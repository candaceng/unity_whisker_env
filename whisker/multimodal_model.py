from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cnn.architecture_whisker import WhiskerArchitecture
from .cnn.whisker_encoder import WhiskerEncoder

class PreprocessedTrialDataset(Dataset):
    def __init__(self, pt_folder):
        self.pt_folder = Path(pt_folder)
        manifest_path = self.pt_folder / "dataset_format.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Right-only dataset manifest not found: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("laterality") != "right":
            raise ValueError(f"{manifest_path} is not a right-only dataset")
        if self.manifest.get("whiskerShape") != [30, 51, 2]:
            raise ValueError(
                f"{manifest_path} has unexpected whiskerShape "
                f"{self.manifest.get('whiskerShape')!r}"
            )

        def trial_id(path):
            return int(path.stem.rsplit("_", 1)[1])

        self.pt_files = sorted(
            self.pt_folder.glob("trial_*.pt"),
            key=trial_id,
        )
        expected_count = self.manifest.get("trialCount")
        if expected_count is not None and len(self.pt_files) != expected_count:
            raise ValueError(
                f"{self.pt_folder} contains {len(self.pt_files)} trial files; "
                f"the manifest declares {expected_count}"
            )
        if not self.pt_files:
            raise ValueError(f"No preprocessed trials found in {self.pt_folder}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        try:
            data = torch.load(
                self.pt_files[idx],
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            data = torch.load(self.pt_files[idx], map_location="cpu")

        image_right = data["image_right"]
        whisker = data["whisker_R"].clone()
        if image_right.shape != (1, 64, 64) or image_right.dtype != torch.uint8:
            raise ValueError(
                f"{self.pt_files[idx]} has an invalid image_right tensor"
            )
        if whisker.shape != (30, 51, 2) or whisker.dtype != torch.float32:
            raise ValueError(
                f"{self.pt_files[idx]} has an invalid whisker_R tensor"
            )
        if not torch.isfinite(whisker).all():
            raise ValueError(
                f"{self.pt_files[idx]} contains non-finite whisker values"
            )

        image_right = image_right.to(torch.float32).div_(255.0)

        # The new sweep is exactly -10 to 40 degrees, so 15 +/- 25 maps
        # theta to [-1, 1]. The s contact coordinate remains unmodified.
        whisker[..., 1].sub_(15.0).div_(25.0)

        return image_right, whisker


class InhibitoryFiLM(nn.Module):
    """
    Channelwise FiLM gate with guaranteed negative scale (inhibition).
    y = (1 + gamma(z)) ⊙ x + beta(z), with gamma(z) <= 0
    """

    def __init__(self, whisker_dim: int, n_channels: int, max_supp: float = 0.25, use_beta: bool = False):
        super().__init__()
        self.fc_gamma = nn.Linear(whisker_dim, n_channels)
        self.use_beta = use_beta
        if use_beta:
            self.fc_beta = nn.Linear(whisker_dim, n_channels)
            nn.init.zeros_(self.fc_beta.weight); nn.init.zeros_(self.fc_beta.bias)

        # init near identity: zero weights so sigmoid(h) ~ 0.5 -> scale ~ 1 - 0.5*max_supp
        nn.init.zeros_(self.fc_gamma.weight); nn.init.zeros_(self.fc_gamma.bias)
        self.max_supp = max_supp  # e.g., 0.3–0.5 => scale in [0.5..1]

    def forward(self, z, fmap):  # z: (B,D), fmap: (B,C,H,W)
        B, C, H, W = fmap.shape
        h = self.fc_gamma(z)                        # (B,C)
        scale = 1 - self.max_supp * torch.sigmoid(h - 2.0)
        y = scale.view(B, C, 1, 1) * fmap

        if self.use_beta:
            beta = self.fc_beta(z).view(B, C, 1, 1)
            y = y + beta

        # logging
        with torch.no_grad():
            self.last_scale_mean = scale.mean().item()
            self.last_scale_min  = scale.min().item()
            self.last_scale_max  = scale.max().item()
            self.last_supp_ratio = (y.abs().mean() / (fmap.abs().mean() + 1e-8)).item()
        return y

class MultimodalMouseModel(nn.Module):
    def __init__(self, visual_net, embed_dim=128, learnable_temp=True, temp=0.1):
        super().__init__()
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(np.log(temp), dtype=torch.float32))
        else:
            self.register_buffer("log_temp", torch.tensor(np.log(temp), dtype=torch.float32))

        self.visual_net = visual_net
        self.whisker_encoder = WhiskerEncoder(
            num_whiskers=30,
            in_dim=2,
            arch=WhiskerArchitecture(),
            output_dim=128,
        )
        self.retinotopic = visual_net.network.retinotopic
        self._visual_in_ch = 2 if self.retinotopic else 1

        dev   = next(self.visual_net.parameters()).device
        dtype = next(self.visual_net.parameters()).dtype
        dummy = torch.zeros(1, self._visual_in_ch, 64, 64, device=dev, dtype=dtype) # used to probe VISrl shape
        visrl_map = self.visual_net.get_img_feature(dummy, ['VISrl5'], flatten=False)

        visual_feat = visrl_map.view(1, -1)
        self.visual_fc = nn.Linear(visual_feat.shape[1], embed_dim)

        visp_map = self.visual_net.get_img_feature(dummy, ['VISp2/3'], flatten=False)
        print(f'visp map shape: {visp_map.shape}')
        C_visp = visp_map.shape[1]
        self.visp_gate = InhibitoryFiLM(whisker_dim=128, n_channels=C_visp)

        # helper: find the module whose name contains 'visp' and '5'
        def _find_module_by_substrings(root, includes):
            cands = [(n, m) for n, m in root.named_modules()
                    if all(s in n.lower() for s in includes)]
            if not cands:
                # fallback: try just 'visp'
                cands = [(n, m) for n, m in root.named_modules() if 'visp' in n.lower()]
            name, module = cands[0]
            return name, module

        self._visp_name, self._visp_module = _find_module_by_substrings(self.visual_net, includes=('visp4visp2/3',))
        print("Hooking:", self._visp_name)

        self._gate_enabled = True
        self._current_z = None

        def _visp_hook(_mod, _inp, out):
            # out: (B, C_visp, H, W)
            if (not self._gate_enabled) or (self._current_z is None):
                return out

            pre = out.detach().abs().mean().item()          # mean magnitude before gating
            gated = self.visp_gate(self._current_z, out)    # apply FiLM (inhibition)
            post = gated.detach().abs().mean().item()       # after gating

            return gated

        # register once
        self._visp_hook_handle = self._visp_module.register_forward_hook(lambda m, i, o: _visp_hook(m, i, o))

    def get_debug_metrics(self):
        m = {
            "visp_scale_mean": getattr(self.visp_gate, "last_scale_mean", None),
            "visp_scale_min":  getattr(self.visp_gate, "last_scale_min",  None),
            "visp_scale_max":  getattr(self.visp_gate, "last_scale_max",  None),
            "visp_supp_ratio": getattr(self.visp_gate, "last_supp_ratio", None),
        }
        return m

    def _to_on_off(self, x):  # x: (B, 1 or 3, H, W)
        # If RGB, convert to gray first
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        assert x.shape[1] == 1, "ON/OFF expects 1-ch input (grayscale)"
        mu  = x.mean(dim=(2,3), keepdim=True)  # global mean
        on  = torch.relu(x - mu)
        off = torch.relu(mu - x)
        return torch.cat([on, off], dim=1)     # (B, 2, H, W)

    def forward(self, image_right, whisker):
        image_right = self._prep_visual_input(image_right)
        if image_right.shape[0] != whisker.shape[0]:
            raise ValueError(
                "image_right and whisker must have the same batch size"
            )

        # Encode tactile input first so the VISp hook can read its latent.
        w_embed = self.whisker_encoder(whisker)
        self._current_z = w_embed
        try:
            v_map = self.visual_net.get_img_feature(
                image_right,
                ["VISrl5"],
                flatten=False,
            )
        finally:
            self._current_z = None

        v_feat = v_map.reshape(image_right.size(0), -1)
        v_embed = self.visual_fc(v_feat)
        return v_embed, w_embed

    # -------- Inference helpers --------
    def _prep_visual_input(self, image_right):
        """Match the channel convention expected by the visual network."""
        if image_right.ndim != 4:
            raise ValueError(
                "image_right must have shape (batch, channels, height, width)"
            )
        if self._visual_in_ch == 2 and image_right.shape[1] == 1:
            image_right = self._to_on_off(image_right)
        elif self._visual_in_ch == 1 and image_right.shape[1] == 2:
            image_right = image_right.mean(dim=1, keepdim=True)
        if image_right.shape[1] != self._visual_in_ch:
            raise ValueError(
                f"Visual network expects {self._visual_in_ch} channels, "
                f"received {image_right.shape[1]}"
            )
        return image_right

    @torch.no_grad()
    def encode_image(self, image_right):
        """
        Return a right-image-only embedding without whisker gating.
        """
        image_right = self._prep_visual_input(image_right)
        previous_gate = self._gate_enabled
        self._gate_enabled = False
        self._current_z = None
        try:
            feature_map = self.visual_net.get_img_feature(
                image_right,
                ["VISrl5"],
                flatten=False,
            )
        finally:
            self._gate_enabled = previous_gate
            self._current_z = None

        visual_features = feature_map.reshape(image_right.size(0), -1)
        return self.visual_fc(visual_features)

    @torch.no_grad()
    def encode_visrl5_pooled(self, image_right):
        image_right = self._prep_visual_input(image_right)
        previous_gate = self._gate_enabled
        self._gate_enabled = False
        self._current_z = None
        try:
            feature_map = self.visual_net.get_img_feature(
                image_right,
                ["VISrl5"],
                flatten=False,
            )
        finally:
            self._gate_enabled = previous_gate
            self._current_z = None
        return feature_map.mean(dim=(2, 3))

    @torch.no_grad()
    def encode_whisker(self, whisker, area="final"):
        """Encode a right-only tensor with shape (batch, 30, 51, 2).

        ``area="final"`` preserves the trained 128-dimensional multimodal
        embedding.  A named anatomical area such as ``"SSp-bfd2/3"``
        returns that encoder activation without introducing a new projection.
        """
        if area == "final":
            return self.whisker_encoder(whisker)
        return self.whisker_encoder.encode(whisker, area=area)

    @property
    def temperature(self):
        return torch.clamp(self.log_temp.exp(), min=0.05, max=0.3)

def clip_loss(vision_embed, whisker_embed, temperature):
    vision_embed = F.normalize(vision_embed, dim=1)
    whisker_embed = F.normalize(whisker_embed, dim=1)
    logits = torch.matmul(vision_embed, whisker_embed.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
