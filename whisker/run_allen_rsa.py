"""Run Allen Brain Observatory RSA for the right-only multimodal model.

The script compares three visual representations using the same natural scenes:

1. The mean RSA across independently initialized, untrained MouseNets (four by
   default, matching the original MouseNet baseline replication count).
2. The trained right-only model with whisker gating disabled. This is a
   visual-only inference ablation, not a separately trained unimodal model.
3. The trained right-only model with the inhibitory gate driven by an all-zero
   right-whisker sequence.

No model training is performed here. The Allen neural pseudo-population and
reliability-threshold procedure intentionally follow the previous RSA script.
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MOUSENET_DIR = SCRIPT_DIR.parent
REPO_ROOT = MOUSENET_DIR.parent

# The project contains imports that use both ``mousenet.*`` and ``whisker.*``.
for import_root in (REPO_ROOT, MOUSENET_DIR):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import numpy as np
import torch
import torch.nn.functional as F
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from mousenet.cmouse.mousenet_complete_pool import MouseNetCompletePool
from mousenet.cmouse.network import load_network_from_pickle
from whisker.multimodal_model import MultimodalMouseModel


DEPTH_GROUPS = {
    "l23": (175, 205),
    "l4": (275, 335, 350),
    "l5": (375, 390, 400),
}
DEPTH_GROUPS["all"] = tuple(
    sorted({depth for depths in DEPTH_GROUPS.values() for depth in depths})
)

MODEL_LAYER_BY_DEPTH_GROUP = {
    "all": "VISp2/3",
    "l23": "VISp2/3",
    "l4": "VISp4",
    "l5": "VISp5",
}

CONDITION_DESCRIPTIONS = {
    "base_mousenet": (
        "Mean RSA across independently initialized, untrained MouseNet visual "
        "networks; whisker gate disabled. The default uses four instances to "
        "match the original MouseNet baseline replication count."
    ),
    "trained_gate_off": (
        "Right-only multimodally trained model evaluated with the whisker "
        "gate disabled (visual-only inference ablation)."
    ),
    "trained_gate_on_zero_whisker": (
        "Right-only multimodally trained model with the gate enabled and "
        "driven by an all-zero (30, 51, 2) right-whisker sequence."
    ),
}


def build_initialized_model(network_path, device, seed):
    """Build one reproducible MouseNet instance from a fresh initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Rebuild the complete model for every seed. In this implementation NumPy
    # initializes the sparse Gaussian masks and PyTorch initializes the weights.
    network = load_network_from_pickle(str(network_path))
    visual_net = MouseNetCompletePool(network)
    model = MultimodalMouseModel(
        visual_net=visual_net,
        learnable_temp=True,
        temp=0.1,
    ).to(device)
    model.eval()
    return model


def zscore_neurons_over_images(values, eps=1e-8):
    """Z-score each neuron over natural-scene images."""
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return (values - mean) / (std + eps)


def compute_neural_responses_for_oeid(
    oeid,
    boc,
    offset=0,
    width=15,
    min_neurons=5,
):
    """Return mean natural-scene responses and split-half reliabilities.

    ``width=15`` integrates approximately 500 ms of event activity at the
    approximately 30 Hz sampling rate used for these recordings.
    """
    data_set = boc.get_ophys_experiment_data(int(oeid))

    # Precomputed Allen L0 event traces: (cells, timepoints).
    events = boc.get_ophys_experiment_events(int(oeid))
    traces = events.astype(np.float32)

    cell_ids = data_set.get_cell_specimen_ids()
    if traces.shape[0] != len(cell_ids):
        raise ValueError(
            f"events rows {traces.shape[0]} != cell ids {len(cell_ids)}"
        )

    stimulus_table = data_set.get_stimulus_table("natural_scenes")
    template = data_set.get_stimulus_template("natural_scenes")

    frames = stimulus_table["frame"].to_numpy().astype(int)
    starts = stimulus_table["start"].to_numpy().astype(int)

    n_stimuli = template.shape[0]
    n_cells = traces.shape[0]
    responses_by_stimulus = [[] for _ in range(n_stimuli)]

    for frame, start in zip(frames, starts):
        if frame < 0 or frame >= n_stimuli:
            continue  # Skip blank presentations.

        window_start = start + offset
        window_end = min(window_start + width, traces.shape[1])
        if window_end <= window_start:
            continue

        response = traces[:, window_start:window_end].sum(axis=1)
        if np.any(np.isnan(response)):
            continue

        responses_by_stimulus[frame].append(response.astype(np.float32))

    half1 = np.zeros((n_stimuli, n_cells), dtype=np.float32)
    half2 = np.zeros((n_stimuli, n_cells), dtype=np.float32)
    mean_response = np.zeros((n_stimuli, n_cells), dtype=np.float32)
    good_stimuli = np.zeros(n_stimuli, dtype=bool)

    for stimulus_index, repetitions in enumerate(responses_by_stimulus):
        if len(repetitions) < 4:
            continue

        repetitions = np.asarray(repetitions, dtype=np.float32)
        half1[stimulus_index] = repetitions[::2].mean(axis=0)
        half2[stimulus_index] = repetitions[1::2].mean(axis=0)
        mean_response[stimulus_index] = repetitions.mean(axis=0)
        good_stimuli[stimulus_index] = True

    half1_good = half1[good_stimuli]
    half2_good = half2[good_stimuli]
    mean_response_good = mean_response[good_stimuli]

    reliabilities = np.full(n_cells, np.nan, dtype=np.float32)
    for cell_index in range(n_cells):
        first = half1_good[:, cell_index]
        second = half2_good[:, cell_index]
        if np.std(first) > 0 and np.std(second) > 0:
            reliabilities[cell_index] = np.corrcoef(first, second)[0, 1]

    valid_cells = np.isfinite(reliabilities)
    print(
        f"oeid={oeid}: {valid_cells.sum()}/{n_cells} neurons have finite "
        "split-half reliability",
        flush=True,
    )

    if valid_cells.sum() < min_neurons:
        raise ValueError(
            f"Too few valid neurons: {valid_cells.sum()} kept, "
            f"minimum required is {min_neurons}"
        )

    response_valid = mean_response_good[:, valid_cells]
    reliability_valid = reliabilities[valid_cells]

    del data_set, traces, stimulus_table, template, events
    gc.collect()

    return response_valid, good_stimuli, reliability_valid


def scene_to_tensor_1x64(image_2d):
    """Convert one Allen natural scene to the model's grayscale input."""
    tensor = torch.from_numpy(image_2d).float()
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0, 1)
    tensor = tensor[None, None, :, :]
    tensor = F.interpolate(
        tensor,
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )
    return tensor[0]


@torch.no_grad()
def compute_model_representation(
    model,
    template,
    device,
    layer_name,
    gate_mode,
    whisker_frames=51,
):
    """Return flattened right-image features and their correlation RDM."""
    images_right = torch.stack(
        [scene_to_tensor_1x64(template[index]) for index in range(template.shape[0])]
    ).to(device)
    images_right = model._prep_visual_input(images_right)

    previous_gate = model._gate_enabled
    previous_whisker_embedding = model._current_z

    try:
        if gate_mode == "off":
            model._gate_enabled = False
            model._current_z = None
        elif gate_mode == "zero_whisker":
            zero_whisker = torch.zeros(
                images_right.shape[0],
                30,
                whisker_frames,
                2,
                device=images_right.device,
                dtype=images_right.dtype,
            )
            model._gate_enabled = True
            model._current_z = model.whisker_encoder(zero_whisker)
        else:
            raise ValueError(f"Unknown gate mode: {gate_mode}")

        feature_map = model.visual_net.get_img_feature(
            images_right,
            [layer_name],
            flatten=False,
        )
        if isinstance(feature_map, dict):
            feature_map = feature_map[layer_name]
    finally:
        model._gate_enabled = previous_gate
        model._current_z = previous_whisker_embedding

    features = feature_map.flatten(start_dim=1).cpu().numpy()
    rdm = squareform(pdist(features, metric="correlation"))
    if not np.all(np.isfinite(rdm)):
        raise ValueError(
            f"Non-finite values found in the {layer_name} {gate_mode} RDM"
        )
    return features, rdm


def extract_state_dict(checkpoint):
    """Accept current, legacy, and bare-state-dict checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a mapping containing a model state dictionary"
        )

    for key in ("model_state_dict", "model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate, key

    if checkpoint and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        return checkpoint, "top_level"

    available = ", ".join(sorted(str(key) for key in checkpoint))
    raise KeyError(
        "Could not find a model state dictionary. Expected one of "
        f"'model_state_dict', 'model', or 'state_dict'. Available keys: {available}"
    )


def load_trained_checkpoint(model, checkpoint_path):
    """Load the right-only checkpoint and return provenance metadata."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, state_dict_field = extract_state_dict(checkpoint)

    checkpoint_laterality = checkpoint.get("laterality")
    if checkpoint_laterality not in (None, "right"):
        raise ValueError(
            f"Checkpoint laterality is {checkpoint_laterality!r}, expected 'right'"
        )
    checkpoint_whiskers = checkpoint.get("whisker_count")
    if checkpoint_whiskers not in (None, 30):
        raise ValueError(
            f"Checkpoint contains {checkpoint_whiskers} whiskers, expected 30"
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as original_error:
        if state_dict and all(key.startswith("module.") for key in state_dict):
            stripped = {
                key[len("module."):]: value
                for key, value in state_dict.items()
            }
            model.load_state_dict(stripped, strict=True)
        else:
            raise original_error

    metadata_keys = (
        "format_version",
        "laterality",
        "whisker_count",
        "source_frame_count",
        "input_features",
        "temperature",
        "training_trial_count",
    )
    checkpoint_metadata = {
        key: checkpoint[key]
        for key in metadata_keys
        if key in checkpoint and not torch.is_tensor(checkpoint[key])
    }

    del state_dict, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "path": str(checkpoint_path),
        "state_dict_field": state_dict_field,
        "metadata": checkpoint_metadata,
    }


def get_experiment_metadata(boc):
    """Index cached Allen experiment metadata by OEID."""
    metadata_path = boc.get_cache_path(None, boc.EXPERIMENTS_KEY)
    if metadata_path is not None and Path(metadata_path).is_file():
        with Path(metadata_path).open("r", encoding="utf-8") as stream:
            records = json.load(stream)
    else:
        # Fall back to the SDK only when the manifest's metadata cache is absent.
        records = boc.get_ophys_experiments(simple=False)
    indexed = {}
    for record in records:
        oeid = record.get("id", record.get("ophys_experiment_id"))
        if oeid is not None:
            indexed[int(oeid)] = record
    return indexed


def summarize_experiment_metadata(record):
    """Select depth and mouse-line fields needed for thesis reporting."""
    targeted_structure = record.get("targeted_structure")
    if isinstance(targeted_structure, dict):
        targeted_structure = targeted_structure.get("acronym")
    targeted_structure = targeted_structure or record.get("targeted_structure_acronym")

    specimen = record.get("specimen") or {}
    donor = specimen.get("donor") or {}
    transgenic_lines = donor.get("transgenic_lines") or []
    driver_lines = [
        line.get("name")
        for line in transgenic_lines
        if (
            line.get("transgenic_line_type_code") == "D"
            or line.get("transgenic_line_type_name") == "driver"
        )
        and line.get("name")
    ]
    if record.get("cre_line") and record["cre_line"] not in driver_lines:
        driver_lines.append(record["cre_line"])

    return {
        "imaging_depth_um": int(record["imaging_depth"]),
        "targeted_structure": targeted_structure,
        "specimen_id": record.get("specimen_id"),
        "specimen_name": specimen.get("name") or record.get("specimen_name"),
        "donor_name": record.get("donor_name"),
        "cre_line": record.get("cre_line"),
        "reporter_line": record.get("reporter_line"),
        "full_genotype": donor.get("full_genotype") or record.get("full_genotype"),
        "driver_lines": driver_lines,
    }


def load_requested_oeids(args):
    """Load one OEID or a one-dimensional NumPy OEID list."""
    if args.oeid is not None:
        requested = [int(args.oeid)]
        source = f"single:{args.oeid}"
    else:
        oeid_path = Path(args.oeid_list).expanduser().resolve()
        values = np.load(oeid_path, allow_pickle=False)
        if values.ndim != 1:
            raise ValueError(
                f"OEID list must be one-dimensional, received shape {values.shape}"
            )
        requested = [int(value) for value in values.tolist()]
        source = str(oeid_path)

    # Preserve order while protecting against accidental duplicate experiments.
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise ValueError("The requested OEID list is empty")
    return requested, source


def select_depth_group(oeids, metadata_by_oeid, depth_group):
    """Select only modeled VISp depths and report exclusions."""
    allowed_depths = set(DEPTH_GROUPS[depth_group])
    selected = []
    excluded = []
    missing_metadata = []

    for oeid in oeids:
        record = metadata_by_oeid.get(int(oeid))
        if record is None or record.get("imaging_depth") is None:
            missing_metadata.append(int(oeid))
            continue

        depth = int(record["imaging_depth"])
        if depth in allowed_depths:
            selected.append(int(oeid))
        else:
            excluded.append({"oeid": int(oeid), "imaging_depth_um": depth})

    if missing_metadata:
        raise ValueError(
            "Missing cached imaging-depth metadata for OEIDs: "
            + ", ".join(str(oeid) for oeid in missing_metadata)
        )
    if not selected:
        raise ValueError(
            f"No experiments matched depth group {depth_group!r} "
            f"with depths {sorted(allowed_depths)}"
        )
    return selected, excluded


def finite_float_or_none(value):
    """Return a JSON-safe finite float."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def build_threshold_results(
    area_responses,
    area_reliabilities,
    model_rdms,
    base_seeds,
):
    """Build the pooled-neuron RSA curve once, after all OEIDs are loaded."""
    base_rdms = np.asarray(model_rdms["base_mousenet"])
    if base_rdms.ndim != 3 or base_rdms.shape[0] != len(base_seeds):
        raise ValueError(
            "Expected one base MouseNet RDM per seed; received "
            f"RDM shape {base_rdms.shape} for {len(base_seeds)} seeds"
        )

    base_similarities = 1.0 - base_rdms
    trained_similarities = {
        name: 1.0 - model_rdms[name]
        for name in ("trained_gate_off", "trained_gate_on_zero_whisker")
    }
    thresholds = np.arange(-0.2, 0.95, 0.05)
    threshold_results = []

    for threshold in thresholds:
        selected_responses = []
        for response, reliability in zip(area_responses, area_reliabilities):
            keep = reliability >= threshold
            if keep.sum() == 0:
                continue
            selected_responses.append(
                zscore_neurons_over_images(response[:, keep])
            )

        if not selected_responses:
            continue

        area_matrix = np.concatenate(selected_responses, axis=1)
        if area_matrix.shape[1] < 10:
            continue

        allen_similarity = np.corrcoef(area_matrix)
        upper_triangle = np.triu_indices_from(allen_similarity, k=1)
        trained_rsa = {
            name: spearmanr(
                allen_similarity[upper_triangle],
                model_similarity[upper_triangle],
            ).correlation
            for name, model_similarity in trained_similarities.items()
        }

        # Match the original MouseNet procedure: compute the similarity for each
        # random initialization independently, then average the four scores.
        base_by_seed = np.asarray(
            [
                spearmanr(
                    allen_similarity[upper_triangle],
                    model_similarity[upper_triangle],
                ).correlation
                for model_similarity in base_similarities
            ],
            dtype=float,
        )
        finite_base = base_by_seed[np.isfinite(base_by_seed)]
        base = float(np.mean(finite_base)) if finite_base.size else np.nan
        base_std = (
            float(np.std(finite_base, ddof=1))
            if finite_base.size > 1
            else np.nan
        )
        trained_off = trained_rsa["trained_gate_off"]
        trained_on = trained_rsa["trained_gate_on_zero_whisker"]
        row = {
            "threshold": float(threshold),
            "n_selected_neurons": int(area_matrix.shape[1]),
            "rsa_base_mousenet": finite_float_or_none(base),
            "rsa_base_mousenet_std": finite_float_or_none(base_std),
            "rsa_base_mousenet_by_seed": [
                {
                    "seed": int(seed),
                    "rsa": finite_float_or_none(score),
                }
                for seed, score in zip(base_seeds, base_by_seed)
            ],
            "rsa_trained_gate_off": finite_float_or_none(trained_off),
            "rsa_trained_gate_on_zero_whisker": finite_float_or_none(trained_on),
            "delta_training_gate_off": finite_float_or_none(trained_off - base),
            "delta_training_gate_on": finite_float_or_none(trained_on - base),
            "delta_gate": finite_float_or_none(trained_on - trained_off),
            # Backward-compatible names used by plot_results_allen.ipynb.
            "rsa_off": finite_float_or_none(trained_off),
            "rsa_on": finite_float_or_none(trained_on),
            "delta": finite_float_or_none(trained_on - trained_off),
        }
        threshold_results.append(row)

    return threshold_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Right-only Allen Brain Observatory RSA rerun"
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    oeid_group = parser.add_mutually_exclusive_group(required=True)
    oeid_group.add_argument("--oeid", type=int)
    oeid_group.add_argument("--oeid-list", type=str)

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--width",
        type=int,
        default=15,
        help="Event-response integration window in imaging frames (default: 15).",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--features-out",
        type=str,
        default=None,
        help=(
            "Optional compressed NPZ for model features/RDMs used in response "
            "visualizations. Omit to write JSON only."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(SCRIPT_DIR / "models" / "model_learnable_0.1_right.pt"),
    )
    parser.add_argument(
        "--network-path",
        type=str,
        default=str(SCRIPT_DIR / "network_test_retinotopic.pkl"),
    )
    parser.add_argument(
        "--base-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help=(
            "Independent seeds for the untrained MouseNet baseline. RSA is "
            "computed per seed and then averaged (default: 0 1 2 3)."
        ),
    )
    parser.add_argument(
        "--depth-group",
        choices=("all", "l23", "l4", "l5"),
        default="all",
        help=(
            "Allen imaging-depth group. 'all' includes only modeled L2/3, L4, "
            "and L5 depths and excludes 550 um/L6 experiments."
        ),
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        default=None,
        help=(
            "Optional model-layer override. By default this is derived from "
            "--depth-group."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.width <= 0:
        raise ValueError("--width must be positive")
    if not args.base_seeds:
        raise ValueError("--base-seeds must contain at least one integer seed")
    if len(set(args.base_seeds)) != len(args.base_seeds):
        raise ValueError("--base-seeds must not contain duplicate seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.base_seeds):
        raise ValueError("--base-seeds values must be in the range [0, 2**32)")

    device = torch.device(args.device)
    boc = BrainObservatoryCache(manifest_file=args.manifest)

    requested_oeids, oeid_source = load_requested_oeids(args)
    experiment_metadata = get_experiment_metadata(boc)
    oeids, excluded_by_depth = select_depth_group(
        requested_oeids,
        experiment_metadata,
        args.depth_group,
    )

    layer_name = args.layer_name or MODEL_LAYER_BY_DEPTH_GROUP[args.depth_group]
    print(
        f"Depth group {args.depth_group}: selected {len(oeids)}/"
        f"{len(requested_oeids)} experiments; model layer {layer_name}",
        flush=True,
    )

    network_path = Path(args.network_path).expanduser().resolve()
    if not network_path.is_file():
        raise FileNotFoundError(f"MouseNet network file not found: {network_path}")

    reference_data_set = boc.get_ophys_experiment_data(int(oeids[0]))
    template = reference_data_set.get_stimulus_template("natural_scenes")
    del reference_data_set
    gc.collect()

    if template.shape[0] != 118:
        raise ValueError(
            f"Expected 118 Allen natural scenes, received {template.shape[0]}"
        )

    base_rdms = []
    base_features_by_seed = [] if args.features_out is not None else None
    for seed in args.base_seeds:
        print(f"Computing untrained MouseNet baseline for seed {seed}", flush=True)
        base_model = build_initialized_model(network_path, device, seed)
        base_features, base_rdm = compute_model_representation(
            base_model,
            template,
            device,
            layer_name=layer_name,
            gate_mode="off",
        )
        base_rdms.append(base_rdm)
        if base_features_by_seed is not None:
            base_features_by_seed.append(base_features)
        del base_features, base_rdm, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_rdms = np.stack(base_rdms, axis=0)

    # Use a fresh container for the trained checkpoint rather than overwriting
    # one of the untrained baseline instances.
    model = build_initialized_model(network_path, device, seed=0)
    checkpoint_provenance = load_trained_checkpoint(model, args.model_path)
    model.eval()

    trained_off_features, trained_off_rdm = compute_model_representation(
        model,
        template,
        device,
        layer_name=layer_name,
        gate_mode="off",
    )
    trained_on_features, trained_on_rdm = compute_model_representation(
        model,
        template,
        device,
        layer_name=layer_name,
        gate_mode="zero_whisker",
        whisker_frames=51,
    )

    model_rdms = {
        "base_mousenet": base_rdms,
        "trained_gate_off": trained_off_rdm,
        "trained_gate_on_zero_whisker": trained_on_rdm,
    }

    area_responses = []
    area_reliabilities = []
    per_experiment = []
    skipped_experiments = []

    for oeid in oeids:
        try:
            response, good_stimuli, reliabilities = (
                compute_neural_responses_for_oeid(
                    oeid,
                    boc,
                    offset=args.offset,
                    width=args.width,
                    min_neurons=5,
                )
            )

            if response.shape[0] != 118 or int(good_stimuli.sum()) != 118:
                raise ValueError(
                    f"Expected 118 usable stimuli, received {response.shape[0]}"
                )

            area_responses.append(response)
            area_reliabilities.append(reliabilities)

            metadata = summarize_experiment_metadata(
                experiment_metadata[int(oeid)]
            )
            per_experiment.append(
                {
                    "oeid": int(oeid),
                    "n_valid_cells": int(response.shape[1]),
                    "mean_cell_reliability": float(np.nanmean(reliabilities)),
                    "median_cell_reliability": float(np.nanmedian(reliabilities)),
                    **metadata,
                }
            )
        except Exception as error:
            message = str(error)
            print(f"Skipping {oeid}: {message}", flush=True)
            skipped_experiments.append(
                {"oeid": int(oeid), "reason": message}
            )

    if not area_responses:
        raise RuntimeError("No Allen experiments produced usable neural responses")

    threshold_results = build_threshold_results(
        area_responses,
        area_reliabilities,
        model_rdms,
        args.base_seeds,
    )
    if not threshold_results:
        raise RuntimeError("No reliability threshold produced a valid RSA result")

    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": 3,
        "analysis": "allen_brain_observatory_natural_scenes_rsa",
        "condition_descriptions": CONDITION_DESCRIPTIONS,
        "n_experiments_requested": len(requested_oeids),
        "n_experiments_after_depth_filter": len(oeids),
        "n_experiments_used": len(area_responses),
        "oeid_source": oeid_source,
        "excluded_by_depth": excluded_by_depth,
        "skipped_experiments": skipped_experiments,
        "offset": int(args.offset),
        "width": int(args.width),
        "offset_frames": int(args.offset),
        "response_window_frames": int(args.width),
        "zero_whisker_shape": [30, 51, 2],
        "neural_depth_group": args.depth_group,
        "allowed_imaging_depths_um": list(DEPTH_GROUPS[args.depth_group]),
        "layer_name": layer_name,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "network_path": str(network_path),
        "base_mousenet": {
            "n_initializations": len(args.base_seeds),
            "seeds": [int(seed) for seed in args.base_seeds],
            "aggregation": "RSA computed per initialization, then averaged",
            "dispersion": (
                "Sample standard deviation across initialization-specific "
                "RSA scores"
            ),
            "reference": "Shi et al. (2022), doi:10.1371/journal.pcbi.1010427",
        },
        "checkpoint": checkpoint_provenance,
        "threshold_results": threshold_results,
        "per_experiment": per_experiment,
    }

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

    if args.features_out is not None:
        features_path = Path(args.features_out).expanduser().resolve()
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            features_path,
            base_seeds=np.asarray(args.base_seeds, dtype=np.int64),
            feats_base_mousenet_by_seed=np.stack(base_features_by_seed, axis=0),
            feats_trained_gate_off=trained_off_features,
            feats_trained_gate_on_zero_whisker=trained_on_features,
            model_rdm_base_mousenet_by_seed=base_rdms,
            model_rdm_base_mousenet_mean=np.mean(base_rdms, axis=0),
            model_rdm_trained_gate_off=trained_off_rdm,
            model_rdm_trained_gate_on_zero_whisker=trained_on_rdm,
            # Backward-compatible keys used by plot_results_allen.ipynb.
            feats_off=trained_off_features,
            feats_on=trained_on_features,
            model_rdm_off=trained_off_rdm,
            model_rdm_on=trained_on_rdm,
        )

    print(f"Wrote RSA results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
