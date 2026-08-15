import sys, os
repo_root = os.path.abspath(os.path.join(".."))

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("PYTHONPATH set to:", repo_root)

sys.path.append('../')
sys.path.append('../cmouse/')
sys.path.append('../mouse_cnn/')
from anatomy import *
from architecture import *

from config import *
import torch.nn as nn
import torch.optim as optim
import torch
from network import load_network_from_pickle
from whisker.multimodal_model import *

import json
import zipfile
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from PIL import Image
import numpy as np


from mcmodels.core import VoxelModelCache
from scipy.spatial import ConvexHull

from mousenet.mouse_cnn.voxel import Target


RIGHT_ONLY_DATA_DIR = Path(
    r"C:\Users\User\Documents\unity_whisker_env\Data\RecordedTrials"
)
RIGHT_ONLY_OUTPUT_DIR = Path(__file__).resolve().parent / "PreprocessedTrialsRight"
RIGHT_WHISKER_COUNT = 30
SOURCE_FRAME_COUNT = 51
TACTILE_COLUMNS = ("s", "theta_deg")
IMAGE_SIZE = (64, 64)


def _numeric_suffix(path):
    """Sort batch_2 before batch_10 and trial_2 before trial_10."""
    try:
        return int(Path(path).stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        raise ValueError(f"Expected a numeric suffix in {path}")


def _load_source_manifest(zip_ref, zip_path):
    try:
        with zip_ref.open("dataset_format.json") as stream:
            manifest = json.load(TextIOWrapper(stream, encoding="utf-8"))
    except KeyError as exc:
        raise ValueError(f"{zip_path.name} has no dataset_format.json") from exc

    expected = {
        "formatVersion": 2,
        "laterality": "right",
        "whiskerCount": RIGHT_WHISKER_COUNT,
        "sourceFrameCount": SOURCE_FRAME_COUNT,
        "tactileFile": "whiskers.csv",
        "tactileOrdering": "frame-major, then whiskerNames order",
        "tactileColumns": list(TACTILE_COLUMNS),
        "imageFile": "frame_right_0000.png",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{zip_path.name}: manifest {key!r} is {manifest.get(key)!r}; "
                f"expected {value!r}"
            )

    whisker_names = manifest.get("whiskerNames")
    if not isinstance(whisker_names, list):
        raise ValueError(f"{zip_path.name}: manifest has no whiskerNames list")
    if len(whisker_names) != RIGHT_WHISKER_COUNT:
        raise ValueError(
            f"{zip_path.name}: expected {RIGHT_WHISKER_COUNT} whisker names, "
            f"found {len(whisker_names)}"
        )
    if len(set(whisker_names)) != len(whisker_names):
        raise ValueError(f"{zip_path.name}: duplicate whisker names in manifest")
    if any(not name.startswith("R") for name in whisker_names):
        raise ValueError(f"{zip_path.name}: manifest contains a non-right whisker")

    return manifest


def _trial_entries(zip_ref, manifest, zip_path):
    """Return validated (trial_id, image_entry, tactile_entry) tuples."""
    files_by_trial = {}
    for entry in zip_ref.infolist():
        if entry.is_dir():
            continue
        parts = PurePosixPath(entry.filename).parts
        if len(parts) != 2 or not parts[0].startswith("trial_"):
            continue
        try:
            trial_id = int(parts[0].split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"{zip_path.name}: invalid trial directory {parts[0]!r}"
            ) from exc
        files_by_trial.setdefault(trial_id, set()).add(parts[1])

    expected_files = {manifest["imageFile"], manifest["tactileFile"]}
    trials = []
    for trial_id in sorted(files_by_trial):
        actual_files = files_by_trial[trial_id]
        if actual_files != expected_files:
            raise ValueError(
                f"{zip_path.name}/trial_{trial_id}: expected files "
                f"{sorted(expected_files)}, found {sorted(actual_files)}"
            )
        prefix = f"trial_{trial_id}"
        trials.append(
            (
                trial_id,
                f"{prefix}/{manifest['imageFile']}",
                f"{prefix}/{manifest['tactileFile']}",
            )
        )

    expected_trials = manifest.get("trialsPerBatch")
    if expected_trials is not None and len(trials) != int(expected_trials):
        raise ValueError(
            f"{zip_path.name}: expected {expected_trials} trials, "
            f"found {len(trials)}"
        )
    if not trials:
        raise ValueError(f"{zip_path.name}: archive contains no trials")

    return trials


def _load_right_only_trial(
    zip_ref,
    image_entry,
    tactile_entry,
    image_size=IMAGE_SIZE,
):
    resampling = getattr(Image, "Resampling", Image)
    with zip_ref.open(image_entry) as stream:
        with Image.open(stream) as image:
            image_array = np.asarray(
                image.convert("L").resize(image_size, resampling.BILINEAR),
                dtype=np.uint8,
            ).copy()
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)

    with zip_ref.open(tactile_entry) as stream:
        with TextIOWrapper(stream, encoding="utf-8", newline="") as text_stream:
            tactile = np.loadtxt(text_stream, delimiter=",", dtype=np.float32)

    expected_rows = SOURCE_FRAME_COUNT * RIGHT_WHISKER_COUNT
    expected_shape = (expected_rows, len(TACTILE_COLUMNS))
    if tactile.shape != expected_shape:
        raise ValueError(
            f"{tactile_entry}: expected CSV shape {expected_shape}, "
            f"found {tactile.shape}"
        )
    if not np.isfinite(tactile).all():
        raise ValueError(f"{tactile_entry}: CSV contains NaN or infinity")

    frame_major = tactile.reshape(
        SOURCE_FRAME_COUNT,
        RIGHT_WHISKER_COUNT,
        len(TACTILE_COLUMNS),
    )
    theta_by_frame = frame_major[:, :, 1]
    if not np.allclose(theta_by_frame, theta_by_frame[:, :1], atol=1e-5):
        raise ValueError(
            f"{tactile_entry}: theta values do not follow frame-major ordering"
        )

    # Encoder convention: (whisker, time, feature).
    whisker_array = np.ascontiguousarray(frame_major.transpose(1, 0, 2))
    whisker_tensor = torch.from_numpy(whisker_array)
    return image_tensor, whisker_tensor


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_valid_preprocessed_trial(path, image_size):
    try:
        data = _torch_load_cpu(path)
    except Exception:
        return False
    return (
        isinstance(data, dict)
        and set(data) == {"image_right", "whisker_R"}
        and isinstance(data["image_right"], torch.Tensor)
        and data["image_right"].shape == (1, image_size[1], image_size[0])
        and data["image_right"].dtype == torch.uint8
        and isinstance(data["whisker_R"], torch.Tensor)
        and data["whisker_R"].shape
        == (RIGHT_WHISKER_COUNT, SOURCE_FRAME_COUNT, len(TACTILE_COLUMNS))
        and data["whisker_R"].dtype == torch.float32
        and torch.isfinite(data["whisker_R"]).all().item()
    )


def check_data_all_zero(zipped_dir=RIGHT_ONLY_DATA_DIR):
    """Report right-only trials in which every whisker has zero contact."""
    zipped_dir = Path(zipped_dir)
    zero_contact_trials = []

    for zip_path in sorted(zipped_dir.glob("batch_*.zip"), key=_numeric_suffix):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            manifest = _load_source_manifest(zip_ref, zip_path)
            for trial_id, _, tactile_entry in _trial_entries(
                zip_ref, manifest, zip_path
            ):
                with zip_ref.open(tactile_entry) as stream:
                    with TextIOWrapper(
                        stream, encoding="utf-8", newline=""
                    ) as text_stream:
                        tactile = np.loadtxt(
                            text_stream, delimiter=",", dtype=np.float32
                        )
                if np.count_nonzero(tactile[:, 0]) == 0:
                    zero_contact_trials.append(trial_id)

    print(f"Trials with all-zero whisker contacts: {len(zero_contact_trials)}")
    print(zero_contact_trials[:10])
    return zero_contact_trials


def preprocess_data(
    zipped_dir=RIGHT_ONLY_DATA_DIR,
    output_dir=RIGHT_ONLY_OUTPUT_DIR,
    overwrite=False,
    image_size=IMAGE_SIZE,
    max_batches=None,
):
    """
    Convert right-only Unity archives into resumable per-trial tensors.

    Archives are read in place; nothing is extracted to disk. Images remain
    uint8 in the saved files and are normalized lazily by the Dataset, reducing
    storage by 4x compared with saving float32 images.
    """
    zipped_dir = Path(zipped_dir)
    output_dir = Path(output_dir)
    image_size = tuple(image_size)

    if len(image_size) != 2 or any(int(size) <= 0 for size in image_size):
        raise ValueError(f"image_size must contain two positive values: {image_size}")
    if not zipped_dir.is_dir():
        raise FileNotFoundError(f"Recorded-trial directory not found: {zipped_dir}")

    zip_paths = sorted(zipped_dir.glob("batch_*.zip"), key=_numeric_suffix)
    if max_batches is not None:
        if int(max_batches) <= 0:
            raise ValueError("max_batches must be positive")
        zip_paths = zip_paths[: int(max_batches)]
    if not zip_paths:
        raise FileNotFoundError(f"No batch_*.zip archives found in {zipped_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    seen_trial_ids = set()
    reference_schema = None
    reference_whisker_names = None

    for batch_index, zip_path in enumerate(zip_paths, start=1):
        print(f"Processing {zip_path.name} ({batch_index}/{len(zip_paths)})")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            bad_member = zip_ref.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"{zip_path.name}: CRC check failed for {bad_member}"
                )

            manifest = _load_source_manifest(zip_ref, zip_path)
            schema = {
                key: manifest[key]
                for key in (
                    "formatVersion",
                    "laterality",
                    "whiskerCount",
                    "sourceFrameCount",
                    "tactileFile",
                    "tactileOrdering",
                    "imageFile",
                )
            }
            if reference_schema is None:
                reference_schema = schema
                reference_whisker_names = list(manifest["whiskerNames"])
            elif (
                schema != reference_schema
                or manifest["whiskerNames"] != reference_whisker_names
            ):
                raise ValueError(
                    f"{zip_path.name}: dataset schema differs from earlier batches"
                )

            trials = _trial_entries(zip_ref, manifest, zip_path)
            for trial_id, image_entry, tactile_entry in trials:
                if trial_id in seen_trial_ids:
                    raise ValueError(
                        f"Duplicate trial_{trial_id} across source archives"
                    )
                seen_trial_ids.add(trial_id)

                save_path = output_dir / f"trial_{trial_id}.pt"
                if (
                    save_path.exists()
                    and not overwrite
                    and _is_valid_preprocessed_trial(save_path, image_size)
                ):
                    skipped += 1
                    continue
                if save_path.exists() and not overwrite:
                    raise ValueError(
                        f"{save_path} exists but is not valid right-only data. "
                        "Use a clean output directory or pass overwrite=True."
                    )

                image_tensor, whisker_tensor = _load_right_only_trial(
                    zip_ref,
                    image_entry,
                    tactile_entry,
                    image_size=image_size,
                )
                partial_path = save_path.with_suffix(".pt.partial")
                try:
                    torch.save(
                        {
                            "image_right": image_tensor,
                            "whisker_R": whisker_tensor,
                        },
                        partial_path,
                    )
                    if not _is_valid_preprocessed_trial(partial_path, image_size):
                        raise RuntimeError(
                            f"Validation failed after saving {partial_path}"
                        )
                    os.replace(partial_path, save_path)
                finally:
                    if partial_path.exists():
                        partial_path.unlink()

                created += 1

        print(
            f"Finished {zip_path.name}: {len(trials)} trials "
            f"({created} created, {skipped} resumed)"
        )

    existing_ids = {
        _numeric_suffix(path) for path in output_dir.glob("trial_*.pt")
    }
    unexpected_ids = existing_ids - seen_trial_ids
    if unexpected_ids:
        sample = sorted(unexpected_ids)[:10]
        raise ValueError(
            f"{output_dir} contains trials not present in the selected archives: "
            f"{sample}. Use a clean output directory."
        )

    output_manifest = {
        "formatVersion": 1,
        "sourceFormatVersion": reference_schema["formatVersion"],
        "laterality": "right",
        "trialCount": len(seen_trial_ids),
        "sourceArchives": [path.name for path in zip_paths],
        "imageKey": "image_right",
        "imageShape": [1, image_size[1], image_size[0]],
        "imageDtype": "uint8",
        "imageNormalization": "divide by 255 at load time",
        "whiskerKey": "whisker_R",
        "whiskerShape": [
            RIGHT_WHISKER_COUNT,
            SOURCE_FRAME_COUNT,
            len(TACTILE_COLUMNS),
        ],
        "whiskerDtype": "float32",
        "whiskerNames": reference_whisker_names,
        "tactileColumns": list(TACTILE_COLUMNS),
        "thetaNormalization": {
            "formula": "(theta_deg - midpoint) / half_range",
            "midpoint": 15.0,
            "halfRange": 25.0,
        },
    }
    manifest_path = output_dir / "dataset_format.json"
    partial_manifest = manifest_path.with_suffix(".json.partial")
    try:
        partial_manifest.write_text(
            json.dumps(output_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(partial_manifest, manifest_path)
    finally:
        if partial_manifest.exists():
            partial_manifest.unlink()

    print(
        f"Done: {len(seen_trial_ids)} validated trials in {output_dir} "
        f"({created} created, {skipped} resumed)."
    )
    return {
        "output_dir": output_dir,
        "trial_count": len(seen_trial_ids),
        "created": created,
        "skipped": skipped,
    }

def get_num_neurons(area, layer):
    """
    :param area: visual area name (e.g. 'VISp')
    :param layer: layer name (e.g. '2/3')
    :return: estimate of number of excitatory neurons in given area/layer
    """
    VISl_counts = {
        '2/3': 15727,
        '4': 6549,
        '5': 10848
    }
    # Ero et al. doesn't include these estimates, so we approximate from
    # density of VISl. Specifically we multiply the estimate from VISl by
    # the ratio of surface areas of L2/3. Surface areas estimated from
    # convex hull of flat map of voxels.

    surface_areas_23 = {
        'VISl': 0.9279064282165419,
        'VISrl': 0.698045549856564,
        'VISli': 0.43560916751267514,
        'VISpor': 1.3936554078054724
    }

    ratio = surface_areas_23[area] / surface_areas_23['VISl']
    return VISl_counts[layer] * ratio

def estimate_area_surface_mm2(structure_acronym, manifest_path='connectivity/voxel_model_manifest.json'):
    """
    Estimate the 2D cortical surface area of a brain region using the Allen voxel model flatmap projection.

    Args:
        structure_acronym (str): e.g. 'SSp-bfd' or 'VISp'
        manifest_path (str): Path to the Allen voxel model manifest JSON

    Returns:
        float: Surface area in mm²
    """
    # Load voxel model
    cache = VoxelModelCache(manifest_file=manifest_path)
    source_mask = cache.get_source_mask()
    structure_tree = cache.get_structure_tree()

    # Get structure ID
    acronym_map = structure_tree.get_id_acronym_map()
    if structure_acronym not in acronym_map:
        raise ValueError(f"Unknown structure acronym: {structure_acronym}")
    structure_id = acronym_map[structure_acronym]

    # Get voxel positions for the structure
    structure_indices = []
    source_keys = source_mask.get_key(structure_ids=None)
    mask_indices = np.array(source_mask.mask.nonzero())

    for i in range(len(source_keys)):
        if structure_tree.structure_descends_from(source_keys[i], structure_id):
            structure_indices.append(i)

    positions_3d = mask_indices[:, structure_indices].T  # shape: (N, 3)

    # Fit sphere to get 2D projection
    center = np.mean(positions_3d, axis=0)
    radius = np.linalg.norm(positions_3d - center, axis=1).mean()

    voxel_size_mm = 0.1
    positions_2d = []
    for pos in positions_3d:
        offset = pos - center
        x = voxel_size_mm * radius * np.arctan(offset[2] / np.linalg.norm(offset[:2]))
        y = voxel_size_mm * radius * np.arctan2(-offset[0], -offset[1])
        positions_2d.append((x, y))

    positions_2d = np.array(positions_2d)
    hull = ConvexHull(positions_2d)
    return hull.volume  # returns mm²

def get_d_w_dict():
    """ Utility function to generate d_w dict to avoid long compute time """

    arch = WhiskerArchitecture()
    anet = gen_anatomy(arch)

    projections = []
    for layer in anet.projections:
        projections.append([layer.pre.area, layer.pre.depth, layer.post.area, layer.post.depth])

    d_w_dict = {}

    for source_area, source_layer, target_area, target_layer in projections:
        if source_area == target_area: # from interlaminar hit rate spatial profile
            d_w_dict[f'{source_area}{source_layer} --> {target_area}{target_layer}'] = 1.0
            continue
            # width_micrometers = self.get_hit_rate_width(source_layer, target_layer)

        target_name = f"{target_area}{target_layer}"
        source_name = f"{source_area}{source_layer}"

        # Use Target object to compute d_w
        try:
            print(f"Calling get_kernel_width_pixels for {source_area}{source_layer} → {target_area}{target_layer}")
            target = Target(target_area, target_layer, external_in_degree=1000)
            d_w_mm = target.get_kernel_width_mm(source_name)  # returns mm
            d_w_um = d_w_mm * 1000
            print('finished calc')
        except Exception as e:
            print(f"[WARNING] Could not estimate d_w for {source_name} → {target_name}: {e}")
            d_w_um = 120  # fallback default

        d_w_dict[f'{source_area}{source_layer} --> {target_area}{target_layer}'] = d_w_um

    return d_w_dict
