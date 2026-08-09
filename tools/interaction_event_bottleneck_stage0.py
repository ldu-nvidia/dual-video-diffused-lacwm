#!/usr/bin/env python3
"""Train-only causal interaction-event bottleneck qualification.

Registration is intentionally separate from execution.  It seals a D405-only
fit256/score64 split before selected RGB or measured state is opened.  The run
then masks rendered robot geometry, extracts low-resolution nonrobot change and
transport fields, fits a future PCA32 bottleneck, and applies fixed causal and
reconstruction gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

# Direct ``python /abs/repo/tools/...py`` execution otherwise places only the
# tools directory on sys.path.  Bootstrap the committed repository root before
# importing shared tool modules; no external checkout is added.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.abc_d405_nominal_geometry_probe import (
    CAMERA_TYPE,
    EXPECTED_ABC_COMMIT,
    OFFICIAL_ASSET_RELATIVE,
    OFFICIAL_SCENE_RELATIVE,
    _decode_top_calibration,
    _joint_qpos_addresses,
    build_robot_only_xml,
    set_observed_pose,
)
from tools.corrected_renderer_attribution import moving_geom_ids, raw_mcap_path


SCHEMA_VERSION = 1
MANIFEST_COUNT = 512
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
PADDED_ACTION_DIM = 23
ACTION_DIM = 14
FIT_CLIPS = 256
SCORE_CLIPS = 64
MOTION_STRATA = 4
FIT_PER_STRATUM = 64
SCORE_PER_STRATUM = 16
SELECTION_SALT = "interaction-event-bottleneck-v1"
MODEL_SEED = 1234
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_SAMPLES = 10_000
PRIMARY_FAMILY_SIZE = 6
RECONSTRUCTION_FAMILY_SIZE = 2
PCA_COMPONENTS = 32
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
WORK_HW = (45, 80)
RENDER_HW = (180, 320)
ROBOT_DILATION_PX = 12
PHOTO_THRESHOLD = 8.0 / 255.0
PHOTO_SCALE = 32.0 / 255.0
FLOW_SCALE_PX = 8.0
PROXIMITY_SCALE_PX = 12.0
PROXIMITY_FLOOR = 0.25
PRIMARY_MIN_IMPROVEMENT_PERCENT = 5.0
RECONSTRUCTION_MIN_IMPROVEMENT_PERCENT = 20.0
MIN_FAVORABLE_FRACTION = 0.60
HISTORY_TRANSITIONS = tuple(range(4))
FUTURE_TRANSITIONS = tuple(range(4, 12))
ACTION_WINDOW = (4, 12)
SHIFT_WINDOWS = {"shift_minus1": (3, 11), "shift_plus1": (5, 13)}
ARMS = (
    "history_only",
    "aligned",
    "episode_shuffled",
    "train_mean",
    "raw_zero",
    "shift_minus1",
    "shift_plus1",
)


class BottleneckError(RuntimeError):
    """Raised when a frozen scientific or artifact contract is violated."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, digest: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
    }
    if digest:
        record["sha256"] = sha256_file(resolved)
    return record


def canonical_identity(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["identity_sha256"] = canonical_identity(result)
    return result


def validate_identity(payload: Mapping[str, Any], location: str) -> None:
    expected = payload.get("identity_sha256")
    if not isinstance(expected, str) or canonical_identity(payload) != expected:
        raise BottleneckError(f"identity mismatch: {location}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def require_false_flags(value: Any, location: str = "root") -> int:
    count = 0
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in {"validation_accessed", "protected_test_accessed"}:
                count += 1
                if child is not False:
                    raise BottleneckError(
                        f"{child_location} must explicitly equal false, got {child!r}"
                    )
            count += require_false_flags(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += require_false_flags(child, f"{location}[{index}]")
    return count


def validate_train_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != MANIFEST_COUNT:
        raise BottleneckError(f"expected {MANIFEST_COUNT} train rows, got {len(rows)}")
    episodes: list[str] = []
    clips: list[str] = []
    for index, row in enumerate(rows):
        if row.get("split") != "train":
            raise BottleneckError(f"row {index} is not train")
        if int(row.get("auxiliary_index", -1)) != index:
            raise BottleneckError(f"row {index} auxiliary index differs")
        if int(row.get("sample_size", -1)) != SAMPLE_SIZE:
            raise BottleneckError(f"row {index} sample size differs")
        if int(row.get("chunk_size", -1)) != CHUNK_SIZE:
            raise BottleneckError(f"row {index} chunk size differs")
        start = int(row["start"])
        expected = list(range(start, start + SAMPLE_SIZE * CHUNK_SIZE, CHUNK_SIZE))
        if row.get("frame_indices") != expected:
            raise BottleneckError(f"row {index} frame map differs")
        episodes.append(str(Path(row["episode_dir"]).resolve()))
        clips.append(str(row["clip_id"]))
    if len(set(episodes)) != len(rows) or len(set(clips)) != len(rows):
        raise BottleneckError("manifest rows must be unique episodes and clips")


def load_cache(cache_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Path, Path]:
    metadata_path = cache_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("split") != "train" or int(metadata.get("clip_count", -1)) != MANIFEST_COUNT:
        raise BottleneckError("only immutable train512 cache is accepted")
    rgb_path = cache_dir / str(metadata["rgb_file"])
    action_path = cache_dir / str(metadata["actions_file"])
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
    if tuple(rgb.shape) != (MANIFEST_COUNT, SAMPLE_SIZE, 3, 180, 960) or rgb.dtype != np.float16:
        raise BottleneckError(f"RGB cache geometry differs: {rgb.shape}/{rgb.dtype}")
    if tuple(actions.shape) != (
        MANIFEST_COUNT,
        SAMPLE_SIZE,
        CHUNK_SIZE,
        PADDED_ACTION_DIM,
    ) or actions.dtype != np.float32:
        raise BottleneckError(f"action cache geometry differs: {actions.shape}/{actions.dtype}")
    if float(np.max(np.abs(actions[..., ACTION_DIM:]))) != 0.0:
        raise BottleneckError("padded action coordinates are nonzero")
    return metadata, rgb, actions, rgb_path, action_path


def selection_hash(clip_id: str) -> str:
    return hashlib.sha256(f"{clip_id}|{SELECTION_SALT}".encode()).hexdigest()


def planned_motion(action: np.ndarray) -> float:
    array = np.asarray(action, dtype=np.float64)
    future = array[ACTION_WINDOW[0] : ACTION_WINDOW[1], -1, :12]
    baseline = array[ACTION_WINDOW[0] - 1, -1, :12]
    return float(np.sqrt(np.mean((future - baseline) ** 2)))


def select_fit_score(eligible: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        (dict(item) for item in eligible),
        key=lambda item: (float(item["planned_joint_motion_rms"]), item["clip_id"]),
    )
    strata = np.array_split(np.asarray(ordered, dtype=object), MOTION_STRATA)
    fit: list[dict[str, Any]] = []
    score: list[dict[str, Any]] = []
    for stratum, values in enumerate(strata):
        candidates = [dict(item) for item in values.tolist()]
        if len(candidates) < FIT_PER_STRATUM + SCORE_PER_STRATUM:
            raise BottleneckError(
                f"motion stratum {stratum} has {len(candidates)} eligible clips; need 80"
            )
        ranked = sorted(candidates, key=lambda item: (selection_hash(item["clip_id"]), item["clip_id"]))
        for rank, item in enumerate(ranked[: FIT_PER_STRATUM + SCORE_PER_STRATUM]):
            item["motion_stratum"] = stratum
            item["selection_hash"] = selection_hash(item["clip_id"])
            item["rank_within_stratum"] = rank
            if rank < FIT_PER_STRATUM:
                item["role"] = "predictor_fit"
                fit.append(item)
            else:
                item["role"] = "heldout_score"
                score.append(item)

    for stratum in range(MOTION_STRATA):
        group = sorted(
            [item for item in score if int(item["motion_stratum"]) == stratum],
            key=lambda item: (item["selection_hash"], item["clip_id"]),
        )
        for position, item in enumerate(group):
            donor = group[(position + 1) % len(group)]
            if donor["episode_dir"] == item["episode_dir"]:
                raise BottleneckError("shuffle donor equals native episode")
            item["donor_manifest_index"] = int(donor["manifest_index"])
            item["donor_clip_id"] = donor["clip_id"]
            item["donor_episode_dir"] = donor["episode_dir"]
    fit = sorted(fit, key=lambda item: (int(item["motion_stratum"]), item["selection_hash"]))
    score = sorted(score, key=lambda item: (int(item["motion_stratum"]), item["selection_hash"]))
    if len(fit) != FIT_CLIPS or len(score) != SCORE_CLIPS:
        raise BottleneckError("selected split geometry differs")
    return fit, score


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(args.train_manifest)
    validate_train_manifest(rows)
    metadata, _, actions, rgb_path, action_path = load_cache(args.train_cache)

    eligible: list[dict[str, Any]] = []
    camera_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        episode = Path(row["episode_dir"]).resolve()
        state_path = episode / "states.npz"
        if not state_path.is_file():
            continue
        try:
            mcap_path = raw_mcap_path(row, args.preprocessed_root, args.raw_root)
            from tools.corrected_renderer_attribution import read_mcap_metadata

            episode_metadata = read_mcap_metadata(mcap_path)
        except (FileNotFoundError, ValueError):
            continue
        camera_type = episode_metadata.get("top_camera_type", "missing")
        camera_counts[camera_type] = camera_counts.get(camera_type, 0) + 1
        if camera_type != CAMERA_TYPE:
            continue
        calibration = _decode_top_calibration(mcap_path)
        eligible.append(
            {
                "manifest_index": index,
                "clip_id": str(row["clip_id"]),
                "episode_dir": str(episode),
                "raw_mcap": str(mcap_path.resolve()),
                "top_camera_type": camera_type,
                "camera_width": int(calibration["camera_width"]),
                "camera_height": int(calibration["camera_height"]),
                "fy": float(calibration["K"][4]),
                "planned_joint_motion_rms": planned_motion(actions[index]),
                "validation_accessed": False,
                "protected_test_accessed": False,
            }
        )

    fit, score = select_fit_score(eligible)
    if set(item["episode_dir"] for item in fit) & set(item["episode_dir"] for item in score):
        raise BottleneckError("fit and score episodes overlap")

    source = Path(__file__).resolve()
    protocol = args.protocol.resolve()
    frozen_source = output / "frozen_interaction_event_bottleneck_stage0.py"
    shutil.copy2(source, frozen_source)
    payload = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_bottleneck_stage0_registration",
            "status": "registered_before_selected_rgb_or_state_access",
            "created_at_utc": now(),
            "source": {
                **file_record(source),
                "git_commit": args.expected_commit or git_commit(source.parents[1]),
                "frozen_copy": file_record(frozen_source),
            },
            "protocol": file_record(protocol),
            "inputs": {
                "train_manifest": file_record(args.train_manifest),
                "cache_metadata": file_record(args.train_cache / "metadata.json"),
                "cache_dir": str(args.train_cache.resolve()),
                "rgb": {
                    **file_record(rgb_path, digest=False),
                    "registered_sha256": metadata["rgb_sha256"],
                    "opened_for_selection": False,
                },
                "actions": {
                    **file_record(action_path, digest=False),
                    "registered_sha256": metadata["actions_sha256"],
                    "opened_for_selection": True,
                },
                "preprocessed_root": str(args.preprocessed_root.resolve()),
                "raw_root": str(args.raw_root.resolve()),
                "camera_type_counts": camera_counts,
                "eligible_d405": len(eligible),
                "selected_state_opened": False,
                "selected_rgb_opened": False,
            },
            "split": {
                "source_split": "train",
                "fit_clips": FIT_CLIPS,
                "score_clips": SCORE_CLIPS,
                "motion_strata": MOTION_STRATA,
                "fit_per_stratum": FIT_PER_STRATUM,
                "score_per_stratum": SCORE_PER_STRATUM,
                "selection_salt": SELECTION_SALT,
                "fit_score_episode_disjoint": True,
                "fit": fit,
                "score": score,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "representation": {
                "history_frames": [0, 1, 2, 3, 4],
                "future_target_frames": list(range(5, 13)),
                "history_transitions": [f"{t}->{t + 1}" for t in HISTORY_TRANSITIONS],
                "future_transitions": [f"{t}->{t + 1}" for t in FUTURE_TRANSITIONS],
                "render_hw": list(RENDER_HW),
                "work_hw": list(WORK_HW),
                "channels": [
                    "positive_change",
                    "negative_change",
                    "event_weighted_horizontal_transport",
                    "event_weighted_vertical_transport",
                ],
                "robot_union_dilation_px_at_180x320": ROBOT_DILATION_PX,
                "photo_threshold": PHOTO_THRESHOLD,
                "photo_scale": PHOTO_SCALE,
                "flow_scale_px": FLOW_SCALE_PX,
                "proximity_scale_work_pixels": PROXIMITY_SCALE_PX,
                "proximity_floor": PROXIMITY_FLOOR,
                "history_pca_components": PCA_COMPONENTS,
                "future_bottleneck_components": PCA_COMPONENTS,
                "future_state_or_rgb_is_predictor_input": False,
            },
            "models": {
                "seed": MODEL_SEED,
                "action_window": list(ACTION_WINDOW),
                "action_pca_components": PCA_COMPONENTS,
                "ridge_alpha_grid": list(ALPHAS),
                "alpha_selection": "five-fold fit256-only KFold target-standardized MSE",
                "history_only_equal_width": "history PCA32 concatenated with constant zeros32",
                "history_plus_action": "history PCA32 concatenated with action PCA32",
                "arms": list(ARMS),
            },
            "statistics": {
                "bootstrap_unit": "heldout train episode",
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "primary_family_size": PRIMARY_FAMILY_SIZE,
                "primary_one_sided_confidence": 1.0 - 0.05 / PRIMARY_FAMILY_SIZE,
                "primary_point_min_percent": PRIMARY_MIN_IMPROVEMENT_PERCENT,
                "reconstruction_family_size": RECONSTRUCTION_FAMILY_SIZE,
                "reconstruction_one_sided_confidence": 1.0 - 0.05 / RECONSTRUCTION_FAMILY_SIZE,
                "reconstruction_point_min_percent": RECONSTRUCTION_MIN_IMPROVEMENT_PERCENT,
                "minimum_favorable_fraction": MIN_FAVORABLE_FRACTION,
                "all_conditions_required": True,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    write_json(output / "registration.json", payload)
    return payload


def _rgb_gray(frame_chw: np.ndarray) -> np.ndarray:
    top = np.asarray(frame_chw[:, :, : RENDER_HW[1]], dtype=np.float32)
    top = np.clip((top + 1.0) * 0.5, 0.0, 1.0)
    gray = 0.2989 * top[0] + 0.5870 * top[1] + 0.1140 * top[2]
    return cv2.resize(
        gray,
        (WORK_HW[1], WORK_HW[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def render_robot_masks(
    *,
    renderer: Any,
    model: Any,
    data: Any,
    mujoco: Any,
    option: Any,
    addresses: Mapping[str, int],
    moving_geoms: set[int],
    camera_id: int,
    boundaries: np.ndarray,
    native_height: int,
    fy: float,
) -> tuple[np.ndarray, list[float]]:
    if boundaries.shape != (SAMPLE_SIZE, ACTION_DIM):
        raise BottleneckError(f"boundary geometry differs: {boundaries.shape}")
    model.cam_fovy[camera_id] = math.degrees(
        2.0 * math.atan(float(native_height) / (2.0 * float(fy)))
    )
    renderer.enable_segmentation_rendering()
    masks: list[np.ndarray] = []
    latencies: list[float] = []
    moving = np.asarray(sorted(moving_geoms), dtype=np.int32)
    for pose in boundaries:
        started = time.perf_counter()
        set_observed_pose(
            model,
            data,
            mujoco,
            dict(addresses),
            np.asarray(pose[:12], dtype=np.float64),
            np.asarray(pose[12:], dtype=np.float64),
        )
        renderer.update_scene(data, camera="top", scene_option=option)
        segmentation = renderer.render().copy()
        latencies.append((time.perf_counter() - started) * 1000.0)
        geom_id = np.asarray(segmentation[..., 0], dtype=np.int32)
        object_type = np.asarray(segmentation[..., 1], dtype=np.int32)
        is_geom = object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = is_geom & np.isin(geom_id, moving)
        if int(mask.sum()) < 10:
            raise BottleneckError("rendered articulated robot mask is empty")
        masks.append(mask)
    renderer.disable_segmentation_rendering()
    return np.stack(masks), latencies


def transition_event_field(
    gray_source: np.ndarray,
    gray_target: np.ndarray,
    robot_source: np.ndarray,
    robot_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if gray_source.shape != WORK_HW or gray_target.shape != WORK_HW:
        raise ValueError("grayscale work geometry differs")
    if robot_source.shape != RENDER_HW or robot_target.shape != RENDER_HW:
        raise ValueError("robot render geometry differs")

    union = np.logical_or(robot_source, robot_target).astype(np.uint8)
    kernel = np.ones((2 * ROBOT_DILATION_PX + 1, 2 * ROBOT_DILATION_PX + 1), np.uint8)
    excluded_render = cv2.dilate(union, kernel) > 0
    excluded = cv2.resize(
        excluded_render.astype(np.uint8),
        (WORK_HW[1], WORK_HW[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    robot_core = cv2.resize(
        union,
        (WORK_HW[1], WORK_HW[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    if not np.any(robot_core):
        raise BottleneckError("downsampled robot support is empty")

    # Distance is defined outside robot support.  A quarter-weight far field is
    # retained so the representation cannot silently become a robot-ring-only
    # statistic.
    distance = cv2.distanceTransform((~robot_core).astype(np.uint8), cv2.DIST_L2, 3)
    proximity = PROXIMITY_FLOOR + (1.0 - PROXIMITY_FLOOR) * np.exp(
        -distance / PROXIMITY_SCALE_PX
    )
    weight = np.where(excluded, 0.0, proximity).astype(np.float32)

    delta = np.asarray(gray_target - gray_source, dtype=np.float32)
    positive = np.clip((delta - PHOTO_THRESHOLD) / PHOTO_SCALE, 0.0, 1.0)
    negative = np.clip((-delta - PHOTO_THRESHOLD) / PHOTO_SCALE, 0.0, 1.0)
    event = np.clip((np.abs(delta) - PHOTO_THRESHOLD) / PHOTO_SCALE, 0.0, 1.0)
    flow = cv2.calcOpticalFlowFarneback(
        np.asarray(gray_source * 255.0, dtype=np.float32),
        np.asarray(gray_target * 255.0, dtype=np.float32),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    u = event * np.clip(flow[..., 0] / FLOW_SCALE_PX, -1.0, 1.0)
    v = event * np.clip(flow[..., 1] / FLOW_SCALE_PX, -1.0, 1.0)
    field = np.stack((positive, negative, u, v), axis=0) * weight[None]
    if not np.isfinite(field).all():
        raise BottleneckError("event field contains non-finite values")
    audit = {
        "robot_core_fraction": float(robot_core.mean()),
        "robot_excluded_fraction": float(excluded.mean()),
        "nonrobot_event_mass": float((event * (~excluded)).mean()),
        "weighted_event_mass": float((event * weight).mean()),
        "weighted_transport_rms": float(
            np.sqrt(np.mean((u * weight) ** 2 + (v * weight) ** 2))
        ),
    }
    return field.astype(np.float32), audit


def clip_event_fields(
    frames: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    if frames.shape != (SAMPLE_SIZE, 3, 180, 960):
        raise BottleneckError(f"clip RGB geometry differs: {frames.shape}")
    if masks.shape != (SAMPLE_SIZE, *RENDER_HW):
        raise BottleneckError(f"clip mask geometry differs: {masks.shape}")
    gray = [_rgb_gray(frames[index]) for index in range(SAMPLE_SIZE)]
    fields: list[np.ndarray] = []
    audits: list[dict[str, float]] = []
    for transition in range(SAMPLE_SIZE - 1):
        field, audit = transition_event_field(
            gray[transition], gray[transition + 1], masks[transition], masks[transition + 1]
        )
        fields.append(field)
        audits.append(audit)
    stacked = np.stack(fields)
    return stacked[:4], stacked[4:12], audits


def _standardizer(values: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.where(std > epsilon, std, 1.0).astype(np.float32)


def _fit_pca(values: np.ndarray, *, components: int = PCA_COMPONENTS) -> tuple[Any, np.ndarray]:
    from sklearn.decomposition import PCA

    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=MODEL_SEED,
        iterated_power=5,
    )
    transformed = pca.fit_transform(flat).astype(np.float32)
    return pca, transformed


def _fit_action_pca(values: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    mean, std = _standardizer(flat)
    active = flat.std(axis=0, dtype=np.float64) > 1e-8
    if int(active.sum()) < PCA_COMPONENTS:
        raise BottleneckError(f"only {int(active.sum())} active planned-action dimensions")
    normalized = ((flat - mean) / std)[:, active]
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    transformed = pca.fit_transform(normalized).astype(np.float32)
    return pca, transformed, mean, std, active


def _transform_action_pca(
    pca: Any,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    return pca.transform(((flat - mean) / std)[:, active]).astype(np.float32)


def _choose_alpha(inputs: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    folds = KFold(n_splits=5, shuffle=True, random_state=MODEL_SEED)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        fold_errors: list[float] = []
        for fit_index, dev_index in folds.split(inputs):
            model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            model.fit(inputs[fit_index], targets[fit_index])
            fold_errors.append(
                float(np.mean((model.predict(inputs[dev_index]) - targets[dev_index]) ** 2))
            )
        scores[str(alpha)] = float(np.mean(fold_errors))
    selected = min(ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    return float(selected), scores


def common_bootstrap_indices(
    units: int,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if units < 2 or samples < 100:
        raise ValueError("bootstrap requires at least two units and 100 samples")
    return np.random.default_rng(seed).integers(0, units, size=(samples, units), dtype=np.int64)


def paired_relative_effect(
    reference: np.ndarray,
    candidate: np.ndarray,
    bootstrap: np.ndarray,
    *,
    family_size: int,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 1 or len(ref) != bootstrap.shape[1]:
        raise ValueError("paired effect geometry differs")
    if (
        not np.isfinite(ref).all()
        or not np.isfinite(cand).all()
        or np.any(ref < 0.0)
        or np.any(cand < 0.0)
        or float(ref.mean()) <= 0.0
    ):
        raise ValueError("paired losses must be finite/nonnegative with positive reference mean")
    point = 100.0 * (float(ref.mean()) - float(cand.mean())) / float(ref.mean())
    sampled_ref = ref[bootstrap].mean(axis=1)
    sampled_cand = cand[bootstrap].mean(axis=1)
    if np.any(sampled_ref <= 0.0):
        raise ValueError("a bootstrap replicate has zero reference loss")
    distribution = 100.0 * (sampled_ref - sampled_cand) / sampled_ref
    lower_probability = 0.05 / float(family_size)
    return {
        "reference_mean": float(ref.mean()),
        "candidate_mean": float(cand.mean()),
        "relative_improvement_percent": point,
        "one_sided_bonferroni_confidence": 1.0 - lower_probability,
        "simultaneous_lower_bound_percent": float(
            np.quantile(distribution, lower_probability, method="linear")
        ),
        "nominal_bootstrap_95_ci_percent": [
            float(np.quantile(distribution, 0.025, method="linear")),
            float(np.quantile(distribution, 0.975, method="linear")),
        ],
        "favorable_count": int(np.sum(cand < ref)),
        "favorable_fraction": float(np.mean(cand < ref)),
        "ties": int(np.sum(cand == ref)),
    }


def effect_gate(effect: Mapping[str, Any], *, minimum_percent: float) -> dict[str, bool]:
    checks = {
        "point_at_least_minimum": float(effect["relative_improvement_percent"]) >= minimum_percent,
        "simultaneous_lower_bound_strictly_positive": float(
            effect["simultaneous_lower_bound_percent"]
        )
        > 0.0,
        "favorable_fraction_at_least_60_percent": float(effect["favorable_fraction"])
        >= MIN_FAVORABLE_FRACTION,
    }
    return {**checks, "passed": all(checks.values())}


def analyze_error_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = common_bootstrap_indices(SCORE_CLIPS)
    effects: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for metric in ("coefficient", "field"):
        candidate = np.asarray(arrays[f"{metric}_error_aligned"])
        for reference in ("history_only", "episode_shuffled", "train_mean"):
            label = f"{metric}:aligned_vs_{reference}"
            effect = paired_relative_effect(
                np.asarray(arrays[f"{metric}_error_{reference}"]),
                candidate,
                bootstrap,
                family_size=PRIMARY_FAMILY_SIZE,
            )
            effects[label] = effect
            gates[label] = effect_gate(effect, minimum_percent=PRIMARY_MIN_IMPROVEMENT_PERCENT)

    for reference in ("fit_mean", "raw_zero"):
        label = f"reconstruction:oracle_vs_{reference}"
        effect = paired_relative_effect(
            np.asarray(arrays[f"reconstruction_error_{reference}"]),
            np.asarray(arrays["reconstruction_error_oracle"]),
            bootstrap,
            family_size=RECONSTRUCTION_FAMILY_SIZE,
        )
        effects[label] = effect
        gates[label] = effect_gate(
            effect, minimum_percent=RECONSTRUCTION_MIN_IMPROVEMENT_PERCENT
        )

    diagnostics: dict[str, Any] = {}
    for metric in ("coefficient", "field"):
        candidate = np.asarray(arrays[f"{metric}_error_aligned"])
        for reference in ("raw_zero", "shift_minus1", "shift_plus1"):
            label = f"{metric}:aligned_vs_{reference}"
            diagnostics[label] = paired_relative_effect(
                np.asarray(arrays[f"{metric}_error_{reference}"]),
                candidate,
                bootstrap,
                family_size=1,
            )
    gates["all_passed"] = all(item["passed"] for item in gates.values())
    return {"mandatory": effects, "diagnostic": diagnostics}, gates


def _selected_clip_arrays(
    *,
    item: Mapping[str, Any],
    row: Mapping[str, Any],
    rgb_cache: np.ndarray,
    action_cache: np.ndarray,
    renderer_context: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[float]]:
    index = int(item["manifest_index"])
    state_path = Path(row["episode_dir"]) / "states.npz"
    with np.load(state_path, allow_pickle=False) as state:
        required = {
            "joint_states",
            "gripper_states",
            "joint_actions",
            "gripper_actions",
            "frame_ts",
        }
        if not required.issubset(state.files):
            raise BottleneckError(f"state file lacks {sorted(required - set(state.files))}: {state_path}")
        frame_indices = np.asarray(row["frame_indices"], dtype=np.int64)
        boundaries = np.concatenate(
            (
                np.asarray(state["joint_states"][frame_indices], dtype=np.float32),
                np.asarray(state["gripper_states"][frame_indices], dtype=np.float32),
            ),
            axis=1,
        )
        start = int(row["start"])
        stop = start + SAMPLE_SIZE * CHUNK_SIZE
        native_actions = np.concatenate(
            (
                np.asarray(state["joint_actions"][start:stop], dtype=np.float32),
                np.asarray(state["gripper_actions"][start:stop], dtype=np.float32),
            ),
            axis=1,
        ).reshape(SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM)
        timestamps = np.asarray(state["frame_ts"][frame_indices], dtype=np.int64)
    cached_actions = np.asarray(action_cache[index, ..., :ACTION_DIM], dtype=np.float32)
    if not np.array_equal(native_actions, cached_actions):
        raise BottleneckError(f"raw/cache action mismatch at manifest row {index}")
    frames = np.asarray(rgb_cache[index], dtype=np.float32)
    masks, latencies = render_robot_masks(
        renderer=renderer_context["renderer"],
        model=renderer_context["model"],
        data=renderer_context["data"],
        mujoco=renderer_context["mujoco"],
        option=renderer_context["option"],
        addresses=renderer_context["addresses"],
        moving_geoms=renderer_context["moving_geoms"],
        camera_id=renderer_context["camera_id"],
        boundaries=boundaries,
        native_height=int(item["camera_height"]),
        fy=float(item["fy"]),
    )
    history, future, transition_audits = clip_event_fields(frames, masks)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "role": item["role"],
        "manifest_index": index,
        "clip_id": item["clip_id"],
        "episode_dir": item["episode_dir"],
        "states_npz": file_record(state_path),
        "frame_indices": frame_indices.tolist(),
        "frame_timestamps_ns": timestamps.tolist(),
        "camera": {
            "type": item["top_camera_type"],
            "native_width": int(item["camera_width"]),
            "native_height": int(item["camera_height"]),
            "fy": float(item["fy"]),
            "render_fovy_degrees": math.degrees(
                2.0
                * math.atan(float(item["camera_height"]) / (2.0 * float(item["fy"])))
            ),
            "nominal_extrinsic": True,
            "principal_point_centered": True,
            "distortion_applied": False,
        },
        "transition_audits": transition_audits,
        "source_split": "train",
        "future_rgb_used_only_for_target": True,
        "future_state_used_only_for_target_mask": True,
        "validation_accessed": False,
        "protected_test_accessed": False,
    }
    return history, future, cached_actions, provenance, latencies


def _model_prediction(
    *,
    model: Any,
    history_pc: np.ndarray,
    action_pc: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    design = np.concatenate((history_pc, action_pc), axis=1)
    standardized = (design - input_mean) / input_std
    prediction = model.predict(standardized).astype(np.float32)
    return prediction * target_std + target_mean


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import mujoco
        from sklearn.linear_model import Ridge
    except ImportError as exc:  # pragma: no cover - cluster dependency guidance
        raise RuntimeError("run requires mujoco and scikit-learn") from exc

    output = args.output.resolve()
    if (output / "analysis.json").exists() or (output / "run_complete.json").exists():
        raise FileExistsError("completed output is immutable")
    registration = read_json(output / "registration.json")
    validate_identity(registration, "registration")
    source = Path(__file__).resolve()
    if sha256_file(source) != registration["source"]["sha256"]:
        raise BottleneckError("execution source bytes differ from registration")
    frozen_source = output / "frozen_interaction_event_bottleneck_stage0.py"
    if sha256_file(frozen_source) != registration["source"]["frozen_copy"]["sha256"]:
        raise BottleneckError("frozen source differs from registration")

    manifest_path = Path(registration["inputs"]["train_manifest"]["path"])
    cache_dir = Path(registration["inputs"]["cache_dir"])
    rows = read_jsonl(manifest_path)
    validate_train_manifest(rows)
    _, rgb_cache, action_cache, _, _ = load_cache(cache_dir)
    selected = [*registration["split"]["fit"], *registration["split"]["score"]]
    fit_items = registration["split"]["fit"]
    score_items = registration["split"]["score"]

    official_root = args.official_abc_root.resolve()
    official_commit = git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise BottleneckError(
            f"official ABC must be {EXPECTED_ABC_COMMIT}, got {official_commit}"
        )
    scene_path = official_root / OFFICIAL_SCENE_RELATIVE
    asset_root = official_root / OFFICIAL_ASSET_RELATIVE
    robot_xml = build_robot_only_xml(scene_path, asset_root)
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    if camera_id < 0:
        raise BottleneckError("official model lacks top camera")
    moving_geoms = moving_geom_ids(model, mujoco)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    renderer = mujoco.Renderer(model, height=RENDER_HW[0], width=RENDER_HW[1])
    renderer_context = {
        "renderer": renderer,
        "model": model,
        "data": data,
        "mujoco": mujoco,
        "option": option,
        "addresses": addresses,
        "moving_geoms": moving_geoms,
        "camera_id": camera_id,
    }

    fit_history: list[np.ndarray] = []
    fit_future: list[np.ndarray] = []
    fit_actions: list[np.ndarray] = []
    score_history: list[np.ndarray] = []
    score_future: list[np.ndarray] = []
    score_actions: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    started = time.time()
    try:
        for position, item in enumerate(selected):
            row = rows[int(item["manifest_index"])]
            history, future, actions, record, latency = _selected_clip_arrays(
                item=item,
                row=row,
                rgb_cache=rgb_cache,
                action_cache=action_cache,
                renderer_context=renderer_context,
            )
            provenance.append(record)
            render_latencies.extend(latency[2:] if len(latency) > 2 else latency)
            if item["role"] == "predictor_fit":
                fit_history.append(history)
                fit_future.append(future)
                fit_actions.append(actions)
            else:
                score_history.append(history)
                score_future.append(future)
                score_actions.append(actions)
            if position == 0 or (position + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "event": "event_field_extraction_progress",
                            "clips": position + 1,
                            "total": len(selected),
                            "elapsed_sec": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        renderer.close()

    fit_h = np.stack(fit_history)
    fit_y_map = np.stack(fit_future)
    fit_a_full = np.stack(fit_actions)
    score_h = np.stack(score_history)
    score_y_map = np.stack(score_future)
    score_a_full = np.stack(score_actions)
    if fit_h.shape != (FIT_CLIPS, 4, 4, *WORK_HW):
        raise BottleneckError(f"fit history geometry differs: {fit_h.shape}")
    if fit_y_map.shape != (FIT_CLIPS, 8, 4, *WORK_HW):
        raise BottleneckError(f"fit target geometry differs: {fit_y_map.shape}")
    if score_y_map.shape != (SCORE_CLIPS, 8, 4, *WORK_HW):
        raise BottleneckError(f"score target geometry differs: {score_y_map.shape}")

    history_pca, fit_history_pc = _fit_pca(fit_h)
    target_pca, fit_target_pc = _fit_pca(fit_y_map)
    score_history_pc = history_pca.transform(score_h.reshape(SCORE_CLIPS, -1)).astype(np.float32)
    score_target_pc = target_pca.transform(score_y_map.reshape(SCORE_CLIPS, -1)).astype(np.float32)

    fit_action_window = fit_a_full[:, ACTION_WINDOW[0] : ACTION_WINDOW[1], :, :ACTION_DIM]
    score_action_window = score_a_full[:, ACTION_WINDOW[0] : ACTION_WINDOW[1], :, :ACTION_DIM]
    action_pca, fit_action_pc, action_mean, action_std, action_active = _fit_action_pca(
        fit_action_window
    )
    score_action_pc = _transform_action_pca(
        action_pca, score_action_window, action_mean, action_std, action_active
    )

    target_mean, target_std = _standardizer(fit_target_pc)
    fit_target_z = (fit_target_pc - target_mean) / target_std
    zeros_fit = np.zeros_like(fit_action_pc)
    history_design = np.concatenate((fit_history_pc, zeros_fit), axis=1)
    action_design = np.concatenate((fit_history_pc, fit_action_pc), axis=1)
    history_input_mean, history_input_std = _standardizer(history_design)
    action_input_mean, action_input_std = _standardizer(action_design)
    history_x = (history_design - history_input_mean) / history_input_std
    action_x = (action_design - action_input_mean) / action_input_std
    history_alpha, history_cv = _choose_alpha(history_x, fit_target_z)
    action_alpha, action_cv = _choose_alpha(action_x, fit_target_z)
    history_model = Ridge(
        alpha=history_alpha, fit_intercept=True, solver="cholesky"
    ).fit(history_x, fit_target_z)
    action_model = Ridge(
        alpha=action_alpha, fit_intercept=True, solver="cholesky"
    ).fit(action_x, fit_target_z)

    zeros_score = np.zeros_like(score_action_pc)
    history_prediction_pc = _model_prediction(
        model=history_model,
        history_pc=score_history_pc,
        action_pc=zeros_score,
        input_mean=history_input_mean,
        input_std=history_input_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    score_index_position = {
        int(item["manifest_index"]): position for position, item in enumerate(score_items)
    }
    donor_positions = np.asarray(
        [score_index_position[int(item["donor_manifest_index"])] for item in score_items],
        dtype=np.int64,
    )
    shifted_pc: dict[str, np.ndarray] = {}
    for label, (start_index, stop_index) in SHIFT_WINDOWS.items():
        shifted = score_a_full[:, start_index:stop_index, :, :ACTION_DIM]
        shifted_pc[label] = _transform_action_pca(
            action_pca, shifted, action_mean, action_std, action_active
        )
    fit_action_mean_raw = fit_action_window.reshape(FIT_CLIPS, -1).mean(axis=0).reshape(
        1, 8, CHUNK_SIZE, ACTION_DIM
    )
    mean_score_raw = np.repeat(fit_action_mean_raw, SCORE_CLIPS, axis=0)
    zero_score_raw = np.zeros_like(score_action_window)
    control_action_pc = {
        "aligned": score_action_pc,
        "episode_shuffled": score_action_pc[donor_positions],
        "train_mean": _transform_action_pca(
            action_pca, mean_score_raw, action_mean, action_std, action_active
        ),
        "raw_zero": _transform_action_pca(
            action_pca, zero_score_raw, action_mean, action_std, action_active
        ),
        **shifted_pc,
    }
    predictions_pc: dict[str, np.ndarray] = {"history_only": history_prediction_pc}
    for label, action_pc in control_action_pc.items():
        predictions_pc[label] = _model_prediction(
            model=action_model,
            history_pc=score_history_pc,
            action_pc=action_pc,
            input_mean=action_input_mean,
            input_std=action_input_std,
            target_mean=target_mean,
            target_std=target_std,
        )

    score_target_flat = score_y_map.reshape(SCORE_CLIPS, -1).astype(np.float32)
    decoded: dict[str, np.ndarray] = {
        label: target_pca.inverse_transform(prediction).astype(np.float32)
        for label, prediction in predictions_pc.items()
    }
    errors: dict[str, np.ndarray] = {}
    for label, prediction in predictions_pc.items():
        errors[f"coefficient_error_{label}"] = np.mean(
            ((prediction - score_target_pc) / target_std) ** 2, axis=1
        ).astype(np.float64)
        errors[f"field_error_{label}"] = np.mean(
            (decoded[label] - score_target_flat) ** 2, axis=1
        ).astype(np.float64)
    oracle_decode = target_pca.inverse_transform(score_target_pc).astype(np.float32)
    fit_mean_decode = np.broadcast_to(target_pca.mean_, score_target_flat.shape)
    raw_zero_decode = np.zeros_like(score_target_flat)
    errors["reconstruction_error_oracle"] = np.mean(
        (oracle_decode - score_target_flat) ** 2, axis=1
    ).astype(np.float64)
    errors["reconstruction_error_fit_mean"] = np.mean(
        (fit_mean_decode - score_target_flat) ** 2, axis=1
    ).astype(np.float64)
    errors["reconstruction_error_raw_zero"] = np.mean(
        (raw_zero_decode - score_target_flat) ** 2, axis=1
    ).astype(np.float64)

    effects, gates = analyze_error_arrays(errors)
    decision = (
        "GO_FOR_GENERATOR_SCREEN" if bool(gates["all_passed"]) else "STOP_INTERACTION_EVENT_BOTTLENECK"
    )

    provenance_path = output / "input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for record in provenance:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    per_clip_path = output / "per_clip_metrics.jsonl"
    with per_clip_path.open("w") as handle:
        for position, item in enumerate(score_items):
            row = {
                "schema_version": SCHEMA_VERSION,
                "score_position": position,
                "manifest_index": int(item["manifest_index"]),
                "clip_id": item["clip_id"],
                "motion_stratum": int(item["motion_stratum"]),
                "donor_manifest_index": int(item["donor_manifest_index"]),
                "errors": {name: float(values[position]) for name, values in errors.items()},
                "source_split": "train",
                "validation_accessed": False,
                "protected_test_accessed": False,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    arrays_path = output / "derived_model_predictions.npz"
    np.savez_compressed(
        arrays_path,
        fit_indices=np.asarray([item["manifest_index"] for item in fit_items], dtype=np.int64),
        score_indices=np.asarray([item["manifest_index"] for item in score_items], dtype=np.int64),
        donor_positions=donor_positions,
        fit_history_pc=fit_history_pc,
        fit_target_pc=fit_target_pc,
        fit_action_pc=fit_action_pc,
        score_history_pc=score_history_pc,
        score_target_pc=score_target_pc,
        score_target_maps=score_y_map.astype(np.float16),
        history_pca_mean=history_pca.mean_,
        history_pca_components=history_pca.components_,
        history_pca_explained_variance_ratio=history_pca.explained_variance_ratio_,
        target_pca_mean=target_pca.mean_,
        target_pca_components=target_pca.components_,
        target_pca_explained_variance_ratio=target_pca.explained_variance_ratio_,
        action_mean=action_mean,
        action_std=action_std,
        action_active=action_active,
        action_pca_mean=action_pca.mean_,
        action_pca_components=action_pca.components_,
        action_pca_explained_variance_ratio=action_pca.explained_variance_ratio_,
        target_mean=target_mean,
        target_std=target_std,
        history_input_mean=history_input_mean,
        history_input_std=history_input_std,
        action_input_mean=action_input_mean,
        action_input_std=action_input_std,
        history_model_coef=history_model.coef_,
        history_model_intercept=history_model.intercept_,
        action_model_coef=action_model.coef_,
        action_model_intercept=action_model.intercept_,
        **{f"prediction_pc_{name}": value for name, value in predictions_pc.items()},
        **errors,
    )

    latency = np.asarray(render_latencies, dtype=np.float64)
    transition_audits = [
        audit
        for record in provenance
        for audit in record["transition_audits"]
    ]
    analysis = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_bottleneck_stage0_analysis",
            "created_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "decision": decision,
            "interpretation": (
                "the masked PCA32 event state cleared every prospective causal, action-specificity, and reconstruction gate"
                if gates["all_passed"]
                else "at least one prospective causal, action-specificity, or reconstruction gate failed; no generator integration is authorized"
            ),
            "population": {
                "source_split": "train",
                "fit_clips": FIT_CLIPS,
                "score_clips": SCORE_CLIPS,
                "fit_score_episode_disjoint": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "models": {
                "history_alpha": history_alpha,
                "action_alpha": action_alpha,
                "history_five_fold_cv": history_cv,
                "action_five_fold_cv": action_cv,
                "history_pca_variance_retained": float(
                    history_pca.explained_variance_ratio_.sum()
                ),
                "target_pca_variance_retained_fit": float(
                    target_pca.explained_variance_ratio_.sum()
                ),
                "action_pca_variance_retained": float(
                    action_pca.explained_variance_ratio_.sum()
                ),
                "prediction_action_sensitivity": {
                    label: float(np.sqrt(np.mean((predictions_pc["aligned"] - predictions_pc[label]) ** 2)))
                    for label in (
                        "episode_shuffled",
                        "train_mean",
                        "raw_zero",
                        "shift_minus1",
                        "shift_plus1",
                    )
                },
            },
            "metrics": {
                "mean_errors": {name: float(values.mean()) for name, values in errors.items()},
                "effects": effects,
                "gates": gates,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "primary_family_size": PRIMARY_FAMILY_SIZE,
                "reconstruction_family_size": RECONSTRUCTION_FAMILY_SIZE,
            },
            "event_field_diagnostics": {
                "transition_rows": len(transition_audits),
                "robot_core_fraction_mean": float(
                    np.mean([row["robot_core_fraction"] for row in transition_audits])
                ),
                "robot_excluded_fraction_mean": float(
                    np.mean([row["robot_excluded_fraction"] for row in transition_audits])
                ),
                "nonrobot_event_mass_mean": float(
                    np.mean([row["nonrobot_event_mass"] for row in transition_audits])
                ),
                "weighted_event_mass_mean": float(
                    np.mean([row["weighted_event_mass"] for row in transition_audits])
                ),
                "weighted_transport_rms_mean": float(
                    np.mean([row["weighted_transport_rms"] for row in transition_audits])
                ),
            },
            "renderer": {
                "official_abc_commit": official_commit,
                "official_scene": file_record(scene_path),
                "robot_only_xml_sha256": hashlib.sha256(robot_xml.encode()).hexdigest(),
                "moving_geom_ids": sorted(moving_geoms),
                "render_pose_mean_ms": float(latency.mean()),
                "render_pose_p50_ms": float(np.percentile(latency, 50)),
                "render_pose_p95_ms": float(np.percentile(latency, 95)),
                "render_pose_count_after_two_warmups_per_clip": int(len(latency)),
            },
            "artifacts": {
                "registration": file_record(output / "registration.json"),
                "frozen_source": file_record(frozen_source),
                "input_provenance": file_record(provenance_path),
                "per_clip_metrics": file_record(per_clip_path),
                "derived_model_predictions": file_record(arrays_path),
            },
            "claim_boundary": (
                "Train-only Stage-0 predictability of a masked change/transport PCA32. "
                "No video generator, FVD, control rollout, semantic object label, physical contact label, validation, or protected test was evaluated."
            ),
            "limitations": [
                "single train-only split and model seed",
                "bootstrap quantifies episode sampling rather than retraining variance",
                "nominal camera extrinsic, centered principal point, and no distortion",
                "rendered masks cannot perfectly remove shadows, grasped objects, or calibration residuals",
                "Farneback and luminance change are observational event proxies",
                "deterministic prediction cannot represent irreducible multimodal futures",
            ],
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    write_json(output / "analysis.json", analysis)
    complete = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_bottleneck_stage0_complete",
            "completed_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": decision,
            "status": "completed",
            "artifacts": {
                "analysis": file_record(output / "analysis.json"),
                **analysis["artifacts"],
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    write_json(output / "run_complete.json", complete)
    return analysis


def audit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    registration = read_json(output / "registration.json")
    analysis = read_json(output / "analysis.json")
    complete = read_json(output / "run_complete.json")
    for name, document in (
        ("registration", registration),
        ("analysis", analysis),
        ("run_complete", complete),
    ):
        validate_identity(document, name)
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise BottleneckError("analysis-registration binding differs")
    if complete["registration_identity_sha256"] != registration["identity_sha256"]:
        raise BottleneckError("completion-registration binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise BottleneckError("completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise BottleneckError("completion decision differs")
    if sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise BottleneckError("audit source bytes differ from registration")
    frozen = output / "frozen_interaction_event_bottleneck_stage0.py"
    if sha256_file(frozen) != registration["source"]["frozen_copy"]["sha256"]:
        raise BottleneckError("frozen source hash differs")

    fit = registration["split"]["fit"]
    score = registration["split"]["score"]
    if len(fit) != FIT_CLIPS or len(score) != SCORE_CLIPS:
        raise BottleneckError("registered split count differs")
    fit_episodes = {item["episode_dir"] for item in fit}
    score_episodes = {item["episode_dir"] for item in score}
    if len(fit_episodes) != FIT_CLIPS or len(score_episodes) != SCORE_CLIPS:
        raise BottleneckError("registered episodes are not unique")
    if fit_episodes & score_episodes:
        raise BottleneckError("fit/score episodes overlap")
    score_indexes = {int(item["manifest_index"]) for item in score}
    for item in score:
        if int(item["donor_manifest_index"]) not in score_indexes:
            raise BottleneckError("donor is outside score set")
        if item["donor_episode_dir"] == item["episode_dir"]:
            raise BottleneckError("donor equals native episode")
        if int(item["motion_stratum"]) not in range(MOTION_STRATA):
            raise BottleneckError("score stratum differs")

    for section in (analysis["artifacts"], complete["artifacts"]):
        for name, record in section.items():
            path = Path(record["path"])
            if not path.is_file():
                path = output / path.name
            if sha256_file(path) != record["sha256"]:
                raise BottleneckError(f"artifact hash mismatch: {name}")

    per_clip = read_jsonl(output / "per_clip_metrics.jsonl")
    provenance = read_jsonl(output / "input_provenance.jsonl")
    if len(per_clip) != SCORE_CLIPS:
        raise BottleneckError(f"per-clip metric row count differs: {len(per_clip)}")
    if len(provenance) != FIT_CLIPS + SCORE_CLIPS:
        raise BottleneckError(f"provenance row count differs: {len(provenance)}")
    with np.load(output / "derived_model_predictions.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if arrays["fit_indices"].shape != (FIT_CLIPS,) or arrays["score_indices"].shape != (
        SCORE_CLIPS,
    ):
        raise BottleneckError("stored index geometry differs")
    if arrays["score_target_maps"].shape != (SCORE_CLIPS, 8, 4, *WORK_HW):
        raise BottleneckError("stored score map geometry differs")
    error_arrays = {
        name: values
        for name, values in arrays.items()
        if name.startswith("coefficient_error_")
        or name.startswith("field_error_")
        or name.startswith("reconstruction_error_")
    }
    recomputed_effects, recomputed_gates = analyze_error_arrays(error_arrays)
    stored_effects = analysis["metrics"]["effects"]
    stored_gates = analysis["metrics"]["gates"]
    if json.dumps(recomputed_effects, sort_keys=True) != json.dumps(
        stored_effects, sort_keys=True
    ):
        raise BottleneckError("recomputed effects differ from analysis")
    if recomputed_gates != stored_gates:
        raise BottleneckError("recomputed gates differ from analysis")
    expected_decision = (
        "GO_FOR_GENERATOR_SCREEN"
        if recomputed_gates["all_passed"]
        else "STOP_INTERACTION_EVENT_BOTTLENECK"
    )
    if analysis["decision"] != expected_decision:
        raise BottleneckError("decision does not follow recomputed gates")

    false_flags = 0
    for name, value in (
        ("registration", registration),
        ("analysis", analysis),
        ("complete", complete),
        ("per_clip", per_clip),
        ("provenance", provenance),
    ):
        false_flags += require_false_flags(value, name)
    result = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_bottleneck_stage0_audit",
            "audited_at_utc": now(),
            "status": "audit_passed",
            "decision": analysis["decision"],
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "fit_clips": FIT_CLIPS,
            "score_clips": SCORE_CLIPS,
            "provenance_rows": len(provenance),
            "per_clip_metric_rows": len(per_clip),
            "recomputed_mandatory_effects": len(recomputed_effects["mandatory"]),
            "recomputed_diagnostic_effects": len(recomputed_effects["diagnostic"]),
            "explicit_false_flags": false_flags,
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--train-cache", type=Path, required=True)
    register_parser.add_argument("--train-manifest", type=Path, required=True)
    register_parser.add_argument("--preprocessed-root", type=Path, required=True)
    register_parser.add_argument("--raw-root", type=Path, required=True)
    register_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/experiments/INTERACTION_EVENT_BOTTLENECK_STAGE0_PROTOCOL.md"),
    )
    register_parser.add_argument("--expected-commit")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--official-abc-root", type=Path, required=True)
    run_parser.add_argument("--mujoco-gl", default="egl")

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        result = register(args)
        event = "registered"
    elif args.command == "run":
        result = run(args)
        event = "completed"
    elif args.command == "audit":
        result = audit(args.output)
        if args.report is not None:
            write_json(args.report, result)
        event = "audited"
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"event": event, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
