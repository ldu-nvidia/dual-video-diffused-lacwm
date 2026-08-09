#!/usr/bin/env python3
"""Train-only attribution gate for predicted tracking-corrected robot renders.

The workflow is deliberately split into three state transitions:

``register``
    Selects the held-out train episodes from action/camera metadata and seals
    the protocol before score-pool state or RGB is opened.
``prepare``
    Fits the causal residual predictor on disjoint train episodes and creates
    compact, hash-addressed RGB/trajectory bundles.
``evaluate``
    Renders every frozen arm, computes region-aware geometry and flow metrics,
    and applies the preregistered gate.

``audit`` is read-only.  Validation and protected test inputs are unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tools.abc_d405_nominal_geometry_probe import (
    CAMERA_TYPE,
    EXPECTED_ABC_COMMIT,
    OFFICIAL_ASSET_RELATIVE,
    OFFICIAL_SCENE_RELATIVE,
    _decode_top_calibration,
    _joint_qpos_addresses,
    _read_video_frames,
    build_robot_only_xml,
    nonwrapping_shift_pairs,
    observed_edges,
    set_observed_pose,
    silhouette_boundary,
)
from tools.stage0_nominal_tracking_residual import construct_clip_arrays


SCHEMA_VERSION = 1
MANIFEST_COUNT = 512
FIT_STOP = 384
SCORE_START = 384
SCORE_STOP = 512
SCORE_CLIPS = 24
MOTION_STRATA = 3
CLIPS_PER_STRATUM = 8
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
ACTION_DIM = 14
PADDED_ACTION_DIM = 23
PCA_COMPONENTS = 64
SEED = 1234
SELECTION_SEED = 20260808
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_SAMPLES = 20_000
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
ROBOT_BAND_PX = 12
FLOW_PIXEL_STRIDE = 2
FLOW_MIN_ORACLE_MOTION_PX = 0.25
GATE_FAVORABLE_CLIP_FRACTION = 0.60
PRIMARY_ALIGNMENT_MIN_RELATIVE_IMPROVEMENT_PERCENT = 5.0
ARMS = (
    "raw_command",
    "predicted_corrected",
    "hold_current",
    "episode_shuffled",
    "measured_oracle",
)
MANDATORY_METRICS = (
    "silhouette_boundary_chamfer_px",
    "rgb_robot_band_chamfer_px",
    "robot_flow_epe_px",
)


class AttributionError(RuntimeError):
    """Raised when an immutable or leakage contract is violated."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, digest: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    result: dict[str, Any] = {"path": str(resolved), "bytes": resolved.stat().st_size}
    if digest:
        result["sha256"] = sha256_file(resolved)
    return result


def canonical_identity(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["identity_sha256"] = canonical_identity(result)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
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


def validate_identity(payload: dict[str, Any], location: str) -> None:
    expected = payload.get("identity_sha256")
    if not isinstance(expected, str) or canonical_identity(payload) != expected:
        raise AttributionError(f"identity mismatch: {location}")


def require_false_flags(value: Any, location: str = "root") -> int:
    count = 0
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "protected_test_accessed":
                count += 1
                if child is not False:
                    raise AttributionError(
                        f"{child_location} must explicitly equal false, got {child!r}"
                    )
            count += require_false_flags(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += require_false_flags(child, f"{location}[{index}]")
    return count


def validate_train_manifest(rows: list[dict[str, Any]]) -> None:
    if len(rows) != MANIFEST_COUNT:
        raise AttributionError(f"train manifest has {len(rows)} rows, expected {MANIFEST_COUNT}")
    episodes: list[str] = []
    clips: list[str] = []
    for index, row in enumerate(rows):
        if row.get("split") != "train":
            raise AttributionError(f"row {index} is not split=train")
        if row.get("auxiliary_index") != index:
            raise AttributionError(f"row {index} auxiliary_index differs")
        if (row.get("sample_size"), row.get("chunk_size"), row.get("action_span")) != (
            SAMPLE_SIZE,
            CHUNK_SIZE,
            SAMPLE_SIZE * CHUNK_SIZE,
        ):
            raise AttributionError(f"row {index} clip geometry differs")
        start = int(row["start"])
        expected = list(range(start, start + SAMPLE_SIZE * CHUNK_SIZE, CHUNK_SIZE))
        if row.get("frame_indices") != expected:
            raise AttributionError(f"row {index} frame boundary ordering differs")
        episodes.append(str(row["episode_dir"]))
        clips.append(str(row["clip_id"]))
    if len(set(episodes)) != len(rows) or len(set(clips)) != len(rows):
        raise AttributionError("train rows must be unique episodes and clip IDs")


def cache_actions(cache_dir: Path) -> tuple[np.ndarray, dict[str, Any], Path]:
    metadata_path = cache_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("split") != "train" or int(metadata.get("clip_count", -1)) != MANIFEST_COUNT:
        raise AttributionError("only the immutable 512-row train cache is accepted")
    actions_path = cache_dir / str(metadata["actions_file"])
    if sha256_file(actions_path) != metadata.get("actions_sha256"):
        raise AttributionError("cached action hash differs from metadata")
    actions = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    expected = (MANIFEST_COUNT, SAMPLE_SIZE, CHUNK_SIZE, PADDED_ACTION_DIM)
    if tuple(actions.shape) != expected or actions.dtype != np.float32:
        raise AttributionError(f"cached action geometry differs: {actions.shape} {actions.dtype}")
    if float(np.max(np.abs(actions[..., ACTION_DIM:]))) != 0.0:
        raise AttributionError("cached padded action coordinates are nonzero")
    return actions, metadata, actions_path


def raw_mcap_path(row: dict[str, Any], preprocessed_root: Path, raw_root: Path) -> Path:
    episode = Path(row["episode_dir"]).resolve()
    try:
        relative = episode.relative_to(preprocessed_root.resolve())
    except ValueError as exc:
        raise AttributionError(
            f"episode {episode} is outside preprocessed root {preprocessed_root}"
        ) from exc
    path = raw_root.resolve() / relative / "episode.mcap"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_mcap_metadata(path: Path) -> dict[str, str]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("register requires mcap") from exc
    with path.open("rb") as handle:
        records = list(make_reader(handle).iter_metadata())
    if not records:
        raise AttributionError(f"MCAP has no metadata: {path}")
    return {str(key): str(value) for key, value in records[0].metadata.items()}


def planned_joint_motion(action: np.ndarray) -> float:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (SAMPLE_SIZE, CHUNK_SIZE, PADDED_ACTION_DIM):
        raise ValueError(f"unexpected action shape: {array.shape}")
    future = array[4:12, -1, :12]
    baseline = array[3, -1, :12]
    return float(np.sqrt(np.mean((future - baseline) ** 2)))


def selection_hash(clip_id: str) -> str:
    return hashlib.sha256(f"{clip_id}|{SELECTION_SEED}".encode()).hexdigest()


def select_motion_stratified(
    eligible: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(eligible) < SCORE_CLIPS:
        raise AttributionError(f"only {len(eligible)} D405 score candidates")
    ordered = sorted(eligible, key=lambda item: (float(item["planned_joint_motion_rms"]), item["clip_id"]))
    bins = np.array_split(np.asarray(ordered, dtype=object), MOTION_STRATA)
    selected: list[dict[str, Any]] = []
    for stratum, values in enumerate(bins):
        candidates = [dict(item) for item in values.tolist()]
        if len(candidates) < CLIPS_PER_STRATUM:
            raise AttributionError(f"motion stratum {stratum} has only {len(candidates)} rows")
        chosen = sorted(candidates, key=lambda item: (selection_hash(item["clip_id"]), item["clip_id"]))[
            :CLIPS_PER_STRATUM
        ]
        for rank, item in enumerate(chosen):
            item["motion_stratum"] = stratum
            item["selection_rank_within_stratum"] = rank
            item["selection_hash"] = selection_hash(item["clip_id"])
        for rank, item in enumerate(chosen):
            donor = chosen[(rank + 1) % len(chosen)]
            item["donor_manifest_index"] = int(donor["manifest_index"])
            item["donor_clip_id"] = str(donor["clip_id"])
            item["donor_episode_dir"] = str(donor["episode_dir"])
        selected.extend(chosen)
    if len(selected) != SCORE_CLIPS:
        raise AttributionError(f"selector produced {len(selected)} rows")
    for order, item in enumerate(selected):
        item["score_order"] = order
    return selected


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = read_jsonl(args.train_manifest)
    validate_train_manifest(rows)
    actions, metadata, actions_path = cache_actions(args.train_cache)

    eligible: list[dict[str, Any]] = []
    camera_type_counts: dict[str, int] = {}
    for index in range(SCORE_START, SCORE_STOP):
        row = rows[index]
        episode = Path(row["episode_dir"])
        state_path = episode / "states.npz"
        video_path = episode / "top.mp4"
        if not state_path.is_file() or not video_path.is_file():
            continue
        mcap_path = raw_mcap_path(row, args.preprocessed_root, args.raw_root)
        episode_metadata = read_mcap_metadata(mcap_path)
        camera_type = episode_metadata.get("top_camera_type", "missing")
        camera_type_counts[camera_type] = camera_type_counts.get(camera_type, 0) + 1
        if camera_type != CAMERA_TYPE:
            continue
        eligible.append(
            {
                "manifest_index": index,
                "clip_id": str(row["clip_id"]),
                "episode_dir": str(episode.resolve()),
                "raw_mcap": str(mcap_path.resolve()),
                "top_camera_type": camera_type,
                "planned_joint_motion_rms": planned_joint_motion(actions[index]),
                "protected_test_accessed": False,
            }
        )

    selected = select_motion_stratified(eligible)
    script = Path(__file__).resolve()
    registration = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "corrected_renderer_attribution_registration",
            "status": "registered_before_score_state_or_rgb_access",
            "created_at_utc": now(),
            "source": {
                **file_record(script),
                "git_commit": git_commit(script.parents[1]),
            },
            "inputs": {
                "train_manifest": file_record(args.train_manifest),
                "train_cache_metadata": file_record(args.train_cache / "metadata.json"),
                "train_actions": {
                    **file_record(actions_path),
                    "shape": list(actions.shape),
                    "dtype": str(actions.dtype),
                    "registered_sha256": metadata["actions_sha256"],
                },
                "preprocessed_root": str(args.preprocessed_root.resolve()),
                "raw_root": str(args.raw_root.resolve()),
                "score_pool_camera_type_counts": camera_type_counts,
                "score_pool_eligible_d405": len(eligible),
                "score_state_opened": False,
                "score_rgb_opened": False,
            },
            "split_contract": {
                "split": "train",
                "manifest_rows": MANIFEST_COUNT,
                "fit_indices": [0, FIT_STOP],
                "score_pool_indices": [SCORE_START, SCORE_STOP],
                "fit_score_episode_disjoint": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "selection": {
                "uses_rgb": False,
                "uses_measured_state": False,
                "uses_alignment_outcome": False,
                "camera_requirement": CAMERA_TYPE,
                "planned_motion_definition": (
                    "RMS future joint-command endpoint displacement at chunks 4..11 "
                    "from chunk-3 endpoint; gripper excluded"
                ),
                "motion_strata": MOTION_STRATA,
                "clips_per_stratum": CLIPS_PER_STRATUM,
                "selection_seed": SELECTION_SEED,
                "selected": selected,
            },
            "protocol": {
                "document": "docs/experiments/CORRECTED_RENDERER_ATTRIBUTION_PROTOCOL.md",
                "fit_model": "fit384 PCA64 history+action multi-output ridge",
                "arms": list(ARMS),
                "target_frames": list(range(5, 13)),
                "robot_band_px": ROBOT_BAND_PX,
                "flow_pixel_stride": FLOW_PIXEL_STRIDE,
                "flow_min_oracle_motion_px": FLOW_MIN_ORACLE_MOTION_PX,
                "timing_diagnostics": [-1, 1],
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "mandatory_metrics": list(MANDATORY_METRICS),
                "references": ["raw_command", "episode_shuffled"],
                "minimum_favorable_clip_fraction": GATE_FAVORABLE_CLIP_FRACTION,
                "primary_alignment_metric": "silhouette_boundary_chamfer_px",
                "primary_minimum_relative_improvement_percent": (
                    PRIMARY_ALIGNMENT_MIN_RELATIVE_IMPROVEMENT_PERCENT
                ),
                "paired_lower_bound_strictly_positive": True,
                "silhouette_iou_nonnegative_lower_bound": True,
                "all_conditions_required": True,
                "measured_oracle_is_privileged_diagnostic": True,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "protected_test_accessed": False,
        }
    )
    write_json(output / "registration.json", registration)
    return registration


def extract_rows(
    rows: list[dict[str, Any]],
    actions: np.ndarray,
    indices: Sequence[int],
    *,
    role: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "history",
            "future_actions",
            "nominal_endpoint",
            "realized_state",
            "target_residual",
            "hold_current_residual",
        )
    }
    provenance: list[dict[str, Any]] = []
    boundaries_all: list[np.ndarray] = []
    timestamps_all: list[np.ndarray] = []
    for position, index in enumerate(indices):
        row = rows[int(index)]
        state_path = Path(row["episode_dir"]) / "states.npz"
        with np.load(state_path, allow_pickle=False) as state:
            required = {"joint_states", "joint_actions", "gripper_states", "gripper_actions", "frame_ts"}
            if not required.issubset(state.files):
                raise AttributionError(f"missing state arrays: {state_path}")
            frame_indices = np.asarray(row["frame_indices"], dtype=np.int64)
            boundaries = np.concatenate(
                (state["joint_states"][frame_indices], state["gripper_states"][frame_indices]),
                axis=1,
            ).astype(np.float32)
            start = int(row["start"])
            stop = start + SAMPLE_SIZE * CHUNK_SIZE
            raw = np.concatenate(
                (state["joint_actions"][start:stop], state["gripper_actions"][start:stop]),
                axis=1,
            ).astype(np.float32).reshape(SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM)
            timestamps = np.asarray(state["frame_ts"][frame_indices], dtype=np.int64)
        cached = np.asarray(actions[int(index), ..., :ACTION_DIM], dtype=np.float32)
        if not np.array_equal(raw, cached):
            raise AttributionError(f"cache/raw action mismatch at train row {index}")
        derived = construct_clip_arrays(boundaries, cached)
        for name, value in derived.items():
            arrays[name].append(value)
        boundaries_all.append(boundaries)
        timestamps_all.append(timestamps)
        provenance.append(
            {
                "schema_version": SCHEMA_VERSION,
                "role": role,
                "position": position,
                "manifest_index": int(index),
                "clip_id": row["clip_id"],
                "episode_dir": row["episode_dir"],
                "states_npz": file_record(state_path),
                "split": "train",
                "validation_accessed": False,
                "protected_test_accessed": False,
            }
        )
    return (
        {name: np.stack(values) for name, values in arrays.items()},
        provenance,
        boundaries_all,
        timestamps_all,
    )


def feature_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = values.reshape(len(values), -1)
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    active = raw_std > 1e-8
    if int(active.sum()) < PCA_COMPONENTS:
        raise AttributionError(f"only {int(active.sum())} active dimensions for PCA64")
    return mean, np.where(active, raw_std, 1.0).astype(np.float32), active


def fit_pca(values: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    mean, std, active = feature_standardizer(values)
    flat = values.reshape(len(values), -1)
    normalized = ((flat - mean) / std)[:, active]
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    transformed = pca.fit_transform(normalized).astype(np.float32)
    return pca, transformed, mean, std, active


def transform_pca(
    pca: Any,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return pca.transform(((flat - mean) / std)[:, active]).astype(np.float32)


def choose_alpha(inputs: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        errors = []
        for fit_index, score_index in folds.split(inputs):
            model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            model.fit(inputs[fit_index], targets[fit_index])
            errors.append(float(np.mean((model.predict(inputs[score_index]) - targets[score_index]) ** 2)))
        scores[str(alpha)] = float(np.mean(errors))
    selected = min(ALPHAS, key=lambda value: (scores[str(value)], value))
    return float(selected), scores


def combine_native_history_with_action(
    native_history: np.ndarray,
    action_features: np.ndarray,
) -> np.ndarray:
    """Combine recipient history with the supplied (possibly donor) action only."""

    history = np.asarray(native_history)
    action = np.asarray(action_features)
    if history.ndim != 2 or action.ndim != 2 or len(history) != len(action):
        raise ValueError(
            f"history/action features must be aligned matrices, got {history.shape}/{action.shape}"
        )
    return np.concatenate((history, action), axis=1)


def render_clip_pose(pose: np.ndarray) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    result[..., 12:] = np.clip(result[..., 12:], 0.0, 1.0)
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    registration_path = output / "registration.json"
    registration = read_json(registration_path)
    validate_identity(registration, str(registration_path))
    if registration.get("kind") != "corrected_renderer_attribution_registration":
        raise AttributionError("unsupported registration kind")
    if sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise AttributionError("execution source differs from registration")
    if (output / "preparation.json").exists():
        raise FileExistsError(output / "preparation.json")

    manifest = Path(registration["inputs"]["train_manifest"]["path"])
    cache_dir = Path(registration["inputs"]["train_cache_metadata"]["path"]).parent
    rows = read_jsonl(manifest)
    validate_train_manifest(rows)
    actions, _, _ = cache_actions(cache_dir)
    fit_indices = list(range(FIT_STOP))
    selected = registration["selection"]["selected"]
    score_indices = [int(item["manifest_index"]) for item in selected]
    donor_indices = [int(item["donor_manifest_index"]) for item in selected]
    if set(fit_indices) & set(score_indices):
        raise AttributionError("fit and score indices overlap")

    fit, fit_provenance, _, _ = extract_rows(
        rows, actions, fit_indices, role="predictor_fit"
    )
    score, score_provenance, score_boundaries, score_timestamps = extract_rows(
        rows, actions, score_indices, role="renderer_score"
    )

    history_pca, fit_history, history_mean, history_std, history_active = fit_pca(
        fit["history"]
    )
    action_pca, fit_action, action_mean, action_std, action_active = fit_pca(
        fit["future_actions"]
    )
    score_history = transform_pca(
        history_pca, score["history"], history_mean, history_std, history_active
    )
    score_action = transform_pca(
        action_pca, score["future_actions"], action_mean, action_std, action_active
    )

    target_fit_flat = fit["target_residual"].reshape(FIT_STOP, -1)
    target_mean = target_fit_flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std_raw = target_fit_flat.std(axis=0, dtype=np.float64).astype(np.float32)
    target_std = np.where(target_std_raw > 1e-8, target_std_raw, 1.0).astype(np.float32)
    fit_target = ((target_fit_flat - target_mean) / target_std).astype(np.float32)

    fit_raw = combine_native_history_with_action(fit_history, fit_action)
    input_mean = fit_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std_raw = fit_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.where(input_std_raw > 1e-8, input_std_raw, 1.0).astype(np.float32)
    fit_input = (fit_raw - input_mean) / input_std
    alpha, cv = choose_alpha(fit_input, fit_target)
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(
        fit_input, fit_target
    )

    def predict(action_features: np.ndarray) -> np.ndarray:
        raw_input = combine_native_history_with_action(score_history, action_features)
        standardized = (raw_input - input_mean) / input_std
        prediction = model.predict(standardized).astype(np.float32)
        return (prediction * target_std + target_mean).reshape(SCORE_CLIPS, 8, ACTION_DIM)

    position_by_index = {index: position for position, index in enumerate(score_indices)}
    donor_positions = np.asarray([position_by_index[index] for index in donor_indices], dtype=np.int64)
    aligned_residual = predict(score_action)
    shuffled_residual = predict(score_action[donor_positions])
    target_residual = score["target_residual"]
    nominal = score["nominal_endpoint"]

    model_path = output / "model_state_and_predictions.npz"
    np.savez_compressed(
        model_path,
        fit_indices=np.asarray(fit_indices, dtype=np.int64),
        score_indices=np.asarray(score_indices, dtype=np.int64),
        donor_indices=np.asarray(donor_indices, dtype=np.int64),
        donor_positions=donor_positions,
        history_feature_mean=history_mean,
        history_feature_std=history_std,
        history_feature_active=history_active,
        history_pca_mean=history_pca.mean_,
        history_pca_components=history_pca.components_,
        history_pca_explained_variance_ratio=history_pca.explained_variance_ratio_,
        action_feature_mean=action_mean,
        action_feature_std=action_std,
        action_feature_active=action_active,
        action_pca_mean=action_pca.mean_,
        action_pca_components=action_pca.components_,
        action_pca_explained_variance_ratio=action_pca.explained_variance_ratio_,
        target_mean=target_mean,
        target_std=target_std,
        input_mean=input_mean,
        input_std=input_std,
        model_coef=model.coef_,
        model_intercept=model.intercept_,
        aligned_residual=aligned_residual,
        shuffled_residual=shuffled_residual,
        target_residual=target_residual,
    )

    bundle_dir = output / "bundles"
    bundle_dir.mkdir()
    bundle_records: list[dict[str, Any]] = []
    source_provenance = [*fit_provenance, *score_provenance]
    preprocessed_root = Path(registration["inputs"]["preprocessed_root"])
    raw_root = Path(registration["inputs"]["raw_root"])
    for position, item in enumerate(selected):
        row = rows[score_indices[position]]
        episode = Path(row["episode_dir"])
        video_path = episode / "top.mp4"
        state_path = episode / "states.npz"
        mcap_path = raw_mcap_path(row, preprocessed_root, raw_root)
        calibration = _decode_top_calibration(mcap_path)
        frame_indices = np.asarray(row["frame_indices"][4:13], dtype=np.int64)
        rgb = _read_video_frames(video_path, frame_indices.tolist())
        boundaries = score_boundaries[position]
        q0 = boundaries[4]
        pose_unclipped = {
            "raw_command": np.concatenate((q0[None], nominal[position]), axis=0),
            "predicted_corrected": np.concatenate(
                (q0[None], nominal[position] + aligned_residual[position]), axis=0
            ),
            "hold_current": np.broadcast_to(q0, (9, ACTION_DIM)).copy(),
            "episode_shuffled": np.concatenate(
                (q0[None], nominal[position] + shuffled_residual[position]), axis=0
            ),
            "measured_oracle": boundaries[4:13].copy(),
        }
        arrays: dict[str, np.ndarray] = {
            "rgb": rgb,
            "frame_indices": frame_indices,
            "frame_ts": score_timestamps[position][4:13],
            "K": np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3),
            "D": np.asarray(calibration["D"], dtype=np.float64),
            "target_residual": target_residual[position],
            "aligned_residual_prediction": aligned_residual[position],
            "shuffled_residual_prediction": shuffled_residual[position],
        }
        for arm, pose in pose_unclipped.items():
            arrays[f"pose_unclipped_{arm}"] = np.asarray(pose, dtype=np.float32)
            arrays[f"pose_rendered_{arm}"] = render_clip_pose(pose)
        bundle_path = bundle_dir / f"{item['clip_id']}.npz"
        np.savez_compressed(bundle_path, **arrays)
        bundle_metadata = seal(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "corrected_renderer_attribution_bundle",
                "registration_identity_sha256": registration["identity_sha256"],
                "score_order": position,
                "manifest_index": score_indices[position],
                "motion_stratum": int(item["motion_stratum"]),
                "clip_id": item["clip_id"],
                "episode_dir": str(episode.resolve()),
                "donor_manifest_index": donor_indices[position],
                "donor_clip_id": item["donor_clip_id"],
                "split": "train",
                "validation_accessed": False,
                "protected_test_accessed": False,
                "camera_type": calibration["camera_type"],
                "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
                "bundle": file_record(bundle_path),
                "source_files": {
                    "states_npz": file_record(state_path),
                    "top_mp4": file_record(video_path),
                    "raw_mcap": file_record(mcap_path),
                },
                "gripper_render_clip": "coordinates 12:14 clipped to [0,1]; unclipped retained",
            }
        )
        metadata_path = bundle_dir / f"{item['clip_id']}.json"
        write_json(metadata_path, bundle_metadata)
        bundle_records.append(
            {
                "clip_id": item["clip_id"],
                "bundle": file_record(bundle_path),
                "metadata": file_record(metadata_path),
                "metadata_identity_sha256": bundle_metadata["identity_sha256"],
                "protected_test_accessed": False,
            }
        )

    provenance_path = output / "input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for item in source_provenance:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    target_error = {
        "predicted_corrected_state_mse": np.mean(
            (nominal + aligned_residual - score["realized_state"]) ** 2, axis=(1, 2)
        ),
        "raw_command_state_mse": np.mean(
            (nominal - score["realized_state"]) ** 2, axis=(1, 2)
        ),
        "episode_shuffled_state_mse": np.mean(
            (nominal + shuffled_residual - score["realized_state"]) ** 2, axis=(1, 2)
        ),
        "hold_current_state_mse": np.mean(
            (score["hold_current_residual"] - target_residual) ** 2, axis=(1, 2)
        ),
    }
    preparation = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "corrected_renderer_attribution_preparation",
            "created_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "status": "prepared_without_renderer_outcome_inspection",
            "split": "train",
            "fit_clips": FIT_STOP,
            "score_clips": SCORE_CLIPS,
            "fit_score_episode_disjoint": True,
            "model": {
                "selected_alpha": alpha,
                "five_fold_fit384_cv_standardized_mse": cv,
                "history_pca_variance_retained": float(history_pca.explained_variance_ratio_.sum()),
                "action_pca_variance_retained": float(action_pca.explained_variance_ratio_.sum()),
            },
            "state_space_diagnostic_only": {
                name: {
                    "mean": float(values.mean()),
                    "per_clip": values.tolist(),
                }
                for name, values in target_error.items()
            },
            "bundles": bundle_records,
            "artifacts": {
                "registration": file_record(registration_path),
                "source": file_record(Path(__file__).resolve()),
                "model_state_and_predictions": file_record(model_path),
                "input_provenance": file_record(provenance_path),
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    write_json(output / "preparation.json", preparation)
    complete = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "corrected_renderer_attribution_preparation_complete",
            "completed_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "artifacts": {
                "preparation": file_record(output / "preparation.json"),
                **preparation["artifacts"],
            },
            "status": "completed",
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    write_json(output / "preparation_complete.json", complete)
    return preparation


@dataclass(frozen=True)
class LoadedBundle:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class PoseRender:
    mask: np.ndarray
    geom_id: np.ndarray
    depth: np.ndarray
    geom_xpos: np.ndarray
    geom_xmat: np.ndarray
    camera_xpos: np.ndarray
    camera_xmat: np.ndarray
    fy: float
    render_ms: float


def load_bundles(output: Path, preparation: dict[str, Any]) -> list[LoadedBundle]:
    bundles: list[LoadedBundle] = []
    for expected_order, record in enumerate(preparation["bundles"]):
        metadata_path = Path(record["metadata"]["path"])
        bundle_path = Path(record["bundle"]["path"])
        if not metadata_path.is_file():
            metadata_path = output / "bundles" / metadata_path.name
        if not bundle_path.is_file():
            bundle_path = output / "bundles" / bundle_path.name
        metadata = read_json(metadata_path)
        validate_identity(metadata, str(metadata_path))
        if metadata.get("score_order") != expected_order:
            raise AttributionError("bundle score order differs")
        if metadata.get("split") != "train":
            raise AttributionError("non-train bundle rejected")
        if metadata.get("validation_accessed") is not False:
            raise AttributionError("bundle validation flag differs")
        if metadata.get("protected_test_accessed") is not False:
            raise AttributionError("bundle protected-test flag differs")
        if sha256_file(bundle_path) != metadata["bundle"]["sha256"]:
            raise AttributionError(f"bundle hash mismatch: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        required = {"rgb", "K", "D", "frame_indices", "frame_ts"}
        required.update(f"pose_rendered_{arm}" for arm in ARMS)
        if not required.issubset(arrays):
            raise AttributionError(f"bundle {bundle_path} lacks required arrays")
        if arrays["rgb"].shape[:1] != (9,):
            raise AttributionError(f"bundle RGB geometry differs: {bundle_path}")
        for arm in ARMS:
            if arrays[f"pose_rendered_{arm}"].shape != (9, ACTION_DIM):
                raise AttributionError(f"bundle pose geometry differs for {arm}")
        bundles.append(LoadedBundle(metadata, arrays))
    if len(bundles) != SCORE_CLIPS:
        raise AttributionError(f"loaded {len(bundles)} bundles, expected {SCORE_CLIPS}")
    return bundles


def moving_geom_ids(model: Any, mujoco: Any) -> set[int]:
    roots = {"left_arm", "right_arm"}
    result: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name in roots:
            continue  # fixed base geometry is deliberately excluded
        ancestor = body_id
        while ancestor > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, ancestor)
            if name in roots:
                result.add(geom_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    if not result:
        raise AttributionError("official model yielded no articulated robot geometries")
    return result


def render_pose(
    *,
    model: Any,
    data: Any,
    mujoco: Any,
    renderer: Any,
    option: Any,
    addresses: dict[str, int],
    camera_id: int,
    moving_geoms: set[int],
    pose: np.ndarray,
    fy: float,
) -> PoseRender:
    started = time.perf_counter()
    set_observed_pose(
        model,
        data,
        mujoco,
        addresses,
        np.asarray(pose[:12], dtype=np.float64),
        np.asarray(pose[12:], dtype=np.float64),
    )
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="top", scene_option=option)
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="top", scene_option=option)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    geom_id = np.asarray(segmentation[..., 0], dtype=np.int32)
    object_type = np.asarray(segmentation[..., 1], dtype=np.int32)
    is_geom = object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
    moving = np.isin(geom_id, np.asarray(sorted(moving_geoms), dtype=np.int32))
    mask = is_geom & moving
    if not np.any(mask):
        raise AttributionError("articulated robot mask is empty")
    return PoseRender(
        mask=mask,
        geom_id=geom_id,
        depth=np.asarray(depth, dtype=np.float32),
        geom_xpos=np.asarray(data.geom_xpos, dtype=np.float64).copy(),
        geom_xmat=np.asarray(data.geom_xmat, dtype=np.float64).reshape(model.ngeom, 3, 3).copy(),
        camera_xpos=np.asarray(data.cam_xpos[camera_id], dtype=np.float64).copy(),
        camera_xmat=np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy(),
        fy=float(fy),
        render_ms=elapsed_ms,
    )


def distance_transform_to(mask: np.ndarray) -> np.ndarray:
    import cv2

    boolean = np.asarray(mask, dtype=bool)
    if not np.any(boolean):
        raise AttributionError("distance-transform target is empty")
    return cv2.distanceTransform((~boolean).astype(np.uint8) * 255, cv2.DIST_L2, 3)


def symmetric_boundary_chamfer(mask: np.ndarray, reference: np.ndarray) -> float:
    candidate_boundary = silhouette_boundary(mask)
    reference_boundary = silhouette_boundary(reference)
    candidate_to_reference = distance_transform_to(reference_boundary)[candidate_boundary]
    reference_to_candidate = distance_transform_to(candidate_boundary)[reference_boundary]
    return float(0.5 * (candidate_to_reference.mean() + reference_to_candidate.mean()))


def mask_iou(mask: np.ndarray, reference: np.ndarray) -> float:
    intersection = int(np.logical_and(mask, reference).sum())
    union = int(np.logical_or(mask, reference).sum())
    if union == 0:
        raise AttributionError("mask union is empty")
    return float(intersection / union)


def rgb_robot_band_metrics(
    candidate_mask: np.ndarray,
    oracle_mask: np.ndarray,
    rgb_edges: np.ndarray,
    *,
    band_px: int = ROBOT_BAND_PX,
) -> dict[str, float]:
    import cv2

    kernel_size = 2 * int(band_px) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    region = cv2.dilate(np.asarray(oracle_mask, dtype=np.uint8), kernel) > 0
    regional_edges = np.asarray(rgb_edges, dtype=bool) & region
    if int(regional_edges.sum()) < 10:
        raise AttributionError("fixed oracle robot band contains fewer than 10 RGB edge pixels")
    boundary = silhouette_boundary(candidate_mask)
    values = distance_transform_to(regional_edges)[boundary]
    return {
        "rgb_robot_band_chamfer_px": float(values.mean()),
        "rgb_robot_band_edge_support_3px": float(np.mean(values <= 3.0)),
        "candidate_boundary_in_oracle_band_fraction": float(np.mean(region[boundary])),
        "fixed_robot_band_fraction": float(region.mean()),
        "fixed_robot_band_rgb_edge_pixels": float(regional_edges.sum()),
    }


def camera_points_from_depth(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    *,
    width: int,
    height: int,
    fy: float,
) -> np.ndarray:
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.stack(
        (
            (x - cx) * depth / fy,
            -(y - cy) * depth / fy,
            -depth,
        ),
        axis=1,
    )


def project_world(
    world: np.ndarray,
    render: PoseRender,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    camera = (world - render.camera_xpos) @ render.camera_xmat
    forward = -camera[:, 2]
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    xy = np.stack(
        (
            cx + render.fy * camera[:, 0] / forward,
            cy - render.fy * camera[:, 1] / forward,
        ),
        axis=1,
    )
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(forward) & (forward > 1e-6)
    return xy, valid


def transport_local_points(
    local: np.ndarray,
    geom_ids: np.ndarray,
    render: PoseRender,
) -> np.ndarray:
    world = np.empty_like(local, dtype=np.float64)
    for geom_id in np.unique(geom_ids):
        keep = geom_ids == geom_id
        world[keep] = (
            render.geom_xpos[int(geom_id)]
            + local[keep] @ render.geom_xmat[int(geom_id)].T
        )
    return world


def robot_flow_error(
    candidate_source: PoseRender,
    candidate_target: PoseRender,
    oracle_source: PoseRender,
    oracle_target: PoseRender,
    *,
    far_depth: float,
    stride: int = FLOW_PIXEL_STRIDE,
    min_oracle_motion_px: float = FLOW_MIN_ORACLE_MOTION_PX,
) -> dict[str, float]:
    height, width = oracle_source.mask.shape
    yy, xx = np.indices((height, width))
    selection = (
        oracle_source.mask
        & (yy % stride == 0)
        & (xx % stride == 0)
        & np.isfinite(oracle_source.depth)
        & (oracle_source.depth > 0.0)
        & (oracle_source.depth < 0.99 * far_depth)
    )
    y = yy[selection].astype(np.float64)
    x = xx[selection].astype(np.float64)
    depth = oracle_source.depth[selection].astype(np.float64)
    geom_ids = oracle_source.geom_id[selection].astype(np.int64)
    if len(x) == 0:
        raise AttributionError("oracle flow source support is empty")

    camera_points = camera_points_from_depth(
        x, y, depth, width=width, height=height, fy=oracle_source.fy
    )
    world_source = oracle_source.camera_xpos + camera_points @ oracle_source.camera_xmat.T
    local = np.empty_like(world_source)
    for geom_id in np.unique(geom_ids):
        keep = geom_ids == geom_id
        local[keep] = (
            (world_source[keep] - oracle_source.geom_xpos[int(geom_id)])
            @ oracle_source.geom_xmat[int(geom_id)]
        )

    oracle_source_world = transport_local_points(local, geom_ids, oracle_source)
    oracle_target_world = transport_local_points(local, geom_ids, oracle_target)
    candidate_source_world = transport_local_points(local, geom_ids, candidate_source)
    candidate_target_world = transport_local_points(local, geom_ids, candidate_target)
    oracle_source_xy, valid_os = project_world(
        oracle_source_world, oracle_source, width=width, height=height
    )
    oracle_target_xy, valid_ot = project_world(
        oracle_target_world, oracle_target, width=width, height=height
    )
    candidate_source_xy, valid_cs = project_world(
        candidate_source_world, candidate_source, width=width, height=height
    )
    candidate_target_xy, valid_ct = project_world(
        candidate_target_world, candidate_target, width=width, height=height
    )
    source_xy = np.stack((x, y), axis=1)
    reprojection = np.linalg.norm(oracle_source_xy - source_xy, axis=1)
    oracle_flow = oracle_target_xy - oracle_source_xy
    candidate_flow = candidate_target_xy - candidate_source_xy
    oracle_motion = np.linalg.norm(oracle_flow, axis=1)
    oracle_target_in_frame = (
        valid_ot
        & (oracle_target_xy[:, 0] >= 0.0)
        & (oracle_target_xy[:, 0] <= width - 1.0)
        & (oracle_target_xy[:, 1] >= 0.0)
        & (oracle_target_xy[:, 1] <= height - 1.0)
    )
    primary = valid_os & oracle_target_in_frame & (oracle_motion >= min_oracle_motion_px)
    primary_count = int(primary.sum())
    candidate_valid = valid_cs & valid_ct
    diagonal = math.hypot(width, height)
    epe = np.full(len(x), diagonal, dtype=np.float64)
    jointly_valid = candidate_valid & np.isfinite(candidate_flow).all(axis=1)
    epe[jointly_valid] = np.linalg.norm(
        candidate_flow[jointly_valid] - oracle_flow[jointly_valid], axis=1
    )
    values = epe[primary]
    if primary_count < 10:
        return {
            "robot_flow_epe_px": None,
            "robot_flow_epe_p95_px": None,
            "robot_flow_epe_sum_px": 0.0,
            "oracle_flow_magnitude_px": None,
            "candidate_projection_valid_fraction": None,
            "oracle_source_reprojection_rmse_px": float(np.sqrt(np.mean(reprojection**2))),
            "oracle_source_sample_count": float(len(x)),
            "oracle_moving_flow_sample_count": float(primary_count),
            "oracle_target_in_frame_fraction": float(oracle_target_in_frame.mean()),
            "flow_transition_scored": False,
        }
    return {
        "robot_flow_epe_px": float(values.mean()),
        "robot_flow_epe_p95_px": float(np.percentile(values, 95)),
        "robot_flow_epe_sum_px": float(values.sum()),
        "oracle_flow_magnitude_px": float(oracle_motion[primary].mean()),
        "candidate_projection_valid_fraction": float(candidate_valid[primary].mean()),
        "oracle_source_reprojection_rmse_px": float(np.sqrt(np.mean(reprojection**2))),
        "oracle_source_sample_count": float(len(x)),
        "oracle_moving_flow_sample_count": float(primary.sum()),
        "oracle_target_in_frame_fraction": float(oracle_target_in_frame.mean()),
        "flow_transition_scored": True,
    }


def paired_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed_offset: int,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) == 0:
        raise ValueError("paired vectors must be nonempty and same-shape")
    difference = candidate - reference if higher_is_better else reference - candidate
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(difference), size=(BOOTSTRAP_SAMPLES, len(difference)))
    samples = difference[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    denominator = float(reference.mean())
    relative = None
    if abs(denominator) > 1e-12:
        if higher_is_better:
            relative = 100.0 * (float(candidate.mean()) - denominator) / abs(denominator)
        else:
            relative = 100.0 * (denominator - float(candidate.mean())) / abs(denominator)
    return {
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "paired_favorable_difference_mean": float(difference.mean()),
        "paired_bootstrap_95_ci": [float(low), float(high)],
        "relative_improvement_percent": relative,
        "favorable_clip_fraction": float(np.mean(difference > 0.0)),
        "paired_clips": int(len(difference)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
        "higher_is_better": higher_is_better,
    }


def aggregate_clip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for clip_id in sorted({str(row["clip_id"]) for row in rows}):
        clip_rows = [row for row in rows if row["clip_id"] == clip_id]
        for arm in ARMS:
            arm_rows = [row for row in clip_rows if row["arm"] == arm]
            if len(arm_rows) != 8:
                raise AttributionError(f"{clip_id}/{arm} has {len(arm_rows)} rows")
            metric_names = sorted(arm_rows[0]["metrics"])
            flow_count = float(
                np.sum(
                    [row["metrics"]["oracle_moving_flow_sample_count"] for row in arm_rows]
                )
            )
            if flow_count < 10:
                raise AttributionError(f"{clip_id}/{arm} has no identifiable robot-flow support")
            aggregated_metrics: dict[str, float] = {}
            for name in metric_names:
                if name == "robot_flow_epe_px":
                    aggregated_metrics[name] = float(
                        np.sum([row["metrics"]["robot_flow_epe_sum_px"] for row in arm_rows])
                        / flow_count
                    )
                    continue
                values = [row["metrics"][name] for row in arm_rows]
                numeric = [float(value) for value in values if value is not None]
                if not numeric:
                    continue
                if name in {"robot_flow_epe_sum_px", "oracle_moving_flow_sample_count"}:
                    aggregated_metrics[name] = float(np.sum(numeric))
                elif name == "flow_transition_scored":
                    aggregated_metrics[name] = float(np.sum(numeric))
                else:
                    aggregated_metrics[name] = float(np.mean(numeric))
            aggregates.append(
                {
                    "clip_id": clip_id,
                    "arm": arm,
                    "metrics": aggregated_metrics,
                    "protected_test_accessed": False,
                }
            )
    return aggregates


def _overlay_boundaries(
    path: Path,
    rgb: np.ndarray,
    raw: np.ndarray,
    predicted: np.ndarray,
    oracle: np.ndarray,
    label: str,
) -> None:
    import cv2

    image = np.asarray(rgb, dtype=np.uint8).copy()
    image[silhouette_boundary(raw)] = np.asarray([245, 70, 70], dtype=np.uint8)
    image[silhouette_boundary(predicted)] = np.asarray([40, 230, 80], dtype=np.uint8)
    image[silhouette_boundary(oracle)] = np.asarray([40, 120, 245], dtype=np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(
        bgr,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write overlay {path}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import cv2  # noqa: F401
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("evaluate requires mujoco and opencv-python-headless") from exc

    output = args.output.resolve()
    if (output / "analysis.json").exists():
        raise FileExistsError(output / "analysis.json")
    registration = read_json(output / "registration.json")
    preparation = read_json(output / "preparation.json")
    validate_identity(registration, "registration")
    validate_identity(preparation, "preparation")
    if preparation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise AttributionError("preparation does not bind registration")
    if sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise AttributionError("evaluation source differs from registration")
    bundles = load_bundles(output, preparation)

    official_root = args.official_abc_root.resolve()
    official_commit = git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise AttributionError(
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
        raise AttributionError("official model lacks top camera")
    moving_geoms = moving_geom_ids(model, mujoco)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    far_depth = float(model.stat.extent * model.vis.map.zfar)

    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    steady_render_latencies: list[float] = []
    overlay_dir = output / "overlays"
    overlay_dir.mkdir()
    overlay_records: list[dict[str, Any]] = []
    renderer = None
    renderer_shape: tuple[int, int] | None = None
    try:
        for bundle in bundles:
            arrays = bundle.arrays
            rgb = arrays["rgb"]
            _, height, width, channels = rgb.shape
            if channels != 3:
                raise AttributionError("RGB channel geometry differs")
            if renderer is not None:
                renderer.close()
            renderer = mujoco.Renderer(model, height=height, width=width)
            renderer_shape = (height, width)
            clip_latency_start = len(render_latencies)
            fy = float(arrays["K"][1, 1])
            model.cam_fovy[camera_id] = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

            rendered: dict[str, list[PoseRender]] = {}
            for arm in ARMS:
                poses = arrays[f"pose_rendered_{arm}"]
                rendered[arm] = [
                    render_pose(
                        model=model,
                        data=data,
                        mujoco=mujoco,
                        renderer=renderer,
                        option=option,
                        addresses=addresses,
                        camera_id=camera_id,
                        moving_geoms=moving_geoms,
                        pose=pose,
                        fy=fy,
                    )
                    for pose in poses
                ]
                render_latencies.extend(item.render_ms for item in rendered[arm])
            clip_latencies = render_latencies[clip_latency_start:]
            steady_render_latencies.extend(
                clip_latencies[2:] if len(clip_latencies) > 2 else clip_latencies
            )

            clip_rows: list[dict[str, Any]] = []
            for target_index in range(1, 9):
                oracle = rendered["measured_oracle"][target_index]
                edges = observed_edges(rgb[target_index])
                for arm in ARMS:
                    candidate = rendered[arm][target_index]
                    rgb_metrics = rgb_robot_band_metrics(candidate.mask, oracle.mask, edges)
                    flow_metrics = robot_flow_error(
                        rendered[arm][target_index - 1],
                        candidate,
                        rendered["measured_oracle"][target_index - 1],
                        oracle,
                        far_depth=far_depth,
                    )
                    metrics = {
                        "silhouette_boundary_chamfer_px": symmetric_boundary_chamfer(
                            candidate.mask, oracle.mask
                        ),
                        "silhouette_iou": mask_iou(candidate.mask, oracle.mask),
                        **rgb_metrics,
                        **flow_metrics,
                    }
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "clip_id": bundle.metadata["clip_id"],
                        "score_order": bundle.metadata["score_order"],
                        "motion_stratum": bundle.metadata["motion_stratum"],
                        "frame_ordinal": target_index,
                        "frame_index": int(arrays["frame_indices"][target_index]),
                        "arm": arm,
                        "metrics": metrics,
                        "split": "train",
                        "validation_accessed": False,
                        "protected_test_accessed": False,
                    }
                    rows.append(row)
                    clip_rows.append(row)

            # Nonwrapping lead/lag timing diagnostics reuse the already rendered poses.
            for shift in (-1, 1):
                source_ordinals, shifted_ordinals = nonwrapping_shift_pairs(8, shift)
                for source_ordinal, shifted_ordinal in zip(source_ordinals, shifted_ordinals):
                    rgb_index = int(source_ordinal + 1)
                    pose_index = int(shifted_ordinal + 1)
                    oracle = rendered["measured_oracle"][rgb_index]
                    edges = observed_edges(rgb[rgb_index])
                    for arm in ARMS:
                        candidate = rendered[arm][pose_index]
                        regional = rgb_robot_band_metrics(candidate.mask, oracle.mask, edges)
                        timing_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "clip_id": bundle.metadata["clip_id"],
                                "arm": arm,
                                "shift_clip_steps": shift,
                                "rgb_frame_ordinal": rgb_index,
                                "pose_frame_ordinal": pose_index,
                                "silhouette_boundary_chamfer_px": symmetric_boundary_chamfer(
                                    candidate.mask, oracle.mask
                                ),
                                "rgb_robot_band_chamfer_px": regional[
                                    "rgb_robot_band_chamfer_px"
                                ],
                                "split": "train",
                                "validation_accessed": False,
                                "protected_test_accessed": False,
                            }
                        )

            predicted_rows = [
                row for row in clip_rows if row["arm"] == "predicted_corrected"
            ]
            worst = max(
                predicted_rows,
                key=lambda row: row["metrics"]["rgb_robot_band_chamfer_px"],
            )
            index = int(worst["frame_ordinal"])
            overlay_path = overlay_dir / f"worst_{bundle.metadata['clip_id'][:12]}.png"
            _overlay_boundaries(
                overlay_path,
                rgb[index],
                rendered["raw_command"][index].mask,
                rendered["predicted_corrected"][index].mask,
                rendered["measured_oracle"][index].mask,
                f"raw=red pred=green oracle=blue frame={int(arrays['frame_indices'][index])}",
            )
            overlay_records.append(
                {
                    "clip_id": bundle.metadata["clip_id"],
                    "worst_predicted_rgb_robot_band_chamfer_px": worst["metrics"][
                        "rgb_robot_band_chamfer_px"
                    ],
                    "file": file_record(overlay_path),
                    "protected_test_accessed": False,
                }
            )
    finally:
        if renderer is not None:
            renderer.close()

    row_path = output / "frame_transition_metrics.jsonl"
    with row_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    timing_path = output / "nonwrapping_timing_metrics.jsonl"
    with timing_path.open("w") as handle:
        for row in timing_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    aggregates = aggregate_clip_rows(rows)
    aggregate_path = output / "clip_metrics.jsonl"
    with aggregate_path.open("w") as handle:
        for row in aggregates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    by_arm: dict[str, dict[str, np.ndarray]] = {}
    clip_order = [bundle.metadata["clip_id"] for bundle in bundles]
    for arm in ARMS:
        arm_rows = {row["clip_id"]: row for row in aggregates if row["arm"] == arm}
        by_arm[arm] = {
            name: np.asarray([arm_rows[clip]["metrics"][name] for clip in clip_order])
            for name in arm_rows[clip_order[0]]["metrics"]
        }

    effects: dict[str, dict[str, Any]] = {}
    offset = 0
    for reference in ("raw_command", "episode_shuffled"):
        for metric in MANDATORY_METRICS:
            offset += 1
            effects[f"predicted_corrected_vs_{reference}:{metric}"] = paired_difference(
                by_arm[reference][metric],
                by_arm["predicted_corrected"][metric],
                seed_offset=offset,
            )
        offset += 1
        effects[f"predicted_corrected_vs_{reference}:silhouette_iou"] = paired_difference(
            by_arm[reference]["silhouette_iou"],
            by_arm["predicted_corrected"]["silhouette_iou"],
            seed_offset=offset,
            higher_is_better=True,
        )

    mandatory_gates: dict[str, bool] = {}
    for label, effect in effects.items():
        if label.endswith(":silhouette_iou"):
            mandatory_gates[label] = bool(
                effect["paired_favorable_difference_mean"] > 0.0
                and effect["paired_bootstrap_95_ci"][0] >= 0.0
            )
        else:
            mandatory_gates[label] = bool(
                effect["paired_favorable_difference_mean"] > 0.0
                and effect["paired_bootstrap_95_ci"][0] > 0.0
                and effect["favorable_clip_fraction"] >= GATE_FAVORABLE_CLIP_FRACTION
            )
            if label.endswith(":silhouette_boundary_chamfer_px"):
                mandatory_gates[label] = bool(
                    mandatory_gates[label]
                    and effect["relative_improvement_percent"]
                    >= PRIMARY_ALIGNMENT_MIN_RELATIVE_IMPROVEMENT_PERCENT
                )
    all_passed = all(mandatory_gates.values())

    timing_summary: dict[str, Any] = {}
    for arm in ARMS:
        timing_summary[arm] = {}
        for shift in (-1, 1):
            selected_rows = [
                row
                for row in timing_rows
                if row["arm"] == arm and row["shift_clip_steps"] == shift
            ]
            timing_summary[arm][str(shift)] = {
                "nonwrap_rows": len(selected_rows),
                "silhouette_boundary_chamfer_px_mean": float(
                    np.mean([row["silhouette_boundary_chamfer_px"] for row in selected_rows])
                ),
                "rgb_robot_band_chamfer_px_mean": float(
                    np.mean([row["rgb_robot_band_chamfer_px"] for row in selected_rows])
                ),
            }

    latency = np.asarray(steady_render_latencies)
    analysis = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "corrected_renderer_attribution_analysis",
            "created_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "decision": "GO_FOR_WAN_SCREEN" if all_passed else "STOP_RENDERER_ATTRIBUTION",
            "interpretation": (
                "predicted correction clears every preregistered raw/shuffled silhouette, regional RGB-edge, and robot-flow attribution gate"
                if all_passed
                else "predicted correction fails at least one preregistered renderer-attribution condition; this study does not authorize Wan integration"
            ),
            "split": "train",
            "fit_clips": FIT_STOP,
            "score_clips": SCORE_CLIPS,
            "score_frames": SCORE_CLIPS * 8,
            "fit_score_episode_disjoint": True,
            "validation_accessed": False,
            "protected_test_accessed": False,
            "metrics_by_arm": {
                arm: {name: float(values.mean()) for name, values in metrics.items()}
                for arm, metrics in by_arm.items()
            },
            "effects": effects,
            "gates": {**mandatory_gates, "all_passed": all_passed},
            "timing_diagnostic": {
                "nonwrapping_clip_steps": [-1, 1],
                "one_step_is_not_subframe_timing": True,
                "by_arm": timing_summary,
            },
            "flow_contract": {
                "type": "rendered geom-local point transport on oracle-visible articulated support",
                "pixel_stride": FLOW_PIXEL_STRIDE,
                "minimum_oracle_motion_px": FLOW_MIN_ORACLE_MOTION_PX,
                "minimum_transition_pixels": 10,
                "zero_support_transition_policy": (
                    "exclude symmetrically for all arms; pool clip EPE by qualifying oracle pixels"
                ),
                "far_depth": far_depth,
                "moving_geom_ids": sorted(moving_geoms),
                "static_arm_bases_excluded": True,
                "object_or_full_scene_flow": False,
            },
            "camera": {
                "official_abc_commit": official_commit,
                "official_scene": file_record(scene_path),
                "robot_only_xml_sha256": hashlib.sha256(robot_xml.encode()).hexdigest(),
                "nominal_extrinsic": True,
                "recorded_fy_as_fovy": True,
                "principal_point_centered": True,
                "distortion_applied": False,
            },
            "latency": {
                "render_pose_mean_ms": float(latency.mean()),
                "render_pose_p50_ms": float(np.percentile(latency, 50)),
                "render_pose_p95_ms": float(np.percentile(latency, 95)),
                "render_pose_count_after_two_warmups": int(len(latency)),
                "warmups_excluded_per_clip": 2,
                "native_resolution_renderer_recreated_per_clip": True,
            },
            "artifacts": {
                "registration": file_record(output / "registration.json"),
                "preparation": file_record(output / "preparation.json"),
                "preparation_complete": file_record(output / "preparation_complete.json"),
                "source": file_record(Path(__file__).resolve()),
                "frame_transition_metrics": file_record(row_path),
                "clip_metrics": file_record(aggregate_path),
                "nonwrapping_timing_metrics": file_record(timing_path),
                "overlays": overlay_records,
            },
            "claim_boundary": (
                "Renderer attribution only. Measured future state defines scoring targets and is never inference input. "
                "No video model, video-quality metric, object/contact flow, or protected test is evaluated."
            ),
            "limitations": [
                "24 motion-stratified held-out train episodes and one fixed seed",
                "raw-command endpoints are not official controller or dynamics rollouts",
                "independently ceiling-resampled command/state streams prevent sub-frame timing claims",
                "nominal camera extrinsics, centered principal point, and no distortion correction",
                "measured state supplies only privileged scoring support and oracle rendered flow",
                "rendered robot-only flow excludes objects, contacts, and scene-object occlusion",
                "passing would authorize a controlled Wan screen, not a generation-quality claim",
            ],
        }
    )
    write_json(output / "analysis.json", analysis)
    complete = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "corrected_renderer_attribution_complete",
            "completed_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": analysis["decision"],
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
    documents = {
        name: read_json(output / name)
        for name in (
            "registration.json",
            "preparation.json",
            "preparation_complete.json",
            "analysis.json",
            "run_complete.json",
        )
    }
    for name, document in documents.items():
        validate_identity(document, name)
    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    analysis = documents["analysis.json"]
    complete = documents["run_complete.json"]
    if preparation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise AttributionError("preparation-registration binding differs")
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise AttributionError("analysis-registration binding differs")
    if analysis["preparation_identity_sha256"] != preparation["identity_sha256"]:
        raise AttributionError("analysis-preparation binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise AttributionError("completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise AttributionError("completion decision differs")
    if sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise AttributionError("current source differs from registered source")

    selected = registration["selection"]["selected"]
    if len(selected) != SCORE_CLIPS:
        raise AttributionError("registration selected count differs")
    selected_indices = [int(item["manifest_index"]) for item in selected]
    if any(index < SCORE_START or index >= SCORE_STOP for index in selected_indices):
        raise AttributionError("selected index is outside frozen score pool")
    if len(set(selected_indices)) != SCORE_CLIPS:
        raise AttributionError("selected indices are not unique")
    for item in selected:
        if item["donor_manifest_index"] == item["manifest_index"]:
            raise AttributionError("shuffle donor equals native clip")
        if item["donor_episode_dir"] == item["episode_dir"]:
            raise AttributionError("shuffle donor equals native episode")
        if item["donor_manifest_index"] not in selected_indices:
            raise AttributionError("shuffle donor is outside selected score rows")

    for section in (preparation["artifacts"], analysis["artifacts"], complete["artifacts"]):
        for name, record in section.items():
            if name == "overlays":
                for item in record:
                    path = Path(item["file"]["path"])
                    if not path.is_file():
                        path = output / "overlays" / path.name
                    if sha256_file(path) != item["file"]["sha256"]:
                        raise AttributionError(f"overlay hash mismatch: {path}")
                continue
            path = Path(record["path"])
            if not path.is_file():
                path = output / path.name
            if sha256_file(path) != record["sha256"]:
                raise AttributionError(f"artifact hash mismatch: {name}")

    bundles = load_bundles(output, preparation)
    frame_rows = read_jsonl(output / "frame_transition_metrics.jsonl")
    clip_rows = read_jsonl(output / "clip_metrics.jsonl")
    timing_rows = read_jsonl(output / "nonwrapping_timing_metrics.jsonl")
    provenance_rows = read_jsonl(output / "input_provenance.jsonl")
    if len(frame_rows) != SCORE_CLIPS * 8 * len(ARMS):
        raise AttributionError(f"frame row count differs: {len(frame_rows)}")
    if len(clip_rows) != SCORE_CLIPS * len(ARMS):
        raise AttributionError(f"clip row count differs: {len(clip_rows)}")
    expected_timing = SCORE_CLIPS * 2 * 7 * len(ARMS)
    if len(timing_rows) != expected_timing:
        raise AttributionError(f"timing row count differs: {len(timing_rows)}")
    if len(provenance_rows) != FIT_STOP + SCORE_CLIPS:
        raise AttributionError(f"provenance row count differs: {len(provenance_rows)}")
    if len(bundles) != SCORE_CLIPS:
        raise AttributionError("bundle count differs")
    for row in frame_rows:
        metrics = row["metrics"]
        if metrics["oracle_source_reprojection_rmse_px"] > 1e-5:
            raise AttributionError("flow source reprojection audit failed")
    for row in clip_rows:
        if row["metrics"]["oracle_moving_flow_sample_count"] < 10:
            raise AttributionError("clip flow support below frozen minimum")
    false_flags = sum(
        require_false_flags(document, name) for name, document in documents.items()
    )
    false_flags += sum(
        require_false_flags(rows, name)
        for name, rows in (
            ("frame_rows", frame_rows),
            ("clip_rows", clip_rows),
            ("timing_rows", timing_rows),
            ("provenance_rows", provenance_rows),
        )
    )
    return {
        "status": "audit_passed",
        "decision": analysis["decision"],
        "registration_identity_sha256": registration["identity_sha256"],
        "preparation_identity_sha256": preparation["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "completion_identity_sha256": complete["identity_sha256"],
        "bundles": len(bundles),
        "frame_rows": len(frame_rows),
        "clip_rows": len(clip_rows),
        "timing_rows": len(timing_rows),
        "provenance_rows": len(provenance_rows),
        "explicit_false_flags": false_flags,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--train-cache", type=Path, required=True)
    register_parser.add_argument("--train-manifest", type=Path, required=True)
    register_parser.add_argument("--preprocessed-root", type=Path, required=True)
    register_parser.add_argument("--raw-root", type=Path, required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--official-abc-root", type=Path, required=True)
    evaluate_parser.add_argument("--mujoco-gl", default="egl")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        result = register(args)
        event = "registered"
    elif args.command == "prepare":
        result = prepare(args)
        event = "prepared"
    elif args.command == "evaluate":
        result = evaluate(args)
        event = "evaluated"
    elif args.command == "audit":
        result = audit(args.output)
        event = "audited"
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"event": event, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
