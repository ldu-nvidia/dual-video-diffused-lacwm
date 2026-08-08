#!/usr/bin/env python3
"""Leakage-controlled Stage-0 raw-command tracking-residual probe for ABC.

The predictor sees measured history through video frame 4 and requested action
chunks 4..11.  Its target is the measured state at frames 5..12 minus the last
raw position command in each preceding five-sample action chunk.  Future
measured state is target/scoring-only.  No RGB is opened.

This is intentionally labelled a *raw-command proxy*.  It is not a replay of
the official ABC controller or MuJoCo dynamics and must not be described as a
simulated nominal-to-realized residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


SCHEMA_VERSION = 1
SEED = 1234
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_SAMPLES = 10_000
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
TRAIN_COUNT = 512
VALIDATION_COUNT = 64
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
ACTION_DIM = 14
PADDED_ACTION_DIM = 23
HISTORY_FRAME_COUNT = 5
FUTURE_CHUNKS = tuple(range(4, 12))
HISTORY_CHUNKS = tuple(range(0, 4))
PCA_COMPONENTS = 64
GATE_MIN_PERCENT = 10.0

# LACWM/cache: [left arm 6, right arm 6, left grip, right grip].
# Official ABC policy/simulator: [left arm 6, left grip, right arm 6, right grip].
CACHE14_TO_OFFICIAL14 = (0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13)


class ProbeError(RuntimeError):
    """Raised when the frozen data or causality contract differs."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_record(path: Path, *, digest: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
    }
    if digest:
        result["sha256"] = sha256_file(path)
    return result


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cache14_to_official14(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected trailing dimension {ACTION_DIM}, got {array.shape}")
    return array[..., CACHE14_TO_OFFICIAL14]


def audit_preprocessing_source(path: Path) -> dict[str, Any]:
    source = path.read_text()
    expression = "np.searchsorted(ts, frame_ts)"
    if expression not in source:
        raise ProbeError("preprocessing source no longer has the registered ceiling resampler")
    return {
        **file_record(path),
        "claimed_comment_mentions_nearest": "nearest" in source.lower(),
        "executed_expression": expression,
        "actual_behavior": "next/ceiling sample, clipped at both ends; not nearest-neighbor",
        "raw_controller_sample_timestamps_preserved_in_states_npz": False,
    }


def cache_record(cache_dir: Path, manifest: Path) -> dict[str, Any]:
    metadata_path = cache_dir / "metadata.json"
    metadata = read_json(metadata_path)
    actions_path = cache_dir / metadata["actions_file"]
    return {
        "cache_dir": str(cache_dir.resolve()),
        "metadata": file_record(metadata_path),
        "manifest": file_record(manifest),
        "split": metadata["split"],
        "clip_count": int(metadata["clip_count"]),
        "actions": {
            "path": str(actions_path.resolve()),
            "bytes": actions_path.stat().st_size,
            "registered_sha256": metadata["actions_sha256"],
            "actual_sha256": sha256_file(actions_path),
            "shape": metadata["actions_shape"],
            "dtype": metadata["actions_dtype"],
        },
        "rgb_opened": False,
        "cached_vjepa_target_opened": False,
    }


def build_registration(args: argparse.Namespace) -> dict[str, Any]:
    train = cache_record(args.train_cache, args.train_manifest)
    validation = cache_record(args.val_cache, args.val_manifest)
    for split, item in (("train", train), ("validation", validation)):
        if item["actions"]["registered_sha256"] != item["actions"]["actual_sha256"]:
            raise ProbeError(f"{split} cached actions hash differs from metadata")
    script = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nominal_tracking_residual_stage0_registration",
        "status": "registered_before_state_target_extraction_or_validation_read",
        "created_at_utc": now(),
        "source": {
            **file_record(script),
            "git_commit": git_commit(script.parents[1]),
        },
        "inputs": {"train": train, "validation": validation},
        "preprocessing_source_audit": audit_preprocessing_source(args.preprocess_source),
        "protocol": {
            "question": (
                "is a compact future raw-command tracking residual predictable from observed "
                "state/history and planned actions strongly enough to justify a stochastic auxiliary state?"
            ),
            "claim_name": "raw-command tracking-residual proxy",
            "forbidden_claim": "controller-simulated or physics-realized nominal residual",
            "seed": SEED,
            "train_clips": TRAIN_COUNT,
            "validation_clips": VALIDATION_COUNT,
            "protected_test_access_allowed": False,
            "frame_action_order": {
                "video_frames": list(range(SAMPLE_SIZE)),
                "observed_frames": list(range(HISTORY_FRAME_COUNT)),
                "future_realized_state_frames_target_only": list(range(5, 13)),
                "action_chunks": [0, 12],
                "raw_samples_per_chunk": CHUNK_SIZE,
                "planned_future_chunks": [4, 12],
                "target_pairing": (
                    "realized state at boundary t+1 minus final raw position command "
                    "in preceding chunk t, for t=4..11"
                ),
                "manifest_requirement": "frame_indices=start+5*t exactly",
            },
            "layout_audit": {
                "cache14": "[left_joint1..6,right_joint1..6,left_gripper,right_gripper]",
                "official14": "[left_joint1..6,left_gripper,right_joint1..6,right_gripper]",
                "cache14_to_official14": list(CACHE14_TO_OFFICIAL14),
                "model_uses_layout": "cache14 only; official permutation is audited but no simulator is run",
            },
            "predictor_inputs": {
                "observed_state_boundaries": "frames 0..4, 5x14",
                "observed_raw_commands": "chunks 0..3, 4x5x14",
                "observed_tracking_residuals": "transitions 0..3, 4x14",
                "planned_actions_candidate_only": "chunks 4..11, 8x5x14",
                "future_measured_state": False,
                "rgb": False,
            },
            "target_only": {
                "shape": [8, 14],
                "definition": "q_measured[t+1] - a_raw[t,last], t=4..11",
                "joint_units": "radians as stored",
                "gripper_units": "normalized aperture as stored",
            },
            "nominal_proxy": {
                "trajectory": "raw absolute position commands already available to the generator",
                "chunk_reduction": "last command before next video-frame boundary (zero-order-hold endpoint)",
                "controller_or_dynamics_replay": False,
            },
            "models": {
                "history_only": "train-standardized history PCA64 -> multi-output ridge",
                "history_plus_action": "history PCA64 + planned-action PCA64 -> multi-output ridge",
                "alpha_grid": list(ALPHAS),
                "alpha_selection": "five-fold train-only KFold standardized-target MSE",
            },
            "controls": {
                "train_mean": "zero in train-standardized target space",
                "zero_residual": "raw command endpoint is treated as realized state",
                "hold_current": "q(frame4) is treated as every future realized state",
                "episode_shuffled": (
                    "different-episode planned-action feature supplied only to the residual predictor; "
                    "native nominal endpoint and native target stay fixed"
                ),
                "mean_action": "train-mean action feature supplied to the same residual predictor",
            },
            "primary_metric": "mean per-clip train-dimension-standardized tracking-residual MSE",
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "paired_clip_resampling": True,
                "effect": "ratio of resampled means, not mean clip ratio",
            },
            "strict_go_gate": {
                "minimum_point_improvement_percent": GATE_MIN_PERCENT,
                "paired_bootstrap_95_low_strictly_positive": True,
                "required_references": [
                    "history_only",
                    "episode_shuffled_same_model",
                    "zero_residual_raw_command",
                    "hold_current_state",
                ],
                "all_required": True,
            },
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "protected_test_accessed": False,
    }
    return seal(payload)


def validate_manifest_rows(
    rows: list[dict[str, Any]], expected_count: int, expected_split: str
) -> None:
    if len(rows) != expected_count:
        raise ProbeError(f"{expected_split} row count {len(rows)} != {expected_count}")
    episodes: list[str] = []
    clip_ids: list[str] = []
    for index, row in enumerate(rows):
        if row.get("split") != expected_split:
            raise ProbeError(f"{expected_split} row {index} has split={row.get('split')!r}")
        if (row.get("sample_size"), row.get("chunk_size"), row.get("action_span")) != (
            SAMPLE_SIZE,
            CHUNK_SIZE,
            SAMPLE_SIZE * CHUNK_SIZE,
        ):
            raise ProbeError(f"{expected_split} row {index} geometry differs")
        start = int(row["start"])
        expected_indices = list(range(start, start + SAMPLE_SIZE * CHUNK_SIZE, CHUNK_SIZE))
        if row.get("frame_indices") != expected_indices:
            raise ProbeError(f"{expected_split} row {index} frame ordering differs")
        episodes.append(str(row["episode_dir"]))
        clip_ids.append(str(row["clip_id"]))
    if len(set(episodes)) != expected_count or len(set(clip_ids)) != expected_count:
        raise ProbeError(f"{expected_split} clips are not unique episodes and IDs")


def construct_clip_arrays(
    state_boundaries: np.ndarray,
    action_chunks: np.ndarray,
) -> dict[str, np.ndarray]:
    states = np.asarray(state_boundaries, dtype=np.float32)
    actions = np.asarray(action_chunks, dtype=np.float32)
    if states.shape != (SAMPLE_SIZE, ACTION_DIM):
        raise ValueError(f"state boundaries shape differs: {states.shape}")
    if actions.shape != (SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"action chunks shape differs: {actions.shape}")
    observed_residual = states[1:HISTORY_FRAME_COUNT] - actions[list(HISTORY_CHUNKS), -1]
    history = np.concatenate(
        (
            states[:HISTORY_FRAME_COUNT].reshape(-1),
            actions[list(HISTORY_CHUNKS)].reshape(-1),
            observed_residual.reshape(-1),
        )
    ).astype(np.float32)
    future_actions = actions[list(FUTURE_CHUNKS)].astype(np.float32)
    nominal_endpoint = future_actions[:, -1].astype(np.float32)
    realized_state = states[5:13].astype(np.float32)
    target_residual = (realized_state - nominal_endpoint).astype(np.float32)
    hold_current_residual = (states[4][None, :] - nominal_endpoint).astype(np.float32)
    return {
        "history": history,
        "future_actions": future_actions,
        "nominal_endpoint": nominal_endpoint,
        "realized_state": realized_state,
        "target_residual": target_residual,
        "hold_current_residual": hold_current_residual,
    }


def extract_split(
    cache_dir: Path,
    rows: list[dict[str, Any]],
    split: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    metadata = read_json(cache_dir / "metadata.json")
    cached = np.load(cache_dir / metadata["actions_file"], mmap_mode="r", allow_pickle=False)
    expected_shape = (len(rows), SAMPLE_SIZE, CHUNK_SIZE, PADDED_ACTION_DIM)
    if tuple(cached.shape) != expected_shape or cached.dtype != np.float32:
        raise ProbeError(f"{split} cached action geometry differs: {cached.shape} {cached.dtype}")
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
    cache_max_abs_error = 0.0
    padded_max_abs = 0.0
    frame_periods: list[float] = []
    for clip_index, row in enumerate(rows):
        state_path = Path(row["episode_dir"]) / "states.npz"
        with np.load(state_path, allow_pickle=False) as state:
            required = {"joint_states", "joint_actions", "gripper_states", "gripper_actions", "frame_ts"}
            if not required.issubset(state.files):
                raise ProbeError(f"{state_path} is missing arrays: {sorted(required - set(state.files))}")
            indices = np.asarray(row["frame_indices"], dtype=np.int64)
            if int(indices.max()) >= len(state["joint_states"]):
                raise ProbeError(f"{split} row {clip_index} exceeds states length")
            boundaries = np.concatenate(
                (state["joint_states"][indices], state["gripper_states"][indices]), axis=1
            ).astype(np.float32)
            start = int(row["start"])
            stop = start + SAMPLE_SIZE * CHUNK_SIZE
            raw_actions = np.concatenate(
                (state["joint_actions"][start:stop], state["gripper_actions"][start:stop]), axis=1
            ).astype(np.float32)
            if raw_actions.shape != (SAMPLE_SIZE * CHUNK_SIZE, ACTION_DIM):
                raise ProbeError(f"{split} row {clip_index} raw action span differs")
            raw_actions = raw_actions.reshape(SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM)
            timestamp = np.asarray(state["frame_ts"][indices], dtype=np.int64)
        cache_values = np.asarray(cached[clip_index, ..., :ACTION_DIM], dtype=np.float32)
        cache_max_abs_error = max(
            cache_max_abs_error,
            float(np.max(np.abs(cache_values - raw_actions))),
        )
        padded_max_abs = max(
            padded_max_abs,
            float(np.max(np.abs(np.asarray(cached[clip_index, ..., ACTION_DIM:])))),
        )
        derived = construct_clip_arrays(boundaries, cache_values)
        for name, value in derived.items():
            arrays[name].append(value)
        frame_periods.extend((np.diff(timestamp).astype(np.float64) / 1e9).tolist())
        provenance.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "clip_index": clip_index,
                "clip_id": row["clip_id"],
                "episode_dir": row["episode_dir"],
                "states_npz": file_record(state_path),
                "frame_indices": row["frame_indices"],
                "start": row["start"],
                "protected_test_accessed": False,
            }
        )
    if cache_max_abs_error != 0.0 or padded_max_abs != 0.0:
        raise ProbeError(
            f"{split} cache/raw contract differs: active={cache_max_abs_error} padded={padded_max_abs}"
        )
    stacked = {name: np.stack(values) for name, values in arrays.items()}
    audit = {
        "clips": len(rows),
        "cached_active14_vs_states_raw_action_max_abs": cache_max_abs_error,
        "cached_padded_coordinates_max_abs": padded_max_abs,
        "official_permutation_roundtrip_exact": False,
        "frame_boundary_period_seconds": {
            "mean": float(np.mean(frame_periods)),
            "median": float(np.median(frame_periods)),
            "min": float(np.min(frame_periods)),
            "max": float(np.max(frame_periods)),
        },
    }
    # The round-trip expression above intentionally exercises the public mapping;
    # verify with the direct inverse as an unambiguous guard.
    basis = np.arange(ACTION_DIM)
    if not np.array_equal(cache14_to_official14(basis)[np.argsort(CACHE14_TO_OFFICIAL14)], basis):
        raise ProbeError("cache-to-official permutation is not invertible")
    audit["official_permutation_roundtrip_exact"] = True
    return stacked, provenance, audit


def standardizer(values: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = values.reshape(len(values), -1)
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    active = raw_std > epsilon
    if int(active.sum()) < PCA_COMPONENTS:
        raise ProbeError(f"only {int(active.sum())} active dimensions; PCA{PCA_COMPONENTS} is impossible")
    return mean, np.where(active, raw_std, 1.0).astype(np.float32), active


def fit_pca(values: np.ndarray) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std, active = standardizer(values)
    flat = values.reshape(len(values), -1)
    normalized = ((flat - mean) / std)[:, active]
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    transformed = pca.fit_transform(normalized).astype(np.float32)
    return pca, transformed, mean, std, active


def transform_pca(
    pca: PCA,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return pca.transform(((flat - mean) / std)[:, active]).astype(np.float32)


def input_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.where(std > 1e-8, std, 1.0).astype(np.float32)


def choose_alpha(inputs: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        fold_scores = []
        for fit_index, score_index in folds.split(inputs):
            model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            model.fit(inputs[fit_index], targets[fit_index])
            prediction = model.predict(inputs[score_index])
            fold_scores.append(float(np.mean((prediction - targets[score_index]) ** 2)))
        scores[str(alpha)] = float(np.mean(fold_scores))
    selected = min(ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    return float(selected), scores


def paired_effect(reference: np.ndarray, candidate: np.ndarray, seed_offset: int) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired errors must be same-shape vectors")
    point = 100.0 * (1.0 - float(candidate.mean()) / float(reference.mean()))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(reference), size=(BOOTSTRAP_SAMPLES, len(reference)))
    ref_mean = reference[indices].mean(axis=1)
    candidate_mean = candidate[indices].mean(axis=1)
    samples = 100.0 * (1.0 - candidate_mean / ref_mean)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": point,
        "paired_bootstrap_95_percent": [float(low), float(high)],
        "favorable_clip_fraction": float(np.mean(candidate < reference)),
        "paired_clips": int(len(reference)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
    }


def cosine_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def run_probe(args: argparse.Namespace, registration: dict[str, Any]) -> dict[str, Any]:
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    validate_manifest_rows(train_rows, TRAIN_COUNT, "train")
    validate_manifest_rows(val_rows, VALIDATION_COUNT, "val")
    train_episodes = {row["episode_dir"] for row in train_rows}
    val_episodes = {row["episode_dir"] for row in val_rows}
    if train_episodes & val_episodes:
        raise ProbeError("train/validation episodes overlap")

    train, train_provenance, train_audit = extract_split(args.train_cache, train_rows, "train")
    val, val_provenance, val_audit = extract_split(args.val_cache, val_rows, "val")

    history_pca, train_history, history_mean, history_std, history_active = fit_pca(train["history"])
    action_pca, train_action, action_mean, action_std, action_active = fit_pca(train["future_actions"])
    val_history = transform_pca(history_pca, val["history"], history_mean, history_std, history_active)
    val_action = transform_pca(action_pca, val["future_actions"], action_mean, action_std, action_active)

    target_flat_train = train["target_residual"].reshape(TRAIN_COUNT, -1)
    target_flat_val = val["target_residual"].reshape(VALIDATION_COUNT, -1)
    target_mean = target_flat_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std_raw = target_flat_train.std(axis=0, dtype=np.float64).astype(np.float32)
    target_std = np.where(target_std_raw > 1e-8, target_std_raw, 1.0).astype(np.float32)
    train_target = ((target_flat_train - target_mean) / target_std).astype(np.float32)
    val_target = ((target_flat_val - target_mean) / target_std).astype(np.float32)

    x_history_mean, x_history_std = input_standardizer(train_history)
    x_action_raw = np.concatenate((train_history, train_action), axis=1)
    x_action_mean, x_action_std = input_standardizer(x_action_raw)
    x_history = (train_history - x_history_mean) / x_history_std
    x_action = (x_action_raw - x_action_mean) / x_action_std
    val_x_history = (val_history - x_history_mean) / x_history_std

    def candidate_inputs(action_features: np.ndarray) -> np.ndarray:
        return (
            np.concatenate((val_history, action_features), axis=1) - x_action_mean
        ) / x_action_std

    alpha_history, cv_history = choose_alpha(x_history, train_target)
    alpha_action, cv_action = choose_alpha(x_action, train_target)
    history_model = Ridge(alpha=alpha_history, fit_intercept=True, solver="cholesky").fit(
        x_history, train_target
    )
    action_model = Ridge(alpha=alpha_action, fit_intercept=True, solver="cholesky").fit(
        x_action, train_target
    )
    donor = np.roll(np.arange(VALIDATION_COUNT), -1)
    if any(val_rows[index]["episode_dir"] == val_rows[int(source)]["episode_dir"] for index, source in enumerate(donor)):
        raise ProbeError("episode-shuffled action donor is not disjoint")
    mean_action_raw = np.broadcast_to(action_mean, (VALIDATION_COUNT, action_mean.size)).copy()
    mean_action = transform_pca(
        action_pca,
        mean_action_raw.reshape(VALIDATION_COUNT, *val["future_actions"].shape[1:]),
        action_mean,
        action_std,
        action_active,
    )

    predictions: dict[str, np.ndarray] = {
        "history_only": history_model.predict(val_x_history).astype(np.float32),
        "history_plus_aligned_action": action_model.predict(candidate_inputs(val_action)).astype(np.float32),
        "history_plus_episode_shuffled_action": action_model.predict(candidate_inputs(val_action[donor])).astype(np.float32),
        "history_plus_mean_action": action_model.predict(candidate_inputs(mean_action)).astype(np.float32),
        "train_mean_residual": np.zeros_like(val_target),
        "zero_residual_raw_command": ((np.zeros_like(target_flat_val) - target_mean) / target_std).astype(np.float32),
        "hold_current_state": (
            (val["hold_current_residual"].reshape(VALIDATION_COUNT, -1) - target_mean) / target_std
        ).astype(np.float32),
    }
    clip_errors = {
        name: np.mean((prediction - val_target) ** 2, axis=1)
        for name, prediction in predictions.items()
    }
    aligned_name = "history_plus_aligned_action"
    contrasts = {
        "aligned_vs_history_only": ("history_only", 1),
        "aligned_vs_episode_shuffled_same_model": ("history_plus_episode_shuffled_action", 2),
        "aligned_vs_zero_residual_raw_command": ("zero_residual_raw_command", 3),
        "aligned_vs_hold_current_state": ("hold_current_state", 4),
        "aligned_vs_train_mean_residual": ("train_mean_residual", 5),
        "aligned_vs_mean_action_same_model": ("history_plus_mean_action", 6),
    }
    effects = {
        label: paired_effect(clip_errors[reference], clip_errors[aligned_name], offset)
        for label, (reference, offset) in contrasts.items()
    }
    required = (
        "aligned_vs_history_only",
        "aligned_vs_episode_shuffled_same_model",
        "aligned_vs_zero_residual_raw_command",
        "aligned_vs_hold_current_state",
    )
    gates = {
        label: (
            effects[label]["relative_improvement_percent"] >= GATE_MIN_PERCENT
            and effects[label]["paired_bootstrap_95_percent"][0] > 0.0
        )
        for label in required
    }
    gates["all_passed"] = all(gates.values())

    provenance_path = args.output / "input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for row in (*train_provenance, *val_provenance):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    per_clip_path = args.output / "per_clip_metrics.jsonl"
    aligned_raw = predictions[aligned_name] * target_std + target_mean
    target_raw = target_flat_val
    with per_clip_path.open("w") as handle:
        for index, row in enumerate(val_rows):
            payload = {
                "schema_version": SCHEMA_VERSION,
                "clip_index": index,
                "clip_id": row["clip_id"],
                "episode_dir": row["episode_dir"],
                "donor_clip_index": int(donor[index]),
                "donor_episode_dir": val_rows[int(donor[index])]["episode_dir"],
                "standardized_mse": {
                    name: float(error[index]) for name, error in clip_errors.items()
                },
                "aligned_raw_residual_mse": float(np.mean((aligned_raw[index] - target_raw[index]) ** 2)),
                "aligned_raw_residual_cosine": float(
                    cosine_rows(aligned_raw[index : index + 1], target_raw[index : index + 1])[0]
                ),
                "protected_test_accessed": False,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    feature_path = args.output / "derived_features_targets.npz"
    np.savez_compressed(
        feature_path,
        train_history=train["history"],
        train_future_actions=train["future_actions"],
        train_nominal_endpoint=train["nominal_endpoint"],
        train_realized_state=train["realized_state"],
        train_target_residual=train["target_residual"],
        train_hold_current_residual=train["hold_current_residual"],
        val_history=val["history"],
        val_future_actions=val["future_actions"],
        val_nominal_endpoint=val["nominal_endpoint"],
        val_realized_state=val["realized_state"],
        val_target_residual=val["target_residual"],
        val_hold_current_residual=val["hold_current_residual"],
        val_episode_shuffled_donor=donor,
    )
    model_path = args.output / "model_state_and_predictions.npz"
    np.savez_compressed(
        model_path,
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
        history_model_coef=history_model.coef_,
        history_model_intercept=history_model.intercept_,
        action_model_coef=action_model.coef_,
        action_model_intercept=action_model.intercept_,
        validation_target_standardized=val_target,
        **{f"prediction_{name}": prediction for name, prediction in predictions.items()},
    )

    aligned_raw_reshaped = aligned_raw.reshape(VALIDATION_COUNT, 8, ACTION_DIM)
    target_raw_reshaped = target_raw.reshape(VALIDATION_COUNT, 8, ACTION_DIM)
    history_raw_reshaped = (
        predictions["history_only"] * target_std + target_mean
    ).reshape(VALIDATION_COUNT, 8, ACTION_DIM)
    future_state_motion = val["target_residual"] - val["hold_current_residual"]
    target_rms = float(np.sqrt(np.mean(target_raw_reshaped**2)))
    future_state_motion_rms = float(np.sqrt(np.mean(future_state_motion**2)))
    zero_raw = np.zeros_like(target_raw_reshaped)
    by_horizon = {}
    for horizon in range(8):
        aligned_error = np.mean(
            ((aligned_raw_reshaped[:, horizon] - target_raw_reshaped[:, horizon]) / target_std.reshape(8, ACTION_DIM)[horizon]) ** 2,
            axis=1,
        )
        zero_error = np.mean(
            ((zero_raw[:, horizon] - target_raw_reshaped[:, horizon]) / target_std.reshape(8, ACTION_DIM)[horizon]) ** 2,
            axis=1,
        )
        by_horizon[str(horizon + 1)] = paired_effect(zero_error, aligned_error, 100 + horizon)

    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nominal_tracking_residual_stage0_analysis",
        "created_at_utc": now(),
        "registration_identity_sha256": registration["identity_sha256"],
        "decision": "GO" if gates["all_passed"] else "NO_GO",
        "interpretation": (
            "the raw-command residual clears every strict incremental, action-specific, nominal, and persistence gate"
            if gates["all_passed"]
            else "the raw-command residual does not clear every strict gate; do not integrate it into the video generator"
        ),
        "gates": gates,
        "effects": effects,
        "validation_mean_standardized_mse": {
            name: float(error.mean()) for name, error in clip_errors.items()
        },
        "validation_aggregate_standardized_r2": {
            name: float(1.0 - np.sum((prediction - val_target) ** 2) / np.sum(val_target**2))
            for name, prediction in predictions.items()
        },
        "raw_metrics": {
            "target_residual_rms": target_rms,
            "target_residual_mean_absolute": float(np.mean(np.abs(target_raw))),
            "target_joint_residual_rms_radians": float(
                np.sqrt(np.mean(target_raw_reshaped[..., :12] ** 2))
            ),
            "target_gripper_residual_rms_normalized_aperture": float(
                np.sqrt(np.mean(target_raw_reshaped[..., 12:] ** 2))
            ),
            "measured_future_state_motion_from_frame4_rms": future_state_motion_rms,
            "target_residual_over_future_state_motion_rms": (
                target_rms / future_state_motion_rms if future_state_motion_rms > 0.0 else None
            ),
            "aligned_prediction_rmse": float(np.sqrt(np.mean((aligned_raw - target_raw) ** 2))),
            "aligned_prediction_mean_absolute_error": float(np.mean(np.abs(aligned_raw - target_raw))),
            "history_only_prediction_rmse": float(
                np.sqrt(np.mean((history_raw_reshaped - target_raw_reshaped) ** 2))
            ),
            "zero_residual_rmse": float(np.sqrt(np.mean(target_raw**2))),
            "hold_current_state_rmse": future_state_motion_rms,
            "aligned_residual_cosine_mean": float(np.mean(cosine_rows(aligned_raw, target_raw))),
            "joint_rmse": float(
                np.sqrt(np.mean((aligned_raw_reshaped[..., :12] - target_raw_reshaped[..., :12]) ** 2))
            ),
            "gripper_rmse": float(
                np.sqrt(np.mean((aligned_raw_reshaped[..., 12:] - target_raw_reshaped[..., 12:]) ** 2))
            ),
        },
        "aligned_vs_zero_by_horizon": by_horizon,
        "models": {
            "history_only": {
                "selected_alpha": alpha_history,
                "train_cv_mse_by_alpha": cv_history,
                "input_dim": int(x_history.shape[1]),
            },
            "history_plus_action": {
                "selected_alpha": alpha_action,
                "train_cv_mse_by_alpha": cv_action,
                "input_dim": int(x_action.shape[1]),
            },
            "target_dim": int(train_target.shape[1]),
            "history_raw_dim": int(train["history"].reshape(TRAIN_COUNT, -1).shape[1]),
            "future_action_raw_dim": int(train["future_actions"].reshape(TRAIN_COUNT, -1).shape[1]),
            "history_pca_explained_variance_ratio_sum": float(history_pca.explained_variance_ratio_.sum()),
            "action_pca_explained_variance_ratio_sum": float(action_pca.explained_variance_ratio_.sum()),
            "active_history_dimensions": int(history_active.sum()),
            "active_action_dimensions": int(action_active.sum()),
            "active_target_dimensions": int(np.sum(target_std_raw > 1e-8)),
        },
        "data_contract": {
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(val_episodes),
            "train_validation_episode_overlap": 0,
            "state_targets_opened_after_registration": True,
            "future_state_used_as_predictor_input": False,
            "future_state_used_for_target_and_scoring_only": True,
            "rgb_opened": False,
            "cached_vjepa_target_opened": False,
            "protected_test_accessed": False,
            "train_ordering_audit": train_audit,
            "validation_ordering_audit": val_audit,
            "cache14_to_official14": list(CACHE14_TO_OFFICIAL14),
            "preprocessing_caveat": (
                "states.npz uses np.searchsorted ceiling assignment independently for state/action streams; "
                "raw controller timestamps are not preserved, so sub-frame command/state timing cannot be recovered"
            ),
        },
        "artifacts": {
            "input_provenance": file_record(provenance_path),
            "per_clip_metrics": file_record(per_clip_path),
            "derived_features_targets": file_record(feature_path),
            "model_state_and_predictions": file_record(model_path),
        },
        "claim_boundary": (
            "This measures a raw absolute-command endpoint proxy. It does not replay controller dynamics, "
            "does not establish a simulator-realized residual, and does not measure video quality."
        ),
        "limitations": [
            "one fixed seed and a repeatedly reused exploratory validation64 split",
            "absolute command and measured state streams were ceiling-resampled independently to video time",
            "the last command in a five-sample chunk is a zero-order-hold endpoint proxy, not a controller rollout",
            "a residual predictor can learn algebraic command cancellation; the mandatory hold-current gate detects the simplest form",
            "episode shuffling diagnoses sample-specific action use but is not a randomized physical intervention",
            "no RGB, FVD, perceptual metric, generator training, latency, or long-horizon rollout is evaluated",
        ],
        "protected_test_accessed": False,
    }
    return seal(analysis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument(
        "--preprocess-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "robot_wm/datasets/abc/preprocessing/abc_preprocess.py",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(SEED)

    registration = build_registration(args)
    write_json(args.output / "registration.json", registration)
    print(
        json.dumps(
            {"event": "registered", "identity_sha256": registration["identity_sha256"]}
        ),
        flush=True,
    )
    analysis = run_probe(args, registration)
    write_json(args.output / "analysis.json", analysis)
    complete = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "nominal_tracking_residual_stage0_complete",
            "completed_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "status": "completed",
            "decision": analysis["decision"],
            "artifacts": {
                "source": file_record(Path(__file__).resolve()),
                "registration": file_record(args.output / "registration.json"),
                "analysis": file_record(args.output / "analysis.json"),
                **analysis["artifacts"],
            },
            "protected_test_accessed": False,
        }
    )
    write_json(args.output / "run_complete.json", complete)
    print(
        json.dumps(
            {
                "event": "completed",
                "decision": analysis["decision"],
                "analysis_identity_sha256": analysis["identity_sha256"],
                "complete_identity_sha256": complete["identity_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
