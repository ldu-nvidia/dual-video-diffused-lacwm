#!/usr/bin/env python3
"""Read-only causal predictor latency replay bound to a sealed renderer study."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.corrected_renderer_attribution import (
    ACTION_DIM,
    CHUNK_SIZE,
    SAMPLE_SIZE,
    canonical_identity,
    read_json,
    read_jsonl,
    seal,
    sha256_file,
    write_json,
)


def causal_predictor_features(
    state_boundaries: np.ndarray,
    action_chunks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(state_boundaries, dtype=np.float32)
    actions = np.asarray(action_chunks, dtype=np.float32)
    if states.shape != (SAMPLE_SIZE, ACTION_DIM):
        raise ValueError(f"state geometry differs: {states.shape}")
    if actions.shape != (SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"action geometry differs: {actions.shape}")
    observed_residual = states[1:5] - actions[0:4, -1]
    history = np.concatenate(
        (states[:5].reshape(-1), actions[:4].reshape(-1), observed_residual.reshape(-1))
    ).astype(np.float32)
    return history, actions[4:12].astype(np.float32)


def pca_project(
    values: np.ndarray,
    model: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    normalized = (
        (flat - model[f"{prefix}_feature_mean"])
        / model[f"{prefix}_feature_std"]
    )[model[f"{prefix}_feature_active"].astype(bool)]
    centered = normalized - model[f"{prefix}_pca_mean"]
    return centered @ model[f"{prefix}_pca_components"].T


def predict_residual(
    state_boundaries: np.ndarray,
    action_chunks: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    history, future_action = causal_predictor_features(state_boundaries, action_chunks)
    history_feature = pca_project(history, model, "history")
    action_feature = pca_project(future_action, model, "action")
    raw_input = np.concatenate((history_feature, action_feature))
    standardized = (raw_input - model["input_mean"]) / model["input_std"]
    prediction_standardized = (
        standardized @ model["model_coef"].T + model["model_intercept"]
    )
    prediction = (
        prediction_standardized * model["target_std"] + model["target_mean"]
    )
    return np.asarray(prediction, dtype=np.float32).reshape(8, ACTION_DIM)


def load_score_inputs(
    registration: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    manifest = Path(registration["inputs"]["train_manifest"]["path"])
    rows = read_jsonl(manifest)
    action_path = Path(registration["inputs"]["train_actions"]["path"])
    actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
    boundaries_all: list[np.ndarray] = []
    actions_all: list[np.ndarray] = []
    for item in registration["selection"]["selected"]:
        index = int(item["manifest_index"])
        row = rows[index]
        if row.get("split") != "train" or not 384 <= index < 512:
            raise RuntimeError("latency replay received a non-score-train row")
        with np.load(Path(row["episode_dir"]) / "states.npz", allow_pickle=False) as state:
            frame_indices = np.asarray(row["frame_indices"], dtype=np.int64)
            # Only frames 0..4 are indexed. Future measured states remain unopened.
            observed_indices = frame_indices[:5]
            observed = np.concatenate(
                (
                    state["joint_states"][observed_indices],
                    state["gripper_states"][observed_indices],
                ),
                axis=1,
            ).astype(np.float32)
        # Fill target-only boundaries with NaNs: causal feature construction must
        # never inspect them, and a violation propagates into the output check.
        boundaries = np.full((SAMPLE_SIZE, ACTION_DIM), np.nan, dtype=np.float32)
        boundaries[:5] = observed
        boundaries_all.append(boundaries)
        actions_all.append(np.asarray(actions[index, ..., :ACTION_DIM], dtype=np.float32))
    return boundaries_all, actions_all


def percentile_summary(milliseconds: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "samples": int(len(values)),
        "mean_ms": float(values.mean()),
        "p10_ms": float(np.percentile(values, 10)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    registration = read_json(artifact / "registration.json")
    preparation = read_json(artifact / "preparation.json")
    if registration.get("identity_sha256") != canonical_identity(registration):
        raise RuntimeError("registration identity differs")
    if preparation.get("identity_sha256") != canonical_identity(preparation):
        raise RuntimeError("preparation identity differs")
    registered_source = args.registered_source.resolve()
    if sha256_file(registered_source) != registration["source"]["sha256"]:
        raise RuntimeError("registered execution source differs")
    model_path = artifact / "model_state_and_predictions.npz"
    if sha256_file(model_path) != preparation["artifacts"]["model_state_and_predictions"][
        "sha256"
    ]:
        raise RuntimeError("model artifact hash differs")
    with np.load(model_path, allow_pickle=False) as payload:
        model = {name: payload[name].copy() for name in payload.files}
    boundaries, actions = load_score_inputs(registration)
    expected = model["aligned_residual"]
    observed = np.stack(
        [predict_residual(state, action, model) for state, action in zip(boundaries, actions)]
    )
    max_abs = float(np.max(np.abs(observed - expected)))
    if max_abs > 1e-5:
        raise RuntimeError(f"manual predictor replay differs by {max_abs}")

    for _ in range(args.warmups):
        position = _ % len(boundaries)
        predict_residual(boundaries[position], actions[position], model)
    per_clip_ms = []
    for repeat in range(args.repeats):
        for position in range(len(boundaries)):
            started = time.perf_counter_ns()
            predict_residual(boundaries[position], actions[position], model)
            per_clip_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    def predict_batch() -> np.ndarray:
        return np.stack(
            [predict_residual(state, action, model) for state, action in zip(boundaries, actions)]
        )

    for _ in range(max(10, args.warmups // 10)):
        predict_batch()
    batch_ms = []
    for _ in range(args.repeats):
        started = time.perf_counter_ns()
        predict_batch()
        batch_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    batch_values = np.asarray(batch_ms, dtype=np.float64)
    result = seal(
        {
            "schema_version": 1,
            "kind": "corrected_renderer_predictor_latency_replay",
            "status": "completed",
            "decision_effect": "none; read-only operational sidecar after frozen quality decision",
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "registered_source_sha256": registration["source"]["sha256"],
            "model_state_and_predictions_sha256": sha256_file(model_path),
            "score_clips": len(boundaries),
            "causal_contract": {
                "measured_state_frames_used": [0, 1, 2, 3, 4],
                "future_measured_state_used": False,
                "future_state_slots_filled_with_nan": True,
                "action_chunks_used": list(range(12)),
                "rgb_used": False,
                "protected_test_accessed": False,
            },
            "replay_fidelity": {
                "expected_predictions": list(expected.shape),
                "maximum_absolute_error": max_abs,
                "tolerance": 1e-5,
            },
            "timing_contract": {
                "includes": (
                    "causal history/residual construction, history and action standardization, "
                    "two PCA64 projections, ridge inference, target destandardization"
                ),
                "excludes": "disk I/O, model fit, robot rendering, video model, and decoder",
                "warmups": args.warmups,
                "repeats": args.repeats,
                "clock": "time.perf_counter_ns",
                "batch_implementation": "Python stack of the same batch-1 causal predictor",
            },
            "batch1_per_clip": percentile_summary(np.asarray(per_clip_ms)),
            "batch24_total": percentile_summary(batch_values),
            "batch24_amortized_per_clip": percentile_summary(batch_values / len(boundaries)),
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
            },
            "protected_test_accessed": False,
        }
    )
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--registered-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
