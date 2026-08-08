#!/usr/bin/env python3
"""Preregistered dense top-view causal action-to-motion Stage-0 follow-up.

Future RGB is used only to construct and score dense optical-flow targets.
Predictors receive observed-history flow and, for the candidate, requested actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
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
FLOW_WORK_WH = (80, 45)
FLOW_TARGET_WH = (20, 12)
HISTORY_TRANSITIONS = tuple(range(0, 4))
FUTURE_TRANSITIONS = tuple(range(4, 12))
ACTION_CHUNKS = (4, 12)
HISTORY_PCA_COMPONENTS = 64
ACTION_PCA_COMPONENTS = 64
TARGET_PCA_COMPONENTS = 192


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(block):
            digest.update(data)
    return digest.hexdigest()


def canonical_identity(payload: dict) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seal(payload: dict) -> dict:
    result = dict(payload)
    result["identity_sha256"] = canonical_identity(result)
    return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_record(path: Path, digest: bool = True) -> dict:
    record = {"path": str(path.resolve()), "bytes": path.stat().st_size}
    if digest:
        record["sha256"] = sha256_file(path)
    return record


def cache_record(cache_dir: Path, manifest_path: Path) -> dict:
    metadata_path = cache_dir / "metadata.json"
    metadata = read_json(metadata_path)
    return {
        "cache_dir": str(cache_dir.resolve()),
        "metadata": file_record(metadata_path),
        "manifest": file_record(manifest_path),
        "split": metadata["split"],
        "clip_count": int(metadata["clip_count"]),
        "rgb": {
            "path": str((cache_dir / metadata["rgb_file"]).resolve()),
            "bytes": (cache_dir / metadata["rgb_file"]).stat().st_size,
            "registered_sha256": metadata["rgb_sha256"],
            "shape": metadata["rgb_shape"],
            "dtype": metadata["rgb_dtype"],
        },
        "actions": {
            "path": str((cache_dir / metadata["actions_file"]).resolve()),
            "bytes": (cache_dir / metadata["actions_file"]).stat().st_size,
            "registered_sha256": metadata["actions_sha256"],
            "shape": metadata["actions_shape"],
            "dtype": metadata["actions_dtype"],
        },
        "cached_vjepa_target_opened": False,
    }


def build_registration(args: argparse.Namespace) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dense_top_flow_stage0_registration",
        "status": "registered_before_dense_feature_or_target_extraction",
        "created_at_utc": now(),
        "source": file_record(Path(__file__).resolve()),
        "inputs": {
            "train": cache_record(args.train_cache, args.train_manifest),
            "validation": cache_record(args.val_cache, args.val_manifest),
        },
        "protocol": {
            "question": "does the prior top-view action gain survive a denser directional flow target?",
            "seed": SEED,
            "train_clips": 512,
            "validation_clips": 64,
            "protected_test_access_allowed": False,
            "predictor_inputs": {
                "history_rgb_frames": [0, 1, 2, 3, 4],
                "history_dense_flow": "top-view transitions 0->1 through 3->4, train-only PCA64",
                "planned_actions": "chunks [4:12], five substeps, train-active coordinates, train-standardized PCA64",
                "future_rgb_or_future_derived_feature": False,
            },
            "target_only": {
                "future_rgb_frames": list(range(4, 13)),
                "transitions": [f"{i}->{i+1}" for i in FUTURE_TRANSITIONS],
                "field": "raw directional Farneback (u,v), top view only",
                "work_resolution_wh": list(FLOW_WORK_WH),
                "target_resolution_wh": list(FLOW_TARGET_WH),
                "raw_target_shape": [8, 2, FLOW_TARGET_WH[1], FLOW_TARGET_WH[0]],
                "raw_target_dim": 8 * 2 * FLOW_TARGET_WH[1] * FLOW_TARGET_WH[0],
                "train_only_target_compression": f"centered PCA{TARGET_PCA_COMPONENTS}",
            },
            "models": {
                "history_only": f"history PCA{HISTORY_PCA_COMPONENTS} -> multi-output Ridge",
                "history_plus_action": f"history PCA{HISTORY_PCA_COMPONENTS} + action PCA{ACTION_PCA_COMPONENTS} -> multi-output Ridge",
                "alpha_grid": list(ALPHAS),
                "alpha_selection": "five-fold train-only KFold coefficient-standardized MSE",
            },
            "validation_controls": {
                "aligned": "native planned actions",
                "episode_shuffled": "deterministic cyclic next validation clip from a different episode",
                "zero": "train-mean planned action",
            },
            "metrics": {
                "primary": "raw reconstructed dense flow MSE",
                "secondary": ["endpoint_error", "flattened_directional_cosine"],
                "compression": "oracle target-PCA reconstruction metrics",
                "latency": "batch-one CPU feature-to-dense-field predictor latency; excludes RGB/flow extraction",
            },
            "bootstrap": {"paired": True, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
            "decision_gate": {
                "aligned_vs_history_dense_mse": {"point_min_percent": 10.0, "bootstrap_95_low_min_percent": 0.0},
                "aligned_vs_shuffled_dense_mse": {"point_min_percent": 10.0, "bootstrap_95_low_min_percent": 0.0},
                "aligned_vs_history_epe_point_min_percent": 0.0,
                "aligned_vs_history_cosine_delta_min": 0.0,
                "target_pca_train_explained_variance_ratio_min": 0.90,
                "all_required": True,
            },
            "latency_reporting": {"warmup": 32, "timed_predictions": 512, "percentiles": [50, 95]},
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "sklearn": sklearn.__version__,
            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "protected_test_accessed": False,
    }
    return seal(payload)


def grayscale_top(frame_chw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame_chw[:, :, :320], dtype=np.float32)
    rgb = (rgb + 1.0) * 127.5
    gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
    return cv2.resize(gray, FLOW_WORK_WH, interpolation=cv2.INTER_AREA)


def dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u = cv2.resize(flow[..., 0], FLOW_TARGET_WH, interpolation=cv2.INTER_AREA)
    v = cv2.resize(flow[..., 1], FLOW_TARGET_WH, interpolation=cv2.INTER_AREA)
    u *= FLOW_TARGET_WH[0] / FLOW_WORK_WH[0]
    v *= FLOW_TARGET_WH[1] / FLOW_WORK_WH[1]
    return np.stack((u, v)).astype(np.float32)


def clip_dense_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = [grayscale_top(frames[index]) for index in range(13)]
    flows = [dense_flow(gray[index], gray[index + 1]) for index in range(12)]
    history = np.stack([flows[index] for index in HISTORY_TRANSITIONS]).reshape(-1)
    future = np.stack([flows[index] for index in FUTURE_TRANSITIONS]).reshape(-1)
    return history.astype(np.float32), future.astype(np.float32)


def extract_split(cache_dir: Path, expected_count: int, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = read_json(cache_dir / "metadata.json")
    rgb = np.load(cache_dir / metadata["rgb_file"], mmap_mode="r", allow_pickle=False)
    actions = np.load(cache_dir / metadata["actions_file"], mmap_mode="r", allow_pickle=False)
    if tuple(rgb.shape) != (expected_count, 13, 3, 180, 960):
        raise RuntimeError(f"{split} RGB geometry differs: {rgb.shape}")
    if tuple(actions.shape) != (expected_count, 13, 5, 23):
        raise RuntimeError(f"{split} action geometry differs: {actions.shape}")
    histories, targets = [], []
    started = time.time()
    for index in range(expected_count):
        history, target = clip_dense_features(rgb[index])
        histories.append(history)
        targets.append(target)
        if index == 0 or (index + 1) % 64 == 0:
            print(json.dumps({"event": "extraction_progress", "split": split, "clips": index + 1, "elapsed_sec": time.time() - started}), flush=True)
    action_window = np.asarray(actions[:, ACTION_CHUNKS[0]:ACTION_CHUNKS[1]], dtype=np.float32)
    return np.stack(histories), action_window, np.stack(targets)


def standardizer(values: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.where(std > epsilon, std, 1.0).astype(np.float32)


def choose_alpha(inputs: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for alpha in ALPHAS:
        fold_scores = []
        for fit, score in folds.split(inputs):
            model = Ridge(alpha=alpha, solver="cholesky").fit(inputs[fit], targets[fit])
            fold_scores.append(float(np.mean((model.predict(inputs[score]) - targets[score]) ** 2)))
        scores[str(alpha)] = float(np.mean(fold_scores))
    return float(min(ALPHAS, key=lambda value: (scores[str(value)], value))), scores


def dense_clip_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred = prediction.reshape(-1, 8, 2, FLOW_TARGET_WH[1], FLOW_TARGET_WH[0])
    truth = target.reshape(-1, 8, 2, FLOW_TARGET_WH[1], FLOW_TARGET_WH[0])
    difference = pred - truth
    mse = np.mean(difference * difference, axis=(1, 2, 3, 4))
    epe = np.mean(np.sqrt(np.sum(difference * difference, axis=2)), axis=(1, 2, 3))
    pred_flat, truth_flat = pred.reshape(len(pred), -1), truth.reshape(len(truth), -1)
    cosine = np.sum(pred_flat * truth_flat, axis=1) / (
        np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(truth_flat, axis=1) + 1e-12
    )
    return {"dense_mse": mse, "endpoint_error": epe, "directional_cosine": cosine}


def lower_is_better_effect(reference: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    point = 100.0 * (1.0 - float(candidate.mean()) / float(reference.mean()))
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(reference), size=(BOOTSTRAP_SAMPLES, len(reference)))
    samples = 100.0 * (1.0 - candidate[indexes].mean(1) / reference[indexes].mean(1))
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": point,
        "paired_bootstrap_95_percent": [float(low), float(high)],
        "favorable_clip_fraction": float(np.mean(candidate < reference)),
    }


def higher_is_better_effect(reference: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    point = float(candidate.mean() - reference.mean())
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(reference), size=(BOOTSTRAP_SAMPLES, len(reference)))
    samples = candidate[indexes].mean(1) - reference[indexes].mean(1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "absolute_improvement": point,
        "paired_bootstrap_95": [float(low), float(high)],
        "favorable_clip_fraction": float(np.mean(candidate > reference)),
    }


def latency_summary(samples_ms: list[float]) -> dict:
    return {
        "median_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
        "timed_predictions": len(samples_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--val-cache", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    np.random.seed(SEED)

    registration = build_registration(args)
    write_json(args.output / "registration.json", registration)
    print(json.dumps({"event": "registered", "identity_sha256": registration["identity_sha256"]}), flush=True)

    train_manifest = read_jsonl(args.train_manifest)
    val_manifest = read_jsonl(args.val_manifest)
    train_episodes = {row["episode_dir"] for row in train_manifest}
    val_episodes = {row["episode_dir"] for row in val_manifest}
    if len(train_manifest) != 512 or len(val_manifest) != 64 or len(train_episodes) != 512 or len(val_episodes) != 64 or train_episodes & val_episodes:
        raise RuntimeError("immutable split cardinality or episode separation differs")

    train_history, train_actions, train_target = extract_split(args.train_cache, 512, "train")
    val_history, val_actions, val_target = extract_split(args.val_cache, 64, "validation")
    if not all(np.isfinite(array).all() for array in (train_history, train_actions, train_target, val_history, val_actions, val_target)):
        raise RuntimeError("derived arrays contain non-finite values")

    action_coordinate_std = train_actions.std(axis=(0, 1, 2), dtype=np.float64)
    active_coordinates = np.flatnonzero(action_coordinate_std > 1e-8)
    train_action = train_actions[..., active_coordinates].reshape(512, -1)
    val_action = val_actions[..., active_coordinates].reshape(64, -1)
    action_mean, action_std = standardizer(train_action)
    train_action_standard = (train_action - action_mean) / action_std
    val_action_standard = (val_action - action_mean) / action_std

    history_pca = PCA(n_components=HISTORY_PCA_COMPONENTS, svd_solver="randomized", random_state=SEED)
    action_pca = PCA(n_components=ACTION_PCA_COMPONENTS, svd_solver="randomized", random_state=SEED)
    target_pca = PCA(n_components=TARGET_PCA_COMPONENTS, svd_solver="randomized", random_state=SEED)
    train_history_pc = history_pca.fit_transform(train_history).astype(np.float32)
    val_history_pc = history_pca.transform(val_history).astype(np.float32)
    train_action_pc = action_pca.fit_transform(train_action_standard).astype(np.float32)
    val_action_pc = action_pca.transform(val_action_standard).astype(np.float32)
    zero_action_pc = action_pca.transform(np.zeros_like(val_action_standard)).astype(np.float32)
    train_target_pc = target_pca.fit_transform(train_target).astype(np.float32)
    val_target_pc = target_pca.transform(val_target).astype(np.float32)
    oracle_val_reconstruction = target_pca.inverse_transform(val_target_pc).astype(np.float32)

    donor = np.roll(np.arange(64), -1)
    if any(val_manifest[index]["episode_dir"] == val_manifest[int(donor[index])]["episode_dir"] for index in range(64)):
        raise RuntimeError("episode-shuffled donor is not episode-disjoint")
    shuffled_action_pc = val_action_pc[donor]

    history_input_mean, history_input_std = standardizer(train_history_pc)
    action_input_train = np.concatenate((train_history_pc, train_action_pc), axis=1)
    action_input_mean, action_input_std = standardizer(action_input_train)
    target_pc_mean, target_pc_std = standardizer(train_target_pc)
    x_history = (train_history_pc - history_input_mean) / history_input_std
    x_action = (action_input_train - action_input_mean) / action_input_std
    y = (train_target_pc - target_pc_mean) / target_pc_std
    val_x_history = (val_history_pc - history_input_mean) / history_input_std

    def val_action_input(action_pc_values: np.ndarray) -> np.ndarray:
        return (np.concatenate((val_history_pc, action_pc_values), axis=1) - action_input_mean) / action_input_std

    alpha_history, cv_history = choose_alpha(x_history, y)
    alpha_action, cv_action = choose_alpha(x_action, y)
    history_model = Ridge(alpha=alpha_history, solver="cholesky").fit(x_history, y)
    action_model = Ridge(alpha=alpha_action, solver="cholesky").fit(x_action, y)

    def reconstruct(standardized_coefficients: np.ndarray) -> np.ndarray:
        coefficients = standardized_coefficients * target_pc_std + target_pc_mean
        return target_pca.inverse_transform(coefficients).astype(np.float32)

    predictions = {
        "history_only": reconstruct(history_model.predict(val_x_history)),
        "history_plus_aligned_action": reconstruct(action_model.predict(val_action_input(val_action_pc))),
        "history_plus_episode_shuffled_action": reconstruct(action_model.predict(val_action_input(shuffled_action_pc))),
        "history_plus_zero_action": reconstruct(action_model.predict(val_action_input(zero_action_pc))),
    }
    metrics = {name: dense_clip_metrics(prediction, val_target) for name, prediction in predictions.items()}
    oracle_metrics = dense_clip_metrics(oracle_val_reconstruction, val_target)

    contrasts = {
        "aligned_vs_history_only": ("history_only", "history_plus_aligned_action"),
        "aligned_vs_episode_shuffled_same_model": ("history_plus_episode_shuffled_action", "history_plus_aligned_action"),
        "aligned_vs_zero_same_model": ("history_plus_zero_action", "history_plus_aligned_action"),
    }
    effects = {}
    for contrast_index, (contrast, (reference, candidate)) in enumerate(contrasts.items()):
        effects[contrast] = {
            "dense_mse": lower_is_better_effect(metrics[reference]["dense_mse"], metrics[candidate]["dense_mse"], BOOTSTRAP_SEED + 10 * contrast_index + 1),
            "endpoint_error": lower_is_better_effect(metrics[reference]["endpoint_error"], metrics[candidate]["endpoint_error"], BOOTSTRAP_SEED + 10 * contrast_index + 2),
            "directional_cosine": higher_is_better_effect(metrics[reference]["directional_cosine"], metrics[candidate]["directional_cosine"], BOOTSTRAP_SEED + 10 * contrast_index + 3),
        }

    def predict_history_one(index: int) -> np.ndarray:
        hp = history_pca.transform(val_history[index:index + 1])
        hx = (hp - history_input_mean) / history_input_std
        return reconstruct(history_model.predict(hx))

    def predict_action_one(index: int, raw_action: np.ndarray) -> np.ndarray:
        hp = history_pca.transform(val_history[index:index + 1])
        active = raw_action[..., active_coordinates].reshape(1, -1)
        ap = action_pca.transform((active - action_mean) / action_std)
        ax = (np.concatenate((hp, ap), axis=1) - action_input_mean) / action_input_std
        return reconstruct(action_model.predict(ax))

    for index in range(32):
        predict_history_one(index % 64)
        predict_action_one(index % 64, val_actions[index % 64])
    history_latencies, action_latencies = [], []
    for iteration in range(512):
        index = iteration % 64
        started = time.perf_counter_ns(); predict_history_one(index); history_latencies.append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns(); predict_action_one(index, val_actions[index]); action_latencies.append((time.perf_counter_ns() - started) / 1e6)

    primary = effects["aligned_vs_history_only"]
    specificity = effects["aligned_vs_episode_shuffled_same_model"]
    gates = {
        "aligned_vs_history_dense_mse": primary["dense_mse"]["relative_improvement_percent"] >= 10.0 and primary["dense_mse"]["paired_bootstrap_95_percent"][0] >= 0.0,
        "aligned_vs_shuffled_dense_mse": specificity["dense_mse"]["relative_improvement_percent"] >= 10.0 and specificity["dense_mse"]["paired_bootstrap_95_percent"][0] >= 0.0,
        "aligned_vs_history_epe_nonnegative": primary["endpoint_error"]["relative_improvement_percent"] >= 0.0,
        "aligned_vs_history_cosine_nonnegative": primary["directional_cosine"]["absolute_improvement"] >= 0.0,
        "target_pca_train_variance": float(target_pca.explained_variance_ratio_.sum()) >= 0.90,
    }
    gates["all_passed"] = all(gates.values())

    per_clip_path = args.output / "per_clip_metrics.jsonl"
    with per_clip_path.open("w") as handle:
        for index, descriptor in enumerate(val_manifest):
            row = {
                "schema_version": SCHEMA_VERSION,
                "clip_index": index,
                "clip_id": descriptor["clip_id"],
                "episode_dir": descriptor["episode_dir"],
                "donor_clip_index": int(donor[index]),
                "donor_episode_dir": val_manifest[int(donor[index])]["episode_dir"],
                "metrics": {name: {metric: float(values[metric][index]) for metric in values} for name, values in metrics.items()},
                "oracle_target_pca": {metric: float(values[index]) for metric, values in oracle_metrics.items()},
                "protected_test_accessed": False,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    derived_path = args.output / "derived_dense_features_targets.npz"
    np.savez_compressed(
        derived_path,
        train_history_dense_flow=train_history,
        train_action_window=train_actions,
        train_future_dense_flow_target=train_target,
        val_history_dense_flow=val_history,
        val_action_window=val_actions,
        val_future_dense_flow_target=val_target,
        active_action_coordinates=active_coordinates,
        val_episode_shuffled_donor=donor,
    )
    model_path = args.output / "model_state_and_predictions.npz"
    np.savez_compressed(
        model_path,
        history_pca_components=history_pca.components_, history_pca_mean=history_pca.mean_,
        action_pca_components=action_pca.components_, action_pca_mean=action_pca.mean_,
        target_pca_components=target_pca.components_, target_pca_mean=target_pca.mean_,
        target_pca_explained_variance_ratio=target_pca.explained_variance_ratio_,
        action_mean=action_mean, action_std=action_std,
        history_input_mean=history_input_mean, history_input_std=history_input_std,
        action_input_mean=action_input_mean, action_input_std=action_input_std,
        target_pc_mean=target_pc_mean, target_pc_std=target_pc_std,
        history_model_coef=history_model.coef_, history_model_intercept=history_model.intercept_,
        action_model_coef=action_model.coef_, action_model_intercept=action_model.intercept_,
        oracle_val_reconstruction=oracle_val_reconstruction,
        **{f"prediction_{name}": prediction for name, prediction in predictions.items()},
    )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dense_top_flow_stage0_analysis",
        "created_at_utc": now(),
        "registration_identity_sha256": registration["identity_sha256"],
        "decision": "GO" if gates["all_passed"] else "NO_GO",
        "interpretation": "dense top-view action signal clears the preregistered strict gate" if gates["all_passed"] else "dense top-view action signal does not clear every preregistered strict gate",
        "gates": gates,
        "effects": effects,
        "validation_means": {name: {metric: float(values[metric].mean()) for metric in values} for name, values in metrics.items()},
        "target_pca": {
            "components": TARGET_PCA_COMPONENTS,
            "train_explained_variance_ratio_sum": float(target_pca.explained_variance_ratio_.sum()),
            "validation_oracle_reconstruction": {metric: float(values.mean()) for metric, values in oracle_metrics.items()},
        },
        "input_compression": {
            "history_components": HISTORY_PCA_COMPONENTS,
            "history_train_explained_variance_ratio_sum": float(history_pca.explained_variance_ratio_.sum()),
            "action_components": ACTION_PCA_COMPONENTS,
            "action_train_explained_variance_ratio_sum": float(action_pca.explained_variance_ratio_.sum()),
            "active_action_coordinates": active_coordinates.tolist(),
            "action_coordinate_std": action_coordinate_std.tolist(),
        },
        "models": {
            "history_only": {"input_dim": int(x_history.shape[1]), "selected_alpha": alpha_history, "train_cv_mse_by_alpha": cv_history},
            "history_plus_action": {"input_dim": int(x_action.shape[1]), "selected_alpha": alpha_action, "train_cv_mse_by_alpha": cv_action},
            "output_dim": int(y.shape[1]),
        },
        "latency_batch_one_cpu_feature_to_dense_field": {
            "history_only": latency_summary(history_latencies),
            "history_plus_aligned_action": latency_summary(action_latencies),
            "excludes": ["RGB decoding", "Farneback history extraction", "data loading"],
        },
        "data_contract": {
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(val_episodes),
            "train_validation_episode_overlap": 0,
            "future_rgb_used_as_predictor_input": False,
            "future_rgb_used_for_target_and_scoring_only": True,
            "cached_vjepa_target_opened": False,
            "protected_test_accessed": False,
        },
        "artifacts": {
            "per_clip_metrics": file_record(per_clip_path),
            "derived_dense_features_targets": file_record(derived_path),
            "model_state_and_predictions": file_record(model_path),
        },
        "limitations": [
            "one seed and a previously reused exploratory ABC validation64 split",
            "Farneback is a pseudo-label and not ground-truth scene flow",
            "top view only; the strict all-view generator gate remains untested",
            "target PCA bounds the predictor to a train-derived linear flow subspace",
            "latency excludes online RGB-to-history-flow extraction",
            "no video generator, FVD, perceptual metric, or rollout was evaluated",
        ],
        "protected_test_accessed": False,
    }
    analysis = seal(analysis)
    write_json(args.output / "analysis.json", analysis)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dense_top_flow_stage0_complete",
        "completed_at_utc": now(),
        "registration_identity_sha256": registration["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "status": "completed",
        "artifacts": {
            "source": file_record(Path(__file__).resolve()),
            "registration": file_record(args.output / "registration.json"),
            "analysis": file_record(args.output / "analysis.json"),
            "per_clip_metrics": file_record(per_clip_path),
            "derived_dense_features_targets": file_record(derived_path),
            "model_state_and_predictions": file_record(model_path),
        },
        "protected_test_accessed": False,
    }
    complete = seal(complete)
    write_json(args.output / "run_complete.json", complete)
    print(json.dumps({"event": "completed", "decision": analysis["decision"], "analysis_identity_sha256": analysis["identity_sha256"], "complete_identity_sha256": complete["identity_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
