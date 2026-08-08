#!/usr/bin/env python3
"""Leakage-controlled Stage-0 action-to-motion proxy on immutable ABC caches.

Predictor inputs are observed RGB frames 0..4 and requested action chunks 4..11.
Targets are optical-flow summaries for future RGB transitions 4->5 .. 11->12.
No future RGB-derived quantity enters either predictor before scoring.
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
VIEWS = ("top", "left_wrist", "right_wrist")
LOWRES_WH = (80, 45)
GRID_HW = (2, 2)
HISTORY_TRANSITIONS = tuple(range(0, 4))
FUTURE_TRANSITIONS = tuple(range(4, 12))
ACTION_CHUNKS = (4, 12)
PCA_COMPONENTS = 64


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def canonical_identity(payload: dict) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seal(payload: dict) -> dict:
    payload = dict(payload)
    payload["identity_sha256"] = canonical_identity(payload)
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_record(path: Path, *, digest: bool = True) -> dict:
    result = {"path": str(path.resolve()), "bytes": path.stat().st_size}
    if digest:
        result["sha256"] = sha256_file(path)
    return result


def cache_record(cache_dir: Path, manifest_path: Path) -> dict:
    metadata_path = cache_dir / "metadata.json"
    meta = read_json(metadata_path)
    return {
        "cache_dir": str(cache_dir.resolve()),
        "metadata": file_record(metadata_path),
        "manifest": file_record(manifest_path),
        "clip_count": int(meta["clip_count"]),
        "split": meta["split"],
        "rgb": {
            "path": str((cache_dir / meta["rgb_file"]).resolve()),
            "bytes": (cache_dir / meta["rgb_file"]).stat().st_size,
            "registered_sha256": meta["rgb_sha256"],
            "shape": meta["rgb_shape"],
            "dtype": meta["rgb_dtype"],
        },
        "actions": {
            "path": str((cache_dir / meta["actions_file"]).resolve()),
            "bytes": (cache_dir / meta["actions_file"]).stat().st_size,
            "registered_sha256": meta["actions_sha256"],
            "shape": meta["actions_shape"],
            "dtype": meta["actions_dtype"],
        },
        "cached_vjepa_target_opened": False,
    }


def registration(args: argparse.Namespace) -> dict:
    script = Path(__file__).resolve()
    train = cache_record(args.train_cache, args.train_manifest)
    val = cache_record(args.val_cache, args.val_manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_flow_proxy_stage0_registration",
        "status": "registered_before_feature_or_target_extraction",
        "created_at_utc": now(),
        "source": file_record(script),
        "inputs": {"train": train, "validation": val},
        "protocol": {
            "question": "do planned actions add validation-set predictive information about future per-view motion beyond observed history?",
            "seed": SEED,
            "train_clips": 512,
            "validation_clips": 64,
            "protected_test_access_allowed": False,
            "predictor_inputs": {
                "history_rgb_frames": [0, 1, 2, 3, 4],
                "history_feature": "Farneback flow summaries for transitions 0->1 through 3->4",
                "planned_actions": "raw action chunks [4:12], five substeps each; train-active coordinates only; train-standardized then PCA64",
                "future_rgb_or_future_derived_feature": False,
            },
            "target_only": {
                "future_rgb_frames": list(range(4, 13)),
                "transitions": [f"{i}->{i+1}" for i in FUTURE_TRANSITIONS],
                "summary": "per view, per transition, 2x2 grid of mean dx, mean dy, mean magnitude, q75 magnitude",
                "flow": "OpenCV Farneback on grayscale 80x45 views",
            },
            "views": list(VIEWS),
            "camera_slicing": "RGB width 960 split into three contiguous 320-pixel views in metadata camera order",
            "models": {
                "history_only": "train-standardized history features -> multi-output Ridge",
                "history_plus_action": "same history plus PCA64 action features -> multi-output Ridge",
                "alpha_grid": list(ALPHAS),
                "alpha_selection": "five-fold train-only KFold standardized-target MSE",
            },
            "validation_controls": {
                "aligned": "native planned action for each clip",
                "episode_shuffled": "deterministic cyclic next validation clip; all donors are different episodes",
                "zero": "train-mean planned action",
            },
            "primary_metric": "mean per-clip dimension-standardized future-flow-summary MSE",
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "paired": True},
            "exploratory_go_gate": {
                "history_plus_aligned_vs_history_only": {"point_min_percent": 1.0, "bootstrap_95_low_min_percent": 0.0},
                "aligned_vs_episode_shuffled_same_model": {"point_min_percent": 1.0, "bootstrap_95_low_min_percent": 0.0},
                "all_required": True,
            },
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "sklearn": sklearn.__version__,
        },
        "protected_test_accessed": False,
    }
    return seal(payload)


def grayscale_view(frame_chw: np.ndarray, view_index: int) -> np.ndarray:
    x0, x1 = view_index * 320, (view_index + 1) * 320
    rgb = np.asarray(frame_chw[:, :, x0:x1], dtype=np.float32)
    rgb = (rgb + 1.0) * 127.5
    gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
    return cv2.resize(gray, LOWRES_WH, interpolation=cv2.INTER_AREA)


def summarize_flow(flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    gy, gx = GRID_HW
    rows: list[float] = []
    for iy in range(gy):
        y0, y1 = (iy * h) // gy, ((iy + 1) * h) // gy
        for ix in range(gx):
            x0, x1 = (ix * w) // gx, ((ix + 1) * w) // gx
            cell = flow[y0:y1, x0:x1]
            mag = np.sqrt(np.sum(cell * cell, axis=-1))
            rows.extend((float(cell[..., 0].mean()), float(cell[..., 1].mean()), float(mag.mean()), float(np.quantile(mag, 0.75))))
    return np.asarray(rows, dtype=np.float32)


def clip_motion_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    summaries: dict[tuple[int, int], np.ndarray] = {}
    for view in range(len(VIEWS)):
        gray = [grayscale_view(frames[t], view) for t in range(13)]
        for transition in range(12):
            flow = cv2.calcOpticalFlowFarneback(
                gray[transition], gray[transition + 1], None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            summaries[(transition, view)] = summarize_flow(flow)
    history = np.concatenate([summaries[(t, v)] for t in HISTORY_TRANSITIONS for v in range(len(VIEWS))])
    future = np.concatenate([summaries[(t, v)] for t in FUTURE_TRANSITIONS for v in range(len(VIEWS))])
    return history.astype(np.float32), future.astype(np.float32)


def extract_split(cache_dir: Path, expected_count: int, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = read_json(cache_dir / "metadata.json")
    rgb = np.load(cache_dir / meta["rgb_file"], mmap_mode="r")
    actions = np.load(cache_dir / meta["actions_file"], mmap_mode="r")
    if tuple(rgb.shape) != (expected_count, 13, 3, 180, 960):
        raise RuntimeError(f"{label} RGB geometry differs: {rgb.shape}")
    if tuple(actions.shape) != (expected_count, 13, 5, 23):
        raise RuntimeError(f"{label} action geometry differs: {actions.shape}")
    histories, targets = [], []
    start = time.time()
    for index in range(expected_count):
        history, target = clip_motion_features(rgb[index])
        histories.append(history)
        targets.append(target)
        if index == 0 or (index + 1) % 64 == 0:
            print(json.dumps({"event": "extraction_progress", "split": label, "clips": index + 1, "elapsed_sec": time.time() - start}), flush=True)
    action_window = np.asarray(actions[:, ACTION_CHUNKS[0]:ACTION_CHUNKS[1]], dtype=np.float32)
    if not np.isfinite(action_window).all():
        raise RuntimeError(f"{label} actions contain non-finite values")
    return np.stack(histories), action_window, np.stack(targets)


def standardizer(x: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > epsilon, std, 1.0).astype(np.float32)
    return mean, std


def choose_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        fold_scores = []
        for fit_index, score_index in folds.split(x):
            model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            model.fit(x[fit_index], y[fit_index])
            pred = model.predict(x[score_index])
            fold_scores.append(float(np.mean((pred - y[score_index]) ** 2)))
        scores[str(alpha)] = float(np.mean(fold_scores))
    selected = min(ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    return float(selected), scores


def paired_effect(reference: np.ndarray, candidate: np.ndarray, seed_offset: int) -> dict:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired errors must be same-shape vectors")
    point = 100.0 * (1.0 - float(candidate.mean()) / float(reference.mean()))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(reference), size=(BOOTSTRAP_SAMPLES, len(reference)))
    ref_means = reference[indices].mean(axis=1)
    cand_means = candidate[indices].mean(axis=1)
    samples = 100.0 * (1.0 - cand_means / ref_means)
    low, high = np.quantile(samples, [0.025, 0.975])
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


def view_horizon_mse(pred: np.ndarray, target: np.ndarray) -> dict:
    error = ((pred - target) ** 2).reshape(len(pred), 8, 3, 4, 4)
    return {
        "by_view": {view: float(error[:, :, i].mean()) for i, view in enumerate(VIEWS)},
        "by_horizon": [float(error[:, h].mean()) for h in range(8)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    np.random.seed(SEED)

    reg = registration(args)
    write_json(args.output / "registration.json", reg)
    print(json.dumps({"event": "registered", "identity_sha256": reg["identity_sha256"]}), flush=True)

    train_manifest = read_jsonl(args.train_manifest)
    val_manifest = read_jsonl(args.val_manifest)
    if len(train_manifest) != 512 or len(val_manifest) != 64:
        raise RuntimeError("immutable manifest counts differ")
    train_episodes = {row["episode_dir"] for row in train_manifest}
    val_episodes = {row["episode_dir"] for row in val_manifest}
    if len(train_episodes) != 512 or len(val_episodes) != 64 or train_episodes & val_episodes:
        raise RuntimeError("episode separation contract differs")

    train_history, train_actions, train_target = extract_split(args.train_cache, 512, "train")
    val_history, val_actions, val_target = extract_split(args.val_cache, 64, "validation")

    action_flat_train_all = train_actions.reshape(512, -1)
    action_coordinate_std = train_actions.std(axis=(0, 1, 2), dtype=np.float64)
    active_coordinates = np.flatnonzero(action_coordinate_std > 1e-8)
    if len(active_coordinates) == 0:
        raise RuntimeError("no active action coordinates")
    train_action_active = train_actions[..., active_coordinates].reshape(512, -1)
    val_action_active = val_actions[..., active_coordinates].reshape(64, -1)

    h_mean, h_std = standardizer(train_history)
    a_mean, a_std = standardizer(train_action_active)
    y_mean, y_std = standardizer(train_target)
    train_h = (train_history - h_mean) / h_std
    val_h = (val_history - h_mean) / h_std
    train_a_raw = (train_action_active - a_mean) / a_std
    val_a_raw = (val_action_active - a_mean) / a_std
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    train_a_pca = pca.fit_transform(train_a_raw).astype(np.float32)
    val_a_pca = pca.transform(val_a_raw).astype(np.float32)
    zero_a_pca = pca.transform(np.zeros_like(val_a_raw)).astype(np.float32)
    donor = np.roll(np.arange(64), -1)
    if any(val_manifest[i]["episode_dir"] == val_manifest[int(donor[i])]["episode_dir"] for i in range(64)):
        raise RuntimeError("shuffled donor is not episode-disjoint")
    shuffled_a_pca = val_a_pca[donor]

    train_y = (train_target - y_mean) / y_std
    val_y = (val_target - y_mean) / y_std
    x_history = train_h
    x_action = np.concatenate((train_h, train_a_pca), axis=1)
    xh_mean, xh_std = standardizer(x_history)
    xa_mean, xa_std = standardizer(x_action)
    x_history = (x_history - xh_mean) / xh_std
    x_action = (x_action - xa_mean) / xa_std
    val_xh = (val_h - xh_mean) / xh_std

    def action_x(action_features: np.ndarray) -> np.ndarray:
        return (np.concatenate((val_h, action_features), axis=1) - xa_mean) / xa_std

    alpha_h, cv_h = choose_alpha(x_history, train_y)
    alpha_a, cv_a = choose_alpha(x_action, train_y)
    history_model = Ridge(alpha=alpha_h, fit_intercept=True, solver="cholesky").fit(x_history, train_y)
    action_model = Ridge(alpha=alpha_a, fit_intercept=True, solver="cholesky").fit(x_action, train_y)
    predictions = {
        "history_only": history_model.predict(val_xh).astype(np.float32),
        "history_plus_aligned_action": action_model.predict(action_x(val_a_pca)).astype(np.float32),
        "history_plus_episode_shuffled_action": action_model.predict(action_x(shuffled_a_pca)).astype(np.float32),
        "history_plus_zero_action": action_model.predict(action_x(zero_a_pca)).astype(np.float32),
    }
    clip_errors = {name: np.mean((pred - val_y) ** 2, axis=1) for name, pred in predictions.items()}

    effects = {
        "aligned_vs_history_only": paired_effect(clip_errors["history_only"], clip_errors["history_plus_aligned_action"], 1),
        "aligned_vs_episode_shuffled_same_model": paired_effect(clip_errors["history_plus_episode_shuffled_action"], clip_errors["history_plus_aligned_action"], 2),
        "aligned_vs_zero_same_model": paired_effect(clip_errors["history_plus_zero_action"], clip_errors["history_plus_aligned_action"], 3),
    }
    gate_h = effects["aligned_vs_history_only"]["relative_improvement_percent"] >= 1.0 and effects["aligned_vs_history_only"]["paired_bootstrap_95_percent"][0] >= 0.0
    gate_s = effects["aligned_vs_episode_shuffled_same_model"]["relative_improvement_percent"] >= 1.0 and effects["aligned_vs_episode_shuffled_same_model"]["paired_bootstrap_95_percent"][0] >= 0.0

    per_clip_path = args.output / "per_clip_metrics.jsonl"
    with per_clip_path.open("w") as handle:
        for i, row in enumerate(val_manifest):
            payload = {
                "schema_version": SCHEMA_VERSION,
                "clip_index": i,
                "clip_id": row["clip_id"],
                "episode_dir": row["episode_dir"],
                "donor_clip_index": int(donor[i]),
                "donor_episode_dir": val_manifest[int(donor[i])]["episode_dir"],
                "errors": {name: float(values[i]) for name, values in clip_errors.items()},
                "protected_test_accessed": False,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    features_path = args.output / "derived_features_targets.npz"
    np.savez_compressed(
        features_path,
        train_history=train_history,
        train_action_window=train_actions,
        train_future_motion_target=train_target,
        val_history=val_history,
        val_action_window=val_actions,
        val_future_motion_target=val_target,
        active_action_coordinates=active_coordinates,
        val_episode_shuffled_donor=donor,
    )
    model_path = args.output / "model_state_and_predictions.npz"
    np.savez_compressed(
        model_path,
        history_mean=h_mean,
        history_std=h_std,
        action_mean=a_mean,
        action_std=a_std,
        target_mean=y_mean,
        target_std=y_std,
        history_input_mean=xh_mean,
        history_input_std=xh_std,
        action_input_mean=xa_mean,
        action_input_std=xa_std,
        action_pca_mean=pca.mean_,
        action_pca_components=pca.components_,
        action_pca_explained_variance_ratio=pca.explained_variance_ratio_,
        history_model_coef=history_model.coef_,
        history_model_intercept=history_model.intercept_,
        action_model_coef=action_model.coef_,
        action_model_intercept=action_model.intercept_,
        validation_target_standardized=val_y,
        **{f"prediction_{name}": value for name, value in predictions.items()},
    )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_flow_proxy_stage0_analysis",
        "created_at_utc": now(),
        "registration_identity_sha256": reg["identity_sha256"],
        "decision": "GO" if gate_h and gate_s else "NO_GO",
        "interpretation": "planned actions clear both exploratory incremental-prediction and shuffled-action-specificity gates" if gate_h and gate_s else "planned actions do not clear both exploratory Stage-0 gates",
        "gates": {"aligned_vs_history_only": gate_h, "aligned_vs_episode_shuffled_same_model": gate_s, "all_passed": gate_h and gate_s},
        "effects": effects,
        "validation_mean_standardized_mse": {name: float(values.mean()) for name, values in clip_errors.items()},
        "validation_aggregate_standardized_r2": {name: float(1.0 - np.sum((pred - val_y) ** 2) / np.sum(val_y ** 2)) for name, pred in predictions.items()},
        "view_and_horizon_mse": {name: view_horizon_mse(pred, val_y) for name, pred in predictions.items()},
        "models": {
            "history_only": {"selected_alpha": alpha_h, "train_cv_mse_by_alpha": cv_h, "input_dim": int(x_history.shape[1])},
            "history_plus_action": {"selected_alpha": alpha_a, "train_cv_mse_by_alpha": cv_a, "input_dim": int(x_action.shape[1])},
            "target_dim": int(train_y.shape[1]),
            "action_pca_components": PCA_COMPONENTS,
            "action_pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "active_action_coordinates": active_coordinates.tolist(),
            "raw_action_coordinate_std": action_coordinate_std.tolist(),
            "raw_action_flat_dim": int(action_flat_train_all.shape[1]),
            "active_future_action_flat_dim": int(train_action_active.shape[1]),
        },
        "data_contract": {
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(val_episodes),
            "train_validation_episode_overlap": 0,
            "future_rgb_used_as_predictor_input": False,
            "future_rgb_used_for_target_and_scoring_only": True,
            "protected_test_accessed": False,
            "cached_vjepa_target_opened": False,
        },
        "artifacts": {
            "per_clip_metrics": file_record(per_clip_path),
            "derived_features_targets": file_record(features_path),
            "model_state_and_predictions": file_record(model_path),
        },
        "limitations": [
            "one immutable ABC train512/val64 split and one fixed seed",
            "validation64 has been reused by earlier exploratory studies and is not a protected confirmation set",
            "Farneback summaries mix robot, object, background, and wrist-camera ego-motion",
            "ridge prediction of a low-resolution summary does not prove a generator can use a dense scaffold",
            "aligned actions can carry task/state correlation; shuffled-action specificity is diagnostic, not randomized causal intervention",
            "no FVD, perceptual metric, generator training, or long-horizon rollout is evaluated",
        ],
        "protected_test_accessed": False,
    }
    analysis = seal(analysis)
    write_json(args.output / "analysis.json", analysis)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_flow_proxy_stage0_complete",
        "completed_at_utc": now(),
        "registration_identity_sha256": reg["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "artifacts": {
            "source": file_record(Path(__file__).resolve()),
            "registration": file_record(args.output / "registration.json"),
            "analysis": file_record(args.output / "analysis.json"),
            "per_clip_metrics": file_record(per_clip_path),
            "derived_features_targets": file_record(features_path),
            "model_state_and_predictions": file_record(model_path),
        },
        "protected_test_accessed": False,
        "status": "completed",
    }
    complete = seal(complete)
    write_json(args.output / "run_complete.json", complete)
    print(json.dumps({"event": "completed", "decision": analysis["decision"], "analysis_identity_sha256": analysis["identity_sha256"], "complete_identity_sha256": complete["identity_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
