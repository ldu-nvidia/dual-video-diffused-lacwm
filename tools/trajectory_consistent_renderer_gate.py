#!/usr/bin/env python3
"""Prospective fresh-episode gate for causal robot-trajectory renderers.

The workflow has immutable ``register``, ``prepare``, ``evaluate``, and
read-only ``audit`` transitions.  Registration is the only transition allowed
to select score clips, and it cannot open their state or RGB.  Validation and
protected test inputs are unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tools import corrected_renderer_attribution as base
from tools.abc_d405_nominal_geometry_probe import (
    CAMERA_TYPE,
    EXPECTED_ABC_COMMIT,
    OFFICIAL_ASSET_RELATIVE,
    OFFICIAL_SCENE_RELATIVE,
    _decode_top_calibration,
    _joint_qpos_addresses,
    _read_video_frames,
    build_robot_only_xml,
    observed_edges,
)


SCHEMA_VERSION = 1
MANIFEST_COUNT = 512
FIT_STOP = 384
SCORE_START = 384
SCORE_STOP = 512
SCORE_CLIPS = 24
EXPECTED_D405_POOL = 103
EXPECTED_UNTOUCHED_D405 = 79
MOTION_STRATA = 3
CLIPS_PER_STRATUM = 8
SELECTION_SALT = "trajectory-consistent-renderer|20260808"
SEED = 1234
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_SAMPLES = 20_000
HOLM_ALPHA = 0.05
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
RECURRENT_GAINS = (0.25, 0.5, 0.75, 1.0)
SELF_ROLLOUT_ROUNDS = 3
PCA_COMPONENTS = 64
ACTION_DIM = 14
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
FLOW_MIN_RELATIVE_IMPROVEMENT_PERCENT = 5.0
MIN_FAVORABLE_CLIP_FRACTION = 0.60
ROBOT_BAND_PX = 12
FLOW_PIXEL_STRIDE = 2
FLOW_MIN_ORACLE_MOTION_PX = 0.25

ARMS = (
    "raw_command",
    "raw_episode_shuffled",
    "absolute_ridge",
    "recurrent_delta",
    "recurrent_delta_shuffled",
    "hybrid_anchor_raw_delta",
    "hold_current",
    "measured_oracle",
)
LOWER_METRICS = (
    "silhouette_boundary_chamfer_px",
    "rgb_robot_band_chamfer_px",
    "robot_flow_epe_px",
)
PRIMARY_FLOW_LABELS = (
    "recurrent_delta_vs_raw_command:robot_flow_epe_px",
    "recurrent_delta_vs_absolute_ridge:robot_flow_epe_px",
    "recurrent_delta_vs_recurrent_delta_shuffled:robot_flow_epe_px",
    "hybrid_anchor_raw_delta_vs_raw_command:robot_flow_epe_px",
    "hybrid_anchor_raw_delta_vs_absolute_ridge:robot_flow_epe_px",
    "raw_command_vs_raw_episode_shuffled:robot_flow_epe_px",
    "raw_command_vs_hold_current:robot_flow_epe_px",
)


class GateError(RuntimeError):
    """Raised when a frozen split, causality, or artifact contract differs."""


def selection_hash(clip_id: str) -> str:
    return hashlib.sha256(f"{clip_id}|{SELECTION_SALT}".encode()).hexdigest()


def select_fresh_motion_stratified(
    eligible: list[dict[str, Any]], excluded_clip_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude Gate-0c score24, then select 8/stratum without outcomes."""

    untouched = [dict(item) for item in eligible if item["clip_id"] not in excluded_clip_ids]
    if len(eligible) != EXPECTED_D405_POOL:
        raise GateError(f"D405 score pool is {len(eligible)}, expected {EXPECTED_D405_POOL}")
    if len(untouched) != EXPECTED_UNTOUCHED_D405:
        raise GateError(
            f"untouched D405 pool is {len(untouched)}, expected {EXPECTED_UNTOUCHED_D405}"
        )
    ordered = sorted(
        untouched,
        key=lambda item: (float(item["planned_joint_motion_rms"]), item["clip_id"]),
    )
    bins = np.array_split(np.asarray(ordered, dtype=object), MOTION_STRATA)
    selected: list[dict[str, Any]] = []
    for stratum, values in enumerate(bins):
        candidates = [dict(item) for item in values.tolist()]
        chosen = sorted(
            candidates,
            key=lambda item: (selection_hash(item["clip_id"]), item["clip_id"]),
        )[:CLIPS_PER_STRATUM]
        if len(chosen) != CLIPS_PER_STRATUM:
            raise GateError(f"fresh stratum {stratum} yielded only {len(chosen)} clips")
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
    for order, item in enumerate(selected):
        item["score_order"] = order
    if len(selected) != SCORE_CLIPS or set(item["clip_id"] for item in selected) & excluded_clip_ids:
        raise GateError("fresh selection count or exclusion differs")
    return selected, untouched


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if len(args.source_commit) != 40 or any(character not in "0123456789abcdef" for character in args.source_commit):
        raise GateError("--source-commit must be a lowercase 40-character Git commit")

    prior_path = args.prior_registration.resolve()
    prior = base.read_json(prior_path)
    base.validate_identity(prior, str(prior_path))
    if prior.get("kind") != "corrected_renderer_attribution_registration":
        raise GateError("prior registration is not the completed Gate-0c registration kind")
    prior_selected = prior["selection"]["selected"]
    excluded_ids = {str(item["clip_id"]) for item in prior_selected}
    if len(excluded_ids) != 24:
        raise GateError("prior Gate-0c registration does not contain 24 unique selected clips")
    prior_copy = output / "prior_gate0c_registration.json"
    prior_copy.write_bytes(prior_path.read_bytes())
    if base.sha256_file(prior_copy) != base.sha256_file(prior_path):
        raise GateError("failed to preserve the prior registration byte-for-byte")

    rows = base.read_jsonl(args.train_manifest)
    base.validate_train_manifest(rows)
    actions, metadata, actions_path = base.cache_actions(args.train_cache)
    eligible: list[dict[str, Any]] = []
    camera_counts: dict[str, int] = {}
    for index in range(SCORE_START, SCORE_STOP):
        row = rows[index]
        episode = Path(row["episode_dir"])
        if not (episode / "states.npz").is_file() or not (episode / "top.mp4").is_file():
            continue
        mcap_path = base.raw_mcap_path(row, args.preprocessed_root, args.raw_root)
        camera_type = base.read_mcap_metadata(mcap_path).get("top_camera_type", "missing")
        camera_counts[camera_type] = camera_counts.get(camera_type, 0) + 1
        if camera_type != CAMERA_TYPE:
            continue
        eligible.append(
            {
                "manifest_index": index,
                "clip_id": str(row["clip_id"]),
                "episode_dir": str(episode.resolve()),
                "raw_mcap": str(mcap_path.resolve()),
                "top_camera_type": camera_type,
                "planned_joint_motion_rms": base.planned_joint_motion(actions[index]),
                "protected_test_accessed": False,
            }
        )
    eligible_ids = {item["clip_id"] for item in eligible}
    if not excluded_ids.issubset(eligible_ids):
        raise GateError("prior selected IDs are not a subset of the current D405 pool")
    selected, untouched = select_fresh_motion_stratified(eligible, excluded_ids)
    script = Path(__file__).resolve()
    registration = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "trajectory_consistent_renderer_gate_registration",
            "status": "registered_before_fresh_score_state_or_rgb_access",
            "created_at_utc": base.now(),
            "source": {
                **base.file_record(script),
                "registered_git_commit": args.source_commit,
                "runtime_worktree_git_commit": base.git_commit(script.parents[1]),
            },
            "prior_gate0c": {
                "registration": base.file_record(prior_copy),
                "source_registration_path": str(prior_path),
                "registration_identity_sha256": prior["identity_sha256"],
                "excluded_clip_ids": sorted(excluded_ids),
                "excluded_count": len(excluded_ids),
            },
            "inputs": {
                "train_manifest": base.file_record(args.train_manifest),
                "train_cache_metadata": base.file_record(args.train_cache / "metadata.json"),
                "train_actions": {
                    **base.file_record(actions_path),
                    "shape": list(actions.shape),
                    "dtype": str(actions.dtype),
                    "registered_sha256": metadata["actions_sha256"],
                },
                "preprocessed_root": str(args.preprocessed_root.resolve()),
                "raw_root": str(args.raw_root.resolve()),
                "score_pool_camera_type_counts": camera_counts,
                "score_pool_eligible_d405": len(eligible),
                "untouched_eligible_d405": len(untouched),
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
                "selection_salt": SELECTION_SALT,
                "motion_strata": MOTION_STRATA,
                "clips_per_stratum": CLIPS_PER_STRATUM,
                "selected": selected,
            },
            "protocol": {
                "document": "docs/experiments/TRAJECTORY_CONSISTENT_RENDERER_GATE_PROTOCOL.md",
                "arms": list(ARMS),
                "absolute_ridge_alphas": list(ALPHAS),
                "recurrent_delta_alphas": list(ALPHAS),
                "recurrent_delta_gains": list(RECURRENT_GAINS),
                "self_rollout_rounds": SELF_ROLLOUT_ROUNDS,
                "pca_components": PCA_COMPONENTS,
                "future_measured_state_predictor_input": False,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "holm_alpha": HOLM_ALPHA,
                "primary_flow_labels": list(PRIMARY_FLOW_LABELS),
                "minimum_flow_relative_improvement_percent": (
                    FLOW_MIN_RELATIVE_IMPROVEMENT_PERCENT
                ),
                "minimum_favorable_clip_fraction": MIN_FAVORABLE_CLIP_FRACTION,
                "all_conditions_within_each_family_required": True,
                "families_cannot_rescue_each_other": True,
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
    base.write_json(output / "registration.json", registration)
    return registration


@dataclass(frozen=True)
class PCAProjection:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_active: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values).reshape(len(values), -1)
        normalized = ((flat - self.feature_mean) / self.feature_std)[:, self.feature_active]
        return ((normalized - self.pca_mean) @ self.pca_components.T).astype(np.float32)


def fit_projection(values: np.ndarray) -> tuple[PCAProjection, np.ndarray]:
    pca, transformed, mean, std, active = base.fit_pca(values)
    state = PCAProjection(
        feature_mean=mean,
        feature_std=std,
        feature_active=active,
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
    )
    replay = state.transform(values)
    if not np.allclose(replay, transformed, atol=2e-4, rtol=2e-4):
        raise GateError("serialized PCA replay differs")
    return state, transformed


@dataclass(frozen=True)
class RidgeState:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features) - self.input_mean) / self.input_std
        standardized = values @ self.coef.T + self.intercept
        return (standardized * self.target_std + self.target_mean).astype(np.float32)


def fit_ridge_state(features: np.ndarray, targets: np.ndarray, alpha: float) -> RidgeState:
    from sklearn.linear_model import Ridge

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    x_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std_raw = x.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std_raw > 1e-8, x_std_raw, 1.0).astype(np.float32)
    y_mean = y.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std_raw = y.std(axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.where(y_std_raw > 1e-8, y_std_raw, 1.0).astype(np.float32)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(
        (x - x_mean) / x_std, (y - y_mean) / y_std
    )
    return RidgeState(
        input_mean=x_mean,
        input_std=x_std,
        target_mean=y_mean,
        target_std=y_std,
        coef=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
    )


def raw_trajectory(q0: np.ndarray, future_actions: np.ndarray) -> np.ndarray:
    q = np.asarray(q0, dtype=np.float32)
    actions = np.asarray(future_actions, dtype=np.float32)
    if q.shape != (len(actions), ACTION_DIM) or actions.shape[1:] != (8, CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"raw trajectory geometry differs: {q.shape}/{actions.shape}")
    return np.concatenate((q[:, None], actions[:, :, -1]), axis=1)


def recurrent_base_features(
    history_context: np.ndarray,
    action_context: np.ndarray,
    future_actions: np.ndarray,
    raw: np.ndarray,
    predicted_source: np.ndarray,
    previous_correction: np.ndarray,
    step: int,
) -> np.ndarray:
    """Build one transition's causal features; no target array is accepted."""

    n = len(history_context)
    time_one_hot = np.zeros((n, 8), dtype=np.float32)
    time_one_hot[:, step] = 1.0
    local = np.asarray(future_actions[:, step], dtype=np.float32).reshape(n, -1)
    raw_delta = raw[:, step + 1] - raw[:, step]
    return np.concatenate(
        (
            history_context,
            action_context,
            local,
            raw[:, step + 1],
            raw_delta,
            time_one_hot,
            predicted_source,
            predicted_source - raw[:, step],
            previous_correction,
        ),
        axis=1,
    ).astype(np.float32)


def rollout_recurrent(
    model: RidgeState,
    history_context: np.ndarray,
    action_context: np.ndarray,
    future_actions: np.ndarray,
    q0: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Closed-loop causal rollout.  Future measured state is not an argument."""

    raw = raw_trajectory(q0, future_actions)
    predicted = np.empty_like(raw)
    predicted[:, 0] = q0
    previous_correction = np.zeros_like(q0)
    for step in range(8):
        features = recurrent_base_features(
            history_context,
            action_context,
            future_actions,
            raw,
            predicted[:, step],
            previous_correction,
            step,
        )
        correction = model.predict(features)
        raw_delta = raw[:, step + 1] - raw[:, step]
        predicted[:, step + 1] = predicted[:, step] + raw_delta + gain * correction
        previous_correction = gain * correction
    return predicted


def _features_from_frozen_rollout(
    history_context: np.ndarray,
    action_context: np.ndarray,
    future_actions: np.ndarray,
    q0: np.ndarray,
    frozen_rollout: np.ndarray,
) -> np.ndarray:
    raw = raw_trajectory(q0, future_actions)
    blocks: list[np.ndarray] = []
    for step in range(8):
        previous = (
            np.zeros_like(q0)
            if step == 0
            else frozen_rollout[:, step] - frozen_rollout[:, step - 1]
            - (raw[:, step] - raw[:, step - 1])
        )
        blocks.append(
            recurrent_base_features(
                history_context,
                action_context,
                future_actions,
                raw,
                frozen_rollout[:, step],
                previous,
                step,
            )
        )
    return np.stack(blocks, axis=1).reshape(len(q0) * 8, -1)


def fit_recurrent_rounds(
    history_context: np.ndarray,
    action_context: np.ndarray,
    future_actions: np.ndarray,
    q0: np.ndarray,
    measured: np.ndarray,
    *,
    alpha: float,
    gain: float,
) -> tuple[RidgeState, np.ndarray, list[dict[str, float]]]:
    """Fit three target-only self-rollout rounds and return final rollout."""

    raw = raw_trajectory(q0, future_actions)
    target_correction = (np.diff(measured, axis=1) - np.diff(raw, axis=1)).reshape(-1, ACTION_DIM)
    frozen = raw.copy()
    rounds: list[dict[str, float]] = []
    model: RidgeState | None = None
    for round_index in range(SELF_ROLLOUT_ROUNDS):
        features = _features_from_frozen_rollout(
            history_context, action_context, future_actions, q0, frozen
        )
        model = fit_ridge_state(features, target_correction, alpha)
        frozen = rollout_recurrent(
            model, history_context, action_context, future_actions, q0, gain
        )
        rounds.append(
            {
                "round": float(round_index),
                "joint_pose_mse": float(np.mean((frozen[:, 1:, :12] - measured[:, 1:, :12]) ** 2)),
                "joint_delta_mse": float(
                    np.mean((np.diff(frozen[..., :12], axis=1) - np.diff(measured[..., :12], axis=1)) ** 2)
                ),
            }
        )
    assert model is not None
    return model, frozen, rounds


def recurrent_cv_score(predicted: np.ndarray, measured: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    pose = float(np.mean((predicted[:, 1:, :12] - measured[:, 1:, :12]) ** 2))
    delta = float(
        np.mean((np.diff(predicted[..., :12], axis=1) - np.diff(measured[..., :12], axis=1)) ** 2)
    )
    raw_pose = float(np.mean((raw[:, 1:, :12] - measured[:, 1:, :12]) ** 2))
    raw_delta = float(
        np.mean((np.diff(raw[..., :12], axis=1) - np.diff(measured[..., :12], axis=1)) ** 2)
    )
    return {
        "joint_pose_mse": pose,
        "joint_delta_mse": delta,
        "raw_joint_pose_mse": raw_pose,
        "raw_joint_delta_mse": raw_delta,
        "normalized_composite": 0.5 * (pose / raw_pose + delta / raw_delta),
    }


def choose_recurrent_hyperparameters(
    history: np.ndarray,
    future_actions: np.ndarray,
    q0: np.ndarray,
    measured: np.ndarray,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Five-fold clip CV; transforms and self-rollout fits are fold-local."""

    from sklearn.model_selection import KFold

    records: dict[tuple[float, float], list[dict[str, float]]] = {
        (alpha, gain): [] for alpha in ALPHAS for gain in RECURRENT_GAINS
    }
    for train_index, score_index in KFold(n_splits=5, shuffle=True, random_state=SEED).split(history):
        h_projection, train_h = fit_projection(history[train_index])
        a_projection, train_a = fit_projection(future_actions[train_index])
        score_h = h_projection.transform(history[score_index])
        score_a = a_projection.transform(future_actions[score_index])
        train_raw = raw_trajectory(q0[train_index], future_actions[train_index])
        score_raw = raw_trajectory(q0[score_index], future_actions[score_index])
        for alpha in ALPHAS:
            for gain in RECURRENT_GAINS:
                model, _, _ = fit_recurrent_rounds(
                    train_h,
                    train_a,
                    future_actions[train_index],
                    q0[train_index],
                    measured[train_index],
                    alpha=alpha,
                    gain=gain,
                )
                prediction = rollout_recurrent(
                    model,
                    score_h,
                    score_a,
                    future_actions[score_index],
                    q0[score_index],
                    gain,
                )
                records[(alpha, gain)].append(
                    recurrent_cv_score(prediction, measured[score_index], score_raw)
                )
    summary: dict[str, dict[str, float]] = {}
    for (alpha, gain), folds in records.items():
        key = f"alpha={alpha}:gain={gain}"
        summary[key] = {
            name: float(np.mean([fold[name] for fold in folds])) for name in folds[0]
        }
    selected_alpha, selected_gain = min(
        records,
        key=lambda pair: (summary[f"alpha={pair[0]}:gain={pair[1]}"]["normalized_composite"], pair),
    )
    return float(selected_alpha), float(selected_gain), summary


def hybrid_anchor_raw_delta(absolute: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return transition-local source/target pairs for the frozen hybrid."""

    absolute = np.asarray(absolute, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    if absolute.shape != raw.shape or absolute.ndim != 3 or absolute.shape[1:] != (9, ACTION_DIM):
        raise ValueError(f"hybrid trajectory geometry differs: {absolute.shape}/{raw.shape}")
    source = absolute[:, :-1].copy()
    target = source + np.diff(raw, axis=1)
    return source, target


def _projection_arrays(prefix: str, value: PCAProjection) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_feature_mean": value.feature_mean,
        f"{prefix}_feature_std": value.feature_std,
        f"{prefix}_feature_active": value.feature_active,
        f"{prefix}_pca_mean": value.pca_mean,
        f"{prefix}_pca_components": value.pca_components,
        f"{prefix}_explained_variance_ratio": value.explained_variance_ratio,
    }


def _ridge_arrays(prefix: str, value: RidgeState) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_input_mean": value.input_mean,
        f"{prefix}_input_std": value.input_std,
        f"{prefix}_target_mean": value.target_mean,
        f"{prefix}_target_std": value.target_std,
        f"{prefix}_coef": value.coef,
        f"{prefix}_intercept": value.intercept,
    }


def _trajectory_pairs(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (9, ACTION_DIM):
        raise ValueError(f"trajectory geometry differs: {values.shape}")
    return values[:, :-1].copy(), values[:, 1:].copy()


def _render_pose(values: np.ndarray) -> np.ndarray:
    rendered = np.asarray(values, dtype=np.float32).copy()
    rendered[..., 12:] = np.clip(rendered[..., 12:], 0.0, 1.0)
    return rendered


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    registration_path = output / "registration.json"
    registration = base.read_json(registration_path)
    base.validate_identity(registration, str(registration_path))
    if registration.get("kind") != "trajectory_consistent_renderer_gate_registration":
        raise GateError("unsupported registration kind")
    if base.sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise GateError("execution source differs from registration")
    if (output / "preparation.json").exists():
        raise FileExistsError(output / "preparation.json")

    manifest = Path(registration["inputs"]["train_manifest"]["path"])
    cache_dir = Path(registration["inputs"]["train_cache_metadata"]["path"]).parent
    rows = base.read_jsonl(manifest)
    base.validate_train_manifest(rows)
    actions, _, _ = base.cache_actions(cache_dir)
    fit_indices = list(range(FIT_STOP))
    selected = registration["selection"]["selected"]
    score_indices = [int(item["manifest_index"]) for item in selected]
    donor_indices = [int(item["donor_manifest_index"]) for item in selected]
    if set(fit_indices) & set(score_indices):
        raise GateError("fit and score indices overlap")
    excluded = set(registration["prior_gate0c"]["excluded_clip_ids"])
    if any(rows[index]["clip_id"] in excluded for index in score_indices):
        raise GateError("fresh score selection reuses a Gate-0c clip")

    fit, fit_provenance, fit_boundaries_list, _ = base.extract_rows(
        rows, actions, fit_indices, role="trajectory_predictor_fit"
    )
    score, score_provenance, score_boundaries_list, score_timestamps = base.extract_rows(
        rows, actions, score_indices, role="fresh_renderer_score"
    )
    fit_boundaries = np.stack(fit_boundaries_list).astype(np.float32)
    score_boundaries = np.stack(score_boundaries_list).astype(np.float32)
    fit_q0 = fit_boundaries[:, 4]
    score_q0 = score_boundaries[:, 4]
    fit_measured = fit_boundaries[:, 4:13]
    score_measured = score_boundaries[:, 4:13]

    history_projection, fit_history_context = fit_projection(fit["history"])
    action_projection, fit_action_context = fit_projection(fit["future_actions"])
    score_history_context = history_projection.transform(score["history"])
    score_action_context = action_projection.transform(score["future_actions"])

    absolute_features = base.combine_native_history_with_action(
        fit_history_context, fit_action_context
    )
    absolute_target = fit["target_residual"].reshape(FIT_STOP, -1)
    feature_mean = absolute_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std_raw = absolute_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.where(feature_std_raw > 1e-8, feature_std_raw, 1.0).astype(np.float32)
    target_mean = absolute_target.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std_raw = absolute_target.std(axis=0, dtype=np.float64).astype(np.float32)
    target_std = np.where(target_std_raw > 1e-8, target_std_raw, 1.0).astype(np.float32)
    absolute_alpha, absolute_cv = base.choose_alpha(
        (absolute_features - feature_mean) / feature_std,
        (absolute_target - target_mean) / target_std,
    )
    absolute_model = fit_ridge_state(absolute_features, absolute_target, absolute_alpha)
    aligned_absolute_residual = absolute_model.predict(
        base.combine_native_history_with_action(score_history_context, score_action_context)
    ).reshape(SCORE_CLIPS, 8, ACTION_DIM)

    recurrent_alpha, recurrent_gain, recurrent_cv = choose_recurrent_hyperparameters(
        fit["history"], fit["future_actions"], fit_q0, fit_measured
    )
    recurrent_model, fit_recurrent, recurrent_rounds = fit_recurrent_rounds(
        fit_history_context,
        fit_action_context,
        fit["future_actions"],
        fit_q0,
        fit_measured,
        alpha=recurrent_alpha,
        gain=recurrent_gain,
    )
    aligned_recurrent = rollout_recurrent(
        recurrent_model,
        score_history_context,
        score_action_context,
        score["future_actions"],
        score_q0,
        recurrent_gain,
    )

    position_by_index = {index: position for position, index in enumerate(score_indices)}
    donor_positions = np.asarray(
        [position_by_index[index] for index in donor_indices], dtype=np.int64
    )
    donor_actions = score["future_actions"][donor_positions]
    shuffled_recurrent = rollout_recurrent(
        recurrent_model,
        score_history_context,
        score_action_context[donor_positions],
        donor_actions,
        score_q0,
        recurrent_gain,
    )

    native_raw = raw_trajectory(score_q0, score["future_actions"])
    shuffled_raw = raw_trajectory(score_q0, donor_actions)
    absolute_trajectory = np.concatenate(
        (
            score_q0[:, None],
            score["nominal_endpoint"] + aligned_absolute_residual,
        ),
        axis=1,
    ).astype(np.float32)
    hold = np.broadcast_to(score_q0[:, None], (SCORE_CLIPS, 9, ACTION_DIM)).copy()
    hybrid_source, hybrid_target = hybrid_anchor_raw_delta(absolute_trajectory, native_raw)

    pair_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw_command": _trajectory_pairs(native_raw),
        "raw_episode_shuffled": _trajectory_pairs(shuffled_raw),
        "absolute_ridge": _trajectory_pairs(absolute_trajectory),
        "recurrent_delta": _trajectory_pairs(aligned_recurrent),
        "recurrent_delta_shuffled": _trajectory_pairs(shuffled_recurrent),
        "hybrid_anchor_raw_delta": (hybrid_source, hybrid_target),
        "hold_current": _trajectory_pairs(hold),
        "measured_oracle": _trajectory_pairs(score_measured),
    }
    if tuple(pair_arrays) != ARMS:
        raise GateError("prepared arms differ from frozen order")

    model_path = output / "model_state_and_predictions.npz"
    np.savez_compressed(
        model_path,
        fit_indices=np.asarray(fit_indices, dtype=np.int64),
        score_indices=np.asarray(score_indices, dtype=np.int64),
        donor_indices=np.asarray(donor_indices, dtype=np.int64),
        donor_positions=donor_positions,
        absolute_alpha=np.asarray(absolute_alpha, dtype=np.float64),
        recurrent_alpha=np.asarray(recurrent_alpha, dtype=np.float64),
        recurrent_gain=np.asarray(recurrent_gain, dtype=np.float64),
        score_history_context=score_history_context,
        score_action_context=score_action_context,
        **_projection_arrays("history", history_projection),
        **_projection_arrays("action", action_projection),
        **_ridge_arrays("absolute", absolute_model),
        **_ridge_arrays("recurrent", recurrent_model),
        aligned_absolute_residual=aligned_absolute_residual,
        aligned_recurrent_trajectory=aligned_recurrent,
        shuffled_recurrent_trajectory=shuffled_recurrent,
        target_measured_trajectory=score_measured,
    )

    bundle_dir = output / "bundles"
    bundle_dir.mkdir()
    bundle_records: list[dict[str, Any]] = []
    preprocessed_root = Path(registration["inputs"]["preprocessed_root"])
    raw_root = Path(registration["inputs"]["raw_root"])
    for position, item in enumerate(selected):
        row = rows[score_indices[position]]
        episode = Path(row["episode_dir"])
        video_path = episode / "top.mp4"
        state_path = episode / "states.npz"
        mcap_path = base.raw_mcap_path(row, preprocessed_root, raw_root)
        calibration = _decode_top_calibration(mcap_path)
        frame_indices = np.asarray(row["frame_indices"][4:13], dtype=np.int64)
        rgb = _read_video_frames(video_path, frame_indices.tolist())
        arrays: dict[str, np.ndarray] = {
            "rgb": rgb,
            "frame_indices": frame_indices,
            "frame_ts": score_timestamps[position][4:13],
            "K": np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3),
            "D": np.asarray(calibration["D"], dtype=np.float64),
            "measured_trajectory": score_measured[position],
            "native_future_actions": score["future_actions"][position],
            "donor_future_actions": donor_actions[position],
        }
        for arm, (source, target) in pair_arrays.items():
            arrays[f"pose_unclipped_source_{arm}"] = source[position].astype(np.float32)
            arrays[f"pose_unclipped_target_{arm}"] = target[position].astype(np.float32)
            arrays[f"pose_rendered_source_{arm}"] = _render_pose(source[position])
            arrays[f"pose_rendered_target_{arm}"] = _render_pose(target[position])
        bundle_path = bundle_dir / f"{item['clip_id']}.npz"
        np.savez_compressed(bundle_path, **arrays)
        metadata = base.seal(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "trajectory_consistent_renderer_gate_bundle",
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
                "bundle": base.file_record(bundle_path),
                "source_files": {
                    "states_npz": base.file_record(state_path),
                    "top_mp4": base.file_record(video_path),
                    "raw_mcap": base.file_record(mcap_path),
                },
                "gripper_render_clip": "coordinates 12:14 clipped to [0,1]; unclipped retained",
                "hybrid_is_transition_local": True,
            }
        )
        metadata_path = bundle_dir / f"{item['clip_id']}.json"
        base.write_json(metadata_path, metadata)
        bundle_records.append(
            {
                "clip_id": item["clip_id"],
                "bundle": base.file_record(bundle_path),
                "metadata": base.file_record(metadata_path),
                "metadata_identity_sha256": metadata["identity_sha256"],
                "protected_test_accessed": False,
            }
        )

    provenance_path = output / "input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for record in [*fit_provenance, *score_provenance]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    state_diagnostics: dict[str, dict[str, Any]] = {}
    oracle_delta = np.diff(score_measured, axis=1)
    for arm, (source, target) in pair_arrays.items():
        pose_mse = np.mean((target - score_measured[:, 1:]) ** 2, axis=(1, 2))
        delta_mse = np.mean((target[..., :12] - source[..., :12] - oracle_delta[..., :12]) ** 2, axis=(1, 2))
        state_diagnostics[arm] = {
            "target_pose_mse_mean": float(pose_mse.mean()),
            "joint_transition_mse_mean": float(delta_mse.mean()),
            "target_pose_mse_per_clip": pose_mse.tolist(),
            "joint_transition_mse_per_clip": delta_mse.tolist(),
        }
    fit_raw = raw_trajectory(fit_q0, fit["future_actions"])
    preparation = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "trajectory_consistent_renderer_gate_preparation",
            "created_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "status": "prepared_after_frozen_registration_without_renderer_metric_inspection",
            "split": "train",
            "fit_clips": FIT_STOP,
            "score_clips": SCORE_CLIPS,
            "fit_score_episode_disjoint": True,
            "future_measured_state_predictor_input": False,
            "models": {
                "absolute_ridge": {
                    "selected_alpha": absolute_alpha,
                    "five_fold_fit384_cv_standardized_mse": absolute_cv,
                },
                "recurrent_delta": {
                    "selected_alpha": recurrent_alpha,
                    "selected_gain": recurrent_gain,
                    "self_rollout_rounds": SELF_ROLLOUT_ROUNDS,
                    "five_fold_fit384_cv": recurrent_cv,
                    "full_fit_round_diagnostics": recurrent_rounds,
                    "full_fit_score": recurrent_cv_score(fit_recurrent, fit_measured, fit_raw),
                },
                "history_pca_variance_retained": float(
                    history_projection.explained_variance_ratio.sum()
                ),
                "action_pca_variance_retained": float(
                    action_projection.explained_variance_ratio.sum()
                ),
            },
            "state_space_diagnostic_only": state_diagnostics,
            "bundles": bundle_records,
            "artifacts": {
                "registration": base.file_record(registration_path),
                "source": base.file_record(Path(__file__).resolve()),
                "model_state_and_predictions": base.file_record(model_path),
                "input_provenance": base.file_record(provenance_path),
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "preparation.json", preparation)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "trajectory_consistent_renderer_gate_preparation_complete",
            "completed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "artifacts": {
                "preparation": base.file_record(output / "preparation.json"),
                **preparation["artifacts"],
            },
            "status": "completed",
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "preparation_complete.json", complete)
    return preparation


def load_bundles(output: Path, preparation: dict[str, Any]) -> list[base.LoadedBundle]:
    bundles: list[base.LoadedBundle] = []
    for expected_order, record in enumerate(preparation["bundles"]):
        metadata_path = Path(record["metadata"]["path"])
        bundle_path = Path(record["bundle"]["path"])
        if not metadata_path.is_file():
            metadata_path = output / "bundles" / metadata_path.name
        if not bundle_path.is_file():
            bundle_path = output / "bundles" / bundle_path.name
        metadata = base.read_json(metadata_path)
        base.validate_identity(metadata, str(metadata_path))
        if metadata.get("score_order") != expected_order or metadata.get("split") != "train":
            raise GateError("bundle order or split differs")
        if metadata.get("protected_test_accessed") is not False:
            raise GateError("bundle protected-test flag differs")
        if base.sha256_file(bundle_path) != metadata["bundle"]["sha256"]:
            raise GateError(f"bundle hash mismatch: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        required = {"rgb", "K", "D", "frame_indices", "frame_ts"}
        for arm in ARMS:
            required.update(
                {
                    f"pose_rendered_source_{arm}",
                    f"pose_rendered_target_{arm}",
                    f"pose_unclipped_source_{arm}",
                    f"pose_unclipped_target_{arm}",
                }
            )
        if not required.issubset(arrays) or arrays["rgb"].shape[0] != 9:
            raise GateError(f"bundle arrays differ: {bundle_path}")
        for arm in ARMS:
            if arrays[f"pose_rendered_source_{arm}"].shape != (8, ACTION_DIM):
                raise GateError(f"source pose geometry differs: {arm}")
            if arrays[f"pose_rendered_target_{arm}"].shape != (8, ACTION_DIM):
                raise GateError(f"target pose geometry differs: {arm}")
        bundles.append(base.LoadedBundle(metadata=metadata, arrays=arrays))
    if len(bundles) != SCORE_CLIPS:
        raise GateError(f"loaded {len(bundles)} bundles, expected {SCORE_CLIPS}")
    return bundles


def aggregate_clip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for clip_id in sorted({str(row["clip_id"]) for row in rows}):
        clip_rows = [row for row in rows if row["clip_id"] == clip_id]
        for arm in ARMS:
            arm_rows = [row for row in clip_rows if row["arm"] == arm]
            if len(arm_rows) != 8:
                raise GateError(f"{clip_id}/{arm} has {len(arm_rows)} frame rows")
            flow_count = float(
                np.sum([row["metrics"]["oracle_moving_flow_sample_count"] for row in arm_rows])
            )
            if flow_count < 10:
                raise GateError(f"{clip_id}/{arm} has insufficient robot-flow support")
            metrics: dict[str, float] = {}
            for name in sorted(arm_rows[0]["metrics"]):
                if name == "robot_flow_epe_px":
                    metrics[name] = float(
                        np.sum([row["metrics"]["robot_flow_epe_sum_px"] for row in arm_rows])
                        / flow_count
                    )
                    continue
                values = [row["metrics"][name] for row in arm_rows]
                numeric = [float(value) for value in values if value is not None]
                if not numeric:
                    continue
                if name in {"robot_flow_epe_sum_px", "oracle_moving_flow_sample_count"}:
                    metrics[name] = float(np.sum(numeric))
                elif name == "flow_transition_scored":
                    metrics[name] = float(np.sum(numeric))
                else:
                    metrics[name] = float(np.mean(numeric))
            aggregates.append(
                {
                    "clip_id": clip_id,
                    "arm": arm,
                    "metrics": metrics,
                    "protected_test_accessed": False,
                }
            )
    return aggregates


def bootstrap_indices() -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, SCORE_CLIPS, size=(BOOTSTRAP_SAMPLES, SCORE_CLIPS))


def paired_effect(
    reference: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    *,
    higher_is_better: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != (SCORE_CLIPS,) or candidate.shape != (SCORE_CLIPS,):
        raise ValueError(f"paired metric shape differs: {reference.shape}/{candidate.shape}")
    difference = candidate - reference if higher_is_better else reference - candidate
    samples = difference[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    denominator = float(reference.mean())
    relative = None
    if abs(denominator) > 1e-12:
        relative = 100.0 * float(difference.mean()) / abs(denominator)
    effect = {
        "reference_mean": denominator,
        "candidate_mean": float(candidate.mean()),
        "paired_favorable_difference_mean": float(difference.mean()),
        "paired_bootstrap_95_ci": [float(low), float(high)],
        "one_sided_bootstrap_p": float((1 + np.sum(samples <= 0.0)) / (len(samples) + 1)),
        "relative_improvement_percent": relative,
        "favorable_clip_fraction": float(np.mean(difference > 0.0)),
        "favorable_clips": int(np.sum(difference > 0.0)),
        "paired_clips": SCORE_CLIPS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "common_bootstrap_indices": True,
        "higher_is_better": higher_is_better,
    }
    return effect, samples


def effect_specs() -> list[tuple[str, str, str, bool]]:
    specs: list[tuple[str, str, str, bool]] = []
    primary = (
        ("recurrent_delta", "raw_command"),
        ("recurrent_delta", "absolute_ridge"),
        ("recurrent_delta", "recurrent_delta_shuffled"),
        ("hybrid_anchor_raw_delta", "raw_command"),
        ("hybrid_anchor_raw_delta", "absolute_ridge"),
        ("raw_command", "raw_episode_shuffled"),
        ("raw_command", "hold_current"),
    )
    for candidate, reference in primary:
        specs.append((candidate, reference, "robot_flow_epe_px", False))
    for candidate in ("recurrent_delta", "hybrid_anchor_raw_delta"):
        for reference in ("raw_command", "absolute_ridge"):
            specs.extend(
                (
                    (candidate, reference, "silhouette_boundary_chamfer_px", False),
                    (candidate, reference, "rgb_robot_band_chamfer_px", False),
                    (candidate, reference, "silhouette_iou", True),
                )
            )
    for reference in ("raw_episode_shuffled", "hold_current"):
        specs.extend(
            (
                ("raw_command", reference, "silhouette_boundary_chamfer_px", False),
                ("raw_command", reference, "rgb_robot_band_chamfer_px", False),
                ("raw_command", reference, "silhouette_iou", True),
            )
        )
    return specs


def apply_holm(
    effects: dict[str, dict[str, Any]], samples: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    ordered_labels = sorted(
        PRIMARY_FLOW_LABELS,
        key=lambda label: (effects[label]["one_sided_bootstrap_p"], label),
    )
    active = True
    rows: list[dict[str, Any]] = []
    family_size = len(ordered_labels)
    for rank_zero, label in enumerate(ordered_labels):
        threshold = HOLM_ALPHA / (family_size - rank_zero)
        p_value = float(effects[label]["one_sided_bootstrap_p"])
        rejected = bool(active and p_value <= threshold)
        if not rejected:
            active = False
        lower_bound = float(np.quantile(samples[label], threshold))
        effects[label]["holm_rank"] = rank_zero + 1
        effects[label]["holm_alpha_threshold"] = threshold
        effects[label]["holm_stepdown_rejected"] = rejected
        effects[label]["holm_one_sided_lower_bound"] = lower_bound
        rows.append(
            {
                "rank": rank_zero + 1,
                "label": label,
                "one_sided_bootstrap_p": p_value,
                "alpha_threshold": threshold,
                "stepdown_rejected": rejected,
                "one_sided_lower_bound": lower_bound,
            }
        )
    return rows


def analyze_aggregates(
    aggregates: list[dict[str, Any]], clip_order: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]], dict[str, Any], dict[str, bool]]:
    if len(clip_order) != SCORE_CLIPS or len(set(clip_order)) != SCORE_CLIPS:
        raise GateError("analysis clip order differs")
    by_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        arm_rows = {row["clip_id"]: row for row in aggregates if row["arm"] == arm}
        if set(arm_rows) != set(clip_order):
            raise GateError(f"aggregate clips differ for {arm}")
        first = arm_rows[clip_order[0]]["metrics"]
        by_arm[arm] = {
            metric: np.asarray([arm_rows[clip]["metrics"][metric] for clip in clip_order])
            for metric in first
        }
    indices = bootstrap_indices()
    effects: dict[str, dict[str, Any]] = {}
    samples: dict[str, np.ndarray] = {}
    for candidate, reference, metric, higher in effect_specs():
        label = f"{candidate}_vs_{reference}:{metric}"
        effects[label], samples[label] = paired_effect(
            by_arm[reference][metric],
            by_arm[candidate][metric],
            indices,
            higher_is_better=higher,
        )
    holm_order = apply_holm(effects, samples)

    primary_pass: dict[str, bool] = {}
    no_margin = {"recurrent_delta_vs_recurrent_delta_shuffled:robot_flow_epe_px"}
    for label in PRIMARY_FLOW_LABELS:
        effect = effects[label]
        primary_pass[label] = bool(
            effect["paired_favorable_difference_mean"] > 0.0
            and effect["holm_stepdown_rejected"]
            and effect["holm_one_sided_lower_bound"] > 0.0
            and effect["favorable_clip_fraction"] >= MIN_FAVORABLE_CLIP_FRACTION
            and (
                label in no_margin
                or effect["relative_improvement_percent"]
                >= FLOW_MIN_RELATIVE_IMPROVEMENT_PERCENT
            )
        )

    retention_pass: dict[str, bool] = {}
    for candidate in ("recurrent_delta", "hybrid_anchor_raw_delta"):
        for reference in ("raw_command", "absolute_ridge"):
            for metric in (
                "silhouette_boundary_chamfer_px",
                "rgb_robot_band_chamfer_px",
                "silhouette_iou",
            ):
                label = f"{candidate}_vs_{reference}:{metric}"
                effect = effects[label]
                retention_pass[label] = bool(
                    effect["paired_favorable_difference_mean"] >= 0.0
                    and effect["paired_bootstrap_95_ci"][0] >= 0.0
                )
    fallback_spatial_pass: dict[str, bool] = {}
    for reference in ("raw_episode_shuffled", "hold_current"):
        for metric in (
            "silhouette_boundary_chamfer_px",
            "rgb_robot_band_chamfer_px",
            "silhouette_iou",
        ):
            label = f"raw_command_vs_{reference}:{metric}"
            effect = effects[label]
            fallback_spatial_pass[label] = bool(
                effect["paired_favorable_difference_mean"] > 0.0
                and effect["paired_bootstrap_95_ci"][0]
                >= (0.0 if metric == "silhouette_iou" else np.nextafter(0.0, 1.0))
            )

    recurrent_labels = [label for label in primary_pass if label.startswith("recurrent_delta_")]
    hybrid_labels = [label for label in primary_pass if label.startswith("hybrid_anchor_")]
    raw_labels = [label for label in primary_pass if label.startswith("raw_command_")]
    family_gates = {
        "recurrent_delta_pass": bool(
            all(primary_pass[label] for label in recurrent_labels)
            and all(
                value
                for label, value in retention_pass.items()
                if label.startswith("recurrent_delta_")
            )
        ),
        "hybrid_anchor_raw_delta_pass": bool(
            all(primary_pass[label] for label in hybrid_labels)
            and all(
                value
                for label, value in retention_pass.items()
                if label.startswith("hybrid_anchor_")
            )
        ),
        "raw_geometry_scaffold_pass": bool(
            all(primary_pass[label] for label in raw_labels)
            and all(fallback_spatial_pass.values())
        ),
    }
    gates: dict[str, Any] = {
        "primary_flow": primary_pass,
        "candidate_spatial_retention": retention_pass,
        "raw_fallback_spatial": fallback_spatial_pass,
        "families": family_gates,
        "families_cannot_rescue_each_other": True,
    }
    metrics_mean = {
        arm: {metric: float(values.mean()) for metric, values in metrics.items()}
        for arm, metrics in by_arm.items()
    }
    return metrics_mean, effects, {"order": holm_order, "family_size": 7}, family_gates | {"details": gates}


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
    registration = base.read_json(output / "registration.json")
    preparation = base.read_json(output / "preparation.json")
    base.validate_identity(registration, "registration")
    base.validate_identity(preparation, "preparation")
    if preparation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise GateError("preparation-registration binding differs")
    if base.sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise GateError("evaluation source differs from registration")
    bundles = load_bundles(output, preparation)

    official_root = args.official_abc_root.resolve()
    official_commit = base.git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise GateError(f"official ABC must be {EXPECTED_ABC_COMMIT}, got {official_commit}")
    scene_path = official_root / OFFICIAL_SCENE_RELATIVE
    asset_root = official_root / OFFICIAL_ASSET_RELATIVE
    robot_xml = build_robot_only_xml(scene_path, asset_root)
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    if camera_id < 0:
        raise GateError("official model lacks top camera")
    moving_geoms = base.moving_geom_ids(model, mujoco)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    far_depth = float(model.stat.extent * model.vis.map.zfar)

    rows: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    steady_latencies: list[float] = []
    overlay_dir = output / "overlays"
    overlay_dir.mkdir()
    overlay_records: list[dict[str, Any]] = []
    renderer = None
    try:
        for bundle in bundles:
            arrays = bundle.arrays
            rgb = arrays["rgb"]
            _, height, width, channels = rgb.shape
            if channels != 3:
                raise GateError("RGB channel geometry differs")
            if renderer is not None:
                renderer.close()
            renderer = mujoco.Renderer(model, height=height, width=width)
            fy = float(arrays["K"][1, 1])
            model.cam_fovy[camera_id] = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
            cache: dict[bytes, base.PoseRender] = {}

            def rendered(pose: np.ndarray) -> base.PoseRender:
                key = np.asarray(pose, dtype=np.float32).tobytes()
                if key not in cache:
                    cache[key] = base.render_pose(
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
                    render_latencies.append(cache[key].render_ms)
                return cache[key]

            per_arm: dict[str, tuple[list[base.PoseRender], list[base.PoseRender]]] = {}
            clip_latency_start = len(render_latencies)
            for arm in ARMS:
                sources = [rendered(pose) for pose in arrays[f"pose_rendered_source_{arm}"]]
                targets = [rendered(pose) for pose in arrays[f"pose_rendered_target_{arm}"]]
                per_arm[arm] = (sources, targets)
            clip_latencies = render_latencies[clip_latency_start:]
            steady_latencies.extend(clip_latencies[2:] if len(clip_latencies) > 2 else clip_latencies)

            clip_rows: list[dict[str, Any]] = []
            for step in range(8):
                oracle_source = per_arm["measured_oracle"][0][step]
                oracle_target = per_arm["measured_oracle"][1][step]
                edges = observed_edges(rgb[step + 1])
                for arm in ARMS:
                    candidate_source = per_arm[arm][0][step]
                    candidate_target = per_arm[arm][1][step]
                    metrics = {
                        "silhouette_boundary_chamfer_px": base.symmetric_boundary_chamfer(
                            candidate_target.mask, oracle_target.mask
                        ),
                        "silhouette_iou": base.mask_iou(candidate_target.mask, oracle_target.mask),
                        **base.rgb_robot_band_metrics(
                            candidate_target.mask, oracle_target.mask, edges
                        ),
                        **base.robot_flow_error(
                            candidate_source,
                            candidate_target,
                            oracle_source,
                            oracle_target,
                            far_depth=far_depth,
                        ),
                    }
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "clip_id": bundle.metadata["clip_id"],
                        "score_order": bundle.metadata["score_order"],
                        "motion_stratum": bundle.metadata["motion_stratum"],
                        "transition_ordinal": step + 1,
                        "source_frame_index": int(arrays["frame_indices"][step]),
                        "target_frame_index": int(arrays["frame_indices"][step + 1]),
                        "arm": arm,
                        "metrics": metrics,
                        "split": "train",
                        "validation_accessed": False,
                        "protected_test_accessed": False,
                    }
                    rows.append(row)
                    clip_rows.append(row)

            for candidate_arm in ("recurrent_delta", "hybrid_anchor_raw_delta"):
                candidate_rows = [row for row in clip_rows if row["arm"] == candidate_arm]
                worst = max(
                    candidate_rows,
                    key=lambda row: row["metrics"]["rgb_robot_band_chamfer_px"],
                )
                step = int(worst["transition_ordinal"] - 1)
                overlay_path = overlay_dir / (
                    f"worst_{candidate_arm}_{bundle.metadata['clip_id'][:12]}.png"
                )
                base._overlay_boundaries(
                    overlay_path,
                    rgb[step + 1],
                    per_arm["raw_command"][1][step].mask,
                    per_arm[candidate_arm][1][step].mask,
                    per_arm["measured_oracle"][1][step].mask,
                    f"raw=red {candidate_arm}=green oracle=blue transition={step + 1}",
                )
                overlay_records.append(
                    {
                        "clip_id": bundle.metadata["clip_id"],
                        "candidate_arm": candidate_arm,
                        "worst_rgb_robot_band_chamfer_px": worst["metrics"][
                            "rgb_robot_band_chamfer_px"
                        ],
                        "file": base.file_record(overlay_path),
                        "protected_test_accessed": False,
                    }
                )
    finally:
        if renderer is not None:
            renderer.close()

    frame_path = output / "frame_transition_metrics.jsonl"
    with frame_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    aggregates = aggregate_clip_rows(rows)
    clip_path = output / "clip_metrics.jsonl"
    with clip_path.open("w") as handle:
        for row in aggregates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    clip_order = [bundle.metadata["clip_id"] for bundle in bundles]
    metrics_by_arm, effects, holm, gate_payload = analyze_aggregates(aggregates, clip_order)
    family_gates = {name: value for name, value in gate_payload.items() if name != "details"}
    passed = [name for name, value in family_gates.items() if value]
    decision = "GO_" + "+".join(passed) if passed else "STOP_ALL_GEOMETRY_HANDOFFS"
    latency = np.asarray(steady_latencies, dtype=np.float64)
    analysis = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "trajectory_consistent_renderer_gate_analysis",
            "created_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "decision": decision,
            "passed_families": passed,
            "family_gates": family_gates,
            "gate_details": gate_payload["details"],
            "split": "train",
            "fit_clips": FIT_STOP,
            "score_clips": SCORE_CLIPS,
            "score_transitions": SCORE_CLIPS * 8,
            "fresh_vs_gate0c": True,
            "fit_score_episode_disjoint": True,
            "validation_accessed": False,
            "protected_test_accessed": False,
            "metrics_by_arm": metrics_by_arm,
            "effects": effects,
            "holm_primary_flow_family": holm,
            "flow_contract": {
                "type": "geom-local point transport on oracle-visible articulated support",
                "pixel_stride": FLOW_PIXEL_STRIDE,
                "minimum_oracle_motion_px": FLOW_MIN_ORACLE_MOTION_PX,
                "minimum_clip_pixels": 10,
                "zero_support_transition_policy": "exclude symmetrically; pool by qualifying oracle pixels",
                "far_depth": far_depth,
                "moving_geom_ids": sorted(moving_geoms),
                "hybrid_transition_local": True,
            },
            "camera": {
                "official_abc_commit": official_commit,
                "official_scene": base.file_record(scene_path),
                "robot_only_xml_sha256": hashlib.sha256(robot_xml.encode()).hexdigest(),
                "nominal_extrinsic": True,
                "recorded_fy_as_fovy": True,
                "principal_point_centered": True,
                "distortion_applied": False,
            },
            "latency": {
                "deduplicated_render_pose_mean_ms": float(latency.mean()),
                "deduplicated_render_pose_p50_ms": float(np.percentile(latency, 50)),
                "deduplicated_render_pose_p95_ms": float(np.percentile(latency, 95)),
                "pose_count_after_two_warmups_per_clip": int(len(latency)),
                "native_resolution_renderer_recreated_per_clip": True,
            },
            "artifacts": {
                "registration": base.file_record(output / "registration.json"),
                "preparation": base.file_record(output / "preparation.json"),
                "preparation_complete": base.file_record(output / "preparation_complete.json"),
                "source": base.file_record(Path(__file__).resolve()),
                "frame_transition_metrics": base.file_record(frame_path),
                "clip_metrics": base.file_record(clip_path),
                "overlays": overlay_records,
            },
            "claim_boundary": (
                "Fresh train-only robot-renderer attribution. Future measured state and RGB are scoring-only. "
                "No generated video, object/contact flow, validation, or protected test is evaluated."
            ),
            "limitations": [
                "24 motion-stratified fresh train episodes and one fixed seed",
                "raw command endpoints are not controller simulation",
                "hybrid is transition-local rather than a coherent recurrent rollout",
                "nominal camera extrinsics, centered principal point, and no distortion",
                "robot-only rendering excludes object/contact dynamics and scene occlusion",
                "a pass authorizes only a controlled Wan conditioning screen",
            ],
        }
    )
    base.write_json(output / "analysis.json", analysis)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "trajectory_consistent_renderer_gate_complete",
            "completed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": decision,
            "status": "completed",
            "artifacts": {"analysis": base.file_record(output / "analysis.json"), **analysis["artifacts"]},
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "run_complete.json", complete)
    return analysis


def _resolve_artifact(output: Path, record: dict[str, Any], subdir: str | None = None) -> Path:
    path = Path(record["path"])
    if path.is_file():
        return path
    candidate = output / (subdir or "") / path.name
    if not candidate.is_file():
        raise FileNotFoundError(path)
    return candidate


def audit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    names = (
        "registration.json",
        "preparation.json",
        "preparation_complete.json",
        "analysis.json",
        "run_complete.json",
    )
    documents = {name: base.read_json(output / name) for name in names}
    for name, document in documents.items():
        base.validate_identity(document, name)
    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    analysis = documents["analysis.json"]
    complete = documents["run_complete.json"]
    if preparation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise GateError("preparation-registration binding differs")
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise GateError("analysis-registration binding differs")
    if analysis["preparation_identity_sha256"] != preparation["identity_sha256"]:
        raise GateError("analysis-preparation binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise GateError("completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise GateError("completion decision differs")
    if base.sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise GateError("current source differs from registered source")

    prior_record = registration["prior_gate0c"]["registration"]
    prior_path = _resolve_artifact(output, prior_record)
    if base.sha256_file(prior_path) != prior_record["sha256"]:
        raise GateError("prior Gate-0c registration hash differs")
    prior = base.read_json(prior_path)
    base.validate_identity(prior, str(prior_path))
    if prior["identity_sha256"] != registration["prior_gate0c"]["registration_identity_sha256"]:
        raise GateError("prior Gate-0c identity differs")
    excluded = set(registration["prior_gate0c"]["excluded_clip_ids"])
    selected = registration["selection"]["selected"]
    selected_ids = [item["clip_id"] for item in selected]
    selected_indices = [int(item["manifest_index"]) for item in selected]
    if len(selected_ids) != SCORE_CLIPS or len(set(selected_ids)) != SCORE_CLIPS:
        raise GateError("fresh selected count differs")
    if set(selected_ids) & excluded:
        raise GateError("fresh selection overlaps Gate-0c")
    if any(index < SCORE_START or index >= SCORE_STOP for index in selected_indices):
        raise GateError("fresh selected index is outside score pool")
    by_index = {int(item["manifest_index"]): item for item in selected}
    for item in selected:
        donor = by_index.get(int(item["donor_manifest_index"]))
        if donor is None:
            raise GateError("donor is outside selected fresh24")
        if donor["clip_id"] == item["clip_id"] or donor["episode_dir"] == item["episode_dir"]:
            raise GateError("donor equals recipient")
        if donor["motion_stratum"] != item["motion_stratum"]:
            raise GateError("donor motion stratum differs")

    hash_count = 0
    for section in (preparation["artifacts"], analysis["artifacts"], complete["artifacts"]):
        for name, record in section.items():
            if name == "overlays":
                for item in record:
                    path = _resolve_artifact(output, item["file"], "overlays")
                    if base.sha256_file(path) != item["file"]["sha256"]:
                        raise GateError(f"overlay hash differs: {path}")
                    hash_count += 1
                continue
            path = _resolve_artifact(output, record)
            if base.sha256_file(path) != record["sha256"]:
                raise GateError(f"artifact hash differs: {name}")
            hash_count += 1

    bundles = load_bundles(output, preparation)
    frame_rows = base.read_jsonl(output / "frame_transition_metrics.jsonl")
    clip_rows = base.read_jsonl(output / "clip_metrics.jsonl")
    provenance_rows = base.read_jsonl(output / "input_provenance.jsonl")
    if len(frame_rows) != SCORE_CLIPS * 8 * len(ARMS):
        raise GateError(f"frame row count differs: {len(frame_rows)}")
    if len(clip_rows) != SCORE_CLIPS * len(ARMS):
        raise GateError(f"clip row count differs: {len(clip_rows)}")
    if len(provenance_rows) != FIT_STOP + SCORE_CLIPS:
        raise GateError(f"provenance row count differs: {len(provenance_rows)}")
    for row in frame_rows:
        if row["metrics"]["oracle_source_reprojection_rmse_px"] > 1e-5:
            raise GateError("flow source reprojection audit failed")
    for row in clip_rows:
        if row["metrics"]["oracle_moving_flow_sample_count"] < 10:
            raise GateError("clip flow support is below frozen minimum")

    clip_order = [bundle.metadata["clip_id"] for bundle in bundles]
    metrics, effects, holm, gate_payload = analyze_aggregates(clip_rows, clip_order)
    family_gates = {name: value for name, value in gate_payload.items() if name != "details"}
    passed = [name for name, value in family_gates.items() if value]
    decision = "GO_" + "+".join(passed) if passed else "STOP_ALL_GEOMETRY_HANDOFFS"
    if metrics != analysis["metrics_by_arm"]:
        raise GateError("recomputed arm metrics differ")
    if effects != analysis["effects"]:
        raise GateError("recomputed paired effects differ")
    if holm != analysis["holm_primary_flow_family"]:
        raise GateError("recomputed Holm family differs")
    if family_gates != analysis["family_gates"] or gate_payload["details"] != analysis["gate_details"]:
        raise GateError("recomputed gate payload differs")
    if decision != analysis["decision"] or passed != analysis["passed_families"]:
        raise GateError("recomputed decision differs")

    model_record = preparation["artifacts"]["model_state_and_predictions"]
    model_path = _resolve_artifact(output, model_record)
    with np.load(model_path, allow_pickle=False) as payload:
        recurrent = RidgeState(
            input_mean=payload["recurrent_input_mean"],
            input_std=payload["recurrent_input_std"],
            target_mean=payload["recurrent_target_mean"],
            target_std=payload["recurrent_target_std"],
            coef=payload["recurrent_coef"],
            intercept=payload["recurrent_intercept"],
        )
        score_h = payload["score_history_context"]
        score_a = payload["score_action_context"]
        donor_positions = payload["donor_positions"].astype(np.int64)
        gain = float(payload["recurrent_gain"])
        stored_aligned = payload["aligned_recurrent_trajectory"]
        stored_shuffled = payload["shuffled_recurrent_trajectory"]
    q0 = np.stack([bundle.arrays["measured_trajectory"][0] for bundle in bundles])
    native_actions = np.stack([bundle.arrays["native_future_actions"] for bundle in bundles])
    donor_actions = np.stack([bundle.arrays["donor_future_actions"] for bundle in bundles])
    replay_aligned = rollout_recurrent(recurrent, score_h, score_a, native_actions, q0, gain)
    replay_shuffled = rollout_recurrent(
        recurrent, score_h, score_a[donor_positions], donor_actions, q0, gain
    )
    replay_error = float(
        max(
            np.max(np.abs(replay_aligned - stored_aligned)),
            np.max(np.abs(replay_shuffled - stored_shuffled)),
        )
    )
    if replay_error > 5e-6:
        raise GateError(f"causal recurrent prediction replay differs: {replay_error}")

    false_flags = sum(base.require_false_flags(document, name) for name, document in documents.items())
    false_flags += sum(
        base.require_false_flags(values, name)
        for name, values in (
            ("frame_rows", frame_rows),
            ("clip_rows", clip_rows),
            ("provenance_rows", provenance_rows),
        )
    )
    return {
        "status": "audit_passed",
        "decision": decision,
        "registration_identity_sha256": registration["identity_sha256"],
        "preparation_identity_sha256": preparation["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "completion_identity_sha256": complete["identity_sha256"],
        "artifact_hashes_verified": hash_count,
        "bundles": len(bundles),
        "frame_rows": len(frame_rows),
        "clip_rows": len(clip_rows),
        "provenance_rows": len(provenance_rows),
        "effects_recomputed": len(effects),
        "holm_primary_tests_recomputed": len(PRIMARY_FLOW_LABELS),
        "causal_prediction_replay_max_abs_error": replay_error,
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
    register_parser.add_argument("--prior-registration", type=Path, required=True)
    register_parser.add_argument("--source-commit", required=True)
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
