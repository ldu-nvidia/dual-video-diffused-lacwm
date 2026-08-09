#!/usr/bin/env python3
"""Prospective distributional object/contact uncertainty qualification.

The deterministic slot predictor has already stopped.  This sequential study
uses its fit256 episodes for conditional-kernel fitting and its opened fresh64
episodes only for bandwidth calibration.  All remaining 31 untouched D405
episodes are sealed before their RGB/state targets are opened and are used once
for proper-score comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import interaction_event_bottleneck_stage0 as base
from tools import object_contact_slot_stage0 as slot
from tools.abc_d405_nominal_geometry_probe import (
    EXPECTED_ABC_COMMIT,
    OFFICIAL_ASSET_RELATIVE,
    OFFICIAL_SCENE_RELATIVE,
    _joint_qpos_addresses,
    build_robot_only_xml,
)
from tools.corrected_renderer_attribution import moving_geom_ids


SCHEMA_VERSION = 1
PARENT_REGISTRATION_IDENTITY = (
    "1acfdb4222bac5057a2ccf774cff626f95d9b14fc6a0cde5938bca57c5ff7125"
)
PARENT_ANALYSIS_IDENTITY = (
    "497e0cd68eaaa7b2cdaf5a1eb0eaad3e91d82360ad06bb3772961cf7115bc619"
)
PARENT_COMPLETION_IDENTITY = (
    "6250a77dc921939764d5c02b504c96157847c72c74601eebf7afdfc4bf25c258"
)

FIT_CLIPS = 256
CALIBRATION_CLIPS = 64
FINAL_CLIPS = 31
TARGET_DIM = 7
TARGET_FEATURES = (
    "log_total_event_mass",
    "event_time_centroid",
    "event_centroid_x",
    "event_centroid_y",
    "signed_motion_u",
    "signed_motion_v",
    "peak_contact_score",
)
ACTION_COMPONENTS = 8
MIXTURE_COMPONENTS = 32
ACTION_DISTANCE_WEIGHT = 1.0
MASS_LOG_SCALE = 1000.0
SIGMA_GRID = (0.20, 0.30, 0.45, 0.65, 0.90, 1.25, 1.75)
CALIBRATION_COVERAGE_80_RANGE = (0.70, 0.90)
CALIBRATION_COVERAGE_95_RANGE = (0.87, 1.00)
MODEL_SEED = 97531
ENERGY_SEED = 20260816
ENERGY_SAMPLES = 256
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_SAMPLES = 20_000

ARMS = (
    "history_only",
    "aligned",
    "episode_shuffled",
    "action_stratum_centroid",
)
REFERENCES = (
    "history_only",
    "episode_shuffled",
    "action_stratum_centroid",
)
PROPER_METRICS = ("nll", "energy", "crps", "wis")
PRIMARY_FAMILY_SIZE = len(REFERENCES) * len(PROPER_METRICS)
MIN_NLL_IMPROVEMENT_NATS_PER_DIM = 0.05
MIN_RELATIVE_IMPROVEMENT_PERCENT = 5.0
MIN_FAVORABLE_FRACTION = 0.60
FINAL_COVERAGE_80_RANGE = (0.70, 0.90)
FINAL_COVERAGE_95_RANGE = (0.87, 1.00)
MIN_FINAL_NONZERO_FRACTION = 0.50


class DistributionError(RuntimeError):
    """Raised when a frozen distributional-study invariant is violated."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_summary(states: np.ndarray) -> np.ndarray:
    """Permutation-invariant seven-scalar summary of a slot sequence."""

    values = np.asarray(states, dtype=np.float64)
    single = values.ndim == 3
    if single:
        values = values[None]
    if values.ndim != 4 or values.shape[2:] != (slot.SLOT_COUNT, slot.SLOT_DIM):
        raise ValueError(f"slot summary geometry differs: {values.shape}")
    batches, transitions = values.shape[:2]
    mass = np.clip(values[..., 4], 0.0, None) + np.clip(
        values[..., 5], 0.0, None
    )
    mass *= np.clip(values[..., slot.PRESENCE_INDEX], 0.0, 1.0)
    total = mass.sum(axis=(1, 2))
    safe = np.where(total > 1e-12, total, 1.0)
    time_axis = np.linspace(-1.0, 1.0, transitions, dtype=np.float64)[None, :, None]
    result = np.zeros((batches, TARGET_DIM), dtype=np.float64)
    result[:, 0] = np.log1p(MASS_LOG_SCALE * total)
    result[:, 1] = np.sum(mass * time_axis, axis=(1, 2)) / safe
    result[:, 2] = np.sum(mass * values[..., 1], axis=(1, 2)) / safe
    result[:, 3] = np.sum(mass * values[..., 2], axis=(1, 2)) / safe
    result[:, 4] = np.sum(mass * values[..., 9], axis=(1, 2)) / safe
    result[:, 5] = np.sum(mass * values[..., 10], axis=(1, 2)) / safe
    contact = np.clip(values[..., 11], 0.0, 1.0) * np.clip(
        values[..., slot.PRESENCE_INDEX], 0.0, 1.0
    )
    result[:, 6] = np.max(contact, axis=(1, 2))
    result[total <= 1e-12] = 0.0
    if not np.isfinite(result).all():
        raise DistributionError("compact target contains non-finite values")
    output = result.astype(np.float32)
    return output[0] if single else output


def _fit_action_pca8(
    values: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    raw_mean, raw_std = base._standardizer(flat)
    active = flat.std(axis=0, dtype=np.float64) > 1e-8
    if int(active.sum()) < ACTION_COMPONENTS:
        raise DistributionError("too few active action coordinates")
    normalized = ((flat - raw_mean) / raw_std)[:, active]
    pca = PCA(n_components=ACTION_COMPONENTS, svd_solver="full")
    pc = pca.fit_transform(normalized).astype(np.float32)
    pc_mean, pc_std = base._standardizer(pc)
    return pca, (pc - pc_mean) / pc_std, raw_mean, raw_std, active, pc_mean, pc_std


def _transform_action_pca8(
    pca: Any,
    values: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    active: np.ndarray,
    pc_mean: np.ndarray,
    pc_std: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    pc = pca.transform(((flat - raw_mean) / raw_std)[:, active]).astype(np.float32)
    return ((pc - pc_mean) / pc_std).astype(np.float32)


def conditional_centers(
    fit_history_z: np.ndarray,
    fit_target_z: np.ndarray,
    query_history_z: np.ndarray,
    *,
    fit_action_z: np.ndarray | None = None,
    query_action_z: np.ndarray | None = None,
    components: int = MIXTURE_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight K-nearest conditional empirical mixture centers."""

    fit_h = np.asarray(fit_history_z, dtype=np.float64)
    fit_y = np.asarray(fit_target_z, dtype=np.float64)
    query_h = np.asarray(query_history_z, dtype=np.float64)
    if fit_h.ndim != 2 or fit_y.shape != (len(fit_h), TARGET_DIM):
        raise ValueError("fit conditional geometry differs")
    if query_h.ndim != 2 or query_h.shape[1] != fit_h.shape[1]:
        raise ValueError("query history geometry differs")
    if components <= 1 or components > len(fit_h):
        raise ValueError("mixture component count differs")
    action_conditioned = fit_action_z is not None or query_action_z is not None
    if action_conditioned:
        if fit_action_z is None or query_action_z is None:
            raise ValueError("both fit and query action are required")
        fit_a = np.asarray(fit_action_z, dtype=np.float64)
        query_a = np.asarray(query_action_z, dtype=np.float64)
        if fit_a.shape != (len(fit_h), ACTION_COMPONENTS):
            raise ValueError("fit action geometry differs")
        if query_a.shape != (len(query_h), ACTION_COMPONENTS):
            raise ValueError("query action geometry differs")
    centers: list[np.ndarray] = []
    indexes: list[np.ndarray] = []
    for row in range(len(query_h)):
        history_distance = np.mean((fit_h - query_h[row]) ** 2, axis=1)
        distance = history_distance
        if action_conditioned:
            action_distance = np.mean((fit_a - query_a[row]) ** 2, axis=1)
            distance = history_distance + ACTION_DISTANCE_WEIGHT * action_distance
        # Stable mergesort plus manifest-order fit arrays fixes all tie breaks.
        nearest = np.argsort(distance, kind="mergesort")[:components]
        centers.append(fit_y[nearest])
        indexes.append(nearest.astype(np.int64))
    return np.stack(centers).astype(np.float32), np.stack(indexes)


def _mixture_quantile(centers: np.ndarray, sigma: float, probability: float) -> float:
    values = np.asarray(centers, dtype=np.float64)
    lower = float(values.min() - 10.0 * sigma)
    upper = float(values.max() + 10.0 * sigma)

    def objective(point: float) -> float:
        return float(np.mean(norm.cdf((point - values) / sigma)) - probability)

    return float(brentq(objective, lower, upper, xtol=1e-10, rtol=1e-12))


def marginal_intervals(
    centers: np.ndarray,
    sigma: float,
) -> dict[str, np.ndarray]:
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (MIXTURE_COMPONENTS, TARGET_DIM):
        raise ValueError(f"mixture geometry differs: {values.shape}")
    result = {
        "q025": np.empty((len(values), TARGET_DIM), dtype=np.float64),
        "q10": np.empty((len(values), TARGET_DIM), dtype=np.float64),
        "q50": np.empty((len(values), TARGET_DIM), dtype=np.float64),
        "q90": np.empty((len(values), TARGET_DIM), dtype=np.float64),
        "q975": np.empty((len(values), TARGET_DIM), dtype=np.float64),
    }
    probabilities = {"q025": 0.025, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q975": 0.975}
    for row in range(len(values)):
        for dimension in range(TARGET_DIM):
            component = values[row, :, dimension]
            for label, probability in probabilities.items():
                result[label][row, dimension] = _mixture_quantile(
                    component, sigma, probability
                )
    return result


def _a_function(delta: np.ndarray, scale: float) -> np.ndarray:
    z = np.asarray(delta, dtype=np.float64) / float(scale)
    return 2.0 * scale * norm.pdf(z) + np.asarray(delta) * (2.0 * norm.cdf(z) - 1.0)


def distribution_scores(
    centers: np.ndarray,
    sigma: float,
    targets: np.ndarray,
    *,
    include_energy: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = np.asarray(centers, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if values.shape != (len(target), MIXTURE_COMPONENTS, TARGET_DIM):
        raise ValueError("score mixture geometry differs")
    if target.ndim != 2 or target.shape[1] != TARGET_DIM or sigma <= 0.0:
        raise ValueError("score target or sigma differs")

    delta = (target[:, None, :] - values) / sigma
    component_log_density = -0.5 * np.sum(delta**2, axis=2)
    component_log_density -= TARGET_DIM * math.log(sigma * math.sqrt(2.0 * math.pi))
    nll = -(
        logsumexp(component_log_density, axis=1) - math.log(MIXTURE_COMPONENTS)
    ) / TARGET_DIM

    first = _a_function(target[:, None, :] - values, sigma).mean(axis=1)
    pair_delta = values[:, :, None, :] - values[:, None, :, :]
    second = _a_function(pair_delta, math.sqrt(2.0) * sigma).mean(axis=(1, 2))
    crps = np.mean(first - 0.5 * second, axis=1)

    intervals = marginal_intervals(values, sigma)
    median_error = np.abs(target - intervals["q50"])

    def interval_score(lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
        score = upper - lower
        score += (2.0 / alpha) * (lower - target) * (target < lower)
        score += (2.0 / alpha) * (target - upper) * (target > upper)
        return score

    interval80 = interval_score(intervals["q10"], intervals["q90"], 0.20)
    interval95 = interval_score(intervals["q025"], intervals["q975"], 0.05)
    wis = np.mean(
        (0.5 * median_error + 0.10 * interval80 + 0.025 * interval95) / 2.5,
        axis=1,
    )
    coverage80 = np.mean(
        (target >= intervals["q10"]) & (target <= intervals["q90"]), axis=1
    )
    coverage95 = np.mean(
        (target >= intervals["q025"]) & (target <= intervals["q975"]), axis=1
    )
    scores: dict[str, np.ndarray] = {
        "nll": nll.astype(np.float64),
        "crps": crps.astype(np.float64),
        "wis": wis.astype(np.float64),
        "coverage80": coverage80.astype(np.float64),
        "coverage95": coverage95.astype(np.float64),
        "width80": np.mean(intervals["q90"] - intervals["q10"], axis=1),
        "width95": np.mean(intervals["q975"] - intervals["q025"], axis=1),
    }
    if include_energy:
        rng = np.random.default_rng(ENERGY_SEED)
        component_first = rng.integers(
            0, MIXTURE_COMPONENTS, size=ENERGY_SAMPLES, dtype=np.int64
        )
        component_second = rng.integers(
            0, MIXTURE_COMPONENTS, size=ENERGY_SAMPLES, dtype=np.int64
        )
        noise_first = rng.standard_normal((ENERGY_SAMPLES, TARGET_DIM))
        noise_second = rng.standard_normal((ENERGY_SAMPLES, TARGET_DIM))
        energy: list[float] = []
        for row in range(len(target)):
            first_sample = values[row, component_first] + sigma * noise_first
            second_sample = values[row, component_second] + sigma * noise_second
            observation_term = np.linalg.norm(first_sample - target[row], axis=1).mean()
            pair_term = np.linalg.norm(first_sample - second_sample, axis=1).mean()
            energy.append(float(observation_term - 0.5 * pair_term))
        scores["energy"] = np.asarray(energy, dtype=np.float64)
    if any(not np.isfinite(item).all() for item in scores.values()):
        raise DistributionError("distribution score contains non-finite values")
    return scores, intervals


def select_calibration_sigma(
    centers: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for sigma in SIGMA_GRID:
        scores, _ = distribution_scores(
            centers, sigma, targets, include_energy=False
        )
        coverage80 = float(scores["coverage80"].mean())
        coverage95 = float(scores["coverage95"].mean())
        coverage_ok = bool(
            CALIBRATION_COVERAGE_80_RANGE[0]
            <= coverage80
            <= CALIBRATION_COVERAGE_80_RANGE[1]
            and CALIBRATION_COVERAGE_95_RANGE[0]
            <= coverage95
            <= CALIBRATION_COVERAGE_95_RANGE[1]
        )
        penalty = abs(coverage80 - 0.80) + abs(coverage95 - 0.95)
        rows.append(
            {
                "sigma": sigma,
                "mean_nll_nats_per_dim": float(scores["nll"].mean()),
                "mean_crps": float(scores["crps"].mean()),
                "coverage80": coverage80,
                "coverage95": coverage95,
                "coverage_constraints_pass": coverage_ok,
                "coverage_penalty": penalty,
            }
        )
    eligible = [item for item in rows if item["coverage_constraints_pass"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (item["mean_nll_nats_per_dim"], item["sigma"]),
        )
    else:
        selected = min(
            rows,
            key=lambda item: (
                item["coverage_penalty"],
                item["mean_nll_nats_per_dim"],
                item["sigma"],
            ),
        )
    return float(selected["sigma"]), rows


def _remaining_final_items(parent_registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    pool = [
        dict(item)
        for item in parent_registration["split"]["untouched_pool"]
        if not bool(item["selected_fresh_score"])
    ]
    if len(pool) != FINAL_CLIPS:
        raise DistributionError(f"expected {FINAL_CLIPS} untouched clips, got {len(pool)}")
    final: list[dict[str, Any]] = []
    for item in pool:
        record = dict(item)
        record["role"] = "distribution_final_score"
        record["validation_accessed"] = False
        record["protected_test_accessed"] = False
        final.append(record)
    for stratum in range(slot.MOTION_STRATA):
        group = sorted(
            [item for item in final if int(item["motion_stratum"]) == stratum],
            key=lambda item: (item["fresh_selection_hash"], item["clip_id"]),
        )
        if len(group) < 2:
            raise DistributionError(f"final stratum {stratum} lacks shuffle donors")
        for position, item in enumerate(group):
            donor = group[(position + 1) % len(group)]
            item["donor_manifest_index"] = int(donor["manifest_index"])
            item["donor_clip_id"] = donor["clip_id"]
            item["donor_episode_dir"] = donor["episode_dir"]
    return sorted(
        final,
        key=lambda item: (int(item["motion_stratum"]), item["fresh_selection_hash"]),
    )


def _load_parent(parent_artifact: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registration = base.read_json(parent_artifact / "registration.json")
    analysis = base.read_json(parent_artifact / "analysis.json")
    complete = base.read_json(parent_artifact / "run_complete.json")
    for name, document in (
        ("parent registration", registration),
        ("parent analysis", analysis),
        ("parent completion", complete),
    ):
        base.validate_identity(document, name)
    if registration["identity_sha256"] != PARENT_REGISTRATION_IDENTITY:
        raise DistributionError("parent registration identity differs")
    if analysis["identity_sha256"] != PARENT_ANALYSIS_IDENTITY:
        raise DistributionError("parent analysis identity differs")
    if complete["identity_sha256"] != PARENT_COMPLETION_IDENTITY:
        raise DistributionError("parent completion identity differs")
    if analysis["decision"] != "STOP_OBJECT_CONTACT_SLOT":
        raise DistributionError("parent deterministic stop is unavailable")
    return registration, analysis, complete


def _calibrate(
    parent_registration: Mapping[str, Any],
    parent_arrays: Mapping[str, np.ndarray],
    action_cache: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fit_items = parent_registration["split"]["fit"]
    calibration_items = parent_registration["split"]["fresh_score"]
    fit_indices = np.asarray([item["manifest_index"] for item in fit_items], dtype=np.int64)
    calibration_indices = np.asarray(
        [item["manifest_index"] for item in calibration_items], dtype=np.int64
    )
    if not np.array_equal(fit_indices, np.asarray(parent_arrays["fit_indices"])):
        raise DistributionError("parent fit array order differs")
    if not np.array_equal(
        calibration_indices, np.asarray(parent_arrays["score_indices"])
    ):
        raise DistributionError("parent calibration array order differs")

    fit_history = compact_summary(np.asarray(parent_arrays["fit_history"]))
    fit_target = compact_summary(np.asarray(parent_arrays["fit_target"]))
    calibration_history = compact_summary(np.asarray(parent_arrays["score_history"]))
    calibration_target = compact_summary(np.asarray(parent_arrays["score_target"]))
    history_mean, history_std = base._standardizer(fit_history)
    target_mean, target_std = base._standardizer(fit_target)
    history_active = fit_history.std(axis=0, dtype=np.float64) > 1e-8
    target_active = fit_target.std(axis=0, dtype=np.float64) > 1e-8
    if int(history_active.sum()) != TARGET_DIM or int(target_active.sum()) != TARGET_DIM:
        raise DistributionError("all seven compact history/target dimensions must be active")
    fit_history_z = (fit_history - history_mean) / history_std
    fit_target_z = (fit_target - target_mean) / target_std
    calibration_history_z = (calibration_history - history_mean) / history_std
    calibration_target_z = (calibration_target - target_mean) / target_std

    fit_actions = np.asarray(
        action_cache[fit_indices, slot.ACTION_WINDOW[0] : slot.ACTION_WINDOW[1], :, : base.ACTION_DIM],
        dtype=np.float32,
    )
    calibration_actions = np.asarray(
        action_cache[
            calibration_indices,
            slot.ACTION_WINDOW[0] : slot.ACTION_WINDOW[1],
            :,
            : base.ACTION_DIM,
        ],
        dtype=np.float32,
    )
    result = _fit_action_pca8(fit_actions)
    action_pca, fit_action_z, action_mean, action_std, action_active, action_pc_mean, action_pc_std = result
    calibration_action_z = _transform_action_pca8(
        action_pca,
        calibration_actions,
        action_mean,
        action_std,
        action_active,
        action_pc_mean,
        action_pc_std,
    )

    history_centers, history_neighbors = conditional_centers(
        fit_history_z, fit_target_z, calibration_history_z
    )
    action_centers, action_neighbors = conditional_centers(
        fit_history_z,
        fit_target_z,
        calibration_history_z,
        fit_action_z=fit_action_z,
        query_action_z=calibration_action_z,
    )
    history_sigma, history_grid = select_calibration_sigma(
        history_centers, calibration_target_z
    )
    action_sigma, action_grid = select_calibration_sigma(
        action_centers, calibration_target_z
    )
    stratum_centroids = np.stack(
        [
            fit_action_z[
                np.asarray(
                    [int(item["motion_stratum"]) == stratum for item in fit_items],
                    dtype=bool,
                )
            ].mean(axis=0)
            for stratum in range(slot.MOTION_STRATA)
        ]
    ).astype(np.float32)
    model_arrays = {
        "fit_indices": fit_indices,
        "calibration_indices": calibration_indices,
        "fit_history_z": fit_history_z.astype(np.float32),
        "fit_target_z": fit_target_z.astype(np.float32),
        "fit_action_z": fit_action_z.astype(np.float32),
        "history_mean": history_mean,
        "history_std": history_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "action_active": action_active,
        "action_pca_mean": action_pca.mean_,
        "action_pca_components": action_pca.components_,
        "action_pca_explained_variance_ratio": action_pca.explained_variance_ratio_,
        "action_pc_mean": action_pc_mean,
        "action_pc_std": action_pc_std,
        "stratum_action_centroids": stratum_centroids,
        "calibration_history_z": calibration_history_z.astype(np.float32),
        "calibration_target_z": calibration_target_z.astype(np.float32),
        "calibration_action_z": calibration_action_z.astype(np.float32),
        "calibration_history_neighbors": history_neighbors,
        "calibration_action_neighbors": action_neighbors,
        "history_sigma": np.asarray(history_sigma, dtype=np.float64),
        "action_sigma": np.asarray(action_sigma, dtype=np.float64),
    }
    calibration = {
        "schema_version": SCHEMA_VERSION,
        "kind": "object_contact_distribution_calibration",
        "created_at_utc": now(),
        "fit_clips": FIT_CLIPS,
        "calibration_clips": CALIBRATION_CLIPS,
        "mixture_components_every_arm": MIXTURE_COMPONENTS,
        "target_features": list(TARGET_FEATURES),
        "history_sigma": history_sigma,
        "action_sigma": action_sigma,
        "history_sigma_grid": history_grid,
        "action_sigma_grid": action_grid,
        "calibration_outcomes_are_evidence": False,
        "remaining_final_rgb_or_state_opened": False,
        "validation_accessed": False,
        "protected_test_accessed": False,
    }
    return calibration, model_arrays


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent_artifact = args.parent_artifact.resolve()
    parent_registration, parent_analysis, parent_complete = _load_parent(parent_artifact)
    final_items = _remaining_final_items(parent_registration)

    manifest_path = Path(parent_registration["inputs"]["train_manifest"]["path"])
    cache_dir = Path(parent_registration["inputs"]["cache_dir"])
    rows = base.read_jsonl(manifest_path)
    base.validate_train_manifest(rows)
    metadata, _, action_cache, rgb_path, action_path = base.load_cache(cache_dir)
    parent_arrays_path = Path(
        parent_analysis["artifacts"]["slot_model_predictions"]["path"]
    )
    if base.file_record(parent_arrays_path) != parent_analysis["artifacts"]["slot_model_predictions"]:
        raise DistributionError("parent array artifact differs")
    with np.load(parent_arrays_path, allow_pickle=False) as loaded:
        parent_arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    calibration, model_arrays = _calibrate(
        parent_registration, parent_arrays, action_cache
    )

    model_path = output / "calibration_model.npz"
    np.savez_compressed(model_path, **model_arrays)
    calibration = base.seal(
        {
            **calibration,
            "parent_registration_identity_sha256": parent_registration["identity_sha256"],
            "parent_analysis_identity_sha256": parent_analysis["identity_sha256"],
            "model_artifact": base.file_record(model_path),
        }
    )
    calibration_path = output / "calibration.json"
    base.write_json(calibration_path, calibration)

    source = Path(__file__).resolve()
    slot_dependency = Path(slot.__file__).resolve()
    base_dependency = Path(base.__file__).resolve()
    protocol = args.protocol.resolve()
    frozen_source = output / "frozen_object_contact_distribution_stage0.py"
    frozen_slot = output / "frozen_object_contact_slot_stage0.py"
    frozen_base = output / "frozen_interaction_event_bottleneck_stage0.py"
    shutil.copy2(source, frozen_source)
    shutil.copy2(slot_dependency, frozen_slot)
    shutil.copy2(base_dependency, frozen_base)

    fit_indexes = {int(item["manifest_index"]) for item in parent_registration["split"]["fit"]}
    calibration_indexes = {
        int(item["manifest_index"])
        for item in parent_registration["split"]["fresh_score"]
    }
    prior_indexes = {
        int(item["manifest_index"])
        for item in parent_registration["split"]["excluded_parent_score"]
    }
    final_indexes = {int(item["manifest_index"]) for item in final_items}
    if final_indexes & (fit_indexes | calibration_indexes | prior_indexes):
        raise DistributionError("final score overlaps any previously opened partition")
    counts_by_stratum = {
        str(stratum): sum(
            int(int(item["motion_stratum"]) == stratum) for item in final_items
        )
        for stratum in range(slot.MOTION_STRATA)
    }
    registration = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_stage0_registration",
            "status": "registered_after_calibration_before_remaining31_rgb_or_state_access",
            "created_at_utc": now(),
            "source": {
                **base.file_record(source),
                "git_commit": args.expected_commit or base.git_commit(REPO_ROOT),
                "frozen_copy": base.file_record(frozen_source),
                "slot_dependency": base.file_record(slot_dependency),
                "frozen_slot_dependency": base.file_record(frozen_slot),
                "base_dependency": base.file_record(base_dependency),
                "frozen_base_dependency": base.file_record(frozen_base),
            },
            "protocol": base.file_record(protocol),
            "parent": {
                "artifact_root": str(parent_artifact),
                "registration": base.file_record(parent_artifact / "registration.json"),
                "analysis": base.file_record(parent_artifact / "analysis.json"),
                "completion": base.file_record(parent_artifact / "run_complete.json"),
                "registration_identity_sha256": parent_registration["identity_sha256"],
                "analysis_identity_sha256": parent_analysis["identity_sha256"],
                "completion_identity_sha256": parent_complete["identity_sha256"],
                "deterministic_stop_preserved": True,
            },
            "calibration": {
                "document": base.file_record(calibration_path),
                "identity_sha256": calibration["identity_sha256"],
                "model": base.file_record(model_path),
                "fit256_used_for_mixture_centers": True,
                "opened_fresh64_used_only_for_bandwidth_calibration": True,
                "calibration64_used_as_final_evidence": False,
                "selected_history_sigma": calibration["history_sigma"],
                "selected_action_sigma": calibration["action_sigma"],
            },
            "inputs": {
                "train_manifest": base.file_record(manifest_path),
                "cache_metadata": base.file_record(cache_dir / "metadata.json"),
                "cache_dir": str(cache_dir),
                "rgb": {
                    **base.file_record(rgb_path, digest=False),
                    "registered_sha256": metadata["rgb_sha256"],
                    "remaining31_pixel_array_indexed": False,
                },
                "actions": {
                    **base.file_record(action_path, digest=False),
                    "registered_sha256": metadata["actions_sha256"],
                    "fit_and_calibration_actions_opened": True,
                    "remaining31_action_is_causal_predictor_and_may_be_opened": True,
                },
                "preprocessed_root": parent_registration["inputs"]["preprocessed_root"],
                "raw_root": parent_registration["inputs"]["raw_root"],
                "remaining31_state_array_opened": False,
            },
            "split": {
                "source_split": "train",
                "fit_clips": FIT_CLIPS,
                "calibration_clips": CALIBRATION_CLIPS,
                "final_score_clips": FINAL_CLIPS,
                "final_score_by_original_motion_stratum": counts_by_stratum,
                "fit": parent_registration["split"]["fit"],
                "calibration": parent_registration["split"]["fresh_score"],
                "excluded_earlier_score": parent_registration["split"]["excluded_parent_score"],
                "final_score": final_items,
                "all_four_partitions_episode_disjoint": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "target": {
                "dimensions": TARGET_DIM,
                "features": list(TARGET_FEATURES),
                "definition": "slot-permutation-invariant mass-weighted summary over eight future transitions",
                "zero_event_maps_to_all_zero": True,
                "fit_standardized": True,
                "future_rgb_or_state_is_predictor_input": False,
            },
            "model": {
                "type": "equal-weight K-nearest conditional Gaussian mixture",
                "mixture_components_every_arm": MIXTURE_COMPONENTS,
                "component_covariance": "isotropic calibrated sigma squared times identity in fit-standardized target space",
                "history_condition": "seven compact observed-history coordinates",
                "action_condition": "fit-standardized PCA8 of action chunks 4:12",
                "combined_distance": "mean squared history distance plus mean squared action distance",
                "action_distance_weight": ACTION_DISTANCE_WEIGHT,
                "arms": list(ARMS),
                "action_stratum_control": "fit action-PCA centroid within the final episode original motion stratum",
                "episode_shuffle": "next remaining31 episode within original motion stratum",
            },
            "scores_and_gate": {
                "proper_scores": [
                    "heldout negative log likelihood in nats per target dimension",
                    "multivariate energy score with fixed common Monte Carlo draws",
                    "exact mean marginal Gaussian-mixture CRPS",
                    "weighted interval score from median and central 80/95 intervals",
                ],
                "coverage_diagnostics": ["central 80%", "central 95%"],
                "mandatory_references": list(REFERENCES),
                "family_size": PRIMARY_FAMILY_SIZE,
                "one_sided_bonferroni_confidence": 1.0 - 0.05 / PRIMARY_FAMILY_SIZE,
                "minimum_nll_improvement_nats_per_dim": MIN_NLL_IMPROVEMENT_NATS_PER_DIM,
                "minimum_other_score_relative_improvement_percent": MIN_RELATIVE_IMPROVEMENT_PERCENT,
                "minimum_favorable_fraction": MIN_FAVORABLE_FRACTION,
                "final_coverage80_range": list(FINAL_COVERAGE_80_RANGE),
                "final_coverage95_range": list(FINAL_COVERAGE_95_RANGE),
                "minimum_final_nonzero_fraction": MIN_FINAL_NONZERO_FRACTION,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "all_twelve_score_effects_and_coverage_salience_required": True,
            },
            "decision_rule": {
                "pass": "GO_FOR_STOCHASTIC_OBJECT_RESIDUAL_SCREEN",
                "fail": "CLOSE_INTERACTION_BRANCH",
                "generator_or_wan_is_part_of_this_stage": False,
            },
            "claim_boundary": (
                "One final train-only proper-score test of compact conditional uncertainty. "
                "No validation, protected test, generator, video metric, or control rollout is evaluated."
            ),
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
    base.write_json(output / "registration.json", registration)
    return registration


def paired_absolute_effect(
    reference: np.ndarray,
    candidate: np.ndarray,
    bootstrap: np.ndarray,
    *,
    family_size: int,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 1 or len(ref) != bootstrap.shape[1]:
        raise ValueError("absolute paired effect geometry differs")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("absolute paired scores must be finite")
    differences = ref - cand
    distribution = differences[bootstrap].mean(axis=1)
    probability = 0.05 / float(family_size)
    return {
        "reference_mean": float(ref.mean()),
        "candidate_mean": float(cand.mean()),
        "absolute_improvement_nats_per_dimension": float(differences.mean()),
        "one_sided_bonferroni_confidence": 1.0 - probability,
        "simultaneous_lower_bound_nats_per_dimension": float(
            np.quantile(distribution, probability, method="linear")
        ),
        "nominal_bootstrap_95_ci_nats_per_dimension": [
            float(np.quantile(distribution, 0.025, method="linear")),
            float(np.quantile(distribution, 0.975, method="linear")),
        ],
        "favorable_count": int(np.sum(cand < ref)),
        "favorable_fraction": float(np.mean(cand < ref)),
        "ties": int(np.sum(cand == ref)),
    }


def analyze_scores(
    errors: Mapping[str, np.ndarray],
    final_target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = base.common_bootstrap_indices(
        FINAL_CLIPS, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    effects: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for metric in PROPER_METRICS:
        candidate = np.asarray(errors[f"{metric}_aligned"])
        for reference in REFERENCES:
            label = f"{metric}:aligned_vs_{reference}"
            if metric == "nll":
                effect = paired_absolute_effect(
                    np.asarray(errors[f"{metric}_{reference}"]),
                    candidate,
                    bootstrap,
                    family_size=PRIMARY_FAMILY_SIZE,
                )
                checks = {
                    "absolute_nll_gain_at_least_0.05_nats_per_dim": float(
                        effect["absolute_improvement_nats_per_dimension"]
                    )
                    >= MIN_NLL_IMPROVEMENT_NATS_PER_DIM,
                    "simultaneous_lower_bound_strictly_positive": float(
                        effect["simultaneous_lower_bound_nats_per_dimension"]
                    )
                    > 0.0,
                    "favorable_fraction_at_least_60_percent": float(
                        effect["favorable_fraction"]
                    )
                    >= MIN_FAVORABLE_FRACTION,
                }
            else:
                effect = base.paired_relative_effect(
                    np.asarray(errors[f"{metric}_{reference}"]),
                    candidate,
                    bootstrap,
                    family_size=PRIMARY_FAMILY_SIZE,
                )
                checks = {
                    "relative_gain_at_least_5_percent": float(
                        effect["relative_improvement_percent"]
                    )
                    >= MIN_RELATIVE_IMPROVEMENT_PERCENT,
                    "simultaneous_lower_bound_strictly_positive": float(
                        effect["simultaneous_lower_bound_percent"]
                    )
                    > 0.0,
                    "favorable_fraction_at_least_60_percent": float(
                        effect["favorable_fraction"]
                    )
                    >= MIN_FAVORABLE_FRACTION,
                }
            effects[label] = effect
            gates[label] = {**checks, "passed": all(checks.values())}

    coverage80 = float(np.mean(errors["coverage80_aligned"]))
    coverage95 = float(np.mean(errors["coverage95_aligned"]))
    nonzero = np.asarray(final_target[:, 0] > 0.0)
    represented_strata = np.asarray(errors["final_motion_strata"], dtype=np.int64)
    salience = {
        "aligned_coverage80": coverage80,
        "aligned_coverage95": coverage95,
        "final_nonzero_count": int(nonzero.sum()),
        "final_nonzero_fraction": float(nonzero.mean()),
        "nonzero_motion_strata": sorted(set(represented_strata[nonzero].tolist())),
    }
    salience_checks = {
        "coverage80_in_70_to_90_percent": FINAL_COVERAGE_80_RANGE[0]
        <= coverage80
        <= FINAL_COVERAGE_80_RANGE[1],
        "coverage95_in_87_to_100_percent": FINAL_COVERAGE_95_RANGE[0]
        <= coverage95
        <= FINAL_COVERAGE_95_RANGE[1],
        "at_least_half_final_targets_nonzero": float(nonzero.mean())
        >= MIN_FINAL_NONZERO_FRACTION,
        "nonzero_targets_cover_all_motion_strata": len(set(represented_strata[nonzero].tolist()))
        == slot.MOTION_STRATA,
    }
    gates["coverage_and_salience"] = {
        **salience_checks,
        "passed": all(salience_checks.values()),
    }
    gates["proper_score_all_passed"] = all(
        item["passed"] for key, item in gates.items() if key != "coverage_and_salience"
    )
    gates["all_passed"] = bool(
        gates["proper_score_all_passed"]
        and gates["coverage_and_salience"]["passed"]
    )
    return {"effects": effects, "coverage_and_salience": salience}, gates


def _load_calibrated_model(
    registration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    calibration_path = Path(registration["calibration"]["document"]["path"])
    model_path = Path(registration["calibration"]["model"]["path"])
    if base.file_record(calibration_path) != registration["calibration"]["document"]:
        raise DistributionError("calibration document differs")
    if base.file_record(model_path) != registration["calibration"]["model"]:
        raise DistributionError("calibration model differs")
    calibration = base.read_json(calibration_path)
    base.validate_identity(calibration, "calibration")
    if calibration["identity_sha256"] != registration["calibration"]["identity_sha256"]:
        raise DistributionError("calibration identity differs")
    with np.load(model_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    return calibration, arrays


def _action_pca_from_arrays(arrays: Mapping[str, np.ndarray]) -> Any:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=ACTION_COMPONENTS, svd_solver="full")
    pca.mean_ = np.asarray(arrays["action_pca_mean"])
    pca.components_ = np.asarray(arrays["action_pca_components"])
    pca.n_components_ = ACTION_COMPONENTS
    pca.n_features_in_ = pca.components_.shape[1]
    return pca


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("run requires mujoco") from exc
    output = args.output.resolve()
    if (output / "analysis.json").exists() or (output / "run_complete.json").exists():
        raise FileExistsError("completed output is immutable")
    registration = base.read_json(output / "registration.json")
    base.validate_identity(registration, "registration")
    checks = (
        (Path(__file__).resolve(), registration["source"]["sha256"], "source"),
        (Path(slot.__file__).resolve(), registration["source"]["slot_dependency"]["sha256"], "slot dependency"),
        (Path(base.__file__).resolve(), registration["source"]["base_dependency"]["sha256"], "base dependency"),
        (output / "frozen_object_contact_distribution_stage0.py", registration["source"]["frozen_copy"]["sha256"], "frozen source"),
        (output / "frozen_object_contact_slot_stage0.py", registration["source"]["frozen_slot_dependency"]["sha256"], "frozen slot"),
        (output / "frozen_interaction_event_bottleneck_stage0.py", registration["source"]["frozen_base_dependency"]["sha256"], "frozen base"),
    )
    for path, expected, label in checks:
        if base.sha256_file(path) != expected:
            raise DistributionError(f"{label} bytes differ")
    calibration, model_arrays = _load_calibrated_model(registration)

    manifest_path = Path(registration["inputs"]["train_manifest"]["path"])
    rows = base.read_jsonl(manifest_path)
    base.validate_train_manifest(rows)
    _, rgb_cache, action_cache, _, _ = base.load_cache(
        Path(registration["inputs"]["cache_dir"])
    )
    final_items = registration["split"]["final_score"]

    official_root = args.official_abc_root.resolve()
    official_commit = base.git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise DistributionError("official ABC commit differs")
    scene_path = official_root / OFFICIAL_SCENE_RELATIVE
    asset_root = official_root / OFFICIAL_ASSET_RELATIVE
    robot_xml = build_robot_only_xml(scene_path, asset_root)
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    moving_geoms = moving_geom_ids(model, mujoco)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    renderer = mujoco.Renderer(model, height=slot.RENDER_HW[0], width=slot.RENDER_HW[1])
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
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    started = time.time()
    try:
        for position, item in enumerate(final_items):
            history, target, _, clip_actions, record, latencies = slot._selected_clip_arrays(
                item=item,
                row=rows[int(item["manifest_index"])],
                rgb_cache=rgb_cache,
                action_cache=action_cache,
                renderer_context=renderer_context,
            )
            histories.append(history)
            targets.append(target)
            actions.append(clip_actions)
            record["distributional_role"] = "final_score_only"
            record["fit_or_calibration_reused_as_final_evidence"] = False
            provenance.append(record)
            render_latencies.extend(latencies[2:] if len(latencies) > 2 else latencies)
            if position == 0 or (position + 1) % 8 == 0:
                print(
                    json.dumps(
                        {
                            "event": "distribution_final_extraction_progress",
                            "clips": position + 1,
                            "total": FINAL_CLIPS,
                            "elapsed_sec": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        renderer.close()

    final_history_slots = np.stack(histories)
    final_target_slots = np.stack(targets)
    final_actions_full = np.stack(actions)
    final_history = compact_summary(final_history_slots)
    final_target = compact_summary(final_target_slots)
    final_history_z = (
        final_history - np.asarray(model_arrays["history_mean"])
    ) / np.asarray(model_arrays["history_std"])
    final_target_z = (
        final_target - np.asarray(model_arrays["target_mean"])
    ) / np.asarray(model_arrays["target_std"])
    action_pca = _action_pca_from_arrays(model_arrays)
    final_action_window = final_actions_full[
        :, slot.ACTION_WINDOW[0] : slot.ACTION_WINDOW[1], :, : base.ACTION_DIM
    ]
    final_action_z = _transform_action_pca8(
        action_pca,
        final_action_window,
        np.asarray(model_arrays["action_mean"]),
        np.asarray(model_arrays["action_std"]),
        np.asarray(model_arrays["action_active"]),
        np.asarray(model_arrays["action_pc_mean"]),
        np.asarray(model_arrays["action_pc_std"]),
    )
    position_by_index = {
        int(item["manifest_index"]): position for position, item in enumerate(final_items)
    }
    donor_positions = np.asarray(
        [position_by_index[int(item["donor_manifest_index"])] for item in final_items],
        dtype=np.int64,
    )
    strata = np.asarray([int(item["motion_stratum"]) for item in final_items], dtype=np.int64)
    stratum_action = np.asarray(model_arrays["stratum_action_centroids"])[strata]
    fit_history_z = np.asarray(model_arrays["fit_history_z"])
    fit_target_z = np.asarray(model_arrays["fit_target_z"])
    fit_action_z = np.asarray(model_arrays["fit_action_z"])
    centers: dict[str, np.ndarray] = {}
    neighbors: dict[str, np.ndarray] = {}
    centers["history_only"], neighbors["history_only"] = conditional_centers(
        fit_history_z, fit_target_z, final_history_z
    )
    for label, query_action in (
        ("aligned", final_action_z),
        ("episode_shuffled", final_action_z[donor_positions]),
        ("action_stratum_centroid", stratum_action),
    ):
        centers[label], neighbors[label] = conditional_centers(
            fit_history_z,
            fit_target_z,
            final_history_z,
            fit_action_z=fit_action_z,
            query_action_z=query_action,
        )

    history_sigma = float(calibration["history_sigma"])
    action_sigma = float(calibration["action_sigma"])
    score_rows: dict[str, dict[str, np.ndarray]] = {}
    intervals: dict[str, dict[str, np.ndarray]] = {}
    for label in ARMS:
        sigma = history_sigma if label == "history_only" else action_sigma
        score_rows[label], intervals[label] = distribution_scores(
            centers[label], sigma, final_target_z
        )
    errors: dict[str, np.ndarray] = {
        f"{metric}_{label}": np.asarray(score_rows[label][metric])
        for label in ARMS
        for metric in (*PROPER_METRICS, "coverage80", "coverage95", "width80", "width95")
    }
    errors["final_motion_strata"] = strata
    metrics, gates = analyze_scores(errors, final_target)
    decision = (
        "GO_FOR_STOCHASTIC_OBJECT_RESIDUAL_SCREEN"
        if gates["all_passed"]
        else "CLOSE_INTERACTION_BRANCH"
    )

    provenance_path = output / "final_input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for record in provenance:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    per_clip_path = output / "final_per_clip_scores.jsonl"
    with per_clip_path.open("w") as handle:
        for position, item in enumerate(final_items):
            record = {
                "schema_version": SCHEMA_VERSION,
                "score_position": position,
                "manifest_index": int(item["manifest_index"]),
                "clip_id": item["clip_id"],
                "motion_stratum": int(item["motion_stratum"]),
                "donor_manifest_index": int(item["donor_manifest_index"]),
                "compact_target": final_target[position].tolist(),
                "scores": {
                    label: {
                        name: float(values[position])
                        for name, values in score_rows[label].items()
                    }
                    for label in ARMS
                },
                "source_split": "train",
                "final_score_only": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    arrays_path = output / "distribution_predictions.npz"
    np.savez_compressed(
        arrays_path,
        final_indices=np.asarray([item["manifest_index"] for item in final_items], dtype=np.int64),
        donor_positions=donor_positions,
        final_motion_strata=strata,
        final_history_slots=final_history_slots,
        final_target_slots=final_target_slots,
        final_history=final_history,
        final_target=final_target,
        final_history_z=final_history_z,
        final_target_z=final_target_z,
        final_action_z=final_action_z,
        **{f"centers_{label}": values for label, values in centers.items()},
        **{f"neighbors_{label}": values for label, values in neighbors.items()},
        **errors,
    )

    latency = np.asarray(render_latencies, dtype=np.float64)
    analysis = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_stage0_analysis",
            "created_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "calibration_identity_sha256": calibration["identity_sha256"],
            "decision": decision,
            "interpretation": (
                "aligned action passed every fresh proper-score, calibration, and salience gate"
                if gates["all_passed"]
                else "aligned action failed at least one final proper-score, calibration, or salience gate; close the interaction branch"
            ),
            "population": {
                "source_split": "train",
                "fit_clips": FIT_CLIPS,
                "calibration_clips": CALIBRATION_CLIPS,
                "final_score_clips": FINAL_CLIPS,
                "final_score_by_motion_stratum": registration["split"]["final_score_by_original_motion_stratum"],
                "all_partitions_episode_disjoint": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "calibrated_model": {
                "mixture_components_every_arm": MIXTURE_COMPONENTS,
                "history_sigma": history_sigma,
                "action_sigma": action_sigma,
                "target_dimensions": TARGET_DIM,
                "target_features": list(TARGET_FEATURES),
                "action_pca_variance_retained": float(
                    np.asarray(model_arrays["action_pca_explained_variance_ratio"]).sum()
                ),
                "mean_neighbor_overlap_aligned_vs_history": float(
                    np.mean(
                        [
                            len(set(neighbors["aligned"][index]) & set(neighbors["history_only"][index]))
                            / MIXTURE_COMPONENTS
                            for index in range(FINAL_CLIPS)
                        ]
                    )
                ),
                "mean_neighbor_overlap_aligned_vs_shuffled": float(
                    np.mean(
                        [
                            len(set(neighbors["aligned"][index]) & set(neighbors["episode_shuffled"][index]))
                            / MIXTURE_COMPONENTS
                            for index in range(FINAL_CLIPS)
                        ]
                    )
                ),
            },
            "metrics": {
                **metrics,
                "gates": gates,
                "mean_scores": {
                    label: {
                        metric: float(score_rows[label][metric].mean())
                        for metric in (*PROPER_METRICS, "coverage80", "coverage95", "width80", "width95")
                    }
                    for label in ARMS
                },
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "family_size": PRIMARY_FAMILY_SIZE,
            },
            "renderer": {
                "official_abc_commit": official_commit,
                "official_scene": base.file_record(scene_path),
                "robot_only_xml_sha256": hashlib.sha256(robot_xml.encode()).hexdigest(),
                "render_pose_mean_ms": float(latency.mean()),
                "render_pose_p95_ms": float(np.percentile(latency, 95)),
                "render_pose_count_after_two_warmups_per_clip": int(len(latency)),
            },
            "artifacts": {
                "registration": base.file_record(output / "registration.json"),
                "calibration": base.file_record(output / "calibration.json"),
                "calibration_model": base.file_record(output / "calibration_model.npz"),
                "frozen_source": base.file_record(output / "frozen_object_contact_distribution_stage0.py"),
                "frozen_slot_dependency": base.file_record(output / "frozen_object_contact_slot_stage0.py"),
                "frozen_base_dependency": base.file_record(output / "frozen_interaction_event_bottleneck_stage0.py"),
                "final_input_provenance": base.file_record(provenance_path),
                "final_per_clip_scores": base.file_record(per_clip_path),
                "distribution_predictions": base.file_record(arrays_path),
            },
            "claim_boundary": (
                "One final 31-episode train-only proper-score comparison of a compact conditional empirical mixture. "
                "No validation, protected test, generator, video metric, or control rollout was evaluated."
            ),
            "limitations": [
                "31 final episodes provide limited power under twelve-way multiplicity correction",
                "equal-weight nearest-neighbor Gaussian mixtures are deliberately simple conditional density models",
                "compact summaries discard object identity and detailed temporal trajectories",
                "connected motion/contact slots remain observational proxies",
                "bootstrap covers final episode sampling, not alternative model fits",
            ],
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "analysis.json", analysis)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_stage0_complete",
            "completed_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": decision,
            "status": "completed",
            "artifacts": {
                "analysis": base.file_record(output / "analysis.json"),
                **analysis["artifacts"],
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "run_complete.json", complete)
    return analysis


def audit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    registration = base.read_json(output / "registration.json")
    analysis = base.read_json(output / "analysis.json")
    complete = base.read_json(output / "run_complete.json")
    for name, document in (
        ("registration", registration),
        ("analysis", analysis),
        ("complete", complete),
    ):
        base.validate_identity(document, name)
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise DistributionError("analysis-registration binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise DistributionError("completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise DistributionError("completion decision differs")
    checks = (
        (Path(__file__).resolve(), registration["source"]["sha256"], "source"),
        (Path(slot.__file__).resolve(), registration["source"]["slot_dependency"]["sha256"], "slot dependency"),
        (Path(base.__file__).resolve(), registration["source"]["base_dependency"]["sha256"], "base dependency"),
        (output / "frozen_object_contact_distribution_stage0.py", registration["source"]["frozen_copy"]["sha256"], "frozen source"),
        (output / "frozen_object_contact_slot_stage0.py", registration["source"]["frozen_slot_dependency"]["sha256"], "frozen slot"),
        (output / "frozen_interaction_event_bottleneck_stage0.py", registration["source"]["frozen_base_dependency"]["sha256"], "frozen base"),
    )
    for path, expected, label in checks:
        if base.sha256_file(path) != expected:
            raise DistributionError(f"audit {label} differs")
    for artifact in analysis["artifacts"].values():
        if base.file_record(Path(artifact["path"])) != artifact:
            raise DistributionError(f"artifact differs: {artifact['path']}")

    fit = {int(item["manifest_index"]) for item in registration["split"]["fit"]}
    calibration = {
        int(item["manifest_index"]) for item in registration["split"]["calibration"]
    }
    earlier = {
        int(item["manifest_index"])
        for item in registration["split"]["excluded_earlier_score"]
    }
    final = {
        int(item["manifest_index"]) for item in registration["split"]["final_score"]
    }
    if len(final) != FINAL_CLIPS or final & (fit | calibration | earlier):
        raise DistributionError("final partition overlap differs")
    with np.load(output / "distribution_predictions.npz", allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    final_target = compact_summary(arrays["final_target_slots"])
    if not np.array_equal(final_target, arrays["final_target"]):
        raise DistributionError("recomputed compact target differs")
    calibration_document, _ = _load_calibrated_model(registration)
    errors: dict[str, np.ndarray] = {
        name: arrays[name]
        for name in arrays
        if any(name.startswith(f"{metric}_") for metric in (*PROPER_METRICS, "coverage80", "coverage95", "width80", "width95"))
    }
    errors["final_motion_strata"] = arrays["final_motion_strata"]
    for label in ARMS:
        sigma = (
            float(calibration_document["history_sigma"])
            if label == "history_only"
            else float(calibration_document["action_sigma"])
        )
        recomputed, _ = distribution_scores(
            arrays[f"centers_{label}"], sigma, arrays["final_target_z"]
        )
        for metric, values in recomputed.items():
            key = f"{metric}_{label}"
            if key in errors and not np.array_equal(values, errors[key]):
                raise DistributionError(f"recomputed score differs: {key}")
    metrics, gates = analyze_scores(errors, final_target)
    if metrics["effects"] != analysis["metrics"]["effects"]:
        raise DistributionError("recomputed proper-score effects differ")
    if metrics["coverage_and_salience"] != analysis["metrics"]["coverage_and_salience"]:
        raise DistributionError("recomputed coverage/salience differs")
    if gates != analysis["metrics"]["gates"]:
        raise DistributionError("recomputed gates differ")
    expected_decision = (
        "GO_FOR_STOCHASTIC_OBJECT_RESIDUAL_SCREEN"
        if gates["all_passed"]
        else "CLOSE_INTERACTION_BRANCH"
    )
    if expected_decision != analysis["decision"]:
        raise DistributionError("recomputed decision differs")
    provenance = base.read_jsonl(output / "final_input_provenance.jsonl")
    per_clip = base.read_jsonl(output / "final_per_clip_scores.jsonl")
    if len(provenance) != FINAL_CLIPS or len(per_clip) != FINAL_CLIPS:
        raise DistributionError("final row counts differ")
    false_flags = 0
    for document in (registration, analysis, complete, provenance, per_clip):
        false_flags += base.require_false_flags(document)
    return base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_stage0_audit",
            "audited_at_utc": now(),
            "status": "audit_passed",
            "decision": analysis["decision"],
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "final_overlap_with_any_previously_opened_partition": 0,
            "provenance_rows": len(provenance),
            "score_rows": len(per_clip),
            "recomputed_proper_score_effects": len(metrics["effects"]),
            "explicit_false_flags": false_flags,
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--parent-artifact", type=Path, required=True)
    register_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/experiments/OBJECT_CONTACT_DISTRIBUTION_STAGE0_PROTOCOL.md"),
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
            base.write_json(args.report, result)
        event = "audited"
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"event": event, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
