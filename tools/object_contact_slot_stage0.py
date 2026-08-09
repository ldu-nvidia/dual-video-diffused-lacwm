#!/usr/bin/env python3
"""Prospective train-only causal object/contact-slot qualification.

Registration reuses the prior predictor-fit episodes but deterministically
selects a fresh score set from D405 episodes that were neither fit nor score in
the earlier interaction-event study.  It seals that split before any selected
RGB pixel or measured state is indexed.  Execution extracts causal persistent
robot-masked moving-object slots and applies frozen causal, timing, salience,
and oracle-reconstruction gates.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import interaction_event_bottleneck_stage0 as base
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
from tools.corrected_renderer_attribution import (
    moving_geom_ids,
    raw_mcap_path,
    read_mcap_metadata,
)


SCHEMA_VERSION = 1
PARENT_REGISTRATION_IDENTITY = (
    "f985c94c708950a6e90530044a9b91f58e539a50a3f7ac2f91b89b1dc945f393"
)
FRESH_SELECTION_SALT = "object-contact-slot-fresh-score-v1"
FIT_CLIPS = 256
SCORE_CLIPS = 64
MOTION_STRATA = 4
SCORE_PER_STRATUM = 16
MODEL_SEED = 2468
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_SAMPLES = 10_000

SLOT_COUNT = 4
SLOT_FEATURES = (
    "presence",
    "centroid_x",
    "centroid_y",
    "area_fraction",
    "positive_mass",
    "negative_mass",
    "covariance_xx",
    "covariance_xy",
    "covariance_yy",
    "signed_motion_u",
    "signed_motion_v",
    "contact_score",
    "contact_onset",
    "track_age",
)
SLOT_DIM = len(SLOT_FEATURES)
PRESENCE_INDEX = SLOT_FEATURES.index("presence")
CORE_FEATURE_NAMES = (
    "presence",
    "centroid_x",
    "centroid_y",
    "signed_motion_u",
    "signed_motion_v",
    "contact_score",
    "contact_onset",
)

WORK_HW = base.WORK_HW
RENDER_HW = base.RENDER_HW
ROBOT_DILATION_PX = 16
PHOTO_THRESHOLD = base.PHOTO_THRESHOLD
PHOTO_SCALE = base.PHOTO_SCALE
FLOW_SCALE_PX = base.FLOW_SCALE_PX
COMPONENT_EVENT_THRESHOLD = 0.20
COMPONENT_FLOW_THRESHOLD_PX = 0.20
COMPONENT_MIN_AREA_PX = 4
COMPONENT_MAX_AREA_FRACTION = 0.20
CONTACT_MAX_DISTANCE_WORK_PX = 12.0
CONTACT_SCALE_WORK_PX = 4.0
MAX_COMPONENT_CANDIDATES = 12
TRACK_MAX_MISSED_TRANSITIONS = 1
TRACK_MAX_DISTANCE_NORM = 0.30
TRACK_MAX_COST = 0.38
DECODER_MIN_VARIANCE = (2.0 / (WORK_HW[1] - 1)) ** 2
DECODER_MAX_VARIANCE = 0.30

ACTION_WINDOW = base.ACTION_WINDOW
SHIFT_WINDOWS = base.SHIFT_WINDOWS
ALPHAS = base.ALPHAS
ACTION_COMPONENTS = base.PCA_COMPONENTS
CAUSAL_METRICS = ("slot", "field")
MANDATORY_REFERENCES = (
    "history_only",
    "episode_shuffled",
    "train_mean",
    "raw_zero",
    "shift_minus1",
    "shift_plus1",
)
CAUSAL_FAMILY_SIZE = len(CAUSAL_METRICS) * len(MANDATORY_REFERENCES)
RECONSTRUCTION_FAMILY_SIZE = 2
MIN_CAUSAL_IMPROVEMENT_PERCENT = 5.0
MIN_RECONSTRUCTION_IMPROVEMENT_PERCENT = 20.0
MIN_FAVORABLE_FRACTION = 0.60
MIN_ACTIVE_TARGET_FRACTION = 0.50
MIN_SCORE_NONEMPTY_FRACTION = 0.90
MIN_SCORE_PERSISTENT_FRACTION = 0.60


class SlotError(RuntimeError):
    """Raised when a frozen slot-screen contract is violated."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fresh_selection_hash(clip_id: str) -> str:
    return hashlib.sha256(f"{clip_id}|{FRESH_SELECTION_SALT}".encode()).hexdigest()


def _motion_stratified_candidates(
    eligible: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(item) for item in eligible),
        key=lambda item: (float(item["planned_joint_motion_rms"]), item["clip_id"]),
    )
    result: list[dict[str, Any]] = []
    for stratum, values in enumerate(
        np.array_split(np.asarray(ordered, dtype=object), MOTION_STRATA)
    ):
        for item in values.tolist():
            record = dict(item)
            record["motion_stratum"] = stratum
            result.append(record)
    return result


def select_fresh_score(
    eligible: Sequence[Mapping[str, Any]],
    parent_registration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reuse parent fit and select fresh score only from untouched episodes."""

    base.validate_identity(parent_registration, "parent registration")
    if parent_registration.get("identity_sha256") != PARENT_REGISTRATION_IDENTITY:
        raise SlotError("parent registration identity differs from frozen prerequisite")
    if parent_registration.get("kind") != "interaction_event_bottleneck_stage0_registration":
        raise SlotError("parent registration kind differs")

    reproduced_fit, reproduced_score = base.select_fit_score(eligible)
    parent_fit = [dict(item) for item in parent_registration["split"]["fit"]]
    parent_score = [dict(item) for item in parent_registration["split"]["score"]]
    for label, reproduced, frozen in (
        ("fit", reproduced_fit, parent_fit),
        ("score", reproduced_score, parent_score),
    ):
        reproduced_signature = [
            (int(item["manifest_index"]), int(item["motion_stratum"]))
            for item in reproduced
        ]
        frozen_signature = [
            (int(item["manifest_index"]), int(item["motion_stratum"]))
            for item in frozen
        ]
        if reproduced_signature != frozen_signature:
            raise SlotError(f"recomputed parent {label} split differs")

    stratified = _motion_stratified_candidates(eligible)
    fit_indexes = {int(item["manifest_index"]) for item in parent_fit}
    prior_score_indexes = {int(item["manifest_index"]) for item in parent_score}
    if fit_indexes & prior_score_indexes:
        raise SlotError("parent fit and score overlap")
    remaining = [
        dict(item)
        for item in stratified
        if int(item["manifest_index"]) not in fit_indexes | prior_score_indexes
    ]

    fresh_score: list[dict[str, Any]] = []
    pool_records: list[dict[str, Any]] = []
    per_stratum: dict[str, Any] = {}
    for stratum in range(MOTION_STRATA):
        pool = [item for item in remaining if int(item["motion_stratum"]) == stratum]
        ranked = sorted(
            pool,
            key=lambda item: (fresh_selection_hash(item["clip_id"]), item["clip_id"]),
        )
        if len(ranked) < SCORE_PER_STRATUM:
            raise SlotError(
                f"fresh stratum {stratum} has {len(ranked)} candidates; need {SCORE_PER_STRATUM}"
            )
        selected_indexes = {
            int(item["manifest_index"]) for item in ranked[:SCORE_PER_STRATUM]
        }
        per_stratum[str(stratum)] = {
            "eligible_d405": sum(
                int(item["motion_stratum"] == stratum) for item in stratified
            ),
            "reused_fit": sum(
                int(int(item["motion_stratum"]) == stratum) for item in parent_fit
            ),
            "excluded_prior_score": sum(
                int(int(item["motion_stratum"]) == stratum) for item in parent_score
            ),
            "untouched_pool": len(ranked),
            "fresh_score": SCORE_PER_STRATUM,
            "unused_after_selection": len(ranked) - SCORE_PER_STRATUM,
        }
        for rank, item in enumerate(ranked):
            record = dict(item)
            record["fresh_selection_hash"] = fresh_selection_hash(item["clip_id"])
            record["fresh_rank_within_stratum"] = rank
            record["selected_fresh_score"] = int(item["manifest_index"]) in selected_indexes
            record["validation_accessed"] = False
            record["protected_test_accessed"] = False
            pool_records.append(record)
            if rank < SCORE_PER_STRATUM:
                chosen = dict(record)
                chosen["role"] = "fresh_heldout_score"
                fresh_score.append(chosen)

    for stratum in range(MOTION_STRATA):
        group = sorted(
            [item for item in fresh_score if int(item["motion_stratum"]) == stratum],
            key=lambda item: (item["fresh_selection_hash"], item["clip_id"]),
        )
        for position, item in enumerate(group):
            donor = group[(position + 1) % len(group)]
            if donor["episode_dir"] == item["episode_dir"]:
                raise SlotError("fresh shuffled donor equals native episode")
            item["donor_manifest_index"] = int(donor["manifest_index"])
            item["donor_clip_id"] = donor["clip_id"]
            item["donor_episode_dir"] = donor["episode_dir"]

    fit = sorted(
        [dict(item, role="predictor_fit") for item in parent_fit],
        key=lambda item: (int(item["motion_stratum"]), item["selection_hash"]),
    )
    fresh_score = sorted(
        fresh_score,
        key=lambda item: (int(item["motion_stratum"]), item["fresh_selection_hash"]),
    )
    if len(fit) != FIT_CLIPS or len(fresh_score) != SCORE_CLIPS:
        raise SlotError("fit/fresh-score geometry differs")
    fresh_indexes = {int(item["manifest_index"]) for item in fresh_score}
    if fresh_indexes & (fit_indexes | prior_score_indexes):
        raise SlotError("fresh score overlaps fit or any prior score")
    counts = {
        "eligible_d405": len(stratified),
        "reused_parent_fit": len(fit_indexes),
        "excluded_parent_score": len(prior_score_indexes),
        "untouched_pool_before_fresh_selection": len(remaining),
        "fresh_score": len(fresh_score),
        "unused_untouched_after_fresh_selection": len(remaining) - len(fresh_score),
        "per_motion_stratum": per_stratum,
    }
    return fit, fresh_score, pool_records, counts


def _scan_eligible(
    rows: Sequence[Mapping[str, Any]],
    actions: np.ndarray,
    preprocessed_root: Path,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[dict[str, Any]] = []
    camera_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        episode = Path(row["episode_dir"]).resolve()
        if not (episode / "states.npz").is_file():
            continue
        try:
            mcap_path = raw_mcap_path(row, preprocessed_root, raw_root)
            metadata = read_mcap_metadata(mcap_path)
        except (FileNotFoundError, ValueError):
            continue
        camera_type = metadata.get("top_camera_type", "missing")
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
                "planned_joint_motion_rms": base.planned_motion(actions[index]),
                "validation_accessed": False,
                "protected_test_accessed": False,
            }
        )
    return eligible, camera_counts


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent_artifact = args.parent_artifact.resolve()
    parent_path = parent_artifact / "registration.json"
    parent = base.read_json(parent_path)
    base.validate_identity(parent, "parent registration")
    if parent["identity_sha256"] != PARENT_REGISTRATION_IDENTITY:
        raise SlotError("wrong parent registration")

    manifest_path = Path(parent["inputs"]["train_manifest"]["path"])
    cache_dir = Path(parent["inputs"]["cache_dir"])
    preprocessed_root = Path(parent["inputs"]["preprocessed_root"])
    raw_root = Path(parent["inputs"]["raw_root"])
    rows = base.read_jsonl(manifest_path)
    base.validate_train_manifest(rows)
    metadata, _, actions, rgb_path, action_path = base.load_cache(cache_dir)
    eligible, camera_counts = _scan_eligible(
        rows, actions, preprocessed_root, raw_root
    )
    fit, fresh_score, untouched_pool, counts = select_fresh_score(eligible, parent)

    source = Path(__file__).resolve()
    dependency = Path(base.__file__).resolve()
    protocol = args.protocol.resolve()
    frozen_source = output / "frozen_object_contact_slot_stage0.py"
    frozen_dependency = output / "frozen_interaction_event_bottleneck_stage0.py"
    shutil.copy2(source, frozen_source)
    shutil.copy2(dependency, frozen_dependency)
    payload = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_slot_stage0_registration",
            "status": "registered_before_fresh_score_rgb_or_state_access",
            "created_at_utc": now(),
            "source": {
                **base.file_record(source),
                "git_commit": args.expected_commit or base.git_commit(REPO_ROOT),
                "frozen_copy": base.file_record(frozen_source),
                "base_dependency": base.file_record(dependency),
                "frozen_base_dependency": base.file_record(frozen_dependency),
            },
            "protocol": base.file_record(protocol),
            "parent": {
                "artifact_root": str(parent_artifact),
                "registration": base.file_record(parent_path),
                "registration_identity_sha256": parent["identity_sha256"],
                "prior_score_outcomes_used_to_design_representation": True,
                "prior_score_episodes_allowed_in_fit_or_fresh_score": False,
            },
            "inputs": {
                "train_manifest": base.file_record(manifest_path),
                "cache_metadata": base.file_record(cache_dir / "metadata.json"),
                "cache_dir": str(cache_dir),
                "rgb": {
                    **base.file_record(rgb_path, digest=False),
                    "registered_sha256": metadata["rgb_sha256"],
                    "pixel_array_indexed_for_selection": False,
                },
                "actions": {
                    **base.file_record(action_path, digest=False),
                    "registered_sha256": metadata["actions_sha256"],
                    "opened_for_selection": True,
                },
                "preprocessed_root": str(preprocessed_root),
                "raw_root": str(raw_root),
                "camera_type_counts": camera_counts,
                "selected_state_array_opened": False,
                "selected_rgb_pixel_opened": False,
            },
            "split": {
                "source_split": "train",
                "selection_salt": FRESH_SELECTION_SALT,
                "reuses_parent_fit": True,
                "fresh_score_excludes_every_parent_score_episode": True,
                "fit_score_episode_disjoint": True,
                "counts": counts,
                "fit": fit,
                "excluded_parent_score": parent["split"]["score"],
                "untouched_pool": untouched_pool,
                "fresh_score": fresh_score,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "representation": {
                "history_frames": [0, 1, 2, 3, 4],
                "future_target_frames": list(range(5, 13)),
                "history_transitions": ["0->1", "1->2", "2->3", "3->4"],
                "future_transitions": [f"{index}->{index + 1}" for index in range(4, 12)],
                "slot_count": SLOT_COUNT,
                "slot_features": list(SLOT_FEATURES),
                "slot_identity": "causal deterministic nearest-predicted-centroid matching; no backward/future matching",
                "null_slot": "all fourteen values exactly zero",
                "render_hw": list(RENDER_HW),
                "work_hw": list(WORK_HW),
                "robot_union_dilation_px_at_180x320": ROBOT_DILATION_PX,
                "component_event_threshold": COMPONENT_EVENT_THRESHOLD,
                "component_flow_threshold_px": COMPONENT_FLOW_THRESHOLD_PX,
                "component_min_area_px": COMPONENT_MIN_AREA_PX,
                "component_max_area_fraction": COMPONENT_MAX_AREA_FRACTION,
                "contact_max_distance_work_px": CONTACT_MAX_DISTANCE_WORK_PX,
                "max_component_candidates": MAX_COMPONENT_CANDIDATES,
                "tracker_max_missed_transitions": TRACK_MAX_MISSED_TRANSITIONS,
                "tracker_max_distance_normalized": TRACK_MAX_DISTANCE_NORM,
                "history_slot_order_uses_future": False,
                "future_state_or_rgb_is_predictor_input": False,
                "oracle_decoder": "per-slot covariance-clipped Gaussian preserving positive/negative mass and signed mean flow",
            },
            "models": {
                "seed": MODEL_SEED,
                "history_design_width": 4 * SLOT_COUNT * SLOT_DIM + ACTION_COMPONENTS,
                "history_only_equal_width": "history slots plus constant zeros32",
                "history_plus_action": "history slots plus fit-standardized action PCA32",
                "target": "active fit dimensions of eight persistent slot states; no learned target encoder",
                "action_window": list(ACTION_WINDOW),
                "action_pca_components": ACTION_COMPONENTS,
                "ridge_alpha_grid": list(ALPHAS),
                "alpha_selection": "five-fold fit256-only KFold target-standardized MSE",
            },
            "statistics": {
                "bootstrap_unit": "fresh heldout train episode",
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "causal_metrics": list(CAUSAL_METRICS),
                "mandatory_references": list(MANDATORY_REFERENCES),
                "causal_family_size": CAUSAL_FAMILY_SIZE,
                "causal_one_sided_confidence": 1.0 - 0.05 / CAUSAL_FAMILY_SIZE,
                "causal_point_min_percent": MIN_CAUSAL_IMPROVEMENT_PERCENT,
                "reconstruction_family_size": RECONSTRUCTION_FAMILY_SIZE,
                "reconstruction_one_sided_confidence": 1.0 - 0.05 / RECONSTRUCTION_FAMILY_SIZE,
                "reconstruction_point_min_percent": MIN_RECONSTRUCTION_IMPROVEMENT_PERCENT,
                "minimum_favorable_fraction": MIN_FAVORABLE_FRACTION,
                "salience": {
                    "minimum_active_target_fraction": MIN_ACTIVE_TARGET_FRACTION,
                    "minimum_nonempty_score_fraction": MIN_SCORE_NONEMPTY_FRACTION,
                    "minimum_persistent_score_fraction": MIN_SCORE_PERSISTENT_FRACTION,
                },
                "all_causal_reconstruction_and_salience_conditions_required": True,
            },
            "claim_boundary": (
                "Prospective fresh-score train-only slot predictability and relevance. "
                "A pass authorizes only a generator screen; no Wan, validation, protected test, video metric, or control rollout is part of this run."
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
    base.write_json(output / "registration.json", payload)
    return payload


def _transition_candidates(
    gray_source: np.ndarray,
    gray_target: np.ndarray,
    robot_source: np.ndarray,
    robot_target: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    if gray_source.shape != WORK_HW or gray_target.shape != WORK_HW:
        raise ValueError("grayscale geometry differs")
    union = np.logical_or(robot_source, robot_target).astype(np.uint8)
    kernel = np.ones((2 * ROBOT_DILATION_PX + 1,) * 2, dtype=np.uint8)
    excluded_render = cv2.dilate(union, kernel) > 0
    excluded = cv2.resize(
        excluded_render.astype(np.uint8),
        (WORK_HW[1], WORK_HW[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    robot_core = cv2.resize(
        union, (WORK_HW[1], WORK_HW[0]), interpolation=cv2.INTER_NEAREST
    ) > 0
    if not np.any(robot_core):
        raise SlotError("downsampled robot support is empty")
    distance = cv2.distanceTransform((~robot_core).astype(np.uint8), cv2.DIST_L2, 3)
    proximity = np.exp(-distance / CONTACT_SCALE_WORK_PX).astype(np.float32)

    delta = np.asarray(gray_target - gray_source, dtype=np.float32)
    positive = np.clip((delta - PHOTO_THRESHOLD) / PHOTO_SCALE, 0.0, 1.0)
    negative = np.clip((-delta - PHOTO_THRESHOLD) / PHOTO_SCALE, 0.0, 1.0)
    event = positive + negative
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
    flow_magnitude = np.sqrt(np.sum(np.asarray(flow, dtype=np.float32) ** 2, axis=-1))
    support = (
        (event >= COMPONENT_EVENT_THRESHOLD)
        & (flow_magnitude >= COMPONENT_FLOW_THRESHOLD_PX)
        & (~excluded)
    ).astype(np.uint8)
    support = cv2.morphologyEx(
        support, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    support[excluded] = 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support, connectivity=8
    )
    height, width = WORK_HW
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = 2.0 * xx.astype(np.float32) / float(width - 1) - 1.0
    y_norm = 2.0 * yy.astype(np.float32) / float(height - 1) - 1.0
    candidates: list[dict[str, Any]] = []
    accepted_mask = np.zeros(WORK_HW, dtype=bool)
    for component_label in range(1, component_count):
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        if area < COMPONENT_MIN_AREA_PX:
            continue
        if area / float(height * width) > COMPONENT_MAX_AREA_FRACTION:
            continue
        mask = labels == component_label
        minimum_distance = float(np.min(distance[mask]))
        if minimum_distance > CONTACT_MAX_DISTANCE_WORK_PX:
            continue
        weights = (event * proximity * mask).astype(np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-8:
            continue
        cx = float(np.sum(weights * x_norm) / weight_sum)
        cy = float(np.sum(weights * y_norm) / weight_sum)
        dx = x_norm - cx
        dy = y_norm - cy
        cov_xx = float(np.sum(weights * dx * dx) / weight_sum)
        cov_xy = float(np.sum(weights * dx * dy) / weight_sum)
        cov_yy = float(np.sum(weights * dy * dy) / weight_sum)
        normalized_u = np.clip(flow[..., 0] / FLOW_SCALE_PX, -1.0, 1.0)
        normalized_v = np.clip(flow[..., 1] / FLOW_SCALE_PX, -1.0, 1.0)
        contact_score = float(
            np.clip(
                (CONTACT_MAX_DISTANCE_WORK_PX - minimum_distance)
                / CONTACT_MAX_DISTANCE_WORK_PX,
                0.0,
                1.0,
            )
        )
        candidates.append(
            {
                "component_label": component_label,
                "mask": mask,
                "centroid": np.asarray([cx, cy], dtype=np.float32),
                "area_fraction": area / float(height * width),
                "positive_mass": float(np.sum(positive * proximity * mask) / (height * width)),
                "negative_mass": float(np.sum(negative * proximity * mask) / (height * width)),
                "covariance_xx": cov_xx,
                "covariance_xy": cov_xy,
                "covariance_yy": cov_yy,
                "signed_motion_u": float(np.sum(weights * normalized_u) / weight_sum),
                "signed_motion_v": float(np.sum(weights * normalized_v) / weight_sum),
                "contact_score": contact_score,
                "salience_mass": weight_sum,
                "minimum_robot_distance_work_px": minimum_distance,
            }
        )
        accepted_mask |= mask
    candidates.sort(
        key=lambda item: (
            -float(item["salience_mass"]),
            float(item["centroid"][1]),
            float(item["centroid"][0]),
            int(item["component_label"]),
        )
    )
    candidates = candidates[:MAX_COMPONENT_CANDIDATES]
    retained_candidate_mask = np.zeros(WORK_HW, dtype=bool)
    for candidate in candidates:
        retained_candidate_mask |= candidate["mask"]
    cleaned = np.stack(
        (
            positive * proximity,
            negative * proximity,
            event * np.clip(flow[..., 0] / FLOW_SCALE_PX, -1.0, 1.0) * proximity,
            event * np.clip(flow[..., 1] / FLOW_SCALE_PX, -1.0, 1.0) * proximity,
        ),
        axis=0,
    ) * retained_candidate_mask[None]
    audit = {
        "robot_core_fraction": float(robot_core.mean()),
        "robot_excluded_fraction": float(excluded.mean()),
        "threshold_support_fraction": float(support.mean()),
        "accepted_component_count_before_cap": int(
            len(np.unique(labels[accepted_mask])) - (1 if np.any(labels[accepted_mask] == 0) else 0)
        ),
        "candidate_count_after_cap": len(candidates),
        "qualified_event_mass": float(np.sum(event * proximity * retained_candidate_mask)),
        "cleaned_field_energy": float(np.mean(cleaned**2)),
    }
    return candidates, cleaned.astype(np.float32), audit


def track_components(
    observations: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Causally match candidates; no assignment can inspect a later transition."""

    states = np.zeros((len(observations), SLOT_COUNT, SLOT_DIM), dtype=np.float32)
    tracks: list[dict[str, Any] | None] = [None] * SLOT_COUNT
    transition_records: list[dict[str, Any]] = []
    for transition, candidates_raw in enumerate(observations):
        candidates = [dict(item) for item in candidates_raw]
        assignments: dict[int, int] = {}
        assigned_candidates: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for slot, track in enumerate(tracks):
            if track is None:
                continue
            dt = transition - int(track["last_seen"])
            if dt > TRACK_MAX_MISSED_TRANSITIONS + 1:
                continue
            prediction = np.asarray(track["centroid"]) + np.asarray(track["velocity"]) * dt
            for candidate_index, candidate in enumerate(candidates):
                distance = float(
                    np.linalg.norm(np.asarray(candidate["centroid"]) - prediction)
                )
                mass_ratio = abs(
                    math.log(
                        (float(candidate["salience_mass"]) + 1e-6)
                        / (float(track["salience_mass"]) + 1e-6)
                    )
                )
                cost = distance + 0.025 * min(mass_ratio, 4.0)
                if distance <= TRACK_MAX_DISTANCE_NORM and cost <= TRACK_MAX_COST:
                    pairs.append((cost, slot, candidate_index))
        for _, slot, candidate_index in sorted(pairs):
            if slot in assignments or candidate_index in assigned_candidates:
                continue
            assignments[slot] = candidate_index
            assigned_candidates.add(candidate_index)

        free_slots = [
            slot
            for slot, track in enumerate(tracks)
            if slot not in assignments
            and (
                track is None
                or transition - int(track["last_seen"]) > TRACK_MAX_MISSED_TRANSITIONS + 1
            )
        ]
        for candidate_index in range(len(candidates)):
            if candidate_index in assigned_candidates or not free_slots:
                continue
            slot = free_slots.pop(0)
            assignments[slot] = candidate_index
            assigned_candidates.add(candidate_index)

        retained_mass = 0.0
        for slot, candidate_index in sorted(assignments.items()):
            candidate = candidates[candidate_index]
            previous = tracks[slot]
            matched_identity = (
                previous is not None
                and transition - int(previous["last_seen"])
                <= TRACK_MAX_MISSED_TRANSITIONS + 1
            )
            if matched_identity:
                dt = transition - int(previous["last_seen"])
                velocity = (
                    np.asarray(candidate["centroid"]) - np.asarray(previous["centroid"])
                ) / float(dt)
                previous_contact = float(previous["contact_score"])
                sightings = int(previous["sightings"]) + 1
            else:
                velocity = np.zeros(2, dtype=np.float32)
                previous_contact = 0.0
                sightings = 1
            contact_score = float(candidate["contact_score"])
            vector = np.asarray(
                [
                    1.0,
                    float(candidate["centroid"][0]),
                    float(candidate["centroid"][1]),
                    float(candidate["area_fraction"]),
                    float(candidate["positive_mass"]),
                    float(candidate["negative_mass"]),
                    float(candidate["covariance_xx"]),
                    float(candidate["covariance_xy"]),
                    float(candidate["covariance_yy"]),
                    float(candidate["signed_motion_u"]),
                    float(candidate["signed_motion_v"]),
                    contact_score,
                    max(0.0, contact_score - previous_contact),
                    min(sightings / 12.0, 1.0),
                ],
                dtype=np.float32,
            )
            states[transition, slot] = vector
            tracks[slot] = {
                "centroid": np.asarray(candidate["centroid"], dtype=np.float32),
                "velocity": np.asarray(velocity, dtype=np.float32),
                "salience_mass": float(candidate["salience_mass"]),
                "contact_score": contact_score,
                "last_seen": transition,
                "sightings": sightings,
            }
            retained_mass += float(candidate["salience_mass"])
        total_mass = float(sum(float(item["salience_mass"]) for item in candidates))
        transition_records.append(
            {
                "transition": transition,
                "candidate_count": len(candidates),
                "assigned_count": len(assignments),
                "assigned_slots": sorted(assignments),
                "retained_candidate_mass_fraction": (
                    retained_mass / total_mass if total_mass > 0.0 else 1.0
                ),
                "future_observations_consulted": False,
            }
        )
    if not np.isfinite(states).all():
        raise SlotError("slot state contains non-finite values")
    return states, transition_records


def clip_slot_state(
    frames: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if frames.shape != (base.SAMPLE_SIZE, 3, 180, 960):
        raise SlotError(f"clip RGB geometry differs: {frames.shape}")
    if masks.shape != (base.SAMPLE_SIZE, *RENDER_HW):
        raise SlotError(f"clip mask geometry differs: {masks.shape}")
    gray = [base._rgb_gray(frames[index]) for index in range(base.SAMPLE_SIZE)]
    observations: list[list[dict[str, Any]]] = []
    fields: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for transition in range(base.SAMPLE_SIZE - 1):
        candidates, field, audit = _transition_candidates(
            gray[transition], gray[transition + 1], masks[transition], masks[transition + 1]
        )
        observations.append(candidates)
        fields.append(field)
        audits.append(audit)
    states, tracking = track_components(observations)
    return states[:4], states[4:12], np.stack(fields[4:12]), audits, tracking


_GRID_Y, _GRID_X = np.mgrid[0 : WORK_HW[0], 0 : WORK_HW[1]]
_GRID_X = 2.0 * _GRID_X.astype(np.float64) / float(WORK_HW[1] - 1) - 1.0
_GRID_Y = 2.0 * _GRID_Y.astype(np.float64) / float(WORK_HW[0] - 1) - 1.0


def decode_slot_sequence(slots: np.ndarray) -> np.ndarray:
    values = np.asarray(slots, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (SLOT_COUNT, SLOT_DIM):
        raise ValueError(f"slot decoder geometry differs: {values.shape}")
    decoded = np.zeros((len(values), 4, *WORK_HW), dtype=np.float64)
    pixels = float(WORK_HW[0] * WORK_HW[1])
    for transition in range(len(values)):
        for slot in range(SLOT_COUNT):
            state = values[transition, slot]
            presence = float(np.clip(state[PRESENCE_INDEX], 0.0, 1.0))
            if presence <= 1e-6:
                continue
            centroid = np.clip(state[1:3], -1.25, 1.25)
            covariance = np.asarray(
                [[state[6], state[7]], [state[7], state[8]]], dtype=np.float64
            )
            covariance = np.nan_to_num(covariance, nan=DECODER_MIN_VARIANCE)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.clip(
                eigenvalues, DECODER_MIN_VARIANCE, DECODER_MAX_VARIANCE
            )
            covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            inverse = np.linalg.inv(covariance)
            dx = _GRID_X - centroid[0]
            dy = _GRID_Y - centroid[1]
            exponent = -0.5 * (
                inverse[0, 0] * dx * dx
                + 2.0 * inverse[0, 1] * dx * dy
                + inverse[1, 1] * dy * dy
            )
            gaussian = np.exp(np.clip(exponent, -80.0, 0.0))
            gaussian_sum = float(gaussian.sum())
            if gaussian_sum <= 0.0:
                continue
            gaussian /= gaussian_sum
            positive_mass = float(np.clip(state[4], 0.0, 1.0))
            negative_mass = float(np.clip(state[5], 0.0, 1.0))
            event_mass = positive_mass + negative_mass
            u = float(np.clip(state[9], -1.0, 1.0))
            v = float(np.clip(state[10], -1.0, 1.0))
            decoded[transition, 0] += presence * positive_mass * pixels * gaussian
            decoded[transition, 1] += presence * negative_mass * pixels * gaussian
            decoded[transition, 2] += presence * event_mass * u * pixels * gaussian
            decoded[transition, 3] += presence * event_mass * v * pixels * gaussian
    return decoded.astype(np.float32)


def _selected_clip_arrays(
    *,
    item: Mapping[str, Any],
    row: Mapping[str, Any],
    rgb_cache: np.ndarray,
    action_cache: np.ndarray,
    renderer_context: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[float]]:
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
            raise SlotError(f"state file lacks required arrays: {state_path}")
        frame_indices = np.asarray(row["frame_indices"], dtype=np.int64)
        boundaries = np.concatenate(
            (
                np.asarray(state["joint_states"][frame_indices], dtype=np.float32),
                np.asarray(state["gripper_states"][frame_indices], dtype=np.float32),
            ),
            axis=1,
        )
        start = int(row["start"])
        stop = start + base.SAMPLE_SIZE * base.CHUNK_SIZE
        native_actions = np.concatenate(
            (
                np.asarray(state["joint_actions"][start:stop], dtype=np.float32),
                np.asarray(state["gripper_actions"][start:stop], dtype=np.float32),
            ),
            axis=1,
        ).reshape(base.SAMPLE_SIZE, base.CHUNK_SIZE, base.ACTION_DIM)
        timestamps = np.asarray(state["frame_ts"][frame_indices], dtype=np.int64)
    cached_actions = np.asarray(
        action_cache[index, ..., : base.ACTION_DIM], dtype=np.float32
    )
    if not np.array_equal(native_actions, cached_actions):
        raise SlotError(f"raw/cache action mismatch at manifest row {index}")
    frames = np.asarray(rgb_cache[index], dtype=np.float32)
    masks, latencies = base.render_robot_masks(
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
    history, future, future_fields, transition_audits, tracking_audits = clip_slot_state(
        frames, masks
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "role": item["role"],
        "manifest_index": index,
        "clip_id": item["clip_id"],
        "episode_dir": item["episode_dir"],
        "states_npz": base.file_record(state_path),
        "frame_indices": frame_indices.tolist(),
        "frame_timestamps_ns": timestamps.tolist(),
        "transition_audits": transition_audits,
        "tracking_audits": tracking_audits,
        "source_split": "train",
        "history_slot_order_uses_future": False,
        "future_rgb_used_only_for_target": True,
        "future_state_used_only_for_target_mask": True,
        "validation_accessed": False,
        "protected_test_accessed": False,
    }
    return history, future, future_fields, cached_actions, provenance, latencies


def _target_salience(
    fit_target: np.ndarray,
    score_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, bool]]:
    flat = fit_target.reshape(FIT_CLIPS, -1)
    raw_std = flat.std(axis=0, dtype=np.float64)
    active = raw_std > 1e-6
    presence = score_target[..., PRESENCE_INDEX] > 0.5
    nonempty = np.any(presence, axis=(1, 2))
    adjacent = presence[:, 1:] & presence[:, :-1]
    persistent = np.any(adjacent, axis=(1, 2))
    metrics = {
        "active_fit_target_dimensions": int(active.sum()),
        "total_target_dimensions": int(active.size),
        "active_fit_target_fraction": float(active.mean()),
        "score_nonempty_fraction": float(nonempty.mean()),
        "score_nonempty_count": int(nonempty.sum()),
        "score_persistent_fraction": float(persistent.mean()),
        "score_persistent_count": int(persistent.sum()),
        "score_mean_present_slot_transitions": float(presence.sum(axis=(1, 2)).mean()),
    }
    checks = {
        "active_target_fraction_at_least_50_percent": float(active.mean())
        >= MIN_ACTIVE_TARGET_FRACTION,
        "score_nonempty_fraction_at_least_90_percent": float(nonempty.mean())
        >= MIN_SCORE_NONEMPTY_FRACTION,
        "score_persistent_fraction_at_least_60_percent": float(persistent.mean())
        >= MIN_SCORE_PERSISTENT_FRACTION,
    }
    return active, metrics, {**checks, "passed": all(checks.values())}


def analyze_errors(
    errors: Mapping[str, np.ndarray],
    salience_gate: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = base.common_bootstrap_indices(
        SCORE_CLIPS, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    causal_effects: dict[str, Any] = {}
    causal_gates: dict[str, Any] = {}
    for metric in CAUSAL_METRICS:
        candidate = np.asarray(errors[f"{metric}_error_aligned"])
        for reference in MANDATORY_REFERENCES:
            label = f"{metric}:aligned_vs_{reference}"
            effect = base.paired_relative_effect(
                np.asarray(errors[f"{metric}_error_{reference}"]),
                candidate,
                bootstrap,
                family_size=CAUSAL_FAMILY_SIZE,
            )
            causal_effects[label] = effect
            causal_gates[label] = base.effect_gate(
                effect, minimum_percent=MIN_CAUSAL_IMPROVEMENT_PERCENT
            )
    reconstruction_effects: dict[str, Any] = {}
    reconstruction_gates: dict[str, Any] = {}
    for reference in ("fit_mean", "raw_zero"):
        label = f"reconstruction:oracle_vs_{reference}"
        effect = base.paired_relative_effect(
            np.asarray(errors[f"reconstruction_error_{reference}"]),
            np.asarray(errors["reconstruction_error_oracle"]),
            bootstrap,
            family_size=RECONSTRUCTION_FAMILY_SIZE,
        )
        reconstruction_effects[label] = effect
        reconstruction_gates[label] = base.effect_gate(
            effect, minimum_percent=MIN_RECONSTRUCTION_IMPROVEMENT_PERCENT
        )
    gates = {
        "causal": causal_gates,
        "reconstruction": reconstruction_gates,
        "target_salience": dict(salience_gate),
    }
    gates["causal_all_passed"] = all(item["passed"] for item in causal_gates.values())
    gates["reconstruction_all_passed"] = all(
        item["passed"] for item in reconstruction_gates.values()
    )
    gates["all_passed"] = bool(
        gates["causal_all_passed"]
        and gates["reconstruction_all_passed"]
        and salience_gate["passed"]
    )
    return {
        "causal": causal_effects,
        "reconstruction": reconstruction_effects,
    }, gates


def _predict_slots(
    *,
    model: Any,
    history: np.ndarray,
    action_pc: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    target_active: np.ndarray,
) -> np.ndarray:
    design = np.concatenate((history, action_pc), axis=1)
    prediction_z = model.predict((design - input_mean) / input_std).astype(np.float32)
    full = np.broadcast_to(target_mean, (len(history), len(target_mean))).copy()
    full[:, target_active] = (
        prediction_z * target_std[target_active] + target_mean[target_active]
    )
    return full.reshape(len(history), 8, SLOT_COUNT, SLOT_DIM).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import mujoco
        from sklearn.linear_model import Ridge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("run requires mujoco and scikit-learn") from exc

    output = args.output.resolve()
    if (output / "analysis.json").exists() or (output / "run_complete.json").exists():
        raise FileExistsError("completed output is immutable")
    registration = base.read_json(output / "registration.json")
    base.validate_identity(registration, "registration")
    source = Path(__file__).resolve()
    dependency = Path(base.__file__).resolve()
    for path, expected, label in (
        (source, registration["source"]["sha256"], "source"),
        (dependency, registration["source"]["base_dependency"]["sha256"], "base dependency"),
        (output / "frozen_object_contact_slot_stage0.py", registration["source"]["frozen_copy"]["sha256"], "frozen source"),
        (output / "frozen_interaction_event_bottleneck_stage0.py", registration["source"]["frozen_base_dependency"]["sha256"], "frozen dependency"),
    ):
        if base.sha256_file(path) != expected:
            raise SlotError(f"{label} bytes differ from registration")

    rows = base.read_jsonl(Path(registration["inputs"]["train_manifest"]["path"]))
    base.validate_train_manifest(rows)
    _, rgb_cache, action_cache, _, _ = base.load_cache(
        Path(registration["inputs"]["cache_dir"])
    )
    fit_items = registration["split"]["fit"]
    score_items = registration["split"]["fresh_score"]
    selected = [*fit_items, *score_items]

    official_root = args.official_abc_root.resolve()
    official_commit = base.git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise SlotError(f"official ABC commit differs: {official_commit}")
    scene_path = official_root / OFFICIAL_SCENE_RELATIVE
    asset_root = official_root / OFFICIAL_ASSET_RELATIVE
    robot_xml = build_robot_only_xml(scene_path, asset_root)
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    if camera_id < 0:
        raise SlotError("official model lacks top camera")
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
    fit_target: list[np.ndarray] = []
    fit_fields: list[np.ndarray] = []
    fit_actions: list[np.ndarray] = []
    score_history: list[np.ndarray] = []
    score_target: list[np.ndarray] = []
    score_fields: list[np.ndarray] = []
    score_actions: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    started = time.time()
    try:
        for position, item in enumerate(selected):
            result = _selected_clip_arrays(
                item=item,
                row=rows[int(item["manifest_index"])],
                rgb_cache=rgb_cache,
                action_cache=action_cache,
                renderer_context=renderer_context,
            )
            history, target, fields, actions, record, latencies = result
            provenance.append(record)
            render_latencies.extend(latencies[2:] if len(latencies) > 2 else latencies)
            if item["role"] == "predictor_fit":
                fit_history.append(history)
                fit_target.append(target)
                fit_fields.append(fields)
                fit_actions.append(actions)
            elif item["role"] == "fresh_heldout_score":
                score_history.append(history)
                score_target.append(target)
                score_fields.append(fields)
                score_actions.append(actions)
            else:
                raise SlotError(f"unexpected role {item['role']}")
            if position == 0 or (position + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "event": "object_slot_extraction_progress",
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
    fit_y = np.stack(fit_target)
    fit_field = np.stack(fit_fields)
    fit_a_full = np.stack(fit_actions)
    score_h = np.stack(score_history)
    score_y = np.stack(score_target)
    score_field = np.stack(score_fields)
    score_a_full = np.stack(score_actions)
    expected_history = (FIT_CLIPS, 4, SLOT_COUNT, SLOT_DIM)
    expected_target = (FIT_CLIPS, 8, SLOT_COUNT, SLOT_DIM)
    if fit_h.shape != expected_history or fit_y.shape != expected_target:
        raise SlotError(f"fit slot geometry differs: {fit_h.shape}/{fit_y.shape}")
    if score_y.shape != (SCORE_CLIPS, 8, SLOT_COUNT, SLOT_DIM):
        raise SlotError(f"score slot geometry differs: {score_y.shape}")

    target_active, salience_metrics, salience_gate = _target_salience(fit_y, score_y)
    if int(target_active.sum()) < 2:
        raise SlotError("too few active target coordinates to fit")
    fit_history_flat = fit_h.reshape(FIT_CLIPS, -1)
    score_history_flat = score_h.reshape(SCORE_CLIPS, -1)
    fit_target_flat = fit_y.reshape(FIT_CLIPS, -1)
    score_target_flat = score_y.reshape(SCORE_CLIPS, -1)
    target_mean, target_std = base._standardizer(fit_target_flat)
    fit_target_z = (
        (fit_target_flat[:, target_active] - target_mean[target_active])
        / target_std[target_active]
    )

    fit_action_window = fit_a_full[:, ACTION_WINDOW[0] : ACTION_WINDOW[1], :, : base.ACTION_DIM]
    score_action_window = score_a_full[:, ACTION_WINDOW[0] : ACTION_WINDOW[1], :, : base.ACTION_DIM]
    action_pca, fit_action_pc, action_mean, action_std, action_active = base._fit_action_pca(
        fit_action_window
    )
    score_action_pc = base._transform_action_pca(
        action_pca, score_action_window, action_mean, action_std, action_active
    )
    zeros_fit = np.zeros_like(fit_action_pc)
    history_design = np.concatenate((fit_history_flat, zeros_fit), axis=1)
    action_design = np.concatenate((fit_history_flat, fit_action_pc), axis=1)
    history_input_mean, history_input_std = base._standardizer(history_design)
    action_input_mean, action_input_std = base._standardizer(action_design)
    history_x = (history_design - history_input_mean) / history_input_std
    action_x = (action_design - action_input_mean) / action_input_std
    history_alpha, history_cv = base._choose_alpha(history_x, fit_target_z)
    action_alpha, action_cv = base._choose_alpha(action_x, fit_target_z)
    history_model = Ridge(alpha=history_alpha, fit_intercept=True, solver="cholesky").fit(
        history_x, fit_target_z
    )
    action_model = Ridge(alpha=action_alpha, fit_intercept=True, solver="cholesky").fit(
        action_x, fit_target_z
    )

    score_index_position = {
        int(item["manifest_index"]): position for position, item in enumerate(score_items)
    }
    donor_positions = np.asarray(
        [score_index_position[int(item["donor_manifest_index"])] for item in score_items],
        dtype=np.int64,
    )
    fit_action_mean_raw = fit_action_window.reshape(FIT_CLIPS, -1).mean(axis=0).reshape(
        1, 8, base.CHUNK_SIZE, base.ACTION_DIM
    )
    mean_score_raw = np.repeat(fit_action_mean_raw, SCORE_CLIPS, axis=0)
    shifted_pc = {
        label: base._transform_action_pca(
            action_pca,
            score_a_full[:, start:stop, :, : base.ACTION_DIM],
            action_mean,
            action_std,
            action_active,
        )
        for label, (start, stop) in SHIFT_WINDOWS.items()
    }
    control_action_pc = {
        "aligned": score_action_pc,
        "episode_shuffled": score_action_pc[donor_positions],
        "train_mean": base._transform_action_pca(
            action_pca, mean_score_raw, action_mean, action_std, action_active
        ),
        "raw_zero": base._transform_action_pca(
            action_pca,
            np.zeros_like(score_action_window),
            action_mean,
            action_std,
            action_active,
        ),
        **shifted_pc,
    }
    predictions: dict[str, np.ndarray] = {
        "history_only": _predict_slots(
            model=history_model,
            history=score_history_flat,
            action_pc=np.zeros_like(score_action_pc),
            input_mean=history_input_mean,
            input_std=history_input_std,
            target_mean=target_mean,
            target_std=target_std,
            target_active=target_active,
        )
    }
    for label, action_pc in control_action_pc.items():
        predictions[label] = _predict_slots(
            model=action_model,
            history=score_history_flat,
            action_pc=action_pc,
            input_mean=action_input_mean,
            input_std=action_input_std,
            target_mean=target_mean,
            target_std=target_std,
            target_active=target_active,
        )

    decoded_predictions = {
        label: np.stack([decode_slot_sequence(clip) for clip in prediction])
        for label, prediction in predictions.items()
    }
    oracle_decode = np.stack([decode_slot_sequence(clip) for clip in score_y])
    fit_mean_field = np.broadcast_to(
        fit_field.mean(axis=0, dtype=np.float64).astype(np.float32), score_field.shape
    )
    errors: dict[str, np.ndarray] = {}
    for label, prediction in predictions.items():
        prediction_flat = prediction.reshape(SCORE_CLIPS, -1)
        errors[f"slot_error_{label}"] = np.mean(
            (
                (prediction_flat[:, target_active] - score_target_flat[:, target_active])
                / target_std[target_active]
            )
            ** 2,
            axis=1,
        ).astype(np.float64)
        errors[f"field_error_{label}"] = np.mean(
            (decoded_predictions[label] - score_field) ** 2, axis=(1, 2, 3, 4)
        ).astype(np.float64)
    errors["reconstruction_error_oracle"] = np.mean(
        (oracle_decode - score_field) ** 2, axis=(1, 2, 3, 4)
    ).astype(np.float64)
    errors["reconstruction_error_fit_mean"] = np.mean(
        (fit_mean_field - score_field) ** 2, axis=(1, 2, 3, 4)
    ).astype(np.float64)
    errors["reconstruction_error_raw_zero"] = np.mean(
        score_field**2, axis=(1, 2, 3, 4)
    ).astype(np.float64)
    effects, gates = analyze_errors(errors, salience_gate)
    decision = (
        "GO_FOR_OBJECT_SLOT_GENERATOR_SCREEN"
        if gates["all_passed"]
        else "STOP_OBJECT_CONTACT_SLOT"
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
                "fresh_score": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    arrays_path = output / "slot_model_predictions.npz"
    np.savez_compressed(
        arrays_path,
        fit_indices=np.asarray([item["manifest_index"] for item in fit_items], dtype=np.int64),
        score_indices=np.asarray([item["manifest_index"] for item in score_items], dtype=np.int64),
        donor_positions=donor_positions,
        fit_history=fit_h,
        fit_target=fit_y,
        score_history=score_h,
        score_target=score_y,
        score_target_fields=score_field.astype(np.float16),
        target_mean=target_mean,
        target_std=target_std,
        target_active=target_active,
        action_mean=action_mean,
        action_std=action_std,
        action_active=action_active,
        action_pca_mean=action_pca.mean_,
        action_pca_components=action_pca.components_,
        action_pca_explained_variance_ratio=action_pca.explained_variance_ratio_,
        history_input_mean=history_input_mean,
        history_input_std=history_input_std,
        action_input_mean=action_input_mean,
        action_input_std=action_input_std,
        history_model_coef=history_model.coef_,
        history_model_intercept=history_model.intercept_,
        action_model_coef=action_model.coef_,
        action_model_intercept=action_model.intercept_,
        **{f"prediction_{label}": value for label, value in predictions.items()},
        **errors,
    )

    transition_audits = [
        audit for record in provenance for audit in record["transition_audits"]
    ]
    tracking_audits = [
        audit for record in provenance for audit in record["tracking_audits"]
    ]
    latency = np.asarray(render_latencies, dtype=np.float64)
    analysis = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_slot_stage0_analysis",
            "created_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "parent_registration_identity_sha256": registration["parent"]["registration_identity_sha256"],
            "decision": decision,
            "interpretation": (
                "fresh-score object/contact slots cleared every causal, timing, salience, and reconstruction gate"
                if gates["all_passed"]
                else "at least one fresh-score causal, timing, salience, or reconstruction gate failed; no generator integration is authorized"
            ),
            "population": {
                **registration["split"]["counts"],
                "source_split": "train",
                "fit_score_episode_disjoint": True,
                "every_parent_score_episode_excluded": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "target_salience": salience_metrics,
            "models": {
                "history_alpha": history_alpha,
                "action_alpha": action_alpha,
                "history_five_fold_cv": history_cv,
                "action_five_fold_cv": action_cv,
                "action_pca_variance_retained": float(
                    action_pca.explained_variance_ratio_.sum()
                ),
                "active_target_dimensions": int(target_active.sum()),
                "prediction_action_sensitivity": {
                    label: float(
                        np.sqrt(np.mean((predictions["aligned"] - predictions[label]) ** 2))
                    )
                    for label in MANDATORY_REFERENCES
                    if label != "history_only"
                },
            },
            "metrics": {
                "mean_errors": {name: float(values.mean()) for name, values in errors.items()},
                "effects": effects,
                "gates": gates,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "causal_family_size": CAUSAL_FAMILY_SIZE,
                "reconstruction_family_size": RECONSTRUCTION_FAMILY_SIZE,
            },
            "slot_diagnostics": {
                "transition_rows": len(transition_audits),
                "robot_excluded_fraction_mean": float(
                    np.mean([item["robot_excluded_fraction"] for item in transition_audits])
                ),
                "candidate_count_after_cap_mean": float(
                    np.mean([item["candidate_count_after_cap"] for item in transition_audits])
                ),
                "qualified_event_mass_mean": float(
                    np.mean([item["qualified_event_mass"] for item in transition_audits])
                ),
                "assigned_slot_count_mean": float(
                    np.mean([item["assigned_count"] for item in tracking_audits])
                ),
                "retained_candidate_mass_fraction_mean": float(
                    np.mean(
                        [item["retained_candidate_mass_fraction"] for item in tracking_audits]
                    )
                ),
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
                "frozen_source": base.file_record(output / "frozen_object_contact_slot_stage0.py"),
                "frozen_base_dependency": base.file_record(output / "frozen_interaction_event_bottleneck_stage0.py"),
                "input_provenance": base.file_record(provenance_path),
                "per_clip_metrics": base.file_record(per_clip_path),
                "slot_model_predictions": base.file_record(arrays_path),
            },
            "claim_boundary": (
                "Fresh heldout train-episode linear predictability and Gaussian reconstruction of deterministic masked object/contact slots. "
                "No Wan, video metric, validation, protected test, semantic label, force label, or control rollout was evaluated."
            ),
            "limitations": [
                "single fresh train-only score split and model seed",
                "connected motion components are observational proxies, not semantic instance or physical-contact labels",
                "causal greedy matching can recycle a slot after more than one missed transition",
                "Gaussian decoding cannot preserve irregular component boundaries",
                "bootstrap quantifies score-episode sampling, not retraining variance",
                "nominal camera calibration and rendered robot masks have residual error",
            ],
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "analysis.json", analysis)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_slot_stage0_complete",
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
        ("run_complete", complete),
    ):
        base.validate_identity(document, name)
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise SlotError("analysis-registration binding differs")
    if complete["registration_identity_sha256"] != registration["identity_sha256"]:
        raise SlotError("completion-registration binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise SlotError("completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise SlotError("completion decision differs")
    for path, expected, label in (
        (Path(__file__).resolve(), registration["source"]["sha256"], "source"),
        (Path(base.__file__).resolve(), registration["source"]["base_dependency"]["sha256"], "base dependency"),
        (output / "frozen_object_contact_slot_stage0.py", registration["source"]["frozen_copy"]["sha256"], "frozen source"),
        (output / "frozen_interaction_event_bottleneck_stage0.py", registration["source"]["frozen_base_dependency"]["sha256"], "frozen dependency"),
    ):
        if base.sha256_file(path) != expected:
            raise SlotError(f"audit {label} hash differs")
    for artifact in analysis["artifacts"].values():
        path = Path(artifact["path"])
        if base.file_record(path) != artifact:
            raise SlotError(f"artifact record differs: {path}")

    fit_indexes = {int(item["manifest_index"]) for item in registration["split"]["fit"]}
    prior_score_indexes = {
        int(item["manifest_index"])
        for item in registration["split"]["excluded_parent_score"]
    }
    fresh_indexes = {
        int(item["manifest_index"]) for item in registration["split"]["fresh_score"]
    }
    if len(fit_indexes) != FIT_CLIPS or len(fresh_indexes) != SCORE_CLIPS:
        raise SlotError("registered split counts differ")
    if fit_indexes & fresh_indexes or prior_score_indexes & (fit_indexes | fresh_indexes):
        raise SlotError("fresh-score exclusion invariant differs")

    arrays = np.load(output / "slot_model_predictions.npz", allow_pickle=False)
    errors = {
        name: np.asarray(arrays[name])
        for name in arrays.files
        if name.startswith("slot_error_")
        or name.startswith("field_error_")
        or name.startswith("reconstruction_error_")
    }
    fit_target = np.asarray(arrays["fit_target"])
    score_target = np.asarray(arrays["score_target"])
    active, salience_metrics, salience_gate = _target_salience(fit_target, score_target)
    if not np.array_equal(active, np.asarray(arrays["target_active"])):
        raise SlotError("target active coordinates differ")
    effects, gates = analyze_errors(errors, salience_gate)
    if effects != analysis["metrics"]["effects"] or gates != analysis["metrics"]["gates"]:
        raise SlotError("recomputed effects or gates differ")
    if salience_metrics != analysis["target_salience"]:
        raise SlotError("recomputed target salience differs")

    provenance = base.read_jsonl(output / "input_provenance.jsonl")
    per_clip = base.read_jsonl(output / "per_clip_metrics.jsonl")
    if len(provenance) != FIT_CLIPS + SCORE_CLIPS or len(per_clip) != SCORE_CLIPS:
        raise SlotError("provenance or score row count differs")
    false_flags = 0
    for document in (registration, analysis, complete, provenance, per_clip):
        false_flags += base.require_false_flags(document)
    expected_decision = (
        "GO_FOR_OBJECT_SLOT_GENERATOR_SCREEN"
        if gates["all_passed"]
        else "STOP_OBJECT_CONTACT_SLOT"
    )
    if analysis["decision"] != expected_decision:
        raise SlotError("decision differs from recomputed gates")
    return base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_slot_stage0_audit",
            "audited_at_utc": now(),
            "status": "audit_passed",
            "decision": analysis["decision"],
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "provenance_rows": len(provenance),
            "score_rows": len(per_clip),
            "fresh_score_overlap_with_prior_score": 0,
            "recomputed_causal_effects": len(effects["causal"]),
            "recomputed_reconstruction_effects": len(effects["reconstruction"]),
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
        default=Path("docs/experiments/OBJECT_CONTACT_SLOT_STAGE0_PROTOCOL.md"),
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
