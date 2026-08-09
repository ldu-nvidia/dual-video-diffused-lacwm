#!/usr/bin/env python3
"""Prospective renderer-endpoint-unopened-55 spatial confirmation.

The workflow is deliberately split into five immutable transitions:

``register -> prepare -> evaluate -> analyze -> audit``.

``prepare`` can only construct causal trajectories.  It seals their hash before
``evaluate`` is allowed to open future measured state or RGB.  The fitted
predictors are copied from, and cryptographically bound to, the completed
trajectory-consistent fresh24 study; this module never fits or tunes a model.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import corrected_renderer_attribution as base
from tools import trajectory_consistent_renderer_gate as predecessor
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
EXPECTED_D405_POOL = 103
GATE0C_CLIPS = 24
FRESH24_CLIPS = 24
RENDERER_UNOPENED_CLIPS = 55
VPM_COLLISION_CLIPS = 26
SCORE_CLIPS = 55
MOTION_STRATA = 3
RENDERER_UNOPENED_STRATUM_COUNTS = (19, 18, 18)
EXPECTED_STRATUM_COUNTS = RENDERER_UNOPENED_STRATUM_COUNTS
ACTION_DIM = 14
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
HISTORY_FRAME_COUNT = 5
HISTORY_CHUNKS = tuple(range(4))
FUTURE_CHUNKS = tuple(range(4, 12))

BOOTSTRAP_SEED = 20260809
BOOTSTRAP_SAMPLES = 100_000
HOLM_ALPHA = 0.05
MIN_FAVORABLE_CLIP_FRACTION = 0.60
FLOW_MIN_RELATIVE_IMPROVEMENT_PERCENT = 5.0
SPATIAL_NI_MARGINS = {
    "silhouette_boundary_chamfer_px": 0.25,
    "rgb_robot_band_chamfer_px": 0.25,
    "silhouette_iou": 0.005,
}
ROBOT_BAND_PX = predecessor.ROBOT_BAND_PX
FLOW_PIXEL_STRIDE = predecessor.FLOW_PIXEL_STRIDE
FLOW_MIN_ORACLE_MOTION_PX = predecessor.FLOW_MIN_ORACLE_MOTION_PX

PRIOR_REGISTRATION_IDENTITY = (
    "9cc556aba53d1defb69b0049dab67d12a3991decb917015bba5153c16cb8c2b1"
)
PRIOR_PREPARATION_IDENTITY = (
    "11444ee94ae6707869f44f00409386701c93b886d650ebc928bc4edec4f41a3a"
)
PRIOR_COMPLETION_IDENTITY = (
    "715fc288ca3e5526cda3a5bb3329db650aa8ed15a61d2959be2349dd3df6dcfc"
)
PRIOR_GATE0C_REGISTRATION_IDENTITY = (
    "494ae93b5b87ca4ed19e1497f6cdd64d51dba0f527ac51a83a8eff3eaa676052"
)
PRIOR_MODEL_SHA256 = (
    "8602fda20b3f0c8365c69efcc48e363a8e8f93666725e1af8665a18d03e80688"
)
PRIOR_RECURRENT_ALPHA = 0.1
PRIOR_RECURRENT_GAIN = 1.0

CAUSAL_ARMS = (
    "raw_command",
    "absolute_ridge",
    "recurrent_delta",
    "recurrent_delta_shuffled",
)
ARMS = (*CAUSAL_ARMS, "measured_oracle")
LOWER_METRICS = (
    "robot_flow_epe_px",
    "silhouette_boundary_chamfer_px",
    "rgb_robot_band_chamfer_px",
)
FLOW_LABELS = (
    "recurrent_delta_vs_raw_command:robot_flow_epe_px",
    "recurrent_delta_vs_absolute_ridge:robot_flow_epe_px",
    "recurrent_delta_vs_recurrent_delta_shuffled:robot_flow_epe_px",
)
SPATIAL_LABELS = tuple(
    f"recurrent_delta_vs_{reference}:{metric}"
    for reference in ("raw_command", "absolute_ridge")
    for metric in (
        "silhouette_boundary_chamfer_px",
        "rgb_robot_band_chamfer_px",
        "silhouette_iou",
    )
)


class ConfirmationError(RuntimeError):
    """Raised when a prospective, causal, or artifact contract differs."""


def _file_bytes_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _resolve_record(root: Path, record: dict[str, Any], *fallback_parts: str) -> Path:
    path = Path(record["path"])
    if path.is_file():
        return path
    candidates = [root / path.name]
    if fallback_parts:
        candidates.insert(0, root.joinpath(*fallback_parts, path.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if base.sha256_file(source) != base.sha256_file(destination):
        raise ConfirmationError(f"copy differs: {source} -> {destination}")
    return base.file_record(destination)


def _validate_prior(prior_output: Path) -> dict[str, Any]:
    documents = {}
    for name in ("registration.json", "preparation.json", "run_complete.json"):
        path = prior_output / name
        document = base.read_json(path)
        base.validate_identity(document, str(path))
        documents[name] = document
    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    complete = documents["run_complete.json"]
    expected = {
        "registration.json": PRIOR_REGISTRATION_IDENTITY,
        "preparation.json": PRIOR_PREPARATION_IDENTITY,
        "run_complete.json": PRIOR_COMPLETION_IDENTITY,
    }
    for name, identity in expected.items():
        if documents[name].get("identity_sha256") != identity:
            raise ConfirmationError(f"prior {name} identity differs")
    if registration.get("kind") != "trajectory_consistent_renderer_gate_registration":
        raise ConfirmationError("prior registration kind differs")
    if preparation.get("kind") != "trajectory_consistent_renderer_gate_preparation":
        raise ConfirmationError("prior preparation kind differs")
    if complete.get("kind") != "trajectory_consistent_renderer_gate_complete":
        raise ConfirmationError("prior completion kind differs")
    if complete.get("status") != "completed":
        raise ConfirmationError("prior trajectory gate is incomplete")
    if preparation.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ConfirmationError("prior preparation-registration binding differs")
    if complete.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ConfirmationError("prior completion-registration binding differs")
    model_path = _resolve_record(
        prior_output, preparation["artifacts"]["model_state_and_predictions"]
    )
    if base.sha256_file(model_path) != PRIOR_MODEL_SHA256:
        raise ConfirmationError("prior frozen model hash differs")
    gate0c_path = prior_output / "prior_gate0c_registration.json"
    if not gate0c_path.is_file():
        gate0c_path = _resolve_record(
            prior_output, registration["prior_gate0c"]["registration"]
        )
    gate0c = base.read_json(gate0c_path)
    base.validate_identity(gate0c, str(gate0c_path))
    if gate0c.get("identity_sha256") != PRIOR_GATE0C_REGISTRATION_IDENTITY:
        raise ConfirmationError("prior Gate-0c registration identity differs")
    return {
        "registration": registration,
        "preparation": preparation,
        "complete": complete,
        "model_path": model_path,
        "gate0c": gate0c,
        "gate0c_path": gate0c_path,
    }


def _stratify_fresh79(
    eligible: list[dict[str, Any]], gate0c_ids: set[str]
) -> dict[str, int]:
    fresh79 = [item for item in eligible if item["clip_id"] not in gate0c_ids]
    if len(fresh79) != EXPECTED_D405_POOL - GATE0C_CLIPS:
        raise ConfirmationError(f"fresh79 count differs: {len(fresh79)}")
    ordered = sorted(
        fresh79,
        key=lambda item: (float(item["planned_joint_motion_rms"]), item["clip_id"]),
    )
    bins = np.array_split(np.asarray(ordered, dtype=object), MOTION_STRATA)
    return {
        str(item["clip_id"]): stratum
        for stratum, values in enumerate(bins)
        for item in values.tolist()
    }


def _renderer_unopened_census(
    eligible: list[dict[str, Any]],
    gate0c_ids: set[str],
    fresh24_ids: set[str],
) -> list[dict[str, Any]]:
    """Return all 55 renderer-unopened rows with predecessor-frozen strata."""

    eligible_ids = {str(item["clip_id"]) for item in eligible}
    if len(eligible) != EXPECTED_D405_POOL or len(eligible_ids) != EXPECTED_D405_POOL:
        raise ConfirmationError(f"eligible D405 pool differs: {len(eligible)}/{len(eligible_ids)}")
    if len(gate0c_ids) != GATE0C_CLIPS or len(fresh24_ids) != FRESH24_CLIPS:
        raise ConfirmationError("prior selected count differs")
    if gate0c_ids & fresh24_ids:
        raise ConfirmationError("prior selected sets overlap")
    if not (gate0c_ids | fresh24_ids).issubset(eligible_ids):
        raise ConfirmationError("prior selected IDs are outside current D405 pool")
    stratum_by_id = _stratify_fresh79(eligible, gate0c_ids)
    remaining = [
        dict(item)
        for item in eligible
        if item["clip_id"] not in gate0c_ids and item["clip_id"] not in fresh24_ids
    ]
    if len(remaining) != RENDERER_UNOPENED_CLIPS:
        raise ConfirmationError(f"renderer-unopened census differs: {len(remaining)}")
    for item in remaining:
        item["motion_stratum"] = int(stratum_by_id[item["clip_id"]])
    counts = tuple(
        sum(item["motion_stratum"] == index for item in remaining)
        for index in range(MOTION_STRATA)
    )
    if counts != RENDERER_UNOPENED_STRATUM_COUNTS:
        raise ConfirmationError(f"renderer-unopened stratum counts differ: {counts}")
    return remaining


def select_remaining_census(
    eligible: list[dict[str, Any]],
    gate0c_ids: set[str],
    fresh24_ids: set[str],
    freshness_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return every renderer-endpoint-unopened row; disclose unrelated overlap."""

    if freshness_audit.get("kind") != "recurrent_delta_spatial_confirmation_freshness_audit":
        raise ConfirmationError("freshness audit kind differs")
    claim = freshness_audit.get("claim_boundary", {})
    if claim.get("outcome_payloads_inspected") is not False:
        raise ConfirmationError("freshness audit inspected outcome payloads")
    counts_payload = freshness_audit.get("counts", {})
    expected_counts = {
        "d405_eligible": EXPECTED_D405_POOL,
        "gate0c_renderer_opened": GATE0C_CLIPS,
        "fresh24_renderer_opened": FRESH24_CLIPS,
        "gate0c_fresh24_overlap": 0,
        "renderer_outcome_unopened": RENDERER_UNOPENED_CLIPS,
        "later_vpm_dev64_opened_collision": VPM_COLLISION_CLIPS,
    }
    for name, expected in expected_counts.items():
        if counts_payload.get(name) != expected:
            raise ConfirmationError(f"freshness count differs: {name}")
    if tuple(
        int(counts_payload["renderer_outcome_unopened_by_fresh24_stratum"][str(index)])
        for index in range(MOTION_STRATA)
    ) != RENDERER_UNOPENED_STRATUM_COUNTS:
        raise ConfirmationError("freshness renderer-unopened strata differ")
    collision = freshness_audit.get("collision_audit", {}).get(
        "vpm_invertible_multirate", {}
    )
    if (
        collision.get("development_clip_range") != [416, 480]
        or collision.get("post_selection_exploratory_dev_opened") is not True
        or collision.get("matched_renderer_reserve_id_count") != VPM_COLLISION_CLIPS
        or collision.get("classification") != "strict_freshness_collision"
    ):
        raise ConfirmationError("VPM collision binding differs")
    for name in (
        "registration_identity_sha256",
        "endpoint_identity_sha256",
        "audit_identity_sha256",
        "run_complete_identity_sha256",
    ):
        value = collision.get(name, "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ConfirmationError(f"VPM collision identity differs: {name}")

    renderer_unopened = _renderer_unopened_census(eligible, gate0c_ids, fresh24_ids)
    by_id = {str(item["clip_id"]): item for item in renderer_unopened}
    receipt_rows = freshness_audit.get("episodes", [])
    if len(receipt_rows) != RENDERER_UNOPENED_CLIPS:
        raise ConfirmationError("freshness episode count differs")
    receipt_by_id = {str(item["clip_id"]): item for item in receipt_rows}
    if len(receipt_by_id) != RENDERER_UNOPENED_CLIPS or set(receipt_by_id) != set(by_id):
        raise ConfirmationError("freshness episode membership differs")
    bindings = freshness_audit.get("canonical_bindings", {})
    if bindings.get("fresh24_registration_identity_sha256") != PRIOR_REGISTRATION_IDENTITY:
        raise ConfirmationError("freshness prior-registration binding differs")
    if int(sum(bool(item.get("later_vpm_dev64_opened")) for item in receipt_rows)) != VPM_COLLISION_CLIPS:
        raise ConfirmationError("freshness collision flags differ")
    for clip_id, base_item in by_id.items():
        receipt_item = receipt_by_id[clip_id]
        if int(receipt_item.get("manifest_index", -1)) != int(base_item["manifest_index"]):
            raise ConfirmationError("freshness manifest index differs")
        if int(receipt_item.get("fresh24_motion_stratum", -1)) != int(
            base_item["motion_stratum"]
        ):
            raise ConfirmationError("freshness motion stratum differs")
        if not isinstance(receipt_item.get("later_vpm_dev64_opened"), bool):
            raise ConfirmationError("freshness unrelated-endpoint disclosure flag differs")
    selected_pool = [dict(item) for item in renderer_unopened]
    by_stratum = {
        stratum: sorted(
            [item for item in selected_pool if item["motion_stratum"] == stratum],
            key=lambda item: item["clip_id"],
        )
        for stratum in range(MOTION_STRATA)
    }
    counts = tuple(len(by_stratum[index]) for index in range(MOTION_STRATA))
    if counts != EXPECTED_STRATUM_COUNTS:
        raise ConfirmationError(f"remaining stratum counts differ: {counts}")
    donor_by_id: dict[str, dict[str, Any]] = {}
    for values in by_stratum.values():
        for index, item in enumerate(values):
            donor_by_id[item["clip_id"]] = values[(index + 1) % len(values)]
    selected = sorted(selected_pool, key=lambda item: int(item["manifest_index"]))
    for order, item in enumerate(selected):
        donor = donor_by_id[item["clip_id"]]
        if donor["clip_id"] == item["clip_id"] or donor["episode_dir"] == item["episode_dir"]:
            raise ConfirmationError("donor equals recipient")
        item["score_order"] = order
        item["donor_manifest_index"] = int(donor["manifest_index"])
        item["donor_clip_id"] = str(donor["clip_id"])
        item["donor_episode_dir"] = str(donor["episode_dir"])
        item["protected_test_accessed"] = False
    return selected


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_freshness_documents(
    audit_v2: dict[str, Any], inventory_v1: dict[str, Any], inventory_file_sha256: str
) -> dict[str, Any]:
    if (
        audit_v2.get("schema_version") != 2
        or audit_v2.get("kind")
        != "recurrent_delta_spatial_confirmation_freshness_audit"
    ):
        raise ConfirmationError("canonical freshness audit v2 schema/kind differs")
    summary = audit_v2.get("summary", {})
    if (
        summary.get("renderer_endpoint_unopened") != SCORE_CLIPS
        or tuple(
            int(summary["renderer_endpoint_unopened_by_motion_stratum"][str(index)])
            for index in range(MOTION_STRATA)
        )
        != EXPECTED_STRATUM_COUNTS
        or summary.get("strict_cross_experiment_unopened") != 0
        or summary.get("outcome_values_read") is not False
    ):
        raise ConfirmationError("canonical freshness summary differs")
    if audit_v2.get("claim_boundary", {}).get("outcome_payloads_inspected") is not False:
        raise ConfirmationError("canonical freshness audit inspected outcome payloads")
    renderer = audit_v2.get("renderer_specific_remainder", {})
    if (
        renderer.get("d405_eligible") != EXPECTED_D405_POOL
        or renderer.get("gate0c_opened") != GATE0C_CLIPS
        or renderer.get("fresh24_opened") != FRESH24_CLIPS
        or renderer.get("gate0c_fresh24_overlap") != 0
        or renderer.get("unopened_count") != SCORE_CLIPS
        or tuple(
            int(renderer["motion_strata_counts"][str(index)])
            for index in range(MOTION_STRATA)
        )
        != EXPECTED_STRATUM_COUNTS
        or renderer.get("inventory_container_file_sha256") != inventory_file_sha256
    ):
        raise ConfirmationError("canonical renderer-remainder binding differs")
    if audit_v2.get("supersedes", {}).get("file_sha256") != inventory_file_sha256:
        raise ConfirmationError("canonical v2/v1 provenance binding differs")
    scan = audit_v2.get("canonical_collision_scan", {})
    populations = scan.get("later_completed_populations", [])
    if (
        scan.get("canonical_metadata_files_scanned") != 370
        or scan.get("canonical_files_with_renderer_remainder_id_hits") != 22
        or scan.get("outcome_payloads_inspected") is not False
        or scan.get("union_matched_renderer_remainder_ids") != SCORE_CLIPS
        or scan.get("strict_cross_experiment_endpoint_unopened_count") != 0
        or scan.get("strict_cross_experiment_endpoint_unopened_ids") != []
        or not populations
    ):
        raise ConfirmationError("canonical cross-experiment collision scan differs")
    matrix_sha = str(scan.get("collision_matrix_sha256", ""))
    if len(matrix_sha) != 64 or any(c not in "0123456789abcdef" for c in matrix_sha):
        raise ConfirmationError("canonical collision-matrix digest differs")
    for item in populations:
        matched = item.get("matched_renderer_remainder_ids")
        if not isinstance(matched, int) or matched <= 0:
            raise ConfirmationError("canonical population overlap count differs")
        evidence_paths = [
            value
            for key, value in item.items()
            if (key.endswith("path") or key.endswith("_root")) and isinstance(value, str)
        ]
        if not evidence_paths:
            raise ConfirmationError("canonical population evidence path is absent")
    if audit_v2.get("readiness_verdict", {}).get("endpoint_specific_reuse_study") != (
        "REGISTERABLE_ONLY_WITH_EXPLICIT_REUSE_DISCLOSURE"
    ):
        raise ConfirmationError("canonical reuse-disclosure verdict differs")
    if inventory_v1.get("kind") != "recurrent_delta_spatial_confirmation_freshness_audit":
        raise ConfirmationError("renderer-remainder inventory kind differs")
    episodes = inventory_v1.get("episodes", [])
    if len(episodes) != SCORE_CLIPS or len({item["clip_id"] for item in episodes}) != SCORE_CLIPS:
        raise ConfirmationError("renderer-remainder inventory membership differs")
    return {
        "collision_matrix_sha256": matrix_sha,
        "later_population_inventory_sha256": _canonical_json_sha256(populations),
        "later_completed_population_count": len(populations),
        "cross_experiment_endpoint_overlap_count": SCORE_CLIPS,
        "strict_cross_experiment_unopened_count": 0,
        "renderer_endpoint_unopened_count": SCORE_CLIPS,
        "cross_experiment_strict_freshness_claimed": False,
    }


def _load_freshness_v2(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    audit_v2 = base.read_json(path)
    inventory_record = audit_v2.get("renderer_specific_remainder", {})
    inventory_path = Path(inventory_record.get("exact_episode_inventory_path", ""))
    if not inventory_path.is_file():
        inventory_path = path.parent / inventory_path.name
    if not inventory_path.is_file():
        raise FileNotFoundError("freshness v1 episode inventory is unavailable")
    inventory_sha = base.sha256_file(inventory_path)
    inventory_v1 = base.read_json(inventory_path)
    disclosure = _validate_freshness_documents(audit_v2, inventory_v1, inventory_sha)
    return audit_v2, inventory_path, inventory_v1, disclosure


def _model_tensor_manifest(model_path: Path) -> dict[str, dict[str, Any]]:
    required = (
        "history_feature_mean",
        "history_feature_std",
        "history_feature_active",
        "history_pca_mean",
        "history_pca_components",
        "action_feature_mean",
        "action_feature_std",
        "action_feature_active",
        "action_pca_mean",
        "action_pca_components",
        "absolute_input_mean",
        "absolute_input_std",
        "absolute_target_mean",
        "absolute_target_std",
        "absolute_coef",
        "absolute_intercept",
        "recurrent_input_mean",
        "recurrent_input_std",
        "recurrent_target_mean",
        "recurrent_target_std",
        "recurrent_coef",
        "recurrent_intercept",
    )
    with np.load(model_path, allow_pickle=False) as payload:
        missing = set(required) - set(payload.files)
        if missing:
            raise ConfirmationError(f"prior model tensors missing: {sorted(missing)}")
        if float(payload["recurrent_alpha"]) != PRIOR_RECURRENT_ALPHA:
            raise ConfirmationError("prior recurrent alpha differs")
        if float(payload["recurrent_gain"]) != PRIOR_RECURRENT_GAIN:
            raise ConfirmationError("prior recurrent gain differs")
        result = {}
        for name in required:
            values = payload[name]
            if values.dtype != np.bool_ and not np.isfinite(values).all():
                raise ConfirmationError(f"prior tensor is non-finite: {name}")
            result[name] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sha256": _file_bytes_sha256(values),
            }
    return result


def _source_records() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "workflow": Path(__file__).resolve(),
        "predecessor": root / "tools" / "trajectory_consistent_renderer_gate.py",
        "renderer_base": root / "tools" / "corrected_renderer_attribution.py",
        "geometry": root / "tools" / "abc_d405_nominal_geometry_probe.py",
        "protocol": root
        / "docs"
        / "experiments"
        / "RECURRENT_DELTA_SPATIAL_CONFIRMATION_PROTOCOL.md",
    }
    return {name: base.file_record(path) for name, path in paths.items()}


def _verify_source_contract(registration: dict[str, Any]) -> None:
    current = _source_records()
    for name, frozen in registration["source"]["files"].items():
        if name not in current or current[name]["sha256"] != frozen["sha256"]:
            raise ConfirmationError(f"registered source differs: {name}")


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ConfirmationError("--source-commit must be lowercase 40-hex")
    output.mkdir(parents=True)
    prior_output = args.prior_output.resolve()
    prior = _validate_prior(prior_output)
    freshness_audit_path = args.freshness_audit.resolve()
    (
        freshness_audit,
        freshness_inventory_path,
        freshness_inventory,
        freshness_disclosure,
    ) = _load_freshness_v2(freshness_audit_path)

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

    gate0c_ids = {
        str(item["clip_id"]) for item in prior["gate0c"]["selection"]["selected"]
    }
    fresh24_ids = {
        str(item["clip_id"])
        for item in prior["registration"]["selection"]["selected"]
    }
    bindings = freshness_audit.get("canonical_renderer_bindings", {})
    if bindings.get("train_manifest_sha256") != base.sha256_file(args.train_manifest):
        raise ConfirmationError("freshness/train-manifest hash binding differs")
    if bindings.get("train_actions_sha256") != base.sha256_file(actions_path):
        raise ConfirmationError("freshness/train-actions hash binding differs")
    selected = select_remaining_census(
        eligible, gate0c_ids, fresh24_ids, freshness_inventory
    )
    prior_dir = output / "prior"
    copied = {
        "trajectory_registration": _copy_exact(
            prior_output / "registration.json", prior_dir / "trajectory_registration.json"
        ),
        "trajectory_preparation": _copy_exact(
            prior_output / "preparation.json", prior_dir / "trajectory_preparation.json"
        ),
        "trajectory_completion": _copy_exact(
            prior_output / "run_complete.json", prior_dir / "trajectory_run_complete.json"
        ),
        "gate0c_registration": _copy_exact(
            prior["gate0c_path"], prior_dir / "gate0c_registration.json"
        ),
        "frozen_model": _copy_exact(
            prior["model_path"], prior_dir / "model_state_and_predictions.npz"
        ),
        "freshness_audit_v2": _copy_exact(
            freshness_audit_path, prior_dir / "freshness_audit_v2.json"
        ),
        "freshness_inventory_v1": _copy_exact(
            freshness_inventory_path, prior_dir / "freshness_audit_v1.json"
        ),
    }
    if copied["frozen_model"]["sha256"] != PRIOR_MODEL_SHA256:
        raise ConfirmationError("copied frozen model hash differs")
    tensor_manifest = _model_tensor_manifest(prior_dir / "model_state_and_predictions.npz")

    source_files = _source_records()
    root = Path(__file__).resolve().parents[1]
    runtime_commit = base.git_commit(root)
    if runtime_commit is not None and runtime_commit != args.source_commit:
        raise ConfirmationError(
            f"runtime/source commit differs: {runtime_commit} != {args.source_commit}"
        )
    registration = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_registration",
            "status": "registered_before_remaining55_future_state_or_rgb_access",
            "created_at_utc": base.now(),
            "source": {
                "registered_git_commit": args.source_commit,
                "runtime_worktree_git_commit": runtime_commit,
                "files": source_files,
            },
            "prior": {
                "registration_identity_sha256": prior["registration"]["identity_sha256"],
                "preparation_identity_sha256": prior["preparation"]["identity_sha256"],
                "completion_identity_sha256": prior["complete"]["identity_sha256"],
                "gate0c_registration_identity_sha256": prior["gate0c"]["identity_sha256"],
                "artifacts": copied,
                "freshness_audit_kind": freshness_audit["kind"],
                "freshness_audit_schema_version": freshness_audit["schema_version"],
                "freshness_audit_sha256": copied["freshness_audit_v2"]["sha256"],
                "freshness_inventory_sha256": copied["freshness_inventory_v1"][
                    "sha256"
                ],
                "freshness_disclosure": freshness_disclosure,
                "frozen_model_tensor_manifest": tensor_manifest,
                "selected_alpha": PRIOR_RECURRENT_ALPHA,
                "selected_gain": PRIOR_RECURRENT_GAIN,
                "model_refit_or_tuning": False,
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
                "future_measured_state_opened": False,
                "future_rgb_opened": False,
                "validation_accessed": False,
                "protected_test_accessed": False,
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
            "population": {
                "rule": (
                    "all renderer-endpoint-unopened episodes after Gate-0c24 and "
                    "trajectory fresh24; unrelated endpoint overlap is disclosed, not excluded"
                ),
                "eligible_d405": eligible,
                "gate0c_excluded_clip_ids": sorted(gate0c_ids),
                "fresh24_excluded_clip_ids": sorted(fresh24_ids),
                "unrelated_vpm_dev64_overlap_clip_ids": sorted(
                    item["clip_id"]
                    for item in freshness_inventory["episodes"]
                    if item["later_vpm_dev64_opened"]
                ),
                "cross_experiment_endpoint_overlap_count": SCORE_CLIPS,
                "cross_experiment_strict_freshness_claimed": False,
                "remaining_count": len(selected),
                "expected_stratum_counts": list(EXPECTED_STRATUM_COUNTS),
                "donor_rule": "next clip_id within same frozen fresh79 motion stratum, wraparound",
                "selected": selected,
                "uses_future_measured_state": False,
                "uses_rgb": False,
                "uses_renderer_outcome": False,
            },
            "protocol": {
                "document": "docs/experiments/RECURRENT_DELTA_SPATIAL_CONFIRMATION_PROTOCOL.md",
                "arms": list(ARMS),
                "causal_arms_closed_before_target_access": list(CAUSAL_ARMS),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_type": "common motion-stratified paired episode bootstrap",
                "holm_alpha_each_family": HOLM_ALPHA,
                "flow_family": list(FLOW_LABELS),
                "spatial_ni_family": list(SPATIAL_LABELS),
                "spatial_ni_margins": SPATIAL_NI_MARGINS,
                "minimum_favorable_clip_fraction": MIN_FAVORABLE_CLIP_FRACTION,
                "flow_minimum_relative_improvement_percent": (
                    FLOW_MIN_RELATIVE_IMPROVEMENT_PERCENT
                ),
                "all_members_of_both_families_required": True,
                "positive_decision": "GO_RECURRENT_DELTA_WAN_SCREEN",
                "negative_decision": "STOP_RECURRENT_DELTA_GEOMETRY",
            },
            "freshness_audit": {
                **copied["freshness_audit_v2"],
                "kind": freshness_audit["kind"],
                "schema_version": freshness_audit["schema_version"],
                "outcome_payloads_inspected": freshness_audit["claim_boundary"][
                    "outcome_payloads_inspected"
                ],
                "inventory": copied["freshness_inventory_v1"],
                **freshness_disclosure,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "registration.json", registration)
    return registration


def _load_projection(payload: Any, prefix: str) -> predecessor.PCAProjection:
    return predecessor.PCAProjection(
        feature_mean=payload[f"{prefix}_feature_mean"],
        feature_std=payload[f"{prefix}_feature_std"],
        feature_active=payload[f"{prefix}_feature_active"],
        pca_mean=payload[f"{prefix}_pca_mean"],
        pca_components=payload[f"{prefix}_pca_components"],
        explained_variance_ratio=payload[f"{prefix}_explained_variance_ratio"],
    )


def _load_ridge(payload: Any, prefix: str) -> predecessor.RidgeState:
    return predecessor.RidgeState(
        input_mean=payload[f"{prefix}_input_mean"],
        input_std=payload[f"{prefix}_input_std"],
        target_mean=payload[f"{prefix}_target_mean"],
        target_std=payload[f"{prefix}_target_std"],
        coef=payload[f"{prefix}_coef"],
        intercept=payload[f"{prefix}_intercept"],
    )


def _causal_clip_input(
    row: dict[str, Any], cached_actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read only measured state through boundary four plus proposed commands."""

    state_path = Path(row["episode_dir"]) / "states.npz"
    frame_indices = np.asarray(row["frame_indices"][:HISTORY_FRAME_COUNT], dtype=np.int64)
    with np.load(state_path, allow_pickle=False) as state:
        required = {
            "joint_states",
            "joint_actions",
            "gripper_states",
            "gripper_actions",
            "frame_ts",
        }
        if not required.issubset(state.files):
            raise ConfirmationError(f"missing causal state arrays: {state_path}")
        measured_history = np.concatenate(
            (
                state["joint_states"][frame_indices],
                state["gripper_states"][frame_indices],
            ),
            axis=1,
        ).astype(np.float32)
        start = int(row["start"])
        stop = start + SAMPLE_SIZE * CHUNK_SIZE
        raw_actions = np.concatenate(
            (
                state["joint_actions"][start:stop],
                state["gripper_actions"][start:stop],
            ),
            axis=1,
        ).astype(np.float32).reshape(SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM)
        causal_timestamps = np.asarray(state["frame_ts"][frame_indices], dtype=np.int64)
    cached = np.asarray(cached_actions[..., :ACTION_DIM], dtype=np.float32)
    if not np.array_equal(raw_actions, cached):
        raise ConfirmationError(f"cache/raw command mismatch: {row['clip_id']}")
    observed_residual = (
        measured_history[1:HISTORY_FRAME_COUNT]
        - raw_actions[list(HISTORY_CHUNKS), -1]
    )
    history = np.concatenate(
        (
            measured_history.reshape(-1),
            raw_actions[list(HISTORY_CHUNKS)].reshape(-1),
            observed_residual.reshape(-1),
        )
    ).astype(np.float32)
    future_actions = raw_actions[list(FUTURE_CHUNKS)].astype(np.float32)
    q4 = measured_history[4].astype(np.float32)
    stat = state_path.stat()
    provenance = {
        "clip_id": row["clip_id"],
        "states_npz_path": str(state_path.resolve()),
        "states_npz_bytes": stat.st_size,
        "states_npz_mtime_ns": stat.st_mtime_ns,
        "causal_frame_indices": frame_indices.tolist(),
        "latest_measured_state_boundary": 4,
        "causal_timestamps_sha256": _file_bytes_sha256(causal_timestamps),
        "measured_history_sha256": _file_bytes_sha256(measured_history),
        "score_state_npz_container_opened_for_causal_prefix": True,
        "future_measured_state_indices_extracted": False,
        "future_measured_state_used": False,
        "future_rgb_opened": False,
        "validation_accessed": False,
        "protected_test_accessed": False,
    }
    return history, future_actions, q4, provenance


def build_closed_trajectories(
    history: np.ndarray,
    future_actions: np.ndarray,
    q4: np.ndarray,
    donor_positions: np.ndarray,
    model_path: Path,
) -> dict[str, np.ndarray]:
    """Construct all causal arms; no target state or RGB argument exists."""

    with np.load(model_path, allow_pickle=False) as payload:
        history_projection = _load_projection(payload, "history")
        action_projection = _load_projection(payload, "action")
        absolute_model = _load_ridge(payload, "absolute")
        recurrent_model = _load_ridge(payload, "recurrent")
        alpha = float(payload["recurrent_alpha"])
        gain = float(payload["recurrent_gain"])
    if alpha != PRIOR_RECURRENT_ALPHA or gain != PRIOR_RECURRENT_GAIN:
        raise ConfirmationError("frozen recurrent hyperparameters differ")
    history_context = history_projection.transform(history)
    action_context = action_projection.transform(future_actions)
    native_raw = predecessor.raw_trajectory(q4, future_actions)
    absolute_residual = absolute_model.predict(
        base.combine_native_history_with_action(history_context, action_context)
    ).reshape(SCORE_CLIPS, 8, ACTION_DIM)
    absolute = np.concatenate(
        (q4[:, None], future_actions[:, :, -1] + absolute_residual), axis=1
    ).astype(np.float32)
    recurrent = predecessor.rollout_recurrent(
        recurrent_model, history_context, action_context, future_actions, q4, gain
    )
    donor_actions = future_actions[donor_positions]
    recurrent_shuffled = predecessor.rollout_recurrent(
        recurrent_model,
        history_context,
        action_context[donor_positions],
        donor_actions,
        q4,
        gain,
    )
    values: dict[str, np.ndarray] = {
        "history": history,
        "future_actions": future_actions,
        "measured_q4": q4,
        "donor_positions": donor_positions.astype(np.int64),
        "history_context": history_context,
        "action_context": action_context,
        "donor_future_actions": donor_actions,
    }
    for arm, trajectory in (
        ("raw_command", native_raw),
        ("absolute_ridge", absolute),
        ("recurrent_delta", recurrent),
        ("recurrent_delta_shuffled", recurrent_shuffled),
    ):
        source, target = predecessor._trajectory_pairs(trajectory)
        values[f"pose_unclipped_source_{arm}"] = source
        values[f"pose_unclipped_target_{arm}"] = target
        values[f"pose_rendered_source_{arm}"] = predecessor._render_pose(source)
        values[f"pose_rendered_target_{arm}"] = predecessor._render_pose(target)
    return values


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    registration = base.read_json(output / "registration.json")
    base.validate_identity(registration, "registration")
    if registration.get("kind") != "recurrent_delta_spatial_confirmation_registration":
        raise ConfirmationError("registration kind differs")
    _verify_source_contract(registration)
    if any(
        (output / name).exists()
        for name in (
            "preparation.json",
            "preparation_complete.json",
            "causal_trajectory_closure.json",
            "evaluation.json",
        )
    ):
        raise FileExistsError("preparation or downstream artifact already exists")
    rows = base.read_jsonl(Path(registration["inputs"]["train_manifest"]["path"]))
    base.validate_train_manifest(rows)
    cache_dir = Path(registration["inputs"]["train_cache_metadata"]["path"]).parent
    actions, _, _ = base.cache_actions(cache_dir)
    selected = registration["population"]["selected"]
    score_indices = np.asarray([int(item["manifest_index"]) for item in selected], dtype=np.int64)
    position_by_index = {int(value): position for position, value in enumerate(score_indices)}
    donor_positions = np.asarray(
        [position_by_index[int(item["donor_manifest_index"])] for item in selected],
        dtype=np.int64,
    )
    histories = []
    futures = []
    q4_values = []
    provenance = []
    for item, index in zip(selected, score_indices, strict=True):
        history, future, q4, record = _causal_clip_input(rows[int(index)], actions[int(index)])
        record.update(
            {
                "score_order": int(item["score_order"]),
                "manifest_index": int(index),
                "motion_stratum": int(item["motion_stratum"]),
                "donor_manifest_index": int(item["donor_manifest_index"]),
                "donor_clip_id": item["donor_clip_id"],
            }
        )
        histories.append(history)
        futures.append(future)
        q4_values.append(q4)
        provenance.append(record)
    history_array = np.stack(histories).astype(np.float32)
    future_array = np.stack(futures).astype(np.float32)
    q4_array = np.stack(q4_values).astype(np.float32)
    model_path = _resolve_record(
        output, registration["prior"]["artifacts"]["frozen_model"], "prior"
    )
    if base.sha256_file(model_path) != PRIOR_MODEL_SHA256:
        raise ConfirmationError("registered frozen model changed")
    trajectories = build_closed_trajectories(
        history_array, future_array, q4_array, donor_positions, model_path
    )
    trajectory_path = output / "causal_trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        score_indices=score_indices,
        motion_strata=np.asarray(
            [int(item["motion_stratum"]) for item in selected], dtype=np.int8
        ),
        **trajectories,
    )
    provenance_path = output / "causal_input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for record in provenance:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    preparation = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_preparation",
            "created_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "status": "all_causal_trajectories_materialized_before_target_access",
            "score_clips": SCORE_CLIPS,
            "causal_arms": list(CAUSAL_ARMS),
            "frozen_model_sha256": PRIOR_MODEL_SHA256,
            "model_refit_or_tuning": False,
            "latest_measured_predictor_boundary": 4,
            "score_state_npz_containers_opened_for_causal_prefix": True,
            "future_measured_state_indices_extracted": False,
            "future_measured_state_used": False,
            "future_rgb_opened": False,
            "artifacts": {
                "registration": base.file_record(output / "registration.json"),
                "causal_trajectories": base.file_record(trajectory_path),
                "causal_input_provenance": base.file_record(provenance_path),
                "frozen_model": base.file_record(model_path),
                "source": base.file_record(Path(__file__).resolve()),
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "preparation.json", preparation)
    closure = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_causal_trajectory_closure",
            "closed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "causal_trajectories": base.file_record(trajectory_path),
            "score_clips": SCORE_CLIPS,
            "trajectory_arrays": sorted(trajectories),
            "score_state_npz_containers_opened_for_causal_prefix": True,
            "future_measured_state_indices_extracted": False,
            "future_measured_state_used": False,
            "future_rgb_opened": False,
            "immutable_before_evaluation": True,
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "causal_trajectory_closure.json", closure)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_preparation_complete",
            "completed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "closure_identity_sha256": closure["identity_sha256"],
            "status": "completed_before_target_access",
            "artifacts": {
                "preparation": base.file_record(output / "preparation.json"),
                "closure": base.file_record(output / "causal_trajectory_closure.json"),
                **preparation["artifacts"],
            },
            "score_state_npz_containers_opened_for_causal_prefix": True,
            "future_measured_state_indices_extracted": False,
            "future_measured_state_used": False,
            "future_rgb_opened": False,
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "preparation_complete.json", complete)
    return preparation


def _load_closed_arrays(output: Path, closure: dict[str, Any]) -> dict[str, np.ndarray]:
    path = _resolve_record(output, closure["causal_trajectories"])
    if base.sha256_file(path) != closure["causal_trajectories"]["sha256"]:
        raise ConfirmationError("causal trajectory artifact changed after closure")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if arrays["score_indices"].shape != (SCORE_CLIPS,):
        raise ConfirmationError("closed score-index geometry differs")
    for arm in CAUSAL_ARMS:
        for kind in ("source", "target"):
            name = f"pose_rendered_{kind}_{arm}"
            if arrays.get(name, np.empty(0)).shape != (SCORE_CLIPS, 8, ACTION_DIM):
                raise ConfirmationError(f"closed trajectory geometry differs: {name}")
    return arrays


def _aggregate_clip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    clip_ids = sorted({str(row["clip_id"]) for row in rows})
    if len(clip_ids) != SCORE_CLIPS:
        raise ConfirmationError(f"metric clip count differs: {len(clip_ids)}")
    for clip_id in clip_ids:
        clip_rows = [row for row in rows if row["clip_id"] == clip_id]
        for arm in ARMS:
            arm_rows = [row for row in clip_rows if row["arm"] == arm]
            if len(arm_rows) != 8:
                raise ConfirmationError(f"{clip_id}/{arm} row count differs: {len(arm_rows)}")
            flow_count = float(
                sum(row["metrics"]["oracle_moving_flow_sample_count"] for row in arm_rows)
            )
            if flow_count < 10:
                raise ConfirmationError(f"{clip_id}/{arm} has insufficient flow support")
            metrics: dict[str, float] = {}
            for name in sorted(arm_rows[0]["metrics"]):
                values = [row["metrics"][name] for row in arm_rows]
                numeric = [float(value) for value in values if value is not None]
                if not numeric:
                    continue
                if name == "robot_flow_epe_px":
                    metrics[name] = float(
                        sum(row["metrics"]["robot_flow_epe_sum_px"] for row in arm_rows)
                        / flow_count
                    )
                elif name in {
                    "robot_flow_epe_sum_px",
                    "oracle_moving_flow_sample_count",
                    "flow_transition_scored",
                }:
                    metrics[name] = float(sum(numeric))
                else:
                    metrics[name] = float(np.mean(numeric))
            aggregates.append(
                {
                    "clip_id": clip_id,
                    "motion_stratum": int(arm_rows[0]["motion_stratum"]),
                    "arm": arm,
                    "metrics": metrics,
                    "split": "train",
                    "validation_accessed": False,
                    "protected_test_accessed": False,
                }
            )
    return aggregates


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import cv2  # noqa: F401
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("evaluate requires mujoco and opencv-python-headless") from exc

    output = args.output.resolve()
    if any((output / name).exists() for name in ("evaluation.json", "analysis.json")):
        raise FileExistsError("evaluation or downstream analysis already exists")
    registration = base.read_json(output / "registration.json")
    preparation = base.read_json(output / "preparation.json")
    closure = base.read_json(output / "causal_trajectory_closure.json")
    for name, document in (
        ("registration", registration),
        ("preparation", preparation),
        ("closure", closure),
    ):
        base.validate_identity(document, name)
    if preparation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise ConfirmationError("preparation-registration binding differs")
    if closure["preparation_identity_sha256"] != preparation["identity_sha256"]:
        raise ConfirmationError("closure-preparation binding differs")
    if (
        closure.get("future_measured_state_indices_extracted") is not False
        or closure.get("future_measured_state_used") is not False
        or closure.get("future_rgb_opened") is not False
    ):
        raise ConfirmationError("closure target-blind flags differ")
    _verify_source_contract(registration)
    closed = _load_closed_arrays(output, closure)

    official_root = args.official_abc_root.resolve()
    official_commit = base.git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise ConfirmationError(
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
        raise ConfirmationError("official model lacks top camera")
    moving_geoms = base.moving_geom_ids(model, mujoco)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    far_depth = float(model.stat.extent * model.vis.map.zfar)

    rows = base.read_jsonl(Path(registration["inputs"]["train_manifest"]["path"]))
    selected = registration["population"]["selected"]
    frame_rows: list[dict[str, Any]] = []
    bundle_records: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    steady_latencies: list[float] = []
    bundle_dir = output / "score_bundles"
    bundle_dir.mkdir()
    overlay_dir = output / "overlays"
    overlay_dir.mkdir()
    renderer = None
    target_opened_at = base.now()
    try:
        for position, item in enumerate(selected):
            row = rows[int(item["manifest_index"])]
            episode = Path(row["episode_dir"])
            state_path = episode / "states.npz"
            video_path = episode / "top.mp4"
            provenance = base.read_jsonl(output / "causal_input_provenance.jsonl")[position]
            state_stat = state_path.stat()
            if (
                state_stat.st_size != provenance["states_npz_bytes"]
                or state_stat.st_mtime_ns != provenance["states_npz_mtime_ns"]
            ):
                raise ConfirmationError("score state file changed after causal closure")
            frame_indices = np.asarray(row["frame_indices"][4:13], dtype=np.int64)
            with np.load(state_path, allow_pickle=False) as state:
                measured = np.concatenate(
                    (
                        state["joint_states"][frame_indices],
                        state["gripper_states"][frame_indices],
                    ),
                    axis=1,
                ).astype(np.float32)
                frame_ts = np.asarray(state["frame_ts"][frame_indices], dtype=np.int64)
            if not np.array_equal(measured[0], closed["measured_q4"][position]):
                raise ConfirmationError("closed q4 differs from evaluation source")
            rgb = _read_video_frames(video_path, frame_indices.tolist())
            mcap_path = base.raw_mcap_path(
                row,
                Path(registration["inputs"]["preprocessed_root"]),
                Path(registration["inputs"]["raw_root"]),
            )
            calibration = _decode_top_calibration(mcap_path)
            K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
            D = np.asarray(calibration["D"], dtype=np.float64)
            arrays: dict[str, np.ndarray] = {
                "rgb": rgb,
                "frame_indices": frame_indices,
                "frame_ts": frame_ts,
                "K": K,
                "D": D,
                "measured_trajectory": measured,
            }
            for arm in CAUSAL_ARMS:
                for kind in ("source", "target"):
                    arrays[f"pose_rendered_{kind}_{arm}"] = closed[
                        f"pose_rendered_{kind}_{arm}"
                    ][position]
                    arrays[f"pose_unclipped_{kind}_{arm}"] = closed[
                        f"pose_unclipped_{kind}_{arm}"
                    ][position]
            oracle_source, oracle_target = predecessor._trajectory_pairs(measured[None])
            arrays["pose_unclipped_source_measured_oracle"] = oracle_source[0]
            arrays["pose_unclipped_target_measured_oracle"] = oracle_target[0]
            arrays["pose_rendered_source_measured_oracle"] = predecessor._render_pose(
                oracle_source[0]
            )
            arrays["pose_rendered_target_measured_oracle"] = predecessor._render_pose(
                oracle_target[0]
            )
            bundle_path = bundle_dir / f"{item['clip_id']}.npz"
            np.savez_compressed(bundle_path, **arrays)
            metadata = base.seal(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "recurrent_delta_spatial_confirmation_score_bundle",
                    "registration_identity_sha256": registration["identity_sha256"],
                    "closure_identity_sha256": closure["identity_sha256"],
                    "score_order": position,
                    "manifest_index": int(item["manifest_index"]),
                    "motion_stratum": int(item["motion_stratum"]),
                    "clip_id": item["clip_id"],
                    "split": "train",
                    "target_access_after_closure": True,
                    "bundle": base.file_record(bundle_path),
                    "source_files": {
                        "states_npz": base.file_record(state_path),
                        "top_mp4": base.file_record(video_path),
                        "raw_mcap": base.file_record(mcap_path),
                    },
                    "validation_accessed": False,
                    "protected_test_accessed": False,
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

            _, height, width, channels = rgb.shape
            if channels != 3:
                raise ConfirmationError("RGB channel geometry differs")
            if renderer is not None:
                renderer.close()
            renderer = mujoco.Renderer(model, height=height, width=width)
            fy = float(K[1, 1])
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
            latency_start = len(render_latencies)
            for arm in ARMS:
                per_arm[arm] = (
                    [rendered(pose) for pose in arrays[f"pose_rendered_source_{arm}"]],
                    [rendered(pose) for pose in arrays[f"pose_rendered_target_{arm}"]],
                )
            clip_latencies = render_latencies[latency_start:]
            steady_latencies.extend(clip_latencies[2:] if len(clip_latencies) > 2 else clip_latencies)

            clip_rows: list[dict[str, Any]] = []
            for step in range(8):
                oracle_source_render = per_arm["measured_oracle"][0][step]
                oracle_target_render = per_arm["measured_oracle"][1][step]
                edges = observed_edges(rgb[step + 1])
                for arm in ARMS:
                    candidate_source = per_arm[arm][0][step]
                    candidate_target = per_arm[arm][1][step]
                    metrics = {
                        "silhouette_boundary_chamfer_px": base.symmetric_boundary_chamfer(
                            candidate_target.mask, oracle_target_render.mask
                        ),
                        "silhouette_iou": base.mask_iou(
                            candidate_target.mask, oracle_target_render.mask
                        ),
                        **base.rgb_robot_band_metrics(
                            candidate_target.mask, oracle_target_render.mask, edges
                        ),
                        **base.robot_flow_error(
                            candidate_source,
                            candidate_target,
                            oracle_source_render,
                            oracle_target_render,
                            far_depth=far_depth,
                        ),
                    }
                    metric_row = {
                        "schema_version": SCHEMA_VERSION,
                        "clip_id": item["clip_id"],
                        "score_order": position,
                        "motion_stratum": int(item["motion_stratum"]),
                        "transition_ordinal": step + 1,
                        "source_frame_index": int(frame_indices[step]),
                        "target_frame_index": int(frame_indices[step + 1]),
                        "arm": arm,
                        "metrics": metrics,
                        "split": "train",
                        "target_access_after_closure": True,
                        "validation_accessed": False,
                        "protected_test_accessed": False,
                    }
                    frame_rows.append(metric_row)
                    clip_rows.append(metric_row)
            recurrent_rows = [row_ for row_ in clip_rows if row_["arm"] == "recurrent_delta"]
            worst = max(
                recurrent_rows,
                key=lambda row_: row_["metrics"]["rgb_robot_band_chamfer_px"],
            )
            step = int(worst["transition_ordinal"] - 1)
            overlay_path = overlay_dir / f"worst_recurrent_{item['clip_id'][:12]}.png"
            base._overlay_boundaries(
                overlay_path,
                rgb[step + 1],
                per_arm["raw_command"][1][step].mask,
                per_arm["recurrent_delta"][1][step].mask,
                per_arm["measured_oracle"][1][step].mask,
                f"raw=red recurrent=green oracle=blue transition={step + 1}",
            )
    finally:
        if renderer is not None:
            renderer.close()

    frame_path = output / "frame_transition_metrics.jsonl"
    with frame_path.open("w") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    latency = np.asarray(steady_latencies, dtype=np.float64)
    evaluation = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_evaluation",
            "created_at_utc": base.now(),
            "target_opened_at_utc": target_opened_at,
            "registration_identity_sha256": registration["identity_sha256"],
            "preparation_identity_sha256": preparation["identity_sha256"],
            "closure_identity_sha256": closure["identity_sha256"],
            "causal_trajectory_sha256_verified_before_target_access": closure[
                "causal_trajectories"
            ]["sha256"],
            "score_clips": SCORE_CLIPS,
            "score_transitions": SCORE_CLIPS * 8,
            "arms": list(ARMS),
            "future_measured_state_opened_only_for_scoring": True,
            "future_rgb_opened_only_for_scoring": True,
            "camera": {
                "official_abc_commit": official_commit,
                "official_scene": base.file_record(scene_path),
                "robot_only_xml_sha256": hashlib.sha256(robot_xml.encode()).hexdigest(),
                "nominal_extrinsic": True,
                "recorded_fy_as_fovy": True,
                "principal_point_centered": True,
                "distortion_applied": False,
            },
            "flow_contract": {
                "pixel_stride": FLOW_PIXEL_STRIDE,
                "minimum_oracle_motion_px": FLOW_MIN_ORACLE_MOTION_PX,
                "minimum_clip_pixels": 10,
                "far_depth": far_depth,
                "moving_geom_ids": sorted(moving_geoms),
            },
            "latency": {
                "deduplicated_render_pose_mean_ms": float(latency.mean()),
                "deduplicated_render_pose_p50_ms": float(np.percentile(latency, 50)),
                "deduplicated_render_pose_p95_ms": float(np.percentile(latency, 95)),
                "pose_count_after_two_warmups_per_clip": int(len(latency)),
            },
            "artifacts": {
                "registration": base.file_record(output / "registration.json"),
                "preparation": base.file_record(output / "preparation.json"),
                "closure": base.file_record(output / "causal_trajectory_closure.json"),
                "frame_transition_metrics": base.file_record(frame_path),
                "bundles": bundle_records,
                "overlays": [base.file_record(path) for path in sorted(overlay_dir.glob("*.png"))],
                "source": base.file_record(Path(__file__).resolve()),
            },
            "split": "train",
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "evaluation.json", evaluation)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_evaluation_complete",
            "completed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "evaluation_identity_sha256": evaluation["identity_sha256"],
            "status": "completed",
            "artifacts": {
                "evaluation": base.file_record(output / "evaluation.json"),
                **evaluation["artifacts"],
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "evaluation_complete.json", complete)
    return evaluation


def stratified_bootstrap_indices(
    strata: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    strata = np.asarray(strata, dtype=np.int64)
    if strata.shape != (SCORE_CLIPS,):
        raise ValueError(f"strata geometry differs: {strata.shape}")
    positions = [np.flatnonzero(strata == value) for value in range(MOTION_STRATA)]
    counts = tuple(len(value) for value in positions)
    if counts != EXPECTED_STRATUM_COUNTS:
        raise ConfirmationError(f"bootstrap stratum counts differ: {counts}")
    rng = np.random.default_rng(seed)
    blocks = [
        rng.choice(value, size=(samples, len(value)), replace=True) for value in positions
    ]
    return np.concatenate(blocks, axis=1).astype(np.int16)


def paired_effect(
    reference: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    *,
    higher_is_better: bool,
    null_boundary: float,
) -> tuple[dict[str, Any], np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != (SCORE_CLIPS,) or candidate.shape != (SCORE_CLIPS,):
        raise ValueError(f"paired metric geometry differs: {reference.shape}/{candidate.shape}")
    difference = candidate - reference if higher_is_better else reference - candidate
    samples = difference[np.asarray(indices, dtype=np.int64)].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    denominator = float(reference.mean())
    relative = None
    if abs(denominator) > 1e-12:
        relative = 100.0 * float(difference.mean()) / abs(denominator)
    return (
        {
            "reference_mean": denominator,
            "candidate_mean": float(candidate.mean()),
            "paired_favorable_difference_mean": float(difference.mean()),
            "paired_bootstrap_95_ci": [float(low), float(high)],
            "null_boundary": float(null_boundary),
            "one_sided_bootstrap_p": float(
                (1 + np.sum(samples <= null_boundary)) / (len(samples) + 1)
            ),
            "relative_improvement_percent": relative,
            "favorable_clip_fraction": float(np.mean(difference > 0.0)),
            "within_null_margin_clip_fraction": float(np.mean(difference > null_boundary)),
            "favorable_clips": int(np.sum(difference > 0.0)),
            "within_null_margin_clips": int(np.sum(difference > null_boundary)),
            "paired_clips": SCORE_CLIPS,
            "bootstrap_samples": int(len(samples)),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "common_stratified_bootstrap_indices": True,
            "higher_is_better": higher_is_better,
        },
        samples,
    )


def apply_holm_family(
    labels: Sequence[str],
    effects: dict[str, dict[str, Any]],
    samples: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    ordered = sorted(labels, key=lambda label: (effects[label]["one_sided_bootstrap_p"], label))
    active = True
    rows = []
    for rank_zero, label in enumerate(ordered):
        threshold = HOLM_ALPHA / (len(ordered) - rank_zero)
        p_value = float(effects[label]["one_sided_bootstrap_p"])
        rejected = bool(active and p_value <= threshold)
        if not rejected:
            active = False
        lower_bound = float(np.quantile(samples[label], threshold))
        effects[label].update(
            {
                "holm_rank": rank_zero + 1,
                "holm_alpha_threshold": threshold,
                "holm_stepdown_rejected": rejected,
                "holm_one_sided_lower_bound": lower_bound,
            }
        )
        rows.append(
            {
                "rank": rank_zero + 1,
                "label": label,
                "one_sided_bootstrap_p": p_value,
                "alpha_threshold": threshold,
                "stepdown_rejected": rejected,
                "one_sided_lower_bound": lower_bound,
                "null_boundary": effects[label]["null_boundary"],
            }
        )
    return rows


def analyze_aggregates(
    aggregates: list[dict[str, Any]],
    clip_order: list[str],
    indices: np.ndarray,
) -> dict[str, Any]:
    if len(clip_order) != SCORE_CLIPS or len(set(clip_order)) != SCORE_CLIPS:
        raise ConfirmationError("analysis clip order differs")
    by_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        arm_rows = {row["clip_id"]: row for row in aggregates if row["arm"] == arm}
        if set(arm_rows) != set(clip_order):
            raise ConfirmationError(f"aggregate clip membership differs: {arm}")
        metrics = arm_rows[clip_order[0]]["metrics"]
        by_arm[arm] = {
            metric: np.asarray(
                [arm_rows[clip_id]["metrics"][metric] for clip_id in clip_order],
                dtype=np.float64,
            )
            for metric in metrics
        }
    effects: dict[str, dict[str, Any]] = {}
    sample_map: dict[str, np.ndarray] = {}
    for reference in ("raw_command", "absolute_ridge", "recurrent_delta_shuffled"):
        label = f"recurrent_delta_vs_{reference}:robot_flow_epe_px"
        effects[label], sample_map[label] = paired_effect(
            by_arm[reference]["robot_flow_epe_px"],
            by_arm["recurrent_delta"]["robot_flow_epe_px"],
            indices,
            higher_is_better=False,
            null_boundary=0.0,
        )
    for reference in ("raw_command", "absolute_ridge"):
        for metric, margin in SPATIAL_NI_MARGINS.items():
            label = f"recurrent_delta_vs_{reference}:{metric}"
            effects[label], sample_map[label] = paired_effect(
                by_arm[reference][metric],
                by_arm["recurrent_delta"][metric],
                indices,
                higher_is_better=(metric == "silhouette_iou"),
                null_boundary=-float(margin),
            )
    flow_holm = apply_holm_family(FLOW_LABELS, effects, sample_map)
    spatial_holm = apply_holm_family(SPATIAL_LABELS, effects, sample_map)
    flow_gates = {}
    for label in FLOW_LABELS:
        effect = effects[label]
        needs_relative = not label.startswith(
            "recurrent_delta_vs_recurrent_delta_shuffled:"
        )
        flow_gates[label] = bool(
            effect["paired_favorable_difference_mean"] > 0.0
            and effect["holm_stepdown_rejected"]
            and effect["holm_one_sided_lower_bound"] > 0.0
            and effect["favorable_clip_fraction"] >= MIN_FAVORABLE_CLIP_FRACTION
            and (
                not needs_relative
                or effect["relative_improvement_percent"]
                >= FLOW_MIN_RELATIVE_IMPROVEMENT_PERCENT
            )
        )
    spatial_gates = {}
    for label in SPATIAL_LABELS:
        effect = effects[label]
        boundary = effect["null_boundary"]
        spatial_gates[label] = bool(
            effect["paired_favorable_difference_mean"] > boundary
            and effect["holm_stepdown_rejected"]
            and effect["holm_one_sided_lower_bound"] > boundary
        )
    passed = bool(all(flow_gates.values()) and all(spatial_gates.values()))
    return {
        "metrics_by_arm": {
            arm: {metric: float(values.mean()) for metric, values in metrics.items()}
            for arm, metrics in by_arm.items()
        },
        "effects": effects,
        "holm_families": {
            "flow_superiority_3": {
                "family_size": 3,
                "alpha": HOLM_ALPHA,
                "order": flow_holm,
            },
            "spatial_noninferiority_6": {
                "family_size": 6,
                "alpha": HOLM_ALPHA,
                "order": spatial_holm,
            },
        },
        "gates": {
            "flow_superiority": flow_gates,
            "spatial_noninferiority": spatial_gates,
            "all_members_of_both_families_required": True,
            "recurrent_delta_spatial_confirmation_pass": passed,
        },
        "decision": (
            "GO_RECURRENT_DELTA_WAN_SCREEN"
            if passed
            else "STOP_RECURRENT_DELTA_GEOMETRY"
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if any((output / name).exists() for name in ("analysis.json", "run_complete.json")):
        raise FileExistsError("analysis or completion already exists")
    registration = base.read_json(output / "registration.json")
    evaluation = base.read_json(output / "evaluation.json")
    closure = base.read_json(output / "causal_trajectory_closure.json")
    for name, document in (
        ("registration", registration),
        ("evaluation", evaluation),
        ("closure", closure),
    ):
        base.validate_identity(document, name)
    if evaluation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise ConfirmationError("evaluation-registration binding differs")
    if evaluation["closure_identity_sha256"] != closure["identity_sha256"]:
        raise ConfirmationError("evaluation-closure binding differs")
    _verify_source_contract(registration)
    frame_rows = base.read_jsonl(output / "frame_transition_metrics.jsonl")
    aggregates = _aggregate_clip_rows(frame_rows)
    clip_path = output / "clip_metrics.jsonl"
    with clip_path.open("w") as handle:
        for row in aggregates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    selected = registration["population"]["selected"]
    clip_order = [str(item["clip_id"]) for item in selected]
    strata = np.asarray([int(item["motion_stratum"]) for item in selected], dtype=np.int8)
    indices = stratified_bootstrap_indices(strata)
    bootstrap_path = output / "bootstrap_indices.npz"
    np.savez_compressed(
        bootstrap_path,
        indices=indices,
        strata=strata,
        seed=np.asarray(BOOTSTRAP_SEED, dtype=np.int64),
        samples=np.asarray(BOOTSTRAP_SAMPLES, dtype=np.int64),
    )
    result = analyze_aggregates(aggregates, clip_order, indices)
    analysis = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_analysis",
            "created_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "evaluation_identity_sha256": evaluation["identity_sha256"],
            "closure_identity_sha256": closure["identity_sha256"],
            "split": "train",
            "score_clips": SCORE_CLIPS,
            "stratum_counts": list(EXPECTED_STRATUM_COUNTS),
            "bootstrap": {
                "type": "common motion-stratified paired episode bootstrap",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "indices": base.file_record(bootstrap_path),
            },
            **result,
            "artifacts": {
                "registration": base.file_record(output / "registration.json"),
                "closure": base.file_record(output / "causal_trajectory_closure.json"),
                "evaluation": base.file_record(output / "evaluation.json"),
                "frame_transition_metrics": base.file_record(
                    output / "frame_transition_metrics.jsonl"
                ),
                "clip_metrics": base.file_record(clip_path),
                "bootstrap_indices": base.file_record(bootstrap_path),
                "source": base.file_record(Path(__file__).resolve()),
            },
            "claim_boundary": (
                "Train-only remaining-D405 robot-renderer confirmation; no generated video, "
                "validation, protected test, object/contact flow, or policy outcome."
            ),
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "analysis.json", analysis)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_spatial_confirmation_complete",
            "completed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": analysis["decision"],
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


def _verify_record(output: Path, record: dict[str, Any], *parts: str) -> Path:
    path = _resolve_record(output, record, *parts)
    if base.sha256_file(path) != record["sha256"]:
        raise ConfirmationError(f"artifact hash differs: {path}")
    return path


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def audit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    names = (
        "registration.json",
        "preparation.json",
        "preparation_complete.json",
        "causal_trajectory_closure.json",
        "evaluation.json",
        "evaluation_complete.json",
        "analysis.json",
        "run_complete.json",
    )
    documents = {name: base.read_json(output / name) for name in names}
    for name, document in documents.items():
        base.validate_identity(document, name)
    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    closure = documents["causal_trajectory_closure.json"]
    evaluation = documents["evaluation.json"]
    analysis = documents["analysis.json"]
    complete = documents["run_complete.json"]
    if preparation["registration_identity_sha256"] != registration["identity_sha256"]:
        raise ConfirmationError("audit preparation-registration binding differs")
    if closure["preparation_identity_sha256"] != preparation["identity_sha256"]:
        raise ConfirmationError("audit closure-preparation binding differs")
    if evaluation["closure_identity_sha256"] != closure["identity_sha256"]:
        raise ConfirmationError("audit evaluation-closure binding differs")
    if analysis["evaluation_identity_sha256"] != evaluation["identity_sha256"]:
        raise ConfirmationError("audit analysis-evaluation binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise ConfirmationError("audit completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise ConfirmationError("audit completion decision differs")
    if _parse_time(closure["closed_at_utc"]) > _parse_time(evaluation["target_opened_at_utc"]):
        raise ConfirmationError("targets were opened before causal trajectory closure")
    _verify_source_contract(registration)

    prior_records = registration["prior"]["artifacts"]
    prior_paths = {
        name: _verify_record(output, record, "prior")
        for name, record in prior_records.items()
    }
    if base.sha256_file(prior_paths["frozen_model"]) != PRIOR_MODEL_SHA256:
        raise ConfirmationError("audit frozen model hash differs")
    if _model_tensor_manifest(prior_paths["frozen_model"]) != registration["prior"][
        "frozen_model_tensor_manifest"
    ]:
        raise ConfirmationError("audit frozen tensor manifest differs")
    gate0c = base.read_json(prior_paths["gate0c_registration"])
    prior_registration = base.read_json(prior_paths["trajectory_registration"])
    freshness_audit_v2 = base.read_json(prior_paths["freshness_audit_v2"])
    freshness_inventory_v1 = base.read_json(prior_paths["freshness_inventory_v1"])
    disclosure = _validate_freshness_documents(
        freshness_audit_v2,
        freshness_inventory_v1,
        base.sha256_file(prior_paths["freshness_inventory_v1"]),
    )
    if disclosure != registration["prior"]["freshness_disclosure"]:
        raise ConfirmationError("audit canonical freshness disclosure differs")
    base.validate_identity(gate0c, "copied Gate-0c")
    base.validate_identity(prior_registration, "copied trajectory registration")
    selected = select_remaining_census(
        registration["population"]["eligible_d405"],
        {item["clip_id"] for item in gate0c["selection"]["selected"]},
        {item["clip_id"] for item in prior_registration["selection"]["selected"]},
        freshness_inventory_v1,
    )
    if selected != registration["population"]["selected"]:
        raise ConfirmationError("audit renderer-endpoint-unopened55 selection/donor map differs")

    trajectory_path = _verify_record(output, closure["causal_trajectories"])
    closed = _load_closed_arrays(output, closure)
    replay = build_closed_trajectories(
        closed["history"],
        closed["future_actions"],
        closed["measured_q4"],
        closed["donor_positions"].astype(np.int64),
        prior_paths["frozen_model"],
    )
    replay_error = 0.0
    for name, values in replay.items():
        if name not in closed:
            raise ConfirmationError(f"audit closed trajectory lacks {name}")
        if values.dtype == np.bool_:
            equal = np.array_equal(values, closed[name])
            error = 0.0 if equal else float("inf")
        else:
            error = float(np.max(np.abs(values - closed[name])))
        replay_error = max(replay_error, error)
    if replay_error > 5e-6:
        raise ConfirmationError(f"causal trajectory replay differs: {replay_error}")

    frame_path = _verify_record(output, evaluation["artifacts"]["frame_transition_metrics"])
    frame_rows = base.read_jsonl(frame_path)
    if len(frame_rows) != SCORE_CLIPS * 8 * len(ARMS):
        raise ConfirmationError(f"audit frame row count differs: {len(frame_rows)}")
    for row in frame_rows:
        if row["metrics"]["oracle_source_reprojection_rmse_px"] > 1e-5:
            raise ConfirmationError("audit flow source reprojection differs")
    aggregates = _aggregate_clip_rows(frame_rows)
    clip_path = _verify_record(output, analysis["artifacts"]["clip_metrics"])
    stored_aggregates = base.read_jsonl(clip_path)
    if aggregates != stored_aggregates:
        raise ConfirmationError("audit clip aggregation differs")

    bootstrap_path = _verify_record(output, analysis["artifacts"]["bootstrap_indices"])
    with np.load(bootstrap_path, allow_pickle=False) as payload:
        indices = payload["indices"].copy()
        strata = payload["strata"].copy()
        seed = int(payload["seed"])
        samples = int(payload["samples"])
    expected_indices = stratified_bootstrap_indices(strata, samples=samples, seed=seed)
    if not np.array_equal(indices, expected_indices):
        raise ConfirmationError("audit bootstrap index matrix differs")
    clip_order = [item["clip_id"] for item in selected]
    recomputed = analyze_aggregates(aggregates, clip_order, indices)
    for key in ("metrics_by_arm", "effects", "holm_families", "gates", "decision"):
        if recomputed[key] != analysis[key]:
            raise ConfirmationError(f"audit recomputed analysis differs: {key}")

    artifact_hashes = 0
    for record in evaluation["artifacts"]["bundles"]:
        _verify_record(output, record["bundle"], "score_bundles")
        metadata_path = _verify_record(output, record["metadata"], "score_bundles")
        metadata = base.read_json(metadata_path)
        base.validate_identity(metadata, str(metadata_path))
        if metadata["closure_identity_sha256"] != closure["identity_sha256"]:
            raise ConfirmationError("bundle-closure binding differs")
        artifact_hashes += 2
    for record in evaluation["artifacts"]["overlays"]:
        _verify_record(output, record, "overlays")
        artifact_hashes += 1
    artifact_hashes += 3

    false_flags = sum(
        base.require_false_flags(document, name) for name, document in documents.items()
    )
    false_flags += base.require_false_flags(frame_rows, "frame_rows")
    false_flags += base.require_false_flags(stored_aggregates, "clip_rows")
    false_flags += base.require_false_flags(
        base.read_jsonl(output / "causal_input_provenance.jsonl"), "causal_provenance"
    )
    return {
        "status": "audit_passed",
        "decision": analysis["decision"],
        "registration_identity_sha256": registration["identity_sha256"],
        "preparation_identity_sha256": preparation["identity_sha256"],
        "closure_identity_sha256": closure["identity_sha256"],
        "evaluation_identity_sha256": evaluation["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "completion_identity_sha256": complete["identity_sha256"],
        "remaining_clips": SCORE_CLIPS,
        "stratum_counts": list(EXPECTED_STRATUM_COUNTS),
        "frame_rows": len(frame_rows),
        "clip_rows": len(stored_aggregates),
        "bootstrap_rows": len(indices),
        "flow_holm_tests_recomputed": len(FLOW_LABELS),
        "spatial_ni_holm_tests_recomputed": len(SPATIAL_LABELS),
        "causal_replay_max_abs_error": replay_error,
        "artifact_hashes_verified": artifact_hashes,
        "explicit_false_flags": false_flags,
        "causal_trajectory_file": str(trajectory_path),
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
    register_parser.add_argument("--prior-output", type=Path, required=True)
    register_parser.add_argument("--freshness-audit", type=Path, required=True)
    register_parser.add_argument("--source-commit", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--official-abc-root", type=Path, required=True)
    evaluate_parser.add_argument("--mujoco-gl", default="egl")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
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
    elif args.command == "analyze":
        result = analyze(args)
        event = "analyzed"
    elif args.command == "audit":
        result = audit(args.output)
        event = "audited"
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"event": event, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
