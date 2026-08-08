#!/usr/bin/env python3
"""Analyze the preregistered paired Action Cycle Stage-1 validation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import action_cycle_stage1 as protocol  # noqa: E402
from tools import action_cycle_stage1_evaluate as evaluation  # noqa: E402
from tools import analyze_lamo_motion_drift as audit_schema  # noqa: E402


SCHEMA_VERSION = 1
KIND_ANALYSIS = "action_cycle_stage1_validation_analysis_v1"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_811
CONTRAST_COUNT = 9
CONFIDENCE = 1.0 - 0.05 / CONTRAST_COUNT
DECODED = "decoded_mse_unit_range"
TEMPORAL = "decoded_temporal_difference_mse_unit_range"
LATENT = "video_future_nmse"
QUALITY_METRICS = (DECODED, TEMPORAL, LATENT)
ATTRIBUTION_METRICS = (DECODED, TEMPORAL)


class ActionCycleAnalysisError(RuntimeError):
    """Evidence is incomplete, unmatched, or outside the frozen protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def paired_effect(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    label: str,
    simultaneous: bool = True,
) -> dict[str, Any]:
    """Paired relative MSE improvement; positive favors candidate."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if (
        left.shape != (64,)
        or right.shape != (64,)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(right <= 0)
    ):
        raise ActionCycleAnalysisError(f"invalid paired effect values: {label}")
    point = float((right.mean() - left.mean()) / right.mean())
    rng = np.random.default_rng(_derived_seed(label))
    indexes = rng.integers(0, 64, size=(BOOTSTRAP_SAMPLES, 64), endpoint=False)
    left_mean = left[indexes].mean(axis=1)
    right_mean = right[indexes].mean(axis=1)
    draws = (right_mean - left_mean) / right_mean
    confidence = CONFIDENCE if simultaneous else 0.95
    return {
        "candidate_mean": float(left.mean()),
        "reference_mean": float(right.mean()),
        "relative_improvement": point,
        "relative_improvement_percent": 100.0 * point,
        "one_sided_lower_bound": {
            "confidence": confidence,
            "low": float(np.quantile(draws, 1.0 - confidence)),
            "family_contrast_count": CONTRAST_COUNT if simultaneous else None,
        },
        "descriptive_two_sided_95_ci": {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
        },
        "favorable_clip_fraction": float(np.mean(right > left)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_derived_seed": _derived_seed(label),
    }


def difference_in_differences(
    *,
    candidate_aligned: Sequence[float],
    candidate_control: Sequence[float],
    reference_aligned: Sequence[float],
    reference_control: Sequence[float],
    label: str,
    simultaneous: bool,
) -> dict[str, Any]:
    """Extra action-alignment benefit, normalized by AC-OFF aligned MSE."""

    ca, cc, ra, rc = (
        np.asarray(value, dtype=np.float64)
        for value in (
            candidate_aligned,
            candidate_control,
            reference_aligned,
            reference_control,
        )
    )
    if any(value.shape != (64,) for value in (ca, cc, ra, rc)) or any(
        not np.isfinite(value).all() for value in (ca, cc, ra, rc)
    ) or np.any(ra <= 0):
        raise ActionCycleAnalysisError(f"invalid DID values: {label}")
    numerator = (cc - ca) - (rc - ra)
    point = float(numerator.mean() / ra.mean())
    rng = np.random.default_rng(_derived_seed(label))
    indexes = rng.integers(0, 64, size=(BOOTSTRAP_SAMPLES, 64), endpoint=False)
    draws = numerator[indexes].mean(axis=1) / ra[indexes].mean(axis=1)
    confidence = CONFIDENCE if simultaneous else 0.95
    return {
        "definition": "((AC-ON_control-AC-ON_aligned)-(AC-OFF_control-AC-OFF_aligned))/mean(AC-OFF_aligned)",
        "relative_difference_in_differences": point,
        "relative_difference_in_differences_percent": 100.0 * point,
        "one_sided_lower_bound": {
            "confidence": confidence,
            "low": float(np.quantile(draws, 1.0 - confidence)),
            "family_contrast_count": CONTRAST_COUNT if simultaneous else None,
        },
        "descriptive_two_sided_95_ci": {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_derived_seed": _derived_seed(label),
    }


def _load_inventory(
    path: Path, arm: protocol.Arm
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve(strict=True)
    inventory = protocol.read_json(path, "evaluation inventory")
    expected_calls = 64 // 2 * sum(value.nfe for value in protocol.ENDPOINTS)
    if (
        not protocol.identity_valid(inventory)
        or inventory.get("kind") != evaluation.KIND_INVENTORY
        or inventory.get("arm") != {
            "code": arm.code,
            "config_name": arm.config_name,
            "run_name": arm.run_name,
            "loss_weight": arm.loss_weight,
        }
        or inventory.get("evaluation_split") != "validation"
        or inventory.get("validation_clips") != 64
        or inventory.get("endpoints")
        != [
            {
                "code": value.code,
                "nfe": value.nfe,
                "action_source": value.action_source,
                "primary": value.primary,
            }
            for value in protocol.ENDPOINTS
        ]
        or inventory.get("row_count") != 64 * len(protocol.ENDPOINTS)
        or inventory.get("actual_wan_call_count") != expected_calls
        or inventory.get("paired_inputs_and_noise_within_arm") is not True
        or inventory.get("critic_loaded_or_called") is not False
        or inventory.get("online_teacher_call_count") != 0
        or inventory.get("target_cache_array_opened") is not False
        or inventory.get("protected_test_accessed") is not False
    ):
        raise ActionCycleAnalysisError("evaluation inventory differs")
    registration_record = inventory.get("registration", {})
    protocol.revalidate_record(registration_record, "registration")
    registration = protocol.validate_registration(
        Path(registration_record["path"]), rehash_inputs=False
    )
    if inventory.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ActionCycleAnalysisError("inventory registration identity differs")
    artifacts = inventory.get("arm_artifacts", {})
    if artifacts.get("run_identity_sha256") != protocol.arm_run_identity(
        registration, arm
    ) or artifacts.get("deployment_critic_loaded") is not False:
        raise ActionCycleAnalysisError("arm artifact identity differs")
    for key in ("snapshot", "resolved_config", "training_completion"):
        protocol.revalidate_record(artifacts[key], f"arm {key}")
    protocol.revalidate_record(
        artifacts["training_trace"]["trace"], "arm training trace"
    )
    protocol.revalidate_record(
        artifacts["training_trace"]["completion"], "arm training trace completion"
    )
    protocol.revalidate_record(
        artifacts["arm_execution_plan"]["record"], "arm execution plan"
    )
    rows = []
    rank_records = inventory.get("rank_manifests")
    if not isinstance(rank_records, list) or len(rank_records) != 8:
        raise ActionCycleAnalysisError("rank manifest inventory differs")
    for expected_rank, record in enumerate(rank_records):
        protocol.revalidate_record(record, "rank manifest")
        rank = protocol.read_json(record["path"], "rank manifest")
        row_record = rank.get("rows", {})
        row_path = Path(str(row_record.get("path", "")))
        if (
            not protocol.identity_valid(rank)
            or rank.get("kind") != evaluation.KIND_RANK
            or rank.get("rank") != expected_rank
            or rank.get("arm") != inventory["arm"]
            or rank.get("assigned_clip_indexes")
            != evaluation._expected_rank_indexes(expected_rank)
            or rank.get("actual_wan_call_count")
            != len(evaluation._expected_rank_indexes(expected_rank)) // 2
            * sum(value.nfe for value in protocol.ENDPOINTS)
            or row_record.get("sha256") != protocol.file_record(row_path)["sha256"]
            or row_record.get("count")
            != len(evaluation._expected_rank_indexes(expected_rank))
            * len(protocol.ENDPOINTS)
            or rank.get("critic_loaded_or_called") is not False
            or rank.get("protected_test_accessed") is not False
        ):
            raise ActionCycleAnalysisError("rank evidence differs")
        with row_path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    evaluation._validate_rows(rows, arm, registration)
    return inventory, registration, rows


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(row["clip_index"]), str(row["endpoint"]["code"])): row
        for row in rows
    }


def _metric(
    rows: Mapping[tuple[int, str], Mapping[str, Any]], endpoint: str, name: str
) -> list[float]:
    return [float(rows[(index, endpoint)]["metrics"][name]) for index in range(64)]


def _assert_evaluation_pair(
    candidate: Mapping[tuple[int, str], Mapping[str, Any]],
    reference: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise ActionCycleAnalysisError("arm clip/endpoint inventories differ")
    fields = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "sampler_actions_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
    )
    for key in sorted(candidate):
        left, right = candidate[key], reference[key]
        if (
            left["clip_id"] != right["clip_id"]
            or left["sampling_id"] != right["sampling_id"]
            or left["actual_wan_call_count"] != right["actual_wan_call_count"]
            or any(
                left["tensor_sha256"][field] != right["tensor_sha256"][field]
                for field in fields
            )
        ):
            raise ActionCycleAnalysisError(f"cross-arm pairing differs at {key}")
    return {
        "paired_clip_endpoint_keys": len(candidate),
        "input_target_noise_and_actions_exact": True,
        "same_nfe_and_actual_wan_calls": True,
        "critic_or_clean_future_feature_at_inference": False,
    }


def _training_events(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(inventory["arm_artifacts"]["training_trace"]["trace"]["path"])
    events = []
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("kind") != "action_cycle_stage1_training_trace_header":
            raise ActionCycleAnalysisError("training trace header differs")
        for line in handle:
            value = json.loads(line)
            if "train_loss/loss" in value.get("metrics", {}):
                events.append(value)
    if len(events) != 200:
        raise ActionCycleAnalysisError("training trace does not contain 200 updates")
    return events


def _assert_training_pair(
    candidate_inventory: Mapping[str, Any], reference_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = _training_events(candidate_inventory)
    reference = _training_events(reference_inventory)
    for index, (left, right) in enumerate(zip(candidate, reference)):
        lm, rm = left["metrics"], right["metrics"]
        if (
            lm.get("iteration") != index
            or rm.get("iteration") != index
            or left.get("total_observations") != (index + 1) * 8
            or right.get("total_observations") != (index + 1) * 8
            or lm.get("learning_rate") != rm.get("learning_rate")
        ):
            raise ActionCycleAnalysisError(f"training schedule differs at {index}")
        for field in audit_schema.AUDIT_FIELDS:
            if (
                isinstance(lm.get(field), bool)
                or not isinstance(lm.get(field), (int, float))
                or not math.isfinite(float(lm[field]))
                or lm[field] != rm.get(field)
            ):
                raise ActionCycleAnalysisError(f"paired input differs at {index}: {field}")
        for field in audit_schema.AUDIT_HASH_FIELDS:
            if (
                not isinstance(lm.get(field), str)
                or protocol.SHA256_RE.fullmatch(lm[field]) is None
                or lm[field] != rm.get(field)
            ):
                raise ActionCycleAnalysisError(f"exact paired hash differs: {field}")
    return {
        "updates": 200,
        "same_exact_data_actions_latents_clocks_and_rng": True,
        "same_learning_rate_schedule": True,
        "same_parent_and_fresh_optimizer_policy": True,
        "loss_off_computes_same_critic_diagnostic": True,
    }


def analyze(on_path: Path, off_path: Path) -> dict[str, Any]:
    on_arm = protocol.ARM_BY_CODE["AC-ON"]
    off_arm = protocol.ARM_BY_CODE["AC-OFF"]
    on_inventory, on_registration, on_rows_raw = _load_inventory(on_path, on_arm)
    off_inventory, off_registration, off_rows_raw = _load_inventory(off_path, off_arm)
    if on_registration["identity_sha256"] != off_registration["identity_sha256"]:
        raise ActionCycleAnalysisError("arms do not share one registration")
    on_rows, off_rows = _index(on_rows_raw), _index(off_rows_raw)
    evaluation_pair = _assert_evaluation_pair(on_rows, off_rows)
    training_pair = _assert_training_pair(on_inventory, off_inventory)

    quality = {
        metric: paired_effect(
            _metric(on_rows, "aligned_nfe_1", metric),
            _metric(off_rows, "aligned_nfe_1", metric),
            label=f"quality:AC-ON-vs-AC-OFF:aligned-nfe1:{metric}",
        )
        for metric in QUALITY_METRICS
    }
    candidate_attribution = {}
    for source in ("shuffled", "zero"):
        candidate_attribution[source] = {
            metric: paired_effect(
                _metric(on_rows, "aligned_nfe_1", metric),
                _metric(on_rows, f"{source}_nfe_1", metric),
                label=f"attribution:AC-ON:aligned-vs-{source}:{metric}",
            )
            for metric in ATTRIBUTION_METRICS
        }
    did_shuffled = {
        metric: difference_in_differences(
            candidate_aligned=_metric(on_rows, "aligned_nfe_1", metric),
            candidate_control=_metric(on_rows, "shuffled_nfe_1", metric),
            reference_aligned=_metric(off_rows, "aligned_nfe_1", metric),
            reference_control=_metric(off_rows, "shuffled_nfe_1", metric),
            label=f"did:aligned-vs-shuffled:{metric}",
            simultaneous=True,
        )
        for metric in ATTRIBUTION_METRICS
    }
    did_zero_descriptive = {
        metric: difference_in_differences(
            candidate_aligned=_metric(on_rows, "aligned_nfe_1", metric),
            candidate_control=_metric(on_rows, "zero_nfe_1", metric),
            reference_aligned=_metric(off_rows, "aligned_nfe_1", metric),
            reference_control=_metric(off_rows, "zero_nfe_1", metric),
            label=f"descriptive-did:aligned-vs-zero:{metric}",
            simultaneous=False,
        )
        for metric in ATTRIBUTION_METRICS
    }
    nfe4_descriptive = {
        metric: paired_effect(
            _metric(on_rows, "aligned_nfe_4", metric),
            _metric(off_rows, "aligned_nfe_4", metric),
            label=f"descriptive:nfe4:{metric}",
            simultaneous=False,
        )
        for metric in QUALITY_METRICS
    }

    checks = {
        "decoded_mse_point_at_least_3pct": quality[DECODED]["relative_improvement"]
        >= 0.03,
        "decoded_mse_lb_at_least_1pct": quality[DECODED]["one_sided_lower_bound"][
            "low"
        ]
        >= 0.01,
        "temporal_mse_point_at_least_3pct": quality[TEMPORAL][
            "relative_improvement"
        ]
        >= 0.03,
        "temporal_mse_lb_at_least_1pct": quality[TEMPORAL][
            "one_sided_lower_bound"
        ]["low"]
        >= 0.01,
        "latent_nmse_point_noninferior_1pct": quality[LATENT]["relative_improvement"]
        >= -0.01,
        "latent_nmse_lb_noninferior_1pct": quality[LATENT]["one_sided_lower_bound"][
            "low"
        ]
        >= -0.01,
    }
    for source in ("shuffled", "zero"):
        for metric in ATTRIBUTION_METRICS:
            effect = candidate_attribution[source][metric]
            checks[f"on_aligned_beats_{source}_{metric}_point"] = (
                effect["relative_improvement"] > 0.0
            )
            checks[f"on_aligned_beats_{source}_{metric}_lb"] = (
                effect["one_sided_lower_bound"]["low"] > 0.0
            )
    for metric in ATTRIBUTION_METRICS:
        effect = did_shuffled[metric]
        checks[f"shuffled_did_{metric}_point_positive"] = (
            effect["relative_difference_in_differences"] > 0.0
        )
        checks[f"shuffled_did_{metric}_lb_positive"] = (
            effect["one_sided_lower_bound"]["low"] > 0.0
        )
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_ANALYSIS,
        "created_at_utc": _now(),
        "status": "validation_only_no_protected_test",
        "registration_identity_sha256": on_registration["identity_sha256"],
        "candidate": {
            "code": on_arm.code,
            "config_name": on_arm.config_name,
            "run_name": on_arm.run_name,
            "loss_weight": on_arm.loss_weight,
        },
        "reference": {
            "code": off_arm.code,
            "config_name": off_arm.config_name,
            "run_name": off_arm.run_name,
            "loss_weight": off_arm.loss_weight,
        },
        "training_pair_audit": training_pair,
        "evaluation_pair_audit": evaluation_pair,
        "quality_aligned_nfe1": quality,
        "candidate_action_attribution_nfe1": candidate_attribution,
        "shuffled_action_difference_in_differences_nfe1": did_shuffled,
        "zero_action_difference_in_differences_descriptive": did_zero_descriptive,
        "aligned_nfe4_descriptive": nfe4_descriptive,
        "gate": {
            "passed": passed,
            "checks": checks,
            "rule": protocol.fixed_protocol()["analysis_gate"],
        },
        "decision": (
            "deployable_action_cycle_advantage_in_quick_screen"
            if passed
            else "no_preregistered_deployable_action_cycle_advantage_in_quick_screen"
        ),
        "claim_scope": (
            "A pass is one-seed val64 evidence for a target-free NFE-1 gain with "
            "action attribution; it is not protected-test, FVD, latency, or broad "
            "physical-realism evidence."
        ),
        "critic_loaded_or_called_at_inference": False,
        "online_teacher_call_count": 0,
        "target_cache_array_opened": False,
        "protected_test_accessed": False,
    }


def command(args: argparse.Namespace) -> int:
    output = args.output.expanduser()
    if not output.is_absolute() or output.name != "analysis.json":
        raise ActionCycleAnalysisError("output must be absolute analysis.json")
    result = protocol.identity_payload(analyze(args.on_inventory, args.off_inventory))
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(protocol.canonical_json(result) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--on-inventory", type=Path, required=True)
    value.add_argument("--off-inventory", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    raise SystemExit(command(parser().parse_args()))
