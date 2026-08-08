#!/usr/bin/env python3
"""Analyze the preregistered paired VPM macro motion-drift validation screen."""

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

from tools import lamo_motion_drift_evaluate as evaluation  # noqa: E402


SCHEMA_VERSION = 1
KIND_ANALYSIS = "lamo_motion_drift_validation_analysis"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_807
CONTRAST_COUNT = len(evaluation.NFE_GRID) * 3
ONE_SIDED_CONFIDENCE = 1.0 - 0.05 / CONTRAST_COUNT
PRIMARY_METRIC = "decoded_temporal_difference_mse_unit_range"
GUARDRAIL_METRICS = ("video_future_nmse", "decoded_mse_unit_range")
CLAIM_METRICS = (PRIMARY_METRIC, *GUARDRAIL_METRICS)
PRIMARY_MINIMUM_IMPROVEMENT = 0.01
GUARDRAIL_MAXIMUM_REGRESSION = 0.01
AUDIT_FIELDS = (
    "train_loss/paired_audit/clip_index_mean",
    "train_loss/paired_audit/clip_index_square_mean",
    "train_loss/paired_audit/timestep_mean",
    "train_loss/paired_audit/timestep_square_mean",
    "train_loss/paired_audit/noisy_probe",
    "train_loss/paired_audit/clean_probe",
)
AUDIT_HASH_FIELDS = tuple(
    f"train_loss/paired_audit/exact_{field}_all_ranks_sha256"
    for field in (
        "clip_index",
        "actions",
        "clean_latent",
        "noisy_latent",
        "timesteps",
        "cpu_rng_state_after_forward",
        "cuda_rng_state_after_forward",
    )
)


class MotionDriftAnalysisError(RuntimeError):
    """Evidence is incomplete, unmatched, or outside the frozen protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MotionDriftAnalysisError(f"refusing to overwrite analysis: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MotionDriftAnalysisError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MotionDriftAnalysisError(f"{label} must contain one object")
    return value


def _sha256(path: Path) -> str:
    return evaluation._sha256(path)


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def paired_effect(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    label: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = ONE_SIDED_CONFIDENCE,
    family_contrast_count: int | None = CONTRAST_COUNT,
) -> dict[str, Any]:
    """Return paired relative improvement; positive always favors candidate."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or left.size != evaluation.EXPECTED_VALIDATION_CLIPS
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(right <= 0)
        or bootstrap_samples < 100
        or not 0.5 < confidence < 1.0
    ):
        raise MotionDriftAnalysisError(f"invalid paired values for {label}")
    relative = (right.mean() - left.mean()) / right.mean()
    seed = _derived_seed(label)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0, left.size, size=(bootstrap_samples, left.size), endpoint=False
    )
    left_means = left[indexes].mean(axis=1)
    right_means = right[indexes].mean(axis=1)
    effects = (right_means - left_means) / right_means
    descriptive_low, descriptive_high = np.quantile(effects, (0.025, 0.975))
    lower_bound = {
        "confidence": confidence,
        "low": float(np.quantile(effects, 1.0 - confidence)),
    }
    bound_key = (
        "one_sided_simultaneous_lower_bound"
        if family_contrast_count is not None
        else "one_sided_descriptive_lower_bound"
    )
    if family_contrast_count is not None:
        lower_bound.update(
            {
                "familywise_alpha": 0.05,
                "family_contrast_count": int(family_contrast_count),
            }
        )
    return {
        "n_paired_clips": int(left.size),
        "candidate_mean": float(left.mean()),
        "reference_mean": float(right.mean()),
        "mean_favorable_delta": float((right - left).mean()),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(100.0 * relative),
        bound_key: lower_bound,
        "descriptive_two_sided_95_ci": {
            "low": float(descriptive_low),
            "high": float(descriptive_high),
        },
        "favorable_clip_fraction": float(np.mean(right > left)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_derived_seed": seed,
        "bootstrap_unit": "paired immutable validation clip",
    }


def endpoint_gate(effects: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, tuple[float, float]] = {}
    for metric in CLAIM_METRICS:
        effect = effects.get(metric)
        if not isinstance(effect, Mapping):
            raise MotionDriftAnalysisError(f"missing effect for {metric}")
        point = effect.get("relative_improvement")
        low = effect.get("one_sided_simultaneous_lower_bound", {}).get("low")
        if (
            isinstance(point, bool)
            or not isinstance(point, (int, float))
            or isinstance(low, bool)
            or not isinstance(low, (int, float))
            or not math.isfinite(float(point))
            or not math.isfinite(float(low))
        ):
            raise MotionDriftAnalysisError(f"non-finite effect for {metric}")
        values[metric] = (float(point), float(low))
    primary_point, primary_low = values[PRIMARY_METRIC]
    checks = {
        "temporal_point_at_least_one_percent": (
            primary_point >= PRIMARY_MINIMUM_IMPROVEMENT
        ),
        "temporal_simultaneous_lb_at_least_one_percent": (
            primary_low >= PRIMARY_MINIMUM_IMPROVEMENT
        ),
    }
    for metric in GUARDRAIL_METRICS:
        point, low = values[metric]
        checks[f"{metric}_point_above_minus_one_percent"] = (
            point > -GUARDRAIL_MAXIMUM_REGRESSION
        )
        checks[f"{metric}_simultaneous_lb_above_minus_one_percent"] = (
            low > -GUARDRAIL_MAXIMUM_REGRESSION
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rule": (
            "decoded temporal MSE point and simultaneous LB >= +1%; latent "
            "NMSE and decoded MSE point/LB each > -1%"
        ),
    }


def _load_inventory(
    path: Path, expected_arm: evaluation.Arm
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve(strict=True)
    inventory = _read_json(path, "evaluation inventory")
    if (
        not evaluation.identity_valid(inventory)
        or inventory.get("kind") != evaluation.KIND_INVENTORY
        or inventory.get("arm") != asdict_arm(expected_arm)
        or inventory.get("evaluation_split") != "validation"
        or inventory.get("validation_clips")
        != evaluation.EXPECTED_VALIDATION_CLIPS
        or inventory.get("endpoints")
        != [asdict_endpoint(endpoint) for endpoint in evaluation.ENDPOINTS]
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("target_cache_array_opened") is not False
        or inventory.get("online_feature_or_teacher_call_count") != 0
        or inventory.get("row_count")
        != evaluation.EXPECTED_VALIDATION_CLIPS * len(evaluation.ENDPOINTS)
        or inventory.get("paired_inputs_and_noise_within_arm") is not True
        or inventory.get("actual_transformer_call_count") != 256
    ):
        raise MotionDriftAnalysisError("evaluation inventory differs")
    registration_record = inventory.get("registration")
    if not isinstance(registration_record, Mapping):
        raise MotionDriftAnalysisError("inventory lacks registration")
    registration_path = Path(str(registration_record.get("path", "")))
    if (
        not registration_path.is_file()
        or registration_record.get("sha256") != _sha256(registration_path)
    ):
        raise MotionDriftAnalysisError("registered evidence changed")
    registration = evaluation._validate_registration(registration_path)
    if (
        inventory.get("registration_identity_sha256")
        != registration["identity_sha256"]
    ):
        raise MotionDriftAnalysisError("inventory registration identity differs")
    arm_artifacts = inventory.get("arm_artifacts")
    if not isinstance(arm_artifacts, Mapping):
        raise MotionDriftAnalysisError("inventory arm artifacts are absent")
    expected_run_identity = evaluation.arm_run_identity(
        registration, expected_arm
    )
    if arm_artifacts.get("run_identity_sha256") != expected_run_identity:
        raise MotionDriftAnalysisError("inventory arm run identity differs")

    def require_record(record: Any, label: str) -> None:
        if not isinstance(record, Mapping):
            raise MotionDriftAnalysisError(f"{label} record is absent")
        record_path = Path(str(record.get("path", "")))
        if (
            not record_path.is_file()
            or record_path.is_symlink()
            or record.get("bytes") != record_path.stat().st_size
            or record.get("sha256") != _sha256(record_path)
        ):
            raise MotionDriftAnalysisError(f"{label} artifact changed")

    require_record(arm_artifacts.get("snapshot"), "arm snapshot")
    require_record(arm_artifacts.get("resolved_config"), "arm config")
    require_record(
        arm_artifacts.get("training_completion"), "training completion"
    )
    require_record(
        arm_artifacts.get("arm_execution_plan", {}).get("record"),
        "arm execution plan",
    )
    trace_artifacts = arm_artifacts.get("training_trace", {})
    require_record(trace_artifacts.get("trace"), "training trace")
    require_record(trace_artifacts.get("completion"), "training trace completion")
    rows: list[dict[str, Any]] = []
    rank_records = inventory.get("rank_manifests")
    if not isinstance(rank_records, list) or len(rank_records) != 8:
        raise MotionDriftAnalysisError("rank inventory count differs")
    for expected_rank, rank_record in enumerate(rank_records):
        rank_path = Path(str(rank_record.get("path", "")))
        if (
            not rank_path.is_file()
            or rank_path.is_symlink()
            or rank_record.get("bytes") != rank_path.stat().st_size
            or rank_record.get("sha256") != _sha256(rank_path)
        ):
            raise MotionDriftAnalysisError("rank manifest changed")
        manifest = _read_json(rank_path, "rank manifest")
        if (
            not evaluation.identity_valid(manifest)
            or manifest.get("kind") != evaluation.KIND_RANK
            or manifest.get("rank") != expected_rank
            or manifest.get("arm") != asdict_arm(expected_arm)
            or manifest.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or manifest.get("world_size") != evaluation.EXPECTED_WORLD_SIZE
            or manifest.get("batch_size_per_rank")
            != evaluation.EXPECTED_BATCH_SIZE_PER_RANK
            or manifest.get("assigned_clip_indexes")
            != evaluation._expected_rank_indexes(expected_rank)
            or manifest.get("endpoints")
            != [asdict_endpoint(endpoint) for endpoint in evaluation.ENDPOINTS]
            or manifest.get("actual_transformer_call_count") != 32
            or manifest.get("protected_test_accessed") is not False
            or manifest.get("target_cache_array_opened") is not False
        ):
            raise MotionDriftAnalysisError("rank manifest identity differs")
        row_record = manifest.get("rows")
        row_path = Path(str(row_record.get("path", "")))
        if (
            not row_path.is_file()
            or row_path.is_symlink()
            or row_record.get("bytes") != row_path.stat().st_size
            or row_record.get("sha256") != _sha256(row_path)
            or row_record.get("count")
            != len(evaluation._expected_rank_indexes(expected_rank))
            * len(evaluation.ENDPOINTS)
        ):
            raise MotionDriftAnalysisError("rank JSONL changed")
        with row_path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    evaluation._validate_rows(rows, expected_arm, registration)
    return inventory, registration, rows


def asdict_arm(arm: evaluation.Arm) -> dict[str, Any]:
    return {
        "code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "motion_drift_weight": arm.motion_drift_weight,
    }


def asdict_endpoint(endpoint: evaluation.Endpoint) -> dict[str, Any]:
    return {
        "code": endpoint.code,
        "nfe": endpoint.nfe,
        "action_source": endpoint.action_source,
        "primary_gate": endpoint.primary_gate,
    }


def _rows_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(row["clip_index"]), str(row["endpoint"]["code"])): row
        for row in rows
    }


def _assert_cross_arm_pairing(
    candidate: Mapping[tuple[int, str], Mapping[str, Any]],
    reference: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise MotionDriftAnalysisError("arm clip/endpoint inventories differ")
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
        left = candidate[key]
        right = reference[key]
        if (
            left.get("clip_id") != right.get("clip_id")
            or left.get("sampling_id") != right.get("sampling_id")
            or left.get("actual_transformer_call_count")
            != right.get("actual_transformer_call_count")
            or any(
                left["tensor_sha256"].get(field)
                != right["tensor_sha256"].get(field)
                for field in fields
            )
        ):
            raise MotionDriftAnalysisError(f"cross-arm pairing differs at {key}")
    return {
        "paired_clip_endpoint_keys": len(candidate),
        "input_target_noise_hashes_exact": True,
        "same_nfe_and_actual_call_counts": True,
        "future_or_feature_input_to_sampler": False,
    }


def _trace_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace_path = Path(str(record["arm_artifacts"]["training_trace"]["trace"]["path"]))
    expected_sha = record["arm_artifacts"]["training_trace"]["trace"]["sha256"]
    if _sha256(trace_path) != expected_sha:
        raise MotionDriftAnalysisError("training trace changed after evaluation")
    events = []
    with trace_path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("kind") != "lamo_motion_drift_training_trace_header":
            raise MotionDriftAnalysisError("training trace header is invalid")
        for line in handle:
            value = json.loads(line)
            metrics = value.get("metrics", {})
            if "train_loss/loss" in metrics:
                events.append(value)
    if len(events) != 200:
        raise MotionDriftAnalysisError(
            f"expected 200 training trace events, found {len(events)}"
        )
    return events


def _assert_training_pair(
    candidate_inventory: Mapping[str, Any],
    reference_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _trace_events(candidate_inventory)
    reference = _trace_events(reference_inventory)
    for index, (left, right) in enumerate(zip(candidate, reference)):
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        if (
            left_metrics.get("iteration") != index
            or right_metrics.get("iteration") != index
            or left.get("total_observations") != (index + 1) * 8
            or right.get("total_observations") != (index + 1) * 8
            or left.get("total_observations") != right.get("total_observations")
            or left_metrics.get("learning_rate") != right_metrics.get("learning_rate")
            or isinstance(left_metrics.get("learning_rate"), bool)
            or not isinstance(left_metrics.get("learning_rate"), (int, float))
            or not math.isfinite(float(left_metrics["learning_rate"]))
            or float(left_metrics["learning_rate"]) <= 0
        ):
            raise MotionDriftAnalysisError(f"training schedule differs at update {index}")
        for field in AUDIT_FIELDS:
            left_value = left_metrics.get(field)
            right_value = right_metrics.get(field)
            if (
                isinstance(left_value, bool)
                or not isinstance(left_value, (int, float))
                or not math.isfinite(float(left_value))
                or left_value != right_value
            ):
                raise MotionDriftAnalysisError(
                    f"paired data/noise audit differs at update {index}: {field}"
                )
        for field in AUDIT_HASH_FIELDS:
            left_value = left_metrics.get(field)
            right_value = right_metrics.get(field)
            if (
                not isinstance(left_value, str)
                or evaluation.SHA256_RE.fullmatch(left_value) is None
                or left_value != right_value
            ):
                raise MotionDriftAnalysisError(
                    f"exact paired tensor/RNG hash differs at update {index}: {field}"
                )
    return {
        "updates": len(candidate),
        "same_clip_order_clock_and_corruption_audit": True,
        "same_exact_actions_latents_clocks_and_rng_hashes": True,
        "same_learning_rate_schedule": True,
        "same_parent_snapshot_sha256": evaluation.PARENT_SNAPSHOT_SHA256,
        "same_fresh_optimizer_policy": "fresh_identical_adamw",
        "same_ema_policy": "none_in_historical_lacwm_and_none_in_both_arms",
        "lambda_zero_executes_same_drift_diagnostic_codepath": True,
    }


def _metric_values(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]],
    endpoint: str,
    metric: str,
) -> list[float]:
    return [
        float(indexed[(index, endpoint)]["metrics"][metric])
        for index in range(evaluation.EXPECTED_VALIDATION_CLIPS)
    ]


def analyze(
    candidate_inventory_path: Path,
    reference_inventory_path: Path,
) -> dict[str, Any]:
    drift_arm = evaluation.ARM_BY_CODE["VPM-DRIFT"]
    baseline_arm = evaluation.ARM_BY_CODE["VPM-CONT"]
    candidate_inventory, candidate_registration, candidate_rows = _load_inventory(
        candidate_inventory_path, drift_arm
    )
    reference_inventory, reference_registration, reference_rows = _load_inventory(
        reference_inventory_path, baseline_arm
    )
    if (
        candidate_registration["identity_sha256"]
        != reference_registration["identity_sha256"]
    ):
        raise MotionDriftAnalysisError("arms do not share one registration")
    candidate = _rows_by_key(candidate_rows)
    reference = _rows_by_key(reference_rows)
    pairing = _assert_cross_arm_pairing(candidate, reference)
    training_pair = _assert_training_pair(candidate_inventory, reference_inventory)

    endpoints: dict[str, Any] = {}
    passing = []
    for nfe in evaluation.NFE_GRID:
        code = f"autonomous_nfe_{nfe}"
        effects = {
            metric: paired_effect(
                _metric_values(candidate, code, metric),
                _metric_values(reference, code, metric),
                label=f"VPM-DRIFT-vs-VPM-CONT:nfe-{nfe}:{metric}",
            )
            for metric in CLAIM_METRICS
        }
        gate = endpoint_gate(effects)
        endpoints[str(nfe)] = {"effects": effects, "gate": gate}
        if gate["passed"]:
            passing.append(nfe)

    # Diagnostic only: action alignment is never part of the primary gate or
    # the endpoint-selection family.
    action_diagnostic: dict[str, Any] = {}
    for arm_code, indexed in (
        ("VPM-CONT", reference),
        ("VPM-DRIFT", candidate),
    ):
        action_diagnostic[arm_code] = {
            metric: paired_effect(
                _metric_values(indexed, "autonomous_nfe_1", metric),
                _metric_values(indexed, "actions_shuffled_nfe_1", metric),
                label=f"diagnostic:{arm_code}:matched-vs-action-shuffled:{metric}",
                confidence=0.95,
                family_contrast_count=None,
            )
            for metric in CLAIM_METRICS
        }
    selected_nfe = min(passing) if passing else None
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_ANALYSIS,
        "created_at_utc": _now(),
        "status": "validation_only_no_protected_test",
        "registration_identity_sha256": candidate_registration["identity_sha256"],
        "candidate": asdict_arm(drift_arm),
        "reference": asdict_arm(baseline_arm),
        "training_pair_audit": training_pair,
        "evaluation_pair_audit": pairing,
        "primary_metric": PRIMARY_METRIC,
        "guardrail_metrics": list(GUARDRAIL_METRICS),
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "family_contrast_count": CONTRAST_COUNT,
            "one_sided_simultaneous_confidence": ONE_SIDED_CONFIDENCE,
        },
        "nfe_endpoints": endpoints,
        "passing_nfe": passing,
        "selected_lowest_passing_nfe": selected_nfe,
        "validation_gate_passed": selected_nfe is not None,
        "action_shuffled_nfe1_diagnostic_not_in_gate": action_diagnostic,
        "claim_scope": (
            "A pass is validation-only evidence that the parameter-free training "
            "loss improves at least one fixed low-NFE endpoint; it is not a "
            "protected-test, latency, or general physical-realism claim."
        ),
        "protected_test_accessed": False,
        "target_cache_array_opened": False,
    }


def command_analyze(args: argparse.Namespace) -> int:
    output = args.output.expanduser()
    if not output.is_absolute() or output.name != "analysis.json":
        raise MotionDriftAnalysisError("analysis output must be absolute analysis.json")
    result = evaluation.identity_payload(
        analyze(args.drift_inventory, args.baseline_inventory)
    )
    _exclusive_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-inventory", type=Path, required=True)
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return command_analyze(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
