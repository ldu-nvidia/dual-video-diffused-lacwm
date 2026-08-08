#!/usr/bin/env python3
"""Analyze the preregistered paired VPM two-clock-consistency screen."""

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

from tools import two_clock_consistency_evaluate as evaluation  # noqa: E402


SCHEMA_VERSION = 1
KIND_ANALYSIS = "two_clock_consistency_validation_analysis"
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
    "train_loss/paired_audit/epsilon_probe",
    "train_loss/paired_audit/clean_probe",
    "train_loss/paired_audit/noisy_hi_probe",
    "train_loss/paired_audit/noisy_lo_probe",
    "train_loss/paired_audit/sigma_hi_mean",
    "train_loss/paired_audit/sigma_lo_mean",
    "train_loss/two_clock_consistency/model_calls",
    "train_loss/two_clock_consistency/shared_epsilon_trajectory",
)
AUDIT_HASH_FIELDS = tuple(
    f"train_loss/paired_audit/exact_{field}_all_ranks_sha256"
    for field in (
        "clip_index",
        "actions",
        "clean_latent",
        "epsilon",
        "sigma_hi",
        "sigma_lo",
        "timestep_hi",
        "timestep_lo",
        "noisy_hi",
        "noisy_lo",
        "cpu_rng_state_after_forward",
        "cuda_rng_state_after_forward",
    )
)


class TwoClockConsistencyAnalysisError(RuntimeError):
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
        raise TwoClockConsistencyAnalysisError(f"refusing to overwrite analysis: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TwoClockConsistencyAnalysisError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TwoClockConsistencyAnalysisError(f"{label} must contain one object")
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
    family_contrast_count: int = CONTRAST_COUNT,
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
        or isinstance(family_contrast_count, bool)
        or not isinstance(family_contrast_count, int)
        or family_contrast_count < 1
    ):
        raise TwoClockConsistencyAnalysisError(f"invalid paired values for {label}")
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
    return {
        "n_paired_clips": int(left.size),
        "candidate_mean": float(left.mean()),
        "reference_mean": float(right.mean()),
        "mean_favorable_delta": float((right - left).mean()),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(100.0 * relative),
        "one_sided_simultaneous_lower_bound": {
            "confidence": confidence,
            "familywise_alpha": float(
                (1.0 - confidence) * family_contrast_count
            ),
            "family_contrast_count": family_contrast_count,
            "low": float(np.quantile(effects, 1.0 - confidence)),
        },
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
            raise TwoClockConsistencyAnalysisError(f"missing effect for {metric}")
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
            raise TwoClockConsistencyAnalysisError(f"non-finite effect for {metric}")
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
        or inventory.get("actual_transformer_call_count")
        != evaluation.EXPECTED_WORLD_SIZE
        * evaluation._expected_rank_transformer_calls()
        or not isinstance(inventory.get("arm_artifacts"), Mapping)
    ):
        raise TwoClockConsistencyAnalysisError("evaluation inventory differs")
    registration_record = inventory.get("registration")
    if not isinstance(registration_record, Mapping):
        raise TwoClockConsistencyAnalysisError("inventory lacks registration")
    registration_path = Path(str(registration_record.get("path", "")))
    observed_registration = (
        evaluation._file_record(registration_path)
        if registration_path.is_file() and not registration_path.is_symlink()
        else None
    )
    if observed_registration is None or any(
        registration_record.get(field) != observed_registration.get(field)
        for field in ("path", "bytes", "sha256")
    ):
        raise TwoClockConsistencyAnalysisError("registered evidence changed")
    registration = evaluation._validate_registration(registration_path)
    if (
        inventory.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or path
        != Path(registration["output_root"])
        / "evaluation"
        / expected_arm.code.lower()
        / "inventory.json"
    ):
        raise TwoClockConsistencyAnalysisError("inventory registration identity differs")
    arm_artifacts = inventory["arm_artifacts"]
    expected_run_identity = evaluation.arm_run_identity(
        registration, expected_arm
    )
    try:
        if arm_artifacts.get("run_identity_sha256") != expected_run_identity:
            raise evaluation.TwoClockConsistencyEvaluationError(
                "arm run identity differs"
            )
        for key in ("snapshot", "resolved_config", "training_completion"):
            evaluation._revalidate_record(
                arm_artifacts[key], f"arm artifact {key}"
            )
        evaluation._revalidate_record(
            arm_artifacts["arm_execution_plan"]["record"],
            "arm execution plan",
        )
        evaluation._revalidate_record(
            arm_artifacts["training_trace"]["trace"],
            "training trace",
        )
        evaluation._revalidate_record(
            arm_artifacts["training_trace"]["completion"],
            "training trace completion",
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        evaluation.TwoClockConsistencyEvaluationError,
    ) as exc:
        raise TwoClockConsistencyAnalysisError(
            "arm artifact provenance differs"
        ) from exc
    rows: list[dict[str, Any]] = []
    rank_records = inventory.get("rank_manifests")
    if not isinstance(rank_records, list) or len(rank_records) != 8:
        raise TwoClockConsistencyAnalysisError("rank inventory count differs")
    for expected_rank, rank_record in enumerate(rank_records):
        expected_rank_path = path.parent / f"rank_{expected_rank:03d}.json"
        rank_path = Path(str(rank_record.get("path", "")))
        observed_rank_record = (
            evaluation._file_record(rank_path)
            if rank_path.is_file() and not rank_path.is_symlink()
            else None
        )
        if (
            rank_path != expected_rank_path
            or observed_rank_record is None
            or any(
                rank_record.get(field) != observed_rank_record.get(field)
                for field in ("path", "bytes", "sha256")
            )
        ):
            raise TwoClockConsistencyAnalysisError("rank manifest changed")
        manifest = _read_json(rank_path, "rank manifest")
        try:
            row_path = evaluation._validate_rank_manifest(
                manifest,
                expected_rank=expected_rank,
                arm=expected_arm,
                registration=registration,
                output_dir=path.parent,
            )
        except evaluation.TwoClockConsistencyEvaluationError as exc:
            raise TwoClockConsistencyAnalysisError(
                "rank manifest identity differs"
            ) from exc
        with row_path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    evaluation._validate_rows(rows, expected_arm, registration)
    if any(
        row.get("tool_git_commit")
        != registration["tool_repository"]["git_commit"]
        or row.get("training_git_commit") != evaluation.TRAINING_COMMIT
        or row.get("arm_snapshot") != arm_artifacts["snapshot"]
        for row in rows
    ):
        raise TwoClockConsistencyAnalysisError(
            "row source or arm-snapshot provenance differs"
        )
    return inventory, registration, rows


def asdict_arm(arm: evaluation.Arm) -> dict[str, Any]:
    return {
        "code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "consistency_weight": arm.consistency_weight,
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
        raise TwoClockConsistencyAnalysisError("arm clip/endpoint inventories differ")
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
            raise TwoClockConsistencyAnalysisError(f"cross-arm pairing differs at {key}")
    return {
        "paired_clip_endpoint_keys": len(candidate),
        "input_target_noise_hashes_exact": True,
        "same_nfe_and_actual_call_counts": True,
        "future_or_feature_input_to_sampler": False,
    }


def _trace_events(
    record: Mapping[str, Any], *, expected_arm: str
) -> list[dict[str, Any]]:
    trace_path = Path(str(record["arm_artifacts"]["training_trace"]["trace"]["path"]))
    expected_sha = record["arm_artifacts"]["training_trace"]["trace"]["sha256"]
    if _sha256(trace_path) != expected_sha:
        raise TwoClockConsistencyAnalysisError("training trace changed after evaluation")
    events = []
    try:
        with trace_path.open(encoding="utf-8") as handle:
            header = json.loads(next(handle))
            if (
                header.get("kind")
                != "two_clock_consistency_training_trace_header"
                or header.get("arm") != expected_arm
            ):
                raise TwoClockConsistencyAnalysisError(
                    "training trace header is invalid"
                )
            for line_number, line in enumerate(handle, start=2):
                if not line.strip():
                    continue
                value = json.loads(line)
                if (
                    not isinstance(value, dict)
                    or value.get("kind")
                    != "two_clock_consistency_training_trace_event"
                    or value.get("arm") != expected_arm
                    or isinstance(value.get("total_observations"), bool)
                    or not isinstance(value.get("total_observations"), int)
                ):
                    raise TwoClockConsistencyAnalysisError(
                        f"invalid training trace event at line {line_number}"
                    )
                metrics = value.get("metrics")
                if not isinstance(metrics, Mapping):
                    raise TwoClockConsistencyAnalysisError(
                        f"training trace metrics are invalid at line {line_number}"
                    )
                if "train_loss/loss" in metrics:
                    events.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, StopIteration) as exc:
        raise TwoClockConsistencyAnalysisError(
            "training trace is invalid JSONL"
        ) from exc
    if len(events) != 200:
        raise TwoClockConsistencyAnalysisError(
            f"expected 200 training trace events, found {len(events)}"
        )
    for update, event in enumerate(events):
        metrics = event["metrics"]
        if (
            metrics.get("iteration") != update
            or metrics.get("total_observations") != (update + 1) * 8
            or event.get("total_observations") != (update + 1) * 8
            or metrics.get("train_loss/two_clock_consistency/model_calls") != 2.0
            or metrics.get(
                "train_loss/two_clock_consistency/shared_epsilon_trajectory"
            )
            != 1.0
        ):
            raise TwoClockConsistencyAnalysisError(
                f"fixed training trace contract differs at update {update}"
            )
        high_sigma = metrics.get("train_loss/paired_audit/sigma_hi_mean")
        low_sigma = metrics.get("train_loss/paired_audit/sigma_lo_mean")
        if (
            isinstance(high_sigma, bool)
            or not isinstance(high_sigma, (int, float))
            or isinstance(low_sigma, bool)
            or not isinstance(low_sigma, (int, float))
            or not 0.8 <= float(high_sigma) <= 1.0
            or not 0.0 <= float(low_sigma) <= 0.4
            or not float(high_sigma) > float(low_sigma)
        ):
            raise TwoClockConsistencyAnalysisError(
                f"training clock bands differ at update {update}"
            )
        learning_rate = metrics.get("learning_rate")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or not 0.0 < float(learning_rate) <= 1e-4
        ):
            raise TwoClockConsistencyAnalysisError(
                f"training learning rate differs at update {update}"
            )
        for field in AUDIT_FIELDS:
            value = metrics.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TwoClockConsistencyAnalysisError(
                    f"training audit field is invalid at update {update}: {field}"
                )
        for field in AUDIT_HASH_FIELDS:
            value = metrics.get(field)
            if (
                not isinstance(value, str)
                or evaluation.SHA256_RE.fullmatch(value) is None
            ):
                raise TwoClockConsistencyAnalysisError(
                    f"training audit hash is invalid at update {update}: {field}"
                )
    return events


def _assert_training_pair(
    candidate_inventory: Mapping[str, Any],
    reference_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _trace_events(candidate_inventory, expected_arm="TC-CONS")
    reference = _trace_events(reference_inventory, expected_arm="TC-CONT")
    for index, (left, right) in enumerate(zip(candidate, reference)):
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        if (
            left_metrics.get("iteration") != right_metrics.get("iteration")
            or left.get("total_observations") != right.get("total_observations")
            or left_metrics.get("learning_rate") != right_metrics.get("learning_rate")
        ):
            raise TwoClockConsistencyAnalysisError(f"training schedule differs at update {index}")
        for field in AUDIT_FIELDS:
            left_value = left_metrics.get(field)
            right_value = right_metrics.get(field)
            if (
                isinstance(left_value, bool)
                or not isinstance(left_value, (int, float))
                or not math.isfinite(float(left_value))
                or left_value != right_value
            ):
                raise TwoClockConsistencyAnalysisError(
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
                raise TwoClockConsistencyAnalysisError(
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
        "zero_weight_executes_same_two_clock_diagnostic_codepath": True,
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
    candidate_arm = evaluation.ARM_BY_CODE["TC-CONS"]
    baseline_arm = evaluation.ARM_BY_CODE["TC-CONT"]
    candidate_inventory, candidate_registration, candidate_rows = _load_inventory(
        candidate_inventory_path, candidate_arm
    )
    reference_inventory, reference_registration, reference_rows = _load_inventory(
        reference_inventory_path, baseline_arm
    )
    if (
        candidate_registration["identity_sha256"]
        != reference_registration["identity_sha256"]
    ):
        raise TwoClockConsistencyAnalysisError("arms do not share one registration")
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
                label=f"TC-CONS-vs-TC-CONT:nfe-{nfe}:{metric}",
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
        ("TC-CONT", reference),
        ("TC-CONS", candidate),
    ):
        action_diagnostic[arm_code] = {
            metric: paired_effect(
                _metric_values(indexed, "autonomous_nfe_1", metric),
                _metric_values(indexed, "actions_shuffled_nfe_1", metric),
                label=f"diagnostic:{arm_code}:matched-vs-action-shuffled:{metric}",
                confidence=0.95,
                family_contrast_count=1,
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
        "candidate": asdict_arm(candidate_arm),
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
            "A pass is validation-only evidence that stopped two-clock training "
            "consistency improves at least one fixed low-NFE endpoint; it is not a "
            "protected-test, latency, or general physical-realism claim."
        ),
        "protected_test_accessed": False,
        "target_cache_array_opened": False,
    }


def command_analyze(args: argparse.Namespace) -> int:
    output = args.output.expanduser()
    if not output.is_absolute() or output.name != "analysis.json":
        raise TwoClockConsistencyAnalysisError("analysis output must be absolute analysis.json")
    result = evaluation.identity_payload(
        analyze(args.candidate_inventory, args.baseline_inventory)
    )
    _exclusive_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return command_analyze(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
