#!/usr/bin/env python3
"""Leakage-safe confidence-gating audit for existing intra-forward rows.

The tool never evaluates a model.  It consumes already-written development
validation rows, refuses target-derived confidence features, and either fits
the prospectively specified nested leave-one-out ridge gate or reports why no
such gate can be fit.  In both cases it reports target-leaking oracle ceilings
as explicitly unattainable diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "confidence-gated-generated-auxiliary-audit-v1"
PRIMARY_NFE = 1
EXPECTED_CLIPS = 64
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 10_000
METRICS = ("video_nmse", "decoded_mse", "temporal_mse")
ALIGNED_SOURCE = "autonomous"
OFF_SOURCE = "off"
CORRUPTED_SOURCE = "autonomous_future_shuffled"

# Every one of these fields uses a clean future target.  They may be outcomes
# or oracle diagnostics but can never enter a deployable confidence rule.
TARGET_DERIVED_FIELDS = frozenset(
    {
        "video_nmse",
        "decoded_mse",
        "temporal_mse",
        "auxiliary_future_nmse",
        "auxiliary_dc_nmse",
        "auxiliary_motion_nmse",
        "auxiliary_future_cosine",
    }
)

# These are semantically meaningful forms of inference-time telemetry that a
# future evaluator could preserve.  The existing artifact records none of
# them.  Explicit names prevent accidental use of latency, memory, IDs, or
# target accuracy simply because those columns happen to be numeric.
ELIGIBLE_CONFIDENCE_FIELDS = (
    "generated_auxiliary_rms",
    "generated_auxiliary_future_rms",
    "generated_auxiliary_motion_rms",
    "midpoint_prediction_rms",
    "midpoint_residual_rms",
    "midpoint_self_consistency_error",
    "midpoint_head_disagreement",
    "predicted_auxiliary_confidence",
)

AUDIT_ONLY_NUMERIC_FIELDS = frozenset(
    {
        "actual_wan_calls",
        "artifact_midpoint_head_calls",
        "audit_batch_size",
        "clip_index",
        "cross_batch_decoded_differing_fraction",
        "cross_batch_decoded_max_abs_uint8",
        "cross_batch_decoded_mean_abs_uint8",
        "decode_latency_ms",
        "effective_clock_gate",
        "effective_state_gate",
        "end_to_end_latency_ms",
        "equivalence_batch_size",
        "equivalence_midpoint_block_calls",
        "equivalence_midpoint_head_calls",
        "equivalence_transform_calls",
        "equivalence_wan_calls",
        "evaluation_generations_per_cell",
        "extra_wan_calls",
        "history_encode_latency_ms",
        "hook_midpoint_block_calls",
        "hook_midpoint_head_calls",
        "hook_wan_calls",
        "midpoint_block_index",
        "midpoint_overhead_latency_ms",
        "nfe",
        "online_teacher_calls",
        "peak_memory_allocated_bytes",
        "profiled_internal_end_to_end_latency_ms",
        "sampler_transform_calls",
        "timed_batch_size",
        "timed_midpoint_block_calls",
        "timed_midpoint_head_calls",
        "timed_wan_calls",
        "total_evaluation_wan_calls",
        "wan_block_count",
        "wan_latency_ms",
    }
)


class AuditError(RuntimeError):
    """Raised when artifact safety or pairing contracts fail closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise AuditError(f"{path}: no rows")
    return rows


def _require_safe_row(row: Mapping[str, Any]) -> None:
    required_false = (
        "protected_test_accessed",
        "oracle_leakage",
        "future_rgb_passed_to_sampler",
        "clean_auxiliary_passed_to_sampler",
    )
    for field in required_false:
        if row.get(field) is not False:
            raise AuditError(f"unsafe row: {field} must be exactly false")
    if row.get("deployable") is not True:
        raise AuditError("unsafe row: deployable must be exactly true")
    if row.get("online_teacher_calls") != 0:
        raise AuditError("unsafe row: online_teacher_calls must be zero")
    if row.get("nfe") != PRIMARY_NFE:
        raise AuditError(f"unexpected non-primary NFE {row.get('nfe')!r}")


def _index_primary_rows(
    rows: Iterable[Mapping[str, Any]], expected_arm: str
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    accepted_sources = {ALIGNED_SOURCE, OFF_SOURCE, CORRUPTED_SOURCE}
    for source_row in rows:
        if source_row.get("nfe") != PRIMARY_NFE:
            continue
        row = dict(source_row)
        if row.get("arm") != expected_arm:
            raise AuditError(
                f"expected arm {expected_arm}, observed {row.get('arm')!r}"
            )
        if row.get("source") not in accepted_sources:
            raise AuditError(f"unexpected source {row.get('source')!r}")
        _require_safe_row(row)
        clip = row.get("clip_index")
        if not isinstance(clip, int) or isinstance(clip, bool):
            raise AuditError("clip_index must be an integer")
        key = (str(row["source"]), clip)
        if key in indexed:
            raise AuditError(f"duplicate row {key}")
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise AuditError(f"{key}: {metric} is not finite numeric")
            if float(value) < 0:
                raise AuditError(f"{key}: {metric} is negative")
        indexed[key] = row

    expected_indices = set(range(EXPECTED_CLIPS))
    for source in accepted_sources:
        observed = {clip for row_source, clip in indexed if row_source == source}
        if observed != expected_indices:
            raise AuditError(
                f"{source}: expected clip indices 0..{EXPECTED_CLIPS - 1}, "
                f"observed {sorted(observed)}"
            )
    for clip in range(EXPECTED_CLIPS):
        rows_for_clip = [indexed[(source, clip)] for source in accepted_sources]
        target_hashes = {row.get("raw_target_sha256") for row in rows_for_clip}
        video_noise_hashes = {row.get("video_initial_sha256") for row in rows_for_clip}
        auxiliary_noise_hashes = {
            row.get("auxiliary_initial_sha256") for row in rows_for_clip
        }
        if len(target_hashes) != 1 or None in target_hashes:
            raise AuditError(f"clip {clip}: target identity is not paired")
        if len(video_noise_hashes) != 1 or None in video_noise_hashes:
            raise AuditError(f"clip {clip}: video noise is not paired")
        if len(auxiliary_noise_hashes) != 1 or None in auxiliary_noise_hashes:
            raise AuditError(f"clip {clip}: auxiliary noise is not paired")
    return indexed


def _numeric_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = sorted(set.intersection(*(set(row) for row in rows)))
    numeric: dict[str, list[float]] = {}
    for field in fields:
        values = [row[field] for row in rows]
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            numeric[field] = [float(value) for value in values]

    eligible: list[str] = []
    eligible_constant: list[str] = []
    for field in ELIGIBLE_CONFIDENCE_FIELDS:
        if field not in numeric:
            continue
        if len(set(numeric[field])) > 1:
            eligible.append(field)
        else:
            eligible_constant.append(field)

    target_fields = sorted(field for field in numeric if field in TARGET_DERIVED_FIELDS)
    audit_fields = sorted(field for field in numeric if field in AUDIT_ONLY_NUMERIC_FIELDS)
    unclassified = sorted(
        field
        for field in numeric
        if field not in TARGET_DERIVED_FIELDS
        and field not in AUDIT_ONLY_NUMERIC_FIELDS
        and field not in ELIGIBLE_CONFIDENCE_FIELDS
    )
    return {
        "eligible_varying_fields": eligible,
        "eligible_constant_fields": eligible_constant,
        "missing_prospectively_eligible_fields": sorted(
            set(ELIGIBLE_CONFIDENCE_FIELDS) - set(numeric)
        ),
        "target_derived_forbidden_fields_present": target_fields,
        "audit_only_numeric_fields_present": audit_fields,
        "unclassified_numeric_fields_fail_closed": unclassified,
        "effective_state_gate_unique": sorted(
            set(numeric.get("effective_state_gate", []))
        ),
        "effective_clock_gate_unique": sorted(
            set(numeric.get("effective_clock_gate", []))
        ),
        "generated_auxiliary_payload_preserved": False,
        "generated_auxiliary_identity_only": "auxiliary_final_sha256" in fields,
    }


def _vectors(
    indexed: Mapping[tuple[str, int], Mapping[str, Any]], source: str, metric: str
) -> np.ndarray:
    return np.asarray(
        [float(indexed[(source, clip)][metric]) for clip in range(EXPECTED_CLIPS)],
        dtype=np.float64,
    )


def _bootstrap_distribution(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise AuditError("paired bootstrap vectors must have equal 1-D shape")
    indices = rng.integers(0, reference.size, size=(replicates, reference.size))
    ref_means = reference[indices].mean(axis=1)
    cand_means = candidate[indices].mean(axis=1)
    if np.any(ref_means <= 0):
        raise AuditError("bootstrap reference mean must be positive")
    return 100.0 * (ref_means - cand_means) / ref_means


def _comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    distribution = _bootstrap_distribution(
        reference, candidate, replicates=replicates, rng=rng
    )
    point = 100.0 * (reference.mean() - candidate.mean()) / reference.mean()
    return {
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": float(point),
        "paired_bootstrap_95_ci_percent": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
    }


def _selector_comparisons(
    indexed: Mapping[tuple[str, int], Mapping[str, Any]],
    selector: np.ndarray,
    selected_source: str,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if selector.shape != (EXPECTED_CLIPS,) or selector.dtype != np.bool_:
        raise AuditError("selector must be a Boolean val64 vector")
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        off = _vectors(indexed, OFF_SOURCE, metric)
        selected = _vectors(indexed, selected_source, metric)
        endpoint = np.where(selector, selected, off)
        metrics[metric] = _comparison(
            off, endpoint, replicates=replicates, rng=rng
        )
    rate_distribution = selector[
        rng.integers(0, EXPECTED_CLIPS, size=(replicates, EXPECTED_CLIPS))
    ].mean(axis=1)
    return {
        "selected_source": selected_source,
        "selected_clips": int(selector.sum()),
        "selection_rate": float(selector.mean()),
        "selection_rate_bootstrap_95_ci": [
            float(np.quantile(rate_distribution, 0.025)),
            float(np.quantile(rate_distribution, 0.975)),
        ],
        "metrics": metrics,
    }


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, ...]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (x - mean) / scale
    y_mean = float(y.mean())
    gram = normalized.T @ normalized + ridge * np.eye(normalized.shape[1])
    beta = np.linalg.solve(gram, normalized.T @ (y - y_mean))
    return mean, scale, beta, np.asarray(y_mean)


def _predict_ridge(model: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    mean, scale, beta, y_mean = model
    return np.asarray(y_mean + ((x - mean) / scale) @ beta, dtype=np.float64)


def _nested_loo_predictions(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[float]]:
    lambdas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    n = x.shape[0]
    predictions = np.empty(n, dtype=np.float64)
    selected_lambdas: list[float] = []
    for outer in range(n):
        outer_train = np.arange(n) != outer
        x_train, y_train = x[outer_train], y[outer_train]
        losses: list[float] = []
        for ridge in lambdas:
            inner_predictions = np.empty(y_train.size, dtype=np.float64)
            for inner in range(y_train.size):
                inner_train = np.arange(y_train.size) != inner
                model = _fit_ridge(x_train[inner_train], y_train[inner_train], ridge)
                inner_predictions[inner] = _predict_ridge(
                    model, x_train[inner : inner + 1]
                )[0]
            losses.append(float(np.mean((inner_predictions - y_train) ** 2)))
        chosen = lambdas[int(np.argmin(losses))]
        selected_lambdas.append(chosen)
        model = _fit_ridge(x_train, y_train, chosen)
        predictions[outer] = _predict_ridge(model, x[outer : outer + 1])[0]
    return predictions, selected_lambdas


def _confidence_matrix(
    indexed: Mapping[tuple[str, int], Mapping[str, Any]],
    source: str,
    fields: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(indexed[(source, clip)][field]) for field in fields]
            for clip in range(EXPECTED_CLIPS)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise AuditError("confidence matrix contains non-finite values")
    return matrix


def analyze(
    mid_on_path: Path,
    mid_off_path: Path,
    protocol_path: Path,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if replicates < 100:
        raise AuditError("at least 100 bootstrap replicates are required")
    on_rows = load_jsonl(mid_on_path)
    off_rows = load_jsonl(mid_off_path)
    indexed = _index_primary_rows(on_rows, "MID-ON")
    package_indexed = _index_primary_rows(off_rows, "MID-OFF")
    ordered_on_rows = [
        indexed[(source, clip)]
        for source in (ALIGNED_SOURCE, OFF_SOURCE, CORRUPTED_SOURCE)
        for clip in range(EXPECTED_CLIPS)
    ]
    inventory = _numeric_inventory(ordered_on_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    always_on: dict[str, Any] = {}
    always_corrupted: dict[str, Any] = {}
    aligned_vs_corrupted: dict[str, Any] = {}
    package_reference: dict[str, Any] = {}
    improved_fraction: dict[str, Any] = {}
    for metric in METRICS:
        off = _vectors(indexed, OFF_SOURCE, metric)
        aligned = _vectors(indexed, ALIGNED_SOURCE, metric)
        corrupted = _vectors(indexed, CORRUPTED_SOURCE, metric)
        matched_trained_off = _vectors(package_indexed, ALIGNED_SOURCE, metric)
        always_on[metric] = _comparison(
            off, aligned, replicates=replicates, rng=rng
        )
        always_corrupted[metric] = _comparison(
            off, corrupted, replicates=replicates, rng=rng
        )
        aligned_vs_corrupted[metric] = _comparison(
            corrupted, aligned, replicates=replicates, rng=rng
        )
        package_reference[metric] = _comparison(
            matched_trained_off, aligned, replicates=replicates, rng=rng
        )
        wins = np.asarray(aligned < off, dtype=np.float64)
        win_distribution = wins[
            rng.integers(0, EXPECTED_CLIPS, size=(replicates, EXPECTED_CLIPS))
        ].mean(axis=1)
        improved_fraction[metric] = {
            "fraction": float(wins.mean()),
            "bootstrap_95_ci": [
                float(np.quantile(win_distribution, 0.025)),
                float(np.quantile(win_distribution, 0.975)),
            ],
        }

    off_temporal = _vectors(indexed, OFF_SOURCE, "temporal_mse")
    aligned_temporal = _vectors(indexed, ALIGNED_SOURCE, "temporal_mse")
    perfect_selector = np.asarray(aligned_temporal < off_temporal, dtype=np.bool_)
    pareto_selector = np.ones(EXPECTED_CLIPS, dtype=np.bool_)
    for metric in METRICS:
        pareto_selector &= _vectors(indexed, ALIGNED_SOURCE, metric) <= _vectors(
            indexed, OFF_SOURCE, metric
        )
    perfect_oracle = _selector_comparisons(
        indexed,
        perfect_selector,
        ALIGNED_SOURCE,
        replicates=replicates,
        rng=rng,
    )
    pareto_oracle = _selector_comparisons(
        indexed,
        pareto_selector,
        ALIGNED_SOURCE,
        replicates=replicates,
        rng=rng,
    )

    eligible_fields = list(inventory["eligible_varying_fields"])
    unclassified = list(inventory["unclassified_numeric_fields_fail_closed"])
    fitted_gate: dict[str, Any]
    if eligible_fields and not unclassified:
        aligned_x = _confidence_matrix(indexed, ALIGNED_SOURCE, eligible_fields)
        corrupted_x = _confidence_matrix(indexed, CORRUPTED_SOURCE, eligible_fields)
        target = (off_temporal - aligned_temporal) / off_temporal
        aligned_predictions, lambdas = _nested_loo_predictions(aligned_x, target)
        # Each outer model must also score the same held-out clip's corrupted
        # telemetry.  Refit the already selected outer lambda without reading
        # that clip's target.
        corrupted_predictions = np.empty(EXPECTED_CLIPS, dtype=np.float64)
        for outer, ridge in enumerate(lambdas):
            outer_train = np.arange(EXPECTED_CLIPS) != outer
            model = _fit_ridge(aligned_x[outer_train], target[outer_train], ridge)
            corrupted_predictions[outer] = _predict_ridge(
                model, corrupted_x[outer : outer + 1]
            )[0]
        aligned_selector = np.asarray(aligned_predictions > 0, dtype=np.bool_)
        corrupted_selector = np.asarray(corrupted_predictions > 0, dtype=np.bool_)
        aligned_result = _selector_comparisons(
            indexed,
            aligned_selector,
            ALIGNED_SOURCE,
            replicates=replicates,
            rng=rng,
        )
        corrupted_result = _selector_comparisons(
            indexed,
            corrupted_selector,
            CORRUPTED_SOURCE,
            replicates=replicates,
            rng=rng,
        )
        oracle_gain = perfect_oracle["metrics"]["temporal_mse"][
            "relative_improvement_percent"
        ]
        gated_gain = aligned_result["metrics"]["temporal_mse"][
            "relative_improvement_percent"
        ]
        fitted_gate = {
            "status": "FITTED_EXPLORATORY",
            "fields": eligible_fields,
            "outer_selected_lambdas": lambdas,
            "aligned": aligned_result,
            "corrupted": corrupted_result,
            "oracle_temporal_gain_retention": (
                float(gated_gain / oracle_gain) if oracle_gain > 0 else None
            ),
        }
    else:
        reasons: list[str] = []
        if not eligible_fields:
            reasons.append(
                "no varying per-clip inference-observable semantic confidence field is preserved"
            )
        if unclassified:
            reasons.append(
                "unclassified numeric fields exist and fail closed: " + ", ".join(unclassified)
            )
        fitted_gate = {
            "status": "NOT_FIT_TELEMETRY_BLOCKER",
            "reasons": reasons,
            "honest_policy": "always_off",
            "honest_policy_exactly_reuses_same_checkpoint_off_endpoint": True,
            "honest_policy_oracle_temporal_gain_retention": 0.0,
            "corrupted_auxiliary_fallback": "exact_off_for_all_clips",
        }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_status": "EXPLORATORY_REUSED_VAL64_NO_PROTECTED_TEST",
        "primary_nfe": PRIMARY_NFE,
        "clip_count": EXPECTED_CLIPS,
        "bootstrap": {
            "replicates": replicates,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired percentile 95%",
        },
        "inputs": {
            "mid_on_rows_sha256": sha256_file(mid_on_path),
            "mid_off_rows_sha256": sha256_file(mid_off_path),
            "protocol_sha256": sha256_file(protocol_path),
        },
        "safety": {
            "protected_test_accessed": False,
            "oracle_sampling_cell_used": False,
            "gpu_jobs_launched_or_changed": False,
            "all_primary_rows_deployable": True,
            "future_rgb_or_clean_auxiliary_passed_to_sampler": False,
            "online_teacher_calls": 0,
        },
        "confidence_telemetry_inventory": inventory,
        "always_on_aligned_vs_same_checkpoint_off": always_on,
        "always_on_corrupted_vs_same_checkpoint_off": always_corrupted,
        "aligned_vs_corrupted_same_checkpoint": aligned_vs_corrupted,
        "aligned_vs_separately_trained_mid_off": package_reference,
        "fraction_of_clips_aligned_beats_off": improved_fraction,
        "perfect_temporal_oracle_unattainable": perfect_oracle,
        "pareto_oracle_unattainable": pareto_oracle,
        "fitted_confidence_gate": fitted_gate,
        "interpretation": {
            "oracle_is_target_leaking_upper_bound": True,
            "confidence_gating_demonstrated": fitted_gate["status"]
            == "FITTED_EXPLORATORY",
            "exact_artifact_blocker": (
                "generated auxiliary content is preserved only by SHA-256; its numeric "
                "accuracy telemetry uses clean future targets; model gates are global "
                "constants; remaining varying numeric fields are quality or audit metadata"
            ),
        },
    }
    result["analysis_identity_sha256"] = _identity(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid-on-rows", type=Path, required=True)
    parser.add_argument("--mid-off-rows", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.mid_on_rows,
        args.mid_off_rows,
        args.protocol,
        replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
