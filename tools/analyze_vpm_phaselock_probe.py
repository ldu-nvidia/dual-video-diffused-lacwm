#!/usr/bin/env python3
"""Analyze the preregistered VPM PhaseLock validation-only probe.

The analyzer never selects or opens a protected-test split.  It evaluates four
fixed aligned-prior candidates against (1) their sample-shuffled prior,
(2) ordinary Euler sampling at exactly the same total transformer calls, and
(3) the already established ordinary one-call VPM frontier point.
"""

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

try:
    from tools import vpm_phaselock_probe as probe
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import vpm_phaselock_probe as probe


SCHEMA_VERSION = 1
KIND_ANALYSIS = "vpm_phaselock_probe_analysis"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_807
# Four candidates x three preregistered contrasts.  A one-sided Bonferroni
# lower bound across all 12 contrasts preserves family-wise alpha=0.05.
CONTRAST_COUNT = 12
ONE_SIDED_CONFIDENCE = 1.0 - 0.05 / CONTRAST_COUNT
TEMPORAL_MINIMUM_IMPROVEMENT = 0.01
GUARDRAIL_MAXIMUM_REGRESSION = 0.01
PRIMARY_METRIC = "decoded_temporal_difference_mse_unit_range"
GUARDRAIL_METRICS = ("video_future_nmse", "decoded_mse_unit_range")
CLAIM_METRICS = (PRIMARY_METRIC, *GUARDRAIL_METRICS)
CANDIDATES = ("k1_f2", "k1_f3", "k2_f2", "k2_f4")


class PhaseLockAnalysisError(RuntimeError):
    """Raised when the validation evidence is incomplete or incomparable."""


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


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def paired_effect(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    label: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    one_sided_confidence: float = ONE_SIDED_CONFIDENCE,
) -> dict[str, Any]:
    """Return paired relative improvement; positive always favors candidate."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or left.size < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(right <= 0)
        or bootstrap_samples < 100
        or not 0.5 < one_sided_confidence < 1.0
    ):
        raise PhaseLockAnalysisError(f"invalid paired input for {label}")
    relative = (right.mean() - left.mean()) / right.mean()
    derived_seed = _derived_seed(label)
    rng = np.random.default_rng(derived_seed)
    indexes = rng.integers(
        0, left.size, size=(bootstrap_samples, left.size), endpoint=False
    )
    left_means = left[indexes].mean(axis=1)
    right_means = right[indexes].mean(axis=1)
    if np.any(right_means <= 0):
        raise PhaseLockAnalysisError(f"non-positive bootstrap reference: {label}")
    effects = (right_means - left_means) / right_means
    alpha = 1.0 - one_sided_confidence
    descriptive_low, descriptive_high = np.quantile(effects, [0.025, 0.975])
    return {
        "n_paired_clips": int(left.size),
        "candidate_mean": float(left.mean()),
        "reference_mean": float(right.mean()),
        "mean_favorable_delta": float((right - left).mean()),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(100.0 * relative),
        "one_sided_simultaneous_lower_bound": {
            "confidence": one_sided_confidence,
            "familywise_alpha": 0.05,
            "family_contrast_count": CONTRAST_COUNT,
            "low": float(np.quantile(effects, alpha)),
        },
        "descriptive_two_sided_95_ci": {
            "low": float(descriptive_low),
            "high": float(descriptive_high),
        },
        "favorable_clip_fraction": float(np.mean(right > left)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_label": label,
        "bootstrap_derived_seed": derived_seed,
        "bootstrap_unit": "paired immutable validation clip",
    }


def contrast_gate(effects: Mapping[str, Any]) -> dict[str, Any]:
    """Gate one candidate/control contrast with a temporal benefit/guardrails."""

    values: dict[str, tuple[float, float]] = {}
    for metric in CLAIM_METRICS:
        effect = effects.get(metric)
        lower = (
            effect.get("one_sided_simultaneous_lower_bound", {}).get("low")
            if isinstance(effect, Mapping)
            else None
        )
        point = effect.get("relative_improvement") if isinstance(effect, Mapping) else None
        if (
            isinstance(point, bool)
            or not isinstance(point, (int, float))
            or isinstance(lower, bool)
            or not isinstance(lower, (int, float))
            or not math.isfinite(float(point))
            or not math.isfinite(float(lower))
        ):
            raise PhaseLockAnalysisError(f"contrast lacks finite effect for {metric}")
        values[metric] = (float(point), float(lower))
    temporal_point, temporal_low = values[PRIMARY_METRIC]
    checks = {
        "temporal_point_at_least_one_percent": (
            temporal_point >= TEMPORAL_MINIMUM_IMPROVEMENT
        ),
        "temporal_simultaneous_lb_at_least_one_percent": (
            temporal_low >= TEMPORAL_MINIMUM_IMPROVEMENT
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
            "decoded temporal MSE point and one-sided simultaneous LB >= +1%; "
            "video-latent NMSE and decoded MSE point/LB each > -1%"
        ),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseLockAnalysisError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PhaseLockAnalysisError(f"{label} must contain one object")
    return payload


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PhaseLockAnalysisError(f"refusing to overwrite output: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_inventory(
    inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    inventory_path = inventory_path.expanduser().resolve()
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise PhaseLockAnalysisError("inventory must be a non-symlink file")
    inventory = _read_json(inventory_path, "probe inventory")
    if (
        not probe.identity_valid(inventory)
        or inventory.get("kind") != probe.KIND_INVENTORY
        or inventory.get("evaluation_split") != "validation"
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("endpoint_grid")
        != [probe.asdict(endpoint) for endpoint in probe.ENDPOINTS]
    ):
        raise PhaseLockAnalysisError("inventory identity/protocol differs")
    registration_record = inventory.get("registration")
    if not isinstance(registration_record, Mapping):
        raise PhaseLockAnalysisError("inventory lacks registration")
    registration_path = Path(str(registration_record.get("path", "")))
    registration = _read_json(registration_path, "probe registration")
    if (
        not probe.identity_valid(registration)
        or registration.get("kind") != probe.KIND_REGISTRATION
        or registration.get("identity_sha256")
        != registration_record.get("identity_sha256")
        or probe._sha256(registration_path) != registration_record.get("sha256")
        or registration.get("fixed_protocol", {}).get("split") != "validation"
        or registration.get("fixed_protocol", {}).get(
            "protected_test_access_allowed"
        )
        is not False
    ):
        raise PhaseLockAnalysisError("registered validation protocol differs")
    rows: list[dict[str, Any]] = []
    rank_records = inventory.get("rank_manifests")
    if not isinstance(rank_records, list) or len(rank_records) != probe.EXPECTED_WORLD_SIZE:
        raise PhaseLockAnalysisError("inventory rank-manifest count differs")
    for expected_rank, record in enumerate(rank_records):
        if not isinstance(record, Mapping):
            raise PhaseLockAnalysisError("rank-manifest record is invalid")
        manifest_path = Path(str(record.get("path", "")))
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or record.get("sha256") != probe._sha256(manifest_path)
            or record.get("bytes") != manifest_path.stat().st_size
        ):
            raise PhaseLockAnalysisError("rank manifest changed after inventory")
        manifest = _read_json(manifest_path, f"rank {expected_rank} manifest")
        if (
            not probe.identity_valid(manifest)
            or manifest.get("kind") != probe.KIND_RANK
            or manifest.get("rank") != expected_rank
            or manifest.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or manifest.get("evaluation_split") != "validation"
            or manifest.get("protected_test_accessed") is not False
        ):
            raise PhaseLockAnalysisError("rank manifest protocol differs")
        rows_record = manifest.get("rows")
        rows_path = Path(str(rows_record.get("path", "")))
        content = rows_path.read_bytes()
        if (
            len(content) != rows_record.get("bytes")
            or hashlib.sha256(content).hexdigest() != rows_record.get("sha256")
        ):
            raise PhaseLockAnalysisError("rank rows changed after manifest")
        for line in content.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise PhaseLockAnalysisError("quality row is not an object")
            rows.append(row)
    validation_manifest = Path(registration["validation"]["manifest"]["path"])
    descriptors = [
        json.loads(line)
        for line in validation_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probe._validate_global_rows(rows, registration, descriptors)
    return inventory, registration, rows


def _row_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    lookup = {}
    for row in rows:
        key = (int(row["clip_index"]), str(row["endpoint"]["code"]))
        if key in lookup:
            raise PhaseLockAnalysisError(f"duplicate row {key}")
        lookup[key] = row
    return lookup


def compare_endpoints(
    lookup: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    candidate_code: str,
    reference_code: str,
    label: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in CLAIM_METRICS:
        candidate = [
            float(lookup[(index, candidate_code)]["metrics"][metric])
            for index in range(probe.EXPECTED_VALIDATION_CLIPS)
        ]
        reference = [
            float(lookup[(index, reference_code)]["metrics"][metric])
            for index in range(probe.EXPECTED_VALIDATION_CLIPS)
        ]
        metrics[metric] = paired_effect(
            candidate,
            reference,
            label=f"{label}:{metric}",
        )
    result = {
        "candidate": candidate_code,
        "reference": reference_code,
        "metrics": metrics,
    }
    result["gate"] = contrast_gate(metrics)
    return result


def analyze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = _row_lookup(rows)
    candidate_results = []
    for candidate in CANDIDATES:
        aligned = f"phaselock_{candidate}_aligned"
        shuffled = f"phaselock_{candidate}_shuffled"
        endpoint = probe.ENDPOINT_BY_CODE[aligned]
        matched = f"ordinary_b{endpoint.total_transformer_calls}"
        comparisons = {
            "sample_alignment_attribution": compare_endpoints(
                lookup,
                candidate_code=aligned,
                reference_code=shuffled,
                label=f"{candidate}:aligned-vs-shuffled",
            ),
            "equal_total_transformer_calls": compare_endpoints(
                lookup,
                candidate_code=aligned,
                reference_code=matched,
                label=f"{candidate}:aligned-vs-{matched}",
            ),
            "existing_vpm_one_call_frontier": compare_endpoints(
                lookup,
                candidate_code=aligned,
                reference_code="ordinary_b1",
                label=f"{candidate}:aligned-vs-ordinary_b1",
            ),
        }
        gates = {key: value["gate"]["passed"] for key, value in comparisons.items()}
        passed = all(gates.values())
        temporal_vs_frontier = comparisons[
            "existing_vpm_one_call_frontier"
        ]["metrics"][PRIMARY_METRIC]["relative_improvement"]
        candidate_results.append(
            {
                "candidate": candidate,
                "aligned_endpoint": aligned,
                "shuffled_endpoint": shuffled,
                "matched_ordinary_endpoint": matched,
                "total_transformer_calls": endpoint.total_transformer_calls,
                "comparisons": comparisons,
                "all_three_gates_passed": passed,
                "gate_summary": gates,
                "tie_break_values": {
                    "total_transformer_calls": endpoint.total_transformer_calls,
                    "temporal_relative_improvement_vs_ordinary_b1": (
                        temporal_vs_frontier
                    ),
                    "fixed_candidate_order": CANDIDATES.index(candidate),
                },
            }
        )
    passed = [result for result in candidate_results if result["all_three_gates_passed"]]
    selected = None
    if passed:
        selected = min(
            passed,
            key=lambda value: (
                value["total_transformer_calls"],
                -value["tie_break_values"][
                    "temporal_relative_improvement_vs_ordinary_b1"
                ],
                value["tie_break_values"]["fixed_candidate_order"],
            ),
        )["candidate"]
    return {
        "fixed_candidate_results": candidate_results,
        "passing_candidates": [value["candidate"] for value in passed],
        "selected_validation_candidate": selected,
        "deployable_latent_delta_benefit_demonstrated_on_validation": bool(passed),
        "interpretation": (
            "At least one fixed candidate beats shuffled-prior, equal-call Euler, "
            "and the established one-call frontier under all simultaneous gates."
            if passed
            else "No fixed candidate passes attribution, equal-call benefit, and the "
            "established one-call frontier together; deployable PhaseLock benefit is "
            "not demonstrated by this validation-only probe."
        ),
        "protected_test_action": "never open or score protected test",
    }


def command_analyze(args: argparse.Namespace) -> int:
    inventory, registration, rows = _load_inventory(args.inventory)
    analysis = analyze(rows)
    output_dir = args.output_dir.expanduser()
    expected = Path(registration["output_root"]) / "analysis"
    if output_dir.absolute() != expected:
        raise PhaseLockAnalysisError(f"analysis output must be exactly {expected}")
    if output_dir.exists():
        raise PhaseLockAnalysisError(f"fresh analysis output exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    payload = probe.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_ANALYSIS,
            "created_at_utc": _now(),
            "registration": {
                "path": str(inventory["registration"]["path"]),
                "sha256": inventory["registration"]["sha256"],
                "identity_sha256": registration["identity_sha256"],
            },
            "inventory": {
                "path": str(args.inventory.resolve()),
                "sha256": probe._sha256(args.inventory.resolve()),
                "identity_sha256": inventory["identity_sha256"],
            },
            "evaluation_split": "validation",
            "protected_test_accessed": False,
            "statistical_protocol": {
                "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "one_sided_simultaneous_confidence": ONE_SIDED_CONFIDENCE,
                "familywise_alpha": 0.05,
                "family_contrast_count": CONTRAST_COUNT,
                "temporal_minimum_relative_improvement": (
                    TEMPORAL_MINIMUM_IMPROVEMENT
                ),
                "guardrail_maximum_relative_regression": (
                    GUARDRAIL_MAXIMUM_REGRESSION
                ),
                "primary_metric": PRIMARY_METRIC,
                "guardrail_metrics": list(GUARDRAIL_METRICS),
                "deterministic_tie_break": (
                    "lower total calls; larger temporal effect vs ordinary_b1; "
                    "fixed candidate order"
                ),
            },
            **analysis,
            "claim_scope": (
                "validation-only mechanistic evidence; no protected-test, FVD, "
                "latency, DAgger-rate, or generalization claim"
            ),
        }
    )
    _exclusive_json(output_dir / "analysis.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return command_analyze(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PhaseLockAnalysisError, probe.PhaseLockProbeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
