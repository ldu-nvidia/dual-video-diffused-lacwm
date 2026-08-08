#!/usr/bin/env python3
"""Apply the preregistered NFE-1 CAMP paired-bootstrap gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_motion_plan_audit as structural


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_808
FAMILY_CELLS = 12
ONE_SIDED_ALPHA = 0.05 / FAMILY_CELLS
METRICS = (
    "decoded_temporal_difference_mse_unit_range",
    "video_future_nmse",
    "decoded_mse_unit_range",
)
REFERENCES = (
    ("independent_plan_off", "PLAN-OFF", "aligned"),
    ("same_checkpoint_off", "PLAN-ON", "off"),
    ("same_checkpoint_shuffled", "PLAN-ON", "shuffled"),
    ("same_checkpoint_action_shuffled", "PLAN-ON", "action_shuffled"),
)
QUALITY_REFERENCE_NAMES = frozenset(name for name, _, _ in REFERENCES[:3])
ACTION_REFERENCE_NAME = REFERENCES[3][0]


class CAMPAnalysisError(RuntimeError):
    """The audited row evidence or fixed statistical gate differs."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CAMPAnalysisError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CAMPAnalysisError(f"{label} must contain one object")
    return value


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str, int], Mapping[str, Any]]:
    result = {}
    for row in rows:
        endpoint = row["endpoint"]
        key = (
            str(row["arm"]),
            int(row["clip_index"]),
            str(endpoint["condition_source"]),
            int(endpoint["nfe"]),
        )
        if key in result:
            raise CAMPAnalysisError("duplicate audited endpoint row")
        result[key] = row
    return result


def paired_relative_improvement(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    bootstrap_indexes: np.ndarray,
) -> dict[str, float]:
    """Return lower-is-better relative improvement and Bonferroni lower bound."""

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (
        candidate.ndim != 1
        or reference.shape != candidate.shape
        or candidate.size < 2
        or bootstrap_indexes.ndim != 2
        or bootstrap_indexes.shape[1] != candidate.size
        or not np.isfinite(candidate).all()
        or not np.isfinite(reference).all()
        or np.any(candidate < 0)
        or np.any(reference <= 0)
    ):
        raise CAMPAnalysisError("paired metric vectors are invalid")
    point = (reference.mean() - candidate.mean()) / reference.mean()
    candidate_means = candidate[bootstrap_indexes].mean(axis=1)
    reference_means = reference[bootstrap_indexes].mean(axis=1)
    effects = (reference_means - candidate_means) / np.maximum(
        reference_means, np.finfo(np.float64).tiny
    )
    lower = np.quantile(effects, ONE_SIDED_ALPHA, method="linear")
    return {
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(reference.mean()),
        "relative_improvement": float(point),
        "simultaneous_one_sided_lower_bound": float(lower),
    }


def _cell_pass(metric: str, effect: Mapping[str, float]) -> bool:
    point = float(effect["relative_improvement"])
    lower = float(effect["simultaneous_one_sided_lower_bound"])
    if metric == "decoded_temporal_difference_mse_unit_range":
        return point >= 0.03 and lower >= 0.01
    return point >= 0.0 and lower > -0.01


def latency_summary(
    seconds: Sequence[float], *, generated_frames: int = 8
) -> dict[str, float]:
    """Separate chunk decision rate from generated-frame throughput."""

    values = np.asarray(seconds, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 1
        or isinstance(generated_frames, bool)
        or not isinstance(generated_frames, int)
        or generated_frames < 1
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise CAMPAnalysisError("latency values are invalid")
    mean = float(values.mean())
    p95 = float(np.quantile(values, 0.95, method="linear"))
    return {
        "mean_seconds": mean,
        "p95_seconds": p95,
        "rollout_hz_from_mean": 1.0 / mean,
        "rollout_hz_from_p95": 1.0 / p95,
        "generated_frame_fps_from_mean": float(generated_frames) / mean,
        "generated_frame_fps_from_p95": float(generated_frames) / p95,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    structural_audit: Mapping[str, Any],
    paired_training_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not structural.identity_valid(structural_audit)
        or structural_audit.get("kind") != structural.AUDIT_KIND
        or structural_audit.get("status") != "passed"
        or structural_audit.get("expected_validation_clips") != 64
        or structural_audit.get("protected_test_accessed") is not False
        or structural_audit.get("primary_gate_endpoints")
        != [
            "aligned_nfe_1",
            "off_nfe_1",
            "shuffled_nfe_1",
            "action_shuffled_nfe_1",
        ]
    ):
        raise CAMPAnalysisError("structural CAMP audit differs")
    if (
        not structural.identity_valid(paired_training_audit)
        or paired_training_audit.get("kind") != "camp_paired_training_audit"
        or paired_training_audit.get("status") != "passed"
        or paired_training_audit.get("updates") != 200
        or paired_training_audit.get("protected_test_accessed") is not False
    ):
        raise CAMPAnalysisError("paired CAMP training audit differs")
    sealed_identities = {
        row.get("sealed_registration_identity_sha256") for row in rows
    }
    if sealed_identities != {
        paired_training_audit.get("sealed_registration_identity_sha256")
    }:
        raise CAMPAnalysisError("evaluation/training sealed registration differs")
    index = _index(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indexes = rng.integers(
        0, 64, size=(BOOTSTRAP_REPLICATES, 64), endpoint=False
    )
    candidate_rows = [index[("PLAN-ON", clip, "aligned", 1)] for clip in range(64)]
    comparisons: dict[str, Any] = {}
    for name, arm, source in REFERENCES:
        reference_rows = [index[(arm, clip, source, 1)] for clip in range(64)]
        effects = {}
        for metric in METRICS:
            candidate = np.asarray(
                [float(row["metrics"][metric]) for row in candidate_rows]
            )
            reference = np.asarray(
                [float(row["metrics"][metric]) for row in reference_rows]
            )
            effect = paired_relative_improvement(
                candidate, reference, bootstrap_indexes=bootstrap_indexes
            )
            passed = _cell_pass(metric, effect)
            effects[metric] = {**effect, "pass": passed}
        comparisons[name] = {
            "reference": {"arm": arm, "condition_source": source, "nfe": 1},
            "metrics": effects,
        }
    quality_nine_cells_pass = all(
        effect["pass"]
        for name in QUALITY_REFERENCE_NAMES
        for effect in comparisons[name]["metrics"].values()
    )
    action_attribution_three_cells_pass = all(
        effect["pass"]
        for effect in comparisons[ACTION_REFERENCE_NAME]["metrics"].values()
    )
    all_twelve_cells_pass = (
        quality_nine_cells_pass and action_attribution_three_cells_pass
    )
    descriptive: dict[str, Any] = {}
    latency_keys = ("history_encode", "planner", "wan", "decode", "end_to_end")
    for arm in structural.ARMS:
        for endpoint in structural.ENDPOINTS:
            endpoint_rows = [
                index[(arm, clip, endpoint.condition_source, endpoint.nfe)]
                for clip in range(64)
            ]
            key = f"{arm}/{endpoint.code}"
            latency_vectors = {
                field: [row["latency_seconds"][field] for row in endpoint_rows]
                for field in latency_keys
            }
            descriptive[key] = {
                "primary_gate": endpoint.primary_gate,
                "metric_means": {
                    metric: float(
                        np.mean([row["metrics"][metric] for row in endpoint_rows])
                    )
                    for metric in METRICS
                },
                "latency_mean_seconds": {
                    field: float(np.mean(latency_vectors[field]))
                    for field in latency_keys
                },
                "latency_p95_seconds": {
                    field: float(
                        np.quantile(latency_vectors[field], 0.95, method="linear")
                    )
                    for field in latency_keys
                },
                "throughput": latency_summary(latency_vectors["end_to_end"]),
                "generated_plan_nmse_mean": float(
                    np.mean(
                        [row["metrics"]["generated_plan_nmse"] for row in endpoint_rows]
                    )
                ),
                "generated_plan_cosine_mean": float(
                    np.mean(
                        [row["metrics"]["generated_plan_cosine"] for row in endpoint_rows]
                    )
                ),
            }
    candidate_throughput = descriptive["PLAN-ON/aligned_nfe_1"]["throughput"]
    candidate_rollout_hz_p95 = candidate_throughput["rollout_hz_from_p95"]
    payload = {
        "schema_version": 1,
        "kind": "causal_motion_plan_preregistered_analysis",
        "status": "passed" if all_twelve_cells_pass else "failed",
        "structural_audit_identity_sha256": structural_audit["identity_sha256"],
        "paired_training_audit_identity_sha256": paired_training_audit[
            "identity_sha256"
        ],
        "protected_test_accessed": False,
        "claim_scope": "one_seed_validation_screen_only",
        "candidate": {"arm": "PLAN-ON", "condition_source": "aligned", "nfe": 1},
        "bootstrap": {
            "unit": "paired_validation_clip",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "family_cells": FAMILY_CELLS,
            "one_sided_alpha_per_cell": ONE_SIDED_ALPHA,
            "common_resamples_across_cells": True,
        },
        "fixed_gate": {
            "comparisons": comparisons,
            "quality_nine_cells_pass": quality_nine_cells_pass,
            "action_attribution_three_cells_pass": action_attribution_three_cells_pass,
            "all_twelve_cells_pass": all_twelve_cells_pass,
            "nfe_2_or_4_can_rescue": False,
            "action_shuffle_required_for_action_conditioned_claim": True,
        },
        "descriptive": descriptive,
        "real_time_dagger": {
            "rate_definition": "action_conditioned_rollouts_per_second_from_p95_end_to_end_latency",
            "candidate_end_to_end_mean_seconds": candidate_throughput[
                "mean_seconds"
            ],
            "candidate_end_to_end_p95_seconds": candidate_throughput["p95_seconds"],
            "candidate_rollout_hz_from_mean": candidate_throughput[
                "rollout_hz_from_mean"
            ],
            "candidate_rollout_hz_from_p95": candidate_rollout_hz_p95,
            "candidate_generated_frame_fps_from_mean": candidate_throughput[
                "generated_frame_fps_from_mean"
            ],
            "candidate_generated_frame_fps_from_p95": candidate_throughput[
                "generated_frame_fps_from_p95"
            ],
            "five_rollout_hz_p95_threshold_met": candidate_rollout_hz_p95 >= 5.0,
            "claim_allowed": bool(
                all_twelve_cells_pass and candidate_rollout_hz_p95 >= 5.0
            ),
        },
    }
    if not all(
        all(
            math.isfinite(float(metric)) and float(metric) > 0
            for metric in value["throughput"].values()
        )
        for value in descriptive.values()
    ):
        raise CAMPAnalysisError("descriptive throughput is invalid")
    return structural.identity_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit_payload = _read_json(args.audit.resolve(strict=True), "CAMP audit")
    pairing_payload = _read_json(
        args.pairing.resolve(strict=True), "paired CAMP training audit"
    )
    rows, source_files = structural.read_jsonl(args.rows)
    observed_audit = structural.audit_rows(rows, source_files=source_files)
    if observed_audit.get("identity_sha256") != audit_payload.get("identity_sha256"):
        raise CAMPAnalysisError("row evidence differs from structural audit")
    result = analyze_rows(
        rows,
        structural_audit=audit_payload,
        paired_training_audit=pairing_payload,
    )
    structural.exclusive_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "identity_sha256": result["identity_sha256"],
                "status": result["status"],
                "all_twelve_cells_pass": result["fixed_gate"][
                    "all_twelve_cells_pass"
                ],
                "candidate_rollout_hz_from_p95": result["real_time_dagger"][
                    "candidate_rollout_hz_from_p95"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
