#!/usr/bin/env python3
"""Frozen paired analysis and decision gate for IPQ-TC1."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools import invertible_two_clock_pilot as pilot
from tools import invertible_two_clock_evaluate as evaluation


PRIMARY_METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
DECODED_METRICS = PRIMARY_METRICS[1:]
BAND_METRICS = ("p_future_nmse", "q_future_nmse")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261201


class IPQAnalysisError(RuntimeError):
    """The frozen paired endpoint family is incomplete or inconsistent."""


def _rows(inventory_path: Path, expected_arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inventory = pilot.read_json(inventory_path.resolve(strict=True), "endpoint inventory")
    if (
        not pilot.identity_valid(inventory)
        or inventory.get("kind") != evaluation.KIND_INVENTORY
        or inventory.get("arm", {}).get("code") != expected_arm
        or inventory.get("target_array_opened") is not False
        or inventory.get("teacher_feature_encoder_calls") != 0
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("global_endpoint_barrier_passed") is not True
        or inventory.get("future_rgb_opened_before_global_endpoint_barrier")
        is not False
        or inventory.get("production_abc_dataset_constructed") is not False
        or inventory.get("validation_content_rehash_after_global_endpoint_barrier")
        is not True
        or float(inventory.get("evaluation_reserved_b200_hours", 0.0)) <= 0.0
        or float(inventory.get("training_reserved_b200_hours_both_arms", 0.0))
        <= 0.0
        or int(inventory.get("artifact_bytes_under_registered_root", 0)) <= 0
    ):
        raise IPQAnalysisError(f"{expected_arm} inventory differs")
    rows = []
    for manifest in inventory.get("rank_manifests", []):
        if (
            not pilot.identity_valid(manifest)
            or manifest.get("kind") != evaluation.KIND_RANK
            or manifest.get("arm", {}).get("code") != expected_arm
            or manifest.get("target_array_opened") is not False
            or manifest.get("teacher_feature_encoder_calls") != 0
            or manifest.get("protected_test_accessed") is not False
            or manifest.get("future_rgb_opened_before_global_endpoint_barrier")
            is not False
            or manifest.get("production_abc_dataset_constructed") is not False
        ):
            raise IPQAnalysisError(f"{expected_arm} rank manifest differs")
        ledger_path = Path(manifest["access_ledger"]["path"])
        if pilot.sha256(ledger_path) != manifest["access_ledger"]["sha256"]:
            raise IPQAnalysisError("rank access ledger bytes changed")
        ledger = pilot.read_json(ledger_path, "rank access ledger")
        if (
            not pilot.identity_valid(ledger)
            or ledger.get("future_rgb_opened_before_global_endpoint_barrier")
            is not False
            or ledger.get("production_abc_dataset_constructed") is not False
            or ledger.get("all_memmap_index_operations_recorded") is not True
            or any(
                value.get("future_rgb_elements_read") != 0
                for value in ledger.get(
                    "pre_global_endpoint_barrier_operations", []
                )
                if value.get("purpose") == "sampler_observed_history_only"
            )
            or any(
                int(value.get("read_event", 0))
                <= int(value.get("global_barrier_event", 0))
                for value in ledger.get(
                    "post_global_endpoint_barrier_operations", []
                )
            )
        ):
            raise IPQAnalysisError("rank access ledger differs")
        path = Path(manifest["rows"]["path"])
        if pilot.sha256(path) != manifest["rows"]["sha256"]:
            raise IPQAnalysisError("rank row bytes changed")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    not pilot.identity_valid(row)
                    or row.get("kind") != evaluation.KIND_ROW
                    or row.get("arm", {}).get("code") != expected_arm
                    or row.get("protected_test_accessed") is not False
                    or row.get("teacher_feature_encoder_calls") != 0
                    or row.get("auxiliary_target_array_opened") is not False
                    or row.get("scoring_constructed_after_all_batch_endpoints")
                    is not True
                    or row.get("event_order", {}).get("endpoint_materialized", 0)
                    >= row.get("event_order", {}).get("first_target_construction", 0)
                    or row.get("event_order", {}).get("endpoint_hashes_closed", 0)
                    >= row.get("event_order", {}).get("first_target_construction", 0)
                    or row.get("future_rgb_opened_before_global_endpoint_barrier")
                    is not False
                    or row.get("latency", {}).get("cuda_synchronized") is not True
                    or int(row.get("latency", {}).get("batch_size", 0)) != 2
                    or int(row.get("actual_wan_calls", -1)) != int(row["nfe"])
                    or row.get("clean_future_rgb_passed_to_sampler") is not False
                    or row.get("clean_future_latent_passed_to_sampler") is not False
                ):
                    raise IPQAnalysisError("endpoint row identity/access differs")
                if any(
                    not math.isfinite(float(row["metrics"].get(metric, float("nan"))))
                    for metric in (*PRIMARY_METRICS, *BAND_METRICS)
                ):
                    raise IPQAnalysisError("endpoint metric is non-finite")
                for value in (
                    row["latency"].get("sampling_seconds_per_batch"),
                    row["latency"].get("decode_seconds_per_batch"),
                    row["latency"].get("end_to_end_seconds_per_batch"),
                    row.get("peak_memory", {}).get("allocated_bytes"),
                    row.get("peak_memory", {}).get("reserved_bytes"),
                ):
                    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                        raise IPQAnalysisError("latency/memory audit is invalid")
                rows.append(row)
    expected = 64 * len(pilot.NOISE_SEEDS) * len(pilot.NFE_GRID) * len(
        pilot.ENDPOINTS_BY_ARM[expected_arm]
    )
    if len(rows) != expected or inventory.get("row_count") != expected:
        raise IPQAnalysisError(
            f"{expected_arm} expected {expected} rows, observed {len(rows)}"
        )
    keys = [
        (
            row["clip_index"],
            row["noise_seed"],
            row["nfe"],
            row["endpoint"]["code"],
        )
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise IPQAnalysisError(f"{expected_arm} endpoint keys are duplicated")
    return inventory, rows


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int, int, str], Mapping[str, Any]]:
    return {
        (
            int(row["clip_index"]),
            int(row["noise_seed"]),
            int(row["nfe"]),
            str(row["endpoint"]["code"]),
        ): row
        for row in rows
    }


def _paired_episode_values(
    candidate: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    reference: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    *,
    nfe_candidate: int,
    endpoint_candidate: str,
    nfe_reference: int,
    endpoint_reference: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    candidate_values = []
    reference_values = []
    episodes = []
    for clip in range(64):
        left = []
        right = []
        for seed in pilot.NOISE_SEEDS:
            left.append(
                float(candidate[(clip, seed, nfe_candidate, endpoint_candidate)]["metrics"][metric])
            )
            right.append(
                float(reference[(clip, seed, nfe_reference, endpoint_reference)]["metrics"][metric])
            )
        candidate_values.append(float(np.mean(left)))
        reference_values.append(float(np.mean(right)))
        episodes.append(clip)
    return np.asarray(candidate_values), np.asarray(reference_values), episodes


def _effect(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(reference.mean())
    if not math.isfinite(denominator) or denominator <= 0:
        raise IPQAnalysisError("relative effect denominator is nonpositive")
    return 100.0 * (denominator - float(candidate.mean())) / denominator


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    rng: np.random.Generator,
    lower_alpha: float,
) -> dict[str, Any]:
    count = candidate.shape[0]
    if reference.shape != candidate.shape or count != 64:
        raise IPQAnalysisError("paired episode arrays differ")
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        choice = rng.integers(0, count, size=count)
        samples[index] = _effect(candidate[choice], reference[choice])
    return {
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(reference.mean()),
        "relative_improvement_percent": _effect(candidate, reference),
        "lower_bound_percent": float(np.quantile(samples, lower_alpha)),
        "ordinary_95_interval_percent": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "favorable_episode_fraction": float(np.mean(candidate < reference)),
    }


def _assert_paired_inputs(
    sync: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    independent: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
) -> None:
    for clip in range(64):
        for seed in pilot.NOISE_SEEDS:
            for nfe in pilot.NFE_GRID:
                base = sync[(clip, seed, nfe, "SYNC")]
                candidate = independent[(clip, seed, nfe, "P_LEADS_ALIGNED")]
                for field in (
                    "sampler_history_rgb",
                    "sampler_actions",
                    "initial_noise",
                    "clean_latent_scoring",
                    "raw_full_rgb_scoring",
                ):
                    if base["tensor_sha256"][field] != candidate["tensor_sha256"][field]:
                        raise IPQAnalysisError(f"paired {field} differs")
            # NFE-1 schedule order must be an exact same-checkpoint no-op.
            hashes = {
                independent[(clip, seed, 1, endpoint)]["tensor_sha256"]["final_latent"]
                for endpoint in ("P_LEADS_ALIGNED", "SYNCHRONOUS", "Q_LEADS_ALIGNED")
            }
            if len(hashes) != 1:
                raise IPQAnalysisError("NFE-1 schedule-order identity failed")


def _latency_p95(
    index: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    *,
    nfe: int,
    endpoint: str,
) -> float:
    by_batch: dict[str, float] = {}
    for row in index.values():
        if int(row["nfe"]) != nfe or row["endpoint"]["code"] != endpoint:
            continue
        key = str(row["batch_key"])
        value = float(row["latency"]["end_to_end_seconds_per_batch"])
        previous = by_batch.setdefault(key, value)
        if previous != value:
            raise IPQAnalysisError("duplicated batch latency differs")
    if len(by_batch) != 128:
        raise IPQAnalysisError(
            f"expected 128 synchronized batch timings, got {len(by_batch)}"
        )
    return float(np.quantile(np.asarray(list(by_batch.values())), 0.95))


def analyze(sync_inventory: Path, independent_inventory: Path) -> dict[str, Any]:
    sync_meta, sync_rows = _rows(sync_inventory, "IPQ-SYNC")
    independent_meta, independent_rows = _rows(
        independent_inventory, "IPQ-INDEP"
    )
    if (
        sync_meta["registration_identity_sha256"]
        != independent_meta["registration_identity_sha256"]
        or sync_meta["source_commit"] != independent_meta["source_commit"]
    ):
        raise IPQAnalysisError("arm registrations/source differ")
    training_b200_hours = float(
        sync_meta["training_reserved_b200_hours_both_arms"]
    )
    if training_b200_hours != float(
        independent_meta["training_reserved_b200_hours_both_arms"]
    ):
        raise IPQAnalysisError("training resource receipts differ by inventory")
    observed_b200_hours = (
        training_b200_hours
        + float(sync_meta["evaluation_reserved_b200_hours"])
        + float(independent_meta["evaluation_reserved_b200_hours"])
    )
    if not math.isfinite(observed_b200_hours) or observed_b200_hours > 96.0:
        raise IPQAnalysisError("IPQ resource envelope exceeded 96 B200-hours")
    sync = _index(sync_rows)
    independent = _index(independent_rows)
    _assert_paired_inputs(sync, independent)
    primary = {}
    primary_rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_alpha = 0.05 / 9.0
    for nfe in pilot.NFE_GRID:
        primary[str(nfe)] = {}
        for metric in PRIMARY_METRICS:
            candidate, reference, _ = _paired_episode_values(
                independent,
                sync,
                nfe_candidate=nfe,
                endpoint_candidate="P_LEADS_ALIGNED",
                nfe_reference=nfe,
                endpoint_reference="SYNC",
                metric=metric,
            )
            primary[str(nfe)][metric] = _comparison(
                candidate,
                reference,
                rng=primary_rng,
                lower_alpha=primary_alpha,
            )
        primary[str(nfe)]["latency_p95_seconds_per_batch"] = {
            "candidate": _latency_p95(
                independent, nfe=nfe, endpoint="P_LEADS_ALIGNED"
            ),
            "reference": _latency_p95(sync, nfe=nfe, endpoint="SYNC"),
        }
    attribution = {}
    controls = (
        "P_LEADS_STATE_OFF",
        "P_LEADS_STATE_SHUFFLED",
        "Q_LEADS_ALIGNED",
        "P_LEADS_CLOCK_TIED",
        "SYNCHRONOUS",
    )
    attribution_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for nfe in pilot.NFE_GRID:
        attribution[str(nfe)] = {}
        for control in controls:
            attribution[str(nfe)][control] = {}
            for metric in PRIMARY_METRICS:
                candidate, reference, _ = _paired_episode_values(
                    independent,
                    independent,
                    nfe_candidate=nfe,
                    endpoint_candidate="P_LEADS_ALIGNED",
                    nfe_reference=nfe,
                    endpoint_reference=control,
                    metric=metric,
                )
                attribution[str(nfe)][control][metric] = _comparison(
                    candidate,
                    reference,
                    rng=attribution_rng,
                    lower_alpha=0.025,
                )
    passing = []
    reasons = {}
    for nfe in pilot.NFE_GRID:
        key = str(nfe)
        p = primary[key]
        gates = {
            "decoded_primary": all(
                p[metric]["relative_improvement_percent"] >= 3.0
                and p[metric]["lower_bound_percent"] > 0.0
                for metric in DECODED_METRICS
            ),
            "latent_guardrail": (
                p[PRIMARY_METRICS[0]]["relative_improvement_percent"] >= 0.0
                and p[PRIMARY_METRICS[0]]["lower_bound_percent"] > -1.0
            ),
            "episode_favorability": all(
                p[metric]["favorable_episode_fraction"] >= 0.60
                for metric in DECODED_METRICS
            ),
            "state_attribution": all(
                attribution[key][control][metric]["relative_improvement_percent"] >= 1.0
                and attribution[key][control][metric]["lower_bound_percent"] > 0.0
                for control in ("P_LEADS_STATE_OFF", "P_LEADS_STATE_SHUFFLED")
                for metric in DECODED_METRICS
            ),
            "access_call_capacity_algebra_latency_identity_audits": True,
        }
        if nfe == 1:
            gates["order_attribution"] = True
            gates["clock_use"] = True
        else:
            gates["order_attribution"] = all(
                attribution[key]["Q_LEADS_ALIGNED"][metric][
                    "relative_improvement_percent"
                ]
                >= 1.0
                and attribution[key]["Q_LEADS_ALIGNED"][metric][
                    "lower_bound_percent"
                ]
                > 0.0
                for metric in DECODED_METRICS
            )
            gates["clock_use"] = any(
                abs(
                    attribution[key]["P_LEADS_CLOCK_TIED"][metric][
                        "relative_improvement_percent"
                    ]
                )
                >= 1.0
                for metric in PRIMARY_METRICS
            )
        reasons[key] = gates
        if all(gates.values()):
            passing.append(nfe)
    training_only = {}
    for nfe in pilot.NFE_GRID:
        key = str(nfe)
        training_only[key] = {}
        for metric in PRIMARY_METRICS:
            candidate, reference, _ = _paired_episode_values(
                independent,
                sync,
                nfe_candidate=nfe,
                endpoint_candidate="SYNCHRONOUS",
                nfe_reference=nfe,
                endpoint_reference="SYNC",
                metric=metric,
            )
            training_only[key][metric] = _comparison(
                candidate,
                reference,
                rng=np.random.default_rng(BOOTSTRAP_SEED + 50_000 + nfe),
                lower_alpha=0.025,
            )

    if passing:
        decision = "GO_IPQ_TWO_CLOCK"
    elif any(
        all(
            primary[str(nfe)][metric]["relative_improvement_percent"] >= 3.0
            for metric in DECODED_METRICS
        )
        and not reasons[str(nfe)]["state_attribution"]
        for nfe in pilot.NFE_GRID
    ):
        decision = "NO_GO_OPTIMIZATION_ONLY"
    elif any(
        all(
            training_only[str(nfe)][metric]["relative_improvement_percent"]
            >= 3.0
            and training_only[str(nfe)][metric]["lower_bound_percent"] > 0.0
            for metric in DECODED_METRICS
        )
        and not all(
            attribution[str(nfe)][control][metric][
                "relative_improvement_percent"
            ]
            >= 1.0
            for control in ("Q_LEADS_ALIGNED", "SYNCHRONOUS")
            for metric in DECODED_METRICS
        )
        for nfe in pilot.NFE_GRID
    ):
        decision = "NO_GO_TRAINING_ONLY_NO_ORDER"
    elif any(
        any(
            _effect(
                _paired_episode_values(
                    independent,
                    sync,
                    nfe_candidate=nfe,
                    endpoint_candidate="P_LEADS_ALIGNED",
                    nfe_reference=nfe,
                    endpoint_reference="SYNC",
                    metric=metric,
                )[0],
                _paired_episode_values(
                    independent,
                    sync,
                    nfe_candidate=nfe,
                    endpoint_candidate="P_LEADS_ALIGNED",
                    nfe_reference=nfe,
                    endpoint_reference="SYNC",
                    metric=metric,
                )[1],
            )
            >= 3.0
            for metric in (*BAND_METRICS, PRIMARY_METRICS[0])
        )
        and not all(
            primary[str(nfe)][metric]["relative_improvement_percent"] >= 3.0
            for metric in DECODED_METRICS
        )
        for nfe in pilot.NFE_GRID
    ):
        decision = "NO_GO_LATENT_ONLY"
    else:
        decision = "NO_GO_IPQ_TWO_CLOCK"
    acceleration = {}
    for candidate_nfe, reference_nfe in ((1, 2), (2, 4)):
        comparisons = {}
        for metric in PRIMARY_METRICS:
            candidate, reference, _ = _paired_episode_values(
                independent,
                sync,
                nfe_candidate=candidate_nfe,
                endpoint_candidate="P_LEADS_ALIGNED",
                nfe_reference=reference_nfe,
                endpoint_reference="SYNC",
                metric=metric,
            )
            comparisons[metric] = _comparison(
                candidate,
                reference,
                rng=np.random.default_rng(
                    BOOTSTRAP_SEED + candidate_nfe * 100 + reference_nfe
                ),
                lower_alpha=0.025,
            )
        acceleration[f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"] = {
            "metrics": comparisons,
            "quality_noninferior": all(
                value["relative_improvement_percent"] >= -1.0
                and value["lower_bound_percent"] > -2.0
                for value in comparisons.values()
            ),
            "candidate_p95_seconds_per_batch": _latency_p95(
                independent,
                nfe=candidate_nfe,
                endpoint="P_LEADS_ALIGNED",
            ),
            "control_p95_seconds_per_batch": _latency_p95(
                sync,
                nfe=reference_nfe,
                endpoint="SYNC",
            ),
        }
        acceleration[f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"][
            "latency_lower"
        ] = (
            acceleration[f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"][
                "candidate_p95_seconds_per_batch"
            ]
            < acceleration[f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"][
                "control_p95_seconds_per_batch"
            ]
        )
        acceleration[f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"][
            "decision"
        ] = (
            "FEWER_CALLS_NONINFERIOR"
            if acceleration[
                f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"
            ]["quality_noninferior"]
            and acceleration[
                f"candidate_{candidate_nfe}_vs_control_{reference_nfe}"
            ]["latency_lower"]
            else "NO_ACCELERATION_CLAIM"
        )
    return pilot.identity_payload(
        {
            "kind": "ipq_tc1_analysis",
            "registration_identity_sha256": sync_meta[
                "registration_identity_sha256"
            ],
            "source_commit": sync_meta["source_commit"],
            "decision": decision,
            "passing_nfe": passing,
            "primary": primary,
            "attribution": attribution,
            "training_only_synchronous_evaluator": training_only,
            "gates": reasons,
            "acceleration": acceleration,
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "episode_clustered_after_four_noise_mean": True,
                "primary_one_sided_bonferroni_alpha": primary_alpha,
            },
            "audit": {
                "paired_inputs": True,
                "nfe1_order_bit_exact": True,
                "row_counts": {
                    "IPQ-SYNC": len(sync_rows),
                    "IPQ-INDEP": len(independent_rows),
                },
                "teacher_feature_encoder_calls": 0,
                "auxiliary_target_array_opened": False,
                "protected_test_accessed": False,
            },
            "resources": {
                "training_reserved_b200_hours_both_arms": training_b200_hours,
                "evaluation_reserved_b200_hours_both_arms": (
                    float(sync_meta["evaluation_reserved_b200_hours"])
                    + float(independent_meta["evaluation_reserved_b200_hours"])
                ),
                "observed_reserved_b200_hours_total": observed_b200_hours,
                "maximum_reserved_b200_hours": 96,
                "artifact_bytes_under_registered_root_at_latest_inventory": max(
                    int(sync_meta["artifact_bytes_under_registered_root"]),
                    int(independent_meta["artifact_bytes_under_registered_root"]),
                ),
                "peak_memory_allocated_bytes": max(
                    int(sync_meta["peak_memory_allocated_bytes"]),
                    int(independent_meta["peak_memory_allocated_bytes"]),
                ),
                "peak_memory_reserved_bytes": max(
                    int(sync_meta["peak_memory_reserved_bytes"]),
                    int(independent_meta["peak_memory_reserved_bytes"]),
                ),
            },
        }
    )


def command_analyze(args: argparse.Namespace) -> int:
    result = analyze(args.sync_inventory, args.independent_inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.exclusive_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-inventory", type=Path, required=True)
    parser.add_argument("--independent-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
