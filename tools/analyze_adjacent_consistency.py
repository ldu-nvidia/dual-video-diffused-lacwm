#!/usr/bin/env python3
"""Frozen paired-bootstrap gate analysis for the ACD-P0 endpoint factorial."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import low_nfe_direct_baseline as pilot  # noqa: E402


METRICS = (
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
    "video_future_nmse",
    "lpips_alex_frame",
)
PRIMARY_CANDIDATE = "CONS_EMA_CONSISTENCY"
PRIMARY_CONTROL = "RF_EMA_NATIVE_RF"
ONLINE_CANDIDATE = "CONS_ONLINE_CONSISTENCY"
ONLINE_CONTROL = "RF_ONLINE_NATIVE_RF"
PARENT = "PARENT_NATIVE_RF"
SHUFFLED_CANDIDATE = "CONS_EMA_CONSISTENCY_ACTION_SHUFFLED"
SHUFFLED_CONTROL = "RF_EMA_NATIVE_RF_ACTION_SHUFFLED"
SHUFFLED_PARENT = "PARENT_NATIVE_RF_ACTION_SHUFFLED"
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20_260_809
BONFERRONI_FAMILY = len(pilot.NFE_GRID) * (len(METRICS) * 2 + 2)


class ACDAnalysisError(RuntimeError):
    """Endpoint rows are incomplete, nonpaired, or outside the frozen family."""


def _load_rows(
    evaluation: Path,
    inventory: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    manifests = inventory.get("rank_manifests", [])
    if (
        not isinstance(manifests, list)
        or len(manifests) != 8
        or {manifest.get("rank") for manifest in manifests} != set(range(8))
    ):
        raise ACDAnalysisError("rank manifest family differs")
    descriptors = registration["validation_descriptors"]
    for manifest in manifests:
        rank = int(manifest["rank"])
        path = Path(manifest["rows"]["path"])
        if (
            path.resolve(strict=True) != (evaluation / f"rank_{rank:03d}.jsonl").resolve(strict=True)
            or pilot.file_record(path) != manifest["rows"]
        ):
            raise ACDAnalysisError("rank row file hash differs")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if (
                    not pilot.identity_valid(value)
                    or value.get("kind") != "acd_p0_endpoint_row"
                    or value.get("registration_identity_sha256")
                    != inventory["registration_identity_sha256"]
                    or value.get("protected_test_accessed") is not False
                    or value.get("future_rgb_opened_before_global_endpoint_barrier")
                    is not False
                    or value.get("teacher_calls") != 0
                    or value.get("feature_calls") != 0
                    or value.get("source_commit")
                    != registration["tool_repository"]["git_commit"]
                    or int(value.get("rank", -1)) != rank
                ):
                    raise ACDAnalysisError("endpoint row identity/access differs")
                rows.append(value)
    expected = 64 * len(pilot.NOISE_SEEDS) * len(pilot.ENDPOINTS) * len(pilot.NFE_GRID)
    if len(rows) != expected or inventory.get("row_count") != expected:
        raise ACDAnalysisError(f"expected {expected} endpoint rows, got {len(rows)}")
    keys = {
        (
            row["endpoint"]["code"],
            int(row["nfe"]),
            int(row["noise_seed"]),
            int(row["clip_index"]),
        )
        for row in rows
    }
    if len(keys) != expected:
        raise ACDAnalysisError("endpoint key family contains duplicates")
    expected_endpoints = {endpoint.code for endpoint in pilot.ENDPOINTS}
    if {row["endpoint"]["code"] for row in rows} != expected_endpoints:
        raise ACDAnalysisError("endpoint codes differ from registration")
    if {int(row["nfe"]) for row in rows} != set(pilot.NFE_GRID):
        raise ACDAnalysisError("NFE grid differs")
    if {int(row["noise_seed"]) for row in rows} != set(pilot.NOISE_SEEDS):
        raise ACDAnalysisError("noise seed family differs")
    for row in rows:
        code = row.get("endpoint", {}).get("code")
        endpoint = pilot.ENDPOINT_BY_CODE.get(str(code))
        if row.get("actual_wan_calls") != row.get("nfe") or row.get(
            "declared_wan_calls"
        ) != row.get("nfe") or endpoint is None or row.get("endpoint") != asdict(endpoint):
            raise ACDAnalysisError("row NFE/call count differs")
        clip_index = int(row["clip_index"])
        descriptor = descriptors[clip_index]
        if (
            row.get("clip_id") != descriptor["clip_id"]
            or row.get("episode_dir") != descriptor["episode_dir"]
            or int(row.get("rank", -1)) != clip_index % 8
            or row.get("action_source") != endpoint.action_source
            or set(row.get("tensor_sha256", {}))
            != {
                "history_rgb",
                "actions",
                "noise_stream",
                "final_latent",
                "decoded_future",
                "clean_latent_scoring",
                "raw_full_rgb_scoring",
            }
            or not all(
                isinstance(value, str) and len(value) == 64
                for value in row["tensor_sha256"].values()
            )
        ):
            raise ACDAnalysisError("row tensor/descriptor identity differs")
        event = row.get("event_order", {})
        latency = row.get("latency", {})
        timings = latency.get("complete_sampler_seconds_per_batch", [])
        if (
            not int(event.get("endpoint_materialized", -1))
            < int(event.get("endpoint_hashes_closed", -1))
            < int(event.get("first_target_construction", -1))
            or latency.get("cuda_synchronized") is not True
            or int(latency.get("batch_size", -1)) != 2
            or not isinstance(timings, list)
            or len(timings) != 4
            or not all(math.isfinite(float(value)) and float(value) > 0 for value in timings)
        ):
            raise ACDAnalysisError("row barrier/timing identity differs")
        if set(row.get("metrics", {})) != {
            *METRICS,
            "video_future_temporal_delta_nmse",
            "decoded_psnr_db",
        }:
            raise ACDAnalysisError("metric family differs")
        if not all(math.isfinite(float(value)) for value in row["metrics"].values()):
            raise ACDAnalysisError("metric is non-finite")
    _validate_paired_rows(rows, registration)
    return rows


def _validate_paired_rows(
    rows: Sequence[Mapping[str, Any]], registration: Mapping[str, Any]
) -> None:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["nfe"]), int(row["noise_seed"]), int(row["clip_index"]))].append(row)
    aligned_codes = {
        endpoint.code
        for endpoint in pilot.ENDPOINTS
        if endpoint.action_source == "aligned"
    }
    shuffled_codes = {
        endpoint.code
        for endpoint in pilot.ENDPOINTS
        if endpoint.action_source == "episode_shuffled"
    }
    for key, family in grouped.items():
        if len(family) != len(pilot.ENDPOINTS):
            raise ACDAnalysisError(f"endpoint pairing is incomplete: {key}")
        for field in ("history_rgb", "noise_stream", "clean_latent_scoring", "raw_full_rgb_scoring"):
            if len({row["tensor_sha256"][field] for row in family}) != 1:
                raise ACDAnalysisError(f"paired {field} differs: {key}")
        aligned = [row for row in family if row["endpoint"]["code"] in aligned_codes]
        shuffled = [row for row in family if row["endpoint"]["code"] in shuffled_codes]
        if len({row["tensor_sha256"]["actions"] for row in aligned}) != 1:
            raise ACDAnalysisError(f"aligned actions differ: {key}")
        if len({row["tensor_sha256"]["actions"] for row in shuffled}) != 1:
            raise ACDAnalysisError(f"shuffled actions differ: {key}")
        aligned_action = aligned[0]["tensor_sha256"]["actions"]
        shuffled_action = shuffled[0]["tensor_sha256"]["actions"]
        if aligned_action == shuffled_action:
            raise ACDAnalysisError(f"shuffled action perturbation is null: {key}")
        descriptors = registration["validation_descriptors"]
        for row in aligned:
            if row.get("action_donor_clip_index") != row.get("clip_index"):
                raise ACDAnalysisError(f"aligned action donor differs: {key}")
        for row in shuffled:
            recipient = int(row["clip_index"])
            donor = int(row.get("action_donor_clip_index", -1))
            if (
                not 0 <= donor < 64
                or donor == recipient
                or descriptors[donor]["episode_dir"]
                == descriptors[recipient]["episode_dir"]
            ):
                raise ACDAnalysisError(f"shuffled donor episode differs: {key}")


def _indexed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, int, int], Mapping[str, Any]]:
    return {
        (
            str(row["endpoint"]["code"]),
            int(row["nfe"]),
            int(row["noise_seed"]),
            int(row["clip_index"]),
        ): row
        for row in rows
    }


def _paired_values(
    index: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
    treatment: str,
    treatment_nfe: int,
    comparator: str,
    comparator_nfe: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    treatment_values = []
    comparator_values = []
    for noise_seed in pilot.NOISE_SEEDS:
        for clip_index in range(64):
            treatment_values.append(
                float(index[(treatment, treatment_nfe, noise_seed, clip_index)]["metrics"][metric])
            )
            comparator_values.append(
                float(index[(comparator, comparator_nfe, noise_seed, clip_index)]["metrics"][metric])
            )
    return np.asarray(treatment_values), np.asarray(comparator_values)


def _improvement(treatment: np.ndarray, comparator: np.ndarray) -> float:
    denominator = float(comparator.mean())
    if not denominator > 0:
        raise ACDAnalysisError("comparator metric mean must be positive")
    return 100.0 * (denominator - float(treatment.mean())) / denominator


def _bootstrap_lower(
    treatment: np.ndarray,
    comparator: np.ndarray,
    *,
    seed: int,
) -> float:
    if treatment.shape != comparator.shape or treatment.shape != (256,):
        raise ACDAnalysisError("paired bootstrap requires 256 clip/noise units")
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, treatment.size, size=(BOOTSTRAPS, treatment.size))
    treatment_means = treatment[indexes].mean(axis=1)
    comparator_means = comparator[indexes].mean(axis=1)
    improvements = 100.0 * (comparator_means - treatment_means) / comparator_means
    alpha = 0.05 / BONFERRONI_FAMILY
    return float(np.quantile(improvements, alpha, method="linear"))


def _comparison(
    index: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
    *,
    treatment: str,
    treatment_nfe: int,
    comparator: str,
    comparator_nfe: int,
    metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    values, reference = _paired_values(
        index,
        treatment,
        treatment_nfe,
        comparator,
        comparator_nfe,
        metric,
    )
    return {
        "treatment": treatment,
        "treatment_nfe": treatment_nfe,
        "comparator": comparator,
        "comparator_nfe": comparator_nfe,
        "metric": metric,
        "pair_count": int(values.size),
        "treatment_mean": float(values.mean()),
        "comparator_mean": float(reference.mean()),
        "relative_improvement_percent": _improvement(values, reference),
        "simultaneous_one_sided_lower_percent": _bootstrap_lower(
            values, reference, seed=BOOTSTRAP_SEED + seed_offset
        ),
    }


def _action_sensitivity_comparison(
    index: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
    *,
    nfe: int,
    metric: str,
    seed_offset: int,
) -> dict[str, Any]:
    candidate, _unused = _paired_values(
        index, PRIMARY_CANDIDATE, nfe, PRIMARY_CANDIDATE, nfe, metric
    )
    candidate_shuffled, _unused = _paired_values(
        index, SHUFFLED_CANDIDATE, nfe, SHUFFLED_CANDIDATE, nfe, metric
    )
    parent, _unused = _paired_values(index, PARENT, 1, PARENT, 1, metric)
    parent_shuffled, _unused = _paired_values(
        index, SHUFFLED_PARENT, 1, SHUFFLED_PARENT, 1, metric
    )

    def sensitivity(aligned: np.ndarray, shuffled: np.ndarray) -> float:
        denominator = float(shuffled.mean())
        if not denominator > 0:
            raise ACDAnalysisError("shuffled action metric mean must be positive")
        return 100.0 * (denominator - float(aligned.mean())) / denominator

    candidate_sensitivity = sensitivity(candidate, candidate_shuffled)
    parent_sensitivity = sensitivity(parent, parent_shuffled)
    difference = candidate_sensitivity - parent_sensitivity
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indexes = generator.integers(0, 256, size=(BOOTSTRAPS, 256))
    candidate_boot = 100.0 * (
        candidate_shuffled[indexes].mean(axis=1) - candidate[indexes].mean(axis=1)
    ) / candidate_shuffled[indexes].mean(axis=1)
    parent_boot = 100.0 * (
        parent_shuffled[indexes].mean(axis=1) - parent[indexes].mean(axis=1)
    ) / parent_shuffled[indexes].mean(axis=1)
    lower = float(
        np.quantile(
            candidate_boot - parent_boot,
            0.05 / BONFERRONI_FAMILY,
            method="linear",
        )
    )
    return {
        "metric": metric,
        "candidate_sensitivity_percent": candidate_sensitivity,
        "parent_nfe1_sensitivity_percent": parent_sensitivity,
        "candidate_minus_parent_percentage_points": difference,
        "simultaneous_one_sided_lower_percentage_points": lower,
        "passed": difference >= -1.0 and lower >= -1.0,
    }


def _latency_p95(rows: Sequence[Mapping[str, Any]], endpoint: str, nfe: int) -> float:
    batches: dict[str, tuple[float, ...]] = {}
    for row in rows:
        if row["endpoint"]["code"] == endpoint and int(row["nfe"]) == nfe:
            key = str(row["batch_key"])
            values = tuple(
                float(value)
                for value in row["latency"]["complete_sampler_seconds_per_batch"]
            )
            if key in batches and batches[key] != values:
                raise ACDAnalysisError("duplicated batch timing differs")
            batches[key] = values
    flattened = [value for values in batches.values() for value in values]
    if len(flattened) < 120:
        raise ACDAnalysisError("fewer than 120 synchronized timing repeats")
    return float(np.quantile(np.asarray(flattened), 0.95, method="linear"))


def analyze(registration: Mapping[str, Any], inventory: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = _indexed(rows)
    comparisons: dict[str, Any] = {}
    decisions = []
    seed_offset = 0
    for nfe in pilot.NFE_GRID:
        control_results = {}
        parent_results = {}
        online_results = {}
        action_sensitivity = {}
        for metric in METRICS:
            control_results[metric] = _comparison(
                index,
                treatment=PRIMARY_CANDIDATE,
                treatment_nfe=nfe,
                comparator=PRIMARY_CONTROL,
                comparator_nfe=nfe,
                metric=metric,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            parent_results[metric] = _comparison(
                index,
                treatment=PRIMARY_CANDIDATE,
                treatment_nfe=nfe,
                comparator=PARENT,
                comparator_nfe=1,
                metric=metric,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            online_results[metric] = _comparison(
                index,
                treatment=ONLINE_CANDIDATE,
                treatment_nfe=nfe,
                comparator=ONLINE_CONTROL,
                comparator_nfe=nfe,
                metric=metric,
                seed_offset=10_000 + seed_offset,
            )
        for metric in (
            "decoded_mse_unit_range",
            "decoded_temporal_difference_mse_unit_range",
        ):
            action_sensitivity[metric] = _action_sensitivity_comparison(
                index,
                nfe=nfe,
                metric=metric,
                seed_offset=20_000 + seed_offset,
            )
            seed_offset += 1
        decoded = (
            "decoded_mse_unit_range",
            "decoded_temporal_difference_mse_unit_range",
        )
        noninferiority = ("video_future_nmse", "lpips_alex_frame")
        control_quality = all(
            control_results[metric]["relative_improvement_percent"] >= 3.0
            and control_results[metric]["simultaneous_one_sided_lower_percent"] >= 1.0
            for metric in decoded
        ) and all(
            control_results[metric]["relative_improvement_percent"] >= 0.0
            and control_results[metric]["simultaneous_one_sided_lower_percent"] > -1.0
            for metric in noninferiority
        )
        parent_quality = all(
            parent_results[metric]["relative_improvement_percent"] > 0.0
            and parent_results[metric]["simultaneous_one_sided_lower_percent"] > 0.0
            for metric in decoded
        ) and all(
            parent_results[metric]["relative_improvement_percent"] > -1.0
            and parent_results[metric]["simultaneous_one_sided_lower_percent"] > -1.0
            for metric in noninferiority
        )
        sign_agreement = all(
            np.sign(control_results[metric]["relative_improvement_percent"])
            == np.sign(online_results[metric]["relative_improvement_percent"])
            for metric in METRICS
        )
        candidate_p95 = _latency_p95(rows, PRIMARY_CANDIDATE, nfe)
        control_p95 = _latency_p95(rows, PRIMARY_CONTROL, nfe)
        latency_ratio = candidate_p95 / control_p95
        latency_gate = latency_ratio <= 1.05
        action_gate = all(value["passed"] for value in action_sensitivity.values())
        passed = (
            control_quality
            and parent_quality
            and sign_agreement
            and action_gate
            and latency_gate
        )
        comparisons[str(nfe)] = {
            "candidate_vs_compatible_control": control_results,
            "candidate_vs_fresh_parent_nfe1": parent_results,
            "online_candidate_vs_online_control": online_results,
            "episode_shuffled_action_sensitivity": action_sensitivity,
            "control_quality_gate": control_quality,
            "parent_frontier_gate": parent_quality,
            "online_ema_sign_agreement": sign_agreement,
            "action_sensitivity_gate": action_gate,
            "candidate_complete_p95_seconds": candidate_p95,
            "control_complete_p95_seconds": control_p95,
            "candidate_control_latency_ratio": latency_ratio,
            "latency_gate": latency_gate,
            "passed": passed,
        }
        if passed:
            decisions.append(nfe)
    means: dict[str, Any] = {}
    for endpoint in pilot.ENDPOINTS:
        means[endpoint.code] = {}
        for nfe in pilot.NFE_GRID:
            selected = [
                row
                for row in rows
                if row["endpoint"]["code"] == endpoint.code and int(row["nfe"]) == nfe
            ]
            means[endpoint.code][str(nfe)] = {
                metric: float(np.mean([row["metrics"][metric] for row in selected]))
                for metric in METRICS
            }
    decision = "GO_ACD" if decisions else "NO_GO_ACD"
    selected_nfe = min(decisions) if decisions else None
    return pilot.identity_payload(
        {
            "kind": "acd_p0_analysis",
            "registration_identity_sha256": registration["identity_sha256"],
            "inventory_identity_sha256": inventory["identity_sha256"],
            "source_commit": registration["tool_repository"]["git_commit"],
            "decision": decision,
            "selected_nfe": selected_nfe,
            "passing_nfe": decisions,
            "comparisons": comparisons,
            "factorial_endpoint_means": means,
            "bootstrap": {
                "resampling_unit": "clip/noise pair",
                "pair_count": 256,
                "replicates": BOOTSTRAPS,
                "seed": BOOTSTRAP_SEED,
                "one_sided_family_alpha": 0.05,
                "bonferroni_family": BONFERRONI_FAMILY,
            },
            "primary_candidate": PRIMARY_CANDIDATE,
            "primary_compatible_control": PRIMARY_CONTROL,
            "fresh_parent_frontier": f"{PARENT}@nfe1",
            "cross_readouts_nonselectable": True,
            "full_2x2_readout_factorial": True,
            "direct_baseline_not_dual_novelty": True,
            "observed_reserved_b200_hours": inventory["observed_reserved_b200_hours"],
            "artifact_bytes_under_registered_root": inventory[
                "artifact_bytes_under_registered_root"
            ],
            "resource_ceiling_passed": (
                float(inventory["observed_reserved_b200_hours"])
                <= pilot.resource_plan()["maximum_reserved_b200_hours"]
                and int(inventory["artifact_bytes_under_registered_root"])
                <= pilot.resource_plan()["artifact_ceiling_bytes"]
            ),
            "wandb_writes": 0,
            "protected_test_accessed": False,
        }
    )


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# ACD-P0 result",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| NFE | vs control | vs parent@1 | online/EMA sign | action guard | latency | pass |",
        "|---:|---|---|---|---|---|---|",
    ]
    for nfe in pilot.NFE_GRID:
        row = result["comparisons"][str(nfe)]
        lines.append(
            f"| {nfe} | {row['control_quality_gate']} | {row['parent_frontier_gate']} | "
            f"{row['online_ema_sign_agreement']} | {row['action_sensitivity_gate']} | "
            f"{row['latency_gate']} | {row['passed']} |"
        )
    lines.extend(
        (
            "",
            "The selectable comparison is consistency-objective + Karras readout versus "
            "RF-MSE + native-RF readout. Cross-readouts are diagnostic only. This is a "
            "feature-free direct baseline and cannot establish dual-diffusion novelty.",
            "",
        )
    )
    return "\n".join(lines)


def command_analyze(args: argparse.Namespace) -> int:
    registration = pilot.validate_registration(args.registration)
    expected_evaluation = Path(registration["output_root"]) / "evaluation"
    if args.evaluation.resolve(strict=True) != expected_evaluation.resolve(strict=True):
        raise ACDAnalysisError("evaluation root differs from registration")
    inventory = pilot.read_json(args.evaluation / "inventory.json", "endpoint inventory")
    if (
        not pilot.identity_valid(inventory)
        or inventory.get("kind") != "acd_p0_endpoint_inventory"
        or inventory.get("registration_identity_sha256") != registration["identity_sha256"]
        or inventory.get("full_2x2_readout_factorial") is not True
        or inventory.get("global_endpoint_barrier_passed") is not True
        or inventory.get("future_rgb_opened_before_global_endpoint_barrier") is not False
        or inventory.get("teacher_calls") != 0
        or inventory.get("feature_calls") != 0
        or inventory.get("wandb_writes") != 0
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("source_commit")
        != registration["tool_repository"]["git_commit"]
        or inventory.get("endpoint_codes")
        != [endpoint.code for endpoint in pilot.ENDPOINTS]
        or inventory.get("nfe_grid") != list(pilot.NFE_GRID)
        or inventory.get("noise_seeds") != list(pilot.NOISE_SEEDS)
    ):
        raise ACDAnalysisError("endpoint inventory differs")
    rows = _load_rows(args.evaluation, inventory, registration)
    result = analyze(registration, inventory, rows)
    for output in (args.output_json, args.output_markdown):
        if output.exists() or output.is_symlink():
            raise ACDAnalysisError("analysis outputs must be fresh")
        if not output.is_absolute() or not output.parent.is_dir():
            raise ACDAnalysisError("analysis output parent must exist and be absolute")
    pilot.exclusive_json(args.output_json, result)
    with args.output_markdown.open("xb") as handle:
        handle.write(_markdown(result).encode("utf-8"))
        handle.flush()
    print(str(args.output_json))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
