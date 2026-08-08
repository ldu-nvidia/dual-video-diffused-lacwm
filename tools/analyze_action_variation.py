#!/usr/bin/env python3
"""Analyze the preregistered paired action-variation VPM screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import action_variation_evaluate as evaluation  # noqa: E402
from tools import action_variation_screen as screen  # noqa: E402
from tools import two_clock_consistency_evaluate as base  # noqa: E402


SCHEMA_VERSION = 1
KIND_ANALYSIS = "action_variation_validation_analysis"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_808
QUALITY_CONTRASTS = 9
ATTRIBUTION_CONTRASTS = 6
PRIMARY_METRIC = "decoded_temporal_difference_mse_unit_range"
GUARDRAILS = ("video_future_nmse", "decoded_mse_unit_range")
CLAIM_METRICS = (PRIMARY_METRIC, *GUARDRAILS)


class ActionVariationAnalysisError(RuntimeError):
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
    contrast_count: int,
) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if (
        left.shape != (evaluation.EXPECTED_VALIDATION_CLIPS,)
        or right.shape != left.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(right <= 0)
    ):
        raise ActionVariationAnalysisError(f"invalid paired values for {label}")
    rng = np.random.default_rng(_derived_seed(label))
    indexes = rng.integers(
        0,
        left.size,
        size=(BOOTSTRAP_SAMPLES, left.size),
        endpoint=False,
    )
    left_means = left[indexes].mean(axis=1)
    right_means = right[indexes].mean(axis=1)
    effects = (right_means - left_means) / right_means
    confidence = 1.0 - 0.05 / contrast_count
    point = (right.mean() - left.mean()) / right.mean()
    return {
        "n_paired_clips": int(left.size),
        "candidate_mean": float(left.mean()),
        "reference_mean": float(right.mean()),
        "relative_improvement": float(point),
        "relative_improvement_percent": float(100.0 * point),
        "one_sided_simultaneous_lower_bound": {
            "confidence": confidence,
            "familywise_alpha": 0.05,
            "family_contrast_count": contrast_count,
            "low": float(np.quantile(effects, 1.0 - confidence)),
        },
        "descriptive_two_sided_95_ci": {
            "low": float(np.quantile(effects, 0.025)),
            "high": float(np.quantile(effects, 0.975)),
        },
        "favorable_clip_fraction": float(np.mean(right > left)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_derived_seed": _derived_seed(label),
    }


def _effect_gate(
    effects: Mapping[str, Any], *, primary_minimum: float
) -> dict[str, Any]:
    values = {}
    for metric in CLAIM_METRICS:
        effect = effects.get(metric, {})
        point = effect.get("relative_improvement")
        low = effect.get("one_sided_simultaneous_lower_bound", {}).get("low")
        if not isinstance(point, (int, float)) or not isinstance(low, (int, float)):
            raise ActionVariationAnalysisError(f"missing effect for {metric}")
        values[metric] = (float(point), float(low))
    point, low = values[PRIMARY_METRIC]
    checks = {
        "primary_point": point >= primary_minimum,
        "primary_simultaneous_lower_bound": low >= primary_minimum,
    }
    for metric in GUARDRAILS:
        point, low = values[metric]
        checks[f"{metric}_point_above_minus_one_percent"] = point > -0.01
        checks[f"{metric}_lower_bound_above_minus_one_percent"] = low > -0.01
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "primary_minimum_relative_improvement": primary_minimum,
    }


def _load_inventory(
    path: Path, arm: screen.Arm
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    inventory = base._read_json(path.resolve(strict=True), "evaluation inventory")
    if (
        not screen.identity_valid(inventory)
        or inventory.get("kind") != evaluation.KIND_INVENTORY
        or inventory.get("arm") != asdict_arm(arm)
        or inventory.get("evaluation_split") != "validation"
        or inventory.get("validation_clips") != 64
        or inventory.get("endpoints")
        != [asdict_endpoint(endpoint) for endpoint in evaluation.ENDPOINTS]
        or inventory.get("row_count") != 64 * len(evaluation.ENDPOINTS)
        or inventory.get("actual_transformer_call_count") != 288
        or inventory.get("paired_inputs_and_noise_within_arm") is not True
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("target_cache_array_opened") is not False
        or inventory.get("future_or_clean_feature_used_at_sampling") is not False
        or not isinstance(inventory.get("arm_artifacts"), Mapping)
    ):
        raise ActionVariationAnalysisError("evaluation inventory differs")
    registration_path = Path(inventory["registration"]["path"])
    if inventory["registration"]["sha256"] != base._sha256(registration_path):
        raise ActionVariationAnalysisError("registration changed")
    registration = screen.validate_registration(registration_path)
    rows = []
    root = path.resolve().parent
    for rank in range(8):
        row_path = root / f"rank_{rank:03d}.jsonl"
        receipt_path = root / f"rank_{rank:03d}_complete.json"
        receipt = base._read_json(receipt_path, "rank receipt")
        if receipt.get("rows", {}).get("sha256") != base._sha256(row_path):
            raise ActionVariationAnalysisError("rank row file changed")
        rows.extend(
            json.loads(line) for line in row_path.read_text().splitlines() if line
        )
    evaluation._validate_rows(rows, arm, registration)
    indexed = {(int(row["clip_index"]), row["endpoint"]["code"]): row for row in rows}
    return {"inventory": inventory, "registration": registration}, indexed


def asdict_arm(arm: screen.Arm) -> dict[str, Any]:
    return {
        "code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "residual_enabled": arm.residual_enabled,
    }


def asdict_endpoint(endpoint: screen.Endpoint) -> dict[str, Any]:
    return {
        "code": endpoint.code,
        "nfe": endpoint.nfe,
        "action_source": endpoint.action_source,
        "primary_gate": endpoint.primary_gate,
    }


def _find_metric(metrics: Mapping[str, Any], suffix: str) -> Any:
    matches = [value for key, value in metrics.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise ActionVariationAnalysisError(
            f"training metric {suffix} is absent/ambiguous"
        )
    return matches[0]


def _assert_training_pair(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    # Trace records are embedded in every row's arm snapshot/training metadata;
    # load the canonical run directories from registration instead.
    registration = control["registration"]
    if registration["identity_sha256"] != candidate["registration"]["identity_sha256"]:
        raise ActionVariationAnalysisError("arms do not share one registration")
    traces = []
    for arm, record in zip(screen.ARMS, (control, candidate)):
        run_dir = Path(registration["output_root"]) / "training" / arm.run_name
        trace_path = run_dir / "action_variation_training_trace.jsonl"
        expected_trace = record["inventory"]["arm_artifacts"]["training_trace"]["trace"]
        if expected_trace.get("path") != str(trace_path) or expected_trace.get(
            "sha256"
        ) != base._sha256(trace_path):
            raise ActionVariationAnalysisError(
                "training trace changed after evaluation"
            )
        rows = [
            json.loads(line) for line in trace_path.read_text().splitlines() if line
        ]
        if not rows:
            raise ActionVariationAnalysisError("training trace is empty")
        train_events = [
            event
            for event in rows[1:]
            if isinstance(event.get("metrics"), Mapping)
            and "train_loss/loss" in event["metrics"]
        ]
        if len(train_events) != 200:
            raise ActionVariationAnalysisError("training trace length differs")
        traces.append((rows[0], train_events))
    header_left, header_right = traces[0][0], traces[1][0]
    if (
        header_left.get("initial_action_residual_state_sha256")
        != header_right.get("initial_action_residual_state_sha256")
        or header_left.get("stats_file_sha256") != header_right.get("stats_file_sha256")
        or header_left.get("initial_effective_gate") != 0.0
        or header_right.get("initial_effective_gate") != 0.0
    ):
        raise ActionVariationAnalysisError("paired initial adapter/state differs")
    input_fields = (
        "actions",
        "future_actions",
        "standardized_action_delta",
        "clean_latent",
        "noisy_latent",
        "timesteps",
        "reference",
    )
    for step, (event_left, event_right) in enumerate(zip(traces[0][1], traces[1][1])):
        if event_left.get("total_observations") != event_right.get(
            "total_observations"
        ):
            raise ActionVariationAnalysisError(
                f"observation count differs at step {step}"
            )
        metrics_left = event_left.get("metrics", {})
        metrics_right = event_right.get("metrics", {})
        for field in input_fields:
            suffix = f"paired_audit/exact_{field}_all_ranks_sha256"
            if _find_metric(metrics_left, suffix) != _find_metric(
                metrics_right, suffix
            ):
                raise ActionVariationAnalysisError(
                    f"paired training input/noise differs at step {step}: {field}"
                )
    return {
        "paired_updates": 200,
        "same_initial_action_residual_state": True,
        "same_stats": True,
        "same_exact_actions_latents_timesteps_noise_and_reference": True,
        "initial_action_residual_state_sha256": header_left[
            "initial_action_residual_state_sha256"
        ],
    }


def _metric(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]], endpoint: str, metric: str
) -> list[float]:
    return [
        float(indexed[(index, endpoint)]["metrics"][metric])
        for index in range(evaluation.EXPECTED_VALIDATION_CLIPS)
    ]


def _effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    probability = energy / max(float(energy.sum()), 1e-30)
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _control_diagnostic(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    aligned = np.asarray(
        [
            indexed[(index, "aligned_nfe_1")]["wan_action_control_probe"]
            for index in range(64)
        ],
        dtype=np.float64,
    )
    shuffled = np.asarray(
        [
            indexed[(index, "global_shuffled_nfe_1")]["wan_action_control_probe"]
            for index in range(64)
        ],
        dtype=np.float64,
    )
    zero = np.asarray(
        [
            indexed[(index, "zero_nfe_1")]["wan_action_control_probe"]
            for index in range(64)
        ],
        dtype=np.float64,
    )
    cosine = np.sum(aligned * shuffled, axis=1) / np.maximum(
        np.linalg.norm(aligned, axis=1) * np.linalg.norm(shuffled, axis=1), 1e-30
    )
    rms = math.sqrt(float(np.mean(aligned**2)))
    return {
        "aligned_centered_effective_rank": _effective_rank(aligned),
        "aligned_rms": rms,
        "aligned_sample_std_to_rms": float(
            np.sqrt(np.mean(np.var(aligned, axis=0, ddof=1))) / max(rms, 1e-30)
        ),
        "aligned_vs_episode_shuffled_cosine_mean": float(cosine.mean()),
        "aligned_vs_episode_shuffled_difference_to_rms": float(
            np.sqrt(np.mean((aligned - shuffled) ** 2)) / max(rms, 1e-30)
        ),
        "aligned_vs_zero_difference_to_rms": float(
            np.sqrt(np.mean((aligned - zero) ** 2)) / max(rms, 1e-30)
        ),
    }


def analyze(
    control_inventory: Path,
    candidate_inventory: Path,
) -> dict[str, Any]:
    control_record, control_rows = _load_inventory(
        control_inventory, screen.ARM_BY_CODE["AV-CONT"]
    )
    candidate_record, candidate_rows = _load_inventory(
        candidate_inventory, screen.ARM_BY_CODE["AV-DELTA"]
    )
    training_pair = _assert_training_pair(control_record, candidate_record)
    if set(control_rows) != set(candidate_rows):
        raise ActionVariationAnalysisError("cross-arm clip/endpoint inventory differs")
    pairing_fields = (
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "auxiliary_initial_noise_sha256",
        "sampler_actions_sha256",
    )
    for key in control_rows:
        if any(
            control_rows[key]["tensor_sha256"][field]
            != candidate_rows[key]["tensor_sha256"][field]
            for field in pairing_fields
        ):
            raise ActionVariationAnalysisError(f"cross-arm paired input differs: {key}")

    quality = {}
    for nfe in screen.NFE_GRID:
        endpoint = f"aligned_nfe_{nfe}"
        effects = {
            metric: paired_effect(
                _metric(candidate_rows, endpoint, metric),
                _metric(control_rows, endpoint, metric),
                label=f"quality:nfe{nfe}:{metric}",
                contrast_count=QUALITY_CONTRASTS,
            )
            for metric in CLAIM_METRICS
        }
        quality[str(nfe)] = {
            "effects": effects,
            "gate": _effect_gate(effects, primary_minimum=0.01),
            "claim_eligible": nfe == 1,
        }

    attribution = {}
    for diagnostic in ("zero_nfe_1", "global_shuffled_nfe_1"):
        effects = {
            metric: paired_effect(
                _metric(candidate_rows, "aligned_nfe_1", metric),
                _metric(candidate_rows, diagnostic, metric),
                label=f"attribution:{diagnostic}:{metric}",
                contrast_count=ATTRIBUTION_CONTRASTS,
            )
            for metric in CLAIM_METRICS
        }
        attribution[diagnostic] = {
            "effects": effects,
            "gate": _effect_gate(effects, primary_minimum=0.005),
        }

    quality_pass = bool(quality["1"]["gate"]["passed"])
    attribution_pass = all(value["gate"]["passed"] for value in attribution.values())
    control_diagnostic = _control_diagnostic(control_rows)
    candidate_diagnostic = _control_diagnostic(candidate_rows)
    return screen.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_ANALYSIS,
            "created_at_utc": _now(),
            "registration_identity_sha256": control_record["registration"][
                "identity_sha256"
            ],
            "protected_test_accessed": False,
            "training_pair": training_pair,
            "quality_candidate_vs_matched_control": quality,
            "candidate_action_attribution": attribution,
            "wan_action_control_diagnostic": {
                "control": control_diagnostic,
                "candidate": candidate_diagnostic,
                "candidate_to_control_effective_rank_ratio": float(
                    candidate_diagnostic["aligned_centered_effective_rank"]
                    / max(control_diagnostic["aligned_centered_effective_rank"], 1e-30)
                ),
            },
            "latency": {
                "control": control_record["inventory"]["latency_ms_per_local_batch"],
                "candidate": candidate_record["inventory"][
                    "latency_ms_per_local_batch"
                ],
                "same_transformer_calls": True,
                "nfe_one_is_one_wan_call": True,
            },
            "learned_action_residual_gate": {
                "control": control_record["inventory"]["arm_artifacts"][
                    "action_variation"
                ],
                "candidate": candidate_record["inventory"]["arm_artifacts"][
                    "action_variation"
                ],
            },
            "decision": {
                "passed": quality_pass and attribution_pass,
                "nfe_1_quality_passed": quality_pass,
                "aligned_beats_zero_and_episode_shuffled": attribution_pass,
                "rule": (
                    "NFE-1 candidate-vs-control temporal point/LB >=1% with >-1% "
                    "guardrails, and candidate aligned-vs-zero plus aligned-vs-episode-"
                    "shuffled temporal point/LB >=0.5% with >-1% guardrails"
                ),
                "protected_test_authorized": False,
            },
        }
    )


def command_analyze(args: argparse.Namespace) -> int:
    result = analyze(args.control_inventory, args.candidate_inventory)
    output = args.output.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ActionVariationAnalysisError(
            "analysis output must be a fresh absolute path"
        )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    base._exclusive_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-inventory", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
