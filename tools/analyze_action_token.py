#!/usr/bin/env python3
"""Analyze the preregistered paired action-token VPM screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import action_token_evaluate as evaluation  # noqa: E402
from tools import action_token_screen as screen  # noqa: E402
from tools import two_clock_consistency_evaluate as base  # noqa: E402

SCHEMA_VERSION = 1
KIND_ANALYSIS = "action_token_validation_analysis"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_808
QUALITY_CONTRASTS = 9
ATTRIBUTION_CONTRASTS = 6
INTERACTION_CONTRASTS = 6
HARD_MASK_CONTRASTS = 3
PRIMARY_METRIC = "decoded_temporal_difference_mse_unit_range"
GUARDRAILS = ("video_future_nmse", "decoded_mse_unit_range")
CLAIM_METRICS = (PRIMARY_METRIC, *GUARDRAILS)


class ActionTokenAnalysisError(RuntimeError):
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
        raise ActionTokenAnalysisError(f"invalid paired values for {label}")
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


def paired_interaction_effect(
    candidate_aligned: Sequence[float],
    candidate_diagnostic: Sequence[float],
    control_aligned: Sequence[float],
    control_diagnostic: Sequence[float],
    *,
    label: str,
    contrast_count: int,
) -> dict[str, Any]:
    """Paired action-specific difference-in-differences, normalized to control."""

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (
            candidate_aligned,
            candidate_diagnostic,
            control_aligned,
            control_diagnostic,
        )
    ]
    if any(
        value.shape != (evaluation.EXPECTED_VALIDATION_CLIPS,)
        or not np.isfinite(value).all()
        for value in arrays
    ) or np.any(arrays[2] <= 0):
        raise ActionTokenAnalysisError(f"invalid interaction values for {label}")
    candidate_gap = arrays[1] - arrays[0]
    control_gap = arrays[3] - arrays[2]
    interaction = candidate_gap - control_gap
    rng = np.random.default_rng(_derived_seed(label))
    indexes = rng.integers(
        0,
        evaluation.EXPECTED_VALIDATION_CLIPS,
        size=(BOOTSTRAP_SAMPLES, evaluation.EXPECTED_VALIDATION_CLIPS),
        endpoint=False,
    )
    denominator = arrays[2][indexes].mean(axis=1)
    effects = interaction[indexes].mean(axis=1) / denominator
    confidence = 1.0 - 0.05 / contrast_count
    point = float(interaction.mean() / arrays[2].mean())
    return {
        "n_paired_clips": evaluation.EXPECTED_VALIDATION_CLIPS,
        "candidate_diagnostic_minus_aligned_mean": float(candidate_gap.mean()),
        "control_diagnostic_minus_aligned_mean": float(control_gap.mean()),
        "relative_improvement": point,
        "relative_improvement_percent": 100.0 * point,
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
        "favorable_clip_fraction": float(np.mean(interaction > 0)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_derived_seed": _derived_seed(label),
        "estimand": (
            "((candidate_diagnostic-candidate_aligned)-"
            "(control_diagnostic-control_aligned))/mean(control_aligned)"
        ),
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
            raise ActionTokenAnalysisError(f"missing effect for {metric}")
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


@contextmanager
def _registered_config_environment(registration: Mapping[str, Any]):
    """Resolve a saved Hydra config only against its registered inputs."""

    environment = {
        "WAN_DIR": registration["runtime"]["wan_dir"],
        "VIDEOX_HOME": registration["runtime"]["videox_home"],
        "ACTION_TOKEN_VPM_SNAPSHOT": registration["controlled_study"][
            "parent_snapshot"
        ]["path"],
        "ACTION_TOKEN_TRAIN_CLIP_MANIFEST": registration["training"]["manifest"][
            "path"
        ],
        "ACTION_TOKEN_TRAIN_CACHE_METADATA": registration["training"][
            "cache_metadata"
        ]["path"],
        "ACTION_TOKEN_VAL_CLIP_MANIFEST": registration["validation"]["manifest"][
            "path"
        ],
        "ACTION_TOKEN_VAL_CACHE_METADATA": registration["validation"][
            "cache_metadata"
        ]["path"],
        "ACTION_TOKEN_STATS": registration["action_token_stats"]["file"]["path"],
        "ACTION_TOKEN_STATS_SHA256": registration["action_token_stats"]["file"][
            "sha256"
        ],
        "ACTION_TOKEN_RUN_ROOT": str(
            Path(registration["output_root"]) / "training"
        ),
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_arm_artifacts(
    inventory: Mapping[str, Any],
    registration: Mapping[str, Any],
    arm: screen.Arm,
) -> None:
    """Revalidate every training artifact referenced by evaluation."""

    from omegaconf import OmegaConf

    artifacts = inventory.get("arm_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ActionTokenAnalysisError("arm artifact inventory is absent")
    run_dir = Path(registration["output_root"]) / "training" / arm.run_name
    expected_identity = screen.arm_run_identity(registration, arm)
    config_path = run_dir / ".hydra" / "config.yaml"
    with _registered_config_environment(registration):
        config = OmegaConf.load(config_path)
        evaluation._validate_config(config, arm, registration)
        plan = evaluation._validate_arm_plan(registration, arm, run_dir)
        if screen.canonical_config_contract(config) != plan["resolved_config_contract"]:
            raise ActionTokenAnalysisError("saved arm config differs from its plan")
    completion_path = run_dir / "training_complete.json"
    completion = base._read_json(completion_path, "training completion")
    runtime_verification = plan["runtime_verification"]
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != 200
        or completion.get("max_iter") != 200
        or completion.get("run_identity_sha256") != expected_identity
        or completion.get("snapshot")
        != str((run_dir / "snapshot.pt").resolve(strict=True))
        or completion.get("runtime_verification_receipt") != runtime_verification
    ):
        raise ActionTokenAnalysisError("training completion differs")
    wandb_completion = evaluation._validate_wandb_completion_receipt(
        registration, arm, run_dir, rehash_snapshot=True
    )
    expected = {
        "snapshot": wandb_completion["trained_snapshot"],
        "config": base._file_record(config_path),
        "training_trace": evaluation._validate_trace(run_dir, arm, registration),
        "training_completion": base._file_record(completion_path),
        "wandb_completion_receipt": wandb_completion,
        "arm_execution_plan": plan,
        "runtime_verification_receipt": runtime_verification,
        "run_identity_sha256": expected_identity,
        "wandb_run_id": expected_identity,
    }
    if set(artifacts) != {*expected, "action_token"} or any(
        artifacts.get(key) != value for key, value in expected.items()
    ):
        raise ActionTokenAnalysisError("arm artifacts changed after evaluation")
    variation = artifacts.get("action_token")
    raw_gate = variation.get("raw_gate") if isinstance(variation, Mapping) else None
    effective_gate = (
        variation.get("effective_gate") if isinstance(variation, Mapping) else None
    )
    if (
        not isinstance(variation, Mapping)
        or set(variation)
        != {
            "enabled",
            "raw_gate",
            "effective_gate",
            "stats_file_sha256",
            "stats_identity_sha256",
        }
        or variation.get("enabled") is not arm.action_token_enabled
        or variation.get("stats_file_sha256")
        != registration["action_token_stats"]["file"]["sha256"]
        or variation.get("stats_identity_sha256")
        != registration["action_token_stats"]["payload"]["identity_sha256"]
        or not isinstance(raw_gate, (int, float))
        or not math.isfinite(float(raw_gate))
        or not isinstance(effective_gate, (int, float))
        or not math.isfinite(float(effective_gate))
        or abs(float(effective_gate)) >= 1.0
        or (not arm.action_token_enabled and float(effective_gate) != 0.0)
    ):
        raise ActionTokenAnalysisError("learned action-token record differs")


def _validate_latency_summary(
    inventory: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    recomputed: dict[str, Any] = {}
    for endpoint in evaluation.ENDPOINTS:
        endpoint_rows = [
            row for row in rows if row["endpoint"]["code"] == endpoint.code
        ]
        recomputed[endpoint.code] = {}
        for stage in (
            "prepare_history_and_action",
            "wan_trajectory",
            "decode",
            "total",
        ):
            values = sorted(
                float(row["latency_ms_per_local_batch"][stage]) for row in endpoint_rows
            )
            recomputed[endpoint.code][stage] = {
                "median": float(values[len(values) // 2]),
                "p95": float(values[math.ceil(0.95 * len(values)) - 1]),
            }
    if inventory.get("latency_ms_batch_one_by_endpoint") != recomputed:
        raise ActionTokenAnalysisError("latency inventory was not row-derived")


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
        or inventory.get("actual_transformer_call_count")
        != evaluation.EXPECTED_WORLD_SIZE
        * evaluation._expected_rank_transformer_calls()
        or inventory.get("batch_size_per_rank") != 1
        or inventory.get("latency_warmup")
        != {
            "rollouts_per_rank": 1,
            "nfe": 1,
            "excluded_from_endpoint_call_accounting": True,
        }
        or inventory.get("latency_semantics") != evaluation.LATENCY_SEMANTICS
        or inventory.get("paired_inputs_and_noise_within_arm") is not True
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("target_cache_array_opened") is not False
        or inventory.get("future_or_clean_feature_used_at_sampling") is not False
        or not isinstance(inventory.get("arm_artifacts"), Mapping)
        or set(inventory.get("latency_ms_batch_one_by_endpoint", {}))
        != {endpoint.code for endpoint in evaluation.ENDPOINTS}
    ):
        raise ActionTokenAnalysisError("evaluation inventory differs")
    registration_path = Path(inventory["registration"]["path"])
    if inventory["registration"]["sha256"] != base._sha256(registration_path):
        raise ActionTokenAnalysisError("registration changed")
    registration = screen.validate_registration(registration_path)
    if inventory.get("registration") != base._file_record(registration_path):
        raise ActionTokenAnalysisError("registration record differs")
    screen.revalidate_execution_environment(registration)
    _validate_arm_artifacts(inventory, registration, arm)
    rows = []
    receipt_records = []
    root = path.resolve().parent
    for rank in range(8):
        row_path = root / f"rank_{rank:03d}.jsonl"
        receipt_path = root / f"rank_{rank:03d}_complete.json"
        receipt = base._read_json(receipt_path, "rank receipt")
        if (
            not screen.identity_valid(receipt)
            or receipt.get("kind") != evaluation.KIND_RANK
            or receipt.get("rank") != rank
            or receipt.get("arm") != asdict_arm(arm)
            or receipt.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or receipt.get("world_size") != evaluation.EXPECTED_WORLD_SIZE
            or receipt.get("batch_size_per_rank") != 1
            or receipt.get("unmeasured_warmup_transformer_calls") != 1
            or receipt.get("assigned_clip_indexes")
            != evaluation._expected_rank_indexes(rank)
            or receipt.get("endpoints")
            != [asdict_endpoint(endpoint) for endpoint in evaluation.ENDPOINTS]
            or receipt.get("actual_transformer_call_count")
            != evaluation._expected_rank_transformer_calls()
            or receipt.get("rows")
            != {
                **base._file_record(row_path),
                "count": len(evaluation._expected_rank_indexes(rank))
                * len(evaluation.ENDPOINTS),
            }
            or receipt.get("protected_test_accessed") is not False
            or receipt.get("target_cache_array_opened") is not False
        ):
            raise ActionTokenAnalysisError("rank row file changed")
        receipt_records.append(base._file_record(receipt_path))
        rows.extend(
            json.loads(line) for line in row_path.read_text().splitlines() if line
        )
    evaluation._validate_rows(
        rows,
        arm,
        registration,
        expected_snapshot=inventory["arm_artifacts"]["snapshot"],
    )
    if inventory.get("rank_receipts") != receipt_records:
        raise ActionTokenAnalysisError("rank receipt inventory differs")
    _validate_latency_summary(inventory, rows)
    indexed = {(int(row["clip_index"]), row["endpoint"]["code"]): row for row in rows}
    return {"inventory": inventory, "registration": registration}, indexed


def asdict_arm(arm: screen.Arm) -> dict[str, Any]:
    return {
        "code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "action_token_enabled": arm.action_token_enabled,
    }


def asdict_endpoint(endpoint: screen.Endpoint) -> dict[str, Any]:
    return {
        "code": endpoint.code,
        "nfe": endpoint.nfe,
        "action_source": endpoint.action_source,
        "token_mode": endpoint.token_mode,
        "primary_gate": endpoint.primary_gate,
    }


def _find_metric(metrics: Mapping[str, Any], suffix: str) -> Any:
    matches = [value for key, value in metrics.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise ActionTokenAnalysisError(
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
        raise ActionTokenAnalysisError("arms do not share one registration")
    traces = []
    for arm, record in zip(screen.ARMS, (control, candidate)):
        run_dir = Path(registration["output_root"]) / "training" / arm.run_name
        trace_path = run_dir / "action_token_training_trace.jsonl"
        expected_trace = record["inventory"]["arm_artifacts"]["training_trace"]["trace"]
        if expected_trace.get("path") != str(trace_path) or expected_trace.get(
            "sha256"
        ) != base._sha256(trace_path):
            raise ActionTokenAnalysisError(
                "training trace changed after evaluation"
            )
        rows = [
            json.loads(line) for line in trace_path.read_text().splitlines() if line
        ]
        if not rows:
            raise ActionTokenAnalysisError("training trace is empty")
        train_events = [
            event
            for event in rows[1:]
            if isinstance(event.get("metrics"), Mapping)
            and "train_loss/loss" in event["metrics"]
        ]
        if len(train_events) != 200:
            raise ActionTokenAnalysisError("training trace length differs")
        traces.append((rows[0], train_events))
    header_left, header_right = traces[0][0], traces[1][0]
    if (
        header_left.get("initial_action_token_state_sha256")
        != header_right.get("initial_action_token_state_sha256")
        or header_left.get("stats_file_sha256") != header_right.get("stats_file_sha256")
        or header_left.get("model_parameter_count")
        != header_right.get("model_parameter_count")
        or header_left.get("trainable_parameter_count")
        != header_right.get("trainable_parameter_count")
        or header_left.get("action_token_parameter_count")
        != header_right.get("action_token_parameter_count")
        or header_left.get("initial_effective_gate") != 0.0
        or header_right.get("initial_effective_gate") != 0.0
    ):
        raise ActionTokenAnalysisError("paired initial adapter/state differs")
    input_fields = (
        "actions",
        "future_actions",
        "standardized_actions",
        "clean_latent",
        "noisy_latent",
        "timesteps",
        "reference",
        "wan_raw_context",
    )
    for step, (event_left, event_right) in enumerate(zip(traces[0][1], traces[1][1])):
        if event_left.get("total_observations") != event_right.get(
            "total_observations"
        ):
            raise ActionTokenAnalysisError(
                f"observation count differs at step {step}"
            )
        metrics_left = event_left.get("metrics", {})
        metrics_right = event_right.get("metrics", {})
        for field in input_fields:
            suffix = f"paired_audit/exact_{field}_all_ranks_sha256"
            if _find_metric(metrics_left, suffix) != _find_metric(
                metrics_right, suffix
            ):
                raise ActionTokenAnalysisError(
                    f"paired training input/noise differs at step {step}: {field}"
                )
    return {
        "paired_updates": 200,
        "same_initial_action_token_state": True,
        "same_model_parameter_count": True,
        "same_trainable_parameter_count": True,
        "same_action_token_parameter_count": True,
        "same_stats": True,
        "same_exact_actions_latents_timesteps_noise_and_reference": True,
        "initial_action_token_state_sha256": header_left[
            "initial_action_token_state_sha256"
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
    total = float(energy.sum())
    if total <= 1e-30:
        return 0.0
    probability = energy / total
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
            indexed[(index, "episode_shuffled_nfe_1")]["wan_action_control_probe"]
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


def _context_diagnostic(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the 40 distinct action-token magnitudes seen by Wan."""

    def values(endpoint: str) -> np.ndarray:
        return np.asarray(
            [
                indexed[(index, endpoint)]["action_context_token_rms"]
                for index in range(evaluation.EXPECTED_VALIDATION_CLIPS)
            ],
            dtype=np.float64,
        )

    aligned = values("aligned_nfe_1")
    shuffled = values("episode_shuffled_nfe_1")
    zero = values("zero_nfe_1")
    masked = values("aligned_tokens_masked_nfe_1")
    if any(array.shape != (64, 40) for array in (aligned, shuffled, zero, masked)):
        raise ActionTokenAnalysisError("action-context diagnostic shape differs")
    aligned_rms = math.sqrt(float(np.mean(aligned**2)))
    cosine = np.sum(aligned * shuffled, axis=1) / np.maximum(
        np.linalg.norm(aligned, axis=1) * np.linalg.norm(shuffled, axis=1), 1e-30
    )
    return {
        "token_count": 40,
        "aligned_centered_effective_rank": _effective_rank(aligned),
        "aligned_rms": aligned_rms,
        "aligned_sample_std_to_rms": float(
            np.sqrt(np.mean(np.var(aligned, axis=0, ddof=1)))
            / max(aligned_rms, 1e-30)
        ),
        "aligned_vs_episode_shuffled_cosine_mean": float(cosine.mean()),
        "aligned_vs_episode_shuffled_difference_to_rms": float(
            np.sqrt(np.mean((aligned - shuffled) ** 2))
            / max(aligned_rms, 1e-30)
        ),
        "aligned_vs_zero_difference_to_rms": float(
            np.sqrt(np.mean((aligned - zero) ** 2))
            / max(aligned_rms, 1e-30)
        ),
        "hard_mask_abs_max": float(np.max(np.abs(masked))),
    }


def analyze(
    control_inventory: Path,
    candidate_inventory: Path,
) -> dict[str, Any]:
    control_record, control_rows = _load_inventory(
        control_inventory, screen.ARM_BY_CODE["AT-OFF"]
    )
    candidate_record, candidate_rows = _load_inventory(
        candidate_inventory, screen.ARM_BY_CODE["AT-ON"]
    )
    training_pair = _assert_training_pair(control_record, candidate_record)
    if set(control_rows) != set(candidate_rows):
        raise ActionTokenAnalysisError("cross-arm clip/endpoint inventory differs")
    pairing_fields = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
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
            raise ActionTokenAnalysisError(f"cross-arm paired input differs: {key}")

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

    candidate_attribution_descriptive = {}
    for diagnostic in ("zero_nfe_1", "episode_shuffled_nfe_1"):
        effects = {
            metric: paired_effect(
                _metric(candidate_rows, "aligned_nfe_1", metric),
                _metric(candidate_rows, diagnostic, metric),
                label=f"attribution:{diagnostic}:{metric}",
                contrast_count=ATTRIBUTION_CONTRASTS,
            )
            for metric in CLAIM_METRICS
        }
        candidate_attribution_descriptive[diagnostic] = {
            "effects": effects,
            "gate": _effect_gate(effects, primary_minimum=0.005),
        }

    interaction_attribution = {}
    for diagnostic in ("zero_nfe_1", "episode_shuffled_nfe_1"):
        effects = {
            metric: paired_interaction_effect(
                _metric(candidate_rows, "aligned_nfe_1", metric),
                _metric(candidate_rows, diagnostic, metric),
                _metric(control_rows, "aligned_nfe_1", metric),
                _metric(control_rows, diagnostic, metric),
                label=f"interaction:{diagnostic}:{metric}",
                contrast_count=INTERACTION_CONTRASTS,
            )
            for metric in CLAIM_METRICS
        }
        interaction_attribution[diagnostic] = {
            "effects": effects,
            "gate": _effect_gate(effects, primary_minimum=0.005),
        }

    hard_mask_effects = {
        metric: paired_effect(
            _metric(candidate_rows, "aligned_nfe_1", metric),
            _metric(candidate_rows, "aligned_tokens_masked_nfe_1", metric),
            label=f"candidate-hard-mask:{metric}",
            contrast_count=HARD_MASK_CONTRASTS,
        )
        for metric in CLAIM_METRICS
    }
    hard_mask_ablation = {
        "effects": hard_mask_effects,
        "gate": _effect_gate(hard_mask_effects, primary_minimum=0.005),
    }

    # In AT-OFF both native and explicit hard-mask modes are exact-zero gates.
    # Their equality qualifies the intervention implementation itself.
    for index in range(evaluation.EXPECTED_VALIDATION_CLIPS):
        native = control_rows[(index, "aligned_nfe_1")]
        masked = control_rows[(index, "aligned_tokens_masked_nfe_1")]
        if (
            native["tensor_sha256"]["video_final_sha256"]
            != masked["tensor_sha256"]["video_final_sha256"]
            or native["tensor_sha256"]["decoded_final_sha256"]
            != masked["tensor_sha256"]["decoded_final_sha256"]
            or native["tensor_sha256"]["action_control_sha256"]
            != masked["tensor_sha256"]["action_control_sha256"]
            or native["tensor_sha256"]["wan_action_control_probe_sha256"]
            != masked["tensor_sha256"]["wan_action_control_probe_sha256"]
            or native["tensor_sha256"]["wan_context_sha256"]
            != masked["tensor_sha256"]["wan_context_sha256"]
            or native["tensor_sha256"]["action_context_delta_sha256"]
            != masked["tensor_sha256"]["action_context_delta_sha256"]
            or native.get("action_context_delta_abs_max") != 0.0
            or masked.get("action_context_delta_abs_max") != 0.0
            or native["metrics"] != masked["metrics"]
        ):
            raise ActionTokenAnalysisError(
                "control native and token-hard-mask endpoints are not identical"
            )

    quality_pass = bool(quality["1"]["gate"]["passed"])
    interaction_pass = all(
        value["gate"]["passed"] for value in interaction_attribution.values()
    )
    hard_mask_pass = bool(hard_mask_ablation["gate"]["passed"])
    control_diagnostic = _control_diagnostic(control_rows)
    candidate_diagnostic = _control_diagnostic(candidate_rows)
    control_context_diagnostic = _context_diagnostic(control_rows)
    candidate_context_diagnostic = _context_diagnostic(candidate_rows)
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
            "candidate_action_attribution_descriptive": (
                candidate_attribution_descriptive
            ),
            "action_specific_difference_in_differences": interaction_attribution,
            "trained_candidate_token_hard_mask_ablation": hard_mask_ablation,
            "wan_action_control_diagnostic": {
                "control": control_diagnostic,
                "candidate": candidate_diagnostic,
                "candidate_to_control_effective_rank_ratio": float(
                    candidate_diagnostic["aligned_centered_effective_rank"]
                    / max(control_diagnostic["aligned_centered_effective_rank"], 1e-30)
                ),
            },
            "wan_action_context_diagnostic": {
                "control": control_context_diagnostic,
                "candidate": candidate_context_diagnostic,
                "candidate_gate_opened": bool(
                    candidate_context_diagnostic["aligned_rms"] > 0.0
                ),
                "candidate_hard_mask_exact_zero": bool(
                    candidate_context_diagnostic["hard_mask_abs_max"] == 0.0
                ),
                "control_native_route_exact_zero": bool(
                    control_context_diagnostic["aligned_rms"] == 0.0
                ),
            },
            "latency": {
                "control": control_record["inventory"][
                    "latency_ms_batch_one_by_endpoint"
                ],
                "candidate": candidate_record["inventory"][
                    "latency_ms_batch_one_by_endpoint"
                ],
                "semantics": evaluation.LATENCY_SEMANTICS,
                "same_transformer_calls": True,
                "nfe_one_is_one_wan_call": True,
            },
            "runtime_verification_receipts": {
                "control": control_record["inventory"]["arm_artifacts"][
                    "runtime_verification_receipt"
                ],
                "candidate": candidate_record["inventory"]["arm_artifacts"][
                    "runtime_verification_receipt"
                ],
            },
            "learned_action_token_route_gate": {
                "control": control_record["inventory"]["arm_artifacts"][
                    "action_token"
                ],
                "candidate": candidate_record["inventory"]["arm_artifacts"][
                    "action_token"
                ],
            },
            "decision": {
                "passed": quality_pass and interaction_pass and hard_mask_pass,
                "nfe_1_quality_passed": quality_pass,
                "incremental_action_specificity_passed": interaction_pass,
                "trained_candidate_token_hard_mask_passed": hard_mask_pass,
                "rule": (
                    "NFE-1 candidate-vs-control temporal point/LB >=1% with >-1% "
                    "guardrails; candidate-vs-control aligned/diagnostic interaction "
                    "temporal point/LB >=0.5%; and trained-candidate native-vs-hard-"
                    "masked temporal point/LB >=0.5%, both with >-1% guardrails"
                ),
                "protected_test_authorized": False,
            },
        }
    )


def command_analyze(args: argparse.Namespace) -> int:
    result = analyze(args.control_inventory, args.candidate_inventory)
    output = args.output.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ActionTokenAnalysisError(
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
