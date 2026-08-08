#!/usr/bin/env python3
"""Deployable paired evaluation for the action-variation VPM screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import action_variation_screen as screen  # noqa: E402
from tools import two_clock_consistency_evaluate as base  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402

SCHEMA_VERSION = 1
KIND_ROW = "action_variation_validation_clip"
KIND_RANK = "action_variation_validation_rank"
KIND_INVENTORY = "action_variation_validation_inventory"
EXPECTED_WORLD_SIZE = screen.EXPECTED_WORLD_SIZE
EXPECTED_BATCH_SIZE_PER_RANK = 1
EXPECTED_VALIDATION_CLIPS = screen.EXPECTED_VALIDATION_CLIPS
VALIDATION_SAMPLE_ID_OFFSET = 2_000_000
ARMS = screen.ARMS
ARM_BY_CODE = screen.ARM_BY_CODE
ENDPOINTS = screen.ENDPOINTS
ENDPOINT_BY_CODE = screen.ENDPOINT_BY_CODE
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TENSOR_HASH_FIELDS = {
    "cached_rgb_input_sha256",
    "sampler_history_rgb_sha256",
    "cached_actions_input_sha256",
    "sampler_actions_sha256",
    "action_control_sha256",
    "wan_action_control_probe_sha256",
    "video_clean_scoring_sha256",
    "raw_ground_truth_sha256",
    "raw_history_last_sha256",
    "video_initial_noise_sha256",
    "auxiliary_initial_noise_sha256",
    "video_final_sha256",
    "decoded_final_sha256",
}


class ActionVariationEvaluationError(RuntimeError):
    """Deployment boundary, exact pairing, or artifact identity changed."""


def _validate_arm_plan(
    registration: Mapping[str, Any], arm: screen.Arm, run_dir: Path
) -> dict[str, Any]:
    path = Path(registration["output_root"]) / "arm_plans" / f"{arm.code.lower()}.json"
    plan = base._read_json(path, "arm execution plan")
    expected_identity = screen.arm_run_identity(registration, arm)
    expected_arm = {
        "code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "residual_enabled": arm.residual_enabled,
    }
    if (
        not screen.identity_valid(plan)
        or plan.get("kind") != "action_variation_arm_execution_plan"
        or plan.get("status") != "planned_before_arm_training_or_metrics"
        or plan.get("registration_identity_sha256") != registration["identity_sha256"]
        or plan.get("arm") != expected_arm
        or plan.get("run_identity_sha256") != expected_identity
        or plan.get("paths", {}).get("run_dir") != str(run_dir)
        or plan.get("training", {}).get("updates") != 200
        or plan.get("training", {}).get("wan_calls_per_update") != 1
        or plan.get("training", {}).get("same_schema_and_forward_compute") is not True
        or plan.get("evaluation", {}).get("batch_size_per_rank") != 1
        or plan.get("evaluation", {}).get("endpoints")
        != [asdict(endpoint) for endpoint in ENDPOINTS]
        or plan.get("evaluation", {}).get("protected_test_accessed") is not False
        or plan.get("evaluation", {}).get("future_or_clean_feature_used_at_sampling")
        is not False
        or plan.get("wandb")
        != {
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "group": None,
            "mode": "online",
            "id": expected_identity,
            "resume": "never",
        }
        or not screen.identity_valid(plan.get("resolved_config_contract", {}))
    ):
        raise ActionVariationEvaluationError("arm execution plan differs")
    return {
        "record": base._file_record(path),
        "identity_sha256": plan["identity_sha256"],
        "resolved_config_contract": plan["resolved_config_contract"],
    }


def _validate_config(
    config: Any, arm: screen.Arm, registration: Mapping[str, Any]
) -> None:
    variation = config.model.action_variation
    dual = config.model.dual_diffusion
    trainer = config.trainer.config
    expected_stats = registration["action_delta_stats"]["file"]
    expected_run_identity = screen.arm_run_identity(registration, arm)
    abc = config.dataset.datasets.ABC
    val_abc = config.val_dataset.datasets.ABC
    if (
        str(config.name) != arm.run_name
        or int(config.seed) != 1234
        or not str(config.model.get("_target_", "")).endswith(
            ".ActionVariationConditionedVPM"
        )
        or bool(variation.enabled) != arm.residual_enabled
        or Path(str(variation.stats_path)) != Path(expected_stats["path"])
        or str(variation.expected_stats_sha256) != expected_stats["sha256"]
        or int(variation.action_dim) != 157
        or int(variation.chunk_size) != 5
        or int(variation.latent_dim) != 64
        or int(variation.morph_dim) != 64
        or int(variation.hidden) != 256
        or int(variation.num_layers) != 3
        or float(variation.clip_value) != 8.0
        or int(variation.initialization_seed) != 20_260_808
        or not bool(dual.enabled)
        or not bool(dual.parameter_matched_control)
        or bool(dual.video_only_control)
        or bool(dual.condition_on_tf)
        or bool(dual.condition_on_tf_clock)
        or str(dual.condition_mode) != "off"
        or float(dual.tf_loss_weight) != 0.0
        or int(dual.tf_channels) != 64
        or list(dual.evaluation_nfe_steps) != [1, 2, 4]
        or int(dual.evaluation_noise_seed) != 20_260_726
        or int(config.model.num_history_frames) != 5
        or int(config.model.num_future_frames) != 8
        or int(config.dataset.padding_dim) != 157
        or not bool(config.dataset.infinite)
        or bool(config.dataset.img_augment)
        or bool(config.dataset.future_validity.enabled)
        or abc.expected_split != "train"
        or int(abc.expected_clip_count) != 512
        or abc.expected_actions_sha256 != base.TRAIN_ACTIONS_SHA256
        or val_abc.expected_split != "val"
        or int(val_abc.expected_clip_count) != 64
        or val_abc.expected_actions_sha256 != phase.EXPECTED_VALIDATION_ACTIONS_SHA256
        or int(config.data_loader.batch_size) != 1
        or int(config.val_data_loader[0].batch_size) != 2
        or len(config.viz_data_loader) != 0
        or int(trainer.max_iter) != 200
        or int(trainer.gradient_accumulation_steps) != 1
        or list(trainer.exclude_keys) != []
        or trainer.transition_handoff_path is not None
        or float(config.optimizer_factory.lr) != 1e-4
        or list(config.optimizer_factory.betas) != [0.9, 0.95]
        or int(config.lr_scheduler_factory.lr_lambda.warmup_steps) != 20
        or int(config.lr_scheduler_factory.lr_lambda.total_steps) != 200
        or config.wandb.entity != "zijiandu"
        or not bool(config.wandb.enabled)
        or config.wandb.project != "dual-video-diffusion-private"
        or config.wandb.group is not None
        or str(config.wandb.mode) != "online"
        or config.wandb.get("id") != expected_run_identity
        or config.wandb.get("resume") != "never"
    ):
        raise ActionVariationEvaluationError("resolved arm configuration differs")


def _validate_trace(
    run_dir: Path, arm: screen.Arm, registration: Mapping[str, Any]
) -> dict[str, Any]:
    trace_path = run_dir / "action_variation_training_trace.jsonl"
    complete_path = run_dir / "action_variation_training_trace_complete.json"
    try:
        rows = [
            json.loads(line) for line in trace_path.read_text().splitlines() if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionVariationEvaluationError("training trace is invalid") from exc
    if not rows or not isinstance(rows[0], dict):
        raise ActionVariationEvaluationError("training trace is empty")
    header = rows[0]
    expected_arm = "AV-DELTA" if arm.residual_enabled else "AV-CONT"
    if (
        header.get("kind") != "action_variation_training_trace_header"
        or header.get("arm") != expected_arm
        or header.get("residual_enabled") is not arm.residual_enabled
        or header.get("parent_snapshot_sha256") != screen.PARENT_SNAPSHOT_SHA256
        or header.get("stats_file_sha256")
        != registration["action_delta_stats"]["file"]["sha256"]
        or header.get("initial_effective_gate") != 0.0
        or header.get("same_schema_modules_and_forward_calls") is not True
        or header.get("parent_function_preserved_at_initialization") is not True
        or header.get("model_calls_per_update") != 1
        or header.get("protected_test_accessed") is not False
    ):
        raise ActionVariationEvaluationError("training trace header differs")
    train_events = [
        event
        for event in rows[1:]
        if isinstance(event.get("metrics"), Mapping)
        and "train_loss/loss" in event["metrics"]
    ]
    validation_events = [
        event
        for event in rows[1:]
        if isinstance(event.get("metrics"), Mapping)
        and any(str(key).startswith("val_loss/") for key in event["metrics"])
    ]
    if len(train_events) != 200 or len(validation_events) != 3:
        raise ActionVariationEvaluationError(
            "training trace must contain 200 train and three validation events"
        )
    expected_observations = list(range(8, 1601, 8))
    for update, (event, count) in enumerate(zip(train_events, expected_observations)):
        if (
            event.get("kind") != "action_variation_training_trace_event"
            or event.get("arm") != expected_arm
            or event.get("total_observations") != count
            or not isinstance(event.get("metrics"), Mapping)
            or event["metrics"].get("iteration") != update
            or event["metrics"].get("total_observations") != count
        ):
            raise ActionVariationEvaluationError("training trace event differs")
    complete = base._read_json(complete_path, "training trace completion")
    if (
        complete.get("completed_updates") != 200
        or complete.get("arm") != expected_arm
        or complete.get("rows") != len(rows)
        or complete.get("trace_sha256") != base._sha256(trace_path)
        or complete.get("protected_test_accessed") is not False
    ):
        raise ActionVariationEvaluationError("training trace completion differs")
    return {
        "trace": base._file_record(trace_path),
        "completion": base._file_record(complete_path),
        "header": header,
        "training_events": 200,
        "validation_events": 3,
    }


def _validate_wandb_completion_receipt(
    registration: Mapping[str, Any], arm: screen.Arm, run_dir: Path
) -> dict[str, Any]:
    """Require API-observed completion of the exact private run."""

    path = run_dir / "wandb_run_complete.json"
    receipt = base._read_json(path, "W&B completion receipt")
    expected_identity = screen.arm_run_identity(registration, arm)
    expected_arm = {
        "code": arm.code,
        "run_name": arm.run_name,
        "residual_enabled": arm.residual_enabled,
    }
    expected_training_completion = base._file_record(run_dir / "training_complete.json")
    wandb_observed = receipt.get("wandb")
    expected_url_fragment = (
        f"/zijiandu/dual-video-diffusion-private/runs/{expected_identity}"
    )
    if (
        not screen.identity_valid(receipt)
        or receipt.get("kind") != "action_variation_wandb_run_complete"
        or receipt.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or receipt.get("arm") != expected_arm
        or receipt.get("run_identity_sha256") != expected_identity
        or receipt.get("training_completion") != expected_training_completion
        or not isinstance(wandb_observed, Mapping)
        or set(wandb_observed)
        != {
            "id",
            "entity",
            "project",
            "group",
            "state",
            "run_identity_summary",
            "url",
        }
        or wandb_observed.get("id") != expected_identity
        or wandb_observed.get("entity") != "zijiandu"
        or wandb_observed.get("project") != "dual-video-diffusion-private"
        or wandb_observed.get("group") is not None
        or wandb_observed.get("state") != "finished"
        or wandb_observed.get("run_identity_summary") != expected_identity
        or not isinstance(wandb_observed.get("url"), str)
        or not wandb_observed["url"].startswith("https://")
        or expected_url_fragment not in wandb_observed["url"]
    ):
        raise ActionVariationEvaluationError("W&B completion receipt differs")
    return {
        "record": base._file_record(path),
        "identity_sha256": receipt["identity_sha256"],
        "training_completion": expected_training_completion,
        "wandb": dict(wandb_observed),
    }


def _load_model(
    registration: Mapping[str, Any], arm: screen.Arm, run_dir: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    tool_repo = Path(registration["tool_repository"]["path"])
    project_root = tool_repo / "projects" / "latent_action_models"
    shim = tool_repo / "tools" / "env" / "videox_shim"
    videox = Path(registration["runtime"]["videox_home"])
    for root in reversed((str(tool_repo), str(project_root), str(shim), str(videox))):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["WAN_DIR"] = registration["runtime"]["wan_dir"]
    os.environ["VIDEOX_HOME"] = registration["runtime"]["videox_home"]
    os.environ["ACTION_VARIATION_STATS"] = registration["action_delta_stats"]["file"][
        "path"
    ]
    os.environ["ACTION_VARIATION_STATS_SHA256"] = registration["action_delta_stats"][
        "file"
    ]["sha256"]
    config_path = run_dir / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    _validate_config(config, arm, registration)
    arm_plan = _validate_arm_plan(registration, arm, run_dir)
    observed_config_contract = screen.canonical_config_contract(config)
    if arm_plan["resolved_config_contract"] != observed_config_contract:
        raise ActionVariationEvaluationError(
            "saved resolved config differs from the pre-training arm contract"
        )
    wandb_completion = _validate_wandb_completion_receipt(registration, arm, run_dir)
    model = instantiate(config.model)
    snapshot_path = run_dir / "snapshot.pt"
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    expected_identity = screen.arm_run_identity(registration, arm)
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != 8
        or snapshot.get("_start_iter") != 200
        or snapshot.get("_total_observations") != 1600
        or snapshot.get("run_identity_sha256") != expected_identity
        or snapshot.get("wandb_run_id") != expected_identity
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise ActionVariationEvaluationError("trained arm snapshot metadata differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ActionVariationEvaluationError("strict trained-arm load failed")
    del snapshot
    model = model.to(device=device).eval()
    model._ensure_video_only_runtime_contract()
    if (
        bool(model.action_variation_enabled) != arm.residual_enabled
        or model.action_delta_residual.stats_file_sha256
        != registration["action_delta_stats"]["file"]["sha256"]
        or model.tf_condition_mode != "off"
        or model.condition_on_tf
        or bool(model.condition_on_tf_clock)
        or model.tf_loss_weight != 0.0
        or not model.parameter_matched_control
        or tuple(model.evaluation_nfe_steps) != (1, 2, 4)
    ):
        raise ActionVariationEvaluationError(
            "trained model violates deployable contract"
        )
    raw_action_gate = float(model.action_delta_residual.raw_gate.detach().float().cpu())
    effective_action_gate = float(
        model.action_delta_residual.effective_gate().detach().float().cpu()
    )
    if (
        not math.isfinite(raw_action_gate)
        or not math.isfinite(effective_action_gate)
        or abs(effective_action_gate) >= 1.0
        or (not arm.residual_enabled and effective_action_gate != 0.0)
    ):
        raise ActionVariationEvaluationError("trained action residual gate is invalid")
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if any(term in identity for term in ("vjepa", "teacher", "dino")):
            suspicious.append(identity)
    if suspicious or getattr(model, "time_frequency_transform", None) is not None:
        raise ActionVariationEvaluationError("online feature/teacher module is present")
    training = _validate_trace(run_dir, arm, registration)
    completion_path = run_dir / "training_complete.json"
    completion = base._read_json(completion_path, "training completion")
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != 200
        or completion.get("run_identity_sha256") != expected_identity
    ):
        raise ActionVariationEvaluationError("training completion differs")
    return model, {
        "snapshot": base._distributed_file_record(snapshot_path),
        "config": base._file_record(config_path),
        "training_trace": training,
        "training_completion": base._file_record(completion_path),
        "wandb_completion_receipt": wandb_completion,
        "arm_execution_plan": arm_plan,
        "run_identity_sha256": expected_identity,
        "wandb_run_id": expected_identity,
        "action_variation": {
            "enabled": arm.residual_enabled,
            "raw_gate": raw_action_gate,
            "effective_gate": effective_action_gate,
            "stats_file_sha256": model.action_delta_residual.stats_file_sha256,
            "stats_identity_sha256": model.action_delta_residual.stats_identity_sha256,
        },
    }


def _dataset(registration: Mapping[str, Any]) -> Any:
    validation = registration["validation"]
    return phase._RegisteredValidationInputs(
        rgb_path=Path(validation["arrays"]["rgb"]["path"]),
        actions_path=Path(validation["arrays"]["actions"]["path"]),
        descriptors=registration["validation_descriptors"],
        padding_dim=157,
    )


def _uint8_future(model: Any, video: Any, out_hw: tuple[int, int]) -> Any:
    import torch

    decoded = model.rgb_tokenizer.decode_temporal(video, out_hw=out_hw)
    return (
        ((decoded[:, :, -8:].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
    )


def _expected_rank_indexes(rank: int) -> list[int]:
    return list(range(rank, EXPECTED_VALIDATION_CLIPS, EXPECTED_WORLD_SIZE))


def _expected_action_donor_index(clip_index: int) -> int:
    return (
        clip_index + EXPECTED_WORLD_SIZE
        if (clip_index // EXPECTED_WORLD_SIZE) % 2 == 0
        else clip_index - EXPECTED_WORLD_SIZE
    )


def _expected_rank_transformer_calls() -> int:
    return (
        len(_expected_rank_indexes(0))
        // EXPECTED_BATCH_SIZE_PER_RANK
        * sum(endpoint.nfe for endpoint in ENDPOINTS)
    )


def _rows_for_endpoint(
    *,
    arm: screen.Arm,
    endpoint: screen.Endpoint,
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    decoded: Any,
    scoring: Mapping[str, Any],
    indexes: Sequence[int],
    clip_ids: Sequence[str],
    sampling_ids: Any,
    history: Any,
    action_input: Any,
    donors: Sequence[int | None],
    calls: int,
    prepare_ms: float,
    probe_ms: float,
    trajectory_ms: float,
    decode_ms: float,
    registration: Mapping[str, Any],
    arm_artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    final = result["video"].detach().cpu().to(torch.float16)
    clean = scoring["video_clean"]
    history_tokens = int(prepared["history_frames"])
    if calls != endpoint.nfe or result["calls"] != endpoint.nfe:
        raise ActionVariationEvaluationError("actual transformer calls differ from NFE")
    latent_nmse = phase._per_sample_nmse(final, clean, history_tokens)
    latent_delta = phase._per_sample_future_delta_nmse(final, clean, history_tokens)
    decoded_metrics = phase._per_sample_decoded(
        decoded, scoring["ground_truth"], scoring["history_last"]
    )
    hashes = {
        "cached_rgb_input_sha256": scoring["rgb_hashes"],
        "sampler_history_rgb_sha256": phase._slice_hashes(history),
        "cached_actions_input_sha256": scoring["actions_hashes"],
        "sampler_actions_sha256": phase._slice_hashes(action_input),
        "action_control_sha256": phase._slice_hashes(prepared["z_control"]),
        "wan_action_control_probe_sha256": phase._slice_hashes(
            prepared["wan_action_control_probe"]
        ),
        "video_clean_scoring_sha256": phase._slice_hashes(clean),
        "raw_ground_truth_sha256": phase._slice_hashes(scoring["ground_truth"]),
        "raw_history_last_sha256": phase._slice_hashes(scoring["history_last"]),
        "video_initial_noise_sha256": phase._slice_hashes(
            prepared["initial_video"].detach().cpu().to(torch.float16)
        ),
        "auxiliary_initial_noise_sha256": phase._slice_hashes(
            prepared["initial_tf"].detach().cpu().to(torch.float16)
        ),
        "video_final_sha256": phase._slice_hashes(final),
        "decoded_final_sha256": phase._slice_hashes(decoded),
    }
    sampling_values = [int(value) for value in sampling_ids.detach().cpu().tolist()]
    control_values = (
        prepared["wan_action_control_probe"].detach().float().cpu().flatten(1).tolist()
    )
    rows = []
    for offset, (clip_index, clip_id) in enumerate(zip(indexes, clip_ids)):
        rows.append(
            screen.identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": KIND_ROW,
                    "registration_identity_sha256": registration["identity_sha256"],
                    "tool_git_commit": registration["tool_repository"]["git_commit"],
                    "arm": asdict(arm),
                    "arm_snapshot": arm_artifacts["snapshot"],
                    "evaluation_split": "validation",
                    "protected_test_accessed": False,
                    "clip_index": int(clip_index),
                    "clip_id": str(clip_id),
                    "sampling_id": sampling_values[offset],
                    "endpoint": asdict(endpoint),
                    "action_donor_sampling_id": donors[offset],
                    "clean_future_rgb_passed_to_sampler": False,
                    "clean_video_latent_passed_to_sampler": False,
                    "clean_auxiliary_passed_to_sampler": False,
                    "target_cache_array_opened": False,
                    "online_feature_or_teacher_call_count": 0,
                    "scoring_constructed_after_all_sampling": True,
                    "history_rgb_frames": 5,
                    "future_rgb_frames": 8,
                    "history_video_latent_tokens": history_tokens,
                    "future_video_latent_tokens": int(final.shape[2] - history_tokens),
                    "actual_transformer_call_count": calls,
                    "declared_nfe": endpoint.nfe,
                    "sampler_action_abs_max": float(
                        action_input[offset].detach().float().abs().max().cpu()
                    ),
                    "wan_action_control_probe": [
                        float(value) for value in control_values[offset]
                    ],
                    "latency_ms_per_local_batch": {
                        "prepare_history_and_action": prepare_ms,
                        "wan_trajectory": trajectory_ms,
                        "decode": decode_ms,
                        "total": prepare_ms + trajectory_ms + decode_ms,
                    },
                    "evaluator_only_action_control_probe_ms": probe_ms,
                    "metrics": {
                        "video_future_nmse": latent_nmse[offset],
                        "video_future_temporal_delta_nmse": latent_delta[offset],
                        "decoded_mse_unit_range": decoded_metrics[
                            "decoded_mse_unit_range"
                        ][offset],
                        "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][offset],
                        "decoded_temporal_difference_mse_unit_range": decoded_metrics[
                            "decoded_temporal_difference_mse_unit_range"
                        ][offset],
                    },
                    "tensor_sha256": {
                        key: values[offset] for key, values in hashes.items()
                    },
                }
            )
        )
    return rows


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: screen.Arm,
    registration: Mapping[str, Any],
    expected_snapshot: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        (index, endpoint.code)
        for index in range(EXPECTED_VALIDATION_CLIPS)
        for endpoint in ENDPOINTS
    }
    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    observed_snapshot_records = set()
    descriptors = registration["validation_descriptors"]
    for row in rows:
        endpoint_payload = row.get("endpoint")
        code = (
            endpoint_payload.get("code")
            if isinstance(endpoint_payload, Mapping)
            else None
        )
        endpoint = ENDPOINT_BY_CODE.get(code)
        index = row.get("clip_index")
        metrics = row.get("metrics")
        latency = row.get("latency_ms_per_local_batch")
        tensor_hashes = row.get("tensor_sha256")
        action_abs_max = row.get("sampler_action_abs_max")
        if (
            not screen.identity_valid(row)
            or row.get("kind") != KIND_ROW
            or row.get("arm") != asdict(arm)
            or row.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or row.get("tool_git_commit")
            != registration["tool_repository"]["git_commit"]
            or row.get("evaluation_split") != "validation"
            or row.get("protected_test_accessed") is not False
            or row.get("clean_future_rgb_passed_to_sampler") is not False
            or row.get("clean_video_latent_passed_to_sampler") is not False
            or row.get("clean_auxiliary_passed_to_sampler") is not False
            or row.get("target_cache_array_opened") is not False
            or row.get("online_feature_or_teacher_call_count") != 0
            or row.get("scoring_constructed_after_all_sampling") is not True
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < EXPECTED_VALIDATION_CLIPS
            or endpoint is None
            or dict(endpoint_payload) != asdict(endpoint)
            or row.get("clip_id") != descriptors[index]["clip_id"]
            or row.get("sampling_id") != VALIDATION_SAMPLE_ID_OFFSET + index
            or row.get("history_rgb_frames") != 5
            or row.get("future_rgb_frames") != 8
            or row.get("history_video_latent_tokens") != 2
            or row.get("future_video_latent_tokens") != 2
            or row.get("actual_transformer_call_count") != endpoint.nfe
            or row.get("declared_nfe") != endpoint.nfe
            or not isinstance(metrics, Mapping)
            or not isinstance(latency, Mapping)
            or set(latency)
            != {
                "prepare_history_and_action",
                "wan_trajectory",
                "decode",
                "total",
            }
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in latency.values()
            )
            or latency["total"]
            != latency["prepare_history_and_action"]
            + latency["wan_trajectory"]
            + latency["decode"]
            or isinstance(action_abs_max, bool)
            or not isinstance(action_abs_max, (int, float))
            or not math.isfinite(action_abs_max)
            or action_abs_max < 0
            or not isinstance(tensor_hashes, Mapping)
            or set(tensor_hashes) != TENSOR_HASH_FIELDS
            or any(
                not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
                for value in tensor_hashes.values()
            )
            or not isinstance(row.get("wan_action_control_probe"), list)
            or len(row["wan_action_control_probe"]) != 32
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in row["wan_action_control_probe"]
            )
            or not isinstance(
                row.get("evaluator_only_action_control_probe_ms"), (int, float)
            )
            or not math.isfinite(row["evaluator_only_action_control_probe_ms"])
            or row["evaluator_only_action_control_probe_ms"] < 0
        ):
            raise ActionVariationEvaluationError("validation row violates protocol")
        for metric in (
            "video_future_nmse",
            "video_future_temporal_delta_nmse",
            "decoded_mse_unit_range",
            "decoded_psnr_db",
            "decoded_temporal_difference_mse_unit_range",
        ):
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ActionVariationEvaluationError(f"invalid metric {metric}")
        for metric in (
            "video_future_nmse",
            "video_future_temporal_delta_nmse",
            "decoded_mse_unit_range",
            "decoded_temporal_difference_mse_unit_range",
        ):
            if metrics[metric] < 0:
                raise ActionVariationEvaluationError(f"negative metric {metric}")
        key = (index, endpoint.code)
        if key in observed:
            raise ActionVariationEvaluationError("duplicate validation row")
        observed[key] = row
        snapshot = row.get("arm_snapshot")
        if not isinstance(snapshot, Mapping) or not isinstance(
            snapshot.get("sha256"), str
        ):
            raise ActionVariationEvaluationError("validation row lacks arm snapshot")
        observed_snapshot_records.add(
            (snapshot.get("path"), snapshot.get("sha256"), snapshot.get("bytes"))
        )
    if set(observed) != expected:
        raise ActionVariationEvaluationError("validation inventory is incomplete")
    if len(observed_snapshot_records) != 1:
        raise ActionVariationEvaluationError(
            "validation rows do not share one snapshot"
        )
    if expected_snapshot is not None and observed_snapshot_records != {
        (
            expected_snapshot.get("path"),
            expected_snapshot.get("sha256"),
            expected_snapshot.get("bytes"),
        )
    }:
        raise ActionVariationEvaluationError(
            "validation rows are not bound to the loaded arm snapshot"
        )
    invariant = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "auxiliary_initial_noise_sha256",
    )
    for index in range(EXPECTED_VALIDATION_CLIPS):
        clip_rows = [observed[(index, endpoint.code)] for endpoint in ENDPOINTS]
        reference = clip_rows[0]["tensor_sha256"]
        if any(
            row["tensor_sha256"].get(field) != reference.get(field)
            for row in clip_rows[1:]
            for field in invariant
        ):
            raise ActionVariationEvaluationError(
                "paired target/noise changed within arm"
            )
        for nfe in screen.NFE_GRID:
            aligned = observed[(index, f"aligned_nfe_{nfe}")]
            if (
                aligned["tensor_sha256"]["sampler_actions_sha256"]
                != reference["cached_actions_input_sha256"]
                or aligned.get("action_donor_sampling_id") is not None
                or aligned["tensor_sha256"]["action_control_sha256"]
                != clip_rows[0]["tensor_sha256"]["action_control_sha256"]
                or aligned["tensor_sha256"]["wan_action_control_probe_sha256"]
                != clip_rows[0]["tensor_sha256"]["wan_action_control_probe_sha256"]
            ):
                raise ActionVariationEvaluationError("aligned actions differ from clip")
        zero = observed[(index, "zero_nfe_1")]
        if (
            zero.get("action_donor_sampling_id") is not None
            or zero.get("sampler_action_abs_max") != 0.0
        ):
            raise ActionVariationEvaluationError("zero endpoint has an action donor")
        shuffled = observed[(index, "global_shuffled_nfe_1")]
        donor_index = _expected_action_donor_index(index)
        donor = observed[(donor_index, "aligned_nfe_1")]
        if (
            shuffled.get("action_donor_sampling_id")
            != VALIDATION_SAMPLE_ID_OFFSET + donor_index
            or shuffled["tensor_sha256"]["sampler_actions_sha256"]
            != donor["tensor_sha256"]["cached_actions_input_sha256"]
            or descriptors[index]["episode_dir"]
            == descriptors[donor_index]["episode_dir"]
        ):
            raise ActionVariationEvaluationError("global shuffled action donor differs")
        masked = observed[(index, "aligned_residual_masked_nfe_1")]
        if (
            masked["tensor_sha256"]["sampler_actions_sha256"]
            != reference["cached_actions_input_sha256"]
            or masked.get("action_donor_sampling_id") is not None
            or masked["endpoint"].get("residual_mode") != "hard_mask"
        ):
            raise ActionVariationEvaluationError(
                "trained residual hard-mask endpoint differs"
            )


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (
        dist.get_world_size() != EXPECTED_WORLD_SIZE
        or args.batch_size_per_rank != EXPECTED_BATCH_SIZE_PER_RANK
    ):
        raise ActionVariationEvaluationError(
            "evaluation requires eight ranks, local batch one"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise ActionVariationEvaluationError("evaluation requires B200 GPUs")
    registration = screen.validate_registration(args.registration)
    if rank == 0:
        screen.revalidate_execution_environment(registration)
        screen.revalidate_registered_inputs(registration)
    dist.barrier()
    arm = ARM_BY_CODE.get(args.arm)
    if arm is None:
        raise ActionVariationEvaluationError("unknown arm")
    output = Path(registration["output_root"]) / "evaluation" / arm.code.lower()
    if args.output_dir.expanduser().absolute() != output:
        raise ActionVariationEvaluationError("evaluation output path differs")
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise ActionVariationEvaluationError("fresh evaluation output exists")
        output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    run_dir = Path(registration["output_root"]) / "training" / arm.run_name
    model, arm_artifacts = _load_model(
        registration, arm, run_dir.resolve(strict=True), device
    )
    dataset = _dataset(registration)
    assigned = _expected_rank_indexes(rank)
    rows: list[dict[str, Any]] = []
    hook_calls = 0

    # One unmeasured history/action -> one-call -> decode rollout warms CUDA,
    # VAE, scheduler, and allocator paths.  The accounting hook is intentionally
    # installed only after this warmup, so declared endpoint calls stay exact.
    warm_sample = phase._move_batch([dataset[assigned[0]]], device)
    warm_history = warm_sample["rgb"][:, :5].clone()
    warm_ids = warm_sample["clip_index"] + VALIDATION_SAMPLE_ID_OFFSET
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
    ):
        warm_prepared = phase._prepare_deployable_rollout(
            model,
            warm_history,
            warm_sample["actions"],
            warm_sample["morphology_index"],
            warm_ids,
        )
        warm_result = phase._run_trajectory(model, warm_prepared, steps=1)
        _uint8_future(
            model,
            warm_result["video"],
            (int(warm_history.shape[-2]), int(warm_history.shape[-1])),
        )
    torch.cuda.synchronize(device)
    del warm_sample, warm_history, warm_ids, warm_prepared, warm_result

    def count_calls(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    hook = model.forward_model.register_forward_hook(count_calls)
    try:
        for clip_index in assigned:
            indexes = [clip_index]
            samples = [dataset[index] for index in indexes]
            batch = phase._move_batch(samples, device)
            history = batch["rgb"][:, :5].clone()
            sampling_ids = batch["clip_index"] + VALIDATION_SAMPLE_ID_OFFSET
            donor_index = _expected_action_donor_index(clip_index)
            donor_actions = dataset.action_only(donor_index).unsqueeze(0).to(device)
            sources = {
                "aligned": (batch["actions"], [None]),
                "zero": (torch.zeros_like(batch["actions"]), [None]),
                "global_shuffled": (
                    donor_actions,
                    [VALIDATION_SAMPLE_ID_OFFSET + donor_index],
                ),
            }
            prepared_by_source: dict[tuple[str, str], tuple[Any, float, float]] = {}
            completed = []
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
            ):
                preparation_keys = list(
                    dict.fromkeys(
                        (endpoint.action_source, endpoint.residual_mode)
                        for endpoint in ENDPOINTS
                    )
                )
                for source, residual_mode in preparation_keys:
                    action_input, _donors = sources[source]
                    mask_context = (
                        model.action_delta_residual.runtime_hard_mask()
                        if residual_mode == "hard_mask"
                        else nullcontext()
                    )
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter_ns()
                    with mask_context:
                        prepared = phase._prepare_deployable_rollout(
                            model,
                            history,
                            action_input,
                            batch["morphology_index"],
                            sampling_ids,
                        )
                    torch.cuda.synchronize(device)
                    prepare_ms = (time.perf_counter_ns() - begin) / 1e6
                    # Exact 32-value state seen by Wan after the trained
                    # ActionToControl projection (two future latent rows x 16
                    # channels). This read-only probe is not fed back to the
                    # sampler and adds no transformer call.
                    begin = time.perf_counter_ns()
                    history_tokens = int(prepared["history_frames"])
                    wan_control = model.forward_model.action_to_control(
                        prepared["z_control"], 1, 1
                    )[:, :, history_tokens:, 0, 0]
                    if tuple(wan_control.shape[1:]) != (16, 2):
                        raise ActionVariationEvaluationError(
                            "Wan action-control probe shape changed"
                        )
                    prepared = {
                        **prepared,
                        "wan_action_control_probe": wan_control,
                    }
                    torch.cuda.synchronize(device)
                    probe_ms = (time.perf_counter_ns() - begin) / 1e6
                    prepared_by_source[(source, residual_mode)] = (
                        prepared,
                        prepare_ms,
                        probe_ms,
                    )
                for endpoint in ENDPOINTS:
                    action_input, donors = sources[endpoint.action_source]
                    prepared, prepare_ms, probe_ms = prepared_by_source[
                        (endpoint.action_source, endpoint.residual_mode)
                    ]
                    before_calls = hook_calls
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter_ns()
                    result = phase._run_trajectory(model, prepared, steps=endpoint.nfe)
                    torch.cuda.synchronize(device)
                    trajectory_ms = (time.perf_counter_ns() - begin) / 1e6
                    calls = hook_calls - before_calls
                    begin = time.perf_counter_ns()
                    decoded = _uint8_future(
                        model,
                        result["video"],
                        (int(history.shape[-2]), int(history.shape[-1])),
                    )
                    torch.cuda.synchronize(device)
                    decode_ms = (time.perf_counter_ns() - begin) / 1e6
                    completed.append(
                        (
                            endpoint,
                            result,
                            prepared,
                            decoded,
                            action_input,
                            donors,
                            calls,
                            prepare_ms,
                            probe_ms,
                            trajectory_ms,
                            decode_ms,
                        )
                    )
            scoring = phase._scoring_targets(model, batch)
            clip_ids = [
                registration["validation_descriptors"][index]["clip_id"]
                for index in indexes
            ]
            for values in completed:
                (
                    endpoint,
                    result,
                    prepared,
                    decoded,
                    action_input,
                    donors,
                    calls,
                    prepare_ms,
                    probe_ms,
                    trajectory_ms,
                    decode_ms,
                ) = values
                rows.extend(
                    _rows_for_endpoint(
                        arm=arm,
                        endpoint=endpoint,
                        result=result,
                        prepared=prepared,
                        decoded=decoded,
                        scoring=scoring,
                        indexes=indexes,
                        clip_ids=clip_ids,
                        sampling_ids=sampling_ids,
                        history=history,
                        action_input=action_input,
                        donors=donors,
                        calls=calls,
                        prepare_ms=prepare_ms,
                        probe_ms=probe_ms,
                        trajectory_ms=trajectory_ms,
                        decode_ms=decode_ms,
                        registration=registration,
                        arm_artifacts=arm_artifacts,
                    )
                )
    finally:
        hook.remove()
    row_path = output / f"rank_{rank:03d}.jsonl"
    content = b"".join(_canonical_row(row) for row in rows)
    base._exclusive_bytes(row_path, content)
    receipt = screen.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RANK,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": asdict(arm),
            "rank": rank,
            "world_size": EXPECTED_WORLD_SIZE,
            "batch_size_per_rank": EXPECTED_BATCH_SIZE_PER_RANK,
            "unmeasured_warmup_transformer_calls": 1,
            "assigned_clip_indexes": assigned,
            "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
            "actual_transformer_call_count": hook_calls,
            "rows": {**base._file_record(row_path), "count": len(rows)},
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        }
    )
    base._exclusive_json(output / f"rank_{rank:03d}_complete.json", receipt)
    dist.barrier()
    if rank == 0:
        all_rows = []
        rank_records = []
        total_calls = 0
        for value in range(EXPECTED_WORLD_SIZE):
            rank_path = output / f"rank_{value:03d}.jsonl"
            rank_receipt_path = output / f"rank_{value:03d}_complete.json"
            rank_receipt = base._read_json(rank_receipt_path, "rank receipt")
            if (
                not screen.identity_valid(rank_receipt)
                or rank_receipt.get("kind") != KIND_RANK
                or rank_receipt.get("rank") != value
                or rank_receipt.get("arm") != asdict(arm)
                or rank_receipt.get("registration_identity_sha256")
                != registration["identity_sha256"]
                or rank_receipt.get("batch_size_per_rank")
                != EXPECTED_BATCH_SIZE_PER_RANK
                or rank_receipt.get("unmeasured_warmup_transformer_calls") != 1
                or rank_receipt.get("actual_transformer_call_count")
                != _expected_rank_transformer_calls()
                or rank_receipt.get("rows", {}).get("sha256") != base._sha256(rank_path)
            ):
                raise ActionVariationEvaluationError("rank receipt differs")
            rank_records.append(base._file_record(rank_receipt_path))
            total_calls += int(rank_receipt["actual_transformer_call_count"])
            all_rows.extend(
                json.loads(line) for line in rank_path.read_text().splitlines() if line
            )
        _validate_rows(
            all_rows, arm, registration, expected_snapshot=arm_artifacts["snapshot"]
        )
        latency_by_endpoint = {}
        for endpoint in ENDPOINTS:
            endpoint_rows = [
                row for row in all_rows if row["endpoint"]["code"] == endpoint.code
            ]
            latency_by_endpoint[endpoint.code] = {}
            for stage in (
                "prepare_history_and_action",
                "wan_trajectory",
                "decode",
                "total",
            ):
                values = sorted(
                    float(row["latency_ms_per_local_batch"][stage])
                    for row in endpoint_rows
                )
                latency_by_endpoint[endpoint.code][stage] = {
                    "median": float(values[len(values) // 2]),
                    "p95": float(values[math.ceil(0.95 * len(values)) - 1]),
                }
        inventory = screen.identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_INVENTORY,
                "registration_identity_sha256": registration["identity_sha256"],
                "registration": base._file_record(
                    args.registration.resolve(strict=True)
                ),
                "arm": asdict(arm),
                "evaluation_split": "validation",
                "validation_clips": EXPECTED_VALIDATION_CLIPS,
                "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
                "row_count": len(all_rows),
                "actual_transformer_call_count": total_calls,
                "batch_size_per_rank": EXPECTED_BATCH_SIZE_PER_RANK,
                "latency_warmup": {
                    "rollouts_per_rank": 1,
                    "nfe": 1,
                    "excluded_from_endpoint_call_accounting": True,
                },
                "paired_inputs_and_noise_within_arm": True,
                "arm_artifacts": arm_artifacts,
                "latency_ms_batch_one_by_endpoint": latency_by_endpoint,
                "rank_receipts": rank_records,
                "protected_test_accessed": False,
                "target_cache_array_opened": False,
                "future_or_clean_feature_used_at_sampling": False,
            }
        )
        base._exclusive_json(output / "inventory.json", inventory)
    dist.barrier()
    return 0


def _canonical_row(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size-per-rank", type=int, default=1)
    parser.set_defaults(func=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
