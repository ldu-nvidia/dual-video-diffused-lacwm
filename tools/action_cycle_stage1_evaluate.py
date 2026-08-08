#!/usr/bin/env python3
"""Target-free paired validation for Action Cycle Stage 1.

Every sampler receives exactly five observed RGB frames, requested actions,
morphology, and deterministic initial noise.  The deployment model is rebuilt
with ``action_cycle.mode=deploy`` and no critic path or digest.  Clean future
RGB/latents are constructed only after all endpoints in a batch finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import action_cycle_recoverability as stage0  # noqa: E402
from tools import action_cycle_stage1 as protocol  # noqa: E402
from tools import lamo_motion_drift_evaluate as common  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


SCHEMA_VERSION = 1
KIND_ROW = "action_cycle_stage1_validation_clip_v1"
KIND_RANK = "action_cycle_stage1_validation_rank_v1"
KIND_INVENTORY = "action_cycle_stage1_validation_inventory_v1"
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE_PER_RANK = 2
VALIDATION_SAMPLE_ID_OFFSET = 3_000_000


class ActionCycleEvaluationError(RuntimeError):
    """A deployment boundary, trained arm, or paired row changed."""


def _expected_rank_indexes(rank: int) -> list[int]:
    return list(range(rank, protocol.EXPECTED_VALIDATION_CLIPS, EXPECTED_WORLD_SIZE))


def action_shuffle(registration: Mapping[str, Any]) -> list[int]:
    descriptors = registration.get("validation_descriptors")
    if not isinstance(descriptors, Sequence) or len(descriptors) != 64:
        raise ActionCycleEvaluationError("validation descriptors are incomplete")
    episodes = [str(row["episode_dir"]) for row in descriptors]
    permutation = stage0.episode_disjoint_permutation(
        episodes, seed=protocol.SHUFFLE_SEED
    )
    values = [int(value) for value in permutation.tolist()]
    if sorted(values) != list(range(64)) or any(
        episodes[index] == episodes[donor] for index, donor in enumerate(values)
    ):
        raise ActionCycleEvaluationError("action shuffle is not episode-disjoint/bijective")
    return values


def _validate_config(config: Any, arm: protocol.Arm, registration: Mapping[str, Any]) -> None:
    dual = config.model.dual_diffusion
    cycle = config.model.action_cycle
    trainer = config.trainer.config
    optimizer = config.optimizer_factory
    scheduler = config.lr_scheduler_factory.lr_lambda
    train_abc = config.dataset.datasets.ABC
    val_abc = config.val_dataset.datasets.ABC
    train_manifest = registration["training"]["manifest"]["path"]
    train_metadata = registration["training"]["cache_metadata"]["path"]
    critic = registration["stage0"]["critic_bundle"]
    if (
        str(config.name) != arm.run_name
        or int(config.seed) != 1234
        or not str(config.model.get("_target_", "")).endswith(
            ".ActionCycleStage1VPM"
        )
        or str(cycle.mode) != ("off" if arm.code == "AC-OFF" else "on")
        or float(cycle.loss_weight) != arm.loss_weight
        or list(cycle.transitions) != [1, 2]
        or str(cycle.critic_path) != critic["path"]
        or str(cycle.critic_sha256) != critic["sha256"]
        or str(cycle.stage0_registration_identity_sha256)
        != registration["stage0"]["registration_identity_sha256"]
        or int(config.model.num_history_frames) != 5
        or int(config.model.num_future_frames) != 8
        or not bool(dual.enabled)
        or not bool(dual.parameter_matched_control)
        or bool(dual.condition_on_tf)
        or bool(dual.condition_on_tf_clock)
        or not bool(dual.head_condition_on_tf_clock)
        or str(dual.condition_mode) != "off"
        or float(dual.tf_loss_weight) != 0.0
        or list(dual.evaluation_nfe_steps) != [1, 4]
        or int(dual.evaluation_noise_seed) != 20_260_726
        or list(dual.evaluation_condition_sources) != ["autonomous"]
        or int(config.dataset.padding_dim) != 157
        or int(config.dataset.transform.sample_size) != 13
        or int(config.dataset.transform.chunk_size) != 5
        or not bool(config.dataset.infinite)
        or bool(config.dataset.img_augment)
        or bool(config.dataset.future_validity.enabled)
        or str(train_abc.clip_manifest) != train_manifest
        or str(train_abc.cache_metadata) != train_metadata
        or str(val_abc.clip_manifest) != train_manifest
        or str(val_abc.cache_metadata) != train_metadata
        or len(config.val_data_loader) != 1
        or int(config.val_data_loader[0].batch_size) != 2
        or len(config.viz_data_loader) != 0
        or int(config.data_loader.batch_size) != 1
        or int(trainer.max_iter) != 200
        or int(trainer.gradient_accumulation_steps) != 1
        or list(trainer.exclude_keys) != []
        or trainer.transition_handoff_path is not None
        or int(trainer.logging.log_every) != 1
        or int(trainer.validation.val_every) != 100
        or bool(trainer.validation.save_best)
        or float(trainer.gradient_clipping.max_norm) != 1.0
        or float(optimizer.lr) != 1.0e-4
        or list(optimizer.betas) != [0.9, 0.95]
        or int(scheduler.warmup_steps) != 20
        or int(scheduler.total_steps) != 200
        or float(scheduler.final_learning_rate) != 1.0e-6
        or config.wandb.entity != protocol.EXPECTED_ENTITY
        or config.wandb.project != protocol.EXPECTED_PROJECT
        or config.wandb.group is not None
    ):
        raise ActionCycleEvaluationError("resolved arm configuration differs")


def _validate_trace(
    run_dir: Path, arm: protocol.Arm, registration: Mapping[str, Any]
) -> dict[str, Any]:
    trace = run_dir / "paired_training_trace.jsonl"
    complete_path = run_dir / "paired_training_trace_complete.json"
    trace_record = protocol.file_record(trace)
    complete = protocol.read_json(complete_path, "training trace completion")
    with trace.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    header = json.loads(lines[0]) if lines else {}
    if (
        complete.get("kind") != "action_cycle_stage1_training_trace_complete"
        or complete.get("arm") != arm.code
        or complete.get("completed_updates") != 200
        or complete.get("trace_sha256") != trace_record["sha256"]
        or complete.get("rows") != len(lines)
        or complete.get("protected_test_accessed") is not False
        or header.get("kind") != "action_cycle_stage1_training_trace_header"
        or header.get("arm") != arm.code
        or header.get("action_cycle_loss_weight") != arm.loss_weight
        or header.get("critic_bundle_sha256")
        != registration["stage0"]["critic_bundle"]["sha256"]
        or header.get("stage0_registration_identity_sha256")
        != registration["stage0"]["registration_identity_sha256"]
        or header.get("parent_snapshot_sha256") != protocol.PARENT_SNAPSHOT_SHA256
        or header.get("parent_run_identity_sha256")
        != protocol.PARENT_RUN_IDENTITY_SHA256
        or header.get("continuation_updates") != 200
        or header.get("run_identity_sha256")
        != protocol.arm_run_identity(registration, arm)
        or header.get("critic_parameters_in_optimizer") != 0
        or header.get("protected_test_accessed") is not False
    ):
        raise ActionCycleEvaluationError("paired training trace differs")
    return {
        "trace": trace_record,
        "completion": protocol.file_record(complete_path),
        "header": header,
    }


def _validate_arm_plan(
    registration: Mapping[str, Any], arm: protocol.Arm, run_dir: Path
) -> dict[str, Any]:
    path = Path(registration["output_root"]) / "arm_plans" / f"{arm.code.lower()}.json"
    plan = protocol.read_json(path, "arm execution plan")
    if (
        not protocol.identity_valid(plan)
        or plan.get("kind") != "action_cycle_stage1_arm_execution_plan_v1"
        or plan.get("status") != "planned_before_arm_training_or_metrics"
        or plan.get("registration_identity_sha256") != registration["identity_sha256"]
        or plan.get("arm") != asdict(arm)
        or plan.get("run_identity_sha256")
        != protocol.arm_run_identity(registration, arm)
        or plan.get("paths", {}).get("run_dir") != str(run_dir)
        or plan.get("training", {}).get("updates") != 200
        or plan.get("training", {}).get("same_data_noise_calls") is not True
        or plan.get("evaluation", {}).get("critic_loaded_or_called") is not False
        or plan.get("evaluation", {}).get("protected_test_accessed") is not False
        or plan.get("input_revalidation", {}).get(
            "all_selected_registered_inputs_rehashed"
        )
        is not True
    ):
        raise ActionCycleEvaluationError("arm execution plan differs")
    return {"record": protocol.file_record(path), "identity_sha256": plan["identity_sha256"]}


def _load_deploy_model(
    registration: Mapping[str, Any], arm: protocol.Arm, run_dir: Path, device: Any
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    repo = Path(registration["tool_repository"]["path"])
    project = repo / "projects" / "latent_action_models"
    shim = repo / "tools" / "env" / "videox_shim"
    videox = Path(registration["runtime"]["videox_home"])
    for root in reversed((str(repo), str(project), str(shim), str(videox))):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["WAN_DIR"] = registration["runtime"]["wan_dir"]
    os.environ["VIDEOX_HOME"] = registration["runtime"]["videox_home"]
    # Config interpolation must resolve exactly as it did during training.
    os.environ["ACTION_CYCLE_CRITIC_BUNDLE"] = registration["stage0"][
        "critic_bundle"
    ]["path"]
    os.environ["ACTION_CYCLE_CRITIC_SHA256"] = registration["stage0"][
        "critic_bundle"
    ]["sha256"]
    os.environ["ACTION_CYCLE_STAGE0_IDENTITY"] = registration["stage0"][
        "registration_identity_sha256"
    ]
    config_path = run_dir / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    _validate_config(config, arm, registration)
    with open_dict(config.model.action_cycle):
        config.model.action_cycle.mode = "deploy"
        config.model.action_cycle.loss_weight = 0.0
        config.model.action_cycle.critic_path = None
        config.model.action_cycle.critic_sha256 = ""
        config.model.action_cycle.stage0_registration_identity_sha256 = ""
    model = instantiate(config.model)
    if getattr(model, "action_cycle_critic", "sentinel") is not None:
        raise ActionCycleEvaluationError("deployment model loaded the training critic")
    model.assert_deployable()
    snapshot_path = run_dir / "snapshot.pt"
    snapshot_record = common._distributed_file_record(snapshot_path)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True, mmap=True)
    expected_identity = protocol.arm_run_identity(registration, arm)
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("_start_iter") != 200
        or snapshot.get("_total_observations") != 1600
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("run_identity_sha256") != expected_identity
        or not isinstance(snapshot.get("rank_states"), Sequence)
        or len(snapshot["rank_states"]) != EXPECTED_WORLD_SIZE
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
        or any("critic" in str(key).lower() for key in snapshot["model"])
    ):
        raise ActionCycleEvaluationError("trained arm snapshot differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ActionCycleEvaluationError("strict critic-free deployment load failed")
    del snapshot
    model = model.to(device=device).eval()
    model.assert_deployable()
    model._ensure_video_only_runtime_contract()
    if (
        model.tf_condition_mode != "off"
        or model.condition_on_tf
        or bool(model.condition_on_tf_clock)
        or model.tf_loss_weight != 0.0
        or not model.parameter_matched_control
        or tuple(model.evaluation_nfe_steps) != (1, 4)
        or tuple(model.evaluation_condition_sources) != ("autonomous",)
    ):
        raise ActionCycleEvaluationError("deployment model is not the VPM no-op arm")
    trace = _validate_trace(run_dir, arm, registration)
    completion_path = run_dir / "training_complete.json"
    completion = protocol.read_json(completion_path, "training completion")
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != 200
        or completion.get("max_iter") != 200
        or completion.get("run_identity_sha256") != expected_identity
        or completion.get("snapshot") != str(snapshot_path.resolve(strict=True))
    ):
        raise ActionCycleEvaluationError("training completion differs")
    return model, config, {
        "snapshot": snapshot_record,
        "resolved_config": protocol.file_record(config_path),
        "training_trace": trace,
        "training_completion": protocol.file_record(completion_path),
        "arm_execution_plan": _validate_arm_plan(registration, arm, run_dir),
        "run_identity_sha256": expected_identity,
        "deployment_critic_loaded": False,
    }


def _uint8_future(model: Any, video: Any, out_hw: tuple[int, int]) -> Any:
    import torch

    decoded = model.rgb_tokenizer.decode_temporal(video, out_hw=out_hw)
    return (
        ((decoded[:, :, -8:].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
    )


def _rows(
    *,
    arm: protocol.Arm,
    endpoint: protocol.Endpoint,
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    decoded: Any,
    scoring: Mapping[str, Any],
    indexes: Sequence[int],
    clip_ids: Sequence[str],
    sampling_ids: Any,
    history: Any,
    actions: Any,
    donors: Sequence[int | None],
    observed_calls: int,
    registration: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    final = result["video"].detach().cpu().to(torch.float16)
    clean = scoring["video_clean"]
    history_tokens = int(prepared["history_frames"])
    if result["calls"] != endpoint.nfe or observed_calls != endpoint.nfe:
        raise ActionCycleEvaluationError("actual Wan call count differs from NFE")
    latent_nmse = phase._per_sample_nmse(final, clean, history_tokens)
    latent_delta = phase._per_sample_future_delta_nmse(final, clean, history_tokens)
    decoded_metrics = phase._per_sample_decoded(
        decoded, scoring["ground_truth"], scoring["history_last"]
    )
    hashes = {
        "cached_rgb_input_sha256": scoring["rgb_hashes"],
        "sampler_history_rgb_sha256": phase._slice_hashes(history),
        "cached_actions_input_sha256": scoring["actions_hashes"],
        "sampler_actions_sha256": phase._slice_hashes(actions),
        "video_clean_scoring_sha256": phase._slice_hashes(clean),
        "raw_ground_truth_sha256": phase._slice_hashes(scoring["ground_truth"]),
        "raw_history_last_sha256": phase._slice_hashes(scoring["history_last"]),
        "video_initial_noise_sha256": phase._slice_hashes(
            prepared["initial_video"].detach().cpu().to(torch.float16)
        ),
        "tf_initial_noise_sha256": phase._slice_hashes(
            prepared["initial_tf"].detach().cpu().to(torch.float16)
        ),
        "video_final_sha256": phase._slice_hashes(final),
        "decoded_final_sha256": phase._slice_hashes(decoded),
    }
    output = []
    sample_values = [int(value) for value in sampling_ids.detach().cpu().tolist()]
    for offset, index in enumerate(indexes):
        output.append(
            protocol.identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": KIND_ROW,
                    "registration_identity_sha256": registration["identity_sha256"],
                    "arm": asdict(arm),
                    "arm_snapshot": artifacts["snapshot"],
                    "evaluation_split": "validation",
                    "clip_index": int(index),
                    "clip_id": str(clip_ids[offset]),
                    "sampling_id": sample_values[offset],
                    "endpoint": asdict(endpoint),
                    "action_donor_clip_index": donors[offset],
                    "clean_future_rgb_passed_to_sampler": False,
                    "clean_video_latent_passed_to_sampler": False,
                    "clean_feature_passed_to_sampler": False,
                    "critic_loaded_or_called": False,
                    "online_teacher_call_count": 0,
                    "target_cache_array_opened": False,
                    "scoring_constructed_after_all_sampling": True,
                    "actual_wan_call_count": observed_calls,
                    "declared_nfe": endpoint.nfe,
                    "history_rgb_frames": 5,
                    "future_rgb_frames": 8,
                    "history_video_latent_tokens": history_tokens,
                    "future_video_latent_tokens": int(final.shape[2] - history_tokens),
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
                    "protected_test_accessed": False,
                }
            )
        )
    return output


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: protocol.Arm,
    registration: Mapping[str, Any],
) -> None:
    expected = {
        (index, endpoint.code)
        for index in range(64)
        for endpoint in protocol.ENDPOINTS
    }
    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    permutation = action_shuffle(registration)
    required_hashes = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "sampler_actions_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
        "video_final_sha256",
        "decoded_final_sha256",
    )
    for row in rows:
        endpoint_value = row.get("endpoint", {})
        endpoint = protocol.ENDPOINT_BY_CODE.get(str(endpoint_value.get("code")))
        index = row.get("clip_index")
        key = (index, str(endpoint_value.get("code")))
        expected_donor = (
            permutation[index]
            if endpoint is not None and endpoint.action_source == "shuffled"
            and isinstance(index, int)
            else None
        )
        hashes = row.get("tensor_sha256", {})
        metrics = row.get("metrics", {})
        if (
            not protocol.identity_valid(row)
            or row.get("kind") != KIND_ROW
            or row.get("registration_identity_sha256") != registration["identity_sha256"]
            or row.get("arm") != asdict(arm)
            or endpoint is None
            or endpoint_value != asdict(endpoint)
            or not isinstance(index, int)
            or not 0 <= index < 64
            or row.get("clip_id")
            != registration["validation_descriptors"][index]["clip_id"]
            or row.get("sampling_id") != VALIDATION_SAMPLE_ID_OFFSET + index
            or row.get("action_donor_clip_index") != expected_donor
            or row.get("critic_loaded_or_called") is not False
            or row.get("online_teacher_call_count") != 0
            or row.get("target_cache_array_opened") is not False
            or row.get("clean_future_rgb_passed_to_sampler") is not False
            or row.get("clean_video_latent_passed_to_sampler") is not False
            or row.get("clean_feature_passed_to_sampler") is not False
            or row.get("scoring_constructed_after_all_sampling") is not True
            or row.get("actual_wan_call_count") != endpoint.nfe
            or row.get("declared_nfe") != endpoint.nfe
            or row.get("history_video_latent_tokens") != 2
            or row.get("future_video_latent_tokens") != 2
            or row.get("protected_test_accessed") is not False
            or any(
                not isinstance(hashes.get(name), str)
                or protocol.SHA256_RE.fullmatch(hashes[name]) is None
                for name in required_hashes
            )
            or any(
                isinstance(metrics.get(name), bool)
                or not isinstance(metrics.get(name), (int, float))
                or not math.isfinite(float(metrics[name]))
                for name in (
                    "video_future_nmse",
                    "video_future_temporal_delta_nmse",
                    "decoded_mse_unit_range",
                    "decoded_psnr_db",
                    "decoded_temporal_difference_mse_unit_range",
                )
            )
            or key in observed
        ):
            raise ActionCycleEvaluationError("validation row violates protocol")
        observed[key] = row
    if set(observed) != expected:
        raise ActionCycleEvaluationError("validation row inventory is incomplete")
    invariant = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
    )
    for index in range(64):
        aligned = observed[(index, "aligned_nfe_1")]
        for endpoint in protocol.ENDPOINTS[1:]:
            row = observed[(index, endpoint.code)]
            if any(
                row["tensor_sha256"][field]
                != aligned["tensor_sha256"][field]
                for field in invariant
            ):
                raise ActionCycleEvaluationError("paired target/noise hash changed")
        if (
            aligned["tensor_sha256"]["sampler_actions_sha256"]
            != aligned["tensor_sha256"]["cached_actions_input_sha256"]
        ):
            raise ActionCycleEvaluationError("aligned action input differs from cache")
        donor = observed[(permutation[index], "aligned_nfe_1")]
        shuffled = observed[(index, "shuffled_nfe_1")]
        if (
            shuffled["tensor_sha256"]["sampler_actions_sha256"]
            != donor["tensor_sha256"]["cached_actions_input_sha256"]
        ):
            raise ActionCycleEvaluationError("shuffled action differs from sealed donor")
        zero = observed[(index, "zero_nfe_1")]
        if zero["tensor_sha256"]["sampler_actions_sha256"] in {
            aligned["tensor_sha256"]["sampler_actions_sha256"],
            shuffled["tensor_sha256"]["sampler_actions_sha256"],
        }:
            raise ActionCycleEvaluationError("zero action control is not distinct")


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE or args.batch_size_per_rank != 2:
        raise ActionCycleEvaluationError("evaluation requires 8 ranks x batch 2")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise ActionCycleEvaluationError("evaluation requires B200 GPUs")
    registration = protocol.validate_registration(args.registration, rehash_inputs=False)
    if rank == 0:
        common.revalidate_registered_inputs(
            registration,
            include_parent=False,
            include_train=False,
            include_validation=True,
        )
    dist.barrier()
    arm = protocol.ARM_BY_CODE.get(args.arm)
    if arm is None:
        raise ActionCycleEvaluationError("unknown arm")
    root = Path(registration["output_root"])
    output = root / "evaluation" / arm.code.lower()
    if args.output_dir.expanduser().absolute() != output:
        raise ActionCycleEvaluationError("evaluation output path differs")
    run_dir = (root / "training" / arm.run_name).resolve(strict=True)
    assigned = _expected_rank_indexes(rank)
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise ActionCycleEvaluationError("fresh evaluation output exists")
        output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    model, _config, artifacts = _load_deploy_model(
        registration, arm, run_dir, device
    )
    dataset = phase._RegisteredValidationInputs(
        rgb_path=Path(registration["validation"]["arrays"]["rgb"]["path"]),
        actions_path=Path(registration["validation"]["arrays"]["actions"]["path"]),
        descriptors=registration["validation_descriptors"],
        padding_dim=157,
    )
    permutation = action_shuffle(registration)
    rows: list[dict[str, Any]] = []
    hook_calls = 0
    total_calls = 0

    def count_calls(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    hook = model.forward_model.register_forward_hook(count_calls)
    try:
        for start in range(0, len(assigned), EXPECTED_BATCH_SIZE_PER_RANK):
            indexes = assigned[start : start + EXPECTED_BATCH_SIZE_PER_RANK]
            samples = [dataset[index] for index in indexes]
            batch = phase._move_batch(samples, device)
            donor_indexes = [permutation[index] for index in indexes]
            donor_samples = [dataset[index] for index in donor_indexes]
            donor_batch = phase._move_batch(donor_samples, device)
            history = batch["rgb"][:, :5].clone()
            sampling_ids = batch["clip_index"] + VALIDATION_SAMPLE_ID_OFFSET
            action_by_source = {
                "aligned": batch["actions"],
                "shuffled": donor_batch["actions"],
                "zero": torch.zeros_like(batch["actions"]),
            }
            clip_ids = [
                registration["validation_descriptors"][index]["clip_id"]
                for index in indexes
            ]
            completed = []
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                prepared_by_source = {
                    source: phase._prepare_deployable_rollout(
                        model,
                        history,
                        actions,
                        batch["morphology_index"],
                        sampling_ids,
                    )
                    for source, actions in action_by_source.items()
                }
                for endpoint in protocol.ENDPOINTS:
                    hook_calls = 0
                    prepared = prepared_by_source[endpoint.action_source]
                    result = phase._run_trajectory(model, prepared, steps=endpoint.nfe)
                    observed = hook_calls
                    decoded = _uint8_future(
                        model,
                        result["video"],
                        (int(history.shape[-2]), int(history.shape[-1])),
                    )
                    donors: list[int | None] = (
                        donor_indexes
                        if endpoint.action_source == "shuffled"
                        else [None] * len(indexes)
                    )
                    completed.append(
                        (
                            endpoint,
                            result,
                            prepared,
                            decoded,
                            action_by_source[endpoint.action_source],
                            donors,
                            observed,
                        )
                    )
                    total_calls += observed
            scoring = phase._scoring_targets(model, batch)
            for endpoint, result, prepared, decoded, actions, donors, observed in completed:
                rows.extend(
                    _rows(
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
                        actions=actions,
                        donors=donors,
                        observed_calls=observed,
                        registration=registration,
                        artifacts=artifacts,
                    )
                )
            del completed, scoring, prepared_by_source, donor_batch, batch
    finally:
        hook.remove()
    expected_calls = len(assigned) // 2 * sum(value.nfe for value in protocol.ENDPOINTS)
    if len(rows) != len(assigned) * len(protocol.ENDPOINTS) or total_calls != expected_calls:
        raise ActionCycleEvaluationError("rank row/call totals differ")
    rows_path = output / f"rank_{rank:03d}.jsonl"
    encoded = b"".join(protocol.canonical_json(row) + b"\n" for row in rows)
    common._exclusive_bytes(rows_path, encoded)
    rank_manifest = protocol.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RANK,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": asdict(arm),
            "rank": rank,
            "world_size": world_size,
            "batch_size_per_rank": 2,
            "assigned_clip_indexes": assigned,
            "endpoints": [asdict(value) for value in protocol.ENDPOINTS],
            "actual_wan_call_count": total_calls,
            "rows": {
                "path": str(rows_path),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "count": len(rows),
            },
            "critic_loaded_or_called": False,
            "protected_test_accessed": False,
        }
    )
    common._exclusive_json(output / f"rank_{rank:03d}.json", rank_manifest)
    dist.barrier()
    if rank == 0:
        global_rows = []
        rank_records = []
        for expected_rank in range(world_size):
            rank_path = output / f"rank_{expected_rank:03d}.json"
            record = protocol.read_json(rank_path, "rank manifest")
            row_path = output / f"rank_{expected_rank:03d}.jsonl"
            if (
                not protocol.identity_valid(record)
                or record.get("kind") != KIND_RANK
                or record.get("rank") != expected_rank
                or record.get("arm") != asdict(arm)
                or record.get("rows", {}).get("sha256") != common._sha256(row_path)
            ):
                raise ActionCycleEvaluationError("rank manifest differs")
            with row_path.open(encoding="utf-8") as handle:
                global_rows.extend(json.loads(line) for line in handle if line.strip())
            rank_records.append(protocol.file_record(rank_path))
        _validate_rows(global_rows, arm, registration)
        inventory = protocol.identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_INVENTORY,
                "registration": protocol.file_record(args.registration),
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": asdict(arm),
                "arm_artifacts": artifacts,
                "evaluation_split": "validation",
                "validation_clips": 64,
                "endpoints": [asdict(value) for value in protocol.ENDPOINTS],
                "row_count": len(global_rows),
                "rank_manifests": rank_records,
                "paired_inputs_and_noise_within_arm": True,
                "actual_wan_call_count": sum(
                    protocol.read_json(record["path"], "rank")[
                        "actual_wan_call_count"
                    ]
                    for record in rank_records
                ),
                "critic_loaded_or_called": False,
                "online_teacher_call_count": 0,
                "target_cache_array_opened": False,
                "protected_test_accessed": False,
            }
        )
        common._exclusive_json(output / "inventory.json", inventory)
    dist.barrier()
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--registration", type=Path, required=True)
    value.add_argument("--arm", choices=tuple(protocol.ARM_BY_CODE), required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--batch-size-per-rank", type=int, default=2)
    return value


if __name__ == "__main__":
    raise SystemExit(command_evaluate(parser().parse_args()))
