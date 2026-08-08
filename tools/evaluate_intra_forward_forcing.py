#!/usr/bin/env python3
"""Distributed validation-only evaluator for the Haar Intra-Forward Latent-Forcing screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
for root in (str(REPO_ROOT / "tools"), str(REPO_ROOT), str(PROJECT_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

import intra_forward_forcing_screen as contract  # noqa: E402


class EvaluationError(RuntimeError):
    """The checkpoint, sampler, or result grid violated the frozen protocol."""


def _canonical_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise EvaluationError(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise EvaluationError(f"{label} must be canonical and nonempty: {path}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must contain one object")
    return value


def _hash_tensor(value: Any) -> list[str]:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise EvaluationError("hash input must be a batched tensor")
    output = []
    for item in value.detach().cpu().contiguous():
        output.append(hashlib.sha256(item.numpy().tobytes()).hexdigest())
    return output


def _cross_batch_uint8_diagnostics(
    candidate: Any, reference: Any
) -> dict[str, Any]:
    """Describe, but never gate on, shape-dependent BF16 output differences."""
    import torch

    if (
        not isinstance(candidate, torch.Tensor)
        or not isinstance(reference, torch.Tensor)
        or candidate.dtype != torch.uint8
        or reference.dtype != torch.uint8
        or candidate.shape != reference.shape
        or candidate.ndim < 1
        or int(candidate.shape[0]) != 1
    ):
        raise EvaluationError(
            "cross-batch diagnostic inputs must be matching B=1 uint8 tensors"
        )
    candidate_cpu = candidate.detach().cpu().contiguous()
    reference_cpu = reference.detach().cpu().contiguous()
    delta = (candidate_cpu.to(torch.int16) - reference_cpu.to(torch.int16)).abs()
    return {
        "timed_decoded_sha256": _hash_tensor(candidate_cpu)[0],
        "audit_decoded_sha256": _hash_tensor(reference_cpu)[0],
        "cross_batch_decoded_max_abs_uint8": int(delta.max().item()),
        "cross_batch_decoded_mean_abs_uint8": float(delta.float().mean().item()),
        "cross_batch_decoded_differing_fraction": float(
            delta.ne(0).float().mean().item()
        ),
    }


def _per_sample_nmse(estimate: Any, target: Any) -> Any:
    import torch

    reduce_dims = tuple(range(1, estimate.ndim))
    numerator = (estimate.float() - target.float()).square().sum(dim=reduce_dims)
    denominator = target.float().square().sum(dim=reduce_dims)
    return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)


def _per_sample_mse(estimate: Any, target: Any) -> Any:
    reduce_dims = tuple(range(1, estimate.ndim))
    return (estimate.float() - target.float()).square().mean(dim=reduce_dims)


def _per_sample_cosine(estimate: Any, target: Any) -> Any:
    import torch.nn.functional as F

    return F.cosine_similarity(
        estimate.float().flatten(1), target.float().flatten(1), dim=1, eps=1e-8
    )


def _decoded_metrics(
    decoded_uint8: Any, rgb: Any, future_frames: int
) -> dict[str, Any]:
    import torch

    target = (
        ((rgb[:, -future_frames:].permute(0, 2, 1, 3, 4).float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    history_last = (
        (
            (
                rgb[:, -(future_frames + 1) : -future_frames]
                .permute(0, 2, 1, 3, 4)
                .float()
                + 1.0
            )
            * 127.5
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    decoded = decoded_uint8.float() / 255.0
    target_float = target.float() / 255.0
    decoded_mse = _per_sample_mse(decoded, target_float)
    pred_sequence = torch.cat([history_last.float() / 255.0, decoded], dim=2)
    target_sequence = torch.cat([history_last.float() / 255.0, target_float], dim=2)
    temporal_mse = _per_sample_mse(
        torch.diff(pred_sequence, dim=2),
        torch.diff(target_sequence, dim=2),
    )
    return {
        "decoded_mse": decoded_mse,
        "temporal_mse": temporal_mse,
        "target_hash": _hash_tensor(target),
    }


def _source_infix(source: str) -> str:
    return "" if source == "autonomous" else f"_{source}"


def _validate_completion_receipt(
    receipt: Mapping[str, Any],
    *,
    snapshot_path: Path,
    arm_identity_sha256: str,
    registration: Mapping[str, Any],
    training_content: Mapping[str, Any],
) -> None:
    """Require the trainer's final, identity-bound completion receipt.

    Merely finding ``snapshot.pt`` is insufficient: interrupted jobs can leave
    a valid intermediate snapshot at the same path.  The trainer writes this
    receipt only after all configured optimizer updates and the final save.
    """
    expected = {
        "schema_version": 1,
        "status": "completed",
        "completed_updates": contract.TRAIN_UPDATES,
        "max_iter": contract.TRAIN_UPDATES,
        "run_identity_sha256": arm_identity_sha256,
        "snapshot": str(snapshot_path),
        "wandb_run_id": arm_identity_sha256,
        "wandb_training_status": "completed",
        "source_commit": registration["source"]["commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "warm_start_sha256": registration["warm_start"]["sha256"],
        "runtime_receipt_sha256": training_content["training_runtime"]["sha256"],
        "resolved_config_sha256": training_content["resolved_config"]["sha256"],
    }
    mismatches = {
        key: {"observed": receipt.get(key), "expected": value}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise EvaluationError(f"training completion receipt differs: {mismatches}")


def _validate_config(config: Any, arm: contract.Arm) -> None:
    from omegaconf import OmegaConf

    # Resolve only the model subtree. A Hydra-authored config retains
    # ``${hydra:runtime.output_dir}`` under trainer paths, and HydraConfig is
    # intentionally unavailable in this standalone evaluator.
    model = OmegaConf.to_container(config.model, resolve=True)
    dual = model["dual_diffusion"]
    transform = model["time_frequency_transform"]
    expected = {
        "enabled": True,
        "auxiliary_history_mode": "diffuse_all",
        "tf_channels": 6,
        "condition_mode": arm.condition_mode,
        "condition_on_tf": arm.condition_on_state,
        "condition_on_tf_clock": arm.condition_on_clock,
        "schedule_mode": arm.schedule_mode,
        "tf_lead_logit": arm.lead_logit,
        "tf_loss_weight": arm.auxiliary_loss_weight,
        "state_gate_init": arm.state_gate_init,
        "clock_gate_init": arm.clock_gate_init,
        "parameter_matched_control": arm.parameter_matched_control,
    }
    for key, wanted in expected.items():
        if dual.get(key) != wanted:
            raise EvaluationError(
                f"resolved arm config {key}={dual.get(key)!r}, expected {wanted!r}"
            )
    if (
        transform.get("_target_")
        != "robot_wm.modeling.dual_diffusion.haar_lowpass.PerViewHaarLowpass"
        or transform.get("output_size") != [24, 120]
        or transform.get("window_size") != 4
        or dual.get("evaluation_nfe_steps") != contract.NFE_GRID
        or dual.get("evaluation_condition_sources") != contract.SOURCES
        or int(dual.get("evaluation_noise_seed", -1)) != contract.EVALUATION_SEED
        or dual.get("head_condition_on_tf_clock") is not True
        or dual.get("intra_forward_forcing")
        != {
            "enabled": True,
            "block_index": contract.MIDPOINT_BLOCK_INDEX,
            "stop_gradient": True,
            "history_bins": contract.AUXILIARY_HISTORY_BINS,
        }
        or bool(model["forward_model"].get("gradient_checkpointing", True))
        or int(config.trainer.config.max_iter) != contract.TRAIN_UPDATES
        or int(config.seed) != contract.SEED
    ):
        raise EvaluationError("resolved representation/training contract changed")
    if len(config.val_data_loader) != 1:
        raise EvaluationError("resolved config must have one validation loader")
    val_loader = config.val_data_loader[0]
    validation = config.trainer.config.validation
    max_iter = int(config.trainer.config.max_iter)
    val_every = int(validation.val_every)
    observed_validation_contract = {
        "dataset_infinite": bool(config.val_dataset.infinite),
        "dataset_seed": int(config.val_dataset.seed),
        "image_augmentation": bool(config.val_dataset.img_augment),
        "future_validity_enabled": bool(config.val_dataset.future_validity.enabled),
        "future_validity_max_retries": int(
            config.val_dataset.future_validity.max_retries
        ),
        "single_iterator_reused": True,
        "drop_last": bool(val_loader.drop_last),
        "batch_size_per_rank": int(val_loader.batch_size),
        "loader_workers_per_rank": int(val_loader.num_workers),
        "persistent_workers": bool(val_loader.persistent_workers),
        "local_batches_per_event": int(validation.n_val_samples),
        "local_clips_per_event": (
            int(val_loader.batch_size) * int(validation.n_val_samples)
        ),
        "global_clips_per_event": (
            contract.WORLD_SIZE
            * int(val_loader.batch_size)
            * int(validation.n_val_samples)
        ),
        "iterations": [
            iteration
            for iteration in range(max_iter)
            if iteration % val_every == 0 or iteration + 1 == max_iter
        ],
        "one_complete_registered_validation_pass_per_event": True,
    }
    try:
        contract.validate_training_validation_contract(observed_validation_contract)
    except contract.ContractError as exc:
        raise EvaluationError(str(exc)) from exc
    for split_name in ("dataset", "val_dataset", "viz_dataset"):
        dataset = config[split_name]
        target = str(dataset.datasets.ABC.get("_target_", ""))
        if not target.endswith(".ABCFixedRGBActionDataset"):
            raise EvaluationError(f"{split_name} can expose an offline target")
    if (
        config.wandb.entity != contract.WANDB_ENTITY
        or config.wandb.project != contract.WANDB_PROJECT
        or config.wandb.group is not None
    ):
        raise EvaluationError("private W&B destination changed")


def _load_arm_inputs(args: argparse.Namespace) -> dict[str, Any]:
    registration_path = _canonical_file(args.registration, "registration")
    # All multi-GB registration/warm-start/snapshot hashes were completed by
    # the single-process Slurm preflight before torchrun created this NCCL
    # process group.  Only small identity-bound receipts are read here.
    registration = contract.load_registration(registration_path, verify_files=False)
    arm = contract._arm(args.array_task_id)
    expected_run = Path(registration["study_root"]) / "runs" / arm.slug
    if args.run_dir != expected_run:
        raise EvaluationError(f"run directory must be {expected_run}")
    try:
        training_content = contract.validate_training_content(
            registration,
            arm,
            args.run_dir,
            verify_files=False,
        )
    except contract.ContractError as exc:
        raise EvaluationError(str(exc)) from exc
    evaluation_runtime_value = os.environ.get("MID_EVAL_RUNTIME_RECEIPT")
    expected_evaluation_runtime = args.run_dir / "evaluation_runtime.json"
    if (
        not evaluation_runtime_value
        or Path(evaluation_runtime_value) != expected_evaluation_runtime
    ):
        raise EvaluationError(
            "MID_EVAL_RUNTIME_RECEIPT must identify the canonical arm runtime receipt"
        )
    try:
        evaluation_runtime = contract.validate_runtime_receipt(
            expected_evaluation_runtime,
            registration,
            label="evaluation runtime",
        )
    except contract.ContractError as exc:
        raise EvaluationError(str(exc)) from exc
    if (
        evaluation_runtime["sha256"]
        != training_content["training_runtime"]["sha256"]
    ):
        raise EvaluationError("training and evaluation B200 runtime receipts differ")
    try:
        evaluation_preflight = contract.validate_evaluation_preflight(
            registration,
            arm,
            args.run_dir,
            training_content=training_content,
            evaluation_runtime=evaluation_runtime,
        )
    except contract.ContractError as exc:
        raise EvaluationError(str(exc)) from exc
    arm_manifest_path = _canonical_file(
        args.run_dir / "arm_manifest.json", "arm manifest"
    )
    arm_manifest = _read_json(arm_manifest_path, "arm manifest")
    if (
        arm_manifest.get("schema") != contract.ARM_SCHEMA
        or not contract.validate_identity(arm_manifest)
        or arm_manifest.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or arm_manifest.get("array_task_id") != args.array_task_id
        or arm_manifest.get("arm") != contract.asdict(arm)
        or arm_manifest.get("run_dir") != str(expected_run)
        or arm_manifest.get("expected_completed_updates") != contract.TRAIN_UPDATES
    ):
        raise EvaluationError("arm manifest identity or intervention differs")
    config_path = _canonical_file(
        args.run_dir / ".hydra/config.yaml", "resolved Hydra config"
    )
    snapshot_path = _canonical_file(args.run_dir / "snapshot.pt", "snapshot")
    completion_path = _canonical_file(
        args.run_dir / "training_complete.json", "training completion receipt"
    )
    completion = _read_json(completion_path, "training completion receipt")
    _validate_completion_receipt(
        completion,
        snapshot_path=snapshot_path,
        arm_identity_sha256=arm_manifest["identity_sha256"],
        registration=registration,
        training_content=training_content,
    )
    initialization_match_path = _canonical_file(
        args.run_dir / "initialization_match.json", "initialization match"
    )
    initialization_match = _read_json(initialization_match_path, "initialization match")
    if (
        initialization_match.get("schema") != contract.INITIALIZATION_MATCH_SCHEMA
        or not contract.validate_identity(initialization_match)
        or initialization_match.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or initialization_match.get("arm") != arm.code
        or initialization_match.get("arm_identity_sha256")
        != arm_manifest["identity_sha256"]
        or initialization_match.get("exact_match") is not True
    ):
        raise EvaluationError("initialization match identity differs")
    return {
        "registration": registration,
        "arm": arm,
        "arm_manifest": arm_manifest,
        "config": config_path,
        "snapshot": snapshot_path,
        "completion": completion,
        "initialization_match": initialization_match,
        "training_content": training_content,
        "evaluation_runtime": evaluation_runtime,
        "evaluation_preflight": evaluation_preflight,
        "snapshot_sha256": training_content["snapshot"]["sha256"],
        "hydra_config_sha256": training_content["hydra_config"]["sha256"],
        "resolved_config_sha256": training_content["resolved_config"]["sha256"],
    }


def _load_model_dataset(inputs: Mapping[str, Any], device: Any):
    import torch
    import torch.distributed as dist
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(inputs["config"])
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    _validate_config(config, inputs["arm"])
    model = instantiate(config.model)
    snapshot = torch.load(
        inputs["snapshot"], map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != contract.TRAIN_UPDATES
        or snapshot.get("run_identity_sha256")
        != inputs["arm_manifest"]["identity_sha256"]
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise EvaluationError("snapshot cursor, schema, or arm identity differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EvaluationError(f"strict checkpoint load failed: {incompatible}")
    del snapshot
    model = model.to(device=device).eval()
    if (
        not bool(model.forward_model.intra_forward_forcing_enabled)
        or model.forward_model.intra_forward_block_index
        != contract.MIDPOINT_BLOCK_INDEX
        or model.forward_model.intra_forward_stop_gradient is not True
        or model.forward_model.intra_forward_history_bins
        != contract.AUXILIARY_HISTORY_BINS
        or len(model.forward_model.transformer.blocks) != contract.WAN_BLOCK_COUNT
        or bool(model.forward_model.transformer.gradient_checkpointing)
    ):
        raise EvaluationError("loaded model violates the frozen midpoint seam")
    if model.time_frequency_transform is None:
        raise EvaluationError("frequency target transform is missing")
    if any(
        parameter.requires_grad
        for parameter in model.time_frequency_transform.parameters()
    ):
        raise EvaluationError("frequency target unexpectedly has trainable parameters")
    dataset = instantiate(config.val_dataset)
    if len(dataset) != contract.EXPECTED_VALIDATION_CLIPS:
        raise EvaluationError("validation dataset is not the frozen 64 clips")
    # ``MultiDataset`` is an IterableDataset whose base class captures the
    # distributed rank/world at construction and shards its iterator itself.
    # Passing a DistributedSampler to DataLoader is both redundant and rejected
    # by PyTorch for IterableDataset instances.  Fail closed if that rank binding
    # was not established before returning the validation dataset.
    if (
        not isinstance(dataset, torch.utils.data.IterableDataset)
        or int(getattr(dataset, "_process_id", -1)) != dist.get_rank()
        or int(getattr(dataset, "_num_processes", -1)) != dist.get_world_size()
    ):
        raise EvaluationError(
            "validation iterable did not capture the active distributed rank/world"
        )
    return model, dataset


def _validate_merged_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm_code: str,
) -> None:
    """Require one row for every validation clip/source/NFE cell."""
    expected = (
        contract.EXPECTED_VALIDATION_CLIPS
        * len(contract.SOURCES)
        * len(contract.NFE_GRID)
    )
    if len(rows) != expected:
        raise EvaluationError(f"arm grid has {len(rows)} rows, expected {expected}")
    keys = []
    for row in rows:
        if row.get("arm") != arm_code:
            raise EvaluationError(
                f"arm result row is labelled {row.get('arm')!r}, expected {arm_code}"
            )
        keys.append(
            (
                str(row.get("source")),
                int(row.get("nfe", -1)),
                int(row.get("clip_index", -1)),
            )
        )
    if len(set(keys)) != expected:
        raise EvaluationError("arm result grid has duplicate/missing keys")
    expected_indexes = list(range(contract.EXPECTED_VALIDATION_CLIPS))
    for source in contract.SOURCES:
        for nfe in contract.NFE_GRID:
            indexes = sorted(
                int(row["clip_index"])
                for row in rows
                if row["source"] == source and int(row["nfe"]) == nfe
            )
            if indexes != expected_indexes:
                raise EvaluationError(
                    f"{source}/NFE{nfe} did not score exact validation indexes"
                )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    output = {}
    for key, value in batch.items():
        output[key] = (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
    return output


def _validation_loader(dataset: Any, *, pin_memory: bool):
    """Build a loader without overriding the iterable's rank sharding."""
    import torch
    from torch.utils.data import DataLoader

    if not isinstance(dataset, torch.utils.data.IterableDataset):
        raise EvaluationError("validation dataset must provide iterable rank sharding")
    return DataLoader(
        dataset,
        batch_size=contract.TRAIN_VALIDATION_BATCH_SIZE_PER_RANK,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _exact_validation_batches(loader: Any):
    """Yield one val64 rank shard even when the trainer dataset is infinite."""
    iterator = iter(loader)
    seen_clip_indexes: set[int] = set()
    for _ in range(contract.TRAIN_VALIDATION_LOCAL_BATCHES_PER_EVENT):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise EvaluationError(
                "validation iterator ended before one exact val64 pass"
            ) from exc
        if (
            not isinstance(batch, Mapping)
            or "clip_index" not in batch
            or int(batch["clip_index"].numel())
            != contract.TRAIN_VALIDATION_BATCH_SIZE_PER_RANK
        ):
            raise EvaluationError("validation batch is not exact batch two")
        clip_indexes = {
            int(value) for value in batch["clip_index"].reshape(-1).tolist()
        }
        if (
            len(clip_indexes) != contract.TRAIN_VALIDATION_BATCH_SIZE_PER_RANK
            or seen_clip_indexes.intersection(clip_indexes)
        ):
            raise EvaluationError("validation rank pass repeats a clip index")
        seen_clip_indexes.update(clip_indexes)
        yield batch
    if len(seen_clip_indexes) != contract.TRAIN_VALIDATION_LOCAL_CLIPS_PER_EVENT:
        raise EvaluationError("validation rank pass is not eight unique clips")


def _sample_cell(
    *,
    model: Any,
    batch: Mapping[str, Any],
    source: str,
    nfe: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch

    model.evaluation_condition_sources = (source,)
    model.evaluation_nfe_steps = (nfe,)
    model.viz_num_steps = nfe
    model.artifact_batch_limit = None
    model.capture_latent_trajectories = False
    if source not in contract.DEPLOYABLE_SOURCES:
        raise EvaluationError("intra-forward evaluation forbids oracle sources")

    def invoke(
        input_batch: Mapping[str, Any], *, collect_artifacts: bool, profile: bool
    ):
        counts = {"wan": 0, "head": 0, "block": 0, "transform": 0}

        def increment(name):
            def hook(_module, _inputs, _output):
                counts[name] += 1

            return hook

        handles = [
            model.forward_model.register_forward_hook(increment("wan")),
            model.forward_model.tf_velocity_head.register_forward_hook(
                increment("head")
            ),
            model.forward_model.transformer.blocks[
                contract.MIDPOINT_BLOCK_INDEX
            ].register_forward_hook(increment("block")),
            model.time_frequency_transform.register_forward_hook(
                increment("transform")
            ),
        ]
        model.profile_sampling_stages = profile
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                predicted = model.sample_future_deployable(
                    input_batch["rgb"][:, : model.num_history_frames],
                    input_batch["actions"],
                    input_batch.get("morphology_index"),
                    collect_artifacts=collect_artifacts,
                    sample_ids=input_batch["clip_index"],
                )
        finally:
            model.profile_sampling_stages = False
            model.forward_model.profile_intra_forward_latency = False
            for handle in handles:
                handle.remove()
        return predicted, counts

    _audit_prediction, audit_counts = invoke(
        batch, collect_artifacts=True, profile=False
    )
    artifacts = model.pop_visualization_artifacts()
    if not isinstance(artifacts, Mapping):
        raise EvaluationError("sampler did not expose audit artifacts")
    if audit_counts["transform"] != 0:
        raise EvaluationError("sampler invoked the clean frequency transform")
    del _audit_prediction
    infix = _source_infix(source)
    audited_decoded = artifacts[f"decoded_future{infix}_nfe_{nfe}"]
    batch_size = int(batch["rgb"].shape[0])
    if batch_size != contract.TRAIN_VALIDATION_BATCH_SIZE_PER_RANK:
        raise EvaluationError("artifact audit requires the registered batch size two")
    # The artifact path and the profiling path must be bit-identical when the
    # numerical batch shape is held fixed.  BF16 kernels are allowed to choose
    # different arithmetic at B=1, so the endpoint-timing calls below are
    # compared to B=2 only diagnostically and never supply quality metrics.
    equivalence_prediction, equivalence_counts = invoke(
        batch, collect_artifacts=False, profile=True
    )
    equivalence_uint8 = (
        ((equivalence_prediction.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    if _hash_tensor(equivalence_uint8) != _hash_tensor(audited_decoded):
        raise EvaluationError(
            "same-batch profiled and artifact deployable outputs differ"
        )
    del equivalence_prediction

    timed = []
    for offset in range(batch_size):
        single = {
            key: (
                value[offset : offset + 1]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
                else value
            )
            for key, value in batch.items()
        }
        torch.cuda.synchronize(batch["rgb"].device)
        torch.cuda.reset_peak_memory_stats(batch["rgb"].device)
        timed_started_ns = time.perf_counter_ns()
        timed_prediction, timed_counts = invoke(
            single, collect_artifacts=False, profile=True
        )
        torch.cuda.synchronize(batch["rgb"].device)
        timed_end_to_end_latency_ms = (
            time.perf_counter_ns() - timed_started_ns
        ) / 1_000_000.0
        peak_memory = int(torch.cuda.max_memory_allocated(batch["rgb"].device))
        profile = getattr(model, "_last_sampling_profile", None)
        if not isinstance(profile, Mapping):
            raise EvaluationError("timed sampler did not expose its stage profile")
        if profile.get("condition_source") != source or profile.get("nfe") != nfe:
            raise EvaluationError("timed sampler profile is labelled for another cell")
        timed_uint8 = (
            ((timed_prediction.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        cross_batch = _cross_batch_uint8_diagnostics(
            timed_uint8, audited_decoded[offset : offset + 1]
        )
        timed.append(
            {
                "counts": timed_counts,
                "profile": dict(profile),
                "timed_end_to_end_latency_ms": timed_end_to_end_latency_ms,
                "peak_memory_allocated_bytes": peak_memory,
                "batch_size": 1,
                **cross_batch,
            }
        )
    audit = {
        "audit_counts": audit_counts,
        "audit_batch_size": batch_size,
        "equivalence_counts": equivalence_counts,
        "equivalence_batch_size": batch_size,
        "equivalence_exact_decoded_bytes": True,
        "equivalence_decoded_sha256": _hash_tensor(equivalence_uint8),
        "timed": timed,
    }
    return artifacts, audit


def _rows_for_cell(
    *,
    inputs: Mapping[str, Any],
    model: Any,
    batch: Mapping[str, Any],
    video_clean: Any,
    auxiliary_clean: Any,
    source: str,
    nfe: int,
    artifacts: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    infix = _source_infix(source)
    video_final = artifacts[f"video_final{infix}_nfe_{nfe}"]
    auxiliary_final = artifacts[f"tf_final{infix}_nfe_{nfe}"]
    decoded = artifacts[f"decoded_future{infix}_nfe_{nfe}"]
    recorded_calls = int(artifacts[f"wan_call_count{infix}_nfe_{nfe}"].reshape(-1)[0])
    recorded_midpoint_calls = int(
        artifacts[f"midpoint_head_call_count{infix}_nfe_{nfe}"].reshape(-1)[0]
    )
    audit_counts = audit["audit_counts"]
    equivalence_counts = audit["equivalence_counts"]
    timed_audits = audit["timed"]
    if (
        recorded_calls != nfe
        or recorded_midpoint_calls != nfe
        or any(audit_counts[name] != nfe for name in ("wan", "head", "block"))
        or audit_counts["transform"] != 0
        or int(audit.get("audit_batch_size", -1)) != 2
        or any(
            equivalence_counts[name] != nfe for name in ("wan", "head", "block")
        )
        or equivalence_counts["transform"] != 0
        or int(audit.get("equivalence_batch_size", -1)) != 2
        or audit.get("equivalence_exact_decoded_bytes") is not True
        or len(timed_audits) != 2
        or any(
            timed_audit.get("batch_size") != 1
            or any(
                timed_audit["counts"][name] != nfe
                for name in ("wan", "head", "block")
            )
            or timed_audit["counts"]["transform"] != 0
            for timed_audit in timed_audits
        )
    ):
        raise EvaluationError(
            f"call topology differs for {source}/NFE{nfe}: "
            f"artifact={recorded_calls}/{recorded_midpoint_calls}, "
            f"audit={audit_counts}, "
            f"equivalence={equivalence_counts}, "
            f"timed={[item.get('counts') for item in timed_audits]}"
        )
    if int(artifacts["online_teacher_call_count"].reshape(-1)[0]) != 0:
        raise EvaluationError("sampler reported a teacher call")
    if int(artifacts["deployment_mode"].reshape(-1)[0]) != 1:
        raise EvaluationError("sampler deployment-mode label differs")
    if int(artifacts["auxiliary_clean_available"].reshape(-1)[0]) != 0:
        raise EvaluationError("clean auxiliary availability differs")
    forbidden = {"video_clean", "tf_clean", "ground_truth_future_uint8"}
    if forbidden.intersection(artifacts):
        raise EvaluationError("deployable sampler artifact exposes a clean target")
    if int(artifacts["auxiliary_history_latent_frames"].reshape(-1)[0]) != 0:
        raise EvaluationError("auxiliary state did not start wholly from noise")
    if (
        int(artifacts["evaluation_noise_seed"].reshape(-1)[0])
        != contract.EVALUATION_SEED
    ):
        raise EvaluationError("sample-keyed evaluation seed changed")
    if not torch.equal(
        artifacts["sample_ids"].reshape(-1).to(torch.int64),
        batch["clip_index"].detach().cpu().reshape(-1).to(torch.int64),
    ):
        raise EvaluationError("sampler sample IDs differ from registered clip indexes")
    if not torch.equal(artifacts["tf_initial_state"], artifacts["tf_initial_noise"]):
        raise EvaluationError("initial auxiliary state is not exact Gaussian noise")

    history = int(artifacts["history_latent_frames"].reshape(-1)[0])
    if video_final.shape != video_clean.shape:
        raise EvaluationError("video final/target shapes differ")
    if tuple(auxiliary_final.shape[1:]) != tuple(contract.TARGET_SHAPE):
        raise EvaluationError("auxiliary final shape differs from [B,6,4,24,120]")
    if auxiliary_final.shape != auxiliary_clean.shape:
        raise EvaluationError("auxiliary final/target shapes differ")
    if decoded.shape[2] != 8:
        raise EvaluationError("decoded output is not eight future frames")

    video_nmse = _per_sample_nmse(
        video_final[:, :, history:], video_clean[:, :, history:]
    )
    auxiliary_nmse = _per_sample_nmse(
        auxiliary_final[:, :, history:], auxiliary_clean[:, :, history:]
    )
    auxiliary_cosine = _per_sample_cosine(
        auxiliary_final[:, :, history:], auxiliary_clean[:, :, history:]
    )
    dc_nmse = _per_sample_nmse(
        auxiliary_final[:, :3, history:], auxiliary_clean[:, :3, history:]
    )
    motion_nmse = _per_sample_nmse(
        auxiliary_final[:, 3:, history:], auxiliary_clean[:, 3:, history:]
    )
    decoded_metrics = _decoded_metrics(decoded, batch["rgb"], 8)
    clip_indexes = [int(value) for value in batch["clip_index"].detach().cpu().tolist()]
    initial_video_hashes = _hash_tensor(artifacts["video_initial_state"])
    initial_aux_hashes = _hash_tensor(artifacts["tf_initial_state"])
    final_video_hashes = _hash_tensor(video_final)
    final_aux_hashes = _hash_tensor(auxiliary_final)
    decoded_hashes = _hash_tensor(decoded)
    rows = []
    for offset, clip_index in enumerate(clip_indexes):
        timed_audit = timed_audits[offset]
        timed_counts = timed_audit["counts"]
        profile = timed_audit["profile"]
        values = {
            "schema": contract.RESULT_SCHEMA,
            "registration_identity_sha256": inputs["registration"]["identity_sha256"],
            "arm_identity_sha256": inputs["arm_manifest"]["identity_sha256"],
            "arm": inputs["arm"].code,
            "source": source,
            "oracle_leakage": False,
            "deployable": True,
            "nfe": nfe,
            "clip_index": clip_index,
            "video_nmse": float(video_nmse[offset]),
            "decoded_mse": float(decoded_metrics["decoded_mse"][offset]),
            "temporal_mse": float(decoded_metrics["temporal_mse"][offset]),
            "auxiliary_future_nmse": float(auxiliary_nmse[offset]),
            "auxiliary_future_cosine": float(auxiliary_cosine[offset]),
            "auxiliary_dc_nmse": float(dc_nmse[offset]),
            "auxiliary_motion_nmse": float(motion_nmse[offset]),
            "actual_wan_calls": recorded_calls,
            "hook_wan_calls": audit_counts["wan"],
            "artifact_midpoint_head_calls": recorded_midpoint_calls,
            "hook_midpoint_head_calls": audit_counts["head"],
            "hook_midpoint_block_calls": audit_counts["block"],
            "timed_wan_calls": timed_counts["wan"],
            "timed_midpoint_head_calls": timed_counts["head"],
            "timed_midpoint_block_calls": timed_counts["block"],
            "equivalence_wan_calls": equivalence_counts["wan"],
            "equivalence_midpoint_head_calls": equivalence_counts["head"],
            "equivalence_midpoint_block_calls": equivalence_counts["block"],
            "equivalence_transform_calls": equivalence_counts["transform"],
            "extra_wan_calls": 0,
            "evaluation_generations_per_cell": 4,
            "audit_batch_size": 2,
            "equivalence_batch_size": 2,
            "equivalence_exact_decoded_bytes": True,
            "timed_batch_size": 1,
            "total_evaluation_wan_calls": (
                audit_counts["wan"]
                + equivalence_counts["wan"]
                + sum(item["counts"]["wan"] for item in timed_audits)
            ),
            "wan_block_count": contract.WAN_BLOCK_COUNT,
            "midpoint_block_index": contract.MIDPOINT_BLOCK_INDEX,
            "midpoint_condition_source": {
                "autonomous": "aligned",
                "off": "off",
                "autonomous_future_shuffled": "future_shuffled",
            }[source],
            "generated_clean_stop_gradient": True,
            "sampler_transform_calls": audit_counts["transform"],
            "online_teacher_calls": 0,
            "clean_auxiliary_passed_to_sampler": False,
            "future_rgb_passed_to_sampler": False,
            "all_auxiliary_bins_initialized_from_noise": True,
            "video_initial_sha256": initial_video_hashes[offset],
            "auxiliary_initial_sha256": initial_aux_hashes[offset],
            "video_final_sha256": final_video_hashes[offset],
            "auxiliary_final_sha256": final_aux_hashes[offset],
            "decoded_final_sha256": decoded_hashes[offset],
            "equivalence_decoded_sha256": audit[
                "equivalence_decoded_sha256"
            ][offset],
            "timed_decoded_sha256": timed_audit["timed_decoded_sha256"],
            "cross_batch_decoded_max_abs_uint8": timed_audit[
                "cross_batch_decoded_max_abs_uint8"
            ],
            "cross_batch_decoded_mean_abs_uint8": timed_audit[
                "cross_batch_decoded_mean_abs_uint8"
            ],
            "cross_batch_decoded_differing_fraction": timed_audit[
                "cross_batch_decoded_differing_fraction"
            ],
            "cross_batch_output_comparison_is_diagnostic_only": True,
            "raw_target_sha256": decoded_metrics["target_hash"][offset],
            "snapshot_sha256": inputs["snapshot_sha256"],
            "hydra_config_sha256": inputs["hydra_config_sha256"],
            "resolved_config_sha256": inputs["resolved_config_sha256"],
            "training_content_identity_sha256": inputs["training_content"][
                "identity_sha256"
            ],
            "training_runtime_sha256": inputs["training_content"][
                "training_runtime"
            ]["sha256"],
            "evaluation_runtime_sha256": inputs["evaluation_runtime"]["sha256"],
            "initialization_match_identity_sha256": inputs["initialization_match"][
                "identity_sha256"
            ],
            "evaluation_preflight_identity_sha256": inputs[
                "evaluation_preflight"
            ]["identity_sha256"],
            "history_encode_latency_ms": float(profile["history_encode_latency_ms"]),
            "wan_latency_ms": float(profile["wan_latency_ms"]),
            "midpoint_overhead_latency_ms": float(
                profile["midpoint_overhead_latency_ms"]
            ),
            "decode_latency_ms": float(profile["decode_latency_ms"]),
            "end_to_end_latency_ms": float(
                timed_audit["timed_end_to_end_latency_ms"]
            ),
            "profiled_internal_end_to_end_latency_ms": float(
                profile["end_to_end_latency_ms"]
            ),
            "peak_memory_allocated_bytes": int(
                timed_audit["peak_memory_allocated_bytes"]
            ),
            "effective_state_gate": float(
                artifacts["effective_state_gate"].reshape(-1)[0]
            ),
            "effective_clock_gate": float(
                artifacts["effective_clock_gate"].reshape(-1)[0]
            ),
            "protected_test_accessed": False,
        }
        if not all(
            math.isfinite(float(values[key]))
            for key in (
                "video_nmse",
                "decoded_mse",
                "temporal_mse",
                "auxiliary_future_nmse",
                "auxiliary_future_cosine",
                "auxiliary_dc_nmse",
                "auxiliary_motion_nmse",
                "history_encode_latency_ms",
                "wan_latency_ms",
                "midpoint_overhead_latency_ms",
                "decode_latency_ms",
                "end_to_end_latency_ms",
                "profiled_internal_end_to_end_latency_ms",
            )
        ):
            raise EvaluationError("non-finite validation metric")
        rows.append(contract.identity_payload(values))
    return rows


def _exclusive_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise EvaluationError("CUDA is required")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != contract.WORLD_SIZE:
        raise EvaluationError(f"evaluation world size {world} != {contract.WORLD_SIZE}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        inputs = _load_arm_inputs(args)
        preflight = [None]
        if rank == 0:
            preflight[0] = {
                "ok": True,
                "registration": inputs["registration"]["identity_sha256"],
                "snapshot_sha256": inputs["snapshot_sha256"],
            }
        dist.broadcast_object_list(preflight, src=0)
        if not preflight[0] or not preflight[0].get("ok"):
            raise EvaluationError("rank-zero registration preflight failed")
        snapshot_sha256 = preflight[0].get("snapshot_sha256")
        if (
            not isinstance(snapshot_sha256, str)
            or contract.SHA_RE.fullmatch(snapshot_sha256) is None
        ):
            raise EvaluationError("rank-zero snapshot digest is invalid")
        inputs["snapshot_sha256"] = snapshot_sha256

        output = args.output_dir
        expected_output = (
            Path(inputs["registration"]["study_root"])
            / "evaluation"
            / inputs["arm"].slug
        )
        if output != expected_output:
            raise EvaluationError(f"evaluation output must be {expected_output}")
        if rank == 0:
            if output.exists():
                raise EvaluationError("fresh evaluation output already exists")
            output.mkdir(parents=True, mode=0o700)
        dist.barrier()

        model, dataset = _load_model_dataset(inputs, device)
        if contract.EXPECTED_VALIDATION_CLIPS % world:
            raise EvaluationError(
                "validation clips must divide evenly across the frozen world size"
            )
        local_clips = contract.EXPECTED_VALIDATION_CLIPS // world
        if local_clips != contract.TRAIN_VALIDATION_LOCAL_CLIPS_PER_EVENT:
            raise EvaluationError(
                "rank-local validation size differs from the registered pass"
            )
        if local_clips % contract.TRAIN_VALIDATION_BATCH_SIZE_PER_RANK:
            raise EvaluationError(
                "rank-local validation clips must divide evenly by evaluation batch two"
            )
        loader = _validation_loader(dataset, pin_memory=True)
        rows = []
        # val_dataset is deliberately infinite for the reusable trainer
        # iterator.  Bound standalone evaluation to one exact rank-local pass.
        for cpu_batch in _exact_validation_batches(loader):
            batch = _move_batch(cpu_batch, device)
            if "auxiliary_target" in batch:
                raise EvaluationError("validation dataset exposed cached V-JEPA target")
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                video_clean = model._encode_clip(batch["rgb"]).to(batch["rgb"].dtype)
                auxiliary_clean = model._tf_clean(batch["rgb"], video_clean.shape)
            if tuple(auxiliary_clean.shape[1:]) != tuple(contract.TARGET_SHAPE):
                raise EvaluationError("online Haar target shape differs")
            for source in contract.SOURCES:
                for nfe in contract.NFE_GRID:
                    artifacts, audit = _sample_cell(
                        model=model,
                        batch=batch,
                        source=source,
                        nfe=nfe,
                    )
                    rows.extend(
                        _rows_for_cell(
                            inputs=inputs,
                            model=model,
                            batch=batch,
                            video_clean=video_clean.detach().cpu().to(torch.float16),
                            auxiliary_clean=auxiliary_clean.detach()
                            .cpu()
                            .to(torch.float16),
                            source=source,
                            nfe=nfe,
                            artifacts=artifacts,
                            audit=audit,
                        )
                    )
                    del artifacts
        expected_local_rows = (
            local_clips * len(contract.SOURCES) * len(contract.NFE_GRID)
        )
        if len(rows) != expected_local_rows:
            raise EvaluationError(
                f"rank {rank} produced {len(rows)} rows, expected {expected_local_rows}"
            )
        rank_path = output / f"rows_rank_{rank:02d}.jsonl"
        _exclusive_rows(rank_path, rows)
        dist.barrier()
        if rank == 0:
            merged = []
            for peer in range(world):
                path = output / f"rows_rank_{peer:02d}.jsonl"
                with path.open(encoding="utf-8") as handle:
                    merged.extend(json.loads(line) for line in handle if line.strip())
            _validate_merged_grid(merged, arm_code=inputs["arm"].code)
            merged.sort(
                key=lambda row: (row["source"], int(row["nfe"]), int(row["clip_index"]))
            )
            merged_path = output / "rows.jsonl"
            _exclusive_rows(merged_path, merged)
            complete = contract.identity_payload(
                {
                    "schema": contract.EVALUATION_COMPLETE_SCHEMA,
                    "registration_identity_sha256": inputs["registration"][
                        "identity_sha256"
                    ],
                    "arm_identity_sha256": inputs["arm_manifest"]["identity_sha256"],
                    "arm": inputs["arm"].code,
                    "rows": len(merged),
                    "validation_clips": contract.EXPECTED_VALIDATION_CLIPS,
                    "nfe": contract.NFE_GRID,
                    "sources": contract.SOURCES,
                    "world_size": world,
                    "rows_sha256": contract.sha256_file(merged_path),
                    "snapshot_sha256": inputs["snapshot_sha256"],
                    "hydra_config_sha256": inputs["hydra_config_sha256"],
                    "resolved_config_sha256": inputs["resolved_config_sha256"],
                    "training_content_identity_sha256": inputs[
                        "training_content"
                    ]["identity_sha256"],
                    "training_runtime_sha256": inputs["training_content"][
                        "training_runtime"
                    ]["sha256"],
                    "evaluation_runtime_sha256": inputs["evaluation_runtime"][
                        "sha256"
                    ],
                    "initialization_match_identity_sha256": inputs[
                        "initialization_match"
                    ]["identity_sha256"],
                    "evaluation_preflight_identity_sha256": inputs[
                        "evaluation_preflight"
                    ]["identity_sha256"],
                    "protected_test_accessed": False,
                }
            )
            contract.exclusive_json(output / "complete.json", complete)
            print(json.dumps(complete, sort_keys=True))
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--array-task-id", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return evaluate(args)
    except (
        EvaluationError,
        contract.ContractError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
