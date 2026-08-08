#!/usr/bin/env python3
"""Distributed validation-only evaluator for the Haar Frequency-Forcing screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
for root in (str(REPO_ROOT / "tools"), str(REPO_ROOT), str(PROJECT_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

import frequency_forcing_screen as contract  # noqa: E402


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


def _decoded_metrics(decoded_uint8: Any, rgb: Any, future_frames: int) -> dict[str, Any]:
    import torch

    target = (
        ((rgb[:, -future_frames:].permute(0, 2, 1, 3, 4).float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    history_last = (
        ((
            rgb[:, -(future_frames + 1) : -future_frames]
            .permute(0, 2, 1, 3, 4)
            .float()
            + 1.0
        ) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    decoded = decoded_uint8.float() / 255.0
    target_float = target.float() / 255.0
    decoded_mse = _per_sample_mse(decoded, target_float)
    pred_sequence = torch.cat([history_last.float() / 255.0, decoded], dim=2)
    target_sequence = torch.cat(
        [history_last.float() / 255.0, target_float], dim=2
    )
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
    }
    mismatches = {
        key: {"observed": receipt.get(key), "expected": value}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise EvaluationError(
            f"training completion receipt differs: {mismatches}"
        )


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
        or int(config.trainer.config.max_iter) != contract.TRAIN_UPDATES
        or int(config.seed) != contract.SEED
    ):
        raise EvaluationError("resolved representation/training contract changed")
    for split_name in ("dataset", "val_dataset", "viz_dataset"):
        dataset = config[split_name]
        target = str(dataset.datasets.ABC.get("_target_", ""))
        if not target.endswith(".ABCFixedRGBActionDataset"):
            raise EvaluationError(f"{split_name} can expose an offline target")


def _load_arm_inputs(args: argparse.Namespace, rank: int) -> dict[str, Any]:
    registration_path = _canonical_file(args.registration, "registration")
    # One rank performs expensive multi-GB hashes; all ranks validate its
    # success before importing/model allocation continues.
    registration = contract.load_registration(
        registration_path, verify_files=(rank == 0)
    )
    arm = contract._arm(args.array_task_id)
    expected_run = Path(registration["study_root"]) / "runs" / arm.slug
    if args.run_dir != expected_run:
        raise EvaluationError(f"run directory must be {expected_run}")
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
        or arm_manifest.get("expected_completed_updates")
        != contract.TRAIN_UPDATES
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
    )
    return {
        "registration": registration,
        "arm": arm,
        "arm_manifest": arm_manifest,
        "config": config_path,
        "snapshot": snapshot_path,
        "completion": completion,
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
    if model.time_frequency_transform is None:
        raise EvaluationError("frequency target transform is missing")
    if any(parameter.requires_grad for parameter in model.time_frequency_transform.parameters()):
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
        batch_size=2,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=True,
    )


def _sample_cell(
    *,
    model: Any,
    batch: Mapping[str, Any],
    auxiliary_clean: Any,
    source: str,
    nfe: int,
) -> tuple[Mapping[str, Any], int, int]:
    import torch

    model.evaluation_condition_sources = (source,)
    model.evaluation_nfe_steps = (nfe,)
    model.viz_num_steps = nfe
    model.artifact_batch_limit = None
    model.capture_latent_trajectories = False
    calls = 0
    transform_calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    def count_transform(_module, _inputs, _output):
        nonlocal transform_calls
        transform_calls += 1

    forward_handle = model.forward_model.register_forward_hook(count_forward)
    transform_handle = model.time_frequency_transform.register_forward_hook(
        count_transform
    )
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            if source in contract.DEPLOYABLE_SOURCES:
                model.sample_future_deployable(
                    batch["rgb"][:, : model.num_history_frames],
                    batch["actions"],
                    batch.get("morphology_index"),
                    collect_artifacts=True,
                    sample_ids=batch["clip_index"],
                )
            else:
                model._sample_future(
                    batch["rgb"],
                    batch["actions"],
                    batch.get("morphology_index"),
                    auxiliary_target=auxiliary_clean,
                    collect_artifacts=True,
                    deployment_mode=False,
                    sample_ids=batch["clip_index"],
                )
    finally:
        forward_handle.remove()
        transform_handle.remove()
    artifacts = model.pop_visualization_artifacts()
    if not isinstance(artifacts, Mapping):
        raise EvaluationError("sampler did not expose audit artifacts")
    if transform_calls != 0:
        raise EvaluationError("sampler invoked the clean frequency transform")
    return artifacts, calls, transform_calls


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
    hook_calls: int,
    transform_calls: int,
) -> list[dict[str, Any]]:
    import torch

    infix = _source_infix(source)
    video_final = artifacts[f"video_final{infix}_nfe_{nfe}"]
    auxiliary_final = artifacts[f"tf_final{infix}_nfe_{nfe}"]
    decoded = artifacts[f"decoded_future{infix}_nfe_{nfe}"]
    recorded_calls = int(
        artifacts[f"wan_call_count{infix}_nfe_{nfe}"].reshape(-1)[0]
    )
    deployable = source in contract.DEPLOYABLE_SOURCES
    if hook_calls != nfe or recorded_calls != nfe:
        raise EvaluationError(
            f"Wan calls differ for {source}/NFE{nfe}: hook={hook_calls}, artifact={recorded_calls}"
        )
    if int(artifacts["online_teacher_call_count"].reshape(-1)[0]) != 0:
        raise EvaluationError("sampler reported a teacher call")
    if int(artifacts["deployment_mode"].reshape(-1)[0]) != int(deployable):
        raise EvaluationError("sampler deployment-mode label differs")
    if int(artifacts["auxiliary_clean_available"].reshape(-1)[0]) != int(not deployable):
        raise EvaluationError("clean auxiliary availability differs")
    forbidden = {"video_clean", "tf_clean", "ground_truth_future_uint8"}
    if deployable and forbidden.intersection(artifacts):
        raise EvaluationError("deployable sampler artifact exposes a clean target")
    if not deployable and not forbidden.issubset(artifacts):
        raise EvaluationError("oracle artifact lacks its leakage-labelled targets")
    if int(artifacts["auxiliary_history_latent_frames"].reshape(-1)[0]) != 0:
        raise EvaluationError("auxiliary state did not start wholly from noise")
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
    rows = []
    for offset, clip_index in enumerate(clip_indexes):
        values = {
            "schema": contract.RESULT_SCHEMA,
            "registration_identity_sha256": inputs["registration"]["identity_sha256"],
            "arm_identity_sha256": inputs["arm_manifest"]["identity_sha256"],
            "arm": inputs["arm"].code,
            "source": source,
            "oracle_leakage": not deployable,
            "deployable": deployable,
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
            "hook_wan_calls": hook_calls,
            "sampler_transform_calls": transform_calls,
            "online_teacher_calls": 0,
            "clean_auxiliary_passed_to_sampler": not deployable,
            "future_rgb_passed_to_sampler": not deployable,
            "all_auxiliary_bins_initialized_from_noise": True,
            "video_initial_sha256": initial_video_hashes[offset],
            "auxiliary_initial_sha256": initial_aux_hashes[offset],
            "video_final_sha256": final_video_hashes[offset],
            "auxiliary_final_sha256": final_aux_hashes[offset],
            "raw_target_sha256": decoded_metrics["target_hash"][offset],
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
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

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
        inputs = _load_arm_inputs(args, rank)
        preflight = [None]
        if rank == 0:
            preflight[0] = {
                "ok": True,
                "registration": inputs["registration"]["identity_sha256"],
            }
        dist.broadcast_object_list(preflight, src=0)
        if not preflight[0] or not preflight[0].get("ok"):
            raise EvaluationError("rank-zero registration preflight failed")

        output = args.output_dir
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
        if local_clips % 2:
            raise EvaluationError(
                "rank-local validation clips must divide evenly by evaluation batch two"
            )
        loader = _validation_loader(dataset, pin_memory=True)
        rows = []
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            if "auxiliary_target" in batch:
                raise EvaluationError("validation dataset exposed cached V-JEPA target")
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                video_clean = model._encode_clip(batch["rgb"]).to(
                    batch["rgb"].dtype
                )
                auxiliary_clean = model._tf_clean(
                    batch["rgb"], video_clean.shape
                )
            if tuple(auxiliary_clean.shape[1:]) != tuple(contract.TARGET_SHAPE):
                raise EvaluationError("online Haar target shape differs")
            for source in contract.SOURCES:
                for nfe in contract.NFE_GRID:
                    artifacts, hook_calls, transform_calls = _sample_cell(
                        model=model,
                        batch=batch,
                        auxiliary_clean=auxiliary_clean,
                        source=source,
                        nfe=nfe,
                    )
                    rows.extend(
                        _rows_for_cell(
                            inputs=inputs,
                            model=model,
                            batch=batch,
                            video_clean=video_clean.detach().cpu().to(torch.float16),
                            auxiliary_clean=auxiliary_clean.detach().cpu().to(torch.float16),
                            source=source,
                            nfe=nfe,
                            artifacts=artifacts,
                            hook_calls=hook_calls,
                            transform_calls=transform_calls,
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
            merged.sort(key=lambda row: (row["source"], int(row["nfe"]), int(row["clip_index"])))
            merged_path = output / "rows.jsonl"
            _exclusive_rows(merged_path, merged)
            complete = contract.identity_payload(
                {
                    "schema": contract.EVALUATION_COMPLETE_SCHEMA,
                    "registration_identity_sha256": inputs["registration"]["identity_sha256"],
                    "arm_identity_sha256": inputs["arm_manifest"]["identity_sha256"],
                    "arm": inputs["arm"].code,
                    "rows": len(merged),
                    "validation_clips": contract.EXPECTED_VALIDATION_CLIPS,
                    "nfe": contract.NFE_GRID,
                    "sources": contract.SOURCES,
                    "world_size": world,
                    "rows_sha256": contract.sha256_file(merged_path),
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
    except (EvaluationError, contract.ContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
