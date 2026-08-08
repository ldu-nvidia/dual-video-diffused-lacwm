#!/usr/bin/env python3
"""Sealed val64 evaluator for the training-only VideoREPA TRD screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
for root in (str(REPO_ROOT), str(PROJECT_ROOT), str(REPO_ROOT / "tools")):
    if root not in sys.path:
        sys.path.insert(0, root)

import videorepa_trd_screen as contract  # noqa: E402


class TRDEvaluationError(RuntimeError):
    """The sealed checkpoint, target-free sampler, or result grid changed."""


def _hash_sample(value) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _per_sample_nmse(estimate, target):
    import torch

    numerator = (estimate.float() - target.float()).square().flatten(1).sum(1)
    denominator = target.float().square().flatten(1).sum(1)
    return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)


def _per_sample_mse(estimate, target):
    return (estimate.float() - target.float()).square().flatten(1).mean(1)


def _validate_config(config, arm: contract.Arm, registration: Mapping[str, Any]):
    from omegaconf import OmegaConf

    model = OmegaConf.to_container(config.model, resolve=True)
    trd = model["token_relation_distillation"]
    dual = model["dual_diffusion"]
    forward = model["forward_model"]
    expected_trd = {
        "mode": arm.mode,
        "loss_weight": arm.loss_weight,
        "margin": 0.05,
        "block_index": 14,
        "num_views": 3,
        "pool_height": 2,
        "pool_width_per_view": 4,
        "exclude_first_temporal_bin": True,
    }
    if trd != expected_trd:
        raise TRDEvaluationError("resolved TRD intervention differs")
    if (
        model.get("_target_")
        != "lam.videorepa_trd_model.VideoRepaTRDModel"
        or bool(forward.get("gradient_checkpointing", True))
        or int(forward.get("lora_rank", -1)) != 64
        or int(forward.get("lora_alpha", -1)) != 128
        or float(forward.get("lora_dropout", -1)) != 0.05
        or int(model.get("num_history_frames", -1)) != 5
        or int(model.get("num_future_frames", -1)) != 8
        or dual.get("enabled") is not True
        or dual.get("parameter_matched_control") is not True
        or dual.get("video_only_control") is not False
        or dual.get("auxiliary_history_mode") != "diffuse_all"
        or dual.get("condition_mode") != "off"
        or dual.get("condition_on_tf") is not False
        or dual.get("condition_on_tf_clock") is not False
        or float(dual.get("tf_loss_weight", -1)) != 0.0
        or int(dual.get("tf_channels", -1)) != 64
        or dual.get("schedule_mode") != "aligned"
        or dual.get("evaluation_nfe_steps") != [1]
        or dual.get("evaluation_condition_sources") != ["off"]
        or dual.get("capture_latent_trajectories") is not False
        or int(config.trainer.config.max_iter) != contract.TRAIN_UPDATES
        or int(config.trainer.config.gradient_accumulation_steps) != 1
        or str(config.trainer.config.load_path)
        != registration["parent_snapshot"]["file"]["path"]
        or list(config.trainer.config.exclude_keys) != []
        or float(config.optimizer_factory.lr) != 1.0e-4
        or list(config.optimizer_factory.betas) != [0.9, 0.95]
        or int(config.lr_scheduler_factory.lr_lambda.warmup_steps) != 20
        or int(config.lr_scheduler_factory.lr_lambda.total_steps)
        != contract.TRAIN_UPDATES
        or float(config.lr_scheduler_factory.lr_lambda.final_learning_rate)
        != 1.0e-6
        or int(config.data_loader.batch_size) != 1
        or int(config.seed) != contract.SEED
        or config.wandb.entity != contract.WANDB_ENTITY
        or config.wandb.project != contract.WANDB_PROJECT
        or config.wandb.group is not None
        or config.wandb.name != arm.run_name
        or str(config.wandb.id) != contract.arm_identity(registration, arm)
        or str(config.wandb.resume) != "never"
    ):
        raise TRDEvaluationError("resolved model/training/W&B contract differs")
    train_manifest = registration["training"]["manifest"]["path"]
    train_metadata = registration["training"]["cache_metadata"]["path"]
    for split in ("dataset", "val_dataset", "viz_dataset"):
        dataset = config[split]
        if (
            str(dataset.datasets.ABC.clip_manifest) != train_manifest
            or str(dataset.datasets.ABC.cache_metadata) != train_metadata
        ):
            raise TRDEvaluationError(
                "training config accessed a non-train cache before sealing"
            )


def _load_model_dataset(registration, seal, arm, device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    paths = contract.arm_paths(registration, arm)
    sealed_arm = next(
        item for item in seal["arms"] if item["arm"]["code"] == arm.code
    )
    config_path = Path(sealed_arm["resolved_config"]["path"])
    snapshot_path = Path(sealed_arm["snapshot"]["path"])
    config = OmegaConf.load(config_path)
    config.wandb.enabled = False
    _validate_config(config, arm, registration)
    model = instantiate(config.model)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True, mmap=True)
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != contract.TRAIN_UPDATES
        or snapshot.get("run_identity_sha256")
        != contract.arm_identity(registration, arm)
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise TRDEvaluationError("sealed arm snapshot differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TRDEvaluationError(f"strict snapshot load failed: {incompatible}")
    del snapshot
    model = model.to(device=device).eval()
    if (
        getattr(model.forward_model.transformer, "teacache", None) is not None
        or int(getattr(model.forward_model.transformer, "sp_world_size", 1)) != 1
    ):
        raise TRDEvaluationError(
            "evaluation requires stateless Wan calls without sequence parallelism"
        )
    if any(
        "token_relation" in name or "trd" in name.lower()
        for name, _parameter in model.named_parameters()
    ):
        raise TRDEvaluationError("TRD unexpectedly added an inference parameter")

    # Build a target-free view of the sealed cache.  The inherited cache
    # validator resolves but never opens the auxiliary target array.
    validation = config.val_dataset
    validation.infinite = False
    validation.seed = contract.SEED
    validation.future_validity.enabled = False
    validation.future_validity.max_retries = 0
    validation.datasets.ABC._target_ = (
        "robot_wm.datasets.abc.fixed_rgb_action_dataset.ABCFixedRGBActionDataset"
    )
    validation.datasets.ABC.clip_manifest = registration["validation"]["manifest"][
        "path"
    ]
    validation.datasets.ABC.cache_metadata = registration["validation"][
        "cache_metadata"
    ]["path"]
    validation.datasets.ABC.infinite = False
    dataset = instantiate(validation)
    if len(dataset) != contract.VALIDATION_CLIPS:
        raise TRDEvaluationError("sealed val64 dataset length differs")
    child = dataset.datasets["ABC"]
    if getattr(child, "_targets", None) is not None:
        raise TRDEvaluationError("clean V-JEPA target array was opened")
    return model, dataset, paths


def _episode_rows(registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    with Path(registration["validation"]["manifest"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != contract.VALIDATION_CLIPS:
        raise TRDEvaluationError("validation manifest changed")
    return rows


def _episode_disjoint_actions(actions, clip_indexes, episode_rows):
    """Choose one global cyclic donor shift with no same-episode recipient."""

    import torch
    import torch.distributed as distributed

    world = distributed.get_world_size()
    if actions.shape[0] != 1 or clip_indexes.numel() != 1:
        raise TRDEvaluationError("frozen evaluator requires batch one per rank")
    gathered_actions = [torch.empty_like(actions) for _ in range(world)]
    gathered_indexes = [torch.empty_like(clip_indexes) for _ in range(world)]
    distributed.all_gather(gathered_actions, actions)
    distributed.all_gather(gathered_indexes, clip_indexes)
    indexes = [int(value.item()) for value in gathered_indexes]
    episodes = [episode_rows[index]["episode_dir"] for index in indexes]
    shift = None
    for candidate in range(1, world):
        if all(
            episodes[index] != episodes[(index + candidate) % world]
            for index in range(world)
        ):
            shift = candidate
            break
    if shift is None:
        raise TRDEvaluationError(
            "global val batch has no episode-disjoint cyclic action donor"
        )
    rank = distributed.get_rank()
    return gathered_actions[(rank + shift) % world], indexes[(rank + shift) % world]


def _sample_cell(model, batch, *, actions, nfe):
    import torch

    model.evaluation_condition_sources = ("off",)
    model.evaluation_nfe_steps = (int(nfe),)
    model.viz_num_steps = int(nfe)
    model.artifact_batch_limit = None
    model.capture_latent_trajectories = False
    before = int(model._trd_forward_hook_installations)
    calls = 0
    auxiliary_calls = 0

    def count_wan(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    def count_auxiliary(_module, _inputs, _output):
        nonlocal auxiliary_calls
        auxiliary_calls += 1

    handles = [
        model.forward_model.transformer.register_forward_hook(count_wan),
        model.forward_model.transformer.control_adapter.register_forward_hook(
            count_auxiliary
        ),
        # The historical dual path calls these leaf methods directly rather
        # than each wrapper's ``forward``.  Hook the executed leaves so a
        # future refactor cannot evade the zero-auxiliary inference check.
        model.forward_model.tf_token_adapter.projection.register_forward_hook(
            count_auxiliary
        ),
        model.forward_model.tf_token_adapter.norm.register_forward_hook(
            count_auxiliary
        ),
        model.forward_model.tf_clock_embedding.net.register_forward_hook(
            count_auxiliary
        ),
        model.forward_model.tf_velocity_head.register_forward_hook(count_auxiliary),
    ]
    try:
        # Materialize an independent history-only allocation.  A mere temporal
        # slice would retain the full 13-frame backing storage even though its
        # visible shape is causal.
        history_only = batch["rgb"][:, : model.num_history_frames].clone(
            memory_format=torch.contiguous_format
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            model.sample_future_deployable(
                history_only,
                actions,
                batch.get("morphology_index"),
                collect_artifacts=True,
                sample_ids=batch["clip_index"],
            )
    finally:
        for handle in handles:
            handle.remove()
    after = int(model._trd_forward_hook_installations)
    artifacts = model.pop_visualization_artifacts()
    if (
        calls != nfe
        or after != before
        or auxiliary_calls != 0
        or not isinstance(artifacts, Mapping)
        or int(artifacts["deployment_mode"].item()) != 1
        or int(artifacts["auxiliary_clean_available"].item()) != 0
        or int(artifacts["online_teacher_call_count"].item()) != 0
        or int(artifacts["trd_inference_branch_call_count"].item()) != 0
        or "video_clean" in artifacts
        or "tf_clean" in artifacts
        or "ground_truth_future_uint8" in artifacts
    ):
        raise TRDEvaluationError("deployable sampler leakage/call contract differs")
    return artifacts, calls, after - before, auxiliary_calls


def _move_sampling_batch(batch, device, *, history_frames: int):
    import torch

    moved = {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
        if key != "rgb"
    }
    # Future RGB remains in the CPU scoring batch until every generation cell
    # has finished.  The sampler-side dictionary physically contains only an
    # independently allocated observed prefix.
    moved["rgb"] = batch["rgb"][:, :history_frames].clone().to(
        device, non_blocking=True
    )
    return moved


def _rows_for_batch(
    *,
    registration,
    seal,
    arm,
    model,
    batch,
    cells,
    donor_clip_index,
):
    import torch

    # Clean future information is constructed only now, after every deployable
    # generation call for this batch has completed.
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        clean_latent = model._encode_clip(batch["rgb"]).detach().cpu()
    target_uint8 = (
        ((batch["rgb"][:, -model.num_future_frames :].permute(0, 2, 1, 3, 4).float() + 1.0)
        * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    history_last = (
        ((batch["rgb"][:, model.num_history_frames - 1 : model.num_history_frames]
        .permute(0, 2, 1, 3, 4).float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    rows = []
    clip_index = int(batch["clip_index"].item())
    for (control, nfe), (artifacts, calls, trd_hooks, auxiliary_calls) in cells.items():
        infix = "_off"
        final_latent = artifacts[f"video_final{infix}_nfe_{nfe}"]
        decoded = artifacts[f"decoded_future{infix}_nfe_{nfe}"]
        history = int(artifacts["history_latent_frames"].item())
        latent_nmse = _per_sample_nmse(
            final_latent[:, :, history:], clean_latent[:, :, history:]
        )[0]
        decoded_mse = _per_sample_mse(
            decoded.float() / 255.0, target_uint8.float() / 255.0
        )[0]
        predicted_sequence = torch.cat(
            [history_last.float() / 255.0, decoded.float() / 255.0], dim=2
        )
        target_sequence = torch.cat(
            [history_last.float() / 255.0, target_uint8.float() / 255.0], dim=2
        )
        temporal_mse = _per_sample_mse(
            torch.diff(predicted_sequence, dim=2),
            torch.diff(target_sequence, dim=2),
        )[0]
        rows.append(
            {
                "schema": contract.RESULT_SCHEMA,
                "registration_identity_sha256": registration["identity_sha256"],
                "post_training_seal_identity_sha256": seal["identity_sha256"],
                "arm": arm.code,
                "run_identity_sha256": contract.arm_identity(registration, arm),
                "action_control": control,
                "action_donor_clip_index": (
                    clip_index
                    if control == "aligned"
                    else donor_clip_index
                    if control == "episode_shuffled"
                    else None
                ),
                "nfe": int(nfe),
                "clip_index": clip_index,
                "latent_nmse": float(latent_nmse),
                "decoded_mse": float(decoded_mse),
                "temporal_mse": float(temporal_mse),
                "actual_wan_calls": int(calls),
                "trd_hook_calls_at_inference": int(trd_hooks),
                "auxiliary_branch_calls_at_inference": int(auxiliary_calls),
                "online_teacher_calls": 0,
                "sampler_received_clean_feature": False,
                "sampler_received_future_rgb": False,
                "score_only_future_rgb_used_after_generation": True,
                "target_array_opened": False,
                "video_initial_sha256": _hash_sample(
                    artifacts["video_initial_state"][0]
                ),
                "video_final_sha256": _hash_sample(final_latent[0]),
                "decoded_sha256": _hash_sample(decoded[0]),
                "raw_target_sha256": _hash_sample(target_uint8[0]),
                "protected_test_accessed": False,
            }
        )
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as distributed
    from torch.utils.data import DataLoader

    if os.environ.get("WANDB_MODE") != "disabled":
        raise TRDEvaluationError("evaluation requires WANDB_MODE=disabled")
    distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = distributed.get_rank()
    try:
        registration = contract.load_registration(args.registration)
        seal = contract.load_seal(registration)
        arm = contract.ARM_BY_CODE[args.arm]
        output = args.output_dir
        expected = Path(contract.arm_paths(registration, arm)["evaluation_dir"])
        if output != expected:
            raise TRDEvaluationError(f"evaluation output must be {expected}")
        if rank == 0:
            if output.exists() or output.is_symlink():
                raise TRDEvaluationError("evaluation output must be fresh")
            output.mkdir(parents=True, mode=0o700)
        distributed.barrier()
        model, dataset, _paths = _load_model_dataset(
            registration, seal, arm, torch.device("cuda", local_rank)
        )
        child = dataset.datasets["ABC"]
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        episode_rows = _episode_rows(registration)
        local_rows = []
        local_indexes = []
        for raw_batch in loader:
            device = torch.device("cuda", local_rank)
            batch = _move_sampling_batch(
                raw_batch,
                device,
                history_frames=model.num_history_frames,
            )
            local_indexes.append(int(batch["clip_index"].item()))
            shuffled_actions, donor_clip = _episode_disjoint_actions(
                batch["actions"], batch["clip_index"], episode_rows
            )
            control_actions = {
                "aligned": batch["actions"],
                "episode_shuffled": shuffled_actions,
                "zero": torch.zeros_like(batch["actions"]),
            }
            cells = {}
            for control in contract.ACTION_CONTROLS:
                for nfe in contract.NFE_GRID:
                    cells[(control, nfe)] = _sample_cell(
                        model,
                        batch,
                        actions=control_actions[control],
                        nfe=nfe,
                    )
            scoring_batch = dict(batch)
            scoring_batch["rgb"] = raw_batch["rgb"].to(
                device, non_blocking=True
            )
            local_rows.extend(
                _rows_for_batch(
                    registration=registration,
                    seal=seal,
                    arm=arm,
                    model=model,
                    batch=scoring_batch,
                    cells=cells,
                    donor_clip_index=donor_clip,
                )
            )
        if len(local_indexes) != contract.VALIDATION_CLIPS // contract.WORLD_SIZE:
            raise TRDEvaluationError("rank did not consume exactly eight val clips")
        if getattr(child, "_targets", None) is not None:
            raise TRDEvaluationError("evaluation opened the clean target array")
        local_path = output / f"rows.rank{rank:03d}.jsonl"
        _write_rows(local_path, local_rows)
        gathered = [None for _ in range(contract.WORLD_SIZE)] if rank == 0 else None
        distributed.gather_object(local_rows, gathered, dst=0)
        gathered_indexes = (
            [None for _ in range(contract.WORLD_SIZE)] if rank == 0 else None
        )
        distributed.gather_object(local_indexes, gathered_indexes, dst=0)
        if rank == 0:
            merged = [row for rank_rows in gathered for row in rank_rows]
            contract.validate_result_rows(merged, arm)
            all_indexes = sorted(index for values in gathered_indexes for index in values)
            if all_indexes != list(range(contract.VALIDATION_CLIPS)):
                raise TRDEvaluationError("val64 clip coverage differs")
            merged_path = output / "rows.jsonl"
            _write_rows(merged_path, merged)
            rows_record = contract._file_record(
                merged_path, f"{arm.code} merged evaluation rows"
            )
            inventory = contract._identity(
                {
                    "schema_version": 1,
                    "kind": "videorepa_trd_evaluation_inventory",
                    "registration_identity_sha256": registration["identity_sha256"],
                    "post_training_seal_identity_sha256": seal["identity_sha256"],
                    "arm": arm.code,
                    "run_identity_sha256": contract.arm_identity(registration, arm),
                    "row_count": len(merged),
                    "rows_file": rows_record,
                    "validation_clips": contract.VALIDATION_CLIPS,
                    "nfe_grid": list(contract.NFE_GRID),
                    "action_controls": list(contract.ACTION_CONTROLS),
                    "target_array_opened": False,
                    "sampler_clean_feature_calls": 0,
                    "sampler_future_rgb_calls": 0,
                    "online_teacher_calls": 0,
                    "trd_inference_parameters": 0,
                    "protected_test_accessed": False,
                }
            )
            contract._exclusive_json(output / "inventory.json", inventory)
            print(json.dumps(inventory, sort_keys=True))
        distributed.barrier()
        return 0
    finally:
        distributed.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--registration", type=Path, required=True)
    evaluate.add_argument("--arm", choices=tuple(contract.ARM_BY_CODE), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
