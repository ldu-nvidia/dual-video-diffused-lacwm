#!/usr/bin/env python3
"""Target-free val64 NFE1 evaluator for LFMREG-OFF versus LFMREG-ON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
for root in (str(REPO_ROOT), str(PROJECT_ROOT), str(REPO_ROOT / "tools")):
    if root not in sys.path:
        sys.path.insert(0, root)

import motion_band_training_only_screen as contract  # noqa: E402


class LFMREGEvaluationError(RuntimeError):
    pass


def _sha_tensor(value) -> str:
    import torch

    octets = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(octets.numpy().tobytes(order="C")).hexdigest()


def _per_sample_nmse(estimate, target):
    numerator = (estimate.float() - target.float()).square().flatten(1).sum(1)
    denominator = target.float().square().flatten(1).sum(1)
    return numerator / denominator.clamp_min(1.0e-30)


def _per_sample_mse(estimate, target):
    return (estimate.float() - target.float()).square().flatten(1).mean(1)


def _load_model_dataset(registration, seal, arm_code, device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    arm = contract.ARM_BY_CODE[arm_code]
    sealed = next(item for item in seal["arms"] if item["arm"] == arm_code)
    config = OmegaConf.load(sealed["resolved_config"]["path"])
    contract.validate_resolved_config(config, registration, arm)
    config.wandb.enabled = False
    model = instantiate(config.model)
    snapshot = torch.load(
        sealed["snapshot"]["path"], map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != contract.TRAIN_UPDATES
        or snapshot.get("run_identity_sha256")
        != contract.arm_identity(registration, arm)
    ):
        raise LFMREGEvaluationError("sealed snapshot metadata differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise LFMREGEvaluationError(f"strict checkpoint load failed: {incompatible}")
    del snapshot
    if any(
        "motion_band" in name.lower() or "tf_" in name.lower()
        for name, _ in model.named_parameters()
    ):
        raise LFMREGEvaluationError("training-only treatment added inference parameters")
    model = model.to(device=device).eval()
    if getattr(model.forward_model.transformer, "teacache", None) is not None:
        raise LFMREGEvaluationError("evaluation requires stateless Wan inference")

    validation = config.val_dataset
    validation.infinite = False
    validation.datasets.ABC.infinite = False
    validation.datasets.ABC.clip_manifest = registration["validation"]["manifest"][
        "path"
    ]
    validation.datasets.ABC.cache_metadata = registration["validation"]["metadata"][
        "path"
    ]
    dataset = instantiate(validation)
    if len(dataset) != contract.VALIDATION_CLIPS:
        raise LFMREGEvaluationError("validation dataset is not val64")
    child = dataset.datasets["ABC"]
    if getattr(child, "_targets", None) is not None:
        raise LFMREGEvaluationError("target-free dataset opened V-JEPA targets")
    return model, dataset, child


def _episode_rows(registration):
    with Path(registration["validation"]["manifest"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != contract.VALIDATION_CLIPS:
        raise LFMREGEvaluationError("val64 manifest cardinality changed")
    return rows


def _episode_disjoint_donor(actions, clip_index, episode_rows):
    import torch
    import torch.distributed as distributed

    world = distributed.get_world_size()
    gathered_actions = [torch.empty_like(actions) for _ in range(world)]
    gathered_indexes = [torch.empty_like(clip_index) for _ in range(world)]
    distributed.all_gather(gathered_actions, actions)
    distributed.all_gather(gathered_indexes, clip_index)
    indexes = [int(value.item()) for value in gathered_indexes]
    episodes = [episode_rows[index]["episode_dir"] for index in indexes]
    shift = next(
        (
            candidate
            for candidate in range(1, world)
            if all(
                episodes[position] != episodes[(position + candidate) % world]
                for position in range(world)
            )
        ),
        None,
    )
    if shift is None:
        raise LFMREGEvaluationError("global batch lacks episode-disjoint action donors")
    donor_rank = (distributed.get_rank() + shift) % world
    return gathered_actions[donor_rank], indexes[donor_rank]


def _sample(model, history, actions, morphology, *, clip_index):
    calls = 0

    def count_wan(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.forward_model.transformer.register_forward_hook(count_wan)
    try:
        sample = model.sample_future_deployable(
            history,
            actions,
            morphology,
            nfe=1,
            noise_seed=contract.EVALUATION_NOISE_SEED + int(clip_index),
        )
    finally:
        handle.remove()
    if calls != 1 or sample.wan_calls != 1:
        raise LFMREGEvaluationError("NFE1 sample did not execute exactly one Wan call")
    return sample


def _write_rows(path: Path, rows) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as distributed
    from torch.utils.data import DataLoader, Subset

    if os.environ.get("WANDB_MODE") != "disabled":
        raise LFMREGEvaluationError("evaluation requires WANDB_MODE=disabled")
    distributed.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = distributed.get_rank()
    try:
        registration = contract.load_registration(args.registration)
        seal = contract.load_seal(registration)
        expected_output = (
            Path(registration["study_root"]) / "evaluation" / args.arm.lower()
        )
        if args.output_dir != expected_output:
            raise LFMREGEvaluationError(f"output must be {expected_output}")
        if rank == 0:
            if args.output_dir.exists() or args.output_dir.is_symlink():
                raise LFMREGEvaluationError("evaluation output must be fresh")
            args.output_dir.mkdir(parents=True, mode=0o700)
        distributed.barrier()
        device = torch.device("cuda", local_rank)
        model, dataset, child = _load_model_dataset(
            registration, seal, args.arm, device
        )
        indexes = list(range(rank, contract.VALIDATION_CLIPS, contract.WORLD_SIZE))
        loader = DataLoader(
            Subset(dataset, indexes), batch_size=1, num_workers=0, pin_memory=True
        )
        episode_rows = _episode_rows(registration)

        import lam.motion_band_training_only_model as model_module

        original_motion_band = model_module.low_frequency_motion_consistency

        def forbidden_motion_band_call(*_args, **_kwargs):
            raise LFMREGEvaluationError(
                "training-only motion loss ran during evaluation"
            )

        model_module.low_frequency_motion_consistency = forbidden_motion_band_call
        rows = []
        consumed = []
        try:
            for raw in loader:
                clip_index = int(raw["clip_index"].item())
                consumed.append(clip_index)
                history = (
                    raw["rgb"][:, : model.num_history_frames]
                    .clone()
                    .to(device, non_blocking=True)
                )
                actions = raw["actions"].to(device, non_blocking=True)
                clip_tensor = raw["clip_index"].to(device, non_blocking=True)
                morphology = raw.get("morphology_index")
                if morphology is not None:
                    morphology = morphology.to(device, non_blocking=True)
                shuffled, donor_index = _episode_disjoint_donor(
                    actions, clip_tensor, episode_rows
                )
                controls = {
                    "aligned": (actions, clip_index),
                    "episode_shuffled": (shuffled, donor_index),
                    "zero": (torch.zeros_like(actions), None),
                }
                generated = {}
                action_hashes = {}
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ),
                ):
                    for control in contract.ACTION_CONTROLS:
                        control_actions, _donor = controls[control]
                        action_hashes[control] = _sha_tensor(control_actions[0])
                        generated[control] = _sample(
                            model,
                            history,
                            control_actions,
                            morphology,
                            clip_index=clip_index,
                        )

                    # Future RGB first reaches the accelerator after every
                    # deployment call; it is used only for metric targets.
                    full_rgb = raw["rgb"].to(device, non_blocking=True)
                    clean_latent = model._encode_clip(full_rgb)
                target_uint8 = (
                    (
                        (
                            raw["rgb"][:, -model.num_future_frames :].permute(
                                0, 2, 1, 3, 4
                            )
                            + 1.0
                        )
                        * 127.5
                    )
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                history_last = (
                    (
                        (
                            raw["rgb"][
                                :,
                                model.num_history_frames - 1 : model.num_history_frames,
                            ].permute(0, 2, 1, 3, 4)
                            + 1.0
                        )
                        * 127.5
                    )
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                for control in contract.ACTION_CONTROLS:
                    sample = generated[control]
                    pred_uint8 = (
                        ((sample.decoded_future.float().clamp(-1, 1) + 1.0) * 127.5)
                        .round()
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .cpu()
                    )
                    history_tokens = sample.history_latent_frames
                    latent_nmse = _per_sample_nmse(
                        sample.video_latent[:, :, history_tokens:],
                        clean_latent[:, :, history_tokens:],
                    )[0]
                    decoded_mse = _per_sample_mse(
                        pred_uint8.float() / 255.0,
                        target_uint8.float() / 255.0,
                    )[0]
                    pred_sequence = torch.cat(
                        [history_last.float() / 255.0, pred_uint8.float() / 255.0],
                        dim=2,
                    )
                    target_sequence = torch.cat(
                        [history_last.float() / 255.0, target_uint8.float() / 255.0],
                        dim=2,
                    )
                    temporal_mse = _per_sample_mse(
                        torch.diff(pred_sequence, dim=2),
                        torch.diff(target_sequence, dim=2),
                    )[0]
                    rows.append(
                        {
                            "schema": contract.RESULT_SCHEMA,
                            "registration_identity_sha256": registration[
                                "identity_sha256"
                            ],
                            "seal_identity_sha256": seal["identity_sha256"],
                            "arm": args.arm,
                            "run_identity_sha256": contract.arm_identity(
                                registration, contract.ARM_BY_CODE[args.arm]
                            ),
                            "clip_index": clip_index,
                            "action_control": control,
                            "action_donor_clip_index": controls[control][1],
                            "action_tensor_sha256": action_hashes[control],
                            "action_tensor_shape": [
                                int(value) for value in controls[control][0][0].shape
                            ],
                            "action_tensor_dtype": str(controls[control][0].dtype),
                            "nfe": 1,
                            "noise_seed": contract.EVALUATION_NOISE_SEED + clip_index,
                            "latent_nmse": float(latent_nmse),
                            "decoded_mse": float(decoded_mse),
                            "temporal_mse": float(temporal_mse),
                            "wan_calls": 1,
                            "history_rgb_frames_received": model.num_history_frames,
                            "future_rgb_frames_received": 0,
                            "motion_band_loss_calls_at_inference": 0,
                            "auxiliary_inputs_at_inference": 0,
                            "auxiliary_modules_at_inference": 0,
                            "online_teacher_calls": 0,
                            "cached_vjepa_target_opened": False,
                            "protected_test_accessed": False,
                            "initial_noise_sha256": _sha_tensor(
                                sample.initial_video_noise[0]
                            ),
                            "video_latent_sha256": _sha_tensor(sample.video_latent[0]),
                            "decoded_sha256": _sha_tensor(pred_uint8[0]),
                            "score_target_sha256": _sha_tensor(target_uint8[0]),
                        }
                    )
        finally:
            model_module.low_frequency_motion_consistency = original_motion_band
        if getattr(child, "_targets", None) is not None:
            raise LFMREGEvaluationError("evaluation opened a cached V-JEPA target")
        local_path = args.output_dir / f"rows.rank{rank:03d}.jsonl"
        _write_rows(local_path, rows)
        gathered_rows = [None] * contract.WORLD_SIZE if rank == 0 else None
        gathered_indexes = [None] * contract.WORLD_SIZE if rank == 0 else None
        distributed.gather_object(rows, gathered_rows, dst=0)
        distributed.gather_object(consumed, gathered_indexes, dst=0)
        if rank == 0:
            merged = [row for part in gathered_rows for row in part]
            contract.validate_result_rows(merged, contract.ARM_BY_CODE[args.arm])
            all_indexes = sorted(index for part in gathered_indexes for index in part)
            if all_indexes != list(range(contract.VALIDATION_CLIPS)):
                raise LFMREGEvaluationError("distributed val64 coverage differs")
            _write_rows(args.output_dir / "rows.jsonl", merged)
            inventory = contract.identity(
                {
                    "schema_version": 1,
                    "kind": "motion_band_training_only_evaluation_inventory",
                    "registration_identity_sha256": registration["identity_sha256"],
                    "seal_identity_sha256": seal["identity_sha256"],
                    "arm": args.arm,
                    "rows": contract.file_record(args.output_dir / "rows.jsonl"),
                    "row_count": len(merged),
                    "validation_clips": contract.VALIDATION_CLIPS,
                    "action_controls": list(contract.ACTION_CONTROLS),
                    "nfe": 1,
                    "motion_band_loss_calls_at_inference": 0,
                    "auxiliary_inputs_at_inference": 0,
                    "auxiliary_modules_at_inference": 0,
                    "cached_vjepa_target_opened": False,
                    "protected_test_accessed": False,
                }
            )
            contract.exclusive_json(args.output_dir / "inventory.json", inventory)
            print(json.dumps(inventory, sort_keys=True))
        distributed.barrier()
        return 0
    finally:
        distributed.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--arm", choices=("LFMREG-OFF", "LFMREG-ON"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(func=command_evaluate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
