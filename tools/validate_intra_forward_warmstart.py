#!/usr/bin/env python3
"""Fail-closed model-only warm-start compatibility check for Intra-Forward Latent-Forcing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
CONFIG_ROOT = PROJECT_ROOT / "configs"
for root in (str(REPO_ROOT / "tools"), str(REPO_ROOT), str(PROJECT_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

# Register the same OmegaConf resolvers as the production training entrypoint.
import custom_resolvers  # noqa: E402,F401
import intra_forward_forcing_screen as contract  # noqa: E402

EXCLUDED_PREFIXES = (
    "forward_model.tf_token_adapter",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_velocity_head",
)


class PreflightError(RuntimeError):
    pass


def _identity(payload):
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {**payload, "identity_sha256": hashlib.sha256(canonical).hexdigest()}


def _exclusive_json(path: Path, payload) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _tensor_state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise PreflightError(f"model state {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        encoded = _canonical_bytes(header)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_schema(model, *, trainable_only: bool = False):
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "numel": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
        if not trainable_only or parameter.requires_grad
    ]


def run(args: argparse.Namespace) -> int:
    registration_path = args.registration.resolve(strict=True)
    registration = contract.load_registration(registration_path, verify_files=False)
    checkpoint = args.warmstart.resolve(strict=True)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise PreflightError("warm start must be a canonical regular file")
    if str(checkpoint) != registration["warm_start"]["path"]:
        raise PreflightError("warm start path differs from registration")
    output = args.output
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise PreflightError(
            "output must be a fresh absolute path in an existing directory"
        )
    matching_arms = [arm for arm in contract.ARMS if arm.selector == args.selector]
    if len(matching_arms) != 1:
        raise PreflightError(
            "selector is not one frozen Intra-Forward Latent-Forcing arm"
        )
    arm = matching_arms[0]
    arm_manifest_path = args.arm_manifest.resolve(strict=True)
    if arm_manifest_path.is_symlink():
        raise PreflightError("arm manifest must not be a symlink")
    arm_manifest = json.loads(arm_manifest_path.read_text(encoding="utf-8"))
    if (
        arm_manifest.get("schema") != contract.ARM_SCHEMA
        or not contract.validate_identity(arm_manifest)
        or arm_manifest.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or arm_manifest.get("arm") != contract.asdict(arm)
        or arm_manifest_path.parent != output.parent
    ):
        raise PreflightError("arm manifest identity or run directory differs")

    _seed_all(1234)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        config = compose(
            config_name="train",
            overrides=[f"+experiments_0908={args.selector}"],
        )
    resolved_model = OmegaConf.to_container(config.model, resolve=True)
    dual = resolved_model["dual_diffusion"]
    transform = resolved_model["time_frequency_transform"]
    expected_arm = {
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
    changed = {
        key: {"observed": dual.get(key), "expected": value}
        for key, value in expected_arm.items()
        if dual.get(key) != value
    }
    if changed:
        raise PreflightError(f"resolved arm intervention changed: {changed}")
    if (
        dual.get("enabled") is not True
        or dual.get("auxiliary_history_mode") != "diffuse_all"
        or dual.get("tf_channels") != 6
        or transform.get("_target_")
        != "robot_wm.modeling.dual_diffusion.haar_lowpass.PerViewHaarLowpass"
        or transform.get("output_size") != [24, 120]
        or transform.get("window_size") != 4
        or int(config.trainer.config.max_iter) != contract.TRAIN_UPDATES
        or int(config.seed) != contract.SEED
        or dual.get("evaluation_nfe_steps") != contract.NFE_GRID
        or dual.get("evaluation_condition_sources") != contract.SOURCES
        or int(dual.get("evaluation_noise_seed", -1)) != contract.EVALUATION_SEED
        or dual.get("head_condition_on_tf_clock") is not True
        or dual.get("intra_forward_forcing")
        != {
            "enabled": True,
            "block_index": contract.MIDPOINT_BLOCK_INDEX,
            "stop_gradient": True,
        }
        or bool(config.model.forward_model.gradient_checkpointing)
    ):
        raise PreflightError("resolved common representation/training contract changed")
    if (
        config.wandb.enabled is not True
        or config.wandb.entity != contract.WANDB_ENTITY
        or config.wandb.project != contract.WANDB_PROJECT
        or config.wandb.group is not None
    ):
        raise PreflightError("private W&B destination changed")
    val_loaders = config.val_data_loader
    validation = config.trainer.config.validation
    if len(val_loaders) != 1:
        raise PreflightError("training requires exactly one validation loader")
    val_loader = val_loaders[0]
    max_iter = int(config.trainer.config.max_iter)
    val_every = int(validation.val_every)
    validation_iterations = [
        iteration
        for iteration in range(max_iter)
        if iteration % val_every == 0 or iteration + 1 == max_iter
    ]
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
        "iterations": validation_iterations,
        "one_complete_registered_validation_pass_per_event": True,
    }
    try:
        contract.validate_training_validation_contract(observed_validation_contract)
        contract.validate_training_validation_contract(
            registration["training"]["validation_iterator"]
        )
    except contract.ContractError as exc:
        raise PreflightError(str(exc)) from exc
    model = instantiate(config.model)
    if (
        bool(getattr(model.forward_model.transformer, "gradient_checkpointing", True))
        or len(model.forward_model.transformer.blocks) != contract.WAN_BLOCK_COUNT
        or model.forward_model.intra_forward_block_index
        != contract.MIDPOINT_BLOCK_INDEX
        or model.forward_model.intra_forward_stop_gradient is not True
    ):
        raise PreflightError("instantiated intra-forward seam changed")
    current = model.state_dict()
    expected_missing = sorted(
        key for key in current if key.startswith(EXCLUDED_PREFIXES)
    )
    if not expected_missing:
        raise PreflightError("new model has no auxiliary state to reinitialize")
    projection_shape = tuple(
        current["forward_model.tf_token_adapter.projection.weight"].shape
    )
    head_shape = tuple(current["forward_model.tf_velocity_head.linear.weight"].shape)
    if projection_shape[1] != 6 or head_shape[0] != 24:
        raise PreflightError(
            f"new six-channel schema changed: projection={projection_shape}, head={head_shape}"
        )

    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != 1000
        or not isinstance(snapshot.get("model"), dict)
    ):
        raise PreflightError("historical VPM snapshot schema/cursor differs")
    historical = snapshot["model"]
    historical_projection = historical.get(
        "forward_model.tf_token_adapter.projection.weight"
    )
    if (
        not isinstance(historical_projection, torch.Tensor)
        or historical_projection.shape[1] != 64
    ):
        raise PreflightError("historical checkpoint is not the 64-channel VPM schema")
    filtered = {
        key: value
        for key, value in historical.items()
        if not key.startswith(EXCLUDED_PREFIXES)
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise PreflightError(
            "non-auxiliary warm-start compatibility failed: "
            f"missing={missing}, expected={expected_missing}, unexpected={unexpected}"
        )
    initialized_state = model.state_dict()
    parameter_schema = _parameter_schema(model)
    trainable_schema = _parameter_schema(model, trainable_only=True)
    optimizer_schema = {
        "optimizer_factory": OmegaConf.to_container(
            config.optimizer_factory, resolve=True
        ),
        "lr_scheduler_factory": OmegaConf.to_container(
            config.lr_scheduler_factory, resolve=True
        ),
        "gradient_clipping": OmegaConf.to_container(
            config.trainer.config.gradient_clipping, resolve=True
        ),
        "trainable_parameters": trainable_schema,
    }
    payload = _identity(
        {
            "schema": contract.PREFLIGHT_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": arm.code,
            "selector": args.selector,
            "arm_identity_sha256": arm_manifest["identity_sha256"],
            "warmstart": str(checkpoint),
            "historical_completed_updates": 1000,
            "historical_auxiliary_channels": 64,
            "new_auxiliary_channels": 6,
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "expected_missing_keys": expected_missing,
            "unexpected_keys": unexpected,
            "non_auxiliary_loaded_keys": len(filtered),
            "new_projection_shape": list(projection_shape),
            "new_velocity_head_shape": list(head_shape),
            "initialized_model_state_sha256": _tensor_state_sha256(initialized_state),
            "parameter_schema_sha256": hashlib.sha256(
                _canonical_bytes(parameter_schema)
            ).hexdigest(),
            "trainable_parameter_schema_sha256": hashlib.sha256(
                _canonical_bytes(trainable_schema)
            ).hexdigest(),
            "optimizer_schema_sha256": hashlib.sha256(
                _canonical_bytes(optimizer_schema)
            ).hexdigest(),
            "total_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "initial_snapshot_tensor_count": len(initialized_state),
            "gradient_checkpointing": False,
            "wan_block_count": contract.WAN_BLOCK_COUNT,
            "midpoint_block_index": contract.MIDPOINT_BLOCK_INDEX,
            "training_validation_iterator": observed_validation_contract,
            "status": "pass",
        }
    )
    _exclusive_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--arm-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (PreflightError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
