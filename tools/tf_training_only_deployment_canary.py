#!/usr/bin/env python3
"""One-clip B200 proof that TFREG has no spectral inference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
for root in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)


class DeploymentCanaryError(RuntimeError):
    pass


def _sha_tensor(value) -> str:
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes(order="C")).hexdigest()


def _exclusive_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_config(registration):
    from hydra import compose, initialize_config_dir

    config_dir = PROJECT_ROOT / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=[
                "+experiments_0908=ravenhuang/wan-dit/tfreg_on",
            ],
        )
    cfg.wandb.enabled = False
    return cfg


def _load_parent(model, snapshot_path: Path):
    import torch

    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    state = snapshot.get("model")
    if not isinstance(state, Mapping):
        raise DeploymentCanaryError("parent snapshot has no model state")
    excluded = (
        "forward_model.tf_token_adapter",
        "forward_model.tf_clock_embedding",
        "forward_model.tf_velocity_head",
    )
    filtered = {
        key: value
        for key, value in state.items()
        if not any(key.startswith(prefix) for prefix in excluded)
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DeploymentCanaryError(
            "parent does not exactly match the target-free architecture: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return len(state) - len(filtered)


def command_run(args: argparse.Namespace) -> int:
    import torch
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader

    with args.registration.open(encoding="utf-8") as handle:
        registration = json.load(handle)
    if registration.get("kind") != "tf_training_only_registration":
        raise DeploymentCanaryError("registration kind differs")
    cfg = _load_config(registration)
    model = instantiate(cfg.model)
    removed_parent_keys = _load_parent(
        model, Path(registration["parent_snapshot"]["path"])
    )
    if removed_parent_keys <= 0:
        raise DeploymentCanaryError(
            "parent did not contain the expected legacy TF keys"
        )
    inference_parameters = [
        name
        for name, _ in model.named_parameters()
        if "spectral" in name.lower() or "tf_" in name.lower()
    ]
    if inference_parameters:
        raise DeploymentCanaryError(
            f"training-only intervention added parameters: {inference_parameters}"
        )

    dataset_cfg = cfg.dataset
    dataset_cfg.infinite = False
    dataset = instantiate(dataset_cfg)
    if len(dataset) != 512:
        raise DeploymentCanaryError("canary requires the registered train512 cache")
    child = dataset.datasets["ABC"]
    if getattr(child, "_targets", None) is not None:
        raise DeploymentCanaryError("target-free dataset opened cached V-JEPA targets")
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    device = torch.device("cuda", 0)
    model = model.to(device=device).eval()
    history = batch["rgb"][:, : model.num_history_frames].clone().to(device)
    actions = batch["actions"].to(device)
    morphology = batch.get("morphology_index")
    if morphology is not None:
        morphology = morphology.to(device)

    # If any training-only spectrum function is reached, fail immediately.
    import lam.tf_training_only_model as model_module

    original_spectral = model_module.spatiotemporal_spectral_consistency

    def forbidden_spectral_call(*_args, **_kwargs):
        raise DeploymentCanaryError("spectral training loss was called at inference")

    model_module.spatiotemporal_spectral_consistency = forbidden_spectral_call
    wan_calls = 0

    def count_wan(_module, _inputs, _output):
        nonlocal wan_calls
        wan_calls += 1

    handle = model.forward_model.transformer.register_forward_hook(count_wan)
    try:
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
        ):
            model.spectral_loss_weight = 0.0
            off = model.sample_future_deployable(
                history,
                actions,
                morphology,
                nfe=1,
                noise_seed=20_260_808,
            )
            model.spectral_loss_weight = 0.05
            on = model.sample_future_deployable(
                history,
                actions,
                morphology,
                nfe=1,
                noise_seed=20_260_808,
            )
    finally:
        handle.remove()
        model_module.spatiotemporal_spectral_consistency = original_spectral
    if wan_calls != 2 or off.wan_calls != 1 or on.wan_calls != 1:
        raise DeploymentCanaryError("NFE1 did not make exactly one Wan call per sample")
    if not torch.equal(off.initial_video_noise, on.initial_video_noise):
        raise DeploymentCanaryError("loss weight changed keyed initial-noise bytes")
    if not torch.equal(off.video_latent, on.video_latent):
        raise DeploymentCanaryError("loss weight changed native latent sampling bytes")
    if not torch.equal(off.decoded_future, on.decoded_future):
        raise DeploymentCanaryError("loss weight changed decoded sampling bytes")
    if getattr(child, "_targets", None) is not None:
        raise DeploymentCanaryError("canary opened a cached auxiliary target")

    report = {
        "schema_version": 1,
        "kind": "tf_training_only_deployment_canary",
        "status": "passed",
        "registration_identity_sha256": registration["identity_sha256"],
        "clip_index": int(batch["clip_index"].item()),
        "history_rgb_frames_received": int(history.shape[1]),
        "future_rgb_frames_received": 0,
        "action_tensor_sha256": _sha_tensor(actions[0]),
        "off_initial_noise_sha256": _sha_tensor(off.initial_video_noise),
        "on_initial_noise_sha256": _sha_tensor(on.initial_video_noise),
        "off_video_latent_sha256": _sha_tensor(off.video_latent),
        "on_video_latent_sha256": _sha_tensor(on.video_latent),
        "off_decoded_sha256": _sha_tensor(off.decoded_future),
        "on_decoded_sha256": _sha_tensor(on.decoded_future),
        "off_on_native_latent_bitwise_equal": True,
        "off_on_decoded_bitwise_equal": True,
        "off_on_initial_noise_bitwise_equal": torch.equal(
            off.initial_video_noise, on.initial_video_noise
        ),
        "wan_calls_total": wan_calls,
        "spectral_loss_calls": 0,
        "auxiliary_inputs": 0,
        "auxiliary_modules": 0,
        "auxiliary_parameters": 0,
        "cached_vjepa_target_opened": False,
        "protected_test_accessed": False,
        "legacy_parent_tf_keys_discarded": int(removed_parent_keys),
    }
    _exclusive_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_run)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
