#!/usr/bin/env python3
"""Run a non-destructive, one-GPU forward/backward/update validation.

In real mode the script loads one batch from every configured dataset. In synthetic
mode it uses deterministic, shape-correct tensors for the configured morphology IDs
while still loading the real model and weights. The dual-video variants use ABC only
and always execute at least four optimizer updates so gradients can pass through their
zero-initialized TF head and residual gates. It checks finite losses/gradients and
gradient flow into the intended trainable groups. An optional production snapshot is
read as a model-only warm start, hash-verified, and never modified. The script does not
save a checkpoint. Run it only through ``run_gradient_smoke.sh``, which guards the GPU
and records provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects" / "latent_action_models"
CONFIG_ROOT = PROJECT_ROOT / "configs"
if os.environ.get("VIDEOX_HOME"):
    sys.path.insert(0, os.environ["VIDEOX_HOME"])
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
# This must stay ahead of VIDEOX_HOME so Python uses the narrow Wan-only
# package initializers while extending their module paths into the pinned tree.
sys.path.insert(0, str(REPO_ROOT / "tools" / "env" / "videox_shim"))

EXPERIMENTS = {
    # Instantiate the exact official model ranks/modules. Batch size and trainer
    # cadence are irrelevant because this utility drives one-sample updates.
    "latent": "ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml",
    "explicit": "ravenhuang/wan-dit/wan_dit_explicit_abc_agibot_droid_egodex.yaml",
    "dual-no-ztf": "ravenhuang/wan-dit/dual_abc_no_ztf_condition.yaml",
    "dual-with-ztf": "ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml",
}
DUAL_VARIANTS = frozenset(("dual-no-ztf", "dual-with-ztf"))
DUAL_CONDITION_ON_TF = {
    "dual-no-ztf": False,
    "dual-with-ztf": True,
}
WARMSTART_EXCLUDE_PREFIXES = (
    "inverse_model",
    "rgb_pos_embed",
    "action_decoder",
    "action_pos_embed",
    "action_pool",
    "morphology_tokens",
)
WARMSTART_ALLOWED_MISSING_PREFIXES = (
    "action_pool.",
    "morphology_tokens.",
    "action_encoder.",
    "forward_model.tf_token_adapter.",
    "forward_model.tf_clock_embedding.",
    "forward_model.tf_velocity_head.",
)


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def group_grad_norm(model: Any, token: str) -> tuple[float, int]:
    total = 0.0
    tensors = 0
    for name, parameter in model.named_parameters():
        if token not in name or parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm().item()
        total += norm * norm
        tensors += 1
    return math.sqrt(total), tensors


def group_trainable_tensor_count(model: Any, token: str) -> int:
    return sum(
        1
        for name, parameter in model.named_parameters()
        if token in name and parameter.requires_grad
    )


def all_trainable_gradients_finite(model: Any) -> tuple[bool, list[str]]:
    import torch

    bad = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            bad.append(name)
    return not bad, bad


def sha256_stream(handle: Any, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _file_identity(file_stat: os.stat_result) -> dict[str, Any]:
    return {
        "device": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
        "size_bytes": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "ctime_ns": int(file_stat.st_ctime_ns),
        "mode_octal": oct(stat.S_IMODE(file_stat.st_mode)),
    }


def load_model_warmstart(
    model: Any,
    checkpoint_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read and audit a model-only warm start without modifying the snapshot."""
    import torch

    original_path = checkpoint_path.expanduser()
    if original_path.is_symlink():
        raise RuntimeError(
            "warm-start checkpoint must be an exact regular path, not a symlink: "
            f"{original_path}"
        )
    checkpoint_path = original_path.resolve(strict=True)
    expected_sha256 = expected_sha256.strip().lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("warm-start SHA-256 must contain exactly 64 hex digits")

    # Hash and deserialize through the same read-only descriptor. This avoids
    # validating one pathname target and then accidentally loading another.
    with checkpoint_path.open("rb") as handle:
        before_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(before_stat.st_mode):
            raise RuntimeError(
                f"warm-start checkpoint is not a regular file: {checkpoint_path}"
            )
        actual_sha256 = sha256_stream(handle)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "warm-start checkpoint SHA-256 mismatch: "
                f"{actual_sha256} != {expected_sha256}"
            )
        handle.seek(0)
        snapshot = torch.load(handle, map_location="cpu", weights_only=True)
        after_stat = os.fstat(handle.fileno())

    before_identity = _file_identity(before_stat)
    after_identity = _file_identity(after_stat)
    if before_identity != after_identity:
        raise RuntimeError("warm-start checkpoint changed while it was being read")
    if not isinstance(snapshot, Mapping) or not isinstance(
        snapshot.get("model"), Mapping
    ):
        raise RuntimeError("warm-start checkpoint does not contain a model state mapping")

    source_state = snapshot["model"]
    excluded_keys = sorted(
        key
        for key in source_state
        if any(key.startswith(prefix) for prefix in WARMSTART_EXCLUDE_PREFIXES)
    )
    excluded_key_set = set(excluded_keys)
    filtered_state = {
        key: value
        for key, value in source_state.items()
        if key not in excluded_key_set
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    disallowed_missing = [
        key
        for key in missing
        if not any(
            key.startswith(prefix)
            for prefix in WARMSTART_ALLOWED_MISSING_PREFIXES
        )
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "warm-start state is not compatible with the dual pilot "
            f"(disallowed missing={disallowed_missing[:20]}, "
            f"unexpected={unexpected[:20]})"
        )

    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "file_identity": before_identity,
        "source_key_count": len(source_state),
        "loaded_key_count": len(filtered_state) - len(unexpected),
        "excluded_prefixes": list(WARMSTART_EXCLUDE_PREFIXES),
        "excluded_key_count": len(excluded_keys),
        "excluded_keys": excluded_keys,
        "allowed_missing_prefixes": list(WARMSTART_ALLOWED_MISSING_PREFIXES),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "model_only": True,
    }


def compose_config(variant: str) -> Any:
    from hydra import compose, initialize_config_dir

    # Importing registers the mul/div OmegaConf resolvers used by the model config.
    import custom_resolvers  # noqa: F401

    experiment = EXPERIMENTS[variant]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(config_name="train", overrides=[f"+experiments_0908={experiment}"])


def morphology_samples(dataset: Any) -> list[tuple[str, Any]]:
    samples = []
    offset = 0
    for dataset_name, child in dataset.datasets.items():
        sample = dataset._get_sample(offset)
        samples.append((dataset_name, sample))
        offset += len(child)
    return samples


def synthetic_morphology_samples(variant: str) -> list[tuple[str, Any]]:
    """Build deterministic, shape-correct batches without reading training data."""
    import torch

    morphologies = (
        (("ABCSynthetic", 9),)
        if variant in DUAL_VARIANTS
        else (
            ("DroidSynthetic", 0),
            ("EgoDexSynthetic", 2),
            ("AgiBotSynthetic", 6),
            ("ABCSynthetic", 9),
        )
    )
    samples = []
    for dataset_name, morphology in morphologies:
        generator = torch.Generator(device="cpu").manual_seed(12_340 + morphology)
        # The real transform produces 13 RGB frames with three 180x320 views stacked
        # along width and normalized to [-1, 1]. Random deterministic pixels ensure
        # each view has nonzero variance and is therefore included by the loss mask.
        rgb = torch.rand((13, 3, 180, 960), generator=generator, dtype=torch.float32) * 2.0 - 1.0
        actions = torch.rand((13, 5, 157), generator=generator, dtype=torch.float32) * 0.2 - 0.1
        sample = {
            "rgb": rgb,
            "actions": actions,
            "mask": torch.ones(13, dtype=torch.bool),
            "morphology_index": torch.tensor(morphology, dtype=torch.long),
        }
        samples.append((dataset_name, sample))
    return samples


def _dual_zero_init_audit(model: Any) -> dict[str, float]:
    values = {
        "tf_state_gate_abs_max": float(
            model.forward_model.tf_token_adapter.gate.detach().float().abs().max().item()
        ),
        "tf_clock_gate_abs_max": float(
            model.forward_model.tf_clock_embedding.gate.detach().float().abs().max().item()
        ),
        "tf_head_weight_abs_max": float(
            model.forward_model.tf_velocity_head.linear.weight.detach()
            .float()
            .abs()
            .max()
            .item()
        ),
        "tf_head_bias_abs_max": float(
            model.forward_model.tf_velocity_head.linear.bias.detach()
            .float()
            .abs()
            .max()
            .item()
        ),
    }
    if any(value != 0.0 or not math.isfinite(value) for value in values.values()):
        raise RuntimeError(
            "dual residual gates/head must remain exact zero-init after warm start: "
            f"{values}"
        )
    return values


def _dual_video_noop_audit(
    model: Any,
    sample: Mapping[str, Any],
    device: Any,
    collate: Any,
) -> dict[str, Any]:
    """Prove the zero-gated dual paths exactly preserve production video output."""
    import torch

    batch = move_to_device(collate([sample]), device)
    rgb = batch["rgb"]
    actions = batch["actions"]
    morphology_index = batch.get("morphology_index")
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        video_clean = model._encode_clip(rgb).to(rgb.dtype)
        batch_size, _, latent_frames, _, _ = video_clean.shape
        history_frames = min(model.num_history_latent, latent_frames)
        reference = torch.zeros_like(video_clean)
        reference[:, :, :history_frames] = video_clean[:, :, :history_frames]
        _, z_control, _ = model._latent_actions(
            rgb,
            actions,
            morphology_index,
            latent_frames,
            history_frames,
        )
        context = model._build_context(batch_size, device, rgb.dtype)
        clip_fea = model._build_clip(batch_size, device, rgb.dtype)
        tf_clean = model._tf_clean(rgb, video_clean.shape)
        timestep = model.noise_scheduler.timesteps[
            len(model.noise_scheduler.timesteps) // 2
        ].to(device)
        timesteps = timestep.expand(batch_size)
        tf_sigma = torch.full(
            (batch_size,), 0.5, device=device, dtype=rgb.dtype
        )

        # The states are generated once and reused. Resetting CUDA RNG around
        # each call also makes the configured LoRA dropout masks identical.
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(95_731)
            video_state = torch.randn_like(video_clean)
            tf_state = torch.randn_like(tf_clean)

            # Exercise the ordinary production Wan branch: dual diffusion is
            # temporarily disabled, so noisy_tf/tf_sigma are omitted and the
            # transformer is called without y_camera. Restore the arm before
            # either dual call even when the baseline forward raises.
            original_dual_enabled = model.forward_model.dual_diffusion_enabled
            if not original_dual_enabled:
                raise RuntimeError("dual no-op audit requires a dual-enabled model")
            try:
                model.forward_model.dual_diffusion_enabled = False
                torch.manual_seed(28_441)
                production_baseline = model.forward_model(
                    video_state,
                    timesteps,
                    z_control,
                    reference,
                    context,
                    clip_fea,
                )
            finally:
                model.forward_model.dual_diffusion_enabled = original_dual_enabled

            torch.manual_seed(28_441)
            disabled = model.forward_model(
                video_state,
                timesteps,
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=tf_state,
                tf_sigma=tf_sigma,
                condition_on_tf=False,
            )
            torch.manual_seed(28_441)
            enabled = model.forward_model(
                video_state,
                timesteps,
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=tf_state,
                tf_sigma=tf_sigma,
                condition_on_tf=True,
            )

    production_video = production_baseline.detach()
    disabled_video = disabled.video_velocity.detach()
    enabled_video = enabled.video_velocity.detach()
    exact_equal = torch.equal(disabled_video, enabled_video)
    max_abs_difference = float(
        (disabled_video.float() - enabled_video.float()).abs().max().item()
    )
    production_exact_equal = torch.equal(production_video, disabled_video)
    production_max_abs_difference = float(
        (production_video.float() - disabled_video.float()).abs().max().item()
    )
    if not exact_equal or max_abs_difference != 0.0:
        raise RuntimeError(
            "zero-gated Ztf changed the pre-update video velocity "
            f"(max_abs_difference={max_abs_difference})"
        )
    if not production_exact_equal or production_max_abs_difference != 0.0:
        raise RuntimeError(
            "zero-gated dual path changed the ordinary production Wan video velocity "
            f"(max_abs_difference={production_max_abs_difference})"
        )
    return {
        "exact_video_velocity_equal": exact_equal,
        "max_abs_difference": max_abs_difference,
        "production_baseline_exact_equal": production_exact_equal,
        "production_baseline_max_abs_difference": production_max_abs_difference,
        "video_velocity_shape": list(disabled_video.shape),
        "comparison": "condition_on_tf=false versus true",
        "production_baseline": (
            "dual_diffusion_enabled=false with noisy_tf, tf_sigma, and y_camera omitted"
        ),
        "rng_seed": 28_441,
    }


def run_validation(
    variant: str,
    requested_steps: int,
    data_mode: str,
    warmstart_model: Path | None = None,
    warmstart_sha256: str | None = None,
) -> dict[str, Any]:
    import torch
    from hydra.utils import instantiate
    from torch.utils.data._utils.collate import default_collate

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"gradient smoke requires exactly one visible GPU; saw {torch.cuda.device_count()} "
            "(set CUDA_VISIBLE_DEVICES through the wrapper)"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    cfg = compose_config(variant)
    if data_mode == "real":
        dataset = instantiate(cfg.dataset)
        available_samples = morphology_samples(dataset)
    else:
        available_samples = synthetic_morphology_samples(variant)
    expected_morphologies = {9} if variant in DUAL_VARIANTS else {0, 2, 6, 9}
    available_morphologies = {
        int(sample["morphology_index"].item())
        for _, sample in available_samples
    }
    if available_morphologies != expected_morphologies:
        raise RuntimeError(
            f"expected configured morphologies {sorted(expected_morphologies)}, "
            f"found {sorted(available_morphologies)}"
        )

    model = instantiate(cfg.model).to(device).train()
    warmstart_audit = None
    if warmstart_model is not None:
        if variant not in DUAL_VARIANTS:
            raise ValueError(
                "model-only production warm start is supported only for dual variants"
            )
        if warmstart_sha256 is None:
            raise ValueError("warm-start path requires its expected SHA-256")
        warmstart_audit = load_model_warmstart(
            model, warmstart_model, warmstart_sha256
        )
    elif warmstart_sha256 is not None:
        raise ValueError("warm-start SHA-256 requires a checkpoint path")

    zero_init_audit = None
    video_noop_audit = None
    if variant in DUAL_VARIANTS:
        expected_condition = DUAL_CONDITION_ON_TF[variant]
        actual_conditions = {
            bool(model.condition_on_tf),
            bool(model.forward_model.condition_on_tf),
        }
        if actual_conditions != {expected_condition}:
            raise RuntimeError(
                f"{variant} composed with condition_on_tf={actual_conditions}, "
                f"expected {expected_condition}"
            )
        zero_init_audit = _dual_zero_init_audit(model)
        video_noop_audit = _dual_video_noop_audit(
            model,
            available_samples[0][1],
            device,
            default_collate,
        )

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=1e-4, betas=(0.9, 0.95))

    # Four steps are required for the dual graph: step 0 opens the zero TF
    # velocity head, later steps open the state/clock gates and their upstream
    # projections. The legacy arms also retain four-morphology coverage.
    steps = max(requested_steps, len(available_samples), 4)
    records = []
    decoder_morphologies_seen: set[int] = set()
    morphologies_seen: set[int] = set()
    group_ever_nonzero: dict[str, bool] = {}
    expected_groups = ["lora_", "forward_model.action_to_control", "action_pool", "morphology_tokens"]
    if variant in DUAL_VARIANTS:
        expected_groups += [
            "action_encoder",
            "forward_model.tf_velocity_head",
            "forward_model.tf_velocity_head.linear",
            "forward_model.tf_velocity_head.norm",
            "forward_model.tf_clock_embedding",
            "forward_model.tf_clock_embedding.gate",
            "forward_model.tf_clock_embedding.net",
            "forward_model.tf_token_adapter",
            "forward_model.tf_token_adapter.gate",
            "forward_model.tf_token_adapter.projection",
            "forward_model.tf_token_adapter.norm",
        ]
    else:
        expected_groups += ["action_encoder"] if variant == "explicit" else ["inverse_model", "action_decoder"]
    matched_trainable_tensors = {
        group: group_trainable_tensor_count(model, group)
        for group in expected_groups
    }
    unmatched_groups = [
        group for group, count in matched_trainable_tensors.items() if count == 0
    ]
    if unmatched_groups:
        raise RuntimeError(
            f"expected trainable parameter groups are absent: {unmatched_groups}"
        )

    for step in range(steps):
        dataset_name, sample = available_samples[step % len(available_samples)]
        batch = move_to_device(default_collate([sample]), device)
        morphology = int(batch["morphology_index"].flatten()[0].item())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()

        finite, bad_gradients = all_trainable_gradients_finite(model)
        if not finite:
            raise RuntimeError(f"non-finite gradients at step {step}: {bad_gradients[:20]}")

        norms = {}
        counts = {}
        for group in expected_groups:
            norm, count = group_grad_norm(model, group)
            norms[group] = norm
            counts[group] = count
            group_ever_nonzero[group] = group_ever_nonzero.get(group, False) or norm > 0.0

        if norms["lora_"] <= 0.0:
            raise RuntimeError(f"LoRA gradient is zero at step {step}")
        if norms["forward_model.action_to_control"] <= 0.0:
            raise RuntimeError(f"ActionToControl gradient is zero at step {step}")
        if variant in DUAL_VARIANTS:
            if norms["forward_model.tf_velocity_head.linear"] <= 0.0:
                raise RuntimeError(f"TF velocity-head gradient is zero at step {step}")
            morphologies_seen.add(morphology)
        if variant == "latent":
            if norms["inverse_model"] <= 0.0:
                raise RuntimeError(f"inverse-model gradient is zero at step {step}")
            decoder_token = f"action_decoder.decoders.{morphology}_"
            decoder_norm, decoder_count = group_grad_norm(model, decoder_token)
            if decoder_norm <= 0.0:
                raise RuntimeError(
                    f"action-decoder gradient is zero for morphology {morphology} "
                    f"at step {step} (matched tensors={decoder_count})"
                )
            decoder_morphologies_seen.add(morphology)

        optimizer.step()
        records.append(
            {
                "step": step,
                "dataset": dataset_name,
                "morphology_index": morphology,
                "loss": float(loss.detach().item()),
                "gradient_norms": norms,
                "gradient_tensor_counts": counts,
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
        print(
            f"step={step} dataset={dataset_name} morphology={morphology} "
            f"loss={loss.detach().item():.6g} lora_grad={norms['lora_']:.6g}"
        )

    # ActionToControl's final projection starts at zero, so upstream control modules can
    # legitimately receive zero gradient on step zero. They must receive signal later.
    delayed_groups = ["action_pool", "morphology_tokens"]
    if variant == "explicit" or variant in DUAL_VARIANTS:
        delayed_groups.append("action_encoder")
    if variant in DUAL_VARIANTS:
        delayed_groups += [
            "forward_model.tf_velocity_head.norm",
            "forward_model.tf_clock_embedding.gate",
            "forward_model.tf_clock_embedding.net",
            "forward_model.tf_token_adapter.gate",
            "forward_model.tf_token_adapter.projection",
            "forward_model.tf_token_adapter.norm",
        ]
    missing_signal = [group for group in delayed_groups if not group_ever_nonzero.get(group, False)]
    if missing_signal:
        raise RuntimeError(f"no gradient reached trainable conditioning groups: {missing_signal}")
    if variant == "latent" and decoder_morphologies_seen != {0, 2, 6, 9}:
        raise RuntimeError(f"did not validate every action-decoder morphology: {sorted(decoder_morphologies_seen)}")
    if variant in DUAL_VARIANTS and morphologies_seen != {9}:
        raise RuntimeError(
            f"dual smoke did not validate ABC morphology 9: {sorted(morphologies_seen)}"
        )

    return {
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "visible_device_count": torch.cuda.device_count(),
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        },
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "condition_on_tf": (
            DUAL_CONDITION_ON_TF[variant] if variant in DUAL_VARIANTS else None
        ),
        "sigma_convention": "sigma=1 is noise; sigma=0 is clean data",
        "warmstart": warmstart_audit,
        "dual_zero_init": zero_init_audit,
        "dual_video_noop": video_noop_audit,
        "matched_trainable_tensors": matched_trainable_tensors,
        "groups_ever_nonzero": group_ever_nonzero,
        "steps": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--data-mode", choices=("real", "synthetic"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="minimum steps; dual variants always run at least four updates",
    )
    parser.add_argument(
        "--warmstart-model",
        type=Path,
        help="optional model-only production snapshot (read-only)",
    )
    parser.add_argument(
        "--warmstart-sha256",
        help="required exact SHA-256 when --warmstart-model is supplied",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": f"lacwm_gradient_smoke_{args.data_mode}",
        "data_mode": args.data_mode,
        "status": "running",
        "variant": args.variant,
        "started_at_utc": started.isoformat(),
        "hostname": socket.gethostname(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "paths": {
            "repo_root": str(REPO_ROOT),
            "wan_dir": os.environ.get("WAN_DIR"),
            "videox_home": os.environ.get("VIDEOX_HOME"),
            "data_root": os.environ.get("LACWM_DATA"),
            "warmstart_model": (
                str(args.warmstart_model.expanduser().resolve(strict=False))
                if args.warmstart_model is not None
                else None
            ),
        },
    }
    try:
        if (args.warmstart_model is None) != (args.warmstart_sha256 is None):
            raise ValueError(
                "--warmstart-model and --warmstart-sha256 must be supplied together"
            )
        payload["validation"] = run_validation(
            args.variant,
            args.steps,
            args.data_mode,
            args.warmstart_model,
            args.warmstart_sha256,
        )
        payload["status"] = "passed"
        return_code = 0
    except Exception as exc:  # keep a machine-readable failed report for diagnosis
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        print(payload["traceback"], file=sys.stderr)
        return_code = 1
    finally:
        finished = datetime.now(timezone.utc)
        payload["finished_at_utc"] = finished.isoformat()
        payload["elapsed_seconds"] = (finished - started).total_seconds()
        atomic_json(args.report, payload)
        print(f"gradient smoke report: {args.report} ({payload['status']})")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
