#!/usr/bin/env python3
"""Run a destructive-free, one-GPU forward/backward/update validation.

In real mode the script loads one batch from every configured dataset. In synthetic
mode it uses deterministic, shape-correct tensors for the same four morphology IDs
while still loading the real model and weights. It checks finite losses/gradients and
gradient flow into the intended trainable groups. It does not save a checkpoint. Run
it only through ``run_gradient_smoke.sh``, which guards the GPU and records provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def all_trainable_gradients_finite(model: Any) -> tuple[bool, list[str]]:
    import torch

    bad = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            bad.append(name)
    return not bad, bad


def compose_config(variant: str) -> Any:
    from hydra import compose, initialize_config_dir

    # Importing registers the mul/div OmegaConf resolvers used by the model config.
    import custom_resolvers  # noqa: F401

    experiment = {
        # Instantiate the exact official model ranks/modules.  Batch size and
        # trainer cadence are irrelevant because this utility drives four
        # one-sample updates directly.
        "latent": "ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml",
        "explicit": "ravenhuang/wan-dit/wan_dit_explicit_abc_agibot_droid_egodex.yaml",
    }[variant]
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


def synthetic_morphology_samples() -> list[tuple[str, Any]]:
    """Build deterministic, shape-correct batches without reading training data."""
    import torch

    morphologies = (
        ("DroidSynthetic", 0),
        ("EgoDexSynthetic", 2),
        ("AgiBotSynthetic", 6),
        ("ABCSynthetic", 9),
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


def run_validation(variant: str, requested_steps: int, data_mode: str) -> dict[str, Any]:
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
        available_samples = synthetic_morphology_samples()
    if len(available_samples) != 4:
        raise RuntimeError(f"expected four configured morphologies, found {[name for name, _ in available_samples]}")

    model = instantiate(cfg.model).to(device).train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=1e-4, betas=(0.9, 0.95))

    steps = max(requested_steps, len(available_samples))
    records = []
    decoder_morphologies_seen: set[int] = set()
    group_ever_nonzero: dict[str, bool] = {}
    expected_groups = ["lora_", "forward_model.action_to_control", "action_pool", "morphology_tokens"]
    expected_groups += ["action_encoder"] if variant == "explicit" else ["inverse_model", "action_decoder"]

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
    if variant == "explicit":
        delayed_groups.append("action_encoder")
    missing_signal = [group for group in delayed_groups if not group_ever_nonzero.get(group, False)]
    if missing_signal:
        raise RuntimeError(f"no gradient reached trainable conditioning groups: {missing_signal}")
    if variant == "latent" and decoder_morphologies_seen != {0, 2, 6, 9}:
        raise RuntimeError(f"did not validate every action-decoder morphology: {sorted(decoder_morphologies_seen)}")

    return {
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "visible_device_count": torch.cuda.device_count(),
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        },
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "groups_ever_nonzero": group_ever_nonzero,
        "steps": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("latent", "explicit"), required=True)
    parser.add_argument("--data-mode", choices=("real", "synthetic"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4, help="minimum steps; always covers all four morphologies")
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
        },
    }
    try:
        payload["validation"] = run_validation(args.variant, args.steps, args.data_mode)
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
