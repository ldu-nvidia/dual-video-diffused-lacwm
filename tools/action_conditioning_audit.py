#!/usr/bin/env python3
"""Audit whether the trained explicit-action path preserves sample variation.

This is a read-only, train-split diagnostic.  It exactly replays the action
encoder, temporal pool, and Wan control projection from a model snapshot
without constructing the video model or reading RGB/future targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


SCHEMA = "lacwm-action-conditioning-audit-v1"
REQUIRED_KEYS = {
    "morphology_tokens.weight",
    "action_encoder.net.0.weight",
    "action_encoder.net.0.bias",
    "action_encoder.net.2.weight",
    "action_encoder.net.2.bias",
    "action_encoder.net.4.weight",
    "action_encoder.net.4.bias",
    "action_pool.0.weight",
    "action_pool.0.bias",
    "action_pool.2.weight",
    "action_pool.2.bias",
    "forward_model.action_to_control.net.0.weight",
    "forward_model.action_to_control.net.0.bias",
    "forward_model.action_to_control.net.2.weight",
    "forward_model.action_to_control.net.2.bias",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_absolute(path: str, *, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise ValueError(f"{label} must be a regular absolute file")
    if any(part.lower() in {"test", "tests", "protected_test"} for part in value.parts):
        raise ValueError(f"{label} may not reference a protected-test path")
    return value


def linear(value: Tensor, state: dict[str, Tensor], prefix: str) -> Tensor:
    return F.linear(
        value,
        state[f"{prefix}.weight"].float(),
        state[f"{prefix}.bias"].float(),
    )


def replay_action_path(
    state: dict[str, Tensor],
    actions: Tensor,
    *,
    morphology_index: int,
    padding_dim: int = 157,
) -> dict[str, Tensor]:
    if actions.ndim != 4 or tuple(actions.shape[1:3]) != (13, 5):
        raise ValueError("actions must have shape [N,13,5,D]")
    if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
        raise ValueError("actions must be finite floating point")
    if not 1 <= actions.shape[-1] <= padding_dim:
        raise ValueError("action width must lie in [1,padding_dim]")
    missing = REQUIRED_KEYS - set(state)
    if missing:
        raise ValueError(f"snapshot is missing action-path keys: {sorted(missing)}")
    if not 0 <= morphology_index < state["morphology_tokens.weight"].shape[0]:
        raise ValueError("morphology index is out of range")

    future = actions[:, 4:12].float()
    if future.shape[-1] < padding_dim:
        future = torch.cat(
            (
                future,
                future.new_zeros(*future.shape[:-1], padding_dim - future.shape[-1]),
            ),
            dim=-1,
        )
    morphology = state["morphology_tokens.weight"][morphology_index].float()
    morphology = morphology.reshape(1, 1, -1).expand(future.shape[0], 8, -1)
    encoded = torch.cat((future.flatten(2), morphology), dim=-1)
    encoded = F.silu(linear(encoded, state, "action_encoder.net.0"))
    encoded = F.silu(linear(encoded, state, "action_encoder.net.2"))
    encoded = linear(encoded, state, "action_encoder.net.4")

    pooled = encoded.reshape(encoded.shape[0], 2, 4 * encoded.shape[-1])
    pooled = F.silu(linear(pooled, state, "action_pool.0"))
    pooled = linear(pooled, state, "action_pool.2")
    control = F.silu(
        linear(pooled, state, "forward_model.action_to_control.net.0")
    )
    control = linear(
        control, state, "forward_model.action_to_control.net.2"
    )
    return {
        "future_actions": future,
        "encoded_actions": encoded,
        "pooled_actions": pooled,
        "control": control,
    }


def tensor_stats(value: Tensor) -> dict[str, float | list[int]]:
    flat = value.float().flatten(1)
    rolled = torch.roll(flat, shifts=-1, dims=0)
    rms = value.float().square().mean().sqrt()
    shuffled_diff = (flat - rolled).square().mean().sqrt()
    centered = value.float() - value.float().mean(dim=0, keepdim=True)
    sample_std_rms = value.float().std(dim=0, unbiased=True).square().mean().sqrt()
    cosine = F.cosine_similarity(flat, rolled, dim=1, eps=1.0e-12)
    return {
        "shape": list(value.shape),
        "rms": float(rms),
        "mean_abs": float(value.float().abs().mean()),
        "max_abs": float(value.float().abs().max()),
        "sample_std_rms": float(sample_std_rms),
        "sample_std_to_rms": float(sample_std_rms / rms.clamp_min(1.0e-12)),
        "centered_rms": float(centered.square().mean().sqrt()),
        "cyclic_shuffle_diff_rms": float(shuffled_diff),
        "cyclic_shuffle_diff_to_rms": float(
            shuffled_diff / rms.clamp_min(1.0e-12)
        ),
        "cyclic_shuffle_cosine_mean": float(cosine.mean()),
        "cyclic_shuffle_cosine_min": float(cosine.min()),
    }


def effective_rank(value: Tensor) -> float:
    centered = value.float().flatten(1)
    centered = centered - centered.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1.0e-20)
    entropy = -(probability * probability.clamp_min(1.0e-20).log()).sum()
    return float(entropy.exp())


def run(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_path = regular_absolute(args.snapshot, label="snapshot")
    actions_path = regular_absolute(args.actions, label="actions")
    snapshot_sha = sha256(snapshot_path)
    actions_sha = sha256(actions_path)
    if snapshot_sha != args.snapshot_sha256:
        raise ValueError("snapshot SHA-256 differs")
    if actions_sha != args.actions_sha256:
        raise ValueError("actions SHA-256 differs")

    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    state = snapshot.get("model")
    if not isinstance(state, dict):
        raise ValueError("snapshot lacks a model state")
    actions_array = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    actions = torch.from_numpy(np.array(actions_array, copy=True)).float()
    replay = replay_action_path(
        state,
        actions,
        morphology_index=args.morphology_index,
        padding_dim=args.padding_dim,
    )
    zero_replay = replay_action_path(
        state,
        torch.zeros_like(actions),
        morphology_index=args.morphology_index,
        padding_dim=args.padding_dim,
    )
    control = replay["control"]
    zero_control = zero_replay["control"]
    action_component = control - zero_control
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "snapshot": str(snapshot_path),
            "snapshot_sha256": snapshot_sha,
            "actions": str(actions_path),
            "actions_sha256": actions_sha,
            "snapshot_start_iter": snapshot.get("_start_iter"),
            "snapshot_run_identity_sha256": snapshot.get("run_identity_sha256"),
            "protected_test_access": False,
        },
        "contract": {
            "morphology_index": args.morphology_index,
            "padding_dim": args.padding_dim,
            "future_action_chunks": [4, 12],
            "clip_count": int(actions.shape[0]),
        },
        "stats": {
            name: tensor_stats(value)
            for name, value in (
                ("future_actions", replay["future_actions"]),
                ("encoded_actions", replay["encoded_actions"]),
                ("pooled_actions", replay["pooled_actions"]),
                ("control", control),
                ("zero_action_control", zero_control),
                ("action_dependent_control", action_component),
            )
        },
        "control_effective_rank": effective_rank(control),
        "action_dependent_control_rms_fraction": float(
            action_component.square().mean().sqrt()
            / control.square().mean().sqrt().clamp_min(1.0e-12)
        ),
        "correct_vs_zero_control_cosine_mean": float(
            F.cosine_similarity(
                control.flatten(1), zero_control.flatten(1), dim=1, eps=1.0e-12
            ).mean()
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["analysis_identity_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--snapshot", required=True)
    value.add_argument("--snapshot-sha256", required=True)
    value.add_argument("--actions", required=True)
    value.add_argument("--actions-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--morphology-index", type=int, default=9)
    value.add_argument("--padding-dim", type=int, default=157)
    return value


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
