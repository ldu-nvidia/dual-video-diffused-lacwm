#!/usr/bin/env python3
"""One-rank, full-geometry, three-copy ACD-P0 peak-memory preflight."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "latent_action_models"
for value in (str(ROOT), str(PROJECT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from tools import low_nfe_direct_baseline as pilot  # noqa: E402


SYNTHETIC_FIXTURE_VERSION = "acd-p0-full-support-v2"


def _synthetic_full_geometry_fixture(torch, *, device):
    """Return a bounded, dataset-free 13-frame clip with three valid views."""

    y = torch.linspace(-1.0, 1.0, 180, device=device, dtype=torch.float32)[
        :, None
    ]
    x = torch.linspace(-1.0, 1.0, 320, device=device, dtype=torch.float32)[
        None, :
    ]
    views = torch.cat(
        (
            0.25 * (x + y),
            0.25 * (x - y),
            0.30 * x + 0.10 * y,
        ),
        dim=-1,
    ).reshape(1, 1, 1, 180, 960)
    frame_offsets = torch.linspace(
        -0.10, 0.10, 13, device=device, dtype=torch.float32
    ).reshape(1, 13, 1, 1, 1)
    channel_offsets = torch.tensor(
        (-0.05, 0.0, 0.05), device=device, dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    rgb = views + frame_offsets + channel_offsets
    temporal_mask = torch.ones((1, 13), device=device, dtype=torch.bool)
    if (
        tuple(rgb.shape) != (1, 13, 3, 180, 960)
        or not bool(torch.isfinite(rgb).all())
        or float(rgb.min()) < -1.0
        or float(rgb.max()) > 1.0
        or not bool(temporal_mask.all())
    ):
        raise pilot.ACDPilotError("synthetic full-support fixture construction failed")
    return rgb, temporal_mask


def command_smoke(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from robot_wm.modeling.low_nfe.adjacent_consistency import (
        CANONICAL_MODEL_STATE_HASH_ALGORITHM,
        RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
        _expanded_mask,
        ema_update_module_,
        require_model_state_hashes,
        tensor_state_sha256,
    )

    source = pilot.clean_source(args.source_repo, args.expected_commit)
    if args.output.exists() or args.output.is_symlink():
        raise pilot.ACDPilotError("memory receipt output must be fresh")
    if not args.output.is_absolute() or not args.output.parent.is_dir():
        raise pilot.ACDPilotError("memory receipt requires an existing absolute parent")
    parent_record = pilot.file_record(args.parent_snapshot)
    config_record = pilot.file_record(args.parent_resolved_config)
    if parent_record["sha256"] != pilot.PARENT_SNAPSHOT_SHA256:
        raise pilot.ACDPilotError("memory smoke parent snapshot differs")
    if config_record["sha256"] != pilot.PARENT_RESOLVED_CONFIG_SHA256:
        raise pilot.ACDPilotError("memory smoke parent config differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise pilot.ACDPilotError("memory smoke requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if "B200" not in device_name.upper():
        raise pilot.ACDPilotError(f"memory smoke requires B200, got {device_name}")
    os.environ["WAN_DIR"] = str(args.wan_dir.resolve(strict=True))
    os.environ["VIDEOX_HOME"] = str(args.videox_home.resolve(strict=True))
    os.environ["ACD_P0_TRAIN_CLIP_MANIFEST"] = "/not-opened/acd/train.jsonl"
    os.environ["ACD_P0_TRAIN_CACHE_METADATA"] = "/not-opened/acd/train.json"
    os.environ["ACD_P0_PARENT_SNAPSHOT"] = str(args.parent_snapshot.resolve(strict=True))
    os.environ["ACD_P0_PARENT_RESOLVED_CONFIG"] = str(
        args.parent_resolved_config.resolve(strict=True)
    )
    os.environ["ACD_P0_RUN_ROOT"] = "/not-created/acd"

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    config_root = PROJECT / "configs"
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "+experiments_0908=ravenhuang/wan-dit/adjacent_consistency_candidate"
            ],
        )
    model = instantiate(config.model)
    snapshot = torch.load(
        parent_record["path"], map_location="cpu", weights_only=True, mmap=True
    )
    parent_state = snapshot.get("model")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("run_identity_sha256") != pilot.PARENT_RUN_IDENTITY_SHA256
        or snapshot.get("_start_iter") != 1000
        or not isinstance(parent_state, dict)
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise pilot.ACDPilotError("memory smoke parent metadata differs")
    if (
        CANONICAL_MODEL_STATE_HASH_ALGORITHM
        != pilot.PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM
        or RUNTIME_TENSOR_STATE_HASH_ALGORITHM
        != pilot.PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM
    ):
        raise pilot.ACDPilotError("memory smoke hash algorithm identity differs")
    try:
        parent_state_hash_receipt = require_model_state_hashes(
            parent_state,
            expected_canonical_sha256=pilot.PARENT_CANONICAL_MODEL_STATE_SHA256,
            expected_runtime_sha256=pilot.PARENT_RUNTIME_TENSOR_STATE_SHA256,
            label="memory smoke raw parent model state",
        )
    except Exception as exc:
        raise pilot.ACDPilotError("memory smoke raw parent hashes differ") from exc
    incompatible = model.load_state_dict(parent_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise pilot.ACDPilotError("memory smoke strict parent load failed")
    loaded_student_runtime_hash = tensor_state_sha256(model.state_dict())
    if loaded_student_runtime_hash != pilot.PARENT_RUNTIME_TENSOR_STATE_SHA256:
        raise pilot.ACDPilotError("memory smoke loaded student runtime state differs")
    del snapshot, parent_state

    model = model.to(device=device)
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    strict_loaded_runtime_hashes = {
        "student": tensor_state_sha256(model.state_dict()),
        "teacher": tensor_state_sha256(teacher.state_dict()),
        "ema_target": tensor_state_sha256(target.state_dict()),
    }
    if any(
        value != pilot.PARENT_RUNTIME_TENSOR_STATE_SHA256
        for value in strict_loaded_runtime_hashes.values()
    ):
        raise pilot.ACDPilotError(
            "memory smoke loaded student/teacher/EMA runtime state differs"
        )
    model.bind_frozen_models(teacher=teacher, target=target)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=5e-6,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rgb, mask = _synthetic_full_geometry_fixture(torch, device=device)
    actions = torch.zeros((1, 13, 5, 157), device=device, dtype=torch.float32)
    morphology = torch.tensor([9], device=device, dtype=torch.long)
    clip_index = torch.tensor([0], device=device, dtype=torch.long)
    latent_geometry = (1, 16, 4, 24, 120)
    loss_mask = model._build_loss_mask(rgb, mask, latent_geometry)
    future_mask = loss_mask[:, :, 2:]
    future_target = torch.zeros(
        (1, 16, 2, 24, 120), device=device, dtype=torch.float32
    )
    expanded_future_mask = _expanded_mask(future_mask, future_target)
    per_view_std = (
        rgb.reshape(1, 13, 3, 180, 3, 320)
        .permute(0, 4, 1, 2, 3, 5)
        .reshape(1, 3, -1)
        .std(dim=-1)
    )
    synthetic_support = {
        "fixture_version": SYNTHETIC_FIXTURE_VERSION,
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "rgb_min": float(rgb.min()),
        "rgb_max": float(rgb.max()),
        "minimum_view_std": float(per_view_std.min()),
        "valid_views": int((per_view_std > 1e-3).sum()),
        "temporal_mask_shape": list(mask.shape),
        "temporal_mask_dtype": str(mask.dtype),
        "temporal_valid_frames": int(mask.sum()),
        "history_valid_frames": int(mask[:, :5].sum()),
        "future_valid_frames": int(mask[:, 5:].sum()),
        "latent_loss_mask_shape": list(loss_mask.shape),
        "history_latent_tokens": 2,
        "future_latent_tokens": int(future_mask.shape[2]),
        "expanded_future_support_per_sample": [
            int(value)
            for value in expanded_future_mask.sum(dim=(1, 2, 3, 4)).tolist()
        ],
        "production_build_loss_mask_exercised": True,
        "expanded_mask_exercised": True,
        "dataset_accessed": False,
    }
    del loss_mask, future_mask, future_target, expanded_future_mask, per_view_std
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss = model(
            rgb,
            actions=actions,
            mask=mask,
            morphology_index=morphology,
            clip_index=clip_index,
        )
    if not bool(torch.isfinite(loss)):
        raise pilot.ACDPilotError("memory smoke loss is non-finite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=1.0, norm_type=2.0, error_if_nonfinite=True
    )
    optimizer.step()
    receipt = ema_update_module_(target, model, decay=0.995)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    headroom = total_memory - peak_reserved
    reserved_fraction = peak_reserved / total_memory
    call_counts = dict(model._acd_call_counts)
    passed = (
        call_counts
        == {"teacher": 1, "online_student": 1, "ema_target": 1}
        and reserved_fraction <= 0.90
        and headroom >= 16 * (1 << 30)
    )
    payload = pilot.identity_payload(
        {
            "kind": "acd_p0_memory_smoke_receipt",
            "schema_version": 2,
            "status": "PASS" if passed else "FAIL",
            "source_commit": source["git_commit"],
            "source_tree_sha": source["git_tree_sha"],
            "parent_snapshot_sha256": parent_record["sha256"],
            "parent_resolved_config_sha256": config_record["sha256"],
            "parent_model_state_hash_receipt": parent_state_hash_receipt,
            "strict_loaded_runtime_tensor_state_sha256": (
                strict_loaded_runtime_hashes
            ),
            "device_name": device_name,
            "device_total_memory_bytes": total_memory,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_fraction": reserved_fraction,
            "headroom_bytes": headroom,
            "model_copies": 3,
            "synthetic_full_geometry": [1, 13, 3, 180, 960],
            "synthetic_support": synthetic_support,
            "teacher_calls": call_counts.get("teacher", 0),
            "online_calls": call_counts.get("online_student", 0),
            "ema_target_calls": call_counts.get("ema_target", 0),
            "loss": float(loss.detach()),
            "gradient_norm": float(grad_norm.detach()),
            "backward_completed": True,
            "optimizer_step_completed": True,
            "ema_step_completed": True,
            "ema_parameter_values": receipt.ema_parameter_values,
            "exact_copied_parameter_values": receipt.copied_parameter_values,
            "training_data_opened": False,
            "synthetic_input_only": True,
            "outcomes_opened": 0,
            "wandb_writes": 0,
            "jobs_submitted": 0,
            "protected_test_accessed": False,
        }
    )
    pilot.exclusive_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise pilot.ACDPilotError("three-copy B200 memory preflight failed")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument("--parent-resolved-config", type=Path, required=True)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--videox-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
