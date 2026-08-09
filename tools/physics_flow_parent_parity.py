#!/usr/bin/env python3
"""Prove the Stage-1 parent endpoint is the native target-blind VPM sampler.

This command runs before validation evaluation.  It indexes only frames 0:5 of
one prospectively registered immutable *training* clip, then compares the
evaluation adapter bit-for-bit with a direct call to the de65 parent's public
``sample_future_deployable`` method at NFE 1/2/4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import physics_flow_stage1 as stage  # noqa: E402
from tools.physics_flow_parent_vpm import (  # noqa: E402
    NativeParentVPMError,
    extract_native_parent_artifacts,
    load_exact_parent_model,
    materialize_native_parent_endpoint,
    set_native_parent_endpoint,
)


class ParentParityError(RuntimeError):
    """The target-blind native-parent parity contract failed."""


def _torch_tensor_hash(value: Any) -> str:
    """Hash a tensor without the scalar ``view(uint8)`` failure mode."""

    import torch

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(stage.canonical_json(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _registered_train_input(
    registration: Mapping[str, Any], device: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    metadata = stage.load_cache_metadata(
        Path(registration["flow_caches"]["train"]["metadata"]["path"]),
        split="train",
    )
    cache_registration = stage.read_json(
        Path(metadata["cache_registration"]["path"]), "cache registration"
    )
    immutable = cache_registration["inputs"]["train"]["cache"]["arrays"]
    rgb = np.load(immutable["rgb"]["path"], mmap_mode="r", allow_pickle=False)
    actions = np.load(
        immutable["actions"]["path"], mmap_mode="r", allow_pickle=False
    )
    if (
        rgb.shape != (stage.TRAIN_COUNT, 13, 3, 180, 960)
        or rgb.dtype != np.float16
        or actions.shape != (stage.TRAIN_COUNT, 13, 5, 23)
        or actions.dtype != np.float32
    ):
        raise ParentParityError("immutable train input geometry differs")
    index = stage.PARENT_PARITY_TRAIN_INDEX
    # This is the only RGB indexing operation in the command.  Frames 5:13 are
    # neither sliced nor hashed, so the parity proof is target-blind.
    history = torch.from_numpy(np.array(rgb[index, 0:5], copy=True)).float()
    action = torch.from_numpy(np.array(actions[index], copy=True))
    action = torch.cat(
        (
            action,
            torch.zeros(
                *action.shape[:-1],
                157 - action.shape[-1],
                dtype=action.dtype,
            ),
        ),
        dim=-1,
    )
    morphology = torch.tensor([9], dtype=torch.long)
    sample_ids = torch.tensor(
        [stage.PARENT_PARITY_SAMPLE_ID], dtype=torch.long
    )
    descriptors = cache_registration["splits"]["train"]["descriptors"]
    descriptor = descriptors[index]
    batch = {
        "history_rgb": history.unsqueeze(0).to(device=device),
        "actions": action.unsqueeze(0).to(device=device),
        "morphology_index": morphology.to(device=device),
        "sample_ids": sample_ids.to(device=device),
    }
    evidence = {
        "split": "train",
        "clip_index": index,
        "clip_id": descriptor["clip_id"],
        "rgb_indexed_frames_half_open": [0, 5],
        "rgb_future_frames_indexed": False,
        "actions_indexed_frames_half_open": [0, 13],
        "sample_id": stage.PARENT_PARITY_SAMPLE_ID,
        "history_rgb_sha256": _torch_tensor_hash(batch["history_rgb"]),
        "actions_sha256": _torch_tensor_hash(batch["actions"]),
        "morphology_index_sha256": _torch_tensor_hash(
            batch["morphology_index"]
        ),
        "immutable_rgb_array": dict(immutable["rgb"]),
        "immutable_actions_array": dict(immutable["actions"]),
        "future_rgb_bytes_opened": False,
        "future_measured_state_opened": False,
        "protected_test_accessed": False,
    }
    return batch, evidence


def _direct_native_call(
    model: Any,
    *,
    batch: Mapping[str, Any],
    nfe: int,
    expected_noise: Any,
    hook_counter: list[int],
) -> dict[str, Any]:
    import torch

    set_native_parent_endpoint(model, nfe)
    before = hook_counter[0]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=batch["history_rgb"].is_cuda,
    ):
        predicted = model.sample_future_deployable(
            batch["history_rgb"],
            batch["actions"],
            morphology_index=batch["morphology_index"],
            collect_artifacts=True,
            sample_ids=batch["sample_ids"],
        )
    result = extract_native_parent_artifacts(
        model,
        nfe=nfe,
        predicted_future=predicted,
        expected_video_noise=expected_noise,
    )
    result["hook_wan_calls"] = hook_counter[0] - before
    return result


def command_parity(args: argparse.Namespace) -> int:
    import torch

    registration = stage.validate_study_registration(args.registration)
    canonical_output = (
        Path(registration["output_root"]) / stage.PARENT_PARITY_FILENAME
    )
    if args.output.expanduser().absolute() != canonical_output:
        raise ParentParityError(f"parity output must be {canonical_output}")
    if canonical_output.exists() or canonical_output.is_symlink():
        raise ParentParityError("fresh parent parity output already exists")
    if not torch.cuda.is_available():
        raise ParentParityError("parent parity requires a B200 GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise ParentParityError("parent parity requires a B200 GPU")

    # Train history is opened before model sampling; no validation dataset or
    # train future RGB is ever indexed by this command.
    batch, input_evidence = _registered_train_input(registration, device)
    model, parent_artifacts = load_exact_parent_model(registration, device)
    hook_counter = [0]

    def count_call(_module: Any, _inputs: Any, _output: Any) -> None:
        hook_counter[0] += 1

    handle = model.forward_model.register_forward_hook(count_call)
    comparisons = []
    try:
        for nfe in stage.NFE_GRID:
            expected_noise = model._evaluation_noise(
                (1, 16, 4, 24, 120),
                device=device,
                dtype=torch.float32,
                base_seed=model.evaluation_noise_seed,
                sample_ids=batch["sample_ids"],
                stream=0,
                rank=0,
            )
            direct = _direct_native_call(
                model,
                batch=batch,
                nfe=nfe,
                expected_noise=expected_noise,
                hook_counter=hook_counter,
            )
            before = hook_counter[0]
            adapted = materialize_native_parent_endpoint(
                model,
                history_rgb=batch["history_rgb"],
                actions=batch["actions"],
                morphology_index=batch["morphology_index"],
                sample_ids=batch["sample_ids"],
                nfe=nfe,
                expected_video_noise=expected_noise,
            )
            adapted_hook_calls = hook_counter[0] - before
            latent_bitwise = torch.equal(
                direct["video_latent"], adapted["video_latent"]
            )
            decoded_bitwise = torch.equal(
                direct["decoded_uint8"], adapted["decoded_uint8"]
            )
            initial_noise_bitwise = torch.equal(
                direct["video_initial_state_fp16"],
                adapted["video_initial_state_fp16"],
            )
            if (
                direct["hook_wan_calls"] != nfe
                or direct["wan_calls"] != nfe
                or adapted_hook_calls != nfe
                or adapted["wan_calls"] != nfe
                or adapted[
                    "full_precision_initial_noise_bitwise_equal"
                ]
                is not True
                or not latent_bitwise
                or not decoded_bitwise
                or not initial_noise_bitwise
            ):
                raise ParentParityError(
                    f"native parent adapter is not bitwise at NFE {nfe}"
                )
            comparisons.append(
                {
                    "nfe": nfe,
                    "direct_native_wan_calls": direct["hook_wan_calls"],
                    "adapter_wan_calls": adapted_hook_calls,
                    "latent_bitwise_equal": latent_bitwise,
                    "decoded_uint8_bitwise_equal": decoded_bitwise,
                    "initial_video_noise_fp16_bitwise_equal": (
                        initial_noise_bitwise
                    ),
                    "adapter_full_precision_initial_noise_bitwise_equal": True,
                    "explicit_video_noise_sha256": _torch_tensor_hash(
                        expected_noise
                    ),
                    "direct_latent_sha256": _torch_tensor_hash(
                        direct["video_latent"]
                    ),
                    "adapter_latent_sha256": _torch_tensor_hash(
                        adapted["video_latent"]
                    ),
                    "direct_decoded_sha256": _torch_tensor_hash(
                        direct["decoded_uint8"]
                    ),
                    "adapter_decoded_sha256": _torch_tensor_hash(
                        adapted["decoded_uint8"]
                    ),
                }
            )
    finally:
        handle.remove()

    receipt = stage.identity_payload(
        {
            "schema_version": stage.SCHEMA_VERSION,
            "kind": stage.PARENT_PARITY_KIND,
            "status": "bitwise_native_parent_sampler_parity_passed",
            "registration_identity_sha256": registration["identity_sha256"],
            "parent": parent_artifacts,
            "public_reference_method": "sample_future_deployable",
            "evaluation_adapter_method": (
                "materialize_native_parent_endpoint"
            ),
            "condition_source": "off",
            "schedule_mode": "aligned",
            "nfe_grid": list(stage.NFE_GRID),
            "input": input_evidence,
            "comparisons": comparisons,
            "bitwise_parity_required": True,
            "all_bitwise": True,
            "validation_dataset_opened": False,
            "future_rgb_bytes_opened": False,
            "future_measured_state_opened": False,
            "online_teacher_or_feature_calls": 0,
            "protected_test_accessed": False,
        }
    )
    stage.exclusive_json(canonical_output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return command_parity(args)
    except (
        ParentParityError,
        NativeParentVPMError,
        stage.PhysicsFlowStage1Error,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
