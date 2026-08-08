#!/usr/bin/env python3
"""Fail-closed GPU canary for the VideoREPA TRD deployment sampler.

The canary uses one immutable TRAIN clip and one Wan evaluation per path. It verifies
that the custom, auxiliary-free deployment path is numerically identical to
the historical VPM condition-off sampler when both start from the same keyed
noise.  Validation data and clean V-JEPA targets are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
for root in (str(REPO_ROOT), str(PROJECT_ROOT), str(REPO_ROOT / "tools")):
    if root not in sys.path:
        sys.path.insert(0, root)

import videorepa_trd_screen as contract  # noqa: E402


CANARY_SCHEMA_VERSION = 1
CANARY_KIND = "videorepa_trd_deployment_sampler_canary"
CANARY_DIRECTORY = "deployment_sampler_canary"
CANARY_REPORT = "report.json"
AUXILIARY_MODULE_NAMES = (
    "control_adapter",
    "tf_projection",
    "tf_norm",
    "tf_clock_net",
    "tf_velocity_head",
)
ZERO_AUXILIARY_MODULE_CALLS = {name: 0 for name in AUXILIARY_MODULE_NAMES}
ORDINARY_VPM_AUXILIARY_MODULE_CALLS = {
    name: 1 for name in AUXILIARY_MODULE_NAMES
}
EXPECTED_COMPARISON_SPECS = {
    "fixed_noise": ([1, 16, 4, 24, 120], "torch.float16"),
    "history_reference_latents": ([1, 16, 4, 24, 120], "torch.float16"),
    "native_video_velocity": ([1, 16, 4, 24, 120], "torch.bfloat16"),
    "final_video_latent_fp16_evidence": (
        [1, 16, 4, 24, 120],
        "torch.float16",
    ),
    "future_output": ([1, 3, 8, 180, 960], "torch.float32"),
    "decoded_future_uint8": ([1, 3, 8, 180, 960], "torch.uint8"),
}


class TRDDeploymentCanaryError(RuntimeError):
    """The deployment sampler or its reference equivalence changed."""


def _tensor_sha256(value) -> str:
    """Hash the exact logical tensor bytes, including scalar tensors safely."""

    contiguous = value.detach().cpu().contiguous().reshape(-1)
    octets = contiguous.view(__import__("torch").uint8)
    return hashlib.sha256(octets.numpy().tobytes(order="C")).hexdigest()


def _comparison(left, right) -> dict[str, Any]:
    import torch

    if left.shape != right.shape or left.dtype != right.dtype:
        raise TRDDeploymentCanaryError(
            "custom/reference output shape or dtype differs: "
            f"{tuple(left.shape)}/{left.dtype} != "
            f"{tuple(right.shape)}/{right.dtype}"
        )
    difference = left.detach().float() - right.detach().float()
    return {
        "shape": [int(value) for value in left.shape],
        "dtype": str(left.dtype),
        "custom_sha256": _tensor_sha256(left),
        "ordinary_vpm_sha256": _tensor_sha256(right),
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
    }


def _run_counted(model, operation: Callable[[], Any]):
    calls = 0
    auxiliary_module_calls = dict(ZERO_AUXILIARY_MODULE_CALLS)
    captured_velocities = []

    def count_wan(_module, _inputs, output):
        nonlocal calls
        calls += 1
        velocity = output[0] if isinstance(output, (list, tuple)) else output
        if not hasattr(velocity, "detach"):
            raise TRDDeploymentCanaryError("Wan hook did not observe a tensor")
        captured_velocities.append(velocity.detach().clone())

    def count_auxiliary(name):
        def callback(_module, _inputs, _output):
            auxiliary_module_calls[name] += 1

        return callback

    before = int(model._trd_forward_hook_installations)
    handles = [
        model.forward_model.transformer.register_forward_hook(count_wan),
        model.forward_model.transformer.control_adapter.register_forward_hook(
            count_auxiliary("control_adapter")
        ),
        model.forward_model.tf_token_adapter.projection.register_forward_hook(
            count_auxiliary("tf_projection")
        ),
        model.forward_model.tf_token_adapter.norm.register_forward_hook(
            count_auxiliary("tf_norm")
        ),
        model.forward_model.tf_clock_embedding.net.register_forward_hook(
            count_auxiliary("tf_clock_net")
        ),
        model.forward_model.tf_velocity_head.register_forward_hook(
            count_auxiliary("tf_velocity_head")
        ),
    ]
    try:
        result = operation()
    finally:
        for handle in handles:
            handle.remove()
    after = int(model._trd_forward_hook_installations)
    if len(captured_velocities) != calls:
        raise TRDDeploymentCanaryError("Wan native-velocity capture count differs")
    counts = {
        "wan_calls": calls,
        "auxiliary_branch_calls": sum(auxiliary_module_calls.values()),
        "auxiliary_module_calls": auxiliary_module_calls,
        "trd_hook_installations": after - before,
    }
    return result, counts, captured_velocities


def _compose_config(registration: Mapping[str, Any]):
    from hydra import compose, initialize_config_dir

    arm = contract.ARM_BY_CODE["TRD-ON"]
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"), version_base=None
    ):
        config = compose(
            config_name="train",
            overrides=[
                f"+experiments_0908={arm.config_name}",
                f"+wandb.id={contract.arm_identity(registration, arm)}",
                "+wandb.resume=never",
            ],
        )
    # Reuse the sealed evaluator's complete architecture/config guard.  The
    # canary composes the same prospective config before either arm trains.
    from evaluate_videorepa_trd import _validate_config

    _validate_config(config, arm, registration)
    return config


def _load_target_free_train_clip(config):
    from hydra.utils import instantiate
    from torch.utils.data import default_collate

    dataset_config = config.dataset
    dataset_config.infinite = False
    dataset_config.future_validity.enabled = False
    dataset_config.future_validity.max_retries = 0
    dataset_config.datasets.ABC._target_ = (
        "robot_wm.datasets.abc.fixed_rgb_action_dataset."
        "ABCFixedRGBActionDataset"
    )
    dataset_config.datasets.ABC.infinite = False
    dataset = instantiate(dataset_config)
    if len(dataset) != contract.TRAIN_CLIPS:
        raise TRDDeploymentCanaryError("target-free train512 length differs")
    sample = default_collate([dataset[0]])
    child = dataset.datasets["ABC"]
    if getattr(child, "_targets", None) is not None:
        raise TRDDeploymentCanaryError("canary opened a clean V-JEPA target array")
    if int(sample["clip_index"].item()) != 0:
        raise TRDDeploymentCanaryError("canary did not select immutable train clip 0")
    return dataset, sample


def _load_parent_model(config, registration, device):
    import torch
    from hydra.utils import instantiate

    model = instantiate(config.model)
    snapshot = torch.load(
        registration["parent_snapshot"]["file"]["path"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(snapshot.get("model"), Mapping):
        raise TRDDeploymentCanaryError("parent snapshot model state is missing")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TRDDeploymentCanaryError(
            f"strict parent snapshot load failed: {incompatible}"
        )
    del snapshot
    model = model.to(device=device).eval()
    if (
        getattr(model.forward_model.transformer, "teacache", None) is not None
        or int(getattr(model.forward_model.transformer, "sp_world_size", 1)) != 1
    ):
        raise TRDDeploymentCanaryError(
            "canary requires stateless Wan calls without sequence parallelism"
        )
    if any(
        "token_relation" in name or "trd" in name.lower()
        for name, _parameter in model.named_parameters()
    ):
        raise TRDDeploymentCanaryError("TRD added an inference parameter")
    return model


def _run_canary(registration: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise TRDDeploymentCanaryError("deployment canary requires CUDA")
    if os.environ.get("WANDB_MODE") != "disabled":
        raise TRDDeploymentCanaryError("deployment canary requires W&B disabled")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise TRDDeploymentCanaryError("deployment canary must be single-process")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config = _compose_config(registration)
    dataset, raw_batch = _load_target_free_train_clip(config)
    model = _load_parent_model(config, registration, device)
    model.evaluation_condition_sources = ("off",)
    model.evaluation_nfe_steps = (1,)
    model.viz_num_steps = 1
    model.artifact_batch_limit = None
    model.capture_latent_trajectories = False

    history = raw_batch["rgb"][:, : model.num_history_frames].clone(
        memory_format=torch.contiguous_format
    ).to(device)
    actions = raw_batch["actions"].to(device)
    morphology = raw_batch.get("morphology_index")
    if morphology is not None:
        morphology = morphology.to(device)
    sample_ids = raw_batch["clip_index"].to(device)
    action_before = _tensor_sha256(actions[0])
    history_is_owned = (
        history.untyped_storage().nbytes()
        == history.numel() * history.element_size()
    )
    torch.cuda.reset_peak_memory_stats(device)

    def custom_operation():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            return model.sample_future_deployable(
                history,
                actions,
                morphology,
                collect_artifacts=True,
                sample_ids=sample_ids,
            )

    custom_future, custom_counts, custom_velocities = _run_counted(
        model, custom_operation
    )
    custom_artifacts = model.pop_visualization_artifacts()
    if not isinstance(custom_artifacts, Mapping):
        raise TRDDeploymentCanaryError("custom deployment artifacts are missing")

    # Invoke the inherited VPM sampler directly so this reference necessarily
    # executes its parameter-matched auxiliary machinery with both condition
    # switches off.  It uses the same keyed video noise and one-step scheduler.
    from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel

    def ordinary_vpm_operation():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            predicted, ground_truth = DualExplicitActionDiTModel._sample_future(
                model,
                history,
                actions,
                morphology,
                auxiliary_target=None,
                collect_artifacts=True,
                deployment_mode=True,
                sample_ids=sample_ids,
            )
        if ground_truth is not None:
            raise TRDDeploymentCanaryError(
                "ordinary condition-off reference constructed future ground truth"
            )
        return predicted[:, :, -model.num_future_frames :]

    ordinary_future, ordinary_counts, ordinary_velocities = _run_counted(
        model, ordinary_vpm_operation
    )
    ordinary_artifacts = model.pop_visualization_artifacts()
    if not isinstance(ordinary_artifacts, Mapping):
        raise TRDDeploymentCanaryError("ordinary VPM artifacts are missing")
    torch.cuda.synchronize(device)

    action_after = _tensor_sha256(actions[0])
    future_comparison = _comparison(custom_future, ordinary_future)
    initial_comparison = _comparison(
        custom_artifacts["video_initial_state"],
        ordinary_artifacts["video_initial_state"],
    )
    reference_comparison = _comparison(
        custom_artifacts["reference_latents"],
        ordinary_artifacts["reference_latents"],
    )
    native_velocity_comparison = _comparison(
        custom_velocities[0], ordinary_velocities[0]
    )
    final_comparison = _comparison(
        custom_artifacts["video_final_off_nfe_1"],
        ordinary_artifacts["video_final_off_nfe_1"],
    )
    decoded_comparison = _comparison(
        custom_artifacts["decoded_future_off_nfe_1"],
        ordinary_artifacts["decoded_future_off_nfe_1"],
    )
    if (
        custom_counts
        != {
            "wan_calls": 1,
            "auxiliary_branch_calls": 0,
            "auxiliary_module_calls": ZERO_AUXILIARY_MODULE_CALLS,
            "trd_hook_installations": 0,
        }
        or ordinary_counts
        != {
            "wan_calls": 1,
            "auxiliary_branch_calls": len(AUXILIARY_MODULE_NAMES),
            "auxiliary_module_calls": ORDINARY_VPM_AUXILIARY_MODULE_CALLS,
            "trd_hook_installations": 0,
        }
        or action_before != action_after
        or not history_is_owned
        or any(
            not comparison["bitwise_equal"]
            or comparison["max_abs_error"] != 0.0
            or comparison["mean_abs_error"] != 0.0
            for comparison in (
                future_comparison,
                initial_comparison,
                reference_comparison,
                native_velocity_comparison,
                final_comparison,
                decoded_comparison,
            )
        )
        or int(custom_artifacts["deployment_mode"].item()) != 1
        or int(custom_artifacts["auxiliary_clean_available"].item()) != 0
        or int(custom_artifacts["online_teacher_call_count"].item()) != 0
        or int(custom_artifacts["trd_inference_branch_call_count"].item()) != 0
        or int(ordinary_artifacts["online_teacher_call_count"].item()) != 0
        or "video_clean" in custom_artifacts
        or "tf_clean" in custom_artifacts
        or "ground_truth_future_uint8" in custom_artifacts
        or getattr(dataset.datasets["ABC"], "_targets", None) is not None
    ):
        raise TRDDeploymentCanaryError(
            "deployment sampler failed its call/leakage/equivalence contract"
        )

    return contract._identity(
        {
            "schema_version": CANARY_SCHEMA_VERSION,
            "kind": CANARY_KIND,
            "status": "passed_before_full_arm_training",
            "registration_identity_sha256": registration["identity_sha256"],
            "source_git_commit": registration["source"]["git_commit"],
            "parent_snapshot_sha256": registration["parent_snapshot"]["file"][
                "sha256"
            ],
            "clip_split": "train",
            "clip_index": 0,
            "nfe": 1,
            "condition_source": "off",
            "sample_id": int(sample_ids.item()),
            "history_tensor_sha256": _tensor_sha256(history[0]),
            "history_tensor_shape": [int(value) for value in history[0].shape],
            "history_tensor_dtype": str(history.dtype),
            "history_input_owned_storage": history_is_owned,
            "action_tensor_sha256": action_before,
            "action_tensor_shape": [int(value) for value in actions[0].shape],
            "action_tensor_dtype": str(actions.dtype),
            "action_tensor_unchanged": action_before == action_after,
            "custom_deployment_counts": custom_counts,
            "ordinary_vpm_condition_off_counts": ordinary_counts,
            "fixed_noise": initial_comparison,
            "history_reference_latents": reference_comparison,
            "native_video_velocity": native_velocity_comparison,
            "future_output": future_comparison,
            "final_video_latent_fp16_evidence": final_comparison,
            "decoded_future_uint8": decoded_comparison,
            "numerical_tolerance": {"rtol": 0.0, "atol": 0.0},
            "custom_path_received_future_rgb": False,
            "custom_path_received_clean_feature": False,
            "target_array_opened": False,
            "validation_split_accessed": False,
            "protected_test_accessed": False,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "passed": True,
            "created_at": contract._now(),
        }
    )


def _validate_report(
    registration: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    comparisons = {
        key: report.get(key) for key in EXPECTED_COMPARISON_SPECS
    }
    digests = (
        report.get("history_tensor_sha256"),
        report.get("action_tensor_sha256"),
    )
    if (
        not contract._valid_identity(report)
        or report.get("schema_version") != CANARY_SCHEMA_VERSION
        or report.get("kind") != CANARY_KIND
        or report.get("status") != "passed_before_full_arm_training"
        or report.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or report.get("source_git_commit")
        != registration["source"]["git_commit"]
        or report.get("parent_snapshot_sha256")
        != registration["parent_snapshot"]["file"]["sha256"]
        or report.get("clip_split") != "train"
        or report.get("clip_index") != 0
        or report.get("sample_id") != 0
        or report.get("nfe") != 1
        or report.get("condition_source") != "off"
        or report.get("history_tensor_shape") != [5, 3, 180, 960]
        or report.get("history_tensor_dtype") != "torch.float32"
        or report.get("history_input_owned_storage") is not True
        or report.get("action_tensor_unchanged") is not True
        or report.get("action_tensor_shape")
        != list(contract.ACTION_SAMPLE_SHAPE)
        or report.get("action_tensor_dtype") != contract.ACTION_TENSOR_DTYPE
        or report.get("custom_deployment_counts")
        != {
            "wan_calls": 1,
            "auxiliary_branch_calls": 0,
            "auxiliary_module_calls": ZERO_AUXILIARY_MODULE_CALLS,
            "trd_hook_installations": 0,
        }
        or report.get("ordinary_vpm_condition_off_counts")
        != {
            "wan_calls": 1,
            "auxiliary_branch_calls": len(AUXILIARY_MODULE_NAMES),
            "auxiliary_module_calls": ORDINARY_VPM_AUXILIARY_MODULE_CALLS,
            "trd_hook_installations": 0,
        }
        or report.get("numerical_tolerance") != {"rtol": 0.0, "atol": 0.0}
        or report.get("custom_path_received_future_rgb") is not False
        or report.get("custom_path_received_clean_feature") is not False
        or report.get("target_array_opened") is not False
        or report.get("validation_split_accessed") is not False
        or report.get("protected_test_accessed") is not False
        or report.get("passed") is not True
        or any(contract.SHA_RE.fullmatch(str(value or "")) is None for value in digests)
        or any(not isinstance(value, Mapping) for value in comparisons.values())
    ):
        raise TRDDeploymentCanaryError("deployment canary report differs")
    for name, comparison in comparisons.items():
        expected_shape, expected_dtype = EXPECTED_COMPARISON_SPECS[name]
        if (
            comparison.get("shape") != expected_shape
            or comparison.get("dtype") != expected_dtype
            or comparison.get("bitwise_equal") is not True
            or float(comparison.get("max_abs_error", -1.0)) != 0.0
            or float(comparison.get("mean_abs_error", -1.0)) != 0.0
            or contract.SHA_RE.fullmatch(
                str(comparison.get("custom_sha256", ""))
            )
            is None
            or comparison.get("custom_sha256")
            != comparison.get("ordinary_vpm_sha256")
        ):
            raise TRDDeploymentCanaryError(
                "deployment canary numerical equivalence differs"
            )


def _report_path(registration: Mapping[str, Any]) -> Path:
    return Path(registration["study_root"]) / CANARY_DIRECTORY / CANARY_REPORT


def command_run(args: argparse.Namespace) -> int:
    registration = contract.load_registration(args.registration)
    output = _report_path(registration)
    if output.parent.exists() or output.parent.is_symlink():
        raise TRDDeploymentCanaryError(
            "deployment canary output directory must be fresh"
        )
    output.parent.mkdir(mode=0o700)
    report = _run_canary(registration)
    _validate_report(registration, report)
    contract._exclusive_json(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    registration = contract.load_registration(args.registration)
    output = _report_path(registration)
    report = contract._read_json(output, "deployment sampler canary")
    _validate_report(registration, report)
    print(
        json.dumps(
            {
                "deployment_sampler_canary": "verified",
                "report": str(output),
                "report_identity_sha256": report["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    for command, function in (("run", command_run), ("verify", command_verify)):
        child = subparsers.add_parser(command)
        child.add_argument("--registration", type=Path, required=True)
        child.set_defaults(func=function)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
