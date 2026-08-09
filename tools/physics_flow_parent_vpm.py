#!/usr/bin/env python3
"""Exact evaluation-only adapter for the untouched de65 VPM parent.

The adapter deliberately delegates generation to the parent's public
``sample_future_deployable`` method.  It accepts only observed RGB history,
candidate actions, morphology, and immutable sample IDs.  A separate
pre-evaluation parity command compares this adapter with a direct invocation of
that public method on a registered train-history input before validation RGB is
opened.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import physics_flow_stage1 as stage  # noqa: E402


class NativeParentVPMError(RuntimeError):
    """The exact parent, native deployable sampler, or evidence changed."""


def _install_registered_imports(registration: Mapping[str, Any]) -> None:
    repo = Path(registration["source_repository"]["path"])
    project = repo / "projects" / "latent_action_models"
    shim = repo / "tools" / "env" / "videox_shim"
    videox = Path(registration["runtime"]["videox_home"])
    for root in reversed((str(repo), str(project), str(shim), str(videox))):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["WAN_DIR"] = registration["runtime"]["wan_dir"]
    os.environ["VIDEOX_HOME"] = registration["runtime"]["videox_home"]


def _validate_parent_config(config: Any, registration: Mapping[str, Any]) -> Any:
    model = config.trainer.model
    dual = model.dual_diffusion
    forward_dual = model.forward_model.dual_diffusion
    expected_snapshot = registration["parent"]
    if (
        str(config.name)
        != "vjepa2-faithful-cascade-20260730-seed1234-6560866-v1-VPM"
        or int(config.seed) != 1234
        or not str(model.get("_target_", "")).endswith(
            ".DualExplicitActionDiTModel"
        )
        or int(model.latent_action_dim) != 64
        or int(model.num_history_frames) != 5
        or int(model.num_future_frames) != 8
        or model.time_frequency_transform is not None
        or not bool(dual.enabled)
        or bool(dual.condition_on_tf)
        or str(dual.condition_mode) != "off"
        or str(dual.auxiliary_history_mode) != "diffuse_all"
        or int(dual.tf_channels) != 64
        or float(dual.tf_loss_weight) != 0.0
        or bool(dual.video_only_control)
        or not bool(dual.parameter_matched_control)
        or bool(dual.condition_on_tf_clock)
        or str(dual.schedule_mode) != "aligned"
        or int(dual.evaluation_noise_seed) != 20260729
        or dict(dual) != dict(forward_dual)
        or Path(str(model.rgb_tokenizer.model_path)).resolve(strict=True)
        != Path(registration["runtime"]["wan_dir"]).resolve(strict=True)
        or Path(str(model.forward_model.model_path)).resolve(strict=True)
        != Path(registration["runtime"]["wan_dir"]).resolve(strict=True)
        or expected_snapshot.get("canonical_model_state_sha256")
        != stage.PARENT_CANONICAL_MODEL_STATE_SHA256
    ):
        raise NativeParentVPMError("de65 parent resolved model config differs")
    return model


def load_exact_parent_model(
    registration: Mapping[str, Any], device: Any
) -> tuple[Any, dict[str, Any]]:
    """Instantiate and strictly load the exact de65 parent model state."""

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _install_registered_imports(registration)
    parent = registration["parent"]
    config_record = parent.get("resolved_config")
    if (
        not isinstance(config_record, Mapping)
        or config_record.get("sha256") != stage.PARENT_RESOLVED_CONFIG_SHA256
    ):
        raise NativeParentVPMError("parent resolved config is not registered")
    config_path = Path(config_record["path"]).resolve(strict=True)
    if stage.file_record(config_path) != dict(config_record):
        raise NativeParentVPMError("parent resolved config changed")
    config = OmegaConf.load(config_path)
    model_config = _validate_parent_config(config, registration)
    model = instantiate(model_config)

    snapshot_record = parent["snapshot"]
    snapshot_path = Path(snapshot_record["path"]).resolve(strict=True)
    if snapshot_record.get("sha256") != stage.PARENT_SNAPSHOT_SHA256:
        raise NativeParentVPMError("parent snapshot registration differs")
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    state = snapshot.get("model")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != 8
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 1000
        or snapshot.get("run_identity_sha256")
        != stage.PARENT_RUN_IDENTITY_SHA256
        or not isinstance(state, Mapping)
        or len(state) != 1686
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise NativeParentVPMError("de65 parent snapshot metadata differs")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise NativeParentVPMError("de65 parent strict state load failed")
    del state, snapshot
    model = model.to(device=device).eval()
    if (
        model.num_history_frames != 5
        or model.num_future_frames != 8
        or model.tf_schedule_mode != "aligned"
        or model.tf_condition_mode != "off"
        or bool(model.condition_on_tf)
        or bool(model.condition_on_tf_clock)
        or not bool(model.parameter_matched_control)
        or int(model.forward_model.tf_token_adapter.tf_channels) != 64
        or model.evaluation_noise_seed != 20260729
        or getattr(model, "time_frequency_transform", None) is not None
    ):
        raise NativeParentVPMError("de65 parent runtime contract differs")
    suspicious = []
    for name, module in model.named_modules():
        identity = (
            f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        )
        if any(token in identity for token in ("vjepa", "teacher", "dino")):
            suspicious.append(identity)
    if suspicious:
        raise NativeParentVPMError("parent contains an online teacher/feature model")
    return model, {
        "snapshot": dict(snapshot_record),
        "resolved_config": dict(config_record),
        "run_identity_sha256": stage.PARENT_RUN_IDENTITY_SHA256,
        "canonical_model_state_sha256": (
            stage.PARENT_CANONICAL_MODEL_STATE_SHA256
        ),
        "model_schema_sha256": stage.PARENT_MODEL_SCHEMA_SHA256,
        "strict_state_load": True,
        "native_public_sampler": "sample_future_deployable",
        "condition_source": "off",
        "continued_training_updates": 0,
        "protected_test_accessed": False,
    }


def set_native_parent_endpoint(model: Any, nfe: int) -> None:
    if isinstance(nfe, bool) or nfe not in stage.NFE_GRID:
        raise NativeParentVPMError("parent NFE is outside the registered grid")
    if model.tf_schedule_mode != "aligned":
        raise NativeParentVPMError("parent no longer uses its native aligned grid")
    model.evaluation_condition_sources = ("off",)
    model.evaluation_nfe_steps = (int(nfe),)
    model.viz_num_steps = int(nfe)
    model.capture_latent_trajectories = False
    model.artifact_batch_limit = None


def extract_native_parent_artifacts(
    model: Any,
    *,
    nfe: int,
    predicted_future: Any,
    expected_video_noise: Any | None,
) -> dict[str, Any]:
    """Validate and extract a target-blind native deployable invocation."""

    import torch

    artifacts = model.pop_visualization_artifacts()
    counters = getattr(model, "_last_sampling_counters", None)
    if not isinstance(artifacts, Mapping) or not isinstance(counters, Mapping):
        raise NativeParentVPMError("native parent exposed no sampler evidence")
    latent_key = f"video_final_off_nfe_{nfe}"
    decoded_key = f"decoded_future_off_nfe_{nfe}"
    forbidden = {"video_clean", "tf_clean", "ground_truth_future_uint8"}
    if (
        forbidden.intersection(artifacts)
        or int(artifacts["deployment_mode"].reshape(-1)[0]) != 1
        or int(artifacts["auxiliary_clean_available"].reshape(-1)[0]) != 0
        or int(artifacts["online_teacher_call_count"].reshape(-1)[0]) != 0
        or tuple(int(value) for value in artifacts["evaluation_nfe_steps"].tolist())
        != (nfe,)
        or counters.get("wan_calls_by_source_nfe") != {f"off:nfe_{nfe}": nfe}
        or counters.get("wan_calls_total") != nfe
        or counters.get("online_teacher_calls") != 0
        or counters.get("auxiliary_clean_available") != 0
        or counters.get("deployment_mode") != 1
        or latent_key not in artifacts
        or decoded_key not in artifacts
    ):
        raise NativeParentVPMError("native parent deployable evidence differs")
    latent = artifacts[latent_key].detach().cpu().contiguous()
    decoded = artifacts[decoded_key].detach().cpu().contiguous()
    predicted_uint8 = (
        ((predicted_future.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
    )
    if not torch.equal(decoded, predicted_uint8):
        raise NativeParentVPMError(
            "native returned prediction and recorded decoded output differ"
        )
    initial = artifacts["video_initial_state"].detach().cpu().contiguous()
    if expected_video_noise is not None and not torch.equal(
        initial, expected_video_noise.detach().cpu().to(torch.float16).contiguous()
    ):
        raise NativeParentVPMError(
            "native parent initial noise differs from paired explicit noise"
        )
    return {
        "video_latent": latent,
        "decoded_uint8": decoded,
        "video_initial_state_fp16": initial,
        "sample_ids": artifacts["sample_ids"].detach().cpu().to(torch.int64),
        "wan_calls": int(counters["wan_calls_total"]),
        "flow_model_calls": 0,
        "online_teacher_or_feature_calls": 0,
        "auxiliary_clean_available": False,
        "deployment_mode": True,
        "native_public_sampler": "sample_future_deployable",
    }


def materialize_native_parent_endpoint(
    model: Any,
    *,
    history_rgb: Any,
    actions: Any,
    morphology_index: Any,
    sample_ids: Any,
    nfe: int,
    expected_video_noise: Any,
) -> dict[str, Any]:
    """Invoke the unmodified public parent sampler and return CPU evidence."""

    import torch

    set_native_parent_endpoint(model, nfe)
    if history_rgb.ndim != 5 or tuple(history_rgb.shape[1:3]) != (5, 3):
        raise NativeParentVPMError("parent receives exactly five observed frames")
    if sample_ids.reshape(-1).numel() != history_rgb.shape[0]:
        raise NativeParentVPMError("parent sample ID population differs")
    observed_first_video_state: list[Any] = []

    def capture_first_video_state(_module: Any, inputs: Any) -> None:
        if not observed_first_video_state:
            observed_first_video_state.append(inputs[0].detach().clone())

    noise_handle = model.forward_model.register_forward_pre_hook(
        capture_first_video_state
    )
    if torch.cuda.is_available() and history_rgb.is_cuda:
        torch.cuda.synchronize(history_rgb.device)
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=history_rgb.is_cuda
        ):
            predicted = model.sample_future_deployable(
                history_rgb,
                actions,
                morphology_index=morphology_index,
                collect_artifacts=True,
                sample_ids=sample_ids,
            )
    finally:
        noise_handle.remove()
    if torch.cuda.is_available() and history_rgb.is_cuda:
        torch.cuda.synchronize(history_rgb.device)
    elapsed = time.perf_counter() - started
    result = extract_native_parent_artifacts(
        model,
        nfe=nfe,
        predicted_future=predicted,
        expected_video_noise=expected_video_noise,
    )
    if not torch.equal(
        result["sample_ids"], sample_ids.detach().cpu().to(torch.int64)
    ):
        raise NativeParentVPMError("native parent sample IDs differ")
    if len(observed_first_video_state) != 1 or not torch.equal(
        observed_first_video_state[0], expected_video_noise
    ):
        raise NativeParentVPMError(
            "native parent first Wan input differs from paired full-precision noise"
        )
    result["full_precision_initial_noise_bitwise_equal"] = True
    result["latency"] = {
        "measurement_scope": "native_public_sampler_end_to_end_only",
        "history_encode_seconds": None,
        "adapter_and_wan_seconds": None,
        "decode_seconds": None,
        "end_to_end_seconds": float(elapsed),
    }
    return result
