#!/usr/bin/env python3
"""Causal equal-Wan-call evaluator for the raw physics-flow Stage-1 gate.

For each D405 validation clip and stateless noise seed, every registered causal
endpoint is fully materialized before this process reads the clean future RGB
bytes for scoring.  Sampler APIs receive only five observed frames, candidate
actions, morphology, explicit Gaussian noise, and a registered fixed field.
The third evaluation-only model is the untouched de65 parent, invoked through
its native public deployment sampler after a pre-validation bitwise parity gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import physics_flow_stage1 as stage  # noqa: E402
from tools.physics_flow_parent_vpm import (  # noqa: E402
    NativeParentVPMError,
    load_exact_parent_model,
    materialize_native_parent_endpoint,
)
from tools.physics_flow_lpips import load_offline_lpips  # noqa: E402


EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE = 2
EXPECTED_EVALUATION_NOISE_SEED = 20_260_729
VALIDATION_SAMPLE_ID_OFFSET = 8_000_000


class PhysicsFlowEvaluationError(RuntimeError):
    """A registered evaluation, model, tensor, or causal contract changed."""


def _distributed_file_record(path: Path) -> dict[str, Any]:
    import torch.distributed as dist

    payload: list[Any] = [None, None]
    if dist.get_rank() == 0:
        try:
            payload[0] = stage.file_record(path)
        except BaseException as exc:  # pragma: no cover - distributed error relay
            payload[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None or not isinstance(payload[0], dict):
        raise PhysicsFlowEvaluationError(
            f"unable to bind shared file: {payload[1] or 'invalid receipt'}"
        )
    return payload[0]


def _tensor_hash(value: Any) -> str:
    import torch

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(stage.canonical_json(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _slice_hashes(value: Any) -> list[str]:
    return [_tensor_hash(item) for item in value]


class RegisteredCausalInputs:
    """Mmap immutable inputs while exposing history and target separately."""

    def __init__(self, registration: Mapping[str, Any]) -> None:
        import numpy as np

        val_cache = registration["flow_caches"]["val"]
        metadata_path = Path(val_cache["metadata"]["path"])
        metadata = stage.load_cache_metadata(metadata_path, split="val")
        cache_registration = stage.read_json(
            Path(metadata["cache_registration"]["path"]), "cache registration"
        )
        self.descriptors = tuple(cache_registration["splits"]["val"]["descriptors"])
        self.d405_indexes = tuple(
            index
            for index, descriptor in enumerate(self.descriptors)
            if descriptor["d405_eligible"]
        )
        if len(self.d405_indexes) != registration["evaluation_contract"]["d405_count"]:
            raise PhysicsFlowEvaluationError("registered D405 population differs")
        immutable = cache_registration["inputs"]["validation"]["cache"]["arrays"]
        self._rgb = np.load(
            immutable["rgb"]["path"], mmap_mode="r", allow_pickle=False
        )
        self._actions = np.load(
            immutable["actions"]["path"], mmap_mode="r", allow_pickle=False
        )
        if (
            self._rgb.shape != (stage.VAL_COUNT, 13, 3, 180, 960)
            or self._rgb.dtype != np.float16
            or self._actions.shape != (stage.VAL_COUNT, 13, 5, 23)
            or self._actions.dtype != np.float32
        ):
            raise PhysicsFlowEvaluationError("immutable validation input geometry differs")
        self._flows = {}
        for source in stage.CACHE_SOURCES:
            record = metadata["arrays"][source]
            path = Path(record["path"]).resolve(strict=True)
            if stage.sha256_file(path) != record["sha256"]:
                raise PhysicsFlowEvaluationError(f"{source} flow cache changed")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            stage.validate_compact_flow(value, count=stage.VAL_COUNT)
            self._flows[source] = value

    @staticmethod
    def _pad_actions(value: Any) -> Any:
        import torch

        if value.shape[-1] > 157:
            raise PhysicsFlowEvaluationError("action width exceeds model padding")
        if value.shape[-1] == 157:
            return value
        return torch.cat(
            (
                value,
                torch.zeros(
                    *value.shape[:-1],
                    157 - value.shape[-1],
                    dtype=value.dtype,
                ),
            ),
            dim=-1,
        )

    def causal_batch(self, indexes: Sequence[int], device: Any) -> dict[str, Any]:
        """Read observed frames only; no future RGB byte is indexed here."""

        import numpy as np
        import torch

        history = torch.from_numpy(
            np.stack([np.array(self._rgb[index, :5], copy=True) for index in indexes])
        ).float()
        actions = torch.from_numpy(
            np.stack([np.array(self._actions[index], copy=True) for index in indexes])
        )
        actions = self._pad_actions(actions)
        morphology = torch.full((len(indexes),), 9, dtype=torch.long)
        return {
            "history_rgb": history.to(device=device),
            "actions": actions.to(device=device),
            "morphology_index": morphology.to(device=device),
            "clip_index": torch.tensor(indexes, dtype=torch.long, device=device),
        }

    def packed_flow(self, source: str, indexes: Sequence[int], device: Any) -> Any:
        import numpy as np
        import torch

        from robot_wm.modeling.dual_diffusion.physics_flow import pack_top_view_flow

        if source == "off":
            transitions = torch.zeros(
                len(indexes), 8, 4, 24, 40, dtype=torch.float32
            )
        else:
            if source not in self._flows:
                raise PhysicsFlowEvaluationError(f"unknown flow source: {source}")
            transitions = torch.from_numpy(
                np.stack(
                    [np.array(self._flows[source][index], copy=True) for index in indexes]
                )
            ).float()
        return pack_top_view_flow(transitions).to(device=device)

    def scoring_batch(self, indexes: Sequence[int], device: Any) -> Any:
        """Read the full clips only after the caller materializes all endpoints."""

        import numpy as np
        import torch

        return torch.from_numpy(
            np.stack([np.array(self._rgb[index], copy=True) for index in indexes])
        ).float().to(device=device)


def _load_model(
    registration: Mapping[str, Any],
    arm: stage.Arm,
    device: Any,
    input_gate_artifact: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

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
    run_dir = Path(str(input_gate_artifact.get("run_dir", ""))).resolve(
        strict=True
    )
    config_path = run_dir / ".hydra" / "config.yaml"
    config_record = stage.file_record(config_path)
    config = OmegaConf.load(config_path)
    model_seed = int(config.model.dual_diffusion.evaluation_noise_seed)
    forward_seed = int(
        config.model.forward_model.dual_diffusion.evaluation_noise_seed
    )
    if (
        str(config.name) != arm.run_name
        or not str(config.model.get("_target_", "")).endswith(".PhysicsFlowVPM")
        or bool(config.model.physics_flow.fuse_flow) is not arm.fuse_flow
        or {model_seed, forward_seed} != {EXPECTED_EVALUATION_NOISE_SEED}
        or int(config.seed) != 1234
        or int(config.data_loader.batch_size) != 1
        or int(config.trainer.config.max_iter) != 200
        or int(config.trainer.config.gradient_accumulation_steps) != 1
        or list(config.val_data_loader) != []
        or list(config.viz_data_loader) != []
        or config.wandb.entity != "zijiandu"
        or config.wandb.project != "dual-video-diffusion-private"
        or config.wandb.group is not None
        or str(config.wandb.mode) != "online"
        or float(config.optimizer_factory.lr) != 1.0e-4
        or tuple(float(value) for value in config.optimizer_factory.betas)
        != (0.9, 0.95)
        or Path(str(config.trainer.config.load_path)).resolve(strict=True)
        != Path(registration["parent"]["snapshot"]["path"]).resolve(strict=True)
    ):
        raise PhysicsFlowEvaluationError(f"{arm.code} resolved config differs")
    abc = config.dataset.datasets.ABC
    flow_record = registration["flow_caches"]["train"]
    if (
        Path(str(abc.flow_metadata)).resolve(strict=True)
        != Path(flow_record["metadata"]["path"]).resolve(strict=True)
        or str(abc.expected_flow_sha256) != flow_record["arrays"]["raw"]["sha256"]
        or str(abc.expected_renderer_analysis_identity_sha256)
        != stage.RENDERER_ANALYSIS_IDENTITY
    ):
        raise PhysicsFlowEvaluationError(f"{arm.code} training cache binding differs")
    model = instantiate(config.model)
    snapshot_path = run_dir / "snapshot.pt"
    snapshot_record = _distributed_file_record(snapshot_path)
    if (
        config_record != input_gate_artifact.get("resolved_config")
        or snapshot_record != input_gate_artifact.get("snapshot")
        or str(run_dir) != input_gate_artifact.get("run_dir")
    ):
        raise PhysicsFlowEvaluationError(
            f"{arm.code} endpoint artifacts differ from causal-input gate"
        )
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 200
        or snapshot.get("run_identity_sha256")
        != registration["arm_run_identity_sha256"][arm.code]
        or snapshot.get("physics_flow_arm") != arm.code
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise PhysicsFlowEvaluationError(f"{arm.code} snapshot differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PhysicsFlowEvaluationError(f"{arm.code} strict state load failed")
    del snapshot
    model = model.to(device=device).eval()
    if (
        bool(model.fuse_flow) is not arm.fuse_flow
        or model.tf_loss_weight != 0.0
        or bool(model.condition_on_tf_clock)
        or model.tf_condition_mode != "off"
        or bool(model.condition_on_tf)
        or model.evaluation_noise_seed != EXPECTED_EVALUATION_NOISE_SEED
        or getattr(model, "time_frequency_transform", None) is not None
    ):
        raise PhysicsFlowEvaluationError(f"{arm.code} runtime contract differs")
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if any(token in identity for token in ("vjepa", "teacher", "dino")):
            suspicious.append(identity)
    if suspicious:
        raise PhysicsFlowEvaluationError("online teacher/feature module is present")
    return model, {
        "snapshot": snapshot_record,
        "resolved_config": config_record,
        "run_dir": str(run_dir.resolve(strict=True)),
        "protected_test_accessed": False,
    }


def _video_noise(model: Any, batch: int, sample_ids: Any, device: Any) -> Any:
    import torch

    return model._evaluation_noise(
        (batch, 16, 4, 24, 120),
        device=device,
        dtype=torch.float32,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sample_ids,
        stream=0,
        rank=0,
    )


def _to_uint8(decoded: Any) -> Any:
    import torch

    return (
        ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )


def _materialize(
    *,
    model: Any,
    endpoint: stage.Endpoint,
    batch: Mapping[str, Any],
    noise: Any,
    flow: Any,
    hook_counter: list[int],
) -> dict[str, Any]:
    import torch

    before = hook_counter[0]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        sample = model.sample_physics_flow(
            batch["history_rgb"],
            batch["actions"],
            batch["morphology_index"],
            video_noise=noise,
            flow_condition=flow,
            steps=endpoint.nfe,
            condition_source=endpoint.condition_source,
        )
    calls = hook_counter[0] - before
    if (
        calls != endpoint.nfe
        or sample.wan_calls != endpoint.nfe
        or sample.flow_model_calls != 0
        or sample.condition_source != endpoint.condition_source
    ):
        raise PhysicsFlowEvaluationError("materialized endpoint call contract differs")
    return {
        "video_latent": sample.video_latent.detach().cpu().to(torch.float16),
        "decoded_uint8": _to_uint8(sample.decoded_future),
        "injected_flow": sample.injected_flow.detach().cpu().to(torch.float16),
        "wan_calls": calls,
        "flow_model_calls": sample.flow_model_calls,
        "native_parent_sampler": False,
        "native_parent_full_precision_noise_verified": False,
        "effective_adapter_gate": float(
            model.forward_model.tf_token_adapter.effective_gate()
            .detach()
            .float()
            .cpu()
        ),
        "latency": {
            "measurement_scope": "component_and_end_to_end",
            "history_encode_seconds": float(sample.history_encode_seconds),
            "adapter_and_wan_seconds": float(sample.adapter_and_wan_seconds),
            "decode_seconds": float(sample.decode_seconds),
            "end_to_end_seconds": float(sample.end_to_end_seconds),
        },
    }


def _materialize_parent(
    *,
    model: Any,
    endpoint: stage.Endpoint,
    batch: Mapping[str, Any],
    noise: Any,
    zero_flow: Any,
    sampling_ids: Any,
    hook_counter: list[int],
) -> dict[str, Any]:
    """Materialize the exact parent's unmodified public deployable sampler."""

    import torch

    before = hook_counter[0]
    sample = materialize_native_parent_endpoint(
        model,
        history_rgb=batch["history_rgb"],
        actions=batch["actions"],
        morphology_index=batch["morphology_index"],
        sample_ids=sampling_ids,
        nfe=endpoint.nfe,
        expected_video_noise=noise,
    )
    calls = hook_counter[0] - before
    if (
        endpoint.arm != stage.PARENT_EVALUATION_MODEL_CODE
        or endpoint.condition_source != "off"
        or calls != endpoint.nfe
        or sample["wan_calls"] != endpoint.nfe
        or sample["flow_model_calls"] != 0
        or sample["online_teacher_or_feature_calls"] != 0
        or sample["auxiliary_clean_available"] is not False
        or sample["deployment_mode"] is not True
    ):
        raise PhysicsFlowEvaluationError(
            "native parent materialization contract differs"
        )
    return {
        "video_latent": sample["video_latent"],
        "decoded_uint8": sample["decoded_uint8"],
        "injected_flow": zero_flow.detach().cpu().to(torch.float16),
        "wan_calls": calls,
        "flow_model_calls": 0,
        "effective_adapter_gate": float(
            model.forward_model.tf_token_adapter.effective_gate()
            .detach()
            .float()
            .cpu()
        ),
        "latency": sample["latency"],
        "native_parent_sampler": True,
        "native_parent_full_precision_noise_verified": sample[
            "full_precision_initial_noise_bitwise_equal"
        ],
    }


def _target_payload(model: Any, full_rgb: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        clean = model._encode_clip(full_rgb).detach().cpu().to(torch.float16)
    raw = full_rgb.permute(0, 2, 1, 3, 4)
    uint8 = _to_uint8(raw)
    return {
        "clean_video": clean,
        "future_uint8": uint8[:, :, -8:],
        "history_last_uint8": uint8[:, :, 4:5],
        "full_rgb_hashes": _slice_hashes(full_rgb),
    }


def _nmse(prediction: Any, target: Any, history_tokens: int = 2) -> list[float]:
    import torch

    pred = prediction[:, :, history_tokens:].double().flatten(1)
    truth = target[:, :, history_tokens:].double().flatten(1)
    denominator = truth.square().sum(dim=1)
    if bool((denominator <= 0).any()):
        raise PhysicsFlowEvaluationError("latent target has zero future energy")
    value = (pred - truth).square().sum(dim=1) / denominator
    if not bool(torch.isfinite(value).all()):
        raise PhysicsFlowEvaluationError("latent NMSE is nonfinite")
    return [float(item) for item in value.tolist()]


def _decoded_metrics(
    prediction: Any, target: Any, history_last: Any
) -> dict[str, list[float]]:
    import torch

    pred = prediction.double() / 255.0
    truth = target.double() / 255.0
    history = history_last.double() / 255.0

    def metrics_for_width(stop: int) -> tuple[Any, Any]:
        p = pred[..., :stop]
        t = truth[..., :stop]
        h = history[..., :stop]
        reduce = tuple(range(1, p.ndim))
        mse = (p - t).square().mean(dim=reduce)
        p_delta = torch.diff(torch.cat((h, p), dim=2), dim=2)
        t_delta = torch.diff(torch.cat((h, t), dim=2), dim=2)
        temporal = (p_delta - t_delta).square().mean(dim=reduce)
        return mse, temporal

    top_mse, top_temporal = metrics_for_width(320)
    all_mse, all_temporal = metrics_for_width(960)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(top_mse, min=1e-12))
    return {
        "top_decoded_mse_unit_range": [float(value) for value in top_mse.tolist()],
        "top_temporal_mse_unit_range": [
            float(value) for value in top_temporal.tolist()
        ],
        "all_decoded_mse_unit_range": [float(value) for value in all_mse.tolist()],
        "all_temporal_mse_unit_range": [
            float(value) for value in all_temporal.tolist()
        ],
        "decoded_psnr_db": [float(value) for value in psnr.tolist()],
    }


def _lpips_top(lpips_model: Any, prediction: Any, target: Any, device: Any) -> list[float]:
    import torch

    batch, channels, frames, height, _width = prediction.shape
    pred = (
        prediction[..., :320]
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * frames, channels, height, 320)
        .float()
        .to(device=device)
        / 127.5
        - 1.0
    )
    truth = (
        target[..., :320]
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * frames, channels, height, 320)
        .float()
        .to(device=device)
        / 127.5
        - 1.0
    )
    values = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        # Chunking limits activation memory and has no effect on the frozen
        # per-frame AlexNet LPIPS definition.
        for start in range(0, len(pred), 8):
            values.append(lpips_model(pred[start : start + 8], truth[start : start + 8]))
    value = torch.cat(values).float().reshape(batch, frames, -1).mean(dim=(1, 2)).cpu()
    if not bool(torch.isfinite(value).all()):
        raise PhysicsFlowEvaluationError("LPIPS is nonfinite")
    return [float(item) for item in value.tolist()]


def _score_rows(
    *,
    materialized: Sequence[tuple[stage.Endpoint, dict[str, Any]]],
    target: Mapping[str, Any],
    batch: Mapping[str, Any],
    indexes: Sequence[int],
    clip_ids: Sequence[str],
    noise_seed_id: int,
    sampling_ids: Any,
    noise: Any,
    lpips_model: Any,
    device: Any,
    registration: Mapping[str, Any],
    model_artifacts: Mapping[str, Mapping[str, Any]],
    input_replay_gate_identity_sha256: str,
    parent_parity_identity_sha256: str,
) -> list[dict[str, Any]]:
    rows = []
    history_hashes = _slice_hashes(batch["history_rgb"])
    action_hashes = _slice_hashes(batch["actions"])
    noise_hashes = _slice_hashes(noise)
    clean_hashes = _slice_hashes(target["clean_video"])
    target_hashes = _slice_hashes(target["future_uint8"])
    sampling_values = [int(value) for value in sampling_ids.detach().cpu().tolist()]
    for endpoint, output in materialized:
        metrics = _decoded_metrics(
            output["decoded_uint8"],
            target["future_uint8"],
            target["history_last_uint8"],
        )
        metrics["future_video_latent_nmse"] = _nmse(
            output["video_latent"], target["clean_video"]
        )
        metrics["top_lpips_alex"] = _lpips_top(
            lpips_model,
            output["decoded_uint8"],
            target["future_uint8"],
            device,
        )
        flow_hashes = _slice_hashes(output["injected_flow"])
        condition_flat = output["injected_flow"].float().flatten(1)
        condition_rms = condition_flat.square().mean(dim=1).sqrt().tolist()
        condition_nonzero = condition_flat.ne(0).float().mean(dim=1).tolist()
        final_hashes = _slice_hashes(output["video_latent"])
        decoded_hashes = _slice_hashes(output["decoded_uint8"])
        for offset, (clip_index, clip_id) in enumerate(zip(indexes, clip_ids)):
            rows.append(
                stage.identity_payload(
                    {
                        "schema_version": stage.SCHEMA_VERSION,
                        "kind": stage.EVALUATION_KIND,
                        "registration_identity_sha256": registration[
                            "identity_sha256"
                        ],
                        "input_replay_gate_identity_sha256": (
                            input_replay_gate_identity_sha256
                        ),
                        "source_commit": registration["source_repository"][
                            "git_commit"
                        ],
                        "clip_index": int(clip_index),
                        "clip_id": str(clip_id),
                        "noise_seed_id": int(noise_seed_id),
                        "sampling_id": sampling_values[offset],
                        "endpoint": asdict(endpoint),
                        "arm_artifacts": model_artifacts[endpoint.arm],
                        "actual_transformer_call_count": output["wan_calls"],
                        "flow_model_call_count": output["flow_model_calls"],
                        "history_rgb_frames": 5,
                        "future_rgb_frames": 8,
                        "history_video_latent_tokens": 2,
                        "future_video_latent_tokens": 2,
                        "all_endpoints_materialized_before_future_rgb_open": True,
                        "future_rgb_sampler_input": False,
                        "future_measured_state_sampler_input": False,
                        "clean_video_latent_sampler_input": False,
                        "online_teacher_or_feature_calls": 0,
                        "native_parent_sampler": output[
                            "native_parent_sampler"
                        ],
                        "native_parent_full_precision_noise_verified": (
                            output[
                                "native_parent_full_precision_noise_verified"
                            ]
                        ),
                        "native_parent_sampler_parity_identity_sha256": (
                            parent_parity_identity_sha256
                        ),
                        "lpips_evaluator_identity_sha256": registration["runtime"][
                            "lpips_alex"
                        ]["identity_sha256"],
                        "seam_diagnostics": {
                            "effective_adapter_gate": output[
                                "effective_adapter_gate"
                            ],
                            "condition_rms": float(condition_rms[offset]),
                            "condition_nonzero_fraction": float(
                                condition_nonzero[offset]
                            ),
                            "fusion_enabled": (
                                endpoint.arm == "RAW-FLOW"
                                and endpoint.condition_source != "off"
                            ),
                            "nonselectable": True,
                        },
                        "protected_test_accessed": False,
                        "metrics": {
                            name: values[offset] for name, values in metrics.items()
                        },
                        "latency_seconds": output["latency"],
                        "tensor_sha256": {
                            "history_rgb_sha256": history_hashes[offset],
                            "actions_sha256": action_hashes[offset],
                            "video_initial_noise_sha256": noise_hashes[offset],
                            "clean_video_latent_sha256": clean_hashes[offset],
                            "future_rgb_target_sha256": target_hashes[offset],
                            "injected_flow_sha256": flow_hashes[offset],
                            "final_video_latent_sha256": final_hashes[offset],
                            "decoded_future_sha256": decoded_hashes[offset],
                        },
                    }
                )
            )
    return rows


def _rank_indexes(d405_indexes: Sequence[int], rank: int) -> list[int]:
    return [int(value) for value in d405_indexes[rank::EXPECTED_WORLD_SIZE]]


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE or args.batch_size != EXPECTED_BATCH_SIZE:
        raise PhysicsFlowEvaluationError("evaluation requires 8 ranks and batch size 2")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise PhysicsFlowEvaluationError("evaluation requires B200 GPUs")
    registration = stage.validate_study_registration(args.registration)
    # The source-pinned gate proves exact v7/v5 causal-input replay, exact frozen
    # tensors, and finite trainable drift without a post-hoc numerical threshold.
    # It preserves v7's failed exact-output qualification and authorizes only a
    # fresh exploratory evaluation of immutable v7 checkpoints.
    input_gate, _input_gate_record = stage.load_input_replay_gate(registration)
    # This receipt is validated before the validation mmap is constructed.  It
    # proves that the third endpoint delegates bit-for-bit to the untouched
    # parent's public target-blind sampler on a registered train history.
    parent_parity, parent_parity_record = stage.load_parent_sampler_parity(
        registration
    )
    output = Path(registration["output_root"]) / "evaluation"
    if args.output_dir.expanduser().absolute() != output:
        raise PhysicsFlowEvaluationError(f"evaluation output must be {output}")
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise PhysicsFlowEvaluationError("fresh evaluation output already exists")
        output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    dataset = RegisteredCausalInputs(registration)
    models = {}
    artifacts = {}
    for arm in stage.ARMS:
        models[arm.code], artifacts[arm.code] = _load_model(
            registration, arm, device, input_gate["artifacts"][arm.code]
        )
    models[stage.PARENT_EVALUATION_MODEL_CODE], artifacts[
        stage.PARENT_EVALUATION_MODEL_CODE
    ] = load_exact_parent_model(registration, device)
    if {
        model.evaluation_noise_seed for model in models.values()
    } != {EXPECTED_EVALUATION_NOISE_SEED}:
        raise PhysicsFlowEvaluationError(
            "training arms and native parent evaluation noise base seeds differ"
        )
    lpips_model, lpips_receipt = load_offline_lpips(device=device)
    if lpips_receipt != registration["runtime"].get("lpips_alex"):
        raise PhysicsFlowEvaluationError(
            "offline LPIPS implementation/weights differ from registration"
        )
    rank_lpips_identities: list[Any] = [None] * world_size
    dist.all_gather_object(
        rank_lpips_identities, lpips_receipt["identity_sha256"]
    )
    if set(rank_lpips_identities) != {lpips_receipt["identity_sha256"]}:
        raise PhysicsFlowEvaluationError("LPIPS identity differs across ranks")
    hook_counts = {code: [0] for code in models}
    handles = []
    for code, model in models.items():
        counter = hook_counts[code]

        def count_call(_module: Any, _inputs: Any, _output: Any, counter=counter) -> None:
            counter[0] += 1

        handles.append(model.forward_model.register_forward_hook(count_call))
    assigned = _rank_indexes(dataset.d405_indexes, rank)
    rows = []
    try:
        for start in range(0, len(assigned), EXPECTED_BATCH_SIZE):
            indexes = assigned[start : start + EXPECTED_BATCH_SIZE]
            batch = dataset.causal_batch(indexes, device)
            clip_ids = [dataset.descriptors[index]["clip_id"] for index in indexes]
            flows = {
                source: dataset.packed_flow(source, indexes, device)
                for source in stage.RUNTIME_SOURCES
            }
            # Materialize the complete registered noise-by-endpoint grid before
            # opening any clean future bytes for this batch.  Retaining the
            # CPU outputs is intentional: it makes the causal boundary a
            # property of execution order, not merely of the sampler API.
            materialized_by_seed = []
            for noise_seed_id in stage.NOISE_SEEDS:
                sampling_ids = (
                    batch["clip_index"]
                    + VALIDATION_SAMPLE_ID_OFFSET
                    + noise_seed_id * 100_000
                )
                noise = _video_noise(
                    models["RAW-FLOW"], len(indexes), sampling_ids, device
                )
                materialized = []
                for endpoint in stage.ENDPOINTS:
                    model = models[endpoint.arm]
                    if endpoint.arm == stage.PARENT_EVALUATION_MODEL_CODE:
                        output_value = _materialize_parent(
                            model=model,
                            endpoint=endpoint,
                            batch=batch,
                            noise=noise,
                            zero_flow=flows["off"],
                            sampling_ids=sampling_ids,
                            hook_counter=hook_counts[endpoint.arm],
                        )
                    else:
                        output_value = _materialize(
                            model=model,
                            endpoint=endpoint,
                            batch=batch,
                            noise=noise,
                            flow=flows[endpoint.condition_source],
                            hook_counter=hook_counts[endpoint.arm],
                        )
                    materialized.append((endpoint, output_value))
                materialized_by_seed.append(
                    (noise_seed_id, sampling_ids, noise, materialized)
                )

            # Causal boundary: only after every endpoint for every registered
            # noise seed above has returned and is resident on CPU may clean
            # future bytes open.
            full_rgb = dataset.scoring_batch(indexes, device)
            target = _target_payload(models["RAW-FLOW"], full_rgb)
            for noise_seed_id, sampling_ids, noise, materialized in materialized_by_seed:
                rows.extend(
                    _score_rows(
                        materialized=materialized,
                        target=target,
                        batch=batch,
                        indexes=indexes,
                        clip_ids=clip_ids,
                        noise_seed_id=noise_seed_id,
                        sampling_ids=sampling_ids,
                        noise=noise,
                        lpips_model=lpips_model,
                        device=device,
                        registration=registration,
                        model_artifacts=artifacts,
                        input_replay_gate_identity_sha256=input_gate[
                            "identity_sha256"
                        ],
                        parent_parity_identity_sha256=parent_parity[
                            "identity_sha256"
                        ],
                    )
                )
            del materialized_by_seed, target, full_rgb
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "rank": rank,
                        "completed_local_clips": min(start + len(indexes), len(assigned)),
                        "local_clips": len(assigned),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    rank_path = output / f"rank_{rank:02d}.jsonl"
    stage.exclusive_jsonl(rank_path, rows)
    local_batches = math.ceil(len(assigned) / EXPECTED_BATCH_SIZE)
    expected_transformer_calls = {
        "FLOW-OFF": local_batches * len(stage.NOISE_SEEDS) * sum(stage.NFE_GRID),
        "RAW-FLOW": (
            local_batches
            * len(stage.NOISE_SEEDS)
            * len(stage.RUNTIME_SOURCES)
            * sum(stage.NFE_GRID)
        ),
        stage.PARENT_EVALUATION_MODEL_CODE: (
            local_batches * len(stage.NOISE_SEEDS) * sum(stage.NFE_GRID)
        ),
    }
    observed_transformer_calls = {
        code: counter[0] for code, counter in hook_counts.items()
    }
    if observed_transformer_calls != expected_transformer_calls:
        raise PhysicsFlowEvaluationError(
            "rank transformer-call inventory differs from endpoint grid"
        )
    receipt = stage.identity_payload(
        {
            "schema_version": stage.SCHEMA_VERSION,
            "kind": "raw_physics_flow_stage1_evaluation_rank",
            "registration_identity_sha256": registration["identity_sha256"],
            "input_replay_gate_identity_sha256": input_gate[
                "identity_sha256"
            ],
            "rank": rank,
            "world_size": world_size,
            "d405_clip_indexes": assigned,
            "noise_seed_ids": list(stage.NOISE_SEEDS),
            "endpoints": [asdict(endpoint) for endpoint in stage.ENDPOINTS],
            "rows": len(rows),
            "rows_file": stage.file_record(rank_path),
            "transformer_calls_by_evaluation_model": {
                **observed_transformer_calls
            },
            "expected_transformer_calls_by_evaluation_model": (
                expected_transformer_calls
            ),
            "native_parent_sampler_parity": parent_parity_record,
            "native_parent_sampler_parity_identity_sha256": parent_parity[
                "identity_sha256"
            ],
            "historical_parent_reference": parent_parity[
                "historical_reference"
            ],
            "historical_vs_current_parent_all_bitwise": True,
            "lpips_evaluator_identity_sha256": lpips_receipt["identity_sha256"],
            "all_endpoints_materialized_before_future_rgb_open": True,
            "future_rgb_sampler_input": False,
            "future_measured_state_sampler_input": False,
            "clean_video_latent_sampler_input": False,
            "protected_test_accessed": False,
        }
    )
    stage.exclusive_json(output / f"rank_{rank:02d}.json", receipt)
    dist.barrier()
    if rank == 0:
        rank_files = []
        rank_receipts = []
        total_rows = 0
        observed_indexes = []
        for source_rank in range(world_size):
            row_path = output / f"rank_{source_rank:02d}.jsonl"
            receipt_path = output / f"rank_{source_rank:02d}.json"
            rank_files.append(stage.file_record(row_path))
            rank_receipts.append(stage.file_record(receipt_path))
            source_receipt = stage.read_json(receipt_path, "rank receipt")
            if (
                not stage.identity_valid(source_receipt)
                or source_receipt.get("kind")
                != "raw_physics_flow_stage1_evaluation_rank"
                or source_receipt.get("rank") != source_rank
                or source_receipt.get("registration_identity_sha256")
                != registration["identity_sha256"]
                or source_receipt.get("input_replay_gate_identity_sha256")
                != input_gate["identity_sha256"]
                or source_receipt.get("rows_file") != rank_files[-1]
                or source_receipt.get("lpips_evaluator_identity_sha256")
                != registration["runtime"]["lpips_alex"]["identity_sha256"]
                or source_receipt.get(
                    "native_parent_sampler_parity_identity_sha256"
                )
                != parent_parity["identity_sha256"]
                or source_receipt.get("native_parent_sampler_parity")
                != parent_parity_record
                or source_receipt.get("historical_parent_reference")
                != parent_parity["historical_reference"]
                or source_receipt.get(
                    "historical_vs_current_parent_all_bitwise"
                )
                is not True
                or source_receipt.get(
                    "transformer_calls_by_evaluation_model"
                )
                != source_receipt.get(
                    "expected_transformer_calls_by_evaluation_model"
                )
                or source_receipt.get(
                    "all_endpoints_materialized_before_future_rgb_open"
                )
                is not True
                or source_receipt.get("clean_video_latent_sampler_input")
                is not False
                or source_receipt.get("protected_test_accessed") is not False
            ):
                raise PhysicsFlowEvaluationError("rank evaluation receipt differs")
            total_rows += int(source_receipt["rows"])
            observed_indexes.extend(source_receipt["d405_clip_indexes"])
        if sorted(observed_indexes) != sorted(dataset.d405_indexes):
            raise PhysicsFlowEvaluationError("D405 shard coverage differs")
        expected_rows = len(dataset.d405_indexes) * len(stage.NOISE_SEEDS) * len(stage.ENDPOINTS)
        if total_rows != expected_rows:
            raise PhysicsFlowEvaluationError("evaluation row count differs")
        inventory = stage.identity_payload(
            {
                "schema_version": stage.SCHEMA_VERSION,
                "kind": stage.EVALUATION_INVENTORY_KIND,
                "status": "complete",
                "registration_identity_sha256": registration["identity_sha256"],
                "input_replay_gate_identity_sha256": input_gate[
                    "identity_sha256"
                ],
                "d405_clips": len(dataset.d405_indexes),
                "d405_clip_indexes": list(dataset.d405_indexes),
                "noise_seed_ids": list(stage.NOISE_SEEDS),
                "endpoints": [asdict(endpoint) for endpoint in stage.ENDPOINTS],
                "rows": total_rows,
                "rank_files": rank_files,
                "rank_receipts": rank_receipts,
                "lpips_evaluator": registration["runtime"]["lpips_alex"],
                "native_parent_sampler_parity": parent_parity_record,
                "native_parent_sampler_parity_identity_sha256": parent_parity[
                    "identity_sha256"
                ],
                "historical_parent_reference": parent_parity[
                    "historical_reference"
                ],
                "historical_vs_current_parent_all_bitwise": True,
                "unmodified_de65_parent_endpoint_included": True,
                "lpips_same_on_all_ranks_and_all_arm_endpoints": True,
                "all_endpoints_materialized_before_future_rgb_open": True,
                "future_rgb_sampler_input": False,
                "future_measured_state_sampler_input": False,
                "clean_video_latent_sampler_input": False,
                "protected_test_accessed": False,
            }
        )
        stage.exclusive_json(output / "inventory.json", inventory)
        # Full semantic replay after the immutable inventory exists.
        stage.load_evaluation_rows(registration)
    dist.barrier()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return command_evaluate(args)
    except (
        PhysicsFlowEvaluationError,
        NativeParentVPMError,
        stage.PhysicsFlowStage1Error,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
