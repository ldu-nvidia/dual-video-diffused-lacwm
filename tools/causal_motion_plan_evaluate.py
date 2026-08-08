#!/usr/bin/env python3
"""Evaluate frozen CAMP arms on val64 with causal, paired endpoints.

Every endpoint receives exactly five observed RGB frames, requested actions,
morphology, and explicit content-keyed noise.  All ten endpoints for a clip are
completed before the evaluator constructs the clean future scoring tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_motion_plan_workflow as workflow  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


SCHEMA_VERSION = 1
ROW_KIND = "causal_motion_plan_validation_clip"
RANK_KIND = "camp_validation_rank_receipt"
INVENTORY_KIND = "camp_validation_inventory"
EXPECTED_WORLD_SIZE = 8
EXPECTED_VALIDATION_CLIPS = 64
SAMPLE_ID_OFFSET = 8_100_000


class CAMPEvaluationError(RuntimeError):
    """A CAMP causal input, trained artifact, or paired endpoint differs."""


@dataclass(frozen=True)
class Endpoint:
    condition_source: str
    nfe: int
    primary_gate: bool

    @property
    def code(self) -> str:
        return f"{self.condition_source}_nfe_{self.nfe}"


ENDPOINTS = tuple(
    Endpoint(source, nfe, nfe == 1)
    for nfe in (1, 2, 4)
    for source in ("aligned", "off", "shuffled")
) + (Endpoint("action_shuffled", 1, True),)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def _file_record(path: Path, label: str, *, rehash: bool = True) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise CAMPEvaluationError(f"{label} must be a regular absolute file")
    path = path.resolve(strict=True)
    record = {"path": str(path), "bytes": path.stat().st_size}
    if rehash:
        record["sha256"] = _sha256(path)
    return record


def _distributed_file_record(path: Path, label: str) -> dict[str, Any]:
    import torch.distributed as dist

    payload: list[Any] = [None, None]
    if dist.get_rank() == 0:
        try:
            payload[0] = _file_record(path, label)
        except BaseException as exc:
            payload[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None or not isinstance(payload[0], dict):
        raise CAMPEvaluationError(f"unable to bind {label}: {payload[1]}")
    return payload[0]


def _exclusive_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CAMPEvaluationError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_bytes(path, _canonical_json(payload) + b"\n")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CAMPEvaluationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CAMPEvaluationError(f"{label} must contain one object")
    return value


def _arm(code: str) -> workflow.Arm:
    arm = workflow.ARM_BY_CODE.get(code)
    if arm is None:
        raise CAMPEvaluationError(f"unknown CAMP arm: {code}")
    return arm


def _configure_environment(
    sealed: Mapping[str, Any], base: Mapping[str, Any]
) -> None:
    roots = (
        str(REPO_ROOT),
        str(REPO_ROOT / "projects" / "latent_action_models"),
        str(REPO_ROOT / "tools" / "env" / "videox_shim"),
        base["runtime"]["videox_home"],
    )
    for root in reversed(roots):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    values = {
        "WAN_DIR": base["runtime"]["wan_dir"],
        "VIDEOX_HOME": base["runtime"]["videox_home"],
        "CAMP_TRAIN_CLIP_MANIFEST": base["training"]["manifest"]["path"],
        "CAMP_TRAIN_CACHE_METADATA": base["training"]["cache_metadata"]["path"],
        "CAMP_VAL_CLIP_MANIFEST": base["validation"]["manifest"]["path"],
        "CAMP_VAL_CACHE_METADATA": base["validation"]["cache_metadata"]["path"],
        "CAMP_PLAN_STATS": sealed["motion_plan_stats"]["path"],
        "CAMP_PLAN_STATS_SHA256": sealed["motion_plan_stats"]["sha256"],
        "CAMP_PLANNER_SNAPSHOT": sealed["planner_snapshot"]["path"],
        "CAMP_PLANNER_SNAPSHOT_SHA256": sealed["planner_snapshot"]["sha256"],
        "CAMP_VPM_SNAPSHOT": base["parent_snapshot"]["path"],
        "CAMP_VPM_SNAPSHOT_SHA256": base["parent_snapshot"]["sha256"],
        "CAMP_RUN_ROOT": str(Path(sealed["output_root"]) / "training"),
    }
    os.environ.update(values)


def _validate_trace(
    run_dir: Path,
    arm: workflow.Arm,
    sealed: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    trace_path = run_dir / "camp_training_trace.jsonl"
    complete_path = run_dir / "camp_training_trace_complete.json"
    training_complete_path = run_dir / "training_complete.json"
    records = {
        "trace": _file_record(trace_path, "CAMP training trace"),
        "trace_complete": _file_record(complete_path, "CAMP trace completion"),
        "training_complete": _file_record(
            training_complete_path, "CAMP training completion"
        ),
    }
    with trace_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not isinstance(rows[0], dict):
        raise CAMPEvaluationError("CAMP training trace is empty")
    header = rows[0]
    if (
        header.get("kind") != "camp_training_trace_header"
        or header.get("arm") != arm.code
        or header.get("fuse_generated_plan") is not arm.fuse_generated_plan
        or header.get("parent_snapshot_sha256") != base["parent_snapshot"]["sha256"]
        or header.get("planner_snapshot_sha256") != sealed["planner_snapshot"]["sha256"]
        or header.get("motion_plan_stats_sha256")
        != sealed["motion_plan_stats"]["sha256"]
        or header.get("continuation_updates") != 200
        or header.get("planner_calls_per_example") != 2
        or header.get("clean_future_plan_conditioning") is not False
        or header.get("protected_test_accessed") is not False
    ):
        raise CAMPEvaluationError("CAMP training trace header differs")
    complete = _read_json(complete_path, "CAMP trace completion")
    training_complete = _read_json(training_complete_path, "CAMP training completion")
    expected_run_identity = sealed["arm_run_identity_sha256"][arm.code]
    if (
        complete.get("kind") != "camp_training_trace_complete"
        or complete.get("arm") != arm.code
        or complete.get("completed_updates") != 200
        or complete.get("trace_sha256") != records["trace"]["sha256"]
        or complete.get("rows") != len(rows)
        or complete.get("protected_test_accessed") is not False
        or training_complete.get("schema_version") != 1
        or training_complete.get("status") != "completed"
        or training_complete.get("completed_updates") != 200
        or training_complete.get("run_identity_sha256") != expected_run_identity
    ):
        raise CAMPEvaluationError("CAMP training completion differs")
    train_events = [
        row for row in rows[1:] if "train_loss/loss" in row.get("metrics", {})
    ]
    audit_names = (
        "train_loss/paired_audit/clip_index_mean",
        "train_loss/paired_audit/clip_index_square_mean",
        "train_loss/paired_audit/timestep_mean",
        "train_loss/paired_audit/timestep_square_mean",
        "train_loss/paired_audit/video_noise_probe",
        "train_loss/paired_audit/plan_noise_probe",
        "train_loss/paired_audit/action_probe",
    )
    if len(train_events) != 200 or any(
        any(name not in row["metrics"] for name in audit_names) for row in train_events
    ):
        raise CAMPEvaluationError("CAMP paired training audit inventory differs")
    return {**records, "header": header, "train_update_events": 200}


def _load_model(
    sealed: Mapping[str, Any],
    base: Mapping[str, Any],
    arm: workflow.Arm,
    run_dir: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    expected_run = Path(sealed["output_root"]) / "training" / arm.run_name
    if run_dir.expanduser().resolve(strict=True) != expected_run.resolve(strict=True):
        raise CAMPEvaluationError("CAMP arm run directory is noncanonical")
    config_path = run_dir / ".hydra" / "config.yaml"
    config_record = _file_record(config_path, "resolved CAMP config")
    config = OmegaConf.load(config_path)
    if (
        OmegaConf.to_container(config.model, resolve=True)
        != OmegaConf.to_container(config.trainer.model, resolve=True)
        or OmegaConf.to_container(config.dataset, resolve=True)
        != OmegaConf.to_container(config.trainer.data_loader.dataset, resolve=True)
        or str(config.name) != arm.run_name
        or not str(config.model.get("_target_", "")).endswith(".CausalMotionPlanVPM")
        or bool(config.model.causal_motion_plan.fuse_generated_plan)
        is not arm.fuse_generated_plan
        or str(config.model.motion_plan_normalizer.path)
        != sealed["motion_plan_stats"]["path"]
        or str(config.model.motion_plan_normalizer.expected_sha256)
        != sealed["motion_plan_stats"]["sha256"]
        or str(config.model.causal_motion_plan.planner_checkpoint)
        != sealed["planner_snapshot"]["path"]
        or str(config.model.causal_motion_plan.planner_checkpoint_sha256)
        != sealed["planner_snapshot"]["sha256"]
        or int(config.trainer.config.max_iter) != 200
        or int(config.trainer.config.gradient_accumulation_steps) != 1
        or bool(config.trainer.config.validation.save_best)
        or config.wandb.entity != "zijiandu"
        or config.wandb.project != "dual-video-diffusion-private"
        or config.wandb.group is not None
    ):
        raise CAMPEvaluationError("resolved CAMP arm configuration differs")
    model = instantiate(config.model)
    snapshot_path = run_dir / "snapshot.pt"
    snapshot_record = _distributed_file_record(snapshot_path, "trained arm snapshot")
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 200
        or snapshot.get("run_identity_sha256")
        != sealed["arm_run_identity_sha256"][arm.code]
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise CAMPEvaluationError("trained CAMP snapshot differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CAMPEvaluationError("trained CAMP strict load failed")
    del snapshot
    model = model.to(device=device).eval()
    if (
        bool(model.fuse_generated_plan) is not arm.fuse_generated_plan
        or any(p.requires_grad for p in model.causal_motion_planner.parameters())
        or model.motion_plan_normalizer.artifact_sha256
        != sealed["motion_plan_stats"]["sha256"]
    ):
        raise CAMPEvaluationError("loaded CAMP causal/frozen contract differs")
    trace = _validate_trace(run_dir, arm, sealed, base)
    return model, {
        "snapshot": snapshot_record,
        "resolved_config": config_record,
        "trace": trace,
    }


class _ValidationInputs:
    def __init__(self, base: Mapping[str, Any]) -> None:
        import numpy as np

        validation = base["validation"]
        self.rgb = np.load(
            validation["arrays"]["rgb"]["path"], mmap_mode="r", allow_pickle=False
        )
        self.actions = np.load(
            validation["arrays"]["actions"]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            tuple(self.rgb.shape) != (64, 13, 3, 180, 960)
            or str(self.rgb.dtype) != "float16"
            or tuple(self.actions.shape) != (64, 13, 5, 23)
            or str(self.actions.dtype) != "float32"
        ):
            raise CAMPEvaluationError("validation RGB/action array differs")
        with Path(validation["manifest"]["path"]).open(encoding="utf-8") as handle:
            self.rows = tuple(json.loads(line) for line in handle if line.strip())
        if len(self.rows) != EXPECTED_VALIDATION_CLIPS:
            raise CAMPEvaluationError("validation manifest count differs")

    def sample(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        rgb = torch.from_numpy(np.array(self.rgb[index], copy=True)).float()
        actions = torch.from_numpy(np.array(self.actions[index], copy=True))
        actions = torch.nn.functional.pad(actions, (0, 157 - actions.shape[-1]))
        return {
            "rgb": rgb.unsqueeze(0),
            "actions": actions.unsqueeze(0),
            "morphology_index": torch.tensor([9], dtype=torch.long),
            "clip_index": torch.tensor([index], dtype=torch.long),
            "clip_id": self.rows[index]["clip_id"],
        }


def _move(sample: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device=device, non_blocking=False)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in sample.items()
    }


def _uint8(decoded: Any) -> Any:
    import torch

    return (
        ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )


def _scoring_targets(model: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Construct evaluator-only future targets after causal sampling completes."""

    import torch

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        video_clean = model._encode_clip(sample["rgb"]).to(sample["rgb"].dtype)
    raw = sample["rgb"].permute(0, 2, 1, 3, 4)
    future = (
        ((raw[:, :, -8:].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    history_last = (
        ((raw[:, :, 4:5].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    return {
        "video_clean": video_clean.detach().cpu(),
        "ground_truth": future,
        "history_last": history_last,
    }


def _noise(model: Any, history_rgb: Any, sample_ids: Any, rank: int) -> tuple[Any, Any]:
    history_shape = (
        int(history_rgb.shape[0]),
        16,
        4,
        24,
        120,
    )
    plan_shape = (int(history_rgb.shape[0]), 16, 2, 6, 30)
    video = model._evaluation_noise(
        history_shape,
        device=history_rgb.device,
        dtype=history_rgb.dtype,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sample_ids,
        stream=71,
        rank=rank,
    )
    plan = model._evaluation_noise(
        plan_shape,
        device=history_rgb.device,
        dtype=history_rgb.dtype,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sample_ids,
        stream=72,
        rank=rank,
    )
    return video, plan


def _plan_metrics(prediction: Any, target: Any) -> tuple[float, float]:
    import torch

    prediction = prediction.double().flatten(1)
    target = target.double().flatten(1)
    denominator = target.square().sum(dim=1)
    if bool((denominator <= 0).any()):
        raise CAMPEvaluationError("normalized plan target has zero energy")
    nmse = (prediction - target).square().sum(dim=1) / denominator
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=1)
    return float(nmse[0]), float(cosine[0])


def _row(
    *,
    arm: workflow.Arm,
    endpoint: Endpoint,
    sample: Any,
    result: Any,
    planner_actions: Any,
    planner_action_donor_clip_index: int,
    injected_plan_donor_clip_index: int,
    video_noise: Any,
    plan_noise: Any,
    scoring: Mapping[str, Any],
    target_plan: Any,
    clip_index: int,
    clip_id: str,
    sampling_id: int,
    sealed: Mapping[str, Any],
    arm_artifacts: Mapping[str, Any],
    peak_memory_bytes: int,
) -> dict[str, Any]:
    video = result.video_latent.detach().cpu().to(dtype=scoring["video_clean"].dtype)
    decoded = _uint8(result.decoded_future)
    latent_nmse = phase._per_sample_nmse(video, scoring["video_clean"], 2)[0]
    latent_delta_nmse = phase._per_sample_future_delta_nmse(
        video, scoring["video_clean"], 2
    )[0]
    decoded_metrics = phase._per_sample_decoded(
        decoded, scoring["ground_truth"], scoring["history_last"]
    )
    plan_nmse, plan_cosine = _plan_metrics(result.generated_plan.cpu(), target_plan)
    latencies = {
        "history_encode_seconds": float(result.history_encode_seconds),
        "planner_seconds": float(result.planner_seconds),
        "wan_seconds": float(result.wan_seconds),
        "decode_seconds": float(result.decode_seconds),
        "end_to_end_seconds": float(result.end_to_end_seconds),
    }
    if any(not math.isfinite(value) or value < 0 for value in latencies.values()):
        raise CAMPEvaluationError("CAMP stage latency is invalid")
    tensors = {
        "cached_rgb_sha256": phase._tensor_sha256(sample["rgb"]),
        "local_actions_sha256": phase._tensor_sha256(sample["actions"]),
        "planner_actions_sha256": phase._tensor_sha256(planner_actions),
        "history_latent_sha256": phase._tensor_sha256(result.history_latent),
        "video_noise_sha256": phase._tensor_sha256(video_noise),
        "plan_noise_sha256": phase._tensor_sha256(plan_noise),
        "generated_plan_sha256": phase._tensor_sha256(result.generated_plan),
        "injected_plan_sha256": phase._tensor_sha256(result.injected_plan),
        "final_latent_sha256": phase._tensor_sha256(result.video_latent),
        "decode_sha256": phase._tensor_sha256(decoded),
        "video_clean_scoring_sha256": phase._tensor_sha256(scoring["video_clean"]),
        "raw_ground_truth_sha256": phase._tensor_sha256(scoring["ground_truth"]),
    }
    return _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": ROW_KIND,
            "sealed_registration_identity_sha256": sealed["identity_sha256"],
            "arm": arm.code,
            "arm_snapshot": arm_artifacts["snapshot"],
            "model_identity": {
                "parameter_schema_sha256": arm_artifacts["trace"]["header"][
                    "parameter_schema_sha256"
                ],
                "planner_checkpoint_sha256": sealed["planner_snapshot"]["sha256"],
                "motion_plan_stats_sha256": sealed["motion_plan_stats"]["sha256"],
            },
            "evaluation_split": "validation",
            "protected_test_accessed": False,
            "clip_index": clip_index,
            "clip_id": clip_id,
            "sampling_id": sampling_id,
            "endpoint": asdict(endpoint),
            "planner_action_donor_clip_index": planner_action_donor_clip_index,
            "injected_plan_donor_clip_index": injected_plan_donor_clip_index,
            "history_rgb_frames": 5,
            "future_rgb_frames_passed_to_sampler": 0,
            "clean_future_rgb_passed_to_sampler": False,
            "clean_future_video_latent_passed_to_sampler": False,
            "clean_future_feature_passed_to_sampler": False,
            "clean_plan_passed_to_sampler": False,
            "teacher_call_count": 0,
            "scoring_constructed_after_all_sampling": True,
            "runtime_plan_fusion_enabled": bool(
                arm.code == "PLAN-ON" and endpoint.condition_source != "off"
            ),
            "call_counts": {
                "planner": int(result.planner_calls),
                "wan": int(result.wan_calls),
            },
            "declared_nfe": endpoint.nfe,
            "latency_device_synchronized": True,
            "latency_seconds": {
                "history_encode": latencies["history_encode_seconds"],
                "planner": latencies["planner_seconds"],
                "wan": latencies["wan_seconds"],
                "decode": latencies["decode_seconds"],
                "end_to_end": latencies["end_to_end_seconds"],
            },
            "peak_memory_bytes": int(peak_memory_bytes),
            "metrics": {
                "video_future_nmse": latent_nmse,
                "video_future_temporal_delta_nmse": latent_delta_nmse,
                "decoded_mse_unit_range": decoded_metrics[
                    "decoded_mse_unit_range"
                ][0],
                "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][0],
                "decoded_temporal_difference_mse_unit_range": decoded_metrics[
                    "decoded_temporal_difference_mse_unit_range"
                ][0],
                "generated_plan_nmse": plan_nmse,
                "generated_plan_cosine": plan_cosine,
            },
            "tensor_sha256": tensors,
        }
    )


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    from robot_wm.modeling.dual_diffusion.causal_motion_plan import motion_plan_target
    from robot_wm.modeling.dual_diffusion.conditioning import roll_across_global_batch

    if not torch.cuda.is_available():
        raise CAMPEvaluationError("CAMP evaluation requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise CAMPEvaluationError("launch CAMP evaluation with torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if dist.get_world_size() != EXPECTED_WORLD_SIZE:
        raise CAMPEvaluationError("CAMP evaluation requires exactly eight ranks")
    rank = dist.get_rank()
    sealed = workflow._sealed(args.sealed_registration)
    base = workflow._registration(Path(sealed["base_registration"]["path"]))
    _configure_environment(sealed, base)
    arm = _arm(args.arm)
    device = torch.device("cuda", local_rank)
    model, arm_artifacts = _load_model(sealed, base, arm, args.run_dir, device)
    dataset = _ValidationInputs(base)
    output_dir = args.output_dir.expanduser()
    expected_output = Path(sealed["output_root"]) / "evaluation" / arm.code.lower()
    if output_dir != expected_output:
        raise CAMPEvaluationError("CAMP evaluation output is noncanonical")
    if rank == 0:
        if output_dir.exists() or output_dir.is_symlink():
            raise CAMPEvaluationError("CAMP evaluation output must be fresh")
        output_dir.mkdir(parents=True, mode=0o700)
    dist.barrier()
    assigned = list(range(rank, EXPECTED_VALIDATION_CLIPS, EXPECTED_WORLD_SIZE))
    rows: list[dict[str, Any]] = []
    for clip_index in assigned:
        sample = _move(dataset.sample(clip_index), device)
        sampling_ids = sample["clip_index"] + SAMPLE_ID_OFFSET
        history_rgb = sample["rgb"][:, :5]
        video_noise, plan_noise = _noise(model, history_rgb, sampling_ids, rank)
        shuffled_actions = roll_across_global_batch(sample["actions"])
        shuffled_clip_indexes = roll_across_global_batch(sample["clip_index"])
        endpoint_results: list[tuple[Endpoint, Any, Any, int, int, int]] = []
        for endpoint in ENDPOINTS:
            planner_actions = (
                shuffled_actions
                if endpoint.condition_source == "action_shuffled"
                else sample["actions"]
            )
            planner_donor = (
                int(shuffled_clip_indexes.item())
                if endpoint.condition_source == "action_shuffled"
                else clip_index
            )
            injected_donor = (
                int(shuffled_clip_indexes.item())
                if endpoint.condition_source == "shuffled"
                else clip_index
            )
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                result = model.sample_causal_motion_plan(
                    history_rgb,
                    sample["actions"],
                    sample["morphology_index"],
                    video_noise=video_noise,
                    plan_noise=plan_noise,
                    steps=endpoint.nfe,
                    condition_source=endpoint.condition_source,
                )
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
            if result.planner_calls != 2 or result.wan_calls != endpoint.nfe:
                raise CAMPEvaluationError("CAMP endpoint call count differs")
            endpoint_results.append(
                (
                    endpoint,
                    result,
                    planner_actions,
                    planner_donor,
                    injected_donor,
                    peak,
                )
            )

        # The future target is evaluator-owned and is deliberately constructed
        # only after every causal endpoint for this clip has completed.
        scoring = _scoring_targets(model, sample)
        history_reference = endpoint_results[0][1].history_latent.detach().cpu()
        history_difference = (
            scoring["video_clean"][:, :, :2].float() - history_reference.float()
        ).abs().max()
        if float(history_difference) > 1e-4:
            raise CAMPEvaluationError(
                "validation full-clip/history-only Wan tokens violate causality"
            )
        raw_target_plan = motion_plan_target(
            scoring["video_clean"].float(), history_reference.float()
        )
        target_plan = model.motion_plan_normalizer.cpu().normalize(raw_target_plan)
        model.motion_plan_normalizer.to(device)
        for (
            endpoint,
            result,
            planner_actions,
            planner_donor,
            injected_donor,
            peak,
        ) in endpoint_results:
            rows.append(
                _row(
                    arm=arm,
                    endpoint=endpoint,
                    sample=sample,
                    result=result,
                    planner_actions=planner_actions,
                    planner_action_donor_clip_index=planner_donor,
                    injected_plan_donor_clip_index=injected_donor,
                    video_noise=video_noise,
                    plan_noise=plan_noise,
                    scoring=scoring,
                    target_plan=target_plan,
                    clip_index=clip_index,
                    clip_id=sample["clip_id"],
                    sampling_id=int(sampling_ids.item()),
                    sealed=sealed,
                    arm_artifacts=arm_artifacts,
                    peak_memory_bytes=peak,
                )
            )
    rows_path = output_dir / f"rows_rank_{rank:02d}.jsonl"
    _exclusive_bytes(
        rows_path,
        b"".join(_canonical_json(row) + b"\n" for row in rows),
    )
    receipt = _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": RANK_KIND,
            "sealed_registration_identity_sha256": sealed["identity_sha256"],
            "arm": asdict(arm),
            "rank": rank,
            "world_size": EXPECTED_WORLD_SIZE,
            "indexes": assigned,
            "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
            "rows": len(rows),
            "rows_file": _file_record(rows_path, "rank rows"),
            "planner_calls": len(assigned) * len(ENDPOINTS) * 2,
            "wan_calls": len(assigned) * sum(endpoint.nfe for endpoint in ENDPOINTS),
            "protected_test_accessed": False,
        }
    )
    _exclusive_json(output_dir / f"receipt_rank_{rank:02d}.json", receipt)
    dist.barrier()
    if rank == 0:
        row_files = [
            _file_record(output_dir / f"rows_rank_{other:02d}.jsonl", "rank rows")
            for other in range(EXPECTED_WORLD_SIZE)
        ]
        receipts = [
            _file_record(
                output_dir / f"receipt_rank_{other:02d}.json", "rank receipt"
            )
            for other in range(EXPECTED_WORLD_SIZE)
        ]
        inventory = _identity(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": INVENTORY_KIND,
                "sealed_registration_identity_sha256": sealed["identity_sha256"],
                "arm": asdict(arm),
                "validation_clips": EXPECTED_VALIDATION_CLIPS,
                "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
                "row_count": EXPECTED_VALIDATION_CLIPS * len(ENDPOINTS),
                "row_files": row_files,
                "receipts": receipts,
                "protected_test_accessed": False,
            }
        )
        _exclusive_json(output_dir / "inventory.json", inventory)
        print(json.dumps(inventory, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--sealed-registration", type=Path, required=True)
    evaluate.add_argument("--arm", choices=tuple(workflow.ARM_BY_CODE), required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
