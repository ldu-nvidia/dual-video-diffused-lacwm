#!/usr/bin/env python3
"""Target-blind, call-counted, full-factorial ACD-P0 endpoint evaluator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "latent_action_models"
for value in (str(ROOT), str(PROJECT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from tools import invertible_two_clock_evaluate as access  # noqa: E402
from tools import low_nfe_direct_baseline as pilot  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE = 2
VALIDATION_ID_OFFSET = 4_000_000
TIMING_REPEATS_PER_LOCAL_BATCH = 4
PAIRED_INVARIANT_FIELDS = (
    "clip_index",
    "actions",
    "clean_latent",
    "noise",
    "clock_indexes",
    "clock_sigmas",
    "clock_timesteps",
    "source_state",
    "teacher_stepped_state",
    "rf_target",
    "teacher_probe",
    "cpu_rng_before_teacher",
    "cuda_rng_before_teacher",
    "cpu_rng_before_online_student",
    "cuda_rng_before_online_student",
    "cpu_rng_before_ema_target",
    "cuda_rng_before_ema_target",
    "cpu_rng_after_forward",
    "cuda_rng_after_forward",
    "call_ledger",
)


class ACDEvaluationError(RuntimeError):
    """A training, access, endpoint, metric, or call identity changed."""


@dataclass
class MaterializedEndpoint:
    latent: Any
    decoded: Any
    tensor_hashes: dict[str, list[str]]
    timing_seconds: list[float]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    materialized_event: int
    hash_closed_event: int
    observed_calls: int
    declared_calls: int


def _run_dir(registration: Mapping[str, Any], arm: pilot.Arm) -> Path:
    return Path(registration["output_root"]) / "training" / arm.run_name


def _read_trace(
    path: Path, arm: pilot.Arm, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != 401:
        raise ACDEvaluationError(f"{arm.code} trace must have header + 400 rows")
    rows = []
    for index, raw in enumerate(raw_lines):
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise ACDEvaluationError(f"invalid trace JSON line {index + 1}") from exc
        if not isinstance(value, dict):
            raise ACDEvaluationError("trace row must be an object")
        rows.append(value)
    header = rows[0]
    expected_label = "ACD-RF-CONT" if arm.arm_mode == "rf_control" else "ACD-CONS"
    if (
        header.get("kind") != "acd_p0_training_trace_header"
        or header.get("protocol_version") != "acd-p0-v1"
        or header.get("arm") != expected_label
        or header.get("arm_mode") != arm.arm_mode
        or header.get("updates") != 400
        or header.get("global_batch_size") != 8
        or header.get("wan_calls_per_update")
        != {"teacher": 1, "online_student": 1, "ema_target": 1}
        or header.get("parent_snapshot_sha256") != pilot.PARENT_SNAPSHOT_SHA256
        or header.get("parent_run_identity_sha256") != pilot.PARENT_RUN_IDENTITY_SHA256
        or header.get("parent_canonical_model_state_sha256")
        != pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
        or header.get("parent_resolved_config_sha256")
        != pilot.PARENT_RESOLVED_CONFIG_SHA256
        or header.get("parent_training_source_commit")
        != pilot.PARENT_TRAINING_SOURCE_COMMIT
        or header.get("paired_invariant_fields") != list(PAIRED_INVARIANT_FIELDS)
        or header.get("endpoint_split_opened") is not False
        or header.get("validation_batches") != 0
        or header.get("wandb_enabled") is not False
        or header.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or header.get("run_identity_sha256")
        != registration["arm_run_identity_sha256"][arm.code]
        or header.get("resolved_arm_config_semantic_sha256")
        != registration["arm_resolved_configs"][arm.code]["semantic_sha256"]
    ):
        raise ACDEvaluationError(f"{arm.code} trace header differs")
    previous_hash = hashlib.sha256(raw_lines[0]).hexdigest()
    events = rows[1:]
    for iteration, (event, raw) in enumerate(zip(events, raw_lines[1:], strict=True)):
        metrics = event.get("metrics")
        if (
            event.get("kind") != "acd_p0_training_trace_event"
            or event.get("phase") != "optimizer_update"
            or event.get("arm") != expected_label
            or event.get("iteration") != iteration
            or event.get("total_observations") != (iteration + 1) * 8
            or event.get("previous_record_sha256") != previous_hash
            or not isinstance(metrics, Mapping)
        ):
            raise ACDEvaluationError(f"{arm.code} update row {iteration} differs")
        for field in PAIRED_INVARIANT_FIELDS:
            value = metrics.get(f"paired_audit/exact_{field}_all_ranks_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise ACDEvaluationError(
                    f"{arm.code} update {iteration} lacks {field} audit"
                )
        previous_hash = hashlib.sha256(raw).hexdigest()
    return header, events, previous_hash


def _validate_completion(
    registration: Mapping[str, Any], arm: pilot.Arm
) -> dict[str, Any]:
    run_dir = _run_dir(registration, arm)
    completion = pilot.read_json(run_dir / "training_complete.json", "training completion")
    trace_complete = pilot.read_json(
        run_dir / "acd_training_trace_complete.json", "ACD trace completion"
    )
    snapshot_path = run_dir / "snapshot.pt"
    expected_identity = registration["arm_run_identity_sha256"][arm.code]
    trace_path = run_dir / "acd_training_trace.jsonl"
    header, events, final_record_hash = _read_trace(trace_path, arm, registration)
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != 400
        or completion.get("max_iter") != 400
        or completion.get("run_identity_sha256") != expected_identity
        or completion.get("snapshot") != str(snapshot_path.resolve(strict=True))
        or trace_complete.get("kind") != "acd_p0_training_trace_complete"
        or trace_complete.get("arm")
        != ("ACD-RF-CONT" if arm.arm_mode == "rf_control" else "ACD-CONS")
        or trace_complete.get("completed_updates") != 400
        or trace_complete.get("trace_update_rows") != 400
        or trace_complete.get("trace_final_record_sha256") != final_record_hash
        or trace_complete.get("trace_sha256") != pilot.sha256(trace_path)
        or trace_complete.get("endpoint_split_opened") is not False
        or trace_complete.get("protected_test_accessed") is not False
        or trace_complete.get("wandb_writes") != 0
        or not isinstance(trace_complete.get("training_wall_seconds_max"), (int, float))
        or float(trace_complete.get("training_wall_seconds_max", 0)) <= 0
        or float(trace_complete.get("training_reserved_b200_hours", 0)) <= 0
        or trace_complete.get("teacher_final_state_receipt")
        != header.get("teacher_initial_state_receipt")
        or trace_complete.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or trace_complete.get("run_identity_sha256") != expected_identity
        or trace_complete.get("resolved_arm_config_semantic_sha256")
        != registration["arm_resolved_configs"][arm.code]["semantic_sha256"]
        or trace_complete.get("initial_rank_state_receipts")
        != header.get("initial_rank_state_receipts")
    ):
        raise ACDEvaluationError(f"{arm.code} completion differs")
    _validate_rank_state_receipts(
        trace_complete.get("initial_rank_state_receipts"),
        label=f"{arm.code} initial",
        require_roles_equal=True,
    )
    _validate_rank_state_receipts(
        trace_complete.get("final_rank_state_receipts"),
        label=f"{arm.code} final",
        require_roles_equal=False,
    )
    return {
        "run_dir": run_dir,
        "snapshot": snapshot_path,
        "snapshot_record": pilot.file_record(snapshot_path),
        "config_record": pilot.file_record(run_dir / ".hydra/config.yaml"),
        "completion_record": pilot.file_record(run_dir / "training_complete.json"),
        "trace_record": pilot.file_record(trace_path),
        "trace_complete_record": pilot.file_record(
            run_dir / "acd_training_trace_complete.json"
        ),
        "trace_complete": trace_complete,
        "header": header,
        "events": events,
    }


def _validate_training_pair(registration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from omegaconf import OmegaConf

    records = {arm.code: _validate_completion(registration, arm) for arm in pilot.ARMS}
    for arm in pilot.ARMS:
        config = OmegaConf.load(records[arm.code]["run_dir"] / ".hydra/config.yaml")
        _validate_config(config, arm)
        payload = OmegaConf.to_container(config, resolve=True)
        if (
            not isinstance(payload, Mapping)
            or pilot.semantic_config_sha256(payload)
            != registration["arm_resolved_configs"][arm.code]["semantic_sha256"]
        ):
            raise ACDEvaluationError(
                f"emitted full resolved config differs: {arm.code}"
            )
    control = records["ACD-RF-CONT"]
    candidate = records["ACD-CONS"]
    for field in (
        "student_initial_state_receipt",
        "teacher_initial_state_receipt",
        "ema_initial_state_receipt",
        "wan_calls_per_update",
        "paired_invariant_fields",
    ):
        if control["header"].get(field) != candidate["header"].get(field):
            raise ACDEvaluationError(f"matched initialization differs: {field}")
    for iteration, (left, right) in enumerate(
        zip(control["events"], candidate["events"], strict=True)
    ):
        if (
            left.get("iteration") != right.get("iteration")
            or left.get("total_observations") != right.get("total_observations")
        ):
            raise ACDEvaluationError(f"paired update index differs: {iteration}")
        for field in PAIRED_INVARIANT_FIELDS:
            key = f"paired_audit/exact_{field}_all_ranks_sha256"
            if left["metrics"].get(key) != right["metrics"].get(key):
                raise ACDEvaluationError(
                    f"paired invariant differs at update {iteration}: {field}"
                )
    return records


def _validate_rank_state_receipts(
    value: Any, *, label: str, require_roles_equal: bool
) -> None:
    if not isinstance(value, list) or len(value) != EXPECTED_WORLD_SIZE:
        raise ACDEvaluationError(f"{label} full-state rank family differs")
    roles = ("student", "teacher", "ema_target")
    for role in roles:
        receipts = [row.get(role) if isinstance(row, Mapping) else None for row in value]
        if any(not isinstance(receipt, Mapping) for receipt in receipts) or any(
            receipt != receipts[0] for receipt in receipts[1:]
        ):
            raise ACDEvaluationError(f"{label} {role} receipt differs across ranks")
    if require_roles_equal and len(
        {value[0][role]["sha256"] for role in roles}
    ) != 1:
        raise ACDEvaluationError(f"{label} role state bytes differ")


def _validate_config(config: Any, arm: pilot.Arm) -> None:
    adjacent = config.model.adjacent_consistency
    dual = config.model.dual_diffusion
    trainer = config.trainer.config
    if (
        str(config.name) != arm.run_name
        or int(config.seed) != 1234
        or not str(config.model._target_).endswith(".AdjacentConsistencyVPM")
        or str(adjacent.arm_mode) != arm.arm_mode
        or int(adjacent.stride) != 500
        or float(adjacent.sigma_data) != 0.5
        or float(adjacent.huber_c) != 0.001
        or float(adjacent.ema_decay) != 0.995
        or str(adjacent.protocol_version) != "acd-p0-v1"
        or not bool(dual.enabled)
        or str(dual.condition_mode) != "off"
        or bool(dual.condition_on_tf)
        or bool(dual.condition_on_tf_clock)
        or float(dual.tf_loss_weight) != 0.0
        or not bool(dual.parameter_matched_control)
        or int(config.model.num_history_frames) != 5
        or int(config.model.num_future_frames) != 8
        or int(trainer.max_iter) != 400
        or int(trainer.gradient_accumulation_steps) != 1
        or int(trainer.logging.log_every) != 1
        or list(trainer.exclude_keys)
        or trainer.transition_handoff_path is not None
        or bool(trainer.validation.save_best)
        or int(trainer.validation.n_val_samples) != 0
        or len(config.val_data_loader) != 0
        or len(config.viz_data_loader) != 0
        or int(config.data_loader.batch_size) != 1
        or float(config.optimizer_factory.lr) != 5e-6
        or list(config.optimizer_factory.betas) != [0.9, 0.95]
        or int(config.lr_scheduler_factory.lr_lambda.warmup_steps) != 40
        or int(config.lr_scheduler_factory.lr_lambda.total_steps) != 400
        or float(trainer.gradient_clipping.max_norm) != 1.0
        or float(trainer.gradient_clipping.norm_type) != 2.0
        or not bool(trainer.gradient_clipping.error_if_nonfinite)
        or bool(config.wandb.enabled)
    ):
        raise ACDEvaluationError(f"resolved configuration differs: {arm.code}")


def _instantiate_model(registration: Mapping[str, Any], training: Mapping[str, Any], device: Any) -> Any:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    arm = pilot.ARM_BY_CODE["ACD-CONS"]
    config = OmegaConf.load(training[arm.code]["run_dir"] / ".hydra/config.yaml")
    _validate_config(config, arm)
    os.environ["WAN_DIR"] = registration["runtime"]["wan_dir"]
    os.environ["VIDEOX_HOME"] = registration["runtime"]["videox_home"]
    model = instantiate(config.model).to(device=device).eval()
    if object.__getattribute__(model, "_acd_teacher_model") is not None or object.__getattribute__(
        model, "_acd_target_model"
    ) is not None:
        raise ACDEvaluationError("deployment model unexpectedly bound teacher/target")
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if any(term in identity for term in ("vjepa", "dino", "rfft", "teacher")):
            suspicious.append(identity)
    if suspicious or model.time_frequency_transform is not None:
        raise ACDEvaluationError("feature/teacher extractor is reachable")
    return model


def _state_record(
    registration: Mapping[str, Any],
    training: Mapping[str, Any],
    endpoint: pilot.Endpoint,
) -> tuple[Mapping[str, Any], str]:
    import torch
    from robot_wm.modeling.low_nfe.adjacent_consistency import tensor_state_sha256

    if endpoint.checkpoint == "parent":
        path = Path(registration["parent"]["snapshot"]["path"])
        snapshot = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = snapshot.get("model")
        expected = pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
    else:
        record = training[endpoint.checkpoint]
        snapshot = torch.load(
            record["snapshot"], map_location="cpu", weights_only=True, mmap=True
        )
        arm = pilot.ARM_BY_CODE[endpoint.checkpoint]
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("acd_schema_version") != 1
            or snapshot.get("world_size") != 8
            or snapshot.get("_start_iter") != 400
            or snapshot.get("_total_observations") != 3200
            or snapshot.get("run_identity_sha256")
            != registration["arm_run_identity_sha256"][arm.code]
            or snapshot.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or snapshot.get("resolved_arm_config_semantic_sha256")
            != registration["arm_resolved_configs"][arm.code]["semantic_sha256"]
            or snapshot.get("teacher_serialized") is not False
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise ACDEvaluationError(f"checkpoint metadata differs: {endpoint.code}")
        key = "model" if endpoint.state == "online" else "target_model"
        receipt_key = (
            "student_final_state_receipt"
            if endpoint.state == "online"
            else "ema_target_state_receipt"
        )
        state = snapshot.get(key)
        snapshot_receipt = snapshot.get(receipt_key)
        trace_receipt = training[arm.code]["trace_complete"].get(receipt_key)
        if (
            snapshot_receipt != trace_receipt
            or snapshot.get("initial_rank_state_receipts")
            != training[arm.code]["trace_complete"].get(
                "initial_rank_state_receipts"
            )
            or snapshot.get("final_rank_state_receipts")
            != training[arm.code]["trace_complete"].get("final_rank_state_receipts")
        ):
            raise ACDEvaluationError(
                f"snapshot/independent completion receipt differs: {endpoint.code}"
            )
        expected = snapshot_receipt.get("sha256") if isinstance(snapshot_receipt, Mapping) else None
    if not isinstance(state, Mapping) or not isinstance(expected, str):
        raise ACDEvaluationError(f"checkpoint state is absent: {endpoint.code}")
    observed = tensor_state_sha256(dict(state))
    if observed != expected:
        raise ACDEvaluationError(f"full state hash differs: {endpoint.code}")
    return state, observed


def _noise_stream(*, steps: int, shape: Sequence[int], device: Any, dtype: Any, sample_ids: Any, seed: int) -> Any:
    import torch

    streams = []
    modulus = (1 << 63) - 1
    for step in range(steps):
        samples = []
        for sample_id in sample_ids.detach().cpu().tolist():
            generator = torch.Generator(device=device)
            generator.manual_seed(
                (
                    int(seed)
                    ^ (int(sample_id) * 0x9E3779B185EBCA87)
                    ^ ((step + 1) * 0xD1B54A32D192ED03)
                )
                % modulus
            )
            samples.append(
                torch.randn(
                    tuple(shape[1:]), device=device, dtype=dtype, generator=generator
                )
            )
        streams.append(torch.stack(samples))
    return torch.stack(streams)


def _episode_shuffle_offset(descriptors: Sequence[Mapping[str, Any]]) -> int:
    if len(descriptors) != 64:
        raise ACDEvaluationError("validation descriptor count differs")
    for offset in range(1, 64):
        if all(
            descriptors[index]["episode_dir"]
            != descriptors[(index + offset) % 64]["episode_dir"]
            for index in range(64)
        ):
            return offset
    raise ACDEvaluationError("no episode-disjoint cyclic action permutation exists")


def _decoded_uint8(value: Any) -> Any:
    return (
        ((value.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(__import__("torch").uint8)
        .cpu()
    )


def _lpips_per_sample(extractor: Any, prediction: Any, target: Any) -> list[float]:
    import torch
    import torch.nn.functional as functional
    from robot_wm.evaluation.video_latent_forcing_quality import (
        lpips_alex_per_frame_per_example,
    )

    def prepare(value: Any) -> Any:
        batch = int(value.shape[0])
        views = []
        normalized = value.float() / 127.5 - 1.0
        for view in range(3):
            camera = normalized[..., view * 320 : (view + 1) * 320]
            frames = camera.permute(0, 2, 1, 3, 4).reshape(batch * 8, 3, 180, 320)
            frames = functional.interpolate(
                frames,
                size=(64, 112),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            views.append(
                frames.reshape(batch, 8, 3, 64, 112).permute(0, 2, 1, 3, 4)
            )
        return torch.cat(views, dim=0)

    scores = lpips_alex_per_frame_per_example(
        prepare(target), prepare(prediction), extractor
    ).reshape(3, int(prediction.shape[0])).mean(0)
    return [float(value) for value in scores.cpu().tolist()]


def _validate_metric_weight_records(
    registration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebind every frozen perceptual-metric weight to its registered bytes."""

    metrics = registration.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ACDEvaluationError("registered metric records are absent")
    validated: dict[str, dict[str, Any]] = {}
    for logical in ("lpips_linear_weight", "alexnet_weight"):
        record = metrics.get(logical)
        if not isinstance(record, Mapping):
            raise ACDEvaluationError(f"registered {logical} record is absent")
        path = Path(str(record.get("path", "")))
        try:
            observed = pilot.file_record(path)
        except Exception as exc:
            raise ACDEvaluationError(
                f"registered {logical} is not a canonical regular file"
            ) from exc
        if observed != dict(record):
            raise ACDEvaluationError(f"registered {logical} bytes differ")
        validated[logical] = observed
    return validated


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    from torch.utils.data import default_collate

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != EXPECTED_WORLD_SIZE:
        raise ACDEvaluationError("ACD evaluation requires exactly eight ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    started = time.perf_counter()
    registration = pilot.validate_registration(args.registration)
    metric_weights = _validate_metric_weight_records(registration)
    os.environ.update(
        {
            "ACD_P0_TRAIN_CLIP_MANIFEST": registration["training"]["manifest"]["path"],
            "ACD_P0_TRAIN_CACHE_METADATA": registration["training"]["cache_metadata"]["path"],
            "ACD_P0_PARENT_SNAPSHOT": registration["parent"]["snapshot"]["path"],
            "ACD_P0_PARENT_RESOLVED_CONFIG": registration["parent"]["resolved_config"]["path"],
            "ACD_P0_RUN_ROOT": str(Path(registration["output_root"]) / "training"),
            "WAN_DIR": registration["runtime"]["wan_dir"],
            "VIDEOX_HOME": registration["runtime"]["videox_home"],
        }
    )
    live_source = pilot.clean_source(
        ROOT, registration["tool_repository"]["git_commit"]
    )
    if live_source["git_tree_sha"] != registration["tool_repository"]["git_tree_sha"]:
        raise ACDEvaluationError("live evaluator source tree differs from registration")
    training = _validate_training_pair(registration)
    for logical in ("rgb", "actions"):
        record = registration["validation"]["arrays"][logical]
        path = Path(record["path"])
        if path.is_symlink() or not path.resolve(strict=True).is_file():
            raise ACDEvaluationError(f"validation {logical} path differs")
        if path.stat().st_size != record["bytes"]:
            raise ACDEvaluationError(f"validation {logical} size differs")
    output = args.output
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise ACDEvaluationError("evaluation output must be fresh")
        output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    assigned = list(range(rank, 64, world))
    if len(assigned) != 8:
        raise ACDEvaluationError("rank assignment differs")
    reader = access.EndpointServingReader(registration=registration, rank=rank)
    batches = []
    for start in range(0, len(assigned), EXPECTED_BATCH_SIZE):
        indexes = assigned[start : start + EXPECTED_BATCH_SIZE]
        batch = default_collate([reader.read_history_and_actions(index) for index in indexes])
        batches.append({"start": start, "indexes": indexes, "batch": batch})
    reader.close()
    endpoint_access_ledger = reader.ledger
    gathered_actions: list[Any] = [None] * world
    local_actions = {
        int(index): stored["batch"]["actions"][offset].clone()
        for stored in batches
        for offset, index in enumerate(stored["indexes"])
    }
    dist.all_gather_object(gathered_actions, local_actions)
    global_actions = {
        int(index): value
        for rank_actions in gathered_actions
        for index, value in rank_actions.items()
    }
    if set(global_actions) != set(range(64)):
        raise ACDEvaluationError("global planned-action family is incomplete")
    shuffle_offset = _episode_shuffle_offset(registration["validation_descriptors"])
    for stored in batches:
        donors = [(index + shuffle_offset) % 64 for index in stored["indexes"]]
        stored["shuffled_actions"] = torch.stack([global_actions[index] for index in donors])
        stored["shuffled_donor_indexes"] = donors
        for recipient, donor, value in zip(
            stored["indexes"], donors, stored["shuffled_actions"], strict=True
        ):
            endpoint_access_ledger.append(
                {
                    "phase": "pre_global_endpoint_barrier",
                    "purpose": "episode_shuffled_planned_actions",
                    "rank": rank,
                    "recipient_row": recipient,
                    "donor_row": donor,
                    "cyclic_offset": shuffle_offset,
                    "returned_tensor_sha256": phase._tensor_sha256(value),
                    "additional_memmap_reads": 0,
                    "future_rgb_elements_read": 0,
                }
            )
    materialized: dict[tuple[int, int, str, int], MaterializedEndpoint] = {}
    state_hashes: dict[str, str] = {}
    event = 0
    total_observed_calls = 0
    timing_invocations: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    source_groups: dict[tuple[str, str], list[pilot.Endpoint]] = {}
    for endpoint in pilot.ENDPOINTS:
        source_groups.setdefault((endpoint.checkpoint, endpoint.state), []).append(endpoint)
    groups = list(source_groups.items())
    rotation = rank % len(groups)
    groups = groups[rotation:] + groups[:rotation]
    warmed: set[tuple[str, int]] = set()
    for (_source_key, endpoints) in groups:
        model = _instantiate_model(registration, training, device)
        state, state_hash = _state_record(registration, training, endpoints[0])
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ACDEvaluationError("strict endpoint state load failed")
        model.eval()
        state_hashes[f"{endpoints[0].checkpoint}:{endpoints[0].state}"] = state_hash
        for stored in batches:
            batch = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                for key, value in stored["batch"].items()
            }
            history = batch["history_rgb"]
            sample_ids = batch["clip_index"] + VALIDATION_ID_OFFSET
            latent_shape = (len(stored["indexes"]), 16, 4, 24, 120)
            for noise_seed in pilot.NOISE_SEEDS:
                ordered_endpoints = (
                    endpoints
                    if (stored["start"] // EXPECTED_BATCH_SIZE + noise_seed + rank) % 2 == 0
                    else list(reversed(endpoints))
                )
                for endpoint in ordered_endpoints:
                    effective_actions = (
                        batch["actions"]
                        if endpoint.action_source == "aligned"
                        else stored["shuffled_actions"].to(device=device)
                    )
                    for nfe in pilot.NFE_GRID:
                        noises = _noise_stream(
                            steps=nfe,
                            shape=latent_shape,
                            device=device,
                            dtype=history.dtype,
                            sample_ids=sample_ids,
                            seed=noise_seed,
                        )
                        warm_key = (endpoint.code, nfe)
                        if warm_key not in warmed:
                            warm_observed = 0

                            def count_warm(_module, _inputs, _output):
                                nonlocal warm_observed
                                warm_observed += 1

                            warm_handle = model.forward_model.transformer.register_forward_hook(
                                count_warm
                            )
                            try:
                                with torch.inference_mode(), torch.autocast(
                                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                                ):
                                    warm = model.sample_consistency_deployable(
                                        history,
                                        effective_actions,
                                        batch["morphology_index"],
                                        renoise_noises=noises,
                                        steps=nfe,
                                        sampler_mode=endpoint.sampler_mode,
                                    )
                            finally:
                                warm_handle.remove()
                            if warm.model_calls != nfe or warm_observed != nfe:
                                raise ACDEvaluationError("warmup call count differs")
                            total_observed_calls += warm_observed
                            warmed.add(warm_key)
                            del warm
                        durations = []
                        first_result = None
                        first_observed = None
                        torch.cuda.reset_peak_memory_stats(device)
                        for repeat in range(TIMING_REPEATS_PER_LOCAL_BATCH):
                            observed = 0

                            def count_call(_module, _inputs, _output):
                                nonlocal observed
                                observed += 1

                            handle = model.forward_model.transformer.register_forward_hook(count_call)
                            try:
                                torch.cuda.synchronize(device)
                                call_started = time.perf_counter()
                                with torch.inference_mode(), torch.autocast(
                                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                                ):
                                    result = model.sample_consistency_deployable(
                                        history,
                                        effective_actions,
                                        batch["morphology_index"],
                                        renoise_noises=noises,
                                        steps=nfe,
                                        sampler_mode=endpoint.sampler_mode,
                                    )
                                torch.cuda.synchronize(device)
                                durations.append(time.perf_counter() - call_started)
                            finally:
                                handle.remove()
                            if (
                                observed != nfe
                                or result.model_calls != nfe
                                or result.teacher_calls != 0
                                or result.feature_calls != 0
                                or result.sampler_mode != endpoint.sampler_mode
                            ):
                                raise ACDEvaluationError("endpoint call/access count differs")
                            total_observed_calls += observed
                            if repeat == 0:
                                first_result = result
                                first_observed = observed
                            else:
                                if not torch.equal(
                                    result.absolute_video_latent,
                                    first_result.absolute_video_latent,
                                ):
                                    raise ACDEvaluationError("timing repeat is not deterministic")
                                del result
                        if first_result is None or first_observed is None:
                            raise ACDEvaluationError("endpoint was not materialized")
                        event += 1
                        materialized_event = event
                        latent = first_result.absolute_video_latent.detach().cpu().to(torch.float16)
                        decoded = _decoded_uint8(first_result.decoded_future)
                        hashes = {
                            "history_rgb": phase._slice_hashes(history),
                            "actions": phase._slice_hashes(effective_actions),
                            "noise_stream": [
                                phase._tensor_sha256(noises[:, offset])
                                for offset in range(int(noises.shape[1]))
                            ],
                            "final_latent": phase._slice_hashes(latent),
                            "decoded_future": phase._slice_hashes(decoded),
                        }
                        event += 1
                        key = (stored["start"], noise_seed, endpoint.code, nfe)
                        if key in materialized:
                            raise ACDEvaluationError("duplicate endpoint key")
                        materialized[key] = MaterializedEndpoint(
                            latent=latent,
                            decoded=decoded,
                            tensor_hashes=hashes,
                            timing_seconds=durations,
                            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                            materialized_event=materialized_event,
                            hash_closed_event=event,
                            observed_calls=first_observed,
                            declared_calls=first_result.model_calls,
                        )
                        timing_key = f"{endpoint.code}:nfe_{nfe}"
                        timing_invocations[timing_key] = timing_invocations.get(timing_key, 0) + len(durations)
                        if rank == 0 and stored["start"] == 0 and noise_seed == pilot.NOISE_SEEDS[0]:
                            evidence[f"{endpoint.code}_nfe_{nfe}_latent"] = latent[:1]
                            evidence[f"{endpoint.code}_nfe_{nfe}_decoded"] = decoded[:1]
                        del first_result, noises
            del batch, history
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
    expected_keys = len(batches) * len(pilot.NOISE_SEEDS) * len(pilot.ENDPOINTS) * len(pilot.NFE_GRID)
    if len(materialized) != expected_keys:
        raise ACDEvaluationError("endpoint family is incomplete")
    event += 1
    local_barrier = access.EndpointBarrierReceipt(
        rank=rank,
        assigned_clip_indexes=tuple(assigned),
        expected_endpoint_keys=expected_keys,
        materialized_endpoint_keys=len(materialized),
        endpoint_tensor_hashes_closed=len(materialized),
        event=event,
    )
    local_barrier.assert_valid()
    dist.barrier()
    event += 1
    global_barrier_event = event
    integrity: list[Any] = [None]
    if rank == 0:
        integrity[0] = {
            logical: pilot.sha256(Path(registration["validation"]["arrays"][logical]["path"]))
            for logical in ("rgb", "actions")
        }
    dist.broadcast_object_list(integrity, src=0)
    observed_integrity = integrity[0]
    if any(
        observed_integrity.get(logical)
        != registration["validation"]["arrays"][logical]["sha256"]
        for logical in ("rgb", "actions")
    ):
        raise ACDEvaluationError("post-barrier validation integrity differs")
    scoring_reader = access.PostBarrierScoringReader(
        registration=registration,
        barrier=local_barrier,
        global_barrier_event=global_barrier_event,
        rank=rank,
    )
    scorings: dict[int, dict[str, Any]] = {}
    target_events: dict[int, int] = {}
    model = _instantiate_model(registration, training, device)
    parent_endpoint = pilot.ENDPOINT_BY_CODE["PARENT_NATIVE_RF"]
    parent_state, _parent_state_hash = _state_record(
        registration, training, parent_endpoint
    )
    incompatible = model.load_state_dict(parent_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ACDEvaluationError("fresh scoring model strict parent load failed")
    model.eval()
    for stored in batches:
        event += 1
        target_events[stored["start"]] = event
        full_rgb = default_collate(
            [scoring_reader.read_full_rgb(index, event=event) for index in stored["indexes"]]
        ).to(device=device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            scoring = phase._scoring_targets(
                model,
                {"rgb": full_rgb, "actions": stored["batch"]["actions"].to(device)},
            )
        scorings[stored["start"]] = scoring
        scoring["clean_hashes"] = phase._slice_hashes(scoring["video_clean"])
        del full_rgb
    scoring_reader.close()
    scoring_access_ledger = scoring_reader.ledger
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from robot_wm.evaluation.video_latent_forcing_quality import FrozenLPIPSAlex

    extractor = FrozenLPIPSAlex(
        linear_weight_path=metric_weights["lpips_linear_weight"]["path"],
        alexnet_weight_path=metric_weights["alexnet_weight"]["path"],
        device=device,
    )
    rows = []
    descriptors = registration["validation_descriptors"]
    for stored in batches:
        scoring = scorings[stored["start"]]
        target_event = target_events[stored["start"]]
        for noise_seed in pilot.NOISE_SEEDS:
            for endpoint in pilot.ENDPOINTS:
                for nfe in pilot.NFE_GRID:
                    item = materialized[(stored["start"], noise_seed, endpoint.code, nfe)]
                    if item.hash_closed_event >= target_event:
                        raise ACDEvaluationError("endpoint hash closed after target access")
                    latent_nmse = phase._per_sample_nmse(item.latent, scoring["video_clean"], 2)
                    delta_nmse = phase._per_sample_future_delta_nmse(
                        item.latent, scoring["video_clean"], 2
                    )
                    decoded_metrics = phase._per_sample_decoded(
                        item.decoded, scoring["ground_truth"], scoring["history_last"]
                    )
                    lpips = _lpips_per_sample(extractor, item.decoded, scoring["ground_truth"])
                    for offset, clip_index in enumerate(stored["indexes"]):
                        descriptor = descriptors[clip_index]
                        rows.append(
                            pilot.identity_payload(
                                {
                                    "kind": "acd_p0_endpoint_row",
                                    "registration_identity_sha256": registration["identity_sha256"],
                                    "source_commit": registration["tool_repository"]["git_commit"],
                                    "endpoint": asdict(endpoint),
                                    "nfe": nfe,
                                    "noise_seed": noise_seed,
                                    "clip_index": clip_index,
                                    "clip_id": descriptor["clip_id"],
                                    "episode_dir": descriptor["episode_dir"],
                                    "action_source": endpoint.action_source,
                                    "action_donor_clip_index": (
                                        clip_index
                                        if endpoint.action_source == "aligned"
                                        else stored["shuffled_donor_indexes"][offset]
                                    ),
                                    "rank": rank,
                                    "batch_start": stored["start"],
                                    "batch_key": (
                                        f"rank-{rank:03d}-batch-{stored['start']:03d}-"
                                        f"seed-{noise_seed}-endpoint-{endpoint.code}-nfe-{nfe}"
                                    ),
                                    "metrics": {
                                        "video_future_nmse": latent_nmse[offset],
                                        "video_future_temporal_delta_nmse": delta_nmse[offset],
                                        "decoded_mse_unit_range": decoded_metrics["decoded_mse_unit_range"][offset],
                                        "decoded_temporal_difference_mse_unit_range": decoded_metrics["decoded_temporal_difference_mse_unit_range"][offset],
                                        "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][offset],
                                        "lpips_alex_frame": lpips[offset],
                                    },
                                    "tensor_sha256": {
                                        key: values[offset] for key, values in item.tensor_hashes.items()
                                    }
                                    | {
                                        "clean_latent_scoring": scoring["clean_hashes"][offset],
                                        "raw_full_rgb_scoring": scoring["rgb_hashes"][offset],
                                    },
                                    "event_order": {
                                        "endpoint_materialized": item.materialized_event,
                                        "endpoint_hashes_closed": item.hash_closed_event,
                                        "first_target_construction": target_event,
                                    },
                                    "latency": {
                                        "cuda_synchronized": True,
                                        "complete_sampler_seconds_per_batch": item.timing_seconds,
                                        "batch_size": len(stored["indexes"]),
                                    },
                                    "peak_memory": {
                                        "allocated_bytes": item.peak_allocated_bytes,
                                        "reserved_bytes": item.peak_reserved_bytes,
                                    },
                                    "actual_wan_calls": item.observed_calls,
                                    "declared_wan_calls": item.declared_calls,
                                    "teacher_calls": 0,
                                    "feature_calls": 0,
                                    "future_rgb_opened_before_global_endpoint_barrier": False,
                                    "protected_test_accessed": False,
                                }
                            )
                        )
    expected_rows = len(assigned) * len(pilot.NOISE_SEEDS) * len(pilot.ENDPOINTS) * len(pilot.NFE_GRID)
    if len(rows) != expected_rows:
        raise ACDEvaluationError("rank endpoint row count differs")
    if any(value * world < 120 for value in timing_invocations.values()):
        raise ACDEvaluationError("global timing repeats are below 120")
    if (
        len(endpoint_access_ledger) != 3 * len(assigned)
        or len(scoring_access_ledger) != len(assigned)
        or any(
            value.get("future_rgb_elements_read", 0) != 0
            for value in endpoint_access_ledger
        )
    ):
        raise ACDEvaluationError("target-blind member-access ledger is incomplete")
    ledger = pilot.identity_payload(
        {
            "kind": "acd_p0_rank_access_ledger",
            "registration_identity_sha256": registration["identity_sha256"],
            "rank": rank,
            "prebarrier_operations": endpoint_access_ledger,
            "postbarrier_operations": scoring_access_ledger,
            "endpoint_barrier_receipt": asdict(local_barrier),
            "global_barrier_event": global_barrier_event,
            "validation_integrity_after_barrier": observed_integrity,
            "generic_prebarrier_getitem_available": False,
            "future_rgb_opened_before_global_endpoint_barrier": False,
            "episode_shuffle_offset": shuffle_offset,
            "episode_shuffle_all_donors_different_episode": True,
            "protected_test_accessed": False,
        }
    )
    ledger_path = output / f"access_ledger_rank_{rank:03d}.json"
    pilot.exclusive_json(ledger_path, ledger)
    rows_path = output / f"rank_{rank:03d}.jsonl"
    with rows_path.open("xb") as handle:
        for row in rows:
            handle.write(pilot.canonical_json(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    evidence_path = output / f"evidence_rank_{rank:03d}.pt"
    torch.save(evidence, evidence_path)
    manifest = pilot.identity_payload(
        {
            "kind": "acd_p0_endpoint_rank",
            "registration_identity_sha256": registration["identity_sha256"],
            "rank": rank,
            "world_size": world,
            "assigned_clip_indexes": assigned,
            "row_count": len(rows),
            "actual_wan_calls_including_warmup_and_repeats": total_observed_calls,
            "timing_invocations": timing_invocations,
            "state_hashes": state_hashes,
            "rows": pilot.file_record(rows_path),
            "evidence": pilot.file_record(evidence_path),
            "access_ledger": pilot.file_record(ledger_path),
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_allocated_bytes": max(
                item.peak_allocated_bytes for item in materialized.values()
            ),
            "peak_memory_reserved_bytes": max(
                item.peak_reserved_bytes for item in materialized.values()
            ),
            "both_training_completions_checked_before_endpoint_reader": True,
            "full_2x2_readout_factorial": True,
            "fresh_parent_native_rf": True,
            "fresh_unbound_model_per_checkpoint_state_group": True,
            "episode_shuffle_offset": shuffle_offset,
            "future_rgb_opened_before_global_endpoint_barrier": False,
            "teacher_calls": 0,
            "feature_calls": 0,
            "protected_test_accessed": False,
        }
    )
    pilot.exclusive_json(output / f"rank_{rank:03d}.json", manifest)
    dist.barrier()
    if rank == 0:
        manifests = [
            pilot.read_json(output / f"rank_{value:03d}.json", "rank manifest")
            for value in range(world)
        ]
        training_b200_hours = sum(
            float(record["trace_complete"]["training_reserved_b200_hours"])
            for record in training.values()
        )
        evaluation_wall = max(float(value["wall_seconds"]) for value in manifests)
        artifact_bytes = sum(
            path.stat().st_size
            for path in Path(registration["output_root"]).rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        inventory = pilot.identity_payload(
            {
                "kind": "acd_p0_endpoint_inventory",
                "registration_identity_sha256": registration["identity_sha256"],
                "source_commit": registration["tool_repository"]["git_commit"],
                "rank_manifests": manifests,
                "row_count": sum(value["row_count"] for value in manifests),
                "training_reserved_b200_hours": training_b200_hours,
                "evaluation_reserved_b200_hours": evaluation_wall * world / 3600.0,
                "observed_reserved_b200_hours": training_b200_hours + evaluation_wall * world / 3600.0,
                "artifact_bytes_under_registered_root": artifact_bytes,
                "endpoint_codes": [endpoint.code for endpoint in pilot.ENDPOINTS],
                "nfe_grid": list(pilot.NFE_GRID),
                "noise_seeds": list(pilot.NOISE_SEEDS),
                "full_2x2_readout_factorial": True,
                "compatible_primary_comparison": {
                    "candidate": "CONS_EMA_CONSISTENCY",
                    "control": "RF_EMA_NATIVE_RF",
                    "parent": "PARENT_NATIVE_RF@nfe1",
                },
                "global_endpoint_barrier_passed": True,
                "future_rgb_opened_before_global_endpoint_barrier": False,
                "teacher_calls": 0,
                "feature_calls": 0,
                "wandb_writes": 0,
                "protected_test_accessed": False,
            }
        )
        pilot.exclusive_json(output / "inventory.json", inventory)
        print(str(output / "inventory.json"))
    dist.barrier()
    dist.destroy_process_group()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
