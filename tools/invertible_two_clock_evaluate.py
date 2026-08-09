#!/usr/bin/env python3
"""Target-blind paired endpoint evaluator for the IPQ-TC1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import invertible_two_clock_pilot as pilot  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


KIND_ROW = "ipq_tc1_endpoint_row"
KIND_RANK = "ipq_tc1_endpoint_rank"
KIND_INVENTORY = "ipq_tc1_endpoint_inventory"
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE = 2
VALIDATION_ID_OFFSET = 3_000_000


class IPQEvaluationError(RuntimeError):
    """A model, access boundary, call count, or endpoint identity changed."""


def _array_sha256(value: Any) -> str:
    """Hash one exact NumPy slice without consulting any other row/slice."""

    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(pilot.canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


@dataclass(frozen=True)
class EndpointBarrierReceipt:
    """Proof that a rank closed every target-blind endpoint before scoring."""

    rank: int
    assigned_clip_indexes: tuple[int, ...]
    expected_endpoint_keys: int
    materialized_endpoint_keys: int
    endpoint_tensor_hashes_closed: int
    event: int

    def assert_valid(self) -> None:
        if (
            len(self.assigned_clip_indexes) != 8
            or len(set(self.assigned_clip_indexes)) != 8
            or self.expected_endpoint_keys <= 0
            or self.materialized_endpoint_keys != self.expected_endpoint_keys
            or self.endpoint_tensor_hashes_closed != self.expected_endpoint_keys
            or self.event <= 0
        ):
            raise IPQEvaluationError("endpoint barrier receipt is incomplete")


class EndpointServingReader:
    """Typed memmap reader whose only RGB operation is the fixed `0:5` slice.

    This is deliberately not a production ABC dataset.  There is no generic
    ``__getitem__`` or caller-supplied RGB slice, so future RGB cannot be
    requested through the pre-barrier interface.
    """

    def __init__(
        self,
        *,
        registration: Mapping[str, Any],
        rank: int,
    ) -> None:
        import numpy as np

        self._rank = int(rank)
        self._rgb_record = dict(registration["validation"]["arrays"]["rgb"])
        self._actions_record = dict(
            registration["validation"]["arrays"]["actions"]
        )
        self._rgb = np.load(
            self._rgb_record["path"], mmap_mode="r", allow_pickle=False
        )
        self._actions = np.load(
            self._actions_record["path"], mmap_mode="r", allow_pickle=False
        )
        if (
            tuple(self._rgb.shape) != (64, 13, 3, 180, 960)
            or str(self._rgb.dtype) != "float16"
            or tuple(self._actions.shape) != (64, 13, 5, 23)
            or str(self._actions.dtype) != "float32"
        ):
            raise IPQEvaluationError("registered endpoint array schema changed")
        self.ledger: list[dict[str, Any]] = []
        self._closed = False

    def read_history_and_actions(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        if self._closed:
            raise IPQEvaluationError("endpoint reader is closed")
        index = int(index)
        if not 0 <= index < 64:
            raise IndexError(index)
        # These are the only two memmap indexing operations in this class.
        raw_history = np.array(self._rgb[index, 0:5], copy=True)
        raw_actions = np.array(self._actions[index, 0:13], copy=True)
        history = torch.from_numpy(raw_history).float()
        actions = torch.from_numpy(raw_actions)
        padding = torch.zeros(
            *actions.shape[:-1],
            157 - actions.shape[-1],
            dtype=actions.dtype,
        )
        actions = torch.cat((actions, padding), dim=-1)
        self.ledger.extend(
            (
                {
                    "phase": "pre_global_endpoint_barrier",
                    "purpose": "sampler_observed_history_only",
                    "rank": self._rank,
                    "file": self._rgb_record,
                    "row": index,
                    "slice": ["0:5", ":", ":", ":"],
                    "array_dtype": str(raw_history.dtype),
                    "array_shape": list(raw_history.shape),
                    "array_sha256": _array_sha256(raw_history),
                    "returned_tensor_sha256": phase._tensor_sha256(history),
                    "future_rgb_elements_read": 0,
                },
                {
                    "phase": "pre_global_endpoint_barrier",
                    "purpose": "sampler_planned_actions",
                    "rank": self._rank,
                    "file": self._actions_record,
                    "row": index,
                    "slice": ["0:13", ":", ":"],
                    "array_dtype": str(raw_actions.dtype),
                    "array_shape": list(raw_actions.shape),
                    "array_sha256": _array_sha256(raw_actions),
                    "returned_tensor_sha256": phase._tensor_sha256(actions),
                },
            )
        )
        return {
            "history_rgb": history,
            "actions": actions,
            "morphology_index": torch.tensor(9, dtype=torch.long),
            "clip_index": torch.tensor(index, dtype=torch.long),
        }

    def close(self) -> None:
        if not self._closed:
            _close_memmap(self._rgb)
            _close_memmap(self._actions)
            self._closed = True


class PostBarrierScoringReader:
    """Full-RGB reader that cannot be constructed without a sealed barrier."""

    def __init__(
        self,
        *,
        registration: Mapping[str, Any],
        barrier: EndpointBarrierReceipt,
        global_barrier_event: int,
        rank: int,
    ) -> None:
        import numpy as np

        barrier.assert_valid()
        if barrier.rank != int(rank):
            raise IPQEvaluationError("scoring-reader rank/barrier differs")
        if int(global_barrier_event) <= barrier.event:
            raise IPQEvaluationError("global barrier event is not after local seal")
        self._rank = int(rank)
        self._allowed = set(barrier.assigned_clip_indexes)
        self._local_barrier_event = barrier.event
        self._global_barrier_event = int(global_barrier_event)
        self._rgb_record = dict(registration["validation"]["arrays"]["rgb"])
        self._rgb = np.load(
            self._rgb_record["path"], mmap_mode="r", allow_pickle=False
        )
        if (
            tuple(self._rgb.shape) != (64, 13, 3, 180, 960)
            or str(self._rgb.dtype) != "float16"
        ):
            raise IPQEvaluationError("registered scoring RGB schema changed")
        self.ledger: list[dict[str, Any]] = []
        self._closed = False

    def read_full_rgb(self, index: int, *, event: int) -> Any:
        import numpy as np
        import torch

        if self._closed:
            raise IPQEvaluationError("scoring reader is closed")
        index = int(index)
        if index not in self._allowed:
            raise IPQEvaluationError("scoring row is outside rank assignment")
        if int(event) <= self._global_barrier_event:
            raise IPQEvaluationError("full RGB requested before endpoint barrier")
        # This is the only full-RGB memmap indexing operation in the evaluator.
        raw = np.array(self._rgb[index, 0:13], copy=True)
        tensor = torch.from_numpy(raw).float()
        self.ledger.append(
            {
                "phase": "post_global_endpoint_barrier",
                "purpose": "evaluator_owned_clean_scoring",
                "rank": self._rank,
                "file": self._rgb_record,
                "row": index,
                "slice": ["0:13", ":", ":", ":"],
                "array_dtype": str(raw.dtype),
                "array_shape": list(raw.shape),
                "array_sha256": _array_sha256(raw),
                "returned_tensor_sha256": phase._tensor_sha256(tensor),
                "local_barrier_event": self._local_barrier_event,
                "global_barrier_event": self._global_barrier_event,
                "read_event": int(event),
            }
        )
        return tensor

    def close(self) -> None:
        if not self._closed:
            _close_memmap(self._rgb)
            self._closed = True


@dataclass
class MaterializedEndpoint:
    """Target-blind endpoint tensors and hashes closed before scoring."""

    latent: Any
    p_state: Any
    q_state: Any
    decoded: Any
    tensor_hashes: dict[str, list[str]]
    observed_calls: int
    declared_calls: int
    sampling_event: int
    hash_closed_event: int
    sampling_seconds: float
    decode_seconds: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int


def _arm(registration: Mapping[str, Any], code: str) -> pilot.Arm:
    arm = pilot.ARM_BY_CODE.get(code)
    if arm is None:
        raise IPQEvaluationError(f"unsupported arm: {code}")
    if registration["arm_run_identity_sha256"].get(code) != pilot.arm_identity(
        registration, arm
    ):
        raise IPQEvaluationError("arm identity differs")
    return arm


def _run_dir(registration: Mapping[str, Any], arm: pilot.Arm) -> Path:
    return Path(registration["output_root"]) / "training" / arm.run_name


def _load_trace(run_dir: Path, arm: pilot.Arm) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = run_dir / "ipq_training_trace.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except Exception as exc:
                raise IPQEvaluationError(f"invalid training trace line {number}") from exc
            if not isinstance(value, dict):
                raise IPQEvaluationError("training trace row is not an object")
            rows.append(value)
    if not rows or rows[0].get("kind") != "ipq_tc1_training_trace_header":
        raise IPQEvaluationError("training trace header differs")
    expected_label = "IPQ-SYNC" if arm.arm_mode == "synchronous" else "IPQ-INDEP"
    if (
        rows[0].get("arm") != expected_label
        or rows[0].get("arm_mode") != arm.arm_mode
        or rows[0].get("updates") != 400
        or rows[0].get("wan_calls_per_update") != 1
        or rows[0].get("parent_snapshot_sha256") != pilot.PARENT_SNAPSHOT_SHA256
        or rows[0].get("parent_run_identity_sha256")
        != pilot.PARENT_RUN_IDENTITY_SHA256
        or rows[0].get("parent_canonical_model_state_sha256")
        != pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
        or rows[0].get("parent_resolved_config_sha256")
        != pilot.PARENT_RESOLVED_CONFIG_SHA256
        or rows[0].get("parent_training_source_commit")
        != pilot.PARENT_TRAINING_SOURCE_COMMIT
        or rows[0].get("endpoint_split_opened") is not False
        or rows[0].get("validation_batches") != 0
        or rows[0].get("protocol_version") != "ipq-tc1-v1"
        or rows[0].get("global_batch_size") != 8
        or not isinstance(rows[0].get("model_parameter_schema"), Mapping)
        or rows[0].get("optimizer_parameter_count")
        != rows[0]["model_parameter_schema"].get("parameter_count")
    ):
        raise IPQEvaluationError("training trace lineage/access header differs")
    events = [
        row
        for row in rows[1:]
        if isinstance(row.get("metrics"), dict)
        and "train_loss/paired_audit/exact_clip_index_all_ranks_sha256"
        in row["metrics"]
    ]
    if len(events) != 400:
        raise IPQEvaluationError(f"expected 400 paired updates, got {len(events)}")
    return rows[0], events


def _validate_completion(
    registration: Mapping[str, Any], arm: pilot.Arm
) -> dict[str, Any]:
    run_dir = _run_dir(registration, arm)
    completion = pilot.read_json(run_dir / "training_complete.json", "training completion")
    trace_complete = pilot.read_json(
        run_dir / "ipq_training_trace_complete.json", "trace completion"
    )
    expected_identity = registration["arm_run_identity_sha256"][arm.code]
    snapshot = run_dir / "snapshot.pt"
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != 400
        or completion.get("max_iter") != 400
        or completion.get("run_identity_sha256") != expected_identity
        or completion.get("snapshot") != str(snapshot.resolve(strict=True))
        or trace_complete.get("kind") != "ipq_tc1_training_trace_complete"
        or trace_complete.get("completed_updates") != 400
        or trace_complete.get("endpoint_split_opened") is not False
        or trace_complete.get("protected_test_accessed") is not False
        or trace_complete.get("trace_sha256")
        != pilot.sha256(run_dir / "ipq_training_trace.jsonl")
        or not isinstance(trace_complete.get("model_parameter_schema"), Mapping)
        or not isinstance(trace_complete.get("training_wall_seconds_max"), (int, float))
        or float(trace_complete.get("training_wall_seconds_max", 0.0)) <= 0.0
        or float(trace_complete.get("training_reserved_b200_hours", 0.0)) <= 0.0
    ):
        raise IPQEvaluationError(f"{arm.code} completion differs")
    header, events = _load_trace(run_dir, arm)
    if (
        trace_complete["model_parameter_schema"] != header["model_parameter_schema"]
        or header.get("run_identity_sha256") != expected_identity
        or header.get("wandb_authorized")
        is not bool(registration["wandb"]["enabled"])
        or trace_complete.get("wandb_authorized")
        is not bool(registration["wandb"]["enabled"])
    ):
        raise IPQEvaluationError(f"{arm.code} completion capacity differs")
    return {
        "run_dir": run_dir,
        "snapshot": snapshot,
        "snapshot_record": pilot.file_record(snapshot),
        "config_record": pilot.file_record(run_dir / ".hydra/config.yaml"),
        "completion_record": pilot.file_record(run_dir / "training_complete.json"),
        "trace_record": pilot.file_record(run_dir / "ipq_training_trace.jsonl"),
        "trace_complete_record": pilot.file_record(
            run_dir / "ipq_training_trace_complete.json"
        ),
        "header": header,
        "events": events,
    }


def _validate_both_training_pair(
    registration: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    records = {arm.code: _validate_completion(registration, arm) for arm in pilot.ARMS}
    sync = records["IPQ-SYNC"]
    independent = records["IPQ-INDEP"]
    if (
        sync["header"].get("initial_auxiliary_state_sha256")
        != independent["header"].get("initial_auxiliary_state_sha256")
        or sync["header"].get("model_parameter_schema")
        != independent["header"].get("model_parameter_schema")
        or sync["header"].get("optimizer_parameter_count")
        != independent["header"].get("optimizer_parameter_count")
    ):
        raise IPQEvaluationError("matched arms did not share auxiliary initialization")
    paired_suffixes = (
        "clip_index",
        "actions",
        "clean_latent",
        "canonical_noise",
        "timestep_p",
        "sigma_p",
        "spare_timestep_q",
        "spare_sigma_q",
        "cpu_rng_before_wan",
        "cuda_rng_before_wan",
        "cpu_rng_after_wan",
        "cuda_rng_after_wan",
        "wan_call_count",
    )
    for index, (left, right) in enumerate(
        zip(sync["events"], independent["events"], strict=True)
    ):
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        for ordinary in ("iteration", "total_observations"):
            if left.get(ordinary) != right.get(ordinary):
                raise IPQEvaluationError(f"paired update {index} {ordinary} differs")
        if left_metrics.get("learning_rate") != right_metrics.get("learning_rate"):
            raise IPQEvaluationError(f"paired update {index} learning rate differs")
        for suffix in paired_suffixes:
            key = f"train_loss/paired_audit/exact_{suffix}_all_ranks_sha256"
            if left_metrics.get(key) != right_metrics.get(key):
                raise IPQEvaluationError(f"paired update {index} {suffix} differs")
    return records


def _validate_config(config: Any, arm: pilot.Arm) -> None:
    dual = config.model.dual_diffusion
    ipq = config.model.invertible_two_clock
    trainer = config.trainer.config
    if (
        str(config.name) != arm.run_name
        or int(config.seed) != 1234
        or not str(config.model._target_).endswith(".InvertibleTwoClockVPM")
        or str(ipq.arm_mode) != arm.arm_mode
        or int(ipq.history_latent_frames) != 2
        or int(ipq.num_views) != 3
        or int(ipq.view_width) != 40
        or float(ipq.loss_identity_tolerance) != 2e-6
        or not bool(dual.enabled)
        or int(dual.tf_channels) != 16
        or str(dual.condition_mode) != "matched"
        or not bool(dual.condition_on_tf)
        or not bool(dual.condition_on_tf_clock)
        or not bool(dual.head_condition_on_tf_clock)
        or float(dual.state_gate_init) != 0.02
        or bool(dual.state_gate_trainable)
        or float(dual.clock_gate_init) != 0.02
        or bool(dual.clock_gate_trainable)
        or str(dual.auxiliary_history_mode) != "diffuse_all"
        or not bool(dual.preserve_zero_support)
        or int(config.model.num_history_frames) != 5
        or int(config.model.num_future_frames) != 8
        or int(trainer.max_iter) != 400
        or int(trainer.gradient_accumulation_steps) != 1
        or list(trainer.exclude_keys)
        != [
            "forward_model.tf_token_adapter",
            "forward_model.tf_clock_embedding",
            "forward_model.tf_velocity_head",
        ]
        or trainer.transition_handoff_path is not None
        or bool(trainer.validation.save_best)
        or int(trainer.validation.n_val_samples) != 0
        or len(config.val_data_loader) != 0
        or len(config.viz_data_loader) != 0
        or int(config.data_loader.batch_size) != 1
        or float(config.optimizer_factory.lr) != 1e-4
        or list(config.optimizer_factory.betas) != [0.9, 0.95]
        or int(config.lr_scheduler_factory.lr_lambda.warmup_steps) != 40
        or int(config.lr_scheduler_factory.lr_lambda.total_steps) != 400
        or config.wandb.group is not None
        or str(config.wandb.entity) != "zijiandu"
        or str(config.wandb.project) != "dual-video-diffusion-private"
    ):
        raise IPQEvaluationError(f"{arm.code} resolved configuration differs")


def _load_model(
    registration: Mapping[str, Any], arm: pilot.Arm, record: Mapping[str, Any], device: Any
) -> tuple[Any, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    os.environ["WAN_DIR"] = registration["runtime"]["wan_dir"]
    os.environ["VIDEOX_HOME"] = registration["runtime"]["videox_home"]
    config = OmegaConf.load(record["run_dir"] / ".hydra/config.yaml")
    _validate_config(config, arm)
    model = instantiate(config.model)
    snapshot = torch.load(
        record["snapshot"], map_location="cpu", weights_only=True, mmap=True
    )
    rank_states = snapshot.get("rank_states")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("_start_iter") != 400
        or snapshot.get("_total_observations") != 3200
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("run_identity_sha256")
        != registration["arm_run_identity_sha256"][arm.code]
        or not isinstance(rank_states, Sequence)
        or isinstance(rank_states, (str, bytes))
        or len(rank_states) != EXPECTED_WORLD_SIZE
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise IPQEvaluationError(f"{arm.code} checkpoint metadata differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise IPQEvaluationError(f"{arm.code} checkpoint strict load failed")
    del snapshot
    model = model.to(device=device).eval()
    model._assert_ipq_model_contract()
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if any(term in identity for term in ("vjepa", "teacher", "dino", "rfft")):
            suspicious.append(identity)
    if suspicious or getattr(model, "time_frequency_transform", None) is not None:
        raise IPQEvaluationError("feature/teacher/TF encoder is reachable")
    return model, config


def _evaluation_noise(
    *, shape: Sequence[int], device: Any, dtype: Any, sample_ids: Any, seed: int
) -> Any:
    import torch

    samples = []
    modulus = (1 << 63) - 1
    for sample_id in sample_ids.detach().cpu().tolist():
        generator = torch.Generator(device=device)
        generator.manual_seed(
            (int(seed) ^ (int(sample_id) * 0x9E3779B185EBCA87)) % modulus
        )
        samples.append(
            torch.randn(tuple(shape[1:]), device=device, dtype=dtype, generator=generator)
        )
    return torch.stack(samples)


def _future_uint8(model: Any, latent: Any, out_hw: tuple[int, int]) -> Any:
    import torch

    decoded = model.rgb_tokenizer.decode_temporal(latent, out_hw=out_hw)
    return (
        ((decoded[:, :, -8:].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )


def _band_metrics(
    final_p: Any,
    final_q: Any,
    clean: Any,
    history: int,
) -> dict[str, list[float]]:
    import torch
    from robot_wm.modeling.dual_diffusion.invertible_two_clock import split_pq

    clean_p, clean_q = split_pq(clean.float(), history_frames=history)
    result: dict[str, list[float]] = {}
    for name, prediction, target in (
        ("p", final_p.float(), clean_p),
        ("q", final_q.float(), clean_q),
    ):
        error = (prediction[:, :, history:] - target[:, :, history:]).double().flatten(1)
        predicted = prediction[:, :, history:].double().flatten(1)
        target_flat = target[:, :, history:].double().flatten(1)
        target_energy = target_flat.square().sum(1)
        prediction_energy = predicted.square().sum(1)
        error_energy = error.square().sum(1)
        nmse = error_energy / target_energy.clamp_min(1e-20)
        quantities = {
            f"{name}_future_nmse": nmse,
            f"{name}_target_energy": target_energy,
            f"{name}_prediction_energy": prediction_energy,
            f"{name}_error_energy": error_energy,
        }
        if not all(bool(torch.isfinite(value).all()) for value in quantities.values()):
            raise IPQEvaluationError("band metric is non-finite")
        result.update(
            {
                key: [float(item) for item in value.tolist()]
                for key, value in quantities.items()
            }
        )
    return result


def _slice_hashes(value: Any) -> list[str]:
    return phase._slice_hashes(value)


def _row(
    *,
    registration: Mapping[str, Any],
    arm: pilot.Arm,
    endpoint: pilot.Endpoint,
    nfe: int,
    noise_seed: int,
    clip_index: int,
    clip_id: str,
    episode_dir: str,
    scoring: Mapping[str, Any],
    offset: int,
    band_metrics: Mapping[str, list[float]],
    target_event: int,
    materialized: MaterializedEndpoint,
    batch_key: str,
) -> dict[str, Any]:
    final = materialized.latent
    clean = scoring["video_clean"]
    latent_nmse = phase._per_sample_nmse(final, clean, 2)
    delta_nmse = phase._per_sample_future_delta_nmse(final, clean, 2)
    decoded_metrics = phase._per_sample_decoded(
        materialized.decoded, scoring["ground_truth"], scoring["history_last"]
    )
    if materialized.declared_calls != nfe or materialized.observed_calls != nfe:
        raise IPQEvaluationError("declared NFE differs from both call counters")
    if materialized.hash_closed_event >= target_event:
        raise IPQEvaluationError("endpoint hashes were not closed before scoring")
    return pilot.identity_payload(
        {
            "schema_version": pilot.SCHEMA_VERSION,
            "kind": KIND_ROW,
            "registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["tool_repository"]["git_commit"],
            "arm": asdict(arm),
            "endpoint": asdict(endpoint),
            "nfe": nfe,
            "noise_seed": noise_seed,
            "clip_index": clip_index,
            "clip_id": clip_id,
            "episode_dir": episode_dir,
            "sampling_id": VALIDATION_ID_OFFSET + clip_index,
            "metrics": {
                "video_future_nmse": latent_nmse[offset],
                "decoded_mse_unit_range": decoded_metrics[
                    "decoded_mse_unit_range"
                ][offset],
                "decoded_temporal_difference_mse_unit_range": decoded_metrics[
                    "decoded_temporal_difference_mse_unit_range"
                ][offset],
                "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][offset],
                "video_future_temporal_delta_nmse": delta_nmse[offset],
                **{
                    key: values[offset] for key, values in band_metrics.items()
                },
            },
            "tensor_sha256": {
                **{
                    key: values[offset]
                    for key, values in materialized.tensor_hashes.items()
                },
                "clean_latent_scoring": _slice_hashes(clean)[offset],
                "raw_full_rgb_scoring": scoring["rgb_hashes"][offset],
            },
            "event_order": {
                "endpoint_materialized": materialized.sampling_event,
                "endpoint_hashes_closed": materialized.hash_closed_event,
                "first_target_construction": target_event,
            },
            "batch_key": batch_key,
            "latency": {
                "cuda_synchronized": True,
                "sampling_seconds_per_batch": materialized.sampling_seconds,
                "decode_seconds_per_batch": materialized.decode_seconds,
                "end_to_end_seconds_per_batch": (
                    materialized.sampling_seconds + materialized.decode_seconds
                ),
                "batch_size": int(final.shape[0]),
            },
            "peak_memory": {
                "allocated_bytes": materialized.peak_memory_allocated_bytes,
                "reserved_bytes": materialized.peak_memory_reserved_bytes,
            },
            "scoring_constructed_after_all_batch_endpoints": (
                target_event > materialized.hash_closed_event
            ),
            "actual_wan_calls": materialized.observed_calls,
            "clean_future_rgb_passed_to_sampler": False,
            "clean_future_latent_passed_to_sampler": False,
            "future_rgb_opened_before_global_endpoint_barrier": False,
            "auxiliary_target_array_opened": False,
            "teacher_feature_encoder_calls": 0,
            "protected_test_accessed": False,
        }
    )


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    from torch.utils.data import default_collate

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != EXPECTED_WORLD_SIZE:
        raise IPQEvaluationError("IPQ evaluation requires exactly eight ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    job_started = time.perf_counter()
    registration = pilot.validate_registration(args.registration)
    arm = _arm(registration, args.arm)
    training = _validate_both_training_pair(registration)
    # Pre-barrier checks bind only path/type/size.  Content hashing would read
    # future bytes and is therefore deferred until the global endpoint barrier.
    for logical in ("rgb", "actions"):
        record = registration["validation"]["arrays"][logical]
        path = Path(record["path"])
        if path.is_symlink():
            raise IPQEvaluationError(f"validation {logical} stat changed")
        path = path.resolve(strict=True)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise IPQEvaluationError(f"validation {logical} stat changed")
    output = args.output
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise IPQEvaluationError("evaluation output must be fresh")
        output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    model, _config = _load_model(registration, arm, training[arm.code], device)
    assigned = list(range(rank, 64, world))
    if len(assigned) != 8:
        raise IPQEvaluationError("rank assignment differs")
    endpoints = pilot.ENDPOINTS_BY_ARM[arm.code]
    rows = []
    evidence: dict[str, Any] = {}
    event_counter = 0
    total_calls = 0
    endpoint_reader = EndpointServingReader(registration=registration, rank=rank)
    batch_materializations: list[dict[str, Any]] = []
    total_materialized = 0
    total_hash_closed = 0
    for batch_start in range(0, len(assigned), EXPECTED_BATCH_SIZE):
        indexes = assigned[batch_start : batch_start + EXPECTED_BATCH_SIZE]
        batch = default_collate(
            [endpoint_reader.read_history_and_actions(index) for index in indexes]
        )
        batch = {
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        history = batch["history_rgb"]
        sampling_ids = batch["clip_index"] + VALIDATION_ID_OFFSET
        latent_shape = (len(indexes), 16, 4, 24, 120)
        materialized: dict[tuple[int, str, int], MaterializedEndpoint] = {}
        for noise_seed in pilot.NOISE_SEEDS:
            initial_noise = _evaluation_noise(
                shape=latent_shape,
                device=device,
                dtype=history.dtype,
                sample_ids=sampling_ids,
                seed=noise_seed,
            )
            for endpoint in endpoints:
                for nfe in pilot.NFE_GRID:
                    observed = 0

                    def count_call(_module, _inputs, _output):
                        nonlocal observed
                        observed += 1

                    handle = model.forward_model.transformer.register_forward_hook(
                        count_call
                    )
                    collect = (
                        endpoint.primary
                        and noise_seed == pilot.NOISE_SEEDS[0]
                        and batch_start == 0
                    )
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)
                        sample_started = time.perf_counter()
                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16, enabled=True
                        ):
                            result = model.sample_ipq_latent(
                                history,
                                actions=batch["actions"],
                                morphology_index=batch["morphology_index"],
                                initial_noise=initial_noise,
                                num_steps=nfe,
                                schedule_mode=endpoint.schedule_mode,
                                condition_source=endpoint.condition_source,
                                collect_trajectory=collect,
                            )
                        torch.cuda.synchronize(device)
                        sampling_seconds = time.perf_counter() - sample_started
                        decode_started = time.perf_counter()
                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16, enabled=True
                        ):
                            decoded = _future_uint8(
                                model, result.latent, (180, 960)
                            )
                        torch.cuda.synchronize(device)
                        decode_seconds = time.perf_counter() - decode_started
                        peak_allocated = int(torch.cuda.max_memory_allocated(device))
                        peak_reserved = int(torch.cuda.max_memory_reserved(device))
                    finally:
                        handle.remove()
                    event_counter += 1
                    sampling_event = event_counter
                    key = (noise_seed, endpoint.code, nfe)
                    if key in materialized:
                        raise IPQEvaluationError("duplicate endpoint materialization")
                    final = result.latent.detach().cpu().to(torch.float16)
                    final_p = result.p_state.detach().cpu().to(torch.float16)
                    final_q = result.q_state.detach().cpu().to(torch.float16)
                    tensor_hashes = {
                        "sampler_history_rgb": _slice_hashes(history),
                        "sampler_actions": _slice_hashes(batch["actions"]),
                        "initial_noise": _slice_hashes(result.initial_noise),
                        "final_latent": _slice_hashes(final),
                        "final_p": _slice_hashes(final_p),
                        "final_q": _slice_hashes(final_q),
                        "decoded_future": _slice_hashes(decoded),
                    }
                    if collect:
                        evidence[f"{endpoint.code}_nfe_{nfe}_p"] = torch.stack(
                            result.trajectory_p
                        )[:, :1].cpu().to(torch.float16)
                        evidence[f"{endpoint.code}_nfe_{nfe}_q"] = torch.stack(
                            result.trajectory_q
                        )[:, :1].cpu().to(torch.float16)
                    event_counter += 1
                    hash_closed_event = event_counter
                    materialized[key] = MaterializedEndpoint(
                        latent=final,
                        p_state=final_p,
                        q_state=final_q,
                        decoded=decoded,
                        tensor_hashes=tensor_hashes,
                        observed_calls=observed,
                        declared_calls=result.wan_calls,
                        sampling_event=sampling_event,
                        hash_closed_event=hash_closed_event,
                        sampling_seconds=float(sampling_seconds),
                        decode_seconds=float(decode_seconds),
                        peak_memory_allocated_bytes=peak_allocated,
                        peak_memory_reserved_bytes=peak_reserved,
                    )
                    total_calls += observed
                    total_materialized += 1
                    total_hash_closed += 1
                    del result, decoded, final, final_p, final_q
            if arm.code == "IPQ-INDEP":
                nfe_one = {
                    endpoint_code: tuple(
                        materialized[(noise_seed, endpoint_code, 1)].tensor_hashes[
                            "final_latent"
                        ]
                    )
                    for endpoint_code in (
                        "P_LEADS_ALIGNED",
                        "SYNCHRONOUS",
                        "Q_LEADS_ALIGNED",
                    )
                }
                if len(set(nfe_one.values())) != 1:
                    raise IPQEvaluationError(
                        "NFE-1 schedule order is not bit-identical"
                    )
        expected_batch_keys = (
            len(pilot.NOISE_SEEDS) * len(endpoints) * len(pilot.NFE_GRID)
        )
        if len(materialized) != expected_batch_keys:
            raise IPQEvaluationError("batch endpoint family is incomplete")
        batch_materializations.append(
            {
                "batch_start": batch_start,
                "indexes": tuple(indexes),
                "history_rgb": history.detach().cpu(),
                "actions": batch["actions"].detach().cpu(),
                "materialized": materialized,
            }
        )
        del batch, history

    endpoint_reader.close()
    expected_total_keys = (
        len(batch_materializations)
        * len(pilot.NOISE_SEEDS)
        * len(endpoints)
        * len(pilot.NFE_GRID)
    )
    event_counter += 1
    local_barrier = EndpointBarrierReceipt(
        rank=rank,
        assigned_clip_indexes=tuple(assigned),
        expected_endpoint_keys=expected_total_keys,
        materialized_endpoint_keys=total_materialized,
        endpoint_tensor_hashes_closed=total_hash_closed,
        event=event_counter,
    )
    local_barrier.assert_valid()
    # This is the target-access boundary: every rank must have closed every
    # endpoint tensor/hash before any rank reads or rehashes full RGB.
    dist.barrier()
    event_counter += 1
    global_barrier_event = event_counter
    integrity_status: list[Any] = [None]
    if rank == 0:
        try:
            integrity_status[0] = {
                logical: pilot.sha256(
                    Path(registration["validation"]["arrays"][logical]["path"])
                )
                for logical in ("rgb", "actions")
            }
        except Exception as exc:  # pragma: no cover - distributed fail relay
            integrity_status[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(integrity_status, src=0)
    observed_integrity = integrity_status[0]
    if not isinstance(observed_integrity, dict) or "error" in observed_integrity:
        raise IPQEvaluationError(f"post-barrier integrity rehash failed: {observed_integrity}")
    for logical in ("rgb", "actions"):
        if (
            observed_integrity.get(logical)
            != registration["validation"]["arrays"][logical]["sha256"]
        ):
            raise IPQEvaluationError(f"validation {logical} bytes changed")

    scoring_reader = PostBarrierScoringReader(
        registration=registration,
        barrier=local_barrier,
        global_barrier_event=global_barrier_event,
        rank=rank,
    )
    for stored in batch_materializations:
        indexes = stored["indexes"]
        event_counter += 1
        target_event = event_counter
        full_rgb = default_collate(
            [
                scoring_reader.read_full_rgb(index, event=target_event)
                for index in indexes
            ]
        ).to(device=device)
        scoring_batch = {
            "rgb": full_rgb,
            "actions": stored["actions"].to(device=device),
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            scoring = phase._scoring_targets(model, scoring_batch)
        descriptors = [
            registration["validation_descriptors"][index] for index in indexes
        ]
        for key, materialized_endpoint in stored["materialized"].items():
            noise_seed, endpoint_code, nfe = key
            endpoint = next(
                value for value in endpoints if value.code == endpoint_code
            )
            band_metrics = _band_metrics(
                materialized_endpoint.p_state,
                materialized_endpoint.q_state,
                scoring["video_clean"],
                2,
            )
            for offset, index in enumerate(indexes):
                rows.append(
                    _row(
                        registration=registration,
                        arm=arm,
                        endpoint=endpoint,
                        nfe=nfe,
                        noise_seed=noise_seed,
                        clip_index=index,
                        clip_id=descriptors[offset]["clip_id"],
                        episode_dir=descriptors[offset]["episode_dir"],
                        scoring=scoring,
                        offset=offset,
                        band_metrics=band_metrics,
                        target_event=target_event,
                        materialized=materialized_endpoint,
                        batch_key=(
                            f"rank-{rank:03d}-batch-{stored['batch_start']:03d}-"
                            f"seed-{noise_seed}-endpoint-{endpoint_code}-nfe-{nfe}"
                        ),
                    )
                )
        del full_rgb, scoring_batch, scoring
    scoring_reader.close()
    endpoint_access = endpoint_reader.ledger
    scoring_access = scoring_reader.ledger
    if (
        len(endpoint_access) != 2 * len(assigned)
        or len(scoring_access) != len(assigned)
        or {
            int(value["row"])
            for value in endpoint_access
            if value["purpose"] == "sampler_observed_history_only"
        }
        != set(assigned)
        or {int(value["row"]) for value in scoring_access} != set(assigned)
        or any(
            value.get("future_rgb_elements_read") != 0
            for value in endpoint_access
            if value["purpose"] == "sampler_observed_history_only"
        )
    ):
        raise IPQEvaluationError("memmap access ledger is incomplete")
    expected_rows = (
        len(assigned)
        * len(pilot.NOISE_SEEDS)
        * len(pilot.NFE_GRID)
        * len(endpoints)
    )
    expected_calls = (
        len(batch_materializations)
        * len(pilot.NOISE_SEEDS)
        * len(endpoints)
        * sum(pilot.NFE_GRID)
    )
    if len(rows) != expected_rows or total_calls != expected_calls:
        raise IPQEvaluationError("endpoint row/call family is incomplete")
    access_ledger_path = output / f"access_ledger_rank_{rank:03d}.json"
    access_ledger = pilot.identity_payload(
        {
            "kind": "ipq_tc1_rank_memmap_access_ledger",
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": asdict(arm),
            "rank": rank,
            "pre_global_endpoint_barrier_operations": endpoint_access,
            "post_global_endpoint_barrier_operations": scoring_access,
            "endpoint_barrier_receipt": asdict(local_barrier),
            "global_barrier_event": global_barrier_event,
            "integrity_rehash_after_global_endpoint_barrier": observed_integrity,
            "prebarrier_rgb_contract": "fixed row, frames 0:5 only",
            "prebarrier_action_contract": "fixed row, planned actions 0:13",
            "generic_prebarrier_getitem_available": False,
            "production_abc_dataset_constructed": False,
            "future_rgb_opened_before_global_endpoint_barrier": False,
            "all_memmap_index_operations_recorded": True,
            "auxiliary_target_array_opened": False,
            "teacher_feature_encoder_calls": 0,
            "protected_test_accessed": False,
        }
    )
    pilot.exclusive_json(access_ledger_path, access_ledger)
    row_path = output / f"rank_{rank:03d}.jsonl"
    with row_path.open("xb") as handle:
        for row in rows:
            handle.write(pilot.canonical_json(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    evidence_path = output / f"trajectory_rank_{rank:03d}.pt"
    torch.save(evidence, evidence_path)
    manifest = pilot.identity_payload(
        {
            "kind": KIND_RANK,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": asdict(arm),
            "rank": rank,
            "world_size": world,
            "assigned_clip_indexes": assigned,
            "row_count": len(rows),
            "actual_wan_calls": total_calls,
            "rows": pilot.file_record(row_path),
            "trajectory": pilot.file_record(evidence_path),
            "access_ledger": pilot.file_record(access_ledger_path),
            "endpoint_barrier_receipt": asdict(local_barrier),
            "global_barrier_event": global_barrier_event,
            "wall_seconds": time.perf_counter() - job_started,
            "peak_memory_allocated_bytes": max(
                row["peak_memory"]["allocated_bytes"] for row in rows
            ),
            "peak_memory_reserved_bytes": max(
                row["peak_memory"]["reserved_bytes"] for row in rows
            ),
            "both_training_completions_checked_before_endpoint_reader": True,
            "validation_content_rehash_after_global_endpoint_barrier": True,
            "production_abc_dataset_constructed": False,
            "future_rgb_opened_before_global_endpoint_barrier": False,
            "target_array_opened": False,
            "teacher_feature_encoder_calls": 0,
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
        training_resource = {
            code: pilot.read_json(
                value["trace_complete_record"]["path"],
                f"{code} training resource receipt",
            )
            for code, value in training.items()
        }
        artifact_bytes = sum(
            path.stat().st_size
            for path in Path(registration["output_root"]).rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        evaluation_wall = max(value["wall_seconds"] for value in manifests)
        evaluation_b200_hours = evaluation_wall * world / 3600.0
        training_b200_hours = sum(
            float(value["training_reserved_b200_hours"])
            for value in training_resource.values()
        )
        inventory = pilot.identity_payload(
            {
                "kind": KIND_INVENTORY,
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": asdict(arm),
                "source_commit": registration["tool_repository"]["git_commit"],
                "rank_manifests": manifests,
                "row_count": sum(value["row_count"] for value in manifests),
                "actual_wan_calls": sum(
                    value["actual_wan_calls"] for value in manifests
                ),
                "evaluation_wall_seconds_max": evaluation_wall,
                "evaluation_reserved_b200_hours": evaluation_b200_hours,
                "training_reserved_b200_hours_both_arms": training_b200_hours,
                "observed_b200_hours_including_this_evaluation": (
                    training_b200_hours + evaluation_b200_hours
                ),
                "artifact_bytes_under_registered_root": artifact_bytes,
                "peak_memory_allocated_bytes": max(
                    value["peak_memory_allocated_bytes"] for value in manifests
                ),
                "peak_memory_reserved_bytes": max(
                    value["peak_memory_reserved_bytes"] for value in manifests
                ),
                "validation_array_sha256": {
                    logical: observed_integrity[logical]
                    for logical in ("rgb", "actions")
                },
                "both_training_completions": {
                    code: {
                        key: value[key]
                        for key in (
                            "snapshot_record",
                            "config_record",
                            "completion_record",
                            "trace_record",
                            "trace_complete_record",
                        )
                    }
                    for code, value in training.items()
                },
                "access_ledgers": [
                    value["access_ledger"] for value in manifests
                ],
                "global_endpoint_barrier_passed": True,
                "validation_content_rehash_after_global_endpoint_barrier": True,
                "production_abc_dataset_constructed": False,
                "future_rgb_opened_before_global_endpoint_barrier": False,
                "target_array_opened": False,
                "teacher_feature_encoder_calls": 0,
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
    parser.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
