#!/usr/bin/env python3
"""Read-only analysis for the preregistered V-JEPA 2.1 controlled study.

The analyzer consumes the immutable study tree produced by
``tools/vjepa2_controlled_study.py``.  It never imports or invokes the V-JEPA
teacher, loads a training checkpoint, or mutates a run.  The quantitative
baseline is VPM, whose parameter schema matches A1/J0/J1.  V0 is retained as
an architecture/provenance control only: its legacy visualization does not
emit the paired latent/source/NFE grid required for a causal comparison.

Scientific held-out reconstruction evidence is paired by immutable
``(clip_id, clip_index)`` across all
128 fixed test clips.  Trainer visualization artifacts are diagnostic only
and never enter an aggregate, confidence interval, or success gate.  The
following fail-closed invariants are checked before any metric is reported:

* all seven completed-update milestones are present and each uses its pinned
  intermediate/final source-by-NFE grid;
* the auxiliary history length is zero (all V-JEPA bins are generated);
* online teacher calls are zero;
* the measured Wan call count equals the declared NFE;
* oracle sources are explicitly labeled leakage-only;
* autonomous and autonomous-shuffled NFE=1 endpoints are bit-identical.
* the final speed comparison is counterbalanced in one process on one B200,
  with both checkpoints resident and bit-identical observable inputs.

LACWM clock convention: ``sigma=1`` is Gaussian noise and ``sigma=0`` is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


SCHEMA_VERSION = 1
STUDY_KIND = "vjepa2_controlled_video_diffusion_study"
SIGMA_CONVENTION = "1=noise,0=clean"
COMPLETED_UPDATES = (1, 50, 100, 200, 400, 800, 1000)
STAGE_ENDPOINTS = (1, 50, 100, 200, 400, 600, 800, 1000)
NFE_STEPS = (1, 2, 4, 6, 8, 12, 20)
OBSERVED_HISTORY_FRAMES = 5
GENERATED_FUTURE_FRAMES = 8
EXPECTED_TEST_CLIPS = 128
EXPECTED_QUALITY_WORLD_SIZE = 8
EXPECTED_QUALITY_BATCH_PER_RANK = 2
SOURCES = (
    "autonomous",
    "off",
    "autonomous_shuffled",
    "oracle_matched",
    "oracle_shuffled",
)
DEPLOYABLE_SOURCES = ("autonomous", "off", "autonomous_shuffled")
ORACLE_SOURCES = ("oracle_matched", "oracle_shuffled")
LATENCY_SOURCES = ("autonomous", "off")
SOURCE_CODES = {
    "autonomous": 0,
    "off": 1,
    "oracle_matched": 2,
    "oracle_shuffled": 3,
    "autonomous_shuffled": 4,
}
ARM_DIRECTORIES = {
    "V0": "v0_original_explicit_action",
    "VPM": "vpm_parameter_matched_video",
    "A1": "a1_auxiliary_objective_only",
    "J0": "j0_joint_aligned",
    "J1": "j1_joint_auxiliary_leads",
}
QUANTITATIVE_ARMS = ("VPM", "A1", "J0", "J1")
EXPECTED_INTERVENTIONS = {
    "VPM": {
        "condition_on_tf": 0,
        "condition_mode_code": 0,
        "tf_loss_weight": 0.0,
        "parameter_matched_control": 1,
    },
    "A1": {
        "condition_on_tf": 0,
        "condition_mode_code": 0,
        "tf_loss_weight": 1.0,
        "parameter_matched_control": 0,
    },
    "J0": {
        "condition_on_tf": 1,
        "condition_mode_code": 1,
        "tf_loss_weight": 1.0,
        "parameter_matched_control": 0,
    },
    "J1": {
        "condition_on_tf": 1,
        "condition_mode_code": 1,
        "tf_loss_weight": 1.0,
        "parameter_matched_control": 0,
    },
}
METRIC_DIRECTIONS = {
    "video_future_nmse": "lower",
    "auxiliary_future_nmse": "lower",
    "auxiliary_future_cosine_similarity": "higher",
    "decoded_mse_unit_range": "lower",
    "decoded_psnr_db": "higher",
    "decoded_temporal_difference_mse_unit_range": "lower",
}
PRIMARY_METRIC = "decoded_temporal_difference_mse_unit_range"
QUALITY_GUARDRAILS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
)
ARTIFACT_RE = re.compile(r"^latent_trajectory_rank_([0-9]+)[.]safetensors$")
SOURCE_TENSOR_RE = re.compile(
    r"^(video_final|tf_final|decoded_future|wan_call_count)"
    r"(?:_(off|autonomous_shuffled|oracle_matched|oracle_shuffled))?"
    r"_nfe_([0-9]+)$"
)
LATENCY_RE = re.compile(
    r"^source_(autonomous|off)_nfe_([0-9]+)[.]json$"
)
PAIRED_LATENCY_BASENAME = "paired_j1_nfe4_vs_vpm_nfe8.json"
PAIRED_LATENCY_KIND = "vjepa2_controlled_study_paired_latency"
PAIRED_LATENCY_COMPARISON = "J1_autonomous_nfe4_vs_VPM_autonomous_nfe8"
PAIRED_WARMUP_PAIRS = 20
PAIRED_TIMED_PAIRS = 100
QUALITY_RANK_RE = re.compile(r"^rank_([0-9]{3})[.]jsonl$")
QUALITY_RANK_MANIFEST_RE = re.compile(
    r"^rank_([0-9]{3})_manifest[.]json$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _quality_grid(update: int) -> tuple[tuple[str, int], ...]:
    if update == 1000:
        return tuple((source, nfe) for source in SOURCES for nfe in NFE_STEPS)
    if update in COMPLETED_UPDATES:
        return (
            ("autonomous", 4),
            ("autonomous", 8),
            ("off", 4),
            ("off", 8),
            ("autonomous_shuffled", 4),
            ("autonomous_shuffled", 8),
            ("oracle_matched", 4),
            ("oracle_shuffled", 4),
        )
    raise StudyValidationError(f"quality grid is not pinned at update {update}")


class StudyValidationError(RuntimeError):
    """Raised when evidence is incomplete, unsafe, or not paired."""


@dataclass(frozen=True)
class ArtifactRecord:
    arm: str
    update: int
    dataset: str
    rank: int
    path: Path
    sidecar: Path
    sha256: str
    tensor_schema: Mapping[str, tuple[tuple[int, ...], str]]

    @property
    def unit(self) -> tuple[str, int]:
        return self.dataset, self.rank


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _hash_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StudyValidationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise StudyValidationError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StudyValidationError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise StudyValidationError(
            f"{label} must be a non-empty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _assert_inside(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StudyValidationError(f"{label} escapes the study root: {path}") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise StudyValidationError(
                    f"{label} contains a symlink component: {current}"
                )
        except FileNotFoundError as exc:
            raise StudyValidationError(
                f"{label} disappeared during validation: {current}"
            ) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StudyValidationError(
                    f"{label} contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyValidationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise StudyValidationError(f"{label} must contain one object: {path}")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StudyValidationError(
                    f"{label} contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise StudyValidationError(
                        f"{label} contains blank row {line_number}: {path}"
                    )
                row = json.loads(
                    line,
                    object_pairs_hook=reject_duplicates,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
                if not isinstance(row, dict):
                    raise StudyValidationError(
                        f"{label} row {line_number} is not an object: {path}"
                    )
                rows.append(row)
    except StudyValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StudyValidationError(f"{label} is invalid JSONL: {path}") from exc
    return rows


def _identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() == recorded


def _scalar_int(tensor: torch.Tensor, key: str) -> int:
    if tensor.numel() != 1 or tensor.dtype not in INTEGER_DTYPES:
        raise StudyValidationError(f"{key} must be one integer scalar")
    return int(tensor.item())


def _scalar_float(tensor: torch.Tensor, key: str) -> float:
    if tensor.numel() != 1 or not tensor.is_floating_point():
        raise StudyValidationError(f"{key} must be one floating-point scalar")
    value = float(tensor.item())
    if not math.isfinite(value):
        raise StudyValidationError(f"{key} must be finite")
    return value


def _validate_latent(
    tensor: torch.Tensor,
    key: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> None:
    if tensor.ndim != 5 or not tensor.is_floating_point():
        raise StudyValidationError(
            f"{key} must be floating [B,C,T,H,W], got "
            f"{tuple(tensor.shape)} {tensor.dtype}"
        )
    if shape is not None and tuple(tensor.shape) != shape:
        raise StudyValidationError(
            f"{key} shape {tuple(tensor.shape)} != {shape}"
        )
    if tensor.shape[0] != 1:
        raise StudyValidationError(
            f"{key} must contain exactly one paired evaluation unit"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise StudyValidationError(f"{key} contains non-finite values")


def _validate_decoded(
    tensor: torch.Tensor,
    key: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> None:
    if tensor.ndim != 5 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise StudyValidationError(
            f"{key} must be [1,3,T,H,W], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.uint8:
        raise StudyValidationError(f"{key} must be uint8")
    if shape is not None and tuple(tensor.shape) != shape:
        raise StudyValidationError(
            f"{key} shape {tuple(tensor.shape)} != {shape}"
        )
    if tensor.shape[2] < 2:
        raise StudyValidationError(f"{key} requires at least two frames")


def _source_infix(source: str) -> str:
    if source not in SOURCES:
        raise StudyValidationError(f"unknown source: {source!r}")
    return "" if source == "autonomous" else f"_{source}"


def _required_keys() -> set[str]:
    required = {
        "video_clean",
        "tf_clean",
        "ground_truth_future_uint8",
        "history_latent_frames",
        "auxiliary_history_latent_frames",
        "auxiliary_clean_available",
        "video_initial_state",
        "tf_initial_state",
        "tf_initial_noise",
        "condition_on_tf",
        "condition_mode_code",
        "video_only_control",
        "parameter_matched_control",
        "tf_loss_weight",
        "effective_state_gate",
        "effective_clock_gate",
        "evaluation_noise_seed",
        "evaluation_nfe_steps",
        "evaluation_condition_source_codes",
        "oracle_sources_are_leakage",
        "online_teacher_call_count",
    }
    for source in SOURCES:
        infix = _source_infix(source)
        for nfe in NFE_STEPS:
            required.update(
                {
                    f"video_final{infix}_nfe_{nfe}",
                    f"tf_final{infix}_nfe_{nfe}",
                    f"decoded_future{infix}_nfe_{nfe}",
                    f"wan_call_count{infix}_nfe_{nfe}",
                }
            )
    return required


def _schema_from_sidecar(
    payload: Mapping[str, Any], path: Path
) -> dict[str, tuple[tuple[int, ...], str]]:
    raw = payload.get("tensors")
    if not isinstance(raw, dict) or not raw:
        raise StudyValidationError(f"sidecar has no tensor schema: {path}")
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for key, record in raw.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise StudyValidationError(f"malformed tensor schema: {path}")
        shape = record.get("shape")
        dtype = record.get("dtype")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in shape
            )
            or not isinstance(dtype, str)
        ):
            raise StudyValidationError(
                f"invalid tensor schema for {key!r}: {path}"
            )
        result[key] = (tuple(shape), dtype)
    return result


def _validate_artifact(
    path: Path,
    *,
    root: Path,
    arm: str,
    update: int,
) -> ArtifactRecord:
    _assert_inside(path, root, "artifact")
    path = _regular_file(path, "quality artifact")
    match = ARTIFACT_RE.fullmatch(path.name)
    if match is None:
        raise StudyValidationError(f"unexpected artifact name: {path}")
    filename_rank = int(match.group(1))
    sidecar = _regular_file(path.with_suffix(".json"), "artifact sidecar")
    _assert_inside(sidecar, root, "artifact sidecar")
    payload = _read_json(sidecar, "artifact sidecar")
    iteration = payload.get("iteration")
    dataset = payload.get("dataset")
    rank = payload.get("global_rank")
    if iteration != update - 1:
        raise StudyValidationError(
            f"artifact iteration {iteration!r} != completed update {update} - 1"
        )
    if not isinstance(dataset, str) or not dataset or dataset != path.parent.name:
        raise StudyValidationError(f"artifact dataset provenance differs: {path}")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank != filename_rank
    ):
        raise StudyValidationError(f"artifact rank provenance differs: {path}")
    if payload.get("sigma_convention") != SIGMA_CONVENTION:
        raise StudyValidationError(f"artifact sigma convention differs: {path}")
    recorded_sha = payload.get("safetensors_sha256")
    if not isinstance(recorded_sha, str) or SHA256_RE.fullmatch(recorded_sha) is None:
        raise StudyValidationError(f"artifact sidecar SHA-256 is invalid: {path}")
    actual_sha = _hash_file(path)
    if actual_sha != recorded_sha:
        raise StudyValidationError(f"artifact SHA-256 mismatch: {path}")
    declared_schema = _schema_from_sidecar(payload, sidecar)
    missing = sorted(_required_keys() - set(declared_schema))
    if missing:
        raise StudyValidationError(
            f"artifact lacks required V-JEPA tensors: {missing}: {path}"
        )
    for key in declared_schema:
        if not key.startswith(
            ("video_final", "tf_final", "decoded_future", "wan_call_count")
        ):
            continue
        tensor_match = SOURCE_TENSOR_RE.fullmatch(key)
        if tensor_match is None:
            if key in {
                "video_initial_state",
                "tf_initial_state",
                "tf_initial_noise",
            }:
                continue
            raise StudyValidationError(
                f"unrecognized source/NFE tensor {key!r}: {path}"
            )
        source = tensor_match.group(2) or "autonomous"
        nfe = int(tensor_match.group(3))
        if source not in SOURCES or nfe not in NFE_STEPS:
            raise StudyValidationError(
                f"undeclared source/NFE tensor {key!r}: {path}"
            )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            actual_schema: dict[str, tuple[tuple[int, ...], str]] = {}
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                actual_schema[key] = (tuple(tensor.shape), str(tensor.dtype))
    except Exception as exc:
        raise StudyValidationError(f"cannot read artifact {path}: {exc}") from exc
    if actual_schema != declared_schema:
        raise StudyValidationError(
            f"artifact/sidecar tensor schema differs: {path}"
        )
    expected_metadata = {
        "iteration": str(update - 1),
        "dataset": dataset,
        "sigma_convention": SIGMA_CONVENTION,
    }
    for key, wanted in expected_metadata.items():
        if metadata.get(key) != wanted:
            raise StudyValidationError(
                f"artifact metadata {key!r} differs: "
                f"{metadata.get(key)!r} != {wanted!r}: {path}"
            )
    return ArtifactRecord(
        arm=arm,
        update=update,
        dataset=dataset,
        rank=rank,
        path=path,
        sidecar=sidecar,
        sha256=actual_sha,
        tensor_schema=actual_schema,
    )


def _discover_update(
    arm_root: Path, *, arm: str, update: int, study_root: Path
) -> dict[tuple[str, int], ArtifactRecord]:
    iteration_root = arm_root / "visualization" / f"iter_{update - 1}"
    iteration_root = _canonical_directory(
        iteration_root, f"{arm} update {update} visualization"
    )
    _assert_inside(iteration_root, study_root, "visualization directory")
    paths = sorted(iteration_root.glob("*/latent_trajectory_rank_*.safetensors"))
    if not paths:
        raise StudyValidationError(
            f"{arm} update {update} contains no latent quality artifacts"
        )
    records: dict[tuple[str, int], ArtifactRecord] = {}
    for path in paths:
        record = _validate_artifact(
            path,
            root=study_root,
            arm=arm,
            update=update,
        )
        if record.unit in records:
            raise StudyValidationError(
                f"{arm} update {update} duplicates unit {record.unit}"
            )
        records[record.unit] = record
    for dataset in sorted({dataset for dataset, _ in records}):
        ranks = sorted(rank for name, rank in records if name == dataset)
        if ranks != list(range(ranks[-1] + 1)):
            raise StudyValidationError(
                f"{arm} update {update} has non-contiguous {dataset} ranks: {ranks}"
            )
    all_artifacts = sorted(iteration_root.rglob("latent_trajectory_rank_*.safetensors"))
    if paths != all_artifacts:
        raise StudyValidationError(
            f"{arm} update {update} has nested or ambiguous artifact paths"
        )
    return records


def _nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history: int,
    key: str,
) -> float:
    prediction = prediction[:, :, history:].double()
    target = target[:, :, history:].double()
    denominator = torch.sum(target.square()).item()
    if not math.isfinite(denominator) or denominator <= 0:
        raise StudyValidationError(f"{key} target energy is not positive")
    value = torch.sum((prediction - target).square()).item() / denominator
    if not math.isfinite(value):
        raise StudyValidationError(f"{key} NMSE is non-finite")
    return float(value)


def _cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history: int,
    key: str,
) -> float:
    prediction = prediction[:, :, history:].double().reshape(-1)
    target = target[:, :, history:].double().reshape(-1)
    denominator = torch.linalg.vector_norm(prediction) * torch.linalg.vector_norm(
        target
    )
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise StudyValidationError(f"{key} cosine norm is not positive")
    value = float(torch.dot(prediction, target).item() / denominator_value)
    if not math.isfinite(value):
        raise StudyValidationError(f"{key} cosine is non-finite")
    return min(1.0, max(-1.0, value))


def _decoded_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.double() / 255.0
    target = target.double() / 255.0
    mse = float(torch.mean((prediction - target).square()).item())
    temporal = float(
        torch.mean(
            (
                torch.diff(prediction, dim=2)
                - torch.diff(target, dim=2)
            ).square()
        ).item()
    )
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    if not all(math.isfinite(value) for value in (mse, temporal, psnr)):
        raise StudyValidationError("decoded metrics are non-finite")
    return {
        "decoded_mse_unit_range": mse,
        "decoded_psnr_db": psnr,
        "decoded_temporal_difference_mse_unit_range": temporal,
    }


def _analyze_artifact(
    record: ArtifactRecord,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    expected = EXPECTED_INTERVENTIONS[record.arm]
    try:
        with safe_open(str(record.path), framework="pt", device="cpu") as handle:
            video_clean = handle.get_tensor("video_clean")
            auxiliary_clean = handle.get_tensor("tf_clean")
            ground_truth = handle.get_tensor("ground_truth_future_uint8")
            initial_video = handle.get_tensor("video_initial_state")
            initial_auxiliary = handle.get_tensor("tf_initial_state")
            initial_auxiliary_noise = handle.get_tensor("tf_initial_noise")
            _validate_latent(video_clean, "video_clean")
            _validate_latent(
                auxiliary_clean,
                "tf_clean",
                shape=(1, 64, 4, 24, 120),
            )
            _validate_decoded(ground_truth, "ground_truth_future_uint8")
            if ground_truth.shape[2] != GENERATED_FUTURE_FRAMES:
                raise StudyValidationError(
                    "decoded future length differs from the pinned eight frames"
                )
            _validate_latent(
                initial_video,
                "video_initial_state",
                shape=tuple(video_clean.shape),
            )
            _validate_latent(
                initial_auxiliary,
                "tf_initial_state",
                shape=tuple(auxiliary_clean.shape),
            )
            _validate_latent(
                initial_auxiliary_noise,
                "tf_initial_noise",
                shape=tuple(auxiliary_clean.shape),
            )
            if initial_video.dtype != video_clean.dtype:
                raise StudyValidationError("video initial/clean dtypes differ")
            if (
                initial_auxiliary.dtype != auxiliary_clean.dtype
                or initial_auxiliary_noise.dtype != auxiliary_clean.dtype
            ):
                raise StudyValidationError("auxiliary initial/clean dtypes differ")

            video_history = _scalar_int(
                handle.get_tensor("history_latent_frames"),
                "history_latent_frames",
            )
            auxiliary_history = _scalar_int(
                handle.get_tensor("auxiliary_history_latent_frames"),
                "auxiliary_history_latent_frames",
            )
            if not 0 <= video_history < video_clean.shape[2]:
                raise StudyValidationError(
                    "video history must leave at least one generated latent frame"
                )
            if auxiliary_history != 0:
                raise StudyValidationError(
                    "V-JEPA auxiliary_history_latent_frames must equal 0; "
                    "all four bins must be generated"
                )
            if _scalar_int(
                handle.get_tensor("auxiliary_clean_available"),
                "auxiliary_clean_available",
            ) != 1:
                raise StudyValidationError(
                    "paired quality/oracle evaluation requires cached clean V-JEPA"
                )
            if not torch.equal(initial_auxiliary, initial_auxiliary_noise):
                raise StudyValidationError(
                    "zero-history auxiliary initial state must equal local noise"
                )
            if _scalar_int(
                handle.get_tensor("online_teacher_call_count"),
                "online_teacher_call_count",
            ) != 0:
                raise StudyValidationError(
                    "V-JEPA teacher was invoked during training/inference"
                )
            if _scalar_int(
                handle.get_tensor("oracle_sources_are_leakage"),
                "oracle_sources_are_leakage",
            ) != 1:
                raise StudyValidationError(
                    "oracle sources are not explicitly labeled leakage-only"
                )
            if _scalar_int(
                handle.get_tensor("video_only_control"),
                "video_only_control",
            ) != 0:
                raise StudyValidationError(
                    "quantitative arms must use the common dual parameter schema"
                )
            for key in ("condition_on_tf", "condition_mode_code"):
                actual = _scalar_int(handle.get_tensor(key), key)
                if actual != expected[key]:
                    raise StudyValidationError(
                        f"{record.arm} {key}={actual} != {expected[key]}"
                    )
            actual_parameter_control = _scalar_int(
                handle.get_tensor("parameter_matched_control"),
                "parameter_matched_control",
            )
            if actual_parameter_control != expected["parameter_matched_control"]:
                raise StudyValidationError(
                    f"{record.arm} parameter-matched flag differs"
                )
            actual_weight = _scalar_float(
                handle.get_tensor("tf_loss_weight"), "tf_loss_weight"
            )
            if not math.isclose(
                actual_weight,
                expected["tf_loss_weight"],
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise StudyValidationError(
                    f"{record.arm} auxiliary loss weight differs"
                )
            effective_gates = {
                key: _scalar_float(handle.get_tensor(key), key)
                for key in ("effective_state_gate", "effective_clock_gate")
            }
            if record.arm in {"VPM", "A1"} and any(
                value != 0.0 for value in effective_gates.values()
            ):
                raise StudyValidationError(
                    f"{record.arm} no-injection gates must remain exactly zero"
                )
            nfe = tuple(
                int(value)
                for value in handle.get_tensor("evaluation_nfe_steps").tolist()
            )
            if nfe != NFE_STEPS:
                raise StudyValidationError(
                    f"evaluation NFE inventory {nfe} != {NFE_STEPS}"
                )
            source_codes = tuple(
                int(value)
                for value in handle.get_tensor(
                    "evaluation_condition_source_codes"
                ).tolist()
            )
            expected_codes = tuple(SOURCE_CODES[source] for source in SOURCES)
            if source_codes != expected_codes:
                raise StudyValidationError(
                    f"evaluation source codes {source_codes} != {expected_codes}"
                )
            noise_seed = _scalar_int(
                handle.get_tensor("evaluation_noise_seed"),
                "evaluation_noise_seed",
            )
            if noise_seed < 0:
                raise StudyValidationError("evaluation noise seed is negative")

            metrics: dict[str, dict[str, dict[str, float]]] = {}
            for source in SOURCES:
                metrics[source] = {}
                infix = _source_infix(source)
                for steps in NFE_STEPS:
                    call_key = f"wan_call_count{infix}_nfe_{steps}"
                    calls = _scalar_int(handle.get_tensor(call_key), call_key)
                    if calls != steps:
                        raise StudyValidationError(
                            f"{call_key}={calls}; actual Wan calls must equal NFE"
                        )
                    video_key = f"video_final{infix}_nfe_{steps}"
                    auxiliary_key = f"tf_final{infix}_nfe_{steps}"
                    decoded_key = f"decoded_future{infix}_nfe_{steps}"
                    video = handle.get_tensor(video_key)
                    auxiliary = handle.get_tensor(auxiliary_key)
                    decoded = handle.get_tensor(decoded_key)
                    _validate_latent(
                        video, video_key, shape=tuple(video_clean.shape)
                    )
                    _validate_latent(
                        auxiliary,
                        auxiliary_key,
                        shape=tuple(auxiliary_clean.shape),
                    )
                    _validate_decoded(
                        decoded, decoded_key, shape=tuple(ground_truth.shape)
                    )
                    if video.dtype != video_clean.dtype:
                        raise StudyValidationError(
                            f"{video_key} dtype differs from video_clean"
                        )
                    if auxiliary.dtype != auxiliary_clean.dtype:
                        raise StudyValidationError(
                            f"{auxiliary_key} dtype differs from tf_clean"
                        )
                    decoded_result = _decoded_metrics(decoded, ground_truth)
                    metrics[source][str(steps)] = {
                        "video_future_nmse": _nmse(
                            video, video_clean, video_history, video_key
                        ),
                        "auxiliary_future_nmse": _nmse(
                            auxiliary,
                            auxiliary_clean,
                            auxiliary_history,
                            auxiliary_key,
                        ),
                        "auxiliary_future_cosine_similarity": _cosine(
                            auxiliary,
                            auxiliary_clean,
                            auxiliary_history,
                            auxiliary_key,
                        ),
                        **decoded_result,
                    }

            prefixes = ("video_final", "tf_final", "decoded_future")
            if record.arm in {"VPM", "A1"}:
                # These controls disable both state and clock injection.  The
                # auxiliary head can evolve, but changing the conditioning
                # source cannot alter either denoising stream within the same
                # checkpoint while both effective gates are exactly zero.
                for steps in NFE_STEPS:
                    for prefix in prefixes:
                        reference = handle.get_tensor(
                            f"{prefix}_nfe_{steps}"
                        )
                        for source in SOURCES[1:]:
                            candidate = handle.get_tensor(
                                f"{prefix}{_source_infix(source)}_nfe_{steps}"
                            )
                            if not torch.equal(reference, candidate):
                                raise StudyValidationError(
                                    f"{record.arm} no-injection source no-op "
                                    f"failed for {prefix}, {source}, NFE={steps}"
                                )
            else:
                # At sigma=1 every non-off source supplies the same local pure
                # noise on the sole call.  Their NFE=1 result is an exact
                # endpoint identity even though later calls can diverge.
                non_off_sources = (
                    "autonomous",
                    "autonomous_shuffled",
                    "oracle_matched",
                    "oracle_shuffled",
                )
                for prefix in prefixes:
                    reference = handle.get_tensor(f"{prefix}_nfe_1")
                    for source in non_off_sources[1:]:
                        candidate = handle.get_tensor(
                            f"{prefix}{_source_infix(source)}_nfe_1"
                        )
                        if not torch.equal(reference, candidate):
                            raise StudyValidationError(
                                "non-off pure-noise NFE=1 identity failed for "
                                f"{record.arm}, {prefix}, {source}"
                            )
            identity = {
                "dataset": record.dataset,
                "rank": record.rank,
                "video_history_latent_frames": video_history,
                "auxiliary_history_latent_frames": auxiliary_history,
                "video_clean_sha256": _hash_tensor(video_clean),
                "auxiliary_clean_sha256": _hash_tensor(auxiliary_clean),
                "ground_truth_sha256": _hash_tensor(ground_truth),
                "video_initial_state_sha256": _hash_tensor(initial_video),
                "auxiliary_initial_state_sha256": _hash_tensor(
                    initial_auxiliary
                ),
                "auxiliary_initial_noise_sha256": _hash_tensor(
                    initial_auxiliary_noise
                ),
                "evaluation_noise_seed": noise_seed,
                "evaluation_nfe_steps": list(NFE_STEPS),
                "evaluation_sources": list(SOURCES),
                "online_teacher_call_count": 0,
                "oracle_sources_are_leakage": True,
                "effective_state_gate": effective_gates[
                    "effective_state_gate"
                ],
                "effective_clock_gate": effective_gates[
                    "effective_clock_gate"
                ],
                "artifact_path": str(record.path),
                "artifact_sha256": record.sha256,
            }
    except StudyValidationError:
        raise
    except Exception as exc:
        raise StudyValidationError(
            f"failed to analyze {record.path}: {exc}"
        ) from exc
    return metrics, identity


def _summary(
    values: Sequence[float],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise StudyValidationError(f"invalid aggregate values for {label}")
    rng = np.random.default_rng(_derived_seed(seed, label))
    indexes = rng.integers(
        0, array.size, size=(bootstrap_samples, array.size), endpoint=False
    )
    means = array[indexes].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {
        "n_paired_units": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
        },
    }


def _paired_effect(
    left: Sequence[float],
    reference: Sequence[float],
    *,
    direction: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        left_array.ndim != 1
        or left_array.shape != reference_array.shape
        or left_array.size == 0
        or not np.isfinite(left_array).all()
        or not np.isfinite(reference_array).all()
    ):
        raise StudyValidationError(f"invalid paired values for {label}")
    if direction not in {"lower", "higher"}:
        raise StudyValidationError(f"unknown metric direction: {direction}")
    left_mean = float(left_array.mean())
    reference_mean = float(reference_array.mean())
    denominator = abs(reference_mean)
    if denominator <= 0:
        relative = None
        bootstrap_ci = None
    else:
        sign = -1.0 if direction == "lower" else 1.0
        relative = sign * (left_mean - reference_mean) / denominator
        rng = np.random.default_rng(_derived_seed(seed, label))
        indexes = rng.integers(
            0,
            left_array.size,
            size=(bootstrap_samples, left_array.size),
            endpoint=False,
        )
        left_means = left_array[indexes].mean(axis=1)
        reference_means = reference_array[indexes].mean(axis=1)
        denominators = np.abs(reference_means)
        if np.any(denominators <= 0):
            bootstrap_ci = None
        else:
            effects = sign * (left_means - reference_means) / denominators
            tail = (1.0 - confidence) / 2.0
            low, high = np.quantile(effects, [tail, 1.0 - tail])
            bootstrap_ci = {
                "confidence": confidence,
                "low": float(low),
                "high": float(high),
            }
    signed_deltas = (
        reference_array - left_array
        if direction == "lower"
        else left_array - reference_array
    )
    return {
        "n_paired_units": int(left_array.size),
        "definition": (
            "positive relative_improvement means left is better; paired "
            "bootstrap resamples immutable clip_id/clip_index units"
        ),
        "left_mean": left_mean,
        "reference_mean": reference_mean,
        "mean_favorable_delta": float(signed_deltas.mean()),
        "relative_improvement": (
            None if relative is None else float(relative)
        ),
        "relative_improvement_percent": (
            None if relative is None else float(100.0 * relative)
        ),
        "bootstrap_ci": bootstrap_ci,
        "favorable_unit_fraction": float(np.mean(signed_deltas > 0)),
    }


def _counterbalanced_paired_latency_effect(
    j1_latency_ms: Sequence[float],
    vpm_latency_ms: Sequence[float],
    execution_orders: Sequence[Sequence[str]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Estimate J1's speedup while preserving the two execution-order strata."""

    j1 = np.asarray(j1_latency_ms, dtype=np.float64)
    vpm = np.asarray(vpm_latency_ms, dtype=np.float64)
    if (
        j1.ndim != 1
        or j1.shape != vpm.shape
        or j1.size != PAIRED_TIMED_PAIRS
        or not np.isfinite(j1).all()
        or not np.isfinite(vpm).all()
        or np.any(j1 <= 0)
        or np.any(vpm <= 0)
        or len(execution_orders) != j1.size
    ):
        raise StudyValidationError(
            f"invalid counterbalanced latency values for {label}"
        )
    strata: dict[str, np.ndarray] = {}
    for first in ("J1", "VPM"):
        indexes = [
            index
            for index, order in enumerate(execution_orders)
            if tuple(order) == (
                (first, "VPM") if first == "J1" else (first, "J1")
            )
        ]
        if len(indexes) != PAIRED_TIMED_PAIRS // 2:
            raise StudyValidationError(
                f"latency order stratum {first}-first is not balanced"
            )
        strata[first] = np.asarray(indexes, dtype=np.int64)

    j1_mean = float(j1.mean())
    vpm_mean = float(vpm.mean())
    if vpm_mean <= 0:
        raise StudyValidationError("paired VPM latency mean is not positive")
    relative = (vpm_mean - j1_mean) / vpm_mean
    rng = np.random.default_rng(_derived_seed(seed, label))
    stratum_effect_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for first in ("J1", "VPM"):
        indexes = strata[first]
        draws = rng.integers(
            0,
            indexes.size,
            size=(bootstrap_samples, indexes.size),
            endpoint=False,
        )
        sampled = indexes[draws]
        stratum_effect_inputs.append(
            (j1[sampled].mean(axis=1), vpm[sampled].mean(axis=1))
        )
    bootstrap_j1 = np.mean(
        np.stack([item[0] for item in stratum_effect_inputs], axis=0),
        axis=0,
    )
    bootstrap_vpm = np.mean(
        np.stack([item[1] for item in stratum_effect_inputs], axis=0),
        axis=0,
    )
    if np.any(bootstrap_vpm <= 0):
        raise StudyValidationError(
            "paired bootstrap produced non-positive VPM latency"
        )
    effects = (bootstrap_vpm - bootstrap_j1) / bootstrap_vpm
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(effects, [tail, 1.0 - tail])
    favorable_differences = vpm - j1
    return {
        "n_paired_rounds": int(j1.size),
        "bootstrap_unit": (
            "paired timing round, resampled separately within J1-first and "
            "VPM-first execution-order strata"
        ),
        "execution_order_strata": {
            "J1_first": int(strata["J1"].size),
            "VPM_first": int(strata["VPM"].size),
        },
        "J1_mean_ms": j1_mean,
        "VPM_mean_ms": vpm_mean,
        "mean_favorable_difference_ms": float(
            favorable_differences.mean()
        ),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(100.0 * relative),
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
        },
        "favorable_pair_fraction": float(
            np.mean(favorable_differences > 0)
        ),
    }


def _paired_gap_closure(
    off: Sequence[float],
    autonomous: Sequence[float],
    oracle: Sequence[float],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    off_array = np.asarray(off, dtype=np.float64)
    autonomous_array = np.asarray(autonomous, dtype=np.float64)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    if (
        off_array.ndim != 1
        or off_array.shape != autonomous_array.shape
        or off_array.shape != oracle_array.shape
        or off_array.size == 0
        or not np.isfinite(off_array).all()
        or not np.isfinite(autonomous_array).all()
        or not np.isfinite(oracle_array).all()
    ):
        raise StudyValidationError(f"invalid paired gap values for {label}")
    numerator = float(np.mean(off_array - autonomous_array))
    denominator = float(np.mean(off_array - oracle_array))
    rng = np.random.default_rng(_derived_seed(seed, label))
    indexes = rng.integers(
        0,
        off_array.size,
        size=(bootstrap_samples, off_array.size),
        endpoint=False,
    )
    bootstrap_numerator = np.mean(
        off_array[indexes] - autonomous_array[indexes], axis=1
    )
    bootstrap_denominator = np.mean(
        off_array[indexes] - oracle_array[indexes], axis=1
    )
    valid = bootstrap_denominator > 0
    if denominator <= 0 or int(np.sum(valid)) < int(0.99 * bootstrap_samples):
        return {
            "n_paired_units": int(off_array.size),
            "definition": (
                "mean(off-autonomous)/mean(off-oracle); positive means the "
                "deployable predictor closes positive oracle headroom"
            ),
            "oracle_headroom_mean": denominator,
            "gap_closure_fraction": None,
            "bootstrap_ci": None,
            "valid_bootstrap_fraction": float(np.mean(valid)),
        }
    values = bootstrap_numerator[valid] / bootstrap_denominator[valid]
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return {
        "n_paired_units": int(off_array.size),
        "definition": (
            "mean(off-autonomous)/mean(off-oracle); positive means the "
            "deployable predictor closes positive oracle headroom"
        ),
        "oracle_headroom_mean": denominator,
        "gap_closure_fraction": numerator / denominator,
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
        },
        "valid_bootstrap_fraction": float(np.mean(valid)),
    }


def _values(
    analyzed: Mapping[
        str, Mapping[int, Mapping[tuple[str, int], Mapping[str, Any]]]
    ],
    *,
    arm: str,
    update: int,
    source: str,
    nfe: int,
    metric: str,
    units: Sequence[tuple[str, int]],
) -> list[float]:
    return [
        float(
            analyzed[arm][update][unit]["metrics"][source][str(nfe)][metric]
        )
        for unit in units
    ]


def _comparison(
    analyzed: Mapping[str, Mapping[int, Mapping[tuple[str, int], Mapping[str, Any]]]],
    *,
    left_arm: str,
    reference_arm: str,
    update: int,
    left_source: str,
    reference_source: str,
    left_nfe: int,
    reference_nfe: int,
    units: Sequence[tuple[str, int]],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        metrics[metric] = _paired_effect(
            _values(
                analyzed,
                arm=left_arm,
                update=update,
                source=left_source,
                nfe=left_nfe,
                metric=metric,
                units=units,
            ),
            _values(
                analyzed,
                arm=reference_arm,
                update=update,
                source=reference_source,
                nfe=reference_nfe,
                metric=metric,
                units=units,
            ),
            direction=direction,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
            label=f"{label}:{metric}",
        )
    oracle = left_source in ORACLE_SOURCES or reference_source in ORACLE_SOURCES
    return {
        "left": {
            "arm": left_arm,
            "completed_updates": update,
            "source": left_source,
            "nfe": left_nfe,
        },
        "reference": {
            "arm": reference_arm,
            "completed_updates": update,
            "source": reference_source,
            "nfe": reference_nfe,
        },
        "paired_by": "identical fixed-test clip_id/clip_index unit",
        "oracle_leakage": oracle,
        "deployable_evidence": not oracle,
        "metrics": metrics,
    }


def _manifest_and_stage_inventory(
    study_root: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    study_path = _regular_file(
        study_root / "study_manifest.json", "study manifest"
    )
    study = _read_json(study_path, "study manifest")
    problems: list[str] = []
    if study.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version differs")
    if study.get("kind") != STUDY_KIND:
        problems.append("kind differs")
    if Path(str(study.get("study_root", ""))) != study_root:
        problems.append("study_root differs")
    if not _identity_valid(study):
        problems.append("identity SHA-256 is invalid")
    schedule = study.get("schedule", {})
    inference = study.get("inference", {})
    if schedule.get("completed_update_milestones") != list(COMPLETED_UPDATES):
        problems.append("completed-update milestones differ")
    if inference.get("nfe") != list(NFE_STEPS):
        problems.append("inference NFE inventory differs")
    if inference.get("sources") != list(SOURCES):
        problems.append("inference source inventory differs")
    if inference.get("deployable_sources") != list(DEPLOYABLE_SOURCES):
        problems.append("deployable source labels differ")
    if inference.get("oracle_leakage_only_sources") != list(ORACLE_SOURCES):
        problems.append("oracle leakage labels differ")
    if inference.get("teacher_invocations_allowed") != 0:
        problems.append("teacher invocation allowance is non-zero")
    if inference.get("actual_wan_calls_must_equal_nfe") is not True:
        problems.append("Wan-call/NFE contract is disabled")
    latency_protocol = inference.get("latency_protocol")
    final_latency = (
        latency_protocol.get("final_claim_comparison")
        if isinstance(latency_protocol, dict)
        else None
    )
    if (
        not isinstance(latency_protocol, dict)
        or not isinstance(
            latency_protocol.get("per_arm_grid_telemetry"), dict
        )
        or latency_protocol["per_arm_grid_telemetry"].get("claim_role")
        != "diagnostic_only"
        or not isinstance(final_latency, dict)
        or final_latency.get("arms")
        != [
            {"arm": "J1", "source": "autonomous", "nfe": 4},
            {"arm": "VPM", "source": "autonomous", "nfe": 8},
        ]
        or final_latency.get("same_slurm_allocation") is not True
        or final_latency.get("same_node") is not True
        or final_latency.get("same_B200") is not True
        or final_latency.get("same_process") is not True
        or final_latency.get("both_models_resident") is not True
        or final_latency.get("identical_immutable_batch_inputs") is not True
        or final_latency.get("warmup_pairs") != PAIRED_WARMUP_PAIRS
        or final_latency.get("timed_pairs") != PAIRED_TIMED_PAIRS
        or final_latency.get("counterbalance")
        != "even pair J1-first; odd pair VPM-first"
        or final_latency.get("forward_hooks_active_during_timing") is not False
        or final_latency.get("future_ground_truth_available_to_sampler")
        is not False
        or final_latency.get("clean_auxiliary_available_to_sampler")
        is not False
        or final_latency.get("online_teacher_calls") != 0
    ):
        problems.append("final paired-latency protocol differs")
    slurm_contract = study.get("slurm")
    paired_slurm = (
        slurm_contract.get("paired_latency_post_study_job")
        if isinstance(slurm_contract, dict)
        else None
    )
    if (
        not isinstance(paired_slurm, dict)
        or paired_slurm.get("dependency") != "afterok:final_stage_array"
        or paired_slurm.get("nodes") != 1
        or paired_slurm.get("gpus") != 1
        or paired_slurm.get("runs_final_analyzer_after_benchmark") is not True
    ):
        problems.append("paired post-study Slurm contract differs")
    wandb = study.get("wandb")
    if (
        not isinstance(wandb, dict)
        or wandb.get("entity") != "zijiandu"
        or wandb.get("project") != "dual-video-diffusion-private"
        or wandb.get("access") != "PRIVATE"
        or wandb.get("group") is not None
        or wandb.get("authenticated_viewer_username") != "zijiandu"
        or not isinstance(wandb.get("authenticated_viewer_email"), str)
        or wandb.get("user_requested_email") != "ldu@nvidia.edu"
        or wandb.get("authenticated_email_matches_user_request")
        != (
            wandb.get("authenticated_viewer_email") == "ldu@nvidia.edu"
        )
    ):
        problems.append("W&B authenticated viewer provenance differs")
    if problems:
        raise StudyValidationError(
            "invalid study manifest: " + "; ".join(problems)
        )

    arm_roots: dict[str, Path] = {}
    stages: dict[str, Any] = {}
    study_arms = study.get("arms", {})
    for index, (code, directory) in enumerate(ARM_DIRECTORIES.items()):
        root = _canonical_directory(study_root / directory, f"{code} arm root")
        _assert_inside(root, study_root, f"{code} arm root")
        arm_roots[code] = root
        arm_manifest_path = _regular_file(
            root / "arm_manifest.json", f"{code} arm manifest"
        )
        arm_manifest = _read_json(arm_manifest_path, f"{code} arm manifest")
        expected_contract = study_arms.get(str(index), {})
        if (
            not _identity_valid(arm_manifest)
            or arm_manifest.get("kind") != "vjepa2_controlled_study_arm"
            or arm_manifest.get("arm") != {
                key: value
                for key, value in expected_contract.items()
                if key
                not in {
                    "array_task_id",
                    "common_dual_schema",
                    "inference_nfe",
                    "evaluation_sources",
                }
            }
            or arm_manifest.get("run_dir") != str(root)
            or arm_manifest.get("study_identity_sha256")
            != study.get("identity_sha256")
        ):
            raise StudyValidationError(f"{code} arm manifest is invalid")

        cumulative_wall = 0
        stage_records: dict[str, Any] = {}
        for endpoint in STAGE_ENDPOINTS:
            stage_manifest_path = _regular_file(
                root / f"stage_manifest_update_{endpoint:04d}.json",
                f"{code} stage {endpoint} manifest",
            )
            stage_manifest = _read_json(
                stage_manifest_path, f"{code} stage {endpoint} manifest"
            )
            if (
                not _identity_valid(stage_manifest)
                or stage_manifest.get("kind")
                != "vjepa2_controlled_study_stage"
                or stage_manifest.get("arm_identity_sha256")
                != arm_manifest.get("identity_sha256")
                or stage_manifest.get("arm_code") != code
                or stage_manifest.get("stage_endpoint_completed_updates")
                != endpoint
                or stage_manifest.get("primary_milestone")
                != (endpoint in COMPLETED_UPDATES)
                or stage_manifest.get("trainer_terminal_iteration")
                != endpoint - 1
            ):
                raise StudyValidationError(
                    f"{code} stage {endpoint} manifest violates provenance"
                )
            resolved_record = stage_manifest.get("resolved_config")
            if not isinstance(resolved_record, dict):
                raise StudyValidationError(
                    f"{code} stage {endpoint} lacks resolved-config provenance"
                )
            resolved_path = _regular_file(
                Path(str(resolved_record.get("path", ""))),
                f"{code} stage {endpoint} resolved config",
            )
            _assert_inside(
                resolved_path, study_root, f"{code} stage {endpoint} config"
            )
            if (
                resolved_path
                != root / f"resolved_update_{endpoint:04d}.yaml"
                or resolved_record.get("sha256") != _hash_file(resolved_path)
                or resolved_record.get("bytes") != resolved_path.stat().st_size
            ):
                raise StudyValidationError(
                    f"{code} stage {endpoint} resolved-config provenance differs"
                )
            path = _regular_file(
                root / f"stage_outcome_update_{endpoint:04d}.json",
                f"{code} stage {endpoint} outcome",
            )
            payload = _read_json(path, f"{code} stage {endpoint} outcome")
            if (
                not _identity_valid(payload)
                or payload.get("kind")
                != "vjepa2_controlled_study_stage_outcome"
                or payload.get("arm_identity_sha256")
                != arm_manifest.get("identity_sha256")
                or payload.get("stage_identity_sha256")
                != stage_manifest.get("identity_sha256")
                or payload.get("arm_code") != code
                or payload.get("completed_updates") != endpoint
                or payload.get("primary_milestone")
                != (endpoint in COMPLETED_UPDATES)
                or payload.get("teacher_invocations_during_training") != 0
                or payload.get("cache_extraction_wall_time_included") is not False
            ):
                raise StudyValidationError(
                    f"{code} stage {endpoint} outcome violates provenance"
                )
            wall = payload.get(
                "stage_wall_seconds_including_validation_and_visualization"
            )
            if isinstance(wall, bool) or not isinstance(wall, int) or wall < 0:
                raise StudyValidationError(
                    f"{code} stage {endpoint} wall time is invalid"
                )
            observed_snapshot = payload.get(
                "snapshot_observed_at_stage_end"
            )
            if (
                not isinstance(observed_snapshot, dict)
                or observed_snapshot.get("path") != str(root / "snapshot.pt")
                or not isinstance(observed_snapshot.get("sha256"), str)
                or SHA256_RE.fullmatch(observed_snapshot["sha256"]) is None
                or isinstance(observed_snapshot.get("bytes"), bool)
                or not isinstance(observed_snapshot.get("bytes"), int)
                or observed_snapshot["bytes"] <= 0
            ):
                raise StudyValidationError(
                    f"{code} stage {endpoint} snapshot provenance differs"
                )
            cumulative_wall += wall
            stage_records[str(endpoint)] = {
                "stage_manifest_path": str(stage_manifest_path),
                "stage_manifest_sha256": _hash_file(stage_manifest_path),
                "stage_identity_sha256": stage_manifest.get("identity_sha256"),
                "resolved_config": {
                    "path": str(resolved_path),
                    "sha256": resolved_record["sha256"],
                    "bytes": resolved_record["bytes"],
                },
                "path": str(path),
                "sha256": _hash_file(path),
                "stage_outcome_identity_sha256": payload[
                    "identity_sha256"
                ],
                "snapshot_observed_at_stage_end": {
                    "path": observed_snapshot["path"],
                    "sha256": observed_snapshot["sha256"],
                    "bytes": observed_snapshot["bytes"],
                },
                "stage_wall_seconds_including_validation_and_visualization": wall,
                "cumulative_wall_seconds_including_validation_and_visualization": (
                    cumulative_wall
                ),
                "primary_milestone": endpoint in COMPLETED_UPDATES,
                "teacher_invocations_during_training": 0,
                "cache_extraction_wall_time_included": False,
            }
        outcome_path = _regular_file(root / "outcome.json", f"{code} outcome")
        outcome = _read_json(outcome_path, f"{code} outcome")
        if (
            not _identity_valid(outcome)
            or outcome.get("kind") != "vjepa2_controlled_study_arm_outcome"
            or outcome.get("arm_identity_sha256")
            != arm_manifest.get("identity_sha256")
            or outcome.get("arm_code") != code
            or outcome.get("completed_updates") != 1000
            or outcome.get("primary_milestones") != list(COMPLETED_UPDATES)
        ):
            raise StudyValidationError(f"{code} final outcome is invalid")
        final_snapshot = outcome.get("final_snapshot")
        if not isinstance(final_snapshot, dict):
            raise StudyValidationError(
                f"{code} outcome lacks final snapshot provenance"
            )
        final_snapshot_path = _regular_file(
            Path(str(final_snapshot.get("path", ""))),
            f"{code} final snapshot",
        )
        _assert_inside(final_snapshot_path, study_root, f"{code} final snapshot")
        if (
            final_snapshot_path != root / "snapshot.pt"
            or final_snapshot.get("sha256") != _hash_file(final_snapshot_path)
            or final_snapshot.get("bytes") != final_snapshot_path.stat().st_size
            or final_snapshot
            != stage_records["1000"]["snapshot_observed_at_stage_end"]
        ):
            raise StudyValidationError(
                f"{code} final snapshot provenance differs"
            )
        stages[code] = {
            "arm_manifest_path": str(arm_manifest_path),
            "arm_manifest_sha256": _hash_file(arm_manifest_path),
            "run_id": arm_manifest.get("run_id"),
            "git_commit": arm_manifest.get("git_commit"),
            "arm_identity_sha256": arm_manifest.get("identity_sha256"),
            "stages": stage_records,
            "outcome_path": str(outcome_path),
            "outcome_sha256": _hash_file(outcome_path),
            "outcome_quality_evidence": outcome.get("quality_evidence"),
            "outcome_latency_evidence": outcome.get("latency_evidence"),
            "final_snapshot": {
                "path": str(final_snapshot_path),
                "sha256": final_snapshot["sha256"],
            },
        }
    return study, arm_roots, stages


def _quality_expected_input_records(
    *,
    study: Mapping[str, Any],
    stages: Mapping[str, Any],
    arm: str,
    update: int,
) -> dict[str, Any]:
    stage = stages[arm]["stages"][str(update)]
    test = study.get("inputs", {}).get("splits", {}).get("test", {})
    clip_manifest = test.get("clip_manifest")
    cache = test.get("cache", {})
    cache_metadata = cache.get("metadata")
    if not isinstance(clip_manifest, dict) or not isinstance(
        cache_metadata, dict
    ):
        raise StudyValidationError(
            "study manifest lacks fixed-test manifest/cache provenance"
        )
    return {
        "resolved_config": dict(stage["resolved_config"]),
        "snapshot": dict(stage["snapshot_observed_at_stage_end"]),
        "arm_manifest": {
            "path": stages[arm]["arm_manifest_path"],
            "sha256": stages[arm]["arm_manifest_sha256"],
        },
        "study_manifest": {
            "path": str(Path(study["study_root"]) / "study_manifest.json"),
            "sha256": _hash_file(
                Path(study["study_root"]) / "study_manifest.json"
            ),
        },
        "stage_manifest": {
            "path": stage["stage_manifest_path"],
            "sha256": stage["stage_manifest_sha256"],
        },
        "stage_outcome": {
            "path": stage["path"],
            "sha256": stage["sha256"],
            "identity_sha256": stage["stage_outcome_identity_sha256"],
        },
        "test_clip_manifest": {
            "path": str(clip_manifest.get("path", "")),
            "sha256": str(clip_manifest.get("sha256", "")),
        },
        "test_cache_metadata": {
            "path": str(cache_metadata.get("path", "")),
            "sha256": str(cache_metadata.get("sha256", "")),
        },
        "test_cache_arrays": {
            name: {
                "path": str(cache.get(name, {}).get("path", "")),
                "sha256": str(cache.get(name, {}).get("sha256", "")),
                "bytes": cache.get(name, {}).get("bytes"),
            }
            for name in ("target", "rgb", "actions")
        },
    }


def _quality_metric_scope_valid(scope: Any) -> bool:
    return (
        isinstance(scope, dict)
        and scope.get("reconstruction_metrics_only") is True
        and scope.get("primary_decoded_target")
        == "cached raw held-out future RGB"
        and scope.get("vae_reconstruction_target_is_diagnostic_only") is True
        and scope.get(
            "temporal_metric_includes_history_to_first_future_boundary"
        )
        is True
        and (
            scope.get("perceptual_metric_available") is False
            or scope.get("lpips_or_video_perceptual_metric") is False
        )
    )


def _finite_metric_mapping(
    value: Any,
    *,
    expected_keys: set[str],
    label: str,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise StudyValidationError(f"{label} metric schema differs")
    normalized: dict[str, float] = {}
    for key in sorted(expected_keys):
        item = value[key]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise StudyValidationError(f"{label} {key} is non-finite")
        normalized[key] = float(item)
    if normalized.get("video_future_nmse", 0.0) < 0:
        raise StudyValidationError(f"{label} video NMSE is negative")
    if normalized.get("auxiliary_future_nmse", 0.0) < 0:
        raise StudyValidationError(f"{label} auxiliary NMSE is negative")
    cosine = normalized.get("auxiliary_future_cosine_similarity")
    if cosine is not None and not -1.0 <= cosine <= 1.0:
        raise StudyValidationError(f"{label} auxiliary cosine is invalid")
    for key in expected_keys:
        if ("mse" in key or "nmse" in key) and normalized[key] < 0:
            raise StudyValidationError(f"{label} {key} is negative")
    return normalized


def _load_quality_evidence(
    *,
    study: Mapping[str, Any],
    study_root: Path,
    arm_roots: Mapping[str, Path],
    stages: Mapping[str, Any],
) -> tuple[
    dict[str, dict[int, dict[tuple[str, int], dict[str, Any]]]],
    dict[str, Any],
    list[tuple[str, int]],
]:
    protocol = study.get("inference", {}).get("quality_protocol")
    expected_intermediate_grid = [
        {"source": source, "nfe": nfe}
        for source, nfe in _quality_grid(COMPLETED_UPDATES[0])
    ]
    expected_final_grid = [
        {"source": source, "nfe": nfe} for source, nfe in _quality_grid(1000)
    ]
    if (
        not isinstance(protocol, dict)
        or protocol.get("fixed_test_clips") != EXPECTED_TEST_CLIPS
        or protocol.get("distributed_world_size")
        != EXPECTED_QUALITY_WORLD_SIZE
        or protocol.get("batch_size_per_rank")
        != EXPECTED_QUALITY_BATCH_PER_RANK
        or protocol.get("trainer_visualization_is_diagnostic_only") is not True
        or protocol.get("stateless_noise_key") != "clip_index"
        or protocol.get(
            "deployable_sources_use_history_only_public_entrypoint"
        )
        is not True
        or protocol.get("oracle_sources_are_leakage_only") is not True
        or protocol.get(
            "rank0_rehashes_full_test_target_rgb_action_arrays"
        )
        is not True
        or protocol.get(
            "temporal_metric_includes_history_to_first_future_boundary"
        )
        is not True
        or protocol.get("intermediate_grid") != expected_intermediate_grid
        or protocol.get("final_grid") != expected_final_grid
    ):
        raise StudyValidationError("study quality protocol differs")

    test_record = (
        study.get("inputs", {})
        .get("splits", {})
        .get("test", {})
        .get("clip_manifest")
    )
    if (
        not isinstance(test_record, dict)
        or test_record.get("entries") != EXPECTED_TEST_CLIPS
    ):
        raise StudyValidationError(
            "study does not pin exactly 128 fixed test clips"
        )
    test_path = _regular_file(
        Path(str(test_record.get("path", ""))), "fixed-test clip manifest"
    )
    if test_record.get("sha256") != _hash_file(test_path):
        raise StudyValidationError("fixed-test clip manifest digest differs")
    descriptors = _read_jsonl(test_path, "fixed-test clip manifest")
    descriptor_units: list[tuple[str, int]] = []
    for index, descriptor in enumerate(descriptors):
        clip_id = descriptor.get("clip_id")
        auxiliary_index = descriptor.get("auxiliary_index")
        if (
            not isinstance(clip_id, str)
            or not clip_id
            or isinstance(auxiliary_index, bool)
            or auxiliary_index != index
        ):
            raise StudyValidationError(
                f"fixed-test descriptor {index} has invalid identity"
            )
        descriptor_units.append((clip_id, index))
    if (
        len(descriptor_units) != EXPECTED_TEST_CLIPS
        or len(set(descriptor_units)) != EXPECTED_TEST_CLIPS
        or len({clip_id for clip_id, _ in descriptor_units})
        != EXPECTED_TEST_CLIPS
    ):
        raise StudyValidationError(
            "fixed-test clip IDs/indexes are not 128 unique dense units"
        )

    metric_keys = set(METRIC_DIRECTIONS)
    diagnostic_keys = {
        "prediction_vs_vae_reconstruction_mse_unit_range",
        "prediction_vs_vae_reconstruction_psnr_db",
        "prediction_vs_vae_reconstruction_temporal_mse_unit_range",
        "vae_reconstruction_vs_raw_mse_unit_range",
        "vae_reconstruction_vs_raw_psnr_db",
        "vae_reconstruction_vs_raw_temporal_mse_unit_range",
    }
    tensor_hash_keys = {
        "video_clean_sha256",
        "auxiliary_clean_sha256",
        "ground_truth_sha256",
        "vae_ground_truth_sha256",
        "raw_history_last_sha256",
        "vae_history_last_sha256",
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_initial_state_sha256",
        "auxiliary_initial_state_sha256",
        "auxiliary_initial_noise_sha256",
        "video_final_sha256",
        "auxiliary_final_sha256",
        "decoded_final_sha256",
    }
    pairing_hash_keys = (
        "video_clean_sha256",
        "auxiliary_clean_sha256",
        "ground_truth_sha256",
        "vae_ground_truth_sha256",
        "raw_history_last_sha256",
        "vae_history_last_sha256",
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_initial_state_sha256",
        "auxiliary_initial_state_sha256",
        "auxiliary_initial_noise_sha256",
    )
    output_hash_keys = (
        "video_final_sha256",
        "auxiliary_final_sha256",
        "decoded_final_sha256",
    )
    analyzed: dict[
        str, dict[int, dict[tuple[str, int], dict[str, Any]]]
    ] = {arm: {} for arm in QUANTITATIVE_ARMS}
    inventory_summary: dict[str, Any] = {
        "scientific_evidence": "fixed 128-clip distributed quality evaluator",
        "trainer_visualization_role": "diagnostic_only",
        "arms": {},
    }
    global_pairing: dict[tuple[str, int], dict[str, str]] = {}

    for arm in QUANTITATIVE_ARMS:
        inventory_summary["arms"][arm] = {}
        outcome_quality = stages[arm].get("outcome_quality_evidence")
        outcome_inventories = (
            outcome_quality.get("inventories")
            if isinstance(outcome_quality, dict)
            else None
        )
        if (
            not isinstance(outcome_quality, dict)
            or outcome_quality.get(
                "trainer_visualization_is_diagnostic_only"
            )
            is not True
            or outcome_quality.get("fixed_test_clip_count")
            != EXPECTED_TEST_CLIPS
            or outcome_quality.get("reconstruction_metrics_only") is not True
            or not isinstance(outcome_inventories, dict)
        ):
            raise StudyValidationError(
                f"{arm} outcome lacks scientific quality evidence"
            )
        for update in COMPLETED_UPDATES:
            grid = _quality_grid(update)
            grid_records = [
                {"source": source, "nfe": nfe} for source, nfe in grid
            ]
            quality_root = _canonical_directory(
                arm_roots[arm] / "quality" / f"update_{update:04d}",
                f"{arm} update {update} quality directory",
            )
            _assert_inside(
                quality_root,
                arm_roots[arm],
                f"{arm} update {update} quality directory",
            )
            expected_names = {"inventory.json"}
            for rank in range(EXPECTED_QUALITY_WORLD_SIZE):
                expected_names.update(
                    {
                        f"rank_{rank:03d}.jsonl",
                        f"rank_{rank:03d}_manifest.json",
                    }
                )
            actual_names = {
                path.name
                for path in quality_root.iterdir()
                if path.is_file() or path.is_symlink()
            }
            if actual_names != expected_names:
                raise StudyValidationError(
                    f"{arm} update {update} quality files differ: "
                    f"missing={sorted(expected_names - actual_names)}, "
                    f"extra={sorted(actual_names - expected_names)}"
                )
            inventory_path = _regular_file(
                quality_root / "inventory.json",
                f"{arm} update {update} quality inventory",
            )
            inventory = _read_json(
                inventory_path, f"{arm} update {update} quality inventory"
            )
            stage = stages[arm]["stages"][str(update)]
            expected_records = EXPECTED_TEST_CLIPS * len(grid)
            expected_validation = {
                "record_count": expected_records,
                "clip_count": EXPECTED_TEST_CLIPS,
                "grid_count": len(grid),
                "all_clip_source_nfe_keys_present_once": True,
                "per_clip_clean_and_initial_noise_pairing_exact": True,
                "causal_no_op_identities_passed": True,
            }
            if (
                not _identity_valid(inventory)
                or inventory.get("schema_version") != SCHEMA_VERSION
                or inventory.get("kind")
                != "vjepa2_controlled_study_quality_inventory"
                or inventory.get("complete") is not True
                or inventory.get("arm_code") != arm
                or inventory.get("arm_identity_sha256")
                != stages[arm]["arm_identity_sha256"]
                or inventory.get("study_identity_sha256")
                != study.get("identity_sha256")
                or inventory.get("stage_identity_sha256")
                != stage["stage_identity_sha256"]
                or inventory.get("stage_outcome_identity_sha256")
                != stage["stage_outcome_identity_sha256"]
                or inventory.get("git_commit") != stages[arm]["git_commit"]
                or inventory.get("completed_updates") != update
                or inventory.get("world_size") != EXPECTED_QUALITY_WORLD_SIZE
                or inventory.get("batch_size_per_rank")
                != EXPECTED_QUALITY_BATCH_PER_RANK
                or inventory.get("clip_count") != EXPECTED_TEST_CLIPS
                or inventory.get("grid") != grid_records
                or inventory.get("expected_record_count") != expected_records
                or inventory.get("observed_record_count") != expected_records
                or inventory.get("validation") != expected_validation
                or inventory.get("oracle_sources_are_leakage_only") is not True
                or inventory.get(
                    "trainer_visualization_is_diagnostic_only"
                )
                is not True
                or not _quality_metric_scope_valid(
                    inventory.get("metric_scope")
                )
            ):
                raise StudyValidationError(
                    f"{arm} update {update} quality inventory differs"
                )
            input_integrity = inventory.get("input_cache_integrity")
            expected_cache_arrays = _quality_expected_input_records(
                study=study, stages=stages, arm=arm, update=update
            )["test_cache_arrays"]
            if (
                not isinstance(input_integrity, dict)
                or input_integrity.get(
                    "rank0_rehashed_full_target_rgb_action_arrays"
                )
                is not True
                or input_integrity.get("arrays") != expected_cache_arrays
            ):
                raise StudyValidationError(
                    f"{arm} update {update} cache-integrity evidence differs"
                )
            outcome_record = outcome_inventories.get(str(update))
            if (
                not isinstance(outcome_record, dict)
                or outcome_record.get("path") != str(inventory_path)
                or outcome_record.get("sha256") != _hash_file(inventory_path)
                or outcome_record.get("identity_sha256")
                != inventory["identity_sha256"]
                or outcome_record.get("record_count") != expected_records
            ):
                raise StudyValidationError(
                    f"{arm} update {update} outcome/inventory binding differs"
                )
            rank_evidence = inventory.get("rank_evidence")
            if (
                not isinstance(rank_evidence, dict)
                or set(rank_evidence)
                != {
                    str(rank) for rank in range(EXPECTED_QUALITY_WORLD_SIZE)
                }
            ):
                raise StudyValidationError(
                    f"{arm} update {update} rank inventory differs"
                )

            analyzed[arm][update] = {
                unit: {
                    "metrics": {},
                    "entries": {},
                    "identity": {
                        "clip_id": unit[0],
                        "clip_index": unit[1],
                    },
                }
                for unit in descriptor_units
            }
            observed_keys: set[tuple[int, str, int]] = set()
            per_update_pairing: dict[int, dict[str, str]] = {}
            rows_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
            expected_inputs = _quality_expected_input_records(
                study=study, stages=stages, arm=arm, update=update
            )
            for rank in range(EXPECTED_QUALITY_WORLD_SIZE):
                manifest_path = _regular_file(
                    quality_root / f"rank_{rank:03d}_manifest.json",
                    f"{arm} update {update} rank {rank} manifest",
                )
                rows_path = _regular_file(
                    quality_root / f"rank_{rank:03d}.jsonl",
                    f"{arm} update {update} rank {rank} rows",
                )
                evidence = rank_evidence[str(rank)]
                if (
                    not isinstance(evidence, dict)
                    or evidence.get("manifest_path") != str(manifest_path)
                    or evidence.get("manifest_sha256")
                    != _hash_file(manifest_path)
                    or evidence.get("rows_path") != str(rows_path)
                    or evidence.get("rows_sha256") != _hash_file(rows_path)
                ):
                    raise StudyValidationError(
                        f"{arm} update {update} rank {rank} evidence differs"
                    )
                manifest = _read_json(
                    manifest_path,
                    f"{arm} update {update} rank {rank} manifest",
                )
                assigned = list(
                    range(rank, EXPECTED_TEST_CLIPS, EXPECTED_QUALITY_WORLD_SIZE)
                )
                expected_rank_rows = len(assigned) * len(grid)
                expected_wan_calls = (
                    len(assigned)
                    // EXPECTED_QUALITY_BATCH_PER_RANK
                    * sum(nfe for _source, nfe in grid)
                )
                rows_record = manifest.get("rows")
                if (
                    not _identity_valid(manifest)
                    or manifest.get("schema_version") != SCHEMA_VERSION
                    or manifest.get("kind")
                    != "vjepa2_controlled_study_quality_rank"
                    or manifest.get("arm_code") != arm
                    or manifest.get("arm_identity_sha256")
                    != stages[arm]["arm_identity_sha256"]
                    or manifest.get("study_identity_sha256")
                    != study.get("identity_sha256")
                    or manifest.get("stage_identity_sha256")
                    != stage["stage_identity_sha256"]
                    or manifest.get("stage_outcome_identity_sha256")
                    != stage["stage_outcome_identity_sha256"]
                    or manifest.get("git_commit") != stages[arm]["git_commit"]
                    or manifest.get("completed_updates") != update
                    or manifest.get("rank") != rank
                    or manifest.get("world_size")
                    != EXPECTED_QUALITY_WORLD_SIZE
                    or manifest.get("batch_size_per_rank")
                    != EXPECTED_QUALITY_BATCH_PER_RANK
                    or manifest.get("assigned_clip_indexes") != assigned
                    or manifest.get("grid") != grid_records
                    or manifest.get("actual_wan_backbone_invocations")
                    != expected_wan_calls
                    or manifest.get("online_teacher_call_count") != 0
                    or manifest.get("inputs") != expected_inputs
                    or not _quality_metric_scope_valid(
                        manifest.get("metric_scope")
                    )
                    or manifest.get("oracle_sources_are_leakage_only")
                    is not True
                    or manifest.get("sigma_convention")
                    != "sigma=1 noise, sigma=0 clean"
                    or not isinstance(rows_record, dict)
                    or rows_record.get("path") != str(rows_path)
                    or rows_record.get("sha256") != _hash_file(rows_path)
                    or rows_record.get("bytes") != rows_path.stat().st_size
                    or rows_record.get("count") != expected_rank_rows
                    or evidence.get("manifest_identity_sha256")
                    != manifest["identity_sha256"]
                    or evidence.get("row_count") != expected_rank_rows
                ):
                    raise StudyValidationError(
                        f"{arm} update {update} rank {rank} manifest differs"
                    )
                device = manifest.get("device")
                if (
                    not isinstance(device, dict)
                    or "B200" not in str(device.get("name", "")).upper()
                    or device.get("local_rank") != rank
                ):
                    raise StudyValidationError(
                        f"{arm} update {update} rank {rank} device differs"
                    )
                rows = _read_jsonl(
                    rows_path, f"{arm} update {update} rank {rank} rows"
                )
                if len(rows) != expected_rank_rows:
                    raise StudyValidationError(
                        f"{arm} update {update} rank {rank} row count differs"
                    )
                expected_rank_keys = {
                    (clip_index, source, nfe)
                    for clip_index in assigned
                    for source, nfe in grid
                }
                rank_keys: set[tuple[int, str, int]] = set()
                for row in rows:
                    if not _identity_valid(row):
                        raise StudyValidationError(
                            f"{arm} update {update} quality row identity differs"
                        )
                    clip_index = row.get("clip_index")
                    source = row.get("source")
                    nfe = row.get("nfe")
                    if (
                        isinstance(clip_index, bool)
                        or not isinstance(clip_index, int)
                        or not isinstance(source, str)
                        or isinstance(nfe, bool)
                        or not isinstance(nfe, int)
                    ):
                        raise StudyValidationError(
                            f"{arm} update {update} row key is invalid"
                        )
                    key = (clip_index, source, nfe)
                    if key not in expected_rank_keys or key in rank_keys:
                        raise StudyValidationError(
                            f"{arm} update {update} rank {rank} row key differs: "
                            f"{key}"
                        )
                    rank_keys.add(key)
                    unit = descriptor_units[clip_index]
                    oracle = source in ORACLE_SOURCES
                    expected_entrypoint = (
                        "DualExplicitActionDiTModel._sample_future"
                        if oracle
                        else (
                            "DualExplicitActionDiTModel."
                            "sample_future_deployable"
                        )
                    )
                    state_gate = row.get("effective_state_gate")
                    clock_gate = row.get("effective_clock_gate")
                    if (
                        row.get("schema_version") != SCHEMA_VERSION
                        or row.get("kind")
                        != "vjepa2_controlled_study_quality_clip"
                        or row.get("arm_code") != arm
                        or row.get("completed_updates") != update
                        or row.get("clip_id") != unit[0]
                        or row.get("oracle_leakage") != oracle
                        or row.get("deployable_evidence") != (not oracle)
                        or row.get("sampler_entrypoint")
                        != expected_entrypoint
                        or row.get(
                            "clean_future_or_auxiliary_passed_to_sampler"
                        )
                        != oracle
                        or row.get("online_teacher_call_count") != 0
                        or row.get("actual_wan_call_count") != nfe
                        or row.get("auxiliary_history_latent_frames") != 0
                        or isinstance(
                            row.get("video_history_latent_frames"), bool
                        )
                        or not isinstance(
                            row.get("video_history_latent_frames"), int
                        )
                        or row["video_history_latent_frames"] < 0
                        or isinstance(state_gate, bool)
                        or not isinstance(state_gate, (int, float))
                        or not math.isfinite(float(state_gate))
                        or isinstance(clock_gate, bool)
                        or not isinstance(clock_gate, (int, float))
                        or not math.isfinite(float(clock_gate))
                    ):
                        raise StudyValidationError(
                            f"{arm} update {update} row provenance differs: "
                            f"{key}"
                        )
                    if arm in {"VPM", "A1"} and (
                        float(state_gate) != 0.0
                        or float(clock_gate) != 0.0
                    ):
                        raise StudyValidationError(
                            f"{arm} update {update} no-op gate is nonzero"
                        )
                    metrics = _finite_metric_mapping(
                        row.get("metrics"),
                        expected_keys=metric_keys,
                        label=f"{arm} update {update} row {key}",
                    )
                    diagnostics = _finite_metric_mapping(
                        row.get("diagnostic_metrics"),
                        expected_keys=diagnostic_keys,
                        label=(
                            f"{arm} update {update} diagnostic row {key}"
                        ),
                    )
                    hashes = row.get("tensor_sha256")
                    if (
                        not isinstance(hashes, dict)
                        or set(hashes) != tensor_hash_keys
                        or any(
                            not isinstance(value, str)
                            or SHA256_RE.fullmatch(value) is None
                            for value in hashes.values()
                        )
                        or hashes["auxiliary_initial_state_sha256"]
                        != hashes["auxiliary_initial_noise_sha256"]
                    ):
                        raise StudyValidationError(
                            f"{arm} update {update} row hashes differ: {key}"
                        )
                    perceptual = row.get("perceptual_metric")
                    if (
                        not isinstance(perceptual, dict)
                        or perceptual.get("available") is not False
                    ):
                        raise StudyValidationError(
                            f"{arm} update {update} row enabled an unpinned "
                            "perceptual metric"
                        )
                    paired_hashes = {
                        field: hashes[field] for field in pairing_hash_keys
                    }
                    prior = per_update_pairing.setdefault(
                        clip_index, paired_hashes
                    )
                    if prior != paired_hashes:
                        raise StudyValidationError(
                            f"{arm} update {update} clip {clip_index} "
                            "clean/noise pairing differs across grid"
                        )
                    global_prior = global_pairing.setdefault(
                        unit, paired_hashes
                    )
                    if global_prior != paired_hashes:
                        raise StudyValidationError(
                            "fixed-test clean/noise pairing differs across "
                            f"arms or milestones for unit {unit}"
                        )
                    analyzed_unit = analyzed[arm][update][unit]
                    analyzed_unit["metrics"].setdefault(source, {})[
                        str(nfe)
                    ] = metrics
                    analyzed_unit["entries"].setdefault(source, {})[
                        str(nfe)
                    ] = {
                        "diagnostic_metrics": diagnostics,
                        "tensor_sha256": dict(hashes),
                        "sampler_entrypoint": expected_entrypoint,
                        "oracle_leakage": oracle,
                        "deployable_evidence": not oracle,
                        "effective_state_gate": float(state_gate),
                        "effective_clock_gate": float(clock_gate),
                        "quality_rows_path": str(rows_path),
                        "quality_row_identity_sha256": row[
                            "identity_sha256"
                        ],
                    }
                    analyzed_unit["identity"].update(
                        {
                            "video_history_latent_frames": row[
                                "video_history_latent_frames"
                            ],
                            **paired_hashes,
                        }
                    )
                    rows_by_key[key] = row
                if rank_keys != expected_rank_keys:
                    raise StudyValidationError(
                        f"{arm} update {update} rank {rank} rows are incomplete"
                    )
                observed_keys.update(rank_keys)

            expected_all_keys = {
                (clip_index, source, nfe)
                for clip_index in range(EXPECTED_TEST_CLIPS)
                for source, nfe in grid
            }
            if observed_keys != expected_all_keys:
                raise StudyValidationError(
                    f"{arm} update {update} global quality keys differ"
                )
            sources_by_nfe: dict[int, list[str]] = {}
            for source, nfe in grid:
                sources_by_nfe.setdefault(nfe, []).append(source)
            for clip_index in range(EXPECTED_TEST_CLIPS):
                for nfe, available_sources in sources_by_nfe.items():
                    if arm in {"VPM", "A1"}:
                        compare_sources = available_sources
                    elif nfe == 1:
                        compare_sources = [
                            source
                            for source in available_sources
                            if source != "off"
                        ]
                    else:
                        continue
                    if len(compare_sources) < 2:
                        continue
                    reference_hashes = rows_by_key[
                        (clip_index, compare_sources[0], nfe)
                    ]["tensor_sha256"]
                    for source in compare_sources[1:]:
                        candidate = rows_by_key[
                            (clip_index, source, nfe)
                        ]["tensor_sha256"]
                        if any(
                            candidate[field] != reference_hashes[field]
                            for field in output_hash_keys
                        ):
                            raise StudyValidationError(
                                f"{arm} update {update} exact causal no-op "
                                f"failed for clip {clip_index}, NFE {nfe}, "
                                f"source {source}"
                            )
            inventory_summary["arms"][arm][str(update)] = {
                "inventory_path": str(inventory_path),
                "inventory_sha256": _hash_file(inventory_path),
                "inventory_identity_sha256": inventory["identity_sha256"],
                "clip_count": EXPECTED_TEST_CLIPS,
                "grid": grid_records,
                "record_count": expected_records,
                "rank_count": EXPECTED_QUALITY_WORLD_SIZE,
                "snapshot_observed_at_stage_end": dict(
                    stage["snapshot_observed_at_stage_end"]
                ),
                "primary_decoded_target": (
                    "cached raw held-out future RGB"
                ),
                "temporal_metric_includes_history_to_first_future_boundary": (
                    True
                ),
                "input_cache_integrity": dict(input_integrity),
            }

    return analyzed, inventory_summary, descriptor_units


def _load_latency(
    arm_roots: Mapping[str, Path],
    stages: Mapping[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    expected = {
        (arm, source, nfe)
        for arm in QUANTITATIVE_ARMS
        for source in LATENCY_SOURCES
        for nfe in NFE_STEPS
    }
    observed: set[tuple[str, str, int]] = set()
    reference_sample_identity: dict[str, Any] | None = None
    latency_study_path = _regular_file(
        arm_roots["VPM"].parent / "study_manifest.json",
        "latency study manifest",
    )
    latency_study = _read_json(latency_study_path, "latency study manifest")
    for arm in QUANTITATIVE_ARMS:
        root = arm_roots[arm]
        latency_root = root / "latency"
        arm_results: dict[str, Any] = {}
        if not latency_root.exists():
            results[arm] = arm_results
            continue
        latency_root = _canonical_directory(latency_root, f"{arm} latency root")
        for path in sorted(latency_root.glob("source_*_nfe_*.json")):
            path = _regular_file(path, f"{arm} latency result")
            match = LATENCY_RE.fullmatch(path.name)
            if match is None:
                raise StudyValidationError(f"invalid latency filename: {path}")
            source = match.group(1)
            nfe = int(match.group(2))
            if nfe not in NFE_STEPS:
                raise StudyValidationError(f"latency NFE is not pinned: {path}")
            key = (arm, source, nfe)
            if key in observed:
                raise StudyValidationError(f"duplicate latency result: {key}")
            payload = _read_json(path, f"{arm} latency result")
            if (
                not _identity_valid(payload)
                or payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("kind") != "vjepa2_controlled_study_latency"
                or payload.get("git_commit") != stages[arm]["git_commit"]
                or payload.get("run_dir") != str(root)
                or payload.get("arm_code") != arm
                or payload.get("arm_identity_sha256")
                != stages[arm]["arm_identity_sha256"]
                or payload.get("source") != source
                or payload.get("source_is_deployable") is not True
                or payload.get("oracle_is_leakage_only") is not False
                or payload.get("nfe") != nfe
                or payload.get("batch_size") != 1
                or payload.get("warmups") != 20
                or payload.get("repetitions") != 100
                or payload.get("sigma_convention")
                != "sigma=1 noise, sigma=0 clean"
            ):
                raise StudyValidationError(
                    f"latency identity/protocol differs: {path}"
                )
            device = payload.get("device")
            if (
                not isinstance(device, dict)
                or "B200" not in str(device.get("name", "")).upper()
                or device.get("index") != 0
            ):
                raise StudyValidationError(
                    f"latency result was not measured on cuda:0 B200: {path}"
                )
            inputs = payload.get("inputs")
            if not isinstance(inputs, dict):
                raise StudyValidationError(f"latency inputs are missing: {path}")
            expected_input_records = {
                "resolved_config": stages[arm]["stages"]["1000"][
                    "resolved_config"
                ],
                "snapshot": stages[arm]["stages"]["1000"][
                    "snapshot_observed_at_stage_end"
                ],
                "arm_manifest": {
                    "path": stages[arm]["arm_manifest_path"],
                    "sha256": stages[arm]["arm_manifest_sha256"],
                },
                "study_manifest": {
                    "path": str(latency_study_path),
                    "sha256": _hash_file(latency_study_path),
                    "identity_sha256": latency_study["identity_sha256"],
                },
                "stage_manifest": {
                    "path": stages[arm]["stages"]["1000"][
                        "stage_manifest_path"
                    ],
                    "sha256": stages[arm]["stages"]["1000"][
                        "stage_manifest_sha256"
                    ],
                    "identity_sha256": stages[arm]["stages"]["1000"][
                        "stage_identity_sha256"
                    ],
                },
                "stage_outcome": {
                    "path": stages[arm]["stages"]["1000"]["path"],
                    "sha256": stages[arm]["stages"]["1000"]["sha256"],
                    "identity_sha256": stages[arm]["stages"]["1000"][
                        "stage_outcome_identity_sha256"
                    ],
                },
            }
            for field, wanted in expected_input_records.items():
                if inputs.get(field) != wanted:
                    raise StudyValidationError(
                        f"latency {field} provenance differs: {path}"
                    )
            sample_index = inputs.get("sample_index")
            if (
                isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index != 0
            ):
                raise StudyValidationError(
                    f"latency sample index is invalid: {path}"
                )
            sample_identity = inputs.get("sample_identity")
            if (
                not isinstance(sample_identity, dict)
                or sample_identity.get("sample_index") != sample_index
                or set(sample_identity)
                != {
                    "sample_index",
                    "full_rgb_sha256",
                    "history_rgb_sha256",
                    "actions_sha256",
                    "morphology_index_sha256",
                }
                or any(
                    not isinstance(sample_identity[field], str)
                    or SHA256_RE.fullmatch(sample_identity[field]) is None
                    for field in (
                        "full_rgb_sha256",
                        "history_rgb_sha256",
                        "actions_sha256",
                        "morphology_index_sha256",
                    )
                )
            ):
                raise StudyValidationError(
                    f"latency sample identity is invalid: {path}"
                )
            if reference_sample_identity is None:
                reference_sample_identity = dict(sample_identity)
            elif sample_identity != reference_sample_identity:
                raise StudyValidationError(
                    f"latency input sample differs across arms/sources/NFEs: {path}"
                )
            expected_scope = {
                "entrypoint": (
                    "DualExplicitActionDiTModel.sample_future_deployable"
                ),
                "dataset_loading_timed": False,
                "future_ground_truth_rgb_available": False,
                "clean_auxiliary_target_available": False,
                "history_video_latent_and_action_preparation_timed": True,
                "wan_backbone_calls_timed": True,
                "vae_decode_timed": True,
                "internal_evidence_tensor_materialization_timed": False,
                "trajectory_capture_timed": False,
                "one_source_and_nfe_per_process": True,
                "source_mapped_to_single_autonomous_sampler_slot": True,
                "full_clip_prediction_equivalence_audit_timed": False,
                "cuda_synchronize_before_and_after": True,
            }
            scope = payload.get("scope")
            if not isinstance(scope, dict) or any(
                scope.get(field) != wanted
                for field, wanted in expected_scope.items()
            ):
                raise StudyValidationError(f"latency timing scope differs: {path}")
            history_frames = scope.get("observed_history_frames")
            generated_frames = scope.get("generated_future_frames")
            if (
                isinstance(history_frames, bool)
                or not isinstance(history_frames, int)
                or history_frames != OBSERVED_HISTORY_FRAMES
                or isinstance(generated_frames, bool)
                or not isinstance(generated_frames, int)
                or generated_frames != GENERATED_FUTURE_FRAMES
            ):
                raise StudyValidationError(
                    f"latency history/future frame counts are invalid: {path}"
                )
            invariance = payload.get("history_invariance_audit")
            expected_invariance = {
                "passed": True,
                "comparison": "full_clip_future_vs_history_only_deployable",
                "full_clip_call_timed": False,
                "full_clip_ground_truth_constructed_for_audit_only": True,
                "deployable_future_ground_truth_available": False,
                "history_frames": history_frames,
                "generated_future_frames": generated_frames,
                "full_wan_calls": nfe,
                "deployable_wan_calls": nfe,
                "absolute_tolerance": 1.0e-6,
            }
            if not isinstance(invariance, dict) or any(
                invariance.get(field) != wanted
                for field, wanted in expected_invariance.items()
            ):
                raise StudyValidationError(
                    f"latency history-only invariance audit differs: {path}"
                )
            for field in ("max_abs", "mean_abs"):
                value = invariance.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    or float(value) > 1.0e-6
                ):
                    raise StudyValidationError(
                        f"latency invariance {field} is invalid: {path}"
                    )
            if not isinstance(invariance.get("exact_equal"), bool):
                raise StudyValidationError(
                    f"latency invariance exact-equality flag is invalid: {path}"
                )
            counters = payload.get("counters")
            expected_audit = {
                "wan_calls": nfe,
                "online_teacher_calls": 0,
                "auxiliary_clean_available": 0,
                "deployment_mode": 1,
            }
            if (
                not isinstance(counters, dict)
                or counters.get("separate_untimed_audit_run") != expected_audit
                or counters.get("teacher_calls_per_repetition") != 0
                or counters.get("teacher_calls_total") != 0
                or counters.get("wan_calls_per_repetition") != nfe
                or counters.get("wan_calls_total") != nfe * 100
                or counters.get("auxiliary_clean_available") != 0
            ):
                raise StudyValidationError(
                    f"latency teacher/Wan/clean-target counters differ: {path}"
                )
            latency = payload.get("latency_ms")
            if not isinstance(latency, dict):
                raise StudyValidationError(f"latency_ms is missing: {path}")
            values: dict[str, float] = {}
            for field in ("p50", "p95", "mean", "min", "max"):
                value = latency.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise StudyValidationError(
                        f"latency {field} is invalid: {path}"
                    )
                values[field] = float(value)
            if not (
                values["min"]
                <= values["p50"]
                <= values["p95"]
                <= values["max"]
                and values["min"] <= values["mean"] <= values["max"]
            ):
                raise StudyValidationError(
                    f"latency order statistics are inconsistent: {path}"
                )
            values_sha = latency.get("values_sha256")
            if not isinstance(values_sha, str) or SHA256_RE.fullmatch(values_sha) is None:
                raise StudyValidationError(
                    f"latency raw-values SHA-256 is invalid: {path}"
                )
            raw_values = latency.get("values")
            if (
                not isinstance(raw_values, list)
                or len(raw_values) != 100
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in raw_values
                )
            ):
                raise StudyValidationError(
                    f"latency raw-value vector is invalid: {path}"
                )
            normalized_raw = [float(value) for value in raw_values]
            expected_values_sha = hashlib.sha256(
                _canonical_json_bytes(
                    [round(value, 9) for value in normalized_raw]
                )
            ).hexdigest()
            if values_sha != expected_values_sha:
                raise StudyValidationError(
                    f"latency raw-value SHA-256 differs: {path}"
                )
            ordered = sorted(normalized_raw)

            def percentile(percent: float) -> float:
                position = (len(ordered) - 1) * percent / 100.0
                lower = int(position)
                upper = min(lower + 1, len(ordered) - 1)
                fraction = position - lower
                return (
                    ordered[lower] * (1.0 - fraction)
                    + ordered[upper] * fraction
                )

            recomputed = {
                "p50": percentile(50.0),
                "p95": percentile(95.0),
                "mean": sum(normalized_raw) / len(normalized_raw),
                "min": min(normalized_raw),
                "max": max(normalized_raw),
            }
            if any(
                not math.isclose(
                    values[field],
                    recomputed[field],
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                for field in recomputed
            ):
                raise StudyValidationError(
                    f"latency summary does not match raw values: {path}"
                )
            throughput = payload.get("throughput")
            expected_throughput = {
                "generated_frames_per_clip": generated_frames,
                "clips_per_second_at_p50_latency": 1000.0 / values["p50"],
                "clips_per_second_at_p95_latency": 1000.0 / values["p95"],
                "generated_frames_per_second_at_p50_latency": (
                    generated_frames * 1000.0 / values["p50"]
                ),
                "generated_frames_per_second_at_p95_latency": (
                    generated_frames * 1000.0 / values["p95"]
                ),
            }
            if not isinstance(throughput, dict):
                raise StudyValidationError(
                    f"latency throughput is missing: {path}"
                )
            for field, wanted in expected_throughput.items():
                actual = throughput.get(field)
                if field == "generated_frames_per_clip":
                    matches = actual == wanted
                else:
                    matches = (
                        isinstance(actual, (int, float))
                        and not isinstance(actual, bool)
                        and math.isclose(
                            float(actual),
                            float(wanted),
                            rel_tol=1e-12,
                            abs_tol=1e-9,
                        )
                    )
                if not matches:
                    raise StudyValidationError(
                        f"latency throughput {field} differs: {path}"
                    )
            observed.add(key)
            arm_results.setdefault(source, {})[str(nfe)] = {
                "path": str(path),
                "sha256": _hash_file(path),
                "identity_sha256": payload["identity_sha256"],
                "oracle_leakage": source in ORACLE_SOURCES,
                "deployable_evidence": source not in ORACLE_SOURCES,
                "teacher_calls": 0,
                "wan_calls_per_repetition": nfe,
                "latency_ms": {
                    **values,
                    "values_sha256": values_sha,
                    "raw_values_omitted_from_analysis": True,
                },
                "throughput": dict(throughput),
                "history_invariance_audit": dict(invariance),
                "sample_identity": dict(sample_identity),
            }
        outcome_latency = stages[arm].get("outcome_latency_evidence")
        outcome_records = (
            outcome_latency.get("records")
            if isinstance(outcome_latency, dict)
            else None
        )
        expected_arm_keys = {
            f"{source}:nfe_{nfe}"
            for source in LATENCY_SOURCES
            for nfe in NFE_STEPS
        }
        if (
            not isinstance(outcome_latency, dict)
            or outcome_latency.get("complete") is not True
            or outcome_latency.get("record_count") != len(expected_arm_keys)
            or outcome_latency.get("expected_record_count")
            != len(expected_arm_keys)
            or outcome_latency.get("fixed_sample_identity")
            != reference_sample_identity
            or not isinstance(outcome_records, dict)
            or set(outcome_records) != expected_arm_keys
        ):
            raise StudyValidationError(
                f"{arm} outcome latency evidence is incomplete"
            )
        for source in LATENCY_SOURCES:
            for nfe in NFE_STEPS:
                result = arm_results.get(source, {}).get(str(nfe))
                record = outcome_records.get(f"{source}:nfe_{nfe}")
                if (
                    not isinstance(result, dict)
                    or not isinstance(record, dict)
                    or record.get("path") != result["path"]
                    or record.get("sha256") != result["sha256"]
                    or record.get("identity_sha256")
                    != result["identity_sha256"]
                    or record.get("sample_identity")
                    != result["sample_identity"]
                ):
                    raise StudyValidationError(
                        f"{arm} outcome latency binding differs for "
                        f"{source} NFE={nfe}"
                    )
        results[arm] = arm_results
    missing = sorted(expected - observed)
    return {
        "protocol": {
            "batch_size": 1,
            "warmups": 20,
            "timed_repetitions": 100,
            "scope": "model preparation + Wan calls + VAE decode",
            "fixed_sample_identity": reference_sample_identity,
            "V0_role": (
                "no compatible deployable latency evaluator; speed claims "
                "compare quantitative dual arms against VPM only"
            ),
        },
        "complete": not missing,
        "expected_record_count": len(expected),
        "observed_record_count": len(observed),
        "missing_records": [
            {"arm": arm, "source": source, "nfe": nfe}
            for arm, source, nfe in missing
        ],
        "arms": results,
    }


def _normalized_slurm_job_id(value: Any, label: str) -> str:
    raw = str(value)
    normalized = raw.split(";", 1)[0].split("_", 1)[0]
    if not normalized.isdigit() or int(normalized) <= 0:
        raise StudyValidationError(f"{label} is not a positive Slurm job ID")
    return normalized


def _validate_paired_file_record(
    record: Any,
    *,
    path: Path,
    label: str,
    identity_sha256: str | None = None,
) -> None:
    if (
        not isinstance(record, dict)
        or record.get("path") != str(path)
        or record.get("sha256") != _hash_file(path)
        or record.get("bytes") != path.stat().st_size
        or (
            identity_sha256 is not None
            and record.get("identity_sha256") != identity_sha256
        )
    ):
        raise StudyValidationError(f"paired latency {label} record differs")


def _validated_paired_latency_vector(
    value: Any,
    *,
    label: str,
) -> tuple[list[float], dict[str, Any]]:
    if not isinstance(value, dict) or value.get("count") != PAIRED_TIMED_PAIRS:
        raise StudyValidationError(f"{label} latency summary is missing")
    raw = value.get("values")
    if (
        not isinstance(raw, list)
        or len(raw) != PAIRED_TIMED_PAIRS
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
            for item in raw
        )
    ):
        raise StudyValidationError(f"{label} latency vector is invalid")
    values = [float(item) for item in raw]
    expected_sha = hashlib.sha256(
        _canonical_json_bytes([round(item, 9) for item in values])
    ).hexdigest()
    if value.get("values_sha256") != expected_sha:
        raise StudyValidationError(f"{label} latency vector digest differs")
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    recomputed = {
        "p50": percentile(50.0),
        "p95": percentile(95.0),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
    for field, expected in recomputed.items():
        observed = value.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed), expected, rel_tol=1e-12, abs_tol=1e-9
            )
        ):
            raise StudyValidationError(
                f"{label} latency {field} does not match raw values"
            )
    return values, {
        **recomputed,
        "count": PAIRED_TIMED_PAIRS,
        "values_sha256": expected_sha,
        "raw_values_omitted_from_analysis": True,
    }


def _load_paired_latency(
    *,
    study: Mapping[str, Any],
    study_root: Path,
    stages: Mapping[str, Any],
    grid_latency: Mapping[str, Any],
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    path = _regular_file(
        study_root / "paired_latency" / PAIRED_LATENCY_BASENAME,
        "paired latency evidence",
    )
    _assert_inside(path, study_root, "paired latency evidence")
    payload = _read_json(path, "paired latency evidence")
    expected_commit = study.get("inputs", {}).get("repository", {}).get(
        "git_commit"
    )
    if (
        not _identity_valid(payload)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != PAIRED_LATENCY_KIND
        or payload.get("comparison") != PAIRED_LATENCY_COMPARISON
        or payload.get("git_commit") != expected_commit
    ):
        raise StudyValidationError(
            "paired latency identity/comparison/commit differs"
        )

    study_path = _regular_file(
        study_root / "study_manifest.json", "paired latency study manifest"
    )
    _validate_paired_file_record(
        payload.get("study"),
        path=study_path,
        label="study manifest",
        identity_sha256=str(study.get("identity_sha256", "")),
    )
    submission_path = _regular_file(
        study_root / "slurm_submission.json",
        "paired latency submission record",
    )
    submission = _read_json(
        submission_path, "paired latency submission record"
    )
    if (
        not _identity_valid(submission)
        or submission.get("kind")
        != "vjepa2_controlled_study_submission"
        or submission.get("study_identity_sha256")
        != study.get("identity_sha256")
        or submission.get("dependency") != "afterok"
    ):
        raise StudyValidationError(
            "paired latency submission provenance differs"
        )
    _validate_paired_file_record(
        payload.get("submission"),
        path=submission_path,
        label="submission",
        identity_sha256=str(submission.get("identity_sha256", "")),
    )
    paired_job = submission.get("paired_latency_job")
    if (
        not isinstance(paired_job, dict)
        or paired_job.get("dependency") != "afterok:final_stage_array"
        or paired_job.get("comparison") != PAIRED_LATENCY_COMPARISON
        or paired_job.get("nodes") != 1
        or paired_job.get("gpus") != 1
        or paired_job.get("same_allocation_pairing") is not True
        or paired_job.get("runs_final_analyzer_after_benchmark") is not True
        or paired_job.get("analysis_output_root")
        != study["slurm"]["paired_latency_post_study_job"][
            "analysis_output_root"
        ]
    ):
        raise StudyValidationError(
            "paired latency submission job contract differs"
        )
    slurm = payload.get("slurm")
    if (
        not isinstance(slurm, dict)
        or slurm.get("same_allocation") is not True
        or not isinstance(slurm.get("same_node"), str)
        or not slurm["same_node"].strip()
        or not isinstance(slurm.get("cuda_visible_devices"), str)
        or not slurm["cuda_visible_devices"].strip()
        or _normalized_slurm_job_id(
            slurm.get("job_id"), "paired evidence Slurm job ID"
        )
        != _normalized_slurm_job_id(
            paired_job.get("job_id"), "paired submission Slurm job ID"
        )
    ):
        raise StudyValidationError(
            "paired latency was not bound to one recorded allocation/node/GPU"
        )
    device = payload.get("device")
    if (
        not isinstance(device, dict)
        or "B200" not in str(device.get("name", "")).upper()
        or device.get("index") != 0
        or isinstance(
            device.get("peak_allocated_bytes_with_both_models_resident"), bool
        )
        or not isinstance(
            device.get("peak_allocated_bytes_with_both_models_resident"), int
        )
        or device["peak_allocated_bytes_with_both_models_resident"] <= 0
    ):
        raise StudyValidationError(
            "paired latency device/resident-model provenance differs"
        )

    protocol = payload.get("protocol")
    expected_protocol = {
        "batch_size": 1,
        "sample_index": 0,
        "warmup_pairs": PAIRED_WARMUP_PAIRS,
        "timed_pairs": PAIRED_TIMED_PAIRS,
        "counterbalance": "even pair J1-first; odd pair VPM-first",
        "J1_first_pairs": 50,
        "VPM_first_pairs": 50,
        "same_process": True,
        "same_B200": True,
        "both_models_resident": True,
        "identical_immutable_batch_inputs": True,
        "entrypoint": (
            "DualExplicitActionDiTModel.sample_future_deployable"
        ),
        "collect_artifacts_timed": False,
        "trajectory_capture_timed": False,
        "forward_hooks_active_during_timing": False,
        "future_ground_truth_rgb_available_to_sampler": False,
        "clean_auxiliary_target_available_to_sampler": False,
        "online_teacher_calls": 0,
        "cuda_synchronize_before_and_after_each_arm": True,
        "timing_scope": (
            "history preparation inside model + Wan calls + VAE decode"
        ),
        "sigma_convention": "sigma=1 noise, sigma=0 clean",
    }
    if not isinstance(protocol, dict) or protocol != expected_protocol:
        raise StudyValidationError("paired latency timing protocol differs")

    immutable = payload.get("immutable_input")
    expected_identity = grid_latency.get("protocol", {}).get(
        "fixed_sample_identity"
    )
    sample_identity_fields = {
        "sample_index",
        "full_rgb_sha256",
        "history_rgb_sha256",
        "actions_sha256",
        "morphology_index_sha256",
    }
    if (
        not isinstance(immutable, dict)
        or set(immutable) != sample_identity_fields | {"test_cache"}
        or immutable.get("sample_index") != 0
        or any(
            not isinstance(immutable.get(field), str)
            or SHA256_RE.fullmatch(immutable[field]) is None
            for field in sample_identity_fields - {"sample_index"}
        )
        or {
            field: immutable[field] for field in sample_identity_fields
        }
        != expected_identity
    ):
        raise StudyValidationError(
            "paired latency immutable input differs from the per-arm grid"
        )

    test = study.get("inputs", {}).get("splits", {}).get("test", {})
    cache = test.get("cache", {})
    paired_cache = immutable.get("test_cache")
    if not isinstance(paired_cache, dict):
        raise StudyValidationError("paired latency test-cache record is absent")
    for field, paired_field, label in (
        ("clip_manifest", "clip_manifest", "test clip manifest"),
        ("metadata", "cache_metadata", "test cache metadata"),
    ):
        study_record = (
            test.get(field) if field == "clip_manifest" else cache.get(field)
        )
        paired_record = paired_cache.get(paired_field)
        if not isinstance(study_record, dict):
            raise StudyValidationError(f"study lacks {label} provenance")
        current = _regular_file(Path(str(study_record.get("path", ""))), label)
        if (
            study_record.get("sha256") != _hash_file(current)
            or not isinstance(paired_record, dict)
            or paired_record.get("path") != str(current)
            or paired_record.get("sha256") != study_record.get("sha256")
            or paired_record.get("bytes") != current.stat().st_size
        ):
            raise StudyValidationError(
                f"paired latency {label} provenance differs"
            )
    paired_arrays = paired_cache.get("arrays")
    if not isinstance(paired_arrays, dict) or set(paired_arrays) != {
        "target",
        "rgb",
        "actions",
    }:
        raise StudyValidationError(
            "paired latency test-cache array inventory differs"
        )
    for name in ("target", "rgb", "actions"):
        study_record = cache.get(name)
        paired_record = paired_arrays.get(name)
        if not isinstance(study_record, dict):
            raise StudyValidationError(
                f"study lacks test {name} array provenance"
            )
        current = _regular_file(
            Path(str(study_record.get("path", ""))),
            f"test {name} cache array",
        )
        actual_sha = _hash_file(current)
        if (
            study_record.get("sha256") != actual_sha
            or study_record.get("bytes") != current.stat().st_size
            or not isinstance(paired_record, dict)
            or paired_record
            != {
                "path": str(current),
                "sha256": actual_sha,
                "bytes": current.stat().st_size,
                "full_sha256_verified": True,
            }
        ):
            raise StudyValidationError(
                f"paired latency test {name} array provenance differs"
            )

    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"J1", "VPM"}:
        raise StudyValidationError("paired latency arm inventory differs")
    raw_vectors: dict[str, list[float]] = {}
    arm_results: dict[str, Any] = {}
    expected_specs = {"J1": 4, "VPM": 8}
    for arm, nfe in expected_specs.items():
        record = arms.get(arm)
        if (
            not isinstance(record, dict)
            or record.get("source") != "autonomous"
            or record.get("nfe") != nfe
        ):
            raise StudyValidationError(
                f"paired latency {arm} source/NFE differs"
            )
        stage = stages[arm]["stages"]["1000"]
        expected_records = {
            "arm_manifest": (
                Path(stages[arm]["arm_manifest_path"]),
                stages[arm]["arm_identity_sha256"],
            ),
            "stage_manifest": (
                Path(stage["stage_manifest_path"]),
                stage["stage_identity_sha256"],
            ),
            "stage_outcome": (
                Path(stage["path"]),
                stage["stage_outcome_identity_sha256"],
            ),
            "arm_outcome": (Path(stages[arm]["outcome_path"]), None),
            "resolved_config": (
                Path(stage["resolved_config"]["path"]),
                None,
            ),
        }
        for field, (expected_path, expected_identity_value) in (
            expected_records.items()
        ):
            expected_path = _regular_file(
                expected_path, f"{arm} paired {field}"
            )
            if field == "arm_outcome":
                expected_payload = _read_json(
                    expected_path, f"{arm} arm outcome"
                )
                expected_identity_value = expected_payload.get(
                    "identity_sha256"
                )
            _validate_paired_file_record(
                record.get(field),
                path=expected_path,
                label=f"{arm} {field}",
                identity_sha256=expected_identity_value,
            )
        snapshot = record.get("snapshot")
        snapshot_path = _regular_file(
            Path(stages[arm]["final_snapshot"]["path"]),
            f"{arm} paired final snapshot",
        )
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("path") != str(snapshot_path)
            or snapshot.get("sha256")
            != stages[arm]["final_snapshot"]["sha256"]
            or snapshot.get("sha256") != _hash_file(snapshot_path)
            or snapshot.get("bytes") != snapshot_path.stat().st_size
            or snapshot.get("checkpoint_start_iter") != 1000
            or snapshot.get("run_identity_sha256")
            != stages[arm]["arm_identity_sha256"]
        ):
            raise StudyValidationError(
                f"paired latency {arm} snapshot provenance differs"
            )
        audit = record.get("audit")
        expected_counters = {
            "wan_calls": nfe,
            "online_teacher_calls": 0,
            "auxiliary_clean_available": 0,
            "artifacts_collected": 1,
            "deployment_mode": 1,
        }
        if (
            not isinstance(audit, dict)
            or audit.get("nfe") != nfe
            or audit.get("wan_calls") != nfe
            or audit.get("online_teacher_calls") != 0
            or audit.get("clean_auxiliary_available") != 0
            or audit.get("deployment_mode") != 1
            or audit.get("trajectory_capture_enabled") is not False
            or audit.get("independent_forward_hook_wan_count") != nfe
            or audit.get("forward_hook_active_during_timing") is not False
            or audit.get("public_entrypoint")
            != "DualExplicitActionDiTModel.sample_future_deployable"
            or audit.get("sampler_counters") != expected_counters
        ):
            raise StudyValidationError(
                f"paired latency {arm} deployable audit differs"
            )
        raw, summary = _validated_paired_latency_vector(
            record.get("latency_ms"), label=f"paired {arm}"
        )
        raw_vectors[arm] = raw
        expected_fps = GENERATED_FUTURE_FRAMES * 1000.0 / summary["p95"]
        observed_fps = record.get(
            "generated_frames_per_second_at_p95"
        )
        if (
            isinstance(observed_fps, bool)
            or not isinstance(observed_fps, (int, float))
            or not math.isclose(
                float(observed_fps),
                expected_fps,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise StudyValidationError(
                f"paired latency {arm} throughput differs"
            )
        arm_results[arm] = {
            "source": "autonomous",
            "nfe": nfe,
            "latency_ms": summary,
            "generated_frames_per_second_at_p95": expected_fps,
            "snapshot": dict(snapshot),
            "audit": dict(audit),
        }

    paired = payload.get("paired")
    rounds = paired.get("rounds") if isinstance(paired, dict) else None
    if not isinstance(rounds, list) or len(rounds) != PAIRED_TIMED_PAIRS:
        raise StudyValidationError("paired timing round inventory differs")
    if paired.get("favorable_direction") != "positive means J1@4 is faster":
        raise StudyValidationError("paired timing favorable direction differs")
    differences: list[float] = []
    relative_per_pair: list[float] = []
    execution_orders: list[list[str]] = []
    for index, round_record in enumerate(rounds):
        expected_order = (
            ["J1", "VPM"] if index % 2 == 0 else ["VPM", "J1"]
        )
        if (
            not isinstance(round_record, dict)
            or round_record.get("pair_index") != index
            or round_record.get("execution_order") != expected_order
        ):
            raise StudyValidationError(
                f"paired timing round {index} order/index differs"
            )
        j1_value = raw_vectors["J1"][index]
        vpm_value = raw_vectors["VPM"][index]
        difference = vpm_value - j1_value
        relative = difference / vpm_value
        for field, expected in (
            ("J1_latency_ms", j1_value),
            ("VPM_latency_ms", vpm_value),
            ("favorable_difference_ms", difference),
            ("relative_improvement", relative),
        ):
            observed = round_record.get(field)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(
                    float(observed),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            ):
                raise StudyValidationError(
                    f"paired timing round {index} {field} differs"
                )
        differences.append(difference)
        relative_per_pair.append(relative)
        execution_orders.append(expected_order)
    if paired.get("rounds_sha256") != hashlib.sha256(
        _canonical_json_bytes(rounds)
    ).hexdigest():
        raise StudyValidationError("paired timing round digest differs")
    for field, expected in (
        ("favorable_difference_ms", differences),
        ("relative_improvement", relative_per_pair),
    ):
        observed = paired.get(field)
        if (
            not isinstance(observed, list)
            or len(observed) != len(expected)
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isclose(
                    float(item),
                    expected[index],
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                for index, item in enumerate(observed)
            )
        ):
            raise StudyValidationError(
                f"paired timing {field} vector differs"
            )
    expected_means = {
        "mean_favorable_difference_ms": sum(differences) / len(differences),
        "mean_relative_improvement": (
            sum(relative_per_pair) / len(relative_per_pair)
        ),
    }
    for field, expected in expected_means.items():
        observed = paired.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isclose(
                float(observed), expected, rel_tol=1e-12, abs_tol=1e-9
            )
        ):
            raise StudyValidationError(f"paired timing {field} differs")
    order_strata = paired.get("order_strata")
    expected_strata: dict[str, Any] = {}
    for first in ("J1", "VPM"):
        indexes = [
            index
            for index, order in enumerate(execution_orders)
            if order[0] == first
        ]
        expected_strata[f"{first}_first"] = {
            "count": len(indexes),
            "mean_favorable_difference_ms": (
                sum(differences[index] for index in indexes) / len(indexes)
            ),
            "mean_relative_improvement": (
                sum(relative_per_pair[index] for index in indexes)
                / len(indexes)
            ),
        }
    if not isinstance(order_strata, dict) or set(order_strata) != set(
        expected_strata
    ):
        raise StudyValidationError("paired timing order strata differ")
    for name, expected in expected_strata.items():
        observed = order_strata.get(name)
        if not isinstance(observed, dict) or observed.get("count") != 50:
            raise StudyValidationError(
                f"paired timing {name} count differs"
            )
        for field in (
            "mean_favorable_difference_ms",
            "mean_relative_improvement",
        ):
            value = observed.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(
                    float(value),
                    expected[field],
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            ):
                raise StudyValidationError(
                    f"paired timing {name} {field} differs"
                )

    effect = _counterbalanced_paired_latency_effect(
        raw_vectors["J1"],
        raw_vectors["VPM"],
        execution_orders,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=bootstrap_seed,
        label="paired-latency:J1-nfe4-vs-VPM-nfe8",
    )
    return {
        "path": str(path),
        "sha256": _hash_file(path),
        "identity_sha256": payload["identity_sha256"],
        "comparison": PAIRED_LATENCY_COMPARISON,
        "claim_role": "sole_final_speed_gate_evidence",
        "same_allocation": True,
        "same_node": slurm["same_node"],
        "same_B200": True,
        "same_process": True,
        "submission_job_id": paired_job["job_id"],
        "protocol": dict(protocol),
        "immutable_input_identity": {
            field: immutable[field] for field in sample_identity_fields
        },
        "arms": arm_results,
        "order_strata": expected_strata,
        "counterbalanced_paired_effect": effect,
    }


def _normalized_auc(
    update_to_mean: Mapping[int, float],
) -> float:
    updates = np.asarray(COMPLETED_UPDATES, dtype=np.float64)
    values = np.asarray(
        [update_to_mean[update] for update in COMPLETED_UPDATES],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise StudyValidationError("training curve contains non-finite values")
    return float(np.trapezoid(values, updates) / (updates[-1] - updates[0]))


def _time_to_threshold(
    update_to_mean: Mapping[int, float],
    *,
    threshold: float,
    direction: str,
    cumulative_wall: Mapping[int, int],
) -> dict[str, Any]:
    def passes(value: float) -> bool:
        return value <= threshold if direction == "lower" else value >= threshold

    first = None
    sustained = None
    for index, update in enumerate(COMPLETED_UPDATES):
        if first is None and passes(update_to_mean[update]):
            first = update
        if all(
            passes(update_to_mean[later])
            for later in COMPLETED_UPDATES[index:]
        ):
            sustained = update
            break
    return {
        "threshold": threshold,
        "direction": direction,
        "first_completed_update": first,
        "first_sustained_completed_update": sustained,
        "cumulative_stage_wall_seconds_at_first": (
            None if first is None else cumulative_wall[first]
        ),
        "cumulative_stage_wall_seconds_at_first_sustained": (
            None if sustained is None else cumulative_wall[sustained]
        ),
        "wall_time_scope": (
            "training stages including validation and visualization; offline "
            "V-JEPA cache extraction excluded"
        ),
    }


def _gate(
    *,
    gate_id: str,
    description: str,
    rule: str,
    value: Any,
    passed: bool | None,
    oracle_leakage: bool = False,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "description": description,
        "rule": rule,
        "value": value,
        "passed": passed,
        "status": (
            "unavailable" if passed is None else ("pass" if passed else "fail")
        ),
        "oracle_leakage": oracle_leakage,
        "deployable_evidence": not oracle_leakage,
    }


def _metric_improvement(comparison: Mapping[str, Any], metric: str) -> float:
    value = comparison["metrics"][metric]["relative_improvement"]
    if value is None:
        raise StudyValidationError(
            f"relative improvement is undefined for {metric}"
        )
    return float(value)


def _metric_improvement_ci_low(
    comparison: Mapping[str, Any], metric: str
) -> float:
    interval = comparison["metrics"][metric].get("bootstrap_ci")
    if (
        not isinstance(interval, dict)
        or isinstance(interval.get("low"), bool)
        or not isinstance(interval.get("low"), (int, float))
        or not math.isfinite(float(interval["low"]))
    ):
        raise StudyValidationError(
            f"paired relative-improvement CI is undefined for {metric}"
        )
    return float(interval["low"])


def _build_analysis(
    study_root: Path,
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    study, arm_roots, stages = _manifest_and_stage_inventory(study_root)
    analyzed, quality_inventory, units = _load_quality_evidence(
        study=study,
        study_root=study_root,
        arm_roots=arm_roots,
        stages=stages,
    )

    aggregates: dict[str, Any] = {}
    per_unit: dict[str, Any] = {}
    for arm in QUANTITATIVE_ARMS:
        aggregates[arm] = {}
        per_unit[arm] = {}
        for update in COMPLETED_UPDATES:
            aggregates[arm][str(update)] = {}
            per_unit[arm][str(update)] = {}
            for unit in units:
                unit_key = f"{unit[0]}|clip_index={unit[1]}"
                per_unit[arm][str(update)][unit_key] = {
                    "clip_id": unit[0],
                    "clip_index": unit[1],
                    "pairing_identity": analyzed[arm][update][unit][
                        "identity"
                    ],
                    "metrics": analyzed[arm][update][unit]["metrics"],
                    "evidence": analyzed[arm][update][unit]["entries"],
                }
            available_grid = _quality_grid(update)
            available_sources = tuple(
                dict.fromkeys(source for source, _nfe in available_grid)
            )
            for source in available_sources:
                aggregates[arm][str(update)][source] = {}
                for nfe in (
                    nfe
                    for grid_source, nfe in available_grid
                    if grid_source == source
                ):
                    aggregates[arm][str(update)][source][str(nfe)] = {}
                    for metric in METRIC_DIRECTIONS:
                        values = _values(
                            analyzed,
                            arm=arm,
                            update=update,
                            source=source,
                            nfe=nfe,
                            metric=metric,
                            units=units,
                        )
                        aggregates[arm][str(update)][source][str(nfe)][metric] = (
                            _summary(
                                values,
                                bootstrap_samples=bootstrap_samples,
                                confidence=confidence,
                                seed=bootstrap_seed,
                                label=(
                                    f"aggregate:{arm}:{update}:{source}:"
                                    f"{nfe}:{metric}"
                                ),
                            )
                        )

    final_comparisons: dict[str, Any] = {}

    def add_comparison(
        name: str,
        left_arm: str,
        reference_arm: str,
        left_source: str,
        reference_source: str,
        left_nfe: int,
        reference_nfe: int,
    ) -> None:
        final_comparisons[name] = _comparison(
            analyzed,
            left_arm=left_arm,
            reference_arm=reference_arm,
            update=1000,
            left_source=left_source,
            reference_source=reference_source,
            left_nfe=left_nfe,
            reference_nfe=reference_nfe,
            units=units,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
            label=name,
        )

    for nfe in NFE_STEPS:
        add_comparison(
            f"J1_autonomous_vs_off_nfe_{nfe}",
            "J1",
            "J1",
            "autonomous",
            "off",
            nfe,
            nfe,
        )
        add_comparison(
            f"J1_autonomous_vs_autonomous_shuffled_nfe_{nfe}",
            "J1",
            "J1",
            "autonomous",
            "autonomous_shuffled",
            nfe,
            nfe,
        )
        add_comparison(
            f"J1_oracle_matched_vs_autonomous_nfe_{nfe}",
            "J1",
            "J1",
            "oracle_matched",
            "autonomous",
            nfe,
            nfe,
        )
        add_comparison(
            f"J1_oracle_matched_vs_oracle_shuffled_nfe_{nfe}",
            "J1",
            "J1",
            "oracle_matched",
            "oracle_shuffled",
            nfe,
            nfe,
        )
        for reference_arm in ("VPM", "A1", "J0"):
            add_comparison(
                f"J1_vs_{reference_arm}_autonomous_nfe_{nfe}",
                "J1",
                reference_arm,
                "autonomous",
                "autonomous",
                nfe,
                nfe,
            )
    add_comparison(
        "J1_autonomous_nfe_4_vs_VPM_autonomous_nfe_8",
        "J1",
        "VPM",
        "autonomous",
        "autonomous",
        4,
        8,
    )

    training: dict[str, Any] = {
        "curve_source": "autonomous",
        "curve_nfe": 4,
        "x_axis": "completed optimizer updates",
        "auc_definition": (
            "trapezoidal area over completed updates 1..1000, divided by 999"
        ),
        "arms": {},
    }
    curve_means: dict[str, dict[str, dict[int, float]]] = {}
    per_unit_auc_values: dict[str, dict[str, list[float]]] = {}
    for arm in QUANTITATIVE_ARMS:
        curve_means[arm] = {}
        per_unit_auc_values[arm] = {}
        training["arms"][arm] = {"metrics": {}}
        cumulative_wall = {
            update: int(
                stages[arm]["stages"][str(update)][
                    "cumulative_wall_seconds_including_validation_and_visualization"
                ]
            )
            for update in COMPLETED_UPDATES
        }
        for metric, direction in METRIC_DIRECTIONS.items():
            update_to_mean = {
                update: float(
                    aggregates[arm][str(update)]["autonomous"]["4"][metric][
                        "mean"
                    ]
                )
                for update in COMPLETED_UPDATES
            }
            curve_means[arm][metric] = update_to_mean
            per_unit_auc = []
            for unit in units:
                unit_curve = {
                    update: float(
                        analyzed[arm][update][unit]["metrics"][
                            "autonomous"
                        ]["4"][metric]
                    )
                    for update in COMPLETED_UPDATES
                }
                per_unit_auc.append(_normalized_auc(unit_curve))
            per_unit_auc_values[arm][metric] = per_unit_auc
            training["arms"][arm]["metrics"][metric] = {
                "direction": direction,
                "milestone_means": {
                    str(update): value
                    for update, value in update_to_mean.items()
                },
                "normalized_auc_of_aggregate_mean_curve": _normalized_auc(
                    update_to_mean
                ),
                "per_unit_normalized_auc": _summary(
                    per_unit_auc,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=bootstrap_seed,
                    label=f"training-auc:{arm}:{metric}",
                ),
            }
    for arm in QUANTITATIVE_ARMS:
        cumulative_wall = {
            update: int(
                stages[arm]["stages"][str(update)][
                    "cumulative_wall_seconds_including_validation_and_visualization"
                ]
            )
            for update in COMPLETED_UPDATES
        }
        for metric, direction in METRIC_DIRECTIONS.items():
            threshold = curve_means["VPM"][metric][1000]
            training["arms"][arm]["metrics"][metric][
                "time_to_vpm_final_threshold"
            ] = _time_to_threshold(
                curve_means[arm][metric],
                threshold=threshold,
                direction=direction,
                cumulative_wall=cumulative_wall,
            )

    final_training_comparisons = {
        reference: _comparison(
            analyzed,
            left_arm="J1",
            reference_arm=reference,
            update=1000,
            left_source="autonomous",
            reference_source="autonomous",
            left_nfe=4,
            reference_nfe=4,
            units=units,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
            label=f"training-final:J1-vs-{reference}",
        )
        for reference in ("VPM", "A1", "J0")
    }
    training["final_J1_comparisons"] = final_training_comparisons
    auc_comparisons = {
        reference: {
            metric: _paired_effect(
                per_unit_auc_values["J1"][metric],
                per_unit_auc_values[reference][metric],
                direction=direction,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=bootstrap_seed,
                label=f"training-auc:J1-vs-{reference}:{metric}",
            )
            for metric, direction in METRIC_DIRECTIONS.items()
        }
        for reference in ("VPM", "A1", "J0")
    }
    training["paired_J1_auc_comparisons"] = auc_comparisons

    latency = _load_latency(arm_roots, stages)
    latency["per_arm_grid_claim_role"] = (
        "diagnostic telemetry only; independent allocations/processes cannot "
        "supply the final comparative speed gate"
    )
    paired_latency = _load_paired_latency(
        study=study,
        study_root=study_root,
        stages=stages,
        grid_latency=latency,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    latency["paired_final_claim"] = paired_latency
    training_vpm = final_training_comparisons["VPM"]
    training_a1 = final_training_comparisons["A1"]
    training_j0 = final_training_comparisons["J0"]
    temporal_vs_vpm = _metric_improvement(training_vpm, PRIMARY_METRIC)
    video_vs_vpm = _metric_improvement(training_vpm, "video_future_nmse")
    decoded_vs_vpm = _metric_improvement(
        training_vpm, "decoded_mse_unit_range"
    )
    temporal_vs_a1 = _metric_improvement(training_a1, PRIMARY_METRIC)
    temporal_vs_j0 = _metric_improvement(training_j0, PRIMARY_METRIC)
    temporal_vs_vpm_ci_low = _metric_improvement_ci_low(
        training_vpm, PRIMARY_METRIC
    )
    video_vs_vpm_ci_low = _metric_improvement_ci_low(
        training_vpm, "video_future_nmse"
    )
    decoded_vs_vpm_ci_low = _metric_improvement_ci_low(
        training_vpm, "decoded_mse_unit_range"
    )
    temporal_vs_a1_ci_low = _metric_improvement_ci_low(
        training_a1, PRIMARY_METRIC
    )
    temporal_vs_j0_ci_low = _metric_improvement_ci_low(
        training_j0, PRIMARY_METRIC
    )
    threshold_record = training["arms"]["J1"]["metrics"][PRIMARY_METRIC][
        "time_to_vpm_final_threshold"
    ]
    threshold_update = threshold_record["first_sustained_completed_update"]
    threshold_wall = threshold_record[
        "cumulative_stage_wall_seconds_at_first_sustained"
    ]
    vpm_final_wall = stages["VPM"]["stages"]["1000"][
        "cumulative_wall_seconds_including_validation_and_visualization"
    ]
    temporal_auc_comparison = auc_comparisons["VPM"][PRIMARY_METRIC]
    temporal_auc_ci_low = float(
        temporal_auc_comparison["bootstrap_ci"]["low"]
    )

    quality_4_vpm = final_comparisons["J1_vs_VPM_autonomous_nfe_4"]
    quality_8_vpm = final_comparisons["J1_vs_VPM_autonomous_nfe_8"]
    quality_4_off = final_comparisons["J1_autonomous_vs_off_nfe_4"]
    quality_4_shuffled = final_comparisons[
        "J1_autonomous_vs_autonomous_shuffled_nfe_4"
    ]
    quality_4_vs_vpm8 = final_comparisons[
        "J1_autonomous_nfe_4_vs_VPM_autonomous_nfe_8"
    ]
    oracle_4 = final_comparisons["J1_oracle_matched_vs_autonomous_nfe_4"]

    def quality_gate(comparison: Mapping[str, Any], primary_min: float) -> bool:
        return (
            _metric_improvement_ci_low(comparison, PRIMARY_METRIC)
            > primary_min
            and all(
                _metric_improvement_ci_low(comparison, metric) > -0.01
                for metric in QUALITY_GUARDRAILS
            )
        )

    off_mean = float(
        quality_4_off["metrics"][PRIMARY_METRIC]["reference_mean"]
    )
    autonomous_mean = float(
        quality_4_off["metrics"][PRIMARY_METRIC]["left_mean"]
    )
    oracle_mean = float(
        oracle_4["metrics"][PRIMARY_METRIC]["left_mean"]
    )
    oracle_gap_closure = _paired_gap_closure(
        _values(
            analyzed,
            arm="J1",
            update=1000,
            source="off",
            nfe=4,
            metric=PRIMARY_METRIC,
            units=units,
        ),
        _values(
            analyzed,
            arm="J1",
            update=1000,
            source="autonomous",
            nfe=4,
            metric=PRIMARY_METRIC,
            units=units,
        ),
        _values(
            analyzed,
            arm="J1",
            update=1000,
            source="oracle_matched",
            nfe=4,
            metric=PRIMARY_METRIC,
            units=units,
        ),
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=bootstrap_seed,
        label="oracle-gap-closure:J1:nfe4:temporal",
    )
    latency_j1 = paired_latency["arms"]["J1"]
    latency_vpm = paired_latency["arms"]["VPM"]
    j1_p95 = float(latency_j1["latency_ms"]["p95"])
    vpm_p95 = float(latency_vpm["latency_ms"]["p95"])
    paired_effect = paired_latency["counterbalanced_paired_effect"]
    order_strata = paired_latency["order_strata"]
    latency_pass = (
        float(paired_effect["bootstrap_ci"]["low"]) > 0.0
        and j1_p95 < vpm_p95
        and all(
            float(order_strata[name]["mean_favorable_difference_ms"]) > 0.0
            for name in ("J1_first", "VPM_first")
        )
    )
    latency_value = {
        "evidence": paired_latency["path"],
        "same_allocation": True,
        "same_node": paired_latency["same_node"],
        "J1_autonomous_nfe4_p95_ms": j1_p95,
        "VPM_autonomous_nfe8_p95_ms": vpm_p95,
        "p95_relative_reduction": (vpm_p95 - j1_p95) / vpm_p95,
        "counterbalanced_paired_mean_effect": paired_effect,
        "execution_order_strata": order_strata,
    }
    j1_p95_fps = float(
        latency_j1["generated_frames_per_second_at_p95"]
    )
    latency["realtime_dagger_diagnostic"] = {
        "J1_autonomous_nfe4_generated_frames_per_second_at_p95": j1_p95_fps,
        "meets_5_fps": j1_p95_fps >= 5.0,
        "meets_10_fps": j1_p95_fps >= 10.0,
        "source": "same-allocation paired final benchmark",
        "preregistered_success_gate": False,
    }

    gates = {
        "training": [
            _gate(
                gate_id="T1",
                description=(
                    "J1 final held-out temporal reconstruction improves over VPM"
                ),
                rule=(
                    "paired 95% CI lower bound on relative improvement >= 3% "
                    "at update 1000, autonomous NFE=4"
                ),
                value=training_vpm["metrics"][PRIMARY_METRIC],
                passed=temporal_vs_vpm_ci_low >= 0.03,
            ),
            _gate(
                gate_id="T2",
                description="J1 final video latent has no material regression",
                rule=(
                    "paired 95% CI lower bound on video-NMSE relative "
                    "improvement > -1% versus VPM"
                ),
                value=training_vpm["metrics"]["video_future_nmse"],
                passed=video_vs_vpm_ci_low > -0.01,
            ),
            _gate(
                gate_id="T3",
                description="J1 final decoded MSE has no material regression",
                rule=(
                    "paired 95% CI lower bound on decoded raw-RGB MSE "
                    "relative improvement > -1% versus VPM"
                ),
                value=training_vpm["metrics"]["decoded_mse_unit_range"],
                passed=decoded_vs_vpm_ci_low > -0.01,
            ),
            _gate(
                gate_id="T4",
                description=(
                    "J1 has CI-supported faster temporal convergence than VPM"
                ),
                rule=(
                    "descriptive sustained threshold update <=800, J1 wall "
                    "time there < VPM wall time at update 1000, and paired "
                    "AUC-improvement 95% CI lower bound >0"
                ),
                value={
                    "descriptive_threshold_crossing": threshold_record,
                    "threshold_crossing_uncertainty_limit": (
                        "milestone crossing is a point-estimate descriptor; "
                        "the paired per-clip AUC CI supplies inferential support"
                    ),
                    "J1_wall_seconds_at_crossing": threshold_wall,
                    "VPM_wall_seconds_at_update_1000": vpm_final_wall,
                    "paired_temporal_auc_effect": temporal_auc_comparison,
                },
                passed=(
                    threshold_update is not None
                    and threshold_update <= 800
                    and threshold_wall is not None
                    and threshold_wall < vpm_final_wall
                    and temporal_auc_ci_low > 0.0
                ),
            ),
            _gate(
                gate_id="T5",
                description="J1 beats auxiliary-objective-only A1",
                rule=(
                    "paired 95% CI lower bound on temporal relative "
                    "improvement >0 at update 1000/NFE=4"
                ),
                value=training_a1["metrics"][PRIMARY_METRIC],
                passed=temporal_vs_a1_ci_low > 0,
            ),
            _gate(
                gate_id="T6",
                description="V-JEPA-leading J1 is at least as good as aligned J0",
                rule=(
                    "paired 95% CI lower bound on temporal relative "
                    "improvement >=0 at update 1000/NFE=4"
                ),
                value=training_j0["metrics"][PRIMARY_METRIC],
                passed=temporal_vs_j0_ci_low >= 0,
            ),
        ],
        "inference": [
            _gate(
                gate_id="I0",
                description=(
                    "The diagnostic grid and final paired latency inventories "
                    "are complete"
                ),
                rule=(
                    "VPM/A1/J0/J1 x autonomous/off x seven NFEs, plus the "
                    "same-allocation J1@4 versus VPM@8 paired benchmark"
                ),
                value={
                    "diagnostic_grid_observed": (
                        latency["observed_record_count"]
                    ),
                    "diagnostic_grid_expected": (
                        latency["expected_record_count"]
                    ),
                    "paired_final_evidence": paired_latency["path"],
                    "paired_rounds": paired_effect["n_paired_rounds"],
                },
                passed=True if latency["complete"] else None,
            ),
            _gate(
                gate_id="I1",
                description="Generated V-JEPA state helps versus same-checkpoint off",
                rule=(
                    "temporal paired-CI lower bound >0 and video/decoded-MSE "
                    "paired-CI lower bounds >-1% at J1 autonomous NFE=4"
                ),
                value=quality_4_off["metrics"],
                passed=quality_gate(quality_4_off, 0.0),
            ),
            _gate(
                gate_id="I2",
                description="Aligned generated V-JEPA helps versus shuffled generated state",
                rule=(
                    "temporal paired-CI lower bound >0 and video/decoded-MSE "
                    "paired-CI lower bounds >-1% at J1 NFE=4"
                ),
                value=quality_4_shuffled["metrics"],
                passed=quality_gate(quality_4_shuffled, 0.0),
            ),
            _gate(
                gate_id="I3",
                description="J1 autonomous beats VPM at the same four-step budget",
                rule=(
                    "temporal paired-CI lower bound >0 and video/decoded-MSE "
                    "paired-CI lower bounds >-1% at NFE=4"
                ),
                value=quality_4_vpm["metrics"],
                passed=quality_gate(quality_4_vpm, 0.0),
            ),
            _gate(
                gate_id="I4",
                description=(
                    "J1 four-step held-out reconstruction approaches VPM "
                    "eight-step reconstruction"
                ),
                rule=(
                    "paired 95% CI lower bounds for temporal/video/decoded "
                    "MSE relative improvement are all > -1%"
                ),
                value=quality_4_vs_vpm8["metrics"],
                passed=all(
                    _metric_improvement_ci_low(
                        quality_4_vs_vpm8, metric
                    )
                    > -0.01
                    for metric in (PRIMARY_METRIC, *QUALITY_GUARDRAILS)
                ),
            ),
            _gate(
                gate_id="I5",
                description="The J1-versus-VPM direction persists at NFE=8",
                rule=(
                    "paired 95% CI lower bound on temporal relative "
                    "improvement >0"
                ),
                value=quality_8_vpm["metrics"][PRIMARY_METRIC],
                passed=(
                    _metric_improvement_ci_low(
                        quality_8_vpm, PRIMARY_METRIC
                    )
                    > 0
                ),
            ),
            _gate(
                gate_id="I6",
                description=(
                    "Four-step J1 has robustly lower end-to-end latency than "
                    "eight-step VPM on the same B200"
                ),
                rule=(
                    "same-allocation counterbalanced paired-bootstrap 95% CI "
                    "lower bound on mean relative speedup >0, J1@4 p95 < "
                    "VPM@8 p95, and favorable mean difference in both "
                    "execution-order strata"
                ),
                value=latency_value,
                passed=latency_pass,
            ),
        ],
        "mechanism": [
            _gate(
                gate_id="M1",
                description="Generated V-JEPA closes a positive fraction of oracle headroom",
                rule=(
                    "paired-bootstrap 95% CI lower bound for "
                    "mean(off-autonomous)/mean(off-oracle) >0 at NFE=4"
                ),
                value={
                    "off_mean": off_mean,
                    "autonomous_mean": autonomous_mean,
                    "oracle_matched_mean": oracle_mean,
                    "paired_oracle_gap_closure": oracle_gap_closure,
                    "oracle_headroom_is_leakage_only": True,
                },
                passed=(
                    isinstance(oracle_gap_closure.get("bootstrap_ci"), dict)
                    and oracle_gap_closure["bootstrap_ci"]["low"] > 0
                ),
                oracle_leakage=True,
            ),
        ],
    }
    literal_training_pass = all(
        gate["passed"] is True for gate in gates["training"]
    )
    literal_inference_pass = all(
        gate["passed"] is True for gate in gates["inference"]
    )
    literal_mechanism_pass = all(
        gate["passed"] is True for gate in gates["mechanism"]
    )
    any_unavailable = any(
        gate["passed"] is None
        for category in gates.values()
        for gate in category
    )
    if literal_training_pass and literal_inference_pass:
        conclusion = (
            "The preregistered evidence supports both faster training convergence "
            "and lower-error/faster deployable held-out reconstruction for J1 "
            "versus VPM. The separate oracle mechanism diagnostic does not "
            "supply or gate this deployable claim."
        )
    elif any_unavailable:
        conclusion = (
            "The study is not yet conclusive because at least one preregistered "
            "gate lacks required evidence; no faster/lower-reconstruction-error "
            "claim is made."
        )
    else:
        conclusion = (
            "The preregistered joint claim is not demonstrated: one or more "
            "training or deployable-inference gates failed."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "vjepa2_controlled_study_analysis",
        "created_at_utc": _utc_now(),
        "read_only": True,
        "study": {
            "study_root": str(study_root),
            "study_id": study.get("study_id"),
            "study_manifest_path": str(study_root / "study_manifest.json"),
            "study_manifest_sha256": _hash_file(
                study_root / "study_manifest.json"
            ),
            "study_identity_sha256": study.get("identity_sha256"),
            "wandb_provenance": dict(study["wandb"]),
        },
        "contract": {
            "quantitative_baseline": "VPM",
            "quantitative_arms": list(QUANTITATIVE_ARMS),
            "V0_role": "architecture/provenance sanity only",
            "V0_quantitative_metrics_available": False,
            "V0_exclusion_reason": (
                "legacy V0 does not emit uncompressed paired latent/source/NFE "
                "artifacts; compressed side-by-side MP4s are never used for claims"
            ),
            "completed_update_milestones": list(COMPLETED_UPDATES),
            "nfe": list(NFE_STEPS),
            "sources": list(SOURCES),
            "deployable_sources": list(DEPLOYABLE_SOURCES),
            "oracle_sources": {
                "names": list(ORACLE_SOURCES),
                "leakage_only": True,
                "claim_eligible": False,
            },
            "sigma_convention": "sigma=1 noise, sigma=0 clean",
            "auxiliary_history_latent_frames_required": 0,
            "online_teacher_calls_required": 0,
            "actual_wan_calls_must_equal_nfe": True,
            "final_speed_comparison": (
                "same-allocation, same-process, counterbalanced paired "
                "J1 autonomous NFE=4 versus VPM autonomous NFE=8"
            ),
            "paired_unit": "immutable fixed-test clip_id/clip_index",
            "fixed_test_clip_count": EXPECTED_TEST_CLIPS,
            "trainer_visualization_role": "diagnostic_only",
            "primary_decoded_metric_target": (
                "cached raw held-out future RGB"
            ),
            "vae_reconstruction_metrics_role": "diagnostic_only",
            "perceptual_metrics_available": False,
            "metric_directions": dict(METRIC_DIRECTIONS),
        },
        "validation": {
            "passed": True,
            "paired_unit_count": len(units),
            "paired_units": [
                {"clip_id": clip_id, "clip_index": clip_index}
                for clip_id, clip_index in units
            ],
            "all_primary_milestones_present": True,
            "all_pinned_quality_grid_rows_present": True,
            "auxiliary_history_zero": True,
            "training_and_inference_teacher_calls_zero": True,
            "wan_call_counts_equal_nfe": True,
            "oracle_leakage_labels_present": True,
            "all_non_off_sources_equal_at_nfe1": True,
            "VPM_A1_all_source_outputs_are_exact_no_ops": True,
            "VPM_A1_effective_state_and_clock_gates_are_exactly_zero": True,
            "trainer_visualization_excluded_from_scientific_aggregates": True,
            "raw_held_out_rgb_is_primary_decoded_target": True,
            "paired_speed_evidence_same_allocation_node_gpu_process": True,
            "paired_speed_inputs_bit_identical": True,
            "paired_speed_future_and_clean_auxiliary_unavailable": True,
            "paired_speed_online_teacher_calls_zero": True,
            "paired_speed_actual_wan_calls_equal_nfe": True,
        },
        "stage_provenance_and_wall_time": stages,
        "quality": {
            "aggregate_bootstrap_unit": (
                "immutable fixed-test clip_id/clip_index"
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence": confidence,
            "evidence_inventory": quality_inventory,
            "arm_milestone_source_nfe_metrics": aggregates,
            "per_unit_metrics": per_unit,
            "final_completed_update_comparisons": final_comparisons,
        },
        "training_convergence": training,
        "latency": latency,
        "success_gates": {
            **gates,
            "training_all_pass": literal_training_pass,
            "inference_all_pass": literal_inference_pass,
            "oracle_mechanism_all_pass": literal_mechanism_pass,
            "oracle_mechanism_gates_are_not_deployable_claim_prerequisites": (
                True
            ),
            "joint_claim_pass": literal_training_pass and literal_inference_pass,
            "has_unavailable_gate": any_unavailable,
            "uncertainty_policy": (
                "reconstruction, noninferiority, mechanism, final training, and "
                "convergence-AUC gates use paired bootstrap confidence-interval "
                "lower bounds over 128 immutable clips. Milestone threshold "
                "crossing is descriptive and cannot pass the faster-training "
                "gate without CI-supported paired AUC and wall-time evidence. "
                "The final speed gate uses a paired bootstrap stratified by "
                "first-executed arm within one shared B200 allocation"
            ),
        },
        "conclusion": conclusion,
        "claim_limits": [
            "Oracle matched/shuffled results consume hidden future V-JEPA targets "
            "and quantify headroom or mechanism only.",
            "Offline V-JEPA extraction cost is reported separately and excluded "
            "from optimizer convergence and autonomous inference latency.",
            "V0 is not a quantitative comparator without a separate compatible "
            "posthoc evaluator; its compressed MP4s are intentionally ignored.",
            "A positive result is limited to this model, dataset, seed, and fixed "
            "test inventory until replicated.",
            "No LPIPS, FVD, or pretrained perceptual metric is reported because "
            "the immutable study pins no such checkpoint; claims are restricted "
            "to held-out latent and raw-RGB reconstruction metrics.",
            "The final latency claim compares J1 autonomous NFE=4 with the "
            "parameter-matched VPM autonomous NFE=8 baseline in one process on "
            "one B200 using identical inputs and counterbalanced paired rounds. "
            "The wider per-arm grid is diagnostic only. V0 has no compatible "
            "deployable latency evaluator and is not part of the speed claim.",
        ],
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    gates = payload["success_gates"]

    def category_status(name: str) -> str:
        records = gates[name]
        if all(record["passed"] is True for record in records):
            return "PASS"
        if any(record["passed"] is None for record in records):
            return "INCOMPLETE"
        return "FAIL"

    lines = [
        "# V-JEPA 2.1 controlled video-diffusion study",
        "",
        f"Study: `{payload['study']['study_id']}`",
        "",
        payload["conclusion"],
        "",
        "## Evidence integrity",
        "",
        (
            f"Validated {payload['validation']['paired_unit_count']} paired "
            "immutable fixed-test clips at updates "
            f"{payload['contract']['completed_update_milestones']} using the "
            "pinned intermediate/final source-by-NFE grids. Auxiliary history "
            "is zero, online "
            "teacher calls are zero, every measured Wan-call count equals NFE, "
            "all non-off sources are identical at final NFE=1, and every VPM/A1 "
            "conditioning-source intervention is an exact no-op. Decoded "
            "metrics use cached raw held-out future RGB; VAE reconstruction "
            "comparisons and trainer visualizations are diagnostic only."
        ),
        "",
        "## Preregistered gates",
        "",
        "| Gate | Result | Rule |",
        "|---|---:|---|",
    ]
    for category in ("training", "inference", "mechanism"):
        for gate in gates[category]:
            lines.append(
                f"| {gate['id']} — {gate['description']} | "
                f"**{gate['status'].upper()}** | {gate['rule']} |"
            )
    lines.extend(
        [
            "",
            (
                f"Training gates: **{category_status('training')}**. "
                f"Inference gates: **{category_status('inference')}**. "
                f"Oracle mechanism diagnostic: "
                f"**{category_status('mechanism')}** (leakage-only; does not "
                "gate the deployable claim). "
                f"Joint claim: **{'PASS' if gates['joint_claim_pass'] else 'NOT DEMONSTRATED'}**."
            ),
            "",
            "## Key final comparisons",
            "",
            "| Comparison | Temporal improvement | Video-NMSE improvement | Decoded-MSE improvement |",
            "|---|---:|---:|---:|",
        ]
    )
    comparisons = payload["quality"]["final_completed_update_comparisons"]
    rows = (
        ("J1@4 vs VPM@4", "J1_vs_VPM_autonomous_nfe_4"),
        ("J1@4 vs same-checkpoint off", "J1_autonomous_vs_off_nfe_4"),
        (
            "J1@4 vs generated-state shuffled",
            "J1_autonomous_vs_autonomous_shuffled_nfe_4",
        ),
        (
            "J1@4 vs VPM@8",
            "J1_autonomous_nfe_4_vs_VPM_autonomous_nfe_8",
        ),
    )
    for label, key in rows:
        metrics = comparisons[key]["metrics"]
        lines.append(
            f"| {label} | "
            f"{_format_number(100 * metrics[PRIMARY_METRIC]['relative_improvement'])}% | "
            f"{_format_number(100 * metrics['video_future_nmse']['relative_improvement'])}% | "
            f"{_format_number(100 * metrics['decoded_mse_unit_range']['relative_improvement'])}% |"
        )
    latency = payload["latency"]
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                f"Latency inventory: {latency['observed_record_count']}/"
                f"{latency['expected_record_count']} records "
                f"({'complete' if latency['complete'] else 'incomplete'}). "
                "This wider grid is diagnostic; the final speed gate uses the "
                "single-allocation counterbalanced paired artifact."
            ),
            "",
            (
                "Positive improvement means the left side is better. Oracle "
                "results are hidden-future leakage and cannot support a "
                "deployable claim. VPM is the parameter-matched quantitative "
                "baseline. V0 remains architecture/provenance-only because its "
                "legacy MP4 visualization is not an uncompressed paired "
                "source/NFE evaluator. No LPIPS/FVD claim is made because no "
                "perceptual checkpoint is pinned. Deployable latency is compared "
                "against parameter-matched VPM with both checkpoints resident "
                "in one process on the same B200 and identical observable "
                "inputs; V0 is outside the speed claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _output_paths(
    json_value: str | Path,
    markdown_value: str | Path | None,
    study_root: Path,
) -> tuple[Path, Path]:
    raw_json = Path(json_value).expanduser()
    if raw_json.suffix.lower() != ".json":
        raise StudyValidationError("JSON output path must end in .json")
    json_parent = _canonical_directory(raw_json.parent, "JSON output parent")
    json_path = json_parent / raw_json.name
    raw_markdown = (
        Path(markdown_value).expanduser()
        if markdown_value is not None
        else raw_json.with_suffix(".md")
    )
    if raw_markdown.suffix.lower() != ".md":
        raise StudyValidationError("Markdown output path must end in .md")
    markdown_parent = _canonical_directory(
        raw_markdown.parent, "Markdown output parent"
    )
    markdown_path = markdown_parent / raw_markdown.name
    if json_path == markdown_path:
        raise StudyValidationError("JSON and Markdown outputs must differ")
    for path in (json_path, markdown_path):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise StudyValidationError(f"output already exists: {path}")
        if path == study_root or path.is_relative_to(study_root):
            raise StudyValidationError(
                f"analysis output must be outside the read-only study root: {path}"
            )
    return json_path, markdown_path


def _exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def analyze(
    study_root: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_729,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Validate all evidence and exclusively create JSON and Markdown outputs."""

    if bootstrap_samples < 100:
        raise StudyValidationError("bootstrap_samples must be at least 100")
    if not 0.5 < confidence < 1.0:
        raise StudyValidationError("confidence must be strictly between 0.5 and 1")
    root = _canonical_directory(study_root, "study root")
    json_path, markdown_path = _output_paths(
        output_json, output_markdown, root
    )
    payload = _build_analysis(
        root,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    json_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown_bytes = _markdown(payload).encode("utf-8")
    # Both output paths were preflighted before reading the run. Exclusive
    # creation still protects against concurrent analyzers.
    _exclusive_write(json_path, json_bytes)
    _exclusive_write(markdown_path, markdown_bytes)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument(
        "--output",
        "--output-json",
        dest="output_json",
        required=True,
        help="exclusive JSON output path outside the study root",
    )
    parser.add_argument(
        "--output-markdown",
        help="exclusive Markdown output path (default: JSON path with .md)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = analyze(
            args.study_root,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
        )
    except StudyValidationError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "study_id": payload["study"]["study_id"],
                "paired_units": payload["validation"]["paired_unit_count"],
                "joint_claim_pass": payload["success_gates"]["joint_claim_pass"],
                "conclusion": payload["conclusion"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
