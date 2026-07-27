#!/usr/bin/env python3
"""Analyze matched dual-diffusion NFE artifacts without modifying a run.

Each arm root must contain one visualization iteration's
``latent_trajectory_rank_<rank>.safetensors`` files and their JSON sidecars.
Artifacts are paired by ``(dataset, global_rank)``.  The output is created with
exclusive-create semantics at a caller-supplied path outside every arm root.

LACWM clock convention: ``sigma=1`` is noise and ``sigma=0`` is clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open


NFE_STEPS = (1, 2, 4, 8)
SIGMA_CONVENTION = "1=noise,0=clean"
ARM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")
SCALED_ARM_NAME_RE = re.compile(r"^(matched|shuffled)_s([0-9]+)$")
ARTIFACT_NAME_RE = re.compile(r"^latent_trajectory_rank_([0-9]+)[.]safetensors$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_TENSOR_RE = re.compile(
    r"^(video_final|tf_final|decoded_future)"
    r"(?:_(off|oracle_matched|oracle_shuffled|autonomous_shuffled|"
    r"autonomous_legacy))?_nfe_([0-9]+)$"
)
METRIC_DIRECTIONS = {
    "video_future_nmse": "lower",
    "tf_future_nmse": "lower",
    "decoded_mse_unit_range": "lower",
    "decoded_psnr_db": "higher",
    "decoded_temporal_difference_mse_unit_range": "lower",
}
CONDITION_MODE_NAMES = {0: "off", 1: "matched", 2: "shuffled"}
EVALUATION_CONDITION_SOURCE_NAMES = {
    0: "autonomous",
    1: "off",
    2: "oracle_matched",
    3: "oracle_shuffled",
    4: "autonomous_shuffled",
    5: "autonomous_legacy",
}
ORACLE_SOURCES = {"oracle_matched", "oracle_shuffled"}
PROMISING_RELATIVE_IMPROVEMENT = 0.03
EQUIVALENCE_MARGIN = 0.02


class ArtifactValidationError(RuntimeError):
    """Raised when an artifact set is unsafe or not causally comparable."""


@dataclass(frozen=True)
class ArtifactRecord:
    arm: str
    root: Path
    path: Path
    sidecar: Path
    sha256: str
    iteration: int
    dataset: str
    global_rank: int
    tensor_schema: Mapping[str, tuple[tuple[int, ...], str]]
    metadata: Mapping[str, str]

    @property
    def pair_key(self) -> tuple[str, int]:
        return self.dataset, self.global_rank


def _required_tensor_keys(nfe_steps: Sequence[int] = NFE_STEPS) -> set[str]:
    keys = {
        "video_clean",
        "tf_clean",
        "ground_truth_future_uint8",
        "history_latent_frames",
        "video_initial_state",
        "tf_initial_state",
        "tf_initial_noise",
        "evaluation_noise_seed",
        "evaluation_nfe_steps",
        "evaluation_condition_source_codes",
        "oracle_sources_are_leakage",
        "condition_on_tf",
        "condition_mode_code",
    }
    for nfe in nfe_steps:
        keys.update(
            {
                f"video_final_nfe_{nfe}",
                f"tf_final_nfe_{nfe}",
                f"decoded_future_nfe_{nfe}",
            }
        )
    return keys


def _source_infix(source: str) -> str:
    if source == "autonomous":
        return ""
    if source not in EVALUATION_CONDITION_SOURCE_NAMES.values():
        raise ArtifactValidationError(
            f"unknown evaluation condition source: {source!r}"
        )
    return f"_{source}"


def _source_tensor_keys(
    source: str, nfe_steps: Sequence[int] = NFE_STEPS
) -> set[str]:
    infix = _source_infix(source)
    return {
        f"{prefix}{infix}_nfe_{nfe}"
        for nfe in nfe_steps
        for prefix in ("video_final", "tf_final", "decoded_future")
    }


def _validate_source_tensor_inventory(
    tensor_keys: set[str],
    declared_sources: Sequence[str],
    nfe_steps: Sequence[int] = NFE_STEPS,
) -> None:
    expected = set()
    for source in declared_sources:
        expected.update(_source_tensor_keys(source, nfe_steps))

    actual = set()
    source_prefixes = ("video_final", "tf_final", "decoded_future")
    for key in tensor_keys:
        if not key.startswith(source_prefixes):
            continue
        match = SOURCE_TENSOR_RE.fullmatch(key)
        if match is None:
            raise ArtifactValidationError(
                f"unrecognized condition-source tensor key: {key!r}"
            )
        source = match.group(2) or "autonomous"
        nfe = int(match.group(3))
        if source not in declared_sources:
            raise ArtifactValidationError(
                f"tensor {key!r} belongs to undeclared source {source!r}"
            )
        if nfe not in nfe_steps:
            raise ArtifactValidationError(
                f"tensor {key!r} uses undeclared NFE={nfe}"
            )
        actual.add(key)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ArtifactValidationError(
            "declared condition-source tensor inventory is incomplete or "
            f"inconsistent; missing={missing}, extra={extra}"
        )


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
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    byte_view = value.view(torch.uint8).numpy()
    digest.update(memoryview(byte_view))
    return digest.hexdigest()


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ArtifactValidationError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise ArtifactValidationError(
            f"{label} must be a non-empty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _ensure_no_symlink_components(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ArtifactValidationError(
                    f"artifact path contains a symlink component: {current}"
                )
        except FileNotFoundError as exc:
            raise ArtifactValidationError(
                f"artifact path disappeared during validation: {current}"
            ) from exc


def _load_json_strict(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactValidationError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON sidecar {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"JSON sidecar must contain an object: {path}")
    return payload


def _manifest_tensor_schema(
    payload: Mapping[str, Any], path: Path
) -> dict[str, tuple[tuple[int, ...], str]]:
    raw_schema = payload.get("tensors")
    if not isinstance(raw_schema, dict) or not raw_schema:
        raise ArtifactValidationError(f"sidecar has no tensor schema: {path}")
    schema: dict[str, tuple[tuple[int, ...], str]] = {}
    for key, record in raw_schema.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise ArtifactValidationError(f"malformed tensor schema in {path}")
        shape = record.get("shape")
        dtype = record.get("dtype")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            )
            or not isinstance(dtype, str)
        ):
            raise ArtifactValidationError(
                f"invalid schema for tensor {key!r} in {path}"
            )
        schema[key] = (tuple(shape), dtype)
    return schema


def _validate_artifact(
    arm: str,
    root: Path,
    raw_path: Path,
    nfe_steps: Sequence[int] = NFE_STEPS,
) -> ArtifactRecord:
    _ensure_no_symlink_components(raw_path, root)
    path = _canonical_regular_file(raw_path, f"{arm} safetensors artifact")
    match = ARTIFACT_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ArtifactValidationError(f"unexpected artifact filename: {path}")
    filename_rank = int(match.group(1))

    raw_sidecar = raw_path.with_suffix(".json")
    _ensure_no_symlink_components(raw_sidecar, root)
    sidecar = _canonical_regular_file(raw_sidecar, f"{arm} artifact sidecar")
    payload = _load_json_strict(sidecar)

    iteration = payload.get("iteration")
    dataset = payload.get("dataset")
    global_rank = payload.get("global_rank")
    sigma_convention = payload.get("sigma_convention")
    expected_sha256 = payload.get("safetensors_sha256")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ArtifactValidationError(f"invalid iteration in {sidecar}")
    if not isinstance(dataset, str) or not dataset or dataset != path.parent.name:
        raise ArtifactValidationError(
            f"dataset provenance does not match the artifact directory in {sidecar}"
        )
    if (
        not isinstance(global_rank, int)
        or isinstance(global_rank, bool)
        or global_rank < 0
        or global_rank != filename_rank
    ):
        raise ArtifactValidationError(
            f"global-rank provenance does not match the filename in {sidecar}"
        )
    if sigma_convention != SIGMA_CONVENTION:
        raise ArtifactValidationError(
            f"unsupported sigma convention in {sidecar}: {sigma_convention!r}"
        )
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ArtifactValidationError(f"invalid safetensors SHA-256 in {sidecar}")
    actual_sha256 = _hash_file(path)
    if actual_sha256 != expected_sha256:
        raise ArtifactValidationError(f"safetensors SHA-256 mismatch: {path}")

    manifest_schema = _manifest_tensor_schema(payload, sidecar)
    missing = sorted(_required_tensor_keys(nfe_steps) - set(manifest_schema))
    if missing:
        raise ArtifactValidationError(
            f"artifact is missing required tensor keys {missing}: {path}"
        )

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            actual_keys = set(handle.keys())
            if actual_keys != set(manifest_schema):
                raise ArtifactValidationError(
                    f"sidecar/artifact tensor-key mismatch: {path}"
                )
            actual_schema: dict[str, tuple[tuple[int, ...], str]] = {}
            for key in sorted(actual_keys):
                tensor = handle.get_tensor(key)
                actual_schema[key] = (tuple(tensor.shape), str(tensor.dtype))
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(
            f"could not read safetensors artifact {path}: {exc}"
        ) from exc

    if actual_schema != manifest_schema:
        raise ArtifactValidationError(
            f"sidecar/artifact tensor-shape or dtype mismatch: {path}"
        )
    expected_metadata = {
        "iteration": str(iteration),
        "dataset": dataset,
        "sigma_convention": SIGMA_CONVENTION,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ArtifactValidationError(
                f"safetensors metadata {key!r} mismatch in {path}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )

    return ArtifactRecord(
        arm=arm,
        root=root,
        path=path,
        sidecar=sidecar,
        sha256=actual_sha256,
        iteration=iteration,
        dataset=dataset,
        global_rank=global_rank,
        tensor_schema=actual_schema,
        metadata=metadata,
    )


def _discover_arm(
    arm: str, root: Path, nfe_steps: Sequence[int] = NFE_STEPS
) -> dict[tuple[str, int], ArtifactRecord]:
    raw_paths = sorted(root.rglob("latent_trajectory_rank_*.safetensors"))
    if not raw_paths:
        raise ArtifactValidationError(f"arm {arm!r} contains no rank artifacts: {root}")

    records: dict[tuple[str, int], ArtifactRecord] = {}
    for raw_path in raw_paths:
        record = _validate_artifact(arm, root, raw_path, nfe_steps)
        if record.pair_key in records:
            raise ArtifactValidationError(
                f"arm {arm!r} has duplicate dataset/rank pair {record.pair_key}; "
                "point the arm at exactly one visualization iteration"
            )
        records[record.pair_key] = record

    iterations = {record.iteration for record in records.values()}
    if len(iterations) != 1:
        raise ArtifactValidationError(
            f"arm {arm!r} mixes visualization iterations: {sorted(iterations)}"
        )
    for dataset in sorted({key[0] for key in records}):
        ranks = sorted(rank for name, rank in records if name == dataset)
        if ranks != list(range(ranks[-1] + 1)):
            raise ArtifactValidationError(
                f"arm {arm!r} dataset {dataset!r} has non-contiguous ranks: {ranks}"
            )
    return records


def _validate_float_latent(
    tensor: torch.Tensor, key: str, expected_shape: tuple[int, ...] | None = None
) -> None:
    if tensor.ndim != 5:
        raise ArtifactValidationError(
            f"{key} must have latent layout [B,C,T,H,W], got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise ArtifactValidationError(f"{key} must be floating point")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ArtifactValidationError(
            f"{key} shape {tuple(tensor.shape)} != {expected_shape}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ArtifactValidationError(f"{key} contains non-finite values")


def _validate_decoded(
    tensor: torch.Tensor, key: str, expected_shape: tuple[int, ...] | None = None
) -> None:
    if tensor.ndim != 5 or tensor.shape[1] != 3:
        raise ArtifactValidationError(
            f"{key} must have RGB layout [B,3,T,H,W], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.uint8:
        raise ArtifactValidationError(f"{key} must have dtype torch.uint8")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ArtifactValidationError(
            f"{key} shape {tuple(tensor.shape)} != {expected_shape}"
        )
    if tensor.shape[2] < 2:
        raise ArtifactValidationError(
            f"{key} needs at least two future frames for temporal-difference MSE"
        )


def _future_nmse(
    prediction: torch.Tensor, target: torch.Tensor, history_frames: int, key: str
) -> float:
    prediction_future = prediction[:, :, history_frames:].double()
    target_future = target[:, :, history_frames:].double()
    denominator = torch.sum(target_future.square()).item()
    if not math.isfinite(denominator) or denominator <= 0:
        raise ArtifactValidationError(
            f"{key} clean future has zero or non-finite energy; NMSE is undefined"
        )
    numerator = torch.sum((prediction_future - target_future).square()).item()
    value = numerator / denominator
    if not math.isfinite(value):
        raise ArtifactValidationError(f"{key} produced a non-finite NMSE")
    return float(value)


def _decoded_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float]:
    prediction_float = prediction.double().div_(255.0)
    target_float = target.double().div_(255.0)
    mse = torch.mean((prediction_float - target_float).square()).item()
    temporal_error = (
        torch.diff(prediction_float, dim=2) - torch.diff(target_float, dim=2)
    )
    temporal_mse = torch.mean(temporal_error.square()).item()
    # The floor keeps exact uint8 matches finite and strict-JSON serializable.
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    if not all(math.isfinite(value) for value in (mse, temporal_mse, psnr)):
        raise ArtifactValidationError("decoded metrics contain a non-finite value")
    return float(mse), float(psnr), float(temporal_mse)


def _analyze_record(
    record: ArtifactRecord,
    nfe_steps: Sequence[int] = NFE_STEPS,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
]:
    try:
        with safe_open(str(record.path), framework="pt", device="cpu") as handle:
            video_clean = handle.get_tensor("video_clean")
            tf_clean = handle.get_tensor("tf_clean")
            ground_truth = handle.get_tensor("ground_truth_future_uint8")
            history_tensor = handle.get_tensor("history_latent_frames")
            initial_video = handle.get_tensor("video_initial_state")
            initial_tf = handle.get_tensor("tf_initial_state")
            initial_tf_noise = handle.get_tensor("tf_initial_noise")
            seed_tensor = handle.get_tensor("evaluation_noise_seed")
            nfe_tensor = handle.get_tensor("evaluation_nfe_steps")
            source_codes_tensor = handle.get_tensor(
                "evaluation_condition_source_codes"
            )
            oracle_leakage_tensor = handle.get_tensor(
                "oracle_sources_are_leakage"
            )
            condition_tensor = handle.get_tensor("condition_on_tf")
            mode_tensor = handle.get_tensor("condition_mode_code")

            _validate_float_latent(video_clean, "video_clean")
            _validate_float_latent(tf_clean, "tf_clean")
            _validate_decoded(ground_truth, "ground_truth_future_uint8")
            _validate_float_latent(
                initial_video, "video_initial_state", tuple(video_clean.shape)
            )
            _validate_float_latent(
                initial_tf, "tf_initial_state", tuple(tf_clean.shape)
            )
            _validate_float_latent(
                initial_tf_noise, "tf_initial_noise", tuple(tf_clean.shape)
            )
            if initial_video.dtype != video_clean.dtype:
                raise ArtifactValidationError(
                    "video_initial_state dtype must match video_clean"
                )
            if initial_tf.dtype != tf_clean.dtype:
                raise ArtifactValidationError(
                    "tf_initial_state dtype must match tf_clean"
                )
            if initial_tf_noise.dtype != tf_clean.dtype:
                raise ArtifactValidationError(
                    "tf_initial_noise dtype must match tf_clean"
                )
            if (
                history_tensor.numel() != 1
                or history_tensor.dtype
                not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            ):
                raise ArtifactValidationError(
                    "history_latent_frames must be one integer scalar"
                )
            history_frames = int(history_tensor.item())
            if (
                history_frames < 0
                or history_frames >= video_clean.shape[2]
                or history_frames >= tf_clean.shape[2]
            ):
                raise ArtifactValidationError(
                    "history_latent_frames must leave a non-empty future in "
                    "both latents"
                )
            integer_dtypes = (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
            if seed_tensor.numel() != 1 or seed_tensor.dtype not in integer_dtypes:
                raise ArtifactValidationError(
                    "evaluation_noise_seed must be one integer scalar"
                )
            evaluation_noise_seed = int(seed_tensor.item())
            if evaluation_noise_seed < 0:
                raise ArtifactValidationError(
                    "evaluation_noise_seed must be non-negative"
                )
            if nfe_tensor.ndim != 1 or nfe_tensor.dtype not in integer_dtypes:
                raise ArtifactValidationError(
                    "evaluation_nfe_steps must be a one-dimensional integer tensor"
                )
            evaluation_nfe_steps = tuple(int(value) for value in nfe_tensor.tolist())
            if evaluation_nfe_steps != tuple(nfe_steps):
                raise ArtifactValidationError(
                    f"evaluation_nfe_steps must be {tuple(nfe_steps)}, "
                    f"got {evaluation_nfe_steps}"
                )
            if (
                source_codes_tensor.ndim != 1
                or source_codes_tensor.numel() < 1
                or source_codes_tensor.dtype not in integer_dtypes
            ):
                raise ArtifactValidationError(
                    "evaluation_condition_source_codes must be a non-empty "
                    "one-dimensional integer tensor"
                )
            source_codes = tuple(
                int(value) for value in source_codes_tensor.tolist()
            )
            if (
                len(set(source_codes)) != len(source_codes)
                or 0 not in source_codes
                or any(
                    code not in EVALUATION_CONDITION_SOURCE_NAMES
                    for code in source_codes
                )
            ):
                raise ArtifactValidationError(
                    "evaluation_condition_source_codes must be unique, include "
                    "autonomous code 0, and contain only supported codes"
                )
            declared_sources = tuple(
                EVALUATION_CONDITION_SOURCE_NAMES[code] for code in source_codes
            )
            if (
                oracle_leakage_tensor.numel() != 1
                or oracle_leakage_tensor.dtype not in integer_dtypes
                or int(oracle_leakage_tensor.item()) != 1
            ):
                raise ArtifactValidationError(
                    "oracle_sources_are_leakage must be the integer scalar 1"
                )
            _validate_source_tensor_inventory(
                set(record.tensor_schema), declared_sources, nfe_steps
            )
            if (
                condition_tensor.numel() != 1
                or condition_tensor.dtype not in integer_dtypes
                or mode_tensor.numel() != 1
                or mode_tensor.dtype not in integer_dtypes
            ):
                raise ArtifactValidationError(
                    "condition_on_tf and condition_mode_code must be integer scalars"
                )
            condition_on_tf = int(condition_tensor.item())
            condition_mode_code = int(mode_tensor.item())
            if condition_mode_code not in CONDITION_MODE_NAMES:
                raise ArtifactValidationError(
                    f"unknown condition_mode_code: {condition_mode_code}"
                )
            expected_enabled = int(condition_mode_code != 0)
            if condition_on_tf != expected_enabled:
                raise ArtifactValidationError(
                    "condition_on_tf is inconsistent with condition_mode_code"
                )
            if not torch.equal(
                initial_tf[:, :, :history_frames],
                tf_clean[:, :, :history_frames],
            ):
                raise ArtifactValidationError(
                    "tf_initial_state does not preserve the exact clean history"
                )
            if not torch.equal(
                initial_tf[:, :, history_frames:],
                initial_tf_noise[:, :, history_frames:],
            ):
                raise ArtifactValidationError(
                    "tf_initial_state future does not match tf_initial_noise"
                )

            source_metrics: dict[str, dict[str, dict[str, float]]] = {}
            for source in declared_sources:
                source_metrics[source] = {}
                infix = _source_infix(source)
                for nfe in nfe_steps:
                    video_key = f"video_final{infix}_nfe_{nfe}"
                    tf_key = f"tf_final{infix}_nfe_{nfe}"
                    decoded_key = f"decoded_future{infix}_nfe_{nfe}"
                    video_final = handle.get_tensor(video_key)
                    tf_final = handle.get_tensor(tf_key)
                    decoded = handle.get_tensor(decoded_key)
                    _validate_float_latent(
                        video_final, video_key, tuple(video_clean.shape)
                    )
                    _validate_float_latent(
                        tf_final, tf_key, tuple(tf_clean.shape)
                    )
                    if video_final.dtype != video_clean.dtype:
                        raise ArtifactValidationError(
                            f"{video_key} dtype must match video_clean"
                        )
                    if tf_final.dtype != tf_clean.dtype:
                        raise ArtifactValidationError(
                            f"{tf_key} dtype must match tf_clean"
                        )
                    _validate_decoded(
                        decoded, decoded_key, tuple(ground_truth.shape)
                    )

                    decoded_mse, decoded_psnr, temporal_mse = _decoded_metrics(
                        decoded, ground_truth
                    )
                    source_metrics[source][str(nfe)] = {
                        "video_future_nmse": _future_nmse(
                            video_final,
                            video_clean,
                            history_frames,
                            video_key,
                        ),
                        "tf_future_nmse": _future_nmse(
                            tf_final,
                            tf_clean,
                            history_frames,
                            tf_key,
                        ),
                        "decoded_mse_unit_range": decoded_mse,
                        "decoded_psnr_db": decoded_psnr,
                        "decoded_temporal_difference_mse_unit_range": (
                            temporal_mse
                        ),
                    }

            identity = {
                "history_latent_frames": history_frames,
                "video_clean_sha256": _hash_tensor(video_clean),
                "tf_clean_sha256": _hash_tensor(tf_clean),
                "ground_truth_future_uint8_sha256": _hash_tensor(ground_truth),
                "video_initial_state_sha256": _hash_tensor(initial_video),
                "tf_initial_state_sha256": _hash_tensor(initial_tf),
                "tf_initial_noise_sha256": _hash_tensor(initial_tf_noise),
                "evaluation_noise_seed": evaluation_noise_seed,
                "evaluation_nfe_steps": list(evaluation_nfe_steps),
                "evaluation_condition_source_codes": list(source_codes),
                "evaluation_condition_sources": list(declared_sources),
                "oracle_sources_are_leakage": True,
                "video_clean_shape": list(video_clean.shape),
                "tf_clean_shape": list(tf_clean.shape),
                "ground_truth_future_uint8_shape": list(ground_truth.shape),
            }
            intervention = {
                "condition_on_tf": bool(condition_on_tf),
                "condition_mode_code": condition_mode_code,
                "condition_mode": CONDITION_MODE_NAMES[condition_mode_code],
            }
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(
            f"failed while analyzing artifact {record.path}: {exc}"
        ) from exc
    return source_metrics, identity, intervention, declared_sources


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


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
        raise ArtifactValidationError(f"invalid values for summary {label!r}")
    rng = np.random.default_rng(_derived_seed(seed, label))
    indices = rng.integers(
        0, array.size, size=(bootstrap_samples, array.size), endpoint=False
    )
    bootstrap_means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
        },
    }


def _relative_effect_summary(
    left_values: Sequence[float],
    reference_values: Sequence[float],
    *,
    favorable_when: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Bootstrap a paired ratio-of-means relative effect.

    The effect is ``(mean(left) - mean(reference)) / mean(reference)``.
    Resampling always uses the same indices for both sides.
    """

    left = np.asarray(left_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    if (
        left.ndim != 1
        or reference.ndim != 1
        or left.size == 0
        or left.shape != reference.shape
        or not np.isfinite(left).all()
        or not np.isfinite(reference).all()
    ):
        raise ArtifactValidationError(
            f"invalid paired values for relative effect {label!r}"
        )
    if favorable_when not in {"lower", "higher"}:
        raise ArtifactValidationError(
            f"invalid metric direction for relative effect {label!r}"
        )

    left_mean = float(left.mean())
    reference_mean = float(reference.mean())
    common = {
        "n": int(left.size),
        "definition": (
            "(mean(left) - mean(reference)) / mean(reference), with paired "
            "bootstrap resampling"
        ),
        "left_mean": left_mean,
        "reference_mean": reference_mean,
        "favorable_when": (
            "relative_effect < 0"
            if favorable_when == "lower"
            else "relative_effect > 0"
        ),
        "equivalence": {
            "margin_fraction": EQUIVALENCE_MARGIN,
            "margin_percent": 100.0 * EQUIVALENCE_MARGIN,
            "ci_entirely_within_margin": False,
        },
    }
    if reference_mean <= 0:
        return {
            **common,
            "defined": False,
            "undefined_reason": "reference mean must be strictly positive",
            "relative_effect": None,
            "relative_effect_percent": None,
            "bootstrap_ci": None,
            "ci_favors_left": False,
            "ci_favors_reference": False,
            "material_improvement_at_least_3pct": False,
            "material_harm_at_least_3pct": False,
        }

    effect = (left_mean - reference_mean) / reference_mean
    rng = np.random.default_rng(_derived_seed(seed, label))
    indices = rng.integers(
        0, left.size, size=(bootstrap_samples, left.size), endpoint=False
    )
    left_means = left[indices].mean(axis=1)
    reference_means = reference[indices].mean(axis=1)
    if np.any(reference_means <= 0):
        return {
            **common,
            "defined": False,
            "undefined_reason": (
                "at least one paired bootstrap resample has non-positive "
                "reference mean"
            ),
            "relative_effect": float(effect),
            "relative_effect_percent": float(100.0 * effect),
            "bootstrap_ci": None,
            "ci_favors_left": False,
            "ci_favors_reference": False,
            "material_improvement_at_least_3pct": False,
            "material_harm_at_least_3pct": False,
        }
    bootstrap_effects = (
        left_means - reference_means
    ) / reference_means
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_effects, [tail, 1.0 - tail])
    if favorable_when == "lower":
        ci_favors_left = high < 0
        ci_favors_reference = low > 0
        material_improvement = effect <= -PROMISING_RELATIVE_IMPROVEMENT
        material_harm = effect >= PROMISING_RELATIVE_IMPROVEMENT
    else:
        ci_favors_left = low > 0
        ci_favors_reference = high < 0
        material_improvement = effect >= PROMISING_RELATIVE_IMPROVEMENT
        material_harm = effect <= -PROMISING_RELATIVE_IMPROVEMENT
    equivalent = low >= -EQUIVALENCE_MARGIN and high <= EQUIVALENCE_MARGIN
    return {
        **common,
        "defined": True,
        "undefined_reason": None,
        "relative_effect": float(effect),
        "relative_effect_percent": float(100.0 * effect),
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
            "low_percent": float(100.0 * low),
            "high_percent": float(100.0 * high),
        },
        "ci_favors_left": bool(ci_favors_left),
        "ci_favors_reference": bool(ci_favors_reference),
        "material_improvement_at_least_3pct": bool(material_improvement),
        "material_harm_at_least_3pct": bool(material_harm),
        "equivalence": {
            "margin_fraction": EQUIVALENCE_MARGIN,
            "margin_percent": 100.0 * EQUIVALENCE_MARGIN,
            "ci_entirely_within_margin": bool(equivalent),
        },
    }


def _effect_value(
    relative_nfe: Mapping[str, Any], nfe: int, metric: str
) -> dict[str, Any]:
    value = relative_nfe.get(str(nfe), {}).get(metric)
    if not isinstance(value, dict):
        raise ArtifactValidationError(
            f"relative effect is missing for NFE={nfe}, metric={metric}"
        )
    return value


def _effect_at_most(value: Mapping[str, Any], threshold: float) -> bool:
    effect = value.get("relative_effect")
    return bool(value.get("defined")) and isinstance(effect, (int, float)) and (
        effect <= threshold
    )


def _effect_at_least(value: Mapping[str, Any], threshold: float) -> bool:
    effect = value.get("relative_effect")
    return bool(value.get("defined")) and isinstance(effect, (int, float)) and (
        effect >= threshold
    )


def _equivalent(value: Mapping[str, Any]) -> bool:
    equivalence = value.get("equivalence")
    return bool(
        value.get("defined")
        and isinstance(equivalence, dict)
        and equivalence.get("ci_entirely_within_margin") is True
    )


def _preregistered_decision(
    relative_nfe: Mapping[str, Any],
    *,
    scale: str,
    oracle_relative_nfe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    temporal_metric = "decoded_temporal_difference_mse_unit_range"
    video_metric = "video_future_nmse"
    temporal_4 = _effect_value(relative_nfe, 4, temporal_metric)
    video_4 = _effect_value(relative_nfe, 4, video_metric)
    temporal_8 = _effect_value(relative_nfe, 8, temporal_metric)
    video_8 = _effect_value(relative_nfe, 8, video_metric)

    temporal_4_improves_3pct = _effect_at_most(
        temporal_4, -PROMISING_RELATIVE_IMPROVEMENT
    )
    video_4_no_regression = _effect_at_most(video_4, 0.0)
    temporal_8_direction_agrees = _effect_at_most(temporal_8, 0.0)
    # Strictly negative is required for direction agreement; an exact zero is
    # not evidence that the favorable direction persisted.
    if temporal_8.get("relative_effect") == 0:
        temporal_8_direction_agrees = False
    literal_metric_gate = (
        temporal_4_improves_3pct
        and video_4_no_regression
        and temporal_8_direction_agrees
    )
    bootstrap_supported_gate = bool(
        literal_metric_gate
        and temporal_4.get("ci_favors_left")
        and video_4.get("ci_favors_left")
        and temporal_8.get("ci_favors_left")
    )

    symmetric_harm_signal = bool(
        _effect_at_least(temporal_4, PROMISING_RELATIVE_IMPROVEMENT)
        and _effect_at_least(video_4, 0.0)
        and _effect_at_least(temporal_8, 0.0)
        and temporal_4.get("ci_favors_reference")
    )
    autonomous_equivalence = all(
        _equivalent(value)
        for value in (temporal_4, video_4, temporal_8, video_8)
    )

    oracle_equivalence = None
    if oracle_relative_nfe is not None:
        oracle_equivalence = all(
            _equivalent(
                _effect_value(oracle_relative_nfe, nfe, metric)
            )
            for nfe in (4, 8)
            for metric in (temporal_metric, video_metric)
        )
    scoped_negative_metric_component = bool(
        scale == "s010"
        and autonomous_equivalence
        and oracle_equivalence is True
    )
    if literal_metric_gate:
        metric_classification = "promising_metric_gate"
    elif symmetric_harm_signal:
        metric_classification = "harm_metric_signal"
    elif scoped_negative_metric_component:
        metric_classification = "scoped_null_metric_equivalence"
    else:
        metric_classification = "inconclusive"

    return {
        "schema_version": 1,
        "scope": (
            "metric-only screen decision; exposure qualification and training-"
            "seed replication are external requirements"
        ),
        "scale": scale,
        "primary_nfe": 4,
        "confirmation_nfe": 8,
        "primary_temporal_metric": temporal_metric,
        "primary_video_metric": video_metric,
        "thresholds": {
            "temporal_relative_improvement_fraction": (
                PROMISING_RELATIVE_IMPROVEMENT
            ),
            "temporal_relative_improvement_percent": (
                100.0 * PROMISING_RELATIVE_IMPROVEMENT
            ),
            "video_nmse_no_regression_fraction": 0.0,
            "equivalence_margin_fraction": EQUIVALENCE_MARGIN,
            "equivalence_margin_percent": 100.0 * EQUIVALENCE_MARGIN,
        },
        "criteria": {
            "temporal_4nfe_at_least_3pct_better": (
                temporal_4_improves_3pct
            ),
            "video_nmse_4nfe_no_regression": video_4_no_regression,
            "temporal_8nfe_direction_agrees": temporal_8_direction_agrees,
            "literal_preregistered_metric_gate_pass": literal_metric_gate,
            "bootstrap_sign_supported_metric_gate_pass": (
                bootstrap_supported_gate
            ),
        },
        "harm_diagnostic": {
            "symmetric_material_harm_signal": symmetric_harm_signal,
            "preregistered": False,
        },
        "equivalence_diagnostic": {
            "autonomous_primary_4nfe_and_8nfe_equivalent_within_2pct": (
                autonomous_equivalence
            ),
            "oracle_primary_4nfe_and_8nfe_equivalent_within_2pct": (
                oracle_equivalence
            ),
            "scoped_negative_metric_component_pass": (
                scoped_negative_metric_component
            ),
            "requires_verified_exposure_through_s010": True,
        },
        "external_requirements": {
            "exposure_qualified": None,
            "exposure_telemetry_required": True,
            "three_fresh_training_seeds_required_after_promising_screen": True,
        },
        "metric_classification": metric_classification,
    }


def _parse_arm_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not ARM_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "arm name must use letters, digits, dot, underscore, or dash"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("arm path must not be empty")
    return name, Path(raw_path)


def _prepare_output_path(output: str | Path, roots: Sequence[Path]) -> Path:
    raw = Path(output).expanduser()
    if raw.suffix.lower() != ".json":
        raise ArtifactValidationError("output path must end in .json")
    parent = _canonical_directory(raw.parent, "output parent")
    path = parent / raw.name
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ArtifactValidationError(f"output path already exists: {path}")
    for root in roots:
        if path == root or path.is_relative_to(root):
            raise ArtifactValidationError(
                f"output must be outside every read-only arm root: {path}"
            )
    return path


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def analyze(
    arms: Mapping[str, str | Path],
    *,
    baseline: str,
    output: str | Path,
    expected_nfe_steps: Sequence[int] = NFE_STEPS,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_726,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Validate, compare, and exclusively write one analysis JSON document."""

    if not arms:
        raise ArtifactValidationError("at least one arm is required")
    if baseline not in arms:
        raise ArtifactValidationError(f"baseline arm {baseline!r} was not supplied")
    if bootstrap_samples < 100:
        raise ArtifactValidationError("bootstrap_samples must be at least 100")
    if not 0.5 < confidence < 1.0:
        raise ArtifactValidationError("confidence must lie strictly between 0.5 and 1")
    nfe_steps = tuple(expected_nfe_steps)
    if (
        not nfe_steps
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in nfe_steps
        )
        or tuple(sorted(set(nfe_steps))) != nfe_steps
    ):
        raise ArtifactValidationError(
            "expected_nfe_steps must be unique positive integers in "
            "strictly increasing order"
        )

    canonical_roots: dict[str, Path] = {}
    for name, raw_root in arms.items():
        if not ARM_NAME_RE.fullmatch(name):
            raise ArtifactValidationError(f"invalid arm name: {name!r}")
        canonical_roots[name] = _canonical_directory(raw_root, f"arm {name!r} root")
    if len(set(canonical_roots.values())) != len(canonical_roots):
        raise ArtifactValidationError("arm roots must be distinct")
    output_path = _prepare_output_path(output, list(canonical_roots.values()))

    inventories = {
        name: _discover_arm(name, root, nfe_steps)
        for name, root in sorted(canonical_roots.items())
    }
    baseline_pairs = set(inventories[baseline])
    for name, records in inventories.items():
        if set(records) != baseline_pairs:
            missing = sorted(baseline_pairs - set(records))
            extra = sorted(set(records) - baseline_pairs)
            raise ArtifactValidationError(
                f"arm {name!r} dataset/rank pairing mismatch; "
                f"missing={missing}, extra={extra}"
            )

    all_records = [
        record for records in inventories.values() for record in records.values()
    ]
    iterations = {record.iteration for record in all_records}
    if len(iterations) != 1:
        raise ArtifactValidationError(
            f"arms have mismatched visualization iterations: {sorted(iterations)}"
        )

    analyzed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for arm, records in inventories.items():
        analyzed[arm] = {}
        for pair_key, record in records.items():
            (
                source_metrics,
                identity,
                intervention,
                condition_sources,
            ) = _analyze_record(record, nfe_steps)
            analyzed[arm][pair_key] = {
                # Retain the original autonomous field for cross-arm analysis
                # and existing consumers.
                "metrics": source_metrics["autonomous"],
                "condition_source_metrics": source_metrics,
                "condition_sources": list(condition_sources),
                "identity": identity,
                "intervention": intervention,
            }

    ordered_pairs = sorted(baseline_pairs)
    baseline_schema = {
        pair: inventories[baseline][pair].tensor_schema for pair in ordered_pairs
    }
    for arm, records in inventories.items():
        for pair in ordered_pairs:
            if records[pair].tensor_schema != baseline_schema[pair]:
                raise ArtifactValidationError(
                    f"tensor-key/shape/dtype mismatch for arm {arm!r}, pair {pair}"
                )
            expected_identity = analyzed[baseline][pair]["identity"]
            actual_identity = analyzed[arm][pair]["identity"]
            if actual_identity != expected_identity:
                raise ArtifactValidationError(
                    "paired provenance mismatch (clean/history/seed/initial "
                    f"state/source declarations) for arm {arm!r}, pair {pair}"
                )
        interventions = {
            (
                result["intervention"]["condition_on_tf"],
                result["intervention"]["condition_mode_code"],
                result["intervention"]["condition_mode"],
            )
            for result in analyzed[arm].values()
        }
        if len(interventions) != 1:
            raise ArtifactValidationError(
                f"arm {arm!r} mixes condition interventions: "
                f"{sorted(interventions)}"
            )
        source_declarations = {
            tuple(result["condition_sources"])
            for result in analyzed[arm].values()
        }
        if len(source_declarations) != 1:
            raise ArtifactValidationError(
                f"arm {arm!r} mixes evaluation condition sources: "
                f"{sorted(source_declarations)}"
            )

    within_arm_condition_sources: dict[str, Any] = {}
    within_arm_source_deltas: dict[str, Any] = {}
    for arm in sorted(inventories):
        declared_sources = analyzed[arm][ordered_pairs[0]][
            "condition_sources"
        ]
        source_aggregates: dict[str, Any] = {}
        for source in declared_sources:
            source_aggregates[source] = {
                "oracle_leakage": source in ORACLE_SOURCES,
                "nfe": {},
            }
            for nfe in nfe_steps:
                nfe_key = str(nfe)
                source_aggregates[source]["nfe"][nfe_key] = {}
                for metric in METRIC_DIRECTIONS:
                    values = [
                        analyzed[arm][pair]["condition_source_metrics"][source][
                            nfe_key
                        ][metric]
                        for pair in ordered_pairs
                    ]
                    source_aggregates[source]["nfe"][nfe_key][metric] = _summary(
                        values,
                        bootstrap_samples=bootstrap_samples,
                        confidence=confidence,
                        seed=bootstrap_seed,
                        label=(
                            f"within-arm:{arm}:source:{source}:nfe:{nfe}:"
                            f"metric:{metric}"
                        ),
                    )
        within_arm_condition_sources[arm] = {
            "declared_sources": declared_sources,
            "sources": source_aggregates,
        }

        source_comparisons: dict[str, Any] = {}
        for left, right in (
            ("autonomous", "autonomous_shuffled"),
            ("autonomous", "autonomous_legacy"),
            ("autonomous", "off"),
            ("oracle_matched", "off"),
            ("oracle_matched", "oracle_shuffled"),
        ):
            if left not in declared_sources or right not in declared_sources:
                continue
            comparison_name = f"{left}_minus_{right}"
            oracle_leakage = left in ORACLE_SOURCES or right in ORACLE_SOURCES
            comparison: dict[str, Any] = {
                "definition": f"{left} minus {right}",
                "oracle_leakage": oracle_leakage,
                "deployable_evidence": not oracle_leakage,
                "paired_by": "identical arm, dataset/rank, and initial states",
                "nfe": {},
                "relative_nfe": {},
            }
            for nfe in nfe_steps:
                nfe_key = str(nfe)
                comparison["nfe"][nfe_key] = {}
                comparison["relative_nfe"][nfe_key] = {}
                for metric, favorable_when in METRIC_DIRECTIONS.items():
                    left_values = [
                        analyzed[arm][pair]["condition_source_metrics"][left][
                            nfe_key
                        ][metric]
                        for pair in ordered_pairs
                    ]
                    right_values = [
                        analyzed[arm][pair]["condition_source_metrics"][right][
                            nfe_key
                        ][metric]
                        for pair in ordered_pairs
                    ]
                    deltas = [
                        left_value - right_value
                        for left_value, right_value in zip(
                            left_values, right_values
                        )
                    ]
                    summary = _summary(
                        deltas,
                        bootstrap_samples=bootstrap_samples,
                        confidence=confidence,
                        seed=bootstrap_seed,
                        label=(
                            f"within-arm-delta:{arm}:{comparison_name}:"
                            f"nfe:{nfe}:metric:{metric}"
                        ),
                    )
                    favorable_count = sum(
                        delta < 0
                        if favorable_when == "lower"
                        else delta > 0
                        for delta in deltas
                    )
                    summary.update(
                        {
                            "favorable_when": (
                                "delta < 0"
                                if favorable_when == "lower"
                                else "delta > 0"
                            ),
                            "favorable_fraction": (
                                favorable_count / len(deltas)
                            ),
                        }
                    )
                    comparison["nfe"][nfe_key][metric] = summary
                    comparison["relative_nfe"][nfe_key][metric] = (
                        _relative_effect_summary(
                            left_values,
                            right_values,
                            favorable_when=favorable_when,
                            bootstrap_samples=bootstrap_samples,
                            confidence=confidence,
                            seed=bootstrap_seed,
                            label=(
                                f"within-arm-relative:{arm}:"
                                f"{comparison_name}:nfe:{nfe}:"
                                f"metric:{metric}"
                            ),
                        )
                    )
            source_comparisons[comparison_name] = comparison
        within_arm_source_deltas[arm] = source_comparisons

    arm_aggregates: dict[str, Any] = {}
    for arm in sorted(inventories):
        arm_aggregates[arm] = {}
        for nfe in nfe_steps:
            nfe_key = str(nfe)
            arm_aggregates[arm][nfe_key] = {}
            for metric in METRIC_DIRECTIONS:
                values = [
                    analyzed[arm][pair]["metrics"][nfe_key][metric]
                    for pair in ordered_pairs
                ]
                arm_aggregates[arm][nfe_key][metric] = _summary(
                    values,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=bootstrap_seed,
                    label=f"arm:{arm}:nfe:{nfe}:metric:{metric}",
                )

    paired_deltas: dict[str, Any] = {}
    for arm in sorted(name for name in inventories if name != baseline):
        comparison: dict[str, Any] = {
            "definition": f"{arm} minus {baseline}",
            "arm": arm,
            "baseline": baseline,
            "condition_source": "autonomous",
            "oracle_leakage": False,
            "nfe": {},
            "relative_nfe": {},
        }
        for nfe in nfe_steps:
            nfe_key = str(nfe)
            comparison["nfe"][nfe_key] = {}
            comparison["relative_nfe"][nfe_key] = {}
            for metric, favorable_when in METRIC_DIRECTIONS.items():
                left_values = [
                    analyzed[arm][pair]["metrics"][nfe_key][metric]
                    for pair in ordered_pairs
                ]
                reference_values = [
                    analyzed[baseline][pair]["metrics"][nfe_key][metric]
                    for pair in ordered_pairs
                ]
                deltas = [
                    left_value - reference_value
                    for left_value, reference_value in zip(
                        left_values, reference_values
                    )
                ]
                summary = _summary(
                    deltas,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=bootstrap_seed,
                    label=(
                        f"delta:{arm}-minus-{baseline}:nfe:{nfe}:metric:{metric}"
                    ),
                )
                favorable_count = sum(
                    delta < 0 if favorable_when == "lower" else delta > 0
                    for delta in deltas
                )
                summary.update(
                    {
                        "favorable_when": (
                            "delta < 0" if favorable_when == "lower" else "delta > 0"
                        ),
                        "favorable_fraction": favorable_count / len(deltas),
                    }
                )
                comparison["nfe"][nfe_key][metric] = summary
                comparison["relative_nfe"][nfe_key][metric] = (
                    _relative_effect_summary(
                        left_values,
                        reference_values,
                        favorable_when=favorable_when,
                        bootstrap_samples=bootstrap_samples,
                        confidence=confidence,
                        seed=bootstrap_seed,
                        label=(
                            f"relative:{arm}-minus-{baseline}:"
                            f"nfe:{nfe}:metric:{metric}"
                        ),
                    )
                )
        paired_deltas[arm] = comparison

    scaled_arm_inventory: dict[str, dict[str, str]] = {}
    for arm in sorted(inventories):
        match = SCALED_ARM_NAME_RE.fullmatch(arm)
        if match is None:
            continue
        named_mode, digits = match.groups()
        actual_mode = analyzed[arm][ordered_pairs[0]]["intervention"][
            "condition_mode"
        ]
        if actual_mode != named_mode:
            raise ArtifactValidationError(
                f"scaled arm name {arm!r} declares mode {named_mode!r}, "
                f"but artifact intervention is {actual_mode!r}"
            )
        scale = f"s{digits}"
        scaled_arm_inventory.setdefault(scale, {})[named_mode] = arm

    direct_same_scale_comparisons: dict[str, Any] = {}
    same_scale_pair_inventory: dict[str, Any] = {}
    for scale, modes in sorted(scaled_arm_inventory.items()):
        missing_modes = sorted({"matched", "shuffled"} - set(modes))
        same_scale_pair_inventory[scale] = {
            "matched_arm": modes.get("matched"),
            "shuffled_arm": modes.get("shuffled"),
            "complete": not missing_modes,
            "missing_modes": missing_modes,
        }
        if missing_modes:
            continue
        matched_arm = modes["matched"]
        shuffled_arm = modes["shuffled"]
        declared_source_sets = {
            tuple(analyzed[arm][ordered_pairs[0]]["condition_sources"])
            for arm in (matched_arm, shuffled_arm)
        }
        if len(declared_source_sets) != 1:
            raise ArtifactValidationError(
                f"same-scale arms at {scale} declare different condition sources"
            )
        declared_sources = next(iter(declared_source_sets))

        def build_source_comparison(source: str) -> dict[str, Any]:
            source_comparison: dict[str, Any] = {
                "definition": (
                    f"{matched_arm} minus {shuffled_arm}, sampled with "
                    f"condition source {source}"
                ),
                "condition_source": source,
                "oracle_leakage": source in ORACLE_SOURCES,
                "paired_by": "identical dataset/rank and initial states",
                "nfe": {},
                "relative_nfe": {},
            }
            for nfe in nfe_steps:
                nfe_key = str(nfe)
                source_comparison["nfe"][nfe_key] = {}
                source_comparison["relative_nfe"][nfe_key] = {}
                for metric, favorable_when in METRIC_DIRECTIONS.items():
                    matched_values = [
                        analyzed[matched_arm][pair][
                            "condition_source_metrics"
                        ][source][nfe_key][metric]
                        for pair in ordered_pairs
                    ]
                    shuffled_values = [
                        analyzed[shuffled_arm][pair][
                            "condition_source_metrics"
                        ][source][nfe_key][metric]
                        for pair in ordered_pairs
                    ]
                    deltas = [
                        matched_value - shuffled_value
                        for matched_value, shuffled_value in zip(
                            matched_values, shuffled_values
                        )
                    ]
                    summary = _summary(
                        deltas,
                        bootstrap_samples=bootstrap_samples,
                        confidence=confidence,
                        seed=bootstrap_seed,
                        label=(
                            f"same-scale-delta:{scale}:{source}:"
                            f"{matched_arm}-minus-{shuffled_arm}:"
                            f"nfe:{nfe}:metric:{metric}"
                        ),
                    )
                    favorable_count = sum(
                        delta < 0
                        if favorable_when == "lower"
                        else delta > 0
                        for delta in deltas
                    )
                    summary.update(
                        {
                            "favorable_when": (
                                "delta < 0"
                                if favorable_when == "lower"
                                else "delta > 0"
                            ),
                            "favorable_fraction": (
                                favorable_count / len(deltas)
                            ),
                        }
                    )
                    source_comparison["nfe"][nfe_key][metric] = summary
                    source_comparison["relative_nfe"][nfe_key][metric] = (
                        _relative_effect_summary(
                            matched_values,
                            shuffled_values,
                            favorable_when=favorable_when,
                            bootstrap_samples=bootstrap_samples,
                            confidence=confidence,
                            seed=bootstrap_seed,
                            label=(
                                f"same-scale-relative:{scale}:{source}:"
                                f"{matched_arm}-minus-{shuffled_arm}:"
                                f"nfe:{nfe}:metric:{metric}"
                            ),
                        )
                    )
            return source_comparison

        source_comparisons = {
            source: build_source_comparison(source)
            for source in ("autonomous", "off")
            if source in declared_sources
        }
        if "autonomous" not in source_comparisons:
            raise ArtifactValidationError(
                f"same-scale arms at {scale} lack autonomous artifacts"
            )
        autonomous = source_comparisons["autonomous"]
        comparison: dict[str, Any] = {
            "definition": f"{matched_arm} minus {shuffled_arm}",
            "matched_arm": matched_arm,
            "shuffled_arm": shuffled_arm,
            "scale": scale,
            "condition_source": "autonomous",
            "oracle_leakage": False,
            "paired_by": "identical dataset/rank and initial states",
            # Retain these aliases for existing decision/report consumers.
            "nfe": autonomous["nfe"],
            "relative_nfe": autonomous["relative_nfe"],
            "condition_source_comparisons": source_comparisons,
            "autonomous_sampler_requirement": (
                "the shuffled sampler must preserve each paired unit's local "
                "TF corruption noise and derange only its denoised content"
            ),
        }
        if "off" in source_comparisons:
            source_comparisons["off"].update(
                {
                    "training_alignment_diagnostic": True,
                    "direct_inference_conditioning_evidence": False,
                    "interpretation": (
                        "Both trained models are sampled with TF injection "
                        "disabled; a difference can reflect whether aligned "
                        "TF exposure changed the learned video weights, but "
                        "cannot establish an inference-time TF benefit."
                    ),
                }
            )

        oracle_comparison = within_arm_source_deltas.get(
            matched_arm, {}
        ).get("oracle_matched_minus_oracle_shuffled")
        oracle_relative_nfe = (
            oracle_comparison.get("relative_nfe")
            if isinstance(oracle_comparison, dict)
            else None
        )
        comparison["oracle_mechanism_diagnostic"] = {
            "available": oracle_relative_nfe is not None,
            "oracle_leakage": True,
            "deployable_evidence": False,
            "source": (
                f"aggregate.within_arm_source_deltas.{matched_arm}."
                "oracle_matched_minus_oracle_shuffled"
            ),
        }
        comparison["preregistered_decision"] = _preregistered_decision(
            comparison["relative_nfe"],
            scale=scale,
            oracle_relative_nfe=oracle_relative_nfe,
        )
        direct_same_scale_comparisons[scale] = comparison

    same_scale_decisions = {
        scale: comparison["preregistered_decision"]
        for scale, comparison in direct_same_scale_comparisons.items()
    }
    if any(
        decision["criteria"]["literal_preregistered_metric_gate_pass"]
        for decision in same_scale_decisions.values()
    ):
        overall_metric_classification = (
            "promising_metric_gate_at_one_or_more_scales"
        )
    elif any(
        decision["harm_diagnostic"]["symmetric_material_harm_signal"]
        for decision in same_scale_decisions.values()
    ):
        overall_metric_classification = (
            "harm_metric_signal_at_one_or_more_scales"
        )
    elif any(
        decision["equivalence_diagnostic"][
            "scoped_negative_metric_component_pass"
        ]
        for decision in same_scale_decisions.values()
    ):
        overall_metric_classification = (
            "scoped_null_metric_equivalence_requires_exposure"
        )
    else:
        overall_metric_classification = "inconclusive"
    preregistered_decisions = {
        "schema_version": 1,
        "comparison": (
            "autonomous matched minus autonomous shuffled at identical "
            "fixed state scale"
        ),
        "same_scale": same_scale_decisions,
        "overall_metric_classification": overall_metric_classification,
        "exposure_qualification_is_external": True,
        "oracle_results_are_leakage_only": True,
    }

    per_pair = []
    for dataset, rank in ordered_pairs:
        per_pair.append(
            {
                "dataset": dataset,
                "global_rank": rank,
                "arms": {
                    arm: {
                        "artifact_path": str(inventories[arm][(dataset, rank)].path),
                        "artifact_sha256": inventories[arm][
                            (dataset, rank)
                        ].sha256,
                        "intervention": analyzed[arm][(dataset, rank)][
                            "intervention"
                        ],
                        "metrics": analyzed[arm][(dataset, rank)]["metrics"],
                        "condition_source_metrics": {
                            source: {
                                "oracle_leakage": source in ORACLE_SOURCES,
                                "metrics": metrics,
                            }
                            for source, metrics in analyzed[arm][
                                (dataset, rank)
                            ]["condition_source_metrics"].items()
                        },
                    }
                    for arm in sorted(inventories)
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dual_video_diffusion_matched_nfe_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sigma_convention": SIGMA_CONVENTION,
        "nfe_steps": list(nfe_steps),
        "baseline_arm": baseline,
        "oracle_diagnostics": {
            "oracle_sources_are_leakage": True,
            "deployable_evidence": False,
            "interpretation": (
                "oracle_matched and oracle_shuffled consume hidden-future TF "
                "content and are leakage-only mechanism diagnostics; they "
                "cannot establish causal or deployable generation quality"
            ),
        },
        "condition_source_definitions": {
            "autonomous": (
                "deployable sampler using independently denoised TF; under the "
                "stage-faithful cascade its TF content is injected only after "
                "the TF phase reaches sigma=0"
            ),
            "autonomous_shuffled": (
                "same-checkpoint stage-faithful causal control: generate TF "
                "with identical local initial corruption, then roll only its "
                "future content globally once before the video phase"
            ),
            "autonomous_legacy": (
                "same-checkpoint historical control with TF-content injection "
                "enabled on both TF-only and video-phase calls"
            ),
            "off": "same trained model sampled with TF-to-video injection disabled",
            "oracle_matched": (
                "LEAKAGE: noisy condition built from the matched clean hidden-future TF"
            ),
            "oracle_shuffled": (
                "LEAKAGE control: noisy condition built from another sample's "
                "clean hidden-future TF"
            ),
        },
        "metric_definitions": {
            "video_future_nmse": (
                "sum((video_final-video_clean)^2) / sum(video_clean^2), "
                "latent time axis >= history_latent_frames"
            ),
            "tf_future_nmse": (
                "sum((tf_final-tf_clean)^2) / sum(tf_clean^2), "
                "latent time axis >= history_latent_frames"
            ),
            "decoded_mse_unit_range": (
                "mean squared RGB error after mapping uint8 values to [0,1]"
            ),
            "decoded_psnr_db": (
                "10*log10(1/max(decoded_mse_unit_range,1e-12))"
            ),
            "decoded_temporal_difference_mse_unit_range": (
                "MSE between consecutive-frame RGB differences in [0,1]"
            ),
            "paired_delta": (
                "left/comparison metric minus right/reference metric, as named "
                "in each comparison definition"
            ),
        },
        "bootstrap": {
            "method": (
                "paired nonparametric percentile bootstrap over dataset/rank units"
            ),
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "per_statistic_seed_derivation": "sha256(seed + NUL + statistic label)",
        },
        "provenance": {
            "iteration": next(iter(iterations)),
            "paired_unit_count": len(ordered_pairs),
            "paired_units": [
                {"dataset": dataset, "global_rank": rank}
                for dataset, rank in ordered_pairs
            ],
            "arms": {
                arm: {
                    "root": str(canonical_roots[arm]),
                    "artifact_count": len(inventories[arm]),
                    "intervention": next(
                        iter(analyzed[arm].values())
                    )["intervention"],
                    "evaluation_condition_sources": next(
                        iter(analyzed[arm].values())
                    )["condition_sources"],
                    "oracle_sources_are_leakage": True,
                    "artifacts": [
                        {
                            "dataset": record.dataset,
                            "global_rank": record.global_rank,
                            "path": str(record.path),
                            "sidecar_path": str(record.sidecar),
                            "sha256": record.sha256,
                        }
                        for _, record in sorted(inventories[arm].items())
                    ],
                }
                for arm in sorted(inventories)
            },
        },
        "per_paired_unit": per_pair,
        "aggregate": {
            "cross_arm_condition_source": "autonomous",
            "arms": arm_aggregates,
            "paired_deltas": paired_deltas,
            "same_scale_pair_inventory": same_scale_pair_inventory,
            "direct_same_scale_matched_vs_shuffled": (
                direct_same_scale_comparisons
            ),
            "preregistered_decisions": preregistered_decisions,
            "within_arm_condition_sources": within_arm_condition_sources,
            "within_arm_source_deltas": within_arm_source_deltas,
        },
    }
    _exclusive_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm_spec,
        required=True,
        metavar="NAME=PATH",
        help="matched arm name and read-only visualization-iteration root (repeat)",
    )
    parser.add_argument("--baseline", required=True, help="name of the baseline arm")
    parser.add_argument(
        "--output",
        required=True,
        help="fresh .json path outside every arm root; parent must already exist",
    )
    parser.add_argument(
        "--nfe-steps",
        type=int,
        nargs="+",
        default=list(NFE_STEPS),
        metavar="N",
        help=(
            "exact increasing NFE vector embedded in every artifact "
            f"(default: {' '.join(str(value) for value in NFE_STEPS)})"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_726)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms: dict[str, Path] = {}
    for name, path in args.arm:
        if name in arms:
            raise ArtifactValidationError(f"duplicate arm name: {name!r}")
        arms[name] = path
    analyze(
        arms,
        baseline=args.baseline,
        output=args.output,
        expected_nfe_steps=args.nfe_steps,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
