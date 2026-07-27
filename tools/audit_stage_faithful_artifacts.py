#!/usr/bin/env python3
"""Bitwise audit a stage-faithful cascade evaluation against its legacy run.

The command is deliberately read-only with respect to both input roots.  It
creates exactly one caller-selected JSON report, using exclusive-create
semantics, outside the evaluated roots.

LACWM clock convention: ``sigma=1`` is noise and ``sigma=0`` is clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open


WORLD_SIZE = 8
ARTIFACT_ITERATION = 199
NFE_STEPS = (2, 4, 8)
NEW_SOURCE_CODES = (0, 4, 5, 1)
LEGACY_SOURCE_CODES = (0, 1, 2, 3)
SIGMA_CONVENTION = "1=noise,0=clean"
SOURCE_NAMES = {
    0: "autonomous",
    1: "off",
    2: "oracle_matched",
    3: "oracle_shuffled",
    4: "autonomous_shuffled",
    5: "autonomous_legacy",
}
IDENTITY_TENSORS = (
    "video_clean",
    "tf_clean",
    "video_initial_state",
    "tf_initial_state",
    "tf_initial_noise",
    "history_latent_frames",
    "evaluation_noise_seed",
    "ground_truth_future_uint8",
)
FINAL_PREFIXES = ("video_final", "tf_final", "decoded_future")
FORBIDDEN_TRAINING_BASENAMES = {
    "_never_write_snapshot.pt",
    "snapshot.pt",
    "training_complete.json",
}
ARTIFACT_RE = re.compile(r"^latent_trajectory_rank_([0-9]+)[.]safetensors$")
FINAL_TENSOR_RE = re.compile(
    r"^(video_final|tf_final|decoded_future)"
    r"(?:_(off|oracle_matched|oracle_shuffled|autonomous_shuffled|"
    r"autonomous_legacy))?_nfe_([0-9]+)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
SAFETENSORS_TO_TORCH_DTYPE = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "I16": "torch.int16",
    "U16": "torch.uint16",
    "I32": "torch.int32",
    "U32": "torch.uint32",
    "I64": "torch.int64",
    "U64": "torch.uint64",
    "F8_E4M3": "torch.float8_e4m3fn",
    "F8_E5M2": "torch.float8_e5m2",
    "BF16": "torch.bfloat16",
    "F16": "torch.float16",
    "F32": "torch.float32",
    "F64": "torch.float64",
    "C64": "torch.complex64",
    "C128": "torch.complex128",
}


class StageArtifactAuditError(RuntimeError):
    """Raised when provenance or a required bitwise invariant fails."""


@dataclass(frozen=True)
class ArtifactRecord:
    root: Path
    scope: Path
    path: Path
    sidecar: Path
    rank: int
    dataset: str
    iteration: int
    artifact_sha256: str
    sidecar_sha256: str
    keys: frozenset[str]


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
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


def _tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8),
            right.detach().cpu().contiguous().view(torch.uint8),
        )
    )


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StageArtifactAuditError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise StageArtifactAuditError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StageArtifactAuditError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise StageArtifactAuditError(
            f"{label} must be a non-empty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _ensure_no_symlink_components(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageArtifactAuditError(f"path escapes input root: {path}") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise StageArtifactAuditError(
                    f"artifact path contains a symlink component: {current}"
                )
        except FileNotFoundError as exc:
            raise StageArtifactAuditError(
                f"artifact path disappeared during audit: {current}"
            ) from exc


def _load_json_strict(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StageArtifactAuditError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageArtifactAuditError(f"invalid JSON sidecar {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageArtifactAuditError(
            f"JSON sidecar must contain an object: {path}"
        )
    return payload


def _select_artifact_scope(root: Path) -> Path:
    candidates = (
        root / "visualization" / f"iter_{ARTIFACT_ITERATION}",
        root / f"iter_{ARTIFACT_ITERATION}",
    )
    for candidate in candidates:
        if candidate.exists():
            return _canonical_directory(candidate, "visualization iteration")
    return root


def _sidecar_schema(
    payload: Mapping[str, Any], sidecar: Path
) -> dict[str, tuple[tuple[int, ...], str]]:
    raw = payload.get("tensors")
    if not isinstance(raw, dict) or not raw:
        raise StageArtifactAuditError(
            f"sidecar tensors must be a non-empty object: {sidecar}"
        )
    schema: dict[str, tuple[tuple[int, ...], str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise StageArtifactAuditError(
                f"invalid tensor schema entry in {sidecar}: {key!r}"
            )
        shape = value.get("shape")
        dtype = value.get("dtype")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in shape
            )
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise StageArtifactAuditError(
                f"invalid tensor shape/dtype for {key!r} in {sidecar}"
            )
        schema[key] = (tuple(shape), dtype)
    return schema


def _validate_artifact(root: Path, scope: Path, raw_path: Path) -> ArtifactRecord:
    _ensure_no_symlink_components(raw_path, root)
    path = _canonical_regular_file(raw_path, "rank safetensors artifact")
    match = ARTIFACT_RE.fullmatch(path.name)
    if match is None:
        raise StageArtifactAuditError(f"invalid artifact filename: {path}")
    filename_rank = int(match.group(1))
    sidecar = path.with_suffix(".json")
    _ensure_no_symlink_components(sidecar, root)
    sidecar = _canonical_regular_file(sidecar, "rank artifact sidecar")
    payload = _load_json_strict(sidecar)

    iteration = payload.get("iteration")
    dataset = payload.get("dataset")
    rank = payload.get("global_rank")
    convention = payload.get("sigma_convention")
    expected_hash = payload.get("safetensors_sha256")
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or iteration != ARTIFACT_ITERATION
    ):
        raise StageArtifactAuditError(
            f"sidecar iteration must be {ARTIFACT_ITERATION}: {sidecar}"
        )
    if not isinstance(dataset, str) or not dataset or dataset != path.parent.name:
        raise StageArtifactAuditError(
            f"sidecar dataset does not match artifact directory: {sidecar}"
        )
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank != filename_rank
    ):
        raise StageArtifactAuditError(
            f"sidecar rank does not match artifact filename: {sidecar}"
        )
    if convention != SIGMA_CONVENTION:
        raise StageArtifactAuditError(
            f"unsupported sidecar sigma convention in {sidecar}: {convention!r}"
        )
    if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
        raise StageArtifactAuditError(
            f"invalid safetensors SHA-256 in sidecar: {sidecar}"
        )
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise StageArtifactAuditError(f"safetensors SHA-256 mismatch: {path}")

    sidecar_schema = _sidecar_schema(payload, sidecar)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            keys = frozenset(handle.keys())
            if keys != frozenset(sidecar_schema):
                raise StageArtifactAuditError(
                    f"sidecar/artifact tensor inventory mismatch: {path}"
                )
            for key in sorted(keys):
                tensor_slice = handle.get_slice(key)
                actual_shape = tuple(tensor_slice.get_shape())
                raw_dtype = str(tensor_slice.get_dtype())
                actual_dtype = SAFETENSORS_TO_TORCH_DTYPE.get(raw_dtype)
                if actual_dtype is None:
                    raise StageArtifactAuditError(
                        f"unsupported safetensors dtype {raw_dtype!r} for {key!r}"
                    )
                if sidecar_schema[key] != (actual_shape, actual_dtype):
                    raise StageArtifactAuditError(
                        f"sidecar/artifact schema mismatch for {key!r}: {path}"
                    )
    except StageArtifactAuditError:
        raise
    except Exception as exc:
        raise StageArtifactAuditError(
            f"could not validate safetensors artifact {path}: {exc}"
        ) from exc

    expected_metadata = {
        "iteration": str(iteration),
        "dataset": dataset,
        "sigma_convention": SIGMA_CONVENTION,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise StageArtifactAuditError(
                f"safetensors metadata {key!r} mismatch in {path}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    return ArtifactRecord(
        root=root,
        scope=scope,
        path=path,
        sidecar=sidecar,
        rank=rank,
        dataset=dataset,
        iteration=iteration,
        artifact_sha256=actual_hash,
        sidecar_sha256=_sha256_file(sidecar),
        keys=keys,
    )


def _discover(root: Path, label: str) -> tuple[Path, dict[int, ArtifactRecord]]:
    scope = _select_artifact_scope(root)
    raw_paths = sorted(scope.rglob("latent_trajectory_rank_*.safetensors"))
    if len(raw_paths) != WORLD_SIZE:
        raise StageArtifactAuditError(
            f"{label} must contain exactly {WORLD_SIZE} rank artifacts at "
            f"iteration {ARTIFACT_ITERATION}, found {len(raw_paths)} in {scope}"
        )
    records: dict[int, ArtifactRecord] = {}
    for raw_path in raw_paths:
        record = _validate_artifact(root, scope, raw_path)
        if record.rank in records:
            raise StageArtifactAuditError(
                f"{label} has duplicate artifact for rank {record.rank}"
            )
        records[record.rank] = record
    expected_ranks = list(range(WORLD_SIZE))
    if sorted(records) != expected_ranks:
        raise StageArtifactAuditError(
            f"{label} ranks must be exactly {expected_ranks}, "
            f"got {sorted(records)}"
        )
    datasets = {record.dataset for record in records.values()}
    if len(datasets) != 1:
        raise StageArtifactAuditError(
            f"{label} must contain one paired dataset, got {sorted(datasets)}"
        )
    return scope, records


def _get_tensor(record: ArtifactRecord, key: str) -> torch.Tensor:
    if key not in record.keys:
        raise StageArtifactAuditError(
            f"required tensor {key!r} is missing from {record.path}"
        )
    try:
        with safe_open(str(record.path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)
    except Exception as exc:
        raise StageArtifactAuditError(
            f"could not load tensor {key!r} from {record.path}: {exc}"
        ) from exc


def _integer_values(
    record: ArtifactRecord, key: str, *, expected_length: int | None = None
) -> tuple[int, ...]:
    tensor = _get_tensor(record, key)
    if (
        tensor.ndim != 1
        or tensor.dtype not in INTEGER_DTYPES
        or (expected_length is not None and tensor.numel() != expected_length)
    ):
        raise StageArtifactAuditError(
            f"{key} must be a one-dimensional integer tensor"
            + (
                ""
                if expected_length is None
                else f" with length {expected_length}"
            )
            + f": {record.path}"
        )
    return tuple(int(value) for value in tensor.tolist())


def _source_infix(source: str) -> str:
    return "" if source == "autonomous" else f"_{source}"


def _expected_final_keys(source_codes: Sequence[int]) -> set[str]:
    return {
        f"{prefix}{_source_infix(SOURCE_NAMES[code])}_nfe_{nfe}"
        for code in source_codes
        for nfe in NFE_STEPS
        for prefix in FINAL_PREFIXES
    }


def _validate_final_inventory(
    record: ArtifactRecord, source_codes: Sequence[int], label: str
) -> None:
    expected = _expected_final_keys(source_codes)
    actual: set[str] = set()
    for key in record.keys:
        if not key.startswith(FINAL_PREFIXES):
            continue
        match = FINAL_TENSOR_RE.fullmatch(key)
        if match is None:
            raise StageArtifactAuditError(
                f"{label} has an unrecognized final tensor: {key!r}"
            )
        actual.add(key)
    if actual != expected:
        raise StageArtifactAuditError(
            f"{label} final tensor inventory mismatch in rank {record.rank}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _comparison(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_hash = _sha256_tensor(left)
    right_hash = _sha256_tensor(right)
    passed = _tensor_bitwise_equal(left, right)
    return {
        "pass": passed,
        "comparison": "bitwise_tensor_identity",
        "left": {
            "label": left_label,
            "sha256": left_hash,
            "shape": list(left.shape),
            "dtype": str(left.dtype),
        },
        "right": {
            "label": right_label,
            "sha256": right_hash,
            "shape": list(right.shape),
            "dtype": str(right.dtype),
        },
    }


def _require_pass(result: Mapping[str, Any], message: str) -> None:
    if result.get("pass") is not True:
        raise StageArtifactAuditError(message)


def _artifact_set_sha256(records: Mapping[int, ArtifactRecord]) -> str:
    inventory = [
        {
            "rank": rank,
            "dataset": record.dataset,
            "artifact_sha256": record.artifact_sha256,
            "sidecar_sha256": record.sidecar_sha256,
        }
        for rank, record in sorted(records.items())
    ]
    encoded = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _forbidden_training_outputs(root: Path) -> list[str]:
    found: list[str] = []
    for path in root.rglob("*"):
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise StageArtifactAuditError(
                f"new root changed while checking training outputs: {path}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            continue
        name = path.name
        is_checkpoint_name = (
            path.suffix.lower() in {".pt", ".pth", ".ckpt"}
            and re.search(r"(?:^|[-_])(snapshot|checkpoint)(?:[-_.]|$)", name)
            is not None
        )
        if name in FORBIDDEN_TRAINING_BASENAMES or is_checkpoint_name:
            found.append(str(path.relative_to(root)))
    return sorted(found)


def _rank_artifact_inventory(record: ArtifactRecord) -> dict[str, Any]:
    return {
        "rank": record.rank,
        "dataset": record.dataset,
        "iteration": record.iteration,
        "artifact": {
            "path": str(record.path),
            "bytes": record.path.stat().st_size,
            "sha256": record.artifact_sha256,
        },
        "sidecar": {
            "path": str(record.sidecar),
            "bytes": record.sidecar.stat().st_size,
            "sha256": record.sidecar_sha256,
            "safetensors_sha256_matches": True,
            "sigma_convention_matches": True,
            "schema_matches": True,
        },
    }


def audit(
    *,
    new_root: str | Path,
    legacy_root: str | Path,
) -> dict[str, Any]:
    """Audit two immutable artifact roots and return a signed JSON payload."""

    new = _canonical_directory(new_root, "new evaluation root")
    legacy = _canonical_directory(legacy_root, "legacy evaluation root")
    if new == legacy or new.is_relative_to(legacy) or legacy.is_relative_to(new):
        raise StageArtifactAuditError(
            "new and legacy roots must be separate, non-nested directories"
        )

    forbidden = _forbidden_training_outputs(new)
    if forbidden:
        raise StageArtifactAuditError(
            "new evaluation root contains forbidden training outputs: "
            + ", ".join(forbidden)
        )

    new_scope, new_records = _discover(new, "new evaluation")
    legacy_scope, legacy_records = _discover(legacy, "legacy evaluation")
    if {
        record.dataset for record in new_records.values()
    } != {record.dataset for record in legacy_records.values()}:
        raise StageArtifactAuditError(
            "new and legacy artifact datasets do not match"
        )

    rank_results: list[dict[str, Any]] = []
    observed_noise_seeds: set[int] = set()
    for rank in range(WORLD_SIZE):
        new_record = new_records[rank]
        legacy_record = legacy_records[rank]

        new_codes = _integer_values(
            new_record,
            "evaluation_condition_source_codes",
            expected_length=len(NEW_SOURCE_CODES),
        )
        legacy_codes = _integer_values(
            legacy_record,
            "evaluation_condition_source_codes",
            expected_length=len(LEGACY_SOURCE_CODES),
        )
        new_nfe = _integer_values(
            new_record,
            "evaluation_nfe_steps",
            expected_length=len(NFE_STEPS),
        )
        legacy_nfe = _integer_values(
            legacy_record,
            "evaluation_nfe_steps",
            expected_length=len(NFE_STEPS),
        )
        stage_flag = _integer_values(
            new_record,
            "cascade_stage_faithful_inference",
            expected_length=1,
        )
        if new_codes != NEW_SOURCE_CODES:
            raise StageArtifactAuditError(
                f"new rank {rank} source codes {new_codes} != {NEW_SOURCE_CODES}"
            )
        if legacy_codes != LEGACY_SOURCE_CODES:
            raise StageArtifactAuditError(
                f"legacy rank {rank} source codes {legacy_codes} != "
                f"{LEGACY_SOURCE_CODES}"
            )
        if new_nfe != NFE_STEPS or legacy_nfe != NFE_STEPS:
            raise StageArtifactAuditError(
                f"rank {rank} NFE contract mismatch: "
                f"new={new_nfe}, legacy={legacy_nfe}, expected={NFE_STEPS}"
            )
        if stage_flag != (1,):
            raise StageArtifactAuditError(
                f"new rank {rank} stage-faithful flag must be 1"
            )
        _validate_final_inventory(new_record, new_codes, "new evaluation")
        _validate_final_inventory(
            legacy_record, legacy_codes, "legacy evaluation"
        )

        identity: dict[str, Any] = {}
        for key in IDENTITY_TENSORS:
            result = _comparison(
                _get_tensor(new_record, key),
                _get_tensor(legacy_record, key),
                left_label=f"new:{key}",
                right_label=f"legacy:{key}",
            )
            _require_pass(
                result,
                f"rank {rank} new/legacy identity mismatch for {key}",
            )
            identity[key] = result
        noise_seed = int(
            _integer_values(new_record, "evaluation_noise_seed", expected_length=1)[
                0
            ]
        )
        observed_noise_seeds.add(noise_seed)

        legacy_reproduction: dict[str, Any] = {}
        stage_tf_equivalence: dict[str, Any] = {}
        for nfe in NFE_STEPS:
            legacy_reproduction[str(nfe)] = {}
            for prefix in FINAL_PREFIXES:
                new_key = f"{prefix}_autonomous_legacy_nfe_{nfe}"
                old_key = f"{prefix}_nfe_{nfe}"
                result = _comparison(
                    _get_tensor(new_record, new_key),
                    _get_tensor(legacy_record, old_key),
                    left_label=f"new:{new_key}",
                    right_label=f"legacy:{old_key}",
                )
                _require_pass(
                    result,
                    f"rank {rank} NFE {nfe} legacy reproduction mismatch "
                    f"for {prefix}",
                )
                legacy_reproduction[str(nfe)][prefix] = result

            new_auto_key = f"tf_final_nfe_{nfe}"
            new_shuffled_key = f"tf_final_autonomous_shuffled_nfe_{nfe}"
            old_off_key = f"tf_final_off_nfe_{nfe}"
            auto_tensor = _get_tensor(new_record, new_auto_key)
            shuffled_tensor = _get_tensor(new_record, new_shuffled_key)
            old_off_tensor = _get_tensor(legacy_record, old_off_key)
            auto_vs_shuffled = _comparison(
                auto_tensor,
                shuffled_tensor,
                left_label=f"new:{new_auto_key}",
                right_label=f"new:{new_shuffled_key}",
            )
            auto_vs_old_off = _comparison(
                auto_tensor,
                old_off_tensor,
                left_label=f"new:{new_auto_key}",
                right_label=f"legacy:{old_off_key}",
            )
            _require_pass(
                auto_vs_shuffled,
                f"rank {rank} NFE {nfe} stage autonomous/shuffled TF mismatch",
            )
            _require_pass(
                auto_vs_old_off,
                f"rank {rank} NFE {nfe} stage autonomous/legacy-off TF mismatch",
            )
            stage_tf_equivalence[str(nfe)] = {
                "pass": True,
                "new_autonomous_vs_new_autonomous_shuffled": auto_vs_shuffled,
                "new_autonomous_vs_legacy_off": auto_vs_old_off,
            }

        rank_results.append(
            {
                "rank": rank,
                "pass": True,
                "contracts": {
                    "pass": True,
                    "new_source_codes": {
                        "observed": list(new_codes),
                        "expected": list(NEW_SOURCE_CODES),
                        "pass": True,
                    },
                    "legacy_source_codes": {
                        "observed": list(legacy_codes),
                        "expected": list(LEGACY_SOURCE_CODES),
                        "pass": True,
                    },
                    "new_stage_faithful_flag": {
                        "observed": stage_flag[0],
                        "expected": 1,
                        "pass": True,
                    },
                    "nfe_steps": {
                        "new": list(new_nfe),
                        "legacy": list(legacy_nfe),
                        "expected": list(NFE_STEPS),
                        "pass": True,
                    },
                },
                "new_legacy_input_identity": {
                    "pass": True,
                    "tensors": identity,
                },
                "legacy_reproduction": {
                    "pass": True,
                    "description": (
                        "new autonomous_legacy finals equal legacy autonomous "
                        "finals bitwise"
                    ),
                    "nfe": legacy_reproduction,
                },
                "stage_tf_equivalence": {
                    "pass": True,
                    "description": (
                        "stage autonomous TF equals stage shuffled TF and "
                        "legacy off TF bitwise"
                    ),
                    "nfe": stage_tf_equivalence,
                },
            }
        )

    if len(observed_noise_seeds) != 1:
        raise StageArtifactAuditError(
            "evaluation_noise_seed is inconsistent across ranks: "
            f"{sorted(observed_noise_seeds)}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "stage_faithful_cascade_bitwise_artifact_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sigma_convention": SIGMA_CONVENTION,
        "read_only_inputs": True,
        "overall_pass": True,
        "contracts": {
            "pass": True,
            "world_size": {
                "observed_new": len(new_records),
                "observed_legacy": len(legacy_records),
                "expected": WORLD_SIZE,
                "pass": True,
            },
            "paired_ranks": {
                "new": sorted(new_records),
                "legacy": sorted(legacy_records),
                "expected": list(range(WORLD_SIZE)),
                "pass": True,
            },
            "artifact_iteration": {
                "new": ARTIFACT_ITERATION,
                "legacy": ARTIFACT_ITERATION,
                "expected": ARTIFACT_ITERATION,
                "pass": True,
            },
            "evaluation_noise_seed_identity": {
                "observed": next(iter(observed_noise_seeds)),
                "pass": True,
            },
            "forbidden_training_outputs": {
                "observed": forbidden,
                "expected": [],
                "pass": True,
            },
            "sidecar_hashes_and_sigma_convention": {"pass": True},
            "new_source_codes": {
                "expected": list(NEW_SOURCE_CODES),
                "pass": True,
            },
            "legacy_source_codes": {
                "expected": list(LEGACY_SOURCE_CODES),
                "pass": True,
            },
            "nfe_steps": {"expected": list(NFE_STEPS), "pass": True},
            "new_stage_faithful_flag": {"expected": 1, "pass": True},
        },
        "inputs": {
            "new": {
                "root": str(new),
                "artifact_scope": str(new_scope),
                "artifact_set_sha256": _artifact_set_sha256(new_records),
                "ranks": [
                    _rank_artifact_inventory(record)
                    for _, record in sorted(new_records.items())
                ],
            },
            "legacy": {
                "root": str(legacy),
                "artifact_scope": str(legacy_scope),
                "artifact_set_sha256": _artifact_set_sha256(legacy_records),
                "ranks": [
                    _rank_artifact_inventory(record)
                    for _, record in sorted(legacy_records.items())
                ],
            },
        },
        "rank_audits": rank_results,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _output_path(value: str | Path, roots: Sequence[Path]) -> Path:
    raw = Path(value).expanduser()
    if raw.exists() or raw.is_symlink():
        raise StageArtifactAuditError(f"output already exists: {raw}")
    parent = _canonical_directory(raw.parent, "output parent")
    output = parent / raw.name
    for root in roots:
        if output == root or output.is_relative_to(root):
            raise StageArtifactAuditError(
                f"output must be outside evaluated input roots: {output}"
            )
    return output


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StageArtifactAuditError(
            f"could not exclusively create output {path}: {exc}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-root",
        required=True,
        help=(
            "Fresh stage-faithful evaluation run root, or its iter_199 "
            "artifact directory"
        ),
    )
    parser.add_argument(
        "--legacy-root",
        required=True,
        help=(
            "Legacy matched-cascade run root, or its iter_199 artifact "
            "directory"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New JSON report path outside both immutable input roots",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        new_root = _canonical_directory(args.new_root, "new evaluation root")
        legacy_root = _canonical_directory(
            args.legacy_root, "legacy evaluation root"
        )
        output = _output_path(args.output, (new_root, legacy_root))
        payload = audit(new_root=new_root, legacy_root=legacy_root)
        output_sha256 = _exclusive_json(output, payload)
    except StageArtifactAuditError as exc:
        print(f"artifact audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "overall_pass": True,
                "output": str(output),
                "output_sha256": output_sha256,
                "identity_sha256": payload["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
