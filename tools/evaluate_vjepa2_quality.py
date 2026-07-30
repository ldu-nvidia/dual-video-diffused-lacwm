#!/usr/bin/env python3
"""Evaluate all 128 pinned test clips for one V-JEPA study arm/milestone.

This is the scientific quality evaluator for the controlled study.  Trainer
visualizations remain useful diagnostics, but their one sample per rank is not
used for a claim.  Run this tool with one process per B200:

``python -m torch.distributed.run --standalone --nproc_per_node=8 ...``

Every rank evaluates a disjoint deterministic slice of the immutable 128-clip
test cache.  Noise is derived statelessly inside the model from each
``clip_index``, so the same clip starts from bit-identical video and auxiliary
noise in every source, NFE, checkpoint, and arm.

Intermediate primary milestones use an efficient preregistered grid:

* autonomous/off/autonomous_shuffled at NFE 4 and 8;
* oracle_matched/oracle_shuffled at NFE 4.

The final 1000-update milestone uses all five sources at all seven NFEs.
Oracle rows are leakage-only headroom diagnostics.  No pretrained perceptual
metric is instantiated: the study manifest does not pin an LPIPS/video-model
checkpoint, and implicit downloads are forbidden.  Claims are therefore
restricted to latent and decoded reconstruction metrics.

The opt-in frontier mode is separate: it evaluates the full autonomous grid on
validation, then only a frozen pair on a newly registered episode-disjoint
lockbox. The already inspected original test split cannot serve as that
lockbox.

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
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools import vjepa2_frontier_lockbox as frontier_lockbox
    from tools import vjepa2_nfe_frontier as frontier_contract
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import vjepa2_frontier_lockbox as frontier_lockbox
    import vjepa2_nfe_frontier as frontier_contract


SCHEMA_VERSION = 1
PRIMARY_UPDATES = (1, 50, 100, 200, 400, 800, 1000)
FINAL_UPDATE = 1000
NFE_STEPS = (1, 2, 4, 6, 8, 12, 20)
SOURCES = (
    "autonomous",
    "off",
    "autonomous_shuffled",
    "oracle_matched",
    "oracle_shuffled",
)
ORACLE_SOURCES = {"oracle_matched", "oracle_shuffled"}
SOURCE_CODES = {
    "autonomous": 0,
    "off": 1,
    "oracle_matched": 2,
    "oracle_shuffled": 3,
    "autonomous_shuffled": 4,
}
QUANTITATIVE_ARMS = {"VPM", "A1", "J0", "J1"}
EXPECTED_TEST_CLIPS = 128
EXPECTED_FRONTIER_CLIPS = {"validation": 64, "lockbox": 128}
FRONTIER_SAMPLE_ID_OFFSETS = {"validation": 1_000_000, "lockbox": 3_000_000}
EXPECTED_WORLD_SIZE = 8
DEFAULT_BATCH_SIZE_PER_RANK = 2
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def intermediate_grid() -> tuple[tuple[str, int], ...]:
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


def final_grid() -> tuple[tuple[str, int], ...]:
    return tuple((source, nfe) for source in SOURCES for nfe in NFE_STEPS)


def frontier_validation_grid() -> tuple[tuple[str, int], ...]:
    """Validation-only deployable grid used to select the NFE frontier."""

    return tuple(("autonomous", nfe) for nfe in NFE_STEPS)


def frontier_test_grid(
    arm: str, selection: Mapping[str, Any]
) -> tuple[tuple[str, int], ...]:
    """Frozen test grid; no candidate selection is permitted on test."""

    pair = selection.get("selected_pair")
    left = pair.get("left") if isinstance(pair, Mapping) else None
    reference = pair.get("reference") if isinstance(pair, Mapping) else None
    k = left.get("nfe") if isinstance(left, Mapping) else None
    m = reference.get("nfe") if isinstance(reference, Mapping) else None
    if (
        arm not in {"J1", "VPM"}
        or not isinstance(k, int)
        or isinstance(k, bool)
        or not isinstance(m, int)
        or isinstance(m, bool)
        or k < 2
        or k >= m
        or k not in NFE_STEPS
        or m not in NFE_STEPS
    ):
        raise QualityEvaluationError("frontier selection pair is invalid")
    nfes = (k,) if arm == "J1" else (k, m)
    return tuple(("autonomous", nfe) for nfe in nfes)


def quality_grid(completed_updates: int) -> tuple[tuple[str, int], ...]:
    if completed_updates not in PRIMARY_UPDATES:
        raise QualityEvaluationError(
            f"quality evaluation is not pinned at update {completed_updates}"
        )
    return (
        final_grid()
        if completed_updates == FINAL_UPDATE
        else intermediate_grid()
    )


class QualityEvaluationError(RuntimeError):
    """Raised when quality evidence would be incomplete or incomparable."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def _identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == recorded


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    import torch

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
        raise QualityEvaluationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise QualityEvaluationError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise QualityEvaluationError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise QualityEvaluationError(
            f"{label} must be a non-empty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QualityEvaluationError(
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
        raise QualityEvaluationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise QualityEvaluationError(f"{label} must contain one object: {path}")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise QualityEvaluationError(
                        f"{label} has blank row {line_number}: {path}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QualityEvaluationError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(row)
    except json.JSONDecodeError as exc:
        raise QualityEvaluationError(f"{label} is invalid JSONL: {path}") from exc
    return rows


def _exclusive_bytes(path: Path, content: bytes) -> None:
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


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise QualityEvaluationError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _assert_clean_commit(repo: Path, expected: str) -> None:
    if COMMIT_RE.fullmatch(expected) is None:
        raise QualityEvaluationError(
            "expected commit must be 40 lowercase hexadecimal characters"
        )
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected:
        raise QualityEvaluationError(f"Git commit differs: {actual} != {expected}")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise QualityEvaluationError(
            "repository must be clean: " + status.replace("\n", "; ")
        )


def _source_infix(source: str) -> str:
    if source not in SOURCES:
        raise QualityEvaluationError(f"unknown quality source: {source!r}")
    return "" if source == "autonomous" else f"_{source}"


def _per_sample_nmse(
    prediction: Any, target: Any, history: int, key: str
) -> list[float]:
    import torch

    prediction = prediction[:, :, history:].double().flatten(1)
    target = target[:, :, history:].double().flatten(1)
    numerator = torch.sum((prediction - target).square(), dim=1)
    denominator = torch.sum(target.square(), dim=1)
    if not bool(torch.isfinite(denominator).all()) or bool(
        (denominator <= 0).any()
    ):
        raise QualityEvaluationError(f"{key} has invalid target energy")
    values = numerator / denominator
    if not bool(torch.isfinite(values).all()):
        raise QualityEvaluationError(f"{key} NMSE is non-finite")
    return [float(value) for value in values.tolist()]


def _per_sample_cosine(
    prediction: Any, target: Any, history: int, key: str
) -> list[float]:
    import torch

    prediction = prediction[:, :, history:].double().flatten(1)
    target = target[:, :, history:].double().flatten(1)
    denominator = torch.linalg.vector_norm(
        prediction, dim=1
    ) * torch.linalg.vector_norm(target, dim=1)
    if not bool(torch.isfinite(denominator).all()) or bool(
        (denominator <= 0).any()
    ):
        raise QualityEvaluationError(f"{key} has invalid cosine norm")
    values = torch.sum(prediction * target, dim=1) / denominator
    if not bool(torch.isfinite(values).all()):
        raise QualityEvaluationError(f"{key} cosine is non-finite")
    return [
        float(min(1.0, max(-1.0, value))) for value in values.tolist()
    ]


def _per_sample_decoded(
    prediction: Any,
    target: Any,
    *,
    prediction_history: Any | None = None,
    target_history: Any | None = None,
) -> dict[str, list[float]]:
    import torch

    prediction = prediction.double() / 255.0
    target = target.double() / 255.0
    if (prediction_history is None) != (target_history is None):
        raise QualityEvaluationError(
            "decoded temporal boundary requires both prediction/target history"
        )
    if prediction_history is not None:
        prediction_history = prediction_history.double() / 255.0
        target_history = target_history.double() / 255.0
        expected_history_shape = (
            prediction.shape[0],
            prediction.shape[1],
            1,
            prediction.shape[3],
            prediction.shape[4],
        )
        if (
            tuple(prediction_history.shape) != expected_history_shape
            or tuple(target_history.shape) != expected_history_shape
        ):
            raise QualityEvaluationError(
                "decoded temporal-boundary history shape differs"
            )
        temporal_prediction = torch.cat(
            (prediction_history, prediction), dim=2
        )
        temporal_target = torch.cat((target_history, target), dim=2)
    else:
        temporal_prediction = prediction
        temporal_target = target
    reduce_dims = tuple(range(1, prediction.ndim))
    mse = torch.mean((prediction - target).square(), dim=reduce_dims)
    temporal = torch.mean(
        (
            torch.diff(temporal_prediction, dim=2)
            - torch.diff(temporal_target, dim=2)
        ).square(),
        dim=reduce_dims,
    )
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
    if not all(
        bool(torch.isfinite(values).all())
        for values in (mse, temporal, psnr)
    ):
        raise QualityEvaluationError("decoded metrics are non-finite")
    return {
        "decoded_mse_unit_range": [float(value) for value in mse.tolist()],
        "decoded_psnr_db": [float(value) for value in psnr.tolist()],
        "decoded_temporal_difference_mse_unit_range": [
            float(value) for value in temporal.tolist()
        ],
    }


def _slice_hashes(tensor: Any) -> list[str]:
    return [_tensor_sha256(tensor[index : index + 1]) for index in range(len(tensor))]


def _move_batch(samples: Sequence[Mapping[str, Any]], device: Any) -> dict[str, Any]:
    import torch
    from torch.utils.data import default_collate

    batch = default_collate(samples)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device=device, non_blocking=False)
    required = {
        "rgb",
        "actions",
        "mask",
        "morphology_index",
        "auxiliary_target",
        "clip_index",
    }
    missing = sorted(required - batch.keys())
    if missing:
        raise QualityEvaluationError(f"quality batch lacks {missing}")
    return batch


def _assert_teacher_absent(model: Any, config: Any) -> None:
    from omegaconf import OmegaConf

    suspicious = []
    for name, module in model.named_modules():
        identity = (
            f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        )
        if "vjepa" in identity or "jepa" in identity or "teacher" in identity:
            suspicious.append(identity)
    if suspicious:
        raise QualityEvaluationError(
            "online teacher-like module is registered: "
            + "; ".join(suspicious[:8])
        )
    serialized = json.dumps(
        OmegaConf.to_container(config.model, resolve=True),
        sort_keys=True,
    ).lower()
    for forbidden in (
        "vjepa2_target",
        "vjepa_source",
        "teacher_checkpoint",
        "teacher_encoder",
    ):
        if forbidden in serialized:
            raise QualityEvaluationError(
                f"model config contains online teacher marker {forbidden!r}"
            )


def _validate_inputs(
    args: argparse.Namespace,
    *,
    verify_snapshot_sha256: bool,
    verify_cache_arrays: bool,
) -> dict[str, Any]:
    frontier_split = getattr(args, "frontier_split", None)
    evaluator_commit = (
        getattr(args, "evaluator_commit", None) or args.expected_commit
    )
    repo = _canonical_directory(args.repo_root, "repository root")
    executed_roots = {
        Path(module.__file__).resolve().parents[1]
        for module in (frontier_contract, frontier_lockbox)
    }
    executed_roots.add(Path(__file__).resolve().parents[1])
    if executed_roots != {repo}:
        raise QualityEvaluationError(
            "evaluator and frontier helpers do not belong to --repo-root"
        )
    _assert_clean_commit(repo, evaluator_commit)
    try:
        inference_compatibility = (
            frontier_contract.git_inference_compatibility(
                repo,
                training_commit=args.expected_commit,
                tool_commit=evaluator_commit,
            )
        )
    except frontier_contract.FrontierError as exc:
        raise QualityEvaluationError(str(exc)) from exc
    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    arm_path = _canonical_file(args.arm_manifest, "arm manifest")
    study_path = _canonical_file(args.study_manifest, "study manifest")
    stage_path = _canonical_file(args.stage_manifest, "stage manifest")
    stage_outcome_path = _canonical_file(
        args.stage_outcome, "stage outcome"
    )
    resolved = _canonical_file(args.resolved_config, "resolved config")
    snapshot = _canonical_file(args.snapshot, "snapshot")
    arm = _read_json(arm_path, "arm manifest")
    study = _read_json(study_path, "study manifest")
    stage = _read_json(stage_path, "stage manifest")
    stage_outcome = _read_json(stage_outcome_path, "stage outcome")
    if (
        not _identity_valid(arm)
        or arm.get("kind") != "vjepa2_controlled_study_arm"
        or arm.get("run_dir") != str(run_dir)
        or arm.get("git_commit") != args.expected_commit
    ):
        raise QualityEvaluationError("arm manifest identity/provenance differs")
    arm_code = str(arm.get("arm", {}).get("code", ""))
    if (
        arm_code not in QUANTITATIVE_ARMS
        or arm.get("arm", {}).get("dual_enabled") is not True
    ):
        raise QualityEvaluationError(
            "paired quality evaluator supports only VPM/A1/J0/J1"
        )
    if (
        not _identity_valid(study)
        or study.get("kind")
        != "vjepa2_controlled_video_diffusion_study"
        or study.get("identity_sha256")
        != arm.get("study_identity_sha256")
    ):
        raise QualityEvaluationError("study manifest identity differs")
    videox_path = _canonical_directory(
        study.get("inputs", {}).get("runtime", {}).get("videox_home", ""),
        "pinned VideoX-Fun checkout",
    )
    try:
        videox_runtime = frontier_contract.git_runtime_provenance(videox_path)
    except frontier_contract.FrontierError as exc:
        raise QualityEvaluationError(str(exc)) from exc
    if (
        not _identity_valid(stage)
        or stage.get("kind") != "vjepa2_controlled_study_stage"
        or stage.get("arm_identity_sha256") != arm.get("identity_sha256")
        or stage.get("arm_code") != arm_code
        or stage.get("stage_endpoint_completed_updates")
        != args.completed_updates
        or stage.get("primary_milestone") is not True
        or stage.get("trainer_terminal_iteration")
        != args.completed_updates - 1
    ):
        raise QualityEvaluationError("stage manifest identity/provenance differs")
    if (
        not _identity_valid(stage_outcome)
        or stage_outcome.get("kind")
        != "vjepa2_controlled_study_stage_outcome"
        or stage_outcome.get("arm_identity_sha256")
        != arm.get("identity_sha256")
        or stage_outcome.get("stage_identity_sha256")
        != stage.get("identity_sha256")
        or stage_outcome.get("arm_code") != arm_code
        or stage_outcome.get("completed_updates") != args.completed_updates
    ):
        raise QualityEvaluationError(
            "stage outcome identity/provenance differs"
        )
    expected_resolved = stage.get("resolved_config")
    if (
        not isinstance(expected_resolved, dict)
        or expected_resolved.get("path") != str(resolved)
        or expected_resolved.get("sha256") != _sha256(resolved)
    ):
        raise QualityEvaluationError("resolved config differs from stage manifest")
    if snapshot != run_dir / "snapshot.pt":
        raise QualityEvaluationError("snapshot must be the live arm snapshot")
    observed_snapshot = stage_outcome.get("snapshot_observed_at_stage_end")
    if (
        not isinstance(observed_snapshot, dict)
        or observed_snapshot.get("path") != str(snapshot)
        or not isinstance(observed_snapshot.get("sha256"), str)
        or SHA256_RE.fullmatch(observed_snapshot["sha256"]) is None
        or (
            verify_snapshot_sha256
            and observed_snapshot.get("sha256") != _sha256(snapshot)
        )
    ):
        raise QualityEvaluationError(
            "live snapshot differs from the immutable stage outcome"
        )
    selection: dict[str, Any] | None = None
    selection_path: Path | None = None
    selection_identity: str | None = None
    lockbox_registration: dict[str, Any] | None = None
    lockbox_validation: dict[str, Any] | None = None
    lockbox_identity: str | None = None
    if frontier_split is None:
        split_name = "test"
        expected_clips = EXPECTED_TEST_CLIPS
        dataset_config_key = "viz_dataset"
        grid = quality_grid(args.completed_updates)
        expected_output = (
            run_dir / "quality" / f"update_{args.completed_updates:04d}"
        )
        if getattr(args, "frontier_selection", None) is not None:
            raise QualityEvaluationError(
                "--frontier-selection requires --frontier-split lockbox"
            )
    else:
        if args.completed_updates != FINAL_UPDATE or arm_code not in {"J1", "VPM"}:
            raise QualityEvaluationError(
                "frontier evaluation requires final J1 or VPM checkpoint"
            )
        if args.batch_size_per_rank != DEFAULT_BATCH_SIZE_PER_RANK:
            raise QualityEvaluationError(
                "frontier evaluation requires batch size 2 per rank"
            )
        split_name = frontier_split
        expected_clips = EXPECTED_FRONTIER_CLIPS[split_name]
        dataset_config_key = (
            "val_dataset" if split_name == "validation" else "viz_dataset"
        )
        if split_name == "validation":
            if getattr(args, "frontier_selection", None) is not None:
                raise QualityEvaluationError(
                    "validation frontier grid must precede selection"
                )
            grid = frontier_validation_grid()
            expected_output = (
                run_dir
                / "frontier_quality"
                / "validation"
                / f"update_{args.completed_updates:04d}"
            )
        else:
            if getattr(args, "frontier_selection", None) is None:
                raise QualityEvaluationError(
                    "frontier lockbox requires a frozen validation selection"
                )
            selection_path = _canonical_file(
                args.frontier_selection, "frontier selection"
            )
            selection = _read_json(selection_path, "frontier selection")
            if (
                not _identity_valid(selection)
                or selection.get("kind") != "vjepa2_nfe_frontier_selection"
                or selection.get("confirmatory_eligible") is not True
                or selection.get("selection_split") != "validation"
                or selection.get("training_git_commit")
                != args.expected_commit
                or selection.get("evaluator_git_commit")
                != evaluator_commit
                or selection.get("study_identity_sha256")
                != study.get("identity_sha256")
                or selection.get("arm_identity_sha256", {}).get(arm_code)
                != arm.get("identity_sha256")
                or selection.get("stage_identity_sha256", {}).get(arm_code)
                != stage.get("identity_sha256")
            ):
                raise QualityEvaluationError(
                    "frontier lockbox selection is not confirmatory "
                    "validation evidence"
                )
            selection_identity = str(selection["identity_sha256"])
            candidate_lockbox = selection.get("lockbox_registration")
            if not isinstance(candidate_lockbox, dict):
                raise QualityEvaluationError(
                    "selection does not bind a registered fresh lockbox"
                )
            try:
                lockbox_validation = frontier_lockbox.validate_registration(
                    candidate_lockbox,
                    study=study,
                    rehash_arrays=False,
                    verify_construction=verify_cache_arrays,
                )
            except frontier_lockbox.LockboxError as exc:
                raise QualityEvaluationError(str(exc)) from exc
            lockbox_registration = candidate_lockbox
            lockbox_identity = str(candidate_lockbox["identity_sha256"])
            if (
                candidate_lockbox.get("registration_git_commit")
                != evaluator_commit
                or candidate_lockbox.get("inference_code_compatibility")
                != inference_compatibility
            ):
                raise QualityEvaluationError(
                    "lockbox registration/evaluator code provenance differs"
                )
            grid = frontier_test_grid(arm_code, selection)
            expected_output = (
                run_dir
                / "frontier_quality"
                / "lockbox"
                / lockbox_identity
                / selection_identity
                / f"update_{args.completed_updates:04d}"
            )
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.absolute() != expected_output:
        raise QualityEvaluationError(
            f"output directory must be exactly {expected_output}"
        )
    if split_name == "lockbox":
        assert lockbox_registration is not None
        evaluation_split = {
            "clip_manifest": lockbox_registration["manifest"],
            "cache": {
                "metadata": lockbox_registration["cache"]["metadata"],
                **lockbox_registration["cache"]["arrays"],
            },
        }
    else:
        evaluation_split = (
            study.get("inputs", {}).get("splits", {}).get(split_name, {})
        )
    quality_protocol = study.get("inference", {}).get("quality_protocol", {})
    expected_grid = [
        {"source": source, "nfe": nfe}
        for source, nfe in quality_grid(args.completed_updates)
    ]
    recorded_grid = (
        quality_protocol.get("final_grid")
        if args.completed_updates == FINAL_UPDATE
        else quality_protocol.get("intermediate_grid")
    )
    if frontier_split is None and (
        quality_protocol.get("fixed_test_clips") != EXPECTED_TEST_CLIPS
        or quality_protocol.get("distributed_world_size")
        != EXPECTED_WORLD_SIZE
        or quality_protocol.get("batch_size_per_rank")
        != args.batch_size_per_rank
        or quality_protocol.get("trainer_visualization_is_diagnostic_only")
        is not True
        or quality_protocol.get("stateless_noise_key") != "clip_index"
        or quality_protocol.get(
            "deployable_sources_use_history_only_public_entrypoint"
        )
        is not True
        or quality_protocol.get("oracle_sources_are_leakage_only") is not True
        or quality_protocol.get(
            "rank0_rehashes_full_test_target_rgb_action_arrays"
        )
        is not True
        or quality_protocol.get(
            "temporal_metric_includes_history_to_first_future_boundary"
        )
        is not True
        or recorded_grid != expected_grid
    ):
        raise QualityEvaluationError(
            "study quality protocol differs from evaluator contract"
        )
    manifest_record = evaluation_split.get("clip_manifest", {})
    cache_record = evaluation_split.get("cache", {}).get("metadata", {})
    if (
        not isinstance(manifest_record, dict)
        or manifest_record.get("entries") != expected_clips
        or not isinstance(cache_record, dict)
    ):
        raise QualityEvaluationError(
            f"study must pin exactly {expected_clips} immutable {split_name} clips"
        )
    evaluation_manifest = _canonical_file(
        manifest_record.get("path", ""), f"{split_name} clip manifest"
    )
    cache_metadata = _canonical_file(
        cache_record.get("path", ""), f"{split_name} cache metadata"
    )
    if (
        manifest_record.get("sha256") != _sha256(evaluation_manifest)
        or cache_record.get("sha256") != _sha256(cache_metadata)
    ):
        raise QualityEvaluationError(
            f"{split_name} manifest/cache digest differs"
        )
    cache = _read_json(cache_metadata, f"{split_name} cache metadata")
    cache_arrays: dict[str, dict[str, Any]] = {}
    for name, file_key, sha_key in (
        ("target", "target_file", "target_sha256"),
        ("rgb", "rgb_file", "rgb_sha256"),
        ("actions", "actions_file", "actions_sha256"),
    ):
        study_record = evaluation_split.get("cache", {}).get(name)
        if not isinstance(study_record, dict):
            raise QualityEvaluationError(
                f"study lacks pinned {split_name} {name} array"
            )
        array_path = _canonical_file(
            study_record.get("path", ""), f"{split_name} {name} cache array"
        )
        metadata_value = cache.get(file_key)
        metadata_path = Path(str(metadata_value))
        if not metadata_path.is_absolute():
            metadata_path = cache_metadata.parent / metadata_path
        metadata_path = _canonical_file(
            metadata_path, f"metadata {split_name} {name} cache array"
        )
        recorded_sha = study_record.get("sha256")
        if (
            array_path != metadata_path
            or not isinstance(recorded_sha, str)
            or SHA256_RE.fullmatch(recorded_sha) is None
            or cache.get(sha_key) != recorded_sha
            or study_record.get("bytes") != array_path.stat().st_size
            or (
                verify_cache_arrays
                and _sha256(array_path) != recorded_sha
            )
        ):
            raise QualityEvaluationError(
                f"{split_name} {name} cache array differs from study record"
            )
        cache_arrays[name] = {
            "path": str(array_path),
            "sha256": recorded_sha,
            "bytes": array_path.stat().st_size,
            "full_sha256_verified_by_rank0": verify_cache_arrays,
        }
    descriptors = []
    with evaluation_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                descriptors.append(json.loads(line))
    if len(descriptors) != expected_clips:
        raise QualityEvaluationError(
            f"{split_name} manifest does not contain {expected_clips} rows"
        )
    indexes = [int(row.get("auxiliary_index", -1)) for row in descriptors]
    clip_ids = [str(row.get("clip_id", "")) for row in descriptors]
    if (
        indexes != list(range(expected_clips))
        or len(set(clip_ids)) != expected_clips
        or any(not clip_id for clip_id in clip_ids)
    ):
        raise QualityEvaluationError(
            f"{split_name} clip IDs/indexes are not unique, dense, and ordered"
        )
    if selection is not None:
        validation_clip_ids = {
            str(unit[0])
            for unit in selection.get("selection_clip_units", [])
            if isinstance(unit, list) and len(unit) == 2
        }
        if (
            len(validation_clip_ids) != EXPECTED_FRONTIER_CLIPS["validation"]
            or validation_clip_ids.intersection(clip_ids)
        ):
            raise QualityEvaluationError(
                "frozen validation selection and lockbox clips are not disjoint"
            )
    return {
        "repo": repo,
        "run_dir": run_dir,
        "arm_manifest_path": arm_path,
        "study_manifest_path": study_path,
        "stage_manifest_path": stage_path,
        "stage_outcome_path": stage_outcome_path,
        "resolved_config": resolved,
        "snapshot": snapshot,
        "snapshot_sha256": observed_snapshot["sha256"],
        "arm_manifest": arm,
        "study_manifest": study,
        "stage_manifest": stage,
        "stage_outcome": stage_outcome,
        "arm_code": arm_code,
        "evaluation_manifest": evaluation_manifest,
        "cache_metadata": cache_metadata,
        "cache_arrays": cache_arrays,
        "descriptors": descriptors,
        "evaluation_split": split_name,
        "frontier_mode": frontier_split is not None,
        "frontier_selection_path": selection_path,
        "frontier_selection": selection,
        "frontier_selection_identity_sha256": selection_identity,
        "lockbox_registration": lockbox_registration,
        "lockbox_registration_identity_sha256": lockbox_identity,
        "lockbox_validation": lockbox_validation,
        "dataset_config_key": dataset_config_key,
        "expected_clips": expected_clips,
        "grid": grid,
        "training_git_commit": args.expected_commit,
        "evaluator_git_commit": evaluator_commit,
        "inference_code_compatibility": inference_compatibility,
        "videox_runtime": videox_runtime,
        "output_dir": expected_output,
    }


def _load_model_and_dataset(
    inputs: Mapping[str, Any], *, device: Any
) -> tuple[Any, Any, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    repo = inputs["repo"]
    project_root = repo / "projects" / "latent_action_models"
    for root in (str(repo), str(project_root)):
        if root not in sys.path:
            sys.path.insert(0, root)
    config = OmegaConf.load(inputs["resolved_config"])
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    target = str(config.model.get("_target_", ""))
    if not target.endswith(".DualExplicitActionDiTModel"):
        raise QualityEvaluationError(
            "resolved model is not DualExplicitActionDiTModel"
        )
    model = instantiate(config.model)
    snapshot = torch.load(
        inputs["snapshot"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        not isinstance(snapshot, Mapping)
        or "model" not in snapshot
        or snapshot.get("run_identity_sha256")
        != inputs["arm_manifest"]["identity_sha256"]
        or snapshot.get("_start_iter")
        != inputs["stage_manifest"]["stage_endpoint_completed_updates"]
    ):
        raise QualityEvaluationError(
            "snapshot identity or completed-update cursor is invalid"
        )
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise QualityEvaluationError(f"strict snapshot load failed: {incompatible}")
    del snapshot
    model = model.to(device=device).eval()
    _assert_teacher_absent(model, config)
    if getattr(model, "time_frequency_transform", None) is not None:
        raise QualityEvaluationError("online auxiliary transform is present")
    dataset_config = config[inputs["dataset_config_key"]]
    if inputs["evaluation_split"] == "lockbox":
        from omegaconf import OmegaConf

        dataset_config = OmegaConf.create(
            OmegaConf.to_container(dataset_config, resolve=True)
        )
        dataset_config.datasets.ABC.clip_manifest = str(
            inputs["evaluation_manifest"]
        )
        dataset_config.datasets.ABC.cache_metadata = str(
            inputs["cache_metadata"]
        )
    dataset = instantiate(dataset_config)
    if len(dataset) != inputs["expected_clips"]:
        raise QualityEvaluationError(
            f"resolved {inputs['evaluation_split']} dataset has {len(dataset)} "
            f"!= {inputs['expected_clips']} clips"
        )
    abc = getattr(dataset, "datasets", {}).get("ABC")
    if (
        abc is None
        or Path(str(getattr(abc, "clip_manifest", "")))
        != inputs["evaluation_manifest"]
        or Path(str(getattr(abc, "cache_metadata", "")))
        != inputs["cache_metadata"]
    ):
        raise QualityEvaluationError(
            f"resolved {inputs['evaluation_split']} dataset differs from "
            "pinned study inputs"
        )
    return model, dataset, config


def _prepare_scoring_targets(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Construct held-out targets without passing them into deployable sampling."""
    import torch

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        video_clean = model._encode_clip(batch["rgb"]).to(batch["rgb"].dtype)
        decoded = model.rgb_tokenizer.decode_temporal(
            video_clean,
            out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
        )
    future_frames = min(model.num_future_frames, decoded.shape[2])
    if future_frames != 8:
        raise QualityEvaluationError(
            f"held-out scoring target has {future_frames} != 8 future frames"
        )
    vae_ground_truth = (
        (
            decoded[:, :, -future_frames:].float().clamp(-1.0, 1.0)
            + 1.0
        )
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    raw_future = batch["rgb"][:, -future_frames:].permute(0, 2, 1, 3, 4)
    raw_ground_truth = (
        (raw_future.float().clamp(-1.0, 1.0) + 1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    if decoded.shape[2] < future_frames + 1 or batch["rgb"].shape[1] < (
        future_frames + 1
    ):
        raise QualityEvaluationError(
            "held-out scoring target lacks a history/future boundary frame"
        )
    vae_history_last = (
        (
            decoded[:, :, -(future_frames + 1) : -future_frames]
            .float()
            .clamp(-1.0, 1.0)
            + 1.0
        )
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    raw_history = batch["rgb"].permute(0, 2, 1, 3, 4)
    raw_history_last = (
        (
            raw_history[:, :, -(future_frames + 1) : -future_frames]
            .float()
            .clamp(-1.0, 1.0)
            + 1.0
        )
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    auxiliary = batch["auxiliary_target"].detach().cpu().to(torch.float16)
    if tuple(auxiliary.shape[1:]) != (64, 4, 24, 120):
        raise QualityEvaluationError(
            "held-out V-JEPA target is not [B,64,4,24,120]"
        )
    return {
        "video_clean": video_clean.detach().cpu().to(torch.float16),
        "auxiliary_clean": auxiliary,
        "ground_truth": raw_ground_truth,
        "vae_ground_truth": vae_ground_truth,
        "raw_history_last": raw_history_last,
        "vae_history_last": vae_history_last,
        "cached_rgb_input_sha256": _slice_hashes(batch["rgb"]),
        "cached_actions_input_sha256": _slice_hashes(batch["actions"]),
    }


def _artifact_rows(
    *,
    artifacts: Mapping[str, Any],
    scoring_video_clean: Any,
    scoring_auxiliary_clean: Any,
    scoring_ground_truth: Any,
    scoring_vae_ground_truth: Any,
    scoring_raw_history_last: Any,
    scoring_vae_history_last: Any,
    scoring_cached_rgb_input_sha256: Sequence[str],
    scoring_cached_actions_input_sha256: Sequence[str],
    arm: str,
    completed_updates: int,
    source: str,
    nfe: int,
    clip_indexes: Sequence[int],
    clip_ids: Sequence[str],
    observed_wan_calls: int,
    evaluation_split: str | None = None,
    frontier_selection_identity_sha256: str | None = None,
    lockbox_registration_identity_sha256: str | None = None,
    study_identity_sha256: str | None = None,
    arm_identity_sha256: str | None = None,
    stage_identity_sha256: str | None = None,
    training_git_commit: str | None = None,
    evaluator_git_commit: str | None = None,
    inference_code_compatibility_sha256: str | None = None,
    videox_runtime_identity_sha256: str | None = None,
    evaluation_world_size: int | None = None,
    evaluation_batch_size_per_rank: int | None = None,
    sampling_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    import torch

    infix = _source_infix(source)
    deployable_source = source not in ORACLE_SOURCES
    video_clean = scoring_video_clean
    auxiliary_clean = scoring_auxiliary_clean
    ground_truth = scoring_ground_truth
    vae_ground_truth = scoring_vae_ground_truth
    raw_history_last = scoring_raw_history_last
    vae_history_last = scoring_vae_history_last
    video_initial = artifacts["video_initial_state"]
    auxiliary_initial = artifacts["tf_initial_state"]
    auxiliary_noise = artifacts["tf_initial_noise"]
    video_final = artifacts[f"video_final{infix}_nfe_{nfe}"]
    auxiliary_final = artifacts[f"tf_final{infix}_nfe_{nfe}"]
    decoded = artifacts[f"decoded_future{infix}_nfe_{nfe}"]
    batch_size = len(clip_indexes)
    if (
        len(scoring_cached_rgb_input_sha256) != batch_size
        or len(scoring_cached_actions_input_sha256) != batch_size
    ):
        raise QualityEvaluationError(
            "cached RGB/action input-hash count differs from batch"
        )
    expected_batch_tensors = {
        "video_clean": video_clean,
        "tf_clean": auxiliary_clean,
        "ground_truth_future_uint8": ground_truth,
        "vae_ground_truth_future_uint8": vae_ground_truth,
        "raw_history_last_uint8": raw_history_last,
        "vae_history_last_uint8": vae_history_last,
        "video_initial_state": video_initial,
        "tf_initial_state": auxiliary_initial,
        "tf_initial_noise": auxiliary_noise,
        "video_final": video_final,
        "tf_final": auxiliary_final,
        "decoded_future": decoded,
    }
    for key, value in expected_batch_tensors.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
            raise QualityEvaluationError(
                f"{key} did not preserve all {batch_size} batch elements"
            )
    if video_final.shape != video_clean.shape:
        raise QualityEvaluationError(
            "generated video latent and held-out clean latent shapes differ"
        )
    if auxiliary_final.shape != auxiliary_clean.shape:
        raise QualityEvaluationError(
            "generated and held-out V-JEPA latent shapes differ"
        )
    if tuple(auxiliary_clean.shape[1:]) != (64, 4, 24, 120):
        raise QualityEvaluationError(
            "clean V-JEPA target shape is not [B,64,4,24,120]"
        )
    video_history = int(artifacts["history_latent_frames"].reshape(-1)[0])
    auxiliary_history = int(
        artifacts["auxiliary_history_latent_frames"].reshape(-1)[0]
    )
    if not 0 <= video_history < video_clean.shape[2]:
        raise QualityEvaluationError("video history does not leave a future")
    if auxiliary_history != 0:
        raise QualityEvaluationError("all four V-JEPA bins must be generated")
    teacher_calls = int(artifacts["online_teacher_call_count"].reshape(-1)[0])
    wan_calls = int(
        artifacts[f"wan_call_count{infix}_nfe_{nfe}"].reshape(-1)[0]
    )
    if teacher_calls != 0:
        raise QualityEvaluationError("quality sampler invoked online teacher")
    expected_deployment_mode = int(deployable_source)
    if (
        int(artifacts["deployment_mode"].reshape(-1)[0])
        != expected_deployment_mode
    ):
        raise QualityEvaluationError(
            "quality sampler entrypoint/deployment-mode label differs"
        )
    expected_clean_available = int(not deployable_source)
    if (
        int(artifacts["auxiliary_clean_available"].reshape(-1)[0])
        != expected_clean_available
    ):
        raise QualityEvaluationError(
            "quality sampler clean-target availability differs"
        )
    target_artifact_keys = {
        "video_clean",
        "tf_clean",
        "ground_truth_future_uint8",
    }
    if deployable_source:
        leaked = target_artifact_keys.intersection(artifacts)
        if leaked:
            raise QualityEvaluationError(
                "deployable quality sampler exposed unavailable targets: "
                f"{sorted(leaked)}"
            )
    else:
        missing_targets = target_artifact_keys - artifacts.keys()
        if missing_targets:
            raise QualityEvaluationError(
                f"oracle quality artifact lacks {sorted(missing_targets)}"
            )
        oracle_targets = {
            "video_clean": video_clean,
            "tf_clean": auxiliary_clean,
            # The model's ordinary full-clip artifact is the VAE
            # reconstruction, not the raw held-out video.  Raw RGB remains an
            # evaluator-owned target and is never passed into the sampler.
            "ground_truth_future_uint8": vae_ground_truth,
        }
        for key, target in oracle_targets.items():
            if not torch.equal(artifacts[key], target):
                raise QualityEvaluationError(
                    f"oracle artifact/scoring target differs: {key}"
                )
    if wan_calls != nfe or observed_wan_calls != nfe:
        raise QualityEvaluationError(
            f"quality Wan calls differ: artifact={wan_calls}, "
            f"hook={observed_wan_calls}, NFE={nfe}"
        )
    if int(artifacts["oracle_sources_are_leakage"].reshape(-1)[0]) != 1:
        raise QualityEvaluationError("oracle sources lack leakage label")
    source_codes = tuple(
        int(value)
        for value in artifacts[
            "evaluation_condition_source_codes"
        ].tolist()
    )
    if source_codes != (SOURCE_CODES[source],):
        raise QualityEvaluationError(
            f"quality source-code record differs: {source_codes}"
        )
    declared_nfe = tuple(
        int(value) for value in artifacts["evaluation_nfe_steps"].tolist()
    )
    if declared_nfe != (nfe,):
        raise QualityEvaluationError(
            f"quality NFE record differs: {declared_nfe}"
        )
    expected_sample_ids = tuple(
        int(value)
        for value in (
            clip_indexes if sampling_ids is None else sampling_ids
        )
    )
    recorded_sample_ids = tuple(
        int(value) for value in artifacts["sample_ids"].tolist()
    )
    if recorded_sample_ids != expected_sample_ids:
        raise QualityEvaluationError(
            f"artifact sample IDs differ: {recorded_sample_ids} != "
            f"{expected_sample_ids}"
        )
    if not torch.equal(auxiliary_initial, auxiliary_noise):
        raise QualityEvaluationError(
            "zero-history auxiliary initial state differs from noise"
        )
    if ground_truth.shape[2] != 8 or decoded.shape != ground_truth.shape:
        raise QualityEvaluationError("decoded future is not eight aligned frames")

    video_nmse = _per_sample_nmse(
        video_final, video_clean, video_history, "video"
    )
    auxiliary_nmse = _per_sample_nmse(
        auxiliary_final, auxiliary_clean, auxiliary_history, "auxiliary"
    )
    auxiliary_cosine = _per_sample_cosine(
        auxiliary_final, auxiliary_clean, auxiliary_history, "auxiliary"
    )
    decoded_metrics = _per_sample_decoded(
        decoded,
        ground_truth,
        prediction_history=raw_history_last,
        target_history=raw_history_last,
    )
    decoded_vs_vae = _per_sample_decoded(
        decoded,
        vae_ground_truth,
        prediction_history=raw_history_last,
        target_history=vae_history_last,
    )
    vae_vs_raw = _per_sample_decoded(
        vae_ground_truth,
        ground_truth,
        prediction_history=vae_history_last,
        target_history=raw_history_last,
    )
    hashes = {
        "video_clean_sha256": _slice_hashes(video_clean),
        "auxiliary_clean_sha256": _slice_hashes(auxiliary_clean),
        "ground_truth_sha256": _slice_hashes(ground_truth),
        "vae_ground_truth_sha256": _slice_hashes(vae_ground_truth),
        "raw_history_last_sha256": _slice_hashes(raw_history_last),
        "vae_history_last_sha256": _slice_hashes(vae_history_last),
        "cached_rgb_input_sha256": list(scoring_cached_rgb_input_sha256),
        "cached_actions_input_sha256": list(
            scoring_cached_actions_input_sha256
        ),
        "video_initial_state_sha256": _slice_hashes(video_initial),
        "auxiliary_initial_state_sha256": _slice_hashes(auxiliary_initial),
        "auxiliary_initial_noise_sha256": _slice_hashes(auxiliary_noise),
        "video_final_sha256": _slice_hashes(video_final),
        "auxiliary_final_sha256": _slice_hashes(auxiliary_final),
        "decoded_final_sha256": _slice_hashes(decoded),
    }
    state_gate = float(artifacts["effective_state_gate"].reshape(-1)[0])
    clock_gate = float(artifacts["effective_clock_gate"].reshape(-1)[0])
    if not math.isfinite(state_gate) or not math.isfinite(clock_gate):
        raise QualityEvaluationError("effective gate is non-finite")
    if arm in {"VPM", "A1"} and (state_gate != 0.0 or clock_gate != 0.0):
        raise QualityEvaluationError(
            f"{arm} no-injection gates must remain exactly zero"
        )
    rows = []
    for index, (clip_index, clip_id) in enumerate(zip(clip_indexes, clip_ids)):
        frontier_fields = (
            {}
            if evaluation_split is None
            else {
                "evaluation_split": evaluation_split,
                "frontier_selection_identity_sha256": (
                    frontier_selection_identity_sha256
                ),
                "lockbox_registration_identity_sha256": (
                    lockbox_registration_identity_sha256
                ),
                "study_identity_sha256": study_identity_sha256,
                "arm_identity_sha256": arm_identity_sha256,
                "stage_identity_sha256": stage_identity_sha256,
                "training_git_commit": training_git_commit,
                "evaluator_git_commit": evaluator_git_commit,
                "inference_code_compatibility_sha256": (
                    inference_code_compatibility_sha256
                ),
                "videox_runtime_identity_sha256": (
                    videox_runtime_identity_sha256
                ),
                "evaluation_world_size": evaluation_world_size,
                "evaluation_batch_size_per_rank": (
                    evaluation_batch_size_per_rank
                ),
                "sampling_id": expected_sample_ids[index],
                "sampling_namespace": evaluation_split,
            }
        )
        row = _identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "vjepa2_controlled_study_quality_clip",
                "arm_code": arm,
                "completed_updates": completed_updates,
                "clip_index": int(clip_index),
                "clip_id": str(clip_id),
                **frontier_fields,
                "source": source,
                "oracle_leakage": source in ORACLE_SOURCES,
                "deployable_evidence": deployable_source,
                "sampler_entrypoint": (
                    "DualExplicitActionDiTModel.sample_future_deployable"
                    if deployable_source
                    else "DualExplicitActionDiTModel._sample_future"
                ),
                "clean_future_or_auxiliary_passed_to_sampler": (
                    not deployable_source
                ),
                "nfe": nfe,
                "video_history_latent_frames": video_history,
                "auxiliary_history_latent_frames": auxiliary_history,
                "online_teacher_call_count": 0,
                "actual_wan_call_count": wan_calls,
                "effective_state_gate": state_gate,
                "effective_clock_gate": clock_gate,
                "metrics": {
                    "video_future_nmse": video_nmse[index],
                    "auxiliary_future_nmse": auxiliary_nmse[index],
                    "auxiliary_future_cosine_similarity": auxiliary_cosine[
                        index
                    ],
                    "decoded_mse_unit_range": decoded_metrics[
                        "decoded_mse_unit_range"
                    ][index],
                    "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][index],
                    "decoded_temporal_difference_mse_unit_range": (
                        decoded_metrics[
                            "decoded_temporal_difference_mse_unit_range"
                        ][index]
                    ),
                },
                "diagnostic_metrics": {
                    "prediction_vs_vae_reconstruction_mse_unit_range": (
                        decoded_vs_vae["decoded_mse_unit_range"][index]
                    ),
                    "prediction_vs_vae_reconstruction_psnr_db": (
                        decoded_vs_vae["decoded_psnr_db"][index]
                    ),
                    "prediction_vs_vae_reconstruction_temporal_mse_unit_range": (
                        decoded_vs_vae[
                            "decoded_temporal_difference_mse_unit_range"
                        ][index]
                    ),
                    "vae_reconstruction_vs_raw_mse_unit_range": (
                        vae_vs_raw["decoded_mse_unit_range"][index]
                    ),
                    "vae_reconstruction_vs_raw_psnr_db": (
                        vae_vs_raw["decoded_psnr_db"][index]
                    ),
                    "vae_reconstruction_vs_raw_temporal_mse_unit_range": (
                        vae_vs_raw[
                            "decoded_temporal_difference_mse_unit_range"
                        ][index]
                    ),
                },
                "tensor_sha256": {
                    key: values[index] for key, values in hashes.items()
                },
                "perceptual_metric": {
                    "available": False,
                    "reason": (
                        "no LPIPS/video-perceptual checkpoint is pinned in the "
                        "immutable study; implicit pretrained downloads forbidden"
                    ),
                },
            }
        )
        rows.append(row)
    return rows


def _expected_rank_indexes(
    rank: int, world_size: int, clip_count: int = EXPECTED_TEST_CLIPS
) -> list[int]:
    return list(range(rank, clip_count, world_size))


def _validate_global_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    completed_updates: int,
    grid: Sequence[tuple[str, int]],
    descriptors: Sequence[Mapping[str, Any]],
    expected_clips: int = EXPECTED_TEST_CLIPS,
    evaluation_split: str | None = None,
    frontier_selection_identity_sha256: str | None = None,
    lockbox_registration_identity_sha256: str | None = None,
) -> dict[str, Any]:
    expected_keys = {
        (index, source, nfe)
        for index in range(expected_clips)
        for source, nfe in grid
    }
    observed: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not _identity_valid(row):
            raise QualityEvaluationError("quality row identity SHA-256 is invalid")
        key = (
            int(row.get("clip_index", -1)),
            str(row.get("source", "")),
            int(row.get("nfe", -1)),
        )
        if key in observed:
            raise QualityEvaluationError(f"duplicate quality row {key}")
        if (
            key not in expected_keys
            or row.get("arm_code") != arm
            or row.get("completed_updates") != completed_updates
            or row.get("clip_id")
            != descriptors[key[0]].get("clip_id")
            or row.get("oracle_leakage")
            != (key[1] in ORACLE_SOURCES)
            or row.get("deployable_evidence")
            != (key[1] not in ORACLE_SOURCES)
            or row.get("sampler_entrypoint")
            != (
                "DualExplicitActionDiTModel._sample_future"
                if key[1] in ORACLE_SOURCES
                else "DualExplicitActionDiTModel.sample_future_deployable"
            )
            or row.get("clean_future_or_auxiliary_passed_to_sampler")
            != (key[1] in ORACLE_SOURCES)
            or row.get("online_teacher_call_count") != 0
            or row.get("actual_wan_call_count") != key[2]
            or row.get("auxiliary_history_latent_frames") != 0
            or (
                evaluation_split is not None
                and row.get("evaluation_split") != evaluation_split
            )
            or (
                evaluation_split in {"test", "lockbox"}
                and row.get("frontier_selection_identity_sha256")
                != frontier_selection_identity_sha256
            )
            or (
                evaluation_split == "lockbox"
                and row.get("lockbox_registration_identity_sha256")
                != lockbox_registration_identity_sha256
            )
            or (
                evaluation_split is not None
                and (
                    row.get("sampling_namespace") != evaluation_split
                    or row.get("sampling_id")
                    != FRONTIER_SAMPLE_ID_OFFSETS[evaluation_split] + key[0]
                )
            )
        ):
            raise QualityEvaluationError(
                f"quality row provenance/counters differ: {key}"
            )
        metrics = row.get("metrics")
        if (
            not isinstance(metrics, dict)
            or len(metrics) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise QualityEvaluationError(f"quality metrics are invalid: {key}")
        diagnostics = row.get("diagnostic_metrics")
        if (
            not isinstance(diagnostics, dict)
            or len(diagnostics) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in diagnostics.values()
            )
        ):
            raise QualityEvaluationError(
                f"quality diagnostic metrics are invalid: {key}"
            )
        tensor_hashes = row.get("tensor_sha256")
        expected_tensor_hashes = {
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
        if (
            not isinstance(tensor_hashes, dict)
            or set(tensor_hashes) != expected_tensor_hashes
            or any(
                not isinstance(value, str)
                or SHA256_RE.fullmatch(value) is None
                for value in tensor_hashes.values()
            )
        ):
            raise QualityEvaluationError(
                f"quality tensor hashes are invalid: {key}"
            )
        perceptual = row.get("perceptual_metric")
        if (
            not isinstance(perceptual, dict)
            or perceptual.get("available") is not False
        ):
            raise QualityEvaluationError(
                f"quality perceptual-metric policy differs: {key}"
            )
        observed[key] = row
    missing = expected_keys - observed.keys()
    extra = observed.keys() - expected_keys
    if missing or extra:
        raise QualityEvaluationError(
            f"quality inventory differs: missing={len(missing)}, extra={len(extra)}"
        )

    identity_fields = (
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
    for clip_index in range(expected_clips):
        clip_rows = [
            observed[(clip_index, source, nfe)] for source, nfe in grid
        ]
        reference_hashes = clip_rows[0]["tensor_sha256"]
        for row in clip_rows[1:]:
            if any(
                row["tensor_sha256"][field] != reference_hashes[field]
                for field in identity_fields
            ):
                raise QualityEvaluationError(
                    f"per-clip clean/noise pairing differs for clip {clip_index}"
                )

    # Causal no-op controls are checked from exact final tensor hashes, never
    # inferred merely from configuration flags.
    output_fields = (
        "video_final_sha256",
        "auxiliary_final_sha256",
        "decoded_final_sha256",
    )
    available_by_nfe: dict[int, list[str]] = {}
    for source, nfe in grid:
        available_by_nfe.setdefault(nfe, []).append(source)
    for clip_index in range(expected_clips):
        for nfe, available_sources in available_by_nfe.items():
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
            reference = observed[
                (clip_index, compare_sources[0], nfe)
            ]["tensor_sha256"]
            for source in compare_sources[1:]:
                candidate = observed[
                    (clip_index, source, nfe)
                ]["tensor_sha256"]
                if any(
                    candidate[field] != reference[field]
                    for field in output_fields
                ):
                    raise QualityEvaluationError(
                        f"{arm} causal source no-op failed: clip={clip_index}, "
                        f"NFE={nfe}, source={source}"
                    )
    return {
        "record_count": len(rows),
        "clip_count": expected_clips,
        "grid_count": len(grid),
        "all_clip_source_nfe_keys_present_once": True,
        "per_clip_clean_and_initial_noise_pairing_exact": True,
        "causal_no_op_identities_passed": True,
    }


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if args.completed_updates not in PRIMARY_UPDATES:
        raise QualityEvaluationError("completed update is not a primary milestone")
    if args.batch_size_per_rank < 1:
        raise QualityEvaluationError("batch size per rank must be positive")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    inputs = _validate_inputs(
        args,
        verify_snapshot_sha256=(rank == 0),
        verify_cache_arrays=(rank == 0),
    )
    if world_size != EXPECTED_WORLD_SIZE:
        raise QualityEvaluationError(
            f"quality evaluation requires {EXPECTED_WORLD_SIZE} ranks"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name.upper():
        raise QualityEvaluationError(
            f"quality evaluation requires B200, found {properties.name}"
        )
    assigned_indexes = _expected_rank_indexes(
        rank, world_size, inputs["expected_clips"]
    )
    if len(assigned_indexes) % args.batch_size_per_rank:
        raise QualityEvaluationError(
            "each rank's clip count must divide batch size exactly"
        )
    if rank == 0:
        if inputs["output_dir"].exists():
            raise QualityEvaluationError(
                f"fresh quality output already exists: {inputs['output_dir']}"
            )
        inputs["output_dir"].mkdir(parents=True, mode=0o700)
    dist.barrier()
    output_dir = _canonical_directory(
        inputs["output_dir"], "quality output directory"
    )
    grid = inputs["grid"]
    model, dataset, _config = _load_model_and_dataset(inputs, device=device)
    # Preserve every batch element in the transient CPU evidence.  The
    # evaluator immediately reduces it to per-clip metrics/hashes and never
    # persists the large tensors themselves.
    model.artifact_batch_limit = None
    model.capture_latent_trajectories = False

    rows: list[dict[str, Any]] = []
    actual_backbone_invocations = 0
    hook_calls = 0

    def count_wan_calls(_module, _inputs, _output):
        nonlocal hook_calls
        hook_calls += 1

    hook = model.forward_model.register_forward_hook(count_wan_calls)
    try:
        for start in range(0, len(assigned_indexes), args.batch_size_per_rank):
            clip_indexes = assigned_indexes[
                start : start + args.batch_size_per_rank
            ]
            samples = [dataset[index] for index in clip_indexes]
            observed_indexes = [
                int(sample["clip_index"].item()) for sample in samples
            ]
            if observed_indexes != clip_indexes:
                raise QualityEvaluationError(
                    "fixed-test dataset substituted a different clip: "
                    f"{observed_indexes} != {clip_indexes}"
                )
            batch = _move_batch(samples, device)
            clip_ids = [
                str(inputs["descriptors"][index]["clip_id"])
                for index in clip_indexes
            ]
            scoring_targets = _prepare_scoring_targets(model, batch)
            history_rgb = batch["rgb"][:, : model.num_history_frames]
            sampling_ids = batch["clip_index"]
            if inputs["frontier_mode"]:
                sampling_ids = sampling_ids + FRONTIER_SAMPLE_ID_OFFSETS[
                    inputs["evaluation_split"]
                ]
            for source, nfe in grid:
                model.evaluation_condition_sources = (source,)
                model.evaluation_nfe_steps = (nfe,)
                model.viz_num_steps = nfe
                model.capture_latent_trajectories = False
                hook_calls = 0
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    if source in ORACLE_SOURCES:
                        model._sample_future(
                            batch["rgb"],
                            batch["actions"],
                            morphology_index=batch["morphology_index"],
                            auxiliary_target=batch["auxiliary_target"],
                            collect_artifacts=True,
                            deployment_mode=False,
                            sample_ids=sampling_ids,
                        )
                    else:
                        model.sample_future_deployable(
                            history_rgb,
                            batch["actions"],
                            morphology_index=batch["morphology_index"],
                            collect_artifacts=True,
                            sample_ids=sampling_ids,
                        )
                artifacts = model.pop_visualization_artifacts()
                if not isinstance(artifacts, Mapping):
                    raise QualityEvaluationError(
                        "quality sampler did not expose artifacts"
                    )
                rows.extend(
                    _artifact_rows(
                        artifacts=artifacts,
                        scoring_video_clean=scoring_targets["video_clean"],
                        scoring_auxiliary_clean=scoring_targets[
                            "auxiliary_clean"
                        ],
                        scoring_ground_truth=scoring_targets["ground_truth"],
                        scoring_vae_ground_truth=scoring_targets[
                            "vae_ground_truth"
                        ],
                        scoring_raw_history_last=scoring_targets[
                            "raw_history_last"
                        ],
                        scoring_vae_history_last=scoring_targets[
                            "vae_history_last"
                        ],
                        scoring_cached_rgb_input_sha256=scoring_targets[
                            "cached_rgb_input_sha256"
                        ],
                        scoring_cached_actions_input_sha256=scoring_targets[
                            "cached_actions_input_sha256"
                        ],
                        arm=inputs["arm_code"],
                        completed_updates=args.completed_updates,
                        source=source,
                        nfe=nfe,
                        clip_indexes=clip_indexes,
                        clip_ids=clip_ids,
                        observed_wan_calls=hook_calls,
                        evaluation_split=(
                            inputs["evaluation_split"]
                            if inputs["frontier_mode"]
                            else None
                        ),
                        frontier_selection_identity_sha256=inputs[
                            "frontier_selection_identity_sha256"
                        ],
                        lockbox_registration_identity_sha256=inputs[
                            "lockbox_registration_identity_sha256"
                        ],
                        study_identity_sha256=inputs["study_manifest"][
                            "identity_sha256"
                        ],
                        arm_identity_sha256=inputs["arm_manifest"][
                            "identity_sha256"
                        ],
                        stage_identity_sha256=inputs["stage_manifest"][
                            "identity_sha256"
                        ],
                        training_git_commit=inputs["training_git_commit"],
                        evaluator_git_commit=inputs["evaluator_git_commit"],
                        inference_code_compatibility_sha256=hashlib.sha256(
                            _canonical_json(
                                inputs["inference_code_compatibility"]
                            )
                        ).hexdigest(),
                        videox_runtime_identity_sha256=hashlib.sha256(
                            _canonical_json(inputs["videox_runtime"])
                        ).hexdigest(),
                        evaluation_world_size=world_size,
                        evaluation_batch_size_per_rank=(
                            args.batch_size_per_rank
                        ),
                        sampling_ids=[
                            int(value)
                            for value in sampling_ids.detach().cpu().tolist()
                        ],
                    )
                )
                actual_backbone_invocations += hook_calls
                del artifacts
            del scoring_targets, batch, samples
    finally:
        hook.remove()

    expected_rows = len(assigned_indexes) * len(grid)
    if len(rows) != expected_rows:
        raise QualityEvaluationError(
            f"rank {rank} emitted {len(rows)} != {expected_rows} rows"
        )
    expected_invocations = (
        len(assigned_indexes)
        // args.batch_size_per_rank
        * sum(nfe for _source, nfe in grid)
    )
    if actual_backbone_invocations != expected_invocations:
        raise QualityEvaluationError(
            f"rank {rank} Wan invocations {actual_backbone_invocations} "
            f"!= {expected_invocations}"
        )
    rows_path = output_dir / f"rank_{rank:03d}.jsonl"
    rows_bytes = b"".join(
        _canonical_json(row) + b"\n" for row in rows
    )
    _exclusive_bytes(rows_path, rows_bytes)
    manifest_inputs = {
        "resolved_config": {
            "path": str(inputs["resolved_config"]),
            "sha256": _sha256(inputs["resolved_config"]),
        },
        "snapshot": {
            "path": str(inputs["snapshot"]),
            "sha256": inputs["snapshot_sha256"],
        },
        "arm_manifest": {
            "path": str(inputs["arm_manifest_path"]),
            "sha256": _sha256(inputs["arm_manifest_path"]),
        },
        "study_manifest": {
            "path": str(inputs["study_manifest_path"]),
            "sha256": _sha256(inputs["study_manifest_path"]),
        },
        "stage_manifest": {
            "path": str(inputs["stage_manifest_path"]),
            "sha256": _sha256(inputs["stage_manifest_path"]),
        },
        "stage_outcome": {
            "path": str(inputs["stage_outcome_path"]),
            "sha256": _sha256(inputs["stage_outcome_path"]),
            "identity_sha256": inputs["stage_outcome"]["identity_sha256"],
        },
        (
            "evaluation_clip_manifest"
            if inputs["frontier_mode"]
            else "test_clip_manifest"
        ): {
            "path": str(inputs["evaluation_manifest"]),
            "sha256": _sha256(inputs["evaluation_manifest"]),
        },
        (
            "evaluation_cache_metadata"
            if inputs["frontier_mode"]
            else "test_cache_metadata"
        ): {
            "path": str(inputs["cache_metadata"]),
            "sha256": _sha256(inputs["cache_metadata"]),
        },
        (
            "evaluation_cache_arrays"
            if inputs["frontier_mode"]
            else "test_cache_arrays"
        ): {
            name: {
                "path": record["path"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
            for name, record in inputs["cache_arrays"].items()
        },
    }
    if inputs["evaluation_split"] == "lockbox":
        manifest_inputs["lockbox_registration"] = {
            "identity_sha256": inputs[
                "lockbox_registration_identity_sha256"
            ],
            "manifest": inputs["lockbox_registration"]["manifest"],
            "cache": inputs["lockbox_registration"]["cache"],
            "episode_isolation": inputs["lockbox_registration"][
                "episode_isolation"
            ],
            "rank0_deterministic_construction_reverified": (
                rank == 0
                and inputs["lockbox_validation"][
                    "deterministic_construction_reverified"
                ]
            ),
        }
    frontier_manifest_fields = (
        {}
        if not inputs["frontier_mode"]
        else {
            "evaluation_split": inputs["evaluation_split"],
            "frontier_mode": True,
            "training_git_commit": inputs["training_git_commit"],
            "evaluator_git_commit": inputs["evaluator_git_commit"],
            "inference_code_compatibility": inputs[
                "inference_code_compatibility"
            ],
            "videox_runtime": inputs["videox_runtime"],
            "frontier_selection_identity_sha256": inputs[
                "frontier_selection_identity_sha256"
            ],
            "lockbox_registration_identity_sha256": inputs[
                "lockbox_registration_identity_sha256"
            ],
            "evaluation_dataset_override": (
                None
                if inputs["evaluation_split"] != "lockbox"
                else {
                    "base_config": "viz_dataset from pinned resolved config",
                    "only_overridden_fields": [
                        "datasets.ABC.clip_manifest",
                        "datasets.ABC.cache_metadata",
                    ],
                    "clip_manifest": str(inputs["evaluation_manifest"]),
                    "cache_metadata": str(inputs["cache_metadata"]),
                    "image_augmentation": False,
                    "online_teacher": False,
                }
            ),
            "frontier_selection": (
                None
                if inputs["frontier_selection_path"] is None
                else {
                    "path": str(inputs["frontier_selection_path"]),
                    "sha256": _sha256(inputs["frontier_selection_path"]),
                    "identity_sha256": inputs[
                        "frontier_selection_identity_sha256"
                    ],
                }
            ),
        }
    )
    rank_manifest = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_quality_rank",
            "created_at_utc": _now(),
            "arm_code": inputs["arm_code"],
            "arm_identity_sha256": inputs["arm_manifest"]["identity_sha256"],
            "study_identity_sha256": inputs["study_manifest"]["identity_sha256"],
            "stage_identity_sha256": inputs["stage_manifest"]["identity_sha256"],
            "stage_outcome_identity_sha256": inputs["stage_outcome"][
                "identity_sha256"
            ],
            "git_commit": args.expected_commit,
            "completed_updates": args.completed_updates,
            "rank": rank,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank,
            "assigned_clip_indexes": assigned_indexes,
            "grid": [
                {"source": source, "nfe": nfe} for source, nfe in grid
            ],
            "rows": {
                "path": str(rows_path),
                "sha256": hashlib.sha256(rows_bytes).hexdigest(),
                "bytes": len(rows_bytes),
                "count": len(rows),
            },
            "actual_wan_backbone_invocations": actual_backbone_invocations,
            "online_teacher_call_count": 0,
            "inputs": manifest_inputs,
            **frontier_manifest_fields,
            "device": {
                "name": properties.name,
                "local_rank": local_rank,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "metric_scope": {
                "reconstruction_metrics_only": True,
                "primary_decoded_target": "cached raw held-out future RGB",
                "vae_reconstruction_target_is_diagnostic_only": True,
                "temporal_metric_includes_history_to_first_future_boundary": (
                    True
                ),
                "lpips_or_video_perceptual_metric": False,
                "reason": (
                    "no perceptual checkpoint is pinned; implicit pretrained "
                    "downloads are forbidden"
                ),
            },
            "oracle_sources_are_leakage_only": True,
            "sigma_convention": "sigma=1 noise, sigma=0 clean",
        }
    )
    rank_manifest_path = output_dir / f"rank_{rank:03d}_manifest.json"
    _exclusive_json(rank_manifest_path, rank_manifest)
    dist.barrier()

    if rank == 0:
        all_rows: list[dict[str, Any]] = []
        rank_records: dict[str, Any] = {}
        for other_rank in range(world_size):
            other_manifest_path = _canonical_file(
                output_dir / f"rank_{other_rank:03d}_manifest.json",
                f"quality rank {other_rank} manifest",
            )
            other_manifest = _read_json(
                other_manifest_path, f"quality rank {other_rank} manifest"
            )
            other_rows_path = _canonical_file(
                output_dir / f"rank_{other_rank:03d}.jsonl",
                f"quality rank {other_rank} rows",
            )
            if (
                not _identity_valid(other_manifest)
                or other_manifest.get("rank") != other_rank
                or other_manifest.get("world_size") != world_size
                or other_manifest.get("arm_code") != inputs["arm_code"]
                or other_manifest.get("completed_updates")
                != args.completed_updates
                or other_manifest.get("stage_outcome_identity_sha256")
                != inputs["stage_outcome"]["identity_sha256"]
                or other_manifest.get("assigned_clip_indexes")
                != _expected_rank_indexes(
                    other_rank, world_size, inputs["expected_clips"]
                )
                or (
                    inputs["frontier_mode"]
                    and (
                        other_manifest.get("evaluation_split")
                        != inputs["evaluation_split"]
                        or other_manifest.get("frontier_mode") is not True
                        or other_manifest.get("training_git_commit")
                        != inputs["training_git_commit"]
                        or other_manifest.get("evaluator_git_commit")
                        != inputs["evaluator_git_commit"]
                        or other_manifest.get(
                            "inference_code_compatibility"
                        )
                        != inputs["inference_code_compatibility"]
                        or other_manifest.get("videox_runtime")
                        != inputs["videox_runtime"]
                        or other_manifest.get(
                            "frontier_selection_identity_sha256"
                        )
                        != inputs["frontier_selection_identity_sha256"]
                        or other_manifest.get(
                            "lockbox_registration_identity_sha256"
                        )
                        != inputs["lockbox_registration_identity_sha256"]
                    )
                )
                or other_manifest.get("rows", {}).get("path")
                != str(other_rows_path)
                or other_manifest.get("rows", {}).get("sha256")
                != _sha256(other_rows_path)
            ):
                raise QualityEvaluationError(
                    f"quality rank {other_rank} evidence differs"
                )
            rank_rows = _read_jsonl(
                other_rows_path, f"quality rank {other_rank} rows"
            )
            if len(rank_rows) != other_manifest["rows"]["count"]:
                raise QualityEvaluationError(
                    f"quality rank {other_rank} row count differs"
                )
            all_rows.extend(rank_rows)
            rank_records[str(other_rank)] = {
                "manifest_path": str(other_manifest_path),
                "manifest_sha256": _sha256(other_manifest_path),
                "manifest_identity_sha256": other_manifest["identity_sha256"],
                "rows_path": str(other_rows_path),
                "rows_sha256": other_manifest["rows"]["sha256"],
                "row_count": len(rank_rows),
            }
        validation = _validate_global_rows(
            all_rows,
            arm=inputs["arm_code"],
            completed_updates=args.completed_updates,
            grid=grid,
            descriptors=inputs["descriptors"],
            expected_clips=inputs["expected_clips"],
            evaluation_split=(
                inputs["evaluation_split"]
                if inputs["frontier_mode"]
                else None
            ),
            frontier_selection_identity_sha256=inputs[
                "frontier_selection_identity_sha256"
            ],
            lockbox_registration_identity_sha256=inputs[
                "lockbox_registration_identity_sha256"
            ],
        )
        inventory = _identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "vjepa2_controlled_study_quality_inventory",
                "created_at_utc": _now(),
                "complete": True,
                "arm_code": inputs["arm_code"],
                "arm_identity_sha256": inputs["arm_manifest"]["identity_sha256"],
                "study_identity_sha256": inputs["study_manifest"][
                    "identity_sha256"
                ],
                "stage_identity_sha256": inputs["stage_manifest"][
                    "identity_sha256"
                ],
                "stage_outcome_identity_sha256": inputs["stage_outcome"][
                    "identity_sha256"
                ],
                "git_commit": args.expected_commit,
                "completed_updates": args.completed_updates,
                "world_size": world_size,
                "batch_size_per_rank": args.batch_size_per_rank,
                "clip_count": inputs["expected_clips"],
                **(
                    {}
                    if not inputs["frontier_mode"]
                    else {
                        "evaluation_split": inputs["evaluation_split"],
                        "frontier_mode": True,
                        "training_git_commit": inputs["training_git_commit"],
                        "evaluator_git_commit": inputs[
                            "evaluator_git_commit"
                        ],
                        "inference_code_compatibility": inputs[
                            "inference_code_compatibility"
                        ],
                        "videox_runtime": inputs["videox_runtime"],
                        "frontier_selection_identity_sha256": inputs[
                            "frontier_selection_identity_sha256"
                        ],
                        "lockbox_registration_identity_sha256": inputs[
                            "lockbox_registration_identity_sha256"
                        ],
                    }
                ),
                "grid": [
                    {"source": source, "nfe": nfe}
                    for source, nfe in grid
                ],
                "expected_record_count": inputs["expected_clips"] * len(grid),
                "observed_record_count": len(all_rows),
                "rank_evidence": rank_records,
                "validation": validation,
                "metric_scope": {
                    "reconstruction_metrics_only": True,
                    "primary_decoded_target": (
                        "cached raw held-out future RGB"
                    ),
                    "vae_reconstruction_target_is_diagnostic_only": True,
                    "temporal_metric_includes_history_to_first_future_boundary": (
                        True
                    ),
                    "perceptual_metric_available": False,
                    "claim_restriction": (
                        "latent NMSE/cosine and decoded raw-video "
                        "pixel/temporal reconstruction metrics only"
                    ),
                },
                "oracle_sources_are_leakage_only": True,
                "trainer_visualization_is_diagnostic_only": True,
                "input_cache_integrity": {
                    "rank0_rehashed_full_target_rgb_action_arrays": True,
                    "arrays": {
                        name: {
                            "path": record["path"],
                            "sha256": record["sha256"],
                            "bytes": record["bytes"],
                        }
                        for name, record in inputs["cache_arrays"].items()
                    },
                },
                "lockbox_integrity": (
                    None
                    if inputs["evaluation_split"] != "lockbox"
                    else {
                        "registration_identity_sha256": inputs[
                            "lockbox_registration_identity_sha256"
                        ],
                        "deterministic_next_unused_construction_reverified": (
                            inputs["lockbox_validation"][
                                "deterministic_construction_reverified"
                            ]
                        ),
                        "episode_isolation_verified": inputs[
                            "lockbox_validation"
                        ]["episode_isolation_verified"],
                        "all_cache_arrays_fully_rehashed_by_rank0": True,
                    }
                ),
                "sigma_convention": "sigma=1 noise, sigma=0 clean",
            }
        )
        _exclusive_json(output_dir / "inventory.json", inventory)
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "arm": inputs["arm_code"],
                    "completed_updates": args.completed_updates,
                    "clips": inputs["expected_clips"],
                    "grid_count": len(grid),
                    "records": inputs["expected_clips"] * len(grid),
                    "output": str(output_dir / "inventory.json"),
                },
                sort_keys=True,
            )
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="immutable training/checkpoint commit recorded by the study",
    )
    parser.add_argument(
        "--evaluator-commit",
        default=None,
        help=(
            "clean evaluator checkout commit; defaults to --expected-commit "
            "for the legacy single-commit protocol"
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--stage-manifest", required=True)
    parser.add_argument("--stage-outcome", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--completed-updates", type=int, required=True)
    parser.add_argument(
        "--batch-size-per-rank",
        type=int,
        default=DEFAULT_BATCH_SIZE_PER_RANK,
    )
    parser.add_argument(
        "--frontier-split",
        choices=("validation", "lockbox"),
        default=None,
        help=(
            "opt-in NFE-frontier protocol; validation emits the full "
            "autonomous grid, lockbox emits only a frozen selected pair"
        ),
    )
    parser.add_argument(
        "--frontier-selection",
        default=None,
        help="frozen validation selection; required only for frontier lockbox",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return command_evaluate(args)
    except QualityEvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
