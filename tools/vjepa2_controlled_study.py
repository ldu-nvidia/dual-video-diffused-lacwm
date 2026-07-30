#!/usr/bin/env python3
"""Immutable contract and provenance helpers for the V-JEPA 2.1 study.

This helper deliberately does not reuse or modify any historical TF-screen
contract.  It defines a new five-arm experiment:

``V0``
    The original explicit-action LACWM (architecture/reference baseline).
``VPM``
    Parameter-matched dual wrapper with zero auxiliary-loss weight and no
    auxiliary state/clock entering the video trunk.
``A1``
    V-JEPA auxiliary denoising loss only; the auxiliary head receives its own
    noisy state and raw clock, but neither enters the video trunk.
``J0``
    Joint denoising with aligned video/auxiliary clocks.
``J1``
    Joint denoising with V-JEPA leading by one logit unit.

The V-JEPA teacher is an *offline target extractor only*.  Training and
autonomous inference consume cached, PCA-whitened targets or generated
auxiliary states.  A teacher invocation during an inference benchmark is a
contract violation.

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
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
VJEPA_SOURCE_COMMIT = "45d025f636dfc58fc2426905fc4a1ab755b1c3e5"
VJEPA_MODEL_NAME = "vjepa2_1_vit_base_384"
VJEPA_CHECKPOINT_KEY = "ema_encoder"
VJEPA_CHECKPOINT_BYTES = 1_664_223_428
VJEPA_SOURCE_DIM = 768
VJEPA_TARGET_SHAPE = (64, 4, 24, 120)
VJEPA_FRAME_MAP = (0, 0, 0, 0, *range(1, 13))
SIGMA_CONVENTION = "sigma=1 noise, sigma=0 clean"
EXPECTED_SPLIT_COUNTS = {"train": 512, "val": 64, "test": 128}
EXPECTED_CACHE_BUILD_SEED = 20_260_729
EXPECTED_CLIPS_PER_EPISODE = 1

# This is intentionally independent of ``tools/vjepa2_phase_gate.py``.  A
# producer and its consumer sharing one mutable source of truth would let a
# semantic regression bless itself.
EXPECTED_PHASE_J0_CONTRACT = {
    "model_target": (
        "lam.dual_explicit_action_dit_model.DualExplicitActionDiTModel"
    ),
    "dual_enabled": True,
    "auxiliary_history_mode": "diffuse_all",
    "tf_channels": 64,
    "condition_mode": "matched",
    "condition_on_tf": True,
    "condition_on_tf_clock": True,
    "schedule_mode": "aligned",
    "tf_lead_logit": 0.0,
    "tf_loss_weight": 1.0,
    "state_gate_init": 0.0,
    "state_gate_trainable": True,
    "clock_gate_init": 0.0,
    "clock_gate_trainable": True,
    "head_condition_on_tf_clock": True,
    "video_only_control": False,
    "parameter_matched_control": False,
    "time_frequency_transform": None,
}
EXPECTED_PHASE_RESET_PREFIXES = (
    "inverse_model",
    "rgb_pos_embed",
    "action_decoder",
    "action_pos_embed",
    "action_pool",
    "morphology_tokens",
    "forward_model.tf_token_adapter",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_velocity_head",
)
# Independent consumer copy of the producer's exact-key warm-start contract.
# Do not import this from vjepa2_phase_gate.py: the validator must be able to
# reject a producer regression rather than sharing its mutable source of truth.
EXPECTED_PHASE_WARMSTART_MISSING_KEYS = (
    "action_encoder.net.0.bias",
    "action_encoder.net.0.weight",
    "action_encoder.net.2.bias",
    "action_encoder.net.2.weight",
    "action_encoder.net.4.bias",
    "action_encoder.net.4.weight",
)
EXPECTED_PHASE_BATCH_SHAPES = {
    "rgb": [1, 13, 3, 180, 960],
    "actions": [1, 13, 5, 157],
    "mask": [1, 13],
    "auxiliary_target": [1, 64, 4, 24, 120],
    "clip_index": [1],
    "morphology_index": [1],
}
EXPECTED_PHASE_GRADIENT_GROUPS = {
    "auxiliary_state_gate",
    "auxiliary_state_projection",
    "auxiliary_state_norm",
    "auxiliary_clock_gate",
    "auxiliary_clock_network",
    "auxiliary_velocity_head",
    "action_control",
    "shared_video_lora",
}

SEED = 1234
TOTAL_UPDATES = 1000
WARMUP_UPDATES = 50
COMPLETED_UPDATE_MILESTONES = (1, 50, 100, 200, 400, 800, 1000)
# The 600-update endpoint bounds every allocation to at most 200 new updates.
# It is a resumability stage, not a primary scientific milestone.
STAGE_ENDPOINTS = (1, 50, 100, 200, 400, 600, 800, 1000)
INFERENCE_NFE = (1, 2, 4, 6, 8, 12, 20)
EVALUATION_NOISE_SEED = 20_260_729
EVALUATION_SOURCES = (
    "autonomous",
    "off",
    "autonomous_shuffled",
    "oracle_matched",
    "oracle_shuffled",
)
DEPLOYABLE_EVALUATION_SOURCES = (
    "autonomous",
    "off",
    "autonomous_shuffled",
)
ORACLE_EVALUATION_SOURCES = ("oracle_matched", "oracle_shuffled")

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[0-9]+(?:[_;][A-Za-z0-9_.%+-]+)?$")
ACTIVE_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")


# Array order and every value below are part of the immutable study contract.
# VPM/A1/J0/J1 intentionally have the same module schema and trainable-parameter
# flags.  The condition flags, schedule, and loss coefficient are interventions.
ARMS: tuple[dict[str, Any], ...] = (
    {
        "code": "V0",
        "name": "v0_original_explicit_action",
        "selector_kind": "baseline",
        "dual_enabled": False,
        "condition_mode": None,
        "condition_on_auxiliary_state": False,
        "condition_on_auxiliary_clock": False,
        "schedule_mode": None,
        "auxiliary_lead_logit": None,
        "auxiliary_loss_weight": 0.0,
        "parameter_matched_control": False,
        "research_role": "original architecture/reference baseline",
    },
    {
        "code": "VPM",
        "name": "vpm_parameter_matched_video",
        "selector_kind": "dual",
        "dual_enabled": True,
        "condition_mode": "off",
        "condition_on_auxiliary_state": False,
        "condition_on_auxiliary_clock": False,
        "schedule_mode": "aligned",
        "auxiliary_lead_logit": 0.0,
        "auxiliary_loss_weight": 0.0,
        "parameter_matched_control": True,
        "research_role": "parameter-matched video-only objective",
    },
    {
        "code": "A1",
        "name": "a1_auxiliary_objective_only",
        "selector_kind": "dual",
        "dual_enabled": True,
        "condition_mode": "off",
        "condition_on_auxiliary_state": False,
        "condition_on_auxiliary_clock": False,
        "schedule_mode": "aligned",
        "auxiliary_lead_logit": 0.0,
        "auxiliary_loss_weight": 1.0,
        "parameter_matched_control": False,
        "research_role": "auxiliary-denoising regularizer without fusion",
    },
    {
        "code": "J0",
        "name": "j0_joint_aligned",
        "selector_kind": "dual",
        "dual_enabled": True,
        "condition_mode": "matched",
        "condition_on_auxiliary_state": True,
        "condition_on_auxiliary_clock": True,
        "schedule_mode": "aligned",
        "auxiliary_lead_logit": 0.0,
        "auxiliary_loss_weight": 1.0,
        "parameter_matched_control": False,
        "research_role": "joint denoising with aligned clocks",
    },
    {
        "code": "J1",
        "name": "j1_joint_auxiliary_leads",
        "selector_kind": "dual",
        "dual_enabled": True,
        "condition_mode": "matched",
        "condition_on_auxiliary_state": True,
        "condition_on_auxiliary_clock": True,
        "schedule_mode": "tf_leads",
        "auxiliary_lead_logit": 1.0,
        "auxiliary_loss_weight": 1.0,
        "parameter_matched_control": False,
        "research_role": "joint denoising with V-JEPA leading by logit 1",
    },
)


class ContractError(RuntimeError):
    """Raised when a mutable input violates the pinned study contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(unsigned)
        ).hexdigest(),
    }


def _identity_is_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or not SHA256_RE.fullmatch(recorded):
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() == recorded


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


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"{label} must be a non-symlink regular file: {path}")
    if info.st_size <= 0:
        raise ContractError(f"{label} is empty: {path}")
    return path.resolve(strict=True)


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def _canonical_episode_directory(value: Any, label: str) -> Path:
    """Require a lexical and physical canonical directory for split identity.

    ``Path.resolve`` alone is unsafe for population-disjointness: two strings
    can name the same episode through ``..`` or a symlink ancestor.  We reject
    those aliases before recording/comparing the resolved directory.
    """
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        raise ContractError(f"{label} is not absolute")
    if ".." in path.parts:
        raise ContractError(f"{label} must not contain '..'")
    if str(path) != value:
        raise ContractError(f"{label} is not lexically canonical: {value!r}")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"{label} must be a non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ContractError(
            f"{label} uses a symlink or other physical alias: {path} -> {resolved}"
        )
    return resolved


def _python_launcher(value: str | Path) -> Path:
    """Validate Python while preserving a virtual-environment launcher path.

    Resolving ``venv/bin/python`` and then executing its base-CPython target
    bypasses the adjacent ``pyvenv.cfg`` and loses the pinned site-packages.
    Callers that execute Python must use this path; immutable provenance can
    separately record the canonical target through ``_python_executable``.
    """
    requested = Path(value).expanduser().absolute()
    if not requested.is_file() or not os.access(requested, os.X_OK):
        raise ContractError(f"Python is missing or not executable: {requested}")
    path = requested.resolve(strict=True)
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not os.access(path, os.X_OK)
    ):
        raise ContractError(
            f"resolved Python is not an executable regular file: {path}"
        )
    return requested


def _python_executable(value: str | Path) -> Path:
    """Return the canonical real binary used for immutable provenance."""
    return _python_launcher(value).resolve(strict=True)


def _validated_id(value: str, label: str) -> str:
    if SAFE_ID_RE.fullmatch(value) is None:
        raise ContractError(
            f"{label} must be a safe 1--127 character identifier"
        )
    return value


def _validated_sha256(value: str, label: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be 64 lowercase hex characters")
    return value


def _validated_commit(value: str, label: str = "Git commit") -> str:
    if COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be 40 lowercase hex characters")
    return value


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ContractError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _assert_clean_commit(repo: Path, expected_commit: str) -> None:
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ContractError(f"repository HEAD differs: {actual} != {expected_commit}")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ContractError(
            "repository must be clean: " + status.replace("\n", "; ")
        )


def _file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractError(
            f"SHA-256 mismatch for {path}: {digest} != {expected_sha256}"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _assert_recorded_file_unchanged(
    record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Re-hash one immutable study input at every allocation boundary."""
    if not isinstance(record, Mapping):
        raise ContractError(f"{label} record is missing")
    path_value = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("bytes")
    if (
        not isinstance(path_value, str)
        or not isinstance(digest, str)
        or not isinstance(byte_count, int)
    ):
        raise ContractError(f"{label} record is incomplete")
    _validated_sha256(digest, f"{label} SHA-256")
    path = _canonical_file(path_value, label)
    if path_value != str(path):
        raise ContractError(f"{label} path is not canonical")
    if path.stat().st_size != byte_count:
        raise ContractError(
            f"{label} byte count changed: {path.stat().st_size} != {byte_count}"
        )
    actual = _sha256(path)
    if actual != digest:
        raise ContractError(
            f"{label} SHA-256 changed: {actual} != {digest}"
        )


def _assert_study_inputs_unchanged(study: Mapping[str, Any]) -> None:
    """Fail a chained stage if any preflighted byte input has mutated."""
    inputs = study.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("study manifest lacks input records")
    records: list[tuple[str, Mapping[str, Any]]] = [
        ("cache validator", inputs["runtime"]["cache_validator"]),
        ("baseline config", inputs["configs"]["baseline"]),
        ("dual config", inputs["configs"]["dual"]),
        ("warm-start checkpoint", inputs["warmstart"]),
        ("V-JEPA checkpoint", inputs["vjepa"]["checkpoint"]),
        ("PCA statistics", inputs["vjepa"]["pca_stats"]),
        (
            "PCA companion metadata",
            inputs["vjepa"]["pca_companion_metadata"],
        ),
        ("phase-gate report", inputs["phase_gate"]["report"]),
        ("phase-gate runtime Python", inputs["phase_gate"]["runtime_python"]),
        ("cache-build completion record", inputs["cache_build"]["complete"]),
        ("cache-build request", inputs["cache_build"]["request"]),
        (
            "cache-build manifest metadata",
            inputs["cache_build"]["manifest_metadata"],
        ),
        (
            "cache-build episode manifest",
            inputs["cache_build"]["episode_manifest"],
        ),
    ]
    for split_name, split in inputs["splits"].items():
        records.extend(
            [
                (
                    f"{split_name} clip manifest",
                    split["clip_manifest"],
                ),
                (
                    f"{split_name} cache metadata",
                    split["cache"]["metadata"],
                ),
                (
                    f"{split_name} cached target",
                    split["cache"]["target"],
                ),
                (
                    f"{split_name} cached RGB",
                    split["cache"]["rgb"],
                ),
                (
                    f"{split_name} cached actions",
                    split["cache"]["actions"],
                ),
            ]
        )
    for label, record in records:
        _assert_recorded_file_unchanged(record, label=label)

    source = _canonical_directory(
        inputs["vjepa"]["source"]["path"], "V-JEPA source"
    )
    if _git(source, "rev-parse", "HEAD") != VJEPA_SOURCE_COMMIT:
        raise ContractError("V-JEPA source commit changed after preflight")
    if _git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("V-JEPA source became dirty after preflight")

    repository = inputs.get("repository")
    if not isinstance(repository, Mapping):
        raise ContractError("study repository record is missing")
    repo = _canonical_directory(repository.get("root", ""), "repository root")
    expected_commit = _validated_commit(
        str(repository.get("git_commit", "")), "study Git commit"
    )
    _assert_clean_commit(repo, expected_commit)
    phase_path = _canonical_file(
        inputs["phase_gate"]["report"]["path"], "V-JEPA phase-gate report"
    )
    phase_gate, phase_report = _validate_phase_gate_report(
        phase_path,
        repo=repo,
        expected_commit=expected_commit,
        warmstart=inputs["warmstart"],
        vjepa_checkpoint=inputs["vjepa"]["checkpoint"],
        pca_stats=inputs["vjepa"]["pca_stats"],
        train_split=inputs["splits"]["train"],
        runtime_python_path=Path(inputs["runtime"]["python"]),
    )
    if phase_gate != inputs["phase_gate"]:
        raise ContractError("phase-gate study binding changed")
    cache_build = _validate_cache_build(
        repo=repo,
        expected_commit=expected_commit,
        phase_report=phase_report,
        phase_gate_record=phase_gate,
        splits=inputs["splits"],
        pca_stats=inputs["vjepa"]["pca_stats"],
        vjepa_source=source,
        vjepa_checkpoint=inputs["vjepa"]["checkpoint"],
        extractor_python=Path(inputs["runtime"]["extractor_python"]),
    )
    if cache_build != inputs["cache_build"]:
        raise ContractError("cache-build study binding changed")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain one JSON object: {path}")
    return payload


def _arm(array_task_id: int) -> dict[str, Any]:
    if array_task_id < 0 or array_task_id >= len(ARMS):
        raise ContractError(
            f"array task ID must be in [0, {len(ARMS) - 1}]"
        )
    return dict(ARMS[array_task_id])


def _arm_by_code(code: str) -> dict[str, Any]:
    matches = [dict(arm) for arm in ARMS if arm["code"] == code]
    if len(matches) != 1:
        raise ContractError(f"unknown arm code: {code!r}")
    return matches[0]


def _arm_contract() -> dict[str, dict[str, Any]]:
    return {
        str(index): {
            "array_task_id": index,
            **arm,
            "common_dual_schema": (
                {
                    "tf_channels": 64,
                    "auxiliary_history_mode": "diffuse_all",
                    "state_gate_init": 0.0,
                    "state_gate_trainable": True,
                    "clock_gate_init": 0.0,
                    "clock_gate_trainable": True,
                    "head_condition_on_tf_clock": True,
                    "video_only_control": False,
                }
                if arm["dual_enabled"]
                else None
            ),
            "inference_nfe": list(INFERENCE_NFE),
            "evaluation_sources": list(EVALUATION_SOURCES),
        }
        for index, arm in enumerate(ARMS)
    }


def _validated_allowed_active_job_ids(
    values: Sequence[str] | None,
) -> list[str]:
    values = list(values or ())
    if any(ACTIVE_JOB_ID_RE.fullmatch(value) is None for value in values):
        raise ContractError(
            "allowed active job IDs must be canonical positive decimal integers"
        )
    if len(set(values)) != len(values):
        raise ContractError("allowed active job IDs must not contain duplicates")
    return sorted(values, key=int)


def _manifest_record(
    path: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    if expected_split not in EXPECTED_SPLIT_COUNTS:
        raise ContractError(f"unsupported expected split: {expected_split!r}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ContractError(
                        f"clip manifest row {line_number} is not an object"
                    )
                rows.append(row)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSONL clip manifest: {path}") from exc
    if not rows:
        raise ContractError(f"clip manifest is empty: {path}")
    required = {
        "clip_id",
        "split",
        "episode_dir",
        "start",
        "auxiliary_index",
    }
    clip_ids: list[str] = []
    episodes: list[str] = []
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ContractError(
                f"clip manifest row {index} lacks {sorted(missing)}"
            )
        if (
            isinstance(row["auxiliary_index"], bool)
            or not isinstance(row["auxiliary_index"], int)
            or row["auxiliary_index"] != index
        ):
            raise ContractError("auxiliary indexes must be dense and ordered")
        if str(row["split"]) != expected_split:
            raise ContractError(
                f"clip manifest row {index} has split={row['split']!r}, "
                f"expected {expected_split!r}"
            )
        if (
            isinstance(row["start"], bool)
            or not isinstance(row["start"], int)
            or row["start"] < 0
        ):
            raise ContractError(f"clip row {index} has an invalid start")
        episode = str(
            _canonical_episode_directory(
                row["episode_dir"],
                f"clip row {index} episode directory",
            )
        )
        clip_id = row["clip_id"]
        if not isinstance(clip_id, str) or not clip_id:
            raise ContractError(f"clip row {index} has an invalid clip ID")
        clip_ids.append(clip_id)
        episodes.append(episode)
    if len(set(clip_ids)) != len(clip_ids):
        raise ContractError("clip IDs must be unique within a manifest")
    expected_count = EXPECTED_SPLIT_COUNTS[expected_split]
    if len(rows) != expected_count:
        raise ContractError(
            f"{expected_split} manifest must contain exactly {expected_count} "
            f"clips, found {len(rows)}"
        )
    if len(set(episodes)) != len(rows):
        raise ContractError(
            f"{expected_split} manifest must contain one unique episode per clip"
        )
    return {
        **_file_record(path),
        "entries": len(rows),
        "unique_episodes": len(set(episodes)),
        "first_clip_id": clip_ids[0],
        "last_clip_id": clip_ids[-1],
    }


def _metadata_provenance_value(
    metadata: Mapping[str, Any],
    key: str,
) -> Any:
    if key in metadata:
        return metadata[key]
    provenance = metadata.get("provenance", {})
    if isinstance(provenance, Mapping):
        return provenance.get(key)
    return None


def _cache_record(
    metadata_path: Path,
    *,
    manifest: Mapping[str, Any],
    expected_split: str,
    train_manifest_sha256: str,
    checkpoint_sha256: str,
    pca_sha256: str,
    external_validation_stdout: str,
) -> dict[str, Any]:
    metadata = _read_json(metadata_path, "V-JEPA cache metadata")
    if int(metadata.get("format_version", -1)) != 1:
        raise ContractError("unsupported V-JEPA cache metadata format")
    if metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache":
        raise ContractError("unexpected V-JEPA cache artifact type")
    if metadata.get("complete") is not True:
        raise ContractError("V-JEPA cache is not marked complete")
    if str(metadata.get("split")) != expected_split:
        raise ContractError(
            f"V-JEPA cache has split={metadata.get('split')!r}, "
            f"expected {expected_split!r}"
        )
    if metadata.get("clip_manifest_sha256") != manifest["sha256"]:
        raise ContractError("cache metadata refers to another clip manifest")
    if int(metadata.get("clip_count", -1)) != int(manifest["entries"]):
        raise ContractError("cache clip_count differs from the clip manifest")
    if metadata.get("train_manifest_sha256") != train_manifest_sha256:
        raise ContractError("cache metadata refers to another training manifest")
    if metadata.get("pca_sha256") != pca_sha256:
        raise ContractError("cache metadata refers to another PCA artifact")
    cache_id = metadata.get("cache_id")
    if not isinstance(cache_id, str):
        raise ContractError("cache metadata lacks cache_id")
    _validated_sha256(cache_id, "V-JEPA cache ID")
    shape = metadata.get("target_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or int(shape[0]) != int(manifest["entries"])
        or tuple(int(value) for value in shape[1:]) != VJEPA_TARGET_SHAPE
    ):
        raise ContractError(
            "cache target_shape must be [N,64,4,24,120] and match the manifest"
        )
    dtype = str(metadata.get("target_dtype", metadata.get("dtype", ""))).lower()
    if dtype not in {"float16", "fp16", "<f2"}:
        raise ContractError(f"cached targets must be float16, found {dtype!r}")
    target_value = metadata.get("target_file")
    if not isinstance(target_value, str) or not target_value:
        raise ContractError("cache metadata lacks target_file")
    target_path = Path(target_value)
    if not target_path.is_absolute():
        target_path = metadata_path.parent / target_path
    target_path = _canonical_file(target_path, "V-JEPA target array")
    recorded_target_sha = metadata.get("target_sha256")
    if not isinstance(recorded_target_sha, str):
        raise ContractError("cache metadata lacks target_sha256")
    _validated_sha256(recorded_target_sha, "cached target SHA-256")
    # ``extract_vjepa2_targets.py validate-cache`` immediately preceded this
    # call and streamed the entire target file while comparing this digest.
    # Avoid reading the multi-GiB array a second time in the same preflight.
    target_record = {
        "path": str(target_path),
        "sha256": recorded_target_sha,
        "bytes": target_path.stat().st_size,
        "full_sha256_verified": True,
    }
    cached_inputs: dict[str, dict[str, Any]] = {}
    input_contracts = {
        "rgb": {
            "file_key": "rgb_file",
            "sha_key": "rgb_sha256",
            "dtype_key": "rgb_dtype",
            "shape_key": "rgb_shape",
            "dtype": "float16",
            "shape": [int(manifest["entries"]), 13, 3, 180, 960],
        },
        "actions": {
            "file_key": "actions_file",
            "sha_key": "actions_sha256",
            "dtype_key": "actions_dtype",
            "shape_key": "actions_shape",
            "dtype": "float32",
            "shape": [int(manifest["entries"]), 13, 5, 23],
        },
    }
    for input_name, contract in input_contracts.items():
        if metadata.get(contract["dtype_key"]) != contract["dtype"]:
            raise ContractError(
                f"cached {input_name} dtype differs from {contract['dtype']}"
            )
        if metadata.get(contract["shape_key"]) != contract["shape"]:
            raise ContractError(
                f"cached {input_name} shape differs from {contract['shape']}"
            )
        file_value = metadata.get(contract["file_key"])
        if not isinstance(file_value, str) or not file_value:
            raise ContractError(f"cache metadata lacks {contract['file_key']}")
        input_path = Path(file_value)
        if not input_path.is_absolute():
            input_path = metadata_path.parent / input_path
        input_path = _canonical_file(
            input_path, f"cached {input_name} array"
        )
        input_sha = metadata.get(contract["sha_key"])
        if not isinstance(input_sha, str):
            raise ContractError(f"cache metadata lacks {contract['sha_key']}")
        _validated_sha256(input_sha, f"cached {input_name} SHA-256")
        cached_inputs[input_name] = {
            "path": str(input_path),
            "sha256": input_sha,
            "bytes": input_path.stat().st_size,
            "shape": contract["shape"],
            "dtype": contract["dtype"],
            "full_sha256_verified": True,
        }
    expected_provenance = {
        "vjepa_source_commit": VJEPA_SOURCE_COMMIT,
        "vjepa_checkpoint_sha256": checkpoint_sha256,
        "pca_stats_sha256": pca_sha256,
    }
    for key, wanted in expected_provenance.items():
        actual = _metadata_provenance_value(metadata, key)
        if actual != wanted:
            raise ContractError(
                f"cache {key} differs: {actual!r} != {wanted!r}"
            )
    return {
        "metadata": _file_record(metadata_path),
        "split": expected_split,
        "clip_count": int(manifest["entries"]),
        "cache_id": cache_id,
        "train_manifest_sha256": train_manifest_sha256,
        "pca_sha256": pca_sha256,
        "target": target_record,
        "rgb": cached_inputs["rgb"],
        "actions": cached_inputs["actions"],
        "target_shape": [int(value) for value in shape],
        "target_dtype": "float16",
        "extraction_provenance": expected_provenance,
        "external_validator_stdout": external_validation_stdout,
    }


def _run_cache_validator(
    *,
    python: Path,
    extractor: Path,
    metadata: Path,
    manifest: Path,
    train_manifest: Path,
    pca: Path,
    vjepa_source: Path,
    vjepa_checkpoint: Path,
    vjepa_checkpoint_sha256: str,
) -> str:
    environment = os.environ.copy()
    repo_root = extractor.parent.parent
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    completed = subprocess.run(
        [
            str(python),
            str(extractor),
            "validate-cache",
            "--cache-metadata",
            str(metadata),
            "--clip-manifest",
            str(manifest),
            "--train-manifest",
            str(train_manifest),
            "--pca",
            str(pca),
            "--source-path",
            str(vjepa_source),
            "--checkpoint",
            str(vjepa_checkpoint),
            "--checkpoint-sha256",
            vjepa_checkpoint_sha256,
            "--finite-check-rows",
            "8",
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ContractError(
            f"external cache validation failed for {metadata}:\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    output = completed.stdout.strip()
    expected_prefix = "cache validation passed: id="
    if expected_prefix not in output or " sha256=" not in output:
        raise ContractError(
            f"external cache validator returned unexpected output: {output!r}"
        )
    return output


def _split_record(
    manifest_value: str,
    metadata_value: str,
    *,
    expected_split: str,
    extractor_python: Path,
    extractor: Path,
    train_manifest: Path,
    pca: Path,
    vjepa_source: Path,
    vjepa_checkpoint: Path,
    checkpoint_sha256: str,
    pca_sha256: str,
) -> dict[str, Any]:
    manifest_path = _canonical_file(manifest_value, "clip manifest")
    metadata_path = _canonical_file(metadata_value, "cache metadata")
    if expected_split not in {"train", "val", "test"}:
        raise ContractError(f"unsupported expected split: {expected_split!r}")
    manifest = _manifest_record(
        manifest_path,
        expected_split=expected_split,
    )
    validation_stdout = _run_cache_validator(
        python=extractor_python,
        extractor=extractor,
        metadata=metadata_path,
        manifest=manifest_path,
        train_manifest=train_manifest,
        pca=pca,
        vjepa_source=vjepa_source,
        vjepa_checkpoint=vjepa_checkpoint,
        vjepa_checkpoint_sha256=checkpoint_sha256,
    )
    cache = _cache_record(
        metadata_path,
        manifest=manifest,
        expected_split=expected_split,
        train_manifest_sha256=_sha256(train_manifest),
        checkpoint_sha256=checkpoint_sha256,
        pca_sha256=pca_sha256,
        external_validation_stdout=validation_stdout,
    )
    return {"clip_manifest": manifest, "cache": cache}


def _assert_disjoint_splits(splits: Mapping[str, Mapping[str, Any]]) -> None:
    episode_sets: dict[str, set[str]] = {}
    clip_sets: dict[str, set[str]] = {}
    for split, record in splits.items():
        manifest_path = Path(record["clip_manifest"]["path"])
        episodes: set[str] = set()
        clips: set[str] = set()
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                episodes.add(
                    str(
                        _canonical_episode_directory(
                            row["episode_dir"],
                            f"{split} split episode directory",
                        )
                    )
                )
                clips.add(str(row["clip_id"]))
        episode_sets[split] = episodes
        clip_sets[split] = clips
    names = list(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            episode_overlap = episode_sets[left] & episode_sets[right]
            clip_overlap = clip_sets[left] & clip_sets[right]
            if episode_overlap or clip_overlap:
                raise ContractError(
                    f"{left}/{right} split overlap: "
                    f"episodes={len(episode_overlap)}, clips={len(clip_overlap)}"
                )


def _embedded_file_record(
    record: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate a phase-gate file record against the bytes on disk."""
    if not isinstance(record, Mapping):
        raise ContractError(f"{label} record is missing")
    path_value = record.get("path")
    sha256 = record.get("sha256")
    byte_count = record.get("bytes")
    if (
        not isinstance(path_value, str)
        or not isinstance(sha256, str)
        or not isinstance(byte_count, int)
    ):
        raise ContractError(f"{label} record is incomplete")
    _validated_sha256(sha256, f"{label} SHA-256")
    path = _canonical_file(path_value, label)
    if path_value != str(path):
        raise ContractError(f"{label} path is not canonical")
    observed = _file_record(path)
    if observed["sha256"] != sha256 or observed["bytes"] != byte_count:
        raise ContractError(f"{label} bytes differ from the phase-gate record")
    return observed


def _assert_same_file_record(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in ("path", "sha256", "bytes"):
        if observed.get(key) != expected.get(key):
            raise ContractError(f"{label} {key} differs")


def _validate_phase_sampler_counters(
    counters: Any,
    *,
    deployment_mode: int,
    label: str,
) -> None:
    if not isinstance(counters, Mapping):
        raise ContractError(f"{label} sampler counters are missing")
    calls = counters.get("wan_calls_by_source_nfe")
    if not isinstance(calls, Mapping):
        raise ContractError(f"{label} Wan-call counters are missing")
    expected = {
        "online_teacher_calls": 0,
        "auxiliary_clean_available": 0,
        "deployment_mode": deployment_mode,
        "artifacts_collected": 1,
        "wan_calls_total": 1,
    }
    if any(counters.get(key) != value for key, value in expected.items()):
        raise ContractError(f"{label} future-free NFE=1 counters differ")
    if dict(calls) != {"autonomous:nfe_1": 1}:
        raise ContractError(
            f"{label} must make exactly one autonomous NFE=1 Wan call"
        )


def _validate_phase_rank_reports(
    ranks: Any,
    *,
    topology_gpus: Sequence[Any],
    training: Mapping[str, Any],
) -> None:
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise ContractError("phase gate must contain exactly eight rank reports")
    observed_ranks: set[int] = set()
    clip_indices: list[int] = []
    sync_records: list[Any] = []
    for expected_rank, rank_report in enumerate(ranks):
        if not isinstance(rank_report, Mapping) or not _identity_is_valid(
            rank_report
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} identity is invalid"
            )
        rank = rank_report.get("rank")
        local_rank = rank_report.get("local_rank")
        if rank != expected_rank or local_rank != expected_rank:
            raise ContractError("phase-gate rank/local-rank topology differs")
        observed_ranks.add(int(rank))
        gpu = rank_report.get("gpu")
        if gpu != topology_gpus[expected_rank] or "B200" not in str(gpu).upper():
            raise ContractError(f"phase-gate rank {expected_rank} is not B200")
        if rank_report.get("batch_shapes") != EXPECTED_PHASE_BATCH_SHAPES:
            raise ContractError(
                f"phase-gate rank {expected_rank} batch shapes differ"
            )
        if rank_report.get("morphology_index") != 9:
            raise ContractError(
                f"phase-gate rank {expected_rank} morphology differs"
            )
        clip_index = rank_report.get("shape_audit_clip_index")
        if (
            isinstance(clip_index, bool)
            or not isinstance(clip_index, int)
            or not 0 <= clip_index < EXPECTED_SPLIT_COUNTS["train"]
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} clip index is invalid"
            )
        clip_indices.append(clip_index)

        zero_gate = rank_report.get("zero_gate_equivalence")
        if (
            not isinstance(zero_gate, Mapping)
            or zero_gate.get("state_gate") != 0.0
            or zero_gate.get("clock_gate") != 0.0
            or zero_gate.get("bitwise_equal") is not True
            or zero_gate.get("max_absolute_difference") != 0.0
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} zero-gate proof differs"
            )
        warmstart_load = rank_report.get("warmstart_load")
        if (
            not isinstance(warmstart_load, Mapping)
            or warmstart_load.get("reset_prefixes")
            != list(EXPECTED_PHASE_RESET_PREFIXES)
            or warmstart_load.get("expected_missing_keys")
            != list(EXPECTED_PHASE_WARMSTART_MISSING_KEYS)
            or warmstart_load.get("non_prefix_missing_keys")
            != list(EXPECTED_PHASE_WARMSTART_MISSING_KEYS)
            or not isinstance(warmstart_load.get("loaded_tensor_count"), int)
            or warmstart_load["loaded_tensor_count"] <= 0
            or not isinstance(warmstart_load.get("loaded_parameter_bytes"), int)
            or warmstart_load["loaded_parameter_bytes"] <= 0
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} warm-start proof differs"
            )

        optimizer = rank_report.get("optimizer")
        if (
            not isinstance(optimizer, Mapping)
            or optimizer.get("completed_updates") != 4
            or optimizer.get("optimizer_step_values") != [1, 2, 3, 4]
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} optimizer proof differs"
            )
        gradients = optimizer.get("gradient_reachability")
        if (
            not isinstance(gradients, Mapping)
            or set(gradients) != EXPECTED_PHASE_GRADIENT_GROUPS
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} gradient groups differ"
            )
        for group, evidence in gradients.items():
            updates = evidence.get("updates") if isinstance(evidence, Mapping) else None
            if not isinstance(updates, list) or len(updates) != 4:
                raise ContractError(
                    f"phase-gate {group} gradient update count differs"
                )
            if not all(
                isinstance(item, Mapping)
                and isinstance(item.get("calls"), int)
                and not isinstance(item.get("calls"), bool)
                and item["calls"] >= 1
                and item.get("finite") is True
                for item in updates
            ):
                raise ContractError(
                    f"phase-gate {group} gradient reachability differs"
                )
            if not any(item.get("nonzero") is True for item in updates):
                raise ContractError(
                    f"phase-gate {group} never has a nonzero gradient"
                )

        inference = rank_report.get("future_free_nfe1")
        if not isinstance(inference, Mapping):
            raise ContractError(
                f"phase-gate rank {expected_rank} inference proof is missing"
            )
        if (
            inference.get("history_shape") != [1, 5, 3, 180, 960]
            or inference.get("actions_shape") != [1, 13, 5, 157]
            or inference.get("prediction_shape") != [1, 3, 8, 180, 960]
            or inference.get("auxiliary_target_argument") is not None
            or inference.get("teacher_constructed") is not False
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} future-free proof differs"
            )
        _validate_phase_sampler_counters(
            inference.get("sampler_counters"),
            deployment_mode=1,
            label=f"phase-gate rank {expected_rank} deployable",
        )
        ordinary = inference.get("ordinary_full_clip_audit")
        if (
            not isinstance(ordinary, Mapping)
            or ordinary.get("condition_source") != "autonomous"
            or ordinary.get("nfe") != 1
            or ordinary.get("auxiliary_target_argument") is not None
            or ordinary.get("generated_future_bitwise_equal") is not True
            or ordinary.get("generated_future_max_absolute_difference") != 0.0
            or ordinary.get("same_initial_noise_and_reference") is not True
            or ordinary.get("ground_truth_used_as_condition") is not False
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} scoring-path proof differs"
            )
        _validate_phase_sampler_counters(
            ordinary.get("sampler_counters"),
            deployment_mode=0,
            label=f"phase-gate rank {expected_rank} scoring audit",
        )
        artifact_hashes = ordinary.get("artifact_sha256")
        if (
            not isinstance(artifact_hashes, Mapping)
            or set(artifact_hashes)
            != {
                "video_initial_state",
                "tf_initial_state",
                "tf_initial_noise",
                "reference_latents",
            }
            or any(
                not isinstance(value, str)
                or SHA256_RE.fullmatch(value) is None
                for value in artifact_hashes.values()
            )
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} initial-noise proof differs"
            )
        sync = rank_report.get("model_sync_sha256")
        sync_keys = set(sync) if isinstance(sync, Mapping) else set()
        required_sync_keys = {
            "forward_model.tf_token_adapter.gate",
            "forward_model.tf_clock_embedding.gate",
            "forward_model.tf_velocity_head.linear.weight",
        }
        if (
            not isinstance(sync, Mapping)
            or len(sync) != 4
            or not required_sync_keys.issubset(sync_keys)
            or len(sync_keys - required_sync_keys) != 1
            or "lora_" not in next(iter(sync_keys - required_sync_keys), "")
            or any(
                not isinstance(value, str)
                or SHA256_RE.fullmatch(value) is None
                for value in sync.values()
            )
        ):
            raise ContractError(
                f"phase-gate rank {expected_rank} model-sync proof differs"
            )
        sync_records.append(dict(sync))

    if observed_ranks != set(range(8)) or len(set(clip_indices)) != 8:
        raise ContractError("phase-gate rank or clip-index population differs")
    if training.get("shape_audit_clip_indices") != clip_indices:
        raise ContractError("phase-gate top-level clip indices differ from ranks")
    if any(record != sync_records[0] for record in sync_records[1:]):
        raise ContractError("phase-gate rank model states are not synchronized")


def _validate_phase_gate_report(
    path: Path,
    *,
    repo: Path,
    expected_commit: str,
    warmstart: Mapping[str, Any],
    vjepa_checkpoint: Mapping[str, Any],
    pca_stats: Mapping[str, Any],
    train_split: Mapping[str, Any],
    runtime_python_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(path, "V-JEPA phase-gate report")
    if not _identity_is_valid(report):
        raise ContractError("V-JEPA phase-gate report identity is invalid")
    exact_fields = {
        "artifact_type": "vjepa2-j0-phase-gate",
        "format_version": 1,
        "passed": True,
        "sigma_convention": SIGMA_CONVENTION,
        "teacher_role": "offline target extractor only",
        "teacher_calls_training": 0,
        "teacher_calls_inference": 0,
        "world_size": 8,
        "j0_contract": EXPECTED_PHASE_J0_CONTRACT,
    }
    for key, expected in exact_fields.items():
        if report.get(key) != expected:
            raise ContractError(f"V-JEPA phase-gate {key} differs")
    resolved_digest = report.get("resolved_config_sha256")
    if (
        not isinstance(resolved_digest, str)
        or SHA256_RE.fullmatch(resolved_digest) is None
    ):
        raise ContractError("V-JEPA phase-gate resolved-config digest is invalid")

    topology = report.get("topology")
    if not isinstance(topology, Mapping):
        raise ContractError("V-JEPA phase-gate topology is missing")
    gpus = topology.get("gpus")
    if (
        topology.get("nodes") != 1
        or topology.get("backend") != "nccl"
        or not isinstance(gpus, list)
        or len(gpus) != 8
        or any("B200" not in str(gpu).upper() for gpu in gpus)
    ):
        raise ContractError("V-JEPA phase gate did not use eight B200 GPUs")

    training = report.get("training")
    if not isinstance(training, Mapping):
        raise ContractError("V-JEPA phase-gate training evidence is missing")
    expected_training = {
        "ddp": True,
        "optimizer_updates": 4,
        "effective_global_batch_size": 8,
        "shape_audit_clip_indices_unique": True,
        "morphology_indices": [9] * 8,
        "morphology_contract": "ABC integer index exactly 9",
        "wandb_enabled": False,
        "checkpoint_writes": 0,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ContractError("V-JEPA phase-gate training contract differs")
    inference = report.get("inference")
    if inference != {
        "source": "autonomous",
        "nfe": 1,
        "observable_history_frames": 5,
        "future_rgb_supplied": False,
        "clean_auxiliary_supplied": False,
    }:
        raise ContractError("V-JEPA phase-gate inference is not future-free NFE=1")
    if report.get("input_invariance") != {
        "repository_clean": True,
        "warmstart_unchanged": True,
        "cache_records_unchanged": True,
    }:
        raise ContractError("V-JEPA phase-gate input invariance differs")
    if report.get("warmstart_policy") != {
        "mode": "strict allowlisted reset",
        "reset_prefixes": list(EXPECTED_PHASE_RESET_PREFIXES),
        "expected_missing_keys": list(
            EXPECTED_PHASE_WARMSTART_MISSING_KEYS
        ),
    }:
        raise ContractError("V-JEPA phase-gate warm-start policy differs")

    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError("V-JEPA phase-gate provenance is missing")
    if provenance.get("repository") != {
        "path": str(repo),
        "commit": expected_commit,
        "clean": True,
    }:
        raise ContractError("V-JEPA phase-gate repository identity differs")
    phase_warmstart = _embedded_file_record(
        provenance.get("warmstart"), label="phase-gate warm start"
    )
    _assert_same_file_record(
        phase_warmstart, warmstart, label="phase-gate warm start"
    )
    cache = provenance.get("cache")
    if not isinstance(cache, Mapping):
        raise ContractError("V-JEPA phase-gate cache provenance is missing")
    phase_train_manifest = _embedded_file_record(
        cache.get("train_manifest"), label="phase-gate training manifest"
    )
    _assert_same_file_record(
        phase_train_manifest,
        train_split["clip_manifest"],
        label="phase-gate training manifest",
    )
    phase_train_metadata = _embedded_file_record(
        cache.get("train_metadata"), label="phase-gate training metadata"
    )
    _assert_same_file_record(
        phase_train_metadata,
        train_split["cache"]["metadata"],
        label="phase-gate training metadata",
    )
    phase_pca = _embedded_file_record(
        cache.get("pca"), label="phase-gate PCA artifact"
    )
    _assert_same_file_record(
        phase_pca, pca_stats, label="phase-gate PCA artifact"
    )
    train_cache = train_split["cache"]
    expected_cache_values = {
        "cache_id": train_cache["cache_id"],
        "target_sha256": train_cache["target"]["sha256"],
        "rgb_sha256": train_cache["rgb"]["sha256"],
        "actions_sha256": train_cache["actions"]["sha256"],
        "checkpoint_sha256": vjepa_checkpoint["sha256"],
        "source_commit": VJEPA_SOURCE_COMMIT,
    }
    if any(cache.get(key) != value for key, value in expected_cache_values.items()):
        raise ContractError("V-JEPA phase-gate cache identity differs")
    phase_complete = _embedded_file_record(
        cache.get("complete"), label="phase-gate cache completion record"
    )
    full_cache = report.get("full_cache_validation")
    if full_cache != {
        "validated": True,
        "split": "train",
        "clip_count": EXPECTED_SPLIT_COUNTS["train"],
        "cache_id": train_cache["cache_id"],
        "target_shape": [512, 64, 4, 24, 120],
        "rgb_shape": [512, 13, 3, 180, 960],
        "actions_shape": [512, 13, 5, 23],
    }:
        raise ContractError("V-JEPA phase-gate full-cache validation differs")
    _validate_phase_rank_reports(
        report.get("rank_reports"),
        topology_gpus=gpus,
        training=training,
    )
    runtime_python = _embedded_file_record(
        provenance.get("runtime_python"), label="phase-gate runtime Python"
    )
    if runtime_python["path"] != str(runtime_python_path):
        raise ContractError("V-JEPA phase-gate runtime Python differs")
    return (
        {
            "report": _file_record(path),
            "identity_sha256": report["identity_sha256"],
            "validated": True,
            "cache_complete": phase_complete,
            "runtime_python": runtime_python,
        },
        report,
    )


def _assert_git_ancestor(
    repo: Path,
    ancestor: str,
    descendant: str,
) -> None:
    _validated_commit(ancestor, "cache-build Git commit")
    completed = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractError(
            f"cache-build commit {ancestor} is not an ancestor of {descendant}"
        )


def _validate_cache_build(
    *,
    repo: Path,
    expected_commit: str,
    phase_report: Mapping[str, Any],
    phase_gate_record: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, Any]],
    pca_stats: Mapping[str, Any],
    vjepa_source: Path,
    vjepa_checkpoint: Mapping[str, Any],
    extractor_python: Path,
) -> dict[str, Any]:
    complete_record = phase_gate_record["cache_complete"]
    complete_path = Path(complete_record["path"])
    complete = _read_json(complete_path, "immutable cache-build completion record")
    if (
        complete.get("artifact_type") != "vjepa2.1-immutable-cache-build"
        or complete.get("format_version") != 1
    ):
        raise ContractError("cache-build completion schema differs")

    request_path = _canonical_file(
        complete_path.with_name("build_request.json"),
        "immutable cache-build request",
    )
    request_record = _file_record(request_path)
    if complete.get("build_request_sha256") != request_record["sha256"]:
        raise ContractError("cache-build request digest differs from complete.json")
    request = _read_json(request_path, "immutable cache-build request")
    expected_request_fields = {
        "artifact_type": "vjepa2.1-immutable-cache-build-request",
        "format_version": 1,
        "build_id": complete.get("build_id"),
        "build_root": str(complete_path.parent),
        "extractor_python": str(extractor_python),
        "vjepa_source": str(vjepa_source),
        "vjepa_source_commit": VJEPA_SOURCE_COMMIT,
        "vjepa_checkpoint": vjepa_checkpoint["path"],
        "vjepa_checkpoint_sha256": vjepa_checkpoint["sha256"],
        "train_clips": EXPECTED_SPLIT_COUNTS["train"],
        "val_clips": EXPECTED_SPLIT_COUNTS["val"],
        "test_clips": EXPECTED_SPLIT_COUNTS["test"],
        "clips_per_episode": EXPECTED_CLIPS_PER_EPISODE,
        "seed": EXPECTED_CACHE_BUILD_SEED,
        "pca_max_clips": 256,
        "pca_max_tokens": 250_000,
    }
    if any(
        request.get(key) != value for key, value in expected_request_fields.items()
    ):
        raise ContractError("immutable cache-build request contract differs")
    build_repo_value = request.get("repo_root")
    if not isinstance(build_repo_value, str):
        raise ContractError("cache-build repository path is missing")
    build_repo = _canonical_directory(
        build_repo_value, "cache-build repository root"
    )
    if str(build_repo) != build_repo_value:
        raise ContractError("cache-build repository path is not canonical")
    if not (build_repo / ".git").exists():
        raise ContractError("cache-build repository root is not a Git checkout")
    build_commit = request.get("git_commit")
    if build_commit != complete.get("git_commit"):
        raise ContractError("cache-build Git commit differs across records")
    _assert_clean_commit(
        build_repo,
        _validated_commit(str(build_commit), "cache-build Git commit"),
    )
    _assert_git_ancestor(repo, str(build_commit), expected_commit)

    manifest_metadata_path = _canonical_file(
        complete.get("manifest_metadata", ""),
        "cache-build manifest metadata",
    )
    manifest_metadata_record = _file_record(manifest_metadata_path)
    if (
        complete.get("manifest_metadata") != manifest_metadata_record["path"]
        or complete.get("manifest_metadata_sha256")
        != manifest_metadata_record["sha256"]
    ):
        raise ContractError("manifest-metadata digest differs from complete.json")
    manifest_metadata = _read_json(
        manifest_metadata_path, "cache-build manifest metadata"
    )
    expected_manifest_fields = {
        "format_version": 1,
        "schema": "abc-vjepa2.1-fixed-clips-v1",
        "seed": EXPECTED_CACHE_BUILD_SEED,
        "sample_size": 13,
        "chunk_size": 5,
        "action_span": 65,
        "frame_offsets": list(range(0, 61, 5)),
        "clips_per_episode": EXPECTED_CLIPS_PER_EPISODE,
        "episode_count_consumed": sum(EXPECTED_SPLIT_COUNTS.values()),
    }
    if any(
        manifest_metadata.get(key) != value
        for key, value in expected_manifest_fields.items()
    ):
        raise ContractError("cache-build manifest metadata contract differs")
    episode_manifest_path = _canonical_file(
        manifest_metadata.get("episode_manifest", ""),
        "cache-build episode manifest",
    )
    episode_manifest_record = _file_record(episode_manifest_path)
    if (
        manifest_metadata.get("episode_manifest_sha256")
        != episode_manifest_record["sha256"]
        or request.get("episode_manifest") != episode_manifest_record["path"]
        or request.get("episode_manifest_sha256")
        != episode_manifest_record["sha256"]
    ):
        raise ContractError("cache-build episode-manifest binding differs")

    pca_path = _canonical_file(complete.get("pca", ""), "cache-build PCA artifact")
    pca_record = _file_record(pca_path)
    if (
        complete.get("pca") != pca_record["path"]
        or complete.get("pca_sha256") != pca_record["sha256"]
        or pca_record != {
            key: pca_stats[key] for key in ("path", "sha256", "bytes")
        }
    ):
        raise ContractError("cache-build PCA binding differs")

    complete_splits = complete.get("splits")
    metadata_splits = manifest_metadata.get("splits")
    if not isinstance(complete_splits, Mapping) or set(complete_splits) != {
        "train",
        "val",
        "test",
    }:
        raise ContractError("complete.json must bind exactly train/val/test")
    if not isinstance(metadata_splits, Mapping) or set(metadata_splits) != {
        "train",
        "val",
        "test",
    }:
        raise ContractError("manifest metadata must bind exactly train/val/test")
    study_keys = {"train": "train", "val": "validation", "test": "test"}
    for split_name, study_key in study_keys.items():
        split_record = splits[study_key]
        manifest_record = split_record["clip_manifest"]
        cache_record = split_record["cache"]
        count = EXPECTED_SPLIT_COUNTS[split_name]
        metadata_entry = metadata_splits[split_name]
        if not isinstance(metadata_entry, Mapping):
            raise ContractError(f"{split_name} manifest metadata is missing")
        expected_manifest_name = f"{split_name}.jsonl"
        if metadata_entry.get("file") != expected_manifest_name:
            raise ContractError(
                f"{split_name} manifest metadata filename differs"
            )
        manifest_path = _canonical_file(
            manifest_metadata_path.parent / expected_manifest_name,
            f"{split_name} cache-build clip manifest",
        )
        if (
            str(manifest_path) != manifest_record["path"]
            or metadata_entry.get("sha256") != manifest_record["sha256"]
            or metadata_entry.get("clip_count") != count
            or metadata_entry.get("episode_count") != count
        ):
            raise ContractError(
                f"{split_name} cache-build manifest binding differs"
            )
        complete_entry = complete_splits[split_name]
        if not isinstance(complete_entry, Mapping):
            raise ContractError(f"{split_name} complete.json entry is missing")
        expected_complete_entry = {
            "metadata": cache_record["metadata"]["path"],
            "metadata_sha256": cache_record["metadata"]["sha256"],
            "cache_id": cache_record["cache_id"],
            "clip_count": count,
            "target_sha256": cache_record["target"]["sha256"],
            "rgb_sha256": cache_record["rgb"]["sha256"],
            "actions_sha256": cache_record["actions"]["sha256"],
        }
        if dict(complete_entry) != expected_complete_entry:
            raise ContractError(
                f"{split_name} cache binding differs from complete.json"
            )

    phase_cache = phase_report.get("provenance", {}).get("cache", {})
    if (
        not isinstance(phase_cache, Mapping)
        or phase_cache.get("complete", {}).get("sha256")
        != complete_record["sha256"]
    ):
        raise ContractError("phase gate refers to another cache build")
    return {
        "complete": dict(complete_record),
        "request": request_record,
        "manifest_metadata": manifest_metadata_record,
        "episode_manifest": episode_manifest_record,
        "build_id": complete["build_id"],
        "git_commit": str(build_commit),
        "seed": EXPECTED_CACHE_BUILD_SEED,
        "clips_per_episode": EXPECTED_CLIPS_PER_EPISODE,
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "validated": True,
    }


def _collect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repo = _canonical_directory(args.repo_root, "repository root")
    expected_commit = _validated_commit(args.git_commit)
    _assert_clean_commit(repo, expected_commit)
    project_root = _canonical_directory(args.project_root, "project root")
    if project_root != repo / "projects" / "latent_action_models":
        raise ContractError(
            "project root must be the pinned repository's latent-action project"
        )
    python = _python_executable(args.python)
    extractor_python = _python_executable(args.extractor_python)
    extractor = _canonical_file(
        repo / "tools" / "extract_vjepa2_targets.py",
        "V-JEPA cache validator",
    )
    wan_dir = _canonical_directory(args.wan_dir, "Wan directory")
    videox_home = _canonical_directory(args.videox_home, "VideoX-Fun checkout")
    baseline_config = _canonical_file(args.baseline_config, "baseline config")
    dual_config = _canonical_file(args.dual_config, "dual config")
    if repo not in baseline_config.parents or repo not in dual_config.parents:
        raise ContractError("experiment configs must be tracked inside the repository")
    warmstart = _canonical_file(args.warmstart, "warm-start checkpoint")
    warmstart_sha = _validated_sha256(
        args.warmstart_sha256, "warm-start SHA-256"
    )
    warmstart_record = _file_record(
        warmstart, expected_sha256=warmstart_sha
    )

    vjepa_source = _canonical_directory(args.vjepa_source, "V-JEPA source")
    actual_vjepa_commit = _git(vjepa_source, "rev-parse", "HEAD")
    if actual_vjepa_commit != VJEPA_SOURCE_COMMIT:
        raise ContractError(
            "V-JEPA source is not pinned to the official release commit: "
            f"{actual_vjepa_commit} != {VJEPA_SOURCE_COMMIT}"
        )
    if _git(vjepa_source, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("V-JEPA source checkout must be clean")
    vjepa_checkpoint = _canonical_file(
        args.vjepa_checkpoint, "V-JEPA checkpoint"
    )
    vjepa_checkpoint_sha = _validated_sha256(
        args.vjepa_checkpoint_sha256, "V-JEPA checkpoint SHA-256"
    )
    if vjepa_checkpoint.stat().st_size != VJEPA_CHECKPOINT_BYTES:
        raise ContractError(
            "V-JEPA checkpoint byte count differs: "
            f"{vjepa_checkpoint.stat().st_size} != {VJEPA_CHECKPOINT_BYTES}"
        )
    vjepa_checkpoint_record = _file_record(
        vjepa_checkpoint, expected_sha256=vjepa_checkpoint_sha
    )
    pca_stats = _canonical_file(args.pca_stats, "PCA whitening statistics")
    pca_sha = _validated_sha256(args.pca_stats_sha256, "PCA statistics SHA-256")
    pca_record = _file_record(pca_stats, expected_sha256=pca_sha)
    pca_companion = _canonical_file(
        pca_stats.with_name(pca_stats.name + ".json"),
        "PCA companion metadata",
    )
    train_manifest = _canonical_file(args.train_manifest, "training clip manifest")
    splits = {
        "train": _split_record(
            str(train_manifest),
            args.train_cache_metadata,
            expected_split="train",
            extractor_python=extractor_python,
            extractor=extractor,
            train_manifest=train_manifest,
            pca=pca_stats,
            vjepa_source=vjepa_source,
            vjepa_checkpoint=vjepa_checkpoint,
            checkpoint_sha256=vjepa_checkpoint_sha,
            pca_sha256=pca_sha,
        ),
        "validation": _split_record(
            args.validation_manifest,
            args.validation_cache_metadata,
            expected_split="val",
            extractor_python=extractor_python,
            extractor=extractor,
            train_manifest=train_manifest,
            pca=pca_stats,
            vjepa_source=vjepa_source,
            vjepa_checkpoint=vjepa_checkpoint,
            checkpoint_sha256=vjepa_checkpoint_sha,
            pca_sha256=pca_sha,
        ),
        "test": _split_record(
            args.test_manifest,
            args.test_cache_metadata,
            expected_split="test",
            extractor_python=extractor_python,
            extractor=extractor,
            train_manifest=train_manifest,
            pca=pca_stats,
            vjepa_source=vjepa_source,
            vjepa_checkpoint=vjepa_checkpoint,
            checkpoint_sha256=vjepa_checkpoint_sha,
            pca_sha256=pca_sha,
        ),
    }
    _assert_disjoint_splits(splits)
    inputs = {
        "repository": {
            "root": str(repo),
            "git_commit": expected_commit,
            "clean": True,
        },
        "runtime": {
            "python": str(python),
            "extractor_python": str(extractor_python),
            "cache_validator": _file_record(extractor),
            "project_root": str(project_root),
            "wan_dir": str(wan_dir),
            "videox_home": str(videox_home),
        },
        "configs": {
            "baseline": {
                "selector": args.baseline_selector,
                **_file_record(baseline_config),
            },
            "dual": {
                "selector": args.dual_selector,
                **_file_record(dual_config),
            },
        },
        "warmstart": warmstart_record,
        "vjepa": {
            "model_name": VJEPA_MODEL_NAME,
            "checkpoint_key": VJEPA_CHECKPOINT_KEY,
            "source": {
                "path": str(vjepa_source),
                "commit": VJEPA_SOURCE_COMMIT,
                "clean": True,
            },
            "checkpoint": vjepa_checkpoint_record,
            "pca_stats": pca_record,
            "pca_companion_metadata": _file_record(pca_companion),
            "source_dim": VJEPA_SOURCE_DIM,
            "target_shape_per_clip": list(VJEPA_TARGET_SHAPE),
            "frame_map": list(VJEPA_FRAME_MAP),
            "camera_transform": "each of 3 views independently",
            "training_imports_teacher": False,
            "inference_imports_teacher": False,
            "one_time_cache_cost_excluded_from_optimizer_convergence": True,
            "one_time_cache_cost_reported_separately": True,
        },
        "splits": splits,
    }
    phase_report_path = _canonical_file(
        args.phase_gate_report, "V-JEPA phase-gate report"
    )
    phase_gate, phase_report = _validate_phase_gate_report(
        phase_report_path,
        repo=repo,
        expected_commit=expected_commit,
        warmstart=warmstart_record,
        vjepa_checkpoint=vjepa_checkpoint_record,
        pca_stats=pca_record,
        train_split=splits["train"],
        runtime_python_path=python,
    )
    cache_build = _validate_cache_build(
        repo=repo,
        expected_commit=expected_commit,
        phase_report=phase_report,
        phase_gate_record=phase_gate,
        splits=splits,
        pca_stats=pca_record,
        vjepa_source=vjepa_source,
        vjepa_checkpoint=vjepa_checkpoint_record,
        extractor_python=extractor_python,
    )
    inputs["phase_gate"] = phase_gate
    inputs["cache_build"] = cache_build
    return inputs


def _study_manifest(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    *,
    wandb_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    study_id = _validated_id(args.study_id, "study ID")
    study_root = _canonical_directory(args.study_root, "study root")
    run_root = _canonical_directory(args.run_root, "run root")
    if study_root.parent != run_root or study_root.name != study_id:
        raise ContractError("study root must be RUN_ROOT/STUDY_ID")
    allowed_jobs = _validated_allowed_active_job_ids(
        args.allow_active_job_id
    )
    return _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_video_diffusion_study",
            "created_at_utc": _now(),
            "study_id": study_id,
            "study_root": str(study_root),
            "inputs": dict(inputs),
            "arms": _arm_contract(),
            "schedule": {
                "seed": SEED,
                "total_completed_updates": TOTAL_UPDATES,
                "warmup_updates": WARMUP_UPDATES,
                "completed_update_milestones": list(
                    COMPLETED_UPDATE_MILESTONES
                ),
                "allocation_stage_endpoints": list(STAGE_ENDPOINTS),
                "trainer_iteration_for_completed_update": "completed_update - 1",
                "batch_size_per_gpu": 1,
                "gpus_per_arm": 8,
                "gradient_accumulation_steps": 1,
                "effective_global_batch_size": 8,
            },
            "inference": {
                "nfe": list(INFERENCE_NFE),
                "noise_seed": EVALUATION_NOISE_SEED,
                "sources": list(EVALUATION_SOURCES),
                "deployable_sources": list(DEPLOYABLE_EVALUATION_SOURCES),
                "oracle_leakage_only_sources": list(ORACLE_EVALUATION_SOURCES),
                "teacher_invocations_required": 0,
                "teacher_invocations_allowed": 0,
                "actual_wan_calls_must_equal_nfe": True,
                "latency_protocol": {
                    "per_arm_grid_telemetry": {
                        "claim_role": "diagnostic_only",
                        "batch_size": 1,
                        "sources": ["autonomous", "off"],
                        "warmups": 20,
                        "timed_repetitions": 100,
                        "statistics": ["p50_ms", "p95_ms"],
                        "scope": (
                            "model preparation + Wan calls + VAE decode"
                        ),
                        "auxiliary_target": None,
                        "clean_auxiliary_available": False,
                        "timed_artifact_materialization": False,
                        "separate_untimed_counter_audit": True,
                    },
                    "final_claim_comparison": {
                        "arms": [
                            {
                                "arm": "J1",
                                "source": "autonomous",
                                "nfe": 4,
                            },
                            {
                                "arm": "VPM",
                                "source": "autonomous",
                                "nfe": 8,
                            },
                        ],
                        "same_slurm_allocation": True,
                        "same_node": True,
                        "same_B200": True,
                        "same_process": True,
                        "both_models_resident": True,
                        "identical_immutable_batch_inputs": True,
                        "batch_size": 1,
                        "warmup_pairs": 20,
                        "timed_pairs": 100,
                        "counterbalance": (
                            "even pair J1-first; odd pair VPM-first"
                        ),
                        "timed_artifact_materialization": False,
                        "forward_hooks_active_during_timing": False,
                        "future_ground_truth_available_to_sampler": False,
                        "clean_auxiliary_available_to_sampler": False,
                        "online_teacher_calls": 0,
                        "claim_statistics": [
                            "paired_stratified_bootstrap_mean_relative_improvement",
                            "p95_ms",
                            "execution_order_strata",
                        ],
                    },
                },
                "autonomous_shuffled_policy": (
                    "quality-only mechanism control with effective batch >=2; "
                    "forbidden in batch-one latency"
                ),
                "quality_protocol": {
                    "fixed_test_clips": 128,
                    "distributed_world_size": 8,
                    "batch_size_per_rank": 2,
                    "trainer_visualization_is_diagnostic_only": True,
                    "stateless_noise_key": "clip_index",
                    "intermediate_completed_updates": [1, 50, 100, 200, 400, 800],
                    "intermediate_grid": [
                        {"source": source, "nfe": nfe}
                        for source, nfe in (
                            ("autonomous", 4),
                            ("autonomous", 8),
                            ("off", 4),
                            ("off", 8),
                            ("autonomous_shuffled", 4),
                            ("autonomous_shuffled", 8),
                            ("oracle_matched", 4),
                            ("oracle_shuffled", 4),
                        )
                    ],
                    "final_completed_updates": 1000,
                    "final_grid": [
                        {"source": source, "nfe": nfe}
                        for source in EVALUATION_SOURCES
                        for nfe in INFERENCE_NFE
                    ],
                    "deployable_sources_use_history_only_public_entrypoint": True,
                    "oracle_sources_are_leakage_only": True,
                    "rank0_rehashes_full_test_target_rgb_action_arrays": True,
                    "temporal_metric_includes_history_to_first_future_boundary": (
                        True
                    ),
                    "perceptual_metric": {
                        "enabled": False,
                        "reason": (
                            "no LPIPS/video-perceptual checkpoint is pinned; "
                            "implicit pretrained downloads are forbidden"
                        ),
                    },
                    "claim_scope": (
                        "latent and decoded raw-video reconstruction metrics "
                        "only; VAE-reconstruction comparisons are diagnostic"
                    ),
                },
            },
            "clock": {
                "convention": SIGMA_CONVENTION,
                "J0": "auxiliary_sigma = video_sigma",
                "J1": "auxiliary_sigma = sigmoid(logit(video_sigma) - 1)",
                "exact_endpoints": True,
            },
            "comparisons": {
                "training_convergence": [
                    "J1 vs VPM",
                    "J1 vs A1",
                    "J1 vs J0",
                    "VPM vs V0 architecture sanity check",
                ],
                "autonomous_inference": [
                    "J1 autonomous vs same-checkpoint off",
                    "J1 autonomous vs same-checkpoint autonomous_shuffled",
                    "J1@NFE4 vs VPM@NFE4 and VPM@NFE8",
                ],
                "oracle_policy": (
                    "oracle results quantify headroom only and cannot support a "
                    "deployable quality or speed claim"
                ),
            },
            "wandb": {
                "entity": EXPECTED_ENTITY,
                "project": EXPECTED_PROJECT,
                "access": "PRIVATE",
                "group": None,
                "authenticated_viewer_username": wandb_attestation.get(
                    "viewer_username"
                ),
                "authenticated_viewer_email": wandb_attestation.get(
                    "viewer_email"
                ),
                "user_requested_email": "ldu@nvidia.edu",
                "authenticated_email_matches_user_request": (
                    wandb_attestation.get("viewer_email")
                    == "ldu@nvidia.edu"
                ),
                "identity_deviation_note": (
                    None
                    if wandb_attestation.get("viewer_email")
                    == "ldu@nvidia.edu"
                    else (
                        "authenticated private personal W&B entity is valid, "
                        "but its viewer email differs from ldu@nvidia.edu; "
                        "no claim is made that the run belongs to that email"
                    )
                ),
            },
            "slurm": {
                "array": f"0-{len(ARMS) - 1}",
                "non_requeueable": True,
                "dependency_chain": "afterok across allocation stage endpoints",
                "paired_latency_post_study_job": {
                    "dependency": "afterok:final_stage_array",
                    "nodes": 1,
                    "gpus": 1,
                    "purpose": (
                        "same-B200 paired J1 autonomous NFE4 versus "
                        "VPM autonomous NFE8 benchmark"
                    ),
                    "runs_final_analyzer_after_benchmark": True,
                    "analysis_output_root": str(
                        run_root / "_analysis" / study_id
                    ),
                },
                "allowed_preexisting_active_job_ids": allowed_jobs,
                "existing_jobs_are_read_only_and_untouched": True,
            },
        }
    )


def _validate_study_manifest(
    path: Path,
    *,
    expected_study_id: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    payload = _read_json(path, "study manifest")
    problems: list[str] = []
    if payload.get("kind") != "vjepa2_controlled_video_diffusion_study":
        problems.append("kind differs")
    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version differs")
    if expected_study_id is not None and payload.get("study_id") != expected_study_id:
        problems.append("study_id differs")
    if (
        expected_commit is not None
        and payload.get("inputs", {}).get("repository", {}).get("git_commit")
        != expected_commit
    ):
        problems.append("Git commit differs")
    if payload.get("arms") != _arm_contract():
        problems.append("five-arm contract differs")
    wandb_record = payload.get("wandb")
    if (
        not isinstance(wandb_record, dict)
        or wandb_record.get("entity") != EXPECTED_ENTITY
        or wandb_record.get("project") != EXPECTED_PROJECT
        or wandb_record.get("access") != "PRIVATE"
        or wandb_record.get("group") is not None
        or wandb_record.get("authenticated_viewer_username")
        != EXPECTED_ENTITY
        or not isinstance(
            wandb_record.get("authenticated_viewer_email"), str
        )
        or wandb_record.get("user_requested_email") != "ldu@nvidia.edu"
        or wandb_record.get("authenticated_email_matches_user_request")
        != (
            wandb_record.get("authenticated_viewer_email")
            == "ldu@nvidia.edu"
        )
        or (
            wandb_record.get("authenticated_email_matches_user_request")
            is False
            and not isinstance(
                wandb_record.get("identity_deviation_note"), str
            )
        )
    ):
        problems.append("W&B authenticated-viewer provenance differs")
    if not _identity_is_valid(payload):
        problems.append("identity SHA-256 is invalid")
    if problems:
        raise ContractError("invalid study manifest: " + "; ".join(problems))
    return payload


def _config_value(config: Any, dotted: str, *, missing: Any = None) -> Any:
    current = config
    for part in dotted.split("."):
        if current is None or part not in current:
            return missing
        current = current[part]
    return current


def _assert_equal(
    problems: list[str],
    config: Any,
    dotted: str,
    wanted: Any,
) -> None:
    marker = object()
    actual = _config_value(config, dotted, missing=marker)
    if actual is marker or actual != wanted:
        problems.append(f"{dotted}: {actual!r} != {wanted!r}")


def _assert_float(
    problems: list[str],
    config: Any,
    dotted: str,
    wanted: float,
) -> None:
    actual = _config_value(config, dotted, missing=None)
    try:
        matches = math.isclose(float(actual), wanted, rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        problems.append(f"{dotted}: {actual!r} != {wanted!r}")


def _assert_fixed_dataset(
    problems: list[str],
    config: Any,
    key: str,
    *,
    manifest: str,
    metadata: str,
) -> None:
    datasets = _config_value(config, f"{key}.datasets", missing={})
    if list(datasets.keys()) != ["ABC"]:
        problems.append(f"{key}.datasets must contain only ABC")
        return
    prefix = f"{key}.datasets.ABC"
    _assert_equal(
        problems,
        config,
        f"{prefix}._target_",
        "robot_wm.datasets.abc.fixed_clip_dataset.ABCFixedClipDataset",
    )
    _assert_equal(problems, config, f"{prefix}.clip_manifest", manifest)
    _assert_equal(problems, config, f"{prefix}.cache_metadata", metadata)
    _assert_equal(problems, config, f"{prefix}.transform", None)


def _assert_resolved_contract(
    content: bytes,
    *,
    arm: Mapping[str, Any],
    stage_endpoint: int,
    manifest: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
) -> None:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ContractError("OmegaConf is required to validate Hydra output") from exc
    config = OmegaConf.create(content.decode("utf-8"))
    inputs = manifest["inputs"]
    splits = inputs["splits"]
    problems: list[str] = []
    expected_common = {
        "name": run_id,
        "seed": SEED,
        "data_loader.batch_size": 1,
        "trainer.config.max_iter": stage_endpoint,
        "trainer.config.gradient_accumulation_steps": 1,
        # ``Trainer`` performs an exact resume whenever the live save path
        # exists, but would subsequently apply ``load_path`` again.  Therefore
        # only the first stage may carry the model-only production warm start.
        "trainer.config.load_path": (
            inputs["warmstart"]["path"] if stage_endpoint == 1 else None
        ),
        "trainer.config.throughput_warmup_steps": WARMUP_UPDATES,
        "trainer.config.saving.save_every": TOTAL_UPDATES,
        "trainer.config.validation.val_every": TOTAL_UPDATES,
        "trainer.config.visualization.viz_every": TOTAL_UPDATES,
        # The original V0 class predates raw latent-trajectory artifacts.  Its
        # decoded visualization is still required on disk by record-stage,
        # while dual arms additionally fail closed on missing latent artifacts.
        "trainer.config.visualization.require_success": bool(
            arm["dual_enabled"]
        ),
        "lr_scheduler_factory.lr_lambda.warmup_steps": WARMUP_UPDATES,
        "lr_scheduler_factory.lr_lambda.total_steps": TOTAL_UPDATES,
        "wandb.enabled": True,
        "wandb.mode": "online",
        "wandb.entity": EXPECTED_ENTITY,
        "wandb.project": EXPECTED_PROJECT,
        "wandb.id": run_id,
        "wandb.group": None,
        "wandb.resume": "allow",
    }
    for dotted, wanted in expected_common.items():
        _assert_equal(problems, config, dotted, wanted)
    save_path = Path(str(_config_value(config, "trainer.config.saving.save_path")))
    viz_path = Path(
        str(_config_value(config, "trainer.config.visualization.viz_path"))
    )
    if save_path != run_dir / "snapshot.pt":
        problems.append("snapshot path is outside the pinned arm directory")
    if viz_path != run_dir / "visualization":
        problems.append("visualization path is outside the pinned arm directory")
    for key in ("dataset", "val_dataset", "viz_dataset"):
        if bool(_config_value(config, f"{key}.img_augment", missing=False)):
            problems.append(f"{key}.img_augment must be false")
    _assert_fixed_dataset(
        problems,
        config,
        "dataset",
        manifest=splits["train"]["clip_manifest"]["path"],
        metadata=splits["train"]["cache"]["metadata"]["path"],
    )
    _assert_fixed_dataset(
        problems,
        config,
        "val_dataset",
        manifest=splits["validation"]["clip_manifest"]["path"],
        metadata=splits["validation"]["cache"]["metadata"]["path"],
    )
    _assert_fixed_dataset(
        problems,
        config,
        "viz_dataset",
        manifest=splits["test"]["clip_manifest"]["path"],
        metadata=splits["test"]["cache"]["metadata"]["path"],
    )

    target = str(_config_value(config, "model._target_", missing=""))
    if arm["dual_enabled"]:
        if not target.endswith(".DualExplicitActionDiTModel"):
            problems.append(f"dual arm has the wrong model target: {target!r}")
        expected_dual = {
            "model.dual_diffusion.enabled": True,
            "model.dual_diffusion.tf_channels": 64,
            "model.dual_diffusion.auxiliary_history_mode": "diffuse_all",
            "model.dual_diffusion.condition_mode": arm["condition_mode"],
            "model.dual_diffusion.condition_on_tf": arm[
                "condition_on_auxiliary_state"
            ],
            "model.dual_diffusion.condition_on_tf_clock": arm[
                "condition_on_auxiliary_clock"
            ],
            "model.dual_diffusion.schedule_mode": arm["schedule_mode"],
            "model.dual_diffusion.state_gate_trainable": True,
            "model.dual_diffusion.clock_gate_trainable": True,
            "model.dual_diffusion.head_condition_on_tf_clock": True,
            "model.dual_diffusion.video_only_control": False,
            "model.dual_diffusion.parameter_matched_control": arm[
                "parameter_matched_control"
            ],
            "model.dual_diffusion.capture_latent_trajectories": True,
            "model.dual_diffusion.evaluation_nfe_steps": list(INFERENCE_NFE),
            "model.dual_diffusion.evaluation_noise_seed": EVALUATION_NOISE_SEED,
            "model.dual_diffusion.evaluation_condition_sources": list(
                EVALUATION_SOURCES
            ),
            "model.forward_model.dual_diffusion.enabled": True,
            "model.forward_model.dual_diffusion.tf_channels": 64,
            "model.forward_model.dual_diffusion.auxiliary_history_mode": (
                "diffuse_all"
            ),
            "model.forward_model.dual_diffusion.condition_mode": arm[
                "condition_mode"
            ],
            "model.forward_model.dual_diffusion.condition_on_tf": arm[
                "condition_on_auxiliary_state"
            ],
            "model.forward_model.dual_diffusion.condition_on_tf_clock": arm[
                "condition_on_auxiliary_clock"
            ],
            "model.forward_model.dual_diffusion.head_condition_on_tf_clock": True,
            "model.forward_model.dual_diffusion.parameter_matched_control": arm[
                "parameter_matched_control"
            ],
        }
        for dotted, wanted in expected_dual.items():
            _assert_equal(problems, config, dotted, wanted)
        for prefix in (
            "model.dual_diffusion",
            "model.forward_model.dual_diffusion",
        ):
            _assert_float(problems, config, f"{prefix}.state_gate_init", 0.0)
            _assert_float(problems, config, f"{prefix}.clock_gate_init", 0.0)
            _assert_float(
                problems,
                config,
                f"{prefix}.tf_loss_weight",
                float(arm["auxiliary_loss_weight"]),
            )
            _assert_float(
                problems,
                config,
                f"{prefix}.tf_lead_logit",
                float(arm["auxiliary_lead_logit"]),
            )
        if _config_value(config, "model.time_frequency_transform", missing=None) is not None:
            problems.append("V-JEPA dual arms must not instantiate an online TF transform")
    else:
        if not target.endswith(".ExplicitActionDiTModel") or target.endswith(
            ".DualExplicitActionDiTModel"
        ):
            problems.append(f"V0 has the wrong model target: {target!r}")
        if bool(
            _config_value(
                config,
                "model.forward_model.dual_diffusion.enabled",
                missing=False,
            )
        ):
            problems.append("V0 forward model unexpectedly enables dual diffusion")
        if _config_value(config, "model.time_frequency_transform", missing=None) is not None:
            problems.append("V0 unexpectedly instantiates an auxiliary transform")

    # The pretrained V-JEPA encoder is an offline cache builder, never part of
    # the trainable model or Trainer. Cache paths are permitted only in dataset
    # nodes already checked above.
    for subtree_name in ("model", "trainer"):
        subtree = _config_value(config, subtree_name, missing={})
        serialized = json.dumps(
            OmegaConf.to_container(subtree, resolve=True),
            sort_keys=True,
        ).lower()
        if "vjepa2_target" in serialized or "vjepa_source" in serialized:
            problems.append(
                f"{subtree_name} config contains a V-JEPA source/teacher call"
            )
        if "teacher_checkpoint" in serialized or "teacher_encoder" in serialized:
            problems.append(
                f"{subtree_name} config contains a pretrained teacher"
            )
        for forbidden_value in (
            str(inputs["vjepa"]["source"]["path"]).lower(),
            str(inputs["vjepa"]["checkpoint"]["path"]).lower(),
            VJEPA_MODEL_NAME.lower(),
        ):
            if forbidden_value in serialized:
                problems.append(
                    f"{subtree_name} config embeds an offline teacher input"
                )

    tags = list(_config_value(config, "wandb.tags", missing=[]))
    required_tags = {
        "vjepa2-controlled-study",
        str(arm["code"]),
        "seed-1234",
    }
    if not required_tags.issubset(tags):
        problems.append(
            f"wandb.tags lacks {sorted(required_tags - set(tags))}"
        )
    if problems:
        raise ContractError(
            "resolved V-JEPA study configuration violates its contract: "
            + "; ".join(problems)
        )


def _compose_config(
    python: Path,
    project_root: Path,
    overrides: Sequence[str],
) -> bytes:
    completed = subprocess.run(
        [
            str(python),
            "train.py",
            *overrides,
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        env=os.environ.copy(),
    )
    if completed.returncode:
        raise ContractError(
            "Hydra config resolution failed:\n"
            + completed.stderr.decode(errors="replace")
        )
    if not completed.stdout.strip():
        raise ContractError("Hydra emitted an empty resolved config")
    return completed.stdout


def command_arm_contract(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    payload = {
        "array_task_id": args.array_task_id,
        **arm,
        "common_dual_schema": (
            _arm_contract()[str(args.array_task_id)]["common_dual_schema"]
        ),
        "completed_update_milestones": list(COMPLETED_UPDATE_MILESTONES),
        "inference_nfe": list(INFERENCE_NFE),
        "evaluation_sources": list(EVALUATION_SOURCES),
    }
    if args.format == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        fields = [
            arm["code"],
            arm["name"],
            arm["selector_kind"],
            str(arm["dual_enabled"]).lower(),
            "" if arm["condition_mode"] is None else arm["condition_mode"],
            str(arm["condition_on_auxiliary_state"]).lower(),
            str(arm["condition_on_auxiliary_clock"]).lower(),
            "" if arm["schedule_mode"] is None else arm["schedule_mode"],
            (
                ""
                if arm["auxiliary_lead_logit"] is None
                else f"{arm['auxiliary_lead_logit']:.2f}"
            ),
            f"{arm['auxiliary_loss_weight']:.2f}",
            str(arm["parameter_matched_control"]).lower(),
        ]
        delimiter = "\t" if args.format == "tsv" else "|"
        if any(
            delimiter in value or "\n" in value or "\r" in value
            for value in fields
        ):
            raise ContractError("arm contract contains a shell delimiter")
        print(delimiter.join(fields))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    inputs = _collect_inputs(args)
    print(
        json.dumps(
            {
                "status": "passed",
                "kind": "vjepa2_controlled_study_preflight",
                "checked_at_utc": _now(),
                "inputs": inputs,
                "arms": _arm_contract(),
                "completed_update_milestones": list(
                    COMPLETED_UPDATE_MILESTONES
                ),
                "allocation_stage_endpoints": list(STAGE_ENDPOINTS),
                "inference_nfe": list(INFERENCE_NFE),
                "sigma_convention": SIGMA_CONVENTION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_wandb_private(args: argparse.Namespace) -> int:
    if args.entity != EXPECTED_ENTITY or args.project != EXPECTED_PROJECT:
        raise ContractError(
            f"study is locked to {EXPECTED_ENTITY}/{EXPECTED_PROJECT}"
        )
    # Reuse the already-audited GraphQL access check without changing the
    # historical pilot or its evidence contract.
    import dual_abc_pilot as pilot

    result = pilot._wandb_private_project(args.entity, args.project)
    print(json.dumps(result, sort_keys=True))
    return 0


def command_create_study(args: argparse.Namespace) -> int:
    inputs = _collect_inputs(args)
    import dual_abc_pilot as pilot

    wandb_attestation = pilot._wandb_private_project(
        EXPECTED_ENTITY, EXPECTED_PROJECT
    )
    payload = _study_manifest(
        args, inputs, wandb_attestation=wandb_attestation
    )
    output = Path(args.output)
    study_root = Path(payload["study_root"])
    if output.parent.resolve(strict=True) != study_root:
        raise ContractError("study manifest must be directly under study_root")
    _exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def command_validate_study_inputs(args: argparse.Namespace) -> int:
    """Revalidate and rehash every immutable study boundary.

    Stage and post-study entrypoints invoke this before creating any new
    evidence/output path, including evidence-only resumptions.
    """
    expected_commit = _validated_commit(args.expected_commit)
    repo = _canonical_directory(args.repo_root, "repository root")
    _assert_clean_commit(repo, expected_commit)
    manifest_path = _canonical_file(args.study_manifest, "study manifest")
    study = _validate_study_manifest(
        manifest_path,
        expected_commit=expected_commit,
    )
    repository = study.get("inputs", {}).get("repository", {})
    if (
        not isinstance(repository, Mapping)
        or repository.get("root") != str(repo)
        or repository.get("git_commit") != expected_commit
        or repository.get("clean") is not True
    ):
        raise ContractError("study manifest repository binding differs")
    _assert_study_inputs_unchanged(study)
    print(
        json.dumps(
            {
                "status": "passed",
                "kind": "vjepa2_controlled_study_input_revalidation",
                "study_identity_sha256": study["identity_sha256"],
                "phase_gate_identity_sha256": study["inputs"]["phase_gate"][
                    "identity_sha256"
                ],
                "phase_gate_report_sha256": study["inputs"]["phase_gate"][
                    "report"
                ]["sha256"],
                "cache_build_complete_sha256": study["inputs"]["cache_build"][
                    "complete"
                ]["sha256"],
                "git_commit": expected_commit,
            },
            sort_keys=True,
        )
    )
    return 0


def command_create_arm(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    study_id = _validated_id(args.study_id, "study ID")
    run_id = _validated_id(args.run_id, "run ID")
    expected_commit = _validated_commit(args.git_commit)
    repo = _canonical_directory(args.repo_root, "repository root")
    _assert_clean_commit(repo, expected_commit)
    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    study_root = _canonical_directory(args.study_root, "study root")
    if run_dir.parent != study_root or run_dir.name != arm["name"]:
        raise ContractError("arm directory must be STUDY_ROOT/immutable-arm-name")
    study_manifest_path = _canonical_file(
        args.study_manifest, "study manifest"
    )
    study = _validate_study_manifest(
        study_manifest_path,
        expected_study_id=study_id,
        expected_commit=expected_commit,
    )
    if Path(study["study_root"]) != study_root:
        raise ContractError("study manifest root differs")
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_arm",
            "created_at_utc": _now(),
            "study_id": study_id,
            "study_identity_sha256": study["identity_sha256"],
            "array_task_id": args.array_task_id,
            "arm": arm,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "git_commit": expected_commit,
            "slurm_array_job_id": args.slurm_array_job_id,
            "completed_update_milestones": list(
                COMPLETED_UPDATE_MILESTONES
            ),
            "allocation_stage_endpoints": list(STAGE_ENDPOINTS),
            "wandb": {
                "entity": EXPECTED_ENTITY,
                "project": EXPECTED_PROJECT,
                "group": None,
                "id": run_id,
                "resume": "allow",
            },
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != run_dir:
        raise ContractError("arm manifest must be directly under run_dir")
    _exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def _validate_arm_manifest(
    path: Path,
    *,
    arm: Mapping[str, Any],
    run_dir: Path,
    expected_commit: str,
) -> dict[str, Any]:
    payload = _read_json(path, "arm manifest")
    problems = []
    expected = {
        "kind": "vjepa2_controlled_study_arm",
        "arm": dict(arm),
        "run_dir": str(run_dir),
        "git_commit": expected_commit,
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            problems.append(f"{key} differs")
    if not _identity_is_valid(payload):
        problems.append("identity SHA-256 is invalid")
    if problems:
        raise ContractError("invalid arm manifest: " + "; ".join(problems))
    return payload


def command_prepare_stage(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    endpoint = int(args.stage_endpoint)
    if endpoint not in STAGE_ENDPOINTS:
        raise ContractError(f"stage endpoint is not pinned: {endpoint}")
    expected_commit = _validated_commit(args.git_commit)
    repo = _canonical_directory(args.repo_root, "repository root")
    _assert_clean_commit(repo, expected_commit)
    project_root = _canonical_directory(args.project_root, "project root")
    python = _python_launcher(args.python)
    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    arm_manifest_path = _canonical_file(args.arm_manifest, "arm manifest")
    arm_manifest = _validate_arm_manifest(
        arm_manifest_path,
        arm=arm,
        run_dir=run_dir,
        expected_commit=expected_commit,
    )
    study_manifest_path = _canonical_file(
        args.study_manifest, "study manifest"
    )
    study = _validate_study_manifest(
        study_manifest_path,
        expected_study_id=arm_manifest["study_id"],
        expected_commit=expected_commit,
    )
    recorded_python = Path(study["inputs"]["runtime"]["python"])
    if python.resolve(strict=True) != recorded_python:
        raise ContractError(
            "stage Python launcher resolves to another runtime binary"
        )
    if arm_manifest["study_identity_sha256"] != study["identity_sha256"]:
        raise ContractError("arm manifest refers to another study identity")
    # Keep the command safe even when invoked outside the audited Slurm
    # entrypoint. The entrypoint also validates before any stage-path mutation,
    # including evidence-only resumes that do not call this command.
    _assert_study_inputs_unchanged(study)
    config_record = study["inputs"]["configs"][arm["selector_kind"]]
    if args.config_selector != config_record["selector"]:
        raise ContractError("Hydra selector differs from the study manifest")
    config_path = _canonical_file(args.config_file, "arm config")
    if _file_record(config_path)["sha256"] != config_record["sha256"]:
        raise ContractError("arm config differs from the study manifest")
    content = _compose_config(python, project_root, args.override)
    _assert_resolved_contract(
        content,
        arm=arm,
        stage_endpoint=endpoint,
        manifest=study,
        run_dir=run_dir,
        run_id=arm_manifest["run_id"],
    )
    resolved_output = Path(args.resolved_config_output)
    stage_output = Path(args.stage_manifest_output)
    if (
        resolved_output.parent.resolve(strict=True) != run_dir
        or stage_output.parent.resolve(strict=True) != run_dir
    ):
        raise ContractError("stage provenance must be written directly under run_dir")
    _exclusive_bytes(resolved_output, content)
    resolved_record = _file_record(resolved_output)
    stage_payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_stage",
            "created_at_utc": _now(),
            "arm_identity_sha256": arm_manifest["identity_sha256"],
            "arm_code": arm["code"],
            "stage_endpoint_completed_updates": endpoint,
            "primary_milestone": endpoint in COMPLETED_UPDATE_MILESTONES,
            "trainer_terminal_iteration": endpoint - 1,
            "resolved_config": resolved_record,
            "config_selector": args.config_selector,
            "snapshot": str(run_dir / "snapshot.pt"),
            "sigma_convention": SIGMA_CONVENTION,
        }
    )
    _exclusive_json(stage_output, stage_payload)
    print(stage_payload["identity_sha256"])
    return 0


def command_record_stage(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    endpoint = int(args.stage_endpoint)
    if endpoint not in STAGE_ENDPOINTS:
        raise ContractError("stage endpoint is not pinned")
    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    arm_manifest_path = _canonical_file(args.arm_manifest, "arm manifest")
    arm_manifest = _validate_arm_manifest(
        arm_manifest_path,
        arm=arm,
        run_dir=run_dir,
        expected_commit=_validated_commit(args.git_commit),
    )
    stage_manifest_path = _canonical_file(
        args.stage_manifest, "stage manifest"
    )
    stage_manifest = _read_json(stage_manifest_path, "stage manifest")
    if (
        not _identity_is_valid(stage_manifest)
        or stage_manifest.get("arm_identity_sha256")
        != arm_manifest["identity_sha256"]
        or stage_manifest.get("stage_endpoint_completed_updates") != endpoint
    ):
        raise ContractError("stage manifest identity or endpoint differs")
    snapshot = _canonical_file(run_dir / "snapshot.pt", "stage snapshot")
    completion = _canonical_file(
        run_dir / "training_complete.json", "training completion marker"
    )
    completion_payload = _read_json(completion, "training completion marker")
    if completion_payload.get("status") != "completed":
        raise ContractError("training completion status is not completed")
    if int(completion_payload.get("completed_updates", -1)) != endpoint:
        raise ContractError("training completion update count differs")
    if (
        completion_payload.get("run_identity_sha256")
        != arm_manifest["identity_sha256"]
    ):
        raise ContractError("training completion run identity differs")
    started = int(args.started_at_epoch)
    finished = int(args.finished_at_epoch)
    if started < 0 or finished < started:
        raise ContractError("stage wall-clock timestamps are invalid")
    visualization = run_dir / "visualization" / f"iter_{endpoint - 1}"
    artifact_count = (
        sum(1 for path in visualization.rglob("*") if path.is_file())
        if visualization.is_dir()
        else 0
    )
    if artifact_count == 0:
        raise ContractError("stage produced no visualization/evaluation artifacts")
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_stage_outcome",
            "recorded_at_utc": _now(),
            "arm_identity_sha256": arm_manifest["identity_sha256"],
            "stage_identity_sha256": stage_manifest["identity_sha256"],
            "arm_code": arm["code"],
            "completed_updates": endpoint,
            "primary_milestone": endpoint in COMPLETED_UPDATE_MILESTONES,
            "trainer_terminal_iteration": endpoint - 1,
            "snapshot_observed_at_stage_end": _file_record(snapshot),
            "training_completion": _file_record(completion),
            "stage_wall_seconds_including_validation_and_visualization": (
                finished - started
            ),
            "visualization_artifact_file_count_at_stage": artifact_count,
            "teacher_invocations_during_training": 0,
            "cache_extraction_wall_time_included": False,
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != run_dir:
        raise ContractError("stage outcome must be directly under run_dir")
    _exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def command_record_outcome(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    arm_manifest = _validate_arm_manifest(
        _canonical_file(args.arm_manifest, "arm manifest"),
        arm=arm,
        run_dir=run_dir,
        expected_commit=_validated_commit(args.git_commit),
    )
    observed: dict[str, Any] = {}
    for endpoint in STAGE_ENDPOINTS:
        path = _canonical_file(
            run_dir / f"stage_outcome_update_{endpoint:04d}.json",
            f"stage {endpoint} outcome",
        )
        payload = _read_json(path, f"stage {endpoint} outcome")
        if (
            not _identity_is_valid(payload)
            or payload.get("arm_identity_sha256")
            != arm_manifest["identity_sha256"]
            or payload.get("completed_updates") != endpoint
        ):
            raise ContractError(f"stage {endpoint} outcome is invalid")
        observed[str(endpoint)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "identity_sha256": payload["identity_sha256"],
            "primary_milestone": payload["primary_milestone"],
        }
    if [
        int(key)
        for key, value in observed.items()
        if value["primary_milestone"]
    ] != list(COMPLETED_UPDATE_MILESTONES):
        raise ContractError("primary completed-update milestone inventory differs")
    snapshot = _canonical_file(run_dir / "snapshot.pt", "final snapshot")
    snapshot_record = _file_record(snapshot)
    final_stage_path = _canonical_file(
        run_dir / "stage_outcome_update_1000.json",
        "final stage outcome",
    )
    final_stage = _read_json(final_stage_path, "final stage outcome")
    study_manifest_path = _canonical_file(
        run_dir.parent / "study_manifest.json", "study manifest"
    )
    study_manifest = _read_json(study_manifest_path, "study manifest")
    final_stage_manifest_path = _canonical_file(
        run_dir / "stage_manifest_update_1000.json",
        "final stage manifest",
    )
    final_stage_manifest = _read_json(
        final_stage_manifest_path, "final stage manifest"
    )
    if final_stage.get("snapshot_observed_at_stage_end") != snapshot_record:
        raise ContractError(
            "live final snapshot differs from the immutable update-1000 stage"
        )
    completion = _read_json(
        _canonical_file(
            run_dir / "training_complete.json", "final completion marker"
        ),
        "final completion marker",
    )
    if completion.get("completed_updates") != TOTAL_UPDATES:
        raise ContractError("final run did not complete 1000 optimizer updates")
    quality_inventories: dict[str, Any] = {}
    latency_records: dict[str, Any] = {}
    if arm["dual_enabled"]:
        for update in COMPLETED_UPDATE_MILESTONES:
            quality_path = _canonical_file(
                run_dir
                / "quality"
                / f"update_{update:04d}"
                / "inventory.json",
                f"quality inventory at update {update}",
            )
            quality = _read_json(
                quality_path, f"quality inventory at update {update}"
            )
            if (
                not _identity_is_valid(quality)
                or quality.get("kind")
                != "vjepa2_controlled_study_quality_inventory"
                or quality.get("complete") is not True
                or quality.get("arm_code") != arm["code"]
                or quality.get("arm_identity_sha256")
                != arm_manifest["identity_sha256"]
                or quality.get("completed_updates") != update
                or quality.get("clip_count") != 128
            ):
                raise ContractError(
                    f"quality inventory at update {update} is invalid"
                )
            quality_inventories[str(update)] = {
                "path": str(quality_path),
                "sha256": _sha256(quality_path),
                "identity_sha256": quality["identity_sha256"],
                "record_count": quality["observed_record_count"],
            }
        expected_latency_names = {
            f"source_{source}_nfe_{nfe}.json"
            for source in ("autonomous", "off")
            for nfe in INFERENCE_NFE
        }
        latency_dir = _canonical_directory(
            run_dir / "latency", "latency evidence directory"
        )
        actual_latency_names = {
            path.name
            for path in latency_dir.iterdir()
            if path.is_file() or path.is_symlink()
        }
        if actual_latency_names != expected_latency_names:
            raise ContractError(
                "latency evidence inventory differs: "
                f"missing={sorted(expected_latency_names - actual_latency_names)}, "
                f"extra={sorted(actual_latency_names - expected_latency_names)}"
            )
        reference_sample_identity = None
        for source in ("autonomous", "off"):
            for nfe in INFERENCE_NFE:
                latency_path = _canonical_file(
                    latency_dir / f"source_{source}_nfe_{nfe}.json",
                    f"{source} NFE={nfe} latency evidence",
                )
                latency = _read_json(
                    latency_path, f"{source} NFE={nfe} latency evidence"
                )
                sample_identity = latency.get("inputs", {}).get(
                    "sample_identity"
                )
                stage_input = latency.get("inputs", {}).get("stage_outcome")
                study_input = latency.get("inputs", {}).get("study_manifest")
                stage_manifest_input = latency.get("inputs", {}).get(
                    "stage_manifest"
                )
                if (
                    not _identity_is_valid(latency)
                    or latency.get("kind")
                    != "vjepa2_controlled_study_latency"
                    or latency.get("arm_code") != arm["code"]
                    or latency.get("arm_identity_sha256")
                    != arm_manifest["identity_sha256"]
                    or latency.get("source") != source
                    or latency.get("nfe") != nfe
                    or latency.get("batch_size") != 1
                    or latency.get("warmups") != 20
                    or latency.get("repetitions") != 100
                    or not isinstance(stage_input, dict)
                    or stage_input.get("path") != str(final_stage_path)
                    or stage_input.get("sha256") != _sha256(final_stage_path)
                    or stage_input.get("identity_sha256")
                    != final_stage["identity_sha256"]
                    or not isinstance(study_input, dict)
                    or study_input.get("path") != str(study_manifest_path)
                    or study_input.get("sha256")
                    != _sha256(study_manifest_path)
                    or study_input.get("identity_sha256")
                    != study_manifest["identity_sha256"]
                    or not isinstance(stage_manifest_input, dict)
                    or stage_manifest_input.get("path")
                    != str(final_stage_manifest_path)
                    or stage_manifest_input.get("sha256")
                    != _sha256(final_stage_manifest_path)
                    or stage_manifest_input.get("identity_sha256")
                    != final_stage_manifest["identity_sha256"]
                    or not isinstance(sample_identity, dict)
                    or sample_identity.get("sample_index") != 0
                ):
                    raise ContractError(
                        f"{source} NFE={nfe} latency evidence is invalid"
                    )
                if reference_sample_identity is None:
                    reference_sample_identity = sample_identity
                elif sample_identity != reference_sample_identity:
                    raise ContractError(
                        "latency sample identity differs across source/NFE"
                    )
                key = f"{source}:nfe_{nfe}"
                latency_records[key] = {
                    "path": str(latency_path),
                    "sha256": _sha256(latency_path),
                    "identity_sha256": latency["identity_sha256"],
                    "sample_identity": dict(sample_identity),
                }
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_arm_outcome",
            "recorded_at_utc": _now(),
            "arm_identity_sha256": arm_manifest["identity_sha256"],
            "arm_code": arm["code"],
            "status": "training_completed_inference_evidence_pending_analysis",
            "completed_updates": TOTAL_UPDATES,
            "primary_milestones": list(COMPLETED_UPDATE_MILESTONES),
            "allocation_stages": observed,
            "final_snapshot": snapshot_record,
            "quality_evidence": {
                "trainer_visualization_is_diagnostic_only": True,
                "fixed_test_clip_count": 128,
                "inventories": quality_inventories,
                "reconstruction_metrics_only": True,
            },
            "latency_evidence": {
                "complete": bool(arm["dual_enabled"]),
                "record_count": len(latency_records),
                "expected_record_count": (
                    2 * len(INFERENCE_NFE) if arm["dual_enabled"] else 0
                ),
                "fixed_sample_identity": (
                    reference_sample_identity if arm["dual_enabled"] else None
                ),
                "records": latency_records,
                "V0_excluded": not arm["dual_enabled"],
            },
            "inference_contract": {
                "nfe": list(INFERENCE_NFE),
                "sources": list(EVALUATION_SOURCES),
                "teacher_invocations_allowed": 0,
                "posthoc_paired_latency_and_quality_analysis_required": True,
            },
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != run_dir:
        raise ContractError("arm outcome must be directly under run_dir")
    _exclusive_json(output, payload)
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    manifest_path = _canonical_file(args.study_manifest, "study manifest")
    study = _validate_study_manifest(manifest_path)
    allowed = _validated_allowed_active_job_ids(args.allow_active_job_id)
    recorded_allowed = study["slurm"]["allowed_preexisting_active_job_ids"]
    if allowed != recorded_allowed:
        raise ContractError("submission active-job allow-list differs")
    job_ids = list(args.job_id)
    if len(job_ids) != len(STAGE_ENDPOINTS):
        raise ContractError(
            f"expected {len(STAGE_ENDPOINTS)} chained job IDs"
        )
    if any(JOB_ID_RE.fullmatch(value) is None for value in job_ids):
        raise ContractError("Slurm returned an invalid job ID")
    if JOB_ID_RE.fullmatch(args.paired_latency_job_id) is None:
        raise ContractError("Slurm returned an invalid paired-latency job ID")
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_submission",
            "submitted_at_utc": _now(),
            "study_identity_sha256": study["identity_sha256"],
            "stage_endpoints": list(STAGE_ENDPOINTS),
            "stage_array_job_ids": [
                {"completed_updates": endpoint, "job_id": job_id}
                for endpoint, job_id in zip(STAGE_ENDPOINTS, job_ids)
            ],
            "dependency": "afterok",
            "paired_latency_job": {
                "job_id": args.paired_latency_job_id,
                "dependency": "afterok:final_stage_array",
                "comparison": "J1_autonomous_nfe4_vs_VPM_autonomous_nfe8",
                "nodes": 1,
                "gpus": 1,
                "same_allocation_pairing": True,
                "runs_final_analyzer_after_benchmark": True,
                "analysis_output_root": study["slurm"][
                    "paired_latency_post_study_job"
                ]["analysis_output_root"],
            },
            "max_concurrent_arms": int(args.max_concurrent_arms),
            "allowed_preexisting_active_job_ids": allowed,
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != manifest_path.parent:
        raise ContractError("submission record must be directly under study root")
    _exclusive_json(output, payload)
    return 0


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--extractor-python", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--videox-home", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--baseline-selector", required=True)
    parser.add_argument("--dual-config", required=True)
    parser.add_argument("--dual-selector", required=True)
    parser.add_argument("--warmstart", required=True)
    parser.add_argument("--warmstart-sha256", required=True)
    parser.add_argument("--vjepa-source", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-sha256", required=True)
    parser.add_argument("--pca-stats", required=True)
    parser.add_argument("--pca-stats-sha256", required=True)
    parser.add_argument("--phase-gate-report", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-cache-metadata", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--validation-cache-metadata", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--test-cache-metadata", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    arm = subparsers.add_parser("arm-contract")
    arm.add_argument("--array-task-id", type=int, required=True)
    arm.add_argument(
        "--format",
        choices=("json", "tsv", "pipe"),
        default="json",
    )
    arm.set_defaults(handler=command_arm_contract)

    preflight = subparsers.add_parser("preflight")
    _add_input_arguments(preflight)
    preflight.set_defaults(handler=command_preflight)

    wandb_private = subparsers.add_parser("wandb-private")
    wandb_private.add_argument("--entity", required=True)
    wandb_private.add_argument("--project", required=True)
    wandb_private.set_defaults(handler=command_wandb_private)

    study = subparsers.add_parser("create-study")
    _add_input_arguments(study)
    study.add_argument("--study-id", required=True)
    study.add_argument("--study-root", required=True)
    study.add_argument("--run-root", required=True)
    study.add_argument("--allow-active-job-id", action="append", default=[])
    study.add_argument("--output", required=True)
    study.set_defaults(handler=command_create_study)

    validate_inputs = subparsers.add_parser("validate-study-inputs")
    validate_inputs.add_argument("--study-manifest", required=True)
    validate_inputs.add_argument("--expected-commit", required=True)
    validate_inputs.add_argument("--repo-root", required=True)
    validate_inputs.set_defaults(handler=command_validate_study_inputs)

    create_arm = subparsers.add_parser("create-arm")
    create_arm.add_argument("--array-task-id", type=int, required=True)
    create_arm.add_argument("--study-id", required=True)
    create_arm.add_argument("--run-id", required=True)
    create_arm.add_argument("--git-commit", required=True)
    create_arm.add_argument("--repo-root", required=True)
    create_arm.add_argument("--study-root", required=True)
    create_arm.add_argument("--run-dir", required=True)
    create_arm.add_argument("--study-manifest", required=True)
    create_arm.add_argument("--slurm-array-job-id", required=True)
    create_arm.add_argument("--output", required=True)
    create_arm.set_defaults(handler=command_create_arm)

    stage = subparsers.add_parser("prepare-stage")
    stage.add_argument("--array-task-id", type=int, required=True)
    stage.add_argument("--stage-endpoint", type=int, required=True)
    stage.add_argument("--git-commit", required=True)
    stage.add_argument("--repo-root", required=True)
    stage.add_argument("--project-root", required=True)
    stage.add_argument("--python", required=True)
    stage.add_argument("--run-dir", required=True)
    stage.add_argument("--arm-manifest", required=True)
    stage.add_argument("--study-manifest", required=True)
    stage.add_argument("--config-selector", required=True)
    stage.add_argument("--config-file", required=True)
    stage.add_argument("--override", action="append", default=[])
    stage.add_argument("--resolved-config-output", required=True)
    stage.add_argument("--stage-manifest-output", required=True)
    stage.set_defaults(handler=command_prepare_stage)

    record_stage = subparsers.add_parser("record-stage")
    record_stage.add_argument("--array-task-id", type=int, required=True)
    record_stage.add_argument("--stage-endpoint", type=int, required=True)
    record_stage.add_argument("--git-commit", required=True)
    record_stage.add_argument("--run-dir", required=True)
    record_stage.add_argument("--arm-manifest", required=True)
    record_stage.add_argument("--stage-manifest", required=True)
    record_stage.add_argument("--started-at-epoch", required=True)
    record_stage.add_argument("--finished-at-epoch", required=True)
    record_stage.add_argument("--output", required=True)
    record_stage.set_defaults(handler=command_record_stage)

    outcome = subparsers.add_parser("record-outcome")
    outcome.add_argument("--array-task-id", type=int, required=True)
    outcome.add_argument("--git-commit", required=True)
    outcome.add_argument("--run-dir", required=True)
    outcome.add_argument("--arm-manifest", required=True)
    outcome.add_argument("--output", required=True)
    outcome.set_defaults(handler=command_record_outcome)

    submission = subparsers.add_parser("record-submission")
    submission.add_argument("--study-manifest", required=True)
    submission.add_argument("--job-id", action="append", required=True)
    submission.add_argument("--paired-latency-job-id", required=True)
    submission.add_argument("--max-concurrent-arms", type=int, required=True)
    submission.add_argument("--allow-active-job-id", action="append", default=[])
    submission.add_argument("--output", required=True)
    submission.set_defaults(handler=command_record_submission)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ContractError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
