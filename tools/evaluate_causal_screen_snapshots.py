#!/usr/bin/env python3
"""Guarded post-training evaluation of the five ABC causal-screen snapshots.

This program is intentionally independent of ``Trainer``.  It never resumes a
run, never initializes W&B, and never writes below a training screen.  A single
ordered list of deterministic ABC visualization batches is cached on each
distributed rank and replayed through every model arm.  Each source/NFE result
is published as an analyzer-compatible safetensors artifact beneath a fresh
evaluation root.

LACWM clock convention: ``sigma=1`` is noise and ``sigma=0`` is clean data.
Oracle condition sources are leakage-only diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import dual_abc_causal_screen as causal
import dual_abc_pilot as pilot


SCHEMA_VERSION = 2
WORLD_SIZE = 8
COMPLETED_UPDATES = 200
EVALUATION_ITERATION = 199
EFFECTIVE_STATE_GATE_ABS_TOL = 1e-6
SIGMA_CONVENTION = "1=noise,0=clean"
BASE_EVALUATION_NOISE_SEED = 20_260_726
NFE_STEPS = (1, 2, 4, 8)
SOURCE_SCREEN_CONDITION_SOURCES = (
    "autonomous",
    "off",
    "oracle_matched",
    "oracle_shuffled",
)
CONDITION_SOURCES = (
    *SOURCE_SCREEN_CONDITION_SOURCES,
    "autonomous_shuffled",
)
SOURCE_CODES = {
    "autonomous": 0,
    "off": 1,
    "oracle_matched": 2,
    "oracle_shuffled": 3,
    "autonomous_shuffled": 4,
}
MODE_CODES = {"off": 0, "matched": 1, "shuffled": 2}
MAX_BATCHES_PER_RANK = 4
EXPECTED_VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JOB_ID_RE = re.compile(r"^[0-9]+(?:[_.;][A-Za-z0-9_.%+-]+)?$")
ACTIVE_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
CORRECTION_PATHS = (
    "robot_wm/modeling/dual_diffusion/conditioning.py",
    "projects/latent_action_models/lam/dual_explicit_action_dit_model.py",
)
CORRECTION_MARKERS = {
    CORRECTION_PATHS[0]: (
        "local_future_noise_component",
        "roll_across_global_batch(generated_future_clean_residual)",
    ),
    CORRECTION_PATHS[1]: (
        "initial_tf_noise,",
        "tf_sigma_expanded,",
        '"autonomous_shuffled"',
    ),
}


def _required_evaluation_tensor_names() -> set[str]:
    """Return the strict five-source posthoc artifact inventory.

    The immutable training-screen contract intentionally remains four-source.
    The fifth source exists only in this corrected posthoc evaluator, so do
    not widen ``causal._required_trajectory_tensor_names()`` and retroactively
    change what the historical training jobs were required to emit.
    """
    required = set(causal._required_trajectory_tensor_names())
    for nfe in NFE_STEPS:
        required.update(
            {
                f"video_final_autonomous_shuffled_nfe_{nfe}",
                f"tf_final_autonomous_shuffled_nfe_{nfe}",
                f"decoded_future_autonomous_shuffled_nfe_{nfe}",
            }
        )
    return required


class EvaluationContractError(RuntimeError):
    """Raised when immutable inputs or paired-evaluation contracts differ."""


def _validated_allowed_active_job_ids(
    values: Sequence[str] | None,
) -> tuple[str, ...]:
    raw = tuple(values or ())
    if any(ACTIVE_JOB_ID_RE.fullmatch(value) is None for value in raw):
        raise EvaluationContractError(
            "allowed active job IDs must be positive decimal Slurm IDs"
        )
    if len(set(raw)) != len(raw):
        raise EvaluationContractError(
            "allowed active job IDs must be unique"
        )
    canonical = tuple(sorted(raw, key=int))
    if raw != canonical:
        raise EvaluationContractError(
            "allowed active job IDs must be supplied in increasing numeric order"
        )
    return canonical


def _validate_active_job_coexistence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(
            "active-job coexistence provenance must be a mapping"
        )
    allowlisted = _validated_allowed_active_job_ids(
        value.get("allowlisted_active_job_ids")
    )
    observed = _validated_allowed_active_job_ids(
        value.get("active_user_job_ids_observed_at_submission_gate")
    )
    expected = {
        "allowlisted_active_job_ids": list(allowlisted),
        "active_user_job_ids_observed_at_submission_gate": list(observed),
        "exact_set_match_required": True,
        "wildcards_and_job_name_bypasses_supported": False,
        "existing_jobs_are_read_only_and_untouched": True,
        "empty_allowlist_requires_no_active_user_jobs": True,
    }
    if allowlisted != observed or dict(value) != expected:
        raise EvaluationContractError(
            "active-job coexistence provenance must record an exact "
            "allowlist/observed-ID set match"
        )
    return expected


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(payload)
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
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationContractError(
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
        raise EvaluationContractError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise EvaluationContractError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationContractError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EvaluationContractError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationContractError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise EvaluationContractError(
            f"{label} must be a nonempty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _is_beneath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_disjoint(
    left: Path,
    right: Path,
    left_label: str,
    right_label: str,
) -> None:
    if left == right or _is_beneath(left, right) or _is_beneath(right, left):
        raise EvaluationContractError(
            f"{left_label} and {right_label} must be disjoint: {left}, {right}"
        )


def _validated_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise EvaluationContractError(
            f"{label} contains unsafe characters or exceeds 127 characters"
        )
    return value


def _validated_commit(value: str, label: str) -> str:
    if not COMMIT_RE.fullmatch(value):
        raise EvaluationContractError(
            f"{label} must be a full lowercase hexadecimal Git commit"
        )
    return value


def _validated_batches_per_rank(value: int) -> int:
    value = int(value)
    if value < 1 or value > MAX_BATCHES_PER_RANK:
        raise EvaluationContractError(
            "batches per rank must be between 1 and "
            f"{MAX_BATCHES_PER_RANK}, got {value}"
        )
    return value


def _selected_arm_names(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw_names = value.split(",")
    else:
        raw_names = list(value)
    names = [str(name).strip() for name in raw_names if str(name).strip()]
    allowed = [arm["name"] for arm in causal.ARMS]
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise EvaluationContractError(
            f"unknown evaluation arms {unknown}; allowed={allowed}"
        )
    if len(names) != len(set(names)):
        raise EvaluationContractError("evaluation arms must be unique")
    if len(names) < 2:
        raise EvaluationContractError(
            "posthoc evaluation requires at least two arms for paired analysis"
        )
    # Collective ordering must be identical on every rank and independent of
    # caller spelling, so retain the immutable causal-screen order.
    selected = [name for name in allowed if name in names]
    return selected


def _git(
    repo_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise EvaluationContractError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed


def _assert_clean_commit(
    repo_root: Path,
    expected_commit: str,
    *,
    label: str = "evaluation repository",
) -> None:
    actual = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    if actual != expected_commit:
        raise EvaluationContractError(
            f"{label} changed: {actual} != {expected_commit}"
        )
    status = _git(
        repo_root, "status", "--porcelain", "--untracked-files=all"
    ).stdout.strip()
    if status:
        raise EvaluationContractError(
            f"{label} must be clean: " + status.replace("\n", "; ")
        )


def _validate_training_repository(
    recorded_repository_root: Any,
    *,
    evaluation_repo_root: Path,
    training_commit: str,
) -> dict[str, Any]:
    if not isinstance(recorded_repository_root, str):
        raise EvaluationContractError(
            "screen-recorded training repository must be a string"
        )
    training_repo_root = _canonical_directory(
        recorded_repository_root,
        "screen-recorded training repository",
    )
    if recorded_repository_root != str(training_repo_root):
        raise EvaluationContractError(
            "screen-recorded training repository path is not canonical"
        )
    top_level = _canonical_directory(
        _git(
            training_repo_root,
            "rev-parse",
            "--show-toplevel",
        ).stdout.strip(),
        "training Git top-level directory",
    )
    if top_level != training_repo_root:
        raise EvaluationContractError(
            "screen-recorded training repository is not its Git top level"
        )
    _require_disjoint(
        training_repo_root,
        evaluation_repo_root,
        "training repository",
        "evaluation repository",
    )
    _assert_clean_commit(
        training_repo_root,
        training_commit,
        label="training repository",
    )
    training_tree = _git(
        training_repo_root,
        "rev-parse",
        f"{training_commit}^{{tree}}",
    ).stdout.strip()
    evaluation_training_tree = _git(
        evaluation_repo_root,
        "rev-parse",
        f"{training_commit}^{{tree}}",
    ).stdout.strip()
    if training_tree != evaluation_training_tree:
        raise EvaluationContractError(
            "training commit tree differs between training and evaluation "
            "repositories"
        )
    return {
        "screen_recorded_path": recorded_repository_root,
        "path": str(training_repo_root),
        "git_commit": training_commit,
        "git_tree": training_tree,
        "clean": True,
        "git_top_level": True,
        "disjoint_from_evaluation_repository": True,
    }


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise EvaluationContractError(
            f"could not read {relative_path} at {commit}: "
            + completed.stderr.decode(errors="replace").strip()
        )
    return completed.stdout


def _validate_corrected_commit(
    repo_root: Path,
    training_commit: str,
    evaluation_commit: str,
) -> dict[str, Any]:
    if training_commit == evaluation_commit:
        raise EvaluationContractError(
            "post-training evaluation must use a fresh corrected commit"
        )
    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        training_commit,
        evaluation_commit,
        check=False,
    )
    if ancestry.returncode != 0:
        raise EvaluationContractError(
            "training commit must be an ancestor of the evaluation commit"
        )

    records = {}
    for relative_path in CORRECTION_PATHS:
        before = _git_blob(repo_root, training_commit, relative_path)
        after = _git_blob(repo_root, evaluation_commit, relative_path)
        if before == after:
            raise EvaluationContractError(
                f"corrected commit does not change {relative_path}"
            )
        decoded = after.decode("utf-8", errors="strict")
        missing = [
            marker
            for marker in CORRECTION_MARKERS[relative_path]
            if marker not in decoded
        ]
        if missing:
            raise EvaluationContractError(
                f"corrected commit lacks required sampler markers in "
                f"{relative_path}: {missing}"
            )
        records[relative_path] = {
            "training_blob_sha256": hashlib.sha256(before).hexdigest(),
            "evaluation_blob_sha256": hashlib.sha256(after).hexdigest(),
            "required_markers": list(CORRECTION_MARKERS[relative_path]),
        }
    return {
        "training_commit_is_ancestor": True,
        "semantic_contract": (
            "evaluation source autonomous_shuffled preserves each local "
            "sample's corruption noise and observed history, shuffles only "
            "its generated noise-subtracted future residual at every step, "
            "and never consumes clean hidden-future TF"
        ),
        "files": records,
    }


def _file_record(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }


def _config_path_value(config: Any, dotted: str) -> Any:
    """Resolve a dotted path through mappings and explicitly indexed lists."""
    current = config
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise EvaluationContractError(
                    f"resolved configuration is missing {dotted}"
                )
            current = current[part]
            continue
        if (
            isinstance(current, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
            and part.isdecimal()
            and str(int(part)) == part
        ):
            index = int(part)
            if index >= len(current):
                raise EvaluationContractError(
                    f"resolved configuration is missing {dotted}"
                )
            current = current[index]
            continue
        raise EvaluationContractError(
            f"resolved configuration is missing {dotted}"
        )
    return current


def _config_container(config: Any, dotted: str) -> Any:
    from omegaconf import OmegaConf

    current = _config_path_value(config, dotted)
    return OmegaConf.to_container(current, resolve=True)


def _config_value(config: Any, dotted: str) -> Any:
    return _config_path_value(config, dotted)


def _hash_config_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_resolved_evaluation_contract(
    config_path: Path,
    *,
    arm: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    expected = {
        "seed": 1234,
        "viz_data_loader.0.batch_size": 1,
        "viz_dataset.img_augment": False,
        "model.dual_diffusion.enabled": True,
        "model.dual_diffusion.condition_on_tf": arm["condition_on_tf"],
        "model.dual_diffusion.condition_mode": arm["condition_mode"],
        "model.dual_diffusion.state_gate_trainable": False,
        "model.dual_diffusion.evaluation_nfe_steps": list(NFE_STEPS),
        "model.dual_diffusion.evaluation_noise_seed": (
            BASE_EVALUATION_NOISE_SEED
        ),
        "model.dual_diffusion.evaluation_condition_sources": list(
            SOURCE_SCREEN_CONDITION_SOURCES
        ),
        "model.dual_diffusion.capture_latent_trajectories": True,
        "model.dual_diffusion.schedule_mode": "tf_leads",
        "model.dual_diffusion.tf_lead_logit": 1.0,
        "trainer.config.dtype": "bfloat16",
        "trainer.config.amp_enabled": True,
    }
    problems = []
    for dotted, wanted in expected.items():
        actual = _config_value(config, dotted)
        if hasattr(actual, "__iter__") and not isinstance(
            actual, (str, bytes, Mapping)
        ):
            actual = list(actual)
        if actual != wanted:
            problems.append(f"{dotted}: {actual!r} != {wanted!r}")
    for dotted in (
        "model.dual_diffusion.state_gate_init",
        "model.forward_model.dual_diffusion.state_gate_init",
    ):
        actual = float(_config_value(config, dotted))
        if not math.isclose(
            actual, float(arm["state_gate_init"]), rel_tol=0.0, abs_tol=1e-12
        ):
            problems.append(
                f"{dotted}: {actual!r} != {arm['state_gate_init']!r}"
            )
    datasets = list(_config_value(config, "viz_dataset.datasets").keys())
    if datasets != ["ABC"]:
        problems.append(f"viz_dataset.datasets: {datasets!r} != ['ABC']")
    if problems:
        raise EvaluationContractError(
            f"invalid resolved evaluation contract for {arm['name']}: "
            + "; ".join(problems)
        )

    viz_dataset = _config_container(config, "viz_dataset")
    viz_loader = _config_container(config, "viz_data_loader.0")
    return config, {
        "viz_dataset_sha256": _hash_config_value(viz_dataset),
        "viz_loader_sha256": _hash_config_value(viz_loader),
        "viz_dataset": viz_dataset,
        "viz_loader": viz_loader,
    }


def _expected_arm_contract() -> dict[str, dict[str, Any]]:
    return {
        arm["name"]: {
            "array_task_id": index,
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
        }
        for index, arm in enumerate(causal.ARMS)
    }


def _collect_source_inputs(
    *,
    repo_root: Path,
    screen_root: Path,
    data_root: Path,
    expected_evaluation_commit: str,
) -> dict[str, Any]:
    _assert_clean_commit(repo_root, expected_evaluation_commit)
    screen_manifest_path = _canonical_file(
        screen_root / "screen_manifest.json", "screen manifest"
    )
    screen = _load_json(screen_manifest_path, "screen manifest")
    if screen.get("kind") != "dual_abc_tf_causal_screen":
        raise EvaluationContractError("screen manifest kind is invalid")
    if not causal._identity_is_valid(screen):
        raise EvaluationContractError("screen manifest identity is invalid")
    training_commit = _validated_commit(
        str(screen.get("git_commit")), "training commit"
    )
    if screen.get("config", {}).get("arms") != causal._arm_manifest_contract():
        raise EvaluationContractError("screen arm contract differs")
    expected_schedule = {
        "optimizer_updates": COMPLETED_UPDATES,
        "gpus_per_arm": WORLD_SIZE,
        "batch_size_per_gpu": 1,
        "evaluation_nfe_steps": list(NFE_STEPS),
        "evaluation_noise_seed": BASE_EVALUATION_NOISE_SEED,
        "evaluation_condition_sources": list(
            SOURCE_SCREEN_CONDITION_SOURCES
        ),
    }
    problems = [
        f"schedule.{key}: {screen.get('schedule', {}).get(key)!r} != {wanted!r}"
        for key, wanted in expected_schedule.items()
        if screen.get("schedule", {}).get(key) != wanted
    ]
    if problems:
        raise EvaluationContractError(
            "screen schedule differs: " + "; ".join(problems)
        )
    recorded_screen_root = Path(screen.get("paths", {}).get("screen_root", ""))
    if recorded_screen_root != screen_root:
        raise EvaluationContractError(
            f"screen root mismatch: {recorded_screen_root} != {screen_root}"
        )
    recorded_data_root = Path(screen.get("paths", {}).get("data_root", ""))
    if recorded_data_root != data_root:
        raise EvaluationContractError(
            f"data root mismatch: {recorded_data_root} != {data_root}"
        )
    training_repository = _validate_training_repository(
        screen.get("repository_root"),
        evaluation_repo_root=repo_root,
        training_commit=training_commit,
    )
    wan_dir = _canonical_directory(
        screen.get("paths", {}).get("wan_dir", ""),
        "screen-recorded Wan directory",
    )
    videox_home = _canonical_directory(
        screen.get("paths", {}).get("videox_home", ""),
        "screen-recorded VideoX-Fun checkout",
    )
    null_prompt_path = _canonical_file(
        wan_dir / "null_prompt_umt5.pt", "null-prompt embedding"
    )
    scheduler_config_path = _canonical_file(
        videox_home / "config" / "wan2.1" / "wan_civitai.yaml",
        "Wan scheduler configuration",
    )
    videox_commit = _git(videox_home, "rev-parse", "HEAD").stdout.strip()
    if videox_commit != EXPECTED_VIDEOX_COMMIT:
        raise EvaluationContractError(
            f"VideoX-Fun commit differs: {videox_commit} != "
            f"{EXPECTED_VIDEOX_COMMIT}"
        )
    current_abc = pilot._abc_manifest_summary(data_root)
    if current_abc != screen.get("data", {}).get("abc_manifest"):
        raise EvaluationContractError("ABC manifest differs from screen provenance")

    correction = _validate_corrected_commit(
        repo_root, training_commit, expected_evaluation_commit
    )
    arms: dict[str, Any] = {}
    viz_contracts = []
    for task_id, arm in enumerate(causal.ARMS):
        arm_name = arm["name"]
        arm_dir = _canonical_directory(
            screen_root / arm_name, f"{arm_name} source arm"
        )
        if arm_dir.parent != screen_root:
            raise EvaluationContractError(
                f"{arm_name} is not a direct child of the screen root"
            )
        manifest_path = _canonical_file(
            arm_dir / "arm_manifest.json", f"{arm_name} arm manifest"
        )
        manifest = _load_json(manifest_path, f"{arm_name} arm manifest")
        if (
            manifest.get("kind") != "dual_abc_tf_causal_screen_arm"
            or not causal._identity_is_valid(manifest)
        ):
            raise EvaluationContractError(f"{arm_name} arm manifest is invalid")
        expected_manifest = {
            "screen_id": screen.get("screen_id"),
            "array_task_id": task_id,
            "arm": arm_name,
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
            "git_commit": training_commit,
            "repository_root": training_repository["path"],
        }
        manifest_problems = [
            f"{key}: {manifest.get(key)!r} != {wanted!r}"
            for key, wanted in expected_manifest.items()
            if manifest.get(key) != wanted
        ]
        if manifest_problems:
            raise EvaluationContractError(
                f"{arm_name} manifest differs: " + "; ".join(manifest_problems)
            )
        if manifest.get("run", {}).get("run_dir") != str(arm_dir):
            raise EvaluationContractError(f"{arm_name} run directory differs")
        if manifest.get("run", {}).get("world_size") != WORLD_SIZE:
            raise EvaluationContractError(f"{arm_name} world size differs")
        if (
            manifest.get("run", {}).get("optimizer_updates")
            != COMPLETED_UPDATES
        ):
            raise EvaluationContractError(
                f"{arm_name} completed-update contract differs"
            )
        recorded_screen = manifest.get("screen_manifest", {})
        if (
            recorded_screen.get("path") != str(screen_manifest_path)
            or recorded_screen.get("sha256") != _sha256(screen_manifest_path)
            or recorded_screen.get("identity_sha256")
            != screen.get("identity_sha256")
        ):
            raise EvaluationContractError(
                f"{arm_name} does not reference the exact screen manifest"
            )

        resolved_path = _canonical_file(
            arm_dir / "resolved_config.yaml", f"{arm_name} resolved config"
        )
        if manifest.get("config", {}).get("resolved_path") != str(resolved_path):
            raise EvaluationContractError(
                f"{arm_name} resolved-config path differs"
            )
        resolved_record = _file_record(resolved_path)
        if (
            manifest.get("config", {}).get("resolved_sha256")
            != resolved_record["sha256"]
        ):
            raise EvaluationContractError(
                f"{arm_name} resolved-config hash differs"
            )
        _, viz_contract = _validate_resolved_evaluation_contract(
            resolved_path, arm=arm
        )
        viz_contracts.append(viz_contract)

        completion_path = _canonical_file(
            arm_dir / "training_complete.json",
            f"{arm_name} training completion",
        )
        completion = _load_json(
            completion_path, f"{arm_name} training completion"
        )
        snapshot_path = _canonical_file(
            arm_dir / "snapshot.pt", f"{arm_name} snapshot"
        )
        expected_completion = {
            "schema_version": 1,
            "status": "completed",
            "completed_updates": COMPLETED_UPDATES,
            "max_iter": COMPLETED_UPDATES,
            "run_identity_sha256": manifest.get("identity_sha256"),
            "snapshot": str(snapshot_path),
        }
        completion_problems = [
            f"{key}: {completion.get(key)!r} != {wanted!r}"
            for key, wanted in expected_completion.items()
            if completion.get(key) != wanted
        ]
        if completion_problems:
            raise EvaluationContractError(
                f"{arm_name} training completion differs: "
                + "; ".join(completion_problems)
            )

        outcome_path = _canonical_file(
            arm_dir / "outcome.json", f"{arm_name} outcome"
        )
        outcome = _load_json(outcome_path, f"{arm_name} outcome")
        if (
            outcome.get("kind")
            != "dual_abc_tf_causal_screen_arm_outcome"
            or outcome.get("completed") is not True
            or outcome.get("exit_status") != 0
        ):
            raise EvaluationContractError(
                f"{arm_name} does not have a successful terminal outcome"
            )
        if (
            outcome.get("manifest", {}).get("sha256") != _sha256(manifest_path)
            or outcome.get("manifest", {}).get("identity_sha256")
            != manifest.get("identity_sha256")
        ):
            raise EvaluationContractError(
                f"{arm_name} outcome/manifest identity differs"
            )
        completion_record = _file_record(completion_path)
        if (
            outcome.get("training_completion", {}).get("sha256")
            != completion_record["sha256"]
            or outcome.get("training_completion", {}).get("path")
            != str(completion_path)
        ):
            raise EvaluationContractError(
                f"{arm_name} outcome/completion hash differs"
            )
        snapshot_record = _file_record(snapshot_path)
        recorded_snapshot = outcome.get("snapshot", {})
        for key in ("path", "sha256", "bytes"):
            if recorded_snapshot.get(key) != snapshot_record[key]:
                raise EvaluationContractError(
                    f"{arm_name} outcome snapshot {key} differs"
                )

        arms[arm_name] = {
            **_expected_arm_contract()[arm_name],
            "source_directory": str(arm_dir),
            "arm_manifest": {
                **_file_record(manifest_path),
                "identity_sha256": manifest.get("identity_sha256"),
            },
            "resolved_config": resolved_record,
            "training_completion": completion_record,
            "outcome": _file_record(outcome_path),
            "snapshot": snapshot_record,
            "viz_dataset_sha256": viz_contract["viz_dataset_sha256"],
            "viz_loader_sha256": viz_contract["viz_loader_sha256"],
        }

    unique_viz_datasets = {
        contract["viz_dataset_sha256"] for contract in viz_contracts
    }
    unique_viz_loaders = {
        contract["viz_loader_sha256"] for contract in viz_contracts
    }
    if len(unique_viz_datasets) != 1 or len(unique_viz_loaders) != 1:
        raise EvaluationContractError(
            "the five arms do not share an identical visualization input contract"
        )
    return {
        "training_commit": training_commit,
        "evaluation_commit": expected_evaluation_commit,
        "training_repository": training_repository,
        "correction": correction,
        "screen": {
            **_file_record(screen_manifest_path),
            "identity_sha256": screen.get("identity_sha256"),
            "screen_id": screen.get("screen_id"),
            "screen_root": str(screen_root),
        },
        "data": {
            "root": str(data_root),
            "datasets": ["ABC"],
            "abc_manifest": current_abc,
            "viz_dataset_sha256": next(iter(unique_viz_datasets)),
            "viz_loader_sha256": next(iter(unique_viz_loaders)),
        },
        "assets": {
            "wan_directory": str(wan_dir),
            "videox_home": str(videox_home),
            "videox_commit": videox_commit,
            "null_prompt": _file_record(null_prompt_path),
            "scheduler_config": _file_record(scheduler_config_path),
        },
        "arms": arms,
    }


def _evaluation_root_candidate(
    *,
    analysis_root: Path,
    evaluation_id: str,
    screen_root: Path,
) -> Path:
    evaluation_id = _validated_id(evaluation_id, "evaluation ID")
    candidate = analysis_root / evaluation_id
    if candidate.exists() or candidate.is_symlink():
        raise EvaluationContractError(
            f"evaluation root must be fresh: {candidate}"
        )
    _require_disjoint(candidate, screen_root, "evaluation root", "screen root")
    return candidate


def _build_evaluation_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    evaluation_commit = _validated_commit(
        args.expected_evaluation_commit, "evaluation commit"
    )
    batches_per_rank = _validated_batches_per_rank(args.batches_per_rank)
    selected_arms = _selected_arm_names(args.arms)
    allowed_active_job_ids = _validated_allowed_active_job_ids(
        args.allow_active_job_id
    )
    repo_root = _canonical_directory(args.repo_root, "repository root")
    if not (repo_root / ".git").is_dir():
        raise EvaluationContractError(f"not a Git repository: {repo_root}")
    screen_root = _canonical_directory(args.screen_root, "screen root")
    analysis_root = _canonical_directory(args.analysis_root, "analysis root")
    data_root = _canonical_directory(args.data_root, "data root")
    evaluation_root = _evaluation_root_candidate(
        analysis_root=analysis_root,
        evaluation_id=args.evaluation_id,
        screen_root=screen_root,
    )
    source = _collect_source_inputs(
        repo_root=repo_root,
        screen_root=screen_root,
        data_root=data_root,
        expected_evaluation_commit=evaluation_commit,
    )
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "dual_abc_tf_causal_screen_posthoc_evaluation",
            "created_at_utc": _now(),
            "evaluation_id": args.evaluation_id,
            "repository_root": str(repo_root),
            "evaluation_root": str(evaluation_root),
            "training_commit": source["training_commit"],
            "evaluation_commit": source["evaluation_commit"],
            "training_repository": source["training_repository"],
            "corrected_sampler": source["correction"],
            "source_screen": source["screen"],
            "data": source["data"],
            "assets": source["assets"],
            "source_arm_inventory": source["arms"],
            "arms": {
                name: source["arms"][name] for name in selected_arms
            },
            "outputs": {
                "analyzer_compatible": True,
                "paired_key": ["dataset", "global_rank"],
                "batch_index_encoding": (
                    "dataset directory suffix __batch_NNN"
                ),
                "arm_roots": {
                    name: str(
                        evaluation_root
                        / "artifacts"
                        / name
                        / f"iter_{EVALUATION_ITERATION}"
                    )
                    for name in selected_arms
                },
            },
            "active_job_coexistence": {
                "allowlisted_active_job_ids": list(
                    allowed_active_job_ids
                ),
                "active_user_job_ids_observed_at_submission_gate": list(
                    allowed_active_job_ids
                ),
                "exact_set_match_required": True,
                "wildcards_and_job_name_bypasses_supported": False,
                "existing_jobs_are_read_only_and_untouched": True,
                "empty_allowlist_requires_no_active_user_jobs": True,
            },
            "paired_evaluation": {
                "world_size": WORLD_SIZE,
                "batches_per_rank": batches_per_rank,
                "paired_unit_count": WORLD_SIZE * batches_per_rank,
                "batch_size": 1,
                "input_loader": (
                    "one rank-sharded ABC viz loader instantiated once after "
                    "NCCL initialization; num_workers=0; cached on CPU"
                ),
                "arm_order": selected_arms,
                "arm_subset_is_explicit": len(selected_arms) < len(causal.ARMS),
                "evaluation_iteration": EVALUATION_ITERATION,
                "nfe_steps": list(NFE_STEPS),
                "source_screen_condition_sources": list(
                    SOURCE_SCREEN_CONDITION_SOURCES
                ),
                "condition_sources": list(CONDITION_SOURCES),
                "runtime_condition_source_override": True,
                "oracle_sources_are_leakage": True,
                "base_evaluation_noise_seed": BASE_EVALUATION_NOISE_SEED,
                "model_seed_for_batch": (
                    "base_evaluation_noise_seed + world_size * batch_index"
                ),
                "effective_generator_seed": (
                    "base_evaluation_noise_seed + global_rank + "
                    "world_size * batch_index"
                ),
                "sigma_convention": SIGMA_CONVENTION,
                "no_cross_example_averaging": True,
            },
            "execution": {
                "evaluation_only": True,
                "strict_snapshot_load": True,
                "sequential_arm_load": True,
                "wandb_enabled": False,
                "resume": False,
                "requeue": False,
                "source_writes_performed": False,
                "condition_source_subset_supported": False,
                "nfe_subset_supported": False,
            },
        }
    )
    return evaluation_root, payload


def command_validate(args: argparse.Namespace) -> int:
    evaluation_root, payload = _build_evaluation_manifest(args)
    print(
        json.dumps(
            {
                "evaluation_root": str(evaluation_root),
                "training_commit": payload["training_commit"],
                "evaluation_commit": payload["evaluation_commit"],
                "screen_id": payload["source_screen"]["screen_id"],
                "arms": list(payload["arms"]),
                "paired_unit_count": payload["paired_evaluation"][
                    "paired_unit_count"
                ],
                "allowlisted_active_job_ids": payload[
                    "active_job_coexistence"
                ]["allowlisted_active_job_ids"],
                "identity_sha256": payload["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    evaluation_root, payload = _build_evaluation_manifest(args)
    os.mkdir(evaluation_root, mode=0o700)
    manifest_path = evaluation_root / "evaluation_manifest.json"
    _exclusive_json(manifest_path, payload)
    print(payload["identity_sha256"])
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    evaluation_root = _canonical_directory(
        args.evaluation_root, "evaluation root"
    )
    manifest_path = _canonical_file(
        args.evaluation_manifest, "evaluation manifest"
    )
    if manifest_path.parent != evaluation_root:
        raise EvaluationContractError(
            "evaluation manifest must be directly under evaluation root"
        )
    manifest = _load_json(manifest_path, "evaluation manifest")
    if (
        manifest.get("kind")
        != "dual_abc_tf_causal_screen_posthoc_evaluation"
        or not _identity_is_valid(manifest)
    ):
        raise EvaluationContractError("evaluation manifest is invalid")
    if not JOB_ID_RE.fullmatch(args.job_id):
        raise EvaluationContractError("Slurm returned an invalid job ID")
    coexistence = _validate_active_job_coexistence(
        manifest.get("active_job_coexistence")
    )
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "dual_abc_tf_causal_screen_posthoc_submission",
            "created_at_utc": _now(),
            "evaluation_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "identity_sha256": manifest.get("identity_sha256"),
            },
            "slurm_job_id": args.job_id,
            "active_job_coexistence": coexistence,
            "requeue": False,
            "resume": False,
        }
    )
    _exclusive_json(evaluation_root / "slurm_submission.json", payload)
    return 0


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(list(value.shape)))
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _update_value_hash(digest: Any, value: Any, path: str) -> None:
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor\0")
        digest.update(_tensor_sha256(value).encode("ascii"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value):
            if not isinstance(key, str):
                raise EvaluationContractError(
                    f"batch mapping key is not a string at {path}: {key!r}"
                )
            _update_value_hash(digest, value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for index, item in enumerate(value):
            _update_value_hash(digest, item, f"{path}[{index}]")
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(b"scalar\0")
        digest.update(_canonical_json_bytes(value))
    else:
        raise EvaluationContractError(
            f"unsupported batch value at {path}: {type(value).__name__}"
        )
    digest.update(b"\0")


def _batch_sha256(batch: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_value_hash(digest, batch, "batch")
    return digest.hexdigest()


def _flatten_batch_tensors(
    value: Any,
    *,
    prefix: str = "",
    output: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if output is None:
        output = {}
    if isinstance(value, torch.Tensor):
        if not prefix:
            raise EvaluationContractError("batch tensor has no key path")
        output[prefix] = value.detach().cpu().contiguous()
    elif isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str):
                raise EvaluationContractError("batch mapping keys must be strings")
            child = key if not prefix else f"{prefix}.{key}"
            _flatten_batch_tensors(value[key], prefix=child, output=output)
    else:
        raise EvaluationContractError(
            "evaluation batches must contain only tensors and nested mappings; "
            f"got {type(value).__name__} at {prefix}"
        )
    return output


def _batch_summary(batch: Mapping[str, Any]) -> dict[str, Any]:
    tensors = _flatten_batch_tensors(batch)
    return {
        "sha256": _batch_sha256(batch),
        "tensor_count": len(tensors),
        "tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
            for key, value in tensors.items()
        },
    }


def _pair_dataset_name(base_dataset: str, batch_index: int) -> str:
    if batch_index < 0:
        raise EvaluationContractError("batch index must be non-negative")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base_dataset).strip("_")
    if not sanitized:
        raise EvaluationContractError("dataset name is empty after sanitization")
    return f"{sanitized}__batch_{batch_index:03d}"


def _model_noise_seed(batch_index: int) -> int:
    if batch_index < 0:
        raise EvaluationContractError("batch index must be non-negative")
    return BASE_EVALUATION_NOISE_SEED + WORLD_SIZE * batch_index


def _effective_generator_seed(global_rank: int, batch_index: int) -> int:
    if global_rank < 0 or global_rank >= WORLD_SIZE:
        raise EvaluationContractError("global rank is outside the evaluation world")
    return _model_noise_seed(batch_index) + global_rank


def _clone_batch_to_device(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone()
        if use_amp:
            return clone.to(device=device, non_blocking=False)
        return clone.to(device=device, dtype=dtype, non_blocking=False)
    if isinstance(value, Mapping):
        return {
            key: _clone_batch_to_device(
                item, device=device, dtype=dtype, use_amp=use_amp
            )
            for key, item in value.items()
        }
    raise EvaluationContractError(
        f"unsupported batch value during device transfer: {type(value).__name__}"
    )


def _strict_state_dict_contract(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, torch.Tensor],
) -> None:
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    extra = sorted(checkpoint_keys - model_keys)
    if missing or extra:
        raise EvaluationContractError(
            f"snapshot/model key mismatch; missing={missing}, extra={extra}"
        )
    problems = []
    for key in sorted(model_keys):
        expected = model_state[key]
        actual = checkpoint_state[key]
        if not isinstance(actual, torch.Tensor):
            problems.append(f"{key}: checkpoint value is not a tensor")
        elif actual.shape != expected.shape:
            problems.append(
                f"{key}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        elif actual.dtype != expected.dtype:
            problems.append(f"{key}: dtype {actual.dtype} != {expected.dtype}")
    if problems:
        raise EvaluationContractError(
            "strict snapshot tensor schema differs: " + "; ".join(problems)
        )


def _validate_snapshot_envelope(
    snapshot: Mapping[str, Any],
    *,
    expected_run_identity: str,
) -> Mapping[str, torch.Tensor]:
    expected = {
        "snapshot_schema_version": 3,
        "_start_iter": COMPLETED_UPDATES,
        "world_size": WORLD_SIZE,
        "gradient_accumulation_steps": 1,
        "run_identity_sha256": expected_run_identity,
    }
    problems = [
        f"{key}: {snapshot.get(key)!r} != {wanted!r}"
        for key, wanted in expected.items()
        if snapshot.get(key) != wanted
    ]
    state = snapshot.get("model")
    if not isinstance(state, Mapping) or not state:
        problems.append("model: missing or empty state dictionary")
    if problems:
        raise EvaluationContractError(
            "snapshot envelope differs: " + "; ".join(problems)
        )
    return state


def _future_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history_frames: int,
    key: str,
) -> float:
    prediction_future = prediction[:, :, history_frames:].double()
    target_future = target[:, :, history_frames:].double()
    denominator = torch.sum(target_future.square()).item()
    numerator = torch.sum(
        (prediction_future - target_future).square()
    ).item()
    if denominator <= 0 or not all(
        math.isfinite(value) for value in (denominator, numerator)
    ):
        raise EvaluationContractError(f"{key} has invalid future NMSE energy")
    return float(numerator / denominator)


def _decoded_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction_float = prediction.double().div(255.0)
    target_float = target.double().div(255.0)
    mse = torch.mean((prediction_float - target_float).square()).item()
    temporal_error = torch.diff(
        prediction_float, dim=2
    ) - torch.diff(target_float, dim=2)
    temporal_mse = torch.mean(temporal_error.square()).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    if not all(math.isfinite(value) for value in (mse, temporal_mse, psnr)):
        raise EvaluationContractError("decoded metrics are non-finite")
    return {
        "decoded_mse_unit_range": float(mse),
        "decoded_psnr_db": float(psnr),
        "decoded_temporal_difference_mse_unit_range": float(temporal_mse),
    }


def _validate_and_measure_artifacts(
    artifacts: Mapping[str, torch.Tensor],
    *,
    arm: Mapping[str, Any],
    expected_model_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = _required_evaluation_tensor_names()
    missing = sorted(required - set(artifacts))
    invalid_types = {
        key: type(value).__name__
        for key, value in artifacts.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    }
    if missing or invalid_types:
        raise EvaluationContractError(
            f"invalid visualization artifacts; missing={missing}, "
            f"invalid_types={invalid_types}"
        )
    expected_scalars = {
        "condition_on_tf": int(arm["condition_on_tf"]),
        "condition_mode_code": MODE_CODES[arm["condition_mode"]],
        "evaluation_noise_seed": expected_model_seed,
        "oracle_sources_are_leakage": 1,
    }
    problems = []
    for key, wanted in expected_scalars.items():
        tensor = artifacts[key]
        if tensor.numel() != 1 or int(tensor.item()) != wanted:
            problems.append(f"{key}: {tensor.tolist()} != [{wanted}]")
    if tuple(int(x) for x in artifacts["evaluation_nfe_steps"].tolist()) != NFE_STEPS:
        problems.append("evaluation_nfe_steps differs")
    expected_source_codes = tuple(SOURCE_CODES[source] for source in CONDITION_SOURCES)
    if tuple(
        int(x)
        for x in artifacts["evaluation_condition_source_codes"].tolist()
    ) != expected_source_codes:
        problems.append("evaluation_condition_source_codes differs")
    if problems:
        raise EvaluationContractError(
            "artifact source/intervention contract differs: " + "; ".join(problems)
        )

    video_clean = artifacts["video_clean"]
    tf_clean = artifacts["tf_clean"]
    ground_truth = artifacts["ground_truth_future_uint8"]
    initial_video = artifacts["video_initial_state"]
    initial_tf = artifacts["tf_initial_state"]
    initial_tf_noise = artifacts["tf_initial_noise"]
    history_frames = int(artifacts["history_latent_frames"].item())
    auxiliary_history_frames = int(
        artifacts.get(
            "auxiliary_history_latent_frames",
            artifacts["history_latent_frames"],
        ).item()
    )
    if (
        video_clean.ndim != 5
        or tf_clean.ndim != 5
        or ground_truth.ndim != 5
        or ground_truth.dtype != torch.uint8
        or history_frames < 0
        or history_frames >= video_clean.shape[2]
        or auxiliary_history_frames < 0
        or auxiliary_history_frames >= tf_clean.shape[2]
    ):
        raise EvaluationContractError("artifact clean/history tensor contract differs")
    if (
        initial_video.shape != video_clean.shape
        or initial_tf.shape != tf_clean.shape
        or initial_tf_noise.shape != tf_clean.shape
    ):
        raise EvaluationContractError("artifact initial-state shape differs")
    if not torch.equal(
        initial_tf[:, :, :auxiliary_history_frames],
        tf_clean[:, :, :auxiliary_history_frames],
    ) or not torch.equal(
        initial_tf[:, :, auxiliary_history_frames:],
        initial_tf_noise[:, :, auxiliary_history_frames:],
    ):
        raise EvaluationContractError(
            "TF initial state does not preserve clean history/local future noise"
        )

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for source in CONDITION_SOURCES:
        infix = "" if source == "autonomous" else f"_{source}"
        metrics[source] = {}
        for nfe in NFE_STEPS:
            video_key = f"video_final{infix}_nfe_{nfe}"
            tf_key = f"tf_final{infix}_nfe_{nfe}"
            decoded_key = f"decoded_future{infix}_nfe_{nfe}"
            video = artifacts[video_key]
            tf = artifacts[tf_key]
            decoded = artifacts[decoded_key]
            if video.shape != video_clean.shape or tf.shape != tf_clean.shape:
                raise EvaluationContractError(
                    f"{source}/NFE={nfe} latent shape differs"
                )
            if decoded.shape != ground_truth.shape or decoded.dtype != torch.uint8:
                raise EvaluationContractError(
                    f"{source}/NFE={nfe} decoded tensor differs"
                )
            metrics[source][str(nfe)] = {
                "video_future_nmse": _future_nmse(
                    video, video_clean, history_frames, video_key
                ),
                "tf_future_nmse": _future_nmse(
                    tf, tf_clean, auxiliary_history_frames, tf_key
                ),
                **_decoded_metrics(decoded, ground_truth),
            }

    for prefix in ("video_final", "tf_final", "decoded_future"):
        if not torch.equal(
            artifacts[f"{prefix}_nfe_1"],
            artifacts[f"{prefix}_autonomous_shuffled_nfe_1"],
        ):
            raise EvaluationContractError(
                "autonomous_shuffled must be an exact pure-noise endpoint "
                f"no-op at NFE=1: {prefix}"
            )

    paired_identity = {
        "history_latent_frames": history_frames,
        "auxiliary_history_latent_frames": auxiliary_history_frames,
        "video_clean_sha256": _tensor_sha256(video_clean),
        "tf_clean_sha256": _tensor_sha256(tf_clean),
        "ground_truth_future_uint8_sha256": _tensor_sha256(ground_truth),
        "video_initial_state_sha256": _tensor_sha256(initial_video),
        "tf_initial_state_sha256": _tensor_sha256(initial_tf),
        "tf_initial_noise_sha256": _tensor_sha256(initial_tf_noise),
        "evaluation_noise_seed": expected_model_seed,
        "evaluation_nfe_steps": list(NFE_STEPS),
        "evaluation_condition_sources": list(CONDITION_SOURCES),
        "oracle_sources_are_leakage": True,
    }
    return metrics, paired_identity


def _save_safetensors_exclusive(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
) -> None:
    from safetensors.torch import save

    normalized = {
        key: value.detach().cpu().contiguous()
        for key, value in tensors.items()
    }
    encoded = save(normalized, metadata=dict(metadata))
    _exclusive_bytes(path, encoded)


def _save_input_batch(
    *,
    evaluation_root: Path,
    dataset: str,
    rank: int,
    batch_index: int,
    batch: Mapping[str, Any],
    batch_summary: Mapping[str, Any],
) -> dict[str, Any]:
    folder = evaluation_root / "inputs" / dataset
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / f"input_rank_{rank}.safetensors"
    tensors = _flatten_batch_tensors(batch)
    _save_safetensors_exclusive(
        path,
        tensors,
        metadata={
            "dataset": dataset,
            "global_rank": str(rank),
            "batch_index": str(batch_index),
            "batch_sha256": str(batch_summary["sha256"]),
        },
    )
    record = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "dual_abc_tf_causal_screen_posthoc_input",
            "dataset": dataset,
            "global_rank": rank,
            "batch_index": batch_index,
            "batch_sha256": batch_summary["sha256"],
            "tensor_schema": batch_summary["tensors"],
            "safetensors_sha256": _sha256(path),
            "safetensors_path": str(path),
        }
    )
    sidecar_path = path.with_suffix(".json")
    _exclusive_json(sidecar_path, record)
    return {
        **record,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
    }


def _save_output_artifact(
    *,
    evaluation_root: Path,
    arm: Mapping[str, Any],
    dataset: str,
    rank: int,
    batch_index: int,
    batch_sha256: str,
    model_seed: int,
    effective_seed: int,
    artifacts: Mapping[str, torch.Tensor],
    metrics: Mapping[str, Any],
    paired_identity: Mapping[str, Any],
    aggregate_sampling_timing: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    evaluation_commit: str,
) -> dict[str, Any]:
    folder = (
        evaluation_root
        / "artifacts"
        / str(arm["name"])
        / f"iter_{EVALUATION_ITERATION}"
        / dataset
    )
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / f"latent_trajectory_rank_{rank}.safetensors"
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in artifacts.items()
    }
    _save_safetensors_exclusive(
        path,
        tensors,
        metadata={
            "iteration": str(EVALUATION_ITERATION),
            "dataset": dataset,
            "sigma_convention": SIGMA_CONVENTION,
            "evaluation_commit": evaluation_commit,
            "snapshot_sha256": str(snapshot["sha256"]),
            "input_batch_sha256": batch_sha256,
        },
    )
    safetensors_sha256 = _sha256(path)
    sidecar = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "dual_abc_tf_causal_screen_posthoc_artifact",
            "iteration": EVALUATION_ITERATION,
            "dataset": dataset,
            "global_rank": rank,
            "batch_index": batch_index,
            "pair_key": {"dataset": dataset, "global_rank": rank},
            "sigma_convention": SIGMA_CONVENTION,
            "arm": arm["name"],
            "intervention": {
                "condition_on_tf": arm["condition_on_tf"],
                "condition_mode": arm["condition_mode"],
                "state_gate_init": arm["state_gate_init"],
                "state_gate_trainable": False,
            },
            "evaluation_commit": evaluation_commit,
            "snapshot": dict(snapshot),
            "resolved_config": dict(resolved_config),
            "input_batch_sha256": batch_sha256,
            "model_evaluation_noise_seed": model_seed,
            "effective_generator_seed": effective_seed,
            "source_contract": {
                "nfe_steps": list(NFE_STEPS),
                "condition_sources": list(CONDITION_SOURCES),
                "source_screen_condition_sources": list(
                    SOURCE_SCREEN_CONDITION_SOURCES
                ),
                "autonomous_shuffled": {
                    "same_checkpoint": True,
                    "uses_clean_hidden_future": False,
                    "preserves_local_corruption_noise": True,
                    "preserves_local_observed_history": True,
                    "rolled_quantity": (
                        "generated noise-subtracted future TF residual"
                    ),
                    "roll_scope": "global 8-rank batch at every denoising step",
                },
                "oracle_sources_are_leakage": True,
                "all_sources_reuse_identical_initial_states": True,
            },
            "paired_identity": dict(paired_identity),
            "per_example_metrics": metrics,
            "aggregate_sampling_timing": aggregate_sampling_timing,
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in tensors.items()
            },
            "safetensors_sha256": safetensors_sha256,
        }
    )
    sidecar_path = path.with_suffix(".json")
    _exclusive_json(sidecar_path, sidecar)
    return {
        "arm": arm["name"],
        "dataset": dataset,
        "global_rank": rank,
        "batch_index": batch_index,
        "batch_sha256": batch_sha256,
        "model_evaluation_noise_seed": model_seed,
        "effective_generator_seed": effective_seed,
        "paired_identity": dict(paired_identity),
        "per_example_metrics": metrics,
        "aggregate_sampling_timing": dict(aggregate_sampling_timing),
        "safetensors_path": str(path),
        "safetensors_sha256": safetensors_sha256,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
    }


def _broadcast_rank0(callable_: Any, *, rank: int) -> Any:
    payload: list[Any] = [None]
    if rank == 0:
        try:
            payload[0] = {"ok": True, "value": callable_()}
        except Exception as exc:  # propagated identically to every rank
            payload[0] = {
                "ok": False,
                "type": type(exc).__name__,
                "error": str(exc),
            }
    torch.distributed.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict) or not result.get("ok"):
        raise EvaluationContractError(
            "rank-0 contract validation failed: "
            f"{result.get('type', 'unknown')}: {result.get('error', result)}"
        )
    return result["value"]


def _validate_runtime_inputs(
    *,
    manifest_path: Path,
    repo_root: Path,
    evaluation_root: Path,
    expected_evaluation_commit: str,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "evaluation manifest")
    if (
        manifest.get("kind")
        != "dual_abc_tf_causal_screen_posthoc_evaluation"
        or not _identity_is_valid(manifest)
    ):
        raise EvaluationContractError("evaluation manifest is invalid")
    expected = {
        "repository_root": str(repo_root),
        "evaluation_root": str(evaluation_root),
        "evaluation_commit": expected_evaluation_commit,
    }
    problems = [
        f"{key}: {manifest.get(key)!r} != {wanted!r}"
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    ]
    if problems:
        raise EvaluationContractError(
            "evaluation runtime identity differs: " + "; ".join(problems)
        )
    _validate_active_job_coexistence(
        manifest.get("active_job_coexistence")
    )
    _assert_clean_commit(repo_root, expected_evaluation_commit)
    screen_root = _canonical_directory(
        manifest["source_screen"]["screen_root"], "source screen root"
    )
    data_root = _canonical_directory(manifest["data"]["root"], "data root")
    rebuilt = _collect_source_inputs(
        repo_root=repo_root,
        screen_root=screen_root,
        data_root=data_root,
        expected_evaluation_commit=expected_evaluation_commit,
    )
    comparisons = {
        "training_commit": rebuilt["training_commit"],
        "evaluation_commit": rebuilt["evaluation_commit"],
        "training_repository": rebuilt["training_repository"],
        "corrected_sampler": rebuilt["correction"],
        "source_screen": rebuilt["screen"],
        "data": rebuilt["data"],
        "assets": rebuilt["assets"],
        "source_arm_inventory": rebuilt["arms"],
    }
    for key, actual in comparisons.items():
        if manifest.get(key) != actual:
            raise EvaluationContractError(
                f"evaluation input mismatch for {key}"
            )
    selected_arm_names = manifest.get("paired_evaluation", {}).get(
        "arm_order"
    )
    if (
        not isinstance(selected_arm_names, list)
        or _selected_arm_names(selected_arm_names) != selected_arm_names
        or set(manifest.get("arms", {})) != set(selected_arm_names)
        or manifest.get("arms")
        != {
            name: rebuilt["arms"][name] for name in selected_arm_names
        }
    ):
        raise EvaluationContractError("selected evaluation-arm inventory differs")
    if (evaluation_root / "execution_started.json").exists():
        raise EvaluationContractError(
            "evaluation cannot resume: execution marker already exists"
        )
    if (evaluation_root / "evaluation_complete.json").exists():
        raise EvaluationContractError("evaluation is already complete")
    for directory in ("inputs", "artifacts"):
        if (evaluation_root / directory).exists():
            raise EvaluationContractError(
                f"evaluation cannot resume into existing {directory}"
            )
    return manifest


def _load_batches(
    *,
    canonical_config: Path,
    batches_per_rank: int,
) -> tuple[list[Mapping[str, Any]], str]:
    import hydra
    from omegaconf import OmegaConf, open_dict

    config = OmegaConf.load(canonical_config)
    loader_config = OmegaConf.create(
        OmegaConf.to_container(config.viz_data_loader[0], resolve=True)
    )
    with open_dict(loader_config):
        loader_config.num_workers = 0
        loader_config.pin_memory = False
        loader_config.pop("prefetch_factor", None)
        loader_config.pop("persistent_workers", None)
    loader = hydra.utils.instantiate(loader_config)
    if int(loader.batch_size) != 1:
        raise EvaluationContractError("evaluation loader batch size must be one")
    dataset_name = loader.dataset.name + "_0"
    if "MultiDataset" in dataset_name:
        dataset_name = loader.dataset.full_name + "_0"
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(batches_per_rank)]
    del iterator
    del loader
    gc.collect()
    if any(not isinstance(batch, Mapping) for batch in batches):
        raise EvaluationContractError("evaluation loader returned a non-mapping batch")
    return batches, dataset_name


def _dtype_from_config(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if value not in mapping:
        raise EvaluationContractError(f"unsupported evaluation dtype: {value!r}")
    return mapping[value]


def _instantiate_and_load_model(
    *,
    config_path: Path,
    snapshot_path: Path,
    arm: Mapping[str, Any],
    expected_run_identity: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.dtype, bool, dict[str, Any]]:
    import hydra
    from omegaconf import OmegaConf, open_dict

    config = OmegaConf.load(config_path)
    # The completed training screen is an immutable four-source input.  Widen
    # only this in-memory evaluation coordinate to add the fifth, generated
    # wrong-content control; the override is parameter-free and therefore
    # preserves the strict checkpoint key/shape/dtype contract below.
    with open_dict(config.model.dual_diffusion):
        config.model.dual_diffusion.evaluation_condition_sources = list(
            CONDITION_SOURCES
        )
    runtime_sources = tuple(
        str(source)
        for source in config.model.dual_diffusion.evaluation_condition_sources
    )
    if runtime_sources != CONDITION_SOURCES:
        raise EvaluationContractError(
            "runtime condition-source override did not resolve exactly"
        )
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    model = hydra.utils.instantiate(config.model)
    if tuple(model.evaluation_condition_sources) != CONDITION_SOURCES:
        raise EvaluationContractError(
            "instantiated model did not retain the five-source runtime override"
        )
    try:
        snapshot = torch.load(
            snapshot_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        snapshot = torch.load(
            snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(snapshot, Mapping):
        raise EvaluationContractError("snapshot must contain a mapping")
    checkpoint_state = _validate_snapshot_envelope(
        snapshot, expected_run_identity=expected_run_identity
    )
    _strict_state_dict_contract(model.state_dict(), checkpoint_state)
    incompatible = model.load_state_dict(checkpoint_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EvaluationContractError(
            "strict snapshot load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    adapter = model.forward_model.tf_token_adapter
    effective_gate = float(adapter.effective_gate().detach().cpu().item())
    if not math.isclose(
        effective_gate,
        float(arm["state_gate_init"]),
        rel_tol=0.0,
        abs_tol=EFFECTIVE_STATE_GATE_ABS_TOL,
    ):
        raise EvaluationContractError(
            f"{arm['name']} effective gate {effective_gate} differs from "
            f"{arm['state_gate_init']}"
        )
    if adapter.gate.requires_grad:
        raise EvaluationContractError(
            f"{arm['name']} state gate is unexpectedly trainable"
        )
    del snapshot
    del checkpoint_state
    model.to(device)
    model.eval()
    dtype = _dtype_from_config(str(config.trainer.config.dtype))
    use_amp = bool(config.trainer.config.amp_enabled)
    return model, dtype, use_amp, {
        "snapshot_schema_version": 3,
        "next_iteration": COMPLETED_UPDATES,
        "world_size": WORLD_SIZE,
        "gradient_accumulation_steps": 1,
        "strict_key_shape_dtype_match": True,
        "effective_state_gate": effective_gate,
        "source_screen_condition_sources": list(
            SOURCE_SCREEN_CONDITION_SOURCES
        ),
        "runtime_condition_sources": list(CONDITION_SOURCES),
        "runtime_override_is_parameter_free": True,
    }


def _record_matches_file(record: Mapping[str, Any]) -> bool:
    path = _canonical_file(record["path"], "recorded immutable input")
    info = path.stat()
    return (
        info.st_size == record["bytes"]
        and info.st_dev == record["device"]
        and info.st_ino == record["inode"]
        and info.st_mtime_ns == record["mtime_ns"]
        and _sha256(path) == record["sha256"]
    )


def _validate_final_inventory(
    *,
    manifest: Mapping[str, Any],
    input_records: list[Mapping[str, Any]],
    output_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    batches_per_rank = int(
        manifest["paired_evaluation"]["batches_per_rank"]
    )
    expected_pairs = {
        (rank, batch_index)
        for rank in range(WORLD_SIZE)
        for batch_index in range(batches_per_rank)
    }
    observed_inputs = {
        (int(record["global_rank"]), int(record["batch_index"]))
        for record in input_records
    }
    if observed_inputs != expected_pairs or len(input_records) != len(expected_pairs):
        raise EvaluationContractError(
            "input rank/batch inventory is incomplete or duplicated"
        )
    input_hashes = [str(record["batch_sha256"]) for record in input_records]
    if len(set(input_hashes)) != len(input_hashes):
        raise EvaluationContractError(
            "rank/batch input hashes must be globally distinct"
        )

    arm_names = list(manifest["paired_evaluation"]["arm_order"])
    expected_outputs = {
        (arm, rank, batch_index)
        for arm in arm_names
        for rank, batch_index in expected_pairs
    }
    observed_outputs = {
        (
            str(record["arm"]),
            int(record["global_rank"]),
            int(record["batch_index"]),
        )
        for record in output_records
    }
    if (
        observed_outputs != expected_outputs
        or len(output_records) != len(expected_outputs)
    ):
        raise EvaluationContractError(
            "output arm/rank/batch inventory is incomplete or duplicated"
        )

    by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for record in output_records:
        pair = (int(record["global_rank"]), int(record["batch_index"]))
        by_pair.setdefault(pair, []).append(record)
        if _sha256(Path(record["safetensors_path"])) != record[
            "safetensors_sha256"
        ]:
            raise EvaluationContractError("output safetensors hash changed")
        if _sha256(Path(record["sidecar_path"])) != record["sidecar_sha256"]:
            raise EvaluationContractError("output sidecar hash changed")
    input_by_pair = {
        (int(record["global_rank"]), int(record["batch_index"])): record
        for record in input_records
    }
    for record in input_records:
        if (
            _sha256(Path(record["safetensors_path"]))
            != record["safetensors_sha256"]
            or _sha256(Path(record["sidecar_path"]))
            != record["sidecar_sha256"]
        ):
            raise EvaluationContractError("input artifact or sidecar hash changed")
    paired_units = []
    for pair in sorted(expected_pairs):
        records = by_pair[pair]
        identities = {
            _canonical_json_bytes(record["paired_identity"])
            for record in records
        }
        batch_hashes = {record["batch_sha256"] for record in records}
        model_seeds = {record["model_evaluation_noise_seed"] for record in records}
        effective_seeds = {record["effective_generator_seed"] for record in records}
        expected_input_hash = input_by_pair[pair]["batch_sha256"]
        if (
            len(records) != len(arm_names)
            or len(identities) != 1
            or batch_hashes != {expected_input_hash}
            or model_seeds != {_model_noise_seed(pair[1])}
            or effective_seeds != {
                _effective_generator_seed(pair[0], pair[1])
            }
        ):
            raise EvaluationContractError(
                f"cross-arm paired identity differs for rank/batch {pair}"
            )
        paired_units.append(
            {
                "global_rank": pair[0],
                "batch_index": pair[1],
                "dataset": records[0]["dataset"],
                "batch_sha256": expected_input_hash,
                "model_evaluation_noise_seed": next(iter(model_seeds)),
                "effective_generator_seed": next(iter(effective_seeds)),
                "paired_identity": records[0]["paired_identity"],
                "input_artifact": {
                    "safetensors_path": input_by_pair[pair][
                        "safetensors_path"
                    ],
                    "safetensors_sha256": input_by_pair[pair][
                        "safetensors_sha256"
                    ],
                    "sidecar_path": input_by_pair[pair]["sidecar_path"],
                    "sidecar_sha256": input_by_pair[pair]["sidecar_sha256"],
                    "tensor_schema": input_by_pair[pair]["tensor_schema"],
                },
                "arm_artifacts": {
                    record["arm"]: {
                        "safetensors_path": record["safetensors_path"],
                        "safetensors_sha256": record["safetensors_sha256"],
                        "sidecar_path": record["sidecar_path"],
                        "sidecar_sha256": record["sidecar_sha256"],
                        "per_example_metrics": record["per_example_metrics"],
                        "aggregate_sampling_timing": record[
                            "aggregate_sampling_timing"
                        ],
                    }
                    for record in sorted(records, key=lambda item: item["arm"])
                },
            }
        )
    return {
        "paired_unit_count": len(paired_units),
        "artifact_count": len(output_records),
        "input_artifact_count": len(input_records),
        "paired_units": paired_units,
    }


def _runtime_wandb_guard() -> None:
    if os.environ.get("WANDB_MODE") != "disabled":
        raise EvaluationContractError("WANDB_MODE must be exactly 'disabled'")
    if os.environ.get("WANDB_DISABLED", "").lower() not in {"true", "1"}:
        raise EvaluationContractError("WANDB_DISABLED must be true")
    forbidden = (
        "WANDB_RUN_ID",
        "WANDB_RUN_GROUP",
        "WANDB_NAME",
        "WANDB_RESUME",
    )
    present = [key for key in forbidden if os.environ.get(key)]
    if present:
        raise EvaluationContractError(
            f"W&B run variables must be unset: {present}"
        )


def command_run(args: argparse.Namespace) -> int:
    import robot_wm.utils.distributed as distributed

    expected_commit = _validated_commit(
        args.expected_evaluation_commit, "evaluation commit"
    )
    repo_root = _canonical_directory(args.repo_root, "repository root")
    evaluation_root = _canonical_directory(
        args.evaluation_root, "evaluation root"
    )
    manifest_path = _canonical_file(
        args.evaluation_manifest, "evaluation manifest"
    )
    if manifest_path.parent != evaluation_root:
        raise EvaluationContractError(
            "evaluation manifest must be directly under evaluation root"
        )
    _runtime_wandb_guard()
    distributed.init_process_group()
    rank = distributed.get_global_rank()
    local_rank = distributed.get_local_rank()
    world_size = distributed.get_world_size()
    if world_size != WORLD_SIZE:
        distributed.destroy_process_group()
        raise EvaluationContractError(
            f"post-training evaluation requires {WORLD_SIZE} ranks, got {world_size}"
        )
    device = torch.device("cuda", local_rank)

    try:
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        total_evaluation_started = time.perf_counter()
        manifest = _broadcast_rank0(
            lambda: _validate_runtime_inputs(
                manifest_path=manifest_path,
                repo_root=repo_root,
                evaluation_root=evaluation_root,
                expected_evaluation_commit=expected_commit,
            ),
            rank=rank,
        )
        batches_per_rank = int(
            manifest["paired_evaluation"]["batches_per_rank"]
        )
        if rank == 0:
            _exclusive_json(
                evaluation_root / "execution_started.json",
                _identity_payload(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": (
                            "dual_abc_tf_causal_screen_posthoc_execution_started"
                        ),
                        "created_at_utc": _now(),
                        "evaluation_manifest_sha256": _sha256(manifest_path),
                        "evaluation_manifest_identity_sha256": manifest[
                            "identity_sha256"
                        ],
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "active_job_coexistence": manifest[
                            "active_job_coexistence"
                        ],
                        "world_size": WORLD_SIZE,
                        "requeue": False,
                        "resume": False,
                        "wandb_enabled": False,
                    }
                ),
            )
            (evaluation_root / "inputs").mkdir(mode=0o700)
            (evaluation_root / "artifacts").mkdir(mode=0o700)
        torch.distributed.barrier()

        selected_arm_names = list(
            manifest["paired_evaluation"]["arm_order"]
        )
        arm_specs_by_name = {arm["name"]: arm for arm in causal.ARMS}
        canonical_arm = selected_arm_names[0]
        canonical_config = Path(
            manifest["arms"][canonical_arm]["resolved_config"]["path"]
        )
        random.seed(1234 + rank)
        np.random.seed(1234 + rank)
        torch.manual_seed(1234 + rank)
        batches, base_dataset = _load_batches(
            canonical_config=canonical_config,
            batches_per_rank=batches_per_rank,
        )
        initial_batch_summaries = [_batch_summary(batch) for batch in batches]
        local_batch_metadata = [
            {
                "global_rank": rank,
                "batch_index": batch_index,
                "dataset": _pair_dataset_name(base_dataset, batch_index),
                **summary,
            }
            for batch_index, summary in enumerate(initial_batch_summaries)
        ]
        gathered_batch_metadata: list[Any] = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(
            gathered_batch_metadata, local_batch_metadata
        )
        flat_batch_metadata = [
            record
            for rank_records in gathered_batch_metadata
            for record in rank_records
        ]

        def validate_batch_metadata() -> bool:
            observed_pairs = {
                (record["global_rank"], record["batch_index"])
                for record in flat_batch_metadata
            }
            expected_pairs = {
                (expected_rank, batch_index)
                for expected_rank in range(WORLD_SIZE)
                for batch_index in range(batches_per_rank)
            }
            hashes = [record["sha256"] for record in flat_batch_metadata]
            if observed_pairs != expected_pairs or len(set(hashes)) != len(hashes):
                raise EvaluationContractError(
                    "deterministic rank/batch inputs are missing, duplicated, "
                    "or content-identical"
                )
            return True

        _broadcast_rank0(validate_batch_metadata, rank=rank)

        local_input_records = []
        for batch_index, (batch, summary) in enumerate(
            zip(batches, initial_batch_summaries)
        ):
            dataset = _pair_dataset_name(base_dataset, batch_index)
            local_input_records.append(
                _save_input_batch(
                    evaluation_root=evaluation_root,
                    dataset=dataset,
                    rank=rank,
                    batch_index=batch_index,
                    batch=batch,
                    batch_summary=summary,
                )
            )
        torch.distributed.barrier()

        local_output_records = []
        paired_identity_reference: dict[int, Mapping[str, Any]] = {}
        for arm_name in selected_arm_names:
            arm_spec = arm_specs_by_name[arm_name]
            arm_name = arm_spec["name"]
            arm_record = manifest["arms"][arm_name]
            torch.distributed.barrier()
            model = None
            dtype = None
            use_amp = None
            load_record = None
            load_error = None
            try:
                (
                    model,
                    dtype,
                    use_amp,
                    load_record,
                ) = _instantiate_and_load_model(
                    config_path=Path(
                        arm_record["resolved_config"]["path"]
                    ),
                    snapshot_path=Path(arm_record["snapshot"]["path"]),
                    arm=arm_spec,
                    expected_run_identity=arm_record["arm_manifest"][
                        "identity_sha256"
                    ],
                    device=device,
                )
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"
            load_errors: list[Any] = [None] * WORLD_SIZE
            torch.distributed.all_gather_object(load_errors, load_error)
            if any(error is not None for error in load_errors):
                raise EvaluationContractError(
                    f"{arm_name} strict load failed by rank: {load_errors}"
                )
            if (
                model is None
                or dtype is None
                or use_amp is None
                or load_record is None
            ):
                raise EvaluationContractError(
                    f"{arm_name} load returned an incomplete model contract"
                )
            for batch_index, batch in enumerate(batches):
                current_batch_hash = _batch_sha256(batch)
                expected_batch_hash = initial_batch_summaries[batch_index][
                    "sha256"
                ]
                if current_batch_hash != expected_batch_hash:
                    raise EvaluationContractError(
                        f"cached CPU batch mutated before {arm_name}/"
                        f"batch {batch_index}"
                    )
                dataset = _pair_dataset_name(base_dataset, batch_index)
                model_seed = _model_noise_seed(batch_index)
                effective_seed = _effective_generator_seed(rank, batch_index)
                model.evaluation_noise_seed = model_seed
                device_batch = _clone_batch_to_device(
                    batch,
                    device=device,
                    dtype=dtype,
                    use_amp=use_amp,
                )
                # This measures one complete visualize call: all five condition
                # sources and every configured NFE. It is aggregate evaluation
                # cost, not single-video generation latency.
                torch.distributed.barrier()
                torch.cuda.synchronize(device)
                sampling_started = time.perf_counter()
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=dtype, enabled=use_amp
                ):
                    visualization = model.visualize(**device_batch)
                torch.cuda.synchronize(device)
                local_sampling_seconds = time.perf_counter() - sampling_started
                rank_sampling_seconds: list[Any] = [None] * WORLD_SIZE
                torch.distributed.all_gather_object(
                    rank_sampling_seconds, float(local_sampling_seconds)
                )
                aggregate_sampling_timing = {
                    "definition": (
                        "synchronized wall-clock for one visualize call "
                        "containing all 5 condition sources and NFE 1/2/4/8"
                    ),
                    "interpretation": (
                        "aggregate evaluation cost; not per-sample generation "
                        "latency and not evidence of generation speedup"
                    ),
                    "seconds_by_global_rank": [
                        float(value) for value in rank_sampling_seconds
                    ],
                    "min_seconds": float(min(rank_sampling_seconds)),
                    "mean_seconds": float(
                        sum(rank_sampling_seconds) / WORLD_SIZE
                    ),
                    "max_seconds": float(max(rank_sampling_seconds)),
                }
                if (
                    not isinstance(visualization, torch.Tensor)
                    or not bool(torch.isfinite(visualization).all())
                    or float(visualization.min().item()) < 0.0
                    or float(visualization.max().item()) > 1.0
                ):
                    raise EvaluationContractError(
                        f"{arm_name}/batch {batch_index} visualization is invalid"
                    )
                artifacts = model.pop_visualization_artifacts()
                if not artifacts:
                    raise EvaluationContractError(
                        f"{arm_name}/batch {batch_index} emitted no artifacts"
                    )
                metrics, paired_identity = _validate_and_measure_artifacts(
                    artifacts,
                    arm=arm_spec,
                    expected_model_seed=model_seed,
                )
                previous_identity = paired_identity_reference.get(batch_index)
                if previous_identity is None:
                    paired_identity_reference[batch_index] = paired_identity
                elif previous_identity != paired_identity:
                    raise EvaluationContractError(
                        f"cross-arm initial/clean identity differs for rank "
                        f"{rank}, batch {batch_index}"
                    )
                local_output_records.append(
                    _save_output_artifact(
                        evaluation_root=evaluation_root,
                        arm=arm_spec,
                        dataset=dataset,
                        rank=rank,
                        batch_index=batch_index,
                        batch_sha256=expected_batch_hash,
                        model_seed=model_seed,
                        effective_seed=effective_seed,
                        artifacts=artifacts,
                        metrics=metrics,
                        paired_identity=paired_identity,
                        aggregate_sampling_timing=aggregate_sampling_timing,
                        snapshot={
                            **arm_record["snapshot"],
                            **load_record,
                        },
                        resolved_config=arm_record["resolved_config"],
                        evaluation_commit=expected_commit,
                    )
                )
                del visualization
                del artifacts
                del device_batch
                if _batch_sha256(batch) != expected_batch_hash:
                    raise EvaluationContractError(
                        f"cached CPU batch mutated after {arm_name}/"
                        f"batch {batch_index}"
                    )
                torch.distributed.barrier()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise EvaluationContractError(
                    f"{arm_name} accumulated gradients during inference"
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.distributed.barrier()

        gathered_inputs: list[Any] = [None] * WORLD_SIZE
        gathered_outputs: list[Any] = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(gathered_inputs, local_input_records)
        torch.distributed.all_gather_object(gathered_outputs, local_output_records)
        flat_inputs = [
            record for rank_records in gathered_inputs for record in rank_records
        ]
        flat_outputs = [
            record for rank_records in gathered_outputs for record in rank_records
        ]

        def validate_for_completion() -> dict[str, Any]:
            _assert_clean_commit(repo_root, expected_commit)
            training_repository = _validate_training_repository(
                manifest["training_repository"]["screen_recorded_path"],
                evaluation_repo_root=repo_root,
                training_commit=manifest["training_commit"],
            )
            if training_repository != manifest["training_repository"]:
                raise EvaluationContractError(
                    "training repository changed during evaluation"
                )
            inventory = _validate_final_inventory(
                manifest=manifest,
                input_records=flat_inputs,
                output_records=flat_outputs,
            )
            for arm in manifest["source_arm_inventory"].values():
                for label in (
                    "arm_manifest",
                    "resolved_config",
                    "training_completion",
                    "outcome",
                    "snapshot",
                ):
                    if not _record_matches_file(arm[label]):
                        raise EvaluationContractError(
                            f"immutable source changed during evaluation: "
                            f"{arm['source_directory']} {label}"
                        )
            if not _record_matches_file(manifest["source_screen"]):
                raise EvaluationContractError(
                    "screen manifest changed during evaluation"
                )
            for label in ("null_prompt", "scheduler_config"):
                if not _record_matches_file(manifest["assets"][label]):
                    raise EvaluationContractError(
                        f"immutable evaluation asset changed: {label}"
                    )
            if (
                _git(
                    Path(manifest["assets"]["videox_home"]),
                    "rev-parse",
                    "HEAD",
                ).stdout.strip()
                != manifest["assets"]["videox_commit"]
            ):
                raise EvaluationContractError(
                    "VideoX-Fun commit changed during evaluation"
                )
            abc_manifest_path = _canonical_file(
                manifest["data"]["abc_manifest"]["path"],
                "ABC manifest",
            )
            if _sha256(abc_manifest_path) != manifest["data"]["abc_manifest"][
                "sha256"
            ]:
                raise EvaluationContractError(
                    "ABC input manifest changed during evaluation"
                )
            return inventory

        inventory = _broadcast_rank0(validate_for_completion, rank=rank)
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        local_total_seconds = time.perf_counter() - total_evaluation_started
        total_seconds_by_rank: list[Any] = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(
            total_seconds_by_rank, float(local_total_seconds)
        )

        def write_completion() -> dict[str, Any]:
            completion = _identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": (
                        "dual_abc_tf_causal_screen_posthoc_evaluation_complete"
                    ),
                    "completed_at_utc": _now(),
                    "evaluation_manifest": {
                        "path": str(manifest_path),
                        "sha256": _sha256(manifest_path),
                        "identity_sha256": manifest["identity_sha256"],
                    },
                    "training_commit": manifest["training_commit"],
                    "evaluation_commit": expected_commit,
                    "active_job_coexistence": manifest[
                        "active_job_coexistence"
                    ],
                    "source_inputs_unchanged": True,
                    "wandb_enabled": False,
                    "resume": False,
                    "requeue": False,
                    "paired_evaluation": manifest["paired_evaluation"],
                    "inventory": inventory,
                    "timing": {
                        "definition": (
                            "end-to-end synchronized posthoc evaluation wall "
                            "clock through source/output validation, excluding "
                            "only completion-manifest publication"
                        ),
                        "interpretation": (
                            "aggregate evaluation cost; not per-video generation "
                            "latency and not evidence of generation speedup"
                        ),
                        "seconds_by_global_rank": [
                            float(value) for value in total_seconds_by_rank
                        ],
                        "min_seconds": float(min(total_seconds_by_rank)),
                        "mean_seconds": float(
                            sum(total_seconds_by_rank) / WORLD_SIZE
                        ),
                        "max_seconds": float(max(total_seconds_by_rank)),
                    },
                }
            )
            _exclusive_json(
                evaluation_root / "evaluation_complete.json", completion
            )
            return completion

        completion = _broadcast_rank0(write_completion, rank=rank)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "evaluation_root": str(evaluation_root),
                        "paired_unit_count": completion["inventory"][
                            "paired_unit_count"
                        ],
                        "artifact_count": completion["inventory"][
                            "artifact_count"
                        ],
                        "identity_sha256": completion["identity_sha256"],
                    },
                    sort_keys=True,
                )
            )
        torch.distributed.barrier()
        return 0
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _add_manifest_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--expected-evaluation-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batches-per-rank", type=int, default=1)
    parser.add_argument(
        "--allow-active-job-id",
        action="append",
        default=[],
        metavar="NUMERIC_ID",
        help=(
            "repeatable exact active Slurm job ID observed and approved by "
            "the launcher; IDs must be unique and numerically sorted"
        ),
    )
    parser.add_argument(
        "--arms",
        default=",".join(arm["name"] for arm in causal.ARMS),
        help=(
            "comma-separated arm subset in causal-screen order; at least two. "
            "All condition sources and NFE 1/2/4/8 remain mandatory."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="read-only validation and dry-run summary"
    )
    _add_manifest_build_arguments(validate)
    validate.set_defaults(func=command_validate)

    prepare = subparsers.add_parser(
        "prepare", help="validate and exclusively create an evaluation manifest"
    )
    _add_manifest_build_arguments(prepare)
    prepare.set_defaults(func=command_prepare)

    submission = subparsers.add_parser(
        "record-submission", help="record an accepted non-requeueable Slurm job"
    )
    submission.add_argument("--evaluation-root", required=True)
    submission.add_argument("--evaluation-manifest", required=True)
    submission.add_argument("--job-id", required=True)
    submission.set_defaults(func=command_record_submission)

    run = subparsers.add_parser(
        "run", help="execute the eight-rank paired post-training evaluation"
    )
    run.add_argument("--expected-evaluation-commit", required=True)
    run.add_argument("--repo-root", required=True)
    run.add_argument("--evaluation-root", required=True)
    run.add_argument("--evaluation-manifest", required=True)
    run.set_defaults(func=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
