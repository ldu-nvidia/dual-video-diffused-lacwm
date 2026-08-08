#!/usr/bin/env python3
"""Train-only eligibility probe for privileged on-policy video distillation.

This tool does *not* train a student.  It asks a narrower prerequisite
question on a frozen dual-video checkpoint:

* roll the feature-free (``off``) policy from its own Gaussian initial state;
* at each video-active state visited by that policy, query the same checkpoint
  with no future feature, the aligned clean V-JEPA feature, and an
  episode-disjoint shuffled clean V-JEPA feature;
* compare all three video velocities with the ordinary rectified-flow target;
* separately compare complete NFE-4 ``off``/aligned/shuffled rollouts.

Only the immutable ABC *training* cache is accepted.  Clean future V-JEPA is
teacher-only evidence, never a deployable result.  The output is an
identity-hashed, append-free artifact suitable for an independent audit.

LACWM convention: sigma=1 is Gaussian noise and sigma=0 is clean data.
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
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "privileged-on-policy-teacher-eligibility-v1"
ROW_SCHEMA = "privileged-on-policy-teacher-eligibility-unit-v1"
ROLLOUT_ROW_SCHEMA = "privileged-on-policy-teacher-rollout-v1"
ANALYSIS_SCHEMA = "privileged-on-policy-teacher-analysis-v1"
COMPLETE_SCHEMA = "privileged-on-policy-teacher-complete-v1"
FAILURE_SCHEMA = "privileged-on-policy-teacher-failure-v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCES = ("off", "oracle_matched", "oracle_shuffled")
SOURCE_INFIX = {
    "off": "_off",
    "oracle_matched": "_oracle_matched",
    "oracle_shuffled": "_oracle_shuffled",
}
METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)


class ProbeError(RuntimeError):
    """Raised when the probe contract would be violated."""


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


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == recorded


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = canonical_file(path, "artifact")
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ProbeError(f"SHA-256 differs for {path}: {observed} != {expected_sha256}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ProbeError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise ProbeError(f"{label} must be a non-empty, non-symlink file: {path}")
    return path.resolve(strict=True)


def canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ProbeError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ProbeError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"JSON root must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ProbeError(f"blank JSONL row {line_number}: {path}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ProbeError(f"non-object JSONL row {line_number}: {path}")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"invalid JSONL: {path}") from exc
    return rows


def exclusive_bytes(path: Path, content: bytes) -> None:
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


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    exclusive_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = b"".join(_canonical_json(row) + b"\n" for row in rows)
    exclusive_bytes(path, content)


def git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise ProbeError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _safe_tensor_sha256(tensor: Any) -> str:
    """Hash tensor bytes, including bfloat16 tensors unsupported by NumPy."""
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _bootstrap_relative_improvement(
    control: Any,
    candidate: Any,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Paired clip-bootstrap ratio; arrays are [clips, repeated states]."""
    import numpy as np

    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if control.shape != candidate.shape or control.ndim not in (1, 2):
        raise ProbeError("paired metric arrays must have identical [N] or [N,K] shape")
    if (
        control.shape[0] < 2
        or not np.isfinite(control).all()
        or not np.isfinite(candidate).all()
    ):
        raise ProbeError("paired metric arrays are too small or non-finite")
    if float(control.mean()) <= 0:
        raise ProbeError("relative-improvement denominator is non-positive")
    point = (
        100.0
        * (float(control.mean()) - float(candidate.mean()))
        / float(control.mean())
    )
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, control.shape[0], size=(replicates, control.shape[0]))
    control_samples = control[indexes].mean(axis=tuple(range(1, control[indexes].ndim)))
    candidate_samples = candidate[indexes].mean(
        axis=tuple(range(1, candidate[indexes].ndim))
    )
    valid = control_samples > 0
    effects = (
        100.0
        * (control_samples[valid] - candidate_samples[valid])
        / control_samples[valid]
    )
    if effects.size != replicates:
        raise ProbeError("bootstrap produced a non-positive denominator")
    low, high = np.quantile(effects, (0.025, 0.975))
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": point,
        "paired_bootstrap_95_percent": [float(low), float(high)],
        "bootstrap_replicates": int(replicates),
        "bootstrap_unit": "clip (all visited timesteps remain clustered)",
    }


def analyze_rows(
    unit_rows: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    *,
    num_clips: int,
    active_steps: int,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    import numpy as np

    expected_units = num_clips * active_steps
    if len(unit_rows) != expected_units:
        raise ProbeError(
            f"unit-row count differs: {len(unit_rows)} != {expected_units}"
        )
    unit_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in unit_rows:
        if not identity_valid(row) or row.get("schema") != ROW_SCHEMA:
            raise ProbeError("invalid unit-row identity/schema")
        if (
            row.get("state_is_student_visited") is not True
            or row.get("video_phase_active") is not True
            or row.get("clean_future_feature_teacher_only") is not True
            or row.get("aligned_feature_differs_from_shuffled") is not True
            or row.get("protected_test_opened") is not False
            or row.get("episode_dir") == row.get("shuffled_donor_episode_dir")
        ):
            raise ProbeError(
                "unit row violates visited-state/sample-specific/train-only contract"
            )
        key = (int(row["clip_index"]), int(row["rollout_step"]))
        if key in unit_by_key:
            raise ProbeError(f"duplicate unit row: {key}")
        unit_by_key[key] = row
    steps = sorted({key[1] for key in unit_by_key})
    if len(steps) != active_steps:
        raise ProbeError("visited-state step inventory differs")
    student = np.empty((num_clips, active_steps), dtype=np.float64)
    aligned = np.empty_like(student)
    shuffled = np.empty_like(student)
    for clip in range(num_clips):
        for position, step in enumerate(steps):
            row = unit_by_key.get((clip, step))
            if row is None:
                raise ProbeError(f"missing unit row {(clip, step)}")
            student[clip, position] = float(row["velocity_mse"]["student_off"])
            aligned[clip, position] = float(row["velocity_mse"]["teacher_aligned"])
            shuffled[clip, position] = float(row["velocity_mse"]["teacher_shuffled"])

    rollout_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rollout_rows:
        if not identity_valid(row) or row.get("schema") != ROLLOUT_ROW_SCHEMA:
            raise ProbeError("invalid rollout-row identity/schema")
        if (
            row.get("protected_test_opened") is not False
            or row.get("deployable_evidence") is not False
            or bool(row.get("oracle_leakage")) != (row.get("source") != "off")
        ):
            raise ProbeError("rollout row violates oracle/protected-data contract")
        key = (int(row["clip_index"]), str(row["source"]))
        if key in rollout_by_key:
            raise ProbeError(f"duplicate rollout row: {key}")
        rollout_by_key[key] = row
    expected_rollouts = num_clips * len(SOURCES)
    if len(rollout_by_key) != expected_rollouts:
        raise ProbeError("rollout-row inventory differs")

    velocity_aligned_vs_student = _bootstrap_relative_improvement(
        student, aligned, seed=seed + 1, replicates=replicates
    )
    velocity_aligned_vs_shuffled = _bootstrap_relative_improvement(
        shuffled, aligned, seed=seed + 2, replicates=replicates
    )
    lower_student = velocity_aligned_vs_student["paired_bootstrap_95_percent"][0]
    lower_shuffled = velocity_aligned_vs_shuffled["paired_bootstrap_95_percent"][0]
    favorable_fraction = float(np.mean(aligned < student))

    rollout_effects: dict[str, Any] = {}
    for metric_position, metric in enumerate(METRICS):
        source_values: dict[str, np.ndarray] = {}
        for source in SOURCES:
            source_values[source] = np.asarray(
                [
                    float(rollout_by_key[(clip, source)]["metrics"][metric])
                    for clip in range(num_clips)
                ],
                dtype=np.float64,
            )
        rollout_effects[metric] = {
            "aligned_vs_off": _bootstrap_relative_improvement(
                source_values["off"],
                source_values["oracle_matched"],
                seed=seed + 10 + metric_position,
                replicates=replicates,
            ),
            "aligned_vs_shuffled": _bootstrap_relative_improvement(
                source_values["oracle_shuffled"],
                source_values["oracle_matched"],
                seed=seed + 20 + metric_position,
                replicates=replicates,
            ),
        }

    gates = {
        "aligned_teacher_better_unit_fraction_at_least_0.60": favorable_fraction
        >= 0.60,
        "aligned_teacher_velocity_gain_at_least_5_percent": (
            velocity_aligned_vs_student["relative_improvement_percent"] >= 5.0
            and lower_student > 0.0
        ),
        "aligned_beats_shuffled_velocity_by_at_least_3_percent": (
            velocity_aligned_vs_shuffled["relative_improvement_percent"] >= 3.0
            and lower_shuffled > 0.0
        ),
        "aligned_full_rollout_improves_decoded_mse": (
            rollout_effects["decoded_mse_unit_range"]["aligned_vs_off"][
                "relative_improvement_percent"
            ]
            > 0.0
        ),
        "aligned_full_rollout_improves_temporal_mse": (
            rollout_effects["decoded_temporal_difference_mse_unit_range"][
                "aligned_vs_off"
            ]["relative_improvement_percent"]
            > 0.0
        ),
    }
    all_passed = all(gates.values())
    return identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "created_at_utc": _now(),
            "clip_count": num_clips,
            "visited_state_steps": steps,
            "visited_state_unit_count": expected_units,
            "teacher_better_unit_fraction": favorable_fraction,
            "velocity_effects": {
                "aligned_vs_student_off": velocity_aligned_vs_student,
                "aligned_vs_episode_shuffled": velocity_aligned_vs_shuffled,
            },
            "rollout_effects": rollout_effects,
            "eligibility_gates": {**gates, "all_passed": all_passed},
            "decision": (
                "ELIGIBLE_FOR_STUDENT_SCREEN"
                if all_passed
                else "STOP_NO_ELIGIBLE_TEACHER"
            ),
            "claim_boundary": (
                "train-only same-checkpoint teacher eligibility; no student was trained, "
                "no deployable quality gain was tested, and no protected split was opened"
            ),
        }
    )


def _manifest_selection(manifest: Path, num_clips: int) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest)
    if len(rows) < num_clips:
        raise ProbeError(f"train manifest has only {len(rows)} < {num_clips} rows")
    for index, row in enumerate(rows):
        if row.get("split") != "train":
            raise ProbeError(f"manifest row {index} is not train-scoped")
        if int(row.get("auxiliary_index", -1)) != index:
            raise ProbeError(f"manifest row {index} has a noncanonical auxiliary index")
    selected = rows[:num_clips]
    episodes: set[str] = set()
    clip_ids: set[str] = set()
    for index, row in enumerate(selected):
        if row.get("split") != "train" or int(row.get("auxiliary_index", -1)) != index:
            raise ProbeError(f"selected row {index} is not the pinned train item")
        episode = row.get("episode_dir")
        clip_id = row.get("clip_id")
        if not isinstance(episode, str) or not episode or not isinstance(clip_id, str):
            raise ProbeError(f"selected row {index} lacks episode/clip identity")
        if episode in episodes or clip_id in clip_ids:
            raise ProbeError(
                "selected calibration clips must be episode- and clip-disjoint"
            )
        episodes.add(episode)
        clip_ids.add(clip_id)
    return selected


def _resolved_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProbeError(f"{label} path is absent")
    try:
        return Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProbeError(f"{label} path is invalid: {value!r}") from exc


def _validate_registered_input_contract(
    *,
    args: argparse.Namespace,
    resolved_config: Path,
    snapshot: Path,
    train_manifest: Path,
    cache_metadata: Path,
    arm_manifest: Path,
    stage_manifest: Path,
    stage_outcome: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-link immutable training provenance instead of trusting filenames."""
    metadata = read_json(cache_metadata)
    if (
        metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache"
        or metadata.get("split") != "train"
        or metadata.get("complete") is not True
        or int(metadata.get("clip_count", -1)) != 512
        or int(metadata.get("written_rows", -1)) != 512
    ):
        raise ProbeError(
            "cache metadata is not the complete pinned 512-clip train cache"
        )
    manifest_sha = records["train_manifest"]["sha256"]
    for field in ("clip_manifest_sha256", "train_manifest_sha256"):
        if metadata.get(field) != manifest_sha:
            raise ProbeError(f"cache metadata {field} does not bind the train manifest")
    for field in ("clip_manifest", "train_manifest"):
        if (
            _resolved_path(metadata.get(field), f"cache metadata {field}")
            != train_manifest
        ):
            raise ProbeError(
                f"cache metadata {field} resolves outside the registered manifest"
            )
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ProbeError("cache metadata lacks provenance")
    if (
        provenance.get("clip_manifest_sha256") != manifest_sha
        or provenance.get("train_manifest_sha256") != manifest_sha
        or provenance.get("pca_stats_sha256") != metadata.get("pca_sha256")
    ):
        raise ProbeError("cache provenance does not bind its train manifest/PCA")
    expected_shapes = {
        "rgb_shape": [512, 13, 3, 180, 960],
        "actions_shape": [512, 13, 5, 23],
        "target_shape": [512, 64, 4, 24, 120],
    }
    for field, expected in expected_shapes.items():
        if metadata.get(field) != expected:
            raise ProbeError(f"cache metadata {field} differs: {metadata.get(field)!r}")

    arm = read_json(arm_manifest)
    stage = read_json(stage_manifest)
    outcome = read_json(stage_outcome)
    for label, payload in (("arm", arm), ("stage", stage), ("stage outcome", outcome)):
        if not identity_valid(payload):
            raise ProbeError(f"{label} manifest identity is invalid")
    arm_identity = arm["identity_sha256"]
    arm_spec = arm.get("arm")
    if (
        arm.get("git_commit") != args.training_source_commit
        or not isinstance(arm_spec, Mapping)
        or arm_spec.get("code") != "J1"
        or arm_spec.get("schedule_mode") != "tf_first_cascaded"
        or arm_spec.get("condition_mode") != "matched"
    ):
        raise ProbeError(
            "arm manifest is not the pinned faithful-cascade J1 training run"
        )
    if (
        stage.get("arm_identity_sha256") != arm_identity
        or stage.get("arm_code") != "J1"
        or int(stage.get("stage_endpoint_completed_updates", -1)) != 1000
        or _resolved_path(stage.get("snapshot"), "stage snapshot") != snapshot
    ):
        raise ProbeError("stage manifest does not bind J1 update 1000 and its snapshot")
    stage_config = stage.get("resolved_config")
    if (
        not isinstance(stage_config, Mapping)
        or _resolved_path(stage_config.get("path"), "stage resolved config")
        != resolved_config
        or stage_config.get("sha256") != records["resolved_config"]["sha256"]
        or int(stage_config.get("bytes", -1)) != records["resolved_config"]["bytes"]
    ):
        raise ProbeError("stage manifest does not bind the registered resolved config")
    snapshot_record = outcome.get("snapshot_observed_at_stage_end")
    if (
        outcome.get("stage_identity_sha256") != stage["identity_sha256"]
        or outcome.get("arm_identity_sha256") != arm_identity
        or outcome.get("arm_code") != "J1"
        or int(outcome.get("completed_updates", -1)) != 1000
        or not isinstance(snapshot_record, Mapping)
        or _resolved_path(snapshot_record.get("path"), "outcome snapshot") != snapshot
        or snapshot_record.get("sha256") != records["snapshot"]["sha256"]
        or int(snapshot_record.get("bytes", -1)) != records["snapshot"]["bytes"]
    ):
        raise ProbeError(
            "stage outcome does not bind the registered update-1000 snapshot"
        )
    return {
        "cache_id": metadata.get("cache_id"),
        "cache_split": metadata.get("split"),
        "cache_manifest_sha256": manifest_sha,
        "cache_target_sha256": metadata.get("target_sha256"),
        "cache_vjepa_checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "arm_identity_sha256": arm_identity,
        "stage_identity_sha256": stage["identity_sha256"],
        "stage_outcome_identity_sha256": outcome["identity_sha256"],
    }


def _assert_train_only_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    if "train" not in path.name.lower() and "/train/" not in lowered:
        raise ProbeError(f"{label} is not explicitly train-scoped: {path}")
    if any(
        marker in lowered
        for marker in ("lockbox", "/test/", "/val/", "test.jsonl", "val.jsonl")
    ):
        raise ProbeError(f"{label} contains a forbidden non-train marker: {path}")


def _prepare_registration(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    if COMMIT_RE.fullmatch(args.expected_source_commit) is None:
        raise ProbeError("--expected-source-commit is invalid")
    if COMMIT_RE.fullmatch(args.training_source_commit) is None:
        raise ProbeError("--training-source-commit is invalid")
    if SHA256_RE.fullmatch(args.snapshot_sha256) is None:
        raise ProbeError("--snapshot-sha256 is invalid")
    if args.num_clips < 8 or args.num_clips % args.batch_size:
        raise ProbeError("num-clips must be >=8 and divisible by batch-size")
    if args.batch_size < 2:
        raise ProbeError("batch-size must be >=2 for episode-shuffled controls")
    if args.nfe != 4:
        raise ProbeError("the v1 probe is frozen to NFE=4")
    if args.bootstrap_replicates < 1000:
        raise ProbeError("bootstrap-replicates must be at least 1000")

    repo = canonical_directory(args.repo_root, "repository")
    actual_commit = git_output(repo, "rev-parse", "HEAD")
    if actual_commit != args.expected_source_commit:
        raise ProbeError(
            f"source commit differs: {actual_commit} != {args.expected_source_commit}"
        )
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProbeError(
            "source repository must be clean: " + status.replace("\n", "; ")
        )

    resolved_config = canonical_file(args.resolved_config, "resolved config")
    snapshot = canonical_file(args.snapshot, "snapshot")
    train_manifest = canonical_file(args.train_manifest, "train manifest")
    cache_metadata = canonical_file(args.cache_metadata, "train cache metadata")
    arm_manifest = canonical_file(args.arm_manifest, "arm manifest")
    stage_manifest = canonical_file(args.stage_manifest, "stage manifest")
    stage_outcome = canonical_file(args.stage_outcome, "stage outcome")
    _assert_train_only_path(train_manifest, "train manifest")
    _assert_train_only_path(cache_metadata, "train cache metadata")
    selected = _manifest_selection(train_manifest, args.num_clips)

    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise ProbeError(f"fresh output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)
    output = output.resolve(strict=True)

    records = {
        "resolved_config": file_record(resolved_config),
        "snapshot": file_record(snapshot, args.snapshot_sha256),
        "train_manifest": file_record(train_manifest),
        "train_cache_metadata": file_record(cache_metadata),
        "arm_manifest": file_record(arm_manifest),
        "stage_manifest": file_record(stage_manifest),
        "stage_outcome": file_record(stage_outcome),
    }
    provenance_links = _validate_registered_input_contract(
        args=args,
        resolved_config=resolved_config,
        snapshot=snapshot,
        train_manifest=train_manifest,
        cache_metadata=cache_metadata,
        arm_manifest=arm_manifest,
        stage_manifest=stage_manifest,
        stage_outcome=stage_outcome,
        records=records,
    )
    registration = identity_payload(
        {
            "schema": SCHEMA,
            "status": "registered_before_model_weights_or_cached_tensor_arrays_open",
            "created_at_utc": _now(),
            "repo_root": str(repo),
            "source_commit": actual_commit,
            "training_source_commit": args.training_source_commit,
            "teacher_student_contract": {
                "student_rollin": (
                    "same checkpoint and production off semantics: causal generated-feature "
                    "prefix, then V-JEPA state/clock hard-off for video-active calls"
                ),
                "teacher_aligned": "same checkpoint, scheduled clean target-video V-JEPA condition",
                "teacher_shuffled": "same checkpoint, scheduled next-item V-JEPA from a different episode",
                "query_state": "identical causal-student-visited video and auxiliary state",
                "flow_target": "fixed initial video noise minus clean video latent",
                "scored_states": "NFE-4 video-active states only; pure auxiliary-prefix calls excluded",
            },
            "data_contract": {
                "split": "train",
                "clip_count": args.num_clips,
                "clip_indices": list(range(args.num_clips)),
                "all_selected_episodes_unique": True,
                "protected_test_opened": False,
                "validation_opened": False,
            },
            "frozen_parameters": {
                "nfe": args.nfe,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "bootstrap_replicates": args.bootstrap_replicates,
                "eligibility": {
                    "teacher_better_unit_fraction_min": 0.60,
                    "aligned_vs_student_velocity_gain_min_percent": 5.0,
                    "aligned_vs_shuffled_velocity_gain_min_percent": 3.0,
                    "paired_lower_bound_strictly_positive": True,
                    "aligned_rollout_decoded_and_temporal_point_gain_strictly_positive": True,
                },
            },
            "selected_clips": [
                {
                    "clip_index": index,
                    "clip_id": row["clip_id"],
                    "episode_dir": row["episode_dir"],
                }
                for index, row in enumerate(selected)
            ],
            "inputs": records,
            "verified_provenance_links": provenance_links,
            "output_dir": str(output),
            "claim_boundary": (
                "teacher-eligibility probe only; teacher uses clean future V-JEPA, "
                "no student optimization or deployable improvement is claimed"
            ),
        }
    )
    exclusive_json(output / "registration.json", registration)
    return output, registration, selected


def _load_model_dataset(
    registration: Mapping[str, Any], device: Any
) -> tuple[Any, Any, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # The executing repository is carried in the registration source commit;
    # derive it from this source file so imports cannot silently come from a
    # different checkout.
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root / "projects" / "latent_action_models"
    for root in (str(repo_root), str(project_root)):
        if root not in sys.path:
            sys.path.insert(0, root)

    config = OmegaConf.load(registration["inputs"]["resolved_config"]["path"])
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    expected_train_manifest = registration["inputs"]["train_manifest"]["path"]
    expected_cache_metadata = registration["inputs"]["train_cache_metadata"]["path"]
    dataset_config = config.dataset
    abc_config = dataset_config.datasets.ABC
    if (
        str(Path(str(abc_config.clip_manifest)).resolve()) != expected_train_manifest
        or str(Path(str(abc_config.cache_metadata)).resolve())
        != expected_cache_metadata
    ):
        raise ProbeError("resolved dataset does not match registered train cache")
    if (
        str(config.model.get("_target_", "")).split(".")[-1]
        != "DualExplicitActionDiTModel"
    ):
        raise ProbeError("resolved model is not DualExplicitActionDiTModel")
    if str(config.model.dual_diffusion.schedule_mode) != "tf_first_cascaded":
        raise ProbeError("probe requires the faithful tf_first_cascaded J1 checkpoint")
    if (
        str(config.model.dual_diffusion.condition_mode) != "matched"
        or config.model.dual_diffusion.auxiliary_history_mode != "diffuse_all"
        or float(config.model.dual_diffusion.cascade_inference_tf_fraction) != 0.5
    ):
        raise ProbeError("resolved J1 inference/conditioning contract differs")
    if config.model.time_frequency_transform is not None:
        raise ProbeError("online auxiliary extractor must remain absent")

    model = instantiate(config.model)
    snapshot = torch.load(
        registration["inputs"]["snapshot"]["path"],
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(snapshot, Mapping) or "model" not in snapshot:
        raise ProbeError("snapshot lacks model state")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ProbeError(f"strict snapshot load failed: {incompatible}")
    del snapshot
    model = model.to(device=device).eval()
    dataset = instantiate(dataset_config)
    if len(dataset) != 512:
        raise ProbeError(f"pinned train dataset length differs: {len(dataset)} != 512")
    return model, dataset, config


def _move_samples(samples: Sequence[Mapping[str, Any]], device: Any) -> dict[str, Any]:
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
    missing = required - batch.keys()
    if missing:
        raise ProbeError(f"batch lacks {sorted(missing)}")
    return batch


def _per_sample_mse(prediction: Any, target: Any, history_frames: int) -> Any:
    return (
        (
            prediction[:, :, history_frames:].float()
            - target[:, :, history_frames:].float()
        )
        .square()
        .flatten(1)
        .mean(1)
    )


def _per_sample_nmse(prediction: Any, target: Any, history_frames: int) -> Any:
    numerator = (
        (
            prediction[:, :, history_frames:].float()
            - target[:, :, history_frames:].float()
        )
        .square()
        .flatten(1)
        .sum(1)
    )
    denominator = target[:, :, history_frames:].float().square().flatten(1).sum(1)
    if bool((denominator <= 0).any()):
        raise ProbeError("latent NMSE target energy is non-positive")
    return numerator / denominator


def _decoded_metrics(prediction: Any, target: Any) -> tuple[Any, Any]:
    prediction = prediction.float() / 255.0
    target = target.float() / 255.0
    mse = (prediction - target).square().flatten(1).mean(1)
    temporal = (prediction.diff(dim=2) - target.diff(dim=2)).square().flatten(1).mean(1)
    return mse, temporal


def _run_batch(
    model: Any,
    batch: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    nfe: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    import torch
    from robot_wm.modeling.dual_diffusion.conditioning import (
        make_oracle_conditioning_tf,
    )
    from robot_wm.modeling.dual_diffusion.flow import euler_flow_step
    from robot_wm.modeling.networks.wan_forward_model import DualWanOutput

    rgb = batch["rgb"]
    actions = batch["actions"]
    clip_indexes = [int(value) for value in batch["clip_index"].detach().cpu().tolist()]
    expected_indexes = [int(item["clip_index"]) for item in selected]
    if clip_indexes != expected_indexes:
        raise ProbeError(
            f"dataset substituted clips: {clip_indexes} != {expected_indexes}"
        )
    episodes = [str(item["episode_dir"]) for item in selected]
    if len(set(episodes)) != len(episodes) or any(
        episodes[index] == episodes[(index + 1) % len(episodes)]
        for index in range(len(episodes))
    ):
        raise ProbeError("batch shuffle donors are not episode-disjoint")

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        video_clean = model._encode_clip(rgb).to(rgb.dtype)
        reference, history_frames = model._history_reference(rgb, video_clean.shape)
        auxiliary_history_frames = model._auxiliary_history_frames(history_frames)
        if auxiliary_history_frames != 0:
            raise ProbeError(
                "offline V-JEPA probe requires diffuse-all auxiliary history"
            )
        _, z_control, _ = model._latent_actions(
            rgb,
            actions,
            batch["morphology_index"],
            video_clean.shape[2],
            history_frames,
        )
        z_control = z_control.to(rgb.dtype)
        context = model._build_context(len(selected), rgb.device, rgb.dtype)
        clip_fea = model._build_clip(len(selected), rgb.device, rgb.dtype)
        tf_clean = model._validate_auxiliary_clean(
            batch["auxiliary_target"], video_clean.shape
        ).to(rgb.dtype)
        wrong_tf_clean = torch.roll(tf_clean, shifts=-1, dims=0)
        initial_video = model._evaluation_noise(
            video_clean.shape,
            device=rgb.device,
            dtype=rgb.dtype,
            base_seed=model.evaluation_noise_seed,
            sample_ids=batch["clip_index"],
            stream=0,
            rank=0,
        )
        initial_tf = model._evaluation_noise(
            tf_clean.shape,
            device=rgb.device,
            dtype=rgb.dtype,
            base_seed=model.evaluation_noise_seed,
            sample_ids=batch["clip_index"],
            stream=1,
            rank=0,
        )
        video_target = initial_video - video_clean
        video_state = initial_video.clone()
        tf_state = initial_tf.clone()
        schedule, timesteps, tf_only_steps = model._sampling_schedule(
            nfe, device=rgb.device
        )
        if tf_only_steps <= 0 or tf_only_steps >= nfe:
            raise ProbeError(
                "faithful cascade did not expose auxiliary and video phases"
            )
        active_steps = nfe - int(tf_only_steps)
        unit_rows: list[dict[str, Any]] = []
        wan_calls = 0

        for step, timestep in enumerate(timesteps):
            video_sigma = schedule.video[step]
            next_video_sigma = schedule.video[step + 1]
            tf_sigma = schedule.time_frequency[step]
            next_tf_sigma = schedule.time_frequency[step + 1]
            t_batch = timestep.expand(len(selected)).to(rgb.device)
            tf_batch_sigma = tf_sigma.expand(len(selected)).to(dtype=rgb.dtype)
            tf_sigma_expanded = model._expand_sigma(tf_batch_sigma, tf_state)

            # Preserve the checkpoint's registered ``off`` semantics.  During
            # the auxiliary-only prefix, ``off`` and ``autonomous`` share the
            # same causal generated-feature trajectory; the video state is not
            # updated.  The intervention becomes hard-off exactly when video
            # updates begin.  No clean target enters either phase.
            causal_prefix = step < tf_only_steps
            student_conditioning = model._sampling_conditioning_tf(
                tf_state,
                initial_tf,
                tf_sigma_expanded,
                history_frames=auxiliary_history_frames,
            )
            student_prediction = model.forward_model(
                video_state,
                t_batch,
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=tf_state,
                conditioning_tf=student_conditioning,
                tf_sigma=tf_batch_sigma,
                condition_on_tf=(model.condition_on_tf if causal_prefix else False),
                condition_on_tf_clock=(
                    model.condition_on_tf_clock if causal_prefix else False
                ),
            )
            wan_calls += 1
            if not isinstance(student_prediction, DualWanOutput):
                raise ProbeError("student forward did not return dual velocities")

            if step >= tf_only_steps:
                aligned_condition = make_oracle_conditioning_tf(
                    tf_clean=tf_clean,
                    tf_noise=initial_tf,
                    tf_sigma_expanded=tf_sigma_expanded,
                    history_frames=auxiliary_history_frames,
                )
                shuffled_condition = make_oracle_conditioning_tf(
                    tf_clean=tf_clean,
                    tf_noise=initial_tf,
                    tf_sigma_expanded=tf_sigma_expanded,
                    history_frames=auxiliary_history_frames,
                    wrong_tf_clean=wrong_tf_clean,
                )
                teacher_aligned = model.forward_model(
                    video_state,
                    t_batch,
                    z_control,
                    reference,
                    context,
                    clip_fea,
                    noisy_tf=tf_state,
                    conditioning_tf=aligned_condition,
                    tf_sigma=tf_batch_sigma,
                    condition_on_tf=True,
                    condition_on_tf_clock=True,
                )
                teacher_shuffled = model.forward_model(
                    video_state,
                    t_batch,
                    z_control,
                    reference,
                    context,
                    clip_fea,
                    noisy_tf=tf_state,
                    conditioning_tf=shuffled_condition,
                    tf_sigma=tf_batch_sigma,
                    condition_on_tf=True,
                    condition_on_tf_clock=True,
                )
                wan_calls += 2
                student_mse = _per_sample_mse(
                    student_prediction.video_velocity, video_target, history_frames
                )
                aligned_mse = _per_sample_mse(
                    teacher_aligned.video_velocity, video_target, history_frames
                )
                shuffled_mse = _per_sample_mse(
                    teacher_shuffled.video_velocity, video_target, history_frames
                )
                distance = _per_sample_mse(
                    teacher_aligned.video_velocity,
                    student_prediction.video_velocity,
                    history_frames,
                )
                for local_index, descriptor in enumerate(selected):
                    unit_rows.append(
                        identity_payload(
                            {
                                "schema": ROW_SCHEMA,
                                "clip_index": int(descriptor["clip_index"]),
                                "clip_id": descriptor["clip_id"],
                                "episode_dir": descriptor["episode_dir"],
                                "shuffled_donor_clip_index": int(
                                    selected[(local_index + 1) % len(selected)][
                                        "clip_index"
                                    ]
                                ),
                                "shuffled_donor_episode_dir": selected[
                                    (local_index + 1) % len(selected)
                                ]["episode_dir"],
                                "rollout_step": step,
                                "video_sigma": float(video_sigma),
                                "tf_sigma": float(tf_sigma),
                                "state_is_student_visited": True,
                                "video_phase_active": True,
                                "velocity_mse": {
                                    "student_off": float(student_mse[local_index]),
                                    "teacher_aligned": float(aligned_mse[local_index]),
                                    "teacher_shuffled": float(
                                        shuffled_mse[local_index]
                                    ),
                                    "teacher_student_distance": float(
                                        distance[local_index]
                                    ),
                                },
                                "tensor_sha256": {
                                    "rgb": _safe_tensor_sha256(
                                        rgb[local_index : local_index + 1]
                                    ),
                                    "actions": _safe_tensor_sha256(
                                        actions[local_index : local_index + 1]
                                    ),
                                    "video_clean": _safe_tensor_sha256(
                                        video_clean[local_index : local_index + 1]
                                    ),
                                    "video_state": _safe_tensor_sha256(
                                        video_state[local_index : local_index + 1]
                                    ),
                                    "video_target_velocity": _safe_tensor_sha256(
                                        video_target[local_index : local_index + 1]
                                    ),
                                    "student_velocity": _safe_tensor_sha256(
                                        student_prediction.video_velocity[
                                            local_index : local_index + 1
                                        ]
                                    ),
                                    "aligned_teacher_velocity": _safe_tensor_sha256(
                                        teacher_aligned.video_velocity[
                                            local_index : local_index + 1
                                        ]
                                    ),
                                    "shuffled_teacher_velocity": _safe_tensor_sha256(
                                        teacher_shuffled.video_velocity[
                                            local_index : local_index + 1
                                        ]
                                    ),
                                    "aligned_clean_feature": _safe_tensor_sha256(
                                        tf_clean[local_index : local_index + 1]
                                    ),
                                    "shuffled_clean_feature": _safe_tensor_sha256(
                                        wrong_tf_clean[local_index : local_index + 1]
                                    ),
                                },
                                "aligned_feature_differs_from_shuffled": (
                                    not torch.equal(
                                        tf_clean[local_index],
                                        wrong_tf_clean[local_index],
                                    )
                                ),
                                "clean_future_feature_teacher_only": True,
                                "protected_test_opened": False,
                            }
                        )
                    )

            if step >= tf_only_steps:
                video_state = model.sample_scheduler.step(
                    student_prediction.video_velocity.float(),
                    timestep,
                    video_state.float(),
                ).prev_sample.to(rgb.dtype)
            history_sigma = next_video_sigma.to(device=rgb.device, dtype=rgb.dtype)
            video_state[:, :, :history_frames] = (1.0 - history_sigma) * reference[
                :, :, :history_frames
            ] + history_sigma * initial_video[:, :, :history_frames]
            tf_state = euler_flow_step(
                tf_state.float(),
                student_prediction.tf_velocity.float(),
                tf_sigma,
                next_tf_sigma,
            ).to(rgb.dtype)

        model.evaluation_condition_sources = SOURCES
        model.evaluation_nfe_steps = (nfe,)
        model.viz_num_steps = nfe
        model.capture_latent_trajectories = False
        model.artifact_batch_limit = None
        model._sample_future(
            rgb,
            actions,
            morphology_index=batch["morphology_index"],
            auxiliary_target=batch["auxiliary_target"],
            collect_artifacts=True,
            deployment_mode=False,
            sample_ids=batch["clip_index"],
        )
        wan_calls += len(SOURCES) * nfe
        artifacts = model.pop_visualization_artifacts()
        if not isinstance(artifacts, Mapping):
            raise ProbeError("full rollout did not return artifacts")
        if int(artifacts["online_teacher_call_count"].item()) != 0:
            raise ProbeError("unexpected registered online teacher module call")
        canonical_target = artifacts["ground_truth_future_uint8"]
        canonical_video_clean = artifacts["video_clean"]
        manual_video_final = video_state.detach().cpu().to(torch.float16)
        manual_tf_final = tf_state.detach().cpu().to(torch.float16)
        if not torch.equal(manual_video_final, artifacts[f"video_final_off_nfe_{nfe}"]):
            raise ProbeError(
                "manual student roll-in differs from production off video rollout"
            )
        if not torch.equal(manual_tf_final, artifacts[f"tf_final_off_nfe_{nfe}"]):
            raise ProbeError(
                "manual student roll-in differs from production off auxiliary rollout"
            )
        rollout_rows: list[dict[str, Any]] = []
        for source in SOURCES:
            infix = SOURCE_INFIX[source]
            final = artifacts[f"video_final{infix}_nfe_{nfe}"]
            decoded = artifacts[f"decoded_future{infix}_nfe_{nfe}"]
            video_nmse = _per_sample_nmse(final, canonical_video_clean, history_frames)
            decoded_mse, temporal_mse = _decoded_metrics(decoded, canonical_target)
            for local_index, descriptor in enumerate(selected):
                rollout_rows.append(
                    identity_payload(
                        {
                            "schema": ROLLOUT_ROW_SCHEMA,
                            "clip_index": int(descriptor["clip_index"]),
                            "clip_id": descriptor["clip_id"],
                            "episode_dir": descriptor["episode_dir"],
                            "source": source,
                            "nfe": nfe,
                            "oracle_leakage": source != "off",
                            "deployable_evidence": False,
                            "metrics": {
                                "video_future_nmse": float(video_nmse[local_index]),
                                "decoded_mse_unit_range": float(
                                    decoded_mse[local_index]
                                ),
                                "decoded_temporal_difference_mse_unit_range": float(
                                    temporal_mse[local_index]
                                ),
                            },
                            "tensor_sha256": {
                                "video_final": _safe_tensor_sha256(
                                    final[local_index : local_index + 1]
                                ),
                                "decoded_final": _safe_tensor_sha256(
                                    decoded[local_index : local_index + 1]
                                ),
                                "vae_decoded_future_target": _safe_tensor_sha256(
                                    canonical_target[local_index : local_index + 1]
                                ),
                            },
                            "protected_test_opened": False,
                        }
                    )
                )
    return unit_rows, rollout_rows, wan_calls, active_steps


def command_run(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise ProbeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name.upper():
        raise ProbeError(f"probe requires a B200, found {properties.name}")
    output, registration, selected_rows = _prepare_registration(args)
    model, dataset, _config = _load_model_dataset(registration, device)

    unit_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    wan_calls = 0
    observed_active_steps: int | None = None
    for start in range(0, args.num_clips, args.batch_size):
        descriptors = [
            {"clip_index": index, **selected_rows[index]}
            for index in range(start, start + args.batch_size)
        ]
        samples = [dataset[index] for index in range(start, start + args.batch_size)]
        batch = _move_samples(samples, device)
        batch_units, batch_rollouts, batch_calls, batch_active_steps = _run_batch(
            model,
            batch,
            descriptors,
            nfe=args.nfe,
        )
        if observed_active_steps is None:
            observed_active_steps = batch_active_steps
        elif observed_active_steps != batch_active_steps:
            raise ProbeError(
                "active video-step count changed across calibration batches"
            )
        unit_rows.extend(batch_units)
        rollout_rows.extend(batch_rollouts)
        wan_calls += batch_calls
        print(
            f"completed clips {start + args.batch_size}/{args.num_clips}; "
            f"Wan calls={wan_calls}",
            flush=True,
        )

    if observed_active_steps is None:
        raise ProbeError("no calibration batches were evaluated")
    active_steps = observed_active_steps
    analysis = analyze_rows(
        unit_rows,
        rollout_rows,
        num_clips=args.num_clips,
        active_steps=active_steps,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    exclusive_jsonl(output / "per_unit_metrics.jsonl", unit_rows)
    exclusive_jsonl(output / "per_clip_rollout_metrics.jsonl", rollout_rows)
    exclusive_json(output / "analysis.json", analysis)
    complete = identity_payload(
        {
            "schema": COMPLETE_SCHEMA,
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": analysis["decision"],
            "unit_row_count": len(unit_rows),
            "rollout_row_count": len(rollout_rows),
            "observed_tf_only_steps": args.nfe - active_steps,
            "observed_video_active_steps": active_steps,
            "observed_wan_forward_calls": wan_calls,
            "expected_wan_forward_calls": (args.num_clips // args.batch_size)
            * ((args.nfe + 2 * active_steps) + len(SOURCES) * args.nfe),
            "artifacts": {
                "registration": file_record(output / "registration.json"),
                "per_unit_metrics": file_record(output / "per_unit_metrics.jsonl"),
                "per_clip_rollout_metrics": file_record(
                    output / "per_clip_rollout_metrics.jsonl"
                ),
                "analysis": file_record(output / "analysis.json"),
            },
            "protected_test_opened": False,
            "validation_opened": False,
            "student_training_launched": False,
        }
    )
    if complete["observed_wan_forward_calls"] != complete["expected_wan_forward_calls"]:
        raise ProbeError("Wan call accounting differs")
    exclusive_json(output / "run_complete.json", complete)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    output = canonical_directory(args.output_dir, "probe output")
    registration = read_json(
        canonical_file(output / "registration.json", "registration")
    )
    complete = read_json(canonical_file(output / "run_complete.json", "completion"))
    analysis = read_json(canonical_file(output / "analysis.json", "analysis"))
    unit_rows = read_jsonl(
        canonical_file(output / "per_unit_metrics.jsonl", "unit rows")
    )
    rollout_rows = read_jsonl(
        canonical_file(output / "per_clip_rollout_metrics.jsonl", "rollout rows")
    )
    for label, payload, schema in (
        ("registration", registration, SCHEMA),
        ("completion", complete, COMPLETE_SCHEMA),
        ("analysis", analysis, ANALYSIS_SCHEMA),
    ):
        if not identity_valid(payload) or payload.get("schema") != schema:
            raise ProbeError(f"{label} identity/schema is invalid")
    frozen = registration["frozen_parameters"]
    recomputed = analyze_rows(
        unit_rows,
        rollout_rows,
        num_clips=int(registration["data_contract"]["clip_count"]),
        active_steps=int(complete["observed_video_active_steps"]),
        seed=int(frozen["seed"]),
        replicates=int(frozen["bootstrap_replicates"]),
    )
    # Timestamp is intentionally not part of scientific equality.
    comparable_analysis = dict(analysis)
    comparable_recomputed = dict(recomputed)
    for payload in (comparable_analysis, comparable_recomputed):
        payload.pop("created_at_utc", None)
        payload.pop("identity_sha256", None)
    if comparable_analysis != comparable_recomputed:
        raise ProbeError("analysis does not reproduce from raw rows")
    if (
        int(complete["observed_tf_only_steps"])
        + int(complete["observed_video_active_steps"])
        != int(frozen["nfe"])
        or int(complete["observed_tf_only_steps"]) <= 0
        or int(complete["observed_video_active_steps"]) <= 0
    ):
        raise ProbeError("recorded cascade phase inventory is invalid")
    expected_outputs = {
        "registration": output / "registration.json",
        "per_unit_metrics": output / "per_unit_metrics.jsonl",
        "per_clip_rollout_metrics": output / "per_clip_rollout_metrics.jsonl",
        "analysis": output / "analysis.json",
    }
    for key, record in complete["artifacts"].items():
        if Path(record["path"]).resolve() != expected_outputs[key]:
            raise ProbeError(f"completion artifact escaped output directory: {key}")
        observed = file_record(Path(record["path"]), record["sha256"])
        if observed != record:
            raise ProbeError(f"completion artifact record differs: {key}")
    for key, record in registration["inputs"].items():
        observed = file_record(Path(record["path"]), record["sha256"])
        if observed != record:
            raise ProbeError(f"registered input record differs: {key}")
    audit = identity_payload(
        {
            "schema": "privileged-on-policy-teacher-audit-v1",
            "created_at_utc": _now(),
            "output_dir": str(output),
            "registration_identity_sha256": registration["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "all_row_identities_valid": True,
            "analysis_recomputed_exactly": True,
            "file_hashes_valid": True,
            "input_hashes_valid": True,
            "observed_tf_only_steps": complete["observed_tf_only_steps"],
            "observed_video_active_steps": complete["observed_video_active_steps"],
            "protected_test_opened": False,
            "student_training_launched": False,
        }
    )
    path = output / "audit.json"
    if path.exists():
        existing = read_json(path)
        if not identity_valid(existing):
            raise ProbeError("existing audit is invalid")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    exclusive_json(path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", required=True)
    run.add_argument("--expected-source-commit", required=True)
    run.add_argument("--training-source-commit", required=True)
    run.add_argument("--resolved-config", required=True)
    run.add_argument("--snapshot", required=True)
    run.add_argument("--snapshot-sha256", required=True)
    run.add_argument("--train-manifest", required=True)
    run.add_argument("--cache-metadata", required=True)
    run.add_argument("--arm-manifest", required=True)
    run.add_argument("--stage-manifest", required=True)
    run.add_argument("--stage-outcome", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--num-clips", type=int, default=64)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--nfe", type=int, default=4)
    run.add_argument("--seed", type=int, default=20260808)
    run.add_argument("--bootstrap-replicates", type=int, default=10_000)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-dir", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return command_run(args)
        return command_audit(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        output_value = getattr(args, "output_dir", None)
        if args.command == "run" and output_value:
            output = Path(output_value).expanduser()
            if output.is_dir() and not (output / "run_complete.json").exists():
                failure = identity_payload(
                    {
                        "schema": FAILURE_SCHEMA,
                        "created_at_utc": _now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "protected_test_opened": False,
                        "student_training_launched": False,
                    }
                )
                try:
                    exclusive_json(output / "failure.json", failure)
                except FileExistsError:
                    pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
