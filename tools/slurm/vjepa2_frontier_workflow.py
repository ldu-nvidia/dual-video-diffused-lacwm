#!/usr/bin/env python3
"""Small fail-closed helpers for the V-JEPA NFE-frontier Slurm workflow.

The GPU entrypoints remain ordinary shell scripts, but JSON/provenance checks
must not depend on ``jq`` or on fragile shell field extraction. This helper
delegates deep evidence validation to the same modules used by the scientific
evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tools import benchmark_vjepa2_paired_latency as paired  # noqa: E402
from tools import vjepa2_nfe_frontier as frontier  # noqa: E402


WELL_FORMED_INELIGIBLE = 3
FINAL_ARRAY_TASK_IDS = frozenset(range(5))
ADOPTED_CACHE_CODE_PATHS = (
    "projects/latent_action_models/lam",
    "robot_wm/datasets",
    "robot_wm/modeling",
    "tools/build_vjepa2_clip_manifests.py",
    "tools/extract_vjepa2_targets.py",
    "tools/vjepa2_frontier_lockbox.py",
    "tools/vjepa2_nfe_frontier.py",
    "tools/evaluate_vjepa2_quality.py",
    "tools/benchmark_vjepa2_inference.py",
    "tools/benchmark_vjepa2_frontier_latency.py",
    "tools/slurm/vjepa2_frontier_cache.sbatch",
    "tools/slurm/vjepa2_frontier_quality.sbatch",
    "tools/slurm/vjepa2_frontier_latency.sbatch",
)
ADOPTED_RECOVERY_ALLOWED_PATHS = frozenset(
    {
        "docs/experiments/VJEPA2_NFE_FRONTIER_PROTOCOL.md",
        "robot_wm/tests/test_vjepa2_frontier_slurm.py",
        "tools/slurm/README.md",
        "tools/slurm/submit_vjepa2_frontier_workflow.sh",
        "tools/slurm/vjepa2_frontier_confirm.sbatch",
        "tools/slurm/vjepa2_frontier_final_gate.sbatch",
        "tools/slurm/vjepa2_frontier_select_and_submit.sbatch",
        "tools/slurm/vjepa2_frontier_workflow.py",
    }
)
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
}
FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


class WorkflowError(RuntimeError):
    """Raised when a workflow transition lacks immutable evidence."""


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise WorkflowError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise WorkflowError(f"{label} must be a non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise WorkflowError(f"{label} must use its canonical absolute path")
    return resolved


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise WorkflowError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file() or info.st_size <= 0:
        raise WorkflowError(
            f"{label} must be a non-empty non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must contain a JSON object")
    return value


def _load_study(
    study_root_value: str | Path,
    *,
    training_commit: str,
) -> tuple[Path, dict[str, Any]]:
    study_root = _canonical_directory(study_root_value, "study root")
    study_path = _canonical_file(
        study_root / "study_manifest.json", "study manifest"
    )
    study = _read_json(study_path, "study manifest")
    if (
        not paired._identity_valid(study)
        or study.get("kind") != "vjepa2_controlled_video_diffusion_study"
        or study.get("study_root") != str(study_root)
        or study.get("inputs", {})
        .get("repository", {})
        .get("git_commit")
        != training_commit
    ):
        raise WorkflowError("study identity or training commit differs")
    return study_root, study


def _require_line_safe(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise WorkflowError(f"{label} is missing or not line-safe")
    return value


def command_study_values(args: argparse.Namespace) -> int:
    """Emit ordered newline-delimited immutable inputs for shell ``mapfile``."""

    study_root, study = _load_study(
        args.study_root, training_commit=args.training_commit
    )
    inputs = study.get("inputs")
    if not isinstance(inputs, Mapping):
        raise WorkflowError("study inputs are missing")
    runtime = inputs.get("runtime")
    vjepa = inputs.get("vjepa")
    splits = inputs.get("splits")
    if not all(isinstance(value, Mapping) for value in (runtime, vjepa, splits)):
        raise WorkflowError("study runtime/V-JEPA/split inputs are missing")
    train = splits.get("train")
    if not isinstance(train, Mapping):
        raise WorkflowError("study training split is missing")
    values = [
        str(study_root),
        _require_line_safe(study.get("study_id"), "study ID"),
        _require_line_safe(
            inputs.get("repository", {}).get("root"), "training repository root"
        ),
        _require_line_safe(runtime.get("python"), "LACWM Python"),
        _require_line_safe(runtime.get("extractor_python"), "extractor Python"),
        _require_line_safe(runtime.get("wan_dir"), "Wan directory"),
        _require_line_safe(runtime.get("videox_home"), "VideoX-Fun checkout"),
        _require_line_safe(
            vjepa.get("source", {}).get("path"), "V-JEPA source checkout"
        ),
        _require_line_safe(
            vjepa.get("checkpoint", {}).get("path"), "V-JEPA checkpoint"
        ),
        _require_line_safe(
            vjepa.get("checkpoint", {}).get("sha256"),
            "V-JEPA checkpoint SHA-256",
        ),
        _require_line_safe(vjepa.get("pca_stats", {}).get("path"), "PCA file"),
        _require_line_safe(
            vjepa.get("pca_stats", {}).get("sha256"), "PCA SHA-256"
        ),
        _require_line_safe(
            train.get("clip_manifest", {}).get("path"), "training manifest"
        ),
    ]
    for value in values:
        print(value)
    return 0


def command_check_final(args: argparse.Namespace) -> int:
    """Fully validate the final VPM/J1 stage evidence and snapshot hashes."""

    study_root, study = _load_study(
        args.study_root, training_commit=args.training_commit
    )
    arms: dict[str, Any] = {}
    for arm in ("VPM", "J1"):
        try:
            provenance = paired._arm_provenance(
                study_root=study_root,
                study=study,
                arm_code=arm,
            )
        except (paired.PairedLatencyError, KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"{arm} final evidence is invalid: {exc}") from exc
        arms[arm] = {
            "arm_identity_sha256": provenance["arm"]["identity_sha256"],
            "stage_identity_sha256": provenance["stage"]["identity_sha256"],
            "snapshot_sha256": provenance["snapshot_sha256"],
            "completed_updates": provenance["outcome"]["completed_updates"],
        }
    print(
        json.dumps(
            {
                "status": "final_update_1000_validated",
                "study_identity_sha256": study["identity_sha256"],
                "training_git_commit": args.training_commit,
                "arms": arms,
            },
            sort_keys=True,
        )
    )
    return 0


def command_check_submission(args: argparse.Namespace) -> int:
    """Bind the requested afterok dependency to the recorded u1000 array."""

    study_root, study = _load_study(
        args.study_root, training_commit=args.training_commit
    )
    submission_path = _canonical_file(
        study_root / "slurm_submission.json", "controlled-study submission"
    )
    submission = _read_json(submission_path, "controlled-study submission")
    jobs = submission.get("stage_array_job_ids")
    if (
        not paired._identity_valid(submission)
        or submission.get("kind") != "vjepa2_controlled_study_submission"
        or submission.get("study_identity_sha256") != study["identity_sha256"]
        or submission.get("dependency") != "afterok"
        or not isinstance(jobs, list)
    ):
        raise WorkflowError("controlled-study Slurm submission is invalid")
    matches = [
        str(record.get("job_id", "")).split(";", 1)[0].split("_", 1)[0]
        for record in jobs
        if isinstance(record, Mapping)
        and record.get("completed_updates") == 1000
    ]
    if matches != [str(args.final_job_id)]:
        raise WorkflowError(
            "requested dependency is not the recorded update-1000 array"
        )
    print(
        json.dumps(
            {
                "status": "final_update_1000_dependency_validated",
                "job_id": str(args.final_job_id),
                "submission": str(submission_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _array_task_ids(job_id: str, observed_job: str) -> tuple[int, ...] | None:
    """Expand a formatted Slurm ``JobID`` for the pinned five-task array."""

    prefix = f"{job_id}_"
    if not observed_job.startswith(prefix):
        if observed_job == job_id:
            raise WorkflowError(
                "final u1000 accounting returned a non-array allocation"
            )
        return None
    suffix = observed_job[len(prefix) :]
    if re.fullmatch(r"[0-9]+", suffix):
        return (int(suffix),)
    compressed = re.fullmatch(r"\[([0-9,:%-]+)\]", suffix)
    if compressed is None:
        raise WorkflowError(
            f"final u1000 accounting has malformed array task ID: {observed_job}"
        )
    specification = re.sub(r"%[1-9][0-9]*$", "", compressed.group(1))
    if "%" in specification:
        raise WorkflowError(
            f"final u1000 accounting has malformed array throttle: {observed_job}"
        )
    task_ids: list[int] = []
    for component in specification.split(","):
        direct = re.fullmatch(r"[0-9]+", component)
        if direct is not None:
            task_ids.append(int(component))
            continue
        interval = re.fullmatch(r"([0-9]+)-([0-9]+)(?::([1-9][0-9]*))?", component)
        if interval is None:
            raise WorkflowError(
                f"final u1000 accounting has malformed array range: {observed_job}"
            )
        start, stop = int(interval.group(1)), int(interval.group(2))
        step = int(interval.group(3) or "1")
        if start > stop:
            raise WorkflowError(
                f"final u1000 accounting has descending array range: {observed_job}"
            )
        task_ids.extend(range(start, stop + 1, step))
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise WorkflowError(
            f"final u1000 accounting has duplicate/empty array tasks: {observed_job}"
        )
    return tuple(task_ids)


def _final_accounting_records(
    job_id: str,
    rows: Sequence[str],
) -> dict[int, tuple[str, str, str]]:
    """Parse the allocation subset currently visible through ``sacct -X``."""

    records: dict[int, tuple[str, str, str]] = {}
    for line in rows:
        if not line.strip():
            continue
        fields = [field.strip() for field in line.strip().split("|")]
        if len(fields) < 3:
            raise WorkflowError(
                "final u1000 accounting row has the wrong schema"
            )
        observed_job, raw_state, exit_code = fields[:3]
        task_ids = _array_task_ids(job_id, observed_job)
        if task_ids is None:
            continue
        # Slurm may append "+" for a truncated state or " by <uid>" to
        # CANCELLED. Neither changes the terminal classification.
        state = raw_state.split("+", 1)[0].split(maxsplit=1)[0].upper()
        for task_id in task_ids:
            if task_id not in FINAL_ARRAY_TASK_IDS:
                raise WorkflowError(
                    f"final u1000 accounting has unexpected array task: {task_id}"
                )
            if task_id in records:
                raise WorkflowError(
                    f"final u1000 accounting duplicates array task: {task_id}"
                )
            records[task_id] = (observed_job, state, exit_code)
    if not records:
        raise WorkflowError(
            "Slurm accounting has no allocation record for final u1000 job"
        )
    task_records = list(records.values())
    failed = [
        record
        for record in task_records
        if record[1] in FAILED_SLURM_STATES
        or record[2] != "0:0"
    ]
    if failed:
        raise WorkflowError(
            f"final u1000 job has failed accounting records: {failed}"
        )
    unknown = [
        record
        for record in task_records
        if record[1] not in ACTIVE_SLURM_STATES
        and record[1] != "COMPLETED"
    ]
    if unknown:
        raise WorkflowError(
            f"final u1000 job has ambiguous accounting states: {unknown}"
        )
    return records


def _final_queue_records(
    job_id: str,
    rows: Sequence[str],
) -> dict[int, tuple[str, str]]:
    """Parse current-user ``squeue -r`` rows by base and array-task ID."""

    records: dict[int, tuple[str, str]] = {}
    for line in rows:
        if not line.strip():
            continue
        fields = [field.strip() for field in line.rstrip("\n").split("|", 3)]
        if len(fields) != 4:
            raise WorkflowError("final u1000 squeue row has the wrong schema")
        base_job_id, raw_task_id, raw_state, reason = fields
        if base_job_id != job_id:
            continue
        if re.fullmatch(r"[0-9]+", raw_task_id) is None:
            raise WorkflowError(
                f"final u1000 squeue has malformed task ID: {raw_task_id}"
            )
        task_id = int(raw_task_id)
        if task_id not in FINAL_ARRAY_TASK_IDS:
            raise WorkflowError(
                f"final u1000 squeue has unexpected array task: {task_id}"
            )
        if task_id in records:
            raise WorkflowError(
                f"final u1000 squeue duplicates array task: {task_id}"
            )
        state = raw_state.split("+", 1)[0].split(maxsplit=1)[0].upper()
        if state not in ACTIVE_SLURM_STATES:
            raise WorkflowError(
                f"final u1000 squeue has ambiguous active state: {raw_state}"
            )
        records[task_id] = (state, reason)
    return records


def classify_final_job_rows(
    job_id: str,
    accounting_rows: Sequence[str],
    squeue_rows: Sequence[str] = (),
) -> str:
    """Combine live queue truth with fail-closed allocation accounting."""

    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise WorkflowError("final job ID must be a positive integer")
    accounting = _final_accounting_records(job_id, accounting_rows)
    queue = _final_queue_records(job_id, squeue_rows)
    if queue:
        observed_queue_tasks = set(queue)
        if observed_queue_tasks != FINAL_ARRAY_TASK_IDS:
            missing = sorted(FINAL_ARRAY_TASK_IDS - observed_queue_tasks)
            extra = sorted(observed_queue_tasks - FINAL_ARRAY_TASK_IDS)
            raise WorkflowError(
                "final u1000 squeue task set differs: "
                f"missing={missing}, extra={extra}"
            )
        for task_id, (_, accounting_state, _) in accounting.items():
            queue_state = queue[task_id][0]
            if accounting_state != queue_state:
                raise WorkflowError(
                    "final u1000 active state mismatch for task "
                    f"{task_id}: sacct={accounting_state}, squeue={queue_state}"
                )
        return "active_afterok"

    observed_accounting_tasks = set(accounting)
    if observed_accounting_tasks != FINAL_ARRAY_TASK_IDS:
        missing = sorted(FINAL_ARRAY_TASK_IDS - observed_accounting_tasks)
        extra = sorted(observed_accounting_tasks - FINAL_ARRAY_TASK_IDS)
        raise WorkflowError(
            "final u1000 terminal accounting task set differs: "
            f"missing={missing}, extra={extra}"
        )
    if all(
        record[1] == "COMPLETED" and record[2] == "0:0"
        for record in accounting.values()
    ):
        return "terminal_success"
    raise WorkflowError(
        "final u1000 left squeue before exact successful terminal accounting"
    )


def command_classify_final_job(args: argparse.Namespace) -> int:
    completed = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            args.final_job_id,
            "--format=JobID%64,State%32,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise WorkflowError(
            "could not query final u1000 Slurm accounting: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    queued = subprocess.run(
        [
            "squeue",
            "-r",
            "--user",
            pwd.getpwuid(os.getuid()).pw_name,
            "-h",
            "-o",
            "%F|%K|%T|%r",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if queued.returncode:
        raise WorkflowError(
            "could not query final u1000 live Slurm queue: "
            f"{queued.stderr.strip() or queued.stdout.strip()}"
        )
    mode = classify_final_job_rows(
        args.final_job_id,
        completed.stdout.splitlines(),
        queued.stdout.splitlines(),
    )
    print(mode)
    return 0


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise WorkflowError(
            f"git {' '.join(arguments)} failed for {repo}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _validate_adopted_cache_repositories(
    *,
    producer_repo: Path,
    producer_commit: str,
    current_repo: Path,
    current_commit: str,
    training_commit: str,
) -> dict[str, Any]:
    if producer_repo == current_repo:
        raise WorkflowError(
            "adopted cache producer and recovery controller need distinct worktrees"
        )
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise WorkflowError("adopted cache evaluator commit is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", current_commit) is None:
        raise WorkflowError("recovery evaluator commit is invalid")
    for repo, commit, label in (
        (producer_repo, producer_commit, "adopted cache producer"),
        (current_repo, current_commit, "recovery controller"),
    ):
        if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
            raise WorkflowError(f"{label} path is not a Git worktree root")
        if _git(repo, "rev-parse", "HEAD") != commit:
            raise WorkflowError(f"{label} worktree HEAD changed")
        if _git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise WorkflowError(f"{label} worktree is dirty")
    compatibility = frontier.git_inference_compatibility(
        producer_repo,
        training_commit=training_commit,
        tool_commit=producer_commit,
    )
    descendant = subprocess.run(
        [
            "git",
            "-C",
            str(current_repo),
            "merge-base",
            "--is-ancestor",
            producer_commit,
            current_commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if descendant.returncode:
        raise WorkflowError(
            "recovery evaluator is not a descendant of the adopted cache evaluator"
        )
    changed_paths = {
        value
        for value in _git(
            current_repo,
            "diff",
            "--name-only",
            producer_commit,
            current_commit,
            "--",
        ).splitlines()
        if value
    }
    unexpected_changes = sorted(changed_paths - ADOPTED_RECOVERY_ALLOWED_PATHS)
    if unexpected_changes:
        raise WorkflowError(
            "recovery commit changes non-orchestration paths: "
            f"{unexpected_changes}"
        )
    code_objects: dict[str, str] = {}
    for path in ADOPTED_CACHE_CODE_PATHS:
        producer_object = _git(
            current_repo, "rev-parse", f"{producer_commit}:{path}"
        )
        current_object = _git(
            current_repo, "rev-parse", f"{current_commit}:{path}"
        )
        if producer_object != current_object:
            raise WorkflowError(
                f"adopted cache/scientific code changed in recovery commit: {path}"
            )
        code_objects[path] = producer_object
    return {
        "producer_commit_is_compatible_with_training": True,
        "recovery_commit_descends_from_producer": True,
        "scientific_code_objects_unchanged": True,
        "recovery_changed_paths": sorted(changed_paths),
        "changes_are_recovery_allowlisted": True,
        "inference_code_compatibility": compatibility,
        "code_objects": code_objects,
    }


def _requested_tres(value: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for component in value.split(","):
        key, separator, item = component.strip().partition("=")
        if not separator or not key or not item or key in records:
            raise WorkflowError("adopted cache ReqTRES is malformed")
        records[key] = item
    return records


def _memory_value(value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(("c", "n")):
        normalized = normalized[:-1]
    if re.fullmatch(r"[1-9][0-9]*[KMGTPE]?", normalized, re.IGNORECASE) is None:
        raise WorkflowError(f"adopted cache memory value is malformed: {value}")
    return normalized.upper()


def _submit_line_tokens(value: str) -> list[str]:
    try:
        tokens = shlex.split(value)
    except ValueError as exc:
        raise WorkflowError("adopted cache SubmitLine is not valid shell syntax") from exc
    if not tokens or Path(tokens[0]).name != "sbatch":
        raise WorkflowError("adopted cache SubmitLine is not an sbatch command")
    return tokens


def _scontrol_fields(value: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for component in value.strip().split():
        key, separator, item = component.partition("=")
        if not separator or not key or key in records:
            raise WorkflowError("adopted cache scontrol record is malformed")
        records[key] = item
    return records


def _validate_adopted_cache_scontrol(
    args: argparse.Namespace,
    row: str,
    *,
    expected_script: Path,
    expected_user: str,
) -> dict[str, str]:
    fields = _scontrol_fields(row)
    expected_dependency = f"afterok:{args.final_job_id}"
    exact = {
        "JobId": str(args.job_id),
        "JobName": "vjepa2-frontier-cache",
        "UserId": f"{expected_user}({os.getuid()})",
        "Account": args.account,
        "QOS": args.qos,
        "JobState": "PENDING",
        "Reason": "Dependency",
        "Requeue": "0",
        "Restarts": "0",
        "BatchFlag": "1",
        "ExitCode": "0:0",
        "TimeLimit": args.cache_time,
        "Partition": args.partition,
        "NumNodes": "1-1",
        "NumCPUs": str(args.cache_cpus),
        "NumTasks": "1",
        "CPUs/Task": str(args.cache_cpus),
        "AllocTRES": "(null)",
        "Command": str(expected_script),
        "WorkDir": pwd.getpwuid(os.getuid()).pw_dir,
        "StdErr": (
            f"{args.log_dir}/vjepa2-frontier-cache-{args.job_id}.err"
        ),
        "StdOut": (
            f"{args.log_dir}/vjepa2-frontier-cache-{args.job_id}.out"
        ),
        "TresPerNode": "gres/gpu:1",
        "TresPerTask": f"cpu={args.cache_cpus}",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            raise WorkflowError(
                f"adopted cache scontrol {key} differs: "
                f"{fields.get(key)!r} != {expected!r}"
            )
    if re.fullmatch(
        rf"{re.escape(expected_dependency)}(?:_\*)?\(unfulfilled\)",
        fields.get("Dependency", ""),
    ) is None:
        raise WorkflowError("adopted cache scontrol dependency differs")
    tres = _requested_tres(fields.get("ReqTRES", ""))
    gpu_values = [
        item
        for key, item in tres.items()
        if key == "gres/gpu" or key.startswith("gres/gpu:")
    ]
    if (
        tres.get("node") != "1"
        or tres.get("cpu") != str(args.cache_cpus)
        or _memory_value(tres.get("mem", "")) != _memory_value(args.cache_memory)
        or gpu_values != ["1"]
    ):
        raise WorkflowError("adopted cache scontrol ReqTRES differs")
    return {
        key: fields[key]
        for key in (
            "JobId",
            "UserId",
            "JobState",
            "Reason",
            "Dependency",
            "Requeue",
            "Restarts",
            "BatchFlag",
            "ExitCode",
            "ReqTRES",
            "Command",
            "StdErr",
            "StdOut",
        )
    }


def validate_adopted_cache_row(
    args: argparse.Namespace,
    row: str,
    scontrol_row: str,
) -> dict[str, Any]:
    fields = row.rstrip("\n").split("|")
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) != 13:
        raise WorkflowError("adopted cache accounting row has the wrong schema")
    (
        job_id,
        job_name,
        user,
        raw_state,
        exit_code,
        account,
        qos,
        partition,
        req_tres,
        req_cpus,
        req_mem,
        time_limit,
        submit_line,
    ) = (value.strip() for value in fields)
    expected_job_id = str(args.job_id)
    if job_id != expected_job_id or re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise WorkflowError("adopted cache accounting JobID differs")
    state = raw_state.split("+", 1)[0].split(maxsplit=1)[0].upper()
    if state != "PENDING" or exit_code != "0:0":
        raise WorkflowError(
            "adopted cache job must still be PENDING with ExitCode 0:0"
        )
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    if user != expected_user:
        raise WorkflowError(
            f"adopted cache owner differs: {user!r} != {expected_user!r}"
        )
    expected_scalars = {
        "JobName": (job_name, "vjepa2-frontier-cache"),
        "Account": (account, args.account),
        "QOS": (qos, args.qos),
        "Partition": (partition, args.partition),
        "ReqCPUS": (req_cpus, str(args.cache_cpus)),
        "Timelimit": (time_limit, args.cache_time),
    }
    for label, (observed, expected) in expected_scalars.items():
        if observed != expected:
            raise WorkflowError(
                f"adopted cache {label} differs: {observed!r} != {expected!r}"
            )
    if _memory_value(req_mem) != _memory_value(args.cache_memory):
        raise WorkflowError("adopted cache ReqMem differs")
    tres = _requested_tres(req_tres)
    gpu_values = [
        value
        for key, value in tres.items()
        if key == "gres/gpu" or key.startswith("gres/gpu:")
    ]
    if any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in gpu_values):
        raise WorkflowError("adopted cache GPU TRES is malformed")
    gpu_count = sum(
        int(value)
        for value in gpu_values
    )
    if (
        tres.get("node") != "1"
        or tres.get("cpu") != str(args.cache_cpus)
        or _memory_value(tres.get("mem", "")) != _memory_value(args.cache_memory)
        or gpu_count != 1
    ):
        raise WorkflowError("adopted cache requested TRES differ")
    expected_dependency = f"afterok:{args.final_job_id}"

    tokens = _submit_line_tokens(submit_line)
    try:
        script_index = next(
            index
            for index, token in enumerate(tokens[1:], start=1)
            if not token.startswith("-")
        )
    except StopIteration as exc:
        raise WorkflowError("adopted cache SubmitLine lacks a batch script") from exc
    script_arguments = tokens[script_index + 1 :]
    if len(script_arguments) % 2:
        raise WorkflowError("adopted cache script arguments are not name/value pairs")
    argument_names = script_arguments[::2]
    if any(not name.startswith("--") for name in argument_names) or len(
        argument_names
    ) != len(set(argument_names)):
        raise WorkflowError("adopted cache script arguments are malformed")
    submitted = dict(zip(argument_names, script_arguments[1::2]))
    producer_repo_value = submitted.get("--repo-root", "")
    producer_commit = submitted.get("--evaluator-commit", "")
    producer_repo = _canonical_directory(
        producer_repo_value, "adopted cache producer repository"
    )
    current_repo = _canonical_directory(
        args.current_repo_root, "recovery controller repository"
    )
    expected_script = producer_repo / "tools/slurm/vjepa2_frontier_cache.sbatch"
    expected_tokens = [
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--gpus-per-node=1",
        f"--cpus-per-task={args.cache_cpus}",
        f"--mem={args.cache_memory}",
        f"--time={args.cache_time}",
        f"--partition={args.partition}",
        f"--dependency={expected_dependency}",
        "--no-requeue",
        "--open-mode=append",
        "--export=ALL",
        "--job-name=vjepa2-frontier-cache",
        f"--output={args.log_dir}/%x-%j.out",
        f"--error={args.log_dir}/%x-%j.err",
        f"--account={args.account}",
        f"--qos={args.qos}",
        str(expected_script),
        "--repo-root",
        str(producer_repo),
        "--study-root",
        args.study_root,
        "--training-commit",
        args.training_commit,
        "--evaluator-commit",
        producer_commit,
        "--python",
        args.python,
        "--extractor-python",
        args.extractor_python,
        "--vjepa-source",
        args.vjepa_source,
        "--vjepa-checkpoint",
        args.vjepa_checkpoint,
        "--vjepa-checkpoint-sha256",
        args.vjepa_checkpoint_sha256,
        "--pca",
        args.pca,
        "--pca-sha256",
        args.pca_sha256,
        "--train-manifest",
        args.train_manifest,
    ]
    if tokens[1:] != expected_tokens:
        raise WorkflowError(
            "adopted cache SubmitLine differs from the exact expected cache command"
        )
    scontrol_evidence = _validate_adopted_cache_scontrol(
        args,
        scontrol_row,
        expected_script=expected_script,
        expected_user=expected_user,
    )
    repository_evidence = _validate_adopted_cache_repositories(
        producer_repo=producer_repo,
        producer_commit=producer_commit,
        current_repo=current_repo,
        current_commit=args.current_evaluator_commit,
        training_commit=args.training_commit,
    )
    return {
        "status": "pending_cache_job_adopted",
        "job_id": job_id,
        "user": user,
        "state": state,
        "exit_code": exit_code,
        "account": account,
        "qos": qos,
        "partition": partition,
        "requested_tres": tres,
        "dependency": expected_dependency,
        "scontrol": scontrol_evidence,
        "submit_line_exact_match": True,
        "submit_line_sha256": hashlib.sha256(
            submit_line.encode("utf-8")
        ).hexdigest(),
        "producer_repo_root": str(producer_repo),
        "producer_evaluator_commit": producer_commit,
        "recovery_repo_root": str(current_repo),
        "recovery_evaluator_commit": args.current_evaluator_commit,
        "repository_evidence": repository_evidence,
    }


def command_validate_adopted_cache(args: argparse.Namespace) -> int:
    completed = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--jobs",
            str(args.job_id),
            "--format="
            "JobID%64,JobName%64,User%64,State%32,ExitCode,Account%64,QOS%64,"
            "Partition%64,ReqTRES%512,ReqCPUS,ReqMem,Timelimit,"
            "SubmitLine%4096",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise WorkflowError(
            "could not query adopted cache Slurm accounting: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise WorkflowError(
            f"adopted cache accounting must contain exactly one row, found {len(rows)}"
        )
    control = subprocess.run(
        ["scontrol", "show", "job", "--oneliner", str(args.job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if control.returncode:
        raise WorkflowError(
            "could not query adopted cache live Slurm state: "
            f"{control.stderr.strip() or control.stdout.strip()}"
        )
    control_rows = [
        line for line in control.stdout.splitlines() if line.strip()
    ]
    if len(control_rows) != 1:
        raise WorkflowError(
            "adopted cache scontrol query must contain exactly one row, "
            f"found {len(control_rows)}"
        )
    evidence = validate_adopted_cache_row(args, rows[0], control_rows[0])
    print(evidence["producer_repo_root"])
    print(evidence["producer_evaluator_commit"])
    print(json.dumps(evidence, sort_keys=True))
    return 0


def command_require_selection(args: argparse.Namespace) -> int:
    """Return 0 only for a reproducible confirmatory validation winner.

    Exit code 3 is reserved for a valid selection artifact that deliberately
    records no eligible pair.  Invalid, stale, or non-reproducible evidence
    returns the ordinary fail-closed error code 2.
    """

    path = _canonical_file(args.selection, "frontier selection")
    selection = _read_json(path, "frontier selection")
    if (
        not frontier.identity_valid(selection)
        or selection.get("kind") != frontier.KIND_SELECTION
    ):
        raise WorkflowError("frontier selection identity/kind is invalid")
    if selection.get("confirmatory_eligible") is not True:
        if (
            selection.get("confirmatory_eligible") is False
            and selection.get("selected_pair") is None
            and selection.get("selection_split") == "validation"
        ):
            print(
                json.dumps(
                    {
                        "status": "no_confirmatory_candidate",
                        "selection": str(path),
                        "lockbox_submission_allowed": False,
                    },
                    sort_keys=True,
                )
            )
            return WELL_FORMED_INELIGIBLE
        raise WorkflowError("selection eligibility field is malformed")
    try:
        validated = frontier.validate_confirmatory_selection(selection)
    except (frontier.FrontierError, KeyError, TypeError, ValueError, OSError) as exc:
        raise WorkflowError(
            f"confirmatory selection does not reproduce: {exc}"
        ) from exc
    pair = validated["selection"]["selected_pair"]
    print(
        json.dumps(
            {
                "status": "confirmatory_candidate_validated",
                "selection_identity_sha256": selection["identity_sha256"],
                "lockbox_registration_identity_sha256": selection[
                    "lockbox_registration"
                ]["identity_sha256"],
                "j1_nfe": pair["left"]["nfe"],
                "vpm_frontier_nfe": pair["reference"]["nfe"],
                "lockbox_submission_allowed": True,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    values = commands.add_parser("study-values")
    values.add_argument("--study-root", required=True)
    values.add_argument("--training-commit", required=True)
    values.set_defaults(handler=command_study_values)

    final = commands.add_parser("check-final")
    final.add_argument("--study-root", required=True)
    final.add_argument("--training-commit", required=True)
    final.set_defaults(handler=command_check_final)

    submission = commands.add_parser("check-submission")
    submission.add_argument("--study-root", required=True)
    submission.add_argument("--training-commit", required=True)
    submission.add_argument("--final-job-id", required=True)
    submission.set_defaults(handler=command_check_submission)

    classify = commands.add_parser("classify-final-job")
    classify.add_argument("--final-job-id", required=True)
    classify.set_defaults(handler=command_classify_final_job)

    adoption = commands.add_parser("validate-adopted-cache")
    adoption.add_argument("--job-id", required=True)
    adoption.add_argument("--study-root", required=True)
    adoption.add_argument("--training-commit", required=True)
    adoption.add_argument("--current-repo-root", required=True)
    adoption.add_argument("--current-evaluator-commit", required=True)
    adoption.add_argument("--final-job-id", required=True)
    adoption.add_argument("--python", required=True)
    adoption.add_argument("--extractor-python", required=True)
    adoption.add_argument("--vjepa-source", required=True)
    adoption.add_argument("--vjepa-checkpoint", required=True)
    adoption.add_argument("--vjepa-checkpoint-sha256", required=True)
    adoption.add_argument("--pca", required=True)
    adoption.add_argument("--pca-sha256", required=True)
    adoption.add_argument("--train-manifest", required=True)
    adoption.add_argument("--partition", required=True)
    adoption.add_argument("--account", required=True)
    adoption.add_argument("--qos", required=True)
    adoption.add_argument("--cache-time", required=True)
    adoption.add_argument("--cache-cpus", required=True)
    adoption.add_argument("--cache-memory", required=True)
    adoption.add_argument("--log-dir", required=True)
    adoption.set_defaults(handler=command_validate_adopted_cache)

    selection = commands.add_parser("require-selection")
    selection.add_argument("--selection", required=True)
    selection.set_defaults(handler=command_require_selection)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
