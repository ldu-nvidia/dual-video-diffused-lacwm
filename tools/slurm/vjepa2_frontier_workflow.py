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
COMPLETED_RECOVERY_CONTROLLER_ALLOWED_PATHS = frozenset(
    {
        "docs/experiments/VJEPA2_NFE_FRONTIER_PROTOCOL.md",
        "robot_wm/tests/test_analyze_vjepa2_controlled_study.py",
        "robot_wm/tests/test_benchmark_vjepa2_paired_latency.py",
        "robot_wm/tests/test_vjepa2_frontier_slurm.py",
        "robot_wm/tests/test_vjepa2_lockbox_causality.py",
        "tools/analyze_vjepa2_controlled_study.py",
        "tools/benchmark_vjepa2_paired_latency.py",
        "tools/evaluate_vjepa2_quality.py",
        "tools/slurm/README.md",
        "tools/slurm/recover_vjepa2_frontier_workflow.sh",
        "tools/slurm/submit_vjepa2_frontier_causality.sh",
        "tools/slurm/submit_vjepa2_frontier_workflow.sh",
        "tools/slurm/vjepa2_frontier_causality_confirm.sbatch",
        "tools/slurm/vjepa2_frontier_causality_quality.sbatch",
        "tools/slurm/vjepa2_frontier_confirm.sbatch",
        "tools/slurm/vjepa2_frontier_final_gate.sbatch",
        "tools/slurm/vjepa2_frontier_select_and_submit.sbatch",
        "tools/slurm/vjepa2_frontier_workflow.py",
        "tools/vjepa2_lockbox_causality.py",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_evidence(path: Path, *, identity: str | None = None) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if identity is not None:
        evidence["identity_sha256"] = identity
    return evidence


def _allocation_row(line: str) -> dict[str, Any]:
    fields = line.rstrip("\n").split("|")
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) != 13:
        raise WorkflowError("recovery accounting row has the wrong schema")
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
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise WorkflowError("recovery accounting JobID is malformed")
    state = raw_state.split("+", 1)[0].split(maxsplit=1)[0].upper()
    return {
        "job_id": job_id,
        "job_name": job_name,
        "user": user,
        "state": state,
        "exit_code": exit_code,
        "account": account,
        "qos": qos,
        "partition": partition,
        "requested_tres": _requested_tres(req_tres),
        "requested_cpus": req_cpus,
        "requested_memory": req_mem,
        "time_limit": time_limit,
        "submit_line": submit_line,
        "submit_line_sha256": hashlib.sha256(
            submit_line.encode("utf-8")
        ).hexdigest(),
    }


def _expected_scheduler_tokens(
    *,
    job_name: str,
    gpus: int,
    cpus: int,
    memory: str,
    time_limit: str,
    dependency: str,
    partition: str,
    account: str,
    qos: str,
    log_dir: Path,
) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        f"--gpus-per-node={gpus}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--partition={partition}",
        f"--dependency={dependency}",
        "--no-requeue",
        "--open-mode=append",
        "--export=ALL",
        f"--job-name={job_name}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        f"--account={account}",
        f"--qos={qos}",
    ]


def _require_exact_submit_line(
    record: Mapping[str, Any],
    *,
    scheduler_tokens: Sequence[str],
    script: Path,
    script_arguments: Sequence[str],
) -> None:
    tokens = _submit_line_tokens(str(record.get("submit_line", "")))
    expected = [*scheduler_tokens, str(script), *script_arguments]
    if tokens != expected:
        raise WorkflowError(
            f"{record.get('job_id')} SubmitLine differs from the recorded DAG"
        )


def _require_allocation(
    record: Mapping[str, Any],
    *,
    job_id: str,
    job_name: str,
    state: str,
    exit_code: str,
    gpus: int,
    cpus: int,
    memory: str,
    time_limit: str,
    partition: str,
    account: str,
    qos: str,
) -> None:
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    expected = {
        "job_id": job_id,
        "job_name": job_name,
        "user": expected_user,
        "state": state,
        "exit_code": exit_code,
        "account": account,
        "qos": qos,
        "partition": partition,
        "requested_cpus": str(cpus),
        "time_limit": time_limit,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise WorkflowError(
                f"{job_id} accounting {field} differs: "
                f"{record.get(field)!r} != {value!r}"
            )
    if _memory_value(str(record.get("requested_memory", ""))) != _memory_value(
        memory
    ):
        raise WorkflowError(f"{job_id} accounting requested memory differs")
    tres = record.get("requested_tres")
    if not isinstance(tres, Mapping):
        raise WorkflowError(f"{job_id} accounting requested TRES are missing")
    gpu_values = [
        value
        for key, value in tres.items()
        if key == "gres/gpu" or key.startswith("gres/gpu:")
    ]
    if (
        tres.get("node") != "1"
        or tres.get("cpu") != str(cpus)
        or _memory_value(str(tres.get("mem", ""))) != _memory_value(memory)
        or any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in gpu_values)
        or sum(int(value) for value in gpu_values) != gpus
    ):
        raise WorkflowError(f"{job_id} accounting requested TRES differ")


def _validated_recovery_lacwm_python(
    *,
    study: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> str:
    runtime = study.get("inputs", {}).get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise WorkflowError("study runtime evidence is missing")
    study_python = _require_line_safe(
        runtime.get("python"), "study LACWM Python"
    )
    interpreters = submission.get("interpreters")
    if not isinstance(interpreters, Mapping):
        raise WorkflowError("predecessor interpreter evidence is missing")
    submitted_python = _require_line_safe(
        interpreters.get("lacwm"), "predecessor submitted LACWM Python"
    )
    study_path = Path(study_python)
    submitted_path = Path(submitted_python)
    if not study_path.is_absolute() or not submitted_path.is_absolute():
        raise WorkflowError("recovery LACWM Python paths must be absolute")
    try:
        study_resolved = study_path.resolve(strict=True)
        submitted_resolved = submitted_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkflowError(
            "recovery LACWM Python evidence does not resolve"
        ) from exc
    if (
        study_path != study_resolved
        or not study_resolved.is_file()
        or not os.access(study_resolved, os.X_OK)
    ):
        raise WorkflowError(
            "study LACWM Python must be a canonical executable file"
        )
    if (
        not submitted_resolved.is_file()
        or not os.access(submitted_path, os.X_OK)
        or submitted_resolved != study_resolved
    ):
        raise WorkflowError(
            "predecessor submitted LACWM Python does not resolve to "
            "the study runtime"
        )
    # Preserve the immutable launcher spelling for exact SubmitLine matching.
    # The study records the canonical target, whereas the original sbatch
    # commands intentionally contain the environment symlink.
    return submitted_python


def _validate_recovery_submission(
    *,
    path: Path,
    study_root: Path,
    training_commit: str,
    cache_job_id: str,
    failed_gate_job_id: str,
    cancelled_vpm_job_id: str,
    cancelled_j1_job_id: str,
    cancelled_selection_job_id: str,
    final_job_id: str,
    scientific_repo: Path,
    scientific_commit: str,
) -> tuple[dict[str, Any], Path, Path, str]:
    expected_path = study_root / "_frontier_slurm" / "submission.json"
    if path != expected_path:
        raise WorkflowError("recovery predecessor submission path differs")
    submission = _read_json(path, "frontier predecessor submission")
    prior_controller_value = submission.get("controller_repo_root")
    prior_controller_commit = submission.get("controller_git_commit")
    if not isinstance(prior_controller_value, str):
        raise WorkflowError("predecessor controller repository is missing")
    prior_controller = _canonical_directory(
        prior_controller_value, "predecessor controller repository"
    )
    expected = {
        "kind": "vjepa2_nfe_frontier_slurm_submission",
        "schema_version": 1,
        "study_root": str(study_root),
        "training_git_commit": training_commit,
        "final_update_1000_job_id": final_job_id,
        "evaluator_git_commit": scientific_commit,
        "scientific_evaluator_repo_root": str(scientific_repo),
        "cache_job_id": cache_job_id,
        "cache_job_adopted": True,
        "final_artifact_gate_job_id": failed_gate_job_id,
        "selection_gate_job_id": cancelled_selection_job_id,
        "lockbox_jobs_submitted_at_initial_submission": False,
        "lockbox_submission_requires_confirmatory_eligible": True,
    }
    for field, value in expected.items():
        if submission.get(field) != value:
            raise WorkflowError(
                f"predecessor submission {field} differs: "
                f"{submission.get(field)!r} != {value!r}"
            )
    if (
        not isinstance(prior_controller_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", prior_controller_commit) is None
        or submission.get("validation_job_ids")
        != {"J1": cancelled_j1_job_id, "VPM": cancelled_vpm_job_id}
        or submission.get("dependencies")
        != {
            "cache": f"afterok:{final_job_id} (adopted pending cache)",
            "final_artifact_gate": f"afterok:{final_job_id}",
            "selection": (
                f"afterok:{cancelled_vpm_job_id}:{cancelled_j1_job_id}"
            ),
            "validation": f"afterok:{cache_job_id}:{failed_gate_job_id}",
        }
    ):
        raise WorkflowError("predecessor submission DAG identities differ")
    interpreters = submission.get("interpreters")
    if not isinstance(interpreters, Mapping):
        raise WorkflowError("predecessor interpreter evidence is missing")
    _require_line_safe(
        interpreters.get("lacwm"), "predecessor submitted LACWM Python"
    )
    adoption = submission.get("cache_adoption_evidence")
    if (
        not isinstance(adoption, Mapping)
        or adoption.get("job_id") != cache_job_id
        or adoption.get("state") != "PENDING"
        or adoption.get("exit_code") != "0:0"
        or adoption.get("producer_repo_root") != str(scientific_repo)
        or adoption.get("producer_evaluator_commit") != scientific_commit
        or adoption.get("submit_line_exact_match") is not True
        or re.fullmatch(
            r"[0-9a-f]{64}", str(adoption.get("submit_line_sha256", ""))
        )
        is None
    ):
        raise WorkflowError("predecessor cache-adoption evidence differs")
    prior_log_dir = _canonical_directory(
        path.parent / "logs", "predecessor Slurm log directory"
    )
    return (
        submission,
        prior_controller,
        prior_log_dir,
        prior_controller_commit,
    )


def validate_completed_recovery_rows(
    args: argparse.Namespace,
    rows: Sequence[str],
    *,
    study: Mapping[str, Any],
    submission: Mapping[str, Any],
    prior_controller: Path,
    prior_controller_commit: str,
    prior_log_dir: Path,
    scientific_repo: Path,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for line in rows:
        if not line.strip():
            continue
        record = _allocation_row(line)
        job_id = record["job_id"]
        if job_id in records:
            raise WorkflowError(f"recovery accounting duplicates job {job_id}")
        records[job_id] = record
    expected_ids = {
        args.cache_job_id,
        args.failed_gate_job_id,
        args.cancelled_vpm_job_id,
        args.cancelled_j1_job_id,
        args.cancelled_selection_job_id,
    }
    if set(records) != expected_ids:
        raise WorkflowError(
            "recovery accounting job set differs: "
            f"missing={sorted(expected_ids - set(records))}, "
            f"extra={sorted(set(records) - expected_ids)}"
        )
    job_specs = {
        args.cache_job_id: (
            "vjepa2-frontier-cache",
            "COMPLETED",
            "0:0",
            1,
            32,
            "256G",
            "01:00:00",
        ),
        args.failed_gate_job_id: (
            "vjepa2-frontier-u1000-gate",
            "FAILED",
            "2:0",
            1,
            16,
            "64G",
            "01:00:00",
        ),
        args.cancelled_vpm_job_id: (
            "vjepa2-frontier-val-vpm",
            "CANCELLED",
            "0:0",
            8,
            160,
            "1000G",
            "04:00:00",
        ),
        args.cancelled_j1_job_id: (
            "vjepa2-frontier-val-j1",
            "CANCELLED",
            "0:0",
            8,
            160,
            "1000G",
            "04:00:00",
        ),
        args.cancelled_selection_job_id: (
            "vjepa2-frontier-select",
            "CANCELLED",
            "0:0",
            1,
            16,
            "64G",
            "01:00:00",
        ),
    }
    for job_id, spec in job_specs.items():
        name, state, exit_code, gpus, cpus, memory, time_limit = spec
        _require_allocation(
            records[job_id],
            job_id=job_id,
            job_name=name,
            state=state,
            exit_code=exit_code,
            gpus=gpus,
            cpus=cpus,
            memory=memory,
            time_limit=time_limit,
            partition=args.partition,
            account=args.account,
            qos=args.qos,
        )

    adoption = submission["cache_adoption_evidence"]
    cache_record = records[args.cache_job_id]
    if cache_record["submit_line_sha256"] != adoption["submit_line_sha256"]:
        raise WorkflowError("completed cache SubmitLine differs from adopted job")

    runtime = study.get("inputs", {}).get("runtime", {})
    python = _validated_recovery_lacwm_python(
        study=study, submission=submission
    )
    wan_dir = _require_line_safe(runtime.get("wan_dir"), "study Wan directory")
    videox_home = _require_line_safe(
        runtime.get("videox_home"), "study VideoX checkout"
    )
    common = {
        "partition": args.partition,
        "account": args.account,
        "qos": args.qos,
        "log_dir": prior_log_dir,
    }
    gate_scheduler = _expected_scheduler_tokens(
        job_name="vjepa2-frontier-u1000-gate",
        gpus=1,
        cpus=16,
        memory="64G",
        time_limit="01:00:00",
        dependency=f"afterok:{args.final_job_id}",
        **common,
    )
    _require_exact_submit_line(
        records[args.failed_gate_job_id],
        scheduler_tokens=gate_scheduler,
        script=prior_controller / "tools/slurm/vjepa2_frontier_final_gate.sbatch",
        script_arguments=[
            "--repo-root",
            str(prior_controller),
            "--study-root",
            args.study_root,
            "--training-commit",
            args.training_commit,
            "--evaluator-commit",
            prior_controller_commit,
            "--scientific-repo-root",
            str(scientific_repo),
            "--scientific-evaluator-commit",
            args.scientific_commit,
            "--python",
            python,
        ],
    )

    validation_dependency = (
        f"afterok:{args.cache_job_id}:{args.failed_gate_job_id}"
    )
    for arm, job_id in (
        ("VPM", args.cancelled_vpm_job_id),
        ("J1", args.cancelled_j1_job_id),
    ):
        scheduler = _expected_scheduler_tokens(
            job_name=f"vjepa2-frontier-val-{arm.lower()}",
            gpus=8,
            cpus=160,
            memory="1000G",
            time_limit="04:00:00",
            dependency=validation_dependency,
            **common,
        )
        _require_exact_submit_line(
            records[job_id],
            scheduler_tokens=scheduler,
            script=scientific_repo / "tools/slurm/vjepa2_frontier_quality.sbatch",
            script_arguments=[
                "--repo-root",
                str(scientific_repo),
                "--study-root",
                args.study_root,
                "--training-commit",
                args.training_commit,
                "--evaluator-commit",
                args.scientific_commit,
                "--python",
                python,
                "--wan-dir",
                wan_dir,
                "--videox-home",
                videox_home,
                "--split",
                "validation",
                "--arm",
                arm,
            ],
        )

    selection_scheduler = _expected_scheduler_tokens(
        job_name="vjepa2-frontier-select",
        gpus=1,
        cpus=16,
        memory="64G",
        time_limit="01:00:00",
        dependency=(
            f"afterok:{args.cancelled_vpm_job_id}:"
            f"{args.cancelled_j1_job_id}"
        ),
        **common,
    )
    _require_exact_submit_line(
        records[args.cancelled_selection_job_id],
        scheduler_tokens=selection_scheduler,
        script=(
            prior_controller
            / "tools/slurm/vjepa2_frontier_select_and_submit.sbatch"
        ),
        script_arguments=[
            "--repo-root",
            str(prior_controller),
            "--study-root",
            args.study_root,
            "--training-commit",
            args.training_commit,
            "--evaluator-commit",
            prior_controller_commit,
            "--scientific-repo-root",
            str(scientific_repo),
            "--scientific-evaluator-commit",
            args.scientific_commit,
            "--python",
            python,
            "--wan-dir",
            wan_dir,
            "--videox-home",
            videox_home,
            "--partition",
            args.partition,
            "--quality-time",
            "04:00:00",
            "--control-time",
            "01:00:00",
            "--timing-time",
            "04:00:00",
            "--quality-cpus",
            "160",
            "--quality-mem",
            "1000G",
            "--control-cpus",
            "16",
            "--control-mem",
            "64G",
            "--timing-cpus",
            "32",
            "--timing-mem",
            "256G",
            "--account",
            args.account,
            "--qos",
            args.qos,
            "--log-dir",
            str(prior_log_dir),
        ],
    )
    return {
        job_id: {
            key: record[key]
            for key in (
                "job_name",
                "state",
                "exit_code",
                "account",
                "qos",
                "partition",
                "requested_tres",
                "requested_cpus",
                "requested_memory",
                "time_limit",
                "submit_line_sha256",
            )
        }
        for job_id, record in sorted(records.items())
    }


def _require_recovery_outputs_absent(study_root: Path) -> list[str]:
    paths = [
        study_root / "vpm_parameter_matched_video" / "frontier_quality",
        study_root / "j1_joint_auxiliary_leads" / "frontier_quality",
        study_root / "frontier_selection.json",
        study_root / "frontier_continuation.json",
        study_root / "frontier_lockbox_confirmation.json",
        study_root / "frontier_latency",
        study_root / "frontier_final_report.json",
    ]
    existing = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise WorkflowError(
            f"recovery refuses pre-existing validation/selection outputs: {existing}"
        )
    return [str(path) for path in paths]


def command_validate_completed_recovery(args: argparse.Namespace) -> int:
    """Validate one failed frontier DAG and its completed immutable cache."""

    study_root, study = _load_study(
        args.study_root, training_commit=args.training_commit
    )
    controller_repo = _canonical_directory(
        args.controller_repo_root, "recovery controller repository"
    )
    scientific_repo = _canonical_directory(
        args.scientific_repo_root, "scientific evaluator repository"
    )
    for repo, commit, label in (
        (controller_repo, args.controller_commit, "recovery controller"),
        (scientific_repo, args.scientific_commit, "scientific evaluator"),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise WorkflowError(f"{label} commit is invalid")
        if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
            raise WorkflowError(f"{label} is not a worktree root")
        if _git(repo, "rev-parse", "HEAD") != commit:
            raise WorkflowError(f"{label} worktree HEAD changed")
        if _git(repo, "status", "--porcelain", "--untracked-files=all"):
            raise WorkflowError(f"{label} worktree is dirty")
    if controller_repo == scientific_repo:
        raise WorkflowError("controller and scientific evaluator must be distinct")
    descendant = subprocess.run(
        [
            "git",
            "-C",
            str(controller_repo),
            "merge-base",
            "--is-ancestor",
            args.scientific_commit,
            args.controller_commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if descendant.returncode:
        raise WorkflowError(
            "recovery controller is not a scientific-evaluator descendant"
        )
    controller_changed_paths = {
        value
        for value in _git(
            controller_repo,
            "diff",
            "--name-only",
            args.scientific_commit,
            args.controller_commit,
            "--",
        ).splitlines()
        if value
    }
    unexpected_controller_changes = sorted(
        controller_changed_paths - COMPLETED_RECOVERY_CONTROLLER_ALLOWED_PATHS
    )
    if unexpected_controller_changes:
        raise WorkflowError(
            "recovery controller changes non-allowlisted paths: "
            f"{unexpected_controller_changes}"
        )
    scientific_code_objects: dict[str, str] = {}
    for path in ADOPTED_CACHE_CODE_PATHS:
        recorded_object = _git(
            scientific_repo, "rev-parse", f"{args.scientific_commit}:{path}"
        )
        live_object = _git(scientific_repo, "rev-parse", f"HEAD:{path}")
        if live_object != recorded_object:
            raise WorkflowError(
                f"frozen scientific evaluator object changed: {path}"
            )
        scientific_code_objects[path] = recorded_object
    selection_diff = subprocess.run(
        [
            "git",
            "-C",
            str(controller_repo),
            "diff",
            "--quiet",
            args.scientific_commit,
            args.controller_commit,
            "--",
            "tools/vjepa2_nfe_frontier.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if selection_diff.returncode:
        raise WorkflowError(
            "controller and frozen scientific selection implementations differ"
        )

    prior_path = _canonical_file(
        args.prior_submission, "frontier predecessor submission"
    )
    (
        submission,
        prior_controller,
        prior_log_dir,
        prior_controller_commit,
    ) = _validate_recovery_submission(
        path=prior_path,
        study_root=study_root,
        training_commit=args.training_commit,
        cache_job_id=args.cache_job_id,
        failed_gate_job_id=args.failed_gate_job_id,
        cancelled_vpm_job_id=args.cancelled_vpm_job_id,
        cancelled_j1_job_id=args.cancelled_j1_job_id,
        cancelled_selection_job_id=args.cancelled_selection_job_id,
        final_job_id=args.final_job_id,
        scientific_repo=scientific_repo,
        scientific_commit=args.scientific_commit,
    )
    if (
        _git(prior_controller, "rev-parse", "--show-toplevel")
        != str(prior_controller)
        or _git(prior_controller, "rev-parse", "HEAD")
        != prior_controller_commit
        or _git(prior_controller, "status", "--porcelain", "--untracked-files=all")
    ):
        raise WorkflowError("predecessor controller worktree changed")

    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            "--jobs",
            ",".join(
                (
                    args.cache_job_id,
                    args.failed_gate_job_id,
                    args.cancelled_vpm_job_id,
                    args.cancelled_j1_job_id,
                    args.cancelled_selection_job_id,
                )
            ),
            "--format="
            "JobID%64,JobName%64,User%64,State%32,ExitCode,Account%64,QOS%64,"
            "Partition%64,ReqTRES%512,ReqCPUS,ReqMem,Timelimit,"
            "SubmitLine%4096",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if accounting.returncode:
        raise WorkflowError(
            "could not query predecessor Slurm accounting: "
            f"{accounting.stderr.strip() or accounting.stdout.strip()}"
        )
    jobs = validate_completed_recovery_rows(
        args,
        accounting.stdout.splitlines(),
        study=study,
        submission=submission,
        prior_controller=prior_controller,
        prior_controller_commit=prior_controller_commit,
        prior_log_dir=prior_log_dir,
        scientific_repo=scientific_repo,
    )

    failed_gate_stderr = _canonical_file(
        prior_log_dir
        / f"vjepa2-frontier-u1000-gate-{args.failed_gate_job_id}.err",
        "failed frontier gate stderr",
    )
    expected_error = (
        "ERROR: VPM final evidence is invalid: "
        "VPM final provenance/snapshot identity differs\n"
    )
    if failed_gate_stderr.read_text(encoding="utf-8") != expected_error:
        raise WorkflowError("failed frontier gate stderr differs")

    registration_path = _canonical_file(
        study_root / "frontier_lockbox" / "registration.json",
        "completed lockbox registration",
    )
    registration = _read_json(registration_path, "completed lockbox registration")
    if registration.get("registration_git_commit") != args.scientific_commit:
        raise WorkflowError("lockbox registration commit differs")
    try:
        cache_validation = frontier.lockbox.validate_registration(
            registration,
            study=study,
            rehash_arrays=True,
            verify_construction=True,
        )
    except (frontier.lockbox.LockboxError, KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            f"completed lockbox registration is invalid: {exc}"
        ) from exc
    absent_outputs = _require_recovery_outputs_absent(study_root)

    evidence = {
        "status": "completed_frontier_recovery_validated",
        "study_identity_sha256": study["identity_sha256"],
        "training_git_commit": args.training_commit,
        "controller_repo_root": str(controller_repo),
        "controller_git_commit": args.controller_commit,
        "scientific_evaluator_repo_root": str(scientific_repo),
        "scientific_evaluator_git_commit": args.scientific_commit,
        "repository_evidence": {
            "controller_descends_from_scientific_commit": True,
            "controller_changed_paths": sorted(controller_changed_paths),
            "controller_changes_allowlisted": True,
            "selection_implementation_identical": True,
            "frozen_scientific_code_objects": scientific_code_objects,
        },
        "predecessor_submission": _file_evidence(prior_path),
        "predecessor_jobs": jobs,
        "failed_gate_stderr": _file_evidence(failed_gate_stderr),
        "completed_cache_reused": True,
        "cache_job_id": args.cache_job_id,
        "lockbox_registration": _file_evidence(
            registration_path,
            identity=str(registration["identity_sha256"]),
        ),
        "lockbox_validation": cache_validation,
        "full_cache_hashes_rechecked": (
            cache_validation.get("full_array_hashes_rechecked") is True
        ),
        "scientific_outputs_absent": absent_outputs,
        "new_cache_submission_allowed": False,
    }
    if evidence["full_cache_hashes_rechecked"] is not True:
        raise WorkflowError("completed lockbox cache was not fully rehashed")
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

    recovery = commands.add_parser("validate-completed-recovery")
    recovery.add_argument("--study-root", required=True)
    recovery.add_argument("--training-commit", required=True)
    recovery.add_argument("--controller-repo-root", required=True)
    recovery.add_argument("--controller-commit", required=True)
    recovery.add_argument("--scientific-repo-root", required=True)
    recovery.add_argument("--scientific-commit", required=True)
    recovery.add_argument("--prior-submission", required=True)
    recovery.add_argument("--cache-job-id", required=True)
    recovery.add_argument("--failed-gate-job-id", required=True)
    recovery.add_argument("--cancelled-vpm-job-id", required=True)
    recovery.add_argument("--cancelled-j1-job-id", required=True)
    recovery.add_argument("--cancelled-selection-job-id", required=True)
    recovery.add_argument("--final-job-id", required=True)
    recovery.add_argument("--partition", required=True)
    recovery.add_argument("--account", required=True)
    recovery.add_argument("--qos", required=True)
    recovery.set_defaults(handler=command_validate_completed_recovery)

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
