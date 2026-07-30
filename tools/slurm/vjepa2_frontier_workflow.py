#!/usr/bin/env python3
"""Small fail-closed helpers for the V-JEPA NFE-frontier Slurm workflow.

The GPU entrypoints remain ordinary shell scripts, but JSON/provenance checks
must not depend on ``jq`` or on fragile shell field extraction. This helper
delegates deep evidence validation to the same modules used by the scientific
evaluators.
"""

from __future__ import annotations

import argparse
import json
import re
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


def classify_final_job_rows(job_id: str, rows: Sequence[str]) -> str:
    """Classify exact formatted ``sacct -X`` rows without trusting stale IDs."""

    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise WorkflowError("final job ID must be a positive integer")
    records: dict[int, tuple[str, str, str]] = {}
    for line in rows:
        fields = [field.strip() for field in line.strip().split("|")]
        if len(fields) < 3:
            continue
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
    observed_tasks = set(records)
    if observed_tasks != FINAL_ARRAY_TASK_IDS:
        missing = sorted(FINAL_ARRAY_TASK_IDS - observed_tasks)
        extra = sorted(observed_tasks - FINAL_ARRAY_TASK_IDS)
        raise WorkflowError(
            "final u1000 accounting task set differs: "
            f"missing={missing}, extra={extra}"
        )
    task_records = list(records.values())
    failed = [
        record
        for record in task_records
        if record[1] in FAILED_SLURM_STATES
        or (record[1] == "COMPLETED" and record[2] != "0:0")
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
    if any(record[1] in ACTIVE_SLURM_STATES for record in task_records):
        return "active_afterok"
    if all(
        record[1] == "COMPLETED" and record[2] == "0:0"
        for record in task_records
    ):
        return "terminal_success"
    raise WorkflowError("final u1000 accounting state is inconsistent")


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
    mode = classify_final_job_rows(
        args.final_job_id, completed.stdout.splitlines()
    )
    print(mode)
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
