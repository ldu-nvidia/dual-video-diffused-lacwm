#!/usr/bin/env python3
"""Fail-closed Slurm planning and explicitly authorized submission for ACD-P0.

Planning is the default and performs no writes or process launches.  Submission
requires both ``--submit`` and a fresh, identity-sealed, single-job user
authorization.  Jobs are submitted held; an immutable submission receipt is
written before ``scontrol release``.  The batch entrypoints validate that
receipt again, which makes a direct invocation of ``sbatch`` fail closed.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_adjacent_consistency as analyzer  # noqa: E402
from tools import low_nfe_direct_baseline as pilot  # noqa: E402


SCHEMA_VERSION = 1
AUTHORIZATION_KIND = "acd_p0_user_slurm_authorization"
SUBMISSION_KIND = "acd_p0_slurm_submission_receipt"
ARM_PLAN_KIND = "acd_p0_arm_execution_plan"
PHASES = ("memory_smoke", "train", "evaluate")
MAX_AUTHORIZATION_WINDOW = timedelta(hours=24)
FUTURE_CLOCK_SKEW = timedelta(minutes=5)
APPROVED_ARTIFACT_ROOTS = pilot.APPROVED_ARTIFACT_ROOTS
JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")


class ACDWorkflowError(RuntimeError):
    """A source, registration, authorization, or submission boundary changed."""


@dataclass(frozen=True)
class SubmissionSpec:
    """One immutable Slurm submission rendered by the workflow."""

    phase: str
    arm: str | None
    source_commit: str
    registration_identity_sha256: str | None
    script: Path
    job_name: str
    log_dir: Path
    authorization: Path
    submission_receipt: Path
    wrapper_argv: tuple[str, ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ACDWorkflowError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ACDWorkflowError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ACDWorkflowError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _approved_artifact_path(path: Path, label: str) -> Path:
    if ".." in path.parts or not path.is_absolute() or not any(
        path == root or root in path.parents for root in APPROVED_ARTIFACT_ROOTS
    ):
        roots = ", ".join(str(root) for root in APPROVED_ARTIFACT_ROOTS)
        raise ACDWorkflowError(
            f"{label} must be under an approved artifact root: {roots}"
        )
    return path


def _existing_file(
    path: Path,
    label: str,
    *,
    executable: bool = False,
    allow_symlink: bool = False,
) -> Path:
    path = path.expanduser()
    if (
        not path.is_absolute()
        or (path.is_symlink() and not allow_symlink)
        or not path.is_file()
    ):
        raise ACDWorkflowError(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path and not allow_symlink:
        raise ACDWorkflowError(f"{label} must be canonical")
    if executable and not path.stat().st_mode & 0o111:
        raise ACDWorkflowError(f"{label} must be executable")
    return path


def _existing_directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ACDWorkflowError(f"{label} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ACDWorkflowError(f"{label} must be canonical")
    return path


def _fresh_path(path: Path, label: str, *, expected_parent: Path | None = None) -> Path:
    path = _approved_artifact_path(path.expanduser(), label)
    if path.exists() or path.is_symlink():
        raise ACDWorkflowError(f"fresh {label} already exists: {path}")
    if expected_parent is None:
        ancestor = path.parent
        while not ancestor.exists():
            if ancestor == ancestor.parent:
                raise ACDWorkflowError(f"{label} has no existing parent")
            ancestor = ancestor.parent
        _existing_directory(ancestor, f"{label} existing ancestor")
        if path.resolve(strict=False) != path:
            raise ACDWorkflowError(f"{label} must not traverse a symlink")
        return path
    else:
        parent = expected_parent
        if path.parent != parent:
            raise ACDWorkflowError(f"{label} must be directly under {parent}")
    canonical = parent / path.name
    if canonical != path:
        raise ACDWorkflowError(f"{label} must be canonical")
    return path


def _log_target(path: Path) -> Path:
    path = _approved_artifact_path(path.expanduser(), "Slurm log directory")
    if path.is_symlink():
        raise ACDWorkflowError("Slurm log directory cannot be a symlink")
    if path.exists():
        return _existing_directory(path, "Slurm log directory")
    parent = _existing_directory(path.parent, "Slurm log parent")
    canonical = parent / path.name
    if canonical != path:
        raise ACDWorkflowError("Slurm log directory must be canonical")
    return path


def _registration(path: Path) -> dict[str, Any]:
    path = _approved_artifact_path(
        _existing_file(path, "registration"), "registration receipt"
    )
    registration = pilot.validate_registration(path)
    output_root = _existing_directory(
        Path(registration["output_root"]), "registered output root"
    )
    _approved_artifact_path(output_root, "registered output root")
    repo = _existing_directory(
        Path(registration["tool_repository"]["path"]),
        "registered source repository",
    )
    if repo != ROOT.resolve(strict=True):
        raise ACDWorkflowError("workflow is not running from the registered repository")
    observed = pilot.clean_source(
        repo, str(registration["tool_repository"]["git_commit"])
    )
    if observed != registration["tool_repository"]:
        raise ACDWorkflowError("registered source record differs from live source")
    python = _existing_file(
        Path(registration["runtime"]["python"]),
        "registered Python",
        executable=True,
        allow_symlink=True,
    )
    if str(python) != registration["runtime"]["python"]:
        raise ACDWorkflowError("registered Python path changed")
    return registration


def _arm(code: str) -> pilot.Arm:
    try:
        return pilot.ARM_BY_CODE[code]
    except KeyError as exc:
        raise ACDWorkflowError(f"unsupported arm: {code}") from exc


def arm_values(
    registration: Mapping[str, Any], arm: pilot.Arm
) -> dict[str, Any]:
    root = Path(str(registration["output_root"]))
    run_dir = root / "training" / arm.run_name
    return {
        "arm_code": arm.code,
        "arm_mode": arm.arm_mode,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "run_identity_sha256": registration["arm_run_identity_sha256"][arm.code],
        "arm_config_semantic_sha256": registration["arm_resolved_configs"][arm.code][
            "semantic_sha256"
        ],
        "registered_arm_config": registration["arm_resolved_configs"][arm.code][
            "yaml"
        ]["path"],
        "registration_identity_sha256": registration["identity_sha256"],
        "source_commit": registration["tool_repository"]["git_commit"],
        "output_root": str(root),
        "run_dir": str(run_dir),
        "arm_plan": str(root / "arm_plans" / f"{arm.code.lower()}.json"),
        "slurm_log_dir": str(root / "_slurm_logs"),
        "parent_snapshot": registration["parent"]["snapshot"]["path"],
        "parent_resolved_config": registration["parent"]["resolved_config"]["path"],
        "train_manifest": registration["training"]["manifest"]["path"],
        "train_cache_metadata": registration["training"]["cache_metadata"]["path"],
        "python": registration["runtime"]["python"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox_home"],
    }


def evaluation_values(registration: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(registration["output_root"]))
    return {
        "registration_identity_sha256": registration["identity_sha256"],
        "source_commit": registration["tool_repository"]["git_commit"],
        "output_root": str(root),
        "evaluation_dir": str(root / "evaluation"),
        "analysis_json": str(root / "analysis" / "analysis.json"),
        "analysis_markdown": str(root / "analysis" / "analysis.md"),
        "slurm_log_dir": str(root / "_slurm_logs"),
        "python": registration["runtime"]["python"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox_home"],
    }


def _format_values(values: Mapping[str, Any], order: Sequence[str], format_: str) -> None:
    if format_ == "json":
        print(json.dumps(dict(values), sort_keys=True))
        return
    fields = [str(values[key]) for key in order]
    if any("\t" in field or "\n" in field for field in fields):
        raise ACDWorkflowError("TSV value contains a delimiter")
    print("\t".join(fields))


def command_arm_values(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    values = arm_values(registration, _arm(args.arm))
    if args.require_training_fresh:
        _fresh_path(Path(values["run_dir"]), "training output")
    _format_values(
        values,
        (
            "arm_code",
            "arm_mode",
            "config_name",
            "run_name",
            "run_identity_sha256",
            "arm_config_semantic_sha256",
            "registered_arm_config",
            "registration_identity_sha256",
            "source_commit",
            "output_root",
            "run_dir",
            "arm_plan",
            "slurm_log_dir",
            "parent_snapshot",
            "parent_resolved_config",
            "train_manifest",
            "train_cache_metadata",
            "python",
            "wan_dir",
            "videox_home",
        ),
        args.format,
    )
    return 0


def command_evaluation_values(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    values = evaluation_values(registration)
    if args.require_evaluation_fresh:
        _fresh_path(Path(values["evaluation_dir"]), "evaluation output")
    _format_values(
        values,
        (
            "registration_identity_sha256",
            "source_commit",
            "output_root",
            "evaluation_dir",
            "analysis_json",
            "analysis_markdown",
            "slurm_log_dir",
            "python",
            "wan_dir",
            "videox_home",
        ),
        args.format,
    )
    return 0


def command_write_arm_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    arm = _arm(args.arm)
    values = arm_values(registration, arm)
    _fresh_path(Path(values["run_dir"]), "training output")
    plan_path = Path(values["arm_plan"])
    if plan_path.exists() or plan_path.is_symlink():
        raise ACDWorkflowError("arm execution plan already exists")
    plan_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = pilot.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": ARM_PLAN_KIND,
            "registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["tool_repository"]["git_commit"],
            "arm": asdict(arm),
            "run_identity_sha256": values["run_identity_sha256"],
            "paths": {
                key: values[key]
                for key in ("output_root", "run_dir", "slurm_log_dir")
            },
            "training": {
                "updates": 400,
                "world_size": 8,
                "local_batch_size": 1,
                "global_batch_size": 8,
                "teacher_calls_per_update": 1,
                "online_calls_per_update": 1,
                "ema_target_calls_per_update": 1,
                "endpoint_data_reachable": False,
            },
            "evaluation": {
                "world_size": 8,
                "nfe_grid": list(pilot.NFE_GRID),
                "noise_seeds": list(pilot.NOISE_SEEDS),
                "endpoints": [asdict(endpoint) for endpoint in pilot.ENDPOINTS],
                "global_endpoint_barrier_before_full_rgb": True,
            },
            "resource_plan": registration["resource_plan"],
            "wandb": {"enabled": False, "writes": 0},
            "jobs_submitted_by_plan_writer": 0,
            "protected_test_accessed": False,
        }
    )
    pilot.exclusive_json(plan_path, payload)
    print(str(plan_path))
    return 0


def validate_user_authorization_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    arm: str | None,
    source_commit: str,
    registration_identity_sha256: str | None,
    submission_receipt: Path,
    at_time: datetime | None = None,
) -> dict[str, Any]:
    """Validate an exact, short-lived, single-job user authorization payload."""

    required_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "expires_at_utc",
        "authorized_by",
        "authorization_basis",
        "slurm_submission_authorized",
        "phase",
        "arm",
        "source_commit",
        "registration_identity_sha256",
        "submission_receipt_path",
        "accepted_parent_not_quality_dominant",
        "accepted_resource_plan",
        "jobs_authorized",
        "wandb_writes_authorized",
        "protected_test_accessed",
        "identity_sha256",
    }
    if set(payload) != required_keys or not pilot.identity_valid(payload):
        raise ACDWorkflowError("user authorization schema/identity differs")
    basis = payload.get("authorization_basis")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != AUTHORIZATION_KIND
        or payload.get("authorized_by") != "user"
        or not isinstance(basis, str)
        or not basis.strip()
        or len(basis) > 2_000
        or payload.get("slurm_submission_authorized") is not True
        or payload.get("phase") != phase
        or payload.get("arm") != arm
        or payload.get("source_commit") != source_commit
        or payload.get("registration_identity_sha256")
        != registration_identity_sha256
        or payload.get("submission_receipt_path") != str(submission_receipt)
        or payload.get("accepted_parent_not_quality_dominant") is not True
        or payload.get("accepted_resource_plan") != pilot.resource_plan()
        or payload.get("jobs_authorized") != 1
        or payload.get("wandb_writes_authorized") != 0
        or payload.get("protected_test_accessed") is not False
    ):
        raise ACDWorkflowError("user authorization does not match this job")
    created = _parse_utc(payload.get("created_at_utc"), "authorization creation")
    expires = _parse_utc(payload.get("expires_at_utc"), "authorization expiry")
    instant = (at_time or _utc_now()).astimezone(timezone.utc)
    if (
        expires <= created
        or expires - created > MAX_AUTHORIZATION_WINDOW
        or created > instant + FUTURE_CLOCK_SKEW
        or instant >= expires
    ):
        raise ACDWorkflowError("user authorization is not fresh")
    return dict(payload)


def validate_user_authorization(
    path: Path,
    *,
    phase: str,
    arm: str | None,
    source_commit: str,
    registration_identity_sha256: str | None,
    submission_receipt: Path,
    at_time: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _approved_artifact_path(
        _existing_file(path, "user authorization"), "user authorization"
    )
    if path.stat().st_mode & 0o222:
        raise ACDWorkflowError("user authorization must be read-only")
    payload = pilot.read_json(path, "ACD-P0 user authorization")
    validated = validate_user_authorization_payload(
        payload,
        phase=phase,
        arm=arm,
        source_commit=source_commit,
        registration_identity_sha256=registration_identity_sha256,
        submission_receipt=submission_receipt,
        at_time=at_time,
    )
    return validated, pilot.file_record(path)


def _submission_receipt_name(phase: str, arm: str | None) -> str:
    suffix = phase if arm is None else f"{phase}-{arm.lower()}"
    return f"acd-p0-{suffix.replace('_', '-')}-submission.json"


def _make_spec(
    *,
    phase: str,
    arm: str | None,
    source_commit: str,
    registration_identity_sha256: str | None,
    script: Path,
    job_name: str,
    log_dir: Path,
    authorization: Path,
    wrapper_argv: Sequence[str],
) -> SubmissionSpec:
    if phase not in PHASES:
        raise ACDWorkflowError(f"unsupported phase: {phase}")
    log_dir = _log_target(log_dir)
    receipt = log_dir / _submission_receipt_name(phase, arm)
    if receipt.exists() or receipt.is_symlink():
        raise ACDWorkflowError(f"fresh submission receipt already exists: {receipt}")
    return SubmissionSpec(
        phase=phase,
        arm=arm,
        source_commit=source_commit,
        registration_identity_sha256=registration_identity_sha256,
        script=_existing_file(script, "Slurm wrapper"),
        job_name=job_name,
        log_dir=log_dir,
        authorization=authorization,
        submission_receipt=receipt,
        wrapper_argv=tuple(wrapper_argv),
    )


def sbatch_argv(spec: SubmissionSpec, *, executable: str = "sbatch") -> list[str]:
    return [
        executable,
        "--hold",
        "--parsable",
        f"--job-name={spec.job_name}",
        f"--chdir={spec.log_dir}",
        f"--output={spec.log_dir}/%x-%j.out",
        f"--error={spec.log_dir}/%x-%j.err",
        str(spec.script),
        *spec.wrapper_argv,
        "--user-authorization",
        str(spec.authorization),
        "--submission-receipt",
        str(spec.submission_receipt),
        "--log-dir",
        str(spec.log_dir),
    ]


def _plan_payload(
    spec: SubmissionSpec,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    command = sbatch_argv(spec)
    return {
        "kind": "acd_p0_slurm_plan",
        "mode": "plan_only_no_commands_executed_no_files_created",
        "phase": spec.phase,
        "arm": spec.arm,
        "source_commit": spec.source_commit,
        "registration_identity_sha256": spec.registration_identity_sha256,
        "authorization_identity_sha256": authorization["identity_sha256"],
        "authorization_expires_at_utc": authorization["expires_at_utc"],
        "submission_receipt_path": str(spec.submission_receipt),
        "slurm_log_dir": str(spec.log_dir),
        "sbatch_argv": command,
        "sbatch_shell": shlex.join(command),
        "held_until_submission_receipt_exists": True,
        "requires_submit_flag": True,
        "jobs_submitted": 0,
    }


def _submit_held(
    spec: SubmissionSpec,
    authorization_payload: Mapping[str, Any],
    authorization_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Submit one held job, seal its receipt, then release it."""

    sbatch = shutil.which("sbatch")
    scontrol = shutil.which("scontrol")
    squeue = shutil.which("squeue")
    if sbatch is None or scontrol is None or squeue is None:
        raise ACDWorkflowError("sbatch, scontrol, and squeue are required for --submit")
    if spec.submission_receipt.exists() or spec.submission_receipt.is_symlink():
        raise ACDWorkflowError("submission receipt is no longer fresh")
    active = subprocess.run(
        [squeue, "--me", "--noheader", "--format=%i|%j|%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if active.returncode:
        raise ACDWorkflowError(active.stderr.strip() or "active-job check failed")
    same_name = [
        line
        for line in active.stdout.splitlines()
        if len(line.split("|", 2)) == 3
        and line.split("|", 2)[1] == spec.job_name
    ]
    if same_name:
        raise ACDWorkflowError(
            f"an active job already has the exact name {spec.job_name}: {same_name}"
        )
    spec.log_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    command = sbatch_argv(spec, executable=sbatch)
    submitted = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if submitted.returncode:
        raise ACDWorkflowError(
            submitted.stderr.strip() or "held sbatch submission failed"
        )
    response = submitted.stdout.strip()
    job_id = response.split(";", 1)[0]
    if JOB_ID_RE.fullmatch(job_id) is None:
        raise ACDWorkflowError(f"sbatch returned an invalid job identifier: {response}")
    receipt = pilot.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": SUBMISSION_KIND,
            "created_at_utc": pilot.now(),
            "phase": spec.phase,
            "arm": spec.arm,
            "source_commit": spec.source_commit,
            "registration_identity_sha256": spec.registration_identity_sha256,
            "authorization": dict(authorization_record),
            "authorization_identity_sha256": authorization_payload[
                "identity_sha256"
            ],
            "submission_receipt_path": str(spec.submission_receipt),
            "script": pilot.file_record(spec.script),
            "slurm_log_dir": str(spec.log_dir),
            "sbatch_argv": command,
            "slurm_job_id": job_id,
            "active_job_check": {
                "command": "squeue --me --noheader --format=%i|%j|%T",
                "same_job_name_before_submission": [],
            },
            "submitted_held": True,
            "release_requested_only_after_receipt": True,
            "jobs_submitted": 1,
        }
    )
    pilot.exclusive_json(spec.submission_receipt, receipt)
    spec.submission_receipt.chmod(0o400)
    released = subprocess.run(
        [scontrol, "release", job_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if released.returncode:
        raise ACDWorkflowError(
            "submission receipt was sealed but the job remains held: "
            + (released.stderr.strip() or job_id)
        )
    return {
        "kind": "acd_p0_slurm_submission_result",
        "phase": spec.phase,
        "arm": spec.arm,
        "slurm_job_id": job_id,
        "submission_receipt": str(spec.submission_receipt),
        "submitted_held": True,
        "released_after_receipt": True,
        "jobs_submitted": 1,
    }


def execute_or_plan(spec: SubmissionSpec, *, submit: bool) -> dict[str, Any]:
    """Validate authorization and either render or explicitly submit one job."""

    authorization, record = validate_user_authorization(
        spec.authorization,
        phase=spec.phase,
        arm=spec.arm,
        source_commit=spec.source_commit,
        registration_identity_sha256=spec.registration_identity_sha256,
        submission_receipt=spec.submission_receipt,
    )
    if not submit:
        return _plan_payload(spec, authorization)
    return _submit_held(spec, authorization, record)


def _memory_spec(args: argparse.Namespace) -> SubmissionSpec:
    repo = _existing_directory(args.source_repo, "source repository")
    if repo != ROOT.resolve(strict=True):
        raise ACDWorkflowError("memory smoke source must be this repository")
    pilot.clean_source(repo, args.expected_commit)
    if pilot.COMMIT_RE.fullmatch(args.expected_commit) is None:
        raise ACDWorkflowError("expected source commit is invalid")
    python = _existing_file(
        args.python, "Python", executable=True, allow_symlink=True
    )
    parent_snapshot = _existing_file(args.parent_snapshot, "parent snapshot")
    parent_config = _existing_file(
        args.parent_resolved_config, "parent resolved config"
    )
    wan_dir = _existing_directory(args.wan_dir, "Wan directory")
    videox_home = _existing_directory(args.videox_home, "VideoX directory")
    output = _fresh_path(args.output, "memory-smoke output")
    log_dir = _log_target(output.parent / "_slurm_logs")
    wrapper = (
        "--source-repo",
        str(repo),
        "--expected-commit",
        args.expected_commit,
        "--python",
        str(python),
        "--parent-snapshot",
        str(parent_snapshot),
        "--parent-resolved-config",
        str(parent_config),
        "--wan-dir",
        str(wan_dir),
        "--videox-home",
        str(videox_home),
        "--output",
        str(output),
    )
    return _make_spec(
        phase="memory_smoke",
        arm=None,
        source_commit=args.expected_commit,
        registration_identity_sha256=None,
        script=ROOT / "tools/slurm/adjacent_consistency_memory_smoke.sbatch",
        job_name=f"acd-p0-memory-{args.expected_commit[:10]}",
        log_dir=log_dir,
        authorization=args.user_authorization,
        wrapper_argv=wrapper,
    )


def _train_spec(args: argparse.Namespace) -> SubmissionSpec:
    registration = _registration(args.registration)
    arm = _arm(args.arm)
    values = arm_values(registration, arm)
    _fresh_path(Path(values["run_dir"]), "training output")
    wrapper = (
        "--registration",
        str(args.registration),
        "--arm",
        arm.code,
        "--repo-root",
        str(ROOT),
        "--python",
        values["python"],
        "--expected-commit",
        values["source_commit"],
    )
    return _make_spec(
        phase="train",
        arm=arm.code,
        source_commit=values["source_commit"],
        registration_identity_sha256=values["registration_identity_sha256"],
        script=ROOT / "tools/slurm/adjacent_consistency.sbatch",
        job_name=f"acd-p0-{arm.code.lower()}",
        log_dir=Path(values["slurm_log_dir"]),
        authorization=args.user_authorization,
        wrapper_argv=wrapper,
    )


def _require_training_outputs(registration: Mapping[str, Any]) -> None:
    for arm in pilot.ARMS:
        run_dir = Path(arm_values(registration, arm)["run_dir"])
        for name in (
            "snapshot.pt",
            "training_complete.json",
            "acd_training_trace.jsonl",
            "acd_training_trace_complete.json",
            ".hydra/config.yaml",
        ):
            path = run_dir / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise ACDWorkflowError(f"training prerequisite is missing: {path}")


def _evaluation_spec(args: argparse.Namespace) -> SubmissionSpec:
    registration = _registration(args.registration)
    _require_training_outputs(registration)
    values = evaluation_values(registration)
    _fresh_path(Path(values["evaluation_dir"]), "evaluation output")
    wrapper = (
        "--registration",
        str(args.registration),
        "--repo-root",
        str(ROOT),
        "--python",
        values["python"],
        "--expected-commit",
        values["source_commit"],
    )
    return _make_spec(
        phase="evaluate",
        arm=None,
        source_commit=values["source_commit"],
        registration_identity_sha256=values["registration_identity_sha256"],
        script=ROOT / "tools/slurm/adjacent_consistency_evaluate.sbatch",
        job_name="acd-p0-evaluate",
        log_dir=Path(values["slurm_log_dir"]),
        authorization=args.user_authorization,
        wrapper_argv=wrapper,
    )


def _emit_phase(spec: SubmissionSpec, submit: bool) -> int:
    print(json.dumps(execute_or_plan(spec, submit=submit), indent=2, sort_keys=True))
    return 0


def command_memory_smoke(args: argparse.Namespace) -> int:
    return _emit_phase(_memory_spec(args), args.submit)


def command_train(args: argparse.Namespace) -> int:
    return _emit_phase(_train_spec(args), args.submit)


def command_evaluate(args: argparse.Namespace) -> int:
    return _emit_phase(_evaluation_spec(args), args.submit)


def _expected_submission_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str, str | None, str | None, Path]:
    registration: dict[str, Any] | None = None
    if args.phase == "memory_smoke":
        if args.registration is not None or args.arm is not None:
            raise ACDWorkflowError("memory smoke forbids registration and arm")
        if pilot.COMMIT_RE.fullmatch(args.source_commit) is None:
            raise ACDWorkflowError("memory smoke source commit is invalid")
        pilot.clean_source(ROOT, args.source_commit)
        registration_identity = None
        expected_script = ROOT / "tools/slurm/adjacent_consistency_memory_smoke.sbatch"
    else:
        if args.registration is None:
            raise ACDWorkflowError("train/evaluate submission requires registration")
        registration = _registration(args.registration)
        registered_commit = registration["tool_repository"]["git_commit"]
        if args.source_commit != registered_commit:
            raise ACDWorkflowError("submission source differs from registration")
        registration_identity = registration["identity_sha256"]
        if args.phase == "train":
            if args.arm is None:
                raise ACDWorkflowError("training submission requires an arm")
            _arm(args.arm)
            expected_script = ROOT / "tools/slurm/adjacent_consistency.sbatch"
        else:
            if args.arm is not None:
                raise ACDWorkflowError("evaluation submission forbids an arm")
            expected_script = ROOT / "tools/slurm/adjacent_consistency_evaluate.sbatch"
    return (
        registration,
        args.source_commit,
        registration_identity,
        args.arm,
        expected_script,
    )


def command_validate_submission(args: argparse.Namespace) -> int:
    (
        _registered,
        source_commit,
        registration_identity,
        arm,
        expected_script,
    ) = _expected_submission_context(args)
    receipt_path = _approved_artifact_path(
        _existing_file(args.submission_receipt, "submission receipt"),
        "submission receipt",
    )
    if receipt_path.stat().st_mode & 0o222:
        raise ACDWorkflowError("submission receipt must be read-only")
    receipt = pilot.read_json(receipt_path, "ACD-P0 submission receipt")
    if not pilot.identity_valid(receipt):
        raise ACDWorkflowError("submission receipt identity differs")
    required = {
        "schema_version",
        "kind",
        "created_at_utc",
        "phase",
        "arm",
        "source_commit",
        "registration_identity_sha256",
        "authorization",
        "authorization_identity_sha256",
        "submission_receipt_path",
        "script",
        "slurm_log_dir",
        "sbatch_argv",
        "slurm_job_id",
        "active_job_check",
        "submitted_held",
        "release_requested_only_after_receipt",
        "jobs_submitted",
        "identity_sha256",
    }
    if set(receipt) != required:
        raise ACDWorkflowError("submission receipt schema differs")
    submitted_at = _parse_utc(receipt.get("created_at_utc"), "submission creation")
    authorization, authorization_record = validate_user_authorization(
        args.user_authorization,
        phase=args.phase,
        arm=arm,
        source_commit=source_commit,
        registration_identity_sha256=registration_identity,
        submission_receipt=receipt_path,
        at_time=submitted_at,
    )
    submitted_argv = receipt.get("sbatch_argv")
    expected_log_prefixes = {
        f"--chdir={receipt.get('slurm_log_dir')}",
        f"--output={receipt.get('slurm_log_dir')}/%x-%j.out",
        f"--error={receipt.get('slurm_log_dir')}/%x-%j.err",
    }
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("kind") != SUBMISSION_KIND
        or receipt.get("phase") != args.phase
        or receipt.get("arm") != arm
        or receipt.get("source_commit") != source_commit
        or receipt.get("registration_identity_sha256") != registration_identity
        or receipt.get("authorization") != authorization_record
        or receipt.get("authorization_identity_sha256")
        != authorization["identity_sha256"]
        or receipt.get("submission_receipt_path") != str(receipt_path)
        or receipt.get("script") != pilot.file_record(expected_script)
        or receipt.get("slurm_job_id") != args.slurm_job_id
        or receipt.get("active_job_check")
        != {
            "command": "squeue --me --noheader --format=%i|%j|%T",
            "same_job_name_before_submission": [],
        }
        or receipt.get("submitted_held") is not True
        or receipt.get("release_requested_only_after_receipt") is not True
        or receipt.get("jobs_submitted") != 1
        or not isinstance(submitted_argv, list)
        or len(submitted_argv) < 8
        or Path(str(submitted_argv[0])).name != "sbatch"
        or submitted_argv[1:3] != ["--hold", "--parsable"]
        or not expected_log_prefixes.issubset(set(submitted_argv))
        or str(expected_script) not in submitted_argv
        or str(args.user_authorization) not in submitted_argv
        or str(receipt_path) not in submitted_argv
    ):
        raise ACDWorkflowError("submission receipt does not authorize this allocation")
    log_dir = _existing_directory(
        Path(str(receipt["slurm_log_dir"])), "submitted Slurm log directory"
    )
    _approved_artifact_path(log_dir, "submitted Slurm log directory")
    if Path.cwd().resolve(strict=True) != log_dir:
        raise ACDWorkflowError("allocation did not start in its registered log directory")
    print(
        json.dumps(
            {
                "status": "AUTHORIZED_SUBMISSION_RECEIPT_VALID",
                "phase": args.phase,
                "arm": arm,
                "slurm_job_id": args.slurm_job_id,
                "authorization_identity_sha256": authorization[
                    "identity_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    root = Path(registration["output_root"])
    payload = {
        "kind": "acd_p0_slurm_workflow_plan",
        "mode": "no_commands_executed_no_files_created",
        "source_commit": registration["tool_repository"]["git_commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "output_root": str(root),
        "output_root_policy": (
            "all scientific and scheduler artifacts under /mnt/data1, "
            "/mnt/data2, or the exact ldu LACWM Lustre root"
        ),
        "phases": [
            {
                "phase": "train",
                "arm": arm.code,
                "script": str(ROOT / "tools/slurm/adjacent_consistency.sbatch"),
                "gpus": 8,
                "gpu_type": "B200",
                "time_limit_hours": 2,
                "submission_receipt": str(
                    root
                    / "_slurm_logs"
                    / _submission_receipt_name("train", arm.code)
                ),
            }
            for arm in pilot.ARMS
        ]
        + [
            {
                "phase": "evaluate",
                "arm": None,
                "script": str(
                    ROOT / "tools/slurm/adjacent_consistency_evaluate.sbatch"
                ),
                "gpus": 8,
                "gpu_type": "B200",
                "time_limit_hours": 2,
                "submission_receipt": str(
                    root
                    / "_slurm_logs"
                    / _submission_receipt_name("evaluate", None)
                ),
            }
        ],
        "memory_smoke": {
            "pre_registration": True,
            "script": str(
                ROOT / "tools/slurm/adjacent_consistency_memory_smoke.sbatch"
            ),
            "gpus": 1,
            "gpu_type": "B200",
            "time_limit_hours": 1,
            "authorization_registration_identity_sha256": None,
        },
        "authorization_contract": {
            "kind": AUTHORIZATION_KIND,
            "maximum_validity_hours": 24,
            "read_only_file_required": True,
            "one_job_per_receipt": True,
            "train_and_evaluate_bind_registration": True,
            "memory_smoke_binds_source_before_registration": True,
            "parent_teacher_limitation_must_be_accepted": True,
        },
        "submission_contract": {
            "default": "plan_only",
            "explicit_flag": "--submit",
            "submitted_held": True,
            "rejects_preexisting_same_name_job": True,
            "release_only_after_immutable_submission_receipt": True,
            "batch_entrypoint_revalidates_receipt": True,
        },
        "resource_plan": pilot.resource_plan(),
        "jobs_submitted": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    values = evaluation_values(registration)
    analysis_dir = Path(values["analysis_json"]).parent
    analysis_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    namespace = argparse.Namespace(
        registration=args.registration,
        evaluation=Path(values["evaluation_dir"]),
        output_json=Path(values["analysis_json"]),
        output_markdown=Path(values["analysis_markdown"]),
    )
    return int(analyzer.command_analyze(namespace))


def _add_submit_guard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user-authorization", type=Path, required=True)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit one held job; default is a read-only plan",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    memory = commands.add_parser("memory-smoke")
    memory.add_argument("--source-repo", type=Path, required=True)
    memory.add_argument("--expected-commit", required=True)
    memory.add_argument("--python", type=Path, required=True)
    memory.add_argument("--parent-snapshot", type=Path, required=True)
    memory.add_argument("--parent-resolved-config", type=Path, required=True)
    memory.add_argument("--wan-dir", type=Path, required=True)
    memory.add_argument("--videox-home", type=Path, required=True)
    memory.add_argument("--output", type=Path, required=True)
    _add_submit_guard(memory)
    memory.set_defaults(func=command_memory_smoke)

    train = commands.add_parser("train")
    train.add_argument("--registration", type=Path, required=True)
    train.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE), required=True)
    _add_submit_guard(train)
    train.set_defaults(func=command_train)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--registration", type=Path, required=True)
    _add_submit_guard(evaluate)
    evaluate.set_defaults(func=command_evaluate)

    values = commands.add_parser("arm-values")
    values.add_argument("--registration", type=Path, required=True)
    values.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE), required=True)
    values.add_argument("--format", choices=("json", "tsv"), default="json")
    values.add_argument("--require-training-fresh", action="store_true")
    values.set_defaults(func=command_arm_values)

    evaluation = commands.add_parser("evaluation-values")
    evaluation.add_argument("--registration", type=Path, required=True)
    evaluation.add_argument("--format", choices=("json", "tsv"), default="json")
    evaluation.add_argument("--require-evaluation-fresh", action="store_true")
    evaluation.set_defaults(func=command_evaluation_values)

    arm_plan = commands.add_parser("write-arm-plan")
    arm_plan.add_argument("--registration", type=Path, required=True)
    arm_plan.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE), required=True)
    arm_plan.set_defaults(func=command_write_arm_plan)

    validate = commands.add_parser("validate-submission")
    validate.add_argument("--phase", choices=PHASES, required=True)
    validate.add_argument("--source-commit", required=True)
    validate.add_argument("--registration", type=Path)
    validate.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE))
    validate.add_argument("--user-authorization", type=Path, required=True)
    validate.add_argument("--submission-receipt", type=Path, required=True)
    validate.add_argument("--slurm-job-id", required=True)
    validate.set_defaults(func=command_validate_submission)

    plan = commands.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.set_defaults(func=command_plan)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
