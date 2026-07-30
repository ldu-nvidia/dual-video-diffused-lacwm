#!/usr/bin/env python3
"""Fail-closed provenance for the one V-JEPA paired-latency recovery.

The original controlled-study job 481133 failed before timing because its
validator rejected the producer's complete ``{path, sha256, bytes}`` records.
That job created an empty ``paired_latency`` directory and the immutable study
submission binds the intended timing allocation to the failed job ID.  Neither
artifact may be edited or removed.

This helper validates that exact incident and creates an external recovery
protocol plus a held-job submission receipt.  It deliberately does not contain
the benchmark itself.  The recovered allocation still executes the original
J1@NFE4 versus VPM@NFE8 same-B200 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vjepa2_nfe_frontier import (  # noqa: E402
    FrontierError,
    git_inference_compatibility,
)


SCHEMA_VERSION = 1
KIND_PROTOCOL = "vjepa2_paired_latency_recovery_protocol"
KIND_SUBMISSION = "vjepa2_paired_latency_recovery_submission"
KIND_RUNTIME = "vjepa2_paired_latency_recovery_runtime"
PAIR_LABEL = "J1_autonomous_nfe4_vs_VPM_autonomous_nfe8"
OUTPUT_BASENAME = "paired_j1_nfe4_vs_vpm_nfe8.json"
STUDY_KIND = "vjepa2_controlled_video_diffusion_study"
STUDY_ID = "vjepa2-controlled-20260730-seed1234-9cf8e69-v3"
TRAINING_COMMIT = "9cf8e6922f35a5d6645e3128545953723bf54da2"
VALIDATOR_FIX_COMMIT = "9ba40731c10a619972877b703a33085724ff84f6"
FAILED_JOB_ID = "481133"
FINAL_STAGE_JOB_ID = "481132"
FAILED_EXIT_CODE = "2:0"
FAILED_ELAPSED = "00:01:08"
FAILED_ERROR = "ERROR: J1 final provenance/snapshot identity differs"
PARTITION = "batch"
ACCOUNT = "coreai_chef_posttrain"
QOS = "normal"
CPUS = 32
MEMORY = "256G"
TIME_LIMIT = "04:00:00"
NODES = 1
GPUS = 1
STUDY_MANIFEST_SHA256 = (
    "6bab0688f644ab03be0943c2a74401ffc835fe8ce20451856ea0f60f12304247"
)
STUDY_MANIFEST_BYTES = 33_387
ORIGINAL_SUBMISSION_SHA256 = (
    "3747b98034395b5453fd7a9fa493f82ad1101571b25ac3af5e4b5b5ee542188d"
)
ORIGINAL_SUBMISSION_BYTES = 1_660
ORIGINAL_SUBMISSION_IDENTITY = (
    "f7aad9848694107a3069725e953b3b591a0730d3003c34053124705c436f1e15"
)
STUDY_IDENTITY = (
    "dc720b8c1d417cb8cbef6f5bd9ab41b35a650c8b64115e15a61e84606b306ac4"
)
FAILED_STDOUT_SHA256 = (
    "254b3a8bfccfd95ae574e8e5c7148df091db141384035b3ed4c453aa4969d083"
)
FAILED_STDOUT_BYTES = 3_762
FAILED_STDERR_SHA256 = (
    "635f922ed2ac43e1d00431168080c37ef95ae429a89c93ef486e459f354b30b2"
)
FAILED_STDERR_BYTES = 948
VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
WAN_ASSET_SPECS = {
    "config.json": {
        "bytes": 249,
        "sha256": (
            "55779e6882d7c0918d8c289d61c4ba4693c1014b53c593f49362dd4f3baead49"
        ),
    },
    "diffusion_pytorch_model.safetensors": {
        "bytes": 3_129_105_448,
        "sha256": (
            "9ff6289322b41bf187206eac2a57e85ce85c9ee5bfe8bc44eabeaeb86b44129a"
        ),
    },
    "Wan2.1_VAE.pth": {
        "bytes": 507_609_880,
        "sha256": (
            "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
        ),
    },
    "null_prompt_umt5.pt": {
        "bytes": 18_045,
        "sha256": (
            "c67a1f04559ac05f8bd475eb7fad68812e1a10e071471514395d25aad3de415d"
        ),
    },
    "null_prompt_umt5.pt.json": {
        "bytes": 1_199,
        "sha256": (
            "be10e37ed98ebff38d9267a75aa5c3f5bac9224ef423d010aa75956a45fc26e7"
        ),
    },
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
RECOVERY_ID_RE = re.compile(
    r"^paired-481133-([0-9a-f]{7,12})-r([1-9][0-9]*)$"
)


class RecoveryError(RuntimeError):
    """Raised when this exact recovery cannot be proven safe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    recorded = payload.get("identity_sha256")
    if (
        not isinstance(recorded, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded) is None
    ):
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == recorded


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_directory(
    value: str | Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RecoveryError(f"{label} must be absolute")
    try:
        info = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            parent = path.parent.resolve(strict=True)
            return parent / path.name
        raise RecoveryError(f"{label} is missing: {path}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RecoveryError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RecoveryError(f"{label} must be absolute")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RecoveryError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise RecoveryError(
            f"{label} must be a non-empty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RecoveryError(f"{label} must be a JSON object: {path}")
    return payload


def file_record(
    path: Path,
    *,
    identity_sha256: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if identity_sha256 is not None:
        result["identity_sha256"] = identity_sha256
    return result


def validate_file_record(
    record: Any,
    path: Path,
    *,
    label: str,
    identity_sha256: str | None = None,
) -> None:
    expected = file_record(path, identity_sha256=identity_sha256)
    if not isinstance(record, Mapping) or dict(record) != expected:
        raise RecoveryError(f"{label} file record differs")


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as exc:
        raise RecoveryError(f"could not exclusively create {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def normalized_job_id(value: Any, label: str) -> str:
    normalized = str(value).split(";", 1)[0].split("_", 1)[0]
    if JOB_ID_RE.fullmatch(normalized) is None:
        raise RecoveryError(f"{label} must be a positive Slurm job ID")
    return normalized


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RecoveryError(
            f"git {' '.join(arguments)} failed for {repo}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def validate_runtime_assets(
    wan_dir: str | Path,
    videox_home: str | Path,
) -> dict[str, Any]:
    """Bind every model asset and the shared VideoX checkout by content."""

    wan = canonical_directory(wan_dir, "Wan directory")
    videox = canonical_directory(videox_home, "VideoX checkout")
    if _git(videox, "rev-parse", "HEAD") != VIDEOX_COMMIT:
        raise RecoveryError("VideoX checkout commit differs")
    if _git(videox, "status", "--porcelain", "--untracked-files=all"):
        raise RecoveryError("VideoX checkout must be clean")
    records: dict[str, dict[str, Any]] = {}
    for name, expected in WAN_ASSET_SPECS.items():
        path = canonical_file(wan / name, f"Wan asset {name}")
        record = file_record(path)
        wanted = {
            "path": str(path),
            "sha256": expected["sha256"],
            "bytes": expected["bytes"],
        }
        if record != wanted:
            raise RecoveryError(f"Wan asset differs: {name}")
        records[name] = record
    return {
        "wan": {
            "root": str(wan),
            "files": records,
            "files_sha256": hashlib.sha256(
                canonical_json(records)
            ).hexdigest(),
        },
        "videox": {
            "root": str(videox),
            "commit": VIDEOX_COMMIT,
            "clean": True,
        },
    }


def validate_worktree(
    root: str | Path,
    commit: str,
    label: str,
) -> Path:
    if COMMIT_RE.fullmatch(commit) is None:
        raise RecoveryError(f"{label} commit must be a full lowercase SHA-1")
    repo = canonical_directory(root, label)
    if Path(_git(repo, "rev-parse", "--show-toplevel")) != repo:
        raise RecoveryError(f"{label} must be a Git worktree root")
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise RecoveryError(f"{label} HEAD differs from the recorded commit")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RecoveryError(f"{label} worktree must be clean")
    return repo


def recovery_paths(
    study_root: Path,
    recovery_root: Path,
) -> dict[str, str]:
    return {
        "protocol": str(recovery_root / "protocol.json"),
        "submission": str(recovery_root / "submission.json"),
        "runtime": str(recovery_root / "runtime.json"),
        "paired_latency": str(
            recovery_root / "paired_latency" / OUTPUT_BASENAME
        ),
        "analysis_json": str(recovery_root / "analysis" / "analysis.json"),
        "analysis_markdown": str(recovery_root / "analysis" / "analysis.md"),
        "legacy_paired_latency_directory": str(
            study_root / "paired_latency"
        ),
        "legacy_analysis_root": str(
            study_root.parent / "_analysis" / STUDY_ID
        ),
    }


def expected_recovery_root(
    study_root: Path,
    recovery_id: str,
    controller_commit: str,
) -> Path:
    match = RECOVERY_ID_RE.fullmatch(recovery_id)
    if match is None or not controller_commit.startswith(match.group(1)):
        raise RecoveryError(
            "recovery ID must bind failed job 481133 and the controller prefix"
        )
    return (
        study_root.parent
        / "_paired_recoveries"
        / STUDY_ID
        / recovery_id
    )


def _validate_study_and_submission(
    study_root: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    if study_root.name != STUDY_ID:
        raise RecoveryError("study root is not the immutable v3 study")
    study_path = canonical_file(
        study_root / "study_manifest.json", "study manifest"
    )
    if (
        study_path.stat().st_size != STUDY_MANIFEST_BYTES
        or sha256(study_path) != STUDY_MANIFEST_SHA256
    ):
        raise RecoveryError("immutable study manifest bytes differ")
    study = read_json(study_path, "study manifest")
    repository = study.get("inputs", {}).get("repository", {})
    if (
        not identity_valid(study)
        or study.get("kind") != STUDY_KIND
        or study.get("study_id") != STUDY_ID
        or study.get("study_root") != str(study_root)
        or study.get("identity_sha256") != STUDY_IDENTITY
        or repository.get("git_commit") != TRAINING_COMMIT
    ):
        raise RecoveryError("immutable study identity/provenance differs")

    submission_path = canonical_file(
        study_root / "slurm_submission.json", "original submission"
    )
    if (
        submission_path.stat().st_size != ORIGINAL_SUBMISSION_BYTES
        or sha256(submission_path) != ORIGINAL_SUBMISSION_SHA256
    ):
        raise RecoveryError("original submission bytes differ")
    submission = read_json(submission_path, "original submission")
    paired = submission.get("paired_latency_job")
    endpoints = submission.get("stage_array_job_ids")
    final = (
        [
            item
            for item in endpoints
            if isinstance(item, Mapping)
            and item.get("completed_updates") == 1000
        ]
        if isinstance(endpoints, list)
        else []
    )
    if (
        not identity_valid(submission)
        or submission.get("kind") != "vjepa2_controlled_study_submission"
        or submission.get("identity_sha256") != ORIGINAL_SUBMISSION_IDENTITY
        or submission.get("study_identity_sha256") != STUDY_IDENTITY
        or submission.get("dependency") != "afterok"
        or not isinstance(paired, Mapping)
        or normalized_job_id(paired.get("job_id"), "original paired job")
        != FAILED_JOB_ID
        or paired.get("dependency") != "afterok:final_stage_array"
        or paired.get("comparison") != PAIR_LABEL
        or paired.get("nodes") != NODES
        or paired.get("gpus") != GPUS
        or paired.get("same_allocation_pairing") is not True
        or paired.get("runs_final_analyzer_after_benchmark") is not True
        or len(final) != 1
        or normalized_job_id(final[0].get("job_id"), "final stage job")
        != FINAL_STAGE_JOB_ID
    ):
        raise RecoveryError("original submission contract differs")
    return study_path, study, submission_path, submission


FAILED_ACCOUNTING_FIELDS = (
    "JobIDRaw",
    "JobName",
    "Partition",
    "Account",
    "QOS",
    "ReqCPUS",
    "ReqMem",
    "ReqNodes",
    "ReqTRES",
    "State",
    "ExitCode",
    "Elapsed",
    "Timelimit",
    "SubmitLine",
    "WorkDir",
    "NodeList",
    "Submit",
    "Start",
    "End",
)


def expected_original_submit_argv(
    *,
    study_root: Path,
    scientific_repo: Path,
    python: Path,
    wan_dir: Path,
    videox_home: Path,
) -> list[str]:
    log_dir = study_root.parent / "_slurm" / "logs"
    job_name = f"vjepa2-{STUDY_ID[:49]}-paired-latency"
    return [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--gpus-per-node=1",
        "--cpus-per-task=32",
        "--mem=256G",
        "--time=04:00:00",
        "--partition=batch",
        "--dependency=afterok:481132",
        "--no-requeue",
        "--open-mode=append",
        "--export=ALL",
        f"--job-name={job_name}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        "--account=coreai_chef_posttrain",
        "--qos=normal",
        str(scientific_repo / "tools/slurm/vjepa2_paired_latency.sbatch"),
        "--study-id",
        STUDY_ID,
        "--expected-commit",
        TRAINING_COMMIT,
        "--repo-root",
        str(scientific_repo),
        "--study-root",
        str(study_root),
        "--python",
        str(python),
        "--wan-dir",
        str(wan_dir),
        "--videox-home",
        str(videox_home),
        "--submission-record",
        str(study_root / "slurm_submission.json"),
    ]


def validate_failed_accounting(
    row: str,
    *,
    study_root: Path,
    scientific_repo: Path,
    python: Path,
    wan_dir: Path,
    videox_home: Path,
) -> dict[str, Any]:
    rows = [line for line in row.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RecoveryError("failed-job accounting must contain exactly one row")
    values = rows[0].split("|")
    if len(values) != len(FAILED_ACCOUNTING_FIELDS):
        raise RecoveryError("failed-job accounting row has the wrong schema")
    record = dict(zip(FAILED_ACCOUNTING_FIELDS, values))
    expected_name = f"vjepa2-{STUDY_ID[:49]}-paired-latency"
    req_tres = set(record["ReqTRES"].split(","))
    if (
        normalized_job_id(record["JobIDRaw"], "failed accounting job")
        != FAILED_JOB_ID
        or record["JobName"] != expected_name
        or record["Partition"] != PARTITION
        or record["Account"] != ACCOUNT
        or record["QOS"] != QOS
        or record["ReqCPUS"] != str(CPUS)
        or record["ReqMem"] != MEMORY
        or record["ReqNodes"] != str(NODES)
        or not {"cpu=32", "gres/gpu=1", "mem=256G", "node=1"}.issubset(
            req_tres
        )
        or record["State"] != "FAILED"
        or record["ExitCode"] != FAILED_EXIT_CODE
        or record["Elapsed"] != FAILED_ELAPSED
        or record["Timelimit"] != TIME_LIMIT
        or Path(record["WorkDir"]) != scientific_repo
        or not record["NodeList"].strip()
    ):
        raise RecoveryError("failed job 481133 accounting differs")
    try:
        submit_argv = shlex.split(record["SubmitLine"])
    except ValueError as exc:
        raise RecoveryError("failed job SubmitLine is not valid shell syntax") from exc
    expected_argv = expected_original_submit_argv(
        study_root=study_root,
        scientific_repo=scientific_repo,
        python=python,
        wan_dir=wan_dir,
        videox_home=videox_home,
    )
    if submit_argv != expected_argv:
        raise RecoveryError("failed job 481133 SubmitLine differs")
    return record


def validate_final_accounting(rows: Sequence[str]) -> list[dict[str, Any]]:
    observed: dict[int, dict[str, Any]] = {}
    expected_name = f"vjepa2-{STUDY_ID[:54]}-u1000"
    for raw in rows:
        if not raw.strip():
            continue
        values = raw.split("|")
        if len(values) != 4:
            raise RecoveryError("final-array accounting row has the wrong schema")
        job, state, exit_code, name = values
        prefix = f"{FINAL_STAGE_JOB_ID}_"
        if not job.startswith(prefix):
            raise RecoveryError("final-array accounting job ID differs")
        task_text = job[len(prefix) :]
        if not task_text.isdigit():
            raise RecoveryError("final-array accounting task ID is malformed")
        task = int(task_text)
        if task in observed:
            raise RecoveryError("final-array accounting duplicates a task")
        if (
            task not in range(5)
            or state != "COMPLETED"
            or exit_code != "0:0"
            or name != expected_name
        ):
            raise RecoveryError("final update-1000 accounting differs")
        observed[task] = {
            "job_id": job,
            "array_task_id": task,
            "state": state,
            "exit_code": exit_code,
            "job_name": name,
        }
    if set(observed) != set(range(5)):
        raise RecoveryError("final update-1000 accounting task set differs")
    return [observed[index] for index in range(5)]


def _validate_original_failure_files(
    *,
    study_root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_log_dir = study_root.parent / "_slurm" / "logs"
    expected_prefix = f"vjepa2-{STUDY_ID[:49]}-paired-latency-481133"
    if (
        stdout_path.parent != expected_log_dir
        or stderr_path.parent != expected_log_dir
        or stdout_path.name != f"{expected_prefix}.out"
        or stderr_path.name != f"{expected_prefix}.err"
        or stdout_path.stat().st_size != FAILED_STDOUT_BYTES
        or stderr_path.stat().st_size != FAILED_STDERR_BYTES
        or sha256(stdout_path) != FAILED_STDOUT_SHA256
        or sha256(stderr_path) != FAILED_STDERR_SHA256
    ):
        raise RecoveryError("failed job 481133 log provenance differs")
    stderr_lines = stderr_path.read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    if not stderr_lines or stderr_lines[-1] != FAILED_ERROR:
        raise RecoveryError("failed job 481133 terminal error differs")
    return file_record(stdout_path), file_record(stderr_path)


def _validate_legacy_absence(study_root: Path) -> dict[str, Any]:
    legacy_output = canonical_directory(
        study_root / "paired_latency", "legacy paired-latency directory"
    )
    if any(legacy_output.iterdir()):
        raise RecoveryError("legacy paired-latency directory is not empty")
    legacy_analysis = study_root.parent / "_analysis" / STUDY_ID
    try:
        legacy_analysis.lstat()
    except FileNotFoundError:
        pass
    else:
        raise RecoveryError("legacy final-analysis path unexpectedly exists")
    return {
        "empty_paired_latency_directory": str(legacy_output),
        "paired_latency_directory_entry_count": 0,
        "analysis_root_absent": str(legacy_analysis),
    }


def _runtime_path(
    value: str | Path,
    expected: Any,
    label: str,
    *,
    directory: bool,
) -> Path:
    path = (
        canonical_directory(value, label)
        if directory
        else canonical_file(value, label)
    )
    if not isinstance(expected, str):
        raise RecoveryError(f"study lacks {label}")
    expected_path = Path(expected)
    try:
        expected_resolved = expected_path.resolve(strict=True)
    except OSError as exc:
        raise RecoveryError(f"recorded {label} is unavailable") from exc
    if path != expected_resolved:
        raise RecoveryError(f"{label} differs from the immutable study runtime")
    return path


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.failed_job_id != FAILED_JOB_ID:
        raise RecoveryError("this recovery is pinned to failed job 481133")
    if args.final_stage_job_id != FINAL_STAGE_JOB_ID:
        raise RecoveryError("this recovery is pinned to final array 481132")
    if args.scientific_commit != TRAINING_COMMIT:
        raise RecoveryError("scientific commit must be the trained 9cf8e69 tree")

    study_root = canonical_directory(args.study_root, "study root")
    study_path, study, submission_path, submission = (
        _validate_study_and_submission(study_root)
    )
    controller = validate_worktree(
        args.controller_repo_root,
        args.controller_commit,
        "recovery controller",
    )
    scientific = validate_worktree(
        args.scientific_repo_root,
        args.scientific_commit,
        "scientific repository",
    )
    if controller == scientific:
        raise RecoveryError(
            "controller and scientific repositories need distinct worktrees"
        )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(controller),
            "merge-base",
            "--is-ancestor",
            VALIDATOR_FIX_COMMIT,
            args.controller_commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RecoveryError("controller does not descend from the validator fix")
    try:
        compatibility = git_inference_compatibility(
            controller,
            training_commit=TRAINING_COMMIT,
            tool_commit=args.controller_commit,
        )
    except FrontierError as exc:
        raise RecoveryError(str(exc)) from exc
    dependency_path = "tools/benchmark_vjepa2_inference.py"
    training_dependency_blob = _git(
        controller,
        "rev-parse",
        f"{TRAINING_COMMIT}:{dependency_path}",
    )
    controller_dependency_blob = _git(
        controller,
        "rev-parse",
        f"{args.controller_commit}:{dependency_path}",
    )
    if training_dependency_blob != controller_dependency_blob:
        raise RecoveryError(
            "controller changed the paired benchmark's inference dependency"
        )
    verifier_path = "tools/env/verify_b200_runtime.py"
    training_verifier_blob = _git(
        controller,
        "rev-parse",
        f"{TRAINING_COMMIT}:{verifier_path}",
    )
    controller_verifier_blob = _git(
        controller,
        "rev-parse",
        f"{args.controller_commit}:{verifier_path}",
    )
    if training_verifier_blob != controller_verifier_blob:
        raise RecoveryError(
            "controller changed the trained runtime-asset verifier"
        )
    compatibility = {
        **compatibility,
        "paired_benchmark_dependency": {
            "path": dependency_path,
            "training_object_sha": training_dependency_blob,
            "controller_object_sha": controller_dependency_blob,
            "unchanged": True,
        },
        "runtime_asset_verifier_dependency": {
            "path": verifier_path,
            "training_object_sha": training_verifier_blob,
            "controller_object_sha": controller_verifier_blob,
            "unchanged": True,
        },
    }

    recorded_repo = study.get("inputs", {}).get("repository", {}).get("root")
    if recorded_repo != str(scientific):
        raise RecoveryError(
            "scientific worktree is not the repository recorded by the study"
        )
    runtime = study.get("inputs", {}).get("runtime", {})
    python_alias = Path(args.python).expanduser()
    if not python_alias.is_absolute() or not os.access(python_alias, os.X_OK):
        raise RecoveryError("LACWM Python must be an absolute executable")
    recorded_python = runtime.get("python")
    if (
        not isinstance(recorded_python, str)
        or python_alias.resolve(strict=True)
        != Path(recorded_python).resolve(strict=True)
    ):
        raise RecoveryError(
            "LACWM Python alias does not resolve to the immutable study runtime"
        )
    wan = _runtime_path(
        args.wan_dir,
        runtime.get("wan_dir"),
        "Wan directory",
        directory=True,
    )
    videox = _runtime_path(
        args.videox_home,
        runtime.get("videox_home"),
        "VideoX checkout",
        directory=True,
    )
    runtime_assets = validate_runtime_assets(wan, videox)

    recovery_root = Path(args.recovery_root).expanduser()
    if not recovery_root.is_absolute():
        raise RecoveryError("recovery root must be absolute")
    wanted_root = expected_recovery_root(
        study_root, args.recovery_id, args.controller_commit
    )
    if recovery_root != wanted_root:
        raise RecoveryError(
            f"recovery root must equal the predecessor-bound path: {wanted_root}"
        )
    if args.require_recovery_absent:
        try:
            recovery_root.lstat()
        except FileNotFoundError:
            parent = recovery_root.parent
            nearest = parent
            while not nearest.exists():
                nearest = nearest.parent
            canonical_directory(nearest, "recovery ancestor")
        else:
            raise RecoveryError("fresh recovery root already exists")
    else:
        recovery_root = canonical_directory(recovery_root, "recovery root")
        for name in ("logs", "paired_latency", "analysis"):
            directory = canonical_directory(
                recovery_root / name, f"recovery {name} directory"
            )
            if any(directory.iterdir()):
                raise RecoveryError(
                    f"recovery {name} directory must start empty"
                )
        for name in ("protocol.json", "submission.json", "runtime.json"):
            if (recovery_root / name).exists():
                raise RecoveryError(f"recovery {name} already exists")

    stdout_path = canonical_file(args.failed_stdout, "failed-job stdout")
    stderr_path = canonical_file(args.failed_stderr, "failed-job stderr")
    stdout_record, stderr_record = _validate_original_failure_files(
        study_root=study_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    failed_accounting = validate_failed_accounting(
        args.failed_accounting_row,
        study_root=study_root,
        scientific_repo=scientific,
        python=python_alias,
        wan_dir=wan,
        videox_home=videox,
    )
    final_accounting = validate_final_accounting(
        args.final_accounting_row
    )
    legacy = _validate_legacy_absence(study_root)
    paths = recovery_paths(study_root, recovery_root)

    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_PROTOCOL,
            "created_at_utc": utc_now(),
            "recovery_id": args.recovery_id,
            "comparison": PAIR_LABEL,
            "reason": (
                "original validator rejected complete path/sha256/bytes records "
                "before model loading or timing"
            ),
            "study": file_record(
                study_path,
                identity_sha256=study["identity_sha256"],
            ),
            "original_submission": file_record(
                submission_path,
                identity_sha256=submission["identity_sha256"],
            ),
            "original_failed_job": {
                "job_id": FAILED_JOB_ID,
                "accounting": failed_accounting,
                "stdout": stdout_record,
                "stderr": stderr_record,
                "terminal_error": FAILED_ERROR,
                "timing_started": False,
                "paired_output_created": False,
                "analysis_created": False,
            },
            "completed_final_stage": {
                "array_job_id": FINAL_STAGE_JOB_ID,
                "tasks": final_accounting,
                "semantic_afterok_predecessor_satisfied": True,
                "new_scheduler_dependency_required": False,
            },
            "controller": {
                "root": str(controller),
                "commit": args.controller_commit,
                "validator_fix_is_ancestor": True,
            },
            "scientific_repository": {
                "root": str(scientific),
                "commit": args.scientific_commit,
                "matches_study_repository": True,
            },
            "inference_code_compatibility": compatibility,
            "runtime": {
                "python": str(python_alias),
                "python_canonical": str(python_alias.resolve(strict=True)),
                "wan_dir": str(wan),
                "videox_home": str(videox),
                "videox_commit": VIDEOX_COMMIT,
                "assets": runtime_assets,
                "asset_contract_source": {
                    "repository_commit": TRAINING_COMMIT,
                    "verifier": "tools/env/verify_b200_runtime.py",
                    "study_runtime_paths_are_identity_bound": True,
                },
            },
            "scheduler": {
                "partition": PARTITION,
                "account": ACCOUNT,
                "qos": QOS,
                "nodes": NODES,
                "tasks": 1,
                "tasks_per_node": 1,
                "gpus_per_node": GPUS,
                "cpus_per_task": CPUS,
                "memory": MEMORY,
                "time_limit": TIME_LIMIT,
                "requeue": False,
                "dependency": None,
                "submitted_held_until_receipt": True,
            },
            "timing_protocol": {
                "J1": {"source": "autonomous", "nfe": 4},
                "VPM": {"source": "autonomous", "nfe": 8},
                "sample_index": 0,
                "batch_size": 1,
                "warmup_pairs": 20,
                "timed_pairs": 100,
                "counterbalance": (
                    "even pair J1-first; odd pair VPM-first"
                ),
                "same_process": True,
                "same_B200": True,
                "both_models_resident": True,
                "future_ground_truth_available": False,
                "online_teacher_calls": 0,
            },
            "legacy_artifacts": legacy,
            "recovery_root": str(recovery_root),
            "paths": paths,
            "immutability": {
                "study_tree_is_read_only": True,
                "legacy_empty_directory_is_preserved": True,
                "original_submission_is_preserved": True,
                "all_new_outputs_are_external": True,
                "exclusive_output_creation": True,
            },
        }
    )


def validate_protocol(
    path: str | Path,
    *,
    controller_repo_root: str | Path | None = None,
    controller_commit: str | None = None,
    scientific_repo_root: str | Path | None = None,
    scientific_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    protocol_path = canonical_file(path, "recovery protocol")
    payload = read_json(protocol_path, "recovery protocol")
    if (
        not identity_valid(payload)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != KIND_PROTOCOL
        or payload.get("comparison") != PAIR_LABEL
        or payload.get("original_failed_job", {}).get("job_id")
        != FAILED_JOB_ID
        or payload.get("completed_final_stage", {}).get("array_job_id")
        != FINAL_STAGE_JOB_ID
        or payload.get("immutability", {}).get(
            "all_new_outputs_are_external"
        )
        is not True
    ):
        raise RecoveryError("recovery protocol identity/contract differs")
    paths = payload.get("paths")
    root = canonical_directory(
        payload.get("recovery_root", ""), "recorded recovery root"
    )
    study_record = payload.get("study")
    if not isinstance(study_record, Mapping):
        raise RecoveryError("recovery protocol lacks its study record")
    study_path = canonical_file(
        str(study_record.get("path", "")), "study manifest"
    )
    study_root = study_path.parent
    expected_paths = recovery_paths(
        canonical_directory(study_root, "recorded study root"),
        root,
    )
    if not isinstance(paths, Mapping) or dict(paths) != expected_paths:
        raise RecoveryError("recovery protocol paths differ")
    if protocol_path != Path(paths["protocol"]):
        raise RecoveryError("recovery protocol path differs")
    controller = payload.get("controller")
    scientific = payload.get("scientific_repository")
    if (
        not isinstance(controller, Mapping)
        or not isinstance(scientific, Mapping)
        or controller.get("validator_fix_is_ancestor") is not True
        or scientific.get("commit") != TRAINING_COMMIT
        or scientific.get("matches_study_repository") is not True
        or root
        != expected_recovery_root(
            study_root,
            str(payload.get("recovery_id", "")),
            str(controller.get("commit", "")),
        )
    ):
        raise RecoveryError("recovery repository/root provenance differs")
    validate_file_record(
        study_record,
        study_path,
        label="study",
        identity_sha256=STUDY_IDENTITY,
    )
    original_submission = canonical_file(
        study_path.parent / "slurm_submission.json",
        "original submission",
    )
    validate_file_record(
        payload.get("original_submission"),
        original_submission,
        label="original submission",
        identity_sha256=ORIGINAL_SUBMISSION_IDENTITY,
    )
    failed = payload.get("original_failed_job")
    completed = payload.get("completed_final_stage")
    scheduler = payload.get("scheduler")
    timing = payload.get("timing_protocol")
    immutability = payload.get("immutability")
    runtime = payload.get("runtime")
    if (
        not isinstance(failed, Mapping)
        or failed.get("job_id") != FAILED_JOB_ID
        or failed.get("terminal_error") != FAILED_ERROR
        or failed.get("timing_started") is not False
        or failed.get("paired_output_created") is not False
        or failed.get("analysis_created") is not False
        or not isinstance(completed, Mapping)
        or completed.get("array_job_id") != FINAL_STAGE_JOB_ID
        or completed.get("semantic_afterok_predecessor_satisfied") is not True
        or completed.get("new_scheduler_dependency_required") is not False
        or scheduler
        != {
            "partition": PARTITION,
            "account": ACCOUNT,
            "qos": QOS,
            "nodes": NODES,
            "tasks": 1,
            "tasks_per_node": 1,
            "gpus_per_node": GPUS,
            "cpus_per_task": CPUS,
            "memory": MEMORY,
            "time_limit": TIME_LIMIT,
            "requeue": False,
            "dependency": None,
            "submitted_held_until_receipt": True,
        }
        or not isinstance(timing, Mapping)
        or timing.get("J1") != {"source": "autonomous", "nfe": 4}
        or timing.get("VPM") != {"source": "autonomous", "nfe": 8}
        or timing.get("sample_index") != 0
        or timing.get("batch_size") != 1
        or timing.get("warmup_pairs") != 20
        or timing.get("timed_pairs") != 100
        or timing.get("same_process") is not True
        or timing.get("same_B200") is not True
        or timing.get("both_models_resident") is not True
        or timing.get("future_ground_truth_available") is not False
        or timing.get("online_teacher_calls") != 0
        or not isinstance(immutability, Mapping)
        or immutability.get("study_tree_is_read_only") is not True
        or immutability.get("legacy_empty_directory_is_preserved") is not True
        or immutability.get("original_submission_is_preserved") is not True
        or immutability.get("all_new_outputs_are_external") is not True
        or immutability.get("exclusive_output_creation") is not True
    ):
        raise RecoveryError("recovery scientific/scheduler contract differs")
    if not isinstance(runtime, Mapping):
        raise RecoveryError("recovery runtime contract is missing")
    python_alias = Path(str(runtime.get("python", "")))
    python_canonical = Path(str(runtime.get("python_canonical", "")))
    if (
        not python_alias.is_absolute()
        or not os.access(python_alias, os.X_OK)
        or python_alias.resolve(strict=True) != python_canonical
        or runtime.get("videox_commit") != VIDEOX_COMMIT
        or runtime.get("asset_contract_source")
        != {
            "repository_commit": TRAINING_COMMIT,
            "verifier": "tools/env/verify_b200_runtime.py",
            "study_runtime_paths_are_identity_bound": True,
        }
    ):
        raise RecoveryError("recovery runtime identity differs")
    runtime_assets = validate_runtime_assets(
        str(runtime.get("wan_dir", "")),
        str(runtime.get("videox_home", "")),
    )
    if runtime.get("assets") != runtime_assets:
        raise RecoveryError("recovery runtime assets differ")
    stdout_path = canonical_file(
        str(failed.get("stdout", {}).get("path", "")),
        "original failed-job stdout",
    )
    stderr_path = canonical_file(
        str(failed.get("stderr", {}).get("path", "")),
        "original failed-job stderr",
    )
    validate_file_record(
        failed.get("stdout"), stdout_path, label="failed-job stdout"
    )
    validate_file_record(
        failed.get("stderr"), stderr_path, label="failed-job stderr"
    )
    _validate_original_failure_files(
        study_root=study_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    legacy = _validate_legacy_absence(study_root)
    if payload.get("legacy_artifacts") != legacy:
        raise RecoveryError("legacy absence evidence differs")
    for field, expected in (
        (controller_repo_root, payload.get("controller", {}).get("root")),
        (controller_commit, payload.get("controller", {}).get("commit")),
        (
            scientific_repo_root,
            payload.get("scientific_repository", {}).get("root"),
        ),
        (
            scientific_commit,
            payload.get("scientific_repository", {}).get("commit"),
        ),
    ):
        if field is not None and str(field) != str(expected):
            raise RecoveryError("runtime repository/commit differs from protocol")
    return protocol_path, payload


def build_submission(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    *,
    job_id: str,
    job_name: str,
    submit_line_tokens: Sequence[str],
) -> dict[str, Any]:
    normalized = normalized_job_id(job_id, "recovery Slurm job")
    if normalized == FAILED_JOB_ID:
        raise RecoveryError("recovery allocation must have a fresh Slurm job ID")
    expected_name = (
        f"vjepa2-paired-recovery-481133-"
        f"{str(protocol['controller']['commit'])[:7]}"
    )
    if job_name != expected_name:
        raise RecoveryError("recovery job name differs")
    paths = protocol["paths"]
    expected_tokens = expected_recovery_submit_argv(
        protocol=protocol,
        job_name=job_name,
    )
    normalized_tokens = [str(value) for value in submit_line_tokens]
    if normalized_tokens != expected_tokens:
        raise RecoveryError("recovery SubmitLine token vector differs")
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_SUBMISSION,
            "submitted_at_utc": utc_now(),
            "protocol": file_record(
                protocol_path,
                identity_sha256=str(protocol["identity_sha256"]),
            ),
            "study_identity_sha256": STUDY_IDENTITY,
            "original_failed_job_id": FAILED_JOB_ID,
            "completed_final_stage_job_id": FINAL_STAGE_JOB_ID,
            "job": {
                "job_id": normalized,
                "job_name": job_name,
                "submitted_held": True,
                "receipt_written_before_release": True,
                "dependency": None,
                "scheduler": dict(protocol["scheduler"]),
                "stdout_pattern": str(
                    Path(protocol["recovery_root"]) / "logs" / "%x-%j.out"
                ),
                "stderr_pattern": str(
                    Path(protocol["recovery_root"]) / "logs" / "%x-%j.err"
                ),
                "submit_line_tokens": normalized_tokens,
                "submit_line_tokens_sha256": hashlib.sha256(
                    canonical_json(normalized_tokens)
                ).hexdigest(),
            },
            "controller": dict(protocol["controller"]),
            "scientific_repository": dict(
                protocol["scientific_repository"]
            ),
            "outputs": {
                "runtime": paths["runtime"],
                "paired_latency": paths["paired_latency"],
                "analysis_json": paths["analysis_json"],
                "analysis_markdown": paths["analysis_markdown"],
            },
        }
    )


def expected_recovery_submit_argv(
    *,
    protocol: Mapping[str, Any],
    job_name: str,
) -> list[str]:
    scheduler = protocol["scheduler"]
    controller = protocol["controller"]
    scientific = protocol["scientific_repository"]
    runtime = protocol["runtime"]
    paths = protocol["paths"]
    recovery_root = Path(str(protocol["recovery_root"]))
    controller_root = Path(str(controller["root"]))
    study_root = Path(str(protocol["study"]["path"])).parent
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--nodes={scheduler['nodes']}",
        f"--ntasks={scheduler['tasks']}",
        f"--ntasks-per-node={scheduler['tasks_per_node']}",
        f"--gpus-per-node={scheduler['gpus_per_node']}",
        f"--cpus-per-task={scheduler['cpus_per_task']}",
        f"--mem={scheduler['memory']}",
        f"--time={scheduler['time_limit']}",
        f"--partition={scheduler['partition']}",
        f"--account={scheduler['account']}",
        f"--qos={scheduler['qos']}",
        f"--chdir={controller_root}",
        "--no-requeue",
        "--open-mode=append",
        "--export=ALL",
        f"--job-name={job_name}",
        f"--output={recovery_root}/logs/%x-%j.out",
        f"--error={recovery_root}/logs/%x-%j.err",
        str(
            controller_root
            / "tools/slurm/vjepa2_paired_latency_recovery.sbatch"
        ),
        "--controller-root",
        str(controller_root),
        "--controller-commit",
        str(controller["commit"]),
        "--scientific-root",
        str(scientific["root"]),
        "--scientific-commit",
        str(scientific["commit"]),
        "--study-root",
        str(study_root),
        "--protocol",
        str(paths["protocol"]),
        "--recovery-submission",
        str(paths["submission"]),
        "--python",
        str(runtime["python"]),
        "--wan-dir",
        str(runtime["wan_dir"]),
        "--videox-home",
        str(runtime["videox_home"]),
    ]


def validate_submission(
    path: str | Path,
    *,
    protocol_path: str | Path,
    slurm_job_id: str,
    output: str | Path | None = None,
    controller_repo_root: str | Path | None = None,
    controller_commit: str | None = None,
    scientific_repo_root: str | Path | None = None,
    scientific_commit: str | None = None,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    checked_protocol_path, protocol = validate_protocol(
        protocol_path,
        controller_repo_root=controller_repo_root,
        controller_commit=controller_commit,
        scientific_repo_root=scientific_repo_root,
        scientific_commit=scientific_commit,
    )
    submission_path = canonical_file(path, "recovery submission")
    payload = read_json(submission_path, "recovery submission")
    job = payload.get("job")
    outputs = payload.get("outputs")
    if (
        not identity_valid(payload)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != KIND_SUBMISSION
        or payload.get("study_identity_sha256") != STUDY_IDENTITY
        or payload.get("original_failed_job_id") != FAILED_JOB_ID
        or payload.get("completed_final_stage_job_id") != FINAL_STAGE_JOB_ID
        or not isinstance(job, Mapping)
        or normalized_job_id(job.get("job_id"), "receipt Slurm job")
        != normalized_job_id(slurm_job_id, "runtime Slurm job")
        or job.get("job_name")
        != (
            "vjepa2-paired-recovery-481133-"
            f"{str(protocol['controller']['commit'])[:7]}"
        )
        or job.get("dependency") is not None
        or job.get("submitted_held") is not True
        or job.get("receipt_written_before_release") is not True
        or job.get("scheduler") != protocol.get("scheduler")
        or job.get("submit_line_tokens")
        != expected_recovery_submit_argv(
            protocol=protocol,
            job_name=str(job.get("job_name", "")),
        )
        or job.get("submit_line_tokens_sha256")
        != hashlib.sha256(
            canonical_json(job.get("submit_line_tokens"))
        ).hexdigest()
        or payload.get("controller") != protocol.get("controller")
        or payload.get("scientific_repository")
        != protocol.get("scientific_repository")
        or not isinstance(outputs, Mapping)
        or dict(outputs)
        != {
            "runtime": protocol["paths"]["runtime"],
            "paired_latency": protocol["paths"]["paired_latency"],
            "analysis_json": protocol["paths"]["analysis_json"],
            "analysis_markdown": protocol["paths"]["analysis_markdown"],
        }
    ):
        raise RecoveryError("recovery submission identity/contract differs")
    validate_file_record(
        payload.get("protocol"),
        checked_protocol_path,
        label="recovery protocol",
        identity_sha256=str(protocol["identity_sha256"]),
    )
    if submission_path != Path(protocol["paths"]["submission"]):
        raise RecoveryError("recovery submission path differs")
    if output is not None and Path(output).expanduser() != Path(
        protocol["paths"]["paired_latency"]
    ):
        raise RecoveryError("paired recovery output differs from receipt")
    return submission_path, payload, checked_protocol_path, protocol


def validate_current_accounting(
    row: str,
    *,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [line for line in row.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RecoveryError(
            "current recovery accounting must contain exactly one row"
        )
    values = rows[0].split("|")
    if len(values) != len(FAILED_ACCOUNTING_FIELDS):
        raise RecoveryError("current recovery accounting row has the wrong schema")
    record = dict(zip(FAILED_ACCOUNTING_FIELDS, values))
    job = submission["job"]
    try:
        submit_tokens = shlex.split(record["SubmitLine"])
    except ValueError as exc:
        raise RecoveryError(
            "current recovery SubmitLine is not valid shell syntax"
        ) from exc
    req_tres = set(record["ReqTRES"].split(","))
    if (
        normalized_job_id(record["JobIDRaw"], "current accounting job")
        != job["job_id"]
        or record["JobName"] != job["job_name"]
        or record["Partition"] != PARTITION
        or record["Account"] != ACCOUNT
        or record["QOS"] != QOS
        or record["ReqCPUS"] != str(CPUS)
        or record["ReqMem"] != MEMORY
        or record["ReqNodes"] != str(NODES)
        or not {"cpu=32", "gres/gpu=1", "mem=256G", "node=1"}.issubset(
            req_tres
        )
        or record["State"] not in {"RUNNING", "COMPLETING"}
        or record["ExitCode"] != "0:0"
        or record["Timelimit"] != TIME_LIMIT
        or Path(record["WorkDir"]) != Path(
            submission["controller"]["root"]
        )
        or not record["NodeList"].strip()
        or submit_tokens != job["submit_line_tokens"]
        or hashlib.sha256(canonical_json(submit_tokens)).hexdigest()
        != job["submit_line_tokens_sha256"]
    ):
        raise RecoveryError("current paired-recovery accounting differs")
    return record


def build_runtime_evidence(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    submission_path: Path,
    submission: Mapping[str, Any],
    current_accounting_row: str,
    slurm_node: str,
    cuda_visible_devices: str,
) -> dict[str, Any]:
    if not slurm_node.strip() or not cuda_visible_devices.strip():
        raise RecoveryError("runtime Slurm node/GPU environment is incomplete")
    accounting = validate_current_accounting(
        current_accounting_row,
        submission=submission,
    )
    if accounting["NodeList"] != slurm_node:
        raise RecoveryError("runtime Slurm node differs from accounting")
    runtime_assets = validate_runtime_assets(
        protocol["runtime"]["wan_dir"],
        protocol["runtime"]["videox_home"],
    )
    if runtime_assets != protocol["runtime"]["assets"]:
        raise RecoveryError("runtime assets changed before model loading")
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RUNTIME,
            "validated_at_utc": utc_now(),
            "protocol": file_record(
                protocol_path,
                identity_sha256=str(protocol["identity_sha256"]),
            ),
            "submission": file_record(
                submission_path,
                identity_sha256=str(submission["identity_sha256"]),
            ),
            "job_id": submission["job"]["job_id"],
            "accounting": accounting,
            "slurm_environment": {
                "job_id": submission["job"]["job_id"],
                "node": slurm_node,
                "cuda_visible_devices": cuda_visible_devices,
                "job_num_nodes": 1,
                "cpus_per_task": CPUS,
            },
            "submit_line_tokens_sha256": submission["job"][
                "submit_line_tokens_sha256"
            ],
            "runtime_assets_before_model_loading": runtime_assets,
            "scheduler_contract_validated": True,
            "actual_job_id_validated": True,
        }
    )


def validate_runtime_evidence(
    path: str | Path,
    *,
    protocol_path: str | Path,
    submission_path: str | Path,
    slurm_job_id: str,
) -> tuple[Path, dict[str, Any]]:
    checked_submission_path, submission, checked_protocol_path, protocol = (
        validate_submission(
            submission_path,
            protocol_path=protocol_path,
            slurm_job_id=slurm_job_id,
        )
    )
    runtime_path = canonical_file(path, "recovery runtime evidence")
    payload = read_json(runtime_path, "recovery runtime evidence")
    environment = payload.get("slurm_environment")
    if (
        runtime_path != Path(protocol["paths"]["runtime"])
        or not identity_valid(payload)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != KIND_RUNTIME
        or payload.get("job_id") != submission["job"]["job_id"]
        or payload.get("submit_line_tokens_sha256")
        != submission["job"]["submit_line_tokens_sha256"]
        or payload.get("scheduler_contract_validated") is not True
        or payload.get("actual_job_id_validated") is not True
        or payload.get("runtime_assets_before_model_loading")
        != protocol["runtime"]["assets"]
        or not isinstance(environment, Mapping)
        or environment.get("job_id") != submission["job"]["job_id"]
        or environment.get("job_num_nodes") != NODES
        or environment.get("cpus_per_task") != CPUS
        or not str(environment.get("node", "")).strip()
        or not str(environment.get("cuda_visible_devices", "")).strip()
    ):
        raise RecoveryError("recovery runtime evidence differs")
    validate_file_record(
        payload.get("protocol"),
        checked_protocol_path,
        label="runtime protocol",
        identity_sha256=str(protocol["identity_sha256"]),
    )
    validate_file_record(
        payload.get("submission"),
        checked_submission_path,
        label="runtime submission",
        identity_sha256=str(submission["identity_sha256"]),
    )
    accounting = payload.get("accounting")
    if not isinstance(accounting, Mapping):
        raise RecoveryError("recovery runtime accounting is absent")
    normalized_row = "|".join(
        str(accounting.get(field, "")) for field in FAILED_ACCOUNTING_FIELDS
    )
    validated = validate_current_accounting(
        normalized_row,
        submission=submission,
    )
    if dict(accounting) != validated:
        raise RecoveryError("recovery runtime accounting record differs")
    if validated["NodeList"] != environment["node"]:
        raise RecoveryError("recovery runtime node binding differs")
    runtime_assets = validate_runtime_assets(
        protocol["runtime"]["wan_dir"],
        protocol["runtime"]["videox_home"],
    )
    if runtime_assets != payload["runtime_assets_before_model_loading"]:
        raise RecoveryError("runtime assets changed after runtime validation")
    return runtime_path, payload


def command_preflight(args: argparse.Namespace) -> int:
    args.require_recovery_absent = True
    protocol = build_protocol(args)
    print(
        json.dumps(
            {
                "status": "paired_recovery_preflight_passed",
                "recovery_id": protocol["recovery_id"],
                "recovery_root": protocol["recovery_root"],
                "failed_job_id": FAILED_JOB_ID,
                "final_stage_job_id": FINAL_STAGE_JOB_ID,
                "controller_commit": protocol["controller"]["commit"],
                "scientific_commit": protocol["scientific_repository"][
                    "commit"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def command_write_protocol(args: argparse.Namespace) -> int:
    args.require_recovery_absent = False
    protocol = build_protocol(args)
    output = Path(args.output).expanduser()
    if output != Path(protocol["paths"]["protocol"]):
        raise RecoveryError("protocol output path differs from recovery contract")
    exclusive_json(output, protocol)
    print(protocol["identity_sha256"])
    return 0


def command_write_submission(args: argparse.Namespace) -> int:
    protocol_path, protocol = validate_protocol(args.protocol)
    output = Path(args.output).expanduser()
    if output != Path(protocol["paths"]["submission"]):
        raise RecoveryError("submission output path differs from protocol")
    payload = build_submission(
        protocol_path,
        protocol,
        job_id=args.job_id,
        job_name=args.job_name,
        submit_line_tokens=args.submit_line_token,
    )
    exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def command_validate_runtime(args: argparse.Namespace) -> int:
    submission_path, submission, protocol_path, protocol = validate_submission(
        args.submission,
        protocol_path=args.protocol,
        slurm_job_id=args.slurm_job_id,
        output=args.output,
        controller_repo_root=args.controller_repo_root,
        controller_commit=args.controller_commit,
        scientific_repo_root=args.scientific_repo_root,
        scientific_commit=args.scientific_commit,
    )
    output_paths = [
        protocol["paths"]["paired_latency"],
        protocol["paths"]["analysis_json"],
        protocol["paths"]["analysis_markdown"],
    ]
    if any(Path(value).exists() for value in output_paths):
        raise RecoveryError("one or more fresh recovery outputs already exist")
    runtime_output = Path(args.runtime_evidence).expanduser()
    if runtime_output != Path(protocol["paths"]["runtime"]):
        raise RecoveryError("runtime evidence output path differs")
    if runtime_output.exists():
        raise RecoveryError("runtime evidence already exists")
    runtime_payload = build_runtime_evidence(
        protocol_path=protocol_path,
        protocol=protocol,
        submission_path=submission_path,
        submission=submission,
        current_accounting_row=args.current_accounting_row,
        slurm_node=args.slurm_node,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    exclusive_json(runtime_output, runtime_payload)
    print(
        json.dumps(
            {
                "status": "paired_recovery_runtime_validated",
                "job_id": submission["job"]["job_id"],
                "protocol_identity_sha256": protocol["identity_sha256"],
                "submission_identity_sha256": submission["identity_sha256"],
                "runtime_identity_sha256": runtime_payload[
                    "identity_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _common_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--controller-repo-root", required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--scientific-repo-root", required=True)
    parser.add_argument("--scientific-commit", required=True)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--failed-job-id", required=True)
    parser.add_argument("--final-stage-job-id", required=True)
    parser.add_argument("--failed-stdout", required=True)
    parser.add_argument("--failed-stderr", required=True)
    parser.add_argument("--failed-accounting-row", required=True)
    parser.add_argument(
        "--final-accounting-row",
        action="append",
        required=True,
    )
    parser.add_argument("--python", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--videox-home", required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _common_protocol_arguments(preflight)
    preflight.set_defaults(handler=command_preflight)
    write_protocol = commands.add_parser("write-protocol")
    _common_protocol_arguments(write_protocol)
    write_protocol.add_argument("--output", required=True)
    write_protocol.set_defaults(handler=command_write_protocol)
    write_submission = commands.add_parser("write-submission")
    write_submission.add_argument("--protocol", required=True)
    write_submission.add_argument("--job-id", required=True)
    write_submission.add_argument("--job-name", required=True)
    write_submission.add_argument(
        "--submit-line-token",
        action="append",
        required=True,
    )
    write_submission.add_argument("--output", required=True)
    write_submission.set_defaults(handler=command_write_submission)
    runtime = commands.add_parser("validate-runtime")
    runtime.add_argument("--protocol", required=True)
    runtime.add_argument("--submission", required=True)
    runtime.add_argument("--slurm-job-id", required=True)
    runtime.add_argument("--output", required=True)
    runtime.add_argument("--controller-repo-root", required=True)
    runtime.add_argument("--controller-commit", required=True)
    runtime.add_argument("--scientific-repo-root", required=True)
    runtime.add_argument("--scientific-commit", required=True)
    runtime.add_argument("--current-accounting-row", required=True)
    runtime.add_argument("--slurm-node", required=True)
    runtime.add_argument("--cuda-visible-devices", required=True)
    runtime.add_argument("--runtime-evidence", required=True)
    runtime.set_defaults(handler=command_validate_runtime)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (RecoveryError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
