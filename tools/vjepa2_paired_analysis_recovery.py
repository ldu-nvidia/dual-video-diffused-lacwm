#!/usr/bin/env python3
"""Fail-closed analyzer-only recovery for completed paired timing job 481826.

The external r1 recovery completed all warmups and 100 paired timing rounds,
then its post-run analyzer failed on a historical-manifest schema regression.
This helper proves that exact incident, keeps every r1 byte immutable, and
binds a fresh external analysis destination.  It never launches or repeats a
model benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import vjepa2_paired_recovery as timing_recovery


SCHEMA_VERSION = 1
KIND_PROTOCOL = "vjepa2_paired_analysis_recovery_protocol"
KIND_SUBMISSION = "vjepa2_paired_analysis_recovery_submission"
RECOVERY_MODE = "external_analyzer_only_recovery"
FAILED_ANALYZER_JOB_ID = "481826"
FAILED_EXIT_CODE = "2:0"
FAILED_ELAPSED = "00:04:27"
FAILED_TERMINAL_ERROR = "ERROR: VPM update 1 rank 0 manifest differs"
FAILED_NODE = "pool0-0077"
FAILED_SUBMIT = "2026-07-30T10:03:34"
FAILED_START = "2026-07-30T10:03:39"
FAILED_END = "2026-07-30T10:08:06"
TIMING_RECOVERY_ID = "paired-481133-43ed5d3-r1"
TIMING_CONTROLLER_COMMIT = "43ed5d3bad6056e257786fefcd602650e2cbe06a"
ANALYZER_FIX_BASE_COMMIT = "131858e583f334e9cbc83a98a803a8aafe07b3fc"
ANALYSIS_CPUS = 8
ANALYSIS_MEMORY = "64G"
ANALYSIS_TIME_LIMIT = "01:00:00"
ANALYSIS_GPUS = 1
PAIRED_IDENTITY = (
    "f273e88fe0c5d6723958974cc93e587b78fc32e6f37d1ce0be1abca95fafe92d"
)
PINNED_FILES = {
    "protocol": {
        "relative": "protocol.json",
        "bytes": 15_126,
        "sha256": (
            "9995673888570a1135fdcf1fe231a2ef93f55b7cf1871e6a773b8dc98aef0aea"
        ),
        "identity_sha256": (
            "ddc03429ea190871213d7ff1a1abf0538c010ecbf77e43d57556fe62d7637a89"
        ),
    },
    "submission": {
        "relative": "submission.json",
        "bytes": 6_649,
        "sha256": (
            "a7d215a26423ff3ec2795ba08a54791840f595155fb3e92204bcec55941e0e86"
        ),
        "identity_sha256": (
            "ef11258af35c8147ecd33b88b32fd7f5b96d79b153aa37196af78eb2aba344b9"
        ),
    },
    "runtime": {
        "relative": "runtime.json",
        "bytes": 6_799,
        "sha256": (
            "3d8d89710873b04dfcb45981744fba3a911bd0a8d053bbcca06722977db46955"
        ),
        "identity_sha256": (
            "e3f7de347ee9b200c13a43dde96f7d545fa02b113a91b5dfcff5c9b9259bf151"
        ),
    },
    "paired_latency": {
        "relative": (
            "paired_latency/"
            + timing_recovery.OUTPUT_BASENAME
        ),
        "bytes": 75_578,
        "sha256": (
            "314a40090a96cd06c6e331f833b689aa06059ba1988cb83955ab9a2b9733d84c"
        ),
        "identity_sha256": PAIRED_IDENTITY,
    },
    "stdout": {
        "relative": (
            "logs/vjepa2-paired-recovery-481133-43ed5d3-481826.out"
        ),
        "bytes": 8_216,
        "sha256": (
            "f946f7383979cf2d88178ae5c549e8c574530c71ec3a2d4c8b6fa14361f04d5e"
        ),
    },
    "stderr": {
        "relative": (
            "logs/vjepa2-paired-recovery-481133-43ed5d3-481826.err"
        ),
        "bytes": 3_327,
        "sha256": (
            "bb95e5d06e538c0495063b23bf1521828e6b42c37a95d7ec4a4d79840aefb121"
        ),
    },
}
RECOVERY_ID_RE = re.compile(
    r"^paired-analysis-481826-([0-9a-f]{7,12})-r([2-9][0-9]*)$"
)


class AnalysisRecoveryError(RuntimeError):
    """Raised when the analyzer-only recovery cannot be proven safe."""


def _convert_error(exc: timing_recovery.RecoveryError) -> AnalysisRecoveryError:
    return AnalysisRecoveryError(str(exc))


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise AnalysisRecoveryError(
            f"git {' '.join(arguments)} failed for {repo}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _validate_controller(root: str | Path, commit: str) -> Path:
    try:
        repo = timing_recovery.validate_worktree(
            root, commit, "analysis controller"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            ANALYZER_FIX_BASE_COMMIT,
            commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AnalysisRecoveryError(
            "analysis controller is not descended from the audited recovery base"
        )
    return repo


def timing_recovery_root(study_root: Path) -> Path:
    return (
        study_root.parent
        / "_paired_recoveries"
        / timing_recovery.STUDY_ID
        / TIMING_RECOVERY_ID
    )


def expected_recovery_root(
    study_root: Path,
    recovery_id: str,
    controller_commit: str,
) -> Path:
    match = RECOVERY_ID_RE.fullmatch(recovery_id)
    if match is None or not controller_commit.startswith(match.group(1)):
        raise AnalysisRecoveryError(
            "analysis recovery ID must bind job 481826, controller, and r2+"
        )
    return (
        study_root.parent
        / "_paired_recoveries"
        / timing_recovery.STUDY_ID
        / recovery_id
    )


def recovery_paths(recovery_root: Path) -> dict[str, str]:
    return {
        "protocol": str(recovery_root / "protocol.json"),
        "submission": str(recovery_root / "submission.json"),
        "analysis_json": str(recovery_root / "analysis" / "analysis.json"),
        "analysis_markdown": str(
            recovery_root / "analysis" / "analysis.md"
        ),
    }


def _pinned_file(
    root: Path,
    key: str,
) -> tuple[Path, dict[str, Any]]:
    spec = PINNED_FILES[key]
    try:
        path = timing_recovery.canonical_file(
            root / str(spec["relative"]), f"r1 {key}"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    record = timing_recovery.file_record(
        path,
        identity_sha256=(
            str(spec["identity_sha256"])
            if "identity_sha256" in spec
            else None
        ),
    )
    expected = {
        "path": str(path),
        "sha256": spec["sha256"],
        "bytes": spec["bytes"],
    }
    if "identity_sha256" in spec:
        expected["identity_sha256"] = spec["identity_sha256"]
    if record != expected:
        raise AnalysisRecoveryError(f"r1 {key} bytes differ")
    return path, record


def _validate_r1_tree(
    study_root: Path,
) -> tuple[
    Path,
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    expected = timing_recovery_root(study_root)
    try:
        root = timing_recovery.canonical_directory(
            expected, "r1 timing recovery"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    if root != expected:
        raise AnalysisRecoveryError("r1 timing-recovery root differs")
    direct = {path.name for path in root.iterdir()}
    if direct != {
        "analysis",
        "logs",
        "paired_latency",
        "protocol.json",
        "runtime.json",
        "submission.json",
    }:
        raise AnalysisRecoveryError("r1 top-level inventory differs")
    analysis_dir = root / "analysis"
    if (
        not analysis_dir.is_dir()
        or analysis_dir.is_symlink()
        or any(analysis_dir.iterdir())
    ):
        raise AnalysisRecoveryError(
            "r1 analysis directory is no longer the preserved empty failure"
        )
    logs = root / "logs"
    paired_dir = root / "paired_latency"
    if (
        not logs.is_dir()
        or logs.is_symlink()
        or not paired_dir.is_dir()
        or paired_dir.is_symlink()
        or {path.name for path in logs.iterdir()}
        != {
            "vjepa2-paired-recovery-481133-43ed5d3-481826.out",
            "vjepa2-paired-recovery-481133-43ed5d3-481826.err",
        }
        or {path.name for path in paired_dir.iterdir()}
        != {timing_recovery.OUTPUT_BASENAME}
    ):
        raise AnalysisRecoveryError("r1 log/timing inventory differs")

    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for key in PINNED_FILES:
        paths[key], records[key] = _pinned_file(root, key)

    try:
        (
            checked_submission,
            submission,
            checked_protocol,
            protocol,
        ) = timing_recovery.validate_submission(
            paths["submission"],
            protocol_path=paths["protocol"],
            slurm_job_id=FAILED_ANALYZER_JOB_ID,
            output=paths["paired_latency"],
        )
        timing_recovery.validate_runtime_evidence(
            paths["runtime"],
            protocol_path=checked_protocol,
            submission_path=checked_submission,
            slurm_job_id=FAILED_ANALYZER_JOB_ID,
        )
        timing_recovery.validate_worktree(
            protocol["controller"]["root"],
            protocol["controller"]["commit"],
            "r1 controller",
        )
        timing_recovery.validate_worktree(
            protocol["scientific_repository"]["root"],
            protocol["scientific_repository"]["commit"],
            "r1 scientific repository",
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc

    paired = timing_recovery.read_json(
        paths["paired_latency"], "r1 paired latency"
    )
    recovery = paired.get("recovery")
    if (
        not timing_recovery.identity_valid(paired)
        or paired.get("identity_sha256") != PAIRED_IDENTITY
        or paired.get("kind")
        != "vjepa2_controlled_study_paired_latency"
        or paired.get("comparison") != timing_recovery.PAIR_LABEL
        or paired.get("git_commit") != timing_recovery.TRAINING_COMMIT
        or paired.get("slurm", {}).get("job_id")
        != FAILED_ANALYZER_JOB_ID
        or not isinstance(recovery, Mapping)
        or recovery.get("mode") != "external_validator_only_recovery"
        or recovery.get("controller", {}).get("commit")
        != TIMING_CONTROLLER_COMMIT
        or recovery.get("legacy_study_tree_mutated") is not False
    ):
        raise AnalysisRecoveryError("r1 paired timing identity differs")
    for key in ("protocol", "submission", "runtime"):
        if recovery.get(key) != records[key]:
            raise AnalysisRecoveryError(
                f"r1 paired timing {key} binding differs"
            )
    return root, records, submission, protocol


def validate_failed_accounting(
    row: str,
    *,
    timing_submission: Mapping[str, Any],
) -> dict[str, str]:
    rows = [line for line in row.splitlines() if line.strip()]
    if len(rows) != 1:
        raise AnalysisRecoveryError(
            "failed analyzer accounting must contain exactly one row"
        )
    values = rows[0].split("|")
    fields = timing_recovery.FAILED_ACCOUNTING_FIELDS
    if len(values) != len(fields):
        raise AnalysisRecoveryError(
            "failed analyzer accounting row has the wrong schema"
        )
    record = dict(zip(fields, values))
    try:
        submit_tokens = shlex.split(record["SubmitLine"])
    except ValueError as exc:
        raise AnalysisRecoveryError(
            "failed analyzer SubmitLine is not valid shell syntax"
        ) from exc
    job = timing_submission["job"]
    req_tres = set(record["ReqTRES"].split(","))
    if (
        timing_recovery.normalized_job_id(
            record["JobIDRaw"], "failed analyzer job"
        )
        != FAILED_ANALYZER_JOB_ID
        or record["JobName"] != job["job_name"]
        or record["Partition"] != timing_recovery.PARTITION
        or record["Account"] != timing_recovery.ACCOUNT
        or record["QOS"] != timing_recovery.QOS
        or record["ReqCPUS"] != str(timing_recovery.CPUS)
        or record["ReqMem"] != timing_recovery.MEMORY
        or record["ReqNodes"] != str(timing_recovery.NODES)
        or not {"cpu=32", "gres/gpu=1", "mem=256G", "node=1"}.issubset(
            req_tres
        )
        or record["State"] != "FAILED"
        or record["ExitCode"] != FAILED_EXIT_CODE
        or record["Elapsed"] != FAILED_ELAPSED
        or record["Timelimit"] != timing_recovery.TIME_LIMIT
        or Path(record["WorkDir"]) != Path(
            timing_submission["controller"]["root"]
        )
        or record["NodeList"] != FAILED_NODE
        or record["Submit"] != FAILED_SUBMIT
        or record["Start"] != FAILED_START
        or record["End"] != FAILED_END
        or submit_tokens != job["submit_line_tokens"]
        or hashlib.sha256(
            timing_recovery.canonical_json(submit_tokens)
        ).hexdigest()
        != job["submit_line_tokens_sha256"]
    ):
        raise AnalysisRecoveryError("failed analyzer accounting differs")
    return record


def _failure_error_is_exact(stderr: Path) -> None:
    lines = [
        line.strip()
        for line in stderr.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines or lines[-1] != FAILED_TERMINAL_ERROR:
        raise AnalysisRecoveryError("failed analyzer terminal error differs")


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    try:
        study_root = timing_recovery.canonical_directory(
            args.study_root, "immutable study root"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    if study_root.name != timing_recovery.STUDY_ID:
        raise AnalysisRecoveryError("study root is not the immutable v3 study")
    controller = _validate_controller(
        args.controller_repo_root, args.controller_commit
    )
    wanted_root = expected_recovery_root(
        study_root, args.recovery_id, args.controller_commit
    )
    if Path(args.recovery_root).expanduser() != wanted_root:
        raise AnalysisRecoveryError("analysis recovery root differs")
    if args.require_recovery_absent and wanted_root.exists():
        raise AnalysisRecoveryError("fresh analysis recovery root already exists")
    if wanted_root.exists():
        try:
            recovery_root = timing_recovery.canonical_directory(
                wanted_root, "analysis recovery root"
            )
        except timing_recovery.RecoveryError as exc:
            raise _convert_error(exc) from exc
    else:
        if not wanted_root.parent.is_dir():
            raise AnalysisRecoveryError(
                "analysis recovery parent does not exist"
            )
        recovery_root = wanted_root

    (
        r1_root,
        records,
        timing_submission,
        timing_protocol,
    ) = _validate_r1_tree(study_root)
    accounting = validate_failed_accounting(
        args.failed_accounting_row,
        timing_submission=timing_submission,
    )
    _failure_error_is_exact(
        Path(records["stderr"]["path"])
    )
    try:
        study_path, study, _, _ = (
            timing_recovery._validate_study_and_submission(study_root)
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    paths = recovery_paths(recovery_root)
    python = Path(args.python).expanduser()
    recorded_python = Path(str(timing_protocol["runtime"]["python"]))
    if (
        not python.is_absolute()
        or not python.exists()
        or python.is_dir()
        or python != recorded_python
        or python.resolve(strict=True)
        != Path(str(timing_protocol["runtime"]["python_canonical"]))
    ):
        raise AnalysisRecoveryError(
            "analysis Python differs from the immutable study runtime"
        )
    return timing_recovery.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_PROTOCOL,
            "mode": RECOVERY_MODE,
            "created_at_utc": timing_recovery.utc_now(),
            "recovery_id": args.recovery_id,
            "recovery_root": str(recovery_root),
            "study": timing_recovery.file_record(
                study_path,
                identity_sha256=str(study["identity_sha256"]),
            ),
            "controller": {
                "root": str(controller),
                "commit": args.controller_commit,
                "audited_base_commit": ANALYZER_FIX_BASE_COMMIT,
                "audited_base_is_ancestor": True,
            },
            "runtime": {
                "python": str(python),
                "python_canonical": str(python.resolve(strict=True)),
            },
            "scheduler": {
                "partition": timing_recovery.PARTITION,
                "account": timing_recovery.ACCOUNT,
                "qos": timing_recovery.QOS,
                "nodes": 1,
                "tasks": 1,
                "tasks_per_node": 1,
                "gpus_per_node": ANALYSIS_GPUS,
                "cpus_per_task": ANALYSIS_CPUS,
                "memory": ANALYSIS_MEMORY,
                "time_limit": ANALYSIS_TIME_LIMIT,
                "requeue": False,
                "submitted_held_until_receipt": True,
            },
            "failed_analyzer_job": {
                "job_id": FAILED_ANALYZER_JOB_ID,
                "accounting": accounting,
                "stdout": records["stdout"],
                "stderr": records["stderr"],
                "terminal_error": FAILED_TERMINAL_ERROR,
                "timing_completed": True,
                "paired_output_created": True,
                "analysis_created": False,
            },
            "timing_recovery": {
                "root": str(r1_root),
                "protocol": records["protocol"],
                "submission": records["submission"],
                "runtime": records["runtime"],
                "paired_latency": records["paired_latency"],
                "paired_latency_identity_sha256": PAIRED_IDENTITY,
                "all_warmup_pairs_completed": True,
                "all_timed_pairs_completed": True,
                "timing_reexecution_allowed": False,
            },
            "historical_schema_regression": {
                "cause": (
                    "analyzer internal byte-count metadata leaked into "
                    "reconstructed historical path+sha256 input records"
                ),
                "first_rejection": FAILED_TERMINAL_ERROR,
                "stage_file_sizes_remain_independently_validated": True,
                "historical_signed_manifests_are_not_rewritten": True,
            },
            "immutability": {
                "study_tree_is_read_only": True,
                "r1_tree_is_read_only": True,
                "r1_analysis_directory_remains_empty": True,
                "paired_timing_is_reused_by_exact_sha256": True,
                "timing_is_not_reexecuted": True,
                "all_new_outputs_are_external": True,
                "exclusive_output_creation": True,
            },
            "paths": paths,
        }
    )


def validate_protocol(
    path: str | Path,
    *,
    paired_latency: str | Path | None = None,
    timing_submission: str | Path | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    try:
        protocol_path = timing_recovery.canonical_file(
            path, "analysis recovery protocol"
        )
        payload = timing_recovery.read_json(
            protocol_path, "analysis recovery protocol"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    study_record = payload.get("study")
    controller = payload.get("controller")
    if (
        not timing_recovery.identity_valid(payload)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != KIND_PROTOCOL
        or payload.get("mode") != RECOVERY_MODE
        or not isinstance(study_record, Mapping)
        or not isinstance(controller, Mapping)
        or controller.get("audited_base_commit")
        != ANALYZER_FIX_BASE_COMMIT
        or controller.get("audited_base_is_ancestor") is not True
    ):
        raise AnalysisRecoveryError(
            "analysis recovery protocol identity/contract differs"
        )
    controller_root = _validate_controller(
        str(controller.get("root", "")),
        str(controller.get("commit", "")),
    )
    try:
        study_path = timing_recovery.canonical_file(
            str(study_record.get("path", "")), "analysis recovery study"
        )
        timing_recovery.validate_file_record(
            study_record,
            study_path,
            label="analysis recovery study",
            identity_sha256=timing_recovery.STUDY_IDENTITY,
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    study_root = study_path.parent
    recovery_root = expected_recovery_root(
        study_root,
        str(payload.get("recovery_id", "")),
        str(controller.get("commit", "")),
    )
    try:
        checked_root = timing_recovery.canonical_directory(
            recovery_root, "analysis recovery root"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    expected_paths = recovery_paths(checked_root)
    if (
        checked_root != Path(str(payload.get("recovery_root", "")))
        or protocol_path != Path(expected_paths["protocol"])
        or payload.get("paths") != expected_paths
        or Path(str(controller.get("root", ""))) != controller_root
    ):
        raise AnalysisRecoveryError(
            "analysis recovery repository/path provenance differs"
        )

    r1_root, records, timing_receipt, timing_protocol = _validate_r1_tree(
        study_root
    )
    failed = payload.get("failed_analyzer_job")
    timing = payload.get("timing_recovery")
    regression = payload.get("historical_schema_regression")
    immutability = payload.get("immutability")
    runtime = payload.get("runtime")
    scheduler = payload.get("scheduler")
    if (
        not isinstance(failed, Mapping)
        or failed.get("job_id") != FAILED_ANALYZER_JOB_ID
        or failed.get("stdout") != records["stdout"]
        or failed.get("stderr") != records["stderr"]
        or failed.get("terminal_error") != FAILED_TERMINAL_ERROR
        or failed.get("timing_completed") is not True
        or failed.get("paired_output_created") is not True
        or failed.get("analysis_created") is not False
        or not isinstance(timing, Mapping)
        or timing
        != {
            "root": str(r1_root),
            "protocol": records["protocol"],
            "submission": records["submission"],
            "runtime": records["runtime"],
            "paired_latency": records["paired_latency"],
            "paired_latency_identity_sha256": PAIRED_IDENTITY,
            "all_warmup_pairs_completed": True,
            "all_timed_pairs_completed": True,
            "timing_reexecution_allowed": False,
        }
        or not isinstance(regression, Mapping)
        or regression.get(
            "stage_file_sizes_remain_independently_validated"
        )
        is not True
        or regression.get("historical_signed_manifests_are_not_rewritten")
        is not True
        or immutability
        != {
            "study_tree_is_read_only": True,
            "r1_tree_is_read_only": True,
            "r1_analysis_directory_remains_empty": True,
            "paired_timing_is_reused_by_exact_sha256": True,
            "timing_is_not_reexecuted": True,
            "all_new_outputs_are_external": True,
            "exclusive_output_creation": True,
        }
        or runtime
        != {
            "python": timing_protocol["runtime"]["python"],
            "python_canonical": timing_protocol["runtime"][
                "python_canonical"
            ],
        }
        or scheduler
        != {
            "partition": timing_recovery.PARTITION,
            "account": timing_recovery.ACCOUNT,
            "qos": timing_recovery.QOS,
            "nodes": 1,
            "tasks": 1,
            "tasks_per_node": 1,
            "gpus_per_node": ANALYSIS_GPUS,
            "cpus_per_task": ANALYSIS_CPUS,
            "memory": ANALYSIS_MEMORY,
            "time_limit": ANALYSIS_TIME_LIMIT,
            "requeue": False,
            "submitted_held_until_receipt": True,
        }
    ):
        raise AnalysisRecoveryError(
            "analysis recovery incident/immutability contract differs"
        )
    accounting = failed.get("accounting")
    if not isinstance(accounting, Mapping):
        raise AnalysisRecoveryError(
            "analysis recovery accounting evidence is absent"
        )
    normalized_row = "|".join(
        str(accounting.get(field, ""))
        for field in timing_recovery.FAILED_ACCOUNTING_FIELDS
    )
    if dict(accounting) != validate_failed_accounting(
        normalized_row,
        timing_submission=timing_receipt,
    ):
        raise AnalysisRecoveryError(
            "analysis recovery accounting record differs"
        )
    _failure_error_is_exact(Path(records["stderr"]["path"]))

    override_pairs = (
        (
            paired_latency,
            records["paired_latency"]["path"],
            "paired latency",
        ),
        (
            timing_submission,
            records["submission"]["path"],
            "timing recovery submission",
        ),
        (output_json, expected_paths["analysis_json"], "analysis JSON"),
        (
            output_markdown,
            expected_paths["analysis_markdown"],
            "analysis Markdown",
        ),
    )
    for actual, expected, label in override_pairs:
        if actual is not None and Path(actual).expanduser() != Path(expected):
            raise AnalysisRecoveryError(f"{label} differs from protocol")
    return protocol_path, payload


def expected_analysis_submit_argv(
    *,
    protocol: Mapping[str, Any],
    job_name: str,
) -> list[str]:
    controller = protocol["controller"]
    runtime = protocol["runtime"]
    scheduler = protocol["scheduler"]
    paths = protocol["paths"]
    timing = protocol["timing_recovery"]
    root = Path(str(protocol["recovery_root"]))
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
        f"--chdir={controller['root']}",
        "--no-requeue",
        "--open-mode=append",
        "--export=ALL",
        f"--job-name={job_name}",
        f"--output={root}/logs/%x-%j.out",
        f"--error={root}/logs/%x-%j.err",
        str(
            Path(str(controller["root"]))
            / "tools/slurm/vjepa2_paired_analysis_recovery.sbatch"
        ),
        "--controller-root",
        str(controller["root"]),
        "--controller-commit",
        str(controller["commit"]),
        "--study-root",
        str(Path(str(protocol["study"]["path"])).parent),
        "--protocol",
        str(paths["protocol"]),
        "--submission",
        str(paths["submission"]),
        "--python",
        str(runtime["python"]),
        "--paired-latency",
        str(timing["paired_latency"]["path"]),
        "--timing-submission",
        str(timing["submission"]["path"]),
    ]


def build_submission(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    *,
    job_id: str,
    job_name: str,
    submit_line_tokens: Sequence[str],
) -> dict[str, Any]:
    normalized = timing_recovery.normalized_job_id(
        job_id, "analysis recovery job"
    )
    if normalized == FAILED_ANALYZER_JOB_ID:
        raise AnalysisRecoveryError(
            "analysis recovery requires a fresh Slurm job"
        )
    expected_name = (
        "vjepa2-paired-analysis-481826-"
        f"{str(protocol['controller']['commit'])[:7]}"
    )
    if job_name != expected_name:
        raise AnalysisRecoveryError("analysis recovery job name differs")
    expected_tokens = expected_analysis_submit_argv(
        protocol=protocol, job_name=job_name
    )
    tokens = [str(token) for token in submit_line_tokens]
    if tokens != expected_tokens:
        raise AnalysisRecoveryError(
            "analysis recovery SubmitLine token vector differs"
        )
    return timing_recovery.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_SUBMISSION,
            "submitted_at_utc": timing_recovery.utc_now(),
            "protocol": timing_recovery.file_record(
                protocol_path,
                identity_sha256=str(protocol["identity_sha256"]),
            ),
            "mode": RECOVERY_MODE,
            "previous_failed_analyzer_job_id": FAILED_ANALYZER_JOB_ID,
            "paired_latency": dict(
                protocol["timing_recovery"]["paired_latency"]
            ),
            "controller": dict(protocol["controller"]),
            "job": {
                "job_id": normalized,
                "job_name": job_name,
                "submitted_held": True,
                "receipt_written_before_release": True,
                "dependency": None,
                "scheduler": dict(protocol["scheduler"]),
                "submit_line_tokens": tokens,
                "submit_line_tokens_sha256": hashlib.sha256(
                    timing_recovery.canonical_json(tokens)
                ).hexdigest(),
            },
            "outputs": {
                "analysis_json": protocol["paths"]["analysis_json"],
                "analysis_markdown": protocol["paths"][
                    "analysis_markdown"
                ],
            },
        }
    )


def validate_submission(
    path: str | Path,
    *,
    protocol_path: str | Path,
    slurm_job_id: str,
    paired_latency: str | Path | None = None,
    timing_submission: str | Path | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    checked_protocol, protocol = validate_protocol(
        protocol_path,
        paired_latency=paired_latency,
        timing_submission=timing_submission,
        output_json=output_json,
        output_markdown=output_markdown,
    )
    try:
        submission_path = timing_recovery.canonical_file(
            path, "analysis recovery submission"
        )
        payload = timing_recovery.read_json(
            submission_path, "analysis recovery submission"
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    job = payload.get("job")
    outputs = payload.get("outputs")
    expected_name = (
        "vjepa2-paired-analysis-481826-"
        f"{str(protocol['controller']['commit'])[:7]}"
    )
    if (
        not timing_recovery.identity_valid(payload)
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != KIND_SUBMISSION
        or payload.get("mode") != RECOVERY_MODE
        or payload.get("previous_failed_analyzer_job_id")
        != FAILED_ANALYZER_JOB_ID
        or payload.get("paired_latency")
        != protocol["timing_recovery"]["paired_latency"]
        or payload.get("controller") != protocol["controller"]
        or not isinstance(job, Mapping)
        or timing_recovery.normalized_job_id(
            job.get("job_id"), "analysis recovery receipt job"
        )
        != timing_recovery.normalized_job_id(
            slurm_job_id, "analysis recovery runtime job"
        )
        or job.get("job_name") != expected_name
        or job.get("submitted_held") is not True
        or job.get("receipt_written_before_release") is not True
        or job.get("dependency") is not None
        or job.get("scheduler") != protocol["scheduler"]
        or job.get("submit_line_tokens")
        != expected_analysis_submit_argv(
            protocol=protocol, job_name=expected_name
        )
        or job.get("submit_line_tokens_sha256")
        != hashlib.sha256(
            timing_recovery.canonical_json(
                job.get("submit_line_tokens")
            )
        ).hexdigest()
        or outputs
        != {
            "analysis_json": protocol["paths"]["analysis_json"],
            "analysis_markdown": protocol["paths"][
                "analysis_markdown"
            ],
        }
    ):
        raise AnalysisRecoveryError(
            "analysis recovery submission identity/contract differs"
        )
    try:
        timing_recovery.validate_file_record(
            payload.get("protocol"),
            checked_protocol,
            label="analysis recovery protocol",
            identity_sha256=str(protocol["identity_sha256"]),
        )
    except timing_recovery.RecoveryError as exc:
        raise _convert_error(exc) from exc
    if submission_path != Path(protocol["paths"]["submission"]):
        raise AnalysisRecoveryError(
            "analysis recovery submission path differs"
        )
    return submission_path, payload, checked_protocol, protocol


def validate_current_accounting(
    row: str,
    *,
    submission: Mapping[str, Any],
) -> dict[str, str]:
    rows = [line for line in row.splitlines() if line.strip()]
    if len(rows) != 1:
        raise AnalysisRecoveryError(
            "analysis job accounting must contain exactly one row"
        )
    fields = timing_recovery.FAILED_ACCOUNTING_FIELDS
    values = rows[0].split("|")
    if len(values) != len(fields):
        raise AnalysisRecoveryError(
            "analysis job accounting row has the wrong schema"
        )
    record = dict(zip(fields, values))
    try:
        tokens = shlex.split(record["SubmitLine"])
    except ValueError as exc:
        raise AnalysisRecoveryError(
            "analysis job SubmitLine is invalid shell syntax"
        ) from exc
    req_tres = set(record["ReqTRES"].split(","))
    job = submission["job"]
    if (
        timing_recovery.normalized_job_id(
            record["JobIDRaw"], "analysis job"
        )
        != job["job_id"]
        or record["JobName"] != job["job_name"]
        or record["Partition"] != timing_recovery.PARTITION
        or record["Account"] != timing_recovery.ACCOUNT
        or record["QOS"] != timing_recovery.QOS
        or record["ReqCPUS"] != str(ANALYSIS_CPUS)
        or record["ReqMem"] != ANALYSIS_MEMORY
        or record["ReqNodes"] != "1"
        or not {
            f"cpu={ANALYSIS_CPUS}",
            f"mem={ANALYSIS_MEMORY}",
            "node=1",
            f"gres/gpu={ANALYSIS_GPUS}",
        }.issubset(req_tres)
        or record["State"] not in {"RUNNING", "COMPLETING"}
        or record["ExitCode"] != "0:0"
        or record["Timelimit"] != ANALYSIS_TIME_LIMIT
        or Path(record["WorkDir"])
        != Path(submission["controller"]["root"])
        or not record["NodeList"].strip()
        or tokens != job["submit_line_tokens"]
        or hashlib.sha256(
            timing_recovery.canonical_json(tokens)
        ).hexdigest()
        != job["submit_line_tokens_sha256"]
    ):
        raise AnalysisRecoveryError(
            "analysis recovery job accounting differs"
        )
    return record


def analysis_record(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    submission_path: Path,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    timing = protocol["timing_recovery"]
    return {
        "mode": RECOVERY_MODE,
        "failed_analyzer_job_id": FAILED_ANALYZER_JOB_ID,
        "protocol": timing_recovery.file_record(
            protocol_path,
            identity_sha256=str(protocol["identity_sha256"]),
        ),
        "submission": timing_recovery.file_record(
            submission_path,
            identity_sha256=str(submission["identity_sha256"]),
        ),
        "analyzer_job_id": submission["job"]["job_id"],
        "analyzer_commit": protocol["controller"]["commit"],
        "timing_recovery_root": timing["root"],
        "paired_latency": dict(timing["paired_latency"]),
        "r1_preserved": True,
        "timing_reexecuted": False,
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--controller-repo-root", required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--failed-accounting-row", required=True)
    parser.add_argument("--python", required=True)


def _command_preflight(args: argparse.Namespace) -> int:
    args.require_recovery_absent = True
    protocol = build_protocol(args)
    print(
        json.dumps(
            {
                "status": "paired_analysis_recovery_preflight_passed",
                "failed_analyzer_job_id": FAILED_ANALYZER_JOB_ID,
                "paired_latency_sha256": protocol["timing_recovery"][
                    "paired_latency"
                ]["sha256"],
                "recovery_root": protocol["recovery_root"],
                "timing_reexecution_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_write_protocol(args: argparse.Namespace) -> int:
    args.require_recovery_absent = False
    protocol = build_protocol(args)
    output = Path(args.output).expanduser()
    if output != Path(protocol["paths"]["protocol"]):
        raise AnalysisRecoveryError("analysis protocol output path differs")
    if output.exists():
        raise AnalysisRecoveryError("analysis protocol already exists")
    timing_recovery.exclusive_json(output, protocol)
    print(
        json.dumps(
            {
                "status": "paired_analysis_recovery_protocol_written",
                "identity_sha256": protocol["identity_sha256"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _command_write_submission(args: argparse.Namespace) -> int:
    protocol_path, protocol = validate_protocol(args.protocol)
    output = Path(args.output).expanduser()
    if output != Path(protocol["paths"]["submission"]):
        raise AnalysisRecoveryError(
            "analysis submission output path differs"
        )
    if output.exists():
        raise AnalysisRecoveryError("analysis submission already exists")
    submission = build_submission(
        protocol_path,
        protocol,
        job_id=args.job_id,
        job_name=args.job_name,
        submit_line_tokens=args.submit_line_token,
    )
    timing_recovery.exclusive_json(output, submission)
    print(
        json.dumps(
            {
                "status": "paired_analysis_recovery_submission_written",
                "job_id": submission["job"]["job_id"],
                "identity_sha256": submission["identity_sha256"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _command_validate_job(args: argparse.Namespace) -> int:
    _, submission, _, _ = validate_submission(
        args.submission,
        protocol_path=args.protocol,
        slurm_job_id=args.slurm_job_id,
        paired_latency=args.paired_latency,
        timing_submission=args.timing_submission,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    accounting = validate_current_accounting(
        args.current_accounting_row,
        submission=submission,
    )
    print(
        json.dumps(
            {
                "status": "paired_analysis_recovery_job_validated",
                "job_id": submission["job"]["job_id"],
                "node": accounting["NodeList"],
                "gpus": ANALYSIS_GPUS,
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _common_arguments(preflight)
    preflight.set_defaults(handler=_command_preflight)
    write = commands.add_parser("write-protocol")
    _common_arguments(write)
    write.add_argument("--output", required=True)
    write.set_defaults(handler=_command_write_protocol)
    submission = commands.add_parser("write-submission")
    submission.add_argument("--protocol", required=True)
    submission.add_argument("--job-id", required=True)
    submission.add_argument("--job-name", required=True)
    submission.add_argument(
        "--submit-line-token",
        action="append",
        default=[],
        required=True,
    )
    submission.add_argument("--output", required=True)
    submission.set_defaults(handler=_command_write_submission)
    validate_job = commands.add_parser("validate-job")
    validate_job.add_argument("--protocol", required=True)
    validate_job.add_argument("--submission", required=True)
    validate_job.add_argument("--slurm-job-id", required=True)
    validate_job.add_argument("--current-accounting-row", required=True)
    validate_job.add_argument("--paired-latency", required=True)
    validate_job.add_argument("--timing-submission", required=True)
    validate_job.add_argument("--output-json", required=True)
    validate_job.add_argument("--output-markdown", required=True)
    validate_job.set_defaults(handler=_command_validate_job)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (AnalysisRecoveryError, timing_recovery.RecoveryError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
