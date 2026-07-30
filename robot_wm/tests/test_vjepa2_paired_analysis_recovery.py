"""Focused tests for the analyzer-only recovery of paired job 481826."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from tools import analyze_vjepa2_controlled_study as analysis
from tools import vjepa2_paired_analysis_recovery as recovery
from tools import vjepa2_paired_recovery as timing_recovery


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT / "tools/slurm/submit_vjepa2_paired_analysis_recovery.sh"
)
WORKER = ROOT / "tools/slurm/vjepa2_paired_analysis_recovery.sbatch"


def _protocol(tmp_path: Path) -> dict:
    controller = tmp_path / "controller"
    study = tmp_path / timing_recovery.STUDY_ID
    recovery_root = (
        study.parent
        / "_paired_recoveries"
        / timing_recovery.STUDY_ID
        / "paired-analysis-481826-abcdef1-r2"
    )
    paths = recovery.recovery_paths(recovery_root)
    protocol = {
        "identity_sha256": "1" * 64,
        "recovery_root": str(recovery_root),
        "study": {"path": str(study / "study_manifest.json")},
        "controller": {
            "root": str(controller),
            "commit": "abcdef1" + "0" * 33,
        },
        "runtime": {"python": "/runtime/bin/python"},
        "scheduler": {
            "partition": timing_recovery.PARTITION,
            "account": timing_recovery.ACCOUNT,
            "qos": timing_recovery.QOS,
            "nodes": 1,
            "tasks": 1,
            "tasks_per_node": 1,
            "gpus_per_node": recovery.ANALYSIS_GPUS,
            "cpus_per_task": recovery.ANALYSIS_CPUS,
            "memory": recovery.ANALYSIS_MEMORY,
            "time_limit": recovery.ANALYSIS_TIME_LIMIT,
            "requeue": False,
            "submitted_held_until_receipt": True,
        },
        "timing_recovery": {
            "paired_latency": {
                "path": "/immutable/paired.json",
                "sha256": recovery.PINNED_FILES["paired_latency"]["sha256"],
                "bytes": recovery.PINNED_FILES["paired_latency"]["bytes"],
                "identity_sha256": recovery.PAIRED_IDENTITY,
            },
            "submission": {"path": "/immutable/submission.json"},
        },
        "paths": paths,
    }
    protocol_path = Path(paths["protocol"])
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(
        json.dumps(protocol, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return protocol


def _accounting_row(submission: dict, *, state: str = "RUNNING") -> str:
    job = submission["job"]
    values = {
        "JobIDRaw": job["job_id"],
        "JobName": job["job_name"],
        "Partition": timing_recovery.PARTITION,
        "Account": timing_recovery.ACCOUNT,
        "QOS": timing_recovery.QOS,
        "ReqCPUS": str(recovery.ANALYSIS_CPUS),
        "ReqMem": recovery.ANALYSIS_MEMORY,
        "ReqNodes": "1",
        "ReqTRES": (
            "billing=8,"
            f"cpu={recovery.ANALYSIS_CPUS},"
            f"gres/gpu={recovery.ANALYSIS_GPUS},"
            f"mem={recovery.ANALYSIS_MEMORY},node=1"
        ),
        "State": state,
        "ExitCode": "0:0",
        "Elapsed": "00:00:05",
        "Timelimit": recovery.ANALYSIS_TIME_LIMIT,
        "SubmitLine": shlex.join(job["submit_line_tokens"]),
        "WorkDir": submission["controller"]["root"],
        "NodeList": "pool0-0001",
        "Submit": "2026-07-30T11:00:00",
        "Start": "2026-07-30T11:00:01",
        "End": "Unknown",
    }
    return "|".join(
        values[field] for field in timing_recovery.FAILED_ACCOUNTING_FIELDS
    )


def test_analysis_recovery_root_binds_failed_job_commit_and_attempt(tmp_path):
    study = tmp_path / timing_recovery.STUDY_ID
    study.mkdir()
    commit = "abcdef1" + "0" * 33
    assert recovery.expected_recovery_root(
        study,
        "paired-analysis-481826-abcdef1-r2",
        commit,
    ).name == "paired-analysis-481826-abcdef1-r2"
    with pytest.raises(recovery.AnalysisRecoveryError):
        recovery.expected_recovery_root(
            study,
            "paired-analysis-481826-deadbee-r2",
            commit,
        )


def test_submission_is_held_and_binds_exact_analyzer_tokens(tmp_path):
    protocol = _protocol(tmp_path)
    protocol_path = Path(protocol["paths"]["protocol"])
    name = "vjepa2-paired-analysis-481826-abcdef1"
    tokens = recovery.expected_analysis_submit_argv(
        protocol=protocol, job_name=name
    )
    submission = recovery.build_submission(
        protocol_path,
        protocol,
        job_id="999101",
        job_name=name,
        submit_line_tokens=tokens,
    )
    assert submission["job"]["submitted_held"] is True
    assert submission["job"]["dependency"] is None
    assert submission["job"]["scheduler"]["gpus_per_node"] == 1
    assert submission["paired_latency"]["sha256"].startswith("314a4009")
    assert submission["job"]["submit_line_tokens_sha256"] == hashlib.sha256(
        timing_recovery.canonical_json(tokens)
    ).hexdigest()
    with pytest.raises(
        recovery.AnalysisRecoveryError, match="token vector differs"
    ):
        recovery.build_submission(
            protocol_path,
            protocol,
            job_id="999101",
            job_name=name,
            submit_line_tokens=[*tokens, "--gpus-per-node=2"],
        )


def test_live_analyzer_job_accounting_is_receipt_bound(tmp_path):
    protocol = _protocol(tmp_path)
    name = "vjepa2-paired-analysis-481826-abcdef1"
    tokens = recovery.expected_analysis_submit_argv(
        protocol=protocol, job_name=name
    )
    submission = recovery.build_submission(
        Path(protocol["paths"]["protocol"]),
        protocol,
        job_id="999101",
        job_name=name,
        submit_line_tokens=tokens,
    )
    result = recovery.validate_current_accounting(
        _accounting_row(submission),
        submission=submission,
    )
    assert result["ReqTRES"].split(",").count("gres/gpu=1") == 1
    with pytest.raises(
        recovery.AnalysisRecoveryError, match="accounting differs"
    ):
        recovery.validate_current_accounting(
            _accounting_row(submission).replace(
                "gres/gpu=1", "gres/gpu=2"
            ),
            submission=submission,
        )


def test_analyzer_recovery_submission_and_job_are_all_or_none(tmp_path):
    with pytest.raises(
        analysis.StudyValidationError,
        match="submission and job ID are both required",
    ):
        analysis.analyze(
            tmp_path / "missing-study",
            output_json=tmp_path / "analysis.json",
            paired_analysis_recovery_submission=tmp_path
            / "submission.json",
        )


def test_analyzer_only_shell_is_held_fail_closed_and_has_no_benchmark():
    for script in (LAUNCHER, WORKER):
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    launcher = LAUNCHER.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    assert "--hold" in launcher
    assert "write-submission" in launcher
    assert launcher.index("write-submission") < launcher.index(
        "scontrol release"
    )
    assert "squeue -h" in launcher
    assert "status --porcelain --untracked-files=all" in launcher
    assert "--gpus-per-node=1" in launcher
    assert "--gpus-per-node=1" in worker
    assert "validate-job" in worker
    assert "--paired-analysis-recovery-submission" in worker
    for forbidden in (
        "benchmark_vjepa2_paired_latency.py",
        "benchmark_vjepa2_inference.py",
        "activate_b200.sh",
        "sample_future_deployable",
        "--future-rgb",
        "--teacher",
    ):
        assert forbidden not in worker
        assert forbidden not in launcher
    assert "torch.load" not in worker
    assert "snapshot.pt" not in worker


def test_pinned_r1_pair_is_exact_completed_timing_artifact():
    pair = recovery.PINNED_FILES["paired_latency"]
    assert pair == {
        "relative": (
            "paired_latency/paired_j1_nfe4_vs_vpm_nfe8.json"
        ),
        "bytes": 75_578,
        "sha256": (
            "314a40090a96cd06c6e331f833b689aa06059ba1988cb83955ab9a2b9733d84c"
        ),
        "identity_sha256": recovery.PAIRED_IDENTITY,
    }
    assert json.loads(
        json.dumps(pair, sort_keys=True)
    ) == pair
