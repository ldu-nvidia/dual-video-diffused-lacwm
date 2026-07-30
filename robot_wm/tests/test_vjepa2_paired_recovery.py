"""Focused tests for the fail-closed paired-latency recovery."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tools import analyze_vjepa2_controlled_study as analysis
from tools import benchmark_vjepa2_paired_latency as benchmark
from tools import vjepa2_paired_recovery as recovery


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT / "tools/slurm/submit_vjepa2_paired_latency_recovery.sh"
)
SBATCH = ROOT / "tools/slurm/vjepa2_paired_latency_recovery.sbatch"


def _minimal_protocol(tmp_path: Path) -> tuple[Path, dict]:
    controller = tmp_path / "controller"
    scientific = tmp_path / "scientific"
    study = tmp_path / recovery.STUDY_ID
    recovery_root = (
        tmp_path
        / "_paired_recoveries"
        / recovery.STUDY_ID
        / "paired-481133-abcdef1-r1"
    )
    for path in (controller, scientific, study, recovery_root):
        path.mkdir(parents=True, exist_ok=True)
    protocol_path = recovery_root / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    paths = recovery.recovery_paths(study, recovery_root)
    protocol = {
        "identity_sha256": "1" * 64,
        "recovery_root": str(recovery_root),
        "study": {"path": str(study / "study_manifest.json")},
        "controller": {
            "root": str(controller),
            "commit": "abcdef1" + "0" * 33,
            "validator_fix_is_ancestor": True,
        },
        "scientific_repository": {
            "root": str(scientific),
            "commit": recovery.TRAINING_COMMIT,
            "matches_study_repository": True,
        },
        "runtime": {
            "python": "/runtime/bin/python",
            "python_canonical": "/runtime/bin/python3.10",
            "wan_dir": "/runtime/wan",
            "videox_home": "/runtime/videox",
            "videox_commit": "1" * 40,
        },
        "scheduler": {
            "partition": recovery.PARTITION,
            "account": recovery.ACCOUNT,
            "qos": recovery.QOS,
            "nodes": recovery.NODES,
            "tasks": 1,
            "tasks_per_node": 1,
            "gpus_per_node": recovery.GPUS,
            "cpus_per_task": recovery.CPUS,
            "memory": recovery.MEMORY,
            "time_limit": recovery.TIME_LIMIT,
            "requeue": False,
            "dependency": None,
            "submitted_held_until_receipt": True,
        },
        "paths": paths,
    }
    return protocol_path, protocol


def _accounting_row(
    *,
    job_id: str,
    job_name: str,
    state: str,
    exit_code: str,
    elapsed: str,
    submit_tokens: list[str],
    workdir: Path,
) -> str:
    values = {
        "JobIDRaw": job_id,
        "JobName": job_name,
        "Partition": recovery.PARTITION,
        "Account": recovery.ACCOUNT,
        "QOS": recovery.QOS,
        "ReqCPUS": str(recovery.CPUS),
        "ReqMem": recovery.MEMORY,
        "ReqNodes": str(recovery.NODES),
        "ReqTRES": "billing=8,cpu=32,gres/gpu=1,mem=256G,node=1",
        "State": state,
        "ExitCode": exit_code,
        "Elapsed": elapsed,
        "Timelimit": recovery.TIME_LIMIT,
        "SubmitLine": shlex.join(submit_tokens),
        "WorkDir": str(workdir),
        "NodeList": "pool0-0173",
        "Submit": "2026-07-30T01:16:06",
        "Start": "2026-07-30T08:55:41",
        "End": "2026-07-30T08:56:49",
    }
    return "|".join(values[field] for field in recovery.FAILED_ACCOUNTING_FIELDS)


def test_exact_failed_accounting_and_submit_line_are_required(tmp_path):
    study = tmp_path / recovery.STUDY_ID
    scientific = tmp_path / "scientific"
    python = tmp_path / "env/bin/python"
    wan = tmp_path / "wan"
    videox = tmp_path / "videox"
    submit = recovery.expected_original_submit_argv(
        study_root=study,
        scientific_repo=scientific,
        python=python,
        wan_dir=wan,
        videox_home=videox,
    )
    name = f"vjepa2-{recovery.STUDY_ID[:49]}-paired-latency"
    row = _accounting_row(
        job_id=recovery.FAILED_JOB_ID,
        job_name=name,
        state="FAILED",
        exit_code=recovery.FAILED_EXIT_CODE,
        elapsed=recovery.FAILED_ELAPSED,
        submit_tokens=submit,
        workdir=scientific,
    )
    result = recovery.validate_failed_accounting(
        row,
        study_root=study,
        scientific_repo=scientific,
        python=python,
        wan_dir=wan,
        videox_home=videox,
    )
    assert shlex.split(result["SubmitLine"]) == submit
    tampered = row.replace("--gpus-per-node=1", "--gpus-per-node=2")
    with pytest.raises(recovery.RecoveryError, match="SubmitLine differs"):
        recovery.validate_failed_accounting(
            tampered,
            study_root=study,
            scientific_repo=scientific,
            python=python,
            wan_dir=wan,
            videox_home=videox,
        )


def test_final_array_requires_exact_five_successful_tasks():
    name = f"vjepa2-{recovery.STUDY_ID[:54]}-u1000"
    rows = [
        f"{recovery.FINAL_STAGE_JOB_ID}_{task}|COMPLETED|0:0|{name}"
        for task in range(5)
    ]
    result = recovery.validate_final_accounting(rows)
    assert [item["array_task_id"] for item in result] == list(range(5))
    with pytest.raises(recovery.RecoveryError, match="task set differs"):
        recovery.validate_final_accounting(rows[:-1])
    with pytest.raises(recovery.RecoveryError, match="accounting differs"):
        recovery.validate_final_accounting(
            [*rows[:-1], rows[-1].replace("COMPLETED", "FAILED")]
        )


def test_recovery_root_binds_failed_job_controller_and_attempt(tmp_path):
    study = tmp_path / recovery.STUDY_ID
    study.mkdir()
    commit = "abcdef1" + "0" * 33
    wanted = recovery.expected_recovery_root(
        study,
        "paired-481133-abcdef1-r2",
        commit,
    )
    assert wanted == (
        study.parent
        / "_paired_recoveries"
        / recovery.STUDY_ID
        / "paired-481133-abcdef1-r2"
    )
    with pytest.raises(recovery.RecoveryError):
        recovery.expected_recovery_root(
            study,
            "paired-481133-deadbee-r2",
            commit,
        )


def test_runtime_assets_are_content_and_videox_commit_bound(
    tmp_path,
    monkeypatch,
):
    wan = tmp_path / "wan"
    videox = tmp_path / "videox"
    wan.mkdir()
    videox.mkdir()
    content = b"pinned-model-asset"
    asset = wan / "weight.bin"
    asset.write_bytes(content)
    monkeypatch.setattr(
        recovery,
        "WAN_ASSET_SPECS",
        {
            asset.name: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        },
    )

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return recovery.VIDEOX_COMMIT
        if arguments == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(recovery, "_git", fake_git)
    result = recovery.validate_runtime_assets(wan, videox)
    assert result["wan"]["files"][asset.name]["bytes"] == len(content)
    assert result["videox"]["commit"] == recovery.VIDEOX_COMMIT
    asset.write_bytes(content + b"-tampered")
    with pytest.raises(recovery.RecoveryError, match="Wan asset differs"):
        recovery.validate_runtime_assets(wan, videox)


def test_submission_binds_exact_held_sbatch_tokens(tmp_path):
    protocol_path, protocol = _minimal_protocol(tmp_path)
    job_name = "vjepa2-paired-recovery-481133-abcdef1"
    tokens = recovery.expected_recovery_submit_argv(
        protocol=protocol,
        job_name=job_name,
    )
    result = recovery.build_submission(
        protocol_path,
        protocol,
        job_id="999001",
        job_name=job_name,
        submit_line_tokens=tokens,
    )
    assert result["job"]["submitted_held"] is True
    assert result["job"]["dependency"] is None
    assert result["job"]["submit_line_tokens"] == tokens
    assert result["job"]["submit_line_tokens_sha256"] == hashlib.sha256(
        recovery.canonical_json(tokens)
    ).hexdigest()
    with pytest.raises(recovery.RecoveryError, match="token vector differs"):
        recovery.build_submission(
            protocol_path,
            protocol,
            job_id="999001",
            job_name=job_name,
            submit_line_tokens=[*tokens, "--dependency=afterok:481132"],
        )


def test_current_job_accounting_is_bound_to_receipt(tmp_path):
    protocol_path, protocol = _minimal_protocol(tmp_path)
    name = "vjepa2-paired-recovery-481133-abcdef1"
    tokens = recovery.expected_recovery_submit_argv(
        protocol=protocol,
        job_name=name,
    )
    submission = recovery.build_submission(
        protocol_path,
        protocol,
        job_id="999001",
        job_name=name,
        submit_line_tokens=tokens,
    )
    row = _accounting_row(
        job_id="999001",
        job_name=name,
        state="RUNNING",
        exit_code="0:0",
        elapsed="00:00:10",
        submit_tokens=tokens,
        workdir=Path(protocol["controller"]["root"]),
    )
    result = recovery.validate_current_accounting(row, submission=submission)
    assert result["Account"] == recovery.ACCOUNT
    assert result["QOS"] == recovery.QOS
    with pytest.raises(recovery.RecoveryError, match="accounting differs"):
        recovery.validate_current_accounting(
            row.replace("|normal|", "|interactive|"),
            submission=submission,
        )


def test_runtime_evidence_records_actual_job_and_submit_line(
    tmp_path,
    monkeypatch,
):
    protocol_path, protocol = _minimal_protocol(tmp_path)
    assets = {
        "wan": {"root": "/runtime/wan", "files": {}, "files_sha256": "0" * 64},
        "videox": {
            "root": "/runtime/videox",
            "commit": recovery.VIDEOX_COMMIT,
            "clean": True,
        },
    }
    protocol["runtime"]["assets"] = assets
    monkeypatch.setattr(
        recovery,
        "validate_runtime_assets",
        lambda _wan, _videox: assets,
    )
    name = "vjepa2-paired-recovery-481133-abcdef1"
    tokens = recovery.expected_recovery_submit_argv(
        protocol=protocol,
        job_name=name,
    )
    submission = recovery.build_submission(
        protocol_path,
        protocol,
        job_id="999001",
        job_name=name,
        submit_line_tokens=tokens,
    )
    submission_path = Path(protocol["paths"]["submission"])
    submission_path.write_text(
        json.dumps(submission, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    row = _accounting_row(
        job_id="999001",
        job_name=name,
        state="RUNNING",
        exit_code="0:0",
        elapsed="00:00:10",
        submit_tokens=tokens,
        workdir=Path(protocol["controller"]["root"]),
    )
    evidence = recovery.build_runtime_evidence(
        protocol_path=protocol_path,
        protocol=protocol,
        submission_path=submission_path,
        submission=submission,
        current_accounting_row=row,
        slurm_node="pool0-0173",
        cuda_visible_devices="0",
    )
    assert recovery.identity_valid(evidence)
    assert evidence["job_id"] == "999001"
    assert evidence["scheduler_contract_validated"] is True
    assert evidence["accounting"]["SubmitLine"] == shlex.join(tokens)
    assert evidence["runtime_assets_before_model_loading"] == assets


def test_write_submission_cli_accepts_option_looking_tokens(
    tmp_path,
    monkeypatch,
):
    protocol_path, protocol = _minimal_protocol(tmp_path)
    name = "vjepa2-paired-recovery-481133-abcdef1"
    tokens = recovery.expected_recovery_submit_argv(
        protocol=protocol,
        job_name=name,
    )
    monkeypatch.setattr(
        recovery,
        "validate_protocol",
        lambda _path: (protocol_path, protocol),
    )
    output = Path(protocol["paths"]["submission"])
    result = recovery.main(
        [
            "write-submission",
            "--protocol",
            str(protocol_path),
            "--job-id",
            "999001",
            "--job-name",
            name,
            "--output",
            str(output),
            *[
                f"--submit-line-token={token}"
                for token in tokens
            ],
        ]
    )
    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["job"]["submit_line_tokens"] == tokens
    assert "--parsable" in payload["job"]["submit_line_tokens"]


def test_analyzer_recovery_overrides_are_all_or_none(tmp_path):
    with pytest.raises(
        analysis.StudyValidationError,
        match="overrides are both required",
    ):
        analysis.analyze(
            tmp_path / "missing-study",
            output_json=tmp_path / "analysis.json",
            paired_latency=tmp_path / "paired.json",
        )


def test_analyzer_outputs_are_bound_to_recovery_receipt(tmp_path):
    receipt = recovery.identity_payload(
        {
            "outputs": {
                "analysis_json": str(tmp_path / "bound" / "analysis.json"),
                "analysis_markdown": str(tmp_path / "bound" / "analysis.md"),
            }
        }
    )
    receipt_path = tmp_path / "submission.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        analysis.StudyValidationError,
        match="outputs differ",
    ):
        analysis.analyze(
            tmp_path / "missing-study",
            output_json=tmp_path / "unbound.json",
            paired_latency=tmp_path / "paired.json",
            paired_recovery_submission=receipt_path,
        )


def test_complete_scientific_module_and_class_provenance(monkeypatch):
    import robot_wm

    project = ROOT / "projects" / "latent_action_models"
    lam_module = types.ModuleType("lam")
    lam_module.__file__ = str(project / "lam/__init__.py")
    monkeypatch.setitem(sys.modules, "lam", lam_module)
    loaded = benchmark._loaded_scientific_module_origins(
        repo=ROOT,
        project_root=project,
    )
    robot_file = str(Path(robot_wm.__file__).resolve(strict=True))
    lam_file = str(Path(lam_module.__file__).resolve(strict=True))
    value = {
        "bootstrap": {"robot_wm": robot_file, "lam": lam_file},
        "instantiated_classes": {
            arm: {
                "dataset": {
                    "qualified_name": "robot_wm.TestDataset",
                    "module": "robot_wm",
                    "file": robot_file,
                },
                "model": {
                    "qualified_name": "lam.TestModel",
                    "module": "lam",
                    "file": lam_file,
                },
            }
            for arm in ("J1", "VPM")
        },
        "before_timing": loaded,
        "after_timing": loaded,
    }
    validated = analysis._validate_scientific_import_origins(
        value,
        scientific_root=ROOT,
    )
    assert validated["after_timing"]["origins_sha256"] == loaded[
        "origins_sha256"
    ]
    tampered = {
        **value,
        "after_timing": {
            **loaded,
            "origins_sha256": "0" * 64,
        },
    }
    with pytest.raises(
        analysis.StudyValidationError,
        match="module map/hash differs",
    ):
        analysis._validate_scientific_import_origins(
            tampered,
            scientific_root=ROOT,
        )

    escaped = types.ModuleType("lam.controller_escape")
    escaped.__file__ = str(ROOT / "tools/README.md")
    monkeypatch.setitem(sys.modules, "lam.controller_escape", escaped)
    with pytest.raises(
        benchmark.PairedLatencyError,
        match="escaped the scientific repository",
    ):
        benchmark._loaded_scientific_module_origins(
            repo=ROOT,
            project_root=project,
        )


def test_scientific_paths_are_repromoted_ahead_of_controller(
    tmp_path,
):
    controller = tmp_path / "controller"
    science = tmp_path / "science"
    project = science / "projects/latent_action_models"
    for package in (
        controller / "robot_wm",
        science / "robot_wm",
        project / "lam",
    ):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f"ORIGIN = {str(package)!r}\n",
            encoding="utf-8",
        )
    code = f"""
import sys
from pathlib import Path
from tools import benchmark_vjepa2_paired_latency as benchmark
controller = {str(controller)!r}
science = Path({str(science)!r})
project = Path({str(project)!r})
sys.path.insert(0, controller)
sys.path.extend([str(project), str(science)])
benchmark._promote_scientific_paths(science, project)
import lam
import robot_wm
assert Path(robot_wm.__file__).resolve().is_relative_to(science)
assert Path(lam.__file__).resolve().is_relative_to(project)
assert sys.path[0:2] == [str(project), str(science)]
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_benchmark_mode_remains_default():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            "/repo",
            "--expected-commit",
            recovery.TRAINING_COMMIT,
            "--study-root",
            "/study",
            "--submission-record",
            "/study/slurm_submission.json",
            "--slurm-job-id",
            recovery.FAILED_JOB_ID,
            "--output",
            "/study/paired_latency/"
            + recovery.OUTPUT_BASENAME,
        ]
    )
    assert args.recovery_protocol is None
    assert args.recovery_submission_record is None
    assert args.recovery_runtime_record is None
    assert args.controller_commit is None


def test_recovery_shell_entrypoints_are_fail_closed():
    for script in (LAUNCHER, SBATCH):
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    launcher = LAUNCHER.read_text(encoding="utf-8")
    worker = SBATCH.read_text(encoding="utf-8")
    assert "--hold" in launcher
    assert "write-submission" in launcher
    assert "scontrol release" in launcher
    assert launcher.index("write-submission") < launcher.index(
        "scontrol release"
    )
    assert "--dependency" not in launcher
    assert "_paired_recoveries" in launcher
    assert 'SUBMISSION_ARGS+=("--submit-line-token=$token")' in launcher
    assert "/vjepa2-paired-recovery-481133-/" in launcher
    assert "current-accounting-row" in worker
    assert "recovery-runtime-record" in worker
    assert "--paired-latency" in worker
    assert "--paired-recovery-submission" in worker
    assert worker.count("git -C \"$VIDEOX_HOME_VALUE\" rev-parse HEAD") == 2
    assert worker.count('"$PYTHON_BIN" "$VERIFY_RUNTIME"') == 2
    assert 'cd "$SCIENTIFIC_ROOT"' in worker


def test_timing_loop_is_still_the_preregistered_implementation():
    source = inspect.getsource(benchmark.command_benchmark)
    assert "range(WARMUP_PAIRS)" in source
    assert "range(TIMED_PAIRS)" in source
    assert "counterbalanced_order(pair_index)" in source
    assert "sample_future_deployable(" in source
    assert "collect_artifacts=False" in source
    assert "loaded robot_wm/lam modules escaped" in source
    assert '"scientific_import_origins"' in source
    assert '"runtime_assets_after_timing"' in source


def test_controller_pins_unchanged_inference_benchmark_dependency():
    source = inspect.getsource(recovery.build_protocol)
    assert 'dependency_path = "tools/benchmark_vjepa2_inference.py"' in source
    assert "training_dependency_blob != controller_dependency_blob" in source
    assert 'verifier_path = "tools/env/verify_b200_runtime.py"' in source
    assert "training_verifier_blob != controller_verifier_blob" in source
