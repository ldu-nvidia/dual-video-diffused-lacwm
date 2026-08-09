"""Static resource, authorization, and no-launch-default tests for ACD-P0."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import low_nfe_direct_baseline as pilot
from tools.slurm import adjacent_consistency_workflow as workflow


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "tools" / "slurm"
LUSTRE_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/"
    "users/ldu/lacwm_train"
)


def _authorization(
    *,
    phase: str = "train",
    arm: str | None = "ACD-CONS",
    source_commit: str = "a" * 40,
    registration_identity: str | None = "b" * 64,
    receipt: Path = Path("/mnt/data1/acd-test/submission.json"),
    created: datetime | None = None,
    expires: datetime | None = None,
) -> dict:
    created = created or datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    expires = expires or created + timedelta(hours=1)
    return pilot.identity_payload(
        {
            "schema_version": workflow.SCHEMA_VERSION,
            "kind": workflow.AUTHORIZATION_KIND,
            "created_at_utc": created.isoformat(),
            "expires_at_utc": expires.isoformat(),
            "authorized_by": "user",
            "authorization_basis": "bounded ACD-P0 execution review",
            "slurm_submission_authorized": True,
            "phase": phase,
            "arm": arm,
            "source_commit": source_commit,
            "registration_identity_sha256": registration_identity,
            "submission_receipt_path": str(receipt),
            "accepted_parent_not_quality_dominant": True,
            "accepted_resource_plan": pilot.resource_plan(),
            "jobs_authorized": 1,
            "wandb_writes_authorized": 0,
            "protected_test_accessed": False,
        }
    )


@pytest.mark.parametrize(
    "filename,gpus,hours",
    (
        ("adjacent_consistency_memory_smoke.sbatch", 1, "01:00:00"),
        ("adjacent_consistency.sbatch", 8, "02:00:00"),
        ("adjacent_consistency_evaluate.sbatch", 8, "02:00:00"),
    ),
)
def test_slurm_resources_are_exact_b200_non_requeueable(
    filename: str, gpus: int, hours: str
) -> None:
    source = (SLURM / filename).read_text(encoding="utf-8")
    assert f"#SBATCH --gpus-per-node={gpus}" in source
    assert "#SBATCH --constraint=B200" in source
    assert f"#SBATCH --time={hours}" in source
    assert "#SBATCH --no-requeue" in source
    assert "validate-submission" in source
    assert "--user-authorization" in source
    assert "--submission-receipt" in source
    assert 'approved_artifact_path "$LOG_DIR" "logs"' in source
    for approved_root in ("/mnt/data1", "/mnt/data2", str(LUSTRE_ROOT)):
        assert approved_root in source
    assert "/lustre/*" not in source
    executable_source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "sbatch " not in executable_source
    for forbidden in (
        "rm -",
        "rmdir ",
        "unlink",
        "scancel",
        "squeue",
        "sacct",
        "git reset",
        "git checkout",
    ):
        assert forbidden not in source


def _sbatch_directives(filename: str) -> dict[str, str]:
    directives = {}
    for line in (SLURM / filename).read_text(encoding="utf-8").splitlines():
        prefix = "#SBATCH --"
        if not line.startswith(prefix):
            continue
        key, separator, value = line[len(prefix) :].partition("=")
        if separator:
            directives[key] = value
    return directives


def _slurm_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


@pytest.mark.parametrize(
    "filename",
    (
        "adjacent_consistency_memory_smoke.sbatch",
        "adjacent_consistency.sbatch",
        "adjacent_consistency_evaluate.sbatch",
    ),
)
def test_each_acd_sbatch_time_fits_site_qos_and_partition_limits(
    filename: str,
) -> None:
    directives = _sbatch_directives(filename)
    assert directives["qos"] == "short"
    assert directives["partition"] == "batch"
    requested_seconds = _slurm_seconds(directives["time"])
    assert requested_seconds <= 2 * 3600  # short QoS MaxWall
    assert requested_seconds <= 4 * 3600  # batch partition MaxTime


def test_workflow_plan_has_two_two_hour_training_specs(
    monkeypatch, capsys
) -> None:
    registration = {
        "output_root": "/mnt/data1/acd-test",
        "tool_repository": {"git_commit": "a" * 40},
        "identity_sha256": "b" * 64,
    }
    monkeypatch.setattr(workflow, "_registration", lambda _path: registration)
    assert workflow.command_plan(
        argparse.Namespace(registration=Path("/unused/registration.json"))
    ) == 0
    rendered = json.loads(capsys.readouterr().out)
    train_specs = [
        spec for spec in rendered["phases"] if spec["phase"] == "train"
    ]
    assert [spec["arm"] for spec in train_specs] == [
        arm.code for arm in pilot.ARMS
    ]
    assert [spec["time_limit_hours"] for spec in train_specs] == [2, 2]
    assert rendered["resource_plan"] == pilot.resource_plan()


@pytest.mark.parametrize(
    "path",
    (
        Path("/mnt/data1/acd/run"),
        Path("/mnt/data2/acd/run"),
        LUSTRE_ROOT / "artifacts/dual_video_diffusion/acd/run",
    ),
)
def test_workflow_accepts_each_exact_site_artifact_root(path: Path) -> None:
    assert workflow._approved_artifact_path(path, "test artifact") == path


@pytest.mark.parametrize(
    "path",
    (
        Path("relative/acd/run"),
        Path("/lustre/acd/run"),
        Path(
            "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/"
            "users/another-user/lacwm_train/acd/run"
        ),
        LUSTRE_ROOT.parent / "another-project/acd/run",
        Path("/mnt/data10/acd/run"),
        Path("/mnt/data1/../etc/acd/run"),
    ),
)
def test_workflow_rejects_paths_outside_exact_site_roots(path: Path) -> None:
    with pytest.raises(workflow.ACDWorkflowError, match="approved artifact root"):
        workflow._approved_artifact_path(path, "test artifact")


@pytest.mark.parametrize(
    "filename,required_guards",
    (
        (
            "adjacent_consistency_memory_smoke.sbatch",
            ("$OUTPUT", "$LOG_DIR", "$USER_AUTHORIZATION", "$SUBMISSION_RECEIPT"),
        ),
        (
            "adjacent_consistency.sbatch",
            (
                "$LOG_DIR",
                "$REGISTRATION",
                "$USER_AUTHORIZATION",
                "$SUBMISSION_RECEIPT",
                "$OUTPUT_ROOT",
            ),
        ),
        (
            "adjacent_consistency_evaluate.sbatch",
            (
                "$LOG_DIR",
                "$REGISTRATION",
                "$USER_AUTHORIZATION",
                "$SUBMISSION_RECEIPT",
                "$OUTPUT_ROOT",
            ),
        ),
    ),
)
def test_wrappers_guard_outputs_logs_and_receipts_with_exact_site_allowlist(
    filename: str, required_guards: tuple[str, ...]
) -> None:
    source = (SLURM / filename).read_text(encoding="utf-8")
    for variable in required_guards:
        assert f'approved_artifact_path "{variable}"' in source
    assert "/mnt/data1|/mnt/data1/*" in source
    assert "/mnt/data2|/mnt/data2/*" in source
    assert str(LUSTRE_ROOT) in source
    assert "/lustre/*" not in source


def test_training_wrapper_binds_exact_registered_hydra_semantics() -> None:
    wrapper = (SLURM / "adjacent_consistency.sbatch").read_text(encoding="utf-8")
    registration = (ROOT / "tools/low_nfe_direct_baseline.py").read_text(
        encoding="utf-8"
    )
    for override in (
        "+experiments_0908=$CONFIG_NAME",
        "hydra.run.dir=$RUN_DIR",
        "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt",
        "trainer.config.visualization.viz_path=$RUN_DIR/visualization",
    ):
        assert override in wrapper
    assert "trainer.config.visualization.viz_path=" in registration
    assert 'export ACD_P0_REGISTRATION_PATH="$REGISTRATION"' in wrapper
    assert (
        'export ACD_P0_ARM_CONFIG_SEMANTIC_SHA256="$ARM_CONFIG_SEMANTIC_SHA256"'
        in wrapper
    )
    assert "ACD_P0_PARENT_CANONICAL_MODEL_STATE_SHA256" in wrapper
    assert "ACD_P0_PARENT_RUNTIME_TENSOR_STATE_SHA256" in wrapper
    assert "arm_config_semantic_sha256" in (
        SLURM / "adjacent_consistency_workflow.py"
    ).read_text(encoding="utf-8")


def test_authorization_is_identity_source_registration_and_job_bound() -> None:
    created = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    receipt = Path("/mnt/data1/acd-test/submission.json")
    payload = _authorization(created=created, receipt=receipt)
    validated = workflow.validate_user_authorization_payload(
        payload,
        phase="train",
        arm="ACD-CONS",
        source_commit="a" * 40,
        registration_identity_sha256="b" * 64,
        submission_receipt=receipt,
        at_time=created + timedelta(minutes=1),
    )
    assert validated["jobs_authorized"] == 1
    assert validated["accepted_resource_plan"] == pilot.resource_plan()

    changed = dict(payload)
    changed.pop("identity_sha256")
    changed["registration_identity_sha256"] = "c" * 64
    changed = pilot.identity_payload(changed)
    with pytest.raises(workflow.ACDWorkflowError, match="does not match"):
        workflow.validate_user_authorization_payload(
            changed,
            phase="train",
            arm="ACD-CONS",
            source_commit="a" * 40,
            registration_identity_sha256="b" * 64,
            submission_receipt=receipt,
            at_time=created + timedelta(minutes=1),
        )


def test_memory_authorization_is_source_bound_before_registration() -> None:
    created = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    receipt = Path("/mnt/data1/acd-memory/_slurm_logs/acd-p0-memory-smoke-submission.json")
    payload = _authorization(
        phase="memory_smoke",
        arm=None,
        registration_identity=None,
        receipt=receipt,
        created=created,
    )
    validated = workflow.validate_user_authorization_payload(
        payload,
        phase="memory_smoke",
        arm=None,
        source_commit="a" * 40,
        registration_identity_sha256=None,
        submission_receipt=receipt,
        at_time=created + timedelta(minutes=1),
    )
    assert validated["registration_identity_sha256"] is None


def test_expired_or_overlong_authorization_fails_closed() -> None:
    created = datetime(2026, 8, 1, tzinfo=timezone.utc)
    payload = _authorization(
        created=created,
        expires=created + timedelta(hours=25),
    )
    with pytest.raises(workflow.ACDWorkflowError, match="not fresh"):
        workflow.validate_user_authorization_payload(
            payload,
            phase="train",
            arm="ACD-CONS",
            source_commit="a" * 40,
            registration_identity_sha256="b" * 64,
            submission_receipt=Path("/mnt/data1/acd-test/submission.json"),
            at_time=created + timedelta(minutes=1),
        )


def test_default_phase_action_only_renders_and_never_submits(monkeypatch) -> None:
    created = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    payload = _authorization(created=created)
    spec = workflow.SubmissionSpec(
        phase="train",
        arm="ACD-CONS",
        source_commit="a" * 40,
        registration_identity_sha256="b" * 64,
        script=SLURM / "adjacent_consistency.sbatch",
        job_name="acd-p0-test",
        log_dir=Path("/mnt/data1/acd-test/_slurm_logs"),
        authorization=Path("/mnt/data1/acd-test/authorization.json"),
        submission_receipt=Path("/mnt/data1/acd-test/submission.json"),
        wrapper_argv=("--registration", "/mnt/data1/acd-test/registration.json"),
    )
    monkeypatch.setattr(
        workflow,
        "validate_user_authorization",
        lambda *_args, **_kwargs: (payload, {"sha256": "d" * 64}),
    )

    def forbidden_submit(*_args, **_kwargs):
        raise AssertionError("default plan attempted submission")

    monkeypatch.setattr(workflow, "_submit_held", forbidden_submit)
    result = workflow.execute_or_plan(spec, submit=False)
    assert result["jobs_submitted"] == 0
    assert result["mode"] == "plan_only_no_commands_executed_no_files_created"
    assert result["sbatch_argv"][1:3] == ["--hold", "--parsable"]


def test_submit_flag_is_separate_from_required_authorization() -> None:
    parser = workflow._parser()
    base = [
        "train",
        "--registration",
        "/mnt/data1/acd/registration.json",
        "--arm",
        "ACD-CONS",
        "--user-authorization",
        "/mnt/data1/acd/authorization.json",
    ]
    assert parser.parse_args(base).submit is False
    assert parser.parse_args([*base, "--submit"]).submit is True
    with pytest.raises(SystemExit):
        parser.parse_args(base[:-2])


def test_submit_path_checks_active_jobs_and_has_no_destructive_job_controls() -> None:
    source = (SLURM / "adjacent_consistency_workflow.py").read_text(
        encoding="utf-8"
    )
    submit_body = source[source.index("def _submit_held") : source.index("def execute_or_plan")]
    assert 'shutil.which("squeue")' in submit_body
    assert 'line.split("|", 2)[1] == spec.job_name' in submit_body
    assert "scancel" not in submit_body
    assert "requeue" not in submit_body
    assert "kill" not in submit_body
