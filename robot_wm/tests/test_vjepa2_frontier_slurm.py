"""Focused tests for the fail-closed V-JEPA frontier Slurm workflow."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import vjepa2_nfe_frontier as frontier
from tools.slurm import vjepa2_frontier_workflow as workflow


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "tools" / "slurm"
LAUNCHER = SLURM / "submit_vjepa2_frontier_workflow.sh"
SELECTION_GATE = SLURM / "vjepa2_frontier_select_and_submit.sbatch"
QUALITY = SLURM / "vjepa2_frontier_quality.sbatch"
TIMING = SLURM / "vjepa2_frontier_latency.sbatch"
CACHE = SLURM / "vjepa2_frontier_cache.sbatch"
SHELL_SCRIPTS = (
    LAUNCHER,
    CACHE,
    SLURM / "vjepa2_frontier_final_gate.sbatch",
    QUALITY,
    SELECTION_GATE,
    SLURM / "vjepa2_frontier_confirm.sbatch",
    TIMING,
)


def _write_json(path: Path, payload: object) -> Path:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


class FrontierWorkflowHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_well_formed_ineligible_selection_returns_three(self) -> None:
        payload = frontier.identity_payload(
            {
                "kind": frontier.KIND_SELECTION,
                "selection_split": "validation",
                "confirmatory_eligible": False,
                "selected_pair": None,
            }
        )
        path = _write_json(self.root / "selection.json", payload)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = workflow.command_require_selection(
                argparse.Namespace(selection=str(path))
            )
        self.assertEqual(status, workflow.WELL_FORMED_INELIGIBLE)
        self.assertFalse(json.loads(output.getvalue())["lockbox_submission_allowed"])

    def test_eligible_selection_is_reproduced_before_success(self) -> None:
        payload = frontier.identity_payload(
            {
                "kind": frontier.KIND_SELECTION,
                "selection_split": "validation",
                "confirmatory_eligible": True,
                "selected_pair": {
                    "left": {"arm": "J1", "nfe": 2},
                    "reference": {"arm": "VPM", "nfe": 4},
                },
                "lockbox_registration": {"identity_sha256": "a" * 64},
            }
        )
        path = _write_json(self.root / "selection.json", payload)
        with mock.patch.object(
            workflow.frontier,
            "validate_confirmatory_selection",
            return_value={"selection": payload},
        ) as validator:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = workflow.command_require_selection(
                    argparse.Namespace(selection=str(path))
                )
        self.assertEqual(status, 0)
        validator.assert_called_once_with(payload)
        self.assertTrue(json.loads(output.getvalue())["lockbox_submission_allowed"])

    def test_study_values_emit_both_pinned_interpreters(self) -> None:
        training_commit = "9" * 40
        study = frontier.identity_payload(
            {
                "kind": "vjepa2_controlled_video_diffusion_study",
                "study_id": "fixture",
                "study_root": str(self.root),
                "inputs": {
                    "repository": {
                        "root": "/repo",
                        "git_commit": training_commit,
                    },
                    "runtime": {
                        "python": "/env/lacwm/bin/python",
                        "extractor_python": "/env/vjepa/bin/python",
                        "wan_dir": "/assets/wan",
                        "videox_home": "/src/videox",
                    },
                    "vjepa": {
                        "source": {"path": "/src/vjepa"},
                        "checkpoint": {
                            "path": "/assets/vjepa.pt",
                            "sha256": "1" * 64,
                        },
                        "pca_stats": {
                            "path": "/cache/pca.pt",
                            "sha256": "2" * 64,
                        },
                    },
                    "splits": {
                        "train": {
                            "clip_manifest": {"path": "/cache/train.jsonl"}
                        }
                    },
                },
            }
        )
        _write_json(self.root / "study_manifest.json", study)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = workflow.command_study_values(
                argparse.Namespace(
                    study_root=str(self.root),
                    training_commit=training_commit,
                )
            )
        values = output.getvalue().splitlines()
        self.assertEqual(status, 0)
        self.assertEqual(len(values), 13)
        self.assertEqual(values[3], "/env/lacwm/bin/python")
        self.assertEqual(values[4], "/env/vjepa/bin/python")

    def test_final_dependency_must_match_recorded_u1000_array(self) -> None:
        training_commit = "9" * 40
        study = frontier.identity_payload(
            {
                "kind": "vjepa2_controlled_video_diffusion_study",
                "study_id": "fixture",
                "study_root": str(self.root),
                "inputs": {
                    "repository": {
                        "root": "/repo",
                        "git_commit": training_commit,
                    }
                },
            }
        )
        _write_json(self.root / "study_manifest.json", study)
        submission = frontier.identity_payload(
            {
                "kind": "vjepa2_controlled_study_submission",
                "study_identity_sha256": study["identity_sha256"],
                "dependency": "afterok",
                "stage_array_job_ids": [
                    {"completed_updates": 800, "job_id": "481131"},
                    {"completed_updates": 1000, "job_id": "481132"},
                ],
            }
        )
        _write_json(self.root / "slurm_submission.json", submission)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = workflow.command_check_submission(
                argparse.Namespace(
                    study_root=str(self.root),
                    training_commit=training_commit,
                    final_job_id="481132",
                )
            )
        self.assertEqual(status, 0)
        with self.assertRaisesRegex(
            workflow.WorkflowError, "not the recorded update-1000"
        ):
            workflow.command_check_submission(
                argparse.Namespace(
                    study_root=str(self.root),
                    training_commit=training_commit,
                    final_job_id="999999",
                )
            )

    def test_final_job_accounting_active_and_terminal_modes(self) -> None:
        self.assertEqual(
            workflow.classify_final_job_rows(
                "481132",
                [
                    "481132_0|RUNNING|0:0",
                    "481132_4|PENDING|0:0",
                ],
                [
                    "999999|0|RUNNING|None",
                    "481132|0|RUNNING|None",
                    "481132|1|RUNNING|None",
                    "481132|2|RUNNING|None",
                    "481132|3|RUNNING|None",
                    "481132|4|PENDING|Dependency",
                ],
            ),
            "active_afterok",
        )
        self.assertEqual(
            workflow.classify_final_job_rows(
                "481132",
                [
                    "481132_0|COMPLETED|0:0",
                    "481132_1|COMPLETED|0:0",
                    "481132_2|COMPLETED|0:0",
                    "481132_3|COMPLETED|0:0",
                    "481132_4|COMPLETED|0:0",
                ],
            ),
            "terminal_success",
        )
        with self.assertRaisesRegex(
            workflow.WorkflowError, "failed accounting"
        ):
            workflow.classify_final_job_rows(
                "481132",
                [
                    "481132_0|FAILED|1:0",
                    "481132_[1-4]|COMPLETED|0:0",
                ],
            )

    def test_active_final_job_requires_exact_consistent_squeue_tasks(self) -> None:
        accounting = [
            "481132_0|RUNNING|0:0",
            "481132_4|PENDING|0:0",
        ]
        with self.assertRaisesRegex(workflow.WorkflowError, "squeue task set differs"):
            workflow.classify_final_job_rows(
                "481132",
                accounting,
                [
                    "481132|0|RUNNING|None",
                    "481132|1|RUNNING|None",
                    "481132|2|RUNNING|None",
                    "481132|4|PENDING|Dependency",
                ],
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "active state mismatch"):
            workflow.classify_final_job_rows(
                "481132",
                accounting,
                [
                    "481132|0|PENDING|Priority",
                    "481132|1|RUNNING|None",
                    "481132|2|RUNNING|None",
                    "481132|3|RUNNING|None",
                    "481132|4|PENDING|Dependency",
                ],
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "ambiguous active state"):
            workflow.classify_final_job_rows(
                "481132",
                accounting,
                [
                    "481132|0|RUNNING|None",
                    "481132|1|FAILED|NonZeroExitCode",
                    "481132|2|RUNNING|None",
                    "481132|3|RUNNING|None",
                    "481132|4|PENDING|Dependency",
                ],
            )

    def test_terminal_accounting_requires_exact_five_task_set(self) -> None:
        with self.assertRaisesRegex(workflow.WorkflowError, "task set differs"):
            workflow.classify_final_job_rows(
                "481132",
                [
                    "481132_0|COMPLETED|0:0",
                    "481132_1|COMPLETED|0:0",
                    "481132_2|COMPLETED|0:0",
                    "481132_3|COMPLETED|0:0",
                ],
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "duplicates array task"):
            workflow.classify_final_job_rows(
                "481132",
                [
                    "481132_[0-4]|COMPLETED|0:0",
                    "481132_4|COMPLETED|0:0",
                ],
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "unexpected array task"):
            workflow.classify_final_job_rows(
                "481132",
                ["481132_[0-5]|COMPLETED|0:0"],
            )

    def test_final_job_query_uses_formatted_job_ids(self) -> None:
        completed = [
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout=(
                    "481132_0|RUNNING|0:0\n"
                    "481132_4|PENDING|0:0\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout=(
                    "481132|0|RUNNING|None\n"
                    "481132|1|RUNNING|None\n"
                    "481132|2|RUNNING|None\n"
                    "481132|3|RUNNING|None\n"
                    "481132|4|PENDING|Dependency\n"
                ),
                stderr="",
            ),
        ]
        with mock.patch.object(
            workflow.subprocess, "run", side_effect=completed
        ) as runner:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = workflow.command_classify_final_job(
                    argparse.Namespace(final_job_id="481132")
                )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), "active_afterok")
        accounting_command = runner.call_args_list[0].args[0]
        queue_command = runner.call_args_list[1].args[0]
        self.assertIn(
            "--format=JobID%64,State%32,ExitCode", accounting_command
        )
        self.assertNotIn("JobIDRaw", " ".join(accounting_command))
        self.assertEqual(
            queue_command,
            [
                "squeue",
                "-r",
                "--user",
                workflow.pwd.getpwuid(os.getuid()).pw_name,
                "-h",
                "-o",
                "%F|%K|%T|%r",
            ],
        )

    def test_final_job_query_terminal_ignores_unrelated_live_arrays(self) -> None:
        completed = [
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="".join(
                    f"481132_{task}|COMPLETED|0:0\n"
                    for task in range(5)
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="999999|0|RUNNING|None\n",
                stderr="",
            ),
        ]
        with mock.patch.object(
            workflow.subprocess, "run", side_effect=completed
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = workflow.command_classify_final_job(
                    argparse.Namespace(final_job_id="481132")
                )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), "terminal_success")

    def _adopted_cache_fixture(
        self,
    ) -> tuple[argparse.Namespace, str, str]:
        producer = self.root / "producer"
        current = self.root / "current"
        log_dir = self.root / "study" / "_frontier_slurm" / "logs"
        producer.mkdir()
        current.mkdir()
        log_dir.mkdir(parents=True)
        args = argparse.Namespace(
            job_id="481556",
            study_root=str(self.root / "study"),
            training_commit="9" * 40,
            current_repo_root=str(current),
            current_evaluator_commit="b" * 40,
            final_job_id="481132",
            python="/env/lacwm/bin/python",
            extractor_python="/env/vjepa/bin/python3.11",
            vjepa_source="/assets/vjepa/source",
            vjepa_checkpoint="/assets/vjepa/checkpoint.pt",
            vjepa_checkpoint_sha256="1" * 64,
            pca="/cache/pca.pt",
            pca_sha256="2" * 64,
            train_manifest="/cache/train.jsonl",
            partition="batch",
            account="coreai_chef_posttrain",
            qos="normal",
            cache_time="01:00:00",
            cache_cpus="32",
            cache_memory="256G",
            log_dir=str(log_dir),
        )
        producer_commit = "a" * 40
        expected_user = workflow.pwd.getpwuid(os.getuid()).pw_name
        tokens = [
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            "--ntasks-per-node=1",
            "--gpus-per-node=1",
            "--cpus-per-task=32",
            "--mem=256G",
            "--time=01:00:00",
            "--partition=batch",
            "--dependency=afterok:481132",
            "--no-requeue",
            "--open-mode=append",
            "--export=ALL",
            "--job-name=vjepa2-frontier-cache",
            f"--output={log_dir}/%x-%j.out",
            f"--error={log_dir}/%x-%j.err",
            "--account=coreai_chef_posttrain",
            "--qos=normal",
            str(producer / "tools/slurm/vjepa2_frontier_cache.sbatch"),
            "--repo-root",
            str(producer),
            "--study-root",
            str(self.root / "study"),
            "--training-commit",
            "9" * 40,
            "--evaluator-commit",
            producer_commit,
            "--python",
            "/env/lacwm/bin/python",
            "--extractor-python",
            "/env/vjepa/bin/python3.11",
            "--vjepa-source",
            "/assets/vjepa/source",
            "--vjepa-checkpoint",
            "/assets/vjepa/checkpoint.pt",
            "--vjepa-checkpoint-sha256",
            "1" * 64,
            "--pca",
            "/cache/pca.pt",
            "--pca-sha256",
            "2" * 64,
            "--train-manifest",
            "/cache/train.jsonl",
        ]
        row = "|".join(
            [
                "481556",
                "vjepa2-frontier-cache",
                expected_user,
                "PENDING",
                "0:0",
                "coreai_chef_posttrain",
                "normal",
                "batch",
                "billing=32,cpu=32,gres/gpu=1,mem=256G,node=1",
                "32",
                "256Gn",
                "01:00:00",
                shlex.join(tokens),
            ]
        )
        control = " ".join(
            [
                "JobId=481556",
                "JobName=vjepa2-frontier-cache",
                f"UserId={expected_user}({os.getuid()})",
                "Account=coreai_chef_posttrain",
                "QOS=normal",
                "JobState=PENDING",
                "Reason=Dependency",
                "Dependency=afterok:481132_*(unfulfilled)",
                "Requeue=0",
                "Restarts=0",
                "BatchFlag=1",
                "ExitCode=0:0",
                "TimeLimit=01:00:00",
                "Partition=batch",
                "NumNodes=1-1",
                "NumCPUs=32",
                "NumTasks=1",
                "CPUs/Task=32",
                "NodeList=",
                "ReqTRES=cpu=32,mem=256G,node=1,billing=8,gres/gpu=1",
                "AllocTRES=(null)",
                (
                    "Command="
                    f"{producer}/tools/slurm/vjepa2_frontier_cache.sbatch"
                ),
                f"WorkDir={workflow.pwd.getpwuid(os.getuid()).pw_dir}",
                (
                    "StdErr="
                    f"{log_dir}/vjepa2-frontier-cache-481556.err"
                ),
                "StdIn=/dev/null",
                (
                    "StdOut="
                    f"{log_dir}/vjepa2-frontier-cache-481556.out"
                ),
                "MemPerTres=gpu:474112",
                "TresPerNode=gres/gpu:1",
                "TresPerTask=cpu=32",
            ]
        )
        return args, row + "|", control

    def test_pending_cache_adoption_binds_exact_command_and_old_evaluator(
        self,
    ) -> None:
        args, row, control = self._adopted_cache_fixture()
        with mock.patch.object(
            workflow,
            "_validate_adopted_cache_repositories",
            return_value={"scientific_code_objects_unchanged": True},
        ) as repository_validator:
            evidence = workflow.validate_adopted_cache_row(args, row, control)
        self.assertEqual(evidence["job_id"], "481556")
        self.assertEqual(evidence["state"], "PENDING")
        self.assertEqual(evidence["producer_evaluator_commit"], "a" * 40)
        self.assertEqual(
            evidence["producer_repo_root"], str(self.root / "producer")
        )
        repository_validator.assert_called_once_with(
            producer_repo=self.root / "producer",
            producer_commit="a" * 40,
            current_repo=self.root / "current",
            current_commit="b" * 40,
            training_commit="9" * 40,
        )

    def test_cache_adoption_rejects_state_resource_and_submitline_drift(
        self,
    ) -> None:
        args, row, control = self._adopted_cache_fixture()
        with self.assertRaisesRegex(workflow.WorkflowError, "must still be PENDING"):
            workflow.validate_adopted_cache_row(
                args, row.replace("|PENDING|", "|RUNNING|", 1), control
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "requested TRES differ"):
            workflow.validate_adopted_cache_row(
                args, row.replace("gres/gpu=1", "gres/gpu=2", 1), control
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "SubmitLine differs"):
            workflow.validate_adopted_cache_row(
                args, row.replace("--no-requeue", "--requeue", 1), control
            )
        with self.assertRaisesRegex(
            workflow.WorkflowError, "scontrol dependency differs"
        ):
            workflow.validate_adopted_cache_row(
                args,
                row,
                control.replace(
                    "Dependency=afterok:481132_*(unfulfilled)",
                    "Dependency=afterok:999999_*(unfulfilled)",
                ),
            )

    def test_adopted_cache_query_rejects_duplicate_accounting_rows(self) -> None:
        args, row, _ = self._adopted_cache_fixture()
        completed = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=row + "\n" + row + "\n",
            stderr="",
        )
        with mock.patch.object(
            workflow.subprocess, "run", return_value=completed
        ) as runner:
            with self.assertRaisesRegex(workflow.WorkflowError, "exactly one row"):
                workflow.command_validate_adopted_cache(args)
        command = runner.call_args.args[0]
        self.assertIn("--duplicates", command)
        self.assertIn("SubmitLine%4096", " ".join(command))

    def test_adopted_cache_query_uses_scontrol_for_live_dependency(self) -> None:
        args, row, control = self._adopted_cache_fixture()
        completed = [
            subprocess.CompletedProcess(
                args=(), returncode=0, stdout=row + "\n", stderr=""
            ),
            subprocess.CompletedProcess(
                args=(), returncode=0, stdout=control + "\n", stderr=""
            ),
        ]
        with (
            mock.patch.object(
                workflow.subprocess, "run", side_effect=completed
            ) as runner,
            mock.patch.object(
                workflow,
                "_validate_adopted_cache_repositories",
                return_value={"scientific_code_objects_unchanged": True},
            ),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = workflow.command_validate_adopted_cache(args)
        self.assertEqual(status, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 3)
        accounting_command = runner.call_args_list[0].args[0]
        self.assertNotIn("Dependency", " ".join(accounting_command))
        self.assertIn("SubmitLine%4096", " ".join(accounting_command))
        self.assertEqual(
            runner.call_args_list[1].args[0],
            ["scontrol", "show", "job", "--oneliner", "481556"],
        )

    def test_adopted_cache_repository_diff_is_orchestration_only(self) -> None:
        current = self.root / "current"
        producer = self.root / "producer"
        current.mkdir()
        subprocess.run(["git", "init", "-q", str(current)], check=True)
        subprocess.run(
            ["git", "-C", str(current), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(current),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        directory_paths = {
            "projects/latent_action_models/lam",
            "robot_wm/datasets",
            "robot_wm/modeling",
            "tools/env/videox_shim",
        }
        file_paths = set(workflow.ADOPTED_CACHE_CODE_PATHS) - directory_paths
        file_paths.add("tools/slurm/vjepa2_frontier_workflow.py")
        for relative in directory_paths:
            marker = current / relative / "marker.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("immutable\n", encoding="utf-8")
        for relative in file_paths:
            path = current / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("immutable\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(current), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(current), "commit", "-qm", "training"],
            check=True,
        )
        training_commit = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        readme = current / "tools/slurm/README.md"
        readme.write_text("producer\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(current), "add", str(readme)], check=True)
        subprocess.run(
            ["git", "-C", str(current), "commit", "-qm", "producer"],
            check=True,
        )
        producer_commit = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            [
                "git",
                "-C",
                str(current),
                "worktree",
                "add",
                "-q",
                "--detach",
                str(producer),
                producer_commit,
            ],
            check=True,
        )
        helper = current / "tools/slurm/vjepa2_frontier_workflow.py"
        helper.write_text("recovery controller\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(current), "add", str(helper)], check=True)
        subprocess.run(
            ["git", "-C", str(current), "commit", "-qm", "recovery"],
            check=True,
        )
        current_commit = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        evidence = workflow._validate_adopted_cache_repositories(
            producer_repo=producer,
            producer_commit=producer_commit,
            current_repo=current,
            current_commit=current_commit,
            training_commit=training_commit,
        )
        self.assertTrue(evidence["changes_are_recovery_allowlisted"])
        self.assertEqual(
            evidence["recovery_changed_paths"],
            ["tools/slurm/vjepa2_frontier_workflow.py"],
        )

        scientific = current / "tools/vjepa2_nfe_frontier.py"
        scientific.write_text("changed science\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(current), "add", str(scientific)], check=True
        )
        subprocess.run(
            ["git", "-C", str(current), "commit", "-qm", "bad science"],
            check=True,
        )
        bad_commit = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with self.assertRaisesRegex(
            workflow.WorkflowError, "non-orchestration paths"
        ):
            workflow._validate_adopted_cache_repositories(
                producer_repo=producer,
                producer_commit=producer_commit,
                current_repo=current,
                current_commit=bad_commit,
                training_commit=training_commit,
            )


class FrontierSlurmContractTest(unittest.TestCase):
    def test_all_shell_entrypoints_parse(self) -> None:
        for path in SHELL_SCRIPTS:
            with self.subTest(path=path.name):
                completed = subprocess.run(
                    ["bash", "-n", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_lockbox_jobs_are_below_reproduced_eligibility_gate(self) -> None:
        source = SELECTION_GATE.read_text(encoding="utf-8")
        eligibility = source.index("require-selection --selection")
        eligibility_case = source.index('case "$ELIGIBILITY_STATUS"')
        lockbox_submit = source.index('LOCKBOX_JOB_ID="$(')
        self.assertLess(eligibility, eligibility_case)
        self.assertLess(eligibility_case, lockbox_submit)
        self.assertIn(
            "No confirmatory validation candidate. Lockbox scoring was not submitted.",
            source,
        )
        self.assertIn('--dependency="afterok:$SLURM_JOB_ID"', source)

    def test_ineligible_selection_executes_no_sbatch_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / "repo"
            study = root / "study"
            bin_dir = root / "bin"
            log_dir = root / "logs"
            bin_dir.mkdir()
            log_dir.mkdir()
            (repo / "tools" / "slurm").mkdir(parents=True)
            registration = study / "frontier_lockbox" / "registration.json"
            registration.parent.mkdir(parents=True)
            _write_json(registration, {"registered": True})
            for directory in (
                study
                / "vpm_parameter_matched_video"
                / "frontier_quality"
                / "validation"
                / "update_1000",
                study
                / "j1_joint_auxiliary_leads"
                / "frontier_quality"
                / "validation"
                / "update_1000",
            ):
                directory.mkdir(parents=True)
                _write_json(directory / "inventory.json", {"complete": True})
                for rank in range(8):
                    (directory / f"rank_{rank:03d}.jsonl").write_text(
                        "{}\n", encoding="utf-8"
                    )

            git = bin_dir / "git"
            git.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$*\" == *\"rev-parse HEAD\"* ]]; then "
                "echo eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee; fi\n"
                f"if [[ \"$*\" == *\"rev-parse --show-toplevel\"* ]]; then "
                f"echo {str(repo)!r}; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            python = bin_dir / "python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == *vjepa2_nfe_frontier.py ]]; then\n"
                "  while (($#)); do\n"
                "    if [[ \"$1\" == --output ]]; then\n"
                "      printf '%s\\n' "
                "'{\"confirmatory_eligible\":false,\"selected_pair\":null}' "
                "> \"$2\"\n"
                "      exit 0\n"
                "    fi\n"
                "    shift\n"
                "  done\n"
                "fi\n"
                "if [[ \"$1\" == *vjepa2_frontier_workflow.py ]]; then exit 3; fi\n"
                "exit 2\n",
                encoding="utf-8",
            )
            sbatch_marker = root / "sbatch-called"
            sbatch = bin_dir / "sbatch"
            sbatch.write_text(
                f"#!/usr/bin/env bash\n: > {str(sbatch_marker)!r}\n",
                encoding="utf-8",
            )
            for executable in (git, python, sbatch):
                executable.chmod(0o755)

            env = dict(os.environ)
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "SLURM_JOB_ID": "12345",
                    "SLURM_RESTART_COUNT": "0",
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(SELECTION_GATE),
                    "--repo-root",
                    str(repo),
                    "--study-root",
                    str(study),
                    "--training-commit",
                    "9" * 40,
                    "--evaluator-commit",
                    "e" * 40,
                    "--python",
                    str(python),
                    "--wan-dir",
                    str(root / "wan"),
                    "--videox-home",
                    str(root / "videox"),
                    "--log-dir",
                    str(log_dir),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(sbatch_marker.exists())
            self.assertIn(
                "Lockbox scoring was not submitted", completed.stdout
            )

    def test_initial_dag_uses_separate_validation_jobs_and_no_lockbox_job(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("VPM_VALIDATION_JOB_ID=\"$(submit_validation VPM)\"", source)
        self.assertIn("J1_VALIDATION_JOB_ID=\"$(submit_validation J1)\"", source)
        self.assertIn(
            '"afterok:$CACHE_DEPENDENCY:$FINAL_GATE_DEPENDENCY"', source
        )
        self.assertIn('FINAL_JOB_MODE="$(', source)
        self.assertIn("active_afterok)", source)
        self.assertIn("terminal_success)", source)
        self.assertIn('"${FINAL_JOB_DEPENDENCY_ARGS[@]}"', source)
        self.assertNotIn("frontier-lockbox", source)
        final_gate = source[source.index('FINAL_GATE_JOB_ID="$(') :]
        final_gate = final_gate[: final_gate.index("QUALITY_JOB_ARGS=(")]
        self.assertIn("--gpus-per-node=1", final_gate)
        selection = source[source.index('SELECT_JOB_ID="$(') :]
        selection = selection[: selection.index('"$PYTHON_BIN" -')]
        self.assertIn("--gpus-per-node=1", selection)
        continuation = SELECTION_GATE.read_text(encoding="utf-8")
        confirmation = continuation[continuation.index('CONFIRM_JOB_ID="$(') :]
        confirmation = confirmation[: confirmation.index('TIMING_JOB_ID="$(')]
        self.assertIn("--gpus-per-node=1", confirmation)

    def test_cache_is_one_b200_fresh_and_never_overwrites(self) -> None:
        source = CACHE.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gpus-per-node=1", source)
        self.assertIn('[[ ! -e "$LOCKBOX_ROOT" ]]', source)
        self.assertIn('"$EXTRACTOR_PYTHON" "$EXTRACTOR" extract', source)
        self.assertIn('"$PYTHON_BIN" "$LOCKBOX_TOOL" register', source)
        self.assertNotIn("--overwrite", source)

    def test_quality_and_timing_paths_are_fresh_only(self) -> None:
        quality = QUALITY.read_text(encoding="utf-8")
        timing = TIMING.read_text(encoding="utf-8")
        self.assertIn('[[ ! -e "$OUTPUT_DIR" ]]', quality)
        self.assertIn('--nproc_per_node=8', quality)
        self.assertIn('[[ ! -e "$LATENCY_PARENT" ]]', timing)
        mkdir_index = timing.index('mkdir "$LATENCY_PARENT"')
        benchmark_index = timing.index(
            '"$PYTHON_BIN" "$REPO_ROOT/tools/benchmark_vjepa2_frontier_latency.py"'
        )
        self.assertLess(mkdir_index, benchmark_index)
        self.assertIn('--benchmark-commit "$EVALUATOR_COMMIT"', timing)

    def test_launcher_defaults_bind_existing_v3_study(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "vjepa2-controlled-20260730-seed1234-9cf8e69-v3", source
        )
        self.assertIn(
            "9cf8e6922f35a5d6645e3128545953723bf54da2", source
        )
        self.assertIn("VJEPA_FINAL_U1000_JOB_ID:-481132", source)
        self.assertIn("lacwm-b200-py310/bin/python", source)
        self.assertIn("vjepa2-extractor-py311/bin/python3.11", source)
        self.assertIn('PARTITION="batch"', source)
        self.assertIn('ACCOUNT="coreai_chef_posttrain"', source)
        self.assertIn('QOS="normal"', source)
        self.assertIn("--adopt-cache-job-id", source)
        self.assertIn(
            'SCIENTIFIC_EVALUATOR_COMMIT="${ADOPTION_VALUES[1]}"', source
        )
        self.assertIn(
            '--repo-root "$SCIENTIFIC_REPO_ROOT"', source
        )
        self.assertIn(
            '--evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT"', source
        )
        self.assertIn('"cache_job_adopted": adoption_evidence is not None', source)
        self.assertIn('"cache_adoption_evidence": adoption_evidence', source)
        self.assertIn('"controller_repo_root": controller_repo_root', source)
        self.assertIn('"controller_git_commit": controller_commit', source)
        self.assertIn(
            '[[ "$TRAINING_REPO_ROOT" != "$REPO_ROOT" ]]', source
        )
        self.assertIn("rev-parse --show-toplevel", source)


if __name__ == "__main__":
    unittest.main()
