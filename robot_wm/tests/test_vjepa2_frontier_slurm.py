"""Focused tests for the fail-closed V-JEPA frontier Slurm workflow."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
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
                    "481132_[0-2%2]|PENDING|0:0",
                    "481132_3|RUNNING|0:0",
                    "481132_4|COMPLETED|0:0",
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
        completed = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout="481132_[0-4%2]|PENDING|0:0\n",
            stderr="",
        )
        with mock.patch.object(
            workflow.subprocess, "run", return_value=completed
        ) as runner:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = workflow.command_classify_final_job(
                    argparse.Namespace(final_job_id="481132")
                )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), "active_afterok")
        command = runner.call_args.args[0]
        self.assertIn("--format=JobID%64,State%32,ExitCode", command)
        self.assertNotIn("JobIDRaw", " ".join(command))


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
        self.assertIn(
            '[[ "$TRAINING_REPO_ROOT" != "$REPO_ROOT" ]]', source
        )
        self.assertIn("rev-parse --show-toplevel", source)


if __name__ == "__main__":
    unittest.main()
