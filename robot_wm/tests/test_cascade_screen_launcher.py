"""Tests for the guarded strict TF-first cascade screen launcher."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "tools" / "dual_abc_cascade_screen.py"
SUBMIT = REPO_ROOT / "tools" / "slurm" / "submit_dual_abc_cascade_screen.sh"
SLOT = REPO_ROOT / "tools" / "slurm" / "dual_abc_cascade_screen.sbatch"


def load_helper():
    sys.path.insert(0, str(HELPER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "dual_abc_cascade_screen_test_module", HELPER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class CascadeScreenLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper()

    def test_condition_modes_remain_strings_when_yaml_is_loaded(self):
        import yaml

        config_root = (
            REPO_ROOT / "projects" / "latent_action_models" / "configs"
        )
        paths = {
            config_root / "models" / "dual_explicit_action_dit_model.yaml": (
                ("dual_diffusion", "condition_mode"),
                "off",
            ),
            config_root
            / "experiments_0908"
            / "ravenhuang"
            / "wan-dit"
            / "dual_abc_no_ztf_condition.yaml": (
                ("model", "dual_diffusion", "condition_mode"),
                "off",
            ),
            config_root
            / "experiments_0908"
            / "ravenhuang"
            / "wan-dit"
            / "dual_abc_with_ztf_condition.yaml": (
                ("model", "dual_diffusion", "condition_mode"),
                "matched",
            ),
        }
        for path, (keys, expected) in paths.items():
            payload = yaml.safe_load(path.read_text())
            actual = payload
            for key in keys:
                actual = actual[key]
            self.assertIsInstance(actual, str, path)
            self.assertEqual(actual, expected, path)

    def test_slurm_default_fits_batch_qos_limit(self):
        source = SUBMIT.read_text()
        self.assertIn('TIME_LIMIT="04:00:00"', source)
        self.assertIn("TIME_SECONDS <= 4 * 3600", source)
        self.assertIn("#SBATCH --time=04:00:00", SLOT.read_text())

    def test_array_contract_has_exact_three_causal_arms(self):
        observed = [
            (
                arm["name"],
                arm["representation"],
                arm["condition_on_tf"],
                arm["condition_mode"],
                arm["state_gate_init"],
                arm["causal_role"],
                arm["smoke_variant"],
            )
            for arm in self.helper.ARMS
        ]
        self.assertEqual(
            observed,
            [
                (
                    "cascade_off_s000",
                    "parseval_rfft",
                    False,
                    "off",
                    0.0,
                    "tf_content_inert_control",
                    "dual-no-ztf",
                ),
                (
                    "cascade_matched_s010",
                    "parseval_rfft",
                    True,
                    "matched",
                    0.10,
                    "matched_tf_content",
                    "dual-with-ztf",
                ),
                (
                    "cascade_shuffled_s010",
                    "parseval_rfft",
                    True,
                    "shuffled",
                    0.10,
                    "wrong_future_tf_content_control",
                    "dual-with-ztf",
                ),
            ],
        )
        self.assertEqual(len({arm["name"] for arm in self.helper.ARMS}), 3)

    def test_arm_contract_cli_rejects_out_of_range_task(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "3",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("array task ID must be in [0, 2]", result.stdout)

    def test_arm_contract_cli_exposes_strict_evaluation_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "2",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["condition_mode"], "shuffled")
        self.assertEqual(payload["representation"], "parseval_rfft")
        self.assertEqual(payload["evaluation_nfe_steps"], [2, 4, 8])
        self.assertNotIn(1, payload["evaluation_nfe_steps"])
        self.assertEqual(payload["evaluation_noise_seed"], 20260726)
        self.assertEqual(
            payload["evaluation_condition_sources"],
            ["autonomous", "off", "oracle_matched", "oracle_shuffled"],
        )
        self.assertFalse(payload["state_gate_trainable"])
        self.assertTrue(payload["condition_only_video_loss_examples"])

    def test_tsv_contract_exposes_off_role_and_parseval_representation(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "0",
                "--format",
                "tsv",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(
            result.stdout.strip().split("\t"),
            [
                "cascade_off_s000",
                "parseval_rfft",
                "false",
                "off",
                "0.00",
                "tf_content_inert_control",
                "dual-no-ztf",
                "no_ztf",
            ],
        )

    def test_shell_entrypoints_parse(self):
        for path in (SUBMIT, SLOT):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stdout}")

    def test_submitter_is_guarded_and_concurrency_is_configurable(self):
        source = SUBMIT.read_text(encoding="utf-8")
        required = (
            "MAX_CONCURRENT_ARMS=3",
            '--array="0-2%$MAX_CONCURRENT_ARMS"',
            '[[ "$MAX_CONCURRENT_ARMS" =~ ^[1-3]$ ]]',
            "--no-requeue",
            'git -C "$REPO_ROOT" status --porcelain',
            '[[ ! -e "$SCREEN_ROOT" ]]',
            "CHECKPOINT_SHA256=",
            "dual-video-diffusion-private",
            (
                'RUN_ROOT="$LACWM_BASE/runs/dual_video_diffusion/'
                'ztf_first_cascade_screen"'
            ),
            "--allow-active-job-id",
            "--format='%F|%i|%T'",
            "unallowed_active_job_rows",
            "allowed active job ID is not active",
            'IMMEDIATE_PRE_SBATCH_JOB_ROWS=("${ACTIVE_JOB_ROWS[@]}")',
            "RECORD_SUBMISSION_ARGS+=(--observed-active-job",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertEqual(source.count("\ncheck_active_user_jobs\n"), 2)
        self.assertNotIn("--states=", source)
        self.assertNotIn("--format='%A|%i|%T'", source)
        self.assertIn("+wandb.group=null", SLOT.read_text(encoding="utf-8"))

    def test_active_job_allow_contract_is_exact_and_fail_closed(self):
        empty = self.helper._active_job_contract([], [])
        self.assertEqual(empty["allowed_array_or_job_ids"], [])
        self.assertEqual(empty["observed_active_jobs"], [])
        self.assertFalse(empty["wildcards_or_names_allowed"])

        allowed = self.helper._active_job_contract(
            ["472562"],
            [
                "472562|472562_0|RUNNING",
                "472562|472562_4|PENDING",
            ],
        )
        self.assertEqual(allowed["allowed_array_or_job_ids"], ["472562"])
        self.assertEqual(len(allowed["observed_active_jobs"]), 2)
        self.assertTrue(allowed["all_observed_jobs_explicitly_allowed"])

        with self.assertRaisesRegex(RuntimeError, "not explicitly allowed"):
            self.helper._active_job_contract(
                [], ["472562|472562_0|RUNNING"]
            )
        with self.assertRaisesRegex(RuntimeError, "were not active"):
            self.helper._active_job_contract(["472562"], [])
        with self.assertRaisesRegex(ValueError, "positive numeric"):
            self.helper._active_job_contract(["472*"], [])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.helper._active_job_contract(["472562", "472562"], [])

    def test_submission_signs_the_immediate_pre_sbatch_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "screen_manifest.json"
            first_inventory = self.helper._active_job_contract(
                ["472562"],
                ["472562|472562_0|RUNNING"],
            )
            manifest.write_text(
                json.dumps(
                    self.helper.pilot._identity_payload(
                        {
                            "kind": "dual_abc_tf_first_cascade_screen",
                            "active_job_coexistence": first_inventory,
                        }
                    )
                )
            )
            output = root / "slurm_submission.json"
            status = self.helper.command_record_submission(
                SimpleNamespace(
                    screen_manifest=str(manifest),
                    job_id="500000",
                    max_concurrent_arms=3,
                    observed_active_job=[
                        "472562|472562_0|COMPLETING",
                    ],
                    output=str(output),
                )
            )
            self.assertEqual(status, 0)
            payload = json.loads(output.read_text())
            self.assertTrue(self.helper._identity_is_valid(payload))
            self.assertEqual(
                payload[
                    "immediate_pre_sbatch_active_job_coexistence"
                ]["observed_active_jobs"][0]["state"],
                "COMPLETING",
            )

    def test_slot_pins_strict_native_cascade_and_evaluation_contract(self):
        source = SLOT.read_text(encoding="utf-8")
        required_fragments = (
            "trainer.config.max_iter=200",
            "seed=1234",
            "model.dual_diffusion.condition_mode='$CONDITION_MODE'",
            "model.dual_diffusion.state_gate_init=$STATE_GATE_INIT",
            "model.dual_diffusion.state_gate_trainable=false",
            "model.time_frequency_transform.representation='$REPRESENTATION'",
            "model.dual_diffusion.schedule_mode=tf_first_cascaded",
            "model.dual_diffusion.cascade_tf_loss_probability=0.4",
            "model.dual_diffusion.cascade_logit_mean=0.0",
            "model.dual_diffusion.cascade_logit_std=1.0",
            "model.dual_diffusion.cascade_tf_condition_max_sigma=0.25",
            "model.dual_diffusion.cascade_validation_tf_sigma=0.125",
            "model.dual_diffusion.cascade_inference_tf_fraction=0.5",
            (
                "model.dual_diffusion."
                "cascade_condition_only_video_loss_examples=true"
            ),
            "model.dual_diffusion.validation_video_sigmas=[0.90,0.75,0.50,0.25]",
            "model.dual_diffusion.evaluation_nfe_steps=[2,4,8]",
            "model.dual_diffusion.evaluation_noise_seed=20260726",
            (
                "model.dual_diffusion.evaluation_condition_sources="
                "[autonomous,off,oracle_matched,oracle_shuffled]"
            ),
            "cascade-contract-smoke",
            "--cascade-contract-report",
            "--nproc_per_node=8",
            "+wandb.resume=never",
            '[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]]',
            '[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-2]$ ]]',
            'RUN_ID="${SCREEN_ID}-${ARM}"',
            '[[ ! -e "$RUN_DIR" ]]',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertNotIn("evaluation_nfe_steps=[1", source)
        self.assertNotIn("scontrol requeue", source)
        self.assertNotIn("--resume", source)
        self.assertNotIn("checkpoint.resume", source)

    def test_manifest_contract_hashes_inputs_and_records_coexistence(self):
        source = HELPER_PATH.read_text(encoding="utf-8")
        required_fragments = (
            "pilot._assert_clean_commit(repo_root, expected_commit)",
            '"checkpoint": checkpoint_summary',
            '"sha256": pilot._sha256(common_config)',
            '"sha256": pilot._sha256(arm_config)',
            '"abc_manifest": pilot._abc_manifest_summary(data_root)',
            '"resolved_sha256": resolved_sha256',
            '"cascade_contract_smoke"',
            '"active_job_coexistence"',
            '"allowed_array_or_job_ids"',
            '"observed_active_jobs"',
            '"immediate_pre_sbatch_active_job_coexistence"',
            "pilot._exclusive_json(output, payload)",
            "pilot._exclusive_bytes(resolved_output, resolved_content)",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_cascade_contract_smoke_exercises_checked_out_primitives(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("local control-plane Python does not provide torch")
        actual_commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contract.json"
            args = SimpleNamespace(
                repo_root=str(REPO_ROOT),
                git_commit=actual_commit,
                output=str(output),
            )
            with mock.patch.object(
                self.helper.pilot, "_assert_clean_commit"
            ) as clean_check:
                status = self.helper.command_cascade_contract_smoke(args)
            self.assertEqual(status, 0)
            clean_check.assert_called_once()
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["git_commit"], actual_commit)
            self.assertEqual(payload["representation"], "parseval_rfft")
            self.assertEqual(payload["evaluation_nfe_steps"], [2, 4, 8])
            self.assertTrue(
                payload["diffusion_clock"][
                    "tf_loss_branch_video_is_pure_noise"
                ]
            )
            self.assertTrue(
                payload["diffusion_clock"][
                    "video_loss_branch_uses_native_video_sigma"
                ]
            )
            self.assertLess(
                payload["transform"]["roundtrip_max_abs"],
                2e-6,
            )
            self.assertTrue(self.helper._identity_is_valid(payload))

    def test_control_scope_and_oracle_leakage_are_explicit(self):
        source = HELPER_PATH.read_text(encoding="utf-8")
        self.assertIn("tf_content_inert_control", source)
        self.assertIn("single-state baseline", source)
        self.assertIn('"oracle_sources_are_leakage": True', source)
        self.assertIn("hidden-future TF", source)
        self.assertIn("exact zero content mask", source)
        self.assertIn('REQUESTED_VIEWER_EMAIL = "ldu@nvidia.edu"', source)
        submit_source = SUBMIT.read_text(encoding="utf-8")
        self.assertIn("Oracle diagnostics leak hidden future targets", submit_source)

    def test_requested_private_wandb_identity_is_recorded(self):
        accepted = {
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "access": "PRIVATE",
            "viewer_username": "zijiandu",
            "viewer_email": "ldu@nvidia.edu",
        }
        with mock.patch.object(
            self.helper.pilot,
            "_wandb_private_project",
            return_value=accepted,
        ):
            summary = self.helper._wandb_private_project(
                "zijiandu",
                "dual-video-diffusion-private",
            )
        self.assertTrue(summary["viewer_email_matches_request"])
        self.assertIsNone(summary["identity_deviation"])
        deviating = dict(accepted, viewer_email="zijiandu@asu.edu")
        with mock.patch.object(
            self.helper.pilot,
            "_wandb_private_project",
            return_value=deviating,
        ):
            summary = self.helper._wandb_private_project(
                "zijiandu",
                "dual-video-diffusion-private",
            )
        self.assertFalse(summary["viewer_email_matches_request"])
        self.assertIn("zijiandu@asu.edu", summary["identity_deviation"])

    def test_successful_outcome_requires_every_strict_nfe_artifact_key(self):
        required = self.helper._required_trajectory_tensor_names()
        for condition_source in (
            "autonomous",
            "off",
            "oracle_matched",
            "oracle_shuffled",
        ):
            infix = (
                "" if condition_source == "autonomous" else f"_{condition_source}"
            )
            for nfe in (2, 4, 8):
                self.assertIn(f"video_final{infix}_nfe_{nfe}", required)
                self.assertIn(f"tf_final{infix}_nfe_{nfe}", required)
                self.assertIn(f"decoded_future{infix}_nfe_{nfe}", required)
            self.assertNotIn(f"video_final{infix}_nfe_1", required)
        self.assertIn("tf_initial_noise", required)
        self.assertIn("oracle_sources_are_leakage", required)
        self.assertIn("condition_only_video_loss_examples", required)
        for schedule_name in (
            "video_sigmas",
            "tf_sigmas",
            "video_trajectory",
            "tf_trajectory",
            "video_x0_trajectory",
            "tf_x0_trajectory",
        ):
            self.assertIn(schedule_name, required)
        self.assertEqual(
            self.helper.VISUALIZATION_UPDATES,
            (0, 50, 100, 150, 199),
        )


if __name__ == "__main__":
    unittest.main()
