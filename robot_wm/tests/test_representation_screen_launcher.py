import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "tools" / "dual_abc_representation_screen.py"
SUBMIT = REPO_ROOT / "tools" / "slurm" / "submit_dual_abc_representation_screen.sh"
SLOT = REPO_ROOT / "tools" / "slurm" / "dual_abc_representation_screen.sbatch"


def load_helper():
    sys.path.insert(0, str(HELPER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "dual_abc_representation_screen_test_module", HELPER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class RepresentationScreenLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper()

    def test_condition_modes_remain_strings_when_yaml_is_loaded(self):
        import yaml

        config_root = (
            REPO_ROOT
            / "projects"
            / "latent_action_models"
            / "configs"
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

    def test_array_contract_keeps_six_representation_arms_and_adds_video_only(self):
        observed = [
            (
                arm["name"],
                arm["representation"],
                arm["condition_on_tf"],
                arm["condition_mode"],
                arm["state_gate_init"],
                arm["state_gate_trainable"],
                arm["clock_gate_init"],
                arm["clock_gate_trainable"],
                arm["tf_loss_weight"],
                arm["video_only_control"],
                arm["smoke_variant"],
            )
            for arm in self.helper.ARMS
        ]
        self.assertEqual(
            observed,
            [
                (
                    "parseval_off_s000",
                    "parseval_rfft",
                    False,
                    "off",
                    0.0,
                    False,
                    0.0,
                    True,
                    1.0,
                    False,
                    "dual-no-ztf",
                ),
                (
                    "parseval_matched_s010",
                    "parseval_rfft",
                    True,
                    "matched",
                    0.10,
                    False,
                    0.0,
                    True,
                    1.0,
                    False,
                    "dual-with-ztf",
                ),
                (
                    "parseval_shuffled_s010",
                    "parseval_rfft",
                    True,
                    "shuffled",
                    0.10,
                    False,
                    0.0,
                    True,
                    1.0,
                    False,
                    "dual-with-ztf",
                ),
                (
                    "time_off_s000",
                    "time_packed",
                    False,
                    "off",
                    0.0,
                    False,
                    0.0,
                    True,
                    1.0,
                    False,
                    "dual-no-ztf",
                ),
                (
                    "time_matched_s010",
                    "time_packed",
                    True,
                    "matched",
                    0.10,
                    False,
                    0.0,
                    True,
                    1.0,
                    False,
                    "dual-with-ztf",
                ),
                (
                    "time_shuffled_s010",
                    "time_packed",
                    True,
                    "shuffled",
                    0.10,
                    False,
                    0.0,
                    True,
                    1.0,
                    False,
                    "dual-with-ztf",
                ),
                (
                    "video_only_s000",
                    "parseval_rfft",
                    False,
                    "off",
                    0.0,
                    False,
                    0.0,
                    False,
                    0.0,
                    True,
                    "dual-no-ztf",
                ),
            ],
        )
        self.assertEqual(len({arm["name"] for arm in self.helper.ARMS}), 7)
        self.assertEqual(
            [arm["name"] for arm in self.helper.ARMS[:6]],
            [
                "parseval_off_s000",
                "parseval_matched_s010",
                "parseval_shuffled_s010",
                "time_off_s000",
                "time_matched_s010",
                "time_shuffled_s010",
            ],
        )

    def test_arm_contract_cli_rejects_out_of_range_task(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "7",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("array task ID must be in [0, 6]", result.stdout)

    def test_arm_contract_cli_exposes_fixed_evaluation_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "5",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["condition_mode"], "shuffled")
        self.assertEqual(payload["representation"], "time_packed")
        self.assertEqual(payload["evaluation_nfe_steps"], [1, 2, 4, 8])
        self.assertEqual(payload["evaluation_noise_seed"], 20260726)
        self.assertEqual(
            payload["evaluation_condition_sources"],
            ["autonomous", "off", "oracle_matched", "oracle_shuffled"],
        )
        self.assertFalse(payload["state_gate_trainable"])

    def test_tsv_contract_keeps_off_quoted_and_exposes_representation(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "3",
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
                "time_off_s000",
                "time_packed",
                "false",
                "off",
                "0.00",
                "false",
                "0.00",
                "true",
                "1.00",
                "false",
                "dual-no-ztf",
                "no_ztf",
            ],
        )

    def test_shell_entrypoints_parse(self):
        result = subprocess.run(
            ["bash", "-n", str(SUBMIT), str(SLOT)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_submitter_is_guarded_and_concurrency_is_configurable(self):
        source = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('MAX_CONCURRENT_ARMS=4', source)
        self.assertIn('--array="0-6%$MAX_CONCURRENT_ARMS"', source)
        self.assertIn(
            '[[ "$MAX_CONCURRENT_ARMS" =~ ^[1-7]$ ]]',
            source,
        )
        self.assertIn("--no-requeue", source)
        self.assertIn(
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED", source
        )
        self.assertIn('git -C "$REPO_ROOT" status --porcelain', source)
        self.assertIn('[[ ! -e "$SCREEN_ROOT" ]]', source)
        self.assertIn("CHECKPOINT_SHA256=", source)
        self.assertIn("dual-video-diffusion-private", source)
        self.assertIn(
            'RUN_ROOT="$LACWM_BASE/runs/dual_video_diffusion/ztf_representation_screen"',
            source,
        )
        self.assertIn("+wandb.group=null", SLOT.read_text(encoding="utf-8"))

    def test_slot_pins_training_and_independent_evaluation_contract(self):
        source = SLOT.read_text(encoding="utf-8")
        required_fragments = (
            "trainer.config.max_iter=200",
            "seed=1234",
            "model.dual_diffusion.condition_mode='$CONDITION_MODE'",
            "model.dual_diffusion.state_gate_init=$STATE_GATE_INIT",
            "model.dual_diffusion.state_gate_trainable=$STATE_GATE_TRAINABLE",
            "model.dual_diffusion.clock_gate_init=$CLOCK_GATE_INIT",
            "model.dual_diffusion.clock_gate_trainable=$CLOCK_GATE_TRAINABLE",
            "model.dual_diffusion.tf_loss_weight=$TF_LOSS_WEIGHT",
            "model.dual_diffusion.video_only_control=$VIDEO_ONLY_CONTROL",
            "model.time_frequency_transform.representation='$REPRESENTATION'",
            "model.dual_diffusion.evaluation_nfe_steps=[1,2,4,8]",
            "model.dual_diffusion.evaluation_noise_seed=20260726",
            (
                "model.dual_diffusion.evaluation_condition_sources="
                "[autonomous,off,oracle_matched,oracle_shuffled]"
            ),
            "--nproc_per_node=8",
            "+wandb.resume=never",
            '[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]]',
            '[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-6]$ ]]',
            'RUN_ID="${SCREEN_ID}-${ARM}"',
            '[[ ! -e "$RUN_DIR" ]]',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertNotIn("scontrol requeue", source)
        self.assertNotIn("--resume", source)
        self.assertNotIn("checkpoint.resume", source)

    def test_manifest_contract_hashes_every_mutable_input_and_fails_closed(self):
        source = HELPER_PATH.read_text(encoding="utf-8")
        required_fragments = (
            "pilot._assert_clean_commit(repo_root, expected_commit)",
            '"checkpoint": checkpoint_summary',
            '"sha256": pilot._sha256(common_config)',
            '"sha256": pilot._sha256(arm_config)',
            '"abc_manifest": pilot._abc_manifest_summary(data_root)',
            '"resolved_sha256": resolved_sha256',
            "pilot._exclusive_json(output, payload)",
            "pilot._exclusive_bytes(resolved_output, resolved_content)",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_raw_rfft_candidate_is_explicitly_not_an_input(self):
        candidate = self.helper.RAW_RFFT_CANDIDATE
        self.assertEqual(candidate["slurm_array_job_id"], "472562")
        self.assertEqual(
            candidate["source_commit"],
            "b1738f9e39e3c8b61403437aa512f8951411f8b3",
        )
        self.assertFalse(candidate["accepted_as_screen_input"])
        self.assertIn("pending", candidate["status"])
        submit_source = SUBMIT.read_text(encoding="utf-8")
        self.assertNotIn("--raw-rfft-screen-manifest", submit_source)
        self.assertIn("not accepted until terminal provenance", submit_source)

    def test_each_arm_has_explicit_representation_and_disjoint_run_id(self):
        for arm in self.helper.ARMS:
            self.assertIn(
                arm["representation"],
                {"parseval_rfft", "time_packed"},
            )
        slot_source = SLOT.read_text(encoding="utf-8")
        self.assertIn(
            "parseval_rfft|time_packed",
            slot_source,
        )
        self.assertIn(
            "ztf-representation-screen,$REPRESENTATION,$ARM,seed-1234",
            slot_source,
        )

    def test_video_only_arm_is_a_fail_closed_causal_noop_contract(self):
        arm = self.helper.ARMS[6]
        self.assertEqual(arm["name"], "video_only_s000")
        self.assertFalse(arm["condition_on_tf"])
        self.assertEqual(arm["condition_mode"], "off")
        self.assertEqual(arm["state_gate_init"], 0.0)
        self.assertFalse(arm["state_gate_trainable"])
        self.assertEqual(arm["clock_gate_init"], 0.0)
        self.assertFalse(arm["clock_gate_trainable"])
        self.assertEqual(arm["tf_loss_weight"], 0.0)
        self.assertTrue(arm["video_only_control"])

        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "6",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        for key in (
            "condition_on_tf",
            "state_gate_trainable",
            "clock_gate_trainable",
        ):
            self.assertFalse(payload[key])
        self.assertTrue(payload["video_only_control"])
        self.assertEqual(payload["tf_loss_weight"], 0.0)

        slot_source = SLOT.read_text(encoding="utf-8")
        for failure_guard in (
            "video-only control must disable TF conditioning",
            "video-only control must freeze the state gate at exact zero",
            "video-only control must freeze the clock gate at exact zero",
            "video-only control must set TF loss weight to exact zero",
        ):
            self.assertIn(failure_guard, slot_source)

    def test_successful_outcome_requires_all_nfe_artifact_keys(self):
        required = self.helper._required_trajectory_tensor_names()
        for condition_source in (
            "autonomous",
            "off",
            "oracle_matched",
            "oracle_shuffled",
        ):
            infix = "" if condition_source == "autonomous" else f"_{condition_source}"
            for nfe in (1, 2, 4, 8):
                self.assertIn(f"video_final{infix}_nfe_{nfe}", required)
                self.assertIn(f"tf_final{infix}_nfe_{nfe}", required)
                self.assertIn(f"decoded_future{infix}_nfe_{nfe}", required)
        self.assertIn("tf_initial_noise", required)
        self.assertIn("video_only_control", required)
        self.assertIn("tf_loss_weight", required)
        self.assertIn("effective_state_gate", required)
        self.assertIn("effective_clock_gate", required)
        self.assertIn("oracle_sources_are_leakage", required)
        self.assertEqual(self.helper.VISUALIZATION_UPDATES, (0, 50, 100, 150, 199))


if __name__ == "__main__":
    unittest.main()
