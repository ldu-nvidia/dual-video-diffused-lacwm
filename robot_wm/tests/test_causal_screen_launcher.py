import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "tools" / "dual_abc_causal_screen.py"
SUBMIT = REPO_ROOT / "tools" / "slurm" / "submit_dual_abc_causal_screen.sh"
SLOT = REPO_ROOT / "tools" / "slurm" / "dual_abc_causal_screen.sbatch"


def load_helper():
    sys.path.insert(0, str(HELPER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "dual_abc_causal_screen_test_module", HELPER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class CausalScreenLauncherTest(unittest.TestCase):
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
        self.assertIn('TIME_LIMIT="04:00:00"', SUBMIT.read_text())
        self.assertIn("#SBATCH --time=04:00:00", SLOT.read_text())

    def test_array_contract_has_exact_five_controlled_arms(self):
        observed = [
            (
                arm["name"],
                arm["condition_on_tf"],
                arm["condition_mode"],
                arm["state_gate_init"],
                arm["smoke_variant"],
            )
            for arm in self.helper.ARMS
        ]
        self.assertEqual(
            observed,
            [
                ("off_s000", False, "off", 0.0, "dual-no-ztf"),
                ("matched_s003", True, "matched", 0.03, "dual-with-ztf"),
                ("shuffled_s003", True, "shuffled", 0.03, "dual-with-ztf"),
                ("matched_s010", True, "matched", 0.10, "dual-with-ztf"),
                ("shuffled_s010", True, "shuffled", 0.10, "dual-with-ztf"),
            ],
        )

    def test_arm_contract_cli_rejects_out_of_range_task(self):
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
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("array task ID must be in [0, 4]", result.stdout)

    def test_arm_contract_cli_exposes_fixed_evaluation_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HELPER_PATH),
                "arm-contract",
                "--array-task-id",
                "4",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["condition_mode"], "shuffled")
        self.assertEqual(payload["evaluation_nfe_steps"], [1, 2, 4, 8])
        self.assertEqual(payload["evaluation_noise_seed"], 20260726)
        self.assertEqual(
            payload["evaluation_condition_sources"],
            ["autonomous", "off", "oracle_matched", "oracle_shuffled"],
        )
        self.assertFalse(payload["state_gate_trainable"])

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
        self.assertIn('--array="0-4%$MAX_CONCURRENT_ARMS"', source)
        self.assertIn("--no-requeue", source)
        self.assertIn(
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED", source
        )
        self.assertIn('git -C "$REPO_ROOT" status --porcelain', source)
        self.assertIn('[[ ! -e "$SCREEN_ROOT" ]]', source)
        self.assertIn("CHECKPOINT_SHA256=", source)
        self.assertIn("dual-video-diffusion-private", source)
        self.assertIn("+wandb.group=null", SLOT.read_text(encoding="utf-8"))

    def test_slot_pins_training_and_independent_evaluation_contract(self):
        source = SLOT.read_text(encoding="utf-8")
        required_fragments = (
            "trainer.config.max_iter=200",
            "seed=1234",
            "model.dual_diffusion.condition_mode=$CONDITION_MODE",
            "model.dual_diffusion.state_gate_init=$STATE_GATE_INIT",
            "model.dual_diffusion.state_gate_trainable=false",
            "model.dual_diffusion.evaluation_nfe_steps=[1,2,4,8]",
            "model.dual_diffusion.evaluation_noise_seed=20260726",
            (
                "model.dual_diffusion.evaluation_condition_sources="
                "[autonomous,off,oracle_matched,oracle_shuffled]"
            ),
            "--nproc_per_node=8",
            "+wandb.resume=never",
            '[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]]',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertNotIn("scontrol requeue", source)
        self.assertNotIn("--resume", source)

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
        self.assertIn("oracle_sources_are_leakage", required)
        self.assertEqual(self.helper.VISUALIZATION_UPDATES, (0, 50, 100, 150, 199))


if __name__ == "__main__":
    unittest.main()
