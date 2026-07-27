"""CPU-only contracts for the guarded privileged-video evaluation array."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "tools" / "privileged_video_eval_launch.py"
SUBMIT = (
    REPO_ROOT / "tools" / "slurm" / "submit_privileged_video_eval.sh"
)
SLOT = REPO_ROOT / "tools" / "slurm" / "privileged_video_eval.sbatch"


def load_helper():
    sys.path.insert(0, str(HELPER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "privileged_video_eval_launch_test_module", HELPER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class PrivilegedVideoEvalLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper()

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

    def test_fixed_serial_array_resources_and_allocation(self):
        submit = SUBMIT.read_text(encoding="utf-8")
        slot = SLOT.read_text(encoding="utf-8")
        for fragment in (
            "--array=0-2%1",
            "--nodes=1",
            "--ntasks=1",
            "--gpus-per-node=8",
            "--cpus-per-task=64",
            "--mem=600G",
            "--time=00:30:00",
            "--partition=batch",
            '--account="$SLURM_ACCOUNT"',
            '--qos="$SLURM_QOS"',
            "--no-requeue",
        ):
            self.assertIn(fragment, submit)
        for fragment in (
            "#SBATCH --array=0-2%1",
            "#SBATCH --nodes=1",
            "#SBATCH --gpus-per-node=8",
            "#SBATCH --time=00:30:00",
            "#SBATCH --partition=batch",
            "#SBATCH --account=coreai_chef_posttrain",
            "#SBATCH --qos=normal",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(fragment, slot)
        self.assertIn('"array_specification": "0-2%1"', HELPER_PATH.read_text())
        self.assertIn('"max_concurrent_tasks": 1', HELPER_PATH.read_text())
        self.assertEqual(
            self.helper.EXPECTED_SLURM_ACCOUNT, "coreai_chef_posttrain"
        )
        self.assertEqual(self.helper.EXPECTED_SLURM_QOS, "normal")

    def test_active_jobs_are_exact_fail_closed_and_checked_twice(self):
        empty = self.helper._active_job_contract([], [])
        self.assertEqual(empty["allowed_array_or_job_ids"], [])
        self.assertFalse(empty["wildcards_or_names_allowed"])
        allowed = self.helper._active_job_contract(
            ["473020"],
            [
                "473020|473020_0|RUNNING",
                "473020|473020_1|PENDING",
            ],
        )
        self.assertTrue(allowed["all_observed_jobs_explicitly_allowed"])
        with self.assertRaisesRegex(RuntimeError, "not explicitly allowed"):
            self.helper._active_job_contract(
                ["473020"],
                [
                    "473020|473020_0|RUNNING",
                    "473021|473021|PENDING",
                ],
            )
        with self.assertRaisesRegex(RuntimeError, "were not active"):
            self.helper._active_job_contract(["473020"], [])
        with self.assertRaisesRegex(ValueError, "positive numeric"):
            self.helper._active_job_contract(["473*"], [])

        source = SUBMIT.read_text(encoding="utf-8")
        self.assertEqual(source.count("\ncheck_active_user_jobs\n"), 2)
        self.assertIn("--format='%F|%i|%T'", source)
        self.assertNotIn("--states=", source)
        self.assertIn("IMMEDIATE_PRE_SBATCH_JOB_ROWS", source)

    def test_array_mapping_and_pinned_parent_contracts(self):
        expected = (
            (
                "trained_off",
                "cascade_off_s000",
                "8861147ccfcc0a2909480400d7f09452ae192298ac758f1cf73f71802d0b5f9b",
                "a147acb27dec8fb9f793d665861149ebc8d203b63ab1e6d107760f62d0b36e6b",
                0.0,
            ),
            (
                "trained_matched",
                "cascade_matched_s010",
                "ea6963718edb2b7827b189f3c622e5affe9b1706fb1cb623159731c8c29486e5",
                "5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d",
                0.1,
            ),
            (
                "trained_shuffled",
                "cascade_shuffled_s010",
                "151cf2b0a349878839f515b34ea4f8edd6a34528c2814c12b1fdc5afc9a3645f",
                "1b5e70982d1a93b4069b8ad1c33b25ba1b4d106c560dcaad63e1d2dd23c3eb76",
                0.1,
            ),
        )
        self.assertEqual(len(self.helper.ARMS), 3)
        for task_id, expected_arm in enumerate(expected):
            arm = self.helper._arm_by_task(task_id)
            observed = (
                arm["label"],
                arm["parent_arm"],
                arm["parent_identity_sha256"],
                arm["snapshot_sha256"],
                arm["state_gate_init"],
            )
            self.assertEqual(observed, expected_arm)
            self.assertEqual(arm["array_task_id"], task_id)
            self.assertEqual(set(arm["file_sha256"]), {
                "resolved_config",
                "arm_manifest",
                "outcome",
                "training_completion",
            })
            self.assertTrue(
                all(len(value) == 64 for value in arm["file_sha256"].values())
            )
        self.assertEqual(self.helper.SNAPSHOT_BYTES, 4_249_340_573)

    def test_evaluation_ids_are_arm_and_new_commit_bound(self):
        commit = "a" * 40
        self.assertEqual(
            self.helper.expected_eval_id(commit, "trained_matched"),
            "abc200-priv-video-trained_matched-s1234-aaaaaaaaaa-v1",
        )
        with self.assertRaises(ValueError):
            self.helper.expected_eval_id("abc", "trained_matched")
        with self.assertRaises(ValueError):
            self.helper.expected_eval_id(commit, "unknown")
        source = HELPER_PATH.read_text(encoding="utf-8")
        self.assertIn("commit == CORE_COMMIT", source)
        self.assertIn("requires a new immutable source commit", source)

    def test_exact_audited_hydra_vectors_match_shell_count(self):
        commit = "a" * 40
        for arm in self.helper.ARMS:
            label = arm["label"]
            eval_id = self.helper.expected_eval_id(commit, label)
            output = Path("/approved/privileged-video") / eval_id
            overrides = self.helper.expected_hydra_overrides(
                label, eval_id, output
            )
            self.assertEqual(len(overrides), 48)
            self.assertEqual(
                overrides[:5],
                [
                    (
                        "+experiments_0908="
                        "ravenhuang/wan-dit/"
                        "dual_abc_with_ztf_condition.yaml"
                    ),
                    f"name={eval_id}",
                    "seed=1234",
                    "data_loader.batch_size=1",
                    "trainer.config.max_iter=200",
                ],
            )
            required = {
                "trainer.config.transition_handoff_path=null",
                "trainer.config.load_path=null",
                "trainer.config.exclude_keys=[]",
                "trainer.config.share_spatial_attention=false",
                (
                    "trainer.config.saving.save_path="
                    f"{output}/_never_write_snapshot.pt"
                ),
                "model.dual_diffusion.enabled=true",
                "model.dual_diffusion.condition_on_tf=false",
                "model.dual_diffusion.condition_mode='off'",
                (
                    "model.dual_diffusion.state_gate_init="
                    f"{arm['state_gate_init']:.2f}"
                ),
                "model.dual_diffusion.state_gate_trainable=false",
                "model.dual_diffusion.schedule_mode=aligned",
                "model.dual_diffusion.evaluation_disable_tf_clock=true",
                (
                    "model.dual_diffusion."
                    "cascade_stage_faithful_inference=false"
                ),
                "model.dual_diffusion.evaluation_nfe_steps=[1,2,4,8]",
                (
                    "model.dual_diffusion.evaluation_condition_sources="
                    "[autonomous,off]"
                ),
                (
                    "model.time_frequency_transform."
                    "representation='parseval_rfft'"
                ),
                "+wandb.group=null",
                "+wandb.resume=never",
                (
                    "+privileged_video_evaluation.parent_arm="
                    f"{arm['parent_arm']}"
                ),
                (
                    "+privileged_video_evaluation."
                    "parent_completed_updates=200"
                ),
                "+privileged_video_evaluation.viz_skip_batches=4",
                "+privileged_video_evaluation.artifact_iteration=199",
            }
            self.assertTrue(required.issubset(set(overrides)))
            self.assertEqual(
                overrides[-7:],
                [
                    (
                        "+privileged_video_evaluation.parent_arm="
                        f"{arm['parent_arm']}"
                    ),
                    (
                        "+privileged_video_evaluation.snapshot_path="
                        f"{self.helper._parent_root(arm) / 'snapshot.pt'}"
                    ),
                    (
                        "+privileged_video_evaluation.snapshot_sha256="
                        f"{arm['snapshot_sha256']}"
                    ),
                    (
                        "+privileged_video_evaluation."
                        "parent_run_identity_sha256="
                        f"{arm['parent_identity_sha256']}"
                    ),
                    (
                        "+privileged_video_evaluation."
                        "parent_completed_updates=200"
                    ),
                    "+privileged_video_evaluation.viz_skip_batches=4",
                    "+privileged_video_evaluation.artifact_iteration=199",
                ],
            )
        slot = SLOT.read_text(encoding="utf-8")
        self.assertIn(
            "[[ ${#HYDRA_OVERRIDES[@]} -eq 48 ]]",
            slot,
        )
        self.assertNotIn("HYDRA_OVERRIDES[@]} -eq 49", slot)

    def test_slot_invokes_zero_update_evaluator_only(self):
        source = SLOT.read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node=8", source)
        self.assertIn("evaluate_privileged_video.py", source)
        self.assertNotIn(" train.py", source)
        self.assertNotIn("--resume", source)
        self.assertIn("_never_write_snapshot.pt", source)
        self.assertIn('"$OUTPUT_ROOT/training_complete.json"', source)
        self.assertIn("privileged_video_evaluation_provenance.json", source)
        self.assertIn("NFE=[1,2,4,8]", source)
        self.assertIn("sources=[autonomous,off]", source)

    def test_python_symlink_and_canonical_target_are_pinned(self):
        self.assertEqual(
            str(self.helper.PYTHON_LINK_TARGET),
            (
                "/lustre/fsw/portfolios/coreai/users/ldu/lacwm_train/"
                "python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
            ),
        )
        self.assertEqual(
            str(self.helper.PYTHON_REAL_BIN),
            (
                "/lustre/fsw/portfolios/coreai/projects/"
                "coreai_chef_pretrain/users/ldu/lacwm_train/python/"
                "cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
            ),
        )
        for source in (
            SUBMIT.read_text(encoding="utf-8"),
            SLOT.read_text(encoding="utf-8"),
        ):
            self.assertIn('[[ -L "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]', source)
            self.assertIn('readlink -f "$PYTHON_BIN"', source)
            self.assertIn('! -L "$PYTHON_REAL_BIN"', source)

    def _populate_rank_artifacts(self, output: Path) -> None:
        dataset = output / "visualization" / "iter_199" / "MultiDatasetABC_0"
        dataset.mkdir(parents=True)
        for rank in range(8):
            trajectory = dataset / f"latent_trajectory_rank_{rank}.safetensors"
            trajectory.write_bytes(f"rank-{rank}".encode())
            sidecar = {
                "global_rank": rank,
                "iteration": 199,
                "dataset": "MultiDatasetABC_0",
                "sigma_convention": "1=noise,0=clean",
                "safetensors_sha256": self.helper.pilot._sha256(trajectory),
            }
            trajectory.with_suffix(".json").write_text(json.dumps(sidecar))
            (dataset / f"viz_MultiDatasetABC_0_{rank}_0.mp4").write_bytes(
                f"video-{rank}".encode()
            )

    def test_completion_and_outcome_emit_privileged_specific_kinds(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._populate_rank_artifacts(output)
            manifest_path = output / "eval_manifest.json"
            manifest = self.helper.pilot._identity_payload(
                {
                    "schema_version": 1,
                    "kind": "privileged_video_posthoc_evaluation_manifest",
                    "evaluation_id": "test-privileged-eval",
                    "array_task_id": 1,
                    "arm_label": "trained_matched",
                    "output": {"output_root": str(output)},
                }
            )
            self.helper.pilot._exclusive_json(manifest_path, manifest)
            inventory_path = output / "eval_artifact_inventory.json"
            completion_path = output / "evaluation_complete.json"
            self.assertEqual(
                self.helper.command_complete(
                    SimpleNamespace(
                        manifest=str(manifest_path),
                        inventory_output=str(inventory_path),
                        completion_output=str(completion_path),
                    )
                ),
                0,
            )
            inventory = json.loads(inventory_path.read_text())
            completion = json.loads(completion_path.read_text())
            self.assertEqual(
                inventory["kind"],
                (
                    "privileged_video_posthoc_evaluation_"
                    "artifact_inventory"
                ),
            )
            self.assertEqual(
                completion["kind"],
                "privileged_video_posthoc_evaluation_completion",
            )
            self.assertEqual(inventory["primary_artifact_count"], 8)
            self.assertEqual(completion["rank_artifact_count"], 8)
            self.assertEqual(completion["training_steps"], 0)
            outcome_path = output / "outcome.json"
            self.assertEqual(
                self.helper.command_record_outcome(
                    SimpleNamespace(
                        manifest=str(manifest_path),
                        completion=str(completion_path),
                        exit_status=0,
                        array_job_id="473020",
                        array_task_id=1,
                        task_job_id="473021",
                        output=str(outcome_path),
                    )
                ),
                0,
            )
            outcome = json.loads(outcome_path.read_text())
            self.assertEqual(
                outcome["kind"],
                "privileged_video_posthoc_evaluation_outcome",
            )
            self.assertTrue(outcome["completed"])
            self.assertTrue(self.helper._identity_is_valid(outcome))

    def test_submission_is_array_specific_and_rejects_task_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "eval_manifest.json"
            manifest = self.helper.pilot._identity_payload(
                {
                    "kind": "privileged_video_posthoc_evaluation_manifest",
                    "array_task_id": 0,
                    "slurm": {
                        "array": True,
                        "array_range": [0, 2],
                        "array_specification": "0-2%1",
                        "max_concurrent_tasks": 1,
                        "active_job_coexistence": (
                            self.helper._active_job_contract([], [])
                        )
                    },
                }
            )
            self.helper.pilot._exclusive_json(manifest_path, manifest)
            submission_path = root / "submission.json"
            self.assertEqual(
                self.helper.command_record_submission(
                    SimpleNamespace(
                        manifest=str(manifest_path),
                        array_job_id="473020",
                        observed_active_job=[],
                        output=str(submission_path),
                    )
                ),
                0,
            )
            submission = json.loads(submission_path.read_text())
            self.assertEqual(
                submission["kind"],
                "privileged_video_posthoc_evaluation_submission",
            )
            self.assertEqual(submission["array_specification"], "0-2%1")
            self.assertEqual(submission["max_concurrent_tasks"], 1)
            with self.assertRaisesRegex(ValueError, "numeric base"):
                self.helper.command_record_submission(
                    SimpleNamespace(
                        manifest=str(manifest_path),
                        array_job_id="473020_0",
                        observed_active_job=[],
                        output=str(root / "invalid.json"),
                    )
                )


if __name__ == "__main__":
    unittest.main()
