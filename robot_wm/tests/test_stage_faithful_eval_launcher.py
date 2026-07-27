"""CPU-only contract tests for the guarded stage-faithful evaluation launcher."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "tools" / "stage_faithful_eval_launch.py"
SUBMIT = REPO_ROOT / "tools" / "slurm" / "submit_stage_faithful_eval.sh"
SLOT = REPO_ROOT / "tools" / "slurm" / "stage_faithful_eval.sbatch"


def load_helper():
    sys.path.insert(0, str(HELPER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "stage_faithful_eval_launch_test_module", HELPER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class StageFaithfulEvalLauncherTest(unittest.TestCase):
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

    def test_job_is_fixed_non_array_single_node_eight_b200(self):
        submit = SUBMIT.read_text(encoding="utf-8")
        slot = SLOT.read_text(encoding="utf-8")
        for fragment in (
            "--nodes=1",
            "--ntasks=1",
            "--gpus-per-node=8",
            "--cpus-per-task=64",
            "--mem=600G",
            "--time=00:30:00",
            "--partition=batch",
            "--no-requeue",
        ):
            self.assertIn(fragment, submit)
        self.assertNotIn("--array=", submit)
        self.assertIn("#SBATCH --nodes=1", slot)
        self.assertIn("#SBATCH --gpus-per-node=8", slot)
        self.assertIn("#SBATCH --time=00:30:00", slot)
        self.assertIn('[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]', slot)

    def test_active_jobs_are_exact_fail_closed_and_checked_twice(self):
        empty = self.helper._active_job_contract([], [])
        self.assertEqual(empty["allowed_array_or_job_ids"], [])
        self.assertFalse(empty["wildcards_or_names_allowed"])

        allowed = self.helper._active_job_contract(
            ["472969", "472974"],
            [
                "472969|472969|PENDING",
                "472974|472974|RUNNING",
            ],
        )
        self.assertTrue(allowed["all_observed_jobs_explicitly_allowed"])
        with self.assertRaisesRegex(RuntimeError, "not explicitly allowed"):
            self.helper._active_job_contract(
                ["472969"],
                ["472969|472969|RUNNING", "472974|472974|PENDING"],
            )
        with self.assertRaisesRegex(RuntimeError, "were not active"):
            self.helper._active_job_contract(["472969"], [])
        with self.assertRaisesRegex(ValueError, "positive numeric"):
            self.helper._active_job_contract(["472*"], [])

        source = SUBMIT.read_text(encoding="utf-8")
        self.assertEqual(source.count("\ncheck_active_user_jobs\n"), 2)
        self.assertIn("--format='%F|%i|%T'", source)
        self.assertNotIn("--states=", source)
        self.assertIn("IMMEDIATE_PRE_SBATCH_JOB_ROWS", source)

    def test_evaluation_id_is_commit_bound(self):
        commit = "a" * 40
        self.assertEqual(
            self.helper.expected_eval_id(commit),
            "abc200-tf-cascade-stage-eval-s1234-aaaaaaaaaa-v1",
        )
        with self.assertRaises(ValueError):
            self.helper.expected_eval_id("abc")

    def test_exact_audited_hydra_vector(self):
        eval_id = self.helper.expected_eval_id("a" * 40)
        output = Path("/approved/eval") / eval_id
        overrides = self.helper.expected_hydra_overrides(eval_id, output)
        self.assertEqual(len(overrides), 52)
        self.assertEqual(
            overrides[:5],
            [
                (
                    "+experiments_0908="
                    "ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml"
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
            (
                "trainer.config.saving.save_path="
                f"{output}/_never_write_snapshot.pt"
            ),
            "model.dual_diffusion.condition_on_tf=true",
            "model.dual_diffusion.condition_mode='matched'",
            "model.dual_diffusion.state_gate_init=0.10",
            "model.dual_diffusion.state_gate_trainable=false",
            "model.dual_diffusion.schedule_mode=tf_first_cascaded",
            "model.dual_diffusion.cascade_stage_faithful_inference=true",
            "model.dual_diffusion.evaluation_nfe_steps=[2,4,8]",
            (
                "model.dual_diffusion.evaluation_condition_sources="
                "[autonomous,autonomous_shuffled,autonomous_legacy,off]"
            ),
            "model.time_frequency_transform.representation='parseval_rfft'",
            "+wandb.group=null",
            "+wandb.resume=never",
            "+stage_faithful_evaluation.viz_skip_batches=4",
            "+stage_faithful_evaluation.artifact_iteration=199",
        }
        self.assertTrue(required.issubset(set(overrides)))
        self.assertEqual(
            overrides[-6:],
            [
                f"+stage_faithful_evaluation.snapshot_path={self.helper.CHECKPOINT}",
                (
                    "+stage_faithful_evaluation.snapshot_sha256="
                    f"{self.helper.CHECKPOINT_SHA256}"
                ),
                (
                    "+stage_faithful_evaluation.parent_run_identity_sha256="
                    f"{self.helper.PARENT_RUN_IDENTITY}"
                ),
                "+stage_faithful_evaluation.parent_completed_updates=200",
                "+stage_faithful_evaluation.viz_skip_batches=4",
                "+stage_faithful_evaluation.artifact_iteration=199",
            ],
        )

    def test_slot_invokes_evaluator_without_training_or_resume(self):
        source = SLOT.read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node=8", source)
        self.assertIn("evaluate_stage_faithful.py", source)
        self.assertNotIn(" train.py", source)
        self.assertNotIn("--resume", source)
        self.assertIn("_never_write_snapshot.pt", source)
        self.assertIn('"$OUTPUT_ROOT/training_complete.json"', source)
        self.assertIn("stage_faithful_evaluation_provenance.json", source)

    def test_parent_checkpoint_and_data_are_immutable_constants(self):
        observed_bytes = (
            self.helper.CHECKPOINT.stat().st_size
            if self.helper.CHECKPOINT.exists()
            else self.helper.CHECKPOINT_BYTES
        )
        self.assertEqual(observed_bytes, 4_249_340_573)
        self.assertEqual(
            self.helper.CHECKPOINT_SHA256,
            "5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d",
        )
        self.assertEqual(
            self.helper.ABC_MANIFEST_SHA256,
            "e52232b49ffec39600aa22e2d708497f22a4ea57fc89f84bc289ae4b1e0a5c09",
        )
        self.assertEqual(
            set(self.helper.PARENT_FILES),
            {
                "resolved_config",
                "arm_manifest",
                "outcome",
                "training_completion",
            },
        )

    def test_completion_requires_exactly_eight_rank_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            dataset = output / "visualization" / "iter_199" / "MultiDatasetABC_0"
            dataset.mkdir(parents=True)
            manifest_path = output / "eval_manifest.json"
            manifest = self.helper.pilot._identity_payload(
                {
                    "schema_version": 1,
                    "kind": "stage_faithful_cascade_evaluation",
                    "evaluation_id": "test-eval",
                    "output": {"output_root": str(output)},
                }
            )
            self.helper.pilot._exclusive_json(manifest_path, manifest)

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

            inventory_path = output / "eval_artifact_inventory.json"
            completion_path = output / "evaluation_complete.json"
            status = self.helper.command_complete(
                SimpleNamespace(
                    manifest=str(manifest_path),
                    inventory_output=str(inventory_path),
                    completion_output=str(completion_path),
                )
            )
            self.assertEqual(status, 0)
            inventory = json.loads(inventory_path.read_text())
            completion = json.loads(completion_path.read_text())
            self.assertTrue(self.helper._identity_is_valid(inventory))
            self.assertTrue(self.helper._identity_is_valid(completion))
            self.assertEqual(inventory["primary_artifact_count"], 8)
            self.assertEqual(
                [record["rank"] for record in inventory["ranks"]],
                list(range(8)),
            )
            self.assertEqual(completion["rank_artifact_count"], 8)
            self.assertEqual(completion["training_steps"], 0)

    def test_completion_rejects_missing_rank_and_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            dataset = output / "visualization" / "iter_199" / "dataset"
            dataset.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "exactly 8"):
                self.helper._artifact_inventory(output)
            (output / "training_complete.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                self.helper._artifact_inventory(output)

    def test_submission_rejects_array_job_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "eval_manifest.json"
            manifest = self.helper.pilot._identity_payload(
                {
                    "slurm": {
                        "active_job_coexistence": self.helper._active_job_contract(
                            [], []
                        )
                    }
                }
            )
            self.helper.pilot._exclusive_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "non-array"):
                self.helper.command_record_submission(
                    SimpleNamespace(
                        manifest=str(manifest_path),
                        job_id="123_0",
                        observed_active_job=[],
                        output=str(root / "submission.json"),
                    )
                )


if __name__ == "__main__":
    unittest.main()
