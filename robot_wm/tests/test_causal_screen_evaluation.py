import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "tools" / "evaluate_causal_screen_snapshots.py"
SUBMIT = REPO_ROOT / "tools" / "slurm" / "submit_causal_screen_evaluation.sh"
SLOT = (
    REPO_ROOT
    / "tools"
    / "slurm"
    / "evaluate_causal_screen_snapshots.sbatch"
)


def load_helper():
    sys.path.insert(0, str(HELPER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "causal_screen_evaluation_test_module", HELPER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class TinyStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("count", torch.zeros(1, dtype=torch.int64))


def make_artifacts(helper, *, mode: str, seed: int, error: float):
    video_clean = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    tf_clean = torch.tensor(
        [2.0, 3.0, 4.0], dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1)
    ground_truth = torch.tensor(
        [
            [
                [[[20]], [[40]]],
                [[[30]], [[50]]],
                [[[40]], [[60]]],
            ]
        ],
        dtype=torch.uint8,
    )
    generator = torch.Generator().manual_seed(seed)
    video_initial = torch.randn(
        video_clean.shape, generator=generator, dtype=torch.float32
    )
    tf_noise = torch.randn(
        tf_clean.shape, generator=generator, dtype=torch.float32
    )
    tf_initial = tf_noise.clone()
    tf_initial[:, :, :1] = tf_clean[:, :, :1]
    artifacts = {
        "video_clean": video_clean,
        "tf_clean": tf_clean,
        "video_initial_state": video_initial,
        "tf_initial_state": tf_initial,
        "tf_initial_noise": tf_noise,
        "ground_truth_future_uint8": ground_truth,
        "history_latent_frames": torch.tensor([1], dtype=torch.int64),
        "condition_on_tf": torch.tensor(
            [int(mode != "off")], dtype=torch.int64
        ),
        "condition_mode_code": torch.tensor(
            [helper.MODE_CODES[mode]], dtype=torch.int64
        ),
        "evaluation_noise_seed": torch.tensor([seed], dtype=torch.int64),
        "evaluation_nfe_steps": torch.tensor(
            helper.NFE_STEPS, dtype=torch.int64
        ),
        "evaluation_condition_source_codes": torch.tensor(
            [helper.SOURCE_CODES[source] for source in helper.CONDITION_SOURCES],
            dtype=torch.int64,
        ),
        "oracle_sources_are_leakage": torch.tensor([1], dtype=torch.int64),
    }
    for source_index, source in enumerate(helper.CONDITION_SOURCES):
        infix = "" if source == "autonomous" else f"_{source}"
        for nfe in helper.NFE_STEPS:
            scale = error * (1.0 + 0.1 * source_index) / nfe
            artifacts[f"video_final{infix}_nfe_{nfe}"] = (
                video_clean + scale
            )
            artifacts[f"tf_final{infix}_nfe_{nfe}"] = (
                tf_clean + 2.0 * scale
            )
            pixel_error = max(1, round(20.0 * scale))
            artifacts[f"decoded_future{infix}_nfe_{nfe}"] = torch.clamp(
                ground_truth.to(torch.int16) + pixel_error, 0, 255
            ).to(torch.uint8)
    return artifacts


class CausalScreenEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper()

    def test_batch_hash_is_order_stable_and_content_sensitive(self):
        left = {
            "rgb": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            "nested": {"mask": torch.tensor([[True, False]])},
        }
        reordered = {"nested": left["nested"], "rgb": left["rgb"].clone()}
        changed = {
            "rgb": left["rgb"].clone(),
            "nested": {"mask": torch.tensor([[False, False]])},
        }
        self.assertEqual(
            self.helper._batch_sha256(left),
            self.helper._batch_sha256(reordered),
        )
        self.assertNotEqual(
            self.helper._batch_sha256(left),
            self.helper._batch_sha256(changed),
        )

    def test_per_pair_seed_is_unique_and_replayable(self):
        observed = {
            self.helper._effective_generator_seed(rank, batch)
            for rank in range(self.helper.WORLD_SIZE)
            for batch in range(self.helper.MAX_BATCHES_PER_RANK)
        }
        self.assertEqual(
            len(observed),
            self.helper.WORLD_SIZE * self.helper.MAX_BATCHES_PER_RANK,
        )
        self.assertEqual(
            self.helper._effective_generator_seed(7, 3),
            self.helper.BASE_EVALUATION_NOISE_SEED + 7 + 8 * 3,
        )

    def test_arm_subset_is_guarded_and_put_in_fixed_collective_order(self):
        self.assertEqual(
            self.helper._selected_arm_names(
                "shuffled_s010,matched_s010"
            ),
            ["matched_s010", "shuffled_s010"],
        )
        with self.assertRaisesRegex(
            self.helper.EvaluationContractError, "at least two"
        ):
            self.helper._selected_arm_names("matched_s010")
        with self.assertRaisesRegex(
            self.helper.EvaluationContractError, "unknown"
        ):
            self.helper._selected_arm_names("matched_s010,unknown")
        with self.assertRaisesRegex(
            self.helper.EvaluationContractError, "unique"
        ):
            self.helper._selected_arm_names(
                "matched_s010,matched_s010"
            )

    def test_batch_index_is_an_explicit_dataset_pair_component(self):
        first = self.helper._pair_dataset_name("MultiDatasetABC_0", 0)
        second = self.helper._pair_dataset_name("MultiDatasetABC_0", 1)
        self.assertEqual(first, "MultiDatasetABC_0__batch_000")
        self.assertEqual(second, "MultiDatasetABC_0__batch_001")
        self.assertNotEqual(first, second)

    def test_strict_state_contract_rejects_key_shape_and_dtype_changes(self):
        model_state = TinyStateModel().state_dict()
        self.helper._strict_state_dict_contract(
            model_state, {key: value.clone() for key, value in model_state.items()}
        )
        cases = {
            "missing": {"weight": model_state["weight"].clone()},
            "extra": {
                **{key: value.clone() for key, value in model_state.items()},
                "extra": torch.zeros(1),
            },
            "shape": {
                "weight": torch.zeros(3, 2),
                "count": model_state["count"].clone(),
            },
            "dtype": {
                "weight": model_state["weight"].double(),
                "count": model_state["count"].clone(),
            },
        }
        for name, checkpoint_state in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(self.helper.EvaluationContractError):
                    self.helper._strict_state_dict_contract(
                        model_state, checkpoint_state
                    )

    def test_snapshot_envelope_is_exact(self):
        identity = "a" * 64
        state = TinyStateModel().state_dict()
        valid = {
            "snapshot_schema_version": 3,
            "_start_iter": 200,
            "world_size": 8,
            "gradient_accumulation_steps": 1,
            "run_identity_sha256": identity,
            "model": state,
        }
        self.assertIs(
            self.helper._validate_snapshot_envelope(
                valid, expected_run_identity=identity
            ),
            state,
        )
        for key, value in (
            ("snapshot_schema_version", 2),
            ("_start_iter", 199),
            ("world_size", 4),
            ("gradient_accumulation_steps", 2),
            ("run_identity_sha256", "b" * 64),
        ):
            broken = dict(valid)
            broken[key] = value
            with self.subTest(key=key):
                with self.assertRaises(self.helper.EvaluationContractError):
                    self.helper._validate_snapshot_envelope(
                        broken, expected_run_identity=identity
                    )

    def test_corrected_commit_contract_is_present_on_current_main(self):
        training_commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "b1738f9"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        evaluation_commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        record = self.helper._validate_corrected_commit(
            REPO_ROOT, training_commit, evaluation_commit
        )
        self.assertTrue(record["training_commit_is_ancestor"])
        self.assertEqual(set(record["files"]), set(self.helper.CORRECTION_PATHS))

    def test_training_and_evaluation_repositories_are_distinct(self):
        def git(repository, *arguments):
            return subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"
            evaluation = root / "evaluation"
            training.mkdir()
            git(training, "init")
            git(training, "config", "user.name", "Evaluator Test")
            git(training, "config", "user.email", "evaluator@example.invalid")
            (training / "model.py").write_text(
                "TRAINING_VERSION = 1\n", encoding="utf-8"
            )
            git(training, "add", "model.py")
            git(training, "commit", "-m", "training state")
            training_commit = git(training, "rev-parse", "HEAD")

            subprocess.run(
                ["git", "clone", "--no-hardlinks", str(training), str(evaluation)],
                check=True,
                capture_output=True,
                text=True,
            )
            git(evaluation, "config", "user.name", "Evaluator Test")
            git(
                evaluation,
                "config",
                "user.email",
                "evaluator@example.invalid",
            )
            (evaluation / "evaluation.py").write_text(
                "CORRECTED_SAMPLER = True\n", encoding="utf-8"
            )
            git(evaluation, "add", "evaluation.py")
            git(evaluation, "commit", "-m", "corrected evaluator")

            record = self.helper._validate_training_repository(
                str(training.resolve()),
                evaluation_repo_root=evaluation.resolve(),
                training_commit=training_commit,
            )
            self.assertEqual(record["path"], str(training.resolve()))
            self.assertEqual(record["git_commit"], training_commit)
            self.assertTrue(record["clean"])
            self.assertTrue(
                record["disjoint_from_evaluation_repository"]
            )
            self.assertNotEqual(
                git(evaluation, "rev-parse", "HEAD"),
                record["git_commit"],
            )
            with self.assertRaisesRegex(
                self.helper.EvaluationContractError, "disjoint"
            ):
                self.helper._validate_training_repository(
                    str(evaluation.resolve()),
                    evaluation_repo_root=evaluation.resolve(),
                    training_commit=training_commit,
                )

    def test_collated_abc_multi_dataset_batch_contains_only_tensors(self):
        sample = {
            "rgb": torch.zeros(3, 4, 8, 8),
            "actions": torch.zeros(4, 32),
            "mask": torch.ones(4, dtype=torch.bool),
            "dataset_index": torch.tensor(0, dtype=torch.long),
            "camera_index": torch.zeros(4, dtype=torch.long),
            "camera_mask": torch.ones(4),
            "morphology_index": torch.tensor(0, dtype=torch.long),
            "ee_action_dim": torch.tensor(23, dtype=torch.int32),
            "decode_camera": True,
        }
        batch = torch.utils.data.default_collate([sample])
        self.assertTrue(
            all(isinstance(value, torch.Tensor) for value in batch.values())
        )
        self.assertEqual(batch["decode_camera"].dtype, torch.bool)
        self.assertEqual(batch["decode_camera"].shape, (1,))
        flattened = self.helper._flatten_batch_tensors(batch)
        self.assertEqual(set(flattened), set(sample))

    def test_output_candidate_must_be_fresh_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            screen = root / "screen"
            analysis.mkdir()
            screen.mkdir()
            candidate = self.helper._evaluation_root_candidate(
                analysis_root=analysis,
                evaluation_id="fresh-eval",
                screen_root=screen,
            )
            self.assertEqual(candidate, analysis / "fresh-eval")
            candidate.mkdir()
            with self.assertRaisesRegex(
                self.helper.EvaluationContractError, "fresh"
            ):
                self.helper._evaluation_root_candidate(
                    analysis_root=analysis,
                    evaluation_id="fresh-eval",
                    screen_root=screen,
                )

    def test_wandb_runtime_guard_requires_disabled_and_no_run_identity(self):
        with mock.patch.dict(
            os.environ,
            {"WANDB_MODE": "disabled", "WANDB_DISABLED": "true"},
            clear=True,
        ):
            self.helper._runtime_wandb_guard()
        with mock.patch.dict(
            os.environ,
            {
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
                "WANDB_RUN_ID": "forbidden",
            },
            clear=True,
        ):
            with self.assertRaises(self.helper.EvaluationContractError):
                self.helper._runtime_wandb_guard()

    def test_artifact_metrics_and_full_source_inventory(self):
        arm = {
            "name": "matched_s010",
            "condition_on_tf": True,
            "condition_mode": "matched",
            "state_gate_init": 0.1,
        }
        artifacts = make_artifacts(
            self.helper,
            mode="matched",
            seed=self.helper._model_noise_seed(0),
            error=0.2,
        )
        metrics, identity = self.helper._validate_and_measure_artifacts(
            artifacts,
            arm=arm,
            expected_model_seed=self.helper._model_noise_seed(0),
        )
        self.assertEqual(set(metrics), set(self.helper.CONDITION_SOURCES))
        for source in self.helper.CONDITION_SOURCES:
            self.assertEqual(
                set(metrics[source]),
                {str(nfe) for nfe in self.helper.NFE_STEPS},
            )
        self.assertEqual(
            identity["evaluation_condition_sources"],
            list(self.helper.CONDITION_SOURCES),
        )

    def test_multiple_batches_are_consumed_by_existing_analyzer(self):
        from tools.analyze_dual_nfe_artifacts import analyze

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_root = root / "evaluation"
            evaluation_root.mkdir()
            arms = (
                {
                    "name": "matched_s010",
                    "condition_on_tf": True,
                    "condition_mode": "matched",
                    "state_gate_init": 0.1,
                },
                {
                    "name": "shuffled_s010",
                    "condition_on_tf": True,
                    "condition_mode": "shuffled",
                    "state_gate_init": 0.1,
                },
            )
            for batch_index in range(2):
                seed = self.helper._model_noise_seed(batch_index)
                dataset = self.helper._pair_dataset_name("ABC_0", batch_index)
                reference_identity = None
                for arm_index, arm in enumerate(arms):
                    artifacts = make_artifacts(
                        self.helper,
                        mode=arm["condition_mode"],
                        seed=seed,
                        error=0.2 + 0.2 * arm_index,
                    )
                    metrics, identity = (
                        self.helper._validate_and_measure_artifacts(
                            artifacts,
                            arm=arm,
                            expected_model_seed=seed,
                        )
                    )
                    if reference_identity is None:
                        reference_identity = identity
                    self.assertEqual(identity, reference_identity)
                    self.helper._save_output_artifact(
                        evaluation_root=evaluation_root,
                        arm=arm,
                        dataset=dataset,
                        rank=0,
                        batch_index=batch_index,
                        batch_sha256=f"{batch_index + 1:064x}",
                        model_seed=seed,
                        effective_seed=seed,
                        artifacts=artifacts,
                        metrics=metrics,
                        paired_identity=identity,
                        aggregate_sampling_timing={
                            "definition": "test aggregate",
                            "interpretation": "not generation latency",
                            "seconds_by_global_rank": [1.0] * 8,
                            "min_seconds": 1.0,
                            "mean_seconds": 1.0,
                            "max_seconds": 1.0,
                        },
                        snapshot={"sha256": "a" * 64},
                        resolved_config={"sha256": "b" * 64},
                        evaluation_commit="c" * 40,
                    )
            analysis_root = root / "analysis"
            analysis_root.mkdir()
            arm_roots = {
                arm["name"]: (
                    evaluation_root
                    / "artifacts"
                    / arm["name"]
                    / f"iter_{self.helper.EVALUATION_ITERATION}"
                )
                for arm in arms
            }
            payload = analyze(
                arm_roots,
                baseline="shuffled_s010",
                output=analysis_root / "result.json",
                bootstrap_samples=100,
                bootstrap_seed=123,
            )
            self.assertEqual(payload["provenance"]["paired_unit_count"], 2)
            pair_names = {
                unit["dataset"] for unit in payload["per_paired_unit"]
            }
            self.assertEqual(
                pair_names,
                {
                    "ABC_0__batch_000",
                    "ABC_0__batch_001",
                },
            )

    def test_shell_entrypoints_are_guarded_and_parse(self):
        result = subprocess.run(
            ["bash", "-n", str(SUBMIT), str(SLOT)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        submit = SUBMIT.read_text(encoding="utf-8")
        slot = SLOT.read_text(encoding="utf-8")
        for fragment in (
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
            'git -C "$REPO_ROOT" status --porcelain',
            "--no-requeue",
            "--batches-per-rank",
            "--arms",
            "W&B: disabled",
            '[[ ! -e "$EVALUATION_ROOT" ]]',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, submit)
        for fragment in (
            "#SBATCH --no-requeue",
            '[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]]',
            "WANDB_MODE=disabled",
            "WANDB_DISABLED=true",
            "--nproc_per_node=8",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, slot)
        self.assertNotIn("scontrol requeue", submit + slot)
        self.assertNotIn("--resume", submit + slot)
        self.assertNotIn("--array", submit + slot)


if __name__ == "__main__":
    unittest.main()
