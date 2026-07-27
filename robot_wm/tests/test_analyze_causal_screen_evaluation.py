import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
WRAPPER_PATH = TOOLS_ROOT / "analyze_causal_screen_evaluation.py"


def load_wrapper():
    sys.path.insert(0, str(TOOLS_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            "completed_evaluation_analysis_test_module",
            WRAPPER_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class CompletedEvaluationFixture:
    def __init__(self, root: Path, wrapper):
        self.root = root
        self.wrapper = wrapper
        self.evaluator = wrapper.evaluator
        self.training_repo = root / "training_repo"
        self.evaluation_repo = root / "evaluation_repo"
        self.screen_root = root / "screen"
        self.evaluation_root = root / "evaluation"
        self.analysis_root = root / "analysis"
        self.data_root = root / "data"
        self.wan_root = root / "wan"
        self.videox_root = root / "videox"
        self.selected_arms = list(wrapper.CANONICAL_ARM_ORDER[:2])
        self._create_repositories()
        self._create_source_inputs()
        self._create_completed_evaluation()

    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _identity_file(self, path: Path, payload) -> dict:
        value = self.evaluator._identity_payload(payload)
        self._write_json(path, value)
        return value

    def _create_repositories(self) -> None:
        self.training_repo.mkdir()
        self._git(self.training_repo, "init")
        self._git(self.training_repo, "config", "user.name", "Fixture")
        self._git(
            self.training_repo,
            "config",
            "user.email",
            "fixture@example.invalid",
        )
        (self.training_repo / "model.py").write_text(
            "TRAINING = True\n",
            encoding="utf-8",
        )
        self._git(self.training_repo, "add", "model.py")
        self._git(self.training_repo, "commit", "-m", "training")
        self.training_commit = self._git(
            self.training_repo,
            "rev-parse",
            "HEAD",
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                str(self.training_repo),
                str(self.evaluation_repo),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self._git(self.evaluation_repo, "config", "user.name", "Fixture")
        self._git(
            self.evaluation_repo,
            "config",
            "user.email",
            "fixture@example.invalid",
        )
        (self.evaluation_repo / "corrected.py").write_text(
            "CORRECTED = True\n",
            encoding="utf-8",
        )
        self._git(self.evaluation_repo, "add", "corrected.py")
        self._git(self.evaluation_repo, "commit", "-m", "evaluation")
        self.evaluation_commit = self._git(
            self.evaluation_repo,
            "rev-parse",
            "HEAD",
        )

        self.videox_root.mkdir()
        self._git(self.videox_root, "init")
        self._git(self.videox_root, "config", "user.name", "Fixture")
        self._git(
            self.videox_root,
            "config",
            "user.email",
            "fixture@example.invalid",
        )
        scheduler = (
            self.videox_root
            / "config"
            / "wan2.1"
            / "wan_civitai.yaml"
        )
        scheduler.parent.mkdir(parents=True)
        scheduler.write_text("scheduler: fixture\n", encoding="utf-8")
        self._git(self.videox_root, "add", "config/wan2.1/wan_civitai.yaml")
        self._git(self.videox_root, "commit", "-m", "scheduler")
        self.videox_commit = self._git(
            self.videox_root,
            "rev-parse",
            "HEAD",
        )

    def _create_source_inputs(self) -> None:
        self.screen_root.mkdir()
        screen = self._identity_file(
            self.screen_root / "screen_manifest.json",
            {
                "schema_version": 1,
                "kind": "dual_abc_tf_causal_screen",
                "screen_id": "fixture-screen",
            },
        )
        self.source_screen = {
            **self.evaluator._file_record(
                self.screen_root / "screen_manifest.json"
            ),
            "identity_sha256": screen["identity_sha256"],
            "screen_id": "fixture-screen",
            "screen_root": str(self.screen_root),
        }
        arm_specs = {
            arm["name"]: arm for arm in self.evaluator.causal.ARMS
        }
        self.source_arms = {}
        for arm_name in self.wrapper.CANONICAL_ARM_ORDER:
            arm_dir = self.screen_root / arm_name
            arm_dir.mkdir()
            arm_manifest = self._identity_file(
                arm_dir / "arm_manifest.json",
                {
                    "schema_version": 1,
                    "kind": "dual_abc_tf_causal_screen_arm",
                    "arm": arm_name,
                    "git_commit": self.training_commit,
                },
            )
            resolved = arm_dir / "resolved_config.yaml"
            resolved.write_text(f"arm: {arm_name}\n", encoding="utf-8")
            snapshot = arm_dir / "snapshot.pt"
            snapshot.write_bytes(f"snapshot:{arm_name}".encode())
            completion = arm_dir / "training_complete.json"
            self._write_json(
                completion,
                {
                    "status": "completed",
                    "completed_updates": self.evaluator.COMPLETED_UPDATES,
                    "snapshot": str(snapshot),
                },
            )
            outcome = arm_dir / "outcome.json"
            self._write_json(
                outcome,
                {"completed": True, "exit_status": 0},
            )
            spec = arm_specs[arm_name]
            self.source_arms[arm_name] = {
                "array_task_id": list(
                    self.wrapper.CANONICAL_ARM_ORDER
                ).index(arm_name),
                "condition_on_tf": spec["condition_on_tf"],
                "condition_mode": spec["condition_mode"],
                "state_gate_init": spec["state_gate_init"],
                "state_gate_trainable": False,
                "source_directory": str(arm_dir),
                "arm_manifest": {
                    **self.evaluator._file_record(
                        arm_dir / "arm_manifest.json"
                    ),
                    "identity_sha256": arm_manifest["identity_sha256"],
                },
                "resolved_config": self.evaluator._file_record(resolved),
                "training_completion": self.evaluator._file_record(completion),
                "outcome": self.evaluator._file_record(outcome),
                "snapshot": self.evaluator._file_record(snapshot),
                "viz_dataset_sha256": "1" * 64,
                "viz_loader_sha256": "2" * 64,
            }

        self.wan_root.mkdir()
        null_prompt = self.wan_root / "null_prompt_umt5.pt"
        null_prompt.write_bytes(b"null prompt")
        abc_manifest = self.data_root / "abc_pp" / "manifest.txt"
        abc_manifest.parent.mkdir(parents=True)
        abc_manifest.write_text("/fixture/episode\n", encoding="utf-8")
        scheduler = (
            self.videox_root
            / "config"
            / "wan2.1"
            / "wan_civitai.yaml"
        )
        self.assets = {
            "wan_directory": str(self.wan_root),
            "videox_home": str(self.videox_root),
            "videox_commit": self.videox_commit,
            "null_prompt": self.evaluator._file_record(null_prompt),
            "scheduler_config": self.evaluator._file_record(scheduler),
        }
        self.data = {
            "root": str(self.data_root),
            "datasets": ["ABC"],
            "abc_manifest": {
                "path": str(abc_manifest),
                "sha256": self.evaluator._sha256(abc_manifest),
                "entries": 1,
                "first_entry": "/fixture/episode",
                "last_entry": "/fixture/episode",
            },
            "viz_dataset_sha256": "1" * 64,
            "viz_loader_sha256": "2" * 64,
        }

    def _paired_contract(self):
        return {
            "world_size": self.evaluator.WORLD_SIZE,
            "batches_per_rank": 1,
            "paired_unit_count": self.evaluator.WORLD_SIZE,
            "batch_size": 1,
            "input_loader": "fixture",
            "arm_order": list(self.selected_arms),
            "arm_subset_is_explicit": True,
            "evaluation_iteration": self.evaluator.EVALUATION_ITERATION,
            "nfe_steps": list(self.evaluator.NFE_STEPS),
            "condition_sources": list(self.evaluator.CONDITION_SOURCES),
            "oracle_sources_are_leakage": True,
            "base_evaluation_noise_seed": (
                self.evaluator.BASE_EVALUATION_NOISE_SEED
            ),
            "model_seed_for_batch": (
                "base_evaluation_noise_seed + world_size * batch_index"
            ),
            "effective_generator_seed": (
                "base_evaluation_noise_seed + global_rank + "
                "world_size * batch_index"
            ),
            "sigma_convention": self.evaluator.SIGMA_CONVENTION,
            "no_cross_example_averaging": True,
        }

    def _create_completed_evaluation(self) -> None:
        self.evaluation_root.mkdir()
        self.analysis_root.mkdir()
        (self.evaluation_root / "inputs").mkdir()
        artifact_root = self.evaluation_root / "artifacts"
        artifact_root.mkdir()
        arm_roots = {}
        for arm_name in self.selected_arms:
            arm_root = (
                artifact_root
                / arm_name
                / f"iter_{self.evaluator.EVALUATION_ITERATION}"
            )
            arm_root.mkdir(parents=True)
            arm_roots[arm_name] = str(arm_root)
        paired = self._paired_contract()
        training_repository = self.evaluator._validate_training_repository(
            str(self.training_repo),
            evaluation_repo_root=self.evaluation_repo,
            training_commit=self.training_commit,
        )
        manifest_unsigned = {
            "schema_version": self.evaluator.SCHEMA_VERSION,
            "kind": self.wrapper.MANIFEST_KIND,
            "created_at_utc": "2026-07-26T00:00:00+00:00",
            "evaluation_id": "fixture-evaluation",
            "repository_root": str(self.evaluation_repo),
            "evaluation_root": str(self.evaluation_root),
            "training_commit": self.training_commit,
            "evaluation_commit": self.evaluation_commit,
            "training_repository": training_repository,
            "corrected_sampler": {"fixture": True},
            "source_screen": self.source_screen,
            "data": self.data,
            "assets": self.assets,
            "source_arm_inventory": self.source_arms,
            "arms": {
                name: self.source_arms[name] for name in self.selected_arms
            },
            "outputs": {
                "analyzer_compatible": True,
                "paired_key": ["dataset", "global_rank"],
                "batch_index_encoding": "dataset suffix",
                "arm_roots": arm_roots,
            },
            "paired_evaluation": paired,
            "execution": {
                "evaluation_only": True,
                "strict_snapshot_load": True,
                "sequential_arm_load": True,
                "wandb_enabled": False,
                "resume": False,
                "requeue": False,
                "source_writes_performed": False,
                "condition_source_subset_supported": False,
                "nfe_subset_supported": False,
            },
        }
        self.manifest = self.evaluator._identity_payload(manifest_unsigned)
        self.manifest_path = self.evaluation_root / self.wrapper.MANIFEST_NAME
        self._write_json(self.manifest_path, self.manifest)

        self.input_records = []
        self.output_records = []
        arm_specs = {
            arm["name"]: arm for arm in self.evaluator.causal.ARMS
        }
        for rank in range(self.evaluator.WORLD_SIZE):
            batch_index = 0
            dataset = "ABC_0__batch_000"
            batch = {
                "rgb": torch.full((1, 2, 3), float(rank)),
                "actions": torch.full((1, 2), float(rank + 1)),
                "mask": torch.ones(1, dtype=torch.bool),
            }
            summary = self.evaluator._batch_summary(batch)
            self.input_records.append(
                self.evaluator._save_input_batch(
                    evaluation_root=self.evaluation_root,
                    dataset=dataset,
                    rank=rank,
                    batch_index=batch_index,
                    batch=batch,
                    batch_summary=summary,
                )
            )
            model_seed = self.evaluator._model_noise_seed(batch_index)
            effective_seed = self.evaluator._effective_generator_seed(
                rank,
                batch_index,
            )
            paired_identity = {
                "fixture_rank": rank,
                "evaluation_noise_seed": model_seed,
            }
            for arm_name in self.selected_arms:
                arm_spec = arm_specs[arm_name]
                source_arm = self.source_arms[arm_name]
                snapshot = {
                    **source_arm["snapshot"],
                    "snapshot_schema_version": 3,
                    "next_iteration": self.evaluator.COMPLETED_UPDATES,
                    "world_size": self.evaluator.WORLD_SIZE,
                    "gradient_accumulation_steps": 1,
                    "strict_key_shape_dtype_match": True,
                    "effective_state_gate": arm_spec["state_gate_init"],
                }
                self.output_records.append(
                    self.evaluator._save_output_artifact(
                        evaluation_root=self.evaluation_root,
                        arm=arm_spec,
                        dataset=dataset,
                        rank=rank,
                        batch_index=batch_index,
                        batch_sha256=summary["sha256"],
                        model_seed=model_seed,
                        effective_seed=effective_seed,
                        artifacts={"fixture": torch.tensor([rank])},
                        metrics={"fixture": rank},
                        paired_identity=paired_identity,
                        aggregate_sampling_timing={"seconds": 1.0},
                        snapshot=snapshot,
                        resolved_config=source_arm["resolved_config"],
                        evaluation_commit=self.evaluation_commit,
                    )
                )
        inventory = self.evaluator._validate_final_inventory(
            manifest=self.manifest,
            input_records=self.input_records,
            output_records=self.output_records,
        )
        execution = self.evaluator._identity_payload(
            {
                "schema_version": self.evaluator.SCHEMA_VERSION,
                "kind": (
                    "dual_abc_tf_causal_screen_posthoc_execution_started"
                ),
                "created_at_utc": "2026-07-26T00:01:00+00:00",
                "evaluation_manifest_sha256": self.evaluator._sha256(
                    self.manifest_path
                ),
                "evaluation_manifest_identity_sha256": self.manifest[
                    "identity_sha256"
                ],
                "slurm_job_id": "123",
                "world_size": self.evaluator.WORLD_SIZE,
                "requeue": False,
                "resume": False,
                "wandb_enabled": False,
            }
        )
        self._write_json(
            self.evaluation_root / "execution_started.json",
            execution,
        )
        self.completion = self.evaluator._identity_payload(
            {
                "schema_version": self.evaluator.SCHEMA_VERSION,
                "kind": self.wrapper.COMPLETION_KIND,
                "completed_at_utc": "2026-07-26T00:02:00+00:00",
                "evaluation_manifest": {
                    "path": str(self.manifest_path),
                    "sha256": self.evaluator._sha256(self.manifest_path),
                    "identity_sha256": self.manifest["identity_sha256"],
                },
                "training_commit": self.training_commit,
                "evaluation_commit": self.evaluation_commit,
                "source_inputs_unchanged": True,
                "wandb_enabled": False,
                "resume": False,
                "requeue": False,
                "paired_evaluation": paired,
                "inventory": inventory,
                "timing": {"seconds": 10.0},
            }
        )
        self.completion_path = (
            self.evaluation_root / self.wrapper.COMPLETION_NAME
        )
        self._write_json(self.completion_path, self.completion)

    def resign_completion(self, mutate) -> str:
        unsigned = dict(self.completion)
        unsigned.pop("identity_sha256")
        mutate(unsigned)
        self.completion = self.evaluator._identity_payload(unsigned)
        self._write_json(self.completion_path, self.completion)
        return self.completion["identity_sha256"]

    def generic_payload(
        self,
        arms,
        *,
        baseline,
        output,
        bootstrap_samples,
        bootstrap_seed,
        confidence,
    ):
        paired_units = sorted(
            [
                {
                    "dataset": unit["dataset"],
                    "global_rank": unit["global_rank"],
                }
                for unit in self.completion["inventory"]["paired_units"]
            ],
            key=lambda item: (item["dataset"], item["global_rank"]),
        )
        payload = {
            "schema_version": 1,
            "kind": "dual_video_diffusion_matched_nfe_analysis",
            "baseline_arm": baseline,
            "bootstrap": {
                "method": (
                    "paired nonparametric percentile bootstrap over "
                    "dataset/rank units"
                ),
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "confidence": confidence,
                "per_statistic_seed_derivation": (
                    "sha256(seed + NUL + statistic label)"
                ),
            },
            "provenance": {
                "iteration": self.evaluator.EVALUATION_ITERATION,
                "paired_unit_count": len(paired_units),
                "paired_units": paired_units,
                "arms": {
                    name: {
                        "root": str(path),
                        "artifact_count": len(paired_units),
                    }
                    for name, path in arms.items()
                },
            },
        }
        Path(output).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        return payload


class CompletedEvaluationAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = load_wrapper()

    def fixture(self, temporary):
        return CompletedEvaluationFixture(Path(temporary), self.wrapper)

    def test_complete_inventory_delegates_only_manifest_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            output = fixture.analysis_root / "analysis.json"
            calls = []

            def analyze(arms, **kwargs):
                calls.append((dict(arms), dict(kwargs)))
                return fixture.generic_payload(arms, **kwargs)

            with mock.patch.object(
                self.wrapper.generic,
                "analyze",
                side_effect=analyze,
            ):
                payload = self.wrapper.analyze_completed_evaluation(
                    fixture.evaluation_root,
                    expected_completion_identity=fixture.completion[
                        "identity_sha256"
                    ],
                    baseline=fixture.selected_arms[0],
                    output=output,
                    bootstrap_samples=100,
                    bootstrap_seed=17,
                )
            expected_roots = {
                name: (
                    fixture.evaluation_root
                    / "artifacts"
                    / name
                    / f"iter_{fixture.evaluator.EVALUATION_ITERATION}"
                )
                for name in fixture.selected_arms
            }
            self.assertEqual(calls[0][0], expected_roots)
            self.assertEqual(
                payload["evaluation"]["paired_unit_count"],
                fixture.evaluator.WORLD_SIZE,
            )
            self.assertTrue(fixture.evaluator._identity_is_valid(payload))
            self.assertTrue(output.is_file())
            self.assertEqual(
                list(fixture.analysis_root.glob(".*.generic-*.json")),
                [],
            )

    def test_rank_zero_only_completion_is_rejected_before_analyzer(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)

            def mutate(completion):
                units = completion["inventory"]["paired_units"][:1]
                completion["inventory"]["paired_units"] = units
                completion["inventory"]["paired_unit_count"] = 1
                completion["inventory"]["input_artifact_count"] = 1
                completion["inventory"]["artifact_count"] = len(
                    fixture.selected_arms
                )

            expected_identity = fixture.resign_completion(mutate)
            with mock.patch.object(
                self.wrapper.generic,
                "analyze",
            ) as analyzer:
                with self.assertRaisesRegex(
                    self.wrapper.CompletedEvaluationError,
                    "counts|paired-unit",
                ):
                    self.wrapper.analyze_completed_evaluation(
                        fixture.evaluation_root,
                        expected_completion_identity=expected_identity,
                        baseline=fixture.selected_arms[0],
                        output=fixture.analysis_root / "partial.json",
                    )
            analyzer.assert_not_called()

    def test_completion_tamper_without_external_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            expected_identity = fixture.completion["identity_sha256"]
            payload = dict(fixture.completion)
            payload["source_inputs_unchanged"] = False
            fixture._write_json(fixture.completion_path, payload)
            with self.assertRaisesRegex(
                self.wrapper.CompletedEvaluationError,
                "identity",
            ):
                self.wrapper._validate_completed_evaluation(
                    fixture.evaluation_root,
                    expected_completion_identity=expected_identity,
                )

    def test_artifact_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            artifact_path = Path(
                fixture.completion["inventory"]["paired_units"][0][
                    "arm_artifacts"
                ][fixture.selected_arms[0]]["safetensors_path"]
            )
            with artifact_path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(
                self.wrapper.CompletedEvaluationError,
                "hash differs",
            ):
                self.wrapper._validate_completed_evaluation(
                    fixture.evaluation_root,
                    expected_completion_identity=fixture.completion[
                        "identity_sha256"
                    ],
                )

    def test_external_identical_input_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            original = Path(
                fixture.completion["inventory"]["paired_units"][0][
                    "input_artifact"
                ]["safetensors_path"]
            )
            external = fixture.root / "external-input.safetensors"
            external.write_bytes(original.read_bytes())

            def mutate(completion):
                completion["inventory"]["paired_units"][0][
                    "input_artifact"
                ]["safetensors_path"] = str(external)

            expected_identity = fixture.resign_completion(mutate)
            with self.assertRaisesRegex(
                self.wrapper.CompletedEvaluationError,
                "path differs",
            ):
                self.wrapper._validate_completed_evaluation(
                    fixture.evaluation_root,
                    expected_completion_identity=expected_identity,
                )

    def test_extra_artifact_is_rejected_by_exact_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            extra = (
                fixture.evaluation_root
                / "artifacts"
                / fixture.selected_arms[0]
                / f"iter_{fixture.evaluator.EVALUATION_ITERATION}"
                / "extra.txt"
            )
            extra.write_text("undeclared", encoding="utf-8")
            with self.assertRaisesRegex(
                self.wrapper.CompletedEvaluationError,
                "allowlist",
            ):
                self.wrapper._validate_completed_evaluation(
                    fixture.evaluation_root,
                    expected_completion_identity=fixture.completion[
                        "identity_sha256"
                    ],
                )

    def test_output_inside_evaluation_is_rejected_before_analyzer(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            with mock.patch.object(
                self.wrapper.generic,
                "analyze",
            ) as analyzer:
                with self.assertRaisesRegex(
                    self.wrapper.CompletedEvaluationError,
                    "outside",
                ):
                    self.wrapper.analyze_completed_evaluation(
                        fixture.evaluation_root,
                        expected_completion_identity=fixture.completion[
                            "identity_sha256"
                        ],
                        baseline=fixture.selected_arms[0],
                        output=fixture.evaluation_root / "analysis.json",
                    )
            analyzer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
