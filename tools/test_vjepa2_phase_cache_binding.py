"""Focused stdlib tests for immutable V-JEPA phase/cache study binding.

These tests intentionally avoid importing the model or benchmark modules.  The
fixtures use real files, directories, JSON identities, and symlink aliases so
the contract checks cannot pass solely because of a mocked filesystem.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from unittest import mock

import vjepa2_controlled_study as study
import vjepa2_phase_gate as phase_gate


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "tools" / "slurm" / "submit_vjepa2_controlled_study.sh"
STAGE_SBATCH = ROOT / "tools" / "slurm" / "vjepa2_controlled_study.sbatch"
PAIRED_SBATCH = ROOT / "tools" / "slurm" / "vjepa2_paired_latency.sbatch"
PHASE_HELPER = ROOT / "tools" / "vjepa2_phase_gate.py"
PHASE_SBATCH = ROOT / "tools" / "slurm" / "vjepa2_phase_gate.sbatch"
BASE_MODEL_CONFIG = (
    ROOT
    / "projects"
    / "latent_action_models"
    / "configs"
    / "models"
    / "dual_explicit_action_dit_model.yaml"
)
VJEPA_MODEL_CONFIG = BASE_MODEL_CONFIG.with_name(
    "dual_explicit_action_dit_vjepa.yaml"
)


def _write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_jsonl_row(path: Path, payload: object) -> Path:
    return _write_bytes(
        path,
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
    )


def _reidentify(payload: dict[str, object]) -> dict[str, object]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("identity_sha256", None)
    return study._identity_payload(unsigned)


class SplitPopulationBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self._serial = 0

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _manifest(
        self,
        split: str,
        count: int,
        *,
        duplicate_episode: bool = False,
    ) -> Path:
        self._serial += 1
        fixture = self.root / f"fixture-{self._serial}-{split}"
        episodes = fixture / "episodes"
        episodes.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        first_episode: Path | None = None
        for index in range(count):
            episode = episodes / f"episode-{index:04d}"
            if duplicate_episode and index == count - 1:
                if first_episode is None:
                    raise AssertionError("duplicate fixture requires two rows")
                episode = first_episode
            else:
                episode.mkdir()
                if first_episode is None:
                    first_episode = episode
            rows.append(
                {
                    "clip_id": f"{split}-clip-{index:04d}",
                    "split": split,
                    "episode_dir": str(episode),
                    "start": index,
                    "auxiliary_index": index,
                }
            )
        path = fixture / f"{split}.jsonl"
        _write_bytes(
            path,
            b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                for row in rows
            ),
        )
        return path

    def test_exact_split_populations_512_64_128_pass(self) -> None:
        for split, count in study.EXPECTED_SPLIT_COUNTS.items():
            with self.subTest(split=split, count=count):
                record = study._manifest_record(
                    self._manifest(split, count),
                    expected_split=split,
                )
                self.assertEqual(record["entries"], count)
                self.assertEqual(record["unique_episodes"], count)

    def test_each_one_clip_undercount_511_63_127_fails(self) -> None:
        for split, expected in study.EXPECTED_SPLIT_COUNTS.items():
            observed = expected - 1
            with self.subTest(split=split, observed=observed):
                with self.assertRaisesRegex(
                    study.ContractError,
                    rf"exactly {expected} clips, found {observed}",
                ):
                    study._manifest_record(
                        self._manifest(split, observed),
                        expected_split=split,
                    )

    def test_duplicate_episode_is_rejected_even_with_exact_clip_count(self) -> None:
        path = self._manifest("train", 512, duplicate_episode=True)
        with self.assertRaisesRegex(
            study.ContractError, "one unique episode per clip"
        ):
            study._manifest_record(path, expected_split="train")

    def test_lexical_parent_alias_is_rejected(self) -> None:
        parent = self.root / "lexical"
        (parent / "left").mkdir(parents=True)
        episode = parent / "episode"
        episode.mkdir()
        aliased = f"{parent}/left/../episode"
        path = parent / "manifest.jsonl"
        _write_jsonl_row(
            path,
            {
                "clip_id": "train-0",
                "split": "train",
                "episode_dir": aliased,
                "start": 0,
                "auxiliary_index": 0,
            },
        )
        with self.assertRaisesRegex(study.ContractError, r"must not contain '\.\.'"):
            study._manifest_record(path, expected_split="train")

    def test_symlink_ancestor_alias_is_rejected(self) -> None:
        real_parent = self.root / "real-parent"
        episode = real_parent / "episode"
        episode.mkdir(parents=True)
        alias_parent = self.root / "alias-parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        path = self.root / "symlink-manifest.jsonl"
        _write_jsonl_row(
            path,
            {
                "clip_id": "train-0",
                "split": "train",
                "episode_dir": str(alias_parent / "episode"),
                "start": 0,
                "auxiliary_index": 0,
            },
        )
        with self.assertRaisesRegex(
            study.ContractError, "symlink or other physical alias"
        ):
            study._manifest_record(path, expected_split="train")

    def test_cross_split_canonical_overlap_and_alias_are_rejected(self) -> None:
        real_parent = self.root / "cross-split-real"
        episode = real_parent / "episode"
        episode.mkdir(parents=True)
        alias_parent = self.root / "cross-split-alias"
        alias_parent.symlink_to(real_parent, target_is_directory=True)

        train_path = self.root / "cross-train.jsonl"
        validation_path = self.root / "cross-validation.jsonl"
        train_row = {
            "clip_id": "train-0",
            "split": "train",
            "episode_dir": str(episode),
            "start": 0,
            "auxiliary_index": 0,
        }
        validation_row = {
            "clip_id": "validation-0",
            "split": "val",
            "episode_dir": str(episode),
            "start": 0,
            "auxiliary_index": 0,
        }
        _write_jsonl_row(train_path, train_row)
        _write_jsonl_row(validation_path, validation_row)
        records = {
            "train": {"clip_manifest": {"path": str(train_path)}},
            "validation": {
                "clip_manifest": {"path": str(validation_path)}
            },
        }
        with self.assertRaisesRegex(study.ContractError, "split overlap"):
            study._assert_disjoint_splits(records)

        validation_row["episode_dir"] = str(alias_parent / "episode")
        _write_jsonl_row(validation_path, validation_row)
        with self.assertRaisesRegex(
            study.ContractError, "symlink or other physical alias"
        ):
            study._assert_disjoint_splits(records)


class PhaseReportSemanticBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.commit = "b" * 40
        self.runtime_python = Path(sys.executable).resolve(strict=True)

        self.warmstart = study._file_record(
            _write_bytes(self.root / "warmstart.pt", b"warm-start")
        )
        self.checkpoint = study._file_record(
            _write_bytes(self.root / "vjepa.pt", b"checkpoint")
        )
        self.pca = study._file_record(
            _write_bytes(self.root / "pca.npz", b"pca")
        )
        self.train_manifest = study._file_record(
            _write_bytes(self.root / "train.jsonl", b'{"clip_id":"train"}\n')
        )
        self.train_metadata = study._file_record(
            _write_bytes(self.root / "train-metadata.json", b"{}")
        )
        self.complete = study._file_record(
            _write_bytes(self.root / "complete.json", b"{}")
        )
        self.cache_id = "1" * 64
        self.train_split = {
            "clip_manifest": self.train_manifest,
            "cache": {
                "metadata": self.train_metadata,
                "cache_id": self.cache_id,
                "target": {"sha256": "2" * 64},
                "rgb": {"sha256": "3" * 64},
                "actions": {"sha256": "4" * 64},
            },
        }
        self.report_path = self.root / "phase-report.json"
        self.report = self._valid_report()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _sampler_counters(deployment_mode: int) -> dict[str, object]:
        return {
            "wan_calls_by_source_nfe": {"autonomous:nfe_1": 1},
            "online_teacher_calls": 0,
            "auxiliary_clean_available": 0,
            "deployment_mode": deployment_mode,
            "artifacts_collected": 1,
            "wan_calls_total": 1,
        }

    def _rank_report(self, rank: int, gpu: str) -> dict[str, object]:
        gradient_update = {
            "calls": 1,
            "finite": True,
            "nonzero": True,
        }
        gradients = {
            group: {"updates": [dict(gradient_update) for _ in range(4)]}
            for group in study.EXPECTED_PHASE_GRADIENT_GROUPS
        }
        digest = f"{rank + 1:x}" * 64
        artifact_hashes = {
            "video_initial_state": digest,
            "tf_initial_state": digest,
            "tf_initial_noise": digest,
            "reference_latents": digest,
        }
        return study._identity_payload(
            {
                "rank": rank,
                "local_rank": rank,
                "gpu": gpu,
                "batch_shapes": copy.deepcopy(
                    study.EXPECTED_PHASE_BATCH_SHAPES
                ),
                "shape_audit_clip_index": rank,
                "morphology_index": 9,
                "zero_gate_equivalence": {
                    "state_gate": 0.0,
                    "clock_gate": 0.0,
                    "bitwise_equal": True,
                    "max_absolute_difference": 0.0,
                },
                "warmstart_load": {
                    "reset_prefixes": list(
                        study.EXPECTED_PHASE_RESET_PREFIXES
                    ),
                    "expected_missing_keys": list(
                        study.EXPECTED_PHASE_WARMSTART_MISSING_KEYS
                    ),
                    "non_prefix_missing_keys": list(
                        study.EXPECTED_PHASE_WARMSTART_MISSING_KEYS
                    ),
                    "loaded_tensor_count": 1,
                    "loaded_parameter_bytes": 4,
                },
                "optimizer": {
                    "completed_updates": 4,
                    "optimizer_step_values": [1, 2, 3, 4],
                    "gradient_reachability": gradients,
                },
                "future_free_nfe1": {
                    "history_shape": [1, 5, 3, 180, 960],
                    "actions_shape": [1, 13, 5, 157],
                    "prediction_shape": [1, 3, 8, 180, 960],
                    "auxiliary_target_argument": None,
                    "teacher_constructed": False,
                    "sampler_counters": self._sampler_counters(1),
                    "ordinary_full_clip_audit": {
                        "condition_source": "autonomous",
                        "nfe": 1,
                        "auxiliary_target_argument": None,
                        "generated_future_bitwise_equal": True,
                        "generated_future_max_absolute_difference": 0.0,
                        "same_initial_noise_and_reference": True,
                        "ground_truth_used_as_condition": False,
                        "sampler_counters": self._sampler_counters(0),
                        "artifact_sha256": artifact_hashes,
                    },
                },
                "model_sync_sha256": {
                    "forward_model.tf_token_adapter.gate": "a" * 64,
                    "forward_model.tf_clock_embedding.gate": "b" * 64,
                    "forward_model.tf_velocity_head.linear.weight": "c" * 64,
                    "forward_model.blocks.0.lora_A.weight": "d" * 64,
                },
            }
        )

    def _valid_report(self) -> dict[str, object]:
        gpus = [f"NVIDIA B200 rank {rank}" for rank in range(8)]
        ranks = [
            self._rank_report(rank, gpu) for rank, gpu in enumerate(gpus)
        ]
        return study._identity_payload(
            {
                "artifact_type": "vjepa2-j0-phase-gate",
                "format_version": 1,
                "passed": True,
                "sigma_convention": study.SIGMA_CONVENTION,
                "teacher_role": "offline target extractor only",
                "teacher_calls_training": 0,
                "teacher_calls_inference": 0,
                "world_size": 8,
                "topology": {
                    "nodes": 1,
                    "gpus": gpus,
                    "backend": "nccl",
                },
                "j0_contract": copy.deepcopy(
                    study.EXPECTED_PHASE_J0_CONTRACT
                ),
                "resolved_config_sha256": "e" * 64,
                "provenance": {
                    "repository": {
                        "path": str(self.repo),
                        "commit": self.commit,
                        "clean": True,
                    },
                    "warmstart": copy.deepcopy(self.warmstart),
                    "cache": {
                        "complete": copy.deepcopy(self.complete),
                        "train_manifest": copy.deepcopy(self.train_manifest),
                        "train_metadata": copy.deepcopy(self.train_metadata),
                        "pca": copy.deepcopy(self.pca),
                        "cache_id": self.cache_id,
                        "target_sha256": "2" * 64,
                        "rgb_sha256": "3" * 64,
                        "actions_sha256": "4" * 64,
                        "checkpoint_sha256": self.checkpoint["sha256"],
                        "source_commit": study.VJEPA_SOURCE_COMMIT,
                    },
                    "runtime_python": study._file_record(
                        self.runtime_python
                    ),
                },
                "full_cache_validation": {
                    "validated": True,
                    "split": "train",
                    "clip_count": 512,
                    "cache_id": self.cache_id,
                    "target_shape": [512, 64, 4, 24, 120],
                    "rgb_shape": [512, 13, 3, 180, 960],
                    "actions_shape": [512, 13, 5, 23],
                },
                "warmstart_policy": {
                    "mode": "strict allowlisted reset",
                    "reset_prefixes": list(
                        study.EXPECTED_PHASE_RESET_PREFIXES
                    ),
                    "expected_missing_keys": list(
                        study.EXPECTED_PHASE_WARMSTART_MISSING_KEYS
                    ),
                },
                "training": {
                    "ddp": True,
                    "optimizer_updates": 4,
                    "effective_global_batch_size": 8,
                    "shape_audit_clip_indices": list(range(8)),
                    "shape_audit_clip_indices_unique": True,
                    "morphology_indices": [9] * 8,
                    "morphology_contract": "ABC integer index exactly 9",
                    "wandb_enabled": False,
                    "checkpoint_writes": 0,
                },
                "inference": {
                    "source": "autonomous",
                    "nfe": 1,
                    "observable_history_frames": 5,
                    "future_rgb_supplied": False,
                    "clean_auxiliary_supplied": False,
                },
                "input_invariance": {
                    "repository_clean": True,
                    "warmstart_unchanged": True,
                    "cache_records_unchanged": True,
                },
                "rank_reports": ranks,
            }
        )

    def _validate(self, report: dict[str, object]) -> None:
        _write_json(self.report_path, report)
        study._validate_phase_gate_report(
            self.report_path,
            repo=self.repo,
            expected_commit=self.commit,
            warmstart=self.warmstart,
            vjepa_checkpoint=self.checkpoint,
            pca_stats=self.pca,
            train_split=self.train_split,
            runtime_python_path=self.runtime_python,
        )

    def test_valid_synthetic_phase_report_passes(self) -> None:
        self._validate(self.report)

    def test_invalid_top_or_rank_identity_is_rejected(self) -> None:
        invalid_top = copy.deepcopy(self.report)
        invalid_top["identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(study.ContractError, "identity is invalid"):
            self._validate(invalid_top)

        invalid_rank = copy.deepcopy(self.report)
        invalid_rank["rank_reports"][0]["identity_sha256"] = "0" * 64
        invalid_rank = _reidentify(invalid_rank)
        with self.assertRaisesRegex(study.ContractError, "rank 0 identity"):
            self._validate(invalid_rank)

    def test_semantic_mutations_fail_after_recomputing_valid_identities(self) -> None:
        def mutate_rank_teacher(report: dict[str, object]) -> None:
            rank = report["rank_reports"][0]
            rank["future_free_nfe1"]["teacher_constructed"] = True
            report["rank_reports"][0] = _reidentify(rank)

        def mutate_rank_optimizer(report: dict[str, object]) -> None:
            rank = report["rank_reports"][0]
            rank["optimizer"]["optimizer_step_values"] = [1, 2, 3, 3]
            report["rank_reports"][0] = _reidentify(rank)

        def mutate_rank_gradient(report: dict[str, object]) -> None:
            rank = report["rank_reports"][0]
            rank["optimizer"]["gradient_reachability"][
                "auxiliary_state_gate"
            ]["updates"][0]["finite"] = False
            report["rank_reports"][0] = _reidentify(rank)

        def mutate_rank_call_key(report: dict[str, object]) -> None:
            rank = report["rank_reports"][0]
            rank["future_free_nfe1"]["sampler_counters"][
                "wan_calls_by_source_nfe"
            ] = {"oracle_matched:nfe_1": 1}
            report["rank_reports"][0] = _reidentify(rank)

        def mutate_rank_scoring_ground_truth(
            report: dict[str, object],
        ) -> None:
            rank = report["rank_reports"][0]
            rank["future_free_nfe1"]["ordinary_full_clip_audit"][
                "ground_truth_used_as_condition"
            ] = True
            report["rank_reports"][0] = _reidentify(rank)

        def mutate_rank_expected_missing(report: dict[str, object]) -> None:
            rank = report["rank_reports"][0]
            rank["warmstart_load"]["expected_missing_keys"] = [
                "action_encoder"
            ]
            report["rank_reports"][0] = _reidentify(rank)

        def mutate_rank_observed_missing(report: dict[str, object]) -> None:
            rank = report["rank_reports"][0]
            rank["warmstart_load"]["non_prefix_missing_keys"] = list(
                study.EXPECTED_PHASE_WARMSTART_MISSING_KEYS[:-1]
            )
            report["rank_reports"][0] = _reidentify(rank)

        mutations = {
            "artifact_type": lambda report: report.__setitem__(
                "artifact_type", "wrong"
            ),
            "format_version": lambda report: report.__setitem__(
                "format_version", 2
            ),
            "missing_world_size": lambda report: report.pop("world_size"),
            "passed_false": lambda report: report.__setitem__(
                "passed", False
            ),
            "sigma_convention": lambda report: report.__setitem__(
                "sigma_convention", "sigma=0 noise"
            ),
            "teacher_role": lambda report: report.__setitem__(
                "teacher_role", "online encoder"
            ),
            "training_teacher_call": lambda report: report.__setitem__(
                "teacher_calls_training", 1
            ),
            "teacher_call": lambda report: report.__setitem__(
                "teacher_calls_inference", 1
            ),
            "two_nodes": lambda report: report["topology"].__setitem__(
                "nodes", 2
            ),
            "wrong_backend": lambda report: report["topology"].__setitem__(
                "backend", "gloo"
            ),
            "seven_gpu_world": lambda report: report["topology"].__setitem__(
                "gpus", report["topology"]["gpus"][:7]
            ),
            "non_b200": lambda report: report["topology"]["gpus"].__setitem__(
                0, "NVIDIA H100"
            ),
            "j0_state_condition_off": lambda report: report[
                "j0_contract"
            ].__setitem__("condition_on_tf", False),
            "j0_video_only": lambda report: report[
                "j0_contract"
            ].__setitem__("video_only_control", True),
            "j0_parameter_control": lambda report: report[
                "j0_contract"
            ].__setitem__("parameter_matched_control", True),
            "non_ddp": lambda report: report["training"].__setitem__(
                "ddp", False
            ),
            "three_updates": lambda report: report["training"].__setitem__(
                "optimizer_updates", 3
            ),
            "global_batch": lambda report: report["training"].__setitem__(
                "effective_global_batch_size", 7
            ),
            "duplicate_clip_audit": lambda report: report[
                "training"
            ].__setitem__(
                "shape_audit_clip_indices", [0, 0, 2, 3, 4, 5, 6, 7]
            ),
            "morphology": lambda report: report["training"].__setitem__(
                "morphology_indices", [8] + [9] * 7
            ),
            "wandb_enabled": lambda report: report["training"].__setitem__(
                "wandb_enabled", True
            ),
            "checkpoint_write": lambda report: report["training"].__setitem__(
                "checkpoint_writes", 1
            ),
            "oracle_inference": lambda report: report[
                "inference"
            ].__setitem__("source", "oracle_matched"),
            "nfe_two": lambda report: report["inference"].__setitem__("nfe", 2),
            "future_rgb": lambda report: report["inference"].__setitem__(
                "future_rgb_supplied", True
            ),
            "clean_auxiliary": lambda report: report[
                "inference"
            ].__setitem__("clean_auxiliary_supplied", True),
            "repository_path": lambda report: report["provenance"][
                "repository"
            ].__setitem__("path", str(self.root)),
            "repository_commit": lambda report: report["provenance"][
                "repository"
            ].__setitem__("commit", "c" * 40),
            "repository_dirty": lambda report: report["provenance"][
                "repository"
            ].__setitem__("clean", False),
            "warmstart_sha": lambda report: report["provenance"][
                "warmstart"
            ].__setitem__("sha256", "f" * 64),
            "manifest_sha": lambda report: report["provenance"][
                "cache"
            ]["train_manifest"].__setitem__("sha256", "f" * 64),
            "metadata_sha": lambda report: report["provenance"][
                "cache"
            ]["train_metadata"].__setitem__("sha256", "f" * 64),
            "pca_sha": lambda report: report["provenance"]["cache"][
                "pca"
            ].__setitem__("sha256", "f" * 64),
            "cache_id": lambda report: report["provenance"]["cache"].__setitem__(
                "cache_id", "f" * 64
            ),
            "target_sha": lambda report: report["provenance"][
                "cache"
            ].__setitem__("target_sha256", "f" * 64),
            "checkpoint_sha": lambda report: report["provenance"][
                "cache"
            ].__setitem__("checkpoint_sha256", "f" * 64),
            "source_commit": lambda report: report["provenance"][
                "cache"
            ].__setitem__("source_commit", "f" * 40),
            "runtime_python": lambda report: report["provenance"][
                "runtime_python"
            ].__setitem__("path", str(self.root / "not-python")),
            "cache_not_invariant": lambda report: report[
                "input_invariance"
            ].__setitem__("cache_records_unchanged", False),
            "repository_not_invariant": lambda report: report[
                "input_invariance"
            ].__setitem__("repository_clean", False),
            "warmstart_not_invariant": lambda report: report[
                "input_invariance"
            ].__setitem__("warmstart_unchanged", False),
            "warmstart_policy": lambda report: report[
                "warmstart_policy"
            ].__setitem__("mode", "permissive"),
            "warmstart_policy_expected_missing": lambda report: report[
                "warmstart_policy"
            ].__setitem__("expected_missing_keys", ["action_encoder"]),
            "full_cache_undercount": lambda report: report[
                "full_cache_validation"
            ].__setitem__("clip_count", 511),
            "full_cache_shape": lambda report: report[
                "full_cache_validation"
            ].__setitem__("actions_shape", [512, 13, 5, 22]),
            "rank_constructs_teacher": mutate_rank_teacher,
            "rank_optimizer": mutate_rank_optimizer,
            "rank_gradient": mutate_rank_gradient,
            "rank_call_key": mutate_rank_call_key,
            "rank_scoring_ground_truth": mutate_rank_scoring_ground_truth,
            "rank_expected_missing": mutate_rank_expected_missing,
            "rank_observed_missing": mutate_rank_observed_missing,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.report)
                mutate(candidate)
                candidate = _reidentify(candidate)
                self.assertTrue(
                    study._identity_is_valid(candidate),
                    "fixture must carry a freshly valid top-level identity",
                )
                for rank in candidate["rank_reports"]:
                    self.assertTrue(study._identity_is_valid(rank))
                with self.assertRaises(study.ContractError):
                    self._validate(candidate)

    def test_phase_report_byte_mutation_is_caught_at_stage_revalidation(self) -> None:
        _write_json(self.report_path, self.report)
        report_record = study._file_record(self.report_path)
        stable = study._file_record(
            _write_bytes(self.root / "stable-input", b"stable")
        )
        cache = {
            "metadata": stable,
            "target": stable,
            "rgb": stable,
            "actions": stable,
        }
        manifest = {
            "inputs": {
                "runtime": {
                    "cache_validator": stable,
                    "python": str(self.runtime_python),
                },
                "configs": {"baseline": stable, "dual": stable},
                "warmstart": stable,
                "vjepa": {
                    "checkpoint": stable,
                    "pca_stats": stable,
                    "pca_companion_metadata": stable,
                    "source": {"path": str(self.root)},
                },
                "phase_gate": {
                    "report": report_record,
                    "runtime_python": stable,
                },
                "cache_build": {
                    "complete": stable,
                    "request": stable,
                    "manifest_metadata": stable,
                    "episode_manifest": stable,
                },
                "splits": {
                    "train": {"clip_manifest": stable, "cache": cache},
                    "validation": {
                        "clip_manifest": stable,
                        "cache": cache,
                    },
                    "test": {"clip_manifest": stable, "cache": cache},
                },
                "repository": {
                    "root": str(self.repo),
                    "git_commit": self.commit,
                },
            }
        }
        with self.report_path.open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(
            study.ContractError, "phase-gate report byte count changed"
        ):
            study._assert_study_inputs_unchanged(manifest)

    def test_python_symlink_is_accepted_and_canonicalized(self) -> None:
        link = self.root / "python-symlink"
        link.symlink_to(self.runtime_python)
        self.assertTrue(link.is_symlink())
        self.assertEqual(
            study._python_executable(link),
            self.runtime_python,
        )


class CacheBuildBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.runtime_python = Path(sys.executable).resolve(strict=True)
        self.expected_commit = "b" * 40
        self.build_commit = "a" * 40

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _fixture(self, name: str) -> dict[str, object]:
        root = self.root / name
        build = root / "build"
        build.mkdir(parents=True)
        repo = root / "repo"
        (repo / ".git").mkdir(parents=True)
        source = root / "vjepa-source"
        source.mkdir()
        checkpoint = study._file_record(
            _write_bytes(root / "vjepa.pt", b"checkpoint")
        )
        pca = study._file_record(_write_bytes(build / "pca.npz", b"pca"))
        episode_manifest = study._file_record(
            _write_bytes(root / "episodes.jsonl", b'{"episode":"all"}\n')
        )

        splits: dict[str, dict[str, object]] = {}
        complete_splits: dict[str, dict[str, object]] = {}
        metadata_splits: dict[str, dict[str, object]] = {}
        study_keys = {"train": "train", "val": "validation", "test": "test"}
        for ordinal, (split_name, study_key) in enumerate(
            study_keys.items(), 1
        ):
            count = study.EXPECTED_SPLIT_COUNTS[split_name]
            manifest_path = _write_bytes(
                build / f"{split_name}.jsonl",
                f'{{"split":"{split_name}"}}\n'.encode("utf-8"),
            )
            manifest_record = study._file_record(manifest_path)
            metadata_path = _write_bytes(
                build / f"{split_name}-metadata.json",
                f'{{"split":"{split_name}"}}\n'.encode("utf-8"),
            )
            metadata_record = study._file_record(metadata_path)
            cache_id = f"{ordinal:x}" * 64
            target_sha = f"{ordinal + 3:x}" * 64
            rgb_sha = f"{ordinal + 6:x}" * 64
            actions_sha = f"{ordinal + 9:x}" * 64
            cache_record = {
                "metadata": metadata_record,
                "cache_id": cache_id,
                "target": {"sha256": target_sha},
                "rgb": {"sha256": rgb_sha},
                "actions": {"sha256": actions_sha},
            }
            splits[study_key] = {
                "clip_manifest": manifest_record,
                "cache": cache_record,
            }
            metadata_splits[split_name] = {
                "file": manifest_path.name,
                "sha256": manifest_record["sha256"],
                "clip_count": count,
                "episode_count": count,
            }
            complete_splits[split_name] = {
                "metadata": metadata_record["path"],
                "metadata_sha256": metadata_record["sha256"],
                "cache_id": cache_id,
                "clip_count": count,
                "target_sha256": target_sha,
                "rgb_sha256": rgb_sha,
                "actions_sha256": actions_sha,
            }

        manifest_metadata = {
            "format_version": 1,
            "schema": "abc-vjepa2.1-fixed-clips-v1",
            "seed": study.EXPECTED_CACHE_BUILD_SEED,
            "sample_size": 13,
            "chunk_size": 5,
            "action_span": 65,
            "frame_offsets": list(range(0, 61, 5)),
            "clips_per_episode": study.EXPECTED_CLIPS_PER_EPISODE,
            "episode_count_consumed": sum(
                study.EXPECTED_SPLIT_COUNTS.values()
            ),
            "episode_manifest": episode_manifest["path"],
            "episode_manifest_sha256": episode_manifest["sha256"],
            "splits": metadata_splits,
        }
        manifest_metadata_path = _write_json(
            build / "manifest_metadata.json", manifest_metadata
        )
        manifest_metadata_record = study._file_record(manifest_metadata_path)
        request = {
            "artifact_type": "vjepa2.1-immutable-cache-build-request",
            "format_version": 1,
            "build_id": f"build-{name}",
            "git_commit": self.build_commit,
            "repo_root": str(repo),
            "build_root": str(build),
            "episode_manifest": episode_manifest["path"],
            "episode_manifest_sha256": episode_manifest["sha256"],
            "extractor_python": str(self.runtime_python),
            "vjepa_source": str(source),
            "vjepa_source_commit": study.VJEPA_SOURCE_COMMIT,
            "vjepa_checkpoint": checkpoint["path"],
            "vjepa_checkpoint_sha256": checkpoint["sha256"],
            "train_clips": 512,
            "val_clips": 64,
            "test_clips": 128,
            "clips_per_episode": 1,
            "seed": 20_260_729,
            "pca_max_clips": 256,
            "pca_max_tokens": 250_000,
        }
        request_path = _write_json(build / "build_request.json", request)
        request_record = study._file_record(request_path)
        complete = {
            "artifact_type": "vjepa2.1-immutable-cache-build",
            "format_version": 1,
            "build_request_sha256": request_record["sha256"],
            "build_id": request["build_id"],
            "git_commit": self.build_commit,
            "manifest_metadata": manifest_metadata_record["path"],
            "manifest_metadata_sha256": manifest_metadata_record["sha256"],
            "pca": pca["path"],
            "pca_sha256": pca["sha256"],
            "splits": complete_splits,
        }
        complete_path = _write_json(build / "complete.json", complete)
        complete_record = study._file_record(complete_path)
        return {
            "repo": repo,
            "source": source,
            "checkpoint": checkpoint,
            "pca": pca,
            "splits": splits,
            "request": request,
            "request_path": request_path,
            "manifest_metadata": manifest_metadata,
            "manifest_metadata_path": manifest_metadata_path,
            "complete": complete,
            "complete_path": complete_path,
            "phase_gate_record": {"cache_complete": complete_record},
            "phase_report": {
                "provenance": {
                    "cache": {"complete": copy.deepcopy(complete_record)}
                }
            },
        }

    def _validate(self, fixture: dict[str, object]) -> dict[str, object]:
        with (
            mock.patch.object(study, "_assert_clean_commit") as clean,
            mock.patch.object(study, "_assert_git_ancestor") as ancestor,
        ):
            result = study._validate_cache_build(
                repo=fixture["repo"],
                expected_commit=self.expected_commit,
                phase_report=fixture["phase_report"],
                phase_gate_record=fixture["phase_gate_record"],
                splits=fixture["splits"],
                pca_stats=fixture["pca"],
                vjepa_source=fixture["source"],
                vjepa_checkpoint=fixture["checkpoint"],
                extractor_python=self.runtime_python,
            )
        clean.assert_called_once_with(fixture["repo"], self.build_commit)
        ancestor.assert_called_once_with(
            fixture["repo"], self.build_commit, self.expected_commit
        )
        return result

    @staticmethod
    def _refresh_complete_binding(fixture: dict[str, object]) -> None:
        _write_json(fixture["complete_path"], fixture["complete"])
        record = study._file_record(fixture["complete_path"])
        fixture["phase_gate_record"]["cache_complete"] = record
        fixture["phase_report"]["provenance"]["cache"]["complete"] = copy.deepcopy(
            record
        )

    @classmethod
    def _refresh_request_binding(cls, fixture: dict[str, object]) -> None:
        _write_json(fixture["request_path"], fixture["request"])
        fixture["complete"]["build_request_sha256"] = study._sha256(
            fixture["request_path"]
        )
        cls._refresh_complete_binding(fixture)

    @classmethod
    def _refresh_manifest_metadata_binding(
        cls, fixture: dict[str, object]
    ) -> None:
        _write_json(
            fixture["manifest_metadata_path"],
            fixture["manifest_metadata"],
        )
        fixture["complete"]["manifest_metadata_sha256"] = study._sha256(
            fixture["manifest_metadata_path"]
        )
        cls._refresh_complete_binding(fixture)

    def test_valid_complete_binds_all_train_val_test_artifacts(self) -> None:
        fixture = self._fixture("valid")
        result = self._validate(fixture)
        self.assertEqual(
            result["split_counts"], {"train": 512, "val": 64, "test": 128}
        )
        self.assertEqual(result["seed"], 20_260_729)
        self.assertEqual(result["clips_per_episode"], 1)

    def test_complete_semantics_bind_validation_and_test_not_only_train(
        self,
    ) -> None:
        for split_name in ("val", "test"):
            with self.subTest(split=split_name):
                fixture = self._fixture(f"mutated-{split_name}")
                fixture["complete"]["splits"][split_name][
                    "target_sha256"
                ] = "f" * 64
                self._refresh_complete_binding(fixture)
                with self.assertRaisesRegex(
                    study.ContractError,
                    rf"{split_name} cache binding differs",
                ):
                    self._validate(fixture)

    def test_complete_requires_exactly_all_three_splits(self) -> None:
        fixture = self._fixture("missing-test")
        fixture["complete"]["splits"].pop("test")
        self._refresh_complete_binding(fixture)
        with self.assertRaisesRegex(
            study.ContractError, "exactly train/val/test"
        ):
            self._validate(fixture)

    def test_build_request_seed_population_and_clip_policy_are_exact(self) -> None:
        mutations = {
            "seed": ("seed", 20_260_728),
            "clips_per_episode": ("clips_per_episode", 2),
            "train_clips": ("train_clips", 511),
            "val_clips": ("val_clips", 63),
            "test_clips": ("test_clips", 127),
            "pca_max_clips": ("pca_max_clips", 255),
            "pca_max_tokens": ("pca_max_tokens", 249_999),
            "source_commit": ("vjepa_source_commit", "f" * 40),
            "checkpoint_sha": ("vjepa_checkpoint_sha256", "f" * 64),
        }
        for name, (field, value) in mutations.items():
            with self.subTest(name=name):
                fixture = self._fixture(f"request-{name}")
                fixture["request"][field] = value
                self._refresh_request_binding(fixture)
                with self.assertRaisesRegex(
                    study.ContractError, "cache-build request contract differs"
                ):
                    self._validate(fixture)

    def test_manifest_metadata_seed_clip_and_episode_policy_are_exact(
        self,
    ) -> None:
        mutations = {
            "seed": lambda metadata: metadata.__setitem__(
                "seed", 20_260_728
            ),
            "clips_per_episode": lambda metadata: metadata.__setitem__(
                "clips_per_episode", 2
            ),
            "episode_count_consumed": lambda metadata: metadata.__setitem__(
                "episode_count_consumed", 703
            ),
            "validation_clip_count": lambda metadata: metadata["splits"][
                "val"
            ].__setitem__("clip_count", 63),
            "test_episode_count": lambda metadata: metadata["splits"][
                "test"
            ].__setitem__("episode_count", 127),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                fixture = self._fixture(f"manifest-{name}")
                mutate(fixture["manifest_metadata"])
                self._refresh_manifest_metadata_binding(fixture)
                with self.assertRaises(study.ContractError):
                    self._validate(fixture)

    def test_build_repository_and_ancestor_fail_closed(self) -> None:
        fixture = self._fixture("bad-build-repo")
        with (
            mock.patch.object(
                study,
                "_assert_clean_commit",
                side_effect=study.ContractError("dirty build repository"),
            ),
            mock.patch.object(study, "_assert_git_ancestor"),
        ):
            with self.assertRaisesRegex(
                study.ContractError, "dirty build repository"
            ):
                study._validate_cache_build(
                    repo=fixture["repo"],
                    expected_commit=self.expected_commit,
                    phase_report=fixture["phase_report"],
                    phase_gate_record=fixture["phase_gate_record"],
                    splits=fixture["splits"],
                    pca_stats=fixture["pca"],
                    vjepa_source=fixture["source"],
                    vjepa_checkpoint=fixture["checkpoint"],
                    extractor_python=self.runtime_python,
                )

        fixture = self._fixture("non-ancestor")
        with (
            mock.patch.object(study, "_assert_clean_commit"),
            mock.patch.object(
                study,
                "_assert_git_ancestor",
                side_effect=study.ContractError("not an ancestor"),
            ),
        ):
            with self.assertRaisesRegex(study.ContractError, "not an ancestor"):
                study._validate_cache_build(
                    repo=fixture["repo"],
                    expected_commit=self.expected_commit,
                    phase_report=fixture["phase_report"],
                    phase_gate_record=fixture["phase_gate_record"],
                    splits=fixture["splits"],
                    pca_stats=fixture["pca"],
                    vjepa_source=fixture["source"],
                    vjepa_checkpoint=fixture["checkpoint"],
                    extractor_python=self.runtime_python,
                )


class StaticEntrypointContractTest(unittest.TestCase):
    def test_phase_launcher_preserves_venv_symlink_for_execution(self) -> None:
        source = PHASE_SBATCH.read_text(encoding="utf-8")
        self.assertNotIn('PYTHON_BIN="$(readlink -f', source)
        self.assertIn(
            'PYTHON_REAL_BIN="$(readlink -f -- "$PYTHON_BIN")"',
            source,
        )
        self.assertIn(
            'die "failed to resolve Python executable: $PYTHON_BIN"',
            source,
        )
        self.assertIn('export LACWM_PYTHON="$PYTHON_BIN"', source)
        self.assertIn('"$PYTHON_BIN" "$VERIFY_RUNTIME"', source)
        self.assertIn('"$PYTHON_BIN" -m torch.distributed.run', source)
        self.assertIn('"$PYTHON_BIN" - "$REPORT"', source)
        command_start = source.index("COMMAND=(")
        command_end = source.index(")", command_start)
        self.assertNotIn("PYTHON_REAL_BIN", source[command_start:command_end])

    def test_phase_helper_records_resolved_binary_not_venv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / "python"
            real = Path(sys.executable).resolve(strict=True)
            link.symlink_to(real)
            record = phase_gate.runtime_python_record(link)
        self.assertEqual(record["path"], str(real))
        self.assertEqual(record["sha256"], study._sha256(real))
        self.assertEqual(record["bytes"], real.stat().st_size)

    def test_executing_venv_symlink_preserves_pyvenv_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "runtime"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            launcher = environment / "bin" / "python"
            real = launcher.resolve(strict=True)
            self.assertTrue(launcher.is_symlink())
            launcher_prefix = subprocess.check_output(
                [str(launcher), "-c", "import sys; print(sys.prefix)"],
                text=True,
            ).strip()
            resolved_prefix = subprocess.check_output(
                [str(real), "-c", "import sys; print(sys.prefix)"],
                text=True,
            ).strip()
        self.assertEqual(Path(launcher_prefix).resolve(), environment.resolve())
        self.assertNotEqual(
            Path(resolved_prefix).resolve(),
            environment.resolve(),
        )

    def test_phase_report_is_mandatory_in_parser_and_submitter(self) -> None:
        parser = study.build_parser()
        subparser_action = next(
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        )
        for command in ("preflight", "create-study"):
            phase_action = next(
                action
                for action in subparser_action.choices[command]._actions
                if "--phase-gate-report" in action.option_strings
            )
            self.assertTrue(phase_action.required)

        source = SUBMIT.read_text(encoding="utf-8")
        self.assertIn(
            'PHASE_GATE_REPORT="${VJEPA_PHASE_GATE_REPORT:-}"', source
        )
        self.assertIn(
            '--phase-gate-report) [[ $# -ge 2 ]] || '
            'die "--phase-gate-report requires a value"',
            source,
        )
        self.assertIn('"phase-gate report:$PHASE_GATE_REPORT"', source)
        self.assertIn('--phase-gate-report "$PHASE_GATE_REPORT"', source)

    def test_generic_dual_configs_fail_closed(self) -> None:
        for path in (BASE_MODEL_CONFIG, VJEPA_MODEL_CONFIG):
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                dual_block = source.split("\ndual_diffusion:\n", 1)[1]
                first_setting = re.search(
                    r"^  ([a-z_]+):\s*(\S+)", dual_block, re.MULTILINE
                )
                self.assertIsNotNone(first_setting)
                self.assertEqual(
                    first_setting.groups(), ("enabled", "false")
                )

    def test_phase_and_study_explicitly_enable_dual_diffusion(self) -> None:
        phase = PHASE_HELPER.read_text(encoding="utf-8")
        stage = STAGE_SBATCH.read_text(encoding="utf-8")
        literal = '"model.dual_diffusion.enabled=true"'
        self.assertIn(literal, phase)
        self.assertIn(literal, stage)

    def test_every_stage_allocation_revalidates_before_arm_branching(self) -> None:
        source = STAGE_SBATCH.read_text(encoding="utf-8")
        call = '"$PYTHON_BIN" "$HELPER" validate-study-inputs \\\n'
        self.assertEqual(source.count(call), 1)
        self.assertLess(source.index(call), source.index("IFS='|' read -r"))
        self.assertLess(source.index(call), source.index('mkdir "$RUN_DIR"'))

    def test_paired_job_revalidates_before_output_or_benchmark(self) -> None:
        source = PAIRED_SBATCH.read_text(encoding="utf-8")
        validation = source.index(
            '"$PYTHON_BIN" "$HELPER" validate-study-inputs \\\n'
        )
        mkdir = source.index('mkdir "$OUTPUT_DIR"')
        benchmark = source.index('"$PYTHON_BIN" "$BENCHMARK" \\\n')
        self.assertLess(validation, mkdir)
        self.assertLess(validation, benchmark)
        validation_block = source[validation:mkdir]
        self.assertIn('--study-manifest "$STUDY_MANIFEST"', validation_block)
        self.assertIn('--expected-commit "$EXPECTED_COMMIT"', validation_block)
        self.assertIn('--repo-root "$REPO_ROOT"', validation_block)


if __name__ == "__main__":
    unittest.main()
