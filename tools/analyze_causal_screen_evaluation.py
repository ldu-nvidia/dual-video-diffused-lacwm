#!/usr/bin/env python3
"""Analyze only a complete, immutable causal-screen posthoc evaluation.

Unlike the generic dual-NFE analyzer, this entrypoint treats
``evaluation_manifest.json`` and ``evaluation_complete.json`` as the inventory
authority.  It requires the completion identity printed by the evaluator,
revalidates every declared input and output, and only then delegates to the
generic analyzer using the iteration roots declared by the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open

import analyze_dual_nfe_artifacts as generic
import evaluate_causal_screen_snapshots as evaluator


SCHEMA_VERSION = 2
MANIFEST_NAME = "evaluation_manifest.json"
SUBMISSION_NAME = "slurm_submission.json"
COMPLETION_NAME = "evaluation_complete.json"
MANIFEST_KIND = "dual_abc_tf_causal_screen_posthoc_evaluation"
SUBMISSION_KIND = "dual_abc_tf_causal_screen_posthoc_submission"
COMPLETION_KIND = "dual_abc_tf_causal_screen_posthoc_evaluation_complete"
INPUT_KIND = "dual_abc_tf_causal_screen_posthoc_input"
ARTIFACT_KIND = "dual_abc_tf_causal_screen_posthoc_artifact"
ANALYSIS_KIND = "dual_abc_tf_causal_screen_completed_analysis"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATASET_RE = re.compile(r"^[A-Za-z0-9._-]+__batch_([0-9]{3})$")
CANONICAL_ARM_ORDER = tuple(arm["name"] for arm in evaluator.causal.ARMS)


class CompletedEvaluationError(RuntimeError):
    """Raised when a completed evaluation cannot be trusted for analysis."""


def _fail(message: str) -> None:
    raise CompletedEvaluationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _snapshot_load_contract_matches(
    snapshot: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Match the load contract, tolerating only serialized gate roundoff."""
    for key, wanted in expected.items():
        actual = snapshot.get(key)
        if key != "effective_state_gate":
            if actual != wanted:
                return False
            continue
        if (
            isinstance(actual, bool)
            or isinstance(wanted, bool)
            or not isinstance(actual, (int, float))
            or not isinstance(wanted, (int, float))
        ):
            return False
        actual_float = float(actual)
        wanted_float = float(wanted)
        if (
            not math.isfinite(actual_float)
            or not math.isfinite(wanted_float)
            or not math.isclose(
                actual_float,
                wanted_float,
                rel_tol=0.0,
                abs_tol=evaluator.EFFECTIVE_STATE_GATE_ABS_TOL,
            )
        ):
            return False
    return True


def _full_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return evaluator._load_json(path, label)
    except Exception as exc:
        raise CompletedEvaluationError(f"invalid {label}: {exc}") from exc


def _identity_json(path: Path, label: str) -> dict[str, Any]:
    payload = _strict_json(path, label)
    if not evaluator._identity_is_valid(payload):
        _fail(f"{label} identity is invalid: {path}")
    return payload


def _canonical_directory(value: str | Path, label: str) -> Path:
    try:
        return evaluator._canonical_directory(value, label)
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc


def _canonical_file(value: str | Path, label: str) -> Path:
    try:
        return evaluator._canonical_file(value, label)
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc


def _sha256(path: Path) -> str:
    return evaluator._sha256(path)


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _exact_file(
    recorded_path: Any,
    *,
    expected: Path,
    root: Path,
    label: str,
) -> Path:
    if recorded_path != str(expected):
        _fail(f"{label} path differs: {recorded_path!r} != {expected}")
    try:
        generic._ensure_no_symlink_components(expected, root)
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc
    canonical = _canonical_file(expected, label)
    if canonical != expected:
        _fail(f"{label} canonical path differs: {canonical} != {expected}")
    return canonical


def _validate_file_record(
    raw_record: Any,
    *,
    expected: Path,
    root: Path,
    label: str,
) -> Path:
    record = _mapping(raw_record, f"{label} record")
    path = _exact_file(
        record.get("path"),
        expected=expected,
        root=root,
        label=label,
    )
    info = path.stat()
    expected_values = {
        "sha256": _sha256(path),
        "bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }
    problems = [
        f"{key}: {record.get(key)!r} != {wanted!r}"
        for key, wanted in expected_values.items()
        if record.get(key) != wanted
    ]
    if problems:
        _fail(f"{label} record differs: " + "; ".join(problems))
    return path


def _assert_git_commit(
    repository: Path,
    expected_commit: Any,
    label: str,
) -> None:
    if not isinstance(expected_commit, str):
        _fail(f"{label} commit must be a string")
    try:
        evaluator._assert_clean_commit(
            repository,
            expected_commit,
            label=label,
        )
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc


def _validate_source_immutability(
    manifest: Mapping[str, Any],
    evaluation_root: Path,
) -> dict[str, Any]:
    evaluation_repo = _canonical_directory(
        manifest.get("repository_root", ""),
        "evaluation repository",
    )
    _assert_git_commit(
        evaluation_repo,
        manifest.get("evaluation_commit"),
        "evaluation repository",
    )
    try:
        training_repository = evaluator._validate_training_repository(
            _mapping(
                manifest.get("training_repository"),
                "training repository",
            ).get("screen_recorded_path"),
            evaluation_repo_root=evaluation_repo,
            training_commit=manifest.get("training_commit"),
        )
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc
    if training_repository != manifest.get("training_repository"):
        _fail("training repository provenance differs")
    training_repo = Path(training_repository["path"])

    source_screen = _mapping(manifest.get("source_screen"), "source screen")
    screen_root = _canonical_directory(
        source_screen.get("screen_root", ""),
        "source screen root",
    )
    for left, right, labels in (
        (
            evaluation_root,
            screen_root,
            ("evaluation root", "source screen root"),
        ),
        (
            evaluation_root,
            evaluation_repo,
            ("evaluation root", "evaluation repository"),
        ),
        (
            evaluation_root,
            training_repo,
            ("evaluation root", "training repository"),
        ),
    ):
        try:
            evaluator._require_disjoint(left, right, *labels)
        except Exception as exc:
            raise CompletedEvaluationError(str(exc)) from exc

    screen_manifest = _validate_file_record(
        source_screen,
        expected=screen_root / "screen_manifest.json",
        root=screen_root,
        label="source screen manifest",
    )
    source_screen_payload = _identity_json(
        screen_manifest,
        "source screen manifest",
    )
    if (
        source_screen_payload.get("identity_sha256")
        != source_screen.get("identity_sha256")
        or source_screen_payload.get("screen_id")
        != source_screen.get("screen_id")
    ):
        _fail("source screen identity differs from evaluation manifest")

    source_arms = _mapping(
        manifest.get("source_arm_inventory"),
        "source arm inventory",
    )
    if set(source_arms) != set(CANONICAL_ARM_ORDER):
        _fail("source arm inventory is not the canonical five-arm screen")
    for arm_name, raw_arm in source_arms.items():
        if (
            not isinstance(arm_name, str)
            or generic.ARM_NAME_RE.fullmatch(arm_name) is None
        ):
            _fail(f"invalid source arm name: {arm_name!r}")
        arm = _mapping(raw_arm, f"source arm {arm_name}")
        arm_directory = _canonical_directory(
            arm.get("source_directory", ""),
            f"source arm {arm_name} directory",
        )
        if arm_directory.parent != screen_root or arm_directory.name != arm_name:
            _fail(f"source arm {arm_name} is not a direct screen child")
        expected_files = {
            "arm_manifest": "arm_manifest.json",
            "resolved_config": "resolved_config.yaml",
            "training_completion": "training_complete.json",
            "outcome": "outcome.json",
            "snapshot": "snapshot.pt",
        }
        validated_paths = {
            key: _validate_file_record(
                arm.get(key),
                expected=arm_directory / filename,
                root=arm_directory,
                label=f"{arm_name} {key}",
            )
            for key, filename in expected_files.items()
        }
        arm_manifest = _identity_json(
            validated_paths["arm_manifest"],
            f"{arm_name} arm manifest",
        )
        if (
            arm_manifest.get("identity_sha256")
            != _mapping(
                arm.get("arm_manifest"),
                f"{arm_name} arm manifest record",
            ).get("identity_sha256")
            or arm_manifest.get("arm") != arm_name
            or arm_manifest.get("git_commit") != manifest.get("training_commit")
        ):
            _fail(f"{arm_name} source manifest identity differs")
        training_completion = _strict_json(
            validated_paths["training_completion"],
            f"{arm_name} training completion",
        )
        if (
            training_completion.get("status") != "completed"
            or training_completion.get("completed_updates")
            != evaluator.COMPLETED_UPDATES
            or training_completion.get("snapshot")
            != str(validated_paths["snapshot"])
        ):
            _fail(f"{arm_name} training completion contract differs")
        outcome = _strict_json(
            validated_paths["outcome"],
            f"{arm_name} outcome",
        )
        if (
            outcome.get("completed") is not True
            or outcome.get("exit_status") != 0
        ):
            _fail(f"{arm_name} outcome is not successfully completed")

    assets = _mapping(manifest.get("assets"), "evaluation assets")
    wan_directory = _canonical_directory(
        assets.get("wan_directory", ""),
        "Wan asset directory",
    )
    videox_home = _canonical_directory(
        assets.get("videox_home", ""),
        "VideoX-Fun checkout",
    )
    _validate_file_record(
        assets.get("null_prompt"),
        expected=wan_directory / "null_prompt_umt5.pt",
        root=wan_directory,
        label="null-prompt embedding",
    )
    _validate_file_record(
        assets.get("scheduler_config"),
        expected=videox_home / "config" / "wan2.1" / "wan_civitai.yaml",
        root=videox_home,
        label="scheduler configuration",
    )
    try:
        videox_commit = evaluator._git(
            videox_home,
            "rev-parse",
            "HEAD",
        ).stdout.strip()
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc
    if videox_commit != assets.get("videox_commit"):
        _fail("VideoX-Fun commit differs")

    data = _mapping(manifest.get("data"), "evaluation data")
    data_root = _canonical_directory(data.get("root", ""), "data root")
    abc_manifest = _mapping(data.get("abc_manifest"), "ABC manifest record")
    abc_path = _exact_file(
        abc_manifest.get("path"),
        expected=data_root / "abc_pp" / "manifest.txt",
        root=data_root,
        label="ABC manifest",
    )
    if _sha256(abc_path) != abc_manifest.get("sha256"):
        _fail("ABC manifest hash differs")
    return {
        "evaluation_repository": str(evaluation_repo),
        "training_repository": str(training_repo),
        "source_screen_root": str(screen_root),
        "source_arm_count": len(source_arms),
        "data_root": str(data_root),
        "wan_directory": str(wan_directory),
        "videox_home": str(videox_home),
        "assets_verified": True,
        "data_verified": True,
    }


def _selected_arm_contract(
    manifest: Mapping[str, Any],
    evaluation_root: Path,
) -> tuple[list[str], dict[str, Path], int]:
    paired = _mapping(manifest.get("paired_evaluation"), "paired evaluation")
    world_size = paired.get("world_size")
    batches_per_rank = paired.get("batches_per_rank")
    paired_count = paired.get("paired_unit_count")
    iteration = paired.get("evaluation_iteration")
    if world_size != evaluator.WORLD_SIZE:
        _fail(f"evaluation world size must be {evaluator.WORLD_SIZE}")
    if (
        not isinstance(batches_per_rank, int)
        or isinstance(batches_per_rank, bool)
        or not 1 <= batches_per_rank <= evaluator.MAX_BATCHES_PER_RANK
    ):
        _fail("batches_per_rank is outside the evaluator contract")
    if paired_count != evaluator.WORLD_SIZE * batches_per_rank:
        _fail("paired unit count is not world_size * batches_per_rank")
    if iteration != evaluator.EVALUATION_ITERATION:
        _fail("evaluation iteration differs")
    expected_paired = {
        "batch_size": 1,
        "nfe_steps": list(evaluator.NFE_STEPS),
        "source_screen_condition_sources": list(
            evaluator.SOURCE_SCREEN_CONDITION_SOURCES
        ),
        "condition_sources": list(evaluator.CONDITION_SOURCES),
        "runtime_condition_source_override": True,
        "oracle_sources_are_leakage": True,
        "base_evaluation_noise_seed": evaluator.BASE_EVALUATION_NOISE_SEED,
        "sigma_convention": evaluator.SIGMA_CONVENTION,
        "no_cross_example_averaging": True,
        "model_seed_for_batch": (
            "base_evaluation_noise_seed + world_size * batch_index"
        ),
        "effective_generator_seed": (
            "base_evaluation_noise_seed + global_rank + "
            "world_size * batch_index"
        ),
    }
    problems = [
        f"{key}: {paired.get(key)!r} != {wanted!r}"
        for key, wanted in expected_paired.items()
        if paired.get(key) != wanted
    ]
    if problems:
        _fail("paired evaluation contract differs: " + "; ".join(problems))

    arm_order = paired.get("arm_order")
    if (
        not isinstance(arm_order, list)
        or len(arm_order) < 2
        or len(arm_order) != len(set(arm_order))
        or any(
            not isinstance(name, str)
            or generic.ARM_NAME_RE.fullmatch(name) is None
            for name in arm_order
        )
    ):
        _fail("selected arm order is invalid")
    canonical_projection = [
        name for name in CANONICAL_ARM_ORDER if name in set(arm_order)
    ]
    if arm_order != canonical_projection:
        _fail("selected arms are not in canonical causal-screen order")
    selected = _mapping(manifest.get("arms"), "selected arms")
    source = _mapping(
        manifest.get("source_arm_inventory"),
        "source arm inventory",
    )
    if (
        set(selected) != set(arm_order)
        or any(selected[name] != source.get(name) for name in arm_order)
    ):
        _fail("selected arms differ from source inventory or arm order")

    outputs = _mapping(manifest.get("outputs"), "evaluation outputs")
    if (
        outputs.get("analyzer_compatible") is not True
        or outputs.get("paired_key") != ["dataset", "global_rank"]
    ):
        _fail("evaluation output analyzer contract differs")
    recorded_roots = _mapping(outputs.get("arm_roots"), "output arm roots")
    if set(recorded_roots) != set(arm_order):
        _fail("declared output roots differ from selected arms")
    artifact_root = _canonical_directory(
        evaluation_root / "artifacts",
        "evaluation artifact root",
    )
    actual_arm_entries = {
        path.name
        for path in artifact_root.iterdir()
        if path.is_dir() or path.is_symlink()
    }
    if actual_arm_entries != set(arm_order):
        _fail(
            "artifact-root arm directories differ from selected arms: "
            f"{sorted(actual_arm_entries)} != {sorted(arm_order)}"
        )
    arm_roots: dict[str, Path] = {}
    for name in arm_order:
        expected = (
            evaluation_root
            / "artifacts"
            / name
            / f"iter_{evaluator.EVALUATION_ITERATION}"
        )
        if recorded_roots.get(name) != str(expected):
            _fail(f"declared root for {name} differs")
        canonical = _canonical_directory(expected, f"{name} iteration root")
        if canonical != expected:
            _fail(f"{name} iteration root canonicalization differs")
        arm_roots[name] = canonical
    return list(arm_order), arm_roots, batches_per_rank


def _validate_input_sidecar(
    *,
    unit: Mapping[str, Any],
    input_record: Mapping[str, Any],
    evaluation_root: Path,
) -> tuple[Path, Path]:
    dataset = unit["dataset"]
    rank = unit["global_rank"]
    batch_index = unit["batch_index"]
    folder = evaluation_root / "inputs" / dataset
    safetensors_path = _exact_file(
        input_record.get("safetensors_path"),
        expected=folder / f"input_rank_{rank}.safetensors",
        root=evaluation_root,
        label="input safetensors",
    )
    sidecar_path = _exact_file(
        input_record.get("sidecar_path"),
        expected=folder / f"input_rank_{rank}.json",
        root=evaluation_root,
        label="input sidecar",
    )
    if (
        _sha256(safetensors_path) != input_record.get("safetensors_sha256")
        or _sha256(sidecar_path) != input_record.get("sidecar_sha256")
    ):
        _fail(f"input hash differs for rank/batch {(rank, batch_index)}")
    if safetensors_path.stat().st_nlink != 1 or sidecar_path.stat().st_nlink != 1:
        _fail("input artifacts must not be hard-linked aliases")
    sidecar = _identity_json(sidecar_path, "input sidecar")
    expected = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "kind": INPUT_KIND,
        "dataset": dataset,
        "global_rank": rank,
        "batch_index": batch_index,
        "batch_sha256": unit["batch_sha256"],
        "tensor_schema": input_record.get("tensor_schema"),
        "safetensors_sha256": input_record.get("safetensors_sha256"),
        "safetensors_path": str(safetensors_path),
    }
    problems = [
        f"{key}: {sidecar.get(key)!r} != {wanted!r}"
        for key, wanted in expected.items()
        if sidecar.get(key) != wanted
    ]
    if problems:
        _fail("input sidecar differs: " + "; ".join(problems))
    tensor_schema = _mapping(sidecar.get("tensor_schema"), "input tensor schema")
    required_keys = {"rgb", "actions", "mask"}
    if not required_keys.issubset(tensor_schema):
        _fail(
            "input tensor schema omits action-bearing ABC tensors: "
            f"{sorted(required_keys - set(tensor_schema))}"
        )
    try:
        with safe_open(
            str(safetensors_path),
            framework="pt",
            device="cpu",
        ) as handle:
            metadata = dict(handle.metadata() or {})
            keys = set(handle.keys())
    except Exception as exc:
        raise CompletedEvaluationError(
            f"could not inspect input safetensors: {exc}"
        ) from exc
    expected_metadata = {
        "dataset": dataset,
        "global_rank": str(rank),
        "batch_index": str(batch_index),
        "batch_sha256": unit["batch_sha256"],
    }
    if keys != set(tensor_schema) or metadata != expected_metadata:
        _fail("input safetensors metadata or tensor keys differ")
    return safetensors_path, sidecar_path


def _validate_output_sidecar(
    *,
    unit: Mapping[str, Any],
    arm_name: str,
    artifact_record: Mapping[str, Any],
    arm_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    dataset = unit["dataset"]
    rank = unit["global_rank"]
    batch_index = unit["batch_index"]
    folder = arm_root / dataset
    safetensors_path = _exact_file(
        artifact_record.get("safetensors_path"),
        expected=folder / f"latent_trajectory_rank_{rank}.safetensors",
        root=arm_root,
        label=f"{arm_name} output safetensors",
    )
    sidecar_path = _exact_file(
        artifact_record.get("sidecar_path"),
        expected=folder / f"latent_trajectory_rank_{rank}.json",
        root=arm_root,
        label=f"{arm_name} output sidecar",
    )
    if (
        _sha256(safetensors_path)
        != artifact_record.get("safetensors_sha256")
        or _sha256(sidecar_path) != artifact_record.get("sidecar_sha256")
    ):
        _fail(
            f"{arm_name} output hash differs for rank/batch "
            f"{(rank, batch_index)}"
        )
    if safetensors_path.stat().st_nlink != 1 or sidecar_path.stat().st_nlink != 1:
        _fail(f"{arm_name} output artifacts must not be hard-linked aliases")
    sidecar = _identity_json(sidecar_path, f"{arm_name} output sidecar")
    arm = _mapping(
        _mapping(manifest.get("arms"), "selected arms").get(arm_name),
        f"selected arm {arm_name}",
    )
    expected_intervention = {
        "condition_on_tf": arm.get("condition_on_tf"),
        "condition_mode": arm.get("condition_mode"),
        "state_gate_init": arm.get("state_gate_init"),
        "state_gate_trainable": False,
    }
    expected = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "iteration": evaluator.EVALUATION_ITERATION,
        "dataset": dataset,
        "global_rank": rank,
        "batch_index": batch_index,
        "pair_key": {"dataset": dataset, "global_rank": rank},
        "sigma_convention": evaluator.SIGMA_CONVENTION,
        "arm": arm_name,
        "intervention": expected_intervention,
        "evaluation_commit": manifest.get("evaluation_commit"),
        "resolved_config": arm.get("resolved_config"),
        "input_batch_sha256": unit["batch_sha256"],
        "model_evaluation_noise_seed": unit[
            "model_evaluation_noise_seed"
        ],
        "effective_generator_seed": unit["effective_generator_seed"],
        "paired_identity": unit["paired_identity"],
        "per_example_metrics": artifact_record.get("per_example_metrics"),
        "aggregate_sampling_timing": artifact_record.get(
            "aggregate_sampling_timing"
        ),
        "safetensors_sha256": artifact_record.get("safetensors_sha256"),
    }
    problems = [
        f"{key}: sidecar value differs"
        for key, wanted in expected.items()
        if sidecar.get(key) != wanted
    ]
    if problems:
        _fail(f"{arm_name} output sidecar differs: " + "; ".join(problems))
    snapshot = _mapping(sidecar.get("snapshot"), "artifact snapshot")
    source_snapshot = _mapping(arm.get("snapshot"), "source snapshot")
    if any(snapshot.get(key) != value for key, value in source_snapshot.items()):
        _fail(f"{arm_name} artifact snapshot provenance differs")
    expected_snapshot_load = {
        "snapshot_schema_version": 3,
        "next_iteration": evaluator.COMPLETED_UPDATES,
        "world_size": evaluator.WORLD_SIZE,
        "gradient_accumulation_steps": 1,
        "strict_key_shape_dtype_match": True,
        "effective_state_gate": arm.get("state_gate_init"),
        "source_screen_condition_sources": list(
            evaluator.SOURCE_SCREEN_CONDITION_SOURCES
        ),
        "runtime_condition_sources": list(evaluator.CONDITION_SOURCES),
        "runtime_override_is_parameter_free": True,
    }
    if not _snapshot_load_contract_matches(
        snapshot,
        expected_snapshot_load,
    ):
        _fail(f"{arm_name} snapshot-load contract differs")
    source_contract = _mapping(
        sidecar.get("source_contract"),
        "artifact source contract",
    )
    expected_autonomous_shuffled = {
        "same_checkpoint": True,
        "uses_clean_hidden_future": False,
        "preserves_local_corruption_noise": True,
        "preserves_local_observed_history": True,
        "rolled_quantity": "generated noise-subtracted future TF residual",
        "roll_scope": "global 8-rank batch at every denoising step",
    }
    if (
        source_contract.get("nfe_steps") != list(evaluator.NFE_STEPS)
        or source_contract.get("condition_sources")
        != list(evaluator.CONDITION_SOURCES)
        or source_contract.get("source_screen_condition_sources")
        != list(evaluator.SOURCE_SCREEN_CONDITION_SOURCES)
        or source_contract.get("autonomous_shuffled")
        != expected_autonomous_shuffled
        or source_contract.get("oracle_sources_are_leakage") is not True
        or source_contract.get("all_sources_reuse_identical_initial_states")
        is not True
    ):
        _fail(f"{arm_name} artifact source contract differs")
    try:
        with safe_open(
            str(safetensors_path),
            framework="pt",
            device="cpu",
        ) as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise CompletedEvaluationError(
            f"could not inspect {arm_name} output safetensors: {exc}"
        ) from exc
    expected_metadata = {
        "iteration": str(evaluator.EVALUATION_ITERATION),
        "dataset": dataset,
        "sigma_convention": evaluator.SIGMA_CONVENTION,
        "evaluation_commit": manifest.get("evaluation_commit"),
        "snapshot_sha256": source_snapshot.get("sha256"),
        "input_batch_sha256": unit["batch_sha256"],
    }
    if metadata != expected_metadata:
        _fail(f"{arm_name} output safetensors metadata differs")
    return safetensors_path, sidecar_path


def _validate_tree_allowlist(root: Path, allowed_files: set[Path]) -> None:
    allowed_directories = {root}
    for path in allowed_files:
        parent = path.parent
        while parent != root:
            allowed_directories.add(parent)
            parent = parent.parent
        allowed_directories.add(root)
    actual_files: set[Path] = set()
    actual_directories = {root}
    for path in root.rglob("*"):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise CompletedEvaluationError(
                f"evaluation inventory changed while scanning: {path}"
            ) from exc
        if stat.S_ISLNK(mode) or stat.S_ISREG(mode):
            actual_files.add(path)
        elif stat.S_ISDIR(mode):
            actual_directories.add(path)
        else:
            _fail(f"unsupported filesystem entry in evaluation inventory: {path}")
    if actual_files != allowed_files or actual_directories != allowed_directories:
        _fail(f"physical tree differs from completion allowlist: {root}")


def _validate_inventory(
    *,
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
    evaluation_root: Path,
    arm_order: Sequence[str],
    arm_roots: Mapping[str, Path],
    batches_per_rank: int,
) -> dict[str, Any]:
    inventory = _mapping(completion.get("inventory"), "completion inventory")
    expected_pair_count = evaluator.WORLD_SIZE * batches_per_rank
    expected_artifact_count = expected_pair_count * len(arm_order)
    expected_counts = {
        "paired_unit_count": expected_pair_count,
        "input_artifact_count": expected_pair_count,
        "artifact_count": expected_artifact_count,
    }
    problems = [
        f"{key}: {inventory.get(key)!r} != {wanted!r}"
        for key, wanted in expected_counts.items()
        if inventory.get(key) != wanted
    ]
    if problems:
        _fail("completion inventory counts differ: " + "; ".join(problems))
    units = inventory.get("paired_units")
    if not isinstance(units, list) or len(units) != expected_pair_count:
        _fail("completion paired-unit list has the wrong length")
    expected_pairs = [
        (rank, batch)
        for rank in range(evaluator.WORLD_SIZE)
        for batch in range(batches_per_rank)
    ]
    observed_pairs = [
        (
            unit.get("global_rank"),
            unit.get("batch_index"),
        )
        if isinstance(unit, Mapping)
        else (None, None)
        for unit in units
    ]
    if observed_pairs != expected_pairs:
        _fail(
            "completion paired-unit inventory is partial, duplicated, or "
            "out of canonical order"
        )

    declared_input_paths: set[Path] = set()
    declared_output_paths: dict[str, set[Path]] = {
        name: set() for name in arm_order
    }
    generic_pairs: list[dict[str, Any]] = []
    datasets_by_batch: dict[int, set[str]] = {
        index: set() for index in range(batches_per_rank)
    }
    batch_hashes: list[str] = []
    for raw_unit in units:
        unit = _mapping(raw_unit, "completion paired unit")
        rank = unit["global_rank"]
        batch_index = unit["batch_index"]
        dataset = unit.get("dataset")
        match = DATASET_RE.fullmatch(dataset) if isinstance(dataset, str) else None
        if match is None or int(match.group(1)) != batch_index:
            _fail(f"dataset does not encode batch index {batch_index}: {dataset!r}")
        datasets_by_batch[batch_index].add(dataset)
        batch_sha256 = _full_sha256(
            unit.get("batch_sha256"),
            "input batch SHA-256",
        )
        batch_hashes.append(batch_sha256)
        expected_model_seed = (
            evaluator.BASE_EVALUATION_NOISE_SEED
            + evaluator.WORLD_SIZE * batch_index
        )
        expected_effective_seed = expected_model_seed + rank
        if (
            unit.get("model_evaluation_noise_seed") != expected_model_seed
            or unit.get("effective_generator_seed") != expected_effective_seed
        ):
            _fail(f"seed contract differs for rank/batch {(rank, batch_index)}")
        input_record = _mapping(
            unit.get("input_artifact"),
            "completion input artifact",
        )
        input_paths = _validate_input_sidecar(
            unit={**unit, "batch_sha256": batch_sha256},
            input_record=input_record,
            evaluation_root=evaluation_root,
        )
        declared_input_paths.update(input_paths)

        arm_artifacts = _mapping(
            unit.get("arm_artifacts"),
            "completion arm artifacts",
        )
        if set(arm_artifacts) != set(arm_order):
            _fail(
                f"completion selected-arm inventory differs for "
                f"{(rank, batch_index)}"
            )
        for arm_name in arm_order:
            record = _mapping(
                arm_artifacts[arm_name],
                f"{arm_name} completion artifact",
            )
            output_paths = _validate_output_sidecar(
                unit={**unit, "batch_sha256": batch_sha256},
                arm_name=arm_name,
                artifact_record=record,
                arm_root=arm_roots[arm_name],
                manifest=manifest,
            )
            declared_output_paths[arm_name].update(output_paths)
        generic_pairs.append({"dataset": dataset, "global_rank": rank})

    if len(set(batch_hashes)) != len(batch_hashes):
        _fail("input batch hashes are not globally unique")
    if any(len(names) != 1 for names in datasets_by_batch.values()):
        _fail("ranks do not share one identical dataset key per batch index")
    projected_pairs = {
        (unit["dataset"], unit["global_rank"]) for unit in units
    }
    if len(projected_pairs) != expected_pair_count:
        _fail("dataset/global-rank analyzer pairing is not injective")
    input_root = _canonical_directory(
        evaluation_root / "inputs",
        "evaluation input root",
    )
    _validate_tree_allowlist(input_root, declared_input_paths)
    for arm_name in arm_order:
        _validate_tree_allowlist(
            arm_roots[arm_name],
            declared_output_paths[arm_name],
        )
    all_output_paths = set().union(*declared_output_paths.values())
    _validate_tree_allowlist(
        evaluation_root / "artifacts",
        all_output_paths,
    )
    return {
        "paired_unit_count": expected_pair_count,
        "artifact_count": expected_artifact_count,
        "paired_units": sorted(
            generic_pairs,
            key=lambda item: (item["dataset"], item["global_rank"]),
        ),
        "input_paths": sorted(str(path) for path in declared_input_paths),
        "artifact_paths": {
            name: sorted(str(path) for path in declared_output_paths[name])
            for name in arm_order
        },
    }


def _validate_completed_evaluation(
    evaluation_root_value: str | Path,
    *,
    expected_completion_identity: str,
) -> dict[str, Any]:
    expected_identity = _full_sha256(
        expected_completion_identity,
        "expected completion identity",
    )
    evaluation_root = _canonical_directory(
        evaluation_root_value,
        "evaluation root",
    )
    manifest_path = _canonical_file(
        evaluation_root / MANIFEST_NAME,
        "evaluation manifest",
    )
    submission_path = _canonical_file(
        evaluation_root / SUBMISSION_NAME,
        "Slurm submission provenance",
    )
    completion_path = _canonical_file(
        evaluation_root / COMPLETION_NAME,
        "evaluation completion",
    )
    if (
        manifest_path.parent != evaluation_root
        or submission_path.parent != evaluation_root
        or completion_path.parent != evaluation_root
    ):
        _fail("manifest and completion must be direct evaluation-root children")
    manifest = _identity_json(manifest_path, "evaluation manifest")
    submission = _identity_json(
        submission_path,
        "Slurm submission provenance",
    )
    completion = _identity_json(completion_path, "evaluation completion")
    if manifest.get("kind") != MANIFEST_KIND:
        _fail("evaluation manifest kind differs")
    if completion.get("kind") != COMPLETION_KIND:
        _fail("evaluation completion kind differs")
    if submission.get("kind") != SUBMISSION_KIND:
        _fail("Slurm submission kind differs")
    if manifest.get("schema_version") != evaluator.SCHEMA_VERSION:
        _fail("evaluation manifest schema differs")
    if completion.get("schema_version") != evaluator.SCHEMA_VERSION:
        _fail("evaluation completion schema differs")
    if submission.get("schema_version") != evaluator.SCHEMA_VERSION:
        _fail("Slurm submission schema differs")
    if manifest.get("evaluation_root") != str(evaluation_root):
        _fail("manifest evaluation root differs")
    if completion.get("identity_sha256") != expected_identity:
        _fail("evaluation completion does not match the externally expected identity")
    try:
        coexistence = evaluator._validate_active_job_coexistence(
            manifest.get("active_job_coexistence")
        )
    except Exception as exc:
        raise CompletedEvaluationError(str(exc)) from exc
    expected_manifest_reference = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "identity_sha256": manifest.get("identity_sha256"),
    }
    if (
        submission.get("evaluation_manifest")
        != expected_manifest_reference
        or submission.get("active_job_coexistence") != coexistence
        or submission.get("requeue") is not False
        or submission.get("resume") is not False
        or not isinstance(submission.get("slurm_job_id"), str)
        or evaluator.JOB_ID_RE.fullmatch(submission["slurm_job_id"]) is None
    ):
        _fail("Slurm submission provenance differs")

    execution_path = _canonical_file(
        evaluation_root / "execution_started.json",
        "evaluation execution marker",
    )
    execution_started = _identity_json(
        execution_path,
        "evaluation execution marker",
    )
    expected_execution_started = {
        "schema_version": evaluator.SCHEMA_VERSION,
        "kind": "dual_abc_tf_causal_screen_posthoc_execution_started",
        "evaluation_manifest_sha256": _sha256(manifest_path),
        "evaluation_manifest_identity_sha256": manifest.get(
            "identity_sha256"
        ),
        "slurm_job_id": submission["slurm_job_id"],
        "active_job_coexistence": coexistence,
        "world_size": evaluator.WORLD_SIZE,
        "requeue": False,
        "resume": False,
        "wandb_enabled": False,
    }
    if any(
        execution_started.get(key) != value
        for key, value in expected_execution_started.items()
    ):
        _fail("evaluation execution marker contract differs")

    completion_manifest = _mapping(
        completion.get("evaluation_manifest"),
        "completion manifest reference",
    )
    if completion_manifest != expected_manifest_reference:
        _fail("completion does not reference the exact evaluation manifest")
    expected_completion = {
        "training_commit": manifest.get("training_commit"),
        "evaluation_commit": manifest.get("evaluation_commit"),
        "active_job_coexistence": coexistence,
        "source_inputs_unchanged": True,
        "wandb_enabled": False,
        "resume": False,
        "requeue": False,
        "paired_evaluation": manifest.get("paired_evaluation"),
    }
    problems = [
        f"{key}: {completion.get(key)!r} != {wanted!r}"
        for key, wanted in expected_completion.items()
        if completion.get(key) != wanted
    ]
    if problems:
        _fail("evaluation completion contract differs: " + "; ".join(problems))
    execution = _mapping(manifest.get("execution"), "evaluation execution")
    expected_execution = {
        "evaluation_only": True,
        "strict_snapshot_load": True,
        "sequential_arm_load": True,
        "wandb_enabled": False,
        "resume": False,
        "requeue": False,
        "source_writes_performed": False,
        "condition_source_subset_supported": False,
        "nfe_subset_supported": False,
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        _fail("evaluation execution contract differs")

    arm_order, arm_roots, batches_per_rank = _selected_arm_contract(
        manifest,
        evaluation_root,
    )
    source_validation = _validate_source_immutability(
        manifest,
        evaluation_root,
    )
    inventory = _validate_inventory(
        manifest=manifest,
        completion=completion,
        evaluation_root=evaluation_root,
        arm_order=arm_order,
        arm_roots=arm_roots,
        batches_per_rank=batches_per_rank,
    )
    return {
        "evaluation_root": evaluation_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "submission_path": submission_path,
        "submission": submission,
        "submission_sha256": _sha256(submission_path),
        "completion_path": completion_path,
        "completion": completion,
        "completion_sha256": _sha256(completion_path),
        "execution_path": execution_path,
        "execution_sha256": _sha256(execution_path),
        "arm_order": arm_order,
        "arm_roots": arm_roots,
        "batches_per_rank": batches_per_rank,
        "source_validation": source_validation,
        "inventory": inventory,
    }


def _prepare_output(
    output_value: str | Path,
    *,
    evaluation_root: Path,
    immutable_roots: Sequence[Path],
) -> Path:
    raw = Path(output_value).expanduser()
    if raw.suffix.lower() != ".json":
        _fail("analysis output must end in .json")
    parent = _canonical_directory(raw.parent, "analysis output parent")
    output = parent / raw.name
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        _fail(f"analysis output must be fresh: {output}")
    forbidden = [evaluation_root, *immutable_roots]
    if any(output == root or _is_beneath(output, root) for root in forbidden):
        _fail("analysis output must be outside all immutable evaluation roots")
    return output


def _temporary_generic_output(output: Path) -> Path:
    for _ in range(32):
        candidate = output.parent / (
            f".{output.name}.generic-{os.getpid()}-{secrets.token_hex(12)}.json"
        )
        try:
            candidate.lstat()
        except FileNotFoundError:
            return candidate
    _fail("could not reserve a fresh generic-analysis temporary name")


def _validate_generic_result(
    payload: Mapping[str, Any],
    *,
    validated: Mapping[str, Any],
    baseline: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> None:
    if payload.get("baseline_arm") != baseline:
        _fail("generic analyzer returned a different baseline")
    if payload.get("bootstrap") != {
        "method": (
            "paired nonparametric percentile bootstrap over dataset/rank units"
        ),
        "samples": bootstrap_samples,
        "seed": bootstrap_seed,
        "confidence": confidence,
        "per_statistic_seed_derivation": (
            "sha256(seed + NUL + statistic label)"
        ),
    }:
        _fail("generic analyzer returned different bootstrap parameters")
    provenance = _mapping(payload.get("provenance"), "generic provenance")
    inventory = validated["inventory"]
    if (
        provenance.get("iteration") != evaluator.EVALUATION_ITERATION
        or provenance.get("paired_unit_count")
        != inventory["paired_unit_count"]
        or provenance.get("paired_units") != inventory["paired_units"]
    ):
        _fail("generic analyzer returned a different paired inventory")
    generic_arms = _mapping(provenance.get("arms"), "generic arm provenance")
    if set(generic_arms) != set(validated["arm_order"]):
        _fail("generic analyzer returned a different arm set")
    for name in validated["arm_order"]:
        arm = _mapping(generic_arms[name], f"generic arm {name}")
        if (
            arm.get("root") != str(validated["arm_roots"][name])
            or arm.get("artifact_count")
            != inventory["paired_unit_count"]
            or arm.get("evaluation_condition_sources")
            != list(evaluator.CONDITION_SOURCES)
        ):
            _fail(f"generic analyzer returned different provenance for {name}")
    aggregate = _mapping(payload.get("aggregate"), "generic aggregate")
    same_checkpoint = _mapping(
        aggregate.get(
            "same_checkpoint_autonomous_vs_autonomous_shuffled"
        ),
        "same-checkpoint autonomous-shuffled analysis",
    )
    if (
        same_checkpoint.get("nfe_1_exact_endpoint_noop_required") is not True
        or set(
            _mapping(
                same_checkpoint.get("contrasts"),
                "same-checkpoint contrasts",
            )
        )
        != set(validated["arm_order"])
    ):
        _fail(
            "generic analyzer omitted the mandatory same-checkpoint "
            "autonomous-shuffled contrast"
        )


def analyze_completed_evaluation(
    evaluation_root: str | Path,
    *,
    expected_completion_identity: str,
    baseline: str,
    output: str | Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_726,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Validate a completed evaluation and exclusively write one analysis."""

    validated = _validate_completed_evaluation(
        evaluation_root,
        expected_completion_identity=expected_completion_identity,
    )
    if baseline not in validated["arm_order"]:
        _fail(f"baseline arm {baseline!r} is not a selected evaluation arm")
    immutable_roots = [
        Path(validated["source_validation"]["evaluation_repository"]),
        Path(validated["source_validation"]["training_repository"]),
        Path(validated["source_validation"]["source_screen_root"]),
        Path(validated["source_validation"]["data_root"]),
        Path(validated["source_validation"]["wan_directory"]),
        Path(validated["source_validation"]["videox_home"]),
    ]
    output_path = _prepare_output(
        output,
        evaluation_root=validated["evaluation_root"],
        immutable_roots=immutable_roots,
    )
    temporary_output = _temporary_generic_output(output_path)
    generic_payload: Mapping[str, Any]
    try:
        generic_payload = generic.analyze(
            validated["arm_roots"],
            baseline=baseline,
            output=temporary_output,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
        persisted = _strict_json(
            _canonical_file(temporary_output, "generic analysis output"),
            "generic analysis output",
        )
        if persisted != generic_payload:
            _fail("generic analyzer return value differs from persisted output")
    finally:
        try:
            temporary_output.unlink()
        except FileNotFoundError:
            pass

    _validate_generic_result(
        generic_payload,
        validated=validated,
        baseline=baseline,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    revalidated = _validate_completed_evaluation(
        validated["evaluation_root"],
        expected_completion_identity=expected_completion_identity,
    )
    for key in (
        "manifest_sha256",
        "submission_sha256",
        "completion_sha256",
        "execution_sha256",
        "arm_order",
        "inventory",
        "source_validation",
    ):
        if revalidated[key] != validated[key]:
            _fail(f"completed evaluation changed during analysis: {key}")

    analyzer_path = _canonical_file(
        Path(generic.__file__).resolve(),
        "generic analyzer source",
    )
    wrapper_path = _canonical_file(
        Path(__file__).resolve(),
        "completion-aware analyzer source",
    )
    payload = evaluator._identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": ANALYSIS_KIND,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "evaluation": {
                "root": str(validated["evaluation_root"]),
                "manifest": {
                    "path": str(validated["manifest_path"]),
                    "sha256": validated["manifest_sha256"],
                    "identity_sha256": validated["manifest"][
                        "identity_sha256"
                    ],
                },
                "completion": {
                    "path": str(validated["completion_path"]),
                    "sha256": validated["completion_sha256"],
                    "identity_sha256": expected_completion_identity,
                },
                "submission": {
                    "path": str(validated["submission_path"]),
                    "sha256": validated["submission_sha256"],
                    "identity_sha256": validated["submission"][
                        "identity_sha256"
                    ],
                    "slurm_job_id": validated["submission"][
                        "slurm_job_id"
                    ],
                    "active_job_coexistence": validated["submission"][
                        "active_job_coexistence"
                    ],
                },
                "execution_started": {
                    "path": str(validated["execution_path"]),
                    "sha256": validated["execution_sha256"],
                },
                "selected_arms": list(validated["arm_order"]),
                "world_size": evaluator.WORLD_SIZE,
                "batches_per_rank": validated["batches_per_rank"],
                "paired_unit_count": validated["inventory"][
                    "paired_unit_count"
                ],
                "artifact_count": validated["inventory"]["artifact_count"],
                "source_validation": validated["source_validation"],
            },
            "analyzer": {
                "wrapper": evaluator._file_record(wrapper_path),
                "generic": evaluator._file_record(analyzer_path),
                "baseline": baseline,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "confidence": confidence,
                "delegated_arm_roots": {
                    name: str(validated["arm_roots"][name])
                    for name in validated["arm_order"]
                },
                "generic_payload_sha256": hashlib.sha256(
                    evaluator._canonical_json_bytes(generic_payload)
                ).hexdigest(),
            },
            "analysis": generic_payload,
        }
    )
    try:
        evaluator._exclusive_json(output_path, payload)
    except Exception as exc:
        raise CompletedEvaluationError(
            f"could not exclusively create analysis output: {exc}"
        ) from exc
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument(
        "--expected-completion-identity",
        required=True,
        help="identity_sha256 printed by the completed evaluator",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="fresh JSON path outside evaluation and immutable source roots",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_726)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_completed_evaluation(
            args.evaluation_root,
            expected_completion_identity=args.expected_completion_identity,
            baseline=args.baseline,
            output=args.output,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
        )
    except (CompletedEvaluationError, generic.ArtifactValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "identity_sha256": result["identity_sha256"],
                "paired_unit_count": result["evaluation"][
                    "paired_unit_count"
                ],
                "selected_arms": result["evaluation"]["selected_arms"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
