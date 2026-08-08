#!/usr/bin/env python3
"""Prospective registration and sealed-critic tooling for Action Cycle Stage 1.

Registration is impossible unless the referenced Stage-0 result has a valid
identity and the preregistered GO decision.  Only its train-fitted aligned
ridge weights and train population normalization are copied into the Stage-1
critic bundle; validation rows, shuffled-control weights, and protected-test
artifacts are neither copied nor opened by training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion.action_cycle import (  # noqa: E402
    FEATURE_DIM,
    FUTURE_RELEVANT_TRANSITIONS,
    TARGET_DIM,
)
from tools import action_cycle_recoverability as stage0  # noqa: E402
from tools import dual_abc_pilot  # noqa: E402
from tools import lamo_motion_drift_evaluate as common  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


SCHEMA_VERSION = 1
KIND_REGISTRATION = "action_cycle_stage1_registration_v1"
KIND_CRITIC = "action_cycle_stage1_train_only_critic_v1"
EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
PARENT_SNAPSHOT_SHA256 = common.PARENT_SNAPSHOT_SHA256
PARENT_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
EXPECTED_VALIDATION_CLIPS = 64
SHUFFLE_SEED = 20_260_810
PROTOCOL_PATH = (
    REPO_ROOT / "docs" / "experiments" / "ACTION_CYCLE_STAGE1_PROTOCOL.md"
)
RUNBOOK_PATH = (
    REPO_ROOT / "docs" / "experiments" / "ACTION_CYCLE_STAGE1_RUNBOOK.md"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ActionCycleStage1Error(RuntimeError):
    """A prerequisite, frozen input, or prospective contract changed."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    loss_weight: float


ARMS = (
    Arm(
        "AC-OFF",
        "ravenhuang/wan-dit/action_cycle_stage1_off",
        "action-cycle-stage1-off-seed1234-u000200",
        0.0,
    ),
    Arm(
        "AC-ON",
        "ravenhuang/wan-dit/action_cycle_stage1_on",
        "action-cycle-stage1-on-seed1234-u000200",
        0.05,
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


@dataclass(frozen=True)
class Endpoint:
    code: str
    nfe: int
    action_source: str
    primary: bool


ENDPOINTS = (
    Endpoint("aligned_nfe_1", 1, "aligned", True),
    Endpoint("aligned_nfe_4", 4, "aligned", False),
    Endpoint("shuffled_nfe_1", 1, "shuffled", True),
    Endpoint("zero_nfe_1", 1, "zero", True),
)
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


def fixed_protocol() -> dict[str, Any]:
    return {
        "stage": "generated_latent_action_cycle_stage1",
        "prerequisite": {
            "kind": stage0.ANALYSIS_KIND,
            "decision": "go_to_generated_latent_action_cycle_stage1",
            "paired_bootstrap_all_passed": True,
            "stage0_fixed_protocol": stage0.fixed_protocol(),
            "validation_metrics_used_to_choose_stage1_hyperparameters": False,
        },
        "arms": [asdict(arm) for arm in ARMS],
        "continuation_updates": 200,
        "seed": 1234,
        "world_size": 8,
        "local_batch_size": 1,
        "global_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "optimizer_state_policy": "fresh_identical_adamw",
        "ema_policy": "none_in_parent_and_none_in_both_arms",
        "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
        "action_cycle": {
            "predicted_clean": "x0_hat=x_sigma-sigma*v_theta",
            "feature": stage0.fixed_protocol()["feature"],
            "target": stage0.fixed_protocol()["target"],
            "critic": "Stage0_train_only_aligned_ridge_and_population_normalization",
            "critic_trainable_parameters": 0,
            "loss": "mean_standardized_inverse_action_mse",
            "loss_weight": 0.05,
            "transitions": list(FUTURE_RELEVANT_TRANSITIONS),
            "transition_rationale": (
                "transition_1_crosses_observed_to_generated_and_transition_2_is_generated; "
                "history_only_transition_0_is_telemetry_not_optimized"
            ),
            "extra_wan_calls": 0,
            "checkpoint_parameter_or_buffer_delta": 0,
        },
        "training_inputs": {
            "split": "train512_only",
            "trainer_validation_split": "train512_only",
            "clean_future_rgb_consumed_by_cycle_target": False,
            "cached_feature_target_opened": False,
        },
        "evaluation": {
            "split": "sealed_val64",
            "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
            "initial_noise_paired_by_immutable_clip_id": True,
            "action_shuffle": {
                "method": "deterministic_episode_disjoint_bijection",
                "seed": SHUFFLE_SEED,
            },
            "zero_action_preserves_morphology": True,
            "scoring_constructed_after_all_sampling": True,
            "clean_future_or_feature_passed_to_sampler": False,
            "critic_loaded_or_called": False,
            "online_teacher_calls": 0,
        },
        "analysis_gate": {
            "primary_endpoint": "aligned_nfe_1",
            "bootstrap_unit": "paired_validation_clip",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 20_260_811,
            "familywise_alpha": 0.05,
            "primary_claim_contrasts": 9,
            "quality": {
                "decoded_mse_relative_improvement_min": 0.03,
                "decoded_temporal_mse_relative_improvement_min": 0.03,
                "both_simultaneous_lower_bounds_min": 0.01,
                "latent_nmse_point_and_lower_bound_min": -0.01,
            },
            "action_attribution": {
                "candidate_aligned_vs_shuffled_decoded_and_temporal_min": 0.0,
                "candidate_aligned_vs_zero_decoded_and_temporal_min": 0.0,
                "shuffled_difference_in_differences_decoded_and_temporal_min": 0.0,
                "all_simultaneous_lower_bounds_strictly_positive": True,
            },
            "all_checks_required": True,
        },
        "wandb": {
            "entity": EXPECTED_ENTITY,
            "project": EXPECTED_PROJECT,
            "access": "PRIVATE",
            "group": None,
        },
        "protected_test_access_allowed": False,
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return bool(
        isinstance(identity, str)
        and SHA256_RE.fullmatch(identity)
        and hashlib.sha256(canonical_json(unsigned)).hexdigest() == identity
    )


def file_record(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser()
    try:
        info = value.lstat()
    except FileNotFoundError as exc:
        raise ActionCycleStage1Error(f"missing artifact: {value}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ActionCycleStage1Error(f"artifact must be a nonempty regular file: {value}")
    value = value.resolve(strict=True)
    return {
        "path": str(value),
        "bytes": int(value.stat().st_size),
        "sha256": common._sha256(value),
    }


def read_json(path: str | Path, label: str) -> dict[str, Any]:
    return common._read_json(Path(path), label)


def revalidate_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    observed = file_record(str(record.get("path", "")))
    if any(observed.get(key) != record.get(key) for key in ("path", "bytes", "sha256")):
        raise ActionCycleStage1Error(f"registered {label} changed")
    return observed


def arm_run_identity(registration: Mapping[str, Any], arm: Arm) -> str:
    return identity_payload(
        {
            "kind": "action_cycle_stage1_arm_identity_v1",
            "registration_identity_sha256": registration["identity_sha256"],
            "tool_git_commit": registration["tool_repository"]["git_commit"],
            "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "critic_bundle_sha256": registration["stage0"]["critic_bundle"]["sha256"],
            "arm": asdict(arm),
            "updates": 200,
            "seed": 1234,
            "world_size": 8,
            "local_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer_state_policy": "fresh_identical_adamw",
            "ema_policy": "none_in_parent_and_none_in_both_arms",
        }
    )["identity_sha256"]


def validate_stage0_go_result(
    registration: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    """Reject every Stage-0 outcome except the sealed preregistered GO."""

    gate = result.get("paired_bootstrap_gate", {})
    if (
        not stage0.identity_valid(result)
        or result.get("kind") != stage0.ANALYSIS_KIND
        or result.get("registration_identity_sha256")
        != registration.get("identity_sha256")
        or result.get("decision") != "go_to_generated_latent_action_cycle_stage1"
        or not isinstance(gate, Mapping)
        or gate.get("all_passed") is not True
        or result.get("validation_fit_observations") != 0
        or result.get("protected_test_accessed") is not False
        or result.get("target_cache_array_opened") is not False
    ):
        raise ActionCycleStage1Error(
            "Stage-0 did not produce the preregistered GO result"
        )


def _validate_stage0(study_root: Path) -> dict[str, Any]:
    root = study_root.expanduser().resolve(strict=True)
    registration_path = root / "registration.json"
    result_path = root / "analysis" / "stage0" / "result.json"
    registration = read_json(registration_path, "Stage-0 registration")
    repository = registration.get("repository", {})
    if (
        not stage0.identity_valid(registration)
        or registration.get("kind") != stage0.REGISTRATION_KIND
        or registration.get("status")
        != "registered_before_encoding_or_validation_metrics"
        or registration.get("fixed_protocol") != stage0.fixed_protocol()
        or registration.get("target_cache_array_opened") is not False
        or registration.get("protected_test_accessed") is not False
        or registration.get("wandb", {}).get("entity") != stage0.EXPECTED_ENTITY
        or registration.get("wandb", {}).get("project")
        != stage0.EXPECTED_PROJECT
        or registration.get("wandb", {}).get("access") != "PRIVATE"
        or registration.get("wandb", {}).get("group") is not None
        or not isinstance(repository, Mapping)
        or not isinstance(repository.get("path"), str)
        or not isinstance(repository.get("commit"), str)
        or COMMIT_RE.fullmatch(repository["commit"]) is None
    ):
        raise ActionCycleStage1Error("Stage-0 registration identity/protocol differs")
    if registration_path.resolve(strict=True) != (root / "registration.json").resolve(
        strict=True
    ):
        raise ActionCycleStage1Error("Stage-0 registration path differs")
    # Validate the source recorded by Stage 0 rather than requiring it to be
    # this Stage-1 worktree. The original validator intentionally binds its own
    # REPO_ROOT and therefore cannot be reused cross-worktree.
    common._clean_source(
        Path(repository["path"]), repository["commit"], "Stage-0 source"
    )
    for name, record in registration.get("tools", {}).items():
        revalidate_record(record, f"Stage-0 tool {name}")
    stage0.validate_temporal_oracle_payload(
        registration.get("train_only_temporal_control_oracle_feasibility", {})
    )
    result = read_json(result_path, "Stage-0 result")
    validate_stage0_go_result(registration, result)
    model_record = result.get("model")
    if not isinstance(model_record, Mapping):
        raise ActionCycleStage1Error("Stage-0 result lacks frozen model metadata")
    model_json_record = revalidate_record(model_record, "Stage-0 model metadata")
    model_metadata = read_json(model_json_record["path"], "Stage-0 model metadata")
    if (
        not stage0.identity_valid(model_metadata)
        or model_metadata.get("kind") != stage0.MODEL_KIND
        or model_metadata.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or model_metadata.get("selection_split") != "train_only_exact_loo"
        or model_metadata.get("validation_fit_observations") != 0
        or model_metadata.get("fit_observations_per_transition_view") != 512
        or model_metadata.get("model_strata") != 9
        or model_metadata.get("protected_test_accessed") is not False
        or model_metadata.get("target_cache_array_opened") is not False
    ):
        raise ActionCycleStage1Error("Stage-0 frozen ridge is not train-only")
    ridge_record = model_metadata.get("artifact")
    if not isinstance(ridge_record, Mapping):
        raise ActionCycleStage1Error("Stage-0 ridge array record is absent")
    ridge_record = revalidate_record(ridge_record, "Stage-0 ridge arrays")
    ridge_path = Path(ridge_record["path"])
    if ridge_path != root / "analysis" / "stage0" / "frozen_ridge.npz":
        raise ActionCycleStage1Error("Stage-0 ridge path is outside canonical analysis root")
    archive = np.load(ridge_path, allow_pickle=False)
    required = {
        "selected_alpha": ((), np.float64),
        "feature_mean": ((3, 3, FEATURE_DIM), np.float32),
        "feature_std": ((3, 3, FEATURE_DIM), np.float32),
        "feature_active": ((3, 3, FEATURE_DIM), np.bool_),
        "target_mean": ((3, TARGET_DIM), np.float32),
        "target_std": ((3, TARGET_DIM), np.float32),
        "target_active": ((3, TARGET_DIM), np.bool_),
        "aligned_weight": ((3, 3, FEATURE_DIM, TARGET_DIM), np.float32),
    }
    copied: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in required.items():
        if name not in archive.files:
            raise ActionCycleStage1Error(f"Stage-0 ridge lacks {name}")
        value = archive[name]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ActionCycleStage1Error(f"Stage-0 ridge {name} shape/dtype differs")
        if value.dtype != np.bool_ and not np.isfinite(value).all():
            raise ActionCycleStage1Error(f"Stage-0 ridge {name} is non-finite")
        copied[name] = np.array(value, copy=True)
    archive.close()
    alpha = float(copied["selected_alpha"].item())
    if alpha != float(model_metadata.get("selected_alpha", math.nan)) or alpha != float(
        result.get("selected_alpha", math.nan)
    ):
        raise ActionCycleStage1Error("Stage-0 selected alpha metadata differs")
    return {
        "root": str(root),
        "registration": file_record(registration_path),
        "registration_identity_sha256": registration["identity_sha256"],
        "result": file_record(result_path),
        "result_identity_sha256": result["identity_sha256"],
        "model_metadata": model_json_record,
        "source_ridge": ridge_record,
        "source_commit": registration["repository"]["commit"],
        "arrays": copied,
        "selected_alpha": alpha,
        "stage0_fixed_protocol": registration["fixed_protocol"],
    }


def _exclusive_npz(path: Path, **arrays: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_critic(root: Path, validated: Mapping[str, Any]) -> dict[str, Any]:
    critic_dir = root / "critic"
    critic_dir.mkdir(mode=0o700)
    path = critic_dir / "train_only_frozen_ridge.npz"
    arrays = dict(validated["arrays"])
    arrays.update(
        {
            "stage0_registration_identity_sha256": np.asarray(
                validated["registration_identity_sha256"]
            ),
            "source_frozen_ridge_sha256": np.asarray(
                validated["source_ridge"]["sha256"]
            ),
        }
    )
    _exclusive_npz(path, **arrays)
    record = file_record(path)
    metadata = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_CRITIC,
            "created_at_utc": now_utc(),
            "artifact": record,
            "stage0_registration_identity_sha256": validated[
                "registration_identity_sha256"
            ],
            "stage0_result_identity_sha256": validated["result_identity_sha256"],
            "source_frozen_ridge": validated["source_ridge"],
            "selection_split": "train_only_exact_loo",
            "validation_fit_observations": 0,
            "copied_arrays": sorted(arrays),
            "excluded_source_arrays": [
                "shuffled_control_weight",
                "validation_episode_disjoint_permutation",
                "train_episode_disjoint_permutation",
                "alpha_grid",
                "alpha_train_loo_scores",
            ],
            "critic_trainable_parameters": 0,
            "protected_test_accessed": False,
        }
    )
    metadata_path = critic_dir / "metadata.json"
    common._exclusive_json(metadata_path, metadata)
    return {
        "bundle": record,
        "metadata": file_record(metadata_path),
        "metadata_identity_sha256": metadata["identity_sha256"],
    }


def command_register(args: argparse.Namespace) -> int:
    tool_repo = args.tool_repo.expanduser().resolve(strict=True)
    historical_repo = args.historical_repo.expanduser().resolve(strict=True)
    source = common._clean_source(tool_repo, args.expected_commit, "tool")
    historical = common._clean_source(
        historical_repo, common.TRAINING_COMMIT, "historical model"
    )
    if tool_repo != REPO_ROOT:
        raise ActionCycleStage1Error("executing tool repository differs")
    if any(not path.is_file() or path.is_symlink() for path in (PROTOCOL_PATH, RUNBOOK_PATH)):
        raise ActionCycleStage1Error("prospective protocol/runbook is absent")
    ancestry = os.spawnlp(
        os.P_WAIT,
        "git",
        "git",
        "-C",
        str(tool_repo),
        "merge-base",
        "--is-ancestor",
        common.TRAINING_COMMIT,
        args.expected_commit,
    )
    if ancestry != 0:
        raise ActionCycleStage1Error("tool commit is not a historical-model descendant")
    output_root = common._fresh_lustre_root(args.output_root)
    parent = phase._validate_study_metadata(
        args.vpm_study_root.expanduser().resolve(strict=True),
        historical_repo,
        rehash_snapshot=True,
    )
    if parent["snapshot_sha256"] != PARENT_SNAPSHOT_SHA256:
        raise ActionCycleStage1Error("exact VPM parent snapshot changed")
    train = common._validate_train_inputs(args.train_manifest, args.train_cache_metadata)
    train_descriptors = train.pop("descriptors")
    validation_descriptors = common._manifest_descriptors(
        Path(parent["validation"]["manifest"]["path"]),
        expected_split="val",
        expected_count=EXPECTED_VALIDATION_CLIPS,
    )
    if validation_descriptors != parent["descriptors"]:
        raise ActionCycleStage1Error("validation descriptors differ from sealed study")
    disjointness = common._split_disjointness(train_descriptors, validation_descriptors)
    stage0_validated = _validate_stage0(args.stage0_study_root)
    if stage0_validated["stage0_fixed_protocol"] != stage0.fixed_protocol():
        raise ActionCycleStage1Error("Stage-0 registered scientific protocol differs")
    if stage0_validated["registration_identity_sha256"] != read_json(
        stage0_validated["registration"]["path"], "Stage-0 registration"
    )["identity_sha256"]:
        raise ActionCycleStage1Error("Stage-0 registration binding differs")
    wandb = dual_abc_pilot._wandb_private_project(EXPECTED_ENTITY, EXPECTED_PROJECT)
    python_input = args.python.expanduser()
    if not python_input.is_absolute():
        raise ActionCycleStage1Error("runtime Python must be absolute")
    python = python_input.parent.resolve(strict=True) / python_input.name
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ActionCycleStage1Error("runtime Python is not executable")

    output_root.mkdir(mode=0o700)
    critic = _write_critic(output_root, stage0_validated)
    stage0_public = {
        key: value for key, value in stage0_validated.items() if key != "arrays"
    }
    stage0_public["critic_bundle"] = critic["bundle"]
    stage0_public["critic_metadata"] = critic["metadata"]
    stage0_public["critic_metadata_identity_sha256"] = critic[
        "metadata_identity_sha256"
    ]
    payload = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": now_utc(),
            "status": "registered_before_stage1_training_or_validation_metrics",
            "output_root": str(output_root),
            "tool_repository": source,
            "historical_model_repository": historical,
            "protocol": file_record(PROTOCOL_PATH),
            "runbook": file_record(RUNBOOK_PATH),
            "controlled_study": {
                "study_root": str(args.vpm_study_root.resolve(strict=True)),
                "study_identity_sha256": parent["study"]["identity_sha256"],
                "arm_identity_sha256": parent["arm"]["identity_sha256"],
                "stage_identity_sha256": parent["stage"]["identity_sha256"],
                "parent_snapshot": file_record(parent["paths"]["snapshot"]),
                "parent_completed_updates": 1000,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "training_git_commit": common.TRAINING_COMMIT,
            },
            "stage0": stage0_public,
            "training": train,
            "validation": parent["validation"],
            "validation_descriptors": validation_descriptors,
            "train_validation_disjointness": disjointness,
            "runtime": {**parent["runtime"], "python": str(python)},
            "fixed_protocol": fixed_protocol(),
            "wandb": {**wandb, "group": None, "mode": "online"},
            "stage1_hyperparameters_frozen_before_stage0_validation_read": True,
            "protected_test_accessed": False,
            "cached_feature_target_opened": False,
        }
    )
    common._exclusive_json(output_root / "registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def validate_registration(path: Path, *, rehash_inputs: bool = False) -> dict[str, Any]:
    value = path.expanduser().resolve(strict=True)
    registration = read_json(value, "Stage-1 registration")
    if (
        not identity_valid(registration)
        or registration.get("kind") != KIND_REGISTRATION
        or registration.get("status")
        != "registered_before_stage1_training_or_validation_metrics"
        or registration.get("fixed_protocol") != fixed_protocol()
        or registration.get("stage1_hyperparameters_frozen_before_stage0_validation_read")
        is not True
        or registration.get("protected_test_accessed") is not False
        or registration.get("cached_feature_target_opened") is not False
    ):
        raise ActionCycleStage1Error("Stage-1 registration identity/protocol differs")
    canonical = Path(registration["output_root"]) / "registration.json"
    if value != canonical.resolve(strict=True):
        raise ActionCycleStage1Error("registration is outside its canonical root")
    source = registration["tool_repository"]
    if common._clean_source(
        Path(source["path"]), source["git_commit"], "registered tool"
    ) != source:
        raise ActionCycleStage1Error("registered source changed")
    executing = common._clean_source(REPO_ROOT, source["git_commit"], "executing tool")
    if executing["git_tree_sha"] != source["git_tree_sha"]:
        raise ActionCycleStage1Error("executing source tree differs")
    stage0_value = registration["stage0"]
    for key in (
        "registration",
        "result",
        "model_metadata",
        "source_ridge",
        "critic_bundle",
        "critic_metadata",
    ):
        revalidate_record(stage0_value[key], f"Stage-0/{key}")
    critic_metadata = read_json(stage0_value["critic_metadata"]["path"], "critic metadata")
    if (
        not identity_valid(critic_metadata)
        or critic_metadata.get("kind") != KIND_CRITIC
        or critic_metadata.get("artifact") != stage0_value["critic_bundle"]
        or critic_metadata.get("stage0_registration_identity_sha256")
        != stage0_value["registration_identity_sha256"]
        or critic_metadata.get("validation_fit_observations") != 0
        or critic_metadata.get("protected_test_accessed") is not False
    ):
        raise ActionCycleStage1Error("sealed critic metadata differs")
    if rehash_inputs:
        common.revalidate_registered_inputs(
            registration,
            include_parent=True,
            include_train=True,
            include_validation=True,
        )
    return registration


def command_validate(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration, rehash_inputs=args.full)
    print(registration["identity_sha256"])
    return 0


def command_plan(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration, rehash_inputs=False)
    root = Path(registration["output_root"])
    payload = {
        "kind": "action_cycle_stage1_dry_run_v1",
        "mode": "no_commands_executed_no_files_created",
        "registration_identity_sha256": registration["identity_sha256"],
        "arms": [
            {
                "arm": asdict(arm),
                "run_identity_sha256": arm_run_identity(registration, arm),
                "training_dir": str(root / "training" / arm.run_name),
                "evaluation_dir": str(root / "evaluation" / arm.code.lower()),
            }
            for arm in ARMS
        ],
        "analysis": str(root / "analysis" / "analysis.json"),
        "protected_test_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--tool-repo", type=Path, required=True)
    register.add_argument("--historical-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.add_argument("--vpm-study-root", type=Path, required=True)
    register.add_argument("--stage0-study-root", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.set_defaults(func=command_register)
    validate = commands.add_parser("validate-registration")
    validate.add_argument("--registration", type=Path, required=True)
    validate.add_argument("--full", action="store_true")
    validate.set_defaults(func=command_validate)
    plan = commands.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.set_defaults(func=command_plan)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
