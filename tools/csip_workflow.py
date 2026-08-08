#!/usr/bin/env python3
"""Register, seal, and render the frozen Phase-0 CSIP workflow.

This program never submits a job.  ``register`` freezes source, immutable
train/validation caches, Wan runtime, protocol, output paths, and a personal
private W&B destination before latent extraction or fitting.  ``seal`` binds
the fixed update-400 checkpoint before the validation cache may be opened.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import csip_contract as contract  # noqa: E402


SEAL_KIND = "csip_phase0_fixed_checkpoint_seal"


def _wandb_attestation(entity: str, project: str) -> dict[str, str]:
    if entity != contract.EXPECTED_ENTITY or project != contract.EXPECTED_PROJECT:
        raise contract.CSIPContractError(
            f"CSIP is locked to {contract.EXPECTED_ENTITY}/{contract.EXPECTED_PROJECT}"
        )
    from tools.dual_abc_pilot import _wandb_private_project

    record = _wandb_private_project(entity, project)
    if str(record.get("access", "")).upper() != "PRIVATE":
        raise contract.CSIPContractError("registered W&B project is not private")
    return record


def _protocol_records() -> dict[str, Any]:
    files = (
        "docs/experiments/CSIP_PHASE0_PROTOCOL.md",
        "docs/experiments/CSIP_PHASE0_RUNBOOK.md",
        "robot_wm/modeling/dual_diffusion/causal_spectral_probe.py",
        "tools/csip_contract.py",
        "tools/csip_workflow.py",
        "tools/csip_latent_cache.py",
        "tools/csip_train.py",
        "tools/csip_evaluate.py",
        "tools/csip_analyze.py",
        "tools/env/activate_b200.sh",
        "tools/env/verify_b200_runtime.py",
        "tools/slurm/csip_phase0_stage.sbatch",
        "tools/slurm/submit_csip_phase0.sh",
    )
    return {
        relative: contract.file_record(REPO_ROOT / relative, relative)
        for relative in files
    }


def command_wandb_check(args: argparse.Namespace) -> int:
    print(json.dumps(_wandb_attestation(args.entity, args.project), sort_keys=True))
    return 0


def command_register(args: argparse.Namespace) -> int:
    if not args.ack_private_wandb_project:
        raise contract.CSIPContractError(
            "registration requires --ack-private-wandb-project"
        )
    if contract.SAFE_ID_RE.fullmatch(args.study_id) is None:
        raise contract.CSIPContractError("study ID is not path-safe")
    source = contract.clean_source(args.tool_repo, args.expected_commit)
    root = contract.artifact_root(args.study_root, must_be_fresh=True)
    if root.name != args.study_id:
        raise contract.CSIPContractError("study-root basename must equal study-id")

    # Registration deliberately performs full hashes of the large source
    # arrays.  Later stages can therefore inherit a content-bound identity
    # instead of trusting only paths and file sizes.
    train, train_rows = contract.source_split_record(
        args.train_manifest,
        args.train_cache_metadata,
        split="train",
        rehash_arrays=True,
    )
    validation, validation_rows = contract.source_split_record(
        args.validation_manifest,
        args.validation_cache_metadata,
        split="val",
        rehash_arrays=True,
    )
    split_isolation = contract.assert_split_isolation(train_rows, validation_rows)
    runtime = contract.runtime_record(args.python, args.wan_dir, args.videox_home)
    wandb = _wandb_attestation(args.wandb_entity, args.wandb_project)
    viewer_email = wandb.get("viewer_email", "")
    planned = {
        "train_latent_root": str(root / "latents" / "train"),
        "train_latent_metadata": str(root / "latents" / "train" / "metadata.json"),
        "validation_latent_root": str(root / "latents" / "validation"),
        "validation_latent_metadata": str(
            root / "latents" / "validation" / "metadata.json"
        ),
        "checkpoint": str(root / "training" / "checkpoint-u000400.pt"),
        "training_report": str(root / "training" / "report.json"),
        "checkpoint_seal": str(root / "checkpoint-seal.json"),
        "evaluation": str(root / "evaluation" / "val64.json"),
        "analysis": str(root / "analysis" / "bootstrap-gate.json"),
        "wandb_local_dir": str(root / "wandb"),
    }
    payload = contract.with_identity(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.REGISTRATION_KIND,
            "created_at_utc": contract.now_utc(),
            "status": "registered_before_latent_extraction_or_training",
            "study_id": args.study_id,
            "study_root": str(root),
            "source": source,
            "protocol_files": _protocol_records(),
            "datasets": {"train": train, "validation": validation},
            "split_isolation": split_isolation,
            "runtime": runtime,
            "representation": {
                "input": "clean_real_wan_future_motion_latents",
                "history_encoding": "independent_five_frame_observed_only",
                "camera_transform": "per_view_never_across_width_stack_seams",
                "spatiotemporal_fft": "centered_coarse_2x4x6_per_view_per_channel",
                "channels": [
                    "log1p_magnitude",
                    "energy_masked_unit_phase_real",
                    "energy_masked_unit_phase_imag",
                    "energy_masked_temporal_phase_increment_real",
                    "energy_masked_temporal_phase_increment_imag",
                ],
                "relative_energy_floor": 1e-3,
                "feature_dim": 9216,
                "phase_comparator": (
                    "matched_9216_input_with_all_phase_coordinates_zeroed"
                ),
                "vae_view_boundary": (
                    "fft_is_per_view_after_established_width_stacked_wan_encoding;"
                    "vae_itself_is_not_claimed_seam_independent"
                ),
                "phase_increment_mask_scale": (
                    "one_rms_shared_across_the_two_motion_tokens_per_sample_view_channel"
                ),
            },
            "action_target": {
                "source": "train_actions_only",
                "chunks": [4, 12],
                "descriptor": "within_chunk_first_differences",
                "raw_dim": 736,
                "fit_only_whitened_pca_dim": 16,
            },
            "training": {
                "fit_partition": "auxiliary_index_mod_8_nonzero",
                "fit_clips": 448,
                "calibration_partition": "auxiliary_index_mod_8_zero",
                "calibration_clips": 64,
                "validation_clips_read": 0,
                "fixed_updates": contract.EXPECTED_UPDATES,
                "batch_size": contract.EXPECTED_BATCH_SIZE,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "seed": contract.EXPECTED_SEED,
                "checkpoint_selection": "fixed_final_update_not_metric_selected",
                "feature_variants": ["full", "magnitude_only"],
                "paired_initialization": "identical_state_before_update_1",
            },
            "evaluation": {
                "sealed_validation_clips": 64,
                "controls": [
                    "aligned",
                    "episode_disjoint_cyclic_shuffled",
                    "zero",
                    "inverse",
                ],
                "bootstrap_replicates": contract.BOOTSTRAP_REPLICATES,
                "bootstrap_seed": contract.BOOTSTRAP_SEED,
                "familywise_alpha": 0.05,
                "family_cells": 8,
                "gate": (
                    "full_probe_beats_three_target_controls_and_matched_"
                    "magnitude_only_under_fixed_practical_effect_thresholds"
                ),
                "control_thresholds": {
                    "relative_mse_point": 0.05,
                    "relative_mse_lower_bound": 0.01,
                    "cosine_point": 0.05,
                    "cosine_lower_bound": 0.01,
                },
                "phase_contribution_thresholds": {
                    "relative_mse_point": 0.03,
                    "relative_mse_lower_bound": 0.01,
                    "cosine_point": 0.02,
                    "cosine_lower_bound": 0.005,
                },
            },
            "wandb": {
                **wandb,
                "entity": contract.EXPECTED_ENTITY,
                "project": contract.EXPECTED_PROJECT,
                "run_id": f"{args.study_id}-probe-u000400",
                "run_name": f"{args.study_id}-probe-u000400",
                "group": None,
                "mode": "online",
                "resume": "never",
                "private_project_acknowledged": True,
                "user_requested_email": "ldu@nvidia.edu",
                "authenticated_email_matches_user_request": viewer_email
                == "ldu@nvidia.edu",
                "identity_deviation_note": (
                    None
                    if viewer_email == "ldu@nvidia.edu"
                    else (
                        "authenticated private personal W&B entity is valid, but "
                        "its viewer email differs from ldu@nvidia.edu"
                    )
                ),
            },
            "planned_paths": planned,
            "protected_test_paths_accepted": False,
            "protected_test_clips_read": 0,
            "generator_changes": 0,
        }
    )
    output = root / "registration.json"
    contract.exclusive_json(output, payload)
    print(
        json.dumps(
            {
                "registration": str(output),
                "identity_sha256": payload["identity_sha256"],
                "file_sha256": contract.sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise contract.CSIPContractError("CSIP checkpoint must contain one mapping")
    return value


def command_seal(args: argparse.Namespace) -> int:
    registration = contract.validate_registration(
        args.registration, require_train_cache=True, open_validation=False
    )
    planned = registration["planned_paths"]
    if str(args.checkpoint.resolve()) != planned["checkpoint"]:
        raise contract.CSIPContractError("checkpoint path differs from registration")
    if str(args.training_report.resolve()) != planned["training_report"]:
        raise contract.CSIPContractError(
            "training report path differs from registration"
        )
    checkpoint_record = contract.file_record(args.checkpoint, "fixed CSIP checkpoint")
    report_record = contract.file_record(args.training_report, "CSIP training report")
    checkpoint = _load_checkpoint(Path(checkpoint_record["path"]))
    report = contract.read_json(report_record["path"], "CSIP training report")
    contract.verify_identity(report, "CSIP training report")
    train_latent_path = Path(planned["train_latent_metadata"])
    train_latent = contract.validate_latent_cache(
        train_latent_path, registration=registration, split="train"
    )
    contract.validate_checkpoint_payload(
        checkpoint,
        registration=registration,
        train_latent_cache_identity_sha256=train_latent["identity_sha256"],
    )
    if (
        report.get("schema_version") != contract.SCHEMA_VERSION
        or report.get("kind") != "csip_phase0_training_report"
        or report.get("status") != "fixed_update_400_complete"
        or report.get("registration_identity_sha256") != registration["identity_sha256"]
        or report.get("train_latent_cache_identity_sha256")
        != train_latent["identity_sha256"]
        or report.get("checkpoint_sha256") != checkpoint_record["sha256"]
        or report.get("completed_updates") != contract.EXPECTED_UPDATES
        or report.get("selection_rule") != "fixed_final_update_not_metric_selected"
        or report.get("feature_variants") != ["full", "magnitude_only"]
        or report.get("paired_initialization") != "identical_state_before_update_1"
        or report.get("wandb", {}).get("run_id") != registration["wandb"]["run_id"]
        or report.get("wandb", {}).get("group") is not None
        or report.get("validation_clips_read") != 0
        or report.get("protected_test_clips_read") != 0
    ):
        raise contract.CSIPContractError(
            "checkpoint is not the preregistered fixed endpoint"
        )
    registration_record = contract.registration_file_record(args.registration)
    seal = contract.with_identity(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": SEAL_KIND,
            "created_at_utc": contract.now_utc(),
            "status": "sealed_before_validation_open",
            "registration": registration_record,
            "registration_identity_sha256": registration["identity_sha256"],
            "checkpoint": checkpoint_record,
            "training_report": report_record,
            "fixed_update": contract.EXPECTED_UPDATES,
            "validation_clips_read_before_seal": 0,
            "protected_test_clips_read": 0,
        }
    )
    output = Path(planned["checkpoint_seal"])
    contract.exclusive_json(output, seal)
    print(
        json.dumps(
            {"seal": str(output), "identity_sha256": seal["identity_sha256"]},
            sort_keys=True,
        )
    )
    return 0


def validate_seal(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal_path = contract.regular_file(path, "CSIP checkpoint seal")
    seal = contract.read_json(seal_path, "CSIP checkpoint seal")
    contract.verify_identity(seal, "CSIP checkpoint seal")
    if (
        seal.get("schema_version") != contract.SCHEMA_VERSION
        or seal.get("kind") != SEAL_KIND
        or seal.get("status") != "sealed_before_validation_open"
        or seal.get("fixed_update") != contract.EXPECTED_UPDATES
        or seal.get("validation_clips_read_before_seal") != 0
        or seal.get("protected_test_clips_read") != 0
    ):
        raise contract.CSIPContractError("CSIP checkpoint seal differs")
    registration_path = seal.get("registration", {}).get("path")
    registration = contract.validate_registration(
        str(registration_path), require_train_cache=True, open_validation=False
    )
    if seal.get("registration_identity_sha256") != registration[
        "identity_sha256"
    ] or seal["registration"] != contract.registration_file_record(registration_path):
        raise contract.CSIPContractError("sealed registration changed")
    for key in ("checkpoint", "training_report"):
        contract.verify_file_record(seal[key], f"sealed {key}")
    return seal, registration


def command_render(args: argparse.Namespace) -> int:
    registration = contract.validate_registration(args.registration)
    root = Path(registration["study_root"])
    python = registration["runtime"]["python"]["launcher_path"]
    repo = registration["source"]["path"]
    registration_path = str(
        contract.regular_file(args.registration, "CSIP registration")
    )

    def q(parts: list[str]) -> str:
        return " ".join(shlex.quote(value) for value in parts)

    commands = {
        "extract_train": q(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(Path(repo) / "tools" / "csip_latent_cache.py"),
                "extract",
                "--registration",
                registration_path,
                "--split",
                "train",
            ]
        ),
        "extract_validation_after_checkpoint_seal": q(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(Path(repo) / "tools" / "csip_latent_cache.py"),
                "extract",
                "--registration",
                registration_path,
                "--split",
                "validation",
                "--seal",
                registration["planned_paths"]["checkpoint_seal"],
            ]
        ),
        "train": q(
            [
                python,
                str(Path(repo) / "tools" / "csip_train.py"),
                "--registration",
                registration_path,
            ]
        ),
        "seal": q(
            [
                python,
                str(Path(repo) / "tools" / "csip_workflow.py"),
                "seal",
                "--registration",
                registration_path,
                "--checkpoint",
                registration["planned_paths"]["checkpoint"],
                "--training-report",
                registration["planned_paths"]["training_report"],
            ]
        ),
        "evaluate": q(
            [
                python,
                str(Path(repo) / "tools" / "csip_evaluate.py"),
                "--seal",
                registration["planned_paths"]["checkpoint_seal"],
            ]
        ),
        "analyze": q(
            [
                python,
                str(Path(repo) / "tools" / "csip_analyze.py"),
                "--seal",
                registration["planned_paths"]["checkpoint_seal"],
                "--evaluation",
                registration["planned_paths"]["evaluation"],
            ]
        ),
    }
    print(
        json.dumps(
            {"study_root": str(root), "commands": commands}, indent=2, sort_keys=True
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wandb = subparsers.add_parser("wandb-check")
    wandb.add_argument("--entity", default=contract.EXPECTED_ENTITY)
    wandb.add_argument("--project", default=contract.EXPECTED_PROJECT)
    wandb.set_defaults(func=command_wandb_check)

    register = subparsers.add_parser("register")
    register.add_argument("--study-id", required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--tool-repo", type=Path, default=REPO_ROOT)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--validation-manifest", type=Path, required=True)
    register.add_argument("--validation-cache-metadata", type=Path, required=True)
    register.add_argument("--wandb-entity", default=contract.EXPECTED_ENTITY)
    register.add_argument("--wandb-project", default=contract.EXPECTED_PROJECT)
    register.add_argument("--ack-private-wandb-project", action="store_true")
    register.set_defaults(func=command_register)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--registration", type=Path, required=True)
    seal.add_argument("--checkpoint", type=Path, required=True)
    seal.add_argument("--training-report", type=Path, required=True)
    seal.set_defaults(func=command_seal)

    render = subparsers.add_parser("render")
    render.add_argument("--registration", type=Path, required=True)
    render.set_defaults(func=command_render)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
