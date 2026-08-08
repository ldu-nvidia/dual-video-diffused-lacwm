from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from tools.slurm import temporal_followup_doe as workflow


SBATCH = workflow.REPO_ROOT / "tools" / "slurm" / "temporal_followup_doe.sbatch"


def inputs(tmp_path: Path) -> workflow.WorkflowInputs:
    return workflow.WorkflowInputs(
        expected_commit="a" * 40,
        study_root=tmp_path / "study-v1",
        data_root=tmp_path / "data",
        semantic_cache_root=tmp_path / "cache",
        train_manifest=tmp_path / "train.jsonl",
        validation_manifest=tmp_path / "val.jsonl",
        normalization=tmp_path / "normalization.json",
    )


def input_records(tmp_path: Path) -> dict:
    return {
        "repository": {"root": str(workflow.REPO_ROOT), "commit": "a" * 40, "dirty": False},
        "data_root": str(tmp_path / "data"),
        "semantic_cache_root": str(tmp_path / "cache"),
        "train_manifest": {"path": str(tmp_path / "train.jsonl"), "sha256": workflow.TRAIN_MANIFEST_SHA256, "bytes": 1},
        "validation_manifest": {"path": str(tmp_path / "val.jsonl"), "sha256": workflow.VALIDATION_MANIFEST_SHA256, "bytes": 1},
        "normalization": {"path": str(tmp_path / "normalization.json"), "sha256": workflow.NORMALIZATION_SHA256, "bytes": 1},
        "train_cache_metadata": {"path": str(tmp_path / "train-metadata.json"), "sha256": workflow.TRAIN_CACHE_METADATA_FILE_SHA256, "bytes": 1},
        "validation_cache_metadata": {"path": str(tmp_path / "val-metadata.json"), "sha256": workflow.VALIDATION_CACHE_METADATA_FILE_SHA256, "bytes": 1},
        "train_target": {"path": str(tmp_path / "train.npy"), "sha256": workflow.TRAIN_TARGET_SHA256, "bytes": 1},
        "validation_target": {"path": str(tmp_path / "val.npy"), "sha256": workflow.VALIDATION_TARGET_SHA256, "bytes": 1},
        "pca_sha256": workflow.PCA_SHA256,
        "protected_test_accessed": False,
    }


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_sbatch_routes_workdir_and_logs_outside_checkout() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    directives = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in text.splitlines()
        if line.startswith("#SBATCH --") and "=" in line
    }
    for name in ("#SBATCH --chdir", "#SBATCH --output", "#SBATCH --error"):
        value = Path(directives[name].replace("%j", "123"))
        assert value.is_absolute()
        assert workflow.REPO_ROOT not in value.parents
        assert Path("/lustre") in (value, *value.parents)


def test_five_arm_commands_freeze_geometry_and_mechanisms(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    expected = {
        "ABS": ("absolute", "0.0", "0.0", False),
        "ABS-T": ("absolute", "1.0", "0.0", True),
        "DELTA": ("delta_pack", "0.0", "0.0", True),
        "DELTA-T": ("delta_pack", "1.0", "0.0", True),
        "DELTA-R": ("delta_pack", "0.0", "0.5", True),
    }
    for arm in workflow.ARMS:
        command = workflow.build_train_command(value, "primary", arm)
        target, temporal, rollin, normalized = expected[arm.name]
        assert command[:6] == [
            os.sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(workflow.TRAINER),
        ]
        assert option(command, "--target-mode") == target
        assert option(command, "--temporal-velocity-loss-weight") == temporal
        assert option(command, "--self-rollin-probability") == rollin
        assert ("--normalization" in command) is normalized
        assert option(command, "--global-batch-size") == "256"
        assert option(command, "--micro-batch-size") == "32"
        assert option(command, "--workers") == "4"
        assert option(command, "--seed") == "1234"
        assert option(command, "--wandb-entity") == "zijiandu"
        assert option(command, "--wandb-project") == "dual-video-diffusion-private"
        assert "--wandb-private-project-ack" in command


def test_plan_orders_all_calibrations_before_primary_and_optional_eval(tmp_path: Path) -> None:
    plan = workflow.build_plan(inputs(tmp_path), input_records(tmp_path))
    kinds = [step["kind"] for step in plan["steps"]]
    assert kinds == ["registration"] + ["calibration"] * 5 + ["primary"] * 5 + ["evaluation"] * 5 + ["analysis"]
    outputs = [step["output"] for step in plan["steps"]]
    assert len(outputs) == len(set(outputs))
    assert [step["kind"] for step in workflow.selected_steps(plan, "train")] == kinds[:11]
    assert [step["kind"] for step in workflow.selected_steps(plan, "evaluate")] == kinds[11:]
    assert workflow.render_plan(plan, "train")["mode"].startswith("dry_run")
    assert plan["protected_test_accessed"] is False


def test_sbatch_contract_is_eight_b200_short_and_never_submits() -> None:
    text = (workflow.REPO_ROOT / "tools" / "slurm" / "temporal_followup_doe.sbatch").read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=8",
        "#SBATCH --time=02:00:00",
        "#SBATCH --partition=batch",
        "#SBATCH --account=coreai_chef_posttrain",
        "#SBATCH --qos=short",
        "#SBATCH --exclude=pool0-0081,pool0-0089",
        "#SBATCH --no-requeue",
    ):
        assert directive in text
    assert "sbatch " not in text


def test_evaluation_commands_bind_both_receipts_and_never_name_test(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    for arm in workflow.ARMS:
        command = workflow.build_evaluation_command(value, arm)
        assert option(command, "--checkpoint").endswith("update_005000.pt")
        assert option(command, "--calibration-checkpoint").endswith("update_000200.pt")
        assert option(command, "--implementation-registration").endswith("implementation_registration.json")
        assert option(command, "--manifest") == str(value.validation_manifest)
        assert "--test-manifest" not in command
        assert "--protected-test-manifest" not in command


def test_sanitized_environment_drops_group_and_unrelated_secrets(tmp_path: Path) -> None:
    source = {
        "PATH": "/bin",
        "SLURM_JOB_ID": "42",
        "WANDB_API_KEY": "secret-required-at-runtime",
        "WANDB_RUN_GROUP": "accidental-group",
        "WANDB_GROUP": "also-bad",
        "UNRELATED_SECRET": "must-not-pass",
        "PYTHONPATH": "/untrusted",
    }
    result = workflow.sanitized_environment(tmp_path, source)
    assert result["WANDB_API_KEY"] == source["WANDB_API_KEY"]
    assert result["WANDB_ENTITY"] == "zijiandu"
    assert result["WANDB_PROJECT"] == "dual-video-diffusion-private"
    assert result["PYTHONPATH"] == str(workflow.REPO_ROOT)
    assert "WANDB_RUN_GROUP" not in result
    assert "WANDB_GROUP" not in result
    assert "UNRELATED_SECRET" not in result


def test_cache_and_normalization_contracts_fail_closed() -> None:
    cache = {
        "schema": "droid-causal-vjepa2.1-v1",
        "complete": True,
        "split": "val",
        "clip_count": 890,
        "manifest_sha256": workflow.VALIDATION_MANIFEST_SHA256,
        "train_manifest_sha256": workflow.TRAIN_MANIFEST_SHA256,
        "target_file": "targets.fp16.npy",
        "target_sha256": workflow.VALIDATION_TARGET_SHA256,
        "target_shape": [890, *workflow.TARGET_SHAPE],
        "auxiliary_target_shape": workflow.TARGET_SHAPE,
        "target_dtype": "float16",
        "pca_sha256": workflow.PCA_SHA256,
        "allowed_splits": ["train", "val"],
        "protected_test_access": False,
        "test_rows_extracted": 0,
        "world_size": 8,
    }
    workflow._validate_cache_metadata(
        cache,
        split="val",
        clips=890,
        manifest_sha256=workflow.VALIDATION_MANIFEST_SHA256,
        target_sha256=workflow.VALIDATION_TARGET_SHA256,
    )
    changed = copy.deepcopy(cache)
    changed["protected_test_access"] = True
    with pytest.raises(workflow.WorkflowError):
        workflow._validate_cache_metadata(
            changed,
            split="val",
            clips=890,
            manifest_sha256=workflow.VALIDATION_MANIFEST_SHA256,
            target_sha256=workflow.VALIDATION_TARGET_SHA256,
        )

    normalization = {
        "schema": "causal-vjepa2-temporal-normalization-v1",
        "status": "complete",
        "complete": True,
        "split": "train",
        "clips": 64_000,
        "train_manifest_sha256": workflow.TRAIN_MANIFEST_SHA256,
        "pca_sha256": workflow.PCA_SHA256,
        "cache_metadata_sha256": workflow.TRAIN_CACHE_METADATA_IDENTITY_SHA256,
        "target_shape": workflow.TARGET_SHAPE,
        "source_storage_dtype": "float16",
        "encode_decode_dtype": "float32",
        "declared_roundtrip_max_abs_tolerance": 2e-5,
        "protected_test_accessed": False,
    }
    workflow._validate_normalization(normalization)
    normalization["split"] = "val"
    with pytest.raises(workflow.WorkflowError):
        workflow._validate_normalization(normalization)


def test_arm_receipt_contract_rejects_mechanism_drift() -> None:
    arm = workflow.ARM_BY_NAME["DELTA-R"]
    config = {
        "doe_arm": "DELTA-R",
        "target_mode": "delta_pack",
        "normalization": {"sha256": workflow.NORMALIZATION_SHA256},
        "loss": {
            "flow_weight": 1.0,
            "normalized_temporal_velocity_weight": 0.0,
            "action_shuffle_margin_weight": 0.0,
        },
        "self_rollin": {
            "probability": 0.5,
            "later_time_rule": "sampled_final_clock_fraction",
        },
    }
    assert workflow._arm_config_valid(config, arm)
    config["self_rollin"]["probability"] = 0.25
    assert not workflow._arm_config_valid(config, arm)


def test_normalized_training_receipt_binds_augmented_record_and_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = inputs(tmp_path)
    value.train_manifest.write_text("train\n", encoding="utf-8")
    value.validation_manifest.write_text("val\n", encoding="utf-8")
    value.normalization.write_text('{"normalization":"payload"}\n', encoding="utf-8")
    arm = workflow.ARM_BY_NAME["DELTA"]
    augmented_normalization = workflow.normalization_file_record(value.normalization)
    monkeypatch.setattr(
        workflow, "NORMALIZATION_SHA256", augmented_normalization["sha256"]
    )
    assert augmented_normalization["payload_sha256"] == workflow.sha256_json(
        {"normalization": "payload"}
    )
    run_dir = workflow.training_run_dir(value.study_root, "calibration", arm)
    checkpoint = run_dir / "checkpoints" / "update_000200.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "schema": workflow.TRAINING_SCHEMA,
        "source": {"commit": value.expected_commit, "dirty": False},
        "entrypoint": workflow.file_record(workflow.TRAINER),
        "command": "train",
        "run_role": "numerical_calibration",
        "doe_arm": "DELTA",
        "initialization": "from_scratch_deterministic_no_pretrained_weights",
        "target_mode": "delta_pack",
        "normalization": augmented_normalization,
        "target_shape": workflow.TARGET_SHAPE,
        "updates": 200,
        "checkpoint_updates": [200],
        "seed": 1234,
        "global_batch_size": 256,
        "world_size": 8,
        "local_optimizer_batch_size": 32,
        "micro_batch_size_per_rank": 32,
        "gradient_accumulation_steps": 1,
        "workers_per_rank": 4,
        "dtype": "bfloat16",
        "parameter_count": workflow.MODEL_PARAMETERS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 5e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_updates": 500,
            "after_warmup": "constant",
            "gradient_clip_norm": 1.0,
        },
        "ema": {
            "decay": 0.9999,
            "schedule": workflow.EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        },
        "loss": {
            "flow_weight": 1.0,
            "normalized_temporal_velocity_weight": 0.0,
            "action_shuffle_margin_weight": 0.0,
        },
        "self_rollin": {
            "probability": 0.0,
            "later_time_rule": "sampled_final_clock_fraction",
        },
        "datasets": {
            "train": workflow.file_record(value.train_manifest),
            "validation": workflow.file_record(value.validation_manifest),
            "semantic_cache": {
                "train": {
                    "target_sha256": workflow.TRAIN_TARGET_SHA256,
                    "pca_sha256": workflow.PCA_SHA256,
                },
                "validation": {
                    "target_sha256": workflow.VALIDATION_TARGET_SHA256,
                    "pca_sha256": workflow.PCA_SHA256,
                },
            },
        },
        "wandb": {
            "enabled": True,
            "entity": workflow.WANDB_ENTITY,
            "project": workflow.WANDB_PROJECT,
            "group": None,
            "private_project_acknowledged": True,
        },
        "video_loss_enabled": False,
        "future_rgb_model_input": False,
        "teacher_model_calls": 0,
        "protected_test_accessed": False,
    }
    config["experiment_identity_sha256"] = workflow.sha256_json(
        {key: item for key, item in config.items() if key != "checkpoint_updates"}
    )
    config_path = run_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    complete = {
        "schema": workflow.TRAINING_SCHEMA,
        "status": "complete",
        "completed_updates": 200,
        "nonfinite_updates": 0,
        "target_mode": "delta_pack",
        "video_loss_enabled": False,
        "future_rgb_model_input": False,
        "teacher_model_calls": 0,
        "protected_test_accessed": False,
        "resolved_config_sha256": workflow.sha256_json(config),
        "experiment_identity_sha256": config["experiment_identity_sha256"],
        "resolved_config": workflow.file_record(config_path),
        "checkpoint": workflow.file_record(checkpoint),
    }
    (run_dir / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    receipt = workflow.validate_training_receipt(
        run_dir,
        kind="calibration",
        arm=arm,
        inputs=value,
    )
    assert receipt["updates"] == 200
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(workflow.WorkflowError):
        workflow.validate_training_receipt(
            run_dir,
            kind="calibration",
            arm=arm,
            inputs=value,
        )


def test_training_set_routes_every_receipt_through_inputs_study_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = inputs(tmp_path)
    calls: list[tuple[Path, str, str]] = []

    def fake_receipt(
        run_dir: Path,
        *,
        kind: str,
        arm: workflow.Arm,
        inputs: workflow.WorkflowInputs,
    ) -> dict:
        assert inputs is value
        calls.append((run_dir, kind, arm.name))
        return {
            "kind": kind,
            "arm": arm.name,
            "config": {
                "loss": {"normalized_temporal_velocity_weight": arm.temporal_weight},
                "self_rollin": {"probability": arm.rollin_probability},
            },
        }

    monkeypatch.setattr(workflow, "validate_training_receipt", fake_receipt)
    receipt = workflow.validate_training_set(value)
    assert receipt["status"] == "complete"
    assert len(calls) == 10
    for run_dir, kind, arm_name in calls:
        assert value.study_root in run_dir.parents
        assert run_dir == workflow.training_run_dir(
            value.study_root, kind, workflow.ARM_BY_NAME[arm_name]
        )
