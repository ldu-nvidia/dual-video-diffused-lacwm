"""Regression tests for the adversarially hardened action-variation screen."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from robot_wm.utils.action_variation_trainer import _load_parent_state_exact
from tools import action_variation_evaluate as evaluation
from tools import action_variation_screen as screen
from tools import vpm_phaselock_probe as phase


def test_registration_recomputes_stats_instead_of_trusting_source_label(tmp_path):
    actions = np.zeros((512, 13, 5, 23), dtype=np.float32)
    actions[..., 0] = np.arange(5, dtype=np.float32).reshape(1, 1, 5) * 2.0
    actions[..., 1] = np.arange(5, dtype=np.float32).reshape(1, 1, 5) * np.arange(
        1, 513, dtype=np.float32
    ).reshape(512, 1, 1)
    actions_path = tmp_path / "actions.npy"
    np.save(actions_path, actions, allow_pickle=False)
    train = {
        "manifest": {"path": "/train.jsonl", "sha256": "m", "bytes": 1},
        "cache_metadata": {"path": "/metadata.json", "sha256": "d", "bytes": 1},
        "actions": {
            "path": str(actions_path),
            "sha256": "a",
            "bytes": actions_path.stat().st_size,
        },
    }
    fitted = screen._computed_action_delta_stats(actions_path)
    stats = {
        **fitted,
        "source": {
            "manifest": train["manifest"],
            "cache_metadata": train["cache_metadata"],
            "actions": train["actions"],
            "rgb_opened": False,
            "auxiliary_target_opened": False,
        },
    }
    screen._validate_stats_against_train(stats, train)
    forged = dict(stats)
    forged["mean"] = list(stats["mean"])
    forged["mean"][0] += 0.25
    with pytest.raises(screen.ActionVariationScreenError, match="recomputed"):
        screen._validate_stats_against_train(forged, train)


def _minimal_config() -> dict:
    dataset = {
        "datasets": {
            "ABC": {"clip_manifest": "/train", "cache_metadata": "/train-meta"}
        }
    }
    validation = {
        "datasets": {"ABC": {"clip_manifest": "/val", "cache_metadata": "/val-meta"}}
    }
    return {
        "name": "arm",
        "seed": 1234,
        "debug": False,
        "dataset": dataset,
        "val_dataset": validation,
        "viz_dataset": validation,
        "data_loader": {"batch_size": 1},
        "val_data_loader": [{"batch_size": 2}],
        "viz_data_loader": [],
        "model": {
            "action_variation": {"stats_path": "/stats"},
            "forward_model": {"lora_alpha": 128, "lora_dropout": 0.05},
        },
        "optimizer_factory": {"lr": 1e-4},
        "lr_scheduler_factory": {"lr_lambda": {"total_steps": 200}},
        "trainer": {
            "config": {
                "load_path": "/parent",
                "saving": {"save_path": "/run/snapshot.pt"},
            }
        },
        "wandb": {"enabled": True, "id": "f" * 64, "resume": "never"},
    }


def test_canonical_config_contract_catches_compute_and_wandb_changes():
    baseline = _minimal_config()
    expected = screen.canonical_config_contract(baseline)
    changed_compute = _minimal_config()
    changed_compute["model"]["forward_model"]["lora_dropout"] = 0.0
    assert screen.canonical_config_contract(changed_compute) != expected
    changed_wandb = _minimal_config()
    changed_wandb["wandb"]["enabled"] = False
    assert screen.canonical_config_contract(changed_wandb) != expected


class _ParentPlusResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parent = nn.Linear(2, 2)
        self.action_delta_residual = nn.Linear(2, 2)


def test_parent_load_misses_exactly_residual_and_rejects_any_other_missing_key():
    source = _ParentPlusResidual()
    target = _ParentPlusResidual()
    parent = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("action_delta_residual.")
    }
    new_keys = {
        key for key in target.state_dict() if key.startswith("action_delta_residual.")
    }
    _load_parent_state_exact(target, parent, new_keys)
    assert torch.equal(target.parent.weight, source.parent.weight)
    incomplete = dict(parent)
    incomplete.pop("parent.bias")
    with pytest.raises(RuntimeError, match="miss exactly"):
        _load_parent_state_exact(target, incomplete, new_keys)


def test_guarded_launcher_sets_registered_pythonpath_and_batch_one():
    root = Path(__file__).resolve().parents[2]
    sbatch = (root / "tools/slurm/action_variation.sbatch").read_text()
    submit = (root / "tools/slurm/submit_action_variation.sh").read_text()
    workflow = (root / "tools/slurm/action_variation_workflow.py").read_text()
    entrypoint = (
        root / "projects/latent_action_models/train_action_variation.py"
    ).read_text()
    trainer = (root / "robot_wm/utils/action_variation_trainer.py").read_text()
    evaluator = (root / "tools/action_variation_evaluate.py").read_text()
    analyzer = (root / "tools/analyze_action_variation.py").read_text()
    assert "--expected-gpus 8 --require-b200" in sbatch
    assert 'export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:' in sbatch
    assert "--batch-size-per-rank 1" in sbatch
    assert "wandb.group=null" in sbatch
    assert '"$WORKFLOW" seal-wandb' in sbatch
    assert '"$WORKFLOW" write-runtime-receipt' in sbatch
    assert "wandb_run_complete.json" in sbatch
    assert "rm -rf" not in sbatch
    assert "Dry-run is the default" in submit
    assert "--gpus-per-node 8" in submit
    assert "--gpus-per-node 0" not in submit
    assert "--gpus-per-node 1" in submit
    assert '--dependency "afterok:$ARM_JOB_ID"' in submit
    excluded = "pool0-0081,pool0-0089,pool0-0200,pool0-0343"
    assert f"#SBATCH --exclude={excluded}" in sbatch
    assert f'EXCLUDE_NODES="{excluded}"' in submit
    assert "--exclude)" in submit
    assert sbatch.index('"$VERIFY_RUNTIME"') < sbatch.index(
        '"$WORKFLOW" write-runtime-receipt'
    ) < sbatch.index('"$WORKFLOW" write-arm-plan') < sbatch.index(
        '"$PYTHON_BIN" -m torch.distributed.run'
    )
    assert entrypoint.index("_validate_launch_contract(cfg)") < entrypoint.index(
        "dist.init_process_group()"
    )
    assert "_sha256(load_path)" not in trainer
    assert "broadcast_object_list" not in trainer
    assert "_distributed_file_record(snapshot_path)" not in evaluator
    assert 'trained_snapshot = base._file_record(run_dir / "snapshot.pt")' in workflow
    assert '"runtime_verification": runtime_verification' in workflow
    assert (
        '"runtime_verification_receipt": self._runtime_verification_receipt'
        in trainer
    )
    assert '"runtime_verification_receipt": runtime_verification' in evaluator
    assert '"runtime_verification_receipts": {' in analyzer


def test_registration_preserves_virtualenv_launcher_and_binds_real_binary(tmp_path):
    real_python = tmp_path / "python-real"
    real_python.write_bytes(b"#!/bin/sh\n")
    real_python.chmod(0o700)
    launcher = tmp_path / "python"
    launcher.symlink_to(real_python)

    # This is the contract used by command_register: keep the launcher for
    # sys.executable/pyvenv.cfg while hashing its resolved executable target.
    assert launcher.is_file()
    assert launcher.resolve(strict=True) == real_python
    record = screen.base._file_record(launcher.resolve(strict=True))
    runtime = {"python": str(launcher), "python_file": record}
    assert Path(runtime["python"]).resolve(strict=True) == Path(
        runtime["python_file"]["path"]
    )


def _runtime_registration(tmp_path: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    record = {"path": "/registered", "bytes": 1, "sha256": "d" * 64}
    return {
        "identity_sha256": "a" * 64,
        "output_root": str(tmp_path),
        "tool_repository": {"path": str(root), "git_commit": "b" * 40},
        "controlled_study": {"parent_snapshot": dict(record)},
        "training": {
            key: dict(record)
            for key in ("manifest", "cache_metadata", "rgb", "actions")
        },
        "validation": {
            "manifest": dict(record),
            "cache_metadata": dict(record),
            "arrays": {"rgb": dict(record), "actions": dict(record)},
        },
        "action_delta_stats": {"file": {**record, "sha256": "c" * 64}},
        "runtime": {
            "python": "/registered/python",
            "videox_commit": "e" * 40,
            "wan_dir": "/registered/wan",
        },
    }


def _write_runtime_receipt(registration: dict, arm: screen.Arm):
    root = Path(__file__).resolve().parents[2]
    verifier = {
        "python": "3.10.20",
        "environment": {"sys_executable": registration["runtime"]["python"]},
        "videox_commit": registration["runtime"]["videox_commit"],
        "videox_status": "clean",
        "weights": {"root": registration["runtime"]["wan_dir"]},
        "gpus": {
            "count": 8,
            "devices": [
                {"index": index, "name": "NVIDIA B200", "capability": [10, 0]}
                for index in range(8)
            ],
            "nccl_available": True,
            "inaccessible_peer_pairs": [],
        },
    }
    core = {
        "schema_version": 1,
        "kind": screen.KIND_RUNTIME_RECEIPT,
        "status": screen.RUNTIME_RECEIPT_STATUS,
        "created_at_utc": "2026-08-08T00:00:00+00:00",
        "registration_identity_sha256": registration["identity_sha256"],
        "tool_git_commit": registration["tool_repository"]["git_commit"],
        "arm": asdict(arm),
        "run_identity_sha256": screen.arm_run_identity(registration, arm),
        "verifier_implementation": evaluation.base._file_record(
            root / "tools/env/verify_b200_runtime.py"
        ),
        "verifier_output": verifier,
        "verifier_identity_sha256": screen.runtime_verifier_identity(verifier),
        "slurm": {
            "job_id": "1001",
            "array_job_id": "1000",
            "array_task_id": list(screen.ARM_BY_CODE).index(arm.code),
            "restart_count": 0,
            "node_list": "pool0-0001",
        },
        "protected_test_accessed": False,
    }
    path = screen.runtime_receipt_path(registration, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(screen.identity_payload(core)))
    return path


def test_registered_input_and_runtime_receipts_are_exact(tmp_path, monkeypatch):
    registration = _runtime_registration(tmp_path)
    expected = screen.registered_input_records(registration)
    assert (
        screen.validate_registered_input_revalidation(expected, registration)
        == expected
    )
    forged = dict(expected)
    forged["parent_snapshot"] = {**expected["parent_snapshot"], "sha256": "0" * 64}
    with pytest.raises(screen.ActionVariationScreenError, match="revalidation differs"):
        screen.validate_registered_input_revalidation(forged, registration)

    arm = screen.ARMS[0]
    receipt_path = _write_runtime_receipt(registration, arm)
    binding = screen.validate_runtime_receipt(registration, arm)
    assert binding["record"]["path"] == str(receipt_path)
    assert binding["verifier_identity_sha256"]
    for key, value in {
        "SLURM_JOB_ID": "1001",
        "SLURM_ARRAY_JOB_ID": "1000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_RESTART_COUNT": "0",
        "SLURM_JOB_NODELIST": "pool0-0001",
    }.items():
        monkeypatch.setenv(key, value)
    assert screen.validate_runtime_receipt(
        registration, arm, require_current_slurm=True
    ) == binding
    monkeypatch.setenv("SLURM_JOB_ID", "9999")
    with pytest.raises(screen.ActionVariationScreenError, match="different Slurm"):
        screen.validate_runtime_receipt(
            registration, arm, require_current_slurm=True
        )


def test_arm_plan_uses_the_complete_canonical_endpoint_schema(tmp_path):
    registration = _runtime_registration(tmp_path)
    arm = screen.ARMS[0]
    _write_runtime_receipt(registration, arm)
    runtime_binding = screen.validate_runtime_receipt(registration, arm)
    run_dir = tmp_path / "training" / arm.run_name
    resolved = screen.identity_payload({"kind": "test_config"})
    plan_core = {
        "schema_version": 1,
        "kind": "action_variation_arm_execution_plan",
        "status": "planned_before_arm_training_or_metrics",
        "registration_identity_sha256": registration["identity_sha256"],
        "arm": asdict(arm),
        "run_identity_sha256": screen.arm_run_identity(registration, arm),
        "resolved_config_contract": resolved,
        "paths": {"run_dir": str(run_dir)},
        "input_revalidation": screen.registered_input_records(registration),
        "runtime_verification": runtime_binding,
        "training": {
            "updates": 200,
            "wan_calls_per_update": 1,
            "same_schema_and_forward_compute": True,
        },
        "evaluation": {
            "batch_size_per_rank": 1,
            "endpoints": screen.endpoint_records(),
            "protected_test_accessed": False,
            "future_or_clean_feature_used_at_sampling": False,
        },
        "wandb": {
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "group": None,
            "mode": "online",
            "id": screen.arm_run_identity(registration, arm),
            "resume": "never",
        },
    }
    plan_path = tmp_path / "arm_plans" / f"{arm.code.lower()}.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(screen.identity_payload(plan_core)))
    observed = evaluation._validate_arm_plan(registration, arm, run_dir)
    assert observed["runtime_verification"] == runtime_binding
    assert all("residual_mode" in endpoint for endpoint in screen.endpoint_records())

    forged = dict(plan_core)
    forged["evaluation"] = dict(plan_core["evaluation"])
    forged["evaluation"]["endpoints"] = [
        {key: value for key, value in endpoint.items() if key != "residual_mode"}
        for endpoint in screen.endpoint_records()
    ]
    plan_path.write_text(json.dumps(screen.identity_payload(forged)))
    with pytest.raises(evaluation.ActionVariationEvaluationError, match="plan differs"):
        evaluation._validate_arm_plan(registration, arm, run_dir)


def test_shuffled_donor_action_loader_never_reads_rgb():
    dataset = object.__new__(phase._RegisteredValidationInputs)
    dataset._actions = np.zeros((1, 13, 5, 23), dtype=np.float32)
    dataset._descriptors = ({"clip_id": "x"},)
    dataset._padding_dim = 157

    class _ForbiddenRGB:
        def __getitem__(self, _index):
            raise AssertionError("RGB cache was read for an action donor")

    dataset._rgb = _ForbiddenRGB()
    action = dataset.action_only(0)
    assert action.shape == (13, 5, 157)


def test_wandb_completion_receipt_is_semantically_bound(tmp_path):
    arm = screen.ARMS[0]
    registration = {
        "identity_sha256": "a" * 64,
        "tool_repository": {"git_commit": "b" * 40},
        "action_delta_stats": {"file": {"sha256": "c" * 64}},
    }
    run_identity = screen.arm_run_identity(registration, arm)
    completion_path = tmp_path / "training_complete.json"
    completion_path.write_text('{"status":"completed"}\n')
    snapshot_path = tmp_path / "snapshot.pt"
    snapshot_path.write_bytes(b"trained snapshot")
    core = {
        "schema_version": 1,
        "kind": "action_variation_wandb_run_complete",
        "registration_identity_sha256": registration["identity_sha256"],
        "arm": {
            "code": arm.code,
            "run_name": arm.run_name,
            "residual_enabled": arm.residual_enabled,
        },
        "run_identity_sha256": run_identity,
        "training_completion": evaluation.base._file_record(completion_path),
        "trained_snapshot": evaluation.base._file_record(snapshot_path),
        "wandb": {
            "id": run_identity,
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "group": None,
            "state": "finished",
            "run_identity_summary": run_identity,
            "url": (
                "https://wandb.ai/zijiandu/dual-video-diffusion-private/runs/"
                + run_identity
            ),
        },
    }
    receipt_path = tmp_path / "wandb_run_complete.json"
    receipt_path.write_text(json.dumps(screen.identity_payload(core)))
    observed = evaluation._validate_wandb_completion_receipt(
        registration, arm, tmp_path
    )
    assert observed["wandb"]["id"] == run_identity
    assert evaluation._validate_wandb_completion_receipt(
        registration, arm, tmp_path, rehash_snapshot=True
    )["trained_snapshot"] == core["trained_snapshot"]
    snapshot_path.write_bytes(b"altered!snapshot")
    with pytest.raises(
        evaluation.ActionVariationEvaluationError,
        match="W&B completion receipt differs",
    ):
        evaluation._validate_wandb_completion_receipt(
            registration, arm, tmp_path, rehash_snapshot=True
        )
    snapshot_path.write_bytes(b"trained snapshot")

    forged = dict(core)
    forged["wandb"] = {**core["wandb"], "group": "hidden-group"}
    receipt_path.write_text(json.dumps(screen.identity_payload(forged)))
    with pytest.raises(
        evaluation.ActionVariationEvaluationError,
        match="W&B completion receipt differs",
    ):
        evaluation._validate_wandb_completion_receipt(registration, arm, tmp_path)
