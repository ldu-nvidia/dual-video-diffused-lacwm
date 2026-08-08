"""Regression tests for the adversarially hardened action-variation screen."""

from __future__ import annotations

import json
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
    assert "--expected-gpus 8 --require-b200" in sbatch
    assert 'export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:' in sbatch
    assert "--batch-size-per-rank 1" in sbatch
    assert "wandb.group=null" in sbatch
    assert '"$WORKFLOW" seal-wandb' in sbatch
    assert "wandb_run_complete.json" in sbatch
    assert "rm -rf" not in sbatch
    assert "Dry-run is the default" in submit
    assert "--gpus-per-node 8" in submit
    assert "--gpus-per-node 0" not in submit
    assert '--dependency "afterok:$ARM_JOB_ID"' in submit


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

    forged = dict(core)
    forged["wandb"] = {**core["wandb"], "group": "hidden-group"}
    receipt_path.write_text(json.dumps(screen.identity_payload(forged)))
    with pytest.raises(
        evaluation.ActionVariationEvaluationError,
        match="W&B completion receipt differs",
    ):
        evaluation._validate_wandb_completion_receipt(registration, arm, tmp_path)
