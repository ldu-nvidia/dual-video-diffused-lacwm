"""Causality, lineage, configuration, and launcher contracts for IPQ-TC1."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tools import invertible_two_clock_evaluate as evaluation
from tools import invertible_two_clock_parent_parity as parent_parity
from tools import invertible_two_clock_pilot as pilot


class _FakeMMap:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _GuardedArray:
    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.operations = []
        self._mmap = _FakeMMap()

    def __getitem__(self, key):
        self.operations.append(key)
        row, frames = key
        assert isinstance(row, int)
        assert isinstance(frames, slice)
        start = 0 if frames.start is None else frames.start
        stop = self.shape[1] if frames.stop is None else frames.stop
        return np.zeros((stop - start, *self.shape[2:]), dtype=self.dtype)


def _registration() -> dict:
    return {
        "validation": {
            "arrays": {
                "rgb": {
                    "path": "/sealed/validation-rgb.npy",
                    "bytes": 123,
                    "sha256": "a" * 64,
                },
                "actions": {
                    "path": "/sealed/validation-actions.npy",
                    "bytes": 456,
                    "sha256": "b" * 64,
                },
            }
        }
    }


def test_prebarrier_reader_can_only_serve_history_and_planned_actions(monkeypatch):
    rgb = _GuardedArray((64, 13, 3, 180, 960), np.float16)
    actions = _GuardedArray((64, 13, 5, 23), np.float32)

    def fake_load(path, *, mmap_mode, allow_pickle):
        assert mmap_mode == "r" and allow_pickle is False
        return rgb if "rgb" in str(path) else actions

    monkeypatch.setattr(np, "load", fake_load)
    reader = evaluation.EndpointServingReader(
        registration=_registration(), rank=0
    )
    assert "__getitem__" not in type(reader).__dict__
    sample = reader.read_history_and_actions(7)
    assert tuple(sample["history_rgb"].shape) == (5, 3, 180, 960)
    assert tuple(sample["actions"].shape) == (13, 5, 157)
    assert rgb.operations == [(7, slice(0, 5, None))]
    assert actions.operations == [(7, slice(0, 13, None))]
    assert reader.ledger[0]["future_rgb_elements_read"] == 0
    assert reader.ledger[0]["slice"][0] == "0:5"
    with pytest.raises(IndexError):
        reader.read_history_and_actions(64)
    reader.close()
    assert rgb._mmap.closed and actions._mmap.closed


def test_full_rgb_reader_is_inconstructible_without_sealed_endpoint_barrier(
    monkeypatch,
):
    rgb = _GuardedArray((64, 13, 3, 180, 960), np.float16)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: rgb)
    invalid = evaluation.EndpointBarrierReceipt(
        rank=0,
        assigned_clip_indexes=tuple(range(8)),
        expected_endpoint_keys=12,
        materialized_endpoint_keys=11,
        endpoint_tensor_hashes_closed=11,
        event=20,
    )
    with pytest.raises(evaluation.IPQEvaluationError):
        evaluation.PostBarrierScoringReader(
            registration=_registration(),
            barrier=invalid,
            global_barrier_event=21,
            rank=0,
        )
    assert rgb.operations == []

    barrier = evaluation.EndpointBarrierReceipt(
        rank=0,
        assigned_clip_indexes=tuple(range(8)),
        expected_endpoint_keys=12,
        materialized_endpoint_keys=12,
        endpoint_tensor_hashes_closed=12,
        event=20,
    )
    reader = evaluation.PostBarrierScoringReader(
        registration=_registration(),
        barrier=barrier,
        global_barrier_event=21,
        rank=0,
    )
    with pytest.raises(evaluation.IPQEvaluationError):
        reader.read_full_rgb(7, event=21)
    with pytest.raises(evaluation.IPQEvaluationError):
        reader.read_full_rgb(9, event=22)
    value = reader.read_full_rgb(7, event=22)
    assert tuple(value.shape) == (13, 3, 180, 960)
    assert rgb.operations == [(7, slice(0, 13, None))]
    assert reader.ledger[0]["phase"] == "post_global_endpoint_barrier"
    reader.close()


def test_evaluator_source_has_no_full_rgb_or_content_rehash_before_barrier():
    source = inspect.getsource(evaluation.command_evaluate)
    prebarrier, postbarrier = source.split(
        "# This is the target-access boundary", maxsplit=1
    )
    assert "pilot.sha256" not in prebarrier
    assert "_scoring_targets" not in prebarrier
    assert "read_full_rgb" not in prebarrier
    assert "_RegisteredValidationInputs" not in source
    assert "pilot.sha256" in postbarrier
    assert "_scoring_targets" in postbarrier
    assert "read_full_rgb" in postbarrier


def test_lineage_selects_de65_and_explicitly_rejects_f67():
    assert pilot.PARENT_SNAPSHOT_SHA256.startswith("de65e832")
    assert pilot.LEGACY_REJECTED_SNAPSHOT_SHA256.startswith("f67c7bae")
    assert pilot.PARENT_SNAPSHOT_SHA256 != pilot.LEGACY_REJECTED_SNAPSHOT_SHA256
    assert pilot.PARENT_CANONICAL_MODEL_STATE_SHA256.startswith("d1231b8b")
    assert pilot.PARENT_RESOLVED_CONFIG_SHA256.startswith("ae3ffd27")


def test_matched_arm_configs_differ_only_in_declared_arm_fields():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    sync = OmegaConf.to_container(
        OmegaConf.load(
            root
            / "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/invertible_two_clock_sync.yaml"
        ),
        resolve=False,
    )
    independent = OmegaConf.to_container(
        OmegaConf.load(
            root
            / "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/invertible_two_clock_independent.yaml"
        ),
        resolve=False,
    )
    assert isinstance(sync, dict) and isinstance(independent, dict)
    assert sync["defaults"] == independent["defaults"]
    assert sync["model"]["invertible_two_clock"]["arm_mode"] == "synchronous"
    assert (
        independent["model"]["invertible_two_clock"]["arm_mode"]
        == "independent"
    )
    for value in (sync, independent):
        value.pop("name")
        value["model"]["invertible_two_clock"].pop("arm_mode")
        value["wandb"].pop("tags")
    assert sync == independent


def test_prepared_slurm_files_never_submit_other_jobs():
    root = Path(__file__).resolve().parents[2]
    sbatch = root / "tools/slurm/invertible_two_clock.sbatch"
    workflow = root / "tools/slurm/invertible_two_clock_workflow.py"
    assert sbatch.is_file() and workflow.is_file()
    assert "sbatch " not in sbatch.read_text(encoding="utf-8")
    assert "subprocess" not in workflow.read_text(encoding="utf-8")


def test_fixed_logging_contract_emits_exactly_400_paired_trace_events(tmp_path):
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    common = OmegaConf.load(
        root
        / "projects/latent_action_models/configs/experiments_0908/invertible_two_clock_common.yaml"
    )
    assert int(common.trainer.config.max_iter) == 400
    assert int(common.trainer.config.logging.log_every) == 1
    arm = pilot.ARM_BY_CODE["IPQ-SYNC"]
    header = {
        "kind": "ipq_tc1_training_trace_header",
        "protocol_version": "ipq-tc1-v1",
        "arm": "IPQ-SYNC",
        "arm_mode": "synchronous",
        "updates": 400,
        "wan_calls_per_update": 1,
        "global_batch_size": 8,
        "parent_snapshot_sha256": pilot.PARENT_SNAPSHOT_SHA256,
        "parent_run_identity_sha256": pilot.PARENT_RUN_IDENTITY_SHA256,
        "parent_canonical_model_state_sha256": (
            pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
        ),
        "parent_resolved_config_sha256": pilot.PARENT_RESOLVED_CONFIG_SHA256,
        "parent_training_source_commit": pilot.PARENT_TRAINING_SOURCE_COMMIT,
        "endpoint_split_opened": False,
        "validation_batches": 0,
        "model_parameter_schema": {"parameter_count": 17},
        "optimizer_parameter_count": 17,
    }
    rows = [header]
    rows.extend(
        {
            "kind": "ipq_tc1_training_trace_event",
            "iteration": index,
            "metrics": {
                "train_loss/paired_audit/exact_clip_index_all_ranks_sha256": (
                    f"{index:064x}"
                )
            },
        }
        for index in range(400)
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace = run_dir / "ipq_training_trace.jsonl"
    trace.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )
    _loaded_header, events = evaluation._load_trace(run_dir, arm)
    assert len(events) == 400
    trace.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows[:-1]),
        encoding="utf-8",
    )
    with pytest.raises(evaluation.IPQEvaluationError):
        evaluation._load_trace(run_dir, arm)


def test_frozen_gates_are_counted_and_matched_aux_initialization_hashes_repeat():
    from robot_wm.modeling.dual_diffusion.adapters import (
        TFSigmaTokenEmbedding,
        TFVelocityHead,
        ZeroInitTFTokenAdapter,
    )
    from robot_wm.modeling.dual_diffusion.invertible_two_clock import (
        parameter_schema,
        tensor_state_sha256,
    )

    def construct(seed: int):
        torch.manual_seed(seed)
        module = nn.ModuleDict(
            {
                "tf_token_adapter": ZeroInitTFTokenAdapter(
                    16,
                    hidden_size=8,
                    gate_init=0.02,
                    gate_trainable=False,
                ),
                "tf_clock_embedding": TFSigmaTokenEmbedding(
                    8,
                    embedding_dim=8,
                    gate_init=0.02,
                    gate_trainable=False,
                ),
                "tf_velocity_head": TFVelocityHead(8, 16),
            }
        )
        state = {
            f"forward_model.{key}": value for key, value in module.state_dict().items()
        }
        return module, tensor_state_sha256(state)

    sync, sync_hash = construct(1234)
    independent, independent_hash = construct(1234)
    assert sync_hash == independent_hash
    assert sync["tf_token_adapter"].gate.requires_grad is False
    assert sync["tf_clock_embedding"].gate.requires_grad is False
    assert torch.equal(
        sync["tf_token_adapter"].gate,
        independent["tf_token_adapter"].gate,
    )
    schema = parameter_schema(sync)
    optimizer = torch.optim.AdamW(sync.parameters())
    optimizer_count = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert optimizer_count == schema["parameter_count"]
    assert schema["trainable_parameter_count"] < schema["parameter_count"]


def test_parent_parity_comparator_requires_exact_nfe_1_2_4_tensor_hashes(tmp_path):
    current_commit = "1" * 40

    def receipt(role: str) -> dict:
        source_commit = (
            pilot.PARENT_TRAINING_SOURCE_COMMIT
            if role == "historical"
            else current_commit
        )
        value = {
            "kind": parent_parity.KIND_REPLAY,
            "source_role": role,
            "source_commit": source_commit,
            "parent_snapshot_sha256": pilot.PARENT_SNAPSHOT_SHA256,
            "parent_run_identity_sha256": pilot.PARENT_RUN_IDENTITY_SHA256,
            "parent_canonical_model_state_sha256": (
                pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
            ),
            "parent_resolved_config_sha256": pilot.PARENT_RESOLVED_CONFIG_SHA256,
            "nfe_grid": list(pilot.NFE_GRID),
            "strict_model_load": True,
            "deployable_target_blind_sampler": True,
            "validation_array_opened": False,
            "protected_test_accessed": False,
            "teacher_feature_encoder_calls": 0,
            "input_sha256": "2" * 64,
            "outputs": {
                str(nfe): {
                    "initial_noise_sha256": "3" * 64,
                    "final_latent_sha256": f"{nfe}" * 64,
                    "decoded_uint8_sha256": f"{nfe + 4}" * 64,
                    "actual_wan_calls": nfe,
                }
                for nfe in pilot.NFE_GRID
            },
        }
        if role == "historical":
            value["sampler_git_blob"] = (
                parent_parity.HISTORICAL_SAMPLER_GIT_BLOB
            )
        else:
            value["sampler_sha256"] = pilot.CURRENT_PARENT_SAMPLER_SHA256
        return pilot.identity_payload(value)

    historical_path = tmp_path / "historical.json"
    current_path = tmp_path / "current.json"
    pilot.exclusive_json(historical_path, receipt("historical"))
    pilot.exclusive_json(current_path, receipt("current"))
    result = parent_parity.compare(
        historical_path,
        current_path,
        current_source_commit=current_commit,
    )
    assert result["status"] == "PASS_BIT_EXACT"
    assert result["nfe_grid"] == [1, 2, 4]
    assert all(
        result["per_nfe"][str(nfe)]["final_latent_sha256"]
        for nfe in pilot.NFE_GRID
    )

    changed = receipt("current")
    changed.pop("identity_sha256")
    changed["outputs"]["2"]["final_latent_sha256"] = "9" * 64
    changed = pilot.identity_payload(changed)
    current_path.unlink()
    pilot.exclusive_json(current_path, changed)
    with pytest.raises(parent_parity.ParentParityError):
        parent_parity.compare(
            historical_path,
            current_path,
            current_source_commit=current_commit,
        )
