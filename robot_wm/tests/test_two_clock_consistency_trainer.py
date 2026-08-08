"""Focused audit tests for the two-clock trainer specialization."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = (
    REPO_ROOT
    / "robot_wm"
    / "utils"
    / "two_clock_consistency_trainer.py"
)
AUDIT_FIELDS = (
    "clip_index",
    "actions",
    "clean_latent",
    "epsilon",
    "sigma_hi",
    "sigma_lo",
    "timestep_hi",
    "timestep_lo",
    "noisy_hi",
    "noisy_lo",
    "cpu_rng_state_after_forward",
    "cuda_rng_state_after_forward",
)


def _load_module(monkeypatch):
    base = types.ModuleType("robot_wm.utils.trainer")

    class Trainer:
        def _step(self):
            return {"loss": 1.0}

    base.Trainer = Trainer
    monkeypatch.setitem(sys.modules, "robot_wm.utils.trainer", base)
    spec = importlib.util.spec_from_file_location(
        "two_clock_consistency_trainer_unit_test", TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _fixture(module, fields=AUDIT_FIELDS):
    trainer = object.__new__(module.TwoClockConsistencyTrainer)
    audit = {field: f"{index:064x}" for index, field in enumerate(fields)}
    trainer.model = types.SimpleNamespace(
        module=types.SimpleNamespace(paired_audit_exact=audit)
    )
    trainer.world_size = 1
    return trainer


def test_step_binds_all_shared_trajectory_and_rng_hashes(monkeypatch):
    module = _load_module(monkeypatch)
    losses = _fixture(module)._step()

    assert losses["loss"] == 1.0
    exact = [key for key in losses if key.startswith("paired_audit/exact_")]
    assert len(exact) == len(AUDIT_FIELDS)
    assert all(len(losses[key]) == 64 for key in exact)


def test_step_fails_closed_when_one_clock_hash_is_missing(monkeypatch):
    module = _load_module(monkeypatch)
    fields = tuple(field for field in AUDIT_FIELDS if field != "sigma_lo")

    with pytest.raises(RuntimeError, match="audit is incomplete"):
        _fixture(module, fields)._step()
