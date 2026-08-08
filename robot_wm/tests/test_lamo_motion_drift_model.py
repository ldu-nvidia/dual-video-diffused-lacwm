"""Unit tests for the parameter-free LaMo RF objective."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    REPO_ROOT
    / "projects"
    / "latent_action_models"
    / "lam"
    / "lamo_motion_drift_model.py"
)
TRAINER_PATH = REPO_ROOT / "robot_wm" / "utils" / "lamo_motion_drift_trainer.py"


def _load_module(monkeypatch):
    """Load the objective without importing the optional VideoX runtime."""

    lam_package = types.ModuleType("lam")
    lam_package.__path__ = []
    base_module = types.ModuleType("lam.dual_explicit_action_dit_model")
    wan_module = types.ModuleType("robot_wm.modeling.networks.wan_forward_model")

    @dataclass
    class DualWanOutput:
        video_velocity: torch.Tensor
        tf_velocity: torch.Tensor | None = None
        tf_condition_tokens: torch.Tensor | None = None
        tf_condition_telemetry: dict | None = None

    class DualExplicitActionDiTModel(nn.Module):
        def forward(self, *_args, **_kwargs):
            self.aux_losses = {}
            self._lamo_clean_full = self.test_clean
            self._lamo_video_noisy = self.test_noisy
            self._lamo_timesteps = self.test_timesteps
            self._lamo_prediction = self.test_prediction
            return self.test_base_loss

        def _get_sigmas(self, *_args, **_kwargs):
            return self.test_sigma[:, None]

    base_module.DualExplicitActionDiTModel = DualExplicitActionDiTModel
    wan_module.DualWanOutput = DualWanOutput
    monkeypatch.setitem(sys.modules, "lam", lam_package)
    monkeypatch.setitem(
        sys.modules, "lam.dual_explicit_action_dit_model", base_module
    )
    monkeypatch.setitem(
        sys.modules, "robot_wm.modeling.networks.wan_forward_model", wan_module
    )
    spec = importlib.util.spec_from_file_location(
        "lamo_motion_drift_model_unit_test", MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rf_x0_conversion_is_exact_for_lacwm_clock(monkeypatch):
    module = _load_module(monkeypatch)
    clean = torch.tensor([1.5, -2.0])[:, None, None]
    noise = torch.tensor([-0.5, 3.0])[:, None, None]
    sigma = torch.tensor([0.25, 0.75])
    noisy = (1.0 - sigma[:, None, None]) * clean + sigma[:, None, None] * noise
    velocity = noise - clean

    recovered = module.rf_predicted_clean(noisy, sigma, velocity)

    torch.testing.assert_close(recovered, clean, rtol=0, atol=1e-7)
    assert module.rf_lamo_schedule_weight(torch.tensor([1.0])).item() == 0.0
    assert module.rf_lamo_schedule_weight(torch.tensor([0.0])).item() == 1.0
    assert module.rf_lamo_schedule_weight(torch.tensor([0.0, 1.0])).item() == 0.5
    assert module.global_rf_lamo_schedule_weight(
        torch.tensor([0.0, 1.0])
    ).item() == 0.5


def test_exact_tensor_hash_binds_bfloat_bytes_dtype_and_shape(monkeypatch):
    module = _load_module(monkeypatch)
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    assert module.tensor_sha256(value) == module.tensor_sha256(value.clone())
    assert module.tensor_sha256(value) != module.tensor_sha256(value.reshape(2, 1))
    assert module.tensor_sha256(value) != module.tensor_sha256(value.float())


def test_schedule_weight_uses_effective_ddp_batch(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def emulate_second_rank(packed, op):
        assert op == torch.distributed.ReduceOp.SUM
        # This rank has sigma=0 -> [signal power 1, count 1]. Emulate a
        # second rank with sigma=1 -> [signal power 0, count 1].
        packed.add_(torch.tensor([0.0, 1.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", emulate_second_rank)

    assert module.global_rf_lamo_schedule_weight(torch.tensor([0.0])).item() == 0.5


def test_macro_loss_excludes_history_and_uses_only_one_future_delta(monkeypatch):
    module = _load_module(monkeypatch)
    clean = torch.zeros(1, 2, 4, 2, 3)
    predicted = torch.zeros_like(clean)
    clean[:, 0, 3] = 1.0
    clean[:, 1, 3] = 2.0
    predicted[:, 0, 3] = 2.0
    predicted[:, 1, 3] = 4.0
    predicted[:, :, :2] = 10_000.0

    loss, predicted_macro, target_macro = module.macro_future_drift_loss(
        predicted,
        clean,
        history_tokens=2,
        epsilon=1e-6,
    )

    torch.testing.assert_close(predicted_macro, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(target_macro, torch.tensor([[1.0, 2.0]]))
    assert loss.item() == pytest.approx(5.0 / (5.0 + 1e-6))
    with pytest.raises(module.LamoMotionDriftError, match="exactly two future"):
        module.macro_future_drift_loss(
            predicted[:, :, :3], clean[:, :, :3], history_tokens=2, epsilon=1e-6
        )


def _forward_fixture(module, weight: float):
    model = object.__new__(module.LamoMotionDriftVPM)
    nn.Module.__init__(model)
    model.motion_drift_weight = weight
    model.motion_drift_epsilon = 1e-6
    model.num_history_frames = 5
    model.num_future_frames = 8
    model.num_history_latent = 2
    model.test_clean = torch.zeros(2, 2, 4, 1, 1)
    model.test_clean[:, :, 3] = 1.0
    model.test_sigma = torch.tensor([0.25, 0.75])
    velocity = torch.zeros_like(model.test_clean, requires_grad=True)
    model.test_noisy = model.test_clean + 0.5
    model.test_timesteps = torch.tensor([250, 750])
    model.test_prediction = module.DualWanOutput(video_velocity=velocity)
    model.test_base_loss = torch.tensor(3.0, requires_grad=True)
    return model


def test_lambda_zero_returns_original_loss_and_does_not_advance_rng(monkeypatch):
    module = _load_module(monkeypatch)
    model = _forward_fixture(module, 0.0)
    state = torch.random.get_rng_state().clone()

    result = model(torch.zeros(2, 13, 3, 1, 1), mask=torch.ones(2, 13))

    assert result is model.test_base_loss
    assert torch.equal(torch.random.get_rng_state(), state)
    assert model.aux_losses["motion_drift/lambda"].item() == 0.0
    assert torch.isfinite(model.aux_losses["motion_drift/raw_loss"])
    assert model.test_prediction.video_velocity.grad is None
    result.backward()
    assert model.test_base_loss.grad.item() == 1.0
    assert model.test_prediction.video_velocity.grad is None


def test_inherited_auxiliary_state_is_zero_and_rejects_cached_target(monkeypatch):
    module = _load_module(monkeypatch)
    model = object.__new__(module.LamoMotionDriftVPM)
    nn.Module.__init__(model)
    model.forward_model = types.SimpleNamespace(
        tf_token_adapter=types.SimpleNamespace(tf_channels=64)
    )
    rgb = torch.ones(2, 13, 3, 2, 2)

    target = model._resolve_auxiliary_clean(rgb, (2, 16, 4, 3, 5), None)

    assert target.shape == (2, 64, 4, 3, 5)
    assert torch.count_nonzero(target) == 0
    with pytest.raises(module.LamoMotionDriftError, match="forbids"):
        model._resolve_auxiliary_clean(rgb, (2, 16, 4, 3, 5), target)


def test_trainer_promotes_every_rank_local_exact_hash_into_trace(monkeypatch):
    base_module = types.ModuleType("robot_wm.utils.trainer")

    class Trainer:
        def _step(self):
            return {"loss": 1.0}

    base_module.Trainer = Trainer
    monkeypatch.setitem(sys.modules, "robot_wm.utils.trainer", base_module)
    spec = importlib.util.spec_from_file_location(
        "lamo_motion_drift_trainer_unit_test", TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    fields = (
        "clip_index",
        "actions",
        "clean_latent",
        "noisy_latent",
        "timesteps",
        "cpu_rng_state_after_forward",
        "cuda_rng_state_after_forward",
    )
    trainer = object.__new__(module.LamoMotionDriftTrainer)
    trainer.world_size = 1
    trainer.model = types.SimpleNamespace(
        module=types.SimpleNamespace(
            paired_audit_exact={field: f"{index:064x}" for index, field in enumerate(fields)}
        )
    )

    losses = trainer._step()

    assert losses["loss"] == 1.0
    for field in fields:
        value = losses[f"paired_audit/exact_{field}_all_ranks_sha256"]
        assert len(value) == 64
