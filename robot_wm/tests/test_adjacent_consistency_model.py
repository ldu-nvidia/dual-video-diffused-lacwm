"""Focused ACD-P0 sampler tests without importing the VideoX runtime."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import pytest
from torch import nn

from robot_wm.modeling.low_nfe.adjacent_consistency import consistency_output


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    ROOT
    / "projects"
    / "latent_action_models"
    / "lam"
    / "adjacent_consistency_model.py"
)


def _load_module(monkeypatch):
    lam_package = types.ModuleType("lam")
    lam_package.__path__ = []
    base_module = types.ModuleType("lam.dual_explicit_action_dit_model")
    wan_module = types.ModuleType("robot_wm.modeling.networks.wan_forward_model")

    @dataclass
    class DualWanOutput:
        video_velocity: torch.Tensor
        tf_velocity: torch.Tensor | None = None

    class DualExplicitActionDiTModel(nn.Module):
        pass

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
        "adjacent_consistency_model_unit_test", MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _Scheduler:
    def __init__(self) -> None:
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)

    def set_timesteps(self, steps, device=None) -> None:
        self.sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
        self.timesteps = self.sigmas[:-1] * 1000.0


def _fixture(module):
    model = object.__new__(module.AdjacentConsistencyVPM)
    nn.Module.__init__(model)
    model.num_history_frames = 5
    model.num_future_frames = 8
    model.acd_sigma_data = 0.5
    model.sample_scheduler = _Scheduler()
    history = torch.full((1, 16, 2, 2, 2), 0.25)
    model._encode_clip = lambda _rgb: history
    model.rgb_tokenizer = types.SimpleNamespace(
        latent_temporal_len=lambda _frames: 4,
        decode_temporal=lambda latent, out_hw: latent,
    )
    model._ensure_video_only_runtime_contract = lambda: None
    model._conditions_for = lambda *_args, **_kwargs: (
        torch.zeros(1),
        None,
        torch.zeros(1),
    )
    model._acd_call_counts = {}
    model._acd_rng_before = {}

    def velocity_call(_model, state, *_args, role, **_kwargs):
        model._acd_call_counts[role] = model._acd_call_counts.get(role, 0) + 1
        return torch.full_like(state, 0.2)

    model._velocity_call = velocity_call
    return model, history


def test_sampler_consumes_trained_karras_map_not_plain_rf_x0(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    model, history = _fixture(module)
    noise = torch.ones(1, 1, 16, 4, 2, 2)
    result = model.sample_consistency_deployable(
        torch.zeros(1, 5, 3, 2, 2),
        torch.zeros(1, 13, 5, 157),
        None,
        renoise_noises=noise,
        steps=1,
    )
    source = noise[0]
    expected = consistency_output(
        source, torch.full_like(source, 0.2), 1.0, sigma_data=0.5
    )
    expected[:, :, :2] = history
    ordinary_x0 = source - 0.2
    assert torch.equal(result.absolute_video_latent, expected)
    assert not torch.equal(result.absolute_video_latent[:, :, 2:], ordinary_x0[:, :, 2:])
    assert result.model_calls == 1
    assert result.teacher_calls == result.feature_calls == 0


def test_sampler_reclamps_observed_prefix_after_every_renoise(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    model, history = _fixture(module)
    original = module.restore_history_forward_path
    restored_prefixes = []

    def audited_restore(*args, **kwargs):
        value = original(*args, **kwargs)
        restored_prefixes.append(value[:, :, :2].clone())
        return value

    monkeypatch.setattr(module, "restore_history_forward_path", audited_restore)
    noises = torch.randn(4, 1, 16, 4, 2, 2)
    result = model.sample_consistency_deployable(
        torch.zeros(1, 5, 3, 2, 2),
        torch.zeros(1, 13, 5, 157),
        None,
        renoise_noises=noises,
        steps=4,
    )
    # Initial corruption plus each of the three stochastic re-noising states.
    assert len(restored_prefixes) == 4
    sigmas = torch.linspace(1.0, 0.0, 5)[:4]
    for index, (prefix, sigma) in enumerate(zip(restored_prefixes, sigmas, strict=True)):
        expected = (1.0 - sigma) * history + sigma * noises[index, :, :, :2]
        torch.testing.assert_close(prefix, expected)
    assert torch.equal(result.absolute_video_latent[:, :, :2], history)
    assert result.model_calls == 4


def test_native_rf_nfe_one_is_ordinary_x0_and_call_matched(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    model, history = _fixture(module)
    noise = torch.ones(1, 1, 16, 4, 2, 2)
    result = model.sample_consistency_deployable(
        torch.zeros(1, 5, 3, 2, 2),
        torch.zeros(1, 13, 5, 157),
        None,
        renoise_noises=noise,
        steps=1,
        sampler_mode="native_rf",
    )
    expected = noise[0] - 0.2
    expected[:, :, :2] = history
    assert torch.equal(result.absolute_video_latent, expected)
    assert result.sampler_mode == "native_rf"
    assert result.model_calls == 1


def test_production_sampler_is_not_overridden(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    assert "_sample_future" not in module.AdjacentConsistencyVPM.__dict__
    assert "sample_future_deployable" not in module.AdjacentConsistencyVPM.__dict__
    assert "sample_consistency_deployable" in module.AdjacentConsistencyVPM.__dict__


@pytest.mark.parametrize(
    "sigmas",
    (
        (0.9, 0.0),
        (1.0, 0.2),
        (1.0, 0.4, 0.5, 0.1, 0.0),
        (1.0, float("nan")),
        (1.0, -0.1),
    ),
)
def test_sampler_fails_closed_on_noncanonical_deployment_clock(
    monkeypatch, sigmas
) -> None:
    module = _load_module(monkeypatch)
    model, _history = _fixture(module)

    class BadScheduler:
        def set_timesteps(self, steps, device=None) -> None:
            values = torch.tensor(sigmas, device=device)
            if values.numel() != steps + 1:
                values = torch.linspace(1.0, 0.0, steps + 1, device=device)
                values[2] = values[1] + 0.1
            self.sigmas = values
            self.timesteps = values[:-1] * 1000.0

    model.sample_scheduler = BadScheduler()
    steps = len(sigmas) - 1 if len(sigmas) in {2, 3, 5} else 1
    with pytest.raises(module.AdjacentConsistencyError, match="deployment clock"):
        model.sample_consistency_deployable(
            torch.zeros(1, 5, 3, 2, 2),
            torch.zeros(1, 13, 5, 157),
            None,
            renoise_noises=torch.ones(steps, 1, 16, 4, 2, 2),
            steps=steps,
        )
