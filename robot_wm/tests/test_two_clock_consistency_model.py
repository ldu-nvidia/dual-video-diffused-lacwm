"""Focused tests for matched two-clock RF self-consistency."""

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
    / "two_clock_consistency_model.py"
)


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
        @staticmethod
        def _expand_sigma(sigma, reference):
            return sigma.to(reference).reshape(-1, 1, 1, 1, 1)

        @staticmethod
        def _masked_per_sample_mse(prediction, target, mask):
            weights = mask.expand_as(prediction)
            dims = tuple(range(1, prediction.ndim))
            return ((prediction.float() - target.float()).square() * weights).sum(
                dim=dims
            ) / weights.sum(dim=dims)

        @staticmethod
        def _masked_per_sample_nmse(estimate, clean, mask):
            weights = mask.expand_as(estimate)
            dims = tuple(range(1, estimate.ndim))
            return ((estimate.float() - clean.float()).square() * weights).sum(
                dim=dims
            ) / (clean.float().square() * weights).sum(dim=dims).clamp_min(1e-8)

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
        "two_clock_consistency_model_unit_test", MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rf_x0_conversion_recovers_clean_on_shared_trajectory(monkeypatch):
    module = _load_module(monkeypatch)
    clean = torch.randn(2, 3, 2, 2, 2)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.25, 0.75])
    expanded = sigma.reshape(-1, 1, 1, 1, 1)
    noisy = (1.0 - expanded) * clean + expanded * noise

    recovered = module.rf_predicted_clean(noisy, sigma, noise - clean)

    torch.testing.assert_close(recovered, clean, rtol=1e-6, atol=1e-6)


def test_consistency_stops_low_teacher_gradient_and_normalizes(monkeypatch):
    module = _load_module(monkeypatch)
    high = torch.full((2, 1, 1, 1, 2), 2.0, requires_grad=True)
    low = torch.full_like(high, 1.0, requires_grad=True)
    clean = torch.full_like(high, 2.0, requires_grad=True)
    mask = torch.ones(2, 1, 1, 1, 2)

    loss, energy = module.normalized_stopped_consistency_loss(
        high, low, clean, mask, epsilon=1e-6
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.0 / (4.0 + 1e-6))
    assert energy.item() == pytest.approx(4.0 + 1e-6)
    assert high.grad is not None and torch.count_nonzero(high.grad) > 0
    assert low.grad is None
    assert clean.grad is None


def test_clock_sampler_uses_only_preregistered_bands(monkeypatch):
    module = _load_module(monkeypatch)
    model = object.__new__(module.TwoClockConsistencyVPM)
    nn.Module.__init__(model)
    model.high_sigma_min, model.high_sigma_max = 0.8, 1.0
    model.low_sigma_min, model.low_sigma_max = 0.0, 0.4
    model.noise_scheduler = types.SimpleNamespace(
        timesteps=torch.arange(1000, -1, -1),
        sigmas=torch.linspace(1.0, 0.0, 1001),
    )

    hi_t, hi_s, lo_t, lo_s = model._sample_two_clocks(4096, torch.device("cpu"))

    assert hi_t.shape == hi_s.shape == lo_t.shape == lo_s.shape == (4096,)
    assert bool(((hi_s >= 0.8) & (hi_s <= 1.0)).all())
    assert bool(((lo_s >= 0.0) & (lo_s <= 0.4)).all())
    assert hi_s.unique().numel() > 100 and lo_s.unique().numel() > 100


class _FakeForward(nn.Module):
    def __init__(self, output_type):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.output_type = output_type
        self.tf_token_adapter = types.SimpleNamespace(tf_channels=4)

    def forward(self, noisy, timestep, *_args, **_kwargs):
        self.calls.append((noisy.detach().clone(), timestep.detach().clone()))
        return self.output_type(video_velocity=self.scale * noisy)


def _forward_fixture(module, weight: float):
    model = object.__new__(module.TwoClockConsistencyVPM)
    nn.Module.__init__(model)
    model.consistency_weight = weight
    model.consistency_epsilon = 1e-6
    model.num_history_frames = 5
    model.num_future_frames = 8
    model.num_history_latent = 2
    model.forward_model = _FakeForward(module.DualWanOutput)
    clean = torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 2 * 2).reshape(
        2, 2, 4, 2, 2
    )
    model._ensure_video_only_runtime_contract = lambda: None
    model._encode_clip = lambda _rgb: clean
    model._history_reference = lambda _rgb, _shape: (torch.zeros_like(clean), 2)
    model._latent_actions = lambda *_args, **_kwargs: (
        None,
        torch.zeros(2, 4, 3),
        None,
    )
    model._build_context = lambda *_args: [torch.zeros(1), torch.zeros(1)]
    model._build_clip = lambda *_args: torch.zeros(2, 1, 1)
    model._build_loss_mask = lambda *_args: torch.ones(2, 1, 4, 1, 2)
    model._sample_two_clocks = lambda *_args: (
        torch.tensor([900.0, 850.0]),
        torch.tensor([0.9, 0.85]),
        torch.tensor([200.0, 300.0]),
        torch.tensor([0.2, 0.3]),
    )
    return model


def test_both_arms_run_two_calls_on_same_epsilon_trajectory(monkeypatch):
    module = _load_module(monkeypatch)
    control = _forward_fixture(module, 0.0)
    candidate = _forward_fixture(module, 0.2)
    rgb = torch.zeros(2, 13, 3, 2, 2)
    actions = torch.zeros(2, 13, 5, 23)
    clips = torch.tensor([4, 7])

    torch.manual_seed(99)
    control_loss = control(rgb, actions=actions, clip_index=clips)
    torch.manual_seed(99)
    candidate_loss = candidate(rgb, actions=actions, clip_index=clips)

    assert len(control.forward_model.calls) == len(candidate.forward_model.calls) == 2
    assert candidate_loss.item() > control_loss.item()
    for key in control.paired_audit_exact:
        assert control.paired_audit_exact[key] == candidate.paired_audit_exact[key]
    noisy_lo, _ = control.forward_model.calls[0]
    noisy_hi, _ = control.forward_model.calls[1]
    clean = control._encode_clip(rgb)
    eps_from_lo = (noisy_lo - 0.8 * clean) / 0.2
    eps_from_hi = (noisy_hi - 0.1 * clean) / 0.9
    torch.testing.assert_close(eps_from_lo[0], eps_from_hi[0], atol=1e-6, rtol=1e-6)


def test_cached_auxiliary_is_forbidden_and_sampling_is_not_overridden(monkeypatch):
    module = _load_module(monkeypatch)
    model = _forward_fixture(module, 0.0)
    rgb = torch.ones(2, 13, 3, 2, 2)
    inert = model._resolve_auxiliary_clean(rgb, (2, 16, 4, 3, 5), None)

    assert inert.shape == (2, 4, 4, 3, 5)
    assert torch.count_nonzero(inert) == 0
    with pytest.raises(module.TwoClockConsistencyError, match="forbids"):
        model._resolve_auxiliary_clean(rgb, (2, 16, 4, 3, 5), inert)
    assert "_sample_future" not in module.TwoClockConsistencyVPM.__dict__
    assert "_sample_future_with_nfe" not in module.TwoClockConsistencyVPM.__dict__
