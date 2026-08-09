"""Focused IPQ-TC1 model tests without importing the VideoX runtime."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    ROOT
    / "projects"
    / "latent_action_models"
    / "lam"
    / "invertible_two_clock_model.py"
)


def _load_module(monkeypatch):
    lam_package = types.ModuleType("lam")
    lam_package.__path__ = []
    base_module = types.ModuleType("lam.dual_explicit_action_dit_model")
    wan_module = types.ModuleType("robot_wm.modeling.networks.wan_forward_model")

    @dataclass
    class DualWanOutput:
        video_velocity: torch.Tensor
        tf_velocity: torch.Tensor
        tf_condition_tokens: torch.Tensor | None = None
        tf_condition_telemetry: dict | None = None

    class DualExplicitActionDiTModel(nn.Module):
        @staticmethod
        def _masked_per_sample_nmse(estimate, clean, mask):
            weights = mask.expand_as(estimate).float()
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
        "invertible_two_clock_model_unit_test", MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeForward(nn.Module):
    def __init__(self, output_type):
        super().__init__()
        self.output_type = output_type
        self.calls = 0
        self.transformer = nn.Identity()
        self.records = []

    def forward(self, noisy, _timestep, *_args, noisy_tf, **kwargs):
        self.calls += 1
        noisy = self.transformer(noisy)
        self.records.append({"noisy_tf": noisy_tf.detach().clone(), **kwargs})
        scalar = noisy.new_tensor(0.01)
        return self.output_type(
            video_velocity=0.1 * noisy,
            tf_velocity=0.2 * noisy_tf,
            tf_condition_telemetry={
                "state_residual_rms": scalar,
                "clock_residual_rms": scalar,
                "combined_rms": scalar,
            },
        )


def _training_fixture(module, arm_mode: str):
    model = object.__new__(module.InvertibleTwoClockVPM)
    nn.Module.__init__(model)
    model.ipq_arm_mode = arm_mode
    model.ipq_history_latent_frames = 2
    model.ipq_view_width = 40
    model.ipq_num_views = 3
    model.ipq_loss_identity_tolerance = 2e-6
    model.num_history_frames = 5
    model.num_future_frames = 8
    model.forward_model = _FakeForward(module.DualWanOutput)
    generator = torch.Generator().manual_seed(17)
    clean = torch.randn(2, 16, 4, 24, 120, generator=generator)
    model._encode_clip = lambda _rgb: clean
    model._history_reference = lambda _rgb, _shape: (torch.zeros_like(clean), 2)
    model._latent_actions = lambda *_args, **_kwargs: (
        None,
        torch.zeros(2, 4, 3),
        None,
    )
    model._build_context = lambda *_args: [torch.zeros(1), torch.zeros(1)]
    model._build_clip = lambda *_args: torch.zeros(2, 1, 1)
    model._build_loss_mask = lambda *_args: torch.ones(2, 1, 4, 24, 120)
    model._draw_training_clocks = lambda *_args: (
        torch.tensor([750.0, 500.0]),
        torch.tensor([0.75, 0.50]),
        torch.tensor([250.0, 900.0]),
        torch.tensor([0.25, 0.90]),
        (
            torch.tensor([0.75, 0.50])
            if arm_mode == "synchronous"
            else torch.tensor([0.25, 0.90])
        ),
    )
    return model


def test_matched_arms_use_one_call_and_pair_every_nontreatment_input(monkeypatch):
    module = _load_module(monkeypatch)
    sync = _training_fixture(module, "synchronous")
    independent = _training_fixture(module, "independent")
    rgb = torch.zeros(2, 13, 3, 2, 2)
    actions = torch.zeros(2, 13, 5, 23)
    clip_index = torch.tensor([7, 11])

    torch.manual_seed(101)
    sync_loss = sync(rgb, actions=actions, clip_index=clip_index)
    torch.manual_seed(101)
    independent_loss = independent(rgb, actions=actions, clip_index=clip_index)

    assert torch.isfinite(sync_loss) and torch.isfinite(independent_loss)
    assert sync.forward_model.calls == independent.forward_model.calls == 1
    paired = {
        "clip_index",
        "actions",
        "clean_latent",
        "canonical_noise",
        "timestep_p",
        "sigma_p",
        "spare_timestep_q",
        "spare_sigma_q",
        "cpu_rng_before_wan",
        "cuda_rng_before_wan",
        "cpu_rng_after_wan",
        "cuda_rng_after_wan",
        "wan_call_count",
    }
    assert all(
        sync.paired_audit_exact[key] == independent.paired_audit_exact[key]
        for key in paired
    )
    assert (
        sync.paired_audit_exact["effective_sigma_q"]
        != independent.paired_audit_exact["effective_sigma_q"]
    )
    assert float(sync.aux_losses["ipq/loss_identity_relative_error"]) <= 2e-6
    assert float(independent.aux_losses["ipq/loss_identity_relative_error"]) <= 2e-6


def _sampling_fixture(module):
    model = object.__new__(module.InvertibleTwoClockVPM)
    nn.Module.__init__(model)
    model.ipq_history_latent_frames = 2
    model.ipq_view_width = 40
    model.ipq_num_views = 3
    model.num_history_frames = 5
    model.num_future_frames = 8
    model.forward_model = _FakeForward(module.DualWanOutput)
    history = torch.randn(2, 16, 2, 24, 120)
    model._encode_clip = lambda _rgb: history
    model.rgb_tokenizer = types.SimpleNamespace(latent_temporal_len=lambda _n: 4)
    model._latent_actions = lambda *_args, **_kwargs: (
        None,
        torch.zeros(2, 4, 3),
        None,
    )
    model._build_context = lambda *_args: [torch.zeros(1), torch.zeros(1)]
    model._build_clip = lambda *_args: torch.zeros(2, 1, 1)
    model._roll_across_global_batch = lambda value: torch.roll(value, 1, 0)
    model.sample_scheduler = types.SimpleNamespace(
        set_timesteps=lambda n, device=None: (
            setattr(model.sample_scheduler, "sigmas", torch.linspace(1, 0, n + 1)),
            setattr(model.sample_scheduler, "timesteps", torch.arange(n, 0, -1)),
        )
    )
    return model


def test_nfe_one_schedule_order_is_bit_identical_within_checkpoint(monkeypatch):
    module = _load_module(monkeypatch)
    model = _sampling_fixture(module)
    history_rgb = torch.zeros(2, 5, 3, 180, 960)
    initial_noise = torch.randn(2, 16, 4, 24, 120)
    kwargs = dict(
        history_rgb=history_rgb,
        actions=torch.zeros(2, 13, 5, 23),
        morphology_index=None,
        initial_noise=initial_noise,
        num_steps=1,
        condition_source="aligned",
    )

    sync = model.sample_ipq_latent(schedule_mode="synchronous", **kwargs)
    p_leads = model.sample_ipq_latent(schedule_mode="p_leads", **kwargs)
    q_leads = model.sample_ipq_latent(schedule_mode="q_leads", **kwargs)

    assert sync.wan_calls == p_leads.wan_calls == q_leads.wan_calls == 1
    assert torch.equal(sync.latent, p_leads.latent)
    assert torch.equal(sync.latent, q_leads.latent)
    assert torch.equal(sync.p_sigmas, p_leads.p_sigmas)
    assert torch.equal(sync.q_sigmas, p_leads.q_sigmas)


def test_production_sampler_is_not_overridden(monkeypatch):
    module = _load_module(monkeypatch)

    assert "_sample_future" not in module.InvertibleTwoClockVPM.__dict__
    assert "sample_future_deployable" not in module.InvertibleTwoClockVPM.__dict__
    assert "sample_ipq_latent" in module.InvertibleTwoClockVPM.__dict__


def test_state_off_disables_only_shared_trunk_not_local_q_head_inputs(monkeypatch):
    module = _load_module(monkeypatch)
    model = _sampling_fixture(module)
    result = model.sample_ipq_latent(
        torch.zeros(2, 5, 3, 180, 960),
        actions=torch.zeros(2, 13, 5, 23),
        morphology_index=None,
        initial_noise=torch.randn(2, 16, 4, 24, 120),
        num_steps=1,
        schedule_mode="p_leads",
        condition_source="state_off",
    )
    record = model.forward_model.records[-1]
    assert record["condition_on_tf"] is False
    assert record["condition_on_tf_clock"] is False
    # Local q and its raw clock remain present for the Q velocity head even
    # though neither residual enters the shared video trunk.
    _expected_p, expected_q = module.split_pq(
        result.initial_noise, history_frames=2, view_width=40
    )
    assert torch.equal(record["noisy_tf"], expected_q)
    assert torch.equal(record["conditioning_tf"], record["noisy_tf"])
    assert tuple(record["tf_sigma"].shape) == (2,)
