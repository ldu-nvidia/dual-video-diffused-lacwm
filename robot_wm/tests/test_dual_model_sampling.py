import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import pytest
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    REPO_ROOT
    / "projects"
    / "latent_action_models"
    / "lam"
    / "dual_explicit_action_dit_model.py"
)


def _load_model_module(monkeypatch):
    """Load the dual model without importing the optional VideoX runtime."""
    lam_package = types.ModuleType("lam")
    lam_package.__path__ = []
    base_module = types.ModuleType("lam.explicit_action_dit_model")

    class ExplicitActionDiTModel:
        pass

    base_module.ExplicitActionDiTModel = ExplicitActionDiTModel

    wan_module = types.ModuleType(
        "robot_wm.modeling.networks.wan_forward_model"
    )

    @dataclass
    class DualWanOutput:
        video_velocity: torch.Tensor
        tf_velocity: torch.Tensor
        tf_condition_tokens: torch.Tensor | None = None
        tf_condition_telemetry: dict | None = None

    wan_module.DualWanOutput = DualWanOutput
    monkeypatch.setitem(sys.modules, "lam", lam_package)
    monkeypatch.setitem(
        sys.modules, "lam.explicit_action_dit_model", base_module
    )
    monkeypatch.setitem(
        sys.modules,
        "robot_wm.modeling.networks.wan_forward_model",
        wan_module,
    )

    spec = importlib.util.spec_from_file_location(
        "dual_explicit_action_dit_model_sampling_test", MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_nfe_model_sampling_pairs_matched_and_shuffled_noise(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    batch_size = 2
    video_clean = torch.zeros(batch_size, 16, 4, 1, 1)
    tf_clean = torch.randn(batch_size, 12, 4, 1, 1)

    model.num_history_latent = 2
    model.num_future_frames = 2
    model.tf_condition_mode = "shuffled"
    model.condition_on_tf = True
    model.video_only_control = False
    model.tf_loss_weight = 1.0
    model.tf_schedule_mode = "aligned"
    model.tf_lead_logit = 1.0
    model.evaluation_noise_seed = 123
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (1,)
    model.viz_num_steps = 1
    model.capture_latent_trajectories = False
    model._visualization_artifacts = None
    model._encode_clip = lambda _rgb: video_clean
    model._tf_clean = lambda _rgb, _shape: tf_clean
    model._latent_actions = (
        lambda _rgb, _actions, _morphology, _latent_frames, _history: (
            None,
            torch.zeros_like(video_clean),
            None,
        )
    )
    model._build_context = (
        lambda _batch_size, _device, _dtype: torch.empty(0)
    )
    model._build_clip = (
        lambda _batch_size, _device, _dtype: torch.empty(0)
    )

    class Tokenizer:
        @staticmethod
        def decode_temporal(latent, out_hw):
            return torch.zeros(
                latent.shape[0],
                3,
                latent.shape[2],
                *out_hw,
                device=latent.device,
                dtype=latent.dtype,
            )

    class Scheduler:
        timesteps = torch.tensor([1.0])
        sigmas = torch.tensor([1.0, 0.0])

        def set_timesteps(self, num_steps, device):
            assert num_steps == 1
            self.timesteps = torch.tensor([1.0], device=device)
            self.sigmas = torch.tensor([1.0, 0.0], device=device)

        @staticmethod
        def step(_velocity, _timestep, state):
            return types.SimpleNamespace(prev_sample=state)

    model.rgb_tokenizer = Tokenizer()
    model.sample_scheduler = Scheduler()
    captured = {}

    def forward_model(
        video_state,
        _timesteps,
        _z_control,
        _reference,
        _context,
        _clip_fea,
        *,
        noisy_tf,
        conditioning_tf,
        tf_sigma,
        condition_on_tf,
    ):
        captured["noisy_tf"] = noisy_tf.detach().clone()
        captured["conditioning_tf"] = conditioning_tf.detach().clone()
        captured["tf_sigma"] = tf_sigma.detach().clone()
        captured["condition_on_tf"] = condition_on_tf
        return module.DualWanOutput(
            video_velocity=torch.zeros_like(video_state),
            tf_velocity=torch.zeros_like(noisy_tf),
        )

    model.forward_model = forward_model
    model.forward_model.tf_token_adapter = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.1)
    )
    model.forward_model.tf_clock_embedding = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.0)
    )
    rgb = torch.zeros(batch_size, 3, 13, 2, 2)

    model._sample_future(rgb)

    assert captured["condition_on_tf"] is True
    torch.testing.assert_close(
        captured["tf_sigma"], torch.ones(batch_size), rtol=0, atol=0
    )
    torch.testing.assert_close(
        captured["conditioning_tf"],
        captured["noisy_tf"],
        rtol=0,
        atol=0,
    )


def test_video_only_control_fails_closed_if_a_tf_path_can_open(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)

    class Gate:
        def __init__(self):
            self.gate = nn.Parameter(torch.zeros(()), requires_grad=False)

        def effective_gate(self):
            return torch.tanh(self.gate)

    model.video_only_control = True
    model.tf_condition_mode = "off"
    model.condition_on_tf = False
    model.tf_loss_weight = 0.0
    model.forward_model = types.SimpleNamespace(
        condition_on_tf=False,
        tf_token_adapter=Gate(),
        tf_clock_embedding=Gate(),
    )

    model._assert_video_only_control_contract()

    model.forward_model.tf_clock_embedding.gate.data.fill_(0.1)
    with pytest.raises(RuntimeError, match="clock gate is not exact zero"):
        model._assert_video_only_control_contract()

    model.forward_model.tf_clock_embedding.gate.data.zero_()
    model.tf_loss_weight = 1.0
    with pytest.raises(RuntimeError, match="TF loss weight"):
        model._assert_video_only_control_contract()


def test_video_only_training_tf_placeholder_preserves_rng_state(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    clean = torch.ones(2, 12, 4, 2, 2)

    torch.manual_seed(20260726)
    before = torch.random.get_rng_state()
    model.video_only_control = True
    placeholder = model._training_tf_noise(clean)
    after = torch.random.get_rng_state()

    assert torch.count_nonzero(placeholder) == 0
    assert torch.equal(before, after)

    model.video_only_control = False
    sampled = model._training_tf_noise(clean)
    assert torch.count_nonzero(sampled) > 0
    assert not torch.equal(after, torch.random.get_rng_state())


def test_video_only_objective_has_no_tf_graph_or_nan_path(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.video_only_control = True
    model.tf_loss_weight = 0.0
    video_loss = torch.tensor(2.0, requires_grad=True)
    tf_loss = torch.tensor(float("nan"), requires_grad=True)

    objective = model._training_objective(video_loss, tf_loss)
    objective.backward()

    assert torch.isfinite(objective)
    assert objective is video_loss
    torch.testing.assert_close(video_loss.grad, torch.tensor(1.0))
    assert tf_loss.grad is None


def test_autonomous_shuffled_rolls_only_generated_future_residual(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    batch_size = 2
    video_clean = torch.zeros(batch_size, 16, 4, 1, 1)
    tf_clean = torch.tensor(
        [
            [[[[0.0]], [[0.0]], [[4.0]], [[6.0]]]],
            [[[[1.0]], [[1.0]], [[8.0]], [[10.0]]]],
        ]
    )

    model.num_history_latent = 2
    model.num_future_frames = 2
    model.tf_condition_mode = "matched"
    model.condition_on_tf = True
    model.video_only_control = False
    model.tf_loss_weight = 1.0
    model.tf_schedule_mode = "aligned"
    model.tf_lead_logit = 1.0
    model.evaluation_noise_seed = 123
    model.evaluation_condition_sources = (
        "autonomous",
        "autonomous_shuffled",
    )
    model.evaluation_nfe_steps = (2,)
    model.viz_num_steps = 2
    model.capture_latent_trajectories = True
    model._visualization_artifacts = None
    model._encode_clip = lambda _rgb: video_clean
    model._tf_clean = lambda _rgb, _shape: tf_clean
    model._latent_actions = (
        lambda _rgb, _actions, _morphology, _latent_frames, _history: (
            None,
            torch.zeros_like(video_clean),
            None,
        )
    )
    model._build_context = (
        lambda _batch_size, _device, _dtype: torch.empty(0)
    )
    model._build_clip = (
        lambda _batch_size, _device, _dtype: torch.empty(0)
    )

    class Tokenizer:
        @staticmethod
        def decode_temporal(latent, out_hw):
            return torch.zeros(
                latent.shape[0],
                3,
                latent.shape[2],
                *out_hw,
                device=latent.device,
                dtype=latent.dtype,
            )

    class Scheduler:
        timesteps = torch.tensor([1.0, 0.5])
        sigmas = torch.tensor([1.0, 0.5, 0.0])

        def set_timesteps(self, num_steps, device):
            assert num_steps == 2
            self.timesteps = torch.tensor([1.0, 0.5], device=device)
            self.sigmas = torch.tensor([1.0, 0.5, 0.0], device=device)

        @staticmethod
        def step(_velocity, _timestep, state):
            return types.SimpleNamespace(prev_sample=state)

    model.rgb_tokenizer = Tokenizer()
    model.sample_scheduler = Scheduler()
    calls = []
    initial_future_noise = None

    def forward_model(
        video_state,
        _timesteps,
        _z_control,
        _reference,
        _context,
        _clip_fea,
        *,
        noisy_tf,
        conditioning_tf,
        tf_sigma,
        condition_on_tf,
    ):
        nonlocal initial_future_noise
        calls.append(
            {
                "noisy_tf": noisy_tf.detach().clone(),
                "conditioning_tf": conditioning_tf.detach().clone(),
                "tf_sigma": tf_sigma.detach().clone(),
                "condition_on_tf": condition_on_tf,
            }
        )
        if initial_future_noise is None:
            initial_future_noise = noisy_tf.detach().clone()
        velocity = torch.zeros_like(noisy_tf)
        velocity[:, :, 2:] = (
            initial_future_noise[:, :, 2:] - tf_clean[:, :, 2:]
        )
        return module.DualWanOutput(
            video_velocity=torch.zeros_like(video_state),
            tf_velocity=velocity,
        )

    model.forward_model = forward_model
    model.forward_model.tf_token_adapter = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.1)
    )
    model.forward_model.tf_clock_embedding = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.0)
    )
    rgb = torch.zeros(batch_size, 3, 13, 2, 2)

    model._sample_future(rgb)

    assert len(calls) == 4
    # The first call of each source is an exact sigma=1 negative control.
    torch.testing.assert_close(
        calls[0]["conditioning_tf"],
        calls[2]["conditioning_tf"],
        rtol=0,
        atol=0,
    )
    local_state = calls[1]["noisy_tf"]
    local_noise = initial_future_noise
    expected_shuffled = (
        torch.roll(
            local_state - 0.5 * local_noise,
            shifts=-1,
            dims=0,
        )
        + 0.5 * local_noise
    )
    expected_shuffled[:, :, :2] = local_state[:, :, :2]
    torch.testing.assert_close(
        calls[1]["conditioning_tf"], local_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        calls[3]["conditioning_tf"],
        expected_shuffled,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        calls[3]["conditioning_tf"][:, :, :2],
        local_state[:, :, :2],
        rtol=0,
        atol=0,
    )
    artifacts = model.pop_visualization_artifacts()
    assert artifacts is not None
    assert artifacts["evaluation_condition_source_codes"].tolist() == [0, 4]
    assert "video_final_autonomous_shuffled_nfe_2" in artifacts
    assert "tf_final_autonomous_shuffled_nfe_2" in artifacts
    assert "decoded_future_autonomous_shuffled_nfe_2" in artifacts
