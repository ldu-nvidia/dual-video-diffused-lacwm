import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


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
    model.tf_schedule_mode = "aligned"
    model.tf_lead_logit = 1.0
    model.evaluation_noise_seed = 123
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (1,)
    model.viz_num_steps = 1
    model.capture_latent_trajectories = False
    model.cascade_condition_only_video_loss_examples = False
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


class NativeScheduler:
    def set_timesteps(self, num_steps, device):
        self.timesteps = torch.arange(
            num_steps,
            0,
            -1,
            device=device,
            dtype=torch.float32,
        )
        self.sigmas = torch.linspace(
            1,
            0,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )


def test_strict_cascade_executes_exact_nfe_and_perfect_velocity_endpoints(
    monkeypatch,
):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    batch_size = 2
    video_clean = torch.zeros(batch_size, 16, 4, 1, 1)
    tf_clean = torch.zeros(batch_size, 12, 4, 1, 1)

    model.num_history_latent = 2
    model.num_future_frames = 2
    model.tf_condition_mode = "matched"
    model.condition_on_tf = True
    model.tf_schedule_mode = "tf_first_cascaded"
    model.tf_lead_logit = 1.0
    model.cascade_inference_tf_fraction = 0.5
    model.evaluation_noise_seed = 123
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (5,)
    model.viz_num_steps = 5
    model.capture_latent_trajectories = True
    model.cascade_condition_only_video_loss_examples = True
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

    class CountingEulerScheduler(NativeScheduler):
        def set_timesteps(self, num_steps, device):
            super().set_timesteps(num_steps, device)
            self.step_calls = 0

        def step(self, velocity, _timestep, state):
            sigma = self.sigmas[self.step_calls]
            next_sigma = self.sigmas[self.step_calls + 1]
            self.step_calls += 1
            return types.SimpleNamespace(
                prev_sample=state + (next_sigma - sigma) * velocity
            )

    model.rgb_tokenizer = Tokenizer()
    model.sample_scheduler = CountingEulerScheduler()
    captured = {
        "forward_calls": 0,
        "video_velocity": None,
        "tf_velocity": None,
    }

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
        assert conditioning_tf.shape == noisy_tf.shape
        assert tf_sigma.shape == (batch_size,)
        assert condition_on_tf is True
        captured["forward_calls"] += 1
        if captured["video_velocity"] is None:
            captured["video_velocity"] = video_state.detach().clone()
            captured["tf_velocity"] = noisy_tf.detach().clone()
        return module.DualWanOutput(
            video_velocity=captured["video_velocity"],
            tf_velocity=captured["tf_velocity"],
        )

    model.forward_model = forward_model
    rgb = torch.zeros(batch_size, 3, 13, 2, 2)

    model._sample_future(rgb)
    artifacts = model.pop_visualization_artifacts()

    assert captured["forward_calls"] == 5
    assert model.sample_scheduler.step_calls == 3
    torch.testing.assert_close(
        artifacts["video_final_nfe_5"].float(),
        torch.zeros_like(artifacts["video_final_nfe_5"].float()),
        atol=1e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        artifacts["tf_final_nfe_5"].float(),
        torch.zeros_like(artifacts["tf_final_nfe_5"].float()),
        atol=1e-3,
        rtol=0,
    )


def test_strict_cascade_sampling_freezes_one_branch_per_step(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "tf_first_cascaded"
    model.cascade_inference_tf_fraction = 0.5
    model.sample_scheduler = NativeScheduler()

    schedule, timesteps, tf_only_steps = model._sampling_schedule(
        8,
        device=torch.device("cpu"),
    )

    assert tf_only_steps == 4
    assert timesteps.shape == (8,)
    assert torch.all(timesteps[:tf_only_steps] == timesteps[0])
    assert schedule.video[0] == schedule.time_frequency[0] == 1
    assert schedule.video[-1] == schedule.time_frequency[-1] == 0
    video_updates = torch.diff(schedule.video) != 0
    tf_updates = torch.diff(schedule.time_frequency) != 0
    assert not torch.any(video_updates & tf_updates)
    assert torch.all(schedule.video[: tf_only_steps + 1] == 1)
    assert torch.all(schedule.time_frequency[tf_only_steps:] == 0)


def test_strict_cascade_sampling_rejects_one_total_nfe(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "tf_first_cascaded"
    model.cascade_inference_tf_fraction = 0.5
    model.sample_scheduler = NativeScheduler()

    with pytest.raises(ValueError, match="at least two"):
        model._sampling_schedule(1, device=torch.device("cpu"))


def test_strict_training_masks_content_out_of_tf_loss_examples(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.condition_on_tf = True
    model.cascade_condition_only_video_loss_examples = True

    mask = model._training_condition_mask(
        torch.tensor([1.0, 0.0, 1.0, 0.0])
    )

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(
        mask,
        torch.tensor([True, False, True, False]),
    )


def test_non_treatment_arm_keeps_content_disabled(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.condition_on_tf = False
    model.cascade_condition_only_video_loss_examples = True

    assert (
        model._training_condition_mask(torch.tensor([1.0, 0.0]))
        is False
    )


@pytest.mark.parametrize(
    ("num_steps", "expected_tf_steps", "expected_video_steps"),
    (
        (2, 1, 1),
        (3, 2, 1),
        (5, 2, 3),
        (7, 4, 3),
    ),
)
def test_strict_cascade_odd_nfe_split_is_deterministic(
    monkeypatch,
    num_steps,
    expected_tf_steps,
    expected_video_steps,
):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "tf_first_cascaded"
    model.cascade_inference_tf_fraction = 0.5
    model.sample_scheduler = NativeScheduler()

    schedule, timesteps, tf_only_steps = model._sampling_schedule(
        num_steps,
        device=torch.device("cpu"),
    )

    assert tf_only_steps == expected_tf_steps
    assert timesteps.numel() == num_steps
    assert schedule.num_steps == num_steps
    assert model.sample_scheduler.timesteps.numel() == expected_video_steps


def test_current_aligned_sampling_schedule_is_unchanged(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "aligned"
    model.tf_lead_logit = 1.0
    model.sample_scheduler = NativeScheduler()

    schedule, timesteps, tf_only_steps = model._sampling_schedule(
        4,
        device=torch.device("cpu"),
    )

    assert tf_only_steps == 0
    torch.testing.assert_close(schedule.video, model.sample_scheduler.sigmas)
    torch.testing.assert_close(schedule.time_frequency, schedule.video)
    torch.testing.assert_close(timesteps, model.sample_scheduler.timesteps)


def test_current_tf_leads_sampling_preserves_native_video_schedule(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "tf_leads"
    model.tf_lead_logit = 1.0
    model.sample_scheduler = NativeScheduler()

    schedule, timesteps, tf_only_steps = model._sampling_schedule(
        4,
        device=torch.device("cpu"),
    )

    assert tf_only_steps == 0
    torch.testing.assert_close(
        schedule.video,
        model.sample_scheduler.sigmas,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(timesteps, model.sample_scheduler.timesteps)
    assert schedule.time_frequency[0] == 1
    assert schedule.time_frequency[-1] == 0
    assert torch.all(
        schedule.time_frequency[1:-1] < schedule.video[1:-1]
    )


def test_cascade_training_clocks_select_exactly_one_branch(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.training = True
    model.tf_schedule_mode = "tf_first_cascaded"
    model.cascade_logit_mean = 0.0
    model.cascade_logit_std = 1.0
    model._cascade_clock_sampler = module.DualClockSampler(
        mode="tf_first_cascaded_noised",
        tf_loss_probability=0.4,
        tf_condition_max_sigma=0.25,
    )
    model.noise_scheduler = types.SimpleNamespace(
        sigmas=torch.linspace(1, 0, 101),
        timesteps=torch.arange(100, 0, -1),
    )
    torch.manual_seed(20260726)

    (
        timesteps,
        video_sigma,
        tf_sigma,
        video_weight,
        tf_weight,
    ) = model._paired_training_clocks(256, torch.device("cpu"), torch.float32)

    torch.testing.assert_close(
        video_weight + tf_weight,
        torch.ones_like(video_weight),
    )
    tf_examples = tf_weight.bool()
    video_examples = video_weight.bool()
    assert tf_examples.any() and video_examples.any()
    assert torch.all(video_sigma[tf_examples] == 1)
    assert torch.all(tf_sigma[video_examples] <= 0.25)
    assert torch.all(timesteps[tf_examples] == 100)
    expected_native = model.noise_scheduler.sigmas[
        100 - timesteps.to(torch.int64)
    ]
    torch.testing.assert_close(video_sigma, expected_native)


def test_cascade_video_clock_uses_native_shifted_wan_index_law(monkeypatch):
    module = _load_model_module(monkeypatch)
    scheduler_module = pytest.importorskip(
        "diffusers.schedulers.scheduling_flow_match_euler_discrete"
    )
    scheduler = scheduler_module.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.noise_scheduler = scheduler
    model.cascade_logit_mean = 0.0
    model.cascade_logit_std = 1.0
    batch_size = 4096
    seed = 20260726

    torch.manual_seed(seed)
    timesteps, video_sigma = model._sample_native_cascade_video_clocks(
        batch_size,
        device=torch.device("cpu"),
    )

    torch.manual_seed(seed)
    schedule_fraction = torch.sigmoid(
        torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size,),
        )
    )
    expected_indices = (
        schedule_fraction * scheduler.timesteps.numel()
    ).long().clamp(0, scheduler.timesteps.numel() - 1)
    torch.testing.assert_close(
        timesteps,
        scheduler.timesteps[expected_indices],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        video_sigma,
        scheduler.sigmas[expected_indices],
        rtol=0,
        atol=0,
    )
    assert video_sigma.median() > 0.8


def test_cascade_logit_mean_keeps_production_clean_noise_direction(monkeypatch):
    module = _load_model_module(monkeypatch)
    scheduler_module = pytest.importorskip(
        "diffusers.schedulers.scheduling_flow_match_euler_discrete"
    )
    scheduler = scheduler_module.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.noise_scheduler = scheduler
    model.cascade_logit_std = 0.5
    batch_size = 4096

    model.cascade_logit_mean = -1.0
    torch.manual_seed(91)
    _, negative_mean_sigma = model._sample_native_cascade_video_clocks(
        batch_size,
        device=torch.device("cpu"),
    )
    model.cascade_logit_mean = 1.0
    torch.manual_seed(91)
    _, positive_mean_sigma = model._sample_native_cascade_video_clocks(
        batch_size,
        device=torch.device("cpu"),
    )

    assert positive_mean_sigma.mean() < negative_mean_sigma.mean()


def test_cascade_validation_clock_is_deterministic_teacher_forced(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.training = False
    model.tf_schedule_mode = "tf_first_cascaded"
    model.cascade_validation_tf_sigma = 0.125
    model.validation_video_sigmas = (0.9, 0.5, 0.25)
    model.noise_scheduler = types.SimpleNamespace(
        sigmas=torch.tensor([1.0, 0.9, 0.5, 0.25, 0.0]),
        timesteps=torch.tensor([100, 90, 50, 25]),
    )

    _, video_sigma, tf_sigma, video_weight, tf_weight = (
        model._paired_training_clocks(
            5,
            torch.device("cpu"),
            torch.float32,
        )
    )

    torch.testing.assert_close(
        video_sigma,
        torch.tensor([0.9, 0.5, 0.25, 0.9, 0.5]),
    )
    torch.testing.assert_close(tf_sigma, torch.full((5,), 0.125))
    assert torch.all(video_weight == 1)
    assert torch.all(tf_weight == 0)

    video_per_sample = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tf_per_sample = torch.full((5,), 1_000.0)
    video_loss = model._branch_weighted_mean(
        video_per_sample,
        video_weight,
    )
    tf_loss = model._branch_weighted_mean(
        tf_per_sample,
        tf_weight,
    )
    assert video_loss == video_per_sample.mean()
    assert tf_loss == 0
    assert video_loss + 7.0 * tf_loss == video_per_sample.mean()


def test_branch_weighted_mean_preserves_default_and_masks_cascade(monkeypatch):
    module = _load_model_module(monkeypatch)
    values = torch.tensor([1.0, 3.0, 5.0, 7.0])

    default = module.DualExplicitActionDiTModel._branch_weighted_mean(
        values,
        torch.ones(4),
    )
    selected = module.DualExplicitActionDiTModel._branch_weighted_mean(
        values,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )

    assert default == values.mean()
    assert selected == torch.tensor((1.0 + 5.0) / 4.0)


def test_branch_weighted_mean_zeroes_inactive_per_sample_gradients(monkeypatch):
    module = _load_model_module(monkeypatch)
    video_values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        requires_grad=True,
    )
    tf_values = torch.tensor(
        [5.0, 6.0, 7.0, 8.0],
        requires_grad=True,
    )
    video_weight = torch.tensor([1.0, 0.0, 1.0, 0.0])
    tf_weight = 1.0 - video_weight

    total = module.DualExplicitActionDiTModel._branch_weighted_mean(
        video_values,
        video_weight,
    ) + 2.0 * module.DualExplicitActionDiTModel._branch_weighted_mean(
        tf_values,
        tf_weight,
    )
    total.backward()

    torch.testing.assert_close(
        video_values.grad,
        torch.tensor([0.25, 0.0, 0.25, 0.0]),
    )
    torch.testing.assert_close(
        tf_values.grad,
        torch.tensor([0.0, 0.5, 0.0, 0.5]),
    )
