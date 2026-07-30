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

    model.num_history_latent = 2
    model.num_history_frames = 5
    model.num_future_frames = 2
    model.auxiliary_history_mode = "diffuse_all"
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
    model._history_reference = lambda _rgb, _shape: (
        torch.zeros_like(video_clean),
        2,
    )
    model._tf_clean = lambda *_args: (_ for _ in ()).throw(
        AssertionError("autonomous diffuse-all sampling invoked its teacher")
    )
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
        tf_channels=12,
        effective_gate=lambda: torch.tensor(0.1)
    )
    model.forward_model.tf_clock_embedding = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.0)
    )
    rgb = torch.zeros(batch_size, 13, 3, 2, 2)

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


def test_historical_training_tf_noise_retains_global_rng_behavior(monkeypatch):
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


def test_cached_target_noise_is_stateless_and_rng_independent(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.video_only_control = False
    model.parameter_matched_control = False
    clean = torch.ones(2, 64, 4, 2, 2)
    sample_ids = torch.tensor([17, 29])
    timesteps = torch.tensor([300, 700])

    torch.manual_seed(20260729)
    before = torch.random.get_rng_state()
    first = model._training_tf_noise(
        clean, sample_ids=sample_ids, timesteps=timesteps
    )
    after = torch.random.get_rng_state()
    second = model._training_tf_noise(
        clean, sample_ids=sample_ids, timesteps=timesteps
    )

    assert torch.equal(before, after)
    assert torch.equal(after, torch.random.get_rng_state())
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


def test_cascade_clock_sampling_is_stateless_and_branch_masked(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.training = True
    model.tf_schedule_mode = "tf_first_cascaded"
    model._cascade_clock_sampler = module.DualClockSampler(
        mode="tf_first_cascaded_noised",
        tf_loss_probability=0.5,
        tf_condition_max_sigma=0.25,
    )
    model.noise_scheduler = types.SimpleNamespace(
        timesteps=torch.tensor([999, 700, 300, 0])
    )
    model._sample_timesteps = lambda batch, device: torch.tensor(
        [700, 300], device=device
    )
    model._get_sigmas = (
        lambda timesteps, n_dim, dtype, device: torch.tensor(
            [[0.7], [0.3]], device=device, dtype=dtype
        )
    )
    sample_ids = torch.tensor([17, 29])

    torch.manual_seed(20260730)
    before = torch.random.get_rng_state()
    first = model._paired_training_clocks(
        2,
        torch.device("cpu"),
        torch.float32,
        sample_ids=sample_ids,
    )
    after = torch.random.get_rng_state()
    second = model._paired_training_clocks(
        2,
        torch.device("cpu"),
        torch.float32,
        sample_ids=sample_ids,
    )

    assert torch.equal(before, after)
    assert torch.equal(after, torch.random.get_rng_state())
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    (
        model_timesteps,
        video_sigma,
        tf_sigma,
        video_weight,
        tf_weight,
        noise_timesteps,
    ) = first
    torch.testing.assert_close(
        video_weight + tf_weight,
        torch.ones_like(video_weight),
    )
    assert torch.all(video_sigma[tf_weight.bool()] == 1)
    assert torch.all(tf_sigma[video_weight.bool()] <= 0.25)
    torch.testing.assert_close(
        noise_timesteps,
        torch.tensor([700, 300]),
        rtol=0,
        atol=0,
    )
    if bool(tf_weight.any()):
        assert torch.all(
            model_timesteps[tf_weight.bool()]
            == model.noise_scheduler.timesteps[0]
        )
        assert torch.any(
            noise_timesteps[tf_weight.bool()]
            != model_timesteps[tf_weight.bool()]
        )


def test_strict_cascade_uses_total_wan_calls_and_native_video_nodes(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "tf_first_cascaded"
    model.cascade_inference_tf_fraction = 0.5

    class Scheduler:
        def set_timesteps(self, num_steps, device):
            assert num_steps == 3
            self.timesteps = torch.tensor(
                [900.0, 450.0, 0.0], device=device
            )
            self.sigmas = torch.tensor(
                [1.0, 0.65, 0.20, 0.0], device=device
            )

    model.sample_scheduler = Scheduler()
    schedule, model_timesteps, tf_only_steps = model._sampling_schedule(
        6,
        device=torch.device("cpu"),
    )

    assert tf_only_steps == 3
    assert model_timesteps.numel() == 6
    torch.testing.assert_close(
        schedule.video[3:],
        torch.tensor([1.0, 0.65, 0.20, 0.0]),
    )
    assert torch.all(schedule.video[:3] == 1)
    assert torch.all(schedule.time_frequency[3:] == 0)


def test_cascade_controls_intervene_only_after_auxiliary_generation(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.tf_schedule_mode = "tf_first_cascaded"

    for source in ("off", "autonomous_shuffled"):
        assert (
            model._sampling_condition_source_for_step(
                source,
                step_index=0,
                tf_only_steps=2,
            )
            == "autonomous"
        )
        assert (
            model._sampling_condition_source_for_step(
                source,
                step_index=1,
                tf_only_steps=2,
            )
            == "autonomous"
        )
        assert (
            model._sampling_condition_source_for_step(
                source,
                step_index=2,
                tf_only_steps=2,
            )
            == source
        )

    assert (
        model._sampling_condition_source_for_step(
            "oracle_matched",
            step_index=0,
            tf_only_steps=2,
        )
        == "oracle_matched"
    )
    model.tf_schedule_mode = "aligned"
    assert (
        model._sampling_condition_source_for_step(
            "off",
            step_index=0,
            tf_only_steps=0,
        )
        == "off"
    )


def test_evaluation_noise_is_keyed_by_clip_not_batch_order(monkeypatch):
    module = _load_model_module(monkeypatch)
    ids = torch.tensor([17, 29])
    first = module.DualExplicitActionDiTModel._evaluation_noise(
        (2, 3, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        base_seed=20260729,
        sample_ids=ids,
        stream=1,
        rank=0,
    )
    reversed_batch = module.DualExplicitActionDiTModel._evaluation_noise(
        (2, 3, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        base_seed=20260729,
        sample_ids=ids.flip(0),
        stream=1,
        rank=7,
    )

    assert torch.equal(first[0], reversed_batch[1])
    assert torch.equal(first[1], reversed_batch[0])
    assert not torch.equal(first[0], first[1])


def test_parameter_matched_control_preserves_rng_and_video_objective(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    model.video_only_control = False
    model.parameter_matched_control = True
    clean = torch.ones(2, 64, 4, 2, 2)

    torch.manual_seed(20260729)
    before = torch.random.get_rng_state()
    placeholder = model._training_tf_noise(clean)
    after = torch.random.get_rng_state()
    assert torch.count_nonzero(placeholder) == 0
    assert torch.equal(before, after)

    video_loss = torch.tensor(2.0, requires_grad=True)
    auxiliary_loss = torch.tensor(float("nan"), requires_grad=True)
    objective = model._training_objective(video_loss, auxiliary_loss)
    objective.backward()
    assert objective is video_loss
    assert auxiliary_loss.grad is None


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
    model.num_history_frames = 5
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
    model._history_reference = lambda _rgb, _shape: (
        torch.zeros_like(video_clean),
        2,
    )
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
        tf_channels=1,
        effective_gate=lambda: torch.tensor(0.1)
    )
    model.forward_model.tf_clock_embedding = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.0)
    )
    rgb = torch.zeros(batch_size, 13, 3, 2, 2)

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
    assert artifacts["online_teacher_call_count"].item() == 0
    assert artifacts["wan_call_count_nfe_2"].item() == 2
    assert (
        artifacts["wan_call_count_autonomous_shuffled_nfe_2"].item() == 2
    )


def test_deployable_sampler_uses_history_only_and_matches_paired_path(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    batch_size = 1
    history_latents = torch.full((batch_size, 16, 2, 1, 1), 0.25)
    full_latents = torch.cat(
        [history_latents, torch.full((batch_size, 16, 2, 1, 1), 0.75)],
        dim=2,
    )

    model.num_history_frames = 5
    model.num_future_frames = 8
    model.num_history_latent = 2
    model.auxiliary_history_mode = "diffuse_all"
    model.tf_condition_mode = "matched"
    model.condition_on_tf = True
    model.condition_on_tf_clock = True
    model.video_only_control = False
    model.parameter_matched_control = False
    model.tf_loss_weight = 1.0
    model.tf_schedule_mode = "aligned"
    model.tf_lead_logit = 0.0
    model.evaluation_noise_seed = 99
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (1,)
    model.viz_num_steps = 1
    model.capture_latent_trajectories = False
    model._visualization_artifacts = None
    model._video_only_runtime_validated = True

    encoded_frame_counts = []

    def encode(rgb):
        encoded_frame_counts.append(int(rgb.shape[1]))
        return history_latents if rgb.shape[1] == 5 else full_latents

    model._encode_clip = encode
    model._latent_actions = (
        lambda rgb, _actions, _morphology, latent_frames, history_frames: (
            None,
            torch.zeros(rgb.shape[0], latent_frames, 1),
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
        temporal_ratio = 4

        @staticmethod
        def latent_temporal_len(frames):
            return (frames - 1) // 4 + 1

        @staticmethod
        def decode_temporal(latent, out_hw):
            # Wan's 4 latent positions decode to 13 frames with temporal
            # multiplicities [1,4,4,4].
            decoded = torch.repeat_interleave(
                latent[:, :1],
                torch.tensor([1, 4, 4, 4]),
                dim=2,
            )
            return decoded.expand(-1, 3, -1, *out_hw)

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
        condition_on_tf_clock,
    ):
        del conditioning_tf, tf_sigma, condition_on_tf, condition_on_tf_clock
        return module.DualWanOutput(
            video_velocity=torch.zeros_like(video_state),
            tf_velocity=torch.zeros_like(noisy_tf),
        )

    model.rgb_tokenizer = Tokenizer()
    model.sample_scheduler = Scheduler()
    model.forward_model = forward_model
    model.forward_model.tf_token_adapter = types.SimpleNamespace(
        tf_channels=1,
        effective_gate=lambda: torch.tensor(0.0),
    )
    model.forward_model.tf_clock_embedding = types.SimpleNamespace(
        effective_gate=lambda: torch.tensor(0.0),
    )

    full_rgb = torch.zeros(batch_size, 13, 3, 2, 2)
    history_rgb = full_rgb[:, :5]
    actions = torch.zeros(batch_size, 13, 5, 23)

    paired, ground_truth = model._sample_future(
        full_rgb,
        actions,
        collect_artifacts=False,
    )
    deployable = model.sample_future_deployable(
        history_rgb,
        actions,
        collect_artifacts=True,
    )

    assert ground_truth is not None
    torch.testing.assert_close(
        deployable,
        paired[:, :, -model.num_future_frames :],
        rtol=0,
        atol=0,
    )
    assert encoded_frame_counts == [13, 5, 5]
    counters = model._last_sampling_counters
    assert counters["deployment_mode"] == 1
    assert counters["auxiliary_clean_available"] == 0
    artifacts = model.pop_visualization_artifacts()
    assert artifacts["deployment_mode"].item() == 1
    assert "video_trajectory" not in artifacts
    assert "tf_trajectory" not in artifacts
    assert "video_clean" not in artifacts
    assert "ground_truth_future_uint8" not in artifacts
    assert "tf_clean" not in artifacts
    # The observed prefix reaches the clean reference at sigma=0.
    torch.testing.assert_close(
        artifacts["video_final_nfe_1"][:, :, :2],
        history_latents.to(torch.float16),
        rtol=0,
        atol=0,
    )

    with pytest.raises(ValueError, match="exactly the observed history"):
        model.sample_future_deployable(full_rgb, actions)
    with pytest.raises(ValueError, match="clean auxiliary target"):
        model._sample_future(
            history_rgb,
            actions,
            auxiliary_target=torch.zeros(1, 1, 4, 1, 1),
            deployment_mode=True,
        )
