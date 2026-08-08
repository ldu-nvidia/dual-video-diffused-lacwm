from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from tools import causal_vjepa2_temporal_targets as temporal


def _normalization(
    *, anchor_mean: float = 0.0, anchor_std: float = 1.0,
    delta_mean: float = 0.0, delta_std: float = 1.0,
) -> temporal.TemporalNormalization:
    return temporal.TemporalNormalization(
        anchor_mean=torch.full((temporal.CHANNELS,), anchor_mean),
        anchor_std=torch.full((temporal.CHANNELS,), anchor_std),
        delta_mean=torch.full((temporal.CHANNELS,), delta_mean),
        delta_std=torch.full((temporal.CHANNELS,), delta_std),
        provenance={"unit_test": True},
    )


def _batch(batch: int = 2):
    generator = torch.Generator().manual_seed(19)
    return {
        "history": torch.randn(batch, 3, 5, 4, 4, generator=generator),
        "future": torch.randn(batch, 3, 8, 4, 4, generator=generator),
        "actions": torch.randn(batch, 16, 7, generator=generator),
        "auxiliary_target": torch.randn(
            batch, *temporal.TARGET_SHAPE, generator=generator
        ),
    }


class _ZeroAuxiliaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.actions = []

    def forward(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        *,
        auxiliary_fusion_mask,
        predict_video,
    ):
        del t_video, t_auxiliary, history
        assert auxiliary_fusion_mask is True
        assert predict_video is False
        self.calls += 1
        self.actions.append(actions.detach().clone())
        return SimpleNamespace(
            video_x=torch.zeros_like(noisy_video),
            auxiliary_x=torch.zeros_like(noisy_auxiliary),
        )


class _ConstantRngConsumingModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))
        self.batch_sizes = []
        self.noisy_auxiliary_inputs = []
        self.auxiliary_times = []

    def forward(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        *,
        auxiliary_fusion_mask,
        predict_video,
    ):
        del t_video, history, actions, auxiliary_fusion_mask
        assert predict_video is False
        self.batch_sizes.append(noisy_auxiliary.shape[0])
        self.noisy_auxiliary_inputs.append(noisy_auxiliary.detach().clone())
        self.auxiliary_times.append(t_auxiliary.detach().clone())
        torch.rand(3)  # the stopped call must restore this consumed RNG state
        return SimpleNamespace(
            video_x=torch.zeros_like(noisy_video),
            auxiliary_x=self.value.expand_as(noisy_auxiliary),
        )


def test_delta_pack_is_exactly_invertible_before_and_after_normalization():
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(2, *temporal.TARGET_SHAPE, generator=generator)
    packed = temporal.temporal_delta_pack(target)
    torch.testing.assert_close(
        temporal.invert_temporal_delta_pack(packed), target, rtol=0, atol=2e-6
    )

    normalization = temporal.TemporalNormalization(
        anchor_mean=torch.linspace(-1.0, 1.0, temporal.CHANNELS),
        anchor_std=torch.linspace(0.5, 2.0, temporal.CHANNELS),
        delta_mean=torch.linspace(-0.2, 0.2, temporal.CHANNELS),
        delta_std=torch.linspace(0.1, 1.1, temporal.CHANNELS),
        provenance={"unit_test": True},
    )
    encoded = normalization.encode(target, "delta_pack")
    decoded = normalization.decode(encoded, "delta_pack")
    torch.testing.assert_close(decoded, target, rtol=2e-6, atol=3e-6)
    assert float((decoded - target).abs().max()) <= 2e-5


def test_temporal_velocity_loss_uses_train_delta_scale():
    shape = (1, *temporal.TARGET_SHAPE)
    noisy = torch.zeros(shape)
    clean = torch.zeros(shape)
    prediction = torch.zeros(shape)
    ramp = 2.0 * torch.arange(8, dtype=torch.float32)
    prediction[:] = ramp.reshape(1, 1, 8, 1, 1)
    result = temporal.per_example_normalized_temporal_velocity_mse(
        prediction,
        noisy,
        clean,
        torch.zeros(1),
        target_mode="absolute",
        normalization=_normalization(delta_std=2.0),
    )
    torch.testing.assert_close(result, torch.ones(1))

    # In normalized delta-pack coordinates, unit error in tokens 1..7 is also
    # unit normalized temporal-velocity error in the original semantic space.
    delta_prediction = torch.zeros(shape)
    delta_prediction[:, :, 1:] = 1.0
    delta_result = temporal.per_example_normalized_temporal_velocity_mse(
        delta_prediction,
        noisy,
        clean,
        torch.zeros(1),
        target_mode="delta_pack",
        normalization=_normalization(delta_std=2.0),
    )
    torch.testing.assert_close(delta_result, torch.ones(1))


def test_action_margin_loss_has_expected_sign_and_zero_region():
    positive = torch.tensor([0.2, 0.5, 0.7])
    shuffled = torch.tensor([0.5, 0.4, 0.7])
    result = temporal.per_example_action_margin_loss(
        positive, shuffled, margin=0.1
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 0.2, 0.1]))
    with pytest.raises(ValueError, match="margin"):
        temporal.per_example_action_margin_loss(positive, shuffled, margin=0.0)


def test_default_training_step_delegates_to_frozen_baseline_exactly(monkeypatch):
    sentinel_loss = torch.tensor(4.25)
    sentinel_telemetry = {"baseline": torch.tensor(7.0)}
    calls = []

    def baseline(model, batch):
        calls.append((model, batch))
        return sentinel_loss, sentinel_telemetry

    monkeypatch.setattr(temporal.screen, "semantic_training_step", baseline)
    model = object()
    batch = _batch()
    loss, telemetry = temporal.temporal_target_training_step(model, batch)
    assert loss is sentinel_loss
    assert telemetry is sentinel_telemetry
    assert calls == [(model, batch)]


def test_action_variant_uses_same_corruption_and_only_rolls_actions():
    batch = _batch()
    model = _ZeroAuxiliaryModel()
    torch.manual_seed(23)
    loss, telemetry = temporal.temporal_target_training_step(
        model,
        batch,
        target_mode="absolute",
        normalization=_normalization(),
        temporal_velocity_loss_weight=1.0,
        action_margin_loss_weight=0.5,
        action_margin=0.2,
    )
    assert model.calls == 1
    torch.testing.assert_close(model.actions[0][:2], batch["actions"])
    torch.testing.assert_close(model.actions[0][2:], batch["actions"].roll(1, 0))
    assert bool(torch.isfinite(loss))
    assert float(telemetry["action_margin_loss"]) >= 0.0
    assert set(telemetry) == {
        "auxiliary_loss",
        "flow_loss",
        "normalized_temporal_velocity_loss",
        "shuffled_temporal_velocity_loss",
        "action_margin_loss",
        "combined_auxiliary_loss",
        "weighted_auxiliary_loss",
        "auxiliary_branch_count",
        "self_rollin_selected_count",
        "self_rollin_selected_fraction",
        "self_rollin_initial_time_mean",
        "self_rollin_time_advance_mean",
        "self_rollin_initial_time_sum",
        "self_rollin_time_advance_sum",
        "self_rollin_no_grad_model_calls",
        "gradient_model_calls",
        "total_model_calls",
    }


def test_counter_hash_rollin_rng_is_stable_and_does_not_consume_torch_rng():
    ids = torch.tensor([101, 102, 103, 104])
    torch.manual_seed(91)
    before = torch.get_rng_state().clone()
    first = temporal.self_rollin_counter_uniforms(
        seed=1234, update=7, global_sample_ids=ids, device=torch.device("cpu")
    )
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    second = temporal.self_rollin_counter_uniforms(
        seed=1234, update=7, global_sample_ids=ids, device=torch.device("cpu")
    )
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert bool(((first[0] > 0) & (first[0] < 1)).all())
    changed = temporal.self_rollin_counter_uniforms(
        seed=1234, update=8, global_sample_ids=ids, device=torch.device("cpu")
    )
    assert not torch.equal(first[0], changed[0])


def test_stopped_rollin_obeys_t0_to_t1_law_and_restores_model_rng():
    batch_size = 2
    shape = (batch_size, *temporal.TARGET_SHAPE)
    clean = torch.ones(shape)
    noise = torch.zeros(shape)
    final_time = torch.full((batch_size,), 0.8)
    true_final = temporal.vlf.corrupt_clean_time(clean, noise, final_time)
    model = _ConstantRngConsumingModel(0.6)
    torch.manual_seed(122)
    rng_before = torch.get_rng_state().clone()
    result = temporal.stopped_one_step_self_rollin(
        model,
        clean=clean,
        noise=noise,
        true_noisy_at_final_time=true_final,
        final_time=final_time,
        noisy_video=torch.zeros(batch_size, 3, 8, 4, 4),
        history=torch.zeros(batch_size, 3, 5, 4, 4),
        actions=torch.zeros(batch_size, 16, 7),
        probability=1.0,
        time_rule="fixed_final_clock_fraction",
        fixed_final_clock_fraction=0.25,
        seed=1234,
        update=3,
        global_sample_ids=torch.tensor([512, 513]),
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    torch.testing.assert_close(result.initial_time, torch.full((2,), 0.2))
    torch.testing.assert_close(result.final_time, final_time)
    # z0=.2, xhat=.6, t1-t0=.6: zR=.2+.6*(.6-.2)/(1-.2)=.5
    torch.testing.assert_close(
        result.noisy_at_final_time,
        torch.full_like(result.noisy_at_final_time, 0.5),
    )
    assert result.no_grad_model_calls == 1
    assert model.batch_sizes == [2]
    assert result.noisy_at_final_time.grad_fn is None


def test_delta_r_gradient_forward_uses_mixed_state_at_unchanged_final_clock(monkeypatch):
    batch = _batch(batch=2)
    time_values = torch.arange(1, 9, dtype=torch.float32)
    batch["auxiliary_target"] = time_values.reshape(1, 1, 8, 1, 1).expand(
        2, *temporal.TARGET_SHAPE
    ).clone()
    clocks = temporal.vlf.TrainingClocks(
        video_time=torch.zeros(2),
        auxiliary_time=torch.full((2,), 0.8),
        video_loss_mask=torch.zeros(2),
        auxiliary_loss_mask=torch.ones(2),
        auxiliary_condition_mask=torch.ones(2, dtype=torch.bool),
    )
    monkeypatch.setattr(temporal.vlf, "sample_training_clocks", lambda *args: clocks)
    monkeypatch.setattr(temporal.torch, "randn_like", lambda value: torch.zeros_like(value))
    model = _ConstantRngConsumingModel(0.6)
    loss, telemetry = temporal.temporal_target_training_step(
        model,
        batch,
        target_mode="delta_pack",
        normalization=_normalization(),
        self_rollin_probability=1.0,
        self_rollin_later_time_rule="fixed_final_clock_fraction",
        self_rollin_fixed_final_clock_fraction=0.25,
        rollin_seed=1234,
        rollin_update=1,
        rollin_global_sample_ids=torch.tensor([0, 1]),
    )
    assert model.batch_sizes == [2, 2]
    torch.testing.assert_close(model.auxiliary_times[0], torch.full((2,), 0.2))
    torch.testing.assert_close(model.auxiliary_times[1], torch.full((2,), 0.8))
    torch.testing.assert_close(
        model.noisy_auxiliary_inputs[1],
        torch.full_like(model.noisy_auxiliary_inputs[1], 0.5),
    )
    assert float(telemetry["self_rollin_selected_count"]) == 2
    assert float(telemetry["self_rollin_no_grad_model_calls"]) == 1
    assert float(telemetry["gradient_model_calls"]) == 1
    assert float(telemetry["total_model_calls"]) == 2
    loss.backward()
    assert model.value.grad is not None and bool(torch.isfinite(model.value.grad))


def test_sampler_api_cannot_accept_clean_future_or_teacher_and_decodes_pack():
    parameters = inspect.signature(temporal.sample_temporal_target).parameters
    assert not {"target", "clean", "future", "teacher"}.intersection(parameters)
    batch = _batch()
    model = _ZeroAuxiliaryModel()
    normalization = _normalization(anchor_mean=3.0, delta_mean=2.0)
    result = temporal.sample_temporal_target(
        model,
        batch["history"],
        batch["actions"],
        video_noise=torch.randn_like(batch["future"]),
        auxiliary_noise=torch.randn_like(batch["auxiliary_target"]),
        steps=1,
        target_mode="delta_pack",
        normalization=normalization,
    )
    assert result.model_calls == 1
    torch.testing.assert_close(
        result.representation_prediction,
        torch.zeros_like(result.representation_prediction),
    )
    expected_time = 3.0 + 2.0 * torch.arange(8, dtype=torch.float32)
    expected = expected_time.reshape(1, 1, 8, 1, 1).expand_as(
        result.semantic_prediction
    )
    torch.testing.assert_close(result.semantic_prediction, expected)


def test_parser_requires_normalization_only_for_nonbaseline_variants():
    base = [
        "train",
        "--artifact-root", "/tmp/artifacts",
        "--run-id", "unit",
        "--data-root", "/tmp/data",
        "--semantic-cache-root", "/tmp/cache",
        "--train-manifest", "/tmp/train.jsonl",
        "--validation-manifest", "/tmp/val.jsonl",
    ]
    parser = temporal.build_parser()
    absolute = parser.parse_args(base)
    temporal.validate_args(absolute)
    assert temporal._doe_arm(absolute) == "ABS"
    delta = parser.parse_args([*base, "--target-mode", "delta_pack"])
    with pytest.raises(temporal.TemporalTargetError, match="normalization"):
        temporal.validate_args(delta)
    valid_delta = parser.parse_args(
        [*base, "--target-mode", "delta_pack", "--normalization", "/tmp/stats.json"]
    )
    temporal.validate_args(valid_delta)
    assert temporal._doe_arm(valid_delta) == "DELTA"
    delta_rollin = parser.parse_args(
        [
            *base,
            "--target-mode", "delta_pack",
            "--normalization", "/tmp/stats.json",
            "--self-rollin-probability", "0.5",
        ]
    )
    temporal.validate_args(delta_rollin)
    assert temporal._doe_arm(delta_rollin) == "DELTA-R"
    invalid_absolute_rollin = parser.parse_args(
        [
            *base,
            "--normalization", "/tmp/stats.json",
            "--self-rollin-probability", "0.5",
        ]
    )
    with pytest.raises(temporal.TemporalTargetError, match="delta_pack"):
        temporal.validate_args(invalid_absolute_rollin)
