"""Exact algebra, boundary, mask, and EMA tests for ACD-P0."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from robot_wm.modeling.low_nfe.adjacent_consistency import (
    AdjacentConsistencyError,
    CANONICAL_MODEL_STATE_HASH_ALGORITHM,
    RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
    adjacent_clock_pair,
    canonical_model_state_sha256,
    consistency_output,
    ema_update_module_,
    karras_boundary_scalings,
    masked_pseudo_huber,
    model_state_hash_receipt,
    module_state_receipt,
    predicted_clean,
    rectified_flow_euler_step,
    rectified_flow_noisy_state,
    require_model_state_hashes,
    restore_history_forward_path,
    validate_shift5_rf_clock,
    tensor_state_sha256,
)


def _state(value: float = 0.0) -> torch.Tensor:
    return torch.full((2, 3, 4, 2, 2), value)


def test_clock_and_rectified_flow_sign_are_noise_to_clean() -> None:
    sigmas = torch.linspace(1.0, 0.0, 1000)
    timesteps = torch.linspace(1000.0, 1.0, 1000)
    pair = adjacent_clock_pair(sigmas, timesteps, start_index=100, stride=500)
    assert pair.end_index == 600
    assert pair.end_sigma < pair.start_sigma

    clean = _state(2.0)
    noise = _state(-1.0)
    source = rectified_flow_noisy_state(clean, noise, pair.start_sigma)
    exact_velocity = noise - clean
    stepped = rectified_flow_euler_step(
        source, exact_velocity, pair.start_sigma, pair.end_sigma
    )
    expected = rectified_flow_noisy_state(clean, noise, pair.end_sigma)
    torch.testing.assert_close(stepped, expected)
    with pytest.raises(AdjacentConsistencyError, match="next_sigma < sigma"):
        rectified_flow_euler_step(source, exact_velocity, 0.25, 0.5)


def test_shift5_training_clock_has_exact_timestep_relation() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=5.0
    )
    receipt = validate_shift5_rf_clock(scheduler)
    assert receipt.train_nodes == receipt.training_sigma_nodes == 1000
    assert receipt.shift == 5.0
    assert receipt.timestep_sigma_relation_exact


def test_karras_boundary_is_exact_identity_at_clean_endpoint() -> None:
    noisy = torch.randn(2, 3, 4, 2, 2)
    velocity = torch.randn_like(noisy)
    c_skip, c_out = karras_boundary_scalings(0.0, sigma_data=0.5)
    assert float(c_skip) == 1.0
    assert float(c_out) == 0.0
    assert torch.equal(
        consistency_output(noisy, velocity, 0.0, sigma_data=0.5), noisy
    )


def test_consistency_map_is_not_ordinary_x0_away_from_endpoint() -> None:
    noisy = _state(1.5)
    velocity = _state(0.25)
    ordinary = predicted_clean(noisy, velocity, 0.8)
    boundary = consistency_output(noisy, velocity, 0.8, sigma_data=0.5)
    assert not torch.equal(boundary, ordinary)


def test_history_reclamp_never_writes_future() -> None:
    state = _state(7.0)
    reference = torch.zeros_like(state)
    reference[:, :, :2] = 3.0
    noise = _state(-2.0)
    restored = restore_history_forward_path(
        state, reference, noise, 0.25, history_tokens=2
    )
    torch.testing.assert_close(restored[:, :, :2], _state(1.75)[:, :, :2])
    assert torch.equal(restored[:, :, 2:], state[:, :, 2:])
    leaking = reference.clone()
    leaking[:, :, 2:] = 1
    with pytest.raises(AdjacentConsistencyError, match="future support"):
        restore_history_forward_path(
            state, leaking, noise, 0.25, history_tokens=2
        )


def test_pseudo_huber_is_future_masked_and_target_stopped() -> None:
    prediction = torch.zeros(2, 3, 2, 2, 2, requires_grad=True)
    target = torch.ones_like(prediction, requires_grad=True)
    mask = torch.zeros(2, 1, 2, 1, 1)
    mask[:, :, 1:] = 1
    loss, per_sample = masked_pseudo_huber(
        prediction, target, mask, c=0.001
    )
    assert tuple(per_sample.shape) == (2,)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[:, :, :1]) == 0
    assert target.grad is None


class _TinyState(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value, value + 1]))
        self.frozen = nn.Parameter(
            torch.tensor([value + 7.1234567]), requires_grad=False
        )
        self.register_buffer("floating", torch.tensor([value + 2]))
        self.register_buffer("integer", torch.tensor([int(value)], dtype=torch.int64))


def test_ema_updates_floats_copies_integer_and_preserves_source() -> None:
    source = _TinyState(4.0)
    target = _TinyState(0.0).requires_grad_(False)
    source_before = {key: value.clone() for key, value in source.state_dict().items()}
    receipt = ema_update_module_(target, source, decay=0.75)
    torch.testing.assert_close(target.weight, torch.tensor([1.0, 2.0]))
    assert torch.equal(target.frozen, source.frozen)
    assert torch.equal(target.floating, source.floating)
    assert int(target.integer) == 4
    assert receipt.ema_parameter_tensors == 1
    assert receipt.copied_parameter_tensors == 1
    assert receipt.copied_floating_buffer_tensors == 1
    assert receipt.copied_nonfloating_buffer_tensors == 1
    assert all(
        torch.equal(source.state_dict()[key], value)
        for key, value in source_before.items()
    )


def test_frozen_parameter_and_buffer_remain_bit_exact_across_many_ema_updates() -> None:
    source = _TinyState(1.234567)
    target = _TinyState(-9.0).requires_grad_(False)
    for index in range(100):
        with torch.no_grad():
            source.weight.add_(0.001 * (index + 1))
        ema_update_module_(target, source, decay=0.995)
        assert torch.equal(target.frozen, source.frozen)
        assert torch.equal(target.floating, source.floating)
        assert torch.equal(target.integer, source.integer)


def test_full_state_receipt_detects_unsampled_state_change() -> None:
    model = nn.Linear(8, 8)
    first = module_state_receipt(model)
    with torch.no_grad():
        model.weight[3, 4].add_(1)
    second = module_state_receipt(model)
    assert first["sha256"] != second["sha256"]
    assert first["parameter_values"] == second["parameter_values"]


def test_canonical_and_runtime_state_hashes_are_not_interchangeable() -> None:
    state = {
        "block.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "block.counter": torch.tensor([7], dtype=torch.int64),
    }
    canonical = canonical_model_state_sha256(state)
    runtime = tensor_state_sha256(state)
    receipt = model_state_hash_receipt(state)

    assert canonical == (
        "bb1e8957eab4a8107d2f85ae0b8c93a66a857ad5978fea749e42a222cd18af61"
    )
    assert runtime == (
        "3d0166ccfd33e7478e10b0cdfb3188d2cdc18fa58859fcead3abecdf2f910812"
    )
    assert canonical != runtime
    assert receipt == {
        "canonical_model_state_sha256": canonical,
        "canonical_hash_algorithm": CANONICAL_MODEL_STATE_HASH_ALGORITHM,
        "runtime_tensor_state_sha256": runtime,
        "runtime_hash_algorithm": RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
        "state_tensors": 2,
        "state_values": 13,
    }
    assert require_model_state_hashes(
        state,
        expected_canonical_sha256=canonical,
        expected_runtime_sha256=runtime,
        label="test state",
    ) == receipt
    # Regression for the original failure: comparing the runtime algorithm's
    # output to the canonical lineage value must fail explicitly.
    with pytest.raises(AdjacentConsistencyError, match="runtime tensor-state"):
        require_model_state_hashes(
            state,
            expected_canonical_sha256=canonical,
            expected_runtime_sha256=canonical,
            label="cross-algorithm comparison",
        )
