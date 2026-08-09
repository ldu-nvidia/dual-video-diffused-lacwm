from __future__ import annotations

import pytest
import torch

from robot_wm.modeling.dual_diffusion.adapters import ZeroInitTFTokenAdapter
from robot_wm.modeling.dual_diffusion.physics_flow import (
    FLOW_CHANNELS,
    HISTORY_LATENT_FRAMES,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    VIEW_LATENT_WIDTH,
    PhysicsFlowError,
    pack_top_view_flow,
    tensor_sha256,
    timeshift_plus_one,
    unpack_top_view_flow,
    validate_packed_flow,
)


def _transitions(batch: int = 2) -> torch.Tensor:
    value = torch.zeros(batch, 8, 4, LATENT_HEIGHT, VIEW_LATENT_WIDTH)
    for transition in range(8):
        value[:, transition, 0] = 0.1 + transition / 100
        value[:, transition, 1] = 0.2 + transition / 100
        value[:, transition, 2] = (transition + 1) / 8
        value[:, transition, 3] = 0.3 + transition / 100
    return value


def test_pack_round_trip_and_exact_support() -> None:
    transitions = _transitions()
    packed = pack_top_view_flow(transitions)
    assert packed.shape == (2, FLOW_CHANNELS, 4, LATENT_HEIGHT, LATENT_WIDTH)
    assert torch.equal(packed[:, :, :HISTORY_LATENT_FRAMES], torch.zeros_like(packed[:, :, :2]))
    assert torch.count_nonzero(packed[..., VIEW_LATENT_WIDTH:]) == 0
    assert torch.equal(unpack_top_view_flow(packed), transitions)


def test_channel_order_is_transition_major() -> None:
    transitions = _transitions(batch=1)
    packed = pack_top_view_flow(transitions)
    for future_latent in range(2):
        for substep in range(4):
            transition = 4 * future_latent + substep
            base = 4 * substep
            observed = packed[0, base : base + 4, 2 + future_latent, 0, 0]
            expected = transitions[0, transition, :, 0, 0]
            assert torch.equal(observed, expected)


def test_nonwrapping_timeshift_uses_next_transition_and_zero_tail() -> None:
    transitions = _transitions(batch=1)
    shifted = timeshift_plus_one(transitions)
    assert torch.equal(shifted[:, :-1], transitions[:, 1:])
    assert torch.count_nonzero(shifted[:, -1]) == 0


def test_validation_rejects_history_wrist_and_visibility_mutations() -> None:
    packed = pack_top_view_flow(_transitions(batch=1))
    bad = packed.clone()
    bad[0, 0, 0, 0, 0] = 1
    with pytest.raises(PhysicsFlowError, match="history"):
        validate_packed_flow(bad)
    bad = packed.clone()
    bad[0, 0, 2, 0, VIEW_LATENT_WIDTH] = 1
    with pytest.raises(PhysicsFlowError, match="wrist"):
        validate_packed_flow(bad)
    bad = packed.clone()
    bad[0, 2, 2, 0, 0] = 1.01
    with pytest.raises(PhysicsFlowError, match="visibility"):
        validate_packed_flow(bad)


def test_validation_rejects_hidden_values_and_unscaled_displacements() -> None:
    transitions = _transitions(batch=1)
    transitions[:, 0, 2, 0, 0] = 0
    with pytest.raises(PhysicsFlowError, match="unsupported"):
        pack_top_view_flow(transitions)
    transitions = _transitions(batch=1)
    transitions[:, 0, 0, 0, 0] = 1.01
    with pytest.raises(PhysicsFlowError, match="displacement"):
        pack_top_view_flow(transitions)


def test_tensor_hash_binds_dtype_shape_and_content() -> None:
    value = pack_top_view_flow(_transitions(batch=1))
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.double())
    changed = value.clone()
    changed[0, 0, 2, 0, 0] += 1
    assert tensor_sha256(value) != tensor_sha256(changed)


def test_zero_support_remains_exact_zero_after_biased_projection() -> None:
    adapter = ZeroInitTFTokenAdapter(
        tf_channels=FLOW_CHANNELS,
        hidden_size=32,
        patch_size=(1, 2, 2),
        gate_init=0.02,
        preserve_zero_support=True,
    )
    with torch.no_grad():
        adapter.projection.bias.fill_(0.75)
        adapter.norm.bias.fill_(0.5)
    condition = pack_top_view_flow(_transitions(batch=1))
    residual, grid = adapter(condition)
    assert grid == (4, 12, 60)
    support = torch.nn.functional.max_pool3d(
        condition.abs().amax(dim=1, keepdim=True),
        kernel_size=(1, 2, 2),
        stride=(1, 2, 2),
    ).flatten(2).transpose(1, 2).ne(0).expand_as(residual)
    assert torch.count_nonzero(residual.masked_select(~support)) == 0
    assert torch.count_nonzero(residual.masked_select(support)) > 0
