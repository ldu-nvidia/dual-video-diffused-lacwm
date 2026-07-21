import torch

from robot_wm.modeling.dual_diffusion.adapters import (
    TFVelocityHead,
    ZeroInitTFTokenAdapter,
)


def test_zero_initialized_adapter_is_exact_noop():
    adapter = ZeroInitTFTokenAdapter(tf_channels=12, hidden_size=16)
    state = torch.randn(2, 12, 4, 8, 12)
    tokens, grid = adapter(state)

    assert grid == (4, 4, 6)
    assert tokens.shape == (2, 4 * 4 * 6, 16)
    assert torch.count_nonzero(tokens) == 0


def test_adapter_becomes_trainable_when_gate_opens():
    adapter = ZeroInitTFTokenAdapter(tf_channels=4, hidden_size=8)
    adapter.gate.data.fill_(0.1)
    state = torch.randn(2, 4, 2, 4, 4, requires_grad=True)
    tokens, _ = adapter(state)
    tokens.square().mean().backward()

    assert state.grad is not None
    assert adapter.projection.weight.grad is not None
    assert torch.isfinite(adapter.projection.weight.grad).all()


def test_zero_initialized_tf_velocity_head_shape():
    head = TFVelocityHead(hidden_size=16, tf_channels=12, patch_size=(1, 2, 2))
    tokens = torch.randn(2, 4 * 4 * 6, 16)
    velocity = head(tokens, (4, 4, 6))

    assert velocity.shape == (2, 12, 4, 8, 12)
    assert torch.count_nonzero(velocity) == 0
