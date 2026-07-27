import pytest
import torch

from robot_wm.modeling.dual_diffusion.adapters import (
    TFSigmaTokenEmbedding,
    TFVelocityHead,
    ZeroInitTFTokenAdapter,
)


def test_zero_initialized_adapter_is_exact_noop():
    adapter = ZeroInitTFTokenAdapter(tf_channels=12, hidden_size=16)
    state = torch.randn(2, 12, 4, 8, 12)
    tokens, grid = adapter(state)
    raw_tokens, raw_grid = adapter.project_tokens(state)

    assert grid == (4, 4, 6)
    assert raw_grid == grid
    assert tokens.shape == (2, 4 * 4 * 6, 16)
    assert torch.count_nonzero(tokens) == 0
    assert torch.count_nonzero(raw_tokens) > 0


def test_adapter_gate_init_is_effective_scale_and_can_be_frozen():
    adapter = ZeroInitTFTokenAdapter(
        tf_channels=4,
        hidden_size=8,
        gate_init=0.1,
        gate_trainable=False,
    )
    state = torch.randn(2, 4, 2, 4, 4)
    raw_tokens, raw_grid = adapter.project_tokens(state)
    residual, grid = adapter(state)

    assert grid == raw_grid
    torch.testing.assert_close(residual, raw_tokens * 0.1)
    torch.testing.assert_close(torch.tanh(adapter.gate), torch.tensor(0.1))
    assert not adapter.gate.requires_grad
    assert adapter.projection.weight.requires_grad


def test_adapter_gate_checkpoint_key_remains_compatible():
    original = ZeroInitTFTokenAdapter(tf_channels=4, hidden_size=8)
    configured = ZeroInitTFTokenAdapter(
        tf_channels=4,
        hidden_size=8,
        gate_init=0.25,
        gate_trainable=False,
    )

    result = configured.load_state_dict(original.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert configured.gate.shape == torch.Size([])
    assert torch.count_nonzero(configured.gate) == 0
    # Loading a checkpoint must not silently change the configured optimizer
    # policy for this scalar.
    assert not configured.gate.requires_grad


@pytest.mark.parametrize("gate_init", [-1.0, 1.0, float("inf"), float("nan")])
def test_adapter_rejects_invalid_effective_gate_init(gate_init):
    with pytest.raises(ValueError, match="gate_init"):
        ZeroInitTFTokenAdapter(
            tf_channels=4,
            hidden_size=8,
            gate_init=gate_init,
        )


def test_adapter_residual_preserves_bfloat16_compute_dtype():
    adapter = ZeroInitTFTokenAdapter(
        tf_channels=4,
        hidden_size=8,
        gate_init=0.1,
    )
    raw_tokens = torch.randn(2, 8, 8, dtype=torch.bfloat16)

    residual = adapter.residual_tokens(raw_tokens)

    assert residual.dtype == torch.bfloat16


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


def test_tf_sigma_embedding_is_exact_noop_then_receives_gradients():
    embedding = TFSigmaTokenEmbedding(hidden_size=16, embedding_dim=8)
    sigma = torch.tensor([0.0, 0.5, 1.0])

    closed = embedding(sigma)
    assert closed.shape == (3, 16)
    assert torch.count_nonzero(closed) == 0

    embedding.gate.data.fill_(0.1)
    opened = embedding(sigma)
    assert torch.count_nonzero(opened) > 0
    opened.square().mean().backward()
    assert embedding.net[0].weight.grad is not None
    assert torch.isfinite(embedding.net[0].weight.grad).all()
