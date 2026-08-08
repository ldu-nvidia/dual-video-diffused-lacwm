import pytest
import torch

from tools import action_conditioning_audit as audit


def _state():
    torch.manual_seed(5)
    return {
        "morphology_tokens.weight": torch.randn(10, 64),
        "action_encoder.net.0.weight": torch.randn(512, 849) * 0.01,
        "action_encoder.net.0.bias": torch.randn(512) * 0.01,
        "action_encoder.net.2.weight": torch.randn(512, 512) * 0.01,
        "action_encoder.net.2.bias": torch.randn(512) * 0.01,
        "action_encoder.net.4.weight": torch.randn(64, 512) * 0.01,
        "action_encoder.net.4.bias": torch.randn(64) * 0.01,
        "action_pool.0.weight": torch.randn(512, 256) * 0.01,
        "action_pool.0.bias": torch.randn(512) * 0.01,
        "action_pool.2.weight": torch.randn(64, 512) * 0.01,
        "action_pool.2.bias": torch.randn(64) * 0.01,
        "forward_model.action_to_control.net.0.weight": torch.randn(256, 64)
        * 0.01,
        "forward_model.action_to_control.net.0.bias": torch.randn(256) * 0.01,
        "forward_model.action_to_control.net.2.weight": torch.randn(16, 256)
        * 0.01,
        "forward_model.action_to_control.net.2.bias": torch.randn(16) * 0.01,
    }


def test_replay_action_path_matches_frozen_shapes_and_is_deterministic():
    state = _state()
    actions = torch.randn(7, 13, 5, 23)
    first = audit.replay_action_path(state, actions, morphology_index=9)
    second = audit.replay_action_path(state, actions, morphology_index=9)
    assert {key: tuple(value.shape) for key, value in first.items()} == {
        "future_actions": (7, 8, 5, 157),
        "encoded_actions": (7, 8, 64),
        "pooled_actions": (7, 2, 64),
        "control": (7, 2, 16),
    }
    for key in first:
        assert torch.equal(first[key], second[key])


def test_tensor_stats_exposes_mean_dominated_collapse():
    value = torch.ones(8, 2, 16) + 1.0e-4 * torch.randn(8, 2, 16)
    stats = audit.tensor_stats(value)
    assert stats["sample_std_to_rms"] < 0.001
    assert stats["cyclic_shuffle_cosine_mean"] > 0.999


def test_replay_rejects_missing_schema_and_nonfinite_actions():
    with pytest.raises(ValueError, match="missing action-path"):
        audit.replay_action_path({}, torch.zeros(2, 13, 5, 23), morphology_index=9)
    actions = torch.zeros(2, 13, 5, 23)
    actions[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        audit.replay_action_path(_state(), actions, morphology_index=9)
