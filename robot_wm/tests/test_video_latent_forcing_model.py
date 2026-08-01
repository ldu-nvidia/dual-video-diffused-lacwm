import inspect

import pytest
import torch

from robot_wm.modeling.video_latent_forcing import (
    VideoLatentForcingConfig,
    VideoLatentForcingModel,
)


def _tiny_config(**overrides):
    values = {
        "future_frames": 2,
        "history_frames": 5,
        "height": 16,
        "width": 16,
        "patch_size": (1, 8, 8),
        "aux_channels": 4,
        "action_steps": 16,
        "action_dim": 7,
        "hidden_size": 16,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
    }
    values.update(overrides)
    return VideoLatentForcingConfig(**values)


def _inputs(config, batch_size=1):
    return (
        torch.randn(batch_size, *config.future_shape),
        torch.randn(batch_size, *config.auxiliary_shape),
        torch.rand(batch_size),
        torch.rand(batch_size),
        torch.randn(batch_size, *config.history_shape),
        torch.randn(batch_size, config.action_steps, config.action_dim),
    )


def test_frozen_default_and_temporal_patch_two_grids():
    default = VideoLatentForcingConfig()
    assert default.future_shape == (3, 8, 64, 112)
    assert default.history_shape == (3, 5, 64, 112)
    assert default.patch_grid == (8, 8, 14)
    assert default.auxiliary_shape == (48, 8, 8, 14)
    assert (default.action_steps, default.action_dim) == (16, 7)
    assert (default.hidden_size, default.depth, default.num_heads) == (512, 12, 8)

    temporal_two = VideoLatentForcingConfig(patch_size=(2, 8, 8))
    assert temporal_two.patch_grid == (4, 8, 14)
    assert temporal_two.auxiliary_shape == (48, 4, 8, 14)
    with pytest.raises(ValueError, match="aux_grid must match"):
        VideoLatentForcingConfig(aux_grid=(4, 8, 14))


def test_full_video_contract_forward_shapes_and_symmetric_projection():
    config = VideoLatentForcingConfig(
        patch_size=(2, 8, 8),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
    )
    model = VideoLatentForcingModel(config).eval()
    noisy_video, noisy_auxiliary, t_video, t_auxiliary, history, actions = _inputs(
        config
    )

    with torch.no_grad():
        video_tokens = (
            model.video_patch_projection(noisy_video).flatten(2).transpose(1, 2)
        )
        auxiliary_tokens = (
            model.auxiliary_patch_projection(noisy_auxiliary)
            .flatten(2)
            .transpose(1, 2)
        )
        fused = model.project_states(noisy_video, noisy_auxiliary)
        output = model(
            noisy_video,
            noisy_auxiliary,
            t_video,
            t_auxiliary,
            history,
            actions,
        )

    torch.testing.assert_close(fused, video_tokens + auxiliary_tokens, rtol=0, atol=0)
    assert output.video_x.shape == (1, 3, 8, 64, 112)
    assert output.auxiliary_x.shape == (1, 48, 4, 8, 14)
    assert model.video_output_head is not model.auxiliary_output_head
    assert not hasattr(model, "auxiliary_gate")
    assert "future" not in inspect.signature(model.forward).parameters


def test_per_example_fusion_mask_is_exact_noop_and_both_clocks_condition():
    torch.manual_seed(20260801)
    config = _tiny_config()
    model = VideoLatentForcingModel(config).eval()
    inputs = list(_inputs(config, batch_size=2))
    inputs[2] = torch.zeros(2)
    inputs[3] = torch.zeros(2)
    mask = torch.tensor([True, False])

    with torch.no_grad():
        baseline = model(*inputs, auxiliary_fusion_mask=mask)
        changed_off = [value.clone() for value in inputs]
        changed_off[1][1].fill_(1000.0)
        off_output = model(*changed_off, auxiliary_fusion_mask=mask)

        changed_on = [value.clone() for value in inputs]
        changed_on[1][0].add_(10.0)
        on_output = model(*changed_on, auxiliary_fusion_mask=mask)

        changed_auxiliary_clock = [value.clone() for value in inputs]
        changed_auxiliary_clock[3][0] = 1.0
        changed_auxiliary_clock[3][1] = 1.0
        auxiliary_clock_output = model(
            *changed_auxiliary_clock,
            auxiliary_fusion_mask=mask,
        )

        changed_video_clock = [value.clone() for value in inputs]
        changed_video_clock[2][0] = 1.0
        video_clock_output = model(*changed_video_clock, auxiliary_fusion_mask=mask)

    torch.testing.assert_close(
        baseline.video_x[1], off_output.video_x[1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline.auxiliary_x[1], off_output.auxiliary_x[1], rtol=0, atol=0
    )
    assert not torch.equal(baseline.auxiliary_x[0], on_output.auxiliary_x[0])
    assert not torch.equal(
        baseline.video_x[0], auxiliary_clock_output.video_x[0]
    )
    assert not torch.equal(
        baseline.video_x[1], auxiliary_clock_output.video_x[1]
    )
    assert not torch.equal(baseline.video_x[0], video_clock_output.video_x[0])

    model.zero_grad(set_to_none=True)
    all_off = model(*inputs, auxiliary_fusion_mask=torch.zeros(2, dtype=torch.bool))
    all_off.video_x.sum().backward()
    assert model.auxiliary_patch_projection.weight.grad is not None
    assert torch.count_nonzero(model.auxiliary_patch_projection.weight.grad) == 0


def test_ordered_action_and_spatial_history_tokens_are_causal_conditions():
    torch.manual_seed(17)
    config = _tiny_config()
    model = VideoLatentForcingModel(config).eval()
    inputs = list(_inputs(config))

    with torch.no_grad():
        baseline = model(*inputs)
        permuted_actions = [value.clone() for value in inputs]
        permuted_actions[5] = permuted_actions[5].flip(1)
        action_output = model(*permuted_actions)
        permuted_history = [value.clone() for value in inputs]
        permuted_history[4] = permuted_history[4].flip(-1)
        history_output = model(*permuted_history)

    assert not torch.equal(baseline.video_x, action_output.video_x)
    assert not torch.equal(baseline.video_x, history_output.video_x)


def test_parameter_matched_video_only_has_equal_schema_and_strict_aux_noop():
    dual_config = _tiny_config(parameter_matched_video_only=False)
    baseline_config = _tiny_config(parameter_matched_video_only=True)
    dual = VideoLatentForcingModel(dual_config)
    baseline = VideoLatentForcingModel(baseline_config).eval()
    assert sum(parameter.numel() for parameter in dual.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert all(parameter.requires_grad for parameter in baseline.parameters())

    inputs = list(_inputs(baseline_config, batch_size=2))
    with torch.no_grad():
        first = baseline(*inputs)
        inputs[1] = torch.randn_like(inputs[1]) * 1000
        inputs[3] = 1.0 - inputs[3]
        second = baseline(*inputs)

    torch.testing.assert_close(first.video_x, second.video_x, rtol=0, atol=0)
    assert torch.count_nonzero(first.auxiliary_x) == 0
    assert torch.count_nonzero(second.auxiliary_x) == 0
