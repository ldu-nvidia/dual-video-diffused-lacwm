import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from torch import nn

from robot_wm.modeling.dual_diffusion.adapters import (
    TFSigmaTokenEmbedding,
    TFVelocityHead,
    ZeroInitTFTokenAdapter,
)


def _load_wan_forward_module():
    """Load the wrapper with light stubs instead of the full VideoX runtime."""
    einops = ModuleType("einops")
    einops.rearrange = lambda value, _pattern: value

    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = object

    peft = ModuleType("peft")
    peft.LoraConfig = object
    peft.inject_adapter_in_model = lambda _config, model: model

    videox_fun = ModuleType("videox_fun")
    videox_models = ModuleType("videox_fun.models")
    wan_transformer = ModuleType("videox_fun.models.wan_transformer3d")
    wan_transformer.WanTransformer3DModel = object

    module_name = "_dual_wan_forward_model_under_test"
    source = (
        Path(__file__).parents[1]
        / "modeling"
        / "networks"
        / "wan_forward_model.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "einops": einops,
        "omegaconf": omegaconf,
        "peft": peft,
        "videox_fun": videox_fun,
        "videox_fun.models": videox_models,
        "videox_fun.models.wan_transformer3d": wan_transformer,
        module_name: module,
    }
    with patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


WAN_FORWARD = _load_wan_forward_module()


class _FakeActionToControl(nn.Module):
    def forward(self, actions, height, width):
        batch, frames, _ = actions.shape
        return actions.new_zeros(batch, 16, frames, height, width)


class _FakeHead(nn.Module):
    def forward(self, tokens, _conditioning=None):
        return tokens


class _FakeTransformer(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            48,
            hidden_size,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=False,
        )
        nn.init.constant_(self.patch_embedding.weight, 0.05)
        self.video_gain = nn.Parameter(torch.tensor(0.25))
        self.head = _FakeHead()
        self.last_shared_tokens = None
        self.last_native_patches = None

    def forward(
        self,
        *,
        x,
        t,
        context,
        seq_len,
        y,
        y_camera=None,
        clip_fea=None,
    ):
        del t, context, seq_len, clip_fea
        native_patches = []
        shared = []
        for index in range(x.shape[0]):
            native = self.patch_embedding(
                torch.cat([x[index], y[index]], dim=0).unsqueeze(0)
            )
            native_patches.append(native)
            if y_camera is not None:
                native = native + y_camera[index]
            shared.append(native.flatten(2).transpose(1, 2))
        self.last_native_patches = native_patches
        self.last_shared_tokens = torch.cat(shared, dim=0)
        self.head(self.last_shared_tokens, None)
        shared_mean = self.last_shared_tokens.mean(dim=(1, 2)).reshape(
            x.shape[0], 1, 1, 1, 1
        )
        return self.video_gain * x + shared_mean


class _FakeBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.last_input = None
        self.last_output = None

    def forward(self, tokens):
        self.last_input = tokens
        self.last_output = tokens + (self.index + 1) * 1.0e-4
        return self.last_output


class _FakeMidpointTransformer(_FakeTransformer):
    def __init__(self, hidden_size=8):
        super().__init__(hidden_size=hidden_size)
        self.blocks = nn.ModuleList([_FakeBlock(index) for index in range(30)])

    def forward(
        self,
        *,
        x,
        t,
        context,
        seq_len,
        y,
        y_camera=None,
        clip_fea=None,
    ):
        del t, context, seq_len, clip_fea
        native_patches = []
        shared = []
        for index in range(x.shape[0]):
            native = self.patch_embedding(
                torch.cat([x[index], y[index]], dim=0).unsqueeze(0)
            )
            native_patches.append(native)
            if y_camera is not None:
                native = native + y_camera[index]
            shared.append(native.flatten(2).transpose(1, 2))
        self.last_native_patches = native_patches
        tokens = torch.cat(shared, dim=0)
        for block in self.blocks:
            tokens = block(tokens)
        self.last_shared_tokens = tokens
        self.head(tokens, None)
        shared_mean = tokens.mean(dim=(1, 2)).reshape(
            x.shape[0], 1, 1, 1, 1
        )
        return self.video_gain * x + shared_mean


class _CapturingTFHead(nn.Module):
    def __init__(self, tf_channels=4, patch_size=(1, 2, 2)):
        super().__init__()
        self.tf_channels = tf_channels
        self.patch_size = patch_size
        self.last_tokens = None
        self.calls = 0

    def forward(self, tokens, grid):
        self.last_tokens = tokens
        self.calls += 1
        pt, ph, pw = self.patch_size
        return tokens.new_zeros(
            tokens.shape[0],
            self.tf_channels,
            grid[0] * pt,
            grid[1] * ph,
            grid[2] * pw,
        )


class _ConstantTFHead(_CapturingTFHead):
    def forward(self, tokens, grid):
        output = super().forward(tokens, grid)
        return torch.ones_like(output)


def _make_dual_model(
    *,
    condition_on_tf=True,
    condition_on_tf_clock=None,
    tf_head_condition_on_clock=False,
    state_gate_init=0.1,
    state_gate_trainable=False,
    clock_gate_init=0.0,
    clock_gate_trainable=True,
    intra_forward_forcing=False,
):
    model = WAN_FORWARD.WanForwardModel.__new__(
        WAN_FORWARD.WanForwardModel
    )
    nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    model.dual_diffusion_enabled = True
    model.condition_on_tf = condition_on_tf
    model.condition_on_tf_clock = (
        condition_on_tf
        if condition_on_tf_clock is None
        else condition_on_tf_clock
    )
    model.tf_head_condition_on_clock = tf_head_condition_on_clock
    model.intra_forward_forcing_enabled = intra_forward_forcing
    model.intra_forward_block_index = 14
    model.intra_forward_stop_gradient = True
    model.intra_forward_history_bins = 2
    model.transformer = (
        _FakeMidpointTransformer()
        if intra_forward_forcing
        else _FakeTransformer()
    )
    model.action_to_control = _FakeActionToControl()
    model.tf_token_adapter = ZeroInitTFTokenAdapter(
        tf_channels=4,
        hidden_size=8,
        patch_size=model.patch_size,
        gate_init=state_gate_init,
        gate_trainable=state_gate_trainable,
    )
    model.tf_clock_embedding = TFSigmaTokenEmbedding(
        hidden_size=8,
        embedding_dim=8,
        gate_init=clock_gate_init,
        gate_trainable=clock_gate_trainable,
    )
    model.tf_velocity_head = _CapturingTFHead()
    return model


def test_intra_forward_forcing_predicts_and_injects_once_inside_one_wan_call():
    torch.manual_seed(101)
    model = _make_dual_model(
        intra_forward_forcing=True,
        condition_on_tf=True,
        condition_on_tf_clock=False,
        state_gate_init=0.2,
    )
    video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(video)
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    common = dict(
        timesteps=torch.tensor([100.0, 200.0]),
        z_control=torch.randn(2, 2, 3),
        ref_latents=reference,
        context=[torch.zeros(1, 4), torch.zeros(1, 4)],
        noisy_tf=noisy_tf,
        conditioning_tf=noisy_tf,
        tf_sigma=torch.ones(2),
        condition_on_tf=True,
        condition_on_tf_clock=False,
    )

    aligned = model(
        video,
        common.pop("timesteps"),
        common.pop("z_control"),
        common.pop("ref_latents"),
        common.pop("context"),
        intra_forward_condition_source="aligned",
        **common,
    )

    assert model.tf_velocity_head.calls == 1
    assert aligned.tf_velocity.shape == noisy_tf.shape
    assert torch.count_nonzero(aligned.tf_condition_tokens) > 0
    assert aligned.tf_condition_telemetry["midpoint_head_calls"].item() == 1
    assert aligned.tf_condition_telemetry["midpoint_block_index"].item() == 14
    assert (
        aligned.tf_condition_telemetry["midpoint_overhead_latency_ms"].item()
        == 0
    )
    # The residual produced after block 14 is the exact change seen by block 15.
    block_14 = model.transformer.blocks[14]
    block_15 = model.transformer.blocks[15]
    torch.testing.assert_close(
        block_15.last_input - block_14.last_output,
        aligned.tf_condition_tokens,
    )
    assert not model.transformer.blocks[14]._forward_hooks


def test_intra_forward_clean_estimate_has_registered_flow_sign_and_endpoints():
    torch.manual_seed(202)
    model = _make_dual_model(
        intra_forward_forcing=True,
        condition_on_tf=True,
        condition_on_tf_clock=False,
        state_gate_init=0.2,
    )
    model.tf_velocity_head = _ConstantTFHead()
    projected_inputs = []
    original_project = model.tf_token_adapter.project_tokens

    def record_project(value):
        projected_inputs.append(value.detach().clone())
        return original_project(value)

    model.tf_token_adapter.project_tokens = record_project
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    model(
        torch.randn(2, 16, 2, 4, 4),
        torch.tensor([0.0, 1000.0]),
        torch.randn(2, 2, 3),
        torch.randn(2, 16, 2, 4, 4),
        [torch.zeros(1, 4), torch.zeros(1, 4)],
        noisy_tf=noisy_tf,
        tf_sigma=torch.tensor([0.0, 1.0]),
        intra_forward_condition_source="aligned",
    )

    # First projection is q_sigma for the head; the final projection is the
    # stopped q0_hat injected into block 15.
    injected_clean = projected_inputs[-1]
    torch.testing.assert_close(injected_clean[0], noisy_tf[0])
    torch.testing.assert_close(injected_clean[1], noisy_tf[1] - 1.0)


def test_intra_forward_off_and_shuffle_are_same_checkpoint_interventions():
    torch.manual_seed(303)
    model = _make_dual_model(
        intra_forward_forcing=True,
        condition_on_tf=True,
        condition_on_tf_clock=False,
        state_gate_init=0.25,
    )
    video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(video)
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    args = (
        video,
        torch.tensor([100.0, 200.0]),
        torch.randn(2, 2, 3),
        reference,
        [torch.zeros(1, 4), torch.zeros(1, 4)],
    )
    kwargs = dict(
        noisy_tf=noisy_tf,
        conditioning_tf=noisy_tf,
        tf_sigma=torch.ones(2),
        condition_on_tf=True,
        condition_on_tf_clock=False,
    )

    aligned = model(
        *args, intra_forward_condition_source="aligned", **kwargs
    )
    off = model(*args, intra_forward_condition_source="off", **kwargs)
    shuffled = model(
        *args, intra_forward_condition_source="shuffled", **kwargs
    )

    assert torch.count_nonzero(off.tf_condition_tokens) == 0
    assert not torch.equal(aligned.video_velocity, off.video_velocity)
    assert not torch.equal(aligned.video_velocity, shuffled.video_velocity)
    assert model.tf_velocity_head.calls == 3
    assert not model.transformer.blocks[14]._forward_hooks


def test_intra_forward_future_shuffle_preserves_history_bins(monkeypatch):
    torch.manual_seed(404)
    model = _make_dual_model(
        intra_forward_forcing=True,
        condition_on_tf=True,
        condition_on_tf_clock=False,
        state_gate_init=0.25,
    )
    projected_inputs = []
    original_project = model.tf_token_adapter.project_tokens

    def record_project(value):
        projected_inputs.append(value.detach().clone())
        return original_project(value)

    def donor_future(value):
        return value + 100.0

    model.tf_token_adapter.project_tokens = record_project
    monkeypatch.setattr(WAN_FORWARD, "roll_across_global_batch", donor_future)
    noisy_tf = torch.randn(2, 4, 4, 4, 4)
    model(
        torch.randn(2, 16, 4, 4, 4),
        torch.tensor([100.0, 200.0]),
        torch.randn(2, 4, 3),
        torch.randn(2, 16, 4, 4, 4),
        [torch.zeros(1, 4), torch.zeros(1, 4)],
        noisy_tf=noisy_tf,
        tf_sigma=torch.tensor([0.3, 0.6]),
        intra_forward_condition_source="future_shuffled",
    )

    generated_clean = projected_inputs[0]
    injected_clean = projected_inputs[-1]
    torch.testing.assert_close(
        injected_clean[:, :, :2], generated_clean[:, :, :2]
    )
    torch.testing.assert_close(
        injected_clean[:, :, 2:], generated_clean[:, :, 2:] + 100.0
    )


def test_intra_forward_forcing_rejects_unregistered_source():
    model = _make_dual_model(intra_forward_forcing=True)
    with pytest.raises(ValueError, match="future_shuffled"):
        model(
            torch.randn(2, 16, 2, 4, 4),
            torch.tensor([100.0, 200.0]),
            torch.randn(2, 2, 3),
            torch.randn(2, 16, 2, 4, 4),
            [torch.zeros(1, 4), torch.zeros(1, 4)],
            noisy_tf=torch.randn(2, 4, 2, 4, 4),
            tf_sigma=torch.tensor([0.3, 0.6]),
            intra_forward_condition_source="oracle",
        )


def test_intra_forward_video_path_stops_gradient_at_generated_clean_estimate():
    torch.manual_seed(707)
    model = _make_dual_model(
        intra_forward_forcing=True,
        condition_on_tf=True,
        condition_on_tf_clock=False,
        state_gate_init=0.2,
    )
    model.tf_velocity_head = TFVelocityHead(
        hidden_size=8, tf_channels=4, patch_size=(1, 2, 2)
    )
    output = model(
        torch.randn(2, 16, 2, 4, 4),
        torch.tensor([100.0, 200.0]),
        torch.randn(2, 2, 3),
        torch.randn(2, 16, 2, 4, 4),
        [torch.zeros(1, 4), torch.zeros(1, 4)],
        noisy_tf=torch.randn(2, 4, 2, 4, 4),
        tf_sigma=torch.tensor([0.3, 0.6]),
        intra_forward_condition_source="aligned",
    )

    output.video_velocity.sum().backward()

    # The video objective may train the post-stop adapter and gate, but must
    # not train the velocity head through q0_hat.
    assert model.tf_velocity_head.linear.weight.grad is None
    assert model.tf_velocity_head.linear.bias.grad is None
    assert model.tf_token_adapter.projection.weight.grad is not None
    assert torch.count_nonzero(
        model.tf_token_adapter.projection.weight.grad
    ) > 0


def test_intra_forward_training_rejects_gradient_checkpoint_recomputation():
    model = _make_dual_model(intra_forward_forcing=True)
    model.transformer.gradient_checkpointing = True

    with pytest.raises(RuntimeError, match="cannot train with gradient checkpointing"):
        model(
            torch.randn(2, 16, 2, 4, 4),
            torch.tensor([100.0, 200.0]),
            torch.randn(2, 2, 3),
            torch.randn(2, 16, 2, 4, 4),
            [torch.zeros(1, 4), torch.zeros(1, 4)],
            noisy_tf=torch.randn(2, 4, 2, 4, 4),
            tf_sigma=torch.tensor([0.3, 0.6]),
            intra_forward_condition_source="aligned",
        )


def _rms(value):
    return value.detach().float().square().mean().sqrt()


def test_video_uses_conditioning_tf_while_tf_head_uses_own_noisy_state():
    torch.manual_seed(7)
    model = _make_dual_model()
    noisy_video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(noisy_video)
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    conditioning_tf = torch.randn_like(noisy_tf) + 3.0
    noisy_tokens, _ = model.tf_token_adapter.project_tokens(noisy_tf)
    conditioning_tokens, _ = model.tf_token_adapter.project_tokens(
        conditioning_tf
    )

    output = model(
        noisy_video,
        torch.tensor([100.0, 200.0]),
        torch.randn(2, 2, 3),
        reference,
        [torch.zeros(1, 4), torch.zeros(1, 4)],
        noisy_tf=noisy_tf,
        conditioning_tf=conditioning_tf,
        tf_sigma=torch.tensor([0.3, 0.6]),
    )

    torch.testing.assert_close(
        output.tf_condition_tokens,
        conditioning_tokens * 0.1,
    )
    torch.testing.assert_close(
        model.tf_velocity_head.last_tokens
        - model.transformer.last_shared_tokens,
        noisy_tokens,
    )
    assert output.video_velocity.shape == noisy_video.shape
    assert output.tf_velocity.shape == noisy_tf.shape

    telemetry = output.tf_condition_telemetry
    assert set(telemetry) == {
        "raw_state_rms",
        "state_residual_rms",
        "clock_residual_rms",
        "combined_rms",
        "native_patch_embedding_rms",
        "state_to_native_ratio",
        "combined_to_native_ratio",
    }
    native_rms = (
        torch.stack(
            [
                value.detach().float().square().sum()
                for value in model.transformer.last_native_patches
            ]
        ).sum()
        / sum(value.numel() for value in model.transformer.last_native_patches)
    ).sqrt()
    torch.testing.assert_close(telemetry["raw_state_rms"], _rms(conditioning_tokens))
    torch.testing.assert_close(
        telemetry["state_residual_rms"],
        _rms(conditioning_tokens * 0.1),
    )
    assert torch.count_nonzero(telemetry["clock_residual_rms"]) == 0
    torch.testing.assert_close(
        telemetry["combined_rms"], _rms(output.tf_condition_tokens)
    )
    torch.testing.assert_close(
        telemetry["native_patch_embedding_rms"], native_rms
    )
    torch.testing.assert_close(
        telemetry["state_to_native_ratio"],
        telemetry["state_residual_rms"] / native_rms,
    )
    torch.testing.assert_close(
        telemetry["combined_to_native_ratio"],
        telemetry["combined_rms"] / native_rms,
    )


def test_conditioning_tf_shape_must_match_noisy_tf():
    model = _make_dual_model()

    with pytest.raises(ValueError, match="conditioning TF state"):
        model(
            torch.randn(2, 16, 2, 4, 4),
            torch.tensor([100.0, 200.0]),
            torch.randn(2, 2, 3),
            torch.randn(2, 16, 2, 4, 4),
            [torch.zeros(1, 4), torch.zeros(1, 4)],
            noisy_tf=torch.randn(2, 4, 2, 4, 4),
            conditioning_tf=torch.randn(2, 4, 2, 4, 2),
            tf_sigma=torch.tensor([0.3, 0.6]),
        )


def test_auxiliary_only_head_sees_state_and_clock_but_video_does_not():
    torch.manual_seed(29)
    model = _make_dual_model(
        condition_on_tf=False,
        condition_on_tf_clock=False,
        tf_head_condition_on_clock=True,
        state_gate_init=0.25,
        clock_gate_init=0.25,
    )
    video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(video)
    actions = torch.randn(2, 2, 3)
    context = [torch.zeros(1, 4), torch.zeros(1, 4)]

    first = model(
        video,
        torch.tensor([100.0, 200.0]),
        actions,
        reference,
        context,
        noisy_tf=torch.randn(2, 4, 2, 4, 4),
        tf_sigma=torch.tensor([0.1, 0.2]),
        condition_on_tf=False,
        condition_on_tf_clock=False,
    )
    first_head_tokens = model.tf_velocity_head.last_tokens.detach().clone()
    second = model(
        video,
        torch.tensor([100.0, 200.0]),
        actions,
        reference,
        context,
        noisy_tf=torch.randn(2, 4, 2, 4, 4) + 50.0,
        tf_sigma=torch.tensor([0.8, 0.9]),
        condition_on_tf=False,
        condition_on_tf_clock=False,
    )
    second_head_tokens = model.tf_velocity_head.last_tokens.detach().clone()

    # A1's private auxiliary inputs can train the shared trunk/head, but cannot
    # directly alter the video prediction in the same forward call.
    assert torch.equal(first.video_velocity, second.video_velocity)
    assert not torch.equal(first_head_tokens, second_head_tokens)
    assert torch.count_nonzero(first.tf_condition_tokens) == 0
    assert torch.count_nonzero(second.tf_condition_tokens) == 0


def test_frozen_zero_state_and_clock_make_video_loss_output_and_gradients_tf_invariant():
    torch.manual_seed(11)
    model = _make_dual_model(
        condition_on_tf=False,
        state_gate_init=0.0,
        state_gate_trainable=False,
        clock_gate_init=0.0,
        clock_gate_trainable=False,
    )
    noisy_video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(noisy_video)
    actions = torch.randn(2, 2, 3)
    timesteps = torch.tensor([100.0, 200.0])
    context = [torch.zeros(1, 4), torch.zeros(1, 4)]
    target = torch.randn_like(noisy_video)

    def run(noisy_tf, conditioning_tf, tf_sigma):
        model.zero_grad(set_to_none=True)
        output = model(
            noisy_video,
            timesteps,
            actions,
            reference,
            context,
            noisy_tf=noisy_tf,
            conditioning_tf=conditioning_tf,
            tf_sigma=tf_sigma,
            condition_on_tf=False,
        )
        video_loss = (output.video_velocity - target).square().mean()
        # The video-only objective is structurally just the video loss; it
        # deliberately does not attach a nominally zero-weight TF graph.
        video_loss.backward()
        gradients = {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            for name, parameter in model.named_parameters()
        }
        return (
            output.video_velocity.detach().clone(),
            video_loss.detach().clone(),
            output.tf_condition_tokens.detach().clone(),
            gradients,
        )

    def run_production_video_path():
        model.zero_grad(set_to_none=True)
        model.dual_diffusion_enabled = False
        try:
            video_velocity = model(
                noisy_video,
                timesteps,
                actions,
                reference,
                context,
            )
        finally:
            model.dual_diffusion_enabled = True
        video_loss = (video_velocity - target).square().mean()
        video_loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.transformer.named_parameters()
            if parameter.grad is not None
        }
        return (
            video_velocity.detach().clone(),
            video_loss.detach().clone(),
            gradients,
        )

    tf_a = torch.randn(2, 4, 2, 4, 4)
    tf_b = torch.randn_like(tf_a) * 100.0 + 37.0
    production = run_production_video_path()
    first = run(tf_a, tf_a.flip(0), torch.tensor([0.0, 1.0]))
    second = run(tf_b, -tf_b, torch.tensor([0.91, 0.17]))

    assert torch.equal(production[0], first[0])
    assert torch.equal(production[1], first[1])
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.count_nonzero(first[2]) == 0
    assert torch.count_nonzero(second[2]) == 0
    assert first[3].keys() == second[3].keys()
    for name in first[3]:
        grad_a = first[3][name]
        grad_b = second[3][name]
        assert (grad_a is None) == (grad_b is None), name
        if grad_a is not None:
            assert torch.equal(grad_a, grad_b), name

    for name, production_gradient in production[2].items():
        dual_gradient = first[3][f"transformer.{name}"]
        assert dual_gradient is not None
        assert torch.equal(production_gradient, dual_gradient), name
    assert model.transformer.video_gain.grad is not None
    assert torch.count_nonzero(model.transformer.video_gain.grad) > 0
    assert not model.tf_token_adapter.gate.requires_grad
    assert not model.tf_clock_embedding.gate.requires_grad
    for module in (model.tf_token_adapter, model.tf_clock_embedding):
        for parameter in module.parameters():
            if parameter.grad is not None:
                assert torch.count_nonzero(parameter.grad) == 0
