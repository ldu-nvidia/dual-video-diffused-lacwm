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
        return torch.zeros_like(x)


class _CapturingTFHead(nn.Module):
    def __init__(self, tf_channels=4, patch_size=(1, 2, 2)):
        super().__init__()
        self.tf_channels = tf_channels
        self.patch_size = patch_size
        self.last_tokens = None

    def forward(self, tokens, grid):
        self.last_tokens = tokens
        pt, ph, pw = self.patch_size
        return tokens.new_zeros(
            tokens.shape[0],
            self.tf_channels,
            grid[0] * pt,
            grid[1] * ph,
            grid[2] * pw,
        )


def _make_dual_model():
    model = WAN_FORWARD.WanForwardModel.__new__(
        WAN_FORWARD.WanForwardModel
    )
    nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    model.dual_diffusion_enabled = True
    model.condition_on_tf = True
    model.transformer = _FakeTransformer()
    model.action_to_control = _FakeActionToControl()
    model.tf_token_adapter = ZeroInitTFTokenAdapter(
        tf_channels=4,
        hidden_size=8,
        patch_size=model.patch_size,
        gate_init=0.1,
        gate_trainable=False,
    )
    model.tf_clock_embedding = TFSigmaTokenEmbedding(
        hidden_size=8,
        embedding_dim=8,
    )
    model.tf_velocity_head = _CapturingTFHead()
    return model


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
        "clock_injection_fraction",
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
    assert telemetry["clock_injection_fraction"] == 1
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


def test_per_sample_condition_mask_neutralizes_tf_content_exactly():
    torch.manual_seed(17)
    model = _make_dual_model()
    noisy_video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(noisy_video)
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    conditioning_tf = torch.randn_like(noisy_tf) + 4.0
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
        condition_on_tf=torch.tensor([True, False]),
    )

    torch.testing.assert_close(
        output.tf_condition_tokens[0],
        conditioning_tokens[0] * 0.1,
    )
    torch.testing.assert_close(
        output.tf_condition_tokens[1],
        torch.zeros_like(output.tf_condition_tokens[1]),
    )


@pytest.mark.parametrize(
    "condition_on_tf,match",
    [
        (torch.tensor([[True], [False]]), r"shape \[batch\]"),
        (torch.tensor([1.0, 0.5]), "exactly zero or one"),
    ],
)
def test_per_sample_condition_mask_rejects_ambiguous_values(
    condition_on_tf,
    match,
):
    model = _make_dual_model()
    with pytest.raises(ValueError, match=match):
        model(
            torch.randn(2, 16, 2, 4, 4),
            torch.tensor([100.0, 200.0]),
            torch.randn(2, 2, 3),
            torch.randn(2, 16, 2, 4, 4),
            [torch.zeros(1, 4), torch.zeros(1, 4)],
            noisy_tf=torch.randn(2, 4, 2, 4, 4),
            conditioning_tf=torch.randn(2, 4, 2, 4, 4),
            tf_sigma=torch.tensor([0.3, 0.6]),
            condition_on_tf=condition_on_tf,
        )


def test_per_sample_clock_mask_is_independent_and_reported_after_masking():
    torch.manual_seed(29)
    model = _make_dual_model()
    with torch.no_grad():
        model.tf_clock_embedding.gate.fill_(0.4)
    noisy_video = torch.randn(2, 16, 2, 4, 4)
    reference = torch.randn_like(noisy_video)
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    tf_sigma = torch.tensor([0.3, 0.6])

    output = model(
        noisy_video,
        torch.tensor([100.0, 200.0]),
        torch.randn(2, 2, 3),
        reference,
        [torch.zeros(1, 4), torch.zeros(1, 4)],
        noisy_tf=noisy_tf,
        conditioning_tf=noisy_tf,
        tf_sigma=tf_sigma,
        condition_on_tf=False,
        condition_on_tf_clock=torch.tensor([1, 0]),
    )

    expected_clock = (
        model.tf_clock_embedding(tf_sigma)
        .unsqueeze(1)
        .expand_as(output.tf_condition_tokens)
    )
    torch.testing.assert_close(
        output.tf_condition_tokens[0],
        expected_clock[0],
    )
    torch.testing.assert_close(
        output.tf_condition_tokens[1],
        torch.zeros_like(output.tf_condition_tokens[1]),
    )
    telemetry = output.tf_condition_telemetry
    torch.testing.assert_close(
        telemetry["clock_residual_rms"],
        _rms(output.tf_condition_tokens),
    )
    assert telemetry["clock_injection_fraction"] == 0.5
    assert telemetry["state_residual_rms"] == 0


def test_default_clock_mask_matches_explicit_enabled_mask():
    torch.manual_seed(31)
    model = _make_dual_model()
    with torch.no_grad():
        model.tf_clock_embedding.gate.fill_(0.4)
    arguments = (
        torch.randn(2, 16, 2, 4, 4),
        torch.tensor([100.0, 200.0]),
        torch.randn(2, 2, 3),
        torch.randn(2, 16, 2, 4, 4),
        [torch.zeros(1, 4), torch.zeros(1, 4)],
    )
    noisy_tf = torch.randn(2, 4, 2, 4, 4)
    kwargs = {
        "noisy_tf": noisy_tf,
        "conditioning_tf": noisy_tf,
        "tf_sigma": torch.tensor([0.3, 0.6]),
        "condition_on_tf": False,
    }

    default_output = model(*arguments, **kwargs)
    enabled_output = model(
        *arguments,
        **kwargs,
        condition_on_tf_clock=True,
    )

    torch.testing.assert_close(
        default_output.tf_condition_tokens,
        enabled_output.tf_condition_tokens,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        default_output.video_velocity,
        enabled_output.video_velocity,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "condition_on_tf_clock,exception,match",
    [
        (
            torch.tensor([[True], [False]]),
            ValueError,
            r"shape \[batch\]",
        ),
        (
            torch.tensor([1.0, 0.5]),
            ValueError,
            "exactly zero or one",
        ),
        (1, TypeError, "must be bool"),
    ],
)
def test_clock_mask_rejects_ambiguous_values(
    condition_on_tf_clock,
    exception,
    match,
):
    model = _make_dual_model()
    with pytest.raises(exception, match=match):
        model(
            torch.randn(2, 16, 2, 4, 4),
            torch.tensor([100.0, 200.0]),
            torch.randn(2, 2, 3),
            torch.randn(2, 16, 2, 4, 4),
            [torch.zeros(1, 4), torch.zeros(1, 4)],
            noisy_tf=torch.randn(2, 4, 2, 4, 4),
            conditioning_tf=torch.randn(2, 4, 2, 4, 4),
            tf_sigma=torch.tensor([0.3, 0.6]),
            condition_on_tf_clock=condition_on_tf_clock,
        )
