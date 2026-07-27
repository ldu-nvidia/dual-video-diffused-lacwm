# Wan2.1-Fun-1.3B-Control as the forward (world) model for the latent-action world model.
#
# The frozen Wan DiT is LoRA-fine-tuned; a small trainable
# ActionToControl module maps the per-latent-frame latent action (64-d) into the Wan
# Fun-Control "control" latent channel (16-ch). History frames go in the "reference"
# channel. The two are channel-concatenated into `y` (32-ch); with the 16-ch noisy
# latent the DiT sees the expected in_dim=48.
#
# y = concat([control(16), reference(16)], dim=channel)  ->  transformer(x=noise, y=y, ...)

import logging
import math
import os
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig, inject_adapter_in_model

from videox_fun.models.wan_transformer3d import WanTransformer3DModel

from robot_wm.modeling.dual_diffusion.adapters import (
    TFSigmaTokenEmbedding,
    TFVelocityHead,
    ZeroInitTFTokenAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class DualWanOutput:
    """Velocities predicted from one shared Wan trunk evaluation."""

    video_velocity: torch.Tensor
    tf_velocity: torch.Tensor
    tf_condition_tokens: torch.Tensor
    tf_condition_telemetry: Mapping[str, torch.Tensor]


class ActionToControl(nn.Module):
    """Project a per-(latent-)frame latent action vector into a 16-ch control latent,
    broadcast over the spatial grid. Zero-initialized so it starts as a no-op."""

    def __init__(self, action_dim: int = 64, latent_ch: int = 16, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_ch),
        )

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.net[0].weight, std=init_std)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)  # zero-init last -> control starts as no-op
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # z: [N, Fp, action_dim] -> [N, latent_ch, Fp, h, w]
        c = self.net(z)  # [N, Fp, latent_ch]
        c = rearrange(c, "n f d -> n d f")
        c = c[:, :, :, None, None].expand(-1, -1, -1, h, w)
        return c


class WanForwardModel(nn.Module):
    def __init__(
        self,
        model_path: str = os.environ.get("WAN_DIR", "/scr/ravenh/wan_fun_1.3b_control"),
        config_path: str = os.path.join(os.environ.get("VIDEOX_HOME", "/scr/ravenh/VideoX-Fun"), "config/wan2.1/wan_civitai.yaml"),
        latent_action_dim: int = 64,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        control_hidden: int = 256,
        gradient_checkpointing: bool = True,
        dual_diffusion: Mapping | None = None,
    ):
        super().__init__()
        cfg = OmegaConf.load(config_path)
        subpath = cfg["transformer_additional_kwargs"].get("transformer_subpath", "./")
        self.transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_path, subpath),
            transformer_additional_kwargs=OmegaConf.to_container(
                cfg["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
        )
        self.patch_size = self.transformer.config.patch_size

        # freeze the full DiT, then inject + unfreeze LoRA adapters
        for p in self.transformer.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q", "k", "v", "o"],
        )
        self.transformer = inject_adapter_in_model(lora_cfg, self.transformer)
        for n, p in self.transformer.named_parameters():
            p.requires_grad = "lora_" in n
            if "lora_" in n:
                # keep LoRA master weights in fp32: the trainer's GradScaler cannot
                # unscale bf16 grads, and autocast casts them to bf16 in the forward.
                p.data = p.data.float()

        if gradient_checkpointing:
            try:
                self.transformer.enable_gradient_checkpointing()
            except Exception as e:  # noqa
                self.transformer.gradient_checkpointing = True
                logger.warning(f"enable_gradient_checkpointing fallback: {e}")

        self.action_to_control = ActionToControl(latent_action_dim, 16, control_hidden)

        dual_config = dict(dual_diffusion or {})
        self.dual_diffusion_enabled = bool(dual_config.get("enabled", False))
        self.condition_on_tf = bool(dual_config.get("condition_on_tf", False))
        if self.dual_diffusion_enabled:
            if self.transformer.control_adapter is not None:
                raise RuntimeError(
                    "dual diffusion requires the pretrained Wan control_adapter seam "
                    "to be unused"
                )
            self.transformer.control_adapter = nn.Identity()
            hidden_size = int(self.transformer.dim)
            patch_size = tuple(int(value) for value in self.transformer.patch_size)
            tf_channels = int(dual_config.get("tf_channels", 12))
            self.tf_token_adapter = ZeroInitTFTokenAdapter(
                tf_channels=tf_channels,
                hidden_size=hidden_size,
                patch_size=patch_size,
                gate_init=float(dual_config.get("state_gate_init", 0.0)),
                gate_trainable=bool(
                    dual_config.get("state_gate_trainable", True)
                ),
            )
            self.tf_clock_embedding = TFSigmaTokenEmbedding(
                hidden_size=hidden_size,
                embedding_dim=int(dual_config.get("clock_embedding_dim", 128)),
                gate_init=float(dual_config.get("clock_gate_init", 0.0)),
                gate_trainable=bool(
                    dual_config.get("clock_gate_trainable", True)
                ),
            )
            self.tf_velocity_head = TFVelocityHead(
                hidden_size=hidden_size,
                tf_channels=tf_channels,
                patch_size=patch_size,
            )

    def init_weights(self):
        self.action_to_control.init_weights()

    def forward(
        self,
        noisy_latents: torch.Tensor,  # [N, 16, Fp, h, w]
        timesteps: torch.Tensor,      # [N]
        z_control: torch.Tensor,      # [N, Fp, action_dim]
        ref_latents: torch.Tensor,    # [N, 16, Fp, h, w]
        context,                      # list of [L, text_dim]
        clip_fea: torch.Tensor = None,
        noisy_tf: torch.Tensor = None,
        conditioning_tf: torch.Tensor = None,
        tf_sigma: torch.Tensor = None,
        condition_on_tf: bool | None = None,
    ) -> torch.Tensor | DualWanOutput:
        n, c, fp, h, w = noisy_latents.shape
        control = self.action_to_control(z_control, h, w).to(noisy_latents.dtype)
        y = torch.cat([control, ref_latents], dim=1)  # [N, 32, Fp, h, w]
        seq_len = int(math.ceil(h * w / (self.patch_size[1] * self.patch_size[2])) * fp)
        if not self.dual_diffusion_enabled:
            if (
                noisy_tf is not None
                or conditioning_tf is not None
                or tf_sigma is not None
            ):
                raise RuntimeError("TF inputs require dual_diffusion.enabled=true")
            out = self.transformer(
                x=noisy_latents,
                t=timesteps,
                context=context,
                seq_len=seq_len,
                y=y,
                clip_fea=clip_fea,
            )
            return out[0] if isinstance(out, (list, tuple)) else out

        if noisy_tf is None or tf_sigma is None:
            raise ValueError("dual diffusion requires noisy_tf and tf_sigma")
        if noisy_tf.shape[0] != n or noisy_tf.shape[2:] != (fp, h, w):
            raise ValueError(
                "TF state must share the video batch and latent grid; "
                f"got video={tuple(noisy_latents.shape)}, TF={tuple(noisy_tf.shape)}"
            )
        noisy_state_tokens, grid = self.tf_token_adapter.project_tokens(noisy_tf)
        if noisy_state_tokens.shape[1] != seq_len:
            raise RuntimeError(
                f"TF token count {noisy_state_tokens.shape[1]} does not match Wan {seq_len}"
            )
        if conditioning_tf is None or conditioning_tf is noisy_tf:
            condition_state_tokens = noisy_state_tokens
        else:
            if (
                conditioning_tf.shape[0] != n
                or conditioning_tf.shape[1] != noisy_tf.shape[1]
                or conditioning_tf.shape[2:] != (fp, h, w)
            ):
                raise ValueError(
                    "conditioning TF state must share the noisy TF shape; "
                    f"got noisy TF={tuple(noisy_tf.shape)}, "
                    f"conditioning TF={tuple(conditioning_tf.shape)}"
                )
            condition_state_tokens, condition_grid = (
                self.tf_token_adapter.project_tokens(conditioning_tf)
            )
            if condition_grid != grid:
                raise RuntimeError(
                    "conditioning TF token grid does not match noisy TF token grid: "
                    f"{condition_grid} != {grid}"
                )
        # The clock MLP deliberately evaluates from an FP32 sigma, but its
        # residual must enter Wan in the same compute dtype as the patch/state
        # tokens.  Leaving an exactly-zero clock residual in FP32 under AMP
        # promotes ``patch_tokens + y_camera`` to FP32, so the nominal zero-gate
        # path is neither functionally nor memory-identical to pretrained Wan.
        clock_tokens = (
            self.tf_clock_embedding(tf_sigma)
            .to(dtype=condition_state_tokens.dtype)
            .unsqueeze(1)
            .expand(-1, seq_len, -1)
        )
        use_tf = self.condition_on_tf if condition_on_tf is None else bool(condition_on_tf)
        state_residual = self.tf_token_adapter.residual_tokens(
            condition_state_tokens
        ) * float(use_tf)
        injected_tokens = clock_tokens + state_residual
        features = injected_tokens.transpose(1, 2).reshape(
            n, injected_tokens.shape[-1], *grid
        )
        y_camera = [features[index : index + 1] for index in range(n)]

        captured_tokens = []
        native_patch_embeddings = []

        def capture_shared_tokens(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Wan head hook did not receive shared tokens")
            captured_tokens.append(inputs[0])

        def capture_native_patch_embedding(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Wan patch embedding hook did not receive a tensor")
            native_patch_embeddings.append(output)

        head_handle = self.transformer.head.register_forward_pre_hook(
            capture_shared_tokens
        )
        patch_handle = self.transformer.patch_embedding.register_forward_hook(
            capture_native_patch_embedding
        )
        try:
            out = self.transformer(
                x=noisy_latents,
                t=timesteps,
                context=context,
                seq_len=seq_len,
                y=y,
                y_camera=y_camera,
                clip_fea=clip_fea,
            )
        finally:
            head_handle.remove()
            patch_handle.remove()
        if len(captured_tokens) != 1:
            raise RuntimeError(
                f"expected one Wan shared-token capture, got {len(captured_tokens)}"
            )
        if not native_patch_embeddings:
            raise RuntimeError("Wan patch-embedding hook did not capture any tokens")
        shared_tokens = captured_tokens[0]
        if shared_tokens.shape != noisy_state_tokens.shape:
            raise RuntimeError(
                "Wan shared-token shape does not match TF token grid: "
                f"{tuple(shared_tokens.shape)} != {tuple(noisy_state_tokens.shape)}"
            )
        # Both ablation arms give the TF head its own noisy state.  The causal
        # difference is only whether that state entered the shared video trunk.
        tf_velocity = self.tf_velocity_head(
            shared_tokens + noisy_state_tokens, grid
        )
        video_velocity = out[0] if isinstance(out, (list, tuple)) else out

        def rms(value):
            return value.detach().float().square().mean().sqrt()

        native_squared_sum = torch.stack(
            [
                value.detach().float().square().sum()
                for value in native_patch_embeddings
            ]
        ).sum()
        native_count = sum(value.numel() for value in native_patch_embeddings)
        native_patch_rms = (native_squared_sum / native_count).sqrt()
        combined_rms = rms(injected_tokens)
        telemetry = {
            "raw_state_rms": rms(condition_state_tokens),
            "state_residual_rms": rms(state_residual),
            "clock_residual_rms": rms(clock_tokens),
            "combined_rms": combined_rms,
            "native_patch_embedding_rms": native_patch_rms,
            "state_to_native_ratio": rms(state_residual)
            / native_patch_rms.clamp_min(torch.finfo(torch.float32).tiny),
            "combined_to_native_ratio": combined_rms
            / native_patch_rms.clamp_min(torch.finfo(torch.float32).tiny),
        }
        return DualWanOutput(
            video_velocity=video_velocity,
            tf_velocity=tf_velocity,
            tf_condition_tokens=injected_tokens,
            tf_condition_telemetry=telemetry,
        )
