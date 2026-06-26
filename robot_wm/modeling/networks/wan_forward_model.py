# Wan2.1-Fun-1.3B-Control as the forward (world) model for the latent-action world model.
#
# Replaces STForwardModel. The frozen Wan DiT is LoRA-fine-tuned; a small trainable
# ActionToControl module maps the per-latent-frame latent action (64-d) into the Wan
# Fun-Control "control" latent channel (16-ch). History frames go in the "reference"
# channel. The two are channel-concatenated into `y` (32-ch); with the 16-ch noisy
# latent the DiT sees the expected in_dim=48.
#
# y = concat([control(16), reference(16)], dim=channel)  ->  transformer(x=noise, y=y, ...)

import logging
import math
import os

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig, inject_adapter_in_model

from videox_fun.models.wan_transformer3d import WanTransformer3DModel

logger = logging.getLogger(__name__)


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
        model_path: str = "/scr/ravenh/wan_fun_1.3b_control",
        config_path: str = "/scr/ravenh/VideoX-Fun/config/wan2.1/wan_civitai.yaml",
        latent_action_dim: int = 64,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        control_hidden: int = 256,
        gradient_checkpointing: bool = True,
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
    ) -> torch.Tensor:
        n, c, fp, h, w = noisy_latents.shape
        control = self.action_to_control(z_control, h, w).to(noisy_latents.dtype)
        y = torch.cat([control, ref_latents], dim=1)  # [N, 32, Fp, h, w]
        seq_len = int(math.ceil(h * w / (self.patch_size[1] * self.patch_size[2])) * fp)
        out = self.transformer(
            x=noisy_latents,
            t=timesteps,
            context=context,
            seq_len=seq_len,
            y=y,
            clip_fea=clip_fea,
        )
        return out[0] if isinstance(out, (list, tuple)) else out
