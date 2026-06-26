# References:
#     Genie: Generative Interactive Environments (https://arxiv.org/abs/2402.15391)
#     Latent Action Pretraining from Videos (https://arxiv.org/abs/2410.11758)
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py

import logging
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from robot_wm.modeling.modules.attention import precompute_freqs_cis
from robot_wm.modeling.modules.video_blocks import (
    STAttentionBlock,
    STCrossAttentionBlock,
    VideoCrossAttentionBlock,
    SpatialAttentionBlock,
    SpatialCrossAttentionBlock,
)

logger = logging.getLogger(__name__)


class STForwardModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        x_dim: int,
        context_dim: int,
        output_dim: int,
        rope_end: int,
        rope_theta: int,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        mlp_drop_prob: float = 0.0,
        residual_drop_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
        mid_checkpointing: bool = False,
        aggressive_checkpointing: bool = False,
        use_video_cross_attention: bool = False,
        normalize_outputs: bool = False,
        use_spatial_attention_only: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.x_dim = x_dim
        self.x_proj = nn.Linear(x_dim, dim, bias=False)
        self.context_proj = nn.Linear(context_dim, dim, bias=False)
        self.normalize_outputs = normalize_outputs

        # Add dropout layers
        self.proj_dropout = nn.Dropout(p=proj_drop_prob)
        assert not (
            aggressive_checkpointing and mid_checkpointing
        ), "Only one of mid_checkpointing and aggressive_checkpointing can be True"

        # If we use aggressive_checkpointing, we checkpoint each block, if not we checkpoint each attention inside the block
        self.use_checkpointing = aggressive_checkpointing

        assert not (
            use_video_cross_attention and use_spatial_attention_only
        ), "Only one of use_video_cross_attention and use_spatial_attention_only can be True"
        
        # Select attention and cross-attention architecture
        AttentionBlock = STAttentionBlock
        CrossAttentionBlock = STCrossAttentionBlock        
        if use_video_cross_attention:
            CrossAttentionBlock = VideoCrossAttentionBlock
        if use_spatial_attention_only:
            AttentionBlock = SpatialAttentionBlock
            CrossAttentionBlock = SpatialCrossAttentionBlock

        # Calculate drop path rates - linear decay across all blocks
        total_blocks = num_layers * 3
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.layers = nn.ModuleDict()
        for id in range(num_layers):
            block_idx = id * 3
            
            self.layers[f"{id}:query_attn"] = AttentionBlock(
                dim=dim,
                num_heads=num_heads,
                attn_drop_prob=attn_drop_prob,
                proj_drop_prob=proj_drop_prob,
                mlp_drop_prob=mlp_drop_prob,
                residual_drop_prob=residual_drop_prob,
                drop_path=dpr[block_idx],
                use_checkpointing=mid_checkpointing,
            )
            self.layers[f"{id}:context_attn"] = AttentionBlock(
                dim=dim,
                num_heads=num_heads,
                attn_drop_prob=attn_drop_prob,
                proj_drop_prob=proj_drop_prob,
                mlp_drop_prob=mlp_drop_prob,
                residual_drop_prob=residual_drop_prob,
                drop_path=dpr[block_idx + 1],
                use_checkpointing=mid_checkpointing,
            )
            self.layers[f"{id}:cross_attn"] = CrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                attn_drop_prob=attn_drop_prob,
                proj_drop_prob=proj_drop_prob,
                mlp_drop_prob=mlp_drop_prob,
                residual_drop_prob=residual_drop_prob,
                drop_path=dpr[block_idx + 2],
                use_checkpointing=mid_checkpointing,
            )
        self.norm = build_norm(dim)
        self.output_proj = nn.Linear(dim, output_dim, bias=False)
        self.register_buffer(
            name="freqs_cis",
            tensor=precompute_freqs_cis(
                dim=dim // num_heads, end=rope_end, theta=rope_theta
            ),
            persistent=False,
        )

    def init_weights(self):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.x_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.context_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, mean=0.0, std=0.02)
        for idx, layer in enumerate(self.layers.values()):
            init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            layer.init_weights(init_std=init_std)

    def maybe_checkpoint(self, func, *args):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)
        return func(*args)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, T, H, W, D]
            context: torch.Tensor [N, T, *, D]
        Returns:
            torch.Tensor [N, T, H, W, D]
        """
        if context.ndim == 3:
            context = context.unsqueeze(2)
        x = self.x_proj(x)
        x = self.proj_dropout(x)
        context = self.context_proj(context)
        context = self.proj_dropout(context)

        for key, layer in self.layers.items():
            if key.endswith("query_attn"):
                x = self.maybe_checkpoint(layer, x, self.freqs_cis)
            elif key.endswith("context_attn"):
                context = self.maybe_checkpoint(layer, context, self.freqs_cis)
            elif key.endswith("cross_attn"):
                x = self.maybe_checkpoint(layer, x, context)
            else:
                raise ValueError("invalid layer name")

        x = self.norm(x)
        x = self.output_proj(x)
        x = self.proj_dropout(x)
        if self.normalize_outputs:
            x = F.layer_norm(x, (x.size(-1),)) 
        return x


def get_arch(arch: str, **kwargs) -> STForwardModel:
    if arch == "small":
        kwargs["dim"] = 512
        kwargs["num_heads"] = 16
        kwargs["num_layers"] = 8
    elif arch == "base":
        kwargs["dim"] = 768
        kwargs["num_heads"] = 16
        kwargs["num_layers"] = 12
    elif arch == "large":
        kwargs["dim"] = 1024
        kwargs["num_heads"] = 16
        kwargs["num_layers"] = 18
    elif arch == "huge":
        kwargs["dim"] = 1536
        kwargs["num_heads"] = 24
        kwargs["num_layers"] = 18
    else:
        raise ValueError(f"invalid arch: {arch}")
    return STForwardModel(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)-8s %(name)s %(asctime)s %(message)s", level=logging.INFO
    )

    N, T, H, W, Dq = 7, 9, 23, 40, 16
    x = torch.randn(N, T, H, W, Dq).to("cuda")
    logger.info(f"{x.shape = }")

    A, Dc = 2, 6
    context = torch.randn(N, T, A, Dc).to("cuda")
    logger.info(f"{context.shape = }")

    dim, x_dim, context_dim = 512, Dq, Dc
    num_layers, num_heads = 8, 16
    rope_end, rope_theta = max(T, H * W), 50_000
    forward_model = STForwardModel(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        x_dim=x_dim,
        context_dim=context_dim,
        output_dim=x_dim,
        rope_end=rope_end,
        rope_theta=rope_theta,
    )
    forward_model = forward_model.to("cuda")

    output = forward_model(x, context)
    logger.info(f"{output.shape = }")

    # print gpu usage
    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    logger.info(f"GPU memory (allocated): {gpu_allocated:0.1f} GB")
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    logger.info(f"GPU memory ( reserved): {gpu_reserved:0.1f} GB")
