# References:
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py
#     https://github.com/pytorch/torchtitan/blob/8a92fb6649de607fbcf5983d13571ca91da593f2/torchtitan/models/llama_multimodal/model.py

from typing import Callable

import torch
import torch.nn as nn

from robot_wm.modeling.modules.attention import Attention
from robot_wm.modeling.modules.cross_attention import CrossAttention
from robot_wm.modeling.modules.mlp import MLP


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = False,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
        mlp_ratio: float = 4.0,
        mlp_build_act: nn.Module = nn.GELU,
        mlp_drop_prob: float = 0.0,
    ):
        super().__init__()

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
        )
        self.attn_norm = build_norm(dim)

        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            build_act=mlp_build_act,
            drop_prob=mlp_drop_prob,
        )
        self.mlp_norm = build_norm(dim)

    def init_weights(self, init_std: float):
        for norm in [self.attn_norm, self.mlp_norm]:
            norm.reset_parameters()
        self.attn.init_weights(init_std=init_std)
        self.mlp.init_weights(init_std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, L, D]
            freqs_cis: torch.Tensor [L_max, D]
        Returns:
            torch.Tensor [N, L, D]
        """
        x = x + self.attn(self.attn_norm(x), freqs_cis=freqs_cis)  # N, L, D
        x = x + self.mlp(self.mlp_norm(x))  # N, L, D
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
        mlp_ratio: float = 4.0,
        mlp_build_act: nn.Module = nn.GELU,
        mlp_drop_prob: float = 0.0,
    ):
        super().__init__()

        self.attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            build_norm=build_norm,
        )
        self.attn_norm = build_norm(dim)
        # TODO: add self.attn_scale = ?

        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            build_act=mlp_build_act,
            drop_prob=mlp_drop_prob,
        )
        self.mlp_norm = build_norm(dim)
        # TODO: add self.mlp_scale = ?

    def init_weights(self, init_std: float):
        for norm in [self.attn_norm, self.mlp_norm]:
            norm.reset_parameters()
        self.attn.init_weights(init_std=init_std)
        self.mlp.init_weights(init_std=init_std)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, L, D]
            freqs_cis: torch.Tensor [L_max, D]
        Returns:
            torch.Tensor [N, L, D]
        """
        query = query + self.attn(query=self.attn_norm(query), context=context)
        query = query + self.mlp(self.mlp_norm(query))
        return query


# usage example
if __name__ == "__main__":
    from robot_wm.modeling.modules.attention import precompute_freqs_cis

    # attention block
    print("-" * 40)
    print("attention block")
    N, L, dim, num_heads = 2, 6, 32, 8
    x = torch.rand(N, L, dim).to("cuda")
    freqs_cis = precompute_freqs_cis(dim // num_heads, end=L).to("cuda")
    block = AttentionBlock(dim=dim, num_heads=num_heads).to("cuda")
    output = block(x, freqs_cis=freqs_cis)
    print(f"{x.shape = }")
    print(f"{freqs_cis.shape = }")
    print(f"{output.shape = }")
    print("-" * 40)

    # cross-attention block
    print("-" * 40)
    print("cross-attention block")
    N, Lq, Lc, dim, num_heads = 2, 6, 10, 32, 8
    query = torch.rand(N, Lq, dim).to("cuda")
    context = torch.rand(N, Lc, dim).to("cuda")
    block = CrossAttentionBlock(dim=dim, num_heads=num_heads).to("cuda")
    output = block(query=query, context=context)
    print(f"{query.shape = }")
    print(f"{context.shape = }")
    print(f"{output.shape = }")
    print("-" * 40)
