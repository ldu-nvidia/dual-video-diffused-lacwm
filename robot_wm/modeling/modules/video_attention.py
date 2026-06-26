# References:
#     https://github.com/myscience/open-genie/blob/732b9f9b746f18fff1a0fb22f83638224f2f7cc6/genie/module/attention.py

import math
from typing import Optional

import torch
from einops import pack, rearrange, repeat, unpack

from robot_wm.modeling.modules.attention import Attention


class VideoAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = True,
    ):
        assert is_causal is True  # attention should be causal over time
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
        )

    @staticmethod
    def _get_causal_attn_mask(x: torch.Tensor) -> torch.Tensor:
        """Returns a block causal attention mask.
        Args:
            x: torch.Tensor [n t h w d]
        Returns:
            torch.Tensor [(t h w) (t h w)]
        """
        device = x.device
        t = x.shape[1]
        hw = math.prod(x.shape[2:-1])
        mask = torch.ones(t, t, device=device, dtype=torch.bool).tril(diagonal=0)
        mask = repeat(mask, "i j -> (i ir) (j jr)", ir=hw, jr=hw)
        return mask.contiguous()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [n t h w d]
            freqs_cis: torch.Tensor [max_len head_dim 2]
        Returns:
            torch.Tensor [n t h w d]
        """
        mask = self._get_causal_attn_mask(x)
        x, ps_thw = pack([x], "n * d")  # n (t h w) d
        x = super().forward(x, freqs_cis, mask)  # n (t h w) d
        x = unpack(x, ps_thw, "n * d")[0]  # n t h w d
        return x


class SpatialAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = False,
    ):
        assert is_causal is False  # attend over the full spatial extent
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [n t * d]
            freqs_cis: torch.Tensor [max_len head_dim 2]
        Returns:
            torch.Tensor [n t * d]
        """
        x, ps_hw = pack([x], "n t * d")  # n t (h w) d
        x, ps_nt = pack([x], "* hw d")  # (n t) (h w) d
        x = super().forward(x, freqs_cis)  # (n t) (h w) d
        x = unpack(x, ps_nt, "* hw d")[0]  # n t (h w) d
        x = unpack(x, ps_hw, "n t * d")[0]  # n t h w d
        return x


class TemporalAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = True,
    ):
        assert is_causal is True  # attention should be causal over time
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [n t * d]
            freqs_cis: torch.Tensor [max_len head_dim 2]
        Returns:
            torch.Tensor [n t * d]
        """
        x, ps_hw = pack([x], "n t * d")  # n t (h w) d
        x = rearrange(x, "n t hw d -> n hw t d")  # n hw t d
        x, ps_nhw = pack([x], "* t d")  # (n h w) t d
        x = super().forward(x, freqs_cis)  # (n h w) t d
        x = unpack(x, ps_nhw, "* t d")[0]  # n h w t d
        x = rearrange(x, "n hw t d -> n t hw d")  # n t h w d
        x = unpack(x, ps_hw, "n t * d")[0]  # n t h w d
        return x


# usage example
if __name__ == "__main__":
    from robot_wm.modeling.modules.attention import precompute_freqs_cis

    N, T, H, W, dim, num_heads = 2, 8, 23, 40, 512, 16
    x = torch.rand(N, T, H, W, dim).to("cuda")
    print(f"{x.shape = }")

    print("-" * 40)
    print("VideoAttention")
    attn = VideoAttention(dim=dim, num_heads=num_heads).to("cuda")
    freqs_cis = precompute_freqs_cis(dim // num_heads, end=T * H * W).to("cuda")
    output = attn(x, freqs_cis)
    print(f"{freqs_cis.shape = }")
    print(f"{output.shape = }")

    # print gpu usage
    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    print(f"GPU memory (allocated): {gpu_allocated:0.1f} GB")
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    print(f"GPU memory ( reserved): {gpu_reserved:0.1f} GB")
    del attn, freqs_cis, output

    print("-" * 40)
    print("SpatialAttention")
    attn = SpatialAttention(dim=dim, num_heads=num_heads).to("cuda")
    freqs_cis = precompute_freqs_cis(dim // num_heads, end=H * W).to("cuda")
    output = attn(x, freqs_cis)
    print(f"{freqs_cis.shape = }")
    print(f"{output.shape = }")

    # print gpu usage
    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    print(f"GPU memory (allocated): {gpu_allocated:0.1f} GB")
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    print(f"GPU memory ( reserved): {gpu_reserved:0.1f} GB")
    del attn, freqs_cis, output

    print("-" * 40)
    print("TemporalAttention")
    attn = TemporalAttention(dim=dim, num_heads=num_heads).to("cuda")
    freqs_cis = precompute_freqs_cis(dim // num_heads, end=T).to("cuda")
    output = attn(x, freqs_cis)
    print(f"{freqs_cis.shape = }")
    print(f"{output.shape = }")

    # print gpu usage
    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    print(f"GPU memory (allocated): {gpu_allocated:0.1f} GB")
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    print(f"GPU memory ( reserved): {gpu_reserved:0.1f} GB")
    del attn, freqs_cis, output
