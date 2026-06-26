# References:
#     https://github.com/lucidrains/vector-quantize-pytorch/blob/976335f03b259536a06ed88a7adb7248cdb3d17c/vector_quantize_pytorch/vector_quantize_pytorch.py
#     https://arxiv.org/abs/2410.06424

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import einsum

logger = logging.getLogger(__name__)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


def safe_div(num, den, eps=1e-6):
    return num / den.clamp(min=eps)


def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        (unpacked,) = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (
        (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy)
        .clamp(min=0)
        .sqrt()
    )


def efficient_rotation_trick_transform(u, q, e):
    """Section 4.2 in https://arxiv.org/abs/2410.06424"""
    e = rearrange(e, "b d -> b 1 d")
    w = l2norm(u + q, dim=1).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    )


def rotate_to(src, tgt):
    """Rotation trick from https://arxiv.org/abs/2410.06424"""
    src, inverse = pack_one(src, "* d")
    tgt, _ = pack_one(tgt, "* d")

    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src), safe_div(tgt, norm_tgt), src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)


class VectorQuantizer(nn.Module):
    """Vector quantization layer that uses the "rotation trick," as in:
    https://arxiv.org/abs/2410.06424
    """

    def __init__(
        self,
        dim: int,
        size: int,
        use_ste: bool = False,
        is_learnable: bool = False,
        beta: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.size = size
        self.use_ste = use_ste
        self.is_learnable = is_learnable
        self.beta = beta
        embed = torch.randn(size, dim)
        if is_learnable:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)
        logger.info(f"codebook size: {self.size:,}")

    def init_weights(self):
        nn.init.kaiming_uniform_(self.embed)

    def compute_codebook_usage(self, indices):
        unique, counts = torch.unique(indices, return_counts=True)
        index_counts = torch.zeros(self.size, dtype=torch.long).to(indices.device)
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()
        return code_usage

    def forward(self, x: torch.Tensor, cb_only=False):
        """
        Args:
          x: torch.Tensor [N, *, D]

        Returns dict with:
          zhat: torch.Tensor [N, *, D]
          loss: torch.Tensor [N, *, D]
          indices: torch.Tensor [N, *]
          logs: Dict[str, Any]
        """
        if cb_only and not self.is_learnable:
            raise ValueError("cb_only is True, but is_learnable is False.")

        assert x.shape[-1] == self.dim, f"{x.shape = } {self.dim = }"
        x, inverse = pack_one(x, "N * D")  # N, *, D --> N, L, D

        # quantization
        dist = cdist(x, self.embed.unsqueeze(0))  # N, L, self.size
        indices = dist.min(dim=-1)[1]  # N, L
        codes = F.embedding(input=indices, weight=self.embed)  # N, L, self.dim

        # commitment loss
        commit_loss = F.mse_loss(x, codes.detach(), reduction="none")  # N, L, self.dim
        codebook_loss = F.mse_loss(x.detach(), codes, reduction="none")
        ci_loss = inverse(commit_loss).mean()
        cb_loss = inverse(codebook_loss).mean()

        if self.is_learnable:
            if cb_only:
                loss = codebook_loss
            else:
                loss = self.beta * commit_loss + codebook_loss
        else:
            loss = self.beta * commit_loss

        # straight-through estimator or 'rotation trick"
        if self.use_ste:
            codes = x + (codes - x).detach()  # N, L, self.dim
        else:
            codes = rotate_to(x, codes)  # N, L, self.dim

        # force D=1 for indices for inverse
        unpacked_indices = inverse(indices.unsqueeze(-1)).squeeze(-1)

        outputs = {
            "zhat": inverse(codes),
            "loss": inverse(loss).mean(),
            "indices": unpacked_indices,
        }

        code_usage = self.compute_codebook_usage(indices)
        outputs["logs"] = {
            "vq_loss": outputs["loss"].detach().clone(),
            "commit_loss": ci_loss.detach().clone(),
            "codebook_loss": cb_loss.detach().clone(),
            "code_usage": code_usage,
        }

        return outputs


if __name__ == "__main__":
    dim, size = 32, 10
    vq = VectorQuantizer(dim=dim, size=size)

    N, L = 3, 5
    z = 2.0 * torch.rand(N, L, dim) - 1.0
    print(f"{z.shape = }")

    vq_out = vq(z)
    print(f"{vq_out['zhat'].shape = }")
    print(f"{vq_out['loss'].shape = }")
    print(f"{vq_out['indices'].shape = }")
