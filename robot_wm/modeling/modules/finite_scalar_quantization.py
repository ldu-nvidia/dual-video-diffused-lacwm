# References:
#     https://arxiv.org/pdf/2309.15505v2 (Appendix A.1)
#     https://github.com/lucidrains/vector-quantize-pytorch/blob/4380fe8b7b4435c28544955639563a00db47e8e6/vector_quantize_pytorch/finite_scalar_quantization.py

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round using straight through estimator."""
    return z + (torch.round(z) - z).detach()


class FSQ(nn.Module):
    def __init__(self, levels: List[int], use_float32: bool = True):
        """`levels` suggested in (https://arxiv.org/abs/2309.15505):
        - [8, 6, 5]
        - [8, 5, 5, 5]
        - [7, 5, 5, 5, 5]
        - [8, 8, 8, 6, 5]
        - [8, 8, 8, 5, 5, 5]
        """
        super().__init__()
        self.use_float32 = use_float32

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

        self.codebook_size = torch.prod(self._levels)
        logger.info(f"codebook size: {self.codebook_size:,}")

    def init_weights(self):
        pass

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        assert zhat.shape[-1] == len(self._levels)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.shape[-1] == len(self._levels)
        orig_dtype = z.dtype
        if self.use_float32:
            z = z.to(torch.float32)
        codes = self.quantize(z)
        codes = codes.to(orig_dtype)

        indices = self.codes_to_indices(codes)
        outputs = {
            "zhat": codes,
            "loss": self._zero,
            "indices": indices,
            "logs": {},
        }

        return outputs


# usage example
if __name__ == "__main__":
    levels = [8, 8, 8, 6, 5]
    fsq = FSQ(levels)

    N, L = 3, 5
    z = 2 * (torch.rand(N, L, len(levels)) - 0.5)
    print(f"{z.shape = }")

    fsq_out = fsq(z)
    print(f"{fsq_out['zhat'].shape = }")
    print(f"{fsq_out['loss'].shape = }")
    print(f"{fsq_out['indices'].shape = }")
