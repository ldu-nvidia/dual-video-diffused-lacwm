import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _get_padding(
    shape: Sequence[int], factors: Sequence[int], offset_time_by_one: bool
) -> Sequence[Sequence[int]]:
    """Get padding resulting in a shape evenly divisible by `factors`
    Notes:
        - Spatial dimensions (last two dims) are center padded
        - Temporal dimension (when available) is 'right' padded
    """
    assert len(shape) > len(factors)
    assert len(factors) in [2, 3]

    # get the end of shape (which will be padded)
    partial_shape = list(shape[-len(factors) :])
    if offset_time_by_one and len(partial_shape) == 3:
        partial_shape[0] -= 1

    # calculate the delta between the old and new shape
    deltas = [0] * len(factors)
    for idx, (ps, f) in enumerate(zip(partial_shape, factors)):
        ns = int(math.ceil(ps / f) * f)
        deltas[idx] = ns - ps

    # divide last two dims by 2 for center padding (for spatial dims)
    if not all(p % 2 == 0 for p in deltas[-2:]):
        raise NotImplementedError("irregular padding is not supported")
    deltas[-2:] = [p // 2 for p in deltas[-2:]]

    if len(factors) == 2:
        padding = [[deltas[i], deltas[i]] for i in range(len(deltas))]
        return padding
    elif len(factors) == 3:
        padding = [[deltas[i], deltas[i]] for i in range(len(deltas))]
        padding[0][0] = 0  # fix for 'right' padding
        return padding
    else:
        raise ValueError(f"invalid {len(factors) =}")


def _get_reverse_padding(
    new_shape: Sequence[int], padding: Sequence[int]
) -> Sequence[slice]:
    """Returns a vector that can be used to undo padding."""
    prefix_len = len(new_shape) - len(padding)
    prefix = [[0, 0] for _ in range(prefix_len)]
    reverse_padding = prefix + padding
    for i in range(len(reverse_padding)):
        reverse_padding[i][1] = new_shape[i] - reverse_padding[i][1]
    reverse_padding = [slice(start, stop) for start, stop in reverse_padding]
    return reverse_padding


class DynamicPad(nn.Module):
    def __init__(self, factors: Sequence[int], offset_time_by_one: bool = True):
        super().__init__()
        assert len(factors) in [2, 3]
        if len(factors) == 2:
            factors = [1] + list(factors)
        self.factors = list(factors)
        self.offset_time_by_one = offset_time_by_one

    def reverse(self, xhat: torch.tensor, out_shape: Sequence[int]) -> torch.Tensor:
        """Removes padding from `xhat`.
        Args:
            xhat: torch.Tensor [N, *, C, Hp, Wp]
            out_shape: Sequence[int]
        Returns:
            torch.Tensor [N, *, C, H, W]
        """
        out_shape = list(out_shape)
        assert xhat.ndim == len(out_shape)
        orig_shape = list(out_shape)
        if len(orig_shape) == 4:
            xhat = xhat.unsqueeze(1)
            out_shape = out_shape.insert(1, 1)
        out_shape[1], out_shape[2] = out_shape[2], out_shape[1]
        xhat = rearrange(xhat, "N T C Hp Wp -> N C T Hp Wp")
        padding = _get_padding(
            out_shape, self.factors, offset_time_by_one=self.offset_time_by_one
        )
        reverse = _get_reverse_padding(xhat.shape, padding)
        xhat = xhat[reverse]
        xhat = rearrange(xhat, "N C T H W -> N T C H W")
        if len(orig_shape) == 4:
            xhat = xhat.squeeze(1)
        assert list(xhat.shape) == orig_shape, f"{list(xhat.shape) = } {orig_shape = }"
        return xhat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a padded tensor.
        Args:
            x: torch.Tensor [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, C, Hp, Wp]
        """
        orig_shape = x.shape
        if len(orig_shape) == 4:
            x = x.unsqueeze(1)
        x = rearrange(x, "N T C H W -> N C T H W")
        padding = _get_padding(
            x.shape, self.factors, offset_time_by_one=self.offset_time_by_one
        )
        x = F.pad(x, [p for ps in reversed(padding) for p in ps], "constant", 0)
        x = rearrange(x, "N C T Hp Wp -> N T C Hp Wp")
        if len(orig_shape) == 4:
            x = x.squeeze(1)
        return x


if __name__ == "__main__":
    N, C, T, H, W = 1, 3, 9, 180, 320
    rgb = torch.randn(N, T, C, H, W).to("cuda").to(torch.bfloat16)
    print(f"{rgb.shape = }")

    factors = (8, 16, 16)
    pad = DynamicPad(factors=factors)

    rgb_padded = pad.forward(rgb)
    print(f"{rgb_padded.shape = }")

    rgb_hat = pad.reverse(rgb_padded, rgb.shape)
    print(f"{rgb_hat.shape = }")

    assert torch.equal(rgb, rgb_hat)
