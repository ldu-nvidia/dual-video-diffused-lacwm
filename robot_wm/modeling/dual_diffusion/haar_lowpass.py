"""Cheap low-frequency video targets for deployable dual diffusion.

The representation in this module is deliberately *not* a complete Fourier
transform.  It keeps only a per-view spatial box low-pass and the two
coarsest temporal Haar coordinates: a DC/static coordinate and a signed
first-half-to-second-half motion coordinate.  It therefore gives the
auxiliary branch a small, easy-to-denoise structural scratchpad instead of a
second copy of every pixel degree of freedom.

No learned encoder or external checkpoint is used.  During training the
target is computed from the same RGB clip that supplies the video-flow target.
During autonomous inference the transform is not called: the six-channel
state starts from Gaussian noise and is generated jointly with the Wan latent.
"""

from __future__ import annotations

from math import log2
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PerViewHaarLowpass(nn.Module):
    """Per-view RGB DC/motion targets aligned to the Wan latent grid.

    Input is ``[B,T,C,H,W_total]`` with views concatenated along width.  For
    the production contract ``T=13``, ``window_size=4``, three RGB views, and
    ``output_size=(24,120)``, output is ``[B,6,4,24,120]``.

    Let ``P_t`` be the independently spatially averaged frame for one view.
    For each four-frame tail window ``(P_0,P_1,P_2,P_3)`` we retain

    ``DC = (P_0 + P_1 + P_2 + P_3) / 2``

    ``motion = (P_2 + P_3 - P_0 - P_1) / 2``.

    These are the orthonormal length-four Haar scaling coordinate and its
    coarsest signed detail.  The singleton anchor bin repeats frame zero, so
    its coordinates are ``DC=2*P_0`` and ``motion=0``.  Channel order is all
    input-channel DC values followed by all motion values.

    Spatial pooling is required to be an integer power-of-two box average
    after right/bottom padding.  It is consequently equal to repeated Haar LL
    averaging, but keeps mean-valued rather than orthonormally amplified
    coefficients so its scale stays compatible with normalized RGB.
    """

    bidirectional = True

    def __init__(
        self,
        *,
        num_views: int = 3,
        output_size: Tuple[int, int] = (24, 120),
        window_size: int = 4,
        pad_multiple: int | None = 16,
        pad_value: float = -1.0,
    ) -> None:
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be positive")
        if window_size != 4:
            raise ValueError(
                "PerViewHaarLowpass currently requires window_size=4"
            )
        if len(output_size) != 2 or any(value < 1 for value in output_size):
            raise ValueError("output_size must contain two positive values")
        if output_size[1] % num_views:
            raise ValueError("output width must be divisible by num_views")
        if pad_multiple is not None and pad_multiple < 1:
            raise ValueError("pad_multiple must be positive or None")
        self.num_views = int(num_views)
        self.output_size = tuple(int(value) for value in output_size)
        self.window_size = int(window_size)
        self.pad_multiple = (
            None if pad_multiple is None else int(pad_multiple)
        )
        self.pad_value = float(pad_value)

    @staticmethod
    def output_channels(input_channels: int) -> int:
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        return 2 * int(input_channels)

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and value == 1 << int(log2(value))

    def _split_and_lowpass(self, video: Tensor) -> Tensor:
        batch, frames, channels, height, total_width = video.shape
        if total_width % self.num_views:
            raise ValueError("input width must be divisible by num_views")
        view_width = total_width // self.num_views
        output_height, output_total_width = self.output_size
        output_view_width = output_total_width // self.num_views

        views = video.reshape(
            batch, frames, channels, height, self.num_views, view_width
        ).permute(0, 4, 1, 2, 3, 5)
        flattened = views.reshape(
            batch * self.num_views * frames,
            channels,
            height,
            view_width,
        )
        if self.pad_multiple is not None:
            pad_height = (-height) % self.pad_multiple
            pad_width = (-view_width) % self.pad_multiple
            flattened = F.pad(
                flattened,
                (0, pad_width, 0, pad_height),
                mode="constant",
                value=self.pad_value,
            )
        padded_height, padded_width = flattened.shape[-2:]
        if (
            padded_height % output_height
            or padded_width % output_view_width
        ):
            raise ValueError(
                "padded view size must be divisible by the requested output grid"
            )
        stride_height = padded_height // output_height
        stride_width = padded_width // output_view_width
        if not (
            self._is_power_of_two(stride_height)
            and self._is_power_of_two(stride_width)
        ):
            raise ValueError(
                "spatial low-pass ratios must be powers of two for the Haar contract"
            )
        pooled = F.avg_pool2d(
            flattened,
            kernel_size=(stride_height, stride_width),
            stride=(stride_height, stride_width),
        )
        return pooled.reshape(
            batch,
            self.num_views,
            frames,
            channels,
            output_height,
            output_view_width,
        )

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"expected video [B,T,C,H,W], received {tuple(video.shape)}"
            )
        if not video.is_floating_point():
            raise TypeError("video must be floating point")
        batch, frames, channels, _, _ = video.shape
        if (frames - 1) % self.window_size:
            raise ValueError(
                f"frames must satisfy T=1+k*{self.window_size}; got {frames}"
            )

        pooled = self._split_and_lowpass(video)
        _, _, _, _, output_height, output_view_width = pooled.shape
        temporal_bins = 1 + (frames - 1) // self.window_size
        anchor = pooled[:, :, :1].expand(
            -1, -1, self.window_size, -1, -1, -1
        )
        tail = pooled[:, :, 1:].reshape(
            batch,
            self.num_views,
            temporal_bins - 1,
            self.window_size,
            channels,
            output_height,
            output_view_width,
        )
        windows = torch.cat([anchor.unsqueeze(2), tail], dim=2)
        # Accumulate in FP32 under bf16/fp16 training, then return the input
        # dtype so the existing dual-flow path stays AMP-compatible.
        compute = windows.float() if windows.dtype in {
            torch.float16,
            torch.bfloat16,
        } else windows
        dc = compute.sum(dim=3) * 0.5
        motion = (
            compute[:, :, :, 2:].sum(dim=3)
            - compute[:, :, :, :2].sum(dim=3)
        ) * 0.5
        coordinates = torch.cat([dc, motion], dim=3)
        output = coordinates.permute(0, 3, 2, 4, 1, 5).reshape(
            batch,
            self.output_channels(channels),
            temporal_bins,
            output_height,
            self.output_size[1],
        )
        return output.to(dtype=video.dtype)
