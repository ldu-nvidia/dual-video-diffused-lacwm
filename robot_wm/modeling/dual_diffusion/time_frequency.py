"""Deterministic time-frequency targets for multi-view robot videos.

The production LACWM tensor width-stacks three camera views.  A temporal
transform must never mix those artificial view boundaries.  This module first
splits the views, downsamples each view independently, and only then computes a
localized temporal STFT at every spatial location.

Real and imaginary coefficients are retained instead of converting to wrapped
phase angles.  They preserve complete phase information while avoiding the
discontinuity at ``-pi``/``pi``.  The implementation is a clean-room adaptation
of the dual-state principle in Latent Forcing; it does not copy its image/DINO
implementation.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PerViewCausalRFFT(nn.Module):
    """Complete local RFFT aligned with Wan's causal temporal bins.

    For the production 13-frame contract and ``window_size=4``, frame 0 forms
    the causal anchor bin and frames 1--4, 5--8, and 9--12 form the remaining
    bins.  The anchor frame is repeated four times solely to give it the same
    coefficient contract.  A real length-4 signal is packed without redundant
    imaginary DC/Nyquist terms as ``[X0.real, X1.real, X1.imag, X2.real]``.
    Consequently, the transform is complete and has exactly
    ``input_channels * window_size`` output channels.
    """

    def __init__(
        self,
        *,
        num_views: int = 3,
        output_size: Tuple[int, int] = (24, 120),
        window_size: int = 4,
        pad_multiple: int | None = 16,
        pad_value: float = -1.0,
        normalization: Literal["none", "sample_rms"] = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be positive")
        if window_size < 2:
            raise ValueError("window_size must be at least two")
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError("output_size dimensions must be positive")
        if output_size[1] % num_views:
            raise ValueError("output width must be divisible by num_views")
        if pad_multiple is not None and pad_multiple < 1:
            raise ValueError("pad_multiple must be positive or None")
        if normalization not in {"none", "sample_rms"}:
            raise ValueError(f"unsupported normalization: {normalization}")
        self.num_views = num_views
        self.output_size = output_size
        self.window_size = window_size
        self.pad_multiple = pad_multiple
        self.pad_value = pad_value
        self.normalization = normalization
        self.eps = eps

    def output_channels(self, input_channels: int) -> int:
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        return input_channels * self.window_size

    @staticmethod
    def _compute_dtype(x: Tensor) -> torch.dtype:
        return torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype

    def _split_and_pool(self, video: Tensor) -> Tensor:
        b, frames, channels, height, total_width = video.shape
        if total_width % self.num_views:
            raise ValueError("input width must be divisible by num_views")
        view_width = total_width // self.num_views
        out_height, out_total_width = self.output_size
        out_view_width = out_total_width // self.num_views
        views = video.reshape(
            b, frames, channels, height, self.num_views, view_width
        ).permute(0, 4, 1, 2, 3, 5)
        flattened = views.reshape(
            b * self.num_views * frames, channels, height, view_width
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
        return F.adaptive_avg_pool2d(
            flattened,
            (out_height, out_view_width),
        ).reshape(
            b,
            self.num_views,
            frames,
            channels,
            out_height,
            out_view_width,
        )

    def _pack(self, spectrum: Tensor) -> Tensor:
        # spectrum: [B,V,Fp,C,H,Wv,K]
        parts = [spectrum[..., 0].real]
        for frequency in range(1, spectrum.shape[-1]):
            parts.append(spectrum[..., frequency].real)
            is_nyquist = self.window_size % 2 == 0 and frequency == self.window_size // 2
            if not is_nyquist:
                parts.append(spectrum[..., frequency].imag)
        if len(parts) != self.window_size:
            raise RuntimeError("internal RFFT packing contract is inconsistent")
        return torch.stack(parts, dim=4)  # [B,V,Fp,C,N,H,Wv]

    def _unpack(self, packed: Tensor) -> Tensor:
        # packed: [B,V,Fp,C,N,H,Wv]
        parts = list(packed.unbind(dim=4))
        coefficients = [torch.complex(parts[0], torch.zeros_like(parts[0]))]
        cursor = 1
        for frequency in range(1, self.window_size // 2 + 1):
            real = parts[cursor]
            cursor += 1
            is_nyquist = self.window_size % 2 == 0 and frequency == self.window_size // 2
            if is_nyquist:
                imag = torch.zeros_like(real)
            else:
                imag = parts[cursor]
                cursor += 1
            coefficients.append(torch.complex(real, imag))
        if cursor != len(parts):
            raise RuntimeError("internal RFFT unpacking contract is inconsistent")
        return torch.stack(coefficients, dim=-1)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"expected video [B,T,C,H,W], received shape {tuple(video.shape)}"
            )
        if not video.is_floating_point():
            raise TypeError("video must be floating point")
        b, frames, channels, _, _ = video.shape
        if (frames - 1) % self.window_size:
            raise ValueError(
                f"frames must satisfy T=1+k*{self.window_size}; received T={frames}"
            )

        pooled = self._split_and_pool(video)
        _, _, _, _, out_height, out_view_width = pooled.shape
        temporal_bins = 1 + (frames - 1) // self.window_size
        anchor = pooled[:, :, :1].expand(-1, -1, self.window_size, -1, -1, -1)
        tail = pooled[:, :, 1:].reshape(
            b,
            self.num_views,
            temporal_bins - 1,
            self.window_size,
            channels,
            out_height,
            out_view_width,
        )
        windows = torch.cat([anchor.unsqueeze(2), tail], dim=2)
        compute = windows.to(dtype=self._compute_dtype(windows))
        spectrum = torch.fft.rfft(compute, dim=3, norm="ortho").movedim(3, -1)
        packed = self._pack(spectrum)
        coefficients = packed.permute(0, 1, 3, 4, 2, 5, 6).reshape(
            b,
            self.num_views,
            self.output_channels(channels),
            temporal_bins,
            out_height,
            out_view_width,
        )
        if self.normalization == "sample_rms":
            rms = coefficients.square().mean(dim=(3, 4, 5), keepdim=True).sqrt()
            coefficients = coefficients / rms.clamp_min(self.eps)
        out_total_width = self.output_size[1]
        return coefficients.permute(0, 2, 3, 4, 1, 5).reshape(
            b,
            self.output_channels(channels),
            temporal_bins,
            out_height,
            out_total_width,
        ).to(dtype=video.dtype)

    def inverse(self, coefficients: Tensor) -> Tensor:
        """Invert unnormalized coefficients to the spatially pooled video."""
        if self.normalization != "none":
            raise RuntimeError("sample_rms coefficients require stored scales to invert")
        if coefficients.ndim != 5:
            raise ValueError("coefficients must have shape [B,Ctf,Fp,H,W]")
        b, packed_channels, temporal_bins, height, total_width = coefficients.shape
        if total_width != self.output_size[1] or height != self.output_size[0]:
            raise ValueError("coefficient spatial shape does not match output_size")
        if total_width % self.num_views:
            raise ValueError("coefficient width must be divisible by num_views")
        if packed_channels % self.window_size:
            raise ValueError("coefficient channel count is not divisible by window_size")

        channels = packed_channels // self.window_size
        view_width = total_width // self.num_views
        views = coefficients.reshape(
            b,
            packed_channels,
            temporal_bins,
            height,
            self.num_views,
            view_width,
        ).permute(0, 4, 2, 1, 3, 5)
        packed = views.reshape(
            b,
            self.num_views,
            temporal_bins,
            channels,
            self.window_size,
            height,
            view_width,
        ).to(dtype=self._compute_dtype(views))
        spectrum = self._unpack(packed).movedim(-1, 4)
        windows = torch.fft.irfft(
            spectrum, n=self.window_size, dim=4, norm="ortho"
        )  # [B,V,Fp,C,N,H,Wv]
        anchor = windows[:, :, 0, :, :1].permute(0, 1, 3, 2, 4, 5)
        tail = windows[:, :, 1:].permute(0, 1, 2, 4, 3, 5, 6).reshape(
            b,
            self.num_views,
            (temporal_bins - 1) * self.window_size,
            channels,
            height,
            view_width,
        )
        pooled = torch.cat([anchor, tail], dim=2)
        return pooled.permute(0, 2, 3, 4, 1, 5).reshape(
            b,
            1 + (temporal_bins - 1) * self.window_size,
            channels,
            height,
            total_width,
        ).to(dtype=coefficients.dtype)


class PerViewTemporalSTFT(nn.Module):
    """Create a Wan-grid-aligned TF state from a full-resolution frame sequence.

    Args:
        num_views: Number of contiguous camera views stacked along image width.
        output_size: Spatial ``(height, total_width)`` before the STFT.  The
            total width must be divisible by ``num_views``.
        target_frames: Number of temporal bins expected by the Wan latent.
        n_fft: Temporal FFT size.  ``n_fft=5`` produces three frequency bins.
        hop_length: Temporal STFT hop.
        win_length: Window length; defaults to ``n_fft``.
        normalization: Optional per-sample, per-view, per-channel RMS scaling.

    Input shape is ``[B, T, C, H, W_total]``.  Output shape is
    ``[B, C * 2 * (n_fft // 2 + 1), target_frames, H_out, W_out]``.
    """

    def __init__(
        self,
        *,
        num_views: int = 3,
        output_size: Tuple[int, int] = (24, 120),
        target_frames: int = 4,
        n_fft: int = 5,
        hop_length: int = 2,
        win_length: int | None = None,
        pad_multiple: int | None = 16,
        pad_value: float = -1.0,
        normalization: Literal["none", "sample_rms"] = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        win_length = n_fft if win_length is None else win_length
        if num_views < 1:
            raise ValueError("num_views must be positive")
        if target_frames < 1:
            raise ValueError("target_frames must be positive")
        if n_fft < 2 or not 1 <= win_length <= n_fft:
            raise ValueError("require n_fft >= 2 and 1 <= win_length <= n_fft")
        if hop_length < 1:
            raise ValueError("hop_length must be positive")
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError("output_size dimensions must be positive")
        if output_size[1] % num_views:
            raise ValueError("output width must be divisible by num_views")
        if pad_multiple is not None and pad_multiple < 1:
            raise ValueError("pad_multiple must be positive or None")
        if normalization not in {"none", "sample_rms"}:
            raise ValueError(f"unsupported normalization: {normalization}")

        self.num_views = num_views
        self.output_size = output_size
        self.target_frames = target_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_multiple = pad_multiple
        self.pad_value = pad_value
        self.normalization = normalization
        self.eps = eps
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=True), persistent=False
        )

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    def output_channels(self, input_channels: int) -> int:
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        return input_channels * self.frequency_bins * 2

    def _align_temporal_bins(self, x: Tensor) -> Tensor:
        # x: [B, V, Ctf, S, H, Wv]
        if x.shape[3] == self.target_frames:
            return x
        b, v, c, steps, h, w = x.shape
        flat = x.permute(0, 1, 2, 4, 5, 3).reshape(-1, 1, steps)
        if steps >= self.target_frames:
            flat = F.adaptive_avg_pool1d(flat, self.target_frames)
        else:
            flat = F.interpolate(
                flat, size=self.target_frames, mode="linear", align_corners=False
            )
        return flat.reshape(b, v, c, h, w, self.target_frames).permute(
            0, 1, 2, 5, 3, 4
        )

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"expected video [B,T,C,H,W], received shape {tuple(video.shape)}"
            )
        if not video.is_floating_point():
            raise TypeError("video must be floating point")

        b, frames, channels, height, total_width = video.shape
        if frames < self.win_length:
            raise ValueError(
                f"video has {frames} frames, shorter than win_length={self.win_length}"
            )
        if total_width % self.num_views:
            raise ValueError("input width must be divisible by num_views")

        view_width = total_width // self.num_views
        out_height, out_total_width = self.output_size
        out_view_width = out_total_width // self.num_views

        # Width chunks are independent views.  Pooling happens after the split,
        # so no spatial kernel can cross a camera boundary.
        views = video.reshape(
            b, frames, channels, height, self.num_views, view_width
        ).permute(0, 4, 1, 2, 3, 5)
        flattened = views.reshape(
            b * self.num_views * frames, channels, height, view_width
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
        pooled = F.adaptive_avg_pool2d(
            flattened,
            (out_height, out_view_width),
        ).reshape(
            b,
            self.num_views,
            frames,
            channels,
            out_height,
            out_view_width,
        )

        # torch.stft does not support bf16 on CPU and is more stable in fp32.
        # Casting back after the deterministic transform preserves the model's
        # configured activation dtype.
        series = pooled.permute(0, 1, 3, 4, 5, 2).reshape(-1, frames)
        compute = series.float() if series.dtype in {torch.float16, torch.bfloat16} else series
        spectrum = torch.stft(
            compute,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=compute.device, dtype=compute.dtype),
            center=False,
            normalized=True,
            onesided=True,
            return_complex=True,
        )
        stft_steps = spectrum.shape[-1]
        coefficients = torch.view_as_real(spectrum).reshape(
            b,
            self.num_views,
            channels,
            out_height,
            out_view_width,
            self.frequency_bins,
            stft_steps,
            2,
        )
        coefficients = coefficients.permute(0, 1, 2, 5, 7, 6, 3, 4).reshape(
            b,
            self.num_views,
            self.output_channels(channels),
            stft_steps,
            out_height,
            out_view_width,
        )
        coefficients = self._align_temporal_bins(coefficients)

        if self.normalization == "sample_rms":
            rms = coefficients.square().mean(dim=(3, 4, 5), keepdim=True).sqrt()
            coefficients = coefficients / rms.clamp_min(self.eps)

        # Restore the production layout: views are contiguous width chunks.
        coefficients = coefficients.permute(0, 2, 3, 4, 1, 5).reshape(
            b,
            self.output_channels(channels),
            self.target_frames,
            out_height,
            out_total_width,
        )
        return coefficients.to(dtype=video.dtype)
