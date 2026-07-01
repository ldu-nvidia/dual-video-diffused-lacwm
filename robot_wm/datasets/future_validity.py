"""Model-aligned validity checks for Wan causal-VAE future supervision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import torch


@dataclass(frozen=True)
class FutureValidityConfig:
    enabled: bool = True
    history_frames: int = 5
    future_frames: int = 8
    temporal_ratio: int = 4
    num_views: int = 3
    view_std_threshold: float = 1e-3
    max_retries: int = 8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        if value is None:
            return cls(enabled=False)
        if not isinstance(value, Mapping):
            raise TypeError("future_validity must be a mapping or null")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown future_validity fields: {unknown}")
        config = cls(**dict(value))
        if not isinstance(config.enabled, bool):
            raise TypeError("future_validity.enabled must be a boolean")
        for field_name in (
            "history_frames",
            "future_frames",
            "temporal_ratio",
            "num_views",
            "max_retries",
        ):
            field_value = getattr(config, field_name)
            minimum = 0 if field_name == "max_retries" else 1
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, Integral)
                or field_value < minimum
            ):
                raise ValueError(
                    f"future_validity.{field_name} must be an integer >= {minimum}"
                )
        if (
            isinstance(config.view_std_threshold, bool)
            or not isinstance(config.view_std_threshold, Real)
            or config.view_std_threshold < 0
        ):
            raise ValueError(
                "future_validity.view_std_threshold must be a nonnegative number"
            )
        return config


@dataclass(frozen=True)
class FutureValidity:
    valid: bool
    reason: str
    expected_frames: int
    observed_frames: int | None
    valid_frame_count: int | None
    future_latent_groups: tuple[tuple[int, ...], ...]
    valid_future_latent_groups: tuple[int, ...]
    view_stds: tuple[float, ...]
    valid_views: tuple[int, ...]

    def diagnostic(self) -> dict[str, Any]:
        return asdict(self)


def _invalid(
    reason: str,
    config: FutureValidityConfig,
    *,
    observed_frames: int | None = None,
    valid_frame_count: int | None = None,
    future_groups=(),
    valid_future_groups=(),
    view_stds=(),
    valid_views=(),
) -> FutureValidity:
    return FutureValidity(
        valid=False,
        reason=reason,
        expected_frames=config.history_frames + config.future_frames,
        observed_frames=observed_frames,
        valid_frame_count=valid_frame_count,
        future_latent_groups=tuple(tuple(group) for group in future_groups),
        valid_future_latent_groups=tuple(valid_future_groups),
        view_stds=tuple(float(value) for value in view_stds),
        valid_views=tuple(valid_views),
    )


def _latent_index(frame_index: int, temporal_ratio: int) -> int:
    # Wan's causal VAE keeps frame zero alone, then groups each subsequent
    # temporal_ratio pixel frames into one latent frame.
    return 0 if frame_index == 0 else (frame_index - 1) // temporal_ratio + 1


def evaluate_future_validity(
    sample: Mapping[str, Any], config: FutureValidityConfig
) -> FutureValidity:
    """Return whether ``sample`` has at least one supervised future pixel.

    This mirrors ``LatentActionDiT._build_loss_mask``: a view is valid when its
    width-stacked pixels have standard deviation above the model threshold, and
    a causal-VAE latent time is valid only when every source pixel frame is
    valid. The flow loss has support exactly when both sets are nonempty.
    """
    expected_frames = config.history_frames + config.future_frames
    rgb = sample.get("rgb")
    if not isinstance(rgb, torch.Tensor):
        return _invalid("missing_or_non_tensor_rgb", config)
    if rgb.ndim != 4:
        return _invalid("rgb_must_have_shape_TCHW", config)
    observed_frames = int(rgb.shape[0])
    if observed_frames != expected_frames:
        return _invalid(
            "unexpected_clip_length",
            config,
            observed_frames=observed_frames,
        )
    if rgb.shape[-1] % config.num_views != 0:
        return _invalid(
            "rgb_width_not_divisible_by_num_views",
            config,
            observed_frames=observed_frames,
        )

    view_width = rgb.shape[-1] // config.num_views
    per_view = rgb.reshape(
        observed_frames,
        rgb.shape[1],
        rgb.shape[2],
        config.num_views,
        view_width,
    ).permute(3, 0, 1, 2, 4)
    view_stds_tensor = per_view.float().reshape(config.num_views, -1).std(dim=-1)
    view_stds = tuple(float(value.item()) for value in view_stds_tensor)
    valid_views = tuple(
        index
        for index, value in enumerate(view_stds)
        if value > config.view_std_threshold
    )

    mask = sample.get("mask")
    if mask is None:
        frame_valid = torch.ones(observed_frames, dtype=torch.bool)
    elif not isinstance(mask, torch.Tensor) or mask.ndim != 1:
        return _invalid(
            "mask_must_have_shape_T",
            config,
            observed_frames=observed_frames,
            view_stds=view_stds,
            valid_views=valid_views,
        )
    elif int(mask.shape[0]) != observed_frames:
        return _invalid(
            "mask_length_does_not_match_rgb",
            config,
            observed_frames=observed_frames,
            valid_frame_count=int(mask.to(dtype=torch.bool).sum().item()),
            view_stds=view_stds,
            valid_views=valid_views,
        )
    else:
        frame_valid = mask.to(dtype=torch.bool, device="cpu")

    first_future_latent = _latent_index(
        config.history_frames - 1, config.temporal_ratio
    ) + 1
    final_latent = _latent_index(observed_frames - 1, config.temporal_ratio)
    future_groups = []
    valid_future_groups = []
    for latent_index in range(first_future_latent, final_latent + 1):
        frames = tuple(
            frame
            for frame in range(observed_frames)
            if _latent_index(frame, config.temporal_ratio) == latent_index
        )
        future_groups.append(frames)
        if frames and bool(frame_valid[list(frames)].all().item()):
            valid_future_groups.append(latent_index)

    valid_frame_count = int(frame_valid.sum().item())
    if not valid_views:
        return _invalid(
            "no_nonconstant_view",
            config,
            observed_frames=observed_frames,
            valid_frame_count=valid_frame_count,
            future_groups=future_groups,
            valid_future_groups=valid_future_groups,
            view_stds=view_stds,
            valid_views=valid_views,
        )
    if not valid_future_groups:
        return _invalid(
            "no_complete_future_latent_group",
            config,
            observed_frames=observed_frames,
            valid_frame_count=valid_frame_count,
            future_groups=future_groups,
            valid_future_groups=valid_future_groups,
            view_stds=view_stds,
            valid_views=valid_views,
        )
    return FutureValidity(
        valid=True,
        reason="valid",
        expected_frames=expected_frames,
        observed_frames=observed_frames,
        valid_frame_count=valid_frame_count,
        future_latent_groups=tuple(future_groups),
        valid_future_latent_groups=tuple(valid_future_groups),
        view_stds=view_stds,
        valid_views=valid_views,
    )
