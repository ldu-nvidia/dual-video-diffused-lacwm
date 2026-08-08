"""Training-only VideoREPA-style token-relation distillation.

The teacher and student channel dimensions are intentionally unrelated.  Both
representations are pooled onto the same per-view space/time grid and only
their cosine-relation matrices are compared.  This avoids a trainable feature
projection and makes the auxiliary path disappear completely after training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TokenRelationLoss:
    """Per-sample spatial, temporal, and summed TRD losses."""

    spatial: Tensor
    temporal: Tensor
    total: Tensor


def _validate_grid(
    value: Tensor,
    *,
    name: str,
    num_views: int,
    pooled_height: int,
    pooled_width_per_view: int,
) -> None:
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B,C,F,H,W]")
    if not all(int(size) > 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if num_views < 1 or value.shape[-1] % num_views:
        raise ValueError(f"{name} width must divide evenly into num_views")
    if pooled_height < 1 or pooled_width_per_view < 1:
        raise ValueError("pooled spatial dimensions must be positive")


def pool_per_view_tokens(
    value: Tensor,
    *,
    num_views: int = 3,
    pooled_height: int = 2,
    pooled_width_per_view: int = 4,
) -> Tensor:
    """Pool ``[B,C,F,H,V*W]`` without ever averaging across view seams.

    Returns ``[B,V,F,S,C]`` where
    ``S = pooled_height * pooled_width_per_view``.  Temporal bins are retained
    exactly; only each view's spatial lattice is adaptively average pooled.
    Relation matrices therefore share a schema even when the Wan and V-JEPA
    encoders use different spatial strides.
    """

    _validate_grid(
        value,
        name="feature grid",
        num_views=num_views,
        pooled_height=pooled_height,
        pooled_width_per_view=pooled_width_per_view,
    )
    view_width = value.shape[-1] // num_views
    pooled_views = []
    for view_index in range(num_views):
        start = view_index * view_width
        stop = start + view_width
        view = value[..., start:stop]
        pooled_views.append(
            F.adaptive_avg_pool3d(
                view.float(),
                output_size=(
                    int(value.shape[2]),
                    pooled_height,
                    pooled_width_per_view,
                ),
            )
        )
    # [B,V,C,F,H,W] -> [B,V,F,H*W,C]
    pooled = torch.stack(pooled_views, dim=1)
    return pooled.permute(0, 1, 3, 4, 5, 2).flatten(3, 4).contiguous()


def wan_tokens_to_grid(
    tokens: Tensor,
    *,
    target_grid_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Reshape Wan's flattened tokens to ``[B,D,F,H,W]``.

    ``target_grid_shape`` is the cached clean target lattice before Wan patch
    stride.  Binding the inferred lattice to that target catches token-order or
    patch-size drift before relational supervision can silently misalign it.
    """

    if tokens.ndim != 3:
        raise ValueError("Wan hidden tokens must have shape [B,N,D]")
    if len(target_grid_shape) != 3 or len(patch_size) != 3:
        raise ValueError("target grid and patch size must be three-dimensional")
    if any(int(value) < 1 for value in (*target_grid_shape, *patch_size)):
        raise ValueError("target grid and patch size must be positive")
    if any(size % stride for size, stride in zip(target_grid_shape, patch_size)):
        raise ValueError("clean target grid is not divisible by Wan patch size")
    grid = tuple(
        int(size) // int(stride)
        for size, stride in zip(target_grid_shape, patch_size)
    )
    expected_tokens = grid[0] * grid[1] * grid[2]
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            "Wan token count does not match target/patch grid: "
            f"{tokens.shape[1]} != {expected_tokens}"
        )
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], *grid)


def _thresholded_l1(left: Tensor, right: Tensor, margin: float) -> Tensor:
    if not 0.0 <= float(margin) < 2.0:
        raise ValueError("TRD margin must lie in [0,2)")
    difference = (left - right).abs()
    # Match the released VideoREPA hinge: the margin is a tolerance that is
    # subtracted from every supra-margin error, not merely a binary mask.
    return difference.sub(float(margin)).clamp_min(0.0)


def token_relation_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    margin: float = 0.05,
) -> TokenRelationLoss:
    """Compare within-frame and cross-frame cosine relations.

    Inputs must both be ``[B,V,F,S,C]`` but may use different channel counts.
    Spatial relations compare all token pairs inside each frame and view.
    Temporal relations compare every spatial pair across all ordered, distinct
    frame pairs within a view.  The two mean L1 components receive equal weight,
    matching the additive form in VideoREPA rather than directly matching
    teacher and student feature vectors.
    """

    if student.ndim != 5 or teacher.ndim != 5:
        raise ValueError("TRD inputs must have shape [B,V,F,S,C]")
    if student.shape[:4] != teacher.shape[:4]:
        raise ValueError(
            "student/teacher token lattices must match before relation loss"
        )
    if student.shape[2] < 2:
        raise ValueError("temporal TRD requires at least two temporal bins")
    if student.shape[3] < 1 or student.shape[-1] < 1 or teacher.shape[-1] < 1:
        raise ValueError("TRD token and channel dimensions must be nonempty")
    if not bool(torch.isfinite(student).all()) or not bool(
        torch.isfinite(teacher).all()
    ):
        raise FloatingPointError("TRD inputs contain non-finite values")

    student_unit = F.normalize(student.float(), dim=-1, eps=1e-8)
    teacher_unit = F.normalize(teacher.detach().float(), dim=-1, eps=1e-8)

    student_spatial = torch.einsum(
        "bvfsc,bvftc->bvfst", student_unit, student_unit
    )
    teacher_spatial = torch.einsum(
        "bvfsc,bvftc->bvfst", teacher_unit, teacher_unit
    )
    spatial = _thresholded_l1(
        student_spatial, teacher_spatial, margin
    ).mean(dim=(1, 2, 3, 4))

    student_temporal = torch.einsum(
        "bvfsc,bvgtc->bvfgst", student_unit, student_unit
    )
    teacher_temporal = torch.einsum(
        "bvfsc,bvgtc->bvfgst", teacher_unit, teacher_unit
    )
    frame_count = int(student.shape[2])
    off_diagonal = ~torch.eye(
        frame_count, device=student.device, dtype=torch.bool
    )
    temporal_difference = _thresholded_l1(
        student_temporal, teacher_temporal, margin
    )
    temporal = temporal_difference[:, :, off_diagonal].mean(dim=(1, 2, 3, 4))

    total = spatial + temporal
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("TRD loss is non-finite")
    return TokenRelationLoss(spatial=spatial, temporal=temporal, total=total)
