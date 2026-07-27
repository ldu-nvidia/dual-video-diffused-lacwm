"""Causal controls for injecting TF state into a video denoiser.

The auxiliary TF denoising branch always retains its own state.  These helpers
only choose the state exposed to the video branch.  For the shuffled control,
observed history and the local corruption noise stay fixed while hidden-future
clean content is deterministically deranged across the effective batch.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor


TFConditionMode = Literal["off", "matched", "shuffled"]
TF_CONDITION_MODES = ("off", "matched", "shuffled")


@torch.no_grad()
def roll_across_global_batch(state: Tensor) -> Tensor:
    """Return the next rank/sample's detached tensor without fixed points."""
    if state.ndim < 1 or state.shape[0] < 1:
        raise ValueError("state must have a nonempty batch dimension")
    local = state.detach().contiguous()
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size < 2:
            raise RuntimeError(
                "shuffled TF conditioning requires a global batch of at least two"
            )
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        source_rank = (dist.get_rank() + 1) % world_size
        return gathered[source_rank]
    if local.shape[0] < 2:
        raise RuntimeError(
            "shuffled TF conditioning requires a global batch of at least two"
        )
    return torch.roll(local, shifts=-1, dims=0)


def _validate_history(state: Tensor, history_frames: int) -> None:
    if state.ndim != 5:
        raise ValueError("TF state must have shape [B,C,F,H,W]")
    if not 0 <= history_frames <= state.shape[2]:
        raise ValueError("history_frames must lie within the TF temporal grid")


@torch.no_grad()
def make_training_conditioning_tf(
    *,
    mode: TFConditionMode,
    tf_clean: Tensor,
    tf_noise: Tensor,
    tf_noisy: Tensor,
    tf_sigma_expanded: Tensor,
    history_frames: int,
) -> Tensor:
    """Build matched or wrong-content TF with the same sigma and local noise."""
    if mode not in TF_CONDITION_MODES:
        raise ValueError(f"unsupported TF condition mode: {mode!r}")
    _validate_history(tf_clean, history_frames)
    if (
        tf_noise.shape != tf_clean.shape
        or tf_noisy.shape != tf_clean.shape
        or tf_sigma_expanded.ndim != tf_clean.ndim
        or tf_sigma_expanded.shape[0] != tf_clean.shape[0]
    ):
        raise ValueError("TF clean/noise/noisy/sigma tensors are not aligned")
    if mode != "shuffled":
        return tf_noisy

    wrong_clean = roll_across_global_batch(tf_clean)
    conditioning = (
        (1.0 - tf_sigma_expanded) * wrong_clean
        + tf_sigma_expanded * tf_noise
    )
    conditioning = conditioning.clone()
    conditioning[:, :, :history_frames] = tf_clean[
        :, :, :history_frames
    ]
    return conditioning


@torch.no_grad()
def make_sampling_conditioning_tf(
    *,
    mode: TFConditionMode,
    tf_state: Tensor,
    tf_noise: Tensor,
    tf_sigma_expanded: Tensor,
    history_frames: int,
) -> Tensor:
    """Choose the autonomous TF state visible to the video sampler.

    The shuffled control retains the local sample's initial corruption noise.
    Only the current noise-subtracted residual is deranged across the effective
    batch.  For an ideal flow state this residual is ``(1 - sigma) * x0_hat``;
    avoiding division by ``1 - sigma`` keeps the construction well-defined at
    the pure-noise endpoint and makes matched/shuffled exactly identical there.
    """
    if mode not in TF_CONDITION_MODES:
        raise ValueError(f"unsupported TF condition mode: {mode!r}")
    _validate_history(tf_state, history_frames)
    if (
        tf_noise.shape != tf_state.shape
        or tf_sigma_expanded.ndim != tf_state.ndim
        or tf_sigma_expanded.shape[0] != tf_state.shape[0]
    ):
        raise ValueError("TF state/noise/sigma tensors are not aligned")
    if mode != "shuffled":
        return tf_state

    conditioning = tf_state.clone()
    if history_frames == tf_state.shape[2]:
        return conditioning
    local_future_noise_component = (
        tf_sigma_expanded * tf_noise
    )[:, :, history_frames:]
    generated_future_clean_residual = (
        tf_state[:, :, history_frames:] - local_future_noise_component
    )
    conditioning[:, :, history_frames:] = (
        roll_across_global_batch(generated_future_clean_residual)
        + local_future_noise_component
    )
    return conditioning


@torch.no_grad()
def make_oracle_conditioning_tf(
    *,
    tf_clean: Tensor,
    tf_noise: Tensor,
    tf_sigma_expanded: Tensor,
    history_frames: int,
    wrong_tf_clean: Tensor | None = None,
) -> Tensor:
    """Construct a scheduled clean-TF oracle for diagnostic evaluation only.

    Passing ``wrong_tf_clean`` creates the marginally matched wrong-content
    oracle.  These tensors leak hidden future targets and are never valid
    deployable inputs; they diagnose whether the representation/injection path
    contains information the video model could use.
    """
    _validate_history(tf_clean, history_frames)
    if tf_noise.shape != tf_clean.shape:
        raise ValueError("TF clean and oracle noise shapes must match")
    if (
        tf_sigma_expanded.ndim != tf_clean.ndim
        or tf_sigma_expanded.shape[0] != tf_clean.shape[0]
    ):
        raise ValueError("oracle TF sigma is not aligned to the batch")
    source = tf_clean if wrong_tf_clean is None else wrong_tf_clean
    if source.shape != tf_clean.shape:
        raise ValueError("wrong oracle TF clean shape must match own clean TF")
    conditioning = (
        (1.0 - tf_sigma_expanded) * source
        + tf_sigma_expanded * tf_noise
    )
    conditioning = conditioning.clone()
    conditioning[:, :, :history_frames] = tf_clean[
        :, :, :history_frames
    ]
    return conditioning
