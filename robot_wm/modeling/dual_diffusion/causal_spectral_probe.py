"""Frozen Phase-0 causal spectral inverse-dynamics probe (CSIP).

CSIP is deliberately *not* a video generator.  It asks the cheaper question
that must be answered before another dual-diffusion arm is justified: do clean
Wan future-motion latents contain action-aligned, phase-sensitive spectral
information that a small fixed-capacity probe can recover?

The representation is computed independently for each of the three camera
views.  Given the independently encoded observed history ``h`` and the full
clip latent ``z``, the two clean future increments are

``d0 = z[:, :, 2] - h[:, :, 1]`` and
``d1 = z[:, :, 3] - z[:, :, 2]``.

For every view and Wan channel we retain a centered low-frequency crop from a
three-dimensional FFT of ``[d0,d1]``.  The channels are log magnitude and an
energy-masked unit phasor.  We additionally retain the energy-masked unit
phase increment ``U(d1) conj(U(d0))`` from per-token spatial FFTs.  No FFT is
ever taken across a width-stacked camera seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


WAN_CHANNELS = 16
HISTORY_TOKENS = 2
FULL_TOKENS = 4
FUTURE_TOKENS = 2
LATENT_HEIGHT = 24
LATENT_WIDTH = 120
NUM_VIEWS = 3
VIEW_WIDTH = LATENT_WIDTH // NUM_VIEWS
SPECTRAL_HEIGHT = 4
SPECTRAL_WIDTH = 6
ACTION_CHUNKS = 13
ACTION_STEPS = 5
ACTION_DIM = 23
FUTURE_ACTION_START = 4
FUTURE_ACTION_STOP = 12
ACTION_DESCRIPTOR_DIM = (
    (FUTURE_ACTION_STOP - FUTURE_ACTION_START) * (ACTION_STEPS - 1) * ACTION_DIM
)
ACTION_TARGET_DIM = 16
ENERGY_RELATIVE_FLOOR = 1e-3

# 3-D log magnitude + real/imaginary unit phase, followed by real/imaginary
# temporal phase increment from the two spatial spectra.
SPECTRAL_FEATURE_DIM = (
    NUM_VIEWS
    * WAN_CHANNELS
    * (
        3 * FUTURE_TOKENS * SPECTRAL_HEIGHT * SPECTRAL_WIDTH
        + 2 * (FUTURE_TOKENS - 1) * SPECTRAL_HEIGHT * SPECTRAL_WIDTH
    )
)
# The flattened representation is ordered as volume log magnitude, volume
# unit-phase real/imaginary, and phase-increment real/imaginary.  The matched
# angle-neutral comparator deliberately keeps the full 9,216-input model
# geometry and every reliability-support mask while replacing each valid unit
# phasor by the fixed neutral direction ``(1, 0)``.
VOLUME_MAGNITUDE_FEATURE_DIM = (
    NUM_VIEWS * WAN_CHANNELS * FUTURE_TOKENS * SPECTRAL_HEIGHT * SPECTRAL_WIDTH
)
VOLUME_PHASE_COMPONENT_DIM = VOLUME_MAGNITUDE_FEATURE_DIM
PHASE_INCREMENT_COMPONENT_DIM = (
    NUM_VIEWS
    * WAN_CHANNELS
    * (FUTURE_TOKENS - 1)
    * SPECTRAL_HEIGHT
    * SPECTRAL_WIDTH
)


class CSIPContractError(RuntimeError):
    """A frozen Phase-0 geometry, split, or control contract was violated."""


def phase0_partition_indexes(count: int, role: str) -> tuple[int, ...]:
    """Return the preregistered train-only fit448/calibration64 partition."""

    if isinstance(count, bool) or not isinstance(count, int) or count != 512:
        raise ValueError("CSIP Phase-0 requires exactly 512 training clips")
    if role == "fit":
        return tuple(index for index in range(count) if index % 8 != 0)
    if role == "calibration":
        return tuple(index for index in range(count) if index % 8 == 0)
    if role == "all":
        return tuple(range(count))
    raise ValueError("role must be one of: fit, calibration, all")


def _finite_float(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contains non-finite values")


def validate_latents(full_latents: Tensor, history_latents: Tensor) -> None:
    expected_full = (WAN_CHANNELS, FULL_TOKENS, LATENT_HEIGHT, LATENT_WIDTH)
    expected_history = (
        WAN_CHANNELS,
        HISTORY_TOKENS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    if full_latents.ndim != 5 or tuple(full_latents.shape[1:]) != expected_full:
        raise ValueError(
            f"full_latents must have shape [B,{expected_full}], got "
            f"{tuple(full_latents.shape)}"
        )
    if (
        history_latents.ndim != 5
        or tuple(history_latents.shape[1:]) != expected_history
        or history_latents.shape[0] != full_latents.shape[0]
    ):
        raise ValueError(
            f"history_latents must have shape [B,{expected_history}] with the "
            "same batch size"
        )
    _finite_float(full_latents, "full_latents")
    _finite_float(history_latents, "history_latents")


def future_motion_latents(full_latents: Tensor, history_latents: Tensor) -> Tensor:
    """Build two future increments without full-clip history leakage."""

    validate_latents(full_latents, history_latents)
    first = full_latents[:, :, 2] - history_latents[:, :, 1]
    second = full_latents[:, :, 3] - full_latents[:, :, 2]
    return torch.stack((first, second), dim=2)


def split_camera_views(value: Tensor) -> Tensor:
    """Split ``[B,C,T,H,120]`` into ``[B,3,C,T,H,40]`` before FFT."""

    if value.ndim != 5 or tuple(value.shape[1:]) != (
        WAN_CHANNELS,
        FUTURE_TOKENS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    ):
        raise ValueError("future motion latent geometry differs")
    return torch.stack(
        tuple(
            value[..., view * VIEW_WIDTH : (view + 1) * VIEW_WIDTH]
            for view in range(NUM_VIEWS)
        ),
        dim=1,
    )


def _center_crop_spectrum(value: Tensor) -> Tensor:
    """Crop centered spatial frequencies from ``[...,H,W]``."""

    if value.shape[-2:] != (LATENT_HEIGHT, VIEW_WIDTH):
        raise ValueError("spectral input must contain one 24x40 camera view")
    shifted = torch.fft.fftshift(value, dim=(-2, -1))
    h0 = (LATENT_HEIGHT - SPECTRAL_HEIGHT) // 2
    w0 = (VIEW_WIDTH - SPECTRAL_WIDTH) // 2
    return shifted[..., h0 : h0 + SPECTRAL_HEIGHT, w0 : w0 + SPECTRAL_WIDTH]


def _energy_mask(value: Tensor, *, relative_floor: float) -> tuple[Tensor, Tensor]:
    if not 0.0 < relative_floor < 1.0:
        raise ValueError("relative energy floor must lie strictly inside (0,1)")
    magnitude = value.abs()
    # Each sample/view/channel gets its own scale, so the mask is deterministic
    # and needs no validation-fitted threshold.  Clamp protects all-zero input.
    rms = (
        magnitude.square()
        .mean(dim=tuple(range(3, magnitude.ndim)), keepdim=True)
        .sqrt()
    )
    threshold = (rms * relative_floor).clamp_min(torch.finfo(magnitude.dtype).eps)
    return magnitude, magnitude >= threshold


def causal_spectral_features(
    full_latents: Tensor,
    history_latents: Tensor,
    *,
    relative_energy_floor: float = ENERGY_RELATIVE_FLOOR,
) -> Tensor:
    """Return the fixed ``[B,9216]`` CSIP spectral representation.

    FFTs run in float32 because CUDA half-complex support is shape-restricted
    and because unit phase is unstable under half precision near the mask.
    """

    motion = split_camera_views(future_motion_latents(full_latents, history_latents))
    motion = motion.float()

    # Spatiotemporal spectrum: [B,V,C,T,H,W].  Temporal frequencies are
    # shifted as well so the DC token is in a documented position.
    volume_fft = torch.fft.fftn(motion, dim=(-3, -2, -1), norm="ortho")
    volume_fft = torch.fft.fftshift(volume_fft, dim=(-3,))
    volume_fft = _center_crop_spectrum(volume_fft)
    volume_magnitude, volume_mask = _energy_mask(
        volume_fft, relative_floor=relative_energy_floor
    )
    safe_volume = volume_magnitude.clamp_min(torch.finfo(volume_magnitude.dtype).eps)
    volume_unit = volume_fft / safe_volume
    volume_mask_float = volume_mask.to(volume_unit.real.dtype)
    volume_parts = (
        torch.log1p(volume_magnitude),
        volume_unit.real * volume_mask_float,
        volume_unit.imag * volume_mask_float,
    )

    # Phase motion between the two future increments.  The product implements
    # exp(i(phi_1 - phi_0)) without an angle branch cut.
    spatial_fft = torch.fft.fftn(motion, dim=(-2, -1), norm="ortho")
    spatial_fft = _center_crop_spectrum(spatial_fft)
    spatial_magnitude, spatial_mask = _energy_mask(
        spatial_fft, relative_floor=relative_energy_floor
    )
    safe_spatial = spatial_magnitude.clamp_min(torch.finfo(spatial_magnitude.dtype).eps)
    spatial_unit = spatial_fft / safe_spatial
    increment = spatial_unit[:, :, :, 1:] * spatial_unit[:, :, :, :-1].conj()
    increment_mask = spatial_mask[:, :, :, 1:] & spatial_mask[:, :, :, :-1]
    increment_mask_float = increment_mask.to(increment.real.dtype)
    phase_increment_parts = (
        increment.real * increment_mask_float,
        increment.imag * increment_mask_float,
    )

    flat = torch.cat(
        tuple(
            part.reshape(part.shape[0], -1)
            for part in (*volume_parts, *phase_increment_parts)
        ),
        dim=1,
    )
    if tuple(flat.shape) != (full_latents.shape[0], SPECTRAL_FEATURE_DIM):
        raise AssertionError(
            f"internal CSIP feature geometry differs: {tuple(flat.shape)}"
        )
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("CSIP representation contains non-finite values")
    return flat


def angle_neutral_spectral_features(features: Tensor) -> Tensor:
    """Remove spectral angle while preserving magnitude and support.

    Each masked unit phasor ``(real, imag)`` becomes ``(mask, 0)``.  This keeps
    the exact 9,216-input geometry and the reliability/energy-support signal in
    both the 3-D volume phase and the per-token phase-increment channels.  A
    comparison with the full representation can therefore be attributed to
    angular information rather than to an omitted binary support mask.
    """

    if features.ndim != 2 or features.shape[1] != SPECTRAL_FEATURE_DIM:
        raise ValueError(f"features must have shape [B,{SPECTRAL_FEATURE_DIM}]")
    _finite_float(features, "features")
    result = features.clone()
    volume_real_start = VOLUME_MAGNITUDE_FEATURE_DIM
    volume_imag_start = volume_real_start + VOLUME_PHASE_COMPONENT_DIM
    increment_real_start = volume_imag_start + VOLUME_PHASE_COMPONENT_DIM
    increment_imag_start = increment_real_start + PHASE_INCREMENT_COMPONENT_DIM
    increment_imag_stop = increment_imag_start + PHASE_INCREMENT_COMPONENT_DIM
    if increment_imag_stop != SPECTRAL_FEATURE_DIM:
        raise AssertionError("internal CSIP phase-coordinate geometry differs")

    volume_real = features[:, volume_real_start:volume_imag_start]
    volume_imag = features[:, volume_imag_start:increment_real_start]
    volume_support = ((volume_real != 0) | (volume_imag != 0)).to(features.dtype)
    result[:, volume_real_start:volume_imag_start] = volume_support
    result[:, volume_imag_start:increment_real_start] = 0

    increment_real = features[:, increment_real_start:increment_imag_start]
    increment_imag = features[:, increment_imag_start:increment_imag_stop]
    increment_support = ((increment_real != 0) | (increment_imag != 0)).to(
        features.dtype
    )
    result[:, increment_real_start:increment_imag_start] = increment_support
    result[:, increment_imag_start:increment_imag_stop] = 0
    return result


def action_descriptor(actions: Tensor) -> Tensor:
    """Flatten future within-chunk action deltas to ``[B,736]``.

    Deltas avoid the nearly rank-one absolute-action shortcut observed in the
    earlier action audit.  Only chunks 4:12 are targets; no validation row is
    involved in fitting their PCA transform.
    """

    if actions.ndim != 4 or tuple(actions.shape[1:]) != (
        ACTION_CHUNKS,
        ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("actions must have shape [B,13,5,23]")
    _finite_float(actions, "actions")
    future = actions[:, FUTURE_ACTION_START:FUTURE_ACTION_STOP]
    descriptor = future[:, :, 1:] - future[:, :, :-1]
    descriptor = descriptor.reshape(actions.shape[0], -1).float()
    if descriptor.shape[1] != ACTION_DESCRIPTOR_DIM:
        raise AssertionError("internal action descriptor geometry differs")
    return descriptor


def _canonicalize_component_sign(components: Tensor) -> Tensor:
    components = components.clone()
    pivots = components.abs().argmax(dim=1)
    signs = torch.sign(components[torch.arange(components.shape[0]), pivots])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return components * signs[:, None]


@dataclass(frozen=True)
class ActionPCATransform:
    mean: Tensor
    components: Tensor
    score_scale: Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != (ACTION_DESCRIPTOR_DIM,):
            raise ValueError("action PCA mean geometry differs")
        if self.components.shape != (ACTION_TARGET_DIM, ACTION_DESCRIPTOR_DIM):
            raise ValueError("action PCA component geometry differs")
        if self.score_scale.shape != (ACTION_TARGET_DIM,):
            raise ValueError("action PCA scale geometry differs")
        if bool((self.score_scale <= 1e-8).any()):
            raise ValueError("action PCA contains a degenerate component")

    def transform_descriptor(self, descriptor: Tensor) -> Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != ACTION_DESCRIPTOR_DIM:
            raise ValueError("action descriptor geometry differs")
        mean = self.mean.to(device=descriptor.device, dtype=descriptor.dtype)
        components = self.components.to(
            device=descriptor.device, dtype=descriptor.dtype
        )
        scale = self.score_scale.to(device=descriptor.device, dtype=descriptor.dtype)
        return ((descriptor - mean) @ components.T) / scale

    def transform_actions(self, actions: Tensor) -> Tensor:
        return self.transform_descriptor(action_descriptor(actions))

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "score_scale": self.score_scale.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "ActionPCATransform":
        return cls(
            mean=state["mean"].detach().float().cpu(),
            components=state["components"].detach().float().cpu(),
            score_scale=state["score_scale"].detach().float().cpu(),
        )


def fit_action_pca(descriptor: Tensor) -> ActionPCATransform:
    """Fit and whiten the fixed 16-D target on fit448 rows only."""

    if descriptor.ndim != 2 or tuple(descriptor.shape) != (448, ACTION_DESCRIPTOR_DIM):
        raise ValueError("action PCA must be fit once on fit448 descriptors")
    descriptor64 = descriptor.detach().double().cpu()
    if not bool(torch.isfinite(descriptor64).all()):
        raise FloatingPointError("action descriptor contains non-finite values")
    mean = descriptor64.mean(dim=0)
    centered = descriptor64 - mean
    _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    if (
        singular.shape[0] < ACTION_TARGET_DIM
        or singular[ACTION_TARGET_DIM - 1] <= 1e-10
    ):
        raise CSIPContractError("fit448 action target has insufficient rank")
    components = _canonicalize_component_sign(vh[:ACTION_TARGET_DIM])
    raw_scores = centered @ components.T
    scale = raw_scores.std(dim=0, correction=0)
    return ActionPCATransform(
        mean=mean.float(), components=components.float(), score_scale=scale.float()
    )


def control_targets(
    actions: Tensor,
    transform: ActionPCATransform,
    donor_indexes: Tensor,
) -> dict[str, Tensor]:
    """Build aligned, paired-shuffled, raw-no-action, and inverse targets."""

    descriptor = action_descriptor(actions)
    if donor_indexes.shape != (actions.shape[0],) or donor_indexes.dtype != torch.long:
        raise ValueError("donor_indexes must be one int64 index per clip")
    expected = torch.arange(actions.shape[0], device=donor_indexes.device)
    if bool((donor_indexes == expected).any()):
        raise CSIPContractError("shuffled control contains a self donor")
    if int(donor_indexes.min()) < 0 or int(donor_indexes.max()) >= actions.shape[0]:
        raise ValueError("donor index is outside the evaluation batch")
    donor = donor_indexes.to(descriptor.device)
    return {
        "aligned": transform.transform_descriptor(descriptor),
        "episode_disjoint_paired_shuffled": transform.transform_descriptor(
            descriptor[donor]
        ),
        "raw_no_action": transform.transform_descriptor(torch.zeros_like(descriptor)),
        "inverse": transform.transform_descriptor(-descriptor),
    }


def episode_disjoint_pair_donors(
    episode_ids: Sequence[str],
) -> tuple[Tensor, tuple[tuple[int, int], ...]]:
    """Pair adjacent unique episodes and swap the target inside every pair.

    The fixed manifest order determines the pairing: ``(0,1), (2,3), ...``.
    Returning disjoint symmetric pairs makes the episode pair, rather than an
    overlapping cyclic edge, a valid independent bootstrap unit.
    """

    if len(episode_ids) < 2 or len(episode_ids) % 2 or any(
        not isinstance(value, str) or not value for value in episode_ids
    ):
        raise ValueError(
            "an even number of at least two nonempty episode IDs is required"
        )
    count = len(episode_ids)
    if len(set(episode_ids)) != count:
        raise CSIPContractError("paired donor assignment requires one clip per episode")
    donors = torch.empty(count, dtype=torch.long)
    pairs: list[tuple[int, int]] = []
    for left in range(0, count, 2):
        right = left + 1
        if episode_ids[left] == episode_ids[right]:
            raise CSIPContractError(
                "paired donor assignment contains one episode twice"
            )
        donors[left] = right
        donors[right] = left
        pairs.append((left, right))
    return donors, tuple(pairs)


class FrozenCausalSpectralProbe(nn.Module):
    """Fixed-capacity MLP mapping immutable CSIP features to action PCA scores."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim != 256
        ):
            raise ValueError("Phase-0 probe hidden_dim is frozen at 256")
        self.feature_norm = nn.LayerNorm(SPECTRAL_FEATURE_DIM)
        self.network = nn.Sequential(
            nn.Linear(SPECTRAL_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ACTION_TARGET_DIM),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != SPECTRAL_FEATURE_DIM:
            raise ValueError(f"features must have shape [B,{SPECTRAL_FEATURE_DIM}]")
        _finite_float(features, "features")
        return self.network(self.feature_norm(features.float()))
