"""Frozen Stage-0 inverse-action critic for training-only action-cycle loss.

The Stage-0 probe fits nine train-only ridges (three causal Wan-latent
transitions by three camera views).  Stage 1 uses the identical deterministic
feature map and frozen normalization to score a predicted clean Wan latent.
This object is intentionally *not* an ``nn.Module``: it contributes no
parameter or persistent buffer to the world-model checkpoint and is never
constructed by the deployment configuration.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


CAMERAS = ("top", "left_wrist", "right_wrist")
LATENT_SHAPE = (16, 4, 24, 120)
POOL_SHAPE = (6, 10)
TRANSITIONS = (0, 1, 2)
FUTURE_RELEVANT_TRANSITIONS = (1, 2)
ACTION_INTERVALS = ((0, 4), (4, 8), (8, 12))
ACTION_SHAPE = (13, 5, 23)
FEATURE_DIM = 16 * POOL_SHAPE[0] * POOL_SHAPE[1]
TARGET_DIM = 4 * ACTION_SHAPE[1] * ACTION_SHAPE[2]
STD_FLOOR = 1.0e-6


class ActionCycleError(RuntimeError):
    """A frozen critic, action target, or latent geometry changed."""


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latent_displacement_features(latents: Tensor) -> Tensor:
    """Reproduce the frozen Stage-0 feature, returning ``[B,3,3,960]``.

    Wan encodes the three-view panorama jointly.  Width is split only after
    encoding; each endpoint is layer-normalized independently over channel,
    height, and per-view width, pooled to 6x10, and adjacent endpoints are
    subtracted.  No operation pools across a camera seam.
    """

    if latents.ndim != 5 or tuple(int(v) for v in latents.shape[1:]) != LATENT_SHAPE:
        raise ActionCycleError(
            f"Wan latent must have shape [B,{','.join(map(str, LATENT_SHAPE))}]"
        )
    batch, channels, bins, height, width = latents.shape
    views = len(CAMERAS)
    if width % views:
        raise ActionCycleError("Wan latent width is not divisible by camera count")
    per_view_width = width // views
    view = latents.float().reshape(
        batch, channels, bins, height, views, per_view_width
    ).permute(0, 2, 4, 1, 3, 5)
    mean = view.mean(dim=(3, 4, 5), keepdim=True)
    variance = (view - mean).square().mean(dim=(3, 4, 5), keepdim=True)
    normalized = (view - mean) * torch.rsqrt(variance + 1.0e-6)
    pooled = F.adaptive_avg_pool2d(
        normalized.reshape(batch * bins * views, channels, height, per_view_width),
        POOL_SHAPE,
    ).reshape(batch, bins, views, channels, *POOL_SHAPE)
    result = (pooled[:, 1:] - pooled[:, :-1]).flatten(3).contiguous()
    expected = (batch, len(TRANSITIONS), views, FEATURE_DIM)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise ActionCycleError("latent displacement feature contract failed")
    return result


def aligned_action_targets(actions: Tensor) -> Tensor:
    """Return Stage-0's aligned requested-action segments ``[B,3,460]``.

    Training actions are zero-padded to 157 dimensions.  The immutable ABC
    cache stores the 23-dimensional pre-padding vector (14 active requested
    joint/gripper coordinates plus nine constant camera coordinates), so only
    the first 23 coordinates may enter this critic target.
    """

    if actions.ndim != 4 or tuple(int(v) for v in actions.shape[1:3]) != (13, 5):
        raise ActionCycleError("actions must have shape [B,13,5,D]")
    if int(actions.shape[-1]) < ACTION_SHAPE[-1]:
        raise ActionCycleError("actions do not contain the frozen 23 coordinates")
    raw = actions[..., : ACTION_SHAPE[-1]].float()
    targets = torch.stack(
        [raw[:, start:stop].reshape(raw.shape[0], -1) for start, stop in ACTION_INTERVALS],
        dim=1,
    )
    if tuple(targets.shape) != (actions.shape[0], 3, TARGET_DIM):
        raise ActionCycleError("aligned action-target construction failed")
    if not bool(torch.isfinite(targets).all()):
        raise ActionCycleError("aligned action target is non-finite")
    return targets


def rf_predicted_clean(noisy: Tensor, sigma: Tensor, velocity: Tensor) -> Tensor:
    """Convert Wan RF velocity to clean estimate under sigma=1 noise/0 clean."""

    if noisy.ndim != 5 or noisy.shape != velocity.shape:
        raise ActionCycleError("noisy state and velocity must share [B,C,T,H,W]")
    if sigma.ndim != 1 or sigma.shape[0] != noisy.shape[0]:
        raise ActionCycleError("sigma must have shape [B]")
    expanded = sigma.float().to(noisy.device).reshape(-1, 1, 1, 1, 1)
    estimate = noisy.float() - expanded * velocity.float()
    if not bool(torch.isfinite(estimate).all()):
        raise ActionCycleError("predicted clean latent is non-finite")
    return estimate


class FrozenStage0RidgeCritic:
    """Plain-object frozen ridge and normalization, absent from state dicts."""

    REQUIRED_ARRAYS = {
        "feature_mean": ((3, 3, FEATURE_DIM), np.float32),
        "feature_std": ((3, 3, FEATURE_DIM), np.float32),
        "feature_active": ((3, 3, FEATURE_DIM), np.bool_),
        "target_mean": ((3, TARGET_DIM), np.float32),
        "target_std": ((3, TARGET_DIM), np.float32),
        "target_active": ((3, TARGET_DIM), np.bool_),
        "aligned_weight": ((3, 3, FEATURE_DIM, TARGET_DIM), np.float32),
    }

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        expected_sha256: str,
        expected_stage0_registration_identity: str,
    ) -> None:
        path = Path(bundle_path).expanduser()
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ActionCycleError("critic bundle must be an absolute regular file")
        path = path.resolve(strict=True)
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256 or len(expected_sha256) != 64:
            raise ActionCycleError("critic bundle digest differs from registration")
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ActionCycleError("critic bundle cannot be opened") from exc
        expected_names = set(self.REQUIRED_ARRAYS) | {
            "stage0_registration_identity_sha256",
            "selected_alpha",
            "source_frozen_ridge_sha256",
        }
        if set(archive.files) != expected_names:
            raise ActionCycleError("critic bundle field inventory differs")
        registration_value = archive["stage0_registration_identity_sha256"]
        if (
            registration_value.shape != ()
            or str(registration_value.item()) != expected_stage0_registration_identity
            or len(expected_stage0_registration_identity) != 64
        ):
            raise ActionCycleError("Stage-0 registration identity differs")
        alpha = archive["selected_alpha"]
        if alpha.shape != () or not math.isfinite(float(alpha.item())) or float(alpha.item()) <= 0:
            raise ActionCycleError("selected train-only ridge alpha is invalid")
        arrays: dict[str, Tensor] = {}
        for name, (shape, dtype) in self.REQUIRED_ARRAYS.items():
            value = archive[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ActionCycleError(f"critic array {name} shape/dtype differs")
            if value.dtype != np.bool_ and not np.isfinite(value).all():
                raise ActionCycleError(f"critic array {name} is non-finite")
            arrays[name] = torch.from_numpy(np.array(value, copy=True))
        archive.close()
        if bool((arrays["feature_std"] <= 0).any()) or bool(
            (arrays["target_std"] <= 0).any()
        ):
            raise ActionCycleError("critic standard deviations must be positive")
        if bool(
            (arrays["feature_std"][arrays["feature_active"]] <= STD_FLOOR).any()
        ) or bool(
            (arrays["target_std"][arrays["target_active"]] <= STD_FLOOR).any()
        ):
            raise ActionCycleError("active critic coordinates violate the std floor")
        if bool((~arrays["target_active"]).all(dim=1).any()):
            raise ActionCycleError("a critic transition has no active action coordinate")
        self.path = path
        self.sha256 = observed_sha256
        self.stage0_registration_identity_sha256 = expected_stage0_registration_identity
        self.selected_alpha = float(alpha.item())
        self._cpu = arrays
        self._device: dict[tuple[str, int | None], dict[str, Tensor]] = {}

    def state_dict(self) -> None:
        """Make accidental persistence fail rather than silently serialize."""

        raise ActionCycleError("training-only critic has no model state_dict")

    def _on(self, device: torch.device) -> Mapping[str, Tensor]:
        key = (device.type, device.index)
        if key not in self._device:
            self._device[key] = {
                name: value.to(device=device, non_blocking=False)
                for name, value in self._cpu.items()
            }
        return self._device[key]

    def predict_and_loss(
        self,
        predicted_clean: Tensor,
        actions: Tensor,
        *,
        transitions: Sequence[int] = FUTURE_RELEVANT_TRANSITIONS,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        """Predict standardized actions and return per-sample aligned MSE."""

        selected = tuple(int(value) for value in transitions)
        if selected != FUTURE_RELEVANT_TRANSITIONS:
            raise ActionCycleError("Stage 1 is frozen to future-relevant transitions 1,2")
        arrays = self._on(predicted_clean.device)
        features = latent_displacement_features(predicted_clean)
        feature_normalized = torch.where(
            arrays["feature_active"].unsqueeze(0),
            (features - arrays["feature_mean"].unsqueeze(0))
            / arrays["feature_std"].unsqueeze(0),
            0.0,
        )
        by_view = torch.einsum(
            "btvf,tvfa->btva", feature_normalized, arrays["aligned_weight"]
        )
        prediction = by_view.mean(dim=2)
        raw_target = aligned_action_targets(actions)
        target = torch.where(
            arrays["target_active"].unsqueeze(0),
            (raw_target - arrays["target_mean"].unsqueeze(0))
            / arrays["target_std"].unsqueeze(0),
            0.0,
        )
        per_transition = []
        for transition in selected:
            active = arrays["target_active"][transition]
            per_transition.append(
                (prediction[:, transition, active] - target[:, transition, active])
                .square()
                .mean(dim=1)
            )
        per_sample = torch.stack(per_transition, dim=1).mean(dim=1)
        pred_selected = torch.cat(
            [prediction[:, value, arrays["target_active"][value]] for value in selected],
            dim=1,
        )
        target_selected = torch.cat(
            [target[:, value, arrays["target_active"][value]] for value in selected],
            dim=1,
        )
        cosine = F.cosine_similarity(pred_selected, target_selected, dim=1, eps=1.0e-8)
        if not bool(torch.isfinite(per_sample).all()) or not bool(
            torch.isfinite(cosine).all()
        ):
            raise ActionCycleError("inverse-action critic produced non-finite values")
        return per_sample, {
            "prediction": prediction,
            "target": target,
            "cosine": cosine,
            "feature_rms": features.square().mean(dim=(1, 2, 3)).sqrt(),
            "prediction_rms": pred_selected.square().mean(dim=1).sqrt(),
            "target_rms": target_selected.square().mean(dim=1).sqrt(),
        }

    def discard(self) -> None:
        self._device.clear()
        self._cpu.clear()


def critic_is_absent_from_model_state(model: Any) -> bool:
    """Return true only when no state key or module names the cycle critic."""

    forbidden = ("action_cycle_critic", "frozen_ridge", "inverse_action_critic")
    state_names = tuple(str(name).lower() for name in model.state_dict())
    module_names = tuple(str(name).lower() for name, _ in model.named_modules())
    return not any(
        token in name
        for name in (*state_names, *module_names)
        for token in forbidden
    )
