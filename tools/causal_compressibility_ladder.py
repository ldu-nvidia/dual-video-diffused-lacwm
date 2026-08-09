#!/usr/bin/env python3
"""Frozen-backbone causal-compressibility ladder for privileged video residuals.

The fit phase is the only phase allowed to open the train V-JEPA target array.
J1 and VPM development evaluation run in fresh processes and instantiate an
RGB/action-only dataset guarded against opening that array. LACWM's clock is
sigma=1 noise and sigma=0 clean data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "causal-compressibility-ladder-registration-v1"
FIT_SCHEMA = "causal-compressibility-ladder-fit-v1"
WEIGHT_SCHEMA = "causal-compressibility-ladder-weights-v1"
ROW_SCHEMA = "causal-compressibility-ladder-row-v1"
ENDPOINT_SCHEMA = "causal-compressibility-ladder-endpoint-v1"
ANALYSIS_SCHEMA = "causal-compressibility-ladder-analysis-v1"
COMPLETE_SCHEMA = "causal-compressibility-ladder-complete-v1"
AUDIT_SCHEMA = "causal-compressibility-ladder-audit-v1"

ARMS = ("ZERO", "DIRECT", "PFD_ALIGNED", "PFD_SHUFFLED")
J1_ENDPOINTS = ("J1_OFF", *ARMS)
ALL_ENDPOINTS = (*J1_ENDPOINTS, "VPM1")
ERROR_METRICS = (
    "corrected_velocity_mse",
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
QUALITY_METRICS = ERROR_METRICS[1:]

OPT_RANGE = (128, 384)
CAL_RANGE = (384, 416)
DEV_RANGE = (416, 480)
RESERVE_RANGE = (480, 512)
EXCLUDED_RANGE = (0, 128)
OPT_NOISE_SEEDS = (20260820, 20260821)
CAL_NOISE_SEEDS = (20260822, 20260823)
DEV_NOISE_SEEDS = (20260824, 20260825, 20260826, 20260827)
PROJECTION_SEED = 20260828
BOOTSTRAP_SEED = 20260829
CAPACITY_RUNGS = (64, 256, 1024)
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TOKEN_SAMPLE_COUNT = 256
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_TRAIN_COUNT = 512


class LadderError(RuntimeError):
    """Raised whenever a prospective or leakage boundary would be violated."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        return False
    try:
        int(recorded, 16)
    except ValueError:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == recorded


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LadderError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise LadderError(f"{label} must be a non-empty non-symlink file: {path}")
    return path.resolve(strict=True)


def canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LadderError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LadderError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def file_record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = canonical_file(path, "registered input")
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise LadderError(f"SHA-256 differs for {path}: {observed} != {expected_sha256}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def stat_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Bind a large immutable cache array through already-validated metadata."""
    path = canonical_file(path, "cache array")
    if len(expected_sha256) != 64:
        raise LadderError("cache metadata lacks a full SHA-256")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": expected_sha256,
        "digest_source": "complete immutable cache metadata",
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LadderError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LadderError(f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    raise LadderError(f"blank JSONL row {number}: {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LadderError(f"non-object JSONL row {number}: {path}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LadderError(f"invalid JSONL: {path}") from exc
    return rows


def exclusive_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    exclusive_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    exclusive_bytes(path, b"".join(_canonical_json(row) + b"\n" for row in rows))


def git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise LadderError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _resolve_metadata_file(metadata_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LadderError(f"cache metadata lacks {label}")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return canonical_file(path, label)


def _safe_tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_partition_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAIN_COUNT:
        raise LadderError(f"train manifest has {len(rows)} rows, expected 512")
    episodes: dict[str, int] = {}
    clips: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.get("split") != "train" or int(row.get("auxiliary_index", -1)) != index:
            raise LadderError(f"manifest row {index} is not the canonical train item")
        episode = row.get("episode_dir")
        clip = row.get("clip_id")
        if not isinstance(episode, str) or not episode or not isinstance(clip, str) or not clip:
            raise LadderError(f"manifest row {index} lacks episode/clip identity")
        if episode in episodes:
            raise LadderError(
                f"episodes are not disjoint: rows {episodes[episode]} and {index}"
            )
        if clip in clips:
            raise LadderError(f"clips are not disjoint: rows {clips[clip]} and {index}")
        episodes[episode] = index
        clips[clip] = index
    seed_sets = (set(OPT_NOISE_SEEDS), set(CAL_NOISE_SEEDS), set(DEV_NOISE_SEEDS))
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise LadderError("optimization/calibration/development noise seeds overlap")
    partitions = {
        "excluded": list(EXCLUDED_RANGE),
        "optimization": list(OPT_RANGE),
        "calibration": list(CAL_RANGE),
        "development": list(DEV_RANGE),
        "reserve": list(RESERVE_RANGE),
    }
    covered: list[int] = []
    for start, stop in partitions.values():
        covered.extend(range(start, stop))
    if sorted(covered) != list(range(EXPECTED_TRAIN_COUNT)) or len(set(covered)) != len(covered):
        raise LadderError("frozen manifest partitions do not form an exact partition")
    return {
        "all_512_episodes_unique": True,
        "all_512_clips_unique": True,
        "ranges": partitions,
        "noise_seed_sets_disjoint": True,
        "optimization_noise_seeds": list(OPT_NOISE_SEEDS),
        "calibration_noise_seeds": list(CAL_NOISE_SEEDS),
        "development_noise_seeds": list(DEV_NOISE_SEEDS),
    }


def patchify_video(value: Any, patch_size: Sequence[int]) -> Any:
    """Inverse of Wan's `[F,H,W,pf,ph,pw,C]` unpatchification."""
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise LadderError("video tensor must have shape [B,C,F,H,W]")
    pf, ph, pw = (int(part) for part in patch_size)
    batch, channels, frames, height, width = value.shape
    if frames % pf or height % ph or width % pw:
        raise LadderError("video shape is not divisible by Wan patch size")
    gf, gh, gw = frames // pf, height // ph, width // pw
    return (
        value.reshape(batch, channels, gf, pf, gh, ph, gw, pw)
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .contiguous()
        .reshape(batch, gf * gh * gw, pf * ph * pw * channels)
    )


def unpatchify_tokens(
    tokens: Any,
    *,
    grid: Sequence[int],
    patch_size: Sequence[int],
    channels: int,
) -> Any:
    import torch

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise LadderError("patch tokens must have shape [B,L,P]")
    gf, gh, gw = (int(part) for part in grid)
    pf, ph, pw = (int(part) for part in patch_size)
    expected = pf * ph * pw * int(channels)
    if tokens.shape[1] != gf * gh * gw or tokens.shape[2] != expected:
        raise LadderError("patch tokens do not match the registered Wan grid")
    return (
        tokens.reshape(tokens.shape[0], gf, gh, gw, pf, ph, pw, channels)
        .permute(0, 7, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .reshape(tokens.shape[0], channels, gf * pf, gh * ph, gw * pw)
    )


def future_token_positions(
    *, grid: Sequence[int], patch_size: Sequence[int], history_frames: int, device=None
) -> Any:
    import torch

    gf, gh, gw = (int(part) for part in grid)
    pf = int(patch_size[0])
    if history_frames % pf:
        raise LadderError("history latent frames are not patch aligned")
    first = (history_frames // pf) * gh * gw
    if not 0 < first < gf * gh * gw:
        raise LadderError("future token inventory is empty or consumes the full grid")
    return torch.arange(first, gf * gh * gw, device=device, dtype=torch.long)


def deterministic_token_subsample(
    positions: Any, *, clip_index: int, noise_seed: int, count: int
) -> Any:
    import torch

    if count <= 0 or count > positions.numel():
        raise LadderError("token subsample count is outside the future inventory")
    modulus = (1 << 63) - 1
    seed = (
        int(clip_index) * 0x9E3779B185EBCA87
        + int(noise_seed) * 0xC2B2AE3D27D4EB4F
        + 0x434F4D5052455353
    ) % modulus
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(positions.numel(), generator=generator)[:count]
    return positions.detach().cpu()[order].to(device=positions.device)


def make_projection(hidden_size: int, max_capacity: int, seed: int = PROJECTION_SEED) -> Any:
    import torch

    if not 0 < max_capacity <= hidden_size:
        raise LadderError("projection capacity must lie in (0, hidden_size]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(hidden_size, max_capacity, generator=generator, dtype=torch.float32)
    projection, _ = torch.linalg.qr(matrix, mode="reduced")
    return projection.contiguous()


def _new_sufficient_statistics(max_capacity: int, patch_dim: int, device: Any) -> dict[str, Any]:
    import torch

    return {
        "count": 0,
        "sum_x": torch.zeros(max_capacity, device=device, dtype=torch.float32),
        "sum_xx": torch.zeros(max_capacity, max_capacity, device=device, dtype=torch.float32),
        "sum_y": {
            arm: torch.zeros(patch_dim, device=device, dtype=torch.float32) for arm in ARMS
        },
        "sum_xy": {
            arm: torch.zeros(max_capacity, patch_dim, device=device, dtype=torch.float32)
            for arm in ARMS
        },
        "sum_y2": {arm: torch.zeros((), device=device, dtype=torch.float64) for arm in ARMS},
    }


def update_sufficient_statistics(stats: dict[str, Any], x: Any, targets: Mapping[str, Any]) -> None:
    import torch

    if x.ndim != 2 or set(targets) != set(ARMS):
        raise LadderError("sufficient-statistics batch schema differs")
    rows = x.shape[0]
    if rows <= 0 or any(value.shape[0] != rows or value.ndim != 2 for value in targets.values()):
        raise LadderError("sufficient-statistics target rows differ")
    x = x.float()
    stats["count"] += int(rows)
    stats["sum_x"].add_(x.sum(dim=0))
    stats["sum_xx"].add_(x.transpose(0, 1) @ x)
    for arm, target in targets.items():
        target = target.float()
        stats["sum_y"][arm].add_(target.sum(dim=0))
        stats["sum_xy"][arm].add_(x.transpose(0, 1) @ target)
        stats["sum_y2"][arm].add_(target.double().square().sum())


def fit_ridge_head(
    stats: Mapping[str, Any], *, arm: str, capacity: int, ridge_lambda: float
) -> dict[str, Any]:
    import torch

    if arm not in ARMS or capacity not in CAPACITY_RUNGS or ridge_lambda not in RIDGE_LAMBDAS:
        raise LadderError("ridge head request is outside the frozen ladder")
    count = int(stats["count"])
    if count <= capacity:
        raise LadderError("ridge fit has no overdetermined token inventory")
    sx = stats["sum_x"][:capacity].double().cpu()
    sxx = stats["sum_xx"][:capacity, :capacity].double().cpu()
    sy = stats["sum_y"][arm].double().cpu()
    sxy = stats["sum_xy"][arm][:capacity].double().cpu()
    mean_x = sx / count
    mean_y = sy / count
    covariance = (sxx - torch.outer(sx, sx) / count) / count
    cross = (sxy - torch.outer(sx, sy) / count) / count
    variance = covariance.diagonal().clamp_min(1e-10)
    scale = variance.sqrt()
    standardized_covariance = covariance / scale[:, None] / scale[None, :]
    standardized_cross = cross / scale[:, None]
    matrix = standardized_covariance + float(ridge_lambda) * torch.eye(
        capacity, dtype=torch.float64
    )
    weight = torch.linalg.solve(matrix, standardized_cross)
    if not torch.isfinite(weight).all():
        raise LadderError("ridge solution is non-finite")
    # Prediction is ((x - mean_x) / scale) @ weight + mean_y.
    result = {
        "arm": arm,
        "capacity": int(capacity),
        "ridge_lambda": float(ridge_lambda),
        "mean_x": mean_x.float().contiguous(),
        "scale_x": scale.float().contiguous(),
        "mean_y": mean_y.float().contiguous(),
        "weight": weight.float().contiguous(),
        "parameter_count": int((capacity + 1) * mean_y.numel()),
        "fit_token_count": count,
    }
    if arm == "ZERO":
        for key in ("mean_y", "weight"):
            if int(torch.count_nonzero(result[key])) != 0:
                raise LadderError("zero-target ridge solution is not exact zero")
    return result


def predict_ridge_head(projected: Any, head: Mapping[str, Any]) -> Any:
    import torch

    capacity = int(head["capacity"])
    mean_x = head["mean_x"].to(device=projected.device, dtype=torch.float32)
    scale_x = head["scale_x"].to(device=projected.device, dtype=torch.float32)
    mean_y = head["mean_y"].to(device=projected.device, dtype=torch.float32)
    weight = head["weight"].to(device=projected.device, dtype=torch.float32)
    value = projected[..., :capacity].float()
    prediction = ((value - mean_x) / scale_x) @ weight + mean_y
    if not torch.isfinite(prediction).all():
        raise LadderError("ridge prediction is non-finite")
    return prediction


class NumpyTargetOpenGuard(AbstractContextManager):
    """Record NumPy inputs and fail before a clean auxiliary array is opened."""

    def __init__(self, target_path: Path):
        self.target_path = target_path.resolve(strict=True)
        self.opened: list[Path] = []
        self._original = None

    def __enter__(self):
        import numpy as np

        self._original = np.load

        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)):
                path = Path(file).expanduser().resolve(strict=True)
                if path == self.target_path:
                    raise LadderError("development attempted to open the V-JEPA target array")
                self.opened.append(path)
            return self._original(file, *args, **kwargs)

        np.load = guarded
        return self

    def __exit__(self, exc_type, exc, traceback):
        import numpy as np

        if self._original is not None:
            np.load = self._original
        return False


def _capture_off_shared_tokens(model: Any, forward_call) -> tuple[Any, Any]:
    import torch

    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise LadderError("Wan head pre-hook did not receive shared trunk tokens")
        captured.append(inputs[0].detach())

    handle = model.forward_model.transformer.head.register_forward_pre_hook(capture)
    try:
        output = forward_call()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise LadderError(f"expected one off shared-token capture, observed {len(captured)}")
    return output, captured[0]


def _batch_samples(dataset: Any, indexes: Sequence[int], device: Any) -> dict[str, Any]:
    import torch
    from torch.utils.data import default_collate

    requested = [int(index) for index in indexes]
    access_counts = {index: 0 for index in requested}
    samples = []
    for index in requested:
        access_counts[index] += 1
        sample = dataset[index]
        observed = int(torch.as_tensor(sample.get("clip_index", -1)).item())
        if observed != index:
            raise LadderError(f"dataset substituted clip {observed} for {index}")
        samples.append(sample)
    if any(count != 1 for count in access_counts.values()):
        raise LadderError(f"dataset access-count invariant differs: {access_counts}")
    batch = default_collate(samples)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device=device, non_blocking=False)
    observed = [int(value) for value in batch["clip_index"].detach().cpu().tolist()]
    if observed != requested:
        raise LadderError(f"dataset substituted clips: {observed} != {requested}")
    return batch


def _model_inputs(
    model: Any,
    batch: Mapping[str, Any],
    *,
    noise_seed: int,
    nfe: int,
    need_clean_auxiliary: bool,
) -> dict[str, Any]:
    import torch

    rgb = batch["rgb"]
    video_clean = model._encode_clip(rgb).to(rgb.dtype)
    reference, history_frames = model._history_reference(rgb, video_clean.shape)
    if model._auxiliary_history_frames(history_frames) != 0:
        raise LadderError("probe requires diffuse-all auxiliary history")
    _, z_control, _ = model._latent_actions(
        rgb,
        batch["actions"],
        batch["morphology_index"],
        video_clean.shape[2],
        history_frames,
    )
    z_control = z_control.to(rgb.dtype)
    context = model._build_context(rgb.shape[0], rgb.device, rgb.dtype)
    clip_fea = model._build_clip(rgb.shape[0], rgb.device, rgb.dtype)
    initial_video = model._evaluation_noise(
        video_clean.shape,
        device=rgb.device,
        dtype=rgb.dtype,
        base_seed=int(noise_seed),
        sample_ids=batch["clip_index"],
        stream=0,
        rank=0,
    )
    auxiliary_shape = (
        rgb.shape[0],
        int(model.forward_model.tf_token_adapter.tf_channels),
        *video_clean.shape[2:],
    )
    initial_auxiliary = model._evaluation_noise(
        auxiliary_shape,
        device=rgb.device,
        dtype=rgb.dtype,
        base_seed=int(noise_seed),
        sample_ids=batch["clip_index"],
        stream=1,
        rank=0,
    )
    schedule, timesteps, tf_only_steps = model._sampling_schedule(nfe, device=rgb.device)
    active_step = int(tf_only_steps)
    if nfe == 2:
        if active_step != 1 or not math.isclose(float(schedule.video[active_step]), 1.0):
            raise LadderError("J1 NFE2 schedule lacks one sigma=1 video node")
    elif nfe == 1:
        if active_step != 0 or not math.isclose(float(schedule.video[active_step]), 1.0):
            raise LadderError("VPM NFE1 schedule lacks the sigma=1 video node")
    else:
        raise LadderError("only frozen J1-NFE2-node and VPM@1 endpoints are supported")
    next_video_sigma = float(schedule.video[active_step + 1])
    if not math.isclose(next_video_sigma, 0.0, rel_tol=0.0, abs_tol=1e-8):
        raise LadderError("one-step endpoint does not terminate at sigma=0")
    tf_sigma = schedule.time_frequency[active_step]
    tf_batch_sigma = tf_sigma.expand(rgb.shape[0]).to(device=rgb.device, dtype=rgb.dtype)
    tf_clean = None
    if need_clean_auxiliary:
        if "auxiliary_target" not in batch:
            raise LadderError("optimization batch lacks clean auxiliary target")
        tf_clean = model._validate_auxiliary_clean(
            batch["auxiliary_target"], video_clean.shape
        ).to(rgb.dtype)
    return {
        "rgb": rgb,
        "video_clean": video_clean,
        "reference": reference,
        "history_frames": int(history_frames),
        "z_control": z_control,
        "context": context,
        "clip_fea": clip_fea,
        "initial_video": initial_video,
        "initial_auxiliary": initial_auxiliary,
        "video_target": initial_video - video_clean,
        "timestep": timesteps[active_step],
        "tf_sigma": tf_sigma,
        "tf_batch_sigma": tf_batch_sigma,
        "tf_clean": tf_clean,
    }


def _off_prediction_and_hidden(model: Any, prepared: Mapping[str, Any]) -> tuple[Any, Any]:
    from robot_wm.modeling.networks.wan_forward_model import DualWanOutput

    batch_size = prepared["initial_video"].shape[0]
    timestep = prepared["timestep"].expand(batch_size).to(prepared["initial_video"].device)

    def call():
        return model.forward_model(
            prepared["initial_video"],
            timestep,
            prepared["z_control"],
            prepared["reference"],
            prepared["context"],
            prepared["clip_fea"],
            noisy_tf=prepared["initial_auxiliary"],
            conditioning_tf=prepared["initial_auxiliary"],
            tf_sigma=prepared["tf_batch_sigma"],
            condition_on_tf=False,
            condition_on_tf_clock=False,
        )

    output, hidden = _capture_off_shared_tokens(model, call)
    if not isinstance(output, DualWanOutput):
        raise LadderError("off J1/VPM call did not return dual velocities")
    return output.video_velocity, hidden


def _teacher_predictions(model: Any, prepared: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch
    from robot_wm.modeling.dual_diffusion.conditioning import make_oracle_conditioning_tf
    from robot_wm.modeling.networks.wan_forward_model import DualWanOutput

    tf_clean = prepared["tf_clean"]
    if tf_clean is None or tf_clean.shape[0] < 2:
        raise LadderError("teacher construction requires paired aligned features")
    wrong = torch.roll(tf_clean, shifts=-1, dims=0)
    if any(torch.equal(tf_clean[index], wrong[index]) for index in range(tf_clean.shape[0])):
        raise LadderError("aligned and shuffled teacher features are identical")
    sigma_expanded = model._expand_sigma(
        prepared["tf_batch_sigma"], prepared["initial_auxiliary"]
    )
    aligned_condition = make_oracle_conditioning_tf(
        tf_clean=tf_clean,
        tf_noise=prepared["initial_auxiliary"],
        tf_sigma_expanded=sigma_expanded,
        history_frames=0,
    )
    shuffled_condition = make_oracle_conditioning_tf(
        tf_clean=tf_clean,
        tf_noise=prepared["initial_auxiliary"],
        tf_sigma_expanded=sigma_expanded,
        history_frames=0,
        wrong_tf_clean=wrong,
    )
    batch_size = prepared["initial_video"].shape[0]
    timestep = prepared["timestep"].expand(batch_size).to(prepared["initial_video"].device)
    outputs = []
    for condition in (aligned_condition, shuffled_condition):
        output = model.forward_model(
            prepared["initial_video"],
            timestep,
            prepared["z_control"],
            prepared["reference"],
            prepared["context"],
            prepared["clip_fea"],
            noisy_tf=prepared["initial_auxiliary"],
            conditioning_tf=condition,
            tf_sigma=prepared["tf_batch_sigma"],
            condition_on_tf=True,
            condition_on_tf_clock=True,
        )
        if not isinstance(output, DualWanOutput):
            raise LadderError("teacher call did not return dual velocities")
        outputs.append(output.video_velocity)
    return outputs[0], outputs[1]


def _grid_contract(model: Any, tensor: Any, hidden: Any) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    patch_size = tuple(int(value) for value in model.forward_model.transformer.patch_size)
    grid = tuple(int(tensor.shape[index + 2] // patch_size[index]) for index in range(3))
    patch_dim = int(tensor.shape[1] * math.prod(patch_size))
    if hidden.ndim != 3 or hidden.shape[1] != math.prod(grid):
        raise LadderError("captured shared-token order does not match the Wan output grid")
    if patchify_video(tensor, patch_size).shape != (tensor.shape[0], math.prod(grid), patch_dim):
        raise LadderError("Wan patch contract changed")
    return grid, patch_size, patch_dim


def _load_model(config_path: Path, snapshot_path: Path, device: Any) -> tuple[Any, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root / "projects" / "latent_action_models"
    for root in (str(repo_root), str(project_root)):
        if root not in sys.path:
            sys.path.insert(0, root)
    config = OmegaConf.load(config_path)
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    model = instantiate(config.model)
    snapshot = torch.load(snapshot_path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(snapshot, Mapping) or "model" not in snapshot:
        raise LadderError("snapshot lacks model state")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise LadderError(f"strict snapshot load differs: {incompatible}")
    del snapshot
    model = model.to(device=device).eval()
    return model, config


def _target_dataset(config: Any) -> Any:
    from hydra.utils import instantiate

    dataset = instantiate(config.dataset)
    if len(dataset) != EXPECTED_TRAIN_COUNT:
        raise LadderError("optimization dataset length differs from 512")
    return dataset


def _no_auxiliary_dataset(config: Any, registration: Mapping[str, Any]) -> Any:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    dataset_config = OmegaConf.create(OmegaConf.to_container(config.dataset, resolve=True))
    abc = dataset_config.datasets.ABC
    arrays = registration["cache_arrays"]
    abc._target_ = (
        "robot_wm.datasets.abc.video_residual_anchor_dataset."
        "ABCVideoResidualAnchorDataset"
    )
    for key in (
        "expected_cache_id",
        "expected_pca_sha256",
        "expected_checkpoint_sha256",
        "validate_target_samples",
    ):
        if key in abc:
            del abc[key]
    abc.expected_split = "train"
    abc.expected_clip_count = EXPECTED_TRAIN_COUNT
    abc.expected_manifest_sha256 = registration["inputs"]["train_manifest"]["sha256"]
    abc.expected_rgb_sha256 = arrays["rgb"]["sha256"]
    abc.expected_actions_sha256 = arrays["actions"]["sha256"]
    dataset = instantiate(dataset_config)
    child = dataset.datasets["ABC"]
    if child.auxiliary_target_array_opened is not False:
        raise LadderError("RGB/action-only dataset reports an auxiliary array open")
    return dataset


def _validate_arm_lineage(
    *,
    label: str,
    code: str,
    resolved_config: Path,
    snapshot: Path,
    arm_manifest: Path,
    stage_manifest: Path,
    stage_outcome: Path,
    training_commit: str,
) -> dict[str, Any]:
    arm = read_json(arm_manifest)
    stage = read_json(stage_manifest)
    outcome = read_json(stage_outcome)
    for name, value in (("arm", arm), ("stage", stage), ("outcome", outcome)):
        if not identity_valid(value):
            raise LadderError(f"{label} {name} identity is invalid")
    spec = arm.get("arm")
    if arm.get("git_commit") != training_commit or not isinstance(spec, Mapping) or spec.get("code") != code:
        raise LadderError(f"{label} arm is not the pinned {code} training run")
    if (
        stage.get("arm_identity_sha256") != arm["identity_sha256"]
        or stage.get("arm_code") != code
        or int(stage.get("stage_endpoint_completed_updates", -1)) != 1000
        or Path(str(stage.get("snapshot"))).resolve(strict=True) != snapshot
    ):
        raise LadderError(f"{label} stage does not bind update 1000")
    stage_config = stage.get("resolved_config")
    if not isinstance(stage_config, Mapping) or Path(str(stage_config.get("path"))).resolve(strict=True) != resolved_config:
        raise LadderError(f"{label} stage does not bind its resolved config")
    snapshot_record = outcome.get("snapshot_observed_at_stage_end")
    if (
        outcome.get("stage_identity_sha256") != stage["identity_sha256"]
        or outcome.get("arm_identity_sha256") != arm["identity_sha256"]
        or outcome.get("arm_code") != code
        or int(outcome.get("completed_updates", -1)) != 1000
        or not isinstance(snapshot_record, Mapping)
        or Path(str(snapshot_record.get("path"))).resolve(strict=True) != snapshot
    ):
        raise LadderError(f"{label} outcome does not bind update 1000")
    return {
        "arm_identity_sha256": arm["identity_sha256"],
        "stage_identity_sha256": stage["identity_sha256"],
        "outcome_identity_sha256": outcome["identity_sha256"],
        "arm_spec": dict(spec),
    }


def _prepare_registration(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if len(args.expected_source_commit) != 40 or len(args.training_source_commit) != 40:
        raise LadderError("source/training commits must be full 40-character hashes")
    repo = canonical_directory(args.repo_root, "repository")
    actual_commit = git_output(repo, "rev-parse", "HEAD")
    if actual_commit != args.expected_source_commit:
        raise LadderError(f"source commit differs: {actual_commit} != {args.expected_source_commit}")
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise LadderError("source repository must be clean: " + status.replace("\n", "; "))

    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise LadderError(f"fresh output already exists: {output}")

    paths = {
        "train_manifest": canonical_file(args.train_manifest, "train manifest"),
        "train_cache_metadata": canonical_file(args.cache_metadata, "train cache metadata"),
        "j1_resolved_config": canonical_file(args.j1_resolved_config, "J1 resolved config"),
        "j1_snapshot": canonical_file(args.j1_snapshot, "J1 snapshot"),
        "j1_arm_manifest": canonical_file(args.j1_arm_manifest, "J1 arm manifest"),
        "j1_stage_manifest": canonical_file(args.j1_stage_manifest, "J1 stage manifest"),
        "j1_stage_outcome": canonical_file(args.j1_stage_outcome, "J1 stage outcome"),
        "vpm_resolved_config": canonical_file(args.vpm_resolved_config, "VPM resolved config"),
        "vpm_snapshot": canonical_file(args.vpm_snapshot, "VPM snapshot"),
        "vpm_arm_manifest": canonical_file(args.vpm_arm_manifest, "VPM arm manifest"),
        "vpm_stage_manifest": canonical_file(args.vpm_stage_manifest, "VPM stage manifest"),
        "vpm_stage_outcome": canonical_file(args.vpm_stage_outcome, "VPM stage outcome"),
    }
    manifest_rows = read_jsonl(paths["train_manifest"])
    partitions = validate_partition_contract(manifest_rows)
    metadata = read_json(paths["train_cache_metadata"])
    manifest_sha = sha256_file(paths["train_manifest"])
    if (
        metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache"
        or metadata.get("split") != "train"
        or metadata.get("complete") is not True
        or int(metadata.get("clip_count", -1)) != EXPECTED_TRAIN_COUNT
        or metadata.get("clip_manifest_sha256") != manifest_sha
        or metadata.get("target_shape") != [512, 64, 4, 24, 120]
        or metadata.get("rgb_shape") != [512, 13, 3, 180, 960]
        or metadata.get("actions_shape") != [512, 13, 5, 23]
    ):
        raise LadderError("train cache metadata differs from the immutable ABC cache")
    target_path = _resolve_metadata_file(paths["train_cache_metadata"], metadata.get("target_file"), "target array")
    rgb_path = _resolve_metadata_file(paths["train_cache_metadata"], metadata.get("rgb_file"), "RGB array")
    actions_path = _resolve_metadata_file(paths["train_cache_metadata"], metadata.get("actions_file"), "action array")

    if len(args.j1_snapshot_sha256) != 64 or len(args.vpm_snapshot_sha256) != 64:
        raise LadderError("J1/VPM snapshot digests must be full SHA-256 values")
    records = {
        key: file_record(
            path,
            args.j1_snapshot_sha256 if key == "j1_snapshot" else (
                args.vpm_snapshot_sha256 if key == "vpm_snapshot" else None
            ),
        )
        for key, path in paths.items()
    }
    j1_lineage = _validate_arm_lineage(
        label="J1",
        code="J1",
        resolved_config=paths["j1_resolved_config"],
        snapshot=paths["j1_snapshot"],
        arm_manifest=paths["j1_arm_manifest"],
        stage_manifest=paths["j1_stage_manifest"],
        stage_outcome=paths["j1_stage_outcome"],
        training_commit=args.training_source_commit,
    )
    vpm_lineage = _validate_arm_lineage(
        label="VPM",
        code="VPM",
        resolved_config=paths["vpm_resolved_config"],
        snapshot=paths["vpm_snapshot"],
        arm_manifest=paths["vpm_arm_manifest"],
        stage_manifest=paths["vpm_stage_manifest"],
        stage_outcome=paths["vpm_stage_outcome"],
        training_commit=args.training_source_commit,
    )
    if j1_lineage["arm_spec"].get("schedule_mode") != "tf_first_cascaded" or j1_lineage["arm_spec"].get("condition_mode") != "matched":
        raise LadderError("J1 lineage is not the faithful cascade")
    if vpm_lineage["arm_spec"].get("parameter_matched_control") is not True or vpm_lineage["arm_spec"].get("condition_mode") != "off":
        raise LadderError("VPM lineage is not the parameter-matched video frontier")

    output.mkdir(parents=True, mode=0o700)
    output = output.resolve(strict=True)
    registration = identity_payload(
        {
            "schema": SCHEMA,
            "status": "registered_before_model_weights_or_cache_arrays_open",
            "created_at_utc": _now(),
            "repo_root": str(repo),
            "source_commit": actual_commit,
            "training_source_commit": args.training_source_commit,
            "output_dir": str(output),
            "inputs": records,
            "cache_arrays": {
                "target": stat_record(target_path, str(metadata["target_sha256"])),
                "rgb": stat_record(rgb_path, str(metadata["rgb_sha256"])),
                "actions": stat_record(actions_path, str(metadata["actions_sha256"])),
            },
            "lineage": {"J1": j1_lineage, "VPM": vpm_lineage},
            "partitions": partitions,
            "frozen_ladder": {
                "projection_seed": PROJECTION_SEED,
                "capacity_rungs": list(CAPACITY_RUNGS),
                "ridge_lambdas": list(RIDGE_LAMBDAS),
                "token_sample_count": TOKEN_SAMPLE_COUNT,
                "arms": list(ARMS),
                "capacity_selection_target": "DIRECT calibration corrected velocity MSE only",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "clock_convention": "sigma=1 Gaussian noise; sigma=0 clean data",
            },
            "access_contract": {
                "teacher_target_allowed_only_on_optimization_indices_128_through_383": True,
                "calibration_feature_calls": 0,
                "development_feature_calls": 0,
                "development_teacher_calls": 0,
                "fresh_processes": ["fit", "evaluate-j1", "evaluate-vpm"],
                "validation_opened": False,
                "protected_test_opened": False,
                "reserve_opened": False,
                "wandb_enabled": False,
            },
            "selected_manifest_rows": {
                role: [
                    {
                        "clip_index": index,
                        "clip_id": manifest_rows[index]["clip_id"],
                        "episode_dir": manifest_rows[index]["episode_dir"],
                    }
                    for index in range(start, stop)
                ]
                for role, (start, stop) in {
                    "optimization": OPT_RANGE,
                    "calibration": CAL_RANGE,
                    "development": DEV_RANGE,
                }.items()
            },
            "claim_boundary": (
                "one frozen J1/VPM parent, linear token-local probe, train-only development; "
                "no protected test, validation, full student, FVD, or closed-loop evidence"
            ),
        }
    )
    exclusive_json(output / "registration.json", registration)
    return output, registration


def _load_registration(output: Path) -> dict[str, Any]:
    registration = read_json(output / "registration.json")
    if registration.get("schema") != SCHEMA or not identity_valid(registration):
        raise LadderError("registration identity/schema is invalid")
    if Path(str(registration.get("output_dir"))).resolve(strict=True) != output:
        raise LadderError("registration output path differs")
    repo = canonical_directory(registration["repo_root"], "registered repository")
    if git_output(repo, "rev-parse", "HEAD") != registration["source_commit"]:
        raise LadderError("registered source commit is no longer checked out")
    if git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise LadderError("registered source repository is no longer clean")
    return registration


def _resolve_input(registration: Mapping[str, Any], key: str) -> Path:
    record = registration["inputs"][key]
    path = canonical_file(record["path"], f"registered {key}")
    if path.stat().st_size != int(record["bytes"]):
        raise LadderError(f"registered {key} size changed")
    # Snapshot hashes are expensive but are already checked before registration;
    # phase transitions recheck all smaller manifests/configs and rely on immutable
    # update-1000 lineage plus recorded size for snapshots.
    if key not in {"j1_snapshot", "vpm_snapshot"} and sha256_file(path) != record["sha256"]:
        raise LadderError(f"registered {key} digest changed")
    return path


def _fit_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise LadderError("CUDA is required")
    output, registration = _prepare_registration(args)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    model, config = _load_model(
        _resolve_input(registration, "j1_resolved_config"),
        _resolve_input(registration, "j1_snapshot"),
        device,
    )
    target_dataset = _target_dataset(config)
    projection_cpu = make_projection(int(model.forward_model.transformer.dim), max(CAPACITY_RUNGS))
    projection = projection_cpu.to(device=device)
    stats = None
    grid = patch_size = None
    patch_dim = None
    off_calls = aligned_calls = shuffled_calls = 0
    teacher_rows = 0

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for noise_seed in OPT_NOISE_SEEDS:
            for start in range(OPT_RANGE[0], OPT_RANGE[1], 2):
                indexes = (start, start + 1)
                batch = _batch_samples(target_dataset, indexes, device)
                prepared = _model_inputs(
                    model, batch, noise_seed=noise_seed, nfe=2, need_clean_auxiliary=True
                )
                off_velocity, hidden = _off_prediction_and_hidden(model, prepared)
                aligned_velocity, shuffled_velocity = _teacher_predictions(model, prepared)
                off_calls += 1
                aligned_calls += 1
                shuffled_calls += 1
                observed_grid, observed_patch, observed_dim = _grid_contract(
                    model, off_velocity, hidden
                )
                if grid is None:
                    grid, patch_size, patch_dim = observed_grid, observed_patch, observed_dim
                    stats = _new_sufficient_statistics(max(CAPACITY_RUNGS), patch_dim, device)
                elif (grid, patch_size, patch_dim) != (
                    observed_grid,
                    observed_patch,
                    observed_dim,
                ):
                    raise LadderError("Wan token/grid contract changed between batches")
                projected = hidden.float() @ projection
                targets_video = {
                    "ZERO": torch.zeros_like(off_velocity),
                    "DIRECT": prepared["video_target"] - off_velocity,
                    "PFD_ALIGNED": aligned_velocity - off_velocity,
                    "PFD_SHUFFLED": shuffled_velocity - off_velocity,
                }
                target_tokens = {
                    arm: patchify_video(value, patch_size) for arm, value in targets_video.items()
                }
                positions = future_token_positions(
                    grid=grid,
                    patch_size=patch_size,
                    history_frames=prepared["history_frames"],
                    device=device,
                )
                feature_rows = []
                target_rows = {arm: [] for arm in ARMS}
                for local, clip_index in enumerate(indexes):
                    selected = deterministic_token_subsample(
                        positions,
                        clip_index=clip_index,
                        noise_seed=noise_seed,
                        count=TOKEN_SAMPLE_COUNT,
                    )
                    feature_rows.append(projected[local, selected])
                    for arm in ARMS:
                        target_rows[arm].append(target_tokens[arm][local, selected])
                update_sufficient_statistics(
                    stats,
                    torch.cat(feature_rows, dim=0),
                    {arm: torch.cat(values, dim=0) for arm, values in target_rows.items()},
                )
                teacher_rows += len(indexes)
                del batch, prepared, hidden, projected, targets_video, target_tokens

    if stats is None or grid is None or patch_size is None or patch_dim is None:
        raise LadderError("optimization produced no sufficient statistics")
    expected_tokens = (OPT_RANGE[1] - OPT_RANGE[0]) * len(OPT_NOISE_SEEDS) * TOKEN_SAMPLE_COUNT
    if int(stats["count"]) != expected_tokens or teacher_rows != 512:
        raise LadderError("optimization row/token accounting differs")

    # Close every object capable of reaching the target array before calibration.
    del target_dataset
    gc.collect()
    cal_features: list[Any] = []
    cal_targets: list[Any] = []
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    rgb_path = Path(registration["cache_arrays"]["rgb"]["path"])
    actions_path = Path(registration["cache_arrays"]["actions"]["path"])
    calibration_calls = 0
    with NumpyTargetOpenGuard(target_path) as guard:
        no_aux_dataset = _no_auxiliary_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in CAL_NOISE_SEEDS:
                for start in range(CAL_RANGE[0], CAL_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = _batch_samples(no_aux_dataset, indexes, device)
                    if "auxiliary_target" in batch:
                        raise LadderError("calibration batch exposed an auxiliary target")
                    prepared = _model_inputs(
                        model,
                        batch,
                        noise_seed=noise_seed,
                        nfe=2,
                        need_clean_auxiliary=False,
                    )
                    off_velocity, hidden = _off_prediction_and_hidden(model, prepared)
                    calibration_calls += 1
                    projected = hidden.float() @ projection
                    direct_tokens = patchify_video(
                        prepared["video_target"] - off_velocity, patch_size
                    )
                    positions = future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                        device=device,
                    )
                    for local, clip_index in enumerate(indexes):
                        selected = deterministic_token_subsample(
                            positions,
                            clip_index=clip_index,
                            noise_seed=noise_seed,
                            count=TOKEN_SAMPLE_COUNT,
                        )
                        cal_features.append(projected[local, selected].cpu().to(torch.float16))
                        cal_targets.append(direct_tokens[local, selected].cpu().to(torch.float16))
                    del batch, prepared, hidden, projected, direct_tokens
        opened = sorted({path.resolve(strict=True) for path in guard.opened})
    if opened != sorted({rgb_path.resolve(strict=True), actions_path.resolve(strict=True)}):
        raise LadderError(f"calibration input graph differs: {opened}")
    del no_aux_dataset
    cal_x = torch.cat(cal_features, dim=0).to(device=device, dtype=torch.float32)
    cal_y = torch.cat(cal_targets, dim=0).to(device=device, dtype=torch.float32)
    expected_cal_tokens = (CAL_RANGE[1] - CAL_RANGE[0]) * len(CAL_NOISE_SEEDS) * TOKEN_SAMPLE_COUNT
    if cal_x.shape[0] != expected_cal_tokens:
        raise LadderError("calibration token accounting differs")

    candidate_results = []
    for capacity in CAPACITY_RUNGS:
        for ridge_lambda in RIDGE_LAMBDAS:
            head = fit_ridge_head(
                stats, arm="DIRECT", capacity=capacity, ridge_lambda=ridge_lambda
            )
            prediction = predict_ridge_head(cal_x, head)
            mse = float((prediction - cal_y).square().mean().detach().cpu())
            candidate_results.append(
                {
                    "capacity": capacity,
                    "ridge_lambda": ridge_lambda,
                    "direct_residual_mse": mse,
                    "corrected_velocity_mse": mse,
                    "parameter_count": head["parameter_count"],
                }
            )
    selected = min(
        candidate_results,
        key=lambda row: (
            row["corrected_velocity_mse"],
            row["capacity"],
            -row["ridge_lambda"],
        ),
    )
    selected_heads = {
        arm: fit_ridge_head(
            stats,
            arm=arm,
            capacity=int(selected["capacity"]),
            ridge_lambda=float(selected["ridge_lambda"]),
        )
        for arm in ARMS
    }
    parameter_counts = {head["parameter_count"] for head in selected_heads.values()}
    if len(parameter_counts) != 1:
        raise LadderError("selected arms do not have equal capacity")

    weights = {
        "schema": WEIGHT_SCHEMA,
        "source_commit": registration["source_commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "projection_seed": PROJECTION_SEED,
        "projection": projection_cpu,
        "grid": list(grid),
        "patch_size": list(patch_size),
        "patch_dim": patch_dim,
        "hidden_size": int(projection_cpu.shape[0]),
        "selected_capacity": int(selected["capacity"]),
        "selected_ridge_lambda": float(selected["ridge_lambda"]),
        "heads": selected_heads,
    }
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    exclusive_bytes(output / "adapter_weights.pt", buffer.getvalue())
    weights_record = file_record(output / "adapter_weights.pt")
    target_rms = {
        arm: math.sqrt(float(stats["sum_y2"][arm].cpu()) / (stats["count"] * patch_dim))
        for arm in ARMS
    }
    fit = identity_payload(
        {
            "schema": FIT_SCHEMA,
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "weights": weights_record,
            "optimization": {
                "clip_indices": list(OPT_RANGE),
                "noise_seeds": list(OPT_NOISE_SEEDS),
                "sample_noise_pairs": teacher_rows,
                "sampled_future_tokens": int(stats["count"]),
                "token_sample_count_per_pair": TOKEN_SAMPLE_COUNT,
                "off_wan_invocations": off_calls,
                "aligned_teacher_wan_invocations": aligned_calls,
                "shuffled_teacher_wan_invocations": shuffled_calls,
                "target_rms": target_rms,
            },
            "calibration": {
                "clip_indices": list(CAL_RANGE),
                "noise_seeds": list(CAL_NOISE_SEEDS),
                "sampled_future_tokens": int(cal_x.shape[0]),
                "off_wan_invocations": calibration_calls,
                "teacher_wan_invocations": 0,
                "auxiliary_target_array_opened": False,
                "opened_numpy_arrays": [str(path) for path in opened],
                "candidate_results": candidate_results,
                "selection_rule": (
                    "minimum DIRECT corrected-velocity MSE; ties prefer smaller "
                    "capacity then larger ridge penalty"
                ),
            },
            "selected": selected,
            "equal_parameter_count_per_arm": parameter_counts.pop(),
            "zero_head_exact_zero": True,
            "validation_opened": False,
            "protected_test_opened": False,
            "reserve_opened": False,
            "wandb_enabled": False,
            "claim_boundary": "closed-form train-only preflight; no deployable outcome yet",
        }
    )
    exclusive_json(output / "fit.json", fit)
    print(json.dumps({"fit_identity_sha256": fit["identity_sha256"], "selected": selected}, sort_keys=True))
    return 0


def _load_weights(output: Path, registration: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    fit = read_json(output / "fit.json")
    if fit.get("schema") != FIT_SCHEMA or not identity_valid(fit):
        raise LadderError("fit identity/schema is invalid")
    if fit.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise LadderError("fit does not bind the registration")
    record = fit.get("weights")
    if not isinstance(record, Mapping):
        raise LadderError("fit lacks a weights record")
    path = canonical_file(record["path"], "adapter weights")
    if path.parent != output or sha256_file(path) != record["sha256"]:
        raise LadderError("adapter weights digest/path differs")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(weights, Mapping)
        or weights.get("schema") != WEIGHT_SCHEMA
        or weights.get("registration_identity_sha256") != registration["identity_sha256"]
        or set(weights.get("heads", {})) != set(ARMS)
    ):
        raise LadderError("adapter weight payload differs")
    return fit, dict(weights)


def _to_uint8_video(value: Any) -> Any:
    import torch

    return value.float().clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)


def _decoded_metrics(
    prediction: Any,
    target: Any,
    *,
    prediction_history: Any,
    target_history: Any,
) -> tuple[Any, Any]:
    import torch

    prediction = prediction.double() / 255.0
    target = target.double() / 255.0
    expected_history_shape = (
        prediction.shape[0],
        prediction.shape[1],
        1,
        prediction.shape[3],
        prediction.shape[4],
    )
    if tuple(prediction_history.shape) != expected_history_shape or tuple(target_history.shape) != expected_history_shape:
        raise LadderError("decoded temporal-boundary history shape differs")
    temporal_prediction = torch.cat(
        (prediction_history.double() / 255.0, prediction), dim=2
    )
    temporal_target = torch.cat(
        (target_history.double() / 255.0, target), dim=2
    )
    mse = (prediction - target).square().flatten(1).mean(1)
    temporal = (
        temporal_prediction.diff(dim=2) - temporal_target.diff(dim=2)
    ).square().flatten(1).mean(1)
    return mse, temporal


def _per_sample_future_metrics(
    *,
    velocity: Any,
    base_velocity: Any,
    direct_residual: Any,
    final: Any,
    clean: Any,
    decoded: Any,
    raw_target: Any,
    raw_history: Any,
    vae_target: Any,
    vae_history: Any,
    history_frames: int,
) -> dict[str, Any]:
    import torch

    residual = velocity[:, :, history_frames:] - base_velocity[:, :, history_frames:]
    target = direct_residual[:, :, history_frames:]
    reduce = tuple(range(1, target.ndim))
    residual_sse = (residual.float() - target.float()).square().sum(dim=reduce)
    residual_energy = target.float().square().sum(dim=reduce).clamp_min(1e-12)
    r2 = 1.0 - residual_sse / residual_energy
    cosine = torch.nn.functional.cosine_similarity(
        residual.float().flatten(1), target.float().flatten(1), dim=1, eps=1e-12
    )
    corrected_velocity_mse = (
        velocity[:, :, history_frames:].float()
        - (base_velocity + direct_residual)[:, :, history_frames:].float()
    ).square().flatten(1).mean(1)
    numerator = (final[:, :, history_frames:].float() - clean[:, :, history_frames:].float()).square().flatten(1).sum(1)
    denominator = clean[:, :, history_frames:].float().square().flatten(1).sum(1).clamp_min(1e-12)
    latent_nmse = numerator / denominator
    decoded_mse, temporal_mse = _decoded_metrics(
        decoded,
        raw_target,
        prediction_history=raw_history,
        target_history=raw_history,
    )
    decoded_vs_vae_mse, decoded_vs_vae_temporal = _decoded_metrics(
        decoded,
        vae_target,
        prediction_history=raw_history,
        target_history=vae_history,
    )
    vae_vs_raw_mse, vae_vs_raw_temporal = _decoded_metrics(
        vae_target,
        raw_target,
        prediction_history=vae_history,
        target_history=raw_history,
    )
    return {
        "residual_r2": r2,
        "residual_cosine": cosine,
        "corrected_velocity_mse": corrected_velocity_mse,
        "video_future_nmse": latent_nmse,
        "decoded_mse_unit_range": decoded_mse,
        "decoded_temporal_difference_mse_unit_range": temporal_mse,
        "prediction_vs_vae_reconstruction_mse_unit_range": decoded_vs_vae_mse,
        "prediction_vs_vae_reconstruction_temporal_mse_unit_range": decoded_vs_vae_temporal,
        "vae_reconstruction_vs_raw_mse_unit_range": vae_vs_raw_mse,
        "vae_reconstruction_vs_raw_temporal_mse_unit_range": vae_vs_raw_temporal,
    }


def _one_step_final(initial: Any, velocity: Any, reference: Any, history_frames: int) -> Any:
    # Euler rectified-flow integration from sigma=1 to sigma=0.
    final = initial.float() - velocity.float()
    final[:, :, :history_frames] = reference[:, :, :history_frames].float()
    return final


def _endpoint_rows_j1(output: Path, registration: Mapping[str, Any], fit: Mapping[str, Any], weights: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    device = torch.device("cuda", 0)
    model, config = _load_model(
        _resolve_input(registration, "j1_resolved_config"),
        _resolve_input(registration, "j1_snapshot"),
        device,
    )
    projection = weights["projection"].to(device=device, dtype=torch.float32)
    grid = tuple(int(value) for value in weights["grid"])
    patch_size = tuple(int(value) for value in weights["patch_size"])
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_opened = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    manifest_rows = read_jsonl(_resolve_input(registration, "train_manifest"))
    rows: list[dict[str, Any]] = []
    wan_invocations = 0
    adapter_batch_timings: dict[str, list[float]] = {arm: [] for arm in ARMS}
    with NumpyTargetOpenGuard(target_path) as guard:
        dataset = _no_auxiliary_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in DEV_NOISE_SEEDS:
                for start in range(DEV_RANGE[0], DEV_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = _batch_samples(dataset, indexes, device)
                    if "auxiliary_target" in batch:
                        raise LadderError("J1 development batch exposed auxiliary_target")
                    prepared = _model_inputs(
                        model, batch, noise_seed=noise_seed, nfe=2, need_clean_auxiliary=False
                    )
                    off_velocity, hidden = _off_prediction_and_hidden(model, prepared)
                    wan_invocations += 1
                    observed_grid, observed_patch, _ = _grid_contract(model, off_velocity, hidden)
                    if observed_grid != grid or observed_patch != patch_size:
                        raise LadderError("development Wan grid differs from the fit")
                    projection_start = torch.cuda.Event(enable_timing=True)
                    projection_end = torch.cuda.Event(enable_timing=True)
                    projection_start.record()
                    projected = hidden.float() @ projection
                    projection_end.record()
                    projection_end.synchronize()
                    projection_ms = projection_start.elapsed_time(projection_end)
                    history_frames = prepared["history_frames"]
                    future_positions = future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=history_frames,
                        device=device,
                    )
                    velocities = {"J1_OFF": off_velocity.float()}
                    residual_tensors = {}
                    for arm in ARMS:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        residual_tokens = predict_ridge_head(projected, weights["heads"][arm])
                        residual_tokens = residual_tokens.clone()
                        residual_tokens[:, : int(future_positions[0])] = 0.0
                        residual = unpatchify_tokens(
                            residual_tokens,
                            grid=grid,
                            patch_size=patch_size,
                            channels=off_velocity.shape[1],
                        )
                        end_event.record()
                        end_event.synchronize()
                        adapter_batch_timings[arm].append(
                            float(projection_ms + start_event.elapsed_time(end_event))
                        )
                        residual_tensors[arm] = residual
                        velocities[arm] = off_velocity.float() + residual.float()
                    if int(torch.count_nonzero(residual_tensors["ZERO"])) != 0:
                        raise LadderError("ZERO adapter is not an exact no-op")
                    video_clean = prepared["video_clean"]
                    decoded_clean = model.rgb_tokenizer.decode_temporal(
                        video_clean, out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1])
                    )
                    future_pixel_frames = min(model.num_future_frames, decoded_clean.shape[2])
                    vae_target = _to_uint8_video(
                        decoded_clean[:, :, -future_pixel_frames:]
                    )
                    vae_history = _to_uint8_video(
                        decoded_clean[
                            :, :, -(future_pixel_frames + 1) : -future_pixel_frames
                        ]
                    )
                    raw_video = batch["rgb"].permute(0, 2, 1, 3, 4)
                    raw_target = _to_uint8_video(
                        raw_video[:, :, -future_pixel_frames:]
                    )
                    raw_history = _to_uint8_video(
                        raw_video[
                            :, :, -(future_pixel_frames + 1) : -future_pixel_frames
                        ]
                    )
                    direct_residual = prepared["video_target"] - off_velocity
                    finals = {}
                    for endpoint, velocity in velocities.items():
                        final = _one_step_final(
                            prepared["initial_video"], velocity, prepared["reference"], history_frames
                        )
                        finals[endpoint] = final
                        decoded = _to_uint8_video(
                            model.rgb_tokenizer.decode_temporal(
                                final.to(batch["rgb"].dtype),
                                out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                            )[:, :, -future_pixel_frames:]
                        )
                        metrics = _per_sample_future_metrics(
                            velocity=velocity,
                            base_velocity=off_velocity,
                            direct_residual=direct_residual,
                            final=final,
                            clean=video_clean,
                            decoded=decoded,
                            raw_target=raw_target,
                            raw_history=raw_history,
                            vae_target=vae_target,
                            vae_history=vae_history,
                            history_frames=history_frames,
                        )
                        for local, clip_index in enumerate(indexes):
                            rows.append(
                                identity_payload(
                                    {
                                        "schema": ROW_SCHEMA,
                                        "endpoint": endpoint,
                                        "clip_index": clip_index,
                                        "clip_id": manifest_rows[clip_index]["clip_id"],
                                        "episode_dir": manifest_rows[clip_index]["episode_dir"],
                                        "noise_seed": noise_seed,
                                        "sigma_video": 1.0,
                                        "sigma_final": 0.0,
                                        "wan_calls": 1,
                                        "teacher_calls": 0,
                                        "auxiliary_target_array_opened": False,
                                        "future_feature_input": False,
                                        "metrics": {
                                            key: float(value[local].detach().cpu())
                                            for key, value in metrics.items()
                                        },
                                        "adapter_latency_ms_per_batch": (
                                            0.0
                                            if endpoint == "J1_OFF"
                                            else adapter_batch_timings[endpoint][-1]
                                        ),
                                        "tensor_sha256": {
                                            "initial_video": _safe_tensor_sha256(
                                                prepared["initial_video"][local : local + 1]
                                            ),
                                            "final_video": _safe_tensor_sha256(
                                                final[local : local + 1]
                                            ),
                                            "raw_future_target": _safe_tensor_sha256(
                                                raw_target[local : local + 1]
                                            ),
                                            "raw_history_boundary": _safe_tensor_sha256(
                                                raw_history[local : local + 1]
                                            ),
                                        },
                                        "validation_opened": False,
                                        "protected_test_opened": False,
                                        "reserve_opened": False,
                                    }
                                )
                            )
                    if not torch.equal(finals["J1_OFF"], finals["ZERO"]):
                        raise LadderError("ZERO final differs bit-for-bit from J1_OFF")
                    del batch, prepared, hidden, projected, velocities, residual_tensors, finals
        opened = sorted({path.resolve(strict=True) for path in guard.opened})
    if set(opened) != expected_opened:
        raise LadderError(f"J1 endpoint input graph differs: {opened}")
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(J1_ENDPOINTS)
    expected_invocations = ((DEV_RANGE[1] - DEV_RANGE[0]) // 2) * len(DEV_NOISE_SEEDS)
    if len(rows) != expected_rows or wan_invocations != expected_invocations:
        raise LadderError("J1 endpoint row/Wan accounting differs")
    receipt = identity_payload(
        {
            "schema": ENDPOINT_SCHEMA,
            "endpoint_family": "J1",
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "row_count": len(rows),
            "wan_invocations": wan_invocations,
            "wan_sample_calls": (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS),
            "teacher_calls": 0,
            "auxiliary_target_array_opened": False,
            "opened_numpy_arrays": [str(path) for path in opened],
            "zero_equals_off_bit_exact": True,
            "primary_decoded_target": "cached raw held-out future RGB uint8",
            "temporal_metric_includes_history_to_first_future_boundary": True,
            "prediction_vs_vae_reconstruction_is_diagnostic": True,
            "adapter_latency_ms_per_batch": {
                arm: {
                    "mean": float(sum(values) / len(values)),
                    "p95": float(__import__("numpy").quantile(values, 0.95)),
                }
                for arm, values in adapter_batch_timings.items()
            },
            "validation_opened": False,
            "protected_test_opened": False,
            "reserve_opened": False,
        }
    )
    return rows, receipt


def _endpoint_rows_vpm(output: Path, registration: Mapping[str, Any], fit: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    device = torch.device("cuda", 0)
    model, config = _load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_opened = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    manifest_rows = read_jsonl(_resolve_input(registration, "train_manifest"))
    rows: list[dict[str, Any]] = []
    wan_invocations = 0
    with NumpyTargetOpenGuard(target_path) as guard:
        dataset = _no_auxiliary_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in DEV_NOISE_SEEDS:
                for start in range(DEV_RANGE[0], DEV_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = _batch_samples(dataset, indexes, device)
                    if "auxiliary_target" in batch:
                        raise LadderError("VPM development batch exposed auxiliary_target")
                    prepared = _model_inputs(
                        model, batch, noise_seed=noise_seed, nfe=1, need_clean_auxiliary=False
                    )
                    velocity, _hidden = _off_prediction_and_hidden(model, prepared)
                    wan_invocations += 1
                    history_frames = prepared["history_frames"]
                    final = _one_step_final(
                        prepared["initial_video"], velocity, prepared["reference"], history_frames
                    )
                    decoded_clean = model.rgb_tokenizer.decode_temporal(
                        prepared["video_clean"],
                        out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                    )
                    future_pixel_frames = min(model.num_future_frames, decoded_clean.shape[2])
                    vae_target = _to_uint8_video(
                        decoded_clean[:, :, -future_pixel_frames:]
                    )
                    vae_history = _to_uint8_video(
                        decoded_clean[
                            :, :, -(future_pixel_frames + 1) : -future_pixel_frames
                        ]
                    )
                    raw_video = batch["rgb"].permute(0, 2, 1, 3, 4)
                    raw_target = _to_uint8_video(
                        raw_video[:, :, -future_pixel_frames:]
                    )
                    raw_history = _to_uint8_video(
                        raw_video[
                            :, :, -(future_pixel_frames + 1) : -future_pixel_frames
                        ]
                    )
                    decoded = _to_uint8_video(
                        model.rgb_tokenizer.decode_temporal(
                            final.to(batch["rgb"].dtype),
                            out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                        )[:, :, -future_pixel_frames:]
                    )
                    direct_residual = prepared["video_target"] - velocity
                    metrics = _per_sample_future_metrics(
                        velocity=velocity,
                        base_velocity=velocity,
                        direct_residual=direct_residual,
                        final=final,
                        clean=prepared["video_clean"],
                        decoded=decoded,
                        raw_target=raw_target,
                        raw_history=raw_history,
                        vae_target=vae_target,
                        vae_history=vae_history,
                        history_frames=history_frames,
                    )
                    for local, clip_index in enumerate(indexes):
                        rows.append(
                            identity_payload(
                                {
                                    "schema": ROW_SCHEMA,
                                    "endpoint": "VPM1",
                                    "clip_index": clip_index,
                                    "clip_id": manifest_rows[clip_index]["clip_id"],
                                    "episode_dir": manifest_rows[clip_index]["episode_dir"],
                                    "noise_seed": noise_seed,
                                    "sigma_video": 1.0,
                                    "sigma_final": 0.0,
                                    "wan_calls": 1,
                                    "teacher_calls": 0,
                                    "auxiliary_target_array_opened": False,
                                    "future_feature_input": False,
                                    "metrics": {
                                        key: float(value[local].detach().cpu())
                                        for key, value in metrics.items()
                                    },
                                    "adapter_latency_ms_per_batch": 0.0,
                                    "tensor_sha256": {
                                        "initial_video": _safe_tensor_sha256(
                                            prepared["initial_video"][local : local + 1]
                                        ),
                                        "final_video": _safe_tensor_sha256(
                                            final[local : local + 1]
                                        ),
                                        "raw_future_target": _safe_tensor_sha256(
                                            raw_target[local : local + 1]
                                        ),
                                        "raw_history_boundary": _safe_tensor_sha256(
                                            raw_history[local : local + 1]
                                        ),
                                    },
                                    "validation_opened": False,
                                    "protected_test_opened": False,
                                    "reserve_opened": False,
                                }
                            )
                        )
                    del batch, prepared, final, decoded, decoded_clean, _hidden
        opened = sorted({path.resolve(strict=True) for path in guard.opened})
    if set(opened) != expected_opened:
        raise LadderError(f"VPM endpoint input graph differs: {opened}")
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS)
    expected_invocations = ((DEV_RANGE[1] - DEV_RANGE[0]) // 2) * len(DEV_NOISE_SEEDS)
    if len(rows) != expected_rows or wan_invocations != expected_invocations:
        raise LadderError("VPM endpoint row/Wan accounting differs")
    receipt = identity_payload(
        {
            "schema": ENDPOINT_SCHEMA,
            "endpoint_family": "VPM",
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "row_count": len(rows),
            "wan_invocations": wan_invocations,
            "wan_sample_calls": (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS),
            "teacher_calls": 0,
            "auxiliary_target_array_opened": False,
            "opened_numpy_arrays": [str(path) for path in opened],
            "primary_decoded_target": "cached raw held-out future RGB uint8",
            "temporal_metric_includes_history_to_first_future_boundary": True,
            "prediction_vs_vae_reconstruction_is_diagnostic": True,
            "validation_opened": False,
            "protected_test_opened": False,
            "reserve_opened": False,
        }
    )
    return rows, receipt


def _evaluate_phase(args: argparse.Namespace, family: str) -> int:
    import torch

    if not torch.cuda.is_available():
        raise LadderError("CUDA is required")
    output = canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, weights = _load_weights(output, registration)
    torch.cuda.set_device(0)
    if family == "J1":
        rows_path = output / "j1_development_rows.jsonl"
        receipt_path = output / "j1_endpoint_complete.json"
        rows, receipt = _endpoint_rows_j1(output, registration, fit, weights)
    elif family == "VPM":
        rows_path = output / "vpm_development_rows.jsonl"
        receipt_path = output / "vpm_endpoint_complete.json"
        rows, receipt = _endpoint_rows_vpm(output, registration, fit)
    else:
        raise LadderError(f"unknown endpoint family: {family}")
    exclusive_jsonl(rows_path, rows)
    receipt = identity_payload(
        {
            **receipt,
            "rows": file_record(rows_path),
        }
    )
    exclusive_json(receipt_path, receipt)
    print(json.dumps({"family": family, "receipt_identity_sha256": receipt["identity_sha256"]}, sort_keys=True))
    return 0


def _bootstrap_relative(
    control: Any, candidate: Any, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    import numpy as np

    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if control.shape != candidate.shape or control.ndim != 2 or control.shape[0] < 8:
        raise LadderError("paired bootstrap arrays must be [episodes, noise_seeds]")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all() or control.mean() <= 0:
        raise LadderError("paired bootstrap input is non-finite/non-positive")
    point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, control.shape[0], size=(replicates, control.shape[0]))
    control_samples = control[indexes].mean(axis=(1, 2))
    candidate_samples = candidate[indexes].mean(axis=(1, 2))
    effects = 100.0 * (control_samples - candidate_samples) / control_samples
    low, high = np.quantile(effects, (0.025, 0.975))
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": float(point),
        "paired_cluster_bootstrap_95_percent": [float(low), float(high)],
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "episode (four development noise seeds remain clustered)",
    }


def analyze_development_rows(
    j1_rows: Sequence[Mapping[str, Any]],
    vpm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    expected_keys = {
        (endpoint, clip, seed)
        for endpoint in ALL_ENDPOINTS
        for clip in range(*DEV_RANGE)
        for seed in DEV_NOISE_SEEDS
    }
    inventory: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in (*j1_rows, *vpm_rows):
        if row.get("schema") != ROW_SCHEMA or not identity_valid(row):
            raise LadderError("development row identity/schema is invalid")
        key = (str(row.get("endpoint")), int(row.get("clip_index", -1)), int(row.get("noise_seed", -1)))
        if key in inventory:
            raise LadderError(f"duplicate development row: {key}")
        if (
            int(row.get("wan_calls", -1)) != 1
            or int(row.get("teacher_calls", -1)) != 0
            or row.get("auxiliary_target_array_opened") is not False
            or row.get("future_feature_input") is not False
            or row.get("validation_opened") is not False
            or row.get("protected_test_opened") is not False
            or row.get("reserve_opened") is not False
        ):
            raise LadderError("development row violates feature-free/train-only contract")
        inventory[key] = row
    if set(inventory) != expected_keys:
        missing = sorted(expected_keys - set(inventory))[:5]
        extra = sorted(set(inventory) - expected_keys)[:5]
        raise LadderError(f"development inventory differs: missing={missing}, extra={extra}")

    for clip in range(*DEV_RANGE):
        for seed in DEV_NOISE_SEEDS:
            for field, label in (
                ("initial_video", "initial Gaussian noise"),
                ("raw_future_target", "raw future target"),
                ("raw_history_boundary", "raw history boundary"),
            ):
                hashes = {
                    inventory[(endpoint, clip, seed)]["tensor_sha256"][field]
                    for endpoint in ALL_ENDPOINTS
                }
                if len(hashes) != 1:
                    raise LadderError(f"J1/VPM {label} differs for {(clip, seed)}")
            off = inventory[("J1_OFF", clip, seed)]
            zero = inventory[("ZERO", clip, seed)]
            if off["tensor_sha256"]["final_video"] != zero["tensor_sha256"]["final_video"] or off["metrics"] != zero["metrics"]:
                raise LadderError(f"ZERO does not reproduce J1_OFF at {(clip, seed)}")

    clips = list(range(*DEV_RANGE))
    seeds = list(DEV_NOISE_SEEDS)

    def values(endpoint: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                [float(inventory[(endpoint, clip, seed)]["metrics"][metric]) for seed in seeds]
                for clip in clips
            ],
            dtype=np.float64,
        )

    aggregates: dict[str, Any] = {}
    for endpoint in ALL_ENDPOINTS:
        aggregates[endpoint] = {
            metric: {
                "mean": float(values(endpoint, metric).mean()),
                "std_over_all_pairs": float(values(endpoint, metric).std(ddof=1)),
            }
            for metric in (*ERROR_METRICS, "residual_r2", "residual_cosine")
        }

    comparisons: dict[str, Any] = {}
    comparison_index = 0
    for candidate, controls in {
        "PFD_ALIGNED": ("J1_OFF", "ZERO", "DIRECT", "PFD_SHUFFLED", "VPM1"),
        "DIRECT": ("J1_OFF", "VPM1"),
        "PFD_SHUFFLED": ("J1_OFF",),
    }.items():
        for control in controls:
            label = f"{candidate}_vs_{control}"
            comparisons[label] = {}
            for metric_index, metric in enumerate(ERROR_METRICS):
                comparisons[label][metric] = _bootstrap_relative(
                    values(control, metric),
                    values(candidate, metric),
                    seed=BOOTSTRAP_SEED + comparison_index * 10 + metric_index,
                )
            comparison_index += 1

    def effect(candidate: str, control: str, metric: str) -> dict[str, Any]:
        return comparisons[f"{candidate}_vs_{control}"][metric]

    velocity_direct = effect("PFD_ALIGNED", "DIRECT", "corrected_velocity_mse")
    velocity_shuffled = effect("PFD_ALIGNED", "PFD_SHUFFLED", "corrected_velocity_mse")
    gates: dict[str, bool] = {
        "aligned_velocity_beats_direct_ci": velocity_direct["relative_improvement_percent"] > 0
        and velocity_direct["paired_cluster_bootstrap_95_percent"][0] > 0,
        "aligned_velocity_beats_shuffled_ci": velocity_shuffled["relative_improvement_percent"] > 0
        and velocity_shuffled["paired_cluster_bootstrap_95_percent"][0] > 0,
    }
    for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range"):
        for control in ("J1_OFF", "ZERO", "DIRECT"):
            result = effect("PFD_ALIGNED", control, metric)
            gates[f"aligned_{metric}_beats_{control}_3pct_ci1"] = (
                result["relative_improvement_percent"] >= 3.0
                and result["paired_cluster_bootstrap_95_percent"][0] > 1.0
            )
        shuffled = effect("PFD_ALIGNED", "PFD_SHUFFLED", metric)
        gates[f"aligned_{metric}_beats_shuffled_1pct"] = (
            shuffled["relative_improvement_percent"] >= 1.0
            and shuffled["paired_cluster_bootstrap_95_percent"][0] > 0.0
        )
        vpm = effect("PFD_ALIGNED", "VPM1", metric)
        gates[f"aligned_{metric}_beats_vpm_ci"] = (
            vpm["relative_improvement_percent"] > 0.0
            and vpm["paired_cluster_bootstrap_95_percent"][0] > 0.0
        )
    for control in ("DIRECT", "VPM1"):
        latent = effect("PFD_ALIGNED", control, "video_future_nmse")
        gates[f"aligned_latent_nonnegative_vs_{control}_guardrail"] = (
            latent["relative_improvement_percent"] >= 0.0
            and latent["paired_cluster_bootstrap_95_percent"][0] > -1.0
        )
    aligned_r2 = aggregates["PFD_ALIGNED"]["residual_r2"]["mean"]
    aligned_cosine = aggregates["PFD_ALIGNED"]["residual_cosine"]["mean"]
    gates["aligned_r2_beats_direct_and_shuffled"] = all(
        aligned_r2 > aggregates[control]["residual_r2"]["mean"]
        for control in ("DIRECT", "PFD_SHUFFLED")
    )
    gates["aligned_cosine_beats_direct_and_shuffled"] = all(
        aligned_cosine > aggregates[control]["residual_cosine"]["mean"]
        for control in ("DIRECT", "PFD_SHUFFLED")
    )
    gates["feature_free_serving_contract"] = True
    gates["zero_equals_off_bit_exact"] = True
    gates["train_only_no_reserve"] = True
    all_passed = all(gates.values())
    return identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "created_at_utc": _now(),
            "development_episode_count": len(clips),
            "noise_seeds_per_episode": len(seeds),
            "paired_outcome_count": len(clips) * len(seeds),
            "primary_decoded_target": "cached raw held-out future RGB uint8",
            "temporal_metric_includes_history_to_first_future_boundary": True,
            "prediction_vs_vae_reconstruction_is_diagnostic": True,
            "aggregates": aggregates,
            "comparisons": comparisons,
            "gates": {**gates, "all_passed": all_passed},
            "decision": (
                "CAUSALLY_COMPRESSIBLE_ADVANCE"
                if all_passed
                else "STOP_PRIVILEGED_NOT_CAUSALLY_COMPRESSIBLE"
            ),
            "claim_boundary": (
                "train-only development, one frozen parent pair, linear token-local adapter; "
                "no validation/protected test, long-horizon, FVD, or closed-loop claim"
            ),
        }
    )


def _analyze_phase(args: argparse.Namespace) -> int:
    output = canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    receipts = {}
    rows = {}
    for family, prefix in (("J1", "j1"), ("VPM", "vpm")):
        receipt = read_json(output / f"{prefix}_endpoint_complete.json")
        if receipt.get("schema") != ENDPOINT_SCHEMA or not identity_valid(receipt):
            raise LadderError(f"{family} endpoint receipt identity/schema is invalid")
        if (
            receipt.get("registration_identity_sha256") != registration["identity_sha256"]
            or receipt.get("fit_identity_sha256") != fit["identity_sha256"]
            or receipt.get("teacher_calls") != 0
            or receipt.get("auxiliary_target_array_opened") is not False
        ):
            raise LadderError(f"{family} endpoint receipt violates serving contract")
        row_path = canonical_file(receipt["rows"]["path"], f"{family} rows")
        if row_path.parent != output or sha256_file(row_path) != receipt["rows"]["sha256"]:
            raise LadderError(f"{family} row artifact differs")
        receipts[family] = receipt
        rows[family] = read_jsonl(row_path)
    analysis = analyze_development_rows(rows["J1"], rows["VPM"])
    analysis = identity_payload(
        {
            **analysis,
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "j1_endpoint_identity_sha256": receipts["J1"]["identity_sha256"],
            "vpm_endpoint_identity_sha256": receipts["VPM"]["identity_sha256"],
        }
    )
    exclusive_json(output / "analysis.json", analysis)
    analysis_record = file_record(output / "analysis.json")
    complete = identity_payload(
        {
            "schema": COMPLETE_SCHEMA,
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "j1_endpoint_identity_sha256": receipts["J1"]["identity_sha256"],
            "vpm_endpoint_identity_sha256": receipts["VPM"]["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "analysis": analysis_record,
            "decision": analysis["decision"],
            "j1_row_count": len(rows["J1"]),
            "vpm_row_count": len(rows["VPM"]),
            "development_teacher_calls": 0,
            "development_auxiliary_target_array_opens": 0,
            "validation_opened": False,
            "protected_test_opened": False,
            "reserve_opened": False,
        }
    )
    exclusive_json(output / "run_complete.json", complete)
    print(json.dumps({"analysis_identity_sha256": analysis["identity_sha256"], "decision": analysis["decision"]}, sort_keys=True))
    return 0


def _audit_phase(args: argparse.Namespace) -> int:
    output = canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    analysis = read_json(output / "analysis.json")
    complete = read_json(output / "run_complete.json")
    for label, value, schema in (
        ("analysis", analysis, ANALYSIS_SCHEMA),
        ("completion", complete, COMPLETE_SCHEMA),
    ):
        if value.get("schema") != schema or not identity_valid(value):
            raise LadderError(f"{label} identity/schema is invalid")
    expected_links = {
        "registration_identity_sha256": registration["identity_sha256"],
        "fit_identity_sha256": fit["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "decision": analysis["decision"],
    }
    if any(complete.get(key) != value for key, value in expected_links.items()):
        raise LadderError("completion links differ")
    if (
        complete.get("development_teacher_calls") != 0
        or complete.get("development_auxiliary_target_array_opens") != 0
        or complete.get("validation_opened") is not False
        or complete.get("protected_test_opened") is not False
        or complete.get("reserve_opened") is not False
    ):
        raise LadderError("completion violates train-only feature-free contract")
    # Re-run the full row inventory and paired-noise/no-op validation.
    j1_rows = read_jsonl(output / "j1_development_rows.jsonl")
    vpm_rows = read_jsonl(output / "vpm_development_rows.jsonl")
    recomputed = analyze_development_rows(j1_rows, vpm_rows)
    for key in ("aggregates", "comparisons", "gates", "decision"):
        if recomputed[key] != analysis[key]:
            raise LadderError(f"audited analysis {key} differs")
    artifact_names = (
        "registration.json",
        "fit.json",
        "adapter_weights.pt",
        "j1_development_rows.jsonl",
        "j1_endpoint_complete.json",
        "vpm_development_rows.jsonl",
        "vpm_endpoint_complete.json",
        "analysis.json",
        "run_complete.json",
    )
    audit = identity_payload(
        {
            "schema": AUDIT_SCHEMA,
            "created_at_utc": _now(),
            "output_dir": str(output),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "decision": analysis["decision"],
            "artifact_sha256": {
                name: sha256_file(canonical_file(output / name, name)) for name in artifact_names
            },
            "row_inventory_recomputed": True,
            "initial_noise_pairing_recomputed": True,
            "zero_noop_recomputed": True,
            "development_teacher_calls": 0,
            "development_auxiliary_target_array_opens": 0,
            "validation_opened": False,
            "protected_test_opened": False,
            "reserve_opened": False,
        }
    )
    exclusive_json(output / "audit.json", audit)
    print(json.dumps({"audit_identity_sha256": audit["identity_sha256"], "decision": audit["decision"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)
    fit = subcommands.add_parser("fit", help="register and fit the train-only ladder")
    fit.add_argument("--repo-root", required=True)
    fit.add_argument("--expected-source-commit", required=True)
    fit.add_argument("--training-source-commit", required=True)
    fit.add_argument("--train-manifest", required=True)
    fit.add_argument("--cache-metadata", required=True)
    fit.add_argument("--j1-resolved-config", required=True)
    fit.add_argument("--j1-snapshot", required=True)
    fit.add_argument("--j1-snapshot-sha256", required=True)
    fit.add_argument("--j1-arm-manifest", required=True)
    fit.add_argument("--j1-stage-manifest", required=True)
    fit.add_argument("--j1-stage-outcome", required=True)
    fit.add_argument("--vpm-resolved-config", required=True)
    fit.add_argument("--vpm-snapshot", required=True)
    fit.add_argument("--vpm-snapshot-sha256", required=True)
    fit.add_argument("--vpm-arm-manifest", required=True)
    fit.add_argument("--vpm-stage-manifest", required=True)
    fit.add_argument("--vpm-stage-outcome", required=True)
    fit.add_argument("--output-dir", required=True)
    for command in ("evaluate-j1", "evaluate-vpm", "analyze", "audit"):
        subcommands.add_parser(command).add_argument("--output-dir", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "fit":
        return _fit_phase(args)
    if args.command == "evaluate-j1":
        return _evaluate_phase(args, "J1")
    if args.command == "evaluate-vpm":
        return _evaluate_phase(args, "VPM")
    if args.command == "analyze":
        return _analyze_phase(args)
    if args.command == "audit":
        return _audit_phase(args)
    raise LadderError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LadderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
