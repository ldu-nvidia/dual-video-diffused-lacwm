#!/usr/bin/env python3
"""Post-selection train-only VPM causal Haar-residual exploration.

The experiment uses a one-level, view-isolated orthogonal 3-D Haar grouping on
the exact VPM@1 flow residual.  Rows 416--479 are explicitly prior-inspected
exploratory outcomes; rows 480--511, validation, and protected test are barred.
LACWM's clock is sigma=1 noise and sigma=0 clean video.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import causal_compressibility_ladder as ladder
from tools import vpm_direct_residual_frontier as frontier


SCHEMA = "vpm-causal-haar-residual-registration-v1"
FIT_SCHEMA = "vpm-causal-haar-residual-fit-v1"
WEIGHT_SCHEMA = "vpm-causal-haar-residual-weights-v1"
ROW_SCHEMA = "vpm-causal-haar-residual-row-v1"
TIMING_SCHEMA = "vpm-causal-haar-residual-timing-v1"
ENDPOINT_SCHEMA = "vpm-causal-haar-residual-endpoint-v1"
ANALYSIS_SCHEMA = "vpm-causal-haar-residual-analysis-v1"
COMPLETE_SCHEMA = "vpm-causal-haar-residual-complete-v1"
AUDIT_SCHEMA = "vpm-causal-haar-residual-audit-v1"

BANDS = ("ST_COARSE", "TEMPORAL_CHANGE", "SPATIOTEMPORAL_DETAIL")
TARGETS = ("FULL_DIRECT", *BANDS)
ALIGNMENTS = ("ALIGNED", "SHUFFLED")
ARMS = ("ZERO", *(f"{target}_{alignment}" for target in TARGETS for alignment in ALIGNMENTS))
DOSES = (32, 64, 128, 256)

FIT_RANGE = (128, 384)
CAL_RANGE = (384, 416)
DEV_RANGE = (416, 480)
RESERVE_RANGE = (480, 511)
STRUCTURAL_EXCLUDED_RANGE = (511, 512)
EXCLUDED_RANGE = (0, 128)
FIT_NOISE_SEEDS = (20260910, 20260911)
CAL_NOISE_SEEDS = (20260912, 20260913)
DEV_NOISE_SEEDS = (20260914, 20260915, 20260916, 20260917)
PROJECTION_SEED = 20260918
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_REPLICATES = 10_000
CAPACITY_RUNGS = (64, 256, 1024)
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TOKEN_SAMPLE_COUNT = 256

LATENT_SHAPE = (16, 4, 24, 120)
HISTORY_LATENT_FRAMES = 2
VIEW_COUNT = 3
VIEW_WIDTH = 40
HAAR_MAX_ABS_TOLERANCE = 2e-6
HAAR_RELATIVE_ENERGY_TOLERANCE = 1e-12
HAAR_ORTHOGONALITY_TOLERANCE = 2e-6
LEAKAGE_TOLERANCE = 2e-6

ERROR_METRICS = (
    "corrected_velocity_mse",
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
BAND_METRICS = (
    "target_r2",
    "target_cosine",
    "target_energy",
    "prediction_energy",
    "preprojection_offband_relative_energy",
    "postprojection_offband_relative_energy",
)


class HaarError(ladder.LadderError):
    """A prospective, decomposition, access, or evidence contract changed."""


def endpoint_name(dose: int, target: str, alignment: str) -> str:
    if dose not in DOSES or target not in TARGETS or alignment not in ALIGNMENTS:
        raise HaarError("invalid endpoint coordinates")
    return f"D{dose:03d}_{target}_{alignment}"


ENDPOINTS = (
    "VPM_OFF",
    "ZERO",
    *(endpoint_name(dose, target, alignment) for dose in DOSES for target in TARGETS for alignment in ALIGNMENTS),
)


def _configure_shared_helpers() -> None:
    # Imported ridge helpers read these globals at call time.
    ladder.ARMS = ARMS
    ladder.CAPACITY_RUNGS = CAPACITY_RUNGS
    ladder.RIDGE_LAMBDAS = RIDGE_LAMBDAS
    ladder.PROJECTION_SEED = PROJECTION_SEED
    ladder.TOKEN_SAMPLE_COUNT = TOKEN_SAMPLE_COUNT


def _spatial_lowpass_view(value: Any) -> Any:
    if value.ndim != 5 or value.shape[-2] % 2 or value.shape[-1] % 2:
        raise HaarError("per-view Haar tensor must be [B,C,T,even-H,even-W]")
    batch, channels, frames, height, width = value.shape
    blocks = value.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
    low = blocks.mean(dim=(4, 6), keepdim=True)
    return low.expand_as(blocks).reshape_as(value)


def _temporal_lowpass(value: Any) -> Any:
    if value.ndim != 5 or value.shape[2] % 2:
        raise HaarError("temporal Haar tensor must have an even frame count")
    batch, channels, frames, height, width = value.shape
    pairs = value.reshape(batch, channels, frames // 2, 2, height, width)
    low = pairs.mean(dim=3, keepdim=True)
    return low.expand_as(pairs).reshape_as(value)


def haar_decompose(value: Any, *, history_frames: int) -> dict[str, Any]:
    """Return three orthogonal full-shape projections, computed per view."""
    import torch

    if value.ndim != 5 or tuple(value.shape[1:]) != LATENT_SHAPE:
        raise HaarError(
            f"pinned VPM latent must be [B,{LATENT_SHAPE}], got {tuple(value.shape)}"
        )
    if history_frames != HISTORY_LATENT_FRAMES:
        raise HaarError("pinned VPM history boundary differs")
    future = value.float()[:, :, history_frames:]
    if future.shape[2] != 2 or future.shape[-1] != VIEW_COUNT * VIEW_WIDTH:
        raise HaarError("pinned future/view geometry differs")
    coarse_views = []
    change_views = []
    detail_views = []
    for view in range(VIEW_COUNT):
        left = view * VIEW_WIDTH
        right = left + VIEW_WIDTH
        current = future[..., left:right]
        spatial = _spatial_lowpass_view(current)
        coarse = _temporal_lowpass(spatial)
        coarse_views.append(coarse)
        change_views.append(spatial - coarse)
        detail_views.append(current - spatial)
    future_bands = {
        "ST_COARSE": torch.cat(coarse_views, dim=-1),
        "TEMPORAL_CHANGE": torch.cat(change_views, dim=-1),
        "SPATIOTEMPORAL_DETAIL": torch.cat(detail_views, dim=-1),
    }
    result = {}
    for band, projected in future_bands.items():
        full = torch.zeros_like(value, dtype=torch.float32)
        full[:, :, history_frames:] = projected
        result[band] = full
    return result


def _future_only(value: Any, history_frames: int) -> Any:
    import torch

    result = torch.zeros_like(value, dtype=torch.float32)
    result[:, :, history_frames:] = value.float()[:, :, history_frames:]
    return result


def haar_contract(value: Any, bands: Mapping[str, Any], *, history_frames: int) -> dict[str, Any]:
    import torch

    if set(bands) != set(BANDS):
        raise HaarError("Haar band inventory differs")
    target = _future_only(value, history_frames)
    reconstruction = sum((bands[name] for name in BANDS), torch.zeros_like(target))
    error = reconstruction - target
    energy = target.square().sum().clamp_min(1e-30)
    max_abs = float(error.abs().max().detach().cpu())
    relative_energy = float((error.square().sum() / energy).detach().cpu())
    pairs = {}
    for first_index, first in enumerate(BANDS):
        for second in BANDS[first_index + 1 :]:
            pairs[f"{first}__{second}"] = float(
                ((bands[first] * bands[second]).sum().abs() / energy).detach().cpu()
            )
    fractions = {
        name: float((bands[name].square().sum() / energy).detach().cpu())
        for name in BANDS
    }
    history_nonzero = sum(
        int(torch.count_nonzero(bands[name][:, :, :history_frames]).detach().cpu())
        for name in BANDS
    )
    if (
        max_abs > HAAR_MAX_ABS_TOLERANCE
        or relative_energy > HAAR_RELATIVE_ENERGY_TOLERANCE
        or any(value > HAAR_ORTHOGONALITY_TOLERANCE for value in pairs.values())
        or history_nonzero != 0
        or not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=5e-6)
    ):
        raise HaarError("Haar reconstruction/orthogonality contract failed")
    return {
        "max_abs_reconstruction_error": max_abs,
        "relative_reconstruction_energy": relative_energy,
        "normalized_pairwise_inner_products": pairs,
        "band_energy_fractions": fractions,
        "history_nonzero": history_nonzero,
    }


def _new_haar_summary() -> dict[str, Any]:
    return {
        "checked_batches": 0,
        "max_abs_reconstruction_error": 0.0,
        "max_relative_reconstruction_energy": 0.0,
        "max_abs_normalized_pairwise_inner_product": 0.0,
        "mean_band_energy_fraction_sums": {name: 0.0 for name in BANDS},
    }


def _update_haar_summary(summary: dict[str, Any], receipt: Mapping[str, Any]) -> None:
    summary["checked_batches"] += 1
    summary["max_abs_reconstruction_error"] = max(
        summary["max_abs_reconstruction_error"],
        float(receipt["max_abs_reconstruction_error"]),
    )
    summary["max_relative_reconstruction_energy"] = max(
        summary["max_relative_reconstruction_energy"],
        float(receipt["relative_reconstruction_energy"]),
    )
    summary["max_abs_normalized_pairwise_inner_product"] = max(
        summary["max_abs_normalized_pairwise_inner_product"],
        *(float(value) for value in receipt["normalized_pairwise_inner_products"].values()),
    )
    for band in BANDS:
        summary["mean_band_energy_fraction_sums"][band] += float(
            receipt["band_energy_fractions"][band]
        )


def _finalize_haar_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    count = int(summary["checked_batches"])
    if count <= 0:
        raise HaarError("no Haar batches were checked")
    return {
        **{key: value for key, value in summary.items() if key != "mean_band_energy_fraction_sums"},
        "mean_band_energy_fraction": {
            name: float(value) / count
            for name, value in summary["mean_band_energy_fraction_sums"].items()
        },
        "passed": (
            float(summary["max_abs_reconstruction_error"]) <= HAAR_MAX_ABS_TOLERANCE
            and float(summary["max_relative_reconstruction_energy"]) <= HAAR_RELATIVE_ENERGY_TOLERANCE
            and float(summary["max_abs_normalized_pairwise_inner_product"]) <= HAAR_ORTHOGONALITY_TOLERANCE
        ),
    }


def _synthetic_haar_contract() -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260918)
    value = torch.randn((2, *LATENT_SHAPE), generator=generator, dtype=torch.float32)
    bands = haar_decompose(value, history_frames=HISTORY_LATENT_FRAMES)
    receipt = haar_contract(value, bands, history_frames=HISTORY_LATENT_FRAMES)
    # View isolation: changing view 0 cannot change either other view.
    mutated = value.clone()
    mutated[..., :VIEW_WIDTH] += 7.0
    changed = haar_decompose(mutated, history_frames=HISTORY_LATENT_FRAMES)
    for band in BANDS:
        if not torch.equal(bands[band][..., VIEW_WIDTH:], changed[band][..., VIEW_WIDTH:]):
            raise HaarError("Haar transform crosses a camera-view boundary")
    return {
        **receipt,
        "latent_shape": list(LATENT_SHAPE),
        "future_shape_per_view": [LATENT_SHAPE[0], 2, LATENT_SHAPE[2], VIEW_WIDTH],
        "view_isolation_bit_exact": True,
    }


def _validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 512:
        raise HaarError("canonical train manifest must contain 512 rows")
    episodes: set[str] = set()
    clips: set[str] = set()
    for index, row in enumerate(rows):
        episode = row.get("episode_dir")
        clip = row.get("clip_id")
        if (
            row.get("split") != "train"
            or int(row.get("auxiliary_index", -1)) != index
            or not isinstance(episode, str)
            or not isinstance(clip, str)
            or episode in episodes
            or clip in clips
        ):
            raise HaarError(f"canonical manifest identity differs at row {index}")
        episodes.add(episode)
        clips.add(clip)
    ranges = {
        "excluded_history": list(EXCLUDED_RANGE),
        "nested_fit": list(FIT_RANGE),
        "calibration": list(CAL_RANGE),
        "prior_inspected_exploratory_development": list(DEV_RANGE),
        "forbidden_fresh_reserve": list(RESERVE_RANGE),
        "excluded_constructor_probe": list(STRUCTURAL_EXCLUDED_RANGE),
    }
    covered = [index for start, stop in ranges.values() for index in range(start, stop)]
    if sorted(covered) != list(range(512)) or len(set(covered)) != 512:
        raise HaarError("registered row ranges are not an exact partition")
    seed_sets = [set(FIT_NOISE_SEEDS), set(CAL_NOISE_SEEDS), set(DEV_NOISE_SEEDS)]
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise HaarError("phase noise seeds overlap")
    prior_seeds = set(range(20260820, 20260842))
    if set().union(*seed_sets, {PROJECTION_SEED, BOOTSTRAP_SEED}) & prior_seeds:
        raise HaarError("exploratory seeds overlap prior ladder/frontier seeds")
    return {
        "ranges": ranges,
        "fit_doses": list(DOSES),
        "fit_noise_seeds": list(FIT_NOISE_SEEDS),
        "calibration_noise_seeds": list(CAL_NOISE_SEEDS),
        "development_noise_seeds": list(DEV_NOISE_SEEDS),
        "all_episode_and_clip_identities_unique": True,
    }


def _validated_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    return frontier._validated_record(record, label)


def _validated_cache_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    return frontier._validated_cache_record(record, label)


def _prepare_registration(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    repo = ladder.canonical_directory(args.repo_root, "repository")
    observed_commit = ladder.git_output(repo, "rev-parse", "HEAD")
    if observed_commit != args.expected_source_commit or len(observed_commit) != 40:
        raise HaarError("source commit differs")
    if ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise HaarError("source repository must be clean")
    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise HaarError("fresh output already exists")

    parent_path = ladder.canonical_file(args.parent_registration, "parent ladder registration")
    runtime_path = ladder.canonical_file(args.runtime_record, "B200 runtime record")
    runtime = ladder.read_json(runtime_path)
    devices = runtime.get("gpus", {}).get("devices", [])
    if (
        runtime.get("gpus", {}).get("count") != 1
        or len(devices) != 1
        or "B200" not in str(devices[0].get("name", "")).upper()
        or devices[0].get("capability") != [10, 0]
    ):
        raise HaarError("runtime receipt is not a one-B200 verification")
    parent = ladder.read_json(parent_path)
    if parent.get("schema") != ladder.SCHEMA or not ladder.identity_valid(parent) or "VPM" not in parent.get("lineage", {}):
        raise HaarError("parent ladder registration is invalid")
    if args.vpm_snapshot_sha256 != parent["inputs"]["vpm_snapshot"].get("sha256"):
        raise HaarError("VPM snapshot digest is not parent-bound")
    required_inputs = (
        "train_manifest",
        "train_cache_metadata",
        "vpm_resolved_config",
        "vpm_snapshot",
        "vpm_arm_manifest",
        "vpm_stage_manifest",
        "vpm_stage_outcome",
    )
    inputs = {key: _validated_record(parent["inputs"][key], key) for key in required_inputs}
    manifest_rows = ladder.read_jsonl(Path(inputs["train_manifest"]["path"]))
    partitions = _validate_manifest(manifest_rows)
    arrays = {
        key: _validated_cache_record(record, f"cache {key}")
        for key, record in parent["cache_arrays"].items()
    }
    if set(arrays) != {"target", "rgb", "actions"}:
        raise HaarError("parent cache array inventory differs")
    vpm_spec = parent["lineage"]["VPM"].get("arm_spec", {})
    if vpm_spec.get("parameter_matched_control") is not True or vpm_spec.get("condition_mode") != "off":
        raise HaarError("registered parent is not the VPM frontier")

    output.mkdir(parents=True, mode=0o700)
    output = output.resolve(strict=True)
    registration = ladder.identity_payload(
        {
            "schema": SCHEMA,
            "created_at_utc": ladder._now(),
            "status": "registered_before_fit_or_prior_dev_outcome_open",
            "repo_root": str(repo),
            "source_commit": observed_commit,
            "output_dir": str(output),
            "parent_registration": ladder.file_record(parent_path),
            "parent_registration_identity_sha256": parent["identity_sha256"],
            "runtime_verification": ladder.file_record(runtime_path),
            "inputs": inputs,
            "cache_arrays": arrays,
            "lineage": {"VPM": parent["lineage"]["VPM"]},
            "partitions": partitions,
            "frozen_design": {
                "bands": list(BANDS),
                "haar_grouping": {
                    "ST_COARSE": "T S r = LLL",
                    "TEMPORAL_CHANGE": "(I-T) S r = HLL",
                    "SPATIOTEMPORAL_DETAIL": "(I-S)r = remaining six 3-D Haar details",
                },
                "arms": list(ARMS),
                "doses": list(DOSES),
                "capacity_rungs": list(CAPACITY_RUNGS),
                "ridge_lambdas": list(RIDGE_LAMBDAS),
                "projection_seed": PROJECTION_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "token_sample_count_per_clip_noise": TOKEN_SAMPLE_COUNT,
                "selection_target": "full-dose FULL_DIRECT_ALIGNED calibration corrected velocity MSE",
                "tie_break": "smaller capacity, then larger ridge penalty",
                "synthetic_haar_contract": _synthetic_haar_contract(),
                "thresholds": {
                    "max_abs_reconstruction_error": HAAR_MAX_ABS_TOLERANCE,
                    "relative_reconstruction_energy": HAAR_RELATIVE_ENERGY_TOLERANCE,
                    "normalized_pairwise_inner_product": HAAR_ORTHOGONALITY_TOLERANCE,
                    "postprojection_offband_relative_energy": LEAKAGE_TOLERANCE,
                },
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
                    "fit": FIT_RANGE,
                    "calibration": CAL_RANGE,
                    "exploratory_development": DEV_RANGE,
                }.items()
            },
            "access_contract": {
                "vjepa_target_array_allowed": False,
                "teacher_calls": 0,
                "future_validity_enabled": False,
                "future_validity_max_retries": 0,
                "fresh_reserve_480_510_opened": False,
                "constructor_probe_511_opened": False,
                "validation_opened": False,
                "protected_test_opened": False,
                "development_opened_during_registration": False,
                "wandb_enabled": False,
            },
            "claim_boundary": (
                "post-selection exploratory reuse of prior-inspected ABC-train rows "
                "416--479; one frozen VPM parent and linear token-local heads; no "
                "reserve, validation, protected-test, or paper claim"
            ),
        }
    )
    ladder.exclusive_json(output / "registration.json", registration)
    return output, registration


def _load_registration(output: Path) -> dict[str, Any]:
    value = ladder.read_json(output / "registration.json")
    if value.get("schema") != SCHEMA or not ladder.identity_valid(value):
        raise HaarError("registration identity/schema differs")
    if Path(value["output_dir"]).resolve(strict=True) != output:
        raise HaarError("registration output path differs")
    repo = ladder.canonical_directory(value["repo_root"], "registered repository")
    if (
        ladder.git_output(repo, "rev-parse", "HEAD") != value["source_commit"]
        or ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise HaarError("registered source state changed")
    for key, label in (("runtime_verification", "runtime receipt"), ("parent_registration", "parent registration")):
        record = value.get(key, {})
        observed = _validated_record(record, label)
        if observed != {name: record[name] for name in ("path", "bytes", "sha256")}:
            raise HaarError(f"registered {label} changed")
    return value


def _resolve_input(registration: Mapping[str, Any], key: str) -> Path:
    return Path(_validated_record(registration["inputs"][key], key)["path"])


class _IndexAuditedDataset:
    def __init__(self, dataset: Any, allowed: Sequence[int]) -> None:
        self.dataset = dataset
        self.allowed = frozenset(int(index) for index in allowed)
        self.access_counts = {index: 0 for index in self.allowed}
        if not self.allowed:
            raise HaarError("dataset phase has no allowed rows")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        index = int(index)
        if index not in self.allowed:
            raise HaarError(f"dataset phase attempted forbidden row {index}")
        self.access_counts[index] += 1
        return self.dataset[index]

    def assert_exact_accesses(self, expected_per_row: int) -> None:
        if any(count != expected_per_row for count in self.access_counts.values()):
            raise HaarError(f"dataset row-access inventory differs: {self.access_counts}")


def _phase_dataset(config: Any, registration: Mapping[str, Any], index_range: tuple[int, int]) -> _IndexAuditedDataset:
    start, stop = index_range
    dataset = ladder._no_auxiliary_dataset(
        config,
        registration,
        validation_sample_indices=(start, stop - 1),
    )
    return _IndexAuditedDataset(dataset, range(start, stop))


def _future_rows(
    projected: Any,
    target_tokens: Mapping[str, Any],
    positions: Any,
    indexes: Sequence[int],
    noise_seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    feature_rows = []
    target_rows = {arm: [] for arm in ARMS}
    for local, clip_index in enumerate(indexes):
        selected = ladder.deterministic_token_subsample(
            positions,
            clip_index=int(clip_index),
            noise_seed=int(noise_seed),
            count=TOKEN_SAMPLE_COUNT,
        )
        feature_rows.append(projected[local, selected])
        for arm in ARMS:
            target_rows[arm].append(target_tokens[arm][local, selected])
    return torch.cat(feature_rows), {
        arm: torch.cat(values) for arm, values in target_rows.items()
    }


def _target_tokens(direct: Any, *, patch_size: Sequence[int], history_frames: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    aligned_bands = haar_decompose(direct, history_frames=history_frames)
    aligned_receipt = haar_contract(direct, aligned_bands, history_frames=history_frames)
    shuffled = torch.roll(direct, shifts=-1, dims=0)
    shuffled_bands = haar_decompose(shuffled, history_frames=history_frames)
    shuffled_receipt = haar_contract(shuffled, shuffled_bands, history_frames=history_frames)
    full = _future_only(direct, history_frames)
    shuffled_full = _future_only(shuffled, history_frames)
    tensors = {
        "FULL_DIRECT_ALIGNED": full,
        "FULL_DIRECT_SHUFFLED": shuffled_full,
        **{f"{band}_ALIGNED": value for band, value in aligned_bands.items()},
        **{f"{band}_SHUFFLED": value for band, value in shuffled_bands.items()},
    }
    tokenized = {arm: ladder.patchify_video(value, patch_size) for arm, value in tensors.items()}
    tokenized["ZERO"] = torch.zeros_like(tokenized["FULL_DIRECT_ALIGNED"])
    return tokenized, {"aligned": aligned_receipt, "shuffled": shuffled_receipt}


def _fit_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise HaarError("CUDA is required")
    _configure_shared_helpers()
    output, registration = _prepare_registration(args)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    model, config = ladder._load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    projection_cpu = ladder.make_projection(
        int(model.forward_model.transformer.dim), max(CAPACITY_RUNGS), seed=PROJECTION_SEED
    )
    projection = projection_cpu.to(device=device)
    stats_by_dose: dict[int, dict[str, Any]] = {}
    grid = patch_size = None
    patch_dim = None
    fit_calls = 0
    fit_haar = _new_haar_summary()
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, FIT_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in FIT_NOISE_SEEDS:
                for start in range(FIT_RANGE[0], FIT_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = ladder._batch_samples(dataset, indexes, device)
                    prepared = ladder._model_inputs(
                        model, batch, noise_seed=noise_seed, nfe=1, need_clean_auxiliary=False
                    )
                    off_velocity, hidden = ladder._off_prediction_and_hidden(model, prepared)
                    fit_calls += 1
                    observed = ladder._grid_contract(model, off_velocity, hidden)
                    if grid is None:
                        grid, patch_size, patch_dim = observed
                        stats_by_dose = {
                            dose: ladder._new_sufficient_statistics(max(CAPACITY_RUNGS), patch_dim, device)
                            for dose in DOSES
                        }
                    elif observed != (grid, patch_size, patch_dim):
                        raise HaarError("Wan grid changed during fit")
                    direct = (prepared["video_target"] - off_velocity).float()
                    targets, receipts = _target_tokens(
                        direct,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                    )
                    _update_haar_summary(fit_haar, receipts["aligned"])
                    _update_haar_summary(fit_haar, receipts["shuffled"])
                    projected = hidden.float() @ projection
                    positions = ladder.future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                        device=device,
                    )
                    x, y = _future_rows(projected, targets, positions, indexes, noise_seed)
                    for dose in DOSES:
                        if start < FIT_RANGE[0] + dose:
                            ladder.update_sufficient_statistics(stats_by_dose[dose], x, y)
                    del batch, prepared, hidden, projected, direct, targets, x, y
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(len(FIT_NOISE_SEEDS))
    if opened != expected_arrays:
        raise HaarError(f"fit input graph differs: {sorted(opened)}")
    if grid is None or patch_size is None or patch_dim is None:
        raise HaarError("fit produced no sufficient statistics")
    for dose, stats in stats_by_dose.items():
        expected = dose * len(FIT_NOISE_SEEDS) * TOKEN_SAMPLE_COUNT
        if int(stats["count"]) != expected:
            raise HaarError(f"fit token count differs at dose {dose}")
    del dataset
    gc.collect()

    cal_x: list[Any] = []
    cal_y: list[Any] = []
    cal_calls = 0
    cal_haar = _new_haar_summary()
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, CAL_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in CAL_NOISE_SEEDS:
                for start in range(CAL_RANGE[0], CAL_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = ladder._batch_samples(dataset, indexes, device)
                    prepared = ladder._model_inputs(
                        model, batch, noise_seed=noise_seed, nfe=1, need_clean_auxiliary=False
                    )
                    off_velocity, hidden = ladder._off_prediction_and_hidden(model, prepared)
                    cal_calls += 1
                    direct = (prepared["video_target"] - off_velocity).float()
                    bands = haar_decompose(direct, history_frames=prepared["history_frames"])
                    _update_haar_summary(
                        cal_haar,
                        haar_contract(direct, bands, history_frames=prepared["history_frames"]),
                    )
                    projected = hidden.float() @ projection
                    target = ladder.patchify_video(
                        _future_only(direct, prepared["history_frames"]), patch_size
                    )
                    positions = ladder.future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                        device=device,
                    )
                    for local, clip_index in enumerate(indexes):
                        selected = ladder.deterministic_token_subsample(
                            positions,
                            clip_index=clip_index,
                            noise_seed=noise_seed,
                            count=TOKEN_SAMPLE_COUNT,
                        )
                        cal_x.append(projected[local, selected].cpu().to(torch.float16))
                        cal_y.append(target[local, selected].cpu().to(torch.float16))
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(len(CAL_NOISE_SEEDS))
    if opened != expected_arrays:
        raise HaarError(f"calibration input graph differs: {sorted(opened)}")
    features = torch.cat(cal_x).to(device=device, dtype=torch.float32)
    targets = torch.cat(cal_y).to(device=device, dtype=torch.float32)
    expected_cal = (CAL_RANGE[1] - CAL_RANGE[0]) * len(CAL_NOISE_SEEDS) * TOKEN_SAMPLE_COUNT
    if features.shape[0] != expected_cal:
        raise HaarError("calibration token count differs")

    candidates = []
    for capacity in CAPACITY_RUNGS:
        for ridge_lambda in RIDGE_LAMBDAS:
            head = ladder.fit_ridge_head(
                stats_by_dose[256],
                arm="FULL_DIRECT_ALIGNED",
                capacity=capacity,
                ridge_lambda=ridge_lambda,
            )
            prediction = ladder.predict_ridge_head(features, head)
            candidates.append(
                {
                    "capacity": capacity,
                    "ridge_lambda": ridge_lambda,
                    "corrected_velocity_mse": float((prediction - targets).square().mean().detach().cpu()),
                    "parameter_count": head["parameter_count"],
                }
            )
    selected = min(
        candidates,
        key=lambda row: (row["corrected_velocity_mse"], row["capacity"], -row["ridge_lambda"]),
    )
    heads = {
        str(dose): {
            arm: ladder.fit_ridge_head(
                stats_by_dose[dose],
                arm=arm,
                capacity=int(selected["capacity"]),
                ridge_lambda=float(selected["ridge_lambda"]),
            )
            for arm in ARMS
        }
        for dose in DOSES
    }
    counts = {
        int(head["parameter_count"])
        for dose_heads in heads.values()
        for head in dose_heads.values()
    }
    if len(counts) != 1:
        raise HaarError("all dose/target heads must have equal capacity")
    if any(
        int(torch.count_nonzero(heads[str(dose)]["ZERO"][name])) != 0
        for dose in DOSES
        for name in ("weight", "mean_y")
    ):
        raise HaarError("ZERO head is not exactly zero")
    weights = {
        "schema": WEIGHT_SCHEMA,
        "source_commit": registration["source_commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "projection_seed": PROJECTION_SEED,
        "projection": projection_cpu,
        "grid": list(grid),
        "patch_size": list(patch_size),
        "patch_dim": int(patch_dim),
        "heads": heads,
        "selected_capacity": int(selected["capacity"]),
        "selected_ridge_lambda": float(selected["ridge_lambda"]),
    }
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    ladder.exclusive_bytes(output / "adapter_weights.pt", buffer.getvalue())
    fit = ladder.identity_payload(
        {
            "schema": FIT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "weights": ladder.file_record(output / "adapter_weights.pt"),
            "optimization": {
                "clip_indices": list(FIT_RANGE),
                "fit_doses": list(DOSES),
                "dataset_array_validation_rows": [FIT_RANGE[0], FIT_RANGE[1] - 1],
                "exact_dataset_accesses_per_row": len(FIT_NOISE_SEEDS),
                "noise_seeds": list(FIT_NOISE_SEEDS),
                "sampled_future_tokens_by_dose": {
                    str(dose): int(stats_by_dose[dose]["count"]) for dose in DOSES
                },
                "off_wan_invocations": fit_calls,
                "teacher_calls": 0,
                "vjepa_target_array_opened": False,
                "haar_contract": _finalize_haar_summary(fit_haar),
            },
            "calibration": {
                "clip_indices": list(CAL_RANGE),
                "dataset_array_validation_rows": [CAL_RANGE[0], CAL_RANGE[1] - 1],
                "exact_dataset_accesses_per_row": len(CAL_NOISE_SEEDS),
                "noise_seeds": list(CAL_NOISE_SEEDS),
                "sampled_future_tokens": int(features.shape[0]),
                "off_wan_invocations": cal_calls,
                "teacher_calls": 0,
                "vjepa_target_array_opened": False,
                "candidate_results": candidates,
                "haar_contract": _finalize_haar_summary(cal_haar),
            },
            "selected": selected,
            "head_parameter_count": counts.pop(),
            "all_heads_equal_parameter_count": True,
            "zero_head_exact_zero": True,
            "prior_dev_opened": False,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "fit.json", fit)
    print(json.dumps({"fit_identity_sha256": fit["identity_sha256"], "selected": selected}, sort_keys=True))
    return 0


def _load_weights(
    output: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    fit = ladder.read_json(output / "fit.json")
    if fit.get("schema") != FIT_SCHEMA or not ladder.identity_valid(fit):
        raise HaarError("fit identity/schema differs")
    if fit.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise HaarError("fit registration link differs")
    record = fit.get("weights", {})
    path = ladder.canonical_file(record.get("path", ""), "adapter weights")
    if (
        path.parent != output
        or path.stat().st_size != int(record.get("bytes", -1))
        or ladder.sha256_file(path) != record.get("sha256")
    ):
        raise HaarError("adapter weight artifact differs")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    selected = fit.get("selected", {})
    capacity = int(selected.get("capacity", -1))
    ridge_lambda = float(selected.get("ridge_lambda", float("nan")))
    heads = weights.get("heads", {})
    if (
        weights.get("schema") != WEIGHT_SCHEMA
        or weights.get("registration_identity_sha256") != registration["identity_sha256"]
        or set(heads) != {str(dose) for dose in DOSES}
        or any(set(dose_heads) != set(ARMS) for dose_heads in heads.values())
        or weights.get("projection_seed") != PROJECTION_SEED
        or int(weights.get("selected_capacity", -1)) != capacity
        or float(weights.get("selected_ridge_lambda", float("nan"))) != ridge_lambda
        or capacity not in CAPACITY_RUNGS
        or ridge_lambda not in RIDGE_LAMBDAS
        or fit.get("optimization", {}).get("clip_indices") != list(FIT_RANGE)
        or fit.get("calibration", {}).get("clip_indices") != list(CAL_RANGE)
        or fit.get("zero_head_exact_zero") is not True
        or fit.get("prior_dev_opened") is not False
        or fit.get("fresh_reserve_480_510_opened") is not False
        or fit.get("constructor_probe_511_opened") is not False
        or fit.get("validation_opened") is not False
        or fit.get("protected_test_opened") is not False
        or fit.get("optimization", {}).get("haar_contract", {}).get("passed") is not True
        or fit.get("calibration", {}).get("haar_contract", {}).get("passed") is not True
    ):
        raise HaarError("sealed fit/weight contract differs")
    parameter_counts = set()
    for dose_heads in heads.values():
        for arm, head in dose_heads.items():
            if (
                head.get("arm") != arm
                or int(head.get("capacity", -1)) != capacity
                or float(head.get("ridge_lambda", float("nan"))) != ridge_lambda
            ):
                raise HaarError("sealed head metadata differs")
            parameter_counts.add(int(head.get("parameter_count", -1)))
    if len(parameter_counts) != 1 or parameter_counts != {int(fit["head_parameter_count"])}:
        raise HaarError("sealed heads are not parameter matched")
    return fit, dict(weights)


def _event_ms(start: Any, end: Any) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or np.any(array < 0):
        raise HaarError("latency trace is empty, negative, or non-finite")
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _prediction_for_endpoint(
    *,
    endpoint: str,
    hidden: Any,
    projection: Any,
    weights: Mapping[str, Any],
    grid: Sequence[int],
    patch_size: Sequence[int],
    channels: int,
    history_frames: int,
) -> tuple[Any, Any]:
    """Return the deployable correction and unprojected head output."""

    if endpoint == "ZERO":
        dose = DOSES[-1]
        arm = "ZERO"
    else:
        prefix, arm = endpoint.split("_", 1)
        dose = int(prefix[1:])
    projected_hidden = hidden.float() @ projection
    tokens = ladder.predict_ridge_head(projected_hidden, weights["heads"][str(dose)][arm]).clone()
    positions = ladder.future_token_positions(
        grid=grid,
        patch_size=patch_size,
        history_frames=history_frames,
        device=hidden.device,
    )
    tokens[:, : int(positions[0])] = 0
    raw = ladder.unpatchify_tokens(
        tokens,
        grid=grid,
        patch_size=patch_size,
        channels=channels,
    ).float()
    raw[:, :, :history_frames] = 0
    target = arm.rsplit("_", 1)[0] if arm != "ZERO" else "ZERO"
    if target in BANDS:
        correction = haar_decompose(raw, history_frames=history_frames)[target]
    else:
        correction = raw
    return correction, raw


def _projection_leakage(
    raw: Any,
    correction: Any,
    *,
    endpoint: str,
    history_frames: int,
) -> tuple[Any, Any]:
    target = next((band for band in BANDS if band in endpoint), None)
    if target is None:
        projected_twice = correction
    else:
        projected_twice = haar_decompose(correction, history_frames=history_frames)[target]
    reduce = tuple(range(1, raw.ndim))
    raw_energy = raw.square().sum(dim=reduce).clamp_min(1e-30)
    projected_energy = correction.square().sum(dim=reduce).clamp_min(1e-30)
    pre_leakage = (raw - correction).square().sum(dim=reduce) / raw_energy
    post_leakage = (correction - projected_twice).square().sum(dim=reduce) / projected_energy
    return pre_leakage, post_leakage


def _per_sample_band_metrics(
    prediction: Any,
    target: Any,
    *,
    history_frames: int,
    pre_leakage: Any,
    post_leakage: Any,
) -> dict[str, Any]:
    import torch

    prediction = prediction.float()[:, :, history_frames:]
    target = target.float()[:, :, history_frames:]
    reduce = tuple(range(1, prediction.ndim))
    error = (prediction - target).square().sum(dim=reduce)
    energy = target.square().sum(dim=reduce).clamp_min(1e-30)
    count = math.prod(target.shape[1:])
    return {
        "target_r2": 1.0 - error / energy,
        "target_cosine": torch.nn.functional.cosine_similarity(
            prediction.flatten(1), target.flatten(1), dim=1, eps=1e-12
        ),
        "target_energy": energy / count,
        "prediction_energy": prediction.square().sum(dim=reduce) / count,
        "preprojection_offband_relative_energy": pre_leakage,
        "postprojection_offband_relative_energy": post_leakage,
    }


def _evaluate_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise HaarError("CUDA is required")
    _configure_shared_helpers()
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, weights = _load_weights(output, registration)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    model, config = ladder._load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    projection = weights["projection"].to(device=device)
    grid = tuple(int(value) for value in weights["grid"])
    patch_size = tuple(int(value) for value in weights["patch_size"])
    manifest = ladder.read_jsonl(_resolve_input(registration, "train_manifest"))
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    adapter_timings = {endpoint: [] for endpoint in ENDPOINTS if endpoint != "VPM_OFF"}
    integration_timings = {endpoint: [] for endpoint in ENDPOINTS}
    decoder_timings = {endpoint: [] for endpoint in ENDPOINTS}
    preparation_timings: list[float] = []
    wan_timings: list[float] = []
    full_primary_timings: list[float] = []
    wan_invocations = 0
    dev_haar = _new_haar_summary()
    primary = endpoint_name(256, "FULL_DIRECT", "ALIGNED")

    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, DEV_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in DEV_NOISE_SEEDS:
                for start in range(DEV_RANGE[0], DEV_RANGE[1], 2):
                    indexes = tuple(range(start, min(start + 2, DEV_RANGE[1])))
                    batch = ladder._batch_samples(dataset, indexes, device)
                    torch.cuda.synchronize()
                    full_started = time.perf_counter()
                    preparation_started = time.perf_counter()
                    try:
                        prepared = frontier._serving_inputs(model, batch, noise_seed=noise_seed)
                    except ladder.LadderError as exc:
                        raise HaarError(str(exc)) from exc
                    torch.cuda.synchronize()
                    preparation_timings.append(1000.0 * (time.perf_counter() - preparation_started))
                    before = torch.cuda.Event(enable_timing=True)
                    after = torch.cuda.Event(enable_timing=True)
                    before.record()
                    off_velocity, hidden = ladder._off_prediction_and_hidden(model, prepared)
                    after.record()
                    wan_timings.append(_event_ms(before, after))
                    wan_invocations += 1
                    observed_grid, observed_patch, _ = ladder._grid_contract(model, off_velocity, hidden)
                    if observed_grid != grid or observed_patch != patch_size:
                        raise HaarError("development Wan grid differs from fit")
                    if tuple(off_velocity.shape[1:]) != LATENT_SHAPE:
                        raise HaarError("development VPM latent shape differs")

                    residuals: dict[str, Any] = {}
                    pre_leakage: dict[str, Any] = {}
                    post_leakage: dict[str, Any] = {}
                    velocities = {"VPM_OFF": off_velocity.float()}

                    def predict_endpoint(endpoint: str, *, audit_leakage: bool = True) -> Any:
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        correction, raw_correction = _prediction_for_endpoint(
                            endpoint=endpoint,
                            hidden=hidden,
                            projection=projection,
                            weights=weights,
                            grid=grid,
                            patch_size=patch_size,
                            channels=off_velocity.shape[1],
                            history_frames=prepared["history_frames"],
                        )
                        after.record()
                        adapter_timings[endpoint].append(_event_ms(before, after))
                        residuals[endpoint] = correction
                        velocities[endpoint] = off_velocity.float() + correction
                        if audit_leakage:
                            pre, post = _projection_leakage(
                                raw_correction,
                                correction,
                                endpoint=endpoint,
                                history_frames=prepared["history_frames"],
                            )
                            pre_leakage[endpoint] = pre
                            post_leakage[endpoint] = post
                        return raw_correction

                    finals: dict[str, Any] = {}
                    decoded_by_endpoint: dict[str, Any] = {}

                    def integrate_decode(endpoint: str) -> None:
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        final = ladder._one_step_final(
                            prepared["initial_video"],
                            velocities[endpoint],
                            prepared["reference"],
                            prepared["history_frames"],
                        )
                        after.record()
                        integration_timings[endpoint].append(_event_ms(before, after))
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        decoded_full = model.rgb_tokenizer.decode_temporal(
                            final.to(batch["rgb"].dtype),
                            out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                        )
                        decoded = ladder._to_uint8_video(decoded_full[:, :, -model.num_future_frames :])
                        after.record()
                        decoder_timings[endpoint].append(_event_ms(before, after))
                        ladder._validate_decoded_horizon(
                            model_future_frames=model.num_future_frames,
                            decoded_frames=decoded_full.shape[2],
                            raw_frames=batch["rgb"].shape[1],
                            endpoint=endpoint,
                        )
                        finals[endpoint] = final
                        decoded_by_endpoint[endpoint] = decoded

                    # Materialize only the frozen primary endpoint before closing
                    # its synchronized end-to-end timer. Other control adapters
                    # must not contaminate this serving measurement.
                    primary_raw = predict_endpoint(primary, audit_leakage=False)
                    integrate_decode(primary)
                    torch.cuda.synchronize()
                    full_primary_timings.append(1000.0 * (time.perf_counter() - full_started))
                    primary_pre, primary_post = _projection_leakage(
                        primary_raw,
                        residuals[primary],
                        endpoint=primary,
                        history_frames=prepared["history_frames"],
                    )
                    pre_leakage[primary] = primary_pre
                    post_leakage[primary] = primary_post
                    for endpoint in ENDPOINTS:
                        if endpoint not in ("VPM_OFF", primary):
                            predict_endpoint(endpoint)
                    zeros = torch.zeros(
                        (len(indexes),), device=device, dtype=torch.float32
                    )
                    residuals["VPM_OFF"] = torch.zeros_like(off_velocity, dtype=torch.float32)
                    pre_leakage["VPM_OFF"] = zeros
                    post_leakage["VPM_OFF"] = zeros
                    if int(torch.count_nonzero(residuals["ZERO"]).detach().cpu()) != 0:
                        raise HaarError("ZERO correction is not exact zero")
                    for endpoint in ENDPOINTS:
                        if endpoint != primary:
                            integrate_decode(endpoint)
                    if not torch.equal(finals["VPM_OFF"], finals["ZERO"]):
                        raise HaarError("ZERO final differs bitwise from VPM_OFF")

                    # Only now may the evaluator encode and inspect clean future video.
                    video_clean = model._encode_clip(batch["rgb"]).to(batch["rgb"].dtype)
                    if video_clean.shape != prepared["initial_video"].shape:
                        raise HaarError("development clean-video grid differs")
                    video_target = prepared["initial_video"] - video_clean
                    direct_target = (video_target - off_velocity).float()
                    target_bands = haar_decompose(
                        direct_target, history_frames=prepared["history_frames"]
                    )
                    _update_haar_summary(
                        dev_haar,
                        haar_contract(
                            direct_target,
                            target_bands,
                            history_frames=prepared["history_frames"],
                        ),
                    )
                    full_target = _future_only(direct_target, prepared["history_frames"])
                    raw_video = batch["rgb"].permute(0, 2, 1, 3, 4)
                    raw_target_exact = raw_video[:, :, -model.num_future_frames :]
                    raw_history_exact = raw_video[
                        :, :, -(model.num_future_frames + 1) : -model.num_future_frames
                    ]
                    raw_target = ladder._to_uint8_video(raw_target_exact)
                    raw_history = ladder._to_uint8_video(raw_history_exact)
                    decoded_clean = model.rgb_tokenizer.decode_temporal(
                        video_clean,
                        out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                    )
                    vae_target = ladder._to_uint8_video(decoded_clean[:, :, -model.num_future_frames :])
                    vae_history = ladder._to_uint8_video(
                        decoded_clean[:, :, -(model.num_future_frames + 1) : -model.num_future_frames]
                    )

                    metrics_by_endpoint = {}
                    for endpoint in ENDPOINTS:
                        standard = ladder._per_sample_future_metrics(
                            velocity=velocities[endpoint],
                            base_velocity=off_velocity,
                            direct_residual=direct_target,
                            final=finals[endpoint],
                            clean=video_clean,
                            decoded=decoded_by_endpoint[endpoint],
                            raw_target=raw_target,
                            raw_history=raw_history,
                            vae_target=vae_target,
                            vae_history=vae_history,
                            history_frames=prepared["history_frames"],
                        )
                        if endpoint in ("VPM_OFF", "ZERO") or "FULL_DIRECT" in endpoint:
                            named_target = full_target
                        else:
                            named_target = next(
                                target_bands[band] for band in BANDS if band in endpoint
                            )
                        band_metrics = _per_sample_band_metrics(
                            residuals[endpoint],
                            named_target,
                            history_frames=prepared["history_frames"],
                            pre_leakage=pre_leakage[endpoint],
                            post_leakage=post_leakage[endpoint],
                        )
                        metrics_by_endpoint[endpoint] = {**standard, **band_metrics}

                    common_hashes = {
                        "initial_video": prepared["initial_video"],
                        "actions": batch["actions"],
                        "morphology_index": batch["morphology_index"],
                        "action_control": prepared["z_control"],
                        "auxiliary_noise": prepared["initial_auxiliary"],
                        "raw_history_input": batch["rgb"][:, : model.num_history_frames],
                        "history_reference": prepared["reference"],
                        "latent_flow_target": video_target[:, :, prepared["history_frames"] :],
                        "raw_future_target": raw_target_exact,
                        "raw_history_boundary": raw_history_exact,
                        "scoring_future_uint8": raw_target,
                        "scoring_history_uint8": raw_history,
                    }
                    for endpoint in ENDPOINTS:
                        metrics = metrics_by_endpoint[endpoint]
                        for local, clip_index in enumerate(indexes):
                            rows.append(
                                ladder.identity_payload(
                                    {
                                        "schema": ROW_SCHEMA,
                                        "endpoint": endpoint,
                                        "clip_index": clip_index,
                                        "clip_id": manifest[clip_index]["clip_id"],
                                        "episode_dir": manifest[clip_index]["episode_dir"],
                                        "noise_seed": noise_seed,
                                        "wan_calls": 1,
                                        "teacher_calls": 0,
                                        "vjepa_target_array_opened": False,
                                        "future_target_entered_correction": False,
                                        "metrics": {
                                            key: float(value[local].detach().cpu())
                                            for key, value in metrics.items()
                                        },
                                        "tensor_sha256": {
                                            **{
                                                name: ladder._safe_tensor_sha256(value[local : local + 1])
                                                for name, value in common_hashes.items()
                                            },
                                            "correction": ladder._safe_tensor_sha256(
                                                residuals[endpoint][local : local + 1]
                                            ),
                                            "final_video": ladder._safe_tensor_sha256(
                                                finals[endpoint][local : local + 1]
                                            ),
                                        },
                                        "scoring_target_constructed_after_all_endpoints": True,
                                        "post_selection_exploratory_dev_opened": True,
                                        "fresh_reserve_480_510_opened": False,
                                        "constructor_probe_511_opened": False,
                                        "validation_opened": False,
                                        "protected_test_opened": False,
                                    }
                                )
                            )
                    timing_rows.append(
                        ladder.identity_payload(
                            {
                                "schema": TIMING_SCHEMA,
                                "batch_ordinal": len(timing_rows),
                                "noise_seed": noise_seed,
                                "clip_indices": list(indexes),
                                "batch_size": len(indexes),
                                "latency_ms": {
                                    "causal_preparation": preparation_timings[-1],
                                    "wan": wan_timings[-1],
                                    "adapter": {
                                        endpoint: adapter_timings[endpoint][-1]
                                        for endpoint in adapter_timings
                                    },
                                    "euler": {
                                        endpoint: integration_timings[endpoint][-1]
                                        for endpoint in ENDPOINTS
                                    },
                                    "decoder": {
                                        endpoint: decoder_timings[endpoint][-1]
                                        for endpoint in ENDPOINTS
                                    },
                                    "full_primary_endpoint": full_primary_timings[-1],
                                },
                            }
                        )
                    )
                    del (
                        batch,
                        prepared,
                        hidden,
                        velocities,
                        residuals,
                        finals,
                        decoded_by_endpoint,
                        video_clean,
                        video_target,
                        direct_target,
                        target_bands,
                    )
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(len(DEV_NOISE_SEEDS))
    if opened != expected_arrays:
        raise HaarError(f"development input graph differs: {sorted(opened)}")
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(ENDPOINTS)
    expected_calls = ((DEV_RANGE[1] - DEV_RANGE[0] + 1) // 2) * len(DEV_NOISE_SEEDS)
    if len(rows) != expected_rows or wan_invocations != expected_calls:
        raise HaarError("development row/Wan accounting differs")
    rows_path = output / "development_rows.jsonl"
    timing_path = output / "timing_rows.jsonl"
    ladder.exclusive_jsonl(rows_path, rows)
    ladder.exclusive_jsonl(timing_path, timing_rows)
    receipt = ladder.identity_payload(
        {
            "schema": ENDPOINT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "rows": ladder.file_record(rows_path),
            "timing_rows": ladder.file_record(timing_path),
            "row_count": len(rows),
            "timing_row_count": len(timing_rows),
            "development_clip_range": list(DEV_RANGE),
            "development_noise_seeds": list(DEV_NOISE_SEEDS),
            "dataset_array_validation_rows": [DEV_RANGE[0], DEV_RANGE[1] - 1],
            "exact_dataset_accesses_per_row": len(DEV_NOISE_SEEDS),
            "shared_wan_invocations": wan_invocations,
            "wan_sample_calls": (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS),
            "teacher_calls": 0,
            "vjepa_target_array_opened": False,
            "future_target_entered_correction": False,
            "scoring_target_constructed_after_all_endpoints": True,
            "zero_equals_off_bit_exact": True,
            "haar_contract": _finalize_haar_summary(dev_haar),
            "adapter_latency_ms_per_batch": {
                endpoint: _latency_summary(values) for endpoint, values in adapter_timings.items()
            },
            "wan_latency_ms_per_batch": _latency_summary(wan_timings),
            "causal_preparation_latency_ms_per_batch": _latency_summary(preparation_timings),
            "euler_latency_ms_per_batch": {
                endpoint: _latency_summary(values) for endpoint, values in integration_timings.items()
            },
            "decoder_latency_ms_per_batch": {
                endpoint: _latency_summary(values) for endpoint, values in decoder_timings.items()
            },
            "full_primary_endpoint": primary,
            "full_primary_endpoint_latency_ms_per_batch": _latency_summary(full_primary_timings),
            "latency_scope": (
                "primary synchronized wall time from resident observed RGB/actions through "
                "causal preparation, one shared Wan call, adapter, Euler, and decode; clean "
                "future target encoding, scoring, and post-projection leakage audit excluded"
            ),
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "endpoint_complete.json", receipt)
    print(json.dumps({"endpoint_identity_sha256": receipt["identity_sha256"]}, sort_keys=True))
    return 0


def _bootstrap_relative(control: Any, candidate: Any, *, seed: int) -> dict[str, Any]:
    import numpy as np

    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    expected_shape = (DEV_RANGE[1] - DEV_RANGE[0], len(DEV_NOISE_SEEDS))
    if (
        control.shape != expected_shape
        or candidate.shape != expected_shape
        or not np.isfinite(control).all()
        or not np.isfinite(candidate).all()
        or control.mean() <= 0
    ):
        raise HaarError("paired bootstrap input differs")
    point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, control.shape[0], size=(BOOTSTRAP_REPLICATES, control.shape[0]))
    control_draw = control[indexes].mean(axis=(1, 2))
    candidate_draw = candidate[indexes].mean(axis=(1, 2))
    effects = 100.0 * (control_draw - candidate_draw) / control_draw
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": float(point),
        "paired_episode_bootstrap_95_percent": [
            float(value) for value in np.quantile(effects, (0.025, 0.975))
        ],
        "favorable_episode_fraction": float(
            np.mean(candidate.mean(axis=1) < control.mean(axis=1))
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _bootstrap_auc_difference(
    off: Any,
    band_curve: Sequence[Any],
    direct_curve: Sequence[Any],
    *,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    off = np.asarray(off, dtype=np.float64)
    bands = np.stack([np.asarray(value, dtype=np.float64) for value in band_curve])
    directs = np.stack([np.asarray(value, dtype=np.float64) for value in direct_curve])
    expected = (len(DOSES), DEV_RANGE[1] - DEV_RANGE[0], len(DEV_NOISE_SEEDS))
    if bands.shape != expected or directs.shape != expected or off.shape != expected[1:]:
        raise HaarError("learning-curve inventory differs")
    off_episode = off.mean(axis=1).clip(min=1e-30)
    band_improvement = 100.0 * (off_episode[None, :] - bands.mean(axis=2)) / off_episode[None, :]
    direct_improvement = 100.0 * (off_episode[None, :] - directs.mean(axis=2)) / off_episode[None, :]
    x = (np.log2(np.asarray(DOSES, dtype=np.float64)) - math.log2(DOSES[0])) / (
        math.log2(DOSES[-1]) - math.log2(DOSES[0])
    )
    band_auc = np.trapezoid(band_improvement, x=x, axis=0)
    direct_auc = np.trapezoid(direct_improvement, x=x, axis=0)
    differences = band_auc - direct_auc
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, differences.shape[0], size=(BOOTSTRAP_REPLICATES, differences.shape[0]))
    draws = differences[indexes].mean(axis=1)
    return {
        "band_mean_relative_improvement_auc": float(band_auc.mean()),
        "full_direct_mean_relative_improvement_auc": float(direct_auc.mean()),
        "band_minus_full_direct_auc": float(differences.mean()),
        "paired_episode_bootstrap_95_difference": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))
        ],
        "favorable_episode_fraction": float(np.mean(differences > 0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    registered_samples: Sequence[Mapping[str, Any]],
    haar_contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    registered = {
        int(sample.get("clip_index", -1)): (sample.get("clip_id"), sample.get("episode_dir"))
        for sample in registered_samples
    }
    if (
        set(registered) != set(range(*DEV_RANGE))
        or len(registered) != len(registered_samples)
        or any(not isinstance(clip, str) or not isinstance(episode, str) for clip, episode in registered.values())
    ):
        raise HaarError("registered exploratory-development identity inventory differs")
    expected = {
        (endpoint, clip, seed)
        for endpoint in ENDPOINTS
        for clip in range(*DEV_RANGE)
        for seed in DEV_NOISE_SEEDS
    }
    inventory: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    required_metrics = {*ERROR_METRICS, *BAND_METRICS}
    paired_hashes = {
        "initial_video",
        "actions",
        "morphology_index",
        "action_control",
        "auxiliary_noise",
        "raw_history_input",
        "history_reference",
        "latent_flow_target",
        "raw_future_target",
        "raw_history_boundary",
        "scoring_future_uint8",
        "scoring_history_uint8",
    }
    required_hashes = {*paired_hashes, "correction", "final_video"}
    for row in rows:
        if row.get("schema") != ROW_SCHEMA or not ladder.identity_valid(row):
            raise HaarError("development row identity/schema differs")
        key = (
            str(row.get("endpoint")),
            int(row.get("clip_index", -1)),
            int(row.get("noise_seed", -1)),
        )
        if key in inventory:
            raise HaarError(f"duplicate development row: {key}")
        if (
            key not in expected
            or (row.get("clip_id"), row.get("episode_dir")) != registered.get(key[1])
            or int(row.get("wan_calls", -1)) != 1
            or int(row.get("teacher_calls", -1)) != 0
            or row.get("vjepa_target_array_opened") is not False
            or row.get("future_target_entered_correction") is not False
            or row.get("scoring_target_constructed_after_all_endpoints") is not True
            or row.get("post_selection_exploratory_dev_opened") is not True
            or row.get("fresh_reserve_480_510_opened") is not False
            or row.get("constructor_probe_511_opened") is not False
            or row.get("validation_opened") is not False
            or row.get("protected_test_opened") is not False
        ):
            raise HaarError("development row violates identity/access/serving contract")
        metrics = row.get("metrics")
        hashes = row.get("tensor_sha256")
        if (
            not isinstance(metrics, Mapping)
            or not required_metrics.issubset(metrics)
            or any(not math.isfinite(float(metrics[name])) for name in required_metrics)
            or not isinstance(hashes, Mapping)
            or not required_hashes.issubset(hashes)
            or any(not isinstance(hashes[name], str) or len(hashes[name]) != 64 for name in required_hashes)
        ):
            raise HaarError("development row metric/hash payload differs")
        inventory[key] = row
    if set(inventory) != expected:
        raise HaarError("development row inventory differs")
    for clip in range(*DEV_RANGE):
        for seed in DEV_NOISE_SEEDS:
            for field in paired_hashes:
                if len({inventory[(endpoint, clip, seed)]["tensor_sha256"][field] for endpoint in ENDPOINTS}) != 1:
                    raise HaarError(f"paired {field} differs for {(clip, seed)}")
            off = inventory[("VPM_OFF", clip, seed)]
            zero = inventory[("ZERO", clip, seed)]
            if (
                off["tensor_sha256"]["final_video"] != zero["tensor_sha256"]["final_video"]
                or off["tensor_sha256"]["correction"] != zero["tensor_sha256"]["correction"]
                or off["metrics"] != zero["metrics"]
            ):
                raise HaarError("ZERO is not bit-exact to VPM_OFF")
    if (
        haar_contract_receipt.get("passed") is not True
        or int(haar_contract_receipt.get("checked_batches", 0))
        != ((DEV_RANGE[1] - DEV_RANGE[0] + 1) // 2) * len(DEV_NOISE_SEEDS)
        or float(haar_contract_receipt.get("max_abs_reconstruction_error", float("inf")))
        > HAAR_MAX_ABS_TOLERANCE
        or float(haar_contract_receipt.get("max_relative_reconstruction_energy", float("inf")))
        > HAAR_RELATIVE_ENERGY_TOLERANCE
        or float(haar_contract_receipt.get("max_abs_normalized_pairwise_inner_product", float("inf")))
        > HAAR_ORTHOGONALITY_TOLERANCE
    ):
        raise HaarError("development Haar receipt differs")

    clips = list(range(*DEV_RANGE))
    seeds = list(DEV_NOISE_SEEDS)

    def values(endpoint: str, metric: str) -> Any:
        return np.asarray(
            [
                [float(inventory[(endpoint, clip, seed)]["metrics"][metric]) for seed in seeds]
                for clip in clips
            ],
            dtype=np.float64,
        )

    aggregates = {
        endpoint: {
            metric: {
                "mean": float(values(endpoint, metric).mean()),
                "std": float(values(endpoint, metric).std(ddof=1)),
                "min": float(values(endpoint, metric).min()),
                "max": float(values(endpoint, metric).max()),
            }
            for metric in (*ERROR_METRICS, *BAND_METRICS)
        }
        for endpoint in ENDPOINTS
    }
    comparisons: dict[str, Any] = {}
    comparison_index = 0
    for dose in DOSES:
        full = endpoint_name(dose, "FULL_DIRECT", "ALIGNED")
        for target in TARGETS:
            aligned = endpoint_name(dose, target, "ALIGNED")
            shuffled = endpoint_name(dose, target, "SHUFFLED")
            controls = {"VPM_OFF": "VPM_OFF", "MATCHED_SHUFFLED": shuffled}
            if target != "FULL_DIRECT":
                controls["MATCHED_FULL_DIRECT"] = full
            for control_label, control in controls.items():
                label = f"{aligned}_vs_{control_label}"
                comparisons[label] = {
                    metric: _bootstrap_relative(
                        values(control, metric),
                        values(aligned, metric),
                        seed=BOOTSTRAP_SEED + 100 * comparison_index + metric_index,
                    )
                    for metric_index, metric in enumerate(ERROR_METRICS)
                }
                comparison_index += 1

    learning_curves = {}
    for band_index, band in enumerate(BANDS):
        learning_curves[band] = {}
        for metric_index, metric in enumerate(
            ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
        ):
            learning_curves[band][metric] = _bootstrap_auc_difference(
                values("VPM_OFF", metric),
                [values(endpoint_name(dose, band, "ALIGNED"), metric) for dose in DOSES],
                [values(endpoint_name(dose, "FULL_DIRECT", "ALIGNED"), metric) for dose in DOSES],
                seed=BOOTSTRAP_SEED + 10_000 + 10 * band_index + metric_index,
            )

    def effect(candidate: str, control_label: str, metric: str) -> Mapping[str, Any]:
        return comparisons[f"{candidate}_vs_{control_label}"][metric]

    band_gates = {}
    any_special = False
    for band in BANDS:
        dose_gates = {}
        auc_ok = all(
            learning_curves[band][metric]["paired_episode_bootstrap_95_difference"][0] > 0
            for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
        )
        full_dose = endpoint_name(256, band, "ALIGNED")
        full_noninferior = all(
            effect(full_dose, "MATCHED_FULL_DIRECT", metric)["relative_improvement_percent"] >= -1.0
            for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
        )
        sample_efficiency = auc_ok and full_noninferior
        for dose in DOSES:
            candidate = endpoint_name(dose, band, "ALIGNED")
            attribution_metrics = (
                "corrected_velocity_mse",
                "decoded_mse_unit_range",
                "decoded_temporal_difference_mse_unit_range",
            )
            attribution = all(
                effect(candidate, "MATCHED_SHUFFLED", metric)["relative_improvement_percent"] >= 3.0
                and effect(candidate, "MATCHED_SHUFFLED", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                for metric in attribution_metrics
            ) and all(
                effect(candidate, "MATCHED_SHUFFLED", metric)["favorable_episode_fraction"] >= 0.60
                for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
            )
            off_quality = all(
                effect(candidate, "VPM_OFF", metric)["relative_improvement_percent"] >= 3.0
                and effect(candidate, "VPM_OFF", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
            )
            latent = effect(candidate, "VPM_OFF", "video_future_nmse")
            latent_guardrail = (
                latent["relative_improvement_percent"] >= 0.0
                and latent["paired_episode_bootstrap_95_percent"][0] > -1.0
            )
            matched_quality = all(
                effect(candidate, "MATCHED_FULL_DIRECT", metric)["relative_improvement_percent"] >= 1.0
                and effect(candidate, "MATCHED_FULL_DIRECT", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
            )
            predictable = (
                aggregates[candidate]["target_r2"]["mean"] > 0
                and aggregates[candidate]["target_cosine"]["mean"] > 0
            )
            leakage = aggregates[candidate]["postprojection_offband_relative_energy"]["max"] <= LEAKAGE_TOLERANCE
            passed = (
                attribution
                and off_quality
                and latent_guardrail
                and (matched_quality or sample_efficiency)
                and predictable
                and leakage
            )
            dose_gates[str(dose)] = {
                "aligned_beats_matched_shuffled": attribution,
                "aligned_beats_vpm_off": off_quality,
                "latent_guardrail": latent_guardrail,
                "matched_full_direct_quality_superiority": matched_quality,
                "learning_curve_sample_efficiency_superiority": sample_efficiency,
                "positive_band_r2_and_cosine": predictable,
                "postprojection_leakage_pass": leakage,
                "all_passed": passed,
            }
            any_special = any_special or passed
        band_gates[band] = {
            "learning_curve_auc_superiority": auc_ok,
            "full_dose_noninferior_within_one_percent": full_noninferior,
            "sample_efficiency_superiority": sample_efficiency,
            "doses": dose_gates,
            "any_dose_passed": any(value["all_passed"] for value in dose_gates.values()),
        }
    global_gates = {
        "zero_equals_off_bit_exact": True,
        "one_wan_call_zero_teacher_feature_calls": True,
        "haar_reconstruction_orthogonality_pass": True,
        "post_selection_dev_only": True,
        "fresh_reserve_480_510_unopened": True,
        "constructor_probe_511_unopened": True,
        "validation_and_protected_test_unopened": True,
    }
    passed = any_special and all(global_gates.values())
    return ladder.identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "created_at_utc": ladder._now(),
            "development_episode_count": len(clips),
            "noise_seeds_per_episode": len(seeds),
            "aggregates": aggregates,
            "comparisons": comparisons,
            "learning_curve_sample_efficiency": learning_curves,
            "band_gates": band_gates,
            "global_gates": {**global_gates, "all_passed": passed},
            "decision": "EXPLORATORY_HAAR_SPECIAL" if passed else "EXPLORATORY_NO_HAAR_ADVANTAGE",
            "claim_boundary": (
                "post-selection exploratory reuse of prior-inspected ABC-train rows 416--479; "
                "one frozen VPM parent and linear token-local heads; no reserve, validation, "
                "protected-test, generalization, or paper claim"
            ),
        }
    )


def _validated_endpoint(
    output: Path,
    registration: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    endpoint = ladder.read_json(output / "endpoint_complete.json")
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(ENDPOINTS)
    expected_batches = ((DEV_RANGE[1] - DEV_RANGE[0] + 1) // 2) * len(DEV_NOISE_SEEDS)
    if (
        endpoint.get("schema") != ENDPOINT_SCHEMA
        or not ladder.identity_valid(endpoint)
        or endpoint.get("registration_identity_sha256") != registration["identity_sha256"]
        or endpoint.get("fit_identity_sha256") != fit["identity_sha256"]
        or int(endpoint.get("row_count", -1)) != expected_rows
        or int(endpoint.get("timing_row_count", -1)) != expected_batches
        or endpoint.get("development_clip_range") != list(DEV_RANGE)
        or endpoint.get("dataset_array_validation_rows") != [DEV_RANGE[0], DEV_RANGE[1] - 1]
        or int(endpoint.get("exact_dataset_accesses_per_row", -1)) != len(DEV_NOISE_SEEDS)
        or int(endpoint.get("shared_wan_invocations", -1)) != expected_batches
        or int(endpoint.get("teacher_calls", -1)) != 0
        or endpoint.get("vjepa_target_array_opened") is not False
        or endpoint.get("zero_equals_off_bit_exact") is not True
        or endpoint.get("post_selection_exploratory_dev_opened") is not True
        or endpoint.get("fresh_reserve_480_510_opened") is not False
        or endpoint.get("constructor_probe_511_opened") is not False
        or endpoint.get("validation_opened") is not False
        or endpoint.get("protected_test_opened") is not False
    ):
        raise HaarError("endpoint receipt contract differs")
    paths = []
    for key, label in (("rows", "development rows"), ("timing_rows", "timing rows")):
        record = endpoint.get(key, {})
        path = ladder.canonical_file(record.get("path", ""), label)
        if (
            path.parent != output
            or path.stat().st_size != int(record.get("bytes", -1))
            or ladder.sha256_file(path) != record.get("sha256")
        ):
            raise HaarError(f"{label} artifact differs")
        paths.append(path)
    timing = ladder.read_jsonl(paths[1])
    expected_timing = [
        (seed, list(range(start, min(start + 2, DEV_RANGE[1]))))
        for seed in DEV_NOISE_SEEDS
        for start in range(DEV_RANGE[0], DEV_RANGE[1], 2)
    ]
    if len(timing) != len(expected_timing):
        raise HaarError("timing row count differs")
    for ordinal, (row, (seed, clips)) in enumerate(zip(timing, expected_timing, strict=True)):
        latency = row.get("latency_ms", {})
        if (
            row.get("schema") != TIMING_SCHEMA
            or not ladder.identity_valid(row)
            or int(row.get("batch_ordinal", -1)) != ordinal
            or int(row.get("noise_seed", -1)) != seed
            or row.get("clip_indices") != clips
            or set(latency.get("adapter", {})) != set(ENDPOINTS) - {"VPM_OFF"}
            or set(latency.get("euler", {})) != set(ENDPOINTS)
            or set(latency.get("decoder", {})) != set(ENDPOINTS)
            or any(
                not math.isfinite(float(value)) or float(value) < 0
                for group in (latency["adapter"], latency["euler"], latency["decoder"])
                for value in group.values()
            )
        ):
            raise HaarError("timing row inventory differs")
    return endpoint, paths[0], paths[1]


def _analyze_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    endpoint, rows_path, _timing_path = _validated_endpoint(output, registration, fit)
    rows = ladder.read_jsonl(rows_path)
    analysis = analyze_rows(
        rows,
        registered_samples=registration["selected_manifest_rows"]["exploratory_development"],
        haar_contract_receipt=endpoint["haar_contract"],
    )
    analysis = ladder.identity_payload(
        {
            **analysis,
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
        }
    )
    ladder.exclusive_json(output / "analysis.json", analysis)
    complete = ladder.identity_payload(
        {
            "schema": COMPLETE_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "analysis": ladder.file_record(output / "analysis.json"),
            "decision": analysis["decision"],
            "development_row_count": len(rows),
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "run_complete.json", complete)
    print(json.dumps({"analysis_identity_sha256": analysis["identity_sha256"], "decision": analysis["decision"]}, sort_keys=True))
    return 0


def _audit_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    endpoint, rows_path, timing_path = _validated_endpoint(output, registration, fit)
    analysis = ladder.read_json(output / "analysis.json")
    complete = ladder.read_json(output / "run_complete.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or not ladder.identity_valid(analysis)
        or complete.get("schema") != COMPLETE_SCHEMA
        or not ladder.identity_valid(complete)
        or complete.get("decision") != analysis.get("decision")
        or complete.get("analysis_identity_sha256") != analysis.get("identity_sha256")
        or complete.get("fresh_reserve_480_510_opened") is not False
        or complete.get("constructor_probe_511_opened") is not False
        or complete.get("validation_opened") is not False
        or complete.get("protected_test_opened") is not False
    ):
        raise HaarError("completion/analysis identity contract differs")
    analysis_record = complete.get("analysis", {})
    analysis_path = ladder.canonical_file(analysis_record.get("path", ""), "analysis")
    if (
        analysis_path != output / "analysis.json"
        or analysis_path.stat().st_size != int(analysis_record.get("bytes", -1))
        or ladder.sha256_file(analysis_path) != analysis_record.get("sha256")
    ):
        raise HaarError("analysis artifact receipt differs")
    rows = ladder.read_jsonl(rows_path)
    recomputed = analyze_rows(
        rows,
        registered_samples=registration["selected_manifest_rows"]["exploratory_development"],
        haar_contract_receipt=endpoint["haar_contract"],
    )
    for key in (
        "aggregates",
        "comparisons",
        "learning_curve_sample_efficiency",
        "band_gates",
        "global_gates",
        "decision",
    ):
        if recomputed[key] != analysis.get(key):
            raise HaarError(f"audited {key} differs")
    artifacts = (
        "registration.json",
        "fit.json",
        "adapter_weights.pt",
        "development_rows.jsonl",
        "timing_rows.jsonl",
        "endpoint_complete.json",
        "analysis.json",
        "run_complete.json",
    )
    audit = ladder.identity_payload(
        {
            "schema": AUDIT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "decision": analysis["decision"],
            "artifact_sha256": {
                name: ladder.sha256_file(ladder.canonical_file(output / name, name))
                for name in artifacts
            },
            "row_inventory_recomputed": True,
            "paired_tensor_hashes_recomputed": True,
            "timing_inventory_recomputed": True,
            "zero_noop_recomputed": True,
            "haar_contract_recomputed_from_sealed_receipt": True,
            "role_scoped_row_access_receipts_validated": True,
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "status": "PASS",
        }
    )
    ladder.exclusive_json(output / "audit.json", audit)
    print(json.dumps({"audit_identity_sha256": audit["identity_sha256"], "decision": analysis["decision"], "status": "PASS"}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--repo-root", required=True)
    fit.add_argument("--expected-source-commit", required=True)
    fit.add_argument("--parent-registration", required=True)
    fit.add_argument("--vpm-snapshot-sha256", required=True)
    fit.add_argument("--runtime-record", required=True)
    fit.add_argument("--output-dir", required=True)
    for command in ("evaluate", "analyze", "audit"):
        commands.add_parser(command).add_argument("--output-dir", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "fit":
        return _fit_phase(args)
    if args.command == "evaluate":
        return _evaluate_phase(args)
    if args.command == "analyze":
        return _analyze_phase(args)
    if args.command == "audit":
        return _audit_phase(args)
    raise HaarError("unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HaarError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
