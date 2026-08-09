#!/usr/bin/env python3
"""Train-only one-step direct-residual frontier on the frozen VPM parent."""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import causal_compressibility_ladder as ladder


SCHEMA = "vpm-direct-residual-frontier-registration-v1"
FIT_SCHEMA = "vpm-direct-residual-frontier-fit-v1"
WEIGHT_SCHEMA = "vpm-direct-residual-frontier-weights-v1"
ROW_SCHEMA = "vpm-direct-residual-frontier-row-v1"
ENDPOINT_SCHEMA = "vpm-direct-residual-frontier-endpoint-v1"
ANALYSIS_SCHEMA = "vpm-direct-residual-frontier-analysis-v1"
COMPLETE_SCHEMA = "vpm-direct-residual-frontier-complete-v1"
AUDIT_SCHEMA = "vpm-direct-residual-frontier-audit-v1"

ARMS = ("ZERO", "DIRECT_ALIGNED", "DIRECT_SHUFFLED")
ENDPOINTS = ("VPM_OFF", *ARMS)
FIT_RANGE = (128, 384)
CAL_RANGE = (384, 416)
PRIOR_DEV_RANGE = (416, 480)
OUTCOME_RANGE = (480, 512)
EXCLUDED_RANGE = (0, 128)
FIT_NOISE_SEEDS = (20260832, 20260833)
CAL_NOISE_SEEDS = (20260834, 20260835)
OUTCOME_NOISE_SEEDS = (20260836, 20260837, 20260838, 20260839)
PROJECTION_SEED = 20260840
BOOTSTRAP_SEED = 20260841
BOOTSTRAP_REPLICATES = 10_000
CAPACITY_RUNGS = (64, 256, 1024)
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TOKEN_SAMPLE_COUNT = 256
ERROR_METRICS = (
    "corrected_velocity_mse",
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)


class FrontierError(ladder.LadderError):
    """A prospective, lineage, or serving contract changed."""


def _configure_shared_helpers() -> None:
    # The imported helpers deliberately read these module globals at call time.
    ladder.ARMS = ARMS
    ladder.CAPACITY_RUNGS = CAPACITY_RUNGS
    ladder.RIDGE_LAMBDAS = RIDGE_LAMBDAS
    ladder.PROJECTION_SEED = PROJECTION_SEED
    ladder.TOKEN_SAMPLE_COUNT = TOKEN_SAMPLE_COUNT


def _validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 512:
        raise FrontierError("canonical train manifest must contain 512 rows")
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
            raise FrontierError(f"canonical manifest identity differs at row {index}")
        episodes.add(episode)
        clips.add(clip)
    ranges = {
        "excluded_teacher_history": list(EXCLUDED_RANGE),
        "fit": list(FIT_RANGE),
        "calibration": list(CAL_RANGE),
        "prior_inspected_development": list(PRIOR_DEV_RANGE),
        "fresh_reserve_outcome": list(OUTCOME_RANGE),
    }
    covered: list[int] = []
    for start, stop in ranges.values():
        covered.extend(range(start, stop))
    if sorted(covered) != list(range(512)) or len(set(covered)) != 512:
        raise FrontierError("registered partitions do not form an exact partition")
    seed_sets = [
        set(FIT_NOISE_SEEDS),
        set(CAL_NOISE_SEEDS),
        set(OUTCOME_NOISE_SEEDS),
    ]
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise FrontierError("fit/calibration/outcome noise seeds overlap")
    return {
        "ranges": ranges,
        "all_episodes_unique": True,
        "all_clips_unique": True,
        "fit_noise_seeds": list(FIT_NOISE_SEEDS),
        "calibration_noise_seeds": list(CAL_NOISE_SEEDS),
        "outcome_noise_seeds": list(OUTCOME_NOISE_SEEDS),
    }


def _validated_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = ladder.canonical_file(record.get("path", ""), label)
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise FrontierError(f"{label} byte size differs")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise FrontierError(f"{label} lacks a full SHA-256")
    # Rehash configs/manifests. The 4.25-GB parent is already content-bound by
    # its immutable parent registration and is rechecked by the Slurm wrapper.
    if "snapshot" not in label and ladder.sha256_file(path) != digest:
        raise FrontierError(f"{label} digest differs")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _validated_cache_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Carry forward a large-array digest already bound by the parent receipt."""
    path = ladder.canonical_file(record.get("path", ""), label)
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise FrontierError(f"{label} byte size differs")
    digest = record.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or record.get("digest_source") != "complete immutable cache metadata"
    ):
        raise FrontierError(f"{label} is not metadata-bound by the parent")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "digest_source": "complete immutable cache metadata",
    }


def _prepare_registration(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    repo = ladder.canonical_directory(args.repo_root, "repository")
    observed_commit = ladder.git_output(repo, "rev-parse", "HEAD")
    if observed_commit != args.expected_source_commit or len(observed_commit) != 40:
        raise FrontierError("source commit differs")
    if ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise FrontierError("source repository must be clean")
    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FrontierError("fresh output already exists")

    parent_path = ladder.canonical_file(
        args.parent_registration, "parent ladder registration"
    )
    parent = ladder.read_json(parent_path)
    if (
        parent.get("schema") != ladder.SCHEMA
        or not ladder.identity_valid(parent)
        or "VPM" not in parent.get("lineage", {})
    ):
        raise FrontierError("parent ladder registration is invalid")
    required_inputs = (
        "train_manifest",
        "train_cache_metadata",
        "vpm_resolved_config",
        "vpm_snapshot",
        "vpm_arm_manifest",
        "vpm_stage_manifest",
        "vpm_stage_outcome",
    )
    inputs = {
        key: _validated_record(parent["inputs"][key], key)
        for key in required_inputs
    }
    manifest_rows = ladder.read_jsonl(Path(inputs["train_manifest"]["path"]))
    partitions = _validate_manifest(manifest_rows)
    arrays = {
        key: _validated_cache_record(record, f"cache {key}")
        for key, record in parent["cache_arrays"].items()
    }
    if set(arrays) != {"target", "rgb", "actions"}:
        raise FrontierError("parent cache array inventory differs")
    vpm_spec = parent["lineage"]["VPM"].get("arm_spec", {})
    if (
        vpm_spec.get("parameter_matched_control") is not True
        or vpm_spec.get("condition_mode") != "off"
    ):
        raise FrontierError("registered parent is not the VPM frontier")

    output.mkdir(parents=True, mode=0o700)
    output = output.resolve(strict=True)
    registration = ladder.identity_payload(
        {
            "schema": SCHEMA,
            "created_at_utc": ladder._now(),
            "status": "registered_before_vpm_fit_or_reserve_outcome_open",
            "repo_root": str(repo),
            "source_commit": observed_commit,
            "output_dir": str(output),
            "parent_registration": ladder.file_record(parent_path),
            "parent_registration_identity_sha256": parent["identity_sha256"],
            "inputs": inputs,
            "cache_arrays": arrays,
            "lineage": {"VPM": parent["lineage"]["VPM"]},
            "partitions": partitions,
            "frozen_design": {
                "arms": list(ARMS),
                "capacity_rungs": list(CAPACITY_RUNGS),
                "ridge_lambdas": list(RIDGE_LAMBDAS),
                "projection_seed": PROJECTION_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "token_sample_count_per_clip_noise": TOKEN_SAMPLE_COUNT,
                "selection_target": "DIRECT_ALIGNED calibration corrected velocity MSE",
                "tie_break": "smaller capacity, then larger ridge penalty",
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
                    "fresh_reserve_outcome": OUTCOME_RANGE,
                }.items()
            },
            "access_contract": {
                "vjepa_target_array_allowed": False,
                "teacher_calls": 0,
                "validation_opened": False,
                "protected_test_opened": False,
                "outcome_opened_during_registration": False,
                "wandb_enabled": False,
            },
            "claim_boundary": (
                "ordinary direct residual hard control on one frozen VPM parent; "
                "not dual diffusion and no validation/protected-test claim"
            ),
        }
    )
    ladder.exclusive_json(output / "registration.json", registration)
    return output, registration


def _load_registration(output: Path) -> dict[str, Any]:
    value = ladder.read_json(output / "registration.json")
    if value.get("schema") != SCHEMA or not ladder.identity_valid(value):
        raise FrontierError("registration identity/schema differs")
    if Path(value["output_dir"]).resolve(strict=True) != output:
        raise FrontierError("registration output path differs")
    repo = ladder.canonical_directory(value["repo_root"], "registered repository")
    if (
        ladder.git_output(repo, "rev-parse", "HEAD") != value["source_commit"]
        or ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise FrontierError("registered source state changed")
    return value


def _resolve_input(registration: Mapping[str, Any], key: str) -> Path:
    record = registration["inputs"][key]
    return Path(_validated_record(record, key)["path"])


def _future_rows(
    projected: Any,
    target_tokens: Mapping[str, Any],
    positions: Any,
    indexes: Sequence[int],
    noise_seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    feature_rows = []
    rows = {arm: [] for arm in ARMS}
    for local, clip_index in enumerate(indexes):
        selected = ladder.deterministic_token_subsample(
            positions,
            clip_index=int(clip_index),
            noise_seed=int(noise_seed),
            count=TOKEN_SAMPLE_COUNT,
        )
        feature_rows.append(projected[local, selected])
        for arm in ARMS:
            rows[arm].append(target_tokens[arm][local, selected])
    return torch.cat(feature_rows), {
        arm: torch.cat(values) for arm, values in rows.items()
    }


def _fit_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise FrontierError("CUDA is required")
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
        int(model.forward_model.transformer.dim),
        max(CAPACITY_RUNGS),
        seed=PROJECTION_SEED,
    )
    projection = projection_cpu.to(device=device)
    stats = None
    grid = patch_size = None
    patch_dim = None
    fit_calls = 0
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = ladder._no_auxiliary_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for noise_seed in FIT_NOISE_SEEDS:
                for start in range(FIT_RANGE[0], FIT_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = ladder._batch_samples(dataset, indexes, device)
                    prepared = ladder._model_inputs(
                        model,
                        batch,
                        noise_seed=noise_seed,
                        nfe=1,
                        need_clean_auxiliary=False,
                    )
                    off_velocity, hidden = ladder._off_prediction_and_hidden(
                        model, prepared
                    )
                    fit_calls += 1
                    observed = ladder._grid_contract(model, off_velocity, hidden)
                    if grid is None:
                        grid, patch_size, patch_dim = observed
                        stats = ladder._new_sufficient_statistics(
                            max(CAPACITY_RUNGS), patch_dim, device
                        )
                    elif observed != (grid, patch_size, patch_dim):
                        raise FrontierError("Wan grid changed during fit")
                    projected = hidden.float() @ projection
                    direct = prepared["video_target"] - off_velocity
                    direct_tokens = ladder.patchify_video(direct, patch_size)
                    shuffled_tokens = torch.roll(direct_tokens, shifts=-1, dims=0)
                    if torch.equal(direct_tokens[0], shuffled_tokens[0]):
                        raise FrontierError("aligned and shuffled residuals coincide")
                    target_tokens = {
                        "ZERO": torch.zeros_like(direct_tokens),
                        "DIRECT_ALIGNED": direct_tokens,
                        "DIRECT_SHUFFLED": shuffled_tokens,
                    }
                    positions = ladder.future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                        device=device,
                    )
                    x, y = _future_rows(
                        projected, target_tokens, positions, indexes, noise_seed
                    )
                    ladder.update_sufficient_statistics(stats, x, y)
                    del batch, prepared, hidden, projected, direct, direct_tokens
        opened = {path.resolve(strict=True) for path in guard.opened}
    if opened != expected_arrays:
        raise FrontierError(f"fit input graph differs: {sorted(opened)}")
    if stats is None or grid is None or patch_size is None or patch_dim is None:
        raise FrontierError("fit produced no sufficient statistics")
    expected_fit_tokens = (
        (FIT_RANGE[1] - FIT_RANGE[0])
        * len(FIT_NOISE_SEEDS)
        * TOKEN_SAMPLE_COUNT
    )
    if int(stats["count"]) != expected_fit_tokens:
        raise FrontierError("fit token count differs")
    del dataset
    gc.collect()

    cal_x: list[Any] = []
    cal_y: list[Any] = []
    cal_calls = 0
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = ladder._no_auxiliary_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for noise_seed in CAL_NOISE_SEEDS:
                for start in range(CAL_RANGE[0], CAL_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = ladder._batch_samples(dataset, indexes, device)
                    prepared = ladder._model_inputs(
                        model,
                        batch,
                        noise_seed=noise_seed,
                        nfe=1,
                        need_clean_auxiliary=False,
                    )
                    off_velocity, hidden = ladder._off_prediction_and_hidden(
                        model, prepared
                    )
                    cal_calls += 1
                    projected = hidden.float() @ projection
                    target = ladder.patchify_video(
                        prepared["video_target"] - off_velocity, patch_size
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
                        cal_x.append(
                            projected[local, selected].cpu().to(torch.float16)
                        )
                        cal_y.append(target[local, selected].cpu().to(torch.float16))
        opened = {path.resolve(strict=True) for path in guard.opened}
    if opened != expected_arrays:
        raise FrontierError(f"calibration input graph differs: {sorted(opened)}")
    features = torch.cat(cal_x).to(device=device, dtype=torch.float32)
    targets = torch.cat(cal_y).to(device=device, dtype=torch.float32)
    expected_cal_tokens = (
        (CAL_RANGE[1] - CAL_RANGE[0])
        * len(CAL_NOISE_SEEDS)
        * TOKEN_SAMPLE_COUNT
    )
    if features.shape[0] != expected_cal_tokens:
        raise FrontierError("calibration token count differs")

    candidates = []
    for capacity in CAPACITY_RUNGS:
        for ridge_lambda in RIDGE_LAMBDAS:
            head = ladder.fit_ridge_head(
                stats,
                arm="DIRECT_ALIGNED",
                capacity=capacity,
                ridge_lambda=ridge_lambda,
            )
            prediction = ladder.predict_ridge_head(features, head)
            candidates.append(
                {
                    "capacity": capacity,
                    "ridge_lambda": ridge_lambda,
                    "corrected_velocity_mse": float(
                        (prediction - targets).square().mean().detach().cpu()
                    ),
                    "parameter_count": head["parameter_count"],
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            row["corrected_velocity_mse"],
            row["capacity"],
            -row["ridge_lambda"],
        ),
    )
    heads = {
        arm: ladder.fit_ridge_head(
            stats,
            arm=arm,
            capacity=int(selected["capacity"]),
            ridge_lambda=float(selected["ridge_lambda"]),
        )
        for arm in ARMS
    }
    parameter_counts = {head["parameter_count"] for head in heads.values()}
    if len(parameter_counts) != 1:
        raise FrontierError("selected heads do not have equal capacity")
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
                "noise_seeds": list(FIT_NOISE_SEEDS),
                "sampled_future_tokens": int(stats["count"]),
                "off_wan_invocations": fit_calls,
                "teacher_calls": 0,
                "vjepa_target_array_opened": False,
            },
            "calibration": {
                "clip_indices": list(CAL_RANGE),
                "noise_seeds": list(CAL_NOISE_SEEDS),
                "sampled_future_tokens": int(features.shape[0]),
                "off_wan_invocations": cal_calls,
                "teacher_calls": 0,
                "candidate_results": candidates,
            },
            "selected": selected,
            "equal_parameter_count_per_arm": parameter_counts.pop(),
            "zero_head_exact_zero": True,
            "fresh_reserve_outcome_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "wandb_enabled": False,
        }
    )
    ladder.exclusive_json(output / "fit.json", fit)
    print(
        json.dumps(
            {"fit_identity_sha256": fit["identity_sha256"], "selected": selected},
            sort_keys=True,
        )
    )
    return 0


def _load_weights(
    output: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    fit = ladder.read_json(output / "fit.json")
    if fit.get("schema") != FIT_SCHEMA or not ladder.identity_valid(fit):
        raise FrontierError("fit identity/schema differs")
    if fit.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise FrontierError("fit registration link differs")
    record = fit["weights"]
    path = ladder.canonical_file(record["path"], "adapter weights")
    if path.parent != output or ladder.sha256_file(path) != record["sha256"]:
        raise FrontierError("adapter weight artifact differs")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if (
        weights.get("schema") != WEIGHT_SCHEMA
        or weights.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or set(weights.get("heads", {})) != set(ARMS)
    ):
        raise FrontierError("adapter weight payload differs")
    return fit, dict(weights)


def _event_ms(start: Any, end: Any) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _evaluate_phase(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise FrontierError("CUDA is required")
    _configure_shared_helpers()
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, weights = _load_weights(output, registration)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, config = ladder._load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    projection = weights["projection"].to(device=device, dtype=torch.float32)
    grid = tuple(int(value) for value in weights["grid"])
    patch_size = tuple(int(value) for value in weights["patch_size"])
    manifest = ladder.read_jsonl(_resolve_input(registration, "train_manifest"))
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    rows: list[dict[str, Any]] = []
    wan_invocations = 0
    adapter_timings = {arm: [] for arm in ARMS}
    wan_timings: list[float] = []
    decoder_timings = {endpoint: [] for endpoint in ENDPOINTS}
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = ladder._no_auxiliary_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for noise_seed in OUTCOME_NOISE_SEEDS:
                for start in range(OUTCOME_RANGE[0], OUTCOME_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = ladder._batch_samples(dataset, indexes, device)
                    prepared = ladder._model_inputs(
                        model,
                        batch,
                        noise_seed=noise_seed,
                        nfe=1,
                        need_clean_auxiliary=False,
                    )
                    before = torch.cuda.Event(enable_timing=True)
                    after = torch.cuda.Event(enable_timing=True)
                    before.record()
                    off_velocity, hidden = ladder._off_prediction_and_hidden(
                        model, prepared
                    )
                    after.record()
                    wan_timings.append(_event_ms(before, after))
                    wan_invocations += 1
                    observed_grid, observed_patch, _ = ladder._grid_contract(
                        model, off_velocity, hidden
                    )
                    if observed_grid != grid or observed_patch != patch_size:
                        raise FrontierError("outcome Wan grid differs from fit")
                    positions = ladder.future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                        device=device,
                    )
                    first_future = int(positions[0])
                    velocities = {"VPM_OFF": off_velocity.float()}
                    residuals = {}
                    for arm in ARMS:
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        projected = hidden.float() @ projection
                        tokens = ladder.predict_ridge_head(
                            projected, weights["heads"][arm]
                        ).clone()
                        tokens[:, :first_future] = 0
                        residual = ladder.unpatchify_tokens(
                            tokens,
                            grid=grid,
                            patch_size=patch_size,
                            channels=off_velocity.shape[1],
                        )
                        after.record()
                        adapter_timings[arm].append(_event_ms(before, after))
                        residuals[arm] = residual
                        velocities[arm] = off_velocity.float() + residual.float()
                    if int(torch.count_nonzero(residuals["ZERO"])) != 0:
                        raise FrontierError("ZERO residual is not exact zero")

                    decoded_clean = model.rgb_tokenizer.decode_temporal(
                        prepared["video_clean"],
                        out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                    )
                    future_frames = ladder._validate_decoded_horizon(
                        model_future_frames=model.num_future_frames,
                        decoded_frames=decoded_clean.shape[2],
                        raw_frames=batch["rgb"].shape[1],
                        endpoint="VPM_DIRECT",
                    )
                    vae_target = ladder._to_uint8_video(
                        decoded_clean[:, :, -future_frames:]
                    )
                    vae_history = ladder._to_uint8_video(
                        decoded_clean[:, :, -(future_frames + 1) : -future_frames]
                    )
                    raw_video = batch["rgb"].permute(0, 2, 1, 3, 4)
                    raw_target = ladder._to_uint8_video(
                        raw_video[:, :, -future_frames:]
                    )
                    raw_history = ladder._to_uint8_video(
                        raw_video[:, :, -(future_frames + 1) : -future_frames]
                    )
                    direct_target = prepared["video_target"] - off_velocity
                    finals = {}
                    metrics_by_endpoint = {}
                    for endpoint, velocity in velocities.items():
                        final = ladder._one_step_final(
                            prepared["initial_video"],
                            velocity,
                            prepared["reference"],
                            prepared["history_frames"],
                        )
                        finals[endpoint] = final
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        decoded = ladder._to_uint8_video(
                            model.rgb_tokenizer.decode_temporal(
                                final.to(batch["rgb"].dtype),
                                out_hw=(
                                    batch["rgb"].shape[-2],
                                    batch["rgb"].shape[-1],
                                ),
                            )[:, :, -future_frames:]
                        )
                        after.record()
                        decoder_timings[endpoint].append(_event_ms(before, after))
                        metrics_by_endpoint[endpoint] = ladder._per_sample_future_metrics(
                            velocity=velocity,
                            base_velocity=off_velocity,
                            direct_residual=direct_target,
                            final=final,
                            clean=prepared["video_clean"],
                            decoded=decoded,
                            raw_target=raw_target,
                            raw_history=raw_history,
                            vae_target=vae_target,
                            vae_history=vae_history,
                            history_frames=prepared["history_frames"],
                        )
                    if not torch.equal(finals["VPM_OFF"], finals["ZERO"]):
                        raise FrontierError("ZERO final differs from VPM_OFF")

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
                                        "adapter_latency_ms_per_batch": (
                                            0.0
                                            if endpoint == "VPM_OFF"
                                            else adapter_timings[endpoint][-1]
                                        ),
                                        "tensor_sha256": {
                                            "initial_video": ladder._safe_tensor_sha256(
                                                prepared["initial_video"][local : local + 1]
                                            ),
                                            "final_video": ladder._safe_tensor_sha256(
                                                finals[endpoint][local : local + 1]
                                            ),
                                            "actions": ladder._safe_tensor_sha256(
                                                batch["actions"][local : local + 1]
                                            ),
                                            "raw_future_target": ladder._safe_tensor_sha256(
                                                raw_target[local : local + 1]
                                            ),
                                            "raw_history_boundary": ladder._safe_tensor_sha256(
                                                raw_history[local : local + 1]
                                            ),
                                        },
                                        "fresh_reserve_outcome_opened": True,
                                        "validation_opened": False,
                                        "protected_test_opened": False,
                                    }
                                )
                            )
                    del batch, prepared, hidden, velocities, residuals, finals
        opened = {path.resolve(strict=True) for path in guard.opened}
    if opened != expected_arrays:
        raise FrontierError(f"outcome input graph differs: {sorted(opened)}")
    expected_rows = (
        (OUTCOME_RANGE[1] - OUTCOME_RANGE[0])
        * len(OUTCOME_NOISE_SEEDS)
        * len(ENDPOINTS)
    )
    expected_calls = (
        (OUTCOME_RANGE[1] - OUTCOME_RANGE[0])
        // 2
        * len(OUTCOME_NOISE_SEEDS)
    )
    if len(rows) != expected_rows or wan_invocations != expected_calls:
        raise FrontierError("outcome row/Wan accounting differs")
    rows_path = output / "outcome_rows.jsonl"
    ladder.exclusive_jsonl(rows_path, rows)
    receipt = ladder.identity_payload(
        {
            "schema": ENDPOINT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "rows": ladder.file_record(rows_path),
            "row_count": len(rows),
            "shared_wan_invocations": wan_invocations,
            "wan_sample_calls": (OUTCOME_RANGE[1] - OUTCOME_RANGE[0])
            * len(OUTCOME_NOISE_SEEDS),
            "teacher_calls": 0,
            "vjepa_target_array_opened": False,
            "future_target_entered_correction": False,
            "zero_equals_off_bit_exact": True,
            "adapter_latency_ms_per_batch": {
                arm: {
                    "mean": float(np.mean(values)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                for arm, values in adapter_timings.items()
            },
            "wan_latency_ms_per_batch": {
                "mean": float(np.mean(wan_timings)),
                "p50": float(np.quantile(wan_timings, 0.50)),
                "p95": float(np.quantile(wan_timings, 0.95)),
            },
            "decoder_latency_ms_per_batch": {
                endpoint: {
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                for endpoint, values in decoder_timings.items()
            },
            "latency_scope": (
                "component timing; evaluator clean-target VAE is excluded from serving latency"
            ),
            "fresh_reserve_outcome_opened": True,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "endpoint_complete.json", receipt)
    print(json.dumps({"endpoint_identity_sha256": receipt["identity_sha256"]}))
    return 0


def _bootstrap_relative(
    control: Any, candidate: Any, *, seed: int
) -> dict[str, Any]:
    import numpy as np

    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        control.shape != candidate.shape
        or control.ndim != 2
        or control.shape[0] != OUTCOME_RANGE[1] - OUTCOME_RANGE[0]
        or not np.isfinite(control).all()
        or not np.isfinite(candidate).all()
        or control.mean() <= 0
    ):
        raise FrontierError("bootstrap input differs")
    point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0, control.shape[0], size=(BOOTSTRAP_REPLICATES, control.shape[0])
    )
    controls = control[indexes].mean(axis=(1, 2))
    candidates = candidate[indexes].mean(axis=(1, 2))
    effects = 100.0 * (controls - candidates) / controls
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


def analyze_rows(
    rows: Sequence[Mapping[str, Any]], *, adapter_p95_ms: float
) -> dict[str, Any]:
    import numpy as np

    expected = {
        (endpoint, clip, seed)
        for endpoint in ENDPOINTS
        for clip in range(*OUTCOME_RANGE)
        for seed in OUTCOME_NOISE_SEEDS
    }
    inventory: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("schema") != ROW_SCHEMA or not ladder.identity_valid(row):
            raise FrontierError("outcome row identity/schema differs")
        key = (
            str(row.get("endpoint")),
            int(row.get("clip_index", -1)),
            int(row.get("noise_seed", -1)),
        )
        if key in inventory:
            raise FrontierError(f"duplicate outcome row: {key}")
        if (
            int(row.get("wan_calls", -1)) != 1
            or int(row.get("teacher_calls", -1)) != 0
            or row.get("vjepa_target_array_opened") is not False
            or row.get("future_target_entered_correction") is not False
            or row.get("fresh_reserve_outcome_opened") is not True
            or row.get("validation_opened") is not False
            or row.get("protected_test_opened") is not False
        ):
            raise FrontierError("outcome row violates serving/split contract")
        metrics = row.get("metrics")
        tensor_hashes = row.get("tensor_sha256")
        required_metrics = {
            *ERROR_METRICS,
            "residual_r2",
            "residual_cosine",
        }
        required_hashes = {
            "initial_video",
            "final_video",
            "actions",
            "raw_future_target",
            "raw_history_boundary",
        }
        if (
            not isinstance(metrics, Mapping)
            or not required_metrics.issubset(metrics)
            or any(not math.isfinite(float(metrics[key])) for key in required_metrics)
            or not isinstance(tensor_hashes, Mapping)
            or not required_hashes.issubset(tensor_hashes)
            or any(
                not isinstance(tensor_hashes[key], str)
                or len(tensor_hashes[key]) != 64
                for key in required_hashes
            )
        ):
            raise FrontierError("outcome row metric/hash payload differs")
        inventory[key] = row
    if set(inventory) != expected:
        raise FrontierError("outcome row inventory differs")
    for clip in range(*OUTCOME_RANGE):
        for seed in OUTCOME_NOISE_SEEDS:
            for field in (
                "initial_video",
                "actions",
                "raw_future_target",
                "raw_history_boundary",
            ):
                if len(
                    {
                        inventory[(endpoint, clip, seed)]["tensor_sha256"][field]
                        for endpoint in ENDPOINTS
                    }
                ) != 1:
                    raise FrontierError(f"paired {field} differs for {(clip, seed)}")
            off = inventory[("VPM_OFF", clip, seed)]
            zero = inventory[("ZERO", clip, seed)]
            if (
                off["tensor_sha256"]["final_video"]
                != zero["tensor_sha256"]["final_video"]
                or off["metrics"] != zero["metrics"]
            ):
                raise FrontierError("ZERO is not bit-exact to VPM_OFF")
    clips = list(range(*OUTCOME_RANGE))
    seeds = list(OUTCOME_NOISE_SEEDS)

    def values(endpoint: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(inventory[(endpoint, clip, seed)]["metrics"][metric])
                    for seed in seeds
                ]
                for clip in clips
            ],
            dtype=np.float64,
        )

    aggregates = {
        endpoint: {
            metric: {
                "mean": float(values(endpoint, metric).mean()),
                "std": float(values(endpoint, metric).std(ddof=1)),
            }
            for metric in (*ERROR_METRICS, "residual_r2", "residual_cosine")
        }
        for endpoint in ENDPOINTS
    }
    comparisons = {}
    offset = 0
    for candidate, controls in {
        "DIRECT_ALIGNED": ("VPM_OFF", "DIRECT_SHUFFLED"),
        "DIRECT_SHUFFLED": ("VPM_OFF",),
    }.items():
        for control in controls:
            label = f"{candidate}_vs_{control}"
            comparisons[label] = {}
            for metric_index, metric in enumerate(ERROR_METRICS):
                comparisons[label][metric] = _bootstrap_relative(
                    values(control, metric),
                    values(candidate, metric),
                    seed=BOOTSTRAP_SEED + 10 * offset + metric_index,
                )
            offset += 1

    def effect(control: str, metric: str) -> Mapping[str, Any]:
        return comparisons[f"DIRECT_ALIGNED_vs_{control}"][metric]

    gates: dict[str, bool] = {}
    for control in ("VPM_OFF", "DIRECT_SHUFFLED"):
        velocity = effect(control, "corrected_velocity_mse")
        gates[f"velocity_beats_{control}_3pct_ci1"] = (
            velocity["relative_improvement_percent"] >= 3.0
            and velocity["paired_episode_bootstrap_95_percent"][0] > 1.0
        )
        for metric in (
            "decoded_mse_unit_range",
            "decoded_temporal_difference_mse_unit_range",
        ):
            result = effect(control, metric)
            gates[f"{metric}_beats_{control}_3pct_ci1"] = (
                result["relative_improvement_percent"] >= 3.0
                and result["paired_episode_bootstrap_95_percent"][0] > 1.0
                and result["favorable_episode_fraction"] >= 0.60
            )
    latent = effect("VPM_OFF", "video_future_nmse")
    gates["latent_nonnegative_guardrail"] = (
        latent["relative_improvement_percent"] >= 0.0
        and latent["paired_episode_bootstrap_95_percent"][0] > -1.0
    )
    gates["adapter_p95_below_1ms"] = float(adapter_p95_ms) < 1.0
    gates["zero_equals_off_bit_exact"] = True
    gates["one_wan_call_feature_free_serving"] = True
    gates["fresh_reserve_only_no_validation_or_test"] = True
    passed = all(gates.values())
    return ladder.identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "created_at_utc": ladder._now(),
            "outcome_episode_count": len(clips),
            "noise_seeds_per_episode": len(seeds),
            "aggregates": aggregates,
            "comparisons": comparisons,
            "adapter_p95_ms_per_batch": float(adapter_p95_ms),
            "gates": {**gates, "all_passed": passed},
            "decision": (
                "ADVANCE_VPM_DIRECT_RESIDUAL"
                if passed
                else "STOP_VPM_DIRECT_RESIDUAL"
            ),
            "claim_boundary": (
                "one train-only reserve, one VPM parent, linear token-local head; "
                "not dual diffusion and no validation/protected-test claim"
            ),
        }
    )


def _analyze_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    endpoint = ladder.read_json(output / "endpoint_complete.json")
    if (
        endpoint.get("schema") != ENDPOINT_SCHEMA
        or not ladder.identity_valid(endpoint)
        or endpoint.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or endpoint.get("fit_identity_sha256") != fit["identity_sha256"]
    ):
        raise FrontierError("endpoint receipt differs")
    rows_path = ladder.canonical_file(endpoint["rows"]["path"], "outcome rows")
    if rows_path.parent != output or ladder.sha256_file(rows_path) != endpoint["rows"]["sha256"]:
        raise FrontierError("outcome rows artifact differs")
    rows = ladder.read_jsonl(rows_path)
    adapter_p95 = float(
        endpoint["adapter_latency_ms_per_batch"]["DIRECT_ALIGNED"]["p95"]
    )
    analysis = analyze_rows(rows, adapter_p95_ms=adapter_p95)
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
            "outcome_row_count": len(rows),
            "teacher_calls": 0,
            "vjepa_target_array_opens": 0,
            "fresh_reserve_outcome_opened": True,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "run_complete.json", complete)
    print(
        json.dumps(
            {
                "analysis_identity_sha256": analysis["identity_sha256"],
                "decision": analysis["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


def _audit_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    endpoint = ladder.read_json(output / "endpoint_complete.json")
    analysis = ladder.read_json(output / "analysis.json")
    complete = ladder.read_json(output / "run_complete.json")
    for value, schema in (
        (endpoint, ENDPOINT_SCHEMA),
        (analysis, ANALYSIS_SCHEMA),
        (complete, COMPLETE_SCHEMA),
    ):
        if value.get("schema") != schema or not ladder.identity_valid(value):
            raise FrontierError("audit identity/schema differs")
    if (
        complete.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or complete.get("fit_identity_sha256") != fit["identity_sha256"]
        or complete.get("endpoint_identity_sha256") != endpoint["identity_sha256"]
        or complete.get("analysis_identity_sha256") != analysis["identity_sha256"]
        or complete.get("decision") != analysis["decision"]
    ):
        raise FrontierError("completion identity links differ")
    rows = ladder.read_jsonl(output / "outcome_rows.jsonl")
    recomputed = analyze_rows(
        rows,
        adapter_p95_ms=float(
            endpoint["adapter_latency_ms_per_batch"]["DIRECT_ALIGNED"]["p95"]
        ),
    )
    for key in ("aggregates", "comparisons", "gates", "decision"):
        if recomputed[key] != analysis[key]:
            raise FrontierError(f"audited {key} differs")
    artifacts = (
        "registration.json",
        "fit.json",
        "adapter_weights.pt",
        "outcome_rows.jsonl",
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
                name: ladder.sha256_file(
                    ladder.canonical_file(output / name, name)
                )
                for name in artifacts
            },
            "row_inventory_recomputed": True,
            "paired_hashes_recomputed": True,
            "zero_noop_recomputed": True,
            "teacher_calls": 0,
            "vjepa_target_array_opens": 0,
            "fresh_reserve_outcome_opened": True,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "audit.json", audit)
    print(
        json.dumps(
            {"audit_identity_sha256": audit["identity_sha256"], "decision": audit["decision"]},
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--repo-root", required=True)
    fit.add_argument("--expected-source-commit", required=True)
    fit.add_argument("--parent-registration", required=True)
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
    raise FrontierError("unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FrontierError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
