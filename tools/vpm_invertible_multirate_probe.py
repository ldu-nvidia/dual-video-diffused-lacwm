#!/usr/bin/env python3
"""Frozen, inference-only invertible multirate VPM probe (ILSF-2).

This evaluator never trains or mutates the production sampler.  It uses only
ABC train rows 416--479, which are already prior-inspected exploratory rows.
Clean targets are constructed only after all deployable endpoint latents have
been materialized behind an explicit event barrier.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_compressibility_ladder as ladder
from tools import vpm_causal_haar_residual_exploration as haar


REGISTRATION_SCHEMA = "vpm-invertible-multirate-registration-v1"
ROW_SCHEMA = "vpm-invertible-multirate-row-v1"
TIMING_SCHEMA = "vpm-invertible-multirate-timing-v1"
ENDPOINT_SCHEMA = "vpm-invertible-multirate-endpoint-v1"
ANALYSIS_SCHEMA = "vpm-invertible-multirate-analysis-v1"
AUDIT_SCHEMA = "vpm-invertible-multirate-audit-v1"
COMPLETE_SCHEMA = "vpm-invertible-multirate-complete-v1"

BASE_COMMIT = "997a9dae79d63627a65773ef19cc41462db85c7d"
DEV_RANGE = (416, 480)
FORBIDDEN_RESERVE_RANGE = (480, 511)
CONSTRUCTOR_EXCLUDED_RANGE = (511, 512)
DEV_NOISE_SEEDS = (20261101, 20261102, 20261103, 20261104)
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_REPLICATES = 10_000

LATENT_SHAPE = (16, 4, 24, 120)
HISTORY_LATENT_FRAMES = 2
VIEW_COUNT = 3
VIEW_WIDTH = 40
BATCH_SIZE = 2

ENDPOINTS = (
    "VPM1",
    "VPM2_ORDINARY",
    "LL_FIRST_ALIGNED",
    "LL_FIRST_EPISODE_SHUFFLED",
    "LL_FIRST_TIME_REVERSED",
    "HH_FIRST_RANK_MATCHED",
)
TWO_CALL_ENDPOINTS = ENDPOINTS[1:]
PRIMARY = "LL_FIRST_ALIGNED"
CONTRASTS = (
    ("VPM2_ORDINARY", PRIMARY),
    ("VPM1", PRIMARY),
    ("LL_FIRST_EPISODE_SHUFFLED", PRIMARY),
    ("LL_FIRST_TIME_REVERSED", PRIMARY),
    ("HH_FIRST_RANK_MATCHED", PRIMARY),
)
PRIMARY_METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
ALL_METRICS = (
    *PRIMARY_METRICS,
    "ll_future_nmse",
    "ll_complement_future_nmse",
    "ll_temporal_delta_mse",
    "ll_temporal_delta_cosine",
    "ll_prediction_energy_fraction",
    "hh_prediction_energy_fraction",
    "p_lock_max_abs_error",
)
SIMULTANEOUS_COMPARISON_COUNT = len(CONTRASTS) * len(PRIMARY_METRICS)
BONFERRONI_ONE_SIDED_ALPHA = 0.05 / SIMULTANEOUS_COMPARISON_COUNT

MAX_ABS_TOLERANCE = 2e-6
RELATIVE_ENERGY_TOLERANCE = 1e-12
ORTHOGONALITY_TOLERANCE = 2e-6
IDEMPOTENCE_TOLERANCE = 2e-6
PROJECTION_P95_LIMIT_MS = 2.0
E2E_OVERHEAD_LIMIT_PERCENT = 5.0


class MultirateError(ladder.LadderError):
    """A frozen causal, numerical, access, or evidence contract changed."""


def _future_only(value: Any, history_frames: int) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise MultirateError("video latent must have shape [B,C,T,H,W]")
    if not 0 < int(history_frames) < int(value.shape[2]):
        raise MultirateError("history boundary is outside the latent timeline")
    result = torch.zeros_like(value, dtype=torch.float32)
    result[:, :, history_frames:] = value.float()[:, :, history_frames:]
    return result


def _project_view(value: Any, band: str) -> Any:
    """Project one view onto an orthonormal first-level spatial Haar band."""
    import torch

    if value.ndim != 5 or value.shape[-2] % 2 or value.shape[-1] % 2:
        raise MultirateError("view tensor must be [B,C,T,even-H,even-W]")
    if band not in {"LL", "HH"}:
        raise MultirateError(f"unsupported Haar projector: {band}")
    batch, channels, frames, height, width = value.shape
    blocks = value.float().reshape(
        batch, channels, frames, height // 2, 2, width // 2, 2
    )
    if band == "LL":
        projected = blocks.mean(dim=(4, 6), keepdim=True).expand_as(blocks)
    else:
        signs = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]],
            device=value.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 1, 2, 1, 2)
        coefficient = (blocks * signs).mean(dim=(4, 6), keepdim=True)
        projected = (coefficient * signs).expand_as(blocks)
    return projected.reshape_as(value).float()


def future_spatial_project(
    value: Any,
    *,
    history_frames: int,
    band: str,
    view_count: int = VIEW_COUNT,
) -> Any:
    """Return a future-only FP32 projector without crossing camera views."""
    import torch

    future = _future_only(value, history_frames)
    if view_count <= 0 or future.shape[-1] % view_count:
        raise MultirateError("latent width is not divisible by the view count")
    view_width = int(future.shape[-1]) // int(view_count)
    if view_width % 2:
        raise MultirateError("per-view latent width must be even")
    projected_views = []
    for view in range(view_count):
        left = view * view_width
        right = left + view_width
        projected_views.append(
            _project_view(future[:, :, history_frames:, :, left:right], band)
        )
    result = torch.zeros_like(future, dtype=torch.float32)
    result[:, :, history_frames:] = torch.cat(projected_views, dim=-1)
    return result


def future_spatial_complement(
    value: Any,
    *,
    history_frames: int,
    band: str,
    view_count: int = VIEW_COUNT,
) -> Any:
    return _future_only(value, history_frames) - future_spatial_project(
        value,
        history_frames=history_frames,
        band=band,
        view_count=view_count,
    )


def projector_contract(
    value: Any,
    *,
    history_frames: int,
    band: str,
    view_count: int = VIEW_COUNT,
) -> dict[str, Any]:
    """Fail closed on reconstruction, orthogonality, support, and isolation."""
    import torch

    target = _future_only(value, history_frames)
    projected = future_spatial_project(
        value,
        history_frames=history_frames,
        band=band,
        view_count=view_count,
    )
    complement = target - projected
    reconstructed = projected + complement
    energy = target.square().sum().clamp_min(1e-30)
    reconstruction = reconstructed - target
    max_abs = float(reconstruction.abs().max().detach().cpu())
    relative_energy = float(
        (reconstruction.square().sum() / energy).detach().cpu()
    )
    inner = float(
        ((projected * complement).sum().abs() / energy).detach().cpu()
    )
    reprojection = future_spatial_project(
        projected,
        history_frames=history_frames,
        band=band,
        view_count=view_count,
    )
    idempotence = float((reprojection - projected).abs().max().detach().cpu())
    history_nonzero = int(
        torch.count_nonzero(projected[:, :, :history_frames]).detach().cpu()
    ) + int(
        torch.count_nonzero(complement[:, :, :history_frames]).detach().cpu()
    )

    view_width = int(value.shape[-1]) // int(view_count)
    mutated = value.float().clone()
    mutated[..., :view_width] += 7.0
    mutated_projection = future_spatial_project(
        mutated,
        history_frames=history_frames,
        band=band,
        view_count=view_count,
    )
    view_isolated = torch.equal(
        projected[..., view_width:], mutated_projection[..., view_width:]
    )
    receipt = {
        "band": band,
        "rank_fraction": 0.25,
        "max_abs_reconstruction_error": max_abs,
        "relative_reconstruction_error_energy": relative_energy,
        "normalized_projected_complement_inner_product": inner,
        "max_abs_idempotence_error": idempotence,
        "history_nonzero": history_nonzero,
        "view_isolation_bit_exact": bool(view_isolated),
        "projected_energy_fraction": float(
            (projected.square().sum() / energy).detach().cpu()
        ),
        "complement_energy_fraction": float(
            (complement.square().sum() / energy).detach().cpu()
        ),
    }
    if (
        max_abs > MAX_ABS_TOLERANCE
        or relative_energy > RELATIVE_ENERGY_TOLERANCE
        or inner > ORTHOGONALITY_TOLERANCE
        or idempotence > IDEMPOTENCE_TOLERANCE
        or history_nonzero != 0
        or not view_isolated
        or not math.isclose(
            receipt["projected_energy_fraction"]
            + receipt["complement_energy_fraction"],
            1.0,
            rel_tol=0.0,
            abs_tol=5e-6,
        )
    ):
        raise MultirateError(f"{band} projector contract failed: {receipt}")
    return receipt


def synthetic_projector_contract() -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(BOOTSTRAP_SEED)
    value = torch.randn((2, *LATENT_SHAPE), generator=generator)
    return {
        band: projector_contract(
            value,
            history_frames=HISTORY_LATENT_FRAMES,
            band=band,
        )
        for band in ("LL", "HH")
    }


@dataclass
class EventLedger:
    """Monotone receipt that makes target-before-endpoint use impossible."""

    events: list[dict[str, Any]] = field(default_factory=list)
    endpoints: set[str] = field(default_factory=set)
    target_constructed: bool = False
    barrier_closed: bool = False

    def record(self, kind: str, **payload: Any) -> int:
        sequence = len(self.events)
        self.events.append({"sequence": sequence, "kind": kind, **payload})
        return sequence

    def require_pre_target(self) -> None:
        if self.target_constructed or self.barrier_closed:
            raise MultirateError("endpoint operation attempted after target barrier")

    def endpoint(self, name: str) -> None:
        self.require_pre_target()
        if name not in ENDPOINTS or name in self.endpoints:
            raise MultirateError(f"endpoint inventory/order differs: {name}")
        self.endpoints.add(name)
        self.record("endpoint_materialized", endpoint=name)

    def close_endpoint_barrier(self) -> None:
        if self.endpoints != set(ENDPOINTS):
            raise MultirateError("cannot close barrier before every endpoint")
        self.barrier_closed = True
        self.record("endpoint_barrier_closed")

    def construct_target(self) -> None:
        if not self.barrier_closed or self.target_constructed:
            raise MultirateError("target construction ordering differs")
        self.target_constructed = True
        self.record("target_constructed")

    def receipt(self) -> dict[str, Any]:
        endpoint_sequences = [
            int(event["sequence"])
            for event in self.events
            if event["kind"] == "endpoint_materialized"
        ]
        target_sequences = [
            int(event["sequence"])
            for event in self.events
            if event["kind"] == "target_constructed"
        ]
        if (
            len(endpoint_sequences) != len(ENDPOINTS)
            or len(target_sequences) != 1
            or min(target_sequences) <= max(endpoint_sequences)
        ):
            raise MultirateError("target did not follow every endpoint")
        return {
            "events": self.events,
            "endpoint_count": len(endpoint_sequences),
            "first_target_sequence": target_sequences[0],
            "last_endpoint_sequence": max(endpoint_sequences),
            "target_after_all_endpoints": True,
            "sha256": hashlib.sha256(
                json.dumps(
                    self.events,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }


def _history_at_sigma(
    reference: Any, initial: Any, history_frames: int, sigma: Any
) -> Any:
    value = float(sigma)
    return (
        (1.0 - value) * reference[:, :, :history_frames].float()
        + value * initial[:, :, :history_frames].float()
    )


def _clamp_history(
    state: Any,
    *,
    reference: Any,
    initial: Any,
    history_frames: int,
    sigma: Any,
) -> Any:
    result = state.float().clone()
    result[:, :, :history_frames] = _history_at_sigma(
        reference, initial, history_frames, sigma
    )
    return result


def _euler(state: Any, velocity: Any, sigma: Any, next_sigma: Any) -> Any:
    from robot_wm.modeling.dual_diffusion.flow import euler_flow_step

    return euler_flow_step(
        state.float(), velocity.float(), sigma.float(), next_sigma.float()
    )


def _time_reversed_future(value: Any, history_frames: int) -> Any:
    result = value.float().clone()
    future = result[:, :, history_frames:]
    if future.shape[2] != 2:
        raise MultirateError("time-reversal control requires two future tokens")
    result[:, :, history_frames:] = future.flip(dims=(2,))
    return result


def _timed_value(function: Callable[[], Any]) -> tuple[Any, float]:
    """Synchronize CUDA events when available; use a host timer in unit tests."""
    import torch

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = function()
        end.record()
        end.synchronize()
        return value, float(start.elapsed_time(end))
    started = time.perf_counter()
    value = function()
    return value, 1000.0 * (time.perf_counter() - started)


@dataclass
class EndpointMaterialization:
    states: dict[str, Any]
    midpoints: dict[str, Any]
    first_velocity: Any
    first_clean_estimate: Any
    locked_states: dict[str, Any]
    conceptual_wan_calls: dict[str, int]
    actual_wan_calls: int
    projection_ms: dict[str, float]
    wan_ms: dict[str, float]
    projector_receipts: dict[str, dict[str, Any]]
    synchronous_identity_max_abs: float


def materialize_split_endpoints(
    *,
    initial: Any,
    reference: Any,
    history_frames: int,
    sigmas: Any,
    timesteps: Sequence[Any],
    velocity_call: Callable[[Any, Any, str, Any], Any],
    ledger: EventLedger,
    view_count: int = VIEW_COUNT,
) -> EndpointMaterialization:
    """Materialize all endpoints before any clean target can be constructed."""
    import torch

    ledger.require_pre_target()
    if len(timesteps) != 2 or sigmas.numel() != 3:
        raise MultirateError("ILSF-2 requires exactly two timesteps/three sigmas")
    s0, s1, s2 = (sigmas[index].float() for index in range(3))
    if (
        not math.isclose(float(s0), 1.0, rel_tol=0.0, abs_tol=1e-8)
        or not 0.0 < float(s1) < 1.0
        or not math.isclose(float(s2), 0.0, rel_tol=0.0, abs_tol=1e-8)
    ):
        raise MultirateError("native NFE-2 sigma schedule differs")

    wan_ms: dict[str, float] = {}
    first_velocity, wan_ms["SHARED_FIRST"] = _timed_value(
        lambda: velocity_call(initial, timesteps[0], "SHARED_FIRST", s0)
    )
    if first_velocity.shape != initial.shape:
        raise MultirateError("shared first velocity shape differs")
    ledger.record("wan_call", call="SHARED_FIRST")
    first_clean = initial.float() - s0 * first_velocity.float()
    first_clean = _clamp_history(
        first_clean,
        reference=reference,
        initial=initial,
        history_frames=history_frames,
        sigma=s2,
    )
    standard_midpoint = _euler(initial, first_velocity, s0, s1)
    standard_midpoint = _clamp_history(
        standard_midpoint,
        reference=reference,
        initial=initial,
        history_frames=history_frames,
        sigma=s1,
    )
    ledger.endpoint("VPM1")

    projector_receipts = {
        band: projector_contract(
            first_clean,
            history_frames=history_frames,
            band=band,
            view_count=view_count,
        )
        for band in ("LL", "HH")
    }
    projection_ms: dict[str, float] = {}

    ll_clean, ll_ms = _timed_value(
        lambda: future_spatial_project(
            first_clean,
            history_frames=history_frames,
            band="LL",
            view_count=view_count,
        )
    )
    ll_mid_detail, q_ms = _timed_value(
        lambda: future_spatial_complement(
            standard_midpoint,
            history_frames=history_frames,
            band="LL",
            view_count=view_count,
        )
    )
    projection_ms["LL_SHARED_PROJECT"] = ll_ms + q_ms
    ll_shuffled = torch.roll(ll_clean, shifts=1, dims=0)
    ll_reversed = _time_reversed_future(ll_clean, history_frames)
    hh_clean, hh_ms = _timed_value(
        lambda: future_spatial_project(
            first_clean,
            history_frames=history_frames,
            band="HH",
            view_count=view_count,
        )
    )
    hh_mid_detail, hh_q_ms = _timed_value(
        lambda: future_spatial_complement(
            standard_midpoint,
            history_frames=history_frames,
            band="HH",
            view_count=view_count,
        )
    )
    projection_ms["HH_SHARED_PROJECT"] = hh_ms + hh_q_ms

    midpoints: dict[str, Any] = {
        "VPM2_ORDINARY": standard_midpoint,
        "LL_FIRST_ALIGNED": ll_clean + ll_mid_detail,
        "LL_FIRST_EPISODE_SHUFFLED": ll_shuffled + ll_mid_detail,
        "LL_FIRST_TIME_REVERSED": ll_reversed + ll_mid_detail,
        "HH_FIRST_RANK_MATCHED": hh_clean + hh_mid_detail,
    }
    for endpoint in TWO_CALL_ENDPOINTS:
        midpoints[endpoint] = _clamp_history(
            midpoints[endpoint],
            reference=reference,
            initial=initial,
            history_frames=history_frames,
            sigma=s1,
        )

    locked_states = {
        "LL_FIRST_ALIGNED": ll_clean,
        "LL_FIRST_EPISODE_SHUFFLED": ll_shuffled,
        "LL_FIRST_TIME_REVERSED": ll_reversed,
        "HH_FIRST_RANK_MATCHED": hh_clean,
    }
    states: dict[str, Any] = {"VPM1": first_clean}
    for endpoint in TWO_CALL_ENDPOINTS:
        ledger.require_pre_target()
        second_velocity, wan_ms[endpoint] = _timed_value(
            lambda endpoint=endpoint: velocity_call(
                midpoints[endpoint], timesteps[1], endpoint, s1
            )
        )
        ledger.record("wan_call", call=endpoint)
        native_final = _euler(midpoints[endpoint], second_velocity, s1, s2)
        if endpoint == "VPM2_ORDINARY":
            final = native_final
            projection_ms[endpoint] = 0.0
        else:
            band = "HH" if endpoint == "HH_FIRST_RANK_MATCHED" else "LL"

            def compose_final() -> Any:
                return locked_states[endpoint] + future_spatial_complement(
                    native_final,
                    history_frames=history_frames,
                    band=band,
                    view_count=view_count,
                )

            final, projection_ms[endpoint] = _timed_value(compose_final)
        states[endpoint] = _clamp_history(
            final,
            reference=reference,
            initial=initial,
            history_frames=history_frames,
            sigma=s2,
        )
        ledger.endpoint(endpoint)

    ordinary_future = _future_only(states["VPM2_ORDINARY"], history_frames)
    synchronous = future_spatial_project(
        states["VPM2_ORDINARY"],
        history_frames=history_frames,
        band="LL",
        view_count=view_count,
    ) + future_spatial_complement(
        states["VPM2_ORDINARY"],
        history_frames=history_frames,
        band="LL",
        view_count=view_count,
    )
    synchronous_error = float(
        (synchronous - ordinary_future).abs().max().detach().cpu()
    )
    if synchronous_error > MAX_ABS_TOLERANCE:
        raise MultirateError("synchronous split is not an identity")
    if len(states) != len(ENDPOINTS):
        raise MultirateError("endpoint state inventory differs")
    ledger.close_endpoint_barrier()
    return EndpointMaterialization(
        states=states,
        midpoints=midpoints,
        first_velocity=first_velocity,
        first_clean_estimate=first_clean,
        locked_states=locked_states,
        conceptual_wan_calls={
            endpoint: 1 if endpoint == "VPM1" else 2 for endpoint in ENDPOINTS
        },
        actual_wan_calls=1 + len(TWO_CALL_ENDPOINTS),
        projection_ms=projection_ms,
        wan_ms=wan_ms,
        projector_receipts=projector_receipts,
        synchronous_identity_max_abs=synchronous_error,
    )


def _validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 512:
        raise MultirateError("canonical ABC train manifest must have 512 rows")
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
            raise MultirateError(f"canonical manifest differs at row {index}")
        episodes.add(episode)
        clips.add(clip)
    if set(DEV_NOISE_SEEDS) & set(range(20260820, 20261101)):
        raise MultirateError("endpoint seeds overlap a prior registered seed range")
    if BOOTSTRAP_SEED in DEV_NOISE_SEEDS:
        raise MultirateError("bootstrap seed overlaps endpoint seeds")
    return {
        "ranges": {
            "excluded_prior_used_or_inspected": [0, DEV_RANGE[0]],
            "prior_inspected_exploratory_development": list(DEV_RANGE),
            "forbidden_prior_consumed_reserve": list(FORBIDDEN_RESERVE_RANGE),
            "excluded_constructor_probe": list(CONSTRUCTOR_EXCLUDED_RANGE),
        },
        "development_noise_seeds": list(DEV_NOISE_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "all_episode_and_clip_identities_unique": True,
    }


def _validated_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    return haar._validated_record(record, label)


def _validated_cache_record(
    record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    return haar._validated_cache_record(record, label)


def _prepare_registration(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    repo = ladder.canonical_directory(args.repo_root, "repository")
    observed_commit = ladder.git_output(repo, "rev-parse", "HEAD")
    if observed_commit != args.expected_source_commit or len(observed_commit) != 40:
        raise MultirateError("source commit differs")
    if ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise MultirateError("source repository must be clean")
    ladder.git_output(repo, "merge-base", "--is-ancestor", BASE_COMMIT, observed_commit)

    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise MultirateError("fresh output already exists")
    parent_path = ladder.canonical_file(
        args.parent_registration, "parent ladder registration"
    )
    runtime_path = ladder.canonical_file(args.runtime_record, "B200 runtime record")
    runtime = ladder.read_json(runtime_path)
    devices = runtime.get("gpus", {}).get("devices", [])
    if (
        runtime.get("gpus", {}).get("count") != 1
        or len(devices) != 1
        or "B200" not in str(devices[0].get("name", "")).upper()
        or devices[0].get("capability") != [10, 0]
    ):
        raise MultirateError("runtime receipt is not one B200")

    parent = ladder.read_json(parent_path)
    if (
        parent.get("schema") != ladder.SCHEMA
        or not ladder.identity_valid(parent)
        or "VPM" not in parent.get("lineage", {})
    ):
        raise MultirateError("parent ladder registration is invalid")
    if args.vpm_snapshot_sha256 != parent["inputs"]["vpm_snapshot"].get(
        "sha256"
    ):
        raise MultirateError("VPM snapshot digest is not parent-bound")
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
        raise MultirateError("parent cache array inventory differs")
    vpm_spec = parent["lineage"]["VPM"].get("arm_spec", {})
    if (
        vpm_spec.get("parameter_matched_control") is not True
        or vpm_spec.get("condition_mode") != "off"
    ):
        raise MultirateError("registered parent is not the frozen VPM frontier")

    output.mkdir(parents=True, mode=0o700)
    output = output.resolve(strict=True)
    registration = ladder.identity_payload(
        {
            "schema": REGISTRATION_SCHEMA,
            "created_at_utc": ladder._now(),
            "status": "registered_before_exploratory_outcome_open",
            "repo_root": str(repo),
            "source_commit": observed_commit,
            "source_base_commit": BASE_COMMIT,
            "output_dir": str(output),
            "parent_registration": ladder.file_record(parent_path),
            "parent_registration_identity_sha256": parent["identity_sha256"],
            "runtime_verification": ladder.file_record(runtime_path),
            "inputs": inputs,
            "cache_arrays": arrays,
            "lineage": {"VPM": parent["lineage"]["VPM"]},
            "partitions": partitions,
            "selected_manifest_rows": [
                {
                    "clip_index": index,
                    "clip_id": manifest_rows[index]["clip_id"],
                    "episode_dir": manifest_rows[index]["episode_dir"],
                }
                for index in range(*DEV_RANGE)
            ],
            "frozen_design": {
                "name": "ILSF-2",
                "endpoints": list(ENDPOINTS),
                "two_call_endpoints": list(TWO_CALL_ENDPOINTS),
                "contrasts": [list(value) for value in CONTRASTS],
                "primary_metrics": list(PRIMARY_METRICS),
                "simultaneous_comparison_count": SIMULTANEOUS_COMPARISON_COUNT,
                "bonferroni_one_sided_alpha": BONFERRONI_ONE_SIDED_ALPHA,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "synthetic_projector_contract": synthetic_projector_contract(),
                "thresholds": {
                    "max_abs": MAX_ABS_TOLERANCE,
                    "relative_energy": RELATIVE_ENERGY_TOLERANCE,
                    "orthogonality": ORTHOGONALITY_TOLERANCE,
                    "idempotence": IDEMPOTENCE_TOLERANCE,
                    "projection_p95_ms": PROJECTION_P95_LIMIT_MS,
                    "e2e_overhead_percent": E2E_OVERHEAD_LIMIT_PERCENT,
                },
                "batch_size": BATCH_SIZE,
                "view_count": VIEW_COUNT,
                "view_width": VIEW_WIDTH,
                "latent_shape": list(LATENT_SHAPE),
                "history_latent_frames": HISTORY_LATENT_FRAMES,
                "projectors": {
                    "LL": "2x2 block constant orthogonal projection",
                    "HH": "2x2 checkerboard orthogonal projection",
                    "rank_fraction": 0.25,
                    "rescale": False,
                    "future_only": True,
                    "view_isolated": True,
                    "compute_dtype": "float32",
                },
            },
            "access_contract": {
                "vjepa_target_array_allowed": False,
                "teacher_calls": 0,
                "feature_encoder_calls": 0,
                "optimizer_updates": 0,
                "new_parameters": 0,
                "future_validity_enabled": False,
                "future_validity_max_retries": 0,
                "prior_inspected_development_opened_during_registration": False,
                "fresh_reserve_480_510_opened": False,
                "constructor_probe_511_opened": False,
                "validation_opened": False,
                "protected_test_opened": False,
                "wandb_enabled": False,
            },
            "claim_boundary": (
                "post-selection exploratory reuse of prior-inspected ABC-train "
                "rows 416--479; frozen VPM, no fit, no clean future serving "
                "feature, no reserve/validation/test/generalization/paper claim"
            ),
        }
    )
    ladder.exclusive_json(output / "registration.json", registration)
    return output, registration


def _load_registration(output: Path) -> dict[str, Any]:
    value = ladder.read_json(output / "registration.json")
    if (
        value.get("schema") != REGISTRATION_SCHEMA
        or not ladder.identity_valid(value)
        or Path(value.get("output_dir", "")).resolve(strict=True) != output
    ):
        raise MultirateError("registration identity/schema/output differs")
    repo = ladder.canonical_directory(value["repo_root"], "registered repository")
    if (
        ladder.git_output(repo, "rev-parse", "HEAD") != value["source_commit"]
        or ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise MultirateError("registered source state changed")
    for key, label in (
        ("runtime_verification", "runtime receipt"),
        ("parent_registration", "parent registration"),
    ):
        record = value.get(key, {})
        observed = _validated_record(record, label)
        if observed != {
            name: record[name] for name in ("path", "bytes", "sha256")
        }:
            raise MultirateError(f"registered {label} changed")
    return value


def _resolve_input(registration: Mapping[str, Any], key: str) -> Path:
    return Path(_validated_record(registration["inputs"][key], key)["path"])


class _IndexAuditedDataset:
    def __init__(self, dataset: Any, allowed: Sequence[int]) -> None:
        self.dataset = dataset
        self.allowed = frozenset(int(index) for index in allowed)
        self.access_counts = {index: 0 for index in self.allowed}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        index = int(index)
        if index not in self.allowed:
            raise MultirateError(f"dataset attempted forbidden row {index}")
        self.access_counts[index] += 1
        return self.dataset[index]

    def assert_exact_accesses(self, expected_per_row: int) -> None:
        if any(value != expected_per_row for value in self.access_counts.values()):
            raise MultirateError(
                f"dataset row-access inventory differs: {self.access_counts}"
            )


def _phase_dataset(
    config: Any, registration: Mapping[str, Any]
) -> _IndexAuditedDataset:
    dataset = ladder._no_auxiliary_dataset(
        config,
        registration,
        validation_sample_indices=(DEV_RANGE[0], DEV_RANGE[1] - 1),
    )
    return _IndexAuditedDataset(dataset, range(*DEV_RANGE))


def _parameter_schema(model: Any) -> dict[str, Any]:
    inventory = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
    ]
    encoded = json.dumps(
        inventory, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "parameter_count": sum(item["numel"] for item in inventory),
        "tensor_count": len(inventory),
        "schema_sha256": hashlib.sha256(encoded).hexdigest(),
        "new_parameters": 0,
    }


def _prepare_serving_inputs(
    model: Any,
    batch: Mapping[str, Any],
    *,
    noise_seed: int,
    ledger: EventLedger,
) -> dict[str, Any]:
    """Build NFE-2 inputs from observed history/actions only."""
    import torch

    ledger.require_pre_target()
    rgb = batch["rgb"]
    if rgb.shape[0] != BATCH_SIZE:
        raise MultirateError("episode-shuffled control requires batch size two")
    history_rgb = rgb[:, : model.num_history_frames]
    ledger.record("history_encode_started")
    history_latents = model._encode_clip(history_rgb).to(rgb.dtype)
    ledger.record("history_encode_finished")
    if history_latents.shape[2] != model.num_history_latent:
        raise MultirateError("deployable history latent length differs")
    latent_frames = model.rgb_tokenizer.latent_temporal_len(
        model.num_history_frames + model.num_future_frames
    )
    video_shape = (
        rgb.shape[0],
        history_latents.shape[1],
        latent_frames,
        history_latents.shape[3],
        history_latents.shape[4],
    )
    if tuple(video_shape[1:]) != LATENT_SHAPE:
        raise MultirateError(f"pinned VPM latent shape differs: {video_shape}")
    history_frames = int(history_latents.shape[2])
    if history_frames != HISTORY_LATENT_FRAMES:
        raise MultirateError("pinned latent history boundary differs")
    reference = history_latents.new_zeros(video_shape)
    reference[:, :, :history_frames] = history_latents
    if model._auxiliary_history_frames(history_frames) != 0:
        raise MultirateError("deployable VPM must diffuse all auxiliary slots")
    _, z_control, _ = model._latent_actions(
        history_rgb,
        batch["actions"],
        batch["morphology_index"],
        latent_frames,
        history_frames,
    )
    z_control = z_control.to(rgb.dtype)
    context = model._build_context(rgb.shape[0], rgb.device, rgb.dtype)
    clip_fea = model._build_clip(rgb.shape[0], rgb.device, rgb.dtype)
    initial_video = model._evaluation_noise(
        video_shape,
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
        *video_shape[2:],
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

    schedule, timesteps, tf_only_steps = model._sampling_schedule(
        2, device=rgb.device
    )
    if int(tf_only_steps) != 0 or len(timesteps) != 2:
        raise MultirateError("VPM NFE-2 is not the aligned native schedule")
    sigmas = schedule.video.detach().clone().to(device=rgb.device, dtype=torch.float32)
    if sigmas.numel() != 3:
        raise MultirateError("VPM NFE-2 sigma inventory differs")

    # The first NFE-2 call must be the actual VPM@1 frontier call.
    one_step_scheduler = copy.deepcopy(model.sample_scheduler)
    one_step_scheduler.set_timesteps(1, device=rgb.device)
    one_sigma = one_step_scheduler.sigmas.to(
        device=rgb.device, dtype=torch.float32
    )[:2]
    if (
        one_sigma.numel() != 2
        or not torch.equal(timesteps[0].float(), one_step_scheduler.timesteps[0].float())
        or not torch.equal(sigmas[0], one_sigma[0])
        or not torch.equal(sigmas[-1], one_sigma[-1])
    ):
        raise MultirateError("NFE-2 first call is not the VPM@1 frontier call")

    # Prove once per paired batch that the explicit helper used to branch the
    # shared trajectory is identical to the production FlowMatchEuler step.
    native_scheduler = copy.deepcopy(model.sample_scheduler)
    synthetic_state = initial_video[:1, :1, :1, :2, :2].float()
    synthetic_first_velocity = torch.full_like(synthetic_state, 0.125)
    explicit_midpoint = _euler(
        synthetic_state, synthetic_first_velocity, sigmas[0], sigmas[1]
    )
    native_midpoint = native_scheduler.step(
        synthetic_first_velocity,
        timesteps[0],
        synthetic_state,
    ).prev_sample.float()
    synthetic_second_velocity = torch.full_like(synthetic_state, -0.25)
    explicit_final = _euler(
        explicit_midpoint, synthetic_second_velocity, sigmas[1], sigmas[2]
    )
    native_final = native_scheduler.step(
        synthetic_second_velocity,
        timesteps[1],
        native_midpoint,
    ).prev_sample.float()
    native_euler_error = max(
        float((native_midpoint - explicit_midpoint).abs().max().detach().cpu()),
        float((native_final - explicit_final).abs().max().detach().cpu()),
    )
    if native_euler_error > MAX_ABS_TOLERANCE:
        raise MultirateError("explicit Euler branch differs from native scheduler")
    ledger.record(
        "serving_inputs_ready",
        noise_seed=int(noise_seed),
        sigma_nodes=[float(value) for value in sigmas.detach().cpu()],
        native_euler_equivalence_max_abs=native_euler_error,
    )
    return {
        "history_rgb": history_rgb,
        "reference": reference,
        "history_frames": history_frames,
        "z_control": z_control,
        "context": context,
        "clip_fea": clip_fea,
        "initial_video": initial_video,
        "initial_auxiliary": initial_auxiliary,
        "sigmas": sigmas,
        "timesteps": tuple(timesteps),
        "native_euler_equivalence_max_abs": native_euler_error,
    }


def _off_velocity_call(
    model: Any,
    prepared: Mapping[str, Any],
    *,
    state: Any,
    timestep: Any,
    sigma: Any,
) -> Any:
    import torch
    from robot_wm.modeling.networks.wan_forward_model import DualWanOutput

    batch_size = int(state.shape[0])
    prediction = model.forward_model(
        state,
        timestep.expand(batch_size).to(state.device),
        prepared["z_control"],
        prepared["reference"],
        prepared["context"],
        prepared["clip_fea"],
        noisy_tf=prepared["initial_auxiliary"],
        conditioning_tf=prepared["initial_auxiliary"],
        tf_sigma=sigma.expand(batch_size).to(device=state.device, dtype=state.dtype),
        condition_on_tf=False,
        condition_on_tf_clock=False,
    )
    if not isinstance(prediction, DualWanOutput):
        raise MultirateError("frozen VPM did not return dual velocities")
    if not torch.isfinite(prediction.video_velocity).all():
        raise MultirateError("VPM video velocity is non-finite")
    return prediction.video_velocity


def _construct_targets_after_barrier(
    model: Any,
    batch: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
    ledger: EventLedger,
) -> dict[str, Any]:
    """This is the only function allowed to encode the full clean clip."""
    ledger.construct_target()
    video_clean = model._encode_clip(batch["rgb"]).to(batch["rgb"].dtype)
    if video_clean.shape != prepared["initial_video"].shape:
        raise MultirateError("clean target latent grid differs")
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
    ladder._validate_decoded_horizon(
        model_future_frames=model.num_future_frames,
        decoded_frames=decoded_clean.shape[2],
        raw_frames=batch["rgb"].shape[1],
        endpoint="TARGET",
    )
    return {
        "video_clean": video_clean,
        "raw_target_exact": raw_target_exact,
        "raw_history_exact": raw_history_exact,
        "raw_target": raw_target,
        "raw_history": raw_history,
        "vae_target": ladder._to_uint8_video(
            decoded_clean[:, :, -model.num_future_frames :]
        ),
        "vae_history": ladder._to_uint8_video(
            decoded_clean[
                :,
                :,
                -(model.num_future_frames + 1) : -model.num_future_frames,
            ]
        ),
    }


def _per_sample_metrics(
    *,
    endpoint: str,
    final: Any,
    locked: Any | None,
    target: Mapping[str, Any],
    decoded: Any,
    history_frames: int,
) -> dict[str, Any]:
    import torch

    clean = target["video_clean"].float()
    prediction = final.float()
    future_prediction = prediction[:, :, history_frames:]
    future_clean = clean[:, :, history_frames:]
    latent_nmse = (
        (future_prediction - future_clean).square().flatten(1).sum(1)
        / future_clean.square().flatten(1).sum(1).clamp_min(1e-12)
    )
    decoded_mse, temporal_mse = ladder._decoded_metrics(
        decoded,
        target["raw_target"],
        prediction_history=target["raw_history"],
        target_history=target["raw_history"],
    )
    ll_prediction = future_spatial_project(
        prediction, history_frames=history_frames, band="LL"
    )[:, :, history_frames:]
    ll_clean = future_spatial_project(
        clean, history_frames=history_frames, band="LL"
    )[:, :, history_frames:]
    q_prediction = future_spatial_complement(
        prediction, history_frames=history_frames, band="LL"
    )[:, :, history_frames:]
    q_clean = future_spatial_complement(
        clean, history_frames=history_frames, band="LL"
    )[:, :, history_frames:]
    ll_nmse = (
        (ll_prediction - ll_clean).square().flatten(1).sum(1)
        / ll_clean.square().flatten(1).sum(1).clamp_min(1e-12)
    )
    q_nmse = (
        (q_prediction - q_clean).square().flatten(1).sum(1)
        / q_clean.square().flatten(1).sum(1).clamp_min(1e-12)
    )
    ll_delta_prediction = ll_prediction.diff(dim=2).flatten(1)
    ll_delta_clean = ll_clean.diff(dim=2).flatten(1)
    ll_delta_mse = (ll_delta_prediction - ll_delta_clean).square().mean(1)
    ll_delta_cosine = torch.nn.functional.cosine_similarity(
        ll_delta_prediction,
        ll_delta_clean,
        dim=1,
        eps=1e-12,
    )
    total_energy = future_prediction.square().flatten(1).sum(1).clamp_min(1e-12)
    hh_prediction = future_spatial_project(
        prediction, history_frames=history_frames, band="HH"
    )[:, :, history_frames:]
    if locked is None:
        p_lock = torch.zeros(
            prediction.shape[0], device=prediction.device, dtype=torch.float32
        )
    else:
        band = "HH" if endpoint == "HH_FIRST_RANK_MATCHED" else "LL"
        observed = future_spatial_project(
            prediction, history_frames=history_frames, band=band
        )
        p_lock = (
            observed[:, :, history_frames:] - locked[:, :, history_frames:].float()
        ).abs().flatten(1).amax(1)
    return {
        "video_future_nmse": latent_nmse,
        "decoded_mse_unit_range": decoded_mse,
        "decoded_temporal_difference_mse_unit_range": temporal_mse,
        "ll_future_nmse": ll_nmse,
        "ll_complement_future_nmse": q_nmse,
        "ll_temporal_delta_mse": ll_delta_mse,
        "ll_temporal_delta_cosine": ll_delta_cosine,
        "ll_prediction_energy_fraction": (
            ll_prediction.square().flatten(1).sum(1) / total_energy
        ),
        "hh_prediction_energy_fraction": (
            hh_prediction.square().flatten(1).sum(1) / total_energy
        ),
        "p_lock_max_abs_error": p_lock,
    }


def _aggregate_projector_receipts(
    receipts: Sequence[Mapping[str, Any]], band: str
) -> dict[str, Any]:
    selected = [value[band] for value in receipts]
    if not selected:
        raise MultirateError("no projector receipts were materialized")
    return {
        "band": band,
        "checked_batches": len(selected),
        "rank_fraction": 0.25,
        "max_abs_reconstruction_error": max(
            float(value["max_abs_reconstruction_error"]) for value in selected
        ),
        "max_relative_reconstruction_error_energy": max(
            float(value["relative_reconstruction_error_energy"])
            for value in selected
        ),
        "max_abs_normalized_projected_complement_inner_product": max(
            float(value["normalized_projected_complement_inner_product"])
            for value in selected
        ),
        "max_abs_idempotence_error": max(
            float(value["max_abs_idempotence_error"]) for value in selected
        ),
        "history_nonzero": sum(int(value["history_nonzero"]) for value in selected),
        "view_isolation_bit_exact": all(
            value["view_isolation_bit_exact"] is True for value in selected
        ),
        "mean_projected_energy_fraction": sum(
            float(value["projected_energy_fraction"]) for value in selected
        )
        / len(selected),
        "mean_complement_energy_fraction": sum(
            float(value["complement_energy_fraction"]) for value in selected
        )
        / len(selected),
    }


def _endpoint_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise MultirateError("CUDA is required")
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    for name in ("endpoint_rows.jsonl", "timing_rows.jsonl", "endpoint_complete.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise MultirateError(f"endpoint artifact already exists: {name}")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    model, config = ladder._load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    if (
        not getattr(model, "parameter_matched_control", False)
        or getattr(model, "condition_on_tf", True)
        or getattr(model, "condition_on_tf_clock", True)
        or getattr(model, "tf_schedule_mode", None) != "aligned"
    ):
        raise MultirateError("loaded checkpoint runtime is not aligned VPM-off")
    parameter_before = _parameter_schema(model)
    manifest = ladder.read_jsonl(_resolve_input(registration, "train_manifest"))
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    projector_receipts: list[Mapping[str, Any]] = []
    synchronous_errors: list[float] = []
    native_euler_errors: list[float] = []
    actual_wan_calls = 0
    teacher_calls = feature_calls = auxiliary_target_calls = optimizer_updates = 0

    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            batch_ordinal = 0
            for noise_seed in DEV_NOISE_SEEDS:
                for start in range(DEV_RANGE[0], DEV_RANGE[1], BATCH_SIZE):
                    indexes = tuple(range(start, start + BATCH_SIZE))
                    if len(indexes) != BATCH_SIZE:
                        raise MultirateError("development batch is incomplete")
                    if manifest[indexes[0]]["episode_dir"] == manifest[indexes[1]]["episode_dir"]:
                        raise MultirateError("shuffle batch is not episode-disjoint")
                    batch = ladder._batch_samples(dataset, indexes, device)
                    ledger = EventLedger()
                    prepared = _prepare_serving_inputs(
                        model, batch, noise_seed=noise_seed, ledger=ledger
                    )
                    native_euler_errors.append(
                        float(prepared["native_euler_equivalence_max_abs"])
                    )
                    calls: list[str] = []

                    def velocity_call(state: Any, timestep: Any, label: str, sigma: Any) -> Any:
                        ledger.require_pre_target()
                        calls.append(label)
                        return _off_velocity_call(
                            model,
                            prepared,
                            state=state,
                            timestep=timestep,
                            sigma=sigma,
                        )

                    materialized = materialize_split_endpoints(
                        initial=prepared["initial_video"],
                        reference=prepared["reference"],
                        history_frames=prepared["history_frames"],
                        sigmas=prepared["sigmas"],
                        timesteps=prepared["timesteps"],
                        velocity_call=velocity_call,
                        ledger=ledger,
                    )
                    expected_calls = ["SHARED_FIRST", *TWO_CALL_ENDPOINTS]
                    if calls != expected_calls or materialized.actual_wan_calls != 6:
                        raise MultirateError("actual Wan call inventory/order differs")
                    actual_wan_calls += len(calls)
                    projector_receipts.append(materialized.projector_receipts)
                    synchronous_errors.append(materialized.synchronous_identity_max_abs)

                    # No target operation occurs above this line.
                    target = _construct_targets_after_barrier(
                        model,
                        batch,
                        prepared=prepared,
                        ledger=ledger,
                    )
                    ledger_receipt = ledger.receipt()
                    decoded_by_endpoint: dict[str, Any] = {}
                    decoder_ms: dict[str, float] = {}
                    for endpoint in ENDPOINTS:
                        def decode(endpoint: str = endpoint) -> Any:
                            decoded_full = model.rgb_tokenizer.decode_temporal(
                                materialized.states[endpoint].to(batch["rgb"].dtype),
                                out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                            )
                            ladder._validate_decoded_horizon(
                                model_future_frames=model.num_future_frames,
                                decoded_frames=decoded_full.shape[2],
                                raw_frames=batch["rgb"].shape[1],
                                endpoint=endpoint,
                            )
                            return ladder._to_uint8_video(
                                decoded_full[:, :, -model.num_future_frames :]
                            )

                        decoded_by_endpoint[endpoint], decoder_ms[endpoint] = _timed_value(decode)

                    metrics_by_endpoint = {
                        endpoint: _per_sample_metrics(
                            endpoint=endpoint,
                            final=materialized.states[endpoint],
                            locked=materialized.locked_states.get(endpoint),
                            target=target,
                            decoded=decoded_by_endpoint[endpoint],
                            history_frames=prepared["history_frames"],
                        )
                        for endpoint in ENDPOINTS
                    }
                    common_hashes = {
                        "initial_video": prepared["initial_video"],
                        "actions": batch["actions"],
                        "morphology_index": batch["morphology_index"],
                        "action_control": prepared["z_control"],
                        "auxiliary_noise": prepared["initial_auxiliary"],
                        "raw_history_input": prepared["history_rgb"],
                        "history_reference": prepared["reference"],
                        "first_velocity": materialized.first_velocity,
                        "first_clean_estimate": materialized.first_clean_estimate,
                    }
                    target_hashes = {
                        "latent_future_target": target["video_clean"][
                            :, :, prepared["history_frames"] :
                        ],
                        "raw_future_target": target["raw_target_exact"],
                        "raw_history_boundary": target["raw_history_exact"],
                    }
                    for endpoint in ENDPOINTS:
                        metrics = metrics_by_endpoint[endpoint]
                        for local, clip_index in enumerate(indexes):
                            rows.append(
                                ladder.identity_payload(
                                    {
                                        "schema": ROW_SCHEMA,
                                        "endpoint": endpoint,
                                        "clip_index": int(clip_index),
                                        "clip_id": manifest[clip_index]["clip_id"],
                                        "episode_dir": manifest[clip_index]["episode_dir"],
                                        "noise_seed": int(noise_seed),
                                        "conceptual_wan_calls": materialized.conceptual_wan_calls[endpoint],
                                        "shared_first_wan_call": True,
                                        "actual_batch_wan_calls": len(calls),
                                        "teacher_calls": 0,
                                        "feature_encoder_calls": 0,
                                        "auxiliary_target_calls": 0,
                                        "optimizer_updates": 0,
                                        "new_parameters": 0,
                                        "condition_on_tf": False,
                                        "condition_on_tf_clock": False,
                                        "target_after_all_endpoints": True,
                                        "event_ledger_sha256": ledger_receipt["sha256"],
                                        "metrics": {
                                            name: float(value[local].detach().cpu())
                                            for name, value in metrics.items()
                                        },
                                        "tensor_sha256": {
                                            **{
                                                name: ladder._safe_tensor_sha256(
                                                    value[local : local + 1]
                                                )
                                                for name, value in common_hashes.items()
                                            },
                                            **{
                                                name: ladder._safe_tensor_sha256(
                                                    value[local : local + 1]
                                                )
                                                for name, value in target_hashes.items()
                                            },
                                            "midpoint": ladder._safe_tensor_sha256(
                                                (
                                                    materialized.first_clean_estimate
                                                    if endpoint == "VPM1"
                                                    else materialized.midpoints[endpoint]
                                                )[local : local + 1]
                                            ),
                                            "final_video": ladder._safe_tensor_sha256(
                                                materialized.states[endpoint][local : local + 1]
                                            ),
                                        },
                                        "access": {
                                            "prior_inspected_development_opened": True,
                                            "fresh_reserve_480_510_opened": False,
                                            "constructor_probe_511_opened": False,
                                            "validation_opened": False,
                                            "protected_test_opened": False,
                                            "vjepa_target_array_opened": False,
                                        },
                                    }
                                )
                            )

                    ll_shared = materialized.projection_ms["LL_SHARED_PROJECT"]
                    hh_shared = materialized.projection_ms["HH_SHARED_PROJECT"]
                    projection_by_endpoint = {
                        "VPM1": 0.0,
                        "VPM2_ORDINARY": 0.0,
                        "LL_FIRST_ALIGNED": ll_shared
                        + materialized.projection_ms["LL_FIRST_ALIGNED"],
                        "LL_FIRST_EPISODE_SHUFFLED": ll_shared
                        + materialized.projection_ms["LL_FIRST_EPISODE_SHUFFLED"],
                        "LL_FIRST_TIME_REVERSED": ll_shared
                        + materialized.projection_ms["LL_FIRST_TIME_REVERSED"],
                        "HH_FIRST_RANK_MATCHED": hh_shared
                        + materialized.projection_ms["HH_FIRST_RANK_MATCHED"],
                    }
                    composed_ms = {}
                    for endpoint in ENDPOINTS:
                        value = materialized.wan_ms["SHARED_FIRST"] + decoder_ms[endpoint]
                        if endpoint != "VPM1":
                            value += materialized.wan_ms[endpoint]
                        value += projection_by_endpoint[endpoint]
                        composed_ms[endpoint] = value
                    timing_rows.append(
                        ladder.identity_payload(
                            {
                                "schema": TIMING_SCHEMA,
                                "batch_ordinal": batch_ordinal,
                                "noise_seed": int(noise_seed),
                                "clip_indices": list(indexes),
                                "wan_call_order": calls,
                                "actual_wan_calls": len(calls),
                                "event_ledger": ledger_receipt,
                                "latency_ms": {
                                    "shared_first_wan": materialized.wan_ms["SHARED_FIRST"],
                                    "second_wan": {
                                        endpoint: materialized.wan_ms[endpoint]
                                        for endpoint in TWO_CALL_ENDPOINTS
                                    },
                                    "projection": projection_by_endpoint,
                                    "decoder": decoder_ms,
                                    "deployable_composed": composed_ms,
                                },
                            }
                        )
                    )
                    batch_ordinal += 1
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(len(DEV_NOISE_SEEDS))
    if opened != expected_arrays:
        raise MultirateError(f"endpoint input graph differs: {sorted(opened)}")
    parameter_after = _parameter_schema(model)
    if parameter_before != parameter_after:
        raise MultirateError("model parameter schema changed during evaluation")
    expected_batches = (DEV_RANGE[1] - DEV_RANGE[0]) // BATCH_SIZE * len(DEV_NOISE_SEEDS)
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(ENDPOINTS)
    if (
        len(timing_rows) != expected_batches
        or len(rows) != expected_rows
        or actual_wan_calls != expected_batches * 6
        or teacher_calls != 0
        or feature_calls != 0
        or auxiliary_target_calls != 0
        or optimizer_updates != 0
    ):
        raise MultirateError("endpoint count/call inventory differs")

    rows_path = output / "endpoint_rows.jsonl"
    timing_path = output / "timing_rows.jsonl"
    ladder.exclusive_jsonl(rows_path, rows)
    ladder.exclusive_jsonl(timing_path, timing_rows)
    receipt = ladder.identity_payload(
        {
            "schema": ENDPOINT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "rows": ladder.file_record(rows_path),
            "timing_rows": ladder.file_record(timing_path),
            "row_count": len(rows),
            "timing_row_count": len(timing_rows),
            "development_clip_range": list(DEV_RANGE),
            "development_noise_seeds": list(DEV_NOISE_SEEDS),
            "batch_size": BATCH_SIZE,
            "actual_wan_calls": actual_wan_calls,
            "actual_decoder_calls": expected_batches * len(ENDPOINTS),
            "conceptual_calls": {
                endpoint: 1 if endpoint == "VPM1" else 2 for endpoint in ENDPOINTS
            },
            "first_wan_call_shared": True,
            "projector_contract": {
                band: _aggregate_projector_receipts(projector_receipts, band)
                for band in ("LL", "HH")
            },
            "synchronous_identity_max_abs": max(synchronous_errors),
            "native_scheduler_euler_equivalence_max_abs": max(native_euler_errors),
            "parameter_schema_before": parameter_before,
            "parameter_schema_after": parameter_after,
            "teacher_calls": 0,
            "feature_encoder_calls": 0,
            "auxiliary_target_calls": 0,
            "optimizer_updates": 0,
            "target_after_all_endpoints": True,
            "exact_dataset_accesses_per_row": len(DEV_NOISE_SEEDS),
            "opened_numpy_arrays": [str(path) for path in sorted(opened)],
            "vjepa_target_array_opened": False,
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "endpoint_complete.json", receipt)
    print(
        json.dumps(
            {
                "endpoint_identity_sha256": receipt["identity_sha256"],
                "rows": len(rows),
                "wan_calls": actual_wan_calls,
            },
            sort_keys=True,
        )
    )
    return 0


def _validated_endpoint(
    output: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint = ladder.read_json(output / "endpoint_complete.json")
    expected_batches = (DEV_RANGE[1] - DEV_RANGE[0]) // BATCH_SIZE * len(DEV_NOISE_SEEDS)
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(ENDPOINTS)
    if (
        endpoint.get("schema") != ENDPOINT_SCHEMA
        or not ladder.identity_valid(endpoint)
        or endpoint.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or int(endpoint.get("row_count", -1)) != expected_rows
        or int(endpoint.get("timing_row_count", -1)) != expected_batches
        or int(endpoint.get("actual_wan_calls", -1)) != expected_batches * 6
        or int(endpoint.get("actual_decoder_calls", -1))
        != expected_batches * len(ENDPOINTS)
        or endpoint.get("conceptual_calls")
        != {endpoint: 1 if endpoint == "VPM1" else 2 for endpoint in ENDPOINTS}
        or endpoint.get("first_wan_call_shared") is not True
        or endpoint.get("target_after_all_endpoints") is not True
        or endpoint.get("teacher_calls") != 0
        or endpoint.get("feature_encoder_calls") != 0
        or endpoint.get("auxiliary_target_calls") != 0
        or endpoint.get("optimizer_updates") != 0
        or endpoint.get("vjepa_target_array_opened") is not False
        or endpoint.get("post_selection_exploratory_dev_opened") is not True
        or endpoint.get("fresh_reserve_480_510_opened") is not False
        or endpoint.get("constructor_probe_511_opened") is not False
        or endpoint.get("validation_opened") is not False
        or endpoint.get("protected_test_opened") is not False
        or endpoint.get("parameter_schema_before")
        != endpoint.get("parameter_schema_after")
        or endpoint.get("parameter_schema_after", {}).get("new_parameters") != 0
        or float(endpoint.get("synchronous_identity_max_abs", math.inf))
        > MAX_ABS_TOLERANCE
        or float(
            endpoint.get("native_scheduler_euler_equivalence_max_abs", math.inf)
        )
        > MAX_ABS_TOLERANCE
    ):
        raise MultirateError("endpoint completion contract differs")
    projector = endpoint.get("projector_contract", {})
    if set(projector) != {"LL", "HH"}:
        raise MultirateError("projector receipt inventory differs")
    for band in ("LL", "HH"):
        receipt = projector[band]
        if (
            receipt.get("band") != band
            or int(receipt.get("checked_batches", -1)) != expected_batches
            or float(receipt.get("rank_fraction", -1.0)) != 0.25
            or float(receipt.get("max_abs_reconstruction_error", math.inf))
            > MAX_ABS_TOLERANCE
            or float(
                receipt.get("max_relative_reconstruction_error_energy", math.inf)
            )
            > RELATIVE_ENERGY_TOLERANCE
            or float(
                receipt.get(
                    "max_abs_normalized_projected_complement_inner_product",
                    math.inf,
                )
            )
            > ORTHOGONALITY_TOLERANCE
            or float(receipt.get("max_abs_idempotence_error", math.inf))
            > IDEMPOTENCE_TOLERANCE
            or int(receipt.get("history_nonzero", -1)) != 0
            or receipt.get("view_isolation_bit_exact") is not True
        ):
            raise MultirateError(f"sealed {band} projector receipt differs")

    loaded = []
    for key, filename, label in (
        ("rows", "endpoint_rows.jsonl", "endpoint rows"),
        ("timing_rows", "timing_rows.jsonl", "timing rows"),
    ):
        record = endpoint.get(key, {})
        path = ladder.canonical_file(record.get("path", ""), label)
        if (
            path != output / filename
            or path.stat().st_size != int(record.get("bytes", -1))
            or ladder.sha256_file(path) != record.get("sha256")
        ):
            raise MultirateError(f"{label} artifact differs")
        loaded.append(ladder.read_jsonl(path))
    rows, timings = loaded
    if len(rows) != expected_rows or len(timings) != expected_batches:
        raise MultirateError("sealed row count differs")
    expected_timing = [
        (seed, [start, start + 1])
        for seed in DEV_NOISE_SEEDS
        for start in range(DEV_RANGE[0], DEV_RANGE[1], BATCH_SIZE)
    ]
    for ordinal, (row, (seed, clips)) in enumerate(
        zip(timings, expected_timing, strict=True)
    ):
        latency = row.get("latency_ms", {})
        ledger = row.get("event_ledger", {})
        if (
            row.get("schema") != TIMING_SCHEMA
            or not ladder.identity_valid(row)
            or int(row.get("batch_ordinal", -1)) != ordinal
            or int(row.get("noise_seed", -1)) != seed
            or row.get("clip_indices") != clips
            or row.get("wan_call_order") != ["SHARED_FIRST", *TWO_CALL_ENDPOINTS]
            or int(row.get("actual_wan_calls", -1)) != 6
            or ledger.get("target_after_all_endpoints") is not True
            or int(ledger.get("endpoint_count", -1)) != len(ENDPOINTS)
            or int(ledger.get("first_target_sequence", -1))
            <= int(ledger.get("last_endpoint_sequence", math.inf))
            or set(latency.get("second_wan", {})) != set(TWO_CALL_ENDPOINTS)
            or set(latency.get("projection", {})) != set(ENDPOINTS)
            or set(latency.get("decoder", {})) != set(ENDPOINTS)
            or set(latency.get("deployable_composed", {})) != set(ENDPOINTS)
        ):
            raise MultirateError("timing/event ledger inventory differs")
        scalar_values = [latency.get("shared_first_wan", math.nan)]
        for group in ("second_wan", "projection", "decoder", "deployable_composed"):
            scalar_values.extend(latency[group].values())
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in scalar_values):
            raise MultirateError("timing payload is non-finite or negative")
    return endpoint, rows, timings


def _bootstrap_relative(
    control: Any,
    candidate: Any,
    *,
    bootstrap_indexes: Any,
) -> dict[str, Any]:
    import numpy as np

    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    expected = (DEV_RANGE[1] - DEV_RANGE[0], len(DEV_NOISE_SEEDS))
    if (
        control.shape != expected
        or candidate.shape != expected
        or not np.isfinite(control).all()
        or not np.isfinite(candidate).all()
        or control.mean() <= 0
        or bootstrap_indexes.shape
        != (BOOTSTRAP_REPLICATES, DEV_RANGE[1] - DEV_RANGE[0])
    ):
        raise MultirateError("paired bootstrap input differs")
    point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    controls = control[bootstrap_indexes].mean(axis=(1, 2))
    candidates = candidate[bootstrap_indexes].mean(axis=(1, 2))
    effects = 100.0 * (controls - candidates) / controls
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": float(point),
        "paired_episode_bootstrap_95_percent": [
            float(value) for value in np.quantile(effects, (0.025, 0.975))
        ],
        "paired_episode_bootstrap_bonferroni_one_sided_lower_percent": float(
            np.quantile(effects, BONFERRONI_ONE_SIDED_ALPHA)
        ),
        "favorable_episode_fraction": float(
            np.mean(candidate.mean(axis=1) < control.mean(axis=1))
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "episode; four noise seeds remain clustered",
    }


def _analysis_payload(
    rows: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
    *,
    endpoint_receipt: Mapping[str, Any],
    registered_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    manifest_lookup = {
        int(value["clip_index"]): (value["clip_id"], value["episode_dir"])
        for value in registered_samples
    }
    if set(manifest_lookup) != set(range(*DEV_RANGE)):
        raise MultirateError("registered sample inventory differs")
    expected_keys = {
        (endpoint, clip, seed)
        for endpoint in ENDPOINTS
        for clip in range(*DEV_RANGE)
        for seed in DEV_NOISE_SEEDS
    }
    inventory: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    required_hashes = {
        "initial_video",
        "actions",
        "morphology_index",
        "action_control",
        "auxiliary_noise",
        "raw_history_input",
        "history_reference",
        "first_velocity",
        "first_clean_estimate",
        "latent_future_target",
        "raw_future_target",
        "raw_history_boundary",
        "midpoint",
        "final_video",
    }
    common_hashes = required_hashes - {"midpoint", "final_video"}
    for row in rows:
        key = (
            str(row.get("endpoint", "")),
            int(row.get("clip_index", -1)),
            int(row.get("noise_seed", -1)),
        )
        metrics = row.get("metrics", {})
        hashes = row.get("tensor_sha256", {})
        access = row.get("access", {})
        expected_calls = 1 if key[0] == "VPM1" else 2
        if (
            row.get("schema") != ROW_SCHEMA
            or not ladder.identity_valid(row)
            or key not in expected_keys
            or key in inventory
            or (row.get("clip_id"), row.get("episode_dir"))
            != manifest_lookup[key[1]]
            or int(row.get("conceptual_wan_calls", -1)) != expected_calls
            or row.get("shared_first_wan_call") is not True
            or int(row.get("actual_batch_wan_calls", -1)) != 6
            or row.get("teacher_calls") != 0
            or row.get("feature_encoder_calls") != 0
            or row.get("auxiliary_target_calls") != 0
            or row.get("optimizer_updates") != 0
            or row.get("new_parameters") != 0
            or row.get("condition_on_tf") is not False
            or row.get("condition_on_tf_clock") is not False
            or row.get("target_after_all_endpoints") is not True
            or set(metrics) != set(ALL_METRICS)
            or any(not math.isfinite(float(value)) for value in metrics.values())
            or set(hashes) != required_hashes
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in hashes.values()
            )
            or access.get("prior_inspected_development_opened") is not True
            or access.get("fresh_reserve_480_510_opened") is not False
            or access.get("constructor_probe_511_opened") is not False
            or access.get("validation_opened") is not False
            or access.get("protected_test_opened") is not False
            or access.get("vjepa_target_array_opened") is not False
        ):
            raise MultirateError(f"endpoint row contract differs: {key}")
        inventory[key] = row
    if set(inventory) != expected_keys:
        raise MultirateError("endpoint row inventory differs")
    for clip in range(*DEV_RANGE):
        for seed in DEV_NOISE_SEEDS:
            paired = [inventory[(endpoint, clip, seed)] for endpoint in ENDPOINTS]
            if len({row["event_ledger_sha256"] for row in paired}) != 1:
                raise MultirateError("paired event ledger hashes differ")
            for name in common_hashes:
                if len({row["tensor_sha256"][name] for row in paired}) != 1:
                    raise MultirateError(
                        f"paired {name} hash differs for {(clip, seed)}"
                    )

    clips = list(range(*DEV_RANGE))
    seeds = list(DEV_NOISE_SEEDS)

    def values(endpoint: str, metric: str) -> Any:
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
                "min": float(values(endpoint, metric).min()),
                "max": float(values(endpoint, metric).max()),
            }
            for metric in ALL_METRICS
        }
        for endpoint in ENDPOINTS
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indexes = rng.integers(
        0,
        len(clips),
        size=(BOOTSTRAP_REPLICATES, len(clips)),
    )
    comparisons = {}
    for control, candidate in CONTRASTS:
        label = f"{candidate}_vs_{control}"
        comparisons[label] = {
            metric: _bootstrap_relative(
                values(control, metric),
                values(candidate, metric),
                bootstrap_indexes=bootstrap_indexes,
            )
            for metric in PRIMARY_METRICS
        }
    if sum(len(value) for value in comparisons.values()) != 15:
        raise MultirateError("simultaneous comparison family is not exactly 15")
    complement_gain = _bootstrap_relative(
        values("VPM1", "ll_complement_future_nmse"),
        values(PRIMARY, "ll_complement_future_nmse"),
        bootstrap_indexes=bootstrap_indexes,
    )

    def comparison(control: str, metric: str) -> Mapping[str, Any]:
        return comparisons[f"{PRIMARY}_vs_{control}"][metric]

    def quality_baseline(control: str) -> dict[str, bool]:
        decoded_temporal = all(
            comparison(control, metric)["relative_improvement_percent"] >= 3.0
            and comparison(control, metric)[
                "paired_episode_bootstrap_bonferroni_one_sided_lower_percent"
            ]
            > 0.0
            and comparison(control, metric)["favorable_episode_fraction"] >= 0.60
            for metric in (
                "decoded_mse_unit_range",
                "decoded_temporal_difference_mse_unit_range",
            )
        )
        latent = comparison(control, "video_future_nmse")
        latent_guardrail = (
            latent["relative_improvement_percent"] >= 0.0
            and latent[
                "paired_episode_bootstrap_bonferroni_one_sided_lower_percent"
            ]
            > -1.0
        )
        return {
            "decoded_and_temporal_three_percent_simultaneous": decoded_temporal,
            "latent_nonnegative_point_lower_above_minus_one": latent_guardrail,
            "all_passed": decoded_temporal and latent_guardrail,
        }

    def specificity(control: str) -> dict[str, bool]:
        passed = all(
            comparison(control, metric)["relative_improvement_percent"] >= 1.0
            and comparison(control, metric)[
                "paired_episode_bootstrap_bonferroni_one_sided_lower_percent"
            ]
            > 0.0
            for metric in (
                "decoded_mse_unit_range",
                "decoded_temporal_difference_mse_unit_range",
            )
        )
        return {"decoded_and_temporal_one_percent_simultaneous": passed, "all_passed": passed}

    vpm2_gate = quality_baseline("VPM2_ORDINARY")
    vpm1_gate = quality_baseline("VPM1")
    shuffled_gate = specificity("LL_FIRST_EPISODE_SHUFFLED")
    reversed_gate = specificity("LL_FIRST_TIME_REVERSED")
    hh_gate = specificity("HH_FIRST_RANK_MATCHED")

    projection_values = {
        endpoint: np.asarray(
            [float(row["latency_ms"]["projection"][endpoint]) for row in timings],
            dtype=np.float64,
        )
        for endpoint in ENDPOINTS
    }
    composed_values = {
        endpoint: np.asarray(
            [
                float(row["latency_ms"]["deployable_composed"][endpoint])
                for row in timings
            ],
            dtype=np.float64,
        )
        for endpoint in ENDPOINTS
    }
    latency = {
        endpoint: {
            "projection_p95_ms": float(np.quantile(projection_values[endpoint], 0.95)),
            "deployable_composed_mean_ms": float(composed_values[endpoint].mean()),
            "deployable_composed_p95_ms": float(
                np.quantile(composed_values[endpoint], 0.95)
            ),
        }
        for endpoint in ENDPOINTS
    }
    overhead = 100.0 * (
        latency[PRIMARY]["deployable_composed_p95_ms"]
        - latency["VPM2_ORDINARY"]["deployable_composed_p95_ms"]
    ) / latency["VPM2_ORDINARY"]["deployable_composed_p95_ms"]
    latency_gate = (
        latency[PRIMARY]["projection_p95_ms"] <= PROJECTION_P95_LIMIT_MS
        and overhead <= E2E_OVERHEAD_LIMIT_PERCENT
    )
    p_lock_gate = aggregates[PRIMARY]["p_lock_max_abs_error"]["max"] <= MAX_ABS_TOLERANCE
    complement_gate = complement_gain["relative_improvement_percent"] >= 3.0
    receipt_gates = {
        "projector_reconstruction_orthogonality_idempotence_view_isolation": True,
        "synchronous_split_identity": True,
        "shared_first_call_and_paired_hashes": True,
        "one_or_two_call_inventory": True,
        "zero_new_parameters": True,
        "target_after_every_endpoint": True,
        "zero_teacher_feature_auxiliary_calls": True,
        "post_selection_dev_only": True,
        "fresh_reserve_480_510_unopened": True,
        "constructor_511_validation_and_test_unopened": True,
    }
    all_passed = (
        vpm2_gate["all_passed"]
        and vpm1_gate["all_passed"]
        and shuffled_gate["all_passed"]
        and reversed_gate["all_passed"]
        and hh_gate["all_passed"]
        and complement_gate
        and p_lock_gate
        and latency_gate
        and all(receipt_gates.values())
    )
    if all_passed:
        decision = "GO_INVERTIBLE_MULTIRATE"
    elif vpm2_gate["all_passed"] and not vpm1_gate["all_passed"]:
        decision = "NO_GO_SOLVER_STABILIZATION_ONLY"
    elif not (shuffled_gate["all_passed"] and reversed_gate["all_passed"]):
        decision = "NO_GO_GENERIC_MIXED_STATE"
    elif not hh_gate["all_passed"]:
        decision = "NO_GO_GENERIC_EARLY_SUBSPACE"
    else:
        decision = "NO_GO_UNTRAINED_MIXED_CLOCK"
    return {
        "development_episode_count": len(clips),
        "noise_seeds_per_episode": len(seeds),
        "simultaneous_family": {
            "contrasts": [list(value) for value in CONTRASTS],
            "metrics": list(PRIMARY_METRICS),
            "test_count": SIMULTANEOUS_COMPARISON_COUNT,
            "familywise_alpha": 0.05,
            "one_sided_bonferroni_alpha": BONFERRONI_ONE_SIDED_ALPHA,
            "shared_bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "aggregates": aggregates,
        "comparisons": comparisons,
        "complement_call1_to_aligned_call2": complement_gain,
        "latency": {
            "endpoints": latency,
            "aligned_vs_vpm2_p95_overhead_percent": float(overhead),
        },
        "gates": {
            "aligned_vs_vpm2": vpm2_gate,
            "aligned_vs_vpm1": vpm1_gate,
            "aligned_vs_episode_shuffled": shuffled_gate,
            "aligned_vs_time_reversed": reversed_gate,
            "aligned_vs_rank_matched_hh": hh_gate,
            "complement_improves_three_percent": complement_gate,
            "p_lock_within_tolerance": p_lock_gate,
            "latency": {
                "projection_p95_at_most_two_ms": latency[PRIMARY][
                    "projection_p95_ms"
                ]
                <= PROJECTION_P95_LIMIT_MS,
                "aligned_p95_overhead_at_most_five_percent": overhead
                <= E2E_OVERHEAD_LIMIT_PERCENT,
                "all_passed": latency_gate,
            },
            "receipt": receipt_gates,
            "all_passed": all_passed,
        },
        "decision": decision,
        "claim_boundary": (
            "post-selection exploratory frozen two-call mixed-clock VPM on "
            "prior-inspected ABC-train rows 416--479; negative rejects only "
            "this untrained schedule; no reserve/validation/test/FVD/speed/paper claim"
        ),
    }


def _analyze_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    if (output / "analysis.json").exists() or (output / "analysis.json").is_symlink():
        raise MultirateError("analysis artifact already exists")
    registration = _load_registration(output)
    endpoint, rows, timings = _validated_endpoint(output, registration)
    core = _analysis_payload(
        rows,
        timings,
        endpoint_receipt=endpoint,
        registered_samples=registration["selected_manifest_rows"],
    )
    analysis = ladder.identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
            **core,
        }
    )
    ladder.exclusive_json(output / "analysis.json", analysis)
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
    if (output / "audit.json").exists() or (output / "audit.json").is_symlink():
        raise MultirateError("audit artifact already exists")
    registration = _load_registration(output)
    endpoint, rows, timings = _validated_endpoint(output, registration)
    analysis = ladder.read_json(output / "analysis.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or not ladder.identity_valid(analysis)
        or analysis.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or analysis.get("endpoint_identity_sha256") != endpoint["identity_sha256"]
    ):
        raise MultirateError("analysis identity chain differs")
    recomputed = _analysis_payload(
        rows,
        timings,
        endpoint_receipt=endpoint,
        registered_samples=registration["selected_manifest_rows"],
    )
    for key, value in recomputed.items():
        if analysis.get(key) != value:
            raise MultirateError(f"audited analysis field differs: {key}")
    artifacts = (
        "registration.json",
        "endpoint_rows.jsonl",
        "timing_rows.jsonl",
        "endpoint_complete.json",
        "analysis.json",
    )
    audit = ladder.identity_payload(
        {
            "schema": AUDIT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": analysis["decision"],
            "artifact_sha256": {
                name: ladder.sha256_file(
                    ladder.canonical_file(output / name, name)
                )
                for name in artifacts
            },
            "row_inventory_recomputed": True,
            "paired_first_call_and_input_hashes_recomputed": True,
            "target_order_and_event_ledgers_recomputed": True,
            "call_and_capacity_inventory_recomputed": True,
            "projector_and_synchronous_identity_receipts_revalidated": True,
            "timing_summary_recomputed": True,
            "fifteen_test_bonferroni_family_recomputed": True,
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "status": "PASS",
        }
    )
    ladder.exclusive_json(output / "audit.json", audit)
    print(
        json.dumps(
            {
                "audit_identity_sha256": audit["identity_sha256"],
                "decision": audit["decision"],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


def _complete_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    if (output / "run_complete.json").exists() or (output / "run_complete.json").is_symlink():
        raise MultirateError("completion artifact already exists")
    registration = _load_registration(output)
    endpoint, rows, _timings = _validated_endpoint(output, registration)
    analysis = ladder.read_json(output / "analysis.json")
    audit = ladder.read_json(output / "audit.json")
    if (
        not ladder.identity_valid(analysis)
        or audit.get("schema") != AUDIT_SCHEMA
        or not ladder.identity_valid(audit)
        or audit.get("status") != "PASS"
        or audit.get("decision") != analysis.get("decision")
        or audit.get("analysis_identity_sha256") != analysis.get("identity_sha256")
    ):
        raise MultirateError("completion identity/audit chain differs")
    complete = ladder.identity_payload(
        {
            "schema": COMPLETE_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "audit_identity_sha256": audit["identity_sha256"],
            "analysis": ladder.file_record(output / "analysis.json"),
            "audit": ladder.file_record(output / "audit.json"),
            "decision": analysis["decision"],
            "development_row_count": len(rows),
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "status": "COMPLETE",
        }
    )
    ladder.exclusive_json(output / "run_complete.json", complete)
    print(
        json.dumps(
            {
                "completion_identity_sha256": complete["identity_sha256"],
                "decision": complete["decision"],
                "status": "COMPLETE",
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--repo-root", required=True)
    register.add_argument("--expected-source-commit", required=True)
    register.add_argument("--parent-registration", required=True)
    register.add_argument("--vpm-snapshot-sha256", required=True)
    register.add_argument("--runtime-record", required=True)
    register.add_argument("--output-dir", required=True)
    for command in ("endpoint", "analyze", "audit", "complete"):
        commands.add_parser(command).add_argument("--output-dir", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "register":
        output, registration = _prepare_registration(args)
        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "registration_identity_sha256": registration["identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "endpoint":
        return _endpoint_phase(args)
    if args.command == "analyze":
        return _analyze_phase(args)
    if args.command == "audit":
        return _audit_phase(args)
    if args.command == "complete":
        return _complete_phase(args)
    raise MultirateError("unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
