#!/usr/bin/env python3
"""Fit train-only action statistics and register the paired AVP screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.networks.action_delta_residual import (  # noqa: E402
    ACTION_DELTA_STATS_SCHEMA,
)
from tools import dual_abc_pilot  # noqa: E402
from tools import two_clock_consistency_evaluate as base  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


SCHEMA_VERSION = 1
KIND_REGISTRATION = "action_variation_registration"
KIND_STATS = ACTION_DELTA_STATS_SCHEMA
PARENT_SNAPSHOT_SHA256 = base.PARENT_SNAPSHOT_SHA256
HISTORICAL_TRAINING_COMMIT = base.TRAINING_COMMIT
EXPECTED_WORLD_SIZE = 8
EXPECTED_VALIDATION_CLIPS = 64
NFE_GRID = (1, 2, 4)
PROTOCOL_PATH = REPO_ROOT / "docs" / "experiments" / "VPM_ACTION_VARIATION_PROTOCOL.md"


class ActionVariationScreenError(RuntimeError):
    """Prospective action-variation evidence changed or is unsafe."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    residual_enabled: bool


ARMS = (
    Arm(
        "AV-CONT",
        "ravenhuang/wan-dit/action_variation_control",
        "action-variation-av-cont-seed1234-u000200",
        False,
    ),
    Arm(
        "AV-DELTA",
        "ravenhuang/wan-dit/action_variation_candidate",
        "action-variation-av-delta-seed1234-u000200",
        True,
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


@dataclass(frozen=True)
class Endpoint:
    code: str
    nfe: int
    action_source: str
    primary_gate: bool


ENDPOINTS = tuple(
    Endpoint(f"aligned_nfe_{nfe}", nfe, "aligned", True) for nfe in NFE_GRID
) + (
    Endpoint("zero_nfe_1", 1, "zero", False),
    Endpoint("global_shuffled_nfe_1", 1, "global_shuffled", False),
)
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


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
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return (
        isinstance(identity, str)
        and len(identity) == 64
        and hashlib.sha256(_canonical_json(unsigned)).hexdigest() == identity
    )


def fixed_protocol() -> dict[str, Any]:
    return {
        "question": (
            "does a train-whitened within-chunk action-velocity residual preserve "
            "clip-specific causal control and improve few-step VPM video?"
        ),
        "arms": [asdict(arm) for arm in ARMS],
        "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
        "continuation_updates": 200,
        "seed": 1234,
        "world_size": EXPECTED_WORLD_SIZE,
        "local_batch_size": 1,
        "global_batch_size": 8,
        "train_clips": 512,
        "validation_clips": EXPECTED_VALIDATION_CLIPS,
        "optimizer_state_policy": "fresh_identical_adamw",
        "ema_policy": "none",
        "action_residual": {
            "source": "requested_actions_only",
            "future_chunks": [4, 12],
            "within_chunk_delta": "a[...,1:,:]-a[...,:-1,:]",
            "delta_steps": 4,
            "padding_dim": 157,
            "normalization": "per_coordinate_population_mean_std_fit_on_train_only",
            "inactive_coordinate_rule": "std<=1e-6_maps_to_exact_zero",
            "clip_value": 8.0,
            "residual_hidden": 256,
            "residual_layers": 3,
            "initialization_seed": 20_260_808,
            "gate": "tanh_scalar_initialized_exact_zero",
            "injection": "z_future_base_plus_gate_times_delta_residual",
            "control_hard_mask": True,
            "same_schema_and_forward_compute": True,
            "parent_function_preserved_at_initialization": True,
        },
        "training_model_calls_per_update": 1,
        "nfe_grid": list(NFE_GRID),
        "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
        "evaluation_noise_seed": 20_260_726,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_808,
        "primary_metric": "decoded_temporal_difference_mse_unit_range",
        "guardrail_metrics": ["video_future_nmse", "decoded_mse_unit_range"],
        "action_attribution_endpoints": ["zero_nfe_1", "global_shuffled_nfe_1"],
        "protected_test_access_allowed": False,
        "future_rgb_or_feature_allowed_at_sampling": False,
        "wandb": {
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "group": None,
        },
    }


def _validate_stats(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    payload = base._read_json(path, "action-delta statistics")
    if (
        not identity_valid(payload)
        or payload.get("schema") != KIND_STATS
        or payload.get("split") != "train"
        or payload.get("protected_test_accessed") is not False
        or payload.get("future_action_chunks") != [4, 12]
        or payload.get("chunk_size") != 5
        or payload.get("delta_steps") != 4
        or payload.get("padding_dim") != 157
        or payload.get("fit_observations") != 512 * 8 * 4
        or payload.get("std_floor") != 1e-6
    ):
        raise ActionVariationScreenError("action-delta statistics contract differs")
    for key in ("mean", "std", "active"):
        if not isinstance(payload.get(key), list) or len(payload[key]) != 157:
            raise ActionVariationScreenError(f"statistics {key} shape differs")
    return payload


def command_fit_stats(args: argparse.Namespace) -> int:
    train = base._validate_train_inputs(args.train_manifest, args.train_cache_metadata)
    actions_record = train["actions"]
    actions = np.load(actions_record["path"], mmap_mode="r", allow_pickle=False)
    if actions.shape != (512, 13, 5, 23) or actions.dtype != np.float32:
        raise ActionVariationScreenError(
            "registered train action array shape/type differs"
        )
    future = np.asarray(actions[:, 4:12], dtype=np.float64)
    delta = future[:, :, 1:, :] - future[:, :, :-1, :]
    flat = delta.reshape(-1, delta.shape[-1])
    if flat.shape != (512 * 8 * 4, 23) or not np.isfinite(flat).all():
        raise ActionVariationScreenError("train action deltas are invalid")
    raw_mean = flat.mean(axis=0)
    raw_std = flat.std(axis=0, ddof=0)
    active_raw = raw_std > 1e-6
    mean = np.zeros(157, dtype=np.float64)
    std = np.ones(157, dtype=np.float64)
    active = np.zeros(157, dtype=bool)
    mean[:23][active_raw] = raw_mean[active_raw]
    std[:23][active_raw] = raw_std[active_raw]
    active[:23] = active_raw
    whitened = np.zeros_like(flat)
    whitened[:, active_raw] = (flat[:, active_raw] - raw_mean[active_raw]) / raw_std[
        active_raw
    ]
    payload = identity_payload(
        {
            "schema": KIND_STATS,
            "created_at_utc": _now(),
            "split": "train",
            "protected_test_accessed": False,
            "future_action_chunks": [4, 12],
            "chunk_size": 5,
            "delta_steps": 4,
            "padding_dim": 157,
            "fit_clips": 512,
            "fit_observations": int(flat.shape[0]),
            "std_floor": 1e-6,
            "active_dimensions": int(active.sum()),
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
            "active": [bool(value) for value in active],
            "whitened_active_mean_abs_max": float(
                np.abs(whitened[:, active_raw].mean(axis=0)).max()
            ),
            "whitened_active_std_error_max": float(
                np.abs(whitened[:, active_raw].std(axis=0) - 1.0).max()
            ),
            "source": {
                "manifest": train["manifest"],
                "cache_metadata": train["cache_metadata"],
                "actions": train["actions"],
                "rgb_opened": False,
                "auxiliary_target_opened": False,
            },
        }
    )
    output = args.output.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ActionVariationScreenError("stats output must be a fresh absolute path")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    base._exclusive_json(output, payload)
    print(json.dumps({"stats": base._file_record(output), **payload}, sort_keys=True))
    return 0


def arm_run_identity(registration: Mapping[str, Any], arm: Arm) -> str:
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "action_variation_arm_identity",
            "registration_identity_sha256": registration["identity_sha256"],
            "tool_git_commit": registration["tool_repository"]["git_commit"],
            "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "stats_sha256": registration["action_delta_stats"]["file"]["sha256"],
            "arm": asdict(arm),
            "updates": 200,
            "seed": 1234,
            "world_size": 8,
            "local_batch_size": 1,
        }
    )["identity_sha256"]


def command_register(args: argparse.Namespace) -> int:
    tool_repo = args.tool_repo.expanduser().resolve(strict=True)
    historical_repo = args.historical_repo.expanduser().resolve(strict=True)
    source = base._clean_source(tool_repo, args.expected_commit, "tool")
    historical = base._clean_source(
        historical_repo, HISTORICAL_TRAINING_COMMIT, "historical model"
    )
    base._git(
        tool_repo,
        "merge-base",
        "--is-ancestor",
        HISTORICAL_TRAINING_COMMIT,
        args.expected_commit,
    )
    if (
        tool_repo != REPO_ROOT
        or not PROTOCOL_PATH.is_file()
        or PROTOCOL_PATH.is_symlink()
    ):
        raise ActionVariationScreenError("registered tool/protocol path differs")
    output_root = base._fresh_lustre_root(args.output_root)
    validated = phase._validate_study_metadata(
        args.study_root.expanduser().resolve(strict=True),
        historical_repo,
        rehash_snapshot=True,
    )
    if validated["snapshot_sha256"] != PARENT_SNAPSHOT_SHA256:
        raise ActionVariationScreenError("parent VPM snapshot changed")
    train = base._validate_train_inputs(args.train_manifest, args.train_cache_metadata)
    train_descriptors = train.pop("descriptors")
    stats = _validate_stats(args.stats)
    stats_record = base._file_record(args.stats.expanduser().resolve(strict=True))
    if stats["source"]["actions"]["sha256"] != train["actions"]["sha256"]:
        raise ActionVariationScreenError(
            "statistics were not fit from registered train actions"
        )
    validation_descriptors = base._manifest_descriptors(
        Path(validated["validation"]["manifest"]["path"]),
        expected_split="val",
        expected_count=EXPECTED_VALIDATION_CLIPS,
    )
    if validation_descriptors != validated["descriptors"]:
        raise ActionVariationScreenError("validation descriptors differ")
    if {row["clip_id"] for row in train_descriptors} & {
        row["clip_id"] for row in validation_descriptors
    } or {row["episode_dir"] for row in train_descriptors} & {
        row["episode_dir"] for row in validation_descriptors
    }:
        raise ActionVariationScreenError("train and validation overlap")
    train["validation_disjointness"] = {
        "clip_id": True,
        "episode_dir": True,
        "episode_dir_and_start": True,
    }
    wandb = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    wandb.pop("viewer_email", None)
    python = args.python.expanduser().resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ActionVariationScreenError("runtime Python is not executable")
    payload = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": _now(),
            "status": "registered_before_candidate_metrics",
            "output_root": str(output_root),
            "tool_repository": source,
            "historical_model_repository": historical,
            "protocol": base._file_record(PROTOCOL_PATH),
            "controlled_study": {
                "study_root": str(args.study_root.resolve(strict=True)),
                "study_identity_sha256": validated["study"]["identity_sha256"],
                "arm_identity_sha256": validated["arm"]["identity_sha256"],
                "stage_identity_sha256": validated["stage"]["identity_sha256"],
                "parent_snapshot": base._file_record(validated["paths"]["snapshot"]),
                "parent_completed_updates": 1000,
                "parent_total_observations": 8000,
            },
            "training": train,
            "validation": validated["validation"],
            "validation_descriptors": validation_descriptors,
            "action_delta_stats": {"file": stats_record, "payload": stats},
            "runtime": {**validated["runtime"], "python": str(python)},
            "fixed_protocol": fixed_protocol(),
            "wandb": {**wandb, "group": None, "mode": "online"},
            "protected_test_accessed": False,
        }
    )
    output_root.mkdir(mode=0o700)
    base._exclusive_json(output_root / "registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def validate_registration(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    registration = base._read_json(path, "action-variation registration")
    stats_record = registration.get("action_delta_stats", {}).get("file", {})
    if (
        not identity_valid(registration)
        or registration.get("kind") != KIND_REGISTRATION
        or registration.get("status") != "registered_before_candidate_metrics"
        or registration.get("fixed_protocol") != fixed_protocol()
        or registration.get("protected_test_accessed") is not False
        or registration.get("controlled_study", {})
        .get("parent_snapshot", {})
        .get("sha256")
        != PARENT_SNAPSHOT_SHA256
        or registration.get("wandb", {}).get("entity") != "zijiandu"
        or registration.get("wandb", {}).get("project")
        != "dual-video-diffusion-private"
        or registration.get("wandb", {}).get("access") != "PRIVATE"
        or registration.get("wandb", {}).get("viewer_username") != "zijiandu"
        or registration.get("wandb", {}).get("group") is not None
    ):
        raise ActionVariationScreenError("registration identity/protocol differs")
    canonical = Path(registration["output_root"]) / "registration.json"
    if path != canonical.resolve(strict=True) or stats_record.get(
        "sha256"
    ) != base._sha256(Path(stats_record.get("path", ""))):
        raise ActionVariationScreenError("registration/statistics artifact changed")
    _validate_stats(Path(stats_record["path"]))
    return registration


def revalidate_registered_inputs(registration: Mapping[str, Any]) -> dict[str, Any]:
    records = {
        "parent_snapshot": registration["controlled_study"]["parent_snapshot"],
        "train_manifest": registration["training"]["manifest"],
        "train_metadata": registration["training"]["cache_metadata"],
        "train_rgb": registration["training"]["rgb"],
        "train_actions": registration["training"]["actions"],
        "validation_manifest": registration["validation"]["manifest"],
        "validation_metadata": registration["validation"]["cache_metadata"],
        "validation_rgb": registration["validation"]["rgb"],
        "validation_actions": registration["validation"]["actions"],
        "action_delta_stats": registration["action_delta_stats"]["file"],
    }
    return {
        key: base._revalidate_record(record, key) for key, record in records.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-stats")
    fit.add_argument("--train-manifest", type=Path, required=True)
    fit.add_argument("--train-cache-metadata", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(func=command_fit_stats)
    register = sub.add_parser("register")
    register.add_argument("--tool-repo", type=Path, required=True)
    register.add_argument("--historical-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--stats", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.set_defaults(func=command_register)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
