#!/usr/bin/env python3
"""Prospective contract and paired analysis for low-frequency motion regularization.

The helper is CPU-only and never submits jobs.  It accepts train512 and val64
artifacts only; there is intentionally no protected-test argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
REGISTRATION_KIND = "motion_band_training_only_registration"
SEAL_KIND = "motion_band_training_only_post_training_seal"
RESULT_SCHEMA = "motion_band_training_only_paired_quality_v1"
TRAIN_UPDATES = 200
SEED = 1234
WORLD_SIZE = 8
TRAIN_CLIPS = 512
VALIDATION_CLIPS = 64
EVALUATION_NOISE_SEED = 20_260_808
ACTION_CONTROLS = ("aligned", "episode_shuffled", "zero")
ACTION_SAMPLE_SHAPE = (13, 5, 157)
ZERO_ACTION_SHA256 = hashlib.sha256(
    bytes(ACTION_SAMPLE_SHAPE[0] * ACTION_SAMPLE_SHAPE[1] * ACTION_SAMPLE_SHAPE[2] * 4)
).hexdigest()
METRICS = ("latent_nmse", "decoded_mse", "temporal_mse")
WANDB_ENTITY = "zijiandu"
WANDB_PROJECT = "dual-video-diffusion-private"
VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
PARENT_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
LEGACY_TF_PREFIXES = (
    "forward_model.tf_token_adapter",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_velocity_head",
)
SPLIT_IDENTITIES = {
    "train": {
        "manifest_sha256": "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74",
        "cache_id": "f0a96fc765e5ad39c8f5ad9e3267b9ff8e5b04400df99b4944f7b957fda2e4b6",
    },
    "val": {
        "manifest_sha256": "8cb39c1f056855e28855c0b944c715d084b709e1421f4efeed3710e7099348c4",
        "cache_id": "6b515f4ca0ef3d0e37d72b9fa44b035cf66de703ceb49943f01101992a90fd8f",
    },
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class LFMREGContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    loss_weight: float


ARMS = (
    Arm(
        "LFMREG-OFF",
        "ravenhuang/wan-dit/lfmreg_off",
        "lfmreg-off-seed1234-u000200",
        0.0,
    ),
    Arm(
        "LFMREG-ON",
        "ravenhuang/wan-dit/lfmreg_on",
        "lfmreg-on-seed1234-u000200",
        0.05,
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _valid_identity(payload: Mapping[str, Any]) -> bool:
    observed = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return (
        isinstance(observed, str)
        and SHA_RE.fullmatch(observed) is not None
        and observed == hashlib.sha256(_canonical(unsigned)).hexdigest()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise LFMREGContractError(f"{label} must be an absolute regular file")
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise LFMREGContractError(f"{label} must be an absolute directory")
    return path.resolve(strict=True)


def file_record(path: Path, label: str = "artifact") -> dict[str, Any]:
    path = _regular_file(path, label)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LFMREGContractError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise LFMREGContractError(f"{label} must contain one object")
    return value


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LFMREGContractError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise LFMREGContractError(result.stderr.strip())
    return result.stdout.strip()


def _source_record(expected_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise LFMREGContractError("expected source commit must be full SHA-1")
    if _git(REPO_ROOT, "rev-parse", "HEAD") != expected_commit:
        raise LFMREGContractError("source HEAD differs from expected commit")
    if _git(REPO_ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise LFMREGContractError("source worktree must be clean")
    return {"path": str(REPO_ROOT), "git_commit": expected_commit, "clean": True}


def _runtime_record(python: Path, videox: Path, wan: Path) -> dict[str, Any]:
    if (
        not python.is_absolute()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise LFMREGContractError("Python launcher must be an absolute executable")
    resolved_python = python.resolve(strict=True)
    videox = _directory(videox, "VideoX source")
    wan = _directory(wan, "Wan directory")
    if _git(videox, "rev-parse", "HEAD") != VIDEOX_COMMIT:
        raise LFMREGContractError("VideoX commit differs")
    if _git(videox, "status", "--porcelain", "--untracked-files=all"):
        raise LFMREGContractError("VideoX source must be clean")
    return {
        "python": {
            "launcher": str(python),
            "resolved": str(resolved_python),
            "sha256": _sha256(resolved_python),
        },
        "videox": {"path": str(videox), "git_commit": VIDEOX_COMMIT, "clean": True},
        "wan_dir": str(wan),
    }


def _cache_file(metadata_path: Path, metadata: Mapping[str, Any], key: str) -> Path:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise LFMREGContractError(f"cache metadata lacks {key}")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return _regular_file(path, key)


def _cache_record(manifest_path: Path, metadata_path: Path, *, split: str, count: int):
    manifest = file_record(manifest_path, f"{split} manifest")
    metadata_file = file_record(metadata_path, f"{split} metadata")
    metadata_path = Path(metadata_file["path"])
    metadata = _read_json(metadata_path, f"{split} metadata")
    expected = SPLIT_IDENTITIES[split]
    with Path(manifest["path"]).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != count:
        raise LFMREGContractError(f"{split} clip count differs")
    for index, row in enumerate(rows):
        if (
            int(row.get("auxiliary_index", -1)) != index
            or not isinstance(row.get("clip_id"), str)
            or not isinstance(row.get("episode_dir"), str)
        ):
            raise LFMREGContractError(f"{split} manifest row {index} differs")
    if (
        manifest["sha256"] != expected["manifest_sha256"]
        or metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache"
        or metadata.get("cache_id") != expected["cache_id"]
        or metadata.get("clip_manifest_sha256") != manifest["sha256"]
        or int(metadata.get("clip_count", -1)) != count
        or metadata.get("rgb_shape") != [count, 13, 3, 180, 960]
        or metadata.get("actions_shape") != [count, 13, 5, 23]
        or metadata.get("rgb_dtype") != "float16"
        or metadata.get("actions_dtype") != "float32"
        or metadata.get("complete") is not True
    ):
        raise LFMREGContractError(f"{split} cache schema/identity differs")
    arrays = {}
    for key, digest_key in (
        ("rgb_file", "rgb_sha256"),
        ("actions_file", "actions_sha256"),
    ):
        record = file_record(
            _cache_file(metadata_path, metadata, key), f"{split} {key}"
        )
        if record["sha256"] != metadata.get(digest_key):
            raise LFMREGContractError(f"{split} {key} digest differs")
        arrays[key] = record
    return {
        "split": split,
        "clip_count": count,
        "manifest": manifest,
        "metadata": metadata_file,
        "arrays": arrays,
        "cached_vjepa_target_opened": False,
    }


def _unchanged(record: Mapping[str, Any], label: str) -> None:
    current = file_record(Path(record["path"]), label)
    if any(current[key] != record.get(key) for key in ("path", "bytes", "sha256")):
        raise LFMREGContractError(f"registered {label} changed")


def _snapshot_record(path: Path):
    import torch

    record = file_record(path, "parent snapshot")
    if record["sha256"] != PARENT_SNAPSHOT_SHA256:
        raise LFMREGContractError("parent snapshot digest differs")
    snapshot = torch.load(
        record["path"], map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != 1000
        or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise LFMREGContractError("parent snapshot metadata differs")
    keys = sorted(snapshot["model"])
    deployable_keys = [
        key
        for key in keys
        if not any(key.startswith(prefix) for prefix in LEGACY_TF_PREFIXES)
    ]
    removed_keys = sorted(set(keys) - set(deployable_keys))
    if len(removed_keys) != 14:
        raise LFMREGContractError("parent legacy TF key count differs")
    deployable_schema = [
        [key, list(snapshot["model"][key].shape), str(snapshot["model"][key].dtype)]
        for key in deployable_keys
    ]
    return {
        **record,
        "completed_updates": 1000,
        "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
        "model_keyset_sha256": hashlib.sha256(_canonical(keys)).hexdigest(),
        "legacy_tf_keys_discarded": len(removed_keys),
        "deployable_model_key_count": len(deployable_keys),
        "deployable_model_schema_sha256": hashlib.sha256(
            _canonical(deployable_schema)
        ).hexdigest(),
    }


def _private_wandb() -> dict[str, Any]:
    from tools.dual_abc_pilot import _wandb_private_project

    result = _wandb_private_project(WANDB_ENTITY, WANDB_PROJECT)
    viewer = result.get("viewer_username", result.get("authenticated_viewer_username"))
    if result.get("access") != "PRIVATE" or viewer != WANDB_ENTITY:
        raise LFMREGContractError("W&B personal project is not private")
    return {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": None,
        "access": "PRIVATE",
        "viewer_username": viewer,
        "checked_at": _now(),
    }


def arm_identity(registration: Mapping[str, Any], arm: Arm) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": asdict(arm),
                "seed": SEED,
                "updates": TRAIN_UPDATES,
            }
        )
    ).hexdigest()


def arm_paths(registration: Mapping[str, Any], arm: Arm) -> dict[str, str]:
    root = Path(registration["study_root"])
    run_dir = root / "training" / arm.run_name
    return {
        "run_dir": str(run_dir),
        "snapshot": str(run_dir / "snapshot.pt"),
        "plan": str(root / "plans" / f"{arm.code.lower()}.json"),
        "evaluation_dir": str(root / "evaluation" / arm.code.lower()),
    }


def load_registration(path: Path) -> dict[str, Any]:
    value = _read_json(path, "registration")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != REGISTRATION_KIND
        or not _valid_identity(value)
        or Path(value.get("study_root", "")) / "registration.json" != path.resolve()
    ):
        raise LFMREGContractError("registration identity/location differs")
    return value


def load_seal(registration: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(registration["study_root"]) / "post_training_seal.json"
    value = _read_json(path, "post-training seal")
    if (
        value.get("kind") != SEAL_KIND
        or value.get("registration_identity_sha256") != registration["identity_sha256"]
        or not _valid_identity(value)
    ):
        raise LFMREGContractError("post-training seal differs")
    return value


def revalidate(registration: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if stage not in {"train", "validation"}:
        raise LFMREGContractError("stage must be train or validation")
    source = registration["source"]
    if _source_record(source["git_commit"]) != source:
        raise LFMREGContractError("registered source changed")
    _unchanged(registration["parent_snapshot"], "parent snapshot")
    split = registration["training" if stage == "train" else "validation"]
    _unchanged(split["manifest"], f"{stage} manifest")
    _unchanged(split["metadata"], f"{stage} metadata")
    for key, record in split["arrays"].items():
        _unchanged(record, f"{stage} {key}")
    return {"stage": stage, "verified_at": _now(), "protected_test_accessed": False}


def validate_resolved_config(config, registration, arm: Arm) -> None:
    from omegaconf import OmegaConf

    model = OmegaConf.to_container(config.model, resolve=True)
    motion_band = model["motion_band_regularization"]
    expected_exclusions = [
        "forward_model.tf_token_adapter",
        "forward_model.tf_clock_embedding",
        "forward_model.tf_velocity_head",
    ]
    if (
        model.get("_target_")
        != "lam.motion_band_training_only_model.LowFrequencyMotionTrainingOnlyExplicitActionDiT"
        or motion_band
        != {
            "loss_weight": arm.loss_weight,
            "num_views": 3,
            "kernel_size": 5,
            "sigma": 1.0,
            "beta": 0.25,
            "epsilon": 1.0e-4,
        }
        or int(model.get("num_history_frames", -1)) != 5
        or int(model.get("num_future_frames", -1)) != 8
        or int(model["forward_model"].get("lora_rank", -1)) != 64
        or int(model["forward_model"].get("lora_alpha", -1)) != 128
        or float(model["forward_model"].get("lora_dropout", -1)) != 0.05
        or int(config.seed) != SEED
        or int(config.trainer.config.max_iter) != TRAIN_UPDATES
        or int(config.trainer.config.gradient_accumulation_steps) != 1
        or str(config.trainer.config.load_path)
        != registration["parent_snapshot"]["path"]
        or list(config.trainer.config.exclude_keys) != expected_exclusions
        or int(config.data_loader.batch_size) != 1
        or float(config.optimizer_factory.lr) != 1.0e-4
        or list(config.optimizer_factory.betas) != [0.9, 0.95]
        or int(config.lr_scheduler_factory.lr_lambda.warmup_steps) != 20
        or int(config.lr_scheduler_factory.lr_lambda.total_steps) != TRAIN_UPDATES
        or config.wandb.entity != WANDB_ENTITY
        or config.wandb.project != WANDB_PROJECT
        or config.wandb.group is not None
        or config.wandb.name != arm.run_name
        or str(config.wandb.id) != arm_identity(registration, arm)
        or str(config.wandb.resume) != "never"
    ):
        raise LFMREGContractError(f"{arm.code} resolved config differs")
    for split, record in (
        ("dataset", registration["training"]),
        ("val_dataset", registration["validation"]),
    ):
        dataset = config[split]
        if (
            str(dataset.datasets.ABC.clip_manifest) != record["manifest"]["path"]
            or str(dataset.datasets.ABC.cache_metadata) != record["metadata"]["path"]
            or str(dataset.datasets.ABC._target_)
            != "robot_wm.datasets.abc.fixed_rgb_action_dataset.ABCFixedRGBActionDataset"
        ):
            raise LFMREGContractError(f"{arm.code} {split} cache differs")


def validate_result_rows(rows, arm: Arm) -> None:
    expected = VALIDATION_CLIPS * len(ACTION_CONTROLS)
    if len(rows) != expected:
        raise LFMREGContractError(f"{arm.code} result row count differs")
    keys = set()
    for row in rows:
        key = (int(row.get("clip_index", -1)), row.get("action_control"))
        keys.add(key)
        if (
            row.get("schema") != RESULT_SCHEMA
            or row.get("arm") != arm.code
            or row.get("action_control") not in ACTION_CONTROLS
            or row.get("nfe") != 1
            or row.get("wan_calls") != 1
            or row.get("history_rgb_frames_received") != 5
            or row.get("future_rgb_frames_received") != 0
            or row.get("motion_band_loss_calls_at_inference") != 0
            or row.get("auxiliary_inputs_at_inference") != 0
            or row.get("auxiliary_modules_at_inference") != 0
            or row.get("online_teacher_calls") != 0
            or row.get("cached_vjepa_target_opened") is not False
            or row.get("protected_test_accessed") is not False
            or row.get("noise_seed") != EVALUATION_NOISE_SEED + key[0]
            or row.get("action_tensor_shape") != list(ACTION_SAMPLE_SHAPE)
            or row.get("action_tensor_dtype") != "torch.float32"
            or any(
                not isinstance(row.get(metric), float)
                or not math.isfinite(row[metric])
                or row[metric] < 0
                for metric in METRICS
            )
        ):
            raise LFMREGContractError(f"invalid {arm.code} evaluation row")
        donor = row.get("action_donor_clip_index")
        if (
            (key[1] == "aligned" and donor != key[0])
            or (
                key[1] == "episode_shuffled"
                and (
                    not isinstance(donor, int)
                    or not 0 <= donor < VALIDATION_CLIPS
                    or donor == key[0]
                )
            )
            or (key[1] == "zero" and donor is not None)
            or (
                key[1] == "zero"
                and row.get("action_tensor_sha256") != ZERO_ACTION_SHA256
            )
        ):
            raise LFMREGContractError(f"invalid {arm.code} action-control provenance")
    expected_keys = {
        (clip, control)
        for clip in range(VALIDATION_CLIPS)
        for control in ACTION_CONTROLS
    }
    if keys != expected_keys:
        raise LFMREGContractError(f"{arm.code} evaluation grid differs")
    for clip in range(VALIDATION_CLIPS):
        clip_rows = [row for row in rows if row["clip_index"] == clip]
        if (
            len({row["noise_seed"] for row in clip_rows}) != 1
            or len({row["initial_noise_sha256"] for row in clip_rows}) != 1
            or len({row["score_target_sha256"] for row in clip_rows}) != 1
        ):
            raise LFMREGContractError(
                f"{arm.code} clip {clip} controls did not share noise/target"
            )


def _read_rows(path: Path):
    with _regular_file(path, "evaluation rows").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_register(args) -> int:
    if (
        not args.study_root.is_absolute()
        or Path("/lustre") not in args.study_root.parents
    ):
        raise LFMREGContractError("study root must be an absolute path under /lustre")
    if args.study_root.exists() or args.study_root.is_symlink():
        raise LFMREGContractError("study root must be fresh")
    source = _source_record(args.expected_commit)
    runtime = _runtime_record(args.python, args.videox_home, args.wan_dir)
    payload = identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": REGISTRATION_KIND,
            "status": "registered_before_training_or_validation_metrics",
            "registered_at": _now(),
            "study_root": str(args.study_root),
            "source": source,
            "runtime": runtime,
            "parent_snapshot": _snapshot_record(args.parent_snapshot),
            "training": _cache_record(
                args.train_manifest,
                args.train_metadata,
                split="train",
                count=TRAIN_CLIPS,
            ),
            "validation": _cache_record(
                args.val_manifest,
                args.val_metadata,
                split="val",
                count=VALIDATION_CLIPS,
            ),
            "arms": [asdict(arm) for arm in ARMS],
            "design": {
                "seed": SEED,
                "world_size": WORLD_SIZE,
                "global_batch_size": WORLD_SIZE,
                "updates": TRAIN_UPDATES,
                "same_parent_data_order_noise_optimizer_scheduler": True,
                "only_intervention": "0.05 times training-only low-frequency motion loss",
                "predicted_clean_formula": "x0_hat = x_sigma - sigma * v_hat",
                "signal_axes": "Wan x0_hat future [B,16,T=2,H=24,W=120]",
                "view_layout": "three width-concatenated views; Wv=40; no filter crosses a view seam",
                "observed_anchor": "prepend the last clean observed Wan token, producing T=3 trajectory",
                "transform": "fixed per-view 5x5 Gaussian low-pass, sigma=1.0, then adjacent-time difference",
                "transitions": [
                    "last observed token -> future token 0",
                    "future token 0 -> future token 1",
                ],
                "loss": "Smooth-L1 beta=0.25 after detached target RMS normalization per sample/view",
                "weight": 0.05,
                "inference_feature_or_branch": False,
                "nfe": 1,
                "action_controls": list(ACTION_CONTROLS),
                "scientific_scope": (
                    "T_future=2 cannot identify frame-level contact timing; this is a coarse "
                    "anchor-augmented latent-motion screen, not a contact claim"
                ),
                "protected_test_accessed": False,
            },
            "decision_gate": {
                "decoded_mse": {
                    "point_min_percent": 3.0,
                    "bootstrap_lb_min_percent": 1.0,
                },
                "temporal_mse": {
                    "point_min_percent": 3.0,
                    "bootstrap_lb_min_percent": 1.0,
                },
                "latent_nmse": {
                    "point_min_percent": 0.0,
                    "bootstrap_lb_min_percent": -1.0,
                },
                "all_required": True,
            },
            "deployment_canary": {
                "split": "train",
                "clips": 1,
                "nfe": 1,
                "off_on_loss_weight_bitwise_equivalence": True,
                "motion_band_calls": 0,
            },
            "wandb": _private_wandb(),
        }
    )
    args.study_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.study_root.mkdir(mode=0o700)
    exclusive_json(args.study_root / "registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_values(args) -> int:
    registration = load_registration(args.registration)
    arm = ARM_BY_CODE[args.arm]
    values = {
        **arm_paths(registration, arm),
        "arm": arm.code,
        "config": arm.config_name,
        "run_name": arm.run_name,
        "run_identity": arm_identity(registration, arm),
        "python": registration["runtime"]["python"]["launcher"],
        "videox_home": registration["runtime"]["videox"]["path"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "parent_snapshot": registration["parent_snapshot"]["path"],
        "train_manifest": registration["training"]["manifest"]["path"],
        "train_metadata": registration["training"]["metadata"]["path"],
        "val_manifest": registration["validation"]["manifest"]["path"],
        "val_metadata": registration["validation"]["metadata"]["path"],
    }
    if args.format == "json":
        print(json.dumps(values, sort_keys=True))
    else:
        order = (
            "arm",
            "config",
            "run_name",
            "run_identity",
            "run_dir",
            "snapshot",
            "plan",
            "evaluation_dir",
            "python",
            "videox_home",
            "wan_dir",
            "parent_snapshot",
            "train_manifest",
            "train_metadata",
            "val_manifest",
            "val_metadata",
        )
        if any("\t" in str(values[key]) or "\n" in str(values[key]) for key in order):
            raise LFMREGContractError("unsafe TSV value")
        print("\t".join(str(values[key]) for key in order))
    return 0


def command_verify_stage(args) -> int:
    registration = load_registration(args.registration)
    print(json.dumps(revalidate(registration, args.stage), sort_keys=True))
    return 0


def command_plan(args) -> int:
    registration = load_registration(args.registration)
    revalidation = revalidate(registration, "train")
    arm = ARM_BY_CODE[args.arm]
    paths = arm_paths(registration, arm)
    for key in ("run_dir", "snapshot", "plan", "evaluation_dir"):
        if Path(paths[key]).exists() or Path(paths[key]).is_symlink():
            raise LFMREGContractError(f"fresh output exists: {paths[key]}")
    payload = identity(
        {
            "schema_version": 1,
            "kind": "motion_band_training_only_arm_plan",
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": arm.code,
            "run_identity_sha256": arm_identity(registration, arm),
            "paths": paths,
            "training_input_revalidation": revalidation,
            "only_difference": "low-frequency motion loss weight 0.0 versus 0.05",
            "same_parameter_schema": True,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(Path(paths["plan"]), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _verified_canary(registration: Mapping[str, Any]) -> dict[str, Any]:
    canary = _read_json(
        Path(registration["study_root"]) / "deployment_canary.json",
        "deployment canary",
    )
    if (
        canary.get("status") != "passed"
        or canary.get("registration_identity_sha256") != registration["identity_sha256"]
        or canary.get("off_on_native_latent_bitwise_equal") is not True
        or canary.get("off_on_decoded_bitwise_equal") is not True
        or canary.get("off_on_initial_noise_bitwise_equal") is not True
        or canary.get("wan_calls_total") != 2
        or canary.get("motion_band_loss_calls") != 0
        or canary.get("auxiliary_inputs") != 0
        or canary.get("auxiliary_modules") != 0
        or canary.get("auxiliary_parameters") != 0
        or canary.get("future_rgb_frames_received") != 0
        or canary.get("cached_vjepa_target_opened") is not False
        or canary.get("protected_test_accessed") is not False
    ):
        raise LFMREGContractError("deployment canary differs")
    return canary


def command_verify_canary(args) -> int:
    registration = load_registration(args.registration)
    print(json.dumps(_verified_canary(registration), sort_keys=True))
    return 0


def command_seal(args) -> int:
    import torch
    from omegaconf import OmegaConf

    registration = load_registration(args.registration)
    revalidate(registration, "train")
    _verified_canary(registration)
    sealed_arms = []
    observed_schemas = []
    for arm in ARMS:
        paths = arm_paths(registration, arm)
        plan = _read_json(Path(paths["plan"]), f"{arm.code} plan")
        completion = _read_json(
            Path(paths["run_dir"]) / "training_complete.json", f"{arm.code} completion"
        )
        expected_identity = arm_identity(registration, arm)
        if (
            not _valid_identity(plan)
            or plan.get("run_identity_sha256") != expected_identity
            or completion.get("status") != "completed"
            or completion.get("completed_updates") != TRAIN_UPDATES
            or completion.get("run_identity_sha256") != expected_identity
        ):
            raise LFMREGContractError(f"{arm.code} completion differs")
        snapshot_record = file_record(Path(paths["snapshot"]), f"{arm.code} snapshot")
        snapshot = torch.load(
            snapshot_record["path"], map_location="cpu", weights_only=True, mmap=True
        )
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("_start_iter") != TRAIN_UPDATES
            or snapshot.get("run_identity_sha256") != expected_identity
        ):
            raise LFMREGContractError(f"{arm.code} snapshot metadata differs")
        state = snapshot.get("model")
        if not isinstance(state, Mapping):
            raise LFMREGContractError(f"{arm.code} snapshot model state differs")
        if any(
            "motion_band" in key.lower()
            or any(key.startswith(prefix) for prefix in LEGACY_TF_PREFIXES)
            for key in state
        ):
            raise LFMREGContractError(
                f"{arm.code} contains an inference motion_band/TF parameter"
            )
        model_schema = [
            [key, list(state[key].shape), str(state[key].dtype)]
            for key in sorted(state)
        ]
        schema_sha256 = hashlib.sha256(_canonical(model_schema)).hexdigest()
        if (
            len(state) != registration["parent_snapshot"]["deployable_model_key_count"]
            or schema_sha256
            != registration["parent_snapshot"]["deployable_model_schema_sha256"]
        ):
            raise LFMREGContractError(
                f"{arm.code} parameter schema differs from stripped parent"
            )
        observed_schemas.append(schema_sha256)
        config_record = file_record(
            Path(paths["run_dir"]) / ".hydra/config.yaml", f"{arm.code} resolved config"
        )
        # ``wandb.id`` remains an oc.env interpolation in the saved Hydra
        # config.  Resolve it against the immutable identity of the arm being
        # sealed, rather than inheriting whichever arm happened to run last.
        os.environ["LFMREG_RUN_IDENTITY"] = expected_identity
        validate_resolved_config(
            OmegaConf.load(config_record["path"]), registration, arm
        )
        sealed_arms.append(
            {
                "arm": arm.code,
                "run_identity_sha256": expected_identity,
                "snapshot": snapshot_record,
                "resolved_config": config_record,
                "completion": completion,
                "model_schema_sha256": schema_sha256,
            }
        )
    if len(set(observed_schemas)) != 1:
        raise LFMREGContractError("OFF/ON parameter schemas differ")
    payload = identity(
        {
            "schema_version": 1,
            "kind": SEAL_KIND,
            "sealed_at": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "deployment_canary": file_record(
                Path(registration["study_root"]) / "deployment_canary.json"
            ),
            "arms": sealed_arms,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(
        Path(registration["study_root"]) / "post_training_seal.json", payload
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_verify_arm(args) -> int:
    registration = load_registration(args.registration)
    seal = load_seal(registration)
    arm = ARM_BY_CODE[args.arm]
    entry = next(item for item in seal["arms"] if item["arm"] == arm.code)
    for key, label in (("snapshot", "snapshot"), ("resolved_config", "config")):
        observed = file_record(Path(entry[key]["path"]), f"{arm.code} {label}")
        if observed != entry[key]:
            raise LFMREGContractError(f"sealed {arm.code} {label} changed")
    completion = _read_json(
        Path(arm_paths(registration, arm)["run_dir"]) / "training_complete.json",
        f"{arm.code} completion",
    )
    if completion != entry["completion"]:
        raise LFMREGContractError(f"sealed {arm.code} completion changed")
    result = {
        "arm": arm.code,
        "run_identity_sha256": arm_identity(registration, arm),
        "verified_at": _now(),
        "protected_test_accessed": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _bootstrap_improvement(off, on, *, seed: int = 20_260_808, draws: int = 10_000):
    import numpy as np

    off = np.asarray(off, dtype=np.float64)
    on = np.asarray(on, dtype=np.float64)
    if off.shape != (VALIDATION_CLIPS,) or on.shape != off.shape:
        raise LFMREGContractError("paired bootstrap requires val64 vectors")
    if not np.isfinite(off).all() or not np.isfinite(on).all() or off.mean() <= 0:
        raise LFMREGContractError("paired bootstrap inputs must be finite/positive")
    point = 100.0 * (off.mean() - on.mean()) / off.mean()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, VALIDATION_CLIPS, size=(draws, VALIDATION_CLIPS))
    off_means = off[indexes].mean(axis=1)
    on_means = on[indexes].mean(axis=1)
    effects = 100.0 * (off_means - on_means) / off_means
    return float(point), [
        float(value) for value in np.quantile(effects, [0.025, 0.975])
    ]


def command_analyze(args) -> int:
    registration = load_registration(args.registration)
    seal = load_seal(registration)
    with Path(registration["validation"]["manifest"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        episode_rows = [json.loads(line) for line in handle if line.strip()]
    rows_by_arm = {}
    for arm in ARMS:
        path = Path(arm_paths(registration, arm)["evaluation_dir"]) / "rows.jsonl"
        rows = _read_rows(path)
        validate_result_rows(rows, arm)
        expected_run_identity = arm_identity(registration, arm)
        for row in rows:
            donor = row["action_donor_clip_index"]
            if (
                row.get("registration_identity_sha256")
                != registration["identity_sha256"]
                or row.get("seal_identity_sha256") != seal["identity_sha256"]
                or row.get("run_identity_sha256") != expected_run_identity
                or any(
                    SHA_RE.fullmatch(str(row.get(field, ""))) is None
                    for field in (
                        "action_tensor_sha256",
                        "initial_noise_sha256",
                        "video_latent_sha256",
                        "decoded_sha256",
                        "score_target_sha256",
                    )
                )
                or (
                    row["action_control"] == "episode_shuffled"
                    and episode_rows[row["clip_index"]]["episode_dir"]
                    == episode_rows[donor]["episode_dir"]
                )
            ):
                raise LFMREGContractError(f"{arm.code} sealed row provenance differs")
        inventory = _read_json(path.parent / "inventory.json", f"{arm.code} inventory")
        if (
            not _valid_identity(inventory)
            or inventory.get("rows") != file_record(path)
            or inventory.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or inventory.get("seal_identity_sha256") != seal["identity_sha256"]
            or inventory.get("motion_band_loss_calls_at_inference") != 0
            or inventory.get("auxiliary_modules_at_inference") != 0
        ):
            raise LFMREGContractError(f"{arm.code} inventory differs")
        rows_by_arm[arm.code] = rows
    keyed = {
        arm: {(row["clip_index"], row["action_control"]): row for row in rows}
        for arm, rows in rows_by_arm.items()
    }
    for key in keyed["LFMREG-OFF"]:
        off = keyed["LFMREG-OFF"][key]
        on = keyed["LFMREG-ON"][key]
        for field in (
            "noise_seed",
            "initial_noise_sha256",
            "action_tensor_sha256",
            "action_donor_clip_index",
            "score_target_sha256",
        ):
            if off[field] != on[field]:
                raise LFMREGContractError(f"paired arm field differs: {field}")
    primary = {}
    gate = registration["decision_gate"]
    for metric in METRICS:
        off_values = [
            keyed["LFMREG-OFF"][(clip, "aligned")][metric]
            for clip in range(VALIDATION_CLIPS)
        ]
        on_values = [
            keyed["LFMREG-ON"][(clip, "aligned")][metric]
            for clip in range(VALIDATION_CLIPS)
        ]
        point, interval = _bootstrap_improvement(off_values, on_values)
        criterion = gate[metric]
        passed = (
            point >= criterion["point_min_percent"]
            and interval[0] >= criterion["bootstrap_lb_min_percent"]
        )
        primary[metric] = {
            "off_mean": sum(off_values) / len(off_values),
            "on_mean": sum(on_values) / len(on_values),
            "relative_improvement_percent": point,
            "paired_bootstrap_95_percent": interval,
            "passed": passed,
        }
    controls = {}
    for arm in ARM_BY_CODE:
        controls[arm] = {}
        for control in ACTION_CONTROLS:
            controls[arm][control] = {
                metric: sum(
                    keyed[arm][(clip, control)][metric]
                    for clip in range(VALIDATION_CLIPS)
                )
                / VALIDATION_CLIPS
                for metric in METRICS
            }
    treatment_by_action_control = {}
    for control in ACTION_CONTROLS:
        treatment_by_action_control[control] = {}
        for metric in METRICS:
            off_values = [
                keyed["LFMREG-OFF"][(clip, control)][metric]
                for clip in range(VALIDATION_CLIPS)
            ]
            on_values = [
                keyed["LFMREG-ON"][(clip, control)][metric]
                for clip in range(VALIDATION_CLIPS)
            ]
            point, interval = _bootstrap_improvement(off_values, on_values)
            treatment_by_action_control[control][metric] = {
                "relative_improvement_percent": point,
                "paired_bootstrap_95_percent": interval,
            }
    action_sensitivity = {}
    for arm in ARM_BY_CODE:
        action_sensitivity[arm] = {}
        for control in ("episode_shuffled", "zero"):
            action_sensitivity[arm][control] = {}
            for metric in METRICS:
                control_values = [
                    keyed[arm][(clip, control)][metric]
                    for clip in range(VALIDATION_CLIPS)
                ]
                aligned_values = [
                    keyed[arm][(clip, "aligned")][metric]
                    for clip in range(VALIDATION_CLIPS)
                ]
                point, interval = _bootstrap_improvement(
                    control_values, aligned_values
                )
                action_sensitivity[arm][control][metric] = {
                    "aligned_advantage_percent": point,
                    "paired_bootstrap_95_percent": interval,
                }
    overall = all(value["passed"] for value in primary.values())
    payload = identity(
        {
            "schema_version": 1,
            "kind": "motion_band_training_only_final_analysis",
            "analyzed_at": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "seal_identity_sha256": seal["identity_sha256"],
            "endpoint": {
                "nfe": 1,
                "action_control": "aligned",
                "clips": VALIDATION_CLIPS,
            },
            "primary": primary,
            "action_controls": controls,
            "treatment_effect_by_action_control": treatment_by_action_control,
            "action_sensitivity": action_sensitivity,
            "guardrails": registration["decision_gate"],
            "decision": "PASS" if overall else "FAIL",
            "interpretation": (
                "training-only low-frequency motion regularization improves deployable NFE1 generation"
                if overall
                else "training-only low-frequency motion regularization did not clear the preregistered deployable NFE1 gate"
            ),
            "motion_band_or_auxiliary_inference_calls": 0,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(Path(registration["study_root"]) / "analysis.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    register = sub.add_parser("register")
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--parent-snapshot", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-metadata", type=Path, required=True)
    register.add_argument("--val-manifest", type=Path, required=True)
    register.add_argument("--val-metadata", type=Path, required=True)
    register.set_defaults(func=command_register)
    values = sub.add_parser("values")
    values.add_argument("--registration", type=Path, required=True)
    values.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    values.add_argument("--format", choices=("json", "tsv"), default="json")
    values.set_defaults(func=command_values)
    verify = sub.add_parser("verify-stage")
    verify.add_argument("--registration", type=Path, required=True)
    verify.add_argument("--stage", choices=("train", "validation"), required=True)
    verify.set_defaults(func=command_verify_stage)
    plan = sub.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    plan.set_defaults(func=command_plan)
    verify_canary = sub.add_parser("verify-canary")
    verify_canary.add_argument("--registration", type=Path, required=True)
    verify_canary.set_defaults(func=command_verify_canary)
    seal = sub.add_parser("seal")
    seal.add_argument("--registration", type=Path, required=True)
    seal.set_defaults(func=command_seal)
    verify_arm = sub.add_parser("verify-arm")
    verify_arm.add_argument("--registration", type=Path, required=True)
    verify_arm.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    verify_arm.set_defaults(func=command_verify_arm)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
