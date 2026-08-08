#!/usr/bin/env python3
"""Immutable contract and analysis for the prospective VideoREPA TRD screen.

This helper performs CPU-side registration, verification, sealing, and
analysis.  It never submits a job and never accepts a protected-test path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = 1
REGISTRATION_KIND = "videorepa_trd_registration"
SEAL_KIND = "videorepa_trd_post_training_seal"
RESULT_SCHEMA = "videorepa_trd_paired_quality_v2"
TRAIN_UPDATES = 200
SEED = 1234
WORLD_SIZE = 8
TRAIN_CLIPS = 512
VALIDATION_CLIPS = 64
TARGET_SHAPE = (64, 4, 24, 120)
VJEPA_MODEL_NAME = "vjepa2_1_vit_base_384"
VJEPA_RELEASE_COMMIT = "45d025f636dfc58fc2426905fc4a1ab755b1c3e5"
VJEPA_CHECKPOINT_SHA256 = (
    "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
)
VJEPA_PCA_SHA256 = (
    "f0f086178a40a0451a5a989260cef1967fed2ff5ff7647d18c79b632e780c3fe"
)
SPLIT_IDENTITIES = {
    "train": {
        "manifest_sha256": (
            "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
        ),
        "cache_id": (
            "f0a96fc765e5ad39c8f5ad9e3267b9ff8e5b04400df99b4944f7b957fda2e4b6"
        ),
    },
    "val": {
        "manifest_sha256": (
            "8cb39c1f056855e28855c0b944c715d084b709e1421f4efeed3710e7099348c4"
        ),
        "cache_id": (
            "6b515f4ca0ef3d0e37d72b9fa44b035cf66de703ceb49943f01101992a90fd8f"
        ),
    },
}
NFE_GRID = (1, 2, 4)
ACTION_CONTROLS = ("aligned", "episode_shuffled", "zero")
ACTION_SAMPLE_SHAPE = (13, 5, 157)
ACTION_TENSOR_DTYPE = "torch.float32"
ZERO_ACTION_SHA256 = hashlib.sha256(
    bytes(ACTION_SAMPLE_SHAPE[0] * ACTION_SAMPLE_SHAPE[1] * ACTION_SAMPLE_SHAPE[2] * 4)
).hexdigest()
PRIMARY_CONTROL = "aligned"
PRIMARY_NFE = 1
VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
PARENT_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
WANDB_ENTITY = "zijiandu"
WANDB_PROJECT = "dual-video-diffusion-private"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class TRDContractError(RuntimeError):
    """A prospective protocol invariant changed or an artifact is stale."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    mode: str
    loss_weight: float


ARMS = (
    Arm(
        "TRD-OFF",
        "ravenhuang/wan-dit/videorepa_trd_off",
        "videorepa-trd-off-seed1234-u000200",
        "off",
        0.0,
    ),
    Arm(
        "TRD-ON",
        "ravenhuang/wan-dit/videorepa_trd_on",
        "videorepa-trd-on-seed1234-u000200",
        "on",
        0.05,
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
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
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise TRDContractError(f"{label} must be an absolute regular file")
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise TRDContractError(f"{label} must be an absolute non-symlink directory")
    return path.resolve(strict=True)


def _file_record(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TRDContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TRDContractError(f"{label} must contain one object")
    return value


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise TRDContractError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise TRDContractError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _clean_source(repo: Path, expected_commit: str) -> dict[str, Any]:
    repo = _directory(repo, "tool source")
    if repo != REPO_ROOT or COMMIT_RE.fullmatch(expected_commit) is None:
        raise TRDContractError("tool source or expected commit differs")
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise TRDContractError("tool source HEAD differs")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise TRDContractError("tool source must be clean")
    return {"path": str(repo), "git_commit": expected_commit, "clean": True}


def _videox_record(path: Path) -> dict[str, Any]:
    path = _directory(path, "VideoX source")
    if _git(path, "rev-parse", "HEAD") != VIDEOX_COMMIT:
        raise TRDContractError("VideoX commit differs")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise TRDContractError("VideoX source must be clean")
    return {"path": str(path), "git_commit": VIDEOX_COMMIT, "clean": True}


def _python_record(path: Path) -> dict[str, Any]:
    launcher = path.expanduser()
    if not launcher.is_absolute() or not launcher.is_file() or not os.access(
        launcher, os.X_OK
    ):
        raise TRDContractError("runtime Python must be an absolute executable")
    resolved = launcher.resolve(strict=True)
    completed = subprocess.run(
        [
            str(launcher),
            "-c",
            "import json,sys; print(json.dumps({'version':sys.version}))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise TRDContractError("runtime Python identity probe failed")
    return {
        "launcher": str(launcher),
        "resolved": str(resolved),
        "sha256": _sha256(resolved),
        **json.loads(completed.stdout),
    }


def _resolve_metadata_file(
    metadata_path: Path, metadata: Mapping[str, Any], key: str
) -> Path:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise TRDContractError(f"cache metadata lacks {key}")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return _regular_file(path, key)


def _cache_record(
    manifest_path: Path,
    metadata_path: Path,
    *,
    split: str,
    expected_count: int,
) -> dict[str, Any]:
    manifest = _file_record(manifest_path, f"{split} manifest")
    metadata_file = _file_record(metadata_path, f"{split} metadata")
    metadata_path = Path(metadata_file["path"])
    metadata = _read_json(metadata_path, f"{split} metadata")
    expected_identity = SPLIT_IDENTITIES.get(split)
    if expected_identity is None:
        raise TRDContractError(f"unsupported cache split: {split}")
    rows = []
    try:
        with Path(manifest["path"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TRDContractError(f"{split} manifest is invalid") from exc
    if len(rows) != expected_count:
        raise TRDContractError(f"{split} clip count differs")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or int(row.get("auxiliary_index", -1)) != index
            or not isinstance(row.get("clip_id"), str)
            or not isinstance(row.get("episode_dir"), str)
        ):
            raise TRDContractError(f"{split} manifest row {index} differs")
        if row.get("split") not in (None, split):
            raise TRDContractError(f"{split} manifest split label differs")
    if (
        manifest["sha256"] != expected_identity["manifest_sha256"]
        or metadata.get("format_version") != 1
        or metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache"
        or metadata.get("model_name") != VJEPA_MODEL_NAME
        or metadata.get("source_commit") != VJEPA_RELEASE_COMMIT
        or metadata.get("checkpoint_sha256") != VJEPA_CHECKPOINT_SHA256
        or metadata.get("pca_sha256") != VJEPA_PCA_SHA256
        or metadata.get("cache_id") != expected_identity["cache_id"]
        or SHA_RE.fullmatch(str(metadata.get("target_sha256", ""))) is None
        or metadata.get("train_manifest_sha256")
        != SPLIT_IDENTITIES["train"]["manifest_sha256"]
        or metadata.get("complete") is not True
        or int(metadata.get("clip_count", -1)) != expected_count
        or metadata.get("clip_manifest_sha256") != manifest["sha256"]
        or metadata.get("target_shape")
        != [expected_count, *TARGET_SHAPE]
        or metadata.get("rgb_shape")
        != [expected_count, 13, 3, 180, 960]
        or metadata.get("actions_shape")
        != [expected_count, 13, 5, 23]
        or metadata.get("target_dtype") != "float16"
        or metadata.get("rgb_dtype") != "float16"
        or metadata.get("actions_dtype") != "float32"
        or int(metadata.get("sample_size", -1)) != 13
        or int(metadata.get("chunk_size", -1)) != 5
    ):
        raise TRDContractError(f"{split} cache schema differs")
    arrays = {}
    files = [
        ("rgb_file", "rgb_sha256"),
        ("actions_file", "actions_sha256"),
    ]
    if split == "train":
        files.insert(0, ("target_file", "target_sha256"))
    for key, digest_key in files:
        record = _file_record(
            _resolve_metadata_file(metadata_path, metadata, key),
            f"{split} {key}",
        )
        if record["sha256"] != metadata.get(digest_key):
            raise TRDContractError(f"{split} {key} digest differs from metadata")
        arrays[key] = record
    return {
        "split": split,
        "clip_count": expected_count,
        "manifest": manifest,
        "cache_metadata": metadata_file,
        "arrays": arrays,
        "target_shape_per_clip": list(TARGET_SHAPE),
    }


def _record_unchanged(record: Mapping[str, Any], label: str) -> None:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise TRDContractError(f"registered {label} record is malformed")
    observed = _file_record(Path(record["path"]), label)
    if (
        observed["sha256"] != record.get("sha256")
        or observed["bytes"] != record.get("bytes")
    ):
        raise TRDContractError(f"registered {label} changed")


def _fresh_lustre_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise TRDContractError("study root must be an absolute named path")
    # ``videorepa_trd`` may be a new artifact namespace.  Resolve existing
    # ancestors and normalize missing suffixes without requiring the immediate
    # parent to have been manually created first.
    canonical = path.resolve(strict=False)
    if canonical != path or Path("/lustre") not in canonical.parents:
        raise TRDContractError("study root must be canonical under /lustre")
    if canonical.exists() or canonical.is_symlink():
        raise TRDContractError("study root must be fresh")
    return canonical


def _wandb_private() -> dict[str, Any]:
    from tools.dual_abc_pilot import _wandb_private_project

    result = _wandb_private_project(WANDB_ENTITY, WANDB_PROJECT)
    viewer = result.get("viewer_username", result.get("authenticated_viewer_username"))
    if result.get("access") != "PRIVATE" or viewer != WANDB_ENTITY:
        raise TRDContractError("W&B personal project privacy check failed")
    return {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": None,
        "access": "PRIVATE",
        "viewer_username": viewer,
        "checked_at": _now(),
    }


def _snapshot_metadata(path: Path) -> dict[str, Any]:
    import torch

    snapshot = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != 1000
        or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise TRDContractError("parent snapshot is not the frozen update-1000 VPM")
    keys = sorted(str(key) for key in snapshot["model"])
    return {
        "snapshot_schema_version": 3,
        "completed_updates": 1000,
        "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
        "model_key_count": len(keys),
        "model_keyset_sha256": hashlib.sha256(_canonical(keys)).hexdigest(),
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
        "trace": str(run_dir / "trd_training_trace.jsonl"),
        "trace_complete": str(run_dir / "trd_training_trace_complete.json"),
        "evaluation_dir": str(root / "evaluation" / arm.code.lower()),
        "plan": str(root / "plans" / f"{arm.code.lower()}.json"),
    }


def load_registration(path: Path) -> dict[str, Any]:
    value = _read_json(path, "TRD registration")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != REGISTRATION_KIND
        or not _valid_identity(value)
        or Path(value.get("study_root", "")) / "registration.json"
        != path.resolve(strict=True)
    ):
        raise TRDContractError("TRD registration identity or location differs")
    return value


def revalidate_stage(
    registration: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    if stage not in {"train", "validation"}:
        raise TRDContractError("stage must be train or validation")
    source = registration["source"]
    if _clean_source(Path(source["path"]), source["git_commit"]) != source:
        raise TRDContractError("registered source changed")
    runtime = registration["runtime"]
    if _videox_record(Path(runtime["videox"]["path"])) != runtime["videox"]:
        raise TRDContractError("registered VideoX source changed")
    if _directory(Path(runtime["wan_dir"]), "Wan directory") != Path(
        runtime["wan_dir"]
    ):
        raise TRDContractError("registered Wan directory changed")
    python = _python_record(Path(runtime["python"]["launcher"]))
    if (
        python["resolved"] != runtime["python"]["resolved"]
        or python["sha256"] != runtime["python"]["sha256"]
    ):
        raise TRDContractError("registered Python changed")
    _record_unchanged(registration["parent_snapshot"]["file"], "parent snapshot")
    split_record = registration[
        "training" if stage == "train" else "validation"
    ]
    _record_unchanged(split_record["manifest"], f"{stage} manifest")
    _record_unchanged(split_record["cache_metadata"], f"{stage} metadata")
    for key, record in split_record["arrays"].items():
        _record_unchanged(record, f"{stage} {key}")
    return {
        "stage": stage,
        "verified_at": _now(),
        "large_files_hashed_before_nccL": True,
        "protected_test_accessed": False,
    }


def command_register(args: argparse.Namespace) -> int:
    source = _clean_source(REPO_ROOT, args.expected_commit)
    study_root = _fresh_lustre_root(args.study_root)
    warmstart = _file_record(args.warmstart, "VPM parent snapshot")
    if warmstart["sha256"] != PARENT_SNAPSHOT_SHA256:
        raise TRDContractError("VPM parent snapshot digest differs")
    parent = {"file": warmstart, **_snapshot_metadata(Path(warmstart["path"]))}
    training = _cache_record(
        args.train_manifest,
        args.train_metadata,
        split="train",
        expected_count=TRAIN_CLIPS,
    )
    validation = _cache_record(
        args.validation_manifest,
        args.validation_metadata,
        split="val",
        expected_count=VALIDATION_CLIPS,
    )
    payload = _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": REGISTRATION_KIND,
            "status": "registered_before_training_or_validation_metrics",
            "registered_at": _now(),
            "study_root": str(study_root),
            "source": source,
            "runtime": {
                "python": _python_record(args.python),
                "videox": _videox_record(args.videox_home),
                "wan_dir": str(_directory(args.wan_dir, "Wan directory")),
            },
            "parent_snapshot": parent,
            "training": training,
            "validation": validation,
            "arms": [asdict(arm) for arm in ARMS],
            "design": {
                "seed": SEED,
                "world_size": WORLD_SIZE,
                "global_batch_size": WORLD_SIZE,
                "updates": TRAIN_UPDATES,
                "teacher": "cached clean V-JEPA 2.1 PCA64 train targets only",
                "teacher_model": VJEPA_MODEL_NAME,
                "teacher_release_commit": VJEPA_RELEASE_COMMIT,
                "teacher_checkpoint_sha256": VJEPA_CHECKPOINT_SHA256,
                "teacher_pca_sha256": VJEPA_PCA_SHA256,
                "student": "Wan block-14 hidden tokens from noisy video",
                "alignment": (
                    "exclude first VAE temporal bin, then per-view adaptive "
                    "pool to [V=3,F=3,H=2,W=4]"
                ),
                "loss": "spatial plus temporal pairwise cosine-relation L1",
                "margin": 0.05,
                "projection": None,
                "inference_teacher_or_trd_branch": False,
                "pre_arm_deployment_sampler_canary": {
                    "split": "train",
                    "clips": 1,
                    "nfe": 1,
                    "ordinary_vpm_condition_off_equivalence": "bitwise",
                },
                "nfe_grid": list(NFE_GRID),
                "action_controls": list(ACTION_CONTROLS),
                "exact_action_tensor_sha256_per_cell": True,
                "primary_endpoint": {"control": PRIMARY_CONTROL, "nfe": PRIMARY_NFE},
                "protected_test_accessed": False,
            },
            "decision_gate": {
                "decoded_mse_relative_improvement_percent": {
                    "point_min": 3.0,
                    "paired_bootstrap_lower_bound_min": 1.0,
                },
                "temporal_mse_relative_improvement_percent": {
                    "point_min": 3.0,
                    "paired_bootstrap_lower_bound_min": 1.0,
                },
                "latent_nmse_relative_improvement_percent": {
                    "point_min": 0.0,
                    "paired_bootstrap_lower_bound_min": -1.0,
                },
            },
            "wandb": _wandb_private(),
        }
    )
    study_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    study_root.mkdir(mode=0o700)
    _exclusive_json(study_root / "registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_verify_stage(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration)
    print(json.dumps(revalidate_stage(registration, stage=args.stage), sort_keys=True))
    return 0


def _arm(args_arm: str) -> Arm:
    arm = ARM_BY_CODE.get(args_arm)
    if arm is None:
        raise TRDContractError(f"unknown arm: {args_arm}")
    return arm


def command_values(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration)
    arm = _arm(args.arm)
    paths = arm_paths(registration, arm)
    values = {
        "arm": arm.code,
        "config": arm.config_name,
        "run_name": arm.run_name,
        "run_identity": arm_identity(registration, arm),
        **paths,
        "python": registration["runtime"]["python"]["launcher"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox"]["path"],
        "warmstart": registration["parent_snapshot"]["file"]["path"],
        "train_manifest": registration["training"]["manifest"]["path"],
        "train_metadata": registration["training"]["cache_metadata"]["path"],
        "validation_manifest": registration["validation"]["manifest"]["path"],
        "validation_metadata": registration["validation"]["cache_metadata"]["path"],
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
            "trace",
            "trace_complete",
            "evaluation_dir",
            "plan",
            "python",
            "wan_dir",
            "videox_home",
            "warmstart",
            "train_manifest",
            "train_metadata",
            "validation_manifest",
            "validation_metadata",
        )
        if any("\t" in values[key] or "\n" in values[key] for key in order):
            raise TRDContractError("unsafe TSV value")
        print("\t".join(values[key] for key in order))
    return 0


def command_write_plan(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration)
    revalidation = revalidate_stage(registration, stage="train")
    arm = _arm(args.arm)
    paths = arm_paths(registration, arm)
    for key in ("run_dir", "snapshot", "evaluation_dir", "plan"):
        path = Path(paths[key])
        if path.exists() or path.is_symlink():
            raise TRDContractError(f"fresh arm output exists: {path}")
    payload = _identity(
        {
            "schema_version": 1,
            "kind": "videorepa_trd_arm_plan",
            "status": "planned_before_arm_training",
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": asdict(arm),
            "run_identity_sha256": arm_identity(registration, arm),
            "paths": paths,
            "training_input_revalidation": revalidation,
            "matched_contract": {
                "same_parent": True,
                "same_model_parameter_schema": True,
                "same_data_order": True,
                "same_seed_noise_optimizer_scheduler": True,
                "same_single_Wan_call_per_update": True,
                "only_difference": "whether 0.05 * TRD relation loss enters objective",
            },
            "validation_exposed_to_training_process": False,
            "protected_test_accessed": False,
        }
    )
    _exclusive_json(Path(paths["plan"]), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _verify_completed_arm(
    registration: Mapping[str, Any], arm: Arm
) -> dict[str, Any]:
    import torch

    paths = arm_paths(registration, arm)
    plan = _read_json(Path(paths["plan"]), f"{arm.code} plan")
    if (
        not _valid_identity(plan)
        or plan.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or plan.get("arm") != asdict(arm)
    ):
        raise TRDContractError(f"{arm.code} arm plan differs")
    completion = _read_json(
        Path(paths["run_dir"]) / "training_complete.json",
        f"{arm.code} completion",
    )
    identity = arm_identity(registration, arm)
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != TRAIN_UPDATES
        or completion.get("max_iter") != TRAIN_UPDATES
        or completion.get("run_identity_sha256") != identity
    ):
        raise TRDContractError(f"{arm.code} training completion differs")
    trace_complete = _read_json(Path(paths["trace_complete"]), f"{arm.code} trace")
    if (
        trace_complete.get("status") != "completed"
        or trace_complete.get("completed_updates") != TRAIN_UPDATES
        or trace_complete.get("run_identity_sha256") != identity
        or int(trace_complete.get("records_after_header", 0)) < 2
    ):
        raise TRDContractError(f"{arm.code} telemetry completion differs")
    trace_record = _file_record(Path(paths["trace"]), f"{arm.code} trace rows")
    trace_rows = _read_rows(Path(trace_record["path"]))
    if (
        not trace_rows
        or trace_rows[0].get("kind") != "videorepa_trd_training_trace_start"
        or trace_rows[0].get("run_identity_sha256") != identity
        or trace_rows[0].get("max_iter") != TRAIN_UPDATES
        or trace_rows[0].get("world_size") != WORLD_SIZE
        or len(trace_rows) - 1 != trace_complete.get("records_after_header")
        or any(
            row.get("run_identity_sha256") != identity
            or row.get("kind") != "videorepa_trd_metric"
            for row in trace_rows[1:]
        )
    ):
        raise TRDContractError(f"{arm.code} telemetry rows differ")
    snapshot_record = _file_record(Path(paths["snapshot"]), f"{arm.code} snapshot")
    snapshot = torch.load(
        snapshot_record["path"], map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != TRAIN_UPDATES
        or snapshot.get("run_identity_sha256") != identity
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise TRDContractError(f"{arm.code} snapshot differs")
    config_record = _file_record(
        Path(paths["run_dir"]) / ".hydra/config.yaml", f"{arm.code} config"
    )
    return {
        "arm": asdict(arm),
        "run_identity_sha256": identity,
        "snapshot": snapshot_record,
        "resolved_config": config_record,
        "completion": completion,
        "trace": trace_record,
        "trace_complete": trace_complete,
        "wandb": _wandb_run_record(identity),
    }


def _wandb_run_record(run_id: str) -> dict[str, Any]:
    """Require the identity-bound personal run to be finished and ungrouped."""

    import wandb

    run = wandb.Api(timeout=30).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    group = getattr(run, "group", None)
    if (
        str(getattr(run, "id", "")) != run_id
        or str(getattr(run, "entity", "")) != WANDB_ENTITY
        or str(getattr(run, "project", "")) != WANDB_PROJECT
        or str(getattr(run, "state", "")).lower() != "finished"
        or group not in (None, "")
    ):
        raise TRDContractError("W&B arm run is absent, unfinished, or grouped")
    return {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "run_id": run_id,
        "state": "finished",
        "group": None,
        "url": str(getattr(run, "url", "")),
        "verified_at": _now(),
    }


def command_seal(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration)
    train_verification = revalidate_stage(registration, stage="train")
    seal_path = Path(registration["study_root"]) / "post_training_seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        raise TRDContractError("post-training seal already exists")
    arms = [_verify_completed_arm(registration, arm) for arm in ARMS]
    payload = _identity(
        {
            "schema_version": 1,
            "kind": SEAL_KIND,
            "status": "sealed_after_both_arms_before_val64_access",
            "sealed_at": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "training_revalidation": train_verification,
            "arms": arms,
            "validation_metrics_observed": False,
            "protected_test_accessed": False,
        }
    )
    _exclusive_json(seal_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_verify_arm(args: argparse.Namespace) -> int:
    """Rehash one completed checkpoint before any evaluation NCCL startup."""

    registration = load_registration(args.registration)
    seal = load_seal(registration)
    arm = _arm(args.arm)
    observed = _verify_completed_arm(registration, arm)
    sealed = next(item for item in seal["arms"] if item["arm"]["code"] == arm.code)
    if (
        observed["snapshot"] != sealed["snapshot"]
        or observed["resolved_config"] != sealed["resolved_config"]
        or observed["trace"] != sealed["trace"]
        or observed["run_identity_sha256"] != sealed["run_identity_sha256"]
    ):
        raise TRDContractError("completed arm changed after post-training seal")
    print(
        json.dumps(
            {
                "arm": arm.code,
                "snapshot_sha256": observed["snapshot"]["sha256"],
                "verified_before_nccL": True,
                "verified_at": _now(),
            },
            sort_keys=True,
        )
    )
    return 0


def load_seal(registration: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(registration["study_root"]) / "post_training_seal.json"
    seal = _read_json(path, "post-training seal")
    if (
        seal.get("kind") != SEAL_KIND
        or seal.get("status") != "sealed_after_both_arms_before_val64_access"
        or seal.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or not _valid_identity(seal)
    ):
        raise TRDContractError("post-training seal differs")
    return seal


def _read_rows(path: Path) -> list[dict[str, Any]]:
    path = _regular_file(path, "evaluation rows")
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TRDContractError("evaluation rows are invalid") from exc
    return rows


def validate_result_rows(rows: Sequence[Mapping[str, Any]], arm: Arm) -> None:
    expected = VALIDATION_CLIPS * len(NFE_GRID) * len(ACTION_CONTROLS)
    if len(rows) != expected:
        raise TRDContractError(f"{arm.code} result grid is incomplete")
    keys = set()
    for row in rows:
        if (
            row.get("schema") != RESULT_SCHEMA
            or row.get("arm") != arm.code
            or row.get("action_control") not in ACTION_CONTROLS
            or int(row.get("nfe", -1)) not in NFE_GRID
            or not 0 <= int(row.get("clip_index", -1)) < VALIDATION_CLIPS
            or row.get("sampler_received_clean_feature") is not False
            or row.get("sampler_received_future_rgb") is not False
            or row.get("trd_hook_calls_at_inference") != 0
            or row.get("auxiliary_branch_calls_at_inference") != 0
            or row.get("online_teacher_calls") != 0
            or int(row.get("actual_wan_calls", -1)) != int(row.get("nfe", -2))
            or row.get("score_only_future_rgb_used_after_generation") is not True
            or row.get("target_array_opened") is not False
            or row.get("protected_test_accessed") is not False
        ):
            raise TRDContractError(f"{arm.code} result row violates protocol")
        for digest_key in (
            "action_tensor_sha256",
            "video_initial_sha256",
            "video_final_sha256",
            "decoded_sha256",
            "raw_target_sha256",
        ):
            if SHA_RE.fullmatch(str(row.get(digest_key, ""))) is None:
                raise TRDContractError(f"{arm.code} result digest is invalid")
        key = (
            row["action_control"],
            int(row["nfe"]),
            int(row["clip_index"]),
        )
        keys.add(key)
        for metric in ("latent_nmse", "decoded_mse", "temporal_mse"):
            value = float(row[metric])
            if not 0.0 <= value < float("inf"):
                raise TRDContractError(f"non-finite {metric}")
    if len(keys) != expected:
        raise TRDContractError(f"{arm.code} result keys are duplicated")
    for clip_index in range(VALIDATION_CLIPS):
        clip_rows = [
            row for row in rows if int(row["clip_index"]) == clip_index
        ]
        if (
            len({row["video_initial_sha256"] for row in clip_rows}) != 1
            or len({row["raw_target_sha256"] for row in clip_rows}) != 1
            or any(
                row.get("action_tensor_shape") != list(ACTION_SAMPLE_SHAPE)
                or row.get("action_tensor_dtype") != ACTION_TENSOR_DTYPE
                for row in clip_rows
            )
        ):
            raise TRDContractError(
                "NFE/control cells did not preserve noise/target/action schema"
            )
        for control in ACTION_CONTROLS:
            control_rows = [
                row for row in clip_rows if row["action_control"] == control
            ]
            if len({row["action_tensor_sha256"] for row in control_rows}) != 1:
                raise TRDContractError("one action control changed across NFE")
        for nfe in NFE_GRID:
            cell = [
                row
                for row in rows
                if int(row["clip_index"]) == clip_index and int(row["nfe"]) == nfe
            ]
            if len(cell) != len(ACTION_CONTROLS) or len(
                {row["video_initial_sha256"] for row in cell}
            ) != 1:
                raise TRDContractError("action controls did not share initial noise")
            if len({row["raw_target_sha256"] for row in cell}) != 1:
                raise TRDContractError("action controls did not share score target")
            aligned = next(row for row in cell if row["action_control"] == "aligned")
            zero = next(row for row in cell if row["action_control"] == "zero")
            shuffled = next(
                row for row in cell if row["action_control"] == "episode_shuffled"
            )
            if (
                int(aligned["action_donor_clip_index"]) != clip_index
                or zero["action_donor_clip_index"] is not None
                or not 0
                <= int(shuffled["action_donor_clip_index"])
                < VALIDATION_CLIPS
                or int(shuffled["action_donor_clip_index"]) == clip_index
            ):
                raise TRDContractError("action donor contract differs")

    aligned_actions = {
        (int(row["nfe"]), int(row["clip_index"])): row["action_tensor_sha256"]
        for row in rows
        if row["action_control"] == "aligned"
    }
    for row in rows:
        if row["action_control"] == "episode_shuffled":
            donor_key = (int(row["nfe"]), int(row["action_donor_clip_index"]))
            if row["action_tensor_sha256"] != aligned_actions.get(donor_key):
                raise TRDContractError(
                    "shuffled action hash does not match its exact donor tensor"
                )
    zero_hashes = {
        row["action_tensor_sha256"]
        for row in rows
        if row["action_control"] == "zero"
    }
    if zero_hashes != {ZERO_ACTION_SHA256}:
        raise TRDContractError(
            "zero action hash is not the exact canonical FP32 zero tensor"
        )


def _bootstrap(values, *, seed: int, samples: int = 10_000) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (VALIDATION_CLIPS,) or not np.isfinite(array).all():
        raise TRDContractError("paired bootstrap input differs")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "point": float(array.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
    }


def _paired_improvement(
    off_rows: Sequence[Mapping[str, Any]],
    on_rows: Sequence[Mapping[str, Any]],
    *,
    control: str,
    nfe: int,
    metric: str,
    seed: int,
) -> dict[str, float]:
    off = {
        int(row["clip_index"]): float(row[metric])
        for row in off_rows
        if row["action_control"] == control and int(row["nfe"]) == nfe
    }
    on = {
        int(row["clip_index"]): float(row[metric])
        for row in on_rows
        if row["action_control"] == control and int(row["nfe"]) == nfe
    }
    if sorted(off) != list(range(VALIDATION_CLIPS)) or sorted(on) != sorted(off):
        raise TRDContractError("paired metric cell is incomplete")
    improvements = [
        100.0 * (off[index] - on[index]) / max(off[index], 1e-12)
        for index in sorted(off)
    ]
    result = _bootstrap(improvements, seed=seed)
    result.update(
        {
            "off_mean": sum(off.values()) / len(off),
            "on_mean": sum(on.values()) / len(on),
            "positive_favors": "TRD-ON",
        }
    )
    return result


def _paired_action_degradation(
    rows: Sequence[Mapping[str, Any]],
    *,
    control: str,
    nfe: int,
    metric: str,
    seed: int,
) -> dict[str, float]:
    """Quantify how much a wrong/zero action worsens one arm versus aligned."""

    if control not in {"episode_shuffled", "zero"}:
        raise TRDContractError("action degradation requires a diagnostic control")
    aligned = {
        int(row["clip_index"]): float(row[metric])
        for row in rows
        if row["action_control"] == "aligned" and int(row["nfe"]) == nfe
    }
    intervention = {
        int(row["clip_index"]): float(row[metric])
        for row in rows
        if row["action_control"] == control and int(row["nfe"]) == nfe
    }
    if sorted(aligned) != list(range(VALIDATION_CLIPS)) or sorted(
        intervention
    ) != sorted(aligned):
        raise TRDContractError("paired action-control cell is incomplete")
    degradation = [
        100.0
        * (intervention[index] - aligned[index])
        / max(aligned[index], 1e-12)
        for index in sorted(aligned)
    ]
    result = _bootstrap(degradation, seed=seed)
    result.update(
        {
            "aligned_mean": sum(aligned.values()) / len(aligned),
            "intervention_mean": sum(intervention.values()) / len(intervention),
            "positive_means_aligned_actions_are_better": True,
        }
    )
    return result


def _trace_endpoint(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    metrics = [row for row in rows if row.get("kind") == "videorepa_trd_metric"]
    if not metrics:
        raise TRDContractError("training trace contains no metrics")
    return metrics[-1]


def command_analyze(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration)
    seal = load_seal(registration)
    revalidate_stage(registration, stage="validation")
    arm_rows = {}
    inventories = {}
    for arm in ARMS:
        paths = arm_paths(registration, arm)
        inventory = _read_json(
            Path(paths["evaluation_dir"]) / "inventory.json",
            f"{arm.code} evaluation inventory",
        )
        if (
            not _valid_identity(inventory)
            or inventory.get("kind") != "videorepa_trd_evaluation_inventory"
            or inventory.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or inventory.get("arm") != arm.code
            or inventory.get("run_identity_sha256")
            != arm_identity(registration, arm)
            or inventory.get("post_training_seal_identity_sha256")
            != seal["identity_sha256"]
            or int(inventory.get("row_count", -1))
            != VALIDATION_CLIPS * len(NFE_GRID) * len(ACTION_CONTROLS)
            or int(inventory.get("validation_clips", -1)) != VALIDATION_CLIPS
            or inventory.get("nfe_grid") != list(NFE_GRID)
            or inventory.get("action_controls") != list(ACTION_CONTROLS)
            or inventory.get("exact_action_tensors_hashed") is not True
            or inventory.get("target_array_opened") is not False
            or inventory.get("sampler_clean_feature_calls") != 0
            or inventory.get("sampler_future_rgb_calls") != 0
            or inventory.get("online_teacher_calls") != 0
            or inventory.get("trd_inference_parameters") != 0
            or inventory.get("protected_test_accessed") is not False
        ):
            raise TRDContractError(f"{arm.code} evaluation inventory differs")
        expected_rows_path = Path(paths["evaluation_dir"]) / "rows.jsonl"
        rows_record = inventory.get("rows_file")
        if (
            not isinstance(rows_record, Mapping)
            or rows_record.get("path") != str(expected_rows_path)
        ):
            raise TRDContractError(f"{arm.code} row-file record differs")
        _record_unchanged(rows_record, f"{arm.code} evaluation rows")
        rows = _read_rows(expected_rows_path)
        validate_result_rows(rows, arm)
        for row in rows:
            if (
                row.get("registration_identity_sha256")
                != registration["identity_sha256"]
                or row.get("post_training_seal_identity_sha256")
                != seal["identity_sha256"]
                or row.get("run_identity_sha256")
                != arm_identity(registration, arm)
            ):
                raise TRDContractError(f"{arm.code} result identity differs")
        arm_rows[arm.code] = rows
        inventories[arm.code] = inventory
    off_rows = arm_rows["TRD-OFF"]
    on_rows = arm_rows["TRD-ON"]
    off_initial = {
        (row["action_control"], int(row["nfe"]), int(row["clip_index"])):
        row["video_initial_sha256"]
        for row in off_rows
    }
    on_initial = {
        (row["action_control"], int(row["nfe"]), int(row["clip_index"])):
        row["video_initial_sha256"]
        for row in on_rows
    }
    if off_initial != on_initial:
        raise TRDContractError("paired arms did not share deterministic initial noise")

    def keyed(rows, field):
        return {
            (row["action_control"], int(row["nfe"]), int(row["clip_index"])):
            row[field]
            for row in rows
        }

    if keyed(off_rows, "raw_target_sha256") != keyed(on_rows, "raw_target_sha256"):
        raise TRDContractError("paired arms did not share exact score targets")
    if keyed(off_rows, "action_tensor_sha256") != keyed(
        on_rows, "action_tensor_sha256"
    ):
        raise TRDContractError("paired arms did not share exact action tensors")
    if keyed(off_rows, "action_donor_clip_index") != keyed(
        on_rows, "action_donor_clip_index"
    ):
        raise TRDContractError("paired arms did not share exact action controls")
    episode_rows = []
    with Path(registration["validation"]["manifest"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        episode_rows = [json.loads(line) for line in handle if line.strip()]
    for row in off_rows:
        if row["action_control"] != "episode_shuffled":
            continue
        recipient = int(row["clip_index"])
        donor = int(row["action_donor_clip_index"])
        if (
            episode_rows[recipient]["episode_dir"]
            == episode_rows[donor]["episode_dir"]
        ):
            raise TRDContractError("shuffled action donor came from recipient episode")
    grid = {}
    seed = 8_800
    for control in ACTION_CONTROLS:
        grid[control] = {}
        for nfe in NFE_GRID:
            grid[control][str(nfe)] = {}
            for metric in ("latent_nmse", "decoded_mse", "temporal_mse"):
                grid[control][str(nfe)][metric] = _paired_improvement(
                    off_rows,
                    on_rows,
                    control=control,
                    nfe=nfe,
                    metric=metric,
                    seed=seed,
                )
                seed += 1
    action_sensitivity = {}
    for arm in ARMS:
        action_sensitivity[arm.code] = {}
        for control in ("episode_shuffled", "zero"):
            action_sensitivity[arm.code][control] = {}
            for nfe in NFE_GRID:
                action_sensitivity[arm.code][control][str(nfe)] = {}
                for metric in ("latent_nmse", "decoded_mse", "temporal_mse"):
                    action_sensitivity[arm.code][control][str(nfe)][metric] = (
                        _paired_action_degradation(
                            arm_rows[arm.code],
                            control=control,
                            nfe=nfe,
                            metric=metric,
                            seed=seed,
                        )
                    )
                    seed += 1
    primary = grid[PRIMARY_CONTROL][str(PRIMARY_NFE)]
    gate = registration["decision_gate"]
    checks = {
        "decoded_point": primary["decoded_mse"]["point"]
        >= gate["decoded_mse_relative_improvement_percent"]["point_min"],
        "decoded_lower": primary["decoded_mse"]["ci95_lower"]
        >= gate["decoded_mse_relative_improvement_percent"][
            "paired_bootstrap_lower_bound_min"
        ],
        "temporal_point": primary["temporal_mse"]["point"]
        >= gate["temporal_mse_relative_improvement_percent"]["point_min"],
        "temporal_lower": primary["temporal_mse"]["ci95_lower"]
        >= gate["temporal_mse_relative_improvement_percent"][
            "paired_bootstrap_lower_bound_min"
        ],
        "latent_point": primary["latent_nmse"]["point"]
        >= gate["latent_nmse_relative_improvement_percent"]["point_min"],
        "latent_lower": primary["latent_nmse"]["ci95_lower"]
        >= gate["latent_nmse_relative_improvement_percent"][
            "paired_bootstrap_lower_bound_min"
        ],
    }
    passed = all(checks.values())
    training_trace_endpoints = {}
    for arm in ARMS:
        sealed_arm = next(
            item for item in seal["arms"] if item["arm"]["code"] == arm.code
        )
        _record_unchanged(sealed_arm["trace"], f"{arm.code} sealed trace")
        training_trace_endpoints[arm.code] = _trace_endpoint(
            Path(sealed_arm["trace"]["path"])
        )
    payload = _identity(
        {
            "schema_version": 1,
            "kind": "videorepa_trd_analysis",
            "registration_identity_sha256": registration["identity_sha256"],
            "post_training_seal_identity_sha256": seal["identity_sha256"],
            "primary_endpoint": {"control": PRIMARY_CONTROL, "nfe": PRIMARY_NFE},
            "paired_relative_improvement_percent": grid,
            "paired_action_degradation_percent": action_sensitivity,
            "primary_gate_checks": checks,
            "primary_gate_passed": passed,
            "training_trace_endpoints": training_trace_endpoints,
            "inventories": inventories,
            "conclusion": (
                "training_only_vjepa_relational_supervision_improves_deployable_nfe1"
                if passed
                else "no_preregistered_deployable_nfe1_trd_advantage_in_quick_screen"
            ),
            "nfe2_nfe4_are_descriptive_only": True,
            "action_shuffle_and_zero_are_mechanism_controls_not_selection_endpoints": True,
            "clean_feature_used_at_inference": False,
            "protected_test_accessed": False,
            "analyzed_at": _now(),
        }
    )
    output = Path(registration["study_root"]) / "analysis.json"
    _exclusive_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_wandb(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(_wandb_private(), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--warmstart", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-metadata", type=Path, required=True)
    register.add_argument("--validation-manifest", type=Path, required=True)
    register.add_argument("--validation-metadata", type=Path, required=True)
    register.set_defaults(func=command_register)

    verify = subparsers.add_parser("verify-stage")
    verify.add_argument("--registration", type=Path, required=True)
    verify.add_argument("--stage", choices=("train", "validation"), required=True)
    verify.set_defaults(func=command_verify_stage)

    values = subparsers.add_parser("values")
    values.add_argument("--registration", type=Path, required=True)
    values.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    values.add_argument("--format", choices=("json", "tsv"), default="json")
    values.set_defaults(func=command_values)

    plan = subparsers.add_parser("write-plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    plan.set_defaults(func=command_write_plan)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--registration", type=Path, required=True)
    seal.set_defaults(func=command_seal)

    verify_arm = subparsers.add_parser("verify-arm")
    verify_arm.add_argument("--registration", type=Path, required=True)
    verify_arm.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    verify_arm.set_defaults(func=command_verify_arm)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)

    wandb = subparsers.add_parser("wandb-check")
    wandb.set_defaults(func=command_wandb)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
