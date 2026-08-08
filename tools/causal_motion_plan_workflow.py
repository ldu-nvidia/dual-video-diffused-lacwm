#!/usr/bin/env python3
"""Register, seal, and render the prospective CAMP workflow.

The tool never submits or executes a GPU job.  Registration occurs before
planner fitting, sealing occurs after the fixed update-400 planner exists but
before either video arm starts, and command rendering revalidates every bound
artifact.  Outputs are exclusive and immutable.
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
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion.causal_motion_plan import (  # noqa: E402
    FROZEN_TRAIN_MANIFEST_SHA256,
    FROZEN_TRAIN_METADATA_SHA256,
    FROZEN_TRAIN_RGB_SHA256,
    load_motion_plan_normalizer,
)


SCHEMA_VERSION = 1
REGISTRATION_KIND = "camp_base_registration"
SEALED_KIND = "camp_video_arm_registration"
TRAIN_ACTIONS_SHA256 = (
    "f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89"
)
PARENT_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
EXPECTED_PARENT_UPDATES = 1000
EXPECTED_PLANNER_UPDATES = 400
EXPECTED_VIDEO_UPDATES = 200
EXPECTED_WORLD_SIZE = 8
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CAMPWorkflowError(RuntimeError):
    """A prospective CAMP registration or phase boundary differs."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    fuse_generated_plan: bool


ARMS = (
    Arm(
        "PLAN-OFF",
        "ravenhuang/wan-dit/causal_motion_plan_off",
        "camp-plan-off-seed1234-u000200",
        False,
    ),
    Arm(
        "PLAN-ON",
        "ravenhuang/wan-dit/causal_motion_plan_on",
        "camp-plan-on-seed1234-u000200",
        True,
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {**unsigned, "identity_sha256": _sha256_json(unsigned)}


def _identity_valid(payload: Mapping[str, Any]) -> bool:
    observed = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return (
        isinstance(observed, str)
        and SHA256_RE.fullmatch(observed) is not None
        and observed == _sha256_json(unsigned)
    )


def _regular_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise CAMPWorkflowError(f"{label} must be a regular absolute file")
    return path.resolve(strict=True)


def _file_record(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _executable_record(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise CAMPWorkflowError("runtime Python must be an absolute executable file")
    completed = subprocess.run(
        [str(path), "-c", "import json,platform,sys; print(json.dumps({'version':sys.version,'platform':platform.platform()}))"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CAMPWorkflowError("runtime Python identity probe failed")
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CAMPWorkflowError("runtime Python identity is invalid") from exc
    return {
        "path": str(path.resolve(strict=True)),
        "launcher_path": str(path),
        "sha256": _sha256(path.resolve(strict=True)),
        **identity,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CAMPWorkflowError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CAMPWorkflowError(f"{label} must contain one object")
    return value


def _record_unchanged(record: Mapping[str, Any], label: str) -> None:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise CAMPWorkflowError(f"registered {label} record is malformed")
    path = Path(record["path"])
    observed = _file_record(path, label)
    if observed["sha256"] != record.get("sha256") or (
        "bytes" in record and observed["bytes"] != record.get("bytes")
    ):
        raise CAMPWorkflowError(f"registered {label} changed")


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CAMPWorkflowError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CAMPWorkflowError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _clean_source(repo: Path, expected_commit: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    if repo != REPO_ROOT or COMMIT_RE.fullmatch(expected_commit) is None:
        raise CAMPWorkflowError("tool repository or commit differs")
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise CAMPWorkflowError("tool HEAD differs from expected commit")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise CAMPWorkflowError("tool checkout must be clean")
    return {"path": str(repo), "git_commit": expected_commit, "clean": True}


def _fresh_lustre_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise CAMPWorkflowError("output root must be an absolute named path")
    canonical = path.parent.resolve(strict=True) / path.name
    if canonical != path or Path("/lustre") not in canonical.parents:
        raise CAMPWorkflowError("output root must be canonical under /lustre")
    if canonical.exists() or canonical.is_symlink():
        raise CAMPWorkflowError("output root must be fresh")
    return canonical


def _resolve_array(metadata_path: Path, metadata: Mapping[str, Any], key: str) -> Path:
    value = metadata.get(f"{key}_file")
    if not isinstance(value, str) or not value:
        raise CAMPWorkflowError(f"metadata lacks {key}_file")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return _regular_file(path, f"{key} array")


def _manifest(path: Path, *, split: str, count: int) -> tuple[dict[str, Any], str]:
    record = _file_record(path, f"{split} manifest")
    rows = []
    try:
        with Path(record["path"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CAMPWorkflowError(f"{split} manifest is invalid") from exc
    if len(rows) != count:
        raise CAMPWorkflowError(f"{split} manifest count differs")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("split") != split
            or row.get("auxiliary_index") != index
            or row.get("sample_size") != 13
            or row.get("chunk_size") != 5
            or not isinstance(row.get("clip_id"), str)
            or SHA256_RE.fullmatch(str(row.get("clip_id"))) is None
        ):
            raise CAMPWorkflowError(f"{split} manifest row {index} differs")
    return record, _sha256_json([row["clip_id"] for row in rows])


def _cached_split(
    manifest_path: Path,
    metadata_path: Path,
    *,
    split: str,
    count: int,
) -> dict[str, Any]:
    manifest, clip_ids_sha256 = _manifest(manifest_path, split=split, count=count)
    metadata_record = _file_record(metadata_path, f"{split} cache metadata")
    metadata = _read_json(Path(metadata_record["path"]), f"{split} cache metadata")
    if (
        metadata.get("complete") is not True
        or metadata.get("split") != split
        or metadata.get("clip_count") != count
        or metadata.get("clip_manifest_sha256") != manifest["sha256"]
        or tuple(metadata.get("rgb_shape", ())) != (count, 13, 3, 180, 960)
        or tuple(metadata.get("actions_shape", ())) != (count, 13, 5, 23)
    ):
        raise CAMPWorkflowError(f"{split} cache metadata differs")
    arrays = {
        key: _file_record(
            _resolve_array(Path(metadata_record["path"]), metadata, key),
            f"{split} {key}",
        )
        for key in ("rgb", "actions")
    }
    if any(metadata.get(f"{key}_sha256") != arrays[key]["sha256"] for key in arrays):
        raise CAMPWorkflowError(f"{split} cache array SHA-256 differs")
    return {
        "split": split,
        "clip_count": count,
        "manifest": manifest,
        "cache_metadata": metadata_record,
        "arrays": arrays,
        "clip_ids_sha256": clip_ids_sha256,
        "target_array_opened": False,
    }


def _registration(path: Path) -> dict[str, Any]:
    value = _read_json(path, "CAMP base registration")
    if (
        not _identity_valid(value)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != REGISTRATION_KIND
        or value.get("status") != "registered_before_stats_or_planner_training"
        or value.get("protected_test_accessed") is not False
    ):
        raise CAMPWorkflowError("CAMP base registration differs")
    canonical = Path(value["output_root"]) / "base_registration.json"
    if path.expanduser().resolve(strict=True) != canonical.resolve(strict=True):
        raise CAMPWorkflowError("CAMP base registration path is noncanonical")
    observed_source = _clean_source(
        Path(value["tool_repository"]["path"]),
        value["tool_repository"]["git_commit"],
    )
    if observed_source != value["tool_repository"]:
        raise CAMPWorkflowError("CAMP source changed after registration")
    _record_unchanged(value["parent_snapshot"], "parent snapshot")
    for split in ("training", "validation"):
        population = value[split]
        _record_unchanged(population["manifest"], f"{split} manifest")
        _record_unchanged(population["cache_metadata"], f"{split} cache metadata")
        for key in ("rgb", "actions"):
            _record_unchanged(population["arrays"][key], f"{split} {key}")
    _record_unchanged(value["runtime"]["python_record"], "runtime Python")
    _record_unchanged(value["runtime"]["wan_config"], "Wan configuration")
    _record_unchanged(value["runtime"]["wan_vae"], "Wan VAE weights")
    videox = Path(value["runtime"]["videox_home"])
    if (
        _git(videox, "rev-parse", "HEAD") != value["runtime"]["videox_git_commit"]
        or _git(videox, "status", "--porcelain", "--untracked-files=all")
    ):
        raise CAMPWorkflowError("VideoX runtime changed after registration")
    return value


def _sealed(path: Path) -> dict[str, Any]:
    value = _read_json(path, "CAMP sealed registration")
    if (
        not _identity_valid(value)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != SEALED_KIND
        or value.get("status") != "sealed_before_video_arm_training"
        or value.get("protected_test_accessed") is not False
    ):
        raise CAMPWorkflowError("CAMP sealed registration differs")
    canonical = Path(value["output_root"]) / "video_registration.json"
    if path.expanduser().resolve(strict=True) != canonical.resolve(strict=True):
        raise CAMPWorkflowError("CAMP sealed registration path is noncanonical")
    base = _registration(Path(value["base_registration"]["path"]))
    if value["base_registration"]["sha256"] != _sha256(
        Path(value["base_registration"]["path"])
    ) or value["base_registration_identity_sha256"] != base["identity_sha256"]:
        raise CAMPWorkflowError("CAMP base registration changed after sealing")
    for key in ("motion_plan_stats", "planner_snapshot", "planner_completion"):
        record = value[key]
        observed = _file_record(Path(record["path"]), key)
        if observed != record:
            raise CAMPWorkflowError(f"sealed {key} changed")
    load_motion_plan_normalizer(
        path=value["motion_plan_stats"]["path"],
        expected_sha256=value["motion_plan_stats"]["sha256"],
    )
    return value


def command_register(args: argparse.Namespace) -> int:
    source = _clean_source(args.tool_repo, args.expected_commit)
    output_root = _fresh_lustre_root(args.output_root)
    parent = _file_record(args.parent_snapshot, "VPM parent snapshot")
    if parent["sha256"] != PARENT_SNAPSHOT_SHA256:
        raise CAMPWorkflowError("VPM parent snapshot differs")
    import torch

    snapshot = torch.load(
        Path(parent["path"]), map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != EXPECTED_PARENT_UPDATES
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise CAMPWorkflowError("VPM parent metadata differs")
    train = _cached_split(
        args.train_manifest, args.train_cache_metadata, split="train", count=512
    )
    if (
        train["manifest"]["sha256"] != FROZEN_TRAIN_MANIFEST_SHA256
        or train["cache_metadata"]["sha256"] != FROZEN_TRAIN_METADATA_SHA256
        or train["arrays"]["rgb"]["sha256"] != FROZEN_TRAIN_RGB_SHA256
        or train["arrays"]["actions"]["sha256"] != TRAIN_ACTIONS_SHA256
    ):
        raise CAMPWorkflowError("registered train population differs")
    validation = _cached_split(
        args.validation_manifest,
        args.validation_cache_metadata,
        split="val",
        count=64,
    )
    python_record = _executable_record(args.python)
    wan_dir = args.wan_dir.expanduser().resolve(strict=True)
    videox_home = args.videox_home.expanduser().resolve(strict=True)
    if not wan_dir.is_dir() or not videox_home.is_dir():
        raise CAMPWorkflowError("Wan/VideoX runtime directory differs")
    from omegaconf import OmegaConf

    wan_config_path = videox_home / "config" / "wan2.1" / "wan_civitai.yaml"
    wan_config = _file_record(wan_config_path, "Wan VAE configuration")
    wan_config_value = OmegaConf.load(wan_config_path)
    vae_subpath = str(
        wan_config_value.get("vae_kwargs", {}).get(
            "vae_subpath", "Wan2.1_VAE.pth"
        )
    )
    wan_vae_path = (wan_dir / vae_subpath).resolve(strict=True)
    if wan_dir not in wan_vae_path.parents:
        raise CAMPWorkflowError("Wan VAE path escapes the registered model root")
    wan_vae = _file_record(wan_vae_path, "Wan VAE weights")
    videox_commit = _git(videox_home, "rev-parse", "HEAD")
    if COMMIT_RE.fullmatch(videox_commit) is None or _git(
        videox_home, "status", "--porcelain", "--untracked-files=all"
    ):
        raise CAMPWorkflowError("VideoX runtime repository must be a clean commit")
    protocol_files = {}
    for relative in (
        "docs/experiments/VPM_CAUSAL_MOTION_PLAN_PROTOCOL.md",
        "tools/causal_motion_plan_stats.py",
        "tools/causal_motion_plan_workflow.py",
        "tools/causal_motion_plan_evaluate.py",
        "tools/causal_motion_plan_audit.py",
        "tools/causal_motion_plan_analyze.py",
        "projects/latent_action_models/train_causal_motion_plan.py",
    ):
        protocol_files[relative] = _file_record(REPO_ROOT / relative, relative)
    payload = _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": REGISTRATION_KIND,
            "created_at_utc": _now(),
            "status": "registered_before_stats_or_planner_training",
            "output_root": str(output_root),
            "tool_repository": source,
            "protocol_files": protocol_files,
            "parent_snapshot": parent,
            "parent_completed_updates": EXPECTED_PARENT_UPDATES,
            "training": train,
            "validation": validation,
            "runtime": {
                "python": python_record["launcher_path"],
                "python_record": python_record,
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
                "videox_git_commit": videox_commit,
                "wan_config": wan_config,
                "wan_vae": wan_vae,
                "world_size": EXPECTED_WORLD_SIZE,
            },
            "planned_paths": {
                "motion_plan_stats": str(output_root / "artifacts" / "motion_plan_stats.json"),
                "planner_run": str(output_root / "planner"),
                "planner_snapshot": str(output_root / "planner" / "snapshot.pt"),
                "video_training_root": str(output_root / "training"),
                "evaluation_root": str(output_root / "evaluation"),
            },
            "fixed_protocol": {
                "planner_fit_clips": 448,
                "planner_calibration_clips": 64,
                "planner_updates": EXPECTED_PLANNER_UPDATES,
                "video_updates": EXPECTED_VIDEO_UPDATES,
                "arms": [asdict(arm) for arm in ARMS],
                "nfe_primary": 1,
                "nfe_descriptive": [2, 4],
                "controls": ["aligned", "off", "shuffled"],
                "diagnostic": "action_shuffled_nfe_1",
                "planner_calls": 2,
                "clean_future_plan_input_allowed": False,
                "planner_selection_on_val64_allowed": False,
            },
            "wandb": {
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
                "mode": "online",
            },
            "protected_test_accessed": False,
        }
    )
    output_root.mkdir(mode=0o700)
    (output_root / "artifacts").mkdir(mode=0o700)
    _exclusive_json(output_root / "base_registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _common_environment(registration: Mapping[str, Any]) -> dict[str, str]:
    return {
        "WAN_DIR": registration["runtime"]["wan_dir"],
        "VIDEOX_HOME": registration["runtime"]["videox_home"],
        "CAMP_TRAIN_CLIP_MANIFEST": registration["training"]["manifest"]["path"],
        "CAMP_TRAIN_CACHE_METADATA": registration["training"]["cache_metadata"]["path"],
        "CAMP_VAL_CLIP_MANIFEST": registration["validation"]["manifest"]["path"],
        "CAMP_VAL_CACHE_METADATA": registration["validation"]["cache_metadata"]["path"],
    }


def command_stats_command(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    output = Path(registration["planned_paths"]["motion_plan_stats"])
    if output.exists() or output.is_symlink():
        raise CAMPWorkflowError("planned stats output already exists")
    command = [
        registration["runtime"]["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(REPO_ROOT / "tools" / "causal_motion_plan_stats.py"),
        "--tool-repo",
        registration["tool_repository"]["path"],
        "--expected-commit",
        registration["tool_repository"]["git_commit"],
        "--train-manifest",
        registration["training"]["manifest"]["path"],
        "--train-cache-metadata",
        registration["training"]["cache_metadata"]["path"],
        "--wan-dir",
        registration["runtime"]["wan_dir"],
        "--videox-home",
        registration["runtime"]["videox_home"],
        "--output",
        str(output),
    ]
    print(json.dumps({"environment": {}, "argv": command}, sort_keys=True))
    return 0


def command_planner_command(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    stats_path = Path(registration["planned_paths"]["motion_plan_stats"])
    stats_sha = _sha256(_regular_file(stats_path, "motion-plan stats"))
    load_motion_plan_normalizer(path=str(stats_path), expected_sha256=stats_sha)
    stats_payload = _read_json(stats_path, "motion-plan stats")
    stats_runtime = stats_payload.get("runtime", {})
    if any(
        stats_runtime.get(key) != registration["runtime"].get(key)
        for key in (
            "wan_dir",
            "videox_home",
            "videox_git_commit",
            "wan_config",
            "wan_vae",
        )
    ):
        raise CAMPWorkflowError("motion-plan stats runtime differs from registration")
    run_dir = Path(registration["planned_paths"]["planner_run"])
    if run_dir.exists() or run_dir.is_symlink():
        raise CAMPWorkflowError("planner run path must be fresh")
    run_identity = _sha256_json(
        {
            "schema": "camp-planner-run-v1",
            "registration": registration["identity_sha256"],
            "stats_sha256": stats_sha,
            "updates": EXPECTED_PLANNER_UPDATES,
            "seed": 1234,
        }
    )
    environment = {
        **_common_environment(registration),
        "CAMP_PLAN_STATS": str(stats_path),
        "CAMP_PLAN_STATS_SHA256": stats_sha,
        "CAMP_PLANNER_RUN_ROOT": str(Path(registration["output_root"])),
        "LACWM_RUN_IDENTITY_SHA256": run_identity,
    }
    command = [
        registration["runtime"]["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(REPO_ROOT / "projects" / "latent_action_models" / "train_causal_motion_plan.py"),
        "+experiments_0908=causal_motion_planner_calibration_common",
        f"hydra.run.dir={run_dir}",
        f"hydra.sweep.dir={run_dir}",
        f"trainer.config.saving.save_path={run_dir / 'snapshot.pt'}",
        "wandb.enabled=true",
        "wandb.mode=online",
        "wandb.entity=zijiandu",
        "wandb.project=dual-video-diffusion-private",
        "wandb.group=null",
        "+wandb.id=camp-planner-seed1234-u000400",
        "+wandb.resume=never",
    ]
    print(json.dumps({"environment": environment, "argv": command}, sort_keys=True))
    return 0


def _validate_planner_snapshot(
    path: Path, stats_path: Path, stats_sha: str, expected_run_identity: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    planner = _file_record(path, "planner snapshot")
    snapshot = torch.load(
        Path(planner["path"]), map_location="cpu", weights_only=True, mmap=True
    )
    state = snapshot.get("model")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != EXPECTED_PLANNER_UPDATES
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("run_identity_sha256") != expected_run_identity
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(state, dict)
        or not any(key.startswith("causal_motion_planner.") for key in state)
    ):
        raise CAMPWorkflowError("planner snapshot metadata/schema differs")
    normalizer = load_motion_plan_normalizer(
        path=str(stats_path), expected_sha256=stats_sha
    )
    expected = normalizer.state_dict()
    observed = {
        key.removeprefix("motion_plan_normalizer."): value
        for key, value in state.items()
        if key.startswith("motion_plan_normalizer.")
    }
    if set(expected) != set(observed) or any(
        not torch.equal(expected[key].cpu(), observed[key].detach().cpu())
        for key in expected
    ):
        raise CAMPWorkflowError("planner snapshot normalization buffers differ")
    return planner, {
        "snapshot_schema_version": 3,
        "completed_updates": EXPECTED_PLANNER_UPDATES,
        "world_size": EXPECTED_WORLD_SIZE,
        "run_identity_sha256": snapshot.get("run_identity_sha256"),
    }


def command_seal(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    output = Path(registration["output_root"]) / "video_registration.json"
    if output.exists() or output.is_symlink():
        raise CAMPWorkflowError("sealed registration already exists")
    stats_path = _regular_file(
        Path(registration["planned_paths"]["motion_plan_stats"]),
        "motion-plan stats",
    )
    stats = _file_record(stats_path, "motion-plan stats")
    normalizer = load_motion_plan_normalizer(
        path=str(stats_path), expected_sha256=stats["sha256"]
    )
    expected_planner_run_identity = _sha256_json(
        {
            "schema": "camp-planner-run-v1",
            "registration": registration["identity_sha256"],
            "stats_sha256": stats["sha256"],
            "updates": EXPECTED_PLANNER_UPDATES,
            "seed": 1234,
        }
    )
    planner, planner_metadata = _validate_planner_snapshot(
        Path(registration["planned_paths"]["planner_snapshot"]),
        stats_path,
        stats["sha256"],
        expected_planner_run_identity,
    )
    completion_path = Path(registration["planned_paths"]["planner_run"]) / "training_complete.json"
    completion = _read_json(completion_path, "planner completion")
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "completed"
        or completion.get("completed_updates") != EXPECTED_PLANNER_UPDATES
        or completion.get("max_iter") != EXPECTED_PLANNER_UPDATES
        or completion.get("run_identity_sha256") != expected_planner_run_identity
        or Path(str(completion.get("snapshot", ""))).resolve(strict=True)
        != Path(planner["path"])
    ):
        raise CAMPWorkflowError("planner completion receipt differs")
    stats_payload = _read_json(stats_path, "motion-plan stats")
    stats_runtime = stats_payload.get("runtime", {})
    base_runtime = registration["runtime"]
    if (
        stats_runtime.get("wan_dir") != base_runtime["wan_dir"]
        or stats_runtime.get("videox_home") != base_runtime["videox_home"]
        or stats_runtime.get("videox_git_commit")
        != base_runtime["videox_git_commit"]
        or stats_runtime.get("wan_config") != base_runtime["wan_config"]
        or stats_runtime.get("wan_vae") != base_runtime["wan_vae"]
    ):
        raise CAMPWorkflowError("motion-plan stats used a different Wan runtime")
    base_record = _file_record(args.registration, "base registration")
    arm_identities = {
        arm.code: _sha256_json(
            {
                "schema": "camp-video-arm-run-v1",
                "base_registration": registration["identity_sha256"],
                "stats_sha256": stats["sha256"],
                "planner_sha256": planner["sha256"],
                "parent_sha256": PARENT_SNAPSHOT_SHA256,
                "arm": asdict(arm),
                "updates": EXPECTED_VIDEO_UPDATES,
                "seed": 1234,
            }
        )
        for arm in ARMS
    }
    payload = _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": SEALED_KIND,
            "created_at_utc": _now(),
            "status": "sealed_before_video_arm_training",
            "output_root": registration["output_root"],
            "base_registration": base_record,
            "base_registration_identity_sha256": registration["identity_sha256"],
            "motion_plan_stats": stats,
            "motion_plan_stats_identity_sha256": stats_payload["identity_sha256"],
            "planner_snapshot": planner,
            "planner_metadata": planner_metadata,
            "planner_completion": _file_record(completion_path, "planner completion"),
            "parent_snapshot": registration["parent_snapshot"],
            "parameter_schema_policy": "identical_model_class_and_shapes_both_arms",
            "arm_run_identity_sha256": arm_identities,
            "fixed_protocol": registration["fixed_protocol"],
            "wandb": registration["wandb"],
            "protected_test_accessed": False,
        }
    )
    _exclusive_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _arm_paths(sealed: Mapping[str, Any], arm: Arm) -> dict[str, Path]:
    root = Path(sealed["output_root"])
    return {
        "run": root / "training" / arm.run_name,
        "snapshot": root / "training" / arm.run_name / "snapshot.pt",
        "evaluation": root / "evaluation" / arm.code.lower(),
    }


def command_arm_command(args: argparse.Namespace) -> int:
    sealed = _sealed(args.sealed_registration)
    base = _registration(Path(sealed["base_registration"]["path"]))
    arm = ARM_BY_CODE.get(args.arm)
    if arm is None:
        raise CAMPWorkflowError("unknown CAMP arm")
    paths = _arm_paths(sealed, arm)
    if paths["run"].exists() or paths["run"].is_symlink():
        raise CAMPWorkflowError("video arm run path must be fresh")
    environment = {
        **_common_environment(base),
        "CAMP_PLAN_STATS": sealed["motion_plan_stats"]["path"],
        "CAMP_PLAN_STATS_SHA256": sealed["motion_plan_stats"]["sha256"],
        "CAMP_PLANNER_SNAPSHOT": sealed["planner_snapshot"]["path"],
        "CAMP_PLANNER_SNAPSHOT_SHA256": sealed["planner_snapshot"]["sha256"],
        "CAMP_VPM_SNAPSHOT": base["parent_snapshot"]["path"],
        "CAMP_VPM_SNAPSHOT_SHA256": base["parent_snapshot"]["sha256"],
        "CAMP_RUN_ROOT": str(Path(sealed["output_root"]) / "training"),
        "LACWM_RUN_IDENTITY_SHA256": sealed["arm_run_identity_sha256"][arm.code],
    }
    command = [
        base["runtime"]["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(REPO_ROOT / "projects" / "latent_action_models" / "train_causal_motion_plan.py"),
        f"+experiments_0908={arm.config_name}",
        f"hydra.run.dir={paths['run']}",
        f"hydra.sweep.dir={paths['run']}",
        f"trainer.config.saving.save_path={paths['snapshot']}",
        "wandb.enabled=true",
        "wandb.mode=online",
        "wandb.entity=zijiandu",
        "wandb.project=dual-video-diffusion-private",
        "wandb.group=null",
        f"+wandb.id={arm.run_name}",
        "+wandb.resume=never",
    ]
    print(json.dumps({"environment": environment, "argv": command}, sort_keys=True))
    return 0


def command_evaluation_command(args: argparse.Namespace) -> int:
    sealed = _sealed(args.sealed_registration)
    base = _registration(Path(sealed["base_registration"]["path"]))
    arm = ARM_BY_CODE.get(args.arm)
    if arm is None:
        raise CAMPWorkflowError("unknown CAMP arm")
    paths = _arm_paths(sealed, arm)
    _regular_file(paths["snapshot"], "trained arm snapshot")
    if paths["evaluation"].exists() or paths["evaluation"].is_symlink():
        raise CAMPWorkflowError("evaluation output must be fresh")
    command = [
        base["runtime"]["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(REPO_ROOT / "tools" / "causal_motion_plan_evaluate.py"),
        "evaluate",
        "--sealed-registration",
        str(args.sealed_registration.resolve(strict=True)),
        "--arm",
        arm.code,
        "--run-dir",
        str(paths["run"]),
        "--output-dir",
        str(paths["evaluation"]),
    ]
    print(json.dumps({"environment": {}, "argv": command}, sort_keys=True))
    return 0


def _training_trace_rows(path: Path, arm: Arm) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    path = _regular_file(path, f"{arm.code} training trace")
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CAMPWorkflowError(f"{arm.code} training trace is invalid") from exc
    if not rows or rows[0].get("kind") != "camp_training_trace_header":
        raise CAMPWorkflowError(f"{arm.code} training trace header differs")
    events: dict[int, dict[str, Any]] = {}
    for row in rows[1:]:
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, dict) or "train_loss/loss" not in metrics:
            continue
        iteration = metrics.get("iteration")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration in events
        ):
            raise CAMPWorkflowError(f"{arm.code} training iteration differs")
        events[iteration] = row
    if set(events) != set(range(EXPECTED_VIDEO_UPDATES)):
        raise CAMPWorkflowError(f"{arm.code} training event inventory differs")
    return rows[0], events


def command_pair_training(args: argparse.Namespace) -> int:
    sealed = _sealed(args.sealed_registration)
    output = Path(sealed["output_root"]) / "training" / "paired_training_audit.json"
    if output.exists() or output.is_symlink():
        raise CAMPWorkflowError("paired training audit output must be fresh")
    traces = {}
    for arm in ARMS:
        trace_path = _arm_paths(sealed, arm)["run"] / "camp_training_trace.jsonl"
        header, events = _training_trace_rows(trace_path, arm)
        traces[arm.code] = {
            "record": _file_record(trace_path, f"{arm.code} training trace"),
            "header": header,
            "events": events,
        }
    shared_header_keys = (
        "parent_snapshot_sha256",
        "parent_run_identity_sha256",
        "planner_snapshot_sha256",
        "motion_plan_stats_sha256",
        "train_manifest_sha256",
        "train_cache_metadata_sha256",
        "parameter_schema_sha256",
        "initial_auxiliary_state_sha256",
        "loaded_planner_state_sha256",
        "continuation_updates",
        "planner_calls_per_example",
        "optimizer_state_policy",
    )
    off_header = traces["PLAN-OFF"]["header"]
    on_header = traces["PLAN-ON"]["header"]
    if any(off_header.get(key) != on_header.get(key) for key in shared_header_keys):
        raise CAMPWorkflowError("CAMP arm training identities/schema differ")
    paired_metric_names = (
        "train_loss/paired_audit/clip_index_mean",
        "train_loss/paired_audit/clip_index_square_mean",
        "train_loss/paired_audit/timestep_mean",
        "train_loss/paired_audit/timestep_square_mean",
        "train_loss/paired_audit/video_noise_probe",
        "train_loss/paired_audit/plan_noise_probe",
        "train_loss/paired_audit/action_probe",
    )
    for iteration in range(EXPECTED_VIDEO_UPDATES):
        off = traces["PLAN-OFF"]["events"][iteration]
        on = traces["PLAN-ON"]["events"][iteration]
        if (
            off.get("total_observations") != on.get("total_observations")
            or any(
                off["metrics"].get(key) != on["metrics"].get(key)
                for key in paired_metric_names
            )
        ):
            raise CAMPWorkflowError(
                f"CAMP data/action/noise pairing differs at update {iteration}"
            )
    receipt = _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "camp_paired_training_audit",
            "status": "passed",
            "sealed_registration_identity_sha256": sealed["identity_sha256"],
            "traces": {
                arm: traces[arm]["record"] for arm in ("PLAN-OFF", "PLAN-ON")
            },
            "updates": EXPECTED_VIDEO_UPDATES,
            "paired_metric_names": list(paired_metric_names),
            "verified": {
                "parent_planner_stats_schema_identical": True,
                "clip_order_identical": True,
                "timesteps_identical": True,
                "video_noise_probe_identical": True,
                "plan_noise_probe_identical": True,
                "action_probe_identical": True,
            },
            "protected_test_accessed": False,
        }
    )
    _exclusive_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def command_audit_command(args: argparse.Namespace) -> int:
    sealed = _sealed(args.sealed_registration)
    base = _registration(Path(sealed["base_registration"]["path"]))
    pairing_path = Path(sealed["output_root"]) / "training" / "paired_training_audit.json"
    pairing = _read_json(pairing_path, "paired training audit")
    if (
        not _identity_valid(pairing)
        or pairing.get("kind") != "camp_paired_training_audit"
        or pairing.get("status") != "passed"
        or pairing.get("sealed_registration_identity_sha256")
        != sealed["identity_sha256"]
    ):
        raise CAMPWorkflowError("paired training audit differs")
    row_paths: list[Path] = []
    for arm in ARMS:
        evaluation = _arm_paths(sealed, arm)["evaluation"]
        inventory_path = evaluation / "inventory.json"
        inventory = _read_json(inventory_path, f"{arm.code} evaluation inventory")
        if (
            inventory.get("kind") != "camp_validation_inventory"
            or inventory.get("sealed_registration_identity_sha256")
            != sealed["identity_sha256"]
            or inventory.get("arm") != asdict(arm)
            or inventory.get("validation_clips") != 64
            or inventory.get("row_count") != 640
            or inventory.get("protected_test_accessed") is not False
        ):
            raise CAMPWorkflowError(f"{arm.code} evaluation inventory differs")
        for rank in range(EXPECTED_WORLD_SIZE):
            row_paths.append(
                _regular_file(
                    evaluation / f"rows_rank_{rank:02d}.jsonl",
                    f"{arm.code} rank rows",
                )
            )
    output = Path(sealed["output_root"]) / "evaluation" / "audit.json"
    if output.exists() or output.is_symlink():
        raise CAMPWorkflowError("CAMP audit output must be fresh")
    command = [
        base["runtime"]["python"],
        str(REPO_ROOT / "tools" / "causal_motion_plan_audit.py"),
    ]
    for path in row_paths:
        command.extend(("--rows", str(path)))
    command.extend(("--output", str(output)))
    print(json.dumps({"environment": {}, "argv": command}, sort_keys=True))
    return 0


def command_analysis_command(args: argparse.Namespace) -> int:
    sealed = _sealed(args.sealed_registration)
    base = _registration(Path(sealed["base_registration"]["path"]))
    audit_path = _regular_file(
        Path(sealed["output_root"]) / "evaluation" / "audit.json",
        "CAMP structural audit",
    )
    pairing_path = _regular_file(
        Path(sealed["output_root"]) / "training" / "paired_training_audit.json",
        "paired training audit",
    )
    pairing = _read_json(pairing_path, "paired training audit")
    if (
        not _identity_valid(pairing)
        or pairing.get("status") != "passed"
        or pairing.get("sealed_registration_identity_sha256")
        != sealed["identity_sha256"]
    ):
        raise CAMPWorkflowError("paired training audit differs")
    row_paths = [
        _regular_file(
            _arm_paths(sealed, arm)["evaluation"] / f"rows_rank_{rank:02d}.jsonl",
            f"{arm.code} rank rows",
        )
        for arm in ARMS
        for rank in range(EXPECTED_WORLD_SIZE)
    ]
    output = Path(sealed["output_root"]) / "evaluation" / "analysis.json"
    if output.exists() or output.is_symlink():
        raise CAMPWorkflowError("CAMP analysis output must be fresh")
    command = [
        base["runtime"]["python"],
        str(REPO_ROOT / "tools" / "causal_motion_plan_analyze.py"),
        "--audit",
        str(audit_path),
        "--pairing",
        str(pairing_path),
    ]
    for path in row_paths:
        command.extend(("--rows", str(path)))
    command.extend(("--output", str(output)))
    print(json.dumps({"environment": {}, "argv": command}, sort_keys=True))
    return 0


def command_wandb_check(_args: argparse.Namespace) -> int:
    from tools import dual_abc_pilot

    result = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    if result.get("access") != "PRIVATE" or result.get("viewer_username") != "zijiandu":
        raise CAMPWorkflowError("private personal W&B project check failed")
    print(json.dumps({**result, "group": None}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--tool-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.add_argument("--parent-snapshot", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--validation-manifest", type=Path, required=True)
    register.add_argument("--validation-cache-metadata", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.set_defaults(function=command_register)

    for name, function in (
        ("stats-command", command_stats_command),
        ("planner-command", command_planner_command),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--registration", type=Path, required=True)
        command.set_defaults(function=function)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--registration", type=Path, required=True)
    seal.set_defaults(function=command_seal)

    for name, function in (
        ("arm-command", command_arm_command),
        ("evaluation-command", command_evaluation_command),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--sealed-registration", type=Path, required=True)
        command.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
        command.set_defaults(function=function)

    audit = subparsers.add_parser("audit-command")
    audit.add_argument("--sealed-registration", type=Path, required=True)
    audit.set_defaults(function=command_audit_command)

    pair = subparsers.add_parser("pair-training")
    pair.add_argument("--sealed-registration", type=Path, required=True)
    pair.set_defaults(function=command_pair_training)

    analysis = subparsers.add_parser("analysis-command")
    analysis.add_argument("--sealed-registration", type=Path, required=True)
    analysis.set_defaults(function=command_analysis_command)

    wandb = subparsers.add_parser("wandb-check")
    wandb.set_defaults(function=command_wandb_check)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
