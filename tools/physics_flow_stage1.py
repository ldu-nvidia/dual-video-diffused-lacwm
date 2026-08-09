#!/usr/bin/env python3
"""Prospective cache, registration, analysis, and audit for raw physics flow.

This module deliberately keeps the analytic field outside the video model.  A
cache row may use only the observed robot state at video frame four, the
already-registered candidate action chunks, a fixed D405 calibration, and the
pinned official ABC robot geometry.  It has no API for future RGB or future
measured robot state.

The workflow has fail-closed state transitions:

``register-cache``
    Bind both immutable RGB/action manifests, camera eligibility, deterministic
    shuffled donors, the independently passed raw-renderer gate, and source.
``build-cache``
    Render compact ``[N,8,4,24,40]`` transition tensors for RAW, SHUFFLED,
    TIME(+1), and HOLD.  OFF is represented at runtime by exact zeros.
``audit-cache``
    Rehash every artifact and reconstruct all deterministic interventions.
``register-study``
    Seal the two-arm training plus untouched-parent equal-Wan-call evaluation
    gate before video model training or generated-video outcome access.
``compare-traces`` / ``analyze`` / ``audit-study``
    Validate matched training, compute the preregistered paired analysis, and
    independently replay the evidence inventory.

Protected test data are unsupported by every command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.abc_d405_nominal_geometry_probe import (  # noqa: E402
    CAMERA_TYPE,
    EXPECTED_ABC_COMMIT,
    OFFICIAL_ASSET_RELATIVE,
    OFFICIAL_SCENE_RELATIVE,
    _decode_top_calibration,
    _joint_qpos_addresses,
    build_robot_only_xml,
)
from tools.corrected_renderer_attribution import (  # noqa: E402
    camera_points_from_depth,
    moving_geom_ids,
    project_world,
    read_mcap_metadata,
    render_pose,
    transport_local_points,
)


SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TRAIN_COUNT = 512
VAL_COUNT = 64
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
ACTION_DIM = 14
PADDED_ACTION_DIM = 23
FUTURE_TRANSITIONS = 8
FLOW_COMPONENTS = 4
OUTPUT_HEIGHT = 180
OUTPUT_WIDTH = 320
PADDED_HEIGHT = 192
POOL = 8
FLOW_HEIGHT = PADDED_HEIGHT // POOL
FLOW_WIDTH = OUTPUT_WIDTH // POOL
MOTION_STRATA = 3
DONOR_SEED = 20260831
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_SAMPLES = 10_000
NOISE_SEEDS = (0, 1, 2, 3)
NFE_GRID = (1, 2, 4)
PRIMARY_NFE = 1
PRIMARY_MIN_POINT_PERCENT = 3.0
PRIMARY_MIN_CI_PERCENT = 1.0
CONTROL_MIN_POINT_PERCENT = 1.0
GUARDRAIL_CI_PERCENT = -1.0
VISIBILITY_ABS_TOLERANCE_M = 0.01
VISIBILITY_REL_TOLERANCE = 0.02

RENDERER_REGISTRATION_IDENTITY = (
    "9cc556aba53d1defb69f0049dab67d12a3991decb917015bba5153c16cb8c2b1"
)
RENDERER_PREPARATION_IDENTITY = (
    "11444ee94ae6707869f44f00409386701c93b886d650ebc928bc4edec4f41a3a"
)
RENDERER_ANALYSIS_IDENTITY = (
    "1471d0bb1f40aabc44d08c71b7eeeba3f0e88b9cb2a07e56dc6f7eb8b11034a0"
)
RENDERER_COMPLETION_IDENTITY = (
    "715fc288ca3e5526cda3a5bb3329db650aa8ed15a61d2959be2349dd3df6dcfc"
)
RENDERER_DECISION = "GO_raw_geometry_scaffold_pass"
RENDERER_SOURCE_COMMIT = "856cd553c051b90f5b8ddf3649103ae1090f4a4d"

PARENT_SNAPSHOT_SHA256 = (
    "de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f"
)
PARENT_CANONICAL_MODEL_STATE_SHA256 = (
    "d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0"
)
PARENT_RESOLVED_CONFIG_SHA256 = (
    "ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
)
PARENT_TRAINING_SOURCE_COMMIT = "656086686dae723c942a4209a9d71cdb17ed6ccc"
PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE = (
    "projects/latent_action_models/lam/dual_explicit_action_dit_model.py"
)
PARENT_NATIVE_SAMPLER_SOURCE_SHA256 = (
    "a10fe3730f7bb3bacd20bd14ebbcfab2b3cf8c63a2db27783f1e8a9787b87ee6"
)
PARENT_NATIVE_SAMPLER_GIT_BLOB = "abc6d4df165f684c2c93920d079585c600562dbb"
PARENT_HISTORICAL_REFERENCE_FILENAME = "parent_historical_reference.pt"
PARENT_TRANSITIVE_SOURCE_FILES = (
    "robot_wm/modeling/dual_diffusion/adapters.py",
    "robot_wm/modeling/networks/wan_forward_model.py",
)
PARENT_HISTORICAL_TRANSITIVE_SHA256 = {
    "robot_wm/modeling/dual_diffusion/adapters.py": (
        "3688f6e5d7a66b80c72f1ff078f2e179323e4c30d7bfd94da10c4707163dfe97"
    ),
    "robot_wm/modeling/networks/wan_forward_model.py": (
        "5cd471dc65940aa32cf953ddb2b05c799ac6c6cc274bda3a6221ef7722e7c4dd"
    ),
}
PARENT_HISTORICAL_TRANSITIVE_BLOBS = {
    "robot_wm/modeling/dual_diffusion/adapters.py": (
        "5b861945d58c3bed751e056382841efc6652e6ce"
    ),
    "robot_wm/modeling/networks/wan_forward_model.py": (
        "e8b2fd4449f9cd56109d3da4257c9d5ca0c267f6"
    ),
}
PARENT_CURRENT_TRANSITIVE_SHA256 = {
    "robot_wm/modeling/dual_diffusion/adapters.py": (
        "f905840fe0ffe54da47279eba01029ef7c228ae8b5595624c88a66b1f317733e"
    ),
    "robot_wm/modeling/networks/wan_forward_model.py": (
        "4d2d49b23c7efc391379b7eb0e2437e6555b40770402d82c17dd03f777e814c1"
    ),
}
PARENT_CURRENT_TRANSITIVE_BLOBS = {
    "robot_wm/modeling/dual_diffusion/adapters.py": (
        "e908586003f761bd819b5864e151713c60258fdb"
    ),
    "robot_wm/modeling/networks/wan_forward_model.py": (
        "53bbbb77d39567574169bc8048ec12e7dc984a15"
    ),
}
PARENT_EVALUATION_MODEL_CODE = "PARENT-VPM"
PARENT_PARITY_KIND = "raw_physics_flow_native_parent_sampler_parity"
PARENT_PARITY_FILENAME = "parent_sampler_parity.json"
PARENT_PARITY_TRAIN_INDEX = 0
PARENT_PARITY_SAMPLE_ID = 7_000_000
LEGACY_REJECTED_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
LEGACY_REJECTED_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
LEGACY_REJECTED_MODEL_STATE_SHA256 = (
    "2b29819672da71cde9d8732e619a50fc2ab941996620b0de8a9d93edcd1a1d9c"
)
PARENT_MODEL_SCHEMA_SHA256 = (
    "9626c924be5ead5fdf5dc7d5887e7435a621845b1b3f739cd72344d94404b97a"
)
CAUSAL_LADDER_REGISTRATION_IDENTITY = (
    "3517eb729ee1b47576ca6a91159121f24fda2b73b7127dcb468628807356b36a"
)
DIRECT_FRONTIER_REGISTRATION_IDENTITY = (
    "e9d4ce9c6299c42d4da0f92302cd8d9ba03ca0b3fcaf6b5c8e622e9707621448"
)
DIRECT_FRONTIER_ANALYSIS_IDENTITY = (
    "6b15b3865925745f2e889585078f66238707315d695ddebad63ae43307ecf67d"
)
DIRECT_FRONTIER_FIT_IDENTITY = (
    "4bfb8a6085db657bbf846f03d3880d4c115e2a4ca025a9395bd75a62e9e1fe7d"
)
DIRECT_FRONTIER_ENDPOINT_IDENTITY = (
    "6d3d4f4f1afeef5bb85046655d33bf019f1fb5367028edbee9d9e9468034242c"
)
DIRECT_FRONTIER_COMPLETION_IDENTITY = (
    "0bf783dccb6105cbd7166014e71bfa24d77d3a91aff624b6e8602532ba16dffc"
)
DIRECT_FRONTIER_AUDIT_IDENTITY = (
    "db093d20647829b67e515f72f4b56f240ddf935a8c11973d2bc917d20b3140fb"
)
DIRECT_FRONTIER_SOURCE_COMMIT = "4f75f9c08dbd64f7ad7d13373a98f25e397f7f4b"
LINEAGE_FAILED_LOG_SHA256 = (
    "18bff874df2ae79bae614520ce80d3dd2d223a86d31bf1c8cc5ad24e05f10714"
)
LINEAGE_COMPARISON_LOG_SHA256 = (
    "048bcddd35ecd2458e5b17f28a48a8a4967111dbb888e8cd2bc9d5ff669c0e2a"
)
LPIPS_RECEIPT_IDENTITY = (
    "04f5013b7161fbf91ed6116d25f7e6ec66afc661024236ad27564b1899cb94be"
)
LPIPS_STATE_DICT_SHA256 = (
    "abc218a76418de010923a57c9c55afb1c1040503b46e5015694ee79ea7c90a7d"
)
LPIPS_ALEXNET_CHECKPOINT_SHA256 = (
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
)
LPIPS_PREFLIGHT_LOG_SHA256 = (
    "db37a417618afa1156cb7226140993755279191edfef8a0c628d550de20fc6af"
)
TRAIN_MANIFEST_SHA256 = (
    "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
)
VAL_MANIFEST_SHA256 = (
    "8cb39c1f056855e28855c0b944c715d084b709e1421f4efeed3710e7099348c4"
)
TRAIN_RGB_SHA256 = (
    "b5bdde4461c75bc88653c38b737021fcbd69b0b22f4c87bc8e8097c3494b64ee"
)
VAL_RGB_SHA256 = (
    "ed82fc0f580baa90c4dc39c1608f97ad7092a4ddcb59c3612eed31711afda404"
)
TRAIN_ACTIONS_SHA256 = (
    "f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89"
)
VAL_ACTIONS_SHA256 = (
    "552a5cf0af156868d2866dfacabe102fc6b5cd24580bb377953e35a14625306a"
)

CACHE_SCHEMA = "raw-physics-flow-cache-v2"
CACHE_REGISTRATION_KIND = "raw_physics_flow_cache_registration"
CACHE_METADATA_KIND = "raw_physics_flow_cache_metadata"
CACHE_AUDIT_KIND = "raw_physics_flow_cache_audit"
STUDY_REGISTRATION_KIND = "raw_physics_flow_stage1_registration"
STUDY_ANALYSIS_KIND = "raw_physics_flow_stage1_analysis"
STUDY_AUDIT_KIND = "raw_physics_flow_stage1_audit"

CACHE_SOURCES = (
    "raw",
    "episode_shuffled",
    "timeshift_plus_one",
    "hold_current",
)
RUNTIME_SOURCES = (
    "raw",
    "off",
    "episode_shuffled",
    "timeshift_plus_one",
    "hold_current",
)


class PhysicsFlowStage1Error(RuntimeError):
    """A causal, immutable, prospective, or evidence contract changed."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    fuse_flow: bool


ARMS = (
    Arm(
        "FLOW-OFF",
        "ravenhuang/wan-dit/physics_flow_off",
        "physics-flow-off-seed1234-u000200",
        False,
    ),
    Arm(
        "RAW-FLOW",
        "ravenhuang/wan-dit/physics_flow_on",
        "physics-flow-raw-seed1234-u000200",
        True,
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


@dataclass(frozen=True)
class Endpoint:
    code: str
    arm: str
    nfe: int
    condition_source: str
    primary: bool


ENDPOINTS = tuple(
    Endpoint(f"matched_off_nfe_{nfe}", "FLOW-OFF", nfe, "off", nfe == 1)
    for nfe in NFE_GRID
) + tuple(
    Endpoint(
        f"raw_checkpoint_{source}_nfe_{nfe}",
        "RAW-FLOW",
        nfe,
        source,
        nfe == 1,
    )
    for source in RUNTIME_SOURCES
    for nfe in NFE_GRID
) + tuple(
    Endpoint(
        f"parent_vpm_off_nfe_{nfe}",
        PARENT_EVALUATION_MODEL_CODE,
        nfe,
        "off",
        nfe == 1,
    )
    for nfe in NFE_GRID
)
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
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
        "identity_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    observed = payload.get("identity_sha256")
    if not isinstance(observed, str) or SHA256_RE.fullmatch(observed) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == observed


def sha256_file(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, digest: bool = True) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PhysicsFlowStage1Error(f"regular absolute file required: {path}")
    path = path.resolve(strict=True)
    result: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    if digest:
        result["sha256"] = sha256_file(path)
    return result


def noncontent_file_stat(path: Path) -> dict[str, Any]:
    """Record file identity metadata without reading any content bytes."""

    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PhysicsFlowStage1Error(f"regular absolute file required: {path}")
    path = path.resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "content_bytes_read_for_provenance": False,
    }


def read_json(path: Path, label: str = "JSON") -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicsFlowStage1Error(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PhysicsFlowStage1Error(f"{label} root must be an object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PhysicsFlowStage1Error(
                    f"invalid JSONL row {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise PhysicsFlowStage1Error(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def exclusive_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PhysicsFlowStage1Error(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    exclusive_bytes(path, canonical_json(payload) + b"\n")


def exclusive_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    exclusive_bytes(path, b"".join(canonical_json(row) + b"\n" for row in rows))


def git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise PhysicsFlowStage1Error(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def clean_repository(repo: Path, expected_commit: str, label: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise PhysicsFlowStage1Error(f"{label} commit must be a full SHA")
    if git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise PhysicsFlowStage1Error(f"{label} path is not a worktree root")
    if git(repo, "rev-parse", "HEAD") != expected_commit:
        raise PhysicsFlowStage1Error(f"{label} commit differs")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise PhysicsFlowStage1Error(f"{label} worktree is dirty")
    return {
        "path": str(repo),
        "git_commit": expected_commit,
        "git_tree_sha": git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def fresh_lustre_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise PhysicsFlowStage1Error("output must be an absolute named path")
    parent = path.parent.resolve(strict=True)
    canonical = parent / path.name
    if canonical != path or Path("/lustre") not in canonical.parents:
        raise PhysicsFlowStage1Error("output must be canonical under /lustre")
    if canonical.exists() or canonical.is_symlink():
        raise PhysicsFlowStage1Error(f"fresh output already exists: {canonical}")
    return canonical


def require_identity(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if not identity_valid(payload) or payload.get("identity_sha256") != expected:
        raise PhysicsFlowStage1Error(f"{label} identity differs")


def require_false_flags(value: Any, location: str = "root") -> int:
    count = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "protected_test_accessed":
                count += 1
                if child is not False:
                    raise PhysicsFlowStage1Error(
                        f"{child_location} must explicitly equal false"
                    )
            count += require_false_flags(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += require_false_flags(child, f"{location}[{index}]")
    return count


def tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def torch_tensor_sha256(value: Any) -> str:
    """Hash a torch tensor, including scalar tensors, with parity semantics."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise PhysicsFlowStage1Error("torch tensor hash requires a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_renderer_gate(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files = {
        "registration": root / "registration.json",
        "preparation": root / "preparation.json",
        "analysis": root / "analysis.json",
        "completion": root / "run_complete.json",
        "audit": root / "audit_stdout.jsonl",
        "source": root / "trajectory_consistent_renderer_gate.py",
    }
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise PhysicsFlowStage1Error("renderer prerequisite inventory is incomplete")
    registration = read_json(files["registration"], "renderer registration")
    preparation = read_json(files["preparation"], "renderer preparation")
    analysis = read_json(files["analysis"], "renderer analysis")
    completion = read_json(files["completion"], "renderer completion")
    require_identity(registration, RENDERER_REGISTRATION_IDENTITY, "renderer registration")
    require_identity(preparation, RENDERER_PREPARATION_IDENTITY, "renderer preparation")
    require_identity(analysis, RENDERER_ANALYSIS_IDENTITY, "renderer analysis")
    require_identity(completion, RENDERER_COMPLETION_IDENTITY, "renderer completion")
    audit_rows = read_jsonl(files["audit"])
    if len(audit_rows) != 1:
        raise PhysicsFlowStage1Error("renderer audit must have exactly one row")
    audit = audit_rows[0]
    if (
        analysis.get("decision") != RENDERER_DECISION
        or analysis.get("family_gates", {}).get("raw_geometry_scaffold_pass") is not True
        or completion.get("decision") != RENDERER_DECISION
        or completion.get("status") != "completed"
        or audit.get("status") != "audit_passed"
        or audit.get("decision") != RENDERER_DECISION
        or audit.get("analysis_identity_sha256") != RENDERER_ANALYSIS_IDENTITY
        or audit.get("completion_identity_sha256") != RENDERER_COMPLETION_IDENTITY
        or audit.get("causal_prediction_replay_max_abs_error", math.inf) > 1e-6
        or int(audit.get("artifact_hashes_verified", 0)) < 100
    ):
        raise PhysicsFlowStage1Error("raw renderer prerequisite did not independently pass")
    # The prospective Wan screen inherits only the independently passed raw
    # family.  Recurrent and hybrid outcomes are explicitly outside scope.
    return {
        "root": str(root),
        "decision": RENDERER_DECISION,
        "source_commit": RENDERER_SOURCE_COMMIT,
        "registration_identity_sha256": RENDERER_REGISTRATION_IDENTITY,
        "preparation_identity_sha256": RENDERER_PREPARATION_IDENTITY,
        "analysis_identity_sha256": RENDERER_ANALYSIS_IDENTITY,
        "completion_identity_sha256": RENDERER_COMPLETION_IDENTITY,
        "raw_geometry_scaffold_pass": True,
        "recurrent_or_hybrid_imported": False,
        "audit_receipt": audit,
        "files": {name: file_record(path) for name, path in files.items()},
        "protected_test_accessed": False,
    }


def validate_manifest(
    path: Path, *, split: str, expected_count: int, expected_sha256: str
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise PhysicsFlowStage1Error(f"{split} manifest digest differs")
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise PhysicsFlowStage1Error(f"{split} manifest row count differs")
    episode_dirs = []
    clip_ids = []
    for index, row in enumerate(rows):
        start = int(row.get("start", -1))
        expected_indices = list(range(start, start + SAMPLE_SIZE * CHUNK_SIZE, CHUNK_SIZE))
        if (
            row.get("split") != split
            or row.get("auxiliary_index") != index
            or row.get("sample_size") != SAMPLE_SIZE
            or row.get("chunk_size") != CHUNK_SIZE
            or row.get("action_span") != SAMPLE_SIZE * CHUNK_SIZE
            or row.get("frame_indices") != expected_indices
            or not isinstance(row.get("clip_id"), str)
            or not Path(str(row.get("episode_dir", ""))).is_absolute()
        ):
            raise PhysicsFlowStage1Error(f"{split} manifest row {index} differs")
        episode_dirs.append(str(row["episode_dir"]))
        clip_ids.append(str(row["clip_id"]))
    if len(set(episode_dirs)) != expected_count or len(set(clip_ids)) != expected_count:
        raise PhysicsFlowStage1Error(f"{split} rows are not unique episodes/clips")
    return rows


def resolve_cache(metadata_path: Path, split: str, expected_count: int) -> dict[str, Any]:
    metadata = read_json(metadata_path, f"{split} RGB/action cache metadata")
    expected_manifest = TRAIN_MANIFEST_SHA256 if split == "train" else VAL_MANIFEST_SHA256
    expected_rgb = TRAIN_RGB_SHA256 if split == "train" else VAL_RGB_SHA256
    expected_actions = TRAIN_ACTIONS_SHA256 if split == "train" else VAL_ACTIONS_SHA256
    if (
        metadata.get("complete") is not True
        or metadata.get("split") != split
        or metadata.get("clip_count") != expected_count
        or metadata.get("clip_manifest_sha256") != expected_manifest
        or metadata.get("rgb_sha256") != expected_rgb
        or metadata.get("actions_sha256") != expected_actions
        or metadata.get("rgb_shape") != [expected_count, 13, 3, 180, 960]
        or metadata.get("actions_shape") != [expected_count, 13, 5, 23]
        or metadata.get("rgb_dtype") != "float16"
        or metadata.get("actions_dtype") != "float32"
    ):
        raise PhysicsFlowStage1Error(f"{split} immutable RGB/action cache differs")
    paths = {}
    for name, expected_digest in (("rgb", expected_rgb), ("actions", expected_actions)):
        value = Path(str(metadata[f"{name}_file"]))
        if not value.is_absolute():
            value = metadata_path.parent / value
        value = value.resolve(strict=True)
        if value.is_symlink() or sha256_file(value) != expected_digest:
            raise PhysicsFlowStage1Error(f"{split} {name} array digest differs")
        paths[name] = file_record(value, digest=False) | {"sha256": expected_digest}
    return {
        "split": split,
        "clip_count": expected_count,
        "metadata": file_record(metadata_path),
        "arrays": paths,
        "payload": metadata,
        "protected_test_accessed": False,
    }


def raw_mcap_path(row: Mapping[str, Any], preprocessed_root: Path, raw_root: Path) -> Path:
    episode = Path(str(row["episode_dir"])).resolve(strict=True)
    try:
        relative = episode.relative_to(preprocessed_root.resolve(strict=True))
    except ValueError as exc:
        raise PhysicsFlowStage1Error("episode lies outside registered preprocessed root") from exc
    path = raw_root.resolve(strict=True) / relative / "episode.mcap"
    if not path.is_file() or path.is_symlink():
        raise PhysicsFlowStage1Error(f"raw MCAP is unavailable: {path}")
    return path.resolve(strict=True)


def planned_motion(action: np.ndarray) -> float:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (SAMPLE_SIZE, CHUNK_SIZE, PADDED_ACTION_DIM):
        raise PhysicsFlowStage1Error(f"action row geometry differs: {value.shape}")
    future = value[4:12, -1, :12]
    return float(np.sqrt(np.mean(np.diff(np.concatenate((value[3:4, -1, :12], future)), axis=0) ** 2)))


def deterministic_donors(
    rows: Sequence[Mapping[str, Any]],
    actions: np.ndarray,
    eligible: Sequence[bool],
) -> tuple[list[int | None], list[int | None], list[float]]:
    motion = [planned_motion(actions[index]) for index in range(len(rows))]
    eligible_indexes = [index for index, value in enumerate(eligible) if value]
    if len(eligible_indexes) < max(3, MOTION_STRATA):
        raise PhysicsFlowStage1Error("too few D405 rows for stratified controls")
    sorted_indexes = sorted(
        eligible_indexes,
        key=lambda index: (motion[index], str(rows[index]["clip_id"])),
    )
    strata: list[int | None] = [None] * len(rows)
    buckets = np.array_split(np.asarray(sorted_indexes, dtype=np.int64), MOTION_STRATA)
    donors: list[int | None] = [None] * len(rows)
    for stratum, raw_bucket in enumerate(buckets):
        bucket = [int(value) for value in raw_bucket.tolist()]
        if len(bucket) < 2:
            raise PhysicsFlowStage1Error("motion stratum lacks two D405 episodes")
        ordered = sorted(
            bucket,
            key=lambda index: hashlib.sha256(
                f"{rows[index]['clip_id']}|{DONOR_SEED}|{stratum}".encode()
            ).hexdigest(),
        )
        for position, index in enumerate(ordered):
            donor = ordered[(position + 1) % len(ordered)]
            if donor == index or rows[donor]["episode_dir"] == rows[index]["episode_dir"]:
                raise PhysicsFlowStage1Error("shuffled donor is not episode-disjoint")
            strata[index] = stratum
            donors[index] = donor
    return donors, strata, motion


def _scan_split(
    *,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    preprocessed_root: Path,
    raw_root: Path,
) -> dict[str, Any]:
    actions_path = Path(str(cache["arrays"]["actions"]["path"]))
    actions = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(rows), SAMPLE_SIZE, CHUNK_SIZE, PADDED_ACTION_DIM)
    if actions.shape != expected_shape or actions.dtype != np.float32:
        raise PhysicsFlowStage1Error(f"{split} cached action geometry changed")
    if float(np.max(np.abs(actions[..., ACTION_DIM:]))) != 0.0:
        raise PhysicsFlowStage1Error(f"{split} padded actions are nonzero")
    camera_types: dict[str, int] = defaultdict(int)
    eligible: list[bool] = []
    mcap_records = []
    for index, row in enumerate(rows):
        mcap = raw_mcap_path(row, preprocessed_root, raw_root)
        metadata = read_mcap_metadata(mcap)
        camera_type = metadata.get("top_camera_type", "missing")
        camera_types[camera_type] += 1
        is_eligible = camera_type == CAMERA_TYPE
        calibration_identity = None
        if is_eligible:
            calibration = _decode_top_calibration(mcap)
            if calibration.get("camera_type") != CAMERA_TYPE:
                raise PhysicsFlowStage1Error(
                    f"{split} D405 calibration disagrees with metadata: {mcap}"
                )
            calibration_identity = hashlib.sha256(
                canonical_json(calibration)
            ).hexdigest()
        eligible.append(is_eligible)
        mcap_records.append(
            {
                "clip_index": index,
                "clip_id": row["clip_id"],
                "raw_mcap": str(mcap),
                "raw_mcap_bytes": mcap.stat().st_size,
                "raw_mcap_mtime_ns": mcap.stat().st_mtime_ns,
                "camera_type": camera_type,
                "d405_eligible": is_eligible,
                "calibration_identity_sha256": calibration_identity,
                "episode_metadata_sha256": hashlib.sha256(
                    canonical_json(metadata)
                ).hexdigest(),
                "protected_test_accessed": False,
            }
        )
    donors, strata, motion = deterministic_donors(rows, actions, eligible)
    descriptors = []
    for index, row in enumerate(rows):
        descriptor = {
            "clip_index": index,
            "clip_id": row["clip_id"],
            "episode_dir": row["episode_dir"],
            "frame_indices": row["frame_indices"],
            "start": row["start"],
            "planned_motion_rms": motion[index],
            "camera_type": mcap_records[index]["camera_type"],
            "d405_eligible": eligible[index],
            "motion_stratum": strata[index],
            "episode_shuffled_donor_index": donors[index],
            "episode_shuffled_donor_clip_id": (
                rows[int(donors[index])]["clip_id"] if donors[index] is not None else None
            ),
            "raw_mcap": mcap_records[index]["raw_mcap"],
            "raw_mcap_bytes": mcap_records[index]["raw_mcap_bytes"],
            "raw_mcap_mtime_ns": mcap_records[index]["raw_mcap_mtime_ns"],
            "episode_metadata_sha256": mcap_records[index]["episode_metadata_sha256"],
            "calibration_identity_sha256": mcap_records[index][
                "calibration_identity_sha256"
            ],
            "action_row_sha256": tensor_sha256(np.asarray(actions[index])),
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "protected_test_accessed": False,
        }
        descriptors.append(descriptor)
    return {
        "split": split,
        "clip_count": len(rows),
        "d405_count": int(sum(eligible)),
        "non_d405_count": int(len(rows) - sum(eligible)),
        "camera_type_counts": dict(sorted(camera_types.items())),
        "descriptors": descriptors,
        "selection_uses_rgb": False,
        "selection_uses_measured_state": False,
        "selection_uses_generator_outcome": False,
        "protected_test_accessed": False,
    }


def command_register_cache(args: argparse.Namespace) -> int:
    output = fresh_lustre_root(args.output)
    source = clean_repository(args.source_repo, args.expected_commit, "physics-flow source")
    official = clean_repository(args.official_abc_root, EXPECTED_ABC_COMMIT, "official ABC")
    renderer = validate_renderer_gate(args.renderer_gate)
    train_rows = validate_manifest(
        args.train_manifest,
        split="train",
        expected_count=TRAIN_COUNT,
        expected_sha256=TRAIN_MANIFEST_SHA256,
    )
    val_rows = validate_manifest(
        args.val_manifest,
        split="val",
        expected_count=VAL_COUNT,
        expected_sha256=VAL_MANIFEST_SHA256,
    )
    train_cache = resolve_cache(args.train_cache_metadata, "train", TRAIN_COUNT)
    val_cache = resolve_cache(args.val_cache_metadata, "val", VAL_COUNT)
    train = _scan_split(
        split="train",
        rows=train_rows,
        cache=train_cache,
        preprocessed_root=args.preprocessed_root,
        raw_root=args.raw_root,
    )
    validation = _scan_split(
        split="val",
        rows=val_rows,
        cache=val_cache,
        preprocessed_root=args.preprocessed_root,
        raw_root=args.raw_root,
    )
    if validation["d405_count"] < 32:
        raise PhysicsFlowStage1Error(
            f"D405 validation population {validation['d405_count']} is below 32"
        )
    train_episode = {row["episode_dir"] for row in train_rows}
    val_episode = {row["episode_dir"] for row in val_rows}
    if train_episode & val_episode:
        raise PhysicsFlowStage1Error("train/validation episodes overlap")
    output.mkdir(mode=0o700)
    registration = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": CACHE_REGISTRATION_KIND,
            "status": "registered_before_flow_rendering_or_video_model_outcomes",
            "created_at_utc": now(),
            "output_root": str(output),
            "source_repository": source,
            "official_abc_repository": official,
            "renderer_prerequisite": renderer,
            "inputs": {
                "preprocessed_root": str(args.preprocessed_root.resolve(strict=True)),
                "raw_root": str(args.raw_root.resolve(strict=True)),
                "train": {
                    "manifest": file_record(args.train_manifest),
                    "cache": train_cache,
                },
                "validation": {
                    "manifest": file_record(args.val_manifest),
                    "cache": val_cache,
                },
            },
            "train_validation_episode_overlap_count": 0,
            "splits": {"train": train, "val": validation},
            "frozen_representation": {
                "native_render_shape": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
                "compact_transition_shape": [
                    FUTURE_TRANSITIONS,
                    FLOW_COMPONENTS,
                    FLOW_HEIGHT,
                    FLOW_WIDTH,
                ],
                "packed_training_shape": [16, 4, 24, 120],
                "components": [
                    "dx_over_width",
                    "dy_over_height",
                    "visibility_fraction",
                    "log_depth_ratio",
                ],
                "trajectory": (
                    "observed frame-4 state followed by raw candidate-action "
                    "chunk endpoints 4..11"
                ),
                "bottom_padding_rows": PADDED_HEIGHT - OUTPUT_HEIGHT,
                "pooling": (
                    "8x8 support-weighted dx/dy/log-depth and area visibility"
                ),
                "visibility_tolerance": {
                    "absolute_m": VISIBILITY_ABS_TOLERANCE_M,
                    "relative": VISIBILITY_REL_TOLERANCE,
                },
                "camera": (
                    "recorded D405 vertical focal length scaled to 180 rows; "
                    "fixed official nominal extrinsics; centered principal point; no distortion"
                ),
                "non_d405_policy": "all four caches exact zero with explicit ineligible row",
                "future_rgb": False,
                "future_measured_state": False,
            },
            "controls": {
                "raw": "native causal raw-command trajectory",
                "episode_shuffled": (
                    "complete compact raw tensor from a different D405 episode in the "
                    "same prospectively frozen planned-motion stratum"
                ),
                "timeshift_plus_one": (
                    "native raw transitions shifted one future slot without wrap; final zero"
                ),
                "hold_current": "observed frame-4 pose repeated for all nine path poses",
                "off": "runtime exact-zero packed tensor; not redundantly stored",
                "donor_seed": DONOR_SEED,
            },
            "causal_inputs_only": True,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
    )
    exclusive_json(output / "cache_registration.json", registration)
    print(json.dumps(registration, sort_keys=True))
    return 0


def validate_cache_registration(path: Path) -> dict[str, Any]:
    registration = read_json(path, "cache registration")
    if (
        not identity_valid(registration)
        or registration.get("kind") != CACHE_REGISTRATION_KIND
        or registration.get("status")
        != "registered_before_flow_rendering_or_video_model_outcomes"
        or registration.get("causal_inputs_only") is not True
        or registration.get("future_rgb_opened") is not False
        or registration.get("future_measured_state_opened") is not False
        or registration.get("generator_outcome_opened") is not False
        or registration.get("protected_test_accessed") is not False
        or registration.get("renderer_prerequisite", {}).get("decision")
        != RENDERER_DECISION
        or registration.get("renderer_prerequisite", {}).get(
            "analysis_identity_sha256"
        )
        != RENDERER_ANALYSIS_IDENTITY
        or registration.get("renderer_prerequisite", {}).get(
            "raw_geometry_scaffold_pass"
        )
        is not True
        or registration.get("renderer_prerequisite", {}).get(
            "recurrent_or_hybrid_imported"
        )
        is not False
    ):
        raise PhysicsFlowStage1Error("cache registration differs")
    source = registration.get("source_repository", {})
    observed_source = clean_repository(
        Path(str(source.get("path", ""))),
        str(source.get("git_commit", "")),
        "physics-flow source",
    )
    if observed_source != source:
        raise PhysicsFlowStage1Error("cache source repository changed after registration")
    official = registration.get("official_abc_repository", {})
    observed_official = clean_repository(
        Path(str(official.get("path", ""))),
        EXPECTED_ABC_COMMIT,
        "official ABC",
    )
    if observed_official != official:
        raise PhysicsFlowStage1Error("official ABC repository changed after registration")
    output = Path(str(registration.get("output_root", ""))).resolve(strict=True)
    if path.resolve(strict=True) != output / "cache_registration.json":
        raise PhysicsFlowStage1Error("cache registration path is noncanonical")
    return registration


def _state_and_actions_for_row(
    *,
    row: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    cached_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read only observed frame-4 state and candidate action samples.

    The function intentionally never indexes joint/gripper state at frames
    five through twelve.  Raw action equality is checked over the complete
    registered candidate span because those actions are sampler inputs, not
    future observations.
    """

    if tensor_sha256(np.asarray(cached_action)) != descriptor.get(
        "action_row_sha256"
    ):
        raise PhysicsFlowStage1Error(
            f"registered candidate-action row changed for {row['clip_id']}"
        )
    state_path = Path(str(row["episode_dir"])) / "states.npz"
    if not state_path.is_file() or state_path.is_symlink():
        raise PhysicsFlowStage1Error(f"states file unavailable: {state_path}")
    frame4 = int(row["frame_indices"][4])
    start = int(row["start"])
    stop = start + SAMPLE_SIZE * CHUNK_SIZE
    with np.load(state_path, allow_pickle=False) as state:
        required = {
            "joint_states",
            "gripper_states",
            "joint_actions",
            "gripper_actions",
        }
        if not required.issubset(state.files):
            raise PhysicsFlowStage1Error(
                f"states file lacks {sorted(required - set(state.files))}: {state_path}"
            )
        # Only this one observed state boundary is read.
        q4 = np.concatenate(
            (
                np.asarray(state["joint_states"][frame4], dtype=np.float32),
                np.asarray(state["gripper_states"][frame4], dtype=np.float32),
            )
        )
        raw_action = np.concatenate(
            (
                np.asarray(state["joint_actions"][start:stop], dtype=np.float32),
                np.asarray(state["gripper_actions"][start:stop], dtype=np.float32),
            ),
            axis=1,
        ).reshape(SAMPLE_SIZE, CHUNK_SIZE, ACTION_DIM)
    cached = np.asarray(cached_action[..., :ACTION_DIM], dtype=np.float32)
    if not np.array_equal(raw_action, cached):
        raise PhysicsFlowStage1Error(
            f"cached/raw candidate actions differ for {row['clip_id']}"
        )
    if q4.shape != (ACTION_DIM,) or not np.isfinite(q4).all():
        raise PhysicsFlowStage1Error("observed frame-4 robot state is invalid")
    future_endpoints = cached[4:12, -1]
    if future_endpoints.shape != (FUTURE_TRANSITIONS, ACTION_DIM):
        raise PhysicsFlowStage1Error("future action endpoint geometry differs")
    provenance = {
        "states_npz": {
            **noncontent_file_stat(state_path),
            "only_arrays_indexed": {
                "joint_states": {"indexes": [frame4]},
                "gripper_states": {"indexes": [frame4]},
                "joint_actions": {
                    "slice_start_inclusive": start,
                    "slice_stop_exclusive": stop,
                },
                "gripper_actions": {
                    "slice_start_inclusive": start,
                    "slice_stop_exclusive": stop,
                },
            },
            "future_measured_state_values_indexed": False,
            "future_measured_state_opened": False,
        },
        "observed_frame4_state_sha256": tensor_sha256(q4),
        "candidate_action_span_sha256": tensor_sha256(cached),
        "candidate_action_endpoints_sha256": tensor_sha256(future_endpoints),
        "registered_action_row_sha256": descriptor["action_row_sha256"],
        "cache_raw_action_max_abs": 0.0,
        "future_rgb_opened": False,
        "future_measured_state_opened": False,
        "protected_test_accessed": False,
    }
    return q4, future_endpoints, provenance


def _local_points(
    source: Any,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width))
    selection = (
        source.mask
        & np.isfinite(source.depth)
        & (source.depth > 1e-6)
    )
    y = yy[selection].astype(np.float64)
    x = xx[selection].astype(np.float64)
    depth = source.depth[selection].astype(np.float64)
    geom_ids = source.geom_id[selection].astype(np.int64)
    if len(x) == 0:
        raise PhysicsFlowStage1Error("articulated source support is empty")
    camera = camera_points_from_depth(
        x,
        y,
        depth,
        width=width,
        height=height,
        fy=source.fy,
    )
    world = source.camera_xpos + camera @ source.camera_xmat.T
    local = np.empty_like(world)
    for geom_id in np.unique(geom_ids):
        keep = geom_ids == geom_id
        local[keep] = (
            (world[keep] - source.geom_xpos[int(geom_id)])
            @ source.geom_xmat[int(geom_id)]
        )
    return np.stack((x, y), axis=1), depth, geom_ids, local


def dense_transition_field(
    source: Any,
    target: Any,
    *,
    width: int = OUTPUT_WIDTH,
    height: int = OUTPUT_HEIGHT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transport every source-visible articulated pixel to the target pose."""

    source_xy, source_depth, geom_ids, local = _local_points(
        source, width=width, height=height
    )
    source_world = transport_local_points(local, geom_ids, source)
    target_world = transport_local_points(local, geom_ids, target)
    source_projected, source_valid = project_world(
        source_world, source, width=width, height=height
    )
    target_xy, target_valid = project_world(
        target_world, target, width=width, height=height
    )
    target_camera = (target_world - target.camera_xpos) @ target.camera_xmat
    target_forward = -target_camera[:, 2]
    in_frame = (
        target_valid
        & (target_xy[:, 0] >= 0.0)
        & (target_xy[:, 0] <= width - 1.0)
        & (target_xy[:, 1] >= 0.0)
        & (target_xy[:, 1] <= height - 1.0)
    )
    nearest_x = np.rint(np.clip(target_xy[:, 0], 0, width - 1)).astype(np.int64)
    nearest_y = np.rint(np.clip(target_xy[:, 1], 0, height - 1)).astype(np.int64)
    z_buffer = target.depth[nearest_y, nearest_x].astype(np.float64)
    tolerance = np.maximum(
        VISIBILITY_ABS_TOLERANCE_M,
        VISIBILITY_REL_TOLERANCE * np.maximum(target_forward, 0.0),
    )
    visible = (
        source_valid
        & in_frame
        & np.isfinite(target_forward)
        & (target_forward > 1e-6)
        & np.isfinite(z_buffer)
        & (z_buffer > 1e-6)
        & (np.abs(z_buffer - target_forward) <= tolerance)
    )
    field = np.zeros((FLOW_COMPONENTS, height, width), dtype=np.float32)
    source_pixel_x = source_xy[:, 0].astype(np.int64)
    source_pixel_y = source_xy[:, 1].astype(np.int64)
    dx = (target_xy[:, 0] - source_projected[:, 0]) / float(width)
    dy = (target_xy[:, 1] - source_projected[:, 1]) / float(height)
    log_depth = np.log((target_forward + 1e-6) / (source_depth + 1e-6))
    valid_indexes = np.flatnonzero(visible)
    if len(valid_indexes):
        yy = source_pixel_y[valid_indexes]
        xx = source_pixel_x[valid_indexes]
        field[0, yy, xx] = dx[valid_indexes].astype(np.float32)
        field[1, yy, xx] = dy[valid_indexes].astype(np.float32)
        field[2, yy, xx] = 1.0
        field[3, yy, xx] = log_depth[valid_indexes].astype(np.float32)
    if not np.isfinite(field).all() or np.max(np.abs(field[:2])) > 1.0 + 1e-6:
        raise PhysicsFlowStage1Error("native dense flow is nonfinite or out of bounds")
    reprojection = np.linalg.norm(source_projected - source_xy, axis=1)
    return field, {
        "source_articulated_pixels": int(len(source_xy)),
        "target_in_frame_pixels": int(in_frame.sum()),
        "zbuffer_visible_pixels": int(visible.sum()),
        "visibility_fraction_of_source": float(visible.mean()),
        "source_reprojection_rmse_px": float(
            np.sqrt(np.mean(reprojection**2))
        ),
        "mean_visible_motion_px": (
            float(
                np.mean(
                    np.linalg.norm(
                        target_xy[visible] - source_projected[visible], axis=1
                    )
                )
            )
            if np.any(visible)
            else 0.0
        ),
        "protected_test_accessed": False,
    }


def pool_transition_field(field: np.ndarray) -> np.ndarray:
    """Bottom-pad and support-pool one native field into 24x40."""

    value = np.asarray(field, dtype=np.float32)
    if value.shape != (FLOW_COMPONENTS, OUTPUT_HEIGHT, OUTPUT_WIDTH):
        raise PhysicsFlowStage1Error(f"native field shape differs: {value.shape}")
    padded = np.zeros((FLOW_COMPONENTS, PADDED_HEIGHT, OUTPUT_WIDTH), dtype=np.float32)
    padded[:, :OUTPUT_HEIGHT] = value
    blocks = padded.reshape(FLOW_COMPONENTS, FLOW_HEIGHT, POOL, FLOW_WIDTH, POOL)
    visibility = blocks[2].sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    pooled = np.zeros((FLOW_COMPONENTS, FLOW_HEIGHT, FLOW_WIDTH), dtype=np.float32)
    pooled[2] = visibility / float(POOL * POOL)
    supported = visibility > 0
    for component in (0, 1, 3):
        weighted = (blocks[component] * blocks[2]).sum(
            axis=(1, 3), dtype=np.float64
        ).astype(np.float32)
        pooled[component, supported] = weighted[supported] / visibility[supported]
    if (
        not np.isfinite(pooled).all()
        or float(pooled[2].min()) < 0.0
        or float(pooled[2].max()) > 1.0
        or float(np.max(np.abs(pooled[:2]))) > 1.0 + 1e-6
    ):
        raise PhysicsFlowStage1Error("pooled transition is invalid")
    unsupported = pooled[2] == 0
    if any(np.any(pooled[component][unsupported] != 0) for component in (0, 1, 3)):
        raise PhysicsFlowStage1Error("unsupported pooled components are nonzero")
    return pooled


def validate_compact_flow(value: np.ndarray, *, count: int | None = None) -> None:
    array = np.asarray(value)
    expected_tail = (FUTURE_TRANSITIONS, FLOW_COMPONENTS, FLOW_HEIGHT, FLOW_WIDTH)
    if (
        array.ndim != 5
        or array.shape[1:] != expected_tail
        or (count is not None and array.shape[0] != count)
        or array.dtype != np.float16
        or not np.isfinite(array).all()
    ):
        raise PhysicsFlowStage1Error(
            f"compact flow contract differs: {array.shape} {array.dtype}"
        )
    if float(array[:, :, 2].min()) < 0 or float(array[:, :, 2].max()) > 1:
        raise PhysicsFlowStage1Error("compact visibility is outside [0,1]")
    if float(np.max(np.abs(array[:, :, :2]))) > 1:
        raise PhysicsFlowStage1Error("compact displacement is outside [-1,1]")
    unsupported = array[:, :, 2] == 0
    for component in (0, 1, 3):
        if np.any(array[:, :, component][unsupported] != 0):
            raise PhysicsFlowStage1Error("compact unsupported components are nonzero")


def nonwrapping_timeshift(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-4:] != (
        FUTURE_TRANSITIONS,
        FLOW_COMPONENTS,
        FLOW_HEIGHT,
        FLOW_WIDTH,
    ):
        raise PhysicsFlowStage1Error("time-shift input geometry differs")
    result = np.zeros_like(array)
    result[..., :-1, :, :, :] = array[..., 1:, :, :, :]
    return result


def _render_trajectory(
    *,
    poses: np.ndarray,
    model: Any,
    data: Any,
    mujoco: Any,
    renderer: Any,
    option: Any,
    addresses: Mapping[str, int],
    camera_id: int,
    moving_geoms: set[int],
    fy: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    renders = []
    for pose in poses:
        renders.append(
            render_pose(
                model=model,
                data=data,
                mujoco=mujoco,
                renderer=renderer,
                option=option,
                addresses=dict(addresses),
                camera_id=camera_id,
                moving_geoms=moving_geoms,
                pose=pose,
                fy=fy,
            )
        )
    transitions = []
    diagnostics = []
    for index in range(FUTURE_TRANSITIONS):
        native, diagnostic = dense_transition_field(renders[index], renders[index + 1])
        transitions.append(pool_transition_field(native))
        diagnostics.append({"transition": index, **diagnostic})
    stacked = np.stack(transitions).astype(np.float16)
    validate_compact_flow(stacked[None], count=1)
    return stacked, {
        "pose_render_ms": [float(value.render_ms) for value in renders],
        "transitions": diagnostics,
        "protected_test_accessed": False,
    }


def _create_renderer(registration: Mapping[str, Any]) -> tuple[Any, ...]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except ImportError as exc:
        raise PhysicsFlowStage1Error("cache rendering requires mujoco") from exc
    official = Path(registration["official_abc_repository"]["path"])
    xml = build_robot_only_xml(
        official / OFFICIAL_SCENE_RELATIVE,
        official / OFFICIAL_ASSET_RELATIVE,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=OUTPUT_HEIGHT, width=OUTPUT_WIDTH)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    if camera_id < 0:
        raise PhysicsFlowStage1Error("official ABC model lacks top camera")
    moving = moving_geom_ids(model, mujoco)
    return model, data, mujoco, renderer, option, addresses, camera_id, moving


def _load_split_registration(
    registration: Mapping[str, Any], split: str
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    input_key = "train" if split == "train" else "validation"
    path = Path(registration["inputs"][input_key]["manifest"]["path"])
    rows = read_jsonl(path)
    descriptors = registration["splits"][split]["descriptors"]
    if len(rows) != len(descriptors):
        raise PhysicsFlowStage1Error("registered split inventory differs")
    action_path = Path(
        registration["inputs"][input_key]["cache"]["arrays"]["actions"]["path"]
    )
    actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
    return rows, actions, descriptors


def command_build_cache(args: argparse.Namespace) -> int:
    registration = validate_cache_registration(args.registration)
    split = args.split
    expected_count = TRAIN_COUNT if split == "train" else VAL_COUNT
    split_root = Path(registration["output_root"]) / split
    if split_root.exists() or split_root.is_symlink():
        raise PhysicsFlowStage1Error(f"fresh split cache already exists: {split_root}")
    split_root.mkdir(mode=0o700)
    rows, actions, descriptors = _load_split_registration(registration, split)
    arrays = {
        source: np.lib.format.open_memmap(
            split_root / f"{source}.float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                expected_count,
                FUTURE_TRANSITIONS,
                FLOW_COMPONENTS,
                FLOW_HEIGHT,
                FLOW_WIDTH,
            ),
        )
        for source in CACHE_SOURCES
    }
    for array in arrays.values():
        array[:] = 0
        array.flush()
    lineage_path = split_root / "row_lineage.jsonl"
    lineage_rows = []
    model, data, mujoco, renderer, option, addresses, camera_id, moving = (
        _create_renderer(registration)
    )
    render_started = time.perf_counter()
    try:
        for index, (row, descriptor) in enumerate(zip(rows, descriptors)):
            if (
                descriptor.get("clip_index") != index
                or descriptor.get("clip_id") != row.get("clip_id")
            ):
                raise PhysicsFlowStage1Error("registered descriptor order changed")
            q4, endpoints, provenance = _state_and_actions_for_row(
                row=row,
                descriptor=descriptor,
                cached_action=np.asarray(actions[index]),
            )
            raw_flow = np.zeros(
                (FUTURE_TRANSITIONS, FLOW_COMPONENTS, FLOW_HEIGHT, FLOW_WIDTH),
                dtype=np.float16,
            )
            hold_flow = np.zeros_like(raw_flow)
            calibration_record: dict[str, Any] | None = None
            render_diagnostics: dict[str, Any] | None = None
            hold_diagnostics: dict[str, Any] | None = None
            if bool(descriptor["d405_eligible"]):
                mcap = Path(descriptor["raw_mcap"])
                stat = mcap.stat()
                if (
                    stat.st_size != descriptor["raw_mcap_bytes"]
                    or stat.st_mtime_ns != descriptor["raw_mcap_mtime_ns"]
                ):
                    raise PhysicsFlowStage1Error("registered MCAP stat changed")
                calibration = _decode_top_calibration(mcap)
                if calibration.get("camera_type") != CAMERA_TYPE:
                    raise PhysicsFlowStage1Error("D405 calibration eligibility changed")
                calibration_identity = hashlib.sha256(
                    canonical_json(calibration)
                ).hexdigest()
                if calibration_identity != descriptor.get(
                    "calibration_identity_sha256"
                ):
                    raise PhysicsFlowStage1Error(
                        "registered D405 calibration changed"
                    )
                native_height = int(calibration["camera_height"])
                native_width = int(calibration["camera_width"])
                K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
                fy = float(K[1, 1] * OUTPUT_HEIGHT / native_height)
                if not math.isfinite(fy) or fy <= 0:
                    raise PhysicsFlowStage1Error("scaled D405 focal length is invalid")
                model.cam_fovy[camera_id] = math.degrees(
                    2.0 * math.atan(OUTPUT_HEIGHT / (2.0 * fy))
                )
                raw_pose = np.concatenate((q4[None], endpoints), axis=0).astype(np.float32)
                raw_pose[:, 12:] = np.clip(raw_pose[:, 12:], 0.0, 1.0)
                hold_pose = np.broadcast_to(q4, (9, ACTION_DIM)).copy().astype(np.float32)
                hold_pose[:, 12:] = np.clip(hold_pose[:, 12:], 0.0, 1.0)
                raw_flow, render_diagnostics = _render_trajectory(
                    poses=raw_pose,
                    model=model,
                    data=data,
                    mujoco=mujoco,
                    renderer=renderer,
                    option=option,
                    addresses=addresses,
                    camera_id=camera_id,
                    moving_geoms=moving,
                    fy=fy,
                )
                hold_flow, hold_diagnostics = _render_trajectory(
                    poses=hold_pose,
                    model=model,
                    data=data,
                    mujoco=mujoco,
                    renderer=renderer,
                    option=option,
                    addresses=addresses,
                    camera_id=camera_id,
                    moving_geoms=moving,
                    fy=fy,
                )
                # Numerical camera projection should make all stationary
                # displacements/depth ratios exact enough to canonicalize to 0.
                hold_flow[:, (0, 1, 3)] = 0
                calibration_record = {
                    "camera_type": calibration["camera_type"],
                    "native_width": native_width,
                    "native_height": native_height,
                    "K": calibration["K"],
                    "D": calibration["D"],
                    "distortion_model": calibration["distortion_model"],
                    "scaled_fy": fy,
                    "calibration_identity_sha256": calibration_identity,
                    "center_principal_offset_native_px": [
                        float(K[0, 2] - (native_width - 1) / 2),
                        float(K[1, 2] - (native_height - 1) / 2),
                    ],
                    "distortion_applied": False,
                    "protected_test_accessed": False,
                }
            arrays["raw"][index] = raw_flow
            arrays["hold_current"][index] = hold_flow
            arrays["timeshift_plus_one"][index] = nonwrapping_timeshift(raw_flow)
            for source in ("raw", "hold_current", "timeshift_plus_one"):
                arrays[source].flush()
            lineage_rows.append(
                identity_payload(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "raw_physics_flow_cache_row",
                        "cache_registration_identity_sha256": registration[
                            "identity_sha256"
                        ],
                        "split": split,
                        "clip_index": index,
                        "clip_id": row["clip_id"],
                        "episode_dir": row["episode_dir"],
                        "frame_indices": row["frame_indices"],
                        "action_window": [int(row["start"]), int(row["start"]) + 65],
                        "d405_eligible": descriptor["d405_eligible"],
                        "camera_type": descriptor["camera_type"],
                        "motion_stratum": descriptor["motion_stratum"],
                        "episode_shuffled_donor_index": descriptor[
                            "episode_shuffled_donor_index"
                        ],
                        "episode_shuffled_donor_clip_id": descriptor[
                            "episode_shuffled_donor_clip_id"
                        ],
                        "input_provenance": provenance,
                        "calibration": calibration_record,
                        "raw_pose_path_sha256": tensor_sha256(
                            np.concatenate((q4[None], endpoints), axis=0).astype(np.float32)
                        ),
                        "tensor_sha256": {
                            "raw": tensor_sha256(raw_flow),
                            "timeshift_plus_one": tensor_sha256(
                                arrays["timeshift_plus_one"][index]
                            ),
                            "hold_current": tensor_sha256(hold_flow),
                        },
                        "render_diagnostics": render_diagnostics,
                        "hold_render_diagnostics": hold_diagnostics,
                        "causal_inputs_only": True,
                        "future_rgb_opened": False,
                        "future_measured_state_opened": False,
                        "generator_outcome_opened": False,
                        "protected_test_accessed": False,
                    }
                )
            )
            if (index + 1) % 16 == 0 or index + 1 == expected_count:
                print(
                    json.dumps(
                        {
                            "event": "flow_cache_progress",
                            "split": split,
                            "completed": index + 1,
                            "total": expected_count,
                            "d405_completed": sum(
                                bool(value["d405_eligible"])
                                for value in descriptors[: index + 1]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        renderer.close()
    # Donor materialization is intentionally delayed until every native RAW row
    # exists, avoiding any order-dependent fallback.
    for index, descriptor in enumerate(descriptors):
        donor = descriptor["episode_shuffled_donor_index"]
        if donor is not None:
            arrays["episode_shuffled"][index] = arrays["raw"][int(donor)]
        else:
            arrays["episode_shuffled"][index] = 0
    for array in arrays.values():
        array.flush()
    del arrays
    arrays_records = {}
    for source in CACHE_SOURCES:
        path = split_root / f"{source}.float16.npy"
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        validate_compact_flow(value, count=expected_count)
        arrays_records[source] = file_record(path)
    # Add the donor hash only after the donor array is finalized.
    for index, row in enumerate(lineage_rows):
        donor_value = np.load(
            split_root / "episode_shuffled.float16.npy", mmap_mode="r", allow_pickle=False
        )[index]
        unsigned = dict(row)
        unsigned.pop("identity_sha256", None)
        unsigned["tensor_sha256"] = {
            **unsigned["tensor_sha256"],
            "episode_shuffled": tensor_sha256(donor_value),
        }
        lineage_rows[index] = identity_payload(unsigned)
    exclusive_jsonl(lineage_path, lineage_rows)
    metadata = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "schema": CACHE_SCHEMA,
            "kind": CACHE_METADATA_KIND,
            "status": "complete",
            "complete": True,
            "created_at_utc": now(),
            "cache_registration": file_record(args.registration),
            "cache_registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["source_repository"]["git_commit"],
            "official_abc_commit": EXPECTED_ABC_COMMIT,
            "renderer_analysis_identity_sha256": RENDERER_ANALYSIS_IDENTITY,
            "renderer_decision": RENDERER_DECISION,
            "renderer_family": "raw_geometry_scaffold",
            "split": split,
            "clip_count": expected_count,
            "d405_count": registration["splits"][split]["d405_count"],
            "non_d405_count": registration["splits"][split]["non_d405_count"],
            "compact_transition_shape": [
                expected_count,
                FUTURE_TRANSITIONS,
                FLOW_COMPONENTS,
                FLOW_HEIGHT,
                FLOW_WIDTH,
            ],
            "flow_dtype": "float16",
            "arrays": arrays_records,
            "aligned_flow_file": arrays_records["raw"]["path"],
            "aligned_flow_sha256": arrays_records["raw"]["sha256"],
            "row_lineage": file_record(lineage_path),
            "render_wall_seconds": time.perf_counter() - render_started,
            "causal_inputs_only": True,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(split_root / "metadata.json", metadata)
    completion = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "raw_physics_flow_cache_complete",
            "status": "completed",
            "split": split,
            "cache_registration_identity_sha256": registration["identity_sha256"],
            "cache_metadata_identity_sha256": metadata["identity_sha256"],
            "metadata": file_record(split_root / "metadata.json"),
            "arrays": arrays_records,
            "row_lineage": file_record(lineage_path),
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(split_root / "complete.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


def load_cache_metadata(path: Path, *, split: str | None = None) -> dict[str, Any]:
    metadata = read_json(path, "flow cache metadata")
    if (
        not identity_valid(metadata)
        or metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("schema") != CACHE_SCHEMA
        or metadata.get("kind") != CACHE_METADATA_KIND
        or metadata.get("status") != "complete"
        or metadata.get("complete") is not True
        or (split is not None and metadata.get("split") != split)
        or metadata.get("renderer_analysis_identity_sha256")
        != RENDERER_ANALYSIS_IDENTITY
        or metadata.get("renderer_decision") != RENDERER_DECISION
        or metadata.get("renderer_family") != "raw_geometry_scaffold"
        or metadata.get("causal_inputs_only") is not True
        or metadata.get("future_rgb_opened") is not False
        or metadata.get("future_measured_state_opened") is not False
        or metadata.get("generator_outcome_opened") is not False
        or metadata.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error("flow cache metadata differs")
    expected_count = TRAIN_COUNT if metadata.get("split") == "train" else VAL_COUNT
    if (
        metadata.get("clip_count") != expected_count
        or metadata.get("compact_transition_shape")
        != [
            expected_count,
            FUTURE_TRANSITIONS,
            FLOW_COMPONENTS,
            FLOW_HEIGHT,
            FLOW_WIDTH,
        ]
        or metadata.get("flow_dtype") != "float16"
    ):
        raise PhysicsFlowStage1Error("flow cache tensor geometry differs")
    return metadata


def _resolve_record(
    record: Mapping[str, Any], *, parent: Path, label: str
) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = parent / path
    path = path.resolve(strict=True)
    if (
        not path.is_file()
        or path.is_symlink()
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != sha256_file(path)
    ):
        raise PhysicsFlowStage1Error(f"{label} artifact differs")
    return path


def audit_cache(metadata_path: Path, *, write: bool) -> dict[str, Any]:
    metadata_path = metadata_path.resolve(strict=True)
    metadata = load_cache_metadata(metadata_path)
    parent = metadata_path.parent
    registration_path = _resolve_record(
        metadata["cache_registration"], parent=parent, label="cache registration"
    )
    registration = validate_cache_registration(registration_path)
    if metadata["cache_registration_identity_sha256"] != registration["identity_sha256"]:
        raise PhysicsFlowStage1Error("cache registration identity linkage differs")
    split = str(metadata["split"])
    expected_count = TRAIN_COUNT if split == "train" else VAL_COUNT
    descriptors = registration["splits"][split]["descriptors"]
    arrays: dict[str, np.ndarray] = {}
    array_paths: dict[str, Path] = {}
    for source in CACHE_SOURCES:
        record = metadata.get("arrays", {}).get(source)
        if not isinstance(record, Mapping):
            raise PhysicsFlowStage1Error(f"cache lacks {source} record")
        path = _resolve_record(record, parent=parent, label=f"{source} array")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        validate_compact_flow(value, count=expected_count)
        arrays[source] = value
        array_paths[source] = path
    if (
        Path(str(metadata["aligned_flow_file"])).resolve(strict=True)
        != array_paths["raw"]
        or metadata["aligned_flow_sha256"] != metadata["arrays"]["raw"]["sha256"]
    ):
        raise PhysicsFlowStage1Error("aligned/raw alias differs")
    lineage_path = _resolve_record(
        metadata["row_lineage"], parent=parent, label="row lineage"
    )
    lineage = read_jsonl(lineage_path)
    if len(lineage) != expected_count:
        raise PhysicsFlowStage1Error("row lineage count differs")
    d405 = 0
    false_flags = 0
    row_hashes_verified = 0
    max_time_error = 0.0
    max_donor_error = 0.0
    max_hold_nonmotion = 0.0
    for index, (row, descriptor) in enumerate(zip(lineage, descriptors)):
        if (
            not identity_valid(row)
            or row.get("kind") != "raw_physics_flow_cache_row"
            or row.get("cache_registration_identity_sha256")
            != registration["identity_sha256"]
            or row.get("split") != split
            or row.get("clip_index") != index
            or row.get("clip_id") != descriptor["clip_id"]
            or row.get("d405_eligible") != descriptor["d405_eligible"]
            or (
                row.get("calibration", {}).get("calibration_identity_sha256")
                if isinstance(row.get("calibration"), Mapping)
                else None
            )
            != descriptor.get("calibration_identity_sha256")
            or row.get("episode_shuffled_donor_index")
            != descriptor["episode_shuffled_donor_index"]
            or row.get("causal_inputs_only") is not True
            or row.get("future_rgb_opened") is not False
            or row.get("future_measured_state_opened") is not False
            or row.get("generator_outcome_opened") is not False
            or row.get("protected_test_accessed") is not False
        ):
            raise PhysicsFlowStage1Error(f"lineage row {index} differs")
        false_flags += require_false_flags(row, f"lineage[{index}]")
        state_access = row.get("input_provenance", {}).get("states_npz", {})
        state_path = Path(str(descriptor["episode_dir"])) / "states.npz"
        expected_state_access = {
            **noncontent_file_stat(state_path),
            "only_arrays_indexed": {
                "joint_states": {"indexes": [int(descriptor["frame_indices"][4])]},
                "gripper_states": {
                    "indexes": [int(descriptor["frame_indices"][4])]
                },
                "joint_actions": {
                    "slice_start_inclusive": int(descriptor["start"]),
                    "slice_stop_exclusive": int(descriptor["start"]) + 65,
                },
                "gripper_actions": {
                    "slice_start_inclusive": int(descriptor["start"]),
                    "slice_stop_exclusive": int(descriptor["start"]) + 65,
                },
            },
            "future_measured_state_values_indexed": False,
            "future_measured_state_opened": False,
        }
        if state_access != expected_state_access or "sha256" in state_access:
            raise PhysicsFlowStage1Error(
                f"lineage state-access boundary differs: row={index}"
            )
        for source in CACHE_SOURCES:
            if row.get("tensor_sha256", {}).get(source) != tensor_sha256(
                arrays[source][index]
            ):
                raise PhysicsFlowStage1Error(
                    f"lineage tensor hash differs: row={index} source={source}"
                )
            row_hashes_verified += 1
        shifted = nonwrapping_timeshift(arrays["raw"][index])
        max_time_error = max(
            max_time_error,
            float(np.max(np.abs(shifted.astype(np.float32) - arrays["timeshift_plus_one"][index].astype(np.float32)))),
        )
        donor = descriptor["episode_shuffled_donor_index"]
        expected_donor = (
            arrays["raw"][int(donor)]
            if donor is not None
            else np.zeros_like(arrays["raw"][index])
        )
        max_donor_error = max(
            max_donor_error,
            float(np.max(np.abs(expected_donor.astype(np.float32) - arrays["episode_shuffled"][index].astype(np.float32)))),
        )
        max_hold_nonmotion = max(
            max_hold_nonmotion,
            float(np.max(np.abs(arrays["hold_current"][index, :, (0, 1, 3)].astype(np.float32)))),
        )
        if descriptor["d405_eligible"]:
            d405 += 1
            if (
                row.get("calibration") is None
                or row.get("render_diagnostics") is None
                or row.get("hold_render_diagnostics") is None
                or int(np.count_nonzero(arrays["raw"][index, :, 2])) == 0
                or int(np.count_nonzero(arrays["hold_current"][index, :, 2])) == 0
            ):
                raise PhysicsFlowStage1Error(f"D405 row {index} lacks rendered support")
        elif any(np.any(arrays[source][index] != 0) for source in CACHE_SOURCES):
            raise PhysicsFlowStage1Error(f"non-D405 row {index} is nonzero")
    if (
        d405 != metadata["d405_count"]
        or d405 != registration["splits"][split]["d405_count"]
        or (split == "val" and d405 < 32)
        or max_time_error != 0.0
        or max_donor_error != 0.0
        or max_hold_nonmotion != 0.0
    ):
        raise PhysicsFlowStage1Error("cache control reconstruction differs")
    completion_path = parent / "complete.json"
    completion = read_json(completion_path, "cache completion")
    if (
        not identity_valid(completion)
        or completion.get("kind") != "raw_physics_flow_cache_complete"
        or completion.get("status") != "completed"
        or completion.get("split") != split
        or completion.get("cache_metadata_identity_sha256")
        != metadata["identity_sha256"]
        or completion.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error("cache completion receipt differs")
    audit = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": CACHE_AUDIT_KIND,
            "status": "audit_passed",
            "audited_at_utc": now(),
            "split": split,
            "cache_registration_identity_sha256": registration["identity_sha256"],
            "cache_metadata_identity_sha256": metadata["identity_sha256"],
            "cache_completion_identity_sha256": completion["identity_sha256"],
            "clip_count": expected_count,
            "d405_count": d405,
            "array_hashes_verified": len(CACHE_SOURCES),
            "row_tensor_hashes_verified": row_hashes_verified,
            "lineage_identities_verified": expected_count,
            "deterministic_controls_reconstructed": list(CACHE_SOURCES[1:]),
            "timeshift_max_abs_error": max_time_error,
            "episode_shuffled_max_abs_error": max_donor_error,
            "hold_nonmotion_max_abs": max_hold_nonmotion,
            "explicit_false_flags": false_flags,
            "causal_inputs_only": True,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    if write:
        exclusive_json(parent / "audit.json", audit)
    return audit


def command_audit_cache(args: argparse.Namespace) -> int:
    audit = audit_cache(args.metadata, write=not args.read_only)
    print(json.dumps(audit, sort_keys=True))
    return 0


def _validated_cache_receipt(metadata_path: Path, split: str) -> dict[str, Any]:
    metadata_path = metadata_path.resolve(strict=True)
    metadata = load_cache_metadata(metadata_path, split=split)
    audit_path = metadata_path.parent / "audit.json"
    audit = read_json(audit_path, f"{split} cache audit")
    replay = audit_cache(metadata_path, write=False)
    # Audit timestamps differ on replay, so compare every invariant except time
    # and the resulting identity.
    for key, value in audit.items():
        if key in {"audited_at_utc", "identity_sha256"}:
            continue
        if replay.get(key) != value:
            raise PhysicsFlowStage1Error(f"{split} cache audit replay differs at {key}")
    if (
        not identity_valid(audit)
        or audit.get("kind") != CACHE_AUDIT_KIND
        or audit.get("status") != "audit_passed"
        or audit.get("cache_metadata_identity_sha256") != metadata["identity_sha256"]
    ):
        raise PhysicsFlowStage1Error(f"{split} cache audit receipt differs")
    return {
        "metadata": file_record(metadata_path),
        "metadata_identity_sha256": metadata["identity_sha256"],
        "audit": file_record(audit_path),
        "audit_identity_sha256": audit["identity_sha256"],
        "d405_count": metadata["d405_count"],
        "arrays": metadata["arrays"],
        "row_lineage": metadata["row_lineage"],
        "protected_test_accessed": False,
    }


def _registered_runtime(args: argparse.Namespace, source_repo: Path) -> dict[str, Any]:
    python = args.python.expanduser().resolve(strict=True)
    if not python.is_file() or python.is_symlink():
        raise PhysicsFlowStage1Error("registered Python runtime is invalid")
    wan = args.wan_dir.expanduser().resolve(strict=True)
    videox = args.videox_home.expanduser().resolve(strict=True)
    if not wan.is_dir() or wan.is_symlink() or not videox.is_dir() or videox.is_symlink():
        raise PhysicsFlowStage1Error("Wan/VideoX runtime directory is invalid")
    null_prompt = wan / "null_prompt_umt5.pt"
    scheduler = videox / "config" / "wan2.1" / "wan_civitai.yaml"
    if not null_prompt.is_file() or not scheduler.is_file():
        raise PhysicsFlowStage1Error("Wan null prompt or scheduler config is unavailable")
    lpips_helper = source_repo / "tools" / "physics_flow_lpips.py"
    if not lpips_helper.is_file() or lpips_helper.is_symlink():
        raise PhysicsFlowStage1Error("offline LPIPS receipt helper is unavailable")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    completed = subprocess.run(
        [str(python), str(lpips_helper), "--receipt-only"],
        cwd=source_repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode:
        raise PhysicsFlowStage1Error(
            "offline LPIPS instrument validation failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        lpips_receipt = json.loads(
            [line for line in completed.stdout.splitlines() if line.strip()][-1]
        )
    except (IndexError, json.JSONDecodeError) as exc:
        raise PhysicsFlowStage1Error("offline LPIPS receipt is invalid") from exc
    if (
        not isinstance(lpips_receipt, dict)
        or not identity_valid(lpips_receipt)
        or lpips_receipt.get("kind")
        != "raw_physics_flow_offline_lpips_alex_v1"
        or lpips_receipt.get("network_access_permitted") is not False
        or lpips_receipt.get("download_performed") is not False
        or lpips_receipt.get("protected_test_accessed") is not False
        or lpips_receipt.get("identity_sha256") != LPIPS_RECEIPT_IDENTITY
        or lpips_receipt.get("loaded_state_dict_sha256")
        != LPIPS_STATE_DICT_SHA256
        or lpips_receipt.get("torchvision_alexnet", {})
        .get("checkpoint", {})
        .get("sha256")
        != LPIPS_ALEXNET_CHECKPOINT_SHA256
        or lpips_receipt.get("versions", {}).get("lpips_distribution") != "0.1.4"
        or lpips_receipt.get("versions", {}).get("torch") != "2.7.1+cu128"
        or lpips_receipt.get("versions", {}).get("torchvision") != "0.22.1+cu128"
    ):
        raise PhysicsFlowStage1Error("offline LPIPS receipt contract differs")
    lpips_preflight = file_record(args.lpips_preflight_log)
    if (
        lpips_preflight["sha256"] != LPIPS_PREFLIGHT_LOG_SHA256
        or lpips_preflight["bytes"] != 6999
        or _last_json_object(args.lpips_preflight_log, "LPIPS preflight")
        != lpips_receipt
    ):
        raise PhysicsFlowStage1Error("offline LPIPS preflight receipt differs")
    return {
        "python": str(python),
        "wan_dir": str(wan),
        "videox_home": str(videox),
        "null_prompt": file_record(null_prompt),
        "null_prompt_provenance": (
            file_record(null_prompt.with_suffix(null_prompt.suffix + ".json"))
            if null_prompt.with_suffix(null_prompt.suffix + ".json").is_file()
            else None
        ),
        "scheduler_config": file_record(scheduler),
        "videox_commit": git(videox, "rev-parse", "HEAD"),
        "lpips_alex": lpips_receipt,
        "lpips_preflight_job_id": 507388,
        "lpips_preflight_log": lpips_preflight,
        "protected_test_accessed": False,
    }


def _study_source_files(repo: Path) -> dict[str, Any]:
    relative_paths = (
        "docs/experiments/PHYSICS_FLOW_WAN_SCREEN_PROTOCOL.md",
        "docs/experiments/PHYSICS_FLOW_PARENT_LINEAGE.md",
        "docs/experiments/PHYSICS_FLOW_STAGE1_LAUNCH_RUNBOOK.md",
        "tools/physics_flow_stage1.py",
        "tools/physics_flow_stage1_evaluate.py",
        "tools/physics_flow_parent_vpm.py",
        "tools/physics_flow_parent_parity.py",
        "tools/physics_flow_parent_reference.py",
        "tools/physics_flow_lpips.py",
        "tools/snapshot_model_state_receipt.py",
        "tools/physics_flow_stage1_workflow.py",
        "tools/slurm/physics_flow_stage1.sbatch",
        "projects/latent_action_models/physics_flow_train.py",
        PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
        "projects/latent_action_models/lam/physics_flow_model.py",
        "projects/latent_action_models/configs/experiments_0908/physics_flow_common.yaml",
        "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/physics_flow_off.yaml",
        "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/physics_flow_on.yaml",
        "projects/latent_action_models/configs/models/physics_flow_model.yaml",
        "robot_wm/datasets/abc/physics_flow_dataset.py",
        "robot_wm/modeling/dual_diffusion/physics_flow.py",
        *PARENT_TRANSITIVE_SOURCE_FILES,
        "robot_wm/utils/physics_flow_trainer.py",
    )
    records = {}
    for relative in relative_paths:
        path = repo / relative
        if not path.is_file():
            raise PhysicsFlowStage1Error(f"study source file is missing: {relative}")
        records[relative] = file_record(path)
    return records


def _last_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        value = json.loads(lines[-1])
    except (OSError, UnicodeError, IndexError, json.JSONDecodeError) as exc:
        raise PhysicsFlowStage1Error(f"invalid final JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PhysicsFlowStage1Error(f"{label} final JSON must be an object")
    return value


def _validate_parent_lineage(
    args: argparse.Namespace, parent: Mapping[str, Any]
) -> dict[str, Any]:
    legacy = file_record(args.legacy_parent_snapshot)
    failed_log = file_record(args.lineage_failed_log)
    comparison_log = file_record(args.lineage_comparison_log)
    if legacy["sha256"] != LEGACY_REJECTED_SNAPSHOT_SHA256:
        raise PhysicsFlowStage1Error("rejected legacy VPM snapshot differs")
    if (
        failed_log["sha256"] != LINEAGE_FAILED_LOG_SHA256
        or failed_log["bytes"] != 849
        or comparison_log["sha256"] != LINEAGE_COMPARISON_LOG_SHA256
        or comparison_log["bytes"] != 37_385
    ):
        raise PhysicsFlowStage1Error("VPM lineage preflight log differs")
    failed_text = args.lineage_failed_log.read_text(encoding="utf-8")
    if (
        "self.dim() cannot be 0" not in failed_text
        or "snapshot_model_state_receipt.py" not in failed_text
    ):
        raise PhysicsFlowStage1Error("failed VPM lineage preflight cause differs")
    comparison = _last_json_object(args.lineage_comparison_log, "lineage comparison")
    snapshots = comparison.get("snapshots")
    result = comparison.get("comparison")
    if (
        comparison.get("kind")
        != "canonical_checkpoint_model_state_comparison_v1"
        or not isinstance(snapshots, list)
        or len(snapshots) != 2
        or not isinstance(result, Mapping)
        or result.get("model_schema_identical") is not True
        or result.get("model_state_bit_identical") is not False
        or result.get("mismatched_tensor_count") != 495
        or not isinstance(result.get("mismatched_tensor_names"), list)
        or len(result["mismatched_tensor_names"]) != 495
    ):
        raise PhysicsFlowStage1Error("canonical VPM lineage comparison differs")
    by_file_sha = {item.get("file_sha256"): item for item in snapshots}
    old = by_file_sha.get(LEGACY_REJECTED_SNAPSHOT_SHA256, {})
    selected = by_file_sha.get(PARENT_SNAPSHOT_SHA256, {})
    if (
        old.get("path") != legacy["path"]
        or old.get("canonical_model_state_sha256")
        != LEGACY_REJECTED_MODEL_STATE_SHA256
        or old.get("model_schema_sha256") != PARENT_MODEL_SCHEMA_SHA256
        or old.get("model_tensor_count") != 1686
        or old.get("snapshot_metadata", {}).get("run_identity_sha256")
        != LEGACY_REJECTED_RUN_IDENTITY_SHA256
        or selected.get("path") != parent["path"]
        or selected.get("canonical_model_state_sha256")
        != PARENT_CANONICAL_MODEL_STATE_SHA256
        or selected.get("model_schema_sha256") != PARENT_MODEL_SCHEMA_SHA256
        or selected.get("model_tensor_count") != 1686
        or selected.get("snapshot_metadata", {}).get("run_identity_sha256")
        != PARENT_RUN_IDENTITY_SHA256
    ):
        raise PhysicsFlowStage1Error("canonical VPM state receipt differs")
    mismatch_names = result["mismatched_tensor_names"]
    mismatch_families = {
        "wan_attention_lora_a_b": sum(
            name.startswith("forward_model.transformer.blocks")
            and ".lora_" in name
            for name in mismatch_names
        ),
        "action_encoder": sum(name.startswith("action_encoder.") for name in mismatch_names),
        "action_pool": sum(name.startswith("action_pool.") for name in mismatch_names),
        "action_to_control": sum(
            name.startswith("forward_model.action_to_control.")
            for name in mismatch_names
        ),
        "morphology_tokens": sum(name == "morphology_tokens.weight" for name in mismatch_names),
    }
    if mismatch_families != {
        "wan_attention_lora_a_b": 480,
        "action_encoder": 6,
        "action_pool": 4,
        "action_to_control": 4,
        "morphology_tokens": 1,
    }:
        raise PhysicsFlowStage1Error("VPM mismatch family inventory differs")

    ladder_path = args.causal_ladder_registration.resolve(strict=True)
    ladder = read_json(ladder_path, "causal ladder registration")
    vpm = ladder.get("lineage", {}).get("VPM", {})
    if (
        not identity_valid(ladder)
        or ladder.get("identity_sha256") != CAUSAL_LADDER_REGISTRATION_IDENTITY
        or ladder.get("status") != "registered_before_model_weights_or_cache_arrays_open"
        or vpm.get("arm_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or vpm.get("stage_identity_sha256")
        != "17149ca2619afed49d933f03c3613c56d86cb858f5a7fe1402a76e1ff9266d00"
        or vpm.get("outcome_identity_sha256")
        != "249b9e755f4aeefcd24e52169cb9f03a3f53815f860a35912c32098c0c741172"
    ):
        raise PhysicsFlowStage1Error("causal ladder VPM lineage differs")

    direct_root = args.direct_frontier_root.resolve(strict=True)
    direct_files = {
        "registration": direct_root / "registration.json",
        "fit": direct_root / "fit.json",
        "endpoint": direct_root / "endpoint_complete.json",
        "analysis": direct_root / "analysis.json",
        "completion": direct_root / "run_complete.json",
        "audit": direct_root / "audit.json",
    }
    expected_identities = {
        "registration": DIRECT_FRONTIER_REGISTRATION_IDENTITY,
        "fit": DIRECT_FRONTIER_FIT_IDENTITY,
        "endpoint": DIRECT_FRONTIER_ENDPOINT_IDENTITY,
        "analysis": DIRECT_FRONTIER_ANALYSIS_IDENTITY,
        "completion": DIRECT_FRONTIER_COMPLETION_IDENTITY,
        "audit": DIRECT_FRONTIER_AUDIT_IDENTITY,
    }
    direct_payloads = {
        name: read_json(path, f"direct frontier {name}")
        for name, path in direct_files.items()
    }
    if any(
        not identity_valid(direct_payloads[name])
        or direct_payloads[name].get("identity_sha256") != identity
        for name, identity in expected_identities.items()
    ):
        raise PhysicsFlowStage1Error("direct-residual frontier identity differs")
    direct_registration = direct_payloads["registration"]
    direct_vpm = direct_registration.get("lineage", {}).get("VPM", {})
    if (
        direct_registration.get("status")
        != "registered_before_vpm_fit_or_reserve_outcome_open"
        or direct_registration.get("source_commit") != DIRECT_FRONTIER_SOURCE_COMMIT
        or direct_registration.get("parent_registration_identity_sha256")
        != CAUSAL_LADDER_REGISTRATION_IDENTITY
        or direct_registration.get("inputs", {}).get("vpm_snapshot", {}).get("sha256")
        != PARENT_SNAPSHOT_SHA256
        or direct_vpm.get("arm_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or direct_payloads["analysis"].get("decision")
        != "STOP_VPM_DIRECT_RESIDUAL"
        or direct_payloads["audit"].get("decision")
        != "STOP_VPM_DIRECT_RESIDUAL"
    ):
        raise PhysicsFlowStage1Error("direct-residual frontier lineage differs")
    return {
        "selected_parent": {
            **parent,
            "canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
            "model_schema_sha256": PARENT_MODEL_SCHEMA_SHA256,
            "model_tensor_count": 1686,
            "selection_reason": (
                "actual faithful-cascade/direct-residual frontier; used identically "
                "by both Stage-1 arms"
            ),
        },
        "rejected_older_parent": {
            **legacy,
            "run_identity_sha256": LEGACY_REJECTED_RUN_IDENTITY_SHA256,
            "canonical_model_state_sha256": LEGACY_REJECTED_MODEL_STATE_SHA256,
            "model_schema_sha256": PARENT_MODEL_SCHEMA_SHA256,
            "rejection_reason": "495 model tensors differ from actual frontier",
        },
        "canonical_comparison": {
            "failed_preflight_job_id": 507379,
            "failed_preflight_log": failed_log,
            "successful_preflight_job_id": 507381,
            "successful_comparison_log": comparison_log,
            "model_schema_identical": True,
            "model_state_bit_identical": False,
            "mismatched_tensor_count": 495,
            "mismatch_families": mismatch_families,
        },
        "causal_ladder_registration": file_record(ladder_path),
        "causal_ladder_registration_identity_sha256": (
            CAUSAL_LADDER_REGISTRATION_IDENTITY
        ),
        "direct_residual_frontier": {
            "root": str(direct_root),
            "decision": "STOP_VPM_DIRECT_RESIDUAL",
            "source_commit": DIRECT_FRONTIER_SOURCE_COMMIT,
            "files": {
                name: file_record(path) for name, path in direct_files.items()
            },
            "identities": expected_identities,
        },
        "direct_residual_weights_or_outcomes_imported": False,
        "protected_test_accessed": False,
    }


def command_register_study(args: argparse.Namespace) -> int:
    output = fresh_lustre_root(args.output)
    source = clean_repository(args.source_repo, args.expected_commit, "study source")
    train_cache = _validated_cache_receipt(args.train_flow_metadata, "train")
    val_cache = _validated_cache_receipt(args.val_flow_metadata, "val")
    train_metadata = load_cache_metadata(args.train_flow_metadata, split="train")
    val_metadata = load_cache_metadata(args.val_flow_metadata, split="val")
    if (
        train_metadata["source_commit"] != args.expected_commit
        or val_metadata["source_commit"] != args.expected_commit
        or train_metadata["cache_registration_identity_sha256"]
        != val_metadata["cache_registration_identity_sha256"]
    ):
        raise PhysicsFlowStage1Error("flow caches and study source are not co-frozen")
    cache_registration = read_json(
        Path(train_metadata["cache_registration"]["path"]), "cache registration"
    )
    if cache_registration["source_repository"] != source:
        raise PhysicsFlowStage1Error("study source differs from cache source")
    parent = file_record(args.parent_snapshot)
    if parent["sha256"] != PARENT_SNAPSHOT_SHA256:
        raise PhysicsFlowStage1Error("VPM parent snapshot differs")
    parent_resolved_config = file_record(args.parent_resolved_config)
    if (
        parent_resolved_config["sha256"] != PARENT_RESOLVED_CONFIG_SHA256
        or Path(parent_resolved_config["path"]).name
        != "resolved_update_1000.yaml"
        or Path(parent_resolved_config["path"]).parent
        != Path(parent["path"]).parent
    ):
        raise PhysicsFlowStage1Error("VPM parent resolved config differs")
    try:
        import torch

        snapshot = torch.load(
            args.parent_snapshot,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise PhysicsFlowStage1Error("unable to validate parent snapshot") from exc
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != 8
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 1000
        or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise PhysicsFlowStage1Error("faithful-cascade VPM parent metadata differs")
    del snapshot
    parent_lineage = _validate_parent_lineage(args, parent)
    runtime = _registered_runtime(args, Path(source["path"]))
    source_files = _study_source_files(Path(source["path"]))
    historical_source = clean_repository(
        args.parent_source_repo,
        PARENT_TRAINING_SOURCE_COMMIT,
        "historical native-parent source",
    )
    if historical_source["path"] == source["path"]:
        raise PhysicsFlowStage1Error(
            "historical parent source requires an isolated worktree"
        )
    native_sampler_source = source_files[PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE]
    if (
        native_sampler_source["sha256"]
        != PARENT_NATIVE_SAMPLER_SOURCE_SHA256
        or git(
            Path(source["path"]),
            "rev-parse",
            f"{PARENT_TRAINING_SOURCE_COMMIT}:"
            f"{PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE}",
        )
        != PARENT_NATIVE_SAMPLER_GIT_BLOB
        or git(
            Path(source["path"]),
            "hash-object",
            PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
        )
        != PARENT_NATIVE_SAMPLER_GIT_BLOB
        or git(
            Path(source["path"]),
            "diff",
            "--exit-code",
            PARENT_TRAINING_SOURCE_COMMIT,
            "--",
            PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
        )
    ):
        raise PhysicsFlowStage1Error(
            "native parent sampler source differs from its training commit"
        )
    historical_native_sampler = file_record(
        Path(historical_source["path"]) / PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE
    )
    if historical_native_sampler["sha256"] != PARENT_NATIVE_SAMPLER_SOURCE_SHA256:
        raise PhysicsFlowStage1Error("historical native parent sampler changed")
    changed_runtime_files = set(
        filter(
            None,
            git(
                Path(source["path"]),
                "diff",
                "--name-only",
                PARENT_TRAINING_SOURCE_COMMIT,
                "--",
                PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
                *PARENT_TRANSITIVE_SOURCE_FILES,
            ).splitlines(),
        )
    )
    if changed_runtime_files != set(PARENT_TRANSITIVE_SOURCE_FILES):
        raise PhysicsFlowStage1Error(
            "native parent transitive runtime delta inventory differs"
        )
    historical_transitive = {}
    current_transitive = {}
    for relative in PARENT_TRANSITIVE_SOURCE_FILES:
        historical_record = file_record(Path(historical_source["path"]) / relative)
        current_record = source_files[relative]
        historical_blob = git(
            Path(historical_source["path"]), "hash-object", relative
        )
        current_blob = git(Path(source["path"]), "hash-object", relative)
        if (
            historical_record["sha256"]
            != PARENT_HISTORICAL_TRANSITIVE_SHA256[relative]
            or historical_blob != PARENT_HISTORICAL_TRANSITIVE_BLOBS[relative]
            or current_record["sha256"]
            != PARENT_CURRENT_TRANSITIVE_SHA256[relative]
            or current_blob != PARENT_CURRENT_TRANSITIVE_BLOBS[relative]
        ):
            raise PhysicsFlowStage1Error(
                f"native parent transitive source differs: {relative}"
            )
        historical_transitive[relative] = {
            "file": historical_record,
            "git_blob": historical_blob,
        }
        current_transitive[relative] = {
            "file": current_record,
            "git_blob": current_blob,
        }
    if b"preserve_zero_support" in Path(
        parent_resolved_config["path"]
    ).read_bytes():
        raise PhysicsFlowStage1Error(
            "historical parent config unexpectedly names preserve_zero_support"
        )
    transitive_runtime_delta = {
        "changed_files": sorted(changed_runtime_files),
        "historical": historical_transitive,
        "current": current_transitive,
        "only_added_runtime_option": "preserve_zero_support",
        "historical_config_key_present": False,
        "current_default_when_key_absent": False,
        "current_runtime_false_required": True,
        "behavioral_equivalence_requires_isolated_bitwise_output_parity": True,
    }
    output.mkdir(mode=0o700)
    arm_identities = {
        arm.code: hashlib.sha256(
            canonical_json(
                {
                    "study_source_commit": args.expected_commit,
                    "arm": asdict(arm),
                    "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                    "train_flow_metadata_identity_sha256": train_metadata[
                        "identity_sha256"
                    ],
                    "seed": 1234,
                    "updates": 200,
                }
            )
        ).hexdigest()
        for arm in ARMS
    }
    registration = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": STUDY_REGISTRATION_KIND,
            "status": "registered_before_training_or_generated_video_outcomes",
            "created_at_utc": now(),
            "output_root": str(output),
            "source_repository": source,
            "source_files": source_files,
            "cache_registration_identity_sha256": train_metadata[
                "cache_registration_identity_sha256"
            ],
            "renderer_prerequisite": cache_registration["renderer_prerequisite"],
            "flow_caches": {"train": train_cache, "val": val_cache},
            "parent": {
                "snapshot": parent,
                "resolved_config": parent_resolved_config,
                "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "canonical_model_state_sha256": (
                    PARENT_CANONICAL_MODEL_STATE_SHA256
                ),
                "model_schema_sha256": PARENT_MODEL_SCHEMA_SHA256,
                "completed_updates": 1000,
                "ema": False,
                "sole_parent_and_control": True,
                "native_evaluation_endpoint": PARENT_EVALUATION_MODEL_CODE,
                "native_sampler_training_source_commit": (
                    PARENT_TRAINING_SOURCE_COMMIT
                ),
                "native_sampler_source": native_sampler_source,
                "historical_source_repository": historical_source,
                "historical_native_sampler_source": historical_native_sampler,
                "native_sampler_git_blob": PARENT_NATIVE_SAMPLER_GIT_BLOB,
                "native_sampler_source_bit_identical_to_training_commit": True,
                "transitive_runtime_source_delta": transitive_runtime_delta,
                "direct_residual_weights_or_outcomes_imported": False,
            },
            "parent_performance_lineage": parent_lineage,
            "runtime": runtime,
            "arms": [asdict(arm) for arm in ARMS],
            "arm_run_identity_sha256": arm_identities,
            "training_contract": {
                "seed": 1234,
                "world_size": 8,
                "batch_size_per_rank": 1,
                "global_batch_size": 8,
                "updates": 200,
                "optimizer": "fresh identical AdamW lr=1e-4 betas=(0.9,0.95)",
                "scheduler": "20-update linear warmup then cosine to 1e-6 at 200",
                "gradient_accumulation_steps": 1,
                "same_clip_order_actions_noise_timesteps_and_flow": True,
                "trainable_parameter_schema_identical": True,
                "initial_auxiliary_state_identical": True,
                "ema": False,
                "Wan_calls_per_example": 1,
                "flow_model_calls_per_example": 0,
                "flow_clock": 0.0,
                "flow_velocity_loss": 0.0,
                "clean_future_or_teacher_conditioning": False,
            },
            "evaluation_contract": {
                "population": "all prospectively registered D405 rows in immutable val64",
                "prospective_scope": (
                    "Stage-1 branch endpoints, gate, and analysis frozen before this "
                    "branch's training or generated-video outcomes"
                ),
                "val64_globally_new_or_untouched_claim": False,
                "val64_may_have_prior_use_elsewhere_in_research_program": True,
                "d405_count": val_cache["d405_count"],
                "noise_seed_ids": list(NOISE_SEEDS),
                "nfe_grid": list(NFE_GRID),
                "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
                "same_noise_across_all_endpoints": True,
                "equal_Wan_calls_within_each_nfe_contrast": True,
                "complete_noise_by_endpoint_materialization_before_future_rgb_open": True,
                "untouched_parent_evaluation_only": True,
                "native_parent_parity_required_before_validation_open": True,
                "native_parent_parity": {
                    "kind": PARENT_PARITY_KIND,
                    "output": str(output / PARENT_PARITY_FILENAME),
                    "input_split": "train",
                    "clip_index": PARENT_PARITY_TRAIN_INDEX,
                    "rgb_indexed_frames_half_open": [0, 5],
                    "sample_id": PARENT_PARITY_SAMPLE_ID,
                    "condition_source": "off",
                    "nfe_grid": list(NFE_GRID),
                    "public_reference_method": "sample_future_deployable",
                    "adapter_method": "materialize_native_parent_endpoint",
                    "historical_reference_output": str(
                        output / PARENT_HISTORICAL_REFERENCE_FILENAME
                    ),
                    "historical_reference_process_isolated": True,
                    "historical_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
                    "historical_vs_current_bitwise_parity_required": True,
                    "bitwise_parity_required": True,
                    "future_rgb_bytes_opened": False,
                    "validation_dataset_opened": False,
                },
                "top_view_pixel_columns": [0, 320],
                "top_view_latent_columns": [0, 40],
                "metrics": [
                    "top_decoded_mse",
                    "top_temporal_mse",
                    "top_lpips_alex",
                    "all_decoded_mse",
                    "all_temporal_mse",
                    "future_video_latent_nmse",
                    "continuation_latency_components",
                    "native_parent_end_to_end_latency_only",
                ],
                "bootstrap": {
                    "episode_clustered": True,
                    "replicates": BOOTSTRAP_SAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "noise_seeds_remain_with_episode_cluster": True,
                },
            },
            "primary_gate": {
                "nfe": PRIMARY_NFE,
                "raw_vs_unmodified_parent_vpm": {
                    "top_decoded_and_temporal_min_point_improvement_percent": (
                        PRIMARY_MIN_POINT_PERCENT
                    ),
                    "top_decoded_and_temporal_lower_bound_percent_strictly_above": (
                        PRIMARY_MIN_CI_PERCENT
                    ),
                    "top_lpips_lower_bound_percent_strictly_positive": True,
                    "all_view_and_latent_point_nonnegative": True,
                    "all_view_and_latent_lower_bound_percent_above": (
                        GUARDRAIL_CI_PERCENT
                    ),
                },
                "raw_vs_matched_flow_off": {
                    "top_decoded_and_temporal_min_point_improvement_percent": (
                        PRIMARY_MIN_POINT_PERCENT
                    ),
                    "top_decoded_and_temporal_lower_bound_percent_strictly_above": (
                        PRIMARY_MIN_CI_PERCENT
                    ),
                    "top_lpips_lower_bound_percent_strictly_positive": True,
                },
                "same_raw_checkpoint_attribution_controls": {
                    "references": [
                        "off",
                        "episode_shuffled",
                        "timeshift_plus_one",
                        "hold_current",
                    ],
                    "top_decoded_and_temporal_min_point_improvement_percent": (
                        CONTROL_MIN_POINT_PERCENT
                    ),
                    "paired_lower_bound_percent_strictly_positive": True,
                },
                "guardrails": {
                    "all_view_decoded_and_temporal_point_nonnegative": True,
                    "all_view_lower_bound_percent_above": GUARDRAIL_CI_PERCENT,
                    "latent_nmse_point_nonnegative": True,
                    "latent_nmse_lower_bound_percent_above": GUARDRAIL_CI_PERCENT,
                },
                "causal_and_compute": {
                    "exactly_one_Wan_call_at_primary": True,
                    "flow_model_calls": 0,
                    "future_rgb_sampler_input": False,
                    "future_measured_state_cache_or_sampler_input": False,
                    "paired_200_update_trace_required": True,
                    "isolated_historical_parent_bitwise_parity_required": True,
                    "protected_test_accessed": False,
                },
                "all_conditions_required": True,
                "pass_decision": "ADVANCE_RAW_FLOW_SCAFFOLD",
                "failure_decision": "STOP_FIXED_RAW_FLOW",
            },
            "claim_boundary": (
                "fixed causal raw-geometry conditioning feasibility only; not dual "
                "diffusion, FVD, DAgger, or a paper-level quality claim"
            ),
            "causal_structure_hypothesis": (
                "RAW-FLOW must add action-aligned geometry beyond ordinary direct-"
                "residual continuation capacity; no direct-residual weights or outcomes "
                "are imported"
            ),
            "recurrent_or_hybrid_condition_included": False,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(output / "registration.json", registration)
    print(json.dumps(registration, sort_keys=True))
    return 0


def validate_study_registration(path: Path) -> dict[str, Any]:
    registration = read_json(path, "study registration")
    parity_contract = registration.get("evaluation_contract", {}).get(
        "native_parent_parity", {}
    )
    if (
        not identity_valid(registration)
        or registration.get("kind") != STUDY_REGISTRATION_KIND
        or registration.get("status")
        != "registered_before_training_or_generated_video_outcomes"
        or registration.get("parent", {}).get("sole_parent_and_control") is not True
        or registration.get("parent", {}).get("snapshot", {}).get("sha256")
        != PARENT_SNAPSHOT_SHA256
        or registration.get("parent", {}).get("resolved_config", {}).get(
            "sha256"
        )
        != PARENT_RESOLVED_CONFIG_SHA256
        or registration.get("parent", {}).get("run_identity_sha256")
        != PARENT_RUN_IDENTITY_SHA256
        or registration.get("parent", {}).get("canonical_model_state_sha256")
        != PARENT_CANONICAL_MODEL_STATE_SHA256
        or registration.get("parent", {}).get("model_schema_sha256")
        != PARENT_MODEL_SCHEMA_SHA256
        or registration.get("parent", {}).get("native_evaluation_endpoint")
        != PARENT_EVALUATION_MODEL_CODE
        or registration.get("parent", {}).get(
            "native_sampler_training_source_commit"
        )
        != PARENT_TRAINING_SOURCE_COMMIT
        or registration.get("parent", {}).get("native_sampler_source", {}).get(
            "sha256"
        )
        != PARENT_NATIVE_SAMPLER_SOURCE_SHA256
        or registration.get("parent", {}).get("native_sampler_git_blob")
        != PARENT_NATIVE_SAMPLER_GIT_BLOB
        or registration.get("parent", {}).get(
            "native_sampler_source_bit_identical_to_training_commit"
        )
        is not True
        or registration.get("parent", {})
        .get("historical_source_repository", {})
        .get("git_commit")
        != PARENT_TRAINING_SOURCE_COMMIT
        or registration.get("parent", {})
        .get("historical_native_sampler_source", {})
        .get("sha256")
        != PARENT_NATIVE_SAMPLER_SOURCE_SHA256
        or registration.get("parent", {})
        .get("transitive_runtime_source_delta", {})
        .get("changed_files")
        != sorted(PARENT_TRANSITIVE_SOURCE_FILES)
        or registration.get("parent", {})
        .get("transitive_runtime_source_delta", {})
        .get("only_added_runtime_option")
        != "preserve_zero_support"
        or registration.get("parent", {})
        .get("transitive_runtime_source_delta", {})
        .get("historical_config_key_present")
        is not False
        or registration.get("parent", {})
        .get("transitive_runtime_source_delta", {})
        .get("current_default_when_key_absent")
        is not False
        or registration.get("parent", {})
        .get("transitive_runtime_source_delta", {})
        .get("current_runtime_false_required")
        is not True
        or registration.get("parent", {})
        .get("transitive_runtime_source_delta", {})
        .get("behavioral_equivalence_requires_isolated_bitwise_output_parity")
        is not True
        or registration.get("parent", {}).get(
            "direct_residual_weights_or_outcomes_imported"
        )
        is not False
        or registration.get("evaluation_contract", {}).get(
            "complete_noise_by_endpoint_materialization_before_future_rgb_open"
        )
        is not True
        or registration.get("evaluation_contract", {}).get(
            "untouched_parent_evaluation_only"
        )
        is not True
        or registration.get("evaluation_contract", {}).get(
            "native_parent_parity_required_before_validation_open"
        )
        is not True
        or registration.get("evaluation_contract", {}).get("endpoints")
        != [asdict(endpoint) for endpoint in ENDPOINTS]
        or registration.get("evaluation_contract", {}).get(
            "val64_globally_new_or_untouched_claim"
        )
        is not False
        or registration.get("evaluation_contract", {}).get(
            "val64_may_have_prior_use_elsewhere_in_research_program"
        )
        is not True
        or parity_contract.get("kind") != PARENT_PARITY_KIND
        or parity_contract.get("input_split") != "train"
        or parity_contract.get("clip_index") != PARENT_PARITY_TRAIN_INDEX
        or parity_contract.get("rgb_indexed_frames_half_open") != [0, 5]
        or parity_contract.get("sample_id") != PARENT_PARITY_SAMPLE_ID
        or parity_contract.get("condition_source") != "off"
        or parity_contract.get("nfe_grid") != list(NFE_GRID)
        or parity_contract.get("public_reference_method")
        != "sample_future_deployable"
        or parity_contract.get("adapter_method")
        != "materialize_native_parent_endpoint"
        or parity_contract.get("historical_reference_process_isolated")
        is not True
        or parity_contract.get("historical_source_commit")
        != PARENT_TRAINING_SOURCE_COMMIT
        or parity_contract.get(
            "historical_vs_current_bitwise_parity_required"
        )
        is not True
        or parity_contract.get("bitwise_parity_required") is not True
        or parity_contract.get("future_rgb_bytes_opened") is not False
        or parity_contract.get("validation_dataset_opened") is not False
        or not isinstance(
            registration.get("runtime", {}).get("lpips_alex"), Mapping
        )
        or not identity_valid(registration["runtime"]["lpips_alex"])
        or registration["runtime"]["lpips_alex"].get("kind")
        != "raw_physics_flow_offline_lpips_alex_v1"
        or registration["runtime"]["lpips_alex"].get("identity_sha256")
        != LPIPS_RECEIPT_IDENTITY
        or registration["runtime"]["lpips_alex"].get(
            "loaded_state_dict_sha256"
        )
        != LPIPS_STATE_DICT_SHA256
        or registration["runtime"]["lpips_alex"].get(
            "network_access_permitted"
        )
        is not False
        or registration["runtime"]["lpips_alex"].get("download_performed")
        is not False
        or registration.get("recurrent_or_hybrid_condition_included") is not False
        or registration.get("future_rgb_opened") is not False
        or registration.get("future_measured_state_opened") is not False
        or registration.get("generator_outcome_opened") is not False
        or registration.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error("study registration differs")
    output = Path(registration["output_root"]).resolve(strict=True)
    if path.resolve(strict=True) != output / "registration.json":
        raise PhysicsFlowStage1Error("study registration path is noncanonical")
    if Path(str(parity_contract.get("output", ""))).absolute() != (
        output / PARENT_PARITY_FILENAME
    ):
        raise PhysicsFlowStage1Error("native parent parity output differs")
    if Path(
        str(parity_contract.get("historical_reference_output", ""))
    ).absolute() != (output / PARENT_HISTORICAL_REFERENCE_FILENAME):
        raise PhysicsFlowStage1Error(
            "historical parent reference output differs"
        )
    source = registration["source_repository"]
    if clean_repository(
        Path(source["path"]), source["git_commit"], "study source"
    ) != source:
        raise PhysicsFlowStage1Error("study source changed after registration")
    for relative, record in registration["source_files"].items():
        observed = file_record(Path(source["path"]) / relative)
        if observed != record:
            raise PhysicsFlowStage1Error(f"registered source file changed: {relative}")
    native_sampler_source = registration["parent"]["native_sampler_source"]
    if (
        native_sampler_source
        != registration["source_files"][PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE]
        or git(
            Path(source["path"]),
            "rev-parse",
            f"{PARENT_TRAINING_SOURCE_COMMIT}:"
            f"{PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE}",
        )
        != PARENT_NATIVE_SAMPLER_GIT_BLOB
        or git(
            Path(source["path"]),
            "hash-object",
            PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
        )
        != PARENT_NATIVE_SAMPLER_GIT_BLOB
        or git(
            Path(source["path"]),
            "diff",
            "--exit-code",
            PARENT_TRAINING_SOURCE_COMMIT,
            "--",
            PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
        )
    ):
        raise PhysicsFlowStage1Error(
            "registered native sampler differs from parent training source"
        )
    historical_source = registration["parent"]["historical_source_repository"]
    if clean_repository(
        Path(historical_source["path"]),
        PARENT_TRAINING_SOURCE_COMMIT,
        "historical native-parent source",
    ) != historical_source:
        raise PhysicsFlowStage1Error(
            "historical native-parent source changed after registration"
        )
    if historical_source["path"] == source["path"]:
        raise PhysicsFlowStage1Error(
            "historical native-parent source is not isolated"
        )
    historical_native = file_record(
        Path(historical_source["path"]) / PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE
    )
    if (
        historical_native
        != registration["parent"]["historical_native_sampler_source"]
        or historical_native["sha256"] != PARENT_NATIVE_SAMPLER_SOURCE_SHA256
    ):
        raise PhysicsFlowStage1Error("historical native sampler source differs")
    delta = registration["parent"]["transitive_runtime_source_delta"]
    changed_runtime_files = set(
        filter(
            None,
            git(
                Path(source["path"]),
                "diff",
                "--name-only",
                PARENT_TRAINING_SOURCE_COMMIT,
                "--",
                PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
                *PARENT_TRANSITIVE_SOURCE_FILES,
            ).splitlines(),
        )
    )
    if changed_runtime_files != set(PARENT_TRANSITIVE_SOURCE_FILES):
        raise PhysicsFlowStage1Error(
            "registered parent transitive runtime delta changed"
        )
    for relative in PARENT_TRANSITIVE_SOURCE_FILES:
        historical_record = file_record(Path(historical_source["path"]) / relative)
        current_record = registration["source_files"][relative]
        expected_historical = {
            "file": historical_record,
            "git_blob": PARENT_HISTORICAL_TRANSITIVE_BLOBS[relative],
        }
        expected_current = {
            "file": current_record,
            "git_blob": PARENT_CURRENT_TRANSITIVE_BLOBS[relative],
        }
        if (
            historical_record["sha256"]
            != PARENT_HISTORICAL_TRANSITIVE_SHA256[relative]
            or git(Path(historical_source["path"]), "hash-object", relative)
            != PARENT_HISTORICAL_TRANSITIVE_BLOBS[relative]
            or current_record["sha256"]
            != PARENT_CURRENT_TRANSITIVE_SHA256[relative]
            or git(Path(source["path"]), "hash-object", relative)
            != PARENT_CURRENT_TRANSITIVE_BLOBS[relative]
            or delta.get("historical", {}).get(relative)
            != expected_historical
            or delta.get("current", {}).get(relative) != expected_current
        ):
            raise PhysicsFlowStage1Error(
                f"registered parent transitive source differs: {relative}"
            )
    if b"preserve_zero_support" in Path(
        registration["parent"]["resolved_config"]["path"]
    ).read_bytes():
        raise PhysicsFlowStage1Error(
            "registered parent config preserve-zero default changed"
        )
    for split in ("train", "val"):
        receipt = registration["flow_caches"][split]
        metadata = Path(receipt["metadata"]["path"])
        observed = _validated_cache_receipt(metadata, split)
        if observed != receipt:
            raise PhysicsFlowStage1Error(f"{split} flow cache changed after registration")
    if file_record(
        Path(registration["runtime"]["lpips_preflight_log"]["path"])
    ) != registration["runtime"]["lpips_preflight_log"]:
        raise PhysicsFlowStage1Error("offline LPIPS preflight log changed")
    if file_record(Path(registration["parent"]["snapshot"]["path"])) != registration[
        "parent"
    ]["snapshot"]:
        raise PhysicsFlowStage1Error("parent snapshot changed after registration")
    if file_record(
        Path(registration["parent"]["resolved_config"]["path"])
    ) != registration["parent"]["resolved_config"]:
        raise PhysicsFlowStage1Error(
            "parent resolved config changed after registration"
        )
    lineage = registration.get("parent_performance_lineage", {})
    if (
        lineage.get("selected_parent", {}).get("canonical_model_state_sha256")
        != PARENT_CANONICAL_MODEL_STATE_SHA256
        or lineage.get("rejected_older_parent", {}).get(
            "canonical_model_state_sha256"
        )
        != LEGACY_REJECTED_MODEL_STATE_SHA256
        or lineage.get("canonical_comparison", {}).get("mismatched_tensor_count")
        != 495
        or lineage.get("canonical_comparison", {}).get(
            "model_state_bit_identical"
        )
        is not False
        or lineage.get("causal_ladder_registration_identity_sha256")
        != CAUSAL_LADDER_REGISTRATION_IDENTITY
        or lineage.get("direct_residual_frontier", {}).get("decision")
        != "STOP_VPM_DIRECT_RESIDUAL"
        or lineage.get("direct_residual_weights_or_outcomes_imported") is not False
    ):
        raise PhysicsFlowStage1Error("parent performance lineage differs")
    lineage_records = [
        lineage["rejected_older_parent"],
        lineage["canonical_comparison"]["failed_preflight_log"],
        lineage["canonical_comparison"]["successful_comparison_log"],
        lineage["causal_ladder_registration"],
        *lineage["direct_residual_frontier"]["files"].values(),
    ]
    for record in lineage_records:
        if file_record(Path(record["path"])) != record:
            raise PhysicsFlowStage1Error("parent lineage artifact changed")
    return registration


def load_parent_sampler_parity(
    registration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the pre-validation target-blind native parent parity receipt."""

    root = Path(registration["output_root"])
    path = root / PARENT_PARITY_FILENAME
    parity = read_json(path, "native parent sampler parity")
    contract = registration["evaluation_contract"]["native_parent_parity"]
    comparisons = parity.get("comparisons")
    if (
        path.resolve(strict=True) != Path(contract["output"]).resolve(strict=True)
        or not identity_valid(parity)
        or parity.get("kind") != PARENT_PARITY_KIND
        or parity.get("status")
        != "bitwise_native_parent_sampler_parity_passed"
        or parity.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or parity.get("public_reference_method")
        != contract["public_reference_method"]
        or parity.get("evaluation_adapter_method") != contract["adapter_method"]
        or parity.get("condition_source") != "off"
        or parity.get("schedule_mode") != "aligned"
        or parity.get("nfe_grid") != list(NFE_GRID)
        or parity.get("bitwise_parity_required") is not True
        or parity.get("all_bitwise") is not True
        or parity.get("historical_reference_process_isolated") is not True
        or parity.get("historical_vs_current_bitwise_parity_required") is not True
        or parity.get("historical_vs_current_all_bitwise") is not True
        or parity.get("temporary_target_blind_bundle_removed") is not True
        or parity.get("validation_dataset_opened") is not False
        or parity.get("future_rgb_bytes_opened") is not False
        or parity.get("future_measured_state_opened") is not False
        or parity.get("online_teacher_or_feature_calls") != 0
        or parity.get("protected_test_accessed") is not False
        or not isinstance(comparisons, list)
        or len(comparisons) != len(NFE_GRID)
    ):
        raise PhysicsFlowStage1Error("native parent sampler parity differs")
    parent = parity.get("parent")
    if (
        not isinstance(parent, Mapping)
        or parent.get("snapshot") != registration["parent"]["snapshot"]
        or parent.get("resolved_config")
        != registration["parent"]["resolved_config"]
        or parent.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or parent.get("canonical_model_state_sha256")
        != PARENT_CANONICAL_MODEL_STATE_SHA256
        or parent.get("model_schema_sha256") != PARENT_MODEL_SCHEMA_SHA256
        or parent.get("strict_state_load") is not True
        or parent.get("native_public_sampler")
        != "sample_future_deployable"
        or parent.get("native_sampler_training_source_commit")
        != PARENT_TRAINING_SOURCE_COMMIT
        or parent.get("native_sampler_source")
        != registration["parent"]["native_sampler_source"]
        or parent.get("native_sampler_git_blob")
        != PARENT_NATIVE_SAMPLER_GIT_BLOB
        or parent.get(
            "native_sampler_source_bit_identical_to_training_commit"
        )
        is not True
        or parent.get("transitive_runtime_source_delta")
        != registration["parent"]["transitive_runtime_source_delta"]
        or parent.get("historical_config_preserve_zero_support_key_present")
        is not False
        or parent.get("current_runtime_preserve_zero_support") is not False
        or parent.get("isolated_historical_output_parity_required") is not True
        or parent.get("continued_training_updates") != 0
        or parent.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error("native parent parity state binding differs")
    historical_reference = parity.get("historical_reference")
    historical_reference_path = Path(
        registration["output_root"]
    ) / PARENT_HISTORICAL_REFERENCE_FILENAME
    if (
        parity.get("historical_source_repository")
        != registration["parent"]["historical_source_repository"]
        or not isinstance(historical_reference, Mapping)
        or file_record(historical_reference_path) != historical_reference
        or Path(historical_reference["path"]).resolve(strict=True)
        != historical_reference_path.resolve(strict=True)
    ):
        raise PhysicsFlowStage1Error(
            "isolated historical parent reference binding differs"
        )
    try:
        import torch

        historical_payload = torch.load(
            historical_reference_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise PhysicsFlowStage1Error(
            "unable to replay isolated historical parent reference"
        ) from exc
    historical_results = (
        historical_payload.get("results")
        if isinstance(historical_payload, Mapping)
        else None
    )
    if (
        not isinstance(historical_payload, Mapping)
        or historical_payload.get("kind")
        != "raw_physics_flow_historical_parent_reference"
        or historical_payload.get("implementation_commit")
        != PARENT_TRAINING_SOURCE_COMMIT
        or historical_payload.get("native_public_sampler")
        != "sample_future_deployable"
        or historical_payload.get("condition_source") != "off"
        or historical_payload.get("schedule_mode") != "aligned"
        or historical_payload.get("nfe_grid") != list(NFE_GRID)
        or historical_payload.get("strict_state_load") is not True
        or historical_payload.get(
            "historical_preserve_zero_support_attribute_absent"
        )
        is not True
        or historical_payload.get("total_wan_calls") != sum(NFE_GRID)
        or historical_payload.get("validation_dataset_opened") is not False
        or historical_payload.get("future_rgb_bytes_opened") is not False
        or historical_payload.get("future_measured_state_opened") is not False
        or historical_payload.get("protected_test_accessed") is not False
        or not isinstance(historical_results, Mapping)
        or set(historical_results) != {str(nfe) for nfe in NFE_GRID}
    ):
        raise PhysicsFlowStage1Error(
            "isolated historical parent reference payload differs"
        )
    input_evidence = parity.get("input")
    train_metadata = load_cache_metadata(
        Path(registration["flow_caches"]["train"]["metadata"]["path"]),
        split="train",
    )
    cache_registration = read_json(
        Path(train_metadata["cache_registration"]["path"]),
        "cache registration",
    )
    immutable = cache_registration["inputs"]["train"]["cache"]["arrays"]
    descriptor = cache_registration["splits"]["train"]["descriptors"][
        PARENT_PARITY_TRAIN_INDEX
    ]
    if (
        not isinstance(input_evidence, Mapping)
        or input_evidence.get("split") != "train"
        or input_evidence.get("clip_index") != PARENT_PARITY_TRAIN_INDEX
        or input_evidence.get("clip_id") != descriptor["clip_id"]
        or input_evidence.get("rgb_indexed_frames_half_open") != [0, 5]
        or input_evidence.get("rgb_future_frames_indexed") is not False
        or input_evidence.get("actions_indexed_frames_half_open") != [0, 13]
        or input_evidence.get("sample_id") != PARENT_PARITY_SAMPLE_ID
        or input_evidence.get("immutable_rgb_array") != immutable["rgb"]
        or input_evidence.get("immutable_actions_array") != immutable["actions"]
        or input_evidence.get("future_rgb_bytes_opened") is not False
        or input_evidence.get("future_measured_state_opened") is not False
        or input_evidence.get("protected_test_accessed") is not False
        or any(
            SHA256_RE.fullmatch(str(input_evidence.get(name, ""))) is None
            for name in (
                "history_rgb_sha256",
                "actions_sha256",
                "morphology_index_sha256",
            )
        )
    ):
        raise PhysicsFlowStage1Error("native parent parity input differs")
    by_nfe = {row.get("nfe"): row for row in comparisons if isinstance(row, Mapping)}
    if set(by_nfe) != set(NFE_GRID):
        raise PhysicsFlowStage1Error("native parent parity NFE population differs")
    for nfe, row in by_nfe.items():
        historical_result = historical_results[str(nfe)]
        if (
            row.get("direct_native_wan_calls") != nfe
            or row.get("adapter_wan_calls") != nfe
            or row.get("latent_bitwise_equal") is not True
            or row.get("decoded_uint8_bitwise_equal") is not True
            or row.get("initial_video_noise_fp16_bitwise_equal") is not True
            or row.get(
                "adapter_full_precision_initial_noise_bitwise_equal"
            )
            is not True
            or row.get(
                "historical_6560866_vs_current_latent_bitwise_equal"
            )
            is not True
            or row.get(
                "historical_6560866_vs_current_decoded_bitwise_equal"
            )
            is not True
            or row.get(
                "historical_6560866_vs_current_initial_noise_bitwise_equal"
            )
            is not True
            or row.get("direct_latent_sha256")
            != row.get("adapter_latent_sha256")
            or row.get("historical_latent_sha256")
            != row.get("adapter_latent_sha256")
            or row.get("direct_decoded_sha256")
            != row.get("adapter_decoded_sha256")
            or row.get("historical_decoded_sha256")
            != row.get("adapter_decoded_sha256")
            or not isinstance(historical_result, Mapping)
            or historical_result.get("wan_calls") != nfe
            or historical_result.get("online_teacher_or_feature_calls") != 0
            or historical_result.get("future_rgb_sampler_input") is not False
            or historical_result.get("clean_video_latent_sampler_input")
            is not False
            or torch_tensor_sha256(historical_result.get("video_latent"))
            != row.get("historical_latent_sha256")
            or torch_tensor_sha256(historical_result.get("decoded_uint8"))
            != row.get("historical_decoded_sha256")
            or torch_tensor_sha256(
                historical_result.get("video_initial_state_full_precision")
            )
            != row.get("explicit_video_noise_sha256")
            or any(
                SHA256_RE.fullmatch(str(row.get(name, ""))) is None
                for name in (
                    "explicit_video_noise_sha256",
                    "direct_latent_sha256",
                    "direct_decoded_sha256",
                )
            )
        ):
            raise PhysicsFlowStage1Error(
                f"native parent parity comparison differs at NFE {nfe}"
            )
    return parity, file_record(path)


def _training_trace(
    registration: Mapping[str, Any], arm: Arm
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    run_dir = Path(registration["output_root"]) / "training" / arm.run_name
    trace_path = run_dir / "physics_flow_training_trace.jsonl"
    complete_path = run_dir / "physics_flow_training_trace_complete.json"
    snapshot_path = run_dir / "snapshot.pt"
    if any(not path.is_file() or path.is_symlink() for path in (trace_path, complete_path, snapshot_path)):
        raise PhysicsFlowStage1Error(f"{arm.code} training inventory is incomplete")
    rows = read_jsonl(trace_path)
    if not rows:
        raise PhysicsFlowStage1Error(f"{arm.code} training trace is empty")
    header = rows[0]
    if (
        header.get("kind") != "physics_flow_training_trace_header"
        or header.get("arm") != arm.code
        or header.get("fuse_flow") is not arm.fuse_flow
        or header.get("parent_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or header.get("parent_run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or header.get("continuation_updates") != 200
        or header.get("wan_calls_per_example") != 1
        or header.get("flow_model_calls_per_example") != 0
        or header.get("future_measured_state_conditioning") is not False
        or header.get("future_rgb_conditioning") is not False
        or header.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error(f"{arm.code} training trace header differs")
    events = {}
    for row in rows[1:]:
        if row.get("kind") != "physics_flow_training_trace_event" or row.get("arm") != arm.code:
            raise PhysicsFlowStage1Error(f"{arm.code} training trace has an unknown row")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PhysicsFlowStage1Error("training metrics row is invalid")
        if "train_loss/loss" not in metrics:
            # Visualization telemetry is allowed but cannot serve as a paired
            # training update.
            continue
        iteration = metrics.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration in events:
            raise PhysicsFlowStage1Error("training iteration identity differs")
        events[iteration] = row
    if set(events) != set(range(200)):
        raise PhysicsFlowStage1Error(f"{arm.code} lacks exactly 200 training updates")
    completion = read_json(complete_path, f"{arm.code} trace completion")
    if (
        completion.get("kind") != "physics_flow_training_trace_complete"
        or completion.get("arm") != arm.code
        or completion.get("completed_updates") != 200
        or completion.get("trace_sha256") != sha256_file(trace_path)
        or completion.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error(f"{arm.code} training completion differs")
    try:
        import torch

        snapshot = torch.load(
            snapshot_path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as exc:
        raise PhysicsFlowStage1Error(f"unable to inspect {arm.code} snapshot") from exc
    expected_identity = registration["arm_run_identity_sha256"][arm.code]
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != 8
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 200
        or snapshot.get("run_identity_sha256") != expected_identity
        or snapshot.get("physics_flow_arm") != arm.code
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise PhysicsFlowStage1Error(f"{arm.code} snapshot contract differs")
    del snapshot
    artifacts = {
        "run_dir": str(run_dir.resolve(strict=True)),
        "trace": file_record(trace_path),
        "completion": file_record(complete_path),
        "snapshot": file_record(snapshot_path),
        "resolved_config": file_record(run_dir / ".hydra" / "config.yaml"),
        "protected_test_accessed": False,
    }
    return header, events, artifacts


def compare_training_traces(
    registration: Mapping[str, Any], *, write: bool
) -> dict[str, Any]:
    off_header, off_events, off_artifacts = _training_trace(
        registration, ARM_BY_CODE["FLOW-OFF"]
    )
    raw_header, raw_events, raw_artifacts = _training_trace(
        registration, ARM_BY_CODE["RAW-FLOW"]
    )
    same_header_fields = (
        "parent_snapshot_sha256",
        "parent_run_identity_sha256",
        "train_flow_metadata_sha256",
        "val_flow_metadata_sha256",
        "parameter_schema_sha256",
        "initial_auxiliary_state_sha256",
        "continuation_updates",
        "wan_calls_per_example",
        "flow_model_calls_per_example",
        "flow_clock",
        "flow_velocity_loss",
        "future_measured_state_conditioning",
        "future_rgb_conditioning",
        "optimizer_state_policy",
    )
    if any(off_header.get(field) != raw_header.get(field) for field in same_header_fields):
        raise PhysicsFlowStage1Error("paired training header identity differs")
    evidence = (
        "actions_sha256",
        "clip_index_sha256",
        "flow_sha256",
        "timesteps_sha256",
        "video_noise_sha256",
    )
    compared = 0
    metric_names = []
    for iteration in range(200):
        left = off_events[iteration]
        right = raw_events[iteration]
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        if (
            left.get("total_observations") != (iteration + 1) * 8
            or right.get("total_observations") != (iteration + 1) * 8
            or left_metrics.get("iteration") != iteration
            or right_metrics.get("iteration") != iteration
        ):
            raise PhysicsFlowStage1Error(
                f"paired training observation accounting differs at {iteration}"
            )
        for field in evidence:
            key = f"train_loss/paired_audit/exact_{field}_all_ranks_sha256"
            if key not in left_metrics or left_metrics.get(key) != right_metrics.get(key):
                raise PhysicsFlowStage1Error(
                    f"paired exact {field} trace differs at update {iteration}"
                )
            if iteration == 0:
                metric_names.append(key)
            compared += 1
    terminal_seam = {
        arm: {
            name: events[199]["metrics"].get(f"train_loss/{name}")
            for name in (
                "physics_flow/condition_rms",
                "physics_flow/condition_nonzero_fraction",
                "physics_flow/effective_adapter_gate",
            )
        }
        for arm, events in (
            ("FLOW-OFF", off_events),
            ("RAW-FLOW", raw_events),
        )
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for values in terminal_seam.values()
        for value in values.values()
    ):
        raise PhysicsFlowStage1Error("terminal seam diagnostic is invalid")
    receipt = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "raw_physics_flow_training_pairing",
            "status": "paired_trace_passed",
            "registration_identity_sha256": registration["identity_sha256"],
            "paired_updates": 200,
            "world_size": 8,
            "global_batch_size": 8,
            "exact_evidence_fields": list(evidence),
            "exact_all_rank_hash_comparisons": compared,
            "metric_names": metric_names,
            "same_parent_parameter_schema_and_initial_auxiliary_state": True,
            "terminal_nonselectable_seam_diagnostics": terminal_seam,
            "artifacts": {
                "FLOW-OFF": off_artifacts,
                "RAW-FLOW": raw_artifacts,
            },
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "protected_test_accessed": False,
        }
    )
    if write:
        exclusive_json(
            Path(registration["output_root"]) / "training_pairing.json", receipt
        )
    return receipt


def command_compare_traces(args: argparse.Namespace) -> int:
    registration = validate_study_registration(args.registration)
    receipt = compare_training_traces(registration, write=not args.read_only)
    print(json.dumps(receipt, sort_keys=True))
    return 0


EVALUATION_KIND = "raw_physics_flow_stage1_evaluation_row"
EVALUATION_INVENTORY_KIND = "raw_physics_flow_stage1_evaluation_inventory"
LOWER_BETTER_METRICS = (
    "top_decoded_mse_unit_range",
    "top_temporal_mse_unit_range",
    "top_lpips_alex",
    "all_decoded_mse_unit_range",
    "all_temporal_mse_unit_range",
    "future_video_latent_nmse",
)


def validate_evaluation_causal_flags(
    value: Mapping[str, Any], *, label: str
) -> None:
    """Require the complete target-blind materialization boundary on evidence."""

    required = {
        "all_endpoints_materialized_before_future_rgb_open": True,
        "future_rgb_sampler_input": False,
        "future_measured_state_sampler_input": False,
        "clean_video_latent_sampler_input": False,
        "protected_test_accessed": False,
    }
    changed = {
        key: value.get(key)
        for key, expected in required.items()
        if value.get(key) is not expected
    }
    if changed:
        raise PhysicsFlowStage1Error(
            f"{label} violates target-blind materialization flags: {changed}"
        )


def load_evaluation_rows(
    registration: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_parity, parent_parity_record = load_parent_sampler_parity(registration)
    root = Path(registration["output_root"]) / "evaluation"
    inventory_path = root / "inventory.json"
    inventory = read_json(inventory_path, "evaluation inventory")
    validate_evaluation_causal_flags(inventory, label="evaluation inventory")
    d405_count = int(registration["evaluation_contract"]["d405_count"])
    expected_rows = d405_count * len(NOISE_SEEDS) * len(ENDPOINTS)
    if (
        not identity_valid(inventory)
        or inventory.get("kind") != EVALUATION_INVENTORY_KIND
        or inventory.get("status") != "complete"
        or inventory.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or inventory.get("d405_clips") != d405_count
        or inventory.get("noise_seed_ids") != list(NOISE_SEEDS)
        or inventory.get("endpoints") != [asdict(endpoint) for endpoint in ENDPOINTS]
        or inventory.get("rows") != expected_rows
        or inventory.get("lpips_evaluator")
        != registration.get("runtime", {}).get("lpips_alex")
        or inventory.get("lpips_same_on_all_ranks_and_all_arm_endpoints") is not True
        or inventory.get("native_parent_sampler_parity")
        != parent_parity_record
        or inventory.get("native_parent_sampler_parity_identity_sha256")
        != parent_parity["identity_sha256"]
        or inventory.get("historical_parent_reference")
        != parent_parity["historical_reference"]
        or inventory.get("historical_vs_current_parent_all_bitwise") is not True
        or inventory.get("unmodified_de65_parent_endpoint_included") is not True
        or inventory.get(
            "all_endpoints_materialized_before_future_rgb_open"
        )
        is not True
        or inventory.get("future_rgb_sampler_input") is not False
        or inventory.get("future_measured_state_sampler_input") is not False
        or inventory.get("clean_video_latent_sampler_input") is not False
        or inventory.get("protected_test_accessed") is not False
    ):
        raise PhysicsFlowStage1Error("evaluation inventory differs")
    rows = []
    files = inventory.get("rank_files")
    receipts = inventory.get("rank_receipts")
    if (
        not isinstance(files, list)
        or len(files) != 8
        or not isinstance(receipts, list)
        or len(receipts) != 8
    ):
        raise PhysicsFlowStage1Error("evaluation rank inventory differs")
    for rank, record in enumerate(files):
        path = _resolve_record(record, parent=root, label=f"rank {rank} rows")
        if path != (root / f"rank_{rank:02d}.jsonl").resolve(strict=True):
            raise PhysicsFlowStage1Error("evaluation rank path is noncanonical")
        rank_rows = read_jsonl(path)
        rows.extend(rank_rows)
        receipt_path = _resolve_record(
            receipts[rank], parent=root, label=f"rank {rank} receipt"
        )
        receipt = read_json(receipt_path, f"rank {rank} receipt")
        validate_evaluation_causal_flags(
            receipt, label=f"rank {rank} evaluation receipt"
        )
        local_indexes = receipt.get("d405_clip_indexes", [])
        local_batches = math.ceil(len(local_indexes) / 2)
        expected_transformer_calls = {
            "FLOW-OFF": local_batches * len(NOISE_SEEDS) * sum(NFE_GRID),
            "RAW-FLOW": (
                local_batches
                * len(NOISE_SEEDS)
                * len(RUNTIME_SOURCES)
                * sum(NFE_GRID)
            ),
            PARENT_EVALUATION_MODEL_CODE: (
                local_batches * len(NOISE_SEEDS) * sum(NFE_GRID)
            ),
        }
        if (
            receipt_path != (root / f"rank_{rank:02d}.json").resolve(strict=True)
            or not identity_valid(receipt)
            or receipt.get("kind")
            != "raw_physics_flow_stage1_evaluation_rank"
            or receipt.get("rank") != rank
            or receipt.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or receipt.get("rows_file") != record
            or receipt.get("world_size") != 8
            or receipt.get("noise_seed_ids") != list(NOISE_SEEDS)
            or receipt.get("endpoints")
            != [asdict(endpoint) for endpoint in ENDPOINTS]
            or receipt.get("rows") != len(rank_rows)
            or receipt.get("lpips_evaluator_identity_sha256")
            != registration["runtime"]["lpips_alex"]["identity_sha256"]
            or receipt.get("native_parent_sampler_parity")
            != parent_parity_record
            or receipt.get("native_parent_sampler_parity_identity_sha256")
            != parent_parity["identity_sha256"]
            or receipt.get("historical_parent_reference")
            != parent_parity["historical_reference"]
            or receipt.get("historical_vs_current_parent_all_bitwise") is not True
            or receipt.get("transformer_calls_by_evaluation_model")
            != expected_transformer_calls
            or receipt.get("expected_transformer_calls_by_evaluation_model")
            != expected_transformer_calls
            or receipt.get(
                "all_endpoints_materialized_before_future_rgb_open"
            )
            is not True
            or receipt.get("clean_video_latent_sampler_input") is not False
            or receipt.get("protected_test_accessed") is not False
        ):
            raise PhysicsFlowStage1Error(f"rank {rank} evaluation receipt differs")
    expected_keys = {
        (clip_index, seed, endpoint.code)
        for clip_index in inventory["d405_clip_indexes"]
        for seed in NOISE_SEEDS
        for endpoint in ENDPOINTS
    }
    observed = {}
    metric_names = (*LOWER_BETTER_METRICS, "decoded_psnr_db")
    for row in rows:
        validate_evaluation_causal_flags(row, label="evaluation row")
        endpoint_value = row.get("endpoint")
        code = endpoint_value.get("code") if isinstance(endpoint_value, Mapping) else None
        endpoint = ENDPOINT_BY_CODE.get(str(code))
        key = (row.get("clip_index"), row.get("noise_seed_id"), str(code))
        metrics = row.get("metrics")
        seam = row.get("seam_diagnostics")
        latency = row.get("latency_seconds")
        latency_values = (
            "history_encode_seconds",
            "adapter_and_wan_seconds",
            "decode_seconds",
            "end_to_end_seconds",
        )
        if endpoint is not None and endpoint.arm == PARENT_EVALUATION_MODEL_CODE:
            latency_valid = (
                isinstance(latency, Mapping)
                and latency.get("measurement_scope")
                == "native_public_sampler_end_to_end_only"
                and all(latency.get(name) is None for name in latency_values[:3])
                and isinstance(latency.get(latency_values[3]), (int, float))
                and not isinstance(latency.get(latency_values[3]), bool)
                and math.isfinite(float(latency[latency_values[3]]))
                and float(latency[latency_values[3]]) >= 0.0
            )
        else:
            latency_valid = (
                isinstance(latency, Mapping)
                and latency.get("measurement_scope")
                == "component_and_end_to_end"
                and all(
                    isinstance(latency.get(name), (int, float))
                    and not isinstance(latency.get(name), bool)
                    and math.isfinite(float(latency[name]))
                    and float(latency[name]) >= 0.0
                    for name in latency_values
                )
            )
        if (
            not identity_valid(row)
            or row.get("kind") != EVALUATION_KIND
            or row.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or endpoint is None
            or endpoint_value != asdict(endpoint)
            or not latency_valid
            or key not in expected_keys
            or key in observed
            or row.get("actual_transformer_call_count") != endpoint.nfe
            or row.get("flow_model_call_count") != 0
            or row.get("online_teacher_or_feature_calls") != 0
            or row.get("native_parent_sampler")
            is not (endpoint.arm == PARENT_EVALUATION_MODEL_CODE)
            or row.get("native_parent_full_precision_noise_verified")
            is not (endpoint.arm == PARENT_EVALUATION_MODEL_CODE)
            or row.get("native_parent_sampler_parity_identity_sha256")
            != parent_parity["identity_sha256"]
            or not isinstance(row.get("arm_artifacts"), Mapping)
            or (
                endpoint.arm == PARENT_EVALUATION_MODEL_CODE
                and (
                    row["arm_artifacts"].get("snapshot")
                    != registration["parent"]["snapshot"]
                    or row["arm_artifacts"].get("resolved_config")
                    != registration["parent"]["resolved_config"]
                    or row["arm_artifacts"].get("canonical_model_state_sha256")
                    != PARENT_CANONICAL_MODEL_STATE_SHA256
                    or row["arm_artifacts"].get("strict_state_load") is not True
                    or row["arm_artifacts"].get("native_public_sampler")
                    != "sample_future_deployable"
                    or row["arm_artifacts"].get(
                        "native_sampler_training_source_commit"
                    )
                    != PARENT_TRAINING_SOURCE_COMMIT
                    or row["arm_artifacts"].get("native_sampler_source")
                    != registration["parent"]["native_sampler_source"]
                    or row["arm_artifacts"].get(
                        "transitive_runtime_source_delta"
                    )
                    != registration["parent"][
                        "transitive_runtime_source_delta"
                    ]
                    or row["arm_artifacts"].get("native_sampler_git_blob")
                    != PARENT_NATIVE_SAMPLER_GIT_BLOB
                    or row["arm_artifacts"].get(
                        "native_sampler_source_bit_identical_to_training_commit"
                    )
                    is not True
                    or row["arm_artifacts"].get(
                        "historical_config_preserve_zero_support_key_present"
                    )
                    is not False
                    or row["arm_artifacts"].get(
                        "current_runtime_preserve_zero_support"
                    )
                    is not False
                    or row["arm_artifacts"].get(
                        "isolated_historical_output_parity_required"
                    )
                    is not True
                    or row["arm_artifacts"].get("continued_training_updates")
                    != 0
                )
            )
            or row.get("future_rgb_sampler_input") is not False
            or row.get("future_measured_state_sampler_input") is not False
            or row.get("clean_video_latent_sampler_input") is not False
            or row.get("all_endpoints_materialized_before_future_rgb_open") is not True
            or row.get("lpips_evaluator_identity_sha256")
            != registration.get("runtime", {})
            .get("lpips_alex", {})
            .get("identity_sha256")
            or row.get("protected_test_accessed") is not False
            or not isinstance(metrics, Mapping)
            or not isinstance(seam, Mapping)
            or seam.get("nonselectable") is not True
            or seam.get("fusion_enabled")
            is not (
                endpoint.arm == "RAW-FLOW"
                and endpoint.condition_source != "off"
            )
            or any(
                isinstance(seam.get(name), bool)
                or not isinstance(seam.get(name), (int, float))
                or not math.isfinite(float(seam[name]))
                for name in (
                    "effective_adapter_gate",
                    "condition_rms",
                    "condition_nonzero_fraction",
                )
            )
            or not 0.0 <= float(seam["condition_nonzero_fraction"]) <= 1.0
            or any(
                isinstance(metrics.get(name), bool)
                or not isinstance(metrics.get(name), (int, float))
                or not math.isfinite(float(metrics[name]))
                for name in metric_names
            )
        ):
            raise PhysicsFlowStage1Error(f"evaluation row violates contract: {key}")
        observed[key] = row
    if set(observed) != expected_keys:
        raise PhysicsFlowStage1Error("evaluation row population is incomplete")
    pairing_hashes = (
        "history_rgb_sha256",
        "actions_sha256",
        "video_initial_noise_sha256",
        "clean_video_latent_sha256",
        "future_rgb_target_sha256",
    )
    for clip_index in inventory["d405_clip_indexes"]:
        for seed in NOISE_SEEDS:
            endpoint_rows = [observed[(clip_index, seed, endpoint.code)] for endpoint in ENDPOINTS]
            reference = endpoint_rows[0]["tensor_sha256"]
            if any(
                row["tensor_sha256"].get(field) != reference.get(field)
                for row in endpoint_rows[1:]
                for field in pairing_hashes
            ):
                raise PhysicsFlowStage1Error(
                    f"paired inputs/noise/targets differ for clip={clip_index} seed={seed}"
                )
            for nfe in NFE_GRID:
                same_nfe = [row for row in endpoint_rows if row["endpoint"]["nfe"] == nfe]
                if any(row["actual_transformer_call_count"] != nfe for row in same_nfe):
                    raise PhysicsFlowStage1Error("equal-Wan-call contrast differs")
    return rows, inventory


def _paired_cluster_effect(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    candidate = {
        (int(row["clip_index"]), int(row["noise_seed_id"])): float(row["metrics"][metric])
        for row in candidate_rows
    }
    reference = {
        (int(row["clip_index"]), int(row["noise_seed_id"])): float(row["metrics"][metric])
        for row in reference_rows
    }
    if set(candidate) != set(reference):
        raise PhysicsFlowStage1Error("paired effect key population differs")
    clips = sorted({key[0] for key in candidate})
    if any(
        {seed for clip, seed in candidate if clip == clip_index} != set(NOISE_SEEDS)
        for clip_index in clips
    ):
        raise PhysicsFlowStage1Error("episode cluster lacks all frozen noise seeds")
    candidate_cluster = np.asarray(
        [np.mean([candidate[(clip, seed)] for seed in NOISE_SEEDS]) for clip in clips],
        dtype=np.float64,
    )
    reference_cluster = np.asarray(
        [np.mean([reference[(clip, seed)] for seed in NOISE_SEEDS]) for clip in clips],
        dtype=np.float64,
    )
    if np.any(reference_cluster <= 0):
        raise PhysicsFlowStage1Error("effect reference metric must be positive")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clips), size=(BOOTSTRAP_SAMPLES, len(clips)))
    candidate_boot = candidate_cluster[draws].mean(axis=1)
    reference_boot = reference_cluster[draws].mean(axis=1)
    improvement_boot = 100.0 * (reference_boot - candidate_boot) / reference_boot
    candidate_mean = float(candidate_cluster.mean())
    reference_mean = float(reference_cluster.mean())
    improvement = 100.0 * (reference_mean - candidate_mean) / reference_mean
    return {
        "metric": metric,
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "relative_improvement_percent": improvement,
        "paired_episode_cluster_bootstrap_95_ci_percent": [
            float(np.quantile(improvement_boot, 0.025)),
            float(np.quantile(improvement_boot, 0.975)),
        ],
        "paired_episode_clusters": len(clips),
        "noise_seeds_per_cluster": len(NOISE_SEEDS),
        "bootstrap_replicates": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _endpoint_rows(
    rows: Sequence[Mapping[str, Any]], code: str
) -> list[Mapping[str, Any]]:
    result = [row for row in rows if row["endpoint"]["code"] == code]
    if not result:
        raise PhysicsFlowStage1Error(f"evaluation endpoint is absent: {code}")
    return result


def analyze_study(
    registration: Mapping[str, Any], *, write: bool
) -> dict[str, Any]:
    pairing_path = Path(registration["output_root"]) / "training_pairing.json"
    pairing = read_json(pairing_path, "training pairing")
    replay_pairing = compare_training_traces(registration, write=False)
    if (
        not identity_valid(pairing)
        or pairing.get("kind") != "raw_physics_flow_training_pairing"
        or pairing.get("status") != "paired_trace_passed"
    ):
        raise PhysicsFlowStage1Error("paired training receipt differs")
    for key, value in pairing.items():
        if key == "identity_sha256":
            continue
        if replay_pairing.get(key) != value:
            raise PhysicsFlowStage1Error(f"training pairing replay differs at {key}")
    rows, inventory = load_evaluation_rows(registration)
    parent_parity, _parent_parity_record = load_parent_sampler_parity(
        registration
    )
    raw_code = "raw_checkpoint_raw_nfe_1"
    baseline_code = "matched_off_nfe_1"
    parent_code = "parent_vpm_off_nfe_1"
    raw_rows = _endpoint_rows(rows, raw_code)
    effects = {}
    effects["raw_vs_matched_off"] = {
        metric: _paired_cluster_effect(
            raw_rows, _endpoint_rows(rows, baseline_code), metric
        )
        for metric in LOWER_BETTER_METRICS
    }
    effects["raw_vs_unmodified_parent_vpm"] = {
        metric: _paired_cluster_effect(
            raw_rows, _endpoint_rows(rows, parent_code), metric
        )
        for metric in LOWER_BETTER_METRICS
    }
    control_codes = {
        "off": "raw_checkpoint_off_nfe_1",
        "episode_shuffled": "raw_checkpoint_episode_shuffled_nfe_1",
        "timeshift_plus_one": "raw_checkpoint_timeshift_plus_one_nfe_1",
        "hold_current": "raw_checkpoint_hold_current_nfe_1",
    }
    for name, code in control_codes.items():
        effects[f"raw_vs_{name}"] = {
            metric: _paired_cluster_effect(
                raw_rows, _endpoint_rows(rows, code), metric
            )
            for metric in ("top_decoded_mse_unit_range", "top_temporal_mse_unit_range")
        }
    # Secondary dose-response is reported, never allowed to rescue NFE-1.
    secondary = {}
    for nfe in (2, 4):
        secondary[f"nfe_{nfe}_raw_vs_matched_off"] = {
            metric: _paired_cluster_effect(
                _endpoint_rows(rows, f"raw_checkpoint_raw_nfe_{nfe}"),
                _endpoint_rows(rows, f"matched_off_nfe_{nfe}"),
                metric,
            )
            for metric in LOWER_BETTER_METRICS
        }
        secondary[f"nfe_{nfe}_raw_vs_unmodified_parent_vpm"] = {
            metric: _paired_cluster_effect(
                _endpoint_rows(rows, f"raw_checkpoint_raw_nfe_{nfe}"),
                _endpoint_rows(rows, f"parent_vpm_off_nfe_{nfe}"),
                metric,
            )
            for metric in LOWER_BETTER_METRICS
        }
    baseline_effect = effects["raw_vs_matched_off"]
    parent_effect = effects["raw_vs_unmodified_parent_vpm"]
    matched_decoded_primary = all(
        baseline_effect[metric]["relative_improvement_percent"]
        >= PRIMARY_MIN_POINT_PERCENT
        and baseline_effect[metric]["paired_episode_cluster_bootstrap_95_ci_percent"][0]
        > PRIMARY_MIN_CI_PERCENT
        for metric in ("top_decoded_mse_unit_range", "top_temporal_mse_unit_range")
    )
    matched_lpips_primary = (
        baseline_effect["top_lpips_alex"][
            "paired_episode_cluster_bootstrap_95_ci_percent"
        ][0]
        > 0.0
    )
    parent_decoded_primary = all(
        parent_effect[metric]["relative_improvement_percent"]
        >= PRIMARY_MIN_POINT_PERCENT
        and parent_effect[metric][
            "paired_episode_cluster_bootstrap_95_ci_percent"
        ][0]
        > PRIMARY_MIN_CI_PERCENT
        for metric in (
            "top_decoded_mse_unit_range",
            "top_temporal_mse_unit_range",
        )
    )
    parent_lpips_primary = (
        parent_effect["top_lpips_alex"][
            "paired_episode_cluster_bootstrap_95_ci_percent"
        ][0]
        > 0.0
    )
    attribution = {}
    for name in control_codes:
        effect = effects[f"raw_vs_{name}"]
        attribution[name] = all(
            effect[metric]["relative_improvement_percent"]
            >= CONTROL_MIN_POINT_PERCENT
            and effect[metric]["paired_episode_cluster_bootstrap_95_ci_percent"][0]
            > 0.0
            for metric in ("top_decoded_mse_unit_range", "top_temporal_mse_unit_range")
        )
    matched_guardrails = {}
    parent_guardrails = {}
    for metric in (
        "all_decoded_mse_unit_range",
        "all_temporal_mse_unit_range",
        "future_video_latent_nmse",
    ):
        effect = baseline_effect[metric]
        matched_guardrails[metric] = (
            effect["relative_improvement_percent"] >= 0.0
            and effect["paired_episode_cluster_bootstrap_95_ci_percent"][0]
            > GUARDRAIL_CI_PERCENT
        )
        parent_metric_effect = parent_effect[metric]
        parent_guardrails[metric] = (
            parent_metric_effect["relative_improvement_percent"] >= 0.0
            and parent_metric_effect[
                "paired_episode_cluster_bootstrap_95_ci_percent"
            ][0]
            > GUARDRAIL_CI_PERCENT
        )
    causal = {
        "native_parent_sampler_bitwise_parity_before_validation": (
            parent_parity["all_bitwise"] is True
            and parent_parity["validation_dataset_opened"] is False
        ),
        "isolated_historical_6560866_vs_current_bitwise_parity": (
            parent_parity["historical_reference_process_isolated"] is True
            and parent_parity["historical_vs_current_all_bitwise"] is True
            and parent_parity["validation_dataset_opened"] is False
        ),
        "paired_200_update_trace": pairing["paired_updates"] == 200,
        "all_primary_rows_one_Wan_call": all(
            row["actual_transformer_call_count"] == 1
            for row in rows
            if row["endpoint"]["nfe"] == 1
        ),
        "zero_flow_model_calls": all(row["flow_model_call_count"] == 0 for row in rows),
        "future_rgb_not_sampler_input": all(
            row["future_rgb_sampler_input"] is False for row in rows
        ),
        "future_measured_state_not_sampler_input": all(
            row["future_measured_state_sampler_input"] is False for row in rows
        ),
        "clean_video_latent_not_sampler_input": all(
            row["clean_video_latent_sampler_input"] is False for row in rows
        ),
        "inventory_complete_materialization_before_future_rgb": (
            inventory["all_endpoints_materialized_before_future_rgb_open"] is True
            and inventory["clean_video_latent_sampler_input"] is False
        ),
        "protected_test_unopened": all(
            row["protected_test_accessed"] is False for row in rows
        ),
    }
    passed = (
        matched_decoded_primary
        and matched_lpips_primary
        and parent_decoded_primary
        and parent_lpips_primary
        and all(attribution.values())
        and all(matched_guardrails.values())
        and all(parent_guardrails.values())
        and all(causal.values())
    )
    decision = "ADVANCE_RAW_FLOW_SCAFFOLD" if passed else "STOP_FIXED_RAW_FLOW"
    raw_primary_by_key = {
        (row["clip_index"], row["noise_seed_id"]): row
        for row in _endpoint_rows(rows, raw_code)
    }
    off_same_checkpoint_by_key = {
        (row["clip_index"], row["noise_seed_id"]): row
        for row in _endpoint_rows(rows, control_codes["off"])
    }
    if set(raw_primary_by_key) != set(off_same_checkpoint_by_key):
        raise PhysicsFlowStage1Error("seam diagnostic pairing differs")
    seam_pairs = len(raw_primary_by_key)
    seam_diagnostics = {
        "nonselectable": True,
        "cannot_rescue_primary_gate": True,
        "measured_geometry_oracle_included": False,
        "measured_geometry_oracle_deferred_to_separate_preregistration": True,
        "training_terminal": pairing[
            "terminal_nonselectable_seam_diagnostics"
        ],
        "raw_checkpoint_nfe1_condition_rms_by_source": {
            source: float(
                np.mean(
                    [
                        row["seam_diagnostics"]["condition_rms"]
                        for row in _endpoint_rows(
                            rows, f"raw_checkpoint_{source}_nfe_1"
                        )
                    ]
                )
            )
            for source in RUNTIME_SOURCES
        },
        "raw_checkpoint_nfe1_effective_adapter_gate_values": sorted(
            {
                float(row["seam_diagnostics"]["effective_adapter_gate"])
                for row in raw_primary_by_key.values()
            }
        ),
        "raw_vs_same_checkpoint_off_paired_rows": seam_pairs,
        "raw_vs_same_checkpoint_off_final_latent_hash_changed_fraction": (
            sum(
                raw_primary_by_key[key]["tensor_sha256"][
                    "final_video_latent_sha256"
                ]
                != off_same_checkpoint_by_key[key]["tensor_sha256"][
                    "final_video_latent_sha256"
                ]
                for key in raw_primary_by_key
            )
            / seam_pairs
        ),
        "raw_vs_same_checkpoint_off_decoded_hash_changed_fraction": (
            sum(
                raw_primary_by_key[key]["tensor_sha256"]["decoded_future_sha256"]
                != off_same_checkpoint_by_key[key]["tensor_sha256"][
                    "decoded_future_sha256"
                ]
                for key in raw_primary_by_key
            )
            / seam_pairs
        ),
    }
    analysis = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": STUDY_ANALYSIS_KIND,
            "analyzed_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "training_pairing_identity_sha256": pairing["identity_sha256"],
            "evaluation_inventory_identity_sha256": inventory["identity_sha256"],
            "decision": decision,
            "primary_nfe": PRIMARY_NFE,
            "effects": effects,
            "secondary_dose_response": secondary,
            "gate": {
                "raw_vs_unmodified_parent_vpm_top_decoded_and_temporal": (
                    parent_decoded_primary
                ),
                "raw_vs_unmodified_parent_vpm_top_lpips": (
                    parent_lpips_primary
                ),
                "raw_vs_unmodified_parent_vpm_guardrails": parent_guardrails,
                "raw_vs_matched_off_top_decoded_and_temporal": (
                    matched_decoded_primary
                ),
                "raw_vs_matched_off_top_lpips": matched_lpips_primary,
                "same_checkpoint_attribution": attribution,
                "raw_vs_matched_off_guardrails": matched_guardrails,
                "causal_and_compute": causal,
                "all_conditions_required": True,
                "passed": passed,
            },
            "seam_diagnostics": seam_diagnostics,
            "claim_boundary": registration["claim_boundary"],
            "recurrent_or_hybrid_condition_included": False,
            "future_measured_state_opened_by_cache_or_sampler": False,
            "protected_test_accessed": False,
        }
    )
    if write:
        analysis_dir = Path(registration["output_root"]) / "analysis"
        analysis_dir.mkdir(mode=0o700)
        exclusive_json(analysis_dir / "analysis.json", analysis)
    return analysis


def command_analyze(args: argparse.Namespace) -> int:
    registration = validate_study_registration(args.registration)
    analysis = analyze_study(registration, write=not args.read_only)
    print(json.dumps(analysis, sort_keys=True))
    return 0


def command_audit_study(args: argparse.Namespace) -> int:
    registration = validate_study_registration(args.registration)
    pairing = compare_training_traces(registration, write=False)
    parent_parity, parent_parity_record = load_parent_sampler_parity(
        registration
    )
    rows, inventory = load_evaluation_rows(registration)
    analysis_path = Path(registration["output_root"]) / "analysis" / "analysis.json"
    analysis = read_json(analysis_path, "study analysis")
    replay = analyze_study(registration, write=False)
    if (
        not identity_valid(analysis)
        or analysis.get("kind") != STUDY_ANALYSIS_KIND
        or analysis.get("registration_identity_sha256")
        != registration["identity_sha256"]
    ):
        raise PhysicsFlowStage1Error("study analysis identity differs")
    # Replay timestamps and identities are intentionally fresh; every scientific
    # value must be exact.
    for key, value in analysis.items():
        if key in {"analyzed_at_utc", "identity_sha256"}:
            continue
        if replay.get(key) != value:
            raise PhysicsFlowStage1Error(f"analysis replay differs at {key}")
    # Target-blind parity JSON plus isolated historical-reference tensor file.
    artifact_hashes = 2
    for record in inventory["rank_files"]:
        _resolve_record(
            record,
            parent=Path(registration["output_root"]) / "evaluation",
            label="evaluation rank",
        )
        artifact_hashes += 1
    for record in inventory["rank_receipts"]:
        _resolve_record(
            record,
            parent=Path(registration["output_root"]) / "evaluation",
            label="evaluation rank receipt",
        )
        artifact_hashes += 1
    false_flags = require_false_flags(registration, "registration")
    false_flags += require_false_flags(pairing, "pairing")
    false_flags += require_false_flags(parent_parity, "parent parity")
    false_flags += require_false_flags(inventory, "inventory")
    false_flags += require_false_flags(rows, "rows")
    false_flags += require_false_flags(analysis, "analysis")
    audit = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": STUDY_AUDIT_KIND,
            "status": "audit_passed",
            "audited_at_utc": now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "training_pairing_identity_sha256": pairing["identity_sha256"],
            "native_parent_sampler_parity_identity_sha256": parent_parity[
                "identity_sha256"
            ],
            "native_parent_sampler_parity": parent_parity_record,
            "historical_parent_reference": parent_parity[
                "historical_reference"
            ],
            "evaluation_inventory_identity_sha256": inventory["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": analysis["decision"],
            "paired_training_updates_replayed": 200,
            "evaluation_rows_replayed": len(rows),
            "episode_cluster_effects_recomputed": sum(
                len(metrics) for metrics in analysis["effects"].values()
            ),
            "artifact_hashes_verified": artifact_hashes,
            "explicit_false_flags": false_flags,
            "equal_Wan_calls_verified": True,
            "unmodified_de65_parent_endpoint_verified": True,
            "native_parent_sampler_bitwise_parity_verified": True,
            "isolated_historical_parent_output_bitwise_parity_verified": True,
            "all_endpoints_materialized_before_future_rgb_open": True,
            "lpips_evaluator_identity_sha256": registration["runtime"][
                "lpips_alex"
            ]["identity_sha256"],
            "lpips_same_on_all_ranks_and_all_arm_endpoints": True,
            "future_rgb_sampler_input": False,
            "future_measured_state_cache_or_sampler_input": False,
            "clean_video_latent_sampler_input": False,
            "recurrent_or_hybrid_condition_included": False,
            "protected_test_accessed": False,
        }
    )
    if not args.read_only:
        exclusive_json(
            Path(registration["output_root"]) / "analysis" / "audit.json", audit
        )
    print(json.dumps(audit, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_cache = subparsers.add_parser("register-cache")
    register_cache.add_argument("--output", type=Path, required=True)
    register_cache.add_argument("--source-repo", type=Path, required=True)
    register_cache.add_argument("--expected-commit", required=True)
    register_cache.add_argument("--official-abc-root", type=Path, required=True)
    register_cache.add_argument("--renderer-gate", type=Path, required=True)
    register_cache.add_argument("--train-manifest", type=Path, required=True)
    register_cache.add_argument("--val-manifest", type=Path, required=True)
    register_cache.add_argument("--train-cache-metadata", type=Path, required=True)
    register_cache.add_argument("--val-cache-metadata", type=Path, required=True)
    register_cache.add_argument("--preprocessed-root", type=Path, required=True)
    register_cache.add_argument("--raw-root", type=Path, required=True)
    register_cache.set_defaults(handler=command_register_cache)

    build_cache = subparsers.add_parser("build-cache")
    build_cache.add_argument("--registration", type=Path, required=True)
    build_cache.add_argument("--split", choices=("train", "val"), required=True)
    build_cache.set_defaults(handler=command_build_cache)

    cache_audit = subparsers.add_parser("audit-cache")
    cache_audit.add_argument("--metadata", type=Path, required=True)
    cache_audit.add_argument("--read-only", action="store_true")
    cache_audit.set_defaults(handler=command_audit_cache)

    study = subparsers.add_parser("register-study")
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--source-repo", type=Path, required=True)
    study.add_argument("--parent-source-repo", type=Path, required=True)
    study.add_argument("--expected-commit", required=True)
    study.add_argument("--train-flow-metadata", type=Path, required=True)
    study.add_argument("--val-flow-metadata", type=Path, required=True)
    study.add_argument("--parent-snapshot", type=Path, required=True)
    study.add_argument("--parent-resolved-config", type=Path, required=True)
    study.add_argument("--legacy-parent-snapshot", type=Path, required=True)
    study.add_argument("--lineage-failed-log", type=Path, required=True)
    study.add_argument("--lineage-comparison-log", type=Path, required=True)
    study.add_argument("--causal-ladder-registration", type=Path, required=True)
    study.add_argument("--direct-frontier-root", type=Path, required=True)
    study.add_argument("--lpips-preflight-log", type=Path, required=True)
    study.add_argument("--python", type=Path, required=True)
    study.add_argument("--wan-dir", type=Path, required=True)
    study.add_argument("--videox-home", type=Path, required=True)
    study.set_defaults(handler=command_register_study)

    traces = subparsers.add_parser("compare-traces")
    traces.add_argument("--registration", type=Path, required=True)
    traces.add_argument("--read-only", action="store_true")
    traces.set_defaults(handler=command_compare_traces)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.add_argument("--read-only", action="store_true")
    analyze.set_defaults(handler=command_analyze)

    audit = subparsers.add_parser("audit-study")
    audit.add_argument("--registration", type=Path, required=True)
    audit.add_argument("--read-only", action="store_true")
    audit.set_defaults(handler=command_audit_study)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PhysicsFlowStage1Error as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
