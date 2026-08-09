#!/usr/bin/env python3
"""Identity, registration, readiness, and resource plan for IPQ-TC1.

No command in this tool submits work. ``readiness`` and ``plan`` are read-only.
``register`` is fail-closed behind an exact-SHA independent audit/ILSF handoff
receipt and creates only a fresh immutable study registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ee4855b3314e0864dece9e832158928570dc93fc"
PROTOCOL_COMMIT = "4609b39"
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "experiments"
    / "VPM_INVERTIBLE_TWO_CLOCK_LATENT_PILOT_PROTOCOL.md"
)
PARENT_SNAPSHOT_SHA256 = (
    "de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
)
LEGACY_REJECTED_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
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
LINEAGE_COMPARISON_LOG_SHA256 = (
    "048bcddd35ecd2458e5b17f28a48a8a4967111dbb888e8cd2bc9d5ff669c0e2a"
)
CURRENT_PARENT_SAMPLER_SHA256 = (
    "a10fe3730f7bb3bacd20bd14ebbcfab2b3cf8c63a2db27783f1e8a9787b87ee6"
)
CURRENT_ADAPTERS_SHA256 = (
    "f905840fe0ffe54da47279eba01029ef7c228ae8b5595624c88a66b1f317733e"
)
CURRENT_WAN_FORWARD_SHA256 = (
    "4d2d49b23c7efc391379b7eb0e2437e6555b40770402d82c17dd03f777e814c1"
)
TRAIN_MANIFEST_SHA256 = (
    "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
)
TRAIN_METADATA_SHA256 = (
    "fa22a213f352ffb8cc0b4dc0d35138b35aac349c03f362c597c621fa3473da43"
)
TRAIN_RGB_SHA256 = (
    "b5bdde4461c75bc88653c38b737021fcbd69b0b22f4c87bc8e8097c3494b64ee"
)
TRAIN_ACTIONS_SHA256 = (
    "f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89"
)
VAL_MANIFEST_SHA256 = (
    "8cb39c1f056855e28855c0b944c715d084b709e1421f4efeed3710e7099348c4"
)
VAL_RGB_SHA256 = (
    "ed82fc0f580baa90c4dc39c1608f97ad7092a4ddcb59c3612eed31711afda404"
)
VAL_ACTIONS_SHA256 = (
    "552a5cf0af156868d2866dfacabe102fc6b5cd24580bb377953e35a14625306a"
)
SCHEMA_VERSION = 1
KIND_REGISTRATION = "ipq_tc1_registration"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
NFE_GRID = (1, 2, 4)
NOISE_SEEDS = (20261202, 20261203, 20261204, 20261205)


class IPQPilotError(RuntimeError):
    """A sealed IPQ-TC1 identity or access contract was violated."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    arm_mode: str


ARMS = (
    Arm(
        "IPQ-SYNC",
        "ravenhuang/wan-dit/invertible_two_clock_sync",
        "ipq-tc1-sync-seed1234-u000400",
        "synchronous",
    ),
    Arm(
        "IPQ-INDEP",
        "ravenhuang/wan-dit/invertible_two_clock_independent",
        "ipq-tc1-independent-seed1234-u000400",
        "independent",
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


@dataclass(frozen=True)
class Endpoint:
    code: str
    schedule_mode: str
    condition_source: str
    primary: bool


SYNC_ENDPOINTS = (Endpoint("SYNC", "synchronous", "aligned", True),)
INDEPENDENT_ENDPOINTS = (
    Endpoint("P_LEADS_ALIGNED", "p_leads", "aligned", True),
    Endpoint("SYNCHRONOUS", "synchronous", "aligned", False),
    Endpoint("Q_LEADS_ALIGNED", "q_leads", "aligned", False),
    Endpoint("P_LEADS_STATE_OFF", "p_leads", "state_off", False),
    Endpoint("P_LEADS_STATE_SHUFFLED", "p_leads", "state_shuffled", False),
    Endpoint("P_LEADS_CLOCK_TIED", "p_leads", "clock_tied", False),
)
ENDPOINTS_BY_ARM = {
    "IPQ-SYNC": SYNC_ENDPOINTS,
    "IPQ-INDEP": INDEPENDENT_ENDPOINTS,
}


REQUIRED_IMPLEMENTATION_FILES = (
    "docs/experiments/VPM_INVERTIBLE_TWO_CLOCK_LATENT_PILOT_PROTOCOL.md",
    "docs/experiments/VPM_INVERTIBLE_TWO_CLOCK_LATENT_PILOT_RUNBOOK.md",
    "robot_wm/modeling/dual_diffusion/invertible_two_clock.py",
    "projects/latent_action_models/lam/invertible_two_clock_model.py",
    "robot_wm/utils/invertible_two_clock_trainer.py",
    "projects/latent_action_models/train_invertible_two_clock.py",
    "projects/latent_action_models/configs/models/dual_explicit_action_dit_invertible_two_clock.yaml",
    "projects/latent_action_models/configs/experiments_0908/invertible_two_clock_common.yaml",
    "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/invertible_two_clock_sync.yaml",
    "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/invertible_two_clock_independent.yaml",
    "tools/invertible_two_clock_pilot.py",
    "tools/invertible_two_clock_parent_parity.py",
    "tools/invertible_two_clock_evaluate.py",
    "tools/analyze_invertible_two_clock.py",
    "tools/slurm/invertible_two_clock.sbatch",
    "tools/slurm/invertible_two_clock_evaluate.sbatch",
    "tools/slurm/invertible_two_clock_workflow.py",
    "robot_wm/tests/test_invertible_two_clock.py",
    "robot_wm/tests/test_invertible_two_clock_model.py",
    "robot_wm/tests/test_invertible_two_clock_pilot.py",
    "robot_wm/tests/test_analyze_invertible_two_clock.py",
)
FORBIDDEN_MODIFIED_PRODUCTION_FILES = (
    "projects/latent_action_models/lam/dual_explicit_action_dit_model.py",
    "robot_wm/modeling/networks/wan_forward_model.py",
    "robot_wm/modeling/dual_diffusion/adapters.py",
)


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
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    expected = payload.get("identity_sha256")
    if not isinstance(expected, str) or SHA_RE.fullmatch(expected) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, rehash: bool = True) -> dict[str, Any]:
    path = path.expanduser()
    if path.is_symlink():
        raise IPQPilotError(f"input must be a regular non-symlink file: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise IPQPilotError(f"input must be a regular non-symlink file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path) if rehash else None,
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise IPQPilotError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise IPQPilotError(f"{label} must be a JSON object")
    return value


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = canonical_json(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise IPQPilotError(
            f"git {' '.join(arguments)} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def clean_source(repo: Path, expected_commit: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise IPQPilotError("expected commit must be a full 40-character SHA")
    if git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise IPQPilotError("source path must be a worktree root")
    if git(repo, "rev-parse", "HEAD") != expected_commit:
        raise IPQPilotError("source HEAD differs from the acknowledged SHA")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise IPQPilotError("source worktree must be clean")
    if git(repo, "merge-base", "--is-ancestor", BASE_COMMIT, expected_commit):
        raise IPQPilotError("implementation does not descend from the frozen base")
    if git(repo, "merge-base", "--is-ancestor", PROTOCOL_COMMIT, expected_commit):
        raise IPQPilotError("implementation does not include the corrected protocol")
    return {
        "path": str(repo),
        "git_commit": expected_commit,
        "git_tree_sha": git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def fixed_protocol() -> dict[str, Any]:
    return {
        "protocol_version": "ipq-tc1-v1",
        "base_commit": BASE_COMMIT,
        "protocol_commit": PROTOCOL_COMMIT,
        "arms": [asdict(arm) for arm in ARMS],
        "endpoints_by_arm": {
            arm: [asdict(endpoint) for endpoint in endpoints]
            for arm, endpoints in ENDPOINTS_BY_ARM.items()
        },
        "nfe_grid": list(NFE_GRID),
        "noise_seeds": list(NOISE_SEEDS),
        "updates": 400,
        "seed": 1234,
        "world_size": 8,
        "local_batch_size": 1,
        "wan_calls_per_training_update": 1,
        "endpoint_split_opened_during_training": False,
        "projector": "view-isolated spatial Haar LL P; Q=I-P",
        "clock_control": "draw both; sync uses q=p, independent uses spare q",
        "inference_schedules": {
            "synchronous": "q=s",
            "p_leads": "q=sqrt(s)",
            "q_leads": "q=s^2",
        },
        "primary_threshold_percent": 3.0,
        "attribution_threshold_percent": 1.0,
        "protected_test_access_allowed": False,
        "clean_future_feature_allowed_at_inference": False,
        "teacher_or_feature_encoder_allowed": False,
    }


def arm_identity(registration: Mapping[str, Any], arm: Arm) -> str:
    registration_core = registration.get(
        "registration_core_identity_sha256", registration["identity_sha256"]
    )
    return identity_payload(
        {
            "kind": "ipq_tc1_arm_identity",
            "registration_core_identity_sha256": registration_core,
            "source_commit": registration["tool_repository"]["git_commit"],
            "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "arm": asdict(arm),
            "updates": 400,
            "seed": 1234,
            "world_size": 8,
            "local_batch_size": 1,
        }
    )["identity_sha256"]


def _manifest(path: Path, split: str, count: int, expected_sha: str) -> list[dict[str, Any]]:
    record = file_record(path)
    if record["sha256"] != expected_sha:
        raise IPQPilotError(f"{split} manifest SHA differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise IPQPilotError(f"invalid manifest row {line_number}") from exc
            if not isinstance(value, dict):
                raise IPQPilotError("manifest row must be an object")
            rows.append(value)
    if len(rows) != count:
        raise IPQPilotError(f"{split} manifest count differs")
    clip_ids = set()
    episodes = set()
    for index, row in enumerate(rows):
        if (
            row.get("split") != split
            or int(row.get("auxiliary_index", -1)) != index
            or not isinstance(row.get("clip_id"), str)
            or row["clip_id"] in clip_ids
            or not isinstance(row.get("episode_dir"), str)
            or row["episode_dir"] in episodes
        ):
            raise IPQPilotError(f"{split} manifest identity differs at {index}")
        clip_ids.add(row["clip_id"])
        episodes.add(row["episode_dir"])
    return rows


def _metadata(
    path: Path,
    *,
    split: str,
    count: int,
    expected_metadata_sha: str | None,
    expected_rgb_sha: str,
    expected_actions_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    if expected_metadata_sha is not None and record["sha256"] != expected_metadata_sha:
        raise IPQPilotError(f"{split} metadata SHA differs")
    value = read_json(path, f"{split} metadata")
    if (
        value.get("complete") is not True
        or value.get("split") != split
        or int(value.get("clip_count", -1)) != count
        or value.get("rgb_sha256") != expected_rgb_sha
        or value.get("actions_sha256") != expected_actions_sha
    ):
        raise IPQPilotError(f"{split} metadata content differs")
    arrays = {}
    for logical, key, digest in (
        ("rgb", "rgb_file", expected_rgb_sha),
        ("actions", "actions_file", expected_actions_sha),
    ):
        array_path = Path(str(value.get(key, "")))
        if not array_path.is_absolute():
            array_path = path.parent / array_path
        if array_path.is_symlink():
            raise IPQPilotError(f"{split} {logical} array is unavailable")
        array_path = array_path.resolve(strict=True)
        if not array_path.is_file():
            raise IPQPilotError(f"{split} {logical} array is unavailable")
        # Do not read endpoint array bytes during registration. The immutable
        # producer digest is rebound and the evaluator rehashes after both arm
        # completion receipts exist.
        arrays[logical] = {
            "path": str(array_path),
            "bytes": array_path.stat().st_size,
            "sha256": digest,
            "rehash_deferred_until_post_training_evaluation": True,
        }
    return record, arrays


def validate_registration(path: Path) -> dict[str, Any]:
    registration = read_json(path.resolve(strict=True), "IPQ registration")
    core = dict(registration)
    core.pop("identity_sha256", None)
    core.pop("registration_core_identity_sha256", None)
    core.pop("arm_run_identity_sha256", None)
    expected_core_identity = identity_payload(core)["identity_sha256"]
    if (
        not identity_valid(registration)
        or registration.get("kind") != KIND_REGISTRATION
        or registration.get("fixed_protocol") != fixed_protocol()
        or registration.get("parent", {}).get("snapshot", {}).get("sha256")
        != PARENT_SNAPSHOT_SHA256
        or registration.get("parent", {}).get("resolved_config", {}).get("sha256")
        != PARENT_RESOLVED_CONFIG_SHA256
        or registration.get("parent", {}).get("run_identity_sha256")
        != PARENT_RUN_IDENTITY_SHA256
        or registration.get("parent", {}).get("canonical_model_state_sha256")
        != PARENT_CANONICAL_MODEL_STATE_SHA256
        or registration.get("registration_core_identity_sha256")
        != expected_core_identity
    ):
        raise IPQPilotError("IPQ registration identity differs")
    clean_source(
        Path(registration["tool_repository"]["path"]),
        registration["tool_repository"]["git_commit"],
    )
    if registration.get("arm_run_identity_sha256") != {
        arm.code: arm_identity(registration, arm) for arm in ARMS
    }:
        raise IPQPilotError("registered arm identities differ")
    return registration


def command_readiness(args: argparse.Namespace) -> int:
    source = clean_source(args.source_repo, args.expected_commit)
    remote_url = git(args.source_repo, "remote", "get-url", args.remote)
    remote_lines = git(args.source_repo, "ls-remote", args.remote).splitlines()
    remote_refs = sorted(
        line.split("\t", 1)[1]
        for line in remote_lines
        if line.startswith(args.expected_commit + "\t") and "\t" in line
    )
    if not remote_refs:
        raise IPQPilotError(
            "exact implementation commit is not reachable from the audited remote"
        )
    missing = [
        relative
        for relative in REQUIRED_IMPLEMENTATION_FILES
        if not (args.source_repo / relative).is_file()
    ]
    if missing:
        raise IPQPilotError(f"required implementation files missing: {missing}")
    changed = set(
        filter(
            None,
            git(args.source_repo, "diff", "--name-only", f"{BASE_COMMIT}..{args.expected_commit}").splitlines(),
        )
    )
    forbidden = sorted(changed & set(FORBIDDEN_MODIFIED_PRODUCTION_FILES))
    if forbidden:
        raise IPQPilotError(f"production sampler dependencies were modified: {forbidden}")
    parent_source_parity = {
        "projects/latent_action_models/lam/dual_explicit_action_dit_model.py": CURRENT_PARENT_SAMPLER_SHA256,
        "robot_wm/modeling/dual_diffusion/adapters.py": CURRENT_ADAPTERS_SHA256,
        "robot_wm/modeling/networks/wan_forward_model.py": CURRENT_WAN_FORWARD_SHA256,
    }
    for relative, expected in parent_source_parity.items():
        if sha256(args.source_repo / relative) != expected:
            raise IPQPilotError(f"parent sampler dependency changed: {relative}")
    files = {
        relative: file_record(args.source_repo / relative)
        for relative in REQUIRED_IMPLEMENTATION_FILES
    }
    test_report = None
    if args.test_report is not None:
        test_report = read_json(args.test_report.resolve(strict=True), "test report")
        if (
            test_report.get("source_commit") != args.expected_commit
            or test_report.get("passed") is not True
            or not isinstance(test_report.get("command"), list)
        ):
            raise IPQPilotError("test report does not seal this exact source")
    payload = identity_payload(
        {
            "kind": "ipq_tc1_readiness_seal",
            "created_at_utc": now(),
            "source": source,
            "remote_reachability": {
                "remote": args.remote,
                "url": remote_url,
                "exact_commit": args.expected_commit,
                "refs": remote_refs,
                "reachable": True,
            },
            "base_commit": BASE_COMMIT,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_sha256": files[
                "docs/experiments/VPM_INVERTIBLE_TWO_CLOCK_LATENT_PILOT_PROTOCOL.md"
            ]["sha256"],
            "changed_files": sorted(changed),
            "required_files": files,
            "forbidden_production_files_modified": forbidden,
            "parent_source_parity_sha256": parent_source_parity,
            "nonduplication_ledger": [
                "not_two_global_points_or_stopped_consistency",
                "not_lossy_rgb_or_semantic_auxiliary",
                "not_frozen_posthoc_residual_projection",
                "not_untrained_scalar-clock_ILSF-2_midpoint",
                "simultaneous_native_PQ_independent_clocks_one_shared_Wan",
            ],
            "parent": {
                "selected_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                "legacy_rejected_snapshot_sha256": LEGACY_REJECTED_SNAPSHOT_SHA256,
                "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
                "resolved_config_sha256": PARENT_RESOLVED_CONFIG_SHA256,
                "training_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
                "current_parent_sampler_sha256": CURRENT_PARENT_SAMPLER_SHA256,
                "current_adapters_sha256": CURRENT_ADAPTERS_SHA256,
                "current_wan_forward_sha256": CURRENT_WAN_FORWARD_SHA256,
            },
            "test_report": test_report,
            "test_report_record": (
                None if args.test_report is None else file_record(args.test_report)
            ),
            "model_config_inventory": {
                relative: files[relative]
                for relative in REQUIRED_IMPLEMENTATION_FILES
                if relative.startswith("projects/latent_action_models/configs/")
                or relative.endswith("invertible_two_clock_model.py")
            },
            "resource_plan": resource_plan(),
            "ilsf_handoff": {
                "decision": args.ilsf_handoff_decision,
                "audit_status": args.ilsf_audit_status,
                "registration_gate_open": (
                    args.ilsf_handoff_decision == "ADVANCE_IPQ_TC1"
                    and args.ilsf_audit_status == "PASS"
                ),
                "terminal_stop": (
                    args.ilsf_handoff_decision
                    == "NO_GO_GENERIC_EARLY_SUBSPACE"
                ),
            },
            "registration_created": False,
            "jobs_submitted": 0,
            "wandb_writes": 0,
            "outcomes_opened": 0,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def resource_plan() -> dict[str, Any]:
    return {
        "training_allocations": 2,
        "training_nodes_each": 1,
        "training_b200_each": 8,
        "training_time_limit_hours_each": 4,
        "evaluation_allocations": 2,
        "evaluation_nodes_each": 1,
        "evaluation_b200_each": 8,
        "evaluation_time_limit_hours_each": 2,
        "maximum_reserved_b200_hours": 96,
        "analysis": "cpu_only",
        "minimum_free_output_bytes": 1 << 40,
        "artifact_filesystem": "registered_fresh_Lustre_root",
        "requeue": False,
    }


def command_resource_plan(_args: argparse.Namespace) -> int:
    print(json.dumps(resource_plan(), indent=2, sort_keys=True))
    return 0


def command_register(args: argparse.Namespace) -> int:
    source = clean_source(args.source_repo, args.expected_commit)
    authorization = read_json(
        args.authorization.resolve(strict=True), "independent launch authorization"
    )
    if (
        authorization.get("kind") != "ipq_tc1_exact_sha_launch_authorization"
        or authorization.get("source_commit") != args.expected_commit
        or authorization.get("ilsf_handoff_decision") != "ADVANCE_IPQ_TC1"
        or authorization.get("independent_auditor_acknowledged") is not True
        or authorization.get("user_execution_authorized") is not True
        or not isinstance(authorization.get("auditor"), str)
        or not authorization.get("auditor")
    ):
        raise IPQPilotError("ILSF handoff/exact-SHA authorization is absent")
    output = args.output.expanduser()
    if not output.is_absolute() or Path("/lustre") not in output.parents:
        raise IPQPilotError("study output must be an absolute Lustre path")
    output = output.parent.resolve(strict=True) / output.name
    if output.exists() or output.is_symlink():
        raise IPQPilotError("study output must be fresh")
    parent_snapshot = file_record(args.parent_snapshot)
    parent_config = file_record(args.parent_resolved_config)
    lineage_log = file_record(args.lineage_comparison_log)
    parent_parity = read_json(
        args.parent_parity_audit.resolve(strict=True), "parent historical parity audit"
    )
    if parent_snapshot["sha256"] != PARENT_SNAPSHOT_SHA256:
        raise IPQPilotError("selected parent is not faithful de65")
    if parent_snapshot["sha256"] == LEGACY_REJECTED_SNAPSHOT_SHA256:
        raise IPQPilotError("legacy f67 parent is forbidden")
    if parent_config["sha256"] != PARENT_RESOLVED_CONFIG_SHA256:
        raise IPQPilotError("parent resolved config differs")
    if lineage_log["sha256"] != LINEAGE_COMPARISON_LOG_SHA256:
        raise IPQPilotError("parent lineage comparison evidence differs")
    if (
        not identity_valid(parent_parity)
        or parent_parity.get("kind") != "ipq_tc1_parent_parity_comparison"
        or parent_parity.get("status") != "PASS_BIT_EXACT"
        or parent_parity.get("parent_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or parent_parity.get("parent_run_identity_sha256")
        != PARENT_RUN_IDENTITY_SHA256
        or parent_parity.get("parent_canonical_model_state_sha256")
        != PARENT_CANONICAL_MODEL_STATE_SHA256
        or parent_parity.get("parent_resolved_config_sha256")
        != PARENT_RESOLVED_CONFIG_SHA256
        or parent_parity.get("historical_source_commit")
        != PARENT_TRAINING_SOURCE_COMMIT
        or parent_parity.get("current_source_commit") != args.expected_commit
        or parent_parity.get("current_sampler_sha256")
        != CURRENT_PARENT_SAMPLER_SHA256
        or parent_parity.get("nfe_grid") != list(NFE_GRID)
        or parent_parity.get("initial_noise_bit_exact") is not True
        or parent_parity.get("final_latent_bit_exact") is not True
        or parent_parity.get("decoded_uint8_bit_exact") is not True
        or parent_parity.get("validation_array_opened") is not False
        or parent_parity.get("protected_test_accessed") is not False
        or parent_parity.get("teacher_feature_encoder_calls") != 0
        or parent_parity.get("strict_model_load_both") is not True
        or parent_parity.get("deployable_target_blind_sampler_both") is not True
        or set(parent_parity.get("per_nfe", {}))
        != {str(nfe) for nfe in NFE_GRID}
        or any(
            not all(
                parent_parity["per_nfe"][str(nfe)].get(field) is True
                for field in (
                    "initial_noise_sha256",
                    "final_latent_sha256",
                    "decoded_uint8_sha256",
                )
            )
            for nfe in NFE_GRID
        )
    ):
        raise IPQPilotError("historical/current parent parity audit differs")
    train_rows = _manifest(args.train_manifest, "train", 512, TRAIN_MANIFEST_SHA256)
    val_rows = _manifest(args.val_manifest, "val", 64, VAL_MANIFEST_SHA256)
    train_meta, train_arrays = _metadata(
        args.train_cache_metadata,
        split="train",
        count=512,
        expected_metadata_sha=TRAIN_METADATA_SHA256,
        expected_rgb_sha=TRAIN_RGB_SHA256,
        expected_actions_sha=TRAIN_ACTIONS_SHA256,
    )
    val_meta, val_arrays = _metadata(
        args.val_cache_metadata,
        split="val",
        count=64,
        expected_metadata_sha=None,
        expected_rgb_sha=VAL_RGB_SHA256,
        expected_actions_sha=VAL_ACTIONS_SHA256,
    )
    train_ids = {row["clip_id"] for row in train_rows}
    val_ids = {row["clip_id"] for row in val_rows}
    train_episodes = {row["episode_dir"] for row in train_rows}
    val_episodes = {row["episode_dir"] for row in val_rows}
    if train_ids & val_ids or train_episodes & val_episodes:
        raise IPQPilotError("train and endpoint manifests overlap")
    protocol_files = {
        relative: file_record(args.source_repo / relative)
        for relative in REQUIRED_IMPLEMENTATION_FILES
    }
    registration = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": now(),
            "output_root": str(output),
            "tool_repository": source,
            "authorization": file_record(args.authorization),
            "authorization_payload": authorization,
            "fixed_protocol": fixed_protocol(),
            "protocol_files": protocol_files,
            "parent": {
                "snapshot": parent_snapshot,
                "resolved_config": parent_config,
                "lineage_comparison_log": lineage_log,
                "historical_current_parity_audit": file_record(
                    args.parent_parity_audit
                ),
                "historical_current_parity_payload": parent_parity,
                "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
                "training_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
                "legacy_rejected_snapshot_sha256": LEGACY_REJECTED_SNAPSHOT_SHA256,
                "current_sampler_sha256": CURRENT_PARENT_SAMPLER_SHA256,
                "current_adapters_sha256": CURRENT_ADAPTERS_SHA256,
                "current_wan_forward_sha256": CURRENT_WAN_FORWARD_SHA256,
            },
            "training": {
                "manifest": file_record(args.train_manifest),
                "cache_metadata": train_meta,
                "arrays": train_arrays,
                "rows": 512,
            },
            "validation": {
                "manifest": file_record(args.val_manifest),
                "cache_metadata": val_meta,
                "arrays": val_arrays,
                "rows": 64,
                "globally_fresh": False,
                "untouched_within_run_until_both_training_completions": True,
            },
            "validation_descriptors": val_rows,
            "runtime": {
                "python": str(args.python.resolve(strict=True)),
                "wan_dir": str(args.wan_dir.resolve(strict=True)),
                "videox_home": str(args.videox_home.resolve(strict=True)),
            },
            "resource_plan": resource_plan(),
            "wandb": {
                "enabled": bool(args.enable_wandb),
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
            },
            "protected_test_accessed": False,
        }
    )
    registration["registration_core_identity_sha256"] = registration[
        "identity_sha256"
    ]
    registration["arm_run_identity_sha256"] = {
        arm.code: arm_identity(registration, arm) for arm in ARMS
    }
    # Arm identities are derived fields; recompute the outer identity once.
    registration.pop("identity_sha256")
    registration = identity_payload(registration)
    output.mkdir(mode=0o700)
    exclusive_json(output / "registration.json", registration)
    print(str(output / "registration.json"))
    return 0


def command_plan(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration)
    root = Path(registration["output_root"])
    source = Path(registration["tool_repository"]["path"])
    python = registration["runtime"]["python"]
    commands = []
    for arm in ARMS:
        run = root / "training" / arm.run_name
        evaluation = root / "evaluation" / arm.code.lower()
        commands.append(
            {
                "arm": arm.code,
                "train": [
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=8",
                    str(
                        source
                        / "projects/latent_action_models/train_invertible_two_clock.py"
                    ),
                    f"+experiments_0908={arm.config_name}",
                    f"hydra.run.dir={run}",
                    f"trainer.config.saving.save_path={run}/snapshot.pt",
                ],
                "evaluate": [
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=8",
                    str(source / "tools/invertible_two_clock_evaluate.py"),
                    "--registration",
                    str(args.registration),
                    "--arm",
                    arm.code,
                    "--output",
                    str(evaluation),
                ],
            }
        )
    print(
        json.dumps(
            {
                "kind": "ipq_tc1_dry_run_plan",
                "mode": "no_commands_executed_no_files_created",
                "source_commit": registration["tool_repository"]["git_commit"],
                "commands": commands,
                "resource_plan": resource_plan(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser("readiness")
    readiness.add_argument("--source-repo", type=Path, required=True)
    readiness.add_argument("--expected-commit", required=True)
    readiness.add_argument("--test-report", type=Path)
    readiness.add_argument("--remote", default="origin")
    readiness.add_argument(
        "--ilsf-handoff-decision",
        choices=("ADVANCE_IPQ_TC1", "NO_GO_GENERIC_EARLY_SUBSPACE"),
        required=True,
    )
    readiness.add_argument(
        "--ilsf-audit-status", choices=("PASS", "FAIL"), required=True
    )
    readiness.set_defaults(func=command_readiness)
    resources = commands.add_parser("resource-plan")
    resources.set_defaults(func=command_resource_plan)
    register = commands.add_parser("register")
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--source-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--authorization", type=Path, required=True)
    register.add_argument("--parent-snapshot", type=Path, required=True)
    register.add_argument("--parent-resolved-config", type=Path, required=True)
    register.add_argument("--lineage-comparison-log", type=Path, required=True)
    register.add_argument("--parent-parity-audit", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--val-manifest", type=Path, required=True)
    register.add_argument("--val-cache-metadata", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--enable-wandb", action="store_true")
    register.set_defaults(func=command_register)
    plan = commands.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.set_defaults(func=command_plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
