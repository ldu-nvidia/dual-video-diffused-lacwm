#!/usr/bin/env python3
"""Fail-closed provenance helpers for the stage-faithful cascade evaluation.

The evaluation is deliberately a one-shot, evaluation-only continuation of the
completed matched strict-cascade checkpoint.  It never trains or writes a
snapshot.  LACWM uses ``sigma=1`` for noise and ``sigma=0`` for clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import dual_abc_pilot as pilot


CORE_COMMIT = "5880d047347ee572fcbdd6b38df98e87bb40e335"
CONFIG_SELECTOR = "ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml"
EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
EXPECTED_SLURM_ACCOUNT = "coreai_chef_posttrain"
EXPECTED_SLURM_QOS = "normal"
REQUESTED_VIEWER_EMAIL = "ldu@nvidia.edu"
WORLD_SIZE = 8
ARTIFACT_ITERATION = 199
VIZ_SKIP_BATCHES = 4
PARENT_COMPLETED_UPDATES = 200
PARENT_RUN_IDENTITY = (
    "ea6963718edb2b7827b189f3c622e5affe9b1706fb1cb623159731c8c29486e5"
)

LACWM_BASE = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/"
    "lacwm_train"
)
RUN_ROOT = LACWM_BASE / "runs/dual_video_diffusion/ztf_first_cascade_stage_eval"
PYTHON_BIN = LACWM_BASE / "envs/lacwm-b200-py310/bin/python"
WAN_DIR = LACWM_BASE / "wan_fun_1.3b_control"
VIDEOX_HOME = LACWM_BASE / "VideoX-Fun-1d6d9c3"
DATA_ROOT = LACWM_BASE / "data/production_v1/fast_mixed_user_waived_v1"
ABC_MANIFEST = DATA_ROOT / "abc_pp/manifest.txt"
ABC_MANIFEST_SHA256 = (
    "e52232b49ffec39600aa22e2d708497f22a4ea57fc89f84bc289ae4b1e0a5c09"
)

PARENT_ROOT = (
    LACWM_BASE
    / "runs/dual_video_diffusion/ztf_first_cascade_screen"
    / "abc200-tf-cascade3-s1234-18318ed-v1"
    / "cascade_matched_s010"
)
CHECKPOINT = PARENT_ROOT / "snapshot.pt"
CHECKPOINT_BYTES = 4_249_340_573
CHECKPOINT_SHA256 = (
    "5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d"
)
PARENT_FILES = {
    "resolved_config": (
        PARENT_ROOT / "resolved_config.yaml",
        "b37b8f7517acbcc9f8e9781412e0d4c548fe60ab9ce614926382cb82d0d5b7cd",
    ),
    "arm_manifest": (
        PARENT_ROOT / "arm_manifest.json",
        "4c39c9f3c22329e082c0e6ceafd1510a2cc610cc61c736ede58c84c41ceb7f9f",
    ),
    "outcome": (
        PARENT_ROOT / "outcome.json",
        "0c8752b28381d9911becfff04e43e002d1498ce3e97d540982ba33b7652ac24f",
    ),
    "training_completion": (
        PARENT_ROOT / "training_complete.json",
        "2d30ac79e6cae4cbdb4e77ccee3a2f7fb1dd65c27d7f8664c4e78e15f96d584f",
    ),
}

NUMERIC_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
RANK_ARTIFACT_RE = re.compile(r"^latent_trajectory_rank_([0-9]+)\.safetensors$")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _identity_is_valid(payload: dict[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest() == recorded


def _signed_json(path: Path, label: str) -> dict[str, Any]:
    payload = _read_json(pilot._canonical_regular_file(str(path), label), label)
    if not _identity_is_valid(payload):
        raise RuntimeError(f"{label} has an invalid identity digest: {path}")
    return payload


def _check_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    canonical = pilot._canonical_regular_file(str(path), label)
    actual = pilot._sha256(canonical)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 changed: {actual} != {expected}")
    return {
        "path": str(canonical),
        "bytes": canonical.stat().st_size,
        "sha256": actual,
    }


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _validate_source(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    repo = pilot._canonical_directory(str(repo_root), "repository root")
    pilot._validated_commit(expected_commit)
    pilot._assert_clean_commit(repo, expected_commit)
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", CORE_COMMIT, expected_commit],
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"launch commit {expected_commit} is not based on core {CORE_COMMIT}"
        )
    evaluator = _check_sha(
        repo
        / "projects/latent_action_models/evaluate_stage_faithful.py",
        pilot._sha256(
            pilot._canonical_regular_file(
                str(
                    repo
                    / "projects/latent_action_models/evaluate_stage_faithful.py"
                ),
                "stage-faithful evaluator",
            )
        ),
        "stage-faithful evaluator",
    )
    return {
        "repository_root": str(repo),
        "git_commit": expected_commit,
        "core_commit": CORE_COMMIT,
        "clean": True,
        "evaluator": evaluator,
    }


def _wandb_private_project(entity: str, project: str) -> dict[str, Any]:
    summary = dict(pilot._wandb_private_project(entity, project))
    viewer_email = str(summary.get("viewer_email", "")).strip().lower()
    summary["requested_viewer_email"] = REQUESTED_VIEWER_EMAIL
    summary["viewer_email_matches_request"] = viewer_email == REQUESTED_VIEWER_EMAIL
    summary["identity_deviation"] = (
        None
        if summary["viewer_email_matches_request"]
        else (
            "authenticated private personal entity owner uses "
            f"{viewer_email!r}, not requested {REQUESTED_VIEWER_EMAIL!r}"
        )
    )
    return summary


def _active_job_contract(
    allowed_values: Sequence[str], observed_values: Sequence[str]
) -> dict[str, Any]:
    allowed = list(allowed_values)
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed active job IDs must be unique")
    if any(NUMERIC_JOB_ID_RE.fullmatch(value) is None for value in allowed):
        raise ValueError("allowed active job IDs must be positive numeric Slurm IDs")

    observed: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in observed_values:
        if "\n" in value or "\r" in value:
            raise ValueError("observed active job record contains a newline")
        fields = value.split("|")
        if len(fields) != 3:
            raise ValueError(
                "observed active job must be BASE_ID|DISPLAY_ID|STATE"
            )
        base_id, display_id, state = fields
        if NUMERIC_JOB_ID_RE.fullmatch(base_id) is None:
            raise ValueError("observed active job base ID must be numeric")
        if (
            not display_id
            or any(character.isspace() for character in display_id)
            or "|" in display_id
        ):
            raise ValueError("observed active job display ID is unsafe")
        if not state or state != state.upper() or not state.replace("_", "").isalpha():
            raise ValueError("observed active job state is malformed")
        if display_id in seen:
            raise ValueError(f"duplicate observed active job: {display_id}")
        seen.add(display_id)
        observed.append(
            {
                "array_or_job_id": base_id,
                "display_job_id": display_id,
                "state": state,
            }
        )

    allowed_set = set(allowed)
    observed_ids = {record["array_or_job_id"] for record in observed}
    unallowed = sorted(observed_ids - allowed_set, key=int)
    missing = sorted(allowed_set - observed_ids, key=int)
    if unallowed:
        raise RuntimeError(
            "active Slurm job IDs were not explicitly allowed: "
            + ", ".join(unallowed)
        )
    if missing:
        raise RuntimeError(
            "allowed Slurm job IDs were not active at the pre-submit check: "
            + ", ".join(missing)
        )
    return {
        "default_policy": "fail_closed_on_any_unlisted_active_user_job",
        "allow_scope": "exact_positive_numeric_array_or_job_IDs_only",
        "wildcards_or_names_allowed": False,
        "allowed_array_or_job_ids": sorted(allowed, key=int),
        "observed_active_jobs": sorted(
            observed,
            key=lambda record: (
                int(record["array_or_job_id"]),
                record["display_job_id"],
            ),
        ),
        "all_observed_jobs_explicitly_allowed": True,
    }


def expected_eval_id(commit: str) -> str:
    pilot._validated_commit(commit)
    return f"abc200-tf-cascade-stage-eval-s1234-{commit[:10]}-v1"


def expected_hydra_overrides(eval_id: str, output_root: Path) -> list[str]:
    pilot._validated_id(eval_id, "evaluation ID")
    out = str(output_root)
    return [
        f"+experiments_0908={CONFIG_SELECTOR}",
        f"name={eval_id}",
        "seed=1234",
        "data_loader.batch_size=1",
        "trainer.config.max_iter=200",
        "trainer.config.transition_handoff_path=null",
        "trainer.config.gradient_accumulation_steps=1",
        "trainer.config.load_path=null",
        "trainer.config.exclude_keys=[]",
        "trainer.config.logging.log_every=5",
        "trainer.config.saving.save_every=50",
        f"trainer.config.saving.save_path={out}/_never_write_snapshot.pt",
        "trainer.config.validation.val_every=10",
        "trainer.config.visualization.viz_every=50",
        f"trainer.config.visualization.viz_path={out}/visualization",
        "trainer.config.visualization.require_success=true",
        "model.dual_diffusion.condition_on_tf=true",
        "model.dual_diffusion.condition_mode='matched'",
        "model.dual_diffusion.state_gate_init=0.10",
        "model.dual_diffusion.state_gate_trainable=false",
        "model.dual_diffusion.schedule_mode=tf_first_cascaded",
        "model.dual_diffusion.cascade_tf_loss_probability=0.4",
        "model.dual_diffusion.cascade_logit_mean=0.0",
        "model.dual_diffusion.cascade_logit_std=1.0",
        "model.dual_diffusion.cascade_tf_condition_max_sigma=0.25",
        "model.dual_diffusion.cascade_validation_tf_sigma=0.125",
        "model.dual_diffusion.cascade_inference_tf_fraction=0.5",
        "model.dual_diffusion.cascade_condition_only_video_loss_examples=true",
        "model.dual_diffusion.cascade_stage_faithful_inference=true",
        "model.dual_diffusion.validation_video_sigmas=[0.90,0.75,0.50,0.25]",
        "model.dual_diffusion.tf_loss_weight=1.0",
        "model.dual_diffusion.capture_latent_trajectories=true",
        "model.dual_diffusion.evaluation_nfe_steps=[2,4,8]",
        "model.dual_diffusion.evaluation_noise_seed=20260726",
        (
            "model.dual_diffusion.evaluation_condition_sources="
            "[autonomous,autonomous_shuffled,autonomous_legacy,off]"
        ),
        "model.time_frequency_transform.representation='parseval_rfft'",
        f"hydra.run.dir={out}",
        f"hydra.sweep.dir={out}",
        "wandb.enabled=true",
        "wandb.mode=online",
        f"wandb.entity={EXPECTED_ENTITY}",
        f"wandb.project={EXPECTED_PROJECT}",
        (
            "wandb.tags=[abc,dual-video-diffusion,evaluation-only,"
            "stage-faithful,same-checkpoint,parseval_rfft,seed-1234]"
        ),
        f"+wandb.id={eval_id}",
        "+wandb.group=null",
        "+wandb.resume=never",
        f"+stage_faithful_evaluation.snapshot_path={CHECKPOINT}",
        f"+stage_faithful_evaluation.snapshot_sha256={CHECKPOINT_SHA256}",
        (
            "+stage_faithful_evaluation.parent_run_identity_sha256="
            f"{PARENT_RUN_IDENTITY}"
        ),
        (
            "+stage_faithful_evaluation.parent_completed_updates="
            f"{PARENT_COMPLETED_UPDATES}"
        ),
        f"+stage_faithful_evaluation.viz_skip_batches={VIZ_SKIP_BATCHES}",
        f"+stage_faithful_evaluation.artifact_iteration={ARTIFACT_ITERATION}",
    ]


def _validate_output_scope(
    output_root: Path, run_root: Path, eval_id: str
) -> tuple[Path, Path]:
    run = pilot._canonical_directory(str(run_root), "evaluation run root")
    out = pilot._canonical_directory(str(output_root), "evaluation output root")
    expected = run / eval_id
    if out != expected:
        raise RuntimeError(f"evaluation output must be exactly {expected}, got {out}")
    if out.is_symlink() or run.is_symlink():
        raise RuntimeError("evaluation output and run root must not be symlinks")
    return out, run


def _validate_parent() -> dict[str, Any]:
    checkpoint = _check_sha(CHECKPOINT, CHECKPOINT_SHA256, "matched checkpoint")
    if checkpoint["bytes"] != CHECKPOINT_BYTES:
        raise RuntimeError(
            f"matched checkpoint size changed: {checkpoint['bytes']} != "
            f"{CHECKPOINT_BYTES}"
        )
    files = {
        name: _check_sha(path, digest, f"parent {name}")
        for name, (path, digest) in PARENT_FILES.items()
    }
    arm = _read_json(PARENT_FILES["arm_manifest"][0], "parent arm manifest")
    if not _identity_is_valid(arm) or arm.get("identity_sha256") != PARENT_RUN_IDENTITY:
        raise RuntimeError("parent arm manifest identity is invalid or unexpected")
    expected_arm = {
        "arm": "cascade_matched_s010",
        "condition_on_tf": True,
        "condition_mode": "matched",
        "representation": "parseval_rfft",
        "state_gate_init": 0.1,
        "state_gate_trainable": False,
    }
    if any(arm.get(key) != value for key, value in expected_arm.items()):
        raise RuntimeError("parent arm manifest is not the matched strict cascade")
    if arm.get("run", {}).get("world_size") != WORLD_SIZE:
        raise RuntimeError("parent arm world size is not 8")
    outcome = _read_json(PARENT_FILES["outcome"][0], "parent outcome")
    if outcome.get("completed") is not True or outcome.get("exit_status") != 0:
        raise RuntimeError("parent outcome is not completed with exit status zero")
    completion = _read_json(
        PARENT_FILES["training_completion"][0], "parent completion"
    )
    expected_completion = {
        "status": "completed",
        "max_iter": PARENT_COMPLETED_UPDATES,
        "completed_updates": PARENT_COMPLETED_UPDATES,
        "run_identity_sha256": PARENT_RUN_IDENTITY,
    }
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise RuntimeError("parent training completion contract changed")
    return {
        "root": str(PARENT_ROOT),
        "checkpoint": checkpoint,
        "run_identity_sha256": PARENT_RUN_IDENTITY,
        "completed_updates": PARENT_COMPLETED_UPDATES,
        "files": files,
    }


def command_hydra_overrides(args: argparse.Namespace) -> int:
    for override in expected_hydra_overrides(args.eval_id, Path(args.output_root)):
        print(override)
    return 0


def command_wandb_private(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            _wandb_private_project(args.entity, args.project),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_create_manifest(args: argparse.Namespace) -> int:
    commit = pilot._validated_commit(args.git_commit)
    eval_id = pilot._validated_id(args.eval_id, "evaluation ID")
    if eval_id != expected_eval_id(commit):
        raise RuntimeError(
            f"evaluation ID must be {expected_eval_id(commit)}, got {eval_id}"
        )
    out, run = _validate_output_scope(
        Path(args.output_root), Path(args.run_root), eval_id
    )
    if any(out.iterdir()):
        raise RuntimeError("fresh evaluation output root must be empty")
    source = _validate_source(Path(args.repo_root), commit)
    parent = _validate_parent()
    data = _check_sha(ABC_MANIFEST, ABC_MANIFEST_SHA256, "ABC manifest")
    wandb = _wandb_private_project(EXPECTED_ENTITY, EXPECTED_PROJECT)
    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "stage_faithful_cascade_evaluation",
            "created_at_utc": pilot._now(),
            "evaluation_id": eval_id,
            "source": source,
            "output": {
                "run_root": str(run),
                "output_root": str(out),
                "fresh_at_manifest_creation": True,
                "snapshot_sentinel": str(out / "_never_write_snapshot.pt"),
                "training_completion_forbidden": str(out / "training_complete.json"),
            },
            "parent": parent,
            "data": {"root": str(DATA_ROOT), "abc_manifest": data},
            "runtime": {
                "python": str(PYTHON_BIN),
                "wan_dir": str(WAN_DIR),
                "videox_home": str(VIDEOX_HOME),
                "nodes": 1,
                "world_size": WORLD_SIZE,
                "gpu_type": "B200",
            },
            "evaluation": {
                "evaluation_only": True,
                "training_steps": 0,
                "seed": 1234,
                "artifact_iteration": ARTIFACT_ITERATION,
                "viz_skip_batches": VIZ_SKIP_BATCHES,
                "nfe_steps": [2, 4, 8],
                "noise_seed": 20260726,
                "condition_sources": [
                    "autonomous",
                    "autonomous_shuffled",
                    "autonomous_legacy",
                    "off",
                ],
                "condition_source_codes": [0, 4, 5, 1],
                "sigma_convention": "sigma=1 noise, sigma=0 clean",
                "hydra_overrides": expected_hydra_overrides(eval_id, out),
            },
            "rank_artifact_contract": {
                "iteration": ARTIFACT_ITERATION,
                "world_size": WORLD_SIZE,
                "expected_ranks": list(range(WORLD_SIZE)),
                "primary_glob": (
                    "visualization/iter_199/*/"
                    "latent_trajectory_rank_*.safetensors"
                ),
                "exact_primary_artifact_count": WORLD_SIZE,
                "sidecar_required": True,
                "decoded_mp4_per_rank": 1,
            },
            "wandb": {
                **wandb,
                "entity": EXPECTED_ENTITY,
                "project": EXPECTED_PROJECT,
                "run_id": eval_id,
                "group": None,
                "resume": "never",
                "mode": "online",
                "evaluation_step": 0,
            },
            "slurm": {
                "array": False,
                "nodes": 1,
                "gpus_per_node": WORLD_SIZE,
                "cpus_per_task": 64,
                "memory": "600G",
                "time_limit": "00:30:00",
                "partition": "batch",
                "account": EXPECTED_SLURM_ACCOUNT,
                "qos": EXPECTED_SLURM_QOS,
                "requeue": False,
                "active_job_coexistence": _active_job_contract(
                    args.allow_active_job_id, args.observed_active_job
                ),
            },
        }
    )
    pilot._exclusive_json(Path(args.output), payload)
    print(payload["identity_sha256"])
    return 0


def command_validate_manifest(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(args.manifest, "eval manifest")
    manifest = _signed_json(manifest_path, "eval manifest")
    commit = pilot._validated_commit(args.git_commit)
    eval_id = expected_eval_id(commit)
    if (
        manifest.get("kind") != "stage_faithful_cascade_evaluation"
        or manifest.get("evaluation_id") != eval_id
        or manifest.get("source", {}).get("git_commit") != commit
    ):
        raise RuntimeError("evaluation manifest identity does not match this launch")
    out, _ = _validate_output_scope(
        Path(manifest["output"]["output_root"]),
        Path(manifest["output"]["run_root"]),
        eval_id,
    )
    if manifest.get("evaluation", {}).get(
        "hydra_overrides"
    ) != expected_hydra_overrides(eval_id, out):
        raise RuntimeError("evaluation manifest Hydra vector changed")
    _validate_source(Path(args.repo_root), commit)
    _validate_parent()
    _check_sha(ABC_MANIFEST, ABC_MANIFEST_SHA256, "ABC manifest")
    print(manifest["identity_sha256"])
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(args.manifest, "eval manifest")
    manifest = _signed_json(manifest_path, "eval manifest")
    if NUMERIC_JOB_ID_RE.fullmatch(args.job_id) is None:
        raise ValueError("Slurm job ID must be a positive non-array numeric ID")
    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "stage_faithful_cascade_evaluation_submission",
            "created_at_utc": pilot._now(),
            "manifest": {
                "path": str(manifest_path),
                "sha256": pilot._sha256(manifest_path),
                "identity_sha256": manifest["identity_sha256"],
            },
            "job_id": args.job_id,
            "array": False,
            "immediate_pre_sbatch_active_job_coexistence": _active_job_contract(
                manifest["slurm"]["active_job_coexistence"][
                    "allowed_array_or_job_ids"
                ],
                args.observed_active_job,
            ),
        }
    )
    pilot._exclusive_json(Path(args.output), payload)
    return 0


def _artifact_inventory(output_root: Path) -> dict[str, Any]:
    forbidden = [
        output_root / "_never_write_snapshot.pt",
        output_root / "training_complete.json",
    ]
    present = [str(path) for path in forbidden if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(
            "evaluation-only forbidden output exists: " + ", ".join(present)
        )
    iteration_root = output_root / "visualization" / f"iter_{ARTIFACT_ITERATION}"
    if not iteration_root.is_dir() or iteration_root.is_symlink():
        raise RuntimeError(f"artifact iteration directory is missing: {iteration_root}")
    paths = sorted(
        iteration_root.glob("*/latent_trajectory_rank_*.safetensors")
    )
    if len(paths) != WORLD_SIZE:
        raise RuntimeError(
            f"expected exactly {WORLD_SIZE} rank artifacts, found {len(paths)}"
        )
    ranks: dict[int, dict[str, Any]] = {}
    for path in paths:
        match = RANK_ARTIFACT_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"malformed rank artifact name: {path}")
        rank = int(match.group(1))
        if rank in ranks or rank not in range(WORLD_SIZE):
            raise RuntimeError(f"duplicate or out-of-range rank artifact: {rank}")
        primary = _check_sha(
            path, pilot._sha256(path), f"rank {rank} trajectory"
        )
        sidecar_path = path.with_suffix(".json")
        sidecar = _read_json(
            pilot._canonical_regular_file(
                str(sidecar_path), f"rank {rank} sidecar"
            ),
            f"rank {rank} sidecar",
        )
        if (
            sidecar.get("global_rank") != rank
            or sidecar.get("iteration") != ARTIFACT_ITERATION
            or sidecar.get("safetensors_sha256") != primary["sha256"]
            or sidecar.get("sigma_convention") != "1=noise,0=clean"
            or sidecar.get("dataset") != path.parent.name
        ):
            raise RuntimeError(f"rank {rank} sidecar contract is invalid")
        video_path = path.parent / f"viz_{sidecar['dataset']}_{rank}_0.mp4"
        if not video_path.is_file() or video_path.is_symlink():
            raise RuntimeError(
                f"rank {rank} decoded MP4 is missing or symlinked: {video_path}"
            )
        video = _check_sha(
            video_path, pilot._sha256(video_path), f"rank {rank} video"
        )
        ranks[rank] = {
            "rank": rank,
            "trajectory": primary,
            "sidecar": {
                "path": str(sidecar_path),
                "bytes": sidecar_path.stat().st_size,
                "sha256": pilot._sha256(sidecar_path),
            },
            "decoded_video": video,
        }
    if sorted(ranks) != list(range(WORLD_SIZE)):
        raise RuntimeError("rank artifact inventory is not exactly ranks 0 through 7")
    return {
        "artifact_iteration": ARTIFACT_ITERATION,
        "world_size": WORLD_SIZE,
        "primary_artifact_count": len(ranks),
        "ranks": [ranks[rank] for rank in range(WORLD_SIZE)],
    }


def command_complete(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(args.manifest, "eval manifest")
    manifest = _signed_json(manifest_path, "eval manifest")
    output_root = Path(manifest["output"]["output_root"])
    inventory_payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "stage_faithful_cascade_evaluation_artifact_inventory",
            "created_at_utc": pilot._now(),
            "manifest": {
                "path": str(manifest_path),
                "sha256": pilot._sha256(manifest_path),
                "identity_sha256": manifest["identity_sha256"],
            },
            **_artifact_inventory(output_root),
        }
    )
    inventory_path = Path(args.inventory_output)
    pilot._exclusive_json(inventory_path, inventory_payload)
    completion_payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "stage_faithful_cascade_evaluation_completion",
            "status": "completed",
            "created_at_utc": pilot._now(),
            "evaluation_id": manifest["evaluation_id"],
            "evaluation_only": True,
            "training_steps": 0,
            "world_size": WORLD_SIZE,
            "rank_artifact_count": WORLD_SIZE,
            "manifest": {
                "path": str(manifest_path),
                "sha256": pilot._sha256(manifest_path),
                "identity_sha256": manifest["identity_sha256"],
            },
            "artifact_inventory": {
                "path": str(inventory_path),
                "sha256": pilot._sha256(inventory_path),
                "identity_sha256": inventory_payload["identity_sha256"],
            },
        }
    )
    pilot._exclusive_json(Path(args.completion_output), completion_payload)
    return 0


def command_record_outcome(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(args.manifest, "eval manifest")
    manifest = _signed_json(manifest_path, "eval manifest")
    exit_status = int(args.exit_status)
    completed = exit_status == 0
    completion: dict[str, Any] | None = None
    if completed:
        completion_path = pilot._canonical_regular_file(
            args.completion, "evaluation completion"
        )
        completion_payload = _signed_json(completion_path, "evaluation completion")
        if completion_payload.get("status") != "completed":
            raise RuntimeError("evaluation completion status is not completed")
        completion = {
            "path": str(completion_path),
            "sha256": pilot._sha256(completion_path),
            "identity_sha256": completion_payload["identity_sha256"],
        }
    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "stage_faithful_cascade_evaluation_outcome",
            "created_at_utc": pilot._now(),
            "completed": completed,
            "exit_status": exit_status,
            "slurm_job_id": args.slurm_job_id,
            "manifest": {
                "path": str(manifest_path),
                "sha256": pilot._sha256(manifest_path),
                "identity_sha256": manifest["identity_sha256"],
            },
            "completion": completion,
        }
    )
    pilot._exclusive_json(Path(args.output), payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    hydra = subparsers.add_parser("hydra-overrides")
    hydra.add_argument("--eval-id", required=True)
    hydra.add_argument("--output-root", required=True)
    hydra.set_defaults(func=command_hydra_overrides)

    wandb = subparsers.add_parser("wandb-private")
    wandb.add_argument("--entity", required=True)
    wandb.add_argument("--project", required=True)
    wandb.set_defaults(func=command_wandb_private)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--eval-id", required=True)
    create.add_argument("--git-commit", required=True)
    create.add_argument("--repo-root", required=True)
    create.add_argument("--run-root", required=True)
    create.add_argument("--output-root", required=True)
    create.add_argument("--allow-active-job-id", action="append", default=[])
    create.add_argument("--observed-active-job", action="append", default=[])
    create.add_argument("--output", required=True)
    create.set_defaults(func=command_create_manifest)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--git-commit", required=True)
    validate.add_argument("--repo-root", required=True)
    validate.set_defaults(func=command_validate_manifest)

    submission = subparsers.add_parser("record-submission")
    submission.add_argument("--manifest", required=True)
    submission.add_argument("--job-id", required=True)
    submission.add_argument("--observed-active-job", action="append", default=[])
    submission.add_argument("--output", required=True)
    submission.set_defaults(func=command_record_submission)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--manifest", required=True)
    complete.add_argument("--inventory-output", required=True)
    complete.add_argument("--completion-output", required=True)
    complete.set_defaults(func=command_complete)

    outcome = subparsers.add_parser("record-outcome")
    outcome.add_argument("--manifest", required=True)
    outcome.add_argument("--completion", required=True)
    outcome.add_argument("--exit-status", required=True, type=int)
    outcome.add_argument("--slurm-job-id", required=True)
    outcome.add_argument("--output", required=True)
    outcome.set_defaults(func=command_record_outcome)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
