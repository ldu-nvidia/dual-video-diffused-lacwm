#!/usr/bin/env python3
"""Fail-closed provenance for the three-arm privileged-video evaluation.

Every arm strict-loads one completed 200-update parent and performs no
optimizer update.  TF content and the independent TF clock residual are both
disabled at evaluation, so all model calls advance the aligned video clock.
LACWM uses ``sigma=1`` for noise and ``sigma=0`` for clean data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import stage_faithful_eval_launch as stage


pilot = stage.pilot
CORE_COMMIT = "af77f8a556cb7204e1aa55b347c123d68b24482b"
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
SNAPSHOT_BYTES = 4_249_340_573
EVALUATION_NFE_STEPS = [1, 2, 4, 8]
EVALUATION_CONDITION_SOURCES = ["autonomous", "off"]

LACWM_BASE = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/"
    "lacwm_train"
)
RUN_ROOT = LACWM_BASE / "runs/dual_video_diffusion/privileged_video_eval"
PYTHON_BIN = LACWM_BASE / "envs/lacwm-b200-py310/bin/python"
PYTHON_LINK_TARGET = Path(
    "/lustre/fsw/portfolios/coreai/users/ldu/lacwm_train/python/"
    "cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
)
PYTHON_REAL_BIN = (
    LACWM_BASE
    / "python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
)
WAN_DIR = LACWM_BASE / "wan_fun_1.3b_control"
VIDEOX_HOME = LACWM_BASE / "VideoX-Fun-1d6d9c3"
DATA_ROOT = LACWM_BASE / "data/production_v1/fast_mixed_user_waived_v1"
ABC_MANIFEST = DATA_ROOT / "abc_pp/manifest.txt"
ABC_MANIFEST_SHA256 = (
    "e52232b49ffec39600aa22e2d708497f22a4ea57fc89f84bc289ae4b1e0a5c09"
)
PARENT_BASE = (
    LACWM_BASE
    / "runs/dual_video_diffusion/ztf_first_cascade_screen"
    / "abc200-tf-cascade3-s1234-18318ed-v1"
)

# Array order, labels, parent identities, and every immutable parent digest are
# one preregistered contract.  Do not derive these values from mutable state.
ARMS: tuple[dict[str, Any], ...] = (
    {
        "array_task_id": 0,
        "label": "trained_off",
        "parent_arm": "cascade_off_s000",
        "parent_identity_sha256": (
            "8861147ccfcc0a2909480400d7f09452ae192298ac758f1cf73f71802d0b5f9b"
        ),
        "snapshot_sha256": (
            "a147acb27dec8fb9f793d665861149ebc8d203b63ab1e6d107760f62d0b36e6b"
        ),
        "condition_on_tf": False,
        "condition_mode": "off",
        "state_gate_init": 0.0,
        "file_sha256": {
            "resolved_config": (
                "55c9bd043268299f605ffaf842293a26a358b91309b4fc7adf086befc4d980b1"
            ),
            "arm_manifest": (
                "048a43e9480618c6b5631c73241b4b62d38f1a1b5366e8c05071c4ad034295c2"
            ),
            "outcome": (
                "a23ad210abc907d6617f2540a082287fa4350169374d52200a0aab172c6bc03a"
            ),
            "training_completion": (
                "bcb6ec14acf746c1f06a8ab7890e32a366a2258e1f87b76245cfcb94e6652a19"
            ),
        },
    },
    {
        "array_task_id": 1,
        "label": "trained_matched",
        "parent_arm": "cascade_matched_s010",
        "parent_identity_sha256": (
            "ea6963718edb2b7827b189f3c622e5affe9b1706fb1cb623159731c8c29486e5"
        ),
        "snapshot_sha256": (
            "5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d"
        ),
        "condition_on_tf": True,
        "condition_mode": "matched",
        "state_gate_init": 0.1,
        "file_sha256": {
            "resolved_config": (
                "b37b8f7517acbcc9f8e9781412e0d4c548fe60ab9ce614926382cb82d0d5b7cd"
            ),
            "arm_manifest": (
                "4c39c9f3c22329e082c0e6ceafd1510a2cc610cc61c736ede58c84c41ceb7f9f"
            ),
            "outcome": (
                "0c8752b28381d9911becfff04e43e002d1498ce3e97d540982ba33b7652ac24f"
            ),
            "training_completion": (
                "2d30ac79e6cae4cbdb4e77ccee3a2f7fb1dd65c27d7f8664c4e78e15f96d584f"
            ),
        },
    },
    {
        "array_task_id": 2,
        "label": "trained_shuffled",
        "parent_arm": "cascade_shuffled_s010",
        "parent_identity_sha256": (
            "151cf2b0a349878839f515b34ea4f8edd6a34528c2814c12b1fdc5afc9a3645f"
        ),
        "snapshot_sha256": (
            "1b5e70982d1a93b4069b8ad1c33b25ba1b4d106c560dcaad63e1d2dd23c3eb76"
        ),
        "condition_on_tf": True,
        "condition_mode": "shuffled",
        "state_gate_init": 0.1,
        "file_sha256": {
            "resolved_config": (
                "1d2090a12746ba28955d2eb9b675b7624c19760807d5270b9a19b738ecbc8ea4"
            ),
            "arm_manifest": (
                "387811b953233611eadcb1f497a74d1fdab603cb0cf6eb2b2d473d7bdd5d295a"
            ),
            "outcome": (
                "9cbf8c442e029dc44695563968a599f98705d8c66d7b6582c6a2ad4ed065d1b1"
            ),
            "training_completion": (
                "b82fad6671c00347e27f16e2a7717c4ecd55eeba00e2b85f42c8ce5e77cc6dc9"
            ),
        },
    },
)

NUMERIC_JOB_ID_RE = stage.NUMERIC_JOB_ID_RE
_active_job_contract = stage._active_job_contract
_identity_is_valid = stage._identity_is_valid


def _arm_by_task(array_task_id: int) -> dict[str, Any]:
    if array_task_id < 0 or array_task_id >= len(ARMS):
        raise ValueError("array task ID must be exactly one of 0, 1, or 2")
    return ARMS[array_task_id]


def _arm_by_label(label: str) -> dict[str, Any]:
    matches = [arm for arm in ARMS if arm["label"] == label]
    if len(matches) != 1:
        raise ValueError(
            "arm label must be trained_off, trained_matched, or trained_shuffled"
        )
    return matches[0]


def expected_eval_id(commit: str, arm_label: str) -> str:
    pilot._validated_commit(commit)
    _arm_by_label(arm_label)
    return f"abc200-priv-video-{arm_label}-s1234-{commit[:10]}-v1"


def _parent_root(arm: dict[str, Any]) -> Path:
    return PARENT_BASE / str(arm["parent_arm"])


def expected_hydra_overrides(
    arm_label: str, eval_id: str, output_root: Path
) -> list[str]:
    arm = _arm_by_label(arm_label)
    pilot._validated_id(eval_id, "evaluation ID")
    expected_id_prefix = f"abc200-priv-video-{arm_label}-s1234-"
    if not eval_id.startswith(expected_id_prefix):
        raise RuntimeError("evaluation ID does not match the privileged arm")
    parent_root = _parent_root(arm)
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
        "+trainer.config.share_spatial_attention=false",
        "trainer.config.logging.log_every=5",
        "trainer.config.saving.save_every=50",
        f"trainer.config.saving.save_path={out}/_never_write_snapshot.pt",
        "trainer.config.validation.val_every=10",
        "trainer.config.visualization.viz_every=50",
        f"trainer.config.visualization.viz_path={out}/visualization",
        "trainer.config.visualization.require_success=true",
        "model.dual_diffusion.enabled=true",
        "model.dual_diffusion.condition_on_tf=false",
        "model.dual_diffusion.condition_mode='off'",
        f"model.dual_diffusion.state_gate_init={arm['state_gate_init']:.2f}",
        "model.dual_diffusion.state_gate_trainable=false",
        "model.dual_diffusion.schedule_mode=aligned",
        "model.dual_diffusion.evaluation_disable_tf_clock=true",
        "model.dual_diffusion.cascade_stage_faithful_inference=false",
        "model.dual_diffusion.tf_loss_weight=1.0",
        "model.dual_diffusion.capture_latent_trajectories=true",
        "model.dual_diffusion.evaluation_nfe_steps=[1,2,4,8]",
        "model.dual_diffusion.evaluation_noise_seed=20260726",
        "model.dual_diffusion.evaluation_condition_sources=[autonomous,off]",
        "model.time_frequency_transform.representation='parseval_rfft'",
        f"hydra.run.dir={out}",
        f"hydra.sweep.dir={out}",
        "wandb.enabled=true",
        "wandb.mode=online",
        f"wandb.entity={EXPECTED_ENTITY}",
        f"wandb.project={EXPECTED_PROJECT}",
        (
            "wandb.tags=[abc,dual-video-diffusion,evaluation-only,"
            f"privileged-video,{arm_label},parseval_rfft,seed-1234]"
        ),
        f"+wandb.id={eval_id}",
        "+wandb.group=null",
        "+wandb.resume=never",
        f"+privileged_video_evaluation.parent_arm={arm['parent_arm']}",
        (
            "+privileged_video_evaluation.snapshot_path="
            f"{parent_root / 'snapshot.pt'}"
        ),
        (
            "+privileged_video_evaluation.snapshot_sha256="
            f"{arm['snapshot_sha256']}"
        ),
        (
            "+privileged_video_evaluation.parent_run_identity_sha256="
            f"{arm['parent_identity_sha256']}"
        ),
        (
            "+privileged_video_evaluation.parent_completed_updates="
            f"{PARENT_COMPLETED_UPDATES}"
        ),
        f"+privileged_video_evaluation.viz_skip_batches={VIZ_SKIP_BATCHES}",
        (
            "+privileged_video_evaluation.artifact_iteration="
            f"{ARTIFACT_ITERATION}"
        ),
    ]


def _validate_source(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    repo = pilot._canonical_directory(str(repo_root), "repository root")
    commit = pilot._validated_commit(expected_commit)
    if commit == CORE_COMMIT:
        raise RuntimeError(
            "privileged evaluation requires a new immutable source commit"
        )
    pilot._assert_clean_commit(repo, commit)
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", CORE_COMMIT, commit],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"launch commit {commit} is not based on privileged core {CORE_COMMIT}"
        )
    files: dict[str, Any] = {}
    for label, relative in (
        (
            "evaluator",
            "projects/latent_action_models/evaluate_privileged_video.py",
        ),
        ("launcher", "tools/privileged_video_eval_launch.py"),
        ("slurm_entrypoint", "tools/slurm/privileged_video_eval.sbatch"),
        ("submission_entrypoint", "tools/slurm/submit_privileged_video_eval.sh"),
    ):
        path = pilot._canonical_regular_file(
            str(repo / relative), f"privileged {label}"
        )
        files[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": pilot._sha256(path),
        }
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "required_ancestor_commit": CORE_COMMIT,
        "new_commit_required": True,
        "clean": True,
        "files": files,
    }


def _validate_output_scope(
    output_root: Path,
    run_root: Path,
    eval_id: str,
) -> tuple[Path, Path]:
    run = pilot._canonical_directory(str(run_root), "evaluation run root")
    out = pilot._canonical_directory(str(output_root), "evaluation output root")
    expected = run / eval_id
    if out != expected:
        raise RuntimeError(f"evaluation output must be exactly {expected}, got {out}")
    if out.is_symlink() or run.is_symlink():
        raise RuntimeError("evaluation output and run root must not be symlinks")
    return out, run


def _validate_parent(arm: dict[str, Any]) -> dict[str, Any]:
    root = _parent_root(arm)
    checkpoint = stage._check_sha(
        root / "snapshot.pt",
        str(arm["snapshot_sha256"]),
        f"{arm['label']} checkpoint",
    )
    if checkpoint["bytes"] != SNAPSHOT_BYTES:
        raise RuntimeError(
            f"{arm['label']} checkpoint size changed: "
            f"{checkpoint['bytes']} != {SNAPSHOT_BYTES}"
        )
    parent_paths = {
        "resolved_config": root / "resolved_config.yaml",
        "arm_manifest": root / "arm_manifest.json",
        "outcome": root / "outcome.json",
        "training_completion": root / "training_complete.json",
    }
    files = {
        name: stage._check_sha(
            parent_paths[name],
            arm["file_sha256"][name],
            f"{arm['label']} parent {name}",
        )
        for name in parent_paths
    }
    parent_manifest = stage._read_json(
        parent_paths["arm_manifest"], f"{arm['label']} parent manifest"
    )
    if (
        not stage._identity_is_valid(parent_manifest)
        or parent_manifest.get("identity_sha256")
        != arm["parent_identity_sha256"]
    ):
        raise RuntimeError(f"{arm['label']} parent identity is invalid")
    expected_parent = {
        "arm": arm["parent_arm"],
        "condition_on_tf": arm["condition_on_tf"],
        "condition_mode": arm["condition_mode"],
        "representation": "parseval_rfft",
        "state_gate_init": arm["state_gate_init"],
        "state_gate_trainable": False,
    }
    if any(
        parent_manifest.get(key) != value
        for key, value in expected_parent.items()
    ):
        raise RuntimeError(f"{arm['label']} parent contract changed")
    if parent_manifest.get("run", {}).get("world_size") != WORLD_SIZE:
        raise RuntimeError(f"{arm['label']} parent world size is not 8")
    if (
        "model.dual_diffusion.tf_loss_weight=1.0"
        not in parent_manifest.get("config", {}).get("hydra_overrides", [])
    ):
        raise RuntimeError(f"{arm['label']} parent TF loss weight is not 1.0")
    outcome = stage._read_json(
        parent_paths["outcome"], f"{arm['label']} parent outcome"
    )
    if outcome.get("completed") is not True or outcome.get("exit_status") != 0:
        raise RuntimeError(f"{arm['label']} parent outcome is not successful")
    completion = stage._read_json(
        parent_paths["training_completion"],
        f"{arm['label']} parent completion",
    )
    expected_completion = {
        "status": "completed",
        "max_iter": PARENT_COMPLETED_UPDATES,
        "completed_updates": PARENT_COMPLETED_UPDATES,
        "run_identity_sha256": arm["parent_identity_sha256"],
    }
    if any(
        completion.get(key) != value
        for key, value in expected_completion.items()
    ):
        raise RuntimeError(f"{arm['label']} parent completion contract changed")
    return {
        "label": arm["label"],
        "arm": arm["parent_arm"],
        "root": str(root),
        "run_identity_sha256": arm["parent_identity_sha256"],
        "checkpoint": checkpoint,
        "completed_updates": PARENT_COMPLETED_UPDATES,
        "training_condition": {
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
            "tf_loss_weight": 1.0,
        },
        "files": files,
    }


def command_arm_contract(args: argparse.Namespace) -> int:
    arm = _arm_by_task(args.array_task_id)
    eval_id = expected_eval_id(args.git_commit, arm["label"])
    payload = {
        "array_task_id": arm["array_task_id"],
        "label": arm["label"],
        "parent_arm": arm["parent_arm"],
        "evaluation_id": eval_id,
    }
    if args.format == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "\t".join(
                str(payload[key])
                for key in (
                    "label",
                    "parent_arm",
                    "evaluation_id",
                )
            )
        )
    return 0


def command_hydra_overrides(args: argparse.Namespace) -> int:
    for override in expected_hydra_overrides(
        args.arm_label, args.eval_id, Path(args.output_root)
    ):
        print(override)
    return 0


def command_wandb_private(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            stage._wandb_private_project(args.entity, args.project),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_create_manifest(args: argparse.Namespace) -> int:
    commit = pilot._validated_commit(args.git_commit)
    arm = _arm_by_label(args.arm_label)
    eval_id = pilot._validated_id(args.eval_id, "evaluation ID")
    expected_id = expected_eval_id(commit, arm["label"])
    if eval_id != expected_id:
        raise RuntimeError(f"evaluation ID must be {expected_id}, got {eval_id}")
    out, run = _validate_output_scope(
        Path(args.output_root), Path(args.run_root), eval_id
    )
    if any(out.iterdir()):
        raise RuntimeError("fresh privileged evaluation output root must be empty")
    source = _validate_source(Path(args.repo_root), commit)
    parent = _validate_parent(arm)
    data = stage._check_sha(
        ABC_MANIFEST, ABC_MANIFEST_SHA256, "ABC manifest"
    )
    wandb = stage._wandb_private_project(EXPECTED_ENTITY, EXPECTED_PROJECT)
    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "privileged_video_posthoc_evaluation_manifest",
            "created_at_utc": pilot._now(),
            "evaluation_id": eval_id,
            "array_task_id": arm["array_task_id"],
            "arm_label": arm["label"],
            "source": source,
            "output": {
                "run_root": str(run),
                "output_root": str(out),
                "fresh_at_manifest_creation": True,
                "snapshot_sentinel": str(out / "_never_write_snapshot.pt"),
                "training_completion_forbidden": str(
                    out / "training_complete.json"
                ),
            },
            "parent": parent,
            "data": {"root": str(DATA_ROOT), "abc_manifest": data},
            "runtime": {
                "python": str(PYTHON_BIN),
                "python_symlink_target": str(PYTHON_LINK_TARGET),
                "python_canonical_executable": str(PYTHON_REAL_BIN),
                "wan_dir": str(WAN_DIR),
                "videox_home": str(VIDEOX_HOME),
                "nodes_per_array_task": 1,
                "world_size_per_array_task": WORLD_SIZE,
                "gpu_type": "B200",
            },
            "evaluation": {
                "evaluation_only": True,
                "training_steps": 0,
                "seed": 1234,
                "artifact_iteration": ARTIFACT_ITERATION,
                "viz_skip_batches": VIZ_SKIP_BATCHES,
                "nfe_steps": EVALUATION_NFE_STEPS,
                "noise_seed": 20260726,
                "condition_sources": EVALUATION_CONDITION_SOURCES,
                "condition_source_codes": [0, 1],
                "schedule_mode": "aligned",
                "tf_content_disabled": True,
                "tf_clock_disabled": True,
                "cascade_stage_faithful_inference": False,
                "sigma_convention": "sigma=1 noise, sigma=0 clean",
                "hydra_overrides": expected_hydra_overrides(
                    arm["label"], eval_id, out
                ),
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
                "array": True,
                "array_range": [0, 2],
                "array_specification": "0-2%1",
                "max_concurrent_tasks": 1,
                "array_task_id": arm["array_task_id"],
                "nodes_per_task": 1,
                "gpus_per_node": WORLD_SIZE,
                "cpus_per_task": 64,
                "memory": "600G",
                "time_limit": "00:30:00",
                "partition": "batch",
                "account": EXPECTED_SLURM_ACCOUNT,
                "qos": EXPECTED_SLURM_QOS,
                "requeue": False,
                "active_job_coexistence": stage._active_job_contract(
                    args.allow_active_job_id, args.observed_active_job
                ),
            },
        }
    )
    pilot._exclusive_json(Path(args.output), payload)
    print(payload["identity_sha256"])
    return 0


def command_validate_manifest(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(
        args.manifest, "privileged evaluation manifest"
    )
    manifest = stage._signed_json(
        manifest_path, "privileged evaluation manifest"
    )
    commit = pilot._validated_commit(args.git_commit)
    arm = _arm_by_label(args.arm_label)
    eval_id = expected_eval_id(commit, arm["label"])
    expected_fields = {
        "kind": "privileged_video_posthoc_evaluation_manifest",
        "evaluation_id": eval_id,
        "array_task_id": arm["array_task_id"],
        "arm_label": arm["label"],
    }
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        raise RuntimeError("privileged evaluation manifest identity changed")
    if manifest.get("source", {}).get("git_commit") != commit:
        raise RuntimeError("privileged evaluation source commit changed")
    out, _ = _validate_output_scope(
        Path(manifest["output"]["output_root"]),
        Path(manifest["output"]["run_root"]),
        eval_id,
    )
    expected_overrides = expected_hydra_overrides(arm["label"], eval_id, out)
    if manifest.get("evaluation", {}).get(
        "hydra_overrides"
    ) != expected_overrides:
        raise RuntimeError("privileged evaluation Hydra vector changed")
    expected_evaluation = {
        "evaluation_only": True,
        "training_steps": 0,
        "seed": 1234,
        "artifact_iteration": ARTIFACT_ITERATION,
        "viz_skip_batches": VIZ_SKIP_BATCHES,
        "nfe_steps": EVALUATION_NFE_STEPS,
        "noise_seed": 20260726,
        "condition_sources": EVALUATION_CONDITION_SOURCES,
        "condition_source_codes": [0, 1],
        "schedule_mode": "aligned",
        "tf_content_disabled": True,
        "tf_clock_disabled": True,
        "cascade_stage_faithful_inference": False,
        "sigma_convention": "sigma=1 noise, sigma=0 clean",
    }
    evaluation = manifest.get("evaluation", {})
    if any(
        evaluation.get(key) != value
        for key, value in expected_evaluation.items()
    ):
        raise RuntimeError("privileged evaluation intervention changed")
    expected_slurm = {
        "array": True,
        "array_range": [0, 2],
        "array_specification": "0-2%1",
        "max_concurrent_tasks": 1,
        "array_task_id": arm["array_task_id"],
        "nodes_per_task": 1,
        "gpus_per_node": WORLD_SIZE,
        "cpus_per_task": 64,
        "memory": "600G",
        "time_limit": "00:30:00",
        "partition": "batch",
        "account": EXPECTED_SLURM_ACCOUNT,
        "qos": EXPECTED_SLURM_QOS,
        "requeue": False,
    }
    slurm = manifest.get("slurm", {})
    if any(slurm.get(key) != value for key, value in expected_slurm.items()):
        raise RuntimeError("privileged evaluation Slurm contract changed")
    expected_runtime = {
        "python": str(PYTHON_BIN),
        "python_symlink_target": str(PYTHON_LINK_TARGET),
        "python_canonical_executable": str(PYTHON_REAL_BIN),
        "wan_dir": str(WAN_DIR),
        "videox_home": str(VIDEOX_HOME),
        "nodes_per_array_task": 1,
        "world_size_per_array_task": WORLD_SIZE,
        "gpu_type": "B200",
    }
    if manifest.get("runtime") != expected_runtime:
        raise RuntimeError("privileged evaluation runtime contract changed")
    source = _validate_source(Path(args.repo_root), commit)
    if manifest.get("source") != source:
        raise RuntimeError("privileged evaluation source provenance changed")
    parent = _validate_parent(arm)
    if manifest.get("parent") != parent:
        raise RuntimeError("privileged evaluation parent provenance changed")
    data = stage._check_sha(
        ABC_MANIFEST, ABC_MANIFEST_SHA256, "ABC manifest"
    )
    if manifest.get("data") != {
        "root": str(DATA_ROOT),
        "abc_manifest": data,
    }:
        raise RuntimeError("privileged evaluation data provenance changed")
    print(manifest["identity_sha256"])
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(
        args.manifest, "privileged evaluation manifest"
    )
    manifest = stage._signed_json(
        manifest_path, "privileged evaluation manifest"
    )
    expected_slurm = {
        "array": True,
        "array_range": [0, 2],
        "array_specification": "0-2%1",
        "max_concurrent_tasks": 1,
    }
    if any(
        manifest.get("slurm", {}).get(key) != value
        for key, value in expected_slurm.items()
    ):
        raise RuntimeError("privileged evaluation array contract changed")
    if NUMERIC_JOB_ID_RE.fullmatch(args.array_job_id) is None:
        raise ValueError("Slurm array job ID must be one positive numeric base ID")
    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "privileged_video_posthoc_evaluation_submission",
            "created_at_utc": pilot._now(),
            "manifest": {
                "path": str(manifest_path),
                "sha256": pilot._sha256(manifest_path),
                "identity_sha256": manifest["identity_sha256"],
            },
            "array_job_id": args.array_job_id,
            "array_task_id": manifest["array_task_id"],
            "array_specification": "0-2%1",
            "max_concurrent_tasks": 1,
            "immediate_pre_sbatch_active_job_coexistence": (
                stage._active_job_contract(
                    manifest["slurm"]["active_job_coexistence"][
                        "allowed_array_or_job_ids"
                    ],
                    args.observed_active_job,
                )
            ),
        }
    )
    pilot._exclusive_json(Path(args.output), payload)
    return 0


def command_complete(args: argparse.Namespace) -> int:
    manifest_path = pilot._canonical_regular_file(
        args.manifest, "privileged evaluation manifest"
    )
    manifest = stage._signed_json(
        manifest_path, "privileged evaluation manifest"
    )
    output_root = Path(manifest["output"]["output_root"])
    inventory_payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": (
                "privileged_video_posthoc_evaluation_artifact_inventory"
            ),
            "created_at_utc": pilot._now(),
            "evaluation_id": manifest["evaluation_id"],
            "array_task_id": manifest["array_task_id"],
            "arm_label": manifest["arm_label"],
            "manifest": {
                "path": str(manifest_path),
                "sha256": pilot._sha256(manifest_path),
                "identity_sha256": manifest["identity_sha256"],
            },
            **stage._artifact_inventory(output_root),
        }
    )
    inventory_path = Path(args.inventory_output)
    pilot._exclusive_json(inventory_path, inventory_payload)
    completion_payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "privileged_video_posthoc_evaluation_completion",
            "status": "completed",
            "created_at_utc": pilot._now(),
            "evaluation_id": manifest["evaluation_id"],
            "array_task_id": manifest["array_task_id"],
            "arm_label": manifest["arm_label"],
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
    manifest_path = pilot._canonical_regular_file(
        args.manifest, "privileged evaluation manifest"
    )
    manifest = stage._signed_json(
        manifest_path, "privileged evaluation manifest"
    )
    if NUMERIC_JOB_ID_RE.fullmatch(args.array_job_id) is None:
        raise ValueError("Slurm array job ID must be one positive numeric base ID")
    if NUMERIC_JOB_ID_RE.fullmatch(args.task_job_id) is None:
        raise ValueError("Slurm task job ID must be one positive numeric ID")
    if args.array_task_id != manifest["array_task_id"]:
        raise RuntimeError("Slurm array task ID differs from the manifest")
    exit_status = int(args.exit_status)
    completed = exit_status == 0
    completion: dict[str, Any] | None = None
    if completed:
        completion_path = pilot._canonical_regular_file(
            args.completion, "privileged evaluation completion"
        )
        completion_payload = stage._signed_json(
            completion_path, "privileged evaluation completion"
        )
        expected_completion = {
            "kind": "privileged_video_posthoc_evaluation_completion",
            "status": "completed",
            "evaluation_id": manifest["evaluation_id"],
            "array_task_id": manifest["array_task_id"],
        }
        if any(
            completion_payload.get(key) != value
            for key, value in expected_completion.items()
        ):
            raise RuntimeError("privileged evaluation completion changed")
        completion = {
            "path": str(completion_path),
            "sha256": pilot._sha256(completion_path),
            "identity_sha256": completion_payload["identity_sha256"],
        }
    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "privileged_video_posthoc_evaluation_outcome",
            "created_at_utc": pilot._now(),
            "completed": completed,
            "exit_status": exit_status,
            "evaluation_id": manifest["evaluation_id"],
            "array_job_id": args.array_job_id,
            "array_task_id": args.array_task_id,
            "task_job_id": args.task_job_id,
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

    arm = subparsers.add_parser("arm-contract")
    arm.add_argument("--array-task-id", required=True, type=int)
    arm.add_argument("--git-commit", required=True)
    arm.add_argument("--format", choices=("json", "tsv"), default="json")
    arm.set_defaults(func=command_arm_contract)

    hydra = subparsers.add_parser("hydra-overrides")
    hydra.add_argument("--arm-label", required=True)
    hydra.add_argument("--eval-id", required=True)
    hydra.add_argument("--output-root", required=True)
    hydra.set_defaults(func=command_hydra_overrides)

    wandb = subparsers.add_parser("wandb-private")
    wandb.add_argument("--entity", required=True)
    wandb.add_argument("--project", required=True)
    wandb.set_defaults(func=command_wandb_private)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--arm-label", required=True)
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
    validate.add_argument("--arm-label", required=True)
    validate.add_argument("--git-commit", required=True)
    validate.add_argument("--repo-root", required=True)
    validate.set_defaults(func=command_validate_manifest)

    submission = subparsers.add_parser("record-submission")
    submission.add_argument("--manifest", required=True)
    submission.add_argument("--array-job-id", required=True)
    submission.add_argument(
        "--observed-active-job", action="append", default=[]
    )
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
    outcome.add_argument("--array-job-id", required=True)
    outcome.add_argument("--array-task-id", required=True, type=int)
    outcome.add_argument("--task-job-id", required=True)
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
