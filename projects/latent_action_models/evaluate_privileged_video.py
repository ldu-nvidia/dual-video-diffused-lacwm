"""No-update TF-free video evaluation of compatible 200-update parents.

This evaluator compares what the video backbone learned under different
training interventions. At inference, both TF-state content and the separate
TF sigma-clock residual are disabled, and every NFE advances the native video
schedule. The auxiliary TF state remains available only as a diagnostic.

LACWM clock convention: sigma=1 is Gaussian noise and sigma=0 is clean data.
"""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from custom_resolvers import *  # noqa: F403 - register training config resolvers
from omegaconf import DictConfig

import train as train_entrypoint
try:
    from evaluate_stage_faithful import (
        ARTIFACT_ITERATION,
        PARENT_COMPLETED_UPDATES,
        SHA256_RE,
        SNAPSHOT_SENTINEL_NAME,
        VIZ_SKIP_BATCHES,
        _advance_visualization_iterators,
        _assert_no_training_outputs,
        _canonical_input_file,
        _exclusive_json,
        _output_path,
        _require_exact_integer,
        _strict_load_parent_snapshot,
    )
except ModuleNotFoundError:
    # Importing by file path in unit tests does not add this script's directory
    # to sys.path; the namespace-package path is the equivalent fallback.
    from projects.latent_action_models.evaluate_stage_faithful import (
        ARTIFACT_ITERATION,
        PARENT_COMPLETED_UPDATES,
        SHA256_RE,
        SNAPSHOT_SENTINEL_NAME,
        VIZ_SKIP_BATCHES,
        _advance_visualization_iterators,
        _assert_no_training_outputs,
        _canonical_input_file,
        _exclusive_json,
        _output_path,
        _require_exact_integer,
        _strict_load_parent_snapshot,
    )

logger = logging.getLogger(__name__)

EVALUATION_CONDITION_SOURCES = ("autonomous", "off")
EVALUATION_NFE_STEPS = (1, 2, 4, 8)
PROVENANCE_NAME = "privileged_video_evaluation_provenance.json"


def _required_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} must be one nonempty string")
    return value.strip()


def _validate_config(cfg: DictConfig) -> dict[str, Any]:
    evaluation = cfg.get("privileged_video_evaluation")
    if evaluation is None:
        raise RuntimeError("privileged_video_evaluation config is required")

    parent_arm = _required_nonempty_string(
        evaluation.get("parent_arm"),
        "parent_arm",
    )
    snapshot_path = _canonical_input_file(
        evaluation.get("snapshot_path"),
        "privileged-video parent snapshot",
    )
    snapshot_sha256 = str(evaluation.get("snapshot_sha256", ""))
    parent_identity = str(
        evaluation.get("parent_run_identity_sha256", "")
    )
    if SHA256_RE.fullmatch(snapshot_sha256) is None:
        raise RuntimeError("snapshot_sha256 must be one lowercase SHA-256")
    if SHA256_RE.fullmatch(parent_identity) is None:
        raise RuntimeError(
            "parent_run_identity_sha256 must be one lowercase SHA-256"
        )
    _require_exact_integer(
        evaluation.get("parent_completed_updates"),
        PARENT_COMPLETED_UPDATES,
        "parent_completed_updates",
    )
    _require_exact_integer(
        evaluation.get("viz_skip_batches"),
        VIZ_SKIP_BATCHES,
        "viz_skip_batches",
    )
    _require_exact_integer(
        evaluation.get("artifact_iteration"),
        ARTIFACT_ITERATION,
        "artifact_iteration",
    )

    trainer_config = cfg.trainer.config
    if trainer_config.get("load_path") is not None:
        raise RuntimeError(
            "trainer.config.load_path must be null; the evaluator performs "
            "one explicit strict model load"
        )
    exclude_keys = trainer_config.get("exclude_keys")
    if exclude_keys is None or list(exclude_keys) != []:
        raise RuntimeError("trainer.config.exclude_keys must be exactly []")
    if bool(trainer_config.get("share_spatial_attention", False)):
        raise RuntimeError(
            "trainer.config.share_spatial_attention must be false"
        )
    if trainer_config.get("transition_handoff_path") is not None:
        raise RuntimeError(
            "trainer.config.transition_handoff_path must be null"
        )

    save_path = _output_path(
        trainer_config.saving.save_path,
        "evaluation snapshot sentinel",
    )
    if save_path.name != SNAPSHOT_SENTINEL_NAME:
        raise RuntimeError(
            "trainer.config.saving.save_path must end in "
            f"{SNAPSHOT_SENTINEL_NAME!r}"
        )
    completion_path = save_path.parent / "training_complete.json"
    if completion_path.exists():
        raise RuntimeError(
            f"evaluation training-completion path already exists: "
            f"{completion_path}"
        )
    viz_path = Path(
        str(trainer_config.visualization.viz_path)
    ).expanduser().resolve(strict=False)
    artifact_root = viz_path / f"iter_{ARTIFACT_ITERATION}"
    if artifact_root.exists():
        raise RuntimeError(
            f"privileged-video artifact root already exists: {artifact_root}"
        )
    provenance_path = save_path.parent / PROVENANCE_NAME
    if provenance_path.exists():
        raise RuntimeError(
            f"privileged-video provenance already exists: {provenance_path}"
        )
    if viz_path != save_path.parent / "visualization":
        raise RuntimeError(
            "visualization path must be the evaluation output's "
            "'visualization' directory"
        )
    for label, output in (
        ("snapshot sentinel", save_path),
        ("visualization", viz_path),
    ):
        if output.is_relative_to(snapshot_path.parent):
            raise RuntimeError(
                f"evaluation {label} must be outside the parent run directory"
            )

    dual = cfg.model.dual_diffusion
    expected_dual_values = {
        "enabled": True,
        "condition_on_tf": False,
        "condition_mode": "off",
        "schedule_mode": "aligned",
        "evaluation_disable_tf_clock": True,
    }
    for key, expected in expected_dual_values.items():
        actual = dual.get(key)
        if actual != expected:
            raise RuntimeError(
                f"model.dual_diffusion.{key} must be {expected!r}, "
                f"got {actual!r}"
            )
    sources = tuple(str(value) for value in dual.evaluation_condition_sources)
    if sources != EVALUATION_CONDITION_SOURCES:
        raise RuntimeError(
            "evaluation_condition_sources must be exactly "
            f"{list(EVALUATION_CONDITION_SOURCES)}, got {list(sources)}"
        )
    nfe_steps = tuple(int(value) for value in dual.evaluation_nfe_steps)
    if nfe_steps != EVALUATION_NFE_STEPS:
        raise RuntimeError(
            "evaluation_nfe_steps must be exactly "
            f"{list(EVALUATION_NFE_STEPS)}, got {list(nfe_steps)}"
        )
    if not bool(trainer_config.visualization.get("require_success", False)):
        raise RuntimeError("visualization.require_success must be true")

    return {
        "parent_arm": parent_arm,
        "snapshot_path": snapshot_path,
        "snapshot_sha256": snapshot_sha256,
        "parent_run_identity_sha256": parent_identity,
        "parent_completed_updates": PARENT_COMPLETED_UPDATES,
        "save_path": save_path,
        "completion_path": completion_path,
        "viz_path": viz_path,
        "artifact_root": artifact_root,
        "provenance_path": provenance_path,
    }


def _update_wandb_provenance(trainer, contract, snapshot_info) -> None:
    if not trainer.use_wandb:
        return
    import wandb

    wandb.summary["evaluation_only"] = True
    wandb.summary["evaluation_optimizer_updates"] = 0
    wandb.summary["evaluation_total_observations"] = 0
    wandb.summary["privileged_video_artifact_iteration"] = ARTIFACT_ITERATION
    wandb.summary["privileged_video_viz_skip_batches"] = VIZ_SKIP_BATCHES
    wandb.summary["privileged_video_tf_content_disabled"] = True
    wandb.summary["privileged_video_tf_clock_disabled"] = True
    wandb.summary["privileged_video_parent_arm"] = contract["parent_arm"]
    wandb.summary["parent_completed_updates"] = PARENT_COMPLETED_UPDATES
    wandb.summary["parent_total_observations"] = snapshot_info[
        "parent_total_observations"
    ]
    wandb.summary["parent_run_identity_sha256"] = contract[
        "parent_run_identity_sha256"
    ]
    wandb.summary["parent_snapshot_sha256"] = contract["snapshot_sha256"]


def _write_provenance(trainer, contract, snapshot_info) -> None:
    if int(trainer.global_rank) == 0:
        _exclusive_json(
            contract["provenance_path"],
            {
                "schema_version": 1,
                "kind": "dual_video_diffusion_privileged_video_evaluation",
                "status": "visualization_completed",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_only": True,
                "evaluation_optimizer_updates": 0,
                "evaluation_total_observations": 0,
                "artifact_iteration": ARTIFACT_ITERATION,
                "viz_skip_batches": VIZ_SKIP_BATCHES,
                "evaluation_condition_sources": list(
                    EVALUATION_CONDITION_SOURCES
                ),
                "evaluation_nfe_steps": list(EVALUATION_NFE_STEPS),
                "runtime_intervention": {
                    "schedule_mode": "aligned",
                    "tf_content_disabled": True,
                    "tf_clock_disabled": True,
                    "all_model_calls_advance_video": True,
                },
                "parent": {
                    "arm": contract["parent_arm"],
                    "snapshot": str(contract["snapshot_path"]),
                    "snapshot_sha256": snapshot_info["snapshot_sha256"],
                    "run_identity_sha256": contract[
                        "parent_run_identity_sha256"
                    ],
                    "completed_updates": PARENT_COMPLETED_UPDATES,
                    "total_observations": snapshot_info[
                        "parent_total_observations"
                    ],
                },
                "artifact_root": str(contract["artifact_root"]),
                "snapshot_written": False,
                "training_completion_written": False,
            },
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_privileged_video_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Strict-load one parent and render its fifth batch without training."""
    contract = _validate_config(cfg)
    trainer = None
    try:
        trainer = train_entrypoint._setup(cfg)
        if trainer.resumed or trainer.transitioned or trainer._start_iter != 0:
            raise RuntimeError(
                "evaluation trainer must start fresh without resume/transition"
            )
        if trainer.metrics.total_observations != 0:
            raise RuntimeError(
                "evaluation W&B step must start at zero observations"
            )
        if Path(trainer.save_path).resolve(strict=False) != contract["save_path"]:
            raise RuntimeError("instantiated trainer save path changed")
        if (
            Path(trainer.completion_path).resolve(strict=False)
            != contract["completion_path"]
        ):
            raise RuntimeError("instantiated trainer completion path changed")

        snapshot_info = _strict_load_parent_snapshot(
            trainer,
            snapshot_path=contract["snapshot_path"],
            snapshot_sha256=contract["snapshot_sha256"],
            parent_run_identity_sha256=contract[
                "parent_run_identity_sha256"
            ],
        )
        _update_wandb_provenance(trainer, contract, snapshot_info)
        _advance_visualization_iterators(trainer, VIZ_SKIP_BATCHES)
        trainer._curr_iter = ARTIFACT_ITERATION
        trainer._viz()

        if not contract["artifact_root"].is_dir():
            raise RuntimeError(
                "visualization did not create the required iter_199 artifact root"
            )
        if trainer.metrics.total_observations != 0:
            raise RuntimeError(
                "evaluation changed total observations despite zero updates"
            )
        _assert_no_training_outputs(contract)
        _write_provenance(trainer, contract, snapshot_info)
        _assert_no_training_outputs(contract)
        return {
            "parent_arm": contract["parent_arm"],
            "artifact_root": str(contract["artifact_root"]),
            "provenance": str(contract["provenance_path"]),
            "snapshot_sha256": snapshot_info["snapshot_sha256"],
        }
    finally:
        if trainer is not None:
            train_entrypoint._teardown(trainer)
        elif train_entrypoint.dist.is_initialized():
            train_entrypoint.dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    torch.multiprocessing.set_start_method("spawn", force=True)
    try:
        run_privileged_video_evaluation(cfg)
    except Exception:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
