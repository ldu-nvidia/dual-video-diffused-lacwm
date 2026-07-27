"""No-update, same-checkpoint evaluation for the strict TF-first cascade.

This entrypoint intentionally reuses ``train._setup`` so model, DDP, data, and
W&B construction match training. It never enters ``Trainer.train`` and never
writes a snapshot or training-completion marker.

LACWM clock convention: sigma=1 is Gaussian noise and sigma=0 is clean data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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

logger = logging.getLogger(__name__)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VIZ_SKIP_BATCHES = 4
ARTIFACT_ITERATION = 199
PARENT_COMPLETED_UPDATES = 200
EVALUATION_CONDITION_SOURCES = (
    "autonomous",
    "autonomous_shuffled",
    "autonomous_legacy",
    "off",
)
SNAPSHOT_SENTINEL_NAME = "_never_write_snapshot.pt"
PROVENANCE_NAME = "stage_faithful_evaluation_provenance.json"


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_input_file(value: Any, label: str) -> Path:
    path = Path(str(value)).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    return resolved


def _output_path(value: Any, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=False)
    if path.exists():
        raise RuntimeError(f"{label} must not already exist: {path}")
    return path


def _require_exact_integer(value: Any, expected: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RuntimeError(f"{label} must be the integer {expected}, got {value!r}")
    return value


def _validate_config(cfg: DictConfig) -> dict[str, Any]:
    evaluation = cfg.get("stage_faithful_evaluation")
    if evaluation is None:
        raise RuntimeError("stage_faithful_evaluation config is required")

    snapshot_path = _canonical_input_file(
        evaluation.get("snapshot_path"),
        "stage-faithful parent snapshot",
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
            f"evaluation training-completion path already exists: {completion_path}"
        )
    viz_path = Path(
        str(trainer_config.visualization.viz_path)
    ).expanduser().resolve(strict=False)
    artifact_root = viz_path / f"iter_{ARTIFACT_ITERATION}"
    if artifact_root.exists():
        raise RuntimeError(
            f"stage-faithful artifact root already exists: {artifact_root}"
        )
    provenance_path = save_path.parent / PROVENANCE_NAME
    if provenance_path.exists():
        raise RuntimeError(
            f"stage-faithful provenance already exists: {provenance_path}"
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
        "condition_on_tf": True,
        "condition_mode": "matched",
        "schedule_mode": "tf_first_cascaded",
        "cascade_stage_faithful_inference": True,
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
    if nfe_steps != (2, 4, 8):
        raise RuntimeError(
            f"evaluation_nfe_steps must be exactly [2, 4, 8], got {nfe_steps}"
        )
    if not bool(trainer_config.visualization.get("require_success", False)):
        raise RuntimeError("visualization.require_success must be true")

    return {
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


def _verified_snapshot_sha256(
    path: Path,
    expected_sha256: str,
    *,
    global_rank: int,
) -> str:
    """Hash once on rank zero and broadcast the result to every evaluator."""
    result = [None]
    if global_rank == 0:
        try:
            actual = _sha256_file(path)
            error = (
                None
                if actual == expected_sha256
                else f"snapshot SHA-256 mismatch: {actual} != {expected_sha256}"
            )
            result[0] = {"actual": actual, "error": error}
        except Exception as exc:
            result[0] = {
                "actual": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(result, src=0)
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank zero did not broadcast snapshot verification")
    if payload.get("error") is not None:
        raise RuntimeError(str(payload["error"]))
    actual = payload.get("actual")
    if not isinstance(actual, str) or actual != expected_sha256:
        raise RuntimeError("snapshot verification returned an invalid digest")
    return actual


def _strict_load_parent_snapshot(
    trainer,
    *,
    snapshot_path: Path,
    snapshot_sha256: str,
    parent_run_identity_sha256: str,
) -> dict[str, Any]:
    actual_sha256 = _verified_snapshot_sha256(
        snapshot_path,
        snapshot_sha256,
        global_rank=int(trainer.global_rank),
    )
    try:
        parameter = next(trainer.model.module.parameters())
        map_location = parameter.device
    except StopIteration:
        map_location = "cpu"
    snapshot = torch.load(
        snapshot_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(snapshot, dict):
        raise RuntimeError("parent snapshot must be a dictionary")
    if snapshot.get("snapshot_schema_version") != 3:
        raise RuntimeError("parent snapshot_schema_version must be 3")
    if snapshot.get("_start_iter") != PARENT_COMPLETED_UPDATES:
        raise RuntimeError(
            "parent snapshot must begin after exactly 200 completed updates"
        )
    checkpoint_identity = snapshot.get("run_identity_sha256")
    if checkpoint_identity != parent_run_identity_sha256:
        raise RuntimeError(
            "parent snapshot run identity mismatch: "
            f"{checkpoint_identity!r} != {parent_run_identity_sha256!r}"
        )
    state_dict = snapshot.get("model")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("parent snapshot model state is missing or empty")
    incompatible = trainer.model.module.load_state_dict(
        state_dict,
        strict=True,
    )
    missing = list(getattr(incompatible, "missing_keys", ()))
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            "strict parent model load was incomplete: "
            f"missing={missing}, unexpected={unexpected}"
        )
    parent_total_observations = snapshot.get("_total_observations")
    if (
        isinstance(parent_total_observations, bool)
        or not isinstance(parent_total_observations, int)
        or parent_total_observations < 0
    ):
        raise RuntimeError(
            "parent snapshot _total_observations must be nonnegative"
        )
    return {
        "snapshot_sha256": actual_sha256,
        "parent_total_observations": parent_total_observations,
    }


def _advance_visualization_iterators(trainer, skip_batches: int) -> None:
    iterators = trainer._viz_data_loader_iters
    loaders = trainer.viz_data_loaders
    if (
        iterators is None
        or loaders is None
        or len(iterators) == 0
        or len(iterators) != len(loaders)
    ):
        raise RuntimeError(
            "visualization iterators must be started and align with loaders"
        )
    for index, iterator in enumerate(iterators):
        try:
            for _ in range(skip_batches):
                next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"visualization iterator {index} ended before {skip_batches} skips"
            ) from exc


def _update_wandb_provenance(trainer, contract, snapshot_info) -> None:
    if not trainer.use_wandb:
        return
    import wandb

    wandb.summary["evaluation_only"] = True
    wandb.summary["evaluation_optimizer_updates"] = 0
    wandb.summary["evaluation_total_observations"] = 0
    wandb.summary["stage_faithful_artifact_iteration"] = ARTIFACT_ITERATION
    wandb.summary["stage_faithful_viz_skip_batches"] = VIZ_SKIP_BATCHES
    wandb.summary["parent_completed_updates"] = PARENT_COMPLETED_UPDATES
    wandb.summary["parent_total_observations"] = snapshot_info[
        "parent_total_observations"
    ]
    wandb.summary["parent_run_identity_sha256"] = contract[
        "parent_run_identity_sha256"
    ]
    wandb.summary["parent_snapshot_sha256"] = contract["snapshot_sha256"]


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_provenance(trainer, contract, snapshot_info) -> None:
    if int(trainer.global_rank) == 0:
        _exclusive_json(
            contract["provenance_path"],
            {
                "schema_version": 1,
                "kind": "dual_video_diffusion_stage_faithful_evaluation",
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
                "parent": {
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


def _assert_no_training_outputs(contract) -> None:
    forbidden = (contract["save_path"], contract["completion_path"])
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            f"evaluation unexpectedly wrote training outputs: {existing}"
        )


def run_stage_faithful_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Strict-load one completed checkpoint and render its fifth viz batch."""
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
        run_stage_faithful_evaluation(cfg)
    except Exception:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
