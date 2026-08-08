#!/usr/bin/env python3
"""Train and evaluate the preregistered ``AINC-OFF`` semantic screen.

The transformer architecture and parameter count are identical to the causal
V-JEPA ABS reference.  Its auxiliary clean state is instead the normalized
observed-anchor increment tensor ``Q``.  The anchor never enters the model; it
is used only after autonomous sampling to reconstruct absolute future semantic
states.  Thus this entry point tests the deterministic-skip hypothesis without
confounding it with an extra conditioning projection.

Only train and validation caches are accepted.  The deployable sampler has no
future RGB, clean semantic target, or oracle feature argument.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.video_latent_forcing import rows_episode_ids  # noqa: E402
from tools import causal_vjepa2_observed_anchor as anchor  # noqa: E402
from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402

RUN_SCHEMA = "causal-vjepa2-observed-anchor-training-v1"
EVALUATION_SCHEMA = "causal-vjepa2-observed-anchor-evaluation-v1"
DOE_ARM = "AINC-OFF"
CALIBRATION_UPDATES = 200
TRAIN_UPDATES = 5_000
TRAIN_CHECKPOINTS = (500, 1_000, 2_000, 5_000)
NFE_GRID = (1, 2, 4)
ACTION_OFFSETS = (1, 17, 101)
EVALUATION_SEED = screen.FROZEN_EVALUATION_SEED
FROZEN_WORLD_SIZE = 8
FROZEN_LOCAL_BATCH_SIZE = screen.FROZEN_GLOBAL_BATCH_SIZE // FROZEN_WORLD_SIZE
FROZEN_MICRO_BATCH_SIZE = 32
FROZEN_TRAIN_WORKERS = 4
FROZEN_EVAL_BATCH_SIZE = 8
FROZEN_EVAL_WORKERS = 2
TEMPORAL_SELECTION_SCHEMA = "causal-vjepa2-temporal-target-frozen-selection-v1"
TEMPORAL_ANALYSIS_SCHEMA = "causal-vjepa2-temporal-target-development-analysis-v1"
TEMPORAL_ARMS = ("ABS", "ABS-T", "DELTA", "DELTA-T", "DELTA-R")
TEMPORAL_CANDIDATE_ARMS = ("ABS-T", "DELTA", "DELTA-T", "DELTA-R")
TEMPORAL_SELECTION_NFE = (1, 2, 4)
TEMPORAL_AUTHORIZATION_COMMIT = (
    "f638b493bc5bf0fad0faa4283a9c56deb5a5f764"
)
EXECUTION_MODES = ("post-temporal-no-pass", "cheap-proxy-validity")
CONTROLS = (
    "autonomous",
    "anchor_static",
    "mean_increment",
    "donor_target",
    "context_shuffled",
    "history_shuffled",
    "actions_offset_1",
    "actions_offset_17",
    "actions_offset_101",
    "anchor_decode_shuffled",
    "zero",
    "oracle_clean",
)
GENERATED_CONTROLS = (
    "autonomous",
    "context_shuffled",
    "history_shuffled",
    "actions_offset_1",
    "actions_offset_17",
    "actions_offset_101",
)


class ObservedAnchorScreenError(RuntimeError):
    """The AINC-OFF training/evaluation contract failed closed."""


def _identity_is_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "identity_sha256"
    }
    return isinstance(identity, str) and identity == screen.sha256_json(unsigned)


def _execution_condition(args: argparse.Namespace) -> dict[str, Any]:
    """Attest the preregistered condition that permits the fallback family."""
    mode = str(args.execution_mode)
    selection_value = args.temporal_selection_record
    if mode == "cheap-proxy-validity":
        if selection_value is not None:
            raise ObservedAnchorScreenError(
                "cheap proxy mode cannot bind a temporal selection record"
            )
        return {
            "mode": mode,
            "temporal_primary_no_pass_attested": False,
            "proxy_validity_only": True,
            "semantic_screen_promotion_eligible": False,
            "video_quality_claim_eligible": False,
            "protected_test_eligible": False,
            "may_become_lockbox_eligible_after_development_gate": False,
            "comparison_reference": "continuation_local_C-ABS_required",
            "external_temporal_abs_numeric_baseline_allowed": False,
        }
    if mode != "post-temporal-no-pass" or selection_value is None:
        raise ObservedAnchorScreenError(
            "post-temporal fallback requires --temporal-selection-record"
        )
    selection_path = Path(selection_value).expanduser().resolve()
    selection = vlf.load_json(selection_path, "temporal DOE frozen selection")
    analysis_record = selection.get("development_analysis")
    if not isinstance(analysis_record, Mapping) or not isinstance(
        analysis_record.get("path"), str
    ):
        raise ObservedAnchorScreenError("temporal selection lacks analysis evidence")
    analysis_path = Path(str(analysis_record["path"])).expanduser().resolve()
    if dict(analysis_record) != vlf.file_record(analysis_path):
        raise ObservedAnchorScreenError("temporal development analysis changed")
    analysis = vlf.load_json(analysis_path, "temporal DOE development analysis")
    input_evaluations = analysis.get("input_evaluations")
    candidate_cells = analysis.get("candidate_cells")
    bootstrap = analysis.get("bootstrap")
    authorization_source = analysis.get("source")
    expected_cells = {
        (arm, nfe)
        for arm in TEMPORAL_CANDIDATE_ARMS
        for nfe in TEMPORAL_SELECTION_NFE
    }
    try:
        observed_cells = {
            (str(cell["arm"]), int(cell["nfe"]))
            for cell in candidate_cells
            if isinstance(cell, Mapping)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ObservedAnchorScreenError(
            "temporal no-pass analysis has malformed candidate cells"
        ) from exc
    input_summaries_are_current = isinstance(input_evaluations, Mapping)
    if input_summaries_are_current:
        for arm in TEMPORAL_ARMS:
            evaluation = input_evaluations.get(arm)
            summary_record = (
                evaluation.get("summary")
                if isinstance(evaluation, Mapping)
                else None
            )
            if not isinstance(summary_record, Mapping) or not isinstance(
                summary_record.get("path"), str
            ):
                input_summaries_are_current = False
                break
            summary_path = Path(str(summary_record["path"])).expanduser().resolve()
            if (
                summary_path.is_symlink()
                or not summary_path.is_file()
                or dict(summary_record) != vlf.file_record(summary_path)
            ):
                input_summaries_are_current = False
                break
    if (
        selection.get("schema") != TEMPORAL_SELECTION_SCHEMA
        or selection.get("status") != "frozen_no_selection"
        or selection.get("selection_count") != 0
        or selection.get("selected_cell") is not None
        or selection.get("selection_split") != "val"
        or selection.get("selection_used_protected_test") is not False
        or selection.get("protected_test_accessed") is not False
        or selection.get("lockbox_may_open") is not False
        or not _identity_is_valid(selection)
        or analysis.get("schema") != TEMPORAL_ANALYSIS_SCHEMA
        or analysis.get("status") != "no_candidate_passed"
        or analysis.get("selected_cell") is not None
        or analysis.get("selection_count") != 0
        or analysis.get("split") != "val"
        or analysis.get("development_selection_split") is not True
        or analysis.get("protected_test_accessed") is not False
        or analysis.get("protected_test_cache_opened") is not False
        or analysis.get("protocol_frozen") is not True
        or not isinstance(authorization_source, Mapping)
        or authorization_source.get("commit") != TEMPORAL_AUTHORIZATION_COMMIT
        or authorization_source.get("dirty") is not False
        or analysis.get("paired_clips") != screen.FROZEN_VALIDATION_CLIPS
        or not isinstance(input_evaluations, Mapping)
        or set(input_evaluations) != set(TEMPORAL_ARMS)
        or selection.get("input_evaluations") != input_evaluations
        or not input_summaries_are_current
        or not isinstance(candidate_cells, list)
        or len(candidate_cells) != len(expected_cells)
        or observed_cells != expected_cells
        or any(
            not isinstance(cell, Mapping)
            or cell.get("composite_gate_passed") is not False
            for cell in candidate_cells
        )
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("samples") != 10_000
        or bootstrap.get("seed") != 20260807
        or bootstrap.get("bonferroni_candidate_cells") != len(expected_cells)
        or bootstrap.get("common_indices_across_all_metrics_cells_controls")
        is not True
        or selection.get("development_analysis_identity_sha256")
        != analysis.get("identity_sha256")
        or not _identity_is_valid(analysis)
    ):
        raise ObservedAnchorScreenError(
            "temporal DOE does not prove a frozen validation-only no-pass result"
        )
    return {
        "mode": mode,
        "temporal_primary_no_pass_attested": True,
        "proxy_validity_only": False,
        "semantic_screen_promotion_eligible": True,
        "video_quality_claim_eligible": False,
        "protected_test_eligible": False,
        "may_become_lockbox_eligible_after_development_gate": True,
        "comparison_reference": "continuation_local_C-ABS_required",
        "external_temporal_abs_numeric_baseline_allowed": False,
        "temporal_no_pass_is_authorization_only": True,
        "temporal_authorization_source_commit": TEMPORAL_AUTHORIZATION_COMMIT,
        "temporal_selection": vlf.file_record(selection_path),
        "temporal_selection_identity_sha256": selection["identity_sha256"],
        "temporal_development_analysis": dict(analysis_record),
        "temporal_development_analysis_identity_sha256": analysis["identity_sha256"],
    }


def action_control(offset: int) -> str:
    if offset not in ACTION_OFFSETS:
        raise ValueError(f"unregistered action offset: {offset}")
    return f"actions_offset_{offset}"


def action_permutation_indices(
    rows: Sequence[Mapping[str, Any]], offset: int
) -> tuple[int, ...]:
    """Return a fixed cyclic action donor map and prove episode disjointness."""
    if offset not in ACTION_OFFSETS:
        raise ValueError(f"unregistered action offset: {offset}")
    size = len(rows)
    if size <= max(ACTION_OFFSETS):
        raise ValueError("validation population is too small for fixed action offsets")
    result = tuple((index + offset) % size for index in range(size))
    for destination, source in enumerate(result):
        if (
            destination == source
            or str(rows[destination]["clip_id"]) == str(rows[source]["clip_id"])
            or int(rows[destination]["episode_index"])
            == int(rows[source]["episode_index"])
        ):
            raise ObservedAnchorScreenError(
                f"action offset {offset} is not clip/episode disjoint"
            )
    return result


def _source_record() -> dict[str, Any]:
    source = vlf.git_record()
    if source.get("dirty") is not False:
        raise ObservedAnchorScreenError(
            "AINC-OFF artifact commands require clean committed source"
        )
    return {
        **source,
        "entrypoint": vlf.file_record(__file__),
        "representation": vlf.file_record(anchor.__file__),
        "cache_bridge": vlf.file_record(
            REPO_ROOT / "tools" / "causal_vjepa2_cache_bridge.py"
        ),
        "shared_screen": vlf.file_record(screen.__file__),
        "shared_runner": vlf.file_record(vlf.__file__),
        "preregistration": anchor.preregistration_record(),
    }


def _manifest_record(
    path: str | Path, *, split: str, expected_clips: int
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    resolved, rows, actual_split = anchor._split_manifest(  # noqa: SLF001
        path, expected_split=split
    )
    if actual_split != split or len(rows) != expected_clips:
        raise ObservedAnchorScreenError(
            f"{split} manifest must contain exactly {expected_clips} clips"
        )
    episodes = [int(row["episode_index"]) for row in rows]
    expected_episodes = 8_000 if split == "train" else screen.FROZEN_VALIDATION_CLIPS
    if len(set(episodes)) != expected_episodes:
        raise ObservedAnchorScreenError(
            f"{split} manifest has an unexpected episode population"
        )
    return (
        resolved,
        rows,
        {
            **vlf.file_record(resolved),
            "split": split,
            "clips": len(rows),
            "episodes": len(set(episodes)),
            "ordered_clip_ids_sha256": screen.sha256_json(
                [str(row["clip_id"]) for row in rows]
            ),
            "ordered_episode_ids_sha256": screen.sha256_json(episodes),
        },
    )


def _construct_datasets(
    args: argparse.Namespace,
) -> tuple[anchor.ObservedAnchorDataset, anchor.ObservedAnchorDataset, dict[str, Any]]:
    train_path, train_rows, train_record = _manifest_record(
        args.train_manifest,
        split="train",
        expected_clips=screen.FROZEN_TRAIN_CLIPS,
    )
    val_path, val_rows, val_record = _manifest_record(
        args.validation_manifest,
        split="val",
        expected_clips=screen.FROZEN_VALIDATION_CLIPS,
    )
    if not rows_episode_ids(train_rows).isdisjoint(rows_episode_ids(val_rows)):
        raise ObservedAnchorScreenError("training and validation episodes overlap")
    train = anchor.construct_observed_anchor_dataset(
        train_path,
        args.data_root,
        args.semantic_cache_root,
        args.anchor_cache_root,
    )
    validation = anchor.construct_observed_anchor_dataset(
        val_path,
        args.data_root,
        args.semantic_cache_root,
        args.anchor_cache_root,
    )
    shared_keys = (
        "pca_sha256",
        "source_commit",
        "checkpoint_sha256",
        "teacher_size",
        "teacher_frames",
        "last_temporal_token",
        "pool_kernel",
        "pooled_token_grid",
        "prefix_frame_map",
        "final_tubelet_history_pair",
    )
    if any(
        train.anchor_cache_metadata.get(key)
        != validation.anchor_cache_metadata.get(key)
        for key in shared_keys
    ):
        raise ObservedAnchorScreenError(
            "train/validation anchors do not share one teacher/PCA/prefix identity"
        )
    return (
        train,
        validation,
        {
            "train": train_record,
            "validation": val_record,
            "semantic_cache": {
                "train": dict(train.base.cache_metadata),
                "validation": dict(validation.base.cache_metadata),
            },
            "anchor_cache": {
                "train": dict(train.anchor_cache_metadata),
                "validation": dict(validation.anchor_cache_metadata),
            },
            "cache_access": {
                "train": train.base.producer_attestation,
                "validation": validation.base.producer_attestation,
            },
            "split_episode_disjoint": True,
            "protected_test_accessed": False,
        },
    )


def _load_normalization_for_datasets(
    args: argparse.Namespace,
    datasets: Mapping[str, Any],
) -> tuple[anchor.ObservedIncrementNormalization, dict[str, Any]]:
    train_manifest = datasets["train"]
    anchors = datasets["anchor_cache"]["train"]
    return anchor.load_increment_normalization(
        args.normalization,
        expected_train_manifest_sha256=str(train_manifest["sha256"]),
        expected_semantic_cache_metadata_sha256=str(
            anchors["semantic_cache_metadata_sha256"]
        ),
        expected_anchor_cache_metadata_sha256=str(
            vlf.file_record(
                Path(args.anchor_cache_root).expanduser().resolve()
                / "train"
                / "metadata.json"
            )["sha256"]
        ),
    )


def _validate_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    batch = screen._validate_batch(raw, device)  # noqa: SLF001
    observed = batch.get("observed_anchor")
    if (
        not isinstance(observed, Tensor)
        or tuple(observed.shape[1:]) != anchor.ANCHOR_SHAPE
        or not observed.is_floating_point()
        or not bool(torch.isfinite(observed).all())
    ):
        raise ObservedAnchorScreenError(
            f"observed anchor must be [B,{anchor.ANCHOR_SHAPE}] and finite"
        )
    batch["observed_anchor"] = observed.float()
    return batch


def ainc_training_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    normalization: anchor.ObservedIncrementNormalization,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run one architecture-matched semantic-only flow update on ``Q``."""
    # The caller enables model bfloat16 autocast.  The representation itself
    # is explicitly outside that context per the preregistered float32 rule.
    with torch.autocast(device_type=batch["auxiliary_target"].device.type, enabled=False):
        clean = anchor.encode_normalized_increment_target(
            batch["auxiliary_target"], batch["observed_anchor"], normalization
        )
    batch_size = clean.shape[0]
    clocks = vlf.sample_training_clocks("phase1", batch_size, clean.device)
    if (
        bool(clocks.video_time.ne(0).any())
        or bool(clocks.video_loss_mask.ne(0).any())
        or bool(clocks.auxiliary_loss_mask.ne(1).any())
    ):
        raise ObservedAnchorScreenError("shared Phase-1 clock implementation changed")
    noisy_video = torch.randn_like(batch["future"])
    auxiliary_noise = torch.randn_like(clean)
    noisy_auxiliary = vlf.corrupt_clean_time(
        clean, auxiliary_noise, clocks.auxiliary_time
    )
    _, prediction_x = vlf.model_forward(
        model,
        noisy_video=noisy_video,
        noisy_auxiliary=noisy_auxiliary,
        t_video=clocks.video_time,
        t_auxiliary=clocks.auxiliary_time,
        history=batch["history"],
        actions=batch["actions"],
        condition_on_auxiliary=True,
        predict_video=False,
    )
    per_example = vlf.per_example_x_prediction_flow_mse(
        prediction_x,
        noisy_auxiliary,
        clean,
        auxiliary_noise,
        clocks.auxiliary_time,
    )
    auxiliary_loss = vlf.masked_branch_loss(per_example, clocks.auxiliary_loss_mask)
    weighted = 0.333 * auxiliary_loss
    return weighted, {
        "auxiliary_loss": auxiliary_loss.detach(),
        "weighted_auxiliary_loss": weighted.detach(),
        "auxiliary_branch_count": clocks.auxiliary_loss_mask.sum().detach(),
    }


def _science_identity(
    *,
    source: Mapping[str, Any],
    model_config: Mapping[str, Any],
    datasets: Mapping[str, Any],
    normalization_record: Mapping[str, Any],
    args: argparse.Namespace,
    context: vlf.DistributedContext,
) -> dict[str, Any]:
    local_batch = args.global_batch_size // context.world_size
    micro_batch = args.micro_batch_size or local_batch
    return {
        "source": dict(source),
        "doe_arm": DOE_ARM,
        "target_kind": anchor.INCREMENT_TARGET_KIND,
        "target_shape": list(anchor.INCREMENT_SHAPE),
        "model": dict(model_config),
        "parameter_count": screen.FROZEN_MODEL_PARAMETERS,
        "seed": args.seed,
        "initialization": "from_scratch_deterministic_no_pretrained_weights",
        "global_batch_size": args.global_batch_size,
        "world_size": context.world_size,
        "local_optimizer_batch_size": local_batch,
        "micro_batch_size_per_rank": micro_batch,
        "gradient_accumulation_steps": local_batch // micro_batch,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "weight_decay": args.weight_decay,
            "warmup_updates": args.warmup_updates,
            "after_warmup": "constant",
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "ema": {
            "decay": args.ema_decay,
            "schedule": vlf.FROZEN_EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        },
        "phase1_schedule": {
            "video_time": 0.0,
            "video_loss": "disabled",
            "auxiliary_logit_normal_mean": -1.2,
            "auxiliary_logit_normal_std": 1.0,
            "auxiliary_loss_coefficient": 0.333,
            "loss_mask_normalization": "unchanged_global_batch",
        },
        "datasets": dict(datasets),
        "normalization": dict(normalization_record),
        "dtype": "bfloat16",
        "target_storage_dtype": "float16",
        "target_flow_compute_dtype": "float32",
        "clean_time_epsilon": vlf.FROZEN_CLEAN_TIME_EPS,
        "workers_per_rank": args.workers,
        "observed_anchor_enters_model": False,
        "future_rgb_model_input": False,
        "protected_test_accessed": False,
        "execution_condition": _execution_condition(args),
    }


def _training_config(
    args: argparse.Namespace,
    context: vlf.DistributedContext,
    *,
    model_config: Mapping[str, Any],
    datasets: Mapping[str, Any],
    normalization_record: Mapping[str, Any],
    total_updates: int,
) -> dict[str, Any]:
    source = _source_record()
    identity = _science_identity(
        source=source,
        model_config=model_config,
        datasets=datasets,
        normalization_record=normalization_record,
        args=args,
        context=context,
    )
    return {
        "schema": RUN_SCHEMA,
        "source": source,
        "entrypoint": vlf.file_record(__file__),
        "command": args.command,
        "doe_arm": DOE_ARM,
        "target_kind": anchor.INCREMENT_TARGET_KIND,
        "target_shape": list(anchor.INCREMENT_SHAPE),
        "clock_convention": vlf.CLOCK_CONVENTION,
        "updates": total_updates,
        "checkpoint_updates": list(
            (CALIBRATION_UPDATES,) if args.command == "calibrate" else TRAIN_CHECKPOINTS
        ),
        "science_identity": identity,
        "science_identity_sha256": screen.sha256_json(identity),
        "normalization": dict(normalization_record),
        "datasets": dict(datasets),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "semantic_cache_root": str(
            Path(args.semantic_cache_root).expanduser().resolve()
        ),
        "anchor_cache_root": str(Path(args.anchor_cache_root).expanduser().resolve()),
        "calibration_record": (
            vlf.file_record(args.calibration_record)
            if args.command == "train"
            else None
        ),
        "wandb": {
            "enabled": args.wandb,
            "entity": args.wandb_entity if args.wandb else None,
            "project": args.wandb_project if args.wandb else None,
            "group": None,
            "private_project_acknowledged": args.wandb_private_project_ack,
        },
    }


def _validate_calibration(
    path: str | Path, *, science_identity_sha256: str
) -> dict[str, Any]:
    record = vlf.load_json(path, "AINC-OFF calibration completion")
    config_record = record.get("resolved_config")
    checkpoint_record = record.get("checkpoint")
    if not isinstance(config_record, Mapping) or not isinstance(
        checkpoint_record, Mapping
    ):
        raise ObservedAnchorScreenError("calibration lacks config/checkpoint evidence")
    config_path = Path(str(config_record.get("path", ""))).resolve()
    checkpoint_path = Path(str(checkpoint_record.get("path", ""))).resolve()
    if (
        record.get("schema") != RUN_SCHEMA
        or record.get("status") != "complete"
        or record.get("command") != "calibrate"
        or record.get("doe_arm") != DOE_ARM
        or record.get("completed_updates") != CALIBRATION_UPDATES
        or record.get("nonfinite_updates") != 0
        or record.get("science_identity_sha256") != science_identity_sha256
        or dict(config_record) != vlf.file_record(config_path)
        or dict(checkpoint_record) != vlf.file_record(checkpoint_path)
    ):
        raise ObservedAnchorScreenError("calibration receipt is invalid")
    config = vlf.load_json(config_path, "AINC-OFF calibration config")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema = payload.get("ema") if isinstance(payload, Mapping) else None
    if (
        screen.sha256_json(config) != payload.get("config_sha256")
        or config.get("science_identity_sha256") != science_identity_sha256
        or payload.get("completed_updates") != CALIBRATION_UPDATES
        or not isinstance(ema, Mapping)
        or ema.get("num_updates") != CALIBRATION_UPDATES
    ):
        raise ObservedAnchorScreenError("calibration checkpoint binding is invalid")
    return record


def training_command(args: argparse.Namespace) -> int:
    context = vlf.initialize_distributed()
    logger: vlf.LocalAndOptionalWandbLogger | None = None
    try:
        total_updates = (
            CALIBRATION_UPDATES if args.command == "calibrate" else TRAIN_UPDATES
        )
        if context.world_size != FROZEN_WORLD_SIZE:
            raise ObservedAnchorScreenError(
                "AINC-OFF calibration/training requires exactly eight ranks"
            )
        if args.global_batch_size % context.world_size:
            raise ObservedAnchorScreenError(
                "global batch 256 must divide by torchrun world size"
            )
        local_batch = args.global_batch_size // context.world_size
        micro_batch = args.micro_batch_size or local_batch
        if (
            local_batch != FROZEN_LOCAL_BATCH_SIZE
            or micro_batch != FROZEN_MICRO_BATCH_SIZE
            or args.workers != FROZEN_TRAIN_WORKERS
            or local_batch % micro_batch
        ):
            raise ObservedAnchorScreenError(
                "AINC-OFF requires local batch 32, microbatch 32, and four workers/rank"
            )
        run_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
        train_dataset, validation_dataset, dataset_records = _construct_datasets(args)
        del validation_dataset
        normalization, normalization_record = _load_normalization_for_datasets(
            args, dataset_records
        )
        vlf.seed_everything(args.seed, 0)
        model, model_config = screen.instantiate_model(args)
        model.to(context.device)
        optimizer, scheduler = vlf.optimizer_and_scheduler(model, args, total_updates)
        ema = vlf.ModelEMA(model, decay=args.ema_decay)
        config = _training_config(
            args,
            context,
            model_config=model_config,
            datasets=dataset_records,
            normalization_record=normalization_record,
            total_updates=total_updates,
        )
        config_sha256 = screen.sha256_json(config)
        screen._assert_distributed_config(context, config_sha256)  # noqa: SLF001
        if args.command == "train":
            _validate_calibration(
                args.calibration_record,
                science_identity_sha256=str(config["science_identity_sha256"]),
            )

        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=False)
            vlf.atomic_write_json(
                run_dir / "resolved_config.json", config, exclusive=True
            )
            vlf.atomic_write_json(
                run_dir / "provenance.json",
                {
                    "schema": RUN_SCHEMA,
                    "source": config["source"],
                    "runtime": vlf.runtime_record(),
                    "command": [sys.executable, *sys.argv],
                    "resolved_config_sha256": config_sha256,
                    "secrets_persisted": False,
                },
                exclusive=True,
            )
        context.barrier()
        vlf.seed_everything(args.seed, context.rank)
        loader = vlf.build_loader(
            train_dataset,
            context=context,
            global_batch_size=args.global_batch_size,
            seed=args.seed,
            start_update=0,
            end_update=total_updates,
            workers=args.workers,
            micro_batch_size=args.micro_batch_size,
        )
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        logger = vlf.LocalAndOptionalWandbLogger(
            run_dir, args, config, primary=context.is_primary
        )
        checkpoints = (
            {CALIBRATION_UPDATES}
            if args.command == "calibrate"
            else set(TRAIN_CHECKPOINTS)
        )
        accumulation_steps = local_batch // micro_batch
        iterator = iter(loader)
        nonfinite_updates = 0
        model.train()
        wall_start = time.perf_counter()
        cumulative_wall = 0.0
        for update in range(1, total_updates + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(3, device=context.device)
            for microstep in range(accumulation_steps):
                batch = _validate_batch(next(iterator), context.device)
                sync = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel)
                    and microstep + 1 < accumulation_steps
                    else contextlib.nullcontext()
                )
                with sync, vlf._autocast(context.device):  # noqa: SLF001
                    weighted, telemetry = ainc_training_step(
                        model, batch, normalization
                    )
                    loss = weighted / accumulation_steps
                if not bool(torch.isfinite(loss)):
                    nonfinite_updates += 1
                    raise ObservedAnchorScreenError(
                        f"nonfinite AINC-OFF loss at update {update}"
                    )
                loss.backward()
                accumulated += torch.stack(
                    (
                        telemetry["weighted_auxiliary_loss"],
                        telemetry["auxiliary_loss"],
                        telemetry["auxiliary_branch_count"],
                    )
                ) / torch.tensor(
                    [accumulation_steps, accumulation_steps, 1.0],
                    device=context.device,
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                nonfinite_updates += 1
                raise ObservedAnchorScreenError(
                    f"nonfinite gradient norm at update {update}"
                )
            optimizer.step()
            scheduler.step()
            ema.update(model)
            packed = context.sum_tensor(accumulated.float())
            observe = (
                update == 1 or update % args.log_every == 0 or update in checkpoints
            )
            if observe:
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                cumulative_wall = time.perf_counter() - wall_start
                logger.log(
                    {
                        "update": update,
                        "doe_arm": DOE_ARM,
                        "weighted_auxiliary_loss": float(
                            packed[0] / context.world_size
                        ),
                        "auxiliary_loss": float(packed[1] / context.world_size),
                        "auxiliary_branch_examples": int(packed[2].item()),
                        "video_loss": 0.0,
                        "gradient_norm": float(gradient_norm.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "cumulative_optimizer_wall_seconds": cumulative_wall,
                    },
                    primary=context.is_primary,
                )
            if update in checkpoints:
                vlf.save_checkpoint(
                    run_dir / "checkpoints" / f"update_{update:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    update=update,
                    arm="phase1",
                    model_config=model_config,
                    config_sha256=config_sha256,
                    context=context,
                    cumulative_optimizer_wall_seconds=cumulative_wall,
                )
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        cumulative_wall = time.perf_counter() - wall_start
        if context.is_primary:
            checkpoint = run_dir / "checkpoints" / f"update_{total_updates:06d}.pt"
            vlf.atomic_write_json(
                run_dir / "complete.json",
                {
                    "schema": RUN_SCHEMA,
                    "status": "complete",
                    "command": args.command,
                    "doe_arm": DOE_ARM,
                    "completed_updates": total_updates,
                    "nonfinite_updates": nonfinite_updates,
                    "target_kind": anchor.INCREMENT_TARGET_KIND,
                    "only_supervised_target": "normalized_anchored_increments",
                    "observed_anchor_enters_model": False,
                    "future_rgb_model_input": False,
                    "video_loss_enabled": False,
                    "protected_test_accessed": False,
                    "execution_condition": config["science_identity"][
                        "execution_condition"
                    ],
                    "science_identity_sha256": config["science_identity_sha256"],
                    "resolved_config_sha256": config_sha256,
                    "source": config["source"],
                    "resolved_config": vlf.file_record(
                        run_dir / "resolved_config.json"
                    ),
                    "provenance": vlf.file_record(run_dir / "provenance.json"),
                    "checkpoint": vlf.file_record(checkpoint),
                    "parameter_counts": vlf.count_parameters(vlf.unwrap_model(model)),
                    "cumulative_optimizer_wall_seconds": cumulative_wall,
                },
                exclusive=True,
            )
        context.barrier()
        return 0
    finally:
        if logger is not None:
            logger.finish()
        vlf.close_distributed(context)


@dataclass(frozen=True)
class AnchoredIncrementSample:
    normalized_prediction: Tensor
    semantic_prediction: Tensor
    model_calls: int
    call_input_sha256_by_example: tuple[tuple[str, ...], ...]


@torch.inference_mode()
def sample_anchored_increments(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    observed_anchor: Tensor,
    *,
    video_noise: Tensor,
    increment_noise: Tensor,
    steps: int,
    normalization: anchor.ObservedIncrementNormalization,
) -> AnchoredIncrementSample:
    """Generate and decode AINC-OFF without any clean future input."""
    forbidden = {"future", "semantic", "target", "clean", "oracle"}
    parameters = set(inspect.signature(sample_anchored_increments).parameters)
    if parameters.intersection(forbidden):
        raise ObservedAnchorScreenError("deployable sampler admits clean future data")
    if increment_noise.dtype != torch.float32:
        raise ValueError("increment sampler state must begin in float32")
    sampled = screen.sample_semantic(
        model,
        history,
        actions,
        video_noise=video_noise,
        auxiliary_noise=increment_noise,
        steps=steps,
    )
    with torch.autocast(device_type=sampled.prediction.device.type, enabled=False):
        semantic = anchor.decode_normalized_increment_prediction(
            sampled.prediction.float(), observed_anchor.float(), normalization
        )
    if sampled.prediction.dtype != torch.float32 or semantic.dtype != torch.float32:
        raise ObservedAnchorScreenError("generated state or decode left float32")
    return AnchoredIncrementSample(
        normalized_prediction=sampled.prediction,
        semantic_prediction=semantic,
        model_calls=sampled.model_calls,
        call_input_sha256_by_example=sampled.call_input_sha256_by_example,
    )


def increment_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Diagnostic normalized-increment metrics per example."""
    if (
        prediction.shape != target.shape
        or tuple(target.shape[1:]) != anchor.INCREMENT_SHAPE
    ):
        raise ValueError("increment metric tensors have the wrong shape")
    prediction = prediction.float()
    target = target.float()
    mse = (prediction - target).square().flatten(1).mean(1)
    power = target.square().flatten(1).mean(1).clamp_min(1e-12)
    cosine = (
        torch.nn.functional.cosine_similarity(
            prediction, target, dim=screen.TARGET_CHANNEL_AXIS, eps=1e-8
        )
        .flatten(1)
        .mean(1)
    )
    return {"increment_nmse": mse / power, "increment_token_cosine": cosine}


def _load_evaluation_checkpoint(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_config: Mapping[str, Any],
    dataset_records: Mapping[str, Any],
    normalization_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.parent.name != "checkpoints":
        raise ObservedAnchorScreenError("checkpoint must remain under run/checkpoints")
    run_root = checkpoint_path.parent.parent
    config_path = run_root / "resolved_config.json"
    complete_path = run_root / "complete.json"
    config = vlf.load_json(config_path, "AINC-OFF training config")
    complete = vlf.load_json(complete_path, "AINC-OFF training completion")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    current_source = _source_record()
    current_execution_condition = _execution_condition(args)
    science_identity = config.get("science_identity")
    provenance_record = complete.get("provenance")
    provenance_path = (
        Path(str(provenance_record.get("path", ""))).resolve()
        if isinstance(provenance_record, Mapping)
        else Path()
    )
    provenance = (
        vlf.load_json(provenance_path, "AINC-OFF training provenance")
        if provenance_path.is_file()
        else {}
    )
    ema = payload.get("ema") if isinstance(payload, Mapping) else None
    if (
        payload.get("schema") != screen.CHECKPOINT_SCHEMA
        or payload.get("arm") != "phase1"
        or payload.get("completed_updates") != TRAIN_UPDATES
        or payload.get("model_config") != model_config
        or config.get("schema") != RUN_SCHEMA
        or config.get("command") != "train"
        or config.get("doe_arm") != DOE_ARM
        or config.get("target_kind") != anchor.INCREMENT_TARGET_KIND
        or config.get("source") != current_source
        or not isinstance(science_identity, Mapping)
        or config.get("science_identity_sha256")
        != screen.sha256_json(science_identity)
        or science_identity.get("execution_condition")
        != current_execution_condition
        or config.get("normalization") != dict(normalization_record)
        or config.get("datasets") != dict(dataset_records)
        or science_identity.get("datasets") != dict(dataset_records)
        or science_identity.get("normalization") != dict(normalization_record)
        or science_identity.get("world_size") != FROZEN_WORLD_SIZE
        or science_identity.get("local_optimizer_batch_size")
        != FROZEN_LOCAL_BATCH_SIZE
        or science_identity.get("micro_batch_size_per_rank")
        != FROZEN_MICRO_BATCH_SIZE
        or science_identity.get("workers_per_rank") != FROZEN_TRAIN_WORKERS
        or science_identity.get("parameter_count") != screen.FROZEN_MODEL_PARAMETERS
        or science_identity.get("model") != model_config
        or science_identity.get("initialization")
        != "from_scratch_deterministic_no_pretrained_weights"
        or screen.sha256_json(config) != payload.get("config_sha256")
        or complete.get("schema") != RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("command") != "train"
        or complete.get("doe_arm") != DOE_ARM
        or complete.get("completed_updates") != TRAIN_UPDATES
        or complete.get("nonfinite_updates") != 0
        or complete.get("observed_anchor_enters_model") is not False
        or complete.get("future_rgb_model_input") is not False
        or complete.get("protected_test_accessed") is not False
        or complete.get("science_identity_sha256")
        != config.get("science_identity_sha256")
        or complete.get("resolved_config_sha256") != screen.sha256_json(config)
        or complete.get("source") != current_source
        or complete.get("checkpoint") != vlf.file_record(checkpoint_path)
        or complete.get("resolved_config") != vlf.file_record(config_path)
        or not isinstance(provenance_record, Mapping)
        or dict(provenance_record) != vlf.file_record(provenance_path)
        or provenance.get("schema") != RUN_SCHEMA
        or provenance.get("source") != current_source
        or provenance.get("resolved_config_sha256") != screen.sha256_json(config)
        or provenance.get("secrets_persisted") is not False
        or not isinstance(ema, Mapping)
        or ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or ema.get("num_updates") != TRAIN_UPDATES
        or not isinstance(ema.get("shadow"), Mapping)
    ):
        raise ObservedAnchorScreenError(
            "checkpoint is not the exact completed AINC-OFF model"
        )
    model.load_state_dict(ema["shadow"], strict=True)
    return (
        dict(payload),
        config,
        vlf.file_record(checkpoint_path),
        vlf.file_record(config_path),
    )


def _metric_bundle(
    semantic_prediction: Tensor,
    semantic_target: Tensor,
    increment_prediction: Tensor,
    increment_target: Tensor,
) -> dict[str, Tensor]:
    return {
        **screen.semantic_metrics(semantic_prediction, semantic_target),
        **increment_metrics(increment_prediction, increment_target),
    }


def _summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault((int(row["nfe"]), str(row["control"])), []).append(row)
    metric_names = (
        "semantic_nmse",
        "semantic_token_cosine",
        "temporal_difference_nmse",
        "temporal_difference_token_cosine",
        "increment_nmse",
        "increment_token_cosine",
    )
    return [
        {
            "nfe": nfe,
            "control": control,
            "clips": len(values),
            **{
                name: float(sum(float(row[name]) for row in values) / len(values))
                for name in metric_names
            },
        }
        for (nfe, control), values in sorted(grouped.items())
    ]


def evaluation_command(args: argparse.Namespace) -> int:
    determinism = screen._configure_deterministic_eval()  # noqa: SLF001
    context = vlf.initialize_distributed()
    logger: vlf.LocalAndOptionalWandbLogger | None = None
    try:
        if context.world_size != FROZEN_WORLD_SIZE:
            raise ObservedAnchorScreenError(
                "AINC-OFF development evaluation requires exactly eight ranks"
            )
        output_dir = vlf.validated_run_dir(
            args.artifact_root, args.run_id, resume=False
        )
        val_path, rows, _ = _manifest_record(
            args.manifest,
            split="val",
            expected_clips=screen.FROZEN_VALIDATION_CLIPS,
        )
        # Build both split records so normalization/checkpoint bindings are
        # revalidated against the exact training population at evaluation.
        proxy = argparse.Namespace(**vars(args))
        proxy.validation_manifest = str(val_path)
        train_dataset, dataset, dataset_records = _construct_datasets(proxy)
        del train_dataset
        normalization, normalization_record = _load_normalization_for_datasets(
            args, dataset_records
        )
        vlf.seed_everything(args.seed, 0)
        model, model_config = screen.instantiate_model(args)
        model.to(context.device)
        payload, _training_config, checkpoint_record, config_record = (
            _load_evaluation_checkpoint(
                args,
                model=model,
                model_config=model_config,
                dataset_records=dataset_records,
                normalization_record=normalization_record,
            )
        )
        model.eval()
        for index in range(0, len(rows), 2):
            if int(rows[index]["episode_index"]) == int(
                rows[index + 1]["episode_index"]
            ):
                raise ObservedAnchorScreenError("adjacent donors share an episode")
        permutations = {
            action_control(offset): action_permutation_indices(rows, offset)
            for offset in ACTION_OFFSETS
        }
        action_bank = torch.stack(
            [dataset[index]["actions"] for index in range(len(dataset))]
        )
        local_indices, local_batches = vlf.paired_rank_evaluation_layout(
            len(dataset),
            args.eval_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        loader = DataLoader(
            torch.utils.data.Subset(dataset, local_indices),
            batch_sampler=local_batches,
            num_workers=args.workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        donor_mapping = {
            str(rows[index]["clip_id"]): str(rows[index ^ 1]["clip_id"])
            for index in range(len(rows))
        }
        config = {
            "schema": EVALUATION_SCHEMA,
            "source": _source_record(),
            "entrypoint": vlf.file_record(__file__),
            "doe_arm": DOE_ARM,
            "split": "val",
            "validation_clips": len(dataset),
            "checkpoint": checkpoint_record,
            "training_config": config_record,
            "checkpoint_update": int(payload["completed_updates"]),
            "normalization": normalization_record,
            "datasets": dataset_records,
            "manifest": dataset_records["validation"],
            "semantic_cache": dataset_records["semantic_cache"]["validation"],
            "anchor_cache": dataset_records["anchor_cache"]["validation"],
            "cache_producer_attestation": dataset_records["cache_access"][
                "validation"
            ],
            "seed": args.seed,
            "nfe_grid": list(NFE_GRID),
            "controls": list(CONTROLS),
            "fixed_clip_noise": True,
            "fixed_noise_key": "sha256(f'{clip_id}:{eval_seed}:video|aux')",
            "donor_rule": "manifest-adjacent xor-1; episode-disjoint",
            "donor_mapping_sha256": screen.sha256_json(donor_mapping),
            "action_offsets": list(ACTION_OFFSETS),
            "action_permutations_sha256": {
                name: screen.sha256_json(list(indices))
                for name, indices in permutations.items()
            },
            "observed_anchor_online_contract": (
                "history-only V-JEPA; cached validation bytes are used for this screen"
            ),
            "anchor_enters_model": False,
            "clean_future_target_entered_sampler": False,
            "future_rgb_entered_sampler": False,
            "protected_test_accessed": False,
            "execution_condition": _execution_condition(args),
            "world_size": context.world_size,
            "eval_batch_size": args.eval_batch_size,
            "workers_per_rank": args.workers,
            "determinism": determinism,
            "wandb": {
                "enabled": args.wandb,
                "entity": args.wandb_entity if args.wandb else None,
                "project": args.wandb_project if args.wandb else None,
                "group": None,
                "private_project_acknowledged": args.wandb_private_project_ack,
            },
        }
        config_sha256 = screen.sha256_json(config)
        screen._assert_distributed_config(context, config_sha256)  # noqa: SLF001
        if context.is_primary:
            output_dir.mkdir(parents=True, exist_ok=False)
            vlf.atomic_write_json(
                output_dir / "resolved_config.json", config, exclusive=True
            )
            vlf.atomic_write_json(
                output_dir / "provenance.json",
                {
                    "schema": EVALUATION_SCHEMA,
                    "source": config["source"],
                    "runtime": vlf.runtime_record(),
                    "command": [sys.executable, *sys.argv],
                    "resolved_config_sha256": config_sha256,
                    "secrets_persisted": False,
                },
                exclusive=True,
            )
        context.barrier()
        logger = vlf.LocalAndOptionalWandbLogger(
            output_dir, args, config, primary=context.is_primary
        )
        clip_to_index = {str(row["clip_id"]): index for index, row in enumerate(rows)}
        episode_by_clip = {
            str(row["clip_id"]): int(row["episode_index"]) for row in rows
        }
        local_records: list[dict[str, Any]] = []
        local_calls = 0
        for raw_batch in loader:
            batch = _validate_batch(raw_batch, context.device)
            clip_ids = [str(value) for value in raw_batch["clip_id"]]
            episode_indices = [int(value) for value in raw_batch["episode_index"]]
            screen._validate_pair_batch(clip_ids, episode_indices)  # noqa: SLF001
            global_indices = [clip_to_index[clip_id] for clip_id in clip_ids]
            donor_index = torch.arange(
                len(clip_ids), device=context.device, dtype=torch.long
            ).bitwise_xor(1)
            donor_ids = [clip_ids[int(value)] for value in donor_index.cpu().tolist()]
            own_semantic = batch["auxiliary_target"].float()
            own_anchor = batch["observed_anchor"].float()
            own_q = anchor.encode_normalized_increment_target(
                own_semantic, own_anchor, normalization
            )
            video_noise = vlf.stable_noise_like(
                batch["future"].float(), clip_ids, args.seed, "video"
            )
            increment_noise = vlf.stable_noise_like(
                own_q, clip_ids, args.seed, "aux"
            ).float()
            sources: dict[str, tuple[Tensor, Tensor, Tensor, list[str], list[str]]] = {
                "autonomous": (
                    batch["history"],
                    batch["actions"],
                    own_anchor,
                    clip_ids,
                    clip_ids,
                ),
                "context_shuffled": (
                    batch["history"].index_select(0, donor_index),
                    batch["actions"].index_select(0, donor_index),
                    own_anchor.index_select(0, donor_index),
                    donor_ids,
                    donor_ids,
                ),
                "history_shuffled": (
                    batch["history"].index_select(0, donor_index),
                    batch["actions"],
                    own_anchor.index_select(0, donor_index),
                    donor_ids,
                    clip_ids,
                ),
            }
            for control, permutation in permutations.items():
                source_indices = [permutation[index] for index in global_indices]
                source_ids = [str(rows[index]["clip_id"]) for index in source_indices]
                source_actions = action_bank.index_select(
                    0, torch.tensor(source_indices, dtype=torch.long)
                ).to(context.device, non_blocking=True)
                sources[control] = (
                    batch["history"],
                    source_actions,
                    own_anchor,
                    clip_ids,
                    source_ids,
                )
            source_input_records = {
                control: [
                    screen.sampler_input_record(
                        history[item],
                        actions[item],
                        video_noise[item],
                        increment_noise[item],
                    )
                    for item in range(len(clip_ids))
                ]
                for control, (history, actions, _, _, _) in sources.items()
            }
            for nfe in NFE_GRID:
                generated: dict[str, AnchoredIncrementSample] = {}
                for control in GENERATED_CONTROLS:
                    history, actions, decode_anchor, _, _ = sources[control]
                    with vlf._autocast(context.device):  # noqa: SLF001
                        generated[control] = sample_anchored_increments(
                            model,
                            history,
                            actions,
                            decode_anchor,
                            video_noise=video_noise,
                            increment_noise=increment_noise,
                            steps=nfe,
                            normalization=normalization,
                        )
                    local_calls += generated[control].model_calls
                static_q, static_semantic = anchor.anchor_static_control(
                    own_anchor, normalization
                )
                mean_q, mean_semantic = anchor.mean_increment_control(
                    own_anchor, normalization
                )
                zero_semantic = torch.zeros_like(own_semantic)
                zero_q = anchor.encode_normalized_increment_target(
                    zero_semantic, own_anchor, normalization
                )
                auto = generated["autonomous"]
                decoded_donor_anchor = anchor.decode_normalized_increment_prediction(
                    auto.normalized_prediction,
                    own_anchor.index_select(0, donor_index),
                    normalization,
                )
                semantic_predictions: dict[str, Tensor] = {
                    **{
                        control: generated[control].semantic_prediction
                        for control in GENERATED_CONTROLS
                    },
                    "anchor_static": static_semantic,
                    "mean_increment": mean_semantic,
                    "donor_target": auto.semantic_prediction,
                    "anchor_decode_shuffled": decoded_donor_anchor,
                    "zero": zero_semantic,
                    "oracle_clean": own_semantic,
                }
                increment_predictions: dict[str, Tensor] = {
                    **{
                        control: generated[control].normalized_prediction
                        for control in GENERATED_CONTROLS
                    },
                    "anchor_static": static_q,
                    "mean_increment": mean_q,
                    "donor_target": auto.normalized_prediction,
                    "anchor_decode_shuffled": auto.normalized_prediction,
                    "zero": zero_q,
                    "oracle_clean": own_q,
                }
                for control in CONTROLS:
                    donor_target = control == "donor_target"
                    semantic_target = (
                        own_semantic.index_select(0, donor_index)
                        if donor_target
                        else own_semantic
                    )
                    increment_target = (
                        own_q.index_select(0, donor_index) if donor_target else own_q
                    )
                    metrics = {
                        key: value.detach().cpu().tolist()
                        for key, value in _metric_bundle(
                            semantic_predictions[control],
                            semantic_target,
                            increment_predictions[control],
                            increment_target,
                        ).items()
                    }
                    generated_sample = (
                        auto
                        if control in {"donor_target", "anchor_decode_shuffled"}
                        else generated.get(control)
                    )
                    for item, clip_id in enumerate(clip_ids):
                        if control in sources:
                            _, _, decode_anchors, history_ids, action_ids = sources[
                                control
                            ]
                            history_source = history_ids[item]
                            action_source = action_ids[item]
                            decode_anchor_source = history_ids[item]
                            sampler_record = source_input_records[control][item]
                            decode_anchor_value = decode_anchors[item]
                        elif control == "anchor_decode_shuffled":
                            history_source = clip_id
                            action_source = clip_id
                            decode_anchor_source = donor_ids[item]
                            sampler_record = source_input_records["autonomous"][item]
                            donor_item = int(donor_index[item].item())
                            decode_anchor_value = own_anchor[donor_item]
                        elif control == "donor_target":
                            history_source = clip_id
                            action_source = clip_id
                            decode_anchor_source = clip_id
                            sampler_record = source_input_records["autonomous"][item]
                            decode_anchor_value = own_anchor[item]
                        else:
                            history_source = None
                            action_source = None
                            decode_anchor_source = clip_id
                            sampler_record = None
                            decode_anchor_value = own_anchor[item]
                        traces = (
                            list(generated_sample.call_input_sha256_by_example[item])
                            if generated_sample is not None
                            else []
                        )
                        local_records.append(
                            {
                                "clip_id": clip_id,
                                "schema": EVALUATION_SCHEMA,
                                "doe_arm": DOE_ARM,
                                "episode_index": episode_indices[item],
                                "donor_clip_id": donor_ids[item],
                                "donor_episode_index": episode_by_clip[donor_ids[item]],
                                "control": control,
                                "nfe": nfe,
                                "evaluation_seed": args.seed,
                                "checkpoint_sha256": checkpoint_record["sha256"],
                                "training_config_sha256": config_record["sha256"],
                                "evaluation_config_sha256": config_sha256,
                                "normalization_artifact_sha256": normalization_record[
                                    "sha256"
                                ],
                                "history_source_clip_id": history_source,
                                "actions_source_clip_id": action_source,
                                "actions_source_episode_index": (
                                    episode_by_clip[action_source]
                                    if action_source is not None
                                    else None
                                ),
                                "decode_anchor_source_clip_id": decode_anchor_source,
                                "decode_anchor_source_episode_index": episode_by_clip[
                                    decode_anchor_source
                                ],
                                "target_source_clip_id": (
                                    donor_ids[item] if donor_target else clip_id
                                ),
                                "generation_reused_from": (
                                    "autonomous"
                                    if control
                                    in {"donor_target", "anchor_decode_shuffled"}
                                    else None
                                ),
                                "conceptual_path_model_calls": (
                                    nfe if generated_sample is not None else 0
                                ),
                                "actual_evaluator_model_calls": (
                                    0
                                    if control
                                    in {"donor_target", "anchor_decode_shuffled"}
                                    else (
                                        generated_sample.model_calls
                                        if generated_sample is not None
                                        else 0
                                    )
                                ),
                                "model_call_input_sha256": traces,
                                "model_call_input_chain_sha256": screen.sha256_json(
                                    traces
                                ),
                                "sampler_input": sampler_record,
                                "sampler_input_sha256": (
                                    screen.sampler_input_sha256(sampler_record)
                                    if sampler_record is not None
                                    else None
                                ),
                                "decode_anchor_sha256": vlf.tensor_sha256(
                                    decode_anchor_value
                                ),
                                "generated_increment_sha256": vlf.tensor_sha256(
                                    increment_predictions[control][item]
                                ),
                                "decoded_semantic_sha256": vlf.tensor_sha256(
                                    semantic_predictions[control][item]
                                ),
                                "metric_target_sha256": vlf.tensor_sha256(
                                    semantic_target[item]
                                ),
                                "metric_increment_target_sha256": vlf.tensor_sha256(
                                    increment_target[item]
                                ),
                                "clean_future_target_entered_sampler": False,
                                "future_rgb_entered_sampler": False,
                                "anchor_entered_model": False,
                                "teacher_model_calls_during_sampling": 0,
                                "trajectory_state_dtype": "float32",
                                "transformer_autocast": "cuda-bfloat16",
                                "metric_target_available_at_inference": False,
                                "generation_deployable": control
                                in GENERATED_CONTROLS,
                                "control_deployable": control
                                in {
                                    *GENERATED_CONTROLS,
                                    "anchor_static",
                                    "mean_increment",
                                },
                                "metric_comparison_only": control
                                in {
                                    "donor_target",
                                    "anchor_decode_shuffled",
                                    "zero",
                                    "oracle_clean",
                                },
                                **{
                                    name: float(values[item])
                                    for name, values in metrics.items()
                                },
                            }
                        )
        rank_path = output_dir / "rank_metrics" / f"rank_{context.rank:04d}.jsonl"
        screen._atomic_jsonl(rank_path, local_records)  # noqa: SLF001
        shard = {
            "rank": context.rank,
            "records": len(local_records),
            "actual_batched_transformer_calls": local_calls,
            "file": vlf.file_record(rank_path),
        }
        shards = context.gather_objects(shard)
        context.barrier()
        if context.is_primary:
            records: list[dict[str, Any]] = []
            for item in sorted(shards, key=lambda value: int(value["rank"])):
                path = Path(item["file"]["path"])
                if item["file"] != vlf.file_record(path):
                    raise ObservedAnchorScreenError("rank metric shard changed")
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ObservedAnchorScreenError("metric row is malformed")
                        records.append(row)
            records.sort(key=lambda row: (row["nfe"], row["control"], row["clip_id"]))
            expected = len(dataset) * len(NFE_GRID) * len(CONTROLS)
            if len(records) != expected:
                raise ObservedAnchorScreenError(
                    f"evaluation produced {len(records)} rows, expected {expected}"
                )
            metrics_path = output_dir / "per_clip_metrics.jsonl"
            screen._atomic_jsonl(metrics_path, records)  # noqa: SLF001
            summaries = _summaries(records)
            summary = {
                "schema": EVALUATION_SCHEMA,
                "status": "complete",
                "doe_arm": DOE_ARM,
                "split": "val",
                "development_selection_split": True,
                "protected_test_accessed": False,
                "execution_condition": config["execution_condition"],
                "checkpoint": checkpoint_record,
                "training_config": config_record,
                "normalization": normalization_record,
                "manifest": config["manifest"],
                "semantic_cache": config["semantic_cache"],
                "anchor_cache": config["anchor_cache"],
                "cache_producer_attestation": config[
                    "cache_producer_attestation"
                ],
                "resolved_config": vlf.file_record(output_dir / "resolved_config.json"),
                "provenance": vlf.file_record(output_dir / "provenance.json"),
                "per_clip_metrics": vlf.file_record(metrics_path),
                "record_count": len(records),
                "cell_count": len(summaries),
                "summaries": summaries,
                "rank_shards": shards,
                "actual_batched_transformer_calls": sum(
                    int(item["actual_batched_transformer_calls"]) for item in shards
                ),
                "nfe_is_actual_transformer_calls_per_generated_path": True,
                "donor_and_anchor_decode_controls_reuse_generation": True,
                "clean_future_target_entered_deployable_sampler": False,
                "future_rgb_entered_deployable_sampler": False,
                "teacher_model_calls": 0,
                "anchor_entered_model": False,
                "increment_metrics_are_diagnostic_only": True,
            }
            vlf.atomic_write_json(output_dir / "summary.json", summary, exclusive=True)
            for index, cell in enumerate(summaries):
                logger.log(
                    {
                        "update": TRAIN_UPDATES + index,
                        "event": "causal_vjepa2_observed_anchor_evaluation",
                        "doe_arm": DOE_ARM,
                        **cell,
                    },
                    primary=True,
                )
        context.barrier()
        return 0
    finally:
        if logger is not None:
            logger.finish()
        vlf.close_distributed(context)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=vlf.FROZEN_MODEL_WIDTH)
    parser.add_argument("--depth", type=int, default=vlf.FROZEN_MODEL_DEPTH)
    parser.add_argument("--heads", type=int, default=vlf.FROZEN_MODEL_HEADS)
    parser.add_argument("--mlp-ratio", type=float, default=vlf.FROZEN_MODEL_MLP_RATIO)


def _add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-private-project-ack", action="store_true")


def _add_shared_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--semantic-cache-root", required=True)
    parser.add_argument("--anchor-cache-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, required=True)
    parser.add_argument("--temporal-selection-record")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "train"):
        command = commands.add_parser(name)
        _add_model_arguments(command)
        _add_shared_data_arguments(command)
        command.add_argument("--validation-manifest", required=True)
        command.add_argument("--artifact-root", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--calibration-record", required=name == "train")
        command.add_argument(
            "--global-batch-size", type=int, default=screen.FROZEN_GLOBAL_BATCH_SIZE
        )
        command.add_argument("--micro-batch-size", type=int)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--seed", type=int, default=screen.FROZEN_SEED)
        command.add_argument(
            "--learning-rate", type=float, default=vlf.FROZEN_LEARNING_RATE
        )
        command.add_argument(
            "--warmup-updates", type=int, default=vlf.FROZEN_WARMUP_UPDATES
        )
        command.add_argument(
            "--weight-decay", type=float, default=vlf.FROZEN_WEIGHT_DECAY
        )
        command.add_argument(
            "--gradient-clip-norm",
            type=float,
            default=vlf.FROZEN_GRADIENT_CLIP_NORM,
        )
        command.add_argument("--ema-decay", type=float, default=vlf.FROZEN_EMA_DECAY)
        command.add_argument("--log-every", type=int, default=10)
        _add_wandb_arguments(command)

    evaluate = commands.add_parser("evaluate")
    _add_model_arguments(evaluate)
    _add_shared_data_arguments(evaluate)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--artifact-root", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--eval-batch-size", type=int, default=8)
    evaluate.add_argument("--workers", type=int, default=2)
    evaluate.add_argument("--seed", type=int, default=EVALUATION_SEED)
    _add_wandb_arguments(evaluate)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.width != vlf.FROZEN_MODEL_WIDTH
        or args.depth != vlf.FROZEN_MODEL_DEPTH
        or args.heads != vlf.FROZEN_MODEL_HEADS
        or args.mlp_ratio != vlf.FROZEN_MODEL_MLP_RATIO
        or args.workers < 0
    ):
        raise ObservedAnchorScreenError(
            "model geometry is frozen at width512/depth12/heads8/MLP4"
        )
    if args.command in {"calibrate", "train"}:
        if (
            args.seed != screen.FROZEN_SEED
            or args.global_batch_size != screen.FROZEN_GLOBAL_BATCH_SIZE
            or args.learning_rate != vlf.FROZEN_LEARNING_RATE
            or args.warmup_updates != vlf.FROZEN_WARMUP_UPDATES
            or args.weight_decay != vlf.FROZEN_WEIGHT_DECAY
            or args.gradient_clip_norm != vlf.FROZEN_GRADIENT_CLIP_NORM
            or args.ema_decay != vlf.FROZEN_EMA_DECAY
            or args.workers != FROZEN_TRAIN_WORKERS
            or args.log_every < 1
            or args.micro_batch_size
            not in (None, FROZEN_MICRO_BATCH_SIZE)
        ):
            raise ObservedAnchorScreenError(
                "AINC-OFF preserves seed1234/global256/AdamW/clip1/EMA.9999"
            )
    elif (
        args.seed != EVALUATION_SEED
        or args.eval_batch_size != FROZEN_EVAL_BATCH_SIZE
        or args.workers != FROZEN_EVAL_WORKERS
    ):
        raise ObservedAnchorScreenError(
            "evaluation preserves seed and even paired batches"
        )
    _execution_condition(args)
    wandb_values = (
        args.wandb_entity,
        args.wandb_project,
        args.wandb_private_project_ack,
    )
    if args.wandb != all(bool(value) for value in wandb_values):
        raise ObservedAnchorScreenError(
            "W&B requires entity, project, and private-project acknowledgement"
        )
    if args.wandb and (
        args.wandb_entity != "zijiandu"
        or args.wandb_project != "dual-video-diffusion-private"
    ):
        raise ObservedAnchorScreenError(
            "W&B is frozen to private zijiandu/dual-video-diffusion-private"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.command in {"calibrate", "train"}:
            return training_command(args)
        return evaluation_command(args)
    except (
        ObservedAnchorScreenError,
        anchor.ObservedAnchorError,
        screen.ScreenError,
        vlf.PocError,
        ValueError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"AINC-OFF screen error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
