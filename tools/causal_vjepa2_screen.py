#!/usr/bin/env python3
"""Train and evaluate the prefix-causal V-JEPA 2 semantic predictor screen.

This executable is deliberately separate from the production LACWM trainer and
from the low-resolution RGB scratchpad experiment.  It reuses the frozen
41,963,760-parameter Video Latent Forcing transformer, but the *only* training
target is the cached prefix-causal V-JEPA 2 tensor ``[48, 8, 8, 14]``.

The flow convention is fixed throughout::

    z(t) = t * clean + (1 - t) * noise
    t=0 is noise, t=1 is clean, velocity = clean - noise.

At evaluation time the deployable sampler accepts only history RGB, actions,
and clip-addressed noise.  A clean semantic target is passed only to metric
code.  In particular, the donor-target control changes the metric target but
reuses the bit-identical autonomous generation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.causal_vjepa2 import (  # noqa: E402
    CACHE_ARTIFACT_TYPE,
    CausalVJEPA2DroidDataset,
    TARGET_KIND as CAUSAL_TARGET_KIND,
)
from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    read_clip_manifest,
    rows_episode_ids,
    sha256_file,
)
from tools import video_latent_forcing_poc as vlf  # noqa: E402


RUN_SCHEMA = "causal-vjepa2-semantic-screen-run-v1"
EVALUATION_SCHEMA = "causal-vjepa2-semantic-screen-evaluation-v1"
TARGET_KIND = CAUSAL_TARGET_KIND
TARGET_SHAPE = (48, 8, 8, 14)
TARGET_CHANNEL_AXIS = 1
TEMPORAL_AXIS = 2
CALIBRATION_UPDATES = 200
TRAIN_UPDATES = 5_000
TRAIN_CHECKPOINTS = (500, 1_000, 2_000, 5_000)
NFE_GRID = (1, 2, 4, 8, 12, 20, 25)
CONTROLS = (
    "autonomous",
    "donor_target",
    "context_shuffled",
    "history_shuffled",
    "actions_shuffled",
    "zero",
    "oracle_clean",
)
GENERATED_CONTROLS = (
    "autonomous",
    "context_shuffled",
    "history_shuffled",
    "actions_shuffled",
)
FROZEN_SEED = 1234
FROZEN_EVALUATION_SEED = 20260801
FROZEN_VALIDATION_CLIPS = 890
FROZEN_TRAIN_CLIPS = 64_000
FROZEN_GLOBAL_BATCH_SIZE = vlf.FROZEN_GLOBAL_BATCH_SIZE
FROZEN_MODEL_PARAMETERS = vlf.FROZEN_PARAMETER_COUNT
CHECKPOINT_SCHEMA = vlf.CHECKPOINT_SCHEMA
HEX64 = re.compile(r"[0-9a-f]{64}")
CACHE_SHARED_UPSTREAM_KEYS = (
    "pca_sha256",
    "implementation",
    "source_commit",
    "checkpoint_sha256",
    "checkpoint_evidence",
    "source_archive_sha256",
    "source_license",
    "train_manifest_sha256",
    "teacher_size",
    "teacher_frames",
    "last_temporal_token",
    "pooled_token_grid",
    "base_droid",
    "runtime",
    "numerical_contract",
)


class ScreenError(RuntimeError):
    """The scientific or provenance contract failed closed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Publish a complete JSONL artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ScreenError(f"immutable JSONL artifact already exists: {path}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ScreenError(f"immutable JSONL artifact appeared while publishing: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _source_record() -> dict[str, Any]:
    source = vlf.git_record()
    if source.get("dirty") is not False:
        raise ScreenError("the semantic screen requires clean, committed source")
    return source


def _dataset_source_record() -> dict[str, Any]:
    source = inspect.getsourcefile(CausalVJEPA2DroidDataset)
    if source is None:
        raise ScreenError("cannot resolve the causal V-JEPA dataset source")
    return vlf.file_record(source)


def validated_cache_metadata(dataset: CausalVJEPA2DroidDataset) -> dict[str, Any]:
    """Read the dataset's already-validated immutable cache metadata.

    The cache implementation owns validation of its tensor shards.  This
    screen additionally verifies every explicit ``{path, sha256}`` evidence
    record and freezes the complete metadata mapping into every run config.
    """
    value = getattr(dataset, "validated_cache_metadata", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(dataset, "cache_metadata", None)
        if callable(value):
            value = value()
    if not isinstance(value, Mapping):
        raise ScreenError(
            "CausalVJEPA2DroidDataset must expose validated_cache_metadata or cache_metadata"
        )
    metadata = _jsonable(value)
    if not isinstance(metadata, dict):
        raise ScreenError("semantic cache metadata must resolve to a JSON mapping")
    if metadata.get("artifact_type") != CACHE_ARTIFACT_TYPE:
        raise ScreenError(
            f"semantic cache metadata does not bind artifact type {CACHE_ARTIFACT_TYPE}"
        )
    if metadata.get("target_kind") != TARGET_KIND:
        raise ScreenError(f"semantic cache metadata does not bind target kind {TARGET_KIND}")
    if list(TARGET_SHAPE) not in (
        metadata.get("per_clip_target_shape"),
        metadata.get("target_shape"),
        metadata.get("auxiliary_target_shape"),
    ):
        raise ScreenError(f"semantic cache metadata does not bind target shape {TARGET_SHAPE}")
    try:
        verified = vlf._verify_embedded_file_records(  # noqa: SLF001 - shared provenance primitive
            metadata, label="causal V-JEPA semantic cache"
        )
    except vlf.PocError as exc:
        raise ScreenError(str(exc)) from exc
    if verified < 1:
        raise ScreenError("semantic cache metadata must contain hashed provenance evidence")
    return metadata


def _base_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: record.get(name) for name in ("path", "sha256", "bytes")}


def _validate_runtime_provenance(value: Any) -> dict[str, Any]:
    """Validate the exact-shaped runtime record emitted by the shared runner."""
    if not isinstance(value, Mapping):
        raise ScreenError("run provenance lacks a runtime record")
    runtime = dict(value)
    expected_keys = {
        "utc",
        "hostname",
        "python",
        "platform",
        "torch_cuda",
        "cuda_available",
        "packages",
        "slurm",
        "gpu_inventory",
    }
    if set(runtime) != expected_keys:
        raise ScreenError("run provenance runtime inventory is malformed")
    if any(
        not isinstance(runtime.get(name), str) or not runtime[name]
        for name in ("utc", "hostname", "python", "platform")
    ):
        raise ScreenError("run provenance runtime identity is malformed")
    if runtime.get("torch_cuda") is not None and not isinstance(
        runtime.get("torch_cuda"), str
    ):
        raise ScreenError("run provenance CUDA runtime is malformed")
    if not isinstance(runtime.get("cuda_available"), bool):
        raise ScreenError("run provenance CUDA availability is malformed")
    for name in ("packages", "slurm"):
        mapping = runtime.get(name)
        if not isinstance(mapping, Mapping) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in mapping.items()
        ):
            raise ScreenError(f"run provenance {name} inventory is malformed")
    inventory = runtime.get("gpu_inventory")
    if not isinstance(inventory, list) or any(
        not isinstance(item, str) for item in inventory
    ):
        raise ScreenError("run provenance GPU inventory is malformed")
    return runtime


def _validate_run_provenance(
    value: Any,
    *,
    schema: str,
    source: Mapping[str, Any],
    config_sha256: str,
    entrypoint: Mapping[str, Any],
    command: str,
) -> dict[str, Any]:
    """Bind runtime and argv evidence to one immutable resolved config."""
    if not isinstance(value, Mapping):
        raise ScreenError("run provenance must be a JSON mapping")
    provenance = dict(value)
    argv = provenance.get("command")
    entrypoint_path = entrypoint.get("path")
    command_valid = (
        isinstance(argv, list)
        and len(argv) >= 3
        and all(isinstance(item, str) and item for item in argv)
        and isinstance(entrypoint_path, str)
        and Path(argv[1]).expanduser().resolve()
        == Path(entrypoint_path).expanduser().resolve()
        and argv[2] == command
    )
    if (
        provenance.get("schema") != schema
        or provenance.get("source") != source
        or provenance.get("resolved_config_sha256") != config_sha256
        or provenance.get("secrets_persisted") is not False
        or not command_valid
    ):
        raise ScreenError("run provenance is not bound to its exact config/source/argv")
    _validate_runtime_provenance(provenance.get("runtime"))
    return provenance


def _validate_training_cache_pair(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    *,
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
) -> None:
    """Require one immutable PCA/upstream identity for both data splits."""
    for split, cache, manifest in (
        ("train", train_cache, train_manifest),
        ("val", val_cache, validation_manifest),
    ):
        evidence = cache.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ScreenError(f"{split} semantic cache lacks evidence records")
        manifest_evidence = evidence.get("manifest")
        train_manifest_evidence = evidence.get("train_manifest")
        pca_evidence = evidence.get("pca")
        if (
            cache.get("split") != split
            or cache.get("manifest_sha256") != manifest.get("sha256")
            or not isinstance(manifest_evidence, Mapping)
            or _base_file_record(manifest_evidence) != _base_file_record(manifest)
            or cache.get("train_manifest_sha256") != train_manifest.get("sha256")
            or not isinstance(train_manifest_evidence, Mapping)
            or _base_file_record(train_manifest_evidence)
            != _base_file_record(train_manifest)
            or not isinstance(pca_evidence, Mapping)
            or pca_evidence.get("sha256") != cache.get("pca_sha256")
        ):
            raise ScreenError(
                f"{split} semantic cache is not bound to the actual frozen manifests/PCA"
            )
    if any(train_cache.get(key) != val_cache.get(key) for key in CACHE_SHARED_UPSTREAM_KEYS):
        raise ScreenError(
            "training and validation semantic caches must share one PCA, "
            "implementation, and upstream teacher identity"
        )


def _manifest_record(
    path: str | Path,
    *,
    split: str,
    expected_clips: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows = read_clip_manifest(resolved, expected_split=split)
    episode_ids = [int(row["episode_index"]) for row in rows]
    if len(rows) != expected_clips or len(
        set(str(row["clip_id"]) for row in rows)
    ) != expected_clips:
        raise ScreenError(f"{split} manifest must contain exactly {expected_clips} unique clips")
    expected_episodes = 8_000 if split == "train" else FROZEN_VALIDATION_CLIPS
    if len(set(episode_ids)) != expected_episodes:
        raise ScreenError(f"{split} manifest must contain exactly {expected_episodes} episodes")
    return resolved, rows, {
        **vlf.file_record(resolved),
        "split": split,
        "clips": len(rows),
        "episodes": len(set(episode_ids)),
        "ordered_clip_ids_sha256": sha256_json([str(row["clip_id"]) for row in rows]),
        "ordered_episode_ids_sha256": sha256_json(episode_ids),
    }


def _construct_dataset(
    manifest: Path,
    data_root: str | Path,
    semantic_cache_root: str | Path,
) -> CausalVJEPA2DroidDataset:
    dataset = CausalVJEPA2DroidDataset(manifest, data_root, semantic_cache_root)
    if len(dataset) < 1:
        raise ScreenError("causal V-JEPA dataset is empty")
    return dataset


def _training_datasets(args: argparse.Namespace):
    train_path, train_rows, train_record = _manifest_record(
        args.train_manifest, split="train", expected_clips=FROZEN_TRAIN_CLIPS
    )
    val_path, val_rows, val_record = _manifest_record(
        args.validation_manifest, split="val", expected_clips=FROZEN_VALIDATION_CLIPS
    )
    if not rows_episode_ids(train_rows).isdisjoint(rows_episode_ids(val_rows)):
        raise ScreenError("training and validation semantic populations overlap by episode")
    train_dataset = _construct_dataset(
        train_path, args.data_root, args.semantic_cache_root
    )
    val_dataset = _construct_dataset(val_path, args.data_root, args.semantic_cache_root)
    train_cache = validated_cache_metadata(train_dataset)
    val_cache = validated_cache_metadata(val_dataset)
    _validate_training_cache_pair(
        train_cache,
        val_cache,
        train_manifest=train_record,
        validation_manifest=val_record,
    )
    return train_dataset, val_dataset, {
        "train": train_record,
        "validation": val_record,
        "semantic_cache": {"train": train_cache, "validation": val_cache},
    }


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        arm="phase1",
        width=args.width,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )


def instantiate_model(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    model, config = vlf.instantiate_model(_model_args(args))
    counts = vlf.count_parameters(model)
    if counts["total"] != FROZEN_MODEL_PARAMETERS:
        raise ScreenError(
            f"model has {counts['total']} parameters, expected {FROZEN_MODEL_PARAMETERS}"
        )
    return model, config


def _validate_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = (
            value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
        )
    # Cache bytes are float16 for compactness.  Interpolation, noise, hashes,
    # and metrics use float32; CUDA model operations remain bfloat16 autocast.
    if isinstance(moved.get("auxiliary_target"), Tensor):
        moved["auxiliary_target"] = moved["auxiliary_target"].float()
    required = {
        "history": (3, 5, 64, 112),
        "future": (3, 8, 64, 112),
        "actions": (16, 7),
        "auxiliary_target": TARGET_SHAPE,
    }
    for key, shape in required.items():
        value = moved.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape[1:]) != shape:
            raise ScreenError(
                f"batch {key} must have trailing shape {shape}, got {getattr(value, 'shape', None)}"
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ScreenError(f"batch {key} must be finite floating point")
    return moved


def semantic_training_step(
    model: nn.Module, batch: Mapping[str, Any]
) -> tuple[Tensor, dict[str, Tensor]]:
    """One exact Phase-1 update whose sole supervised target is semantic."""
    clean = batch["auxiliary_target"]
    batch_size = clean.shape[0]
    clocks = vlf.sample_training_clocks("phase1", batch_size, clean.device)
    if (
        bool(clocks.video_time.ne(0).any())
        or bool(clocks.video_loss_mask.ne(0).any())
        or bool(clocks.auxiliary_loss_mask.ne(1).any())
    ):
        raise ScreenError("shared Phase-1 clock implementation changed")
    # Future RGB is deliberately not used as a model input or loss target.
    # It supplies only the frozen video tensor shape through randn_like.
    noisy_video = torch.randn_like(batch["future"])
    auxiliary_noise = torch.randn_like(clean)
    noisy_auxiliary = vlf.corrupt_clean_time(clean, auxiliary_noise, clocks.auxiliary_time)
    _, auxiliary_x = vlf.model_forward(
        model,
        noisy_video=noisy_video,
        noisy_auxiliary=noisy_auxiliary,
        t_video=clocks.video_time,
        t_auxiliary=clocks.auxiliary_time,
        history=batch["history"],
        actions=batch["actions"],
        condition_on_auxiliary=True,
    )
    per_example = vlf.per_example_x_prediction_flow_mse(
        auxiliary_x,
        noisy_auxiliary,
        clean,
        auxiliary_noise,
        clocks.auxiliary_time,
    )
    auxiliary_loss = vlf.masked_branch_loss(per_example, clocks.auxiliary_loss_mask)
    loss = 0.333 * auxiliary_loss
    return loss, {
        "auxiliary_loss": auxiliary_loss.detach(),
        "weighted_auxiliary_loss": loss.detach(),
        "auxiliary_branch_count": clocks.auxiliary_loss_mask.sum().detach(),
    }


def _training_config(
    args: argparse.Namespace,
    context: vlf.DistributedContext,
    *,
    model_config: Mapping[str, Any],
    datasets: Mapping[str, Any],
    total_updates: int,
) -> dict[str, Any]:
    local_batch = args.global_batch_size // context.world_size
    micro_batch = args.micro_batch_size or local_batch
    config: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "source": _source_record(),
        "entrypoint": vlf.file_record(__file__),
        "dataset_source": _dataset_source_record(),
        "command": args.command,
        "target_kind": TARGET_KIND,
        "cache_artifact_type": CACHE_ARTIFACT_TYPE,
        "target_shape": list(TARGET_SHAPE),
        "clock_convention": vlf.CLOCK_CONVENTION,
        "clean_time_epsilon": vlf.FROZEN_CLEAN_TIME_EPS,
        "updates": total_updates,
        "checkpoint_updates": list(
            (CALIBRATION_UPDATES,) if args.command == "calibrate" else TRAIN_CHECKPOINTS
        ),
        "seed": args.seed,
        "global_batch_size": args.global_batch_size,
        "world_size": context.world_size,
        "local_optimizer_batch_size": local_batch,
        "micro_batch_size_per_rank": micro_batch,
        "gradient_accumulation_steps": local_batch // micro_batch,
        "dtype": "bfloat16",
        "target_storage_dtype": "float16",
        "target_flow_compute_dtype": "float32",
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
        "model": dict(model_config),
        "parameter_count": FROZEN_MODEL_PARAMETERS,
        "datasets": dict(datasets),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "semantic_cache_root": str(
            Path(args.semantic_cache_root).expanduser().resolve()
        ),
        "workers_per_rank": args.workers,
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
    identity_keys = (
        "source",
        "entrypoint",
        "dataset_source",
        "target_kind",
        "cache_artifact_type",
        "target_shape",
        "clock_convention",
        "clean_time_epsilon",
        "seed",
        "global_batch_size",
        "world_size",
        "local_optimizer_batch_size",
        "micro_batch_size_per_rank",
        "gradient_accumulation_steps",
        "dtype",
        "target_storage_dtype",
        "target_flow_compute_dtype",
        "optimizer",
        "ema",
        "phase1_schedule",
        "model",
        "parameter_count",
        "datasets",
        "data_root",
        "semantic_cache_root",
        "workers_per_rank",
    )
    config["experiment_identity_sha256"] = sha256_json(
        {key: config[key] for key in identity_keys}
    )
    return config


def _validate_calibration(
    path: str | Path,
    *,
    identity_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    record = vlf.load_json(path, "semantic-screen calibration record")
    checkpoint = record.get("checkpoint")
    if (
        record.get("schema") != RUN_SCHEMA
        or record.get("status") != "complete"
        or record.get("command") != "calibrate"
        or record.get("completed_updates") != CALIBRATION_UPDATES
        or record.get("nonfinite_updates") != 0
        or record.get("experiment_identity_sha256") != identity_sha256
        or not isinstance(record.get("source"), Mapping)
        or record["source"].get("commit") != source_commit
        or not isinstance(checkpoint, Mapping)
    ):
        raise ScreenError("calibration record does not match this exact 5k run")
    try:
        vlf._verify_embedded_file_records(checkpoint, label="semantic calibration")  # noqa: SLF001
    except vlf.PocError as exc:
        raise ScreenError(str(exc)) from exc
    resolved_config_record = record.get("resolved_config")
    provenance_record = record.get("provenance")
    if not isinstance(resolved_config_record, Mapping):
        raise ScreenError("calibration completion lacks its resolved config")
    try:
        vlf._verify_embedded_file_records(  # noqa: SLF001
            resolved_config_record, label="semantic calibration config"
        )
    except vlf.PocError as exc:
        raise ScreenError(str(exc)) from exc
    calibration_config = vlf.load_json(
        resolved_config_record["path"], "semantic calibration config"
    )
    if not isinstance(provenance_record, Mapping):
        raise ScreenError("calibration completion lacks its provenance record")
    try:
        vlf._verify_embedded_file_records(  # noqa: SLF001
            provenance_record, label="semantic calibration provenance"
        )
    except vlf.PocError as exc:
        raise ScreenError(str(exc)) from exc
    calibration_provenance = vlf.load_json(
        provenance_record["path"], "semantic calibration provenance"
    )
    _validate_run_provenance(
        calibration_provenance,
        schema=RUN_SCHEMA,
        source=calibration_config.get("source", {}),
        config_sha256=sha256_json(calibration_config),
        entrypoint=calibration_config.get("entrypoint", {}),
        command="calibrate",
    )
    calibration_checkpoint = torch.load(
        checkpoint["path"], map_location="cpu", weights_only=False
    )
    calibration_ema = calibration_checkpoint.get("ema")
    if (
        calibration_config.get("schema") != RUN_SCHEMA
        or calibration_config.get("command") != "calibrate"
        or calibration_config.get("updates") != CALIBRATION_UPDATES
        or calibration_config.get("experiment_identity_sha256") != identity_sha256
        or sha256_json(calibration_config) != record.get("resolved_config_sha256")
        or calibration_checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or calibration_checkpoint.get("arm") != "phase1"
        or calibration_checkpoint.get("completed_updates") != CALIBRATION_UPDATES
        or calibration_checkpoint.get("config_sha256")
        != sha256_json(calibration_config)
        or not isinstance(calibration_ema, Mapping)
        or calibration_ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or calibration_ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or calibration_ema.get("num_updates") != CALIBRATION_UPDATES
    ):
        raise ScreenError("calibration checkpoint/config binding is invalid")
    return record


def _assert_distributed_config(
    context: vlf.DistributedContext, config_sha256: str
) -> None:
    hashes = context.gather_objects(config_sha256)
    if len(set(str(value) for value in hashes)) != 1:
        raise ScreenError("ranks resolved different semantic experiment configurations")


def training_command(args: argparse.Namespace) -> int:
    context = vlf.initialize_distributed()
    logger: vlf.LocalAndOptionalWandbLogger | None = None
    try:
        total_updates = (
            CALIBRATION_UPDATES if args.command == "calibrate" else TRAIN_UPDATES
        )
        if args.global_batch_size % context.world_size:
            raise ScreenError("global batch 256 must divide by torchrun world size")
        local_batch = args.global_batch_size // context.world_size
        micro_batch = args.micro_batch_size or local_batch
        if local_batch % micro_batch:
            raise ScreenError("microbatch must divide the per-rank optimizer batch")
        run_dir = vlf.validated_run_dir(
            args.artifact_root, args.run_id, resume=args.resume is not None
        )
        train_dataset, validation_dataset, dataset_records = _training_datasets(args)
        del validation_dataset
        vlf.seed_everything(args.seed, 0)
        model, model_config = instantiate_model(args)
        source = _source_record()
        model.to(context.device)
        optimizer, scheduler = vlf.optimizer_and_scheduler(model, args, total_updates)
        ema = vlf.ModelEMA(model, decay=args.ema_decay)
        config = _training_config(
            args,
            context,
            model_config=model_config,
            datasets=dataset_records,
            total_updates=total_updates,
        )
        config_sha256 = sha256_json(config)
        _assert_distributed_config(context, config_sha256)
        if args.command == "train":
            _validate_calibration(
                args.calibration_record,
                identity_sha256=config["experiment_identity_sha256"],
                source_commit=str(source["commit"]),
            )

        start_update = 0
        prior_wall = 0.0
        if args.resume is not None:
            existing = vlf.load_json(run_dir / "resolved_config.json", "resolved config")
            if sha256_json(existing) != config_sha256:
                raise ScreenError("resume arguments differ from immutable resolved config")
            payload = vlf.load_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                expected_config_sha256=config_sha256,
                context=context,
            )
            start_update = int(payload["completed_updates"])
            prior_wall = float(payload.get("cumulative_optimizer_wall_seconds", 0.0))
            if not 0 <= start_update < total_updates:
                raise ScreenError("resume checkpoint update is outside this run")
            if context.is_primary:
                vlf.reconcile_resume_artifacts(run_dir, start_update)
            context.barrier()
        else:
            if context.is_primary:
                run_dir.mkdir(parents=True, exist_ok=False)
                vlf.atomic_write_json(run_dir / "resolved_config.json", config, exclusive=True)
                vlf.atomic_write_json(
                    run_dir / "provenance.json",
                    {
                        "schema": RUN_SCHEMA,
                        "source": source,
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
            start_update=start_update,
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
        checkpoints = {
            CALIBRATION_UPDATES
        } if args.command == "calibrate" else set(TRAIN_CHECKPOINTS)
        accumulation_steps = local_batch // micro_batch
        loader_iterator = iter(loader)
        nonfinite_updates = 0
        model.train()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        wall_start = time.perf_counter()
        cumulative_wall = prior_wall
        for update in range(start_update + 1, total_updates + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_weighted = torch.zeros((), device=context.device)
            accumulated_raw = torch.zeros((), device=context.device)
            accumulated_count = torch.zeros((), device=context.device)
            for microstep in range(accumulation_steps):
                batch = _validate_batch(next(loader_iterator), context.device)
                sync = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel)
                    and microstep + 1 < accumulation_steps
                    else contextlib.nullcontext()
                )
                with sync, vlf._autocast(context.device):  # noqa: SLF001
                    weighted, telemetry = semantic_training_step(model, batch)
                    loss = weighted / accumulation_steps
                if not bool(torch.isfinite(loss)):
                    nonfinite_updates += 1
                    raise ScreenError(
                        f"nonfinite semantic loss at update {update}, microstep {microstep}"
                    )
                loss.backward()
                accumulated_weighted += weighted.detach() / accumulation_steps
                accumulated_raw += telemetry["auxiliary_loss"] / accumulation_steps
                accumulated_count += telemetry["auxiliary_branch_count"]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                nonfinite_updates += 1
                raise ScreenError(f"nonfinite gradient norm at update {update}")
            optimizer.step()
            scheduler.step()
            ema.update(model)
            packed = context.sum_tensor(
                torch.stack(
                    [
                        accumulated_weighted.float(),
                        accumulated_raw.float(),
                        accumulated_count.float(),
                    ]
                )
            )
            observe = update == 1 or update % args.log_every == 0 or update in checkpoints
            if observe:
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                cumulative_wall = prior_wall + time.perf_counter() - wall_start
                logger.log(
                    {
                        "update": update,
                        "weighted_auxiliary_loss": float(packed[0] / context.world_size),
                        "auxiliary_loss": float(packed[1] / context.world_size),
                        "auxiliary_branch_examples": int(packed[2].item()),
                        "video_loss": 0.0,
                        "gradient_norm": float(gradient_norm.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "cumulative_optimizer_wall_seconds": cumulative_wall,
                        "cumulative_examples_per_second": (
                            update
                            * args.global_batch_size
                            / max(cumulative_wall, 1e-12)
                        ),
                        "peak_gpu_memory_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated(context.device))
                            if context.device.type == "cuda"
                            else None
                        ),
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
        cumulative_wall = prior_wall + time.perf_counter() - wall_start
        if context.is_primary:
            checkpoint_path = run_dir / "checkpoints" / f"update_{total_updates:06d}.pt"
            vlf.atomic_write_json(
                run_dir / "complete.json",
                {
                    "schema": RUN_SCHEMA,
                    "status": "complete",
                    "command": args.command,
                    "target_kind": TARGET_KIND,
                    "completed_updates": total_updates,
                    "nonfinite_updates": nonfinite_updates,
                    "video_loss_enabled": False,
                    "only_supervised_target": "auxiliary_target",
                    "clock_convention": vlf.CLOCK_CONVENTION,
                    "resolved_config_sha256": config_sha256,
                    "experiment_identity_sha256": config[
                        "experiment_identity_sha256"
                    ],
                    "source": source,
                    "resolved_config": vlf.file_record(run_dir / "resolved_config.json"),
                    "provenance": vlf.file_record(run_dir / "provenance.json"),
                    "checkpoint": vlf.file_record(checkpoint_path),
                    "parameter_counts": vlf.count_parameters(vlf.unwrap_model(model)),
                    "cumulative_optimizer_wall_seconds": cumulative_wall,
                    "cumulative_examples_per_second": total_updates
                    * args.global_batch_size
                    / max(cumulative_wall, 1e-12),
                },
                exclusive=True,
            )
        context.barrier()
        return 0
    finally:
        if logger is not None:
            logger.finish()
        vlf.close_distributed(context)


def tensor_sha256(value: Tensor) -> str:
    return vlf.tensor_sha256(value)


def tensor_sha256_by_example(value: Tensor) -> tuple[str, ...]:
    """Hash a batch with one device-to-host transfer, preserving row hashes."""
    if value.ndim < 1:
        raise ValueError("per-example hashing requires a batch dimension")
    host = value.detach().contiguous().cpu()
    return tuple(tensor_sha256(host[item]) for item in range(host.shape[0]))


def sampler_input_record(
    history: Tensor,
    actions: Tensor,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
) -> dict[str, str]:
    return {
        "history_sha256": tensor_sha256(history),
        "actions_sha256": tensor_sha256(actions),
        "initial_video_noise_sha256": tensor_sha256(video_noise),
        "initial_auxiliary_noise_sha256": tensor_sha256(auxiliary_noise),
    }


def sampler_input_sha256(record: Mapping[str, str]) -> str:
    required = (
        "history_sha256",
        "actions_sha256",
        "initial_video_noise_sha256",
        "initial_auxiliary_noise_sha256",
    )
    if set(record) != set(required) or any(
        not isinstance(record[name], str) or not HEX64.fullmatch(record[name])
        for name in required
    ):
        raise ValueError("sampler input record is malformed")
    return sha256_json({"schema": "causal-vjepa2-sampler-input-v1", **dict(record)})


@dataclass(frozen=True)
class SemanticSample:
    prediction: Tensor
    model_calls: int
    call_input_sha256_by_example: tuple[tuple[str, ...], ...]


@torch.inference_mode()
def sample_semantic(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    *,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    steps: int,
) -> SemanticSample:
    """Generate semantic tokens without accepting a clean target argument."""
    if steps < 1:
        raise ValueError("semantic sampling requires at least one transformer call")
    batch = history.shape[0]
    if (
        actions.shape[0] != batch
        or video_noise.shape[0] != batch
        or auxiliary_noise.shape[0] != batch
    ):
        raise ValueError("semantic sampler inputs must share a batch dimension")
    video = video_noise.clone()
    auxiliary = auxiliary_noise.clone()
    video_boundary = video.clone()
    traces: list[list[str]] = [[] for _ in range(batch)]
    host_schedule = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)
    schedule = host_schedule.to(device=auxiliary.device)
    # These tensors are immutable for the whole trajectory.  Hash them once;
    # only the evolving auxiliary state needs a device-to-host hash per call.
    video_hashes = tensor_sha256_by_example(video)
    history_hashes = tensor_sha256_by_example(history)
    action_hashes = tensor_sha256_by_example(actions)
    fixed_hashes = [
        {
            "noisy_video_sha256": video_hashes[item],
            "history_sha256": history_hashes[item],
            "actions_sha256": action_hashes[item],
        }
        for item in range(batch)
    ]
    tv_hash = tensor_sha256(torch.zeros(1, dtype=torch.float32))
    ta_hashes = [
        tensor_sha256(host_schedule[index : index + 1])
        for index in range(steps)
    ]
    calls = 0
    for index in range(steps):
        ta = torch.full((batch,), schedule[index], device=auxiliary.device)
        ta_next = torch.full((batch,), schedule[index + 1], device=auxiliary.device)
        tv = torch.zeros(batch, device=auxiliary.device)
        auxiliary_hashes = tensor_sha256_by_example(auxiliary)
        for item in range(batch):
            payload = {
                "schema": "causal-vjepa2-model-call-input-v1",
                "step_index": index,
                "total_steps": steps,
                "t_video_sha256": tv_hash,
                "t_auxiliary_sha256": ta_hashes[index],
                "noisy_auxiliary_sha256": auxiliary_hashes[item],
                **fixed_hashes[item],
                "auxiliary_fusion": True,
            }
            traces[item].append(sha256_json(payload))
        _, auxiliary_x = vlf.model_forward(
            model,
            noisy_video=video,
            noisy_auxiliary=auxiliary,
            t_video=tv,
            t_auxiliary=ta,
            history=history,
            actions=actions,
            condition_on_auxiliary=True,
        )
        calls += 1
        auxiliary = vlf.clean_time_euler_from_x(
            auxiliary, auxiliary_x, ta, ta_next
        )
        if not torch.equal(video, video_boundary):
            raise ScreenError("video noise changed during semantic-only sampling")
    if calls != steps or any(len(trace) != steps for trace in traces):
        raise ScreenError("actual semantic model calls differ from requested NFE")
    return SemanticSample(
        prediction=auxiliary,
        model_calls=calls,
        call_input_sha256_by_example=tuple(tuple(trace) for trace in traces),
    )


def semantic_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Return ordinary and temporal semantic metrics per example."""
    if prediction.shape != target.shape or tuple(target.shape[1:]) != TARGET_SHAPE:
        raise ValueError(f"semantic tensors must have shape [B,{TARGET_SHAPE}]")
    prediction_f = prediction.float()
    target_f = target.float()
    error = (prediction_f - target_f).square().flatten(1).mean(1)
    power = target_f.square().flatten(1).mean(1).clamp_min(1e-12)
    nmse = error / power
    cosine = torch.nn.functional.cosine_similarity(
        prediction_f, target_f, dim=TARGET_CHANNEL_AXIS, eps=1e-8
    ).flatten(1).mean(1)
    prediction_dt = prediction_f.diff(dim=TEMPORAL_AXIS)
    target_dt = target_f.diff(dim=TEMPORAL_AXIS)
    temporal_error = (prediction_dt - target_dt).square().flatten(1).mean(1)
    temporal_power = target_dt.square().flatten(1).mean(1).clamp_min(1e-12)
    temporal_nmse = temporal_error / temporal_power
    temporal_cosine = torch.nn.functional.cosine_similarity(
        prediction_dt, target_dt, dim=TARGET_CHANNEL_AXIS, eps=1e-8
    ).flatten(1).mean(1)
    return {
        "semantic_nmse": nmse,
        "semantic_token_cosine": cosine,
        "temporal_difference_nmse": temporal_nmse,
        "temporal_difference_token_cosine": temporal_cosine,
        "retained_utility": 1.0 - nmse,
        "temporal_retained_utility": 1.0 - temporal_nmse,
    }


def _configure_deterministic_eval() -> dict[str, Any]:
    expected_env = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
    }
    for name, expected in expected_env.items():
        existing = os.environ.get(name)
        if existing not in (None, expected):
            raise ScreenError(f"{name} differs from the semantic evaluator contract")
        os.environ[name] = expected
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        **{name.lower(): value for name, value in expected_env.items()},
        "autocast": "cuda-bfloat16",
    }


def _load_evaluation_checkpoint(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_config: Mapping[str, Any],
    cache_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.parent.name != "checkpoints":
        raise ScreenError("checkpoint must remain under its immutable run/checkpoints directory")
    training_config_path = checkpoint_path.parent.parent / "resolved_config.json"
    training_complete_path = checkpoint_path.parent.parent / "complete.json"
    training_config = vlf.load_json(training_config_path, "semantic training config")
    training_complete = vlf.load_json(training_complete_path, "semantic training completion")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = _source_record()
    training_datasets = training_config.get("datasets")
    if not isinstance(training_datasets, Mapping):
        raise ScreenError("training config lacks immutable dataset/cache records")
    semantic_caches = training_datasets.get("semantic_cache")
    if not isinstance(semantic_caches, Mapping):
        raise ScreenError("training config lacks train/validation semantic caches")
    _validate_training_cache_pair(
        semantic_caches.get("train", {}),
        semantic_caches.get("validation", {}),
        train_manifest=training_datasets.get("train", {}),
        validation_manifest=training_datasets.get("validation", {}),
    )
    config_cache = semantic_caches.get("validation")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("arm") != "phase1"
        or payload.get("completed_updates") != TRAIN_UPDATES
        or payload.get("model_config") != model_config
        or training_config.get("schema") != RUN_SCHEMA
        or training_config.get("command") != "train"
        or training_config.get("target_kind") != TARGET_KIND
        or training_config.get("cache_artifact_type") != CACHE_ARTIFACT_TYPE
        or training_config.get("target_shape") != list(TARGET_SHAPE)
        or training_config.get("target_storage_dtype") != "float16"
        or training_config.get("target_flow_compute_dtype") != "float32"
        or training_config.get("seed") != FROZEN_SEED
        or training_config.get("model") != model_config
        or training_config.get("source", {}).get("commit") != source["commit"]
        or training_config.get("source", {}).get("dirty") is not False
        or training_config.get("entrypoint") != vlf.file_record(__file__)
        or training_config.get("dataset_source") != _dataset_source_record()
        or sha256_json(training_config) != payload.get("config_sha256")
        or config_cache != cache_metadata
        or training_complete.get("schema") != RUN_SCHEMA
        or training_complete.get("status") != "complete"
        or training_complete.get("command") != "train"
        or training_complete.get("completed_updates") != TRAIN_UPDATES
        or training_complete.get("nonfinite_updates") != 0
        or training_complete.get("only_supervised_target") != "auxiliary_target"
        or training_complete.get("video_loss_enabled") is not False
        or training_complete.get("resolved_config")
        != vlf.file_record(training_config_path)
        or training_complete.get("checkpoint") != vlf.file_record(checkpoint_path)
        or training_complete.get("resolved_config_sha256")
        != sha256_json(training_config)
    ):
        raise ScreenError("checkpoint is not the exact completed causal V-JEPA 2 screen model")
    ema = payload.get("ema")
    if (
        not isinstance(ema, Mapping)
        or ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or ema.get("num_updates") != TRAIN_UPDATES
        or not isinstance(ema.get("shadow"), Mapping)
    ):
        raise ScreenError("checkpoint lacks the exact 5k EMA state")
    model.load_state_dict(ema["shadow"], strict=True)
    return (
        payload,
        training_config,
        vlf.file_record(checkpoint_path),
        vlf.file_record(training_config_path),
    )


def _metric_record(
    *,
    control: str,
    prediction: Tensor,
    target: Tensor,
    item: int,
    nfe: int,
    clip_id: str,
    episode_index: int,
    donor_clip_id: str,
    donor_episode_index: int,
    source_clip_id: str,
    source_history_sha256: str,
    source_actions_sha256: str,
    destination_inputs: Mapping[str, str],
    sample: SemanticSample | None,
    sample_item: int,
    generation_reused_from: str | None,
    target_source_clip_id: str,
    checkpoint_sha256: str,
    training_config_sha256: str,
    metric_values: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    input_record = {
        "history_sha256": source_history_sha256,
        "actions_sha256": source_actions_sha256,
        "initial_video_noise_sha256": destination_inputs[
            "initial_video_noise_sha256"
        ],
        "initial_auxiliary_noise_sha256": destination_inputs[
            "initial_auxiliary_noise_sha256"
        ],
    }
    call_hashes: tuple[str, ...] = ()
    actual_calls = 0
    conceptual_calls = 0
    if sample is not None:
        call_hashes = sample.call_input_sha256_by_example[sample_item]
        conceptual_calls = nfe
        actual_calls = 0 if generation_reused_from is not None else sample.model_calls
    clean_target_entered_sampler = False
    return {
        "clip_id": clip_id,
        "episode_index": episode_index,
        "donor_clip_id": donor_clip_id,
        "donor_episode_index": donor_episode_index,
        "control": control,
        "nfe": nfe,
        "evaluation_seed": FROZEN_EVALUATION_SEED,
        "checkpoint_sha256": checkpoint_sha256,
        "training_config_sha256": training_config_sha256,
        "source_clip_id": source_clip_id,
        "target_source_clip_id": target_source_clip_id,
        **input_record,
        "sampler_input_sha256": sampler_input_sha256(input_record),
        "model_call_input_sha256": list(call_hashes),
        "model_call_input_chain_sha256": sha256_json(list(call_hashes)),
        "conceptual_path_model_calls": conceptual_calls,
        "actual_evaluator_model_calls": actual_calls,
        "generation_reused_from": generation_reused_from,
        "generated_auxiliary_sha256": tensor_sha256(prediction[item]),
        "metric_target_sha256": tensor_sha256(target[item]),
        "generation_deployable": control in {*GENERATED_CONTROLS, "donor_target"},
        "control_deployable": control in GENERATED_CONTROLS,
        "metric_comparison_only": control in {"donor_target", "zero", "oracle_clean"},
        "metric_target_available_at_inference": False,
        "clean_future_target_entered_sampler": clean_target_entered_sampler,
        "teacher_model_calls": 0,
        **{name: float(value[item]) for name, value in metric_values.items()},
    }


def _validate_pair_batch(
    clip_ids: Sequence[str], episode_indices: Sequence[int]
) -> None:
    if len(clip_ids) % 2:
        raise ScreenError("adjacent donor pairs were split across an evaluation batch")
    for index in range(0, len(clip_ids), 2):
        if (
            clip_ids[index] == clip_ids[index + 1]
            or episode_indices[index] == episode_indices[index + 1]
        ):
            raise ScreenError("semantic donor pairs must be clip- and episode-disjoint")


def _summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault((int(record["nfe"]), str(record["control"])), []).append(record)
    metrics = (
        "semantic_nmse",
        "semantic_token_cosine",
        "temporal_difference_nmse",
        "temporal_difference_token_cosine",
        "retained_utility",
        "temporal_retained_utility",
    )
    return [
        {
            "nfe": nfe,
            "control": control,
            "clips": len(group),
            **{
                metric: float(np.mean([float(row[metric]) for row in group]))
                for metric in metrics
            },
        }
        for (nfe, control), group in sorted(groups.items())
    ]


def evaluation_logging_payloads(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Give every evaluation cell a monotonic W&B step without faking its checkpoint."""
    return [
        {
            "update": TRAIN_UPDATES + index,
            "checkpoint_update": TRAIN_UPDATES,
            "evaluation_cell_index": index,
            "event": "causal_vjepa2_semantic_evaluation",
            **dict(cell),
        }
        for index, cell in enumerate(summaries)
    ]


def evaluation_command(args: argparse.Namespace) -> int:
    determinism = _configure_deterministic_eval()
    context = vlf.initialize_distributed()
    logger: vlf.LocalAndOptionalWandbLogger | None = None
    try:
        output_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
        manifest_path, rows, manifest_record = _manifest_record(
            args.manifest, split="val", expected_clips=FROZEN_VALIDATION_CLIPS
        )
        for index in range(0, len(rows), 2):
            if int(rows[index]["episode_index"]) == int(rows[index + 1]["episode_index"]):
                raise ScreenError("adjacent global semantic donors are not episode-disjoint")
        dataset = _construct_dataset(
            manifest_path, args.data_root, args.semantic_cache_root
        )
        cache_metadata = validated_cache_metadata(dataset)
        vlf.seed_everything(args.seed, 0)
        model, model_config = instantiate_model(args)
        model.to(context.device)
        payload, training_config, checkpoint_record, training_config_record = (
            _load_evaluation_checkpoint(
                args,
                model=model,
                model_config=model_config,
                cache_metadata=cache_metadata,
            )
        )
        model.eval()
        local_indexes, local_batches = vlf.paired_rank_evaluation_layout(
            len(dataset),
            args.eval_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        loader = DataLoader(
            torch.utils.data.Subset(dataset, local_indexes),
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
            "dataset_source": _dataset_source_record(),
            "target_kind": TARGET_KIND,
            "cache_artifact_type": CACHE_ARTIFACT_TYPE,
            "target_storage_dtype": "float16",
            "target_flow_compute_dtype": "float32",
            "target_shape": list(TARGET_SHAPE),
            "clock_convention": vlf.CLOCK_CONVENTION,
            "split": "val",
            "validation_clips": FROZEN_VALIDATION_CLIPS,
            "checkpoint": checkpoint_record,
            "training_config": training_config_record,
            "checkpoint_update": int(payload["completed_updates"]),
            "weights": {
                "kind": "ema",
                "decay": vlf.FROZEN_EMA_DECAY,
                "schedule": vlf.FROZEN_EMA_SCHEDULE,
                "updates": TRAIN_UPDATES,
            },
            "manifest": manifest_record,
            "semantic_cache": cache_metadata,
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "semantic_cache_root": str(
                Path(args.semantic_cache_root).expanduser().resolve()
            ),
            "seed": args.seed,
            "nfe_grid": list(NFE_GRID),
            "controls": list(CONTROLS),
            "fixed_clip_noise": True,
            "fixed_noise_key": "sha256(f'{clip_id}:{eval_seed}:video|aux')",
            "donor_rule": "manifest-adjacent xor-1; pairs are episode-disjoint",
            "donor_mapping_sha256": sha256_json(donor_mapping),
            "clean_target_sampler_policy": (
                "forbidden for every generated control; oracle-clean is metric-only"
            ),
            "world_size": context.world_size,
            "eval_batch_size": args.eval_batch_size,
            "determinism": determinism,
            "wandb": {
                "enabled": args.wandb,
                "entity": args.wandb_entity if args.wandb else None,
                "project": args.wandb_project if args.wandb else None,
                "group": None,
                "private_project_acknowledged": args.wandb_private_project_ack,
            },
        }
        config_sha256 = sha256_json(config)
        _assert_distributed_config(context, config_sha256)
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

        local_records: list[dict[str, Any]] = []
        local_actual_batched_calls = 0
        for raw_batch in loader:
            batch = _validate_batch(raw_batch, context.device)
            clip_ids = [str(value) for value in raw_batch["clip_id"]]
            episode_indices = [int(value) for value in raw_batch["episode_index"]]
            _validate_pair_batch(clip_ids, episode_indices)
            donor_index = torch.arange(
                len(clip_ids), device=context.device, dtype=torch.long
            ).bitwise_xor(1)
            donor_ids = [clip_ids[int(index)] for index in donor_index.cpu().tolist()]
            if any(
                donor_mapping[destination] != donor
                for destination, donor in zip(clip_ids, donor_ids, strict=True)
            ):
                raise ScreenError("runtime semantic donor mapping changed")
            video_noise = vlf.stable_noise_like(
                batch["future"], clip_ids, args.seed, "video"
            )
            auxiliary_noise = vlf.stable_noise_like(
                batch["auxiliary_target"], clip_ids, args.seed, "aux"
            )
            own_inputs = [
                sampler_input_record(
                    batch["history"][item],
                    batch["actions"][item],
                    video_noise[item],
                    auxiliary_noise[item],
                )
                for item in range(len(clip_ids))
            ]
            for nfe in NFE_GRID:
                generated: dict[str, SemanticSample] = {}
                sources = {
                    "autonomous": (batch["history"], batch["actions"]),
                    "context_shuffled": (
                        batch["history"].index_select(0, donor_index),
                        batch["actions"].index_select(0, donor_index),
                    ),
                    "history_shuffled": (
                        batch["history"].index_select(0, donor_index),
                        batch["actions"],
                    ),
                    "actions_shuffled": (
                        batch["history"],
                        batch["actions"].index_select(0, donor_index),
                    ),
                }
                for control in GENERATED_CONTROLS:
                    history, actions = sources[control]
                    with vlf._autocast(context.device):  # noqa: SLF001
                        generated[control] = sample_semantic(
                            model,
                            history,
                            actions,
                            video_noise=video_noise,
                            auxiliary_noise=auxiliary_noise,
                            steps=nfe,
                        )
                    local_actual_batched_calls += generated[control].model_calls
                predictions = {
                    **{name: result.prediction for name, result in generated.items()},
                    "donor_target": generated["autonomous"].prediction,
                    "zero": torch.zeros_like(batch["auxiliary_target"]),
                    "oracle_clean": batch["auxiliary_target"],
                }
                for control in CONTROLS:
                    target = (
                        batch["auxiliary_target"].index_select(0, donor_index)
                        if control == "donor_target"
                        else batch["auxiliary_target"]
                    )
                    sample = (
                        generated["autonomous"]
                        if control == "donor_target"
                        else generated.get(control)
                    )
                    metric_values = {
                        name: value.detach().cpu().tolist()
                        for name, value in semantic_metrics(
                            predictions[control], target
                        ).items()
                    }
                    for item, clip_id in enumerate(clip_ids):
                        donor_item = int(donor_index[item])
                        history_source_item = (
                            donor_item
                            if control in {"context_shuffled", "history_shuffled"}
                            else item
                        )
                        action_source_item = (
                            donor_item
                            if control in {"context_shuffled", "actions_shuffled"}
                            else item
                        )
                        source_clip_id = (
                            donor_ids[item]
                            if control == "context_shuffled"
                            else clip_id
                        )
                        local_records.append(
                            _metric_record(
                                control=control,
                                prediction=predictions[control],
                                target=target,
                                item=item,
                                nfe=nfe,
                                clip_id=clip_id,
                                episode_index=episode_indices[item],
                                donor_clip_id=donor_ids[item],
                                donor_episode_index=episode_indices[donor_item],
                                source_clip_id=source_clip_id,
                                source_history_sha256=own_inputs[
                                    history_source_item
                                ]["history_sha256"],
                                source_actions_sha256=own_inputs[
                                    action_source_item
                                ]["actions_sha256"],
                                destination_inputs=own_inputs[item],
                                sample=sample,
                                sample_item=item,
                                generation_reused_from=(
                                    "autonomous" if control == "donor_target" else None
                                ),
                                target_source_clip_id=(
                                    donor_ids[item]
                                    if control == "donor_target"
                                    else clip_id
                                ),
                                checkpoint_sha256=checkpoint_record["sha256"],
                                training_config_sha256=training_config_record["sha256"],
                                metric_values=metric_values,
                            )
                        )

        rank_path = output_dir / "rank_metrics" / f"rank_{context.rank:04d}.jsonl"
        _atomic_jsonl(rank_path, local_records)
        shard_record = {
            "rank": context.rank,
            "records": len(local_records),
            "actual_batched_transformer_calls": local_actual_batched_calls,
            "file": vlf.file_record(rank_path),
        }
        shards = context.gather_objects(shard_record)
        context.barrier()
        if context.is_primary:
            all_records: list[dict[str, Any]] = []
            for shard in sorted(shards, key=lambda value: int(value["rank"])):
                path = Path(shard["file"]["path"])
                if vlf.file_record(path) != shard["file"]:
                    raise ScreenError("rank metric shard changed before aggregation")
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ScreenError("rank metric row is not a JSON object")
                        all_records.append(row)
            all_records.sort(key=lambda row: (row["nfe"], row["control"], row["clip_id"]))
            expected = FROZEN_VALIDATION_CLIPS * len(NFE_GRID) * len(CONTROLS)
            if len(all_records) != expected:
                raise ScreenError(
                    f"evaluation produced {len(all_records)} rows, expected {expected}"
                )
            records_path = output_dir / "per_clip_metrics.jsonl"
            _atomic_jsonl(records_path, all_records)
            summaries = _summaries(all_records)
            summary = {
                "schema": EVALUATION_SCHEMA,
                "status": "complete",
                "target_kind": TARGET_KIND,
                "split": "val",
                "checkpoint": checkpoint_record,
                "training_config": training_config_record,
                "manifest": manifest_record,
                "semantic_cache": cache_metadata,
                "provenance": vlf.file_record(output_dir / "provenance.json"),
                "record_count": len(all_records),
                "cell_count": len(summaries),
                "summaries": summaries,
                "per_clip_metrics": vlf.file_record(records_path),
                "rank_shards": shards,
                "actual_batched_transformer_calls": sum(
                    int(shard["actual_batched_transformer_calls"])
                    for shard in shards
                ),
                "conceptual_path_calls_are_reported_per_clip": True,
                "donor_target_generation_reused_bit_identically": True,
                "clean_future_target_entered_deployable_sampler": False,
                "teacher_model_calls": 0,
                "zero_nmse_reference": 1.0,
                "oracle_clean_nmse_reference": 0.0,
                "retained_utility_definition": "1 - semantic_nmse",
            }
            vlf.atomic_write_json(
                output_dir / "summary.json", summary, exclusive=True
            )
            for payload in evaluation_logging_payloads(summaries):
                logger.log(payload, primary=True)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "train"):
        subparser = subparsers.add_parser(command)
        _add_model_arguments(subparser)
        subparser.add_argument("--artifact-root", required=True)
        subparser.add_argument("--run-id", required=True)
        subparser.add_argument("--data-root", required=True)
        subparser.add_argument("--semantic-cache-root", required=True)
        subparser.add_argument("--train-manifest", required=True)
        subparser.add_argument("--validation-manifest", required=True)
        subparser.add_argument("--seed", type=int, default=FROZEN_SEED)
        subparser.add_argument(
            "--global-batch-size", type=int, default=FROZEN_GLOBAL_BATCH_SIZE
        )
        subparser.add_argument("--micro-batch-size", type=int)
        subparser.add_argument("--workers", type=int, default=4)
        subparser.add_argument(
            "--learning-rate", type=float, default=vlf.FROZEN_LEARNING_RATE
        )
        subparser.add_argument(
            "--warmup-updates", type=int, default=vlf.FROZEN_WARMUP_UPDATES
        )
        subparser.add_argument(
            "--weight-decay", type=float, default=vlf.FROZEN_WEIGHT_DECAY
        )
        subparser.add_argument(
            "--gradient-clip-norm",
            type=float,
            default=vlf.FROZEN_GRADIENT_CLIP_NORM,
        )
        subparser.add_argument(
            "--ema-decay", type=float, default=vlf.FROZEN_EMA_DECAY
        )
        subparser.add_argument("--log-every", type=int, default=10)
        subparser.add_argument("--resume")
        subparser.add_argument(
            "--calibration-record", required=command == "train"
        )
        _add_wandb_arguments(subparser)

    evaluate = subparsers.add_parser("eval")
    _add_model_arguments(evaluate)
    evaluate.add_argument("--artifact-root", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--semantic-cache-root", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--seed", type=int, default=FROZEN_EVALUATION_SEED)
    evaluate.add_argument("--workers", type=int, default=2)
    evaluate.add_argument("--eval-batch-size", type=int, default=8)
    _add_wandb_arguments(evaluate)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.width != vlf.FROZEN_MODEL_WIDTH
        or args.depth != vlf.FROZEN_MODEL_DEPTH
        or args.heads != vlf.FROZEN_MODEL_HEADS
        or args.mlp_ratio != vlf.FROZEN_MODEL_MLP_RATIO
    ):
        raise ScreenError("model contract is frozen at width512/depth12/heads8/MLP4")
    if args.workers < 0:
        raise ScreenError("worker count must be nonnegative")
    if args.command in {"calibrate", "train"}:
        if (
            args.seed != FROZEN_SEED
            or args.global_batch_size != FROZEN_GLOBAL_BATCH_SIZE
            or args.learning_rate != vlf.FROZEN_LEARNING_RATE
            or args.warmup_updates != vlf.FROZEN_WARMUP_UPDATES
            or args.weight_decay != vlf.FROZEN_WEIGHT_DECAY
            or args.gradient_clip_norm != vlf.FROZEN_GRADIENT_CLIP_NORM
            or args.ema_decay != vlf.FROZEN_EMA_DECAY
            or args.log_every < 1
            or (args.micro_batch_size is not None and args.micro_batch_size < 1)
        ):
            raise ScreenError(
                "training is frozen: seed1234, global batch256, AdamW lr5e-5 "
                "betas(.9,.95), wd0, warmup500, clip1, EMA.9999"
            )
        if args.command == "calibrate" and args.resume is not None:
            raise ScreenError("the exact 200-update calibration cannot resume")
    else:
        if args.seed != FROZEN_EVALUATION_SEED:
            raise ScreenError("evaluation seed is frozen at 20260801")
        if args.eval_batch_size < 2 or args.eval_batch_size % 2:
            raise ScreenError("evaluation batch size must be even and at least two")
    wandb_values = (
        args.wandb_entity,
        args.wandb_project,
        args.wandb_private_project_ack,
    )
    if args.wandb != all(bool(value) for value in wandb_values):
        raise ScreenError(
            "W&B is optional; enabling it requires entity, project, and private acknowledgement"
        )
    if args.wandb and (
        args.wandb_entity != "zijiandu"
        or args.wandb_project != "dual-video-diffusion-private"
    ):
        raise ScreenError(
            "this screen is frozen to private zijiandu/dual-video-diffusion-private"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.command in {"calibrate", "train"}:
            return training_command(args)
        return evaluation_command(args)
    except (ScreenError, vlf.PocError, ValueError, OSError) as exc:
        print(f"Causal V-JEPA 2 semantic screen error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
