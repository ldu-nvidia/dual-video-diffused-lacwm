#!/usr/bin/env python3
"""Fail-closed paired gate for the causal V-JEPA 2 semantic screen.

Scientific failure is a valid result and exits zero.  Invalid, incomplete, or
changed evidence exits nonzero and never publishes a gate record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    read_clip_manifest,
    sha256_file,
)
from robot_wm.datasets.droid.causal_vjepa2 import (  # noqa: E402
    CACHE_ARTIFACT_TYPE,
    TARGET_KIND as CAUSAL_TARGET_KIND,
)


RUN_SCHEMA = "causal-vjepa2-semantic-screen-run-v1"
EVALUATION_SCHEMA = "causal-vjepa2-semantic-screen-evaluation-v1"
GATE_SCHEMA = "causal-vjepa2-semantic-screen-gate-v1"
TARGET_KIND = CAUSAL_TARGET_KIND
TARGET_SHAPE = [48, 8, 8, 14]
NFE_GRID = (1, 2, 4, 8, 12, 20, 25)
GATE_NFE_GRID = (1, 2, 4, 8, 12)
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
CLIPS = 890
TRAIN_CLIPS = 64_000
TRAIN_EPISODES = 8_000
CALIBRATION_UPDATES = 200
TRAIN_UPDATES = 5_000
TRAIN_CHECKPOINTS = [500, 1_000, 2_000, 5_000]
TRAIN_SEED = 1234
EVALUATION_SEED = 20260801
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_BASE_SEED = 20260801
METRICS = (
    "semantic_nmse",
    "semantic_token_cosine",
    "temporal_difference_nmse",
    "temporal_difference_token_cosine",
    "retained_utility",
    "temporal_retained_utility",
)
HEX64 = re.compile(r"[0-9a-f]{64}")
APPROVED_ROOTS = (Path("/lustre"), Path("/mnt/data1"), Path("/mnt/data2"))
CLOCK_CONVENTION = "clean-time: t=0 noise, t=1 clean; velocity=clean-noise"
CLEAN_TIME_EPSILON = 0.05
GLOBAL_BATCH_SIZE = 256
MODEL_PARAMETER_COUNT = 41_963_760
EMA_SCHEDULE = "min(target_decay,(1+completed_updates)/(10+completed_updates))-v1"
PHASE1_SCHEDULE = {
    "video_time": 0.0,
    "video_loss": "disabled",
    "auxiliary_logit_normal_mean": -1.2,
    "auxiliary_logit_normal_std": 1.0,
    "auxiliary_loss_coefficient": 0.333,
    "loss_mask_normalization": "unchanged_global_batch",
}
PRODUCTION_NUMERICAL_CONTRACT = {
    "encoder_device": "cuda",
    "encoder_dtype": "bfloat16",
    "encoder_batch_size": 1,
    "pca_device": "cuda",
    "pca_algorithm": "exact-centered-covariance-eigh",
    "pca_covariance_dtype": "float32",
    "pca_tf32": False,
}
MODEL_CONFIG = {
    "video_channels": 3,
    "future_frames": 8,
    "history_frames": 5,
    "height": 64,
    "width": 112,
    "patch_size": [1, 8, 8],
    "aux_channels": 48,
    "aux_grid": None,
    "action_steps": 16,
    "action_dim": 7,
    "hidden_size": 512,
    "depth": 12,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "parameter_matched_video_only": False,
    "initialization": "latent-forcing-zero-adaln-and-output-heads-v1",
}
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
TRAINING_IDENTITY_KEYS = (
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


class GateError(RuntimeError):
    """Gate evidence is incomplete, inconsistent, or mutable."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file_local(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    return sha256_file(path, chunk_bytes=chunk_bytes)


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise GateError(f"required evidence file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file_local(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid metric JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise GateError(f"metric line {line_number} is not an object")
            rows.append(value)
    return rows


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_path(path: str | Path, evaluation_root: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise GateError(f"gate output already exists: {output}")
    if _is_relative_to(output, evaluation_root) or _is_relative_to(output, REPO_ROOT):
        raise GateError("gate output must be outside the evaluation root and Git repository")
    if not any(_is_relative_to(output, root) for root in APPROVED_ROOTS):
        raise GateError("gate output must be under /lustre, /mnt/data1, or /mnt/data2")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def atomic_publish_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GateError(f"gate output appeared while publishing: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def git_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _verify_file_record(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise GateError(f"{label} file record is missing")
    if not isinstance(record.get("path"), str) or not HEX64.fullmatch(
        str(record.get("sha256", ""))
    ):
        raise GateError(f"{label} file record is malformed")
    actual = file_record(record["path"])
    if actual["sha256"] != record["sha256"] or (
        record.get("bytes") is not None and actual["bytes"] != record["bytes"]
    ):
        raise GateError(f"{label} evidence changed")
    return actual


def _verify_embedded_file_records(
    value: Any, *, label: str, _seen: set[tuple[Any, ...]] | None = None
) -> int:
    """Recursively re-hash each explicit ``{path, sha256}`` cache record."""
    seen = set() if _seen is None else _seen
    verified = 0
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            identity = (value.get("path"), value.get("sha256"), value.get("bytes"))
            if identity not in seen:
                _verify_file_record(value, label)
                seen.add(identity)
                verified += 1
        for child in value.values():
            verified += _verify_embedded_file_records(child, label=label, _seen=seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            verified += _verify_embedded_file_records(child, label=label, _seen=seen)
    return verified


def _snapshot_embedded_file_records(
    value: Any,
    *,
    label: str,
    _seen: set[tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Re-hash and retain every unique explicit ``{path, sha256}`` record."""
    seen = set() if _seen is None else _seen
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            identity = (value.get("path"), value.get("sha256"), value.get("bytes"))
            if identity not in seen:
                records.append(_verify_file_record(value, label))
                seen.add(identity)
        for child in value.values():
            records.extend(
                _snapshot_embedded_file_records(child, label=label, _seen=seen)
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            records.extend(
                _snapshot_embedded_file_records(child, label=label, _seen=seen)
            )
    return records


def _base_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: record.get(name) for name in ("path", "sha256", "bytes")}


def _validate_runtime_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GateError("run provenance lacks a runtime record")
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
        raise GateError("run provenance runtime inventory is malformed")
    if any(
        not isinstance(runtime.get(name), str) or not runtime[name]
        for name in ("utc", "hostname", "python", "platform")
    ):
        raise GateError("run provenance runtime identity is malformed")
    if runtime.get("torch_cuda") is not None and not isinstance(
        runtime.get("torch_cuda"), str
    ):
        raise GateError("run provenance CUDA runtime is malformed")
    if not isinstance(runtime.get("cuda_available"), bool):
        raise GateError("run provenance CUDA availability is malformed")
    for name in ("packages", "slurm"):
        mapping = runtime.get(name)
        if not isinstance(mapping, Mapping) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in mapping.items()
        ):
            raise GateError(f"run provenance {name} inventory is malformed")
    inventory = runtime.get("gpu_inventory")
    if not isinstance(inventory, list) or any(
        not isinstance(item, str) for item in inventory
    ):
        raise GateError("run provenance GPU inventory is malformed")
    return runtime


def _validate_run_provenance(
    provenance: Any,
    *,
    schema: str,
    source: Mapping[str, Any],
    config_sha256: str,
    entrypoint: Mapping[str, Any],
    command: str,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise GateError("run provenance must be a JSON mapping")
    value = dict(provenance)
    argv = value.get("command")
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
        value.get("schema") != schema
        or value.get("source") != source
        or value.get("resolved_config_sha256") != config_sha256
        or value.get("secrets_persisted") is not False
        or not command_valid
    ):
        raise GateError("run provenance is not bound to its exact config/source/argv")
    _validate_runtime_provenance(value.get("runtime"))
    return value


def _validate_frozen_training_config(
    config: Mapping[str, Any], *, command: str
) -> None:
    expected_updates = CALIBRATION_UPDATES if command == "calibrate" else TRAIN_UPDATES
    expected_checkpoints = (
        [CALIBRATION_UPDATES] if command == "calibrate" else TRAIN_CHECKPOINTS
    )
    optimizer = config.get("optimizer")
    ema = config.get("ema")
    wandb = config.get("wandb")
    world_size = config.get("world_size")
    local_batch = config.get("local_optimizer_batch_size")
    micro_batch = config.get("micro_batch_size_per_rank")
    accumulation = config.get("gradient_accumulation_steps")
    topology_valid = (
        isinstance(world_size, int)
        and not isinstance(world_size, bool)
        and world_size >= 1
        and GLOBAL_BATCH_SIZE % world_size == 0
        and isinstance(local_batch, int)
        and not isinstance(local_batch, bool)
        and local_batch == GLOBAL_BATCH_SIZE // world_size
        and isinstance(micro_batch, int)
        and not isinstance(micro_batch, bool)
        and 1 <= micro_batch <= local_batch
        and local_batch % micro_batch == 0
        and isinstance(accumulation, int)
        and not isinstance(accumulation, bool)
        and accumulation == local_batch // micro_batch
    )
    wandb_valid = (
        isinstance(wandb, Mapping)
        and isinstance(wandb.get("enabled"), bool)
        and wandb.get("group") is None
        and (
            wandb.get("enabled") is False
            or (
                wandb.get("entity") == "zijiandu"
                and wandb.get("project") == "dual-video-diffusion-private"
                and wandb.get("private_project_acknowledged") is True
            )
        )
    )
    identity_payload = {
        key: config.get(key) for key in TRAINING_IDENTITY_KEYS
    }
    if (
        config.get("schema") != RUN_SCHEMA
        or config.get("command") != command
        or config.get("updates") != expected_updates
        or config.get("checkpoint_updates") != expected_checkpoints
        or config.get("seed") != TRAIN_SEED
        or config.get("global_batch_size") != GLOBAL_BATCH_SIZE
        or not topology_valid
        or config.get("dtype") != "bfloat16"
        or config.get("target_storage_dtype") != "float16"
        or config.get("target_flow_compute_dtype") != "float32"
        or config.get("target_kind") != TARGET_KIND
        or config.get("cache_artifact_type") != CACHE_ARTIFACT_TYPE
        or config.get("target_shape") != TARGET_SHAPE
        or config.get("clock_convention") != CLOCK_CONVENTION
        or config.get("clean_time_epsilon") != CLEAN_TIME_EPSILON
        or canonical_json(optimizer)
        != canonical_json({
            "name": "AdamW",
            "learning_rate": 5e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_updates": 500,
            "after_warmup": "constant",
            "gradient_clip_norm": 1.0,
        })
        or canonical_json(ema)
        != canonical_json({
            "decay": 0.9999,
            "schedule": EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        })
        or canonical_json(config.get("phase1_schedule"))
        != canonical_json(PHASE1_SCHEDULE)
        or canonical_json(config.get("model")) != canonical_json(MODEL_CONFIG)
        or config.get("parameter_count") != MODEL_PARAMETER_COUNT
        or not isinstance(config.get("workers_per_rank"), int)
        or isinstance(config.get("workers_per_rank"), bool)
        or config.get("workers_per_rank") < 0
        or not isinstance(config.get("data_root"), str)
        or not Path(config["data_root"]).is_absolute()
        or not isinstance(config.get("semantic_cache_root"), str)
        or not Path(config["semantic_cache_root"]).is_absolute()
        or not wandb_valid
        or config.get("experiment_identity_sha256")
        != sha256_json(identity_payload)
        or (command == "calibrate" and config.get("calibration_record") is not None)
        or (command == "train" and not isinstance(config.get("calibration_record"), Mapping))
    ):
        raise GateError(f"{command} config violates the exact frozen training contract")


def _validate_training_cache_pair(
    train_cache: Any,
    val_cache: Any,
    *,
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(train_cache, Mapping) or not isinstance(val_cache, Mapping):
        raise GateError("training config lacks both semantic cache records")
    snapshots: list[dict[str, Any]] = []
    for split, count, cache, manifest in (
        ("train", TRAIN_CLIPS, train_cache, train_manifest),
        ("val", CLIPS, val_cache, validation_manifest),
    ):
        split_snapshots = _snapshot_embedded_file_records(
            cache, label=f"{split} semantic cache"
        )
        if len(split_snapshots) < 4:
            raise GateError(f"{split} semantic cache lacks complete provenance")
        snapshots.extend(split_snapshots)
        artifact_identity = cache.get("artifact_identity")
        runtime = cache.get("runtime")
        evidence = cache.get("evidence")
        manifest_evidence = evidence.get("manifest") if isinstance(evidence, Mapping) else None
        train_manifest_evidence = (
            evidence.get("train_manifest") if isinstance(evidence, Mapping) else None
        )
        pca_evidence = evidence.get("pca") if isinstance(evidence, Mapping) else None
        if (
            not isinstance(artifact_identity, Mapping)
            or cache.get("cache_id") != sha256_json(artifact_identity)
            or any(cache.get(key) != value for key, value in artifact_identity.items())
            or not isinstance(cache.get("base_droid"), Mapping)
            or not isinstance(runtime, Mapping)
            or set(runtime) != {"python", "torch", "cuda", "numpy"}
            or any(not isinstance(value, str) for value in runtime.values())
            or canonical_json(cache.get("numerical_contract"))
            != canonical_json(PRODUCTION_NUMERICAL_CONTRACT)
            or cache.get("artifact_type") != CACHE_ARTIFACT_TYPE
            or cache.get("target_kind") != TARGET_KIND
            or cache.get("complete") is not True
            or cache.get("split") != split
            or cache.get("clip_count") != count
            or cache.get("auxiliary_target_shape") != TARGET_SHAPE
            or cache.get("target_shape") != [count, *TARGET_SHAPE]
            or cache.get("target_dtype") != "float16"
            or cache.get("teacher_size") != [384, 672]
            or cache.get("teacher_frames") != 16
            or cache.get("last_temporal_token") != 7
            or cache.get("pooled_token_grid") != [8, 14]
            or cache.get("protected_test_access") is not False
            or cache.get("allowed_splits") != ["train", "val"]
            or cache.get("test_rows_extracted") != 0
            or cache.get("manifest_sha256") != manifest.get("sha256")
            or not isinstance(manifest_evidence, Mapping)
            or _base_file_record(manifest_evidence) != _base_file_record(manifest)
            or cache.get("train_manifest_sha256") != train_manifest.get("sha256")
            or not isinstance(train_manifest_evidence, Mapping)
            or _base_file_record(train_manifest_evidence)
            != _base_file_record(train_manifest)
            or not isinstance(pca_evidence, Mapping)
            or pca_evidence.get("sha256") != cache.get("pca_sha256")
            or not isinstance(cache.get("implementation"), Mapping)
        ):
            raise GateError(
                f"{split} semantic cache is not bound to the exact manifest/PCA contract"
            )
        _hash(cache.get("pca_sha256"), "pca_sha256")
    if any(train_cache.get(key) != val_cache.get(key) for key in CACHE_SHARED_UPSTREAM_KEYS):
        raise GateError(
            "train/validation caches do not share one PCA, implementation, "
            "and upstream teacher identity"
        )
    unique = {
        (record["path"], record["sha256"], record["bytes"]): record
        for record in snapshots
    }
    return [unique[key] for key in sorted(unique)]


def _finite(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GateError(f"{name} must be finite for every metric row")
    return float(value)


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise GateError(f"{name} must be a full lowercase SHA-256")
    return value


def _bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_BASE_SEED}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def relative_improvement(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(reference.mean())
    if denominator <= 0.0:
        return float("nan")
    return float((denominator - candidate.mean()) / denominator)


def mean_first_minus_second(first: np.ndarray, second: np.ndarray) -> float:
    return float(first.mean() - second.mean())


def paired_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    label: str,
    statistic: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    if first.shape != (CLIPS,) or second.shape != (CLIPS,):
        raise GateError(f"paired bootstrap requires exactly {CLIPS} aligned clips")
    estimate = float(statistic(first, second))
    if not math.isfinite(estimate):
        raise GateError(f"nonfinite point statistic: {label}")
    generator = np.random.default_rng(_bootstrap_seed(label))
    values = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    chunk = 1_000
    for start in range(0, BOOTSTRAP_SAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_SAMPLES - start)
        indexes = generator.integers(0, CLIPS, size=(count, CLIPS), endpoint=False)
        first_means = first[indexes].mean(axis=1)
        second_means = second[indexes].mean(axis=1)
        if statistic is relative_improvement:
            with np.errstate(divide="ignore", invalid="ignore"):
                values[start : start + count] = (
                    second_means - first_means
                ) / second_means
        elif statistic is mean_first_minus_second:
            values[start : start + count] = first_means - second_means
        else:
            for offset in range(count):
                values[start + offset] = statistic(
                    first[indexes[offset]], second[indexes[offset]]
                )
    if not np.isfinite(values).all():
        raise GateError(f"nonfinite paired bootstrap: {label}")
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) * 0.5
    low, high = np.quantile(values, (tail, 1.0 - tail), method="linear")
    return {
        "label": label,
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "seed": _bootstrap_seed(label),
    }


def gate_cell(
    nfe: int, values: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    autonomous = values["autonomous"]
    checks: list[dict[str, Any]] = []

    def add(identifier: str, estimate: float, rule: str, passed: bool, **extra: Any) -> None:
        checks.append(
            {
                "id": identifier,
                "estimate": float(estimate),
                "rule": rule,
                "passed": bool(passed),
                **extra,
            }
        )

    nmse = float(autonomous["semantic_nmse"].mean())
    cosine = float(autonomous["semantic_token_cosine"].mean())
    temporal_nmse = float(autonomous["temporal_difference_nmse"].mean())
    add("autonomous_semantic_nmse", nmse, "mean <= 0.50", nmse <= 0.50)
    add("autonomous_token_cosine", cosine, "mean >= 0.70", cosine >= 0.70)
    add(
        "autonomous_temporal_difference_nmse",
        temporal_nmse,
        "mean <= 0.50",
        temporal_nmse <= 0.50,
    )

    for reference_control in ("donor_target", "context_shuffled"):
        for metric in ("semantic_nmse", "temporal_difference_nmse"):
            label = f"nfe={nfe}|{metric}|autonomous_vs_{reference_control}|relative"
            result = paired_bootstrap(
                autonomous[metric],
                values[reference_control][metric],
                label=label,
                statistic=relative_improvement,
            )
            add(
                f"{metric}_vs_{reference_control}",
                result["estimate"],
                "relative improvement >= 0.05 and paired 95% CI low > 0",
                result["estimate"] >= 0.05 and result["ci_low"] > 0.0,
                bootstrap=result,
            )
        label = (
            f"nfe={nfe}|semantic_token_cosine|autonomous_vs_"
            f"{reference_control}|difference"
        )
        result = paired_bootstrap(
            autonomous["semantic_token_cosine"],
            values[reference_control]["semantic_token_cosine"],
            label=label,
            statistic=mean_first_minus_second,
        )
        add(
            f"semantic_token_cosine_vs_{reference_control}",
            result["estimate"],
            "paired mean advantage 95% CI low > 0",
            result["ci_low"] > 0.0,
            bootstrap=result,
        )
    return {
        "nfe": nfe,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "diagnostics": {
            control: {
                metric: float(values[control][metric].mean()) for metric in METRICS
            }
            for control in ("history_shuffled", "actions_shuffled", "zero", "oracle_clean")
        },
    }


def _sampler_input_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        "schema": "causal-vjepa2-sampler-input-v1",
        "history_sha256": _hash(row.get("history_sha256"), "history_sha256"),
        "actions_sha256": _hash(row.get("actions_sha256"), "actions_sha256"),
        "initial_video_noise_sha256": _hash(
            row.get("initial_video_noise_sha256"), "initial_video_noise_sha256"
        ),
        "initial_auxiliary_noise_sha256": _hash(
            row.get("initial_auxiliary_noise_sha256"),
            "initial_auxiliary_noise_sha256",
        ),
    }
    return sha256_json(payload)


def _validate_row_common(
    row: Mapping[str, Any],
    *,
    nfe: int,
    control: str,
    clip_id: str,
    episode_index: int,
    checkpoint_sha256: str,
    training_config_sha256: str,
) -> None:
    if (
        row.get("clip_id") != clip_id
        or not isinstance(row.get("episode_index"), int)
        or isinstance(row.get("episode_index"), bool)
        or row.get("episode_index") != episode_index
        or row.get("control") != control
        or not isinstance(row.get("nfe"), int)
        or isinstance(row.get("nfe"), bool)
        or row.get("nfe") != nfe
        or row.get("evaluation_seed") != EVALUATION_SEED
        or row.get("checkpoint_sha256") != checkpoint_sha256
        or row.get("training_config_sha256") != training_config_sha256
        or row.get("teacher_model_calls") != 0
        or row.get("clean_future_target_entered_sampler") is not False
    ):
        raise GateError(f"row identity/provenance mismatch: {nfe}/{control}/{clip_id}")
    for name in (
        "donor_clip_id",
        "source_clip_id",
        "target_source_clip_id",
    ):
        if not isinstance(row.get(name), str):
            raise GateError(f"row lacks {name}: {nfe}/{control}/{clip_id}")
    if not isinstance(row.get("donor_episode_index"), int) or isinstance(
        row.get("donor_episode_index"), bool
    ):
        raise GateError(f"row donor episode is malformed: {nfe}/{control}/{clip_id}")
    for name in METRICS:
        _finite(row, name)
    if not math.isclose(
        _finite(row, "retained_utility"),
        1.0 - _finite(row, "semantic_nmse"),
        rel_tol=0.0,
        abs_tol=1e-7,
    ) or not math.isclose(
        _finite(row, "temporal_retained_utility"),
        1.0 - _finite(row, "temporal_difference_nmse"),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise GateError("retained utility does not equal 1-NMSE")
    if row.get("sampler_input_sha256") != _sampler_input_sha256(row):
        raise GateError("sampler input digest is not reproducible from exact input hashes")
    _hash(row.get("generated_auxiliary_sha256"), "generated_auxiliary_sha256")
    _hash(row.get("metric_target_sha256"), "metric_target_sha256")
    call_hashes = row.get("model_call_input_sha256")
    if not isinstance(call_hashes, list) or any(
        not isinstance(value, str) or not HEX64.fullmatch(value)
        for value in call_hashes
    ):
        raise GateError("model call input hash trace is malformed")
    if row.get("model_call_input_chain_sha256") != sha256_json(call_hashes):
        raise GateError("model call hash-chain digest changed")
    if control in GENERATED_CONTROLS:
        expected = (nfe, nfe, nfe, None, True, True, False)
    elif control == "donor_target":
        expected = (nfe, 0, nfe, "autonomous", True, False, True)
    else:
        expected = (0, 0, 0, None, False, False, True)
    actual = (
        row.get("conceptual_path_model_calls"),
        row.get("actual_evaluator_model_calls"),
        len(call_hashes),
        row.get("generation_reused_from"),
        row.get("generation_deployable"),
        row.get("control_deployable"),
        row.get("metric_comparison_only"),
    )
    if any(
        not isinstance(row.get(name), int) or isinstance(row.get(name), bool)
        for name in ("conceptual_path_model_calls", "actual_evaluator_model_calls")
    ):
        raise GateError(f"model-call counts are malformed: {nfe}/{control}/{clip_id}")
    if actual != expected:
        raise GateError(f"model-call accounting mismatch: {nfe}/{control}/{clip_id}")
    if row.get("metric_target_available_at_inference") is not False:
        raise GateError("evaluation ground-truth metric target was mislabeled inference-available")


def _summary_index(summary: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    values = summary.get("summaries")
    if not isinstance(values, list):
        raise GateError("summary cells are missing")
    index: dict[tuple[int, str], Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise GateError("summary cell is malformed")
        key = (value.get("nfe"), value.get("control"))
        if key in index:
            raise GateError("duplicate summary cell")
        if (
            not isinstance(key[0], int)
            or isinstance(key[0], bool)
            or not isinstance(key[1], str)
        ):
            raise GateError("summary cell key is malformed")
        index[(key[0], key[1])] = value
    expected = {(nfe, control) for nfe in NFE_GRID for control in CONTROLS}
    if index.keys() != expected:
        raise GateError("summary cell inventory is incomplete or contains extras")
    return index


def _validate_manifest_inventory(
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    clips: int,
    episodes: int,
) -> None:
    clip_ids = [str(row["clip_id"]) for row in rows]
    episode_ids = [int(row["episode_index"]) for row in rows]
    if (
        len(rows) != clips
        or len(set(clip_ids)) != clips
        or len(set(episode_ids)) != episodes
        or record.get("split") != split
        or record.get("clips") != clips
        or record.get("episodes") != episodes
        or record.get("ordered_clip_ids_sha256") != sha256_json(clip_ids)
        or record.get("ordered_episode_ids_sha256") != sha256_json(episode_ids)
    ):
        raise GateError(f"{split} manifest inventory differs from the frozen population")


def analyze_evaluation(evaluation_root: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root).expanduser().resolve()
    if not root.is_dir():
        raise GateError(f"evaluation root is missing: {root}")
    evidence_before = {
        "resolved_config": file_record(root / "resolved_config.json"),
        "provenance": file_record(root / "provenance.json"),
        "summary": file_record(root / "summary.json"),
    }
    config = load_json(root / "resolved_config.json", "evaluation config")
    provenance = load_json(root / "provenance.json", "evaluation provenance")
    summary = load_json(root / "summary.json", "evaluation summary")
    source = git_record()
    expected_determinism = {
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "nvidia_tf32_override": "0",
        "torch_allow_tf32_cublas_override": "0",
        "autocast": "cuda-bfloat16",
    }
    evaluation_wandb = config.get("wandb")
    evaluation_weights = config.get("weights")
    if source["dirty"]:
        raise GateError("gate analysis requires clean committed source")
    if (
        config.get("schema") != EVALUATION_SCHEMA
        or summary.get("schema") != EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or provenance.get("schema") != EVALUATION_SCHEMA
        or config.get("source", {}).get("commit") != source["commit"]
        or config.get("source", {}).get("dirty") is not False
        or provenance.get("source") != config.get("source")
        or provenance.get("resolved_config_sha256") != sha256_json(config)
        or config.get("target_kind") != TARGET_KIND
        or config.get("cache_artifact_type") != CACHE_ARTIFACT_TYPE
        or config.get("target_storage_dtype") != "float16"
        or config.get("target_flow_compute_dtype") != "float32"
        or config.get("target_shape") != TARGET_SHAPE
        or config.get("clock_convention") != CLOCK_CONVENTION
        or config.get("split") != "val"
        or config.get("validation_clips") != CLIPS
        or config.get("checkpoint_update") != TRAIN_UPDATES
        or config.get("seed") != EVALUATION_SEED
        or config.get("nfe_grid") != list(NFE_GRID)
        or config.get("controls") != list(CONTROLS)
        or not isinstance(evaluation_weights, Mapping)
        or evaluation_weights.get("kind") != "ema"
        or evaluation_weights.get("decay") != 0.9999
        or evaluation_weights.get("schedule")
        != "min(target_decay,(1+completed_updates)/(10+completed_updates))-v1"
        or evaluation_weights.get("updates") != TRAIN_UPDATES
        or config.get("determinism") != expected_determinism
        or config.get("fixed_clip_noise") is not True
        or config.get("fixed_noise_key")
        != "sha256(f'{clip_id}:{eval_seed}:video|aux')"
        or config.get("donor_rule")
        != "manifest-adjacent xor-1; pairs are episode-disjoint"
        or not isinstance(config.get("data_root"), str)
        or not Path(config["data_root"]).is_absolute()
        or not isinstance(config.get("semantic_cache_root"), str)
        or not Path(config["semantic_cache_root"]).is_absolute()
        or not isinstance(evaluation_wandb, Mapping)
        or evaluation_wandb.get("group") is not None
        or (
            evaluation_wandb.get("enabled") is True
            and (
                evaluation_wandb.get("entity") != "zijiandu"
                or evaluation_wandb.get("project")
                != "dual-video-diffusion-private"
                or evaluation_wandb.get("private_project_acknowledged") is not True
            )
        )
        or config.get("clean_target_sampler_policy")
        != "forbidden for every generated control; oracle-clean is metric-only"
        or summary.get("clean_future_target_entered_deployable_sampler") is not False
        or summary.get("teacher_model_calls") != 0
        or summary.get("donor_target_generation_reused_bit_identically") is not True
        or summary.get("zero_nmse_reference") != 1.0
        or summary.get("oracle_clean_nmse_reference") != 0.0
        or summary.get("provenance") != evidence_before["provenance"]
    ):
        raise GateError("evaluation configuration/summary violates the frozen semantic screen")

    _validate_run_provenance(
        provenance,
        schema=EVALUATION_SCHEMA,
        source=config.get("source", {}),
        config_sha256=sha256_json(config),
        entrypoint=config.get("entrypoint", {}),
        command="eval",
    )

    checkpoint = _verify_file_record(config.get("checkpoint"), "checkpoint")
    _verify_file_record(config.get("entrypoint"), "evaluation entrypoint")
    _verify_file_record(config.get("dataset_source"), "causal dataset source")
    training_config_record = _verify_file_record(
        config.get("training_config"), "training config"
    )
    training_config = load_json(training_config_record["path"], "training config")
    _validate_frozen_training_config(training_config, command="train")
    datasets = training_config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise GateError("training config lacks immutable dataset records")
    train_manifest_record = _verify_file_record(
        datasets.get("train"), "training manifest"
    )
    training_validation_manifest_record = _verify_file_record(
        datasets.get("validation"), "training validation manifest"
    )
    semantic_caches = datasets.get("semantic_cache")
    if not isinstance(semantic_caches, Mapping):
        raise GateError("training config lacks semantic cache records")
    cache_evidence_records = _validate_training_cache_pair(
        semantic_caches.get("train"),
        semantic_caches.get("validation"),
        train_manifest=datasets["train"],
        validation_manifest=datasets["validation"],
    )
    if (
        training_config.get("source") != config.get("source")
        or semantic_caches.get("validation") != config.get("semantic_cache")
        or datasets.get("validation") != config.get("manifest")
        or summary.get("checkpoint") != config.get("checkpoint")
        or summary.get("training_config") != config.get("training_config")
        or summary.get("semantic_cache") != config.get("semantic_cache")
    ):
        raise GateError("evaluation is not bound to the exact auxiliary-only training run")
    _verify_file_record(training_config.get("entrypoint"), "training entrypoint")
    _verify_file_record(training_config.get("dataset_source"), "training dataset source")
    calibration_record = _verify_file_record(
        training_config.get("calibration_record"), "calibration completion"
    )
    calibration = load_json(calibration_record["path"], "calibration completion")
    calibration_provenance_path = (
        Path(calibration_record["path"]).parent / "provenance.json"
    )
    calibration_provenance_record = file_record(calibration_provenance_path)
    calibration_provenance = load_json(
        calibration_provenance_path, "calibration provenance"
    )
    if (
        calibration.get("schema") != RUN_SCHEMA
        or calibration.get("status") != "complete"
        or calibration.get("command") != "calibrate"
        or calibration.get("completed_updates") != 200
        or calibration.get("nonfinite_updates") != 0
        or calibration.get("only_supervised_target") != "auxiliary_target"
        or calibration.get("video_loss_enabled") is not False
        or calibration.get("clock_convention") != CLOCK_CONVENTION
        or calibration.get("parameter_counts")
        != {"total": MODEL_PARAMETER_COUNT, "trainable": MODEL_PARAMETER_COUNT}
        or calibration.get("experiment_identity_sha256")
        != training_config.get("experiment_identity_sha256")
        or calibration.get("source") != training_config.get("source")
        or calibration.get("provenance") != calibration_provenance_record
        or _verify_embedded_file_records(
            calibration.get("resolved_config"), label="calibration config"
        )
        < 1
        or _verify_embedded_file_records(
            calibration.get("checkpoint"), label="calibration checkpoint"
        )
        < 1
    ):
        raise GateError("training is not bound to the exact valid 200-update calibration")
    calibration_config_record = calibration["resolved_config"]
    calibration_checkpoint_record = calibration["checkpoint"]
    calibration_config_file = _verify_file_record(
        calibration_config_record, "calibration config"
    )
    calibration_checkpoint_file = _verify_file_record(
        calibration_checkpoint_record, "calibration checkpoint"
    )
    calibration_config = load_json(
        calibration_config_file["path"], "calibration config"
    )
    _validate_frozen_training_config(calibration_config, command="calibrate")
    _validate_run_provenance(
        calibration_provenance,
        schema=RUN_SCHEMA,
        source=calibration_config.get("source", {}),
        config_sha256=sha256_json(calibration_config),
        entrypoint=calibration_config.get("entrypoint", {}),
        command="calibrate",
    )
    calibration_checkpoint = torch.load(
        calibration_checkpoint_file["path"], map_location="cpu", weights_only=False
    )
    calibration_ema = (
        calibration_checkpoint.get("ema")
        if isinstance(calibration_checkpoint, Mapping)
        else None
    )
    if (
        calibration_config.get("experiment_identity_sha256")
        != training_config.get("experiment_identity_sha256")
        or sha256_json(calibration_config)
        != calibration.get("resolved_config_sha256")
        or not isinstance(calibration_checkpoint, Mapping)
        or calibration_checkpoint.get("schema")
        != "video-latent-forcing-poc-checkpoint-v1"
        or calibration_checkpoint.get("arm") != "phase1"
        or calibration_checkpoint.get("completed_updates") != 200
        or canonical_json(calibration_checkpoint.get("model_config"))
        != canonical_json(MODEL_CONFIG)
        or calibration_checkpoint.get("config_sha256")
        != sha256_json(calibration_config)
        or not isinstance(calibration_ema, Mapping)
        or calibration_ema.get("decay") != 0.9999
        or calibration_ema.get("schedule") != EMA_SCHEDULE
        or calibration_ema.get("num_updates") != 200
        or not isinstance(calibration_ema.get("shadow"), Mapping)
    ):
        raise GateError("calibration checkpoint/config binding is invalid")
    del calibration_checkpoint
    checkpoint_payload = torch.load(
        checkpoint["path"], map_location="cpu", weights_only=False
    )
    checkpoint_ema = (
        checkpoint_payload.get("ema")
        if isinstance(checkpoint_payload, Mapping)
        else None
    )
    if (
        not isinstance(checkpoint_payload, Mapping)
        or checkpoint_payload.get("schema")
        != "video-latent-forcing-poc-checkpoint-v1"
        or checkpoint_payload.get("arm") != "phase1"
        or checkpoint_payload.get("completed_updates") != TRAIN_UPDATES
        or canonical_json(checkpoint_payload.get("model_config"))
        != canonical_json(MODEL_CONFIG)
        or checkpoint_payload.get("config_sha256") != sha256_json(training_config)
        or not isinstance(checkpoint_ema, Mapping)
        or checkpoint_ema.get("decay") != 0.9999
        or checkpoint_ema.get("schedule")
        != "min(target_decay,(1+completed_updates)/(10+completed_updates))-v1"
        or checkpoint_ema.get("num_updates") != TRAIN_UPDATES
        or not isinstance(checkpoint_ema.get("shadow"), Mapping)
    ):
        raise GateError("bound checkpoint is not the exact 5k Phase-1 EMA model")
    complete_path = Path(checkpoint["path"]).parent.parent / "complete.json"
    complete_record = file_record(complete_path)
    complete = load_json(complete_path, "training completion")
    training_provenance_path = complete_path.parent / "provenance.json"
    training_provenance_record = file_record(training_provenance_path)
    training_provenance = load_json(training_provenance_path, "training provenance")
    if (
        complete.get("schema") != RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("command") != "train"
        or complete.get("completed_updates") != TRAIN_UPDATES
        or complete.get("nonfinite_updates") != 0
        or complete.get("only_supervised_target") != "auxiliary_target"
        or complete.get("video_loss_enabled") is not False
        or complete.get("resolved_config") != training_config_record
        or complete.get("provenance") != training_provenance_record
        or complete.get("checkpoint") != checkpoint
        or complete.get("resolved_config_sha256") != sha256_json(training_config)
        or complete.get("parameter_counts")
        != {"total": MODEL_PARAMETER_COUNT, "trainable": MODEL_PARAMETER_COUNT}
    ):
        raise GateError("training completion does not bind exact config/checkpoint evidence")
    _validate_run_provenance(
        training_provenance,
        schema=RUN_SCHEMA,
        source=training_config.get("source", {}),
        config_sha256=sha256_json(training_config),
        entrypoint=training_config.get("entrypoint", {}),
        command="train",
    )

    rows_train_manifest = read_clip_manifest(
        train_manifest_record["path"], expected_split="train"
    )
    _validate_manifest_inventory(
        datasets["train"],
        rows_train_manifest,
        split="train",
        clips=TRAIN_CLIPS,
        episodes=TRAIN_EPISODES,
    )
    manifest_record = _verify_file_record(config.get("manifest"), "validation manifest")
    rows_manifest = read_clip_manifest(manifest_record["path"], expected_split="val")
    _validate_manifest_inventory(
        config["manifest"],
        rows_manifest,
        split="val",
        clips=CLIPS,
        episodes=CLIPS,
    )
    if len(rows_manifest) != CLIPS:
        raise GateError("validation manifest must contain exactly 890 clips")
    clip_ids = [str(row["clip_id"]) for row in rows_manifest]
    episode_by_clip = {
        str(row["clip_id"]): int(row["episode_index"]) for row in rows_manifest
    }
    if len(set(clip_ids)) != CLIPS or len(set(episode_by_clip.values())) != CLIPS:
        raise GateError("validation must contain one unique clip from each of 890 episodes")
    if not set(episode_by_clip.values()).isdisjoint(
        int(row["episode_index"]) for row in rows_train_manifest
    ):
        raise GateError("training and validation populations overlap by episode")
    donor = {clip_ids[index]: clip_ids[index ^ 1] for index in range(CLIPS)}
    if any(
        episode_by_clip[clip_id] == episode_by_clip[donor[clip_id]]
        for clip_id in clip_ids
    ):
        raise GateError("adjacent donor population is not episode-disjoint")
    if config.get("donor_mapping_sha256") != sha256_json(donor):
        raise GateError("donor mapping digest differs from the validation manifest")

    records_file = _verify_file_record(summary.get("per_clip_metrics"), "per-clip metrics")
    records = _read_jsonl(Path(records_file["path"]))
    expected_count = CLIPS * len(NFE_GRID) * len(CONTROLS)
    if (
        len(records) != expected_count
        or summary.get("record_count") != expected_count
        or summary.get("cell_count") != len(NFE_GRID) * len(CONTROLS)
    ):
        raise GateError("semantic evaluation record count is incomplete")
    index: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (row.get("nfe"), row.get("control"), row.get("clip_id"))
        if key in index:
            raise GateError(f"duplicate semantic metric key: {key}")
        if (
            not isinstance(key[0], int)
            or isinstance(key[0], bool)
            or not isinstance(key[1], str)
            or not isinstance(key[2], str)
            or key[0] not in NFE_GRID
            or key[1] not in CONTROLS
            or key[2] not in episode_by_clip
        ):
            raise GateError(f"unexpected semantic metric key: {key}")
        index[(int(key[0]), str(key[1]), str(key[2]))] = row
    expected_keys = {
        (nfe, control, clip_id)
        for nfe in NFE_GRID
        for control in CONTROLS
        for clip_id in clip_ids
    }
    if index.keys() != expected_keys:
        raise GateError("semantic metric key inventory is incomplete or contains extras")

    # The NFE frontier is paired only when every trajectory for one destination
    # starts from the same clip-addressed noise and immutable context/target.
    # Anchor these values once, outside the NFE loop, so a consistently changed
    # row set at one NFE cannot satisfy only within-cell control checks.
    clip_anchors: dict[str, dict[str, str]] = {}
    for clip_id in clip_ids:
        autonomous = index[NFE_GRID[0], "autonomous", clip_id]
        clip_anchors[clip_id] = {
            "history_sha256": _hash(
                autonomous.get("history_sha256"), "history_sha256"
            ),
            "actions_sha256": _hash(
                autonomous.get("actions_sha256"), "actions_sha256"
            ),
            "initial_video_noise_sha256": _hash(
                autonomous.get("initial_video_noise_sha256"),
                "initial_video_noise_sha256",
            ),
            "initial_auxiliary_noise_sha256": _hash(
                autonomous.get("initial_auxiliary_noise_sha256"),
                "initial_auxiliary_noise_sha256",
            ),
            "metric_target_sha256": _hash(
                autonomous.get("metric_target_sha256"), "metric_target_sha256"
            ),
        }

    values: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for nfe in NFE_GRID:
        values[nfe] = {}
        for control in CONTROLS:
            values[nfe][control] = {
                metric: np.asarray(
                    [_finite(index[nfe, control, clip_id], metric) for clip_id in clip_ids],
                    dtype=np.float64,
                )
                for metric in METRICS
            }
        for clip_id in clip_ids:
            own_episode = episode_by_clip[clip_id]
            donor_id = donor[clip_id]
            donor_episode = episode_by_clip[donor_id]
            by_control = {
                control: index[nfe, control, clip_id] for control in CONTROLS
            }
            for control, row in by_control.items():
                _validate_row_common(
                    row,
                    nfe=nfe,
                    control=control,
                    clip_id=clip_id,
                    episode_index=own_episode,
                    checkpoint_sha256=checkpoint["sha256"],
                    training_config_sha256=training_config_record["sha256"],
                )
                if (
                    row.get("donor_clip_id") != donor_id
                    or row.get("donor_episode_index") != donor_episode
                ):
                    raise GateError("row donor identity differs from manifest-adjacent pairing")
            autonomous = by_control["autonomous"]
            donor_target = by_control["donor_target"]
            donor_autonomous = index[nfe, "autonomous", donor_id]
            if (
                donor_target.get("generated_auxiliary_sha256")
                != autonomous.get("generated_auxiliary_sha256")
                or donor_target.get("model_call_input_sha256")
                != autonomous.get("model_call_input_sha256")
                or donor_target.get("sampler_input_sha256")
                != autonomous.get("sampler_input_sha256")
                or donor_target.get("target_source_clip_id") != donor_id
                or donor_target.get("metric_target_sha256")
                != donor_autonomous.get("metric_target_sha256")
            ):
                raise GateError("donor-target is not metric-only reuse of autonomous generation")

            own_anchor = clip_anchors[clip_id]
            donor_anchor = clip_anchors[donor_id]
            own_history = own_anchor["history_sha256"]
            own_actions = own_anchor["actions_sha256"]
            donor_history = donor_anchor["history_sha256"]
            donor_actions = donor_anchor["actions_sha256"]
            own_video_noise = own_anchor["initial_video_noise_sha256"]
            own_aux_noise = own_anchor["initial_auxiliary_noise_sha256"]
            expected_inputs = {
                "autonomous": (own_history, own_actions),
                "donor_target": (own_history, own_actions),
                "context_shuffled": (donor_history, donor_actions),
                "history_shuffled": (donor_history, own_actions),
                "actions_shuffled": (own_history, donor_actions),
                "zero": (own_history, own_actions),
                "oracle_clean": (own_history, own_actions),
            }
            for control, row in by_control.items():
                expected_target = (
                    donor_anchor["metric_target_sha256"]
                    if control == "donor_target"
                    else own_anchor["metric_target_sha256"]
                )
                if (
                    row.get("initial_video_noise_sha256") != own_video_noise
                    or row.get("initial_auxiliary_noise_sha256") != own_aux_noise
                    or row.get("metric_target_sha256") != expected_target
                ):
                    raise GateError(
                        "cross-NFE fixed noise/target anchor changed: "
                        f"{nfe}/{control}/{clip_id}"
                    )
                if (
                    (row.get("history_sha256"), row.get("actions_sha256"))
                    != expected_inputs[control]
                    or (
                        control != "donor_target"
                        and row.get("target_source_clip_id") != clip_id
                    )
                    or (
                        control != "donor_target"
                        and row.get("metric_target_sha256")
                        != autonomous.get("metric_target_sha256")
                    )
                    or row.get("source_clip_id")
                    != (donor_id if control == "context_shuffled" else clip_id)
                ):
                    raise GateError(f"control input/target alignment changed: {control}")
            zero = by_control["zero"]
            oracle = by_control["oracle_clean"]
            if (
                not math.isclose(_finite(zero, "semantic_nmse"), 1.0, rel_tol=0.0, abs_tol=1e-7)
                or not math.isclose(
                    _finite(zero, "temporal_difference_nmse"),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    _finite(oracle, "semantic_nmse"), 0.0, rel_tol=0.0, abs_tol=1e-9
                )
                or not math.isclose(
                    _finite(oracle, "temporal_difference_nmse"),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or _finite(oracle, "semantic_token_cosine") < 0.99999
                or _finite(oracle, "temporal_difference_token_cosine") < 0.99999
                or oracle.get("generated_auxiliary_sha256")
                != oracle.get("metric_target_sha256")
            ):
                raise GateError("zero/oracle analytic diagnostics are inconsistent")

    summaries = _summary_index(summary)
    for key, cell in summaries.items():
        nfe, control = key
        if cell.get("clips") != CLIPS:
            raise GateError(f"summary cell {key} does not contain 890 clips")
        for metric in METRICS:
            reported = _finite(cell, metric)
            recomputed = float(values[nfe][control][metric].mean())
            if not math.isclose(reported, recomputed, rel_tol=1e-12, abs_tol=1e-12):
                raise GateError(f"summary mean changed: {nfe}/{control}/{metric}")

    shards = summary.get("rank_shards")
    world_size = config.get("world_size")
    batch_size = config.get("eval_batch_size")
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size < 1
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 2
        or batch_size % 2
    ):
        raise GateError("evaluation topology is malformed")
    if not isinstance(shards, list) or len(shards) != world_size:
        raise GateError("rank-shard provenance is incomplete")
    shards_by_rank: dict[int, Mapping[str, Any]] = {}
    for shard in shards:
        rank = shard.get("rank") if isinstance(shard, Mapping) else None
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or rank >= world_size
            or rank in shards_by_rank
        ):
            raise GateError("rank-shard identity is malformed")
        shards_by_rank[rank] = shard
    if shards_by_rank.keys() != set(range(world_size)):
        raise GateError("rank-shard inventory is incomplete")

    actual_calls = 0
    shard_records = 0
    total_batches = 0
    pair_count = CLIPS // 2
    shard_union: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    shard_file_records: list[dict[str, Any]] = []
    for expected_rank in range(world_size):
        shard = shards_by_rank[expected_rank]
        shard_file = _verify_file_record(
            shard.get("file"), f"rank {expected_rank} metrics"
        )
        shard_file_records.append(shard_file)
        shard_rows = _read_jsonl(Path(shard_file["path"]))
        assigned_pairs = list(range(expected_rank, pair_count, world_size))
        if not assigned_pairs:
            raise GateError("an evaluation rank received no donor pairs")
        assigned_clip_ids = {
            clip_ids[index]
            for pair in assigned_pairs
            for index in (2 * pair, 2 * pair + 1)
        }
        expected_shard_keys = {
            (nfe, control, clip_id)
            for nfe in NFE_GRID
            for control in CONTROLS
            for clip_id in assigned_clip_ids
        }
        local_index: dict[tuple[int, str, str], Mapping[str, Any]] = {}
        for row in shard_rows:
            key = (row.get("nfe"), row.get("control"), row.get("clip_id"))
            if key in local_index or key not in expected_shard_keys:
                raise GateError(f"rank {expected_rank} shard has duplicate/unassigned row")
            normalized_key = (int(key[0]), str(key[1]), str(key[2]))
            merged_row = index.get(normalized_key)
            if merged_row is None or canonical_json(row) != canonical_json(merged_row):
                raise GateError(f"rank {expected_rank} shard differs from merged metrics")
            local_index[normalized_key] = row
        if local_index.keys() != expected_shard_keys:
            raise GateError(f"rank {expected_rank} shard row inventory is incomplete")
        if any(key in shard_union for key in local_index):
            raise GateError("rank metric shards overlap")
        shard_union.update(local_index)
        local_batches = math.ceil(len(assigned_clip_ids) / batch_size)
        derived_calls = len(GENERATED_CONTROLS) * sum(NFE_GRID) * local_batches
        if (
            shard.get("records") != len(shard_rows)
            or shard.get("actual_batched_transformer_calls") != derived_calls
        ):
            raise GateError(f"rank {expected_rank} shard sidecar accounting is not derived")
        shard_records += len(shard_rows)
        actual_calls += derived_calls
        total_batches += local_batches
    if shard_union.keys() != index.keys() or any(
        canonical_json(shard_union[key]) != canonical_json(index[key]) for key in index
    ):
        raise GateError("rank shards are not the exact disjoint union of merged metrics")
    if (
        actual_calls != summary.get("actual_batched_transformer_calls")
        or shard_records != expected_count
    ):
        raise GateError("rank-shard call/record accounting differs from the summary")
    expected_actual_calls = len(GENERATED_CONTROLS) * sum(NFE_GRID) * total_batches
    if actual_calls != expected_actual_calls:
        raise GateError(
            f"actual transformer calls {actual_calls} differ from exact {expected_actual_calls}"
        )

    criteria = {
        str(nfe): gate_cell(nfe, values[nfe]) for nfe in GATE_NFE_GRID
    }
    passing = [nfe for nfe in GATE_NFE_GRID if criteria[str(nfe)]["passed"]]
    selected = min(passing) if passing else None
    evidence_after = {
        "resolved_config": file_record(root / "resolved_config.json"),
        "provenance": file_record(root / "provenance.json"),
        "summary": file_record(root / "summary.json"),
    }
    if (
        evidence_after != evidence_before
        or file_record(records_file["path"]) != records_file
        or file_record(checkpoint["path"]) != checkpoint
        or file_record(training_config_record["path"]) != training_config_record
        or file_record(calibration_record["path"]) != calibration_record
        or file_record(calibration_config_file["path"]) != calibration_config_file
        or file_record(calibration_checkpoint_file["path"])
        != calibration_checkpoint_file
        or file_record(calibration_provenance_path)
        != calibration_provenance_record
        or file_record(complete_path) != complete_record
        or file_record(training_provenance_path) != training_provenance_record
        or file_record(train_manifest_record["path"]) != train_manifest_record
        or file_record(training_validation_manifest_record["path"])
        != training_validation_manifest_record
        or any(
            file_record(record["path"]) != record
            for record in cache_evidence_records
        )
        or any(file_record(record["path"]) != record for record in shard_file_records)
    ):
        raise GateError("semantic evidence changed while it was being analyzed")
    payload: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "phase": "prefix_causal_vjepa2_semantic_predictor_screen",
        "status": "pass" if selected is not None else "fail",
        "frozen": True,
        "validation_only": True,
        "scientific_failure_is_valid": True,
        "source_commit": source["commit"],
        "target_kind": TARGET_KIND,
        "checkpoint": checkpoint,
        "training_config": training_config_record,
        "training_provenance": training_provenance_record,
        "calibration": calibration_record,
        "calibration_provenance": calibration_provenance_record,
        "evaluation": {
            "root": str(root),
            **evidence_before,
            "per_clip_metrics": records_file,
        },
        "criteria_by_nfe": criteria,
        "gate_nfe_grid": list(GATE_NFE_GRID),
        "selected_nfe": selected,
        "selected_nfe_pair": [selected, 0] if selected is not None else None,
        "selection_rule": (
            "smallest NFE <=12 passing absolute own-target quality and both "
            "episode-disjoint donor/context paired advantages"
        ),
        "phase3_semantic_gate_passed": selected is not None,
        "history_and_action_shuffles_are_diagnostic_only": True,
        "retained_utility_definition": "1 - semantic_nmse",
        "bootstrap": {
            "method": "paired clip bootstrap with replacement",
            "samples": BOOTSTRAP_SAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "base_seed": BOOTSTRAP_BASE_SEED,
        },
    }
    payload["decision_sha256"] = sha256_json(payload)
    return payload


def gate(evaluation_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root).expanduser().resolve()
    destination = validate_output_path(output, root)
    payload = analyze_evaluation(root)
    atomic_publish_exclusive(destination, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = gate(args.evaluation_root, args.output)
        print(
            canonical_json(
                {
                    "status": payload["status"],
                    "phase3_semantic_gate_passed": payload[
                        "phase3_semantic_gate_passed"
                    ],
                    "selected_nfe": payload["selected_nfe"],
                    "output": str(Path(args.output).expanduser().resolve()),
                }
            )
        )
        # A valid negative scientific result is not an operational error.
        return 0
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Causal V-JEPA 2 semantic gate error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
