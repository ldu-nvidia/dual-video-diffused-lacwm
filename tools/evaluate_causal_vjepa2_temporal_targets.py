#!/usr/bin/env python3
"""Generated-only development evaluator for the temporal-target DOE.

The sampler in this module cannot accept future RGB, a clean semantic target,
or a teacher.  Clean targets enter only the metric path after an autonomous
trajectory has completed.  The executable evaluates the exact update-5,000
EMA checkpoint of one preregistered arm on the frozen 890-clip validation
split and writes immutable per-clip records for paired analysis.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import causal_vjepa2_cache_bridge as cache_bridge  # noqa: E402
from tools import causal_vjepa2_temporal_targets as temporal  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402


EVALUATION_SCHEMA = "causal-vjepa2-temporal-target-evaluation-v1"
IMPLEMENTATION_REGISTRATION_SCHEMA = (
    "causal-vjepa2-temporal-doe-implementation-registration-v1"
)
EVALUATION_SEED = screen.FROZEN_EVALUATION_SEED
NFE_GRID = screen.NFE_GRID
ACTION_PERMUTATION_OFFSETS = (1, 17, 101)
ACTION_CONTROLS = tuple(
    f"actions_shuffled_offset_{offset:03d}" for offset in ACTION_PERMUTATION_OFFSETS
)
GENERATED_CONTROLS = (
    "autonomous",
    "context_shuffled",
    "history_shuffled",
    *ACTION_CONTROLS,
)
CONTROLS = (
    "autonomous",
    "donor_target",
    "context_shuffled",
    "history_shuffled",
    *ACTION_CONTROLS,
    "zero",
    "oracle_clean",
)
DOE_ARMS = ("ABS", "ABS-T", "DELTA", "DELTA-T", "DELTA-R")
ARM_CONTRACTS: dict[str, dict[str, Any]] = {
    "ABS": {
        "target_mode": "absolute",
        "temporal_weight": 0.0,
        "rollin_probability": 0.0,
        "normalization": False,
    },
    "ABS-T": {
        "target_mode": "absolute",
        "temporal_weight": 1.0,
        "rollin_probability": 0.0,
        "normalization": True,
    },
    "DELTA": {
        "target_mode": "delta_pack",
        "temporal_weight": 0.0,
        "rollin_probability": 0.0,
        "normalization": True,
    },
    "DELTA-T": {
        "target_mode": "delta_pack",
        "temporal_weight": 1.0,
        "rollin_probability": 0.0,
        "normalization": True,
    },
    "DELTA-R": {
        "target_mode": "delta_pack",
        "temporal_weight": 0.0,
        "rollin_probability": 0.5,
        "normalization": True,
    },
}
PRIMARY_METRICS = (
    "semantic_nmse",
    "semantic_token_cosine",
    "temporal_difference_nmse",
    "temporal_difference_token_cosine",
    "retained_utility",
    "temporal_retained_utility",
)
PACKED_METRICS = tuple(f"packed_{name}" for name in PRIMARY_METRICS)
FORBIDDEN_SAMPLER_TERMS = ("future", "target", "teacher", "oracle", "clean")
INFERENCE_CRITICAL_PATHS = (
    "tools/video_latent_forcing_poc.py",
    "tools/causal_vjepa2_screen.py",
    "tools/causal_vjepa2_temporal_targets.py",
    "tools/causal_vjepa2_cache_bridge.py",
    "robot_wm/modeling/video_latent_forcing",
)
FROZEN_D1_NORMALIZATION_SHA256 = (
    "157b9dedbd95c8381eda5a69dfa1c646c79b8e408604975c369a8e1e1f54cff9"
)
PREREGISTRATION_COMMIT = "be7e76d97543ccc97253e76d1d234abe1c5c4387"
PREREGISTRATION_PATH = (
    "docs/experiments/VIDEO_LATENT_FORCING_TEMPORAL_FOLLOWUP_PROTOCOL.md"
)
PREREGISTRATION_GIT_OBJECT = "b85e924e4adfb7cab6a26cb35778f00cfa5601b5"
PREREGISTRATION_CONTENT_SHA256 = (
    "d1d5b22853c598fdfa62984f12975220534341eaf5b6f492d8757a1b3b2a7947"
)
FROZEN_TRAIN_WORLD_SIZE = 8
FROZEN_LOCAL_BATCH_SIZE = screen.FROZEN_GLOBAL_BATCH_SIZE // FROZEN_TRAIN_WORLD_SIZE


class TemporalEvaluationError(RuntimeError):
    """A temporal-target evaluation or provenance contract failed closed."""


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TemporalEvaluationError(
            f"cannot validate training/evaluator source compatibility: {' '.join(arguments)}"
        ) from exc


def training_source_compatibility(
    training_source: Any, current_source: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a clean ancestor whose inference-critical Git objects are exact."""
    if not isinstance(training_source, Mapping):
        raise TemporalEvaluationError("training config lacks a source record")
    training_commit = training_source.get("commit")
    current_commit = current_source.get("commit")
    if (
        not isinstance(training_commit, str)
        or not isinstance(current_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", training_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", current_commit) is None
        or training_source.get("dirty") is not False
        or current_source.get("dirty") is not False
    ):
        raise TemporalEvaluationError("training and evaluator sources must be clean commits")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                training_commit,
                current_commit,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TemporalEvaluationError(
            "recorded training commit is not an ancestor of evaluator source"
        ) from exc
    paths: dict[str, Any] = {}
    for path in INFERENCE_CRITICAL_PATHS:
        training_object = _git("rev-parse", f"{training_commit}:{path}")
        evaluator_object = _git("rev-parse", f"{current_commit}:{path}")
        if training_object != evaluator_object:
            raise TemporalEvaluationError(
                f"inference-critical source changed after training: {path}"
            )
        paths[path] = {
            "training_object": training_object,
            "evaluator_object": evaluator_object,
            "unchanged": True,
        }
    return {
        "training_commit": training_commit,
        "evaluator_commit": current_commit,
        "training_is_ancestor": True,
        "inference_critical_paths_unchanged": True,
        "paths": paths,
    }


def preregistration_attestation(current_source: Mapping[str, Any]) -> dict[str, Any]:
    current_commit = str(current_source.get("commit", ""))
    if re.fullmatch(r"[0-9a-f]{40}", current_commit) is None:
        raise TemporalEvaluationError("current source does not identify a clean commit")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                PREREGISTRATION_COMMIT,
                current_commit,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TemporalEvaluationError("preregistration is not an ancestor") from exc
    registered_object = _git(
        "rev-parse", f"{PREREGISTRATION_COMMIT}:{PREREGISTRATION_PATH}"
    )
    current_object = _git("rev-parse", f"{current_commit}:{PREREGISTRATION_PATH}")
    current_record = vlf.file_record(REPO_ROOT / PREREGISTRATION_PATH)
    if (
        registered_object != PREREGISTRATION_GIT_OBJECT
        or current_object != PREREGISTRATION_GIT_OBJECT
        or current_record["sha256"] != PREREGISTRATION_CONTENT_SHA256
    ):
        raise TemporalEvaluationError("frozen preregistration bytes changed")
    return {
        "commit": PREREGISTRATION_COMMIT,
        "path": PREREGISTRATION_PATH,
        "git_object": PREREGISTRATION_GIT_OBJECT,
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "current_file": current_record,
        "current_source_is_descendant": True,
        "bytes_unchanged": True,
    }


def training_doe_common_identity(training_config: Mapping[str, Any]) -> str:
    """Hash every training field except the preregistered arm mechanism."""
    common = json.loads(screen.canonical_json(training_config))
    for name in (
        "doe_arm",
        "promotion_status",
        "target_mode",
        "normalization",
        "experiment_identity_sha256",
    ):
        common.pop(name, None)
    loss = common.get("loss")
    rollin = common.get("self_rollin")
    if not isinstance(loss, dict) or not isinstance(rollin, dict):
        raise TemporalEvaluationError("training config lacks DOE mechanism fields")
    loss.pop("normalized_temporal_velocity_weight", None)
    rollin.pop("probability", None)
    return screen.sha256_json(
        {"schema": "causal-vjepa2-temporal-doe-common-training-v1", **common}
    )


def _implementation_files() -> dict[str, Any]:
    paths = {
        "evaluator": Path(__file__),
        "analyzer": REPO_ROOT / "tools" / "analyze_causal_vjepa2_temporal_targets.py",
        "trainer": Path(temporal.__file__),
        "semantic_screen": Path(screen.__file__),
        "cache_bridge": Path(cache_bridge.__file__),
        "shared_sampler": Path(vlf.__file__),
    }
    return {name: vlf.file_record(path) for name, path in paths.items()}


def create_implementation_registration(path: str | Path) -> dict[str, Any]:
    output = Path(path).expanduser().resolve()
    source = screen._source_record()  # noqa: SLF001
    unsigned = {
        "schema": IMPLEMENTATION_REGISTRATION_SCHEMA,
        "status": "frozen_before_candidate_metrics",
        "source": source,
        "implementation_files": _implementation_files(),
        "preregistration": preregistration_attestation(source),
        "d1_normalization_sha256": FROZEN_D1_NORMALIZATION_SHA256,
        "candidate_metrics_visible_at_registration": False,
        "protected_test_accessed": False,
    }
    payload = {**unsigned, "identity_sha256": screen.sha256_json(unsigned)}
    output.parent.mkdir(parents=True, exist_ok=True)
    vlf.atomic_write_json(output, payload, exclusive=True)
    return payload


def load_implementation_registration(
    path: str | Path, *, current_source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = vlf.load_json(resolved, "temporal DOE implementation registration")
    unsigned = dict(payload)
    identity = unsigned.pop("identity_sha256", None)
    if (
        payload.get("schema") != IMPLEMENTATION_REGISTRATION_SCHEMA
        or payload.get("status") != "frozen_before_candidate_metrics"
        or payload.get("source") != current_source
        or payload.get("implementation_files") != _implementation_files()
        or payload.get("preregistration") != preregistration_attestation(current_source)
        or payload.get("d1_normalization_sha256")
        != FROZEN_D1_NORMALIZATION_SHA256
        or payload.get("candidate_metrics_visible_at_registration") is not False
        or payload.get("protected_test_accessed") is not False
        or identity != screen.sha256_json(unsigned)
    ):
        raise TemporalEvaluationError(
            "implementation registration is not exact or predates no metrics"
        )
    return payload, vlf.file_record(resolved)


def action_control(offset: int) -> str:
    if offset not in ACTION_PERMUTATION_OFFSETS:
        raise ValueError(f"unregistered action permutation offset: {offset}")
    return f"actions_shuffled_offset_{offset:03d}"


def action_permutation_indices(size: int, offset: int) -> tuple[int, ...]:
    """Return the frozen manifest rotation used for factual action attribution."""
    if size < 2 or offset not in ACTION_PERMUTATION_OFFSETS or offset >= size:
        raise ValueError("action permutation requires a registered nonzero offset")
    result = tuple((index + offset) % size for index in range(size))
    if len(set(result)) != size or any(index == source for index, source in enumerate(result)):
        raise TemporalEvaluationError("action permutation is not a fixed-point-free bijection")
    return result


def validate_action_permutations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate and describe the three episode-disjoint manifest rotations."""
    clip_ids = [str(row["clip_id"]) for row in rows]
    episodes = [int(row["episode_index"]) for row in rows]
    if len(clip_ids) != len(set(clip_ids)) or len(episodes) != len(set(episodes)):
        raise TemporalEvaluationError(
            "development manifest must contain one unique episode per clip"
        )
    result: dict[str, dict[str, Any]] = {}
    for offset in ACTION_PERMUTATION_OFFSETS:
        permutation = action_permutation_indices(len(rows), offset)
        if any(episodes[index] == episodes[source] for index, source in enumerate(permutation)):
            raise TemporalEvaluationError("action permutation is not episode-disjoint")
        mapping = {clip_ids[index]: clip_ids[source] for index, source in enumerate(permutation)}
        result[action_control(offset)] = {
            "offset": offset,
            "rule": f"source_index=(destination_index+{offset}) modulo {len(rows)}",
            "mapping_sha256": screen.sha256_json(mapping),
            "episode_disjoint": True,
            "fixed_point_free": True,
        }
    return result


def normalization_binding_sha256(
    target_mode: str, normalization_record: Mapping[str, Any] | None
) -> str:
    payload = {
        "schema": "causal-vjepa2-temporal-sampler-representation-v1",
        "target_mode": target_mode,
        "normalization": dict(normalization_record) if normalization_record else None,
    }
    return screen.sha256_json(payload)


def _sampler_input_record(
    history: Tensor,
    actions: Tensor,
    initial_video_noise: Tensor,
    initial_representation_noise: Tensor,
    *,
    representation_mode: str,
    normalization_binding: str,
) -> dict[str, str]:
    record = {
        "history_sha256": screen.tensor_sha256(history),
        "actions_sha256": screen.tensor_sha256(actions),
        "initial_video_noise_sha256": screen.tensor_sha256(initial_video_noise),
        "initial_representation_noise_sha256": screen.tensor_sha256(
            initial_representation_noise
        ),
        "representation_mode": representation_mode,
        "normalization_binding_sha256": normalization_binding,
    }
    if set(record).intersection(FORBIDDEN_SAMPLER_TERMS):
        raise TemporalEvaluationError("forbidden clean-data field entered sampler record")
    if representation_mode not in temporal.TARGET_MODES:
        raise TemporalEvaluationError("sampler input record has an invalid target mode")
    hash_fields = (
        "history_sha256",
        "actions_sha256",
        "initial_video_noise_sha256",
        "initial_representation_noise_sha256",
        "normalization_binding_sha256",
    )
    if any(not screen.HEX64.fullmatch(record[name]) for name in hash_fields):
        raise TemporalEvaluationError("sampler input record contains an invalid hash")
    return record


def sampler_input_sha256(record: Mapping[str, str]) -> str:
    expected = {
        "history_sha256",
        "actions_sha256",
        "initial_video_noise_sha256",
        "initial_representation_noise_sha256",
        "representation_mode",
        "normalization_binding_sha256",
    }
    if set(record) != expected or any(term in record for term in FORBIDDEN_SAMPLER_TERMS):
        raise ValueError("temporal sampler input record is malformed or leaks clean data")
    return screen.sha256_json(
        {"schema": "causal-vjepa2-temporal-sampler-input-v1", **dict(record)}
    )


@dataclass(frozen=True)
class GeneratedTemporalSample:
    representation_prediction: Tensor
    semantic_prediction: Tensor
    model_calls: int
    call_input_sha256_by_example: tuple[tuple[str, ...], ...]


@torch.inference_mode()
def sample_generated_temporal(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    *,
    video_noise: Tensor,
    representation_noise: Tensor,
    steps: int,
    representation_mode: temporal.TargetMode,
    normalization: temporal.TemporalNormalization | None,
) -> GeneratedTemporalSample:
    """Run the frozen uniform-clean-time Euler sampler with no clean-data API."""
    signature_names = set(inspect.signature(sample_generated_temporal).parameters)
    if any(term in name for name in signature_names for term in FORBIDDEN_SAMPLER_TERMS):
        raise TemporalEvaluationError("generated sampler signature admits clean future data")
    if steps not in NFE_GRID:
        raise ValueError(f"NFE must be one of {NFE_GRID}")
    if video_noise.dtype != torch.float32 or representation_noise.dtype != torch.float32:
        raise ValueError("generated sampler states must begin in float32")
    if not bool(torch.isfinite(video_noise).all()) or not bool(
        torch.isfinite(representation_noise).all()
    ):
        raise ValueError("generated sampler noise must be finite")
    sampled = temporal.sample_temporal_target(
        model,
        history,
        actions,
        video_noise=video_noise,
        auxiliary_noise=representation_noise,
        steps=steps,
        target_mode=representation_mode,
        normalization=normalization,
    )
    if (
        sampled.representation_prediction.dtype != torch.float32
        or sampled.semantic_prediction.dtype != torch.float32
    ):
        raise TemporalEvaluationError("Euler trajectory or decoded prediction left float32")
    if sampled.model_calls != steps:
        raise TemporalEvaluationError("actual transformer calls differ from NFE")
    return GeneratedTemporalSample(
        representation_prediction=sampled.representation_prediction,
        semantic_prediction=sampled.semantic_prediction,
        model_calls=sampled.model_calls,
        call_input_sha256_by_example=sampled.call_input_sha256_by_example,
    )


def _metric_bundle(
    semantic_prediction: Tensor,
    semantic_target: Tensor,
    representation_prediction: Tensor,
    representation_target: Tensor,
) -> dict[str, Tensor]:
    semantic = screen.semantic_metrics(semantic_prediction, semantic_target)
    packed = screen.semantic_metrics(representation_prediction, representation_target)
    result = {**semantic, **{f"packed_{name}": value for name, value in packed.items()}}
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise TemporalEvaluationError("evaluation metric is non-finite")
    return result


def _arm_contract(training_config: Mapping[str, Any]) -> tuple[str, str]:
    arm = training_config.get("doe_arm")
    if arm not in DOE_ARMS:
        raise TemporalEvaluationError("checkpoint is not a preregistered DOE arm")
    expected = ARM_CONTRACTS[str(arm)]
    loss = training_config.get("loss")
    rollin = training_config.get("self_rollin")
    if not isinstance(loss, Mapping) or not isinstance(rollin, Mapping):
        raise TemporalEvaluationError("training config lacks loss/self-roll-in contracts")
    normalization_present = training_config.get("normalization") is not None
    if (
        training_config.get("target_mode") != expected["target_mode"]
        or loss.get("flow_weight") != 1.0
        or loss.get("normalized_temporal_velocity_weight")
        != expected["temporal_weight"]
        or loss.get("action_shuffle_margin_weight") != 0.0
        or rollin.get("probability") != expected["rollin_probability"]
        or normalization_present is not expected["normalization"]
        or (
            arm == "DELTA-R"
            and rollin.get("later_time_rule") != "sampled_final_clock_fraction"
        )
    ):
        raise TemporalEvaluationError(f"training config does not implement arm {arm}")
    return str(arm), str(expected["target_mode"])


def _load_normalization_from_training(
    training_config: Mapping[str, Any], arm: str
) -> tuple[temporal.TemporalNormalization | None, dict[str, Any] | None]:
    expected = ARM_CONTRACTS[arm]
    record = training_config.get("normalization")
    if not expected["normalization"]:
        if record is not None:
            raise TemporalEvaluationError("ABS unexpectedly binds normalization")
        return None, None
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise TemporalEvaluationError(f"{arm} lacks its train-only normalization record")
    if record.get("sha256") != FROZEN_D1_NORMALIZATION_SHA256:
        raise TemporalEvaluationError(
            f"{arm} does not bind the preregistered completed D1 normalization"
        )
    datasets = training_config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise TemporalEvaluationError("training config lacks datasets")
    caches = datasets.get("semantic_cache")
    train_manifest = datasets.get("train")
    if not isinstance(caches, Mapping) or not isinstance(train_manifest, Mapping):
        raise TemporalEvaluationError("training config lacks train cache bindings")
    train_cache = caches.get("train")
    if not isinstance(train_cache, Mapping):
        raise TemporalEvaluationError("training config lacks train semantic cache")
    normalization, loaded_record = temporal.load_temporal_normalization(
        record["path"],
        expected_train_manifest_sha256=train_manifest.get("sha256"),
        expected_pca_sha256=train_cache.get("pca_sha256"),
        expected_cache_metadata_sha256=screen.sha256_json(train_cache),
    )
    if dict(record) != loaded_record:
        raise TemporalEvaluationError("normalization artifact changed after training")
    return normalization, loaded_record


def _load_checkpoint(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_config: Mapping[str, Any],
    validation_cache: Mapping[str, Any],
    producer_attestation: Mapping[str, Any],
) -> tuple[
    str,
    temporal.TargetMode,
    temporal.TemporalNormalization | None,
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if (
        checkpoint_path.name != "update_005000.pt"
        or checkpoint_path.parent.name != "checkpoints"
    ):
        raise TemporalEvaluationError("evaluation requires exact update_005000.pt")
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "resolved_config.json"
    complete_path = run_dir / "complete.json"
    training_config = vlf.load_json(config_path, "temporal training config")
    complete = vlf.load_json(complete_path, "temporal training completion")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = screen._source_record()  # noqa: SLF001 - frozen provenance primitive
    source_compatibility = training_source_compatibility(
        training_config.get("source"), source
    )
    checkpoint_record = vlf.file_record(checkpoint_path)
    config_record = vlf.file_record(config_path)
    arm, target_mode_string = _arm_contract(training_config)
    datasets = training_config.get("datasets")
    caches = datasets.get("semantic_cache") if isinstance(datasets, Mapping) else None
    config_validation_cache = (
        caches.get("validation") if isinstance(caches, Mapping) else None
    )
    cache_access = datasets.get("cache_access") if isinstance(datasets, Mapping) else None
    normalization, normalization_record = _load_normalization_from_training(
        training_config, arm
    )
    doe_common_identity = training_doe_common_identity(training_config)
    if (
        payload.get("schema") != screen.CHECKPOINT_SCHEMA
        or payload.get("arm") != "phase1"
        or payload.get("completed_updates") != screen.TRAIN_UPDATES
        or payload.get("model_config") != model_config
        or payload.get("config_sha256") != screen.sha256_json(training_config)
        or training_config.get("schema") != temporal.RUN_SCHEMA
        or training_config.get("command") != "train"
        or training_config.get("run_role") != "primary_5k"
        or training_config.get("updates") != screen.TRAIN_UPDATES
        or training_config.get("seed") != screen.FROZEN_SEED
        or training_config.get("global_batch_size")
        != screen.FROZEN_GLOBAL_BATCH_SIZE
        or training_config.get("world_size") != FROZEN_TRAIN_WORLD_SIZE
        or training_config.get("local_optimizer_batch_size")
        != FROZEN_LOCAL_BATCH_SIZE
        or training_config.get("micro_batch_size_per_rank")
        != FROZEN_LOCAL_BATCH_SIZE
        or training_config.get("gradient_accumulation_steps") != 1
        or training_config.get("workers_per_rank") != 4
        or training_config.get("checkpoint_updates")
        != list(temporal._checkpoint_updates(screen.TRAIN_UPDATES, 500))  # noqa: SLF001
        or training_config.get("initialization")
        != "from_scratch_deterministic_no_pretrained_weights"
        or training_config.get("dtype") != "bfloat16"
        or training_config.get("optimizer")
        != {
            "name": "AdamW",
            "learning_rate": vlf.FROZEN_LEARNING_RATE,
            "betas": [0.9, 0.95],
            "weight_decay": vlf.FROZEN_WEIGHT_DECAY,
            "warmup_updates": vlf.FROZEN_WARMUP_UPDATES,
            "after_warmup": "constant",
            "gradient_clip_norm": vlf.FROZEN_GRADIENT_CLIP_NORM,
        }
        or training_config.get("ema")
        != {
            "decay": vlf.FROZEN_EMA_DECAY,
            "schedule": vlf.FROZEN_EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        }
        or training_config.get("target_kind") != screen.TARGET_KIND
        or training_config.get("target_shape") != list(screen.TARGET_SHAPE)
        or training_config.get("model") != model_config
        or training_config.get("parameter_count") != screen.FROZEN_MODEL_PARAMETERS
        or training_config.get("entrypoint") != vlf.file_record(temporal.__file__)
        or training_config.get("protected_test_accessed") is not False
        or training_config.get("video_loss_enabled") is not False
        or training_config.get("teacher_model_calls") != 0
        or config_validation_cache != validation_cache
        or not isinstance(cache_access, Mapping)
        or cache_access.get("bridge") != vlf.file_record(cache_bridge.__file__)
        or cache_access.get("validation") != producer_attestation
        or cache_access.get("cache_relabelled_as_current_source") is not False
        or cache_access.get("protected_test_accessed") is not False
        or complete.get("schema") != temporal.RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("completed_updates") != screen.TRAIN_UPDATES
        or complete.get("nonfinite_updates") != 0
        or complete.get("target_mode") != target_mode_string
        or complete.get("video_loss_enabled") is not False
        or complete.get("future_rgb_model_input") is not False
        or complete.get("teacher_model_calls") != 0
        or complete.get("protected_test_accessed") is not False
        or complete.get("resolved_config_sha256") != screen.sha256_json(training_config)
        or complete.get("resolved_config") != config_record
        or complete.get("checkpoint") != checkpoint_record
    ):
        raise TemporalEvaluationError(
            "checkpoint/completion/config is not the exact final temporal DOE model"
        )
    ema = payload.get("ema")
    if (
        not isinstance(ema, Mapping)
        or ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or ema.get("num_updates") != screen.TRAIN_UPDATES
        or not isinstance(ema.get("shadow"), Mapping)
    ):
        raise TemporalEvaluationError("checkpoint lacks exact update-5,000 EMA")
    model.load_state_dict(ema["shadow"], strict=True)
    return (
        arm,
        target_mode_string,  # type: ignore[return-value]
        normalization,
        normalization_record,
        training_config,
        checkpoint_record,
        config_record,
        source_compatibility,
        doe_common_identity,
    )


def _primary_calibration_common(config: Mapping[str, Any]) -> dict[str, Any]:
    common = json.loads(screen.canonical_json(config))
    for name in (
        "run_role",
        "updates",
        "checkpoint_updates",
        "experiment_identity_sha256",
    ):
        common.pop(name, None)
    return common


def load_calibration_receipt(
    path: str | Path,
    *,
    primary_config: Mapping[str, Any],
    arm: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.name != "update_000200.pt" or checkpoint.parent.name != "checkpoints":
        raise TemporalEvaluationError("calibration receipt must be update_000200.pt")
    run_dir = checkpoint.parent.parent
    config_path = run_dir / "resolved_config.json"
    complete_path = run_dir / "complete.json"
    config = vlf.load_json(config_path, "calibration config")
    complete = vlf.load_json(complete_path, "calibration completion")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_record = vlf.file_record(checkpoint)
    config_record = vlf.file_record(config_path)
    complete_record = vlf.file_record(complete_path)
    calibration_arm, _ = _arm_contract(config)
    ema = payload.get("ema")
    if (
        calibration_arm != arm
        or config.get("schema") != temporal.RUN_SCHEMA
        or config.get("run_role") != "numerical_calibration"
        or config.get("updates") != screen.CALIBRATION_UPDATES
        or config.get("checkpoint_updates") != [screen.CALIBRATION_UPDATES]
        or _primary_calibration_common(config)
        != _primary_calibration_common(primary_config)
        or payload.get("schema") != screen.CHECKPOINT_SCHEMA
        or payload.get("arm") != "phase1"
        or payload.get("completed_updates") != screen.CALIBRATION_UPDATES
        or payload.get("model_config") != model_config
        or payload.get("config_sha256") != screen.sha256_json(config)
        or not isinstance(ema, Mapping)
        or ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or ema.get("num_updates") != screen.CALIBRATION_UPDATES
        or not isinstance(ema.get("shadow"), Mapping)
        or complete.get("schema") != temporal.RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("completed_updates") != screen.CALIBRATION_UPDATES
        or complete.get("nonfinite_updates") != 0
        or complete.get("target_mode") != primary_config.get("target_mode")
        or complete.get("video_loss_enabled") is not False
        or complete.get("future_rgb_model_input") is not False
        or complete.get("teacher_model_calls") != 0
        or complete.get("protected_test_accessed") is not False
        or complete.get("resolved_config_sha256") != screen.sha256_json(config)
        or complete.get("resolved_config") != config_record
        or complete.get("checkpoint") != checkpoint_record
    ):
        raise TemporalEvaluationError(
            "200-update calibration is not the exact primary-arm numerical receipt"
        )
    return {
        "schema": "causal-vjepa2-temporal-arm-calibration-receipt-v1",
        "arm": arm,
        "updates": screen.CALIBRATION_UPDATES,
        "checkpoint": checkpoint_record,
        "resolved_config": config_record,
        "complete": complete_record,
        "primary_equivalent_except_updates_and_run_role": True,
        "common_config_sha256": screen.sha256_json(
            _primary_calibration_common(primary_config)
        ),
        "nonfinite_updates": 0,
    }


def _distributed_action_bank(
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    context: vlf.DistributedContext,
) -> Tensor:
    """Read the small action bank once, sharded across ranks.

    Object collectives on an NCCL process group serialize their payload through
    CUDA tensors.  That is unnecessary for this CPU-only metadata and can fail
    before evaluation when the NCCL object buffer requests device memory.  Use
    a temporary Gloo subgroup so the action-bank exchange remains entirely on
    CPU; the model process group and all numerical evaluation collectives stay
    unchanged.
    """
    local: list[tuple[int, str, np.ndarray]] = []
    for index in range(context.rank, len(dataset), context.world_size):
        sample = dataset[index]
        actions = sample.get("actions")
        if (
            str(sample.get("clip_id")) != str(rows[index]["clip_id"])
            or not isinstance(actions, Tensor)
            or tuple(actions.shape) != (16, 7)
            or actions.dtype != torch.float32
            or not bool(torch.isfinite(actions).all())
        ):
            raise TemporalEvaluationError("global action-bank sample is malformed")
        local.append((index, str(sample["clip_id"]), actions.numpy().copy()))
    if context.world_size == 1:
        gathered: list[Any] = [local]
    else:
        if not torch.distributed.is_gloo_available():
            raise TemporalEvaluationError(
                "distributed action-bank exchange requires the CPU Gloo backend"
            )
        cpu_group = torch.distributed.new_group(backend="gloo")
        try:
            gathered = [None] * context.world_size
            torch.distributed.all_gather_object(
                gathered,
                local,
                group=cpu_group,
            )
        finally:
            torch.distributed.destroy_process_group(cpu_group)
    merged: dict[int, Tensor] = {}
    for shard in gathered:
        for index, clip_id, value in shard:
            if index in merged or clip_id != str(rows[index]["clip_id"]):
                raise TemporalEvaluationError("global action bank changed manifest order")
            merged[int(index)] = torch.from_numpy(value)
    if set(merged) != set(range(len(dataset))):
        raise TemporalEvaluationError("global action bank is incomplete")
    return torch.stack([merged[index] for index in range(len(dataset))]).contiguous()


def _encode_target(
    semantic: Tensor,
    target_mode: temporal.TargetMode,
    normalization: temporal.TemporalNormalization | None,
) -> Tensor:
    if target_mode == "absolute":
        return semantic
    if normalization is None:
        raise TemporalEvaluationError("delta target encoding lacks normalization")
    return normalization.encode(semantic, target_mode)


def _summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault((int(record["nfe"]), str(record["control"])), []).append(record)
    summaries: list[dict[str, Any]] = []
    for (nfe, control), rows in sorted(groups.items()):
        summaries.append(
            {
                "nfe": nfe,
                "control": control,
                "clips": len(rows),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in (*PRIMARY_METRICS, *PACKED_METRICS)
                },
            }
        )
    return summaries


def _row_record(
    *,
    arm: str,
    target_mode: str,
    normalization_binding: str,
    normalization_record: Mapping[str, Any] | None,
    control: str,
    nfe: int,
    clip_id: str,
    episode_index: int,
    history_source_clip_id: str | None,
    actions_source_clip_id: str | None,
    target_source_clip_id: str,
    donor_clip_id: str,
    donor_episode_index: int,
    semantic_prediction: Tensor,
    representation_prediction: Tensor,
    semantic_target: Tensor,
    representation_target: Tensor,
    item: int,
    sampler_record: Mapping[str, str] | None,
    sample: GeneratedTemporalSample | None,
    sample_item: int,
    generation_reused_from: str | None,
    checkpoint_record: Mapping[str, Any],
    training_config_record: Mapping[str, Any],
    evaluation_config_sha256: str,
    implementation_registration_identity_sha256: str,
    calibration_checkpoint_sha256: str,
    metrics: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    generated = sample is not None
    call_hashes = (
        list(sample.call_input_sha256_by_example[sample_item]) if generated else []
    )
    actual_calls = (
        0
        if not generated or generation_reused_from is not None
        else sample.model_calls
    )
    conceptual_calls = nfe if generated else 0
    normalization_artifact = (
        str(normalization_record["sha256"]) if normalization_record else None
    )
    return {
        "schema": EVALUATION_SCHEMA,
        "arm": arm,
        "target_mode": target_mode,
        "normalization_binding_sha256": normalization_binding,
        "normalization_artifact_sha256": normalization_artifact,
        "clip_id": clip_id,
        "episode_index": episode_index,
        "control": control,
        "nfe": nfe,
        "evaluation_seed": EVALUATION_SEED,
        "checkpoint_sha256": checkpoint_record["sha256"],
        "training_config_sha256": training_config_record["sha256"],
        "evaluation_config_sha256": evaluation_config_sha256,
        "implementation_registration_identity_sha256": (
            implementation_registration_identity_sha256
        ),
        "calibration_checkpoint_sha256": calibration_checkpoint_sha256,
        "history_source_clip_id": history_source_clip_id,
        "actions_source_clip_id": actions_source_clip_id,
        "target_source_clip_id": target_source_clip_id,
        "donor_clip_id": donor_clip_id,
        "donor_episode_index": donor_episode_index,
        "sampler_input": dict(sampler_record) if sampler_record is not None else None,
        "sampler_input_sha256": (
            sampler_input_sha256(sampler_record) if sampler_record is not None else None
        ),
        "model_call_input_sha256": call_hashes,
        "model_call_input_chain_sha256": screen.sha256_json(call_hashes),
        "conceptual_path_model_calls": conceptual_calls,
        "actual_evaluator_model_calls": actual_calls,
        "generation_reused_from": generation_reused_from,
        "generated_semantic_sha256": screen.tensor_sha256(semantic_prediction[item]),
        "generated_representation_sha256": screen.tensor_sha256(
            representation_prediction[item]
        ),
        "metric_target_sha256": screen.tensor_sha256(semantic_target[item]),
        "metric_representation_target_sha256": screen.tensor_sha256(
            representation_target[item]
        ),
        "generation_deployable": control in GENERATED_CONTROLS,
        "control_deployable": control in GENERATED_CONTROLS,
        "metric_comparison_only": control in {"donor_target", "zero", "oracle_clean"},
        "clean_future_target_entered_sampler": False,
        "future_rgb_entered_sampler": False,
        "teacher_model_calls": 0,
        "trajectory_state_dtype": "float32",
        "transformer_autocast": "cuda-bfloat16",
        **{name: float(values[item]) for name, values in metrics.items()},
    }


def evaluation_command(args: argparse.Namespace) -> int:
    determinism = screen._configure_deterministic_eval()  # noqa: SLF001
    context = vlf.initialize_distributed()
    logger: vlf.LocalAndOptionalWandbLogger | None = None
    try:
        if context.world_size != FROZEN_TRAIN_WORLD_SIZE:
            raise TemporalEvaluationError(
                f"development evaluation requires {FROZEN_TRAIN_WORLD_SIZE} ranks"
            )
        output_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
        manifest_path, rows, manifest_record = screen._manifest_record(  # noqa: SLF001
            args.manifest, split="val", expected_clips=screen.FROZEN_VALIDATION_CLIPS
        )
        permutation_records = validate_action_permutations(rows)
        for index in range(0, len(rows), 2):
            if int(rows[index]["episode_index"]) == int(rows[index + 1]["episode_index"]):
                raise TemporalEvaluationError("adjacent donors are not episode-disjoint")
        dataset = cache_bridge.construct_producer_attested_dataset(
            manifest_path, args.data_root, args.semantic_cache_root
        )
        cache_metadata = screen.validated_cache_metadata(dataset)
        action_bank = _distributed_action_bank(dataset, rows, context)
        action_bank_sha256 = screen.tensor_sha256(action_bank)
        vlf.seed_everything(args.seed, 0)
        model, model_config = screen.instantiate_model(args)
        model.to(context.device)
        (
            arm,
            target_mode,
            normalization,
            normalization_record,
            training_config,
            checkpoint_record,
            training_config_record,
            training_source_compatibility_record,
            training_doe_common_identity_sha256,
        ) = _load_checkpoint(
            args,
            model=model,
            model_config=model_config,
            validation_cache=cache_metadata,
            producer_attestation=dataset.producer_attestation,
        )
        calibration_receipt = load_calibration_receipt(
            args.calibration_checkpoint,
            primary_config=training_config,
            arm=arm,
            model_config=model_config,
        )
        model.eval()
        normalization_binding = normalization_binding_sha256(
            target_mode, normalization_record
        )
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
        clip_to_index = {str(row["clip_id"]): index for index, row in enumerate(rows)}
        donor_mapping = {
            str(rows[index]["clip_id"]): str(rows[index ^ 1]["clip_id"])
            for index in range(len(rows))
        }
        evaluation_source = screen._source_record()  # noqa: SLF001
        registered_protocol = preregistration_attestation(evaluation_source)
        implementation_registration, implementation_registration_record = (
            load_implementation_registration(
                args.implementation_registration,
                current_source=evaluation_source,
            )
        )
        config = {
            "schema": EVALUATION_SCHEMA,
            "status": "configured",
            "source": evaluation_source,
            "entrypoint": vlf.file_record(__file__),
            "dataset_source": vlf.file_record(cache_bridge.__file__),
            "cache_producer_attestation": dataset.producer_attestation,
            "split": "val",
            "development_selection_split": True,
            "protected_test_accessed": False,
            "arm": arm,
            "target_kind": screen.TARGET_KIND,
            "target_shape": list(screen.TARGET_SHAPE),
            "target_mode": target_mode,
            "normalization": normalization_record,
            "normalization_binding_sha256": normalization_binding,
            "checkpoint": checkpoint_record,
            "training_config": training_config_record,
            "training_source_compatibility": training_source_compatibility_record,
            "training_doe_common_identity_sha256": (
                training_doe_common_identity_sha256
            ),
            "preregistration": registered_protocol,
            "implementation_registration": implementation_registration_record,
            "implementation_registration_identity_sha256": (
                implementation_registration["identity_sha256"]
            ),
            "calibration_receipt": calibration_receipt,
            "training_experiment_identity_sha256": training_config[
                "experiment_identity_sha256"
            ],
            "weights": {
                "kind": "ema",
                "decay": vlf.FROZEN_EMA_DECAY,
                "schedule": vlf.FROZEN_EMA_SCHEDULE,
                "updates": screen.TRAIN_UPDATES,
            },
            "manifest": manifest_record,
            "semantic_cache": cache_metadata,
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "semantic_cache_root": str(
                Path(args.semantic_cache_root).expanduser().resolve()
            ),
            "evaluation_seed": args.seed,
            "nfe_grid": list(NFE_GRID),
            "selection_eligible_nfe": [1, 2, 4],
            "controls": list(CONTROLS),
            "generated_controls": list(GENERATED_CONTROLS),
            "action_permutations": permutation_records,
            "action_bank_sha256": action_bank_sha256,
            "fixed_clip_noise": True,
            "noise_key": "sha256(f'{clip_id}:{evaluation_seed}:video|aux')",
            "sampler": {
                "schedule": "uniform-clean-time-euler",
                "trajectory_state_dtype": "float32",
                "transformer_autocast": "cuda-bfloat16",
                "actual_calls_equal_nfe": True,
                "clean_target_argument": "forbidden",
                "future_rgb_argument": "forbidden",
                "teacher_argument": "forbidden",
                "delta_decode": "after_generated_trajectory_only",
            },
            "donor_rule": "manifest-adjacent xor-1; episode-disjoint",
            "donor_mapping_sha256": screen.sha256_json(donor_mapping),
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
        config_sha256 = screen.sha256_json(config)
        screen._assert_distributed_config(context, config_sha256)  # noqa: SLF001
        if context.is_primary:
            output_dir.mkdir(parents=True, exist_ok=False)
            vlf.atomic_write_json(output_dir / "resolved_config.json", config, exclusive=True)
            vlf.atomic_write_json(
                output_dir / "provenance.json",
                {
                    "schema": EVALUATION_SCHEMA,
                    "source": config["source"],
                    "runtime": vlf.runtime_record(),
                    "command": [sys.executable, *sys.argv],
                    "resolved_config_sha256": config_sha256,
                    "protected_test_accessed": False,
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
        permutations = {
            action_control(offset): action_permutation_indices(len(rows), offset)
            for offset in ACTION_PERMUTATION_OFFSETS
        }
        for raw_batch in loader:
            batch = screen._validate_batch(raw_batch, context.device)  # noqa: SLF001
            clip_ids = [str(value) for value in raw_batch["clip_id"]]
            episode_indices = [int(value) for value in raw_batch["episode_index"]]
            screen._validate_pair_batch(clip_ids, episode_indices)  # noqa: SLF001
            global_indices = [clip_to_index[clip_id] for clip_id in clip_ids]
            donor_index = torch.arange(
                len(clip_ids), device=context.device, dtype=torch.long
            ).bitwise_xor(1)
            donor_ids = [clip_ids[int(index)] for index in donor_index.cpu().tolist()]
            if any(
                donor_mapping[destination] != donor
                for destination, donor in zip(clip_ids, donor_ids, strict=True)
            ):
                raise TemporalEvaluationError("runtime donor mapping changed")

            video_reference = torch.empty(
                len(clip_ids), 3, 8, 64, 112, device=context.device, dtype=torch.float32
            )
            representation_reference = torch.empty(
                len(clip_ids), *screen.TARGET_SHAPE,
                device=context.device,
                dtype=torch.float32,
            )
            video_noise = vlf.stable_noise_like(
                video_reference, clip_ids, args.seed, "video"
            )
            representation_noise = vlf.stable_noise_like(
                representation_reference, clip_ids, args.seed, "aux"
            )
            own_semantic_target = batch["auxiliary_target"].float()
            own_representation_target = _encode_target(
                own_semantic_target, target_mode, normalization
            )
            for nfe in NFE_GRID:
                sources: dict[str, tuple[Tensor, Tensor, list[str], list[str]]] = {
                    "autonomous": (
                        batch["history"], batch["actions"], clip_ids, clip_ids
                    ),
                    "context_shuffled": (
                        batch["history"].index_select(0, donor_index),
                        batch["actions"].index_select(0, donor_index),
                        donor_ids,
                        donor_ids,
                    ),
                    "history_shuffled": (
                        batch["history"].index_select(0, donor_index),
                        batch["actions"],
                        donor_ids,
                        clip_ids,
                    ),
                }
                for control, permutation in permutations.items():
                    source_indices = [permutation[index] for index in global_indices]
                    source_ids = [str(rows[index]["clip_id"]) for index in source_indices]
                    source_actions = action_bank.index_select(
                        0, torch.tensor(source_indices, dtype=torch.long)
                    ).to(device=context.device, non_blocking=True)
                    sources[control] = (
                        batch["history"], source_actions, clip_ids, source_ids
                    )

                generated: dict[str, GeneratedTemporalSample] = {}
                sampler_records: dict[str, list[dict[str, str]]] = {}
                for control in GENERATED_CONTROLS:
                    history, actions, _, _ = sources[control]
                    sampler_records[control] = [
                        _sampler_input_record(
                            history[item],
                            actions[item],
                            video_noise[item],
                            representation_noise[item],
                            representation_mode=target_mode,
                            normalization_binding=normalization_binding,
                        )
                        for item in range(len(clip_ids))
                    ]
                    with vlf._autocast(context.device):  # noqa: SLF001
                        generated[control] = sample_generated_temporal(
                            model,
                            history,
                            actions,
                            video_noise=video_noise,
                            representation_noise=representation_noise,
                            steps=nfe,
                            representation_mode=target_mode,
                            normalization=normalization,
                        )
                    local_actual_batched_calls += generated[control].model_calls

                semantic_predictions = {
                    **{
                        control: generated[control].semantic_prediction
                        for control in GENERATED_CONTROLS
                    },
                    "donor_target": generated["autonomous"].semantic_prediction,
                    "zero": torch.zeros_like(own_semantic_target),
                    "oracle_clean": own_semantic_target,
                }
                representation_predictions = {
                    **{
                        control: generated[control].representation_prediction
                        for control in GENERATED_CONTROLS
                    },
                    "donor_target": generated["autonomous"].representation_prediction,
                    "zero": _encode_target(
                        torch.zeros_like(own_semantic_target), target_mode, normalization
                    ),
                    "oracle_clean": own_representation_target,
                }
                for control in CONTROLS:
                    donor_target = control == "donor_target"
                    semantic_target = (
                        own_semantic_target.index_select(0, donor_index)
                        if donor_target
                        else own_semantic_target
                    )
                    representation_target = (
                        own_representation_target.index_select(0, donor_index)
                        if donor_target
                        else own_representation_target
                    )
                    sample = (
                        generated["autonomous"]
                        if donor_target
                        else generated.get(control)
                    )
                    sampler_control = "autonomous" if donor_target else control
                    metrics = {
                        name: value.detach().cpu().tolist()
                        for name, value in _metric_bundle(
                            semantic_predictions[control],
                            semantic_target,
                            representation_predictions[control],
                            representation_target,
                        ).items()
                    }
                    for item, clip_id in enumerate(clip_ids):
                        donor_item = int(donor_index[item])
                        if sample is None:
                            history_source = None
                            actions_source = None
                            sampler_record = None
                        else:
                            _, _, history_ids, action_ids = sources[sampler_control]
                            history_source = history_ids[item]
                            actions_source = action_ids[item]
                            sampler_record = sampler_records[sampler_control][item]
                        local_records.append(
                            _row_record(
                                arm=arm,
                                target_mode=target_mode,
                                normalization_binding=normalization_binding,
                                normalization_record=normalization_record,
                                control=control,
                                nfe=nfe,
                                clip_id=clip_id,
                                episode_index=episode_indices[item],
                                history_source_clip_id=history_source,
                                actions_source_clip_id=actions_source,
                                target_source_clip_id=(
                                    donor_ids[item] if donor_target else clip_id
                                ),
                                donor_clip_id=donor_ids[item],
                                donor_episode_index=episode_indices[donor_item],
                                semantic_prediction=semantic_predictions[control],
                                representation_prediction=representation_predictions[
                                    control
                                ],
                                semantic_target=semantic_target,
                                representation_target=representation_target,
                                item=item,
                                sampler_record=sampler_record,
                                sample=sample,
                                sample_item=item,
                                generation_reused_from=(
                                    "autonomous" if donor_target else None
                                ),
                                checkpoint_record=checkpoint_record,
                                training_config_record=training_config_record,
                                evaluation_config_sha256=config_sha256,
                                implementation_registration_identity_sha256=(
                                    implementation_registration["identity_sha256"]
                                ),
                                calibration_checkpoint_sha256=calibration_receipt[
                                    "checkpoint"
                                ]["sha256"],
                                metrics=metrics,
                            )
                        )

        rank_path = output_dir / "rank_metrics" / f"rank_{context.rank:04d}.jsonl"
        screen._atomic_jsonl(rank_path, local_records)  # noqa: SLF001
        shard = {
            "rank": context.rank,
            "records": len(local_records),
            "actual_batched_transformer_calls": local_actual_batched_calls,
            "file": vlf.file_record(rank_path),
        }
        shards = context.gather_objects(shard)
        context.barrier()
        if context.is_primary:
            all_records: list[dict[str, Any]] = []
            for item in sorted(shards, key=lambda value: int(value["rank"])):
                path = Path(item["file"]["path"])
                if vlf.file_record(path) != item["file"]:
                    raise TemporalEvaluationError("rank metric shard changed")
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise TemporalEvaluationError("metric row is not an object")
                        all_records.append(row)
            all_records.sort(key=lambda row: (row["nfe"], row["control"], row["clip_id"]))
            expected = screen.FROZEN_VALIDATION_CLIPS * len(NFE_GRID) * len(CONTROLS)
            if len(all_records) != expected:
                raise TemporalEvaluationError(
                    f"evaluation produced {len(all_records)} rows, expected {expected}"
                )
            metrics_path = output_dir / "per_clip_metrics.jsonl"
            screen._atomic_jsonl(metrics_path, all_records)  # noqa: SLF001
            summaries = _summaries(all_records)
            summary = {
                "schema": EVALUATION_SCHEMA,
                "status": "complete",
                "split": "val",
                "development_selection_split": True,
                "protected_test_accessed": False,
                "arm": arm,
                "target_mode": target_mode,
                "normalization": normalization_record,
                "normalization_binding_sha256": normalization_binding,
                "checkpoint": checkpoint_record,
                "training_config": training_config_record,
                "training_source_compatibility": training_source_compatibility_record,
                "training_doe_common_identity_sha256": (
                    training_doe_common_identity_sha256
                ),
                "preregistration": registered_protocol,
                "implementation_registration": implementation_registration_record,
                "implementation_registration_identity_sha256": (
                    implementation_registration["identity_sha256"]
                ),
                "calibration_receipt": calibration_receipt,
                "manifest": manifest_record,
                "semantic_cache": cache_metadata,
                "dataset_source": vlf.file_record(cache_bridge.__file__),
                "cache_producer_attestation": dataset.producer_attestation,
                "provenance": vlf.file_record(output_dir / "provenance.json"),
                "resolved_config": vlf.file_record(output_dir / "resolved_config.json"),
                "per_clip_metrics": vlf.file_record(metrics_path),
                "record_count": len(all_records),
                "cell_count": len(summaries),
                "summaries": summaries,
                "rank_shards": shards,
                "actual_batched_transformer_calls": sum(
                    int(item["actual_batched_transformer_calls"]) for item in shards
                ),
                "nfe_is_actual_transformer_calls_per_generated_path": True,
                "donor_target_generation_reused_bit_identically": True,
                "clean_future_target_entered_deployable_sampler": False,
                "future_rgb_entered_deployable_sampler": False,
                "teacher_model_calls": 0,
                "packed_metrics_are_diagnostic_only": True,
            }
            vlf.atomic_write_json(output_dir / "summary.json", summary, exclusive=True)
            for index, cell in enumerate(summaries):
                logger.log(
                    {
                        "update": screen.TRAIN_UPDATES + index,
                        "checkpoint_update": screen.TRAIN_UPDATES,
                        "event": "causal_vjepa2_temporal_target_evaluation",
                        "arm": arm,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--semantic-cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-checkpoint", required=True)
    parser.add_argument("--implementation-registration", required=True)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    screen._add_model_arguments(parser)  # noqa: SLF001
    screen._add_wandb_arguments(parser)  # noqa: SLF001
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.seed != EVALUATION_SEED
        or args.workers < 0
        or args.eval_batch_size < 2
        or args.eval_batch_size % 2
        or args.width != vlf.FROZEN_MODEL_WIDTH
        or args.depth != vlf.FROZEN_MODEL_DEPTH
        or args.heads != vlf.FROZEN_MODEL_HEADS
        or args.mlp_ratio != vlf.FROZEN_MODEL_MLP_RATIO
    ):
        raise TemporalEvaluationError(
            "evaluation preserves seed, model geometry, and even paired batches"
        )
    wandb_values = (
        args.wandb_entity,
        args.wandb_project,
        args.wandb_private_project_ack,
    )
    if args.wandb != all(bool(value) for value in wandb_values):
        raise TemporalEvaluationError("W&B requires all private-project arguments")
    if args.wandb and (
        args.wandb_entity != "zijiandu"
        or args.wandb_project != "dual-video-diffusion-private"
    ):
        raise TemporalEvaluationError(
            "W&B is frozen to zijiandu/dual-video-diffusion-private"
        )


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "register":
        registration_parser = argparse.ArgumentParser(
            description="Freeze the exact temporal DOE evaluator/analyzer revision."
        )
        registration_parser.add_argument("--output", required=True)
        registration_args = registration_parser.parse_args(raw[1:])
        try:
            payload = create_implementation_registration(registration_args.output)
            print(screen.canonical_json(payload))
            return 0
        except (TemporalEvaluationError, screen.ScreenError, vlf.PocError, OSError) as exc:
            print(f"Temporal-target registration error: {exc}", file=sys.stderr)
            return 2
    if raw and raw[0] == "eval":
        raw = raw[1:]
    args = build_parser().parse_args(raw)
    try:
        validate_args(args)
        return evaluation_command(args)
    except (
        TemporalEvaluationError,
        temporal.TemporalTargetError,
        screen.ScreenError,
        cache_bridge.ProducerCacheBridgeError,
        vlf.PocError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Temporal-target evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
