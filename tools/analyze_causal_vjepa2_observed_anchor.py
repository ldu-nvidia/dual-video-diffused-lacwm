#!/usr/bin/env python3
"""Development-only paired gate for the corrected AINC-OFF screen.

The analyzer consumes one completed AINC-OFF validation evaluation and the
exact ABS reference from the temporal-target DOE.  It verifies source, cache,
checkpoint, control, pairing, and no-leakage evidence before computing the one
preregistered three-cell family.  It never accepts a protected-test artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_causal_vjepa2_temporal_targets as temporal_analysis  # noqa: E402
from tools import causal_vjepa2_observed_anchor as anchor  # noqa: E402
from tools import causal_vjepa2_observed_anchor_screen as ainc  # noqa: E402
from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402


ANALYSIS_SCHEMA = "causal-vjepa2-observed-anchor-development-analysis-v1"
SELECTION_SCHEMA = "causal-vjepa2-observed-anchor-frozen-selection-v1"
SELECTION_NFE = ainc.NFE_GRID
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260807
CELLWISE_CONFIDENCE = 1.0 - 0.05 / len(SELECTION_NFE)
EXPECTED_CLIPS = screen.FROZEN_VALIDATION_CLIPS
MIN_RELATIVE_IMPROVEMENT = 0.05
MAX_NMSE = 0.50
MIN_COSINE = 0.70
METRICS = (
    "semantic_nmse",
    "semantic_token_cosine",
    "temporal_difference_nmse",
    "temporal_difference_token_cosine",
    "increment_nmse",
    "increment_token_cosine",
)
TEMPORAL_CONTROLS = (
    "anchor_static",
    "mean_increment",
    "context_shuffled",
    "donor_target",
)


class ObservedAnchorAnalysisError(RuntimeError):
    """An input artifact or paired-gate contract failed closed."""


@dataclass(frozen=True)
class CellData:
    nfe: int
    control: str
    clip_ids: tuple[str, ...]
    metrics: Mapping[str, np.ndarray]
    pairing_identity_sha256: str
    generation_identity_sha256: str
    increment_identity_sha256: str
    sampling_identity_sha256: str = ""


@dataclass(frozen=True)
class EvaluationData:
    summary_record: Mapping[str, Any]
    checkpoint_record: Mapping[str, Any]
    training_config_record: Mapping[str, Any]
    training_config: Mapping[str, Any]
    execution_condition: Mapping[str, Any]
    manifest_record: Mapping[str, Any]
    cells: Mapping[tuple[int, str], CellData]


def _verified_file_record(
    value: Any, *, label: str, extra_keys: Sequence[str] = ()
) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise ObservedAnchorAnalysisError(f"{label} lacks a file record")
    path = Path(str(value["path"])).expanduser().resolve()
    actual = vlf.file_record(path) if path.is_file() and not path.is_symlink() else None
    expected_keys = {"path", "sha256", "bytes", *extra_keys}
    core = (
        {name: value.get(name) for name in ("path", "sha256", "bytes")}
        if isinstance(value, Mapping)
        else {}
    )
    if actual is None or core != actual or set(value) != expected_keys:
        raise ObservedAnchorAnalysisError(f"{label} is missing, linked, or changed")
    return path


def _finite_metric(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservedAnchorAnalysisError(f"metric row lacks numeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ObservedAnchorAnalysisError(f"metric row has non-finite {name}")
    return result


def _hex(value: Any) -> bool:
    return isinstance(value, str) and screen.HEX64.fullmatch(value) is not None


def _sampler_hash(record: Mapping[str, str]) -> str:
    try:
        return screen.sampler_input_sha256(record)
    except (TypeError, ValueError) as exc:
        raise ObservedAnchorAnalysisError("row has malformed sampler inputs") from exc


def _control_source_valid(
    row: Mapping[str, Any],
    *,
    donor_clip: str,
    action_source: Mapping[str, Mapping[str, str]],
) -> bool:
    control = str(row["control"])
    clip_id = str(row["clip_id"])
    history = row.get("history_source_clip_id")
    actions = row.get("actions_source_clip_id")
    decode_anchor = row.get("decode_anchor_source_clip_id")
    target = row.get("target_source_clip_id")
    if control == "autonomous":
        return (history, actions, decode_anchor, target) == (
            clip_id,
            clip_id,
            clip_id,
            clip_id,
        )
    if control == "context_shuffled":
        return (history, actions, decode_anchor, target) == (
            donor_clip,
            donor_clip,
            donor_clip,
            clip_id,
        )
    if control == "history_shuffled":
        return (history, actions, decode_anchor, target) == (
            donor_clip,
            clip_id,
            donor_clip,
            clip_id,
        )
    if control in action_source:
        return (history, actions, decode_anchor, target) == (
            clip_id,
            action_source[control][clip_id],
            clip_id,
            clip_id,
        )
    if control == "donor_target":
        return (history, actions, decode_anchor, target) == (
            clip_id,
            clip_id,
            clip_id,
            donor_clip,
        )
    if control == "anchor_decode_shuffled":
        return (history, actions, decode_anchor, target) == (
            clip_id,
            clip_id,
            donor_clip,
            clip_id,
        )
    if control in {"anchor_static", "mean_increment", "zero", "oracle_clean"}:
        return history is None and actions is None and decode_anchor == clip_id and target == clip_id
    return False


def _validate_row(
    row: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    training_config_sha256: str,
    evaluation_config_sha256: str,
    normalization_sha256: str,
    donor_by_clip: Mapping[str, str],
    episode_by_clip: Mapping[str, int],
    action_source: Mapping[str, Mapping[str, str]],
) -> tuple[int, str, str]:
    try:
        nfe = int(row["nfe"])
        control = str(row["control"])
        clip_id = str(row["clip_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ObservedAnchorAnalysisError("metric row lacks cell identity") from exc
    if nfe not in SELECTION_NFE or control not in ainc.CONTROLS or not clip_id:
        raise ObservedAnchorAnalysisError("metric row identifies an unregistered cell")
    donor = donor_by_clip.get(clip_id)
    call_hashes = row.get("model_call_input_sha256")
    hash_fields = (
        "decode_anchor_sha256",
        "generated_increment_sha256",
        "decoded_semantic_sha256",
        "metric_target_sha256",
        "metric_increment_target_sha256",
        "model_call_input_chain_sha256",
    )
    if (
        donor is None
        or row.get("schema") != ainc.EVALUATION_SCHEMA
        or row.get("doe_arm") != ainc.DOE_ARM
        or row.get("evaluation_seed") != ainc.EVALUATION_SEED
        or row.get("checkpoint_sha256") != checkpoint_sha256
        or row.get("training_config_sha256") != training_config_sha256
        or row.get("evaluation_config_sha256") != evaluation_config_sha256
        or row.get("normalization_artifact_sha256") != normalization_sha256
        or row.get("episode_index") != episode_by_clip.get(clip_id)
        or row.get("donor_clip_id") != donor
        or row.get("donor_episode_index") != episode_by_clip.get(donor)
        or row.get("decode_anchor_source_episode_index")
        != episode_by_clip.get(str(row.get("decode_anchor_source_clip_id")))
        or row.get("actions_source_episode_index")
        != (
            episode_by_clip.get(str(row.get("actions_source_clip_id")))
            if row.get("actions_source_clip_id") is not None
            else None
        )
        or row.get("clean_future_target_entered_sampler") is not False
        or row.get("future_rgb_entered_sampler") is not False
        or row.get("anchor_entered_model") is not False
        or row.get("teacher_model_calls_during_sampling") != 0
        or row.get("trajectory_state_dtype") != "float32"
        or row.get("transformer_autocast") != "cuda-bfloat16"
        or row.get("metric_target_available_at_inference") is not False
        or row.get("generation_deployable")
        is not (control in ainc.GENERATED_CONTROLS)
        or row.get("control_deployable")
        is not (
            control
            in {
                *ainc.GENERATED_CONTROLS,
                "anchor_static",
                "mean_increment",
            }
        )
        or row.get("metric_comparison_only")
        is not (
            control
            in {"donor_target", "anchor_decode_shuffled", "zero", "oracle_clean"}
        )
        or not isinstance(call_hashes, list)
        or any(not _hex(value) for value in call_hashes)
        or any(not _hex(row.get(name)) for name in hash_fields)
        or row.get("model_call_input_chain_sha256")
        != screen.sha256_json(call_hashes)
        or not _control_source_valid(
            row, donor_clip=donor, action_source=action_source
        )
    ):
        raise ObservedAnchorAnalysisError("metric row violates the generated-only contract")

    sampler = row.get("sampler_input")
    reused = control in {"donor_target", "anchor_decode_shuffled"}
    generated = control in ainc.GENERATED_CONTROLS
    if generated:
        expected_actual = expected_conceptual = expected_hashes = nfe
        if not isinstance(sampler, Mapping) or row.get("generation_reused_from") is not None:
            raise ObservedAnchorAnalysisError("generated row lacks its own sampler inputs")
    elif reused:
        expected_actual, expected_conceptual, expected_hashes = 0, nfe, nfe
        if not isinstance(sampler, Mapping) or row.get("generation_reused_from") != "autonomous":
            raise ObservedAnchorAnalysisError("reuse control did not reuse autonomous generation")
    else:
        expected_actual = expected_conceptual = expected_hashes = 0
        if sampler is not None or row.get("generation_reused_from") is not None:
            raise ObservedAnchorAnalysisError("metric-only row unexpectedly sampled")
    if isinstance(sampler, Mapping):
        if row.get("sampler_input_sha256") != _sampler_hash(sampler):
            raise ObservedAnchorAnalysisError("sampler-input hash changed")
    elif row.get("sampler_input_sha256") is not None:
        raise ObservedAnchorAnalysisError("metric-only row has a sampler hash")
    if (
        row.get("actual_evaluator_model_calls") != expected_actual
        or row.get("conceptual_path_model_calls") != expected_conceptual
        or len(call_hashes) != expected_hashes
    ):
        raise ObservedAnchorAnalysisError("row has incorrect transformer-call accounting")
    for name in METRICS:
        _finite_metric(row, name)
    return nfe, control, clip_id


def _cell_from_rows(
    nfe: int, control: str, rows: Sequence[Mapping[str, Any]]
) -> CellData:
    ordered = sorted(rows, key=lambda row: str(row["clip_id"]))
    clip_ids = tuple(str(row["clip_id"]) for row in ordered)
    if len(ordered) != EXPECTED_CLIPS or len(set(clip_ids)) != EXPECTED_CLIPS:
        raise ObservedAnchorAnalysisError(
            f"{control}/NFE{nfe} lacks the exact 890 unique clips"
        )
    pairing_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    for row in ordered:
        sampler = row.get("sampler_input")
        sampler_pair = None
        if isinstance(sampler, Mapping):
            sampler_pair = {
                "history_sha256": sampler.get("history_sha256"),
                "actions_sha256": sampler.get("actions_sha256"),
                "initial_video_noise_sha256": sampler.get(
                    "initial_video_noise_sha256"
                ),
                # Match the ABS analyzer's representation-agnostic key.
                "initial_representation_noise_sha256": sampler.get(
                    "initial_auxiliary_noise_sha256"
                ),
            }
        pairing_rows.append(
            {
                "clip_id": row["clip_id"],
                "episode_index": row["episode_index"],
                "history_source_clip_id": row["history_source_clip_id"],
                "actions_source_clip_id": row["actions_source_clip_id"],
                "target_source_clip_id": row["target_source_clip_id"],
                "donor_clip_id": row["donor_clip_id"],
                "donor_episode_index": row["donor_episode_index"],
                "metric_target_sha256": row["metric_target_sha256"],
                "sampler_inputs_without_representation_binding": sampler_pair,
            }
        )
        generation_rows.append(
            {
                "clip_id": row["clip_id"],
                "decoded_semantic_sha256": row["decoded_semantic_sha256"],
                "model_call_input_chain_sha256": row[
                    "model_call_input_chain_sha256"
                ],
            }
        )
        increment_rows.append(
            {
                "clip_id": row["clip_id"],
                "generated_increment_sha256": row["generated_increment_sha256"],
            }
        )
        sampling_rows.append(
            {
                "clip_id": row["clip_id"],
                "sampler_input_sha256": row["sampler_input_sha256"],
                "model_call_input_chain_sha256": row[
                    "model_call_input_chain_sha256"
                ],
            }
        )
    return CellData(
        nfe=nfe,
        control=control,
        clip_ids=clip_ids,
        metrics={
            name: np.asarray(
                [_finite_metric(row, name) for row in ordered], dtype=np.float64
            )
            for name in METRICS
        },
        pairing_identity_sha256=screen.sha256_json(pairing_rows),
        generation_identity_sha256=screen.sha256_json(generation_rows),
        increment_identity_sha256=screen.sha256_json(increment_rows),
        sampling_identity_sha256=screen.sha256_json(sampling_rows),
    )


def _validate_training_fairness(
    ainc_config: Mapping[str, Any], abs_config: Mapping[str, Any]
) -> None:
    science = ainc_config.get("science_identity")
    if not isinstance(science, Mapping):
        raise ObservedAnchorAnalysisError("AINC training config lacks science identity")
    abs_datasets = abs_config.get("datasets")
    ainc_datasets = science.get("datasets")
    if not isinstance(abs_datasets, Mapping) or not isinstance(ainc_datasets, Mapping):
        raise ObservedAnchorAnalysisError("training configs lack dataset bindings")
    exact_pairs = (
        (science.get("seed"), abs_config.get("seed")),
        (science.get("global_batch_size"), abs_config.get("global_batch_size")),
        (science.get("world_size"), abs_config.get("world_size")),
        (
            science.get("local_optimizer_batch_size"),
            abs_config.get("local_optimizer_batch_size"),
        ),
        (
            science.get("micro_batch_size_per_rank"),
            abs_config.get("micro_batch_size_per_rank"),
        ),
        (
            science.get("gradient_accumulation_steps"),
            abs_config.get("gradient_accumulation_steps"),
        ),
        (science.get("optimizer"), abs_config.get("optimizer")),
        (science.get("ema"), abs_config.get("ema")),
        (science.get("model"), abs_config.get("model")),
        (science.get("parameter_count"), abs_config.get("parameter_count")),
        (science.get("workers_per_rank"), abs_config.get("workers_per_rank")),
        (ainc_datasets.get("train"), abs_datasets.get("train")),
        (ainc_datasets.get("validation"), abs_datasets.get("validation")),
        (
            ainc_datasets.get("semantic_cache"),
            abs_datasets.get("semantic_cache"),
        ),
    )
    if any(left != right for left, right in exact_pairs):
        raise ObservedAnchorAnalysisError(
            "AINC and continuation-local C-ABS do not share exact "
            "initialization/data/optimizer geometry"
        )
    abs_loss = abs_config.get("loss")
    abs_rollin = abs_config.get("self_rollin")
    phase1 = science.get("phase1_schedule")
    if (
        science.get("initialization")
        != "from_scratch_deterministic_no_pretrained_weights"
        or science.get("target_shape") != abs_config.get("target_shape")
        or science.get("dtype") != abs_config.get("dtype")
        or science.get("clean_time_epsilon") != vlf.FROZEN_CLEAN_TIME_EPS
        or not isinstance(science.get("source"), Mapping)
        or science["source"].get("commit")
        != abs_config.get("source", {}).get("commit")
        or science["source"].get("commit") == ainc.TEMPORAL_AUTHORIZATION_COMMIT
        or not isinstance(abs_loss, Mapping)
        or abs_loss.get("flow_weight") != 1.0
        or abs_loss.get("normalized_temporal_velocity_weight") != 0.0
        or abs_loss.get("action_shuffle_margin_weight") != 0.0
        or not isinstance(abs_rollin, Mapping)
        or abs_rollin.get("probability") != 0.0
        or not isinstance(phase1, Mapping)
        or phase1.get("video_time") != 0.0
        or phase1.get("video_loss") != "disabled"
        or phase1.get("auxiliary_logit_normal_mean") != -1.2
        or phase1.get("auxiliary_logit_normal_std") != 1.0
        or phase1.get("auxiliary_loss_coefficient") != 0.333
        or phase1.get("loss_mask_normalization") != "unchanged_global_batch"
        or ainc_config.get("updates") != ainc.TRAIN_UPDATES
        or abs_config.get("updates") != screen.TRAIN_UPDATES
    ):
        raise ObservedAnchorAnalysisError(
            "AINC and continuation-local C-ABS training mechanisms are not matched"
        )


def load_evaluation(
    summary_path: str | Path,
    *,
    abs_evaluation: temporal_analysis.EvaluationData,
) -> EvaluationData:
    resolved_summary = Path(summary_path).expanduser().resolve()
    if resolved_summary.is_symlink() or not resolved_summary.is_file():
        raise ObservedAnchorAnalysisError("AINC summary must be a regular file")
    summary = vlf.load_json(resolved_summary, "AINC development summary")
    config_path = _verified_file_record(summary.get("resolved_config"), label="AINC evaluation config")
    provenance_path = _verified_file_record(summary.get("provenance"), label="AINC evaluation provenance")
    metric_path = _verified_file_record(summary.get("per_clip_metrics"), label="AINC per-clip metrics")
    checkpoint_path = _verified_file_record(summary.get("checkpoint"), label="AINC checkpoint")
    training_config_path = _verified_file_record(summary.get("training_config"), label="AINC training config")
    normalization_path = _verified_file_record(
        summary.get("normalization"),
        label="AINC normalization",
        extra_keys=("payload_sha256",),
    )
    config = vlf.load_json(config_path, "AINC evaluation config")
    provenance = vlf.load_json(provenance_path, "AINC evaluation provenance")
    training_config = vlf.load_json(training_config_path, "AINC training config")
    current_source = ainc._source_record()  # noqa: SLF001
    execution = config.get("execution_condition")
    evaluation_wandb = config.get("wandb")
    training_wandb = training_config.get("wandb")
    if (
        summary.get("schema") != ainc.EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("doe_arm") != ainc.DOE_ARM
        or summary.get("split") != "val"
        or summary.get("development_selection_split") is not True
        or summary.get("protected_test_accessed") is not False
        or summary.get("clean_future_target_entered_deployable_sampler") is not False
        or summary.get("future_rgb_entered_deployable_sampler") is not False
        or summary.get("teacher_model_calls") != 0
        or summary.get("anchor_entered_model") is not False
        or summary.get("nfe_is_actual_transformer_calls_per_generated_path")
        is not True
        or summary.get("donor_and_anchor_decode_controls_reuse_generation")
        is not True
        or summary.get("increment_metrics_are_diagnostic_only") is not True
        or config.get("schema") != ainc.EVALUATION_SCHEMA
        or config.get("source") != current_source
        or config.get("entrypoint") != vlf.file_record(ainc.__file__)
        or config.get("doe_arm") != ainc.DOE_ARM
        or config.get("split") != "val"
        or config.get("validation_clips") != EXPECTED_CLIPS
        or config.get("nfe_grid") != list(SELECTION_NFE)
        or config.get("controls") != list(ainc.CONTROLS)
        or config.get("world_size") != ainc.FROZEN_WORLD_SIZE
        or config.get("eval_batch_size") != ainc.FROZEN_EVAL_BATCH_SIZE
        or config.get("workers_per_rank") != ainc.FROZEN_EVAL_WORKERS
        or config.get("seed") != ainc.EVALUATION_SEED
        or config.get("protected_test_accessed") is not False
        or config.get("clean_future_target_entered_sampler") is not False
        or config.get("future_rgb_entered_sampler") is not False
        or config.get("anchor_enters_model") is not False
        or summary.get("execution_condition") != execution
        or not isinstance(execution, Mapping)
        or provenance.get("schema") != ainc.EVALUATION_SCHEMA
        or provenance.get("source") != current_source
        or provenance.get("resolved_config_sha256") != screen.sha256_json(config)
        or provenance.get("secrets_persisted") is not False
        or not isinstance(evaluation_wandb, Mapping)
        or evaluation_wandb.get("enabled") is not True
        or evaluation_wandb.get("entity") != "zijiandu"
        or evaluation_wandb.get("project") != "dual-video-diffusion-private"
        or evaluation_wandb.get("group") is not None
        or evaluation_wandb.get("private_project_acknowledged") is not True
        or not isinstance(training_wandb, Mapping)
        or training_wandb.get("enabled") is not True
        or training_wandb.get("entity") != "zijiandu"
        or training_wandb.get("project") != "dual-video-diffusion-private"
        or training_wandb.get("group") is not None
        or training_wandb.get("private_project_acknowledged") is not True
    ):
        raise ObservedAnchorAnalysisError("AINC input is not a completed development-only evaluation")
    current_execution = ainc._execution_condition(  # noqa: SLF001
        argparse.Namespace(
            execution_mode=execution.get("mode"),
            temporal_selection_record=(
                execution.get("temporal_selection", {}).get("path")
                if isinstance(execution.get("temporal_selection"), Mapping)
                else None
            ),
        )
    )
    if dict(execution) != current_execution:
        raise ObservedAnchorAnalysisError("AINC execution contingency changed")
    if (
        execution.get("comparison_reference")
        != "continuation_local_C-ABS_required"
        or execution.get("external_temporal_abs_numeric_baseline_allowed") is not False
    ):
        raise ObservedAnchorAnalysisError(
            "AINC comparison is not bound to its continuation-local C-ABS"
        )

    manifest_path = _verified_file_record(
        summary.get("manifest"),
        label="AINC validation manifest",
        extra_keys=(
            "split",
            "clips",
            "episodes",
            "ordered_clip_ids_sha256",
            "ordered_episode_ids_sha256",
        ),
    )
    _, manifest_rows, manifest_record = ainc._manifest_record(  # noqa: SLF001
        manifest_path, split="val", expected_clips=EXPECTED_CLIPS
    )
    if manifest_record != dict(summary["manifest"]) or config.get("manifest") != summary.get("manifest"):
        raise ObservedAnchorAnalysisError("AINC development manifest binding changed")
    validation_anchor_metadata = config.get("anchor_cache")
    validation_semantic_metadata = config.get("semantic_cache")
    validation_cache_attestation = config.get("cache_producer_attestation")
    if (
        not isinstance(validation_anchor_metadata, Mapping)
        or not isinstance(validation_semantic_metadata, Mapping)
        or not isinstance(validation_cache_attestation, Mapping)
        or summary.get("anchor_cache") != validation_anchor_metadata
        or summary.get("semantic_cache") != validation_semantic_metadata
        or summary.get("cache_producer_attestation") != validation_cache_attestation
        or screen.sha256_json(validation_semantic_metadata)
        != abs_evaluation.semantic_cache_identity_sha256
        or screen.sha256_json(validation_cache_attestation)
        != abs_evaluation.cache_producer_attestation_identity_sha256
        or dict(summary["manifest"]) != dict(abs_evaluation.manifest_record)
    ):
        raise ObservedAnchorAnalysisError(
            "AINC and ABS validation cache/population evidence differs"
        )
    anchor_file_record = validation_anchor_metadata.get("anchor_file")
    if not isinstance(anchor_file_record, Mapping) or not isinstance(
        anchor_file_record.get("path"), str
    ):
        raise ObservedAnchorAnalysisError("validation anchor metadata lacks its array")
    validation_anchor_root = (
        Path(str(anchor_file_record["path"])).expanduser().resolve().parent.parent
    )
    revalidated_anchor, _ = anchor.validate_anchor_cache(
        manifest_path=manifest_path,
        anchor_cache_root=validation_anchor_root,
        expected_split="val",
    )
    if revalidated_anchor != dict(validation_anchor_metadata):
        raise ObservedAnchorAnalysisError("validation anchor cache changed")
    ordered_ids = [str(row["clip_id"]) for row in manifest_rows]
    episode_by_clip = {str(row["clip_id"]): int(row["episode_index"]) for row in manifest_rows}
    donor_by_clip = {clip_id: ordered_ids[index ^ 1] for index, clip_id in enumerate(ordered_ids)}
    action_source = {
        ainc.action_control(offset): {
            clip_id: ordered_ids[source]
            for clip_id, source in zip(
                ordered_ids,
                ainc.action_permutation_indices(manifest_rows, offset),
                strict=True,
            )
        }
        for offset in ainc.ACTION_OFFSETS
    }
    if (
        config.get("donor_mapping_sha256") != screen.sha256_json(donor_by_clip)
        or config.get("action_permutations_sha256")
        != {
            name: screen.sha256_json(
                list(ainc.action_permutation_indices(manifest_rows, int(name.rsplit("_", 1)[1])))
            )
            for name in action_source
        }
    ):
        raise ObservedAnchorAnalysisError("AINC control mappings changed")

    science = training_config.get("science_identity")
    complete_path = checkpoint_path.parent.parent / "complete.json"
    complete = vlf.load_json(complete_path, "AINC training completion")
    complete_record = vlf.file_record(complete_path)
    if (
        training_config.get("schema") != ainc.RUN_SCHEMA
        or training_config.get("source") != current_source
        or training_config.get("command") != "train"
        or training_config.get("doe_arm") != ainc.DOE_ARM
        or training_config.get("updates") != ainc.TRAIN_UPDATES
        or not isinstance(science, Mapping)
        or training_config.get("science_identity_sha256")
        != screen.sha256_json(science)
        or training_config.get("normalization") != summary.get("normalization")
        or training_config.get("datasets") != config.get("datasets")
        or science.get("execution_condition") != execution
        or complete.get("schema") != ainc.RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("command") != "train"
        or complete.get("completed_updates") != ainc.TRAIN_UPDATES
        or complete.get("nonfinite_updates") != 0
        or complete.get("checkpoint") != summary.get("checkpoint")
        or complete.get("resolved_config") != summary.get("training_config")
        or complete.get("resolved_config_sha256") != screen.sha256_json(training_config)
        or complete.get("science_identity_sha256")
        != training_config.get("science_identity_sha256")
        or complete.get("protected_test_accessed") is not False
        or complete.get("observed_anchor_enters_model") is not False
        or complete.get("future_rgb_model_input") is not False
        or complete_record.get("path") != str(complete_path)
    ):
        raise ObservedAnchorAnalysisError("AINC training/checkpoint receipt changed")
    normalization_payload = vlf.load_json(
        normalization_path, "AINC normalization payload"
    )
    normalization, normalization_record = anchor.load_increment_normalization(
        normalization_path,
        expected_train_manifest_sha256=str(science["datasets"]["train"]["sha256"]),
        expected_semantic_cache_metadata_sha256=str(
            science["datasets"]["anchor_cache"]["train"]["semantic_cache_metadata_sha256"]
        ),
        expected_anchor_cache_metadata_sha256=str(
            normalization_payload["anchor_cache_metadata_sha256"]
        ),
    )
    del normalization
    if normalization_record != dict(summary["normalization"]):
        raise ObservedAnchorAnalysisError("AINC normalization file record changed")
    calibration_record = training_config.get("calibration_record")
    calibration_path = _verified_file_record(calibration_record, label="AINC calibration completion")
    ainc._validate_calibration(  # noqa: SLF001
        calibration_path,
        science_identity_sha256=str(training_config["science_identity_sha256"]),
    )

    abs_training_config_path = _verified_file_record(
        abs_evaluation.training_config_record, label="ABS primary training config"
    )
    abs_training_config = vlf.load_json(abs_training_config_path, "ABS primary training config")
    _validate_training_fairness(training_config, abs_training_config)
    if execution.get("mode") == "post-temporal-no-pass":
        analysis_path = _verified_file_record(
            execution.get("temporal_development_analysis"),
            label="temporal no-pass analysis",
        )
        temporal_no_pass = vlf.load_json(analysis_path, "temporal no-pass analysis")
        abs_input = temporal_no_pass.get("input_evaluations", {}).get("ABS", {})
        external_abs_summary = (
            abs_input.get("summary") if isinstance(abs_input, Mapping) else None
        )
        if (
            execution.get("temporal_no_pass_is_authorization_only") is not True
            or execution.get("temporal_authorization_source_commit")
            != ainc.TEMPORAL_AUTHORIZATION_COMMIT
            or not isinstance(external_abs_summary, Mapping)
            or dict(external_abs_summary) == dict(abs_evaluation.summary_record)
        ):
            raise ObservedAnchorAnalysisError(
                "running-DOE ABS must authorize only and cannot be the C-ABS baseline"
            )

    rank_shards = summary.get("rank_shards")
    if (
        not isinstance(rank_shards, list)
        or len(rank_shards) != ainc.FROZEN_WORLD_SIZE
        or sorted(int(item.get("rank", -1)) for item in rank_shards if isinstance(item, Mapping))
        != list(range(ainc.FROZEN_WORLD_SIZE))
    ):
        raise ObservedAnchorAnalysisError("AINC rank-shard inventory changed")
    expected_total_calls = 0
    for shard in rank_shards:
        if not isinstance(shard, Mapping):
            raise ObservedAnchorAnalysisError("AINC rank shard is malformed")
        _verified_file_record(shard.get("file"), label="AINC rank metric shard")
        rank = int(shard["rank"])
        local_indices, local_batches = vlf.paired_rank_evaluation_layout(
            EXPECTED_CLIPS,
            ainc.FROZEN_EVAL_BATCH_SIZE,
            rank=rank,
            world_size=ainc.FROZEN_WORLD_SIZE,
        )
        expected_rank_records = (
            len(local_indices) * len(SELECTION_NFE) * len(ainc.CONTROLS)
        )
        expected_rank_calls = (
            len(local_batches)
            * len(ainc.GENERATED_CONTROLS)
            * sum(SELECTION_NFE)
        )
        if (
            shard.get("records") != expected_rank_records
            or shard.get("actual_batched_transformer_calls")
            != expected_rank_calls
        ):
            raise ObservedAnchorAnalysisError(
                "AINC rank shard has incorrect population or call accounting"
            )
        expected_total_calls += expected_rank_calls

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    records = 0
    try:
        with metric_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ObservedAnchorAnalysisError("metric JSONL row is not an object")
                nfe, control, _ = _validate_row(
                    row,
                    checkpoint_sha256=str(summary["checkpoint"]["sha256"]),
                    training_config_sha256=str(summary["training_config"]["sha256"]),
                    evaluation_config_sha256=screen.sha256_json(config),
                    normalization_sha256=str(summary["normalization"]["sha256"]),
                    donor_by_clip=donor_by_clip,
                    episode_by_clip=episode_by_clip,
                    action_source=action_source,
                )
                groups.setdefault((nfe, control), []).append(row)
                records += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObservedAnchorAnalysisError("cannot parse AINC per-clip metrics") from exc
    expected_cells = {(nfe, control) for nfe in SELECTION_NFE for control in ainc.CONTROLS}
    expected_records = EXPECTED_CLIPS * len(expected_cells)
    if (
        set(groups) != expected_cells
        or records != expected_records
        or summary.get("record_count") != expected_records
        or summary.get("cell_count") != len(expected_cells)
        or sum(int(shard.get("records", -1)) for shard in rank_shards) != expected_records
        or summary.get("actual_batched_transformer_calls") != expected_total_calls
    ):
        raise ObservedAnchorAnalysisError("AINC evaluation population is incomplete")
    cells = {key: _cell_from_rows(key[0], key[1], rows) for key, rows in groups.items()}
    reference_clips = abs_evaluation.cells[(1, "autonomous")].clip_ids
    for nfe in SELECTION_NFE:
        auto = cells[(nfe, "autonomous")]
        donor = cells[(nfe, "donor_target")]
        shuffled_anchor = cells[(nfe, "anchor_decode_shuffled")]
        if auto.clip_ids != reference_clips:
            raise ObservedAnchorAnalysisError(
                "AINC and continuation-local C-ABS clip identities are not paired"
            )
        if auto.pairing_identity_sha256 != abs_evaluation.cells[(nfe, "autonomous")].pairing_identity_sha256:
            raise ObservedAnchorAnalysisError(
                "AINC and continuation-local C-ABS immutable sampler inputs differ"
            )
        if (
            donor.generation_identity_sha256 != auto.generation_identity_sha256
            or donor.increment_identity_sha256 != auto.increment_identity_sha256
            or shuffled_anchor.increment_identity_sha256 != auto.increment_identity_sha256
            or donor.sampling_identity_sha256 != auto.sampling_identity_sha256
            or shuffled_anchor.sampling_identity_sha256
            != auto.sampling_identity_sha256
        ):
            raise ObservedAnchorAnalysisError("reuse controls changed autonomous generation")
        if not np.allclose(
            shuffled_anchor.metrics["temporal_difference_nmse"],
            auto.metrics["temporal_difference_nmse"],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ObservedAnchorAnalysisError(
                "decode-anchor shuffle unexpectedly changed temporal differences"
            )
    return EvaluationData(
        summary_record=vlf.file_record(resolved_summary),
        checkpoint_record=dict(summary["checkpoint"]),
        training_config_record=dict(summary["training_config"]),
        training_config=training_config,
        execution_condition=dict(execution),
        manifest_record=dict(summary["manifest"]),
        cells=cells,
    )


def _effect(
    candidate: np.ndarray,
    control: np.ndarray,
    indices: np.ndarray,
    *,
    kind: str,
    confidence: float,
) -> dict[str, Any]:
    return temporal_analysis.paired_effect(
        candidate,
        control,
        indices,
        kind=kind,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _passes_relative(effect: Mapping[str, Any]) -> bool:
    return (
        float(effect["point_estimate"]) >= MIN_RELATIVE_IMPROVEMENT
        and float(effect["one_sided_lower_bound"]) >= MIN_RELATIVE_IMPROVEMENT
    )


def _passes_positive(effect: Mapping[str, Any]) -> bool:
    return float(effect["point_estimate"]) > 0.0 and float(
        effect["one_sided_lower_bound"]
    ) > 0.0


def build_analysis(
    evaluation: EvaluationData,
    abs_evaluation: temporal_analysis.EvaluationData,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    confidence: float = CELLWISE_CONFIDENCE,
) -> dict[str, Any]:
    clip_ids = evaluation.cells[(1, "autonomous")].clip_ids
    indices = temporal_analysis.common_bootstrap_indices(
        len(clip_ids), samples=bootstrap_samples, seed=bootstrap_seed
    )
    cells: list[dict[str, Any]] = []
    for nfe in SELECTION_NFE:
        autonomous = evaluation.cells[(nfe, "autonomous")]
        abs_auto = abs_evaluation.cells[(nfe, "autonomous")]
        effects: dict[str, Any] = {
            "semantic_nmse_vs_c_abs": _effect(
                autonomous.metrics["semantic_nmse"],
                abs_auto.metrics["semantic_nmse"],
                indices,
                kind="relative_lower",
                confidence=confidence,
            ),
            "temporal_nmse_vs_c_abs": _effect(
                autonomous.metrics["temporal_difference_nmse"],
                abs_auto.metrics["temporal_difference_nmse"],
                indices,
                kind="relative_lower",
                confidence=confidence,
            ),
        }
        for control in TEMPORAL_CONTROLS:
            effects[f"temporal_nmse_vs_{control}"] = _effect(
                autonomous.metrics["temporal_difference_nmse"],
                evaluation.cells[(nfe, control)].metrics[
                    "temporal_difference_nmse"
                ],
                indices,
                kind="relative_lower",
                confidence=confidence,
            )
        effects["semantic_nmse_vs_anchor_decode_shuffled"] = _effect(
            autonomous.metrics["semantic_nmse"],
            evaluation.cells[(nfe, "anchor_decode_shuffled")].metrics[
                "semantic_nmse"
            ],
            indices,
            kind="relative_lower",
            confidence=confidence,
        )
        effects["semantic_cosine_vs_anchor_decode_shuffled"] = _effect(
            autonomous.metrics["semantic_token_cosine"],
            evaluation.cells[(nfe, "anchor_decode_shuffled")].metrics[
                "semantic_token_cosine"
            ],
            indices,
            kind="difference_higher",
            confidence=confidence,
        )
        means = {
            name: float(values.mean()) for name, values in autonomous.metrics.items()
        }
        absolute_gate = (
            means["semantic_nmse"] <= MAX_NMSE
            and means["semantic_token_cosine"] >= MIN_COSINE
            and means["temporal_difference_nmse"] <= MAX_NMSE
        )
        relative_gate = all(
            _passes_relative(effect)
            for name, effect in effects.items()
            if name != "semantic_cosine_vs_anchor_decode_shuffled"
        )
        cosine_gate = _passes_positive(
            effects["semantic_cosine_vs_anchor_decode_shuffled"]
        )
        passed = absolute_gate and relative_gate and cosine_gate
        cells.append(
            {
                "doe_arm": ainc.DOE_ARM,
                "nfe": nfe,
                "paired_clips": len(clip_ids),
                "autonomous_means": means,
                "absolute_quality_gate_passed": absolute_gate,
                "effects": effects,
                "all_relative_nmse_gates_passed": relative_gate,
                "anchor_cosine_attribution_gate_passed": cosine_gate,
                "passed": passed,
            }
        )
    passing = [cell for cell in cells if cell["passed"]]
    selected = None
    if passing and evaluation.execution_condition.get("semantic_screen_promotion_eligible") is True:
        selected = min(
            passing,
            key=lambda cell: (
                int(cell["nfe"]),
                float(cell["autonomous_means"]["temporal_difference_nmse"]),
                float(cell["autonomous_means"]["semantic_nmse"]),
            ),
        )
    proxy = evaluation.execution_condition.get("proxy_validity_only") is True
    status = (
        "proxy_nonpromotable"
        if proxy
        else "one_candidate_selected"
        if selected is not None
        else "no_candidate_passed"
    )
    return temporal_analysis.identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": status,
            "source": ainc._source_record(),  # noqa: SLF001
            "split": "val",
            "development_selection_split": True,
            "protected_test_accessed": False,
            "protected_test_cache_opened": False,
            "protocol_frozen": True,
            "prospective_mathematical_correction_applied": True,
            "execution_condition": dict(evaluation.execution_condition),
            "comparison_design": {
                "kind": "self_contained_matched_two_arm_continuation",
                "control_display_name": "C-ABS",
                "control_machine_arm": "ABS",
                "same_clean_commit_required": True,
                "running_temporal_doe_used_for_authorization_only": (
                    evaluation.execution_condition.get("mode")
                    == "post-temporal-no-pass"
                ),
                "external_temporal_abs_numeric_baseline_allowed": False,
            },
            "input_evaluations": {
                "AINC-OFF": dict(evaluation.summary_record),
                "ABS": dict(abs_evaluation.summary_record),
            },
            "paired_clips": len(clip_ids),
            "ordered_clip_ids_sha256": screen.sha256_json(list(clip_ids)),
            "bootstrap": {
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "unit": "paired immutable development clip",
                "common_indices_across_all_metrics_cells_controls": True,
                "indices_sha256": temporal_analysis._array_sha256(indices),  # noqa: SLF001
                "one_sided_confidence_per_candidate_cell": confidence,
                "alpha_per_candidate_cell": 1.0 - confidence,
                "bonferroni_candidate_cells": len(SELECTION_NFE),
                "intersection_union_within_cell": True,
                "ratio_effect": "ratio of resampled means, never mean of clip ratios",
            },
            "thresholds": {
                "autonomous_semantic_nmse_max": MAX_NMSE,
                "autonomous_semantic_token_cosine_min": MIN_COSINE,
                "autonomous_temporal_difference_nmse_max": MAX_NMSE,
                "relative_nmse_point_and_lower_bound_min": MIN_RELATIVE_IMPROVEMENT,
                "anchor_decode_semantic_cosine_point_and_lower_bound": "strictly_positive",
            },
            "candidate_cells": cells,
            "selected_cell": selected,
            "selection_count": int(selected is not None),
            "passing_cells_before_contingency": len(passing),
            "proxy_results_are_nonpromotable": proxy,
            "deterministic_tie_break": [
                "smallest_nfe",
                "lowest_autonomous_temporal_difference_nmse",
                "lowest_autonomous_semantic_nmse",
            ],
            "increment_metrics_promotion_eligible": False,
            "action_permutations_promotion_eligible": False,
        }
    )


def _selection_record(
    analysis: Mapping[str, Any], analysis_record: Mapping[str, Any]
) -> dict[str, Any]:
    selected = analysis.get("selected_cell")
    promotable = (
        selected is not None
        and analysis.get("proxy_results_are_nonpromotable") is False
        and analysis.get("protocol_frozen") is True
    )
    return temporal_analysis.identity_payload(
        {
            "schema": SELECTION_SCHEMA,
            "status": (
                "frozen_candidate"
                if promotable
                else "frozen_proxy_no_selection"
                if analysis.get("proxy_results_are_nonpromotable") is True
                else "frozen_no_selection"
            ),
            "development_analysis": dict(analysis_record),
            "development_analysis_identity_sha256": analysis["identity_sha256"],
            "selected_cell": selected if promotable else None,
            "selection_count": int(promotable),
            "selection_split": "val",
            "selection_used_protected_test": False,
            "protected_test_accessed": False,
            "lockbox_may_open": bool(promotable),
            "lockbox_not_opened_by_this_tool": True,
            "tie_break": analysis["deterministic_tie_break"],
            "input_evaluations": analysis["input_evaluations"],
        }
    )


def analysis_command(args: argparse.Namespace) -> int:
    abs_evaluation = temporal_analysis.load_evaluation(
        args.abs_summary, expected_arm="ABS"
    )
    evaluation = load_evaluation(
        args.ainc_summary, abs_evaluation=abs_evaluation
    )
    analysis = build_analysis(evaluation, abs_evaluation)
    output_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    analysis_path = output_dir / "development_analysis.json"
    vlf.atomic_write_json(analysis_path, analysis, exclusive=True)
    selection = _selection_record(analysis, vlf.file_record(analysis_path))
    vlf.atomic_write_json(output_dir / "frozen_selection.json", selection, exclusive=True)
    print(screen.canonical_json(selection))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ainc-summary", required=True)
    parser.add_argument("--abs-summary", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return analysis_command(args)
    except (
        ObservedAnchorAnalysisError,
        temporal_analysis.TemporalAnalysisError,
        ainc.ObservedAnchorScreenError,
        anchor.ObservedAnchorError,
        screen.ScreenError,
        vlf.PocError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"AINC development analysis error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
