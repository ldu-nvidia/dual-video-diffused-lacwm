#!/usr/bin/env python3
"""Paired development analysis for the temporal-target video DOE.

This tool consumes exactly one completed validation evaluation for each of
ABS, ABS-T, DELTA, DELTA-T, and DELTA-R.  It never opens a protected-test
artifact.  A single clip-index bootstrap matrix is shared across all metrics,
controls, and 12 selectable cells, and at most one arm/NFE pair is frozen for
the later one-shot lockbox evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import evaluate_causal_vjepa2_temporal_targets as evaluator  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402


ANALYSIS_SCHEMA = "causal-vjepa2-temporal-target-development-analysis-v1"
SELECTION_SCHEMA = "causal-vjepa2-temporal-target-frozen-selection-v1"
ARMS = evaluator.DOE_ARMS
CANDIDATE_ARMS = ("ABS-T", "DELTA", "DELTA-T", "DELTA-R")
ARM_ORDER = {arm: index for index, arm in enumerate(CANDIDATE_ARMS)}
SELECTION_NFE = (1, 2, 4)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260807
CELLWISE_CONFIDENCE = 1.0 - 0.05 / (len(CANDIDATE_ARMS) * len(SELECTION_NFE))
EXPECTED_CLIPS = screen.FROZEN_VALIDATION_CLIPS
MIN_RELATIVE_IMPROVEMENT = 0.05
MAX_NMSE = 0.50
MIN_COSINE = 0.70
LOWER_METRICS = ("semantic_nmse", "temporal_difference_nmse")
PRIMARY_METRICS = (
    "semantic_nmse",
    "semantic_token_cosine",
    "temporal_difference_nmse",
    "temporal_difference_token_cosine",
)
ALL_METRICS = (*evaluator.PRIMARY_METRICS, *evaluator.PACKED_METRICS)
GATE_CONTROLS = ("donor_target", "context_shuffled")
EffectKind = Literal["relative_lower", "difference_higher"]


class TemporalAnalysisError(RuntimeError):
    """An input artifact or paired-analysis contract failed closed."""


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(screen.canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def common_bootstrap_indices(
    clips: int,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Construct the one shared clip-resampling matrix for the whole screen."""
    if clips < 2 or samples < 100 or seed < 0:
        raise ValueError("bootstrap requires >=2 clips, >=100 samples, and a seed")
    return np.random.default_rng(seed).integers(
        0, clips, size=(samples, clips), dtype=np.int32
    )


def _bootstrap_effects(
    candidate: np.ndarray,
    control: np.ndarray,
    indices: np.ndarray,
    *,
    kind: EffectKind,
) -> np.ndarray:
    if (
        candidate.ndim != 1
        or control.shape != candidate.shape
        or indices.ndim != 2
        or indices.shape[1] != candidate.size
    ):
        raise ValueError("paired vectors and bootstrap indices have incompatible shapes")
    if not (
        np.isfinite(candidate).all()
        and np.isfinite(control).all()
        and np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("paired bootstrap inputs must be finite")
    output = np.empty(indices.shape[0], dtype=np.float64)
    # Bound peak memory while preserving the exact common index matrix.
    for start in range(0, indices.shape[0], 256):
        selected = indices[start : start + 256]
        candidate_mean = candidate[selected].mean(axis=1)
        control_mean = control[selected].mean(axis=1)
        if kind == "relative_lower":
            if np.any(control_mean <= 0):
                raise TemporalAnalysisError(
                    "relative-improvement bootstrap has non-positive control mean"
                )
            output[start : start + len(selected)] = (
                control_mean - candidate_mean
            ) / control_mean
        elif kind == "difference_higher":
            output[start : start + len(selected)] = candidate_mean - control_mean
        else:  # pragma: no cover - Literal plus defensive runtime guard
            raise ValueError(f"unknown paired effect kind: {kind}")
    if not np.isfinite(output).all():
        raise TemporalAnalysisError("paired bootstrap produced a non-finite effect")
    return output


def paired_effect(
    candidate: np.ndarray,
    control: np.ndarray,
    indices: np.ndarray,
    *,
    kind: EffectKind,
    confidence: float = CELLWISE_CONFIDENCE,
) -> dict[str, Any]:
    """Compute a ratio-of-means improvement or mean cosine advantage."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("one-sided confidence must lie in (0.5,1)")
    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    candidate_mean = float(candidate.mean())
    control_mean = float(control.mean())
    if kind == "relative_lower":
        if control_mean <= 0:
            raise TemporalAnalysisError("relative improvement has non-positive control")
        point = (control_mean - candidate_mean) / control_mean
        definition = "(mean(control)-mean(candidate))/mean(control)"
    else:
        point = candidate_mean - control_mean
        definition = "mean(candidate)-mean(control)"
    effects = _bootstrap_effects(candidate, control, indices, kind=kind)
    alpha = 1.0 - confidence
    lower = float(np.quantile(effects, alpha, method="linear"))
    return {
        "kind": kind,
        "definition": definition,
        "paired_clips": int(candidate.size),
        "candidate_mean": candidate_mean,
        "control_mean": control_mean,
        "point_estimate": float(point),
        "one_sided_lower_bound": lower,
        "confidence": confidence,
        "alpha": alpha,
        "bootstrap_samples": int(indices.shape[0]),
        "bootstrap_effect_mean": float(effects.mean()),
        "bootstrap_effect_std": float(effects.std(ddof=1)),
    }


@dataclass(frozen=True)
class CellData:
    arm: str
    nfe: int
    control: str
    clip_ids: tuple[str, ...]
    metrics: Mapping[str, np.ndarray]
    pairing_identity_sha256: str
    generation_identity_sha256: str


@dataclass(frozen=True)
class EvaluationData:
    arm: str
    target_mode: str
    normalization_binding_sha256: str
    summary_record: Mapping[str, Any]
    checkpoint_record: Mapping[str, Any]
    training_config_record: Mapping[str, Any]
    training_doe_common_identity_sha256: str
    calibration_receipt_identity_sha256: str
    implementation_registration_identity_sha256: str
    preregistration_identity_sha256: str
    manifest_record: Mapping[str, Any]
    semantic_cache_identity_sha256: str
    cache_producer_attestation_identity_sha256: str
    cells: Mapping[tuple[int, str], CellData]


def _verified_file_record(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise TemporalAnalysisError(f"{label} lacks a file record")
    path = Path(value["path"]).expanduser().resolve()
    if path.is_symlink() or vlf.file_record(path) != dict(value):
        raise TemporalAnalysisError(f"{label} changed or is not a regular immutable file")
    return path


def _finite_metric(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalAnalysisError(f"metric row lacks numeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise TemporalAnalysisError(f"metric row has non-finite {name}")
    return result


def _validate_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    target_mode: str,
    normalization_binding: str,
    normalization_artifact_sha256: str | None,
    checkpoint_sha256: str,
    training_config_sha256: str,
    evaluation_config_sha256: str,
    implementation_registration_identity_sha256: str,
    calibration_checkpoint_sha256: str,
    donor_by_clip: Mapping[str, str],
    action_source_by_control: Mapping[str, Mapping[str, str]],
    episode_by_clip: Mapping[str, int],
) -> tuple[int, str, str]:
    try:
        nfe = int(row["nfe"])
        control = str(row["control"])
        clip_id = str(row["clip_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalAnalysisError("metric row lacks its cell identity") from exc
    call_hashes = row.get("model_call_input_sha256")
    sampler = row.get("sampler_input")
    hash_fields = (
        "generated_semantic_sha256",
        "generated_representation_sha256",
        "metric_target_sha256",
        "metric_representation_target_sha256",
        "model_call_input_chain_sha256",
    )
    if (
        row.get("schema") != evaluator.EVALUATION_SCHEMA
        or row.get("arm") != arm
        or row.get("target_mode") != target_mode
        or row.get("normalization_binding_sha256") != normalization_binding
        or row.get("normalization_artifact_sha256")
        != normalization_artifact_sha256
        or row.get("checkpoint_sha256") != checkpoint_sha256
        or row.get("training_config_sha256") != training_config_sha256
        or row.get("evaluation_config_sha256") != evaluation_config_sha256
        or row.get("implementation_registration_identity_sha256")
        != implementation_registration_identity_sha256
        or row.get("calibration_checkpoint_sha256")
        != calibration_checkpoint_sha256
        or nfe not in evaluator.NFE_GRID
        or control not in evaluator.CONTROLS
        or not clip_id
        or row.get("evaluation_seed") != evaluator.EVALUATION_SEED
        or row.get("clean_future_target_entered_sampler") is not False
        or row.get("future_rgb_entered_sampler") is not False
        or row.get("teacher_model_calls") != 0
        or row.get("trajectory_state_dtype") != "float32"
        or row.get("transformer_autocast") != "cuda-bfloat16"
        or row.get("generation_deployable")
        is not (control in evaluator.GENERATED_CONTROLS)
        or row.get("control_deployable")
        is not (control in evaluator.GENERATED_CONTROLS)
        or row.get("metric_comparison_only")
        is not (control in {"donor_target", "zero", "oracle_clean"})
        or not isinstance(call_hashes, list)
        or any(not isinstance(value, str) or not screen.HEX64.fullmatch(value) for value in call_hashes)
        or any(
            not isinstance(row.get(name), str)
            or not screen.HEX64.fullmatch(str(row.get(name)))
            for name in hash_fields
        )
        or row.get("model_call_input_chain_sha256")
        != screen.sha256_json(call_hashes)
    ):
        raise TemporalAnalysisError("metric row violates the generated-only contract")
    donor_clip = donor_by_clip.get(clip_id)
    history_source = row.get("history_source_clip_id")
    actions_source = row.get("actions_source_clip_id")
    target_source = row.get("target_source_clip_id")
    provenance_valid = False
    if control == "autonomous":
        provenance_valid = (
            history_source == clip_id
            and actions_source == clip_id
            and target_source == clip_id
        )
    elif control == "donor_target":
        provenance_valid = (
            history_source == clip_id
            and actions_source == clip_id
            and target_source == donor_clip
        )
    elif control == "context_shuffled":
        provenance_valid = (
            history_source == donor_clip
            and actions_source == donor_clip
            and target_source == clip_id
        )
    elif control == "history_shuffled":
        provenance_valid = (
            history_source == donor_clip
            and actions_source == clip_id
            and target_source == clip_id
        )
    elif control in evaluator.ACTION_CONTROLS:
        provenance_valid = (
            history_source == clip_id
            and actions_source == action_source_by_control[control].get(clip_id)
            and target_source == clip_id
        )
    elif control in {"zero", "oracle_clean"}:
        provenance_valid = (
            history_source is None
            and actions_source is None
            and target_source == clip_id
        )
    if (
        row.get("episode_index") != episode_by_clip.get(clip_id)
        or row.get("donor_clip_id") != donor_clip
        or row.get("donor_episode_index") != episode_by_clip.get(str(donor_clip))
        or not provenance_valid
    ):
        raise TemporalAnalysisError("control-specific source provenance changed")
    if control in evaluator.GENERATED_CONTROLS:
        expected_actual = nfe
        expected_conceptual = nfe
        expected_hashes = nfe
        if not isinstance(sampler, Mapping) or row.get("generation_reused_from") is not None:
            raise TemporalAnalysisError("generated row lacks sampler inputs")
    elif control == "donor_target":
        expected_actual = 0
        expected_conceptual = nfe
        expected_hashes = nfe
        if row.get("generation_reused_from") != "autonomous" or not isinstance(
            sampler, Mapping
        ):
            raise TemporalAnalysisError("donor row did not reuse autonomous generation")
    else:
        expected_actual = expected_conceptual = expected_hashes = 0
        if sampler is not None or row.get("generation_reused_from") is not None:
            raise TemporalAnalysisError("metric-only row unexpectedly has sampler inputs")
    if isinstance(sampler, Mapping):
        try:
            computed_sampler_hash = evaluator.sampler_input_sha256(sampler)
        except (ValueError, TypeError) as exc:
            raise TemporalAnalysisError("metric row has malformed sampler inputs") from exc
        if row.get("sampler_input_sha256") != computed_sampler_hash:
            raise TemporalAnalysisError("metric row sampler-input hash changed")
    elif row.get("sampler_input_sha256") is not None:
        raise TemporalAnalysisError("metric-only row has a sampler-input hash")
    if (
        row.get("actual_evaluator_model_calls") != expected_actual
        or row.get("conceptual_path_model_calls") != expected_conceptual
        or len(call_hashes) != expected_hashes
    ):
        raise TemporalAnalysisError("metric row has incorrect actual-call accounting")
    for name in ALL_METRICS:
        _finite_metric(row, name)
    return nfe, control, clip_id


def _cell_from_rows(
    arm: str,
    nfe: int,
    control: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_clips: int,
) -> CellData:
    ordered = sorted(rows, key=lambda value: str(value["clip_id"]))
    clip_ids = tuple(str(row["clip_id"]) for row in ordered)
    if len(ordered) != expected_clips or len(set(clip_ids)) != expected_clips:
        raise TemporalAnalysisError(
            f"{arm}/{control}/NFE{nfe} does not contain {expected_clips} unique clips"
        )
    pairing_rows = []
    generation_rows = []
    for row in ordered:
        sampler = row.get("sampler_input")
        sampler_pair = None
        if isinstance(sampler, Mapping):
            sampler_pair = {
                name: sampler.get(name)
                for name in (
                    "history_sha256",
                    "actions_sha256",
                    "initial_video_noise_sha256",
                    "initial_representation_noise_sha256",
                )
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
                "generated_semantic_sha256": row["generated_semantic_sha256"],
                "generated_representation_sha256": row[
                    "generated_representation_sha256"
                ],
                "model_call_input_chain_sha256": row[
                    "model_call_input_chain_sha256"
                ],
            }
        )
    metrics = {
        name: np.asarray([_finite_metric(row, name) for row in ordered], dtype=np.float64)
        for name in ALL_METRICS
    }
    return CellData(
        arm=arm,
        nfe=nfe,
        control=control,
        clip_ids=clip_ids,
        metrics=metrics,
        pairing_identity_sha256=screen.sha256_json(pairing_rows),
        generation_identity_sha256=screen.sha256_json(generation_rows),
    )


def load_evaluation(
    summary_path: str | Path,
    *,
    expected_arm: str,
    expected_clips: int = EXPECTED_CLIPS,
) -> EvaluationData:
    """Load one completed development artifact into compact metric arrays."""
    resolved_summary = Path(summary_path).expanduser().resolve()
    if resolved_summary.is_symlink() or not resolved_summary.is_file():
        raise TemporalAnalysisError("evaluation summary must be a regular file")
    summary = vlf.load_json(resolved_summary, f"{expected_arm} evaluation summary")
    config_path = _verified_file_record(summary.get("resolved_config"), label="config")
    config = vlf.load_json(config_path, "evaluation config")
    provenance_path = _verified_file_record(
        summary.get("provenance"), label="evaluation provenance"
    )
    provenance = vlf.load_json(provenance_path, "evaluation provenance")
    current_source = screen._source_record()  # noqa: SLF001
    if (
        expected_arm not in ARMS
        or summary.get("schema") != evaluator.EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("split") != "val"
        or summary.get("development_selection_split") is not True
        or summary.get("protected_test_accessed") is not False
        or summary.get("arm") != expected_arm
        or summary.get("clean_future_target_entered_deployable_sampler") is not False
        or summary.get("future_rgb_entered_deployable_sampler") is not False
        or summary.get("teacher_model_calls") != 0
        or config.get("schema") != evaluator.EVALUATION_SCHEMA
        or config.get("status") != "configured"
        or config.get("source") != current_source
        or config.get("entrypoint") != vlf.file_record(evaluator.__file__)
        or config.get("split") != "val"
        or config.get("protected_test_accessed") is not False
        or config.get("arm") != expected_arm
        or config.get("controls") != list(evaluator.CONTROLS)
        or config.get("nfe_grid") != list(evaluator.NFE_GRID)
        or config.get("selection_eligible_nfe") != list(SELECTION_NFE)
        or config.get("generated_controls") != list(evaluator.GENERATED_CONTROLS)
        or config.get("world_size") != evaluator.FROZEN_TRAIN_WORLD_SIZE
        or config.get("sampler")
        != {
            "schedule": "uniform-clean-time-euler",
            "trajectory_state_dtype": "float32",
            "transformer_autocast": "cuda-bfloat16",
            "actual_calls_equal_nfe": True,
            "clean_target_argument": "forbidden",
            "future_rgb_argument": "forbidden",
            "teacher_argument": "forbidden",
            "delta_decode": "after_generated_trajectory_only",
        }
        or provenance.get("schema") != evaluator.EVALUATION_SCHEMA
        or provenance.get("source") != current_source
        or provenance.get("resolved_config_sha256") != screen.sha256_json(config)
        or provenance.get("protected_test_accessed") is not False
        or provenance.get("secrets_persisted") is not False
    ):
        raise TemporalAnalysisError(
            f"{expected_arm} input is not a completed development-only evaluation"
        )
    target_mode = str(summary.get("target_mode"))
    normalization_binding = str(summary.get("normalization_binding_sha256"))
    normalization_record = summary.get("normalization")
    normalization_required = evaluator.ARM_CONTRACTS[expected_arm]["normalization"]
    if normalization_required:
        if (
            not isinstance(normalization_record, Mapping)
            or not isinstance(normalization_record.get("sha256"), str)
            or not screen.HEX64.fullmatch(str(normalization_record["sha256"]))
            or normalization_record.get("sha256")
            != evaluator.FROZEN_D1_NORMALIZATION_SHA256
        ):
            raise TemporalAnalysisError("evaluation lacks its normalization artifact")
        if set(normalization_record) != {"path", "sha256", "bytes", "payload_sha256"}:
            raise TemporalAnalysisError("normalization file record shape changed")
        normalization_path = _verified_file_record(
            {
                name: normalization_record[name]
                for name in ("path", "sha256", "bytes")
            },
            label=f"{expected_arm} D1 normalization",
        )
        normalization_payload = vlf.load_json(
            normalization_path, f"{expected_arm} D1 normalization"
        )
        if normalization_record.get("payload_sha256") != screen.sha256_json(
            normalization_payload
        ):
            raise TemporalAnalysisError("normalization payload identity changed")
        normalization_artifact_sha256: str | None = str(
            normalization_record["sha256"]
        )
    else:
        if normalization_record is not None:
            raise TemporalAnalysisError("ABS unexpectedly binds normalization")
        normalization_artifact_sha256 = None
    if (
        target_mode != evaluator.ARM_CONTRACTS[expected_arm]["target_mode"]
        or not screen.HEX64.fullmatch(normalization_binding)
        or config.get("target_mode") != target_mode
        or config.get("normalization") != normalization_record
        or config.get("normalization_binding_sha256") != normalization_binding
    ):
        raise TemporalAnalysisError("evaluation target representation binding changed")
    checkpoint_binding = summary.get("checkpoint")
    training_config_binding = summary.get("training_config")
    if (
        not isinstance(checkpoint_binding, Mapping)
        or not isinstance(training_config_binding, Mapping)
        or not isinstance(checkpoint_binding.get("sha256"), str)
        or not isinstance(training_config_binding.get("sha256"), str)
        or not screen.HEX64.fullmatch(str(checkpoint_binding.get("sha256")))
        or not screen.HEX64.fullmatch(str(training_config_binding.get("sha256")))
    ):
        raise TemporalAnalysisError("summary lacks checkpoint/training hash bindings")
    _verified_file_record(
        checkpoint_binding, label=f"{expected_arm} primary checkpoint"
    )
    training_config_path = _verified_file_record(
        training_config_binding, label="primary training config"
    )
    primary_training_config = vlf.load_json(
        training_config_path, "primary training config"
    )
    training_common_identity = summary.get("training_doe_common_identity_sha256")
    calibration_receipt = summary.get("calibration_receipt")
    implementation_registration_record = summary.get("implementation_registration")
    implementation_registration_identity = summary.get(
        "implementation_registration_identity_sha256"
    )
    registered_protocol = summary.get("preregistration")
    if (
        not isinstance(training_common_identity, str)
        or not screen.HEX64.fullmatch(training_common_identity)
        or training_common_identity
        != evaluator.training_doe_common_identity(primary_training_config)
        or config.get("training_doe_common_identity_sha256")
        != training_common_identity
        or not isinstance(calibration_receipt, Mapping)
        or config.get("calibration_receipt") != calibration_receipt
        or calibration_receipt.get("schema")
        != "causal-vjepa2-temporal-arm-calibration-receipt-v1"
        or calibration_receipt.get("arm") != expected_arm
        or calibration_receipt.get("updates") != screen.CALIBRATION_UPDATES
        or calibration_receipt.get(
            "primary_equivalent_except_updates_and_run_role"
        )
        is not True
        or calibration_receipt.get("nonfinite_updates") != 0
        or not isinstance(implementation_registration_record, Mapping)
        or config.get("implementation_registration")
        != implementation_registration_record
        or not isinstance(implementation_registration_identity, str)
        or not screen.HEX64.fullmatch(implementation_registration_identity)
        or config.get("implementation_registration_identity_sha256")
        != implementation_registration_identity
        or not isinstance(registered_protocol, Mapping)
        or config.get("preregistration") != registered_protocol
        or registered_protocol != evaluator.preregistration_attestation(current_source)
    ):
        raise TemporalAnalysisError(
            "training DOE, calibration, implementation, or preregistration binding changed"
        )
    calibration_paths = {
        name: _verified_file_record(
            calibration_receipt.get(name), label=f"{expected_arm} calibration {name}"
        )
        for name in ("checkpoint", "resolved_config", "complete")
    }
    calibration_config = vlf.load_json(
        calibration_paths["resolved_config"], "calibration config"
    )
    primary_calibration_common = evaluator._primary_calibration_common(  # noqa: SLF001
        primary_training_config
    )
    if (
        evaluator._primary_calibration_common(calibration_config)  # noqa: SLF001
        != primary_calibration_common
        or calibration_receipt.get("common_config_sha256")
        != screen.sha256_json(primary_calibration_common)
    ):
        raise TemporalAnalysisError("calibration config is not primary-equivalent")
    registration_path = _verified_file_record(
        implementation_registration_record, label="implementation registration"
    )
    registration_payload, recomputed_registration_record = (
        evaluator.load_implementation_registration(
            registration_path, current_source=current_source
        )
    )
    if (
        recomputed_registration_record != dict(implementation_registration_record)
        or registration_payload.get("identity_sha256")
        != implementation_registration_identity
    ):
        raise TemporalAnalysisError("implementation registration identity changed")
    manifest_binding = summary.get("manifest")
    if not isinstance(manifest_binding, Mapping):
        raise TemporalAnalysisError("summary lacks development manifest binding")
    manifest_path = _verified_file_record(
        manifest_binding, label="development manifest"
    )
    _, manifest_rows, recomputed_manifest = screen._manifest_record(  # noqa: SLF001
        manifest_path, split="val", expected_clips=expected_clips
    )
    if recomputed_manifest != dict(manifest_binding):
        raise TemporalAnalysisError("development manifest identity changed")
    ordered_clip_ids = [str(row["clip_id"]) for row in manifest_rows]
    episode_by_clip = {
        str(row["clip_id"]): int(row["episode_index"]) for row in manifest_rows
    }
    donor_by_clip = {
        clip_id: ordered_clip_ids[index ^ 1]
        for index, clip_id in enumerate(ordered_clip_ids)
    }
    action_source_by_control = {}
    for offset in evaluator.ACTION_PERMUTATION_OFFSETS:
        control_name = evaluator.action_control(offset)
        permutation = evaluator.action_permutation_indices(expected_clips, offset)
        action_source_by_control[control_name] = {
            clip_id: ordered_clip_ids[permutation[index]]
            for index, clip_id in enumerate(ordered_clip_ids)
        }
    if (
        config.get("action_permutations")
        != evaluator.validate_action_permutations(manifest_rows)
        or config.get("donor_mapping_sha256")
        != screen.sha256_json(donor_by_clip)
    ):
        raise TemporalAnalysisError("registered control mappings changed")
    rank_shards = summary.get("rank_shards")
    if (
        not isinstance(rank_shards, list)
        or len(rank_shards) != evaluator.FROZEN_TRAIN_WORLD_SIZE
        or not all(isinstance(shard, Mapping) for shard in rank_shards)
        or sorted(int(shard.get("rank", -1)) for shard in rank_shards)
        != list(range(evaluator.FROZEN_TRAIN_WORLD_SIZE))
    ):
        raise TemporalAnalysisError("evaluation rank-shard inventory changed")
    for shard in rank_shards:
        _verified_file_record(shard.get("file"), label="evaluation rank shard")
    metric_path = _verified_file_record(
        summary.get("per_clip_metrics"), label="per-clip metrics"
    )
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    records = 0
    try:
        with metric_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TemporalAnalysisError("metric JSONL row is not an object")
                nfe, control, _ = _validate_row(
                    row,
                    arm=expected_arm,
                    target_mode=target_mode,
                    normalization_binding=normalization_binding,
                    normalization_artifact_sha256=normalization_artifact_sha256,
                    checkpoint_sha256=str(checkpoint_binding["sha256"]),
                    training_config_sha256=str(training_config_binding["sha256"]),
                    evaluation_config_sha256=screen.sha256_json(config),
                    implementation_registration_identity_sha256=(
                        implementation_registration_identity
                    ),
                    calibration_checkpoint_sha256=str(
                        calibration_receipt["checkpoint"]["sha256"]
                    ),
                    donor_by_clip=donor_by_clip,
                    action_source_by_control=action_source_by_control,
                    episode_by_clip=episode_by_clip,
                )
                groups.setdefault((nfe, control), []).append(row)
                records += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemporalAnalysisError("cannot parse immutable per-clip metrics") from exc
    expected_cells = {
        (nfe, control) for nfe in evaluator.NFE_GRID for control in evaluator.CONTROLS
    }
    expected_records = expected_clips * len(expected_cells)
    if (
        set(groups) != expected_cells
        or records != expected_records
        or summary.get("record_count") != expected_records
        or sum(int(shard.get("records", -1)) for shard in rank_shards)
        != expected_records
        or summary.get("actual_batched_transformer_calls")
        != sum(
            int(shard.get("actual_batched_transformer_calls", -1))
            for shard in rank_shards
        )
    ):
        raise TemporalAnalysisError(
            f"{expected_arm} has {records} rows; expected {expected_records}"
        )
    cells = {
        key: _cell_from_rows(
            expected_arm, key[0], key[1], rows, expected_clips=expected_clips
        )
        for key, rows in groups.items()
    }
    for nfe in evaluator.NFE_GRID:
        if (
            cells[(nfe, "donor_target")].generation_identity_sha256
            != cells[(nfe, "autonomous")].generation_identity_sha256
        ):
            raise TemporalAnalysisError("donor target did not reuse autonomous output")
    checkpoint = checkpoint_binding
    training_config = training_config_binding
    manifest = manifest_binding
    semantic_cache = summary.get("semantic_cache")
    cache_attestation = summary.get("cache_producer_attestation")
    fresh_training_source_compatibility = evaluator.training_source_compatibility(
        primary_training_config.get("source"), current_source
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            checkpoint,
            training_config,
            manifest,
            semantic_cache,
            cache_attestation,
        )
    ):
        raise TemporalAnalysisError("summary lacks immutable provenance records")
    if (
        config.get("cache_producer_attestation") != cache_attestation
        or summary.get("dataset_source") != config.get("dataset_source")
        or config.get("dataset_source")
        != vlf.file_record(evaluator.cache_bridge.__file__)
        or summary.get("semantic_cache") != config.get("semantic_cache")
        or summary.get("manifest") != config.get("manifest")
        or summary.get("checkpoint") != config.get("checkpoint")
        or summary.get("training_config") != config.get("training_config")
        or summary.get("training_source_compatibility")
        != config.get("training_source_compatibility")
        or config.get("training_source_compatibility")
        != fresh_training_source_compatibility
        or not isinstance(config.get("training_source_compatibility"), Mapping)
        or config["training_source_compatibility"].get("training_is_ancestor")
        is not True
        or config["training_source_compatibility"].get(
            "inference_critical_paths_unchanged"
        )
        is not True
        or cache_attestation.get("protected_test_accessed") is not False
        or cache_attestation.get("split") != "val"
        or cache_attestation.get("clips") != expected_clips
    ):
        raise TemporalAnalysisError("producer cache attestation changed or accessed test")
    return EvaluationData(
        arm=expected_arm,
        target_mode=target_mode,
        normalization_binding_sha256=normalization_binding,
        summary_record=vlf.file_record(resolved_summary),
        checkpoint_record=dict(checkpoint),
        training_config_record=dict(training_config),
        training_doe_common_identity_sha256=training_common_identity,
        calibration_receipt_identity_sha256=screen.sha256_json(
            calibration_receipt
        ),
        implementation_registration_identity_sha256=(
            implementation_registration_identity
        ),
        preregistration_identity_sha256=screen.sha256_json(registered_protocol),
        manifest_record=dict(manifest),
        semantic_cache_identity_sha256=screen.sha256_json(semantic_cache),
        cache_producer_attestation_identity_sha256=screen.sha256_json(
            cache_attestation
        ),
        cells=cells,
    )


def validate_common_pairing(evaluations: Mapping[str, EvaluationData]) -> tuple[str, ...]:
    if set(evaluations) != set(ARMS):
        raise TemporalAnalysisError(f"analysis requires exactly {ARMS}")
    reference = evaluations["ABS"]
    reference_clips = reference.cells[(1, "autonomous")].clip_ids
    for arm, evaluation in evaluations.items():
        if (
            evaluation.manifest_record != reference.manifest_record
            or evaluation.semantic_cache_identity_sha256
            != reference.semantic_cache_identity_sha256
            or evaluation.cache_producer_attestation_identity_sha256
            != reference.cache_producer_attestation_identity_sha256
            or evaluation.training_doe_common_identity_sha256
            != reference.training_doe_common_identity_sha256
            or evaluation.implementation_registration_identity_sha256
            != reference.implementation_registration_identity_sha256
            or evaluation.preregistration_identity_sha256
            != reference.preregistration_identity_sha256
        ):
            raise TemporalAnalysisError(f"{arm} uses a different development population")
        for key, cell in evaluation.cells.items():
            if cell.clip_ids != reference_clips:
                raise TemporalAnalysisError(f"{arm}/{key} clip identities are not paired")
            reference_cell = reference.cells[key]
            if cell.pairing_identity_sha256 != reference_cell.pairing_identity_sha256:
                raise TemporalAnalysisError(f"{arm}/{key} immutable paired inputs differ")
    return reference_clips


def _effect_pass(effect: Mapping[str, Any], threshold: float) -> bool:
    return (
        float(effect["point_estimate"]) >= threshold
        and float(effect["one_sided_lower_bound"]) >= threshold
    )


def analyze_candidate_cell(
    evaluations: Mapping[str, EvaluationData],
    *,
    arm: str,
    nfe: int,
    indices: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    if arm not in CANDIDATE_ARMS or nfe not in SELECTION_NFE:
        raise ValueError("candidate cell is outside the preregistered 12-cell screen")
    autonomous = evaluations[arm].cells[(nfe, "autonomous")]
    means = {name: float(values.mean()) for name, values in autonomous.metrics.items()}
    absolute = {
        "semantic_nmse_at_most_0.50": means["semantic_nmse"] <= MAX_NMSE,
        "semantic_token_cosine_at_least_0.70": means["semantic_token_cosine"]
        >= MIN_COSINE,
        "temporal_difference_nmse_at_most_0.50": means[
            "temporal_difference_nmse"
        ]
        <= MAX_NMSE,
    }
    comparisons: dict[str, Any] = {}
    comparison_cells = {
        "ABS_same_nfe": evaluations["ABS"].cells[(nfe, "autonomous")],
        "donor_target": evaluations[arm].cells[(nfe, "donor_target")],
        "context_shuffled": evaluations[arm].cells[(nfe, "context_shuffled")],
    }
    comparison_passes: list[bool] = []
    for label, control in comparison_cells.items():
        ordinary = paired_effect(
            autonomous.metrics["semantic_nmse"],
            control.metrics["semantic_nmse"],
            indices,
            kind="relative_lower",
            confidence=confidence,
        )
        temporal_effect = paired_effect(
            autonomous.metrics["temporal_difference_nmse"],
            control.metrics["temporal_difference_nmse"],
            indices,
            kind="relative_lower",
            confidence=confidence,
        )
        comparison = {
            "ordinary_nmse_relative_improvement": ordinary,
            "temporal_nmse_relative_improvement": temporal_effect,
            "ordinary_passed": _effect_pass(
                ordinary, MIN_RELATIVE_IMPROVEMENT
            ),
            "temporal_passed": _effect_pass(
                temporal_effect, MIN_RELATIVE_IMPROVEMENT
            ),
        }
        if label != "ABS_same_nfe":
            cosine = paired_effect(
                autonomous.metrics["semantic_token_cosine"],
                control.metrics["semantic_token_cosine"],
                indices,
                kind="difference_higher",
                confidence=confidence,
            )
            comparison["semantic_cosine_advantage"] = cosine
            comparison["cosine_passed"] = _effect_pass(cosine, 0.0) and (
                float(cosine["point_estimate"]) > 0.0
                and float(cosine["one_sided_lower_bound"]) > 0.0
            )
        else:
            comparison["semantic_cosine_advantage"] = None
            comparison["cosine_passed"] = True
        comparison["passed"] = bool(
            comparison["ordinary_passed"]
            and comparison["temporal_passed"]
            and comparison["cosine_passed"]
        )
        comparison_passes.append(bool(comparison["passed"]))
        comparisons[label] = comparison

    factual_action_attribution: dict[str, Any] = {}
    for control_name in evaluator.ACTION_CONTROLS:
        shuffled = evaluations[arm].cells[(nfe, control_name)]
        factual_action_attribution[control_name] = {
            "promotion_criterion": False,
            "label": "factual_action_attribution_not_counterfactual_causality",
            "ordinary_nmse_relative_improvement": paired_effect(
                autonomous.metrics["semantic_nmse"],
                shuffled.metrics["semantic_nmse"],
                indices,
                kind="relative_lower",
                confidence=confidence,
            ),
            "temporal_nmse_relative_improvement": paired_effect(
                autonomous.metrics["temporal_difference_nmse"],
                shuffled.metrics["temporal_difference_nmse"],
                indices,
                kind="relative_lower",
                confidence=confidence,
            ),
            "semantic_cosine_advantage": paired_effect(
                autonomous.metrics["semantic_token_cosine"],
                shuffled.metrics["semantic_token_cosine"],
                indices,
                kind="difference_higher",
                confidence=confidence,
            ),
        }
    passed = all(absolute.values()) and all(comparison_passes)
    return {
        "arm": arm,
        "nfe": nfe,
        "selection_eligible": True,
        "autonomous_means": means,
        "absolute_thresholds": absolute,
        "comparisons": comparisons,
        "factual_action_attribution": factual_action_attribution,
        "composite_gate_passed": bool(passed),
    }


def select_cell(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [cell for cell in cells if cell.get("composite_gate_passed") is True]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda cell: (
            int(cell["nfe"]),
            float(cell["autonomous_means"]["temporal_difference_nmse"]),
            float(cell["autonomous_means"]["semantic_nmse"]),
            ARM_ORDER[str(cell["arm"])],
        ),
    )
    return {
        "arm": str(selected["arm"]),
        "nfe": int(selected["nfe"]),
        "autonomous_semantic_nmse": float(
            selected["autonomous_means"]["semantic_nmse"]
        ),
        "autonomous_semantic_token_cosine": float(
            selected["autonomous_means"]["semantic_token_cosine"]
        ),
        "autonomous_temporal_difference_nmse": float(
            selected["autonomous_means"]["temporal_difference_nmse"]
        ),
    }


def _descriptive_summaries(
    evaluations: Mapping[str, EvaluationData]
) -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "nfe": nfe,
            "control": control,
            "clips": len(cell.clip_ids),
            **{name: float(value.mean()) for name, value in cell.metrics.items()},
        }
        for arm in ARMS
        for (nfe, control), cell in sorted(evaluations[arm].cells.items())
    ]


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {**unsigned, "identity_sha256": screen.sha256_json(unsigned)}


def build_analysis(
    evaluations: Mapping[str, EvaluationData],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    confidence: float = CELLWISE_CONFIDENCE,
    expected_clips: int = EXPECTED_CLIPS,
) -> dict[str, Any]:
    clip_ids = validate_common_pairing(evaluations)
    if len(clip_ids) != expected_clips:
        raise TemporalAnalysisError("paired development population size changed")
    indices = common_bootstrap_indices(
        expected_clips, samples=bootstrap_samples, seed=bootstrap_seed
    )
    cells = [
        analyze_candidate_cell(
            evaluations,
            arm=arm,
            nfe=nfe,
            indices=indices,
            confidence=confidence,
        )
        for arm in CANDIDATE_ARMS
        for nfe in SELECTION_NFE
    ]
    selected = select_cell(cells)
    protocol_frozen = (
        bootstrap_samples == BOOTSTRAP_SAMPLES
        and bootstrap_seed == BOOTSTRAP_SEED
        and confidence == CELLWISE_CONFIDENCE
        and expected_clips == EXPECTED_CLIPS
        and len(
            {
                evaluation.training_doe_common_identity_sha256
                for evaluation in evaluations.values()
            }
        )
        == 1
        and len(
            {
                evaluation.implementation_registration_identity_sha256
                for evaluation in evaluations.values()
            }
        )
        == 1
        and len(
            {
                evaluation.preregistration_identity_sha256
                for evaluation in evaluations.values()
            }
        )
        == 1
    )
    return identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "one_candidate_selected" if selected else "no_candidate_passed",
            "source": screen._source_record(),  # noqa: SLF001
            "split": "val",
            "development_selection_split": True,
            "protected_test_accessed": False,
            "input_evaluations": {
                arm: {
                    "summary": dict(evaluations[arm].summary_record),
                    "checkpoint": dict(evaluations[arm].checkpoint_record),
                    "training_config": dict(evaluations[arm].training_config_record),
                    "training_doe_common_identity_sha256": evaluations[
                        arm
                    ].training_doe_common_identity_sha256,
                    "calibration_receipt_identity_sha256": evaluations[
                        arm
                    ].calibration_receipt_identity_sha256,
                    "implementation_registration_identity_sha256": evaluations[
                        arm
                    ].implementation_registration_identity_sha256,
                    "preregistration_identity_sha256": evaluations[
                        arm
                    ].preregistration_identity_sha256,
                    "target_mode": evaluations[arm].target_mode,
                    "normalization_binding_sha256": evaluations[
                        arm
                    ].normalization_binding_sha256,
                }
                for arm in ARMS
            },
            "paired_clips": expected_clips,
            "ordered_clip_ids_sha256": screen.sha256_json(list(clip_ids)),
            "bootstrap": {
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "unit": "paired immutable development clip",
                "common_indices_across_all_metrics_cells_controls": True,
                "indices_sha256": _array_sha256(indices),
                "one_sided_confidence_per_candidate_cell": confidence,
                "alpha_per_candidate_cell": 1.0 - confidence,
                "bonferroni_candidate_cells": 12,
                "interpretation": (
                    "12-cell Bonferroni screen of composite intersection-union gates"
                ),
                "ratio_effect": "ratio of resampled means, never mean of clip ratios",
            },
            "thresholds": {
                "autonomous_semantic_nmse_max": MAX_NMSE,
                "autonomous_semantic_token_cosine_min": MIN_COSINE,
                "autonomous_temporal_difference_nmse_max": MAX_NMSE,
                "relative_nmse_point_and_lower_bound_min": MIN_RELATIVE_IMPROVEMENT,
                "donor_and_context_cosine_point_and_lower_bound": "strictly_positive",
            },
            "candidate_cells": cells,
            "selected_cell": selected,
            "selection_count": int(selected is not None),
            "deterministic_tie_break": [
                "smallest_nfe",
                "lowest_autonomous_temporal_difference_nmse",
                "lowest_autonomous_semantic_nmse",
                "arm_order_ABS-T_DELTA_DELTA-T_DELTA-R",
            ],
            "descriptive_summaries": _descriptive_summaries(evaluations),
            "packed_metrics_promotion_eligible": False,
            "action_permutations_promotion_eligible": False,
            "protected_test_cache_opened": False,
            "protocol_frozen": protocol_frozen,
        }
    )


def _selection_record(
    analysis: Mapping[str, Any], analysis_record: Mapping[str, Any]
) -> dict[str, Any]:
    selected = analysis.get("selected_cell")
    return identity_payload(
        {
            "schema": SELECTION_SCHEMA,
            "status": "frozen_candidate" if selected else "frozen_no_selection",
            "development_analysis": dict(analysis_record),
            "development_analysis_identity_sha256": analysis["identity_sha256"],
            "selected_cell": selected,
            "selection_count": int(selected is not None),
            "selection_split": "val",
            "selection_used_protected_test": False,
            "protected_test_accessed": False,
            "lockbox_may_open": selected is not None and analysis.get("protocol_frozen") is True,
            "lockbox_policy": (
                "evaluate only ABS and the exact selected arm/NFE once; no retuning"
            ),
            "tie_break": analysis["deterministic_tie_break"],
            "input_evaluations": analysis["input_evaluations"],
        }
    )


def analysis_command(args: argparse.Namespace) -> int:
    summaries = {
        "ABS": args.abs_summary,
        "ABS-T": args.abs_t_summary,
        "DELTA": args.delta_summary,
        "DELTA-T": args.delta_t_summary,
        "DELTA-R": args.delta_r_summary,
    }
    evaluations = {
        arm: load_evaluation(path, expected_arm=arm) for arm, path in summaries.items()
    }
    analysis = build_analysis(evaluations)
    if analysis.get("protocol_frozen") is not True:
        raise TemporalAnalysisError("production analysis protocol is not frozen")
    output_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    analysis_path = output_dir / "development_analysis.json"
    vlf.atomic_write_json(analysis_path, analysis, exclusive=True)
    selection = _selection_record(analysis, vlf.file_record(analysis_path))
    vlf.atomic_write_json(
        output_dir / "frozen_selection.json", selection, exclusive=True
    )
    print(screen.canonical_json(selection))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--abs-summary", required=True)
    parser.add_argument("--abs-t-summary", required=True)
    parser.add_argument("--delta-summary", required=True)
    parser.add_argument("--delta-t-summary", required=True)
    parser.add_argument("--delta-r-summary", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return analysis_command(args)
    except (
        TemporalAnalysisError,
        evaluator.TemporalEvaluationError,
        screen.ScreenError,
        vlf.PocError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Temporal-target development analysis error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
