#!/usr/bin/env python3
"""Validate and summarize the stage-faithful cascade diagnostic.

The three inputs are read-only evidence:

* a single-arm ``analyze_dual_nfe_artifacts.py`` result for the new evaluation;
* the validated cross-screen reference containing per-rank video-only metrics;
* the bitwise new-versus-legacy artifact audit.

The output is deterministic for identical input bytes: it intentionally has no
wall-clock timestamp.  LACWM uses ``sigma=1`` for noise and ``sigma=0`` for
clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SIGMA_CONVENTION = "1=noise,0=clean"
NFE_STEPS = (2, 4, 8)
WORLD_SIZE = 8
EXPECTED_DATASET = "MultiDatasetABC_0"
EXPECTED_PAIRS = tuple((EXPECTED_DATASET, rank) for rank in range(WORLD_SIZE))
EXPECTED_SOURCES = (
    "autonomous",
    "autonomous_shuffled",
    "autonomous_legacy",
    "off",
)
SOURCE_COMPARISONS = (
    "autonomous_minus_autonomous_shuffled",
    "autonomous_minus_autonomous_legacy",
    "autonomous_minus_off",
)
METRIC_DIRECTIONS = {
    "video_future_nmse": "lower",
    "tf_future_nmse": "lower",
    "decoded_mse_unit_range": "lower",
    "decoded_psnr_db": "higher",
    "decoded_temporal_difference_mse_unit_range": "lower",
}
VIDEO_ONLY_METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_psnr_db",
    "decoded_temporal_difference_mse_unit_range",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_726
CONFIDENCE = 0.95
PRIMARY_TEMPORAL_IMPROVEMENT = -0.03
PRIMARY_VIDEO_NO_REGRESSION = 0.0


class StageEvaluationSummaryError(RuntimeError):
    """Raised when any evidence or comparison contract is not satisfied."""


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_json_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StageEvaluationSummaryError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or path.suffix.lower() != ".json"
    ):
        raise StageEvaluationSummaryError(
            f"{label} must be a non-empty, non-symlink JSON file: {path}"
        )
    return path.resolve(strict=True)


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StageEvaluationSummaryError(
                    f"duplicate JSON key {key!r} in {label}: {path}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise StageEvaluationSummaryError(
            f"non-finite JSON number {value!r} in {label}: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except StageEvaluationSummaryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageEvaluationSummaryError(
            f"could not parse {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise StageEvaluationSummaryError(
            f"{label} must contain one JSON object: {path}"
        )
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise StageEvaluationSummaryError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise StageEvaluationSummaryError(f"{label} must be an array")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise StageEvaluationSummaryError(
            f"{label} must be one finite JSON number"
        )
    return float(value)


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageEvaluationSummaryError(
            f"{label} must be one lowercase hexadecimal SHA-256"
        )
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageEvaluationSummaryError(message)


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _normal_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not value.startswith("/"):
        raise StageEvaluationSummaryError(
            f"{label} must be a non-empty absolute path string"
        )
    return os.path.normpath(value)


def _pairs(
    raw: Any, label: str, *, rank_key: str = "global_rank"
) -> tuple[tuple[str, int], ...]:
    records = _sequence(raw, label)
    pairs: list[tuple[str, int]] = []
    for index, record in enumerate(records):
        item = _mapping(record, f"{label}[{index}]")
        dataset = item.get("dataset")
        rank = item.get(rank_key)
        if (
            not isinstance(dataset, str)
            or not dataset
            or isinstance(rank, bool)
            or not isinstance(rank, int)
        ):
            raise StageEvaluationSummaryError(
                f"{label}[{index}] has an invalid dataset/rank identity"
            )
        pairs.append((dataset, rank))
    return tuple(pairs)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _summary(
    values: Sequence[float],
    *,
    label: str,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size != WORLD_SIZE
        or not np.isfinite(array).all()
    ):
        raise StageEvaluationSummaryError(f"invalid paired values for {label}")
    rng = np.random.default_rng(_derived_seed(BOOTSTRAP_SEED, label))
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_SAMPLES, array.size),
        endpoint=False,
    )
    bootstrap_means = array[indices].mean(axis=1)
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "bootstrap_ci": {
            "confidence": CONFIDENCE,
            "low": float(low),
            "high": float(high),
        },
    }


def _relative_effect(
    left_values: Sequence[float],
    reference_values: Sequence[float],
    *,
    direction: str,
    label: str,
) -> dict[str, Any]:
    left = np.asarray(left_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    if (
        left.ndim != 1
        or reference.ndim != 1
        or left.shape != (WORLD_SIZE,)
        or reference.shape != (WORLD_SIZE,)
        or not np.isfinite(left).all()
        or not np.isfinite(reference).all()
        or direction not in {"lower", "higher"}
    ):
        raise StageEvaluationSummaryError(
            f"invalid paired values or direction for {label}"
        )
    reference_mean = float(reference.mean())
    if reference_mean <= 0:
        raise StageEvaluationSummaryError(
            f"reference mean must be positive for {label}"
        )
    left_mean = float(left.mean())
    effect = (left_mean - reference_mean) / reference_mean
    rng = np.random.default_rng(_derived_seed(BOOTSTRAP_SEED, label))
    indices = rng.integers(
        0,
        WORLD_SIZE,
        size=(BOOTSTRAP_SAMPLES, WORLD_SIZE),
        endpoint=False,
    )
    left_means = left[indices].mean(axis=1)
    reference_means = reference[indices].mean(axis=1)
    if np.any(reference_means <= 0):
        raise StageEvaluationSummaryError(
            f"bootstrap reference mean is non-positive for {label}"
        )
    effects = (left_means - reference_means) / reference_means
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(effects, [tail, 1.0 - tail])
    ci_favors_left = high < 0 if direction == "lower" else low > 0
    return {
        "n": WORLD_SIZE,
        "definition": (
            "(mean(left) - mean(reference)) / mean(reference), with paired "
            "bootstrap resampling"
        ),
        "left_mean": left_mean,
        "reference_mean": reference_mean,
        "relative_effect": float(effect),
        "relative_effect_percent": float(100.0 * effect),
        "bootstrap_ci": {
            "confidence": CONFIDENCE,
            "low": float(low),
            "high": float(high),
            "low_percent": float(100.0 * low),
            "high_percent": float(100.0 * high),
        },
        "favorable_when": (
            "relative_effect < 0"
            if direction == "lower"
            else "relative_effect > 0"
        ),
        "ci_favors_left": bool(ci_favors_left),
    }


def _validate_summary_record(
    observed_raw: Any,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    observed = _mapping(observed_raw, label)
    _require(observed.get("n") == WORLD_SIZE, f"{label}.n must be 8")
    for key in ("mean", "sample_std"):
        actual = _finite_number(observed.get(key), f"{label}.{key}")
        wanted = _finite_number(expected.get(key), f"expected {label}.{key}")
        _require(
            _same_number(actual, wanted),
            f"{label}.{key} does not match its eight paired values",
        )
    actual_ci = _mapping(observed.get("bootstrap_ci"), f"{label}.bootstrap_ci")
    expected_ci = _mapping(
        expected.get("bootstrap_ci"), f"expected {label}.bootstrap_ci"
    )
    _require(
        _same_number(
            _finite_number(actual_ci.get("confidence"), f"{label}.confidence"),
            CONFIDENCE,
        ),
        f"{label} confidence must be {CONFIDENCE}",
    )
    for key in ("low", "high"):
        actual = _finite_number(actual_ci.get(key), f"{label}.bootstrap_ci.{key}")
        wanted = _finite_number(
            expected_ci.get(key), f"expected {label}.bootstrap_ci.{key}"
        )
        _require(
            _same_number(actual, wanted),
            f"{label}.bootstrap_ci.{key} does not match recomputation",
        )


def _validate_relative_record(
    observed_raw: Any,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    observed = _mapping(observed_raw, label)
    _require(observed.get("defined") is True, f"{label} must be defined")
    _require(observed.get("n") == WORLD_SIZE, f"{label}.n must be 8")
    for key in (
        "left_mean",
        "reference_mean",
        "relative_effect",
        "relative_effect_percent",
    ):
        actual = _finite_number(observed.get(key), f"{label}.{key}")
        wanted = _finite_number(expected.get(key), f"expected {label}.{key}")
        _require(
            _same_number(actual, wanted),
            f"{label}.{key} does not match paired recomputation",
        )
    actual_ci = _mapping(observed.get("bootstrap_ci"), f"{label}.bootstrap_ci")
    expected_ci = _mapping(
        expected.get("bootstrap_ci"), f"expected {label}.bootstrap_ci"
    )
    for key in ("confidence", "low", "high", "low_percent", "high_percent"):
        actual = _finite_number(actual_ci.get(key), f"{label}.bootstrap_ci.{key}")
        wanted = _finite_number(
            expected_ci.get(key), f"expected {label}.bootstrap_ci.{key}"
        )
        _require(
            _same_number(actual, wanted),
            f"{label}.bootstrap_ci.{key} does not match paired recomputation",
        )
    _require(
        observed.get("ci_favors_left") is expected["ci_favors_left"],
        f"{label}.ci_favors_left is inconsistent with its confidence interval",
    )
    return {
        key: expected[key]
        for key in (
            "n",
            "definition",
            "left_mean",
            "reference_mean",
            "relative_effect",
            "relative_effect_percent",
            "bootstrap_ci",
            "favorable_when",
            "ci_favors_left",
        )
    }


def _analysis_evidence(
    analysis: Mapping[str, Any],
) -> tuple[
    str,
    tuple[tuple[str, int], ...],
    dict[str, dict[str, dict[str, list[float]]]],
    dict[str, Any],
]:
    _require(
        analysis.get("schema_version") == 1
        and analysis.get("kind")
        == "dual_video_diffusion_matched_nfe_analysis",
        "analysis schema/kind is unsupported",
    )
    _require(
        analysis.get("sigma_convention") == SIGMA_CONVENTION,
        "analysis sigma convention mismatch",
    )
    _require(
        analysis.get("nfe_steps") == list(NFE_STEPS),
        f"analysis NFE steps must be {list(NFE_STEPS)}",
    )
    bootstrap = _mapping(analysis.get("bootstrap"), "analysis.bootstrap")
    _require(
        bootstrap.get("samples") == BOOTSTRAP_SAMPLES
        and bootstrap.get("seed") == BOOTSTRAP_SEED
        and _same_number(
            _finite_number(
                bootstrap.get("confidence"), "analysis.bootstrap.confidence"
            ),
            CONFIDENCE,
        ),
        "analysis bootstrap contract mismatch",
    )

    provenance = _mapping(analysis.get("provenance"), "analysis.provenance")
    _require(
        provenance.get("paired_unit_count") == WORLD_SIZE,
        "analysis must contain exactly eight paired units",
    )
    pairs = _pairs(provenance.get("paired_units"), "analysis paired_units")
    _require(
        pairs == EXPECTED_PAIRS,
        f"analysis paired units/order must be exactly {EXPECTED_PAIRS}",
    )
    arms = _mapping(provenance.get("arms"), "analysis.provenance.arms")
    _require(len(arms) == 1, "analysis must contain exactly one arm")
    arm = next(iter(arms))
    _require(
        analysis.get("baseline_arm") == arm,
        "single analysis arm must also be its baseline arm",
    )
    arm_provenance = _mapping(arms.get(arm), f"analysis arm {arm}")
    _require(
        arm_provenance.get("evaluation_condition_sources")
        == list(EXPECTED_SOURCES),
        f"analysis sources must be exactly {list(EXPECTED_SOURCES)}",
    )
    _require(
        arm_provenance.get("artifact_count") == WORLD_SIZE,
        "analysis arm must contain exactly eight artifacts",
    )
    artifacts = _sequence(
        arm_provenance.get("artifacts"), "analysis arm artifacts"
    )
    _require(
        _pairs(artifacts, "analysis arm artifacts") == EXPECTED_PAIRS,
        "analysis artifact identity/order mismatch",
    )

    per_units = _sequence(
        analysis.get("per_paired_unit"), "analysis.per_paired_unit"
    )
    _require(
        _pairs(per_units, "analysis.per_paired_unit") == EXPECTED_PAIRS,
        "analysis per-unit identity/order mismatch",
    )
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        source: {
            str(nfe): {metric: [] for metric in METRIC_DIRECTIONS}
            for nfe in NFE_STEPS
        }
        for source in EXPECTED_SOURCES
    }
    for index, record in enumerate(per_units):
        unit = _mapping(record, f"analysis.per_paired_unit[{index}]")
        unit_arms = _mapping(
            unit.get("arms"), f"analysis.per_paired_unit[{index}].arms"
        )
        _require(
            list(unit_arms) == [arm],
            f"analysis pair {index} must contain only arm {arm!r}",
        )
        source_records = _mapping(
            _mapping(unit_arms[arm], f"analysis pair {index} arm").get(
                "condition_source_metrics"
            ),
            f"analysis pair {index} source metrics",
        )
        _require(
            set(source_records) == set(EXPECTED_SOURCES),
            f"analysis pair {index} source inventory mismatch",
        )
        for source in EXPECTED_SOURCES:
            source_record = _mapping(
                source_records[source],
                f"analysis pair {index} source {source}",
            )
            _require(
                source_record.get("oracle_leakage") is False,
                f"deployable source {source} must not be marked oracle leakage",
            )
            source_metrics = _mapping(
                source_record.get("metrics"),
                f"analysis pair {index} source {source} metrics",
            )
            _require(
                set(source_metrics) == {str(nfe) for nfe in NFE_STEPS},
                f"analysis pair {index} source {source} NFE inventory mismatch",
            )
            for nfe in NFE_STEPS:
                metrics = _mapping(
                    source_metrics[str(nfe)],
                    f"analysis pair {index} source {source} NFE {nfe}",
                )
                for metric in METRIC_DIRECTIONS:
                    values[source][str(nfe)][metric].append(
                        _finite_number(
                            metrics.get(metric),
                            (
                                f"analysis pair {index} source {source} "
                                f"NFE {nfe} {metric}"
                            ),
                        )
                    )
    return arm, pairs, values, {
        "arm_provenance": arm_provenance,
        "artifacts": artifacts,
    }


def _validate_audit(
    audit: Mapping[str, Any],
    *,
    analysis_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        audit.get("schema_version") == 1
        and audit.get("kind")
        == "stage_faithful_cascade_bitwise_artifact_audit",
        "audit schema/kind is unsupported",
    )
    _require(
        audit.get("sigma_convention") == SIGMA_CONVENTION
        and audit.get("read_only_inputs") is True
        and audit.get("overall_pass") is True,
        "bitwise artifact audit did not pass",
    )
    contracts = _mapping(audit.get("contracts"), "audit.contracts")
    _require(contracts.get("pass") is True, "audit contracts did not pass")
    required_contracts = (
        "world_size",
        "paired_ranks",
        "artifact_iteration",
        "evaluation_noise_seed_identity",
        "forbidden_training_outputs",
        "sidecar_hashes_and_sigma_convention",
        "new_source_codes",
        "legacy_source_codes",
        "nfe_steps",
        "new_stage_faithful_flag",
    )
    for name in required_contracts:
        record = _mapping(contracts.get(name), f"audit.contracts.{name}")
        _require(record.get("pass") is True, f"audit contract {name} failed")
    _require(
        contracts["world_size"].get("observed_new") == WORLD_SIZE
        and contracts["world_size"].get("observed_legacy") == WORLD_SIZE
        and contracts["world_size"].get("expected") == WORLD_SIZE,
        "audit world-size contract is not exactly eight",
    )
    expected_ranks = list(range(WORLD_SIZE))
    _require(
        contracts["paired_ranks"].get("new") == expected_ranks
        and contracts["paired_ranks"].get("legacy") == expected_ranks
        and contracts["paired_ranks"].get("expected") == expected_ranks,
        "audit paired-rank identity/order mismatch",
    )
    _require(
        contracts["new_source_codes"].get("expected") == [0, 4, 5, 1]
        and contracts["legacy_source_codes"].get("expected") == [0, 1, 2, 3]
        and contracts["nfe_steps"].get("expected") == list(NFE_STEPS)
        and contracts["new_stage_faithful_flag"].get("expected") == 1,
        "audit source/NFE/stage contract mismatch",
    )
    _require(
        contracts["forbidden_training_outputs"].get("observed") == [],
        "audit found forbidden training outputs",
    )

    rank_audits = _sequence(audit.get("rank_audits"), "audit.rank_audits")
    _require(
        [record.get("rank") for record in rank_audits] == expected_ranks,
        "audit rank records must be exactly ranks 0..7 in order",
    )
    for index, raw in enumerate(rank_audits):
        record = _mapping(raw, f"audit.rank_audits[{index}]")
        _require(record.get("pass") is True, f"audit rank {index} failed")
        for name in (
            "contracts",
            "new_legacy_input_identity",
            "legacy_reproduction",
            "stage_tf_equivalence",
        ):
            _require(
                _mapping(record.get(name), f"audit rank {index} {name}").get(
                    "pass"
                )
                is True,
                f"audit rank {index} {name} failed",
            )

    inputs = _mapping(audit.get("inputs"), "audit.inputs")
    new_input = _mapping(inputs.get("new"), "audit.inputs.new")
    new_ranks = _sequence(new_input.get("ranks"), "audit.inputs.new.ranks")
    _require(
        _pairs(new_ranks, "audit.inputs.new.ranks", rank_key="rank")
        == EXPECTED_PAIRS,
        "audit new artifact identity/order mismatch",
    )
    analysis_arm = _mapping(
        analysis_provenance.get("arm_provenance"),
        "analysis arm provenance",
    )
    analysis_root = _normal_path(
        analysis_arm.get("root"), "analysis arm root"
    )
    accepted_new_roots = {
        _normal_path(new_input.get("artifact_scope"), "audit new scope"),
        _normal_path(new_input.get("root"), "audit new root"),
    }
    _require(
        analysis_root in accepted_new_roots,
        "analysis arm root matches neither audited new root nor artifact scope",
    )
    analysis_artifacts = _sequence(
        analysis_provenance.get("artifacts"), "analysis artifacts"
    )
    for index, (analysis_raw, audit_raw) in enumerate(
        zip(analysis_artifacts, new_ranks)
    ):
        analysis_record = _mapping(
            analysis_raw, f"analysis artifact {index}"
        )
        audit_record = _mapping(audit_raw, f"audit new rank {index}")
        audit_artifact = _mapping(
            audit_record.get("artifact"), f"audit new rank {index} artifact"
        )
        analysis_sha256 = _sha256_value(
            analysis_record.get("sha256"),
            f"analysis artifact {index} SHA-256",
        )
        audit_sha256 = _sha256_value(
            audit_artifact.get("sha256"),
            f"audit artifact {index} SHA-256",
        )
        _require(
            analysis_sha256 == audit_sha256
            and _normal_path(
                analysis_record.get("path"), f"analysis artifact {index} path"
            )
            == _normal_path(
                audit_artifact.get("path"), f"audit artifact {index} path"
            ),
            f"analysis/audit artifact provenance mismatch at rank {index}",
        )
    identity_sha256 = _sha256_value(
        audit.get("identity_sha256"), "audit identity SHA-256"
    )
    return {
        "identity_sha256": identity_sha256,
        "new_artifact_set_sha256": new_input.get("artifact_set_sha256"),
        "legacy_input": _mapping(inputs.get("legacy"), "audit.inputs.legacy"),
    }


def _validate_baseline(
    baseline: Mapping[str, Any],
    *,
    audit_evidence: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, Any],
]:
    _require(
        baseline.get("schema_version") == 1
        and baseline.get("kind")
        == "strict_cascade_cross_screen_efficiency_audit",
        "baseline-reference schema/kind is unsupported",
    )
    baseline_bootstrap = _mapping(
        baseline.get("bootstrap"), "baseline.bootstrap"
    )
    _require(
        baseline_bootstrap.get("samples") == BOOTSTRAP_SAMPLES
        and baseline_bootstrap.get("seed") == BOOTSTRAP_SEED
        and _same_number(
            _finite_number(
                baseline_bootstrap.get("confidence"),
                "baseline.bootstrap.confidence",
            ),
            CONFIDENCE,
        ),
        "baseline-reference bootstrap contract mismatch",
    )
    scope = _mapping(baseline.get("scope"), "baseline.scope")
    _require(
        scope.get("common_total_nfe_steps") == list(NFE_STEPS)
        and scope.get("nfe_is_total_model_calls") is True,
        "baseline-reference total-NFE contract mismatch",
    )
    arms = _mapping(baseline.get("arms"), "baseline.arms")
    video_only_arm = _mapping(
        arms.get("video_only_s000"), "baseline arms.video_only_s000"
    )
    video_only_terminal = _mapping(
        video_only_arm.get("terminal"), "baseline video-only terminal"
    )
    video_only_artifacts = _sequence(
        video_only_terminal.get("artifacts"),
        "baseline video-only terminal artifacts",
    )
    _require(
        video_only_terminal.get("artifact_count") == WORLD_SIZE
        and video_only_terminal.get("iteration") == 199
        and _pairs(
            video_only_artifacts, "baseline video-only terminal artifacts"
        )
        == EXPECTED_PAIRS,
        "baseline video-only terminal artifact contract mismatch",
    )
    video_only_outcome = _mapping(
        video_only_terminal.get("outcome"),
        "baseline video-only terminal outcome",
    )
    video_only_completion = _mapping(
        video_only_terminal.get("training_completion"),
        "baseline video-only terminal training completion",
    )
    _require(
        video_only_outcome.get("completed") is True
        and video_only_outcome.get("exit_status") == 0
        and video_only_completion.get("completed_updates") == 200,
        "baseline video-only run was not a clean 200-update completion",
    )
    legacy_arm = _mapping(
        arms.get("cascade_matched_s010"),
        "baseline arms.cascade_matched_s010",
    )
    legacy_input = _mapping(
        audit_evidence.get("legacy_input"), "audit legacy input"
    )
    _require(
        _normal_path(
            legacy_input.get("artifact_scope"), "audit legacy artifact scope"
        )
        == _normal_path(legacy_arm.get("root"), "baseline legacy arm root"),
        "audit legacy scope does not match baseline cascade reference",
    )
    terminal = _mapping(
        legacy_arm.get("terminal"), "baseline legacy arm terminal"
    )
    legacy_artifacts = _sequence(
        terminal.get("artifacts"), "baseline legacy artifacts"
    )
    audit_legacy_ranks = _sequence(
        legacy_input.get("ranks"), "audit legacy ranks"
    )
    _require(
        _pairs(legacy_artifacts, "baseline legacy artifacts") == EXPECTED_PAIRS
        and _pairs(
            audit_legacy_ranks, "audit legacy ranks", rank_key="rank"
        )
        == EXPECTED_PAIRS,
        "baseline/audit legacy artifact identity/order mismatch",
    )
    for index, (baseline_raw, audit_raw) in enumerate(
        zip(legacy_artifacts, audit_legacy_ranks)
    ):
        baseline_record = _mapping(
            baseline_raw, f"baseline legacy artifact {index}"
        )
        audit_artifact = _mapping(
            _mapping(audit_raw, f"audit legacy rank {index}").get("artifact"),
            f"audit legacy rank {index} artifact",
        )
        baseline_sha256 = _sha256_value(
            baseline_record.get("sha256"),
            f"baseline legacy artifact {index} SHA-256",
        )
        audit_sha256 = _sha256_value(
            audit_artifact.get("sha256"),
            f"audit legacy artifact {index} SHA-256",
        )
        _require(
            baseline_sha256 == audit_sha256
            and _normal_path(
                baseline_record.get("path"),
                f"baseline legacy artifact {index} path",
            )
            == _normal_path(
                audit_artifact.get("path"),
                f"audit legacy artifact {index} path",
            ),
            f"baseline/audit legacy artifact mismatch at rank {index}",
        )

    comparisons = _mapping(
        baseline.get("comparisons"), "baseline.comparisons"
    )
    comparison = _mapping(
        comparisons.get("video_only_s000"),
        "baseline comparison video_only_s000",
    )
    _require(
        comparison.get("computed") is True
        and comparison.get("candidate_condition_source") == "autonomous"
        and comparison.get("reference_condition_source") == "autonomous",
        "baseline video-only comparison contract mismatch",
    )
    identity = _mapping(
        comparison.get("identity_audit"), "baseline video-only identity audit"
    )
    _require(
        identity.get("paired_unit_count") == WORLD_SIZE
        and identity.get("common_total_nfe_steps") == list(NFE_STEPS)
        and identity.get("video_relevant_identity_exact_for_every_pair") is True
        and identity.get("tf_identity_exact_for_every_pair") is True,
        "baseline video-only identity audit did not pass",
    )
    _require(
        _pairs(identity.get("paired_units"), "baseline identity paired_units")
        == EXPECTED_PAIRS,
        "baseline identity paired-unit order mismatch",
    )
    identity_per_pair = _sequence(
        identity.get("per_pair"), "baseline identity per_pair"
    )
    _require(
        _pairs(identity_per_pair, "baseline identity per_pair")
        == EXPECTED_PAIRS,
        "baseline per-pair identity order mismatch",
    )
    for index, raw in enumerate(identity_per_pair):
        item = _mapping(raw, f"baseline identity pair {index}")
        _require(
            item.get("candidate_evaluation_nfe_steps") == list(NFE_STEPS)
            and item.get("reference_evaluation_nfe_steps") == [1, 2, 4, 8],
            f"baseline identity pair {index} NFE provenance mismatch",
        )
        for field in (
            "video_identity_fields_equal",
            "tf_identity_fields_equal",
        ):
            flags = _mapping(
                item.get(field), f"baseline identity pair {index} {field}"
            )
            _require(
                bool(flags) and all(value is True for value in flags.values()),
                f"baseline identity pair {index} {field} did not pass",
            )

    per_units = _sequence(
        comparison.get("per_paired_unit"),
        "baseline video-only per_paired_unit",
    )
    _require(
        _pairs(per_units, "baseline video-only per_paired_unit")
        == EXPECTED_PAIRS,
        "baseline video-only per-unit identity/order mismatch",
    )
    values: dict[str, dict[str, list[float]]] = {
        str(nfe): {metric: [] for metric in VIDEO_ONLY_METRICS}
        for nfe in NFE_STEPS
    }
    for index, raw in enumerate(per_units):
        item = _mapping(raw, f"baseline per-unit {index}")
        nfe_records = _mapping(
            item.get("nfe"), f"baseline per-unit {index} nfe"
        )
        _require(
            set(nfe_records) == {str(nfe) for nfe in NFE_STEPS},
            f"baseline per-unit {index} NFE inventory mismatch",
        )
        for nfe in NFE_STEPS:
            metrics = _mapping(
                nfe_records[str(nfe)],
                f"baseline per-unit {index} NFE {nfe}",
            )
            for metric in VIDEO_ONLY_METRICS:
                record = _mapping(
                    metrics.get(metric),
                    f"baseline per-unit {index} NFE {nfe} {metric}",
                )
                candidate = _finite_number(
                    record.get("candidate"), "baseline candidate"
                )
                reference = _finite_number(
                    record.get("reference"), "baseline reference"
                )
                delta = _finite_number(record.get("delta"), "baseline delta")
                _require(
                    _same_number(delta, candidate - reference),
                    (
                        f"baseline per-unit {index} NFE {nfe} {metric} "
                        "delta is inconsistent"
                    ),
                )
                values[str(nfe)][metric].append(reference)

    aggregates = _mapping(
        comparison.get("aggregate"), "baseline video-only aggregate"
    )
    for nfe in NFE_STEPS:
        nfe_aggregate = _mapping(
            aggregates.get(str(nfe)), f"baseline aggregate NFE {nfe}"
        )
        for metric in VIDEO_ONLY_METRICS:
            reference_summary = _mapping(
                _mapping(
                    nfe_aggregate.get(metric),
                    f"baseline aggregate NFE {nfe} {metric}",
                ).get("reference"),
                f"baseline aggregate NFE {nfe} {metric} reference",
            )
            observed_mean = _finite_number(
                reference_summary.get("mean"),
                f"baseline aggregate NFE {nfe} {metric} reference mean",
            )
            expected_mean = float(np.mean(values[str(nfe)][metric]))
            _require(
                reference_summary.get("n") == WORLD_SIZE
                and _same_number(observed_mean, expected_mean),
                (
                    f"baseline aggregate NFE {nfe} {metric} reference "
                    "does not match per-unit values"
                ),
            )
    return values, {
        "comparison": "video_only_s000",
        "legacy_artifact_scope": legacy_input.get("artifact_scope"),
    }


def _within_checkpoint_summary(
    analysis: Mapping[str, Any],
    *,
    arm: str,
    values: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> dict[str, Any]:
    aggregate = _mapping(analysis.get("aggregate"), "analysis.aggregate")
    source_deltas = _mapping(
        _mapping(
            aggregate.get("within_arm_source_deltas"),
            "analysis within-arm source deltas",
        ).get(arm),
        f"analysis within-arm source deltas {arm}",
    )
    source_aggregates = _mapping(
        _mapping(
            aggregate.get("within_arm_condition_sources"),
            "analysis within-arm source aggregates",
        ).get(arm),
        f"analysis within-arm source aggregates {arm}",
    )
    _require(
        source_aggregates.get("declared_sources") == list(EXPECTED_SOURCES),
        "aggregate source declaration mismatch",
    )
    source_summaries = _mapping(
        source_aggregates.get("sources"), "analysis source summaries"
    )

    comparisons: dict[str, Any] = {}
    for comparison_name in SOURCE_COMPARISONS:
        right_source = comparison_name.removeprefix("autonomous_minus_")
        comparison = _mapping(
            source_deltas.get(comparison_name),
            f"analysis comparison {comparison_name}",
        )
        _require(
            comparison.get("oracle_leakage") is False
            and comparison.get("deployable_evidence") is True,
            f"comparison {comparison_name} must be deployable and non-oracle",
        )
        observed_relative = _mapping(
            comparison.get("relative_nfe"),
            f"analysis comparison {comparison_name} relative_nfe",
        )
        output_nfe: dict[str, Any] = {}
        for nfe in NFE_STEPS:
            observed_metrics = _mapping(
                observed_relative.get(str(nfe)),
                f"analysis comparison {comparison_name} NFE {nfe}",
            )
            output_metrics: dict[str, Any] = {}
            for metric, direction in METRIC_DIRECTIONS.items():
                expected = _relative_effect(
                    values["autonomous"][str(nfe)][metric],
                    values[right_source][str(nfe)][metric],
                    direction=direction,
                    label=(
                        f"within-arm-relative:{arm}:{comparison_name}:"
                        f"nfe:{nfe}:metric:{metric}"
                    ),
                )
                output_metrics[metric] = _validate_relative_record(
                    observed_metrics.get(metric),
                    expected,
                    (
                        f"analysis comparison {comparison_name} NFE {nfe} "
                        f"{metric}"
                    ),
                )
            output_nfe[str(nfe)] = output_metrics
        comparisons[comparison_name] = {
            "definition": f"autonomous minus {right_source}",
            "paired_unit_count": WORLD_SIZE,
            "nfe": output_nfe,
        }

    absolute_tf: dict[str, Any] = {}
    sources = _mapping(source_summaries, "analysis source summaries")
    for source in EXPECTED_SOURCES:
        source_record = _mapping(
            sources.get(source), f"analysis source summary {source}"
        )
        _require(
            source_record.get("oracle_leakage") is False,
            f"source summary {source} must not be oracle leakage",
        )
        source_nfe = _mapping(
            source_record.get("nfe"), f"analysis source summary {source} nfe"
        )
        absolute_tf[source] = {}
        for nfe in NFE_STEPS:
            metric_record = _mapping(
                _mapping(
                    source_nfe.get(str(nfe)),
                    f"analysis source summary {source} NFE {nfe}",
                ).get("tf_future_nmse"),
                f"analysis source summary {source} NFE {nfe} TF NMSE",
            )
            expected = _summary(
                values[source][str(nfe)]["tf_future_nmse"],
                label=(
                    f"within-arm:{arm}:source:{source}:nfe:{nfe}:"
                    "metric:tf_future_nmse"
                ),
            )
            _validate_summary_record(
                metric_record,
                expected,
                f"analysis source summary {source} NFE {nfe} TF NMSE",
            )
            absolute_tf[source][str(nfe)] = expected

    primary = comparisons["autonomous_minus_autonomous_shuffled"]["nfe"]
    temporal_4 = primary["4"][
        "decoded_temporal_difference_mse_unit_range"
    ]
    video_4 = primary["4"]["video_future_nmse"]
    temporal_8 = primary["8"][
        "decoded_temporal_difference_mse_unit_range"
    ]
    point_criteria = {
        "nfe4_temporal_at_least_3pct_better": (
            temporal_4["relative_effect"] <= PRIMARY_TEMPORAL_IMPROVEMENT
        ),
        "nfe4_video_nmse_no_regression": (
            video_4["relative_effect"] <= PRIMARY_VIDEO_NO_REGRESSION
        ),
        "nfe8_temporal_direction_agrees_strictly": (
            temporal_8["relative_effect"] < 0
        ),
    }
    point_pass = all(point_criteria.values())
    sign_criteria = {
        "nfe4_temporal_ci_upper_below_zero": (
            temporal_4["bootstrap_ci"]["high"] < 0
        ),
        "nfe4_video_nmse_ci_upper_below_zero": (
            video_4["bootstrap_ci"]["high"] < 0
        ),
        "nfe8_temporal_ci_upper_below_zero": (
            temporal_8["bootstrap_ci"]["high"] < 0
        ),
    }
    return {
        "primary_comparison": "autonomous_minus_autonomous_shuffled",
        "comparisons": comparisons,
        "absolute_tf_future_nmse": absolute_tf,
        "preregistered_gate": {
            "scope": (
                "single-checkpoint stage-faithful autonomous TF versus one "
                "future-only shuffled generated-TF control"
            ),
            "thresholds": {
                "nfe4_temporal_relative_effect_max": (
                    PRIMARY_TEMPORAL_IMPROVEMENT
                ),
                "nfe4_video_nmse_relative_effect_max": (
                    PRIMARY_VIDEO_NO_REGRESSION
                ),
                "nfe8_temporal_relative_effect_strictly_below": 0.0,
                "sign_supported_ci_upper_strictly_below": 0.0,
            },
            "point_criteria": point_criteria,
            "point_gate_pass": point_pass,
            "sign_supported_criteria": sign_criteria,
            "sign_supported_gate_pass": (
                point_pass and all(sign_criteria.values())
            ),
        },
    }


def _video_only_comparisons(
    stage_values: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    video_only_values: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    equal_total: dict[str, Any] = {}
    for nfe in NFE_STEPS:
        equal_total[str(nfe)] = {}
        for metric in VIDEO_ONLY_METRICS:
            equal_total[str(nfe)][metric] = _relative_effect(
                stage_values["autonomous"][str(nfe)][metric],
                video_only_values[str(nfe)][metric],
                direction=METRIC_DIRECTIONS[metric],
                label=(
                    "stage-vs-video-only:equal-total-nfe:"
                    f"{nfe}:metric:{metric}"
                ),
            )
    cross_nfe: dict[str, Any] = {}
    for metric in VIDEO_ONLY_METRICS:
        cross_nfe[metric] = _relative_effect(
            stage_values["autonomous"]["8"][metric],
            video_only_values["2"][metric],
            direction=METRIC_DIRECTIONS[metric],
            label=(
                "stage-vs-video-only:stage-nfe8-vs-video-nfe2:"
                f"metric:{metric}"
            ),
        )
    return {
        "scope": (
            "secondary cross-checkpoint/cross-training-regime package "
            "comparison; not primary causal evidence"
        ),
        "nfe_is_total_model_calls": True,
        "equal_total_nfe": {
            "definition": (
                "stage-faithful autonomous minus video-only autonomous at "
                "equal total model calls"
            ),
            "nfe": equal_total,
        },
        "stage_nfe8_vs_video_only_nfe2": {
            "definition": (
                "stage-faithful autonomous at eight total model calls minus "
                "video-only autonomous at two model calls"
            ),
            "nfe": {"stage": 8, "video_only": 2},
            "metrics": cross_nfe,
        },
    }


def build_summary(
    analysis: Mapping[str, Any],
    baseline_reference: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic, fail-closed summary from parsed evidence."""

    _require(
        set(input_sha256)
        == {"analysis", "baseline_reference", "audit"},
        "input_sha256 must identify exactly the three evidence inputs",
    )
    for name, value in input_sha256.items():
        _sha256_value(value, f"input SHA-256 for {name}")
    arm, pairs, stage_values, analysis_evidence = _analysis_evidence(analysis)
    audit_evidence = _validate_audit(
        audit, analysis_provenance=analysis_evidence
    )
    video_only_values, baseline_evidence = _validate_baseline(
        baseline_reference, audit_evidence=audit_evidence
    )
    within = _within_checkpoint_summary(
        analysis, arm=arm, values=stage_values
    )
    cross = _video_only_comparisons(stage_values, video_only_values)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "stage_faithful_cascade_evaluation_summary",
        "sigma_convention": SIGMA_CONVENTION,
        "deterministic_output": True,
        "bootstrap": {
            "method": (
                "paired nonparametric percentile bootstrap over the exact "
                "eight dataset/rank units"
            ),
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "per_statistic_seed_derivation": (
                "first uint64 little-endian of sha256(seed + NUL + label)"
            ),
        },
        "validation": {
            "overall_pass": True,
            "exact_pair_identity_and_order_pass": True,
            "paired_unit_count": WORLD_SIZE,
            "paired_units": [
                {"dataset": dataset, "global_rank": rank}
                for dataset, rank in pairs
            ],
            "bitwise_audit_pass": True,
            "bitwise_audit_identity_sha256": audit_evidence[
                "identity_sha256"
            ],
            "analysis_arm": arm,
            "baseline_reference_comparison": baseline_evidence["comparison"],
            "input_sha256": dict(sorted(input_sha256.items())),
        },
        "within_checkpoint_stage_faithful": within,
        "video_only_reference": cross,
        "interpretation_limits": {
            "single_checkpoint_single_dataset_screen": True,
            "within_checkpoint_generated_tf_comparison_is_primary": True,
            "video_only_comparisons_are_cross_training_regime": True,
            "wall_time_or_fps_measured": False,
        },
    }


def _output_path(value: str | Path, inputs: Sequence[Path]) -> Path:
    raw = Path(value).expanduser()
    if raw.suffix.lower() != ".json":
        raise StageEvaluationSummaryError("output path must end in .json")
    if raw.exists() or raw.is_symlink():
        raise StageEvaluationSummaryError(f"output already exists: {raw}")
    try:
        info = raw.parent.lstat()
    except FileNotFoundError as exc:
        raise StageEvaluationSummaryError(
            f"output parent is missing: {raw.parent}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise StageEvaluationSummaryError(
            f"output parent must be a non-symlink directory: {raw.parent}"
        )
    path = raw.parent.resolve(strict=True) / raw.name
    _require(
        path not in inputs,
        "output path must not overwrite an input evidence file",
    )
    return path


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StageEvaluationSummaryError(
            f"could not exclusively create output {path}: {exc}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def summarize(
    *,
    analysis: str | Path,
    baseline_reference: str | Path,
    audit: str | Path,
    output: str | Path,
) -> tuple[dict[str, Any], str]:
    input_paths = {
        "analysis": _regular_json_path(analysis, "analysis"),
        "baseline_reference": _regular_json_path(
            baseline_reference, "baseline-reference"
        ),
        "audit": _regular_json_path(audit, "audit"),
    }
    output_path = _output_path(output, list(input_paths.values()))
    payloads = {
        name: _load_json_strict(path, name)
        for name, path in input_paths.items()
    }
    payload = build_summary(
        payloads["analysis"],
        payloads["baseline_reference"],
        payloads["audit"],
        input_sha256={
            name: _sha256_file(path) for name, path in input_paths.items()
        },
    )
    return payload, _exclusive_json(output_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        required=True,
        help="single-arm analyzer JSON for the stage-faithful evaluation",
    )
    parser.add_argument(
        "--baseline-reference",
        required=True,
        help="validated cascade cross-screen efficiency JSON",
    )
    parser.add_argument(
        "--audit",
        required=True,
        help="passing stage-faithful bitwise artifact audit JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="fresh deterministic JSON output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload, output_sha256 = summarize(
            analysis=args.analysis,
            baseline_reference=args.baseline_reference,
            audit=args.audit,
            output=args.output,
        )
    except StageEvaluationSummaryError as exc:
        print(f"stage evaluation summary failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "overall_pass": payload["validation"]["overall_pass"],
                "point_gate_pass": payload[
                    "within_checkpoint_stage_faithful"
                ]["preregistered_gate"]["point_gate_pass"],
                "sign_supported_gate_pass": payload[
                    "within_checkpoint_stage_faithful"
                ]["preregistered_gate"]["sign_supported_gate_pass"],
                "output": str(Path(args.output).expanduser().resolve()),
                "output_sha256": output_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
