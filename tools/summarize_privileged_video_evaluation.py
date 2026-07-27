#!/usr/bin/env python3
"""Summarize the preregistered privileged-TF Stage-A evaluation.

The command validates a passed bitwise artifact audit against a generic
``analyze_dual_nfe_artifacts.py`` result, then recomputes the two primary
cross-checkpoint comparisons with a deterministic paired bootstrap.  Artifact
validity and the scientific promotion gate are deliberately reported as
separate outcomes.
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

try:
    from tools.audit_privileged_video_artifacts import EXPECTED_PARENTS
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from audit_privileged_video_artifacts import (  # type: ignore[no-redef]
        EXPECTED_PARENTS,
    )


SCHEMA_VERSION = 1
SIGMA_CONVENTION = "1=noise,0=clean"
WORLD_SIZE = 8
ARTIFACT_ITERATION = 199
EVALUATION_NOISE_SEED = 20_260_726
NFE_STEPS = (1, 2, 4, 8)
ARM_NAMES = ("trained_matched", "trained_shuffled", "trained_off")
EXPECTED_SOURCES = ("autonomous", "off")
COMPARISONS = {
    "trained_matched_minus_trained_shuffled": (
        "trained_matched",
        "trained_shuffled",
    ),
    "trained_matched_minus_trained_off": (
        "trained_matched",
        "trained_off",
    ),
}
LOWER_IS_BETTER_METRICS = (
    "video_future_nmse",
    "tf_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
NOOP_EQUAL_METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_psnr_db",
    "decoded_temporal_difference_mse_unit_range",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_726
CONFIDENCE = 0.95
TEMPORAL_IMPROVEMENT = -0.03
POINT_NO_REGRESSION = 0.0
NONINFERIORITY_MARGIN = 0.02


class PrivilegedEvaluationSummaryError(RuntimeError):
    """Raised when evidence or a preregistered comparison contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrivilegedEvaluationSummaryError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PrivilegedEvaluationSummaryError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PrivilegedEvaluationSummaryError(f"{label} must be an array")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PrivilegedEvaluationSummaryError(
            f"{label} must be one finite JSON number"
        )
    return float(value)


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrivilegedEvaluationSummaryError(
            f"{label} must be one lowercase hexadecimal SHA-256"
        )
    return value


def _normal_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise PrivilegedEvaluationSummaryError(
            f"{label} must be a nonempty absolute path"
        )
    return os.path.normpath(value)


def _pairs(
    raw: Any,
    label: str,
    *,
    rank_key: str = "global_rank",
) -> tuple[tuple[str, int], ...]:
    records = _sequence(raw, label)
    result: list[tuple[str, int]] = []
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"{label}[{index}]")
        dataset = record.get("dataset")
        rank = record.get(rank_key)
        if (
            not isinstance(dataset, str)
            or not dataset
            or isinstance(rank, bool)
            or not isinstance(rank, int)
        ):
            raise PrivilegedEvaluationSummaryError(
                f"{label}[{index}] has invalid dataset/rank identity"
            )
        result.append((dataset, rank))
    return tuple(result)


def _regular_json_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PrivilegedEvaluationSummaryError(
            f"{label} is missing: {path}"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or path.suffix.lower() != ".json"
    ):
        raise PrivilegedEvaluationSummaryError(
            f"{label} must be a nonempty, non-symlink JSON file: {path}"
        )
    return path.resolve(strict=True)


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_strict(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PrivilegedEvaluationSummaryError(
                    f"duplicate key {key!r} in {label}: {path}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PrivilegedEvaluationSummaryError(
            f"non-finite JSON number {value!r} in {label}: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except PrivilegedEvaluationSummaryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivilegedEvaluationSummaryError(
            f"could not parse {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PrivilegedEvaluationSummaryError(
            f"{label} must contain one JSON object"
        )
    return payload


def _derived_seed(label: str) -> int:
    digest = hashlib.sha256(
        f"{BOOTSTRAP_SEED}\0{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _relative_effect(
    left_values: Sequence[float],
    reference_values: Sequence[float],
    *,
    label: str,
) -> dict[str, Any]:
    left = np.asarray(left_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    if (
        left.shape != (WORLD_SIZE,)
        or reference.shape != (WORLD_SIZE,)
        or not np.isfinite(left).all()
        or not np.isfinite(reference).all()
    ):
        raise PrivilegedEvaluationSummaryError(
            f"invalid paired values for {label}"
        )
    reference_mean = float(reference.mean())
    if reference_mean <= 0:
        raise PrivilegedEvaluationSummaryError(
            f"reference mean must be positive for {label}"
        )
    left_mean = float(left.mean())
    effect = (left_mean - reference_mean) / reference_mean
    rng = np.random.default_rng(_derived_seed(label))
    indices = rng.integers(
        0,
        WORLD_SIZE,
        size=(BOOTSTRAP_SAMPLES, WORLD_SIZE),
        endpoint=False,
    )
    left_means = left[indices].mean(axis=1)
    reference_means = reference[indices].mean(axis=1)
    if np.any(reference_means <= 0):
        raise PrivilegedEvaluationSummaryError(
            f"bootstrap reference mean is non-positive for {label}"
        )
    effects = (left_means - reference_means) / reference_means
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(effects, [tail, 1.0 - tail])
    return {
        "n": WORLD_SIZE,
        "definition": (
            "(mean(left)-mean(reference))/mean(reference), using exact paired "
            "dataset/rank resampling"
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
        "favorable_when": "relative_effect < 0",
        "ci_favors_left": bool(high < 0),
    }


def _all_pass_fields(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pass":
                _require(child is True, f"{label}.pass is not true")
            else:
                _all_pass_fields(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _all_pass_fields(child, f"{label}[{index}]")


def _validate_audit(
    audit: Mapping[str, Any],
) -> tuple[
    tuple[tuple[str, int], ...],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    _require(
        audit.get("schema_version") == SCHEMA_VERSION
        and audit.get("kind")
        == "privileged_tf_video_bitwise_artifact_audit",
        "unsupported artifact-audit schema/kind",
    )
    _require(
        audit.get("sigma_convention") == SIGMA_CONVENTION
        and audit.get("read_only_inputs") is True
        and audit.get("overall_pass") is True,
        "artifact audit did not pass",
    )
    _all_pass_fields(audit, "audit")
    identity_sha256 = _sha256_value(
        audit.get("identity_sha256"),
        "audit identity_sha256",
    )
    unsigned_audit = dict(audit)
    unsigned_audit.pop("identity_sha256", None)
    computed_identity = hashlib.sha256(
        json.dumps(
            unsigned_audit,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    _require(
        identity_sha256 == computed_identity,
        "artifact-audit identity signature does not match its contents",
    )
    contracts = _mapping(audit.get("contracts"), "audit.contracts")
    expected_ranks = list(range(WORLD_SIZE))
    exact_contracts = {
        "arm_names": (
            contracts.get("arm_names"),
            {
                "observed": list(ARM_NAMES),
                "expected": list(ARM_NAMES),
                "pass": True,
            },
        ),
        "world_size": (
            contracts.get("world_size"),
            {
                "observed": {arm: WORLD_SIZE for arm in ARM_NAMES},
                "expected": WORLD_SIZE,
                "pass": True,
            },
        ),
        "paired_ranks": (
            contracts.get("paired_ranks"),
            {
                "observed": {arm: expected_ranks for arm in ARM_NAMES},
                "expected": expected_ranks,
                "pass": True,
            },
        ),
        "artifact_iteration": (
            contracts.get("artifact_iteration"),
            {
                "observed": ARTIFACT_ITERATION,
                "expected": ARTIFACT_ITERATION,
                "pass": True,
            },
        ),
        "source_codes": (
            contracts.get("source_codes"),
            {
                "expected": [0, 1],
                "names": list(EXPECTED_SOURCES),
                "pass": True,
            },
        ),
        "nfe_steps": (
            contracts.get("nfe_steps"),
            {"expected": list(NFE_STEPS), "pass": True},
        ),
        "evaluation_noise_seed": (
            contracts.get("evaluation_noise_seed"),
            {"expected": EVALUATION_NOISE_SEED, "pass": True},
        ),
        "tf_content_disabled": (
            contracts.get("tf_content_disabled"),
            {
                "condition_on_tf": 0,
                "condition_mode_code": 0,
                "pass": True,
            },
        ),
        "tf_clock_disabled": (
            contracts.get("tf_clock_disabled"),
            {
                "evaluation_disable_tf_clock": 1,
                "evaluation_tf_clock_enabled": 0,
                "pass": True,
            },
        ),
        "noncascade_all_video_schedule": (
            contracts.get("noncascade_all_video_schedule"),
            {
                "cascade_stage_faithful_inference": 0,
                "evaluation_all_video_schedule": 1,
                "pass": True,
            },
        ),
        "raw_causal_inputs_present": (
            contracts.get("raw_causal_inputs_present"),
            {
                "raw_actions_present": 1,
                "raw_morphology_index_present": 1,
                "pass": True,
            },
        ),
        "exact_parent_provenance": (
            contracts.get("exact_parent_provenance"),
            {"expected": EXPECTED_PARENTS, "pass": True},
        ),
        "raw_action_morphology_input_identity": (
            contracts.get("raw_action_morphology_input_identity"),
            {
                "tensors": ["raw_actions", "raw_morphology_index"],
                "meaning": (
                    "exact raw causal inputs supplied to each checkpoint's "
                    "independently learned action encoder"
                ),
                "pass": True,
            },
        ),
        "learned_action_control_diagnostic": (
            contracts.get("learned_action_control_diagnostic"),
            {
                "tensor": "z_control",
                "cross_arm_equality_required": False,
                "meaning": (
                    "checkpoint-specific learned action control retained for "
                    "mechanism analysis, not treated as a paired raw input"
                ),
                "pass": True,
            },
        ),
    }
    for name, (observed, expected) in exact_contracts.items():
        _require(
            observed == expected,
            f"audit contract {name} is not exact",
        )
    forbidden = _mapping(
        contracts.get("forbidden_training_outputs"),
        "audit forbidden outputs",
    )
    _require(
        forbidden.get("observed") == {arm: [] for arm in ARM_NAMES}
        and forbidden.get("expected") == {arm: [] for arm in ARM_NAMES},
        "artifact audit found or misdeclared forbidden training outputs",
    )

    inputs = _mapping(audit.get("inputs"), "audit.inputs")
    _require(set(inputs) == set(ARM_NAMES), "audit arm inventory mismatch")
    input_evidence: dict[str, dict[str, Any]] = {}
    expected_pairs: tuple[tuple[str, int], ...] | None = None
    for arm in ARM_NAMES:
        arm_input = _mapping(inputs.get(arm), f"audit.inputs.{arm}")
        _normal_path(arm_input.get("root"), f"audit {arm} root")
        _normal_path(
            arm_input.get("artifact_scope"),
            f"audit {arm} artifact scope",
        )
        _sha256_value(
            arm_input.get("artifact_set_sha256"),
            f"audit {arm} artifact-set SHA-256",
        )
        provenance = _mapping(
            arm_input.get("evaluation_provenance"),
            f"audit {arm} evaluation provenance",
        )
        _normal_path(
            provenance.get("path"),
            f"audit {arm} provenance path",
        )
        _sha256_value(
            provenance.get("sha256"),
            f"audit {arm} provenance SHA-256",
        )
        _require(
            provenance.get("parent") == EXPECTED_PARENTS[arm]
            and provenance.get("completed_updates") == 200
            and isinstance(provenance.get("total_observations"), int)
            and not isinstance(provenance.get("total_observations"), bool)
            and provenance["total_observations"] > 0
            and provenance.get("runtime_intervention")
            == {
                "schedule_mode": "aligned",
                "tf_content_disabled": True,
                "tf_clock_disabled": True,
                "all_model_calls_advance_video": True,
            }
            and provenance.get("pass") is True,
            f"audit {arm} provenance evidence is not exact",
        )
        ranks = _sequence(
            arm_input.get("ranks"),
            f"audit.inputs.{arm}.ranks",
        )
        pairs = _pairs(ranks, f"audit {arm} ranks", rank_key="rank")
        _require(
            len(pairs) == WORLD_SIZE
            and [rank for _, rank in pairs] == expected_ranks,
            f"audit {arm} rank order is not exactly 0..7",
        )
        if expected_pairs is None:
            expected_pairs = pairs
        _require(
            pairs == expected_pairs,
            f"audit {arm} dataset/rank pairing differs",
        )
        artifacts: list[dict[str, Any]] = []
        for index, raw_rank in enumerate(ranks):
            rank_record = _mapping(raw_rank, f"audit {arm} rank {index}")
            artifact = _mapping(
                rank_record.get("artifact"),
                f"audit {arm} rank {index} artifact",
            )
            artifacts.append(
                {
                    "path": _normal_path(
                        artifact.get("path"),
                        f"audit {arm} rank {index} artifact path",
                    ),
                    "sha256": _sha256_value(
                        artifact.get("sha256"),
                        f"audit {arm} rank {index} artifact SHA-256",
                    ),
                }
            )
        input_evidence[arm] = {
            "root": _normal_path(
                arm_input.get("root"),
                f"audit {arm} root",
            ),
            "artifact_scope": _normal_path(
                arm_input.get("artifact_scope"),
                f"audit {arm} scope",
            ),
            "artifacts": artifacts,
            "evaluation_provenance": dict(provenance),
        }

    rank_audits = _sequence(audit.get("rank_audits"), "audit.rank_audits")
    _require(
        len(rank_audits) == WORLD_SIZE
        and [record.get("rank") for record in rank_audits] == expected_ranks,
        "audit rank records/order must be exactly 0..7",
    )
    for index, raw_rank in enumerate(rank_audits):
        rank_record = _mapping(raw_rank, f"audit rank {index}")
        _require(rank_record.get("pass") is True, f"audit rank {index} failed")
        for name in (
            "contracts",
            "cross_arm_input_identity",
            "autonomous_off_runtime_noop",
        ):
            _require(
                _mapping(
                    rank_record.get(name),
                    f"audit rank {index} {name}",
                ).get("pass")
                is True,
                f"audit rank {index} {name} failed",
            )
    assert expected_pairs is not None
    return expected_pairs, input_evidence, {
        "identity_sha256": identity_sha256,
        "artifact_set_sha256": {
            arm: _mapping(inputs[arm], f"audit {arm}").get(
                "artifact_set_sha256"
            )
            for arm in ARM_NAMES
        },
    }


def _validate_analysis(
    analysis: Mapping[str, Any],
    *,
    expected_pairs: tuple[tuple[str, int], ...],
    audit_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    _require(
        analysis.get("schema_version") == 1
        and analysis.get("kind")
        == "dual_video_diffusion_matched_nfe_analysis",
        "unsupported analysis schema/kind",
    )
    _require(
        analysis.get("sigma_convention") == SIGMA_CONVENTION,
        "analysis sigma convention mismatch",
    )
    _require(
        analysis.get("nfe_steps") == list(NFE_STEPS),
        f"analysis NFE steps must be exactly {list(NFE_STEPS)}",
    )
    bootstrap = _mapping(analysis.get("bootstrap"), "analysis.bootstrap")
    _require(
        bootstrap.get("samples") == BOOTSTRAP_SAMPLES
        and bootstrap.get("seed") == BOOTSTRAP_SEED
        and _finite_number(
            bootstrap.get("confidence"),
            "analysis bootstrap confidence",
        )
        == CONFIDENCE,
        "analysis bootstrap contract mismatch",
    )
    provenance = _mapping(analysis.get("provenance"), "analysis.provenance")
    _require(
        provenance.get("iteration") == ARTIFACT_ITERATION
        and provenance.get("paired_unit_count") == WORLD_SIZE,
        "analysis iteration/paired-unit count mismatch",
    )
    _require(
        _pairs(provenance.get("paired_units"), "analysis paired units")
        == expected_pairs,
        "analysis paired-unit identity/order differs from audit",
    )
    arms = _mapping(provenance.get("arms"), "analysis.provenance.arms")
    _require(set(arms) == set(ARM_NAMES), "analysis arm inventory mismatch")
    for arm in ARM_NAMES:
        arm_provenance = _mapping(arms.get(arm), f"analysis arm {arm}")
        _require(
            arm_provenance.get("artifact_count") == WORLD_SIZE
            and arm_provenance.get("evaluation_condition_sources")
            == list(EXPECTED_SOURCES),
            f"analysis arm {arm} artifact/source contract mismatch",
        )
        intervention = _mapping(
            arm_provenance.get("intervention"),
            f"analysis arm {arm} intervention",
        )
        _require(
            intervention.get("condition_on_tf") is False
            and intervention.get("condition_mode_code") == 0
            and intervention.get("condition_mode") == "off",
            f"analysis arm {arm} is not TF-content-off",
        )
        analysis_root = _normal_path(
            arm_provenance.get("root"),
            f"analysis arm {arm} root",
        )
        _require(
            analysis_root
            in {
                audit_inputs[arm]["root"],
                audit_inputs[arm]["artifact_scope"],
            },
            f"analysis arm {arm} root is not the audited root/scope",
        )
        artifacts = _sequence(
            arm_provenance.get("artifacts"),
            f"analysis arm {arm} artifacts",
        )
        _require(
            _pairs(artifacts, f"analysis arm {arm} artifacts")
            == expected_pairs,
            f"analysis arm {arm} artifact order differs from audit",
        )
        for index, raw_artifact in enumerate(artifacts):
            artifact = _mapping(
                raw_artifact,
                f"analysis arm {arm} artifact {index}",
            )
            audited = audit_inputs[arm]["artifacts"][index]
            _require(
                _normal_path(
                    artifact.get("path"),
                    f"analysis arm {arm} artifact {index} path",
                )
                == audited["path"]
                and _sha256_value(
                    artifact.get("sha256"),
                    f"analysis arm {arm} artifact {index} SHA-256",
                )
                == audited["sha256"],
                f"analysis/audit provenance mismatch for {arm} rank {index}",
            )

    values = {
        arm: {
            str(nfe): {metric: [] for metric in LOWER_IS_BETTER_METRICS}
            for nfe in NFE_STEPS
        }
        for arm in ARM_NAMES
    }
    per_units = _sequence(
        analysis.get("per_paired_unit"),
        "analysis.per_paired_unit",
    )
    _require(
        _pairs(per_units, "analysis per_paired_unit") == expected_pairs,
        "analysis per-unit identity/order differs from audit",
    )
    for unit_index, raw_unit in enumerate(per_units):
        unit = _mapping(raw_unit, f"analysis unit {unit_index}")
        unit_arms = _mapping(
            unit.get("arms"),
            f"analysis unit {unit_index} arms",
        )
        _require(
            set(unit_arms) == set(ARM_NAMES),
            f"analysis unit {unit_index} arm inventory mismatch",
        )
        for arm in ARM_NAMES:
            arm_record = _mapping(
                unit_arms[arm],
                f"analysis unit {unit_index} arm {arm}",
            )
            source_records = _mapping(
                arm_record.get("condition_source_metrics"),
                f"analysis unit {unit_index} arm {arm} source metrics",
            )
            _require(
                set(source_records) == set(EXPECTED_SOURCES),
                f"analysis unit {unit_index} arm {arm} source mismatch",
            )
            parsed_sources: dict[str, Mapping[str, Any]] = {}
            for source in EXPECTED_SOURCES:
                source_record = _mapping(
                    source_records[source],
                    (
                        f"analysis unit {unit_index} arm {arm} "
                        f"source {source}"
                    ),
                )
                _require(
                    source_record.get("oracle_leakage") is False,
                    f"{arm}/{source} cannot be marked oracle leakage",
                )
                source_metrics = _mapping(
                    source_record.get("metrics"),
                    (
                        f"analysis unit {unit_index} arm {arm} "
                        f"source {source} metrics"
                    ),
                )
                _require(
                    set(source_metrics) == {str(nfe) for nfe in NFE_STEPS},
                    f"{arm}/{source} NFE inventory mismatch",
                )
                parsed_sources[source] = source_metrics
            for nfe in NFE_STEPS:
                autonomous = _mapping(
                    parsed_sources["autonomous"][str(nfe)],
                    f"{arm} autonomous NFE {nfe}",
                )
                off = _mapping(
                    parsed_sources["off"][str(nfe)],
                    f"{arm} off NFE {nfe}",
                )
                for metric in NOOP_EQUAL_METRICS:
                    left = _finite_number(
                        autonomous.get(metric),
                        f"{arm} autonomous NFE {nfe} {metric}",
                    )
                    right = _finite_number(
                        off.get(metric),
                        f"{arm} off NFE {nfe} {metric}",
                    )
                    _require(
                        left == right,
                        f"{arm} rank {unit_index} NFE {nfe} autonomous/off "
                        f"metric mismatch for {metric}",
                    )
                for metric in LOWER_IS_BETTER_METRICS:
                    values[arm][str(nfe)][metric].append(
                        _finite_number(
                            autonomous.get(metric),
                            f"{arm} autonomous NFE {nfe} {metric}",
                        )
                    )
    return values


def _stage_a_gate(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    comparison_gates: dict[str, Any] = {}
    for name in COMPARISONS:
        result = _mapping(comparisons.get(name), f"comparison {name}")
        nfe = _mapping(result.get("nfe"), f"comparison {name} nfe")
        temporal_2 = _mapping(nfe["2"], f"{name} NFE 2")[
            "decoded_temporal_difference_mse_unit_range"
        ]
        video_4 = _mapping(nfe["4"], f"{name} NFE 4")[
            "video_future_nmse"
        ]
        decoded_4 = _mapping(nfe["4"], f"{name} NFE 4")[
            "decoded_mse_unit_range"
        ]
        temporal_4 = _mapping(nfe["4"], f"{name} NFE 4")[
            "decoded_temporal_difference_mse_unit_range"
        ]
        temporal_8 = _mapping(nfe["8"], f"{name} NFE 8")[
            "decoded_temporal_difference_mse_unit_range"
        ]
        criteria = {
            "nfe4_temporal_at_least_3pct_better": (
                temporal_4["relative_effect"] <= TEMPORAL_IMPROVEMENT
            ),
            "nfe4_video_point_no_regression": (
                video_4["relative_effect"] <= POINT_NO_REGRESSION
            ),
            "nfe4_decoded_point_no_regression": (
                decoded_4["relative_effect"] <= POINT_NO_REGRESSION
            ),
            "nfe4_temporal_ci_upper_below_zero": (
                temporal_4["bootstrap_ci"]["high"] < 0.0
            ),
            "nfe4_video_ci_upper_below_2pct": (
                video_4["bootstrap_ci"]["high"] < NONINFERIORITY_MARGIN
            ),
            "nfe4_decoded_ci_upper_below_2pct": (
                decoded_4["bootstrap_ci"]["high"]
                < NONINFERIORITY_MARGIN
            ),
            "nfe2_temporal_direction_agrees": (
                temporal_2["relative_effect"] < 0.0
            ),
            "nfe8_temporal_direction_agrees": (
                temporal_8["relative_effect"] < 0.0
            ),
        }
        comparison_gates[name] = {
            "criteria": criteria,
            "pass": all(criteria.values()),
        }
    return {
        "name": "preregistered_privileged_tf_stage_a_promotion_gate",
        "scope": (
            "trained_matched must pass every criterion against both "
            "trained_shuffled and trained_off"
        ),
        "thresholds": {
            "nfe4_temporal_relative_effect_max": TEMPORAL_IMPROVEMENT,
            "nfe4_video_and_decoded_point_max": POINT_NO_REGRESSION,
            "nfe4_temporal_ci_upper_strictly_below": 0.0,
            "nfe4_video_and_decoded_ci_upper_strictly_below": (
                NONINFERIORITY_MARGIN
            ),
            "nfe2_and_nfe8_temporal_effect_strictly_below": 0.0,
        },
        "comparisons": comparison_gates,
        "scientific_gate_pass": all(
            result["pass"] for result in comparison_gates.values()
        ),
    }


def build_summary(
    analysis: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Validate evidence and build one deterministic Stage-A summary."""

    _require(
        set(input_sha256) == {"analysis", "audit"},
        "input_sha256 must identify exactly analysis and audit",
    )
    for label, value in input_sha256.items():
        _sha256_value(value, f"{label} input SHA-256")
    pairs, audit_inputs, audit_evidence = _validate_audit(audit)
    values = _validate_analysis(
        analysis,
        expected_pairs=pairs,
        audit_inputs=audit_inputs,
    )
    comparisons: dict[str, Any] = {}
    for comparison_name, (left_arm, reference_arm) in COMPARISONS.items():
        nfe_results: dict[str, Any] = {}
        for nfe in NFE_STEPS:
            nfe_results[str(nfe)] = {
                metric: _relative_effect(
                    values[left_arm][str(nfe)][metric],
                    values[reference_arm][str(nfe)][metric],
                    label=(
                        f"privileged-stage-a:{comparison_name}:"
                        f"nfe:{nfe}:metric:{metric}"
                    ),
                )
                for metric in LOWER_IS_BETTER_METRICS
            }
        comparisons[comparison_name] = {
            "definition": f"{left_arm} minus {reference_arm}",
            "condition_source": "autonomous",
            "deployment_intervention": (
                "all-video native schedule; TF state content and TF clock "
                "disabled"
            ),
            "paired_unit_count": WORLD_SIZE,
            "nfe": nfe_results,
        }
    gate = _stage_a_gate(comparisons)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "privileged_tf_video_stage_a_evaluation_summary",
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
        "artifact_validation": {
            "artifact_audit_pass": True,
            "artifact_audit_identity_sha256": audit_evidence[
                "identity_sha256"
            ],
            "artifact_set_sha256": audit_evidence[
                "artifact_set_sha256"
            ],
            "exact_pair_identity_and_order_pass": True,
            "raw_action_morphology_input_identity_pass": True,
            "learned_z_control_cross_arm_equality_required": False,
            "paired_unit_count": WORLD_SIZE,
            "paired_units": [
                {"dataset": dataset, "global_rank": rank}
                for dataset, rank in pairs
            ],
            "input_sha256": dict(sorted(input_sha256.items())),
        },
        "scientific_evaluation": {
            "comparisons": comparisons,
            "preregistered_stage_a_gate": gate,
        },
        "decision": {
            "artifact_audit_pass": True,
            "scientific_gate_pass": gate["scientific_gate_pass"],
            "promote_to_stage_b": gate["scientific_gate_pass"],
            "audit_pass_does_not_imply_scientific_gate_pass": True,
        },
        "claim_boundary": {
            "single_training_seed": True,
            "eight_ranks_are_paired_evaluation_units_not_model_seeds": True,
            "temporal_difference_mse_is_not_perceptual_temporal_quality": True,
            "no_wall_clock_speed_or_fps_measurement": True,
            "primary_gate_does_not_depend_on_true_video_only": True,
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    payload["summary_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _output_path(value: str | Path, inputs: Sequence[Path]) -> Path:
    raw = Path(value).expanduser()
    if raw.suffix.lower() != ".json":
        raise PrivilegedEvaluationSummaryError(
            "output path must end in .json"
        )
    if raw.exists() or raw.is_symlink():
        raise PrivilegedEvaluationSummaryError(
            f"output already exists: {raw}"
        )
    try:
        info = raw.parent.lstat()
    except FileNotFoundError as exc:
        raise PrivilegedEvaluationSummaryError(
            f"output parent is missing: {raw.parent}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PrivilegedEvaluationSummaryError(
            f"output parent must be a non-symlink directory: {raw.parent}"
        )
    output = raw.parent.resolve(strict=True) / raw.name
    _require(output not in inputs, "output must not overwrite an input")
    return output


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
        raise PrivilegedEvaluationSummaryError(
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
    analysis_path: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load immutable evidence and exclusively write one summary."""

    analysis_file = _regular_json_path(analysis_path, "analysis")
    audit_file = _regular_json_path(audit_path, "artifact audit")
    _require(
        analysis_file != audit_file,
        "analysis and audit inputs must be distinct files",
    )
    output = _output_path(output_path, (analysis_file, audit_file))
    payload = build_summary(
        _load_json_strict(analysis_file, "analysis"),
        _load_json_strict(audit_file, "artifact audit"),
        input_sha256={
            "analysis": _sha256_file(analysis_file),
            "audit": _sha256_file(audit_file),
        },
    )
    _exclusive_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        required=True,
        help="Generic three-arm analyzer JSON",
    )
    parser.add_argument(
        "--audit",
        required=True,
        help="Passed privileged-video bitwise artifact audit JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New external .json output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = summarize(
            analysis_path=args.analysis,
            audit_path=args.audit,
            output_path=args.output,
        )
        output = Path(args.output).expanduser().resolve(strict=True)
        output_sha256 = _sha256_file(output)
    except (PrivilegedEvaluationSummaryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": output_sha256,
                "summary_identity_sha256": payload[
                    "summary_identity_sha256"
                ],
                "artifact_audit_pass": payload["decision"][
                    "artifact_audit_pass"
                ],
                "scientific_gate_pass": payload["decision"][
                    "scientific_gate_pass"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
