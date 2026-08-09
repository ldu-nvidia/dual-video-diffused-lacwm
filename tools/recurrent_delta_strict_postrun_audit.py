#!/usr/bin/env python3
"""Strict read-only audit for the completed recurrent-delta renderer census.

The original run directory is treated as immutable input.  This auditor closes
the deliberately narrow gaps left by the in-run audit: it reopens the frozen
registered inputs, reconstructs every causal predictor input, validates every
row/index/donor binding, and independently recomputes the bootstrap decisions.
Only a sealed report outside the canonical run root may be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from tools import corrected_renderer_attribution as base
from tools import recurrent_delta_spatial_confirmation as study
from tools import trajectory_consistent_renderer_gate as predecessor


SCHEMA_VERSION = 1
EXPECTED_SOURCE_COMMIT = "92cc9584298d2a395f3a79152768c9d0b88b3864"
EXPECTED_RUN_BASENAME = (
    "recurrent-delta-spatial-confirmation-renderer55-seed20260809-92cc958-v1"
)
CAUSAL_REPLAY_ATOL = 5e-6

DOCUMENT_NAMES = (
    "registration.json",
    "preparation.json",
    "preparation_complete.json",
    "causal_trajectory_closure.json",
    "evaluation.json",
    "evaluation_complete.json",
    "analysis.json",
    "run_complete.json",
)

METRIC_NAMES = {
    "silhouette_boundary_chamfer_px",
    "silhouette_iou",
    "rgb_robot_band_chamfer_px",
    "rgb_robot_band_edge_support_3px",
    "candidate_boundary_in_oracle_band_fraction",
    "fixed_robot_band_fraction",
    "fixed_robot_band_rgb_edge_pixels",
    "robot_flow_epe_px",
    "robot_flow_epe_p95_px",
    "robot_flow_epe_sum_px",
    "oracle_flow_magnitude_px",
    "candidate_projection_valid_fraction",
    "oracle_source_reprojection_rmse_px",
    "oracle_source_sample_count",
    "oracle_moving_flow_sample_count",
    "oracle_target_in_frame_fraction",
    "flow_transition_scored",
}

FLOW_OPTIONAL_NAMES = {
    "robot_flow_epe_px",
    "robot_flow_epe_p95_px",
    "oracle_flow_magnitude_px",
    "candidate_projection_valid_fraction",
}

UNIT_INTERVAL_METRICS = {
    "silhouette_iou",
    "rgb_robot_band_edge_support_3px",
    "candidate_boundary_in_oracle_band_fraction",
    "fixed_robot_band_fraction",
    "candidate_projection_valid_fraction",
    "oracle_target_in_frame_fraction",
}


class StrictAuditError(RuntimeError):
    """Raised when canonical evidence violates the frozen audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StrictAuditError(message)


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    canonical = root.resolve()
    return resolved == canonical or canonical in resolved.parents


def validate_report_destination(canonical_root: Path, report: Path) -> tuple[Path, Path]:
    canonical = canonical_root.resolve()
    destination = report.resolve()
    _require(canonical.is_dir(), f"canonical run root is unavailable: {canonical}")
    _require(
        canonical.name == EXPECTED_RUN_BASENAME,
        f"canonical run basename differs: {canonical.name}",
    )
    _require(
        not _is_within(destination, canonical),
        "external report must not be inside the canonical run root",
    )
    _require(not destination.exists(), f"external report already exists: {destination}")
    return canonical, destination


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without overwriting any prior report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise StrictAuditError(f"external report already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


class RecordVerifier:
    """Hash file records once and retain a digest of the verified evidence set."""

    def __init__(self, canonical_root: Path):
        self.canonical_root = canonical_root.resolve()
        self._actual: dict[Path, tuple[int, str]] = {}
        self.record_checks = 0

    def resolve(self, record: Mapping[str, Any], *, fallback: Path | None = None) -> Path:
        raw = Path(str(record.get("path", "")))
        candidates = [raw]
        if fallback is not None:
            candidates.append(fallback)
        candidates.extend(
            (
                self.canonical_root / raw.name,
                self.canonical_root / "prior" / raw.name,
                self.canonical_root / "score_bundles" / raw.name,
                self.canonical_root / "overlays" / raw.name,
            )
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise StrictAuditError(f"recorded file is unavailable: {raw}")

    def verify(
        self,
        record: Mapping[str, Any],
        label: str,
        *,
        fallback: Path | None = None,
    ) -> Path:
        _require(
            isinstance(record.get("sha256"), str)
            and len(str(record["sha256"])) == 64,
            f"{label} lacks a SHA-256 record",
        )
        _require(isinstance(record.get("bytes"), int), f"{label} lacks a byte count")
        path = self.resolve(record, fallback=fallback)
        if path not in self._actual:
            self._actual[path] = (path.stat().st_size, base.sha256_file(path))
        actual_bytes, actual_sha = self._actual[path]
        _require(actual_bytes == int(record["bytes"]), f"{label} byte count differs: {path}")
        _require(actual_sha == record["sha256"], f"{label} SHA-256 differs: {path}")
        self.record_checks += 1
        return path

    def walk(self, value: Any, label: str = "root") -> None:
        if isinstance(value, Mapping):
            if {"path", "bytes", "sha256"}.issubset(value):
                self.verify(value, label)
            for key, child in value.items():
                self.walk(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.walk(child, f"{label}[{index}]")

    @property
    def unique_files(self) -> int:
        return len(self._actual)

    @property
    def unique_bytes(self) -> int:
        return sum(value[0] for value in self._actual.values())

    def evidence_manifest_sha256(self) -> str:
        rows = []
        for path, (size, digest) in sorted(self._actual.items(), key=lambda item: str(item[0])):
            location = (
                str(path.relative_to(self.canonical_root))
                if _is_within(path, self.canonical_root)
                else str(path)
            )
            rows.append({"path": location, "bytes": size, "sha256": digest})
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _require_false_access_flags(value: Any, location: str = "root") -> int:
    count = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in {"validation_accessed", "protected_test_accessed"}:
                count += 1
                _require(child is False, f"{child_location} must equal false")
            count += _require_false_access_flags(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += _require_false_access_flags(child, f"{location}[{index}]")
    return count


def load_documents(canonical_root: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for name in DOCUMENT_NAMES:
        path = canonical_root / name
        _require(path.is_file(), f"canonical phase document is absent: {path}")
        document = base.read_json(path)
        try:
            base.validate_identity(document, name)
        except Exception as exc:
            raise StrictAuditError(f"sealed identity differs: {name}") from exc
        documents[name] = document
    return documents


def _require_artifact_points_to(
    verifier: RecordVerifier,
    record: Mapping[str, Any],
    expected: Path,
    label: str,
) -> None:
    path = verifier.verify(record, label, fallback=expected)
    _require(path == expected.resolve(), f"{label} points to {path}, expected {expected}")


def validate_phase_contract(
    canonical_root: Path,
    documents: Mapping[str, dict[str, Any]],
    verifier: RecordVerifier,
) -> int:
    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    preparation_complete = documents["preparation_complete.json"]
    closure = documents["causal_trajectory_closure.json"]
    evaluation = documents["evaluation.json"]
    evaluation_complete = documents["evaluation_complete.json"]
    analysis = documents["analysis.json"]
    completion = documents["run_complete.json"]

    expected_kind_status = {
        "registration.json": (
            "recurrent_delta_spatial_confirmation_registration",
            "registered_before_remaining55_future_state_or_rgb_access",
        ),
        "preparation.json": (
            "recurrent_delta_spatial_confirmation_preparation",
            "all_causal_trajectories_materialized_before_target_access",
        ),
        "preparation_complete.json": (
            "recurrent_delta_spatial_confirmation_preparation_complete",
            "completed_before_target_access",
        ),
        "causal_trajectory_closure.json": (
            "recurrent_delta_causal_trajectory_closure",
            None,
        ),
        "evaluation.json": ("recurrent_delta_spatial_confirmation_evaluation", None),
        "evaluation_complete.json": (
            "recurrent_delta_spatial_confirmation_evaluation_complete",
            "completed",
        ),
        "analysis.json": ("recurrent_delta_spatial_confirmation_analysis", None),
        "run_complete.json": (
            "recurrent_delta_spatial_confirmation_complete",
            "completed",
        ),
    }
    for name, (kind, status) in expected_kind_status.items():
        _require(documents[name].get("kind") == kind, f"{name} kind differs")
        if status is not None:
            _require(documents[name].get("status") == status, f"{name} status differs")

    registration_identity = registration["identity_sha256"]
    preparation_identity = preparation["identity_sha256"]
    closure_identity = closure["identity_sha256"]
    evaluation_identity = evaluation["identity_sha256"]
    analysis_identity = analysis["identity_sha256"]
    _require(
        preparation.get("registration_identity_sha256") == registration_identity,
        "preparation-registration binding differs",
    )
    _require(
        preparation_complete.get("registration_identity_sha256") == registration_identity
        and preparation_complete.get("preparation_identity_sha256") == preparation_identity
        and preparation_complete.get("closure_identity_sha256") == closure_identity,
        "preparation-complete chain differs",
    )
    _require(
        closure.get("registration_identity_sha256") == registration_identity
        and closure.get("preparation_identity_sha256") == preparation_identity,
        "closure phase chain differs",
    )
    _require(
        evaluation.get("registration_identity_sha256") == registration_identity
        and evaluation.get("preparation_identity_sha256") == preparation_identity
        and evaluation.get("closure_identity_sha256") == closure_identity,
        "evaluation phase chain differs",
    )
    _require(
        evaluation_complete.get("registration_identity_sha256") == registration_identity
        and evaluation_complete.get("evaluation_identity_sha256") == evaluation_identity,
        "evaluation-complete chain differs",
    )
    _require(
        analysis.get("registration_identity_sha256") == registration_identity
        and analysis.get("closure_identity_sha256") == closure_identity
        and analysis.get("evaluation_identity_sha256") == evaluation_identity,
        "analysis phase chain differs",
    )
    _require(
        completion.get("registration_identity_sha256") == registration_identity
        and completion.get("analysis_identity_sha256") == analysis_identity,
        "run-completion chain differs",
    )
    _require(
        completion.get("decision") == analysis.get("decision"),
        "completion-analysis decision differs",
    )

    _require(
        registration["source"].get("registered_git_commit") == EXPECTED_SOURCE_COMMIT
        and registration["source"].get("runtime_worktree_git_commit")
        == EXPECTED_SOURCE_COMMIT,
        "registered source commit differs",
    )
    _require(
        set(registration["source"].get("files", {}))
        == {"workflow", "predecessor", "renderer_base", "geometry", "protocol"},
        "registered source-file set differs",
    )
    try:
        study._verify_source_contract(registration)
    except Exception as exc:
        raise StrictAuditError("current scientific source hashes differ from registration") from exc

    for document in (preparation, closure):
        for key in (
            "future_measured_state_indices_extracted",
            "future_measured_state_used",
            "future_rgb_opened",
        ):
            _require(document.get(key) is False, f"{document['kind']}.{key} differs")
    _require(closure.get("immutable_before_evaluation") is True, "closure immutability flag differs")
    _require(
        evaluation.get("future_measured_state_opened_only_for_scoring") is True
        and evaluation.get("future_rgb_opened_only_for_scoring") is True,
        "evaluation scoring-only target flags differ",
    )
    _require(
        study._parse_time(closure["closed_at_utc"])
        <= study._parse_time(evaluation["target_opened_at_utc"]),
        "target access predates causal closure",
    )

    _require_artifact_points_to(
        verifier,
        preparation_complete["artifacts"]["preparation"],
        canonical_root / "preparation.json",
        "preparation_complete.artifacts.preparation",
    )
    _require_artifact_points_to(
        verifier,
        preparation_complete["artifacts"]["closure"],
        canonical_root / "causal_trajectory_closure.json",
        "preparation_complete.artifacts.closure",
    )
    _require_artifact_points_to(
        verifier,
        evaluation_complete["artifacts"]["evaluation"],
        canonical_root / "evaluation.json",
        "evaluation_complete.artifacts.evaluation",
    )
    _require_artifact_points_to(
        verifier,
        completion["artifacts"]["analysis"],
        canonical_root / "analysis.json",
        "run_complete.artifacts.analysis",
    )

    false_flags = 0
    for name, document in documents.items():
        false_flags += _require_false_access_flags(document, name)
    return false_flags


def validate_registered_inputs(
    registration: Mapping[str, Any], verifier: RecordVerifier
) -> tuple[list[dict[str, Any]], np.ndarray, Path, Path]:
    inputs = registration["inputs"]
    manifest_path = verifier.verify(inputs["train_manifest"], "inputs.train_manifest")
    metadata_path = verifier.verify(
        inputs["train_cache_metadata"], "inputs.train_cache_metadata"
    )
    actions_path = verifier.verify(inputs["train_actions"], "inputs.train_actions")
    rows = base.read_jsonl(manifest_path)
    try:
        base.validate_train_manifest(rows)
    except Exception as exc:
        raise StrictAuditError("registered train manifest contract differs") from exc
    try:
        actions, metadata, resolved_actions_path = base.cache_actions(metadata_path.parent)
    except Exception as exc:
        raise StrictAuditError("registered train cache contract differs") from exc
    _require(resolved_actions_path.resolve() == actions_path, "cache action path differs")
    _require(
        metadata.get("actions_sha256") == inputs["train_actions"].get("registered_sha256")
        == inputs["train_actions"].get("sha256"),
        "cache/registration action SHA binding differs",
    )
    _require(
        list(actions.shape) == inputs["train_actions"].get("shape")
        and str(actions.dtype) == inputs["train_actions"].get("dtype"),
        "registered action tensor geometry differs",
    )
    return rows, actions, Path(inputs["preprocessed_root"]), Path(inputs["raw_root"])


def validate_selected_rows(
    registration: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    preprocessed_root: Path,
    raw_root: Path,
) -> list[dict[str, Any]]:
    selected = registration["population"]["selected"]
    _require(len(selected) == study.SCORE_CLIPS, "selected clip count differs")
    _require(
        registration["population"].get("remaining_count") == study.SCORE_CLIPS,
        "registered remaining clip count differs",
    )
    indices: set[int] = set()
    clips: set[str] = set()
    strata = []
    for position, item in enumerate(selected):
        index = int(item["manifest_index"])
        _require(index not in indices, f"duplicate selected manifest index: {index}")
        _require(str(item["clip_id"]) not in clips, f"duplicate selected clip: {item['clip_id']}")
        indices.add(index)
        clips.add(str(item["clip_id"]))
        _require(item.get("score_order") == position, f"selected score order differs: {position}")
        _require(
            study.SCORE_START <= index < study.SCORE_STOP,
            f"selected manifest index outside frozen score pool: {index}",
        )
        row = rows[index]
        _require(str(row["clip_id"]) == str(item["clip_id"]), f"selected clip binding differs: {index}")
        _require(
            Path(row["episode_dir"]).resolve() == Path(item["episode_dir"]).resolve(),
            f"selected episode binding differs: {item['clip_id']}",
        )
        raw_mcap = base.raw_mcap_path(row, preprocessed_root, raw_root).resolve()
        _require(
            raw_mcap == Path(item["raw_mcap"]).resolve(),
            f"selected raw-MCAP binding differs: {item['clip_id']}",
        )
        _require(item.get("top_camera_type") == study.CAMERA_TYPE, "selected camera type differs")
        _require(item.get("protected_test_accessed") is False, "selected protected flag differs")
        strata.append(int(item["motion_stratum"]))
    counts = tuple(strata.count(value) for value in range(study.MOTION_STRATA))
    _require(counts == study.EXPECTED_STRATUM_COUNTS, f"selected strata differ: {counts}")
    return list(selected)


def _array_error(expected: np.ndarray, actual: np.ndarray, label: str, *, exact: bool) -> float:
    _require(expected.shape == actual.shape, f"{label} shape differs: {expected.shape}/{actual.shape}")
    _require(expected.dtype == actual.dtype, f"{label} dtype differs: {expected.dtype}/{actual.dtype}")
    if expected.dtype != np.bool_:
        _require(np.isfinite(expected).all(), f"expected {label} contains non-finite values")
        _require(np.isfinite(actual).all(), f"actual {label} contains non-finite values")
    if exact or expected.dtype == np.bool_:
        _require(np.array_equal(expected, actual), f"{label} differs")
        return 0.0
    if expected.size == 0:
        return 0.0
    error = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
    _require(error <= CAUSAL_REPLAY_ATOL, f"{label} replay differs: {error}")
    return error


def reconstruct_and_validate_causal_inputs(
    canonical_root: Path,
    registration: Mapping[str, Any],
    preparation: Mapping[str, Any],
    closure: Mapping[str, Any],
    selected: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    actions: np.ndarray,
    verifier: RecordVerifier,
) -> tuple[dict[str, np.ndarray], float, int]:
    provenance_record = preparation["artifacts"]["causal_input_provenance"]
    provenance_path = verifier.verify(
        provenance_record,
        "preparation.artifacts.causal_input_provenance",
        fallback=canonical_root / "causal_input_provenance.jsonl",
    )
    _require(
        provenance_path == (canonical_root / "causal_input_provenance.jsonl").resolve(),
        "causal provenance path differs",
    )
    provenance = base.read_jsonl(provenance_path)
    _require(len(provenance) == study.SCORE_CLIPS, "causal provenance row count differs")

    score_indices = np.asarray(
        [int(item["manifest_index"]) for item in selected], dtype=np.int64
    )
    position_by_index = {int(value): position for position, value in enumerate(score_indices)}
    donor_positions = np.asarray(
        [position_by_index[int(item["donor_manifest_index"])] for item in selected],
        dtype=np.int64,
    )
    expected_strata = np.asarray(
        [int(item["motion_stratum"]) for item in selected], dtype=np.int8
    )
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    q4_values: list[np.ndarray] = []
    false_flags = 0
    for position, (item, manifest_index) in enumerate(
        zip(selected, score_indices, strict=True)
    ):
        row = rows[int(manifest_index)]
        history, future, q4, expected_record = study._causal_clip_input(
            row, actions[int(manifest_index)]
        )
        expected_record.update(
            {
                "score_order": int(item["score_order"]),
                "manifest_index": int(manifest_index),
                "motion_stratum": int(item["motion_stratum"]),
                "donor_manifest_index": int(item["donor_manifest_index"]),
                "donor_clip_id": item["donor_clip_id"],
            }
        )
        _require(
            provenance[position] == expected_record,
            f"causal provenance differs: {item['clip_id']}",
        )
        _require(
            int(item["donor_manifest_index"]) in position_by_index,
            f"donor is outside selected population: {item['clip_id']}",
        )
        donor = selected[donor_positions[position]]
        _require(
            donor["clip_id"] == item["donor_clip_id"]
            and Path(donor["episode_dir"]).resolve()
            == Path(item["donor_episode_dir"]).resolve()
            and int(donor["motion_stratum"]) == int(item["motion_stratum"])
            and donor["clip_id"] != item["clip_id"],
            f"donor binding differs: {item['clip_id']}",
        )
        histories.append(history)
        futures.append(future)
        q4_values.append(q4)
        false_flags += _require_false_access_flags(provenance[position], "causal_provenance")

    history_array = np.stack(histories).astype(np.float32)
    future_array = np.stack(futures).astype(np.float32)
    q4_array = np.stack(q4_values).astype(np.float32)
    model_path = verifier.verify(
        registration["prior"]["artifacts"]["frozen_model"],
        "prior.frozen_model",
        fallback=canonical_root / "prior" / "model_state_and_predictions.npz",
    )
    _require(base.sha256_file(model_path) == study.PRIOR_MODEL_SHA256, "frozen model differs")
    replay = study.build_closed_trajectories(
        history_array, future_array, q4_array, donor_positions, model_path
    )
    expected: dict[str, np.ndarray] = {
        "score_indices": score_indices,
        "motion_strata": expected_strata,
        **replay,
    }

    trajectory_path = verifier.verify(
        closure["causal_trajectories"],
        "closure.causal_trajectories",
        fallback=canonical_root / "causal_trajectories.npz",
    )
    with np.load(trajectory_path, allow_pickle=False) as payload:
        closed = {name: payload[name].copy() for name in payload.files}
    _require(set(closed) == set(expected), "closed causal array-name set differs")
    exact_names = {
        "score_indices",
        "motion_strata",
        "history",
        "future_actions",
        "measured_q4",
        "donor_positions",
    }
    replay_error = 0.0
    for name in sorted(expected):
        replay_error = max(
            replay_error,
            _array_error(expected[name], closed[name], name, exact=name in exact_names),
        )
    _require(
        closure.get("trajectory_arrays") == sorted(replay),
        "closure trajectory-array inventory differs",
    )
    return closed, replay_error, false_flags


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{label} is not numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{label} is non-finite")
    return number


def _validate_metrics(metrics: Mapping[str, Any], label: str) -> None:
    _require(set(metrics) == METRIC_NAMES, f"{label} metric-name set differs")
    scored = metrics["flow_transition_scored"]
    _require(isinstance(scored, bool), f"{label}.flow_transition_scored is not boolean")
    count = _finite_number(metrics["oracle_moving_flow_sample_count"], f"{label}.count")
    if scored:
        _require(count >= 10.0, f"{label} scored flow has insufficient support")
        for name in FLOW_OPTIONAL_NAMES:
            _finite_number(metrics[name], f"{label}.{name}")
        epe = float(metrics["robot_flow_epe_px"])
        total = float(metrics["robot_flow_epe_sum_px"])
        _require(
            math.isclose(epe * count, total, rel_tol=2e-6, abs_tol=2e-5),
            f"{label} flow mean/sum/count differ",
        )
    else:
        _require(count < 10.0, f"{label} unscored flow has too much support")
        for name in FLOW_OPTIONAL_NAMES:
            _require(metrics[name] is None, f"{label}.{name} must be null")
        _require(
            float(metrics["robot_flow_epe_sum_px"]) == 0.0,
            f"{label} unscored flow sum differs",
        )
    for name, value in metrics.items():
        if name == "flow_transition_scored" or (name in FLOW_OPTIONAL_NAMES and value is None):
            continue
        number = _finite_number(value, f"{label}.{name}")
        if name in UNIT_INTERVAL_METRICS:
            _require(0.0 <= number <= 1.0, f"{label}.{name} is outside [0,1]")
        elif name not in {"oracle_source_reprojection_rmse_px"}:
            _require(number >= 0.0, f"{label}.{name} is negative")
    _require(
        float(metrics["oracle_source_reprojection_rmse_px"]) <= 1e-5,
        f"{label} oracle source reprojection differs",
    )


def validate_metric_rows(
    rows: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
) -> int:
    expected_count = study.SCORE_CLIPS * 8 * len(study.ARMS)
    _require(len(rows) == expected_count, f"frame metric row count differs: {len(rows)}")
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    false_flags = 0
    selected_by_clip = {str(item["clip_id"]): item for item in selected}
    for row_index, row in enumerate(rows):
        clip_id = str(row.get("clip_id"))
        arm = str(row.get("arm"))
        transition = int(row.get("transition_ordinal", -1))
        _require(clip_id in selected_by_clip, f"unknown metric clip: {clip_id}")
        _require(arm in study.ARMS, f"unknown metric arm: {arm}")
        _require(1 <= transition <= 8, f"invalid transition ordinal: {transition}")
        key = (clip_id, arm, transition)
        _require(key not in by_key, f"duplicate metric row: {key}")
        by_key[key] = row
        item = selected_by_clip[clip_id]
        manifest = manifest_rows[int(item["manifest_index"])]
        frame_indices = manifest["frame_indices"][4:13]
        _require(row.get("score_order") == item["score_order"], f"metric score order differs: {key}")
        _require(
            int(row.get("motion_stratum", -1)) == int(item["motion_stratum"]),
            f"metric stratum differs: {key}",
        )
        _require(row.get("split") == "train", f"metric split differs: {key}")
        _require(
            row.get("target_access_after_closure") is True,
            f"metric target-access flag differs: {key}",
        )
        _require(
            int(row.get("source_frame_index", -1)) == int(frame_indices[transition - 1])
            and int(row.get("target_frame_index", -1)) == int(frame_indices[transition]),
            f"metric frame-index binding differs: {key}",
        )
        _validate_metrics(row["metrics"], f"metric_rows[{row_index}]")
        false_flags += _require_false_access_flags(row, f"metric_rows[{row_index}]")

    expected_keys = {
        (str(item["clip_id"]), arm, transition)
        for item in selected
        for arm in study.ARMS
        for transition in range(1, 9)
    }
    _require(set(by_key) == expected_keys, "metric row Cartesian product differs")

    shared_names = {
        "oracle_source_reprojection_rmse_px",
        "oracle_source_sample_count",
        "oracle_moving_flow_sample_count",
        "oracle_flow_magnitude_px",
        "oracle_target_in_frame_fraction",
        "flow_transition_scored",
        "fixed_robot_band_fraction",
        "fixed_robot_band_rgb_edge_pixels",
    }
    for item in selected:
        clip_id = str(item["clip_id"])
        for transition in range(1, 9):
            reference = by_key[(clip_id, study.ARMS[0], transition)]["metrics"]
            for arm in study.ARMS[1:]:
                candidate = by_key[(clip_id, arm, transition)]["metrics"]
                for name in shared_names:
                    _require(
                        candidate[name] == reference[name],
                        f"oracle-fixed support differs: {clip_id}/{transition}/{arm}/{name}",
                    )
            oracle = by_key[(clip_id, "measured_oracle", transition)]["metrics"]
            _require(
                abs(float(oracle["silhouette_boundary_chamfer_px"])) <= 1e-7
                and abs(float(oracle["silhouette_iou"]) - 1.0) <= 1e-7,
                f"measured-oracle silhouette diagnostic differs: {clip_id}/{transition}",
            )
            if oracle["flow_transition_scored"]:
                _require(
                    abs(float(oracle["robot_flow_epe_px"])) <= 1e-7
                    and abs(float(oracle["robot_flow_epe_sum_px"])) <= 1e-6,
                    f"measured-oracle flow diagnostic differs: {clip_id}/{transition}",
                )
    return false_flags


def validate_bundles(
    canonical_root: Path,
    evaluation: Mapping[str, Any],
    registration: Mapping[str, Any],
    closure: Mapping[str, Any],
    selected: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
    closed: Mapping[str, np.ndarray],
    verifier: RecordVerifier,
) -> int:
    records = evaluation["artifacts"]["bundles"]
    _require(len(records) == study.SCORE_CLIPS, "score-bundle record count differs")
    false_flags = 0
    for position, (item, record) in enumerate(zip(selected, records, strict=True)):
        clip_id = str(item["clip_id"])
        _require(record.get("clip_id") == clip_id, f"bundle record clip differs: {position}")
        bundle_path = verifier.verify(
            record["bundle"],
            f"bundle[{position}].npz",
            fallback=canonical_root / "score_bundles" / f"{clip_id}.npz",
        )
        metadata_path = verifier.verify(
            record["metadata"],
            f"bundle[{position}].metadata",
            fallback=canonical_root / "score_bundles" / f"{clip_id}.json",
        )
        metadata = base.read_json(metadata_path)
        try:
            base.validate_identity(metadata, str(metadata_path))
        except Exception as exc:
            raise StrictAuditError(f"bundle metadata identity differs: {clip_id}") from exc
        _require(
            metadata["identity_sha256"] == record.get("metadata_identity_sha256"),
            f"bundle metadata record identity differs: {clip_id}",
        )
        _require(
            metadata.get("registration_identity_sha256") == registration["identity_sha256"]
            and metadata.get("closure_identity_sha256") == closure["identity_sha256"],
            f"bundle phase binding differs: {clip_id}",
        )
        _require(
            metadata.get("score_order") == position
            and int(metadata.get("manifest_index", -1)) == int(item["manifest_index"])
            and metadata.get("clip_id") == clip_id
            and int(metadata.get("motion_stratum", -1)) == int(item["motion_stratum"])
            and metadata.get("split") == "train"
            and metadata.get("target_access_after_closure") is True,
            f"bundle population binding differs: {clip_id}",
        )
        verifier.walk(metadata.get("source_files", {}), f"bundle_metadata[{position}].source_files")
        manifest = manifest_rows[int(item["manifest_index"])]
        _require(
            Path(metadata["source_files"]["states_npz"]["path"]).resolve()
            == (Path(manifest["episode_dir"]) / "states.npz").resolve()
            and Path(metadata["source_files"]["top_mp4"]["path"]).resolve()
            == (Path(manifest["episode_dir"]) / "top.mp4").resolve()
            and Path(metadata["source_files"]["raw_mcap"]["path"]).resolve()
            == Path(item["raw_mcap"]).resolve(),
            f"bundle source path binding differs: {clip_id}",
        )
        false_flags += _require_false_access_flags(metadata, f"bundle_metadata[{position}]")
        false_flags += _require_false_access_flags(record, f"bundle_record[{position}]")

        with np.load(bundle_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        expected_names = {
            "rgb",
            "frame_indices",
            "frame_ts",
            "K",
            "D",
            "measured_trajectory",
        }
        for arm in study.ARMS:
            for rendered in ("rendered", "unclipped"):
                for endpoint in ("source", "target"):
                    expected_names.add(f"pose_{rendered}_{endpoint}_{arm}")
        _require(set(arrays) == expected_names, f"bundle array-name set differs: {clip_id}")
        _require(
            arrays["rgb"].shape == (9, 480, 640, 3) and arrays["rgb"].dtype == np.uint8,
            f"bundle RGB geometry differs: {clip_id}",
        )
        frame_indices = np.asarray(manifest["frame_indices"][4:13], dtype=np.int64)
        _require(
            np.array_equal(arrays["frame_indices"], frame_indices),
            f"bundle frame indices differ: {clip_id}",
        )
        _require(
            arrays["frame_ts"].shape == (9,) and arrays["frame_ts"].dtype == np.int64,
            f"bundle timestamps differ: {clip_id}",
        )
        _require(
            arrays["K"].shape == (3, 3)
            and arrays["K"].dtype == np.float64
            and arrays["D"].ndim == 1
            and arrays["D"].dtype == np.float64,
            f"bundle calibration geometry differs: {clip_id}",
        )
        measured = arrays["measured_trajectory"]
        _require(
            measured.shape == (9, study.ACTION_DIM)
            and measured.dtype == np.float32
            and np.isfinite(measured).all(),
            f"bundle measured trajectory differs: {clip_id}",
        )
        _require(
            np.array_equal(measured[0], closed["measured_q4"][position]),
            f"bundle q4/closure binding differs: {clip_id}",
        )
        for arm in study.CAUSAL_ARMS:
            for rendered in ("rendered", "unclipped"):
                for endpoint in ("source", "target"):
                    name = f"pose_{rendered}_{endpoint}_{arm}"
                    _require(
                        np.array_equal(arrays[name], closed[name][position]),
                        f"bundle/closure trajectory differs: {clip_id}/{name}",
                    )
        oracle_source, oracle_target = predecessor._trajectory_pairs(measured[None])
        oracle_expected = {
            "pose_unclipped_source_measured_oracle": oracle_source[0],
            "pose_unclipped_target_measured_oracle": oracle_target[0],
            "pose_rendered_source_measured_oracle": predecessor._render_pose(oracle_source[0]),
            "pose_rendered_target_measured_oracle": predecessor._render_pose(oracle_target[0]),
        }
        for name, values in oracle_expected.items():
            _require(
                np.array_equal(arrays[name], values),
                f"bundle measured-oracle trajectory differs: {clip_id}/{name}",
            )
    overlays = evaluation["artifacts"]["overlays"]
    _require(len(overlays) == study.SCORE_CLIPS, "overlay artifact count differs")
    _require(
        len({verifier.verify(record, "overlay") for record in overlays}) == study.SCORE_CLIPS,
        "overlay paths are not unique",
    )
    return false_flags


def validate_bootstrap(
    bootstrap_path: Path,
    expected_strata: np.ndarray,
) -> np.ndarray:
    with np.load(bootstrap_path, allow_pickle=False) as payload:
        _require(
            set(payload.files) == {"indices", "strata", "seed", "samples"},
            "bootstrap payload array-name set differs",
        )
        indices = payload["indices"].copy()
        strata = payload["strata"].copy()
        seed = int(payload["seed"])
        samples = int(payload["samples"])
    _require(seed == study.BOOTSTRAP_SEED, f"bootstrap seed differs: {seed}")
    _require(samples == study.BOOTSTRAP_SAMPLES, f"bootstrap sample count differs: {samples}")
    _require(
        strata.dtype == np.int8 and np.array_equal(strata, expected_strata),
        "bootstrap strata differ from registered selected order",
    )
    _require(
        indices.shape == (study.BOOTSTRAP_SAMPLES, study.SCORE_CLIPS)
        and indices.dtype == np.int16,
        f"bootstrap index geometry differs: {indices.shape}/{indices.dtype}",
    )
    expected = study.stratified_bootstrap_indices(
        expected_strata, samples=study.BOOTSTRAP_SAMPLES, seed=study.BOOTSTRAP_SEED
    )
    _require(np.array_equal(indices, expected), "bootstrap index matrix differs")
    return indices


def strict_audit(canonical_root: Path) -> dict[str, Any]:
    documents = load_documents(canonical_root)
    verifier = RecordVerifier(canonical_root)
    false_flags = validate_phase_contract(canonical_root, documents, verifier)
    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    closure = documents["causal_trajectory_closure.json"]
    evaluation = documents["evaluation.json"]
    analysis = documents["analysis.json"]
    completion = documents["run_complete.json"]

    # Verify every file record reachable from every sealed phase document.
    for name, document in documents.items():
        verifier.walk(document, name)

    manifest_rows, actions, preprocessed_root, raw_root = validate_registered_inputs(
        registration, verifier
    )
    selected = validate_selected_rows(
        registration, manifest_rows, preprocessed_root, raw_root
    )
    closed, replay_error, provenance_false_flags = reconstruct_and_validate_causal_inputs(
        canonical_root,
        registration,
        preparation,
        closure,
        selected,
        manifest_rows,
        actions,
        verifier,
    )
    false_flags += provenance_false_flags

    frame_path = verifier.verify(
        evaluation["artifacts"]["frame_transition_metrics"],
        "evaluation.frame_transition_metrics",
        fallback=canonical_root / "frame_transition_metrics.jsonl",
    )
    frame_rows = base.read_jsonl(frame_path)
    false_flags += validate_metric_rows(frame_rows, selected, manifest_rows)
    false_flags += validate_bundles(
        canonical_root,
        evaluation,
        registration,
        closure,
        selected,
        manifest_rows,
        closed,
        verifier,
    )

    aggregates = study._aggregate_clip_rows(frame_rows)
    clip_path = verifier.verify(
        analysis["artifacts"]["clip_metrics"],
        "analysis.clip_metrics",
        fallback=canonical_root / "clip_metrics.jsonl",
    )
    stored_aggregates = base.read_jsonl(clip_path)
    _require(aggregates == stored_aggregates, "stored clip aggregation differs")
    false_flags += _require_false_access_flags(stored_aggregates, "clip_metrics")

    expected_strata = np.asarray(
        [int(item["motion_stratum"]) for item in selected], dtype=np.int8
    )
    bootstrap_record = analysis["artifacts"]["bootstrap_indices"]
    bootstrap_path = verifier.verify(
        bootstrap_record,
        "analysis.bootstrap_indices",
        fallback=canonical_root / "bootstrap_indices.npz",
    )
    _require(
        analysis["bootstrap"].get("seed") == study.BOOTSTRAP_SEED
        and analysis["bootstrap"].get("samples") == study.BOOTSTRAP_SAMPLES
        and analysis["bootstrap"].get("type")
        == "common motion-stratified paired episode bootstrap"
        and analysis["bootstrap"]["indices"]["sha256"] == bootstrap_record["sha256"],
        "analysis bootstrap declaration differs",
    )
    indices = validate_bootstrap(bootstrap_path, expected_strata)
    clip_order = [str(item["clip_id"]) for item in selected]
    recomputed = study.analyze_aggregates(aggregates, clip_order, indices)
    for key in ("metrics_by_arm", "effects", "holm_families", "gates", "decision"):
        _require(recomputed[key] == analysis[key], f"recomputed analysis differs: {key}")

    _require(
        recomputed["decision"] == analysis["decision"] == completion["decision"],
        "recomputed/analysis/completion decisions differ",
    )
    builtin = study.audit(canonical_root)
    _require(builtin.get("status") == "audit_passed", "built-in audit did not pass")
    _require(
        builtin.get("decision") == recomputed["decision"],
        "built-in/strict audit decisions differ",
    )

    _require(
        evaluation.get("score_clips") == study.SCORE_CLIPS
        and evaluation.get("score_transitions") == study.SCORE_CLIPS * 8
        and evaluation.get("arms") == list(study.ARMS),
        "evaluation declared geometry differs",
    )
    _require(
        analysis.get("score_clips") == study.SCORE_CLIPS
        and analysis.get("stratum_counts") == list(study.EXPECTED_STRATUM_COUNTS),
        "analysis declared population differs",
    )

    return {
        "status": "strict_audit_passed",
        "canonical_run_root": str(canonical_root),
        "expected_source_commit": EXPECTED_SOURCE_COMMIT,
        "canonical_identities": {
            name: document["identity_sha256"] for name, document in documents.items()
        },
        "population": {
            "clips": study.SCORE_CLIPS,
            "transitions_per_clip": 8,
            "motion_strata": list(study.EXPECTED_STRATUM_COUNTS),
            "metric_rows": len(frame_rows),
            "clip_metric_rows": len(stored_aggregates),
        },
        "causal_reconstruction": {
            "arrays_compared": len(closed),
            "max_abs_error": replay_error,
            "exact_input_arrays": [
                "score_indices",
                "motion_strata",
                "history",
                "future_actions",
                "measured_q4",
                "donor_positions",
            ],
        },
        "bootstrap": {
            "seed": study.BOOTSTRAP_SEED,
            "samples": study.BOOTSTRAP_SAMPLES,
            "shape": list(indices.shape),
            "strata_exactly_registered": True,
            "index_matrix_reconstructed": True,
        },
        "decisions": {
            "recomputed": recomputed["decision"],
            "analysis": analysis["decision"],
            "run_complete": completion["decision"],
            "built_in_audit": builtin["decision"],
            "all_equal": True,
        },
        "artifact_verification": {
            "record_checks": verifier.record_checks,
            "unique_files": verifier.unique_files,
            "unique_bytes": verifier.unique_bytes,
            "evidence_manifest_sha256": verifier.evidence_manifest_sha256(),
        },
        "explicit_false_access_flags": false_flags,
        "built_in_audit": builtin,
        "canonical_root_mutated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical_root, report_path = validate_report_destination(
        args.canonical_root, args.report
    )
    canonical_stat_before = canonical_root.stat()
    result = strict_audit(canonical_root)
    canonical_stat_after = canonical_root.stat()
    _require(
        (canonical_stat_before.st_size, canonical_stat_before.st_mtime_ns)
        == (canonical_stat_after.st_size, canonical_stat_after.st_mtime_ns),
        "canonical run-root metadata changed during read-only audit",
    )
    report = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_strict_external_postrun_audit",
            "created_at_utc": base.now(),
            **result,
            "auditor": {
                "source": base.file_record(Path(__file__).resolve()),
                "git_commit": base.git_commit(REPO_ROOT),
                "python": os.sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "external_report_path": str(report_path),
            "report_is_outside_canonical_root": True,
        }
    )
    write_json_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "event": "strict_external_postrun_audit_completed",
                "report": str(report_path),
                "identity_sha256": report["identity_sha256"],
                "decision": report["decisions"]["recomputed"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
