#!/usr/bin/env python3
"""Fresh-lockbox within-J1 causality sidecar for the V-JEPA frontier study.

This module is deliberately separate from the validation-selected NFE
frontier.  It never selects an NFE, changes the main lockbox confirmation, or
feeds the frontier final report.  Only after an eligible validation selection
and the already-submitted J1 autonomous lockbox evaluation exist, it compares
that frozen J1@k endpoint with two same-checkpoint, same-noise controls:

* ``off``: generated auxiliary state is not fused into the video branch;
* ``autonomous_shuffled``: generated auxiliary state is paired with the wrong
  clip inside the fixed evaluation batch.

All three sources use the public history-only deployable sampler.  The
confirmation is based on 10,000 paired clip bootstraps with seed 1234.
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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools import vjepa2_nfe_frontier as frontier
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import vjepa2_nfe_frontier as frontier


SCHEMA_VERSION = 1
KIND_CONTINUATION = "vjepa2_nfe_frontier_slurm_continuation"
KIND_QUALITY_INVENTORY = "vjepa2_controlled_study_quality_inventory"
KIND_QUALITY_RANK = "vjepa2_controlled_study_quality_rank"
KIND_QUALITY_ROW = "vjepa2_controlled_study_quality_clip"
KIND_CONFIRMATION = "vjepa2_lockbox_within_j1_causality_confirmation"
SIDE_CAR_SOURCES = ("off", "autonomous_shuffled")
ALL_CAUSAL_SOURCES = ("autonomous", *SIDE_CAR_SOURCES)
EXPECTED_CLIPS = 128
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE_PER_RANK = 2
PRACTICAL_TEMPORAL_THRESHOLD = 0.03
JOB_ID_RE = re.compile(r"^[1-9][0-9]*(?:;[A-Za-z0-9_.%+-]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PAIRING_HASH_FIELDS = (
    "video_clean_sha256",
    "auxiliary_clean_sha256",
    "ground_truth_sha256",
    "vae_ground_truth_sha256",
    "raw_history_last_sha256",
    "vae_history_last_sha256",
    "cached_rgb_input_sha256",
    "cached_actions_input_sha256",
    "video_initial_state_sha256",
    "auxiliary_initial_state_sha256",
    "auxiliary_initial_noise_sha256",
)
OUTPUT_HASH_FIELDS = (
    "video_final_sha256",
    "auxiliary_final_sha256",
    "decoded_final_sha256",
)
EXPECTED_TENSOR_HASH_FIELDS = frozenset((*PAIRING_HASH_FIELDS, *OUTPUT_HASH_FIELDS))


class CausalityError(RuntimeError):
    """Raised when sidecar evidence is incomplete, reused, or unpaired."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == recorded


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CausalityError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file() or info.st_size <= 0:
        raise CausalityError(
            f"{label} must be a non-empty non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise CausalityError(f"{label} must use its canonical absolute path")
    return resolved


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise CausalityError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise CausalityError(f"{label} must be a non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise CausalityError(f"{label} must use its canonical absolute path")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CausalityError(
                    f"{label} contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalityError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CausalityError(f"{label} must contain one JSON object")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise CausalityError(
                        f"{label} has blank row {line_number}: {path}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CausalityError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalityError(f"{label} is invalid JSONL: {path}") from exc
    return rows


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CausalityError(f"refusing to overwrite output: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def validate_selection(selection: Mapping[str, Any]) -> int:
    """Reproduce the eligible validation selection and return frozen J1 NFE k."""

    try:
        reproduced = frontier.validate_confirmatory_selection(selection)
    except frontier.FrontierError as exc:
        raise CausalityError(str(exc)) from exc
    if reproduced.get("selection") != dict(selection):
        raise CausalityError("selection reproduction returned different evidence")
    pair = selection.get("selected_pair")
    left = pair.get("left") if isinstance(pair, Mapping) else None
    k = left.get("nfe") if isinstance(left, Mapping) else None
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k < frontier.MIN_CAUSAL_J1_NFE
        or k not in frontier.NFE_GRID
    ):
        raise CausalityError("selection lacks a valid causal J1 NFE")
    return k


def sidecar_grid(selection: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    k = validate_selection(selection)
    return tuple((source, k) for source in SIDE_CAR_SOURCES)


def _normalized_job_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or JOB_ID_RE.fullmatch(value) is None:
        raise CausalityError(f"{label} is not a valid Slurm job ID")
    return value.split(";", 1)[0]


def validate_continuation(
    continuation_path_value: str | Path,
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the sidecar to the existing conditional main-lockbox submission."""

    continuation_path = _canonical_file(
        continuation_path_value, "frontier continuation"
    )
    expected_path = selection_path.parent / "frontier_continuation.json"
    if continuation_path != expected_path:
        raise CausalityError(
            f"continuation must be the study record {expected_path}"
        )
    continuation = _read_json(continuation_path, "frontier continuation")
    selection_record = continuation.get("selection")
    if (
        continuation.get("kind") != KIND_CONTINUATION
        or continuation.get("schema_version") != SCHEMA_VERSION
        or continuation.get(
            "lockbox_jobs_created_only_after_confirmatory_eligibility"
        )
        is not True
        or not isinstance(selection_record, Mapping)
        or selection_record.get("path") != str(selection_path)
        or selection_record.get("sha256") != sha256_file(selection_path)
        or selection_record.get("identity_sha256")
        != selection.get("identity_sha256")
        or selection_record.get("confirmatory_eligible") is not True
        or continuation.get("scientific_evaluator_git_commit")
        != selection.get("evaluator_git_commit")
    ):
        raise CausalityError(
            "frontier continuation does not bind the eligible selection"
        )
    selection_job = _normalized_job_id(
        continuation.get("selection_gate_job_id"), "selection-gate job"
    )
    lockbox_job = _normalized_job_id(
        continuation.get("lockbox_quality_array_job_id"),
        "lockbox quality array job",
    )
    confirmation_job = _normalized_job_id(
        continuation.get("confirmation_job_id"), "main confirmation job"
    )
    _normalized_job_id(
        continuation.get("timing_and_finalization_job_id"),
        "main timing/finalization job",
    )
    if (
        continuation.get("lockbox_quality_dependency")
        != f"afterok:{selection_job}"
        or continuation.get("confirmation_dependency")
        != f"afterok:{lockbox_job}"
        or continuation.get("timing_dependency")
        != f"afterok:{confirmation_job}"
    ):
        raise CausalityError("frontier continuation dependency chain differs")
    for field in ("controller_git_commit", "scientific_evaluator_git_commit"):
        if (
            not isinstance(continuation.get(field), str)
            or COMMIT_RE.fullmatch(str(continuation[field])) is None
        ):
            raise CausalityError(f"continuation {field} is invalid")
    return {
        "path": str(continuation_path),
        "sha256": sha256_file(continuation_path),
        "payload": continuation,
        "lockbox_quality_array_job_id": lockbox_job,
        "j1_array_task_dependency": f"afterok:{lockbox_job}_1",
    }


def autonomous_inventory_path(
    study_root: Path, selection: Mapping[str, Any]
) -> Path:
    selection_id = str(selection["identity_sha256"])
    lockbox = selection.get("lockbox_registration")
    if not isinstance(lockbox, Mapping):
        raise CausalityError("selection lacks lockbox registration")
    lockbox_id = str(lockbox.get("identity_sha256", ""))
    if SHA256_RE.fullmatch(lockbox_id) is None:
        raise CausalityError("selection lockbox identity is invalid")
    return (
        study_root
        / "j1_joint_auxiliary_leads"
        / "frontier_quality"
        / "lockbox"
        / lockbox_id
        / selection_id
        / "update_1000"
        / "inventory.json"
    )


def sidecar_output_directory(
    run_dir: Path, selection: Mapping[str, Any]
) -> Path:
    lockbox = selection.get("lockbox_registration")
    if not isinstance(lockbox, Mapping):
        raise CausalityError("selection lacks lockbox registration")
    return (
        run_dir
        / "frontier_causality"
        / "lockbox"
        / str(lockbox["identity_sha256"])
        / str(selection["identity_sha256"])
        / "update_1000"
    )


def _load_inventory_rows(
    inventory_path_value: str | Path,
    *,
    label: str,
    expected_grid: Sequence[tuple[str, int]],
    expected_evaluator_commit: str,
    sidecar: bool,
) -> dict[str, Any]:
    """Fully validate rank manifests and hashed raw row evidence."""

    inventory_path = _canonical_file(inventory_path_value, f"{label} inventory")
    inventory = _read_json(inventory_path, f"{label} inventory")
    expected_grid_json = [
        {"source": source, "nfe": nfe} for source, nfe in expected_grid
    ]
    if (
        not identity_valid(inventory)
        or inventory.get("kind") != KIND_QUALITY_INVENTORY
        or inventory.get("schema_version") != SCHEMA_VERSION
        or inventory.get("complete") is not True
        or inventory.get("arm_code") != "J1"
        or inventory.get("completed_updates") != 1000
        or inventory.get("world_size") != EXPECTED_WORLD_SIZE
        or inventory.get("batch_size_per_rank")
        != EXPECTED_BATCH_SIZE_PER_RANK
        or inventory.get("clip_count") != EXPECTED_CLIPS
        or inventory.get("expected_record_count")
        != EXPECTED_CLIPS * len(expected_grid)
        or inventory.get("observed_record_count")
        != EXPECTED_CLIPS * len(expected_grid)
        or inventory.get("evaluation_split") != "lockbox"
        or inventory.get("frontier_mode") is not True
        or inventory.get("grid") != expected_grid_json
        or inventory.get("evaluator_git_commit")
        != expected_evaluator_commit
        or inventory.get("frontier_causality_sidecar", False) is not sidecar
    ):
        raise CausalityError(f"{label} inventory contract differs")
    rank_evidence = inventory.get("rank_evidence")
    if (
        not isinstance(rank_evidence, Mapping)
        or set(rank_evidence) != {str(rank) for rank in range(EXPECTED_WORLD_SIZE)}
    ):
        raise CausalityError(f"{label} rank inventory is incomplete")
    rows: list[dict[str, Any]] = []
    common_inputs: dict[str, Any] | None = None
    common_provenance: dict[str, Any] | None = None
    for rank in range(EXPECTED_WORLD_SIZE):
        record = rank_evidence[str(rank)]
        if not isinstance(record, Mapping):
            raise CausalityError(f"{label} rank {rank} evidence is invalid")
        manifest_path = _canonical_file(
            record.get("manifest_path", ""), f"{label} rank {rank} manifest"
        )
        rows_path = _canonical_file(
            record.get("rows_path", ""), f"{label} rank {rank} rows"
        )
        expected_parent = inventory_path.parent
        if (
            manifest_path != expected_parent / f"rank_{rank:03d}_manifest.json"
            or rows_path != expected_parent / f"rank_{rank:03d}.jsonl"
            or record.get("manifest_sha256") != sha256_file(manifest_path)
            or record.get("rows_sha256") != sha256_file(rows_path)
        ):
            raise CausalityError(f"{label} rank {rank} file evidence differs")
        manifest = _read_json(manifest_path, f"{label} rank {rank} manifest")
        rank_rows = _read_jsonl(rows_path, f"{label} rank {rank} rows")
        expected_indexes = list(range(rank, EXPECTED_CLIPS, EXPECTED_WORLD_SIZE))
        expected_wan_invocations = (
            len(expected_indexes)
            // EXPECTED_BATCH_SIZE_PER_RANK
            * sum(nfe for _source, nfe in expected_grid)
        )
        if (
            not identity_valid(manifest)
            or manifest.get("kind") != KIND_QUALITY_RANK
            or manifest.get("rank") != rank
            or manifest.get("world_size") != EXPECTED_WORLD_SIZE
            or manifest.get("batch_size_per_rank")
            != EXPECTED_BATCH_SIZE_PER_RANK
            or manifest.get("assigned_clip_indexes") != expected_indexes
            or manifest.get("grid") != expected_grid_json
            or manifest.get("evaluation_split") != "lockbox"
            or manifest.get("evaluator_git_commit")
            != expected_evaluator_commit
            or manifest.get("frontier_causality_sidecar", False) is not sidecar
            or manifest.get("rows", {}).get("path") != str(rows_path)
            or manifest.get("rows", {}).get("sha256") != sha256_file(rows_path)
            or manifest.get("rows", {}).get("count") != len(rank_rows)
            or manifest.get("actual_wan_backbone_invocations")
            != expected_wan_invocations
            or manifest.get("online_teacher_call_count") != 0
            or record.get("manifest_identity_sha256")
            != manifest.get("identity_sha256")
            or record.get("row_count") != len(rank_rows)
            or len(rank_rows) != len(expected_indexes) * len(expected_grid)
        ):
            raise CausalityError(f"{label} rank {rank} manifest differs")
        inputs = manifest.get("inputs")
        provenance = {
            field: manifest.get(field)
            for field in (
                "arm_identity_sha256",
                "study_identity_sha256",
                "stage_identity_sha256",
                "stage_outcome_identity_sha256",
                "git_commit",
                "training_git_commit",
                "frontier_selection_identity_sha256",
                "lockbox_registration_identity_sha256",
                "inference_code_compatibility",
                "videox_runtime",
            )
        }
        if not isinstance(inputs, Mapping):
            raise CausalityError(f"{label} rank {rank} lacks input evidence")
        normalized_inputs = dict(inputs)
        registration_input = inputs.get("lockbox_registration")
        if not isinstance(registration_input, Mapping):
            raise CausalityError(
                f"{label} rank {rank} lacks lockbox registration evidence"
            )
        rank0_reverified = registration_input.get(
            "rank0_deterministic_construction_reverified"
        )
        if rank0_reverified is not (rank == 0):
            raise CausalityError(
                f"{label} rank {rank} construction-verification label differs"
            )
        normalized_registration = dict(registration_input)
        normalized_registration.pop(
            "rank0_deterministic_construction_reverified", None
        )
        normalized_inputs["lockbox_registration"] = normalized_registration
        if common_inputs is None:
            common_inputs = normalized_inputs
            common_provenance = provenance
        elif normalized_inputs != common_inputs or provenance != common_provenance:
            raise CausalityError(f"{label} rank input/provenance evidence differs")
        rows.extend(rank_rows)
    assert common_inputs is not None and common_provenance is not None
    return {
        "path": str(inventory_path),
        "sha256": sha256_file(inventory_path),
        "identity_sha256": inventory["identity_sha256"],
        "payload": inventory,
        "rows": rows,
        "rank_inputs": common_inputs,
        "rank_provenance": common_provenance,
    }


def _valid_metric_map(metrics: Any) -> bool:
    return isinstance(metrics, Mapping) and all(
        not isinstance(metrics.get(metric), bool)
        and isinstance(metrics.get(metric), (int, float))
        and math.isfinite(float(metrics[metric]))
        and float(metrics[metric]) >= 0.0
        for metric in frontier.CLAIM_METRICS
    )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str],
    k: int,
    selection: Mapping[str, Any],
    evaluator_commit: str,
    sidecar_provenance: Mapping[str, Any] | None,
) -> dict[tuple[str, int], dict[str, Mapping[str, Any]]]:
    selection_id = str(selection["identity_sha256"])
    lockbox_id = str(selection["lockbox_registration"]["identity_sha256"])
    expected_compatibility = str(
        selection["inference_code_compatibility_sha256"]
    )
    expected_videox = str(selection["videox_runtime_identity_sha256"])
    expected_units: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if not identity_valid(row):
            raise CausalityError("causality quality row identity is invalid")
        source = row.get("source")
        unit = (str(row.get("clip_id", "")), int(row.get("clip_index", -1)))
        if (
            source not in sources
            or row.get("kind") != KIND_QUALITY_ROW
            or row.get("arm_code") != "J1"
            or row.get("completed_updates") != 1000
            or not unit[0]
            or not 0 <= unit[1] < EXPECTED_CLIPS
            or row.get("nfe") != k
            or row.get("evaluation_split") != "lockbox"
            or row.get("frontier_selection_identity_sha256") != selection_id
            or row.get("lockbox_registration_identity_sha256") != lockbox_id
            or row.get("study_identity_sha256")
            != selection.get("study_identity_sha256")
            or row.get("arm_identity_sha256")
            != selection.get("arm_identity_sha256", {}).get("J1")
            or row.get("stage_identity_sha256")
            != selection.get("stage_identity_sha256", {}).get("J1")
            or row.get("training_git_commit")
            != selection.get("training_git_commit")
            or row.get("evaluator_git_commit") != evaluator_commit
            or row.get("inference_code_compatibility_sha256")
            != expected_compatibility
            or row.get("videox_runtime_identity_sha256") != expected_videox
            or row.get("evaluation_world_size") != EXPECTED_WORLD_SIZE
            or row.get("evaluation_batch_size_per_rank")
            != EXPECTED_BATCH_SIZE_PER_RANK
            or row.get("sampling_namespace") != "lockbox"
            or row.get("sampling_id")
            != frontier.SAMPLE_ID_OFFSETS["lockbox"] + unit[1]
            or row.get("oracle_leakage") is not False
            or row.get("deployable_evidence") is not True
            or row.get("sampler_entrypoint")
            != "DualExplicitActionDiTModel.sample_future_deployable"
            or row.get("clean_future_or_auxiliary_passed_to_sampler") is not False
            or row.get("online_teacher_call_count") != 0
            or row.get("auxiliary_history_latent_frames") != 0
            or row.get("actual_wan_call_count") != k
            or isinstance(row.get("effective_state_gate"), bool)
            or not isinstance(row.get("effective_state_gate"), (int, float))
            or not math.isfinite(float(row["effective_state_gate"]))
            or isinstance(row.get("effective_clock_gate"), bool)
            or not isinstance(row.get("effective_clock_gate"), (int, float))
            or not math.isfinite(float(row["effective_clock_gate"]))
            or not _valid_metric_map(row.get("metrics"))
        ):
            raise CausalityError(
                f"{source!r} row violates within-J1 causality provenance"
            )
        hashes = row.get("tensor_sha256")
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != EXPECTED_TENSOR_HASH_FIELDS
            or any(
                not isinstance(value, str)
                or SHA256_RE.fullmatch(value) is None
                for value in hashes.values()
            )
            or hashes["auxiliary_initial_state_sha256"]
            != hashes["auxiliary_initial_noise_sha256"]
        ):
            raise CausalityError(f"{source} row tensor hashes are invalid")
        if sidecar_provenance is None:
            if row.get("frontier_causality_sidecar", False) is not False:
                raise CausalityError("autonomous row is mislabeled as sidecar")
        else:
            expected_sidecar = {
                "frontier_causality_sidecar": True,
                **dict(sidecar_provenance),
            }
            if any(row.get(key) != value for key, value in expected_sidecar.items()):
                raise CausalityError(
                    f"{source} row sidecar provenance differs"
                )
        bucket = expected_units.setdefault(unit, {})
        if source in bucket:
            raise CausalityError(f"duplicate {source} row for {unit}")
        bucket[str(source)] = row
    if len(expected_units) != EXPECTED_CLIPS:
        raise CausalityError(
            f"causality clip count {len(expected_units)} != {EXPECTED_CLIPS}"
        )
    required = set(sources)
    if any(set(by_source) != required for by_source in expected_units.values()):
        raise CausalityError("causality source grid is incomplete or contains extras")
    expected_indexes = list(range(EXPECTED_CLIPS))
    if sorted(index for _clip, index in expected_units) != expected_indexes:
        raise CausalityError("causality clip indexes are not dense and unique")
    return expected_units


def _cross_check_inventory_provenance(
    autonomous: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    sidecar_evaluator_commit: str,
) -> None:
    left_inputs = autonomous["rank_inputs"]
    right_inputs = sidecar["rank_inputs"]
    exact_input_keys = {
        "resolved_config",
        "snapshot",
        "arm_manifest",
        "study_manifest",
        "stage_manifest",
        "stage_outcome",
        "evaluation_clip_manifest",
        "evaluation_cache_metadata",
        "evaluation_cache_arrays",
        "lockbox_registration",
    }
    if (
        set(left_inputs) != set(right_inputs)
        or not exact_input_keys.issubset(left_inputs)
        or any(left_inputs[key] != right_inputs[key] for key in exact_input_keys)
    ):
        raise CausalityError(
            "autonomous/sidecar checkpoint, stage, lockbox, or cache inputs differ"
        )
    left = autonomous["rank_provenance"]
    right = sidecar["rank_provenance"]
    exact_fields = (
        "arm_identity_sha256",
        "study_identity_sha256",
        "stage_identity_sha256",
        "stage_outcome_identity_sha256",
        "git_commit",
        "training_git_commit",
        "frontier_selection_identity_sha256",
        "lockbox_registration_identity_sha256",
        "inference_code_compatibility",
        "videox_runtime",
    )
    if any(left[field] != right[field] for field in exact_fields):
        raise CausalityError("autonomous/sidecar rank provenance differs")
    if (
        left["training_git_commit"] != selection.get("training_git_commit")
        or autonomous["payload"].get("evaluator_git_commit")
        != selection.get("evaluator_git_commit")
        or sidecar["payload"].get("evaluator_git_commit")
        != sidecar_evaluator_commit
    ):
        raise CausalityError("inventory evaluator/training commits differ")


def validate_prerequisites(
    *,
    study_root: Path,
    selection_path: Path,
    continuation_path: Path,
    sidecar_evaluator_commit: str,
    current_inference_compatibility: Mapping[str, Any],
    require_autonomous_inventory: bool,
) -> dict[str, Any]:
    """Validate conditional eligibility and, when ready, J1 autonomous evidence."""

    study_root = _canonical_directory(study_root, "study root")
    selection_path = _canonical_file(selection_path, "frontier selection")
    if selection_path != study_root / "frontier_selection.json":
        raise CausalityError("selection must be the study frontier_selection.json")
    selection = _read_json(selection_path, "frontier selection")
    k = validate_selection(selection)
    continuation = validate_continuation(
        continuation_path,
        selection_path=selection_path,
        selection=selection,
    )
    if (
        COMMIT_RE.fullmatch(sidecar_evaluator_commit) is None
        or selection.get("training_git_commit") is None
        or selection.get("inference_code_compatibility_sha256")
        != hashlib.sha256(
            canonical_json(dict(current_inference_compatibility))
        ).hexdigest()
        or selection.get("lockbox_registration", {}).get(
            "inference_code_compatibility"
        )
        != dict(current_inference_compatibility)
    ):
        raise CausalityError(
            "sidecar evaluator does not preserve selected inference semantics"
        )
    inventory_path = autonomous_inventory_path(study_root, selection)
    autonomous = None
    if require_autonomous_inventory:
        autonomous = _load_inventory_rows(
            inventory_path,
            label="J1 autonomous lockbox",
            expected_grid=(("autonomous", k),),
            expected_evaluator_commit=str(selection["evaluator_git_commit"]),
            sidecar=False,
        )
    return {
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": sha256_file(selection_path),
        "selected_j1_nfe": k,
        "continuation": continuation,
        "autonomous_inventory_path": inventory_path,
        "autonomous": autonomous,
    }


def _comparison(
    left: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    reference: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
    *,
    reference_source: str,
    k: int,
) -> dict[str, Any]:
    units = tuple(sorted(left))
    if units != tuple(sorted(reference)):
        raise CausalityError("autonomous/control clip units differ")
    metrics = {}
    for metric in frontier.CLAIM_METRICS:
        metrics[metric] = frontier.paired_effect(
            [float(left[unit]["autonomous"]["metrics"][metric]) for unit in units],
            [
                float(reference[unit][reference_source]["metrics"][metric])
                for unit in units
            ],
            bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
            confidence=frontier.DEFAULT_CONFIDENCE,
            seed=frontier.DEFAULT_SEED,
            label=(
                f"fresh-lockbox-within-J1:J1-autonomous-{k}-vs-"
                f"J1-{reference_source}-{k}"
            ),
        )
    lows = {
        metric: float(effect["bootstrap_ci"]["low"])
        for metric, effect in metrics.items()
    }
    checks = {
        "temporal_relative_improvement_ci_low_strictly_positive": (
            lows[frontier.PRIMARY_METRIC] > 0.0
        ),
        "video_latent_nmse_relative_improvement_ci_low_above_minus_one_percent": (
            lows["video_future_nmse"] > -0.01
        ),
        "decoded_mse_relative_improvement_ci_low_above_minus_one_percent": (
            lows["decoded_mse_unit_range"] > -0.01
        ),
    }
    temporal_effect = float(
        metrics[frontier.PRIMARY_METRIC]["relative_improvement"]
    )
    return {
        "left": {"arm": "J1", "source": "autonomous", "nfe": k},
        "reference": {"arm": "J1", "source": reference_source, "nfe": k},
        "metrics": metrics,
        "ci_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "ci_lows": lows,
            "rule": (
                "temporal relative-improvement 95% paired-bootstrap CI-low > 0; "
                "video-latent NMSE and decoded-MSE CI-lows > -0.01"
            ),
        },
        "temporal_practical_materiality": {
            "threshold": PRACTICAL_TEMPORAL_THRESHOLD,
            "threshold_percent": PRACTICAL_TEMPORAL_THRESHOLD * 100.0,
            "relative_improvement": temporal_effect,
            "relative_improvement_percent": temporal_effect * 100.0,
            "at_least_three_percent": (
                temporal_effect >= PRACTICAL_TEMPORAL_THRESHOLD
            ),
            "role": (
                "reported practical-effect threshold; the preregistered "
                "statistical gate remains CI-low > 0"
            ),
        },
    }


def build_confirmation(
    *,
    selection: Mapping[str, Any],
    continuation: Mapping[str, Any],
    autonomous: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    sidecar_evaluator_commit: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the separate, non-frontier within-J1 causal confirmation."""

    k = validate_selection(selection)
    if (
        continuation.get("payload", {}).get("selection", {}).get(
            "identity_sha256"
        )
        != selection.get("identity_sha256")
        or continuation.get("payload", {}).get(
            "lockbox_jobs_created_only_after_confirmatory_eligibility"
        )
        is not True
    ):
        raise CausalityError("continuation is not eligible-selection evidence")
    _cross_check_inventory_provenance(
        autonomous,
        sidecar,
        selection=selection,
        sidecar_evaluator_commit=sidecar_evaluator_commit,
    )
    sidecar_provenance = {
        "frontier_primary_evaluator_git_commit": selection[
            "evaluator_git_commit"
        ],
        "frontier_continuation_sha256": continuation["sha256"],
        "frontier_continuation_lockbox_quality_array_job_id": continuation[
            "lockbox_quality_array_job_id"
        ],
        "frontier_autonomous_inventory_identity_sha256": autonomous[
            "identity_sha256"
        ],
        "frontier_autonomous_inventory_sha256": autonomous["sha256"],
    }
    autonomous_rows = _validate_rows(
        autonomous["rows"],
        sources=("autonomous",),
        k=k,
        selection=selection,
        evaluator_commit=str(selection["evaluator_git_commit"]),
        sidecar_provenance=None,
    )
    sidecar_rows = _validate_rows(
        sidecar["rows"],
        sources=SIDE_CAR_SOURCES,
        k=k,
        selection=selection,
        evaluator_commit=sidecar_evaluator_commit,
        sidecar_provenance=sidecar_provenance,
    )
    if tuple(sorted(autonomous_rows)) != tuple(sorted(sidecar_rows)):
        raise CausalityError("autonomous and sidecar clip units differ")
    for unit in sorted(autonomous_rows):
        reference = autonomous_rows[unit]["autonomous"]["tensor_sha256"]
        for source in SIDE_CAR_SOURCES:
            candidate_row = sidecar_rows[unit][source]
            candidate = candidate_row["tensor_sha256"]
            if any(
                candidate[field] != reference[field]
                for field in PAIRING_HASH_FIELDS
            ):
                raise CausalityError(
                    f"clean/raw/input/initial-noise pairing differs for {unit}, "
                    f"source={source}"
                )
            if any(
                candidate_row[field]
                != autonomous_rows[unit]["autonomous"][field]
                for field in ("effective_state_gate", "effective_clock_gate")
            ):
                raise CausalityError(
                    f"learned gate values differ for {unit}, source={source}"
                )
    comparisons = {
        source: _comparison(
            autonomous_rows,
            sidecar_rows,
            reference_source=source,
            k=k,
        )
        for source in SIDE_CAR_SOURCES
    }
    all_ci_gates = all(
        comparison["ci_gate"]["passed"] for comparison in comparisons.values()
    )
    all_material = all(
        comparison["temporal_practical_materiality"][
            "at_least_three_percent"
        ]
        for comparison in comparisons.values()
    )
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_CONFIRMATION,
            "created_at_utc": created_at_utc or _now(),
            "evaluation_split": "lockbox",
            "study_identity_sha256": selection["study_identity_sha256"],
            "training_git_commit": selection["training_git_commit"],
            "primary_frontier_evaluator_git_commit": selection[
                "evaluator_git_commit"
            ],
            "sidecar_evaluator_git_commit": sidecar_evaluator_commit,
            "selection_identity_sha256": selection["identity_sha256"],
            "lockbox_registration_identity_sha256": selection[
                "lockbox_registration"
            ]["identity_sha256"],
            "selected_j1_nfe": k,
            "expected_clip_count": EXPECTED_CLIPS,
            "observed_clip_count": len(autonomous_rows),
            "bootstrap": {
                "samples": frontier.DEFAULT_BOOTSTRAP_SAMPLES,
                "confidence": frontier.DEFAULT_CONFIDENCE,
                "seed": frontier.DEFAULT_SEED,
                "unit": "paired immutable clip_id/clip_index",
            },
            "sources": {
                "left": "autonomous generated V-JEPA state",
                "controls": list(SIDE_CAR_SOURCES),
                "all_history_only_deployable": True,
                "online_teacher_calls": 0,
                "clean_future_or_auxiliary_passed_to_sampler": False,
            },
            "pairing_checks": {
                "same_128_clip_units": True,
                "same_raw_and_vae_targets": True,
                "same_cached_rgb_and_actions": True,
                "same_checkpoint_and_final_stage": True,
                "same_selection_and_lockbox": True,
                "same_initial_video_and_auxiliary_noise": True,
                "same_nfe_and_exact_wan_calls": True,
            },
            "comparisons": comparisons,
            "both_preregistered_ci_gates_passed": all_ci_gates,
            "both_temporal_effects_at_least_three_percent": all_material,
            "within_j1_generated_state_causality_supported": all_ci_gates,
            "within_j1_generated_state_causality_practically_material": (
                all_ci_gates and all_material
            ),
            "frontier_isolation": {
                "selection_decisions_from_sidecar": 0,
                "main_lockbox_confirmation_modified": False,
                "main_timing_modified": False,
                "main_final_report_modified": False,
                "result_is_a_separate_causality_claim": True,
            },
            "input_evidence": {
                "selection": {
                    "path": continuation["payload"]["selection"]["path"],
                    "sha256": continuation["payload"]["selection"]["sha256"],
                    "identity_sha256": selection["identity_sha256"],
                },
                "continuation": {
                    "path": continuation["path"],
                    "sha256": continuation["sha256"],
                },
                "autonomous_inventory": {
                    key: autonomous[key]
                    for key in ("path", "sha256", "identity_sha256")
                },
                "sidecar_inventory": {
                    key: sidecar[key]
                    for key in ("path", "sha256", "identity_sha256")
                },
            },
        }
    )


def command_confirm(args: argparse.Namespace) -> int:
    repo = _canonical_directory(args.repo_root, "sidecar evaluator repository")
    if (
        _git(repo, "rev-parse", "HEAD") != args.sidecar_evaluator_commit
        or _git(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise CausalityError(
            "sidecar confirmation repository is not clean/frozen"
        )
    try:
        compatibility = frontier.git_inference_compatibility(
            repo,
            training_commit=args.training_commit,
            tool_commit=args.sidecar_evaluator_commit,
        )
    except frontier.FrontierError as exc:
        raise CausalityError(str(exc)) from exc
    selection_path = _canonical_file(args.selection, "frontier selection")
    selection = _read_json(selection_path, "frontier selection")
    if (
        selection.get("training_git_commit") != args.training_commit
        or selection.get("inference_code_compatibility_sha256")
        != hashlib.sha256(canonical_json(compatibility)).hexdigest()
    ):
        raise CausalityError(
            "confirmation evaluator/training inference provenance differs"
        )
    k = validate_selection(selection)
    continuation = validate_continuation(
        args.continuation,
        selection_path=selection_path,
        selection=selection,
    )
    autonomous = _load_inventory_rows(
        args.autonomous_inventory,
        label="J1 autonomous lockbox",
        expected_grid=(("autonomous", k),),
        expected_evaluator_commit=str(selection["evaluator_git_commit"]),
        sidecar=False,
    )
    sidecar = _load_inventory_rows(
        args.sidecar_inventory,
        label="J1 causality sidecar",
        expected_grid=tuple((source, k) for source in SIDE_CAR_SOURCES),
        expected_evaluator_commit=args.sidecar_evaluator_commit,
        sidecar=True,
    )
    result = build_confirmation(
        selection=selection,
        continuation=continuation,
        autonomous=autonomous,
        sidecar=sidecar,
        sidecar_evaluator_commit=args.sidecar_evaluator_commit,
    )
    _exclusive_json(Path(args.output).expanduser().absolute(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise CausalityError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def command_values(args: argparse.Namespace) -> int:
    """Emit fail-closed, line-safe paths/identities for Slurm entrypoints."""

    repo = _canonical_directory(args.repo_root, "evaluator repository")
    if (
        COMMIT_RE.fullmatch(args.sidecar_evaluator_commit) is None
        or _git(repo, "rev-parse", "HEAD") != args.sidecar_evaluator_commit
        or _git(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise CausalityError("sidecar evaluator repository is not clean/frozen")
    try:
        compatibility = frontier.git_inference_compatibility(
            repo,
            training_commit=args.training_commit,
            tool_commit=args.sidecar_evaluator_commit,
        )
    except frontier.FrontierError as exc:
        raise CausalityError(str(exc)) from exc
    prerequisites = validate_prerequisites(
        study_root=Path(args.study_root),
        selection_path=Path(args.selection),
        continuation_path=Path(args.continuation),
        sidecar_evaluator_commit=args.sidecar_evaluator_commit,
        current_inference_compatibility=compatibility,
        require_autonomous_inventory=args.require_autonomous_inventory,
    )
    selection = prerequisites["selection"]
    run_dir = _canonical_directory(
        Path(args.study_root) / "j1_joint_auxiliary_leads", "J1 run directory"
    )
    sidecar_dir = sidecar_output_directory(run_dir, selection)
    confirmation = (
        Path(args.study_root) / "frontier_lockbox_j1_causality_confirmation.json"
    ).absolute()
    values = (
        str(prerequisites["selected_j1_nfe"]),
        str(selection["identity_sha256"]),
        str(selection["lockbox_registration"]["identity_sha256"]),
        str(prerequisites["autonomous_inventory_path"]),
        str(sidecar_dir),
        str(sidecar_dir / "inventory.json"),
        str(confirmation),
        prerequisites["continuation"]["j1_array_task_dependency"],
        prerequisites["continuation"]["sha256"],
        str(selection["evaluator_git_commit"]),
    )
    if any(not value or "\n" in value or "\r" in value for value in values):
        raise CausalityError("sidecar value is missing or not line-safe")
    print("\n".join(values))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    confirm = subparsers.add_parser(
        "confirm", help="confirm autonomous versus off/shuffled on lockbox"
    )
    confirm.add_argument("--repo-root", required=True)
    confirm.add_argument("--training-commit", required=True)
    confirm.add_argument("--selection", required=True)
    confirm.add_argument("--continuation", required=True)
    confirm.add_argument("--autonomous-inventory", required=True)
    confirm.add_argument("--sidecar-inventory", required=True)
    confirm.add_argument("--sidecar-evaluator-commit", required=True)
    confirm.add_argument("--output", required=True)
    confirm.set_defaults(func=command_confirm)
    values = subparsers.add_parser(
        "values", help="emit validated paths and identities for Slurm"
    )
    values.add_argument("--repo-root", required=True)
    values.add_argument("--study-root", required=True)
    values.add_argument("--training-commit", required=True)
    values.add_argument("--sidecar-evaluator-commit", required=True)
    values.add_argument("--selection", required=True)
    values.add_argument("--continuation", required=True)
    values.add_argument(
        "--require-autonomous-inventory", action="store_true"
    )
    values.set_defaults(func=command_values)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (CausalityError, frontier.FrontierError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
