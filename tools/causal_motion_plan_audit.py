#!/usr/bin/env python3
"""Fail-closed collation audit for prospective CAMP validation rows.

This tool does not evaluate videos or compute scientific results.  It verifies
that independently written evaluation shards implement the frozen causal
motion-plan endpoint grid, bind all paired tensors, account for every model
call and synchronized stage, and contain no clean-future or teacher input.  A
successful audit is written once with ``O_EXCL`` and can therefore be bound by
the later statistical analysis without silently replacing raw evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ROW_KIND = "causal_motion_plan_validation_clip"
AUDIT_KIND = "causal_motion_plan_evaluation_audit"
ARMS = ("PLAN-OFF", "PLAN-ON")
NFE_GRID = (1, 2, 4)
BASE_CONDITION_SOURCES = ("aligned", "off", "shuffled")
EXPECTED_VALIDATION_CLIPS = 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CausalMotionPlanAuditError(RuntimeError):
    """A row, endpoint, pairing invariant, or immutable output is invalid."""


@dataclass(frozen=True)
class Endpoint:
    condition_source: str
    nfe: int
    primary_gate: bool

    @property
    def code(self) -> str:
        return f"{self.condition_source}_nfe_{self.nfe}"


ENDPOINTS = tuple(
    Endpoint(source, nfe, nfe == 1)
    for nfe in NFE_GRID
    for source in BASE_CONDITION_SOURCES
) + (Endpoint("action_shuffled", 1, False),)
ENDPOINT_BY_KEY = {
    (endpoint.condition_source, endpoint.nfe): endpoint for endpoint in ENDPOINTS
}

TENSOR_HASH_KEYS = (
    "cached_rgb_sha256",
    "local_actions_sha256",
    "planner_actions_sha256",
    "history_latent_sha256",
    "video_noise_sha256",
    "plan_noise_sha256",
    "generated_plan_sha256",
    "injected_plan_sha256",
    "final_latent_sha256",
    "decode_sha256",
)
PAIR_INPUT_HASH_KEYS = (
    "cached_rgb_sha256",
    "local_actions_sha256",
    "history_latent_sha256",
    "video_noise_sha256",
    "plan_noise_sha256",
)
PRE_WAN_HASH_KEYS = PAIR_INPUT_HASH_KEYS + (
    "planner_actions_sha256",
    "generated_plan_sha256",
    "injected_plan_sha256",
)
OUTPUT_HASH_KEYS = ("final_latent_sha256", "decode_sha256")
MODEL_IDENTITY_KEYS = (
    "parameter_schema_sha256",
    "planner_checkpoint_sha256",
    "motion_plan_stats_sha256",
)
LATENCY_KEYS = (
    "history_encode",
    "planner",
    "wan",
    "decode",
    "end_to_end",
)
REQUIRED_METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CausalMotionPlanAuditError("payload is not canonical finite JSON") from exc


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    if "identity_sha256" in unsigned:
        raise CausalMotionPlanAuditError("identity field is already present")
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    if not isinstance(identity, str) or SHA256_RE.fullmatch(identity) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    try:
        expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    except CausalMotionPlanAuditError:
        return False
    return identity == expected


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_json_line(line: str, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        parsed = json.loads(line, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CausalMotionPlanAuditError(f"invalid JSON in {label}") from exc
    if not isinstance(parsed, Mapping):
        raise CausalMotionPlanAuditError(f"row in {label} is not an object")
    return parsed


def read_jsonl(paths: Sequence[Path]) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Read regular immutable-looking shards and bind their exact bytes."""

    if not paths:
        raise CausalMotionPlanAuditError("at least one row shard is required")
    rows: list[Mapping[str, Any]] = []
    records: list[dict[str, Any]] = []
    resolved_paths: set[Path] = set()
    for supplied in paths:
        path = supplied.expanduser()
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise CausalMotionPlanAuditError(
                f"row shard must be a regular absolute file: {path}"
            )
        resolved = path.resolve(strict=True)
        if resolved in resolved_paths:
            raise CausalMotionPlanAuditError(f"duplicate row shard: {resolved}")
        resolved_paths.add(resolved)
        content = resolved.read_bytes()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CausalMotionPlanAuditError(f"row shard is not UTF-8: {resolved}") from exc
        shard_rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise CausalMotionPlanAuditError(
                    f"blank JSONL line at {resolved}:{line_number}"
                )
            shard_rows.append(
                _parse_json_line(line, label=f"{resolved}:{line_number}")
            )
        if not shard_rows:
            raise CausalMotionPlanAuditError(f"row shard is empty: {resolved}")
        rows.extend(shard_rows)
        records.append(
            {
                "path": str(resolved),
                "bytes": len(content),
                "rows": len(shard_rows),
                "sha256": _sha256_bytes(content),
            }
        )
    return rows, records


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _exact_int(value: Any, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _endpoint_from_row(row: Mapping[str, Any]) -> Endpoint | None:
    value = row.get("endpoint")
    if not isinstance(value, Mapping):
        return None
    source = value.get("condition_source")
    nfe = value.get("nfe")
    if isinstance(nfe, bool) or not isinstance(nfe, int):
        return None
    endpoint = ENDPOINT_BY_KEY.get((source, nfe))
    if (
        endpoint is None
        or set(value) != {"condition_source", "nfe", "primary_gate"}
        or value.get("primary_gate") is not endpoint.primary_gate
    ):
        return None
    return endpoint


def _validate_single_row(
    row: Mapping[str, Any], *, expected_clips: int
) -> tuple[str, int, Endpoint]:
    endpoint = _endpoint_from_row(row)
    arm = row.get("arm")
    clip_index = row.get("clip_index")
    hashes = row.get("tensor_sha256")
    model_identity = row.get("model_identity")
    calls = row.get("call_counts")
    latencies = row.get("latency_seconds")
    metrics = row.get("metrics")
    if (
        not identity_valid(row)
        or not _exact_int(row.get("schema_version"), SCHEMA_VERSION)
        or row.get("kind") != ROW_KIND
        or not isinstance(arm, str)
        or arm not in ARMS
        or isinstance(clip_index, bool)
        or not isinstance(clip_index, int)
        or not 0 <= clip_index < expected_clips
        or not isinstance(row.get("clip_id"), str)
        or not row.get("clip_id")
        or endpoint is None
        or row.get("evaluation_split") != "validation"
        or row.get("protected_test_accessed") is not False
        or row.get("clean_future_rgb_passed_to_sampler") is not False
        or row.get("clean_future_video_latent_passed_to_sampler") is not False
        or row.get("clean_future_feature_passed_to_sampler") is not False
        or row.get("clean_plan_passed_to_sampler") is not False
        or not _exact_int(row.get("teacher_call_count"), 0)
        or row.get("scoring_constructed_after_all_sampling") is not True
        or row.get("runtime_plan_fusion_enabled")
        is not (arm == "PLAN-ON" and endpoint.condition_source != "off")
        or not isinstance(hashes, Mapping)
        or any(
            not isinstance(hashes.get(key), str)
            or SHA256_RE.fullmatch(str(hashes.get(key))) is None
            for key in TENSOR_HASH_KEYS
        )
        or any(
            not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
            for key, value in hashes.items()
        )
        or not isinstance(model_identity, Mapping)
        or set(model_identity) != set(MODEL_IDENTITY_KEYS)
        or any(
            not isinstance(model_identity.get(key), str)
            or SHA256_RE.fullmatch(str(model_identity.get(key))) is None
            for key in MODEL_IDENTITY_KEYS
        )
        or not isinstance(calls, Mapping)
        or set(calls) != {"planner", "wan"}
        or not _exact_int(calls.get("planner"), 2)
        or not _exact_int(calls.get("wan"), endpoint.nfe)
        or row.get("latency_device_synchronized") is not True
        or not isinstance(latencies, Mapping)
        or set(latencies) != set(LATENCY_KEYS)
        or any(not _finite_nonnegative(latencies.get(key)) for key in LATENCY_KEYS)
        or not isinstance(metrics, Mapping)
        or any(not _finite_nonnegative(metrics.get(key)) for key in REQUIRED_METRICS)
    ):
        raise CausalMotionPlanAuditError("evaluation row violates the CAMP schema")
    stage_total = sum(float(latencies[key]) for key in LATENCY_KEYS[:-1])
    if float(latencies["end_to_end"]) + 1e-12 < stage_total:
        raise CausalMotionPlanAuditError(
            "end-to-end latency is smaller than synchronized stage latency"
        )
    planner_donor = row.get("planner_action_donor_clip_index")
    injected_donor = row.get("injected_plan_donor_clip_index")
    for label, donor in (
        ("planner action", planner_donor),
        ("injected plan", injected_donor),
    ):
        if (
            isinstance(donor, bool)
            or not isinstance(donor, int)
            or not 0 <= donor < expected_clips
        ):
            raise CausalMotionPlanAuditError(f"invalid {label} donor index")
    return str(arm), int(clip_index), endpoint


def _require_permutation(
    index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    *,
    arm: str,
    endpoint: Endpoint,
    donor_field: str,
    expected_clips: int,
) -> None:
    donors = [
        index[(arm, clip_index, endpoint.condition_source, endpoint.nfe)][donor_field]
        for clip_index in range(expected_clips)
    ]
    if sorted(donors) != list(range(expected_clips)) or any(
        clip_index == donor for clip_index, donor in enumerate(donors)
    ):
        raise CausalMotionPlanAuditError(
            f"{endpoint.code} {donor_field} is not a global derangement"
        )


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_clips: int = EXPECTED_VALIDATION_CLIPS,
    source_files: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate and content-bind a complete two-arm prospective row grid."""

    if (
        isinstance(expected_clips, bool)
        or not isinstance(expected_clips, int)
        or expected_clips < 2
    ):
        raise CausalMotionPlanAuditError("expected clip count must be at least two")
    normalized_source_files: list[dict[str, Any]] = []
    for record in source_files:
        path = record.get("path") if isinstance(record, Mapping) else None
        size = record.get("bytes") if isinstance(record, Mapping) else None
        count = record.get("rows") if isinstance(record, Mapping) else None
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "rows", "sha256"}
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise CausalMotionPlanAuditError("source file record violates schema")
        normalized_source_files.append(dict(record))
    if normalized_source_files and sum(
        record["rows"] for record in normalized_source_files
    ) != len(rows):
        raise CausalMotionPlanAuditError("source file row counts differ from collation")
    expected_inventory = {
        (arm, clip_index, endpoint.condition_source, endpoint.nfe)
        for arm in ARMS
        for clip_index in range(expected_clips)
        for endpoint in ENDPOINTS
    }
    indexed: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        arm, clip_index, endpoint = _validate_single_row(
            row, expected_clips=expected_clips
        )
        key = (arm, clip_index, endpoint.condition_source, endpoint.nfe)
        if key in indexed:
            raise CausalMotionPlanAuditError("duplicate CAMP endpoint row")
        indexed[key] = row
    if set(indexed) != expected_inventory:
        missing = len(expected_inventory - set(indexed))
        unexpected = len(set(indexed) - expected_inventory)
        raise CausalMotionPlanAuditError(
            f"endpoint inventory differs (missing={missing}, unexpected={unexpected})"
        )

    shared_identities = {
        tuple(row["model_identity"][key] for key in MODEL_IDENTITY_KEYS)
        for row in indexed.values()
    }
    if len(shared_identities) != 1:
        raise CausalMotionPlanAuditError(
            "parameter schema, planner checkpoint, or normalization stats differ"
        )

    for clip_index in range(expected_clips):
        clip_ids = {
            indexed[(arm, clip_index, endpoint.condition_source, endpoint.nfe)][
                "clip_id"
            ]
            for arm in ARMS
            for endpoint in ENDPOINTS
        }
        if len(clip_ids) != 1:
            raise CausalMotionPlanAuditError(
                f"clip identity changed for clip index {clip_index}"
            )
        for arm in ARMS:
            arm_rows = [
                indexed[(arm, clip_index, endpoint.condition_source, endpoint.nfe)]
                for endpoint in ENDPOINTS
            ]
            reference_hashes = arm_rows[0]["tensor_sha256"]
            if any(
                row["tensor_sha256"][field] != reference_hashes[field]
                for row in arm_rows[1:]
                for field in PAIR_INPUT_HASH_KEYS
            ):
                raise CausalMotionPlanAuditError(
                    f"paired RGB/action/history/noise changed within {arm} clip {clip_index}"
                )

            base_rows = [
                indexed[(arm, clip_index, source, nfe)]
                for nfe in NFE_GRID
                for source in BASE_CONDITION_SOURCES
            ]
            local_action_hash = reference_hashes["local_actions_sha256"]
            if any(
                row["planner_action_donor_clip_index"] != clip_index
                or row["tensor_sha256"]["planner_actions_sha256"]
                != local_action_hash
                for row in base_rows
            ):
                raise CausalMotionPlanAuditError(
                    f"aligned/off/shuffled planner actions are not local for {arm} clip {clip_index}"
                )
            generated_hash = base_rows[0]["tensor_sha256"]["generated_plan_sha256"]
            if any(
                row["tensor_sha256"]["generated_plan_sha256"] != generated_hash
                for row in base_rows[1:]
            ):
                raise CausalMotionPlanAuditError(
                    f"two-call local plan changed across endpoints for {arm} clip {clip_index}"
                )
            for nfe in NFE_GRID:
                for source in ("aligned", "off"):
                    row = indexed[(arm, clip_index, source, nfe)]
                    if (
                        row["injected_plan_donor_clip_index"] != clip_index
                        or row["tensor_sha256"]["injected_plan_sha256"]
                        != row["tensor_sha256"]["generated_plan_sha256"]
                    ):
                        raise CausalMotionPlanAuditError(
                            f"{source} plan is not local for {arm} clip {clip_index}"
                        )
            action_row = indexed[(arm, clip_index, "action_shuffled", 1)]
            if (
                action_row["injected_plan_donor_clip_index"] != clip_index
                or action_row["tensor_sha256"]["injected_plan_sha256"]
                != action_row["tensor_sha256"]["generated_plan_sha256"]
            ):
                raise CausalMotionPlanAuditError(
                    f"action-shuffled generated plan was not injected locally for {arm}"
                )

    for arm in ARMS:
        for nfe in NFE_GRID:
            endpoint = ENDPOINT_BY_KEY[("shuffled", nfe)]
            _require_permutation(
                indexed,
                arm=arm,
                endpoint=endpoint,
                donor_field="injected_plan_donor_clip_index",
                expected_clips=expected_clips,
            )
            for clip_index in range(expected_clips):
                row = indexed[(arm, clip_index, "shuffled", nfe)]
                donor = row["injected_plan_donor_clip_index"]
                donor_row = indexed[(arm, donor, "shuffled", nfe)]
                if (
                    row["tensor_sha256"]["injected_plan_sha256"]
                    != donor_row["tensor_sha256"]["generated_plan_sha256"]
                ):
                    raise CausalMotionPlanAuditError(
                        "shuffled injected plan hash differs from its declared donor"
                    )
        action_endpoint = ENDPOINT_BY_KEY[("action_shuffled", 1)]
        _require_permutation(
            indexed,
            arm=arm,
            endpoint=action_endpoint,
            donor_field="planner_action_donor_clip_index",
            expected_clips=expected_clips,
        )
        for clip_index in range(expected_clips):
            row = indexed[(arm, clip_index, "action_shuffled", 1)]
            donor = row["planner_action_donor_clip_index"]
            donor_row = indexed[(arm, donor, "aligned", 1)]
            if (
                row["tensor_sha256"]["planner_actions_sha256"]
                != donor_row["tensor_sha256"]["local_actions_sha256"]
            ):
                raise CausalMotionPlanAuditError(
                    "action-shuffled planner actions differ from their declared donor"
                )

    for clip_index in range(expected_clips):
        for endpoint in ENDPOINTS:
            off_arm = indexed[
                ("PLAN-OFF", clip_index, endpoint.condition_source, endpoint.nfe)
            ]
            on_arm = indexed[
                ("PLAN-ON", clip_index, endpoint.condition_source, endpoint.nfe)
            ]
            if (
                off_arm["planner_action_donor_clip_index"]
                != on_arm["planner_action_donor_clip_index"]
                or off_arm["injected_plan_donor_clip_index"]
                != on_arm["injected_plan_donor_clip_index"]
                or any(
                    off_arm["tensor_sha256"][field]
                    != on_arm["tensor_sha256"][field]
                    for field in PRE_WAN_HASH_KEYS
                )
            ):
                raise CausalMotionPlanAuditError(
                    f"paired pre-Wan evidence differs across arms for clip {clip_index} "
                    f"endpoint {endpoint.code}"
                )

        for nfe in NFE_GRID:
            plan_off_controls = [
                indexed[("PLAN-OFF", clip_index, source, nfe)]
                for source in BASE_CONDITION_SOURCES
            ]
            reference = plan_off_controls[0]["tensor_sha256"]
            if any(
                row["tensor_sha256"][field] != reference[field]
                for row in plan_off_controls[1:]
                for field in OUTPUT_HASH_KEYS
            ):
                raise CausalMotionPlanAuditError(
                    f"PLAN-OFF is not bit-identical across controls at NFE {nfe}"
                )

    ordered_rows = [indexed[key] for key in sorted(indexed)]
    row_stream = b"".join(_canonical_json(row) + b"\n" for row in ordered_rows)
    identity_values = next(iter(shared_identities))
    counts = Counter(
        f"{arm}/{source}/nfe_{nfe}" for arm, _, source, nfe in indexed
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": "passed",
        "expected_validation_clips": expected_clips,
        "protected_test_accessed": False,
        "clean_future_or_teacher_sampler_inputs": 0,
        "arms": list(ARMS),
        "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
        "primary_gate_endpoints": [
            endpoint.code for endpoint in ENDPOINTS if endpoint.primary_gate
        ],
        "descriptive_endpoints": [
            endpoint.code for endpoint in ENDPOINTS if not endpoint.primary_gate
        ],
        "model_identity": dict(zip(MODEL_IDENTITY_KEYS, identity_values)),
        "rows": {
            "count": len(ordered_rows),
            "canonical_sha256": _sha256_bytes(row_stream),
            "counts_by_arm_endpoint": dict(sorted(counts.items())),
            "source_files": normalized_source_files,
        },
        "verified": {
            "exact_endpoint_inventory": True,
            "nfe1_only_primary_gate": True,
            "planner_calls_exactly_two": True,
            "wan_calls_equal_declared_nfe": True,
            "synchronized_stage_latencies_complete": True,
            "zero_clean_future_or_teacher_inputs": True,
            "paired_tensor_hashes_across_arms": True,
            "global_plan_and_action_derangements": True,
            "plan_off_control_outputs_bit_identical": True,
            "shared_parameter_planner_stats_identity": True,
        },
    }
    return identity_payload(payload)


def exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one canonical JSON receipt without replacing any existing path."""

    output = path.expanduser()
    parent = output.parent
    if not output.is_absolute() or not parent.is_dir() or parent.is_symlink():
        raise CausalMotionPlanAuditError(
            "audit output must be an absolute path in an existing regular directory"
        )
    content = _canonical_json(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(output, flags, 0o440)
    except FileExistsError as exc:
        raise CausalMotionPlanAuditError(f"audit output already exists: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Deliberately leave a partial exclusive path rather than make a failed
        # attempt look reusable.  A new run must choose a fresh output root.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        action="append",
        required=True,
        type=Path,
        help="absolute path to a prospective evaluation JSONL shard; repeatable",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="fresh absolute path for the immutable audit JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows, source_files = read_jsonl(args.rows)
    audit = audit_rows(rows, source_files=source_files)
    exclusive_write_json(args.output, audit)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "identity_sha256": audit["identity_sha256"],
                "rows": audit["rows"]["count"],
                "status": audit["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
