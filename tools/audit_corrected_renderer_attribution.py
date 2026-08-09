#!/usr/bin/env python3
"""Read-only hardened audit for a completed renderer-attribution artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.corrected_renderer_attribution import (
    ARMS,
    BOOTSTRAP_SAMPLES,
    GATE_FAVORABLE_CLIP_FRACTION,
    MANDATORY_METRICS,
    PRIMARY_ALIGNMENT_MIN_RELATIVE_IMPROVEMENT_PERCENT,
    SCORE_CLIPS,
    canonical_identity,
    load_bundles,
    paired_difference,
    read_json,
    read_jsonl,
    require_false_flags,
    seal,
    sha256_file,
    write_json,
)


class AuditError(RuntimeError):
    """Raised when a sealed renderer artifact fails independent checks."""


def require_identity(payload: dict[str, Any], location: str) -> None:
    if payload.get("identity_sha256") != canonical_identity(payload):
        raise AuditError(f"identity mismatch: {location}")


def resolve_record(
    output: Path,
    name: str,
    record: dict[str, Any],
    registered_source: Path,
) -> Path:
    candidates = [Path(record["path"])]
    if name == "source":
        candidates.append(registered_source)
    candidates.extend((output / Path(record["path"]).name, output / name))
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == record["sha256"]:
            return candidate.resolve()
    raise AuditError(
        f"cannot resolve {name} with expected hash {record['sha256']}: "
        f"{[str(path) for path in candidates]}"
    )


def verify_artifact_section(
    output: Path,
    section: dict[str, Any],
    registered_source: Path,
) -> int:
    checked = 0
    for name, record in section.items():
        if name == "overlays":
            for item in record:
                resolve_record(output / "overlays", "overlay", item["file"], registered_source)
                checked += 1
            continue
        resolve_record(output, name, record, registered_source)
        checked += 1
    return checked


def recompute_effects_and_gates(
    clip_rows: list[dict[str, Any]],
    clip_order: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    if set(clip_order) != {str(row["clip_id"]) for row in clip_rows}:
        raise AuditError("registered bundle order does not cover clip rows")
    by_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        indexed = {row["clip_id"]: row for row in clip_rows if row["arm"] == arm}
        if set(indexed) != set(clip_order):
            raise AuditError(f"clip coverage differs for arm {arm}")
        names = indexed[clip_order[0]]["metrics"].keys()
        by_arm[arm] = {
            name: np.asarray([indexed[clip]["metrics"][name] for clip in clip_order])
            for name in names
        }

    effects: dict[str, dict[str, Any]] = {}
    offset = 0
    for reference in ("raw_command", "episode_shuffled"):
        for metric in MANDATORY_METRICS:
            offset += 1
            label = f"predicted_corrected_vs_{reference}:{metric}"
            effects[label] = paired_difference(
                by_arm[reference][metric],
                by_arm["predicted_corrected"][metric],
                seed_offset=offset,
            )
        offset += 1
        label = f"predicted_corrected_vs_{reference}:silhouette_iou"
        effects[label] = paired_difference(
            by_arm[reference]["silhouette_iou"],
            by_arm["predicted_corrected"]["silhouette_iou"],
            seed_offset=offset,
            higher_is_better=True,
        )

    gates: dict[str, bool] = {}
    for label, effect in effects.items():
        if label.endswith(":silhouette_iou"):
            gates[label] = bool(
                effect["paired_favorable_difference_mean"] > 0.0
                and effect["paired_bootstrap_95_ci"][0] >= 0.0
            )
            continue
        passed = bool(
            effect["paired_favorable_difference_mean"] > 0.0
            and effect["paired_bootstrap_95_ci"][0] > 0.0
            and effect["favorable_clip_fraction"] >= GATE_FAVORABLE_CLIP_FRACTION
        )
        if label.endswith(":silhouette_boundary_chamfer_px"):
            passed = bool(
                passed
                and effect["relative_improvement_percent"]
                >= PRIMARY_ALIGNMENT_MIN_RELATIVE_IMPROVEMENT_PERCENT
            )
        gates[label] = passed
    gates["all_passed"] = all(gates.values())
    return effects, gates


def exact_json_equal(left: Any, right: Any, location: str) -> None:
    left_json = json.dumps(left, sort_keys=True, separators=(",", ":"))
    right_json = json.dumps(right, sort_keys=True, separators=(",", ":"))
    if left_json != right_json:
        raise AuditError(f"recomputed value differs: {location}")


def audit(output: Path, registered_source: Path) -> dict[str, Any]:
    output = output.resolve()
    registered_source = registered_source.resolve()
    names = (
        "registration.json",
        "preparation.json",
        "preparation_complete.json",
        "analysis.json",
        "run_complete.json",
    )
    documents = {name: read_json(output / name) for name in names}
    for name, document in documents.items():
        require_identity(document, name)

    registration = documents["registration.json"]
    preparation = documents["preparation.json"]
    prep_complete = documents["preparation_complete.json"]
    analysis = documents["analysis.json"]
    complete = documents["run_complete.json"]
    if sha256_file(registered_source) != registration["source"]["sha256"]:
        raise AuditError("registered execution source hash differs")
    bindings = (
        (
            preparation["registration_identity_sha256"],
            registration["identity_sha256"],
            "preparation->registration",
        ),
        (
            prep_complete["preparation_identity_sha256"],
            preparation["identity_sha256"],
            "preparation-complete->preparation",
        ),
        (
            analysis["registration_identity_sha256"],
            registration["identity_sha256"],
            "analysis->registration",
        ),
        (
            analysis["preparation_identity_sha256"],
            preparation["identity_sha256"],
            "analysis->preparation",
        ),
        (
            complete["analysis_identity_sha256"],
            analysis["identity_sha256"],
            "completion->analysis",
        ),
    )
    for observed, expected, location in bindings:
        if observed != expected:
            raise AuditError(f"identity binding differs: {location}")
    if complete["decision"] != analysis["decision"]:
        raise AuditError("completion decision differs from analysis")

    artifact_hashes = sum(
        verify_artifact_section(output, document["artifacts"], registered_source)
        for document in (preparation, prep_complete, analysis, complete)
    )
    bundles = load_bundles(output, preparation)
    frame_rows = read_jsonl(output / "frame_transition_metrics.jsonl")
    clip_rows = read_jsonl(output / "clip_metrics.jsonl")
    timing_rows = read_jsonl(output / "nonwrapping_timing_metrics.jsonl")
    provenance_rows = read_jsonl(output / "input_provenance.jsonl")
    expected_counts = {
        "bundles": SCORE_CLIPS,
        "frame_rows": SCORE_CLIPS * 8 * len(ARMS),
        "clip_rows": SCORE_CLIPS * len(ARMS),
        "timing_rows": SCORE_CLIPS * 2 * 7 * len(ARMS),
        "provenance_rows": 384 + SCORE_CLIPS,
    }
    observed_counts = {
        "bundles": len(bundles),
        "frame_rows": len(frame_rows),
        "clip_rows": len(clip_rows),
        "timing_rows": len(timing_rows),
        "provenance_rows": len(provenance_rows),
    }
    if observed_counts != expected_counts:
        raise AuditError(f"row counts differ: {observed_counts} != {expected_counts}")

    selected = registration["selection"]["selected"]
    if len(selected) != SCORE_CLIPS:
        raise AuditError("selected count differs")
    selected_indices = {int(item["manifest_index"]) for item in selected}
    if len(selected_indices) != SCORE_CLIPS:
        raise AuditError("selected indices are not unique")
    for item in selected:
        if not 384 <= int(item["manifest_index"]) < 512:
            raise AuditError("score index is outside frozen pool")
        if item["donor_manifest_index"] == item["manifest_index"]:
            raise AuditError("donor is recipient")
        if item["donor_manifest_index"] not in selected_indices:
            raise AuditError("donor is outside score clips")
        if item["donor_episode_dir"] == item["episode_dir"]:
            raise AuditError("donor episode equals recipient")

    for row in frame_rows:
        metrics = row["metrics"]
        if metrics["oracle_source_reprojection_rmse_px"] > 1e-5:
            raise AuditError("source reprojection check failed")
    for row in clip_rows:
        if row["metrics"]["oracle_moving_flow_sample_count"] < 10:
            raise AuditError("clip-total moving flow support is below ten")
    recomputed_effects, recomputed_gates = recompute_effects_and_gates(
        clip_rows, [bundle.metadata["clip_id"] for bundle in bundles]
    )
    exact_json_equal(recomputed_effects, analysis["effects"], "effects")
    exact_json_equal(recomputed_gates, analysis["gates"], "gates")
    expected_decision = (
        "GO_FOR_WAN_SCREEN" if recomputed_gates["all_passed"] else "STOP_RENDERER_ATTRIBUTION"
    )
    if analysis["decision"] != expected_decision:
        raise AuditError("decision differs from recomputed gate")

    false_flags = sum(
        require_false_flags(document, name) for name, document in documents.items()
    )
    false_flags += sum(
        require_false_flags(rows, name)
        for name, rows in (
            ("frame_rows", frame_rows),
            ("clip_rows", clip_rows),
            ("timing_rows", timing_rows),
            ("provenance_rows", provenance_rows),
        )
    )
    return seal({
        "schema_version": 1,
        "kind": "corrected_renderer_attribution_hardened_audit",
        "status": "hardened_audit_passed",
        "decision": analysis["decision"],
        "registration_identity_sha256": registration["identity_sha256"],
        "preparation_identity_sha256": preparation["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "completion_identity_sha256": complete["identity_sha256"],
        "registered_source_sha256": registration["source"]["sha256"],
        "artifact_hashes_checked": artifact_hashes,
        "counts": observed_counts,
        "bootstrap_samples_recomputed_per_contrast": BOOTSTRAP_SAMPLES,
        "effects_recomputed": len(recomputed_effects),
        "gates_recomputed": len(recomputed_gates),
        "explicit_false_flags": false_flags,
        "auditor_source_sha256": sha256_file(Path(__file__).resolve()),
        "protected_test_accessed": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registered-source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = audit(args.output, args.registered_source)
    if args.receipt is not None:
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        write_json(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
