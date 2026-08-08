#!/usr/bin/env python3
"""Read-only integrity audit for the nominal tracking-residual Stage-0 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_KINDS = {
    "registration.json": "nominal_tracking_residual_stage0_registration",
    "analysis.json": "nominal_tracking_residual_stage0_analysis",
    "run_complete.json": "nominal_tracking_residual_stage0_complete",
}


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_identity(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_false_flags(value: Any, location: str = "root") -> int:
    checked = 0
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "protected_test_accessed":
                checked += 1
                if child is not False:
                    raise AuditError(f"{child_location} is not false")
            checked += require_false_flags(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            checked += require_false_flags(child, f"{location}[{index}]")
    return checked


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    payloads: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    protected_flags = 0
    for filename, expected_kind in EXPECTED_KINDS.items():
        path = root / filename
        payload = read_json(path)
        if payload.get("kind") != expected_kind:
            raise AuditError(f"{filename} kind differs: {payload.get('kind')!r}")
        expected_identity = canonical_identity(payload)
        if payload.get("identity_sha256") != expected_identity:
            raise AuditError(f"{filename} identity differs")
        protected_flags += require_false_flags(payload, filename)
        payloads[filename] = payload
        identities[filename] = expected_identity

    registration = payloads["registration.json"]
    analysis = payloads["analysis.json"]
    complete = payloads["run_complete.json"]
    if analysis.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise AuditError("analysis registration identity differs")
    if complete.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise AuditError("completion registration identity differs")
    if complete.get("analysis_identity_sha256") != analysis["identity_sha256"]:
        raise AuditError("completion analysis identity differs")
    if complete.get("status") != "completed" or complete.get("decision") != analysis.get("decision"):
        raise AuditError("completion status/decision differs")

    hashes_checked = 0
    for name, record in complete["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file():
            raise AuditError(f"missing artifact {name}: {path}")
        if path.stat().st_size != record["bytes"]:
            raise AuditError(f"artifact size differs for {name}")
        if sha256_file(path) != record["sha256"]:
            raise AuditError(f"artifact hash differs for {name}")
        hashes_checked += 1

    per_clip = root / "per_clip_metrics.jsonl"
    per_clip_rows = [json.loads(line) for line in per_clip.open() if line.strip()]
    if len(per_clip_rows) != 64 or [row["clip_index"] for row in per_clip_rows] != list(range(64)):
        raise AuditError("per-clip artifact is not one ordered val64 pass")
    if len({row["clip_id"] for row in per_clip_rows}) != 64:
        raise AuditError("per-clip artifact has repeated clip IDs")
    for row in per_clip_rows:
        if row["episode_dir"] == row["donor_episode_dir"]:
            raise AuditError("shuffled action donor shares an episode")
        protected_flags += require_false_flags(row, "per_clip")

    provenance = root / "input_provenance.jsonl"
    provenance_rows = [json.loads(line) for line in provenance.open() if line.strip()]
    if len(provenance_rows) != 576:
        raise AuditError("input provenance does not contain train512+val64")
    if [row["split"] for row in provenance_rows].count("train") != 512:
        raise AuditError("input provenance train count differs")
    if [row["split"] for row in provenance_rows].count("val") != 64:
        raise AuditError("input provenance validation count differs")
    protected_flags += require_false_flags(provenance_rows, "input_provenance")

    return {
        "status": "passed",
        "root": str(root),
        "decision": analysis["decision"],
        "identities": identities,
        "artifact_hashes_checked": hashes_checked,
        "per_clip_rows": len(per_clip_rows),
        "input_provenance_rows": len(provenance_rows),
        "protected_test_false_flags_checked": protected_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.artifact_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
