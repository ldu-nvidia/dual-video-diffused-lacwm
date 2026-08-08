#!/usr/bin/env python3
"""Verify sealed Stage-0 action-to-motion proxy artifacts using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "3be829af2fc71c83e211709fc28fd9c101bcabecad9a134b5d92ef17aff21818"


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(block):
            digest.update(data)
    return digest.hexdigest()


def canonical_identity(payload: dict) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_identity(path: Path) -> dict:
    payload = read_json(path)
    expected = payload.get("identity_sha256")
    actual = canonical_identity(payload)
    if expected != actual:
        raise ValueError(f"identity mismatch for {path}: {expected} != {actual}")
    return payload


def audit(run_root: Path) -> dict:
    registration = verify_identity(run_root / "registration.json")
    analysis = verify_identity(run_root / "analysis.json")
    complete = verify_identity(run_root / "run_complete.json")
    if analysis.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ValueError("analysis registration binding differs")
    if complete.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ValueError("completion registration binding differs")
    if complete.get("analysis_identity_sha256") != analysis["identity_sha256"]:
        raise ValueError("completion analysis binding differs")
    if any(record.get("protected_test_accessed") for record in (registration, analysis, complete)):
        raise ValueError("protected-test flag is not false")

    for name, record in complete["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"artifact hash mismatch for {name}: {actual}")
    if complete["artifacts"]["source"]["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("executed source is not the preserved Stage-0 source")

    rows_path = Path(complete["artifacts"]["per_clip_metrics"]["path"])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if len(rows) != 64 or len({row["clip_id"] for row in rows}) != 64:
        raise ValueError("validation row cardinality differs")
    if any(row.get("protected_test_accessed") for row in rows):
        raise ValueError("a validation row accessed protected test")
    if any(row["episode_dir"] == row["donor_episode_dir"] for row in rows):
        raise ValueError("an episode-shuffled donor came from the same episode")
    return {
        "status": "verified",
        "registration_identity_sha256": registration["identity_sha256"],
        "analysis_identity_sha256": analysis["identity_sha256"],
        "completion_identity_sha256": complete["identity_sha256"],
        "decision": analysis["decision"],
        "validation_rows": len(rows),
        "protected_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
