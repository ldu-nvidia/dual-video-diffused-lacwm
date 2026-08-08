#!/usr/bin/env python3
"""Verify the sealed dense top-view Stage-0 run using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "70f8cf1ba9f1246745ecceb0e8e7b35f94b3b52c3370d4f6e33790f54c80a1a8"
EXPECTED_REGISTRATION_IDENTITY = "1d2f45fbd7209b94fff8b0703e3b5c670d834acb0fd3a3dc991feae13b39bf5b"
EXPECTED_ANALYSIS_IDENTITY = "0cce95d9fd971dbae243ebd10d078225a759d6dad696a9591be0d4b72c94c8df"
EXPECTED_COMPLETION_IDENTITY = "e2dbe55dce37159cd23c6dbd4fe29cc28047496f62eb1d1ed7cb9b8239e7eb1e"


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


def validate_protocol(registration: dict, analysis: dict) -> None:
    protocol = registration["protocol"]
    inputs = protocol["predictor_inputs"]
    target = protocol["target_only"]
    gate = protocol["decision_gate"]

    if inputs["future_rgb_or_future_derived_feature"] is not False:
        raise ValueError("future RGB entered the predictor input contract")
    if inputs["history_rgb_frames"] != [0, 1, 2, 3, 4]:
        raise ValueError("observed history frame contract differs")
    if target["future_rgb_frames"] != [4, 5, 6, 7, 8, 9, 10, 11, 12]:
        raise ValueError("future target frame contract differs")
    if target["raw_target_shape"] != [8, 2, 12, 20]:
        raise ValueError("dense target shape differs")
    if target["train_only_target_compression"] != "centered PCA192":
        raise ValueError("target compression contract differs")
    if gate["aligned_vs_history_dense_mse"]["point_min_percent"] != 10.0:
        raise ValueError("history gate differs")
    if gate["aligned_vs_shuffled_dense_mse"]["point_min_percent"] != 10.0:
        raise ValueError("shuffled-action gate differs")
    if not gate["all_required"]:
        raise ValueError("decision gate no longer requires every criterion")

    data = analysis["data_contract"]
    if data["future_rgb_used_as_predictor_input"] is not False:
        raise ValueError("analysis reports future RGB predictor leakage")
    if data["future_rgb_used_for_target_and_scoring_only"] is not True:
        raise ValueError("future RGB target-only contract differs")
    if data["train_episode_count"] != 512 or data["validation_episode_count"] != 64:
        raise ValueError("split cardinality differs")
    if data["train_validation_episode_overlap"] != 0:
        raise ValueError("train and validation episodes overlap")
    if analysis["target_pca"]["components"] != 192:
        raise ValueError("analysis target PCA dimension differs")
    if analysis["input_compression"]["history_components"] != 64:
        raise ValueError("analysis history PCA dimension differs")
    if analysis["input_compression"]["action_components"] != 64:
        raise ValueError("analysis action PCA dimension differs")
    if analysis["decision"] != "NO_GO" or analysis["gates"]["all_passed"]:
        raise ValueError("sealed decision differs")


def audit(run_root: Path) -> dict:
    registration = verify_identity(run_root / "registration.json")
    analysis = verify_identity(run_root / "analysis.json")
    complete = verify_identity(run_root / "run_complete.json")

    exact_identities = {
        "registration": EXPECTED_REGISTRATION_IDENTITY,
        "analysis": EXPECTED_ANALYSIS_IDENTITY,
        "completion": EXPECTED_COMPLETION_IDENTITY,
    }
    records = {
        "registration": registration,
        "analysis": analysis,
        "completion": complete,
    }
    for name, expected in exact_identities.items():
        if records[name]["identity_sha256"] != expected:
            raise ValueError(f"unexpected {name} identity")

    if analysis.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ValueError("analysis registration binding differs")
    if complete.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise ValueError("completion registration binding differs")
    if complete.get("analysis_identity_sha256") != analysis["identity_sha256"]:
        raise ValueError("completion analysis binding differs")
    if any(record.get("protected_test_accessed") for record in records.values()):
        raise ValueError("protected-test flag is not false")

    validate_protocol(registration, analysis)

    for name, record in complete["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"artifact hash mismatch for {name}: {actual}")
    if complete["artifacts"]["source"]["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("executed source is not the preserved dense Stage-0 source")

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
        "same_episode_donors": 0,
        "protected_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
