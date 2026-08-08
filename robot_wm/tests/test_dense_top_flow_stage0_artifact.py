import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_dense_top_flow_stage0 import (
    EXPECTED_SOURCE_SHA256,
    canonical_identity,
    validate_protocol,
    verify_identity,
)


def test_executed_dense_stage0_source_is_preserved_byte_for_byte():
    root = Path(__file__).resolve().parents[2]
    source = root / "tools" / "stage0_dense_top_flow_proxy.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256


def test_canonical_identity_ignores_embedded_identity():
    payload = {"schema_version": 1, "kind": "example", "value": [2, 7, 1]}
    identity = canonical_identity(payload)
    assert canonical_identity({**payload, "identity_sha256": identity}) == identity


def test_verify_identity_rejects_mutation(tmp_path):
    payload = {"schema_version": 1, "kind": "example", "value": 7}
    payload["identity_sha256"] = canonical_identity(payload)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload))
    assert verify_identity(path) == payload
    payload["value"] = 8
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_identity(path)


def test_protocol_validator_rejects_future_input_leakage():
    registration = {
        "protocol": {
            "predictor_inputs": {
                "future_rgb_or_future_derived_feature": False,
                "history_rgb_frames": [0, 1, 2, 3, 4],
            },
            "target_only": {
                "future_rgb_frames": [4, 5, 6, 7, 8, 9, 10, 11, 12],
                "raw_target_shape": [8, 2, 12, 20],
                "train_only_target_compression": "centered PCA192",
            },
            "decision_gate": {
                "aligned_vs_history_dense_mse": {"point_min_percent": 10.0},
                "aligned_vs_shuffled_dense_mse": {"point_min_percent": 10.0},
                "all_required": True,
            },
        }
    }
    analysis = {
        "data_contract": {
            "future_rgb_used_as_predictor_input": False,
            "future_rgb_used_for_target_and_scoring_only": True,
            "train_episode_count": 512,
            "validation_episode_count": 64,
            "train_validation_episode_overlap": 0,
        },
        "target_pca": {"components": 192},
        "input_compression": {"history_components": 64, "action_components": 64},
        "decision": "NO_GO",
        "gates": {"all_passed": False},
    }
    validate_protocol(registration, analysis)
    registration["protocol"]["predictor_inputs"]["future_rgb_or_future_derived_feature"] = True
    with pytest.raises(ValueError, match="future RGB"):
        validate_protocol(registration, analysis)
