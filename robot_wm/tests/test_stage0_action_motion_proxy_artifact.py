import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_stage0_action_motion_proxy import (
    EXPECTED_SOURCE_SHA256,
    canonical_identity,
    verify_identity,
)


def test_executed_stage0_source_is_preserved_byte_for_byte():
    root = Path(__file__).resolve().parents[2]
    source = root / "tools" / "stage0_action_motion_proxy.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256


def test_canonical_identity_ignores_embedded_identity():
    payload = {"schema_version": 1, "kind": "example", "value": [3, 1, 4]}
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
