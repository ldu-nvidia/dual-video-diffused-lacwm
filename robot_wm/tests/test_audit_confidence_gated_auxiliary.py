from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_confidence_gated_auxiliary as audit


def _row(source: str, clip: int, temporal: float) -> dict[str, object]:
    suffix = f"{clip:064x}"
    return {
        "arm": "MID-ON",
        "source": source,
        "nfe": 1,
        "clip_index": clip,
        "video_nmse": temporal * 2.0,
        "decoded_mse": temporal * 1.5,
        "temporal_mse": temporal,
        "auxiliary_future_nmse": 0.1 + clip,
        "auxiliary_dc_nmse": 0.2 + clip,
        "auxiliary_motion_nmse": 0.3 + clip,
        "auxiliary_future_cosine": 0.9 - clip * 1e-4,
        "protected_test_accessed": False,
        "oracle_leakage": False,
        "future_rgb_passed_to_sampler": False,
        "clean_auxiliary_passed_to_sampler": False,
        "deployable": True,
        "online_teacher_calls": 0,
        "raw_target_sha256": "target" + suffix,
        "video_initial_sha256": "video" + suffix,
        "auxiliary_initial_sha256": "aux" + suffix,
        "auxiliary_final_sha256": "generated" + suffix,
        "effective_state_gate": 0.01,
        "effective_clock_gate": 0.0,
        "wan_latency_ms": 50.0 + clip,
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    mid_on: list[dict[str, object]] = []
    mid_off: list[dict[str, object]] = []
    for clip in range(audit.EXPECTED_CLIPS):
        base = 1.0 + clip / 100.0
        values = {
            audit.OFF_SOURCE: base,
            audit.ALIGNED_SOURCE: base * (0.9 if clip % 2 == 0 else 1.1),
            audit.CORRUPTED_SOURCE: base * 1.2,
        }
        for source, temporal in values.items():
            mid_on.append(_row(source, clip, temporal))
            package_row = _row(source, clip, base * 1.05)
            package_row["arm"] = "MID-OFF"
            mid_off.append(package_row)
    on_path = tmp_path / "on.jsonl"
    off_path = tmp_path / "off.jsonl"
    protocol_path = tmp_path / "protocol.md"
    on_path.write_text("".join(json.dumps(row) + "\n" for row in mid_on))
    off_path.write_text("".join(json.dumps(row) + "\n" for row in mid_off))
    protocol_path.write_text("frozen\n")
    return on_path, off_path, protocol_path


def test_target_accuracy_is_rejected_and_oracle_ceiling_is_reported(tmp_path: Path):
    on_path, off_path, protocol_path = _write_fixture(tmp_path)
    result = audit.analyze(
        on_path, off_path, protocol_path, replicates=200
    )

    inventory = result["confidence_telemetry_inventory"]
    assert inventory["eligible_varying_fields"] == []
    assert "auxiliary_future_nmse" in inventory[
        "target_derived_forbidden_fields_present"
    ]
    assert result["fitted_confidence_gate"]["status"] == (
        "NOT_FIT_TELEMETRY_BLOCKER"
    )
    oracle = result["perfect_temporal_oracle_unattainable"]
    assert oracle["selected_clips"] == audit.EXPECTED_CLIPS // 2
    assert oracle["metrics"]["temporal_mse"][
        "relative_improvement_percent"
    ] > 0
    assert result["fitted_confidence_gate"][
        "honest_policy_exactly_reuses_same_checkpoint_off_endpoint"
    ]


def test_any_protected_row_fails_closed(tmp_path: Path):
    on_path, off_path, protocol_path = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in on_path.read_text().splitlines()]
    rows[0]["protected_test_accessed"] = True
    on_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(audit.AuditError, match="protected_test_accessed"):
        audit.analyze(on_path, off_path, protocol_path, replicates=100)


def test_unpaired_noise_fails_closed(tmp_path: Path):
    on_path, off_path, protocol_path = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in on_path.read_text().splitlines()]
    rows[-1]["video_initial_sha256"] = "wrong"
    on_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(audit.AuditError, match="video noise is not paired"):
        audit.analyze(on_path, off_path, protocol_path, replicates=100)
