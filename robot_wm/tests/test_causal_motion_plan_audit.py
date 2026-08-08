"""Focused tests for the prospective CAMP evaluation collation audit."""

from __future__ import annotations

import hashlib
import json

import pytest

from tools import causal_motion_plan_audit as audit


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _row(
    arm: str,
    clip_index: int,
    endpoint: audit.Endpoint,
    *,
    clips: int,
) -> dict:
    source = endpoint.condition_source
    shuffle_donor = (clip_index + 1) % clips
    planner_donor = shuffle_donor if source == "action_shuffled" else clip_index
    injected_donor = shuffle_donor if source == "shuffled" else clip_index
    generated_plan = (
        _hash(f"action-plan:{clip_index}:actions:{planner_donor}")
        if source == "action_shuffled"
        else _hash(f"plan:{clip_index}")
    )
    injected_plan = (
        _hash(f"plan:{injected_donor}") if source == "shuffled" else generated_plan
    )
    if arm == "PLAN-OFF" and source in audit.BASE_CONDITION_SOURCES:
        final_latent = _hash(f"PLAN-OFF:final:{clip_index}:nfe:{endpoint.nfe}")
        decode = _hash(f"PLAN-OFF:decode:{clip_index}:nfe:{endpoint.nfe}")
    else:
        final_latent = _hash(
            f"{arm}:final:{clip_index}:{source}:nfe:{endpoint.nfe}"
        )
        decode = _hash(f"{arm}:decode:{clip_index}:{source}:nfe:{endpoint.nfe}")
    payload = {
        "schema_version": audit.SCHEMA_VERSION,
        "kind": audit.ROW_KIND,
        "arm": arm,
        "clip_index": clip_index,
        "clip_id": f"validation-clip-{clip_index}",
        "endpoint": {
            "condition_source": source,
            "nfe": endpoint.nfe,
            "primary_gate": endpoint.primary_gate,
        },
        "evaluation_split": "validation",
        "protected_test_accessed": False,
        "clean_future_rgb_passed_to_sampler": False,
        "clean_future_video_latent_passed_to_sampler": False,
        "clean_future_feature_passed_to_sampler": False,
        "clean_plan_passed_to_sampler": False,
        "teacher_call_count": 0,
        "scoring_constructed_after_all_sampling": True,
        "runtime_plan_fusion_enabled": arm == "PLAN-ON" and source != "off",
        "planner_action_donor_clip_index": planner_donor,
        "injected_plan_donor_clip_index": injected_donor,
        "model_identity": {
            "parameter_schema_sha256": _hash("parameter-schema"),
            "planner_checkpoint_sha256": _hash("planner-checkpoint"),
            "motion_plan_stats_sha256": _hash("motion-plan-stats"),
        },
        "call_counts": {"planner": 2, "wan": endpoint.nfe},
        "latency_device_synchronized": True,
        "latency_seconds": {
            "history_encode": 0.01,
            "planner": 0.02,
            "wan": 0.03,
            "decode": 0.04,
            "end_to_end": 0.11,
        },
        "metrics": {
            "video_future_nmse": 1.0,
            "decoded_mse_unit_range": 0.2,
            "decoded_temporal_difference_mse_unit_range": 0.1,
        },
        "tensor_sha256": {
            "cached_rgb_sha256": _hash(f"rgb:{clip_index}"),
            "local_actions_sha256": _hash(f"actions:{clip_index}"),
            "planner_actions_sha256": _hash(f"actions:{planner_donor}"),
            "history_latent_sha256": _hash(f"history:{clip_index}"),
            "video_noise_sha256": _hash(f"video-noise:{clip_index}"),
            "plan_noise_sha256": _hash(f"plan-noise:{clip_index}"),
            "generated_plan_sha256": generated_plan,
            "injected_plan_sha256": injected_plan,
            "final_latent_sha256": final_latent,
            "decode_sha256": decode,
        },
    }
    return audit.identity_payload(payload)


def _rows(clips: int = 2) -> list[dict]:
    return [
        _row(arm, clip_index, endpoint, clips=clips)
        for arm in audit.ARMS
        for clip_index in range(clips)
        for endpoint in audit.ENDPOINTS
    ]


def _replace_identity(row: dict) -> dict:
    unsigned = dict(row)
    unsigned.pop("identity_sha256", None)
    return audit.identity_payload(unsigned)


def test_endpoint_grid_has_only_nfe1_primary_and_action_shuffle_descriptive() -> None:
    assert len(audit.ENDPOINTS) == 10
    assert {
        (endpoint.condition_source, endpoint.nfe)
        for endpoint in audit.ENDPOINTS
    } == {
        (source, nfe)
        for source in audit.BASE_CONDITION_SOURCES
        for nfe in audit.NFE_GRID
    } | {("action_shuffled", 1)}
    assert [endpoint.code for endpoint in audit.ENDPOINTS if endpoint.primary_gate] == [
        "aligned_nfe_1",
        "off_nfe_1",
        "shuffled_nfe_1",
    ]
    assert audit.ENDPOINT_BY_KEY[("action_shuffled", 1)].primary_gate is False


def test_complete_paired_grid_passes_and_is_content_bound() -> None:
    result = audit.audit_rows(_rows(), expected_clips=2)
    assert result["status"] == "passed"
    assert result["rows"]["count"] == 2 * 2 * 10
    assert len(result["rows"]["canonical_sha256"]) == 64
    assert audit.identity_valid(result)
    assert result["verified"]["plan_off_control_outputs_bit_identical"] is True
    assert all("nfe_1" in code for code in result["primary_gate_endpoints"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clean_future_rgb_passed_to_sampler", True, "schema"),
        ("teacher_call_count", 1, "schema"),
        ("latency_device_synchronized", False, "schema"),
    ],
)
def test_clean_future_teacher_and_unsynchronized_timing_fail_closed(
    field: str, value: object, message: str
) -> None:
    rows = _rows()
    changed = dict(rows[0])
    changed[field] = value
    rows[0] = _replace_identity(changed)
    with pytest.raises(audit.CausalMotionPlanAuditError, match=message):
        audit.audit_rows(rows, expected_clips=2)


def test_exact_planner_and_wan_calls_are_required() -> None:
    rows = _rows()
    changed = dict(rows[0])
    changed["call_counts"] = {"planner": 1, "wan": 1}
    rows[0] = _replace_identity(changed)
    with pytest.raises(audit.CausalMotionPlanAuditError, match="schema"):
        audit.audit_rows(rows, expected_clips=2)


def test_cross_arm_noise_pairing_and_shared_identity_are_required() -> None:
    rows = _rows()
    for index, row in enumerate(rows):
        if row["arm"] == "PLAN-ON" and row["clip_index"] == 0:
            changed = dict(row)
            hashes = dict(changed["tensor_sha256"])
            hashes["video_noise_sha256"] = _hash("unpaired-video-noise")
            changed["tensor_sha256"] = hashes
            rows[index] = _replace_identity(changed)
    with pytest.raises(audit.CausalMotionPlanAuditError, match="across arms"):
        audit.audit_rows(rows, expected_clips=2)

    rows = _rows()
    changed = dict(rows[-1])
    identity = dict(changed["model_identity"])
    identity["motion_plan_stats_sha256"] = _hash("different-stats")
    changed["model_identity"] = identity
    rows[-1] = _replace_identity(changed)
    with pytest.raises(audit.CausalMotionPlanAuditError, match="normalization stats"):
        audit.audit_rows(rows, expected_clips=2)


def test_shuffled_plan_must_match_global_donor() -> None:
    rows = _rows()
    for index, row in enumerate(rows):
        if (
            row["arm"] == "PLAN-ON"
            and row["clip_index"] == 0
            and row["endpoint"]["condition_source"] == "shuffled"
            and row["endpoint"]["nfe"] == 1
        ):
            changed = dict(row)
            hashes = dict(changed["tensor_sha256"])
            hashes["injected_plan_sha256"] = _hash("not-the-declared-donor")
            changed["tensor_sha256"] = hashes
            rows[index] = _replace_identity(changed)
            break
    with pytest.raises(audit.CausalMotionPlanAuditError, match="donor|across arms"):
        audit.audit_rows(rows, expected_clips=2)


def test_plan_off_outputs_must_be_bit_identical_across_controls() -> None:
    rows = _rows()
    for index, row in enumerate(rows):
        if (
            row["arm"] == "PLAN-OFF"
            and row["clip_index"] == 0
            and row["endpoint"]["condition_source"] == "shuffled"
            and row["endpoint"]["nfe"] == 2
        ):
            changed = dict(row)
            hashes = dict(changed["tensor_sha256"])
            hashes["decode_sha256"] = _hash("changed-off-output")
            changed["tensor_sha256"] = hashes
            rows[index] = _replace_identity(changed)
            break
    with pytest.raises(audit.CausalMotionPlanAuditError, match="bit-identical"):
        audit.audit_rows(rows, expected_clips=2)


def test_jsonl_sources_are_hashed_and_audit_write_is_exclusive(tmp_path) -> None:
    rows = _rows()
    shard = tmp_path / "rows.jsonl"
    shard.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    loaded, records = audit.read_jsonl([shard.resolve()])
    result = audit.audit_rows(loaded, expected_clips=2, source_files=records)
    assert result["rows"]["source_files"][0]["rows"] == len(rows)
    output = tmp_path / "audit.json"
    audit.exclusive_write_json(output.resolve(), result)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    with pytest.raises(audit.CausalMotionPlanAuditError, match="already exists"):
        audit.exclusive_write_json(output.resolve(), result)
