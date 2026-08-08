"""Schema bridge tests from the CAMP evaluator into the independent audit."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from tools import causal_motion_plan_audit as audit  # noqa: E402
from tools import causal_motion_plan_evaluate as evaluate  # noqa: E402
from tools import causal_motion_plan_workflow as workflow  # noqa: E402


def test_rendered_environment_binds_registered_source_and_disables_user_site() -> None:
    registration = {
        "tool_repository": {"path": "/registered/camp"},
        "runtime": {
            "videox_home": "/registered/videox",
            "wan_dir": "/registered/wan",
        },
        "training": {
            "manifest": {"path": "/data/train.jsonl"},
            "cache_metadata": {"path": "/data/train.json"},
        },
        "validation": {
            "manifest": {"path": "/data/val.jsonl"},
            "cache_metadata": {"path": "/data/val.json"},
        },
    }
    environment = workflow._common_environment(registration)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        "/registered/camp",
        "/registered/camp/projects/latent_action_models",
        "/registered/camp/tools/env/videox_shim",
        "/registered/videox",
    ]


def test_evaluator_row_is_accepted_by_fail_closed_audit_schema() -> None:
    clean = torch.zeros(1, 16, 4, 1, 1)
    clean[:, :, 1] = 0.5
    clean[:, :, 2] = 1.0
    clean[:, :, 3] = 2.0
    plan = torch.randn(1, 16, 2, 6, 30)
    result = SimpleNamespace(
        video_latent=clean.clone(),
        decoded_future=torch.zeros(1, 3, 8, 2, 2),
        history_latent=clean[:, :, :2].clone(),
        generated_plan=plan.clone(),
        injected_plan=plan.clone(),
        planner_calls=2,
        wan_calls=1,
        history_encode_seconds=0.01,
        planner_seconds=0.02,
        wan_seconds=0.03,
        decode_seconds=0.04,
        end_to_end_seconds=0.11,
    )
    sample = {
        "rgb": torch.zeros(1, 13, 3, 2, 2),
        "actions": torch.zeros(1, 13, 5, 157),
    }
    scoring = {
        "video_clean": clean,
        "ground_truth": torch.zeros(1, 3, 8, 2, 2, dtype=torch.uint8),
        "history_last": torch.zeros(1, 3, 1, 2, 2, dtype=torch.uint8),
    }
    arm = workflow.ARM_BY_CODE["PLAN-OFF"]
    row = evaluate._row(
        arm=arm,
        endpoint=evaluate.ENDPOINTS[0],
        sample=sample,
        result=result,
        planner_actions=sample["actions"],
        planner_action_donor_clip_index=0,
        injected_plan_donor_clip_index=0,
        video_noise=torch.zeros_like(clean),
        plan_noise=torch.zeros_like(plan),
        scoring=scoring,
        target_plan=plan,
        clip_index=0,
        clip_id="clip-zero",
        sampling_id=8_100_000,
        sealed={
            "identity_sha256": "1" * 64,
            "planner_snapshot": {"sha256": "2" * 64},
            "motion_plan_stats": {"sha256": "3" * 64},
        },
        arm_artifacts={
            "snapshot": {"path": "/tmp/snapshot.pt", "sha256": "4" * 64},
            "trace": {"header": {"parameter_schema_sha256": "5" * 64}},
        },
        peak_memory_bytes=123,
    )
    observed_arm, observed_clip, endpoint = audit._validate_single_row(
        row, expected_clips=2
    )
    assert (observed_arm, observed_clip) == ("PLAN-OFF", 0)
    assert endpoint.condition_source == "aligned"
