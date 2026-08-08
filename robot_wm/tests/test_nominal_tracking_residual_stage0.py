from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module("nominal_tracking_residual", "tools/stage0_nominal_tracking_residual.py")
audit_tool = load_module(
    "audit_nominal_tracking_residual", "tools/audit_nominal_tracking_residual_stage0.py"
)


def test_cache_to_official_mapping_and_inverse() -> None:
    cache = np.arange(14)
    official = probe.cache14_to_official14(cache)
    assert official.tolist() == [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    inverse = np.argsort(probe.CACHE14_TO_OFFICIAL14)
    np.testing.assert_array_equal(official[inverse], cache)


def test_exact_frame_action_target_and_causality() -> None:
    states = np.arange(13 * 14, dtype=np.float32).reshape(13, 14) / 10.0
    actions = np.arange(13 * 5 * 14, dtype=np.float32).reshape(13, 5, 14) / 20.0
    result = probe.construct_clip_arrays(states, actions)

    assert result["history"].shape == (5 * 14 + 4 * 5 * 14 + 4 * 14,)
    assert result["future_actions"].shape == (8, 5, 14)
    np.testing.assert_array_equal(result["nominal_endpoint"], actions[4:12, -1])
    np.testing.assert_array_equal(
        result["target_residual"], states[5:13] - actions[4:12, -1]
    )
    np.testing.assert_array_equal(
        result["hold_current_residual"], states[4][None, :] - actions[4:12, -1]
    )

    changed_future_state = states.copy()
    changed_future_state[5:] += 100.0
    changed = probe.construct_clip_arrays(changed_future_state, actions)
    np.testing.assert_array_equal(changed["history"], result["history"])
    np.testing.assert_array_equal(changed["future_actions"], result["future_actions"])
    assert not np.array_equal(changed["target_residual"], result["target_residual"])


def test_future_action_changes_no_observed_history() -> None:
    states = np.arange(13 * 14, dtype=np.float32).reshape(13, 14)
    actions = np.arange(13 * 5 * 14, dtype=np.float32).reshape(13, 5, 14)
    base = probe.construct_clip_arrays(states, actions)
    modified_actions = actions.copy()
    modified_actions[4:12] += 7.0
    changed = probe.construct_clip_arrays(states, modified_actions)
    np.testing.assert_array_equal(changed["history"], base["history"])
    assert not np.array_equal(changed["future_actions"], base["future_actions"])
    assert not np.array_equal(changed["target_residual"], base["target_residual"])


def test_manifest_requires_exact_stride_and_split() -> None:
    rows = [
        {
            "split": "train",
            "sample_size": 13,
            "chunk_size": 5,
            "action_span": 65,
            "start": index * 100,
            "frame_indices": list(range(index * 100, index * 100 + 65, 5)),
            "episode_dir": f"/episode/{index}",
            "clip_id": f"clip-{index}",
        }
        for index in range(2)
    ]
    probe.validate_manifest_rows(rows, 2, "train")
    rows[0]["frame_indices"][4] += 1
    with pytest.raises(probe.ProbeError, match="frame ordering"):
        probe.validate_manifest_rows(rows, 2, "train")


def test_preprocessing_audit_detects_ceiling_not_nearest(tmp_path: Path) -> None:
    source = tmp_path / "abc_preprocess.py"
    source.write_text(
        "# nearest resampling\nidx = np.clip(np.searchsorted(ts, frame_ts), 0, len(ts)-1)\n"
    )
    result = probe.audit_preprocessing_source(source)
    assert result["claimed_comment_mentions_nearest"] is True
    assert result["actual_behavior"].startswith("next/ceiling")


def _seal(payload: dict) -> dict:
    payload = dict(payload)
    payload["identity_sha256"] = audit_tool.canonical_identity(payload)
    return payload


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": audit_tool.sha256_file(path),
    }


def test_artifact_audit_rejects_protected_flag(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    per_clip = tmp_path / "per_clip_metrics.jsonl"
    per_clip.write_text(
        "".join(
            json.dumps(
                {
                    "clip_index": index,
                    "clip_id": f"clip-{index}",
                    "episode_dir": f"/val/{index}",
                    "donor_episode_dir": f"/val/{(index + 1) % 64}",
                    "protected_test_accessed": False,
                }
            )
            + "\n"
            for index in range(64)
        )
    )
    provenance = tmp_path / "input_provenance.jsonl"
    provenance.write_text(
        "".join(
            json.dumps(
                {
                    "split": "train" if index < 512 else "val",
                    "protected_test_accessed": False,
                }
            )
            + "\n"
            for index in range(576)
        )
    )
    registration = _seal(
        {
            "kind": "nominal_tracking_residual_stage0_registration",
            "protected_test_accessed": False,
        }
    )
    analysis = _seal(
        {
            "kind": "nominal_tracking_residual_stage0_analysis",
            "registration_identity_sha256": registration["identity_sha256"],
            "decision": "NO_GO",
            "protected_test_accessed": False,
        }
    )
    for filename, payload in (("registration.json", registration), ("analysis.json", analysis)):
        (tmp_path / filename).write_text(json.dumps(payload))
    complete = _seal(
        {
            "kind": "nominal_tracking_residual_stage0_complete",
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "status": "completed",
            "decision": "NO_GO",
            "artifacts": {
                "source": _record(source),
                "per_clip_metrics": _record(per_clip),
                "input_provenance": _record(provenance),
            },
            "protected_test_accessed": False,
        }
    )
    (tmp_path / "run_complete.json").write_text(json.dumps(complete))
    assert audit_tool.audit(tmp_path)["status"] == "passed"

    complete["protected_test_accessed"] = True
    complete = _seal(complete)
    (tmp_path / "run_complete.json").write_text(json.dumps(complete))
    with pytest.raises(audit_tool.AuditError, match="not false"):
        audit_tool.audit(tmp_path)
