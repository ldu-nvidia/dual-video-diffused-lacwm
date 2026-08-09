from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools import recurrent_flow_wan_bridge as bridge


def _valid_compact(count: int) -> np.ndarray:
    value = np.zeros((count, 8, 4, 24, 40), dtype=np.float16)
    for index in range(count):
        value[index, :, 2, index % 24, index % 40] = np.float16(0.5)
        value[index, :, 0, index % 24, index % 40] = np.float16(index / 100.0)
    return value


def test_endpoint_grid_is_exact_seven_equal_nfe1_calls() -> None:
    endpoints = bridge.endpoint_grid()
    assert len(endpoints) == 7
    assert {item.nfe for item in endpoints} == {1}
    assert [item.arm for item in endpoints[:2]] == ["PARENT-VPM", "FLOW-OFF"]
    assert [item.condition_source for item in endpoints[2:]] == list(
        bridge.RUNTIME_SOURCES
    )
    assert len({item.code for item in endpoints}) == len(endpoints)


def test_plan_authorization_switch_is_receipt_only() -> None:
    args = bridge.build_parser().parse_args(
        [
            "plan",
            "--registration",
            "/tmp/registration.json",
            "--authorize-screen",
            "--authorization-token",
            "AUTHORIZE_RECURRENT_FLOW_WAN_NFE1",
        ]
    )
    assert args.authorize_screen is True
    assert args.authorization_token == "AUTHORIZE_RECURRENT_FLOW_WAN_NFE1"


def test_build_predictor_features_matches_sealed_406_width() -> None:
    measured = np.arange(2 * 5 * 14, dtype=np.float32).reshape(2, 5, 14)
    actions = np.arange(2 * 13 * 5 * 14, dtype=np.float32).reshape(2, 13, 5, 14)
    history, future, q4 = bridge.build_predictor_features(measured, actions)
    assert history.shape == (2, 406)
    assert future.shape == (2, 8, 5, 14)
    assert np.array_equal(q4, measured[:, 4])
    expected_residual = measured[:, 1:] - actions[:, :4, -1]
    assert np.array_equal(history[:, -56:], expected_residual.reshape(2, -1))


def test_derive_interventions_is_episode_shuffle_and_nonwrapping() -> None:
    aligned = _valid_compact(3)
    shuffled, wrong = bridge.derive_interventions(
        aligned, donor_indexes=(1, 2, 0), eligible=(True, True, True)
    )
    assert np.array_equal(shuffled[0], aligned[1])
    assert np.array_equal(shuffled[1], aligned[2])
    assert np.array_equal(shuffled[2], aligned[0])
    assert np.array_equal(wrong[:, :-1], aligned[:, 1:])
    assert not np.any(wrong[:, -1])


def test_derive_interventions_keeps_ineligible_rows_zero() -> None:
    aligned = _valid_compact(3)
    aligned[1] = 0
    shuffled, _ = bridge.derive_interventions(
        aligned, donor_indexes=(2, None, 0), eligible=(True, False, True)
    )
    assert not np.any(shuffled[1])
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="donor"):
        bridge.derive_interventions(
            aligned, donor_indexes=(1, None, 0), eligible=(True, False, True)
        )


def test_strict_npz_reader_never_opens_forbidden_member(tmp_path: Path) -> None:
    archive = tmp_path / "privileged.npz"
    np.savez(
        archive,
        recurrent_alpha=np.asarray(0.1),
        recurrent_gain=np.asarray(1.0),
        target_measured_trajectory=np.ones((1, 9, 14), dtype=np.float32),
    )
    values, receipt = bridge._strict_npz_members(
        archive,
        ("recurrent_alpha", "recurrent_gain"),
        require_exact_inventory=False,
    )
    assert set(values) == {"recurrent_alpha", "recurrent_gain"}
    assert receipt["opened_member_names"] == [
        "recurrent_alpha.npy",
        "recurrent_gain.npy",
    ]
    assert "target_measured_trajectory.npy" in receipt[
        "member_streams_not_opened_by_extractor"
    ]
    assert receipt["np_load_container_used"] is False


def test_external_strict_audit_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    report = bridge.raw_stage.identity_payload(
        {
            "schema_version": 1,
            "kind": "recurrent_delta_strict_external_postrun_audit",
            "status": "strict_audit_passed",
            "expected_source_commit": bridge.CONFIRMATION_SOURCE_COMMIT,
            "auditor": {"git_commit": bridge.STRICT_AUDITOR_SOURCE_COMMIT},
            "canonical_run_root": str(root),
            "canonical_root_mutated": False,
            "report_is_outside_canonical_root": True,
            "canonical_identities": {
                "registration.json": bridge.CONFIRMATION_REGISTRATION_IDENTITY,
                "preparation.json": bridge.CONFIRMATION_PREPARATION_IDENTITY,
                "causal_trajectory_closure.json": bridge.CONFIRMATION_CLOSURE_IDENTITY,
                "analysis.json": bridge.CONFIRMATION_ANALYSIS_IDENTITY,
                "run_complete.json": bridge.CONFIRMATION_COMPLETION_IDENTITY,
            },
            "decisions": {
                "all_equal": True,
                "analysis": bridge.POSITIVE_DECISION,
                "built_in_audit": bridge.POSITIVE_DECISION,
                "recomputed": bridge.POSITIVE_DECISION,
                "run_complete": bridge.POSITIVE_DECISION,
            },
            "causal_reconstruction": {"arrays_compared": 25, "max_abs_error": 0.0},
            "built_in_audit": {
                "artifact_hashes_verified": 168,
                "causal_replay_max_abs_error": 0.0,
                "status": "audit_passed",
            },
            "explicit_false_access_flags": 5513,
        }
    )
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(report, sort_keys=True))
    monkeypatch.setattr(bridge, "STRICT_AUDIT_IDENTITY", report["identity_sha256"])
    monkeypatch.setattr(
        bridge, "STRICT_AUDIT_FILE_SHA256", bridge.raw_stage.sha256_file(path)
    )
    result = bridge.validate_strict_external_audit(path, root)
    assert result["status"] == "strict_audit_passed"
    assert result["auditor_source_commit"] == bridge.STRICT_AUDITOR_SOURCE_COMMIT
    alias = tmp_path / "strict-alias.json"
    alias.symlink_to(path)
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="symlink"):
        bridge.validate_strict_external_audit(alias, root)


def test_registration_source_must_be_executing_physical_checkout(tmp_path: Path) -> None:
    assert bridge._validate_executing_source_repo(bridge.REPO_ROOT) == bridge.REPO_ROOT
    other = tmp_path / "other-source"
    other.mkdir()
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="executing bridge checkout"):
        bridge._validate_executing_source_repo(other)
    alias = tmp_path / "source-alias"
    alias.symlink_to(bridge.REPO_ROOT, target_is_directory=True)
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="executing bridge checkout"):
        bridge._validate_executing_source_repo(alias)


def test_registered_source_revalidates_full_executing_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path("/sealed/source")
    record = {
        "path": str(repository),
        "git_commit": "a" * 40,
        "git_tree_sha": "b" * 40,
        "clean": True,
    }
    files = {"bridge": {"sha256": "c" * 64}}
    monkeypatch.setattr(
        bridge, "_validate_executing_source_repo", lambda supplied: repository
    )
    monkeypatch.setattr(
        bridge.raw_stage, "clean_repository", lambda *args, **kwargs: record
    )
    monkeypatch.setattr(bridge, "_source_records", lambda: files)
    registration = {
        "source": {
            "repository": str(repository),
            "git_commit": record["git_commit"],
            "git_tree_sha": record["git_tree_sha"],
            "clean": True,
            "files": files,
        }
    }
    bridge._validate_source(registration)
    registration["source"]["git_tree_sha"] = "d" * 40
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="source record"):
        bridge._validate_source(registration)
    registration["source"]["git_tree_sha"] = record["git_tree_sha"]
    registration["source"]["files"] = {}
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="source file"):
        bridge._validate_source(registration)


def test_cache_runtime_rebind_changes_only_helper_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_helper = tmp_path / "old" / "tools" / "physics_flow_cache_runtime.py"
    new_root = tmp_path / "new"
    new_helper = new_root / "tools" / "physics_flow_cache_runtime.py"
    old_helper.parent.mkdir(parents=True)
    new_helper.parent.mkdir(parents=True)
    old_helper.write_text("identical runtime helper\n")
    new_helper.write_text("identical runtime helper\n")
    sealed = bridge.raw_stage.identity_payload(
        {
            "schema_version": 1,
            "kind": "runtime-fixture",
            "helper_source": bridge.raw_stage.file_record(old_helper),
        }
    )
    base_registration = {
        "source_repository": {"path": str(old_helper.parents[1])},
        "cache_renderer_runtime": sealed,
    }
    monkeypatch.setattr(bridge, "REPO_ROOT", new_root)
    monkeypatch.setattr(
        bridge.raw_stage,
        "validate_cache_renderer_runtime_receipt",
        lambda receipt, source_repo=None: dict(receipt),
    )
    rebound = bridge._rebind_cache_renderer_runtime(base_registration)
    assert rebound["helper_source"] == bridge.raw_stage.file_record(new_helper)
    assert rebound["helper_source"]["sha256"] == sealed["helper_source"]["sha256"]
    assert bridge.raw_stage.identity_valid(rebound)
    new_helper.write_text("different helper\n")
    with pytest.raises(bridge.RecurrentFlowBridgeError, match="byte-identical"):
        bridge._rebind_cache_renderer_runtime(base_registration)


def test_history_reader_reads_state_prefix_and_action_slice_once(tmp_path: Path) -> None:
    rows = 100
    archive = tmp_path / "states.npz"
    np.savez(
        archive,
        joint_states=np.arange(rows * 12, dtype=np.float32).reshape(rows, 12),
        joint_actions=np.arange(rows * 12, dtype=np.float32).reshape(rows, 12) + 10,
        gripper_states=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        gripper_actions=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2) + 20,
        frame_ts=np.arange(rows, dtype=np.int64),
        instruction=np.asarray("test"),
    )
    history_rows = (2, 5, 9, 14, 20)
    measured, actions, receipt = bridge._read_history_prefix_and_actions_once(
        archive.resolve(),
        history_rows=history_rows,
        action_start=25,
        action_stop=90,
    )
    assert measured.shape == (5, 14)
    assert actions.shape == (65, 14)
    assert receipt["observed_state_array_data_bytes_returned"] == 280
    assert receipt["registered_action_array_data_bytes_returned"] == 3_640
    assert receipt["logical_candidate_action_slice_read_count"] == 1
    assert receipt["action_member_pread_count"] == 2
    assert receipt["future_measured_state_array_data_bytes_returned"] == 0


def test_inference_extraction_excludes_every_privileged_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.npz"
    arrays = {
        name: np.asarray(0.1 if name == "recurrent_alpha" else 1.0, dtype=np.float32)
        for name in bridge.INFERENCE_PREDICTOR_KEYS
    }
    arrays.update(
        {
            name: np.ones((1,), dtype=np.float32)
            for name in bridge.FORBIDDEN_SOURCE_PREDICTOR_KEYS
        }
    )
    np.savez(source, **arrays)
    monkeypatch.setattr(bridge, "FROZEN_PREDICTOR_SHA256", bridge.raw_stage.sha256_file(source))
    destination = tmp_path / "inference_only.npz"
    receipt = bridge.extract_inference_predictor(source, destination)
    assert receipt["source_archive_bytes_hashed"] is True
    assert receipt["forbidden_members_deserialized"] == []
    assert receipt["target_measured_trajectory_deserialized"] is False
    with zipfile.ZipFile(destination) as archive:
        names = {Path(name).stem for name in archive.namelist()}
    assert names == set(bridge.INFERENCE_PREDICTOR_KEYS)
    assert not names.intersection(bridge.FORBIDDEN_SOURCE_PREDICTOR_KEYS)


def test_recurrent_dataset_metadata_is_not_raw_family(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    from robot_wm.datasets.abc.recurrent_physics_flow_dataset import (
        ABCRecurrentPhysicsFlowDataset,
    )
    from robot_wm.datasets.abc.physics_flow_dataset import _canonical_identity, _sha256

    flow = tmp_path / "aligned.npy"
    np.save(flow, _valid_compact(1))
    digest = _sha256(flow)
    obj = ABCRecurrentPhysicsFlowDataset.__new__(ABCRecurrentPhysicsFlowDataset)
    obj.expected_flow_sha256 = digest
    obj.expected_confirmation_analysis_identity_sha256 = "a" * 64
    obj.expected_confirmation_completion_identity_sha256 = "b" * 64
    obj.expected_split = "train"
    obj.expected_clip_count = 1
    obj._transition_shape = (1, 8, 4, 24, 40)
    obj.flow_metadata_path = str(tmp_path / "metadata.json")
    metadata = {
        "schema": bridge.CACHE_SCHEMA,
        "complete": True,
        "split": "train",
        "clip_count": 1,
        "compact_transition_shape": [1, 8, 4, 24, 40],
        "flow_dtype": "float16",
        "aligned_flow_file": str(flow),
        "aligned_flow_sha256": digest,
        "condition_family": "sealed_recurrent_delta_geometry",
        "confirmation_decision": bridge.POSITIVE_DECISION,
        "confirmation_analysis_identity_sha256": "a" * 64,
        "confirmation_completion_identity_sha256": "b" * 64,
        "predictor_refit_or_tuning": False,
        "causal_inputs_only": True,
        "future_rgb_opened": False,
        "future_measured_state_opened": False,
        "generator_outcome_opened": False,
        "protected_test_accessed": False,
    }
    metadata["identity_sha256"] = _canonical_identity(metadata)
    obj.flow_metadata = metadata
    assert obj._validate_flow_metadata() == str(flow.resolve())
    obj.flow_metadata["condition_family"] = "raw_geometry_scaffold"
    with pytest.raises(RuntimeError, match="recurrent-flow metadata"):
        obj._validate_flow_metadata()
    obj.flow_metadata["condition_family"] = "sealed_recurrent_delta_geometry"
    alias = tmp_path / "aligned-alias.npy"
    alias.symlink_to(flow)
    obj.flow_metadata["aligned_flow_file"] = str(alias)
    obj.flow_metadata["identity_sha256"] = _canonical_identity(obj.flow_metadata)
    with pytest.raises(RuntimeError, match="symlink"):
        obj._validate_flow_metadata()


def test_model_binds_source_labels_to_family_and_off_to_zero() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "projects/latent_action_models/lam/physics_flow_model.py"
    ).read_text()
    assert 'self.condition_family == "raw_geometry_scaffold"' in source
    assert 'self.condition_family == "sealed_recurrent_delta_geometry"' in source
    assert "condition_source not in self.flow_condition_sources" in source
    assert 'condition_source == "off" and bool(condition.ne(0).any())' in source
