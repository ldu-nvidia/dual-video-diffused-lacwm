from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest

from tools import video_residual_anchor_evaluate as evaluation


def test_endpoint_grid_is_fixed_and_action_shuffle_is_diagnostic_only() -> None:
    assert [asdict(endpoint) for endpoint in evaluation.ENDPOINTS] == [
        {
            "code": "autonomous_nfe_1",
            "nfe": 1,
            "action_source": "matched",
            "primary_gate": True,
        },
        {
            "code": "autonomous_nfe_2",
            "nfe": 2,
            "action_source": "matched",
            "primary_gate": True,
        },
        {
            "code": "autonomous_nfe_4",
            "nfe": 4,
            "action_source": "matched",
            "primary_gate": True,
        },
        {
            "code": "actions_shuffled_nfe_1",
            "nfe": 1,
            "action_source": "shuffled",
            "primary_gate": False,
        },
    ]


def test_registration_identity_roundtrip_and_tamper_detection() -> None:
    payload = evaluation.identity_payload(
        {
            "kind": evaluation.KIND_REGISTRATION,
            "fixed_protocol": {"claim_scope": "adjacent_structural_baseline"},
        }
    )
    assert evaluation.identity_valid(payload)
    payload["fixed_protocol"]["claim_scope"] = "dual_diffusion"
    assert not evaluation.identity_valid(payload)


def test_arm_run_identity_binds_coordinate_mode(tmp_path) -> None:
    commit = "a" * 40
    absolute = evaluation._arm_run_identity(tmp_path, commit, evaluation.ARMS[0])
    residual = evaluation._arm_run_identity(tmp_path, commit, evaluation.ARMS[1])
    assert absolute != residual
    assert len(absolute) == len(residual) == 64


def _descriptor(split: str, index: int, episode: int) -> dict:
    start = 100 + 100 * index
    return {
        "action_span": 65,
        "auxiliary_index": index,
        "chunk_size": 5,
        "clip_id": hashlib.sha256(f"{split}-{index}".encode()).hexdigest(),
        "episode_dir": f"/immutable/{split}/episode-{episode}",
        "frame_indices": [start + 5 * offset for offset in range(13)],
        "sample_size": 13,
        "split": split,
        "start": start,
    }


def test_manifest_validation_proves_episode_disjointness(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(
        "".join(json.dumps(_descriptor("train", i, i)) + "\n" for i in range(2))
    )
    val_path.write_text(json.dumps(_descriptor("val", 0, 7)) + "\n")
    train = evaluation._manifest_descriptors(
        train_path, expected_split="train", expected_count=2
    )
    val = evaluation._manifest_descriptors(
        val_path, expected_split="val", expected_count=1
    )
    evidence = evaluation._split_disjointness(train, val)
    assert evidence["episode_disjoint"] is True
    assert evidence["clip_id_overlap_count"] == 0

    val[0]["episode_dir"] = train[0]["episode_dir"]
    with pytest.raises(evaluation.VideoResidualAnchorEvaluationError, match="overlap"):
        evaluation._split_disjointness(train, val)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _paired_validation_rows():
    arm = evaluation.ARMS[0]
    descriptors = [_descriptor("val", index, index) for index in range(64)]
    registration = {
        "identity_sha256": "a" * 64,
        "output_root": "/immutable/output",
        "tool_repository": {"git_commit": "b" * 40},
        "validation_descriptors": descriptors,
    }
    snapshot_record = {
        "path": f"/immutable/output/training/{arm.run_name}/snapshot.pt",
        "bytes": 17,
        "sha256": "f" * 64,
    }
    donors = {}
    for rank in range(evaluation.EXPECTED_WORLD_SIZE):
        assigned = evaluation._expected_rank_indexes(rank)
        for start in range(0, len(assigned), 2):
            left, right = assigned[start : start + 2]
            donors[left], donors[right] = right, left
    rows = []
    for clip_index in range(64):
        common_hashes = {
            name: _digest(f"{name}-{clip_index}")
            for name in (
                "cached_rgb_input_sha256",
                "cached_actions_input_sha256",
                "video_clean_scoring_sha256",
                "raw_ground_truth_sha256",
                "raw_history_last_sha256",
                "video_initial_noise_sha256",
                "auxiliary_initial_noise_sha256",
            )
        }
        for endpoint in evaluation.ENDPOINTS:
            donor_index = donors[clip_index] if endpoint.action_source == "shuffled" else None
            sampler_action_hash = (
                _digest(f"cached_actions_input_sha256-{donor_index}")
                if donor_index is not None
                else common_hashes["cached_actions_input_sha256"]
            )
            tensor_hashes = {
                **common_hashes,
                "sampler_actions_sha256": sampler_action_hash,
                "generated_representation_sha256": _digest(
                    f"representation-{clip_index}-{endpoint.code}"
                ),
                "absolute_video_final_sha256": _digest(
                    f"absolute-{clip_index}-{endpoint.code}"
                ),
                "decoded_final_sha256": _digest(
                    f"decoded-{clip_index}-{endpoint.code}"
                ),
            }
            rows.append(
                evaluation.identity_payload(
                    {
                        "schema_version": evaluation.SCHEMA_VERSION,
                        "kind": evaluation.KIND_ROW,
                        "registration_identity_sha256": registration["identity_sha256"],
                        "tool_git_commit": "b" * 40,
                        "training_git_commit": evaluation.TRAINING_COMMIT,
                        "arm": asdict(arm),
                        "arm_snapshot": snapshot_record,
                        "evaluation_split": "validation",
                        "protected_test_accessed": False,
                        "clip_index": clip_index,
                        "clip_id": descriptors[clip_index]["clip_id"],
                        "sampling_id": evaluation.VALIDATION_SAMPLE_ID_OFFSET + clip_index,
                        "endpoint": asdict(endpoint),
                        "action_donor_sampling_id": (
                            None
                            if donor_index is None
                            else evaluation.VALIDATION_SAMPLE_ID_OFFSET + donor_index
                        ),
                        "clean_future_rgb_passed_to_sampler": False,
                        "clean_video_latent_passed_to_sampler": False,
                        "clean_auxiliary_passed_to_sampler": False,
                        "target_cache_array_opened": False,
                        "online_feature_or_teacher_call_count": 0,
                        "scoring_constructed_after_all_sampling": True,
                        "representation_inverted_before_decode_and_metrics": True,
                        "known_history_latents_exact": True,
                        "history_rgb_frames": 5,
                        "future_rgb_frames": 8,
                        "history_video_latent_tokens": 2,
                        "future_video_latent_tokens": 2,
                        "actual_transformer_call_count": endpoint.nfe,
                        "declared_nfe": endpoint.nfe,
                        "metrics": {
                            "video_future_nmse": 1.0,
                            "video_future_temporal_delta_nmse": 1.0,
                            "decoded_mse_unit_range": 1.0,
                            "decoded_psnr_db": 1.0,
                            "decoded_temporal_difference_mse_unit_range": 1.0,
                        },
                        "tensor_sha256": tensor_hashes,
                    }
                )
            )
    return rows, arm, registration


def test_validation_rows_bind_shuffled_actions_to_changed_donor() -> None:
    rows, arm, registration = _paired_validation_rows()
    evaluation._validate_rows(rows, arm, registration)
    target_index = next(
        index
        for index, row in enumerate(rows)
        if row["endpoint"]["code"] == "actions_shuffled_nfe_1"
    )
    unsigned = dict(rows[target_index])
    unsigned.pop("identity_sha256")
    unsigned["tensor_sha256"] = dict(unsigned["tensor_sha256"])
    unsigned["tensor_sha256"]["sampler_actions_sha256"] = unsigned[
        "tensor_sha256"
    ]["cached_actions_input_sha256"]
    rows[target_index] = evaluation.identity_payload(unsigned)
    with pytest.raises(
        evaluation.VideoResidualAnchorEvaluationError,
        match="episode-disjoint donor input",
    ):
        evaluation._validate_rows(rows, arm, registration)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scoring_constructed_after_all_sampling", False),
        ("history_rgb_frames", 4),
        ("future_rgb_frames", 9),
        ("schema_version", evaluation.SCHEMA_VERSION + 1),
    ),
)
def test_validation_rows_enforce_causal_boundary_and_rgb_geometry(
    field: str, value: object
) -> None:
    rows, arm, registration = _paired_validation_rows()
    unsigned = dict(rows[0])
    unsigned.pop("identity_sha256")
    unsigned[field] = value
    rows[0] = evaluation.identity_payload(unsigned)
    with pytest.raises(
        evaluation.VideoResidualAnchorEvaluationError,
        match="validation row violates protocol",
    ):
        evaluation._validate_rows(rows, arm, registration)


def test_rank_receipt_binds_rows_and_transformer_calls() -> None:
    arm = evaluation.ARMS[0]
    rank = 0
    rows_record = {"path": "/tmp/rank.jsonl", "bytes": 17, "sha256": "c" * 64}
    receipt = evaluation.identity_payload(
        {
            "schema_version": evaluation.SCHEMA_VERSION,
            "kind": evaluation.KIND_RANK,
            "arm": asdict(arm),
            "rank": rank,
            "world_size": evaluation.EXPECTED_WORLD_SIZE,
            "indexes": evaluation._expected_rank_indexes(rank),
            "rows": 32,
            "rows_file": rows_record,
            "transformer_calls": 32,
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        }
    )
    evaluation._validate_rank_receipt_payload(
        receipt, rank=rank, arm=arm, rows_record=rows_record
    )


def test_training_trace_receipt_requires_all_paired_updates(tmp_path) -> None:
    arm = evaluation.ARMS[0]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / "snapshot.pt"
    parent = tmp_path / "parent.pt"
    snapshot.write_bytes(b"trained")
    parent.write_bytes(b"parent")
    run_identity = "d" * 64
    registration = {
        "identity_sha256": "e" * 64,
        "arm_run_identity_sha256": {arm.code: run_identity},
        "controlled_study": {"parent_snapshot": {"path": str(parent)}},
    }
    header = {
        "kind": "video_residual_anchor_training_trace_header",
        "arm": arm.code,
        "representation_mode": arm.representation_mode,
        "normalization": "none",
        "known_history_policy": "exact_history_only_vae_tokens",
        "parent_snapshot": str(parent),
        "parent_snapshot_sha256": evaluation.PARENT_SNAPSHOT_SHA256,
        "parent_run_identity_sha256": "f" * 64,
        "parent_completed_updates": 1000,
        "continuation_updates": 200,
        "optimizer_state_policy": "fresh_identical_adamw",
        "ema_policy": "none_in_parent_and_none_in_both_arms",
        "auxiliary_feature_access": False,
    }
    rows = [header]
    audit = {
        "train_loss/paired_audit/clip_index_mean": 1.0,
        "train_loss/paired_audit/clip_index_square_mean": 1.0,
        "train_loss/paired_audit/timestep_mean": 1.0,
        "train_loss/paired_audit/timestep_square_mean": 1.0,
        "train_loss/paired_audit/noise_probe": 1.0,
    }
    for iteration in range(200):
        observations = (iteration + 1) * 8
        rows.append(
            {
                "kind": "video_residual_anchor_training_trace_event",
                "arm": arm.code,
                "total_observations": observations,
                "metrics": {
                    "iteration": iteration,
                    "total_observations": observations,
                    "train_loss/loss": 1.0,
                    **audit,
                },
            }
        )
        if iteration in {0, 100, 199}:
            rows.append(
                {
                    "kind": "video_residual_anchor_training_trace_event",
                    "arm": arm.code,
                    "total_observations": observations,
                    "metrics": {
                        "iteration": iteration,
                        "total_observations": observations,
                        "val_loss/ABC/loss": 1.0,
                    },
                }
            )
    trace = run_dir / "video_residual_anchor_training_trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "video_residual_anchor_training_trace_complete.json").write_text(
        json.dumps(
            {
                "kind": "video_residual_anchor_training_trace_complete",
                "arm": arm.code,
                "rows": len(rows),
                "trace_sha256": evaluation._sha256(trace),
                "completed_updates": 200,
                "protected_test_accessed": False,
            }
        )
    )
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "completed_updates": 200,
                "max_iter": 200,
                "run_identity_sha256": run_identity,
                "snapshot": str(snapshot),
            }
        )
    )
    receipt = evaluation._validate_training_trace(run_dir, arm, registration)
    assert receipt["train_update_events"] == 200
    assert receipt["validation_events"] == [0, 100, 199]
