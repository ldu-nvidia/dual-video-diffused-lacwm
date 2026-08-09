"""End-to-end synthetic contract for the frozen IPQ-TC1 analyzer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from tools import analyze_invertible_two_clock as analysis
from tools import invertible_two_clock_evaluate as evaluation
from tools import invertible_two_clock_pilot as pilot


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_arm(root: Path, arm: pilot.Arm) -> Path:
    arm_root = root / arm.code.lower()
    arm_root.mkdir()
    endpoints = pilot.ENDPOINTS_BY_ARM[arm.code]
    rank_manifests = []
    row_count = 0
    for rank in range(8):
        rows = []
        assigned = list(range(rank, 64, 8))
        for clip in assigned:
            position = clip // 8
            batch_start = (position // 2) * 2
            for seed in pilot.NOISE_SEEDS:
                for nfe in pilot.NFE_GRID:
                    for endpoint in endpoints:
                        baseline = arm.code == "IPQ-SYNC"
                        aligned = endpoint.code == "P_LEADS_ALIGNED"
                        value = 1.0 if baseline or not aligned else 0.9
                        # NFE-1 aligned/sync/reverse schedules are one exact
                        # trajectory within the independent checkpoint.
                        if (
                            arm.code == "IPQ-INDEP"
                            and nfe == 1
                            and endpoint.code
                            in {
                                "P_LEADS_ALIGNED",
                                "SYNCHRONOUS",
                                "Q_LEADS_ALIGNED",
                            }
                        ):
                            final_hash = _hash(f"independent-nfe1-{clip}-{seed}")
                        else:
                            final_hash = _hash(
                                f"{arm.code}-{endpoint.code}-{nfe}-{clip}-{seed}"
                            )
                        paired = {
                            field: _hash(f"paired-{field}-{clip}-{seed}-{nfe}")
                            for field in (
                                "sampler_history_rgb",
                                "sampler_actions",
                                "initial_noise",
                                "clean_latent_scoring",
                                "raw_full_rgb_scoring",
                            )
                        }
                        rows.append(
                            pilot.identity_payload(
                                {
                                    "kind": evaluation.KIND_ROW,
                                    "arm": asdict(arm),
                                    "endpoint": asdict(endpoint),
                                    "clip_index": clip,
                                    "noise_seed": seed,
                                    "nfe": nfe,
                                    "metrics": {
                                        "video_future_nmse": value,
                                        "decoded_mse_unit_range": value,
                                        "decoded_temporal_difference_mse_unit_range": value,
                                        "p_future_nmse": value,
                                        "q_future_nmse": value,
                                    },
                                    "tensor_sha256": {
                                        **paired,
                                        "final_latent": final_hash,
                                    },
                                    "event_order": {
                                        "endpoint_materialized": 10,
                                        "endpoint_hashes_closed": 11,
                                        "first_target_construction": 20,
                                    },
                                    "batch_key": (
                                        f"rank-{rank:03d}-batch-{batch_start:03d}-"
                                        f"seed-{seed}-endpoint-{endpoint.code}-nfe-{nfe}"
                                    ),
                                    "latency": {
                                        "cuda_synchronized": True,
                                        "sampling_seconds_per_batch": 0.01 * nfe,
                                        "decode_seconds_per_batch": 0.005,
                                        "end_to_end_seconds_per_batch": 0.01 * nfe + 0.005,
                                        "batch_size": 2,
                                    },
                                    "peak_memory": {
                                        "allocated_bytes": 1024,
                                        "reserved_bytes": 2048,
                                    },
                                    "scoring_constructed_after_all_batch_endpoints": True,
                                    "actual_wan_calls": nfe,
                                    "clean_future_rgb_passed_to_sampler": False,
                                    "clean_future_latent_passed_to_sampler": False,
                                    "future_rgb_opened_before_global_endpoint_barrier": False,
                                    "auxiliary_target_array_opened": False,
                                    "teacher_feature_encoder_calls": 0,
                                    "protected_test_accessed": False,
                                }
                            )
                        )
        row_path = arm_root / f"rank_{rank:03d}.jsonl"
        with row_path.open("wb") as handle:
            for row in rows:
                handle.write(pilot.canonical_json(row) + b"\n")
        ledger_path = arm_root / f"ledger_{rank:03d}.json"
        pilot.exclusive_json(
            ledger_path,
            pilot.identity_payload(
                {
                    "future_rgb_opened_before_global_endpoint_barrier": False,
                    "production_abc_dataset_constructed": False,
                    "all_memmap_index_operations_recorded": True,
                    "pre_global_endpoint_barrier_operations": [],
                    "post_global_endpoint_barrier_operations": [
                        {"read_event": 3, "global_barrier_event": 2}
                    ],
                }
            ),
        )
        manifest = pilot.identity_payload(
            {
                "kind": evaluation.KIND_RANK,
                "arm": asdict(arm),
                "rows": pilot.file_record(row_path),
                "access_ledger": pilot.file_record(ledger_path),
                "target_array_opened": False,
                "teacher_feature_encoder_calls": 0,
                "protected_test_accessed": False,
                "future_rgb_opened_before_global_endpoint_barrier": False,
                "production_abc_dataset_constructed": False,
            }
        )
        rank_manifests.append(manifest)
        row_count += len(rows)
    inventory = pilot.identity_payload(
        {
            "kind": evaluation.KIND_INVENTORY,
            "arm": asdict(arm),
            "registration_identity_sha256": "registration",
            "source_commit": "c" * 40,
            "rank_manifests": rank_manifests,
            "row_count": row_count,
            "target_array_opened": False,
            "teacher_feature_encoder_calls": 0,
            "protected_test_accessed": False,
            "global_endpoint_barrier_passed": True,
            "future_rgb_opened_before_global_endpoint_barrier": False,
            "production_abc_dataset_constructed": False,
            "validation_content_rehash_after_global_endpoint_barrier": True,
            "evaluation_reserved_b200_hours": 1.0,
            "training_reserved_b200_hours_both_arms": 2.0,
            "artifact_bytes_under_registered_root": 4096,
            "peak_memory_allocated_bytes": 1024,
            "peak_memory_reserved_bytes": 2048,
        }
    )
    inventory_path = arm_root / "inventory.json"
    pilot.exclusive_json(inventory_path, inventory)
    return inventory_path


def test_analyzer_runs_complete_paired_family_and_reaches_frozen_go(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "BOOTSTRAP_SAMPLES", 64)
    sync = _write_arm(tmp_path, pilot.ARM_BY_CODE["IPQ-SYNC"])
    independent = _write_arm(tmp_path, pilot.ARM_BY_CODE["IPQ-INDEP"])
    result = analysis.analyze(sync, independent)
    assert result["decision"] == "GO_IPQ_TWO_CLOCK"
    assert result["passing_nfe"] == [1, 2, 4]
    assert result["audit"]["nfe1_order_bit_exact"] is True
    assert result["audit"]["row_counts"] == {
        "IPQ-SYNC": 768,
        "IPQ-INDEP": 4608,
    }
    assert result["resources"]["observed_reserved_b200_hours_total"] == 4.0
