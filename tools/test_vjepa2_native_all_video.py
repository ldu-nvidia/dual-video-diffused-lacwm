"""Contract tests for the post-study native-all-video evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vjepa2_native_all_video as native


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tools" / "slurm" / "vjepa2_native_all_video.sbatch"
SUBMIT = (
    ROOT / "tools" / "slurm" / "submit_vjepa2_native_all_video.sh"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
IDENTITY_STUDY = "1" * 64


def _arm_inputs() -> dict[str, native.ArmInput]:
    result = {}
    for index, code in enumerate(native.ARMS, 2):
        token = str(index) * 64
        result[code] = native.ArmInput(
            code=code,
            run_dir=Path(f"/immutable/{code}"),
            arm_manifest_path=Path(f"/immutable/{code}/arm_manifest.json"),
            arm_manifest={
                "identity_sha256": token,
                "study_identity_sha256": IDENTITY_STUDY,
            },
            stage_manifest_path=Path(
                f"/immutable/{code}/stage_manifest_update_1000.json"
            ),
            stage_manifest={"identity_sha256": str(index + 3) * 64},
            stage_outcome_path=Path(
                f"/immutable/{code}/stage_outcome_update_1000.json"
            ),
            stage_outcome={"identity_sha256": str(index + 4) * 64},
            resolved_config=Path(f"/immutable/{code}/resolved_update_1000.yaml"),
            snapshot=Path(f"/immutable/{code}/snapshot.pt"),
            snapshot_sha256=str(index + 5) * 64,
        )
    return result


def _split(expected_clips: int = 2) -> native.SplitInput:
    return native.SplitInput(
        name="validation",
        expected_clips=expected_clips,
        dataset_config_key="val_dataset",
        manifest=Path("/immutable/validation.jsonl"),
        cache_metadata=Path("/immutable/validation-cache.json"),
        cache_arrays={},
        descriptors=tuple(
            {"clip_id": f"clip-{index}", "auxiliary_index": index}
            for index in range(expected_clips)
        ),
        lockbox_registration=None,
        validation_report=None,
    )


def _linear_schedule(nfe: int) -> dict[str, object]:
    nodes = [1.0 - index / nfe for index in range(nfe + 1)]
    return {
        "video_sigma_nodes": nodes,
        "auxiliary_sigma_nodes": list(nodes),
        "model_timesteps": [float(nfe - index) for index in range(nfe)],
        "video_sigma_sha256": SHA_A,
        "auxiliary_sigma_sha256": SHA_A,
        "model_timesteps_sha256": SHA_A,
        "tf_only_steps": 0,
    }


def _valid_native_counter(nfe: int) -> dict[str, object]:
    return {
        "actual_wan_calls": nfe,
        "actual_video_scheduler_calls": nfe,
        "actual_auxiliary_euler_calls": nfe,
        "actual_auxiliary_nonzero_sigma_transitions": nfe,
        "schedule": _linear_schedule(nfe),
    }


def _valid_rows(
    *,
    expected_clips: int = 2,
) -> tuple[
    list[dict[str, object]],
    dict[str, native.ArmInput],
    native.SplitInput,
]:
    arm_inputs = _arm_inputs()
    split = _split(expected_clips)
    hash_fields = {
        "video_clean_sha256",
        "auxiliary_clean_sha256",
        "ground_truth_sha256",
        "vae_ground_truth_sha256",
        "raw_history_last_sha256",
        "vae_history_last_sha256",
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_initial_state_sha256",
        "auxiliary_initial_state_sha256",
        "auxiliary_initial_noise_sha256",
        "video_final_sha256",
        "auxiliary_final_sha256",
        "decoded_final_sha256",
        "reference_latents_sha256",
    }
    rows: list[dict[str, object]] = []
    for arm in native.ARMS:
        arm_input = arm_inputs[arm]
        for source in native.RUNTIME_SOURCES[arm]:
            for nfe in native.NFE_GRID:
                for clip_index in range(expected_clips):
                    hashes = {field: SHA_A for field in hash_fields}
                    payload = {
                        "schema_version": native.SCHEMA_VERSION,
                        "kind": native.KIND_ROW,
                        "split": "validation",
                        "arm_code": arm,
                        "runtime_source": source,
                        "sampler_condition_source": native.SAMPLER_SOURCE[source],
                        "nfe": nfe,
                        "clip_index": clip_index,
                        "clip_id": f"clip-{clip_index}",
                        "sampling_id": (
                            native.SAMPLE_ID_OFFSETS["validation"] + clip_index
                        ),
                        "completed_updates": native.FINAL_UPDATE,
                        "study_identity_sha256": IDENTITY_STUDY,
                        "arm_identity_sha256": arm_input.arm_manifest[
                            "identity_sha256"
                        ],
                        "stage_identity_sha256": arm_input.stage_manifest[
                            "identity_sha256"
                        ],
                        "stage_outcome_identity_sha256": arm_input.stage_outcome[
                            "identity_sha256"
                        ],
                        "snapshot_sha256": arm_input.snapshot_sha256,
                        "training_git_commit": native.TRAINING_COMMIT,
                        "evaluator_git_commit": "f" * 40,
                        "inference_code_compatibility_sha256": SHA_B,
                        "videox_runtime_identity_sha256": SHA_C,
                        "evaluation_world_size": native.EXPECTED_WORLD_SIZE,
                        "evaluation_batch_size_per_rank": 2,
                        "lockbox_registration_identity_sha256": None,
                        "runtime_intervention": {
                            "schedule_mode": "aligned_native_all_video",
                            "all_wan_calls_advance_video": True,
                            "generated_auxiliary_state": True,
                            "auxiliary_state_injected_into_video": (
                                source == "aligned"
                            ),
                            "auxiliary_clock_injected_into_video": (
                                source == "aligned"
                            ),
                            "model_and_dataset_source_modified": False,
                        },
                        "actual_call_counts": {
                            "wan": nfe,
                            "video_scheduler": nfe,
                            "auxiliary_euler": nfe,
                            "auxiliary_nonzero_sigma_transitions": nfe,
                            "online_teacher": 0,
                        },
                        "schedule": _linear_schedule(nfe),
                        "clean_future_or_auxiliary_passed_to_sampler": False,
                        "oracle_leakage": False,
                        "deployable_evidence": True,
                        "effective_state_gate": (
                            0.02 if arm == "J1" else 0.0
                        ),
                        "effective_clock_gate": (
                            0.02 if arm == "J1" else 0.0
                        ),
                        "metrics": {
                            "video_future_nmse": 1.0,
                            "decoded_mse_unit_range": 1.0,
                            "decoded_temporal_difference_mse_unit_range": 1.0,
                            "decoded_psnr_db": 1.0,
                            "auxiliary_future_nmse": 1.0,
                            "auxiliary_future_cosine_similarity": 0.0,
                        },
                        "tensor_sha256": hashes,
                    }
                    rows.append(native._identity_payload(payload))
    return rows, arm_inputs, split


def test_fixed_scientific_contract():
    assert native.ARMS == ("VPM", "A1", "J1")
    assert native.RUNTIME_SOURCES == {
        "VPM": ("off",),
        "A1": ("off",),
        "J1": ("aligned", "off"),
    }
    assert native.NFE_GRID == (1, 2, 4, 6, 8, 12, 20)
    assert native.ENDPOINT_AUDIT_K == (1, 2, 4)
    assert native.PRIMARY_CLAIM_NFE == 4
    assert native.BOOTSTRAP_SAMPLES == 10_000
    assert native.BOOTSTRAP_SEED == 1234
    assert set(native.COMPOSITE_COMPONENTS) == {
        "training_objective",
        "generated_feature_use",
        "end_to_end",
    }


def test_native_and_cascade_counter_contracts():
    native._validate_native_counter(_valid_native_counter(2), nfe=2)
    cascade = {
        "actual_wan_calls": 4,
        "actual_video_scheduler_calls": 2,
        "actual_auxiliary_euler_calls": 4,
        "actual_auxiliary_nonzero_sigma_transitions": 2,
        "schedule": {
            "video_sigma_nodes": [1.0, 1.0, 1.0, 0.5, 0.0],
            "auxiliary_sigma_nodes": [1.0, 0.5, 0.0, 0.0, 0.0],
            "tf_only_steps": 2,
        },
    }
    native._validate_cascade_counter(cascade, native_nfe=2)
    broken = dict(_valid_native_counter(2))
    broken["actual_video_scheduler_calls"] = 1
    with pytest.raises(native.NativeAllVideoError):
        native._validate_native_counter(broken, nfe=2)


def test_expected_actual_call_totals_include_endpoint_audits():
    assert native._expected_rank_call_counts(
        split_name="validation", batches=1
    ) == {
        "wan": 254,
        "video_scheduler": 233,
        "auxiliary_euler": 254,
        "auxiliary_nonzero_sigma_transitions": 233,
        "online_teacher": 0,
    }
    assert native._expected_rank_call_counts(
        split_name="lockbox", batches=1
    ) == {
        "wan": 212,
        "video_scheduler": 212,
        "auxiliary_euler": 212,
        "auxiliary_nonzero_sigma_transitions": 212,
        "online_teacher": 0,
    }


def test_sampling_instrumentation_counts_real_methods(monkeypatch):
    callbacks = []

    class Handle:
        @staticmethod
        def remove():
            callbacks.clear()

    class Forward:
        @staticmethod
        def register_forward_hook(callback):
            callbacks.append(callback)
            return Handle()

        @staticmethod
        def invoke():
            for callback in callbacks:
                callback(None, None, None)

    class Scheduler:
        @staticmethod
        def step(*_args, **_kwargs):
            return "video"

    module = SimpleNamespace(
        euler_flow_step=lambda state, _velocity, _sigma, _next: state
    )
    model = SimpleNamespace()
    model.forward_model = Forward()
    model.sample_scheduler = Scheduler()
    model._sampling_schedule = lambda *_args, **_kwargs: (
        SimpleNamespace(
            video=torch.tensor([1.0, 0.5, 0.0]),
            time_frequency=torch.tensor([1.0, 0.5, 0.0]),
        ),
        torch.tensor([2.0, 1.0]),
        0,
    )
    model._sample_future = lambda: None
    monkeypatch.setattr(native.inspect, "getmodule", lambda _value: module)
    with native.SamplingInstrumentation(model) as instrumentation:
        model._sampling_schedule(2, device="cpu")
        model.forward_model.invoke()
        model.forward_model.invoke()
        model.sample_scheduler.step(None)
        model.sample_scheduler.step(None)
        module.euler_flow_step(
            torch.zeros(1), torch.zeros(1), torch.tensor(1.0), torch.tensor(0.5)
        )
        module.euler_flow_step(
            torch.zeros(1), torch.zeros(1), torch.tensor(0.5), torch.tensor(0.0)
        )
        record = instrumentation.record()
    assert record["actual_wan_calls"] == 2
    assert record["actual_video_scheduler_calls"] == 2
    assert record["actual_auxiliary_euler_calls"] == 2
    assert record["actual_auxiliary_nonzero_sigma_transitions"] == 2
    native._validate_native_counter(record, nfe=2)


def test_lockbox_path_is_not_inspected_before_validation_pass(
    tmp_path, monkeypatch
):
    report = native._identity_payload(
        {
            "kind": native.KIND_REPORT,
            "split": "validation",
            "complete": True,
            "lockbox_eligible": False,
            "lockbox_inspected": False,
            "study_identity_sha256": IDENTITY_STUDY,
            "training_git_commit": native.TRAINING_COMMIT,
            "evaluator_git_commit": "f" * 40,
            "nfe_grid": list(native.NFE_GRID),
            "primary_claim_nfe": native.PRIMARY_CLAIM_NFE,
            "primary_composite_gate": {"passed": False},
            "endpoint_equivalence_audit": {"passed": True},
        }
    )
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    original = native._canonical_file
    labels = []

    def observed(value, label):
        labels.append(label)
        if "lockbox registration" in label:
            raise AssertionError("lockbox path was inspected")
        return original(value, label)

    monkeypatch.setattr(native, "_canonical_file", observed)
    with pytest.raises(
        native.NativeAllVideoError,
        match="not eligible native-all-video evidence",
    ):
        native._resolve_split(
            split_name="lockbox",
            study={"identity_sha256": IDENTITY_STUDY},
            validation_report_value=report_path,
            lockbox_registration_value=tmp_path / "must-not-be-opened.json",
            training_commit=native.TRAINING_COMMIT,
            evaluator_commit="f" * 40,
            verify_cache_arrays=True,
        )
    assert labels == ["validation report"]


def test_registered_lockbox_cache_arrays_are_normalized_after_gate():
    registration = {
        "manifest": {"path": "/lockbox/manifest.jsonl"},
        "cache": {
            "metadata": {"path": "/lockbox/cache.json"},
            "arrays": {
                "target": {"path": "/lockbox/target.npy"},
                "rgb": {"path": "/lockbox/rgb.npy"},
                "actions": {"path": "/lockbox/actions.npy"},
            },
        },
    }
    split = native._registered_lockbox_split_record(registration)
    assert set(split["cache"]) == {
        "metadata",
        "target",
        "rgb",
        "actions",
    }
    assert split["cache"]["target"] == registration["cache"]["arrays"]["target"]
    malformed = {
        **registration,
        "cache": {
            **registration["cache"],
            "arrays": {"target": {}, "rgb": {}},
        },
    }
    with pytest.raises(native.NativeAllVideoError):
        native._registered_lockbox_split_record(malformed)


def test_rank_zero_creates_output_before_all_ranks_canonicalize(tmp_path):
    repo = tmp_path / "repo"
    study = tmp_path / "study"
    output_parent = tmp_path / "outputs"
    repo.mkdir()
    study.mkdir()
    output_parent.mkdir()
    output = output_parent / "evaluation"

    # Rank zero alone enforces freshness and creates the directory.
    native._create_fresh_output_directory(
        output,
        repo=repo.resolve(),
        study_root=study.resolve(),
    )

    # Every rank, including a late nonzero rank, validates the created path.
    assert native._canonical_evaluation_output_directory(
        output,
        repo=repo.resolve(),
        study_root=study.resolve(),
    ) == output.resolve()
    assert native._canonical_evaluation_output_directory(
        output,
        repo=repo.resolve(),
        study_root=study.resolve(),
    ) == output.resolve()
    with pytest.raises(native.NativeAllVideoError):
        native._create_fresh_output_directory(
            output,
            repo=repo.resolve(),
            study_root=study.resolve(),
        )
    inside_repo = repo / "existing-output"
    inside_repo.mkdir()
    with pytest.raises(
        native.NativeAllVideoError,
        match="outside evaluator and immutable study trees",
    ):
        native._canonical_evaluation_output_directory(
            inside_repo,
            repo=repo.resolve(),
            study_root=study.resolve(),
        )


def test_study_manifest_directly_binds_training_commit(tmp_path):
    study_root = tmp_path / "study"
    study_root.mkdir()
    payload = {
        "kind": "vjepa2_controlled_video_diffusion_study",
        "inputs": {
            "repository": {
                "git_commit": native.TRAINING_COMMIT,
            }
        },
        "arms": {
            code: {
                "code": code,
                "name": native.ARM_NAMES[code],
            }
            for code in native.ARMS
        },
        "inference": {
            "quality_protocol": {
                "deployable_sources_use_history_only_public_entrypoint": True,
                "temporal_metric_includes_history_to_first_future_boundary": True,
            }
        },
        "clock": {"convention": "sigma=1 noise, sigma=0 clean"},
    }
    path = study_root / "study_manifest.json"
    path.write_text(
        json.dumps(native._identity_payload(payload)),
        encoding="utf-8",
    )
    _path, observed = native._validate_study(
        study_root.resolve(),
        training_commit=native.TRAINING_COMMIT,
    )
    assert observed["inputs"]["repository"]["git_commit"] == native.TRAINING_COMMIT

    payload["inputs"]["repository"]["git_commit"] = "0" * 40
    path.write_text(
        json.dumps(native._identity_payload(payload)),
        encoding="utf-8",
    )
    with pytest.raises(native.NativeAllVideoError, match="identity differs"):
        native._validate_study(
            study_root.resolve(),
            training_commit=native.TRAINING_COMMIT,
        )


def test_global_rows_reconstruct_provenance_and_reject_leakage():
    rows, arm_inputs, split = _valid_rows()
    lookup = native._validate_global_rows(
        rows,
        split=split,
        arm_inputs=arm_inputs,
        study_identity_sha256=IDENTITY_STUDY,
        training_commit=native.TRAINING_COMMIT,
        evaluator_commit="f" * 40,
        inference_compatibility_sha256=SHA_B,
        videox_runtime_identity_sha256=SHA_C,
        world_size=native.EXPECTED_WORLD_SIZE,
        batch_size_per_rank=2,
    )
    assert len(lookup) == 2 * 28
    tampered = [dict(value) for value in rows]
    tampered[0].pop("identity_sha256")
    tampered[0]["clean_future_or_auxiliary_passed_to_sampler"] = True
    tampered[0] = native._identity_payload(tampered[0])
    with pytest.raises(native.NativeAllVideoError):
        native._validate_global_rows(
            tampered,
            split=split,
            arm_inputs=arm_inputs,
            study_identity_sha256=IDENTITY_STUDY,
            training_commit=native.TRAINING_COMMIT,
            evaluator_commit="f" * 40,
            inference_compatibility_sha256=SHA_B,
            videox_runtime_identity_sha256=SHA_C,
            world_size=native.EXPECTED_WORLD_SIZE,
            batch_size_per_rank=2,
        )


def test_endpoint_audit_is_reconstructed_from_exact_native_and_cascade_hashes():
    rows, arm_inputs, split = _valid_rows()
    lookup = native._validate_global_rows(
        rows,
        split=split,
        arm_inputs=arm_inputs,
        study_identity_sha256=IDENTITY_STUDY,
        training_commit=native.TRAINING_COMMIT,
        evaluator_commit="f" * 40,
        inference_compatibility_sha256=SHA_B,
        videox_runtime_identity_sha256=SHA_C,
        world_size=native.EXPECTED_WORLD_SIZE,
        batch_size_per_rank=2,
    )
    hash_names = {
        "video_initial_state_sha256",
        "auxiliary_initial_noise_sha256",
        "reference_latents_sha256",
        "video_final_sha256",
        "decoded_final_sha256",
    }
    check_names = {
        *hash_names,
        "video_phase_schedule_equal",
        "native_wan_calls_equal_k",
        "cascade_wan_calls_equal_2k",
        "native_and_cascade_video_updates_equal_k",
        "online_teacher_calls_zero",
        "clean_future_or_auxiliary_not_passed",
    }
    audits = []
    for arm in native.ARMS:
        arm_input = arm_inputs[arm]
        for k in native.ENDPOINT_AUDIT_K:
            cascade_schedule = {
                "video_sigma_nodes": (
                    [1.0] * k
                    + [1.0 - index / k for index in range(k + 1)]
                ),
                "auxiliary_sigma_nodes": (
                    [1.0 - index / k for index in range(k + 1)]
                    + [0.0] * k
                ),
                "model_timesteps": [float(2 * k - index) for index in range(2 * k)],
                "video_sigma_sha256": SHA_A,
                "auxiliary_sigma_sha256": SHA_A,
                "model_timesteps_sha256": SHA_A,
                "tf_only_steps": k,
            }
            for clip_index in range(split.expected_clips):
                native_row = lookup[(arm, "off", k, clip_index)]
                hashes = {
                    name: native_row["tensor_sha256"][name]
                    for name in hash_names
                }
                audits.append(
                    native._identity_payload(
                        {
                            "schema_version": native.SCHEMA_VERSION,
                            "kind": native.KIND_ENDPOINT_AUDIT,
                            "split": "validation",
                            "arm_code": arm,
                            "native_nfe_k": k,
                            "cascade_nfe_2k": 2 * k,
                            "clip_index": clip_index,
                            "clip_id": native_row["clip_id"],
                            "sampling_id": native_row["sampling_id"],
                            "study_identity_sha256": IDENTITY_STUDY,
                            "arm_identity_sha256": arm_input.arm_manifest[
                                "identity_sha256"
                            ],
                            "stage_identity_sha256": arm_input.stage_manifest[
                                "identity_sha256"
                            ],
                            "snapshot_sha256": arm_input.snapshot_sha256,
                            "training_git_commit": native.TRAINING_COMMIT,
                            "evaluator_git_commit": "f" * 40,
                            "native_runtime_source": "off",
                            "native_schedule_mode": "aligned_native_all_video",
                            "cascade_schedule_mode": "tf_first_cascaded",
                            "native_actual_call_counts": {
                                "wan": k,
                                "video_scheduler": k,
                                "auxiliary_euler": k,
                                "auxiliary_nonzero_sigma_transitions": k,
                                "online_teacher": 0,
                            },
                            "native_schedule": native_row["schedule"],
                            "native_tensor_sha256": hashes,
                            "cascade_tensor_sha256": dict(hashes),
                            "checks": {name: True for name in check_names},
                            "passed": True,
                            "cascade_actual_call_counts": {
                                "wan": 2 * k,
                                "video_scheduler": k,
                                "auxiliary_euler": 2 * k,
                                "auxiliary_nonzero_sigma_transitions": k,
                                "online_teacher": 0,
                            },
                            "cascade_schedule": cascade_schedule,
                        }
                    )
                )
    summary = native._validate_endpoint_audits(
        audits,
        split=split,
        row_lookup=lookup,
        arm_inputs=arm_inputs,
        study_identity_sha256=IDENTITY_STUDY,
        training_commit=native.TRAINING_COMMIT,
        evaluator_commit="f" * 40,
    )
    assert summary["passed"] is True
    assert summary["record_count"] == 18
    tampered = [dict(value) for value in audits]
    tampered[0].pop("identity_sha256")
    tampered[0]["cascade_tensor_sha256"] = dict(
        tampered[0]["cascade_tensor_sha256"]
    )
    tampered[0]["cascade_tensor_sha256"]["video_final_sha256"] = SHA_B
    tampered[0] = native._identity_payload(tampered[0])
    with pytest.raises(native.NativeAllVideoError):
        native._validate_endpoint_audits(
            tampered,
            split=split,
            row_lookup=lookup,
            arm_inputs=arm_inputs,
            study_identity_sha256=IDENTITY_STUDY,
            training_commit=native.TRAINING_COMMIT,
            evaluator_commit="f" * 40,
        )
    tampered_counts = [dict(value) for value in audits]
    tampered_counts[0].pop("identity_sha256")
    tampered_counts[0]["native_actual_call_counts"] = dict(
        tampered_counts[0]["native_actual_call_counts"]
    )
    tampered_counts[0]["native_actual_call_counts"]["wan"] += 1
    tampered_counts[0] = native._identity_payload(tampered_counts[0])
    with pytest.raises(native.NativeAllVideoError):
        native._validate_endpoint_audits(
            tampered_counts,
            split=split,
            row_lookup=lookup,
            arm_inputs=arm_inputs,
            study_identity_sha256=IDENTITY_STUDY,
            training_commit=native.TRAINING_COMMIT,
            evaluator_commit="f" * 40,
        )


def test_canonical_paired_comparison_positive_means_left_better():
    lookup = {}
    for clip_index in range(4):
        common = {
            "clip_id": f"clip-{clip_index}",
            "sampling_id": 100 + clip_index,
        }
        lookup[("A1", "off", 4, clip_index)] = {
            **common,
            "metrics": {metric: 0.8 for metric in native.CLAIM_METRICS},
        }
        lookup[("VPM", "off", 4, clip_index)] = {
            **common,
            "metrics": {metric: 1.0 for metric in native.CLAIM_METRICS},
        }
    comparison = native._paired_comparison(
        lookup,
        split="validation",
        expected_clips=4,
        left_arm="A1",
        left_source="off",
        reference_arm="VPM",
        reference_source="off",
        nfe=4,
        label_prefix="unit",
    )
    assert comparison["same_nfe_attribution_gate"]["passed"] is True
    for metric in native.CLAIM_METRICS:
        effect = comparison["metrics"][metric]
        assert effect["relative_improvement"] == pytest.approx(0.2)
        assert effect["bootstrap_ci"]["low"] == pytest.approx(0.2)


def test_composite_gate_requires_three_material_fixed_k4_effects():
    gate = {
        "passed": True,
        "checks": {
            "temporal_ci_low_strictly_positive": True,
            "video_nmse_ci_low_above_minus_one_percent": True,
            "decoded_mse_ci_low_above_minus_one_percent": True,
            "temporal_ci_low_at_least_three_percent": True,
        },
        "ci_lows": {
            native.PRIMARY_METRIC: 0.04,
            "video_future_nmse": 0.0,
            "decoded_mse_unit_range": 0.0,
        },
        "rule": "unit",
    }
    comparisons = {
        name: {
            "left": definition["left"],
            "reference": definition["reference"],
            "claim_nfe": 4,
            "metrics": {},
            "same_nfe_attribution_gate": dict(gate),
        }
        for name, definition in native.COMPOSITE_COMPONENTS.items()
    }
    composite = native._build_composite_gate(comparisons)
    assert composite["passed"] is True
    native._validate_passed_composite_gate(composite)
    broken = {
        name: dict(value) for name, value in comparisons.items()
    }
    broken_gate = {
        **gate,
        "checks": dict(gate["checks"]),
        "ci_lows": dict(gate["ci_lows"]),
    }
    broken_gate["ci_lows"][native.PRIMARY_METRIC] = 0.029
    broken["generated_feature_use"] = {
        **broken["generated_feature_use"],
        "same_nfe_attribution_gate": broken_gate,
    }
    with pytest.raises(native.NativeAllVideoError):
        native._build_composite_gate(broken)


def test_tool_is_evaluation_only_and_uses_public_deployable_sampler():
    source = Path(native.__file__).read_text(encoding="utf-8")
    assert "sample_future_deployable(" in source
    assert "optimizer.step(" not in source
    assert "torch.save(" not in source
    assert "model.tf_schedule_mode = schedule_mode" in source
    assert "clean_future_or_auxiliary_passed_to_sampler" in source
    assert "online_teacher_call_count" in source


def test_slurm_wrapper_is_eight_rank_and_has_no_implicit_scheduler_routing():
    source = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gpus-per-node=8" in source
    assert "--nproc_per_node=8" in source
    for forbidden in (
        "#SBATCH --account",
        "#SBATCH --qos",
        "#SBATCH --partition",
        "#SBATCH --output",
        "#SBATCH --error",
    ):
        assert forbidden not in source
    assert "check-validation-gate" in source
    assert source.index("check-validation-gate") < source.index(
        '[[ -s "$LOCKBOX_REGISTRATION"'
    )


def test_submitter_is_dry_run_first_and_requires_explicit_outputs_and_routing():
    source = SUBMIT.read_text(encoding="utf-8")
    assert 'ACCOUNT=""' in source
    assert 'QOS=""' in source
    assert 'PARTITION=""' in source
    assert 'OUTPUT_DIR=""' in source
    assert 'LOG_DIR=""' in source
    assert 'EXECUTE=0' in source
    assert "--account=" in source
    assert "--qos=" in source
    assert "--partition=" in source
    assert "--output=" in source
    assert "--error=" in source
    assert source.index("check-validation-gate") < source.index(
        '[[ -s "$LOCKBOX_REGISTRATION"'
    )
