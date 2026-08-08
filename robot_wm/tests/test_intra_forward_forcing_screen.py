from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tools import intra_forward_forcing_screen as screen
from tools import validate_intra_forward_warmstart as preflight


def _registration(study_root: Path) -> dict:
    return screen.identity_payload(
        {
            "schema": screen.SCHEMA,
            "study_id": study_root.name,
            "study_root": str(study_root),
            "source": {"commit": "1" * 40},
            "runtime": {
                "python": {"path": "/usr/bin/python3"},
                "wan_dir": "/tmp/wan",
                "videox_home": "/tmp/videox",
            },
            "warm_start": {"sha256": "2" * 64},
            "protected_test": {"allowed": False},
            "architecture": {
                "wan_block_count": 30,
                "midpoint_block_index": 14,
                "midpoint_head_calls_per_wan_call": 1,
                "additional_wan_calls": 0,
                "generated_clean_formula": "q0_hat=q_sigma-sigma*v_hat",
                "generated_clean_stop_gradient": True,
                "history_preserved_future_shuffle": True,
                "auxiliary_history_bins": 2,
                "input_level_auxiliary_state_residual": "exact_zero",
                "input_level_auxiliary_clock_residual": "exact_zero",
                "gradient_checkpointing": False,
            },
            "training": {
                "validation_iterator": dict(screen.TRAIN_VALIDATION_CONTRACT),
                "update_zero_memory_smoke_required": True,
                "arms": [asdict(arm) for arm in screen.ARMS],
            },
            "evaluation": {
                "split": "validation",
                "clips": 64,
                "nfe": [1, 2, 4],
                "sources": list(screen.SOURCES),
                "deployable_sources": list(screen.DEPLOYABLE_SOURCES),
                "sample_keyed_noise_seed": screen.EVALUATION_SEED,
                "teacher_calls": 0,
                "clean_future_feature_available_to_deployable_sampler": False,
                "oracle_sources": [],
                "timing": {
                    "cuda_synchronize_before_and_after": True,
                    "records_end_to_end_and_peak_memory": True,
                    "records_midpoint_head_elapsed": True,
                    "artifact_audit_batch_size": 2,
                    "endpoint_timing_batch_size": 1,
                    "generations_per_two_clip_cell": 3,
                    "latency_claim_scope": "descriptive_equal_nfe_only",
                },
            },
            "wandb": {
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
            },
        }
    )


def _write_memory_smoke(study_root: Path, registration: dict) -> dict:
    runtime = study_root / "memory_smoke_runtime.json"
    runtime.write_text("{}\n", encoding="utf-8")
    ranks = [
        {
            "rank": rank,
            "local_rank": rank,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 120,
            "total_memory_bytes": 1000,
            "finite_loss": True,
            "finite_gradients": True,
        }
        for rank in range(screen.WORLD_SIZE)
    ]
    receipt = screen.identity_payload(
        {
            "schema": screen.MEMORY_SMOKE_SCHEMA,
            "status": "pass",
            "registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["source"]["commit"],
            "selector": screen.ARMS[1].selector,
            "world_size": screen.WORLD_SIZE,
            "synthetic_batch_shapes": {
                "rgb": [1, 13, 3, 180, 960],
                "actions": [1, 13, 5, 157],
                "mask": [1, 13],
                "morphology_index": [1],
                "clip_index": [1],
            },
            "gradient_checkpointing": False,
            "forward_completed": True,
            "backward_completed": True,
            "optimizer_step_executed": False,
            "completed_optimizer_updates": 0,
            "optimizer_state_entries": 0,
            "runtime_receipt_sha256": screen.sha256_file(runtime),
            "ranks": ranks,
            "maximum_peak_allocated_bytes": 100,
            "minimum_headroom_bytes": 900,
            "scientific_metrics_emitted": False,
        }
    )
    (study_root / "memory_smoke.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return receipt


def test_frozen_arms_differ_only_by_midpoint_injection_and_have_no_oracle():
    assert [arm.code for arm in screen.ARMS] == ["MID-OFF", "MID-ON"]
    off, on = screen.ARMS
    assert not off.midpoint_injection
    assert on.midpoint_injection
    ignored = {
        "code",
        "slug",
        "selector",
        "condition_mode",
        "condition_on_state",
        "midpoint_injection",
        "estimand",
    }
    off_common = {
        key: value for key, value in asdict(off).items() if key not in ignored
    }
    on_common = {key: value for key, value in asdict(on).items() if key not in ignored}
    assert off_common == on_common
    assert screen.NFE_GRID == [1, 2, 4]
    assert screen.SOURCES == [
        "autonomous",
        "off",
        "autonomous_future_shuffled",
    ]
    assert screen.SOURCES == screen.DEPLOYABLE_SOURCES
    assert not any("oracle" in source for source in screen.SOURCES)

    register = screen.build_parser()._subparsers._group_actions[0].choices["register"]
    options = {
        option for action in register._actions for option in action.option_strings
    }
    assert not any("test" in option for option in options)


def test_self_signed_registration_cannot_enable_oracle(tmp_path):
    registration = _registration(tmp_path)
    registration["evaluation"]["oracle_sources"] = ["oracle_matched"]
    registration.pop("identity_sha256")
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(screen.identity_payload(registration)), encoding="utf-8")
    with pytest.raises(screen.ContractError, match="deployable evaluation"):
        screen.load_registration(path, verify_files=False)


def test_memory_smoke_requires_backward_without_optimizer_update(tmp_path):
    registration = _registration(tmp_path)
    receipt = _write_memory_smoke(tmp_path, registration)
    assert screen.validate_memory_smoke(registration)["status"] == "pass"

    receipt.pop("identity_sha256")
    receipt["optimizer_step_executed"] = True
    (tmp_path / "memory_smoke.json").write_text(
        json.dumps(screen.identity_payload(receipt)), encoding="utf-8"
    )
    with pytest.raises(screen.ContractError, match="memory smoke contract"):
        screen.validate_memory_smoke(registration)


def test_training_content_binds_full_config_and_rejects_tamper(
    tmp_path, monkeypatch
):
    registration = _registration(tmp_path)
    registration_path = tmp_path / "protocol_registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    _write_memory_smoke(tmp_path, registration)
    arm = screen.ARMS[0]
    run_dir = tmp_path / "runs" / arm.slug
    (run_dir / ".hydra").mkdir(parents=True)
    manifest = screen.identity_payload(
        {
            "schema": screen.ARM_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "array_task_id": 0,
            "arm": asdict(arm),
        }
    )
    (run_dir / "arm_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    initialization = screen.identity_payload(
        {
            "schema": screen.INITIALIZATION_MATCH_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": arm.code,
            "arm_identity_sha256": manifest["identity_sha256"],
            "exact_match": True,
        }
    )
    (run_dir / "initialization_match.json").write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    snapshot = run_dir / "snapshot.pt"
    snapshot.write_bytes(b"checkpoint-bytes")
    (run_dir / ".hydra/config.yaml").write_text("seed: 1234\n", encoding="utf-8")
    resolved_config = {
        "seed": screen.SEED,
        "name": f"{registration['study_id']}-{arm.code}",
        "trainer": {"config": {"max_iter": screen.TRAIN_UPDATES}},
        "model": {
            "dual_diffusion": {
                "condition_mode": arm.condition_mode,
                "condition_on_tf": arm.condition_on_state,
                "intra_forward_forcing": {
                    "history_bins": screen.AUXILIARY_HISTORY_BINS
                },
            }
        },
        "wandb": {
            "entity": screen.WANDB_ENTITY,
            "project": screen.WANDB_PROJECT,
            "group": None,
            "id": manifest["identity_sha256"],
        },
    }
    resolved_path = run_dir / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved_config), encoding="utf-8")
    runtime_path = run_dir / "runtime.json"
    runtime_path.write_text("{}\n", encoding="utf-8")
    completion = {
        "schema_version": 1,
        "status": "completed",
        "completed_updates": screen.TRAIN_UPDATES,
        "max_iter": screen.TRAIN_UPDATES,
        "run_identity_sha256": manifest["identity_sha256"],
        "snapshot": str(snapshot.resolve()),
        "wandb_run_id": manifest["identity_sha256"],
        "wandb_training_status": "completed",
        "source_commit": registration["source"]["commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "warm_start_sha256": registration["warm_start"]["sha256"],
        "runtime_receipt_sha256": screen.sha256_file(runtime_path),
        "resolved_config_sha256": screen.sha256_file(resolved_path),
    }
    (run_dir / "training_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    monkeypatch.setattr(
        screen,
        "validate_runtime_receipt",
        lambda path, _registration, label: {
            "path": str(path),
            "sha256": screen.sha256_file(path),
        },
    )
    args = type(
        "Args",
        (),
        {
            "registration": registration_path,
            "array_task_id": 0,
            "run_dir": run_dir,
        },
    )()
    assert screen.command_bind_training_output(args) == 0
    content = screen.validate_training_content(registration, arm, run_dir)
    assert content["resolved_config"]["sha256"] == screen.sha256_file(resolved_path)

    resolved_path.write_text(json.dumps({**resolved_config, "seed": 999}), encoding="utf-8")
    with pytest.raises(screen.ContractError, match="resolved_config changed"):
        screen.validate_training_content(registration, arm, run_dir)


def test_resolved_configs_bind_midpoint_seam_and_exact_val64():
    common = OmegaConf.load(screen.COMMON_CONFIG)
    model = OmegaConf.load(screen.MODEL_CONFIG)
    assert common.model.forward_model.gradient_checkpointing is False
    assert model.dual_diffusion.intra_forward_forcing == {
        "enabled": True,
        "block_index": 14,
        "stop_gradient": True,
        "history_bins": 2,
    }
    assert list(model.dual_diffusion.evaluation_nfe_steps) == screen.NFE_GRID
    assert list(model.dual_diffusion.evaluation_condition_sources) == screen.SOURCES
    assert common.wandb.entity == "zijiandu"
    assert common.wandb.project == "dual-video-diffusion-private"
    assert common.wandb.group is None
    loader = common.val_data_loader[0]
    validation = common.trainer.config.validation
    assert (
        int(loader.batch_size) * int(validation.n_val_samples) * screen.WORLD_SIZE
        == screen.EXPECTED_VALIDATION_CLIPS
    )


def test_state_byte_hash_is_order_stable_and_content_sensitive():
    first = {
        "b": torch.tensor([2.0], dtype=torch.bfloat16),
        "a": torch.tensor([1, 3], dtype=torch.int64),
        "scalar": torch.tensor(0.02, dtype=torch.float32),
    }
    reordered = {
        "scalar": first["scalar"].clone(),
        "a": first["a"].clone(),
        "b": first["b"].clone(),
    }
    assert preflight._tensor_state_sha256(first) == preflight._tensor_state_sha256(
        reordered
    )
    reordered["a"][0] = 9
    assert preflight._tensor_state_sha256(first) != preflight._tensor_state_sha256(
        reordered
    )


def test_initialization_anchor_requires_exact_cross_arm_bytes(tmp_path, monkeypatch):
    registration = _registration(tmp_path)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    monkeypatch.setattr(
        screen,
        "load_registration",
        lambda _path, verify_files=False: registration,
    )
    shared = {
        "initialized_model_state_sha256": "1" * 64,
        "parameter_schema_sha256": "2" * 64,
        "trainable_parameter_schema_sha256": "3" * 64,
        "optimizer_schema_sha256": "4" * 64,
        "total_parameter_count": 100,
        "trainable_parameter_count": 10,
        "initial_snapshot_tensor_count": 20,
    }
    for task_id, arm in enumerate(screen.ARMS):
        run_dir = tmp_path / "runs" / arm.slug
        run_dir.mkdir(parents=True)
        manifest = screen.identity_payload(
            {
                "schema": screen.ARM_SCHEMA,
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": asdict(arm),
            }
        )
        (run_dir / "arm_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        receipt = screen.identity_payload(
            {
                "schema": screen.PREFLIGHT_SCHEMA,
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": arm.code,
                "selector": arm.selector,
                "arm_identity_sha256": manifest["identity_sha256"],
                "status": "pass",
                **shared,
            }
        )
        receipt_path = run_dir / "warmstart_preflight.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        args = type(
            "Args",
            (),
            {
                "registration": registration_path,
                "array_task_id": task_id,
                "run_dir": run_dir,
                "preflight": receipt_path,
            },
        )()
        assert screen.command_bind_initialization(args) == 0
        assert (run_dir / "initialization_match.json").is_file()


def _write_evaluation_grid(
    study_root: Path,
    registration: dict,
    arm,
    arm_identity: str,
    match_identity: str,
) -> Path:
    output = study_root / "evaluation" / arm.slug
    output.mkdir(parents=True)
    rows_path = output / "rows.jsonl"
    rows = []
    snapshot_sha256 = ("a" if arm.code == "MID-OFF" else "b") * 64
    config_sha256 = ("e" if arm.code == "MID-OFF" else "f") * 64
    content_identity = ("7" if arm.code == "MID-OFF" else "8") * 64
    runtime_sha256 = "9" * 64
    hydra_config_sha256 = "0" * 64
    for source in screen.SOURCES:
        for nfe in screen.NFE_GRID:
            for clip in range(screen.EXPECTED_VALIDATION_CLIPS):
                candidate = arm.code == "MID-ON" and source == "autonomous"
                video = 0.99 if candidate else 1.0
                decoded = 0.99 if candidate else 1.0
                temporal = 0.90 if candidate else 1.0
                values = {
                    "schema": screen.RESULT_SCHEMA,
                    "registration_identity_sha256": registration["identity_sha256"],
                    "arm_identity_sha256": arm_identity,
                    "arm": arm.code,
                    "source": source,
                    "oracle_leakage": False,
                    "deployable": True,
                    "nfe": nfe,
                    "clip_index": clip,
                    "video_nmse": video,
                    "decoded_mse": decoded,
                    "temporal_mse": temporal,
                    "auxiliary_future_nmse": 1.0,
                    "auxiliary_future_cosine": 0.0,
                    "auxiliary_dc_nmse": 1.0,
                    "auxiliary_motion_nmse": 1.0,
                    "actual_wan_calls": nfe,
                    "hook_wan_calls": nfe,
                    "artifact_midpoint_head_calls": nfe,
                    "hook_midpoint_head_calls": nfe,
                    "hook_midpoint_block_calls": nfe,
                    "timed_wan_calls": nfe,
                    "timed_midpoint_head_calls": nfe,
                    "timed_midpoint_block_calls": nfe,
                    "extra_wan_calls": 0,
                    "evaluation_generations_per_cell": 3,
                    "audit_batch_size": 2,
                    "timed_batch_size": 1,
                    "total_evaluation_wan_calls": 3 * nfe,
                    "wan_block_count": 30,
                    "midpoint_block_index": 14,
                    "midpoint_condition_source": {
                        "autonomous": "aligned",
                        "off": "off",
                        "autonomous_future_shuffled": "future_shuffled",
                    }[source],
                    "generated_clean_stop_gradient": True,
                    "sampler_transform_calls": 0,
                    "online_teacher_calls": 0,
                    "clean_auxiliary_passed_to_sampler": False,
                    "future_rgb_passed_to_sampler": False,
                    "all_auxiliary_bins_initialized_from_noise": True,
                    "video_initial_sha256": f"{clip:064x}",
                    "auxiliary_initial_sha256": f"{clip + 1000:064x}",
                    "video_final_sha256": (
                        f"{clip + 2000:064x}"
                        if arm.code == "MID-OFF"
                        else f"{clip + 3000 + 100 * screen.SOURCES.index(source):064x}"
                    ),
                    "auxiliary_final_sha256": (
                        f"{clip + 4000:064x}"
                        if arm.code == "MID-OFF"
                        else f"{clip + 5000 + 100 * screen.SOURCES.index(source):064x}"
                    ),
                    "decoded_final_sha256": (
                        f"{clip + 7000:064x}"
                        if arm.code == "MID-OFF"
                        else f"{clip + 8000 + 100 * screen.SOURCES.index(source):064x}"
                    ),
                    "raw_target_sha256": f"{clip + 6000:064x}",
                    "snapshot_sha256": snapshot_sha256,
                    "hydra_config_sha256": hydra_config_sha256,
                    "resolved_config_sha256": config_sha256,
                    "training_content_identity_sha256": content_identity,
                    "training_runtime_sha256": runtime_sha256,
                    "evaluation_runtime_sha256": runtime_sha256,
                    "initialization_match_identity_sha256": match_identity,
                    "history_encode_latency_ms": 1.0,
                    "wan_latency_ms": float(nfe * 10),
                    "midpoint_overhead_latency_ms": float(nfe),
                    "decode_latency_ms": 2.0,
                    "end_to_end_latency_ms": float(nfe * 10 + 5),
                    "profiled_internal_end_to_end_latency_ms": float(nfe * 10 + 4),
                    "peak_memory_allocated_bytes": 1024,
                    "effective_state_gate": 0.02,
                    "effective_clock_gate": 0.0,
                    "protected_test_accessed": False,
                }
                rows.append(screen.identity_payload(values))
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = screen.identity_payload(
        {
            "schema": screen.EVALUATION_COMPLETE_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm_identity_sha256": arm_identity,
            "arm": arm.code,
            "rows": len(rows),
            "validation_clips": screen.EXPECTED_VALIDATION_CLIPS,
            "nfe": screen.NFE_GRID,
            "sources": screen.SOURCES,
            "world_size": screen.WORLD_SIZE,
            "rows_sha256": screen.sha256_file(rows_path),
            "snapshot_sha256": snapshot_sha256,
            "hydra_config_sha256": hydra_config_sha256,
            "resolved_config_sha256": config_sha256,
            "training_content_identity_sha256": content_identity,
            "training_runtime_sha256": runtime_sha256,
            "evaluation_runtime_sha256": runtime_sha256,
            "initialization_match_identity_sha256": match_identity,
            "protected_test_accessed": False,
        }
    )
    (output / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")
    return rows_path


def test_analyzer_requires_all_three_nfe1_references_and_passes_clean_grid(
    tmp_path, monkeypatch
):
    registration = _registration(tmp_path)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    initialization = {
        "initialized_model_state_sha256": "1" * 64,
        "parameter_schema_sha256": "2" * 64,
        "trainable_parameter_schema_sha256": "3" * 64,
        "optimizer_schema_sha256": "4" * 64,
        "total_parameter_count": 100,
        "trainable_parameter_count": 10,
        "initial_snapshot_tensor_count": 20,
    }
    anchor = screen.identity_payload(
        {
            "schema": screen.INITIALIZATION_ANCHOR_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "source_arm": "MID-OFF",
            "source_preflight_identity_sha256": "5" * 64,
            "initialization": initialization,
        }
    )
    (tmp_path / "initialization_anchor.json").write_text(
        json.dumps(anchor), encoding="utf-8"
    )
    paths = []
    content_by_arm = {}
    for arm in screen.ARMS:
        run_dir = tmp_path / "runs" / arm.slug
        run_dir.mkdir(parents=True)
        arm_identity = ("c" if arm.code == "MID-OFF" else "d") * 64
        match = screen.identity_payload(
            {
                "schema": screen.INITIALIZATION_MATCH_SCHEMA,
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": arm.code,
                "arm_identity_sha256": arm_identity,
                "preflight_identity_sha256": "6" * 64,
                "anchor_identity_sha256": anchor["identity_sha256"],
                "initialization": initialization,
                "exact_match": True,
            }
        )
        (run_dir / "initialization_match.json").write_text(
            json.dumps(match), encoding="utf-8"
        )
        content_by_arm[arm.code] = {
            "identity_sha256": ("7" if arm.code == "MID-OFF" else "8") * 64,
            "snapshot": {
                "sha256": ("a" if arm.code == "MID-OFF" else "b") * 64
            },
            "hydra_config": {"sha256": "0" * 64},
            "resolved_config": {
                "sha256": ("e" if arm.code == "MID-OFF" else "f") * 64
            },
            "training_runtime": {"sha256": "9" * 64},
        }
        paths.append(
            _write_evaluation_grid(
                tmp_path,
                registration,
                arm,
                arm_identity,
                match["identity_sha256"],
            )
        )
    monkeypatch.setattr(
        screen,
        "validate_training_content",
        lambda _registration, arm, _run_dir, verify_files=True: content_by_arm[
            arm.code
        ],
    )
    output = tmp_path / "analysis.json"
    args = type(
        "Args",
        (),
        {
            "registration": registration_path,
            "rows": paths,
            "output": output,
        },
    )()

    assert screen.command_analyze(args) == 0
    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["decision"]["passes"] is True
    assert len(analysis["decision"]["comparisons"]) == 3
    assert analysis["conclusion"] == "pass_one_call_future_scratchpad_forcing_screen"


def test_launchers_use_explicit_dependencies_and_never_reference_test():
    training = screen.TRAIN_SBATCH.read_text(encoding="utf-8")
    evaluation = screen.EVALUATE_SBATCH.read_text(encoding="utf-8")
    submit = screen.SUBMIT_SCRIPT.read_text(encoding="utf-8")
    memory = screen.MEMORY_SMOKE_SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --array=" not in training
    assert "gradient_checkpointing=false" in training
    assert '"+wandb.id=$ARM_ID"' in training
    assert "LACWM_REQUIRE_PROVENANCE=1" in training
    assert "bind-training-output" in training
    assert "--array=0 --dependency=\"afterok:$memory_job\"" in submit
    assert "--array=1 --dependency=\"afterok:$off_job\"" in submit
    assert "--array=0-1%2 --dependency=\"afterok:$on_job\"" in submit
    assert "verify_b200_runtime.py" in evaluation
    assert "smoke_intra_forward_memory.py" in memory
    for source in (training, evaluation, submit, memory):
        assert "TEST_" not in source
