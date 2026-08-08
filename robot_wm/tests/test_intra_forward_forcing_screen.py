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
            "study_root": str(study_root),
            "protected_test": {"allowed": False},
            "architecture": {
                "wan_block_count": 30,
                "midpoint_block_index": 14,
                "midpoint_head_calls_per_wan_call": 1,
                "additional_wan_calls": 0,
                "generated_clean_formula": "q0_hat=q_sigma-sigma*v_hat",
                "generated_clean_stop_gradient": True,
                "input_level_auxiliary_state_residual": "exact_zero",
                "input_level_auxiliary_clock_residual": "exact_zero",
                "gradient_checkpointing": False,
            },
            "training": {
                "validation_iterator": dict(screen.TRAIN_VALIDATION_CONTRACT),
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
            },
            "wandb": {
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
            },
        }
    )


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
    assert screen.SOURCES == ["autonomous", "off", "autonomous_shuffled"]
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


def test_resolved_configs_bind_midpoint_seam_and_exact_val64():
    common = OmegaConf.load(screen.COMMON_CONFIG)
    model = OmegaConf.load(screen.MODEL_CONFIG)
    assert common.model.forward_model.gradient_checkpointing is False
    assert model.dual_diffusion.intra_forward_forcing == {
        "enabled": True,
        "block_index": 14,
        "stop_gradient": True,
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
                    "evaluation_generations_per_cell": 2,
                    "total_evaluation_wan_calls": 2 * nfe,
                    "wan_block_count": 30,
                    "midpoint_block_index": 14,
                    "midpoint_condition_source": {
                        "autonomous": "aligned",
                        "off": "off",
                        "autonomous_shuffled": "shuffled",
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
                    "snapshot_sha256": ("a" if arm.code == "MID-OFF" else "b") * 64,
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
            "snapshot_sha256": ("a" if arm.code == "MID-OFF" else "b") * 64,
            "initialization_match_identity_sha256": match_identity,
            "protected_test_accessed": False,
        }
    )
    (output / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")
    return rows_path


def test_analyzer_requires_all_three_nfe1_references_and_passes_clean_grid(
    tmp_path,
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
        paths.append(
            _write_evaluation_grid(
                tmp_path,
                registration,
                arm,
                arm_identity,
                match["identity_sha256"],
            )
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
    assert analysis["conclusion"] == "pass_one_call_intra_forward_forcing_screen"


@pytest.mark.parametrize(
    "path",
    [
        screen.TRAIN_SBATCH,
        screen.EVALUATE_SBATCH,
    ],
)
def test_launchers_are_two_arm_fail_closed_and_never_reference_test(path):
    source = path.read_text(encoding="utf-8")
    assert "--array=0-1" in source
    assert "gradient_checkpointing=false" in source or path == screen.EVALUATE_SBATCH
    assert "TEST_" not in source
    assert "/mnt/data2/" not in source
