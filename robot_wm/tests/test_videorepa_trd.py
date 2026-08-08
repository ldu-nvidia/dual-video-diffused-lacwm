"""Focused tests for the prospective training-only TRD screen."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from robot_wm.modeling.dual_diffusion.token_relation_distillation import (
    _thresholded_l1,
    pool_per_view_tokens,
    token_relation_loss,
    wan_tokens_to_grid,
)
from tools import videorepa_trd_screen as screen
from tools import videorepa_trd_deployment_canary as deployment_canary
from tools.evaluate_videorepa_trd import _hash_sample


ROOT = Path(__file__).resolve().parents[2]


def test_per_view_pool_never_mixes_camera_seams():
    value = torch.zeros(1, 2, 4, 6, 15)
    value[..., :5] = 1.0
    value[..., 5:10] = 7.0
    value[..., 10:] = -3.0
    pooled = pool_per_view_tokens(
        value, num_views=3, pooled_height=2, pooled_width_per_view=2
    )
    assert pooled.shape == (1, 3, 4, 4, 2)
    assert torch.allclose(pooled[:, 0], torch.ones_like(pooled[:, 0]))
    assert torch.allclose(pooled[:, 1], torch.full_like(pooled[:, 1], 7.0))
    assert torch.allclose(pooled[:, 2], torch.full_like(pooled[:, 2], -3.0))


def test_relations_do_not_require_matching_channel_dimensions():
    generator = torch.Generator().manual_seed(17)
    teacher = torch.randn(2, 3, 4, 8, 7, generator=generator)
    # An isometric embedding preserves every pairwise cosine while changing D.
    matrix = torch.randn(11, 11, generator=generator)
    orthogonal, _ = torch.linalg.qr(matrix)
    padded = torch.cat([teacher, torch.zeros(*teacher.shape[:-1], 4)], dim=-1)
    student = padded @ orthogonal
    loss = token_relation_loss(student, teacher, margin=0.0)
    assert loss.spatial.shape == (2,)
    assert loss.temporal.shape == (2,)
    assert torch.allclose(loss.total, torch.zeros_like(loss.total), atol=2e-6)


def test_temporal_relation_component_detects_frame_permutation():
    generator = torch.Generator().manual_seed(19)
    teacher = torch.randn(1, 3, 4, 8, 12, generator=generator)
    student = teacher.clone()
    student[:, :, 2] = torch.roll(student[:, :, 2], 3, dims=2)
    loss = token_relation_loss(student, teacher, margin=0.0)
    assert loss.temporal.item() > 0.01


def test_margin_is_videorepa_hinge_tolerance():
    difference = torch.tensor([0.02, 0.20])
    actual = _thresholded_l1(difference, torch.zeros_like(difference), 0.05)
    assert torch.allclose(actual, torch.tensor([0.0, 0.15]))


def test_margin_zeros_small_relation_errors():
    teacher = torch.randn(1, 1, 2, 2, 8, generator=torch.Generator().manual_seed(3))
    student = teacher + 1e-5
    unmasked = token_relation_loss(student, teacher, margin=0.0)
    masked = token_relation_loss(student, teacher, margin=0.05)
    assert unmasked.total.item() >= 0.0
    assert masked.total.item() == 0.0


def test_wan_grid_is_bound_to_target_and_patch_stride():
    tokens = torch.randn(2, 4 * 12 * 60, 9)
    grid = wan_tokens_to_grid(
        tokens, target_grid_shape=(4, 24, 120), patch_size=(1, 2, 2)
    )
    assert grid.shape == (2, 9, 4, 12, 60)
    with pytest.raises(ValueError, match="token count"):
        wan_tokens_to_grid(
            tokens[:, :-1],
            target_grid_shape=(4, 24, 120),
            patch_size=(1, 2, 2),
        )


def test_arm_contract_has_one_gradient_level_difference():
    off, on = screen.ARMS
    assert off.code == "TRD-OFF" and on.code == "TRD-ON"
    assert off.mode == "off" and on.mode == "on"
    assert off.loss_weight == 0.0 and on.loss_weight == 0.05
    assert screen.NFE_GRID == (1, 2, 4)
    assert screen.ACTION_CONTROLS == ("aligned", "episode_shuffled", "zero")
    assert screen.TRAIN_CLIPS == 512 and screen.VALIDATION_CLIPS == 64
    assert set(screen.SPLIT_IDENTITIES) == {"train", "val"}
    assert all(
        len(value["manifest_sha256"]) == 64 and len(value["cache_id"]) == 64
        for value in screen.SPLIT_IDENTITIES.values()
    )
    assert _hash_sample(torch.zeros(screen.ACTION_SAMPLE_SHAPE)) == (
        screen.ZERO_ACTION_SHA256
    )


def test_deployable_override_has_no_auxiliary_wan_call():
    source = (
        ROOT
        / "projects/latent_action_models/lam/videorepa_trd_model.py"
    ).read_text(encoding="utf-8")
    video_only = source.split("    def _video_only_wan(", 1)[1].split(
        "    @torch.no_grad()", 1
    )[0]
    deployment = source.split("    def sample_future_deployable(", 1)[1]
    assert "tf_token_adapter" not in video_only
    assert "tf_clock_embedding" not in video_only
    assert "tf_velocity_head" not in video_only
    assert "auxiliary_target" not in deployment
    assert "_video_only_wan" in deployment
    visualization = source.split("    def visualize(", 1)[1]
    assert "sample_future_deployable" in visualization
    assert 'kwargs.get("auxiliary_target")' not in visualization
    evaluator = (ROOT / "tools/evaluate_videorepa_trd.py").read_text(
        encoding="utf-8"
    )
    sampling_move = evaluator.split("def _move_sampling_batch(", 1)[1].split(
        "def _rows_for_batch(", 1
    )[0]
    assert 'if key != "rgb"' in sampling_move
    assert "[:, :history_frames].clone()" in sampling_move


def _synthetic_rows(arm: screen.Arm):
    rows = []
    for clip in range(screen.VALIDATION_CLIPS):
        for nfe in screen.NFE_GRID:
            initial = f"{clip:02d}".ljust(64, "0")
            for control in screen.ACTION_CONTROLS:
                donor = (
                    clip
                    if control == "aligned"
                    else (clip + 1) % screen.VALIDATION_CLIPS
                    if control == "episode_shuffled"
                    else None
                )
                action_digest = (
                    f"{donor:02x}" + "a" * 62
                    if donor is not None
                    else screen.ZERO_ACTION_SHA256
                )
                rows.append(
                    {
                        "schema": screen.RESULT_SCHEMA,
                        "arm": arm.code,
                        "action_control": control,
                        "action_donor_clip_index": (
                            donor
                        ),
                        "action_tensor_sha256": action_digest,
                        "action_tensor_shape": list(screen.ACTION_SAMPLE_SHAPE),
                        "action_tensor_dtype": screen.ACTION_TENSOR_DTYPE,
                        "nfe": nfe,
                        "clip_index": clip,
                        "latent_nmse": 0.3,
                        "decoded_mse": 0.02,
                        "temporal_mse": 0.01,
                        "actual_wan_calls": nfe,
                        "trd_hook_calls_at_inference": 0,
                        "auxiliary_branch_calls_at_inference": 0,
                        "online_teacher_calls": 0,
                        "sampler_received_clean_feature": False,
                        "sampler_received_future_rgb": False,
                        "score_only_future_rgb_used_after_generation": True,
                        "target_array_opened": False,
                        "video_initial_sha256": initial,
                        "video_final_sha256": "1" * 64,
                        "decoded_sha256": "2" * 64,
                        "raw_target_sha256": f"{clip:02x}".ljust(64, "0"),
                        "protected_test_accessed": False,
                    }
                )
    return rows


def test_result_grid_requires_target_free_inference_and_paired_noise():
    rows = _synthetic_rows(screen.ARMS[0])
    screen.validate_result_rows(rows, screen.ARMS[0])
    diagnostic = screen._paired_action_degradation(
        rows,
        control="episode_shuffled",
        nfe=1,
        metric="decoded_mse",
        seed=4,
    )
    assert diagnostic["point"] == pytest.approx(0.0)
    rows[0]["sampler_received_clean_feature"] = True
    with pytest.raises(screen.TRDContractError, match="violates protocol"):
        screen.validate_result_rows(rows, screen.ARMS[0])


def test_result_grid_binds_shuffled_actions_to_exact_donor_tensor():
    rows = _synthetic_rows(screen.ARMS[0])
    for shuffled in rows:
        if (
            shuffled["action_control"] == "episode_shuffled"
            and shuffled["clip_index"] == 0
        ):
            shuffled["action_tensor_sha256"] = "f" * 64
    with pytest.raises(screen.TRDContractError, match="exact donor tensor"):
        screen.validate_result_rows(rows, screen.ARMS[0])


def test_deployment_canary_report_is_identity_bound_and_exact():
    registration = {
        "identity_sha256": "1" * 64,
        "source": {"git_commit": "2" * 40},
        "parent_snapshot": {"file": {"sha256": "3" * 64}},
    }
    comparison = {
        "shape": [1],
        "dtype": "torch.float32",
        "custom_sha256": "4" * 64,
        "ordinary_vpm_sha256": "4" * 64,
        "bitwise_equal": True,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
    }
    report = screen._identity(
        {
            "schema_version": deployment_canary.CANARY_SCHEMA_VERSION,
            "kind": deployment_canary.CANARY_KIND,
            "status": "passed_before_full_arm_training",
            "registration_identity_sha256": registration["identity_sha256"],
            "source_git_commit": registration["source"]["git_commit"],
            "parent_snapshot_sha256": registration["parent_snapshot"]["file"][
                "sha256"
            ],
            "clip_split": "train",
            "clip_index": 0,
            "nfe": 1,
            "condition_source": "off",
            "sample_id": 0,
            "history_tensor_sha256": "5" * 64,
            "history_input_owned_storage": True,
            "action_tensor_sha256": "6" * 64,
            "action_tensor_shape": list(screen.ACTION_SAMPLE_SHAPE),
            "action_tensor_dtype": screen.ACTION_TENSOR_DTYPE,
            "action_tensor_unchanged": True,
            "custom_deployment_counts": {
                "wan_calls": 1,
                "auxiliary_branch_calls": 0,
                "auxiliary_module_calls": dict(
                    deployment_canary.ZERO_AUXILIARY_MODULE_CALLS
                ),
                "trd_hook_installations": 0,
            },
            "ordinary_vpm_condition_off_counts": {
                "wan_calls": 1,
                "auxiliary_branch_calls": 5,
                "auxiliary_module_calls": dict(
                    deployment_canary.ORDINARY_VPM_AUXILIARY_MODULE_CALLS
                ),
                "trd_hook_installations": 0,
            },
            "fixed_noise": dict(comparison),
            "history_reference_latents": dict(comparison),
            "native_video_velocity": dict(comparison),
            "future_output": dict(comparison),
            "final_video_latent_fp16_evidence": dict(comparison),
            "decoded_future_uint8": dict(comparison),
            "numerical_tolerance": {"rtol": 0.0, "atol": 0.0},
            "custom_path_received_future_rgb": False,
            "custom_path_received_clean_feature": False,
            "target_array_opened": False,
            "validation_split_accessed": False,
            "protected_test_accessed": False,
            "passed": True,
        }
    )
    for name, (shape, dtype) in (
        deployment_canary.EXPECTED_COMPARISON_SPECS.items()
    ):
        report[name]["shape"] = shape
        report[name]["dtype"] = dtype
    report["history_tensor_shape"] = [5, 3, 180, 960]
    report["history_tensor_dtype"] = "torch.float32"
    report = screen._identity(
        {key: value for key, value in report.items() if key != "identity_sha256"}
    )
    deployment_canary._validate_report(registration, report)

    partial_reference = deepcopy(report)
    partial_reference["ordinary_vpm_condition_off_counts"] = {
        "wan_calls": 1,
        "auxiliary_branch_calls": 1,
        "auxiliary_module_calls": {
            name: int(name == "control_adapter")
            for name in deployment_canary.AUXILIARY_MODULE_NAMES
        },
        "trd_hook_installations": 0,
    }
    partial_reference = screen._identity(
        {
            key: value
            for key, value in partial_reference.items()
            if key != "identity_sha256"
        }
    )
    with pytest.raises(
        deployment_canary.TRDDeploymentCanaryError,
        match="report differs",
    ):
        deployment_canary._validate_report(registration, partial_reference)

    impossible_schema = deepcopy(report)
    impossible_schema["fixed_noise"]["shape"] = [999]
    impossible_schema["fixed_noise"]["dtype"] = "torch.int8"
    impossible_schema = screen._identity(
        {
            key: value
            for key, value in impossible_schema.items()
            if key != "identity_sha256"
        }
    )
    with pytest.raises(
        deployment_canary.TRDDeploymentCanaryError,
        match="numerical equivalence",
    ):
        deployment_canary._validate_report(registration, impossible_schema)

    report["future_output"]["bitwise_equal"] = False
    report = screen._identity(
        {key: value for key, value in report.items() if key != "identity_sha256"}
    )
    with pytest.raises(
        deployment_canary.TRDDeploymentCanaryError,
        match="numerical equivalence",
    ):
        deployment_canary._validate_report(registration, report)


def test_registration_cli_has_no_protected_test_argument():
    parser = screen.build_parser()
    help_text = parser.format_help().lower()
    source = inspect.getsource(screen.build_parser).lower()
    assert "test-manifest" not in help_text
    assert "test_manifest" not in source


def test_configs_freeze_train_only_cache_and_zero_projection(monkeypatch):
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    environment = {
        "TRD_TRAIN_CLIP_MANIFEST": "/train/manifest.jsonl",
        "TRD_TRAIN_CACHE_METADATA": "/train/cache.json",
        "TRD_VPM_SNAPSHOT": "/parent/snapshot.pt",
        "TRD_RUN_ROOT": "/runs/trd",
        "WAN_DIR": "/wan",
        "VIDEOX_HOME": "/videox",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        config_dir=str(ROOT / "projects/latent_action_models/configs"),
        version_base=None,
    ):
        off = compose(
            config_name="train",
            overrides=[
                "+experiments_0908=ravenhuang/wan-dit/videorepa_trd_off"
            ],
        )
        on = compose(
            config_name="train",
            overrides=[
                "+experiments_0908=ravenhuang/wan-dit/videorepa_trd_on"
            ],
        )
    for config in (off, on):
        assert config.dataset.datasets.ABC.clip_manifest == "/train/manifest.jsonl"
        assert config.val_dataset.datasets.ABC.clip_manifest == "/train/manifest.jsonl"
        assert config.viz_dataset.datasets.ABC.clip_manifest == "/train/manifest.jsonl"
        assert config.model.forward_model.gradient_checkpointing is False
        assert config.model.dual_diffusion.parameter_matched_control is True
        assert config.model.dual_diffusion.condition_mode == "off"
        assert (
            config.model.token_relation_distillation.exclude_first_temporal_bin
            is True
        )
        assert config.wandb.entity == "zijiandu"
        assert config.wandb.project == "dual-video-diffusion-private"
        assert config.wandb.group is None
    assert off.model.token_relation_distillation.loss_weight == 0.0
    assert on.model.token_relation_distillation.loss_weight == 0.05
    assert set(off.model.keys()) == set(on.model.keys())


def test_slurm_workflow_is_nonrequeueable_and_excludes_all_bad_nodes():
    sbatch = (
        ROOT / "tools/slurm/videorepa_trd_workflow.sbatch"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "tools/slurm/submit_videorepa_trd_screen.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in sbatch
    for node in ("pool0-0081", "pool0-0089", "pool0-0200", "pool0-0343"):
        assert node in sbatch and node in launcher
    assert "--exclude must retain known bad node" in launcher
    assert "verify-stage" in sbatch
    assert sbatch.index("verify-stage") < sbatch.index("torch.distributed.run")
    assert "rev-parse --is-inside-work-tree" in sbatch
    assert "Dry-run is the default" in launcher
    assert "--execute" in launcher
    assert "TIME_LIMIT=02:00:00" in launcher
    assert "short QOS requires the cluster-compatible" in launcher
    assert launcher.count("--gpus-per-node 1") == 2
    assert "afterok:$TRAIN_ID" in launcher
    assert "afterok:$SEAL_ID" in launcher
    assert "same-name TRD workflow stage is already active" in launcher
    assert '"viz_data_loader=[]"' in sbatch
    assert '"viz_data_loader=null"' not in sbatch
    assert "videorepa_trd_deployment_canary.py" in sbatch
    assert sbatch.index('"$DEPLOYMENT_CANARY" run') < sbatch.index(
        'torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER"'
    )
    assert '"$DEPLOYMENT_CANARY" verify' in sbatch
    evaluate_branch = sbatch.index('if [[ "$MODE" == evaluate ]]')
    for required_export in (
        'export TRD_VPM_SNAPSHOT="$WARMSTART"',
        'export TRD_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"',
        'export TRD_TRAIN_CACHE_METADATA="$TRAIN_METADATA"',
        'export TRD_RUN_ROOT="$STUDY_ROOT/training"',
    ):
        assert sbatch.count(required_export) == 1
        assert sbatch.index(required_export) < evaluate_branch


@pytest.mark.parametrize(
    "script",
    [
        "tools/slurm/videorepa_trd_workflow.sbatch",
        "tools/slurm/submit_videorepa_trd_screen.sh",
    ],
)
def test_shell_scripts_parse(script):
    completed = subprocess.run(
        ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
