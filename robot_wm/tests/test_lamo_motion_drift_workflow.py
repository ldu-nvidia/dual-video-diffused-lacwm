"""Static and identity tests for the non-launching LaMo workflow."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tools import lamo_motion_drift_evaluate as evaluation
from tools.slurm import lamo_motion_drift_workflow as workflow


ROOT = Path(__file__).resolve().parents[2]


def _registration(tmp_path: Path) -> dict:
    return {
        "identity_sha256": "a" * 64,
        "output_root": str(tmp_path / "study"),
        "tool_repository": {"git_commit": "b" * 40, "path": str(ROOT)},
        "controlled_study": {"parent_snapshot": {"path": "/inputs/vpm.pt"}},
        "training": {
            "manifest": {"path": "/inputs/train.jsonl"},
            "cache_metadata": {"path": "/inputs/train.json"},
        },
        "validation": {
            "manifest": {"path": "/inputs/val.jsonl"},
            "cache_metadata": {"path": "/inputs/val.json"},
        },
        "runtime": {
            "wan_dir": "/assets/wan",
            "videox_home": "/src/videox",
            "python": "/env/bin/python",
        },
    }


def test_arm_identity_and_paths_are_distinct_but_protocol_matched(tmp_path):
    registration = _registration(tmp_path)
    baseline = evaluation.ARM_BY_CODE["VPM-CONT"]
    drift = evaluation.ARM_BY_CODE["VPM-DRIFT"]
    baseline_values = workflow.arm_values(registration, baseline)
    drift_values = workflow.arm_values(registration, drift)

    assert baseline_values["run_identity_sha256"] != drift_values["run_identity_sha256"]
    assert baseline_values["parent_snapshot"] == drift_values["parent_snapshot"]
    assert baseline_values["train_manifest"] == drift_values["train_manifest"]
    assert baseline_values["run_dir"] != drift_values["run_dir"]


def test_composed_training_arms_differ_only_name_lambda_and_tag():
    config_root = ROOT / "projects" / "latent_action_models" / "configs"

    def load(name: str) -> dict:
        with initialize_config_dir(config_dir=str(config_root), version_base=None):
            config = compose(
                config_name="train",
                overrides=[f"+experiments_0908=ravenhuang/wan-dit/{name}"],
            )
        return OmegaConf.to_container(config, resolve=False)

    baseline = load("lamo_motion_drift_baseline")
    drift = load("lamo_motion_drift_aux")
    evaluation._validate_resolved_config(
        OmegaConf.create(baseline), evaluation.ARM_BY_CODE["VPM-CONT"]
    )
    evaluation._validate_resolved_config(
        OmegaConf.create(drift), evaluation.ARM_BY_CODE["VPM-DRIFT"]
    )
    for config in (baseline, drift):
        config.pop("name")
        config["wandb"].pop("tags")
        config["model"]["motion_drift"].pop("weight")

    assert baseline == drift


def test_rendered_train_command_has_eight_ranks_private_wandb_and_no_target(tmp_path):
    registration = _registration(tmp_path)
    values = workflow.arm_values(
        registration, evaluation.ARM_BY_CODE["VPM-DRIFT"]
    )
    command = workflow._train_command(values, ROOT)
    rendered = " ".join(command).lower()
    assert "--nproc_per_node=8" in command
    assert "wandb.entity=zijiandu" in command
    assert "wandb.project=dual-video-diffusion-private" in command
    assert "wandb.group=null" in command
    assert f"+wandb.id={values['run_identity_sha256']}" in command
    assert "+wandb.resume=never" in command
    assert not any(
        term in rendered
        for term in ("target_cache", "teacher", "test_split", "lockbox")
    )


def test_sbatch_is_short_eight_b200_non_requeueable_and_does_not_submit():
    script = (ROOT / "tools" / "slurm" / "lamo_motion_drift.sbatch").read_text()
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --qos=short" in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --exclude=pool0-0081,pool0-0089" in script
    assert "sbatch " not in script
    assert "targets.fp16.npy" not in script
    assert '[[ "$REGISTERED_PYTHON" == "$PYTHON_BIN" ]]' in script


def test_wandb_finish_is_private_bounded_and_nonraising():
    config_root = ROOT / "projects" / "latent_action_models" / "configs"
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "+experiments_0908=ravenhuang/wan-dit/lamo_motion_drift_baseline"
            ],
        )
    settings = OmegaConf.to_container(config.wandb.settings, resolve=True)
    assert config.wandb.entity == "zijiandu"
    assert config.wandb.project == "dual-video-diffusion-private"
    assert config.wandb.group is None
    assert settings == {
        "start_method": "thread",
        "save_code": False,
        "finish_timeout": 120.0,
        "finish_timeout_raises": False,
    }
