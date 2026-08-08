"""Static and identity tests for the non-launching two-clock consistency workflow."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tools import two_clock_consistency_evaluate as evaluation
from tools.slurm import two_clock_consistency_workflow as workflow


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
    baseline = evaluation.ARM_BY_CODE["TC-CONT"]
    candidate = evaluation.ARM_BY_CODE["TC-CONS"]
    baseline_values = workflow.arm_values(registration, baseline)
    candidate_values = workflow.arm_values(registration, candidate)

    assert baseline_values["run_identity_sha256"] != candidate_values["run_identity_sha256"]
    assert baseline_values["wandb_run_id"] == baseline_values["run_identity_sha256"]
    assert candidate_values["wandb_run_id"] == candidate_values["run_identity_sha256"]
    assert baseline_values["parent_snapshot"] == candidate_values["parent_snapshot"]
    assert baseline_values["train_manifest"] == candidate_values["train_manifest"]
    assert baseline_values["run_dir"] != candidate_values["run_dir"]


def test_composed_training_arms_differ_only_name_lambda_and_tag():
    config_root = ROOT / "projects" / "latent_action_models" / "configs"

    def load(name: str) -> dict:
        with initialize_config_dir(config_dir=str(config_root), version_base=None):
            config = compose(
                config_name="train",
                overrides=[f"+experiments_0908=ravenhuang/wan-dit/{name}"],
            )
        return OmegaConf.to_container(config, resolve=False)

    baseline = load("two_clock_consistency_baseline")
    candidate = load("two_clock_consistency_candidate")
    evaluation._validate_resolved_config(
        OmegaConf.create(baseline), evaluation.ARM_BY_CODE["TC-CONT"]
    )
    evaluation._validate_resolved_config(
        OmegaConf.create(candidate), evaluation.ARM_BY_CODE["TC-CONS"]
    )
    for config in (baseline, candidate):
        config.pop("name")
        config["wandb"].pop("tags")
        config["model"]["two_clock_consistency"].pop("weight")

    assert baseline == candidate


def test_rendered_train_command_has_eight_ranks_private_wandb_and_no_target(tmp_path):
    registration = _registration(tmp_path)
    values = workflow.arm_values(
        registration, evaluation.ARM_BY_CODE["TC-CONS"]
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
    script = (ROOT / "tools" / "slurm" / "two_clock_consistency.sbatch").read_text()
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --qos=short" in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --exclude=pool0-0081,pool0-0089" in script
    assert "sbatch " not in script
    assert script.index('export WAN_DIR="$WAN_DIR_VALUE"') < script.index(
        'source "$ACTIVATE"'
    )
    assert script.index('export VIDEOX_HOME="$VIDEOX_HOME_VALUE"') < script.index(
        'source "$ACTIVATE"'
    )
    assert "targets.fp16.npy" not in script
