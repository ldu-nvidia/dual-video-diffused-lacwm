from __future__ import annotations

import argparse
from pathlib import Path

from tools.slurm import observed_anchor_followup as workflow


def _args(tmp_path: Path, *, phase: str = "cache") -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        expected_commit="a" * 40,
        study_root=tmp_path / "study",
        data_root=tmp_path / "data",
        semantic_cache_root=tmp_path / "semantic",
        train_manifest=tmp_path / "train.jsonl",
        validation_manifest=tmp_path / "val.jsonl",
        vjepa_source=tmp_path / "vjepa",
        vjepa_checkpoint=tmp_path / "vjepa.pt",
        execution_mode="cheap-proxy-validity",
        temporal_selection_record=None,
        ack_private_wandb_project=True,
        dry_run=True,
        execute=False,
    )


def test_command_graph_is_three_phase_exact_eight_b200_geometry(tmp_path: Path) -> None:
    args = _args(tmp_path)
    records = {
        "source": {"commit": "a" * 40, "dirty": False},
        "execution_condition": {"proxy_validity_only": True},
        "protected_test_accessed": False,
    }
    plan = workflow.build_plan(args, records)

    assert plan["ordered_phases"] == ["cache", "train", "evaluate"]
    assert plan["geometry"] == {
        "nodes": 1,
        "gpus": 8,
        "gpu_model": "NVIDIA B200",
        "global_batch_size": 256,
        "local_batch_size": 32,
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "workers_per_rank": 4,
    }
    assert len(plan["commands"]["cache"]) == 4
    assert len(plan["commands"]["train"]) == 4
    assert len(plan["commands"]["evaluate"]) == 3
    assert plan["comparison_design"]["arms"] == ["C-ABS", "AINC-OFF"]
    assert plan["comparison_design"]["same_clean_commit"] is True
    assert (
        plan["comparison_design"]["external_temporal_abs_numeric_baseline_allowed"]
        is False
    )
    assert str(workflow.temporal_workflow.EVALUATOR) in plan["commands"]["cache"][0]
    assert "register" in plan["commands"]["cache"][0]
    assert str(workflow.temporal_workflow.TRAINER) in plan["commands"]["train"][0]
    assert "--target-mode" in plan["commands"]["train"][0]
    assert (
        plan["commands"]["train"][0][
            plan["commands"]["train"][0].index("--target-mode") + 1
        ]
        == "absolute"
    )
    assert plan["commands"]["train"][1][
        plan["commands"]["train"][1].index(str(Path(workflow.ainc.__file__))) + 1
    ] == "calibrate"
    assert str(workflow.temporal_workflow.TRAINER) in plan["commands"]["train"][2]
    assert plan["commands"]["train"][0][
        plan["commands"]["train"][0].index("--updates") + 1
    ] == "200"
    assert plan["commands"]["train"][2][
        plan["commands"]["train"][2].index("--updates") + 1
    ] == "5000"
    assert plan["commands"]["train"][3][
        plan["commands"]["train"][3].index(str(Path(workflow.ainc.__file__))) + 1
    ] == "train"
    assert "--calibration-record" in plan["commands"]["train"][3]
    for command in plan["commands"]["train"]:
        assert command[command.index("--seed") + 1] == "1234"
        assert command[command.index("--global-batch-size") + 1] == "256"
        assert command[command.index("--micro-batch-size") + 1] == "32"
        assert command[command.index("--workers") + 1] == "4"
    for phase_commands in plan["commands"].values():
        for command in phase_commands:
            if "torch.distributed.run" in command:
                assert "--nproc-per-node=8" in command
    assert plan["protected_test_commands"] == []
    assert plan["protected_test_accessed"] is False


def test_wandb_environment_is_private_and_has_no_group(tmp_path: Path) -> None:
    environment = workflow.sanitized_environment(tmp_path)
    assert environment["WANDB_ENTITY"] == "zijiandu"
    assert environment["WANDB_PROJECT"] == "dual-video-diffusion-private"
    assert "WANDB_RUN_GROUP" not in environment
    assert "WANDB_GROUP" not in environment
    assert "--wandb" in workflow.training_command(_args(tmp_path), "calibration")
    assert "--group" not in workflow.training_command(_args(tmp_path), "calibration")


def test_post_no_pass_execution_argument_is_explicit(tmp_path: Path) -> None:
    args = _args(tmp_path, phase="train")
    args.execution_mode = "post-temporal-no-pass"
    args.temporal_selection_record = tmp_path / "frozen-no-pass.json"
    command = workflow.training_command(args, "primary")

    assert command[command.index("--execution-mode") + 1] == "post-temporal-no-pass"
    assert command[command.index("--temporal-selection-record") + 1] == str(
        args.temporal_selection_record
    )


def test_evaluation_phase_contains_only_development_evaluation_and_analysis(
    tmp_path: Path,
) -> None:
    commands = workflow.evaluation_commands(_args(tmp_path, phase="evaluate"))
    assert len(commands) == 3
    assert str(workflow.temporal_workflow.EVALUATOR) in commands[0]
    assert "eval" in commands[0]
    assert "--implementation-registration" in commands[0]
    assert str(Path(workflow.ainc.__file__)) in commands[1]
    assert "evaluate" in commands[1]
    assert str(Path(workflow.analyzer.__file__)) in commands[2]
    assert "--ainc-summary" in commands[2]
    assert "--abs-summary" in commands[2]
    assert commands[2][commands[2].index("--abs-summary") + 1] == str(
        workflow.local_abs_evaluation_dir(_args(tmp_path).study_root) / "summary.json"
    )
    assert all("protected" not in token.lower() for command in commands for token in command)


def test_sbatch_accepts_linked_git_worktrees_without_using_spool_source() -> None:
    wrapper = (
        workflow.REPO_ROOT / "tools" / "slurm" / "observed_anchor_followup.sbatch"
    ).read_text(encoding="utf-8")

    assert 'REPO_ROOT="${OBSERVED_ANCHOR_REPO_ROOT:-}"' in wrapper
    assert 'rev-parse --is-inside-work-tree' in wrapper
    assert 'rev-parse --show-toplevel' in wrapper
    assert '-d "$REPO_ROOT/.git"' not in wrapper
    assert '${BASH_SOURCE[0]}' not in wrapper
