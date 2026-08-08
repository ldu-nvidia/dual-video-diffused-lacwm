from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_only_new_named_implementation_has_no_sampler_future_target_argument() -> None:
    source = ROOT / "projects" / "latent_action_models" / "lam" / "video_residual_anchor_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "sample_video_residual_anchor"
    ]
    assert len(methods) == 1
    names = [argument.arg for argument in methods[0].args.args + methods[0].args.kwonlyargs]
    assert names == [
        "self",
        "history_rgb",
        "actions",
        "morphology_index",
        "video_noise",
        "auxiliary_noise",
        "steps",
    ]
    assert not any(term in name for name in names for term in ("clean", "target", "teacher"))


def test_protocol_explicitly_disclaims_dual_diffusion_claim() -> None:
    protocol = (
        ROOT / "docs" / "experiments" / "video_residual_anchor_protocol.md"
    ).read_text(encoding="utf-8")
    assert "adjacent structural baseline" in protocol
    assert "not dual diffusion" in protocol
    assert "Protected test" in protocol


def test_validation_stream_repeats_and_vpm_head_clock_is_preserved() -> None:
    config = (
        ROOT
        / "projects"
        / "latent_action_models"
        / "configs"
        / "experiments_0908"
        / "video_residual_anchor_common.yaml"
    ).read_text(encoding="utf-8")
    assert "val_dataset: &validation_dataset\n  <<: *training_dataset" in config
    assert "head_condition_on_tf_clock: true" in config
    assert "validation: {val_every: 100, n_val_samples: 8, save_best: false}" in config


def test_slurm_exposes_registered_runtime_before_activation() -> None:
    script = (
        ROOT / "tools" / "slurm" / "video_residual_anchor_screen.sbatch"
    ).read_text(encoding="utf-8")
    source = 'source "$ACTIVATE"'
    assert script.index('export WAN_DIR="$WAN_DIR_VALUE"') < script.index(source)
    assert script.index('export VIDEOX_HOME="$VIDEOX_HOME_VALUE"') < script.index(source)
