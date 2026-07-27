import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    REPO_ROOT
    / "projects"
    / "latent_action_models"
    / "evaluate_privileged_video.py"
)
PARENT_IDENTITY = "b" * 64
PARENT_ARMS = (
    "parseval_off_s000",
    "parseval_matched_s010",
    "parseval_shuffled_s010",
    "video_only_s000",
)


def _load_entrypoint(monkeypatch):
    train_module = types.ModuleType("train")
    train_module._setup = lambda _cfg: None
    train_module._teardown = lambda _trainer: None
    train_module.dist = types.SimpleNamespace(
        is_initialized=lambda: False,
        destroy_process_group=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "train", train_module)
    monkeypatch.setitem(
        sys.modules,
        "custom_resolvers",
        types.ModuleType("custom_resolvers"),
    )
    spec = importlib.util.spec_from_file_location(
        "privileged_video_evaluation_test",
        ENTRYPOINT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path, model, *, model_state=None):
    torch.save(
        {
            "snapshot_schema_version": 3,
            "_start_iter": 200,
            "_total_observations": 1600,
            "run_identity_sha256": PARENT_IDENTITY,
            "model": (
                model.state_dict()
                if model_state is None
                else model_state
            ),
        },
        path,
    )
    return _sha256(path)


def _config(
    tmp_path,
    snapshot_path,
    snapshot_sha256,
    *,
    parent_arm="parseval_off_s000",
):
    output = tmp_path / "evaluation"
    return OmegaConf.create(
        {
            "privileged_video_evaluation": {
                "parent_arm": parent_arm,
                "snapshot_path": str(snapshot_path),
                "snapshot_sha256": snapshot_sha256,
                "parent_run_identity_sha256": PARENT_IDENTITY,
                "parent_completed_updates": 200,
                "viz_skip_batches": 4,
                "artifact_iteration": 199,
            },
            "trainer": {
                "config": {
                    "load_path": None,
                    "exclude_keys": [],
                    "transition_handoff_path": None,
                    "saving": {
                        "save_path": str(
                            output / "_never_write_snapshot.pt"
                        )
                    },
                    "visualization": {
                        "viz_path": str(output / "visualization"),
                        "require_success": True,
                    },
                }
            },
            "model": {
                "dual_diffusion": {
                    "enabled": True,
                    "condition_on_tf": False,
                    "condition_mode": "off",
                    "schedule_mode": "aligned",
                    "evaluation_disable_tf_clock": True,
                    "evaluation_nfe_steps": [1, 2, 4, 8],
                    "evaluation_condition_sources": [
                        "autonomous",
                        "off",
                    ],
                }
            },
        }
    )


class FakeTrainer:
    def __init__(self, module, cfg):
        self.model = types.SimpleNamespace(module=module)
        self.global_rank = 0
        self.resumed = False
        self.transitioned = False
        self._start_iter = 0
        self._curr_iter = 0
        self.metrics = types.SimpleNamespace(total_observations=0)
        self.use_wandb = False
        self.save_path = Path(cfg.trainer.config.saving.save_path)
        self.completion_path = (
            self.save_path.parent / "training_complete.json"
        )
        self.viz_path = Path(cfg.trainer.config.visualization.viz_path)
        self.viz_data_loaders = [object(), object()]
        self._viz_data_loader_iters = [
            iter([0, 1, 2, 3, 4]),
            iter([10, 11, 12, 13, 14]),
        ]
        self.visualized_values = None

    def _viz(self):
        self.visualized_values = [
            next(iterator) for iterator in self._viz_data_loader_iters
        ]
        (self.viz_path / f"iter_{self._curr_iter}").mkdir(
            parents=True,
            exist_ok=False,
        )

    def train(self):
        raise AssertionError("evaluation must not call train")

    def _step(self):
        raise AssertionError("evaluation must not take an optimizer step")

    def _save_snapshot(self):
        raise AssertionError("evaluation must not save a snapshot")

    def _write_completion_marker(self):
        raise AssertionError(
            "evaluation must not write training completion"
        )


def test_no_update_entrypoint_strict_loads_and_emits_only_iter_199(
    monkeypatch,
    tmp_path,
):
    entrypoint = _load_entrypoint(monkeypatch)
    parent = tmp_path / "parent"
    parent.mkdir()
    snapshot_path = parent / "snapshot.pt"
    source_model = torch.nn.Linear(1, 1, bias=False)
    source_model.weight.data.fill_(9.0)
    snapshot_sha256 = _snapshot(snapshot_path, source_model)
    cfg = _config(tmp_path, snapshot_path, snapshot_sha256)

    target_model = torch.nn.Linear(1, 1, bias=False)
    target_model.weight.data.zero_()
    trainer = FakeTrainer(target_model, cfg)
    calls = []
    monkeypatch.setattr(
        entrypoint.train_entrypoint,
        "_setup",
        lambda actual_cfg: calls.append(("setup", actual_cfg)) or trainer,
    )
    monkeypatch.setattr(
        entrypoint.train_entrypoint,
        "_teardown",
        lambda actual_trainer: calls.append(
            ("teardown", actual_trainer)
        ),
    )

    result = entrypoint.run_privileged_video_evaluation(cfg)

    assert [name for name, _ in calls] == ["setup", "teardown"]
    assert trainer.visualized_values == [4, 14]
    assert trainer._curr_iter == 199
    assert trainer.metrics.total_observations == 0
    torch.testing.assert_close(
        target_model.weight,
        torch.full_like(target_model.weight, 9.0),
    )
    output = tmp_path / "evaluation"
    assert not (output / "_never_write_snapshot.pt").exists()
    assert not (output / "training_complete.json").exists()
    assert list((output / "visualization").iterdir()) == [
        output / "visualization" / "iter_199"
    ]
    assert Path(result["artifact_root"]) == (
        output / "visualization" / "iter_199"
    )
    provenance = json.loads(
        (
            output / "privileged_video_evaluation_provenance.json"
        ).read_text()
    )
    assert provenance["evaluation_optimizer_updates"] == 0
    assert provenance["evaluation_total_observations"] == 0
    assert provenance["viz_skip_batches"] == 4
    assert provenance["artifact_iteration"] == 199
    assert provenance["evaluation_nfe_steps"] == [1, 2, 4, 8]
    assert provenance["evaluation_condition_sources"] == [
        "autonomous",
        "off",
    ]
    assert provenance["runtime_intervention"] == {
        "all_model_calls_advance_video": True,
        "schedule_mode": "aligned",
        "tf_clock_disabled": True,
        "tf_content_disabled": True,
    }
    assert provenance["parent"]["arm"] == "parseval_off_s000"
    assert provenance["parent"]["completed_updates"] == 200
    assert provenance["parent"]["total_observations"] == 1600
    assert provenance["snapshot_written"] is False
    assert provenance["training_completion_written"] is False


@pytest.mark.parametrize("parent_arm", PARENT_ARMS)
def test_config_is_generic_over_compatible_parent_arms(
    monkeypatch,
    tmp_path,
    parent_arm,
):
    entrypoint = _load_entrypoint(monkeypatch)
    parent = tmp_path / "parent"
    parent.mkdir()
    snapshot_path = parent / "snapshot.pt"
    model = torch.nn.Linear(1, 1, bias=False)
    snapshot_sha256 = _snapshot(snapshot_path, model)
    cfg = _config(
        tmp_path,
        snapshot_path,
        snapshot_sha256,
        parent_arm=parent_arm,
    )

    contract = entrypoint._validate_config(cfg)

    assert contract["parent_arm"] == parent_arm


def test_strict_parent_load_rejects_incompatible_model_keys(
    monkeypatch,
    tmp_path,
):
    entrypoint = _load_entrypoint(monkeypatch)
    parent = tmp_path / "parent"
    parent.mkdir()
    snapshot_path = parent / "snapshot.pt"
    model = torch.nn.Linear(1, 1, bias=False)
    snapshot_sha256 = _snapshot(
        snapshot_path,
        model,
        model_state={"wrong.weight": torch.ones(1, 1)},
    )
    trainer = types.SimpleNamespace(
        global_rank=0,
        model=types.SimpleNamespace(module=model),
    )

    with pytest.raises(RuntimeError, match="Error.*state_dict|Missing key"):
        entrypoint._strict_load_parent_snapshot(
            trainer,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            parent_run_identity_sha256=PARENT_IDENTITY,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda cfg: cfg.model.dual_diffusion.__setattr__(
                "condition_on_tf", True
            ),
            "condition_on_tf",
        ),
        (
            lambda cfg: cfg.model.dual_diffusion.__setattr__(
                "condition_mode", "matched"
            ),
            "condition_mode",
        ),
        (
            lambda cfg: cfg.model.dual_diffusion.__setattr__(
                "schedule_mode", "tf_leads"
            ),
            "schedule_mode",
        ),
        (
            lambda cfg: cfg.model.dual_diffusion.__setattr__(
                "evaluation_disable_tf_clock", False
            ),
            "evaluation_disable_tf_clock",
        ),
        (
            lambda cfg: cfg.model.dual_diffusion.__setattr__(
                "evaluation_nfe_steps", [2, 4, 8]
            ),
            "evaluation_nfe_steps",
        ),
        (
            lambda cfg: cfg.model.dual_diffusion.__setattr__(
                "evaluation_condition_sources", ["autonomous"]
            ),
            "evaluation_condition_sources",
        ),
        (
            lambda cfg: cfg.privileged_video_evaluation.__setattr__(
                "viz_skip_batches", 3
            ),
            "viz_skip_batches",
        ),
    ),
)
def test_config_contract_fails_closed(
    monkeypatch,
    tmp_path,
    mutation,
    message,
):
    entrypoint = _load_entrypoint(monkeypatch)
    parent = tmp_path / "parent"
    parent.mkdir()
    snapshot_path = parent / "snapshot.pt"
    model = torch.nn.Linear(1, 1, bias=False)
    snapshot_sha256 = _snapshot(snapshot_path, model)
    cfg = _config(tmp_path, snapshot_path, snapshot_sha256)
    mutation(cfg)

    with pytest.raises(RuntimeError, match=message):
        entrypoint._validate_config(cfg)
