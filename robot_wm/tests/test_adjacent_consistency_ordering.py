"""Structural ordering guards for target-forward and post-step EMA semantics."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _method_calls(path: Path, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return [ast.unparse(node) for node in ast.walk(method) if isinstance(node, ast.Call)]


def test_target_forward_is_built_inside_training_forward_before_loss_return() -> None:
    path = (
        ROOT
        / "projects"
        / "latent_action_models"
        / "lam"
        / "adjacent_consistency_model.py"
    )
    source = path.read_text(encoding="utf-8")
    online = source.index('role="online_student"')
    target = source.index('role="ema_target"')
    returned = source.index("return returned")
    assert online < target < returned
    assert "with torch.no_grad():\n            target_velocity" in source


def test_ema_update_is_strictly_after_base_optimizer_step() -> None:
    path = ROOT / "robot_wm" / "utils" / "adjacent_consistency_trainer.py"
    source = path.read_text(encoding="utf-8")
    step_start = source.index("    def _step(self)")
    step_end = source.index("    def _build_snapshot", step_start)
    body = source[step_start:step_end]
    # Trainer._step performs forward/backward/optimizer.step and returns only
    # after success; the ACD target is updated afterward and nowhere earlier.
    assert body.index("losses = super()._step()") < body.index("ema_update_module_(")
    assert body.count("ema_update_module_(") == 1
