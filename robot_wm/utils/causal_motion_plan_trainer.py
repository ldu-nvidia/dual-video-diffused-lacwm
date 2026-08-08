"""Fail-closed trainer for the matched CAMP video continuations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from robot_wm.utils.trainer import Trainer


PARENT_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
PARENT_AUXILIARY_PREFIXES = (
    "forward_model.tf_token_adapter",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_velocity_head",
)
PLANNER_PREFIX = "causal_motion_planner."
NORMALIZER_PREFIX = "motion_plan_normalizer."
TRAIN_MANIFEST_SHA256 = (
    "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
)
TRAIN_METADATA_SHA256 = (
    "fa22a213f352ffb8cc0b4dc0d35138b35aac349c03f362c597c621fa3473da43"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    content = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _required_input_from_environment(name: str, expected_sha256: str) -> Path:
    path = Path(os.environ.get(name, "")).expanduser()
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or _sha256(path) != expected_sha256
    ):
        raise RuntimeError(f"CAMP registered input {name} differs")
    return path.resolve(strict=True)


def _state_digest(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    selected = sorted(
        (name, value) for name, value in state.items() if name.startswith(prefixes)
    )
    if not selected:
        raise RuntimeError("CAMP state digest selection is empty")
    for name, value in selected:
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


class CausalMotionPlanTrainer(Trainer):
    """Validate the common parent/planner before either arm can optimize."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError("CAMP requires a regular absolute VPM parent snapshot")
        expected_parent_sha = os.environ.get("CAMP_VPM_SNAPSHOT_SHA256")
        if (
            expected_parent_sha != PARENT_SNAPSHOT_SHA256
            or _sha256(load_path) != PARENT_SNAPSHOT_SHA256
        ):
            raise RuntimeError("CAMP VPM parent snapshot SHA-256 differs")
        snapshot = torch.load(load_path, map_location="cpu", weights_only=True, mmap=True)
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("world_size") != 8
            or snapshot.get("gradient_accumulation_steps") != 1
            or snapshot.get("_start_iter") != 1000
            or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("historical VPM parent metadata differs")
        parent_state = snapshot.get("model")
        if not isinstance(parent_state, dict):
            raise RuntimeError("historical VPM parent lacks a model state")
        model_state = model.state_dict()

        def shared(state: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in state.items()
                if not key.startswith(PARENT_AUXILIARY_PREFIXES)
                and not key.startswith(PLANNER_PREFIX)
                and not key.startswith(NORMALIZER_PREFIX)
            }

        parent_shared = shared(parent_state)
        model_shared = shared(model_state)
        if set(parent_shared) != set(model_shared):
            missing = sorted(set(parent_shared) - set(model_shared))[:8]
            added = sorted(set(model_shared) - set(parent_shared))[:8]
            raise RuntimeError(
                f"CAMP shared VPM schema differs; missing={missing}, added={added}"
            )
        mismatched = [
            key
            for key in parent_shared
            if tuple(parent_shared[key].shape) != tuple(model_shared[key].shape)
        ]
        if mismatched:
            raise RuntimeError(f"CAMP shared VPM shapes differ: {mismatched[:8]}")
        if tuple(config.get("exclude_keys", ())) != PARENT_AUXILIARY_PREFIXES:
            raise RuntimeError("CAMP must exclude exactly the historical auxiliary schema")
        if int(config.get("max_iter", -1)) != 200:
            raise RuntimeError("CAMP matched continuations require exactly 200 updates")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("CAMP is a fresh matched continuation, not a resume")
        planner = getattr(model, "causal_motion_planner", None)
        if planner is None or any(parameter.requires_grad for parameter in planner.parameters()):
            raise RuntimeError("CAMP planner must be loaded and frozen before Trainer init")
        expected_planner_sha = os.environ.get("CAMP_PLANNER_SNAPSHOT_SHA256")
        if (
            expected_planner_sha != getattr(model, "planner_checkpoint_sha256", None)
            or len(str(expected_planner_sha or "")) != 64
        ):
            raise RuntimeError("CAMP planner identity differs between model and environment")
        expected_stats_sha = os.environ.get("CAMP_PLAN_STATS_SHA256")
        normalizer = getattr(model, "motion_plan_normalizer", None)
        if (
            normalizer is None
            or expected_stats_sha != getattr(normalizer, "artifact_sha256", None)
            or len(str(expected_stats_sha or "")) != 64
        ):
            raise RuntimeError(
                "CAMP normalization identity differs between model and environment"
            )
        train_manifest = _required_input_from_environment(
            "CAMP_TRAIN_CLIP_MANIFEST", TRAIN_MANIFEST_SHA256
        )
        train_metadata = _required_input_from_environment(
            "CAMP_TRAIN_CACHE_METADATA", TRAIN_METADATA_SHA256
        )
        del snapshot, parent_state, model_state, parent_shared, model_shared
        super().__init__(*args, **kwargs)
        # The treatment switch is non-parametric. Both arm schemas, frozen
        # planner execution, optimizer parameter groups, and Wan-call topology
        # therefore remain identical.
        module = self.model.module
        if any(parameter.requires_grad for parameter in module.causal_motion_planner.parameters()):
            raise RuntimeError("Trainer construction unfroze the CAMP planner")
        self._camp_arm = "PLAN-ON" if bool(module.fuse_generated_plan) else "PLAN-OFF"
        self._trace_path = self.save_path.parent / "camp_training_trace.jsonl"
        self._trace_complete_path = (
            self.save_path.parent / "camp_training_trace_complete.json"
        )
        schema = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in module.named_parameters()
        ]
        schema_sha256 = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        module_state = module.state_dict()
        initial_auxiliary_state_sha256 = _state_digest(
            module_state, PARENT_AUXILIARY_PREFIXES
        )
        loaded_planner_state_sha256 = _state_digest(
            module_state, (PLANNER_PREFIX, NORMALIZER_PREFIX)
        )
        if self.is_main_process:
            if self._trace_path.exists() or self._trace_complete_path.exists():
                raise RuntimeError("fresh CAMP training trace output already exists")
            _exclusive_json(
                self._trace_path,
                {
                    "kind": "camp_training_trace_header",
                    "arm": self._camp_arm,
                    "fuse_generated_plan": bool(module.fuse_generated_plan),
                    "parent_snapshot": str(load_path.resolve(strict=True)),
                    "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                    "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                    "planner_snapshot": str(
                        Path(module.planner_checkpoint_path).resolve(strict=True)
                    ),
                    "planner_snapshot_sha256": expected_planner_sha,
                    "motion_plan_stats_sha256": expected_stats_sha,
                    "train_manifest": str(train_manifest),
                    "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
                    "train_cache_metadata": str(train_metadata),
                    "train_cache_metadata_sha256": TRAIN_METADATA_SHA256,
                    "parameter_schema_sha256": schema_sha256,
                    "initial_auxiliary_state_sha256": initial_auxiliary_state_sha256,
                    "loaded_planner_state_sha256": loaded_planner_state_sha256,
                    "continuation_updates": 200,
                    "planner_calls_per_example": 2,
                    "clean_future_plan_conditioning": False,
                    "optimizer_state_policy": "fresh_identical_adamw",
                    "protected_test_accessed": False,
                },
            )

    def _log(self, metrics: dict[str, Any]) -> None:
        if self.is_main_process:
            record = {
                "kind": "camp_training_trace_event",
                "arm": self._camp_arm,
                "total_observations": int(self.metrics.total_observations),
                "metrics": metrics,
            }
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        super()._log(metrics)

    def train(self):
        result = super().train()
        if result == "completed" and self.is_main_process:
            _exclusive_json(
                self._trace_complete_path,
                {
                    "kind": "camp_training_trace_complete",
                    "arm": self._camp_arm,
                    "rows": sum(1 for _ in self._trace_path.open("rb")),
                    "trace_sha256": _sha256(self._trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
