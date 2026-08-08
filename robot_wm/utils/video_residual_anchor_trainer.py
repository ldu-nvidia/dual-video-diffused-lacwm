"""Guarded Trainer for the matched 200-update residual-anchor continuation."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class VideoResidualAnchorTrainer(Trainer):
    """Strict-load one VPM parent and record arm-pairing evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        # Match Trainer's positional signature: model, optimizer, scheduler,
        # train loader, validation loader, visualization loader, config.
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError(
                "video residual-anchor continuation requires a regular absolute VPM snapshot"
            )
        expected_sha = os.environ.get("VIDEO_RESIDUAL_ANCHOR_VPM_SNAPSHOT_SHA256")
        if expected_sha != PARENT_SNAPSHOT_SHA256 or _sha256(load_path) != expected_sha:
            raise RuntimeError("parent VPM snapshot SHA-256 is absent or differs")
        snapshot = torch.load(
            load_path, map_location="cpu", weights_only=True, mmap=True
        )
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("world_size") != 8
            or snapshot.get("gradient_accumulation_steps") != 1
            or snapshot.get("_start_iter") != 1000
            or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("historical VPM checkpoint metadata differs")
        model_state = model.state_dict()
        parent_state = snapshot.get("model")
        if not isinstance(parent_state, dict) or set(parent_state) != set(model_state):
            raise RuntimeError("residual-anchor model parameter schema is not VPM-identical")
        mismatched = [
            key
            for key in model_state
            if tuple(model_state[key].shape) != tuple(parent_state[key].shape)
        ]
        if mismatched:
            raise RuntimeError(f"VPM parameter shapes differ: {mismatched[:8]}")
        if list(config.get("exclude_keys", [])):
            raise RuntimeError("residual-anchor continuation must strictly load every key")
        if int(config.get("max_iter", -1)) != 200:
            raise RuntimeError("the matched quick screen fixes exactly 200 updates")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("screen is a fresh matched optimizer continuation, not resume")
        del snapshot, parent_state
        super().__init__(*args, **kwargs)

        module = self.model.module
        mode = str(getattr(module, "video_representation_mode", ""))
        if mode not in {"absolute", "cumulative_residual"}:
            raise RuntimeError("resolved residual-anchor arm is invalid")
        self._residual_anchor_arm = (
            "VPM-ABS" if mode == "absolute" else "VPM-RESIDUAL"
        )
        self._trace_path = self.save_path.parent / "video_residual_anchor_training_trace.jsonl"
        self._complete_path = (
            self.save_path.parent / "video_residual_anchor_training_trace_complete.json"
        )
        if self.is_main_process:
            if self._trace_path.exists() or self._complete_path.exists():
                raise RuntimeError("fresh residual-anchor trace output already exists")
            header = {
                "kind": "video_residual_anchor_training_trace_header",
                "arm": self._residual_anchor_arm,
                "representation_mode": mode,
                "normalization": "none",
                "known_history_policy": "exact_history_only_vae_tokens",
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": expected_sha,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_completed_updates": 1000,
                "continuation_updates": 200,
                "optimizer_state_policy": "fresh_identical_adamw",
                "ema_policy": "none_in_parent_and_none_in_both_arms",
                "auxiliary_feature_access": False,
            }
            _exclusive_json(self._trace_path, header)

    def _log(self, metrics: dict[str, Any]) -> None:
        if self.is_main_process:
            record = {
                "kind": "video_residual_anchor_training_trace_event",
                "arm": self._residual_anchor_arm,
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
                self._complete_path,
                {
                    "kind": "video_residual_anchor_training_trace_complete",
                    "arm": self._residual_anchor_arm,
                    "rows": sum(1 for _ in self._trace_path.open("rb")),
                    "trace_sha256": _sha256(self._trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
