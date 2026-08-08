"""Guarded trainer for the paired Stage-1 action-cycle continuation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
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


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


class ActionCycleStage1Trainer(Trainer):
    """Strict model-only continuation with exact paired-input audit hashes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError("action-cycle continuation requires the regular VPM snapshot")
        if os.environ.get("ACTION_CYCLE_VPM_SNAPSHOT_SHA256") != PARENT_SNAPSHOT_SHA256:
            raise RuntimeError("registered parent snapshot digest is absent or differs")
        observed_sha = (
            _sha256(load_path)
            if not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
            else None
        )
        if torch.distributed.is_initialized():
            values = [observed_sha]
            torch.distributed.broadcast_object_list(values, src=0)
            observed_sha = values[0]
        if observed_sha != PARENT_SNAPSHOT_SHA256:
            raise RuntimeError("historical VPM snapshot content changed")
        snapshot = torch.load(load_path, map_location="cpu", weights_only=True, mmap=True)
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("world_size") != 8
            or snapshot.get("gradient_accumulation_steps") != 1
            or snapshot.get("_start_iter") != 1000
            or snapshot.get("_total_observations") != 8000
            or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
            or not isinstance(snapshot.get("rank_states"), Sequence)
            or len(snapshot["rank_states"]) != 8
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("historical VPM snapshot metadata differs")
        model_state = model.state_dict()
        parent_state = snapshot.get("model")
        if not isinstance(parent_state, Mapping) or set(parent_state) != set(model_state):
            raise RuntimeError("action-cycle model schema is not exactly VPM-compatible")
        mismatched = [
            key
            for key in model_state
            if tuple(model_state[key].shape) != tuple(parent_state[key].shape)
            or model_state[key].dtype != parent_state[key].dtype
        ]
        if mismatched:
            raise RuntimeError(f"VPM parameter shape/dtype differs: {mismatched[:8]}")
        if list(config.get("exclude_keys", [])):
            raise RuntimeError("action-cycle continuation must strictly load every VPM key")
        if int(config.get("max_iter", -1)) != 200:
            raise RuntimeError("prospective screen fixes exactly 200 updates")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("screen is model-only matched fine-tuning, not resume")
        del snapshot, parent_state
        super().__init__(*args, **kwargs)
        if self.resumed or self.transitioned or self._start_iter != 0 or self.optimizer.state:
            raise RuntimeError("both arms require a fresh identical optimizer")
        if (
            not isinstance(self.run_identity_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.run_identity_sha256) is None
        ):
            raise RuntimeError("registered run identity is absent or invalid")
        module = self.model.module
        mode = str(getattr(module, "action_cycle_mode", ""))
        weight = float(getattr(module, "action_cycle_loss_weight", float("nan")))
        if (mode, weight) not in {("off", 0.0), ("on", 0.05)}:
            raise RuntimeError("screen arm must be loss-off/0.0 or loss-on/0.05")
        self._cycle_arm = "AC-OFF" if mode == "off" else "AC-ON"
        self._trace_path = self.save_path.parent / "paired_training_trace.jsonl"
        self._trace_complete = self.save_path.parent / "paired_training_trace_complete.json"
        if self.is_main_process:
            if self._trace_path.exists() or self._trace_complete.exists():
                raise RuntimeError("fresh paired-training trace already exists")
            header = {
                "kind": "action_cycle_stage1_training_trace_header",
                "arm": self._cycle_arm,
                "action_cycle_loss_weight": weight,
                "critic_bundle_sha256": str(
                    getattr(module.action_cycle_critic, "sha256", "")
                ),
                "stage0_registration_identity_sha256": str(
                    getattr(
                        module.action_cycle_critic,
                        "stage0_registration_identity_sha256",
                        "",
                    )
                ),
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_completed_updates": 1000,
                "continuation_updates": 200,
                "run_identity_sha256": self.run_identity_sha256,
                "optimizer_state_policy": "fresh_identical_adamw",
                "initial_optimizer_state_entries": 0,
                "ema_policy": "none_in_parent_and_none_in_both_arms",
                "critic_parameters_in_optimizer": 0,
                "protected_test_accessed": False,
            }
            descriptor = os.open(
                self._trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    (
                        json.dumps(header, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )
                handle.flush()
                os.fsync(handle.fileno())

    def _step(self) -> dict[str, Any]:
        losses = super()._step()
        local = getattr(self.model.module, "paired_audit_exact", None)
        fields = (
            "clip_index",
            "actions",
            "clean_latent",
            "noisy_latent",
            "timesteps",
            "cpu_rng_state_after_forward",
            "cuda_rng_state_after_forward",
        )
        if not isinstance(local, dict) or set(local) != set(fields) or any(
            not isinstance(local.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", local[field]) is None
            for field in fields
        ):
            raise RuntimeError("exact paired-training audit is incomplete")
        if torch.distributed.is_initialized() and self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if len(gathered) != self.world_size or any(
            not isinstance(value, dict) for value in gathered
        ):
            raise RuntimeError("exact paired-training rank audit differs")
        for field in fields:
            payload = [
                {"rank": rank, "tensor_sha256": value.get(field)}
                for rank, value in enumerate(gathered)
            ]
            losses[f"paired_audit/exact_{field}_all_ranks_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        return losses

    def _log(self, metrics: dict[str, Any]) -> None:
        if self.is_main_process:
            record = {
                "kind": "action_cycle_stage1_training_trace_event",
                "arm": self._cycle_arm,
                "total_observations": int(self.metrics.total_observations),
                "metrics": metrics,
            }
            with self._trace_path.open("ab") as handle:
                handle.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )
                handle.flush()
                os.fsync(handle.fileno())
        super()._log(metrics)

    def train(self):
        result = super().train()
        if result == "completed" and self.is_main_process:
            _exclusive_json(
                self._trace_complete,
                {
                    "kind": "action_cycle_stage1_training_trace_complete",
                    "arm": self._cycle_arm,
                    "rows": sum(1 for _ in self._trace_path.open("rb")),
                    "trace_sha256": _sha256(self._trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
