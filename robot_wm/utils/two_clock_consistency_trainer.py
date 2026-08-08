"""Guarded Trainer specialization for the paired two-clock consistency continuation screen."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


class TwoClockConsistencyTrainer(Trainer):
    """Reuse the production loop while recording paired-input audit telemetry."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError(
                "two-clock consistency continuation requires the regular "
                "absolute VPM snapshot"
            )
        expected_sha = os.environ.get("TWO_CLOCK_CONSISTENCY_VPM_SNAPSHOT_SHA256")
        if expected_sha != PARENT_SNAPSHOT_SHA256:
            raise RuntimeError("TWO_CLOCK_CONSISTENCY_VPM_SNAPSHOT_SHA256 is absent or differs")
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
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("historical VPM snapshot metadata differs")
        model_state = model.state_dict()
        parent_state = snapshot.get("model")
        if not isinstance(parent_state, dict) or set(parent_state) != set(model_state):
            raise RuntimeError("new model parameter schema is not exactly VPM-compatible")
        mismatched = [
            key
            for key in model_state
            if tuple(model_state[key].shape) != tuple(parent_state[key].shape)
        ]
        if mismatched:
            raise RuntimeError(f"VPM parameter shapes differ: {mismatched[:8]}")
        if list(config.get("exclude_keys", [])):
            raise RuntimeError(
                "two-clock consistency continuation must strictly load every VPM key"
            )
        if int(config.get("max_iter", -1)) != 200:
            raise RuntimeError("the prospective screen fixes exactly 200 updates")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("screen uses a model-only matched fine-tune, not resume")
        del snapshot, parent_state
        super().__init__(*args, **kwargs)
        if (
            self.resumed
            or self.transitioned
            or self._start_iter != 0
            or len(self.optimizer.state) != 0
        ):
            raise RuntimeError(
                "two-clock arms require a fresh optimizer/model-only continuation"
            )
        if (
            not isinstance(self.run_identity_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.run_identity_sha256) is None
        ):
            raise RuntimeError("LACWM_RUN_IDENTITY_SHA256 is absent or invalid")

        module = self.model.module
        weight = float(getattr(module, "consistency_weight", float("nan")))
        if weight not in {0.0, 0.2}:
            raise RuntimeError("screen arm must use consistency weight 0.0 or 0.2")
        self._two_clock_arm = "TC-CONT" if weight == 0.0 else "TC-CONS"
        self._two_clock_trace_path = (
            self.save_path.parent / "two_clock_consistency_training_trace.jsonl"
        )
        self._two_clock_trace_complete = (
            self.save_path.parent
            / "two_clock_consistency_training_trace_complete.json"
        )
        if self.is_main_process:
            if (
                self._two_clock_trace_path.exists()
                or self._two_clock_trace_complete.exists()
            ):
                raise RuntimeError("fresh paired-training trace output already exists")
            header = {
                "kind": "two_clock_consistency_training_trace_header",
                "arm": self._two_clock_arm,
                "consistency_weight": weight,
                "clock_bands": {"high": [0.8, 1.0], "low": [0.0, 0.4]},
                "model_calls_per_update": 2,
                "shared_epsilon_trajectory": True,
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": expected_sha,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_completed_updates": 1000,
                "parent_total_observations": 8000,
                "continuation_updates": 200,
                "run_identity_sha256": self.run_identity_sha256,
                "optimizer_state_policy": "fresh_identical_adamw",
                "initial_optimizer_state_entries": 0,
                "ema_policy": "none_in_historical_lacwm_and_none_in_both_arms",
            }
            data = (json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n").encode()
            descriptor = os.open(
                self._two_clock_trace_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

    def _step(self) -> dict[str, Any]:
        losses = super()._step()
        local = getattr(self.model.module, "paired_audit_exact", None)
        expected_fields = (
            "clip_index",
            "actions",
            "clean_latent",
            "epsilon",
            "sigma_hi",
            "sigma_lo",
            "timestep_hi",
            "timestep_lo",
            "noisy_hi",
            "noisy_lo",
            "cpu_rng_state_after_forward",
            "cuda_rng_state_after_forward",
        )
        if (
            not isinstance(local, dict)
            or tuple(sorted(local)) != tuple(sorted(expected_fields))
            or any(
                not isinstance(local[field], str) or len(local[field]) != 64
                for field in expected_fields
            )
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
        for field in expected_fields:
            payload = [
                {"rank": rank, "tensor_sha256": value.get(field)}
                for rank, value in enumerate(gathered)
            ]
            if any(
                not isinstance(value["tensor_sha256"], str)
                or len(value["tensor_sha256"]) != 64
                for value in payload
            ):
                raise RuntimeError(f"rank audit field is invalid: {field}")
            losses[f"paired_audit/exact_{field}_all_ranks_sha256"] = hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
        return losses

    def _log(self, metrics: dict[str, Any]) -> None:
        if self.is_main_process:
            record = {
                "kind": "two_clock_consistency_training_trace_event",
                "arm": self._two_clock_arm,
                "total_observations": int(self.metrics.total_observations),
                "metrics": metrics,
            }
            with self._two_clock_trace_path.open("ab") as handle:
                handle.write(
                    (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
                handle.flush()
                os.fsync(handle.fileno())
        super()._log(metrics)

    def train(self):
        result = super().train()
        if result == "completed" and self.is_main_process:
            rows = sum(1 for _ in self._two_clock_trace_path.open("rb"))
            _exclusive_json(
                self._two_clock_trace_complete,
                {
                    "kind": "two_clock_consistency_training_trace_complete",
                    "arm": self._two_clock_arm,
                    "rows": rows,
                    "trace_sha256": _sha256(self._two_clock_trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
