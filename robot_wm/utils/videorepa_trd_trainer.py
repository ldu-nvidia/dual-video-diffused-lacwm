"""Trainer wrapper that persists reviewable TRD telemetry beside W&B."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

import robot_wm.utils.distributed as dist
from robot_wm.utils.trainer import Trainer


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class VideoRepaTRDTrainer(Trainer):
    """Add an identity-bound local JSONL trace without changing optimization."""

    def _load_model_snapshot(
        self, load_path, exclude_keys=None, share_spatial_attention=False
    ):
        """Warm-start the exact VPM state with no silent key exclusions."""

        if exclude_keys not in (None, [], ()) or share_spatial_attention:
            raise RuntimeError("TRD requires an exact, unremapped VPM warm start")
        snapshot = torch.load(
            load_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=True,
        )
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("_start_iter") != 1000
            or snapshot.get("run_identity_sha256")
            != "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
            or not isinstance(snapshot.get("model"), Mapping)
        ):
            raise RuntimeError("TRD warm start is not the frozen update-1000 VPM")
        incompatible = self.model.module.load_state_dict(
            snapshot["model"], strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict VPM warm start failed: {incompatible}")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw_path = os.environ.get("TRD_TRAIN_TRACE")
        if not raw_path:
            raise RuntimeError("TRD_TRAIN_TRACE is required")
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink():
            raise RuntimeError("TRD_TRAIN_TRACE must be an absolute fresh file")
        self.trd_trace_path = path
        self.trd_trace_complete_path = path.with_name(
            "trd_training_trace_complete.json"
        )
        self._trd_trace_records = 0
        if self.is_main_process:
            if (
                path.exists()
                or path.is_symlink()
                or self.trd_trace_complete_path.exists()
                or self.trd_trace_complete_path.is_symlink()
            ):
                raise RuntimeError("TRD telemetry output must be fresh")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    _canonical(
                        {
                            "kind": "videorepa_trd_training_trace_start",
                            "run_identity_sha256": self.run_identity_sha256,
                            "max_iter": self.max_iter,
                            "world_size": self.world_size,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
        if dist.is_initialized():
            dist.barrier()

    def _log(self, metrics):
        super()._log(metrics)
        if not self.is_main_process:
            return
        record = {
            "kind": "videorepa_trd_metric",
            "run_identity_sha256": self.run_identity_sha256,
            "phase": "validation"
            if any(str(key).startswith("val_loss/") for key in metrics)
            else "training",
            "metrics": dict(metrics),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.trd_trace_path.open("ab") as handle:
            handle.write(_canonical(record))
            handle.flush()
            os.fsync(handle.fileno())
        self._trd_trace_records += 1

    def train(self):
        result = super().train()
        if self.is_main_process and result == "completed":
            payload = {
                "schema_version": 1,
                "kind": "videorepa_trd_training_trace_complete",
                "status": "completed",
                "run_identity_sha256": self.run_identity_sha256,
                "completed_updates": self.max_iter,
                "records_after_header": self._trd_trace_records,
                "trace": str(self.trd_trace_path),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            descriptor = os.open(
                self.trd_trace_complete_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_canonical(payload))
                handle.flush()
                os.fsync(handle.fileno())
        if dist.is_initialized():
            dist.barrier()
        return result
