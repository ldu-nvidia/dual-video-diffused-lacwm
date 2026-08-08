"""Guarded trainer for the paired action-variation VPM continuation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from robot_wm.utils.trainer import Trainer

PARENT_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba"
)
NEW_STATE_PREFIX = "action_delta_residual."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().contiguous().cpu()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _load_parent_state_exact(
    model: torch.nn.Module,
    parent_state: Mapping[str, torch.Tensor],
    new_keys: set[str],
) -> None:
    """Load a parent once and prove every overlapping tensor is exact."""

    incompatible = model.load_state_dict(parent_state, strict=False)
    if set(incompatible.missing_keys) != new_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict parent load must miss exactly the new action residual state"
        )
    loaded_parent = {
        key: value for key, value in model.state_dict().items() if key in parent_state
    }
    if _state_sha256(loaded_parent) != _state_sha256(parent_state):
        raise RuntimeError("loaded parent tensors differ from the validated snapshot")


class ActionVariationTrainer(Trainer):
    """Load all parent VPM keys and audit paired data/noise every update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        if (
            not load_path.is_absolute()
            or not load_path.is_file()
            or load_path.is_symlink()
        ):
            raise RuntimeError(
                "action-variation continuation requires a regular parent snapshot"
            )
        expected_sha = os.environ.get("ACTION_VARIATION_VPM_SNAPSHOT_SHA256")
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
        if expected_sha != PARENT_SNAPSHOT_SHA256 or observed_sha != expected_sha:
            raise RuntimeError("historical VPM snapshot digest differs")
        snapshot = torch.load(
            load_path, map_location="cpu", weights_only=True, mmap=True
        )
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
        parent_state = snapshot.get("model")
        model_state = model.state_dict()
        new_keys = {key for key in model_state if key.startswith(NEW_STATE_PREFIX)}
        if (
            not isinstance(parent_state, Mapping)
            or not new_keys
            or set(parent_state) != set(model_state) - new_keys
            or any(
                tuple(parent_state[key].shape) != tuple(model_state[key].shape)
                for key in parent_state
            )
        ):
            raise RuntimeError(
                "new model is not parent VPM plus only the action residual"
            )
        if list(config.get("exclude_keys", [])):
            raise RuntimeError(
                "every parent VPM key must load; exclusions are forbidden"
            )
        if (
            int(config.get("max_iter", -1)) != 200
            or config.get("transition_handoff_path") is not None
            or bool(config.get("share_spatial_attention", False))
        ):
            raise RuntimeError("screen fixes a 200-update model-only continuation")
        module = model
        enabled = bool(module.action_variation_enabled)
        if float(module.action_delta_residual.raw_gate.detach()) != 0.0:
            raise RuntimeError("action residual gate must initialize to exact zero")
        stats_sha = os.environ.get("ACTION_VARIATION_STATS_SHA256")
        if stats_sha != module.action_delta_residual.stats_file_sha256:
            raise RuntimeError("action statistics environment/model digests differ")
        adapter_state = {
            key: value for key, value in model_state.items() if key in new_keys
        }
        initial_adapter_sha256 = _state_sha256(adapter_state)
        _load_parent_state_exact(model, parent_state, new_keys)
        del snapshot, parent_state, model_state, adapter_state

        # The base trainer's historical helper reopens the checkpoint and loads
        # with strict=False.  Suppress that second read: the exact parent state
        # has already been loaded and audited above, before DDP/optimizer setup.
        original_load_path = config.get("load_path")
        config["load_path"] = None
        try:
            super().__init__(*args, **kwargs)
        finally:
            config["load_path"] = original_load_path
        if (
            self.resumed
            or self.transitioned
            or self._start_iter != 0
            or len(self.optimizer.state) != 0
        ):
            raise RuntimeError("both arms require a fresh identical optimizer")
        if (
            not isinstance(self.run_identity_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.run_identity_sha256) is None
        ):
            raise RuntimeError("LACWM_RUN_IDENTITY_SHA256 is absent or invalid")
        module = self.model.module
        if bool(module.action_variation_enabled) != enabled:
            raise RuntimeError("action-variation arm flag changed during construction")
        self._action_variation_arm = "AV-DELTA" if enabled else "AV-CONT"
        self._trace_path = (
            self.save_path.parent / "action_variation_training_trace.jsonl"
        )
        self._trace_complete = (
            self.save_path.parent / "action_variation_training_trace_complete.json"
        )
        if self.is_main_process:
            if self._trace_path.exists() or self._trace_complete.exists():
                raise RuntimeError("fresh action-variation trace output already exists")
            header = {
                "kind": "action_variation_training_trace_header",
                "arm": self._action_variation_arm,
                "residual_enabled": enabled,
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": expected_sha,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_completed_updates": 1000,
                "continuation_updates": 200,
                "run_identity_sha256": self.run_identity_sha256,
                "stats_file_sha256": stats_sha,
                "stats_identity_sha256": module.action_delta_residual.stats_identity_sha256,
                "initial_action_residual_state_sha256": initial_adapter_sha256,
                "initial_effective_gate": float(
                    module.action_delta_residual.effective_gate().detach().cpu()
                ),
                "same_schema_modules_and_forward_calls": True,
                "parent_function_preserved_at_initialization": True,
                "model_calls_per_update": 1,
                "optimizer_state_policy": "fresh_identical_adamw",
                "ema_policy": "none",
                "protected_test_accessed": False,
            }
            descriptor = os.open(
                self._trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    (
                        json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
                handle.flush()
                os.fsync(handle.fileno())

    def _step(self) -> dict[str, Any]:
        losses = super()._step()
        local = getattr(self.model.module, "paired_audit_exact", None)
        paired_input_fields = (
            "actions",
            "future_actions",
            "standardized_action_delta",
            "clean_latent",
            "noisy_latent",
            "timesteps",
            "reference",
        )
        diagnostic_fields = (
            "base_action_latent",
            "delta_action_latent",
            "augmented_action_latent",
            "action_control",
            "z_control",
        )
        fields = (*paired_input_fields, *diagnostic_fields)
        if (
            not isinstance(local, dict)
            or set(local) != set(fields)
            or any(
                not isinstance(local[field], str) or len(local[field]) != 64
                for field in fields
            )
        ):
            raise RuntimeError("exact action-variation training audit is incomplete")
        if torch.distributed.is_initialized() and self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if len(gathered) != self.world_size:
            raise RuntimeError("rank audit count differs")
        for field in fields:
            payload = [
                {"rank": rank, "tensor_sha256": value[field]}
                for rank, value in enumerate(gathered)
            ]
            losses[f"paired_audit/exact_{field}_all_ranks_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        return losses

    def _log(self, metrics: dict[str, Any]) -> None:
        if self.is_main_process:
            event = {
                "kind": "action_variation_training_trace_event",
                "arm": self._action_variation_arm,
                "total_observations": int(self.metrics.total_observations),
                "metrics": metrics,
            }
            with self._trace_path.open("ab") as handle:
                handle.write(
                    (
                        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
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
                    "kind": "action_variation_training_trace_complete",
                    "arm": self._action_variation_arm,
                    "rows": sum(1 for _ in self._trace_path.open("rb")),
                    "trace_sha256": _sha256(self._trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
