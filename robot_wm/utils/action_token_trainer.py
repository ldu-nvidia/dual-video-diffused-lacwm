"""Guarded trainer for the paired action-token VPM continuation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
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
NEW_STATE_PREFIX = "action_token_adapter."


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
        # ``view(dtype)`` rejects a zero-dimensional tensor when the target
        # element size differs.  Flatten first so scalar parameters (for
        # example the residual gate) have the same canonical byte treatment
        # as every other dense state tensor.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
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
            "strict parent load must miss exactly the new action-token state"
        )
    loaded_parent = {
        key: value for key, value in model.state_dict().items() if key in parent_state
    }
    if _state_sha256(loaded_parent) != _state_sha256(parent_state):
        raise RuntimeError("loaded parent tensors differ from the validated snapshot")


class ActionTokenTrainer(Trainer):
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
                "action-token continuation requires a regular parent snapshot"
            )
        expected_sha = os.environ.get("ACTION_TOKEN_VPM_SNAPSHOT_SHA256")
        preflight_sha = os.environ.get("ACTION_TOKEN_PARENT_PREFLIGHT_SHA256")
        if expected_sha != PARENT_SNAPSHOT_SHA256 or preflight_sha != expected_sha:
            raise RuntimeError("historical VPM snapshot digest differs")
        try:
            runtime_binding = json.loads(
                os.environ["ACTION_TOKEN_RUNTIME_RECEIPT_BINDING_JSON"]
            )
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("sealed runtime receipt binding is absent") from exc
        runtime_record = (
            runtime_binding.get("record")
            if isinstance(runtime_binding, Mapping)
            else None
        )
        runtime_path = (
            Path(str(runtime_record.get("path", "")))
            if isinstance(runtime_record, Mapping)
            else Path("")
        )
        if (
            not isinstance(runtime_binding, Mapping)
            or set(runtime_binding)
            != {"record", "identity_sha256", "verifier_identity_sha256"}
            or not isinstance(runtime_record, Mapping)
            or set(runtime_record) != {"path", "bytes", "sha256"}
            or not runtime_path.is_absolute()
            or not runtime_path.is_file()
            or runtime_path.is_symlink()
            or runtime_path.resolve(strict=True) != runtime_path
            or runtime_path.stat().st_size != runtime_record.get("bytes")
            or _sha256(runtime_path) != runtime_record.get("sha256")
            or re.fullmatch(
                r"[0-9a-f]{64}", str(runtime_binding.get("identity_sha256", ""))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(runtime_binding.get("verifier_identity_sha256", "")),
            )
            is None
        ):
            raise RuntimeError("sealed runtime receipt binding differs")
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
                "new model is not parent VPM plus only the action-token adapter"
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
        enabled = bool(module.action_token_enabled)
        raw_gate = float(module.action_token_adapter.raw_gate.detach())
        native_gate = float(torch.tanh(module.action_token_adapter.raw_gate).detach())
        if (
            not math.isclose(raw_gate, math.atanh(0.1), rel_tol=0.0, abs_tol=1e-7)
            or not math.isclose(native_gate, 0.1, rel_tol=0.0, abs_tol=1e-7)
            or module.action_token_adapter.raw_gate.requires_grad
        ):
            raise RuntimeError("action-token gate must be frozen at effective 0.1")
        stats_sha = os.environ.get("ACTION_TOKEN_STATS_SHA256")
        if stats_sha != module.action_token_adapter.stats_file_sha256:
            raise RuntimeError("action statistics environment/model digests differ")
        adapter_state = {
            key: value for key, value in model_state.items() if key in new_keys
        }
        initial_adapter_sha256 = _state_sha256(adapter_state)
        total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        action_token_parameter_count = sum(
            parameter.numel()
            for parameter in model.action_token_adapter.parameters()
        )
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
        if bool(module.action_token_enabled) != enabled:
            raise RuntimeError("action-token arm flag changed during construction")
        self._runtime_verification_receipt = dict(runtime_binding)
        self._action_token_arm = "AT-ON" if enabled else "AT-OFF"
        self._trace_path = (
            self.save_path.parent / "action_token_training_trace.jsonl"
        )
        self._trace_complete = (
            self.save_path.parent / "action_token_training_trace_complete.json"
        )
        if self.is_main_process:
            if self._trace_path.exists() or self._trace_complete.exists():
                raise RuntimeError("fresh action-token trace output already exists")
            header = {
                "kind": "action_token_training_trace_header",
                "arm": self._action_token_arm,
                "action_token_enabled": enabled,
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": expected_sha,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_completed_updates": 1000,
                "continuation_updates": 200,
                "run_identity_sha256": self.run_identity_sha256,
                "stats_file_sha256": stats_sha,
                "stats_identity_sha256": module.action_token_adapter.stats_identity_sha256,
                "initial_action_token_state_sha256": initial_adapter_sha256,
                "model_parameter_count": total_parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "action_token_parameter_count": action_token_parameter_count,
                "initial_effective_gate": float(
                    module.action_token_adapter.effective_gate().detach().cpu()
                ),
                "initial_native_gate_before_arm_mask": native_gate,
                "gate_trainable": False,
                "same_schema_modules_and_forward_calls": True,
                "current_code_parent_path_no_op_at_initialization": not enabled,
                "raw_context_geometry": [40, 4096],
                "injection": "add_to_last_40_null_context_tokens_before_wan_text_embedding",
                "wan_pretrained_parameter_shapes_changed": False,
                "historical_forward_bit_identity_claimed": False,
                "runtime_verification_receipt": self._runtime_verification_receipt,
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

    def _write_completion_marker(self) -> None:
        if not self.is_main_process:
            return
        self._atomic_write_json(
            self.completion_path,
            {
                "schema_version": 1,
                "status": "completed",
                "completed_updates": int(self.max_iter),
                "max_iter": int(self.max_iter),
                "run_identity_sha256": self.run_identity_sha256,
                "snapshot": str(self.save_path.resolve(strict=False)),
                "runtime_verification_receipt": self._runtime_verification_receipt,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _step(self) -> dict[str, Any]:
        losses = super()._step()
        local = getattr(self.model.module, "paired_audit_exact", None)
        paired_input_fields = (
            "actions",
            "future_actions",
            "standardized_actions",
            "clean_latent",
            "noisy_latent",
            "timesteps",
            "reference",
            "wan_raw_context",
        )
        diagnostic_fields = (
            "base_action_latent",
            "raw_action_context_tokens",
            "gated_action_context_tokens",
            "action_control",
            "z_control",
            "wan_action_context",
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
            raise RuntimeError("exact action-token training audit is incomplete")
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
                "kind": "action_token_training_trace_event",
                "arm": self._action_token_arm,
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
                    "kind": "action_token_training_trace_complete",
                    "arm": self._action_token_arm,
                    "rows": sum(1 for _ in self._trace_path.open("rb")),
                    "trace_sha256": _sha256(self._trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
