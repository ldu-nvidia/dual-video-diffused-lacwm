"""Fail-closed trainer for the prospective IPQ-TC1 matched continuation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from robot_wm.modeling.dual_diffusion.invertible_two_clock import (
    parameter_schema,
    tensor_state_sha256,
)
from robot_wm.utils.trainer import Trainer


PARENT_SNAPSHOT_SHA256 = (
    "de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
)
LEGACY_REJECTED_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f"
)
PARENT_CANONICAL_MODEL_STATE_SHA256 = (
    "d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0"
)
PARENT_RESOLVED_CONFIG_SHA256 = (
    "ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
)
PARENT_TRAINING_SOURCE_COMMIT = "656086686dae723c942a4209a9d71cdb17ed6ccc"
EXCLUDED_PREFIXES = (
    "forward_model.tf_token_adapter",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_velocity_head",
)
PAIRED_FIELDS = (
    "clip_index",
    "actions",
    "clean_latent",
    "canonical_noise",
    "timestep_p",
    "sigma_p",
    "spare_timestep_q",
    "spare_sigma_q",
    "cpu_rng_before_wan",
    "cuda_rng_before_wan",
    "cpu_rng_after_wan",
    "cuda_rng_after_wan",
    "wan_call_count",
)
TREATMENT_FIELDS = (
    "effective_sigma_q",
    "noisy_p",
    "noisy_q",
    "native_noisy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


class InvertibleTwoClockTrainer(Trainer):
    """Train fixed final checkpoints without opening an endpoint split."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        resolved_config = Path(
            str(config.get("parent_resolved_config", ""))
        ).expanduser()
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError("IPQ-TC1 requires an absolute regular parent snapshot")
        if (
            not resolved_config.is_absolute()
            or not resolved_config.is_file()
            or resolved_config.is_symlink()
        ):
            raise RuntimeError("IPQ-TC1 requires the parent resolved config")
        expected_snapshot = os.environ.get("IPQ_TC1_PARENT_SNAPSHOT_SHA256")
        expected_config = os.environ.get("IPQ_TC1_PARENT_CONFIG_SHA256")
        if expected_snapshot != PARENT_SNAPSHOT_SHA256:
            raise RuntimeError("IPQ parent snapshot environment identity differs")
        if expected_config != PARENT_RESOLVED_CONFIG_SHA256:
            raise RuntimeError("IPQ parent config environment identity differs")
        observed_snapshot = _sha256(load_path)
        if observed_snapshot == LEGACY_REJECTED_SNAPSHOT_SHA256:
            raise RuntimeError("legacy f67 VPM is explicitly forbidden")
        if observed_snapshot != PARENT_SNAPSHOT_SHA256:
            raise RuntimeError("faithful de65 VPM snapshot content changed")
        if _sha256(resolved_config) != PARENT_RESOLVED_CONFIG_SHA256:
            raise RuntimeError("faithful parent resolved config content changed")
        snapshot = torch.load(
            load_path, map_location="cpu", weights_only=True, mmap=True
        )
        parent_state = snapshot.get("model")
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
            or snapshot.get("_start_iter") != 1000
            or not isinstance(parent_state, dict)
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("faithful parent checkpoint metadata differs")
        model_state = model.state_dict()
        parent_core = {
            key: value
            for key, value in parent_state.items()
            if not key.startswith(EXCLUDED_PREFIXES)
        }
        model_core = {
            key: value
            for key, value in model_state.items()
            if not key.startswith(EXCLUDED_PREFIXES)
        }
        if set(parent_core) != set(model_core):
            missing = sorted(set(model_core) - set(parent_core))
            unexpected = sorted(set(parent_core) - set(model_core))
            raise RuntimeError(
                f"strict parent core schema differs; missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
        mismatched = [
            key
            for key in model_core
            if tuple(model_core[key].shape) != tuple(parent_core[key].shape)
        ]
        if mismatched:
            raise RuntimeError(f"strict parent core shapes differ: {mismatched[:8]}")
        requested_excludes = tuple(config.get("exclude_keys", []))
        if requested_excludes != EXCLUDED_PREFIXES:
            raise RuntimeError("IPQ-TC1 fixes the exact three auxiliary excludes")
        if int(config.get("max_iter", -1)) != 400:
            raise RuntimeError("IPQ-TC1 fixes exactly 400 updates")
        if bool(config.get("validation", {}).get("save_best", True)):
            raise RuntimeError("IPQ-TC1 forbids validation-based checkpoint selection")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("IPQ-TC1 uses a fresh optimizer, not a transition resume")
        del snapshot, parent_state, parent_core, model_core
        super().__init__(*args, **kwargs)
        if self.val_data_loaders or self.viz_data_loaders:
            raise RuntimeError("IPQ training cannot construct validation/viz loaders")
        if self.resumed or self.transitioned or self._start_iter != 0:
            raise RuntimeError("IPQ arms require a fresh matched continuation")
        if len(self.optimizer.state) != 0:
            raise RuntimeError("IPQ arms require a fresh AdamW optimizer")
        if (
            not isinstance(self.run_identity_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.run_identity_sha256) is None
        ):
            raise RuntimeError("LACWM_RUN_IDENTITY_SHA256 is absent or invalid")
        module = self.model.module
        self._ipq_training_started = time.perf_counter()
        self._ipq_arm = (
            "IPQ-SYNC"
            if module.ipq_arm_mode == "synchronous"
            else "IPQ-INDEP"
        )
        auxiliary_state = {
            key: value
            for key, value in module.state_dict().items()
            if key.startswith(EXCLUDED_PREFIXES)
        }
        self._initial_auxiliary_state_sha256 = tensor_state_sha256(auxiliary_state)
        self._ipq_parameter_schema = parameter_schema(module)
        optimizer_parameter_count = sum(
            parameter.numel()
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        if optimizer_parameter_count != self._ipq_parameter_schema["parameter_count"]:
            raise RuntimeError("optimizer/model parameter capacity differs")
        self._wandb_authorized = os.environ.get("IPQ_TC1_WANDB_AUTHORIZED") == "1"
        self._ipq_trace_path = self.save_path.parent / "ipq_training_trace.jsonl"
        self._ipq_trace_complete = (
            self.save_path.parent / "ipq_training_trace_complete.json"
        )
        if self.is_main_process:
            if self._ipq_trace_path.exists() or self._ipq_trace_complete.exists():
                raise RuntimeError("fresh IPQ trace output already exists")
            header = {
                "kind": "ipq_tc1_training_trace_header",
                "protocol_version": module.ipq_protocol_version,
                "arm": self._ipq_arm,
                "arm_mode": module.ipq_arm_mode,
                "updates": 400,
                "wan_calls_per_update": 1,
                "global_batch_size": self.batch_size * self.world_size,
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                "legacy_parent_rejected_sha256": LEGACY_REJECTED_SNAPSHOT_SHA256,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
                "parent_resolved_config": str(resolved_config.resolve(strict=True)),
                "parent_resolved_config_sha256": PARENT_RESOLVED_CONFIG_SHA256,
                "parent_training_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
                "excluded_prefixes": list(EXCLUDED_PREFIXES),
                "initial_auxiliary_state_sha256": self._initial_auxiliary_state_sha256,
                "model_parameter_schema": self._ipq_parameter_schema,
                "optimizer_parameter_count": optimizer_parameter_count,
                "run_identity_sha256": self.run_identity_sha256,
                "endpoint_split_opened": False,
                "validation_batches": 0,
                "wandb_authorized": self._wandb_authorized,
            }
            self._ipq_trace_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._ipq_trace_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    (json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
                handle.flush()
                os.fsync(handle.fileno())

    def _load_model_snapshot(
        self,
        load_path: str,
        exclude_keys: list[str] | None = None,
        share_spatial_attention: bool = False,
    ) -> None:
        """Strict-load every faithful-parent key outside the new P/Q modules."""

        if share_spatial_attention:
            raise RuntimeError("IPQ-TC1 forbids topology remapping")
        if tuple(exclude_keys or ()) != EXCLUDED_PREFIXES:
            raise RuntimeError("IPQ-TC1 auxiliary exclude list changed")
        snapshot = torch.load(
            load_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=True,
        )
        state = {
            key: value
            for key, value in snapshot["model"].items()
            if not key.startswith(EXCLUDED_PREFIXES)
        }
        missing, unexpected = self.model.module.load_state_dict(state, strict=False)
        expected_missing = {
            key
            for key in self.model.module.state_dict()
            if key.startswith(EXCLUDED_PREFIXES)
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "IPQ strict parent load differs: "
                f"missing={sorted(set(missing) ^ expected_missing)[:8]}, "
                f"unexpected={list(unexpected)[:8]}"
            )

    @torch.inference_mode()
    def _validate(self) -> dict[str, float]:
        """Satisfy the base lifecycle without constructing endpoint data."""

        if self.val_data_loaders or self._val_data_loader_iters:
            raise RuntimeError("IPQ endpoint data became reachable during training")
        self.model.train()
        return {"train_only_no_endpoint_access/loss": 0.0}

    def _step(self) -> dict[str, Any]:
        losses = super()._step()
        local = getattr(self.model.module, "paired_audit_exact", None)
        expected = set(PAIRED_FIELDS) | set(TREATMENT_FIELDS)
        if (
            not isinstance(local, dict)
            or set(local) != expected
            or any(
                not isinstance(local[field], str) or len(local[field]) != 64
                for field in expected
            )
        ):
            raise RuntimeError("IPQ paired-training tensor audit is incomplete")
        if torch.distributed.is_initialized() and self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        for field in sorted(expected):
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
            record = {
                "kind": "ipq_tc1_training_trace_event",
                "arm": self._ipq_arm,
                "iteration": int(metrics.get("iteration", -1)),
                "total_observations": int(self.metrics.total_observations),
                "metrics": metrics,
            }
            with self._ipq_trace_path.open("ab") as handle:
                handle.write(
                    (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
                handle.flush()
                os.fsync(handle.fileno())
        super()._log(metrics)

    def train(self):
        result = super().train()
        elapsed = torch.tensor(
            [time.perf_counter() - self._ipq_training_started],
            device=self.local_rank,
            dtype=torch.float64,
        )
        peak = torch.tensor(
            [
                torch.cuda.max_memory_allocated(self.local_rank),
                torch.cuda.max_memory_reserved(self.local_rank),
            ],
            device=self.local_rank,
            dtype=torch.float64,
        )
        if torch.distributed.is_initialized() and self.world_size > 1:
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        if result == "completed" and self.is_main_process:
            _exclusive_json(
                self._ipq_trace_complete,
                {
                    "kind": "ipq_tc1_training_trace_complete",
                    "arm": self._ipq_arm,
                    "completed_updates": 400,
                    "trace_sha256": _sha256(self._ipq_trace_path),
                    "endpoint_split_opened": False,
                    "protected_test_accessed": False,
                    "wandb_authorized": self._wandb_authorized,
                    "training_wall_seconds_max": float(elapsed.item()),
                    "training_reserved_b200_hours": float(
                        elapsed.item() * self.world_size / 3600.0
                    ),
                    "peak_memory_allocated_bytes": int(peak[0].item()),
                    "peak_memory_reserved_bytes": int(peak[1].item()),
                    "model_parameter_schema": self._ipq_parameter_schema,
                },
            )
        return result
