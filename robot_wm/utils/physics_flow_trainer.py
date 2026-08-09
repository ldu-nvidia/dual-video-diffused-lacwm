"""Fail-closed trainer for paired fixed-physics-flow Wan continuations."""

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
EVIDENCE_FIELDS = (
    "actions_sha256",
    "clip_index_sha256",
    "flow_sha256",
    "timesteps_sha256",
    "video_noise_sha256",
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


def _registered_file(path_name: str, digest_name: str) -> tuple[Path, str]:
    path = Path(os.environ.get(path_name, "")).expanduser()
    digest = os.environ.get(digest_name, "")
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or len(digest) != 64
        or _sha256(path) != digest
    ):
        raise RuntimeError(f"registered physics-flow input differs: {path_name}")
    return path.resolve(strict=True), digest


def _shared(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if not key.startswith(PARENT_AUXILIARY_PREFIXES)
    }


def _state_sha256(state: dict[str, Any]) -> str:
    """Hash a small state dictionary without relying on pickle serialization."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"non-tensor model state rejected: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


class PhysicsFlowTrainer(Trainer):
    """Validate the parent/cache and record exact all-rank pairing evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError("physics-flow continuation requires a regular VPM snapshot")
        save_path = Path(str(config.get("saving", {}).get("save_path", ""))).expanduser()
        if not save_path.is_absolute():
            raise RuntimeError("physics-flow save path must be absolute")
        if save_path.resolve(strict=False) == load_path.resolve(strict=True):
            raise RuntimeError("physics-flow output may not overwrite its parent snapshot")
        self._physics_flow_parent_load_path = load_path.resolve(strict=True)
        if (
            os.environ.get("PHYSICS_FLOW_VPM_SNAPSHOT_SHA256")
            != PARENT_SNAPSHOT_SHA256
            or _sha256(load_path) != PARENT_SNAPSHOT_SHA256
        ):
            raise RuntimeError("physics-flow parent snapshot identity differs")
        snapshot = torch.load(load_path, map_location="cpu", weights_only=True, mmap=True)
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("world_size") != 8
            or snapshot.get("gradient_accumulation_steps") != 1
            or snapshot.get("_start_iter") != 1000
            or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("historical VPM metadata differs")
        parent_state = snapshot.get("model")
        model_state = model.state_dict()
        if not isinstance(parent_state, dict):
            raise RuntimeError("historical VPM lacks model state")
        parent_shared = _shared(parent_state)
        model_shared = _shared(model_state)
        if set(parent_shared) != set(model_shared):
            raise RuntimeError("physics-flow shared VPM parameter schema differs")
        mismatched = [
            key
            for key in parent_shared
            if tuple(parent_shared[key].shape) != tuple(model_shared[key].shape)
        ]
        if mismatched:
            raise RuntimeError(f"physics-flow shared VPM shapes differ: {mismatched[:8]}")
        if tuple(config.get("exclude_keys", ())) != PARENT_AUXILIARY_PREFIXES:
            raise RuntimeError("physics-flow must exclude exactly the parent auxiliary schema")
        if int(config.get("max_iter", -1)) != 200:
            raise RuntimeError("physics-flow screen requires exactly 200 updates")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("physics-flow is a fresh model-only continuation")
        train_metadata, train_metadata_sha = _registered_file(
            "PHYSICS_FLOW_TRAIN_FLOW_METADATA",
            "PHYSICS_FLOW_TRAIN_FLOW_METADATA_SHA256",
        )
        val_metadata, val_metadata_sha = _registered_file(
            "PHYSICS_FLOW_VAL_FLOW_METADATA",
            "PHYSICS_FLOW_VAL_FLOW_METADATA_SHA256",
        )
        for path, expected_split in ((train_metadata, "train"), (val_metadata, "val")):
            payload = json.loads(path.read_text())
            if (
                payload.get("schema") != "physics-flow-cache-v1"
                or payload.get("complete") is not True
                or payload.get("split") != expected_split
                or payload.get("renderer_decision") != "GO_FOR_WAN_SCREEN"
                or payload.get("causal_inputs_only") is not True
                or payload.get("future_rgb_opened") is not False
                or payload.get("future_measured_state_opened") is not False
                or payload.get("protected_test_accessed") is not False
            ):
                raise RuntimeError(f"physics-flow {expected_split} cache contract differs")
        del snapshot, parent_state, model_state, parent_shared, model_shared
        super().__init__(*args, **kwargs)
        module = self.model.module
        self._physics_flow_arm = "FLOW-ON" if bool(module.fuse_flow) else "FLOW-OFF"
        self._trace_path = self.save_path.parent / "physics_flow_training_trace.jsonl"
        self._complete_path = self.save_path.parent / "physics_flow_training_trace_complete.json"
        schema = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in module.named_parameters()
        ]
        schema_sha = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        auxiliary_state = {
            key: value
            for key, value in module.state_dict().items()
            if key.startswith(PARENT_AUXILIARY_PREFIXES)
        }
        current_auxiliary_state_sha = _state_sha256(auxiliary_state)
        self._physics_flow_initial_auxiliary_state_sha256 = (
            current_auxiliary_state_sha
        )
        expected_trace_sha = None
        if self.resumed:
            resume_snapshot = torch.load(
                self.save_path, map_location="cpu", weights_only=True, mmap=True
            )
            initial_auxiliary_state_sha = resume_snapshot.get(
                "physics_flow_initial_auxiliary_state_sha256"
            )
            expected_trace_sha = resume_snapshot.get("physics_flow_trace_sha256")
            if (
                not isinstance(initial_auxiliary_state_sha, str)
                or len(initial_auxiliary_state_sha) != 64
                or not isinstance(expected_trace_sha, str)
                or len(expected_trace_sha) != 64
                or resume_snapshot.get("physics_flow_arm")
                != self._physics_flow_arm
            ):
                raise RuntimeError("physics-flow resume checkpoint evidence differs")
            self._physics_flow_initial_auxiliary_state_sha256 = (
                initial_auxiliary_state_sha
            )
            del resume_snapshot
        if self.is_main_process:
            header = {
                    "kind": "physics_flow_training_trace_header",
                    "arm": self._physics_flow_arm,
                    "fuse_flow": bool(module.fuse_flow),
                    "parent_snapshot": str(load_path.resolve(strict=True)),
                    "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                    "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                    "train_flow_metadata": str(train_metadata),
                    "train_flow_metadata_sha256": train_metadata_sha,
                    "val_flow_metadata": str(val_metadata),
                    "val_flow_metadata_sha256": val_metadata_sha,
                    "parameter_schema_sha256": schema_sha,
                    "initial_auxiliary_state_sha256": (
                        self._physics_flow_initial_auxiliary_state_sha256
                    ),
                    "continuation_updates": 200,
                    "wan_calls_per_example": 1,
                    "flow_model_calls_per_example": 0,
                    "flow_clock": 0.0,
                    "flow_velocity_loss": 0.0,
                    "future_measured_state_conditioning": False,
                    "future_rgb_conditioning": False,
                    "optimizer_state_policy": "fresh_identical_adamw",
                    "protected_test_accessed": False,
                }
            if self.resumed:
                self._validate_resumed_trace(header, expected_trace_sha)
            else:
                if self._trace_path.exists() or self._complete_path.exists():
                    raise RuntimeError("fresh physics-flow trace output already exists")
                _exclusive_json(self._trace_path, header)

    def _load_model_snapshot(
        self, load_path, exclude_keys=None, share_spatial_attention=False
    ):
        """Warm-start once, but never overwrite a restored continuation."""

        if Path(load_path).resolve(strict=True) != self._physics_flow_parent_load_path:
            raise RuntimeError("physics-flow parent path changed during construction")
        if self.resumed:
            return None
        return super()._load_model_snapshot(
            load_path,
            exclude_keys=exclude_keys,
            share_spatial_attention=share_spatial_attention,
        )

    def _validate_resumed_trace(
        self, expected_header: dict[str, Any], expected_trace_sha: str
    ) -> None:
        if (
            not self._trace_path.is_file()
            or self._trace_path.is_symlink()
            or self._complete_path.exists()
        ):
            raise RuntimeError("physics-flow resume requires one incomplete regular trace")
        rows = [
            json.loads(line)
            for line in self._trace_path.read_text().splitlines()
            if line.strip()
        ]
        if not rows or rows[0] != expected_header:
            raise RuntimeError("physics-flow resume trace header differs")
        if _sha256(self._trace_path) != expected_trace_sha:
            raise RuntimeError("physics-flow resume trace hash differs from checkpoint")
        train_iterations = [
            int(row["metrics"]["iteration"])
            for row in rows[1:]
            if row.get("kind") == "physics_flow_training_trace_event"
            and isinstance(row.get("metrics"), dict)
            and any(str(key).startswith("train_loss/") for key in row["metrics"])
        ]
        expected_iterations = list(range(int(self._start_iter)))
        if train_iterations != expected_iterations:
            raise RuntimeError(
                "physics-flow trace/checkpoint boundary differs: "
                f"trace={train_iterations[-3:]}, start={self._start_iter}"
            )

    def _build_snapshot(self, rank_states):
        snapshot = super()._build_snapshot(rank_states)
        if not self._trace_path.is_file() or self._trace_path.is_symlink():
            raise RuntimeError("physics-flow trace is unavailable at checkpoint")
        snapshot.update(
            {
                "physics_flow_arm": self._physics_flow_arm,
                "physics_flow_initial_auxiliary_state_sha256": (
                    self._physics_flow_initial_auxiliary_state_sha256
                ),
                "physics_flow_trace_sha256": _sha256(self._trace_path),
            }
        )
        return snapshot

    def _step(self) -> dict[str, Any]:
        losses = super()._step()
        local = getattr(self.model.module, "_physics_flow_step_evidence", None)
        if (
            not isinstance(local, dict)
            or tuple(sorted(local)) != tuple(sorted(EVIDENCE_FIELDS))
            or any(not isinstance(local[field], str) or len(local[field]) != 64 for field in EVIDENCE_FIELDS)
        ):
            raise RuntimeError("exact physics-flow paired audit is incomplete")
        if torch.distributed.is_initialized() and self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if len(gathered) != self.world_size or any(not isinstance(x, dict) for x in gathered):
            raise RuntimeError("physics-flow all-rank audit inventory differs")
        for field in EVIDENCE_FIELDS:
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
                "kind": "physics_flow_training_trace_event",
                "arm": self._physics_flow_arm,
                "total_observations": int(self.metrics.total_observations),
                "metrics": metrics,
            }
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        super()._log(metrics)

    def train(self):
        result = super().train()
        if result == "completed" and self.is_main_process:
            _exclusive_json(
                self._complete_path,
                {
                    "kind": "physics_flow_training_trace_complete",
                    "arm": self._physics_flow_arm,
                    "rows": sum(1 for _ in self._trace_path.open("rb")),
                    "trace_sha256": _sha256(self._trace_path),
                    "completed_updates": 200,
                    "protected_test_accessed": False,
                },
            )
        return result
