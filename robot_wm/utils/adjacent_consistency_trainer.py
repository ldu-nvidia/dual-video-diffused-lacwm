"""Fail-closed trainer for the matched ACD-P0 direct low-NFE baseline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from lam.adjacent_consistency_model import (
    AdjacentConsistencyVPM,
)
from robot_wm.modeling.low_nfe.adjacent_consistency import (
    CANONICAL_MODEL_STATE_HASH_ALGORITHM,
    RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
    ema_update_module_,
    module_state_receipt,
    receipt_dict,
    require_model_state_hashes,
    state_probe,
    tensor_sha256,
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
PARENT_RUNTIME_TENSOR_STATE_SHA256 = (
    "82ffa76e99574f5202831897411e4b22eb281c1e2512dc358238bef72d5a4d5a"
)
PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM = "snapshot_model_state_receipt_v1"
PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM = "tensor_state_sha256_v1"
PARENT_RESOLVED_CONFIG_SHA256 = (
    "ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
)
PARENT_TRAINING_SOURCE_COMMIT = "656086686dae723c942a4209a9d71cdb17ed6ccc"
PAIRED_INVARIANT_FIELDS = (
    "clip_index",
    "actions",
    "clean_latent",
    "noise",
    "clock_indexes",
    "clock_sigmas",
    "clock_timesteps",
    "source_state",
    "teacher_stepped_state",
    "rf_target",
    "teacher_probe",
    "cpu_rng_before_teacher",
    "cuda_rng_before_teacher",
    "cpu_rng_before_online_student",
    "cuda_rng_before_online_student",
    "cpu_rng_before_ema_target",
    "cuda_rng_before_ema_target",
    "cpu_rng_after_forward",
    "cuda_rng_after_forward",
    "call_ledger",
)
TREATMENT_FIELDS = (
    "online_consistency",
    "target_consistency",
    "ema_probe_before_update",
)
AUDIT_FIELDS = PAIRED_INVARIANT_FIELDS + TREATMENT_FIELDS


def _parent_model_state_lineage() -> dict[str, str]:
    return {
        "canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
        "canonical_hash_algorithm": PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM,
        "runtime_tensor_state_sha256": PARENT_RUNTIME_TENSOR_STATE_SHA256,
        "runtime_hash_algorithm": PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
    }


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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity_valid(value: Mapping[str, Any]) -> bool:
    expected = value.get("identity_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        return False
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == expected


def _registered_context() -> tuple[dict[str, Any], Path]:
    path = Path(os.environ.get("ACD_P0_REGISTRATION_PATH", "")).expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("ACD registration path is absent or invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("ACD registration is unreadable") from exc
    if (
        not isinstance(value, dict)
        or not _identity_valid(value)
        or value.get("kind") != "acd_p0_registration"
        or value.get("identity_sha256")
        != os.environ.get("ACD_P0_REGISTRATION_IDENTITY")
    ):
        raise RuntimeError("ACD registration identity differs")
    # Re-run the source-owned core/arm/config-file derivations; a merely
    # self-consistent replacement JSON is not sufficient.
    from tools.low_nfe_direct_baseline import validate_registration

    validated = validate_registration(path)
    if validated != value:
        raise RuntimeError("ACD registration validation is not bit-exact")
    return value, path.resolve(strict=True)


def validate_runtime_resolved_config(config: Any) -> tuple[str, str]:
    """Bind the actual Hydra job, including all inherited fields, to registration."""

    from omegaconf import OmegaConf

    registration, _path = _registered_context()
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError("resolved ACD Hydra job is not a mapping")
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    mode = str(payload.get("model", {}).get("adjacent_consistency", {}).get("arm_mode"))
    arm = "ACD-RF-CONT" if mode == "rf_control" else "ACD-CONS" if mode == "consistency" else ""
    expected = registration.get("arm_resolved_configs", {}).get(arm, {}).get(
        "semantic_sha256"
    )
    if (
        not arm
        or digest != expected
        or digest != os.environ.get("ACD_P0_ARM_CONFIG_SEMANTIC_SHA256")
        or registration.get("arm_run_identity_sha256", {}).get(arm)
        != os.environ.get("LACWM_RUN_IDENTITY_SHA256")
    ):
        raise RuntimeError("actual full resolved Hydra config differs from registration")
    os.environ["ACD_P0_RUNTIME_CONFIG_VALIDATED_SHA256"] = digest
    os.environ["ACD_P0_RUNTIME_ARM_CODE"] = arm
    return arm, digest


def _validate_registered_training_inputs(
    registration: Mapping[str, Any], *, local_rank: int
) -> dict[str, Any]:
    """Rehash admissible train bytes once per arm, then broadcast the receipt."""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    result: dict[str, Any] | None = None
    if rank == 0:
        try:
            import numpy as np

            training = registration["training"]
            if (
                Path(os.environ.get("ACD_P0_TRAIN_CLIP_MANIFEST", "")).resolve(
                    strict=True
                )
                != Path(training["manifest"]["path"])
                or Path(os.environ.get("ACD_P0_TRAIN_CACHE_METADATA", "")).resolve(
                    strict=True
                )
                != Path(training["cache_metadata"]["path"])
            ):
                raise RuntimeError("live train manifest/cache path differs")
            records: dict[str, Any] = {}
            expected = {
                "rgb": ((512, 13, 3, 180, 960), np.dtype("float16")),
                "actions": ((512, 13, 5, 23), np.dtype("float32")),
            }
            for logical, (shape, dtype) in expected.items():
                record = training["arrays"][logical]
                path = Path(record["path"])
                if (
                    path.is_symlink()
                    or path.stat().st_size != int(record["bytes"])
                    or _sha256(path) != record["sha256"]
                ):
                    raise RuntimeError(f"live training {logical} bytes differ")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if tuple(array.shape) != shape or array.dtype != dtype:
                    raise RuntimeError(f"live training {logical} schema differs")
                del array
                records[logical] = {
                    "path": str(path.resolve(strict=True)),
                    "bytes": int(record["bytes"]),
                    "sha256": record["sha256"],
                    "shape": list(shape),
                    "dtype": str(dtype),
                }
            for logical in ("manifest", "cache_metadata"):
                record = training[logical]
                if _sha256(Path(record["path"])) != record["sha256"]:
                    raise RuntimeError(f"live training {logical} bytes differ")
            result = {"status": "PASS", "rank_zero_actual_byte_hashes": records}
        except BaseException as exc:
            result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    payload: list[Any] = [result]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(payload, src=0)
    observed = payload[0]
    if not isinstance(observed, dict) or observed.get("status") != "PASS":
        raise RuntimeError(f"registered training input verification failed: {observed}")
    return observed


def _gather_exact_role_receipts(
    local: Mapping[str, Mapping[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    payload = {role: dict(receipt) for role, receipt in local.items()}
    if torch.distributed.is_initialized():
        gathered: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if any(not isinstance(item, dict) for item in gathered):
        raise RuntimeError(f"{label} state receipt collective is invalid")
    for role in payload:
        observed = [item.get(role) for item in gathered]
        if any(value != observed[0] for value in observed[1:]):
            raise RuntimeError(f"{label} {role} full receipt differs across ranks")
    state_hashes = {
        item[role]["sha256"]
        for item in gathered
        for role in payload
    }
    if label == "initial" and len(state_hashes) != 1:
        raise RuntimeError("initial student/teacher/EMA state bytes differ")
    return [dict(item) for item in gathered]


def _require_exact_live_source() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[2]
    expected = os.environ.get("ACD_P0_SOURCE_COMMIT", "")
    registration = os.environ.get("ACD_P0_REGISTRATION_IDENTITY", "")
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None or re.fullmatch(
        r"[0-9a-f]{64}", registration
    ) is None:
        raise RuntimeError("ACD exact source/registration environment is absent")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected or status:
        raise RuntimeError("ACD live source is not the registered clean commit")
    return expected, registration


class AdjacentConsistencyTrainer(Trainer):
    """Own external teacher/EMA copies while checkpointing only both students."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._acd_training_started = time.perf_counter()
        self._acd_source_commit, self._acd_registration_identity = (
            _require_exact_live_source()
        )
        self._acd_registration, self._acd_registration_path = _registered_context()
        if (
            self._acd_registration.get("tool_repository", {}).get("git_commit")
            != self._acd_source_commit
        ):
            raise RuntimeError("registered source commit differs from live source")
        registered_parent = self._acd_registration.get("parent")
        if (
            not isinstance(registered_parent, Mapping)
            or any(
                registered_parent.get(key) != value
                for key, value in _parent_model_state_lineage().items()
            )
            or CANONICAL_MODEL_STATE_HASH_ALGORITHM
            != PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM
            or RUNTIME_TENSOR_STATE_HASH_ALGORITHM
            != PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM
        ):
            raise RuntimeError("registered dual model-state lineage differs")
        model = kwargs.get("model") if "model" in kwargs else args[0]
        config = kwargs.get("config") if "config" in kwargs else args[6]
        if not isinstance(model, AdjacentConsistencyVPM):
            raise RuntimeError("ACD trainer requires AdjacentConsistencyVPM")
        expected_arm = (
            "ACD-RF-CONT" if model.acd_arm_mode == "rf_control" else "ACD-CONS"
        )
        self._acd_resolved_config_sha256 = os.environ.get(
            "ACD_P0_RUNTIME_CONFIG_VALIDATED_SHA256", ""
        )
        if (
            os.environ.get("ACD_P0_RUNTIME_ARM_CODE") != expected_arm
            or self._acd_resolved_config_sha256
            != self._acd_registration.get("arm_resolved_configs", {})
            .get(expected_arm, {})
            .get("semantic_sha256")
        ):
            raise RuntimeError("runtime full-config validation did not precede trainer")
        self._acd_training_input_receipt = _validate_registered_training_inputs(
            self._acd_registration, local_rank=int(os.environ.get("LOCAL_RANK", "0"))
        )
        load_path = Path(str(config.get("load_path", ""))).expanduser()
        resolved_config = Path(
            str(config.get("parent_resolved_config", ""))
        ).expanduser()
        save_path = Path(str(config.get("saving", {}).get("save_path", "")))
        if save_path.exists():
            raise RuntimeError("ACD-P0 is a fresh non-resumable controlled screen")
        if not load_path.is_absolute() or not load_path.is_file() or load_path.is_symlink():
            raise RuntimeError("ACD-P0 requires an absolute regular parent snapshot")
        if (
            not resolved_config.is_absolute()
            or not resolved_config.is_file()
            or resolved_config.is_symlink()
        ):
            raise RuntimeError("ACD-P0 requires the faithful parent resolved config")
        if os.environ.get("ACD_P0_PARENT_SNAPSHOT_SHA256") != PARENT_SNAPSHOT_SHA256:
            raise RuntimeError("ACD parent snapshot environment identity differs")
        if os.environ.get("ACD_P0_PARENT_CONFIG_SHA256") != PARENT_RESOLVED_CONFIG_SHA256:
            raise RuntimeError("ACD parent config environment identity differs")
        if (
            os.environ.get("ACD_P0_PARENT_CANONICAL_MODEL_STATE_SHA256")
            != PARENT_CANONICAL_MODEL_STATE_SHA256
            or os.environ.get("ACD_P0_PARENT_RUNTIME_TENSOR_STATE_SHA256")
            != PARENT_RUNTIME_TENSOR_STATE_SHA256
        ):
            raise RuntimeError("ACD parent dual-hash environment identity differs")
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
        model_state = model.state_dict()
        if (
            snapshot.get("snapshot_schema_version") != 3
            or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
            or snapshot.get("_start_iter") != 1000
            or not isinstance(parent_state, dict)
            or set(parent_state) != set(model_state)
            or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        ):
            raise RuntimeError("faithful parent checkpoint metadata/schema differs")
        mismatched = [
            key
            for key in model_state
            if parent_state[key].shape != model_state[key].shape
            or parent_state[key].dtype != model_state[key].dtype
        ]
        if mismatched:
            raise RuntimeError(f"faithful parent tensor schema differs: {mismatched[:8]}")
        try:
            self._parent_state_hash_receipt = require_model_state_hashes(
                parent_state,
                expected_canonical_sha256=PARENT_CANONICAL_MODEL_STATE_SHA256,
                expected_runtime_sha256=PARENT_RUNTIME_TENSOR_STATE_SHA256,
                label="trainer raw parent model state",
            )
        except Exception as exc:
            raise RuntimeError("trainer raw parent model-state hashes differ") from exc
        if self._parent_state_hash_receipt != registered_parent.get(
            "model_state_hash_receipt"
        ):
            raise RuntimeError("runtime raw parent hash receipt differs from registration")
        if list(config.get("exclude_keys", [])):
            raise RuntimeError("ACD-P0 strictly loads every parent model key")
        if int(config.get("max_iter", -1)) != 400:
            raise RuntimeError("ACD-P0 fixes exactly 400 optimizer updates")
        if int(config.get("gradient_accumulation_steps", -1)) != 1:
            raise RuntimeError("ACD-P0 fixes gradient accumulation to one")
        if int(config.get("logging", {}).get("log_every", -1)) != 1:
            raise RuntimeError("ACD-P0 persists an audit event every update")
        if not bool(
            config.get("gradient_clipping", {}).get("error_if_nonfinite", False)
        ):
            raise RuntimeError("ACD-P0 requires fail-closed nonfinite gradients")
        if (
            float(config.get("gradient_clipping", {}).get("max_norm", -1.0))
            != 1.0
            or float(config.get("gradient_clipping", {}).get("norm_type", -1.0))
            != 2.0
        ):
            raise RuntimeError("ACD-P0 fixes gradient clipping to L2 max_norm=1")
        if bool(config.get("validation", {}).get("save_best", True)):
            raise RuntimeError("ACD-P0 forbids validation checkpoint selection")
        if config.get("transition_handoff_path") is not None:
            raise RuntimeError("ACD-P0 uses a fresh optimizer, not transition resume")
        del snapshot, parent_state, model_state
        super().__init__(*args, **kwargs)
        if self.val_data_loaders or self.viz_data_loaders:
            raise RuntimeError("ACD training cannot construct endpoint loaders")
        if self.resumed or self.transitioned or self._start_iter != 0:
            raise RuntimeError("ACD arms require a fresh matched continuation")
        if len(self.optimizer.state) != 0:
            raise RuntimeError("ACD arms require a fresh AdamW optimizer")
        if (
            not isinstance(self.run_identity_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.run_identity_sha256) is None
        ):
            raise RuntimeError("LACWM_RUN_IDENTITY_SHA256 is absent or invalid")
        if self.run_identity_sha256 != self._acd_registration.get(
            "arm_run_identity_sha256", {}
        ).get(expected_arm):
            raise RuntimeError("runtime arm identity differs from registration")
        if self.use_wandb:
            raise RuntimeError("ACD-P0 prepared source forbids W&B")

        module = self.model.module
        # DDP and the optimizer already own only `module`. These two full-model
        # copies are deliberately external/non-registered and never serialized
        # under the student's ordinary `model` key.
        teacher = copy.deepcopy(module)
        target = copy.deepcopy(module)
        for frozen in (teacher, target):
            frozen.requires_grad_(False)
            frozen.eval()
        module.bind_frozen_models(teacher=teacher, target=target)
        self.teacher_model = teacher
        self.target_model = target
        self._student_initial_state_receipt = module_state_receipt(module)
        self._teacher_initial_state_receipt = module_state_receipt(teacher)
        self._target_initial_state_receipt = module_state_receipt(target)
        initial_full_hashes = {
            self._student_initial_state_receipt["sha256"],
            self._teacher_initial_state_receipt["sha256"],
            self._target_initial_state_receipt["sha256"],
        }
        if len(initial_full_hashes) != 1:
            raise RuntimeError("teacher/student/EMA full states are not bit-exact")
        if any(
            receipt.get("hash_algorithm")
            != PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM
            or receipt.get("sha256") != PARENT_RUNTIME_TENSOR_STATE_SHA256
            for receipt in (
                self._student_initial_state_receipt,
                self._teacher_initial_state_receipt,
                self._target_initial_state_receipt,
            )
        ):
            raise RuntimeError(
                "strictly loaded student/teacher/EMA runtime state differs from parent"
            )
        initial_counts = {
            tuple(
                receipt[key]
                for key in (
                    "state_tensors",
                    "state_values",
                    "parameter_tensors",
                    "parameter_values",
                    "buffer_tensors",
                    "buffer_values",
                )
            )
            for receipt in (
                self._student_initial_state_receipt,
                self._teacher_initial_state_receipt,
                self._target_initial_state_receipt,
            )
        }
        if len(initial_counts) != 1:
            raise RuntimeError("teacher/student/EMA capacity receipts differ")
        self._initial_rank_state_receipts = _gather_exact_role_receipts(
            {
                "student": self._student_initial_state_receipt,
                "teacher": self._teacher_initial_state_receipt,
                "ema_target": self._target_initial_state_receipt,
            },
            label="initial",
        )
        self._teacher_initial_probe = tensor_sha256(state_probe(teacher))
        self._target_initial_probe = tensor_sha256(state_probe(target))
        student_probe = tensor_sha256(state_probe(module))
        if not (
            self._teacher_initial_probe
            == self._target_initial_probe
            == student_probe
        ):
            raise RuntimeError("teacher/student/EMA initialization is not bit-exact")
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if any(
            id(parameter) in optimizer_ids
            for frozen in (teacher, target)
            for parameter in frozen.parameters()
        ):
            raise RuntimeError("teacher or EMA target leaked into optimizer")

        self._acd_arm = (
            "ACD-RF-CONT"
            if module.acd_arm_mode == "rf_control"
            else "ACD-CONS"
        )
        self._trace_path = self.save_path.parent / "acd_training_trace.jsonl"
        self._trace_complete = self.save_path.parent / "acd_training_trace_complete.json"
        if self.is_main_process:
            if self._trace_path.exists() or self._trace_complete.exists():
                raise RuntimeError("fresh ACD trace output already exists")
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            header = {
                "kind": "acd_p0_training_trace_header",
                "protocol_version": module.acd_protocol_version,
                "arm": self._acd_arm,
                "arm_mode": module.acd_arm_mode,
                "updates": 400,
                "global_batch_size": self.batch_size * self.world_size,
                "wan_calls_per_update": {
                    "teacher": 1,
                    "online_student": 1,
                    "ema_target": 1,
                },
                "stride": module.acd_stride,
                "sigma_data": module.acd_sigma_data,
                "huber_c": module.acd_huber_c,
                "ema_decay": module.acd_ema_decay,
                "parent_snapshot": str(load_path.resolve(strict=True)),
                "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                "parent_run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                "parent_canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
                "parent_canonical_model_state_hash_algorithm": (
                    PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM
                ),
                "parent_runtime_tensor_state_sha256": (
                    PARENT_RUNTIME_TENSOR_STATE_SHA256
                ),
                "parent_runtime_tensor_state_hash_algorithm": (
                    PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM
                ),
                "parent_model_state_hash_receipt": self._parent_state_hash_receipt,
                "parent_resolved_config": str(resolved_config.resolve(strict=True)),
                "parent_resolved_config_sha256": PARENT_RESOLVED_CONFIG_SHA256,
                "parent_training_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
                "teacher_initial_probe_sha256": self._teacher_initial_probe,
                "ema_initial_probe_sha256": self._target_initial_probe,
                "student_initial_state_receipt": self._student_initial_state_receipt,
                "teacher_initial_state_receipt": self._teacher_initial_state_receipt,
                "ema_initial_state_receipt": self._target_initial_state_receipt,
                "paired_invariant_fields": list(PAIRED_INVARIANT_FIELDS),
                "treatment_fields": list(TREATMENT_FIELDS),
                "run_identity_sha256": self.run_identity_sha256,
                "resolved_arm_config_semantic_sha256": self._acd_resolved_config_sha256,
                "source_commit": self._acd_source_commit,
                "registration_identity_sha256": self._acd_registration_identity,
                "registration_path": str(self._acd_registration_path),
                "training_input_receipt": self._acd_training_input_receipt,
                "initial_rank_state_receipts": self._initial_rank_state_receipts,
                "endpoint_split_opened": False,
                "validation_batches": 0,
                "wandb_enabled": False,
            }
            descriptor = os.open(
                self._trace_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                encoded_header = (
                    json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                handle.write(encoded_header)
                handle.flush()
                os.fsync(handle.fileno())
            self._trace_chain_sha256 = hashlib.sha256(encoded_header).hexdigest()
            self._trace_update_rows = 0
        else:
            self._trace_chain_sha256 = ""
            self._trace_update_rows = 0

    def _load_model_snapshot(
        self,
        load_path: str,
        exclude_keys: list[str] | None = None,
        share_spatial_attention: bool = False,
    ) -> None:
        """Strict-load the complete faithful VPM parent before cloning."""

        if share_spatial_attention or list(exclude_keys or []):
            raise RuntimeError("ACD-P0 forbids topology remapping/excluded keys")
        snapshot = torch.load(
            load_path,
            map_location=f"cuda:{self.local_rank}",
            weights_only=True,
        )
        parent_state = snapshot.get("model")
        if not isinstance(parent_state, dict):
            raise RuntimeError("ACD strict-load snapshot lacks model state")
        try:
            receipt = require_model_state_hashes(
                parent_state,
                expected_canonical_sha256=PARENT_CANONICAL_MODEL_STATE_SHA256,
                expected_runtime_sha256=PARENT_RUNTIME_TENSOR_STATE_SHA256,
                label="strict-load parent model state",
            )
        except Exception as exc:
            raise RuntimeError("ACD strict-load parent model-state hashes differ") from exc
        if receipt != self._parent_state_hash_receipt:
            raise RuntimeError("ACD strict-load parent receipt changed after preflight")
        incompatible = self.model.module.load_state_dict(parent_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("ACD strict parent load returned incompatible keys")
        if (
            tensor_state_sha256(self.model.module.state_dict())
            != PARENT_RUNTIME_TENSOR_STATE_SHA256
        ):
            raise RuntimeError("ACD strictly loaded model runtime state differs")

    def initialize_wandb(self, cfg) -> None:
        """Prepared ACD-P0 source never writes external telemetry."""

        if bool(cfg.wandb.enabled):
            raise RuntimeError("ACD-P0 source fixes W&B disabled")
        self.use_wandb = False

    @torch.inference_mode()
    def _validate(self) -> dict[str, float]:
        if self.val_data_loaders or self._val_data_loader_iters:
            raise RuntimeError("ACD endpoint data became reachable during training")
        self.model.train()
        self.teacher_model.eval()
        self.target_model.eval()
        return {"train_only_no_endpoint_access/loss": 0.0}

    def _step(self) -> dict[str, Any]:
        target_probe_before = tensor_sha256(state_probe(self.target_model))
        amp_scale_before = (
            float(self.scaler.get_scale()) if self.use_amp else None
        )
        losses = super()._step()
        if (
            amp_scale_before is not None
            and float(self.scaler.get_scale()) < amp_scale_before
        ):
            # GradScaler skipped optimizer.step. Never advance EMA against an
            # online student that did not complete the registered update.
            raise RuntimeError("AMP optimizer step was skipped; EMA was not updated")
        local = getattr(self.model.module, "paired_audit_exact", None)
        if (
            not isinstance(local, dict)
            or tuple(sorted(local)) != tuple(sorted(AUDIT_FIELDS))
            or any(
                not isinstance(local[field], str) or len(local[field]) != 64
                for field in AUDIT_FIELDS
            )
            or local["ema_probe_before_update"] != target_probe_before
        ):
            raise RuntimeError("ACD paired/EMA pre-update audit is incomplete")
        receipt = ema_update_module_(
            self.target_model,
            self.model.module,
            decay=self.model.module.acd_ema_decay,
        )
        target_probe_after = tensor_sha256(state_probe(self.target_model))
        teacher_probe_after = tensor_sha256(state_probe(self.teacher_model))
        if teacher_probe_after != self._teacher_initial_probe:
            raise RuntimeError("frozen teacher changed after optimizer step")
        if torch.distributed.is_initialized() and self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(
                gathered,
                {
                    "forward": local,
                    "ema_before": target_probe_before,
                    "ema_after": target_probe_after,
                    "teacher_after": teacher_probe_after,
                },
            )
        else:
            gathered = [
                {
                    "forward": local,
                    "ema_before": target_probe_before,
                    "ema_after": target_probe_after,
                    "teacher_after": teacher_probe_after,
                }
            ]
        if len(gathered) != self.world_size:
            raise RuntimeError("ACD distributed audit rank count differs")
        for field in AUDIT_FIELDS:
            payload = [
                {"rank": rank, "sha256": item["forward"][field]}
                for rank, item in enumerate(gathered)
            ]
            losses[f"paired_audit/exact_{field}_all_ranks_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        for field in ("ema_before", "ema_after", "teacher_after"):
            if len({item[field] for item in gathered}) != 1:
                raise RuntimeError(f"ACD {field} probe differs across ranks")
            payload = [
                {"rank": rank, "sha256": item[field]}
                for rank, item in enumerate(gathered)
            ]
            losses[f"paired_audit/{field}_all_ranks_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        for key, value in receipt_dict(receipt).items():
            if key != "decay":
                losses[f"adjacent_consistency/ema_update_{key}"] = value
        losses["adjacent_consistency/ema_update_decay"] = receipt.decay
        if self.is_main_process:
            serializable = {
                key: (
                    value.detach().item()
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                    else value
                )
                for key, value in losses.items()
            }
            if any(isinstance(value, torch.Tensor) for value in serializable.values()):
                raise RuntimeError("ACD update trace contains a nonscalar tensor")
            record = {
                "kind": "acd_p0_training_trace_event",
                "phase": "optimizer_update",
                "arm": self._acd_arm,
                "iteration": int(self._curr_iter),
                "total_observations": int(self.metrics.total_observations),
                "previous_record_sha256": self._trace_chain_sha256,
                "metrics": serializable,
            }
            encoded = (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            with self._trace_path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._trace_chain_sha256 = hashlib.sha256(encoded).hexdigest()
            self._trace_update_rows += 1
        return losses

    def _build_snapshot(self, rank_states):
        student_receipt, teacher_receipt, target_receipt = (
            self._final_state_receipts()
        )
        snapshot = super()._build_snapshot(rank_states)
        snapshot.update(
            {
                "acd_schema_version": 1,
                "acd_arm": self._acd_arm,
                "target_model": self.target_model.state_dict(),
                "teacher_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                "parent_canonical_model_state_sha256": (
                    PARENT_CANONICAL_MODEL_STATE_SHA256
                ),
                "parent_runtime_tensor_state_sha256": (
                    PARENT_RUNTIME_TENSOR_STATE_SHA256
                ),
                "parent_model_state_hash_receipt": self._parent_state_hash_receipt,
                "teacher_initial_probe_sha256": self._teacher_initial_probe,
                "teacher_final_probe_sha256": tensor_sha256(
                    state_probe(self.teacher_model)
                ),
                "student_final_state_receipt": student_receipt,
                "teacher_final_state_receipt": teacher_receipt,
                "ema_target_probe_sha256": tensor_sha256(
                    state_probe(self.target_model)
                ),
                "ema_target_state_receipt": target_receipt,
                "ema_decay": self.model.module.acd_ema_decay,
                "teacher_serialized": False,
                "registration_identity_sha256": self._acd_registration_identity,
                "resolved_arm_config_semantic_sha256": self._acd_resolved_config_sha256,
                "initial_rank_state_receipts": self._initial_rank_state_receipts,
                "final_rank_state_receipts": self._last_acd_rank_final_receipts,
            }
        )
        return snapshot

    def _gather_rank_states(self):
        rank_states = super()._gather_rank_states()
        student, teacher, target = self._final_state_receipts()
        gathered = _gather_exact_role_receipts(
            {"student": student, "teacher": teacher, "ema_target": target},
            label="checkpoint",
        )
        if len(gathered) != self.world_size or len(rank_states) != self.world_size:
            raise RuntimeError("ACD checkpoint full-state rank family differs")
        for state, receipt in zip(rank_states, gathered, strict=True):
            state["acd_full_state_receipts"] = receipt
        self._last_acd_rank_final_receipts = gathered
        return rank_states

    def _final_state_receipts(self):
        student = module_state_receipt(self.model.module)
        teacher = module_state_receipt(self.teacher_model)
        target = module_state_receipt(self.target_model)
        if teacher != self._teacher_initial_state_receipt:
            raise RuntimeError("immutable teacher full state/capacity changed")
        count_fields = (
            "state_tensors",
            "state_values",
            "parameter_tensors",
            "parameter_values",
            "buffer_tensors",
            "buffer_values",
        )
        if any(
            student[field] != target[field]
            or student[field] != teacher[field]
            for field in count_fields
        ):
            raise RuntimeError("student/teacher/EMA final capacity differs")
        return student, teacher, target

    def train(self):
        result = super().train()
        student_receipt, teacher_receipt, target_receipt = self._final_state_receipts()
        final_rank_receipts = _gather_exact_role_receipts(
            {
                "student": student_receipt,
                "teacher": teacher_receipt,
                "ema_target": target_receipt,
            },
            label="final",
        )
        if len(final_rank_receipts) != self.world_size:
            raise RuntimeError("ACD final full-state rank family differs")
        self._last_acd_rank_final_receipts = final_rank_receipts
        local_wall = time.perf_counter() - self._acd_training_started
        local_peak_reserved = (
            int(torch.cuda.max_memory_reserved(self.local_rank))
            if torch.cuda.is_available()
            else 0
        )
        if torch.distributed.is_initialized() and self.world_size > 1:
            wall_tensor = torch.tensor(
                [local_wall], device=self.local_rank, dtype=torch.float64
            )
            peak_tensor = torch.tensor(
                [local_peak_reserved], device=self.local_rank, dtype=torch.int64
            )
            torch.distributed.all_reduce(
                wall_tensor, op=torch.distributed.ReduceOp.MAX
            )
            torch.distributed.all_reduce(
                peak_tensor, op=torch.distributed.ReduceOp.MAX
            )
            wall_max = float(wall_tensor.item())
            peak_reserved_max = int(peak_tensor.item())
        else:
            wall_max = local_wall
            peak_reserved_max = local_peak_reserved
        if result == "completed" and self.is_main_process:
            if self._trace_update_rows != 400:
                raise RuntimeError("ACD trace does not contain exactly 400 updates")
            _exclusive_json(
                self._trace_complete,
                {
                    "kind": "acd_p0_training_trace_complete",
                    "arm": self._acd_arm,
                    "completed_updates": 400,
                    "trace_sha256": _sha256(self._trace_path),
                    "trace_update_rows": self._trace_update_rows,
                    "trace_final_record_sha256": self._trace_chain_sha256,
                    "parent_canonical_model_state_sha256": (
                        PARENT_CANONICAL_MODEL_STATE_SHA256
                    ),
                    "parent_runtime_tensor_state_sha256": (
                        PARENT_RUNTIME_TENSOR_STATE_SHA256
                    ),
                    "parent_model_state_hash_receipt": (
                        self._parent_state_hash_receipt
                    ),
                    "teacher_initial_probe_sha256": self._teacher_initial_probe,
                    "teacher_final_probe_sha256": tensor_sha256(
                        state_probe(self.teacher_model)
                    ),
                    "student_final_state_receipt": student_receipt,
                    "teacher_final_state_receipt": teacher_receipt,
                    "ema_final_probe_sha256": tensor_sha256(
                        state_probe(self.target_model)
                    ),
                    "ema_target_state_receipt": target_receipt,
                    "initial_rank_state_receipts": self._initial_rank_state_receipts,
                    "final_rank_state_receipts": final_rank_receipts,
                    "registration_identity_sha256": self._acd_registration_identity,
                    "resolved_arm_config_semantic_sha256": self._acd_resolved_config_sha256,
                    "run_identity_sha256": self.run_identity_sha256,
                    "training_input_receipt": self._acd_training_input_receipt,
                    "protected_test_accessed": False,
                    "wandb_writes": 0,
                    "endpoint_split_opened": False,
                    "training_wall_seconds_max": wall_max,
                    "training_reserved_b200_hours": (
                        wall_max * self.world_size / 3600.0
                    ),
                    "peak_memory_reserved_bytes_max": peak_reserved_max,
                },
            )
        return result
