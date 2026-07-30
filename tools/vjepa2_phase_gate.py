#!/usr/bin/env python3
"""Fail-closed, eight-B200 phase gate for the V-JEPA 2.1 J0 study.

This is deliberately a *real model* gate rather than a tensor-only smoke test.
It composes the J0 Hydra configuration, instantiates the production
``Trainer``/DDP stack and immutable ABC cache, strictly loads the pinned LACWM
warm start, performs four optimizer updates, and then calls the public
future-only sampler at NFE=1.

The V-JEPA teacher is never imported or constructed.  It was used only to
produce the immutable offline training target.  Autonomous inference begins
the auxiliary state from noise and receives exactly five observed RGB frames.

LACWM clock convention: sigma=1 is Gaussian noise and sigma=0 is clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_VERSION = 1
EXPECTED_WORLD_SIZE = 8
EXPECTED_UPDATES = 4
EXPECTED_BATCH_SHAPES = {
    "rgb": (1, 13, 3, 180, 960),
    "actions": (1, 13, 5, 157),
    "mask": (1, 13),
    "auxiliary_target": (1, 64, 4, 24, 120),
    "clip_index": (1,),
    "morphology_index": (1,),
}
EXPECTED_CACHE_TAIL_SHAPES = {
    "target_shape": (64, 4, 24, 120),
    "rgb_shape": (13, 3, 180, 960),
    "actions_shape": (13, 5, 23),
}
EXPECTED_RESET_PREFIXES = (
    "inverse_model",
    "rgb_pos_embed",
    "action_decoder",
    "action_pos_embed",
    "action_pool",
    "morphology_tokens",
    "forward_model.tf_token_adapter",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_velocity_head",
)
# The pinned legacy LACWM warm start predates the explicit three-layer action
# encoder.  These six tensors are therefore newly initialized.  Keep this as
# an exact-key contract rather than adding the broad ``action_encoder`` module
# to EXPECTED_RESET_PREFIXES: any future action-encoder architecture drift must
# fail the phase gate.
EXPECTED_WARMSTART_MISSING_KEYS = (
    "action_encoder.net.0.bias",
    "action_encoder.net.0.weight",
    "action_encoder.net.2.bias",
    "action_encoder.net.2.weight",
    "action_encoder.net.4.bias",
    "action_encoder.net.4.weight",
)
EXPECTED_GRADIENT_PARAMETERS = {
    "auxiliary_state_gate": "forward_model.tf_token_adapter.gate",
    "auxiliary_state_projection": (
        "forward_model.tf_token_adapter.projection.weight"
    ),
    "auxiliary_state_norm": "forward_model.tf_token_adapter.norm.weight",
    "auxiliary_clock_gate": "forward_model.tf_clock_embedding.gate",
    "auxiliary_clock_network": (
        "forward_model.tf_clock_embedding.net.2.weight"
    ),
    "auxiliary_velocity_head": (
        "forward_model.tf_velocity_head.linear.weight"
    ),
    "action_control": "forward_model.action_to_control.net.0.weight",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class GateError(RuntimeError):
    """A phase-gate contract failed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def exclusive_write(path: Path, content: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(path, mode)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    exclusive_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise GateError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise GateError(f"{label} must be a non-symlink regular file: {path}")
    if info.st_size <= 0:
        raise GateError(f"{label} is empty: {path}")
    return path.resolve(strict=True)


def canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise GateError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise GateError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def file_record(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mtime_ns": int(info.st_mtime_ns),
    }


def runtime_python_record(
    executable: str | Path | None = None,
) -> dict[str, Any]:
    """Record the real binary without using it to execute the venv.

    A venv's ``bin/python`` may be a symlink to base CPython. Executing the
    symlink activates ``pyvenv.cfg`` semantics, while executing its resolved
    target does not. Provenance nevertheless uses the canonical target so it
    agrees with the controlled-study input record.
    """
    requested = Path(sys.executable if executable is None else executable)
    resolved = requested.expanduser().resolve(strict=True)
    return file_record(canonical_file(resolved, "runtime Python"))


def unchanged_file(record: Mapping[str, Any]) -> bool:
    path = Path(str(record["path"]))
    try:
        info = path.stat()
    except FileNotFoundError:
        return False
    return (
        int(info.st_size) == int(record["bytes"])
        and int(info.st_dev) == int(record["device"])
        and int(info.st_ino) == int(record["inode"])
        and int(info.st_mtime_ns) == int(record["mtime_ns"])
    )


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise GateError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def validate_repository(repo: Path, expected_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise GateError("expected commit must be a full lowercase SHA-1")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise GateError(f"repository commit differs: {actual} != {expected_commit}")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise GateError(
            "repository must be clean: " + dirty.replace("\n", "; ")
        )
    return {"path": str(repo), "commit": actual, "clean": True}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError(f"{label} must contain a JSON object")
    return payload


def validate_cache_build_links(
    *,
    complete_path: Path,
    train_manifest: Path,
    train_metadata: Path,
    pca_path: Path,
) -> dict[str, Any]:
    """Validate the immutable build record before the expensive array audit."""
    complete = _load_json(complete_path, "cache build completion record")
    if (
        complete.get("artifact_type") != "vjepa2.1-immutable-cache-build"
        or int(complete.get("format_version", -1)) != 1
    ):
        raise GateError("unexpected cache build completion schema")

    split = complete.get("splits", {}).get("train")
    if not isinstance(split, dict):
        raise GateError("cache build completion record lacks the train split")
    if Path(str(split.get("metadata", ""))).resolve() != train_metadata:
        raise GateError("phase-gate training metadata differs from complete.json")
    metadata_digest = sha256_file(train_metadata)
    if split.get("metadata_sha256") != metadata_digest:
        raise GateError("training cache metadata digest differs from complete.json")

    if Path(str(complete.get("pca", ""))).resolve() != pca_path:
        raise GateError("phase-gate PCA artifact differs from complete.json")
    pca_digest = sha256_file(pca_path)
    if complete.get("pca_sha256") != pca_digest:
        raise GateError("PCA digest differs from complete.json")

    metadata = _load_json(train_metadata, "training cache metadata")
    if metadata.get("complete") is not True:
        raise GateError("training cache metadata is not marked complete")
    if Path(str(metadata.get("clip_manifest", ""))).resolve() != train_manifest:
        raise GateError("training manifest path differs from cache metadata")
    if metadata.get("clip_manifest_sha256") != sha256_file(train_manifest):
        raise GateError("training manifest digest differs from cache metadata")
    if int(metadata.get("clip_count", -1)) != 512:
        raise GateError("phase gate requires the pinned 512-clip training split")
    for key, tail in EXPECTED_CACHE_TAIL_SHAPES.items():
        shape = tuple(int(value) for value in metadata.get(key, ()))
        if shape[1:] != tail:
            raise GateError(
                f"cached {key} differs: {shape}; expected [N,{','.join(map(str, tail))}]"
            )
    for key in ("cache_id", "target_sha256", "rgb_sha256", "actions_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise GateError(f"training cache metadata has invalid {key}")
        if split.get(key) != value:
            raise GateError(f"training cache {key} differs from complete.json")
    if int(split.get("clip_count", -1)) != 512:
        raise GateError("complete.json training clip count is not 512")

    return {
        "complete": file_record(complete_path),
        "train_manifest": file_record(train_manifest),
        "train_metadata": file_record(train_metadata),
        "pca": file_record(pca_path),
        "cache_id": metadata["cache_id"],
        "target_sha256": metadata["target_sha256"],
        "rgb_sha256": metadata["rgb_sha256"],
        "actions_sha256": metadata["actions_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_commit": metadata["source_commit"],
    }


def strict_warmstart_load(
    model: Any,
    warmstart: Path,
    allowed_reset_prefixes: Sequence[str],
    *,
    expected_missing_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Load every compatible warm-start key and permit only named resets.

    ``strict=False`` by itself is intentionally insufficient here: the returned
    incompatibility lists are checked against an exact, reviewable prefix
    allowlist.  Non-prefix model keys may be absent only when the complete
    observed set exactly equals ``expected_missing_keys``; this prevents a
    broad module prefix from silently accepting later architecture drift.
    """
    import torch

    snapshot = torch.load(warmstart, map_location="cpu", weights_only=True)
    if not isinstance(snapshot, Mapping) or not isinstance(
        snapshot.get("model"), Mapping
    ):
        raise GateError("warm start must contain a mapping at snapshot['model']")
    checkpoint_state = dict(snapshot["model"])
    # Historical model snapshots may contain this removed, parameter-free
    # scheduler marker.  The model's own load override accepts it as legacy.
    checkpoint_state.pop("loss_scheduler", None)
    current_state = model.state_dict()

    def is_reset(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in allowed_reset_prefixes)

    declared_missing = list(expected_missing_keys)
    if any(not isinstance(name, str) or not name for name in declared_missing):
        raise GateError("expected warm-start missing keys must be non-empty strings")
    if len(set(declared_missing)) != len(declared_missing):
        raise GateError("expected warm-start missing keys contain duplicates")
    declared_missing = sorted(declared_missing)
    non_model_declarations = sorted(set(declared_missing) - set(current_state))
    if non_model_declarations:
        raise GateError(
            "expected warm-start missing keys are not exact model tensors: "
            f"{non_model_declarations[:8]}"
        )
    prefix_declarations = [name for name in declared_missing if is_reset(name)]
    if prefix_declarations:
        raise GateError(
            "expected warm-start missing keys overlap reset prefixes: "
            f"{prefix_declarations[:8]}"
        )

    accepted: dict[str, Any] = {}
    explicitly_reset_checkpoint_keys: list[str] = []
    unexpected: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for name, value in checkpoint_state.items():
        if is_reset(name):
            explicitly_reset_checkpoint_keys.append(name)
            continue
        if name not in current_state:
            unexpected.append(name)
            continue
        if tuple(value.shape) != tuple(current_state[name].shape):
            shape_mismatches.append(
                {
                    "name": name,
                    "checkpoint": list(value.shape),
                    "model": list(current_state[name].shape),
                }
            )
            continue
        accepted[name] = value

    missing = sorted(set(current_state) - set(accepted))
    reset_model_keys = [name for name in missing if is_reset(name)]
    observed_non_prefix_missing = [
        name for name in missing if not is_reset(name)
    ]
    disallowed_missing = sorted(
        set(observed_non_prefix_missing) - set(declared_missing)
    )
    expected_but_not_missing = sorted(
        set(declared_missing) - set(observed_non_prefix_missing)
    )
    if (
        unexpected
        or shape_mismatches
        or observed_non_prefix_missing != declared_missing
    ):
        raise GateError(
            "strict warm-start audit failed: "
            f"unexpected={unexpected[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}, "
            f"disallowed_missing={disallowed_missing[:8]}, "
            f"expected_but_not_missing={expected_but_not_missing[:8]}"
        )
    incompatible = model.load_state_dict(accepted, strict=False)
    if incompatible.unexpected_keys:
        raise GateError(
            f"strict warm-start load returned unexpected keys: "
            f"{incompatible.unexpected_keys}"
        )
    if sorted(incompatible.missing_keys) != missing:
        raise GateError("strict warm-start load missing-key report changed")
    if not accepted:
        raise GateError("strict warm-start audit accepted no model tensors")
    return {
        "checkpoint_tensor_count": len(checkpoint_state),
        "loaded_tensor_count": len(accepted),
        "loaded_parameter_bytes": sum(
            int(value.numel() * value.element_size())
            for value in accepted.values()
        ),
        "reset_prefixes": list(allowed_reset_prefixes),
        "expected_missing_keys": declared_missing,
        "non_prefix_missing_keys": observed_non_prefix_missing,
        "reset_model_keys": reset_model_keys,
        "excluded_checkpoint_key_count": len(explicitly_reset_checkpoint_keys),
    }


def validate_batch_shapes(batch: Mapping[str, Any]) -> dict[str, list[int]]:
    observed: dict[str, list[int]] = {}
    for key, expected in EXPECTED_BATCH_SHAPES.items():
        if key not in batch or not hasattr(batch[key], "shape"):
            raise GateError(f"training batch lacks tensor {key!r}")
        shape = tuple(int(value) for value in batch[key].shape)
        if shape != expected:
            raise GateError(
                f"training batch {key} shape differs: {shape} != {expected}"
            )
        observed[key] = list(shape)
    morphology = batch["morphology_index"]
    if str(morphology.dtype) not in {"torch.int64", "torch.long"}:
        raise GateError(
            "ABC morphology_index must be an int64 tensor, got "
            f"{morphology.dtype}"
        )
    morphology_value = int(morphology.reshape(-1)[0].item())
    if morphology_value != 9:
        raise GateError(
            f"ABC morphology_index must be exactly 9, got {morphology_value}"
        )
    return observed


def validate_unique_clip_indices(indices: Sequence[int], world_size: int) -> list[int]:
    observed = [int(value) for value in indices]
    if len(observed) != world_size:
        raise GateError(
            f"clip-index audit expected {world_size} ranks, got {len(observed)}"
        )
    if len(set(observed)) != world_size:
        raise GateError(
            f"shape-audit batches are not unique across ranks: {observed}"
        )
    if any(value < 0 or value >= 512 for value in observed):
        raise GateError(f"shape-audit clip index is outside [0,512): {observed}")
    return observed


def validate_j0_config(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(cfg, resolve=True)
    model = resolved["model"]
    dual = model["dual_diffusion"]
    expected = {
        "model_target": "lam.dual_explicit_action_dit_model.DualExplicitActionDiTModel",
        "dual_enabled": True,
        "auxiliary_history_mode": "diffuse_all",
        "tf_channels": 64,
        "condition_mode": "matched",
        "condition_on_tf": True,
        "condition_on_tf_clock": True,
        "schedule_mode": "aligned",
        "tf_lead_logit": 0.0,
        "tf_loss_weight": 1.0,
        "state_gate_init": 0.0,
        "state_gate_trainable": True,
        "clock_gate_init": 0.0,
        "clock_gate_trainable": True,
        "head_condition_on_tf_clock": True,
        "video_only_control": False,
        "parameter_matched_control": False,
        "time_frequency_transform": None,
    }
    observed = {
        "model_target": model["_target_"],
        "dual_enabled": dual["enabled"],
        "auxiliary_history_mode": dual["auxiliary_history_mode"],
        "tf_channels": dual["tf_channels"],
        "condition_mode": dual["condition_mode"],
        "condition_on_tf": dual["condition_on_tf"],
        "condition_on_tf_clock": dual["condition_on_tf_clock"],
        "schedule_mode": dual["schedule_mode"],
        "tf_lead_logit": dual["tf_lead_logit"],
        "tf_loss_weight": dual["tf_loss_weight"],
        "state_gate_init": dual["state_gate_init"],
        "state_gate_trainable": dual["state_gate_trainable"],
        "clock_gate_init": dual["clock_gate_init"],
        "clock_gate_trainable": dual["clock_gate_trainable"],
        "head_condition_on_tf_clock": dual["head_condition_on_tf_clock"],
        "video_only_control": dual["video_only_control"],
        "parameter_matched_control": dual["parameter_matched_control"],
        "time_frequency_transform": model["time_frequency_transform"],
    }
    if observed != expected:
        raise GateError(f"resolved configuration is not exact J0: {observed}")
    if list(dual["evaluation_nfe_steps"]) != [1]:
        raise GateError("phase gate must evaluate exactly NFE=1")
    if list(dual["evaluation_condition_sources"]) != ["autonomous"]:
        raise GateError("phase gate inference must be autonomous only")
    if bool(dual["capture_latent_trajectories"]):
        raise GateError("phase gate must disable latent trajectory capture")
    if resolved["wandb"]["enabled"] or resolved["wandb"]["mode"] != "disabled":
        raise GateError("phase gate must disable W&B")
    if resolved["trainer"]["config"]["load_path"] is not None:
        raise GateError("Trainer's permissive warm-start loader must be disabled")
    reset = tuple(resolved["trainer"]["config"]["exclude_keys"])
    if reset != EXPECTED_RESET_PREFIXES:
        raise GateError(f"warm-start reset prefixes changed: {reset}")
    if resolved["dataset"]["img_augment"]:
        raise GateError("phase gate cache must not use image augmentation")
    child = resolved["dataset"]["datasets"]["ABC"]
    if child["transform"] is not None:
        raise GateError("phase gate must read immutable cached tensors directly")
    return observed


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _first_lora_parameter(model: Any) -> tuple[str, Any]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name:
            return name, parameter
    raise GateError("J0 model has no trainable Wan LoRA parameter")


def gradient_representatives(model: Any) -> dict[str, tuple[str, Any]]:
    parameters = dict(model.named_parameters())
    output: dict[str, tuple[str, Any]] = {}
    for group, name in EXPECTED_GRADIENT_PARAMETERS.items():
        parameter = parameters.get(name)
        if parameter is None or not parameter.requires_grad:
            raise GateError(
                f"required trainable gradient parameter is unavailable: {name}"
            )
        output[group] = (name, parameter)
    output["shared_video_lora"] = _first_lora_parameter(model)
    return output


def _optimizer_step_value(optimizer: Any, parameter: Any) -> int:
    state = optimizer.state.get(parameter)
    if not state or "step" not in state:
        return 0
    value = state["step"]
    return int(value.item() if hasattr(value, "item") else value)


def _capture_video_output(model: Any, batch: Mapping[str, Any]) -> Any:
    import torch

    captured: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if not hasattr(output, "video_velocity"):
            raise GateError("dual forward did not return a video velocity")
        captured.append(output.video_velocity.detach().clone())

    handle = model.forward_model.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            loss = model(**batch)
        if not torch.isfinite(loss):
            raise GateError("zero-gate equivalence forward produced non-finite loss")
    finally:
        handle.remove()
    if len(captured) != 1:
        raise GateError(f"expected one video output capture, got {len(captured)}")
    return captured[0]


def zero_gate_equivalence(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that enabled J0 injection is an exact no-op at zero gates."""
    import torch

    state_gate = model.forward_model.tf_token_adapter.effective_gate()
    clock_gate = model.forward_model.tf_clock_embedding.effective_gate()
    if float(state_gate.detach().float().item()) != 0.0:
        raise GateError("J0 state gate is not exact zero before training")
    if float(clock_gate.detach().float().item()) != 0.0:
        raise GateError("J0 clock gate is not exact zero before training")
    if not model.condition_on_tf or not model.condition_on_tf_clock:
        raise GateError("J0 injection flags are not enabled")

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(torch.cuda.current_device())
    was_training = model.training
    model.train()
    enabled = _capture_video_output(model, batch)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, torch.cuda.current_device())

    original_state = model.condition_on_tf
    original_clock = model.condition_on_tf_clock
    original_forward_state = model.forward_model.condition_on_tf
    original_forward_clock = model.forward_model.condition_on_tf_clock
    try:
        model.condition_on_tf = False
        model.condition_on_tf_clock = False
        model.forward_model.condition_on_tf = False
        model.forward_model.condition_on_tf_clock = False
        disabled = _capture_video_output(model, batch)
    finally:
        model.condition_on_tf = original_state
        model.condition_on_tf_clock = original_clock
        model.forward_model.condition_on_tf = original_forward_state
        model.forward_model.condition_on_tf_clock = original_forward_clock
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, torch.cuda.current_device())
        model.train(was_training)

    difference = (enabled.float() - disabled.float()).abs()
    max_abs = float(difference.max().item())
    exact = bool(torch.equal(enabled, disabled))
    if not exact:
        raise GateError(
            f"zero-gate J0 video output differs from injection-off: max={max_abs}"
        )
    return {
        "state_gate": 0.0,
        "clock_gate": 0.0,
        "video_output_shape": list(enabled.shape),
        "bitwise_equal": exact,
        "max_absolute_difference": max_abs,
    }


def four_optimizer_updates(trainer: Any) -> dict[str, Any]:
    """Run four production Trainer/DDP optimizer updates and audit gradients."""
    import torch

    module = trainer.model.module
    representatives = gradient_representatives(module)
    current_update = {"value": -1}
    histories: dict[str, list[dict[str, Any]]] = {
        group: [
            {"calls": 0, "finite": True, "nonzero": False, "l2": 0.0}
            for _ in range(EXPECTED_UPDATES)
        ]
        for group in representatives
    }
    handles = []
    for group, (_name, parameter) in representatives.items():
        def record(gradient: Any, *, group_name: str = group) -> Any:
            update = current_update["value"]
            if update < 0 or update >= EXPECTED_UPDATES:
                raise GateError("gradient hook fired outside an optimizer update")
            item = histories[group_name][update]
            item["calls"] += 1
            finite = bool(torch.isfinite(gradient).all().item())
            norm = float(gradient.detach().float().norm().item())
            item["finite"] = bool(item["finite"] and finite)
            item["nonzero"] = bool(item["nonzero"] or norm > 0.0)
            item["l2"] += norm
            return gradient

        handles.append(parameter.register_hook(record))

    lora_name, lora_parameter = representatives["shared_video_lora"]
    losses: list[dict[str, float]] = []
    optimizer_steps: list[int] = []
    scaler_scales: list[dict[str, float]] = []
    trainer.model.train()
    try:
        for update in range(EXPECTED_UPDATES):
            current_update["value"] = update
            trainer._curr_iter = update
            scale_before = float(trainer.scaler.get_scale())
            step_losses = trainer._step()
            scale_after = float(trainer.scaler.get_scale())
            if scale_after < scale_before:
                raise GateError(
                    f"AMP skipped optimizer update {update + 1}: "
                    f"{scale_before} -> {scale_after}"
                )
            if not all(float("-inf") < float(value) < float("inf") for value in step_losses.values()):
                raise GateError(f"optimizer update {update + 1} produced non-finite losses")
            step_value = _optimizer_step_value(trainer.optimizer, lora_parameter)
            if step_value != update + 1:
                raise GateError(
                    f"Adam did not complete update {update + 1} for {lora_name}: "
                    f"step={step_value}"
                )
            losses.append({key: float(value) for key, value in step_losses.items()})
            optimizer_steps.append(step_value)
            scaler_scales.append({"before": scale_before, "after": scale_after})
    finally:
        current_update["value"] = -1
        for handle in handles:
            handle.remove()

    gradient_report: dict[str, Any] = {}
    for group, (name, _parameter) in representatives.items():
        updates = histories[group]
        if not all(item["calls"] >= 1 for item in updates):
            raise GateError(f"{group} gradient was absent on one or more updates")
        if not all(item["finite"] for item in updates):
            raise GateError(f"{group} gradient was non-finite")
        if not any(item["nonzero"] for item in updates):
            raise GateError(f"{group} never received a nonzero gradient")
        gradient_report[group] = {"parameter": name, "updates": updates}

    return {
        "completed_updates": EXPECTED_UPDATES,
        "optimizer_anchor": lora_name,
        "optimizer_step_values": optimizer_steps,
        "scaler": scaler_scales,
        "losses": losses,
        "gradient_reachability": gradient_report,
    }


def future_free_nfe1(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    history = batch["rgb"][:, : model.num_history_frames]
    if tuple(history.shape[1:]) != (5, 3, 180, 960):
        raise GateError(f"deployable history shape differs: {tuple(history.shape)}")
    actions = batch["actions"]
    sample_ids = batch["clip_index"]
    morphology = batch.get("morphology_index")
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        prediction = model.sample_future_deployable(
            history,
            actions,
            morphology,
            collect_artifacts=True,
            sample_ids=sample_ids,
        )
    deployable_counters = dict(model._last_sampling_counters)
    deployable_artifacts = model.pop_visualization_artifacts()
    if not isinstance(deployable_artifacts, Mapping):
        raise GateError("deployable NFE=1 audit did not expose its initial noise")
    expected = (history.shape[0], 3, model.num_future_frames, 180, 960)
    if tuple(prediction.shape) != expected:
        raise GateError(
            f"deployable NFE=1 output shape differs: {tuple(prediction.shape)} "
            f"!= {expected}"
        )
    if not torch.isfinite(prediction).all():
        raise GateError("deployable NFE=1 output is non-finite")
    calls = deployable_counters.get("wan_calls_by_source_nfe", {})
    if (
        deployable_counters.get("online_teacher_calls") != 0
        or deployable_counters.get("auxiliary_clean_available") != 0
        or deployable_counters.get("deployment_mode") != 1
        or deployable_counters.get("artifacts_collected") != 1
        or deployable_counters.get("wan_calls_total") != 1
        or sum(int(value) for value in calls.values()) != 1
    ):
        raise GateError(
            "future-free NFE=1 sampler counters differ: "
            f"{deployable_counters}"
        )

    # The ordinary full-clip scoring path is allowed to decode a held-out
    # target after generation, but its generated rollout must be identical to
    # the public history-only path.  No auxiliary target is passed, autonomous
    # is the sole source, and immutable clip IDs key the exact same video/JEPA
    # initial noise.  This is an audit call, never a latency measurement.
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        scored_prediction, scored_ground_truth = model._sample_future(
            batch["rgb"],
            actions,
            morphology,
            auxiliary_target=None,
            collect_artifacts=True,
            deployment_mode=False,
            sample_ids=sample_ids,
        )
    scoring_counters = dict(model._last_sampling_counters)
    scoring_artifacts = model.pop_visualization_artifacts()
    if scored_ground_truth is None:
        raise GateError("ordinary full-clip audit did not construct its score target")
    if not isinstance(scoring_artifacts, Mapping):
        raise GateError("ordinary full-clip audit did not expose its initial noise")
    scored_future = scored_prediction[:, :, -model.num_future_frames :]
    output_difference = (prediction.float() - scored_future.float()).abs()
    output_equal = bool(torch.equal(prediction, scored_future))
    if not output_equal:
        raise GateError(
            "public history-only and ordinary full-clip autonomous predictions "
            f"differ: max={float(output_difference.max().item())}"
        )
    noise_keys = (
        "video_initial_state",
        "tf_initial_state",
        "tf_initial_noise",
        "reference_latents",
    )
    noise_hashes: dict[str, str] = {}
    for key in noise_keys:
        if key not in deployable_artifacts or key not in scoring_artifacts:
            raise GateError(f"NFE=1 comparison lacks artifact {key!r}")
        if not torch.equal(deployable_artifacts[key], scoring_artifacts[key]):
            raise GateError(
                f"history-only/full-clip NFE=1 artifact differs: {key}"
            )
        noise_hashes[key] = _tensor_sha256(deployable_artifacts[key])
    scoring_calls = scoring_counters.get("wan_calls_by_source_nfe", {})
    if (
        scoring_counters.get("online_teacher_calls") != 0
        or scoring_counters.get("auxiliary_clean_available") != 0
        or scoring_counters.get("deployment_mode") != 0
        or scoring_counters.get("artifacts_collected") != 1
        or scoring_counters.get("wan_calls_total") != 1
        or sum(int(value) for value in scoring_calls.values()) != 1
    ):
        raise GateError(
            f"ordinary full-clip NFE=1 audit counters differ: {scoring_counters}"
        )
    return {
        "history_shape": list(history.shape),
        "actions_shape": list(actions.shape),
        "prediction_shape": list(prediction.shape),
        "history_sha256": _tensor_sha256(history),
        "actions_sha256": _tensor_sha256(actions),
        "prediction_sha256": _tensor_sha256(prediction.float()),
        "prediction_min": float(prediction.float().min().item()),
        "prediction_max": float(prediction.float().max().item()),
        "sampler_counters": deployable_counters,
        "auxiliary_target_argument": None,
        "teacher_constructed": False,
        "ordinary_full_clip_audit": {
            "purpose": "untimed scoring-path equivalence audit",
            "condition_source": "autonomous",
            "nfe": 1,
            "auxiliary_target_argument": None,
            "generated_future_bitwise_equal": output_equal,
            "generated_future_max_absolute_difference": float(
                output_difference.max().item()
            ),
            "same_initial_noise_and_reference": True,
            "artifact_sha256": noise_hashes,
            "sampler_counters": scoring_counters,
            "ground_truth_used_as_condition": False,
        },
    }


def model_sync_digests(model: Any) -> dict[str, str]:
    parameters = dict(model.named_parameters())
    names = [
        EXPECTED_GRADIENT_PARAMETERS["auxiliary_state_gate"],
        EXPECTED_GRADIENT_PARAMETERS["auxiliary_clock_gate"],
        EXPECTED_GRADIENT_PARAMETERS["auxiliary_velocity_head"],
        _first_lora_parameter(model)[0],
    ]
    return {name: _tensor_sha256(parameters[name]) for name in names}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-build-complete", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache-metadata", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--warmstart-sha256", required=True)
    return parser.parse_args(argv)


def _rank0_broadcast(callable_: Any, rank: int) -> Any:
    import torch.distributed as distributed

    result = [None]
    if rank == 0:
        try:
            result[0] = {"ok": True, "value": callable_()}
        except Exception as exc:  # broadcast the exact fail-closed reason
            result[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    distributed.broadcast_object_list(result, src=0)
    payload = result[0]
    if not payload["ok"]:
        raise GateError(payload["error"])
    return payload["value"]


def _compose_config(
    *,
    project_root: Path,
    output_dir: Path,
    train_manifest: Path,
    train_metadata: Path,
) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = project_root / "configs"
    overrides = [
        "+experiments_0908=ravenhuang/wan-dit/vjepa_j0.yaml",
        "name=vjepa2-j0-phase-gate",
        "seed=1234",
        f"dataset.datasets.ABC.clip_manifest={train_manifest}",
        f"dataset.datasets.ABC.cache_metadata={train_metadata}",
        f"val_dataset.datasets.ABC.clip_manifest={train_manifest}",
        f"val_dataset.datasets.ABC.cache_metadata={train_metadata}",
        f"viz_dataset.datasets.ABC.clip_manifest={train_manifest}",
        f"viz_dataset.datasets.ABC.cache_metadata={train_metadata}",
        "dataset.infinite=true",
        "dataset.img_augment=false",
        "data_loader.batch_size=1",
        "data_loader.num_workers=0",
        "data_loader.prefetch_factor=null",
        "data_loader.persistent_workers=false",
        "val_data_loader=[]",
        "viz_data_loader=[]",
        "trainer.config.max_iter=4",
        "trainer.config.gradient_accumulation_steps=1",
        "trainer.config.throughput_warmup_steps=0",
        "trainer.config.load_path=null",
        "trainer.config.logging.log_every=1",
        "trainer.config.saving.save_every=2147483647",
        f"trainer.config.saving.save_path={output_dir / 'CHECKPOINT_FORBIDDEN.pt'}",
        "trainer.config.validation.val_every=2147483647",
        "trainer.config.validation.save_best=false",
        "trainer.config.visualization.viz_every=2147483647",
        f"trainer.config.visualization.viz_path={output_dir / 'visualization_forbidden'}",
        "wandb.enabled=false",
        "wandb.mode=disabled",
        "model.viz_num_steps=1",
        "model.dual_diffusion.enabled=true",
        "model.dual_diffusion.capture_latent_trajectories=false",
        "model.dual_diffusion.artifact_batch_limit=1",
        "model.dual_diffusion.evaluation_nfe_steps=[1]",
        "model.dual_diffusion.evaluation_condition_sources=[autonomous]",
        f"hydra.run.dir={output_dir}",
        f"hydra.sweep.dir={output_dir}",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        return compose(config_name="train", overrides=overrides)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = canonical_directory(args.repo_root, "repository root")
    project = canonical_directory(args.project_root, "project root")
    if project != repo / "projects/latent_action_models":
        raise GateError("project root must be the repository LACWM project")
    complete = canonical_file(
        args.cache_build_complete, "cache build completion record"
    )
    manifest = canonical_file(args.train_manifest, "training clip manifest")
    metadata = canonical_file(
        args.train_cache_metadata, "training cache metadata"
    )
    pca = canonical_file(args.pca, "PCA artifact")
    warmstart = canonical_file(args.warmstart, "LACWM warm start")
    if SHA256_RE.fullmatch(args.warmstart_sha256) is None:
        raise GateError("warm-start SHA-256 must be 64 lowercase hex characters")
    output = args.output_dir.expanduser()
    if not output.is_absolute():
        raise GateError("output directory must be absolute")
    output_parent = canonical_directory(output.parent, "output parent")
    output = output_parent / output.name

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(project))
    # Registers the project's OmegaConf resolvers before Hydra composition.
    __import__("custom_resolvers")

    import numpy as np
    import torch
    import torch.distributed as distributed
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != EXPECTED_WORLD_SIZE or not 0 <= rank < world_size:
        raise GateError(
            f"phase gate requires torchrun world size 8, got rank={rank}, "
            f"world_size={world_size}"
        )
    if local_rank != rank:
        raise GateError("single-node phase gate requires LOCAL_RANK == RANK")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise GateError("phase gate requires exactly eight visible CUDA devices")
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")
    try:
        if distributed.get_world_size() != 8:
            raise GateError("initialized DDP world size is not eight")
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_names: list[Any] = [None] * world_size
        distributed.all_gather_object(gpu_names, gpu_name)
        if any("B200" not in str(name).upper() for name in gpu_names):
            raise GateError(f"phase gate requires eight B200 GPUs: {gpu_names}")

        provenance = _rank0_broadcast(
            lambda: {
                "repository": validate_repository(repo, args.expected_commit),
                "cache": validate_cache_build_links(
                    complete_path=complete,
                    train_manifest=manifest,
                    train_metadata=metadata,
                    pca_path=pca,
                ),
                "warmstart": file_record(warmstart),
                "runtime_python": runtime_python_record(),
            },
            rank,
        )
        if provenance["warmstart"]["sha256"] != args.warmstart_sha256:
            raise GateError(
                "warm-start digest differs: "
                f"{provenance['warmstart']['sha256']} != "
                f"{args.warmstart_sha256}"
            )

        def create_output() -> str:
            output.mkdir(mode=0o700, exist_ok=False)
            return str(output)

        _rank0_broadcast(create_output, rank)

        # The extractor's validator performs full finite scans and SHA-256
        # validation of every cached target/RGB/action array.
        def validate_full_cache() -> dict[str, Any]:
            from tools.extract_vjepa2_targets import validate_cache

            payload = validate_cache(
                cache_metadata=metadata,
                clip_manifest=manifest,
                train_manifest=manifest,
                pca_path=pca,
                provenance=None,
                finite_check_rows=32,
            )
            return {
                "validated": True,
                "split": payload["split"],
                "clip_count": payload["clip_count"],
                "cache_id": payload["cache_id"],
                "target_shape": payload["target_shape"],
                "rgb_shape": payload["rgb_shape"],
                "actions_shape": payload["actions_shape"],
            }

        full_cache = _rank0_broadcast(validate_full_cache, rank)

        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        cfg = _compose_config(
            project_root=project,
            output_dir=output,
            train_manifest=manifest,
            train_metadata=metadata,
        )
        config_contract = validate_j0_config(cfg)
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
        config_digest = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
        if rank == 0:
            exclusive_write(
                output / "resolved_config.yaml",
                resolved_yaml.encode("utf-8"),
            )
        distributed.barrier()

        trainer = instantiate(cfg.trainer)
        if trainer.model.__class__.__name__ != "DistributedDataParallel":
            raise GateError("production Trainer did not construct DDP")
        if trainer.use_wandb:
            raise GateError("W&B unexpectedly initialized")
        if trainer.save_path.exists():
            raise GateError("forbidden phase-gate checkpoint path already exists")
        warmstart_load = strict_warmstart_load(
            trainer.model.module,
            warmstart,
            EXPECTED_RESET_PREFIXES,
            expected_missing_keys=EXPECTED_WARMSTART_MISSING_KEYS,
        )
        trainer.start_data_loaders()
        audit_batch = next(trainer._data_loader_iter)
        batch_shapes = validate_batch_shapes(audit_batch)
        local_clip_index = int(audit_batch["clip_index"].reshape(-1)[0].item())
        local_morphology_index = int(
            audit_batch["morphology_index"].reshape(-1)[0].item()
        )
        gathered_clip_indices: list[Any] = [None] * world_size
        distributed.all_gather_object(gathered_clip_indices, local_clip_index)
        unique_clip_indices = validate_unique_clip_indices(
            gathered_clip_indices,
            world_size,
        )
        gathered_morphology_indices: list[Any] = [None] * world_size
        distributed.all_gather_object(
            gathered_morphology_indices,
            local_morphology_index,
        )
        if gathered_morphology_indices != [9] * world_size:
            raise GateError(
                "ABC morphology index differs across ranks: "
                f"{gathered_morphology_indices}"
            )
        audit_batch = trainer._to_device(audit_batch)

        equivalence = zero_gate_equivalence(
            trainer.model.module,
            audit_batch,
        )
        update_report = four_optimizer_updates(trainer)
        inference_report = future_free_nfe1(
            trainer.model.module,
            audit_batch,
        )
        sync_digests = model_sync_digests(trainer.model.module)
        gathered_sync: list[Any] = [None] * world_size
        distributed.all_gather_object(gathered_sync, sync_digests)
        if any(value != gathered_sync[0] for value in gathered_sync[1:]):
            raise GateError("DDP model parameters differ after four updates")
        if trainer.save_path.exists():
            raise GateError("phase gate wrote a forbidden checkpoint")
        if trainer.use_wandb:
            raise GateError("phase gate enabled W&B")

        local = {
            "rank": rank,
            "local_rank": local_rank,
            "gpu": gpu_name,
            "batch_shapes": batch_shapes,
            "shape_audit_clip_index": local_clip_index,
            "morphology_index": local_morphology_index,
            "zero_gate_equivalence": equivalence,
            "warmstart_load": warmstart_load,
            "optimizer": update_report,
            "future_free_nfe1": inference_report,
            "model_sync_sha256": sync_digests,
        }
        local = identity_payload(local)
        exclusive_json(output / f"rank_{rank:02d}.json", local)
        distributed.barrier()
        ranks: list[Any] = [None] * world_size
        distributed.all_gather_object(ranks, local)

        final_invariance = _rank0_broadcast(
            lambda: {
                "repository_clean": validate_repository(
                    repo, args.expected_commit
                )["clean"],
                "warmstart_unchanged": unchanged_file(
                    provenance["warmstart"]
                ),
                "cache_records_unchanged": all(
                    unchanged_file(record)
                    for key, record in provenance["cache"].items()
                    if isinstance(record, Mapping) and "path" in record
                ),
            },
            rank,
        )
        if not all(final_invariance.values()):
            raise GateError(
                f"an immutable input changed during the gate: {final_invariance}"
            )

        report = identity_payload(
            {
                "artifact_type": "vjepa2-j0-phase-gate",
                "format_version": FORMAT_VERSION,
                "created_at": _now(),
                "passed": True,
                "sigma_convention": "sigma=1 noise, sigma=0 clean",
                "teacher_role": "offline target extractor only",
                "teacher_calls_training": 0,
                "teacher_calls_inference": 0,
                "world_size": world_size,
                "topology": {
                    "nodes": 1,
                    "gpus": gpu_names,
                    "backend": str(distributed.get_backend()),
                },
                "j0_contract": config_contract,
                "resolved_config_sha256": config_digest,
                "provenance": provenance,
                "full_cache_validation": full_cache,
                "warmstart_policy": {
                    "mode": "strict allowlisted reset",
                    "reset_prefixes": list(EXPECTED_RESET_PREFIXES),
                    "expected_missing_keys": list(
                        EXPECTED_WARMSTART_MISSING_KEYS
                    ),
                },
                "training": {
                    "ddp": True,
                    "optimizer_updates": EXPECTED_UPDATES,
                    "effective_global_batch_size": 8,
                    "shape_audit_clip_indices": unique_clip_indices,
                    "shape_audit_clip_indices_unique": True,
                    "morphology_indices": gathered_morphology_indices,
                    "morphology_contract": "ABC integer index exactly 9",
                    "wandb_enabled": False,
                    "checkpoint_writes": 0,
                },
                "inference": {
                    "source": "autonomous",
                    "nfe": 1,
                    "observable_history_frames": 5,
                    "future_rgb_supplied": False,
                    "clean_auxiliary_supplied": False,
                },
                "input_invariance": final_invariance,
                "rank_reports": ranks,
            }
        )
        if rank == 0:
            exclusive_json(output / "phase_gate_report.json", report)
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        distributed.barrier()
        return report
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "identity_sha256": report["identity_sha256"],
                    "output": str(args.output_dir / "phase_gate_report.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
