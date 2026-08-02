#!/usr/bin/env python3
"""Build immutable prefix-causal V-JEPA2.1 targets for DROID clips.

The heavy teacher is used only by ``fit-pca`` and ``extract``.  Both commands
support ``torchrun``: ranks own disjoint clips, publish hash-bound rank
sidecars, and rank zero performs the atomic final publication.  An interrupted
run can be resumed only when every immutable identity matches exactly.
Existing complete outputs and ambiguous partial directories fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.causal_vjepa2 import (  # noqa: E402
    CACHE_ARTIFACT_TYPE,
    FROZEN_BASE_CLIP_COUNTS,
    FROZEN_BASE_EPISODE_COUNTS,
    FROZEN_BASE_MANIFEST_SHA256,
    FROZEN_BASE_PROVENANCE_SHA256,
    FROZEN_DROID_DATA_ROOT,
    FROZEN_ELIGIBLE_INVENTORY_SHA256,
    FROZEN_SPLIT_EPISODE_IDS_SHA256,
    FORMAT_VERSION,
    LAST_TEMPORAL_TOKEN,
    PCA_ARTIFACT_TYPE,
    PCA_CLIP_COUNT,
    PCA_RANK_PREFIX,
    POOLED_TOKEN_GRID,
    PRODUCTION_NUMERICAL_CONTRACT,
    SCHEMA,
    TARGET_CHANNELS,
    TARGET_KIND,
    TARGET_SHAPE,
    TEACHER_FRAMES,
    TEACHER_SIZE,
    TOKENS_PER_CLIP,
    VJEPA2_CHECKPOINT_SHA256,
    VJEPA2_LICENSE_SHA256,
    VJEPA2_SOURCE_COMMIT,
    WHITENING_EPS,
    CausalVJEPA2Error,
    extract_causal_vjepa2_target,
    extract_causal_vjepa2_tokens,
    manifest_order_sha256,
    identity_sha256,
    prepare_prefix_causal_teacher_input,
    select_pca_rows,
    validate_causal_cache,
    validate_frozen_base_record,
    validate_pca_artifact,
)
from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    CAMERA,
    SCHEMA as BASE_DROID_SCHEMA,
    DroidVideoLatentForcingDataset,
    canonical_json,
    read_clip_manifest,
    rows_episode_ids,
    sha256_file,
)
from robot_wm.modeling.dual_diffusion.vjepa2_target import (  # noqa: E402
    PCAWhiteningStats,
    VJEPA2_1_CHECKPOINT_BYTES,
    VJEPA2_1_MODEL_NAME,
    VJEPA2_1_SOURCE_DIM,
    load_vjepa2_1_vit_base_encoder,
    validate_vjepa_source,
)


TARGET_FILE = "targets.fp16.npy"
METADATA_FILE = "metadata.json"
PROGRESS_CHECKPOINT_ROWS = 64
APPROVED_ARTIFACT_ROOTS = (Path("/lustre"), Path("/mnt/data1"), Path("/mnt/data2"))


class BuildError(CausalVJEPA2Error):
    """An offline build could not preserve the immutable artifact contract."""


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise BuildError(f"required implementation file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def validate_builder_source() -> dict[str, Any]:
    """Bind targets to one clean project commit and exact extractor sources."""
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise BuildError("causal target extraction requires a clean committed project tree")
    return {
        "repo_root": str(REPO_ROOT.resolve()),
        "repo_commit": commit,
        "builder_source": _file_record(Path(__file__)),
        "dataset_source": _file_record(
            REPO_ROOT / "robot_wm" / "datasets" / "droid" / "causal_vjepa2.py"
        ),
    }


def runtime_record() -> dict[str, str]:
    """Versions that can change encoder/PCA floating-point results."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "numpy": str(np.__version__),
    }


def execution_evidence(*, rank: int, world_size: int) -> dict[str, Any]:
    """Operational evidence deliberately excluded from resumable identities."""
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_STEP_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
    )
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv),
        "rank": int(rank),
        "world_size": int(world_size),
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def production_numerical_contract(
    *,
    context: "DistributedContext",
    encoder_dtype: torch.dtype,
    batch_size: int,
    pca_device: str,
) -> dict[str, Any]:
    actual = {
        "encoder_device": context.device.type,
        "encoder_dtype": str(encoder_dtype).removeprefix("torch."),
        "encoder_batch_size": int(batch_size),
        "pca_device": str(pca_device),
        "pca_algorithm": "exact-centered-covariance-eigh",
        "pca_covariance_dtype": "float32",
        "pca_tf32": False,
    }
    if actual != PRODUCTION_NUMERICAL_CONTRACT:
        raise BuildError(
            "production causal targets require CUDA, bfloat16 encoder, batch-size 1, "
            "and CUDA exact-eigh PCA"
        )
    return actual


def _episode_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = list(dict.fromkeys(int(row["episode_index"]) for row in rows))
    return hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_frozen_base_droid(
    *, train_manifest: str | Path, data_root: str | Path
) -> dict[str, Any]:
    """Validate all three exact POC manifests without decoding protected RGB."""
    train = Path(train_manifest).expanduser().resolve()
    root = train.parent
    paths = {
        "train": train,
        "val": root / "val.jsonl",
        "test": root / "test.identifiers.jsonl",
    }
    provenance_path = root / "provenance.json"
    expected_names = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.identifiers.jsonl",
    }
    if train.name != expected_names["train"] or not provenance_path.is_file():
        raise BuildError("train manifest is not inside the frozen base-DROID artifact")
    if sha256_file(provenance_path) != FROZEN_BASE_PROVENANCE_SHA256:
        raise BuildError("base-DROID provenance SHA-256 mismatch")
    provenance = _read_json(provenance_path, label="base-DROID provenance")
    resolved_data_root = str(Path(data_root).expanduser().resolve())
    if (
        provenance.get("schema") != BASE_DROID_SCHEMA
        or provenance.get("complete") is not True
        or provenance.get("camera") != CAMERA
        or provenance.get("camera_count") != 1
        or provenance.get("clip_start_seed") != 20260801
        or provenance.get("clips_per_episode")
        != {"train": 8, "val": 1, "test": 1}
        or provenance.get("eligible_episode_count") != 9_780
        or provenance.get("eligible_inventory_sha256")
        != FROZEN_ELIGIBLE_INVENTORY_SHA256
        or provenance.get("split_counts") != FROZEN_BASE_EPISODE_COUNTS
        or provenance.get("split_episode_ids_sha256")
        != FROZEN_SPLIT_EPISODE_IDS_SHA256
        or provenance.get("source_root") != FROZEN_DROID_DATA_ROOT
        or resolved_data_root != FROZEN_DROID_DATA_ROOT
        or provenance.get("protected_test_policy")
        != "identifier-only; no video decode or cache during construction"
    ):
        raise BuildError("base-DROID provenance violates the frozen POC contract")
    manifest_metadata = provenance.get("manifests")
    if not isinstance(manifest_metadata, Mapping):
        raise BuildError("base-DROID provenance lacks manifest records")

    manifests: dict[str, dict[str, Any]] = {}
    episode_sets: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        path = paths[split]
        expected_sha = FROZEN_BASE_MANIFEST_SHA256[split]
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise BuildError(f"base-DROID {split} manifest SHA-256 mismatch")
        rows = read_clip_manifest(path, expected_split=split)
        episode_ids = rows_episode_ids(rows)
        entry = manifest_metadata.get(split)
        if (
            not isinstance(entry, Mapping)
            or entry.get("path") != expected_names[split]
            or entry.get("sha256") != expected_sha
            or entry.get("clip_count") != FROZEN_BASE_CLIP_COUNTS[split]
            or entry.get("episode_count") != FROZEN_BASE_EPISODE_COUNTS[split]
            or entry.get("protected") != (split == "test")
            or entry.get("cached") != (split != "test")
            or len(rows) != FROZEN_BASE_CLIP_COUNTS[split]
            or len(episode_ids) != FROZEN_BASE_EPISODE_COUNTS[split]
            or _episode_ids_sha256(rows)
            != FROZEN_SPLIT_EPISODE_IDS_SHA256[split]
        ):
            raise BuildError(f"base-DROID {split} population identity mismatch")
        if split == "test" and any(
            "cache_relpath" in row or "cache_sha256" in row for row in rows
        ):
            raise BuildError("protected test manifest contains cached payload references")
        episode_sets[split] = episode_ids
        manifests[split] = {
            **_file_record(path),
            "clip_count": len(rows),
            "episode_count": len(episode_ids),
            "episode_ids_sha256": _episode_ids_sha256(rows),
            "protected": split == "test",
            "cached": split != "test",
        }
    if any(
        episode_sets[left].intersection(episode_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise BuildError("frozen base-DROID splits overlap")
    record = {
        "schema": "frozen-droid-video-latent-forcing-poc-v1",
        "artifact_root": str(root),
        "data_root": resolved_data_root,
        "provenance": _file_record(provenance_path),
        "manifests": manifests,
        "eligible_inventory_sha256": FROZEN_ELIGIBLE_INVENTORY_SHA256,
        "split_episode_ids_sha256": dict(FROZEN_SPLIT_EPISODE_IDS_SHA256),
        "clip_counts": dict(FROZEN_BASE_CLIP_COUNTS),
        "episode_counts": dict(FROZEN_BASE_EPISODE_COUNTS),
        "split_disjoint": True,
        "protected_test_payload_cached": False,
    }
    return validate_frozen_base_record(record)


def approved_artifact_path(path: str | Path) -> Path:
    """Keep multi-gigabyte PCA/cache artifacts off the repo and root disk."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(REPO_ROOT.resolve()):
        raise BuildError(f"large artifacts cannot be written inside the Git repo: {resolved}")
    if not any(resolved.is_relative_to(root) for root in APPROVED_ARTIFACT_ROOTS):
        raise BuildError("large artifacts must be under /lustre, /mnt/data1, or /mnt/data2")
    return resolved


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BuildError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BuildError(f"{label} must be a JSON object")
    return payload


def _hash_identity(payload: Mapping[str, Any]) -> str:
    return identity_sha256(payload)


def _row_sha256(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def _encoder_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_process_group: bool

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _distributed_context(device_name: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    owns = False
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise BuildError("CUDA was requested but is unavailable")
        # NCCL may select a device while forming its first communicator.  Bind
        # the local rank before process-group initialization, never afterwards.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise BuildError("device must be exactly 'cpu' or 'cuda'")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        owns = True
    if world_size > 1:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return DistributedContext(rank, world_size, local_rank, device, owns)


def validate_teacher_inputs(
    *, source_path: str | Path, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Verify the exact clean source commit and exact official checkpoint."""
    source = validate_vjepa_source(
        source_path, expected_commit=VJEPA2_SOURCE_COMMIT
    )
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise BuildError("V-JEPA source has tracked modifications")
    archive = subprocess.Popen(
        ["git", "-C", str(source), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    source_digest = hashlib.sha256()
    while chunk := archive.stdout.read(8 * 1024 * 1024):
        source_digest.update(chunk)
    stderr = archive.stderr.read().decode("utf-8", errors="replace") if archive.stderr else ""
    if archive.wait() != 0:
        raise BuildError(f"cannot hash the pinned V-JEPA source archive: {stderr}")
    license_candidates = [
        candidate
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
        if (candidate := source / name).is_file()
    ]
    if len(license_candidates) != 1:
        raise BuildError(
            "the pinned V-JEPA checkout must expose exactly one recognized root LICENSE file"
        )
    license_path = license_candidates[0]
    license_digest = sha256_file(license_path)
    if license_digest != VJEPA2_LICENSE_SHA256:
        raise BuildError(
            f"V-JEPA LICENSE hash mismatch: expected {VJEPA2_LICENSE_SHA256}, "
            f"found {license_digest}"
        )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise BuildError(f"V-JEPA checkpoint is missing: {checkpoint}")
    if checkpoint.stat().st_size != VJEPA2_1_CHECKPOINT_BYTES:
        raise BuildError(
            "V-JEPA checkpoint byte count mismatch: "
            f"expected {VJEPA2_1_CHECKPOINT_BYTES}, found {checkpoint.stat().st_size}"
        )
    digest = sha256_file(checkpoint)
    if digest != VJEPA2_CHECKPOINT_SHA256:
        raise BuildError(
            f"V-JEPA checkpoint hash mismatch: expected {VJEPA2_CHECKPOINT_SHA256}, "
            f"found {digest}"
        )
    return {
        "model_name": VJEPA2_1_MODEL_NAME,
        "source_path": str(source),
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "source_archive_sha256": source_digest.hexdigest(),
        "source_license": {
            "path": str(license_path),
            "sha256": license_digest,
            "bytes": license_path.stat().st_size,
        },
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": VJEPA2_1_CHECKPOINT_BYTES,
        "checkpoint_sha256": digest,
        "checkpoint_evidence": {
            "path": str(checkpoint),
            "sha256": digest,
            "bytes": checkpoint.stat().st_size,
        },
    }


def _load_teacher(
    provenance: Mapping[str, Any], *, device: torch.device
) -> torch.nn.Module:
    encoder = load_vjepa2_1_vit_base_encoder(
        source_path=provenance["source_path"],
        checkpoint_path=provenance["checkpoint_path"],
        expected_source_commit=VJEPA2_SOURCE_COMMIT,
        expected_checkpoint_sha256=VJEPA2_CHECKPOINT_SHA256,
    )
    encoder.to(device=device)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _stack_samples(
    dataset: DroidVideoLatentForcingDataset,
    indices: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [dataset[int(index)] for index in indices]
    history = torch.stack([sample["history"] for sample in samples]).to(
        device=device, non_blocking=True
    )
    future = torch.stack([sample["future"] for sample in samples]).to(
        device=device, non_blocking=True
    )
    return history, future


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def run_teacher_preflight(
    *,
    train_manifest: Path,
    data_root: Path,
    source_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    context: DistributedContext,
) -> dict[str, Any] | None:
    """Exercise the real non-square teacher and every causal future boundary."""
    if context.world_size != 1 or context.rank != 0:
        raise BuildError("teacher preflight requires exactly one CUDA rank")
    numerical_contract = production_numerical_contract(
        context=context,
        encoder_dtype=torch.bfloat16,
        batch_size=1,
        pca_device="cuda",
    )
    runtime = runtime_record()
    train_manifest = train_manifest.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise BuildError("teacher preflight evidence is immutable; choose a new output")
    base_droid = validate_frozen_base_droid(
        train_manifest=train_manifest, data_root=data_root
    )
    rows = read_clip_manifest(train_manifest, expected_split="train")
    selected_index, selected_row = select_pca_rows(rows)[0]
    provenance = validate_teacher_inputs(
        source_path=source_path, checkpoint_path=checkpoint_path
    )
    implementation = validate_builder_source()
    dataset = DroidVideoLatentForcingDataset(train_manifest, data_root)
    history, future = _stack_samples(
        dataset, [selected_index], device=context.device
    )
    prepared = prepare_prefix_causal_teacher_input(history, future)
    if tuple(prepared.shape) != (8, 3, 16, 384, 672):
        raise BuildError(f"real teacher preflight prepared wrong shape: {prepared.shape}")
    encoder = _load_teacher(provenance, device=context.device)
    baseline = extract_causal_vjepa2_tokens(
        history, future, encoder=encoder, encoder_dtype=torch.bfloat16
    )
    repeated = extract_causal_vjepa2_tokens(
        history, future, encoder=encoder, encoder_dtype=torch.bfloat16
    )
    expected_shape = (1, 8, 8, 14, VJEPA2_1_SOURCE_DIM)
    if tuple(baseline.shape) != expected_shape or not torch.isfinite(baseline).all():
        raise BuildError(f"real teacher returned invalid pooled tokens: {baseline.shape}")
    if not torch.equal(baseline, repeated):
        raise BuildError("real teacher is not bitwise repeatable for an identical input")
    checks: list[dict[str, Any]] = []
    for boundary in range(1, 8):
        changed = future.clone()
        suffix = changed[:, :, boundary:]
        changed[:, :, boundary:] = torch.where(
            suffix >= 0,
            (suffix - 0.25).clamp(-1.0, 1.0),
            (suffix + 0.25).clamp(-1.0, 1.0),
        )
        perturbed = extract_causal_vjepa2_tokens(
            history, changed, encoder=encoder, encoder_dtype=torch.bfloat16
        )
        earlier_equal = torch.equal(baseline[:, :boundary], perturbed[:, :boundary])
        affected_changed = not torch.equal(
            baseline[:, boundary:], perturbed[:, boundary:]
        )
        if not earlier_equal or not affected_changed:
            raise BuildError(
                f"real teacher causal perturbation failed at future boundary {boundary}"
            )
        checks.append(
            {
                "changed_future_from": boundary,
                "earlier_targets_bitwise_equal": earlier_equal,
                "affected_targets_changed": affected_changed,
                "perturbed_tokens_sha256": _tensor_sha256(perturbed),
            }
        )
    result = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "droid-causal-vjepa2.1-real-teacher-preflight",
        "complete": True,
        "production_cache": False,
        "clip_id": str(selected_row["clip_id"]),
        "manifest_index": int(selected_index),
        "prepared_shape": list(prepared.shape),
        "pooled_token_shape": list(baseline.shape),
        "baseline_tokens_sha256": _tensor_sha256(baseline),
        "repeat_tokens_sha256": _tensor_sha256(repeated),
        "bitwise_repeatable": True,
        "causality_checks": checks,
        "teacher_calls": 2 + len(checks),
        "protected_test_access": False,
        "base_droid": base_droid,
        "teacher": provenance,
        "implementation": implementation,
        "runtime": runtime,
        "numerical_contract": numerical_contract,
        "execution_evidence": execution_evidence(rank=0, world_size=1),
    }
    result["evidence_id"] = _hash_identity(
        {key: value for key, value in result.items() if key != "execution_evidence"}
    )
    _atomic_write_json(output_path, result)
    return result


def _write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        target = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=array.dtype, shape=array.shape
        )
        target[...] = array
        target.flush()
        del target
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _rank_pca_shard_record(
    *,
    rank: int,
    world_size: int,
    selected: Sequence[tuple[int, Mapping[str, Any]]],
    artifact_id: str | None = None,
    runtime: Mapping[str, Any] | None = None,
    numerical_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    positions = list(range(rank, len(selected), world_size))
    record = {
        "rank": rank,
        "world_size": world_size,
        "selected_positions": positions,
        "manifest_indices": [int(selected[position][0]) for position in positions],
        "clip_ids": [str(selected[position][1]["clip_id"]) for position in positions],
        "token_shape": [len(positions) * TOKENS_PER_CLIP, VJEPA2_1_SOURCE_DIM],
    }
    if artifact_id is not None:
        record.update(
            {
                "artifact_id": artifact_id,
                "runtime": dict(runtime or {}),
                "numerical_contract": dict(numerical_contract or {}),
            }
        )
    return record


def _validate_pca_rank_shard(
    work_dir: Path, expected: Mapping[str, Any]
) -> Path | None:
    sidecar = work_dir / f"rank-{int(expected['rank']):05d}.json"
    if not sidecar.exists():
        return None
    actual = _read_json(sidecar, label="PCA rank sidecar")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise BuildError(f"PCA resume sidecar differs at {key!r}")
    token_file = work_dir / str(actual.get("token_file", ""))
    if not token_file.is_file():
        raise BuildError("PCA rank sidecar points to a missing token shard")
    if sha256_file(token_file) != actual.get("token_sha256"):
        raise BuildError("PCA rank token shard hash mismatch")
    array = np.load(token_file, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(expected["token_shape"]) or array.dtype != np.float32:
        raise BuildError("PCA rank token shard shape/dtype mismatch")
    if not np.isfinite(array).all():
        raise BuildError("PCA rank token shard is non-finite")
    return token_file


def _extract_pca_rank_shard(
    *,
    work_dir: Path,
    selected: Sequence[tuple[int, Mapping[str, Any]]],
    dataset: DroidVideoLatentForcingDataset,
    encoder: torch.nn.Module,
    context: DistributedContext,
    encoder_dtype: torch.dtype,
    batch_size: int,
    resume: bool,
    artifact_id: str,
    runtime: Mapping[str, Any],
    numerical_contract: Mapping[str, Any],
) -> None:
    expected = _rank_pca_shard_record(
        rank=context.rank,
        world_size=context.world_size,
        selected=selected,
        artifact_id=artifact_id,
        runtime=runtime,
        numerical_contract=numerical_contract,
    )
    if resume and _validate_pca_rank_shard(work_dir, expected) is not None:
        return
    sidecar = work_dir / f"rank-{context.rank:05d}.json"
    if sidecar.exists():
        raise BuildError("PCA rank sidecar collision; pass --resume only for an exact run")
    positions = list(expected["selected_positions"])
    output = np.empty(tuple(expected["token_shape"]), dtype=np.float32)
    cursor = 0
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        manifest_indices = [int(selected[position][0]) for position in batch_positions]
        history, future = _stack_samples(dataset, manifest_indices, device=context.device)
        tokens = extract_causal_vjepa2_tokens(
            history,
            future,
            encoder=encoder,
            encoder_dtype=encoder_dtype,
        ).float().cpu().numpy()
        if tuple(tokens.shape[1:]) != (
            8,
            *POOLED_TOKEN_GRID,
            VJEPA2_1_SOURCE_DIM,
        ):
            raise BuildError(f"unexpected causal teacher token shape: {tokens.shape}")
        flat = tokens.reshape(len(batch_positions) * TOKENS_PER_CLIP, VJEPA2_1_SOURCE_DIM)
        output[cursor : cursor + len(flat)] = flat
        cursor += len(flat)
    if cursor != len(output) or not np.isfinite(output).all():
        raise BuildError("PCA rank extraction was incomplete or non-finite")
    token_path = work_dir / f"rank-{context.rank:05d}-tokens.f32.npy"
    _write_numpy_atomic(token_path, output)
    record = {
        **expected,
        "complete": True,
        "token_file": token_path.name,
        "token_sha256": sha256_file(token_path),
        "execution_evidence": execution_evidence(
            rank=context.rank, world_size=context.world_size
        ),
    }
    _atomic_write_json(sidecar, record)


def _fit_whitening(
    matrix: torch.Tensor,
    *,
    channels: int,
    eps: float = WHITENING_EPS,
) -> PCAWhiteningStats:
    """Fit exact centered-covariance PCA whitening using every input row."""
    if matrix.ndim != 2 or matrix.shape[0] <= channels or matrix.shape[1] < channels:
        raise BuildError("PCA matrix is too small for the requested projection")
    if not torch.isfinite(matrix).all():
        raise BuildError("PCA matrix is non-finite")
    matrix = matrix.float()
    mean = matrix.mean(dim=0)
    matrix = matrix - mean
    previous_tf32 = None
    if matrix.is_cuda:
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        covariance = matrix.transpose(0, 1) @ matrix
        covariance.div_(matrix.shape[0] - 1)
        eigenvalues_all, eigenvectors = torch.linalg.eigh(covariance)
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    order = torch.arange(
        eigenvalues_all.numel() - 1,
        eigenvalues_all.numel() - channels - 1,
        -1,
        device=eigenvalues_all.device,
    )
    eigenvalues = eigenvalues_all.index_select(0, order).clamp_min_(0.0)
    components = eigenvectors.index_select(1, order).transpose(0, 1).contiguous()
    pivots = components.abs().argmax(dim=1, keepdim=True)
    signs = components.gather(1, pivots).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    components.mul_(signs)
    return PCAWhiteningStats(
        mean=mean.detach().cpu(),
        components=components.detach().cpu(),
        eigenvalues=eigenvalues.detach().cpu(),
        eps=eps,
    )


def _pca_identity(
    *,
    train_manifest: Path,
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[tuple[int, Mapping[str, Any]]],
    provenance: Mapping[str, Any],
    implementation: Mapping[str, Any],
    base_droid: Mapping[str, Any],
    runtime: Mapping[str, Any],
    numerical_contract: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    selected_ids = [str(row["clip_id"]) for _, row in selected]
    selected_indices = [int(index) for index, _ in selected]
    return {
        "schema": SCHEMA,
        "artifact_type": PCA_ARTIFACT_TYPE,
        "train_manifest_sha256": sha256_file(train_manifest),
        "train_manifest_order_sha256": manifest_order_sha256(rows),
        "selected_clip_ids": selected_ids,
        "selected_manifest_indices": selected_indices,
        "selected_clip_ranks": [
            hashlib.sha256(f"{PCA_RANK_PREFIX}{clip_id}".encode()).hexdigest()
            for clip_id in selected_ids
        ],
        "pca_rank_prefix": PCA_RANK_PREFIX,
        "pca_clip_count": PCA_CLIP_COUNT,
        "tokens_per_clip": TOKENS_PER_CLIP,
        "sampled_token_count": PCA_CLIP_COUNT * TOKENS_PER_CLIP,
        "pca_training_split_only": True,
        "test_rows_used": 0,
        "source_commit": provenance["source_commit"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_evidence": provenance["checkpoint_evidence"],
        "source_archive_sha256": provenance["source_archive_sha256"],
        "source_license": provenance["source_license"],
        "implementation": dict(implementation),
        "base_droid": dict(base_droid),
        "runtime": dict(runtime),
        "numerical_contract": dict(numerical_contract),
        "teacher_size": list(TEACHER_SIZE),
        "teacher_frames": TEACHER_FRAMES,
        "last_temporal_token": LAST_TEMPORAL_TOKEN,
        "pooled_token_grid": list(POOLED_TOKEN_GRID),
        "source_dimension": VJEPA2_1_SOURCE_DIM,
        "target_channels": TARGET_CHANNELS,
        "pca_algorithm": "exact-centered-covariance-eigh",
        "pca_covariance_dtype": "float32",
        "pca_tf32": False,
        "whitening_eps": WHITENING_EPS,
        "world_size": world_size,
    }


def _pca_companion_result(
    *, output_path: Path, payload: Mapping[str, Any], digest: str
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "artifact_type": PCA_ARTIFACT_TYPE,
        "artifact_id": payload["artifact_id"],
        "artifact_file": output_path.name,
        "artifact_sha256": digest,
        "train_manifest_sha256": payload["train_manifest_sha256"],
        "selection_sha256": _hash_identity(
            {
                "selected_clip_ids": payload["selected_clip_ids"],
                "selected_manifest_indices": payload["selected_manifest_indices"],
                "selected_clip_ranks": payload["selected_clip_ranks"],
            }
        ),
        "selected_clip_count": PCA_CLIP_COUNT,
        "sampled_token_count": PCA_CLIP_COUNT * TOKENS_PER_CLIP,
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
        "input_dimension": VJEPA2_1_SOURCE_DIM,
        "output_dimension": TARGET_CHANNELS,
    }


def fit_pca(
    *,
    train_manifest: Path,
    data_root: Path,
    source_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    context: DistributedContext,
    encoder_dtype: torch.dtype,
    batch_size: int,
    pca_device: str,
    resume: bool,
) -> dict[str, Any] | None:
    train_manifest = train_manifest.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    numerical_contract = production_numerical_contract(
        context=context,
        encoder_dtype=encoder_dtype,
        batch_size=batch_size,
        pca_device=pca_device,
    )
    runtime = runtime_record()
    base_droid = validate_frozen_base_droid(
        train_manifest=train_manifest, data_root=data_root
    )
    rows = read_clip_manifest(train_manifest, expected_split="train")
    selected = select_pca_rows(rows)
    provenance = validate_teacher_inputs(
        source_path=source_path, checkpoint_path=checkpoint_path
    )
    implementation = validate_builder_source()
    identity = _pca_identity(
        train_manifest=train_manifest,
        rows=rows,
        selected=selected,
        provenance=provenance,
        implementation=implementation,
        base_droid=base_droid,
        runtime=runtime,
        numerical_contract=numerical_contract,
        world_size=context.world_size,
    )
    artifact_id = _hash_identity(identity)
    companion = output_path.with_suffix(output_path.suffix + ".metadata.json")
    if output_path.exists() or companion.exists():
        if not resume or not output_path.is_file():
            raise BuildError("PCA output collision; choose a new path or exact --resume")
        if not companion.exists():
            if context.rank == 0:
                _, recovery_payload, recovery_digest = validate_pca_artifact(
                    output_path,
                    train_manifest_path=train_manifest,
                    expected_base_droid=base_droid,
                    expected_runtime=runtime,
                    expected_numerical_contract=numerical_contract,
                )
                if recovery_payload.get("artifact_id") != artifact_id:
                    raise BuildError("orphan PCA output is not this exact invocation")
                _atomic_write_json(
                    companion,
                    _pca_companion_result(
                        output_path=output_path,
                        payload=recovery_payload,
                        digest=recovery_digest,
                    ),
                )
            context.barrier()
        if not companion.is_file():
            raise BuildError("PCA companion recovery did not publish a complete pair")
        _, payload, digest = validate_pca_artifact(
            output_path,
            train_manifest_path=train_manifest,
            expected_base_droid=base_droid,
            expected_runtime=runtime,
            expected_numerical_contract=numerical_contract,
        )
        metadata = _read_json(companion, label="PCA companion")
        if payload.get("artifact_id") != artifact_id or metadata.get("artifact_sha256") != digest:
            raise BuildError("completed PCA output does not match this exact invocation")
        context.barrier()
        return metadata if context.rank == 0 else None

    work_dir = output_path.parent / f".{output_path.name}.work"
    work_metadata = work_dir / "work.json"
    expected_work = {
        "format_version": FORMAT_VERSION,
        "artifact_id": artifact_id,
        "complete": False,
        "identity": identity,
    }
    if context.rank == 0:
        if work_dir.exists():
            if not resume or not work_metadata.is_file():
                raise BuildError("PCA work-directory collision")
            actual = _read_json(work_metadata, label="PCA work metadata")
            if actual != expected_work:
                raise BuildError("PCA resume identity mismatch")
        else:
            work_dir.mkdir(parents=True, exist_ok=False)
            _atomic_write_json(work_metadata, expected_work)
    context.barrier()
    actual_work = _read_json(work_metadata, label="PCA work metadata")
    if actual_work != expected_work:
        raise BuildError("PCA work metadata changed across ranks")

    dataset = DroidVideoLatentForcingDataset(train_manifest, data_root)
    encoder = _load_teacher(provenance, device=context.device)
    _extract_pca_rank_shard(
        work_dir=work_dir,
        selected=selected,
        dataset=dataset,
        encoder=encoder,
        context=context,
        encoder_dtype=encoder_dtype,
        batch_size=batch_size,
        resume=resume,
        artifact_id=artifact_id,
        runtime=runtime,
        numerical_contract=numerical_contract,
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    context.barrier()

    result: dict[str, Any] | None = None
    if context.rank == 0:
        matrix = np.empty(
            (PCA_CLIP_COUNT * TOKENS_PER_CLIP, VJEPA2_1_SOURCE_DIM),
            dtype=np.float32,
        )
        covered: set[int] = set()
        for rank in range(context.world_size):
            expected = _rank_pca_shard_record(
                rank=rank,
                world_size=context.world_size,
                selected=selected,
                artifact_id=artifact_id,
                runtime=runtime,
                numerical_contract=numerical_contract,
            )
            token_path = _validate_pca_rank_shard(work_dir, expected)
            if token_path is None:
                raise BuildError(f"PCA rank {rank} did not publish a complete shard")
            shard = np.load(token_path, mmap_mode="r", allow_pickle=False)
            for local, position in enumerate(expected["selected_positions"]):
                if int(position) in covered:
                    raise BuildError("PCA rank shards overlap")
                source_start = local * TOKENS_PER_CLIP
                target_start = int(position) * TOKENS_PER_CLIP
                matrix[target_start : target_start + TOKENS_PER_CLIP] = shard[
                    source_start : source_start + TOKENS_PER_CLIP
                ]
                covered.add(int(position))
        if covered != set(range(PCA_CLIP_COUNT)) or not np.isfinite(matrix).all():
            raise BuildError("PCA rank shards do not cover the exact selection")
        tensor = torch.from_numpy(matrix)
        if pca_device == "cuda":
            if not torch.cuda.is_available():
                raise BuildError("CUDA PCA was requested but CUDA is unavailable")
            tensor = tensor.to(context.device)
        elif pca_device != "cpu":
            raise BuildError("pca-device must be cpu or cuda")
        stats = _fit_whitening(tensor, channels=TARGET_CHANNELS)
        payload = stats.to_payload()
        payload.update(
            {
                "artifact_type": PCA_ARTIFACT_TYPE,
                "artifact_identity": identity,
                "artifact_id": artifact_id,
                "model_name": VJEPA2_1_MODEL_NAME,
                **identity,
            }
        )
        _atomic_torch_save(output_path, payload)
        digest = sha256_file(output_path)
        _, validated_payload, validated_digest = validate_pca_artifact(
            output_path,
            train_manifest_path=train_manifest,
            expected_base_droid=base_droid,
            expected_runtime=runtime,
            expected_numerical_contract=numerical_contract,
        )
        if (
            validated_payload.get("artifact_id") != artifact_id
            or validated_digest != digest
        ):
            raise BuildError("new PCA artifact failed its immutable self-validation")
        result = _pca_companion_result(
            output_path=output_path, payload=validated_payload, digest=digest
        )
        _atomic_write_json(companion, result)
        _atomic_write_json(
            work_metadata,
            {**expected_work, "complete": True, "artifact_sha256": digest},
        )
    context.barrier()
    return result


def _cache_identity(
    *,
    manifest: Path,
    rows: Sequence[Mapping[str, Any]],
    train_manifest: Path,
    pca_path: Path,
    pca_digest: str,
    provenance: Mapping[str, Any],
    implementation: Mapping[str, Any],
    base_droid: Mapping[str, Any],
    runtime: Mapping[str, Any],
    numerical_contract: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    split = str(rows[0]["split"])
    return {
        "schema": SCHEMA,
        "artifact_type": CACHE_ARTIFACT_TYPE,
        "target_kind": TARGET_KIND,
        "auxiliary_target_shape": list(TARGET_SHAPE),
        "split": split,
        "clip_count": len(rows),
        "manifest_sha256": sha256_file(manifest),
        "manifest_order_sha256": manifest_order_sha256(rows),
        "train_manifest_sha256": sha256_file(train_manifest),
        "pca_file": str(pca_path),
        "pca_sha256": pca_digest,
        "source_commit": provenance["source_commit"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_evidence": provenance["checkpoint_evidence"],
        "source_archive_sha256": provenance["source_archive_sha256"],
        "source_license": provenance["source_license"],
        "implementation": dict(implementation),
        "base_droid": dict(base_droid),
        "runtime": dict(runtime),
        "numerical_contract": dict(numerical_contract),
        "target_shape": [len(rows), *TARGET_SHAPE],
        "target_dtype": "float16",
        "teacher_size": list(TEACHER_SIZE),
        "teacher_frames": TEACHER_FRAMES,
        "last_temporal_token": LAST_TEMPORAL_TOKEN,
        "pooled_token_grid": list(POOLED_TOKEN_GRID),
        "world_size": world_size,
        "protected_test_access": False,
        "allowed_splits": ["train", "val"],
        "test_rows_extracted": 0,
    }


def _cache_rank_expected(
    rank: int,
    world_size: int,
    row_count: int,
    *,
    cache_id: str | None = None,
    runtime: Mapping[str, Any] | None = None,
    numerical_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "rank": rank,
        "world_size": world_size,
        "assigned_indices": list(range(rank, row_count, world_size)),
    }
    if cache_id is not None:
        record.update(
            {
                "cache_id": cache_id,
                "runtime": dict(runtime or {}),
                "numerical_contract": dict(numerical_contract or {}),
            }
        )
    return record


def _load_completed_rows(
    sidecar: Path,
    *,
    expected: Mapping[str, Any],
    target: np.memmap,
) -> dict[int, str]:
    if not sidecar.exists():
        return {}
    actual = _read_json(sidecar, label="cache rank sidecar")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise BuildError(f"cache resume sidecar differs at {key!r}")
    records = actual.get("completed_rows")
    if not isinstance(records, list):
        raise BuildError("cache rank sidecar lacks completed_rows")
    complete: dict[int, str] = {}
    assigned = set(int(value) for value in expected["assigned_indices"])
    for record in records:
        if not isinstance(record, Mapping):
            raise BuildError("cache rank row record is malformed")
        index = int(record.get("index", -1))
        digest = str(record.get("sha256", ""))
        if index not in assigned or index in complete:
            raise BuildError("cache rank sidecar has duplicate/unassigned rows")
        if _row_sha256(np.asarray(target[index])) != digest:
            raise BuildError(f"cached target row {index} does not match its resume hash")
        complete[index] = digest
    return complete


def _write_cache_rank_sidecar(
    path: Path,
    *,
    expected: Mapping[str, Any],
    completed: Mapping[int, str],
) -> None:
    attempt = execution_evidence(
        rank=int(expected["rank"]), world_size=int(expected["world_size"])
    )
    attempts: list[dict[str, Any]] = []
    if path.is_file():
        previous = _read_json(path, label="previous cache rank sidecar")
        previous_attempts = previous.get("execution_attempts")
        if isinstance(previous_attempts, list):
            attempts = [dict(value) for value in previous_attempts if isinstance(value, Mapping)]
        elif isinstance(previous.get("execution_evidence"), Mapping):
            attempts = [dict(previous["execution_evidence"])]
    attempt_identity = {
        key: value for key, value in attempt.items() if key != "recorded_at_utc"
    }
    if not attempts or {
        key: value for key, value in attempts[-1].items() if key != "recorded_at_utc"
    } != attempt_identity:
        attempts.append(attempt)
    _atomic_write_json(
        path,
        {
            **expected,
            "completed_rows": [
                {"index": index, "sha256": completed[index]}
                for index in sorted(completed)
            ],
            "complete": len(completed) == len(expected["assigned_indices"]),
            "execution_evidence": attempt,
            "execution_attempts": attempts,
        },
    )


SMOKE_CACHE_ARTIFACT_TYPE = "droid-causal-vjepa2.1-mini-cache-smoke-only"


def _run_smoke_cache_pass(
    *,
    label: str,
    phase: str,
    smoke_root: Path,
    identity: Mapping[str, Any],
    selected: Sequence[tuple[int, Mapping[str, Any]]],
    dataset: DroidVideoLatentForcingDataset,
    encoder: torch.nn.Module,
    stats: PCAWhiteningStats,
    context: DistributedContext,
    runtime: Mapping[str, Any],
    numerical_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build/reference or intentionally interrupt/resume a smoke-only cache."""
    if phase not in {"fresh", "partial", "resume"}:
        raise BuildError(f"invalid mini-cache smoke phase: {phase}")
    smoke_id = _hash_identity(identity)
    final_dir = smoke_root / label
    staging = smoke_root / f".{label}.building"
    metadata_path = staging / METADATA_FILE
    target_shape = (len(selected), *TARGET_SHAPE)
    initial_metadata = {
        "format_version": FORMAT_VERSION,
        "artifact_type": SMOKE_CACHE_ARTIFACT_TYPE,
        "complete": False,
        "production_eligible": False,
        "production_cache_validation_forbidden": True,
        "synthetic_pca": True,
        "smoke_identity": dict(identity),
        "smoke_id": smoke_id,
        "target_file": TARGET_FILE,
        "target_shape": list(target_shape),
        "target_dtype": "float16",
    }

    if phase == "resume" and final_dir.is_dir():
        complete = _read_json(final_dir / METADATA_FILE, label="completed smoke cache")
        if (
            complete.get("smoke_id") != smoke_id
            or complete.get("complete") is not True
            or complete.get("byte_equivalent_to_reference") is not True
        ):
            raise BuildError("completed smoke resume output is not this exact smoke")
        context.barrier()
        return complete if context.rank == 0 else None
    if final_dir.exists():
        raise BuildError(f"immutable smoke output already exists: {final_dir}")
    if context.rank == 0:
        smoke_root.mkdir(parents=True, exist_ok=True)
        if phase in {"fresh", "partial"}:
            if staging.exists():
                raise BuildError(f"mini-cache smoke staging collision: {staging}")
            staging.mkdir(parents=False, exist_ok=False)
            target = np.lib.format.open_memmap(
                staging / TARGET_FILE,
                mode="w+",
                dtype=np.float16,
                shape=target_shape,
            )
            target.flush()
            del target
            _atomic_write_json(metadata_path, initial_metadata)
        else:
            if not staging.is_dir() or not metadata_path.is_file():
                raise BuildError("mini-cache resume requires a graceful partial staging cache")
            if _read_json(metadata_path, label="smoke staging metadata") != initial_metadata:
                raise BuildError("mini-cache resume identity mismatch")
    context.barrier()
    if _read_json(metadata_path, label="smoke staging metadata") != initial_metadata:
        raise BuildError("mini-cache smoke metadata changed across ranks")

    target_path = staging / TARGET_FILE
    target = np.load(target_path, mmap_mode="r+", allow_pickle=False)
    expected_rank = _cache_rank_expected(
        context.rank,
        context.world_size,
        len(selected),
        cache_id=smoke_id,
        runtime=runtime,
        numerical_contract=numerical_contract,
    )
    sidecar = staging / f"rank-{context.rank:05d}.json"
    completed = (
        _load_completed_rows(sidecar, expected=expected_rank, target=target)
        if phase == "resume"
        else {}
    )
    pending = [
        int(index)
        for index in expected_rank["assigned_indices"]
        if int(index) not in completed
    ]
    if phase == "partial":
        pending = pending[:1]
    for target_index in pending:
        manifest_index = int(selected[target_index][0])
        history, future = _stack_samples(
            dataset, [manifest_index], device=context.device
        )
        projected = extract_causal_vjepa2_target(
            history,
            future,
            encoder=encoder,
            stats=stats,
            encoder_dtype=torch.bfloat16,
        ).cpu().numpy()
        if tuple(projected.shape) != (1, *TARGET_SHAPE) or projected.dtype != np.float16:
            raise BuildError("mini-cache smoke teacher returned invalid target")
        target[target_index] = projected[0]
        completed[target_index] = _row_sha256(np.asarray(target[target_index]))
    target.flush()
    _write_cache_rank_sidecar(
        sidecar, expected=expected_rank, completed=completed
    )
    del target
    context.barrier()
    if phase == "partial":
        result = None
        if context.rank == 0:
            result = {
                **initial_metadata,
                "status": "intentional_graceful_stop",
                "rows_completed": sum(
                    len(
                        _read_json(
                            staging / f"rank-{rank:05d}.json",
                            label="smoke rank sidecar",
                        )["completed_rows"]
                    )
                    for rank in range(context.world_size)
                ),
                "execution_evidence": execution_evidence(
                    rank=0, world_size=context.world_size
                ),
            }
            _atomic_write_json(staging / "graceful-stop.json", result)
        context.barrier()
        return result

    result: dict[str, Any] | None = None
    if context.rank == 0:
        verify = np.load(target_path, mmap_mode="r", allow_pickle=False)
        coverage: set[int] = set()
        for rank in range(context.world_size):
            expected = _cache_rank_expected(
                rank,
                context.world_size,
                len(selected),
                cache_id=smoke_id,
                runtime=runtime,
                numerical_contract=numerical_contract,
            )
            records = _load_completed_rows(
                staging / f"rank-{rank:05d}.json", expected=expected, target=verify
            )
            if set(records) != set(expected["assigned_indices"]):
                raise BuildError(f"mini-cache smoke rank {rank} is incomplete")
            if coverage.intersection(records):
                raise BuildError("mini-cache smoke rank assignments overlap")
            coverage.update(records)
        if coverage != set(range(len(selected))) or not np.isfinite(verify).all():
            raise BuildError("mini-cache smoke target coverage/content is invalid")
        del verify
        digest = sha256_file(target_path)
        result = {
            **initial_metadata,
            "complete": True,
            "target_sha256": digest,
            "byte_equivalent_to_reference": label == "reference",
            "reference_target_sha256": digest if label == "reference" else None,
            "execution_evidence": execution_evidence(
                rank=0, world_size=context.world_size
            ),
        }
        if label != "reference":
            reference_metadata = _read_json(
                smoke_root / "reference" / METADATA_FILE,
                label="reference smoke cache",
            )
            reference_target = smoke_root / "reference" / TARGET_FILE
            if (
                reference_metadata.get("smoke_id") != smoke_id
                or reference_metadata.get("complete") is not True
                or not reference_target.is_file()
                or reference_metadata.get("target_sha256") != sha256_file(reference_target)
                or digest != reference_metadata.get("target_sha256")
                or target_path.stat().st_size != reference_target.stat().st_size
            ):
                raise BuildError("resumed mini-cache is not byte-equivalent to reference")
            result["byte_equivalent_to_reference"] = True
            result["reference_target_sha256"] = reference_metadata["target_sha256"]
        _atomic_write_json(metadata_path, result)
        os.replace(staging, final_dir)
    context.barrier()
    return result


def run_mini_cache_smoke(
    *,
    mode: str,
    train_manifest: Path,
    data_root: Path,
    source_path: Path,
    checkpoint_path: Path,
    smoke_root: Path,
    rows: int,
    context: DistributedContext,
) -> dict[str, Any] | None:
    """Run an 8-rank, real-teacher cache stop/resume equivalence smoke."""
    if context.world_size != 8:
        raise BuildError("mini-cache resume smoke requires exactly eight ranks")
    if rows < 16 or rows > PCA_CLIP_COUNT or rows % context.world_size:
        raise BuildError("smoke rows must be a multiple of 8 in [16,256]")
    if mode not in {"reference", "partial", "resume"}:
        raise BuildError("invalid mini-cache smoke mode")
    numerical_contract = production_numerical_contract(
        context=context,
        encoder_dtype=torch.bfloat16,
        batch_size=1,
        pca_device="cuda",
    )
    runtime = runtime_record()
    train_manifest = train_manifest.expanduser().resolve()
    smoke_root = smoke_root.expanduser().resolve()
    base_droid = validate_frozen_base_droid(
        train_manifest=train_manifest, data_root=data_root
    )
    train_rows = read_clip_manifest(train_manifest, expected_split="train")
    selected = select_pca_rows(train_rows)[:rows]
    provenance = validate_teacher_inputs(
        source_path=source_path, checkpoint_path=checkpoint_path
    )
    implementation = validate_builder_source()
    identity = {
        "schema": SCHEMA,
        "artifact_type": SMOKE_CACHE_ARTIFACT_TYPE,
        "production_eligible": False,
        "synthetic_pca": {
            "mean": "zeros-768",
            "components": "first-48-coordinate-axes",
            "eigenvalues": "ones-48",
            "eps": WHITENING_EPS,
        },
        "selected_clip_ids": [str(row["clip_id"]) for _, row in selected],
        "selected_manifest_indices": [int(index) for index, _ in selected],
        "world_size": context.world_size,
        "base_droid": base_droid,
        "source_commit": provenance["source_commit"],
        "source_archive_sha256": provenance["source_archive_sha256"],
        "source_license": provenance["source_license"],
        "checkpoint_evidence": provenance["checkpoint_evidence"],
        "implementation": implementation,
        "runtime": runtime,
        "numerical_contract": numerical_contract,
        "protected_test_access": False,
    }
    dataset = DroidVideoLatentForcingDataset(train_manifest, data_root)
    encoder = _load_teacher(provenance, device=context.device)
    stats = PCAWhiteningStats(
        mean=torch.zeros(VJEPA2_1_SOURCE_DIM),
        components=torch.eye(VJEPA2_1_SOURCE_DIM)[:TARGET_CHANNELS].contiguous(),
        eigenvalues=torch.ones(TARGET_CHANNELS),
        eps=WHITENING_EPS,
    )
    result: dict[str, Any] | None = None
    if mode == "reference":
        result = _run_smoke_cache_pass(
            label="reference",
            phase="fresh",
            smoke_root=smoke_root,
            identity=identity,
            selected=selected,
            dataset=dataset,
            encoder=encoder,
            stats=stats,
            context=context,
            runtime=runtime,
            numerical_contract=numerical_contract,
        )
    if mode == "partial":
        if not (smoke_root / "reference" / METADATA_FILE).is_file():
            raise BuildError("partial smoke requires a completed reference cache")
        result = _run_smoke_cache_pass(
            label="resumed",
            phase="partial",
            smoke_root=smoke_root,
            identity=identity,
            selected=selected,
            dataset=dataset,
            encoder=encoder,
            stats=stats,
            context=context,
            runtime=runtime,
            numerical_contract=numerical_contract,
        )
    if mode == "resume":
        result = _run_smoke_cache_pass(
            label="resumed",
            phase="resume",
            smoke_root=smoke_root,
            identity=identity,
            selected=selected,
            dataset=dataset,
            encoder=encoder,
            stats=stats,
            context=context,
            runtime=runtime,
            numerical_contract=numerical_contract,
        )
    return result


def extract_cache(
    *,
    manifest: Path,
    train_manifest: Path,
    data_root: Path,
    source_path: Path,
    checkpoint_path: Path,
    pca_path: Path,
    semantic_cache_root: Path,
    context: DistributedContext,
    encoder_dtype: torch.dtype,
    batch_size: int,
    resume: bool,
) -> dict[str, Any] | None:
    manifest = manifest.expanduser().resolve()
    train_manifest = train_manifest.expanduser().resolve()
    pca_path = pca_path.expanduser().resolve()
    numerical_contract = production_numerical_contract(
        context=context,
        encoder_dtype=encoder_dtype,
        batch_size=batch_size,
        pca_device="cuda",
    )
    runtime = runtime_record()
    base_droid = validate_frozen_base_droid(
        train_manifest=train_manifest, data_root=data_root
    )
    rows = read_clip_manifest(manifest)
    split = str(rows[0]["split"])
    if split not in {"train", "val"}:
        raise BuildError("offline semantic caches are limited to train and val")
    expected_manifest = Path(base_droid["manifests"][split]["path"])
    if manifest != expected_manifest:
        raise BuildError(f"{split} cache manifest is not the frozen base manifest")
    if split == "train" and manifest != train_manifest:
        raise BuildError("train cache extraction requires manifest=train-manifest")
    read_clip_manifest(train_manifest, expected_split="train")
    stats, pca_payload, pca_digest = validate_pca_artifact(
        pca_path,
        train_manifest_path=train_manifest,
        expected_base_droid=base_droid,
        expected_runtime=runtime,
        expected_numerical_contract=numerical_contract,
    )
    provenance = validate_teacher_inputs(
        source_path=source_path, checkpoint_path=checkpoint_path
    )
    implementation = validate_builder_source()
    if (
        pca_payload.get("source_commit") != provenance["source_commit"]
        or pca_payload.get("checkpoint_sha256") != provenance["checkpoint_sha256"]
        or pca_payload.get("source_archive_sha256")
        != provenance["source_archive_sha256"]
        or pca_payload.get("source_license") != provenance["source_license"]
        or pca_payload.get("checkpoint_evidence")
        != provenance["checkpoint_evidence"]
        or pca_payload.get("implementation") != implementation
        or pca_payload.get("runtime") != runtime
        or pca_payload.get("numerical_contract") != numerical_contract
        or pca_payload.get("base_droid") != base_droid
    ):
        raise BuildError("PCA/source/implementation/runtime identity differs before extraction")
    cache_dir = semantic_cache_root.expanduser().resolve() / split
    metadata_path = cache_dir / METADATA_FILE
    identity = _cache_identity(
        manifest=manifest,
        rows=rows,
        train_manifest=train_manifest,
        pca_path=pca_path,
        pca_digest=pca_digest,
        provenance=provenance,
        implementation=implementation,
        base_droid=base_droid,
        runtime=runtime,
        numerical_contract=numerical_contract,
        world_size=context.world_size,
    )
    cache_id = _hash_identity(identity)
    staging = cache_dir.parent / f".{cache_dir.name}.building"
    staging_metadata = staging / METADATA_FILE
    if context.rank == 0 and resume and not cache_dir.exists() and staging.is_dir():
        possible_complete = _read_json(
            staging_metadata, label="semantic-cache staging metadata"
        )
        if possible_complete.get("complete") is True:
            if (
                possible_complete.get("artifact_identity") != identity
                or possible_complete.get("cache_id") != cache_id
            ):
                raise BuildError("complete cache staging belongs to another invocation")
            validate_causal_cache(
                manifest_path=manifest,
                cache_metadata_path=staging_metadata,
                expected_split=split,
            )
            os.replace(staging, cache_dir)
    context.barrier()
    if cache_dir.exists():
        if not resume or not metadata_path.is_file():
            raise BuildError("semantic cache collision; choose a new root or exact --resume")
        metadata, _ = validate_causal_cache(
            manifest_path=manifest,
            cache_metadata_path=metadata_path,
            expected_split=split,
        )
        if metadata.get("cache_id") != cache_id:
            raise BuildError("completed semantic cache does not match this invocation")
        context.barrier()
        return metadata if context.rank == 0 else None

    initial_metadata = {
        "format_version": FORMAT_VERSION,
        **identity,
        "artifact_identity": identity,
        "cache_id": cache_id,
        "complete": False,
        "target_file": TARGET_FILE,
        "target_sha256": None,
    }
    if context.rank == 0:
        if staging.exists():
            if not resume or not staging_metadata.is_file():
                raise BuildError("semantic-cache staging collision")
            if _read_json(staging_metadata, label="staging metadata") != initial_metadata:
                raise BuildError("semantic-cache resume identity mismatch")
        else:
            staging.mkdir(parents=True, exist_ok=False)
            target = np.lib.format.open_memmap(
                staging / TARGET_FILE,
                mode="w+",
                dtype=np.float16,
                shape=tuple(identity["target_shape"]),
            )
            target.flush()
            del target
            _atomic_write_json(staging_metadata, initial_metadata)
    context.barrier()
    if _read_json(staging_metadata, label="staging metadata") != initial_metadata:
        raise BuildError("semantic-cache staging metadata changed across ranks")

    target_path = staging / TARGET_FILE
    target = np.load(target_path, mmap_mode="r+", allow_pickle=False)
    expected_rank = _cache_rank_expected(
        context.rank,
        context.world_size,
        len(rows),
        cache_id=cache_id,
        runtime=runtime,
        numerical_contract=numerical_contract,
    )
    sidecar = staging / f"rank-{context.rank:05d}.json"
    completed = (
        _load_completed_rows(sidecar, expected=expected_rank, target=target)
        if resume
        else {}
    )
    if sidecar.exists() and not resume:
        raise BuildError("cache rank sidecar collision")
    pending = [
        index for index in expected_rank["assigned_indices"] if int(index) not in completed
    ]
    dataset = DroidVideoLatentForcingDataset(manifest, data_root)
    encoder = _load_teacher(provenance, device=context.device)
    rows_since_progress = 0
    for start in range(0, len(pending), batch_size):
        indices = [int(value) for value in pending[start : start + batch_size]]
        history, future = _stack_samples(dataset, indices, device=context.device)
        projected = extract_causal_vjepa2_target(
            history,
            future,
            encoder=encoder,
            stats=stats,
            encoder_dtype=encoder_dtype,
        ).cpu().numpy()
        if tuple(projected.shape[1:]) != TARGET_SHAPE or projected.dtype != np.float16:
            raise BuildError(f"unexpected projected target shape/dtype: {projected.shape}")
        if not np.isfinite(projected).all():
            raise BuildError("projected cache batch is non-finite")
        target[indices] = projected
        for local, index in enumerate(indices):
            completed[index] = _row_sha256(np.asarray(target[index]))
        rows_since_progress += len(indices)
        if (
            rows_since_progress >= PROGRESS_CHECKPOINT_ROWS
            or start + len(indices) >= len(pending)
        ):
            target.flush()
            _write_cache_rank_sidecar(
                sidecar, expected=expected_rank, completed=completed
            )
            rows_since_progress = 0
    target.flush()
    _write_cache_rank_sidecar(
        sidecar, expected=expected_rank, completed=completed
    )
    del encoder
    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    context.barrier()

    result: dict[str, Any] | None = None
    if context.rank == 0:
        verify = np.load(target_path, mmap_mode="r", allow_pickle=False)
        coverage: set[int] = set()
        for rank in range(context.world_size):
            expected = _cache_rank_expected(
                rank,
                context.world_size,
                len(rows),
                cache_id=cache_id,
                runtime=runtime,
                numerical_contract=numerical_contract,
            )
            rank_sidecar = staging / f"rank-{rank:05d}.json"
            records = _load_completed_rows(
                rank_sidecar, expected=expected, target=verify
            )
            if set(records) != set(expected["assigned_indices"]):
                raise BuildError(f"semantic-cache rank {rank} is incomplete")
            if coverage.intersection(records):
                raise BuildError("semantic-cache rank sidecars overlap")
            coverage.update(records)
        if coverage != set(range(len(rows))):
            raise BuildError("semantic-cache rows do not have exact coverage")
        for start in range(0, len(rows), 64):
            if not np.isfinite(np.asarray(verify[start : start + 64])).all():
                raise BuildError(f"semantic-cache target is non-finite near row {start}")
        del verify
        digest = sha256_file(target_path)
        result = {
            **initial_metadata,
            "complete": True,
            "target_sha256": digest,
            "publication_execution": execution_evidence(
                rank=context.rank, world_size=context.world_size
            ),
            "evidence": {
                "target": {
                    "path": str((cache_dir / TARGET_FILE).resolve()),
                    "sha256": digest,
                    "bytes": target_path.stat().st_size,
                },
                "manifest": {
                    "path": str(manifest),
                    "sha256": identity["manifest_sha256"],
                    "bytes": manifest.stat().st_size,
                },
                "train_manifest": {
                    "path": str(train_manifest),
                    "sha256": identity["train_manifest_sha256"],
                    "bytes": train_manifest.stat().st_size,
                },
                "pca": {
                    "path": str(pca_path),
                    "sha256": pca_digest,
                    "bytes": pca_path.stat().st_size,
                },
                "source_license": provenance["source_license"],
            },
        }
        _atomic_write_json(staging_metadata, result)
        if cache_dir.exists():
            raise BuildError("semantic cache appeared during atomic publication")
        os.replace(staging, cache_dir)
        validate_causal_cache(
            manifest_path=manifest,
            cache_metadata_path=cache_dir / METADATA_FILE,
            expected_split=split,
        )
    context.barrier()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    teacher = argparse.ArgumentParser(add_help=False)
    teacher.add_argument("--source-path", type=Path, required=True)
    teacher.add_argument("--checkpoint-path", type=Path, required=True)

    validate_inputs = subparsers.add_parser("validate-inputs", parents=[teacher])

    preflight = subparsers.add_parser("preflight-teacher", parents=[teacher])
    preflight.add_argument("--train-manifest", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--device", choices=("cuda",), default="cuda")

    smoke = subparsers.add_parser("smoke-mini-cache", parents=[teacher])
    smoke.add_argument(
        "--mode",
        choices=("reference", "partial", "resume"),
        required=True,
    )
    smoke.add_argument("--train-manifest", type=Path, required=True)
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--smoke-root", type=Path, required=True)
    smoke.add_argument("--rows", type=int, default=16)
    smoke.add_argument("--device", choices=("cuda",), default="cuda")

    fit = subparsers.add_parser("fit-pca", parents=[teacher])
    fit.add_argument("--train-manifest", type=Path, required=True)
    fit.add_argument("--data-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", choices=("cuda",), default="cuda")
    fit.add_argument("--pca-device", choices=("cuda",), default="cuda")
    fit.add_argument(
        "--encoder-dtype",
        choices=("bfloat16",),
        default="bfloat16",
    )
    fit.add_argument("--batch-size", type=int, default=1)
    fit.add_argument("--resume", action="store_true")

    extract = subparsers.add_parser("extract", parents=[teacher])
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--train-manifest", type=Path, required=True)
    extract.add_argument("--data-root", type=Path, required=True)
    extract.add_argument("--pca", type=Path, required=True)
    extract.add_argument("--semantic-cache-root", type=Path, required=True)
    extract.add_argument("--device", choices=("cuda",), default="cuda")
    extract.add_argument(
        "--encoder-dtype",
        choices=("bfloat16",),
        default="bfloat16",
    )
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--resume", action="store_true")

    validate_pca = subparsers.add_parser("validate-pca")
    validate_pca.add_argument("--pca", type=Path, required=True)
    validate_pca.add_argument("--train-manifest", type=Path, required=True)

    validate_cache = subparsers.add_parser("validate-cache")
    validate_cache.add_argument("--manifest", type=Path, required=True)
    validate_cache.add_argument("--semantic-cache-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "batch_size", 1) != 1:
        raise BuildError("production encoder batch-size is frozen at exactly 1")
    if args.command == "validate-inputs":
        print(
            json.dumps(
                validate_teacher_inputs(
                    source_path=args.source_path,
                    checkpoint_path=args.checkpoint_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-pca":
        _, payload, digest = validate_pca_artifact(
            args.pca, train_manifest_path=args.train_manifest
        )
        print(
            json.dumps(
                {
                    "artifact_id": payload["artifact_id"],
                    "artifact_sha256": digest,
                    "sampled_token_count": payload["sampled_token_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-cache":
        rows = read_clip_manifest(args.manifest)
        metadata, _ = validate_causal_cache(
            manifest_path=args.manifest,
            cache_metadata_path=(
                args.semantic_cache_root / str(rows[0]["split"]) / METADATA_FILE
            ),
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    context = _distributed_context(args.device)
    try:
        if args.command == "preflight-teacher":
            result = run_teacher_preflight(
                train_manifest=args.train_manifest,
                data_root=args.data_root,
                source_path=args.source_path,
                checkpoint_path=args.checkpoint_path,
                output_path=approved_artifact_path(args.output),
                context=context,
            )
        elif args.command == "smoke-mini-cache":
            result = run_mini_cache_smoke(
                mode=args.mode,
                train_manifest=args.train_manifest,
                data_root=args.data_root,
                source_path=args.source_path,
                checkpoint_path=args.checkpoint_path,
                smoke_root=approved_artifact_path(args.smoke_root),
                rows=args.rows,
                context=context,
            )
        elif args.command == "fit-pca":
            result = fit_pca(
                train_manifest=args.train_manifest,
                data_root=args.data_root,
                source_path=args.source_path,
                checkpoint_path=args.checkpoint_path,
                output_path=approved_artifact_path(args.output),
                context=context,
                encoder_dtype=_encoder_dtype(args.encoder_dtype),
                batch_size=args.batch_size,
                pca_device=args.pca_device,
                resume=args.resume,
            )
        elif args.command == "extract":
            result = extract_cache(
                manifest=args.manifest,
                train_manifest=args.train_manifest,
                data_root=args.data_root,
                source_path=args.source_path,
                checkpoint_path=args.checkpoint_path,
                pca_path=args.pca,
                semantic_cache_root=approved_artifact_path(args.semantic_cache_root),
                context=context,
                encoder_dtype=_encoder_dtype(args.encoder_dtype),
                batch_size=args.batch_size,
                resume=args.resume,
            )
        else:  # pragma: no cover - argparse exhausts choices
            raise AssertionError(args.command)
        if context.rank == 0 and result is not None:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
