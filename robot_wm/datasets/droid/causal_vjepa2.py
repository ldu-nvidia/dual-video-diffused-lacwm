"""Prefix-causal V-JEPA2.1 targets for fixed DROID forecasting clips.

The V-JEPA teacher is an offline data-construction dependency only.  Training
uses :class:`DroidCausalVJEPA2Dataset`, which wraps the immutable one-view
DROID dataset and reads a split-specific, read-only float16 target memmap.

For future step ``j`` the teacher input is the native-length prefix

``[h0] * (10-j) + [h0,h1,h2,h3,h4] + [f0,...,fj]``.

It is exactly 16 frames.  Only temporal token seven (the final tubelet) is
retained, so its local pair is ``(h4,f0)`` for ``j=0`` and
``(f[j-1],f[j])`` thereafter.  No prefix contains a future frame after ``j``.
The 24x42 spatial grid is average-pooled in non-overlapping 3x3 windows and a
train-only PCA whitening transform projects 768 channels to 48, producing
``[48,8,8,14]`` per clip.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from robot_wm.datasets.droid.video_latent_forcing import (
    FUTURE_SIZE,
    HISTORY_SIZE,
    DroidVideoLatentForcingDataset,
    canonical_json,
    sha256_file,
)
from robot_wm.modeling.dual_diffusion.vjepa2_target import (
    PCAWhiteningStats,
    VJEPA2_1_MODEL_NAME,
    VJEPA2_1_PATCH_SIZE,
    VJEPA2_1_RELEASE_COMMIT,
    VJEPA2_1_SOURCE_DIM,
    VJEPA2_1_TUBELET_SIZE,
    prepare_vjepa2_1_views,
)


SCHEMA = "droid-causal-vjepa2.1-v1"
PCA_ARTIFACT_TYPE = "droid-causal-vjepa2.1-pca48-whitening"
CACHE_ARTIFACT_TYPE = "droid-causal-vjepa2.1-target-cache"
TARGET_KIND = "prefix-causal-vjepa2-pca48-whitened-v1"
FORMAT_VERSION = 1

VJEPA2_SOURCE_COMMIT = VJEPA2_1_RELEASE_COMMIT
VJEPA2_CHECKPOINT_SHA256 = (
    "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
)
VJEPA2_LICENSE_SHA256 = (
    "cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173"
)
TEACHER_SIZE = (384, 672)
TEACHER_FRAMES = 16
TEACHER_TEMPORAL_TOKENS = 8
LAST_TEMPORAL_TOKEN = 7
TEACHER_TOKEN_GRID = (24, 42)
POOL_KERNEL = (3, 3)
POOLED_TOKEN_GRID = (8, 14)
TARGET_CHANNELS = 48
TARGET_SHAPE = (TARGET_CHANNELS, FUTURE_SIZE, *POOLED_TOKEN_GRID)
TOKENS_PER_CLIP = FUTURE_SIZE * POOLED_TOKEN_GRID[0] * POOLED_TOKEN_GRID[1]
PCA_CLIP_COUNT = 256
PCA_RANK_PREFIX = "causal-vjepa2-pca-v1:"
WHITENING_EPS = 1e-6

# Immutable Phase-3 input population.  These are byte identities of the
# already-published, one-view DROID POC artifact, not merely expected labels.
FROZEN_BASE_PROVENANCE_SHA256 = (
    "3320244e843ccaa84828b4bbecb9c227870706be3bbbc0e6e1c28eda1ac317e0"
)
FROZEN_BASE_MANIFEST_SHA256 = {
    "train": "cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e",
    "val": "b8773a8627e887bf0c0a31cfae6ff537ba6a99e0b1b4f11efec011e3983d8d99",
    "test": "e05bb46087152c4ea07820ce8290ae99c0a084dbcf5d576aac370a61486e925d",
}
FROZEN_BASE_CLIP_COUNTS = {"train": 64_000, "val": 890, "test": 890}
FROZEN_BASE_EPISODE_COUNTS = {"train": 8_000, "val": 890, "test": 890}
FROZEN_ELIGIBLE_INVENTORY_SHA256 = (
    "3bc6f2c06abe74f1a60ddc4f9a44ce734fb8fa85f9ec94ac99e7bcc954993651"
)
FROZEN_SPLIT_EPISODE_IDS_SHA256 = {
    "train": "cea3449f2a7cf3e9251b1fb1859f4fd8c3717a4ce29a344f2f48c65452e3ac12",
    "val": "58f1a863a7be8f273212030c902c568b32ed75df5aa79993a8aa5c1a7a0252e6",
    "test": "aa36a731cde260ddb94875a0678435b339be18ebeaa4584b507cdbaeac71ba11",
}
FROZEN_DROID_DATA_ROOT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/"
    "lacwm_train/data/production_v1/droid_lerobot"
)
PRODUCTION_NUMERICAL_CONTRACT = {
    "encoder_device": "cuda",
    "encoder_dtype": "bfloat16",
    "encoder_batch_size": 1,
    "pca_device": "cuda",
    "pca_algorithm": "exact-centered-covariance-eigh",
    "pca_covariance_dtype": "float32",
    "pca_tf32": False,
}


class CausalVJEPA2Error(RuntimeError):
    """A causal target or immutable cache violated its frozen contract."""


def manifest_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact manifest row order without duplicating source bytes."""
    return hashlib.sha256(
        canonical_json(
            {
                "schema": SCHEMA,
                "ordered_clip_ids": [str(row["clip_id"]) for row in rows],
            }
        ).encode("utf-8")
    ).hexdigest()


def pca_clip_rank(clip_id: str) -> str:
    """Stable rank used to choose the frozen train-only PCA population."""
    return hashlib.sha256(f"{PCA_RANK_PREFIX}{clip_id}".encode("utf-8")).hexdigest()


def select_pca_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, Mapping[str, Any]]]:
    """Return exactly 256 hash-ranked training rows and their indices."""
    if len(rows) < PCA_CLIP_COUNT:
        raise CausalVJEPA2Error(
            f"PCA requires {PCA_CLIP_COUNT} train clips but the manifest has {len(rows)}"
        )
    if any(str(row.get("split")) != "train" for row in rows):
        raise CausalVJEPA2Error("PCA selection may use only training rows")
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            pca_clip_rank(str(item[1]["clip_id"])),
            str(item[1]["clip_id"]),
            item[0],
        ),
    )
    return ranked[:PCA_CLIP_COUNT]


def build_prefix_causal_clips(history: Tensor, future: Tensor) -> Tensor:
    """Construct the eight exact 16-frame prefixes.

    Parameters use the DROID dataset layout ``history=[B,3,5,H,W]`` and
    ``future=[B,3,8,H,W]``.  The result is ``[B,8,3,16,H,W]``.
    """
    if history.ndim != 5 or future.ndim != 5:
        raise ValueError("history and future must be [B,C,T,H,W]")
    if history.shape[:3] != (history.shape[0], 3, HISTORY_SIZE):
        raise ValueError(
            f"history must be [B,3,{HISTORY_SIZE},H,W], got {tuple(history.shape)}"
        )
    if future.shape[:3] != (future.shape[0], 3, FUTURE_SIZE):
        raise ValueError(
            f"future must be [B,3,{FUTURE_SIZE},H,W], got {tuple(future.shape)}"
        )
    if history.shape[0] != future.shape[0] or history.shape[-2:] != future.shape[-2:]:
        raise ValueError("history and future batch/spatial dimensions must match")
    if history.dtype != future.dtype or history.device != future.device:
        raise ValueError("history and future dtype/device must match")
    if not history.is_floating_point() or not future.is_floating_point():
        raise TypeError("history and future must be floating point RGB")
    if not torch.isfinite(history).all() or not torch.isfinite(future).all():
        raise FloatingPointError("history/future contains non-finite RGB")
    if (
        torch.any(history < -1.0)
        or torch.any(history > 1.0)
        or torch.any(future < -1.0)
        or torch.any(future > 1.0)
    ):
        raise ValueError("history/future must use the [-1,1] RGB convention")

    h0 = history[:, :, :1]
    prefixes = []
    for future_index in range(FUTURE_SIZE):
        padding = h0.expand(-1, -1, 10 - future_index, -1, -1)
        prefix = torch.cat(
            (padding, history, future[:, :, : future_index + 1]), dim=2
        )
        if prefix.shape[2] != TEACHER_FRAMES:  # defensive against contract edits
            raise AssertionError("prefix construction did not produce 16 frames")
        prefixes.append(prefix)
    return torch.stack(prefixes, dim=1)


def prepare_prefix_causal_teacher_input(
    history: Tensor,
    future: Tensor,
    *,
    teacher_size: tuple[int, int] = TEACHER_SIZE,
) -> Tensor:
    """Normalize prefixes to ``[B*8,3,16,H_teacher,W_teacher]``."""
    prefixes = build_prefix_causal_clips(history, future)
    batch, steps, channels, frames, height, width = prefixes.shape
    flat = prefixes.permute(0, 1, 3, 2, 4, 5).reshape(
        batch * steps, frames, channels, height, width
    )
    return prepare_vjepa2_1_views(
        flat,
        num_views=1,
        frame_map=tuple(range(TEACHER_FRAMES)),
        expected_view_size=(height, width),
        padded_view_size=(height, width),
        teacher_size=teacher_size,
    )


def select_and_pool_last_tubelet(
    tokens: Tensor,
    *,
    batch_size: int,
    teacher_size: tuple[int, int] = TEACHER_SIZE,
    patch_size: int = VJEPA2_1_PATCH_SIZE,
    tubelet_size: int = VJEPA2_1_TUBELET_SIZE,
    pool_kernel: tuple[int, int] = POOL_KERNEL,
    expected_source_dim: int | None = VJEPA2_1_SOURCE_DIM,
) -> Tensor:
    """Keep temporal token seven and pool to ``[B,8,8,14,D]``."""
    if tokens.ndim != 3:
        raise ValueError("V-JEPA output must be [B*8,N,D]")
    if tokens.shape[0] != batch_size * FUTURE_SIZE:
        raise ValueError("V-JEPA batch does not match B*8 causal prefixes")
    if teacher_size[0] % patch_size or teacher_size[1] % patch_size:
        raise ValueError("teacher size must divide by the patch size")
    temporal_tokens = TEACHER_FRAMES // tubelet_size
    token_height = teacher_size[0] // patch_size
    token_width = teacher_size[1] // patch_size
    expected_tokens = temporal_tokens * token_height * token_width
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} dense tokens, got {tokens.shape[1]}"
        )
    if expected_source_dim is not None and tokens.shape[-1] != expected_source_dim:
        raise ValueError(
            f"expected token width {expected_source_dim}, got {tokens.shape[-1]}"
        )
    if temporal_tokens <= LAST_TEMPORAL_TOKEN:
        raise ValueError("teacher does not expose temporal token index seven")
    if token_height % pool_kernel[0] or token_width % pool_kernel[1]:
        raise ValueError("spatial token grid must divide by the pooling kernel")

    grid = tokens.reshape(
        batch_size,
        FUTURE_SIZE,
        temporal_tokens,
        token_height,
        token_width,
        tokens.shape[-1],
    )[:, :, LAST_TEMPORAL_TOKEN]
    channels_first = grid.permute(0, 1, 4, 2, 3).reshape(
        batch_size * FUTURE_SIZE,
        tokens.shape[-1],
        token_height,
        token_width,
    )
    pooled = F.avg_pool2d(
        channels_first, kernel_size=pool_kernel, stride=pool_kernel
    )
    pooled_height, pooled_width = pooled.shape[-2:]
    return pooled.reshape(
        batch_size,
        FUTURE_SIZE,
        tokens.shape[-1],
        pooled_height,
        pooled_width,
    ).permute(0, 1, 3, 4, 2).contiguous()


@torch.inference_mode()
def extract_causal_vjepa2_tokens(
    history: Tensor,
    future: Tensor,
    *,
    encoder: nn.Module,
    encoder_dtype: torch.dtype = torch.bfloat16,
    teacher_size: tuple[int, int] = TEACHER_SIZE,
    patch_size: int = VJEPA2_1_PATCH_SIZE,
    pool_kernel: tuple[int, int] = POOL_KERNEL,
    expected_source_dim: int | None = VJEPA2_1_SOURCE_DIM,
) -> Tensor:
    """Run the frozen teacher and return unprojected causal pooled tokens."""
    prepared = prepare_prefix_causal_teacher_input(
        history, future, teacher_size=teacher_size
    )
    encoder.eval()
    with torch.autocast(
        device_type=prepared.device.type,
        dtype=encoder_dtype,
        enabled=prepared.device.type == "cuda",
    ):
        tokens = encoder(prepared, training=False)
    if not isinstance(tokens, Tensor):
        raise TypeError("V-JEPA encoder must return one dense token tensor")
    pooled = select_and_pool_last_tubelet(
        tokens,
        batch_size=history.shape[0],
        teacher_size=teacher_size,
        patch_size=patch_size,
        pool_kernel=pool_kernel,
        expected_source_dim=expected_source_dim,
    )
    if not torch.isfinite(pooled).all():
        raise FloatingPointError("V-JEPA causal tokens are non-finite")
    return pooled


def project_causal_vjepa2_tokens(
    tokens: Tensor,
    stats: PCAWhiteningStats,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Whiten ``[B,8,8,14,768]`` to ``[B,48,8,8,14]``."""
    if tokens.ndim != 5:
        raise ValueError("causal tokens must be [B,8,H,W,D]")
    if tokens.shape[1:4] != (FUTURE_SIZE, *POOLED_TOKEN_GRID):
        raise ValueError(
            "causal tokens do not use the frozen [8,8,14] target grid"
        )
    if stats.input_dim != VJEPA2_1_SOURCE_DIM or stats.output_dim != TARGET_CHANNELS:
        raise ValueError("PCA statistics must project exactly 768->48")
    projected = stats.project(tokens)
    if not torch.isfinite(projected).all():
        raise FloatingPointError("projected causal V-JEPA target is non-finite")
    return projected.permute(0, 4, 1, 2, 3).contiguous().to(output_dtype)


@torch.inference_mode()
def extract_causal_vjepa2_target(
    history: Tensor,
    future: Tensor,
    *,
    encoder: nn.Module,
    stats: PCAWhiteningStats,
    encoder_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Construct the production ``[B,48,8,8,14]`` offline target."""
    tokens = extract_causal_vjepa2_tokens(
        history,
        future,
        encoder=encoder,
        encoder_dtype=encoder_dtype,
    )
    return project_causal_vjepa2_tokens(tokens, stats)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CausalVJEPA2Error(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CausalVJEPA2Error(f"{label} must contain a JSON object")
    return payload


def _resolve_bound_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CausalVJEPA2Error(f"cache metadata lacks {label}")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise CausalVJEPA2Error(f"bound {label} is missing: {path}")
    return path


def identity_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical content identity used by PCA and cache artifacts."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_file_record(value: Any, *, label: str, verify: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CausalVJEPA2Error(f"{label} evidence is missing")
    path_value = value.get("path")
    digest = str(value.get("sha256", ""))
    size = value.get("bytes")
    if not isinstance(path_value, str) or not path_value or len(digest) != 64:
        raise CausalVJEPA2Error(f"{label} evidence is malformed")
    try:
        int(digest, 16)
        size = int(size)
    except (TypeError, ValueError) as exc:
        raise CausalVJEPA2Error(f"{label} evidence is malformed") from exc
    if size < 0:
        raise CausalVJEPA2Error(f"{label} byte count is malformed")
    record = {
        "path": str(Path(path_value).expanduser().resolve()),
        "sha256": digest,
        "bytes": size,
    }
    if verify:
        path = Path(record["path"])
        if not path.is_file():
            raise CausalVJEPA2Error(f"{label} evidence is missing: {path}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise CausalVJEPA2Error(f"{label} evidence content does not match its record")
    return record


def validate_frozen_base_record(
    value: Any,
    *,
    expected_split: str | None = None,
    manifest_path: str | Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate the exact immutable DROID POC population bound to an artifact."""
    if not isinstance(value, Mapping):
        raise CausalVJEPA2Error("artifact lacks frozen base-DROID provenance")
    record = dict(value)
    if (
        record.get("schema") != "frozen-droid-video-latent-forcing-poc-v1"
        or record.get("eligible_inventory_sha256")
        != FROZEN_ELIGIBLE_INVENTORY_SHA256
        or record.get("split_episode_ids_sha256")
        != FROZEN_SPLIT_EPISODE_IDS_SHA256
        or record.get("clip_counts") != FROZEN_BASE_CLIP_COUNTS
        or record.get("episode_counts") != FROZEN_BASE_EPISODE_COUNTS
        or record.get("data_root") != FROZEN_DROID_DATA_ROOT
        or record.get("split_disjoint") is not True
        or record.get("protected_test_payload_cached") is not False
    ):
        raise CausalVJEPA2Error("base-DROID identity is not the frozen POC population")
    provenance = _validate_file_record(
        record.get("provenance"), label="base-DROID provenance", verify=verify_files
    )
    if provenance["sha256"] != FROZEN_BASE_PROVENANCE_SHA256:
        raise CausalVJEPA2Error("base-DROID provenance SHA-256 mismatch")
    manifests = record.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {"train", "val", "test"}:
        raise CausalVJEPA2Error("base-DROID manifest evidence is incomplete")
    normalized_manifests: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        entry = manifests[split]
        if not isinstance(entry, Mapping):
            raise CausalVJEPA2Error(f"base-DROID {split} manifest record is malformed")
        evidence = _validate_file_record(
            entry, label=f"base-DROID {split} manifest", verify=verify_files
        )
        if (
            evidence["sha256"] != FROZEN_BASE_MANIFEST_SHA256[split]
            or int(entry.get("clip_count", -1)) != FROZEN_BASE_CLIP_COUNTS[split]
            or int(entry.get("episode_count", -1)) != FROZEN_BASE_EPISODE_COUNTS[split]
            or entry.get("episode_ids_sha256")
            != FROZEN_SPLIT_EPISODE_IDS_SHA256[split]
            or entry.get("protected") != (split == "test")
            or entry.get("cached") != (split != "test")
        ):
            raise CausalVJEPA2Error(f"base-DROID {split} manifest identity mismatch")
        normalized_manifests[split] = {**dict(entry), **evidence}
    if expected_split is not None:
        if expected_split not in normalized_manifests:
            raise CausalVJEPA2Error("unexpected base-DROID split")
        if manifest_path is None:
            raise CausalVJEPA2Error("manifest path is required for split binding")
        actual = Path(manifest_path).expanduser().resolve()
        if actual != Path(normalized_manifests[expected_split]["path"]):
            raise CausalVJEPA2Error(
                f"{expected_split} manifest is not the frozen base-DROID manifest"
            )
    return {
        **record,
        "provenance": provenance,
        "manifests": normalized_manifests,
    }


def _validate_runtime_record(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CausalVJEPA2Error("artifact lacks runtime provenance")
    expected_keys = {"python", "torch", "cuda", "numpy"}
    if set(value) != expected_keys or any(
        not isinstance(value[key], str) for key in expected_keys
    ):
        raise CausalVJEPA2Error("artifact runtime provenance is malformed")
    return {key: str(value[key]) for key in sorted(expected_keys)}


def _validate_numerical_contract(value: Any) -> dict[str, Any]:
    if value != PRODUCTION_NUMERICAL_CONTRACT:
        raise CausalVJEPA2Error("artifact numerical contract is not production CUDA/BF16")
    return dict(value)


def _validate_implementation_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CausalVJEPA2Error("target artifact lacks implementation provenance")
    commit = str(value.get("repo_commit", ""))
    if len(commit) != 40:
        raise CausalVJEPA2Error("implementation repository commit is malformed")
    try:
        int(commit, 16)
    except ValueError as exc:
        raise CausalVJEPA2Error("implementation repository commit is malformed") from exc
    repo_root_value = value.get("repo_root")
    if not isinstance(repo_root_value, str):
        raise CausalVJEPA2Error("implementation repository root is missing")
    repo_root = Path(repo_root_value).expanduser().resolve()
    expected_paths = {
        "builder_source": repo_root / "tools" / "build_causal_vjepa2_droid.py",
        "dataset_source": Path(__file__).resolve(),
    }
    normalized = dict(value)
    normalized["repo_root"] = str(repo_root)
    for label, expected_path in expected_paths.items():
        record = _validate_file_record(
            value.get(label), label=f"implementation {label}", verify=True
        )
        if Path(record["path"]) != expected_path.resolve():
            raise CausalVJEPA2Error(f"implementation {label} path is not the active source")
        normalized[label] = record
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CausalVJEPA2Error("cannot validate implementation Git provenance") from exc
    if actual_commit != commit or status:
        raise CausalVJEPA2Error("implementation is not the recorded clean Git commit")
    return normalized


def _validate_explicit_identity(
    payload: Mapping[str, Any], *, identity_key: str, digest_key: str
) -> dict[str, Any]:
    identity = payload.get(identity_key)
    if not isinstance(identity, Mapping):
        raise CausalVJEPA2Error(f"artifact lacks explicit {identity_key}")
    identity = dict(identity)
    if payload.get(digest_key) != identity_sha256(identity):
        raise CausalVJEPA2Error(f"artifact {digest_key} does not match its explicit identity")
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise CausalVJEPA2Error(f"artifact identity differs from top-level field {key!r}")
    return identity


def validate_pca_artifact(
    pca_path: str | Path,
    *,
    train_manifest_path: str | Path | None = None,
    expected_base_droid: Mapping[str, Any] | None = None,
    expected_runtime: Mapping[str, Any] | None = None,
    expected_numerical_contract: Mapping[str, Any] | None = None,
) -> tuple[PCAWhiteningStats, dict[str, Any], str]:
    """Load a safely-pickled PCA artifact and verify all frozen identities."""
    path = Path(pca_path).expanduser().resolve()
    if not path.is_file():
        raise CausalVJEPA2Error(f"PCA artifact is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise CausalVJEPA2Error(f"cannot safely load PCA artifact: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CausalVJEPA2Error("PCA artifact payload must be a mapping")
    identity = _validate_explicit_identity(
        payload, identity_key="artifact_identity", digest_key="artifact_id"
    )
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise CausalVJEPA2Error("unsupported PCA artifact version")
    if payload.get("artifact_type") != PCA_ARTIFACT_TYPE:
        raise CausalVJEPA2Error("unexpected PCA artifact type")
    if payload.get("model_name") != VJEPA2_1_MODEL_NAME:
        raise CausalVJEPA2Error("PCA model identity mismatch")
    if payload.get("source_commit") != VJEPA2_SOURCE_COMMIT:
        raise CausalVJEPA2Error("PCA source commit mismatch")
    if payload.get("checkpoint_sha256") != VJEPA2_CHECKPOINT_SHA256:
        raise CausalVJEPA2Error("PCA checkpoint identity mismatch")
    source_archive_sha256 = str(payload.get("source_archive_sha256", ""))
    source_license = payload.get("source_license")
    if len(source_archive_sha256) != 64 or not isinstance(source_license, Mapping):
        raise CausalVJEPA2Error("PCA source/LICENSE hashes are missing")
    try:
        int(source_archive_sha256, 16)
        int(str(source_license.get("sha256", "")), 16)
    except ValueError as exc:
        raise CausalVJEPA2Error("PCA source/LICENSE hash is malformed") from exc
    if len(str(source_license.get("sha256", ""))) != 64:
        raise CausalVJEPA2Error("PCA LICENSE SHA-256 is malformed")
    if source_license.get("sha256") != VJEPA2_LICENSE_SHA256:
        raise CausalVJEPA2Error("PCA V-JEPA LICENSE identity mismatch")
    checkpoint_evidence = payload.get("checkpoint_evidence")
    if (
        not isinstance(checkpoint_evidence, Mapping)
        or checkpoint_evidence.get("sha256") != VJEPA2_CHECKPOINT_SHA256
        or not isinstance(checkpoint_evidence.get("path"), str)
    ):
        raise CausalVJEPA2Error("PCA checkpoint evidence record is malformed")
    stats = PCAWhiteningStats.from_payload(payload)
    if stats.eps != WHITENING_EPS or payload.get("whitening_eps") != WHITENING_EPS:
        raise CausalVJEPA2Error("PCA whitening epsilon must be exactly 1e-6")
    if stats.input_dim != VJEPA2_1_SOURCE_DIM or stats.output_dim != TARGET_CHANNELS:
        raise CausalVJEPA2Error("PCA dimensions must be exactly 768->48")
    if payload.get("teacher_size") != list(TEACHER_SIZE):
        raise CausalVJEPA2Error("PCA teacher geometry mismatch")
    if payload.get("pooled_token_grid") != list(POOLED_TOKEN_GRID):
        raise CausalVJEPA2Error("PCA pooled grid mismatch")
    if int(payload.get("tokens_per_clip", -1)) != TOKENS_PER_CLIP:
        raise CausalVJEPA2Error("PCA token-count contract mismatch")
    selected_ids = payload.get("selected_clip_ids")
    selected_indices = payload.get("selected_manifest_indices")
    if not isinstance(selected_ids, list) or not isinstance(selected_indices, list):
        raise CausalVJEPA2Error("PCA artifact does not record its exact selection")
    if len(selected_ids) != PCA_CLIP_COUNT or len(selected_indices) != PCA_CLIP_COUNT:
        raise CausalVJEPA2Error("PCA selection metadata is inconsistent")
    if len(set(map(str, selected_ids))) != PCA_CLIP_COUNT or len(
        set(map(int, selected_indices))
    ) != PCA_CLIP_COUNT:
        raise CausalVJEPA2Error("PCA selection contains duplicate clips or indices")
    expected_ranks = [pca_clip_rank(str(clip_id)) for clip_id in selected_ids]
    if payload.get("pca_rank_prefix") != PCA_RANK_PREFIX or payload.get(
        "selected_clip_ranks"
    ) != expected_ranks:
        raise CausalVJEPA2Error("PCA selection does not use the frozen exact rank")
    if expected_ranks != sorted(expected_ranks):
        raise CausalVJEPA2Error("PCA selection is not in increasing hash-rank order")
    if int(payload.get("sampled_token_count", -1)) != PCA_CLIP_COUNT * TOKENS_PER_CLIP:
        raise CausalVJEPA2Error("PCA artifact did not use every selected pooled token")
    if (
        int(payload.get("pca_clip_count", -1)) != PCA_CLIP_COUNT
        or payload.get("pca_training_split_only") is not True
        or int(payload.get("test_rows_used", -1)) != 0
    ):
        raise CausalVJEPA2Error("PCA artifact does not prove train-only fitting")
    if (
        payload.get("pca_algorithm") != "exact-centered-covariance-eigh"
        or payload.get("pca_covariance_dtype") != "float32"
        or payload.get("pca_tf32") is not False
    ):
        raise CausalVJEPA2Error("PCA artifact does not use the frozen exact solver")
    implementation = _validate_implementation_record(payload.get("implementation"))
    if identity.get("implementation") != implementation:
        raise CausalVJEPA2Error("PCA implementation is outside its explicit identity")
    runtime = _validate_runtime_record(payload.get("runtime"))
    numerical_contract = _validate_numerical_contract(
        payload.get("numerical_contract")
    )
    base_droid = validate_frozen_base_record(payload.get("base_droid"))
    if identity.get("runtime") != runtime or identity.get(
        "numerical_contract"
    ) != numerical_contract or identity.get("base_droid") != base_droid:
        raise CausalVJEPA2Error("PCA runtime/base identity is inconsistent")
    if expected_runtime is not None and runtime != dict(expected_runtime):
        raise CausalVJEPA2Error("PCA runtime differs from this extraction runtime")
    if expected_numerical_contract is not None and numerical_contract != dict(
        expected_numerical_contract
    ):
        raise CausalVJEPA2Error("PCA numerical contract differs from extraction")
    if expected_base_droid is not None and base_droid != dict(expected_base_droid):
        raise CausalVJEPA2Error("PCA is bound to another base-DROID artifact")
    if train_manifest_path is not None:
        train_manifest = Path(train_manifest_path).expanduser().resolve()
        if sha256_file(train_manifest) != payload.get("train_manifest_sha256"):
            raise CausalVJEPA2Error("PCA artifact is bound to another train manifest")
        from robot_wm.datasets.droid.video_latent_forcing import read_clip_manifest

        train_rows = read_clip_manifest(train_manifest, expected_split="train")
        selected = select_pca_rows(train_rows)
        if [index for index, _ in selected] != list(map(int, selected_indices)) or [
            str(row["clip_id"]) for _, row in selected
        ] != list(map(str, selected_ids)):
            raise CausalVJEPA2Error("PCA selection does not match the bound train manifest")
    return stats, dict(payload), sha256_file(path)


def validate_causal_cache(
    *,
    manifest_path: str | Path,
    cache_metadata_path: str | Path,
    expected_split: str | None = None,
    verify_target_hash: bool = True,
) -> tuple[dict[str, Any], np.memmap]:
    """Validate and memory-map a complete split target cache read-only."""
    manifest = Path(manifest_path).expanduser().resolve()
    metadata_path = Path(cache_metadata_path).expanduser().resolve()
    if not manifest.is_file() or not metadata_path.is_file():
        raise CausalVJEPA2Error("manifest and cache metadata must both exist")
    # Constructing the base dataset is deliberately left to the caller; here
    # we parse through its canonical validator by importing lazily.
    from robot_wm.datasets.droid.video_latent_forcing import read_clip_manifest

    rows = read_clip_manifest(manifest, expected_split=expected_split)
    split = str(rows[0]["split"])
    if split not in {"train", "val"}:
        raise CausalVJEPA2Error("causal semantic caches are forbidden for protected test rows")
    metadata = _read_json(metadata_path, label="cache metadata")
    cache_identity = _validate_explicit_identity(
        metadata, identity_key="artifact_identity", digest_key="cache_id"
    )
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        raise CausalVJEPA2Error("unsupported causal cache version")
    if metadata.get("artifact_type") != CACHE_ARTIFACT_TYPE:
        raise CausalVJEPA2Error("unexpected causal cache type")
    if metadata.get("complete") is not True:
        raise CausalVJEPA2Error("causal cache is incomplete")
    if metadata.get("split") != split or (expected_split and split != expected_split):
        raise CausalVJEPA2Error("causal cache split mismatch")
    if int(metadata.get("clip_count", -1)) != len(rows):
        raise CausalVJEPA2Error("causal cache clip count mismatch")
    if metadata.get("manifest_sha256") != sha256_file(manifest):
        raise CausalVJEPA2Error("causal cache is bound to another manifest")
    if metadata.get("manifest_order_sha256") != manifest_order_sha256(rows):
        raise CausalVJEPA2Error("causal cache manifest order mismatch")
    if metadata.get("source_commit") != VJEPA2_SOURCE_COMMIT:
        raise CausalVJEPA2Error("causal cache source commit mismatch")
    if metadata.get("checkpoint_sha256") != VJEPA2_CHECKPOINT_SHA256:
        raise CausalVJEPA2Error("causal cache checkpoint mismatch")
    if metadata.get("target_kind") != TARGET_KIND:
        raise CausalVJEPA2Error("causal cache target-kind mismatch")
    if metadata.get("auxiliary_target_shape") != list(TARGET_SHAPE):
        raise CausalVJEPA2Error("causal cache auxiliary target shape mismatch")
    if (
        metadata.get("protected_test_access") is not False
        or metadata.get("allowed_splits") != ["train", "val"]
        or int(metadata.get("test_rows_extracted", -1)) != 0
    ):
        raise CausalVJEPA2Error("causal cache does not prove test-set exclusion")
    if metadata.get("target_dtype") != "float16":
        raise CausalVJEPA2Error("causal cache target dtype mismatch")
    expected_shape = [len(rows), *TARGET_SHAPE]
    if metadata.get("target_shape") != expected_shape:
        raise CausalVJEPA2Error("causal cache target shape mismatch")

    implementation = _validate_implementation_record(metadata.get("implementation"))
    runtime = _validate_runtime_record(metadata.get("runtime"))
    numerical_contract = _validate_numerical_contract(
        metadata.get("numerical_contract")
    )
    base_droid = validate_frozen_base_record(
        metadata.get("base_droid"),
        expected_split=split,
        manifest_path=manifest,
    )
    if (
        cache_identity.get("implementation") != implementation
        or cache_identity.get("runtime") != runtime
        or cache_identity.get("numerical_contract") != numerical_contract
        or cache_identity.get("base_droid") != base_droid
    ):
        raise CausalVJEPA2Error("cache runtime/base identity is inconsistent")

    pca_path = _resolve_bound_file(
        metadata_path.parent, metadata.get("pca_file"), label="PCA artifact"
    )
    _, pca_payload, pca_digest = validate_pca_artifact(
        pca_path,
        expected_base_droid=base_droid,
        expected_runtime=runtime,
        expected_numerical_contract=numerical_contract,
    )
    if metadata.get("pca_sha256") != pca_digest:
        raise CausalVJEPA2Error("causal cache PCA hash mismatch")
    if metadata.get("train_manifest_sha256") != pca_payload.get(
        "train_manifest_sha256"
    ):
        raise CausalVJEPA2Error("cache and PCA train-manifest identities differ")
    if metadata.get("source_archive_sha256") != pca_payload.get(
        "source_archive_sha256"
    ) or metadata.get("source_license") != pca_payload.get("source_license"):
        raise CausalVJEPA2Error("cache and PCA source/LICENSE identities differ")
    if metadata.get("checkpoint_evidence") != pca_payload.get("checkpoint_evidence"):
        raise CausalVJEPA2Error("cache and PCA checkpoint evidence differs")
    if implementation != pca_payload.get("implementation"):
        raise CausalVJEPA2Error("cache and PCA implementation identities differ")

    target_path = _resolve_bound_file(
        metadata_path.parent, metadata.get("target_file"), label="target memmap"
    )
    if verify_target_hash and sha256_file(target_path) != metadata.get("target_sha256"):
        raise CausalVJEPA2Error("causal target content hash mismatch")
    try:
        target = np.load(target_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise CausalVJEPA2Error(f"cannot memory-map causal targets: {exc}") from exc
    if tuple(target.shape) != tuple(expected_shape) or target.dtype != np.float16:
        raise CausalVJEPA2Error("causal target array shape/dtype mismatch")
    if target.flags.writeable:
        raise CausalVJEPA2Error("causal target memmap unexpectedly permits writes")
    return metadata, target


class DroidCausalVJEPA2Dataset(Dataset):
    """Immutable DROID samples augmented with offline causal V-JEPA targets."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        cache_metadata_path: str | Path,
        *,
        allow_protected_test: bool = False,
        protected_test_purpose: str | None = None,
        verify_target_hash: bool = True,
    ) -> None:
        self.base = DroidVideoLatentForcingDataset(
            manifest_path,
            data_root,
            allow_protected_test=allow_protected_test,
            protected_test_purpose=protected_test_purpose,
        )
        self.cache_metadata, self._targets = validate_causal_cache(
            manifest_path=manifest_path,
            cache_metadata_path=cache_metadata_path,
            expected_split=self.base.split,
            verify_target_hash=verify_target_hash,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        # Copy one row out of the immutable mapping.  The returned tensor may
        # be transformed by a trainer without ever mutating the cache bytes.
        auxiliary = torch.from_numpy(np.array(self._targets[index], copy=True))
        if auxiliary.dtype != torch.float16 or tuple(auxiliary.shape) != TARGET_SHAPE:
            raise CausalVJEPA2Error("cached auxiliary target row is malformed")
        if not torch.isfinite(auxiliary).all():
            raise CausalVJEPA2Error("cached auxiliary target row is non-finite")
        sample["auxiliary_target"] = auxiliary
        sample["auxiliary_cache_id"] = str(self.cache_metadata["cache_id"])
        return sample


class CausalVJEPA2DroidDataset(DroidCausalVJEPA2Dataset):
    """Trainer-facing dataset with a deterministic semantic-cache layout.

    ``semantic_cache_root`` contains one directory per immutable manifest
    split.  This class always resolves exactly
    ``<semantic_cache_root>/<split>/metadata.json``; it never searches or
    guesses among cache artifacts.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        semantic_cache_root: str | Path,
        *,
        allow_protected_test: bool = False,
        protected_test_purpose: str | None = None,
        verify_target_hash: bool = True,
    ) -> None:
        from robot_wm.datasets.droid.video_latent_forcing import read_clip_manifest

        rows = read_clip_manifest(manifest_path)
        split = str(rows[0]["split"])
        metadata_path = (
            Path(semantic_cache_root).expanduser().resolve() / split / "metadata.json"
        )
        super().__init__(
            manifest_path,
            data_root,
            metadata_path,
            allow_protected_test=allow_protected_test,
            protected_test_purpose=protected_test_purpose,
            verify_target_hash=verify_target_hash,
        )
