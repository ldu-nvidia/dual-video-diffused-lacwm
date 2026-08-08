"""Shared immutable contracts for the Phase-0 CSIP feasibility screen.

This module contains no launcher and never opens a protected-test path.  It is
shared by registration, latent extraction, fitting, sealed validation, and
analysis so every stage checks the same content identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_BASE_COMMIT = "e4fda453ed4b92173848ea12b5538749069c9032"
SCHEMA_VERSION = 1
REGISTRATION_KIND = "csip_phase0_registration"
CACHE_KIND = "csip_wan_latent_cache"
CHECKPOINT_KIND = "csip_phase0_fixed_checkpoint"
EVALUATION_KIND = "csip_phase0_sealed_validation"
ANALYSIS_KIND = "csip_phase0_bootstrap_analysis"

EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
EXPECTED_WORLD_SIZE = 8
EXPECTED_TRAIN_CLIPS = 512
EXPECTED_VALIDATION_CLIPS = 64
EXPECTED_VALIDATION_PAIR_BLOCKS = EXPECTED_VALIDATION_CLIPS // 2
EXPECTED_UPDATES = 400
EXPECTED_BATCH_SIZE = 64
EXPECTED_SEED = 1234
EXPECTED_VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260808
CAUSAL_TOLERANCE = 1e-4

TRAIN_MANIFEST_SHA256 = (
    "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
)
TRAIN_METADATA_SHA256 = (
    "fa22a213f352ffb8cc0b4dc0d35138b35aac349c03f362c597c621fa3473da43"
)
TRAIN_RGB_SHA256 = "b5bdde4461c75bc88653c38b737021fcbd69b0b22f4c87bc8e8097c3494b64ee"
TRAIN_ACTIONS_SHA256 = (
    "f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89"
)
VALIDATION_MANIFEST_SHA256 = (
    "8cb39c1f056855e28855c0b944c715d084b709e1421f4efeed3710e7099348c4"
)
VALIDATION_METADATA_SHA256 = (
    "9bb873cf373aea4aa0e28319365a7e0492e63e110afb28dd8d358d5ee1cbb3f6"
)
VALIDATION_RGB_SHA256 = (
    "ed82fc0f580baa90c4dc39c1608f97ad7092a4ddcb59c3612eed31711afda404"
)
VALIDATION_ACTIONS_SHA256 = (
    "552a5cf0af156868d2866dfacabe102fc6b5cd24580bb377953e35a14625306a"
)

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")


class CSIPContractError(RuntimeError):
    """A registered artifact, data, runtime, or split boundary changed."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise CSIPContractError(f"{label} must be a regular absolute file")
    return value.resolve(strict=True)


def canonical_directory(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or not value.is_dir() or value.is_symlink():
        raise CSIPContractError(f"{label} must be a canonical absolute directory")
    resolved = value.resolve(strict=True)
    if value != resolved:
        raise CSIPContractError(f"{label} must not traverse a symlink")
    return resolved


def file_record(
    path: str | Path, label: str, *, hash_content: bool = True
) -> dict[str, Any]:
    value = regular_file(path, label)
    record: dict[str, Any] = {"path": str(value), "bytes": value.stat().st_size}
    if hash_content:
        record["sha256"] = sha256_file(value)
    return record


def verify_file_record(record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record.get("path"), str):
        raise CSIPContractError(f"{label} file record is malformed")
    path = regular_file(str(record["path"]), label)
    if (
        record.get("bytes") != path.stat().st_size
        or not isinstance(record.get("sha256"), str)
        or SHA256_RE.fullmatch(str(record["sha256"])) is None
        or sha256_file(path) != record["sha256"]
    ):
        raise CSIPContractError(f"{label} changed after registration")
    return path


def read_json(path: str | Path, label: str) -> dict[str, Any]:
    value_path = regular_file(path, label)
    try:
        value = json.loads(value_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CSIPContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CSIPContractError(f"{label} must contain one object")
    return value


def with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return {**unsigned, "identity_sha256": sha256_json(unsigned)}


def verify_identity(payload: Mapping[str, Any], label: str) -> None:
    observed = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if (
        not isinstance(observed, str)
        or SHA256_RE.fullmatch(observed) is None
        or observed != sha256_json(unsigned)
    ):
        raise CSIPContractError(f"{label} identity is invalid")


def exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    value = Path(path).expanduser()
    if not value.is_absolute() or not value.name or value.name in {".", ".."}:
        raise CSIPContractError("output JSON must be an absolute named path")
    value.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    canonical = value.parent.resolve(strict=True) / value.name
    if canonical != value or value.exists() or value.is_symlink():
        raise CSIPContractError(f"refusing to overwrite or redirect {value}")
    try:
        descriptor = os.open(value, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CSIPContractError(f"refusing to overwrite {value}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CSIPContractError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def clean_source(repo: str | Path, expected_commit: str) -> dict[str, Any]:
    value = canonical_directory(repo, "tool repository")
    if value != REPO_ROOT or COMMIT_RE.fullmatch(expected_commit) is None:
        raise CSIPContractError("repository or expected commit differs")
    if git_output(value, "rev-parse", "HEAD") != expected_commit:
        raise CSIPContractError("tool HEAD differs from the expected commit")
    git_output(
        value,
        "merge-base",
        "--is-ancestor",
        INTEGRATION_BASE_COMMIT,
        expected_commit,
    )
    if git_output(value, "status", "--porcelain", "--untracked-files=all"):
        raise CSIPContractError("execution requires a clean source checkout")
    return {
        "path": str(value),
        "git_commit": expected_commit,
        "integration_base_commit": INTEGRATION_BASE_COMMIT,
        "clean": True,
    }


def clean_pinned_checkout(
    path: str | Path, *, label: str, expected_commit: str
) -> dict[str, str]:
    """Bind a non-symlinked Git checkout to its exact clean commit and tree."""

    value = canonical_directory(path, label)
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise CSIPContractError(f"{label} expected commit is malformed")
    if git_output(value, "rev-parse", "HEAD") != expected_commit:
        raise CSIPContractError(f"{label} commit changed")
    if git_output(value, "status", "--porcelain", "--untracked-files=all"):
        raise CSIPContractError(f"{label} checkout is dirty")
    tree = git_output(value, "rev-parse", "HEAD^{tree}")
    if COMMIT_RE.fullmatch(tree) is None:
        raise CSIPContractError(f"{label} tree identity is malformed")
    return {"path": str(value), "git_commit": expected_commit, "git_tree": tree}


def executable_record(path: str | Path) -> dict[str, Any]:
    launcher = Path(path).expanduser()
    if (
        not launcher.is_absolute()
        or not launcher.is_file()
        or not os.access(launcher, os.X_OK)
    ):
        raise CSIPContractError(
            "runtime Python must be an absolute executable file or symlink"
        )
    value = launcher.resolve(strict=True)
    if not value.is_file():
        raise CSIPContractError("runtime Python symlink target is not a file")
    completed = subprocess.run(
        [
            # Execute through the registered virtual-environment launcher.
            # Calling the resolved target directly bypasses pyvenv.cfg and can
            # hide the environment's installed packages.  We still hash and
            # bind ``value`` below, so both launcher and executable are sealed.
            str(launcher),
            "-c",
            (
                "import json,platform,sys;from importlib.metadata import version;"
                "print(json.dumps({'version':sys.version,'platform':platform.platform(),"
                "'packages':{name:version(name) for name in "
                "('torch','numpy','wandb','omegaconf')}}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CSIPContractError("runtime Python identity probe failed")
    try:
        runtime = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CSIPContractError("runtime Python identity is malformed") from exc
    return {
        **file_record(value, "runtime Python"),
        **runtime,
        "launcher_path": str(launcher),
    }


def manifest_rows(path: str | Path, *, split: str, count: int) -> list[dict[str, Any]]:
    manifest = regular_file(path, f"{split} manifest")
    rows: list[dict[str, Any]] = []
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("manifest row is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise CSIPContractError(f"{split} manifest is invalid") from exc
    if len(rows) != count:
        raise CSIPContractError(f"{split} manifest must contain exactly {count} rows")
    clip_ids: set[str] = set()
    episodes: set[str] = set()
    for index, row in enumerate(rows):
        clip_id = row.get("clip_id")
        episode = row.get("episode_dir")
        if (
            row.get("split") != split
            or row.get("auxiliary_index") != index
            or row.get("sample_size") != 13
            or row.get("chunk_size") != 5
            or not isinstance(clip_id, str)
            or SHA256_RE.fullmatch(clip_id) is None
            or not isinstance(episode, str)
            or not episode
            or clip_id in clip_ids
            or episode in episodes
        ):
            raise CSIPContractError(f"{split} manifest row {index} violates identity")
        clip_ids.add(clip_id)
        episodes.add(episode)
    return rows


def _resolve_array(metadata_path: Path, metadata: Mapping[str, Any], name: str) -> Path:
    raw = metadata.get(f"{name}_file")
    if not isinstance(raw, str) or not raw:
        raise CSIPContractError(f"cache metadata lacks {name}_file")
    path = Path(raw)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return regular_file(path, f"{name} array")


def source_split_record(
    manifest_path: str | Path,
    metadata_path: str | Path,
    *,
    split: str,
    rehash_arrays: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split not in {"train", "val"}:
        raise CSIPContractError("CSIP accepts only train and val source splits")
    count = EXPECTED_TRAIN_CLIPS if split == "train" else EXPECTED_VALIDATION_CLIPS
    expected = {
        "train": (
            TRAIN_MANIFEST_SHA256,
            TRAIN_METADATA_SHA256,
            TRAIN_RGB_SHA256,
            TRAIN_ACTIONS_SHA256,
        ),
        "val": (
            VALIDATION_MANIFEST_SHA256,
            VALIDATION_METADATA_SHA256,
            VALIDATION_RGB_SHA256,
            VALIDATION_ACTIONS_SHA256,
        ),
    }[split]
    manifest = file_record(manifest_path, f"{split} manifest")
    metadata_record = file_record(metadata_path, f"{split} cache metadata")
    if manifest["sha256"] != expected[0] or metadata_record["sha256"] != expected[1]:
        raise CSIPContractError(f"{split} manifest/cache metadata digest differs")
    rows = manifest_rows(manifest["path"], split=split, count=count)
    metadata = read_json(metadata_record["path"], f"{split} cache metadata")
    if (
        metadata.get("complete") is not True
        or metadata.get("split") != split
        or metadata.get("clip_count") != count
        or metadata.get("clip_manifest_sha256") != expected[0]
        or metadata.get("rgb_shape") != [count, 13, 3, 180, 960]
        or metadata.get("rgb_dtype") != "float16"
        or metadata.get("actions_shape") != [count, 13, 5, 23]
        or metadata.get("actions_dtype") != "float32"
        or metadata.get("rgb_sha256") != expected[2]
        or metadata.get("actions_sha256") != expected[3]
    ):
        raise CSIPContractError(f"{split} immutable RGB/action cache contract differs")
    arrays: dict[str, Any] = {}
    for name, digest in (("rgb", expected[2]), ("actions", expected[3])):
        path = _resolve_array(Path(metadata_record["path"]), metadata, name)
        record = file_record(path, f"{split} {name}", hash_content=rehash_arrays)
        if rehash_arrays and record["sha256"] != digest:
            raise CSIPContractError(f"{split} {name} content digest differs")
        record["sha256"] = digest
        record["full_sha256_verified"] = bool(rehash_arrays)
        arrays[name] = record
    return (
        {
            "split": split,
            "clips": count,
            "manifest": manifest,
            "cache_metadata": metadata_record,
            "arrays": arrays,
            "clip_ids_sha256": sha256_json([row["clip_id"] for row in rows]),
            "episode_dirs_sha256": sha256_json([row["episode_dir"] for row in rows]),
            "one_clip_per_episode": True,
            "protected_test": False,
        },
        rows,
    )


def assert_split_isolation(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_clips = {str(row["clip_id"]) for row in train_rows}
    val_clips = {str(row["clip_id"]) for row in validation_rows}
    train_episodes = {str(row["episode_dir"]) for row in train_rows}
    val_episodes = {str(row["episode_dir"]) for row in validation_rows}
    if train_clips & val_clips or train_episodes & val_episodes:
        raise CSIPContractError("train and validation overlap by clip or episode")
    return {
        "clip_overlap": 0,
        "episode_overlap": 0,
        "one_clip_per_episode": True,
        "validation_sealed_until_fixed_checkpoint": True,
        "protected_test_clips_read": 0,
    }


def runtime_record(
    python: str | Path, wan_dir: str | Path, videox_home: str | Path
) -> dict[str, Any]:
    python_record = executable_record(python)
    if Path(sys.executable).resolve(strict=True) != Path(python_record["path"]):
        raise CSIPContractError(
            "registration must run under the same Python it binds with --python"
        )
    wan = canonical_directory(wan_dir, "Wan model directory")
    videox_record = clean_pinned_checkout(
        videox_home,
        label="VideoX checkout",
        expected_commit=EXPECTED_VIDEOX_COMMIT,
    )
    videox = Path(videox_record["path"])
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise CSIPContractError(
            "OmegaConf is required to bind the Wan runtime"
        ) from exc
    config_path = videox / "config" / "wan2.1" / "wan_civitai.yaml"
    config_record = file_record(config_path, "Wan VAE config")
    config = OmegaConf.load(config_path)
    vae_subpath = str(config.get("vae_kwargs", {}).get("vae_subpath", "Wan2.1_VAE.pth"))
    weights = (wan / vae_subpath).resolve(strict=True)
    if wan not in weights.parents:
        raise CSIPContractError("Wan VAE weights escape the registered model root")
    return {
        "python": python_record,
        "wan_dir": str(wan),
        "videox_home": str(videox),
        "videox_git_commit": videox_record["git_commit"],
        "videox_git_tree": videox_record["git_tree"],
        "wan_config": config_record,
        "wan_vae": file_record(weights, "Wan VAE weights"),
        "world_size": EXPECTED_WORLD_SIZE,
    }


def artifact_root(path: str | Path, *, must_be_fresh: bool) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or not value.name or value.name in {".", ".."}:
        raise CSIPContractError("artifact root must be an absolute named path")
    parent = value.parent.resolve(strict=True)
    canonical = parent / value.name
    allowed = (Path("/mnt/data1"), Path("/mnt/data2"), Path("/lustre"))
    if canonical != value or not any(
        root == canonical or root in canonical.parents for root in allowed
    ):
        raise CSIPContractError(
            "large artifacts must live under /mnt/data1, /mnt/data2, or /lustre"
        )
    if must_be_fresh and (canonical.exists() or canonical.is_symlink()):
        raise CSIPContractError("prospective artifact root must be fresh")
    return canonical


def validate_registration(
    path: str | Path,
    *,
    require_train_cache: bool = False,
    require_validation_cache: bool = False,
    open_validation: bool = True,
) -> dict[str, Any]:
    registration_path = regular_file(path, "CSIP registration")
    payload = read_json(registration_path, "CSIP registration")
    verify_identity(payload, "CSIP registration")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != REGISTRATION_KIND
        or payload.get("status") != "registered_before_latent_extraction_or_training"
        or payload.get("protected_test_clips_read") != 0
    ):
        raise CSIPContractError("CSIP registration contract differs")
    source = payload.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("integration_base_commit") != INTEGRATION_BASE_COMMIT
    ):
        raise CSIPContractError("registration source binding is absent")
    clean_source(str(source.get("path", "")), str(source.get("git_commit", "")))
    splits = ("train", "validation") if open_validation else ("train",)
    for split in splits:
        record = payload.get("datasets", {}).get(split)
        if not isinstance(record, Mapping):
            raise CSIPContractError(f"registration lacks {split} data")
        verify_file_record(record["manifest"], f"{split} manifest")
        verify_file_record(record["cache_metadata"], f"{split} cache metadata")
        for name in ("rgb", "actions"):
            verify_file_record(record["arrays"][name], f"{split} {name}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CSIPContractError("registration lacks runtime binding")
    for key in ("python", "wan_config", "wan_vae"):
        verify_file_record(runtime[key], f"runtime {key}")
    launcher = Path(str(runtime["python"].get("launcher_path", "")))
    if (
        not launcher.is_absolute()
        or not launcher.is_file()
        or launcher.resolve(strict=True) != Path(runtime["python"]["path"])
    ):
        raise CSIPContractError("registered Python launcher target changed")
    videox = clean_pinned_checkout(
        str(runtime.get("videox_home", "")),
        label="registered VideoX checkout",
        expected_commit=EXPECTED_VIDEOX_COMMIT,
    )
    if videox["git_commit"] != runtime.get("videox_git_commit") or videox[
        "git_tree"
    ] != runtime.get("videox_git_tree"):
        raise CSIPContractError("registered VideoX checkout identity changed")
    wan = canonical_directory(str(runtime.get("wan_dir", "")), "registered Wan root")
    if (
        str(wan) != runtime.get("wan_dir")
        or runtime.get("world_size") != EXPECTED_WORLD_SIZE
    ):
        raise CSIPContractError("registered Wan/runtime geometry changed")
    config_path = Path(str(runtime["wan_config"].get("path", "")))
    weights_path = Path(str(runtime["wan_vae"].get("path", "")))
    videox_path = Path(videox["path"])
    if videox_path not in config_path.parents or wan not in weights_path.parents:
        raise CSIPContractError("registered Wan config/weights escaped their roots")
    required = []
    if require_train_cache:
        required.append("train")
    if require_validation_cache:
        if not open_validation:
            raise CSIPContractError(
                "validation cache cannot be required while validation is sealed"
            )
        required.append("validation")
    for split in required:
        cache_path = Path(payload["planned_paths"][f"{split}_latent_metadata"])
        if not cache_path.is_file():
            raise CSIPContractError(f"registered {split} latent cache is absent")
    return payload


def validate_latent_cache(
    metadata_path: str | Path,
    *,
    registration: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    path = regular_file(metadata_path, f"{split} latent metadata")
    payload = read_json(path, f"{split} latent metadata")
    verify_identity(payload, f"{split} latent metadata")
    expected_split = "train" if split == "train" else "val"
    expected_count = (
        EXPECTED_TRAIN_CLIPS if split == "train" else EXPECTED_VALIDATION_CLIPS
    )
    source_key = "train" if split == "train" else "validation"
    source = registration["datasets"][source_key]
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != CACHE_KIND
        or payload.get("status") != "complete"
        or payload.get("split") != expected_split
        or payload.get("clips") != expected_count
        or payload.get("world_size") != EXPECTED_WORLD_SIZE
        or payload.get("full_latent_shape") != [expected_count, 16, 4, 24, 120]
        or payload.get("history_latent_shape") != [expected_count, 16, 2, 24, 120]
        or payload.get("dtype") != "float16"
        or payload.get("source_manifest_sha256") != source["manifest"]["sha256"]
        or payload.get("source_rgb_sha256") != source["arrays"]["rgb"]["sha256"]
        or payload.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or payload.get("causal_history_tolerance") != CAUSAL_TOLERANCE
        or float(payload.get("causal_history_max_abs_observed", float("inf")))
        > CAUSAL_TOLERANCE
        or payload.get("protected_test_clips_read") != 0
    ):
        raise CSIPContractError(f"{split} latent cache contract differs")
    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_WORLD_SIZE:
        raise CSIPContractError(f"{split} latent shard inventory differs")
    indexes: list[int] = []
    for rank, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("rank") != rank:
            raise CSIPContractError(f"{split} latent shard rank differs")
        for name in ("indexes", "full_latents", "history_latents"):
            verify_file_record(shard[name], f"{split} rank {rank} {name}")
        import numpy as np

        shard_indexes = np.load(shard["indexes"]["path"], allow_pickle=False)
        indexes.extend(int(value) for value in shard_indexes.tolist())
    if (
        sorted(indexes) != list(range(expected_count))
        or len(set(indexes)) != expected_count
    ):
        raise CSIPContractError(f"{split} latent shard indexes are incomplete")
    return payload


def load_latent_cache_arrays(
    metadata: Mapping[str, Any],
) -> tuple["Any", "Any"]:
    """Materialize dense full/history arrays from the verified rank shards."""

    import numpy as np

    count = int(metadata["clips"])
    full = np.empty((count, 16, 4, 24, 120), dtype=np.float16)
    history = np.empty((count, 16, 2, 24, 120), dtype=np.float16)
    seen: set[int] = set()
    for shard in metadata["shards"]:
        indexes = np.load(shard["indexes"]["path"], allow_pickle=False)
        shard_full = np.load(
            shard["full_latents"]["path"], mmap_mode="r", allow_pickle=False
        )
        shard_history = np.load(
            shard["history_latents"]["path"], mmap_mode="r", allow_pickle=False
        )
        expected_rows = len(indexes)
        if (
            indexes.dtype != np.int64
            or tuple(shard_full.shape) != (expected_rows, 16, 4, 24, 120)
            or tuple(shard_history.shape) != (expected_rows, 16, 2, 24, 120)
            or shard_full.dtype != np.float16
            or shard_history.dtype != np.float16
        ):
            raise CSIPContractError("latent shard array geometry differs")
        for local, raw_index in enumerate(indexes.tolist()):
            index = int(raw_index)
            if index in seen or index < 0 or index >= count:
                raise CSIPContractError(
                    "latent shard contains a duplicate/out-of-range row"
                )
            full[index] = shard_full[local]
            history[index] = shard_history[local]
            seen.add(index)
    if seen != set(range(count)):
        raise CSIPContractError("latent cache does not cover the dense population")
    return full, history


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    train_latent_cache_identity_sha256: str,
) -> None:
    """Validate the fixed endpoint and its fit448/cal64 contamination guards."""

    import torch

    from robot_wm.modeling.dual_diffusion.causal_spectral_probe import (
        ACTION_DESCRIPTOR_DIM,
        ACTION_TARGET_DIM,
        SPECTRAL_FEATURE_DIM,
        phase0_partition_indexes,
    )

    fit = payload.get("fit_indexes")
    calibration = payload.get("calibration_indexes")
    pca = payload.get("action_pca")
    model_states = payload.get("model_states")
    calibration_metrics = payload.get("calibration_metrics")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != CHECKPOINT_KIND
        or payload.get("registration_identity_sha256")
        != registration.get("identity_sha256")
        or payload.get("train_latent_cache_identity_sha256")
        != train_latent_cache_identity_sha256
        or payload.get("completed_updates") != EXPECTED_UPDATES
        or payload.get("selected_update") != EXPECTED_UPDATES
        or payload.get("selection_rule") != "fixed_final_update_not_metric_selected"
        or payload.get("seed") != EXPECTED_SEED
        or payload.get("model_hidden_dim") != 256
        or payload.get("spectral_feature_dim") != SPECTRAL_FEATURE_DIM
        or payload.get("action_target_dim") != ACTION_TARGET_DIM
        or payload.get("feature_variants") != ["full", "angle_neutral"]
        or payload.get("paired_initialization") != "identical_state_before_update_1"
        or payload.get("wandb_run_id") != registration.get("wandb", {}).get("run_id")
        or payload.get("validation_clips_read") != 0
        or payload.get("protected_test_clips_read") != 0
        or not isinstance(fit, torch.Tensor)
        or fit.dtype != torch.long
        or fit.tolist() != list(phase0_partition_indexes(512, "fit"))
        or not isinstance(calibration, torch.Tensor)
        or calibration.dtype != torch.long
        or calibration.tolist() != list(phase0_partition_indexes(512, "calibration"))
        or not isinstance(pca, Mapping)
        or set(pca) != {"mean", "components", "score_scale"}
        or not isinstance(pca.get("mean"), torch.Tensor)
        or tuple(pca["mean"].shape) != (ACTION_DESCRIPTOR_DIM,)
        or not isinstance(pca.get("components"), torch.Tensor)
        or tuple(pca["components"].shape) != (ACTION_TARGET_DIM, ACTION_DESCRIPTOR_DIM)
        or not isinstance(pca.get("score_scale"), torch.Tensor)
        or tuple(pca["score_scale"].shape) != (ACTION_TARGET_DIM,)
        or not all(torch.isfinite(value).all().item() for value in pca.values())
        or not isinstance(model_states, Mapping)
        or set(model_states) != {"full", "angle_neutral"}
        or any(
            not isinstance(state, Mapping)
            or not state
            or any(
                not isinstance(value, torch.Tensor)
                or not torch.isfinite(value).all().item()
                for value in state.values()
            )
            for state in model_states.values()
        )
        or not isinstance(calibration_metrics, Mapping)
        or set(calibration_metrics) != {"0", "100", "200", "300", "400"}
        or any(
            not isinstance(update_metrics, Mapping)
            or set(update_metrics) != {"full", "angle_neutral"}
            or any(
                not isinstance(metrics, Mapping)
                or set(metrics) != {"mse", "cosine"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not torch.isfinite(torch.tensor(float(value))).item()
                    for value in metrics.values()
                )
                for metrics in update_metrics.values()
            )
            for update_metrics in calibration_metrics.values()
        )
    ):
        raise CSIPContractError("CSIP fixed checkpoint payload differs")


def registration_file_record(path: str | Path) -> dict[str, Any]:
    payload = read_json(path, "CSIP registration")
    verify_identity(payload, "CSIP registration")
    return {
        **file_record(path, "CSIP registration"),
        "identity_sha256": payload["identity_sha256"],
    }


def current_python_matches(record: Mapping[str, Any]) -> None:
    from importlib.metadata import version

    current = file_record(Path(sys.executable).resolve(), "current Python")
    if current["sha256"] != record.get("sha256"):
        raise CSIPContractError("current Python differs from the registered runtime")
    packages = {
        name: version(name) for name in ("torch", "numpy", "wandb", "omegaconf")
    }
    if packages != record.get("packages"):
        raise CSIPContractError(
            "current Python package runtime changed after registration"
        )
