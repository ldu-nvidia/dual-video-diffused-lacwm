#!/usr/bin/env python3
"""Read-only host, asset, dataset, and GPU preflight for lacwm_dit training.

This program never downloads data, creates environments, allocates CUDA tensors, or
stops processes.  It exits non-zero when a required check fails.  The launch wrappers
run it immediately before ``exec`` so an occupied selected GPU cannot be used accidentally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Keep this CLI directly executable (``python tools/training_preflight.py``)
# without relying on a caller-provided PYTHONPATH.  The guarded launchers add
# the repository root already, but standalone incident/preflight use should be
# equally deterministic.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_root_policy import (
    RunRootPolicyError,
    canonical_allowed_run_root,
    configured_allowed_run_roots,
)


PROJECT_ROOT = REPO_ROOT / "projects" / "latent_action_models"
TRAIN_MARKERS = (
    "torchrun",
    "torch.distributed.run",
    "train.py",
    "accelerate launch",
    "deepspeed",
)
PYTHON_MODULES = (
    "torch",
    "torchvision",
    "hydra",
    "omegaconf",
    "torchdata",
    "diffusers",
    "accelerate",
    "peft",
    "safetensors",
    "huggingface_hub",
    "transformers",
    "ftfy",
    "sentencepiece",
    "wandb",
    "einops",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "decord",
    "av",
    "cv2",
    "h5py",
    "kornia",
    "imageio",
    "imageio_ffmpeg",
    "colorlog",
)
TRAINING_IMPORT_PROBES = (
    "robot_wm.modeling.tokenizers.rgb.wan_vae",
    "robot_wm.modeling.networks.wan_forward_model",
    "lam.latent_action_dit_model",
    "lam.explicit_action_dit_model",
    "videox_fun.models.wan_vae",
    "videox_fun.models.wan_transformer3d",
)
DEFAULT_MIN_GPU_MEMORY_MIB = 78_000
DATASET_NAMES = ("droid", "egodex", "agibot", "abc")
STRICT_DATA_POLICY = "strict"
FAST_DATA_POLICY = "files_only_user_waived_v1"
DATA_VALIDATION_POLICIES = (STRICT_DATA_POLICY, FAST_DATA_POLICY)
FAST_SOURCE_ORDER = ("Droid", "EgoDex", "Agibot", "ABC")
FAST_SOURCE_LENGTHS = (10_000, 10_000, 5_671, 10_000)
FAST_WAIVER_KIND = "lacwm_user_authorized_fast_mixed_overlay"
FAST_AUTHORIZATION_KIND = "lacwm_fast_training_authorization"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


class Results:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        self.checks.append(Check(name=name, ok=bool(ok), detail=str(detail), required=required))

    @property
    def passed(self) -> bool:
        return all(item.ok or not item.required for item in self.checks)


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_regular_file(path: Path, label: str) -> Path:
    """Reject a symlink at the supplied path before canonicalizing it."""

    candidate = path.expanduser()
    try:
        info = candidate.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"{label} is not a nonempty non-symlink regular file: {candidate}")
    return candidate.resolve(strict=True)


def read_json_object(path: Path, label: str) -> dict[str, object]:
    path = canonical_regular_file(path, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def parse_aware_timestamp(value: object, label: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include an explicit timezone")
    return parsed


def parse_gpu_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid GPU list {raw!r}") from exc
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("GPU indices must be unique non-negative integers")
    return values


def supported_gpu_memory_mib(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "minimum GPU memory must be an integer number of MiB"
        ) from exc
    if value < DEFAULT_MIN_GPU_MEMORY_MIB:
        raise argparse.ArgumentTypeError(
            f"minimum GPU memory must be at least {DEFAULT_MIN_GPU_MEMORY_MIB} MiB"
        )
    return value


def normalized_dataset_names(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError("dataset names must not contain duplicates")
    selected = set(values)
    if not selected or not selected <= set(DATASET_NAMES):
        raise ValueError(
            "datasets must be a non-empty subset of " + ", ".join(DATASET_NAMES)
        )
    return tuple(name for name in DATASET_NAMES if name in selected)


def existing_writable_ancestor(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.exists() and current.is_dir() and os.access(current, os.W_OK | os.X_OK):
        return current
    return None


def count_nonempty_lines(path: Path, *, skip_header: bool = False) -> tuple[int, list[str]]:
    count = 0
    examples: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            if skip_header and line_number == 0:
                continue
            count += 1
            if len(examples) < 2:
                examples.append(stripped)
    return count, examples


def check_file(results: Results, name: str, path: Path, minimum_bytes: int = 1) -> None:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    results.add(name, exists and size >= minimum_bytes, f"{path} ({size:,} bytes)")


def check_repo(results: Results, profile: str) -> None:
    required_files = (
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "configs" / "train.yaml",
        PROJECT_ROOT
        / "configs"
        / "experiments_0908"
        / "ravenhuang"
        / "wan-dit"
        / "wan_dit_smoke.yaml",
        PROJECT_ROOT
        / "configs"
        / "experiments_0908"
        / "ravenhuang"
        / "wan-dit"
        / "wan_dit_abc_agibot_droid_egodex.yaml",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    results.add("repository layout", not missing, "ok" if not missing else f"missing: {missing}")

    revision = run_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    results.add("git revision", revision.returncode == 0, revision.stdout.strip() or revision.stderr.strip())
    status = run_command(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else True
    results.add(
        "clean worktree",
        not dirty,
        "clean" if not dirty else status.stdout.strip().replace("\n", "; "),
        required=profile == "full",
    )


def check_output_root(results: Results, run_root: Path, profile: str) -> None:
    try:
        allowed_roots = configured_allowed_run_roots()
        resolved_run_root = canonical_allowed_run_root(run_root, allowed_roots)
        results.add(
            "run root policy",
            True,
            f"{resolved_run_root} is under an allowed root; "
            f"allowed={':'.join(str(path) for path in allowed_roots)}",
        )
    except RunRootPolicyError as exc:
        results.add("run root policy", False, str(exc))
    ancestor = existing_writable_ancestor(run_root)
    results.add("run root writable", ancestor is not None, f"writable ancestor: {ancestor}")
    if ancestor is None:
        return
    usage = shutil.disk_usage(ancestor)
    free_gib = usage.free / 2**30
    minimum = 200.0 if profile == "full" else 20.0
    results.add("run volume free space", free_gib >= minimum, f"{free_gib:.1f} GiB free; require {minimum:.0f} GiB")


def check_host(results: Results, profile: str) -> None:
    cpu_count = os.cpu_count() or 0
    minimum_cpus = 64 if profile == "full" else 4
    results.add("host CPU count", cpu_count >= minimum_cpus, f"{cpu_count} logical CPUs; require {minimum_cpus}")
    mem_total_kib = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kib = int(line.split()[1])
                break
    except OSError:
        pass
    ram_gib = mem_total_kib / 2**20
    minimum_ram = 256.0 if profile == "full" else 32.0
    results.add("host RAM", ram_gib >= minimum_ram, f"{ram_gib:.1f} GiB; require {minimum_ram:.0f} GiB")
    results.add("flock available", shutil.which("flock") is not None, shutil.which("flock") or "not found")
    results.add("timeout available", shutil.which("timeout") is not None, shutil.which("timeout") or "not found")
    results.add("sha256sum available", shutil.which("sha256sum") is not None, shutil.which("sha256sum") or "not found")
    results.add("stat available", shutil.which("stat") is not None, shutil.which("stat") or "not found")


def check_python(results: Results, python_bin: Path, video_home: Path) -> None:
    results.add("Python executable", python_bin.is_file() and os.access(python_bin, os.X_OK), str(python_bin))
    if not python_bin.is_file():
        return
    probe = "\n".join(
        (
            "import importlib, json, sys",
            f"modules = {list(PYTHON_MODULES + TRAINING_IMPORT_PROBES)!r}",
            "errors = {}",
            "for module in modules:",
            "    try:",
            "        importlib.import_module(module)",
            "    except Exception as exc:",
            "        errors[module] = f'{type(exc).__name__}: {exc}'",
            "print('__LACWM_IMPORT_PROBE__' + json.dumps({'version': sys.version.split()[0], 'errors': errors}))",
        )
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    additions = [
        str(REPO_ROOT / "tools" / "env" / "videox_shim"),
        str(video_home),
        str(PROJECT_ROOT),
        str(REPO_ROOT),
    ]
    env["PYTHONPATH"] = os.pathsep.join(additions + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    completed = run_command([str(python_bin), "-c", probe], env=env, timeout=120)
    detail = completed.stdout.strip() or completed.stderr.strip()
    ok = False
    if completed.returncode == 0:
        try:
            probe_line = next(
                line for line in reversed(completed.stdout.splitlines()) if line.startswith("__LACWM_IMPORT_PROBE__")
            )
            payload = json.loads(probe_line.removeprefix("__LACWM_IMPORT_PROBE__"))
            ok = not payload["errors"]
            detail = f"Python {payload['version']}; import_errors={payload['errors']}"
        except (KeyError, StopIteration, json.JSONDecodeError):
            pass
    results.add("Python training dependencies", ok, detail)


def check_assets(results: Results, wan_dir: Path, video_home: Path) -> None:
    check_file(results, "Wan DiT checkpoint", wan_dir / "diffusion_pytorch_model.safetensors", 1_000_000_000)
    check_file(results, "Wan VAE checkpoint", wan_dir / "Wan2.1_VAE.pth", 100_000_000)
    check_file(results, "null prompt embedding", wan_dir / "null_prompt_umt5.pt", 1_000)
    check_file(results, "VideoX-Fun Wan config", video_home / "config" / "wan2.1" / "wan_civitai.yaml", 500)
    check_file(results, "VideoX-Fun VAE source", video_home / "videox_fun" / "models" / "wan_vae.py", 1_000)
    check_file(
        results,
        "VideoX-Fun transformer source",
        video_home / "videox_fun" / "models" / "wan_transformer3d.py",
        1_000,
    )


def resolve_manifest_path(raw: str, manifest: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else manifest.parent / path


def check_droid(results: Results, data_root: Path, minimum: int) -> None:
    root = data_root / "droid_lerobot"
    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet")) if root.is_dir() else []
    results.add("DROID episode count", len(parquets) >= minimum, f"{len(parquets):,}; require {minimum:,}")
    if not parquets:
        return
    parquet = parquets[0]
    try:
        episode = int(parquet.stem.split("_")[1])
        chunk = episode // 1000
    except (IndexError, ValueError):
        results.add("DROID sample layout", False, f"cannot parse {parquet}")
        return
    missing = []
    for camera in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
        path = root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    results.add("DROID sample layout", not missing, f"episode {episode}; missing={missing}")


def check_egodex(results: Results, data_root: Path, minimum: int) -> None:
    manifest = data_root / "egodex_cdn" / "manifest.csv"
    if not manifest.is_file():
        results.add("EgoDex manifest", False, str(manifest))
        return
    count, examples = count_nonempty_lines(manifest)
    results.add("EgoDex episode count", count >= minimum, f"{count:,}; require {minimum:,}")
    missing = []
    for raw in examples:
        h5_path = resolve_manifest_path(raw.split()[0], manifest)
        for path in (h5_path, h5_path.with_suffix(".mp4")):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    results.add("EgoDex sample layout", bool(examples) and not missing, f"missing={missing}")


def check_abc(results: Results, data_root: Path, minimum: int) -> None:
    manifest = data_root / "abc_pp" / "manifest.txt"
    if not manifest.is_file():
        results.add("ABC manifest", False, str(manifest))
        return
    count, examples = count_nonempty_lines(manifest)
    results.add("ABC episode count", count >= minimum, f"{count:,}; require {minimum:,}")
    missing = []
    for raw in examples:
        episode = resolve_manifest_path(raw.split()[0], manifest)
        for filename in ("states.npz", "top.mp4", "left_wrist.mp4", "right_wrist.mp4"):
            path = episode / filename
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    results.add("ABC sample layout", bool(examples) and not missing, f"missing={missing}")


def agibot_root(data_root: Path, dataset_name: str) -> Path:
    key = dataset_name.strip().lower()
    if key == "alpha":
        return Path(os.environ.get("AGIBOT_ALPHA_ROOT", data_root / "agibot"))
    if key == "beta":
        return Path(os.environ.get("AGIBOT_BETA_ROOT", data_root / "agibot"))
    if key == "viscam":
        return data_root / "agibot_combined"
    return data_root / "agibot"


def check_agibot(results: Results, data_root: Path, minimum: int) -> None:
    manifest = data_root / "agibot" / "manifest.csv"
    if not manifest.is_file():
        results.add("AgiBot manifest", False, str(manifest))
        return
    count, examples = count_nonempty_lines(manifest, skip_header=True)
    results.add("AgiBot episode count", count >= minimum, f"{count:,}; require {minimum:,}")
    missing = []
    parsed = 0
    for raw in examples:
        row = next(csv.reader([raw]))
        if len(row) < 3:
            missing.append(f"malformed manifest row: {raw}")
            continue
        task, episode, dataset_name = row[:3]
        root = agibot_root(data_root, dataset_name)
        parsed += 1
        paths = [
            root / "proprio_stats" / task / episode / "proprio_stats.h5",
            *[
                root / "observations" / task / episode / "videos" / filename
                for filename in ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")
            ],
            *[
                root / "parameters" / task / episode / "parameters" / "camera" / filename
                for filename in (
                    "head_extrinsic_params_aligned.json",
                    "hand_left_extrinsic_params_aligned.json",
                    "hand_right_extrinsic_params_aligned.json",
                )
            ],
        ]
        missing.extend(str(path) for path in paths if not path.is_file() or path.stat().st_size == 0)
    results.add("AgiBot sample layout", parsed > 0 and not missing, f"missing={missing}")


def _manifest_entries(path: Path, limit: int) -> list[Path]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            value = raw.strip()
            if value:
                entries.append(Path(value).resolve(strict=False))
            if len(entries) >= limit:
                break
    return entries


def _current_data_fingerprints(
    data_root: Path,
    dataset_names: tuple[str, ...] = DATASET_NAMES,
    *,
    agibot_profile: str = "production",
) -> dict[str, dict[str, object]]:
    """Recompute the validator's cheap identity fingerprint for active files."""
    from tools.validate_training_data import (
        _agibot_lineage_findings,
        _find_agibot_qualification_provenance,
        stat_fingerprint,
    )

    selected = set(dataset_names)
    if not selected or not selected <= set(DATASET_NAMES):
        raise ValueError(f"invalid selected datasets: {sorted(selected)!r}")
    fingerprints: dict[str, dict[str, object]] = {}

    if "droid" in selected:
        droid_root = data_root / "droid_lerobot"
        droid_files: list[Path] = []
        for parquet in sorted(
            (droid_root / "data").glob("chunk-*/episode_*.parquet")
        )[:10_000]:
            match = parquet.stem.removeprefix("episode_")
            episode = int(match)
            chunk = episode // 1_000
            droid_files.append(parquet)
            droid_files.extend(
                droid_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / f"observation.images.{camera}"
                / f"episode_{episode:06d}.mp4"
                for camera in (
                    "exterior_image_1_left",
                    "exterior_image_2_left",
                    "wrist_image_left",
                )
            )
        fingerprints["droid"] = stat_fingerprint(droid_files)

    if "egodex" in selected:
        egodex_manifest = data_root / "egodex_cdn" / "manifest.csv"
        egodex_entries = _manifest_entries(egodex_manifest, 10_000)
        egodex_files = [
            item
            for path in egodex_entries
            for item in (path, path.with_suffix(".mp4"))
        ]
        fingerprints["egodex"] = stat_fingerprint(egodex_files, egodex_manifest)

    if "agibot" in selected:
        agibot_manifest = data_root / "agibot" / "manifest.csv"
        agibot_files: list[Path] = []
        agibot_roots: set[Path] = set()
        agibot_entry_count = 0
        with agibot_manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, skipinitialspace=True)
            next(reader)
            for row_index, row in enumerate(reader):
                if row_index >= 10_000:
                    break
                agibot_entry_count += 1
                task, episode, dataset_name = (value.strip() for value in row)
                root = agibot_root(data_root, dataset_name)
                agibot_roots.add(root)
                agibot_files.append(
                    root / "proprio_stats" / task / episode / "proprio_stats.h5"
                )
                agibot_files.extend(
                    root / "observations" / task / episode / "videos" / filename
                    for filename in (
                        "head_color.mp4",
                        "hand_left_color.mp4",
                        "hand_right_color.mp4",
                    )
                )
                agibot_files.extend(
                    root
                    / "parameters"
                    / task
                    / episode
                    / "parameters"
                    / "camera"
                    / filename
                    for filename in (
                        "head_extrinsic_params_aligned.json",
                        "hand_left_extrinsic_params_aligned.json",
                        "hand_right_extrinsic_params_aligned.json",
                    )
                )

        agibot_fingerprint = stat_fingerprint(agibot_files, agibot_manifest)
        lineage_paths: list[Path] = []
        if agibot_profile == "production":
            lineage_findings, lineage_paths = _agibot_lineage_findings(
                agibot_roots,
                agibot_manifest,
                agibot_entry_count,
                "production",
                expected_payloads=set(agibot_files),
                verify_upstream=False,
                verify_payload_hashes=False,
            )
            lineage_errors = [
                item for item in lineage_findings if item.severity == "error"
            ]
            if lineage_errors:
                examples = "; ".join(item.message for item in lineage_errors[:3])
                raise ValueError(
                    f"current AgiBot production lineage is invalid: {examples}"
                )
            if lineage_paths:
                agibot_fingerprint["lineage"] = stat_fingerprint(lineage_paths)
        elif agibot_profile != "qualification":
            raise ValueError(f"unsupported AgiBot profile: {agibot_profile!r}")
        agibot_fingerprint["agibot_profile"] = agibot_profile
        agibot_fingerprint["qualification_markers"] = [
            str(path)
            for path, _signals in _find_agibot_qualification_provenance(
                agibot_roots
            )
        ]
        agibot_fingerprint["lineage_files"] = [
            str(path) for path in lineage_paths
        ]
        fingerprints["agibot"] = agibot_fingerprint

    if "abc" in selected:
        abc_manifest = data_root / "abc_pp" / "manifest.txt"
        abc_entries = _manifest_entries(abc_manifest, 10_000)
        abc_files = [
            item
            for path in abc_entries
            for item in (
                path / "states.npz",
                path / "top.mp4",
                path / "left_wrist.mp4",
                path / "right_wrist.mp4",
            )
        ]
        fingerprints["abc"] = stat_fingerprint(abc_files, abc_manifest)

    return {name: fingerprints[name] for name in DATASET_NAMES if name in selected}


def _fast_agibot_metadata_fingerprint(
    data_root: Path, waiver: dict[str, object]
) -> tuple[str, int, int]:
    """Recompute the waived overlay's metadata-only AgiBot seal.

    This deliberately reads only the active manifest and inode metadata.  It
    does not decode videos or hash payload contents; that omission is recorded
    by the immutable waiver and must be separately authorized.
    """

    staging_value = waiver.get("staging_root")
    if not isinstance(staging_value, str):
        raise ValueError("fast waiver lacks staging_root")
    staging = Path(staging_value).expanduser().resolve(strict=True)
    manifest = data_root / "agibot" / "manifest.csv"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"fast AgiBot manifest is missing/symlinked: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["task_id", "episode_id", "dataset"]:
            raise ValueError("fast AgiBot manifest has an unexpected header")
        rows = [tuple(value.strip() for value in row) for row in reader]
    if (
        len(rows) != 5_671
        or len(set(rows)) != 5_671
        or any(
            len(row) != 3
            or not row[0].isdigit()
            or not row[1].isdigit()
            or row[2] != "scr"
            or row[0] == "475"
            for row in rows
        )
    ):
        raise ValueError("fast AgiBot manifest is not the approved 5,671-row schema")

    videos = (
        "head_color.mp4",
        "hand_left_color.mp4",
        "hand_right_color.mp4",
    )
    extrinsics = (
        "head_extrinsic_params_aligned.json",
        "hand_left_extrinsic_params_aligned.json",
        "hand_right_extrinsic_params_aligned.json",
    )
    digest = hashlib.sha256()
    payload_count = 0
    payload_bytes = 0
    for task, episode, _dataset in rows:
        paths = [
            data_root
            / "agibot"
            / "proprio_stats"
            / task
            / episode
            / "proprio_stats.h5"
        ]
        paths.extend(
            data_root / "agibot" / "observations" / task / episode / "videos" / name
            for name in videos
        )
        paths.extend(
            data_root
            / "agibot"
            / "parameters"
            / task
            / episode
            / "parameters"
            / "camera"
            / name
            for name in extrinsics
        )
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"fast AgiBot payload must not itself be a symlink: {path}")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(staging):
                raise ValueError(
                    f"fast AgiBot payload escapes the sealed staging root: {path} -> {resolved}"
                )
            info = resolved.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                raise ValueError(f"fast AgiBot payload is not a nonempty file: {resolved}")
            relative = resolved.relative_to(staging).as_posix()
            digest.update(
                f"{relative}\0{info.st_size}\0{info.st_mtime_ns}\0"
                f"{info.st_ctime_ns}\0{info.st_ino}\n".encode()
            )
            payload_count += 1
            payload_bytes += info.st_size
    return digest.hexdigest(), payload_count, payload_bytes


def check_fast_training_evidence(
    results: Results,
    data_root: Path,
    data_report_path: Path | None,
    authorization_path: Path | None,
    mixed_report_path: Path | None,
) -> None:
    """Validate explicit user authorization and live fast-overlay evidence."""

    label = "fast user-waived training evidence"
    if (
        data_report_path is None
        or authorization_path is None
        or mixed_report_path is None
    ):
        results.add(
            label,
            False,
            "fast policy requires --fast-training-authorization and --mixed-loader-report",
        )
        return
    try:
        authorization_path = canonical_regular_file(
            authorization_path, "fast training authorization"
        )
        data_report_path = canonical_regular_file(
            data_report_path, "files-only data report"
        )
        mixed_report_path = canonical_regular_file(
            mixed_report_path, "mixed-loader report"
        )
        authorization = read_json_object(
            authorization_path, "fast training authorization"
        )
        mixed = read_json_object(mixed_report_path, "mixed-loader report")
        waiver_path = data_root / "fast_validation_waiver.json"
        waiver = read_json_object(waiver_path, "original fast validation waiver")

        if authorization.get("schema_version") != 1:
            raise ValueError("fast training authorization schema_version must be 1")
        if authorization.get("kind") != FAST_AUTHORIZATION_KIND:
            raise ValueError("fast training authorization has an unexpected kind")
        if authorization.get("training_authorized") is not True:
            raise ValueError("fast training authorization is not explicitly true")
        if authorization.get("policy") != FAST_DATA_POLICY:
            raise ValueError("fast training authorization policy mismatch")
        if Path(str(authorization.get("data_root", ""))).expanduser().resolve(
            strict=False
        ) != data_root.resolve(strict=False):
            raise ValueError("fast training authorization is bound to another data root")
        if authorization.get("branch") != "lora":
            raise ValueError("fast training authorization is for another branch")
        if authorization.get("authorization_scope") != (
            "one_branch_one_commit_one_fast_overlay"
        ):
            raise ValueError("fast training authorization scope mismatch")
        if authorization.get("authorized_by") != "user" or not isinstance(
            authorization.get("authorization_basis"), str
        ) or not str(authorization["authorization_basis"]).strip():
            raise ValueError("fast training authorization lacks explicit user authority")
        if authorization.get("source_order") != list(FAST_SOURCE_ORDER) or authorization.get(
            "source_lengths"
        ) != list(FAST_SOURCE_LENGTHS):
            raise ValueError("fast training authorization source topology mismatch")
        current_commit = run_command(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
        ).stdout.strip()
        if authorization.get("expected_commit") != current_commit:
            raise ValueError("fast training authorization is for another commit")
        parse_aware_timestamp(authorization.get("created_at_utc"), "created_at_utc")
        if Path(str(authorization.get("certificate_path", ""))).expanduser().resolve(
            strict=False
        ) != authorization_path:
            raise ValueError("fast training authorization was moved from its bound path")
        if authorization_path.stat().st_mode & 0o222:
            raise ValueError("fast training authorization must be read-only")

        waiver_hash = sha256_file(waiver_path)
        inputs = authorization.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != {
            "waiver",
            "files_only_report",
            "mixed_loader_report",
            "gradient_report",
        }:
            raise ValueError("authorization input bindings are incomplete")

        def require_input_record(name: str, expected_path: Path | None = None) -> Path:
            record = inputs.get(name)
            if not isinstance(record, dict):
                raise ValueError(f"authorization input binding is malformed: {name}")
            path = canonical_regular_file(
                Path(str(record.get("path", ""))),
                f"authorization input {name}",
            )
            if expected_path is not None and path != expected_path:
                raise ValueError(f"authorization input path mismatch: {name}")
            if path.stat().st_size != record.get("size") or sha256_file(path) != record.get(
                "sha256"
            ):
                raise ValueError(f"authorization input bytes changed: {name}")
            return path

        require_input_record("waiver", waiver_path)
        require_input_record("files_only_report", data_report_path)
        require_input_record("mixed_loader_report", mixed_report_path)
        require_input_record("gradient_report")
        if waiver.get("schema_version") != 1 or waiver.get("kind") != FAST_WAIVER_KIND:
            raise ValueError("original fast waiver kind/schema mismatch")
        if (
            waiver.get("logical_read_skipped") is not True
            or waiver.get("strict_validated") is not False
        ):
            raise ValueError("original fast waiver does not describe the skipped checks")
        if waiver.get("selected_episodes") != 5_671 or waiver.get(
            "required_payloads"
        ) != 39_697:
            raise ValueError("original fast waiver corpus counts mismatch")
        recorded_fingerprint = waiver.get("metadata_fingerprint_sha256")
        if not isinstance(recorded_fingerprint, str) or len(recorded_fingerprint) != 64:
            raise ValueError("original fast waiver lacks a metadata fingerprint")
        live_fingerprint, payload_count, payload_bytes = (
            _fast_agibot_metadata_fingerprint(data_root, waiver)
        )
        if live_fingerprint != recorded_fingerprint:
            raise ValueError("live AgiBot metadata fingerprint differs from the waiver")
        if payload_count != waiver.get("required_payloads") or payload_bytes != waiver.get(
            "required_payload_bytes"
        ):
            raise ValueError("live AgiBot payload metadata counts differ from the waiver")
        authorized_agibot = authorization.get("agibot")
        if (
            not isinstance(authorized_agibot, dict)
            or authorized_agibot.get("metadata_fingerprint_sha256")
            != live_fingerprint
            or authorized_agibot.get("required_payloads") != payload_count
            or authorized_agibot.get("required_payload_bytes") != payload_bytes
        ):
            raise ValueError("authorization AgiBot metadata seal differs from live evidence")
        authorized_validation = authorization.get("validation")
        if not isinstance(authorized_validation, dict):
            raise ValueError("authorization validation summary is absent")
        original_summary = authorized_validation.get("original_waiver")
        files_summary = authorized_validation.get("files_only")
        mixed_summary = authorized_validation.get("mixed_loader")
        gradient_summary = authorized_validation.get("real_gradient")
        if (
            not isinstance(original_summary, dict)
            or original_summary.get("logical_read_skipped") is not True
            or original_summary.get("strict_validated") is not False
            or original_summary.get("training_authorized") is not False
            or not isinstance(files_summary, dict)
            or files_summary.get("passed") is not True
            or files_summary.get("files_only") is not True
            or not isinstance(mixed_summary, dict)
            or mixed_summary.get("passed") is not True
            or not isinstance(gradient_summary, dict)
            or gradient_summary.get("passed") is not True
        ):
            raise ValueError("authorization validation summaries are incomplete")

        if mixed.get("schema_version") != 1 or mixed.get("kind") != (
            "lacwm_real_mixed_stateful_dataloader_smoke"
        ):
            raise ValueError("mixed-loader report kind/schema mismatch")
        if (
            mixed.get("status") != "passed"
            or mixed.get("git_commit") != current_commit
            or mixed.get("git_status") != ""
            or Path(str(mixed.get("requested_data_root", ""))).expanduser().resolve(
                strict=False
            )
            != data_root.resolve(strict=False)
        ):
            raise ValueError("mixed-loader report is not passing and commit-bound")
        validation = mixed.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("mixed-loader report lacks validation evidence")
        mixed_data = validation.get("data")
        mix = validation.get("mix")
        resume = validation.get("resume")
        if not all(isinstance(value, dict) for value in (mixed_data, mix, resume)):
            raise ValueError("mixed-loader report has malformed data/mix/resume evidence")
        if Path(str(mixed_data.get("root", ""))).expanduser().resolve(
            strict=False
        ) != data_root.resolve(strict=False):
            raise ValueError("mixed-loader report is bound to another data root")
        if mixed_data.get("source_order") != list(FAST_SOURCE_ORDER) or mixed_data.get(
            "source_lengths"
        ) != list(FAST_SOURCE_LENGTHS):
            raise ValueError("mixed-loader source order/lengths mismatch")
        if mixed_data.get("total_episodes") != sum(FAST_SOURCE_LENGTHS):
            raise ValueError("mixed-loader total episode count mismatch")
        observed = mix.get("observed_source_counts")
        if (
            not isinstance(observed, dict)
            or set(observed) != set(FAST_SOURCE_ORDER)
            or any(int(observed.get(name, 0)) <= 0 for name in FAST_SOURCE_ORDER)
            or not isinstance(mix.get("batches_checked"), int)
            or mix.get("batches_checked", 0) <= 0
            or mix.get("mixed_batches") != mix.get("batches_checked")
        ):
            raise ValueError("mixed-loader report did not observe every source and a mixed batch")
        if (
            resume.get("exact_continuation") is not True
            or resume.get("reference_signature") != resume.get("restored_signature")
        ):
            raise ValueError("mixed-loader report lacks exact resume continuity")
        state_path = canonical_regular_file(
            Path(str(resume.get("state_path", ""))),
            "mixed-loader state artifact",
        )
        if (
            state_path.stat().st_size != resume.get("state_size")
            or sha256_file(state_path) != resume.get("state_sha256")
        ):
            raise ValueError("mixed-loader state artifact hash mismatch")

        evidence = mixed_data.get("evidence")
        expected_evidence = {
            ".prepared/manifests.ready": data_root / ".prepared/manifests.ready",
            "egodex_cdn/manifest.csv": data_root / "egodex_cdn/manifest.csv",
            "agibot/manifest.csv": data_root / "agibot/manifest.csv",
            "abc_pp/manifest.txt": data_root / "abc_pp/manifest.txt",
        }
        if not isinstance(evidence, dict) or set(evidence) != set(expected_evidence):
            raise ValueError("mixed-loader manifest evidence set mismatch")
        for name, expected_path in expected_evidence.items():
            record = evidence.get(name)
            if not isinstance(record, dict):
                raise ValueError(f"mixed-loader evidence is malformed for {name}")
            resolved = expected_path.resolve(strict=True)
            if Path(str(record.get("path", ""))).expanduser().resolve(
                strict=False
            ) != resolved:
                raise ValueError(f"mixed-loader evidence path mismatch for {name}")
            if resolved.stat().st_size != record.get("size") or sha256_file(
                resolved
            ) != record.get("sha256"):
                raise ValueError(f"live manifest evidence changed for {name}")

        results.add(
            label,
            True,
            f"authorization_sha256={sha256_file(authorization_path)}; "
            f"waiver_sha256={waiver_hash}; mixed_sha256={sha256_file(mixed_report_path)}; "
            f"agibot_metadata_sha256={live_fingerprint}; payloads={payload_count}",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        results.add(label, False, str(exc))


def check_strict_data_report(
    results: Results,
    report_path: Path | None,
    data_root: Path,
    python_bin: Path,
    args: argparse.Namespace,
) -> None:
    dataset_names = normalized_dataset_names(args.datasets)
    policy = getattr(args, "data_validation_policy", STRICT_DATA_POLICY)
    fast_policy = policy == FAST_DATA_POLICY
    validation_label = (
        "files-only user-waived data validation"
        if fast_policy
        else "strict data validation"
    )
    payload: dict[str, object] | None = None
    detail_prefix = ""
    if report_path is not None:
        try:
            payload = json.loads(report_path.read_text())
            generated_at = datetime.fromisoformat(str(payload.get("generated_at_utc")))
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
            detail_prefix = f"cached report {report_path}; internal_age={age_hours:.1f}h; "
            if age_hours > args.max_data_report_age_hours:
                results.add(
                    validation_label,
                    False,
                    detail_prefix + f"older than {args.max_data_report_age_hours:.1f}h",
                )
                return
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            results.add(validation_label, False, f"cannot read {report_path}: {exc}")
            return
    else:
        if fast_policy:
            results.add(
                validation_label,
                False,
                "fast policy requires an explicit commit-bound files-only report",
            )
            return
        command = [
            str(python_bin),
            str(REPO_ROOT / "tools" / "validate_training_data.py"),
            "--data-root",
            str(data_root),
            "--datasets",
            *dataset_names,
            "--workers",
            str(args.data_validation_workers),
            "--agibot-profile",
            "production",
            "--json",
        ]
        completed = run_command(command, timeout=21_600)
        if completed.returncode not in {0, 1}:
            results.add(
                validation_label,
                False,
                f"validator failed to run: {completed.stderr.strip() or completed.stdout.strip()}",
            )
            return
        try:
            payload = json.loads(completed.stdout)
            detail_prefix = "fresh strict validator run; "
        except json.JSONDecodeError as exc:
            results.add(validation_label, False, f"validator emitted invalid JSON: {exc}")
            return

    reports = payload.get("reports", []) if isinstance(payload, dict) else []
    report_names = [
        str(report.get("name")) for report in reports if isinstance(report, dict)
    ]
    names = set(report_names)
    all_sources = {
        "droid": (data_root / "droid_lerobot").resolve(strict=False),
        "egodex": (data_root / "egodex_cdn" / "manifest.csv").resolve(strict=False),
        "agibot": (data_root / "agibot" / "manifest.csv").resolve(strict=False),
        "abc": (data_root / "abc_pp" / "manifest.txt").resolve(strict=False),
    }
    expected_sources = {name: all_sources[name] for name in dataset_names}
    sources_ok = all(
        isinstance(report, dict)
        and str(report.get("name")) in expected_sources
        and Path(str(report.get("source", "/nonexistent"))).resolve(strict=False)
        == expected_sources[str(report.get("name"))]
        for report in reports
    )
    expected_by_name = {
        "droid": args.min_droid,
        "egodex": args.min_egodex,
        "agibot": args.min_agibot,
        "abc": args.min_abc,
    }
    content_counts_ok = all(
        isinstance(report, dict)
        and (
            fast_policy
            or int(report.get("checked", 0))
            >= expected_by_name.get(str(report.get("name")), 10**18)
        )
        and int(report.get("selected", 0))
        >= expected_by_name.get(str(report.get("name")), 10**18)
        and int(report.get("active_complete", 0))
        >= expected_by_name.get(str(report.get("name")), 10**18)
        and (
            not fast_policy
            or int(report.get("complete", 0))
            >= expected_by_name.get(str(report.get("name")), 10**18)
        )
        for report in reports
    )
    current_commit = run_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    validator_path = REPO_ROOT / "tools" / "validate_training_data.py"
    validator_hash = hashlib.sha256(validator_path.read_bytes()).hexdigest()
    invocation = payload.get("invocation") if isinstance(payload, dict) else None
    selected_caps = {name: 10_000 for name in dataset_names}
    selected_expected = {
        name: (5_671 if name == "agibot" else 10_000)
        for name in dataset_names
    }
    invocation_sources = invocation.get("sources") if isinstance(invocation, dict) else None
    invocation_caps = invocation.get("caps") if isinstance(invocation, dict) else None
    invocation_expected = invocation.get("expected") if isinstance(invocation, dict) else None
    invocation_external_roots = (
        invocation.get("allowed_external_roots")
        if isinstance(invocation, dict)
        else None
    )
    fast_external_roots_ok = bool(
        not fast_policy
        or (
            isinstance(invocation_external_roots, dict)
            and Path(
                str(invocation_external_roots.get("egodex", "/nonexistent"))
            ).expanduser().resolve(strict=False)
            == (data_root.parent / "egodex_cdn").resolve(strict=False)
            and Path(
                str(invocation_external_roots.get("abc", "/nonexistent"))
            ).expanduser().resolve(strict=False)
            == (data_root.parent / "abc_pp").resolve(strict=False)
        )
    )
    invocation_ok = bool(
        isinstance(invocation, dict)
        and invocation.get("data_root") == str(data_root.resolve(strict=False))
        and invocation.get("datasets") == list(dataset_names)
        and invocation.get("min_timesteps") == 66
        and invocation.get("validate_all") is False
        and invocation.get("files_only") is fast_policy
        and isinstance(invocation_sources, dict)
        and {
            name: str(
                Path(str(invocation_sources.get(name, "/nonexistent"))).resolve(
                    strict=False
                )
            )
            for name in dataset_names
        }
        == {name: str(path) for name, path in expected_sources.items()}
        and isinstance(invocation_caps, dict)
        and {name: invocation_caps.get(name) for name in dataset_names}
        == selected_caps
        and isinstance(invocation_expected, dict)
        and {name: invocation_expected.get(name) for name in dataset_names}
        == selected_expected
        and fast_external_roots_ok
        and (
            "agibot" not in dataset_names
            or (
                invocation.get("agibot_profile")
                == ("qualification" if fast_policy else "production")
                and invocation.get("agibot_roots")
                == {
                    "scr": str(
                        agibot_root(data_root, "scr").resolve(strict=False)
                    ),
                    "alpha": str(
                        agibot_root(data_root, "alpha").resolve(strict=False)
                    ),
                    "beta": str(
                        agibot_root(data_root, "beta").resolve(strict=False)
                    ),
                    "viscam": str(
                        agibot_root(data_root, "viscam").resolve(strict=False)
                    ),
                }
            )
        )
    )
    provenance_ok = bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("git_commit") == current_commit
        and payload.get("git_status") == ""
        and payload.get("validator_sha256") == validator_hash
    )
    fingerprints_ok = False
    if (
        isinstance(payload, dict)
        and len(report_names) == len(dataset_names)
        and names == set(dataset_names)
    ):
        try:
            if fast_policy:
                current_fingerprints = _current_data_fingerprints(
                    data_root, dataset_names, agibot_profile="qualification"
                )
            else:
                current_fingerprints = _current_data_fingerprints(
                    data_root, dataset_names
                )
            reported_fingerprints = {
                str(report["name"]): report.get("fingerprint")
                for report in reports
                if isinstance(report, dict)
            }
            fingerprints_ok = reported_fingerprints == current_fingerprints
        except (OSError, ValueError, KeyError, csv.Error):
            fingerprints_ok = False
    ok = bool(
        isinstance(payload, dict)
        and payload.get("read_only") is True
        and payload.get("passed") is True
        and len(report_names) == len(dataset_names)
        and names == set(dataset_names)
        and sources_ok
        and content_counts_ok
        and invocation_ok
        and provenance_ok
        and fingerprints_ok
    )
    errors = sum(int(report.get("error_count", 0)) for report in reports if isinstance(report, dict))
    checked = sum(int(report.get("checked", 0)) for report in reports if isinstance(report, dict))
    results.add(
        validation_label,
        ok,
        detail_prefix
        + f"datasets={sorted(str(name) for name in names)}; checked={checked:,}; "
        + f"errors={errors}; sources_match={sources_ok}; strict_counts={content_counts_ok}; "
        + f"invocation={invocation_ok}; provenance={provenance_ok}; fingerprints={fingerprints_ok}",
    )


def check_data(
    results: Results,
    data_root: Path,
    profile: str,
    args: argparse.Namespace,
    python_bin: Path,
) -> None:
    dataset_names = normalized_dataset_names(args.datasets)
    results.add("data root", data_root.is_dir(), str(data_root))
    if not data_root.is_dir():
        return
    if profile == "full":
        report_path = args.data_validation_report.resolve(strict=False) if args.data_validation_report else None
        check_strict_data_report(results, report_path, data_root, python_bin, args)
        if args.data_validation_policy == FAST_DATA_POLICY:
            check_fast_training_evidence(
                results,
                data_root,
                report_path,
                args.fast_training_authorization,
                args.mixed_loader_report,
            )
    checks = {
        "droid": (check_droid, args.min_droid),
        "egodex": (check_egodex, args.min_egodex),
        "agibot": (check_agibot, args.min_agibot),
        "abc": (check_abc, args.min_abc),
    }
    for name in dataset_names:
        check, configured_minimum = checks[name]
        check(results, data_root, configured_minimum if profile == "full" else 1)


def list_training_processes() -> list[dict[str, str | int]]:
    found: list[dict[str, str | int]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (OSError, PermissionError):
            continue
        lowered = command.lower()
        if command and any(marker in lowered for marker in TRAIN_MARKERS):
            found.append({"pid": int(entry.name), "command": command[:500]})
    return sorted(found, key=lambda item: int(item["pid"]))


def query_gpus() -> tuple[list[dict[str, object]], str]:
    fields = "index,uuid,name,memory.total,memory.free,compute_mode"
    completed = run_command(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if completed.returncode != 0:
        return [], completed.stderr.strip() or "nvidia-smi failed"
    gpus: list[dict[str, object]] = []
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(row[0]),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": int(row[3]),
                    "memory_free_mib": int(row[4]),
                    "compute_mode": row[5].strip(),
                }
            )
        except ValueError:
            continue
    return gpus, completed.stdout.strip()


def query_compute_apps() -> tuple[list[dict[str, str]], str | None]:
    fields = "gpu_uuid,pid,process_name,used_memory"
    completed = run_command(["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"])
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or (
            f"nvidia-smi compute-app query exited {completed.returncode}"
        )
        return [], detail
    apps = []
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) == 4:
            apps.append({"gpu_uuid": row[0].strip(), "pid": row[1].strip(), "name": row[2].strip(), "used_mib": row[3].strip()})
    return apps, None


def check_gpus(
    results: Results,
    selected: list[int],
    profile: str,
    min_gpu_memory_mib: int = DEFAULT_MIN_GPU_MEMORY_MIB,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    gpus, inventory = query_gpus()
    results.add("nvidia-smi", bool(gpus), inventory or "no GPUs reported")
    by_index = {int(gpu["index"]): gpu for gpu in gpus}
    expected = 8 if profile == "full" else 1
    if profile in {"smoke", "full"}:
        results.add("selected GPU count", len(selected) == expected, f"selected={selected}; require {expected}")
    missing = [index for index in selected if index not in by_index]
    results.add("selected GPUs exist", not missing, f"missing={missing}")
    chosen = [by_index[index] for index in selected if index in by_index]
    if profile == "full":
        non_b200 = [f"{gpu['index']}:{gpu['name']}" for gpu in chosen if "B200" not in str(gpu["name"]).upper()]
        results.add("B200 model enforcement", not non_b200 and len(chosen) == 8, f"non-B200={non_b200}")
        low_memory = [
            f"{gpu['index']}:{gpu['memory_total_mib']}MiB"
            for gpu in chosen
            if int(gpu["memory_total_mib"]) < min_gpu_memory_mib
        ]
        results.add(
            "B200 memory capacity",
            not low_memory and len(chosen) == 8,
            f"minimum={min_gpu_memory_mib} MiB; below minimum={low_memory}",
        )
    elif profile == "smoke":
        low_memory = [f"{gpu['index']}:{gpu['memory_free_mib']}MiB" for gpu in chosen if int(gpu["memory_free_mib"]) < 40_000]
        results.add("smoke GPU free memory", not low_memory and len(chosen) == 1, f"below 40000 MiB free={low_memory}")

    apps, apps_error = query_compute_apps()
    results.add(
        "GPU compute-process query",
        apps_error is None,
        "query succeeded" if apps_error is None else apps_error,
        required=profile in {"smoke", "full"},
    )
    selected_uuids = {str(gpu["uuid"]) for gpu in chosen}
    selected_apps = [app for app in apps if app["gpu_uuid"] in selected_uuids]
    unselected_apps = [app for app in apps if app["gpu_uuid"] not in selected_uuids]
    results.add(
        "selected GPUs idle",
        apps_error is None and not selected_apps,
        (
            "occupancy unknown because the compute-process query failed"
            if apps_error is not None
            else "no active compute applications"
            if not selected_apps
            else json.dumps(selected_apps, sort_keys=True)
        ),
        required=profile in {"smoke", "full"},
    )
    results.add(
        "jobs on unselected GPUs",
        not unselected_apps,
        "none" if not unselected_apps else json.dumps(unselected_apps, sort_keys=True),
        required=False,
    )
    processes = list_training_processes()
    results.add(
        "no active training processes",
        not processes,
        "none" if not processes else json.dumps(processes, sort_keys=True),
        required=profile == "full",
    )
    return chosen, apps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("assets", "smoke", "full"), default="assets")
    parser.add_argument("--gpus", help="comma-separated physical GPU indices")
    parser.add_argument("--python", default=os.environ.get("LACWM_PYTHON", sys.executable))
    parser.add_argument("--wan-dir", default=os.environ.get("WAN_DIR"))
    parser.add_argument("--videox-home", default=os.environ.get("VIDEOX_HOME"))
    parser.add_argument("--data-root", default=os.environ.get("LACWM_DATA"))
    parser.add_argument("--run-root", default=os.environ.get("LACWM_RUNS"))
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="skip dataset paths only for explicitly synthetic smoke validation",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--data-validation-report",
        type=Path,
        help=(
            "consume a JSON report from tools/validate_training_data.py; strict "
            "full mode invokes it when omitted, while the fast policy requires "
            "an explicit --files-only report"
        ),
    )
    parser.add_argument(
        "--data-validation-policy",
        choices=DATA_VALIDATION_POLICIES,
        default=STRICT_DATA_POLICY,
        help="strict by default; the files-only policy requires explicit waiver evidence",
    )
    parser.add_argument(
        "--fast-training-authorization",
        type=Path,
        help="explicit immutable authorization JSON required by files_only_user_waived_v1",
    )
    parser.add_argument(
        "--mixed-loader-report",
        type=Path,
        help="commit/data-bound mixed StatefulDataLoader report required by the fast policy",
    )
    parser.add_argument("--max-data-report-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--min-gpu-memory-mib",
        type=supported_gpu_memory_mib,
        default=DEFAULT_MIN_GPU_MEMORY_MIB,
        help=(
            "minimum total memory required on each B200 in full mode "
            f"(default: {DEFAULT_MIN_GPU_MEMORY_MIB} MiB)"
        ),
    )
    parser.add_argument("--data-validation-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="exact dataset set required by this training stage",
    )
    parser.add_argument("--min-droid", type=int, default=10_000)
    parser.add_argument("--min-egodex", type=int, default=10_000)
    parser.add_argument("--min-agibot", type=int, default=5_671)
    parser.add_argument("--min-abc", type=int, default=10_000)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.skip_data and args.profile != "smoke":
        parser.error("--skip-data is allowed only with --profile smoke")
    fast_evidence_supplied = bool(
        args.fast_training_authorization or args.mixed_loader_report
    )
    if args.data_validation_policy == FAST_DATA_POLICY:
        if args.profile != "full":
            parser.error("the files-only user-waived policy is allowed only with --profile full")
        if not args.data_validation_report:
            parser.error("the files-only user-waived policy requires --data-validation-report")
        if not args.fast_training_authorization or not args.mixed_loader_report:
            parser.error(
                "the files-only user-waived policy requires --fast-training-authorization "
                "and --mixed-loader-report"
            )
    elif fast_evidence_supplied:
        parser.error(
            "fast authorization/mixed evidence is forbidden unless --data-validation-policy "
            f"is {FAST_DATA_POLICY}"
        )
    try:
        args.datasets = list(normalized_dataset_names(args.datasets))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        selected = parse_gpu_list(args.gpus)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    results = Results()
    check_repo(results, args.profile)
    check_host(results, args.profile)

    required_paths = {
        "WAN_DIR": args.wan_dir,
        "VIDEOX_HOME": args.videox_home,
        "LACWM_RUNS": args.run_root,
    }
    if not args.skip_data:
        required_paths["LACWM_DATA"] = args.data_root
    for name, value in required_paths.items():
        results.add(f"{name} set", bool(value), value or "unset")

    wan_dir = Path(args.wan_dir or "/nonexistent/WAN_DIR")
    video_home = Path(args.videox_home or "/nonexistent/VIDEOX_HOME")
    data_root = Path(args.data_root or "/nonexistent/LACWM_DATA")
    run_root = Path(args.run_root or "/nonexistent/LACWM_RUNS")
    # Do not resolve a venv's ``bin/python`` symlink to its base interpreter;
    # executing the resolved target would discard the venv's site-packages.
    python_bin = Path(args.python).expanduser().absolute()

    check_output_root(results, run_root, args.profile)
    check_python(results, python_bin, video_home)
    check_assets(results, wan_dir, video_home)
    if args.skip_data:
        results.add("dataset checks", True, "SKIPPED: synthetic gradient smoke only")
    else:
        check_data(results, data_root, args.profile, args, python_bin)
    chosen_gpus, apps = check_gpus(
        results, selected, args.profile, args.min_gpu_memory_mib
    )

    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "profile": args.profile,
        "data_validation_policy": args.data_validation_policy,
        "dataset_names": args.datasets,
        "gpu_requirements": {
            "model": "B200" if args.profile == "full" else None,
            "minimum_memory_mib": (
                args.min_gpu_memory_mib if args.profile == "full" else None
            ),
        },
        "passed": results.passed,
        "selected_gpus": chosen_gpus,
        "compute_apps": apps,
        "paths": {
            "repo_root": str(REPO_ROOT),
            "python": str(python_bin),
            "wan_dir": str(wan_dir),
            "videox_home": str(video_home),
            "data_root": str(data_root),
            "run_root": str(run_root),
        },
        "checks": [asdict(item) for item in results.checks],
    }

    for item in results.checks:
        if item.ok:
            label = "PASS"
        elif item.required:
            label = "FAIL"
        else:
            label = "WARN"
        print(f"[{label}] {item.name}: {item.detail}")
    print(f"\nPREFLIGHT {'PASSED' if results.passed else 'FAILED'}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if results.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
