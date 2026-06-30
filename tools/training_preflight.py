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
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects" / "latent_action_models"
ALLOWED_RUN_ROOTS = (Path("/mnt/data1"), Path("/mnt/data2"))
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


def existing_writable_ancestor(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.exists() and current.is_dir() and os.access(current, os.W_OK | os.X_OK):
        return current
    return None


def path_is_under(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in roots)


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
    allowed = path_is_under(run_root, ALLOWED_RUN_ROOTS)
    results.add("run root policy", allowed, f"{run_root.resolve(strict=False)} must be under /mnt/data1 or /mnt/data2")
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


def _current_data_fingerprints(data_root: Path) -> dict[str, dict[str, object]]:
    """Recompute the validator's cheap identity fingerprint for active files."""
    from tools.validate_training_data import stat_fingerprint

    droid_root = data_root / "droid_lerobot"
    droid_files: list[Path] = []
    for parquet in sorted((droid_root / "data").glob("chunk-*/episode_*.parquet"))[:10_000]:
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
            for camera in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
        )

    egodex_manifest = data_root / "egodex_cdn" / "manifest.csv"
    egodex_entries = _manifest_entries(egodex_manifest, 10_000)
    egodex_files = [item for path in egodex_entries for item in (path, path.with_suffix(".mp4"))]

    agibot_manifest = data_root / "agibot" / "manifest.csv"
    agibot_files: list[Path] = []
    with agibot_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        next(reader)
        for row_index, row in enumerate(reader):
            if row_index >= 10_000:
                break
            task, episode, dataset_name = (value.strip() for value in row)
            root = agibot_root(data_root, dataset_name)
            agibot_files.append(root / "proprio_stats" / task / episode / "proprio_stats.h5")
            agibot_files.extend(
                root / "observations" / task / episode / "videos" / filename
                for filename in ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")
            )
            agibot_files.extend(
                root / "parameters" / task / episode / "parameters" / "camera" / filename
                for filename in (
                    "head_extrinsic_params_aligned.json",
                    "hand_left_extrinsic_params_aligned.json",
                    "hand_right_extrinsic_params_aligned.json",
                )
            )

    abc_manifest = data_root / "abc_pp" / "manifest.txt"
    abc_entries = _manifest_entries(abc_manifest, 10_000)
    abc_files = [
        item
        for path in abc_entries
        for item in (path / "states.npz", path / "top.mp4", path / "left_wrist.mp4", path / "right_wrist.mp4")
    ]

    return {
        "droid": stat_fingerprint(droid_files),
        "egodex": stat_fingerprint(egodex_files, egodex_manifest),
        "agibot": stat_fingerprint(agibot_files, agibot_manifest),
        "abc": stat_fingerprint(abc_files, abc_manifest),
    }


def check_strict_data_report(
    results: Results,
    report_path: Path | None,
    data_root: Path,
    python_bin: Path,
    args: argparse.Namespace,
) -> None:
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
                    "strict data validation",
                    False,
                    detail_prefix + f"older than {args.max_data_report_age_hours:.1f}h",
                )
                return
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            results.add("strict data validation", False, f"cannot read {report_path}: {exc}")
            return
    else:
        command = [
            str(python_bin),
            str(REPO_ROOT / "tools" / "validate_training_data.py"),
            "--data-root",
            str(data_root),
            "--workers",
            str(args.data_validation_workers),
            "--json",
        ]
        completed = run_command(command, timeout=21_600)
        if completed.returncode not in {0, 1}:
            results.add(
                "strict data validation",
                False,
                f"validator failed to run: {completed.stderr.strip() or completed.stdout.strip()}",
            )
            return
        try:
            payload = json.loads(completed.stdout)
            detail_prefix = "fresh strict validator run; "
        except json.JSONDecodeError as exc:
            results.add("strict data validation", False, f"validator emitted invalid JSON: {exc}")
            return

    reports = payload.get("reports", []) if isinstance(payload, dict) else []
    names = {report.get("name") for report in reports if isinstance(report, dict)}
    expected_sources = {
        "droid": (data_root / "droid_lerobot").resolve(strict=False),
        "egodex": (data_root / "egodex_cdn" / "manifest.csv").resolve(strict=False),
        "agibot": (data_root / "agibot" / "manifest.csv").resolve(strict=False),
        "abc": (data_root / "abc_pp" / "manifest.txt").resolve(strict=False),
    }
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
        and int(report.get("checked", 0)) >= expected_by_name.get(str(report.get("name")), 10**18)
        and int(report.get("selected", 0)) >= expected_by_name.get(str(report.get("name")), 10**18)
        and int(report.get("active_complete", 0)) >= expected_by_name.get(str(report.get("name")), 10**18)
        for report in reports
    )
    current_commit = run_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    validator_path = REPO_ROOT / "tools" / "validate_training_data.py"
    validator_hash = hashlib.sha256(validator_path.read_bytes()).hexdigest()
    expected_invocation = {
        "data_root": str(data_root.resolve(strict=False)),
        "datasets": ["droid", "egodex", "agibot", "abc"],
        "min_timesteps": 66,
        "validate_all": False,
        "files_only": False,
        "caps": {"droid": 10_000, "egodex": 10_000, "agibot": 10_000, "abc": 10_000},
        "expected": {"droid": 10_000, "egodex": 10_000, "agibot": 5_671, "abc": 10_000},
        "sources": {name: str(path) for name, path in expected_sources.items()},
        "agibot_roots": {
            "scr": str(agibot_root(data_root, "scr").resolve(strict=False)),
            "alpha": str(agibot_root(data_root, "alpha").resolve(strict=False)),
            "beta": str(agibot_root(data_root, "beta").resolve(strict=False)),
            "viscam": str(agibot_root(data_root, "viscam").resolve(strict=False)),
        },
    }
    invocation_ok = isinstance(payload, dict) and payload.get("invocation") == expected_invocation
    provenance_ok = bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("git_commit") == current_commit
        and payload.get("git_status") == ""
        and payload.get("validator_sha256") == validator_hash
    )
    fingerprints_ok = False
    if isinstance(payload, dict) and names == set(expected_sources):
        try:
            current_fingerprints = _current_data_fingerprints(data_root)
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
        and names == {"droid", "egodex", "agibot", "abc"}
        and sources_ok
        and content_counts_ok
        and invocation_ok
        and provenance_ok
        and fingerprints_ok
    )
    errors = sum(int(report.get("error_count", 0)) for report in reports if isinstance(report, dict))
    checked = sum(int(report.get("checked", 0)) for report in reports if isinstance(report, dict))
    results.add(
        "strict data validation",
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
    results.add("data root", data_root.is_dir(), str(data_root))
    if not data_root.is_dir():
        return
    if profile == "full":
        report_path = args.data_validation_report.resolve(strict=False) if args.data_validation_report else None
        check_strict_data_report(results, report_path, data_root, python_bin, args)
    if profile == "full":
        minimums = (args.min_droid, args.min_egodex, args.min_agibot, args.min_abc)
    else:
        minimums = (1, 1, 1, 1)
    check_droid(results, data_root, minimums[0])
    check_egodex(results, data_root, minimums[1])
    check_agibot(results, data_root, minimums[2])
    check_abc(results, data_root, minimums[3])


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
        help="consume a strict JSON report from tools/validate_training_data.py; full mode invokes it when omitted",
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
