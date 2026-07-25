#!/usr/bin/env python3
"""Capture reproducibility metadata for a guarded lacwm_dit launch."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "LACWM_DATA",
    "LACWM_RUNS",
    "MPLCONFIGDIR",
    "PYTHONPATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TMPDIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "VIDEOX_HOME",
    "WANDB_CACHE_DIR",
    "WANDB_CONFIG_DIR",
    "WANDB_DATA_DIR",
    "WANDB_DIR",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_PROJECT",
    "WAN_DIR",
    "XDG_CACHE_HOME",
)


def command_output(command: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
        output = completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        return completed.returncode, output.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def write_command_output(path: Path, command: list[str], timeout: int = 60) -> dict[str, object]:
    return_code, output = command_output(command, timeout=timeout)
    path.write_text(output + ("\n" if output else ""))
    return {"command": command, "return_code": return_code, "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--kind", choices=("gradient-smoke", "full-8xb200"), required=True)
    parser.add_argument(
        "--variant",
        choices=("latent", "explicit", "dual-no-ztf", "dual-with-ztf"),
        required=True,
    )
    parser.add_argument("--data-mode", choices=("real", "synthetic"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    artifacts.append(
        write_command_output(
            args.output_dir / "git_status.txt",
            ["git", "-C", str(REPO_ROOT), "status", "--short", "--branch"],
        )
    )
    artifacts.append(
        write_command_output(
            args.output_dir / "git_diff_stat.txt",
            ["git", "-C", str(REPO_ROOT), "diff", "--stat", "--", "."],
        )
    )
    artifacts.append(
        write_command_output(
            args.output_dir / "nvidia_smi.txt",
            ["nvidia-smi", "-q"],
        )
    )
    artifacts.append(
        write_command_output(
            args.output_dir / "nvidia_topology.txt",
            ["nvidia-smi", "topo", "-m"],
        )
    )
    # Canonical training environments may be created by uv and intentionally have no
    # pip module. importlib.metadata is part of Python and inventories either layout.
    package_probe = "\n".join(
        (
            "from importlib import metadata",
            "rows = []",
            "for dist in metadata.distributions():",
            "    name = dist.metadata.get('Name') or '<unnamed>'",
            "    rows.append((name.lower(), name, dist.version))",
            "for _, name, version in sorted(rows):",
            "    print(f'{name}=={version}')",
        )
    )
    artifacts.append(
        write_command_output(
            args.output_dir / "python_packages.txt",
            [str(args.python), "-c", package_probe],
            timeout=120,
        )
    )
    revision_code, revision = command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "variant": args.variant,
        "data_mode": args.data_mode,
        "run_name": args.run_name,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": str(args.python),
        "repo_root": str(REPO_ROOT),
        "git_commit": revision if revision_code == 0 else None,
        "command_file": str(args.command_file),
        "environment": {key: os.environ.get(key) for key in SAFE_ENV_KEYS},
        "artifacts": artifacts,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
