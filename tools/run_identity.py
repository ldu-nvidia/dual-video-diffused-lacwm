#!/usr/bin/env python3
"""Create or validate immutable identity for a guarded full training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MIN_SUPPORTED_GPU_MEMORY_MIB = 78_000


def canonical(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def absolute_executable(value: str) -> str:
    # Keep the venv entrypoint path rather than resolving its ``python`` symlink
    # to a shared managed/base interpreter.
    return str(Path(value).expanduser().absolute())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def supported_gpu_memory_mib(value: str) -> int:
    parsed = positive_int(value)
    if parsed < MIN_SUPPORTED_GPU_MEMORY_MIB:
        raise argparse.ArgumentTypeError(
            "minimum GPU memory must be at least "
            f"{MIN_SUPPORTED_GPU_MEMORY_MIB} MiB"
        )
    return parsed


def expected_payload(args: argparse.Namespace) -> dict:
    data = json.loads(args.data_report.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    reports = data.get("reports", [])
    fingerprints = {
        str(report.get("name")): report.get("fingerprint")
        for report in reports
        if isinstance(report, dict)
    }
    if set(fingerprints) != {"droid", "egodex", "agibot", "abc"}:
        raise RuntimeError("data report does not contain all four fingerprints")
    effective_global_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * 8
    )
    payload = {
        "schema_version": 3,
        "variant": args.variant,
        "git_commit": args.git_commit,
        # ``batch_size`` is the physical per-rank microbatch.  Keep all three
        # batching values explicit so a resume cannot silently alter optimizer
        # semantics while retaining the same nominal update count.
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": 8,
        "effective_global_batch_size": effective_global_batch_size,
        "gpu_profile": {
            "model": "B200",
            "minimum_memory_mib": args.min_gpu_memory_mib,
        },
        "run_name": args.run_name,
        "paths": {
            "python": absolute_executable(args.python),
            "wan_dir": canonical(args.wan_dir),
            "videox_home": canonical(args.videox_home),
            "data_root": canonical(args.data_root),
            "run_root": canonical(args.run_root),
            "run_dir": canonical(args.run_dir),
        },
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
        },
        "data": {
            "fingerprints": fingerprints,
            "invocation": data.get("invocation"),
            "validator_sha256": data.get("validator_sha256"),
        },
        "runtime": {
            "python": runtime.get("python"),
            "packages": runtime.get("packages"),
            "distributions": runtime.get("distributions"),
            "environment": runtime.get("environment"),
            "videox_commit": runtime.get("videox_commit"),
            "videox_status": runtime.get("videox_status"),
            "weights": runtime.get("weights"),
        },
        "gradient_smoke": {
            "sha256": sha256(args.smoke_report),
            "kind": smoke.get("kind"),
            "variant": smoke.get("variant"),
        },
    }
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(canonical_json).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "validate"))
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--variant", choices=("latent", "explicit"), required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--batch-size", type=positive_int, required=True)
    parser.add_argument(
        "--gradient-accumulation-steps", type=positive_int, required=True
    )
    parser.add_argument(
        "--min-gpu-memory-mib", type=supported_gpu_memory_mib, required=True
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--videox-home", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wandb-mode", required=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    args = parser.parse_args(argv)

    expected = expected_payload(args)
    if args.action == "create":
        if args.identity.exists():
            raise RuntimeError(f"refusing to replace existing identity: {args.identity}")
        payload = dict(expected)
        payload["state"] = "prelaunch"
        temporary = args.identity.with_suffix(args.identity.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.identity)
        print(f"Created run identity: {args.identity}")
        return 0

    actual = json.loads(args.identity.read_text(encoding="utf-8"))
    problems = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if problems:
        raise RuntimeError(f"run identity mismatch for fields: {problems}")
    print(f"Validated run identity: {args.identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
