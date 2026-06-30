#!/usr/bin/env python3
"""Validate the semantic evidence required from a real-data gradient smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def canonical(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def validate(
    report: Path,
    *,
    variant: str,
    git_commit: str,
    wan_dir: str,
    videox_home: str,
    data_root: str,
) -> dict:
    payload = json.loads(report.read_text(encoding="utf-8"))
    problems: list[str] = []
    expected_top = {
        "schema_version": 1,
        "kind": "lacwm_gradient_smoke_real",
        "data_mode": "real",
        "status": "passed",
        "variant": variant,
        "git_commit": git_commit,
        "git_status": "",
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            problems.append(f"{key}={payload.get(key)!r}, expected {expected!r}")

    report_paths = payload.get("paths", {})
    for key, expected in (
        ("wan_dir", canonical(wan_dir)),
        ("videox_home", canonical(videox_home)),
        ("data_root", canonical(data_root)),
    ):
        actual = report_paths.get(key) if isinstance(report_paths, dict) else None
        if not isinstance(actual, str) or canonical(actual) != expected:
            problems.append(f"{key} path differs")

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        problems.append("validation payload is missing")
        validation = {}
    trainable = validation.get("trainable_parameters")
    if not isinstance(trainable, int) or trainable <= 0:
        problems.append("trainable parameter count is invalid")
    gpu = validation.get("gpu", {})
    memory = gpu.get("max_memory_allocated_gib") if isinstance(gpu, dict) else None
    if not isinstance(memory, (int, float)) or not math.isfinite(memory) or memory <= 0:
        problems.append("GPU allocation evidence is invalid")

    steps = validation.get("steps")
    if not isinstance(steps, list) or len(steps) < 4:
        problems.append("fewer than four gradient steps were recorded")
        steps = []
    morphologies = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            problems.append(f"step {index} is malformed")
            continue
        loss = step.get("loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss <= 0:
            problems.append(f"step {index} has invalid loss")
        morphology = step.get("morphology_index")
        if isinstance(morphology, int):
            morphologies.add(morphology)
    if morphologies != {0, 2, 6, 9}:
        problems.append(f"morphology coverage is {sorted(morphologies)}")

    groups = validation.get("groups_ever_nonzero")
    required_groups = {
        "lora_",
        "forward_model.action_to_control",
        "action_pool",
        "morphology_tokens",
    }
    required_groups |= (
        {"action_encoder"}
        if variant == "explicit"
        else {"inverse_model", "action_decoder"}
    )
    if not isinstance(groups, dict):
        problems.append("gradient-group evidence is missing")
    else:
        missing = sorted(group for group in required_groups if groups.get(group) is not True)
        if missing:
            problems.append(f"required gradient groups were not nonzero: {missing}")

    if problems:
        raise RuntimeError(f"invalid smoke report {report}: " + "; ".join(problems))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=("latent", "explicit"), required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--videox-home", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    validate(
        args.report,
        variant=args.variant,
        git_commit=args.git_commit,
        wan_dir=args.wan_dir,
        videox_home=args.videox_home,
        data_root=args.data_root,
    )
    print(f"Validated real-data gradient smoke: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
