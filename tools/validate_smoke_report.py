#!/usr/bin/env python3
"""Validate the semantic evidence required from a real-data gradient smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


VARIANTS = ("latent", "explicit", "dual-no-ztf", "dual-with-ztf")
DUAL_VARIANTS = frozenset(("dual-no-ztf", "dual-with-ztf"))
DUAL_CONDITION_ON_TF = {
    "dual-no-ztf": False,
    "dual-with-ztf": True,
}
DUAL_GRADIENT_GROUPS = {
    "action_encoder",
    "forward_model.tf_velocity_head",
    "forward_model.tf_velocity_head.linear",
    "forward_model.tf_velocity_head.norm",
    "forward_model.tf_clock_embedding",
    "forward_model.tf_clock_embedding.gate",
    "forward_model.tf_clock_embedding.net",
    "forward_model.tf_token_adapter",
    "forward_model.tf_token_adapter.gate",
    "forward_model.tf_token_adapter.projection",
    "forward_model.tf_token_adapter.norm",
}


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
    warmstart_model: str | None = None,
    warmstart_sha256: str | None = None,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported smoke variant: {variant}")
    if (warmstart_model is None) != (warmstart_sha256 is None):
        raise ValueError(
            "warm-start model and SHA-256 must either both be supplied or both omitted"
        )
    if warmstart_sha256 is not None:
        normalized_sha256 = warmstart_sha256.lower()
        if len(normalized_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_sha256
        ):
            raise ValueError("warm-start SHA-256 must contain exactly 64 hex digits")
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
    if warmstart_model is not None:
        actual = report_paths.get("warmstart_model") if isinstance(report_paths, dict) else None
        if not isinstance(actual, str) or canonical(actual) != canonical(warmstart_model):
            problems.append("warmstart_model path differs")

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
    expected_morphologies = {9} if variant in DUAL_VARIANTS else {0, 2, 6, 9}
    if morphologies != expected_morphologies:
        problems.append(f"morphology coverage is {sorted(morphologies)}")

    groups = validation.get("groups_ever_nonzero")
    required_groups = {
        "lora_",
        "forward_model.action_to_control",
        "action_pool",
        "morphology_tokens",
    }
    if variant in DUAL_VARIANTS:
        required_groups |= DUAL_GRADIENT_GROUPS
        # After decoupling the TF-head representation from the video residual,
        # the no-ZTF arm deliberately leaves the video-only state gate unused.
        # Its projection/norm must still receive TF-head gradients.
        if not DUAL_CONDITION_ON_TF[variant]:
            required_groups.discard("forward_model.tf_token_adapter.gate")
    else:
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

    if variant in DUAL_VARIANTS:
        expected_condition = DUAL_CONDITION_ON_TF[variant]
        if validation.get("condition_on_tf") is not expected_condition:
            problems.append(
                f"condition_on_tf={validation.get('condition_on_tf')!r}, "
                f"expected {expected_condition!r}"
            )
        if validation.get("sigma_convention") != "sigma=1 is noise; sigma=0 is clean data":
            problems.append("dual sigma convention evidence is missing")
        zero_init = validation.get("dual_zero_init")
        if not isinstance(zero_init, dict) or not zero_init:
            problems.append("dual zero-init evidence is missing")
        elif any(value != 0.0 for value in zero_init.values()):
            problems.append("dual TF gates/head were not exact zero-init")
        noop = validation.get("dual_video_noop")
        if not isinstance(noop, dict):
            problems.append("dual video-velocity no-op evidence is missing")
        elif (
            noop.get("exact_video_velocity_equal") is not True
            or noop.get("max_abs_difference") != 0.0
        ):
            problems.append("zero-gated Ztf changed the pre-update video velocity")
        elif (
            noop.get("production_baseline_exact_equal") is not True
            or noop.get("production_baseline_max_abs_difference") != 0.0
        ):
            problems.append(
                "zero-gated dual path changed the ordinary production Wan video velocity"
            )
        matched = validation.get("matched_trainable_tensors")
        if not isinstance(matched, dict):
            problems.append("matched trainable-tensor evidence is missing")
        else:
            unmatched = sorted(
                group
                for group in required_groups
                if not isinstance(matched.get(group), int) or matched[group] <= 0
            )
            if unmatched:
                problems.append(
                    f"required trainable parameter groups were absent: {unmatched}"
                )

    if warmstart_model is not None and warmstart_sha256 is not None:
        expected_sha256 = warmstart_sha256.lower()
        warmstart = validation.get("warmstart")
        if not isinstance(warmstart, dict):
            problems.append("warm-start load audit is missing")
        else:
            if canonical(str(warmstart.get("path", ""))) != canonical(warmstart_model):
                problems.append("warm-start audited path differs")
            if warmstart.get("sha256") != expected_sha256:
                problems.append("warm-start audited SHA-256 differs")
            if warmstart.get("model_only") is not True:
                problems.append("warm-start was not recorded as model-only")
            if warmstart.get("unexpected_keys") != []:
                problems.append("warm-start had unexpected model keys")
            identity = warmstart.get("file_identity")
            size = identity.get("size_bytes") if isinstance(identity, dict) else None
            if not isinstance(size, int) or size <= 0:
                problems.append("warm-start file identity is invalid")

    if problems:
        raise RuntimeError(f"invalid smoke report {report}: " + "; ".join(problems))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--videox-home", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--warmstart-model")
    parser.add_argument("--warmstart-sha256")
    args = parser.parse_args()
    validate(
        args.report,
        variant=args.variant,
        git_commit=args.git_commit,
        wan_dir=args.wan_dir,
        videox_home=args.videox_home,
        data_root=args.data_root,
        warmstart_model=args.warmstart_model,
        warmstart_sha256=args.warmstart_sha256,
    )
    print(f"Validated real-data gradient smoke: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
