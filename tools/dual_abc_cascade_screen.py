#!/usr/bin/env python3
"""Provenance and contract checks for the strict TF-first ABC cascade screen.

This helper is intentionally separate from ``dual_abc_pilot.py`` and the
overlapping-clock screens.  All three arms use the same strict two-phase
schedule and orthonormal Parseval RFFT state.  They differ only in whether the
video branch receives no TF content, matched TF content, or globally deranged
future TF content with the local corruption noise and observed history held
fixed.  Oracle sources are hidden-future leakage diagnostics, never deployable
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dual_abc_pilot as pilot


EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
REQUESTED_VIEWER_EMAIL = "ldu@nvidia.edu"
CONFIG_SELECTOR = "ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml"
EVALUATION_NFE_STEPS = [2, 4, 8]
EVALUATION_NOISE_SEED = 20260726
EVALUATION_CONDITION_SOURCES = [
    "autonomous",
    "off",
    "oracle_matched",
    "oracle_shuffled",
]
OPTIMIZER_UPDATES = 200
WARMUP_UPDATES = 20
VISUALIZATION_UPDATES = (0, 50, 100, 150, 199)
ARRAY_JOB_ID_RE = re.compile(r"^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$")
NUMERIC_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
REPRESENTATION = "parseval_rfft"
SCHEDULE_MODE = "tf_first_cascaded"
CASCADE_TF_LOSS_PROBABILITY = 0.4
CASCADE_LOGIT_MEAN = 0.0
CASCADE_LOGIT_STD = 1.0
CASCADE_TF_CONDITION_MAX_SIGMA = 0.25
CASCADE_VALIDATION_TF_SIGMA = 0.125
CASCADE_INFERENCE_TF_FRACTION = 0.5
CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES = True
VALIDATION_VIDEO_SIGMAS = [0.90, 0.75, 0.50, 0.25]
CONDITION_MODE_CODES = {"off": 0, "matched": 1, "shuffled": 2}
EVALUATION_CONDITION_SOURCE_CODES = {
    source: index for index, source in enumerate(EVALUATION_CONDITION_SOURCES)
}

# Array order is part of the immutable experimental contract.
ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "cascade_off_s000",
        "representation": REPRESENTATION,
        "condition_on_tf": False,
        "condition_mode": "off",
        "state_gate_init": 0.0,
        "causal_role": "tf_content_inert_control",
        "smoke_variant": "dual-no-ztf",
        "smoke_report_key": "no_ztf",
    },
    {
        "name": "cascade_matched_s010",
        "representation": REPRESENTATION,
        "condition_on_tf": True,
        "condition_mode": "matched",
        "state_gate_init": 0.10,
        "causal_role": "matched_tf_content",
        "smoke_variant": "dual-with-ztf",
        "smoke_report_key": "with_ztf",
    },
    {
        "name": "cascade_shuffled_s010",
        "representation": REPRESENTATION,
        "condition_on_tf": True,
        "condition_mode": "shuffled",
        "state_gate_init": 0.10,
        "causal_role": "wrong_future_tf_content_control",
        "smoke_variant": "dual-with-ztf",
        "smoke_report_key": "with_ztf",
    },
)


def _arm(task_id: int) -> dict[str, Any]:
    if task_id < 0 or task_id >= len(ARMS):
        raise ValueError(f"array task ID must be in [0, {len(ARMS) - 1}]")
    return ARMS[task_id]


def _identity_is_valid(payload: dict[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest() == recorded


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _wandb_private_project(entity: str, project: str) -> dict[str, Any]:
    """Record the requested-email deviation after strict private-owner checks."""
    summary = dict(pilot._wandb_private_project(entity, project))
    viewer_email = summary.get("viewer_email", "").strip().lower()
    summary["requested_viewer_email"] = REQUESTED_VIEWER_EMAIL
    summary["viewer_email_matches_request"] = (
        viewer_email == REQUESTED_VIEWER_EMAIL
    )
    summary["identity_deviation"] = (
        None
        if summary["viewer_email_matches_request"]
        else (
            "authenticated private personal entity owner uses "
            f"{viewer_email!r}, not requested {REQUESTED_VIEWER_EMAIL!r}"
        )
    )
    return summary


def _active_job_contract(
    allowed_values: Sequence[str],
    observed_values: Sequence[str],
) -> dict[str, Any]:
    """Validate exact numeric coexistence scopes captured immediately pre-submit."""
    allowed = list(allowed_values)
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed active job IDs must be unique")
    if any(NUMERIC_JOB_ID_RE.fullmatch(value) is None for value in allowed):
        raise ValueError(
            "allowed active job IDs must be positive numeric Slurm IDs"
        )

    observed: list[dict[str, str]] = []
    seen_display_ids: set[str] = set()
    for value in observed_values:
        if "\n" in value or "\r" in value:
            raise ValueError("observed active job record contains a newline")
        fields = value.split("|")
        if len(fields) != 3:
            raise ValueError(
                "observed active job must be BASE_ID|DISPLAY_ID|STATE"
            )
        base_id, display_id, state = fields
        if NUMERIC_JOB_ID_RE.fullmatch(base_id) is None:
            raise ValueError("observed active job base ID must be numeric")
        if (
            not display_id
            or "|" in display_id
            or any(character.isspace() for character in display_id)
        ):
            raise ValueError("observed active job display ID is unsafe")
        if not state or not state.replace("_", "").isalpha() or state != state.upper():
            raise ValueError("observed active job state is malformed")
        if display_id in seen_display_ids:
            raise ValueError(f"duplicate observed active job: {display_id}")
        seen_display_ids.add(display_id)
        observed.append(
            {
                "array_or_job_id": base_id,
                "display_job_id": display_id,
                "state": state,
            }
        )

    allowed_set = set(allowed)
    observed_ids = {record["array_or_job_id"] for record in observed}
    unallowed = sorted(observed_ids - allowed_set, key=int)
    unused = sorted(allowed_set - observed_ids, key=int)
    if unallowed:
        raise RuntimeError(
            "active Slurm job IDs were not explicitly allowed: "
            + ", ".join(unallowed)
        )
    if unused:
        raise RuntimeError(
            "allowed Slurm job IDs were not active at the pre-submit check: "
            + ", ".join(unused)
        )
    return {
        "default_policy": "fail_closed_on_any_unlisted_active_user_job",
        "allow_scope": "exact_positive_numeric_array_or_job_IDs_only",
        "wildcards_or_names_allowed": False,
        "allowed_array_or_job_ids": sorted(allowed, key=int),
        "observed_active_jobs": sorted(
            observed,
            key=lambda record: (
                int(record["array_or_job_id"]),
                record["display_job_id"],
            ),
        ),
        "all_observed_jobs_explicitly_allowed": True,
    }


def _report_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "no_ztf": pilot._canonical_regular_file(
            args.no_ztf_smoke_report, "no-ZTF gradient smoke report"
        ),
        "with_ztf": pilot._canonical_regular_file(
            args.with_ztf_smoke_report, "with-ZTF gradient smoke report"
        ),
    }


def _arm_manifest_contract() -> dict[str, dict[str, Any]]:
    return {
        str(task_id): {
            "array_task_id": task_id,
            "name": arm["name"],
            "representation": arm["representation"],
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
            "condition_only_video_loss_examples": (
                CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES
            ),
            "causal_role": arm["causal_role"],
            "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
            "evaluation_noise_seed": EVALUATION_NOISE_SEED,
            "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
            "smoke_variant": arm["smoke_variant"],
        }
        for task_id, arm in enumerate(ARMS)
    }


def command_arm_contract(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    payload = {
        "array_task_id": args.array_task_id,
        **arm,
        "state_gate_trainable": False,
        "condition_only_video_loss_examples": (
            CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES
        ),
        "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
        "evaluation_noise_seed": EVALUATION_NOISE_SEED,
        "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
    }
    if args.format == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        fields = (
            payload["name"],
            payload["representation"],
            str(payload["condition_on_tf"]).lower(),
            payload["condition_mode"],
            f"{payload['state_gate_init']:.2f}",
            payload["causal_role"],
            payload["smoke_variant"],
            payload["smoke_report_key"],
        )
        if any("\t" in str(field) or "\n" in str(field) for field in fields):
            raise RuntimeError("arm contract contains a shell-unsafe delimiter")
        print("\t".join(str(field) for field in fields))
    return 0


def command_wandb_private(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            _wandb_private_project(args.entity, args.project),
            sort_keys=True,
        )
    )
    return 0


def command_cascade_contract_smoke(args: argparse.Namespace) -> int:
    """Exercise the exact checked-out cascade and Parseval contracts on CPU."""
    import torch

    from robot_wm.modeling.dual_diffusion import (
        DualClockSampler,
        PerViewCausalRFFT,
        pair_native_cascaded_sigma_schedule,
    )

    repo_root = pilot._canonical_directory(args.repo_root, "repository root")
    expected_commit = pilot._validated_commit(args.git_commit)
    pilot._assert_clean_commit(repo_root, expected_commit)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    video = torch.randn(2, 13, 3, 8, 24, generator=generator)
    transform = PerViewCausalRFFT(
        num_views=3,
        output_size=(8, 24),
        window_size=4,
        pad_multiple=None,
        normalization="none",
        representation=REPRESENTATION,
    )
    coefficients = transform(video)
    reconstructed = transform.inverse(coefficients)
    reconstruction_max_abs = float((reconstructed - video).abs().max())
    if coefficients.shape != (2, 12, 4, 8, 24):
        raise RuntimeError(
            f"Parseval RFFT smoke shape differs: {tuple(coefficients.shape)}"
        )
    if reconstruction_max_abs > 2e-6:
        raise RuntimeError(
            "Parseval RFFT round-trip exceeded tolerance: "
            f"{reconstruction_max_abs}"
        )

    native_video_sigmas = torch.tensor([1.0, 0.6, 0.0])
    schedule = pair_native_cascaded_sigma_schedule(
        native_video_sigmas,
        total_steps=4,
        tf_fraction=CASCADE_INFERENCE_TF_FRACTION,
    )
    expected_video = torch.tensor([1.0, 1.0, 1.0, 0.6, 0.0])
    expected_tf = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0])
    if not torch.equal(schedule.video, expected_video) or not torch.equal(
        schedule.time_frequency, expected_tf
    ):
        raise RuntimeError("strict cascade did not preserve the native video nodes")

    native_batch = torch.tensor([0.9, 0.4, 0.1])
    tf_branch = DualClockSampler(
        mode="tf_first_cascaded_noised",
        logit_mean=CASCADE_LOGIT_MEAN,
        logit_std=CASCADE_LOGIT_STD,
        tf_loss_probability=1.0,
        tf_condition_max_sigma=CASCADE_TF_CONDITION_MAX_SIGMA,
    )(
        native_batch.numel(),
        device="cpu",
        generator=generator,
        native_video_sigma=native_batch,
    )
    video_branch = DualClockSampler(
        mode="tf_first_cascaded_noised",
        logit_mean=CASCADE_LOGIT_MEAN,
        logit_std=CASCADE_LOGIT_STD,
        tf_loss_probability=0.0,
        tf_condition_max_sigma=CASCADE_TF_CONDITION_MAX_SIGMA,
    )(
        native_batch.numel(),
        device="cpu",
        generator=generator,
        native_video_sigma=native_batch,
    )
    if not (
        torch.equal(tf_branch.video_sigma, torch.ones_like(native_batch))
        and torch.equal(tf_branch.video_loss_weight, torch.zeros_like(native_batch))
        and torch.equal(tf_branch.tf_loss_weight, torch.ones_like(native_batch))
    ):
        raise RuntimeError("TF-loss branch did not freeze video at pure noise")
    if not (
        torch.equal(video_branch.video_sigma, native_batch)
        and torch.equal(video_branch.video_loss_weight, torch.ones_like(native_batch))
        and torch.equal(video_branch.tf_loss_weight, torch.zeros_like(native_batch))
        and bool(
            (
                (video_branch.tf_sigma >= 0)
                & (
                    video_branch.tf_sigma
                    <= CASCADE_TF_CONDITION_MAX_SIGMA
                )
            ).all()
        )
    ):
        raise RuntimeError(
            "video-loss branch violated its native-video/imperfect-TF contract"
        )

    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_tf_first_cascade_contract_smoke",
            "created_at_utc": pilot._now(),
            "status": "passed",
            "git_commit": expected_commit,
            "git_status": "",
            "repository_root": str(repo_root),
            "seed": 1234,
            "representation": REPRESENTATION,
            "transform": {
                "input_shape": list(video.shape),
                "output_shape": list(coefficients.shape),
                "roundtrip_max_abs": reconstruction_max_abs,
                "per_view": True,
            },
            "diffusion_clock": {
                "convention": "sigma=1 noise, sigma=0 clean",
                "mode": SCHEDULE_MODE,
                "native_video_nodes": native_video_sigmas.tolist(),
                "schedule_video": schedule.video.tolist(),
                "schedule_time_frequency": schedule.time_frequency.tolist(),
                "tf_loss_branch_video_is_pure_noise": True,
                "video_loss_branch_uses_native_video_sigma": True,
                "video_loss_branch_tf_sigma_max": (
                    CASCADE_TF_CONDITION_MAX_SIGMA
                ),
            },
            "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
        }
    )
    output = Path(args.output)
    if not output.is_absolute():
        raise ValueError("cascade contract output must be absolute")
    output.parent.resolve(strict=True)
    pilot._exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def _validate_cascade_contract_smoke(
    path: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    payload = _read_json_file(path, "cascade contract smoke")
    expected = {
        "kind": "dual_abc_tf_first_cascade_contract_smoke",
        "status": "passed",
        "git_commit": expected_commit,
        "git_status": "",
        "representation": REPRESENTATION,
        "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
    }
    problems = [
        f"{key}: {payload.get(key)!r} != {wanted!r}"
        for key, wanted in expected.items()
        if payload.get(key) != wanted
    ]
    clock = payload.get("diffusion_clock", {})
    if clock.get("convention") != "sigma=1 noise, sigma=0 clean":
        problems.append("sigma convention differs")
    if clock.get("mode") != SCHEDULE_MODE:
        problems.append("schedule mode differs")
    if clock.get("tf_loss_branch_video_is_pure_noise") is not True:
        problems.append("TF-loss branch smoke did not freeze video")
    if clock.get("video_loss_branch_uses_native_video_sigma") is not True:
        problems.append("video-loss branch smoke did not use native Wan sigmas")
    if not _identity_is_valid(payload):
        problems.append("identity SHA-256 is invalid")
    if problems:
        raise RuntimeError(
            "invalid cascade contract smoke: " + "; ".join(problems)
        )
    return payload


def command_create_screen(args: argparse.Namespace) -> int:
    screen_id = pilot._validated_id(args.screen_id, "screen ID")
    expected_commit = pilot._validated_commit(args.git_commit)
    repo_root = pilot._canonical_directory(args.repo_root, "repository root")
    run_root = pilot._canonical_directory(args.run_root, "run root")
    screen_root = pilot._canonical_directory(args.screen_root, "screen root")
    if screen_root.parent != run_root:
        raise ValueError(
            f"screen root must be a direct child of run root: {screen_root}"
        )
    if screen_root.name != screen_id:
        raise ValueError(
            f"screen root basename {screen_root.name!r} does not match "
            f"screen ID {screen_id!r}"
        )
    pilot._assert_clean_commit(repo_root, expected_commit)

    checkpoint = pilot._canonical_regular_file(
        args.checkpoint, "warm-start checkpoint"
    )
    checkpoint_summary = pilot._verify_checkpoint(
        checkpoint, args.checkpoint_sha256
    )
    data_root = pilot._canonical_directory(args.data_root, "data root")
    python = pilot._python_executable(args.python)
    wan_dir = pilot._canonical_directory(args.wan_dir, "Wan directory")
    videox_home = pilot._canonical_directory(
        args.videox_home, "VideoX-Fun checkout"
    )
    wandb_summary = _wandb_private_project(
        args.wandb_entity, args.wandb_project
    )
    common_config = pilot._canonical_regular_file(
        args.common_config, "common experiment config"
    )
    arm_config = pilot._canonical_regular_file(
        args.arm_config, "base arm experiment config"
    )
    cascade_helper = pilot._canonical_regular_file(
        str(repo_root / "tools" / "dual_abc_cascade_screen.py"),
        "cascade screen helper",
    )
    smoke_paths = _report_paths(args)
    active_job_coexistence = _active_job_contract(
        args.allow_active_job_id,
        args.observed_active_job,
    )
    max_concurrent_arms = int(args.max_concurrent_arms)
    if max_concurrent_arms < 1 or max_concurrent_arms > len(ARMS):
        raise ValueError(
            f"max concurrent arms must be between 1 and {len(ARMS)}"
        )

    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_tf_first_cascade_screen",
            "created_at_utc": pilot._now(),
            "screen_id": screen_id,
            "git_commit": expected_commit,
            "repository_root": str(repo_root),
            "paths": {
                "python": str(python),
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
                "data_root": str(data_root),
                "run_root": str(run_root),
                "screen_root": str(screen_root),
            },
            "checkpoint": checkpoint_summary,
            "data": {
                "datasets": ["ABC"],
                "abc_manifest": pilot._abc_manifest_summary(data_root),
            },
            "config": {
                "common": {
                    "path": str(common_config),
                    "sha256": pilot._sha256(common_config),
                },
                "base_arm": {
                    "selector": CONFIG_SELECTOR,
                    "path": str(arm_config),
                    "sha256": pilot._sha256(arm_config),
                },
                "arms": _arm_manifest_contract(),
                "controlled_factors": [
                    "model.dual_diffusion.condition_on_tf",
                    "model.dual_diffusion.condition_mode",
                    "model.dual_diffusion.state_gate_init",
                ],
                "matched_factors": [
                    "source_commit",
                    "warm_start",
                    "ABC_manifest",
                    "seed",
                    "optimizer",
                    "strict_cascade_training_schedule",
                    "TF_content_injection_only_on_video_loss_examples",
                    "native_Wan_video_sigma_index_law",
                    "Parseval_RFFT_representation",
                    "evaluation_noise",
                    "NFE_schedules",
                ],
            },
            "gradient_smoke_reports": {
                key: {
                    "path": str(path),
                    "sha256": pilot._sha256(path),
                    "variant": (
                        "dual-no-ztf" if key == "no_ztf" else "dual-with-ztf"
                    ),
                    "data_mode": "real",
                    "warmstart_sha256": args.checkpoint_sha256,
                    "configuration_scope": (
                        "ordinary raw-RFFT/tf-leads compatibility smoke; "
                        "not a full strict Parseval cascade GPU smoke"
                    ),
                }
                for key, path in smoke_paths.items()
            },
            "cascade_contract_smoke": {
                "required_per_arm": True,
                "kind": "dual_abc_tf_first_cascade_contract_smoke",
                "tool_path": str(cascade_helper),
                "tool_sha256": pilot._sha256(cascade_helper),
                "git_commit": expected_commit,
            },
            "schedule": {
                "seed": 1234,
                "optimizer_updates": OPTIMIZER_UPDATES,
                "warmup_updates": WARMUP_UPDATES,
                "batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "gpus_per_arm": 8,
                "effective_global_batch_size_per_arm": 8,
                "log_every": 5,
                "validate_every": 10,
                "save_every": 50,
                "visualize_every": 50,
                "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
                "evaluation_noise_seed": EVALUATION_NOISE_SEED,
                "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
            },
            "diffusion_clock": {
                "convention": "sigma=1 noise, sigma=0 clean",
                "mode": SCHEDULE_MODE,
                "training": {
                    "branch_selection": (
                        "one active loss per example; TF-loss examples hold video "
                        "at sigma=1; video-loss examples draw the exact native Wan "
                        "schedule index and condition on TF sigma in [0,0.25]"
                    ),
                    "tf_loss_probability": CASCADE_TF_LOSS_PROBABILITY,
                    "condition_only_video_loss_examples": (
                        CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES
                    ),
                    "native_video_logit_mean": CASCADE_LOGIT_MEAN,
                    "native_video_logit_std": CASCADE_LOGIT_STD,
                    "tf_condition_max_sigma": CASCADE_TF_CONDITION_MAX_SIGMA,
                    "validation_tf_sigma": CASCADE_VALIDATION_TF_SIGMA,
                    "validation_video_sigmas": VALIDATION_VIDEO_SIGMAS,
                },
                "inference": {
                    "tf_fraction": CASCADE_INFERENCE_TF_FRACTION,
                    "nfe_is_total_model_calls": True,
                    "tf_phase": "video fixed at sigma=1",
                    "video_phase": "TF fixed at sigma=0; exact native Wan nodes",
                },
            },
            "research_question": (
                "After a strict Parseval-RFFT TF-first phase, does aligned TF "
                "content improve low-NFE video denoising relative to both an "
                "exact TF-content-off path and a same-marginal shuffled-content "
                "control?"
            ),
            "causal_interpretation": {
                "off_control_scope": (
                    "exactly removes TF state content from the video trunk by "
                    "condition_on_tf=false and a fixed zero state gate; the "
                    "schedule-only TF clock and auxiliary TF loss allocation/"
                    "objective remain matched, so this is not a production "
                    "single-state baseline"
                ),
                "matched_vs_shuffled": (
                    "changes future-content alignment only on video-loss examples "
                    "at identical state scale; shuffling preserves observed "
                    "history, local corruption noise, and TF sigma. TF-loss "
                    "examples receive an exact zero content mask, preventing a "
                    "direct condition-mode intervention on their forward/loss; "
                    "downstream shared-parameter mediation remains part of the "
                    "treatment effect"
                ),
                "oracle_sources_are_leakage": True,
                "oracle_policy": (
                    "oracle_matched and oracle_shuffled consume hidden-future TF "
                    "targets and diagnose mechanism only; they are nondeployable"
                ),
            },
            "wandb": {
                **wandb_summary,
                "group": None,
            },
            "active_job_coexistence": active_job_coexistence,
            "slurm": {
                "nodes_per_arm": 1,
                "gpus_per_node": 8,
                "array": f"0-{len(ARMS) - 1}%{max_concurrent_arms}",
                "requeue": False,
            },
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != screen_root:
        raise ValueError(
            f"screen manifest must be written directly under {screen_root}"
        )
    pilot._exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def _config_value(config: Any, dotted: str) -> Any:
    return pilot._config_value(config, dotted)


def _assert_equal(
    problems: list[str],
    config: Any,
    dotted: str,
    wanted: Any,
) -> None:
    actual = _config_value(config, dotted)
    if actual != wanted:
        problems.append(f"{dotted}: {actual!r} != {wanted!r}")


def _assert_float(
    problems: list[str],
    config: Any,
    dotted: str,
    wanted: float,
) -> None:
    actual = _config_value(config, dotted)
    try:
        matches = math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        problems.append(f"{dotted}: {actual!r} != {wanted!r}")


def _assert_resolved_contract(
    content: bytes,
    *,
    arm: dict[str, Any],
    checkpoint: Path,
    run_dir: Path,
    wandb_run_id: str,
) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.create(content.decode())
    problems: list[str] = []
    expected = {
        "name": wandb_run_id,
        "seed": 1234,
        "data_loader.batch_size": 1,
        "trainer.config.max_iter": OPTIMIZER_UPDATES,
        "trainer.config.gradient_accumulation_steps": 1,
        "trainer.config.load_path": str(checkpoint),
        "trainer.config.logging.log_every": 5,
        "trainer.config.saving.save_every": 50,
        "trainer.config.validation.val_every": 10,
        "trainer.config.visualization.viz_every": 50,
        "trainer.config.visualization.require_success": True,
        "model.viz_num_steps": 8,
        "model.num_history_frames": 5,
        "model.num_future_frames": 8,
        "lr_scheduler_factory.lr_lambda.warmup_steps": WARMUP_UPDATES,
        "lr_scheduler_factory.lr_lambda.total_steps": OPTIMIZER_UPDATES,
        "optimizer_factory.lr": 1e-4,
        "optimizer_factory.betas": [0.9, 0.95],
        "model.dual_diffusion.enabled": True,
        "model.dual_diffusion.condition_on_tf": arm["condition_on_tf"],
        "model.dual_diffusion.condition_mode": arm["condition_mode"],
        "model.dual_diffusion.state_gate_trainable": False,
        "model.dual_diffusion.evaluation_nfe_steps": EVALUATION_NFE_STEPS,
        "model.dual_diffusion.evaluation_noise_seed": EVALUATION_NOISE_SEED,
        "model.dual_diffusion.evaluation_condition_sources": (
            EVALUATION_CONDITION_SOURCES
        ),
        "model.dual_diffusion.schedule_mode": SCHEDULE_MODE,
        "model.dual_diffusion.cascade_tf_loss_probability": (
            CASCADE_TF_LOSS_PROBABILITY
        ),
        "model.dual_diffusion.cascade_logit_mean": CASCADE_LOGIT_MEAN,
        "model.dual_diffusion.cascade_logit_std": CASCADE_LOGIT_STD,
        "model.dual_diffusion.cascade_tf_condition_max_sigma": (
            CASCADE_TF_CONDITION_MAX_SIGMA
        ),
        "model.dual_diffusion.cascade_validation_tf_sigma": (
            CASCADE_VALIDATION_TF_SIGMA
        ),
        "model.dual_diffusion.cascade_inference_tf_fraction": (
            CASCADE_INFERENCE_TF_FRACTION
        ),
        "model.dual_diffusion.cascade_condition_only_video_loss_examples": (
            CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES
        ),
        "model.dual_diffusion.validation_video_sigmas": (
            VALIDATION_VIDEO_SIGMAS
        ),
        "model.dual_diffusion.tf_loss_weight": 1.0,
        "model.dual_diffusion.tf_channels": 12,
        "model.dual_diffusion.capture_latent_trajectories": True,
        "model.forward_model.dual_diffusion.condition_on_tf": arm[
            "condition_on_tf"
        ],
        "model.forward_model.dual_diffusion.condition_mode": arm[
            "condition_mode"
        ],
        "model.forward_model.dual_diffusion.state_gate_trainable": False,
        "model.forward_model.dual_diffusion.evaluation_nfe_steps": (
            EVALUATION_NFE_STEPS
        ),
        "model.forward_model.dual_diffusion.evaluation_noise_seed": (
            EVALUATION_NOISE_SEED
        ),
        "model.forward_model.dual_diffusion.evaluation_condition_sources": (
            EVALUATION_CONDITION_SOURCES
        ),
        "model.forward_model.dual_diffusion.schedule_mode": SCHEDULE_MODE,
        "model.forward_model.dual_diffusion.cascade_tf_loss_probability": (
            CASCADE_TF_LOSS_PROBABILITY
        ),
        "model.forward_model.dual_diffusion.cascade_logit_mean": (
            CASCADE_LOGIT_MEAN
        ),
        "model.forward_model.dual_diffusion.cascade_logit_std": (
            CASCADE_LOGIT_STD
        ),
        "model.forward_model.dual_diffusion.cascade_tf_condition_max_sigma": (
            CASCADE_TF_CONDITION_MAX_SIGMA
        ),
        "model.forward_model.dual_diffusion.cascade_validation_tf_sigma": (
            CASCADE_VALIDATION_TF_SIGMA
        ),
        "model.forward_model.dual_diffusion.cascade_inference_tf_fraction": (
            CASCADE_INFERENCE_TF_FRACTION
        ),
        (
            "model.forward_model.dual_diffusion."
            "cascade_condition_only_video_loss_examples"
        ): CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES,
        "model.forward_model.dual_diffusion.validation_video_sigmas": (
            VALIDATION_VIDEO_SIGMAS
        ),
        "model.forward_model.dual_diffusion.tf_loss_weight": 1.0,
        "model.forward_model.dual_diffusion.tf_channels": 12,
        "model.forward_model.dual_diffusion.capture_latent_trajectories": True,
        "model.time_frequency_transform.num_views": 3,
        "model.time_frequency_transform.output_size": [24, 120],
        "model.time_frequency_transform.window_size": 4,
        "model.time_frequency_transform.pad_multiple": 16,
        "model.time_frequency_transform.pad_value": -1.0,
        "model.time_frequency_transform.normalization": "none",
        "model.time_frequency_transform.representation": REPRESENTATION,
        "wandb.enabled": True,
        "wandb.mode": "online",
        "wandb.entity": EXPECTED_ENTITY,
        "wandb.project": EXPECTED_PROJECT,
        "wandb.id": wandb_run_id,
        "wandb.group": None,
        "wandb.resume": "never",
    }
    for dotted, wanted in expected.items():
        _assert_equal(problems, config, dotted, wanted)
    _assert_float(
        problems,
        config,
        "model.dual_diffusion.state_gate_init",
        float(arm["state_gate_init"]),
    )
    _assert_float(
        problems,
        config,
        "model.forward_model.dual_diffusion.state_gate_init",
        float(arm["state_gate_init"]),
    )

    for loader in ("dataset", "val_dataset", "viz_dataset"):
        names = list(_config_value(config, f"{loader}.datasets").keys())
        if names != ["ABC"]:
            problems.append(f"{loader}.datasets: {names!r} != ['ABC']")

    save_path = Path(str(_config_value(config, "trainer.config.saving.save_path")))
    viz_path = Path(
        str(_config_value(config, "trainer.config.visualization.viz_path"))
    )
    if save_path != run_dir / "snapshot.pt":
        problems.append(f"saving.save_path: {save_path} != {run_dir / 'snapshot.pt'}")
    if viz_path != run_dir / "visualization":
        problems.append(
            f"visualization.viz_path: {viz_path} != {run_dir / 'visualization'}"
        )

    tags = list(_config_value(config, "wandb.tags"))
    for expected_tag in (
        "ztf-first-cascade-screen",
        REPRESENTATION,
        str(arm["name"]),
        "seed-1234",
    ):
        if expected_tag not in tags:
            problems.append(
                f"wandb.tags does not contain {expected_tag!r}: {tags!r}"
            )
    if problems:
        raise RuntimeError(
            "resolved TF-first cascade configuration violates its contract: "
            + "; ".join(problems)
        )


def _validate_screen_manifest(
    path: Path,
    *,
    screen_id: str,
    expected_commit: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = _read_json_file(path, "screen manifest")
    problems = []
    expected = {
        "kind": "dual_abc_tf_first_cascade_screen",
        "screen_id": screen_id,
        "git_commit": expected_commit,
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            problems.append(f"{key}: {payload.get(key)!r} != {wanted!r}")
    if payload.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        problems.append("checkpoint SHA-256 differs")
    if payload.get("config", {}).get("arms") != _arm_manifest_contract():
        problems.append("arm contract differs")
    cascade_smoke = payload.get("cascade_contract_smoke", {})
    if cascade_smoke.get("required_per_arm") is not True:
        problems.append("per-arm cascade contract smoke is not required")
    if cascade_smoke.get("git_commit") != expected_commit:
        problems.append("cascade contract smoke commit differs")
    interpretation = payload.get("causal_interpretation", {})
    if interpretation.get("oracle_sources_are_leakage") is not True:
        problems.append("oracle leakage policy is missing")
    coexistence = payload.get("active_job_coexistence", {})
    try:
        rebuilt_coexistence = _active_job_contract(
            coexistence.get("allowed_array_or_job_ids", []),
            [
                "|".join(
                    (
                        record["array_or_job_id"],
                        record["display_job_id"],
                        record["state"],
                    )
                )
                for record in coexistence.get("observed_active_jobs", [])
            ],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        problems.append(f"active-job coexistence contract is invalid: {exc}")
    else:
        if coexistence != rebuilt_coexistence:
            problems.append("active-job coexistence contract differs")
    if not _identity_is_valid(payload):
        problems.append("identity SHA-256 is invalid")
    if problems:
        raise RuntimeError("invalid screen manifest: " + "; ".join(problems))
    return payload


def command_prepare_arm(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    screen_id = pilot._validated_id(args.screen_id, "screen ID")
    wandb_run_id = pilot._validated_id(args.wandb_run_id, "W&B run ID")
    expected_commit = pilot._validated_commit(args.git_commit)
    repo_root = pilot._canonical_directory(args.repo_root, "repository root")
    project_root = pilot._canonical_directory(args.project_root, "project root")
    run_dir = pilot._canonical_directory(args.run_dir, "arm run directory")
    screen_root = pilot._canonical_directory(args.screen_root, "screen root")
    if run_dir.parent != screen_root:
        raise ValueError(f"arm run directory must be directly beneath {screen_root}")
    if run_dir.name != arm["name"]:
        raise ValueError(
            f"run directory basename {run_dir.name!r} != arm {arm['name']!r}"
        )
    if screen_root.name != screen_id:
        raise ValueError("screen root and screen ID disagree")
    pilot._assert_clean_commit(repo_root, expected_commit)

    python = pilot._python_executable(args.python)
    checkpoint = pilot._canonical_regular_file(
        args.checkpoint, "warm-start checkpoint"
    )
    checkpoint_summary = pilot._verify_checkpoint(
        checkpoint, args.checkpoint_sha256
    )
    data_root = pilot._canonical_directory(args.data_root, "data root")
    wan_dir = pilot._canonical_directory(args.wan_dir, "Wan directory")
    videox_home = pilot._canonical_directory(
        args.videox_home, "VideoX-Fun checkout"
    )
    wandb_summary = _wandb_private_project(
        args.wandb_entity, args.wandb_project
    )

    screen_manifest = pilot._canonical_regular_file(
        args.screen_manifest, "screen manifest"
    )
    screen_payload = _validate_screen_manifest(
        screen_manifest,
        screen_id=screen_id,
        expected_commit=expected_commit,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    smoke_report = pilot._canonical_regular_file(
        args.smoke_report, f"{arm['name']} gradient smoke report"
    )
    recorded_smoke = screen_payload.get("gradient_smoke_reports", {}).get(
        arm["smoke_report_key"], {}
    )
    if recorded_smoke.get("sha256") != pilot._sha256(smoke_report):
        raise RuntimeError(
            "gradient smoke report differs from the submitted screen manifest"
        )
    if recorded_smoke.get("variant") != arm["smoke_variant"]:
        raise RuntimeError("gradient smoke variant differs from the arm contract")
    cascade_contract_report = pilot._canonical_regular_file(
        args.cascade_contract_report,
        f"{arm['name']} cascade contract smoke",
    )
    cascade_contract = _validate_cascade_contract_smoke(
        cascade_contract_report,
        expected_commit=expected_commit,
    )
    recorded_contract = screen_payload.get("cascade_contract_smoke", {})
    helper_path = pilot._canonical_regular_file(
        str(repo_root / "tools" / "dual_abc_cascade_screen.py"),
        "cascade screen helper",
    )
    if recorded_contract.get("tool_sha256") != pilot._sha256(helper_path):
        raise RuntimeError(
            "cascade contract smoke tool differs from the screen manifest"
        )

    arm_config = pilot._canonical_regular_file(args.arm_config, "base arm config")
    common_config = pilot._canonical_regular_file(
        args.common_config, "common config"
    )
    recorded_configs = screen_payload.get("config", {})
    if recorded_configs.get("base_arm", {}).get("sha256") != pilot._sha256(
        arm_config
    ):
        raise RuntimeError("base arm config differs from the screen manifest")
    if recorded_configs.get("common", {}).get("sha256") != pilot._sha256(
        common_config
    ):
        raise RuntimeError("common config differs from the screen manifest")
    current_abc = pilot._abc_manifest_summary(data_root)
    if (
        screen_payload.get("data", {}).get("abc_manifest", {}).get("sha256")
        != current_abc["sha256"]
    ):
        raise RuntimeError("ABC manifest differs from the screen manifest")

    resolved_content = pilot._compose_resolved_config(
        python, project_root, args.override
    )
    _assert_resolved_contract(
        resolved_content,
        arm=arm,
        checkpoint=checkpoint,
        run_dir=run_dir,
        wandb_run_id=wandb_run_id,
    )
    resolved_output = Path(args.resolved_config_output)
    manifest_output = Path(args.manifest_output)
    if (
        resolved_output.parent.resolve(strict=True) != run_dir
        or manifest_output.parent.resolve(strict=True) != run_dir
    ):
        raise ValueError("arm provenance files must be written directly in run_dir")
    pilot._exclusive_bytes(resolved_output, resolved_content)
    resolved_sha256 = pilot._sha256(resolved_output)

    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_tf_first_cascade_screen_arm",
            "created_at_utc": pilot._now(),
            "screen_id": screen_id,
            "screen_manifest": {
                "path": str(screen_manifest),
                "sha256": pilot._sha256(screen_manifest),
                "identity_sha256": screen_payload.get("identity_sha256"),
            },
            "array_task_id": args.array_task_id,
            "arm": arm["name"],
            "representation": arm["representation"],
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
            "causal_role": arm["causal_role"],
            "git_commit": expected_commit,
            "repository_root": str(repo_root),
            "config": {
                "selector": CONFIG_SELECTOR,
                "base_arm_path": str(arm_config),
                "base_arm_sha256": pilot._sha256(arm_config),
                "common_path": str(common_config),
                "common_sha256": pilot._sha256(common_config),
                "resolved_path": str(resolved_output),
                "resolved_sha256": resolved_sha256,
                "hydra_overrides": list(args.override),
            },
            "checkpoint": checkpoint_summary,
            "gradient_smoke_report": {
                "path": str(smoke_report),
                "sha256": pilot._sha256(smoke_report),
                "variant": arm["smoke_variant"],
                "data_mode": "real",
                "configuration_scope": (
                    "ordinary raw-RFFT/tf-leads compatibility smoke; "
                    "strict Parseval cascade is contract-smoked on CPU and "
                    "validated by the actual arm run"
                ),
            },
            "cascade_contract_smoke": {
                "path": str(cascade_contract_report),
                "sha256": pilot._sha256(cascade_contract_report),
                "identity_sha256": cascade_contract.get("identity_sha256"),
                "status": cascade_contract.get("status"),
                "git_commit": cascade_contract.get("git_commit"),
            },
            "data": {
                "root": str(data_root),
                "datasets": ["ABC"],
                "abc_manifest": current_abc,
            },
            "assets": {
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
            },
            "run": {
                "run_dir": str(run_dir),
                "wandb_run_id": wandb_run_id,
                "seed": 1234,
                "optimizer_updates": OPTIMIZER_UPDATES,
                "fresh_optimizer": True,
                "world_size": 8,
                "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
                "evaluation_noise_seed": EVALUATION_NOISE_SEED,
                "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
            },
            "wandb": {
                **wandb_summary,
                "run_id": wandb_run_id,
                "group": None,
            },
            "diffusion_clock": {
                "convention": "sigma=1 noise, sigma=0 clean",
                "mode": SCHEDULE_MODE,
                "tf_loss_probability": CASCADE_TF_LOSS_PROBABILITY,
                "condition_only_video_loss_examples": (
                    CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES
                ),
                "native_video_logit_mean": CASCADE_LOGIT_MEAN,
                "native_video_logit_std": CASCADE_LOGIT_STD,
                "tf_condition_max_sigma": CASCADE_TF_CONDITION_MAX_SIGMA,
                "validation_tf_sigma": CASCADE_VALIDATION_TF_SIGMA,
                "inference_tf_fraction": CASCADE_INFERENCE_TF_FRACTION,
                "nfe_is_total_model_calls": True,
            },
            "slurm": {
                "job_id": args.slurm_job_id,
                "array_job_id": args.slurm_array_job_id,
                "array_task_id": args.array_task_id,
                "requeue": False,
            },
        }
    )
    pilot._exclusive_json(manifest_output, payload)
    print(payload["identity_sha256"])
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    screen_manifest = pilot._canonical_regular_file(
        args.screen_manifest, "screen manifest"
    )
    screen_payload = _read_json_file(screen_manifest, "screen manifest")
    if screen_payload.get("kind") != "dual_abc_tf_first_cascade_screen":
        raise RuntimeError("screen manifest kind is invalid")
    if not _identity_is_valid(screen_payload):
        raise RuntimeError("screen manifest identity SHA-256 is invalid")
    if not ARRAY_JOB_ID_RE.fullmatch(args.job_id):
        raise ValueError("Slurm returned an invalid array job ID")
    max_concurrent_arms = int(args.max_concurrent_arms)
    if max_concurrent_arms < 1 or max_concurrent_arms > len(ARMS):
        raise ValueError(
            f"max concurrent arms must be between 1 and {len(ARMS)}"
        )
    manifest_inventory = screen_payload.get("active_job_coexistence")
    if not isinstance(manifest_inventory, dict):
        raise RuntimeError("screen manifest active-job inventory is missing")
    allowed_job_ids = manifest_inventory.get("allowed_array_or_job_ids")
    if not isinstance(allowed_job_ids, list):
        raise RuntimeError("screen manifest allowed active-job IDs are invalid")
    immediate_pre_sbatch_inventory = _active_job_contract(
        allowed_job_ids,
        args.observed_active_job,
    )
    payload = pilot._identity_payload({
        "schema_version": 2,
        "kind": "dual_abc_tf_first_cascade_screen_slurm_submission",
        "created_at_utc": pilot._now(),
        "screen_manifest": {
            "path": str(screen_manifest),
            "sha256": pilot._sha256(screen_manifest),
            "identity_sha256": screen_payload.get("identity_sha256"),
        },
        "slurm_array_job_id": args.job_id,
        "array": f"0-{len(ARMS) - 1}%{max_concurrent_arms}",
        "requeue": False,
        "active_job_coexistence": manifest_inventory,
        "immediate_pre_sbatch_active_job_coexistence": (
            immediate_pre_sbatch_inventory
        ),
    })
    output = Path(args.output)
    if output.parent.resolve(strict=True) != screen_manifest.parent:
        raise ValueError("submission record must be directly under screen root")
    pilot._exclusive_json(output, payload)
    return 0


def _required_trajectory_tensor_names() -> set[str]:
    required_tensor_names = {
        "video_clean",
        "tf_clean",
        "video_initial_state",
        "tf_initial_state",
        "tf_initial_noise",
        "ground_truth_future_uint8",
        "history_latent_frames",
        "condition_on_tf",
        "condition_only_video_loss_examples",
        "condition_mode_code",
        "evaluation_noise_seed",
        "evaluation_nfe_steps",
        "evaluation_condition_source_codes",
        "oracle_sources_are_leakage",
        "video_trajectory",
        "tf_trajectory",
        "video_x0_trajectory",
        "tf_x0_trajectory",
        "video_sigmas",
        "tf_sigmas",
    }
    for source in EVALUATION_CONDITION_SOURCES:
        source_infix = "" if source == "autonomous" else f"_{source}"
        for nfe in EVALUATION_NFE_STEPS:
            required_tensor_names.update(
                {
                    f"video_final{source_infix}_nfe_{nfe}",
                    f"tf_final{source_infix}_nfe_{nfe}",
                    f"decoded_future{source_infix}_nfe_{nfe}",
                }
            )
    return required_tensor_names


def _trajectory_summary(
    run_dir: Path,
    *,
    arm_payload: dict[str, Any],
) -> dict[str, Any]:
    from safetensors import safe_open

    required_tensor_names = _required_trajectory_tensor_names()

    summary: dict[str, Any] = {}
    for iteration in VISUALIZATION_UPDATES:
        iteration_dir = pilot._canonical_directory(
            str(run_dir / "visualization" / f"iter_{iteration}"),
            f"iteration {iteration} visualization directory",
        )
        trajectories = sorted(
            iteration_dir.glob("*/latent_trajectory_rank_*.safetensors")
        )
        sidecars = sorted(iteration_dir.glob("*/latent_trajectory_rank_*.json"))
        decoded_videos = sorted(iteration_dir.glob("*/*.mp4"))
        if (
            len(trajectories) != 8
            or len(sidecars) != 8
            or len(decoded_videos) != 8
        ):
            raise RuntimeError(
                f"iteration {iteration} artifact count mismatch: "
                f"trajectories={len(trajectories)}, sidecars={len(sidecars)}, "
                f"decoded_videos={len(decoded_videos)}; expected 8 each"
            )

        records = []
        ranks: set[int] = set()
        for trajectory in trajectories:
            if trajectory.is_symlink() or trajectory.stat().st_size <= 0:
                raise RuntimeError(f"invalid trajectory file: {trajectory}")
            match = re.search(
                r"latent_trajectory_rank_([0-9]+)[.]safetensors$",
                trajectory.name,
            )
            if match is None:
                raise RuntimeError(f"unexpected trajectory filename: {trajectory}")
            rank = int(match.group(1))
            ranks.add(rank)
            sidecar_path = trajectory.with_suffix(".json")
            sidecar = _read_json_file(sidecar_path, "trajectory sidecar")
            sidecar_expected = {
                "iteration": iteration,
                "global_rank": rank,
                "sigma_convention": "1=noise,0=clean",
            }
            problems = [
                f"{key}: {sidecar.get(key)!r} != {wanted!r}"
                for key, wanted in sidecar_expected.items()
                if sidecar.get(key) != wanted
            ]
            actual_sha256 = pilot._sha256(trajectory)
            if sidecar.get("safetensors_sha256") != actual_sha256:
                problems.append("safetensors_sha256 differs")
            with safe_open(trajectory, framework="pt", device="cpu") as handle:
                tensor_names = set(handle.keys())
                tensor_shapes = {
                    name: tuple(handle.get_slice(name).get_shape())
                    for name in tensor_names
                }
                tensor_dtypes = {
                    name: str(handle.get_slice(name).get_dtype())
                    for name in tensor_names
                }
                small_tensors = {
                    name: handle.get_tensor(name)
                    for name in (
                        "history_latent_frames",
                        "condition_on_tf",
                        "condition_only_video_loss_examples",
                        "condition_mode_code",
                        "evaluation_noise_seed",
                        "evaluation_nfe_steps",
                        "evaluation_condition_source_codes",
                        "oracle_sources_are_leakage",
                        "video_sigmas",
                        "tf_sigmas",
                    )
                    if name in tensor_names
                }
            missing = sorted(required_tensor_names - tensor_names)
            if missing:
                problems.append(f"missing tensors: {missing}")
            expected_small_values = {
                "condition_on_tf": [int(arm_payload["condition_on_tf"])],
                "condition_only_video_loss_examples": [
                    int(CASCADE_CONDITION_ONLY_VIDEO_LOSS_EXAMPLES)
                ],
                "condition_mode_code": [
                    CONDITION_MODE_CODES[str(arm_payload["condition_mode"])]
                ],
                "evaluation_noise_seed": [EVALUATION_NOISE_SEED],
                "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
                "evaluation_condition_source_codes": [
                    EVALUATION_CONDITION_SOURCE_CODES[source]
                    for source in EVALUATION_CONDITION_SOURCES
                ],
                "oracle_sources_are_leakage": [1],
            }
            for name, wanted in expected_small_values.items():
                tensor = small_tensors.get(name)
                if tensor is not None and tensor.reshape(-1).tolist() != wanted:
                    problems.append(
                        f"{name}: {tensor.reshape(-1).tolist()!r} != {wanted!r}"
                    )

            expected_metadata_shapes = {
                "history_latent_frames": (1,),
                "condition_on_tf": (1,),
                "condition_only_video_loss_examples": (1,),
                "condition_mode_code": (1,),
                "evaluation_noise_seed": (1,),
                "evaluation_nfe_steps": (len(EVALUATION_NFE_STEPS),),
                "evaluation_condition_source_codes": (
                    len(EVALUATION_CONDITION_SOURCES),
                ),
                "oracle_sources_are_leakage": (1,),
                "video_sigmas": (9,),
                "tf_sigmas": (9,),
            }
            for name, wanted_shape in expected_metadata_shapes.items():
                if (
                    name in tensor_shapes
                    and tensor_shapes[name] != wanted_shape
                ):
                    problems.append(
                        f"{name} shape differs: "
                        f"{tensor_shapes[name]} != {wanted_shape}"
                    )
            for name in (
                "history_latent_frames",
                "condition_on_tf",
                "condition_only_video_loss_examples",
                "condition_mode_code",
                "evaluation_noise_seed",
                "evaluation_nfe_steps",
                "evaluation_condition_source_codes",
                "oracle_sources_are_leakage",
            ):
                if name in tensor_dtypes and tensor_dtypes[name] != "I64":
                    problems.append(
                        f"{name} dtype differs: {tensor_dtypes[name]} != I64"
                    )
            for name in ("video_sigmas", "tf_sigmas"):
                if name in tensor_dtypes and tensor_dtypes[name] != "F32":
                    problems.append(
                        f"{name} dtype differs: {tensor_dtypes[name]} != F32"
                    )
            fp16_names = {
                "video_clean",
                "tf_clean",
                "video_initial_state",
                "tf_initial_state",
                "tf_initial_noise",
                "video_trajectory",
                "tf_trajectory",
                "video_x0_trajectory",
                "tf_x0_trajectory",
            }
            for source in EVALUATION_CONDITION_SOURCES:
                source_infix = "" if source == "autonomous" else f"_{source}"
                for nfe in EVALUATION_NFE_STEPS:
                    fp16_names.update(
                        {
                            f"video_final{source_infix}_nfe_{nfe}",
                            f"tf_final{source_infix}_nfe_{nfe}",
                        }
                    )
            for name in fp16_names:
                if name in tensor_dtypes and tensor_dtypes[name] != "F16":
                    problems.append(
                        f"{name} dtype differs: {tensor_dtypes[name]} != F16"
                    )
            if (
                "ground_truth_future_uint8" in tensor_dtypes
                and tensor_dtypes["ground_truth_future_uint8"] != "U8"
            ):
                problems.append(
                    "ground_truth_future_uint8 dtype differs: "
                    f"{tensor_dtypes['ground_truth_future_uint8']} != U8"
                )

            video_shape = tensor_shapes.get("video_clean")
            tf_shape = tensor_shapes.get("tf_clean")
            if video_shape is not None and (
                len(video_shape) != 5
                or video_shape[0] != 1
                or video_shape[1] != 16
            ):
                problems.append(f"video_clean shape is invalid: {video_shape}")
            if tf_shape is not None and (
                len(tf_shape) != 5
                or tf_shape[0] != 1
                or tf_shape[1] != 12
                or (
                    video_shape is not None
                    and tf_shape[2:] != video_shape[2:]
                )
            ):
                problems.append(f"tf_clean shape is invalid: {tf_shape}")
            history_tensor = small_tensors.get("history_latent_frames")
            if history_tensor is not None and video_shape is not None:
                history_frames = int(history_tensor.reshape(-1)[0])
                if not 0 < history_frames < video_shape[2]:
                    problems.append(
                        "history_latent_frames must select a nonempty proper "
                        f"prefix: {history_frames} for latent frames {video_shape[2]}"
                    )
            for name in ("video_initial_state",):
                if (
                    video_shape is not None
                    and name in tensor_shapes
                    and tensor_shapes[name] != video_shape
                ):
                    problems.append(
                        f"{name} shape differs from video_clean: "
                        f"{tensor_shapes[name]} != {video_shape}"
                    )
            for name in ("tf_initial_state", "tf_initial_noise"):
                if (
                    tf_shape is not None
                    and name in tensor_shapes
                    and tensor_shapes[name] != tf_shape
                ):
                    problems.append(
                        f"{name} shape differs from tf_clean: "
                        f"{tensor_shapes[name]} != {tf_shape}"
                    )
            trajectory_shape_contracts = {
                "video_trajectory": (
                    None if video_shape is None else (9, *video_shape)
                ),
                "tf_trajectory": (
                    None if tf_shape is None else (9, *tf_shape)
                ),
                "video_x0_trajectory": (
                    None if video_shape is None else (8, *video_shape)
                ),
                "tf_x0_trajectory": (
                    None if tf_shape is None else (8, *tf_shape)
                ),
            }
            for name, wanted_shape in trajectory_shape_contracts.items():
                if (
                    wanted_shape is not None
                    and name in tensor_shapes
                    and tensor_shapes[name] != wanted_shape
                ):
                    problems.append(
                        f"{name} shape differs: "
                        f"{tensor_shapes[name]} != {wanted_shape}"
                    )
            for source in EVALUATION_CONDITION_SOURCES:
                source_infix = "" if source == "autonomous" else f"_{source}"
                for nfe in EVALUATION_NFE_STEPS:
                    shape_contracts = {
                        f"video_final{source_infix}_nfe_{nfe}": video_shape,
                        f"tf_final{source_infix}_nfe_{nfe}": tf_shape,
                        f"decoded_future{source_infix}_nfe_{nfe}": (
                            tensor_shapes.get("ground_truth_future_uint8")
                        ),
                    }
                    for name, wanted_shape in shape_contracts.items():
                        if (
                            wanted_shape is not None
                            and name in tensor_shapes
                            and tensor_shapes[name] != wanted_shape
                        ):
                            problems.append(
                                f"{name} shape differs: "
                                f"{tensor_shapes[name]} != {wanted_shape}"
                            )
                    decoded_name = (
                        f"decoded_future{source_infix}_nfe_{nfe}"
                    )
                    if (
                        decoded_name in tensor_dtypes
                        and tensor_dtypes[decoded_name] != "U8"
                    ):
                        problems.append(
                            f"{decoded_name} dtype differs: "
                            f"{tensor_dtypes[decoded_name]} != U8"
                        )

            video_sigmas = small_tensors.get("video_sigmas")
            tf_sigmas = small_tensors.get("tf_sigmas")
            expected_tf_sigmas = [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
            if video_sigmas is not None:
                values = [float(value) for value in video_sigmas.reshape(-1)]
                if (
                    len(values) != 9
                    or any(abs(value - 1.0) > 1e-6 for value in values[:5])
                    or values[5] >= 1.0 - 1e-6
                    or values[-2] <= 0.0
                    or abs(values[-1]) > 1e-6
                    or any(
                        right - left > 1e-6
                        for left, right in zip(values, values[1:])
                    )
                ):
                    problems.append(
                        f"video_sigmas do not encode a 4+4 strict cascade: {values}"
                    )
            if tf_sigmas is not None:
                values = [float(value) for value in tf_sigmas.reshape(-1)]
                if len(values) != len(expected_tf_sigmas) or any(
                    abs(actual - wanted) > 1e-6
                    for actual, wanted in zip(values, expected_tf_sigmas)
                ):
                    problems.append(
                        f"tf_sigmas do not encode a 4+4 strict cascade: {values}"
                    )
            if problems:
                raise RuntimeError(
                    f"invalid trajectory artifact {trajectory}: "
                    + "; ".join(problems)
                )
            records.append(
                {
                    "path": str(trajectory),
                    "sha256": actual_sha256,
                    "bytes": trajectory.stat().st_size,
                    "rank": rank,
                    "tensor_names": sorted(tensor_names),
                    "tensor_shapes": {
                        name: list(shape)
                        for name, shape in sorted(tensor_shapes.items())
                    },
                    "tensor_dtypes": dict(sorted(tensor_dtypes.items())),
                }
            )
        if ranks != set(range(8)):
            raise RuntimeError(
                f"iteration {iteration} trajectory ranks differ: {sorted(ranks)}"
            )
        summary[str(iteration)] = {
            "trajectory_count": 8,
            "sidecar_count": 8,
            "decoded_video_count": 8,
            "ranks": list(range(8)),
            "trajectories": records,
        }
    return summary


def command_record_outcome(args: argparse.Namespace) -> int:
    manifest = pilot._canonical_regular_file(args.manifest, "arm manifest")
    arm_payload = _read_json_file(manifest, "arm manifest")
    if arm_payload.get("kind") != "dual_abc_tf_first_cascade_screen_arm":
        raise RuntimeError("arm manifest kind is invalid")
    if not _identity_is_valid(arm_payload):
        raise RuntimeError("arm manifest identity SHA-256 is invalid")

    snapshot = Path(args.snapshot)
    status = int(args.exit_status)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dual_abc_tf_first_cascade_screen_arm_outcome",
        "created_at_utc": pilot._now(),
        "manifest": {
            "path": str(manifest),
            "sha256": pilot._sha256(manifest),
            "identity_sha256": arm_payload.get("identity_sha256"),
        },
        "exit_status": status,
        "completed": status == 0,
    }
    if status == 0:
        run_dir = snapshot.parent.resolve(strict=True)
        completion_path = pilot._canonical_regular_file(
            str(run_dir / "training_complete.json"), "training completion marker"
        )
        completion = _read_json_file(completion_path, "training completion marker")
        expected_completion = {
            "schema_version": 1,
            "status": "completed",
            "completed_updates": OPTIMIZER_UPDATES,
            "max_iter": OPTIMIZER_UPDATES,
            "run_identity_sha256": arm_payload.get("identity_sha256"),
            "snapshot": str(snapshot.resolve(strict=False)),
        }
        problems = [
            f"{key}: {completion.get(key)!r} != {wanted!r}"
            for key, wanted in expected_completion.items()
            if completion.get(key) != wanted
        ]
        if problems:
            raise RuntimeError(
                "training completion marker is invalid: " + "; ".join(problems)
            )
        payload["training_completion"] = {
            "path": str(completion_path),
            "sha256": pilot._sha256(completion_path),
            "completed_updates": OPTIMIZER_UPDATES,
            "max_iter": OPTIMIZER_UPDATES,
            "run_identity_sha256": arm_payload.get("identity_sha256"),
        }
        payload["visualization_artifacts"] = _trajectory_summary(
            run_dir,
            arm_payload=arm_payload,
        )

    if snapshot.is_file() and not snapshot.is_symlink():
        payload["snapshot"] = {
            "path": str(snapshot.resolve(strict=True)),
            "bytes": snapshot.stat().st_size,
            "sha256": pilot._sha256(snapshot),
        }
    elif status == 0:
        raise RuntimeError(
            f"training exited successfully without a snapshot: {snapshot}"
        )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != manifest.parent:
        raise ValueError(
            "outcome must be written directly under the arm run directory"
        )
    pilot._exclusive_json(output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    arm_contract = subparsers.add_parser(
        "arm-contract", help="emit the immutable mapping for one array task"
    )
    arm_contract.add_argument("--array-task-id", required=True, type=int)
    arm_contract.add_argument("--format", choices=("json", "tsv"), default="json")
    arm_contract.set_defaults(func=command_arm_contract)

    wandb_parser = subparsers.add_parser(
        "wandb-private", help="verify the authenticated personal project is private"
    )
    wandb_parser.add_argument("--entity", default=EXPECTED_ENTITY)
    wandb_parser.add_argument("--project", default=EXPECTED_PROJECT)
    wandb_parser.set_defaults(func=command_wandb_private)

    contract_parser = subparsers.add_parser(
        "cascade-contract-smoke",
        help="exercise the checked-out strict cascade and Parseval contracts",
    )
    contract_parser.add_argument("--repo-root", required=True)
    contract_parser.add_argument("--git-commit", required=True)
    contract_parser.add_argument("--output", required=True)
    contract_parser.set_defaults(func=command_cascade_contract_smoke)

    screen_parser = subparsers.add_parser(
        "create-screen", help="write the immutable TF-first cascade manifest"
    )
    for name in (
        "screen-id",
        "git-commit",
        "repo-root",
        "run-root",
        "screen-root",
        "python",
        "wan-dir",
        "videox-home",
        "data-root",
        "checkpoint",
        "checkpoint-sha256",
        "no-ztf-smoke-report",
        "with-ztf-smoke-report",
        "common-config",
        "arm-config",
        "wandb-entity",
        "wandb-project",
        "max-concurrent-arms",
        "output",
    ):
        screen_parser.add_argument(f"--{name}", required=True)
    screen_parser.add_argument(
        "--allow-active-job-id", action="append", default=[]
    )
    screen_parser.add_argument(
        "--observed-active-job", action="append", default=[]
    )
    screen_parser.set_defaults(func=command_create_screen)

    prepare_parser = subparsers.add_parser(
        "prepare-arm", help="resolve and validate one arm before torchrun"
    )
    prepare_parser.add_argument("--array-task-id", required=True, type=int)
    for name in (
        "screen-id",
        "git-commit",
        "repo-root",
        "project-root",
        "screen-root",
        "run-dir",
        "python",
        "wan-dir",
        "videox-home",
        "data-root",
        "checkpoint",
        "checkpoint-sha256",
        "arm-config",
        "common-config",
        "wandb-entity",
        "wandb-project",
        "wandb-run-id",
        "screen-manifest",
        "smoke-report",
        "cascade-contract-report",
        "slurm-job-id",
        "slurm-array-job-id",
        "resolved-config-output",
        "manifest-output",
    ):
        prepare_parser.add_argument(f"--{name}", required=True)
    prepare_parser.add_argument(
        "--override", action="append", default=[], required=True
    )
    prepare_parser.set_defaults(func=command_prepare_arm)

    submission_parser = subparsers.add_parser(
        "record-submission", help="record the accepted Slurm array job ID"
    )
    submission_parser.add_argument("--screen-manifest", required=True)
    submission_parser.add_argument("--job-id", required=True)
    submission_parser.add_argument("--max-concurrent-arms", required=True, type=int)
    submission_parser.add_argument(
        "--observed-active-job", action="append", default=[]
    )
    submission_parser.add_argument("--output", required=True)
    submission_parser.set_defaults(func=command_record_submission)

    outcome_parser = subparsers.add_parser(
        "record-outcome", help="write a terminal arm outcome"
    )
    outcome_parser.add_argument("--manifest", required=True)
    outcome_parser.add_argument("--snapshot", required=True)
    outcome_parser.add_argument("--exit-status", required=True, type=int)
    outcome_parser.add_argument("--output", required=True)
    outcome_parser.set_defaults(func=command_record_outcome)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
