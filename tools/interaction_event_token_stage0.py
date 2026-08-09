#!/usr/bin/env python3
"""Sequential explicit-token fallback for masked interaction events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import interaction_event_bottleneck_stage0 as base
from tools.abc_d405_nominal_geometry_probe import (
    EXPECTED_ABC_COMMIT,
    OFFICIAL_ASSET_RELATIVE,
    OFFICIAL_SCENE_RELATIVE,
    _joint_qpos_addresses,
    build_robot_only_xml,
)
from tools.corrected_renderer_attribution import moving_geom_ids


SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_SAMPLES = 10_000
FAMILY_SIZE = 3
MIN_IMPROVEMENT_PERCENT = 5.0
MIN_FAVORABLE_FRACTION = 0.60
MIN_ACTIVE_TOKEN_DIMS = 24
ACTIVE_TOKEN_STD_THRESHOLD = 1e-6
MIN_ACTIVE_SCORE_FRACTION = 0.90
MIN_SCORE_EVENT_MASS = 1e-5
ARMS = base.ARMS


class TokenError(RuntimeError):
    """Raised when the sequential fallback contract is violated."""


def token_state(field: np.ndarray) -> np.ndarray:
    """Return exact per-transition/channel spatial integrals (means)."""
    array = np.asarray(field, dtype=np.float32)
    if array.ndim != 4 or array.shape[1:] != (4, *base.WORK_HW):
        raise ValueError(f"event-field geometry differs: {array.shape}")
    result = array.mean(axis=(-2, -1), dtype=np.float64).astype(np.float32)
    if not np.isfinite(result).all():
        raise TokenError("explicit event tokens are non-finite")
    return result


def register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = args.parent_artifact.resolve()
    parent_documents = {
        name: base.read_json(parent / name)
        for name in ("registration.json", "analysis.json", "run_complete.json", "audit.json")
    }
    for name, document in parent_documents.items():
        base.validate_identity(document, f"parent/{name}")
    parent_registration = parent_documents["registration.json"]
    parent_analysis = parent_documents["analysis.json"]
    parent_complete = parent_documents["run_complete.json"]
    parent_audit = parent_documents["audit.json"]
    if parent_registration["identity_sha256"] != (
        "f985c94c708950a6e90530044a9b91f58e539a50a3f7ac2f91b89b1dc945f393"
    ):
        raise TokenError("parent registration is not the frozen fit256/score64 split")
    if parent_analysis["decision"] != "STOP_INTERACTION_EVENT_BOTTLENECK":
        raise TokenError("fallback requires the frozen PCA32 stop decision")
    if parent_complete["analysis_identity_sha256"] != parent_analysis["identity_sha256"]:
        raise TokenError("parent completion binding differs")
    if parent_audit.get("status") != "audit_passed":
        raise TokenError("parent read-only audit did not pass")

    source = Path(__file__).resolve()
    protocol = args.protocol.resolve()
    frozen_source = output / "frozen_interaction_event_token_stage0.py"
    shutil.copy2(source, frozen_source)
    payload = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_token_stage0_registration",
            "status": "sequential_fallback_registered_before_token_scoring",
            "created_at_utc": base.now(),
            "source": {
                **base.file_record(source),
                "git_commit": args.expected_commit or base.git_commit(source.parents[1]),
                "frozen_copy": base.file_record(frozen_source),
            },
            "protocol": base.file_record(protocol),
            "parent": {
                "artifact_root": str(parent),
                "registration": {
                    **base.file_record(parent / "registration.json"),
                    "identity_sha256": parent_registration["identity_sha256"],
                },
                "analysis": {
                    **base.file_record(parent / "analysis.json"),
                    "identity_sha256": parent_analysis["identity_sha256"],
                    "decision": parent_analysis["decision"],
                },
                "completion": {
                    **base.file_record(parent / "run_complete.json"),
                    "identity_sha256": parent_complete["identity_sha256"],
                },
                "audit": {
                    **base.file_record(parent / "audit.json"),
                    "identity_sha256": parent_audit["identity_sha256"],
                },
            },
            "inputs": parent_registration["inputs"],
            "split": parent_registration["split"],
            "representation": {
                "raw_event_field_contract": parent_registration["representation"],
                "future_shape": [8, 4],
                "flattened_future_dimensions": 32,
                "history_shape": [4, 4],
                "flattened_history_dimensions": 16,
                "tokens": [
                    "positive_change_spatial_mean",
                    "negative_change_spatial_mean",
                    "signed_horizontal_transport_spatial_mean",
                    "signed_vertical_transport_spatial_mean",
                ],
                "learned_target_encoder": False,
                "spatial_localization_retained": False,
            },
            "models": {
                "history_only_equal_width": "history16 concatenated with zeros32",
                "history_plus_action": "history16 concatenated with fit-only action PCA32",
                "ridge_alpha_grid": list(base.ALPHAS),
                "model_seed": base.MODEL_SEED,
                "arms": list(ARMS),
            },
            "statistics": {
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "family_size": FAMILY_SIZE,
                "one_sided_bonferroni_confidence": 1.0 - 0.05 / FAMILY_SIZE,
                "minimum_relative_improvement_percent": MIN_IMPROVEMENT_PERCENT,
                "minimum_favorable_fraction": MIN_FAVORABLE_FRACTION,
                "minimum_active_token_dimensions": MIN_ACTIVE_TOKEN_DIMS,
                "active_token_std_threshold": ACTIVE_TOKEN_STD_THRESHOLD,
                "minimum_active_score_fraction": MIN_ACTIVE_SCORE_FRACTION,
                "minimum_score_event_mass": MIN_SCORE_EVENT_MASS,
            },
            "claim_boundary": (
                "Sequential train-only explicit-token diagnostic. A pass can authorize only a spatial-localization screen, never direct generator integration."
            ),
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "registration.json", payload)
    return payload


def common_bootstrap_indices() -> np.ndarray:
    return base.common_bootstrap_indices(
        base.SCORE_CLIPS, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )


def analyze_errors(
    errors: Mapping[str, np.ndarray],
    *,
    active_token_dims: int,
    active_score_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = common_bootstrap_indices()
    mandatory: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for reference in ("history_only", "episode_shuffled", "train_mean"):
        label = f"all_tokens:aligned_vs_{reference}"
        effect = base.paired_relative_effect(
            np.asarray(errors[f"all_token_error_{reference}"]),
            np.asarray(errors["all_token_error_aligned"]),
            bootstrap,
            family_size=FAMILY_SIZE,
        )
        mandatory[label] = effect
        checks = {
            "point_at_least_5_percent": effect["relative_improvement_percent"]
            >= MIN_IMPROVEMENT_PERCENT,
            "simultaneous_lower_bound_strictly_positive": effect[
                "simultaneous_lower_bound_percent"
            ]
            > 0.0,
            "favorable_fraction_at_least_60_percent": effect["favorable_fraction"]
            >= MIN_FAVORABLE_FRACTION,
        }
        gates[label] = {**checks, "passed": all(checks.values())}

    diagnostics: dict[str, Any] = {}
    for subset in ("change", "transport"):
        for reference in (
            "history_only",
            "episode_shuffled",
            "train_mean",
            "raw_zero",
            "shift_minus1",
            "shift_plus1",
        ):
            label = f"{subset}:aligned_vs_{reference}"
            diagnostics[label] = base.paired_relative_effect(
                np.asarray(errors[f"{subset}_error_{reference}"]),
                np.asarray(errors[f"{subset}_error_aligned"]),
                bootstrap,
                family_size=1,
            )
    for reference in ("raw_zero", "shift_minus1", "shift_plus1"):
        label = f"all_tokens:aligned_vs_{reference}"
        diagnostics[label] = base.paired_relative_effect(
            np.asarray(errors[f"all_token_error_{reference}"]),
            np.asarray(errors["all_token_error_aligned"]),
            bootstrap,
            family_size=1,
        )

    salience_checks = {
        "at_least_24_active_fit_token_dimensions": active_token_dims
        >= MIN_ACTIVE_TOKEN_DIMS,
        "at_least_90_percent_score_clips_have_event_mass": active_score_fraction
        >= MIN_ACTIVE_SCORE_FRACTION,
    }
    gates["target_salience"] = {
        **salience_checks,
        "passed": all(salience_checks.values()),
    }
    gates["all_passed"] = all(item["passed"] for item in gates.values())
    return {"mandatory": mandatory, "diagnostic": diagnostics}, gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import mujoco
        from sklearn.linear_model import Ridge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("run requires mujoco and scikit-learn") from exc

    output = args.output.resolve()
    if (output / "analysis.json").exists():
        raise FileExistsError("completed token output is immutable")
    registration = base.read_json(output / "registration.json")
    base.validate_identity(registration, "registration")
    if base.sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise TokenError("execution source differs from registration")
    frozen_source = output / "frozen_interaction_event_token_stage0.py"
    if base.sha256_file(frozen_source) != registration["source"]["frozen_copy"]["sha256"]:
        raise TokenError("frozen source differs")

    manifest_path = Path(registration["inputs"]["train_manifest"]["path"])
    cache_dir = Path(registration["inputs"]["cache_dir"])
    rows = base.read_jsonl(manifest_path)
    base.validate_train_manifest(rows)
    _, rgb_cache, action_cache, _, _ = base.load_cache(cache_dir)
    fit_items = registration["split"]["fit"]
    score_items = registration["split"]["score"]
    selected = [*fit_items, *score_items]

    official_root = args.official_abc_root.resolve()
    official_commit = base.git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT:
        raise TokenError(f"official ABC commit differs: {official_commit}")
    scene_path = official_root / OFFICIAL_SCENE_RELATIVE
    asset_root = official_root / OFFICIAL_ASSET_RELATIVE
    robot_xml = build_robot_only_xml(scene_path, asset_root)
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    if camera_id < 0:
        raise TokenError("official model lacks top camera")
    moving_geoms = moving_geom_ids(model, mujoco)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    renderer = mujoco.Renderer(
        model, height=base.RENDER_HW[0], width=base.RENDER_HW[1]
    )
    context = {
        "renderer": renderer,
        "model": model,
        "data": data,
        "mujoco": mujoco,
        "option": option,
        "addresses": addresses,
        "moving_geoms": moving_geoms,
        "camera_id": camera_id,
    }

    fit_history: list[np.ndarray] = []
    fit_target: list[np.ndarray] = []
    fit_actions: list[np.ndarray] = []
    score_history: list[np.ndarray] = []
    score_target: list[np.ndarray] = []
    score_actions: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    render_latencies: list[float] = []
    started = time.time()
    try:
        for position, item in enumerate(selected):
            history_field, future_field, actions, record, latencies = (
                base._selected_clip_arrays(
                    item=item,
                    row=rows[int(item["manifest_index"])],
                    rgb_cache=rgb_cache,
                    action_cache=action_cache,
                    renderer_context=context,
                )
            )
            history_token = token_state(history_field)
            target_token = token_state(future_field)
            record = dict(record)
            record["token_contract"] = {
                "history_shape": list(history_token.shape),
                "future_shape": list(target_token.shape),
                "learned_target_encoder": False,
                "future_tokens_are_spatial_means_of_raw_event_channels": True,
            }
            provenance.append(record)
            render_latencies.extend(latencies[2:] if len(latencies) > 2 else latencies)
            if item["role"] == "predictor_fit":
                fit_history.append(history_token)
                fit_target.append(target_token)
                fit_actions.append(actions)
            else:
                score_history.append(history_token)
                score_target.append(target_token)
                score_actions.append(actions)
            if position == 0 or (position + 1) % 32 == 0:
                print(
                    json.dumps(
                        {
                            "event": "explicit_token_extraction_progress",
                            "clips": position + 1,
                            "total": len(selected),
                            "elapsed_sec": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        renderer.close()

    fit_h = np.stack(fit_history).reshape(base.FIT_CLIPS, -1)
    fit_y = np.stack(fit_target).reshape(base.FIT_CLIPS, -1)
    fit_a_full = np.stack(fit_actions)
    score_h = np.stack(score_history).reshape(base.SCORE_CLIPS, -1)
    score_y = np.stack(score_target).reshape(base.SCORE_CLIPS, -1)
    score_a_full = np.stack(score_actions)
    if fit_h.shape != (base.FIT_CLIPS, 16) or fit_y.shape != (base.FIT_CLIPS, 32):
        raise TokenError(f"fit token geometry differs: {fit_h.shape}/{fit_y.shape}")
    if score_h.shape != (base.SCORE_CLIPS, 16) or score_y.shape != (
        base.SCORE_CLIPS,
        32,
    ):
        raise TokenError(f"score token geometry differs: {score_h.shape}/{score_y.shape}")

    fit_action_window = fit_a_full[
        :, base.ACTION_WINDOW[0] : base.ACTION_WINDOW[1], :, : base.ACTION_DIM
    ]
    score_action_window = score_a_full[
        :, base.ACTION_WINDOW[0] : base.ACTION_WINDOW[1], :, : base.ACTION_DIM
    ]
    action_pca, fit_action_pc, action_mean, action_std, action_active = (
        base._fit_action_pca(fit_action_window)
    )
    score_action_pc = base._transform_action_pca(
        action_pca, score_action_window, action_mean, action_std, action_active
    )

    target_raw_std = fit_y.std(axis=0, dtype=np.float64)
    target_mean, target_std = base._standardizer(fit_y)
    fit_target_z = (fit_y - target_mean) / target_std
    history_design = np.concatenate((fit_h, np.zeros_like(fit_action_pc)), axis=1)
    action_design = np.concatenate((fit_h, fit_action_pc), axis=1)
    history_input_mean, history_input_std = base._standardizer(history_design)
    action_input_mean, action_input_std = base._standardizer(action_design)
    history_x = (history_design - history_input_mean) / history_input_std
    action_x = (action_design - action_input_mean) / action_input_std
    history_alpha, history_cv = base._choose_alpha(history_x, fit_target_z)
    action_alpha, action_cv = base._choose_alpha(action_x, fit_target_z)
    history_model = Ridge(
        alpha=history_alpha, fit_intercept=True, solver="cholesky"
    ).fit(history_x, fit_target_z)
    action_model = Ridge(
        alpha=action_alpha, fit_intercept=True, solver="cholesky"
    ).fit(action_x, fit_target_z)

    history_prediction = base._model_prediction(
        model=history_model,
        history_pc=score_h,
        action_pc=np.zeros_like(score_action_pc),
        input_mean=history_input_mean,
        input_std=history_input_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    position_by_index = {
        int(item["manifest_index"]): position for position, item in enumerate(score_items)
    }
    donor_positions = np.asarray(
        [position_by_index[int(item["donor_manifest_index"])] for item in score_items],
        dtype=np.int64,
    )
    fit_action_mean = fit_action_window.reshape(base.FIT_CLIPS, -1).mean(axis=0).reshape(
        1, 8, base.CHUNK_SIZE, base.ACTION_DIM
    )
    control_action_pc: dict[str, np.ndarray] = {
        "aligned": score_action_pc,
        "episode_shuffled": score_action_pc[donor_positions],
        "train_mean": base._transform_action_pca(
            action_pca,
            np.repeat(fit_action_mean, base.SCORE_CLIPS, axis=0),
            action_mean,
            action_std,
            action_active,
        ),
        "raw_zero": base._transform_action_pca(
            action_pca,
            np.zeros_like(score_action_window),
            action_mean,
            action_std,
            action_active,
        ),
    }
    for label, (start_index, stop_index) in base.SHIFT_WINDOWS.items():
        control_action_pc[label] = base._transform_action_pca(
            action_pca,
            score_a_full[:, start_index:stop_index, :, : base.ACTION_DIM],
            action_mean,
            action_std,
            action_active,
        )
    predictions: dict[str, np.ndarray] = {"history_only": history_prediction}
    for label, action_pc in control_action_pc.items():
        predictions[label] = base._model_prediction(
            model=action_model,
            history_pc=score_h,
            action_pc=action_pc,
            input_mean=action_input_mean,
            input_std=action_input_std,
            target_mean=target_mean,
            target_std=target_std,
        )

    change_indexes = np.asarray(
        [horizon * 4 + channel for horizon in range(8) for channel in (0, 1)],
        dtype=np.int64,
    )
    transport_indexes = np.asarray(
        [horizon * 4 + channel for horizon in range(8) for channel in (2, 3)],
        dtype=np.int64,
    )
    errors: dict[str, np.ndarray] = {}
    for label, prediction in predictions.items():
        standardized = (prediction - score_y) / target_std
        errors[f"all_token_error_{label}"] = np.mean(standardized**2, axis=1).astype(
            np.float64
        )
        errors[f"change_error_{label}"] = np.mean(
            standardized[:, change_indexes] ** 2, axis=1
        ).astype(np.float64)
        errors[f"transport_error_{label}"] = np.mean(
            standardized[:, transport_indexes] ** 2, axis=1
        ).astype(np.float64)

    score_tokens = score_y.reshape(base.SCORE_CLIPS, 8, 4)
    score_event_mass = np.sum(score_tokens[:, :, :2], axis=(1, 2))
    active_token_dims = int(np.sum(target_raw_std > ACTIVE_TOKEN_STD_THRESHOLD))
    active_score_fraction = float(np.mean(score_event_mass > MIN_SCORE_EVENT_MASS))
    effects, gates = analyze_errors(
        errors,
        active_token_dims=active_token_dims,
        active_score_fraction=active_score_fraction,
    )
    decision = (
        "GO_FOR_SPATIAL_LOCALIZATION_SCREEN"
        if gates["all_passed"]
        else "STOP_EXPLICIT_EVENT_TOKENS"
    )

    provenance_path = output / "input_provenance.jsonl"
    with provenance_path.open("w") as handle:
        for record in provenance:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    per_clip_path = output / "per_clip_metrics.jsonl"
    with per_clip_path.open("w") as handle:
        for position, item in enumerate(score_items):
            handle.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "score_position": position,
                        "manifest_index": int(item["manifest_index"]),
                        "clip_id": item["clip_id"],
                        "motion_stratum": int(item["motion_stratum"]),
                        "donor_manifest_index": int(item["donor_manifest_index"]),
                        "score_event_mass": float(score_event_mass[position]),
                        "errors": {
                            name: float(values[position]) for name, values in errors.items()
                        },
                        "source_split": "train",
                        "validation_accessed": False,
                        "protected_test_accessed": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    arrays_path = output / "token_model_predictions.npz"
    np.savez_compressed(
        arrays_path,
        fit_indices=np.asarray([item["manifest_index"] for item in fit_items], dtype=np.int64),
        score_indices=np.asarray(
            [item["manifest_index"] for item in score_items], dtype=np.int64
        ),
        donor_positions=donor_positions,
        fit_history=fit_h,
        fit_target=fit_y,
        fit_action_pc=fit_action_pc,
        score_history=score_h,
        score_target=score_y,
        score_event_mass=score_event_mass,
        target_mean=target_mean,
        target_std=target_std,
        target_raw_std=target_raw_std,
        action_mean=action_mean,
        action_std=action_std,
        action_active=action_active,
        action_pca_mean=action_pca.mean_,
        action_pca_components=action_pca.components_,
        action_pca_explained_variance_ratio=action_pca.explained_variance_ratio_,
        history_input_mean=history_input_mean,
        history_input_std=history_input_std,
        action_input_mean=action_input_mean,
        action_input_std=action_input_std,
        history_model_coef=history_model.coef_,
        history_model_intercept=history_model.intercept_,
        action_model_coef=action_model.coef_,
        action_model_intercept=action_model.intercept_,
        **{f"prediction_{name}": value for name, value in predictions.items()},
        **errors,
    )

    latency = np.asarray(render_latencies, dtype=np.float64)
    analysis = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_token_stage0_analysis",
            "created_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "parent_registration_identity_sha256": registration["parent"]["registration"][
                "identity_sha256"
            ],
            "decision": decision,
            "interpretation": (
                "explicit global event integrals clear the causal gates and may proceed only to a location-bearing screen"
                if gates["all_passed"]
                else "explicit global event integrals fail at least one causal or salience gate; stop this interaction-event line before video training"
            ),
            "population": {
                "source_split": "train",
                "fit_clips": base.FIT_CLIPS,
                "score_clips": base.SCORE_CLIPS,
                "reuses_parent_score_for_sequential_diagnostic": True,
                "validation_accessed": False,
                "protected_test_accessed": False,
            },
            "models": {
                "history_alpha": history_alpha,
                "action_alpha": action_alpha,
                "history_five_fold_cv": history_cv,
                "action_five_fold_cv": action_cv,
                "action_pca_variance_retained": float(
                    action_pca.explained_variance_ratio_.sum()
                ),
                "prediction_action_sensitivity": {
                    label: float(np.sqrt(np.mean((predictions["aligned"] - predictions[label]) ** 2)))
                    for label in (
                        "episode_shuffled",
                        "train_mean",
                        "raw_zero",
                        "shift_minus1",
                        "shift_plus1",
                    )
                },
            },
            "target_salience": {
                "active_fit_token_dimensions": active_token_dims,
                "total_token_dimensions": 32,
                "raw_std_threshold": ACTIVE_TOKEN_STD_THRESHOLD,
                "active_score_fraction": active_score_fraction,
                "score_event_mass_threshold": MIN_SCORE_EVENT_MASS,
                "score_event_mass_mean": float(score_event_mass.mean()),
                "integral_reconstruction": (
                    "exact by construction: token channels are spatial means of the raw event fields"
                ),
            },
            "metrics": {
                "mean_errors": {name: float(values.mean()) for name, values in errors.items()},
                "effects": effects,
                "gates": gates,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "family_size": FAMILY_SIZE,
            },
            "renderer": {
                "official_abc_commit": official_commit,
                "official_scene": base.file_record(scene_path),
                "robot_only_xml_sha256": hashlib.sha256(robot_xml.encode()).hexdigest(),
                "render_pose_mean_ms": float(latency.mean()),
                "render_pose_p95_ms": float(np.percentile(latency, 95)),
                "render_pose_count_after_two_warmups_per_clip": int(len(latency)),
            },
            "artifacts": {
                "registration": base.file_record(output / "registration.json"),
                "frozen_source": base.file_record(frozen_source),
                "input_provenance": base.file_record(provenance_path),
                "per_clip_metrics": base.file_record(per_clip_path),
                "token_model_predictions": base.file_record(arrays_path),
            },
            "claim_boundary": (
                "Sequential train-only global-integral predictability. No spatial condition, generator, video metric, control rollout, validation, or protected test was evaluated."
            ),
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "analysis.json", analysis)
    complete = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_token_stage0_complete",
            "completed_at_utc": base.now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "decision": decision,
            "status": "completed",
            "artifacts": {
                "analysis": base.file_record(output / "analysis.json"),
                **analysis["artifacts"],
            },
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(output / "run_complete.json", complete)
    return analysis


def audit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    registration = base.read_json(output / "registration.json")
    analysis = base.read_json(output / "analysis.json")
    complete = base.read_json(output / "run_complete.json")
    for name, document in (
        ("registration", registration),
        ("analysis", analysis),
        ("complete", complete),
    ):
        base.validate_identity(document, name)
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise TokenError("analysis-registration binding differs")
    if complete["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise TokenError("completion-analysis binding differs")
    if complete["decision"] != analysis["decision"]:
        raise TokenError("completion decision differs")
    if base.sha256_file(Path(__file__).resolve()) != registration["source"]["sha256"]:
        raise TokenError("audit source differs")
    frozen = output / "frozen_interaction_event_token_stage0.py"
    if base.sha256_file(frozen) != registration["source"]["frozen_copy"]["sha256"]:
        raise TokenError("frozen source differs")

    for section in (analysis["artifacts"], complete["artifacts"]):
        for name, record in section.items():
            path = Path(record["path"])
            if not path.is_file():
                path = output / path.name
            if base.sha256_file(path) != record["sha256"]:
                raise TokenError(f"artifact hash differs: {name}")

    provenance = base.read_jsonl(output / "input_provenance.jsonl")
    per_clip = base.read_jsonl(output / "per_clip_metrics.jsonl")
    if len(provenance) != base.FIT_CLIPS + base.SCORE_CLIPS:
        raise TokenError("provenance count differs")
    if len(per_clip) != base.SCORE_CLIPS:
        raise TokenError("score row count differs")
    with np.load(output / "token_model_predictions.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if arrays["fit_target"].shape != (base.FIT_CLIPS, 32):
        raise TokenError("fit target geometry differs")
    if arrays["score_target"].shape != (base.SCORE_CLIPS, 32):
        raise TokenError("score target geometry differs")
    errors = {
        name: value
        for name, value in arrays.items()
        if name.startswith("all_token_error_")
        or name.startswith("change_error_")
        or name.startswith("transport_error_")
    }
    active_dims = int(
        np.sum(arrays["target_raw_std"] > ACTIVE_TOKEN_STD_THRESHOLD)
    )
    active_fraction = float(
        np.mean(arrays["score_event_mass"] > MIN_SCORE_EVENT_MASS)
    )
    effects, gates = analyze_errors(
        errors,
        active_token_dims=active_dims,
        active_score_fraction=active_fraction,
    )
    if json.dumps(effects, sort_keys=True) != json.dumps(
        analysis["metrics"]["effects"], sort_keys=True
    ):
        raise TokenError("recomputed effects differ")
    if gates != analysis["metrics"]["gates"]:
        raise TokenError("recomputed gates differ")
    expected_decision = (
        "GO_FOR_SPATIAL_LOCALIZATION_SCREEN"
        if gates["all_passed"]
        else "STOP_EXPLICIT_EVENT_TOKENS"
    )
    if analysis["decision"] != expected_decision:
        raise TokenError("decision does not follow gates")

    false_flags = 0
    for name, value in (
        ("registration", registration),
        ("analysis", analysis),
        ("complete", complete),
        ("provenance", provenance),
        ("per_clip", per_clip),
    ):
        false_flags += base.require_false_flags(value, name)
    return base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "interaction_event_token_stage0_audit",
            "audited_at_utc": base.now(),
            "status": "audit_passed",
            "decision": analysis["decision"],
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "provenance_rows": len(provenance),
            "score_rows": len(per_clip),
            "recomputed_mandatory_effects": len(effects["mandatory"]),
            "recomputed_diagnostic_effects": len(effects["diagnostic"]),
            "explicit_false_flags": false_flags,
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--parent-artifact", type=Path, required=True)
    register_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/experiments/INTERACTION_EVENT_TOKEN_STAGE0_PROTOCOL.md"),
    )
    register_parser.add_argument("--expected-commit")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--official-abc-root", type=Path, required=True)
    run_parser.add_argument("--mujoco-gl", default="egl")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        result = register(args)
        event = "registered"
    elif args.command == "run":
        result = run(args)
        event = "completed"
    elif args.command == "audit":
        result = audit(args.output)
        if args.report is not None:
            base.write_json(args.report, result)
        event = "audited"
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"event": event, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
