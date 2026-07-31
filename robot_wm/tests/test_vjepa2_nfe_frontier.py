import hashlib
import copy
import json
from pathlib import Path
from functools import lru_cache
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
from tools import vjepa2_nfe_frontier as frontier
from tools import evaluate_vjepa2_quality as evaluator
from tools import vjepa2_frontier_lockbox as lockbox
from tools import benchmark_vjepa2_frontier_latency as latency_benchmark
from tools.slurm import vjepa2_frontier_workflow as workflow


_TEST_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="vjepa2-nfe-frontier-tests-"
)
_TEST_ROOT = Path(_TEST_DIRECTORY.name)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _file_record(label: str) -> dict:
    return {"path": f"/unused/{label}", "sha256": _hash(label), "bytes": 1}


def _lockbox_registration() -> dict:
    compatibility = {
        "training_commit_is_ancestor": True,
        "inference_critical_paths_unchanged": True,
        "paths": {},
    }
    return lockbox.identity_payload(
        {
            "schema_version": 1,
            "kind": lockbox.KIND_REGISTRATION,
            "study_identity_sha256": _hash("study"),
            "training_git_commit": "1" * 40,
            "registration_git_commit": "2" * 40,
            "inference_code_compatibility": compatibility,
            "construction": {
                **_file_record("construction"),
                "identity_sha256": _hash("construction-identity"),
            },
            "manifest": {
                **_file_record("manifest"),
                "entries": lockbox.LOCKBOX_CLIPS,
            },
            "cache": {
                "metadata": _file_record("metadata"),
                "cache_id": _hash("cache"),
                "clip_count": lockbox.LOCKBOX_CLIPS,
                "arrays": {
                    name: _file_record(name)
                    for name in ("target", "rgb", "actions")
                },
            },
            "episode_isolation": {
                "lockbox_original_episode_overlap_counts": {
                    "train": 0,
                    "validation": 0,
                    "test": 0,
                }
            },
            "operator_attestation": {
                "lockbox_never_scored_before_registration": True,
                "text": lockbox.ATTESTATION_TEXT,
            },
            "selection_must_bind_registration_before_evaluation": True,
        }
    )


def _videox_runtime() -> dict:
    return {
        "path": "/unused/VideoX-Fun-1d6d9c3",
        "git_commit": frontier.EXPECTED_VIDEOX_COMMIT,
        "git_tree_sha": _hash("videox-tree"),
        "clean": True,
    }


def _row(
    *,
    arm: str,
    split: str | None,
    clip_prefix: str,
    clip_index: int,
    nfe: int,
    value: float,
    selection_identity: str | None = None,
    lockbox_identity: str | None = None,
) -> dict:
    input_hash = _hash(f"{clip_prefix}-{clip_index}")
    payload = {
        "schema_version": 1,
        "kind": "vjepa2_controlled_study_quality_clip",
        "arm_code": arm,
        "completed_updates": 1000,
        "clip_index": clip_index,
        "clip_id": f"{clip_prefix}-{clip_index}",
        "source": "autonomous",
        "oracle_leakage": False,
        "deployable_evidence": True,
        "sampler_entrypoint": (
            "DualExplicitActionDiTModel.sample_future_deployable"
        ),
        "clean_future_or_auxiliary_passed_to_sampler": False,
        "nfe": nfe,
        "video_history_latent_frames": 1,
        "auxiliary_history_latent_frames": 0,
        "online_teacher_call_count": 0,
        "actual_wan_call_count": nfe,
        "metrics": {
            "decoded_temporal_difference_mse_unit_range": value,
            "video_future_nmse": value,
            "decoded_mse_unit_range": value,
        },
        "tensor_sha256": {
            "video_clean_sha256": input_hash,
            "auxiliary_clean_sha256": input_hash,
            "ground_truth_sha256": input_hash,
            "vae_ground_truth_sha256": input_hash,
            "raw_history_last_sha256": input_hash,
            "vae_history_last_sha256": input_hash,
            "cached_rgb_input_sha256": input_hash,
            "cached_actions_input_sha256": input_hash,
            "video_initial_state_sha256": input_hash,
            "auxiliary_initial_state_sha256": input_hash,
            "auxiliary_initial_noise_sha256": input_hash,
        },
    }
    if split is not None:
        payload["evaluation_split"] = split
        payload["frontier_selection_identity_sha256"] = selection_identity
        payload["study_identity_sha256"] = _hash("study")
        payload["arm_identity_sha256"] = _hash(f"arm-{arm}")
        payload["stage_identity_sha256"] = _hash(f"stage-{arm}")
        payload["training_git_commit"] = "1" * 40
        payload["evaluator_git_commit"] = "2" * 40
        payload["inference_code_compatibility_sha256"] = hashlib.sha256(
            frontier._canonical_json(
                _lockbox_registration()["inference_code_compatibility"]
            )
        ).hexdigest()
        payload["videox_runtime_identity_sha256"] = hashlib.sha256(
            frontier._canonical_json(_videox_runtime())
        ).hexdigest()
        payload["evaluation_world_size"] = 8
        payload["evaluation_batch_size_per_rank"] = 2
        payload["sampling_namespace"] = split
        payload["sampling_id"] = (
            frontier.SAMPLE_ID_OFFSETS[split] + clip_index
        )
        if split == "lockbox":
            payload["lockbox_registration_identity_sha256"] = lockbox_identity
    return frontier.identity_payload(payload)


def _grid_rows(
    arm: str,
    *,
    split: str | None,
    clip_prefix: str,
    clips: int,
    values: dict[int, float],
    selection_identity: str | None = None,
    lockbox_identity: str | None = None,
) -> list[dict]:
    return [
        _row(
            arm=arm,
            split=split,
            clip_prefix=clip_prefix,
            clip_index=clip_index,
            nfe=nfe,
            value=values[nfe],
            selection_identity=selection_identity,
            lockbox_identity=lockbox_identity,
        )
        for clip_index in range(clips)
        for nfe in values
    ]


def _resign(payload: dict) -> dict:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("identity_sha256", None)
    return frontier.identity_payload(unsigned)


def _write_jsonl(label: str, rows: list[dict]) -> Path:
    path = _TEST_ROOT / f"{label}.jsonl"
    path.write_bytes(
        b"".join(frontier._canonical_json(row) + b"\n" for row in rows)
    )
    return path.resolve()


def _row_evidence(
    label: str,
    j1_rows: list[dict],
    vpm_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    j1_path = _write_jsonl(f"{label}-j1", j1_rows)
    vpm_path = _write_jsonl(f"{label}-vpm", vpm_rows)
    return (
        frontier._input_records([j1_path]),
        frontier._input_records([vpm_path]),
    )


def _registration_evidence() -> tuple[dict, dict]:
    registration = _lockbox_registration()
    path = (_TEST_ROOT / "lockbox-registration.json").resolve()
    path.write_text(
        json.dumps(registration, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    record = {
        **frontier._input_records([path])[0],
        "identity_sha256": registration["identity_sha256"],
    }
    return registration, record


@lru_cache(maxsize=1)
def _validation_selection(clips: int = 64) -> dict:
    vpm_values = {
        1: 0.50,
        2: 0.30,
        4: 0.20,
        6: 0.25,
        8: 0.30,
        12: 0.35,
        20: 0.40,
    }
    j1_values = {
        1: 0.001,  # Must not become a causal candidate.
        2: 0.15,
        4: 0.19,
        6: 0.24,
        8: 0.29,
        12: 0.34,
        20: 0.39,
    }
    j1_rows = _grid_rows(
        "J1",
        split="validation",
        clip_prefix="val",
        clips=clips,
        values=j1_values,
    )
    vpm_rows = _grid_rows(
        "VPM",
        split="validation",
        clip_prefix="val",
        clips=clips,
        values=vpm_values,
    )
    j1_inputs, vpm_inputs = _row_evidence(
        f"validation-{clips}", j1_rows, vpm_rows
    )
    registration, registration_input = _registration_evidence()
    return frontier.build_selection(
        j1_rows=j1_rows,
        vpm_rows=vpm_rows,
        split="validation",
        expected_clips=clips,
        bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=0.95,
        seed=1234,
        allow_posthoc=False,
        lockbox_registration=registration,
        lockbox_input=registration_input,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )


@lru_cache(maxsize=1)
def _lockbox_confirmation() -> dict:
    selection = _validation_selection()
    selection_id = selection["identity_sha256"]
    lockbox_id = selection["lockbox_registration"]["identity_sha256"]
    k = selection["selected_pair"]["left"]["nfe"]
    m = selection["selected_pair"]["reference"]["nfe"]
    j1_rows = _grid_rows(
        "J1",
        split="lockbox",
        clip_prefix="lockbox",
        clips=128,
        values={k: 0.10},
        selection_identity=selection_id,
        lockbox_identity=lockbox_id,
    )
    vpm_rows = _grid_rows(
        "VPM",
        split="lockbox",
        clip_prefix="lockbox",
        clips=128,
        values={k: 0.11, m: 0.12},
        selection_identity=selection_id,
        lockbox_identity=lockbox_id,
    )
    j1_inputs, vpm_inputs = _row_evidence(
        "lockbox-confirmation", j1_rows, vpm_rows
    )
    return frontier.build_confirmation(
        selection=selection,
        j1_rows=j1_rows,
        vpm_rows=vpm_rows,
        expected_clips=128,
        allow_posthoc=False,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )


def _write_canonical_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frontier._canonical_json(payload) + b"\n")
    return path.resolve()


def _retarget_rows(
    rows: list[dict],
    *,
    study_identity: str,
    training_commit: str,
    evaluator_commit: str,
    registration: dict,
) -> list[dict]:
    compatibility_identity = hashlib.sha256(
        frontier._canonical_json(
            registration["inference_code_compatibility"]
        )
    ).hexdigest()
    videox_identity = hashlib.sha256(
        frontier._canonical_json(_videox_runtime())
    ).hexdigest()
    rebound = []
    for row in rows:
        value = copy.deepcopy(row)
        value.pop("identity_sha256", None)
        value.update(
            {
                "study_identity_sha256": study_identity,
                "training_git_commit": training_commit,
                "evaluator_git_commit": evaluator_commit,
                "inference_code_compatibility_sha256": (
                    compatibility_identity
                ),
                "videox_runtime_identity_sha256": videox_identity,
            }
        )
        rebound.append(frontier.identity_payload(value))
    return rebound


def _bound_cli_fixture(
    root: Path,
    *,
    quality_passes: bool,
) -> dict:
    study_root = (root / "study").resolve()
    study_root.mkdir(parents=True)
    training_commit = "1" * 40
    scientific_commit = "2" * 40
    controller_commit = "4" * 40
    study = frontier.identity_payload(
        {
            "kind": "vjepa2_controlled_video_diffusion_study",
            "study_id": f"fixture-{root.name}",
            "study_root": str(study_root),
            "inputs": {
                "repository": {
                    "root": "/unused/training",
                    "git_commit": training_commit,
                }
            },
        }
    )
    _write_canonical_json(study_root / "study_manifest.json", study)

    registration = copy.deepcopy(_lockbox_registration())
    registration.pop("identity_sha256")
    registration.update(
        {
            "study_identity_sha256": study["identity_sha256"],
            "training_git_commit": training_commit,
            "registration_git_commit": scientific_commit,
        }
    )
    registration = lockbox.identity_payload(registration)
    registration_path = _write_canonical_json(
        study_root / "frontier_lockbox" / "registration.json",
        registration,
    )
    registration_input = {
        **frontier._input_records([registration_path])[0],
        "identity_sha256": registration["identity_sha256"],
    }

    vpm_values = {
        1: 0.50,
        2: 0.30,
        4: 0.20,
        6: 0.25,
        8: 0.30,
        12: 0.35,
        20: 0.40,
    }
    j1_values = {
        1: 0.001,
        2: 0.15,
        4: 0.19,
        6: 0.24,
        8: 0.29,
        12: 0.34,
        20: 0.39,
    }
    validation_j1 = _retarget_rows(
        _grid_rows(
            "J1",
            split="validation",
            clip_prefix=f"{root.name}-val",
            clips=64,
            values=j1_values,
        ),
        study_identity=study["identity_sha256"],
        training_commit=training_commit,
        evaluator_commit=scientific_commit,
        registration=registration,
    )
    validation_vpm = _retarget_rows(
        _grid_rows(
            "VPM",
            split="validation",
            clip_prefix=f"{root.name}-val",
            clips=64,
            values=vpm_values,
        ),
        study_identity=study["identity_sha256"],
        training_commit=training_commit,
        evaluator_commit=scientific_commit,
        registration=registration,
    )
    j1_inputs, vpm_inputs = _row_evidence(
        f"{root.name}-validation", validation_j1, validation_vpm
    )
    selection = frontier.build_selection(
        j1_rows=validation_j1,
        vpm_rows=validation_vpm,
        split="validation",
        expected_clips=64,
        bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=frontier.DEFAULT_CONFIDENCE,
        seed=frontier.DEFAULT_SEED,
        allow_posthoc=False,
        lockbox_registration=registration,
        lockbox_input=registration_input,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )
    selection_path = _write_canonical_json(
        study_root / "frontier_selection.json", selection
    )

    k = selection["selected_pair"]["left"]["nfe"]
    m = selection["selected_pair"]["reference"]["nfe"]
    same_nfe_reference = 0.11 if quality_passes else 0.09
    lockbox_j1 = _retarget_rows(
        _grid_rows(
            "J1",
            split="lockbox",
            clip_prefix=f"{root.name}-lockbox",
            clips=128,
            values={k: 0.10},
            selection_identity=selection["identity_sha256"],
            lockbox_identity=registration["identity_sha256"],
        ),
        study_identity=study["identity_sha256"],
        training_commit=training_commit,
        evaluator_commit=scientific_commit,
        registration=registration,
    )
    lockbox_vpm = _retarget_rows(
        _grid_rows(
            "VPM",
            split="lockbox",
            clip_prefix=f"{root.name}-lockbox",
            clips=128,
            values={k: same_nfe_reference, m: 0.12},
            selection_identity=selection["identity_sha256"],
            lockbox_identity=registration["identity_sha256"],
        ),
        study_identity=study["identity_sha256"],
        training_commit=training_commit,
        evaluator_commit=scientific_commit,
        registration=registration,
    )
    j1_inputs, vpm_inputs = _row_evidence(
        f"{root.name}-confirmation", lockbox_j1, lockbox_vpm
    )
    confirmation = frontier.build_confirmation(
        selection=selection,
        j1_rows=lockbox_j1,
        vpm_rows=lockbox_vpm,
        expected_clips=128,
        allow_posthoc=False,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )
    confirmation_path = _write_canonical_json(
        study_root / "frontier_lockbox_confirmation.json",
        confirmation,
    )
    return {
        "study_root": study_root,
        "study": study,
        "training_commit": training_commit,
        "scientific_commit": scientific_commit,
        "controller_commit": controller_commit,
        "selection": selection,
        "selection_path": selection_path,
        "confirmation": confirmation,
        "confirmation_path": confirmation_path,
        "output": study_root / "frontier_final_report.json",
    }


def _run_outcome_cli(
    fixture: dict, **path_overrides: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "tools"
                / "slurm"
                / "vjepa2_frontier_workflow.py"
            ),
            "finalize-quality-failure",
            "--study-root",
            str(path_overrides.get("study_root", fixture["study_root"])),
            "--training-commit",
            fixture["training_commit"],
            "--selection",
            str(path_overrides.get("selection", fixture["selection_path"])),
            "--confirmation",
            str(
                path_overrides.get(
                    "confirmation", fixture["confirmation_path"]
                )
            ),
            "--output",
            str(path_overrides.get("output", fixture["output"])),
            "--controller-commit",
            fixture["controller_commit"],
            "--scientific-commit",
            fixture["scientific_commit"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _latency_artifact() -> dict:
    selection = _validation_selection()
    compatibility = copy.deepcopy(
        selection["lockbox_registration"]["inference_code_compatibility"]
    )
    k = selection["selected_pair"]["left"]["nfe"]
    m = selection["selected_pair"]["reference"]["nfe"]
    values = {
        "J1_k": [5.0 + (index % 3) * 0.01 for index in range(120)],
        "VPM_k": [4.5 + (index % 3) * 0.01 for index in range(120)],
        "VPM_m": [10.0 + (index % 3) * 0.01 for index in range(120)],
    }
    orders = [
        list(latency_benchmark.balanced_order(index)) for index in range(120)
    ]
    summaries = {
        name: latency_benchmark.latency_summary(endpoint_values)
        for name, endpoint_values in values.items()
    }
    frontier_effect = latency_benchmark.paired_timing_effect(
        values["J1_k"],
        values["VPM_m"],
        orders,
        left_label="J1_k",
        reference_label="VPM_m",
        bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=frontier.DEFAULT_CONFIDENCE,
        seed=frontier.DEFAULT_SEED,
        label=f"frontier-latency:J1-{k}-vs-VPM-{m}",
    )
    same_effect = latency_benchmark.paired_timing_effect(
        values["J1_k"],
        values["VPM_k"],
        orders,
        left_label="J1_k",
        reference_label="VPM_k",
        bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=frontier.DEFAULT_CONFIDENCE,
        seed=frontier.DEFAULT_SEED,
        label=f"same-nfe-latency:J1-{k}-vs-VPM-{k}",
    )
    timing_gate = latency_benchmark.timing_gate(
        frontier_effect,
        left_p95=summaries["J1_k"]["p95"],
        reference_p95=summaries["VPM_m"]["p95"],
    )
    endpoints = {}
    for name, arm, nfe in (
        ("J1_k", "J1", k),
        ("VPM_k", "VPM", k),
        ("VPM_m", "VPM", m),
    ):
        p95 = summaries[name]["p95"]
        endpoints[name] = {
            "arm": arm,
            "source": "autonomous",
            "nfe": nfe,
            "actual_wan_calls": nfe,
            "audit": {"actual_wan_calls": nfe},
            "latency_ms": summaries[name],
            "generated_frames_per_second_at_p95": 8000.0 / p95,
            "peak_allocated_bytes_with_both_models_resident": 100,
            "audit_peak_allocated_bytes_with_artifact_capture": 120,
        }
    return frontier.identity_payload(
        {
            "schema_version": 1,
            "kind": frontier.KIND_LATENCY,
            "training_git_commit": selection["training_git_commit"],
            "evaluator_git_commit": selection["evaluator_git_commit"],
            "benchmark_git_commit": "3" * 40,
            "inference_code_compatibility": {
                "evaluator": copy.deepcopy(compatibility),
                "benchmark": copy.deepcopy(compatibility),
            },
            "videox_runtime": _videox_runtime(),
            "lockbox_registration_identity_sha256": selection[
                "lockbox_registration"
            ]["identity_sha256"],
            "selection": {
                **_file_record("selection"),
                "identity_sha256": selection["identity_sha256"],
            },
            "protocol": {
                "confirmatory_protocol": True,
                "warmup_rounds": 18,
                "timed_rounds": 120,
                "bootstrap_samples": frontier.DEFAULT_BOOTSTRAP_SAMPLES,
                "confidence": frontier.DEFAULT_CONFIDENCE,
                "bootstrap_seed": frontier.DEFAULT_SEED,
                "same_process": True,
                "same_B200": True,
                "both_models_resident": True,
                "sampling_namespace": "lockbox",
                "sampling_id": frontier.SAMPLE_ID_OFFSETS["lockbox"],
                "balanced_order_cycle": [
                    list(order) for order in latency_benchmark.BALANCED_ORDERS
                ],
            },
            "endpoints": endpoints,
            "model_provenance": {
                arm: {
                    "arm_manifest": {
                        "identity_sha256": selection[
                            "arm_identity_sha256"
                        ][arm]
                    },
                    "stage_manifest": {
                        "identity_sha256": selection[
                            "stage_identity_sha256"
                        ][arm]
                    },
                }
                for arm in ("J1", "VPM")
            },
            "frontier_acceleration": {
                "comparison": f"J1@{k} vs VPM@{m}",
                "paired_speed_effect": frontier_effect,
                "timing_gate": timing_gate,
            },
            "same_nfe_overhead": {
                "comparison": f"J1@{k} vs VPM@{k}",
                "definition": "positive relative_overhead means J1 is slower",
                "relative_overhead": -same_effect["relative_improvement"],
                "relative_overhead_percent": (
                    -100.0 * same_effect["relative_improvement"]
                ),
                "bootstrap_ci": {
                    "confidence": same_effect["bootstrap_ci"]["confidence"],
                    "low": -same_effect["bootstrap_ci"]["high"],
                    "high": -same_effect["bootstrap_ci"]["low"],
                },
                "paired_speed_effect": same_effect,
            },
            "device": {
                "name": "NVIDIA B200",
                "both_models_resident": True,
                "resident_allocated_bytes_before_timing": 100,
                "peak_allocated_bytes_during_timing": 120,
            },
            "rounds_sha256": hashlib.sha256(
                frontier._canonical_json(
                    [
                        {
                            "round_index": index,
                            "execution_order": orders[index],
                            "latency_ms": {
                                name: values[name][index]
                                for name in latency_benchmark.ENDPOINT_LABELS
                            },
                        }
                        for index in range(120)
                    ]
                )
            ).hexdigest(),
            "rounds": [
                {
                    "round_index": index,
                    "execution_order": orders[index],
                    "latency_ms": {
                        name: values[name][index]
                        for name in latency_benchmark.ENDPOINT_LABELS
                    },
                }
                for index in range(120)
            ],
        }
    )


def test_validation_selects_only_non_dominated_vpm_frontier():
    selection = _validation_selection()

    assert selection["vpm_non_dominated_nfe_frontier"] == [1, 2, 4]
    assert selection["selected_pair"]["left"]["nfe"] == 2
    assert selection["selected_pair"]["reference"]["nfe"] == 4
    assert selection["selected_pair"]["quality_gate"]["passed"] is True
    assert selection["confirmatory_eligible"] is True
    assert selection["status"] == "validation_selected_confirmatory_candidate"


def test_nfe_one_is_excluded_from_j1_causal_candidates():
    selection = _validation_selection()

    assert selection["selected_pair"]["left"]["nfe"] != 1
    assert all(
        candidate["left"]["nfe"] >= frontier.MIN_CAUSAL_J1_NFE
        for candidate in selection["candidates"]
    )


def test_confirmatory_selection_requires_registered_lockbox():
    values = {nfe: 0.2 for nfe in frontier.NFE_GRID}
    with pytest.raises(frontier.FrontierError, match="explicit validation"):
        frontier.build_selection(
            j1_rows=_grid_rows(
                "J1",
                split="validation",
                clip_prefix="val",
                clips=64,
                values=values,
            ),
            vpm_rows=_grid_rows(
                "VPM",
                split="validation",
                clip_prefix="val",
                clips=64,
                values=values,
            ),
            split="validation",
            expected_clips=64,
            bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
            confidence=frontier.DEFAULT_CONFIDENCE,
            seed=frontier.DEFAULT_SEED,
            allow_posthoc=False,
        )


def test_equal_quality_vpm_grid_keeps_only_lowest_compute_point():
    values = {nfe: 0.2 for nfe in frontier.NFE_GRID}
    rows = _grid_rows(
        "VPM",
        split="validation",
        clip_prefix="equal",
        clips=64,
        values=values,
    )
    validated, _bound = frontier.validate_arm_rows(
        rows,
        arm="VPM",
        split="validation",
        required_nfes=frontier.NFE_GRID,
        expected_clips=64,
        selection_identity=None,
        allow_posthoc=False,
    )

    nfe_frontier, _evidence = frontier.robust_vpm_frontier(
        validated,
        bootstrap_samples=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=frontier.DEFAULT_CONFIDENCE,
        seed=frontier.DEFAULT_SEED,
    )

    assert nfe_frontier == [1]


def test_lockbox_confirmation_is_frozen_bound_and_disjoint():
    selection = _validation_selection()
    selection_id = selection["identity_sha256"]
    k = selection["selected_pair"]["left"]["nfe"]
    m = selection["selected_pair"]["reference"]["nfe"]
    lockbox_id = selection["lockbox_registration"]["identity_sha256"]
    j1_rows = _grid_rows(
        "J1",
        split="lockbox",
        clip_prefix="lockbox",
        clips=128,
        values={k: 0.10},
        selection_identity=selection_id,
        lockbox_identity=lockbox_id,
    )
    vpm_rows = _grid_rows(
        "VPM",
        split="lockbox",
        clip_prefix="lockbox",
        clips=128,
        values={k: 0.11, m: 0.12},
        selection_identity=selection_id,
        lockbox_identity=lockbox_id,
    )
    j1_inputs, vpm_inputs = _row_evidence(
        "lockbox-confirmation-direct", j1_rows, vpm_rows
    )

    result = frontier.build_confirmation(
        selection=selection,
        j1_rows=j1_rows,
        vpm_rows=vpm_rows,
        expected_clips=128,
        allow_posthoc=False,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )

    assert result["lockbox_selection_decisions"] == 0
    assert result["validation_lockbox_clip_units_disjoint"] is True
    assert result["rows_explicitly_bound_to_frozen_selection"] is True
    assert result["confirmatory_evidence"] is True
    assert result["frontier_quality_gate_passed"] is True
    assert result["same_nfe_quality_attribution_gate_passed"] is True
    assert result["quality_gate_passed"] is True


def test_frontier_gain_cannot_hide_failed_same_nfe_attribution():
    selection = _validation_selection()
    selection_id = selection["identity_sha256"]
    k = selection["selected_pair"]["left"]["nfe"]
    m = selection["selected_pair"]["reference"]["nfe"]
    lockbox_id = selection["lockbox_registration"]["identity_sha256"]
    j1_rows = _grid_rows(
        "J1",
        split="lockbox",
        clip_prefix="lockbox",
        clips=128,
        values={k: 0.10},
        selection_identity=selection_id,
        lockbox_identity=lockbox_id,
    )
    vpm_rows = _grid_rows(
        "VPM",
        split="lockbox",
        clip_prefix="lockbox",
        clips=128,
        values={k: 0.09, m: 0.12},
        selection_identity=selection_id,
        lockbox_identity=lockbox_id,
    )
    j1_inputs, vpm_inputs = _row_evidence(
        "lockbox-confirmation-failed-attribution", j1_rows, vpm_rows
    )

    result = frontier.build_confirmation(
        selection=selection,
        j1_rows=j1_rows,
        vpm_rows=vpm_rows,
        expected_clips=128,
        allow_posthoc=False,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )

    assert result["frontier_quality_gate_passed"] is True
    assert result["same_nfe_quality_attribution_gate_passed"] is False
    assert result["quality_gate_passed"] is False


def test_negative_cli_is_study_bound_atomic_and_idempotent(tmp_path):
    fixture = _bound_cli_fixture(tmp_path, quality_passes=False)

    first = _run_outcome_cli(fixture)

    assert first.returncode == 0, first.stderr
    first_bytes = fixture["output"].read_bytes()
    report = json.loads(first_bytes)
    assert frontier.identity_valid(report)
    assert report["kind"] == frontier.KIND_FINAL
    assert report["status"] == "NOT_DEMONSTRATED"
    assert report["termination_reason"] == (
        "HELD_OUT_LOCKBOX_QUALITY_GATE_FAILED"
    )
    assert report["study_identity_sha256"] == fixture["study"][
        "identity_sha256"
    ]
    assert report["videox_runtime_identity_sha256"] == fixture["selection"][
        "videox_runtime_identity_sha256"
    ]
    assert report["quality_gate_passed"] is False
    assert report["timing_performed"] is False
    assert report["timing_gate_passed"] is None
    assert report["latency_identity_sha256"] is None
    assert report["frontier_acceleration"] is None
    assert report["faster_with_better_held_out_reconstruction_demonstrated"] is False
    assert report["timing"]["claim_permitted"] is False
    assert not list(
        fixture["study_root"].glob(".frontier_final_report.json.tmp.*")
    )

    second = _run_outcome_cli(fixture)

    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["idempotent_recovery"] is True
    assert fixture["output"].read_bytes() == first_bytes


def test_negative_cli_rejects_cross_study_artifact_paths(tmp_path):
    first = _bound_cli_fixture(tmp_path / "first", quality_passes=False)
    second = _bound_cli_fixture(tmp_path / "second", quality_passes=False)

    completed = _run_outcome_cli(
        second,
        selection=first["selection_path"],
        confirmation=first["confirmation_path"],
    )

    assert completed.returncode == 2
    assert "exact study artifact" in completed.stderr
    assert not second["output"].exists()

    second["selection_path"].write_bytes(first["selection_path"].read_bytes())
    second["confirmation_path"].write_bytes(
        first["confirmation_path"].read_bytes()
    )
    embedded_mismatch = _run_outcome_cli(second)

    assert embedded_mismatch.returncode == 2
    assert "does not bind the requested study" in embedded_mismatch.stderr
    assert not second["output"].exists()


def test_negative_cli_rejects_noncanonical_output_path(tmp_path):
    fixture = _bound_cli_fixture(tmp_path, quality_passes=False)
    wrong_output = fixture["study_root"] / "another-report.json"

    completed = _run_outcome_cli(fixture, output=wrong_output)

    assert completed.returncode == 2
    assert "exact study artifact" in completed.stderr
    assert not wrong_output.exists()
    assert not fixture["output"].exists()


def test_negative_cli_rejects_invalid_evidence_without_output(tmp_path):
    fixture = _bound_cli_fixture(tmp_path, quality_passes=False)
    confirmation = copy.deepcopy(fixture["confirmation"])
    confirmation["quality_gate_passed"] = True
    confirmation = _resign(confirmation)
    _write_canonical_json(fixture["confirmation_path"], confirmation)

    completed = _run_outcome_cli(fixture)

    assert completed.returncode == 2
    assert "differs from recomputed raw evidence" in completed.stderr
    assert not fixture["output"].exists()


def test_negative_cli_never_replaces_non_equivalent_existing_report(tmp_path):
    fixture = _bound_cli_fixture(tmp_path, quality_passes=False)
    existing = b'{"status":"partial-or-foreign"}\n'
    fixture["output"].write_bytes(existing)

    completed = _run_outcome_cli(fixture)

    assert completed.returncode == 2
    assert "identity is invalid" in completed.stderr
    assert fixture["output"].read_bytes() == existing


def test_atomic_negative_publication_never_exposes_partial_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "frontier_final_report.json"

    def fail_link(_temporary, _output):
        raise OSError("injected publication failure")

    monkeypatch.setattr(workflow.os, "link", fail_link)
    with pytest.raises(OSError, match="injected publication failure"):
        workflow._atomic_publish_no_replace(output, {"complete": True})

    assert not output.exists()
    assert not list(tmp_path.glob(".frontier_final_report.json.tmp.*"))


def test_positive_cli_returns_four_without_output(tmp_path):
    fixture = _bound_cli_fixture(tmp_path, quality_passes=True)

    completed = _run_outcome_cli(fixture)

    assert completed.returncode == workflow.HELD_OUT_QUALITY_PASSED
    payload = json.loads(completed.stdout)
    assert payload["paired_timing_required"] is True
    assert payload["final_report_written"] is False
    assert not fixture["output"].exists()


def test_legacy_test_grid_is_always_labeled_posthoc():
    values = {nfe: 0.2 + nfe / 1000 for nfe in frontier.NFE_GRID}
    result = frontier.build_selection(
        j1_rows=_grid_rows(
            "J1",
            split=None,
            clip_prefix="legacy",
            clips=8,
            values=values,
        ),
        vpm_rows=_grid_rows(
            "VPM",
            split=None,
            clip_prefix="legacy",
            clips=8,
            values=values,
        ),
        split="test",
        expected_clips=8,
        bootstrap_samples=500,
        confidence=0.95,
        seed=1234,
        allow_posthoc=True,
    )

    assert result["status"] == "posthoc_exploratory_selection"
    assert result["confirmatory_eligible"] is False
    assert result["selection_used_test_metrics"] is True


def test_frontier_evaluator_grid_is_validation_full_then_lockbox_frozen():
    selection = _validation_selection()

    assert evaluator.frontier_validation_grid() == tuple(
        ("autonomous", nfe) for nfe in frontier.NFE_GRID
    )
    assert evaluator.frontier_test_grid("J1", selection) == (
        ("autonomous", 2),
    )
    assert evaluator.frontier_test_grid("VPM", selection) == (
        ("autonomous", 2),
        ("autonomous", 4),
    )


@pytest.mark.parametrize(
    ("expected_clips", "bootstrap_samples", "confidence", "seed"),
    [
        (8, frontier.DEFAULT_BOOTSTRAP_SAMPLES, 0.95, 1234),
        (64, 500, 0.95, 1234),
        (64, frontier.DEFAULT_BOOTSTRAP_SAMPLES, 0.51, 1234),
        (64, frontier.DEFAULT_BOOTSTRAP_SAMPLES, 0.95, 7),
    ],
)
def test_confirmatory_selection_cli_rejects_unpinned_protocol(
    expected_clips, bootstrap_samples, confidence, seed
):
    args = SimpleNamespace(
        allow_posthoc=False,
        split="validation",
        expected_clips=expected_clips,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
        lockbox_registration="lockbox.json",
    )

    with pytest.raises(frontier.FrontierError, match="pinned 64-clip"):
        frontier._command_select(args)


def test_confirmatory_test_cli_rejects_subsets():
    args = SimpleNamespace(allow_posthoc=False, expected_clips=10)

    with pytest.raises(frontier.FrontierError, match="all 128"):
        frontier._command_confirm(args)


def test_final_report_recomputes_valid_bound_evidence():
    result = frontier.build_final_report(
        selection=_validation_selection(),
        confirmation=_lockbox_confirmation(),
        latency=_latency_artifact(),
    )

    assert result["status"] == "PASS"
    assert (
        result["faster_with_better_held_out_reconstruction_demonstrated"]
        is True
    )


def test_resigned_posthoc_selection_cannot_be_promoted_to_confirmatory():
    selection = copy.deepcopy(_validation_selection())
    selection.update(
        {
            "status": "posthoc_exploratory_selection",
            "selection_split": "test",
            "selection_used_test_metrics": True,
            "legacy_or_unbound_rows_used": True,
        }
    )
    selection = _resign(selection)

    with pytest.raises(frontier.FrontierError, match="pinned confirmatory"):
        frontier.build_final_report(
            selection=selection,
            confirmation=_lockbox_confirmation(),
            latency=_latency_artifact(),
        )


def test_resigned_selection_cannot_replace_computed_winner():
    selection = copy.deepcopy(_validation_selection())
    selection["vpm_non_dominated_nfe_frontier"].append(8)
    selection["selected_pair"]["reference"]["nfe"] = 8
    selection = _resign(selection)

    with pytest.raises(frontier.FrontierError, match="reproduce"):
        frontier.build_final_report(
            selection=selection,
            confirmation=_lockbox_confirmation(),
            latency=_latency_artifact(),
        )


def test_final_report_rejects_missing_or_posthoc_quality_evidence():
    confirmation = copy.deepcopy(_lockbox_confirmation())
    confirmation["input_evidence"] = {"J1": [], "VPM": []}
    confirmation["posthoc_exploratory"] = True
    confirmation = _resign(confirmation)

    with pytest.raises(
        frontier.FrontierError, match="posthoc or lacks raw"
    ):
        frontier.build_final_report(
            selection=_validation_selection(),
            confirmation=confirmation,
            latency=_latency_artifact(),
        )


def test_final_report_recomputes_quality_from_raw_rows():
    confirmation = copy.deepcopy(_lockbox_confirmation())
    effect = confirmation["frontier_quality_comparison"]["metrics"][
        frontier.PRIMARY_METRIC
    ]
    effect["relative_improvement"] = 0.999
    effect["relative_improvement_percent"] = 99.9
    confirmation = _resign(confirmation)

    with pytest.raises(frontier.FrontierError, match="does not reproduce"):
        frontier.build_final_report(
            selection=_validation_selection(),
            confirmation=confirmation,
            latency=_latency_artifact(),
        )


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    [
        (
            "latency",
            lambda value: value["endpoints"]["VPM_m"].update({"nfe": 8}),
            "endpoint accounting",
        ),
        (
            "latency",
            lambda value: value.update(
                {"lockbox_registration_identity_sha256": "9" * 64}
            ),
            "selection/lockbox",
        ),
        (
            "latency",
            lambda value: value["model_provenance"]["J1"][
                "stage_manifest"
            ].update({"identity_sha256": "9" * 64}),
            "arm/stage identity",
        ),
        (
            "latency",
            lambda value: value["inference_code_compatibility"][
                "benchmark"
            ].update({"training_commit_is_ancestor": False}),
            "Git or study provenance",
        ),
        (
            "confirmation",
            lambda value: value.update({"quality_gate_passed": False}),
            "posthoc or lacks raw",
        ),
    ],
)
def test_final_report_rejects_resigned_mismatched_artifacts(
    target, mutate, message
):
    confirmation = copy.deepcopy(_lockbox_confirmation())
    latency = _latency_artifact()
    artifact = confirmation if target == "confirmation" else latency
    mutate(artifact)
    artifact = _resign(artifact)
    if target == "confirmation":
        confirmation = artifact
    else:
        latency = artifact

    with pytest.raises(frontier.FrontierError, match=message):
        frontier.build_final_report(
            selection=_validation_selection(),
            confirmation=confirmation,
            latency=latency,
        )
