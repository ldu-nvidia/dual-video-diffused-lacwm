"""Focused tests for the separate fresh-lockbox within-J1 causality sidecar."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools import evaluate_vjepa2_quality as quality
from tools import vjepa2_lockbox_causality as causality
from tools import vjepa2_nfe_frontier as frontier


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "tools/slurm/submit_vjepa2_frontier_causality.sh"
QUALITY_SBATCH = ROOT / "tools/slurm/vjepa2_frontier_causality_quality.sbatch"
CONFIRM_SBATCH = ROOT / "tools/slurm/vjepa2_frontier_causality_confirm.sbatch"
OLD_COMMIT = "a" * 40
NEW_COMMIT = "b" * 40
SELECTION_ID = "c" * 64
LOCKBOX_ID = "d" * 64
STUDY_ID = "e" * 64
ARM_ID = "f" * 64
STAGE_ID = "1" * 64
COMPATIBILITY_ID = "2" * 64
VIDEOX_ID = "3" * 64
CONTINUATION_SHA = "4" * 64
AUTONOMOUS_INVENTORY_ID = "5" * 64
AUTONOMOUS_INVENTORY_SHA = "6" * 64


def _selection() -> dict:
    return {
        "identity_sha256": SELECTION_ID,
        "selected_pair": {
            "left": {"arm": "J1", "nfe": 2},
            "reference": {"arm": "VPM", "nfe": 4},
        },
        "study_identity_sha256": STUDY_ID,
        "training_git_commit": "9" * 40,
        "evaluator_git_commit": OLD_COMMIT,
        "inference_code_compatibility_sha256": COMPATIBILITY_ID,
        "videox_runtime_identity_sha256": VIDEOX_ID,
        "arm_identity_sha256": {"J1": ARM_ID},
        "stage_identity_sha256": {"J1": STAGE_ID},
        "lockbox_registration": {"identity_sha256": LOCKBOX_ID},
    }


def _rank_inputs() -> dict:
    return {
        key: {"path": f"/immutable/{key}", "sha256": "7" * 64}
        for key in (
            "resolved_config",
            "snapshot",
            "arm_manifest",
            "study_manifest",
            "stage_manifest",
            "stage_outcome",
            "evaluation_clip_manifest",
            "evaluation_cache_metadata",
            "evaluation_cache_arrays",
            "lockbox_registration",
        )
    }


def _rank_provenance() -> dict:
    return {
        "arm_identity_sha256": ARM_ID,
        "study_identity_sha256": STUDY_ID,
        "stage_identity_sha256": STAGE_ID,
        "stage_outcome_identity_sha256": "8" * 64,
        "git_commit": "9" * 40,
        "training_git_commit": "9" * 40,
        "frontier_selection_identity_sha256": SELECTION_ID,
        "lockbox_registration_identity_sha256": LOCKBOX_ID,
        "inference_code_compatibility": {"unchanged": True},
        "videox_runtime": {"commit": "fixed"},
    }


def _sidecar_provenance() -> dict:
    return {
        "frontier_primary_evaluator_git_commit": OLD_COMMIT,
        "frontier_continuation_sha256": CONTINUATION_SHA,
        "frontier_continuation_lockbox_quality_array_job_id": "481700",
        "frontier_autonomous_inventory_identity_sha256": (
            AUTONOMOUS_INVENTORY_ID
        ),
        "frontier_autonomous_inventory_sha256": AUTONOMOUS_INVENTORY_SHA,
    }


def _row(index: int, source: str, metric: float) -> dict:
    tensor_hashes = {
        field: hashlib.sha256(f"{index}:{field}".encode()).hexdigest()
        for field in causality.PAIRING_HASH_FIELDS
    }
    tensor_hashes["auxiliary_initial_state_sha256"] = tensor_hashes[
        "auxiliary_initial_noise_sha256"
    ]
    tensor_hashes.update(
        {
            field: hashlib.sha256(
                f"{index}:{source}:{field}".encode()
            ).hexdigest()
            for field in causality.OUTPUT_HASH_FIELDS
        }
    )
    sidecar = (
        {}
        if source == "autonomous"
        else {"frontier_causality_sidecar": True, **_sidecar_provenance()}
    )
    return causality.identity_payload(
        {
            "schema_version": 1,
            "kind": causality.KIND_QUALITY_ROW,
            "arm_code": "J1",
            "completed_updates": 1000,
            "clip_id": f"lockbox-{index:03d}",
            "clip_index": index,
            "source": source,
            "nfe": 2,
            "evaluation_split": "lockbox",
            "frontier_selection_identity_sha256": SELECTION_ID,
            "lockbox_registration_identity_sha256": LOCKBOX_ID,
            "study_identity_sha256": STUDY_ID,
            "arm_identity_sha256": ARM_ID,
            "stage_identity_sha256": STAGE_ID,
            "training_git_commit": "9" * 40,
            "evaluator_git_commit": (
                OLD_COMMIT if source == "autonomous" else NEW_COMMIT
            ),
            "inference_code_compatibility_sha256": COMPATIBILITY_ID,
            "videox_runtime_identity_sha256": VIDEOX_ID,
            "evaluation_world_size": 8,
            "evaluation_batch_size_per_rank": 2,
            "sampling_namespace": "lockbox",
            "sampling_id": frontier.SAMPLE_ID_OFFSETS["lockbox"] + index,
            "oracle_leakage": False,
            "deployable_evidence": True,
            "sampler_entrypoint": (
                "DualExplicitActionDiTModel.sample_future_deployable"
            ),
            "clean_future_or_auxiliary_passed_to_sampler": False,
            "online_teacher_call_count": 0,
            "auxiliary_history_latent_frames": 0,
            "actual_wan_call_count": 2,
            "effective_state_gate": 0.01,
            "effective_clock_gate": 0.02,
            "metrics": {
                "video_future_nmse": metric,
                "decoded_mse_unit_range": metric,
                "decoded_temporal_difference_mse_unit_range": metric,
            },
            "tensor_sha256": tensor_hashes,
            **sidecar,
        }
    )


def _evidence(source_metrics: dict[str, float]) -> tuple[dict, dict, dict]:
    autonomous_rows = [
        _row(index, "autonomous", source_metrics["autonomous"])
        for index in range(128)
    ]
    sidecar_rows = [
        _row(index, source, source_metrics[source])
        for index in range(128)
        for source in causality.SIDE_CAR_SOURCES
    ]
    base = {
        "rank_inputs": _rank_inputs(),
        "rank_provenance": _rank_provenance(),
    }
    autonomous = {
        **base,
        "path": "/evidence/autonomous.json",
        "sha256": AUTONOMOUS_INVENTORY_SHA,
        "identity_sha256": AUTONOMOUS_INVENTORY_ID,
        "payload": {"evaluator_git_commit": OLD_COMMIT},
        "rows": autonomous_rows,
    }
    sidecar = {
        **copy.deepcopy(base),
        "path": "/evidence/sidecar.json",
        "sha256": "a" * 64,
        "identity_sha256": "b" * 64,
        "payload": {"evaluator_git_commit": NEW_COMMIT},
        "rows": sidecar_rows,
    }
    continuation = {
        "path": "/study/frontier_continuation.json",
        "sha256": CONTINUATION_SHA,
        "lockbox_quality_array_job_id": "481700",
        "payload": {
            "selection": {
                "path": "/study/frontier_selection.json",
                "sha256": "0" * 64,
                "identity_sha256": SELECTION_ID,
            },
            "lockbox_jobs_created_only_after_confirmatory_eligibility": True,
        },
    }
    return autonomous, sidecar, continuation


def test_sidecar_grid_is_frozen_controls_only(monkeypatch):
    monkeypatch.setattr(causality, "validate_selection", lambda _selection: 2)
    assert quality.frontier_causality_sidecar_grid("J1", _selection()) == (
        ("off", 2),
        ("autonomous_shuffled", 2),
    )
    with pytest.raises(quality.QualityEvaluationError, match="only for J1"):
        quality.frontier_causality_sidecar_grid("VPM", _selection())


def test_continuation_binds_eligible_selection_and_exact_j1_task(tmp_path):
    selection_path = tmp_path / "frontier_selection.json"
    selection_bytes = json.dumps(_selection(), sort_keys=True).encode()
    selection_path.write_bytes(selection_bytes)
    continuation_path = tmp_path / "frontier_continuation.json"
    continuation_path.write_text(
        json.dumps(
            {
                "kind": causality.KIND_CONTINUATION,
                "schema_version": 1,
                "selection": {
                    "path": str(selection_path),
                    "sha256": hashlib.sha256(selection_bytes).hexdigest(),
                    "identity_sha256": SELECTION_ID,
                    "confirmatory_eligible": True,
                },
                "selection_gate_job_id": "481600",
                "controller_git_commit": NEW_COMMIT,
                "scientific_evaluator_git_commit": OLD_COMMIT,
                "lockbox_quality_array_job_id": "481700",
                "lockbox_quality_dependency": "afterok:481600",
                "confirmation_job_id": "481800",
                "confirmation_dependency": "afterok:481700",
                "timing_and_finalization_job_id": "481900",
                "timing_dependency": "afterok:481800",
                "lockbox_jobs_created_only_after_confirmatory_eligibility": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = causality.validate_continuation(
        continuation_path,
        selection_path=selection_path,
        selection=_selection(),
    )
    assert result["j1_array_task_dependency"] == "afterok:481700_1"

    payload = json.loads(continuation_path.read_text(encoding="utf-8"))
    payload["selection"]["confirmatory_eligible"] = False
    continuation_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(causality.CausalityError, match="eligible selection"):
        causality.validate_continuation(
            continuation_path,
            selection_path=selection_path,
            selection=_selection(),
        )


def test_both_paired_causal_gates_and_materiality_pass(monkeypatch):
    monkeypatch.setattr(causality, "validate_selection", lambda _selection: 2)
    autonomous, sidecar, continuation = _evidence(
        {"autonomous": 0.90, "off": 1.0, "autonomous_shuffled": 1.0}
    )

    result = causality.build_confirmation(
        selection=_selection(),
        continuation=continuation,
        autonomous=autonomous,
        sidecar=sidecar,
        sidecar_evaluator_commit=NEW_COMMIT,
        created_at_utc="2026-07-30T00:00:00+00:00",
    )

    assert result["both_preregistered_ci_gates_passed"] is True
    assert result["both_temporal_effects_at_least_three_percent"] is True
    assert result["within_j1_generated_state_causality_supported"] is True
    assert result["frontier_isolation"]["selection_decisions_from_sidecar"] == 0
    assert causality.identity_valid(result)


def test_cross_source_initial_noise_or_target_mismatch_fails(monkeypatch):
    monkeypatch.setattr(causality, "validate_selection", lambda _selection: 2)
    autonomous, sidecar, continuation = _evidence(
        {"autonomous": 0.90, "off": 1.0, "autonomous_shuffled": 1.0}
    )
    tampered = sidecar["rows"][0]
    unsigned = dict(tampered)
    unsigned.pop("identity_sha256")
    unsigned["tensor_sha256"] = dict(unsigned["tensor_sha256"])
    unsigned["tensor_sha256"]["video_initial_state_sha256"] = "0" * 64
    sidecar["rows"][0] = causality.identity_payload(unsigned)

    with pytest.raises(causality.CausalityError, match="pairing differs"):
        causality.build_confirmation(
            selection=_selection(),
            continuation=continuation,
            autonomous=autonomous,
            sidecar=sidecar,
            sidecar_evaluator_commit=NEW_COMMIT,
        )


def test_temporal_gate_fails_when_autonomous_is_worse(monkeypatch):
    monkeypatch.setattr(causality, "validate_selection", lambda _selection: 2)
    autonomous, sidecar, continuation = _evidence(
        {"autonomous": 1.10, "off": 1.0, "autonomous_shuffled": 1.0}
    )
    result = causality.build_confirmation(
        selection=_selection(),
        continuation=continuation,
        autonomous=autonomous,
        sidecar=sidecar,
        sidecar_evaluator_commit=NEW_COMMIT,
    )
    assert result["both_preregistered_ci_gates_passed"] is False
    assert result["within_j1_generated_state_causality_supported"] is False


def test_conditional_launcher_is_separate_and_recoverable():
    for path in (LAUNCHER, QUALITY_SBATCH, CONFIRM_SBATCH):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'CONTINUATION="$STUDY_ROOT/frontier_continuation.json"' in launcher
    assert 'MAIN_J1_DEPENDENCY="${VALUES[7]}"' in launcher
    assert '--dependency="$MAIN_J1_DEPENDENCY"' in launcher
    assert "QUALITY_RECEIPT=" in launcher
    assert "Recovered accepted causality quality job" in launcher
    assert "main_frontier_artifacts_modified" in launcher
    assert "frontier_lockbox_confirmation.json" not in launcher
    assert "frontier_final_report.json" not in launcher
