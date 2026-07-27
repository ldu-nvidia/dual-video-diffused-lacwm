import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_privileged_video_artifacts import EXPECTED_PARENTS
from tools.summarize_privileged_video_evaluation import (
    ARM_NAMES,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    LOWER_IS_BETTER_METRICS,
    NFE_STEPS,
    PrivilegedEvaluationSummaryError,
    build_summary,
    main,
)


DATASET = "MultiDatasetABC_0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _artifact(arm: str, rank: int) -> dict:
    return {
        "dataset": DATASET,
        "global_rank": rank,
        "path": (
            f"/evidence/{arm}/visualization/iter_199/{DATASET}/"
            f"latent_trajectory_rank_{rank}.safetensors"
        ),
        "sha256": _sha256_text(f"{arm}:{rank}"),
    }


def _signed_audit() -> dict:
    expected_ranks = list(range(8))
    contracts = {
        "pass": True,
        "arm_names": {
            "observed": list(ARM_NAMES),
            "expected": list(ARM_NAMES),
            "pass": True,
        },
        "world_size": {
            "observed": {arm: 8 for arm in ARM_NAMES},
            "expected": 8,
            "pass": True,
        },
        "paired_ranks": {
            "observed": {arm: expected_ranks for arm in ARM_NAMES},
            "expected": expected_ranks,
            "pass": True,
        },
        "artifact_iteration": {
            "observed": 199,
            "expected": 199,
            "pass": True,
        },
        "source_codes": {
            "expected": [0, 1],
            "names": ["autonomous", "off"],
            "pass": True,
        },
        "nfe_steps": {"expected": list(NFE_STEPS), "pass": True},
        "evaluation_noise_seed": {"expected": 20_260_726, "pass": True},
        "tf_content_disabled": {
            "condition_on_tf": 0,
            "condition_mode_code": 0,
            "pass": True,
        },
        "tf_clock_disabled": {
            "evaluation_disable_tf_clock": 1,
            "evaluation_tf_clock_enabled": 0,
            "pass": True,
        },
        "noncascade_all_video_schedule": {
            "cascade_stage_faithful_inference": 0,
            "evaluation_all_video_schedule": 1,
            "pass": True,
        },
        "raw_causal_inputs_present": {
            "raw_actions_present": 1,
            "raw_morphology_index_present": 1,
            "pass": True,
        },
        "cross_arm_input_identity": {"pass": True},
        "autonomous_off_runtime_noop": {"pass": True},
        "exact_parent_provenance": {
            "expected": EXPECTED_PARENTS,
            "pass": True,
        },
        "raw_action_morphology_input_identity": {
            "tensors": ["raw_actions", "raw_morphology_index"],
            "meaning": (
                "exact raw causal inputs supplied to each checkpoint's "
                "independently learned action encoder"
            ),
            "pass": True,
        },
        "learned_action_control_diagnostic": {
            "tensor": "z_control",
            "cross_arm_equality_required": False,
            "meaning": (
                "checkpoint-specific learned action control retained for "
                "mechanism analysis, not treated as a paired raw input"
            ),
            "pass": True,
        },
        "forbidden_training_outputs": {
            "observed": {arm: [] for arm in ARM_NAMES},
            "expected": {arm: [] for arm in ARM_NAMES},
            "pass": True,
        },
        "sidecar_hashes_schema_and_sigma_convention": {"pass": True},
    }
    inputs = {}
    for arm in ARM_NAMES:
        inputs[arm] = {
            "root": f"/evidence/{arm}",
            "artifact_scope": f"/evidence/{arm}/visualization/iter_199",
            "artifact_set_sha256": _sha256_text(f"set:{arm}"),
            "evaluation_provenance": {
                "path": f"/evidence/{arm}/privileged_video_evaluation_provenance.json",
                "sha256": _sha256_text(f"provenance:{arm}"),
                "parent": EXPECTED_PARENTS[arm],
                "completed_updates": 200,
                "total_observations": 1600,
                "runtime_intervention": {
                    "schedule_mode": "aligned",
                    "tf_content_disabled": True,
                    "tf_clock_disabled": True,
                    "all_model_calls_advance_video": True,
                },
                "pass": True,
            },
            "ranks": [
                {
                    "rank": rank,
                    "dataset": DATASET,
                    "artifact": {
                        "path": _artifact(arm, rank)["path"],
                        "sha256": _artifact(arm, rank)["sha256"],
                    },
                }
                for rank in range(8)
            ],
        }
    payload = {
        "schema_version": 1,
        "kind": "privileged_tf_video_bitwise_artifact_audit",
        "created_at_utc": "2026-07-26T00:00:00+00:00",
        "sigma_convention": "1=noise,0=clean",
        "read_only_inputs": True,
        "overall_pass": True,
        "contracts": contracts,
        "inputs": inputs,
        "rank_audits": [
            {
                "rank": rank,
                "dataset": DATASET,
                "pass": True,
                "contracts": {"pass": True},
                "cross_arm_input_identity": {"pass": True},
                "autonomous_off_runtime_noop": {"pass": True},
            }
            for rank in range(8)
        ],
    }
    payload["identity_sha256"] = _sha256_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return payload


def _resign(audit: dict) -> None:
    audit.pop("identity_sha256", None)
    audit["identity_sha256"] = _sha256_text(
        json.dumps(
            audit,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _base_metric(metric: str, rank: int, nfe: int) -> float:
    scale = 1.0 + 0.025 * rank
    base = {
        "video_future_nmse": 0.8,
        "tf_future_nmse": 0.9,
        "decoded_mse_unit_range": 0.08,
        "decoded_temporal_difference_mse_unit_range": 0.04,
    }[metric]
    return base * scale / (1.0 + 0.01 * nfe)


def _metric(
    arm: str,
    metric: str,
    rank: int,
    nfe: int,
) -> float:
    value = _base_metric(metric, rank, nfe)
    if arm == "trained_shuffled":
        return value
    if arm == "trained_off":
        return value * 1.02
    if metric == "decoded_temporal_difference_mse_unit_range":
        return value * (0.88 if nfe in (2, 4, 8) else 0.98)
    if metric in ("video_future_nmse", "decoded_mse_unit_range"):
        return value * (0.98 if nfe == 4 else 1.0)
    return value * 0.97


def _analysis() -> dict:
    arms_provenance = {}
    for arm in ARM_NAMES:
        arms_provenance[arm] = {
            "root": f"/evidence/{arm}/visualization/iter_199",
            "artifact_count": 8,
            "intervention": {
                "condition_on_tf": False,
                "condition_mode_code": 0,
                "condition_mode": "off",
            },
            "evaluation_condition_sources": ["autonomous", "off"],
            "artifacts": [_artifact(arm, rank) for rank in range(8)],
        }
    per_units = []
    for rank in range(8):
        unit_arms = {}
        for arm in ARM_NAMES:
            nfe_metrics = {}
            for nfe in NFE_STEPS:
                metrics = {
                    metric: _metric(arm, metric, rank, nfe)
                    for metric in LOWER_IS_BETTER_METRICS
                }
                metrics["decoded_psnr_db"] = 20.0 + 0.1 * rank + nfe
                nfe_metrics[str(nfe)] = metrics
            source = {
                "oracle_leakage": False,
                "metrics": copy.deepcopy(nfe_metrics),
            }
            unit_arms[arm] = {
                "artifact_path": _artifact(arm, rank)["path"],
                "artifact_sha256": _artifact(arm, rank)["sha256"],
                "intervention": {
                    "condition_on_tf": False,
                    "condition_mode_code": 0,
                    "condition_mode": "off",
                },
                "metrics": copy.deepcopy(nfe_metrics),
                "condition_source_metrics": {
                    "autonomous": copy.deepcopy(source),
                    "off": copy.deepcopy(source),
                },
            }
        per_units.append(
            {
                "dataset": DATASET,
                "global_rank": rank,
                "arms": unit_arms,
            }
        )
    return {
        "schema_version": 1,
        "kind": "dual_video_diffusion_matched_nfe_analysis",
        "generated_at_utc": "2026-07-26T00:00:00+00:00",
        "sigma_convention": "1=noise,0=clean",
        "nfe_steps": list(NFE_STEPS),
        "baseline_arm": "trained_off",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
        },
        "provenance": {
            "iteration": 199,
            "paired_unit_count": 8,
            "paired_units": [
                {"dataset": DATASET, "global_rank": rank}
                for rank in range(8)
            ],
            "arms": arms_provenance,
        },
        "per_paired_unit": per_units,
        "aggregate": {},
    }


def _build(analysis=None, audit=None):
    return build_summary(
        _analysis() if analysis is None else analysis,
        _signed_audit() if audit is None else audit,
        input_sha256={"analysis": "a" * 64, "audit": "b" * 64},
    )


def test_build_summary_passes_artifact_and_scientific_gates():
    payload = _build()
    assert payload["decision"] == {
        "artifact_audit_pass": True,
        "scientific_gate_pass": True,
        "promote_to_stage_b": True,
        "audit_pass_does_not_imply_scientific_gate_pass": True,
    }
    comparisons = payload["scientific_evaluation"]["comparisons"]
    assert set(comparisons) == {
        "trained_matched_minus_trained_shuffled",
        "trained_matched_minus_trained_off",
    }
    for comparison in comparisons.values():
        assert set(comparison["nfe"]) == {"1", "2", "4", "8"}
        assert set(comparison["nfe"]["4"]) == set(LOWER_IS_BETTER_METRICS)
    assert (
        payload["artifact_validation"][
            "raw_action_morphology_input_identity_pass"
        ]
        is True
    )
    assert (
        payload["artifact_validation"][
            "learned_z_control_cross_arm_equality_required"
        ]
        is False
    )


def test_scientific_gate_can_fail_while_artifact_audit_passes():
    analysis = _analysis()
    for unit in analysis["per_paired_unit"]:
        rank = unit["global_rank"]
        bad = 1.05 * _base_metric(
            "decoded_temporal_difference_mse_unit_range",
            rank,
            8,
        )
        for source in ("autonomous", "off"):
            unit["arms"]["trained_matched"]["condition_source_metrics"][
                source
            ]["metrics"]["8"][
                "decoded_temporal_difference_mse_unit_range"
            ] = bad
    payload = _build(analysis=analysis)
    assert payload["decision"]["artifact_audit_pass"] is True
    assert payload["decision"]["scientific_gate_pass"] is False
    assert payload["decision"]["promote_to_stage_b"] is False


def test_output_is_deterministic_for_identical_evidence():
    assert _build() == _build()


def test_rejects_mutated_audit_signature():
    audit = _signed_audit()
    audit["contracts"]["tf_clock_disabled"][
        "evaluation_disable_tf_clock"
    ] = 0
    with pytest.raises(
        PrivilegedEvaluationSummaryError,
        match="identity signature",
    ):
        _build(audit=audit)


def test_rejects_swapped_parent_even_when_resigned():
    audit = _signed_audit()
    audit["inputs"]["trained_off"]["evaluation_provenance"][
        "parent"
    ] = EXPECTED_PARENTS["trained_matched"]
    _resign(audit)
    with pytest.raises(
        PrivilegedEvaluationSummaryError,
        match="provenance evidence is not exact",
    ):
        _build(audit=audit)


def test_rejects_analysis_artifact_hash_mismatch():
    analysis = _analysis()
    analysis["provenance"]["arms"]["trained_matched"]["artifacts"][3][
        "sha256"
    ] = "f" * 64
    with pytest.raises(
        PrivilegedEvaluationSummaryError,
        match="analysis/audit provenance mismatch",
    ):
        _build(analysis=analysis)


def test_rejects_analysis_pair_order_mutation():
    analysis = _analysis()
    analysis["per_paired_unit"][1], analysis["per_paired_unit"][2] = (
        analysis["per_paired_unit"][2],
        analysis["per_paired_unit"][1],
    )
    with pytest.raises(
        PrivilegedEvaluationSummaryError,
        match="per-unit identity/order",
    ):
        _build(analysis=analysis)


def test_rejects_autonomous_off_metric_disagreement():
    analysis = _analysis()
    analysis["per_paired_unit"][0]["arms"]["trained_matched"][
        "condition_source_metrics"
    ]["off"]["metrics"]["4"]["decoded_mse_unit_range"] += 0.001
    with pytest.raises(
        PrivilegedEvaluationSummaryError,
        match="autonomous/off metric mismatch",
    ):
        _build(analysis=analysis)


def test_cli_exclusively_writes_summary(tmp_path, capsys):
    analysis_path = tmp_path / "analysis.json"
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "summary.json"
    analysis_path.write_text(json.dumps(_analysis()), encoding="utf-8")
    audit_path.write_text(json.dumps(_signed_audit()), encoding="utf-8")
    args = [
        "--analysis",
        str(analysis_path),
        "--audit",
        str(audit_path),
        "--output",
        str(output_path),
    ]
    assert main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["artifact_audit_pass"] is True
    assert stdout["scientific_gate_pass"] is True
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert main(args) == 2
    assert "output already exists" in capsys.readouterr().err
