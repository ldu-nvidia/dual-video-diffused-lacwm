import hashlib
import json
from pathlib import Path

import pytest

from tools.summarize_stage_faithful_evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    EXPECTED_DATASET,
    EXPECTED_SOURCES,
    METRIC_DIRECTIONS,
    NFE_STEPS,
    SOURCE_COMPARISONS,
    StageEvaluationSummaryError,
    _relative_effect,
    _summary,
    build_summary,
    main,
    summarize,
)


ARM = "stage_faithful"
NEW_SCOPE = "/evidence/new/visualization/iter_199"
LEGACY_SCOPE = "/evidence/legacy/visualization/iter_199"


def _artifact(scope: str, rank: int, prefix: str) -> dict:
    path = (
        f"{scope}/{EXPECTED_DATASET}/"
        f"latent_trajectory_rank_{rank}.safetensors"
    )
    return {
        "dataset": EXPECTED_DATASET,
        "global_rank": rank,
        "path": path,
        "sha256": hashlib.sha256(f"{prefix}{rank}".encode()).hexdigest(),
    }


def _metric_value(source: str, nfe: int, metric: str, rank: int) -> float:
    rank_scale = 1.0 + 0.05 * rank
    source_scale = {
        "autonomous": 1.0,
        "autonomous_shuffled": 1.0,
        "autonomous_legacy": 1.14,
        "off": 1.08,
    }[source]
    if metric == "decoded_psnr_db":
        value = 18.0 / source_scale + 0.1 * nfe + 0.02 * rank
    else:
        base = {
            "video_future_nmse": 0.7,
            "tf_future_nmse": 0.4,
            "decoded_mse_unit_range": 0.05,
            "decoded_temporal_difference_mse_unit_range": 0.03,
        }[metric]
        value = base * source_scale * rank_scale / (1.0 + 0.04 * nfe)

    # The preregistered autonomous-vs-shuffled gate passes with strict,
    # bootstrap-supported improvements on the three required cells.
    if source == "autonomous":
        if nfe == 4 and metric == "video_future_nmse":
            value *= 0.95
        if nfe == 4 and metric == "decoded_temporal_difference_mse_unit_range":
            value *= 0.90
        if nfe == 8 and metric == "decoded_temporal_difference_mse_unit_range":
            value *= 0.97
    return value


def _analysis() -> dict:
    values = {
        source: {
            str(nfe): {
                metric: [
                    _metric_value(source, nfe, metric, rank)
                    for rank in range(8)
                ]
                for metric in METRIC_DIRECTIONS
            }
            for nfe in NFE_STEPS
        }
        for source in EXPECTED_SOURCES
    }
    per_units = []
    for rank in range(8):
        source_metrics = {}
        for source in EXPECTED_SOURCES:
            source_metrics[source] = {
                "oracle_leakage": False,
                "metrics": {
                    str(nfe): {
                        metric: values[source][str(nfe)][metric][rank]
                        for metric in METRIC_DIRECTIONS
                    }
                    for nfe in NFE_STEPS
                },
            }
        per_units.append(
            {
                "dataset": EXPECTED_DATASET,
                "global_rank": rank,
                "arms": {
                    ARM: {
                        "condition_source_metrics": source_metrics,
                    }
                },
            }
        )

    sources = {}
    for source in EXPECTED_SOURCES:
        sources[source] = {
            "oracle_leakage": False,
            "nfe": {
                str(nfe): {
                    metric: _summary(
                        values[source][str(nfe)][metric],
                        label=(
                            f"within-arm:{ARM}:source:{source}:nfe:{nfe}:"
                            f"metric:{metric}"
                        ),
                    )
                    for metric in METRIC_DIRECTIONS
                }
                for nfe in NFE_STEPS
            },
        }

    comparisons = {}
    for name in SOURCE_COMPARISONS:
        right = name.removeprefix("autonomous_minus_")
        comparisons[name] = {
            "oracle_leakage": False,
            "deployable_evidence": True,
            "relative_nfe": {
                str(nfe): {
                    metric: {
                        **_relative_effect(
                            values["autonomous"][str(nfe)][metric],
                            values[right][str(nfe)][metric],
                            direction=direction,
                            label=(
                                f"within-arm-relative:{ARM}:{name}:"
                                f"nfe:{nfe}:metric:{metric}"
                            ),
                        ),
                        "defined": True,
                    }
                    for metric, direction in METRIC_DIRECTIONS.items()
                }
                for nfe in NFE_STEPS
            },
        }

    return {
        "schema_version": 1,
        "kind": "dual_video_diffusion_matched_nfe_analysis",
        "sigma_convention": "1=noise,0=clean",
        "nfe_steps": list(NFE_STEPS),
        "baseline_arm": ARM,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
        },
        "provenance": {
            "paired_unit_count": 8,
            "paired_units": [
                {"dataset": EXPECTED_DATASET, "global_rank": rank}
                for rank in range(8)
            ],
            "arms": {
                ARM: {
                    "root": NEW_SCOPE,
                    "artifact_count": 8,
                    "evaluation_condition_sources": list(EXPECTED_SOURCES),
                    "artifacts": [
                        _artifact(NEW_SCOPE, rank, "new-hash-")
                        for rank in range(8)
                    ],
                }
            },
        },
        "per_paired_unit": per_units,
        "aggregate": {
            "within_arm_condition_sources": {
                ARM: {
                    "declared_sources": list(EXPECTED_SOURCES),
                    "sources": sources,
                }
            },
            "within_arm_source_deltas": {ARM: comparisons},
        },
    }


def _baseline() -> dict:
    per_units = []
    references = {
        str(nfe): {
            metric: [
                (
                    (0.35 + 0.02 * rank) / (1.0 + 0.02 * nfe)
                    if metric == "video_future_nmse"
                    else (0.025 + 0.002 * rank) / (1.0 + 0.02 * nfe)
                    if metric == "decoded_mse_unit_range"
                    else (0.014 + 0.001 * rank) / (1.0 + 0.02 * nfe)
                    if metric
                    == "decoded_temporal_difference_mse_unit_range"
                    else 19.0 + 0.1 * nfe + 0.05 * rank
                )
                for rank in range(8)
            ]
            for metric in (
                "video_future_nmse",
                "decoded_mse_unit_range",
                "decoded_psnr_db",
                "decoded_temporal_difference_mse_unit_range",
            )
        }
        for nfe in NFE_STEPS
    }
    for rank in range(8):
        per_units.append(
            {
                "dataset": EXPECTED_DATASET,
                "global_rank": rank,
                "nfe": {
                    str(nfe): {
                        metric: {
                            "candidate": reference + 0.1,
                            "reference": reference,
                            "delta": 0.1,
                        }
                        for metric, reference in (
                            (metric, references[str(nfe)][metric][rank])
                            for metric in references[str(nfe)]
                        )
                    }
                    for nfe in NFE_STEPS
                },
            }
        )
    identity_per_pair = [
        {
            "dataset": EXPECTED_DATASET,
            "global_rank": rank,
            "candidate_evaluation_nfe_steps": list(NFE_STEPS),
            "reference_evaluation_nfe_steps": [1, 2, 4, 8],
            "video_identity_fields_equal": {"clean": True, "noise": True},
            "tf_identity_fields_equal": {"clean": True, "noise": True},
        }
        for rank in range(8)
    ]
    return {
        "schema_version": 1,
        "kind": "strict_cascade_cross_screen_efficiency_audit",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
        },
        "scope": {
            "common_total_nfe_steps": list(NFE_STEPS),
            "nfe_is_total_model_calls": True,
        },
        "arms": {
            "cascade_matched_s010": {
                "root": LEGACY_SCOPE,
                "terminal": {
                    "artifacts": [
                        _artifact(LEGACY_SCOPE, rank, "legacy-hash-")
                        for rank in range(8)
                    ]
                },
            },
            "video_only_s000": {
                "root": "/evidence/video-only/visualization/iter_199",
                "terminal": {
                    "artifact_count": 8,
                    "iteration": 199,
                    "artifacts": [
                        _artifact(
                            "/evidence/video-only/visualization/iter_199",
                            rank,
                            "video-only-hash-",
                        )
                        for rank in range(8)
                    ],
                    "outcome": {"completed": True, "exit_status": 0},
                    "training_completion": {"completed_updates": 200},
                },
            },
        },
        "comparisons": {
            "video_only_s000": {
                "computed": True,
                "candidate_condition_source": "autonomous",
                "reference_condition_source": "autonomous",
                "identity_audit": {
                    "paired_unit_count": 8,
                    "common_total_nfe_steps": list(NFE_STEPS),
                    "video_relevant_identity_exact_for_every_pair": True,
                    "tf_identity_exact_for_every_pair": True,
                    "paired_units": [
                        {"dataset": EXPECTED_DATASET, "global_rank": rank}
                        for rank in range(8)
                    ],
                    "per_pair": identity_per_pair,
                },
                "per_paired_unit": per_units,
                "aggregate": {
                    str(nfe): {
                        metric: {
                            "reference": {
                                "n": 8,
                                "mean": sum(metric_values) / 8,
                            }
                        }
                        for metric, metric_values in references[str(nfe)].items()
                    }
                    for nfe in NFE_STEPS
                },
            }
        },
    }


def _audit() -> dict:
    def rank_record(scope: str, rank: int, prefix: str) -> dict:
        artifact = _artifact(scope, rank, prefix)
        return {
            "dataset": artifact["dataset"],
            "rank": artifact["global_rank"],
            "artifact": {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
            },
        }

    return {
        "schema_version": 1,
        "kind": "stage_faithful_cascade_bitwise_artifact_audit",
        "sigma_convention": "1=noise,0=clean",
        "read_only_inputs": True,
        "overall_pass": True,
        "identity_sha256": "a" * 64,
        "contracts": {
            "pass": True,
            "world_size": {
                "pass": True,
                "observed_new": 8,
                "observed_legacy": 8,
                "expected": 8,
            },
            "paired_ranks": {
                "pass": True,
                "new": list(range(8)),
                "legacy": list(range(8)),
                "expected": list(range(8)),
            },
            "artifact_iteration": {"pass": True},
            "evaluation_noise_seed_identity": {"pass": True},
            "forbidden_training_outputs": {"pass": True, "observed": []},
            "sidecar_hashes_and_sigma_convention": {"pass": True},
            "new_source_codes": {"pass": True, "expected": [0, 4, 5, 1]},
            "legacy_source_codes": {"pass": True, "expected": [0, 1, 2, 3]},
            "nfe_steps": {"pass": True, "expected": list(NFE_STEPS)},
            "new_stage_faithful_flag": {"pass": True, "expected": 1},
        },
        "inputs": {
            "new": {
                "root": "/evidence/new",
                "artifact_scope": NEW_SCOPE,
                "artifact_set_sha256": "b" * 64,
                "ranks": [
                    rank_record(NEW_SCOPE, rank, "new-hash-")
                    for rank in range(8)
                ],
            },
            "legacy": {
                "root": "/evidence/legacy",
                "artifact_scope": LEGACY_SCOPE,
                "artifact_set_sha256": "c" * 64,
                "ranks": [
                    rank_record(LEGACY_SCOPE, rank, "legacy-hash-")
                    for rank in range(8)
                ],
            },
        },
        "rank_audits": [
            {
                "rank": rank,
                "pass": True,
                "contracts": {"pass": True},
                "new_legacy_input_identity": {"pass": True},
                "legacy_reproduction": {"pass": True},
                "stage_tf_equivalence": {"pass": True},
            }
            for rank in range(8)
        ],
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_build_summary_passes_exact_gate_and_cross_reference():
    analysis = _analysis()
    payload = build_summary(
        analysis,
        _baseline(),
        _audit(),
        input_sha256={
            "analysis": "1" * 64,
            "baseline_reference": "2" * 64,
            "audit": "3" * 64,
        },
    )

    assert payload["validation"]["overall_pass"] is True
    assert payload["validation"]["paired_unit_count"] == 8
    gate = payload["within_checkpoint_stage_faithful"][
        "preregistered_gate"
    ]
    assert gate["point_gate_pass"] is True
    assert gate["sign_supported_gate_pass"] is True
    assert all(gate["point_criteria"].values())
    assert all(gate["sign_supported_criteria"].values())

    expected_stage = sum(
        _metric_value("autonomous", 4, "video_future_nmse", rank)
        for rank in range(8)
    ) / 8
    observed = payload["video_only_reference"]["equal_total_nfe"]["nfe"]["4"][
        "video_future_nmse"
    ]
    assert observed["left_mean"] == pytest.approx(expected_stage)
    assert observed["n"] == 8
    assert (
        payload["video_only_reference"]["stage_nfe8_vs_video_only_nfe2"][
            "nfe"
        ]
        == {"stage": 8, "video_only": 2}
    )


def test_cli_output_is_exclusive_and_byte_deterministic(tmp_path, capsys):
    analysis_path = tmp_path / "analysis.json"
    baseline_path = tmp_path / "baseline.json"
    audit_path = tmp_path / "audit.json"
    _write(analysis_path, _analysis())
    _write(baseline_path, _baseline())
    _write(audit_path, _audit())
    first = tmp_path / "summary-first.json"
    second = tmp_path / "summary-second.json"

    first_payload, first_sha = summarize(
        analysis=analysis_path,
        baseline_reference=baseline_path,
        audit=audit_path,
        output=first,
    )
    second_payload, second_sha = summarize(
        analysis=analysis_path,
        baseline_reference=baseline_path,
        audit=audit_path,
        output=second,
    )
    assert first_payload == second_payload
    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.stat().st_mode & 0o777 == 0o600

    assert (
        main(
            [
                "--analysis",
                str(analysis_path),
                "--baseline-reference",
                str(baseline_path),
                "--audit",
                str(audit_path),
                "--output",
                str(first),
            ]
        )
        == 2
    )
    assert "output already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutation",
    ("audit_failure", "pair_reorder", "baseline_mean", "artifact_hash"),
)
def test_fail_closed_on_audit_pair_reference_and_artifact_changes(mutation):
    analysis = _analysis()
    baseline = _baseline()
    audit = _audit()
    if mutation == "audit_failure":
        audit["overall_pass"] = False
    elif mutation == "pair_reorder":
        analysis["provenance"]["paired_units"][0], analysis["provenance"][
            "paired_units"
        ][1] = (
            analysis["provenance"]["paired_units"][1],
            analysis["provenance"]["paired_units"][0],
        )
    elif mutation == "baseline_mean":
        baseline["comparisons"]["video_only_s000"]["aggregate"]["4"][
            "video_future_nmse"
        ]["reference"]["mean"] += 0.01
    else:
        audit["inputs"]["new"]["ranks"][3]["artifact"]["sha256"] = "changed"

    with pytest.raises(StageEvaluationSummaryError):
        build_summary(
            analysis,
            baseline,
            audit,
            input_sha256={
                "analysis": "1" * 64,
                "baseline_reference": "2" * 64,
                "audit": "3" * 64,
            },
        )


def test_rejects_tampered_analyzer_relative_effect():
    analysis = _analysis()
    analysis["aggregate"]["within_arm_source_deltas"][ARM][
        "autonomous_minus_autonomous_shuffled"
    ]["relative_nfe"]["4"]["video_future_nmse"]["relative_effect"] += 0.001

    with pytest.raises(
        StageEvaluationSummaryError, match="paired recomputation"
    ):
        build_summary(
            analysis,
            _baseline(),
            _audit(),
            input_sha256={
                "analysis": "1" * 64,
                "baseline_reference": "2" * 64,
                "audit": "3" * 64,
            },
        )
