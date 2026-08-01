from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tools import analyze_video_latent_forcing_poc as gate


def _constant_values(size: int) -> dict[str, dict[str, np.ndarray]]:
    def values(lpips: float, temporal: float, nmse: float, cosine: float):
        return {
            "lpips_alex_frame": np.full(size, lpips, dtype=np.float64),
            "temporal_difference_mse": np.full(size, temporal, dtype=np.float64),
            "auxiliary_nmse": np.full(size, nmse, dtype=np.float64),
            "auxiliary_cosine": np.full(size, cosine, dtype=np.float64),
        }

    return {
        "autonomous": values(0.4, 0.4, 0.2, 0.9),
        "off": values(0.8, 0.8, 0.2, 0.9),
        "shuffled": values(0.8, 0.8, 0.2, 0.9),
        "oracle_clean": values(0.2, 0.2, 0.2, 0.9),
        "context_shuffled": values(0.4, 0.4, 0.4, 0.7),
    }


def test_phase1_gate_cell_passes_only_complete_representation_screen(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    decision = gate._gate_cell(4, _constant_values(8))
    assert decision["passed"] is True
    assert decision["nfe_pair"] == [4, 0]
    assert {check["id"] for check in decision["checks"]} == {
        "auxiliary_nmse",
        "auxiliary_cosine",
        "lpips_alex_frame_vs_off",
        "lpips_alex_frame_vs_shuffled",
        "lpips_alex_frame_retained_utility",
        "temporal_difference_mse_vs_off",
        "temporal_difference_mse_vs_shuffled",
        "temporal_difference_mse_retained_utility",
        "context_causality_nmse",
        "context_causality_cosine",
    }


def test_nonpositive_oracle_utility_denominator_is_valid_json_failure(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    values = _constant_values(8)
    values["oracle_clean"]["lpips_alex_frame"][:] = 0.9
    decision = gate._gate_cell(4, values)
    retained = next(
        check
        for check in decision["checks"]
        if check["id"] == "lpips_alex_frame_retained_utility"
    )
    assert retained["passed"] is False
    assert retained["estimate"] is None
    assert decision["passed"] is False
    assert "NaN" not in json.dumps(decision, allow_nan=False)


def test_paired_bootstrap_is_label_deterministic_and_label_separated(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    first = np.arange(1, 9, dtype=np.float64)
    second = first + np.arange(2, 10, dtype=np.float64)
    one = gate.paired_bootstrap(
        first,
        second,
        label="registered-statistic",
        statistic=gate.relative_improvement,
    )
    two = gate.paired_bootstrap(
        first,
        second,
        label="registered-statistic",
        statistic=gate.relative_improvement,
    )
    other = gate.paired_bootstrap(
        first,
        second,
        label="different-statistic",
        statistic=gate.relative_improvement,
    )
    assert one == two
    assert one["seed"] != other["seed"]


def test_bootstrap_rejects_nonfinite_or_unpaired_inputs(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 8)
    with pytest.raises(gate.GateError, match="exactly 8 aligned clips"):
        gate.paired_bootstrap(
            np.ones(7),
            np.ones(7),
            label="bad-size",
            statistic=gate.relative_improvement,
        )
    with pytest.raises(gate.GateError, match="nonfinite point statistic"):
        gate.paired_bootstrap(
            np.ones(8),
            np.zeros(8),
            label="zero-reference",
            statistic=gate.relative_improvement,
        )


def test_output_validation_rejects_evidence_and_git_roots(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(gate.GateError, match="outside both evidence and Git roots"):
        gate.validate_output_path(evidence / "gate.json", evidence)
    with pytest.raises(gate.GateError, match="under /lustre"):
        gate.validate_output_path(tmp_path / "gate.json", evidence)


def _phase2_cell(size: int, frechet: float, paired_value: float) -> dict:
    return {
        "means": {
            gate.DISTRIBUTION_PRIMARY_METRIC: frechet,
            **{metric: paired_value for metric in gate.PAIRED_PRIMARY_METRICS},
        },
        "paired": {
            metric: np.full(size, paired_value, dtype=np.float64)
            for metric in gate.PAIRED_PRIMARY_METRICS
        },
        "real_feature_sha256": "a" * 64,
        "generated_feature_sha256": "b" * 64,
    }


def _phase2_decision_inputs(size: int) -> tuple[dict, dict]:
    cells = {
        ("B0", 20_000, "off"): _phase2_cell(size, 10.0, 1.0),
        ("A1", 20_000, "off"): _phase2_cell(size, 9.0, 0.9),
        ("L1", 20_000, "off"): _phase2_cell(size, 9.0, 0.9),
        ("L1", 20_000, "shuffled"): _phase2_cell(size, 9.0, 0.9),
    }
    for update in gate.TRAINING_EFFICIENCY_UPDATES:
        cells[("L1", update, "autonomous")] = _phase2_cell(
            size,
            9.0 if update == 16_000 else 11.0,
            0.9 if update == 16_000 else 1.1,
        )
    cells[("L1", 20_000, "autonomous")] = _phase2_cell(size, 8.5, 0.8)
    walls = {("B0", 20_000): 200.0}
    walls.update(
        {("L1", update): float(index + 1) * 20 for index, update in enumerate(
            gate.TRAINING_EFFICIENCY_UPDATES
        )}
    )
    return cells, walls


def test_phase2_decision_positive_and_exact_earliest_reach(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    cells, walls = _phase2_decision_inputs(8)
    decision = gate.phase2_gate_decision(cells, walls)
    assert decision["passed"] is True
    assert (
        decision["criteria"]["training_efficiency"]["selected_reach"]["update"]
        == 16_000
    )
    assert "same_update_sample_nfe_budget" in decision["criteria"]
    assert "same_budget_quality" not in decision["criteria"]
    assert decision["criteria"]["attribution"]["passed"] is True


def test_phase2_decision_fails_nonpositive_denominator_without_nan(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 32)
    cells, walls = _phase2_decision_inputs(8)
    cells[("B0", 20_000, "off")]["means"]["r3d18_frechet"] = 0.0
    decision = gate.phase2_gate_decision(cells, walls)
    criterion = decision["criteria"]["same_update_sample_nfe_budget"]
    assert criterion["passed"] is False
    assert criterion["checks"][0]["relative_improvement"] is None
    assert "NaN" not in json.dumps(decision, allow_nan=False)


def test_phase2_decision_requires_attribution_and_lower_wall(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 32)
    cells, walls = _phase2_decision_inputs(8)
    cells[("L1", 20_000, "off")] = copy.deepcopy(
        cells[("L1", 20_000, "autonomous")]
    )
    decision = gate.phase2_gate_decision(cells, walls)
    assert decision["criteria"]["attribution"]["autonomous_vs_off"]["passed"] is False
    walls[("L1", 16_000)] = walls[("B0", 20_000)]
    decision = gate.phase2_gate_decision(cells, walls)
    assert decision["criteria"]["training_efficiency"]["passed"] is False


def _complete_walls() -> dict[tuple[str, int], float]:
    return {
        (arm, update): float(index + 1)
        for arm in gate.DUAL_ARMS
        for index, update in enumerate(gate.DUAL_CHECKPOINT_UPDATES)
    }


def test_phase2_checkpoint_walls_fail_closed_on_missing_or_nonmonotone():
    walls = _complete_walls()
    gate._validate_phase2_checkpoint_walls(walls)
    missing = dict(walls)
    missing.pop(("A1", 1_000))
    with pytest.raises(gate.GateError, match="not positive and finite"):
        gate._validate_phase2_checkpoint_walls(missing)
    nonmonotone = dict(walls)
    nonmonotone[("L1", 1_000)] = nonmonotone[("L1", 500)]
    with pytest.raises(gate.GateError, match="strictly increasing"):
        gate._validate_phase2_checkpoint_walls(nonmonotone)


def _complete_gate_cells(size: int = 2) -> dict:
    cells = {
        (arm, update, gate.PRIMARY_CONTROL[arm]): _phase2_cell(size, 10.0, 1.0)
        for arm in gate.DUAL_ARMS
        for update in gate.DUAL_CHECKPOINT_UPDATES
    }
    cells[("L1", 20_000, "off")] = _phase2_cell(size, 10.0, 1.0)
    cells[("L1", 20_000, "shuffled")] = _phase2_cell(size, 10.0, 1.0)
    return cells


def test_phase2_gate_cells_require_complete_inventory_and_shared_real_features():
    cells = _complete_gate_cells()
    assert gate._validate_phase2_gate_cells(cells) == "a" * 64
    missing = dict(cells)
    missing.pop(("B0", 500, "off"))
    with pytest.raises(gate.GateError, match="inventory"):
        gate._validate_phase2_gate_cells(missing)
    mismatched = copy.deepcopy(cells)
    mismatched[("A1", 500, "off")]["real_feature_sha256"] = "c" * 64
    with pytest.raises(gate.GateError, match="bit-identical R3D target"):
        gate._validate_phase2_gate_cells(mismatched)


def test_phase2_matrix_index_rejects_missing_and_duplicate(monkeypatch):
    with pytest.raises(gate.GateError, match="exactly 21"):
        gate._index_phase2_audits([str(index) for index in range(20)])
    monkeypatch.setattr(
        gate,
        "_audit_phase2_evaluation",
        lambda root: {"arm": "B0", "update": 500},
    )
    with pytest.raises(gate.GateError, match="duplicate"):
        gate._index_phase2_audits([str(index) for index in range(21)])


def test_phase2_cli_collects_repeated_evaluation_roots():
    args = gate.build_parser().parse_args(
        [
            "phase2",
            "--evaluation-root",
            "/mnt/data1/eval-b0",
            "--evaluation-root",
            "/mnt/data1/eval-l1",
            "--output",
            "/mnt/data1/gate.json",
        ]
    )
    assert args.command == "phase2"
    assert args.evaluation_root == ["/mnt/data1/eval-b0", "/mnt/data1/eval-l1"]


def test_phase2_summary_rejects_wrong_nfe_or_control(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 2)
    for video_nfe, control in ((25, "off"), (50, "autonomous")):
        summary = {
            "summaries": [
                {
                    "auxiliary_nfe": 0,
                    "video_nfe": video_nfe,
                    "control": control,
                    "clips": 2,
                    "total_nfe": video_nfe,
                    **{metric: 1.0 for metric in gate.ALL_PRIMARY_METRICS},
                }
            ]
        }
        with pytest.raises(gate.GateError, match="inventory differs"):
            gate._phase2_summary_index(summary, {(0, 50, "off")})


def _control_rows(arm: str) -> tuple[str, str, dict[str, dict]]:
    clip_id = "a" * 64
    donor_id = "b" * 64
    common = {
        "checkpoint_sha256": "0" * 64,
        "training_config_sha256": "1" * 64,
        "shuffle_mapping_sha256": "2" * 64,
        "history_sha256": "3" * 64,
        "actions_sha256": "4" * 64,
        "initial_video_noise_sha256": "5" * 64,
        "target_auxiliary_sha256": "6" * 64,
        "zero_auxiliary_sha256": "7" * 64,
        "generated_video_sha256": "8" * 64,
        "teacher_model_calls": 0,
        "lpips_alex_frame": 0.5,
        "lpips_alex_temporal_difference": 0.5,
        "temporal_difference_mse": 0.5,
        "rgb_mse": 0.5,
        "rgb_psnr": 20.0,
        "generated_pixel_clipped_fraction": 0.0,
    }
    if arm == "B0":
        return clip_id, donor_id, {
            "off": {
                **common,
                "control": "off",
                "deployable": True,
                "clean_future_used_as_condition": False,
                "conditioning_source_clip_id": clip_id,
                "auxiliary_conditioning_source_clip_id": clip_id,
                "history_action_source_clip_id": clip_id,
                "auxiliary_frozen_assertion_executed": False,
                "phase_boundary_sha256": None,
                "conditioning_auxiliary_sha256": None,
                "pre_video_auxiliary_sha256": None,
                "post_video_auxiliary_sha256": None,
                "initial_auxiliary_noise_sha256": None,
            }
        }
    boundary = "9" * 64
    donor_boundary = "c" * 64
    controls = {}
    for control in gate.DUAL_CONTROLS:
        conditioning = (
            donor_boundary
            if control == "shuffled"
            else "6" * 64
            if control == "oracle_clean"
            else boundary
        )
        source = donor_id if control == "shuffled" else clip_id
        controls[control] = {
            **common,
            "control": control,
            "deployable": control != "oracle_clean",
            "clean_future_used_as_condition": control == "oracle_clean",
            "conditioning_source_clip_id": source,
            "auxiliary_conditioning_source_clip_id": source,
            "history_action_source_clip_id": clip_id,
            "auxiliary_frozen_assertion_executed": True,
            "phase_boundary_sha256": boundary,
            "conditioning_auxiliary_sha256": conditioning,
            "pre_video_auxiliary_sha256": conditioning,
            "post_video_auxiliary_sha256": conditioning,
            "initial_auxiliary_noise_sha256": "d" * 64,
        }
    return clip_id, donor_id, controls


def test_phase2_control_group_audits_freeze_shuffle_leakage_and_a1_noop():
    clip_id, donor_id, rows = _control_rows("A1")
    counts = gate._validate_phase2_control_group(
        arm="A1",
        clip_id=clip_id,
        donor_id=donor_id,
        rows_by_control=rows,
        donor_phase_boundary_sha256="c" * 64,
    )
    assert counts["a1_generated_video_noop_comparisons"] == 1

    tampered = copy.deepcopy(rows)
    tampered["autonomous"]["post_video_auxiliary_sha256"] = "e" * 64
    with pytest.raises(gate.GateError, match="changed during"):
        gate._validate_phase2_control_group(
            arm="A1", clip_id=clip_id, donor_id=donor_id,
            rows_by_control=tampered, donor_phase_boundary_sha256="c" * 64,
        )
    tampered = copy.deepcopy(rows)
    tampered["shuffled"]["conditioning_auxiliary_sha256"] = "e" * 64
    tampered["shuffled"]["pre_video_auxiliary_sha256"] = "e" * 64
    tampered["shuffled"]["post_video_auxiliary_sha256"] = "e" * 64
    with pytest.raises(gate.GateError, match="registered auxiliary"):
        gate._validate_phase2_control_group(
            arm="A1", clip_id=clip_id, donor_id=donor_id,
            rows_by_control=tampered, donor_phase_boundary_sha256="c" * 64,
        )
    tampered = copy.deepcopy(rows)
    tampered["autonomous"]["clean_future_used_as_condition"] = True
    with pytest.raises(gate.GateError, match="leakage"):
        gate._validate_phase2_control_group(
            arm="A1", clip_id=clip_id, donor_id=donor_id,
            rows_by_control=tampered, donor_phase_boundary_sha256="c" * 64,
        )
    tampered = copy.deepcopy(rows)
    tampered["oracle_clean"]["generated_video_sha256"] = "f" * 64
    with pytest.raises(gate.GateError, match="video tensor changed"):
        gate._validate_phase2_control_group(
            arm="A1", clip_id=clip_id, donor_id=donor_id,
            rows_by_control=tampered, donor_phase_boundary_sha256="c" * 64,
        )
    tampered = copy.deepcopy(rows)
    tampered["off"]["initial_video_noise_sha256"] = "f" * 64
    with pytest.raises(gate.GateError, match="initial_video_noise"):
        gate._validate_phase2_control_group(
            arm="A1", clip_id=clip_id, donor_id=donor_id,
            rows_by_control=tampered, donor_phase_boundary_sha256="c" * 64,
        )


def test_phase2_b0_has_no_auxiliary_freeze_evidence():
    clip_id, _, rows = _control_rows("B0")
    counts = gate._validate_phase2_control_group(
        arm="B0",
        clip_id=clip_id,
        donor_id=None,
        rows_by_control=rows,
        donor_phase_boundary_sha256=None,
    )
    assert counts["boundary_control_comparisons"] == 0


def test_phase2_structural_criteria_are_explicit_and_fail_closed(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 2)

    def evidence(arm: str) -> dict:
        pair_clips = 2 * len(gate._expected_phase2_pairs(arm))
        rows = pair_clips * len(gate._expected_phase2_controls(arm))
        return {
            "audited_rows": rows,
            "teacher_model_call_sum": 0,
            "deployable_rows": rows if arm == "B0" else pair_clips * 3,
            "oracle_clean_rows": 0 if arm == "B0" else pair_clips,
            "clean_future_condition_rows": 0 if arm == "B0" else pair_clips,
            "deployable_clean_future_condition_rows": 0,
            "dual_auxiliary_freeze_rows": 0 if arm == "B0" else rows,
            "b0_auxiliary_strict_noop_rows": rows if arm == "B0" else 0,
            "boundary_control_comparisons": 0 if arm == "B0" else pair_clips,
            "registered_auxiliary_bindings": 0 if arm == "B0" else pair_clips * 4,
            "deployable_generated_auxiliary_bindings": (
                0 if arm == "B0" else pair_clips * 3
            ),
            "shuffled_donor_bindings": 0 if arm == "B0" else pair_clips,
            "a1_generated_video_noop_comparisons": pair_clips if arm == "A1" else 0,
        }

    audits = {
        (arm, 500): {"structural_evidence": evidence(arm)}
        for arm in gate.DUAL_ARMS
    }
    criteria = gate._phase2_structural_criteria(audits)
    assert criteria["auxiliary_mechanism"]["passed"] is True
    assert criteria["no_inference_leakage"]["passed"] is True
    audits[("L1", 500)]["structural_evidence"][
        "deployable_clean_future_condition_rows"
    ] = 1
    with pytest.raises(gate.GateError, match="structural evidence count"):
        gate._phase2_structural_criteria(audits)


def test_phase2_cross_root_inputs_frechet_source_and_output_fail_closed(
    monkeypatch, tmp_path
):
    reference = {"clip": {"history_sha256": "a" * 64}}
    gate._require_identical_phase2_inputs(reference, copy.deepcopy(reference), label="inputs")
    with pytest.raises(gate.GateError, match="changed inputs"):
        gate._require_identical_phase2_inputs(
            reference,
            {"clip": {"history_sha256": "b" * 64}},
            label="inputs",
        )
    gate._require_frechet_match(1.0, 1.0, label="primary")
    with pytest.raises(gate.GateError, match="differs from pinned"):
        gate._require_frechet_match(1.0, 1.1, label="primary")
    monkeypatch.setattr(gate, "git_record", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(gate.GateError, match="dirty"):
        gate._validate_source_binding({"commit": "abc", "dirty": False})
    monkeypatch.setattr(gate, "git_record", lambda: {"commit": "other", "dirty": False})
    with pytest.raises(gate.GateError, match="differs"):
        gate._validate_source_binding({"commit": "abc", "dirty": False})

    monkeypatch.setattr(gate, "APPROVED_ROOTS", (tmp_path,))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    output = tmp_path / "gate.json"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(gate.GateError, match="already exists"):
        gate.validate_phase2_output_path(output, [evidence])


def test_phase2_feature_payload_tamper_fails_before_frechet(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "CLIPS", 2)
    path = tmp_path / "features.pt"
    import torch

    torch.save(
        {
            "clip_ids": ["a", "b"],
            "r3d18_real_features": torch.zeros(2, 512),
            "r3d18_generated_features": torch.zeros(2, 512),
            "feature_name": "tampered",
            "feature_dimension": 512,
        },
        path,
    )
    with pytest.raises(gate.GateError, match="violates the frozen contract"):
        gate._recompute_primary_frechet(gate.file_record(path), clip_ids=["a", "b"])


def test_exact_quality_provenance_rejects_weight_tamper(monkeypatch, tmp_path):
    from robot_wm.evaluation import video_latent_forcing_quality as quality

    paths = {}
    for name in ("linear", "alexnet", "r3d18"):
        path = tmp_path / f"{name}.pth"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    hashes = {name: gate.sha256_file(path) for name, path in paths.items()}
    monkeypatch.setattr(quality, "LPIPS_LINEAR_SHA256", hashes["linear"])
    monkeypatch.setattr(quality, "ALEXNET_SHA256", hashes["alexnet"])
    monkeypatch.setattr(quality, "R3D18_SHA256", hashes["r3d18"])

    def weight(name: str, role: str, source_url: str) -> dict:
        return {
            "role": role,
            "path": str(paths[name].resolve()),
            "size_bytes": paths[name].stat().st_size,
            "sha256": hashes[name],
            "expected_sha256": hashes[name],
            "expected_sha256_prefix": None,
            "source_url": source_url,
        }

    preprocessing = quality.preprocessing_specification()
    unsigned = {
        "metrics": [
            quality.LPIPS_ALEX_FRAME_METRIC,
            quality.LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC,
            quality.R3D18_FRECHET_METRIC,
        ],
        "perceptual_extractor": {
            "extractor": "FrozenLPIPSAlex",
            "package": {"name": "lpips", "version": quality.LPIPS_PACKAGE_VERSION},
            "weights": [
                weight(
                    "linear",
                    "LPIPS-Alex v0.1 linear calibration",
                    "https://github.com/richzhang/PerceptualSimilarity/"
                    "tree/master/lpips/weights/v0.1",
                ),
                weight("alexnet", "ImageNet AlexNet backbone", quality.ALEXNET_URL),
            ],
            "preprocessing": preprocessing,
        },
        "video_feature_extractor": {
            "extractor": "FrozenR3D18AvgPool",
            "package": {
                "name": "torchvision",
                "version": quality.TORCHVISION_PACKAGE_VERSION,
            },
            "weights_enum": "R3D_18_Weights.KINETICS400_V1",
            "weights": [
                weight(
                    "r3d18",
                    "torchvision R3D-18 Kinetics-400 V1",
                    quality.R3D18_URL,
                )
            ],
            "preprocessing": preprocessing,
        },
        "preprocessing": preprocessing,
    }
    provenance = {**unsigned, "sha256": gate.sha256_json(unsigned)}
    audit = gate._validate_quality_provenance(provenance)
    assert audit["weights"]["r3d18"]["sha256"] == hashes["r3d18"]

    paths["r3d18"].write_bytes(b"changed")
    with pytest.raises(gate.GateError, match="missing, changed, or not pinned"):
        gate._validate_quality_provenance(provenance)
