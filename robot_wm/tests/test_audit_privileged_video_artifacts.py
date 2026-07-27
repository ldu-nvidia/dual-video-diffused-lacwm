import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.audit_privileged_video_artifacts import (
    ARM_NAMES,
    ARTIFACT_ITERATION,
    EXPECTED_PARENTS,
    NFE_STEPS,
    PROVENANCE_NAME,
    PrivilegedArtifactAuditError,
    audit,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensors(rank: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1000 + rank)
    video = torch.randn((1, 2, 3, 2, 2), generator=generator)
    tf = torch.randn((1, 4, 3, 2, 2), generator=generator)
    video_initial = torch.randn(video.shape, generator=generator)
    tf_noise = torch.randn(tf.shape, generator=generator)
    tf_initial = tf_noise.clone()
    tf_initial[:, :, :1] = tf[:, :, :1]
    target = torch.randint(
        0, 256, (1, 3, 4, 2, 2), generator=generator, dtype=torch.uint8
    )
    tensors = {
        "video_clean": video,
        "tf_clean": tf,
        "video_initial_state": video_initial,
        "tf_initial_state": tf_initial,
        "tf_initial_noise": tf_noise,
        "history_latent_frames": torch.tensor([1]),
        "evaluation_noise_seed": torch.tensor([20_260_726]),
        "ground_truth_future_uint8": target,
        "raw_actions": torch.tensor(
            [[[0.125, -0.25], [0.375, -0.5]]],
            dtype=torch.float32,
        ),
        "raw_morphology_index": torch.tensor([3], dtype=torch.int64),
        "raw_actions_present": torch.tensor([1]),
        "raw_morphology_index_present": torch.tensor([1]),
        "z_control": torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4),
        "condition_on_tf": torch.tensor([0]),
        "condition_mode_code": torch.tensor([0]),
        "cascade_stage_faithful_inference": torch.tensor([0]),
        "evaluation_disable_tf_clock": torch.tensor([1]),
        "evaluation_tf_clock_enabled": torch.tensor([0]),
        "evaluation_all_video_schedule": torch.tensor([1]),
        "evaluation_condition_source_codes": torch.tensor([0, 1]),
        "evaluation_nfe_steps": torch.tensor(list(NFE_STEPS)),
    }
    for nfe in NFE_STEPS:
        video_final = video + rank + nfe / 100
        tf_final = tf + rank + nfe / 100
        decoded = torch.clamp(
            target.to(torch.int16) + rank + nfe, 0, 255
        ).to(torch.uint8)
        tensors[f"video_final_nfe_{nfe}"] = video_final
        tensors[f"video_final_off_nfe_{nfe}"] = video_final.clone()
        tensors[f"tf_final_nfe_{nfe}"] = tf_final
        tensors[f"tf_final_off_nfe_{nfe}"] = tf_final.clone()
        tensors[f"decoded_future_nfe_{nfe}"] = decoded
        tensors[f"decoded_future_off_nfe_{nfe}"] = decoded.clone()
    return tensors


def _write_rank(root: Path, rank: int, tensors: dict[str, torch.Tensor]) -> None:
    dataset = "MultiDatasetABC_0"
    folder = root / "visualization" / "iter_199" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"latent_trajectory_rank_{rank}.safetensors"
    save_file(
        tensors,
        str(path),
        metadata={
            "iteration": str(ARTIFACT_ITERATION),
            "dataset": dataset,
            "sigma_convention": "1=noise,0=clean",
        },
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "iteration": ARTIFACT_ITERATION,
                "dataset": dataset,
                "global_rank": rank,
                "sigma_convention": "1=noise,0=clean",
                "tensors": {
                    key: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for key, value in tensors.items()
                },
                "safetensors_sha256": _sha256(path),
            }
        ),
        encoding="utf-8",
    )


def _write_provenance(root: Path, arm: str) -> None:
    (root / PROVENANCE_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "dual_video_diffusion_privileged_video_evaluation",
                "status": "visualization_completed",
                "generated_at_utc": "2026-07-26T00:00:00+00:00",
                "evaluation_only": True,
                "evaluation_optimizer_updates": 0,
                "evaluation_total_observations": 0,
                "artifact_iteration": ARTIFACT_ITERATION,
                "viz_skip_batches": 4,
                "evaluation_condition_sources": ["autonomous", "off"],
                "evaluation_nfe_steps": list(NFE_STEPS),
                "runtime_intervention": {
                    "schedule_mode": "aligned",
                    "tf_content_disabled": True,
                    "tf_clock_disabled": True,
                    "all_model_calls_advance_video": True,
                },
                "parent": {
                    **EXPECTED_PARENTS[arm],
                    "completed_updates": 200,
                    "total_observations": 1600,
                },
                "artifact_root": str(
                    (root / "visualization" / "iter_199").resolve()
                ),
                "snapshot_written": False,
                "training_completion_written": False,
            }
        ),
        encoding="utf-8",
    )


def _roots(tmp_path: Path) -> dict[str, Path]:
    roots = {arm: tmp_path / arm for arm in ARM_NAMES}
    for arm, root in roots.items():
        for rank in range(8):
            _write_rank(root, rank, _tensors(rank))
        _write_provenance(root, arm)
    return roots


def _rewrite_tensor(
    root: Path, rank: int, key: str, value: torch.Tensor
) -> None:
    path = next(root.rglob(f"latent_trajectory_rank_{rank}.safetensors"))
    tensors = load_file(str(path), device="cpu")
    tensors[key] = value
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    save_file(tensors, str(path), metadata=metadata)
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["tensors"] = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in tensors.items()
    }
    sidecar["safetensors_sha256"] = _sha256(path)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")


def _run_audit(roots: dict[str, Path]) -> dict:
    return audit(
        trained_matched_root=roots["trained_matched"],
        trained_shuffled_root=roots["trained_shuffled"],
        trained_off_root=roots["trained_off"],
    )


def test_audit_passes_and_does_not_mutate_inputs(tmp_path):
    roots = _roots(tmp_path)
    before = {
        arm: sorted(
            (str(path.relative_to(root)), _sha256(path))
            for path in root.rglob("*")
            if path.is_file()
        )
        for arm, root in roots.items()
    }
    payload = _run_audit(roots)
    after = {
        arm: sorted(
            (str(path.relative_to(root)), _sha256(path))
            for path in root.rglob("*")
            if path.is_file()
        )
        for arm, root in roots.items()
    }
    assert before == after
    assert payload["overall_pass"] is True
    assert payload["contracts"]["exact_parent_provenance"]["pass"] is True
    assert (
        payload["contracts"]["raw_action_morphology_input_identity"][
            "tensors"
        ]
        == ["raw_actions", "raw_morphology_index"]
    )
    assert (
        payload["contracts"]["learned_action_control_diagnostic"][
            "cross_arm_equality_required"
        ]
        is False
    )
    assert len(payload["rank_audits"]) == 8


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("evaluation_condition_source_codes", torch.tensor([1, 0])),
        ("evaluation_nfe_steps", torch.tensor([1, 2, 4, 7])),
        ("evaluation_noise_seed", torch.tensor([20_260_725])),
        ("condition_on_tf", torch.tensor([1])),
        ("condition_mode_code", torch.tensor([1])),
        ("cascade_stage_faithful_inference", torch.tensor([1])),
        ("evaluation_disable_tf_clock", torch.tensor([0])),
        ("evaluation_tf_clock_enabled", torch.tensor([1])),
        ("evaluation_all_video_schedule", torch.tensor([0])),
        ("raw_actions_present", torch.tensor([0])),
        ("raw_morphology_index_present", torch.tensor([0])),
    ],
)
def test_rejects_runtime_contract_mutations(tmp_path, key, value):
    roots = _roots(tmp_path)
    _rewrite_tensor(roots["trained_matched"], 3, key, value)
    with pytest.raises(PrivilegedArtifactAuditError):
        _run_audit(roots)


def test_rejects_raw_action_morphology_input_mismatch(tmp_path):
    roots = _roots(tmp_path)
    changed = _tensors(2)["raw_actions"].clone()
    changed.view(-1)[0] += 1
    _rewrite_tensor(roots["trained_shuffled"], 2, "raw_actions", changed)
    with pytest.raises(
        PrivilegedArtifactAuditError,
        match="identity mismatch for raw_actions",
    ):
        _run_audit(roots)


def test_allows_checkpoint_specific_learned_z_control(tmp_path):
    roots = _roots(tmp_path)
    changed = _tensors(2)["z_control"].clone()
    changed.view(-1)[0] += 1
    _rewrite_tensor(roots["trained_shuffled"], 2, "z_control", changed)

    payload = _run_audit(roots)

    assert payload["overall_pass"] is True


@pytest.mark.parametrize("key", ["video_final_nfe_4", "decoded_future_nfe_4"])
def test_rejects_autonomous_off_noop_mutation(tmp_path, key):
    roots = _roots(tmp_path)
    original = _tensors(5)[key].clone()
    original.view(-1)[0] += 1
    off_key = key.replace("_nfe_", "_off_nfe_")
    _rewrite_tensor(roots["trained_off"], 5, off_key, original)
    with pytest.raises(PrivilegedArtifactAuditError, match="not bitwise"):
        _run_audit(roots)


def test_rejects_swapped_parent_provenance(tmp_path):
    roots = _roots(tmp_path)
    path = roots["trained_shuffled"] / PROVENANCE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parent"].update(EXPECTED_PARENTS["trained_matched"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PrivilegedArtifactAuditError,
        match="parent mapping mismatch",
    ):
        _run_audit(roots)


def test_rejects_forbidden_training_output(tmp_path):
    roots = _roots(tmp_path)
    (roots["trained_off"] / "snapshot.pt").write_bytes(b"forbidden")
    with pytest.raises(
        PrivilegedArtifactAuditError,
        match="forbidden training outputs",
    ):
        _run_audit(roots)


def test_cli_exclusive_external_output(tmp_path, capsys):
    roots = _roots(tmp_path)
    output = tmp_path / "audit.json"
    args = [
        "--trained-matched-root",
        str(roots["trained_matched"]),
        "--trained-shuffled-root",
        str(roots["trained_shuffled"]),
        "--trained-off-root",
        str(roots["trained_off"]),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["overall_pass"] is True
    assert stdout["output_sha256"] == _sha256(output)
    assert output.stat().st_mode & 0o777 == 0o600
    assert main(args) == 2
    assert "output already exists" in capsys.readouterr().err
