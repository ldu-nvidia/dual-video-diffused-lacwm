import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.audit_stage_faithful_artifacts import (
    ARTIFACT_ITERATION,
    IDENTITY_TENSORS,
    LEGACY_SOURCE_CODES,
    NEW_SOURCE_CODES,
    NFE_STEPS,
    SIGMA_CONVENTION,
    StageArtifactAuditError,
    audit,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _infix(source: str) -> str:
    return "" if source == "autonomous" else f"_{source}"


def _base_tensors(rank: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20_260_726 + rank)
    video_clean = (
        torch.arange(12, dtype=torch.float16).reshape(1, 2, 3, 2, 1)
        + rank
    )
    tf_clean = (
        torch.arange(18, dtype=torch.float16).reshape(1, 3, 3, 2, 1)
        + 2 * rank
    )
    video_initial = torch.randn(
        video_clean.shape, dtype=torch.float16, generator=generator
    )
    tf_noise = torch.randn(
        tf_clean.shape, dtype=torch.float16, generator=generator
    )
    tf_initial = tf_noise.clone()
    tf_initial[:, :, :1] = tf_clean[:, :, :1]
    decoded_target = (
        torch.arange(24, dtype=torch.uint8).reshape(1, 3, 2, 2, 2) + rank
    )
    return {
        "video_clean": video_clean,
        "tf_clean": tf_clean,
        "video_initial_state": video_initial,
        "tf_initial_state": tf_initial,
        "tf_initial_noise": tf_noise,
        "history_latent_frames": torch.tensor([1], dtype=torch.int64),
        "evaluation_noise_seed": torch.tensor([20_260_726], dtype=torch.int64),
        "ground_truth_future_uint8": decoded_target,
        "condition_on_tf": torch.tensor([1], dtype=torch.int64),
        "condition_only_video_loss_examples": torch.tensor(
            [1], dtype=torch.int64
        ),
        "condition_mode_code": torch.tensor([1], dtype=torch.int64),
        "oracle_sources_are_leakage": torch.tensor([1], dtype=torch.int64),
        "video_sigmas": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "tf_sigmas": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }


def _legacy_tensors(rank: int) -> dict[str, torch.Tensor]:
    tensors = _base_tensors(rank)
    tensors["evaluation_nfe_steps"] = torch.tensor(
        NFE_STEPS, dtype=torch.int64
    )
    tensors["evaluation_condition_source_codes"] = torch.tensor(
        LEGACY_SOURCE_CODES, dtype=torch.int64
    )
    for nfe in NFE_STEPS:
        for source_index, source in enumerate(
            ("autonomous", "off", "oracle_matched", "oracle_shuffled")
        ):
            infix = _infix(source)
            tensors[f"video_final{infix}_nfe_{nfe}"] = (
                tensors["video_clean"] + (10 * source_index + nfe)
            )
            if source == "off":
                # Stage-faithful TF generation has no content injection and
                # therefore must exactly reproduce this legacy off state.
                tf_final = tensors["tf_clean"] + (100 + nfe)
            else:
                tf_final = tensors["tf_clean"] + (20 * source_index + nfe)
            tensors[f"tf_final{infix}_nfe_{nfe}"] = tf_final
            tensors[f"decoded_future{infix}_nfe_{nfe}"] = torch.clamp(
                tensors["ground_truth_future_uint8"].to(torch.int16)
                + source_index
                + nfe,
                0,
                255,
            ).to(torch.uint8)
    return tensors


def _new_tensors(
    rank: int, legacy: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    tensors = {
        key: value.clone()
        for key, value in legacy.items()
        if key in set(IDENTITY_TENSORS)
        | {
            "condition_on_tf",
            "condition_only_video_loss_examples",
            "condition_mode_code",
            "oracle_sources_are_leakage",
            "video_sigmas",
            "tf_sigmas",
        }
    }
    tensors["evaluation_nfe_steps"] = torch.tensor(
        NFE_STEPS, dtype=torch.int64
    )
    tensors["evaluation_condition_source_codes"] = torch.tensor(
        NEW_SOURCE_CODES, dtype=torch.int64
    )
    tensors["cascade_stage_faithful_inference"] = torch.tensor(
        [1], dtype=torch.int64
    )
    for nfe in NFE_STEPS:
        for source_index, source in enumerate(
            ("autonomous", "autonomous_shuffled", "autonomous_legacy", "off")
        ):
            infix = _infix(source)
            if source == "autonomous_legacy":
                for prefix in ("video_final", "tf_final", "decoded_future"):
                    tensors[f"{prefix}{infix}_nfe_{nfe}"] = legacy[
                        f"{prefix}_nfe_{nfe}"
                    ].clone()
                continue
            tensors[f"video_final{infix}_nfe_{nfe}"] = (
                tensors["video_clean"] + (50 * source_index + nfe)
            )
            if source in {"autonomous", "autonomous_shuffled"}:
                tf_final = legacy[f"tf_final_off_nfe_{nfe}"].clone()
            else:
                tf_final = tensors["tf_clean"] + (70 * source_index + nfe)
            tensors[f"tf_final{infix}_nfe_{nfe}"] = tf_final
            tensors[f"decoded_future{infix}_nfe_{nfe}"] = torch.clamp(
                tensors["ground_truth_future_uint8"].to(torch.int16)
                + 3 * source_index
                + nfe,
                0,
                255,
            ).to(torch.uint8)
    return tensors


def _write_rank(
    root: Path, rank: int, tensors: dict[str, torch.Tensor]
) -> Path:
    dataset = "ABC_0"
    folder = (
        root
        / "visualization"
        / f"iter_{ARTIFACT_ITERATION}"
        / dataset
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"latent_trajectory_rank_{rank}.safetensors"
    save_file(
        tensors,
        str(path),
        metadata={
            "iteration": str(ARTIFACT_ITERATION),
            "dataset": dataset,
            "sigma_convention": SIGMA_CONVENTION,
        },
    )
    sidecar = {
        "iteration": ARTIFACT_ITERATION,
        "dataset": dataset,
        "global_rank": rank,
        "sigma_convention": SIGMA_CONVENTION,
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in tensors.items()
        },
        "safetensors_sha256": _sha256(path),
    }
    path.with_suffix(".json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return path


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    new_root = tmp_path / "new"
    legacy_root = tmp_path / "legacy"
    for rank in range(8):
        legacy = _legacy_tensors(rank)
        _write_rank(legacy_root, rank, legacy)
        _write_rank(new_root, rank, _new_tensors(rank, legacy))
    return new_root, legacy_root


def _rewrite_tensor(
    root: Path, rank: int, key: str, value: torch.Tensor
) -> None:
    path = next(
        root.rglob(f"latent_trajectory_rank_{rank}.safetensors")
    )
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


def test_audit_passes_all_eight_rank_bitwise_contracts(tmp_path):
    new_root, legacy_root = _roots(tmp_path)
    before_new = sorted(
        (str(path.relative_to(new_root)), _sha256(path))
        for path in new_root.rglob("*")
        if path.is_file()
    )
    payload = audit(new_root=new_root, legacy_root=legacy_root)
    after_new = sorted(
        (str(path.relative_to(new_root)), _sha256(path))
        for path in new_root.rglob("*")
        if path.is_file()
    )

    assert before_new == after_new
    assert payload["overall_pass"] is True
    assert payload["contracts"]["pass"] is True
    assert payload["contracts"]["world_size"]["pass"] is True
    assert payload["contracts"]["paired_ranks"]["expected"] == list(range(8))
    assert payload["contracts"]["new_source_codes"]["expected"] == [0, 4, 5, 1]
    assert payload["contracts"]["legacy_source_codes"]["expected"] == [0, 1, 2, 3]
    assert payload["contracts"]["nfe_steps"]["expected"] == [2, 4, 8]
    assert payload["contracts"]["new_stage_faithful_flag"]["pass"] is True
    assert payload["contracts"]["forbidden_training_outputs"]["pass"] is True
    assert len(payload["rank_audits"]) == 8
    assert all(result["pass"] is True for result in payload["rank_audits"])
    assert all(
        result["new_legacy_input_identity"]["pass"] is True
        and result["legacy_reproduction"]["pass"] is True
        and result["stage_tf_equivalence"]["pass"] is True
        for result in payload["rank_audits"]
    )
    assert len(payload["identity_sha256"]) == 64
    assert len(payload["inputs"]["new"]["artifact_set_sha256"]) == 64
    assert all(
        rank["sidecar"]["safetensors_sha256_matches"] is True
        and rank["sidecar"]["sigma_convention_matches"] is True
        for rank in payload["inputs"]["new"]["ranks"]
    )


def test_cli_exclusively_writes_only_external_report(tmp_path, capsys):
    new_root, legacy_root = _roots(tmp_path)
    output = tmp_path / "audit.json"
    assert (
        main(
            [
                "--new-root",
                str(new_root),
                "--legacy-root",
                str(legacy_root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["overall_pass"] is True
    assert stdout["output_sha256"] == _sha256(output)
    assert written["overall_pass"] is True
    assert output.stat().st_mode & 0o777 == 0o600

    assert (
        main(
            [
                "--new-root",
                str(new_root),
                "--legacy-root",
                str(legacy_root),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "output already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("root_name", "key", "value"),
    [
        (
            "new",
            "evaluation_condition_source_codes",
            torch.tensor([0, 5, 4, 1], dtype=torch.int64),
        ),
        (
            "legacy",
            "evaluation_condition_source_codes",
            torch.tensor([0, 1, 3, 2], dtype=torch.int64),
        ),
        (
            "new",
            "evaluation_nfe_steps",
            torch.tensor([2, 4, 7], dtype=torch.int64),
        ),
        (
            "new",
            "cascade_stage_faithful_inference",
            torch.tensor([0], dtype=torch.int64),
        ),
    ],
)
def test_rejects_source_nfe_and_stage_contract_changes(
    tmp_path, root_name, key, value
):
    new_root, legacy_root = _roots(tmp_path)
    root = new_root if root_name == "new" else legacy_root
    _rewrite_tensor(root, 3, key, value)
    with pytest.raises(StageArtifactAuditError):
        audit(new_root=new_root, legacy_root=legacy_root)


@pytest.mark.parametrize(
    ("root_name", "key"),
    [
        ("new", "video_clean"),
        ("new", "video_initial_state"),
        ("new", "tf_initial_noise"),
        ("new", "history_latent_frames"),
        ("new", "evaluation_noise_seed"),
        ("new", "ground_truth_future_uint8"),
    ],
)
def test_rejects_new_legacy_input_identity_changes(tmp_path, root_name, key):
    new_root, legacy_root = _roots(tmp_path)
    root = new_root if root_name == "new" else legacy_root
    path = next(root.rglob("latent_trajectory_rank_2.safetensors"))
    current = load_file(str(path), device="cpu")[key]
    changed = current.clone()
    changed.view(-1)[0] = changed.view(-1)[0] + 1
    _rewrite_tensor(root, 2, key, changed)
    with pytest.raises(
        StageArtifactAuditError, match="new/legacy identity mismatch"
    ):
        audit(new_root=new_root, legacy_root=legacy_root)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (
            "video_final_autonomous_legacy_nfe_4",
            "legacy reproduction mismatch",
        ),
        (
            "tf_final_autonomous_shuffled_nfe_8",
            "stage autonomous/shuffled TF mismatch",
        ),
        (
            "tf_final_off_nfe_2",
            "stage autonomous/legacy-off TF mismatch",
        ),
    ],
)
def test_rejects_bitwise_control_mismatches(tmp_path, key, message):
    new_root, legacy_root = _roots(tmp_path)
    root = legacy_root if key == "tf_final_off_nfe_2" else new_root
    path = next(root.rglob("latent_trajectory_rank_6.safetensors"))
    current = load_file(str(path), device="cpu")[key]
    changed = current.clone()
    changed.view(torch.uint8).view(-1)[0] ^= 1
    _rewrite_tensor(root, 6, key, changed)
    with pytest.raises(StageArtifactAuditError, match=message):
        audit(new_root=new_root, legacy_root=legacy_root)


def test_rejects_missing_rank_tampered_sidecar_and_bad_sigma(tmp_path):
    new_root, legacy_root = _roots(tmp_path)
    missing = next(new_root.rglob("latent_trajectory_rank_7.safetensors"))
    missing.unlink()
    missing.with_suffix(".json").unlink()
    with pytest.raises(StageArtifactAuditError, match="exactly 8"):
        audit(new_root=new_root, legacy_root=legacy_root)

    new_root, legacy_root = _roots(tmp_path / "tamper")
    sidecar_path = next(
        new_root.rglob("latent_trajectory_rank_1.json")
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["safetensors_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(StageArtifactAuditError, match="SHA-256 mismatch"):
        audit(new_root=new_root, legacy_root=legacy_root)

    new_root, legacy_root = _roots(tmp_path / "sigma")
    sidecar_path = next(
        legacy_root.rglob("latent_trajectory_rank_0.json")
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["sigma_convention"] = "0=noise,1=clean"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(StageArtifactAuditError, match="sigma convention"):
        audit(new_root=new_root, legacy_root=legacy_root)


@pytest.mark.parametrize(
    "forbidden_name",
    ["_never_write_snapshot.pt", "snapshot.pt", "training_complete.json"],
)
def test_rejects_forbidden_training_outputs(tmp_path, forbidden_name):
    new_root, legacy_root = _roots(tmp_path)
    (new_root / forbidden_name).write_bytes(b"forbidden")
    with pytest.raises(
        StageArtifactAuditError, match="forbidden training outputs"
    ):
        audit(new_root=new_root, legacy_root=legacy_root)


def test_rejects_output_inside_inputs(tmp_path, capsys):
    new_root, legacy_root = _roots(tmp_path)
    output = new_root / "audit.json"
    assert (
        main(
            [
                "--new-root",
                str(new_root),
                "--legacy-root",
                str(legacy_root),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "outside evaluated input roots" in capsys.readouterr().err
    assert not output.exists()
