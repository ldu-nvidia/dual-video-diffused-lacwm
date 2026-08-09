from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage = _load("physics_flow_stage1_test", "tools/physics_flow_stage1.py")
cache_runtime = _load(
    "physics_flow_cache_runtime_test",
    "tools/physics_flow_cache_runtime.py",
)
snapshot_receipt = _load(
    "snapshot_model_state_receipt_test",
    "tools/snapshot_model_state_receipt.py",
)
parent_adapter = _load(
    "physics_flow_parent_vpm_test",
    "tools/physics_flow_parent_vpm.py",
)
parent_parity_tool = _load(
    "physics_flow_parent_parity_test",
    "tools/physics_flow_parent_parity.py",
)
abc_probe = sys.modules["tools.abc_d405_nominal_geometry_probe"]


def test_endpoint_grid_is_equal_call_raw_only() -> None:
    assert len(stage.ENDPOINTS) == 21
    assert {arm.code for arm in stage.ARMS} == {"FLOW-OFF", "RAW-FLOW"}
    assert {endpoint.arm for endpoint in stage.ENDPOINTS} == {
        "FLOW-OFF",
        "RAW-FLOW",
        "PARENT-VPM",
    }
    assert {
        endpoint.condition_source
        for endpoint in stage.ENDPOINTS
        if endpoint.arm == "RAW-FLOW"
    } == set(stage.RUNTIME_SOURCES)
    assert all(endpoint.nfe in (1, 2, 4) for endpoint in stage.ENDPOINTS)
    assert all(
        endpoint.condition_source == "off"
        for endpoint in stage.ENDPOINTS
        if endpoint.arm in {"FLOW-OFF", "PARENT-VPM"}
    )
    assert [
        endpoint.nfe
        for endpoint in stage.ENDPOINTS
        if endpoint.arm == "PARENT-VPM"
    ] == [1, 2, 4]
    source = (ROOT / "tools/physics_flow_stage1.py").read_text()
    assert "recurrent_delta" not in source
    assert "hybrid_anchor_raw_delta" not in source
    assert "file_record(state_path)" not in source
    assert '"only_arrays_indexed"' in source
    assert '"content_bytes_read_for_provenance": False' in source
    assert stage.PARENT_SNAPSHOT_SHA256.startswith("de65e832")
    assert stage.RENDERER_REGISTRATION_IDENTITY == (
        "9cc556aba53d1defb69b0049dab67d12a3991decb917015bba5153c16cb8c2b1"
    )
    assert stage.PARENT_RUN_IDENTITY_SHA256.startswith("d79c3699")
    assert stage.PARENT_CANONICAL_MODEL_STATE_SHA256.startswith("d1231b8b")
    assert stage.PARENT_RESOLVED_CONFIG_SHA256.startswith("ae3ffd27")
    assert stage.PARENT_TRAINING_SOURCE_COMMIT.startswith("6560866")
    assert stage.PARENT_NATIVE_SAMPLER_SOURCE_SHA256.startswith("a10fe373")
    assert stage.PARENT_NATIVE_SAMPLER_GIT_BLOB.startswith("abc6d4df")
    assert stage.sha256_file(
        ROOT / stage.PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE
    ) == stage.PARENT_NATIVE_SAMPLER_SOURCE_SHA256
    changed_parent_runtime = set(
        filter(
            None,
            stage.git(
                ROOT,
                "diff",
                "--name-only",
                stage.PARENT_TRAINING_SOURCE_COMMIT,
                "--",
                stage.PARENT_NATIVE_SAMPLER_SOURCE_RELATIVE,
                *stage.PARENT_TRANSITIVE_SOURCE_FILES,
            ).splitlines(),
        )
    )
    assert changed_parent_runtime == set(stage.PARENT_TRANSITIVE_SOURCE_FILES)
    assert stage.LEGACY_REJECTED_SNAPSHOT_SHA256.startswith("f67c7bae")


def test_snapshot_receipt_hashes_scalar_and_records_lineage(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.pt"
    torch.save(
        {
            "model": {
                "scalar": torch.tensor(0.25),
                "matrix": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            },
            "snapshot_schema_version": 3,
            "run_identity_sha256": "a" * 64,
            "_start_iter": 1,
            "world_size": 1,
            "gradient_accumulation_steps": 1,
        },
        path,
    )
    receipt, tensors = snapshot_receipt.model_state_receipt(path)
    assert receipt["model_tensor_count"] == 2
    assert len(receipt["canonical_model_state_sha256"]) == 64
    assert set(tensors) == {"matrix", "scalar"}


def test_parent_parity_hash_handles_scalar_and_native_artifacts() -> None:
    assert len(parent_parity_tool._torch_tensor_hash(torch.tensor(0.25))) == 64
    noise = torch.randn(1, 16, 4, 2, 3)
    predicted = torch.linspace(-1.0, 1.0, 1 * 3 * 8 * 2 * 3).reshape(
        1, 3, 8, 2, 3
    )
    decoded = (
        ((predicted.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
    )

    class FakeParent:
        _last_sampling_counters = {
            "wan_calls_by_source_nfe": {"off:nfe_1": 1},
            "wan_calls_total": 1,
            "online_teacher_calls": 0,
            "auxiliary_clean_available": 0,
            "deployment_mode": 1,
        }

        def pop_visualization_artifacts(self):
            return {
                "deployment_mode": torch.tensor([1]),
                "auxiliary_clean_available": torch.tensor([0]),
                "online_teacher_call_count": torch.tensor([0]),
                "evaluation_nfe_steps": torch.tensor([1]),
                "video_final_off_nfe_1": torch.zeros(1, 16, 4, 2, 3),
                "decoded_future_off_nfe_1": decoded,
                "video_initial_state": noise.to(torch.float16),
                "sample_ids": torch.tensor([7_000_000]),
            }

    result = parent_adapter.extract_native_parent_artifacts(
        FakeParent(),
        nfe=1,
        predicted_future=predicted,
        expected_video_noise=noise,
    )
    assert result["wan_calls"] == 1
    assert result["native_public_sampler"] == "sample_future_deployable"
    assert torch.equal(result["decoded_uint8"], decoded)


def test_support_weighted_pool_and_bottom_padding() -> None:
    native = np.zeros((4, 180, 320), dtype=np.float32)
    native[0] = 0.25
    native[1] = -0.5
    native[2] = 1.0
    native[3] = 0.125
    pooled = stage.pool_transition_field(native)
    assert pooled.shape == (4, 24, 40)
    np.testing.assert_array_equal(pooled[0, :23], 0.25)
    np.testing.assert_array_equal(pooled[1, :23], -0.5)
    np.testing.assert_array_equal(pooled[3, :23], 0.125)
    np.testing.assert_array_equal(pooled[2, :22], 1.0)
    np.testing.assert_array_equal(pooled[2, 22], 0.5)
    np.testing.assert_array_equal(pooled[:, 23], 0.0)


def test_cache_python_identity_preserves_venv_symlink_entry(tmp_path: Path) -> None:
    shared = tmp_path / "shared" / "bin" / "python3.10"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"pinned interpreter\n")
    shared.chmod(0o700)
    model_entry = tmp_path / "model" / "bin" / "python"
    model_entry.parent.mkdir(parents=True)
    model_entry.symlink_to(shared)
    cache_entry = tmp_path / "cache" / "bin" / "python"
    cache_entry.parent.mkdir(parents=True)
    cache_entry.symlink_to(model_entry)

    identity = cache_runtime.python_entry_identity(cache_entry)

    assert identity["entry_path"] == str(cache_entry)
    assert identity["entry_is_symlink"] is True
    assert identity["symlink_chain"] == [
        {"path": str(cache_entry), "link_target": str(model_entry)},
        {"path": str(model_entry), "link_target": str(shared)},
    ]
    assert identity["resolved_executable"]["path"] == str(shared)
    # Regression: resolving before execution would select the model/shared
    # environment rather than the cache virtual environment.
    assert Path(identity["entry_path"]) != Path(identity["entry_path"]).resolve()


def test_main_python_runtime_preserves_lexical_venv_and_packages(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "main-runtime"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    entry = prefix / "bin" / "python"
    assert entry.is_symlink()
    site_packages = next((prefix / "lib").glob("python*/site-packages"))
    for distribution, (import_name, version) in (
        stage.MAIN_PYTHON_RUNTIME_DISTRIBUTIONS.items()
    ):
        module = site_packages / import_name / "__init__.py"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(f"# isolated {distribution} {version}\n")
        dist_info = site_packages / f"{distribution}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
        )
        (dist_info / "top_level.txt").write_text(f"{import_name}\n")
        (dist_info / "RECORD").write_text(
            f"{import_name}/__init__.py,,\n"
            f"{dist_info.name}/METADATA,,\n"
            f"{dist_info.name}/top_level.txt,,\n"
            f"{dist_info.name}/RECORD,,\n"
        )

    receipt = stage._collect_main_python_runtime(entry)

    assert receipt["python_entry"]["entry_path"] == str(entry)
    assert receipt["python"]["sys_executable"] == str(entry)
    assert receipt["python"]["sys_prefix"] == str(prefix)
    assert receipt["python"]["no_user_site"] is True
    assert receipt["site_packages"] == [str(site_packages)]
    assert set(receipt["packages"]) == set(
        stage.MAIN_PYTHON_RUNTIME_DISTRIBUTIONS
    )
    assert stage.validate_main_python_runtime_receipt(receipt) == receipt

    # The old implementation resolved the symlink before subprocess launch.
    # The base interpreter may have an unrelated global LPIPS, but it cannot
    # see this venv's deliberately isolated package or prefix.
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    resolved = subprocess.run(
        [
            str(entry.resolve(strict=True)),
            "-c",
            (
                "import importlib.util,json,sys;"
                "s=importlib.util.find_spec('lpips');"
                "print(json.dumps({'prefix':sys.prefix,'origin':"
                "None if s is None else s.origin}))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    base_probe = json.loads(resolved.stdout)
    assert base_probe["prefix"] != str(prefix)
    assert str(site_packages) not in str(base_probe["origin"])

    pyvenv = prefix / "pyvenv.cfg"
    original_pyvenv = pyvenv.read_bytes()
    pyvenv.write_text("mutated runtime\n")
    with pytest.raises(stage.PhysicsFlowStage1Error):
        stage.validate_main_python_runtime_receipt(receipt)
    pyvenv.write_bytes(original_pyvenv)
    lpips_record = Path(
        receipt["packages"]["lpips"]["distribution_record"]["path"]
    )
    lpips_record.write_text(lpips_record.read_text() + "mutated,,\n")
    with pytest.raises(stage.PhysicsFlowStage1Error):
        stage.validate_main_python_runtime_receipt(receipt)


def test_compute_main_runtime_preflight_orders_venv_before_lpips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = tmp_path / "main" / "bin" / "python"
    source = tmp_path / "source"
    frozen_log = tmp_path / "lpips-preflight.log"
    calls: list[tuple[str, Path]] = []
    main_receipt = {"identity_sha256": "a" * 64}
    lpips_receipt = {
        "identity_sha256": stage.LPIPS_RECEIPT_IDENTITY,
        "loaded_state_dict_sha256": stage.LPIPS_STATE_DICT_SHA256,
    }
    log_receipt = {"path": str(frozen_log), "bytes": 6999, "sha256": "b" * 64}

    def collect(path: Path):
        calls.append(("venv", path))
        return main_receipt

    def lpips(path: Path, repo: Path, log: Path):
        calls.append(("lpips", path))
        assert repo == source
        assert log == frozen_log
        return lpips_receipt, log_receipt

    monkeypatch.setattr(stage, "_collect_main_python_runtime", collect)
    monkeypatch.setattr(stage, "_validated_offline_lpips", lpips)

    receipt = stage._main_runtime_preflight(entry, source, frozen_log)

    assert calls == [("venv", entry), ("lpips", entry)]
    assert receipt["kind"] == stage.MAIN_RUNTIME_PREFLIGHT_KIND
    assert receipt["python"] == str(entry)
    assert receipt["python_runtime"] == main_receipt
    assert receipt["lpips_alex"] == lpips_receipt
    assert receipt["network_access_permitted"] is False
    assert stage.identity_valid(receipt)


def test_cache_runtime_declares_complete_mcap_decode_stack() -> None:
    assert stage.CACHE_RUNTIME_CALIBRATION_PREFLIGHT_TRAIN_INDEX == 1
    assert stage.CACHE_RUNTIME_CALIBRATION_PREFLIGHT_IDENTITY == (
        "abc8d36380d88196e2cde46b24a57c52994fc6b518af028b89f75fffc1ee346c"
    )
    assert stage.CACHE_RUNTIME_DISTRIBUTIONS == {
        "lz4": "4.4.5",
        "mcap": "1.4.0",
        "mcap-protobuf-support": "0.5.4",
        "mujoco": "3.3.7",
        "numpy": "2.0.1",
        "protobuf": "7.35.1",
        "zstandard": "0.25.0",
    }
    assert stage.CACHE_RUNTIME_IMPORT_NAMES == {
        "lz4": "lz4",
        "mcap": "mcap",
        "mcap-protobuf-support": "mcap_protobuf",
        "mujoco": "mujoco",
        "numpy": "numpy",
        "protobuf": "google.protobuf",
        "zstandard": "zstandard",
    }


def test_calibration_preflight_requires_registered_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = {
        "camera_type": stage.CAMERA_TYPE,
        "camera_width": 640,
        "camera_height": 480,
        "distortion_model": "plumb_bob",
        "frame_id": "top_camera",
        "K": [1.0] * 9,
        "D": [0.0] * 5,
        "R": [1.0] * 9,
        "P": [1.0] * 12,
        "topics": ["/top-camera", "/top-camera-info"],
        "episode_metadata": {"top_camera_type": stage.CAMERA_TYPE},
    }
    monkeypatch.setattr(
        abc_probe,
        "_decode_top_calibration",
        lambda _: calibration,
    )
    raw_mcap = tmp_path / "episode.mcap"
    raw_mcap.write_bytes(b"zstd fixture")
    identity = hashlib.sha256(
        cache_runtime.canonical_json(calibration)
    ).hexdigest()

    receipt = cache_runtime._calibration_preflight(raw_mcap, identity)

    assert receipt["passed"] is True
    assert receipt["calibration_identity_sha256"] == identity
    assert receipt["future_rgb_message_decoded"] is False
    with pytest.raises(cache_runtime.CacheRuntimeError):
        cache_runtime._calibration_preflight(raw_mcap, "0" * 64)


def test_cache_runtime_receipt_rejects_symlink_or_record_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage,
        "CACHE_RUNTIME_RECORD_AMBIGUITIES",
        {name: [] for name in stage.CACHE_RUNTIME_DISTRIBUTIONS},
    )
    monkeypatch.setattr(
        stage,
        "CACHE_RUNTIME_RECORD_INVENTORIES",
        {name: {} for name in stage.CACHE_RUNTIME_DISTRIBUTIONS},
    )
    shared = tmp_path / "shared" / "python3.10"
    shared.parent.mkdir()
    shared.write_bytes(b"interpreter\n")
    shared.chmod(0o700)
    prefix = tmp_path / "cache"
    entry = prefix / "bin" / "python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(shared)
    pyvenv = prefix / "pyvenv.cfg"
    pyvenv.write_text("version = 3.10.20\n")
    site_packages = prefix / "lib" / "python3.10" / "site-packages"
    site_packages.mkdir(parents=True)
    packages = {}
    native_files = {}
    for name, version in stage.CACHE_RUNTIME_DISTRIBUTIONS.items():
        module = site_packages / name / "__init__.py"
        module.parent.mkdir()
        module_bytes = f"# {name} {version}\n".encode()
        module.write_bytes(module_bytes)
        native = module.parent / f"_{name}.so"
        native_bytes = f"native {name} {version}\n".encode()
        native.write_bytes(native_bytes)
        native_files[name] = native
        record = site_packages / f"{name}-{version}.dist-info" / "RECORD"
        record.parent.mkdir()
        rows = []
        for relative, content in (
            (f"{name}/__init__.py", module_bytes),
            (f"{name}/_{name}.so", native_bytes),
        ):
            encoded = (
                base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                .decode()
                .rstrip("=")
            )
            rows.append(f"{relative},sha256={encoded},{len(content)}")
        rows.append(f"{name}-{version}.dist-info/RECORD,,")
        record.write_text("\n".join(rows) + "\n")
        packages[name] = {
            "distribution": name,
            "import_name": stage.CACHE_RUNTIME_IMPORT_NAMES[name],
            "version": version,
            "module": stage.file_record(module),
            "distribution_record": stage.file_record(record),
            "verified_record_inventory": cache_runtime.verify_distribution_record(
                record, prefix
            ),
        }
    helper = tmp_path / "physics_flow_cache_runtime.py"
    helper.write_text("# receipt helper\n")
    raw_mcap = tmp_path / "episode.mcap"
    raw_mcap.write_bytes(b"registered calibration fixture\n")
    raw_mcap_stat = raw_mcap.stat()
    receipt = stage.identity_payload(
        {
            "schema_version": stage.CACHE_RENDERER_RUNTIME_SCHEMA_VERSION,
            "kind": stage.CACHE_RENDERER_RUNTIME_KIND,
            "python_entry": cache_runtime.python_entry_identity(entry),
            "pyvenv_cfg": stage.file_record(pyvenv),
            "python": {
                "sys_executable": str(entry),
                "sys_prefix": str(prefix),
                "sys_base_prefix": str(shared.parent.parent),
                "version": "3.10.20",
                "platform": "test",
                "no_user_site": True,
            },
            "packages": packages,
            "renderer_preflight": {
                "passed": True,
                "headless_backend": "egl",
                "render_shape": [8, 8, 3],
                "render_dtype": "uint8",
                "mujoco_library_version": "3.3.7",
            },
            "calibration_preflight": {
                "passed": True,
                "raw_mcap": {
                    "path": str(raw_mcap),
                    "bytes": raw_mcap_stat.st_size,
                    "mtime_ns": raw_mcap_stat.st_mtime_ns,
                    "device": raw_mcap_stat.st_dev,
                    "inode": raw_mcap_stat.st_ino,
                    "content_digest_computed": False,
                },
                "calibration_identity_sha256": "a" * 64,
                "camera_type": stage.CAMERA_TYPE,
                "camera_width": 640,
                "camera_height": 480,
                "intrinsic_count": 9,
                "topics": ["/top-camera", "/top-camera-info"],
                "registered_d405_calibration_decoded": True,
                "future_rgb_message_decoded": False,
                "protected_test_accessed": False,
            },
            "helper_source": stage.file_record(helper),
            "protected_test_accessed": False,
        }
    )
    assert stage.validate_cache_renderer_runtime_receipt(receipt) == receipt

    # The RECORD itself remains bit-identical; mutation of a declared native
    # library must still fail the static replay.
    native_files["mujoco"].write_bytes(b"mutated native library\n")
    with pytest.raises(stage.PhysicsFlowStage1Error):
        stage.validate_cache_renderer_runtime_receipt(receipt)


def test_record_verifier_only_allows_content_bound_duplicate_pyc(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "venv"
    site_packages = prefix / "lib" / "python3.10" / "site-packages"
    bytecode = site_packages / "demo" / "__pycache__" / "module.pyc"
    bytecode.parent.mkdir(parents=True)
    observed = b"observed generated bytecode"
    declared = b"wheel build bytecode"
    bytecode.write_bytes(observed)
    source = site_packages / "demo" / "__init__.py"
    source_bytes = b"# demo\n"
    source.write_bytes(source_bytes)
    record = site_packages / "demo-1.0.dist-info" / "RECORD"
    record.parent.mkdir()
    encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(declared).digest())
        .decode()
        .rstrip("=")
    )
    source_encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(source_bytes).digest())
        .decode()
        .rstrip("=")
    )
    record.write_text(
        f"demo/__init__.py,sha256={source_encoded},{len(source_bytes)}\n"
        "demo/__pycache__/module.pyc,,\n"
        f"demo/__pycache__/module.pyc,sha256={encoded},{len(declared)}\n"
        "demo-1.0.dist-info/RECORD,,\n"
    )

    inventory = cache_runtime.verify_distribution_record(record, prefix)

    assert inventory["duplicate_entry_count"] == 1
    assert inventory["ambiguous_duplicate_entries"] == [
        {
            "path": "demo/__pycache__/module.pyc",
            "bytes": len(observed),
            "sha256": hashlib.sha256(observed).hexdigest(),
            "declared_sha256": hashlib.sha256(declared).hexdigest(),
            "declared_bytes": len(declared),
            "duplicate_unhashed_declaration": True,
        }
    ]

    unique = site_packages / "demo" / "native.so"
    unique.write_bytes(b"mutated")
    record.write_text(
        record.read_text()
        + f"demo/native.so,sha256={encoded},{len(declared)}\n"
    )
    with pytest.raises(cache_runtime.CacheRuntimeError):
        cache_runtime.verify_distribution_record(record, prefix)


def test_support_weighting_does_not_attenuate_motion() -> None:
    native = np.zeros((4, 180, 320), dtype=np.float32)
    native[0, 0:4, 0:4] = 0.4
    native[1, 0:4, 0:4] = -0.2
    native[2, 0:4, 0:4] = 1.0
    native[3, 0:4, 0:4] = 0.1
    pooled = stage.pool_transition_field(native)
    assert pooled[2, 0, 0] == 0.25
    assert pooled[0, 0, 0] == np.float32(0.4)
    assert pooled[1, 0, 0] == np.float32(-0.2)
    assert pooled[3, 0, 0] == np.float32(0.1)
    assert np.count_nonzero(pooled[:, 1:, :]) == 0


def test_compact_control_shift_is_nonwrapping() -> None:
    value = np.zeros((2, 8, 4, 24, 40), dtype=np.float16)
    for transition in range(8):
        value[:, transition, 2] = 1
        value[:, transition, 0] = transition / 10
    shifted = stage.nonwrapping_timeshift(value)
    np.testing.assert_array_equal(shifted[:, :-1], value[:, 1:])
    assert np.count_nonzero(shifted[:, -1]) == 0
    stage.validate_compact_flow(value, count=2)
    stage.validate_compact_flow(shifted, count=2)


def test_cluster_bootstrap_keeps_noise_seeds_inside_episode() -> None:
    candidate = []
    reference = []
    for clip in range(6):
        for seed in stage.NOISE_SEEDS:
            candidate.append(
                {
                    "clip_index": clip,
                    "noise_seed_id": seed,
                    "metrics": {"metric": 0.8 + clip / 1000 + seed / 10000},
                }
            )
            reference.append(
                {
                    "clip_index": clip,
                    "noise_seed_id": seed,
                    "metrics": {"metric": 1.0 + clip / 1000 + seed / 10000},
                }
            )
    effect = stage._paired_cluster_effect(candidate, reference, "metric")
    assert effect["paired_episode_clusters"] == 6
    assert effect["noise_seeds_per_cluster"] == 4
    assert effect["relative_improvement_percent"] > 19
    assert effect["paired_episode_cluster_bootstrap_95_ci_percent"][0] > 0


def test_target_blind_evaluation_flags_fail_closed() -> None:
    valid = {
        "all_endpoints_materialized_before_future_rgb_open": True,
        "future_rgb_sampler_input": False,
        "future_measured_state_sampler_input": False,
        "clean_video_latent_sampler_input": False,
        "protected_test_accessed": False,
    }
    stage.validate_evaluation_causal_flags(valid, label="test")
    for field, invalid in (
        ("clean_video_latent_sampler_input", True),
        ("all_endpoints_materialized_before_future_rgb_open", False),
    ):
        changed = dict(valid)
        changed[field] = invalid
        with pytest.raises(stage.PhysicsFlowStage1Error):
            stage.validate_evaluation_causal_flags(changed, label="test")
    missing = dict(valid)
    missing.pop("clean_video_latent_sampler_input")
    with pytest.raises(stage.PhysicsFlowStage1Error):
        stage.validate_evaluation_causal_flags(missing, label="test")


def test_protocol_and_launcher_have_causal_guards() -> None:
    protocol = (
        ROOT / "docs/experiments/PHYSICS_FLOW_WAN_SCREEN_PROTOCOL.md"
    ).read_text()
    launcher = (ROOT / "tools/slurm/physics_flow_stage1.sbatch").read_text()
    evaluator = (ROOT / "tools/physics_flow_stage1_evaluate.py").read_text()
    for token in (
        "episode_shuffled",
        "timeshift_plus_one",
        "hold_current",
        "future measured state",
        "ADVANCE_RAW_FLOW_SCAFFOLD",
        "04f5013b7161fbf91ed6116d25f7e6ec66afc661024236ad27564b1899cb94be",
        "MEASURED_GEOMETRY_ORACLE",
        "no claim that val64 is globally",
        "separate Python process",
        "preserve_zero_support",
        stage.RENDERER_REGISTRATION_IDENTITY,
    ):
        assert token in protocol
    assert "--no-requeue" in launcher
    assert "#SBATCH --time=02:00:00" in launcher
    assert "04:00:00" not in launcher
    assert "compare-traces" in launcher
    assert "physics_flow_parent_parity.py" in launcher
    assert launcher.index('"$PYTHON_BIN" "$PARENT_PARITY"') < launcher.rindex(
        '"$PYTHON_BIN" -m torch.distributed.run'
    )
    assert "every endpoint for every registered" in evaluator
    assert '"clean_video_latent_sampler_input": False' in evaluator
    assert evaluator.index("materialized_by_seed.append") < evaluator.index(
        "dataset.scoring_batch"
    )
    lpips_helper = (ROOT / "tools/physics_flow_lpips.py").read_text()
    assert "offline AlexNet checkpoint is absent" in lpips_helper
    assert "network access is forbidden" in lpips_helper
    assert "loaded_state_dict_sha256" in lpips_helper
    parent_adapter = (ROOT / "tools/physics_flow_parent_vpm.py").read_text()
    parent_parity = (ROOT / "tools/physics_flow_parent_parity.py").read_text()
    parent_reference = (
        ROOT / "tools/physics_flow_parent_reference.py"
    ).read_text()
    launch_runbook = (
        ROOT / "docs/experiments/PHYSICS_FLOW_STAGE1_LAUNCH_RUNBOOK.md"
    ).read_text()
    stage_source = (ROOT / "tools/physics_flow_stage1.py").read_text()
    cache_runtime_source = (
        ROOT / "tools/physics_flow_cache_runtime.py"
    ).read_text()
    corrected_source = (
        ROOT / "tools/corrected_renderer_attribution.py"
    ).read_text()
    assert "model.sample_future_deployable(" in parent_adapter
    assert "model.sample_future_deployable(" in parent_parity
    assert "rgb[index, 0:5]" in parent_parity
    assert "rgb[index, 5" not in parent_parity
    assert "bitwise_parity_required" in parent_parity
    assert "physics_flow_stage1" not in parent_reference
    assert "implementation_repo" in parent_reference
    assert "historical_preserve_zero_support_attribute_absent" in parent_reference
    corrected_preamble = corrected_source.split("def extract_rows", 1)[0]
    assert "stage0_nominal_tracking_residual" not in corrected_preamble
    assert "stage0_nominal_tracking_residual" in corrected_source.split(
        "def extract_rows", 1
    )[1]
    assert "prepared only; do not execute" in launch_runbook
    assert "REPLACE_WITH_AUDITOR_ACKNOWLEDGED_40_CHARACTER_COMMIT" in launch_runbook
    assert 'BASH_PREFIX="/bin/bash -lc' in launch_runbook
    assert "CACHE_PYTHON_BIN=$BASE/envs/interaction-event-py310-v1/bin/python" in launch_runbook
    assert "PYTHONNOUSERSITE=1" in launch_runbook
    assert "--cache-python $CACHE_PYTHON_BIN" in launch_runbook
    assert launch_runbook.count("-20260808-$SHORT-v4") == 3
    assert "-20260808-$SHORT-v3" not in launch_runbook
    assert "-20260808-$SHORT-v2" not in launch_runbook
    assert "-20260808-$SHORT-v1" not in launch_runbook
    assert launch_runbook.count(
        "$CACHE_PYTHON_BIN tools/physics_flow_stage1.py build-cache"
    ) == 2
    assert "$CACHE_PYTHON_BIN tools/physics_flow_stage1.py audit-cache" not in launch_runbook
    assert "p._collect_main_python_runtime" in launch_runbook
    assert "args.python.expanduser().resolve" not in stage_source
    assert "python = lexical_absolute(python_entry)" in stage_source
    assert '"python_runtime": main_python_runtime' in stage_source
    login_preflight = launch_runbook.split("## 2. Freeze", 1)[1].split(
        "## 3. Submit", 1
    )[0]
    assert "physics_flow_lpips.py" not in login_preflight
    register_wrapper = launch_runbook.split("REGISTER_JOB=", 1)[1].split(
        "CACHE_TRAIN_JOB=", 1
    )[0]
    assert register_wrapper.index("preflight-main-runtime") < register_wrapper.index(
        "register-cache"
    )
    assert "--lpips-preflight-log $LPIPS_PREFLIGHT_LOG" in register_wrapper
    compute_preflight = stage_source.split(
        "def _main_runtime_preflight", 1
    )[1].split("def command_preflight_main_runtime", 1)[0]
    assert compute_preflight.index("_collect_main_python_runtime") < (
        compute_preflight.index("_validated_offline_lpips")
    )
    assert '"compute_preflight_passed_before_cache_output"' in stage_source
    register_body = stage_source.split("def command_register_cache", 1)[1].split(
        "def validate_cache_registration", 1
    )[0]
    assert register_body.index("_register_cache_renderer_runtime") < register_body.index(
        "output.mkdir(mode=0o700)"
    )
    assert register_body.index("train = _scan_split") < register_body.index(
        "_register_cache_renderer_runtime"
    )
    assert "--calibration-mcap" in stage_source
    assert "expected_calibration_identity_sha256" in stage_source
    for token in (
        '"mcap-protobuf-support": "mcap_protobuf"',
        '"protobuf": "google.protobuf"',
        '"lz4": "lz4"',
        '"zstandard": "zstandard"',
        "_calibration_preflight(",
    ):
        assert token in cache_runtime_source
    assert '"tools/abc_d405_nominal_geometry_probe.py"' in stage_source
    assert '"tools/corrected_renderer_attribution.py"' in stage_source
    build_body = stage_source.split("def command_build_cache", 1)[1].split(
        "def load_cache_metadata", 1
    )[0]
    assert build_body.index("enforce_current_cache_renderer_runtime") < build_body.index(
        "split_root.mkdir(mode=0o700)"
    )
    assert 'mkdir -p "$(dirname "$CACHE_ROOT")"' in launch_runbook
    assert "p.validate_renderer_gate" in launch_runbook
    assert launch_runbook.count("--gpus-per-node=1") == 4
    assert "04:00:00" not in launch_runbook
    assert launch_runbook.count("--time=02:00:00") == 4
    assert "--parent-source-repo $PARENT_SOURCE_REPO" in launch_runbook
    assert "--dependency=afterok:$FLOW_OFF_JOB:$RAW_FLOW_JOB" in launch_runbook
