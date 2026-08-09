from __future__ import annotations

import base64
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

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
    assert "np.load(state_path" not in source
    assert '"all_array_data_reads_within_selected_ranges": True' in source
    assert stage.SCHEMA_VERSION == 3
    assert stage.CACHE_SCHEMA == "raw-physics-flow-cache-v3"
    assert stage.STATE_ACCESS_SCHEMA == "raw-physics-flow-selected-npy-bytes-v1"
    assert stage.EVALUATION_NOISE_SEED == 20260729
    assert {arm.run_name for arm in stage.ARMS} == {
        "physics-flow-off-strict-v7-seed1234-u000200",
        "physics-flow-raw-strict-v7-seed1234-u000200",
    }
    dataset_source = (
        ROOT / "robot_wm/datasets/abc/physics_flow_dataset.py"
    ).read_text()
    trainer_source = (
        ROOT / "robot_wm/utils/physics_flow_trainer.py"
    ).read_text()
    for consumer in (dataset_source, trainer_source):
        assert '"raw-physics-flow-cache-v3"' in consumer
        assert '"raw-physics-flow-cache-v2"' not in consumer
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


def _state_archive_arrays(frame_count: int = 80) -> dict[str, np.ndarray]:
    return {
        "joint_states": np.arange(frame_count * 12, dtype=np.float32).reshape(
            frame_count, 12
        ),
        "joint_actions": (
            np.arange(frame_count * 12, dtype=np.float32).reshape(frame_count, 12)
            / 100
        ),
        "gripper_states": (
            np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2)
            / 10
        ),
        "gripper_actions": (
            np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2)
            / 100
        ),
        "frame_ts": np.arange(frame_count, dtype=np.int64),
        "instruction": np.asarray("strict byte fixture"),
    }


def _write_state_archive(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    compressed: bool = False,
) -> None:
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **arrays)


def _first_local_crc32(path: Path) -> int:
    with path.open("rb") as handle:
        handle.seek(14)
        value = handle.read(4)
    assert len(value) == 4
    return int(struct.unpack("<L", value)[0])


def _rewrite_numpy_force_zip64_as_legacy_redundant_extra(path: Path) -> None:
    """Emulate the NumPy layout used by the immutable ABC state archives.

    Current NumPy writes ZIP64 sentinels/version 45 in each local header.
    Older NumPy wrote the same 20-byte ZIP64 size extra redundantly beside
    ordinary 32-bit sizes/version 20; its central directory also uses version
    20 and no extra.  Rewriting only metadata preserves every NPY payload byte.
    """

    content = bytearray(path.read_bytes())
    local = stage._ZIP_LOCAL_HEADER
    offset = 0
    for expected_name in stage.STATE_ARCHIVE_MEMBERS:
        fields = list(local.unpack_from(content, offset))
        assert fields[0] == b"PK\x03\x04"
        assert fields[1] == 45
        assert fields[7] == fields[8] == 0xFFFFFFFF
        name_bytes, extra_bytes = fields[9], fields[10]
        variable_start = offset + local.size
        variable = content[
            variable_start : variable_start + name_bytes + extra_bytes
        ]
        assert bytes(variable[:name_bytes]).decode("ascii") == expected_name
        extra = bytes(variable[name_bytes:])
        payload_bytes, compressed_bytes = stage._parse_zip64_local_sizes(
            extra, expected_name
        )
        assert payload_bytes == compressed_bytes
        struct.pack_into("<H", content, offset + 4, 20)
        struct.pack_into("<L", content, offset + 18, compressed_bytes)
        struct.pack_into("<L", content, offset + 22, payload_bytes)
        offset = variable_start + name_bytes + extra_bytes + payload_bytes

    eocd_offset = len(content) - stage._ZIP_EOCD.size
    eocd = stage._ZIP_EOCD.unpack_from(content, eocd_offset)
    assert eocd[0] == b"PK\x05\x06"
    central_cursor = int(eocd[6])
    assert central_cursor == offset
    for expected_name in stage.STATE_ARCHIVE_MEMBERS:
        fields = stage._ZIP_CENTRAL_HEADER.unpack_from(content, central_cursor)
        assert fields[0] == b"PK\x01\x02"
        name_bytes, extra_bytes, comment_bytes = fields[10:13]
        name_start = central_cursor + stage._ZIP_CENTRAL_HEADER.size
        assert bytes(content[name_start : name_start + name_bytes]).decode(
            "ascii"
        ) == expected_name
        assert extra_bytes == comment_bytes == 0
        struct.pack_into("<H", content, central_cursor + 6, 20)
        central_cursor += (
            stage._ZIP_CENTRAL_HEADER.size
            + name_bytes
            + extra_bytes
            + comment_bytes
        )
    assert central_cursor == eocd_offset
    path.write_bytes(content)


def test_legacy_numpy_redundant_zip64_local_extra_is_strictly_addressable(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "states.npz"
    arrays = _state_archive_arrays()
    _write_state_archive(state_path, arrays)
    _rewrite_numpy_force_zip64_as_legacy_redundant_extra(state_path)

    physical_reads: list[dict[str, object]] = []
    observed = stage._read_selected_state_action_bytes(
        state_path,
        frame4=20,
        action_start=0,
        action_stop=65,
        read_observer=physical_reads.append,
    )

    np.testing.assert_array_equal(
        observed[0],
        np.concatenate(
            (arrays["joint_states"][20], arrays["gripper_states"][20])
        ),
    )
    assert observed[1].shape == (65, 14)
    assert observed[2]["array_byte_access_audit"][
        "future_measured_state_array_data_bytes_returned"
    ] == 0
    assert {
        member["zip_local_size_encoding"]
        for member in observed[2]["members"].values()
    } == {"zip64_redundant_sizes"}
    assert {
        member["zip_local_extra_bytes"]
        for member in observed[2]["members"].values()
    } == {20}
    assert sum(
        int(record["returned_bytes"])
        for record in physical_reads
        if str(record["label"]).startswith("selected_array_data")
    ) == (14 + 65 * 14) * 4


def test_redundant_zip64_local_extra_size_mismatch_fails_before_payload(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "states.npz"
    _write_state_archive(state_path, _state_archive_arrays())
    _rewrite_numpy_force_zip64_as_legacy_redundant_extra(state_path)
    content = bytearray(state_path.read_bytes())
    # First local extra begins after the fixed header and member name. Corrupt
    # only its redundant uncompressed size; no member payload is touched.
    name_bytes = stage._ZIP_LOCAL_HEADER.unpack_from(content, 0)[9]
    extra_payload_offset = stage._ZIP_LOCAL_HEADER.size + name_bytes + 4
    original = struct.unpack_from("<Q", content, extra_payload_offset)[0]
    struct.pack_into("<Q", content, extra_payload_offset, original + 1)
    state_path.write_bytes(content)
    physical_reads: list[dict[str, object]] = []

    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="inconsistent redundant ZIP64 sizes",
    ):
        stage._read_selected_state_action_bytes(
            state_path,
            frame4=20,
            action_start=0,
            action_stop=65,
            read_observer=physical_reads.append,
        )

    assert physical_reads
    assert {record["label"] for record in physical_reads} == {
        "zip_local_header"
    }


def test_state_reader_is_future_byte_invariant_and_frame4_sensitive(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    state_path = episode / "states.npz"
    arrays = _state_archive_arrays()
    _write_state_archive(state_path, arrays)
    start = 0
    stop = stage.SAMPLE_SIZE * stage.CHUNK_SIZE
    frame_indices = list(range(0, stop, stage.CHUNK_SIZE))
    frame4 = frame_indices[4]
    raw_action = np.concatenate(
        (
            arrays["joint_actions"][start:stop],
            arrays["gripper_actions"][start:stop],
        ),
        axis=1,
    ).reshape(stage.SAMPLE_SIZE, stage.CHUNK_SIZE, stage.ACTION_DIM)
    cached_action = np.zeros(
        (stage.SAMPLE_SIZE, stage.CHUNK_SIZE, stage.PADDED_ACTION_DIM),
        dtype=np.float32,
    )
    cached_action[..., : stage.ACTION_DIM] = raw_action
    row = {
        "clip_id": "strict-fixture",
        "episode_dir": str(episode),
        "frame_indices": frame_indices,
        "start": start,
    }
    descriptor = {
        "action_row_sha256": stage.tensor_sha256(cached_action),
    }
    preflight = stage._preflight_state_archive_access(
        state_path,
        frame4=frame4,
        action_start=start,
        action_stop=stop,
    )
    assert preflight["preflight_access_audit"]["array_data_bytes_returned"] == 0
    assert all(
        value["api"] == "os.pread"
        for value in preflight["planned_direct_reads"].values()
    )

    q4, endpoints, provenance = stage._state_and_actions_for_row(
        row=row,
        descriptor=descriptor,
        cached_action=cached_action,
    )
    access = provenance["states_npz"]
    assert access["required_member_compression"] == "ZIP_STORED"
    assert access["compressed_members_supported"] is False
    assert access["array_byte_access_audit"] == {
        **access["array_byte_access_audit"],
        "future_measured_state_array_data_bytes_returned": 0,
        "unregistered_action_array_data_bytes_returned": 0,
        "unused_member_payload_bytes_returned": 0,
        "all_array_data_reads_within_selected_ranges": True,
    }
    assert access["array_byte_access_audit"][
        "selected_array_data_bytes_returned"
    ] == (14 + 65 * 14) * 4
    assert access["array_byte_access_audit"][
        "selected_observed_state_array_data_bytes_returned"
    ] == 56
    assert access["array_byte_access_audit"][
        "selected_registered_action_array_data_bytes_returned"
    ] == 3_640
    assert access["array_byte_access_audit"][
        "all_archive_read_calls_use_os_pread"
    ] is True
    assert access["registered_candidate_actions_are_inference_available"] is True
    assert len(access["selected_array_data_pread_ledger"]) == 4
    assert all(
        row["api"] == "os.pread"
        and row["requested_bytes"] == row["returned_bytes"]
        for row in access["selected_array_data_pread_ledger"]
    )
    assert access["whole_member_crc_content_validated"] is False
    assert access["local_central_crc_metadata_agree"] is True
    assert access["zip_crc_value_returned_in_receipt"] is False
    np.testing.assert_array_equal(
        q4,
        np.concatenate(
            (arrays["joint_states"][frame4], arrays["gripper_states"][frame4])
        ),
    )

    # Rewrite a structurally valid archive after mutating both future measured
    # state members.  This updates the whole-member CRC metadata too, proving
    # that neither the future values nor their CRC summaries enter the receipt.
    original_crc32 = _first_local_crc32(state_path)
    arrays["joint_states"][frame4 + 1, 0] = 123_456.0
    arrays["gripper_states"][frame4 + 1, 0] = 654_321.0
    _write_state_archive(state_path, arrays)
    assert _first_local_crc32(state_path) != original_crc32
    future_q4, future_endpoints, future_provenance = (
        stage._state_and_actions_for_row(
            row=row,
            descriptor=descriptor,
            cached_action=cached_action,
        )
    )
    np.testing.assert_array_equal(future_q4, q4)
    np.testing.assert_array_equal(future_endpoints, endpoints)
    assert future_provenance == provenance
    assert stage._preflight_state_archive_access(
        state_path,
        frame4=frame4,
        action_start=start,
        action_stop=stop,
    ) == preflight

    arrays["joint_states"][frame4, 0] = 777_777.0
    _write_state_archive(state_path, arrays)
    changed_q4, changed_endpoints, changed_provenance = (
        stage._state_and_actions_for_row(
            row=row,
            descriptor=descriptor,
            cached_action=cached_action,
        )
    )
    assert changed_q4[0] == np.float32(777_777.0)
    assert not np.array_equal(changed_q4, q4)
    np.testing.assert_array_equal(changed_endpoints, endpoints)
    assert changed_provenance != provenance
    assert (
        changed_provenance["observed_frame4_state_sha256"]
        != provenance["observed_frame4_state_sha256"]
    )


def test_registered_state_preflight_ledgers_every_row(tmp_path: Path) -> None:
    descriptors = []
    for index in range(2):
        episode = tmp_path / f"episode-{index}"
        episode.mkdir()
        _write_state_archive(episode / "states.npz", _state_archive_arrays())
        descriptors.append(
            {
                "clip_index": index,
                "clip_id": f"clip-{index}",
                "episode_dir": str(episode),
                "frame_indices": list(range(0, 65, 5)),
                "start": 0,
            }
        )

    receipt = stage._preflight_registered_state_access("train", descriptors)

    assert stage.identity_valid(receipt)
    assert receipt["row_count"] == 2
    assert len(receipt["rows"]) == 2
    assert receipt["all_members_zip_stored"] is True
    assert receipt["all_local_central_metadata_agree"] is True
    assert receipt["preflight_array_data_bytes_returned"] == 0
    for index, row in enumerate(receipt["rows"]):
        assert stage.identity_valid(row)
        assert row["clip_index"] == index
        assert row["preflight_array_data_bytes_returned"] == 0
        assert row["planned_selected_array_data_bytes"] == (14 + 65 * 14) * 4
        assert row["planned_observed_state_array_data_bytes"] == 56
        assert row["planned_registered_action_array_data_bytes"] == 3_640
        assert row["registered_candidate_actions_are_inference_available"] is True
        assert row["preflight"]["preflight_access_audit"][
            "all_archive_read_calls_use_os_pread"
        ] is True
        assert stage.identity_valid(row["preflight"])


def test_compressed_state_archive_fails_before_member_payload_read(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "states.npz"
    _write_state_archive(state_path, _state_archive_arrays(), compressed=True)
    physical_reads: list[dict[str, object]] = []

    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="compressed; strict selected-byte access requires ZIP_STORED",
    ):
        stage._read_selected_state_action_bytes(
            state_path,
            frame4=20,
            action_start=0,
            action_stop=65,
            read_observer=physical_reads.append,
        )

    assert physical_reads
    assert {record["label"] for record in physical_reads} == {
        "zip_local_header"
    }
    assert not any(
        str(record["label"]).startswith("selected_array_data")
        or record["label"] == "npy_header"
        for record in physical_reads
    )


def test_state_archive_structure_fails_closed(tmp_path: Path) -> None:
    valid = _state_archive_arrays()

    wrong_dtype = dict(valid)
    wrong_dtype["joint_states"] = wrong_dtype["joint_states"].astype(np.float64)
    wrong_dtype_path = tmp_path / "wrong-dtype.npz"
    _write_state_archive(wrong_dtype_path, wrong_dtype)
    with pytest.raises(stage.PhysicsFlowStage1Error, match="dtype/shape/C-order"):
        stage._read_selected_state_action_bytes(
            wrong_dtype_path, frame4=20, action_start=0, action_stop=65
        )

    fortran = dict(valid)
    fortran["joint_states"] = np.asfortranarray(fortran["joint_states"])
    fortran_path = tmp_path / "fortran.npz"
    _write_state_archive(fortran_path, fortran)
    with pytest.raises(stage.PhysicsFlowStage1Error, match="dtype/shape/C-order"):
        stage._read_selected_state_action_bytes(
            fortran_path, frame4=20, action_start=0, action_stop=65
        )

    wrong_order_path = tmp_path / "wrong-order.npz"
    wrong_order = {
        "joint_actions": valid["joint_actions"],
        "joint_states": valid["joint_states"],
        "gripper_states": valid["gripper_states"],
        "gripper_actions": valid["gripper_actions"],
        "frame_ts": valid["frame_ts"],
        "instruction": valid["instruction"],
    }
    _write_state_archive(wrong_order_path, wrong_order)
    with pytest.raises(stage.PhysicsFlowStage1Error, match="order/name differs"):
        stage._read_selected_state_action_bytes(
            wrong_order_path, frame4=20, action_start=0, action_stop=65
        )

    extra_member_path = tmp_path / "extra-member.npz"
    _write_state_archive(
        extra_member_path,
        {**valid, "future_state_summary": np.asarray([1], dtype=np.int64)},
    )
    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="central directory/count/trailing bytes differ",
    ):
        stage._read_selected_state_action_bytes(
            extra_member_path, frame4=20, action_start=0, action_stop=65
        )

    flagged_path = tmp_path / "data-descriptor-flag.npz"
    _write_state_archive(flagged_path, valid)
    with flagged_path.open("r+b") as handle:
        handle.seek(6)  # first local header general-purpose flags
        handle.write(struct.pack("<H", 0x08))
    with pytest.raises(stage.PhysicsFlowStage1Error, match="unsupported ZIP flags"):
        stage._read_selected_state_action_bytes(
            flagged_path, frame4=20, action_start=0, action_stop=65
        )

    crc_mismatch_path = tmp_path / "local-central-crc-mismatch.npz"
    _write_state_archive(crc_mismatch_path, valid)
    with crc_mismatch_path.open("r+b") as handle:
        handle.seek(14)  # first local header CRC32
        original = handle.read(4)
        handle.seek(14)
        handle.write(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="central/local ZIP metadata differs",
    ):
        stage._read_selected_state_action_bytes(
            crc_mismatch_path, frame4=20, action_start=0, action_stop=65
        )


@pytest.mark.parametrize("payload_delta", [b"trailing", None])
def test_state_npy_member_rejects_trailing_or_premature_bytes(
    tmp_path: Path,
    payload_delta: bytes | None,
) -> None:
    arrays = _state_archive_arrays()
    state_path = tmp_path / f"member-{payload_delta is None}.npz"
    with zipfile.ZipFile(state_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in stage.STATE_ARCHIVE_MEMBERS:
            key = name[:-4]
            buffer = io.BytesIO()
            np.save(buffer, arrays[key], allow_pickle=False)
            payload = buffer.getvalue()
            if name == "joint_states.npy":
                payload = (
                    payload + payload_delta
                    if payload_delta is not None
                    else payload[:-1]
                )
            archive.writestr(name, payload)

    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="trailing or truncated NPY array bytes",
    ):
        stage._read_selected_state_action_bytes(
            state_path, frame4=20, action_start=0, action_stop=65
        )


def test_state_archive_rejects_trailing_and_truncated_archive_bytes(
    tmp_path: Path,
) -> None:
    arrays = _state_archive_arrays()
    trailing_path = tmp_path / "trailing-archive.npz"
    _write_state_archive(trailing_path, arrays)
    with trailing_path.open("ab") as handle:
        handle.write(b"forbidden trailing bytes")
    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="central directory/count/trailing bytes differ",
    ):
        stage._read_selected_state_action_bytes(
            trailing_path, frame4=20, action_start=0, action_stop=65
        )

    truncated_path = tmp_path / "truncated-archive.npz"
    _write_state_archive(truncated_path, arrays)
    truncated_path.write_bytes(truncated_path.read_bytes()[:-1])
    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="central directory/count/trailing bytes differ",
    ):
        stage._read_selected_state_action_bytes(
            truncated_path, frame4=20, action_start=0, action_stop=65
        )


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


def test_enriched_lineage_file_receipt_revalidates_only_file_fields(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "legacy-snapshot.pt"
    artifact.write_bytes(b"sealed legacy snapshot\n")
    enriched = {
        **stage.file_record(artifact),
        "run_identity_sha256": "a" * 64,
        "canonical_model_state_sha256": "b" * 64,
        "model_schema_sha256": "c" * 64,
        "rejection_reason": "separately validated semantic metadata",
    }

    assert stage._absolute_file_record_matches(
        enriched, "legacy parent"
    ) == artifact.resolve()

    for field, value in (
        ("bytes", enriched["bytes"] + 1),
        ("sha256", "0" * 64),
    ):
        changed = dict(enriched)
        changed[field] = value
        with pytest.raises(
            stage.PhysicsFlowStage1Error,
            match="legacy parent file differs",
        ):
            stage._absolute_file_record_matches(changed, "legacy parent")

    different = tmp_path / "different-snapshot.pt"
    different.write_bytes(b"different artifact\n")
    changed_path = {**enriched, "path": str(different)}
    with pytest.raises(
        stage.PhysicsFlowStage1Error,
        match="legacy parent file differs",
    ):
        stage._absolute_file_record_matches(changed_path, "legacy parent")


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


def test_v5_cache_equivalence_requires_values_equal_and_provenance_distinct() -> None:
    arrays = {
        source: np.asarray([[[float(index + 1)]]], dtype=np.float16)
        for index, source in enumerate(stage.CACHE_SOURCES)
    }
    tensor_hashes = {
        source: stage.tensor_sha256(value[0])
        for source, value in arrays.items()
    }
    shared_provenance = {
        "observed_frame4_state_sha256": "1" * 64,
        "registered_action_row_sha256": "2" * 64,
        "candidate_action_span_sha256": "3" * 64,
        "candidate_action_endpoints_sha256": "4" * 64,
        "cache_raw_action_max_abs": 0.0,
    }
    base = {
        "kind": "raw_physics_flow_cache_row",
        "split": "train",
        "clip_index": 0,
        "clip_id": "fixture",
        "episode_dir": "/fixture/episode",
        "d405_eligible": True,
        "frame_indices": list(range(13)),
        "action_window": [0, 65],
        "camera_type": "D405",
        "calibration": {"calibration_identity_sha256": "5" * 64},
        "motion_stratum": 1,
        "episode_shuffled_donor_index": 7,
        "episode_shuffled_donor_clip_id": "donor",
        "render_diagnostics": {"transitions": [{"support": 3}]},
        "hold_render_diagnostics": {"transitions": [{"support": 4}]},
        "raw_pose_path_sha256": "6" * 64,
        "tensor_sha256": tensor_hashes,
        "protected_test_accessed": False,
    }
    prior = stage.identity_payload(
        {
            **base,
            "schema_version": 2,
            "input_provenance": {
                **shared_provenance,
                "states_npz": {"legacy_full_member_materialization": True},
            },
        }
    )
    current = stage.identity_payload(
        {
            **base,
            "schema_version": 3,
            "input_provenance": {
                **shared_provenance,
                "states_npz": {
                    "state_access_schema": stage.STATE_ACCESS_SCHEMA,
                    "array_byte_access_audit": {
                        "selected_observed_state_array_data_bytes_returned": 56,
                        "selected_registered_action_array_data_bytes_returned": 3640,
                        "future_measured_state_array_data_bytes_returned": 0,
                        "unregistered_action_array_data_bytes_returned": 0,
                    },
                },
            },
        }
    )
    records = {
        source: {
            "path": f"/fixture/{source}.npy",
            "bytes": int(value.nbytes),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for source, value in arrays.items()
    }
    reference = {
        "count": 1,
        "metadata": {"path": "/fixture/metadata.json"},
        "metadata_identity_sha256": "7" * 64,
        "row_lineage": {"path": "/fixture/row_lineage.jsonl"},
        "arrays": records,
    }
    receipt = stage._v5_cache_numeric_equivalence(
        split="train",
        current_metadata={"arrays": records},
        current_rows=[current],
        reference=reference,
        reference_rows=[prior],
        current_arrays=arrays,
        reference_arrays={key: value.copy() for key, value in arrays.items()},
    )
    assert receipt["status"] == "bitwise_numeric_equivalence_passed"
    assert receipt["row_tensor_hashes_recomputed_per_version"] == 8
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["raw"][0, 0, 0] += np.float16(1)
    with pytest.raises(stage.PhysicsFlowStage1Error):
        stage._v5_cache_numeric_equivalence(
            split="train",
            current_metadata={"arrays": records},
            current_rows=[current],
            reference=reference,
            reference_rows=[prior],
            current_arrays=changed,
            reference_arrays=arrays,
        )


def test_same_arm_v5_training_equivalence_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_registration = {
        "path": str(tmp_path / "v5-registration.json"),
        "bytes": 1,
        "sha256": "8" * 64,
    }
    old_registration_identity = "9" * 64
    reference_arms = {}
    current = {}
    for arm in stage.ARMS:
        arm_root = tmp_path / arm.code.lower()
        arm_root.mkdir()
        common_header = {
            "kind": "physics_flow_training_trace_header",
            "arm": arm.code,
            "fuse_flow": arm.fuse_flow,
            "parent_snapshot": "/fixture/parent.pt",
            "parent_snapshot_sha256": stage.PARENT_SNAPSHOT_SHA256,
            "parent_run_identity_sha256": stage.PARENT_RUN_IDENTITY_SHA256,
            "continuation_updates": 200,
            "wan_calls_per_example": 1,
            "flow_model_calls_per_example": 0,
            "flow_clock": 0.0,
            "flow_velocity_loss": 0.0,
            "future_measured_state_conditioning": False,
            "future_rgb_conditioning": False,
            "optimizer_state_policy": "fresh_identical_adamw",
            "parameter_schema_sha256": "a" * 64,
            "initial_auxiliary_state_sha256": "b" * 64,
            "protected_test_accessed": False,
        }
        old_header = {
            **common_header,
            "study_registration": old_registration["path"],
            "study_registration_identity_sha256": old_registration_identity,
            "study_registration_sha256": old_registration["sha256"],
            "train_flow_metadata": "/fixture/v5/train/metadata.json",
            "train_flow_metadata_sha256": "c" * 64,
            "val_flow_metadata": "/fixture/v5/val/metadata.json",
            "val_flow_metadata_sha256": "d" * 64,
        }
        new_header = {
            **common_header,
            "study_registration": "/fixture/v6/registration.json",
            "study_registration_identity_sha256": "e" * 64,
            "study_registration_sha256": "f" * 64,
            "train_flow_metadata": "/fixture/v6/train/metadata.json",
            "train_flow_metadata_sha256": "0" * 64,
            "val_flow_metadata": "/fixture/v6/val/metadata.json",
            "val_flow_metadata_sha256": "1" * 64,
        }
        old_events = []
        new_events = {}
        for iteration in range(200):
            deterministic = {
                "iteration": iteration,
                "train_loss/loss": float(iteration) / 100,
                "system/world_size": 8,
            }
            old_metrics = {
                **deterministic,
                **{
                    key: float(iteration)
                    for key in stage.V5_TRAINING_METRIC_EXCLUSIONS
                },
            }
            new_metrics = {
                **deterministic,
                **{
                    key: float(iteration + 1)
                    for key in stage.V5_TRAINING_METRIC_EXCLUSIONS
                },
            }
            old_events.append(
                {
                    "kind": "physics_flow_training_trace_event",
                    "arm": arm.code,
                    "metrics": old_metrics,
                    "total_observations": (iteration + 1) * 8,
                }
            )
            new_events[iteration] = {
                "kind": "physics_flow_training_trace_event",
                "arm": arm.code,
                "metrics": new_metrics,
                "total_observations": (iteration + 1) * 8,
            }
        trace_path = arm_root / "v5-trace.jsonl"
        trace_path.write_bytes(
            b"".join(
                stage.canonical_json(row) + b"\n"
                for row in [old_header, *old_events]
            )
        )
        new_trace_path = arm_root / "v6-trace.jsonl"
        new_trace_path.write_bytes(
            b"".join(
                stage.canonical_json(row) + b"\n"
                for row in [new_header, *new_events.values()]
            )
        )
        old_completion = {
            "kind": "physics_flow_training_trace_complete",
            "arm": arm.code,
            "completed_updates": 200,
            "rows": 201,
            "trace_sha256": stage.sha256_file(trace_path),
            "protected_test_accessed": False,
        }
        old_completion_path = arm_root / "v5-complete.json"
        old_completion_path.write_bytes(
            stage.canonical_json(old_completion) + b"\n"
        )
        new_completion = {**old_completion, "trace_sha256": "2" * 64}
        new_completion_path = arm_root / "v6-complete.json"
        new_completion_path.write_bytes(
            stage.canonical_json(new_completion) + b"\n"
        )
        model = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        old_snapshot_path = arm_root / "v5-snapshot.pt"
        new_snapshot_path = arm_root / "v6-snapshot.pt"
        snapshot_common = {
            "snapshot_schema_version": 3,
            "_start_iter": 200,
            "world_size": 8,
            "gradient_accumulation_steps": 1,
            "model": model,
        }
        torch.save(
            {**snapshot_common, "run_identity_sha256": "3" * 64},
            old_snapshot_path,
        )
        torch.save(
            {**snapshot_common, "run_identity_sha256": "4" * 64},
            new_snapshot_path,
        )
        reference_arms[arm.code] = {
            "run_name": f"v5-{arm.code}",
            "trace": stage.file_record(trace_path),
            "completion": stage.file_record(old_completion_path),
            "snapshot": stage.file_record(old_snapshot_path),
        }
        current[arm.code] = (
            new_header,
            new_events,
            {
                "trace": stage.file_record(new_trace_path),
                "completion": stage.file_record(new_completion_path),
                "snapshot": stage.file_record(new_snapshot_path),
            },
        )
    reference = stage.identity_payload(
        {
            "root": str(tmp_path),
            "study_registration": old_registration,
            "study_registration_identity_sha256": old_registration_identity,
            "arms": reference_arms,
        }
    )
    registration = {"v5_training_reference": reference}
    monkeypatch.setattr(
        stage,
        "_validated_v5_study_reference",
        lambda _root, verify_snapshot_digests: reference,
    )
    receipt = stage._same_arm_v5_training_equivalence(registration, current)
    assert receipt["same_arm_updates_compared_exactly"] == 400
    assert receipt["model_tensor_hashes_compared_exactly"] == 2
    assert all(
        arm_receipt["all_per_tensor_hashes_exact"]
        for arm_receipt in receipt["arms"].values()
    )


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
    evaluator_source = (
        ROOT / "tools/physics_flow_stage1_evaluate.py"
    ).read_text()
    trainer_source = (
        ROOT / "projects/latent_action_models/physics_flow_train.py"
    ).read_text()
    model_config = (
        ROOT
        / "projects/latent_action_models/configs/models/physics_flow_model.yaml"
    ).read_text()
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
    assert launch_runbook.count("-20260808-$SHORT-v7") == 3
    assert "--v5-reference-cache-root $V5_CACHE_ROOT" in launch_runbook
    assert "--v5-reference-study-root $V5_STUDY_ROOT" in launch_runbook
    assert "-20260808-$SHORT-v6" not in launch_runbook
    assert "-20260808-$SHORT-v4" not in launch_runbook
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
    assert register_body.index("_preflight_registered_state_access") < (
        register_body.index("_register_cache_renderer_runtime")
    )
    assert register_body.index("_preflight_registered_state_access") < (
        register_body.index("output.mkdir(mode=0o700)")
    )
    validate_registration_body = stage_source.split(
        "def validate_cache_registration", 1
    )[1].split("def _state_and_actions_for_row", 1)[0]
    assert "_preflight_registered_state_access" in validate_registration_body
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
    assert "evaluation_noise_seed: 20260729" in model_config
    assert trainer_source.index("_validate_protocol_config(cfg)") < (
        trainer_source.index("dist.init_process_group()")
    )
    evaluator_config_gate = evaluator_source.index(
        "{model_seed, forward_seed} != {EXPECTED_EVALUATION_NOISE_SEED}"
    )
    assert evaluator_config_gate < evaluator_source.index(
        "model = instantiate(config.model)"
    )
    assert evaluator_source.index("stage.load_training_pairing(registration)") < (
        evaluator_source.index("dataset = RegisteredCausalInputs(registration)")
    )
    assert "model.evaluation_noise_seed != EXPECTED_EVALUATION_NOISE_SEED" in (
        evaluator_source
    )
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
    compare_index = launcher.index(
        '"$PYTHON_BIN" "$STAGE_TOOL" compare-traces'
    )
    parity_index = launcher.index('"$PYTHON_BIN" "$PARENT_PARITY"')
    evaluate_index = launcher.rindex(
        '"$PYTHON_BIN" -m torch.distributed.run'
    )
    assert compare_index < parity_index < evaluate_index
    assert "V5_TRAINING_METRIC_EXCLUSIONS" in stage_source
    assert "snapshot_pickle_bytes_compared" in stage_source
