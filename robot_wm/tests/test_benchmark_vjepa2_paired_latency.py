import hashlib
import inspect

import pytest

from tools import benchmark_vjepa2_inference as single
from tools import benchmark_vjepa2_paired_latency as paired


def test_counterbalance_is_exact_and_deterministic():
    orders = [
        paired.counterbalanced_order(index)
        for index in range(paired.TIMED_PAIRS)
    ]
    assert orders[0] == ("J1", "VPM")
    assert orders[1] == ("VPM", "J1")
    assert sum(order[0] == "J1" for order in orders) == 50
    assert sum(order[0] == "VPM" for order in orders) == 50
    with pytest.raises(paired.PairedLatencyError):
        paired.counterbalanced_order(-1)


def test_latency_summary_binds_all_raw_values():
    values = [float(index + 1) for index in range(100)]
    result = paired._summary(values)
    assert result["count"] == 100
    assert result["p50"] == pytest.approx(50.5)
    assert result["p95"] == pytest.approx(95.05)
    assert result["values_sha256"] == hashlib.sha256(
        single._canonical_json([round(value, 9) for value in values])
    ).hexdigest()


def test_file_record_requires_exact_path_hash_and_size(tmp_path):
    snapshot = tmp_path / "snapshot.pt"
    snapshot.write_bytes(b"sealed checkpoint bytes")
    record = paired._record(snapshot)

    assert paired._file_record_matches(record, snapshot)
    assert not paired._file_record_matches(
        {key: value for key, value in record.items() if key != "bytes"},
        snapshot,
    )
    assert not paired._file_record_matches(
        {**record, "bytes": record["bytes"] + 1},
        snapshot,
    )
    assert not paired._file_record_matches(
        {**record, "unexpected": True},
        snapshot,
    )


def test_timed_implementation_uses_only_public_deployable_sampler():
    source = inspect.getsource(paired.command_benchmark)
    assert "sample_future_deployable(" in source
    assert "collect_artifacts=False" in source
    assert "torch.cuda.synchronize(device)" in source
    assert "._sample_future(" not in source
    assert "auxiliary_target=" not in source
