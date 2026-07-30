from types import SimpleNamespace

import pytest

from tools import benchmark_vjepa2_frontier_latency as benchmark


def test_balanced_orders_cover_all_permutations_and_positions():
    orders = [benchmark.balanced_order(index) for index in range(12)]

    assert len(set(orders[:6])) == 6
    for endpoint in benchmark.ENDPOINT_LABELS:
        for position in range(3):
            assert sum(order[position] == endpoint for order in orders) == 4


def test_frontier_timing_gate_and_same_nfe_overhead_are_separable():
    orders = [benchmark.balanced_order(index) for index in range(120)]
    j1 = [5.0 + (index % 3) * 0.01 for index in range(120)]
    vpm_same = [4.5 + (index % 3) * 0.01 for index in range(120)]
    vpm_frontier = [10.0 + (index % 3) * 0.01 for index in range(120)]

    acceleration = benchmark.paired_timing_effect(
        j1,
        vpm_frontier,
        orders,
        left_label="J1_k",
        reference_label="VPM_m",
        bootstrap_samples=500,
        confidence=0.95,
        seed=1234,
        label="frontier",
    )
    overhead_speed_effect = benchmark.paired_timing_effect(
        j1,
        vpm_same,
        orders,
        left_label="J1_k",
        reference_label="VPM_k",
        bootstrap_samples=500,
        confidence=0.95,
        seed=1234,
        label="same",
    )
    gate = benchmark.timing_gate(
        acceleration,
        left_p95=benchmark.latency_summary(j1)["p95"],
        reference_p95=benchmark.latency_summary(vpm_frontier)["p95"],
    )

    assert gate["passed"] is True
    assert acceleration["n_counterbalance_blocks"] == 20
    assert acceleration["bootstrap_unit"].startswith(
        "complete six-round counterbalance block"
    )
    assert acceleration["bootstrap_ci"]["low"] > 0
    assert all(
        value["mean_favorable_difference_ms"] > 0
        for value in acceleration["order_strata"].values()
    )
    assert overhead_speed_effect["relative_improvement"] < 0


def test_timing_gate_fails_if_one_order_stratum_reverses():
    orders = [benchmark.balanced_order(index) for index in range(120)]
    j1 = []
    vpm = []
    for order in orders:
        j1_first = order.index("J1_k") < order.index("VPM_m")
        j1.append(12.0 if j1_first else 4.0)
        vpm.append(10.0)
    effect = benchmark.paired_timing_effect(
        j1,
        vpm,
        orders,
        left_label="J1_k",
        reference_label="VPM_m",
        bootstrap_samples=500,
        confidence=0.95,
        seed=1234,
        label="order-reversal",
    )
    gate = benchmark.timing_gate(
        effect,
        left_p95=12.0,
        reference_p95=10.0,
    )

    assert gate["passed"] is False
    assert gate["checks"]["both_execution_order_strata_favorable"] is False


def test_benchmark_command_rejects_underpowered_confirmatory_protocol():
    args = SimpleNamespace(
        warmup_rounds=6,
        timed_rounds=6,
        bootstrap_samples=100,
        confidence=0.51,
        seed=7,
    )

    with pytest.raises(
        benchmark.FrontierLatencyError, match="confirmatory timing requires"
    ):
        benchmark.command_benchmark(args)
