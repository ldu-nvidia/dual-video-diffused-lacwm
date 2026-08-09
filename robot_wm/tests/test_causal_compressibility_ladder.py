import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "causal_compressibility_ladder.py"
SPEC = importlib.util.spec_from_file_location("causal_compressibility_ladder", SCRIPT)
ladder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ladder)


def test_wan_patch_roundtrip_preserves_token_and_grid_order():
    value = torch.arange(2 * 3 * 4 * 6 * 10, dtype=torch.float32).reshape(2, 3, 4, 6, 10)
    patch_size = (1, 2, 2)
    tokens = ladder.patchify_video(value, patch_size)
    restored = ladder.unpatchify_tokens(
        tokens,
        grid=(4, 3, 5),
        patch_size=patch_size,
        channels=3,
    )
    assert tokens.shape == (2, 60, 12)
    assert torch.equal(restored, value)


def test_future_positions_exclude_history_and_subsample_is_keyed():
    positions = ladder.future_token_positions(
        grid=(4, 12, 60), patch_size=(1, 2, 2), history_frames=2
    )
    assert positions[0].item() == 2 * 12 * 60
    assert positions[-1].item() == 4 * 12 * 60 - 1
    first = ladder.deterministic_token_subsample(
        positions, clip_index=128, noise_seed=20260820, count=256
    )
    repeated = ladder.deterministic_token_subsample(
        positions, clip_index=128, noise_seed=20260820, count=256
    )
    changed = ladder.deterministic_token_subsample(
        positions, clip_index=128, noise_seed=20260821, count=256
    )
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert len(first.unique()) == 256


def test_partition_contract_is_episode_disjoint_and_rejects_overlap():
    rows = [
        {
            "split": "train",
            "auxiliary_index": index,
            "clip_id": f"clip-{index}",
            "episode_dir": f"episode-{index}",
        }
        for index in range(512)
    ]
    result = ladder.validate_partition_contract(rows)
    assert result["ranges"]["optimization"] == [128, 384]
    assert result["ranges"]["development"] == [416, 480]
    rows[416] = {**rows[416], "episode_dir": rows[128]["episode_dir"]}
    with pytest.raises(ladder.LadderError, match="episodes are not disjoint"):
        ladder.validate_partition_contract(rows)


def test_equal_capacity_ridge_recovers_linear_targets_and_zero_is_exact():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(2048, 64, generator=generator)
    truth = torch.randn(64, 8, generator=generator) * 0.1
    direct = x @ truth
    targets = {
        "ZERO": torch.zeros_like(direct),
        "DIRECT": direct,
        "PFD_ALIGNED": direct * 0.75,
        "PFD_SHUFFLED": torch.roll(direct, 1, 0),
    }
    stats = ladder._new_sufficient_statistics(64, 8, torch.device("cpu"))
    ladder.update_sufficient_statistics(stats, x, targets)
    old_rungs = ladder.CAPACITY_RUNGS
    try:
        ladder.CAPACITY_RUNGS = (64,)
        direct_head = ladder.fit_ridge_head(
            stats, arm="DIRECT", capacity=64, ridge_lambda=1e-4
        )
        zero_head = ladder.fit_ridge_head(
            stats, arm="ZERO", capacity=64, ridge_lambda=1e-4
        )
    finally:
        ladder.CAPACITY_RUNGS = old_rungs
    prediction = ladder.predict_ridge_head(x, direct_head)
    zero = ladder.predict_ridge_head(x, zero_head)
    assert direct_head["parameter_count"] == zero_head["parameter_count"]
    assert torch.mean((prediction - direct) ** 2) < 1e-5
    assert torch.count_nonzero(zero) == 0


def test_target_open_guard_records_rgb_actions_and_fails_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "target.npy"
        rgb = root / "rgb.npy"
        actions = root / "actions.npy"
        np.save(target, np.zeros((1,), dtype=np.float32))
        np.save(rgb, np.ones((1,), dtype=np.float32))
        np.save(actions, np.full((1,), 2, dtype=np.float32))
        with ladder.NumpyTargetOpenGuard(target) as guard:
            assert np.load(rgb)[0] == 1
            assert np.load(actions)[0] == 2
            with pytest.raises(ladder.LadderError, match="V-JEPA target array"):
                np.load(target)
        assert set(guard.opened) == {rgb.resolve(), actions.resolve()}


def test_batch_samples_accesses_each_requested_item_exactly_once():
    class CountingDataset:
        def __init__(self):
            self.counts = {}

        def __getitem__(self, index):
            self.counts[index] = self.counts.get(index, 0) + 1
            return {
                "clip_index": torch.tensor(index),
                "value": torch.tensor(float(index)),
            }

    dataset = CountingDataset()
    batch = ladder._batch_samples(dataset, (3, 7), torch.device("cpu"))
    assert dataset.counts == {3: 1, 7: 1}
    assert batch["clip_index"].tolist() == [3, 7]


def test_decoded_temporal_metric_includes_raw_history_boundary():
    prediction = torch.full((1, 3, 2, 1, 1), 100, dtype=torch.uint8)
    target = prediction.clone()
    prediction_history = torch.zeros((1, 3, 1, 1, 1), dtype=torch.uint8)
    target_history = torch.full((1, 3, 1, 1, 1), 100, dtype=torch.uint8)
    mse, temporal = ladder._decoded_metrics(
        prediction,
        target,
        prediction_history=prediction_history,
        target_history=target_history,
    )
    assert mse.item() == 0.0
    assert temporal.item() > 0.0


def test_decoded_endpoint_fails_closed_on_horizon_or_missing_boundary():
    assert ladder._validate_decoded_horizon(
        model_future_frames=8, decoded_frames=13, raw_frames=13, endpoint="J1"
    ) == 8
    with pytest.raises(ladder.LadderError, match="eight future frames"):
        ladder._validate_decoded_horizon(
            model_future_frames=7, decoded_frames=13, raw_frames=13, endpoint="J1"
        )
    with pytest.raises(ladder.LadderError, match="history-boundary"):
        ladder._validate_decoded_horizon(
            model_future_frames=8, decoded_frames=8, raw_frames=13, endpoint="VPM"
        )


def test_identity_detects_mutation_and_one_step_sign_is_noise_minus_velocity():
    payload = ladder.identity_payload({"value": 3})
    assert ladder.identity_valid(payload)
    payload["value"] = 4
    assert not ladder.identity_valid(payload)
    initial = torch.tensor([[[[[3.0]], [[5.0]]]]])
    velocity = torch.tensor([[[[[1.0]], [[2.0]]]]])
    reference = torch.tensor([[[[[9.0]], [[0.0]]]]])
    final = ladder._one_step_final(initial, velocity, reference, history_frames=1)
    assert torch.equal(final, torch.tensor([[[[[9.0]], [[3.0]]]]]))


def _row(endpoint: str, clip: int, seed: int, value: float):
    return ladder.identity_payload(
        {
            "schema": ladder.ROW_SCHEMA,
            "endpoint": endpoint,
            "clip_index": clip,
            "noise_seed": seed,
            "wan_calls": 1,
            "teacher_calls": 0,
            "auxiliary_target_array_opened": False,
            "future_feature_input": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "reserve_opened": False,
            "metrics": {
                "residual_r2": 0.5 if endpoint == "PFD_ALIGNED" else 0.2,
                "residual_cosine": 0.7 if endpoint == "PFD_ALIGNED" else 0.3,
                **{metric: value for metric in ladder.ERROR_METRICS},
            },
            "tensor_sha256": {
                "initial_video": f"noise-{clip}-{seed}",
                "raw_future_target": f"raw-future-{clip}",
                "raw_history_boundary": f"raw-history-{clip}",
                "final_video": (
                    f"off-{clip}-{seed}"
                    if endpoint in {"J1_OFF", "ZERO"}
                    else f"{endpoint}-{clip}-{seed}"
                ),
            },
        }
    )


def test_analysis_fails_when_fresh_vpm_noise_is_not_paired(monkeypatch):
    # Keep this inventory test cheap; bootstrap values are irrelevant because
    # the noise-pair assertion precedes analysis.
    j1 = []
    vpm = []
    for clip in range(*ladder.DEV_RANGE):
        for seed in ladder.DEV_NOISE_SEEDS:
            for endpoint in ladder.J1_ENDPOINTS:
                j1.append(_row(endpoint, clip, seed, 1.0))
            vpm.append(_row("VPM1", clip, seed, 1.0))
    vpm[0] = ladder.identity_payload(
        {
            **vpm[0],
            "tensor_sha256": {
                **vpm[0]["tensor_sha256"],
                "initial_video": "different-noise",
            },
        }
    )
    with pytest.raises(ladder.LadderError, match="initial Gaussian noise differs"):
        ladder.analyze_development_rows(j1, vpm)


def test_protocol_constants_are_frozen_in_document():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "experiments"
        / "CAUSAL_COMPRESSIBILITY_LADDER_PROTOCOL.md"
    ).read_text()
    for text in (
        "128--383",
        "384--415",
        "416--479",
        "20260824",
        "1024",
        "actual frozen VPM",
    ):
        assert text in protocol
