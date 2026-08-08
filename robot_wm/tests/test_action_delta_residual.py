"""Focused tests for the train-whitened action-velocity residual."""

from __future__ import annotations

import hashlib
import json

import torch

from robot_wm.modeling.networks.action_delta_residual import (
    ACTION_DELTA_STATS_SCHEMA,
    WhitenedActionDeltaResidual,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _stats(tmp_path):
    active = [False] * 157
    active[0] = True
    active[1] = True
    mean = [0.0] * 157
    mean[0] = 1.0
    mean[1] = -2.0
    std = [1.0] * 157
    std[0] = 2.0
    std[1] = 4.0
    unsigned = {
        "schema": ACTION_DELTA_STATS_SCHEMA,
        "split": "train",
        "protected_test_accessed": False,
        "future_action_chunks": [4, 12],
        "chunk_size": 5,
        "delta_steps": 4,
        "padding_dim": 157,
        "active_dimensions": 2,
        "mean": mean,
        "std": std,
        "active": active,
    }
    payload = {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _module(tmp_path, enabled: bool):
    path, digest = _stats(tmp_path)
    return WhitenedActionDeltaResidual(
        stats_path=str(path.resolve()),
        expected_stats_sha256=digest,
        enabled=enabled,
        action_dim=157,
        chunk_size=5,
        latent_dim=64,
        morph_dim=64,
        hidden=256,
        num_layers=3,
        clip_value=8.0,
        initialization_seed=20_260_808,
    )


def test_delta_whitening_uses_only_within_chunk_differences_and_masks_padding(tmp_path):
    module = _module(tmp_path, True)
    actions = torch.zeros(2, 8, 5, 157)
    # Deltas are +3 and +2 in the two active coordinates.
    actions[..., 0] = torch.arange(5).reshape(1, 1, 5) * 3.0
    actions[..., 1] = torch.arange(5).reshape(1, 1, 5) * 2.0
    actions[..., 2] = torch.arange(5).reshape(1, 1, 5) * 1000.0
    whitened = module.standardized_deltas(actions)
    assert whitened.shape == (2, 8, 4, 157)
    assert torch.equal(whitened[..., 0], torch.ones_like(whitened[..., 0]))
    assert torch.equal(whitened[..., 1], torch.ones_like(whitened[..., 1]))
    assert torch.count_nonzero(whitened[..., 2:]) == 0


def test_control_and_candidate_have_identical_state_and_parent_function_at_zero_gate(
    tmp_path,
):
    control = _module(tmp_path, False)
    candidate = _module(tmp_path, True)
    assert set(control.state_dict()) == set(candidate.state_dict())
    for key in control.state_dict():
        assert torch.equal(control.state_dict()[key], candidate.state_dict()[key])
    actions = torch.randn(2, 8, 5, 157)
    morphology = torch.randn(2, 64)
    residual_control, _ = control(actions, morphology)
    residual_candidate, _ = candidate(actions, morphology)
    assert torch.equal(residual_control, residual_candidate)
    base = torch.randn_like(residual_control)
    parent = base.clone()
    assert torch.equal(base + control.effective_gate() * residual_control, parent)
    assert torch.equal(base + candidate.effective_gate() * residual_candidate, parent)

    with torch.no_grad():
        control.raw_gate.fill_(0.5)
        candidate.raw_gate.fill_(0.5)
    assert float(control.effective_gate().detach()) == 0.0
    assert float(candidate.effective_gate().detach()) > 0.0
    assert not torch.equal(
        base + candidate.effective_gate() * residual_candidate,
        parent,
    )


def test_statistics_file_digest_and_identity_fail_closed(tmp_path):
    path, digest = _stats(tmp_path)
    with path.open("a") as handle:
        handle.write(" ")
    try:
        WhitenedActionDeltaResidual(
            stats_path=str(path.resolve()),
            expected_stats_sha256=digest,
            enabled=True,
        )
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("mutated statistics file was accepted")
