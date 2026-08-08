"""Focused tests for the per-substep action context adapter."""

from __future__ import annotations

import hashlib
import json

import torch

from robot_wm.modeling.networks.action_token_context import (
    ACTION_TOKEN_STATS_SCHEMA,
    ActionTokenContextAdapter,
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
        "schema": ACTION_TOKEN_STATS_SCHEMA,
        "split": "train",
        "protected_test_accessed": False,
        "future_action_chunks": [4, 12],
        "chunk_size": 5,
        "num_transitions": 8,
        "token_count": 40,
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
    return ActionTokenContextAdapter(
        stats_path=str(path.resolve()),
        expected_stats_sha256=digest,
        enabled=enabled,
        action_dim=157,
        chunk_size=5,
        num_transitions=8,
        morph_dim=64,
        hidden=256,
        text_dim=4096,
        clip_value=8.0,
        initialization_seed=20_260_808,
    )


def test_action_whitening_preserves_all_substeps_and_masks_padding(tmp_path):
    module = _module(tmp_path, True)
    actions = torch.zeros(2, 8, 5, 157)
    actions[..., 0] = 3.0
    actions[..., 1] = 2.0
    actions[..., 2] = 1000.0
    whitened = module.standardized_actions(actions)
    assert whitened.shape == (2, 8, 5, 157)
    assert torch.equal(whitened[..., 0], torch.ones_like(whitened[..., 0]))
    assert torch.equal(whitened[..., 1], torch.ones_like(whitened[..., 1]))
    assert torch.count_nonzero(whitened[..., 2:]) == 0


def test_tokens_are_transition_major_and_each_substep_can_change_independently(tmp_path):
    module = _module(tmp_path, True)
    actions = torch.zeros(1, 8, 5, 157)
    morphology = torch.zeros(1, 64)
    baseline, _ = module(actions, morphology)
    changed = actions.clone()
    changed[:, 3, 2, 0] = 5.0
    tokens, _ = module(changed, morphology)
    difference = (tokens - baseline).abs().sum(dim=-1)
    assert tokens.shape == (1, 40, 4096)
    assert torch.count_nonzero(difference) == 1
    assert float(difference[0, 3 * 5 + 2]) > 0.0


def test_control_and_candidate_have_identical_state_and_exact_zero_gate(tmp_path):
    control = _module(tmp_path, False)
    candidate = _module(tmp_path, True)
    assert set(control.state_dict()) == set(candidate.state_dict())
    for key in control.state_dict():
        assert torch.equal(control.state_dict()[key], candidate.state_dict()[key])
    actions = torch.randn(2, 8, 5, 157)
    morphology = torch.randn(2, 64)
    tokens_control, _ = control(actions, morphology)
    tokens_candidate, _ = candidate(actions, morphology)
    assert torch.equal(tokens_control, tokens_candidate)
    null = torch.randn(2, 40, 4096)
    assert torch.equal(null + control.effective_gate() * tokens_control, null)
    assert torch.equal(null + candidate.effective_gate() * tokens_candidate, null)

    with torch.no_grad():
        control.raw_gate.fill_(0.5)
        candidate.raw_gate.fill_(0.5)
    assert float(control.effective_gate().detach()) == 0.0
    assert float(candidate.effective_gate().detach()) > 0.0
    with candidate.runtime_hard_mask():
        assert float(candidate.effective_gate().detach()) == 0.0
    assert float(candidate.effective_gate().detach()) > 0.0
    assert not torch.equal(
        null + candidate.effective_gate() * tokens_candidate,
        null,
    )


def test_statistics_file_digest_and_identity_fail_closed(tmp_path):
    path, digest = _stats(tmp_path)
    with path.open("a") as handle:
        handle.write(" ")
    try:
        ActionTokenContextAdapter(
            stats_path=str(path.resolve()),
            expected_stats_sha256=digest,
            enabled=True,
        )
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("mutated statistics file was accepted")
