from __future__ import annotations

import pytest
import torch

from robot_wm.modeling.dual_diffusion.invertible_two_clock import (
    InvertibleTwoClockError,
    assert_projector_receipt,
    band_sigma_schedule,
    euler_band_step,
    make_mixed_clock_state,
    project_p,
    project_q,
    project_velocities_and_loss,
    projector_receipt,
    split_pq,
)


def _native(batch: int = 2, seed: int = 20261201) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 16, 4, 24, 120, generator=generator)


def test_pq_is_exact_orthogonal_future_partition() -> None:
    value = _native()
    p, q = split_pq(value)
    receipt = projector_receipt(value)

    assert_projector_receipt(receipt)
    torch.testing.assert_close(p[:, :, :2], torch.zeros_like(p[:, :, :2]))
    torch.testing.assert_close(q[:, :, :2], torch.zeros_like(q[:, :, :2]))
    torch.testing.assert_close(p[:, :, 2:] + q[:, :, 2:], value[:, :, 2:])
    assert receipt.rank_fraction_p == 0.25
    assert receipt.rank_fraction_q == 0.75


def test_projector_does_not_cross_three_view_seams() -> None:
    value = torch.zeros(1, 1, 4, 24, 120)
    value[:, :, 2, 0, 39] = 4.0

    p = project_p(value)

    assert torch.count_nonzero(p[:, :, :, :, 40:]) == 0
    expected = torch.zeros_like(p)
    expected[:, :, 2, 0:2, 38:40] = 1.0
    torch.testing.assert_close(p, expected)


def test_diagonal_corruption_reconstructs_ordinary_rf_state() -> None:
    clean = _native()
    noise = _native(seed=20261202)
    sigma = torch.tensor([0.25, 0.75])

    mixed = make_mixed_clock_state(clean, noise, sigma, sigma)
    expanded = sigma.reshape(-1, 1, 1, 1, 1)
    ordinary = (1.0 - expanded) * clean + expanded * noise

    torch.testing.assert_close(mixed.native_noisy, ordinary)
    torch.testing.assert_close(mixed.noisy_p + mixed.noisy_q, ordinary * torch.tensor(
        [0.0, 0.0, 1.0, 1.0]
    ).reshape(1, 1, 4, 1, 1))


def test_projected_loss_equals_combined_full_coordinate_loss() -> None:
    target = _native()
    target_p, target_q = split_pq(target)
    native_prediction = _native(seed=20261203)
    q_prediction = _native(seed=20261204)
    mask = torch.ones(2, 1, 2, 24, 120)

    result = project_velocities_and_loss(
        native_prediction[:, :, 2:],
        q_prediction[:, :, 2:],
        target_p[:, :, 2:],
        target_q[:, :, 2:],
        mask,
        history_frames=0,
    )

    torch.testing.assert_close(result.loss, result.combined_loss)
    assert float(result.loss_identity_relative_error) <= 2e-6
    assert float(project_q(result.velocity_p, history_frames=0).abs().max()) <= 2e-6
    assert float(project_p(result.velocity_q, history_frames=0).abs().max()) <= 2e-6


@pytest.mark.parametrize("num_steps", [1, 2, 4])
def test_schedules_have_exact_endpoints_and_nfe_one_collapses(num_steps: int) -> None:
    base = torch.linspace(1.0, 0.0, num_steps + 1)
    sync = band_sigma_schedule(base, "synchronous")
    p_leads = band_sigma_schedule(base, "p_leads")
    q_leads = band_sigma_schedule(base, "q_leads")

    for p, q in (sync, p_leads, q_leads):
        assert float(p[0]) == float(q[0]) == 1.0
        assert float(p[-1]) == float(q[-1]) == 0.0
        assert torch.all(p[1:] <= p[:-1])
        assert torch.all(q[1:] <= q[:-1])
    if num_steps == 1:
        for schedule in (p_leads, q_leads):
            torch.testing.assert_close(schedule[0], sync[0])
            torch.testing.assert_close(schedule[1], sync[1])
    else:
        assert torch.all(p_leads[1][1:-1] > p_leads[0][1:-1])
        assert torch.all(q_leads[1][1:-1] < q_leads[0][1:-1])


def test_diagonal_euler_sum_equals_full_euler() -> None:
    state = _native()
    p, q = split_pq(state)
    velocity = _native(seed=20261205)
    velocity_p = project_p(velocity)
    velocity_q = project_q(velocity)

    next_p, next_q = euler_band_step(
        p, q, velocity_p, velocity_q, 0.75, 0.5, 0.75, 0.5
    )

    expected = p + q - 0.25 * (velocity_p + velocity_q)
    torch.testing.assert_close(next_p + next_q, expected)


def test_projector_receipt_fails_closed_when_tampered() -> None:
    receipt = projector_receipt(_native())
    tampered = receipt.__class__(
        **{**receipt.__dict__, "history_nonzero": 1}
    )
    with pytest.raises(InvertibleTwoClockError, match="history_nonzero"):
        assert_projector_receipt(tampered)
