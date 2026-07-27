import pytest
import torch

from robot_wm.modeling.dual_diffusion.conditioning import (
    make_oracle_conditioning_tf,
    make_sampling_conditioning_tf,
    make_training_conditioning_tf,
    roll_across_global_batch,
)


def test_local_global_roll_is_deranged_and_multiset_preserving():
    state = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1, 1)
    rolled = roll_across_global_batch(state)

    assert torch.equal(rolled[:, 0, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 0.0]))
    assert sorted(rolled.flatten().tolist()) == sorted(state.flatten().tolist())
    assert not torch.any(rolled.flatten() == state.flatten())


def test_roll_fails_closed_for_single_sample_without_ddp():
    with pytest.raises(RuntimeError, match="global batch"):
        roll_across_global_batch(torch.zeros(1, 2, 3, 4, 5))


def test_shuffled_training_condition_preserves_history_sigma_and_local_noise():
    clean = torch.stack(
        [
            torch.zeros(1, 4, 1, 1),
            torch.full((1, 4, 1, 1), 10.0),
        ]
    )
    noise = torch.stack(
        [
            torch.full((1, 4, 1, 1), 2.0),
            torch.full((1, 4, 1, 1), 3.0),
        ]
    )
    sigma = torch.tensor([0.25, 0.75]).reshape(2, 1, 1, 1, 1)
    noisy = (1.0 - sigma) * clean + sigma * noise
    noisy[:, :, :2] = clean[:, :, :2]

    shuffled = make_training_conditioning_tf(
        mode="shuffled",
        tf_clean=clean,
        tf_noise=noise,
        tf_noisy=noisy,
        tf_sigma_expanded=sigma,
        history_frames=2,
    )

    torch.testing.assert_close(shuffled[:, :, :2], clean[:, :, :2])
    expected_future = (1.0 - sigma) * torch.roll(clean, -1, 0) + sigma * noise
    torch.testing.assert_close(shuffled[:, :, 2:], expected_future[:, :, 2:])


def test_matched_and_off_return_the_exact_own_state():
    state = torch.randn(2, 3, 4, 5, 6)
    sigma = torch.full((2, 1, 1, 1, 1), 0.5)
    for mode in ("matched", "off"):
        returned = make_training_conditioning_tf(
            mode=mode,
            tf_clean=state,
            tf_noise=state,
            tf_noisy=state,
            tf_sigma_expanded=sigma,
            history_frames=2,
        )
        assert returned is state


def test_shuffled_sampling_preserves_local_noise_and_rolls_only_residual():
    clean = torch.stack(
        [
            torch.tensor([[[[0.0]], [[0.0]], [[4.0]], [[6.0]]]]),
            torch.tensor([[[[1.0]], [[1.0]], [[8.0]], [[10.0]]]]),
        ]
    )
    noise = torch.stack(
        [
            torch.tensor([[[[20.0]], [[20.0]], [[2.0]], [[3.0]]]]),
            torch.tensor([[[[30.0]], [[30.0]], [[5.0]], [[7.0]]]]),
        ]
    )
    sigma = torch.full((2, 1, 1, 1, 1), 0.25)
    state = (1.0 - sigma) * clean + sigma * noise
    state[:, :, :2] = clean[:, :, :2]
    condition = make_sampling_conditioning_tf(
        mode="shuffled",
        tf_state=state,
        tf_noise=noise,
        tf_sigma_expanded=sigma,
        history_frames=2,
    )

    torch.testing.assert_close(condition[:, :, :2], state[:, :, :2])
    expected = (1.0 - sigma) * torch.roll(clean, -1, 0) + sigma * noise
    torch.testing.assert_close(
        condition[:, :, 2:], expected[:, :, 2:]
    )


def test_shuffled_sampling_is_exactly_matched_at_pure_noise():
    noise = torch.randn(2, 3, 4, 2, 2)
    clean_history = torch.randn(2, 3, 2, 2, 2)
    state = noise.clone()
    state[:, :, :2] = clean_history

    matched = make_sampling_conditioning_tf(
        mode="matched",
        tf_state=state,
        tf_noise=noise,
        tf_sigma_expanded=torch.ones(2, 1, 1, 1, 1),
        history_frames=2,
    )
    shuffled = make_sampling_conditioning_tf(
        mode="shuffled",
        tf_state=state,
        tf_noise=noise,
        tf_sigma_expanded=torch.ones(2, 1, 1, 1, 1),
        history_frames=2,
    )

    assert matched is state
    torch.testing.assert_close(shuffled, matched, rtol=0, atol=0)


def test_oracle_conditioning_respects_noise_and_clean_endpoints():
    clean = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4, 1, 1)
    wrong = torch.roll(clean, -1, 0)
    noise = torch.full_like(clean, 13.0)

    at_noise = make_oracle_conditioning_tf(
        tf_clean=clean,
        tf_noise=noise,
        tf_sigma_expanded=torch.ones(2, 1, 1, 1, 1),
        history_frames=2,
        wrong_tf_clean=wrong,
    )
    at_clean = make_oracle_conditioning_tf(
        tf_clean=clean,
        tf_noise=noise,
        tf_sigma_expanded=torch.zeros(2, 1, 1, 1, 1),
        history_frames=2,
        wrong_tf_clean=wrong,
    )

    # Observed history is always the correct local history.
    torch.testing.assert_close(at_noise[:, :, :2], clean[:, :, :2])
    torch.testing.assert_close(at_clean[:, :, :2], clean[:, :, :2])
    # At sigma=1 hidden future is pure noise, so aligned and wrong clean
    # content are indistinguishable. At sigma=0 the selected clean source wins.
    torch.testing.assert_close(at_noise[:, :, 2:], noise[:, :, 2:])
    torch.testing.assert_close(at_clean[:, :, 2:], wrong[:, :, 2:])
