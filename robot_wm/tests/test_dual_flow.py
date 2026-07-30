import torch

from robot_wm.modeling.dual_diffusion.flow import (
    DualClockSampler,
    cascaded_step_counts,
    corrupt_flow,
    derive_tf_sigma,
    euler_flow_step,
    make_paired_sigma_schedule,
    pair_native_cascaded_sigma_schedule,
    pair_video_sigma_schedule,
)


def test_flow_corruption_endpoints_and_velocity_sign():
    clean = torch.randn(3, 2, 4)
    noise = torch.randn_like(clean)

    at_data = corrupt_flow(clean, torch.zeros(3), noise=noise)
    at_noise = corrupt_flow(clean, torch.ones(3), noise=noise)

    torch.testing.assert_close(at_data.noisy, clean)
    torch.testing.assert_close(at_noise.noisy, noise)
    torch.testing.assert_close(at_noise.velocity_target, noise - clean)
    reconstructed = euler_flow_step(
        noise, at_noise.velocity_target, torch.tensor(1.0), torch.tensor(0.0)
    )
    torch.testing.assert_close(reconstructed, clean)


def test_independent_clocks_are_actually_independent():
    generator = torch.Generator().manual_seed(7)
    clocks = DualClockSampler(mode="independent")(
        64, device="cpu", generator=generator
    )
    assert not torch.equal(clocks.video_sigma, clocks.tf_sigma)
    assert torch.all(clocks.video_loss_weight == 1)
    assert torch.all(clocks.tf_loss_weight == 1)


def test_tf_lead_is_cleaner_in_lacwm_sigma_convention():
    generator = torch.Generator().manual_seed(9)
    clocks = DualClockSampler(mode="tf_leads", tf_lead_logit=1.5)(
        32, device="cpu", generator=generator
    )
    assert torch.all(clocks.tf_sigma < clocks.video_sigma)


def test_cascaded_noised_training_contract():
    generator = torch.Generator().manual_seed(11)
    clocks = DualClockSampler(
        mode="tf_first_cascaded_noised",
        tf_loss_probability=0.5,
        tf_condition_max_sigma=0.25,
    )(256, device="cpu", generator=generator)

    torch.testing.assert_close(
        clocks.video_loss_weight + clocks.tf_loss_weight,
        torch.ones_like(clocks.video_loss_weight),
    )
    tf_examples = clocks.tf_loss_weight.bool()
    video_examples = clocks.video_loss_weight.bool()
    assert tf_examples.any() and video_examples.any()
    assert torch.all(clocks.video_sigma[tf_examples] == 1)
    assert torch.all(clocks.tf_sigma[video_examples] <= 0.25)


def test_cascade_preserves_supplied_native_video_clocks():
    native = torch.tensor([0.91, 0.67, 0.23, 0.02])
    generator = torch.Generator().manual_seed(19)
    clocks = DualClockSampler(
        mode="tf_first_cascaded_noised",
        tf_loss_probability=0.5,
        tf_condition_max_sigma=0.25,
    )(
        native.numel(),
        device="cpu",
        generator=generator,
        native_video_sigma=native,
    )

    video_examples = clocks.video_loss_weight.bool()
    tf_examples = clocks.tf_loss_weight.bool()
    torch.testing.assert_close(
        clocks.video_sigma[video_examples],
        native[video_examples],
    )
    assert torch.all(clocks.video_sigma[tf_examples] == 1)


def test_inference_schedules_have_correct_endpoints_and_order():
    for mode in ("aligned", "tf_leads", "tf_first_cascaded"):
        schedule = make_paired_sigma_schedule(8, mode=mode)
        assert schedule.num_steps == 8
        assert schedule.video[0] == schedule.time_frequency[0] == 1
        assert schedule.video[-1] == schedule.time_frequency[-1] == 0
        assert torch.all(torch.diff(schedule.video) <= 0)
        assert torch.all(torch.diff(schedule.time_frequency) <= 0)

    leading = make_paired_sigma_schedule(8, mode="tf_leads")
    assert torch.all(leading.time_frequency <= leading.video)

    cascaded = make_paired_sigma_schedule(8, mode="tf_first_cascaded")
    video_updates = torch.diff(cascaded.video) != 0
    tf_updates = torch.diff(cascaded.time_frequency) != 0
    assert not torch.any(video_updates & tf_updates)


def test_native_cascade_preserves_every_video_scheduler_node():
    native = torch.tensor([1.0, 0.80, 0.35, 0.0])
    paired = pair_native_cascaded_sigma_schedule(
        native,
        total_steps=7,
        tf_fraction=4 / 7,
    )

    assert cascaded_step_counts(7, tf_fraction=4 / 7) == (4, 3)
    torch.testing.assert_close(paired.video[4:], native)
    assert torch.all(paired.video[:4] == 1)
    assert torch.all(paired.time_frequency[4:] == 0)
    assert paired.num_steps == 7
    assert not torch.any(
        (torch.diff(paired.video) != 0)
        & (torch.diff(paired.time_frequency) != 0)
    )


def test_derived_tf_clock_preserves_native_video_schedule_and_exact_endpoints():
    video = torch.tensor([1.0, 0.91, 0.70, 0.33, 0.0])
    paired = pair_video_sigma_schedule(
        video, mode="tf_leads", tf_lead_logit=1.0
    )

    torch.testing.assert_close(paired.video, video)
    assert paired.time_frequency[0] == 1
    assert paired.time_frequency[-1] == 0
    assert torch.all(paired.time_frequency[1:-1] < video[1:-1])
    assert torch.all(torch.diff(paired.time_frequency) <= 0)


def test_derived_aligned_clock_is_an_exact_copy():
    video = torch.rand(11).sort(descending=True).values
    torch.testing.assert_close(
        derive_tf_sigma(video, mode="aligned"),
        video,
        rtol=0,
        atol=0,
    )
