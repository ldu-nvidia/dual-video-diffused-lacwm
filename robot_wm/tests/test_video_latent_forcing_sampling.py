import torch

from robot_wm.modeling.video_latent_forcing import (
    VideoLatentForcingConfig,
    VideoLatentForcingOutput,
    sample_auxiliary_only,
    sample_cascade,
    sample_video_only,
)


class _RecordingPerfectXModel:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def __call__(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        *,
        condition_on_auxiliary,
    ):
        self.calls.append(
            {
                "video": noisy_video.clone(),
                "auxiliary": (
                    None if noisy_auxiliary is None else noisy_auxiliary.clone()
                ),
                "t_video": t_video.clone(),
                "t_auxiliary": t_auxiliary.clone(),
                "condition_on_auxiliary": condition_on_auxiliary,
            }
        )
        if noisy_auxiliary is None:
            noisy_auxiliary = noisy_video.new_zeros(
                noisy_video.shape[0], *self.config.auxiliary_shape
            )
        auxiliary_target = history[:, :1, :1, :1, :1].expand_as(noisy_auxiliary)
        if condition_on_auxiliary:
            video_target = noisy_auxiliary.mean(dim=(1, 2, 3, 4), keepdim=True)
            video_target = video_target.expand_as(noisy_video)
        else:
            video_target = torch.zeros_like(noisy_video)
        return VideoLatentForcingOutput(
            video_x=video_target,
            auxiliary_x=auxiliary_target,
        )


def _sampler_inputs():
    config = VideoLatentForcingConfig(
        video_channels=1,
        future_frames=2,
        history_frames=1,
        height=8,
        width=8,
        patch_size=(1, 8, 8),
        aux_channels=1,
        action_steps=2,
        action_dim=1,
        hidden_size=4,
        depth=1,
        num_heads=1,
    )
    history = torch.stack((torch.ones(config.history_shape), 2 * torch.ones(config.history_shape)))
    actions = torch.zeros(2, config.action_steps, config.action_dim)
    video_noise = torch.full((2, *config.future_shape), 7.0)
    auxiliary_noise = torch.full((2, *config.auxiliary_shape), -3.0)
    return config, history, actions, video_noise, auxiliary_noise


def test_strict_configurable_cascade_freezes_inactive_state_and_counts_calls():
    config, history, actions, video_noise, auxiliary_noise = _sampler_inputs()
    model = _RecordingPerfectXModel(config)
    sample = sample_cascade(
        model,
        history,
        actions,
        auxiliary_steps=3,
        video_steps=2,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
    )

    assert sample.model_calls == 5 == len(model.calls)
    for call in model.calls[:3]:
        torch.testing.assert_close(call["video"], video_noise, rtol=0, atol=0)
        assert call["condition_on_auxiliary"] is True
        assert torch.all(call["t_video"] == 0)
    torch.testing.assert_close(
        torch.stack([call["t_auxiliary"][0] for call in model.calls[:3]]),
        torch.tensor([0.0, 1 / 3, 2 / 3]),
    )
    for call in model.calls[3:]:
        torch.testing.assert_close(
            call["auxiliary"], sample.conditioning_auxiliary, rtol=0, atol=0
        )
        assert torch.all(call["t_auxiliary"] == 1)
    torch.testing.assert_close(sample.initial_video_noise, video_noise, rtol=0, atol=0)
    torch.testing.assert_close(
        sample.initial_auxiliary_noise, auxiliary_noise, rtol=0, atol=0
    )
    expected_auxiliary = history[:, :1, :1, :1, :1].expand_as(
        sample.generated_auxiliary
    )
    torch.testing.assert_close(sample.generated_auxiliary, expected_auxiliary)


def test_public_auxiliary_boundary_keeps_video_noise_exactly_frozen():
    config, history, actions, video_noise, auxiliary_noise = _sampler_inputs()
    model = _RecordingPerfectXModel(config)
    sample = sample_auxiliary_only(
        model,
        history,
        actions,
        auxiliary_steps=4,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
    )

    assert sample.model_calls == 4 == len(model.calls)
    for call in model.calls:
        torch.testing.assert_close(call["video"], video_noise, rtol=0, atol=0)
        assert call["condition_on_auxiliary"] is True
    expected = history[:, :1, :1, :1, :1].expand_as(sample.generated_auxiliary)
    torch.testing.assert_close(sample.generated_auxiliary, expected)


def test_public_b0_sampler_uses_exactly_50_video_only_calls():
    config, history, actions, video_noise, _ = _sampler_inputs()
    model = _RecordingPerfectXModel(config)
    sample = sample_video_only(
        model,
        history,
        actions,
        video_noise=video_noise,
    )

    assert sample.model_calls == 50 == len(model.calls)
    assert all(call["auxiliary"] is None for call in model.calls)
    assert all(
        call["condition_on_auxiliary"] is False for call in model.calls
    )
    assert all(torch.all(call["t_auxiliary"] == 0) for call in model.calls)
    # The released inference denominator is clamped at 0.05, so the final two
    # 0.02 Euler increments intentionally under-relax a perfect x=0 predictor.
    torch.testing.assert_close(
        sample.video,
        torch.full_like(sample.video, 0.1008),
        atol=1e-6,
        rtol=0,
    )


def test_default_is_exact_25_plus_25_and_controls_share_phase_one():
    config, history, actions, video_noise, auxiliary_noise = _sampler_inputs()
    samples = {}
    models = {}
    for mode in ("generated", "off", "shuffled"):
        models[mode] = _RecordingPerfectXModel(config)
        samples[mode] = sample_cascade(
            models[mode],
            history,
            actions,
            auxiliary_condition=mode,
            video_noise=video_noise,
            auxiliary_noise=auxiliary_noise,
        )

    assert all(sample.model_calls == 50 for sample in samples.values())
    torch.testing.assert_close(
        samples["generated"].generated_auxiliary,
        samples["off"].generated_auxiliary,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        samples["generated"].generated_auxiliary,
        samples["shuffled"].generated_auxiliary,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        samples["off"].conditioning_auxiliary,
        samples["off"].generated_auxiliary,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        samples["shuffled"].conditioning_auxiliary,
        torch.roll(samples["shuffled"].generated_auxiliary, 1, 0),
        rtol=0,
        atol=0,
    )
    assert all(
        call["condition_on_auxiliary"] is False
        for call in models["off"].calls[25:]
    )
    assert all(
        call["condition_on_auxiliary"] is True
        for call in models["off"].calls[:25]
    )
