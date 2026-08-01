import torch

from robot_wm.modeling.video_latent_forcing import (
    clean_time_euler_step,
    corrupt_clean_time,
    x_prediction_to_velocity,
    x_prediction_v_loss,
)


def test_clean_time_corruption_endpoints_and_velocity_sign():
    clean = torch.tensor([[[2.0, -1.0]], [[4.0, 3.0]]])
    noise = torch.tensor([[[-3.0, 5.0]], [[1.0, -2.0]]])

    at_noise = corrupt_clean_time(clean, 0.0, noise=noise)
    at_clean = corrupt_clean_time(clean, 1.0, noise=noise)

    torch.testing.assert_close(at_noise.noisy, noise, rtol=0, atol=0)
    torch.testing.assert_close(at_clean.noisy, clean, rtol=0, atol=0)
    torch.testing.assert_close(
        at_noise.velocity_target,
        clean - noise,
        rtol=0,
        atol=0,
    )
    reconstructed = clean_time_euler_step(
        noise,
        at_noise.velocity_target,
        0.0,
        1.0,
    )
    torch.testing.assert_close(reconstructed, clean, rtol=0, atol=0)


def test_x_prediction_v_loss_matches_velocity_mse_and_full_batch_masking():
    clean = torch.randn(2, 3, 2, 2, dtype=torch.float64)
    noise = torch.randn_like(clean)
    clean_time = torch.tensor([0.2, 0.7], dtype=torch.float64)
    corruption = corrupt_clean_time(clean, clean_time, noise=noise)
    predicted_clean = clean + torch.tensor([0.1, -0.2], dtype=clean.dtype).reshape(
        2, 1, 1, 1
    )

    predicted_velocity = x_prediction_to_velocity(
        predicted_clean,
        corruption.noisy,
        clean_time,
    )
    velocity_mse = (predicted_velocity - (clean - noise)).square().flatten(1).mean(1)
    x_loss = x_prediction_v_loss(
        predicted_clean,
        clean,
        clean_time,
        reduction="none",
    )
    torch.testing.assert_close(x_loss, velocity_mse)

    masked = x_prediction_v_loss(
        predicted_clean,
        clean,
        clean_time,
        sample_weight=torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(masked, x_loss[0] / 2)


def test_released_default_clamps_the_near_clean_denominator_at_point_zero_five():
    state = torch.tensor([[2.0]])
    predicted_clean = torch.tensor([[3.0]])
    velocity = x_prediction_to_velocity(
        predicted_clean,
        state,
        torch.tensor([0.99]),
    )
    torch.testing.assert_close(velocity, torch.tensor([[20.0]]))

    loss = x_prediction_v_loss(
        predicted_clean,
        torch.tensor([[2.5]]),
        torch.tensor([0.99]),
    )
    torch.testing.assert_close(loss, torch.tensor(100.0))
