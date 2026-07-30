import torch

from robot_wm.modeling.dual_diffusion.vjepa2_target import (
    PCAWhiteningStats,
    align_vjepa2_1_tokens,
    prepare_vjepa2_1_views,
    project_and_stack_views,
)


def test_prepare_views_keeps_camera_batches_isolated():
    video = torch.full((1, 3, 3, 2, 4), -1.0)
    video[..., :2] = 1.0

    prepared = prepare_vjepa2_1_views(
        video,
        num_views=2,
        frame_map=(0, 1, 1, 2),
        expected_view_size=(2, 2),
        padded_view_size=(2, 2),
        teacher_size=(2, 2),
    )

    assert prepared.shape == (2, 3, 4, 2, 2)
    # The first view was all +1 and the second all -1 before the fixed
    # ImageNet normalization.  No spatial operation may mix them.
    assert torch.all(prepared[0] > prepared[1])


def test_temporal_pairing_matches_wan_bins():
    # Four two-frame tubelet positions, one spatial token each.
    values = torch.arange(4, dtype=torch.float32)
    tokens = values.reshape(1, 4, 1)

    aligned = align_vjepa2_1_tokens(
        tokens,
        batch_size=1,
        num_views=1,
        teacher_frames=8,
        teacher_size=(2, 2),
        output_frames=2,
        patch_size=2,
        tubelet_size=2,
    )

    assert aligned.shape == (1, 1, 2, 1, 1, 1)
    torch.testing.assert_close(
        aligned.flatten(), torch.tensor([0.5, 2.5])
    )


def test_projection_whitens_and_width_stacks_views():
    # [B=1,V=2,F=1,H=1,W=2,D=2]
    aligned = torch.tensor(
        [
            [
                [[[[1.0, 2.0], [3.0, 4.0]]]],
                [[[[5.0, 6.0], [7.0, 8.0]]]],
            ]
        ]
    )
    stats = PCAWhiteningStats(
        mean=torch.tensor([1.0, 2.0]),
        components=torch.eye(2),
        eigenvalues=torch.tensor([1.0, 4.0]),
    )

    target = project_and_stack_views(
        aligned, stats, output_dtype=torch.float32
    )

    assert target.shape == (1, 2, 1, 1, 4)
    # Camera zero occupies the first two width locations, camera one the last
    # two.  Channel one is divided by sqrt(eigenvalue)=2.
    torch.testing.assert_close(
        target[0, 0, 0, 0], torch.tensor([0.0, 2.0, 4.0, 6.0])
    )
    torch.testing.assert_close(
        target[0, 1, 0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
    )


def test_pca_payload_round_trip_is_exact():
    stats = PCAWhiteningStats(
        mean=torch.tensor([1.0, 2.0, 3.0]),
        components=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        eigenvalues=torch.tensor([2.0, 4.0]),
        eps=1e-5,
    )

    restored = PCAWhiteningStats.from_payload(stats.to_payload())

    assert restored.eps == stats.eps
    assert torch.equal(restored.mean, stats.mean)
    assert torch.equal(restored.components, stats.components)
    assert torch.equal(restored.eigenvalues, stats.eigenvalues)
