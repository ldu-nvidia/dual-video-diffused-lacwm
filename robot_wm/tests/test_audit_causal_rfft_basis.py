import math

import pytest
import torch

from robot_wm.modeling.dual_diffusion.time_frequency import PerViewCausalRFFT
from tools.audit_causal_rfft_basis import (
    apply_component_matrix,
    audit_tf_clean_artifacts,
    build_report,
    causal_time_domain_packing,
    packed_rfft_matrix,
    parseval_channel_scales,
)


def test_length_four_matrix_formula_singular_values_and_parseval_scaling():
    matrix = packed_rfft_matrix()
    expected = 0.5 * torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(matrix, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.linalg.svdvals(matrix),
        torch.tensor(
            [1.0, 1.0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
            dtype=torch.float64,
        ),
    )
    assert torch.linalg.cond(matrix).item() == pytest.approx(math.sqrt(2))

    scaled = torch.diag(parseval_channel_scales()) @ matrix
    torch.testing.assert_close(
        scaled.T @ scaled,
        torch.eye(4, dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )


def test_same_shape_time_packing_maps_exactly_to_production_coefficients():
    generator = torch.Generator().manual_seed(17)
    video = torch.randn(
        2, 13, 3, 5, 18, dtype=torch.float64, generator=generator
    )
    transform = PerViewCausalRFFT(
        num_views=3,
        output_size=(4, 12),
        window_size=4,
        pad_multiple=None,
        normalization="none",
    )

    packed_time, pooled_video = causal_time_domain_packing(transform, video)
    expected = apply_component_matrix(
        packed_time, packed_rfft_matrix(), num_views=3
    )
    actual = transform(video)

    assert packed_time.shape == actual.shape == (2, 12, 4, 4, 12)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        transform.inverse(actual), pooled_video, rtol=1e-14, atol=1e-14
    )


def test_component_matrix_round_trip_and_view_isolation():
    generator = torch.Generator().manual_seed(23)
    packed = torch.zeros(1, 12, 4, 3, 15, dtype=torch.float64)
    packed[..., :5] = torch.randn(
        packed[..., :5].shape, dtype=packed.dtype, generator=generator
    )
    matrix = packed_rfft_matrix()

    coefficients = apply_component_matrix(packed, matrix, num_views=3)
    reconstructed = apply_component_matrix(
        coefficients, torch.linalg.inv(matrix), num_views=3
    )

    torch.testing.assert_close(reconstructed, packed, rtol=1e-14, atol=1e-14)
    assert coefficients[..., :5].abs().sum() > 0
    assert torch.count_nonzero(coefficients[..., 5:]) == 0


def test_theoretical_white_variance_and_coefficient_noise_covariance():
    matrix = packed_rfft_matrix()
    torch.testing.assert_close(
        matrix @ matrix.T,
        torch.diag(torch.tensor([1.0, 0.5, 0.5, 1.0], dtype=torch.float64)),
        rtol=1e-14,
        atol=1e-14,
    )
    inverse = torch.linalg.inv(matrix)
    eigenvalues = torch.linalg.eigvalsh(inverse @ inverse.T)
    torch.testing.assert_close(
        eigenvalues,
        torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64),
    )


def test_artifact_audit_recovers_time_packing_and_energy(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    generator = torch.Generator().manual_seed(29)
    time_packed = torch.randn(
        2, 12, 4, 2, 6, dtype=torch.float64, generator=generator
    )
    matrix = packed_rfft_matrix()
    coefficients = apply_component_matrix(time_packed, matrix, num_views=3)
    path = tmp_path / "artifact.safetensors"
    safetensors_torch.save_file({"tf_clean": coefficients}, path)

    report = audit_tf_clean_artifacts([path], matrix=matrix, num_views=3)

    assert report is not None
    assert report["round_trip_coefficient_max_abs_error"] < 1e-14
    assert report[
        "sqrt2_weighted_coefficient_energy_over_time_packed_energy"
    ] == pytest.approx(1.0)
    assert report["artifacts"][0]["sha256"]


def test_report_states_scoped_claims_and_exact_synthetic_contract():
    report = build_report(seed=31)

    assert report["linear_algebra"]["packed_window_condition_number_2"] == (
        pytest.approx(math.sqrt(2))
    )
    assert report["linear_algebra"]["original_operator_rank"] == 13
    assert report["synthetic_verification"]["view_isolation_exact"]
    assert (
        report["synthetic_verification"]["production_vs_matrix_max_abs_error"]
        < 1e-14
    )
    assert len(report["claims"]["forbidden_without_further_controls"]) >= 5
    assert report["identity_sha256"]
