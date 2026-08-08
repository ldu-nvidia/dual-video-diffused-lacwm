from __future__ import annotations

import argparse

import pytest

from tools import vpm_phaselock_probe as probe


def test_endpoint_grid_is_fixed_and_exactly_compute_matched() -> None:
    assert len(probe.ENDPOINTS) == 12
    assert tuple(
        endpoint.code
        for endpoint in probe.ENDPOINTS
        if endpoint.kind == "ordinary"
    ) == ("ordinary_b1", "ordinary_b3", "ordinary_b4", "ordinary_b6")
    for endpoint in probe.ENDPOINTS:
        if endpoint.kind == "ordinary":
            assert endpoint.few_steps == 0
            assert endpoint.full_steps == endpoint.total_transformer_calls
            continue
        assert endpoint.prior_alignment in {"aligned", "shuffled"}
        assert (
            endpoint.few_steps + endpoint.full_steps
            == endpoint.total_transformer_calls
        )
        assert f"ordinary_b{endpoint.total_transformer_calls}" in probe.ENDPOINT_BY_CODE


def test_guidance_schedule_is_fixed_early_half_linear_decay() -> None:
    assert probe.guidance_end_step(1) == 1
    assert probe.guidance_end_step(2) == 1
    assert probe.guidance_end_step(4) == 2
    assert probe.linear_guidance_strength(0, 4) == pytest.approx(0.05)
    assert probe.linear_guidance_strength(1, 4) == pytest.approx(0.025)
    assert probe.linear_guidance_strength(2, 4) == 0.0
    with pytest.raises(probe.PhaseLockProbeError):
        probe.guidance_end_step(0)


def test_registration_identity_detects_protocol_mutation() -> None:
    payload = probe.identity_payload(
        {
            "kind": probe.KIND_REGISTRATION,
            "fixed_protocol": {
                "split": "validation",
                "endpoint_grid": [probe.asdict(value) for value in probe.ENDPOINTS],
            },
        }
    )
    assert probe.identity_valid(payload)
    payload["fixed_protocol"]["split"] = "test"
    assert not probe.identity_valid(payload)


def test_exact_controlled_study_identity_is_pinned() -> None:
    payload = {
        "kind": probe.EXPECTED_STUDY_KIND,
        "study_id": probe.EXPECTED_STUDY_ID,
        "identity_sha256": probe.EXPECTED_STUDY_IDENTITY_SHA256,
        "study_root": str(probe.EXPECTED_STUDY_ROOT),
    }
    probe._validate_expected_study_identity(  # noqa: SLF001
        payload, probe.EXPECTED_STUDY_ROOT
    )
    for key, replacement in (
        ("kind", "other"),
        ("study_id", "other"),
        ("identity_sha256", "0" * 64),
        ("study_root", "/lustre/other"),
    ):
        changed = {**payload, key: replacement}
        with pytest.raises(probe.PhaseLockProbeError):
            probe._validate_expected_study_identity(  # noqa: SLF001
                changed, probe.EXPECTED_STUDY_ROOT
            )


def test_output_root_rejects_lexical_escape(tmp_path) -> None:
    lustre = probe.Path("/lustre")
    if not lustre.is_dir():
        pytest.skip("unit environment has no /lustre mount")
    with pytest.raises(probe.PhaseLockProbeError):
        probe._canonical_fresh_lustre_output(  # noqa: SLF001
            lustre / ".." / "tmp" / "escaped-probe"
        )


def test_cli_exposes_no_test_or_lockbox_split_switch() -> None:
    parser = probe._parser()
    options = {
        option
        for action in parser._actions
        if isinstance(action, argparse.Action)
        for option in action.option_strings
    }
    # Subparser-specific options are inspected recursively.
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                options.update(
                    option
                    for subaction in subparser._actions
                    for option in subaction.option_strings
                )
    assert "--split" not in options
    assert "--test" not in options
    assert "--lockbox" not in options


def test_future_delta_guidance_preserves_history_and_first_future() -> None:
    torch = pytest.importorskip("torch")
    latents = torch.tensor([[[[[0.0]], [[1.0]], [[3.0]], [[7.0]]]]])
    # Shape is [B=1,C=1,T=4,H=1,W=1], with h=2 and future delta 4.
    prior = torch.tensor([[[[[2.0]]]]])
    guided = probe.apply_future_delta_guidance(latents, prior, 2, 0.5)
    assert torch.equal(guided[:, :, :3], latents[:, :, :3])
    assert guided[0, 0, 3, 0, 0].item() == pytest.approx(6.0)


def test_shuffled_prior_records_nonself_donor() -> None:
    torch = pytest.importorskip("torch")
    prior = torch.tensor([[[[[1.0]]]], [[[[2.0]]]]])
    ids = torch.tensor([1_000_000, 1_000_008])
    shuffled, donors = probe.shuffled_motion_prior(prior, ids)
    assert torch.equal(shuffled[0], prior[1])
    assert torch.equal(shuffled[1], prior[0])
    assert donors.tolist() == [1_000_008, 1_000_000]


def test_guided_endpoint_enforces_exact_composite_call_count() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Scheduler:
        def set_timesteps(self, steps, device):
            self.timesteps = torch.arange(steps - 1, -1, -1, device=device)
            self.sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)

        def step(self, velocity, timestep, state):
            del timestep
            return SimpleNamespace(prev_sample=state - velocity / len(self.timesteps))

    class Forward(torch.nn.Module):
        def forward(self, video, timestep, z_control, reference, context, clip, **kwargs):
            del timestep, z_control, reference, context, clip, kwargs
            return SimpleNamespace(
                video_velocity=torch.ones_like(video) * 0.1,
                tf_velocity=torch.ones_like(video) * 0.1,
            )

    model = SimpleNamespace(
        sample_scheduler=Scheduler(),
        forward_model=Forward(),
        tf_schedule_mode="aligned",
        tf_lead_logit=0.0,
    )
    initial = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4, 1, 1)
    prepared = {
        "initial_video": initial,
        "initial_tf": initial.clone(),
        "reference": torch.zeros_like(initial),
        "history_frames": 2,
        "z_control": torch.zeros(2, 1),
        "context": torch.zeros(2, 1),
        "clip_fea": torch.zeros(2, 1),
        "batch_size": 2,
        "dtype": initial.dtype,
        "device": initial.device,
    }
    calls = 0

    def hook(*_args):
        nonlocal calls
        calls += 1

    handle = model.forward_model.register_forward_hook(hook)
    try:
        endpoint = probe.ENDPOINT_BY_CODE["phaselock_k1_f2_aligned"]
        result = probe.run_endpoint(
            model, prepared, endpoint, torch.tensor([1_000_000, 1_000_008])
        )
    finally:
        handle.remove()
    assert result["calls"] == endpoint.total_transformer_calls == 3
    assert calls == 3
    assert result["guidance_strengths"] == [0.05, 0.0]


def test_registered_input_loader_uses_only_rgb_actions_and_pads_actions(
    tmp_path, monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    class FakeArray:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = np.dtype(dtype)

        def __getitem__(self, index):
            del index
            return np.zeros(self.shape[1:], dtype=self.dtype)

    rgb_path = tmp_path / "rgb.npy"
    actions_path = tmp_path / "actions.npy"
    arrays = {
        rgb_path: FakeArray((64, 13, 3, 180, 960), np.float16),
        actions_path: FakeArray((64, 13, 5, 23), np.float32),
    }
    monkeypatch.setattr(
        np,
        "load",
        lambda path, **_kwargs: arrays[path],
    )
    dataset = probe._RegisteredValidationInputs(
        rgb_path=rgb_path,
        actions_path=actions_path,
        descriptors=[{"clip_id": str(index)} for index in range(64)],
        padding_dim=157,
    )
    sample = dataset[7]
    assert set(sample) == {"rgb", "actions", "morphology_index", "clip_index"}
    assert tuple(sample["actions"].shape) == (13, 5, 157)
    assert torch.count_nonzero(sample["actions"]) == 0
    assert sample["morphology_index"].item() == 9
    assert sample["clip_index"].item() == 7
