import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    REPO_ROOT
    / "projects"
    / "latent_action_models"
    / "lam"
    / "dual_explicit_action_dit_model.py"
)


def _load_model_module(monkeypatch):
    """Load the dual model without importing the optional VideoX runtime."""
    lam_package = types.ModuleType("lam")
    lam_package.__path__ = []
    base_module = types.ModuleType("lam.explicit_action_dit_model")

    class ExplicitActionDiTModel:
        pass

    base_module.ExplicitActionDiTModel = ExplicitActionDiTModel

    wan_module = types.ModuleType(
        "robot_wm.modeling.networks.wan_forward_model"
    )

    @dataclass
    class DualWanOutput:
        video_velocity: torch.Tensor
        tf_velocity: torch.Tensor
        tf_condition_tokens: torch.Tensor | None = None
        tf_condition_telemetry: dict | None = None

    wan_module.DualWanOutput = DualWanOutput
    monkeypatch.setitem(sys.modules, "lam", lam_package)
    monkeypatch.setitem(
        sys.modules, "lam.explicit_action_dit_model", base_module
    )
    monkeypatch.setitem(
        sys.modules,
        "robot_wm.modeling.networks.wan_forward_model",
        wan_module,
    )

    spec = importlib.util.spec_from_file_location(
        "dual_explicit_action_dit_model_sampling_test", MODEL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_nfe_model_sampling_pairs_matched_and_shuffled_noise(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = object.__new__(module.DualExplicitActionDiTModel)
    batch_size = 2
    video_clean = torch.zeros(batch_size, 16, 4, 1, 1)
    tf_clean = torch.randn(batch_size, 12, 4, 1, 1)

    model.num_history_latent = 2
    model.num_future_frames = 2
    model.tf_condition_mode = "shuffled"
    model.condition_on_tf = True
    model.tf_schedule_mode = "aligned"
    model.tf_lead_logit = 1.0
    model.evaluation_noise_seed = 123
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (1,)
    model.viz_num_steps = 1
    model.capture_latent_trajectories = False
    model._visualization_artifacts = None
    model._encode_clip = lambda _rgb: video_clean
    model._tf_clean = lambda _rgb, _shape: tf_clean
    model._latent_actions = (
        lambda _rgb, _actions, _morphology, _latent_frames, _history: (
            None,
            torch.zeros_like(video_clean),
            None,
        )
    )
    model._build_context = (
        lambda _batch_size, _device, _dtype: torch.empty(0)
    )
    model._build_clip = (
        lambda _batch_size, _device, _dtype: torch.empty(0)
    )

    class Tokenizer:
        @staticmethod
        def decode_temporal(latent, out_hw):
            return torch.zeros(
                latent.shape[0],
                3,
                latent.shape[2],
                *out_hw,
                device=latent.device,
                dtype=latent.dtype,
            )

    class Scheduler:
        timesteps = torch.tensor([1.0])
        sigmas = torch.tensor([1.0, 0.0])

        def set_timesteps(self, num_steps, device):
            assert num_steps == 1
            self.timesteps = torch.tensor([1.0], device=device)
            self.sigmas = torch.tensor([1.0, 0.0], device=device)

        @staticmethod
        def step(_velocity, _timestep, state):
            return types.SimpleNamespace(prev_sample=state)

    model.rgb_tokenizer = Tokenizer()
    model.sample_scheduler = Scheduler()
    captured = {}

    def forward_model(
        video_state,
        _timesteps,
        _z_control,
        _reference,
        _context,
        _clip_fea,
        *,
        noisy_tf,
        conditioning_tf,
        tf_sigma,
        condition_on_tf,
    ):
        captured["noisy_tf"] = noisy_tf.detach().clone()
        captured["conditioning_tf"] = conditioning_tf.detach().clone()
        captured["tf_sigma"] = tf_sigma.detach().clone()
        captured["condition_on_tf"] = condition_on_tf
        return module.DualWanOutput(
            video_velocity=torch.zeros_like(video_state),
            tf_velocity=torch.zeros_like(noisy_tf),
        )

    model.forward_model = forward_model
    rgb = torch.zeros(batch_size, 3, 13, 2, 2)

    model._sample_future(rgb)

    assert captured["condition_on_tf"] is True
    torch.testing.assert_close(
        captured["tf_sigma"], torch.ones(batch_size), rtol=0, atol=0
    )
    torch.testing.assert_close(
        captured["conditioning_tf"],
        captured["noisy_tf"],
        rtol=0,
        atol=0,
    )
