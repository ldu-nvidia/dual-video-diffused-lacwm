import copy
import unittest

import numpy as np
import torch

from robot_wm.datasets.future_validity import (
    FutureValidityConfig,
    evaluate_future_validity,
)
from robot_wm.datasets.multi_dataset import MultiDataset


def _varying_view(frames=13, height=4, width=4, offset=0.0):
    pixels = torch.linspace(-0.8 + offset, 0.8 + offset, height * width)
    return pixels.reshape(1, 1, height, width).expand(frames, 3, -1, -1).clone()


def _sample(*, style="droid", valid_frames=13, marker=0.0):
    real = _varying_view(offset=marker * 0.01)
    if style == "droid":
        rgb = torch.cat([real, real * 0.8, real * 0.6], dim=-1)
    elif style == "egodex":
        pad = torch.full_like(real, -1.0)
        rgb = torch.cat([real, pad, pad], dim=-1)
    elif style == "constant":
        rgb = torch.zeros(13, 3, 4, 12)
    else:
        raise ValueError(style)
    mask = torch.arange(13) < valid_frames
    return {
        "rgb": rgb,
        "mask": mask,
        "actions": torch.full((13, 10), marker),
    }


class FutureValidityTest(unittest.TestCase):
    def setUp(self):
        self.config = FutureValidityConfig()

    def test_short_padded_clip_has_no_complete_future_latent(self):
        result = evaluate_future_validity(
            _sample(style="droid", valid_frames=8), self.config
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "no_complete_future_latent_group")
        self.assertEqual(result.future_latent_groups, ((5, 6, 7, 8), (9, 10, 11, 12)))
        self.assertEqual(result.valid_future_latent_groups, ())

    def test_all_constant_views_are_rejected(self):
        result = evaluate_future_validity(_sample(style="constant"), self.config)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "no_nonconstant_view")
        self.assertEqual(result.valid_views, ())

    def test_valid_droid_style_three_view_clip(self):
        result = evaluate_future_validity(_sample(style="droid"), self.config)
        self.assertTrue(result.valid)
        self.assertEqual(result.valid_views, (0, 1, 2))
        self.assertEqual(result.valid_future_latent_groups, (2, 3))

    def test_valid_egodex_style_one_real_two_padded_views(self):
        result = evaluate_future_validity(_sample(style="egodex"), self.config)
        self.assertTrue(result.valid)
        self.assertEqual(result.valid_views, (0,))
        self.assertEqual(result.valid_future_latent_groups, (2, 3))


class _FakeChildDataset:
    name = "DroidLeRobotDataset"
    ee_action_dim = 10
    decode_camera = True

    def __init__(self, samples):
        self.samples = samples
        self.loaded_state = None

    def __len__(self):
        return len(self.samples)

    def _get_sample(self, index):
        return copy.deepcopy(self.samples[int(index)])

    def state_dict(self):
        return {"child": 7}

    def load_state_dict(self, state):
        self.loaded_state = state


def _retry_dataset(samples, *, seed=31, max_retries=8):
    dataset = MultiDataset.__new__(MultiDataset)
    child = _FakeChildDataset(samples)
    dataset.datasets = {"Droid": child}
    dataset.dataset_lengths = [len(child)]
    dataset.cumulative_lengths = np.cumsum(dataset.dataset_lengths)
    dataset.padding_dim = 0
    dataset.img_augment = False
    dataset.emit_crop_rgb = False
    dataset.future_validity = FutureValidityConfig(max_retries=max_retries)
    dataset._seed = seed
    dataset._process_id = 0
    dataset._start_idx = 0
    dataset._gen = torch.Generator().manual_seed(101)
    dataset._augment_gen = torch.Generator().manual_seed(202)
    dataset._augment_initialized = False
    dataset._validity_gen = torch.Generator()
    dataset._validity_initialized = False
    return dataset


class MultiDatasetFutureValidityRetryTest(unittest.TestCase):
    def test_retry_stream_is_deterministic_and_checkpointed(self):
        samples = [
            _sample(style="constant", marker=0),
            _sample(style="droid", marker=1),
            _sample(style="egodex", marker=2),
            _sample(style="droid", marker=3),
        ]
        first = _retry_dataset(samples)
        second = _retry_dataset(samples)
        first_result = first._get_sample(0)
        second_result = second._get_sample(0)
        self.assertEqual(
            first_result["actions"][0, 0].item(),
            second_result["actions"][0, 0].item(),
        )

        checkpoint = first.state_dict()
        uninterrupted = first._get_sample(0)
        restored = _retry_dataset(samples)
        restored.load_state_dict(checkpoint)
        resumed = restored._get_sample(0)
        self.assertEqual(
            uninterrupted["actions"][0, 0].item(),
            resumed["actions"][0, 0].item(),
        )
        self.assertEqual(restored.datasets["Droid"].loaded_state, {"child": 7})

    def test_retry_exhaustion_raises_actionable_diagnostics(self):
        dataset = _retry_dataset(
            [_sample(style="constant", marker=0)], max_retries=2
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "future-validity retries exhausted.*Droid.*no_nonconstant_view",
        ):
            dataset._get_sample(0)


if __name__ == "__main__":
    unittest.main()
