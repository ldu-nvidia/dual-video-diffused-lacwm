import unittest
from types import SimpleNamespace

import torch

from lam.latent_action_dit_model import LatentActionDiTModel


class ActionDecodingLossTest(unittest.TestCase):
    def setUp(self):
        # The loss helpers do not depend on initialized model state.
        self.model = LatentActionDiTModel.__new__(LatentActionDiTModel)
        self.model.aux_losses = {}
        self.model.action_decoder = SimpleNamespace(
            split_type={
                "action_type_split": {"ee": [0, 1]},
                "0": {"ee": 1},
                "2": {"ee": 1},
            }
        )

    def test_masked_entries_are_excluded_from_denominator(self):
        decoded = torch.zeros(1, 2, 1)
        target = torch.tensor([[[[1.0]], [[100.0]]]])
        mask = torch.tensor([[True, False]])
        loss = self.model._action_decoding_loss(decoded, target, loss_mask=mask)
        self.assertAlmostEqual(float(loss), 1.0)

    def test_multi_morphology_loss_is_transition_weighted(self):
        actions = torch.tensor([[[[1.0]]], [[[1.0]]], [[[9.0]]]])
        decoded = {
            "0": {
                "mask": torch.tensor([True, True, False]),
                "actions": torch.zeros(2, 1, 1),
            },
            "2": {
                "mask": torch.tensor([False, False, True]),
                "actions": torch.zeros(1, 1, 1),
            },
        }
        loss = self.model._multi_action_decoding_loss(decoded, actions)
        self.assertAlmostEqual(float(loss), 11.0 / 3.0, places=6)

    def test_future_actions_start_at_last_history_transition(self):
        self.model.num_history_frames = 5
        self.model.num_future_frames = 8
        actions = torch.arange(13).view(1, 13, 1, 1)
        aligned = self.model._future_action_chunks(actions)
        self.assertEqual(aligned.flatten().tolist(), list(range(4, 12)))

        # Transition chunk 4 targets future frame 5, so mask alignment starts
        # one index later than action-chunk alignment.
        mask = torch.tensor([[False, False, False, False, False, True, True, True, True, True, True, True, False]])
        aligned_mask = self.model._future_target_mask(mask)
        self.assertEqual(
            aligned_mask.flatten().tolist(),
            [True, True, True, True, True, True, True, False],
        )

    def test_split_decoder_target_is_component_major(self):
        self.model.action_decoder = SimpleNamespace(
            split_type={
                "action_type_split": {
                    "ee": [0, 1],
                    "camera": [1, 2],
                },
                # Two low-level steps: EE has two values/step, camera one.
                "0": {"ee": 4, "camera": 2},
            }
        )
        # Raw layout is step-major:
        #   step 0 = [ee0, ee1, camera], step 1 = [ee0, ee1, camera].
        actions = torch.tensor([[[[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]]]])
        predicted_component_major = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 10.0, 20.0]]])
        decoded = {
            "0": {
                "mask": torch.tensor([True]),
                "actions": predicted_component_major,
            }
        }
        loss = self.model._multi_action_decoding_loss(decoded, actions)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
