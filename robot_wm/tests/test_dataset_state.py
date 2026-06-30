import unittest
import pickle

import torch

from robot_wm.datasets.abc.transform import ABCTransform
from robot_wm.datasets.agibot.transform import AgibotTransform
from robot_wm.datasets.agios.transform import AgiosTransform
from robot_wm.datasets.droid.transform import DroidTransform
from robot_wm.datasets.base import Dataset
from robot_wm.datasets.multi_dataset import MultiDataset


class _DummyDataset(Dataset):
    @property
    def name(self):
        return "DummyDataset"

    def _get_length(self):
        return 1

    def __len__(self):
        return self._get_length()

    def _get_sample(self, index):
        return {"index": index}


class DatasetStateTest(unittest.TestCase):
    def test_multi_dataset_restores_child_retry_generator(self):
        child = _DummyDataset(seed=17)
        dataset = MultiDataset(datasets={"child": child}, seed=5)
        _ = torch.randint(0, 100, (4,), generator=child._gen)
        state = dataset.state_dict()
        expected = torch.randint(0, 100, (4,), generator=child._gen)

        dataset.load_state_dict(state)
        actual = torch.randint(0, 100, (4,), generator=child._gen)

        self.assertTrue(torch.equal(actual, expected))

    def test_spawn_pickle_preserves_parent_and_child_generator_streams(self):
        child = _DummyDataset(seed=17)
        dataset = MultiDataset(datasets={"child": child}, seed=5)
        _ = torch.randint(0, 100, (4,), generator=dataset._gen)
        _ = torch.randint(0, 100, (4,), generator=dataset._augment_gen)
        _ = torch.randint(0, 100, (4,), generator=child._gen)

        restored = pickle.loads(pickle.dumps(dataset))

        for original, copy in (
            (dataset._gen, restored._gen),
            (dataset._augment_gen, restored._augment_gen),
            (child._gen, restored.datasets["child"]._gen),
        ):
            expected = torch.randint(0, 100, (8,), generator=original)
            actual = torch.randint(0, 100, (8,), generator=copy)
            self.assertTrue(torch.equal(actual, expected))

    def test_abc_transform_generator_round_trip(self):
        transform = ABCTransform(
            cameras=["top"],
            output_keys=["rgb", "actions", "mask"],
            sample_size=13,
            chunk_size=5,
            seed=123,
        )
        _ = torch.randint(0, 100, (8,), generator=transform._gen)
        state = transform.state_dict()
        expected = torch.randint(0, 100, (8,), generator=transform._gen)

        transform.load_state_dict(state)
        actual = torch.randint(0, 100, (8,), generator=transform._gen)

        self.assertTrue(torch.equal(actual, expected))

    def test_abc_transform_is_spawn_picklable(self):
        transform = ABCTransform(
            cameras=["top"],
            output_keys=["rgb", "actions", "mask"],
            sample_size=13,
            chunk_size=5,
        )
        restored = pickle.loads(pickle.dumps(transform))
        self.assertEqual(restored._sample_size, 13)

    def test_all_active_transforms_are_spawn_picklable(self):
        common = {
            "output_keys": ["rgb", "actions", "mask"],
            "sample_size": 13,
            "chunk_size": 5,
            "resize_to": [180, 320],
        }
        transforms = [
            DroidTransform(cameras=["exterior_image_1_left"], **common),
            AgiosTransform(cameras=["ego_centric_image"], **common),
            AgibotTransform(cameras=["head_color"], **common),
        ]
        for transform in transforms:
            restored = pickle.loads(pickle.dumps(transform))
            self.assertEqual(restored._sample_size, 13)


if __name__ == "__main__":
    unittest.main()
