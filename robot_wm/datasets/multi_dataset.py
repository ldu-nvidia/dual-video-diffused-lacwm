# a dataset that contains multiple datasets such as the droid dataset and the egodex dataset, it needs to take in args for each dataset and then combine them
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F

from robot_wm.datasets.base import Dataset

Morphology_Mapping = {
    "MPKDataset": 0,
    "Hot3dDataset": 1,
    "EgoDexDataset": 2,
    "MurpDataset": 4,
    "ManiSkillDataset": 5,
    "AgibotDataset": 6,
    "SimplerDataset": 7,
    "FrankasimDataset": 8,
    "LiberoDataset": 5,
    "DroidLeRobotDataset": 0,
    "ABCDataset": 9,
}


def random_crop_resize_same(batch: torch.Tensor, ratio_range=(0.6, 1.0)):
    """
    Apply the same random crop (based on a sampled ratio) to all images in the batch,
    then resize them back to original size.

    Args:
        batch: (B, C, H, W)
        ratio_range: tuple of (min_ratio, max_ratio)

    Returns:
        Transformed batch: (B, C, H, W)
    """
    B, C, H, W = batch.shape

    # Sample crop ratio
    ratio = torch.empty(1).uniform_(*ratio_range).item()
    crop_h = int(H * ratio)
    crop_w = int(W * ratio)

    # Sample top-left crop corner (same for all images)
    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()

    # Crop and resize each image
    cropped = batch[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

    return resized


class MultiDataset(Dataset):
    """
    A dataset that contains multiple datasets such as the droid dataset and the egodex dataset, it needs to take in args for each dataset and then combine them, an example is:
    datasets:
        droid:
            _target_: robot_wm.datasets.droid.lerobot_dataset.DroidLeRobotDataset
            data_dir: /path/to/droid_lerobot
            seed: 0
            infinite: True
            transform: None
            to_numpy: False
        egodex:
            _target_: robot_wm.datasets.egodex.dataset.EgoDexDataset
            manifest: /path/to/egodex/manifest
            seed: 0
            infinite: True
            transform: None
            to_numpy: False
    """

    def __init__(
        self,
        datasets: dict,
        transform: Optional[dict] = None,
        seed: int = 0,
        infinite: bool = True,
        padding_dim: int = 0,
        img_augment: bool = False,
        emit_crop_rgb: bool = True,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.datasets = datasets

        self.dataset_lengths = [len(dataset) for dataset in self.datasets.values()]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        for name, dataset in self.datasets.items():
            logger.info(f"MultiDataset: {name} has {len(dataset):,} episodes")
        logger.info(f"MultiDataset: total {self.cumulative_lengths[-1]:,} episodes")
        self.padding_dim = padding_dim
        self.img_augment = img_augment
        self.emit_crop_rgb = emit_crop_rgb
        self._augment_gen = torch.Generator()
        self._augment_initialized = False
        self.augment = torch.nn.Sequential(
            K.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=1.0,
                same_on_batch=True,
            ),
            K.RandomGamma(
                gamma=(0.7, 1.5), p=0.5, same_on_batch=True
            ),  # simulates lighting change
        )
        self.rot_augment = K.RandomRotation(degrees=20, same_on_batch=True, p=1.0)

    @property
    def name(self) -> str:
        return "MultiDataset"

    @property
    def full_name(self) -> str:
        all_dataset_names = list(self.datasets.keys())
        return "MultiDataset" + " - ".join(all_dataset_names)

    def _get_length(self):
        return self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample(self, index):
        # fine the which dataset the index is in using the dataset_lengths
        dataset_index = np.searchsorted(self.cumulative_lengths, index, "right")
        # get the index of the sample in the dataset

        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumulative_lengths[dataset_index - 1]

        dataset = self.datasets[list(self.datasets.keys())[dataset_index]]
        episode = dataset._get_sample(sample_index)
        _camera = None
        if isinstance(episode, tuple):
            episode, _camera = episode
        episode["dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)

        # if camera index is not in the episode, add it
        if "camera_index" not in episode:
            episode["camera_index"] = torch.zeros(
                *(episode["actions"].shape[:-1]), dtype=torch.long
            )
        # if camera mask is not in the episode, add it
        if "camera_mask" not in episode:
            episode["camera_mask"] = torch.ones(*(episode["actions"].shape[:-1]))

        if dataset.ee_action_dim < self.padding_dim:
            # episode["actions"] is a tensor of shape (N, T, D), padding at last dim
            # create a tensor of shape (N, T, self.padding_dim - action_dim)
            padding = torch.zeros(
                *(episode["actions"].shape[:-1]),
                self.padding_dim - episode["actions"].shape[-1],
            )
            episode["actions"] = torch.cat(
                [
                    episode["actions"],
                    padding,
                ],
                dim=-1,
            )

        assert dataset.name in Morphology_Mapping, (
            f"Dataset {dataset.name} not found in Morphology_Mapping. "
            f"Available datasets: {list(Morphology_Mapping.keys())}"
        )
        morphology_index = Morphology_Mapping[dataset.name]

        episode["morphology_index"] = torch.tensor(
            morphology_index, dtype=torch.long
        )  # Add morphology as a tensor

        episode["ee_action_dim"] = torch.tensor(
            dataset.ee_action_dim + 9, dtype=torch.int
        )

        episode["decode_camera"] = getattr(dataset, "decode_camera", True)

        if self.img_augment and "rgb" in episode:
            original_rgb = episode["rgb"]
            # rgb is normalized to [-1,1] (mean=std=0.5). kornia ColorJitter/RandomGamma assume
            # [0,1] input (they clamp to [0,1]), so augment in [0,1] -- i.e. BEFORE normalization --
            # then re-normalize back to [-1,1]. (Augmenting the normalized [-1,1] tensor directly
            # collapses the range to [0,1] and feeds the VAE out-of-distribution data.)
            if not self._augment_initialized:
                worker = torch.utils.data.get_worker_info()
                worker_id = worker.id if worker is not None else 0
                self._augment_gen.manual_seed(
                    self._seed + 1_000_003 * self._process_id + 9_176 * worker_id
                )
                self._augment_initialized = True
            # Kornia samples from the worker-global CPU RNG. Isolate it behind a
            # stateful generator so checkpointed loader workers can resume the
            # exact augmentation stream.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._augment_gen.get_state())
                rgb01 = ((original_rgb + 1.0) * 0.5).clamp(0.0, 1.0)
                episode["rgb"] = self.augment(rgb01) * 2.0 - 1.0
                if self.emit_crop_rgb:
                    episode["crop_rgb"] = random_crop_resize_same(
                        self.rot_augment(self.augment(rgb01)) * 2.0 - 1.0
                    )
                self._augment_gen.set_state(torch.get_rng_state())
            # mediapy.write_video(
            #     f"augmented_video_cropped_{index}.mp4",
            #     episode["crop_rgb"].permute(0, 2, 3, 1).numpy(),
            #     fps=30,
            # )

        return episode

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self._get_sample(index)
        return episode

    def state_dict(self) -> dict[str, Any]:
        output = {
            "_start_idx": self._start_idx,
            "_gen": self._gen.get_state(),
            "_augment_gen": self._augment_gen.get_state(),
            "_augment_initialized": self._augment_initialized,
            "_datasets": {},
        }
        for dataset_name, dataset in self.datasets.items():
            # Child dataset state includes its retry RNG and transform RNG. The
            # retry generator is consumed only on decode/sample failures, but it
            # must still continue exactly after a checkpoint.
            output["_datasets"][dataset_name] = dataset.state_dict()
        return output

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_idx = state_dict["_start_idx"]
        self._gen.set_state(state_dict["_gen"])
        if "_augment_gen" in state_dict:
            self._augment_gen.set_state(state_dict["_augment_gen"])
            self._augment_initialized = bool(
                state_dict.get("_augment_initialized", True)
            )
        child_states = state_dict.get("_datasets")
        if not isinstance(child_states, dict) or set(child_states) != set(self.datasets):
            raise RuntimeError("checkpoint child-dataset state does not match MultiDataset")
        for dataset_name, dataset in self.datasets.items():
            dataset.load_state_dict(child_states[dataset_name])
