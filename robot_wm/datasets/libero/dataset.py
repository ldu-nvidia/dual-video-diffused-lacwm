import csv
import json
import logging
import os
import random
from typing import Any, Optional

import h5py
import numpy as np

from robot_wm.datasets.base import Dataset
from robot_wm.datasets.libero.transform import LiberoTransform

logger = logging.getLogger(__name__)

CAMERAS = [
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
]


class LiberoDataset(Dataset):
    def __init__(
        self,
        manifest: Optional[str] = None,
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[LiberoTransform] = None,
        subsample_traj: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.manifest = manifest
        self.all_task_episodes = self.get_manifest(self.manifest)
        if subsample_traj is not None:
            self.all_task_episodes = self.all_task_episodes[:subsample_traj]
        logger.info(f"{self.manifest = }")
        self.ee_action_dim = 7  # dx, dy, dz, rx, ry, rz, gripper
        if "joint_states" in transform._action_type:
            self.ee_action_dim += 7

        self.sample_zero_action = kwargs.get("sample_zero_action", False)

    @staticmethod
    def get_manifest(csv_path: str):
        """
        Get a list of task_id and episode_id pairs, these is how the data is stored.
        It will return a list of tuples, where each tuple contains a task_id and an episode_id.

        Args:
            csv_path (str): The base directory path.

        Returns:
            List[tuple]: A list of (task_id, episode_id,dataset) pairs.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Manifest file not found: {csv_path}")
        with open(csv_path, mode="r") as f:
            reader = csv.reader(f, delimiter=" ")
            next(reader)  # Skip the header line
            task_episode_pairs = [row[0].split(",") for row in reader]
        return task_episode_pairs

    @staticmethod
    def get_proprio(src_path: str, episode_id: str):
        """Get proprioception data for a specific task_id and episode_id."""
        with h5py.File(src_path, "r") as f:
            episode = f["data"][episode_id]
            return {
                "timestamp": np.array(episode["obs/joint_states"].shape[0]),
                "agentview_rgb": np.array(episode["obs/agentview_rgb"])[
                    :, ::-1, ::-1
                ],  # consistent with pi training
                "wrist_rgb": np.array(episode["obs/eye_in_hand_rgb"])[
                    :, ::-1, ::-1
                ],  # consistent with pi training
                "ee_states": np.array(episode["obs/ee_states"]),  # pos + axis_angle
                "joint_states": np.array(episode["obs/joint_states"]),
                "gripper_states": np.array(episode["obs/gripper_states"][:, 0])
                - np.array(episode["obs/gripper_states"][:, 1]),
                "action": np.array(
                    episode["actions"]
                ),  # position + axis_angle + gripper, 7 dof
            }

    @property
    def name(self) -> str:
        return "LiberoDataset"

    def _get_length(self) -> int:
        return len(self.all_task_episodes)

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample_from_ids(self, src_path, episode_id) -> dict[str, Any]:
        toret = dict()
        toret["episode_id"] = episode_id
        toret["obs"] = self.get_proprio(src_path, episode_id)
        return toret

    def __get_sample(self, index: int) -> dict[str, Any]:
        src_path, episode_id = self.all_task_episodes[index]
        episode = dict()
        episode["episode_id"] = episode_id
        episode["obs"] = self.get_proprio(src_path, episode_id)
        if self._transform is not None:
            episode = self._transform(episode)
        if self.sample_zero_action:
            if random.random() < 0.1:
                episode["actions"] = episode["zero_action"]
                episode["rgb"][:] = episode["rgb"][0:1]
        return episode

    def _get_sample(self, index: int) -> dict[str, Any]:
        return self.__get_sample(index)
        # try:
        #     return self.__get_sample(index)
        # except Exception as e:
        #     new_index = np.random.choice(self._get_length())
        #     logger.warning(f"{index = } failed; using {new_index = }; {e = }")
        #     return self._get_sample(new_index)

    def __getitem__(self, index):
        return self._get_sample(index)


if __name__ == "__main__":
    CSV_PATH = "/home/ravenh/lacwm-dit/robot_wm/datasets/csv_files/libero.csv"
    transform = LiberoTransform(
        cameras=["agentview_rgb", "wrist_rgb"],
        output_keys=["rgb", "actions", "mask", "zero_action"],
        sample_size=9,
        chunk_size=5,
        resize_to=[256, 256],
        action_type="action+camera",
        multiview=False,
    )
    dataset = LiberoDataset(CSV_PATH, sample_zero_action=True, transform=transform)
    idx = np.random.choice(len(dataset))
    sample = dataset[idx]
    print(f"{sample.keys() = }")
    print(f"{sample['rgb'].shape = }")
    print(f"{sample['rgb'].max() = }")
    print(f"{sample['rgb'].min() = }")
    print(f"{sample['actions'].shape = }")
    print(f"{sample['mask'].shape = }")
    # save the sampled rgb to a file
    import imageio

    imageio.imwrite(
        "sampled_rgb.png",
        (sample["rgb"][0].permute(1, 2, 0) * 255.0).numpy().astype(np.uint8),
    )
