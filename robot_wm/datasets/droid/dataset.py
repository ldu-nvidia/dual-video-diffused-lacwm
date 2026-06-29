import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import numpy as np
import torch

from robot_wm.datasets.base import Dataset, _h5py_to_dict
from robot_wm.datasets.droid.transform import DroidTransform

logger = logging.getLogger(__name__)


class DroidDataset(Dataset):
    def __init__(
        self,
        manifest: Union[str, Path],
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[DroidTransform] = None,
        to_numpy: bool = False,
        normalize_keys: Optional[list[str]] = None,
        normalization_type: str = "mean",
        statistic_manifest: Optional[Union[str, Path]] = None,
        random_mask_camera: float = 0.0,
        subsample_traj: Optional[int] = None,
    ):
        self.normalize_keys = normalize_keys
        if self.normalize_keys is not None:
            self.preprocessing = True
        self.random_mask_camera = random_mask_camera
        if statistic_manifest is None:
            statistic_manifest = manifest

        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self._to_numpy = to_numpy

        self.stats = None
        self.normalization_type = normalization_type
        overwrite_camera_motion = (
            "camera" in self._transform._action_type
            and "right_wrist_rgb" in self._transform._cameras
        )
        if self.normalize_keys is not None:
            stats_path = self.compute_and_save_stats(
                statistic_manifest,
                self.normalize_keys,
                overwrite_camera_motion=overwrite_camera_motion,
            )
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
        self.preprocessing = False
        if self._transform is not None and self.stats is not None:
            self._transform.quantile_range = {
                key: [self.stats[key]["q1"], self.stats[key]["q99"]]
                for key in self.normalize_keys
            }

        self._manifest = manifest
        self._paths = self._load_manifest(
            manifest
        )  # override the path, otherwise it would be the statistic manifest loaded in compute_and_save_stats
        if subsample_traj is not None and subsample_traj > 0:
            self._paths = self._paths[:subsample_traj]
            logger.info(
                f"Subsampled trajectories to {subsample_traj}, new dataset size: {len(self._paths):,}"
            )
        self.ee_action_dim = 10
        logger.info(f"full dataset size ({manifest}): {len(self._paths):,}")

    def _load_manifest(
        self, manifest: Union[str, Path], fix_paths: bool = True
    ) -> Sequence[Path]:
        with Path(manifest).open() as f:
            reader = csv.reader(f, delimiter=" ")
            paths = [Path(row[0]) / "episode.h5" for row in reader]
        if fix_paths:
            logger.warning(f"fixing paths in manifest file ({manifest})")
            paths = [Path(str(p).replace("011925", "011825")) for p in paths]
        return paths

    @property
    def name(self) -> str:
        return "DroidDataset"

    def _get_length(self) -> int:
        return len(self._paths)

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample(self, index: int) -> dict[str, Any]:
        path = self._paths[index]
        with h5py.File(path, "r") as h5_file:
            if self._to_numpy:
                episode = _h5py_to_dict(h5_file)
            else:
                episode = h5_file
            if self._transform is not None:
                if self.preprocessing:
                    episode, camera = self._transform(episode, self.normalize_keys)
                else:
                    episode, camera = self._transform(episode)

        episode["dataset_index"] = torch.tensor(0, dtype=torch.long)

        assert (
            episode["actions"].shape[-1] == self.ee_action_dim + 9
        ), f"Expected action shape to be {self.ee_action_dim + 9}, got {episode['actions'].shape[-1]}"

        # if sampled writs camera then normalize the wrist camera, otherwise no normalization

        if self.stats is not None:
            for key in self.normalize_keys:
                x = episode[key]
                if camera == "wrist_image_left":
                    action_dim = x.shape[-1]
                else:
                    action_dim = self.ee_action_dim

                if self.normalization_type == "mean":
                    episode[key][..., :action_dim] = (
                        x[..., :action_dim]
                        - np.array(self.stats[key]["mean"][:action_dim])
                    ) / (np.array(self.stats[key]["std"][:action_dim]) + 1e-8)
                elif self.normalization_type == "quantile":
                    q1, q99 = np.array(self.stats[key]["q1"][:action_dim]), np.array(
                        self.stats[key]["q99"][:action_dim]
                    )
                    episode[key][..., :action_dim] = (x[..., :action_dim] - q1) / (
                        q99 - q1 + 1e-6
                    ) * 2.0 - 1.0
                else:
                    raise NotImplementedError(
                        f"Unsupported normalization: {self.normalization_type}"
                    )

        episode["camera_index"] = torch.zeros(
            *(episode["actions"].shape[:-1]), dtype=torch.long
        )
        if self.random_mask_camera > 0.0:
            mask = (
                torch.rand(*(episode["actions"].shape[:-1])) > self.random_mask_camera
            )
            episode["camera_mask"] = mask.float()
        else:
            episode["camera_mask"] = torch.ones(*(episode["actions"].shape[:-1]))

        return episode

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self._get_sample(index)
        return episode


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    transform = DroidTransform(
        cameras=["exterior_image_1_left", "exterior_image_2_left"],
        output_keys=["rgb", "actions"],
        sample_size=9,
        chunk_size=5,
    )

    # example usage
    dataset = DroidDataset(
        "data/droid/episode_paths.csv",
        transform=transform,
        normalize_keys=["actions"],
        statistic_manifest="data/droid/episode_paths.csv",
    )
    N = dataset._get_length()
    print(f"{N = :,}")
    # select a random episode
    idx = np.random.choice(N)
    episode = dataset._get_sample(idx)
    episode_path = str(dataset._paths[idx])
    print(f"{idx = }")
    print(f"{str(episode_path) = }")
    # parse episode
    exterior_image_1_left = episode["episode_data"]["observation"][
        "exterior_image_1_left"
    ]
    exterior_image_2_left = episode["episode_data"]["observation"][
        "exterior_image_2_left"
    ]
    wrist_image_left = episode["episode_data"]["observation"]["wrist_image_left"]
    state = episode["episode_data"]["observation"]["cartesian_position"]
    action = episode["episode_data"]["action"]
    gripper = action[:, -1]
    T = 5
    # select T timesteps
    indices = np.linspace(0, len(exterior_image_1_left), num=T + 2)[1:-1]
    indices = indices.astype(int).tolist()
    # show timestep
    fig, axs = plt.subplots(T, 3, figsize=(15, T * 3))
    _ = [ax.axis("off") for ax in axs.flat]
    for ax_idx, idx in enumerate(indices):
        _ = axs[ax_idx, 0].imshow(exterior_image_1_left[idx])
        _ = axs[ax_idx, 1].imshow(exterior_image_2_left[idx])
        _ = axs[ax_idx, 2].imshow(wrist_image_left[idx])
        _ = axs[ax_idx, 1].set_title(f"t = {idx}")
        _ = axs[ax_idx, 2].set_title(f"gripper: {gripper[idx]:0.2f}")
    # show state and action trajectories
    _, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    _ = ax.plot(state[:, 0], state[:, 1], state[:, 2])
    _ = ax.plot(action[:, 0], action[:, 1], action[:, 2])
    plt.show()
