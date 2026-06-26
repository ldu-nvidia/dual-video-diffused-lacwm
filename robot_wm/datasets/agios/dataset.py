import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import numpy as np
import torch

from robot_wm.datasets.agios.transform import AgiosTransform
from robot_wm.datasets.base import Dataset, _h5py_to_dict

logger = logging.getLogger(__name__)


class AgiosDataset(Dataset):
    def __init__(
        self,
        manifest: Union[str, Path],
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[AgiosTransform] = None,
        to_numpy: bool = False,
        normalize_keys: Optional[list[str]] = None,
        normalization_type: str = "mean",
        statistic_manifest: Optional[Union[str, Path]] = None,
        stats_path: Optional[Union[str, Path]] = None,
    ):
        self.normalize_keys = normalize_keys
        if self.normalize_keys is not None:
            self.preprocessing = True
        if statistic_manifest is None:
            statistic_manifest = manifest

        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self._to_numpy = to_numpy

        self.stats = None
        self.normalization_type = normalization_type
        if self.normalize_keys is not None:
            if stats_path is None:
                stats_path = self.compute_and_save_stats(
                    statistic_manifest,
                    self.normalize_keys,
                    interpolation_step=transform.interpolation_step,
                )
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
        self.preprocessing = False
        if self._transform is not None and self.stats is not None:
            self._transform.quantile_range = {
                key: [self.stats[key]["q1"], self.stats[key]["q99"]]
                for key in self.normalize_keys
            }

        self._paths = self._load_manifest(
            manifest
        )  # override the path, otherwise it would be the statistic manifest loaded in compute_and_save_stats
        self.ee_action_dim = 72
        self.action_norm_dim = 72 if "touch" in self._transform._action_type else 62
        logger.info(f"full dataset size ({manifest}): {len(self._paths):,}")

    def _load_manifest(
        self, manifest: Union[str, Path], fix_paths: bool = False
    ) -> Sequence[Path]:
        with Path(manifest).open() as f:
            reader = csv.reader(f, delimiter=" ")
            paths = [Path(row[0]) / "episode.h5" for row in reader]

        return paths

    @property
    def name(self) -> str:
        return "AgiosDataset"

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
                    episode = self._transform(
                        episode, self.normalize_keys, include_touch=True
                    )
                else:
                    episode = self._transform(episode)

        if self.stats is not None:
            for key in self.normalize_keys:
                x = episode[key]
                action_dim = self.action_norm_dim
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
        episode["camera_mask"] = torch.zeros(*(episode["actions"].shape[:-1]))

        return episode

    def __getitem__(self, index: int) -> dict[str, any]:
        episode = self._get_sample(index)
        return episode


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    transform = AgiosTransform(
        cameras=["ego_centric_image"],
        output_keys=["rgb", "actions"],
        sample_size=9,
        chunk_size=5,
    )
    # example usage
    dataset = AgiosDataset(
        "/home/ravenhuang/h2r/robot_world_models/data/datasets/agios/agios_dataset_h5/all_episode_paths.csv",
        transform=transform,
        normalize_keys=["actions"],
    )
    N = dataset._get_length()
    print(f"{N = :,}")
    # select a random episode
    idx = np.random.choice(N)
    episode = dataset._get_sample(idx)
    episode = dataset.__getitem__(idx)
    episode_path = str(dataset._paths[idx])
    print(f"{idx = }")
    print(f"{str(episode_path) = }")
    # parse episode
    exterior_image_1_left = episode["rgb"]
    actions = episode["actions"]
    T = 5
    # select T timesteps
    indices = np.linspace(0, len(exterior_image_1_left), num=T + 2)[1:-1]
    indices = indices.astype(int).tolist()
    # show timestep
    fig, axs = plt.subplots(T, 3, figsize=(15, T * 3))
    _ = [ax.axis("off") for ax in axs.flat]
    for ax_idx, idx in enumerate(indices):
        _ = axs[ax_idx, 0].imshow(exterior_image_1_left[idx].permute(1, 2, 0))
        _ = axs[ax_idx, 1].set_title(f"t = {idx}")
        # _ = axs[ax_idx, 2].set_title(f"gripper: {gripper[idx]:0.2f}")
    # show state and action trajectories
    plt.savefig("test_obs.png")
    _, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    _ = ax.plot(actions[:, 0], actions[:, 1], actions[:, 2])
    # _ = ax.plot(action[:, 0], action[:, 1], action[:, 2])
    plt.savefig("test.png")
