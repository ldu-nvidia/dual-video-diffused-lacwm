import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robot_wm.datasets.agios.transform import AgiosTransform
from robot_wm.datasets.base import Dataset
from robot_wm.datasets.utils import euler_to_rot_6d

logger = logging.getLogger(__name__)


# ALL_JOINTS = ['camera', 'hip', 'leftArm', 'leftForearm', 'leftHand', 'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 'leftIndexFingerKnuckle', 'leftIndexFingerMetacarpal', 'leftIndexFingerTip', 'leftLittleFingerIntermediateBase', 'leftLittleFingerIntermediateTip', 'leftLittleFingerKnuckle', 'leftLittleFingerMetacarpal', 'leftLittleFingerTip', 'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip', 'leftMiddleFingerKnuckle', 'leftMiddleFingerMetacarpal', 'leftMiddleFingerTip', 'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip', 'leftRingFingerKnuckle', 'leftRingFingerMetacarpal', 'leftRingFingerTip', 'leftShoulder', 'leftThumbIntermediateBase', 'leftThumbIntermediateTip', 'leftThumbKnuckle', 'leftThumbTip', 'neck1', 'neck2', 'neck3', 'neck4', 'rightArm', 'rightForearm', 'rightHand', 'rightIndexFingerIntermediateBase', 'rightIndexFingerIntermediateTip', 'rightIndexFingerKnuckle', 'rightIndexFingerMetacarpal', 'rightIndexFingerTip', 'rightLittleFingerIntermediateBase', 'rightLittleFingerIntermediateTip', 'rightLittleFingerKnuckle', 'rightLittleFingerMetacarpal', 'rightLittleFingerTip', 'rightMiddleFingerIntermediateBase', 'rightMiddleFingerIntermediateTip', 'rightMiddleFingerKnuckle', 'rightMiddleFingerMetacarpal', 'rightMiddleFingerTip', 'rightRingFingerIntermediateBase', 'rightRingFingerIntermediateTip', 'rightRingFingerKnuckle', 'rightRingFingerMetacarpal', 'rightRingFingerTip', 'rightShoulder', 'rightThumbIntermediateBase', 'rightThumbIntermediateTip', 'rightThumbKnuckle', 'rightThumbTip', 'spine1', 'spine2', 'spine3', 'spine4', 'spine5', 'spine6', 'spine7']

# TODO: move to a file
LEFT_HAND_JOINTS = [
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbTip",
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]
RIGHT_HAND_JOINTS = [
    "rightThumbKnuckle",  # 22
    "rightThumbIntermediateBase",  # 20
    "rightThumbIntermediateTip",  # 21
    "rightThumbTip",  # 23
    "rightIndexFingerKnuckle",  # 2
    "rightIndexFingerIntermediateBase",  # 0
    "rightIndexFingerIntermediateTip",  # 1
    "rightIndexFingerTip",  # 4
    "rightMiddleFingerKnuckle",  # 12
    "rightMiddleFingerIntermediateBase",  # 10
    "rightMiddleFingerIntermediateTip",  # 11
    "rightMiddleFingerTip",  # 14
    "rightRingFingerKnuckle",  # 17
    "rightRingFingerIntermediateBase",  # 15
    "rightRingFingerIntermediateTip",  # 16
    "rightRingFingerTip",  # 19
    "rightLittleFingerKnuckle",  # 7
    "rightLittleFingerIntermediateBase",  # 5
    "rightLittleFingerIntermediateTip",  # 6
    "rightLittleFingerTip",  # 9
]


STATE_ROT_JOINTS = ["camera", "leftHand", "rightHand"]


def _h5py_to_dict(file: Union[h5py.File, h5py.Group]) -> dict:
    def _str_or_numpy(
        item: h5py.Dataset, encoding: str = "utf-8"
    ) -> Union[str, np.ndarray]:
        return item.asstr(encoding)[()] if item.dtype == "object" else np.array(item)

    output = {}
    for key in file:
        if isinstance(file[key], h5py.Dataset):
            output[key] = _str_or_numpy(file[key])
        else:
            output[key] = _h5py_to_dict(file[key])
    return output


class EgoDexDataset(Dataset):
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
        subsample_traj: Optional[int] = None,
    ):
        self.normalize_keys = normalize_keys
        if self.normalize_keys is not None:
            self.preprocessing = True
        if statistic_manifest is None:
            statistic_manifest = manifest

        transform.hand_state_dim = 9 + 20 * 3
        transform.hand_pos_scale = 1.0  # scale for hand positions, used in

        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self._manifest = manifest
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

        self.action_norm_dim = 138
        self.ee_action_dim = 148
        self._paths = self._load_manifest(manifest)
        if subsample_traj is not None:
            self._paths = self._paths[:subsample_traj]
        assert transform._action_type in [
            "state",
            "delta-state",
            "delta-state+camera",
            "delta-state+camera+abs_finger",
        ]

        logger.info(f"full dataset size ({manifest}): {len(self._paths):,}")

    def _load_manifest(self, manifest: Union[str, Path]) -> Sequence[Path]:
        with Path(manifest).open() as f:
            reader = csv.reader(f, delimiter=" ")
            paths = [Path(row[0]) for row in reader]
        return paths

    @property
    def name(self) -> str:
        return "EgoDexDataset"

    def _get_length(self) -> int:
        return len(self._paths)

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample(self, index: int) -> dict[str, Any]:
        h5_path = self._paths[index]
        episode = h5py.File(h5_path)
        if not self.preprocessing:
            video_path = h5_path.with_suffix(".mp4")
            frames = []
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(frame, (640, 360))[..., ::-1])

            if len(frames) == 0:
                logger.warning(f"Failed to get sample at index {index}")
                new_index = int(
                    torch.randint(0, len(self), (1,), generator=self._gen).item()
                )
                return self._get_sample(new_index)

            frames = np.stack(frames)
        else:
            frames = None

        poses_6d = {}
        for joint in STATE_ROT_JOINTS:
            pose_matrix = episode["transforms"][joint]
            poses_6d[joint] = np.concatenate(
                [
                    pose_matrix[:, :3, 3],
                    euler_to_rot_6d(
                        Rotation.from_matrix(pose_matrix[:, :3, :3]).as_euler(
                            "xyz", degrees=False
                        )
                    ),
                ],
                axis=-1,
            )
        left_hand_states = np.concatenate(
            [episode["transforms"][joint][:, :3, 3] for joint in LEFT_HAND_JOINTS],
            axis=-1,
        )
        right_hand_states = np.concatenate(
            [episode["transforms"][joint][:, :3, 3] for joint in RIGHT_HAND_JOINTS],
            axis=-1,
        )

        episode = {
            "episode_data": {
                "observation": {
                    "ego_centric_image": frames,
                    "ekf_hand_pose_left/WristState": poses_6d["leftHand"],
                    "ekf_hand_pose_right/WristState": poses_6d["rightHand"],
                    "ekf_hand_pose_left/JointAngles": left_hand_states,
                    "ekf_hand_pose_right/JointAngles": right_hand_states,
                    "camera": poses_6d["camera"],
                }
            },
            "action": None,  # Will be filled in by transform
            # TODO: switch to action space in Section 4.1 of EgoDex paper: https://arxiv.org/pdf/2505.11709
        }
        if self._to_numpy:
            episode = _h5py_to_dict(episode)
        if self._transform is not None:
            if self.preprocessing:
                episode = self._transform(episode, self.normalize_keys, skip_rgb=True)
            else:
                episode = self._transform(episode)

        assert (
            episode["actions"].shape[-1] == self.ee_action_dim + 9
        ), f"Expected action shape to be {self.ee_action_dim + 9}, got {episode['actions'].shape[-1]}"

        if self.stats is not None:
            for key in self.normalize_keys:
                x = episode[key]
                action_dim = self.action_norm_dim
                if self.normalization_type == "mean":
                    episode[key][..., :action_dim] = (
                        x[..., :action_dim]
                        - np.array(self.stats[key]["mean"][:action_dim])
                    ) / (
                        np.array(self.stats[key]["std"][:action_dim]) + 1e-8
                    )  # normalize the ee action

                    episode[key][..., self.ee_action_dim :] = (
                        x[..., self.ee_action_dim :]
                        - np.array(self.stats[key]["mean"][self.ee_action_dim :])
                    ) / (
                        np.array(self.stats[key]["std"][self.ee_action_dim :]) + 1e-8
                    )  # normalize the camera action

                elif self.normalization_type == "quantile":
                    q1, q99 = np.array(self.stats[key]["q1"]), np.array(
                        self.stats[key]["q99"]
                    )

                    episode[key][..., :action_dim] = (
                        x[..., :action_dim] - q1[:action_dim]
                    ) / (q99[:action_dim] - q1[:action_dim] + 1e-6) * 2.0 - 1.0

                    episode[key][..., self.ee_action_dim :] = (
                        x[..., self.ee_action_dim :] - q1[self.ee_action_dim :]
                    ) / (
                        q99[self.ee_action_dim :] - q1[self.ee_action_dim :] + 1e-6
                    ) * 2.0 - 1.0

                else:
                    raise NotImplementedError(
                        f"Unsupported normalization: {self.normalization_type}"
                    )

        episode["camera_index"] = torch.zeros(
            *(episode["actions"].shape[:-1]), dtype=torch.long
        )
        episode["camera_mask"] = torch.ones(*(episode["actions"].shape[:-1]))

        return episode

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get_sample(index)


if __name__ == "__main__":
    transform = AgiosTransform(
        cameras=["ego_centric_image"],
        output_keys=["rgb", "actions", "mask", "state", "goal"],
        sample_size=9,
        chunk_size=5,
        hand_state_dim=81,
        hand_pos_scale=1.0,
    )
    # example usage
    dataset = EgoDexDataset(
        "data/egodex/manifest.csv",
        transform=transform,
    )
    N = dataset._get_length()
    print(f"{N = :,}")
    # select a random episode
    idx = np.random.choice(N)
    episode = dataset._get_sample(idx)
    episode_path = str(dataset._paths[idx])
    print(f"{idx = }")
    print(f"{str(episode_path) = }")
