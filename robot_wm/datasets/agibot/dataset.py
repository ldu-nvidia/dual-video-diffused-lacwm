import csv
import json
import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

import h5py
import numpy as np
import torch

from robot_wm.datasets.agibot.transform import AgibotTransform
from robot_wm.datasets.base import Dataset

logger = logging.getLogger(__name__)


def _default_dataset_roots() -> dict[str, str]:
    """Resolve dataset roots from the environment at construction time."""
    agibot_data = os.environ.get("LACWM_DATA", "/scr/ravenh/lacwm_data")
    return {
        "alpha": os.environ.get("AGIBOT_ALPHA_ROOT", f"{agibot_data}/agibot"),
        "beta": os.environ.get("AGIBOT_BETA_ROOT", f"{agibot_data}/agibot"),
        "viscam": f"{agibot_data}/agibot_combined",
        "scr": f"{agibot_data}/agibot",
    }


# Retain the legacy module-level defaults for external imports. Dataset instances
# resolve the environment again so worker construction is not tied to import time.
DATASET_ROOTS = _default_dataset_roots()


class AgibotDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        manifest: Optional[str] = None,
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[AgibotTransform] = None,
        load_camera_params: bool = True,
        load_task_info: bool = False,
        subsample_traj: Optional[int] = None,
        dataset_roots: Optional[Mapping[str, os.PathLike | str]] = None,
        max_retries: int = 8,
        **kwargs,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        resolved_roots = _default_dataset_roots()
        if dataset_roots is not None:
            if not isinstance(dataset_roots, Mapping):
                raise TypeError(
                    "dataset_roots must be a mapping of dataset IDs to paths"
                )
            for dataset_id, root in dataset_roots.items():
                if not isinstance(dataset_id, str) or not dataset_id.strip():
                    raise ValueError("dataset_roots keys must be non-empty strings")
                try:
                    resolved_root = os.fspath(root)
                except TypeError as exc:
                    raise TypeError(
                        f"dataset root for {dataset_id!r} must be path-like"
                    ) from exc
                if not resolved_root:
                    raise ValueError(
                        f"dataset root for {dataset_id!r} must not be empty"
                    )
                resolved_roots[dataset_id] = resolved_root
        self.dataset_roots = resolved_roots
        self._max_retries = max_retries

        self.csv_path = csv_path
        if manifest is not None:
            self.csv_path = manifest
        if self.csv_path is None:
            raise ValueError("either csv_path or manifest must be provided")
        self.all_task_episodes = self.get_manifest(
            self.csv_path, allowed_dataset_ids=self.dataset_roots
        )
        if subsample_traj is not None:
            self.all_task_episodes = self.all_task_episodes[:subsample_traj]
        if not self.all_task_episodes:
            raise ValueError(f"AgiBot manifest contains no episodes: {self.csv_path}")
        self.load_camera_params = load_camera_params
        self.load_task_info = load_task_info
        logger.info(f"{self.csv_path = }")
        self.ee_action_dim = 34  # 20d ee (10/arm) + 14d joint (7/arm); +9 camera -> 43 total
        self.decode_camera = False

    @staticmethod
    def get_manifest(
        csv_path: str,
        allowed_dataset_ids: Optional[Mapping[str, object]] = None,
    ):
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
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            header = [value.strip() for value in next(reader)]
            if header != ["task_id", "episode_id", "dataset"]:
                raise ValueError(f"invalid AgiBot manifest header: {header}")
            task_episode_pairs = []
            for line_number, row in enumerate(reader, 2):
                values = [value.strip() for value in row]
                if len(values) != 3 or any(not value for value in values):
                    raise ValueError(
                        f"invalid AgiBot manifest row {line_number}: {row}"
                    )
                dataset_id = values[2]
                if (
                    allowed_dataset_ids is not None
                    and dataset_id not in allowed_dataset_ids
                ):
                    allowed = ", ".join(sorted(allowed_dataset_ids))
                    raise ValueError(
                        f"invalid AgiBot dataset ID {dataset_id!r} at manifest "
                        f"row {line_number}; configured IDs: {allowed}"
                    )
                task_episode_pairs.append(values)
        return task_episode_pairs

    @staticmethod
    def get_camera_params(src_path: str, task_id: str, episode_id: str):
        """Load JSON files and return their contents in a dictionary."""
        specific_files = [
            # "head_extrinsic_params.json",
            "head_extrinsic_params_aligned.json",
            # "head_intrinsic_params.json",
            # "hand_left_extrinsic_params.json",
            "hand_left_extrinsic_params_aligned.json",
            # "hand_left_intrinsic_params.json",
            # "hand_right_extrinsic_params.json",
            "hand_right_extrinsic_params_aligned.json",
            # "hand_right_intrinsic_params.json",
        ]
        folder_path = os.path.join(
            src_path, "parameters", task_id, episode_id, "parameters", "camera"
        )
        camera_params_dict = {}
        for file_name in specific_files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Camera params file not found: {file_path}")
            file_stem = file_name.replace(".json", "")
            with open(file_path, "r") as f:
                camera_params_dict[file_stem] = json.load(f)
        return camera_params_dict

    @staticmethod
    def get_video_paths(
        src_path: str,
        task_id: str,
        episode_id: str,
    ):
        """Get video path for a specific task_id and episode_id."""
        specific_files = [
            "head_color.mp4",
            "hand_left_color.mp4",
            "hand_right_color.mp4",
        ]
        folder_path = os.path.join(
            src_path, "observations", task_id, episode_id, "videos"
        )
        video_paths_dict = {}
        for file_name in specific_files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Video file not found: {file_path}")
            file_stem = file_name.replace(".mp4", "")
            video_paths_dict[file_stem] = file_path
        return video_paths_dict

    @staticmethod
    def get_task_info(src_path: str, task_id: str, episode_id: str):
        """Get task information for a specific task_id and episode_id."""
        task_info_path = os.path.join(src_path, f"task_info/task_{task_id}.json")
        if not os.path.exists(task_info_path):
            raise FileNotFoundError(f"Text goal file not found: {task_info_path}")
        with open(task_info_path, "r") as f:
            task_info = {str(item["episode_id"]): item for item in json.load(f)}
        task_info_of_episode = task_info.get(episode_id, None)
        return task_info_of_episode

    @staticmethod
    def get_proprio(src_path: str, task_id: str, episode_id: str):
        """Get proprioception data for a specific task_id and episode_id."""
        proprio_path = os.path.join(
            src_path, "proprio_stats", f"{task_id}", f"{episode_id}", "proprio_stats.h5"
        )
        if not os.path.exists(proprio_path):
            raise FileNotFoundError(f"Proprio file not found: {proprio_path}")
        with h5py.File(proprio_path, "r") as f:
            return {
                "timestamp": np.array(f["/timestamp"]),
                # state
                "state_effector_position": np.array(f["/state/effector/position"]),
                "state_end_orientation": np.array(f["/state/end/orientation"]),
                "state_end_position": np.array(f["/state/end/position"]),
                "state_head_position": np.array(f["/state/head/position"]),
                "state_joint_positon": np.array(f["state/joint/position"]),
                "state_robot_orientation": np.array(f["/state/robot/orientation"]),
                "state_robot_position": np.array(f["/state/robot/position"]),
                "state_waist_position": np.array(f["/state/waist/position"]),
                # action
                "action_effector_position": np.array(f["/action/effector/position"]),
                "action_end_orientation": np.array(f["/action/end/orientation"]),
                "action_end_position": np.array(f["/action/end/position"]),
                "action_head_position": np.array(f["/action/head/position"]),
                "action_joint_positon": np.array(f["action/joint/position"]),
                # "action_robot_orientation": np.array(f["/action/robot/orientation"]),
                # "action_robot_position": np.array(f["/action/robot/position"]),
                "action_waist_position": np.array(f["/action/waist/position"]),
            }

    @property
    def name(self) -> str:
        return "AgibotDataset"

    def _get_length(self) -> int:
        return len(self.all_task_episodes)

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample_from_ids(self, src_path, task_id, episode_id) -> dict[str, Any]:
        toret = dict()
        toret["task_id"] = task_id
        toret["episode_id"] = episode_id
        toret["proprio_stats"] = self.get_proprio(src_path, task_id, episode_id)
        if self.load_camera_params:
            toret["camera_params"] = self.get_camera_params(
                src_path, task_id, episode_id
            )
        if self.load_task_info:
            toret["task_info"] = self.get_task_info(src_path, task_id, episode_id)
        toret["video_path"] = self.get_video_paths(src_path, task_id, episode_id)
        return toret

    def __get_sample(self, index: int) -> dict[str, Any]:
        task_id, episode_id, dataset_id = self.all_task_episodes[index]
        src_path = self.dataset_roots[dataset_id]
        episode = self._get_sample_from_ids(src_path, task_id, episode_id)
        if self._transform is not None:
            episode = self._transform(episode)
        return episode

    def _get_sample(self, index: int) -> dict[str, Any]:
        initial_index = int(index)
        current_index = initial_index
        failures: list[tuple[int, Exception]] = []
        total_attempts = self._max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                return self.__get_sample(current_index)
            except Exception as exc:
                failures.append((current_index, exc))
                if attempt == total_attempts:
                    break
                new_index = int(
                    torch.randint(
                        0, self._get_length(), (1,), generator=self._gen
                    ).item()
                )
                logger.warning(
                    "AgiBot sample %s failed on attempt %s/%s (%s: %s); "
                    "retrying index %s",
                    current_index,
                    attempt,
                    total_attempts,
                    type(exc).__name__,
                    exc,
                    new_index,
                )
                current_index = new_index

        details = []
        for attempt, (failed_index, exc) in enumerate(failures, 1):
            try:
                task_id, episode_id, dataset_id = self.all_task_episodes[failed_index]
                sample = (
                    f"task_id={task_id}, episode_id={episode_id}, "
                    f"dataset_id={dataset_id}"
                )
            except (IndexError, TypeError):
                sample = "manifest row unavailable"
            details.append(
                f"attempt {attempt}/{total_attempts} index={failed_index} "
                f"({sample}): {type(exc).__name__}: {exc}"
            )
        message = (
            f"failed to load an AgiBot sample after {total_attempts} attempts "
            f"(initial_index={initial_index}, max_retries={self._max_retries}). "
            + " | ".join(details)
        )
        raise RuntimeError(message) from failures[-1][1]

    def __getitem__(self, index):
        return self._get_sample(index)


if __name__ == "__main__":
    CSV_PATH = "data/agibot/task_episode_alpha.csv"
    dataset = AgibotDataset(CSV_PATH)
    idx = np.random.choice(len(dataset))
    sample = dataset[idx]
    N = len(sample["proprio_stats"]["timestamp"])
    print(f"{N = }")
    print(f"{sample['task_id'] = }")
    print(f"{sample['episode_id'] = }")
    print(f"{sample['task_info']['task_name'] = }")
    print(f"{sample['video_path']['head_color'] = }")


"""
/timestamp                          [N] timestamp in nanoseconds
/state/effector/position (gripper)  [N, 2] left [:, 0], right [:, 1], gripper open range in mm
/state/effector/position (dexhand)  [N, 12] left [:, :6], right [:, 6:], joint angle in rad
/state/end/orientation              [N, 2, 4] left [:, 0, :], right [:, 1, :], flange quaternion with xyzw
/state/end/position                 [N, 2, 3] left [:, 0, :], right [:, 1, :], flange xyz in meters
/state/head/position                [N, 2] yaw [:, 0], pitch [:, 1], rad
/state/joint/current_value          [N, 14] left arm [:, :7], right arm [:, 7:]
/state/joint/position               [N, 14] left arm [:, :7], right arm [:, 7:], rad
/state/robot/orientation            [N, 4] quaternion in xyzw, yaw only
/state/robot/position               [N, 3] xyz position, where z is always 0 in meters
/state/waist/position               [N, 2] pitch [:, 0] in rad, lift [:, 1]in meters
/action/*/index                     [M] actions indexes refer to when the control source is actually sending signals
/action/effector/position (gripper) [N, 2] left [:, 0], right [:, 1], 0 for full open and 1 for full close
/action/effector/position (dexhand) [N, 12] same as /state/effector/position
/action/effector/index              [M_1] index when the control source for end effector is sending control signals
/action/end/orientation             [N, 2, 4] same as /state/end/orientation
/action/end/position                [N, 2, 3] same as /state/end/position
/action/end/index                   [M_2] same as other indexes
/action/head/position               [N, 2] same as /state/head/position
/action/head/index                  [M_3] same as other indexes
/action/joint/position              [N, 14] same as /state/joint/position
/action/joint/index                 [M_4] same as other indexes
/action/robot/velocity              [N, 2] vel along x axis [:, 0], yaw rate [:, 1]
/action/robot/index                 [M_5] same as other indexes
/action/waist/position              [N, 2] same as /state/waist/position
/action/waist/index                 [M_6] same as other indexes
"""
