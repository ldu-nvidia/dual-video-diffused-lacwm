import csv
import json
import logging
import os
from typing import Any, Optional

import h5py
import numpy as np

from robot_wm.datasets.agibot.transform import AgibotTransform
from robot_wm.datasets.base import Dataset

logger = logging.getLogger(__name__)

# TODO: make configurable
DATASET_ROOTS = {
    "alpha": "/checkpoint/cortex/ravenhuang/agibot_sync/AgiBotWorld-Alpha",
    "beta": "/checkpoint/cortex/ravenhuang/agibot_sync/AgiBotWorld-Beta",
    "viscam": "/scr/ravenh/lacwm_data/agibot_combined",
    "scr": "/scr/ravenh/lacwm_data/agibot",
}


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
        **kwargs,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.csv_path = csv_path
        if manifest is not None:
            self.csv_path = manifest
        self.all_task_episodes = self.get_manifest(self.csv_path)
        if subsample_traj is not None:
            self.all_task_episodes = self.all_task_episodes[:subsample_traj]
        self.load_camera_params = load_camera_params
        self.load_task_info = load_task_info
        logger.info(f"{self.csv_path = }")
        self.ee_action_dim = 34  # 20d ee (10/arm) + 14d joint (7/arm); +9 camera -> 43 total
        self.decode_camera = False

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
        src_path = DATASET_ROOTS[dataset_id]
        episode = self._get_sample_from_ids(src_path, task_id, episode_id)
        if self._transform is not None:
            episode = self._transform(episode)
        return episode

    def _get_sample(self, index: int) -> dict[str, Any]:
        # return self.__get_sample(index)
        try:
            return self.__get_sample(index)
        except Exception as e:
            new_index = np.random.choice(self._get_length())
            logger.warning(f"{index = } failed; using {new_index = }; {e = }")
            return self._get_sample(new_index)

    def __getitem__(self, index):
        return self._get_sample(index)


if __name__ == "__main__":
    CSV_PATH = "/fsx-cortex-datacache/shared/datasets/agibot/manifest_csv/task_episode_alpha.csv"
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
