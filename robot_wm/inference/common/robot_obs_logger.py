import json
import os
import time

import h5py
import numpy as np

from robot_wm.inference.actor.base import Action
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import ImageProprioGoal


class DataLogger:
    def __init__(self):
        """
        Initialize the logger with a log file name.
        If no filename is provided, it defaults to "robot_time_steps.log".
        """
        self.start_time = time.time()
        os.makedirs("./log", exist_ok=True)

    def log_obs_to_the_actor(self, obs: RobotObs, idx):
        with h5py.File("./log/obs_log.h5", "a") as f:
            log_entry = f.create_group(f"obs_{idx}")
            log_entry.create_dataset("time", data=float(time.time() - self.start_time))
            log_entry.create_dataset("ee_pose", data=np.array(obs.end_effector_pose))
            log_entry.create_dataset("joints", data=np.array(obs.joints))
            log_entry.create_dataset(
                "image",
                data=np.array(obs.image),
                compression="gzip",
                compression_opts=4,
            )

    def log_action_from_the_actor(self, action: Action, idx):
        with h5py.File("./log/action_log.h5", "a") as f:
            log_entry = f.create_group(f"action_{idx}")
            log_entry.create_dataset("action", data=np.array(action.action))
            log_entry.create_dataset("time", data=float(time.time() - self.start_time))

    def log_goal_to_the_actor(self, goal: ImageProprioGoal, idx):
        with h5py.File("./log/goal_log.h5", "a") as f:
            log_entry = f.create_group(f"goal_{idx}")
            log_entry.create_dataset("goal_proprio", data=np.array(goal.proprio_tensor))
            log_entry.create_dataset("goal_image", data=np.array(goal.image))
            log_entry.create_dataset(
                "goal_time", data=float(time.time() - self.start_time)
            )

    def eval_log(self, score, aux_info, termination, metric_info_list):
        """
        Logs evaluation metrics to a JSON file.
        """
        with open("./eval.json", "w") as file:
            json.dump(
                {
                    "start_time": self.start_time,
                    "end_time": time.time(),
                    "score": score,
                    "aux_info": aux_info,
                    "termination": termination,
                    "metric_info": metric_info_list,
                },
                file,
                indent=4,
            )


class DataLoggerFranka:
    def __init__(self):
        self.start_time = time.time()
        os.makedirs("./logRT", exist_ok=True)
        self.h5_file = h5py.File("./logRT/franka_data.h5", "a")
        self.frame_count = 0

    def log_obs_franka(self, obs: RobotObs):
        # Create a group for this timestep
        timestamp = time.time() - self.start_time
        group_name = f"_frame_{self.frame_count}"

        # Check if the group already exists
        if group_name in self.h5_file:
            # Either use a unique name or update the existing group
            group_name = f"_frame_{self.frame_count}_{int(timestamp * 1000)}"  # Make name unique with timestamp

        # Now create the group with the possibly updated name
        grp = self.h5_file.create_group(group_name)

        # Store metadata and smaller arrays
        grp.attrs["time"] = timestamp
        grp.create_dataset("ee_pose", data=np.array(obs.end_effector_pose))
        grp.create_dataset("joints", data=np.array(obs.joints))

        # Store image efficiently
        grp.create_dataset(
            "image", data=np.array(obs.image), compression="gzip", compression_opts=4
        )

        self.frame_count += 1
        self.h5_file.flush()  # Ensure data is written to disk

    def close(self):
        self.h5_file.close()
