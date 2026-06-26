import time

import numpy as np

from robot_wm.inference.robot.base import RobotAction, RobotObs
from robot_wm.inference.task.metrics import Metric
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)


def quatereion_difference(q1, q2):
    return np.arccos(2 * (np.dot(q1, q2) ** 2) - 1)


# def angle_difference(q1, q2):


class PoseErrorMetric(Metric):
    def __init__(self):
        pass

    def reset(self):
        self.start_time = time.time()

    def compute(self, obs: RobotObs, action: RobotAction, goal: ImageProprioGoal):

        # Compute the position difference as the sum of absolute differences
        # between the goal and reached positions (x, y, z coordinates)
        xyz_delta = np.array(goal.proprio_tensor[:3] - obs.end_effector_pose[:3])
        # angle_delta = np.array(goal_pose[3:7]-reached_pose[3:7])

        # orientation_diff = np.sum(np.abs(angle_delta))
        carteian_dist = np.linalg.norm(xyz_delta)

        # angle_delta = quatereion_difference(goal.proprio_tensor[3:7],obs.end_effector_pose[3:7])
        angle_diff = np.array(goal.proprio_tensor[3:6] - obs.end_effector_pose[3:6])
        angle_delta = np.sum(np.abs(angle_diff))

        numerial_diff = np.array(goal.proprio_tensor[:7] - obs.end_effector_pose[:7])
        error = np.sum(np.abs(numerial_diff))

        score = carteian_dist  # + angle_delta + error
        return (
            score,
            {
                "carteian_dist": carteian_dist,
                "angle_diff": angle_delta,
                "error": error,
                "time": time.time() - self.start_time,
            },
        )
