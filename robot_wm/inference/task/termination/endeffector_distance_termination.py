import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_wm.inference.actor.base import Action
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import ImageProprioGoal
from robot_wm.inference.task.termination import Termination


def quaternion_angle_difference(quat_a_wxyz, quat_b_wxyz):
    quat_a_xyzw = np.array(quat_a_wxyz)[[1, 2, 3, 0]]
    quat_b_xyzw = np.array(quat_b_wxyz)[[1, 2, 3, 0]]
    r_a = R.from_quat(quat_a_xyzw)
    r_b = R.from_quat(quat_b_xyzw)
    r_diff = r_a.inv() * r_b
    # angle_axis = r_diff.as_rotvec()
    angle = r_diff.magnitude()
    return angle  # radians


class EndEffectorDistanceTermination(Termination):
    def __init__(self, pos_tolerance, angle_tolerance):
        self.pos_tolerance = pos_tolerance
        self.angle_tolerance = angle_tolerance

    def reset(self):
        pass

    def compute(self, obs: RobotObs, action: Action, goal: ImageProprioGoal):
        pos_deltas = np.abs(obs.end_effector_pose[:3] - goal.proprio[:3])
        quat_a_wxyz = obs.end_effector_pose[3:]
        rpy = goal.proprio[3:6]
        quat_b_wxyz = R.from_euler("xyz", rpy).as_quat()
        angle_rad = quaternion_angle_difference(quat_a_wxyz, quat_b_wxyz)
        angle_tol_rad = np.deg2rad(self.angle_tolerance)

        is_terminated = (
            np.all(pos_deltas < self.pos_tolerance) and angle_rad < angle_tol_rad
        )
        info = {
            "pos_deltas": pos_deltas,
            "angle_rad": angle_rad,
            "pos_tolerance": self.pos_tolerance,
            "angle_tolerance": angle_tol_rad,
        }
        if is_terminated:
            print("Terminated")  # NOCOMMIT
        return is_terminated, info
