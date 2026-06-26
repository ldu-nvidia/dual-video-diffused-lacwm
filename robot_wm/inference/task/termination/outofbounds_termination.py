import numpy as np

from robot_wm.inference.actor.base import Action
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import ImageProprioGoal
from robot_wm.inference.task.termination import Termination


class BoxOutOfBoundsTermination(Termination):
    def __init__(self, box_min, box_max):
        self.box_min = box_min
        self.box_max = box_max

    def reset(self):
        pass

    def compute(self, obs: RobotObs, action: Action, goal: ImageProprioGoal):
        pos = obs.end_effector_pose[:3]
        is_terminated = np.any(pos < self.box_min) and np.any(pos > self.box_max)
        info = {"pos": pos, "box_min": self.box_min, "box_max": self.box_max}
        if is_terminated:
            print("Terminated")  # NOCOMMIT
        return is_terminated, info
