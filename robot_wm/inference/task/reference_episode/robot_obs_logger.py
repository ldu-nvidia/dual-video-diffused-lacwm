import time

from robot_wm.inference.actor.base import Action
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import ImageProprioGoal


class TimeStepLogger:
    def __init__(
        self,
        log_filename="/home/abhagejji/Code/franka_deoxys_ros/logs/robot_time_steps.log",
    ):
        """
        Initialize the logger with a log file name.
        If no filename is provided, it defaults to "robot_time_steps.log".
        """
        self.log_filename = log_filename

    def log(self, obs: RobotObs, action: Action, goal: ImageProprioGoal):

        info_packet = {
            "timestamp": time.time(),
            "obs_end_effector_pose": obs.end_effector_pose,
            "obs_joints": obs.joints,
            "obs_image": obs.image,
            # eventually would add action and goal
        }

        # log to a file for persistent storage
        with open(self.log_filename, "a") as log_file:
            log_file.write(info_packet)
