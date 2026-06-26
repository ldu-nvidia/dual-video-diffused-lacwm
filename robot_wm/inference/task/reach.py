import time  # Added for timeout tracking
from typing import List

import hydra

from robot_wm.inference.actor.base import Action
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task import Task
from robot_wm.inference.task.metrics import Metric
from robot_wm.inference.task.reference_episode import ReferenceEpisode
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)
from robot_wm.inference.task.termination import Termination


class ReachTask(Task):
    def __init__(
        self,
        terminations: List[Termination],
        reference_episode: ReferenceEpisode,
        metrics: List[Metric],
    ):  # Added max_time parameter
        # h5 load file
        self.metrics = metrics
        self.terminations = terminations
        self.reference_episode = reference_episode
        # variables
        self.alive_start_time = 0
        self.max_time = 5  # Maximum allowed time in seconds

    def reset(self):
        self.alive_start_time = time.time()  # Record start time on reset
        init_state = self.reference_episode.get_initial_state()
        goal = self.reference_episode.goal
        return goal, init_state

    def start(self):
        for metric in self.metrics:
            metric.reset()
        for termination in self.terminations:
            termination.reset()

    def compute_termination(self):
        # timeout
        current_time = time.time()
        elapsed_time = current_time - self.alive_start_time
        timeout = elapsed_time > self.max_time

        # goal reached
        # Add goal-reaching condition here
        goal_reached = False

        return timeout or goal_reached

    def eval(self, obs: RobotObs, action: Action, goal: ImageProprioGoal):
        scores = {}
        aux_info = {"termination_reason": {}}  # Placeholder for auxiliary metrics
        for metric in self.metrics:
            score, metric_info = metric.compute(obs, action, goal)
            scores[metric.__class__.__name__] = score

        # Check for termination
        any_terminate = False
        for termination in self.terminations:
            terminate, terminate_info = termination.compute(obs, action, goal)
            any_terminate = any_terminate or terminate
            if terminate:
                aux_info["termination_reason"][
                    termination.__class__.__name__
                ] = terminate_info

        # Compute score and other metrics

        return any_terminate, scores, aux_info, metric_info


@hydra.main(
    config_path="../../config/task", config_name="reach.yaml", version_base="1.1"
)
def main(cfg):
    # Load the config
    print(cfg)
    task = hydra.utils.instantiate(cfg)
    print(task)


if __name__ == "__main__":
    main()
