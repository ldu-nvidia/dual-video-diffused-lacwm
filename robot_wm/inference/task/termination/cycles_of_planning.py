from robot_wm.inference.actor.base import Action
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import ImageProprioGoal
from robot_wm.inference.task.termination import Termination


class CyclesOfPlanningTermination(Termination):
    def __init__(self, max_cycles):
        self.max_cycles = max_cycles
        self.current_cycles = 0

    def reset(self):
        self.current_cycles = 0

    def compute(self, obs: RobotObs, action: Action, goal: ImageProprioGoal):
        self.current_cycles += 1
        is_terminated = self.current_cycles >= self.max_cycles
        return is_terminated, {}
