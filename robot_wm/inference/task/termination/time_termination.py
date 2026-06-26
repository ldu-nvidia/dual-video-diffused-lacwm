import time

from robot_wm.inference.task.termination import Termination


class TimeTermination(Termination):
    def __init__(self, max_time):
        self.max_time = max_time

        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def compute(self, obs, action, goal):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        return elapsed_time > self.max_time, {}
