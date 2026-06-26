import time

from robot_wm.inference.task.metrics import Metric


class ExecutionTimeMetric(Metric):
    def __init__(self):
        pass

    def reset(self):
        self.start_time = time.time()

    def compute(self, obs, action, goal):
        wall_time = time.time() - self.start_time
        return wall_time, {}
