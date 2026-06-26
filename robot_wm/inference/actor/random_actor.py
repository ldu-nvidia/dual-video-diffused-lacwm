from robot_wm.inference.actor.base import Actor
from robot_wm.inference.robot.base import RobotEEDeltaAction


class RandomActor(Actor):
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, obs, goal):
        return RobotEEDeltaAction.RANDOM()


if __name__ == "__main__":
    actor = RandomActor()
    action = actor.act(None, None)
