from robot_wm.inference.robot.base import Robot, RobotObs, RobotState


class DummyRobot(Robot):
    def reset(self, init_state: RobotState) -> RobotObs:
        return self.reset_to_state(init_state)

    def execute_delta_action(self, action) -> RobotObs:
        print("DummyFranka: execute ", action)
        return RobotObs.ZERO(), {}

    def reset_to_state(self, state: RobotState) -> RobotObs:
        print("DummyFranka: reset to ", state)
        return RobotObs.ZERO()
