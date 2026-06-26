from robot_wm.inference.world_model.base import WMState


class Cost:
    name: str

    def __call__(self, goal: WMState, state: WMState):
        """
        Predicts the cost between a given World Model state and a goal
        """
        raise NotImplementedError("Cost function not implemented")
