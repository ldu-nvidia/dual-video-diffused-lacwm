import torch

from robot_wm.inference.actor.cost import Cost
from robot_wm.inference.world_model.base import WMState


class L2VisualLatentsCost(Cost):
    name: str = "L2VisualLatentsCost"

    def __call__(self, goal: WMState, state: WMState):
        """
        Predicts the L2 cost between a given World Model state and a goal
        """
        goal_latents = goal.visual_state
        state_latents = state.visual_state
        diff = state_latents - goal_latents  # (B, T, P, Z)
        err = torch.norm(diff, dim=-1)  # (B, T, P)
        cost = torch.mean(err, dim=-1)  # (B, T)
        return cost
