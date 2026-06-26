from typing import List

import hydra
import torch
from tqdm import tqdm

from robot_wm.inference.actor.base import Action, RobotActionHistory, RobotObsHistory
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)
from robot_wm.inference.world_model.base import WorldModel


class ITMPC:
    def __init__(self, config):
        self.sample_size = config.sample_size
        self.iterations = config.iterations
        self.pred_horizon_s = config.pred_horizon_s
        self.SIGMA = config.SIGMA
        self.cost = config.cost
        self.counter = 0

    def reset(self):
        pass

    @torch.inference_mode()
    def plan(
        self,
        obs_history: RobotObsHistory,
        action_history: RobotActionHistory,
        goal: ImageProprioGoal,
        wm: WorldModel,
        initial_action: RobotActionHistory = None,
    ) -> List[Action]:

        encoded_goal = wm.encode_goal(goal)  # WMLatentGoal
        action_frequency = action_history.freq
        FTR = int(self.pred_horizon_s * action_frequency)
        action_dim = action_history.action_dim
        print("Planning...")
        encoded_context = wm.encode_history(obs_history, action_history)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_cost = float("inf")
        initial_action_tensor = torch.zeros(1, FTR, action_dim).to(device)
        # get initial action if provided
        if initial_action is not None:
            _1, _FTR, _A = initial_action.actions_tensor.shape
            assert (
                _1 == 1 and _A == action_dim
            ), f"initial_action shape should be (1, {FTR}, {action_dim}) but is ({_1}, {_FTR}, {_A})"
            if _FTR != FTR:
                print(
                    f"WARNING: initial_action shape should be (1, {FTR}, {action_dim}) but is ({_1}, {_FTR}, {_A}). Padding time axis with zeros."
                )
            initial_action_tensor[:, :_FTR, :] = initial_action.actions_tensor
        best_action = initial_action_tensor * 1.0  # (1, FTR, A)

        pbar = tqdm(range(self.iterations))
        for iter in pbar:
            # Generate random action samples
            noised_actions = (
                best_action
                + torch.randn(self.sample_size, FTR, action_dim, device=device)
                * self.SIGMA
            )

            # Create action histories for world model
            action_histories = RobotActionHistory.from_tensor(
                1000, action_frequency, noised_actions
            )

            # Rollout trajectories
            latent_rollouts = wm.rollout(encoded_context, action_histories)

            # Compute costs (distance to goal in latent space)
            costs = self.cost(encoded_goal, latent_rollouts)
            costs = costs[:, -1]  # take only the last cost
            print(costs)

            # Find minimum cost in this batch
            min_cost = torch.min(costs).item()

            # Update best action if improved
            if min_cost < best_cost:
                weights = torch.exp(-costs)
                weights = weights / weights.sum()
                # 0 for all but min cost
                # weights = torch.zeros(self.sample_size, device=device)
                # weights[torch.argmin(costs)] = 1.0

                best_action = torch.sum(
                    weights[:, None, None] * noised_actions.to(device),
                    dim=0,
                    keepdim=True,
                )
                best_cost = min_cost
            pbar.set_postfix({"Best Cost": best_cost})

            if iter % 1 == 0:
                print(
                    f"Iter {iter}, Min cost: {min_cost:.5f}, Best: {best_cost:.5f}, Best XYZ {best_action[0, :, :3].sum(dim=0).cpu().numpy()}"
                )

        best_actions = RobotActionHistory.from_tensor(
            1000, action_frequency, best_action.cpu()
        )
        print(f"Found an action sequence {best_actions} with cost {best_cost}")

        RENDER_RESULT = True
        if RENDER_RESULT:
            import os

            os.makedirs("./viz", exist_ok=True)
            latent_rollouts = wm.rollout(encoded_context, best_actions)
            rollout_obs = wm.decode_latents(latent_rollouts)
            rollout_obs.store(f"./viz/rollout_{self.counter}.mp4")
            best_actions.plot_cumulative(
                show=False, save_path=f"./viz/actions_{self.counter}.png"
            )
            self.counter += 1

        # Already in robot frequency and everything
        # wm.rollout(encoded_context, best_actions, store_vid=True)#TODO: solve error store_vid=True
        return best_actions


@hydra.main(
    config_path="../../config/actor/planning",
    config_name="naive_trajectory_optimization.yaml",
)
def main(cfg):
    planner = hydra.utils.instantiate(cfg)
    print(planner)
