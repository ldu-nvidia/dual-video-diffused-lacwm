from typing import List

import hydra
import torch
from tqdm import tqdm

from robot_wm.inference.actor.base import Action, RobotActionHistory, RobotObsHistory
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)
from robot_wm.inference.world_model.base import WorldModel


class CEM:
    """
    Cross-Entropy Method for trajectory optimization.
    Maintains a distribution over actions and iteratively refines it using elite samples.
    """

    def __init__(self, config):
        # CEM hyperparameters
        self.n_particles = config.n_particles  # Number of particles to sample
        self.iterations = config.iterations  # Number of CEM iterations
        self.pred_horizon_s = config.pred_horizon_s  # Prediction horizon in seconds
        self.elite_frac = config.elite_frac  # Fraction of top particles to keep
        self.var_scale = config.var_scale  # Initial variance scale
        self.cost = config.cost  # Cost function
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
        print("Planning with CEM...")
        encoded_context = wm.encode_history(obs_history, action_history)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate the number of elite particles
        n_elite = int(self.n_particles * self.elite_frac)

        # Initialize action distribution parameters
        # Mean starts as zeros or initial action if provided
        initial_action_tensor = torch.zeros(1, FTR, action_dim).to(device)
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

        # Initialize CEM distribution parameters
        mu = initial_action_tensor * 1.0  # (1, FTR, A)
        sigma = self.var_scale * torch.ones_like(mu)  # Initial standard deviation

        # Best action and cost tracking
        best_action = mu.clone()
        best_cost = float("inf")

        pbar = tqdm(range(self.iterations))
        for iter in pbar:
            # Sample action particles from the distribution
            action_particles = mu.repeat(self.n_particles, 1, 1) + sigma.repeat(
                self.n_particles, 1, 1
            ) * torch.randn(self.n_particles, FTR, action_dim, device=device)

            # Clamp actions to valid range if needed
            action_particles = torch.clamp(action_particles, min=-1.0, max=1.0)

            # Create action histories for world model
            action_histories = RobotActionHistory.from_tensor(
                1000,
                action_frequency,
                action_particles,  # Using 1000 to match original code
            )

            # Rollout trajectories
            latent_rollouts = wm.rollout(encoded_context, action_histories)

            # Compute costs (distance to goal in latent space)
            costs = self.cost(encoded_goal, latent_rollouts)
            costs = costs[:, -1]  # Take only the last cost

            # Print costs statistics
            print(
                f"Costs: min={costs.min().item():.5f}, mean={costs.mean().item():.5f}, max={costs.max().item():.5f}"
            )

            # Sort by cost and select elite indices
            elite_indices = torch.topk(costs, k=n_elite, largest=False).indices
            elite_actions = action_particles[elite_indices]

            # Find minimum cost in this batch
            min_cost = torch.min(costs).item()
            min_cost_idx = torch.argmin(costs).item()

            # Update best action if improved
            if min_cost < best_cost:
                best_action = action_particles[
                    min_cost_idx : min_cost_idx + 1
                ]  # Take the best action
                best_cost = min_cost

            # Update distribution parameters based on elite samples
            mu = elite_actions.mean(dim=0, keepdim=True)  # Update mean
            sigma = elite_actions.std(dim=0, keepdim=True)  # Update std deviation

            # Ensure minimum exploration by clamping sigma
            sigma = torch.clamp(sigma, min=self.var_scale * 0.1)

            pbar.set_postfix({"Best Cost": best_cost})

            if iter % 1 == 0:
                print(
                    f"Iter {iter}, Min cost: {min_cost:.5f}, Best: {best_cost:.5f}, Best XYZ {best_action[0, :, :3].sum(dim=0).cpu().numpy()}"
                )

        best_actions = RobotActionHistory.from_tensor(
            1000, action_frequency, best_action.cpu()
        )
        print(f"Found an action sequence {best_actions} with cost {best_cost}")

        RENDER_RESULT = False
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

        return best_actions


@hydra.main(
    config_path="../../config/actor/planning",
    config_name="cem.yaml",
)
def main(cfg):
    planner = hydra.utils.instantiate(cfg)
    print(planner)
