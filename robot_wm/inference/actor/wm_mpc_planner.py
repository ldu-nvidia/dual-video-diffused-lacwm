import os

import hydra
import matplotlib.pyplot as plt

from robot_wm.inference.actor.base import Actor, RobotActionHistory, RobotObsHistory
from robot_wm.inference.robot.franka import RobotEEDeltaAction, RobotObs
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)


class WMMPCPlanner(Actor):
    def __init__(self, config):
        self.trajectory_optimization = (
            config.trajectory_optimization
        )  # TrajectoryOptimization
        self.worldmodel = config.world_model  # WorldModel

        self.robot_frequency = config.robot_frequency
        self.actor_frequency = config.actor_frequency
        self.planning_frequency = config.planning_frequency

        self.step = 0
        self.max_history_size = config.max_history_size

        self.action_aggregation_horizon = config.action_aggregation_horizon
        self.observation_history = RobotObsHistory(
            self.max_history_size, self.robot_frequency
        )
        self.action_history = RobotActionHistory(
            self.max_history_size, self.robot_frequency
        )
        self.action_queue = []

    def reset(self):
        self.observation_history.reset()
        self.action_history.reset()
        self.action_history.push(
            RobotEEDeltaAction.ZERO()
        )  # initialize world model with zero action
        self.worldmodel.reset()
        self.trajectory_optimization.reset()

    def __store_obs(self, obs, idx):
        os.makedirs("./viz", exist_ok=True)
        image = (obs.image * 255).astype("uint8")
        plt.imsave(f"./viz/obs_{idx}.png", image.transpose(1, 2, 0))

    def act(self, obs: RobotObs, goal: ImageProprioGoal):

        self.observation_history.push(obs)
        self.__store_obs(obs, self.step)

        # Plan to get the action trajectory
        action_trajectory = self.trajectory_optimization.plan(
            self.observation_history,
            self.action_history,
            goal,
            self.worldmodel,
        )

        # Integrate it to give a higher level delta action to the robot
        action = action_trajectory.aggregate_first_mpc_action(
            horizon_s=self.action_aggregation_horizon
        )
        action = RobotEEDeltaAction(action.action[0, :])

        # In theory we need now to store the action but this is somewhat broken
        # --- We'd want to store the trajectory of actions but we don't have observations at same frequency
        # ----We cannot simply store the integrated action because its not how the model was trained
        # --- This should be fixed once we deploy models with smaller frequency at training
        # --- For now models only use context = 1 so we dont consider an action history for replanning

        # for a in action_trajectory.actions:
        #     print('Adding action')
        #     self.action_history.push(a)

        # self.action_history.push(action)

        # action_chunks = action_trajectory.chunk(self.max_history_size, 6, 'sum', from_last=False)
        # for action_chunk in action_chunks.actions:
        #     self.action_history.push(action_chunk)

        self.step += 1
        return action


@hydra.main(
    config_path="../config/actor", config_name="wm_mpc_planner_5hz_dino_p209.yaml"
)
def main(cfg):
    actor = hydra.utils.instantiate(cfg)
    print(actor)


if __name__ == "__main__":
    main()
