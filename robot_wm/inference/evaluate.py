import os
import time

import hydra
import matplotlib.pyplot as plt
from termcolor import colored

from robot_wm.inference.common.robot_obs_logger import DataLogger


class Evaluator:
    def __init__(self, actor, robot, task, cfg):
        self.actor = actor  # Actor()
        self.robot = robot  # Robot()
        self.task = task  # Task()

        if "dir" in cfg:
            if os.path.exists(cfg.dir):
                print(colored(f"Expt logging already exists {cfg.dir}", "red"))
            formatted_time = time.strftime("%Y-%m-%d_%H:%M")
            expt_dir = os.path.join(cfg.dir, formatted_time)

            os.makedirs(expt_dir)
            os.chdir(expt_dir)
            print(colored(f"Expt logging is in {expt_dir}", "yellow"))

        self.logger = DataLogger()

    def run_evaluation(self):
        # initialize
        goal, init_state = self.task.reset()
        obs = self.robot.reset_to_state(init_state)
        self.actor.reset()
        time.sleep(5)
        print(colored("Starting Evaluation", "blue", attrs=["bold"]))
        png_path = "./viz/goal.png"
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        img = goal.image.transpose(1, 2, 0)
        plt.imsave(png_path, img)

        # evaluation loop
        self.task.start()
        self.robot.start()
        metric_info_list = []
        for i in range(10000000000000000):
            print(colored(f"MPC cycle {i}", "yellow"))
            self.logger.log_goal_to_the_actor(goal, i)
            self.logger.log_obs_to_the_actor(obs, i)
            action = self.actor.act(obs, goal)
            self.logger.log_action_from_the_actor(action, i)

            obs, robot_info = self.robot.execute_delta_action(action)
            termination, score, aux_info, metric_info = self.task.eval(
                obs, action, goal
            )
            metric_info_list.append(metric_info)
            if termination:
                print(colored("Evaluation Complete", "blue", attrs=["bold"]))
                print("Score: ", score)
                print("Auxiliary Info: ", aux_info)
                print("Metric Info: ", metric_info_list)
                self.logger.eval_log(score, aux_info, termination, metric_info_list)
                return

            time.sleep(5)
            obs = self.robot.get_last_observation()


@hydra.main(config_path="config", config_name="evaluator.yaml")
def main(cfg):

    actor = hydra.utils.instantiate(cfg.actor)
    robot = hydra.utils.instantiate(cfg.robot)
    task = hydra.utils.instantiate(cfg.task)
    evaluator = Evaluator(actor, robot, task, cfg)

    try:
        evaluator.run_evaluation()
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
    except Exception as e:
        import traceback

        print(f"Evaluation failed with error: {e}")
        traceback.print_exc()
    finally:
        # Ensure the robot's ROS spinning thread is stopped
        if hasattr(evaluator, "robot") and hasattr(evaluator.robot, "stop"):
            evaluator.robot.stop()
            # The robot class will handle ROS shutdown if needed


if __name__ == "__main__":
    main()
