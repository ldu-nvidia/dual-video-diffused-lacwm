import h5py
import numpy as np
from huggingface_hub import hf_hub_download

from robot_wm.inference.robot.base import RobotState
from robot_wm.inference.task.reference_episode import ImageProprioGoal, ReferenceEpisode


class DummyImageGoalReferenceEpisode(ReferenceEpisode):
    def __init__(self):
        pass

    def get_initial_state(self) -> RobotState:

        return RobotState.ZERO()

    @property
    def goal(self):
        return ImageProprioGoal.ZERO()


class H5ImageGoalReferenceEpisode(ReferenceEpisode):
    """image goal h5 reference episode"""

    def __init__(self, episode_path, bgr=False, huggingface=False):
        if huggingface:
            self.episode_path = hf_hub_download(repo_id="facebook/robotics-world-models", filename=episode_path)
        else:
            self.episode_path = episode_path
        self.episode = h5py.File(self.episode_path, "r")
        self.bgr = bgr

    def __validate_episode(self):
        assert "episode_data" in self.episode
        assert "observation" in self.episode["episode_data"]
        assert "joint_position" in self.episode["episode_data"]["observation"]
        assert "exterior_image_1_left" in self.episode["episode_data"]["observation"]
        # TODO: Assert shapes too

    def get_initial_state(self) -> RobotState:
        # messy logic to decide what is the init state
        init_joints = self.episode["episode_data"]["observation"]["joint_position"][0]
        return RobotState(init_joints)

    @property
    def _reference_episode(self):
        return self.episode

    @property
    def goal(self):
        image_hwc = self.episode["episode_data"]["observation"][
            "exterior_image_1_left"
        ][-1]
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        image_chw_01 = image_chw / 255.0
        if self.bgr:
            image_chw_01 = image_chw_01[[2, 1, 0], :, :]
        ee_pose = self.episode['episode_data']['observation']['cartesian_position'][-1]
        ee_pose = np.concatenate((ee_pose,[0,]))

        return ImageProprioGoal(image_chw_01, ee_pose)


import hydra


@hydra.main(
    config_path="../../config/task", config_name="reach.yaml", version_base="1.1"
)
def main(cfg):
    episode = hydra.utils.instantiate(cfg.reference_episode)
    print(episode.goal)
    print(episode.get_initial_state())


if __name__ == "__main__":
    main()

    # episode_path = '/home/abhagejji/Code/franka_deoxys_ros/evaluation_tasks/mpk/reach/v0/run_0003/episode.h5'
    # episode = H5ImageGoalReferenceEpisode(episode_path)
    # print(episode.goal)
    # print(episode.get_initial_state())
