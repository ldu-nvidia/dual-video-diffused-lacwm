# pip install mani_skill
import gymnasium as gym  # noqa: F401
import mani_skill.envs  # noqa: F401
import numpy as np
import gymnasium as gym
import sapien
from transforms3d.euler import euler2quat
import mani_skill.envs
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv # noqa: F401
from mani_skill.utils.registration import register_env # noqa: F401
from mani_skill.utils import sapien_utils # noqa: F401
from mani_skill.sensors.camera import CameraConfig # noqa: F401

import robot_wm.inference.common.deoxys_transform_utils as transform_utils
from robot_wm.inference.robot.base import (
    Robot,
    RobotEEDeltaAction,
    RobotObs,
    RobotState,
)


# This registers the environments
from projects.droid_maniskill.droid_pick_cube_env import DROIDPickCubeEnv


# Old robot (franka gripper)
class SimulatedRobot(Robot):
    def __init__(self, config):
        self.env = gym.make(
            "DROIDPickCube-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode="state",  # there is also "state_dict", "rgbd", ...
            control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
            render_mode="sensors",  # there is also "human", "sensors", "rgb_array", "all", ...
            robot_uids="panda_wristcam",
        )

    def start(self):
        pass

    def reset(self, init_state: RobotState) -> RobotObs:
        return self.reset_to_state(init_state)

    def execute_delta_action(self, action : RobotEEDeltaAction) -> RobotObs:
        xyzrpygripper = action.action[0] * 1. # dx d
        xyzrpygripper[-1] = -xyzrpygripper[-1] # gripper is open when 0, closed when 1
        REPEAT = 10
        for _ in range(REPEAT):
            obs, reward, terminated, truncated, info = self.env.step(xyzrpygripper)
        robot_obs = self._get_robot_obs()
        return robot_obs, {}

    def _get_robot_obs(self) -> RobotObs:
        # image
        panoptic = self.env.render()[0].cpu()
        H, W, C = 180, 320, 3
        ext1 = np.array(panoptic[:H, :W, :C])
        # ext2 = panoptic[:H, W:W*2, :C]
        ext1_chw_01 = ext1.transpose(2, 0, 1) / 255.0
        # proprio
        joints_9 = self.env.agent.robot.get_qpos()[0, :]
        limits_9 = self.env.agent.robot.get_qlimits()[0, :, :]  # (9, 2) [lower, upper]
        joints_7 = np.array(joints_9[:7])
        # we flip around x by pi as droid has tcp pointing down
        tcp_in_root = self.env.agent.robot.get_root_pose().inv() * self.env.agent.tcp.pose * sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi, 0, 0))# world_in_root * tcp_in_world * droidtcp_in_tcp
        ee_pos = tcp_in_root.p[0, :]
        ee_wxyz = tcp_in_root.q[0, :]
        ee_xyzw = np.array([ee_wxyz[3], ee_wxyz[0], ee_wxyz[1], ee_wxyz[2]])
        mat = transform_utils.quat2mat(ee_xyzw)
        rpy = transform_utils.mat2euler(mat)
        # sim has a joint for each gripper finger, 0 means open, 0.04 means closed
        assert limits_9[7][1] == 0.04
        gripper_01 = 1.0 - (joints_9[7] / 0.04)
        xyzrpygripper = np.concatenate([ee_pos, rpy, [gripper_01]])
        return RobotObs(
            joints=joints_7, end_effector_pose=xyzrpygripper, image=ext1_chw_01
        )

    def get_last_observation(self) -> RobotObs:
        obs = self._get_robot_obs()
        return obs

    def reset_to_state(self, state: RobotState) -> RobotObs:
        target_joints = state.joints.tolist()
        target_qpos = np.zeros((9,))  # 7 joints + 2 gripper
        target_qpos[:7] = np.array(target_joints)
        obs, _ = self.env.reset(seed=0)  # reset with a seed for determinism
        self.env.agent.robot.set_qpos(target_qpos)
        return self._get_robot_obs()


# New robot: Robotiq gripper
class NewSimulatedRobot(Robot):
    def __init__(self, config):
        style_seed = config.get("style_seed", 0)
        self.env = gym.make(
            "DROIDPickCube-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode="state",  # there is also "state_dict", "rgbd", ...
            control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
            render_mode="sensors",  # there is also "human", "sensors", "rgb_array", "all", ...
            robot_uids="panda_robotiq",
            style_seed=style_seed,
        )
        self.target_gripper_01 = None

    def start(self):
        pass

    def reset(self, init_state: RobotState) -> RobotObs:
        return self.reset_to_state(init_state)

    def execute_delta_action(self, action : RobotEEDeltaAction) -> RobotObs:
        xyzrpygripper = action.action[0] * 1. # dx d
        if True: # relative to absolute
            gripper_delta = xyzrpygripper[-1]
            if self.target_gripper_01 is None:
                self.target_gripper_01 = self._get_robot_obs().end_effector_pose[-1]
            self.target_gripper_01 = np.clip(self.target_gripper_01 + gripper_delta, 0, 0.81)
            xyzrpygripper[-1] = self.target_gripper_01
        else:
            xyzrpygripper[-1] = -xyzrpygripper[-1] # gripper is open when 0, closed when 1
        REPEAT = 10
        for _ in range(REPEAT):
            obs, reward, terminated, truncated, info = self.env.step(xyzrpygripper)
        robot_obs = self._get_robot_obs()
        return robot_obs, {}

    def _get_robot_obs(self) -> RobotObs:
        # image
        panoptic = self.env.render()[0].cpu()
        H, W, C = 180, 320, 3
        ext1 = np.array(panoptic)
        # ext1 = np.array(panoptic[:H, :W, :C])
        # ext2 = panoptic[:H, W:W*2, :C]
        ext1_chw_01 = ext1.transpose(2, 0, 1) / 255.0
        # proprio
        joints_9 = self.env.agent.robot.get_qpos()[0, :]
        limits_9 = self.env.agent.robot.get_qlimits()[0, :, :]  # (9, 2) [lower, upper]
        joints_7 = np.array(joints_9[:7])
        # we flip around x by pi as droid has tcp pointing down
        tcp_in_root = self.env.agent.robot.get_root_pose().inv() * self.env.agent.tcp.pose * sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi, 0, 0))# world_in_root * tcp_in_world * droidtcp_in_tcp
        ee_pos = tcp_in_root.p[0, :]
        ee_wxyz = tcp_in_root.q[0, :]
        ee_xyzw = np.array([ee_wxyz[3], ee_wxyz[0], ee_wxyz[1], ee_wxyz[2]])
        mat = transform_utils.quat2mat(ee_xyzw)
        rpy = transform_utils.mat2euler(mat)
        # sim has a joint for each gripper finger, 0 means open, 0.04 means closed
        assert limits_9[7][1] == 0.81 # robotiq
        gripper_01 = (joints_9[7] / limits_9[7][1])
        xyzrpygripper = np.concatenate([ee_pos, rpy, [gripper_01]])
        return RobotObs(
            joints=joints_7, end_effector_pose=xyzrpygripper, image=ext1_chw_01.astype(np.float32)
        )

    def get_last_observation(self) -> RobotObs:
        obs = self._get_robot_obs()
        return obs

    def move_to_joints_and_gripper(self, state: RobotState) -> RobotObs:
        # print(self.env.agent.supported_control_modes)
        self.env.agent.set_control_mode("pd_joint_pos")
        self.env.agent.controller.reset()
        # print(self.env.agent.single_action_space)
        target_joints = state.joints.tolist()
        jointsgripper = np.zeros((8,))
        jointsgripper[:7] = np.array(target_joints)
        gripper_01 = state.gripper
        gripper = np.clip(gripper_01 * 0.81, 0, 0.81)  # robotiq gripper range is [0, 0.81]
        jointsgripper[-1] = gripper
        REPEAT = 10
        for _ in range(REPEAT):
            obs, reward, terminated, truncated, info = self.env.step(jointsgripper[None, :]) 
        self.env.agent.set_control_mode("pd_ee_delta_pose")
        self.env.agent.controller.reset()
        return self._get_robot_obs()

    def move_to_pose_and_gripper(self, xyzrpygripper: np.ndarray) -> RobotObs:
        """ example: [0, 0, 0, pi, 0, 0, 0.81] """
        self.env.agent.set_control_mode("pd_ee_pose")
        self.env.agent.controller.reset()
        # print(self.env.agent.single_action_space)
        REPEAT = 10
        for _ in range(REPEAT):
            obs, reward, terminated, truncated, info = self.env.step(xyzrpygripper) 
        self.env.agent.set_control_mode("pd_ee_delta_pose")
        self.env.agent.controller.reset()
        return self._get_robot_obs()

    def reset_to_state(self, state: RobotState) -> RobotObs:
        target_joints = state.joints.tolist()
        target_qpos = np.zeros((13,))  # 7 joints + 6 gripper
        target_qpos[:7] = np.array(target_joints)
        obs, _ = self.env.reset(seed=0)  # reset with a seed for determinism
        self.env.agent.robot.set_qpos(target_qpos)
        return self._get_robot_obs()

if __name__ == "__main__":
    from robot_wm.inference.actor.base import RobotObsHistory
    from tqdm import tqdm

    config = {"style_seed": 0}
    robot = NewSimulatedRobot(config)
    history = RobotObsHistory(100, 30, [])
    init_state = RobotState(
        joints=np.array(
            [
                2.0078e-03,
                3.7137e-01,
                3.6554e-02,
                -2.0464e+00,
                9.5134e-02,
                2.4619e+00,
                7.4044e-01
            ]
        )
    )
    default_action_values = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.03])
    for AIDX in range(7):
        obs = robot.reset_to_state(init_state)
        AVAL = default_action_values[AIDX]
        action = np.zeros((1, 7))
        action[:, AIDX] = AVAL
        for i in tqdm(range(100)):
            if i == 50:
                action[:, AIDX] = -AVAL
            obs, info = robot.execute_delta_action(RobotEEDeltaAction(action))
            history.push(obs)
        history.store(f"action_history_{AIDX}.mp4")
