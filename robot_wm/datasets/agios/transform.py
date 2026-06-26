"""
References:
    https://droid-dataset.github.io/droid/the-droid-dataset.html#-dataset-schema
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from scipy.interpolate import interp1d
from torchvision.transforms.v2 import functional as F

from robot_wm.datasets.droid.transform import DroidTransform
from robot_wm.datasets.utils import _get_T_delta, gram_schmidt

logger = logging.getLogger(__name__)

AGIOS_RGB_CAMERAS = [
    "ego_centric_image",
    "wrist_image_left",
    "wrist_image_right",
]
AGIOS_ACTION_TYPES = [
    "state",
    "delta-state",
    "state+touch",
    "delta-state+touch",
    "delta-state+touch+camera",
    "delta-state+camera",
    "delta-state+camera+abs_finger",
]


def _get_delta(state: np.ndarray, abs_finger=False) -> np.ndarray:
    finger_pose = state[:, 9:]
    if abs_finger:
        finger_pose = finger_pose.reshape(-1, 20, 3)
        finger_pose = (
            finger_pose - state[:, :3][:, None, :3]
        )  # finger pose relative to wrist
        finger_pose = finger_pose.reshape(-1, 20 * 3)
        finger_delta = np.zeros_like(finger_pose)
        finger_delta[:-1] = finger_pose[1:]
    else:
        finger_delta = np.zeros_like(finger_pose)
        finger_delta[:-1] = finger_pose[1:] - finger_pose[:-1]

    hand_delta = _get_T_delta(state[:, :9])
    delta = np.concatenate([hand_delta, finger_delta], axis=-1)
    return delta


def _integrate_chunks(
    state: torch.Tensor, actions: torch.Tensor, action_type: str
) -> torch.Tensor:
    assert state.ndim == 3
    assert state.shape == actions.shape
    assert action_type == "delta-state"
    state = state[:, 0, :].numpy()
    actions = _get_delta(state)
    state = torch.from_numpy(state).unsqueeze(1)
    actions = torch.from_numpy(actions).unsqueeze(1)
    return state, actions


def interpolate_6d_state(states, factor, state_dim):
    """
    Linearly interpolate states
    """
    T, M = states.shape
    t_original = np.arange(T)
    t_interp = np.linspace(0, T - 1, T * factor)
    f = interp1d(t_original, states, kind="linear", axis=0)
    interpolated = f(t_interp)  # shape: (T * factor, M)

    # for the rot6d, run gram_schmidt
    left_rot6d = interpolated[:, 3:9].reshape(-1, 2, 3)
    right_rot6d = interpolated[:, state_dim + 3 : state_dim + 9].reshape(-1, 2, 3)
    if not np.allclose(np.sum(left_rot6d[:, 0] * left_rot6d[:, 1], axis=-1), 0):
        left_rot6d = gram_schmidt(left_rot6d)
    left_rot6d = left_rot6d.reshape(-1, 6)

    if not np.allclose(np.sum(right_rot6d[:, 0] * right_rot6d[:, 1], axis=-1), 0):
        right_rot6d = gram_schmidt(right_rot6d)
    right_rot6d = right_rot6d.reshape(-1, 6)

    interpolated[:, 3:9] = left_rot6d
    if state_dim > 0:
        interpolated[:, state_dim + 3 : state_dim + 9] = right_rot6d

    return interpolated


class ResizeThenCenterCrop:
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, img):
        # Get original dimensions
        orig_width, orig_height = img.shape[-1], img.shape[-2]

        # Compute scale factor to ensure both dimensions > target
        scale_h = self.target_height / orig_height
        scale_w = self.target_width / orig_width
        scale = max(scale_h, scale_w)

        # Compute new size keeping aspect ratio
        new_width = int(round(orig_width * scale))
        new_height = int(round(orig_height * scale))

        # Resize and center crop
        img_resized = F.resize(
            img,
            (new_height, new_width),
            max_size=None,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias="warn",
        )
        img_cropped = F.center_crop(
            img_resized, (self.target_height, self.target_width)
        )
        return img_cropped


class AgiosTransform(DroidTransform):
    def __init__(
        self,
        action_type: str = "delta-state",
        integrate_chunks: bool = False,
        multiview: bool = False,
        num_views=None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        resize_to=None,
        hand_state_dim=9 + 22,
        hand_pos_scale=1000.0,
        **kwargs,
    ):
        if integrate_chunks and action_type != "delta-state":
            raise ValueError(
                "`integrate_chunks` can only be used with action_type='delta-state'."
            )

        if multiview:
            raise NotImplementedError(
                "Multiview is not supported for AgiosTransform (AGIOS 20)."
            )

        super().__init__(
            action_type=action_type,
            integrate_chunks=integrate_chunks,
            multiview=multiview,
            **kwargs,
        )

        self._num_views = num_views
        self._rgb_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (T, C, H, W)
                ResizeThenCenterCrop(
                    target_height=resize_to[0], target_width=resize_to[1]
                )
                if resize_to is not None
                else transforms.Lambda(lambda x: x),  # resize
                transforms.Normalize(mean=mean, std=std, inplace=True)
                if mean is not None
                else transforms.Lambda(lambda x: x.clamp(0, 1)),  # normalize
            ]
        )

        self.include_touch = "touch" in self._action_type
        self.quantile_range = None
        self.interpolation_step = kwargs.get("interpolation_step", 1)
        self.hand_state_dim = hand_state_dim
        self.hand_pos_scale = hand_pos_scale

    def _get_rgb(self, episode: dict[str, Any], camera: str) -> np.ndarray:
        """Returns data from the selected `camera`."""
        assert camera in AGIOS_RGB_CAMERAS
        # interpolate the observation data by repeating each frame interleavedly by `self.interpolation_step`
        rgb = episode["episode_data"]["observation"][camera]
        # if self.interpolation_step > 1:
        #     rgb = np.repeat(rgb, self.interpolation_step, axis=0)
        return rgb

    def _get_state(self, episode: dict[str, Any], include_tow) -> np.ndarray:
        """Returns the robot state."""
        hand_state_dim = self.hand_state_dim
        cartesian_pos_left = episode["episode_data"]["observation"][
            "ekf_hand_pose_left/WristState"
        ]  # 9

        cartesian_pos_right = episode["episode_data"]["observation"][
            "ekf_hand_pose_right/WristState"
        ]  # 9

        finger_pos_left = episode["episode_data"]["observation"][
            "ekf_hand_pose_left/JointAngles"
        ]  # 22
        finger_pos_right = episode["episode_data"]["observation"][
            "ekf_hand_pose_right/JointAngles"
        ]  # 22
        state = np.concatenate(
            [
                cartesian_pos_left,
                finger_pos_left,
                cartesian_pos_right,
                finger_pos_right,
            ],
            axis=-1,
        )
        state[:, :3] /= self.hand_pos_scale
        state[:, hand_state_dim : hand_state_dim + 3] /= self.hand_pos_scale

        tow = np.zeros((state.shape[0], 10))
        if include_tow:
            tow[:, 5:] = episode["episode_data"]["observation"]["tow_raw"]

        state = np.concatenate([state, tow], axis=-1)

        if self.quantile_range is not None and "state" in self.quantile_range:
            state_dim = state.shape[1]
            q1, q99 = self.quantile_range["state"][0], self.quantile_range["state"][1]
            q1, q99 = np.array(q1)[:state_dim], np.array(q99)[:state_dim]
            mask = (state >= q1) & (state <= q99)
            state = np.where(mask, state, 0)

        # interpolate the state data by linearly interpolating each frame interleavedly by `self.interpolation_step`
        if self.interpolation_step > 1:
            state = interpolate_6d_state(
                state, self.interpolation_step, self.hand_state_dim
            )

        return state

    def _get_actions(
        self, episode: dict[str, Any], action_type: str = "action", include_touch=False
    ) -> np.ndarray:
        assert action_type in AGIOS_ACTION_TYPES, f"Invalid action_type: {action_type}"
        hand_state_dim = self.hand_state_dim
        include_tow = "touch" in action_type or include_touch
        state = self._get_state(episode, include_tow)

        if "delta" not in action_type:
            # For non-delta action types, return next-state (shifted forward)
            action = np.zeros_like(state)
            action[:-1] = state[1:]

        else:
            # Handle delta-state action
            left_state = state[:, :hand_state_dim]
            right_state = state[:, hand_state_dim : hand_state_dim * 2]

            left_delta = _get_delta(left_state, abs_finger="abs_finger" in action_type)
            right_delta = _get_delta(
                right_state, abs_finger="abs_finger" in action_type
            )
            action = np.concatenate([left_delta, right_delta], axis=1)

            touch = state[:, hand_state_dim * 2 :]  # shape: (T, D_touch)
            touch_delta = np.zeros_like(touch)
            touch_delta[:-1] = touch[1:] - touch[:-1]
            action = np.concatenate([action, touch_delta], axis=1)

        if "camera" in action_type:
            if "camera" not in episode["episode_data"]["observation"]:
                # concatenate the zeros camera motion of 9d to the action
                camera_action = np.zeros((action.shape[0], 9))
            else:
                camera_action = episode["episode_data"]["observation"]["camera"]
                if "delta" in action_type:
                    camera_action = _get_T_delta(camera_action)

                if self.interpolation_step > 1:
                    camera_action = interpolate_6d_state(
                        camera_action, self.interpolation_step, 0
                    )
            action = np.concatenate([action, camera_action], axis=-1)

        if self.quantile_range is not None and "action" in self.quantile_range:
            action_dim = action.shape[1]
            q1, q99 = self.quantile_range["action"][0], self.quantile_range["action"][1]
            q1, q99 = np.array(q1)[:action_dim], np.array(q99)[:action_dim]
            mask = (action >= q1) & (action <= q99)
            action = np.where(mask, action, 0)
        return action

    def __call__(
        self,
        episode: dict[str, Any],
        output_keys=None,
        include_touch=False,
        skip_rgb=False,
    ) -> dict[str, Any]:
        state = self._get_state(episode, self.include_touch or include_touch)
        actions = self._get_actions(
            episode, action_type=self._action_type, include_touch=include_touch
        )
        rgb_chunk_size = int(self._chunk_size / self.interpolation_step)
        # select a camera
        if not skip_rgb:
            if self._multiview:
                # vertically stacks all cameras in random order
                rgbs_h5 = []
                shuffled_camera_idxs = torch.randperm(
                    len(self._cameras), generator=self._gen
                ).tolist()
                for camera_idx in shuffled_camera_idxs:
                    camera = self._cameras[camera_idx]
                    rgb_h5 = self._get_rgb(episode, camera=camera)
                    rgbs_h5.append(rgb_h5)
                    assert (
                        rgb_h5.shape[3] == 3
                    ), f"Invalid {camera = }, {rgb_h5.shape = }"

                # rgb = np.concatenate(rgbs_h5, axis=1)  # (T, H, W, C)
            else:
                low, high = 0, len(self._cameras)
                camera_idx = torch.randint(low, high, (1,), generator=self._gen).item()
                camera = self._cameras[camera_idx]

                # get data
                rgb_h5 = self._get_rgb(episode, camera=camera)
        assert len(state) == len(actions)

        # sample chunk
        # Note: `high` is set such that we never include the last timestep in
        # the chunk, because the action at the last timestep may not be valid.
        # Specifically, when using 'delta' actions the last timestep actions
        # will be all zeros. Some applications (such as world modeling) may not
        # require a valid last action, and in these cases this choice is
        # suboptimal.
        T = len(state)  # e.g. 100
        total_size = self._sample_size * self._chunk_size  # e.g. 10
        rgb_total_size = self._sample_size * rgb_chunk_size
        low, high = 0, max(
            1, T - total_size - 1
        )  # -total_size to fit the whole history, -1 to leave at least 1 future goal image # e.g. (0, 89)
        start_idx = torch.randint(
            low, high, (1,), generator=self._gen
        ).item()  # e.g. 88
        rgb_start_idx = start_idx // self.interpolation_step

        if not skip_rgb:
            if self._multiview:
                rgb = np.concatenate(
                    [
                        rgb_h5[
                            rgb_start_idx : rgb_start_idx
                            + rgb_total_size : rgb_chunk_size
                        ]
                        for rgb_h5 in rgbs_h5
                    ],
                    axis=1,
                )  # (T, n*H, W, C)
            else:
                rgb = rgb_h5[
                    rgb_start_idx : rgb_start_idx + rgb_total_size : rgb_chunk_size
                ]  # (T, H, W, C)
        state = state[start_idx : start_idx + total_size]  # (T, D)
        actions = actions[start_idx : start_idx + total_size]  # (T, D)

        goal_rgb = None
        if self._goal_type == "image" and not skip_rgb:
            end_idx = start_idx + total_size  # e.g. 98
            min_goal_idx = end_idx + self._min_image_goal_horizon
            max_goal_idx = end_idx + self._max_image_goal_horizon
            if self._max_image_goal_horizon == -1:
                max_goal_idx = T
            min_goal_idx = min(min_goal_idx, T - 1)
            max_goal_idx = min(max_goal_idx, T - 1)
            goal_idx = torch.randint(
                min_goal_idx, max_goal_idx + 1, (1,), generator=self._gen
            ).item()
            if self._multiview:
                goal_rgb = np.concatenate(
                    [rgb_h5[goal_idx : goal_idx + 1] for rgb_h5 in rgbs_h5], axis=1
                )  # (1, n*H, W, C)
            else:
                goal_rgb = rgb_h5[goal_idx : goal_idx + 1]  # (1, H, W, C)

        if not skip_rgb:
            rgb = self._to_tensor(rgb)
            goal_rgb = self._to_tensor(goal_rgb)
        state = self._to_tensor(state)
        actions = self._to_tensor(actions)

        if not skip_rgb:
            mask = torch.ones(rgb.shape[:1], dtype=bool)
            # apply padding (if needed)
            if len(rgb) != self._sample_size:
                pad = self._sample_size - len(rgb)
                rgb = nn.functional.pad(rgb, 6 * [0] + [0, pad])
                mask = nn.functional.pad(mask, [0, pad])
        if len(state) != total_size:
            pad = total_size - len(state)
            state = nn.functional.pad(state, 2 * [0] + [0, pad])
        if len(actions) != total_size:
            pad = total_size - len(actions)
            actions = nn.functional.pad(actions, 2 * [0] + [0, pad])

        # reshape state and actions
        state = state.reshape(self._sample_size, self._chunk_size, -1)
        actions = actions.reshape(self._sample_size, self._chunk_size, -1)

        # integrate action chunks
        if self._integrate_chunks:
            state, actions = _integrate_chunks(state, actions, self._action_type)

        # apply transforms
        goal = None
        if not skip_rgb:
            rgb = self._rgb_transform(rgb)
            if self._num_views is not None and self._num_views > 1:
                rgb = torch.concatenate(
                    [rgb] + [torch.zeros_like(rgb)] * (self._num_views - 1), dim=-1
                )  # (T, C, H, num_views*W) -- ego + padded views
            if self._goal_type == "image":
                goal = self._rgb_transform(goal_rgb)
        state = self._state_transform(state)
        actions = self._actions_transform(actions)

        output = {
            "rgb": rgb if not skip_rgb else None,
            "mask": mask if not skip_rgb else None,
            "state": state,
            "actions": actions,
            "goal": goal,
        }
        if output_keys is not None:
            return {k: v for k, v in output.items() if k in output_keys}

        return {k: v for k, v in output.items() if k in self._output_keys}
