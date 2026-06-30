"""
References:
    https://droid-dataset.github.io/droid/the-droid-dataset.html#-dataset-schema
"""

import json
import logging
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

import robot_wm.utils.distributed as dist
from robot_wm.datasets.base import Transform
from robot_wm.datasets.utils import _get_T_delta, euler_to_rot_6d, random_wrist_mask, pad_to_num_views

logger = logging.getLogger(__name__)


def _divide_255(x):
    return x / 255.0


def _multiply_255(x):
    return x * 255.0


def _thwc_to_tchw(x):
    return x.permute(0, 3, 1, 2)


def _tchw_to_thwc(x):
    return x.permute(0, 2, 3, 1)


def _identity(x):
    return x


def _clamp_01(x):
    return x.clamp(0, 1)

DROID_RGB_CAMERAS = [
    "wrist_image_left",
    "exterior_image_1_left",
    "exterior_image_2_left",
]

MURP_RGB_CAMERAS = [
    "head_rgb",
    "torso_rgb",
    "right_wrist_rgb",
]

DROID_ACTION_TYPES = [
    "action",
    "action+camera",
    "delta-action",
    "delta-action+camera",
    "state",
    "state+camera",
    "delta-state",
    "delta-state+camera",
    "delta-state+camera+abs_finger",
]
STATE_DIM = 10


def _get_action(episode: dict[str, Any]) -> np.ndarray:
    """Returns the robot action."""
    # get the action
    action = episode["episode_data"]["action"]  # xyz, euler angles, gripper position
    # get the gripper position
    gripper_pos = action[:, 6:]
    # get the cartesian position
    cartesian_pos = action[:, :6]
    # convert cartesian_pos to 3d translation and 6d rotation
    euler_angles = cartesian_pos[:, 3:6]
    rot_6d = euler_to_rot_6d(euler_angles)
    vect9d = np.concatenate([cartesian_pos[:, :3], rot_6d], axis=1)

    action = np.concatenate([vect9d, gripper_pos], axis=1)
    return action


def _get_delta(values: np.ndarray, abs_finger=False, values_type="state") -> np.ndarray:
    """
    values: (T, 10)
    """
    vec9d = values[:, :9]
    gripper_pos = values[:, 9:]
    ee_delta = _get_T_delta(vec9d)
    gripper_delta = np.zeros_like(gripper_pos)

    if abs_finger:
        if values_type == "state":
            gripper_delta[:-1] = gripper_pos[1:]
        elif values_type == "action":
            gripper_delta = gripper_pos
    else:
        gripper_delta[:-1] = gripper_pos[1:] - gripper_pos[:-1]
    delta = np.concatenate([ee_delta, gripper_delta], axis=1)
    return delta


def _integrate_chunks(
    state: torch.Tensor,
    actions: torch.Tensor,
    action_type: str,
) -> torch.Tensor:
    assert state.ndim == 3
    assert state.shape == actions.shape
    assert action_type == "delta-state"
    state = state[:, 0, :].numpy()
    actions = _get_delta(state)
    state = torch.from_numpy(state).unsqueeze(1)
    actions = torch.from_numpy(actions).unsqueeze(1)
    return state, actions


class DroidTransform(Transform):
    def __init__(
        self,
        cameras: Sequence[str],
        output_keys: Sequence[str],
        sample_size: int,
        chunk_size: int = 1,
        action_type: str = "delta-state",
        integrate_chunks: bool = False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        resize_to=None,
        random_crop=None,  # if not None, will apply random crop as percentage of image size. 0.8 means random crop down to 80% of the original image size
        random_stretch=None,
        seed=0,
        multiview=False,
        wrist_mask_prob: float = 0.0,
        num_views=None,
        separate_views=False,
        try_to_order_cameras=False,
        goal_type="none",
        min_image_goal_horizon=1,
        max_image_goal_horizon=-1,
        **kwargs,
    ):
        assert not integrate_chunks or action_type == "delta-state"

        self._cameras = cameras
        self._multiview = multiview
        self._wrist_mask_prob = wrist_mask_prob
        self._num_views = num_views
        self._separate_views = separate_views
        self._try_to_order_cameras = try_to_order_cameras
        self._output_keys = output_keys
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._action_type = action_type
        self._integrate_chunks = integrate_chunks
        self._mean = mean
        self._std = std
        self._resize_to = resize_to
        self._random_crop = random_crop
        self._random_stretch = random_stretch
        self._seed = seed
        self._goal_type = goal_type
        self._min_image_goal_horizon = min_image_goal_horizon
        self._max_image_goal_horizon = max_image_goal_horizon

        self._rgb_transform = transforms.Compose(
            [
                transforms.Lambda(_divide_255),
                transforms.Lambda(_thwc_to_tchw),  # (T, C, H, W)
                transforms.Resize(tuple(int(x) for x in resize_to))
                if resize_to is not None
                else transforms.Lambda(_identity),  # resize
                transforms.Normalize(mean=mean, std=std, inplace=True)
                if mean is not None
                else transforms.Lambda(_clamp_01),  # normalize
            ]
        )
        self._state_transform = transforms.Identity()
        self._actions_transform = transforms.Identity()

        self._gen = torch.Generator()
        if dist.is_initialized():
            self._seed = dist.get_global_rank() + seed
        self._gen.manual_seed(self._seed)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        if x is None:
            return None
        else:
            return torch.from_numpy(x).float()

    @staticmethod
    def _get_rgb(episode: dict[str, Any], camera: str) -> np.ndarray:
        """Returns data from the selected `camera`."""
        assert camera in DROID_RGB_CAMERAS or camera in MURP_RGB_CAMERAS
        return episode["episode_data"]["observation"][camera]

    @staticmethod
    def _get_state(episode: dict[str, Any]) -> np.ndarray:
        """Returns the robot state."""
        cartesian_pos = episode["episode_data"]["observation"][
            "cartesian_position"
        ]  # this is euler angles
        # convert cartesian_pos to 3d translation and 6d rotation
        trans = cartesian_pos[:, :3]
        euler_angles = cartesian_pos[:, 3:6]
        rot_6d = euler_to_rot_6d(euler_angles)
        vect9d = np.concatenate([trans, rot_6d], axis=1)

        gripper_pos = episode["episode_data"]["observation"]["gripper_position"]
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos[:].reshape(vect9d.shape[0], -1)
        state = np.concatenate([vect9d, gripper_pos], axis=1)
        return state

    def _get_actions(
        self,
        episode: dict[str, Any],
        action_type: str = "action",
    ) -> np.ndarray:
        assert action_type in DROID_ACTION_TYPES
        if "action" in action_type:
            action = _get_action(episode)
            if "delta" in action_type:
                action = _get_delta(
                    action, abs_finger="abs_finger" in action_type, values_type="action"
                )
        elif "state" in action_type:
            state = self._get_state(episode)
            if "delta" in action_type:
                action = _get_delta(state, abs_finger="abs_finger" in action_type)
        else:
            raise ValueError(f"invalid {action_type = }")

        if "camera" in action_type:
            camera_motion = np.zeros((action.shape[0], 9))
            action = np.concatenate([action, camera_motion], axis=-1)
        return action

    def __call__(self, episode: dict[str, Any], output_keys=None) -> dict[str, Any]:
        state = self._get_state(episode)
        actions = self._get_actions(
            episode,
            action_type=self._action_type,
        )
        # select a camera
        if self._multiview:
            # multiview: keep DETERMINISTIC config order (ego/exterior first, then
            # wrist) so the stacked grid stays consistent with rgb_pos_embed
            rgbs_h5 = []
            rgbs_names = []
            shuffled_camera_idxs = list(range(len(self._cameras)))
            # Try to get camera order from metadata.json
            if self._try_to_order_cameras:
                shuffled_camera_idxs = range(
                    len(self._cameras)
                )  # if we can't get metadata, at least follow the config order
                try:
                    metadata_filepath = episode.filename.replace(
                        "episode.h5", "metadata.json"
                    )
                    with open(metadata_filepath, "r") as f:
                        metadata = json.load(f)
                    desired_order = ["right", "left", "wrist"]  # MPK order
                    is_1_left = (
                        metadata["ext1_cam_extrinsics"][1]
                        > metadata["ext2_cam_extrinsics"][1]
                    )
                    if is_1_left:
                        cam_mapping = {
                            "left": "exterior_image_1_left",
                            "right": "exterior_image_2_left",
                            "wrist": "wrist_image_left",
                        }
                    else:
                        cam_mapping = {
                            "left": "exterior_image_2_left",
                            "right": "exterior_image_1_left",
                            "wrist": "wrist_image_left",
                        }
                    ordered_camera_names = [
                        cam_mapping[which_cam]
                        for which_cam in desired_order
                        if cam_mapping[which_cam] in self._cameras
                    ]
                    shuffled_camera_idxs = [
                        self._cameras.index(cam) for cam in ordered_camera_names
                    ]
                except Exception:
                    pass

            for camera_idx in shuffled_camera_idxs:
                camera = self._cameras[camera_idx]
                rgb_h5 = self._get_rgb(episode, camera=camera)
                rgbs_h5.append(rgb_h5)
                rgbs_names.append(camera)
                assert rgb_h5.shape[3] == 3, f"Invalid {camera = }, {rgb_h5.shape = }"
                assert len(rgb_h5) == len(state)
        else:
            low, high = 0, len(self._cameras)
            camera_idx = torch.randint(low, high, (1,), generator=self._gen).item()
            camera = self._cameras[camera_idx]

            # get data
            rgb_h5 = self._get_rgb(episode, camera=camera)
            assert len(rgb_h5) == len(state)
            rgbs_h5 = [rgb_h5]  # only one image
            rgbs_names = [camera]
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
        low, high = 0, max(
            1, T - total_size - 1
        )  # -total_size to fit the whole history, -1 to leave at least 1 future goal image # e.g. (0, 89)
        start_idx = torch.randint(
            low, high, (1,), generator=self._gen
        ).item()  # e.g. 88

        rgbs = [
            rgb_h5[start_idx : start_idx + total_size : self._chunk_size]
            for rgb_h5 in rgbs_h5
        ]  # [(T, H, W, C) * n_cam]
        state = state[start_idx : start_idx + total_size]  # (T, D)
        actions = actions[start_idx : start_idx + total_size]  # (T, D)

        if (not self._multiview) and "wrist" in camera:
            # single-view only: a wrist cam moves with the ee, so camera motion = ee
            # pose. In stacked multiview there is no single wrist view -> leave zeros.
            # set all state other than gripper position to 0
            state[:, :-1] = 0
            # set camera motion to be the first 9 elements of actions
            actions[:, -9:] = actions[:, :9].copy()
            # then set the first 9 elements of actions to be 0
            actions[:, :9] = 0.0

        goal_rgb = None
        if self._goal_type == "image":
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
                goal_rgb = rgbs_h5[0][goal_idx : goal_idx + 1]  # (1, H, W, C)

        # Add random crop augmentations
        if self._random_crop is not None:
            min_ratio = self._random_crop  # [0.0, 1.0]
            max_ratio = 1.0
            assert 0.0 < min_ratio <= max_ratio <= 1.0, "Invalid random crop ratio"
            # Randomly crop each view independently
            cropped_rgbs = []
            for rgb in rgbs:
                # sample random ratio between min and max
                ratio = (
                    torch.FloatTensor(1)
                    .uniform_(min_ratio, max_ratio, generator=self._gen)
                    .item()
                )
                T, H, W, C = rgb.shape
                H_c, W_c = int(H * ratio), int(W * ratio)
                tf_crop_then_resize_to_original = transforms.Compose(
                    [
                        transforms.Lambda(_thwc_to_tchw),  # (T, C, H, W)
                        transforms.Lambda(_divide_255),  # normalize to [0, 1]
                        transforms.RandomCrop(
                            (H_c, W_c)
                        ),  # , generator=self._gen), no generator arg
                        transforms.Resize((H, W)),  # resize to original size
                        transforms.Lambda(_multiply_255),  # rescale to [0, 255]
                        transforms.Lambda(_tchw_to_thwc),  # (T, H, W, C)
                    ]
                )
                rgb_np = tf_crop_then_resize_to_original(
                    torch.from_numpy(rgb).float()
                ).numpy()
                cropped_rgbs.append(rgb_np)
            rgbs = cropped_rgbs

        if self._random_stretch is not None:
            raise NotImplementedError(
                f"Random stretch augmentation is not implemented yet. {self._random_stretch = }"
            )

        # convert to tensors
        rgbs = [self._to_tensor(rgb) for rgb in rgbs]
        state = self._to_tensor(state)
        actions = self._to_tensor(actions)
        goal_rgb = self._to_tensor(goal_rgb)

        # apply padding (if needed)
        mask = torch.ones(rgbs[0].shape[:1], dtype=bool)
        if (
            len(rgbs[0]) != self._sample_size
        ):  # sometimes we don't have enough frames. Add 0s, pad mask with False
            pad = self._sample_size - len(rgbs[0])
            rgbs = [nn.functional.pad(rgb, 6 * [0] + [0, pad]) for rgb in rgbs]
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
            state, actions = _integrate_chunks(
                state,
                actions,
                self._action_type,
            )

        # apply transforms
        if self._separate_views:
            rgbs = [
                self._rgb_transform(rgb) for rgb in rgbs
            ]  # (T, C, H, W) for each camera
            rgb = None
        else:
            rgbs = [self._rgb_transform(r) for r in rgbs]  # per-view -> (T,C,H,W)
            rgbs, rgbs_names = pad_to_num_views(rgbs, rgbs_names, self._num_views)
            rgbs = random_wrist_mask(rgbs, rgbs_names, self._wrist_mask_prob, self._gen)
            rgb = torch.concatenate(rgbs, dim=-1)  # (T, C, H, n*W)
        state = self._state_transform(state)
        actions = self._actions_transform(actions)
        goal = None
        if self._goal_type == "image":
            goal = self._rgb_transform(goal_rgb)

        output = {
            "rgb": rgb,
            "mask": mask,
            "state": state,
            "actions": actions,
            "goal": goal,
        }

        if self._separate_views:
            for name, rgb in zip(rgbs_names, rgbs):
                output[f"{name}"] = rgb

        if output_keys is not None:
            return {k: v for k, v in output.items() if k in output_keys}, camera

        return {k: v for k, v in output.items() if k in self._output_keys}, camera

    def state_dict(self) -> dict[str, Any]:
        return {"_gen": self._gen.get_state()}

    def load_state_dict(self, state_dict):
        self._gen.set_state(state_dict["_gen"])
