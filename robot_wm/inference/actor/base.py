import os
from dataclasses import dataclass, field
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_wm.inference.robot.base import RobotAction, RobotEEDeltaAction, RobotObs
from robot_wm.utils.notebook import create_video_from_frames, display_video_in_notebook


@dataclass
class Action:
    pass


@dataclass
class Actor:
    pass


@dataclass
class ConcatAction(RobotAction):
    action: np.ndarray
    prechunk_action_dim: int


@dataclass
class RobotObsHistory:
    max_context: int
    freq: int  # hz
    obs: List[RobotObs] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.max_context, int) or self.max_context <= 0:
            raise ValueError(
                f"max_context must be a positive integer, got {self.max_context}"
            )
        if not isinstance(self.freq, int) or self.freq <= 0:
            raise ValueError(f"freq must be a positive integer, got {self.freq}")

        if not isinstance(self.obs, list):
            raise TypeError("obs must be a list")

        # for i, item in enumerate(self.obs):
        #     if not isinstance(item, RobotObs):
        # raise TypeError(f"All items in obs must be RobotObs instances, item at index {i} is {type(item)}")

    def __repr__(self):
        return f"RobotObsHistory(freq={self.freq}, max_context={self.max_context}, n_observations_in_the_context={len(self.obs)})"

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.obs[key]
        elif isinstance(key, slice):
            sliced_obs = self.obs[key]
            return RobotObsHistory(
                max_context=self.max_context, freq=self.freq, obs=sliced_obs
            )
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Expected int or slice.")

    def reset(self):
        self.obs = []

    def push(self, obs: RobotObs):
        assert isinstance(
            obs, RobotObs
        ), f"obs must be a RobotObs instance, got {type(obs)}"
        self.obs.append(obs)
        if len(self.obs) > self.max_context:
            self.obs.pop(0)

    def get_subsampled_history(self, max_context: int, skip: int = 1):
        """
        Returns a subset of observations by sampling every `skip`-th observation
        from the last `max_context * skip` observations.

        Args:
            max_context: The maximum number of observations to return.
            skip: Return every `skip`-th observation. Default is 1 (no skipping).

        Returns:
            A list of RobotObs instances, ordered from oldest to most recent.
        """
        # Ensure max_context and skip are positive integers
        if not isinstance(max_context, int) or max_context <= 0:
            raise ValueError(
                f"max_context must be a positive integer, got {max_context}"
            )
        if not isinstance(skip, int) or skip <= 0:
            raise ValueError(f"skip must be a positive integer, got {skip}")

        # Calculate the window size
        window_size = max_context * skip

        # Get the window of observations
        window = self.obs[-window_size:] if len(self.obs) >= window_size else self.obs

        # Return every `skip`-th observation from the window
        return RobotObsHistory(
            max_context=max_context, freq=self.freq // skip, obs=window[::skip]
        )

    def store(self, filepath="robot_history.mp4", caption=None):
        if len(self.obs) == 0:
            print("No observations to save.")
            return None

        def get_text_info(i):
            text = ""
            if caption is not None:
                text += f"{caption}\n"
            text += f"Frame: {i}/{len(self.obs)-1}"
            if (
                hasattr(self.obs[i], "end_effector_pose")
                and self.obs[i].end_effector_pose is not None
            ):
                text += f"\nEE pos: {self.obs[i].end_effector_pose.round(2)}"
            return text

        return create_video_from_frames(
            frames=[obs.image for obs in self.obs],
            fps=self.freq,
            filepath=filepath,
            text_info_fn=get_text_info,
        )   

    def store_side_by_side(list_of_histories, filepath="side_by_side.mp4"):
        video_paths = []
        for idx, history in enumerate(list_of_histories):
            tmp_filepath = filepath + f"temp_{idx}.mp4"
            video_paths.append(history.store(filepath=tmp_filepath))

        # Load the videos and concatenate
        from moviepy.editor import VideoClip, VideoFileClip, clips_array
        # Load video clips
        clips = []
        for video_path in video_paths:
            clips.append(VideoFileClip(video_path))
        
        # Determine the target resolution
        target_height = min([clip.h for clip in clips])
        
        # Resize clips to match target height while preserving aspect ratio
        clips_resized = []
        for clip in clips:
            # clips_resized.append(clip.resize(height=target_height))
            clips_resized.append(clip)
        
        # Calculate the duration of the output video (take the shorter one)
        final_duration = min([clip.duration for clip in clips])
        
        # Trim both clips to the same duration
        clips_final = []
        for clip in clips_resized:
            clips_final.append(clip.subclip(0, final_duration))
        
        print(f"Processing videos with duration: {final_duration:.2f} seconds")
        
        # Arrange clips side by side
        final_clip = clips_array([clips_final]) # 2d array
        
        # Make sure we have a valid fps value (use fps from first clip, or default to 24 if not available)
        fps = clips_final[0].fps
        if fps is None:
            print("Warning: FPS not detected in source video. Using default fps=24.")
            fps = 24

        # Write the result to a file
        print(f"Writing output to {filepath}...")
        final_clip.write_videofile(
            filepath,
            # codec='libx264',
            fps=fps,
            # audio_codec='aac',
            # temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Close the clips to free up resources
        for clip in clips:
            clip.close()

        # try to delete clips
        for video_path in video_paths:
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"WARNING: Failed to delete {video_path}: {e}")
        
        print("Done!")
        return filepath

    def store_gif(self, filepath="robot_history.gif"): 
        if len(self.obs) == 0:
            print("No observations to save.")
            return None

        img_shape = self.obs[0].image.shape
        aspect_ratio = img_shape[2] / img_shape[1]  # width / height

        # Create a figure with dimensions that match the image aspect ratio
        fig = plt.figure(figsize=(8, 8 / aspect_ratio))
        ax = plt.gca()
        ax.set_axis_off()

        img_array = self.obs[0].image.transpose(
            1, 2, 0
        )  # Convert from (3,H,W) to (H,W,3)
        img_plot = ax.imshow(img_array)

        text = ax.text(8, 8, "", color="white", backgroundcolor="black")

        def update(frame):
            img = self.obs[frame].image.transpose(1, 2, 0)
            img_plot.set_array(img)
            text.set_text(
                f"Frame: {frame}/{len(self.obs)-1}\n"
                f"EE pos: {self.obs[frame].end_effector_pose.round(2)}"
            )
            return [img_plot, text]

        # Create and save animation as GIF
        ani = animation.FuncAnimation(fig, update, frames=len(self.obs), blit=True)
        ani.save(filepath, writer=animation.PillowWriter(fps=self.freq))

    def show(self, filepath=None):
        _filepath = "temp.mp4" if filepath is None else filepath
        _filepath = self.store(filepath=_filepath)
        display_video_in_notebook(_filepath)
        if filepath is None:
            os.remove(_filepath)

    # TODO: The following functions depend on the RobotObs class attributes
    # making this abstraction less general. Consider refactoring this on the future
    # when we move to other robot embodiements.
    @property
    def joints_tensor(self):
        return torch.tensor([obs.joints for obs in self.obs])

    @property
    def end_effector_pose_tensor(self):
        return torch.tensor([obs.end_effector_pose for obs in self.obs])

    @property
    def image_tensor(self):
        return torch.tensor([obs.image for obs in self.obs])

    def get_ee_pose_deltas(self):
        """
        Returns a list of end effector pose deltas between consecutive observations.

        Returns:
            A list of RobotEEDeltaAction instances, ordered from oldest to most recent.
        """
        ee_poses = self.end_effector_pose_tensor
        delta_state = ee_poses[1:] - ee_poses[:-1]
        delta_state[delta_state > 5] -= 2 * np.pi
        delta_state[delta_state < -5] += 2 * np.pi
        return delta_state

    def get_action_history_from_ee_deltas(self):
        ee_deltas = self.get_ee_pose_deltas().unsqueeze(0)
        action_history = RobotActionHistory.from_tensor(
            max_context=self.max_context, freq=self.freq, tensor=ee_deltas
        )
        return action_history

    def to_h5(self, h5_path="episode.h5"):
        """
        Converts the history to an h5 file with the DROID structure
        """
        import h5py
        RCAM = "right_camera"
        LCAM = "left_camera"
        data = {}
        data["ee_states"] = []
        data["joint_states"] = []
        data["gripper_states"] = []
        data[f"{RCAM}_rgb_images"] = []
        data[f"{LCAM}_rgb_images"] = []
        for obs in self.obs:
            data["ee_states"].append(obs.end_effector_pose[:6])
            data["joint_states"].append(obs.joints)
            data["gripper_states"].append(obs.end_effector_pose[-1])
            image_hwc_255 = (obs.image.transpose(1, 2, 0) * 255).astype("uint8")
            data[f"{RCAM}_rgb_images"].append(image_hwc_255) # T H W C [0, 255] uint 8
            # data[f"{LCAM}_rgb_images"].append(obs.image)
         # Save all the collected data
        with h5py.File(h5_path, "w") as h5py_file:
            grp = h5py_file.create_group("episode_data")
            subgrp = grp.create_group("observation")
            
            # Save configuration info
            # subgrp.attrs["config"] = json.dumps(config)
            
            # Save robot state data
            subgrp.create_dataset("cartesian_position", data=np.array(data["ee_states"]))
            subgrp.create_dataset("joint_position", data=np.array(data["joint_states"]))
            print(f"Saved {len(data['ee_states'])} end-effector states")
            print(f"Saved {len(data['joint_states'])} joint states")
            
            if data["gripper_states"]:
                subgrp.create_dataset("gripper_position", data=np.array(data["gripper_states"]))
                print(f"Saved {len(data['gripper_states'])} gripper states")
            
            # Save image data if available
            if data[f"{RCAM}_rgb_images"]:
                rgb_data = np.array(data[f"{RCAM}_rgb_images"])
                subgrp.create_dataset("exterior_image_1_left", data=rgb_data, 
                                compression="gzip", compression_opts=4)
                print(f"Saved {len(rgb_data)} {RCAM} RGB images")
            if data[f"{LCAM}_rgb_images"]:
                rgb_data = np.array(data[f"{LCAM}_rgb_images"])
                subgrp.create_dataset("exterior_image_2_left", data=rgb_data, 
                                compression="gzip", compression_opts=4)
                print(f"Saved {len(rgb_data)} {LCAM} RGB images")
        
        print(f"Data saved to {h5_path}")


@dataclass
class RobotActionHistory:
    max_context: int
    freq: int  # hz
    actions: List[RobotAction] = field(default_factory=list)

    def __post_init__(self):
        # Validate max_context and freq are positive integer
        if not isinstance(self.max_context, int) or self.max_context <= 0:
            raise ValueError(
                f"max_context must be a positive integer, got {self.max_context}"
            )
        if not isinstance(self.freq, int) or self.freq <= 0:
            raise ValueError(f"freq must be a positive integer, got {self.freq}")

        # Validate actions is a list
        if not isinstance(self.actions, list):
            raise TypeError("actions must be a list")

        # Validate all items in actions are RobotAction instances
        for i, item in enumerate(self.actions):
            if not isinstance(item, (RobotAction, ConcatAction)):
                raise TypeError(
                    f"All items in actions must be RobotAction or ConcatAction instances, item at index {i} is {type(item)}"
                )

        # Validate action_dim consistency if there are any actions
        if len(self.actions) > 1:
            action_dim = self.actions[0].action_dim
            for i, action in enumerate(self.actions[1:], 1):
                if action.action_dim != action_dim:
                    raise ValueError(
                        f"All actions must have the same action_dim. Expected {action_dim}, got {action.action_dim} at index {i}"
                    )

    def reset(self):
        self.actions = []

    def aggregate_first_mpc_action(self, horizon_s: float, aggregation="sum"):
        n_actions = int(horizon_s * self.freq)
        assert aggregation == "sum"
        return sum(self.actions[:n_actions])

    @classmethod
    def from_tensor(cls, max_context: int, freq: int, tensor: torch.Tensor):
        assert (
            tensor.dim() == 3
        ), f"tensor must have 3 dimensions (batch_size, seq_len, action_dim), got {tensor.dim()}"
        return cls(
            max_context=max_context,
            freq=freq,
            actions=[
                RobotAction(action.cpu().numpy()) for action in tensor.permute(1, 0, 2)
            ],
        )

    def push(self, action: RobotAction):
        assert isinstance(
            action, RobotAction
        ), f"action must be a RobotAction instance, got {type(action)}"

        # Check action_dim consistency if there are existing actions
        if self.actions and action.action_dim != self.action_dim:
            raise ValueError(
                f"New action has inconsistent action_dim. Expected {self.action_dim}, got {action.action_dim}"
            )

        self.actions.append(action)
        if len(self.actions) > self.max_context:
            self.actions.pop(0)

    def plot_cumulative(self, show=True, save_path=None):
        # pass
        actions_tensor = self.actions_tensor.numpy()  # (B, T, A)
        B, T, A = actions_tensor.shape
        cumulative_actions_tensor = np.cumsum(actions_tensor, axis=1)
        # create A subplots
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(A, 1, figsize=(10, 5))
        for a in range(A):
            for b in range(B):
                axs[a].plot(cumulative_actions_tensor[b, :, a])
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        return fig, axs

    def chunk(
        self,
        max_context: int,
        chunk_size: int = 1,
        aggregation="concat",
        from_last=True,
        fill_missing=RobotEEDeltaAction.ZERO(),
    ):
        """
        Returns a subset of actions by getting every `chunk_size`-th chunk of actions
        from the last `max_context * chunk_size` actions. Every chunk is either concatenated,
        averaged or summed.

        Args:
            max_context: The maximum number of actions to return.
            chunk_size: Return every `chunk_size`-th chunk of actions. Default is 1 (no skipping).
            aggregation: The operation to perform on each chunk. Options are 'concat', 'average', 'sum'.
            from_last: If True, get the last `max_context * chunk_size` actions, otherwise get the first `max_context * chunk_size` actions.
        """
        assert aggregation in ["concat", "average", "sum"]
        assert isinstance(max_context, int) and max_context > 0
        assert isinstance(chunk_size, int) and chunk_size > 0

        if len(self.actions) == 0:
            return RobotActionHistory(
                max_context=max_context, freq=self.freq // chunk_size, actions=[]
            )

        # Get chunk
        window_size = max_context * chunk_size
        window_size = min(window_size, len(self.actions))
        if from_last:
            window = self.actions[-window_size:]
        else:
            window = self.actions[:window_size]

        chunks = [window[i : i + chunk_size] for i in range(0, len(window), chunk_size)]

        # Pad
        S = self.actions[0].action.shape[0]
        fill_missing_tiled = fill_missing.batch_tile(S)
        chunks[-1] = chunks[-1] + [
            fill_missing_tiled for _ in range(chunk_size - len(chunks[-1]))
        ]

        # Aggregate
        if aggregation == "concat":
            chunked_actions = []
            for chunk in chunks:
                chunk_action = np.concatenate(
                    [action.action for action in chunk], axis=1
                )
                chunked_actions.append(
                    ConcatAction(chunk_action, prechunk_action_dim=self.action_dim)
                )
        elif aggregation == "average":
            chunked_actions = [sum(chunk) / len(chunk) for chunk in chunks]
        elif aggregation == "sum":
            chunked_actions = [sum(chunk) for chunk in chunks]

        return RobotActionHistory(
            max_context=max_context,
            freq=self.freq // chunk_size,
            actions=chunked_actions,
        )

    @property
    def action_dim(self):
        return self.actions[0].action_dim

    @property
    def actions_tensor(self):
        if len(self.actions) == 0:
            return torch.zeros(0, 0, 0)
        return torch.tensor([action.action for action in self.actions]).permute(
            1, 0, 2
        )  # (T, B, A) -> (B, T, A)

    def __repr__(self):
        return f"RobotActionHistory(freq={self.freq}, max_context={self.max_context}, n_actions_in_the_context={len(self.actions)})"

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.actions[key]
        elif isinstance(key, slice):
            sliced_actions = self.actions[key]
            return RobotActionHistory(
                max_context=self.max_context, freq=self.freq, actions=sliced_actions
            )
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Expected int or slice.")
