import os

# Adding JEPA to the path
import sys
from dataclasses import dataclass
from typing import Tuple

import hydra
import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from projects.mido_jepa_wm_rope.mpc_utils import compute_new_pose
from projects.mido_jepa_wm_rope.transforms import make_transforms

## Linking Mido JEPA code
# 1. Downloaded code from hf (now just in projects)
# from robot_wm.utils.config import get_robot_wm_path
# robot_wm_path = get_robot_wm_path()
# jepa_path = os.path.join(robot_wm_path, "projects", "mido_jepa_wm_hf")
# from huggingface_hub import snapshot_download
# H5_PATH_PARENT = snapshot_download(repo_id="facebook/robotics-world-models", allow_patterns=["*.py", "*.yaml", "*.json"], local_dir=jepa_path)
# 2. git cloned jepa-internal to mido-jepa-internal, commit b1481e45
# cd ~/Code/robot_world_models/projects && git clone git@github.com:fairinternal/jepa-internal.git mido-jepa-internal && cd mido-jepa-internal && git checkout b1481e45
# 3. import
from projects.mido_jepa_wm_rope.utils import init_video_model, load_checkpoint
from projects.mido_jepa_wm_rope.world_model_wrapper import WorldModel as JEPAWorldModel
from robot_wm.inference.actor.base import RobotActionHistory, RobotObsHistory
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)
from robot_wm.inference.world_model.base import WMState, WorldModel
from robot_wm.utils.config import get_robot_wm_path

robot_wm_path = get_robot_wm_path()
jepa_path = os.path.join(robot_wm_path, "projects", "mido-jepa-internal")
sys.path.append(jepa_path)

from app.droid_wm.world_model_heads import (
    WorldModelDiTCosmosHead,
    WorldModelViTImageHead,
)


@dataclass
class JepaWMState:
    action_traj: torch.Tensor  # (B, T-1, 7)
    frame_traj: torch.Tensor  # (B, T, TOK, D)
    pose_traj: torch.Tensor  # (B, T, 7)

    def __init__(self, action_traj, frame_traj, pose_traj):
        self.action_traj = action_traj
        self.frame_traj = frame_traj
        self.pose_traj = pose_traj
        assert (
            len(action_traj.shape) == 3
        ), f"action_traj should be (b, t, 7) but is {action_traj.shape}"
        ba, ta, _7 = action_traj.shape
        assert _7 == 7, f"action_traj should be (b, t, 7) but is {action_traj.shape}"
        assert (
            len(frame_traj.shape) == 4
        ), f"frame_traj should be (b, t, tok, d) but is {frame_traj.shape}"
        bf, tf, tok, d = frame_traj.shape
        assert (
            bf == ba
        ), f"action_traj and frame_traj should have the same batch size but are {ba} and {bf}"
        assert (
            tf == ta + 1
        ), f"frame_traj should have one more time step than action_traj but are {tf} and {ta}"
        assert (
            len(pose_traj.shape) == 3
        ), f"pose_traj should be (b, t, 7) but is {pose_traj.shape}"
        bp, tp, _7 = pose_traj.shape
        assert (
            bp == ba
        ), f"action_traj and pose_traj should have the same batch size but are {ba} and {bp}"
        assert (
            tp == tf
        ), f"pose_traj should have the same time steps as frame_traj but are {tf} and {tp}"
        assert _7 == 7, f"pose_traj should be (b, t, 7) but is {pose_traj.shape}"

    def to(self, *args, **kwargs):
        self.action_traj = self.action_traj.to(*args, **kwargs)
        self.frame_traj = self.frame_traj.to(*args, **kwargs)
        self.pose_traj = self.pose_traj.to(*args, **kwargs)
        return self


class JepaWM(WorldModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.size = config.size
        self.window = config.window
        self.tubulet_size = config.tubulet_size
        self.max_context = config.max_context
        self.wm_frequency = config.wm_frequency
        self.max_rollout_length = config.max_rollout_length
        self.grid_size = config.grid_size
        device = "cuda:0"
        self.device = device
        # chekc if ckpt in the local cache folder already
        config.local_path = (
            "" if not hasattr(config, "local_path") else config.local_path
        )
        if os.path.exists(
            f"{config.local_path}/mido-jepa-wm-rope/{config.resume_path}"
        ):
            checkpoint_path = (
                f"{config.local_path}/mido-jepa-wm-rope/{config.resume_path}"
            )
        else:
            # download checkpoint
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id="facebook/robotics-world-models",
                filename=f"mido-jepa-wm-rope/{config.resume_path}",
            )
            print(f"Checkpoint downloaded to {checkpoint_path}")

        self.tokens_per_frame = int((config.crop_size // config.patch_size) ** 2)
        wm_image_head = WorldModelViTImageHead(
            head_config=dict(
                grid_size=16,
                num_frames=1,
                patch_size=8,
                embed_dim=1408,
                decoder_embed_dim=1024,
                depth=12,
                num_heads=32,
                num_views=1,
                action_tokens=0,
                state_tokens=0,
                local_window=(-1, -1, -1),
                use_activation_checkpointing=True,
                use_rope=False,
            ),
            device="cuda",
        )
        if os.path.exists(
            f"{config.local_path}/mido-jepa-wm-rope/{config.decoder_path}"
        ):
            decoder_checkpoint_path = (
                f"{config.local_path}/mido-jepa-wm-rope/{config.decoder_path}"
            )
        else:
            from huggingface_hub import hf_hub_download

            decoder_checkpoint_path = hf_hub_download(
                repo_id="facebook/robotics-world-models",
                filename=f"mido-jepa-wm-rope/{config.decoder_path}",
            )
        dec_state = torch.load(decoder_checkpoint_path, map_location="cpu")
        msg = wm_image_head.model.load_state_dict(
            {k.replace("module.", ""): v for k, v in dec_state["model"].items()}
        )
        print(f"loaded pretrained target decoder with msg: {msg}")
        self.decoder = wm_image_head  # WorldModelViTImageHead
        encoder, predictor = init_video_model(
            uniform_power=config.uniform_power,
            device=device,
            patch_size=config.patch_size,
            max_num_frames=512,
            tubelet_size=config.tubulet_size,
            model_name=config.model_name,
            crop_size=config.crop_size,
            pred_depth=config.pred_depth,
            pred_num_heads=config.pred_num_heads,
            pred_embed_dim=config.pred_embed_dim,
            conditional_layer_norm=config.conditional_layer_norm,
            action_embed_dim=7,
            pred_is_frame_causal=config.pred_is_frame_causal,
            use_extrinsics=config.use_extrinsics,
            use_sdpa=True,
            use_silu=config.use_silu,
            use_pred_silu=config.use_pred_silu,
            wide_silu=config.wide_silu,
            use_rope=config.use_rope,
            use_activation_checkpointing=config.use_activation_checkpointing,
        )
        (_, predictor, encoder, _, _, _,) = load_checkpoint(
            r_path=checkpoint_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=encoder,
            replace_kw=["backbone.", "module."],
        )
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=256,
        )
        self.wm = JEPAWorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=self.tokens_per_frame,
            transform=transform,
            normalize_reps=config.normalize_reps,
            mpc_args=config.mpc,
            device=device,
        )

    def to(self, *args, **kwargs):
        self.wm.to(*args, **kwargs)
        self.decoder.model = self.decoder.model.to(*args, **kwargs)
        self.device = self.wm.device
        return self

    def video_encode(self, video):
        # video: (T, H, W, C) [0, 1]
        t, h, w, c = video.shape
        video = (video * 255).to(torch.uint8)  # (T, H, W, C) [0, 255]
        clip = self.wm.transform(video)[None, :]  # (1, C, T, H, W) [-1, 1]
        B, C, T, H, W = clip.size()
        assert B == 1
        assert C == 3
        assert T == t
        # daniel: according to Mido, we separate each image, and duplicate it to a video tubelet, I'm guessing to make it comparable to image goal
        clip = (
            clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        )  # (B*T, C, 2, H, W)
        clip = clip.to(self.device, non_blocking=True)
        h = self.wm.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # (B, T*TOK, D)
        if self.wm.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        h = h.view(B, T, -1, h.size(-1))  # (B, T, HW, D)
        return h  # (1, T, HW, D)

    def __pred(self, reps, actions, poses):
        # x, a, p (0 to t) -> x(t+1), p(t+1)
        B, T, N_T, D = reps.size()
        B, T, _7 = actions.size()
        B, T, _7 = poses.size()
        reps = reps.flatten(1, 2)  # (B, T*TOK_PER_FRAME, D)
        next_rep = self.wm.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
        if self.wm.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        next_rep = next_rep.view(B, 1, N_T, D)
        next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])  # (B, 1, 7)
        return next_rep, next_pose

    def __step(self, action, state: JepaWMState):
        """one prediction step from a state context -> longer state"""
        # action: (B, 1, 7)
        # clip to maximum context size handled by model
        MAX_CTX = 1
        if MAX_CTX == 1:
            reps = state.frame_traj[:, -1:]  # (B, T, TOK, D)
            poses = state.pose_traj[:, -1:]  # (B, T, 7)
            actions = state.action_traj[:, :0]  # (B, T-1, 7)
        else:
            MAX_ACTX = MAX_CTX - 1
            reps = state.frame_traj[:, -MAX_CTX:]  # (B, T, TOK, D)
            poses = state.pose_traj[:, -MAX_CTX:]  # (B, T, 7)
            actions = state.action_traj[:, -MAX_ACTX:]  # (B, T-1, 7)
        B, _1, _7 = action.shape
        # send to gpu
        action = action.to(self.device)  # (B, 1, 7)
        reps = reps.to(self.device)
        actions = actions.to(self.device)
        poses = poses.to(self.device)
        actions = torch.cat([actions, action], dim=1)  # (B, T, 7)
        next_rep, next_pose = self.__pred(reps, actions, poses)
        state.frame_traj = torch.cat(
            [state.frame_traj, next_rep], dim=1
        )  # (B, T+1, TOK, D)
        state.pose_traj = torch.cat([state.pose_traj, next_pose], dim=1)  # (B, T+1, 7)
        state.action_traj = torch.cat([state.action_traj, action], dim=1)  # (B, T, 7)
        feature = next_rep  # (B, 1, TOK, D)
        return feature, state

    def reset(self):
        pass

    def _encode_goal(self, goal: ImageProprioGoal):
        video = goal.image_tensor  # (T, 3, H, W) [0, 1]
        T, C, H, W = video.shape
        if T > 1:
            frame_skip = int(goal.freq // self.wm_frequency)
            video = video[::frame_skip]  # T, 3, H, W
        # if N is odd we need to add a dummy image to at least have one tubelet
        # if video.shape[0] % 2 == 1:
        #     video = np.concatenate([video, video[-1:]], axis=0)
        video_thwc = rearrange(video, "t c h w -> t h w c")  # (T, H, W, 3)
        goal_img_features = self.video_encode(video_thwc)  # (1, T, hw, D)
        return WMState(visual_state=goal_img_features, proprio_state=None, actions=None)

    @torch.inference_mode()
    def _encode_history(
        self, obs_history: RobotObsHistory, action_history: RobotActionHistory
    ):
        # subsample context
        frame_skip = int(obs_history.freq // self.wm_frequency)
        obs_context = obs_history.get_subsampled_history(
            max_context=self.max_context, skip=frame_skip
        )

        # Proprio
        proprio = obs_context.end_effector_pose_tensor  # (CTX, 7)
        proprio = proprio.unsqueeze(0)  # (1, CTX, 7)
        proprio = proprio.float()

        # Preprocess images
        video = obs_context.image_tensor  # N_images, 3, H, W
        # if N is odd we need to add a dummy image to at least have one tubelet
        # if video.shape[0] % 2 == 1:
        # video = np.concatenate([video, video[-1:]], axis=0)
        # proprio = torch.cat([proprio, proprio[:, -1:]], dim=1) # we need to pad this too
        video_thwc = rearrange(video, "t c h w -> t h w c")  # (T, H, W, 3)
        video_features = self.video_encode(video_thwc)  # (1, T, hw, D)

        # Now get actions
        if video_features.shape[1] - 1 == 0:
            actions_tensor = torch.zeros(1, 0, 7)
        else:
            action_context = action_history.chunk(
                max_context=video_features.shape[1] - 1,
                chunk_size=frame_skip,
                aggregation="sum",
            )
            actions_tensor = action_context.actions_tensor

        _1, tp, _7 = proprio.shape
        _1, tf, tok, d = video_features.shape
        assert (
            tf == tp
        ), f"video features and proprio should have the same time steps but shapes are f:{video_features.shape} and p:{proprio.shape}"
        _1, ta, _7 = actions_tensor.shape
        if ta == tp:
            actions_tensor = actions_tensor[:, :-1, :]
        assert (
            ta == tp - 1
        ), f"actions should be one less than proprio but are a:{actions_tensor.shape} and p:{proprio.shape}"
        actions_tensor = actions_tensor.float()

        # The 1 is camera view
        return WMState(
            visual_state=video_features,
            proprio_state=proprio,
            actions=actions_tensor,
        )

    @torch.inference_mode()
    def _rollout(
        self,
        context: WMState,
        action_sequence_to_rollout: RobotActionHistory,
        store_vid=False,
    ):
        # Chunk the action sequence
        action_sequence = action_sequence_to_rollout.chunk(
            max_context=self.max_rollout_length,
            chunk_size=int(action_sequence_to_rollout.freq // self.wm_frequency),
            aggregation="sum",
            from_last=False,
        )
        # Repeat context for each sample
        n_samples = action_sequence.actions_tensor.shape[0]
        context_actions = context.actions  # (1, CTX-1, 7)
        context_actions = context_actions.repeat(n_samples, 1, 1)
        context_features = context.visual_state  # (1, CTX, HW, D)
        context_features = context_features.repeat(n_samples, 1, 1, 1).float()
        context_proprio = context.proprio_state  # (1, CTX, 7)
        context_proprio = context_proprio.repeat(n_samples, 1, 1).float()

        # Do the rollout
        s, t_1, _7 = context_actions.shape
        s, t, tok, d = context_features.shape
        s, t, _7 = context_proprio.shape
        state = JepaWMState(
            frame_traj=context_features,
            action_traj=context_actions,
            pose_traj=context_proprio,
        )
        state = state.to(self.device)
        s, ftr, _7 = action_sequence.actions_tensor.shape
        # features = []
        # with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        if True:
            for robot_action in action_sequence.actions:
                action = torch.tensor(robot_action.action).float().unsqueeze(1).cuda()
                feature, state = self.__step(action, state)
                b, _1, tok, d = feature.shape
                # features.append(feature)

        # Return WMLatentRollout
        # features = torch.cat(features, dim=1) # (B, T, TOK, D)
        return WMState(
            visual_state=state.frame_traj,
            proprio_state=state.pose_traj,
            actions=state.action_traj,
        )  # only future state

    @torch.inference_mode()
    def _decode_latents(self, latents: WMState) -> RobotObsHistory:
        # Can't find a decoder on huggingface
        images = []
        B, T, TOK, D = latents.visual_state.shape
        assert B == 1, "Assuming batch size of 1"
        z_visual = latents.visual_state[0]

        # with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        if True:
            for z in z_visual:
                image = self.decoder.decode(
                    z.reshape(1, 1, 1, 16, 16, 1408)
                )  # (b, t, v, h, w, c) [0, 255]
                image_CHW_01 = (
                    rearrange(image, "1 1 1 h w c -> c h w").float() / 255.0
                )  # (3, 256, 256) [0, 1]
                images.append(image_CHW_01.cpu().detach().numpy())
        trajectory = RobotObsHistory(
            1000,
            int(self.wm_frequency),
            [RobotObs(np.zeros(7), np.zeros(7), img) for img in images],
        )
        return trajectory

    def input_frames_per_prediction(self, input_freq):
        assert input_freq == 30, "Input frequency must be 30 fps, is {}".format(
            input_freq
        )
        return 5

    @property
    def action_chunk_size(self):
        return 5  # TODO: action_freq // self.wm_frequency

    def _one_step_latent_pred(
        self, encoded_context: WMState, action_chunk: torch.Tensor
    ):
        B, _1, a = action_chunk.shape  # a is the chunk dimension
        A = 7
        assert (
            a == A * self.action_chunk_size
        ), f"Expected shape of action_chunk to be (B, 1, {A*self.action_chunk_size}), but got {action_chunk.shape}"
        action_unchunked = action_chunk.reshape(B, self.action_chunk_size, A)
        action_chunk_as_history = RobotActionHistory.from_tensor(
            max_context=self.action_chunk_size, freq=30, tensor=action_unchunked
        )

        new_context = self.rollout(encoded_context, action_chunk_as_history)
        z_new_visual = new_context.visual_state[:, -1:, :, :]  # (b, 1, p, d)

        return new_context, z_new_visual


@hydra.main(config_path="config", config_name="mido_jepa_wm.yaml")
def main(cfg):
    wm = hydra.utils.instantiate(cfg)
    print(wm)


if __name__ == "__main__":
    main()
