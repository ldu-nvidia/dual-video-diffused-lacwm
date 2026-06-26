import os
import sys

import hydra
import numpy as np
import torch
from einops import rearrange

from robot_wm.inference.actor.base import RobotActionHistory, RobotObsHistory
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import (
    ImageProprioGoal,
    VisualDemonstrationGoal,
)
from robot_wm.inference.world_model.base import WMState, WorldModel
from robot_wm.utils.config import get_robot_wm_path

# from robot_wm.inference.evaluator.world_model import test_world_model_on_droid

BIG = 1000  # arbitrarily big context length

# Adding JEPA to the path
robot_wm_path = get_robot_wm_path()
jepa_path = os.path.join(robot_wm_path, "projects", "jepa-internal")
sys.path.append(jepa_path)
from app.droid_wm.world_model import WorldModel as JepaModel  # noqa: F401
from app.droid_wm.world_model import WorldModelState as JepaWMState  # noqa: F401


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
        self.wm = JepaModel.load_pretrained(
            config.weight_path,
            device="cuda",
        )

    def reset(self):
        pass

    def __preprocess_image(self, image):
        assert image.shape[0] == 3 and len(image.shape) == 3
        image = rearrange(image, "C H W -> H W C").cpu().numpy()
        # add a little noise for JEPA
        noise = np.random.normal(0, 0.005, image.shape).astype(np.float32)
        image2 = (np.clip(image + noise, 0, 1) * 255).astype(np.uint8)
        image = (image * 255).astype(np.uint8)
        image = self.wm.preprocess_frames(
            [image, image2], size=self.size
        )  # 2 images for tubelet
        return image

    def __preprocess_video(self, video, video_freq):
        T, C, H, W = video.shape
        assert C == 3
        frame_skip = int(video_freq // self.wm_frequency)
        video_s = video[::frame_skip]  # T, C, H, W
        # if T is odd we copy the last frame twice
        if video_s.shape[0] % 2 == 1:
            video_s = np.concatenate([video_s, video_s[-1:]], axis=0)
        video_hwc = np.moveaxis(video_s, 1, -1)  # T, H, W, C
        video_hwc_255 = (video_hwc * 255).astype(np.uint8)
        video_proc = self.wm.preprocess_frames(video_hwc_255, size=self.size)
        return video_proc

    def _encode_goal(self, goal):
        T, C, H, W = goal.image_tensor.shape
        if T == 1:
            image = self.__preprocess_image(goal.image_tensor[0])
        else:
            image = self.__preprocess_video(goal.image_tensor, goal.freq)
        goal_img_features = self.wm.encode_features(
            image,
            window=self.window,
            tubelet_size=self.tubulet_size,
            grid_size=self.grid_size,
        )
        goal_img_features = rearrange(goal_img_features, "t h w c -> 1 t (h w) c")
        return WMState(visual_state=goal_img_features, proprio_state=None, actions=None)

    def _encode_history(
        self, obs_history: RobotObsHistory, action_history: RobotActionHistory
    ):
        # Preprocess images
        frame_skip = int(obs_history.freq // self.wm_frequency)
        frame_skip = (
            frame_skip // 2
        )  # Daniel: This ensures latent history has same freq as rollout, but may be the wrong "model framerate"
        # TODO: decide if keep above or skip every other output latent in rollout
        obs_context = obs_history.get_subsampled_history(
            max_context=BIG, skip=frame_skip
        )
        imgs = obs_context.image_tensor  # N_images, 3, H, W
        imgs = (np.moveaxis(imgs.numpy(), 1, -1) * 255).astype(np.uint8)  # N, H, W, 3

        # if N is odd we need to add a dummy image to at least have one tubelet
        if imgs.shape[0] % 2 == 1:
            imgs = np.concatenate([imgs, imgs[-1:]], axis=0)
        imgs = self.wm.preprocess_frames(imgs, size=256)  # 3, N, size, size

        # Encode them in video features
        video_features = self.wm.encode_features(
            imgs,
            window=self.window,
            tubelet_size=self.tubulet_size,
            grid_size=self.grid_size,
        )  # N // tubelet_size, grid_size, grid_size, latent_size
        video_features = rearrange(video_features, "n h w c -> 1 n 1 h w c")

        # Now get actions
        if video_features.shape[1] - 1 == 0:
            actions_tensor = torch.zeros(1, 0, 7)
        else:
            action_context = action_history.chunk(
                max_context=video_features.shape[1] - 1,
                chunk_size=frame_skip * self.tubulet_size,
                aggregation="sum",
            )
            actions_tensor = action_context.actions_tensor

        # The 1 is camera view
        return WMState(
            visual_state=rearrange(video_features, "b t 1 h w z ->  b t (h w) z"),
            proprio_state=None,
            actions=actions_tensor,
        )

    def input_frames_per_prediction(self, input_freq):
        # 30 fps / 7.5 fps = 4 frames
        return int(input_freq / self.wm_frequency)  # 4

    @property
    def action_chunk_size(self):
        return 30.0 / 7.5  # 4

    @torch.inference_mode()
    def _rollout(
        self, context, action_sequence_to_rollout: RobotActionHistory, store_vid=False
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
        context_actions = context.actions
        context_actions = context_actions.repeat(n_samples, 1, 1)
        context_features = rearrange(
            context.visual_state, "b t (h w) z -> b t 1 h w z", h=16, w=16
        )
        context_features = context_features.repeat(n_samples, 1, 1, 1, 1, 1).float()

        # Do the rollout
        state = JepaWMState(features=context_features, actions=context_actions)
        features = [context_features]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            for robot_action in action_sequence.actions:
                action = torch.tensor(robot_action.action).unsqueeze(1).cuda()
                feature, state = self.wm.step(
                    action, state, use_cache=True, no_grad=True
                )
                features.append(
                    feature[:, -1:]
                )  # first loop feature contains context too

        # Return WMLatentRollout
        features = torch.cat(features, dim=1)
        features = rearrange(features[:, :, 0], "b t h w z -> b t (h w) z")
        return WMState(visual_state=features, proprio_state=None, actions=None)

    @torch.inference_mode()
    def _decode_latents(self, latents) -> RobotObsHistory:
        images = []
        assert latents.visual_state.shape[0] == 1, "Assuming batch size of 1"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            for latent in latents.visual_state[0]:
                latent = rearrange(latent, "(h w) z -> 1 1 1 h w z", h=16, w=16)
                image = self.wm.heads["image_head"].decode(latent)
                image_HWC_255 = image[0, -1, 0].cpu().numpy()  # (256, 256, 3) [0-255]
                image_CHW_01 = (
                    np.moveaxis(image_HWC_255, -1, 0) / 255.0
                )  # (3, 256, 256) [0-1]
                images.append(image_CHW_01)
        trajectory = RobotObsHistory(
            1000,
            int(self.wm_frequency),
            [RobotObs(np.zeros(7), np.zeros(7), img) for img in images],
        )
        return trajectory


@hydra.main(config_path="config", config_name="jepa_wm.yaml")
def main(cfg):
    wm = hydra.utils.instantiate(cfg)
    print(wm)


@hydra.main(config_path="config", config_name="jepa_wm.yaml")
def test_on_droid(cfg):
    wm_name = "jepa_wm"
    wm = JepaWM({})
    # TODO: Fix this function to use the new API
    test_world_model_on_droid(wm, wm_name)  # noqa: F821


if __name__ == "__main__":
    main()
    # test_on_droid()
