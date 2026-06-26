import copy

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from robot_wm.inference.actor.base import RobotActionHistory, RobotObsHistory
from robot_wm.inference.robot.base import RobotObs
from robot_wm.inference.task.reference_episode import ImageProprioGoal
from robot_wm.inference.world_model.base import WMState, WorldModel
from robot_wm.utils.notebook import _to_image


class STWM(WorldModel):
    def __init__(self, config):
        # Path checking kung-fu
        checkpoint_path = config.ckpt_path

        valid_extensions = ["snapshot.best", "snapshot.pt", "snapshot.best.pt"]
        if not any(checkpoint_path.endswith(ext) for ext in valid_extensions):
            raise AssertionError(
                f"Checkpoint path must end with one of: {valid_extensions}"
            )

        base_path = checkpoint_path
        for ext in valid_extensions:
            if checkpoint_path.endswith(ext):
                base_path = checkpoint_path[: -len(ext)]

        config_path = (
            base_path + ".hydra/config.yaml"
            if ".hydra" not in base_path
            else base_path + "config.yaml"
        )

        if config.ckpt_source == "huggingface":
            dl_checkpoint_path = hf_hub_download(repo_id="facebook/robotics-world-models", filename=checkpoint_path)
            dl_config_path = hf_hub_download(repo_id="facebook/robotics-world-models", filename=config_path)
            self.ckpt_path = dl_checkpoint_path
            self.cfg = OmegaConf.load(dl_config_path)
        elif config.ckpt_source == "local":
            self.ckpt_path = checkpoint_path
            self.cfg = OmegaConf.load(config_path)
        else:
            raise ValueError

        self.cfg.model.rgb_tokenizer.tokenizer_type = "both"
        model = hydra.utils.instantiate(self.cfg.model).to("cuda")
        model.eval()

        snapshot = torch.load(self.ckpt_path, map_location="cuda")
        msg = model.load_state_dict(snapshot["model"], strict=False)
        allowed_missing_keys = [
            "rgb_tokenizer.custom_decoder", # WM checkpoint doesn't contain the custom decoder weights, they were trained separately and are loaded in the tokenizer directly
            "rgb_tokenizer.model._dec_model", # Daniel: not 100% sure why this one is needed. Sergio?
        ]
        remaining_missing_keys = [k for k in msg.missing_keys if not any(k.startswith(prefix) for prefix in allowed_missing_keys)]
        assert len(remaining_missing_keys) == 0, f"Missing keys: {remaining_missing_keys}"

        self.model = model.to(torch.bfloat16)
        self.transform = hydra.utils.instantiate(self.cfg.dataset.transform)

    def preprocess_imgs(self, images):
        assert images.ndim == 4
        assert images.shape[1] == 3
        assert images.max() <= 1.0 and images.min() >= 0.0
        images = images.to("cuda") * 255
        images = rearrange(images, "t c h w -> t h w c")
        images = self.transform._rgb_transform(images)
        return images.unsqueeze(0) # (1, t, h, w, c)

    def reset(self):
        pass

    def _encode_goal(self, goal: ImageProprioGoal) -> WMState:
        img = goal.image_tensor
        img = self.preprocess_imgs(img)
        # img = img.to(torch.bfloat16)
        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            encoded_img = self.model._forward_rgb_tokenizer_encode(rgb=img)
        encoded_img = rearrange(encoded_img, "b t h w z -> b t (h w) z")
        return WMState(visual_state=encoded_img, proprio_state=None, actions=None)

    def encode_image(self, image: torch.Tensor) -> WMState:
        """Encodes a single image into the world model state."""
        img = self.preprocess_imgs(image)
        img = img.to(torch.bfloat16)
        return self.model._forward_rgb_tokenizer_encode(rgb=img)

    def decode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Decodes a single image from the world model state."""
        out_shape = list(self.inputs_shape)
        out_shape[1] = 1
        images = self.model._forward_rgb_tokenizer_decode(image, out_shape)
        out_list = [_to_image(image) for image in images[0]]
        return RobotObsHistory(
            1000,
            5,
            [
                RobotObs(np.zeros(7), np.zeros(7), np.transpose(obs, (2, 0, 1)) / 255.0)
                for obs in out_list
            ],
        )

    @torch.inference_mode()
    def _encode_history(
        self, obs_history: RobotObsHistory, action_history: RobotActionHistory
    ) -> WMState:

        obs_context = obs_history.get_subsampled_history(max_context=1000, skip=self.action_chunk_size)
        images = obs_context.image_tensor
        images = self.preprocess_imgs(images) # (1, t, h, w, c) [-1, 1] 

        if len(obs_context) > 1:
            # TODO: agg is sum if integrate_chunks else concat
            action_context = action_history.chunk(
                max_context=len(obs_context) - 1,
                chunk_size=self.action_chunk_size,
                aggregation="concat",
            )
            actions = action_context.actions_tensor.cuda()
        else:
            action_dim = self.cfg.model.action_tokenizer.input_dim
            actions = torch.zeros(1, 0, action_dim * self.action_chunk_size).cuda()
        actions = rearrange(actions, "b t (a d) -> b t a d", a=self.action_chunk_size)

        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            x_hat = self.model._forward_rgb_tokenizer_encode(
                rgb=images.to(torch.bfloat16)
            )
            a = self.model._forward_action_tokenizer_encode(
                actions=actions.to(torch.bfloat16)
            )

        self.h, self.w = x_hat.shape[2], x_hat.shape[3]
        self.inputs_shape = images.shape

        return WMState(
            visual_state=rearrange(x_hat, "b t h w z -> b t (h w) z"),
            proprio_state=None,
            actions=a,
        )

    def _rollout(
        self, context: WMState, actions_sequence_to_rollout: RobotActionHistory
    ) -> WMState:
        new_actions = actions_sequence_to_rollout.chunk(
            max_context=1000, chunk_size=self.action_chunk_size, aggregation="concat", from_last=False
        )
        actions = new_actions.actions_tensor.cuda()
        actions = rearrange(actions, "b t (a d) -> b t a d", a=self.action_chunk_size)
        inputs = {
            "context_obs": rearrange(
                context.visual_state, "b t (h w) z -> b t h w z", h=self.h, w=self.w
            ),
            "context_actions": context.actions,
            "future_actions": actions.to(torch.bfloat16),
        }

        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            x_hat, a = self.__generate_from_encoded(**inputs)

        return WMState(
            visual_state=rearrange(x_hat, "b t h w z -> b t (h w) z"),
            proprio_state=None,
            actions=a,
        )

    def input_frames_per_prediction(self, input_freq):
        assert input_freq == 30, "Input frequency must be 30 fps, is {}".format(input_freq)
        return self.action_chunk_size

    @property
    def action_chunk_size(self):
        return self.cfg.dataset.transform.chunk_size

    def _one_step_latent_pred(self, encoded_context : WMState, action_chunk : torch.Tensor):
        B, _1, a = action_chunk.shape # a is the chunk dimension
        A = 7
        assert a == A * self.action_chunk_size, f"Expected shape of action_chunk to be (B, 1, {A*self.action_chunk_size}), but got {action_chunk.shape}"
        action_unchunked = action_chunk.reshape(B, self.action_chunk_size, A)
        action_chunk_as_history = RobotActionHistory.from_tensor(
            max_context=self.action_chunk_size, freq=30, tensor=action_unchunked)

        new_context = self.rollout(encoded_context, action_chunk_as_history)
        z_new_visual = new_context.visual_state[:, -1:, :, :] # (b, 1, p, d)

        return new_context, z_new_visual

    @torch.inference_mode()
    def _decode_latents(self, state: WMState) -> RobotObsHistory:
        visual_latents = rearrange(
            state.visual_state, "b t (h w) z -> b t h w z", h=self.h, w=self.w
        )
        # We want to decode the latents to input shape
        out_shape = list(self.inputs_shape)
        # But we have a sequence of actions length
        if state.actions is not None:
            out_shape[1] = state.actions.shape[1] + 1
        else:
            out_shape[1] = 1
        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            images = self.model._forward_rgb_tokenizer_decode(visual_latents, out_shape) # (b, t, c, h, w) [-1, 1]
        obs_list = [_to_image(image, mean=self.transform._mean, std=self.transform._std) for image in images[0]] # (t, h, w, c) [0, 255]
        robot_obs_list = [
            RobotObs(
                np.zeros(7),
                np.zeros(7),
                np.transpose(obs, (2, 0, 1)) / 255.0,
            )
            for obs in obs_list
        ]
        return RobotObsHistory(1000, 5, robot_obs_list)

    @torch.no_grad()
    def __generate_from_encoded(
        self, context_obs: torch.Tensor, context_actions: torch.Tensor, future_actions
    ) -> torch.Tensor:
        """Generates future predictions from encoded context of RGB, actions and future actions to take.
        Args:
            x: torch.Tensor [N, T1, h, w, d]
            a: torch.Tensor [N, T2, *, A]
            actions: torch.Tensor [N, T2, *, A]
        Returns:
            torch.Tensor [N, T2, h, w, d]
        """
        new_a = self.model._forward_action_tokenizer_encode(actions=future_actions)
        a = context_actions.repeat(new_a.shape[0], 1, 1, 1)
        x = context_obs.repeat(new_a.shape[0], 1, 1, 1, 1)
        a = torch.cat([a, new_a], dim=1)
        (n, t1, h, w, d), t2 = x.shape, a.shape[1]
        xhat = torch.empty((n, t2 + 1, h, w, d), device=x.device, dtype=x.dtype)
        xhat[:, :t1] = x.detach().clone()
        
        max_context_size = self.cfg.dataset.transform.sample_size - 1
        for idx in range(t1, t2 + 1):
            xhat[:, idx] = self.model._forward_forward_model(
                xhat[:, max(0, idx - max_context_size) : idx], a[:, max(0, idx - max_context_size) : idx]
            )[:, -1]
        return xhat, a


@hydra.main(config_path="config", config_name="st_wm.yaml")
def main(cfg):
    wm = hydra.utils.instantiate(cfg)
    print(wm)


if __name__ == "__main__":
    main()
    # class DummyConfig:
        # ckpt_path = "/home/dud/Code/robot_world_models/data/experiments/JEPA2 48N/2025-04-24/20-27-01/0/snapshot.best"
    # STWM(DummyConfig)
