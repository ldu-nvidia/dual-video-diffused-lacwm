import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.cuda.amp as amp
from einops import rearrange, repeat
from omegaconf import OmegaConf

from robot_wm.inference.actor.base import RobotActionHistory, RobotObsHistory
from robot_wm.inference.robot.base import RobotEEDeltaAction, RobotObs
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)
from robot_wm.inference.world_model.base import WMState, WorldModel
from robot_wm.utils.config import get_robot_wm_path

# Adding Dino-WM to the path
robot_wm_path = get_robot_wm_path()
dino_wm_path = os.path.join(robot_wm_path, "projects", "dino_wm")
sys.path.append(dino_wm_path)

from datasets.img_transforms import default_transform

# from plan import load_model

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path, weights_only=False)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    # if "image_size" is in model config, we are dealing with a legacy model. Replace image_size in config with image_height and image_width (both equal)
    if hasattr(train_cfg.model, "image_size"):
        train_cfg.model.image_height = train_cfg.model.image_size
        train_cfg.model.image_width = train_cfg.model.image_size
        train_cfg.model.pop("image_size")

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model

class DinoWM(WorldModel):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg_path = config.model_cfg_path
        self.model_ckpt_path = Path(config.model_ckpt_path)
        self.max_context = config.max_context
        self.max_rollout_length = config.max_rollout_length
        self.obs_frame_skip = config.obs_frame_skip
        self.frame_skip = config.frame_skip
        self.use_amp = config.use_amp
        self.counter = 0
        self.wm_frequency = 25 / self.frame_skip

        # Load model configuration
        with open(self.model_cfg_path, "r") as f:
            self.model_cfg = OmegaConf.load(f)

        self.num_action_repeat = self.model_cfg.num_action_repeat

        # Load world model and transforms
        print("Loading world model")
        self.wm = load_model(
            self.model_ckpt_path,
            self.model_cfg,
            self.num_action_repeat,
            device=self.device,
        )
        if config.override_decoder:
            decoder_ckpt_path = Path(config.decoder_ckpt_path)
            decoder_cfg_path = config.decoder_cfg_path
            with open(decoder_cfg_path, "r") as f:
                decoder_cfg = OmegaConf.load(f)
            decoder_model = load_model(
                decoder_ckpt_path,
                decoder_cfg,
                decoder_cfg.num_action_repeat,
                device=self.device,
            )
            self.wm.decoder = decoder_model.decoder
            print(f"Overridden Decoder, loaded from {decoder_ckpt_path}")
            self.has_decoder = True
        print("World model loaded")
        self.wm.eval()
        self.transform = default_transform((self.wm.image_height, self.wm.image_width))

    def reset(self):
        pass

    def __prepend_zero(self, actions):
        return torch.cat(
            [
                torch.zeros(
                    (actions.shape[0], 1, actions.shape[2]),
                    device=actions.device,
                ),
                actions,
            ],
            dim=1,
        )

    @torch.inference_mode()
    def _encode_goal(self, goal: ImageProprioGoal):
        visual = self.transform(goal.image_tensor)
        visual = (
            rearrange(visual, "t c h w -> 1 t c h w").cuda().float()
        )  # (1, 1, 3, H, W)
        proprio = (
            rearrange(goal.proprio_tensor, "t d -> 1 t d").cuda().float()
        )  # (1, 1, 8)
        # remove gripper (WM doesn't know about it
        proprio = proprio[:, :, :7]

        with amp.autocast(enabled=self.use_amp):
            encoded = self.wm.encode_obs({"visual": visual, "proprio": proprio})

        encoded_proprio = rearrange(encoded["proprio"], "1 t d -> 1 t d")
        encoded_visual = rearrange(encoded["visual"], "1 t p f -> 1 t p f")  # (patch, )

        return WMState(
            visual_state=encoded_visual, proprio_state=encoded_proprio, actions=None
        )

    @torch.inference_mode()
    def _encode_history(
        self, obs_history: RobotObsHistory, action_history: RobotActionHistory
    ):
        """
        Encode observation and action history into context for the world model.

        Input shapes:
        - obs_history: Contains image and joint position history
        - action_history: Contains robot action history
        """
        # Get subsampled observation history
        obs_context = obs_history.get_subsampled_history(
            max_context=self.max_context,
            skip=self.obs_frame_skip,  # this frame skip needs to have hertz of
        )
        # Preprocess images add batch dimension to observations
        visual = self.transform(obs_context.image_tensor)
        context_obs = {
            "visual": rearrange(visual, "t c h w -> 1 t c h w").cuda().float(),
            "proprio": rearrange(obs_context.joints_tensor, "t d -> 1 t d")
            .cuda()
            .float(),
        }

        # Process action history

        # TODO might need clean up

        if len(obs_context) > 1:
            # Process action history
            action_context = action_history.chunk(
                max_context=len(obs_context) - 1,
                chunk_size=self.frame_skip,
                aggregation="concat",
                fill_missing=RobotEEDeltaAction.ZERO(),
            )
            action_context = action_context.actions_tensor.cuda().float()
            action_context = self.__prepend_zero(action_context)
        else:

            action_context = torch.zeros(
                (1, 1, 35), device=self.device
            ).float()  # TODO change 35
        # Encode context

        # action_context = action_history.chunk(
        #     max_context=len(obs_context) - 1,
        #     chunk_size=self.frame_skip,
        #     aggregation="concat",
        #     fill_missing=RobotEEDeltaAction.ZERO(),
        # )
        # action_context = action_context.actions_tensor.cuda().float()
        # action_context = self.__prepend_zero(action_context)

        # Encode context
        with amp.autocast(enabled=self.use_amp):
            z = self.wm.encode(context_obs, action_context)
            z_obs, z_act = self.wm.separate_emb(z)

        return WMState(
            visual_state=z_obs["visual"], proprio_state=z_obs["proprio"], actions=z_act
        )

    def input_frames_per_prediction(self, input_freq):
        assert input_freq == 30, "Input frequency must be 30 fps, is {}".format(input_freq)
        return self.frame_skip

    @property
    def action_chunk_size(self):
        return self.frame_skip

    @torch.inference_mode()
    def _rollout(
        self,
        encoded_context,
        actions_sequence_to_rollout: RobotActionHistory,
        store_vid=False,
    ):
        z_visual = encoded_context.visual_state
        z_proprio = encoded_context.proprio_state
        z_act = encoded_context.actions
        z = self.wm.join_emb(z_visual, z_proprio, z_act)

        # chunk all action_sequences and concat them
        action_sequence = actions_sequence_to_rollout.chunk(
            max_context=self.max_rollout_length,
            chunk_size=self.frame_skip,
            aggregation="concat",
            from_last=False,
        )
        action_sequence = action_sequence.actions_tensor.cuda().float()

        # Repeat encoded context to match the batch size of the action sequence
        z = repeat(z, "1 ... -> n ...", n=action_sequence.shape[0])
        with amp.autocast(enabled=self.use_amp):
            rollout_zobs, rollout_z = self.__rollout_from_encoding(
                z_0=z,
                action=action_sequence,
            )

        return WMState(
            visual_state=rollout_zobs["visual"],
            proprio_state=rollout_zobs["proprio"],
            actions=None,
        )

    @torch.inference_mode()
    def _decode_latents(self, latents) -> RobotObsHistory:
        assert latents.visual_state.shape[0] == 1, "Batch size must be 1"
        BATCH = 0
        dict = {"visual": latents.visual_state, "proprio": latents.proprio_state}
        with amp.autocast(enabled=self.use_amp):
            decoded, _ = self.wm.decode_obs(dict)
        robot_obs_list = []
        for i in range(len(decoded["visual"][BATCH])):
            image_CHW = decoded["visual"][BATCH, i].cpu().numpy()
            image_CHW_01 = np.clip(image_CHW * 0.5 + 0.5, 0, 1).astype(np.float32)
            obs = RobotObs(
                np.zeros(7),
                np.zeros(
                    7
                ),  # decoded['proprio'][i].cpu().numpy(), # no proprio decoder yet
                image_CHW_01,
            )
            robot_obs_list.append(obs)
        trajectory = RobotObsHistory(1000, int(self.wm_frequency), robot_obs_list)
        return trajectory

    @torch.inference_mode()
    def decode_to_images(self, latents):
        dict = {"visual": latents.visual_latents, "proprio": latents.proprio_latents}
        with amp.autocast(enabled=self.use_amp):
            obs, diff = self.wm.decode_obs(dict)
        images = obs["visual"]
        B, T, C, H, W = images.shape
        return images  # (B, T, C, H, W)

    @torch.inference_mode()
    def __rollout_from_encoding(self, z_0, action):
        """
        Rollout from pre-encoded observations.

        Args:
            z_0: Pre-encoded states (b, num_init, num_patches, emb_dim)
            action: Actions for rollout (b, T, action_dim)

        Returns:
            z_obs, z: Predicted observation embeddings and full state
        """

        # Start with the provided encoded state
        z = z_0.clone()

        # Rollout with the actions
        t, inc = 0, 1

        while t < action.shape[1]:
            z_pred = self.wm.predict(z[:, -self.wm.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.wm.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        # Final prediction step
        z_pred = self.wm.predict(z[:, -self.wm.num_hist :])
        z_new = z_pred[:, -1:, ...]  # Take only the next prediction
        z = torch.cat([z, z_new], dim=1)
        z_obs, z_acts = self.wm.separate_emb(z)

        return z_obs, z

    @torch.inference_mode()
    def _one_step_latent_pred(self, encoded_context : WMState, action_chunk : torch.Tensor):
        B, _1, a = action_chunk.shape # a is the chunk dimension
        z_visual = encoded_context.visual_state # (B, t, p, d)
        z_proprio = encoded_context.proprio_state
        z_act = encoded_context.actions # (B, t-1, d)
        CTX = min(self.wm.num_hist, self.max_context, z_act.shape[1]+1, z_visual.shape[1])
        assert CTX > 0, f"Context length must be at least 1, got {CTX} from hist {self.wm.num_hist}, max {self.max_context}, z_act {z_act.shape[1]}, z_visual {z_visual.shape[1]}"
        ACTX = CTX - 1

        # trim context to max
        z_visual = z_visual[:, -CTX:, :, :]
        z_proprio = z_proprio[:, -CTX:, :]
        z_act = z_act[:, -ACTX:, :]

        # add new action chunk
        z_new_act = self.wm.encode_act(action_chunk) # (b, 1, d)
        z_act = torch.cat([z_act, z_new_act], dim=1) # (b, t, d)

        # predict
        z = self.wm.join_emb(z_visual, z_proprio, z_act)
        z_pred = self.wm.predict(z)

        # autoregressive state update
        z_pred_obs, _ = self.wm.separate_emb(z_pred)
        z_new_visual = z_pred_obs["visual"][:, -1:, ...] # (b, 1, p, d)
        z_new_proprio = z_pred_obs["proprio"][:, -1:, ...]

        new_context = WMState(
            visual_state=torch.cat([encoded_context.visual_state, z_new_visual], dim=1)[:, -self.max_context:, :, :],
            proprio_state=torch.cat([encoded_context.proprio_state, z_new_proprio], dim=1)[:, -self.max_context:, :],
            actions=torch.cat([encoded_context.actions, z_new_act], dim=1)[:, -self.max_context:, :],
        )

        return new_context, z_new_visual




@hydra.main(config_path="config", config_name="dino_wm.yaml")
def main(cfg):
    wm = hydra.utils.instantiate(cfg)
    print(wm)


@hydra.main(config_path="config", config_name="dino_wm.yaml")
def test_on_droid(cfg):
    wm = hydra.utils.instantiate(cfg)  # DinoWM(cfg)
    wm_name = "droid_wm"

    # TODO
    test_world_model_on_droid(wm, wm_name)  # noqa: F821


if __name__ == "__main__":
    main()
    # test_on_droid()