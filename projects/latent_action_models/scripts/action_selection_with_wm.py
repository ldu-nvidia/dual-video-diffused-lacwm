import random

import imageio
import mediapy as media
import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Register custom resolver to handle multiplication in OmegaConf interpolation
OmegaConf.register_new_resolver("mul", lambda x, y: float(x) * float(y))
OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b))

import os
import sys
import time

sys.path.append(
    os.path.abspath(
        "/svl/u/ravenh/lacwm/robot_world_models-raven-lam/projects/latent_action_models"
    )
)
os.environ["COSMOS_HOME"] = "/svl/u/ravenh/lacwm/Cosmos"

import torchvision.transforms.v2 as transforms

_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)


def _make_rgb_transform(resize_to):
    return transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (T, C, H, W)
            (
                transforms.Resize(tuple(int(x) for x in resize_to))
                if resize_to is not None
                else transforms.Lambda(lambda x: x)
            ),  # resize
            transforms.Normalize(mean=_RGB_MEAN, std=_RGB_STD, inplace=True),
        ]
    )


def load_wm_model(snapshot_dir):
    # snapshot_dir = "/svl/u/ravenh/lacwm/robot_world_models-raven-lam/projects/latent_action_models/data/experiments_0908/libero_sim_scratch_action_9/2025-11-24/19-41-14/"
    config_path = f"{snapshot_dir}/.hydra"
    print(config_path)

    cfg = OmegaConf.load(config_path + "/config.yaml")

    model = instantiate(cfg.model)
    model = model.cuda().eval()
    snapshot = torch.load(
        f"{snapshot_dir}/snapshot.pt", map_location="cuda:0", weights_only=True
    )
    model.load_state_dict(snapshot["model"])

    sample_size = cfg.dataset.transform.sample_size
    chunk_size = cfg.dataset.transform.chunk_size
    # Read resize_to from the first dataset's transform config; fall back to (256, 256).
    resize_to = (256, 256)
    datasets_cfg = cfg.get("dataset", {}).get("datasets", {})
    for ds_cfg in datasets_cfg.values():
        ds_resize = ds_cfg.get("transform", {}).get("resize_to", None)
        if ds_resize is not None:
            resize_to = tuple(ds_resize)
            break
    return model, sample_size, chunk_size, resize_to


class ActionSelectionWithWM:
    def __init__(self, snapshot_dir, use_wrist_image=False, multi_view=False):
        self.model, sample_size, chunk_size, resize_to = load_wm_model(snapshot_dir)
        self.model.ae_only = True
        self.wm_history_size = sample_size
        self.wm_action_chunk_size = chunk_size
        self.morphology_index = 5
        self.ee_action_dim = 19
        self.use_wrist_image = use_wrist_image
        self.multi_view = multi_view
        self.rgb_transform = _make_rgb_transform(resize_to)

    def get_batch_obs(self, all_action_chunks, all_img_obs):
        all_actions = np.array(all_action_chunks)[
            :, : self.wm_history_size * self.wm_action_chunk_size
        ]
        batch_size = all_actions.shape[0]
        zero_camera_action = np.zeros(
            (batch_size, self.wm_history_size * self.wm_action_chunk_size, 9)
        )
        all_actions = np.concatenate([all_actions, zero_camera_action], axis=-1)
        action_history_reshaped = all_actions.reshape(
            batch_size,
            self.wm_history_size,
            self.wm_action_chunk_size,
            all_actions.shape[-1],
        )
        cur_obs = {
            "rgb": all_img_obs[None].repeat(batch_size, 1, 1, 1, 1),
            "actions": action_history_reshaped,
        }
        if cur_obs["rgb"].shape[1] != self.wm_history_size:
            cur_obs["rgb"] = cur_obs["rgb"][:, :1].repeat(
                1, self.wm_history_size, 1, 1, 1
            )

        kwargs = {
            "morphology_index": np.repeat(
                np.array([self.morphology_index]), batch_size, axis=0
            ),
            "ee_action_dim": np.repeat(
                np.array([self.ee_action_dim]), batch_size, axis=0
            ),
        }
        for k, v in cur_obs.items():
            if isinstance(v, np.ndarray):
                cur_obs[k] = torch.from_numpy(v).to(torch.float32).cuda()
            else:
                cur_obs[k] = v.cuda()
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                dtype = torch.long if k == "morphology_index" else torch.float32
                kwargs[k] = torch.from_numpy(v).to(dtype).cuda()
            else:
                kwargs[k] = v.cuda()
        return cur_obs, kwargs

    def wm_rollout(
        self,
        all_action_chunks,
        agent_obs,  # (T, H, W, C), uint8
        wrist_obs,
        batch_size=4,
    ):  # (T, H, W, C), uint8

        if agent_obs.ndim == 3:
            agent_obs = agent_obs[None]
        if wrist_obs.ndim == 3:
            wrist_obs = wrist_obs[None]

        agent_obs = self.rgb_transform(
            torch.from_numpy(agent_obs.copy()).float()
        )  # (T, H, W, C), float32
        wrist_obs = self.rgb_transform(
            torch.from_numpy(wrist_obs.copy()).float()
        )  # (T, H, W, C), float32

        if self.multi_view:
            all_img_obs = torch.cat([agent_obs, wrist_obs], dim=-1)
        elif self.use_wrist_image:
            all_img_obs = wrist_obs
        else:
            all_img_obs = agent_obs

        cur_obs, kwargs = self.get_batch_obs(all_action_chunks, all_img_obs)

        all_predictions = []
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for i in range(0, cur_obs["actions"].shape[0], batch_size):
                    batched_kwargs = {
                        k: v[i : i + batch_size] for k, v in kwargs.items()
                    }
                    start_time = time.time()
                    predictions = self.model._generate_future(
                        cur_obs["rgb"][i : i + batch_size],
                        cur_obs["actions"][i : i + batch_size],
                        **batched_kwargs,
                    )
                    print("inference time: ", time.time() - start_time)
                    all_predictions.append(predictions)
        all_predictions_raw = torch.cat(
            all_predictions, dim=0
        )  # B, T, C, H, W, float32

        all_predictions_latents = self.model._forward_tokenizer_encode(
            all_predictions_raw[:, -1:]
        )  # get the latents of the last prediction

        C = all_predictions_raw.shape[-3]
        device = all_predictions_raw.device
        mean = torch.tensor(self.model.rgb_tokenizer._input_mean, device=device)
        std = torch.tensor(self.model.rgb_tokenizer._input_std, device=device)
        all_predictions = torch.clamp(
            all_predictions_raw * std.view(C, 1, 1) + mean.view(C, 1, 1), 0, 1
        )

        return all_predictions_latents, all_predictions

    def action_selection(
        self,
        all_action_chunks,
        agent_obs,
        wrist_obs,
        gt_goal_image,
        prediction_steps=None,
    ):
        self.wm_history_size = (
            prediction_steps if prediction_steps is not None else self.wm_history_size
        )
        # gt_goal_image is a single image of shape (H, W, C)
        all_predictions_latents, all_predictions = self.wm_rollout(
            all_action_chunks, agent_obs, wrist_obs
        )

        gt_goal_image_input = self.rgb_transform(
            torch.from_numpy(gt_goal_image.copy()).float().cuda()[None]
        )[
            None
        ]  # 1, 1, C, H, W
        gt_goal_image_latents = self.model._forward_tokenizer_encode(
            gt_goal_image_input
        )

        mse_distance = torch.nn.functional.mse_loss(
            all_predictions_latents, gt_goal_image_latents, reduction="none"
        )
        mse_distance = mse_distance.mean(dim=(1, 2, 3, 4))

        index = mse_distance.argmin()

        results = {
            "action_chunk": all_action_chunks[index],
            "index": index,
            "mse_distance": mse_distance,
            "all_predictions": all_predictions,
        }

        return results
