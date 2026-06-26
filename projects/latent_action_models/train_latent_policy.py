import argparse
import logging
import os
import sys
from dataclasses import dataclass
from math import prod
from typing import List, Optional, Union

os.environ["MANI_SKILL_ASSET_DIR"] = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

import time

import debugpy
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from robot_wm.modeling.modules.mlp import MLP

sys.path.append("projects/latent_action_models")
from custom_resolvers import *  # noqa: F401, F403

logger = logging.getLogger(__name__)


class RenderEnv:
    """
    Wrapper to render the environment and return the rendered observation.
    If we use "rgb" as the obs_mode in mani_skill we will get a different camera
    perspective for rendering and for obs
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def process_obs(self, obs):
        return obs
        """
        extracts the first one image of 3 stacked images
        """
        obs = obs.squeeze()
        H, W, C = obs.shape
        h = H // 3
        return obs[:h, :, :].clone()

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):
        obs, info = self.env.reset(*args, seed=seed, options=options, **kwargs)
        info["original_obs"] = obs.clone()  # Store the original observation
        obs = self.env.render()
        return self.process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_obs"] = obs.clone()  # Store the original observation
        obs = self.env.render()
        return self.process_obs(obs), reward, terminated, truncated, info


class Policy(nn.Module):
    def __init__(
        self,
        lam_model,
        latent_action_dim,
        input_dim,
        num_actions,
    ):
        super().__init__()
        self.lam_model = lam_model
        self.model = MLP(
            input_dim=input_dim,
            output_dim=512,
            hidden_dim=[1024, 512, 512],
        )
        self.out = nn.Linear(512, latent_action_dim)

    def _forward_tokenizer_encode(self, obs):
        if obs.dim() == 4:
            obs = obs.unsqueeze(0)
        x = self.lam_model(obs, mode="tokenize")
        x = rearrange(x, "N T d h w -> N T h w d")
        x = rearrange(x, "N T h w d -> N T (h w d)")
        return x

    def _generate_decoded_action(self, obs):
        z = self.forward(obs)
        return self.lam_model(z, mode="action_decode")

    def forward(self, obs, decode=False):
        z = self._forward_tokenizer_encode(obs)
        h = self.model(z)
        latent_action = self.out(h)
        if decode:
            return self.lam_model(latent_action, mode="action_decode")
        return latent_action


def eval_policy(policy, env, step, args, num_episodes=10, deterministic=True):
    policy.eval()

    success_rate = 0.0
    average_episode_reward = 0.0

    with torch.no_grad():
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                obs = rearrange(obs, "N H W C -> N C H W")
                obs = obs / 255.0
                action = policy(obs.unsqueeze(0), decode=True)
                action = action.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action[:7])
                done = terminated or truncated
                episode_reward += reward

            # logger.info(f"Episode {episode + 1} reward: {episode_reward}")
            average_episode_reward += episode_reward
            success_rate += info["success"]

    average_episode_reward /= num_episodes
    success_rate /= num_episodes

    print(f"Average episode reward: {average_episode_reward}")
    print(f"Success rate: {success_rate}")
    logger.info(f"Average episode reward: {average_episode_reward}")
    logger.info(f"Success rate: {success_rate}")

    if not args.no_wandb:
        wandb.log(
            {
                "average_episode_reward": average_episode_reward.item(),
                "success_rate": success_rate.item(),
            },
            step=step,
        )


@torch.no_grad()
def get_inverse_action(obs, model):
    # logger.info(f"obs shape: {obs.shape}")
    dist.barrier()
    z = model(obs, mode="tokenize")
    # logger.info(f"z shape: {z.shape}")
    with torch.no_grad():
        latent_actions = model(z, mode="inverse")
    return latent_actions


def train_latent_policy(
    model, env, rank, local_rank, world_size, cfg, data_iters, dataloaders, args
):
    output_dir = args.output_dir
    output_dir = os.path.join(
        output_dir,
        f"run_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}",
    )
    os.makedirs(output_dir, exist_ok=True)

    assert len(data_iters) > 0, "Validation iterator is empty"
    if rank == 0 and not args.no_wandb:
        wandb.init(
            project="latent_policy_training_ravenhuang",
            entity="robot_world_models",
            name=args.wandb_name,
            config=args,
        )

    device = torch.device(f"cuda:{local_rank}")
    num_actions = model.module.model.inverse_model.output_proj.query.num_embeddings

    # Ensure all processes are synchronized before getting z_dim
    dist.barrier()
    z_dim = get_z_dim(model, data_iters[0], local_rank)

    # Reset model to training mode after get_z_dim
    model.train()

    pi = Policy(
        lam_model=model,
        input_dim=z_dim,
        latent_action_dim=cfg.model.quantizer.dim,
        num_actions=num_actions,
    ).to(device)
    pi = DDP(pi, device_ids=[local_rank], find_unused_parameters=True)

    # Ensure all processes are synchronized after DDP initialization
    dist.barrier()

    optimizer = torch.optim.Adam(pi.parameters(), lr=1e-4)

    avg_loss = [0 for _ in range(len(data_iters))]
    pi.train()
    for step in tqdm(range(args.num_steps)):
        for i, (d_iter, data_loader) in enumerate(zip(data_iters, dataloaders)):
            batch = next(d_iter)
            optimizer.zero_grad()

            obs = batch["rgb"].to(device)
            # logger.info(f"obs shape: {obs.shape}")

            action_labels = get_inverse_action(obs, model).squeeze()
            # logger.info(f"action_labels shape: {action_labels.shape}")

            latent_actions = pi(obs)
            # logger.info(f"latent_actions shape: {latent_actions.shape}")

            loss = F.mse_loss(latent_actions[:, :-1, :], action_labels[:, 1:, :])

            loss.backward()
            optimizer.step()

            avg_loss[i] += loss.item()
            if step % args.log_every == 0 and rank == 0:
                if step > 0:
                    avg_loss[i] /= args.log_every
                logger.info(f"Average Loss: {avg_loss[i]:.4f}")
                if not args.no_wandb:
                    wandb.log(
                        {
                            f"{data_loader.dataset.name}/loss": loss.item(),
                            "avg_loss": avg_loss[i],
                        },
                        step=step,
                    )
                avg_loss[i] = 0

        if step % args.save_every == 0 and rank == 0:
            logger.info(f"Saving snapshot at step {step}")
            torch.save(
                {
                    "model": pi.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"{output_dir}/snapshot-{step}.pt",
            )
        if step % args.eval_every == 0 and rank == 0:
            logger.info(f"Evaluating at step {step}")
            eval_policy(
                pi, env, step, args, num_episodes=args.num_eval_eps, deterministic=True
            )


def get_z_dim(model, val_iter, local_rank):
    batch = next(val_iter)
    device = torch.device(f"cuda:{local_rank}")
    obs = batch["rgb"].to(device)

    # Debug: Check device placement
    logger.info(f"obs device: {obs.device}")
    logger.info(f"target device: {device}")

    # Ensure we're in eval mode and use no_grad for this operation
    model.eval()
    with torch.no_grad():
        z = model(obs, mode="tokenize")

    # Clean up GPU memory
    torch.cuda.empty_cache()

    return prod(z.shape[2:])  # Assuming z is of shape (N, T, H, W, D)


def get_model(cfg, snapshot_dir, local_rank, snapshot_name="snapshot.pt"):
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Loading model on device: {device}")

    model = instantiate(cfg.model)
    model = model.custom_to(device)

    state_dict = torch.load(f"{snapshot_dir}/{snapshot_name}", map_location=device)
    logger.info(state_dict["model"].keys())

    model.load_state_dict(state_dict["model"])

    # Explicitly move all model components to the target device after loading
    model = model.custom_to(device)

    # # Ensure all parameters are on the correct device
    # for name, param in model.named_parameters():
    #     if param.device != device:
    #         logger.warning(f"Parameter {name} is on device {param.device}, moving to {device}")
    #         param.data = param.data.to(device)

    # # Also check buffers
    # for name, buffer in model.named_buffers():
    #     if buffer.device != device:
    #         logger.warning(f"Buffer {name} is on device {buffer.device}, moving to {device}")
    #         buffer.data = buffer.data.to(device)

    logger.info(f"Model successfully loaded on device: {device}")

    # Wrap the model first
    model = WrappedModel(model)

    # Move the wrapped model to device to be sure
    model = model.to(device)

    # Initialize DDP
    model = DDP(model, device_ids=[local_rank]).eval()

    return model


class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, a=None, mode=None):
        # Ensure input tensor is on the same device as model
        if hasattr(self, "device"):
            x = x.to(self.device)

        if mode == "tokenize":
            return self.model._forward_tokenizer_encode(x)
        elif mode == "action_decode":
            morphology_index = torch.tensor([5], dtype=torch.long, device=x.device)
            return self.model.action_decoder(x, morphology_index[None])["5"]["actions"]
        elif mode == "forward":
            assert a is not None, "Action must be provided for forward mode"
            return self.model._forward_forward_model(x, a)
        elif mode == "inverse":
            return self.model._forward_inverse_model(x)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Supported modes are 'tokenize', 'action_decode', and 'forward'."
            )


@dataclass
class CLIArgs:
    snapshot_dir: str = "/home/ravenhuang/h2r/robot_world_models/projects/latent_action_models/data/experiments/32_aug_maniskill-finetune/2025-08-12/06-27-39/0/"
    debug: bool = False
    no_wandb: bool = False
    batch_size: int = 8
    num_eval_eps: int = 10
    num_steps: int = 50000
    save_every: int = 500
    eval_every: int = 50
    seed: int = 42
    log_every: int = 50
    style_seed: int = 0
    dataset_names: tuple[str] = ("ManiSkillDataset",)
    wandb_name: str = "train_latent_policy"
    env_name: str = "DROIDPickCube-v1"
    output_dir: str = "./data/latent_policy_outputs"

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env_name",
            type=str,
            default=cls.__dataclass_fields__["env_name"].default,
            help="Name of the environment to use",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=cls.__dataclass_fields__["seed"].default,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--dataset_names",
            type=list,
            default=cls.__dataclass_fields__["dataset_names"].default,
            help="List of dataset names to use",
        )
        parser.add_argument(
            "--snapshot_dir",
            type=str,
            default=cls.__dataclass_fields__["snapshot_dir"].default,
            help="Directory containing the snapshot",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=cls.__dataclass_fields__["batch_size"].default,
            help="Batch size for training",
        )
        parser.add_argument(
            "--num_eval_eps",
            type=int,
            default=cls.__dataclass_fields__["num_eval_eps"].default,
            help="Number of evaluation episodes",
        )
        parser.add_argument(
            "--num_steps",
            type=int,
            default=cls.__dataclass_fields__["num_steps"].default,
            help="Number of training steps",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=cls.__dataclass_fields__["debug"].default,
            help="Enable debug mode",
        )
        parser.add_argument(
            "--no_wandb",
            action="store_true",
            default=cls.__dataclass_fields__["no_wandb"].default,
            help="Enable/Disable Weights & Biases logging",
        )
        parser.add_argument(
            "--save_every",
            type=int,
            default=cls.__dataclass_fields__["save_every"].default,
            help="Save model every N steps",
        )
        parser.add_argument(
            "--eval_every",
            type=int,
            default=cls.__dataclass_fields__["eval_every"].default,
            help="Evaluate model every N steps",
        )
        parser.add_argument(
            "--log_every",
            type=int,
            default=cls.__dataclass_fields__["log_every"].default,
            help="Log metrics every N steps",
        )
        parser.add_argument(
            "--style_seed",
            type=int,
            default=cls.__dataclass_fields__["style_seed"].default,
            help="Style seed for the environment",
        )
        parser.add_argument(
            "--wandb_name",
            type=str,
            default=cls.__dataclass_fields__["wandb_name"].default,
            help="Name for the Weights & Biases run",
        )
        args = parser.parse_args()
        return cls(**vars(args))


def get_config(args: CLIArgs):
    config_path = f"{args.snapshot_dir}/.hydra/config.yaml"
    logger.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    return cfg


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"

    args = CLIArgs.from_cli()
    logger.info(f"Parsed CLI args: {args}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rank = int(os.environ["RANK"])
    if args.debug and rank == 1:
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

    cfg = get_config(args)
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    cfg.data_loader.batch_size = args.batch_size
    val_dataloader = instantiate(cfg.val_data_loader)
    # select dataloaders that have the same dataset name as the ones in args.dataset_names
    dataloaders = []
    for dl in val_dataloader:
        if "MultiDataset" in dl.dataset.name:
            for _, dataset in dl.dataset.datasets.items():
                if dataset.name in args.dataset_names:
                    dataloaders.append(dl)
        elif dl.dataset.name in args.dataset_names:
            dataloaders.append(dl)
    val_iters = [iter(dl) for dl in dataloaders]

    model = get_model(cfg, args.snapshot_dir, local_rank)
    env = gym.make(
        args.env_name,
        robot_uids="panda_robotiq",
        training_mode=False,
        style_seed=args.style_seed,
        num_envs=1,
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        render_mode="sensors",
    )
    env = RenderEnv(env)

    train_latent_policy(
        model, env, rank, local_rank, world_size, cfg, val_iters, dataloaders, args
    )

    if rank == 0 and not args.no_wandb:
        wandb.finish()
    env.close()
