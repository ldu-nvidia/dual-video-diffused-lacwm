import dataclasses
import logging
import os
import sys
from typing import Dict

import numpy as np
import tyro
import torch
import websocket_policy_server
from openpi_client import base_policy as _base_policy

# Make sure we can import ActionSelectionWithWM when running the script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from scripts.action_selection_with_wm import ActionSelectionWithWM  # noqa: E402


class WorldModelPolicy(_base_policy.BasePolicy):
    """Wraps ActionSelectionWithWM with the BasePolicy interface for websocket serving."""

    def __init__(self, snapshot_dir: str, use_wrist_image: bool = False, multi_view: bool = False) -> None:
        self._wm = ActionSelectionWithWM(snapshot_dir, use_wrist_image=use_wrist_image, multi_view=multi_view)

    def infer(self, obs: Dict, noise=None) -> Dict:
        """
        Expects obs to contain:
        - action_chunks: array-like [N, T, action_dim]
        - agent_obs: uint8 agent camera image (H, W, 3)
        - wrist_obs: optional uint8 wrist camera image (H, W, 3)
        - goal_image: uint8 goal image (H, W, 3)
        """
        action_chunks = np.asarray(obs["action_chunks"])
        agent_obs = np.asarray(obs["agent_obs"])
        wrist_obs = np.asarray(obs.get("wrist_obs", agent_obs))
        goal_image = np.asarray(obs["goal_image"])
        prediction_steps = int(obs.get("prediction_steps") // self._wm.wm_action_chunk_size)

        results = self._wm.action_selection(action_chunks, agent_obs, wrist_obs, goal_image,prediction_steps)

        # Convert tensors to numpy for msgpack serialization.
        action_chunk = np.asarray(results["action_chunk"])
        mse_distance = results["mse_distance"].to(torch.float32).detach().cpu().numpy()

        return {
            "action": action_chunk,
            "index": int(results["index"]),
            "mse_distance": mse_distance,
            "all_predictions": results["all_predictions"].to(torch.float32).detach().cpu().numpy(),
        }


@dataclasses.dataclass
class Args:
    snapshot_dir: str = (
        "/svl/u/ravenh/lacwm/robot_world_models-raven-lam/projects/latent_action_models/"
        "data/experiments_0908/libero_sim_scratch_action_9/2025-11-24/19-41-14/"
    )
    # Bind to all interfaces by default so the websocket client can reach us.
    host: str = "0.0.0.0"
    port: int = 9100
    use_wrist_image: bool = False
    multi_view: bool = False


def main(args: Args) -> None:
    policy = WorldModelPolicy(
        snapshot_dir=args.snapshot_dir, use_wrist_image=args.use_wrist_image, multi_view=args.multi_view
    )
    metadata = {"model": "world_model", "snapshot_dir": args.snapshot_dir}
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info("Serving world model on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
