#!/usr/bin/env python3
"""CPU-safe smoke test for the dual-state tensor and schedule contracts."""

import argparse
import json
import sys
from pathlib import Path

import torch

# Support direct execution from a clean checkout without an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion import (
    DualClockSampler,
    PerViewCausalRFFT,
    TFVelocityHead,
    ZeroInitTFTokenAdapter,
    corrupt_dual_flow,
    make_paired_sigma_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    video = torch.randn(args.batch_size, 13, 3, args.height, args.width)
    transform = PerViewCausalRFFT(
        num_views=3, output_size=(args.height, args.width), window_size=4
    )
    tf_clean = transform(video)
    video_clean = torch.randn(args.batch_size, 16, 4, args.height, args.width)
    clocks = DualClockSampler(mode="tf_first_cascaded_noised")(
        args.batch_size, device=video.device
    )
    corruption = corrupt_dual_flow(video_clean, tf_clean, clocks)

    adapter = ZeroInitTFTokenAdapter(
        tf_channels=tf_clean.shape[1], hidden_size=32, patch_size=(1, 2, 2)
    )
    tokens, grid = adapter(corruption.time_frequency.noisy)
    head = TFVelocityHead(
        hidden_size=32, tf_channels=tf_clean.shape[1], patch_size=(1, 2, 2)
    )
    tf_velocity = head(tokens, grid)
    schedule = make_paired_sigma_schedule(8, mode="tf_first_cascaded")

    report = {
        "schema_version": 1,
        "status": "pass",
        "video_clean_shape": list(video_clean.shape),
        "tf_clean_shape": list(tf_clean.shape),
        "tf_token_shape": list(tokens.shape),
        "tf_velocity_shape": list(tf_velocity.shape),
        "adapter_initially_noop": bool(torch.count_nonzero(tokens) == 0),
        "head_initially_zero": bool(torch.count_nonzero(tf_velocity) == 0),
        "schedule_steps": schedule.num_steps,
        "schedule_video": schedule.video.tolist(),
        "schedule_time_frequency": schedule.time_frequency.tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
