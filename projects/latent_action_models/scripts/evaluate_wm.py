"""Evaluate world model image prediction quality.

Metrics: PSNR, LPIPS, FID, FVD.
Usage:
    python scripts/evaluate_wm.py --snapshot_dir <dir> --gpu <gpu_id>
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        "/svl/u/ravenh/lacwm/robot_world_models-raven-lam/projects/latent_action_models"
    )
)
os.environ["COSMOS_HOME"] = "/svl/u/ravenh/lacwm/Cosmos"

from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: float(x) * float(y), replace=True)
OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b), replace=True)

import lpips

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── I3D for FVD ────────────────────────────────────────────────────────────
# Lightweight FVD using a torchvision inception v3 on per-frame features,
# then temporal averaging. This avoids needing an external I3D checkpoint.
from torchvision.models import inception_v3


def load_inception(device):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # output 2048-D features
    model = model.to(device).eval()
    return model


@torch.no_grad()
def get_inception_features(frames, inception_model, device, batch_size=64):
    """Extract inception features from frames [N, C, H, W] in [0,1]."""
    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size].to(device)
        # inception expects 299x299
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        # inception normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        feats = inception_model(batch)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()


def compute_fid(real_feats, gen_feats):
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def compute_fvd(real_video_feats, gen_video_feats):
    """FVD: same as FID but over temporally-averaged video features.
    real_video_feats: list of arrays, each [T, 2048] -> averaged to [2048]
    """
    real_avg = np.stack([f.mean(0) for f in real_video_feats])
    gen_avg = np.stack([f.mean(0) for f in gen_video_feats])
    return compute_fid(real_avg, gen_avg)


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_psnr(img1, img2):
    """Compute PSNR between two images in [0,1], shape [C,H,W]."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--snapshot_path", type=str, default=None,
                        help="Override snapshot file path (default: <snapshot_dir>/snapshot.pt)")
    parser.add_argument("--dataset_key", type=str, default="val_dataset",
                        help="Config key for dataset (e.g. val_dataset, libero_unseen_val_dataset)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # ── Load model ──────────────────────────────────────────────────────
    config_path = os.path.join(args.snapshot_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(config_path)

    logger.info(f"Loading model from {args.snapshot_dir}")
    model = instantiate(cfg.model)
    model = model.to(device).eval()
    snapshot_file = args.snapshot_path or os.path.join(args.snapshot_dir, "snapshot.pt")
    logger.info(f"Loading snapshot from {snapshot_file}")
    snapshot = torch.load(
        snapshot_file,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(snapshot["model"])

    # ── Load val dataset ────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset_key}")
    val_dataset = instantiate(cfg[args.dataset_key])

    # ── Load LPIPS and Inception ────────────────────────────────────────
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    inception_model = load_inception(device)

    # ── Sample indices ──────────────────────────────────────────────────
    rng = np.random.RandomState(42)
    dataset_len = len(val_dataset)
    num_samples = min(args.num_samples, dataset_len)
    indices = rng.choice(dataset_len, size=num_samples, replace=False)

    # ── Run evaluation ──────────────────────────────────────────────────
    all_psnr = []
    all_lpips = []
    all_real_frames = []
    all_gen_frames = []
    real_video_feats = []
    gen_video_feats = []

    logger.info(f"Evaluating {num_samples} samples on GPU {args.gpu}")

    for batch_start in range(0, num_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, num_samples)
        batch_indices = indices[batch_start:batch_end]
        cur_bs = len(batch_indices)

        # Collect batch
        samples = [val_dataset[int(idx)] for idx in batch_indices]

        rgb = torch.stack([s["rgb"] for s in samples]).to(device)  # [B, T, C, H, W]
        actions = torch.stack([s["actions"] for s in samples]).to(device)
        morphology_index = torch.stack([s["morphology_index"] for s in samples]).to(device)
        ee_action_dim = torch.stack([s["ee_action_dim"] for s in samples]).to(device)

        # Generate predictions
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            rgbhat = model._generate_future(
                rgb,
                actions,
                morphology_index=morphology_index,
                ee_action_dim=ee_action_dim,
            )  # [B, T, C, H, W]

        # Denormalize to [0,1] for metrics
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        rgb_denorm = (rgb * std + mean).clamp(0, 1)
        rgbhat_denorm = (rgbhat * std + mean).clamp(0, 1)

        # Only evaluate predicted frames (skip first context frame)
        gt_frames = rgb_denorm[:, 1:]  # [B, T-1, C, H, W]
        pred_frames = rgbhat_denorm[:, 1:]

        # Per-frame PSNR and LPIPS
        for b in range(cur_bs):
            sample_psnr = []
            sample_lpips = []
            for t in range(gt_frames.shape[1]):
                gt_f = gt_frames[b, t]  # [C, H, W]
                pred_f = pred_frames[b, t]
                sample_psnr.append(compute_psnr(pred_f, gt_f))
                # LPIPS expects [-1, 1]
                lp = lpips_fn(
                    pred_f.unsqueeze(0) * 2 - 1,
                    gt_f.unsqueeze(0) * 2 - 1,
                ).item()
                sample_lpips.append(lp)
            all_psnr.append(np.mean(sample_psnr))
            all_lpips.append(np.mean(sample_lpips))

        # Collect frames for FID (all predicted frames flattened)
        B, T_pred = gt_frames.shape[:2]
        all_real_frames.append(gt_frames.reshape(B * T_pred, *gt_frames.shape[2:]).cpu())
        all_gen_frames.append(pred_frames.reshape(B * T_pred, *pred_frames.shape[2:]).cpu())

        # Collect per-video features for FVD
        for b in range(cur_bs):
            real_clip = gt_frames[b].cpu()  # [T-1, C, H, W]
            gen_clip = pred_frames[b].cpu()
            real_feats = get_inception_features(real_clip, inception_model, device)
            gen_feats = get_inception_features(gen_clip, inception_model, device)
            real_video_feats.append(real_feats)
            gen_video_feats.append(gen_feats)

        logger.info(
            f"  [{batch_start + cur_bs}/{num_samples}] "
            f"PSNR={np.mean(all_psnr):.2f} LPIPS={np.mean(all_lpips):.4f}"
        )

    # ── Compute FID ─────────────────────────────────────────────────────
    logger.info("Computing FID...")
    all_real_frames = torch.cat(all_real_frames, dim=0)
    all_gen_frames = torch.cat(all_gen_frames, dim=0)

    real_inc_feats = get_inception_features(all_real_frames, inception_model, device)
    gen_inc_feats = get_inception_features(all_gen_frames, inception_model, device)
    fid = compute_fid(real_inc_feats, gen_inc_feats)

    # ── Compute FVD ─────────────────────────────────────────────────────
    logger.info("Computing FVD...")
    fvd = compute_fvd(real_video_feats, gen_video_feats)

    # ── Results ─────────────────────────────────────────────────────────
    results = {
        "model": args.snapshot_dir,
        "num_samples": num_samples,
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "lpips_mean": float(np.mean(all_lpips)),
        "lpips_std": float(np.std(all_lpips)),
        "fid": fid,
        "fvd": fvd,
    }

    logger.info(f"Results: {json.dumps(results, indent=2)}")

    # Save
    if args.output_dir is None:
        args.output_dir = args.snapshot_dir

    model_name = cfg.get("name", "model")
    snap_stem = os.path.splitext(os.path.basename(snapshot_file))[0]
    output_path = os.path.join(args.output_dir, f"eval_results_{model_name}_{args.dataset_key}_{snap_stem}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
