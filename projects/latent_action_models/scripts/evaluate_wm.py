"""Evaluate Wan-DiT world-model future-frame prediction quality.

Metrics: PSNR, LPIPS, FID, FVD (frame-level inception features; FVD over
temporally-averaged per-clip features, so no external I3D checkpoint is needed).

Usage:
    python scripts/evaluate_wm.py --snapshot_dir <run_dir> --gpu <gpu_id>

<run_dir> is a training run directory containing .hydra/config.yaml and snapshot.pt.
Works for both the latent-action and explicit-action DiT variants.
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

# make the `lam` package importable (it lives in the project root, one level up)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: float(x) * float(y), replace=True)
OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b), replace=True)

import lpips
from torchvision.models import Inception_V3_Weights, inception_v3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Inception features (FID / FVD) ──────────────────────────────────────────

def load_inception(device):
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = torch.nn.Identity()  # output 2048-D features
    return model.to(device).eval()


@torch.no_grad()
def get_inception_features(frames, inception_model, device, batch_size=64):
    """Extract inception features from frames [N, C, H, W] in [0, 1] -> [N, 2048]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        all_feats.append(inception_model(batch).cpu())
    return torch.cat(all_feats, dim=0).numpy()


def compute_fid(real_feats, gen_feats):
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


def compute_fvd(real_video_feats, gen_video_feats):
    """FVD: FID over temporally-averaged per-clip features.
    Each element of *_video_feats is a [T, 2048] array, averaged to [2048]."""
    real_avg = np.stack([f.mean(0) for f in real_video_feats])
    gen_avg = np.stack([f.mean(0) for f in gen_video_feats])
    return compute_fid(real_avg, gen_avg)


def compute_psnr(img1, img2):
    """PSNR between two images in [0, 1]."""
    mse = F.mse_loss(img1, img2).item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


# ─── Main ────────────────────────────────────────────────────────────────────

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
                        help="Config key for the eval dataset (e.g. val_dataset)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # ── Load model from the run's hydra config + snapshot ───────────────
    cfg = OmegaConf.load(os.path.join(args.snapshot_dir, ".hydra", "config.yaml"))
    logger.info(f"Loading model from {args.snapshot_dir}")
    model = instantiate(cfg.model).to(device).eval()

    snapshot_file = args.snapshot_path or os.path.join(args.snapshot_dir, "snapshot.pt")
    logger.info(f"Loading snapshot from {snapshot_file}")
    state = torch.load(snapshot_file, map_location=device, weights_only=True)["model"]
    # tolerate a DDP "module." prefix
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)

    # ── Dataset + metric nets ───────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset_key}")
    val_dataset = instantiate(cfg[args.dataset_key])
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    inception_model = load_inception(device)

    rng = np.random.RandomState(42)
    num_samples = min(args.num_samples, len(val_dataset))
    indices = rng.choice(len(val_dataset), size=num_samples, replace=False)
    n_future = model.num_future_frames

    all_psnr, all_lpips = [], []
    all_real_frames, all_gen_frames = [], []
    real_video_feats, gen_video_feats = [], []

    logger.info(f"Evaluating {num_samples} samples on GPU {args.gpu}")
    for batch_start in range(0, num_samples, args.batch_size):
        batch_indices = indices[batch_start : batch_start + args.batch_size]
        samples = [val_dataset[int(idx)] for idx in batch_indices]
        cur_bs = len(samples)

        rgb = torch.stack([s["rgb"] for s in samples]).to(device)              # [B, T, C, H, W]
        actions = torch.stack([s["actions"] for s in samples]).to(device)
        morphology_index = torch.stack([s["morphology_index"] for s in samples]).to(device)

        # Sample the future. _sample_future decodes through the Wan VAE and returns
        # predicted + ground-truth pixels as [B, C, F, H, W] in [-1, 1].
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_pix, gt_pix = model._sample_future(rgb, actions, morphology_index=morphology_index)

        # keep the last n_future frames, map [-1,1] -> [0,1], reorder to [B, F, C, H, W]
        nf = min(n_future, pred_pix.shape[2])
        to01 = lambda x: ((x[:, :, -nf:].float() + 1) / 2).clamp(0, 1).permute(0, 2, 1, 3, 4)
        gt_frames, pred_frames = to01(gt_pix), to01(pred_pix)

        # Per-frame PSNR / LPIPS
        for b in range(cur_bs):
            psnr_b, lpips_b = [], []
            for t in range(nf):
                gt_f, pred_f = gt_frames[b, t], pred_frames[b, t]            # [C, H, W]
                psnr_b.append(compute_psnr(pred_f, gt_f))
                lpips_b.append(lpips_fn(pred_f.unsqueeze(0) * 2 - 1,         # LPIPS expects [-1,1]
                                        gt_f.unsqueeze(0) * 2 - 1).item())
            all_psnr.append(np.mean(psnr_b))
            all_lpips.append(np.mean(lpips_b))

        # Collect frames (FID) and per-clip inception features (FVD)
        all_real_frames.append(gt_frames.reshape(cur_bs * nf, *gt_frames.shape[2:]).cpu())
        all_gen_frames.append(pred_frames.reshape(cur_bs * nf, *pred_frames.shape[2:]).cpu())
        for b in range(cur_bs):
            real_video_feats.append(get_inception_features(gt_frames[b].cpu(), inception_model, device))
            gen_video_feats.append(get_inception_features(pred_frames[b].cpu(), inception_model, device))

        logger.info(f"  [{batch_start + cur_bs}/{num_samples}] "
                    f"PSNR={np.mean(all_psnr):.2f} LPIPS={np.mean(all_lpips):.4f}")

    logger.info("Computing FID...")
    fid = compute_fid(
        get_inception_features(torch.cat(all_real_frames), inception_model, device),
        get_inception_features(torch.cat(all_gen_frames), inception_model, device),
    )
    logger.info("Computing FVD...")
    fvd = compute_fvd(real_video_feats, gen_video_feats)

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

    output_dir = args.output_dir or args.snapshot_dir
    model_name = cfg.get("name", "model")
    snap_stem = os.path.splitext(os.path.basename(snapshot_file))[0]
    output_path = os.path.join(output_dir, f"eval_results_{model_name}_{args.dataset_key}_{snap_stem}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
