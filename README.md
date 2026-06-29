# lacwm-dit — Latent-Action World Model with a Wan2.1-Fun DiT forward model

A latent-action world model whose forward (world) model is the
**Wan2.1-Fun-1.3B-Control** video-diffusion transformer. The whole pipeline runs on the
**Wan VAE**: the world model is a conditional video-diffusion model that, given a few
history frames and a latent action, denoises the future frames.

The inverse model (image frames → latent action) and the per-morphology action decoder
are the standard latent-action-model components; only the forward model is diffusion-based.

## Architecture

```
rgb ──WanVAE(per-frame)──▶ per-frame latents ──inverse_model──▶ latent action z
history frames ──WanVAE(temporal)──▶ reference latent ─┐
latent action z ──ActionToControl──▶ control latent   ┼─ y = [control(16) | reference(16)]  (32ch)
future frames ──WanVAE(temporal)+noise──▶ noisy (16ch) ┘
        │
        ▼
   Wan DiT  (frozen 1.3B + LoRA)  x=noisy(16) ⊕ y(32) = in_dim 48  ──flow-matching──▶ future frames
        │
z ──action_decoder──▶ GT robot action            (L1 supervision, weight 0.1)

loss = masked_flow_matching_mse(future)  +  action_decoding_weight · action_decoding_l1
```

Two variants share the same skeleton:

| Variant | Model class | Conditioning latent |
|---|---|---|
| **Latent action** (default) | `LatentActionDiTModel` | inverse model derives `z` from image frames; `z` also supervised against GT actions |
| **Explicit action** | `ExplicitActionDiTModel` | an `ActionEncoderMLP` encodes the **GT robot actions** directly into `z`; no inverse model, no action-decoding loss |

### Key facts (verified against the Wan-Fun config + VideoX-Fun source)

- **Wan DiT** (`WanTransformer3DModel`): `dim 1536, ffn 8960, 12 heads, 30 layers,
  in_dim 48, out_dim 16`, patch `(1,2,2)`, `model_type i2v`. `in_dim 48` is the
  channel concat `[noisy(16) | control(16) | reference(16)]`. We reuse the pretrained
  patch-embed and only repurpose the *control* slot (latent action) and *reference*
  slot (history). LoRA on `q/k/v/o`; LoRA master weights kept in **fp32** (the trainer's
  `GradScaler` can't unscale bf16 grads).
- **Wan VAE** (`AutoencoderKLWan`): 16-ch latent, 8× spatial / 4× temporal causal.
  Runs in **fp32** (bf16 conv VAE is numerically unstable → NaN). Pixel H,W are padded
  to a multiple of 16; clips need `F ≡ 1 (mod 4)`.
- **Clip = 5 history + 8 future = 13 frames** (satisfies `4k+1`). 3 camera views are
  width-stacked per frame (180×960).
- **Latent actions**: exactly `num_future` (8) actions, one per future transition, from
  a 9-frame window `[last history frame | 8 future frames]`. A learnable MLP
  (`action_pool`) pools the per-latent-frame actions into the control channel.
- **Masked flow loss**: masked/missing camera views (constant pixels) and zero-padded
  frames (short trajectories) are kept as *inputs* but excluded from supervision.
- **Flow matching**: `noisy=(1-σ)x+σ·noise`, `target=noise-x`, logit-normal timesteps,
  `FlowMatchEulerDiscreteScheduler`. Text conditioning is a cached null umT5 embedding
  (no T5 in the train loop).
- **Attention backend**: short-sequence attention (e.g. the inverse model's temporal
  attention) uses the exact **MATH** SDPA backend — FlashAttention's fused backward
  emits NaN on degenerate (constant-across-time) keys. See
  `robot_wm/modeling/modules/attention.py`.

## Setup

Built and run on `ravenh@38.213.24.3` (8×B200), conda env `lacwm-dit`.

To provision a **new server** in one shot (env + weights + VideoX-Fun + datasets +
manifests + validation), use [`setup_training.sh`](setup_training.sh) — see **Data** below.
The manual steps here are what it automates.

1. **Environment** — Python 3.10 + PyTorch 2.7.1 (cu128). One-shot from the repo root:
   ```bash
   conda env create -f environment.yaml   # env "lacwm-dit": cu128 torch + requirements.txt + this package
   conda activate lacwm-dit
   ```
   or manually:
   ```bash
   conda create -n lacwm-dit python=3.10 && conda activate lacwm-dit
   pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt && pip install -e .
   ```
2. **VideoX-Fun** (provides `videox_fun.models.*`): cloned at `/scr/ravenh/VideoX-Fun`.
   Import the submodules directly (e.g. `from videox_fun.models.wan_vae import
   AutoencoderKLWan`) to avoid the librosa-heavy package `__init__`.
3. **Weights** — `alibaba-pai/Wan2.1-Fun-1.3B-Control` (VideoX-Fun single-file format)
   at `/scr/ravenh/wan_fun_1.3b_control/` (DiT `safetensors`, `Wan2.1_VAE.pth`, umT5/CLIP).
   The cached null-prompt embedding is `null_prompt_umt5.pt` in that dir.
4. **Data** — the 4 datasets under `/scr/ravenh/lacwm_data/` (`abc_pp`, `agibot`,
   `droid_lerobot`, `egodex_cdn`). The active run uses 35,671 episodes (DROID 10k,
   EgoDex 10k, ABC 10k, Agibot 5,671). See **Data** below for sources + the setup script.

## Data

`setup_training.sh` provisions a fresh server end-to-end (env, weights, VideoX-Fun,
datasets, manifests, path repointing, validation):

```bash
# download from the public sources, with per-dataset toggles + caps:
./setup_training.sh
AGIBOT_LIMIT=2000 EGODEX_PARTS=part2 ABC_ENABLE=0 ./setup_training.sh

# just (re)generate manifests on data already in place:
./setup_training.sh manifests
```

Each dataset has `<DS>_ENABLE` (1/0) and `<DS>_LIMIT` (episodes kept in the manifest;
`all` = no cap), plus download-volume knobs (`AGIBOT_TASKS`, `EGODEX_PARTS`; DROID volume
follows `DROID_LIMIT` → `ceil(limit/1000)` chunks).
Set `BASE` to relocate everything off `/scr/ravenh` and the script repoints the configs.

### Sources & expected layout (under `$DATA_ROOT`, default `/scr/ravenh/lacwm_data`)

| Dataset | Source | On-disk layout (what the loader reads) | Notes |
|---|---|---|---|
| DROID | [cadene/droid](https://huggingface.co/datasets/cadene/droid) | `droid_lerobot/{data,meta,videos}`, `data/chunk-*/episode_*.parquet` | LeRobot **v2.1**, loads directly. (`lerobot/droid_1.0.1` is **v3.0** file-packed parquet and is *not* loadable as-is.) `DROID_LIMIT` → `ceil(limit/1000)` chunks |
| Agibot | [agibot-world/AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) | `agibot/{observations,proprio_stats,parameters,task_info}/<task>/<ep>` | layout matches directly; loader uses `*_aligned.json` camera params (may need an alignment prep step) |
| EgoDex | [apple/ml-egodex](https://github.com/apple/ml-egodex) (zips on Apple's CDN) | `egodex_cdn/<part>/<task>/<n>.hdf5` | direct, no preprocessing; the active run used `part2.zip` |
| ABC | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) | `abc_pp/<task>/episode_*/{top,left_wrist,right_wrist}.mp4 + states.npz` | HF is **raw** — preprocess with `robot_wm/datasets/abc/preprocessing/abc_batch_preprocess.py` |

### Manifests

Loaders read manifests of **absolute** paths, so they are regenerated per server by
`./setup_training.sh manifests` (never copy egodex/abc manifests between machines):
- `egodex_cdn/manifest.csv` — one `*.hdf5` path per line (enumerated from the data)
- `abc_pp/manifest.txt` — one `episode_*` dir per line (enumerated)
- `agibot/manifest.csv` — `task_id,episode_id,dataset` (path-portable; an existing curated
  manifest, e.g. the active run's 5,671-episode subset, is preserved when present and
  uncapped, otherwise regenerated from the downloaded tasks)
- DROID needs no manifest — the loader globs `data/chunk-*/episode_*.parquet`

## Training

Launch scripts live in `/scr/ravenh/`. From `projects/latent_action_models/`:

```bash
# smoke test (tiny, single GPU)
/scr/ravenh/run_wan_smoke.sh

# full latent-action run (8 GPU, batch per-GPU = first arg)
/scr/ravenh/run_wan_full.sh 16
# == torchrun --standalone --nproc_per_node=8 train.py \
#      +experiments_0908=ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml \
#      data_loader.batch_size=16

# explicit action-conditioned variant
/scr/ravenh/run_wan_explicit_smoke.sh
# full: +experiments_0908=ravenhuang/wan-dit/wan_dit_explicit_abc_agibot_droid_egodex.yaml
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. If the NVLink fabric is flaky,
add `NCCL_P2P_DISABLE=1` (LoRA grads are small, so the PCIe fallback is cheap). To run on
fewer GPUs, e.g. with a dead device: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ...
--nproc_per_node=7`. Resume a crashed run with `--resume` (loads `snapshot.pt`).

Outputs (checkpoints, val losses, viz mp4s) land in `/scr/ravenh/lacwm_runs/<name>/<date>/`.
Metrics log to wandb project `lacwm`. Validation/viz run every 1000 iters; viz renders a
`[ground-truth | predicted]` future-frame strip via the flow-matching sampler.

## Layout (Wan-DiT additions)

```
projects/latent_action_models/
  lam/latent_action_dit_model.py        # main model (LatentActionDiTModel)
  lam/explicit_action_dit_model.py      # ExplicitActionDiTModel variant
  configs/models/{latent,explicit}_action_dit_model.yaml
  configs/experiments_0908/ravenhuang/wan-dit/*.yaml   # full + smoke experiments
robot_wm/modeling/
  tokenizers/rgb/wan_vae.py             # WanVAETokenizer (fp32 Wan VAE wrapper)
  networks/wan_forward_model.py         # WanForwardModel: frozen DiT + LoRA + ActionToControl
  networks/action_encoder.py            # ActionEncoderMLP (explicit variant)
  modules/attention.py                  # MATH SDPA for short sequences (NaN-in-backward fix)
```

The remaining infrastructure (datasets, transforms, inverse model, action decoder, trainer,
optimizer, LR schedule, DDP, wandb) is shared across the codebase. See the docstrings at the
top of each file above for the precise tensor shapes and conditioning layout.

## Config knobs worth knowing

- `model.forward_model`: `lora_rank` (64), `lora_alpha` (128), `gradient_checkpointing`.
- `model.config.action_decoding_weight` (0.1) and `flow_loss_on_future_only` (true).
- `model.num_history_frames` / `num_future_frames` (5 / 8) — must keep
  `num_future == (Fp-K)·temporal_ratio` so the learnable action pool aligns to latent frames.
- `dataset.img_augment` — on for train, off for val/viz; augmentation is applied in `[0,1]`
  before re-normalizing to `[-1,1]`.
- `data_loader.batch_size` — per-GPU; the full run uses 16 on B200.
