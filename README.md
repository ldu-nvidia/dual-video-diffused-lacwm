# dual-video-diffused-lacwm

Research code for adding a jointly denoised, phase-preserving
time-frequency state to the production LACWM Wan video world model.  The goal
is to test whether an auxiliary motion representation can improve temporal
coherence, sample efficiency, and low-NFE generation without sacrificing
action conditioning.

This repository preserves the production LACWM history at commit
`f227b3b5108cd63c0fc853a08a26ca705606387c`.  The completed reference LoRA run
used 64 B200 GPUs and reached all 60,000 planned optimizer updates.  See
[`docs/UPSTREAM_LACWM.md`](docs/UPSTREAM_LACWM.md) for exact provenance.

## Current status

The broader deployable study is now summarized in
[`TWELVE_HOUR_DUAL_VIDEO_RESEARCH_REPORT.md`](docs/experiments/TWELVE_HOUR_DUAL_VIDEO_RESEARCH_REPORT.md)
and the living evidence map
[`DEPLOYABLE_DUAL_VIDEO_DIRECTIONS.md`](docs/experiments/DEPLOYABLE_DUAL_VIDEO_DIRECTIONS.md).
Oracle clean-future TF/V-JEPA features remain upper bounds, while the tested
autonomous semantic/frequency branches and training-only relation/spectrum
losses have not improved the low-NFE generator. The selected next hypothesis is
an inference-causal robot-only flow scaffold computed from planned actions,
robot geometry, and calibrated cameras; its staged gate is specified in
[`ACTION_DERIVED_FLOW_PROTOCOL.md`](docs/experiments/ACTION_DERIVED_FLOW_PROTOCOL.md).
After the learned dense-flow proxy failed its strict handoff threshold, the
selected analytic-flow, stochastic-residual, and few-call generator protocol is
frozen in
[`PHYSICS_ANCHORED_RESIDUAL_FLOW_PROTOCOL.md`](docs/experiments/PHYSICS_ANCHORED_RESIDUAL_FLOW_PROTOCOL.md).
The first train-only nominal D405 geometry diagnostic is also complete: aligned
YAM rendering beat a preregistered +4-clip-step pose control by 4.58 pixels in
Chamfer distance (95% paired interval 3.32--5.86) at 2.77 ms p95 render
latency. This is a calibration-feasibility result, not a generation gain; exact
evidence and limitations are in
[`ABC_D405_NOMINAL_GEOMETRY_PROBE.md`](docs/experiments/ABC_D405_NOMINAL_GEOMETRY_PROBE.md).
A separate leakage-controlled Stage-0 now shows that observed history plus
planned actions predicts the compact 14-D command-tracking residual: validation
standardized MSE improved 60.08% over history and 72.45% over shuffled actions,
with all registered lower bounds positive. This is a `GO` only for rendering
the predicted corrected trajectory—not for Wan integration; see
[`NOMINAL_TRACKING_RESIDUAL_STAGE0_RESULT.md`](docs/experiments/NOMINAL_TRACKING_RESIDUAL_STAGE0_RESULT.md).
The sealed dense proxy result and post-hoc horizon diagnostics are in
[`DENSE_TOP_FLOW_STAGE0.md`](docs/experiments/DENSE_TOP_FLOW_STAGE0.md).
A complementary training-only route that distills a privileged clean-feature
teacher into a feature-free on-policy student is frozen in
[`PRIVILEGED_ON_POLICY_VIDEO_DISTILLATION_PROTOCOL.md`](docs/experiments/PRIVILEGED_ON_POLICY_VIDEO_DISTILLATION_PROTOCOL.md).
Its first NFE-4 train-only eligibility gate stopped before student training:
clean V-JEPA was uniformly helpful at the pure-noise video update but slightly
harmful at the final low-noise update, leaving only 50% favorable units versus
the registered 60% requirement. The exact timestep split and immutable evidence
are in
[`PRIVILEGED_ON_POLICY_TEACHER_ELIGIBILITY.md`](docs/experiments/PRIVILEGED_ON_POLICY_TEACHER_ELIGIBILITY.md).
A preregistered replication on 64 disjoint train episodes then confirmed that
the clean-feature teacher is uniformly strong at the sole NFE-2 pure-noise
video update (89.95% velocity-MSE gain over off; 64/64 favorable). This passes
only the prerequisite for a feature-free `PFD-VIDEO` student screen; no student
or deployable gain exists yet. See
[`PRIVILEGED_TEACHER_HIGH_NOISE_NFE2_RESULT.md`](docs/experiments/PRIVILEGED_TEACHER_HIGH_NOISE_NFE2_RESULT.md).
The read-only confidence-gating audit and its fail-closed result are documented
in [`CONFIDENCE_GATED_AUXILIARY_AUDIT_RESULT.md`](docs/experiments/CONFIDENCE_GATED_AUXILIARY_AUDIT_RESULT.md).

The prefix-causal V-JEPA 2 semantic representation screen is complete. It
learned history-specific scene semantics but failed the frozen temporal and
absolute-quality gate at every selectable NFE; the video branch therefore
remains disabled. See
[`docs/experiments/VIDEO_LATENT_FORCING_CAUSAL_VJEPA2_RESULTS.md`](docs/experiments/VIDEO_LATENT_FORCING_CAUSAL_VJEPA2_RESULTS.md)
for the measured result, evidence hashes, scope, and next diagnostic.

The initial bootstrap implements and tests the contracts required before Wan
integration:

- complete per-view causal RFFT features on the pre-VAE 13-frame sequence;
- localized per-view STFT features as an ablation;
- separate video/TF Gaussian noise, clocks, velocity targets, and loss masks;
- aligned, independent, TF-leading, and TF-first cascaded schedules;
- a zero-initialized TF token adapter and velocity head that are exact no-ops at
  initialization.

The feature flag is intentionally **off** for production training.  This commit
does not claim a quality improvement and does not launch a B200 job.  Run the
CPU-safe contract checks with:

```bash
pytest -q \
  robot_wm/tests/test_dual_time_frequency.py \
  robot_wm/tests/test_dual_flow.py \
  robot_wm/tests/test_dual_adapters.py
python tools/dual_diffusion_contract_smoke.py
```

The implementation and experimental gates are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).  The mapping from Latent
Forcing is explicit in
[`docs/LATENT_FORCING_PORT.md`](docs/LATENT_FORCING_PORT.md).

## Upstream LACWM baseline

The remainder of this README describes the inherited production baseline.

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
  (`action_pool`) pools the per-latent-frame actions into the control channel. Because
  dataset action chunk `i` moves sampled frame `i` to `i+1`, supervision/explicit
  conditioning uses chunks 4–11 for the five-history/eight-future layout.
- **Masked flow loss**: masked/missing camera views (constant pixels) and zero-padded
  frames (short trajectories) are kept as *inputs* but excluded from supervision.
  The masked pixel mean is computed per sample before batch averaging, so a
  three-view robot does not automatically receive three times EgoDex's weight.
- **Flow matching**: `noisy=(1-σ)x+σ·noise`, `target=noise-x`, logit-normal timesteps,
  `FlowMatchEulerDiscreteScheduler`. Text conditioning is a cached null umT5 embedding
  (no T5 in the train loop).
- **Attention backend**: short-sequence attention (e.g. the inverse model's temporal
  attention) uses the exact **MATH** SDPA backend — FlashAttention's fused backward
  emits NaN on degenerate (constant-across-time) keys. See
  `robot_wm/modeling/modules/attention.py`.

## Setup

The supported training stack is native **PyTorch 2.7.1 + CUDA 12.8**, launched
with `torchrun`/NCCL DDP on one to four 8xB200 nodes. The 1.3B backbone fits on each B200 and only the LoRA and
robot-conditioning modules are trainable, so Megatron tensor/pipeline parallelism
would add conversion and communication complexity without solving a memory problem.

Create the exact environment, activate the pinned VideoX-Fun overlay, and prepare
the official Wan assets from the repo root:

```bash
tools/env/create_b200_env.sh
source /mnt/data2/$USER/lacwm_runtime/envs/lacwm-b200-py310/bin/activate
source tools/env/activate_b200.sh
tools/env/prepare_wan_assets.sh --device cuda:0
python tools/env/verify_b200_runtime.py --expected-gpus 8 --require-b200 \
  --wan-dir "$WAN_DIR"
```

The scripts pin Python packages, VideoX-Fun commit
`1d6d9c3e1540968466937129fef4b288041e06de`, and Wan model revision
`ce96ebd52b1134d2c8a903ceb491ab27aa1e5b7c`. Large environments, model assets,
data, and outputs default to `/mnt/data2/$USER/lacwm_runtime`; override
`LACWM_BASE` to use another `/mnt/data1` or `/mnt/data2` volume. If the host does
not already provide Python 3.10.20, the environment script checksum-verifies and
installs pinned `uv` 0.10.9 under that runtime volume, then uses it to provision
the exact Python patch release without changing shell profiles.

`environment.yaml` remains a Conda convenience definition, but the B200 scripts
above are the canonical, fully validated path. To use the Conda alternative:

1. **Environment** — Python 3.10.20 + PyTorch 2.7.1 (cu128):
   ```bash
   conda env create -f environment.yaml
   conda activate lacwm-dit
   export LACWM_BASE=/mnt/data2/$USER/lacwm_runtime
   export VIDEOX_HOME=$LACWM_BASE/VideoX-Fun-1d6d9c3
   VIDEOX_HOME="$VIDEOX_HOME" tools/env/prepare_videox_fun.sh
   source tools/env/activate_b200.sh
   ENV_DIR="$CONDA_PREFIX" tools/env/prepare_wan_assets.sh --device cuda:0
   python tools/env/verify_b200_runtime.py --expected-gpus 8 --require-b200 \
     --wan-dir "$WAN_DIR"
   ```
2. **VideoX-Fun** supplies `videox_fun.models.*`; use the pinned checkout and
   namespace overlay prepared by the commands above.
3. **Weights** — `alibaba-pai/Wan2.1-Fun-1.3B-Control` (VideoX-Fun single-file format)
   under `$WAN_DIR`. `prepare_wan_assets.sh` generates the cached
   `null_prompt_umt5.pt`; the 11 GB text encoder is not loaded during training.
4. **Data** — the 4 datasets under `$LACWM_DATA` (`abc_pp`, `agibot`,
   `droid_lerobot`, `egodex_cdn`). The active run uses 35,671 episodes (DROID 10k,
   EgoDex 10k, ABC 10k, Agibot 5,671). See **Data** below for sources + the setup script.

## Data

`setup_training.sh` provisions the environment, manifests, paths, and validation,
but dataset downloads are deliberately opt-in. A bare invocation uses `FETCH=skip`
with every dataset disabled and cannot start a multi-terabyte transfer.

```bash
# Explicit finite DROID download (10 chunks, approximately 10k episodes):
FETCH=download ALLOW_DATA_DOWNLOAD=1 DROID_ENABLE=1 DROID_LIMIT=10000 \
  ./setup_training.sh datasets

# Build manifests for already prepared data, then run the strict launch gate:
FETCH=skip DROID_ENABLE=1 EGODEX_ENABLE=1 EGODEX_PARTS=part2 AGIBOT_ENABLE=1 ABC_ENABLE=1 \
  ./setup_training.sh manifests
FETCH=skip DROID_ENABLE=1 EGODEX_ENABLE=1 AGIBOT_ENABLE=1 ABC_ENABLE=1 \
  ./setup_training.sh validate

# Run the validator directly (read-only; exits nonzero on any gap):
python tools/validate_training_data.py --data-root "$LACWM_DATA"
```

Each dataset has `<DS>_ENABLE` (default `0`) and a required positive finite
`<DS>_LIMIT`; `all` is rejected. DROID volume follows `DROID_LIMIT` →
`ceil(limit/1000)` chunks. EgoDex additionally requires explicit `EGODEX_PARTS`;
downloads also require an `EGODEX_SHA256_PLAN` with one trusted `<part> <sha256>`
line per archive because the upstream project does not publish checksums.
ABC requires `ABC_DOWNLOAD_PLAN`, containing exactly `ABC_LIMIT` Hugging Face
`episode.mcap` paths. AgiBot requires `AGIBOT_ARCHIVE_PLAN`, with one reviewed
`<section> <HF .tar path> <sha256>` record per official archive, plus an exact
`AGIBOT_EPISODE_PLAN` CSV (`task_id,episode_id,dataset`). The setup script downloads
only those paths at the pinned upstream revision, verifies every hash, safely extracts
them, and qualifies only the planned episodes. Qualification strictly decodes every
video frame, hashes each of the seven runtime payloads per episode, and never derives
motion from static calibration or fills missing base poses.

For an already extracted tree, run a read-only structural preview. It deliberately
cannot certify lineage or publish a production manifest:

```bash
python tools/prepare_agibot.py \
  --root "$LACWM_DATA/agibot" --limit 5671 \
  --episode-plan /path/to/agibot_episodes.csv --validate-all
```

Only the verified archive flow accepts `--execute`; it extracts into a plan-bound clean
staging tree and atomically publishes `manifest.success.csv`, `manifest.csv`, and
`preparation_report.json` plus `payloads.sha256` with the final corpus. Production
validation re-hashes those payloads and independently rechecks the claimed archive
path/hash/size records against the pinned Hugging Face revision, so it needs authorized
Hugging Face network access. Incomplete episodes fail publication.
Qualification-only sample data and its synthesized camera/base streams are rejected by
the production preparer and validator.

### Paths (no hardcoded locations)

All filesystem locations are environment-driven. `source tools/env/activate_b200.sh`
sets the canonical defaults:

| Var | Default | What |
|---|---|---|
| `LACWM_DATA` | `/mnt/data2/$USER/lacwm_runtime/data` | dataset root |
| `LACWM_RUNS` | `/mnt/data2/$USER/lacwm_runtime/runs` | training outputs |
| `WAN_DIR` | `/mnt/data2/$USER/lacwm_runtime/wan_fun_1.3b_control` | Wan weights + null prompt |
| `VIDEOX_HOME` | `/mnt/data2/$USER/lacwm_runtime/VideoX-Fun-1d6d9c3` | pinned VideoX-Fun |

To relocate, set `BASE` (or the individual vars) when running `setup_training.sh`; it writes
`.lacwm_env` with the exports. Source that file before training/evaluation.

### Sources & expected layout (under `$LACWM_DATA`)

| Dataset | Source | On-disk layout (what the loader reads) | Notes |
|---|---|---|---|
| DROID | [cadene/droid](https://huggingface.co/datasets/cadene/droid) | `droid_lerobot/{data,meta,videos}`, `data/chunk-*/episode_*.parquet` | LeRobot **v2.1**, loads directly. (`lerobot/droid_1.0.1` is **v3.0** file-packed parquet and is *not* loadable as-is.) `DROID_LIMIT` → `ceil(limit/1000)` chunks |
| Agibot | [agibot-world/AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) | `agibot/{observations,proprio_stats,parameters}/<task>/<ep>` (`task_info` optional) | exact checksummed archive plan required; production preparation requires genuine `*_aligned.json` camera series and T-sized robot base streams and never synthesizes them |
| EgoDex | [apple/ml-egodex](https://github.com/apple/ml-egodex) (zips on Apple's CDN) | `egodex_cdn/<part>/<task>/<n>.hdf5` | direct, no preprocessing; the active run used `part2.zip` |
| ABC | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) | `abc_pp/<task>/episode_*/{top,left_wrist,right_wrist}.mp4 + states.npz` | HF ships raw `episode.mcap`; setup downloads only paths explicitly listed in `ABC_DOWNLOAD_PLAN`, then runs the mcap→`abc_pp` preprocessor |

### Manifests

Loaders read manifests of **absolute** paths, so they are regenerated per server by
`./setup_training.sh manifests` (never copy egodex/abc manifests between machines):
- `egodex_cdn/manifest.csv` — one `*.hdf5` path per line (enumerated from the data)
- `abc_pp/manifest.success.txt` — atomic preprocessing provenance containing only
  episodes with `states.npz` and all three MP4s
- `abc_pp/manifest.txt` — finite runtime manifest derived only from the success
  manifest; partial `episode_*` directories are never enumerated
- `agibot/manifest.csv` — `task_id,episode_id,dataset` (path-portable; an existing curated
  manifest is preserved by default; production regeneration is permitted only through
  the clean checksummed-archive flow—`REBUILD_AGIBOT_MANIFEST=1` intentionally fails
  rather than relabel an unbound extracted tree)
- DROID needs no manifest — the loader globs `data/chunk-*/episode_*.parquet`

## Training

Use the fail-closed wrappers in `tools/`; see `tools/README.md` for every option.
First validate gradients on one idle GPU with real samples from all four datasets:

```bash
tools/run_gradient_smoke.sh \
  --gpu 0 --variant latent --data-mode real \
  --python "$LACWM_PYTHON" --wan-dir "$WAN_DIR" \
  --videox-home "$VIDEOX_HOME" --data-root "$LACWM_DATA" \
  --run-root "$LACWM_RUNS" --wandb-mode disabled --execute
```

Then run the strict data validator and preview the full command. The wrapper requires
eight idle B200s, exact runtime/assets, a clean Git tree, a recent fingerprint-bound
data report, and a real-data gradient report from the same commit. Before the long run
it executes both an eight-rank NCCL probe and three real-data accumulated DDP
optimizer/scheduler updates using the production physical microbatch and loader settings:

```bash
mkdir -p "$LACWM_RUNS"
python tools/validate_training_data.py --data-root "$LACWM_DATA" --workers 16 --json \
  > "$LACWM_RUNS/data_validation.json"

tools/launch_8xb200.sh \
  --gpus 0,1,2,3,4,5,6,7 --variant latent \
  --python "$LACWM_PYTHON" --wan-dir "$WAN_DIR" \
  --videox-home "$VIDEOX_HOME" --data-root "$LACWM_DATA" \
  --run-root "$LACWM_RUNS" --run-name lacwm_latent_v1 \
  --smoke-report /path/to/gradient_report.json \
  --data-validation-report "$LACWM_RUNS/data_validation.json" \
  --wandb-mode offline --wandb-project lacwm
# Add --execute only after reviewing the dry run.
```

Resume an interrupted wrapper-owned run with the same arguments plus
`--resume --execute`; atomic `snapshot.pt` replacement prevents a partial checkpoint from
becoming the live resume file.

For shared Slurm clusters with shorter wall-time slots, use
`tools/slurm/submit_8xb200.sh`. The Trainer retains the configured periodic checkpoint
cadence and additionally performs an exact-state checkpoint when Slurm gives advance
notice. The batch job self-requeues only after the durable checkpoint ACK, then selects
`--resume` automatically on the next allocation. See
[`tools/slurm/README.md`](tools/slurm/README.md); the run directory must be on storage
shared by every eligible B200 node.

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
- `data_loader.batch_size` — physical per-GPU microbatch; the 80 GiB latent profile
  uses 4 with `trainer.config.gradient_accumulation_steps=4`. Across eight ranks the
  effective global batch remains `4 × 4 × 8 = 128` samples per optimizer update.
