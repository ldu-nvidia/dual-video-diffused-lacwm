# VPM LaMo macro motion-drift screen (prospective)

## Status and research question

This protocol is fixed before any metric from either continuation arm is
computed. It tests a deployable alternative to future-feature dual diffusion:
can self-supervised motion structure improve low-step video generation when it
is supplied only as a **training target**, with no future feature or auxiliary
model at inference?

The answer is scoped to the immutable 512-train/64-validation ABC screen. The
protected 128-clip test split is not configured, opened, scored, or selected
on. A positive gate is validation evidence, not a final physical-realism,
latency, or generalization claim.

## Source and adaptation

[LaMo](https://arxiv.org/abs/2605.23878v1) defines a parameter-free macro
latent motion statistic and its scale-normalized loss in Eqs. 5–6. For clean
video latents `z`, predicted clean latents `x0_hat`, spatial positions `(h,w)`,
and temporal lag `tau`, its statistic is

```text
mu*     = mean_hw(z[i+tau]      - z[i])
mu_hat  = mean_hw(x0_hat[i+tau] - x0_hat[i])

L_drift = mean_bi ||mu_hat - mu*||_2^2
                   / stopgrad(||mu*||_2^2 + epsilon).
```

LaMo adds `lambda_drift * mean(alpha_bar) * L_drift` to its denoising loss,
using `lambda_drift=0.4`. This screen keeps that weight and fixes
`epsilon=1e-6` prospectively.

LACWM uses rectified flow with the opposite-noise clock convention:

```text
sigma = 1 : Gaussian noise
sigma = 0 : clean data

x_sigma = (1-sigma) x0 + sigma epsilon
v*      = epsilon - x0
x0_hat  = x_sigma - sigma v_theta.
```

The clean-signal coefficient is `1-sigma`, so the exact signal-power analogue
of LaMo's `alpha_bar` is `(1-sigma)^2`. The adapted objective is therefore

```text
L_train = L_video_flow
        + lambda_drift * mean_global_b((1-sigma_b)^2) * L_drift.
```

The schedule mean is explicitly all-reduced across the effective eight-sample
DDP batch. With local batch one, silently using a per-rank mean would instead
optimize `mean_b(alpha_b * L_drift,b)`, which is not Eq. 6's product of batch
means.

This has the intended high-noise limit: at `sigma=1`, the drift term receives
zero schedule weight because `x0_hat` is least reliable. No loss is placed on
`x_sigma`, and no epsilon-prediction formula is mixed into the RF model.

## Exact temporal geometry and lag

The frozen VPM uses a Wan causal VAE with temporal ratio four:

```text
13 RGB frames = 5 observed history + 8 predicted future
Wan latent length = floor((13-1)/4)+1 = 4 tokens
history latent length = floor((5-1)/4)+1 = 2 tokens
future latent length = 4-2 = 2 tokens.
```

Only the two future tokens enter the macro statistic:

```text
z_future = z[:,:,2:4]
Delta_future = z_future[:,:,1] - z_future[:,:,0]
mu = mean_hw(Delta_future).
```

There is exactly one future-to-future difference. We therefore fix `tau=1`.
LaMo's default `tau=2` is not estimable here: it would require a third future
Wan token. The observed-history-to-first-future boundary difference is
explicitly excluded so the auxiliary target cannot be satisfied by copying
history appearance rather than learning predicted-horizon motion.

## Frozen starting point and arms

Both arms load every model key from the exact update-1,000 parameter-matched
video-only VPM snapshot in
`vjepa2-controlled-20260730-seed1234-9cf8e69-v3`, trained at
`9cf8e6922f35a5d6645e3128545953723bf54da2`:

```text
snapshot SHA-256:
f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21
```

| Arm | Video-flow loss | Macro drift weight | Role |
|---|---:|---:|---|
| `VPM-CONT` | 1 | 0.0 | ordinary matched continuation |
| `VPM-DRIFT` | 1 | 0.4 | LaMo macro-drift candidate |

The model class, parameter names/shapes/count, Wan input/output, action
conditioning, and deployment sampler are identical. Both arms execute the same
capture, RF `x0_hat`, macro-drift, and telemetry code path. The zero-weight arm
returns the original VPM loss object unchanged, so a `0 * auxiliary_graph`
cannot introduce a gradient or turn a non-finite diagnostic into the control
objective.

The VPM auxiliary feature state remains an exact no-op:

- auxiliary-state conditioning off;
- auxiliary-clock conditioning off;
- auxiliary loss weight zero;
- state and clock gates exactly zero;
- parameter-matched unused auxiliary schema retained.

The inherited 64-channel auxiliary head receives a deterministic all-zero
placeholder in both arms. This preserves the historical VPM parameter and call
topology without opening the old V-JEPA target array or constructing any new
clean auxiliary feature. The placeholder has no video-path effect because both
injection gates and the auxiliary objective are exact no-ops. The historical
auxiliary-head-only clock remains enabled for configuration fidelity; it never
enters the shared video trunk or the training objective.

Thus this screen tests training-time motion supervision, not future-feature
conditioning or an inference-time dual stream.

## Matched fine-tuning

Each arm performs exactly 200 fresh fine-tuning optimizer updates from the
same VPM model weights. With global batch eight, this is 1,600 clip exposures,
or 3.125 nominal passes over the 512-clip population. The endpoint is fixed
before metrics.

Across arms, the following are exact matches:

- seed `1234`, eight B200 ranks, local batch one, global batch eight;
- immutable 512-row train manifest and RGB/action arrays;
- deterministic manifest permutation, rank/worker sharding, clip order;
- video corruption, timestep, LoRA dropout, and loader RNG initialization;
- AdamW hyperparameters (`lr=1e-4`, betas `0.9/0.95`, `eps=1e-8`,
  `weight_decay=0.01`), 20-step warmup, 200-step decay to `1e-6`, gradient
  clipping, bfloat16 AMP, and checkpoint cadence;
- model-only initialization from the same parent snapshot, with a fresh
  identical optimizer in each arm;
- no EMA in the historical LACWM Trainer and no EMA in either arm. Both are
  evaluated from their non-EMA update-200 snapshots.

Trainer-side validation at updates 0, 100, and 199 consumes four local batches
per rank. This is exactly one global pass over the 64 validation clips each
time (`8 ranks x 2 clips x 4 batches`); the iterator cycles the same fixed
rank shard so later validations cannot exhaust it. These training-loop losses
do not select a checkpoint (`save_best=false`); the dedicated all-64 paired
evaluator scores only the fixed update-200 endpoint.

The loss adds no random draw. A rank-reduced trace records, at every update,
the clip-index first and second moments, timestep first and second moments, and
fixed projections of clean/noisy Wan inputs. It additionally stores all-rank
SHA-256 digests of the exact clip indexes, actions, clean/noisy Wan latents,
timesteps, and CPU/CUDA RNG states after the shared forward call. Analysis
fails before inspecting quality effects unless all 200 paired audit records,
exact hashes, and the learning-rate schedule match exactly.

The clean 13-frame video is encoded by the already-frozen Wan VAE. Its two
future latent tokens are a training target only. They are never passed to the
Wan transformer as conditioning.

Telemetry is written online to the authenticated private personal project
`zijiandu/dual-video-diffusion-private`, with `group=null`. Registration and
each arm job re-query W&B and require `access=PRIVATE` and viewer
`zijiandu`. The content-bound arm identity is also the W&B run ID, preventing a
fresh study root from silently resuming or merging with an earlier attempt.
W&B finalization is bounded at 120 seconds and non-raising after durable local
outputs exist; unsent local files are retained for a later `wandb sync`.

## Deployable evaluation boundary

Evaluation uses all 64 immutable validation clips, eight B200 ranks, and two
clips per rank. The sampler receives only:

- five observed RGB frames;
- the requested future action chunks and morphology ID;
- deterministic Gaussian video/unused-auxiliary noise keyed by validation
  clip ID with fixed base seed `20260726`.

It does not receive the remaining eight RGB frames, a clean Wan future latent,
a V-JEPA/DINO/other feature, a target-cache tensor, or an online teacher. The
RGB/action-only loader never resolves or opens `targets.fp16.npy`.

For each arm and clip, ordinary autonomous Euler sampling is run independently
from the same initial noise at fixed NFE:

```text
NFE in {1, 2, 4}.
```

A forward hook counts actual Wan/transformer calls. A row is invalid unless
actual calls equal declared NFE. Per-clip hashes bind the cached full clip, the
sampler's exact five-frame RGB input, cached and sampler action tensors,
scoring target, initial video noise, and initial unused-auxiliary noise within
and across arms.

Only after all sampler endpoints for a batch have completed does the evaluator
access the held-back eight RGB frames and encode the complete clip to construct
metrics. Ground truth is evaluator-owned scoring data, not sampler input.

### Action-shuffled diagnostic

At NFE 1 only, each arm also receives the other sample's actions within its
fixed local two-clip batch while retaining local history and noise. This is a
preregistered diagnostic of action sensitivity. It is not part of the primary
gate, does not add a tunable candidate, and cannot promote an arm.

## Metrics and decision rule

Primary metric, lower is better:

- decoded temporal-difference MSE on `[last observed RGB, 8 predicted RGB]`
  against `[last observed RGB, 8 ground-truth RGB]`, in `[0,1]` units.

Guardrails, lower is better:

- future Wan-latent NMSE;
- decoded future RGB MSE in `[0,1]` units.

Future-latent delta NMSE and decoded PSNR are diagnostics only.

For each of the three NFE values and each claim metric, paired relative
improvement is

```text
I = (mean(VPM-CONT) - mean(VPM-DRIFT)) / mean(VPM-CONT).
```

The analyzer uses 10,000 paired clip bootstrap replicates, seed `20260807`.
The family contains `3 NFE x 3 metrics = 9` predeclared contrasts. One-sided
Bonferroni lower bounds therefore use confidence

```text
1 - 0.05/9 = 0.9944444444444445.
```

An NFE endpoint passes only if:

- decoded temporal MSE point estimate and simultaneous lower bound are both at
  least `+1%`; and
- latent NMSE and decoded MSE point estimates and simultaneous lower bounds are
  each greater than `-1%`.

The validation screen passes if any fixed NFE endpoint passes. The selected
diagnostic endpoint is the lowest passing NFE. No hyperparameter, checkpoint,
or NFE is changed after metrics. The action-shuffled result is reported
descriptively and cannot enter this gate.

## Provenance and fresh-output rules

Registration, before training or candidate metrics, rehashes:

- the clean tool commit (required to descend from `9cf8e69`) and exact clean
  historical `9cf8e69` checkout;
- the VPM study/arm/stage identities and 4.25 GB parent snapshot;
- the 512-row train manifest, metadata, 6.9 GB RGB array, and action array;
- the 64-row validation manifest, metadata, RGB array, and action array;
- the protocol bytes.

It also records the pinned virtual-environment launcher without resolving away
its final symlink, validates the Wan asset directory, and requires the exact
clean VideoX commit. The Python environment and Wan directory are path/runtime
gates rather than recursively content-hashed inputs; the strict full-model
snapshot load and parameter shape/dtype gate bind the model state actually
evaluated.

It parses every train and validation manifest row, verifies the exact 13-frame
geometry, dense identities, and one-clip-per-source-episode property, and fails
if any clip ID or source episode occurs in both splits. The registered
disjointness hashes are recomputed at each later point of use. The registered
Python path preserves the pinned virtual-environment launcher rather than
resolving it to the environment's base interpreter.

Each arm plan rehashes the registered parent snapshot plus the selected
train/validation manifests, metadata, RGB, and action arrays again immediately
before training. Rank zero rehashes the registered validation inputs once more
before evaluation creates any output. Neither revalidation resolves or opens
the auxiliary target array.

Every artifact is exclusive-create. A failed attempt is retained and cannot be
silently resumed or overwritten; a retry requires a fresh study root. Slurm
jobs are non-requeueable, use one node with eight B200 GPUs, short QOS, and
exclude `pool0-0081,pool0-0089`.

The static Lustre Slurm-log directory declared by the sbatch file must exist on
the cluster before submission. The implemented workflow intentionally neither
creates that external directory nor submits a job during this prospective
phase.

## Implementation and execution order

- `projects/latent_action_models/lam/lamo_motion_drift_model.py`: exact RF
  `x0_hat` and macro-drift loss;
- `robot_wm/datasets/abc/lamo_motion_drift_fixed_dataset.py`: immutable
  RGB/action-only training data;
- `robot_wm/utils/lamo_motion_drift_trainer.py`: strict parent-schema gate and
  per-update paired audit;
- `tools/lamo_motion_drift_evaluate.py`: registration, deployable sampler,
  scoring, call/hash inventories;
- `tools/analyze_lamo_motion_drift.py`: paired bootstrap and fixed gate;
- `tools/slurm/lamo_motion_drift_workflow.py`: arm receipts and dry-run plan;
- `tools/slurm/lamo_motion_drift.sbatch`: non-requeueable eight-B200 entrypoint.

Execution order is:

1. register a fresh root;
2. render and inspect the dry-run plan;
3. run the two independent `arm` jobs (training followed by validation);
4. after both succeed, run `analyze`;
5. do not open protected test under this protocol, regardless of the result.
