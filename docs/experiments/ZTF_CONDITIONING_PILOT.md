# ZTF conditioning pilot

## Question and estimand

This pilot asks one narrow question:

> With the TF prediction task, parameters, clocks, optimizer budget, samples,
> seed, and Wan initialization held fixed, does allowing the current generated
> TF state into the video denoiser improve early video denoising?

The two causal models are:

```text
no-ZTF condition:   p(v_video, v_tf | z_video, sigma_video, sigma_tf, history, actions)
with-ZTF condition: p(v_video, v_tf | z_video, z_tf, sigma_video, sigma_tf,
                                        history, actions)
```

Both arms give the TF output head its noisy TF state and train the same future
TF flow loss. Only `condition_on_tf` controls whether TF-state tokens enter
Wan's shared trunk and therefore the video prediction. This avoids conflating
TF conditioning with extra parameters or an auxiliary-loss regularizer.

## Causality and representation

- The model is the explicit-action LACWM. Proposed future actions, not hidden
  future RGB, produce action conditioning.
- A complete per-view length-4 RFFT is computed independently for each of the
  three width-stacked cameras.
- Real and imaginary members are retained. For RGB input the TF state is
  `[B, 12, 4, 24, 120]`; the Wan video state is `[B, 16, 4, 24, 120]`.
- TF bins 0--1 depend only on the five observed history frames and remain
  clamped. Only future bins 2--3 begin at Gaussian noise and receive TF loss.
- Raw coefficients are used in this pilot. Per-sample RMS normalization is
  forbidden because it would make the history scale depend on hidden future
  frames. Fixed training-split whitening is a later representation ablation.

LACWM's clock convention is `sigma=1` for noise and `sigma=0` for clean data.
The production shifted Wan scheduler supplies `sigma_video`; the leading clock
is derived without replacing that schedule:

```text
sigma_tf = sigmoid(logit(sigma_video) - 1)
```

with exact `(1, 1)` and `(0, 0)` endpoints.

## Integration

The pretrained 48-channel Wan patch input and video head are unchanged.
Zero-gated TF and TF-clock tokens use the pinned Wan implementation's existing
additive `y_camera/control_adapter` seam. A guarded pre-head hook exposes the
final shared Wan tokens to a separate zero-initialized TF velocity head.

At initialization both arms have an identical video function. The no-ZTF arm
continues to compute the same TF adapter and consumes the same TF random draws,
but multiplies the state residual by zero before the shared trunk.

## Pilot data and budget

ABC is used for this integration screen because it supplies three native camera
views and avoids a padded-view confound:

```text
dataset: ABC only
episodes sampled: 1,000
seed: 1234
hardware: 1 node x 8 B200 per arm
physical batch: 1 per GPU
optimizer updates: 100
warmup: 20
validation: every 10 updates
trajectory/video probe: updates 1, 51, and 100
sampler: 8 Wan NFEs
```

The 100-update result is a screening observation, not publication evidence.
Phase 4 still requires controlled DROID splits, at least three seeds, and the
random/parameter-matched representation controls in `docs/RESEARCH_PLAN.md`.

## Denoising and autonomous-rollout metrics

For rectified flow,

```text
z_sigma = z_clean + sigma * (noise - z_clean)
z_clean_hat = z_sigma - sigma * v_pred
```

The fixed-sigma forward-pass diagnostic is future-only, view-masked
video-latent NMSE at native Wan scheduler nodes nearest
`sigma_video = 0.90` and `0.75`; validation also records `0.50` and `0.25`.
Those calls corrupt the TF state from the ground-truth clip, so their telemetry
is explicitly prefixed `teacher_forced/`. They measure whether a usable TF
condition could accelerate one-call video denoising, but cannot by themselves
establish faster autonomous generation.

The primary inference endpoint is therefore the corresponding future-only
video-latent NMSE along the saved 8-NFE **joint rollout**, where both future
video and future TF start from Gaussian noise and every later TF state is
model-generated. Joint-rollout NMSE is computed from
`video_x0_trajectory`, `video_clean`, and `history_latent_frames`; the raw
trajectory remains the auditable source. Decoded video quality is evaluated at
the same NFE.

Interpret the pilot as positive only if the conditioned arm:

1. has lower teacher-forced high-noise video NMSE on the paired probe at the
   same update;
2. also has lower generated-TF joint-rollout video NMSE on the paired probe at
   the same update and NFE;
3. retains finite video/TF losses and gradients; and
4. does not merely improve TF NMSE while video NMSE remains unchanged.

This 100-update screen advances the validation/visualization loaders between
checkpoints. Therefore it compares arms only at matching checkpoints; it does
not estimate a within-run time-to-threshold across updates. A confirmatory
learning-speed experiment must cache a fixed validation clip set and fixed
video/TF noise across all checkpoints. At every pilot checkpoint, clean states
and initial noise tensors must hash-identically across the two arms before a
paired difference is interpreted.

The later go gate remains the stricter Phase-4 criterion: at least a 5% early
high-noise gain and a 3% preregistered temporal gain at 8 NFE without a material
decoded-quality regression.

## Telemetry and artifacts

W&B destination:

```text
entity:  zijiandu
project: dual-video-diffusion-private
group:   unset
access:  PRIVATE (verified before upload)
```

Scalars include video/TF flow loss, explicitly teacher-forced video/TF
clean-estimate NMSE, video and TF sigma, state RMS, TF token RMS,
state/clock gates, gradients, throughput, step time, and GPU memory.
Autonomous joint-rollout NMSE is derived from each saved trajectory and is the
inference-side endpoint.

Each visualization checkpoint stores and uploads a safetensors artifact with:

```text
video_trajectory       [S+1, 1, 16, 4, 24, 120]
tf_trajectory          [S+1, 1, 12, 4, 24, 120]
video_x0_trajectory    [S,   1, 16, 4, 24, 120]
tf_x0_trajectory       [S,   1, 12, 4, 24, 120]
video_clean            [1, 16, 4, 24, 120]
tf_clean               [1, 12, 4, 24, 120]
video_sigmas, tf_sigmas
```

Decoded side-by-side ground-truth/prediction videos are logged separately.
All raw tensors and videos remain under the approved Lustre run directory.
