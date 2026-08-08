# Low-frequency motion regularizer (LFMREG) controlled screen

## Question

Can a parameter-free, training-only structural loss improve deployable NFE-1
video generation without supplying a clean or predicted future feature at
inference?

This is deliberately distinct from TFREG. It uses no FFT, complex coefficient,
phase target, auxiliary encoder, auxiliary latent, second sampler, or inference
branch.

## Fixed treatment

For Wan's clean future latent `z` and flow endpoint estimate
`x0_hat = x_sigma - sigma * v_hat`, both shaped `[B,16,2,24,120]`:

1. Split the width into the three camera views, each `[B,16,2,24,40]`.
2. Prepend the last clean *observed* Wan token to each trajectory. This creates
   `observed -> future-0 -> future-1`; the observed token is available causally
   and is identical on the prediction and target sides.
3. Apply the same fixed 5x5, sigma-1 Gaussian independently to every view,
   channel, and time. Filtering cannot cross camera seams.
4. Take the two adjacent-time differences.
5. Normalize prediction and target by detached target-motion RMS per
   sample/view, clamped at `1e-4`.
6. Use Smooth-L1 with beta `0.25`. LFMREG-ON adds this loss at fixed weight
   `0.05`; LFMREG-OFF uses `0.0`. The model parameter schema is identical.

The dose is prospective and conservative: the normalized loss is
dimensionless, and flow matching remains the primary objective.

## Scientific limitation of the current geometry

Eight future RGB frames collapse to only two future Wan temporal tokens. A
future-only derivative therefore contains only one coarse transition. Adding
the last observed token gives two transitions and is the nearest defensible
motion-trajectory loss available without changing the parent, cache, horizon,
or model geometry.

This screen cannot localize frame-level contact onset, impact timing, or
high-frequency motion. A positive result would support only coarse
low-frequency latent-motion regularization. It would not establish a contact
loss or a general spatiotemporal-frequency advantage. A contact claim requires
a less temporally compressed tokenizer or supervision in decoded RGB/feature
space at the original eight-frame cadence.

## Paired protocol

- Exact parent VPM snapshot SHA-256: `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`.
- Exact immutable ABC train512 and val64 cache identities inherited from the
  controlled VPM studies.
- Seed 1234, eight B200s, global batch 8, 200 updates, identical optimizer,
  schedule, order, and diffusion noise.
- Only arm difference: motion-loss weight `0.0` versus `0.05`.
- Private personal W&B project `zijiandu/dual-video-diffusion-private`, no
  group.
- One ordinary Wan call at NFE-1. Inference accepts only five observed RGB
  frames and the registered action tensor. The training loss is monkey-patched
  to fail if reached during the deployment canary or evaluation.
- Val64 reports latent NMSE, decoded MSE, and decoded temporal MSE under
  aligned, episode-disjoint shuffled, and zero action controls with keyed,
  identical noise across arms and controls.
- The primary endpoint is aligned-action NFE-1. Treatment effects are also
  reported separately for all action controls, together with within-arm action
  sensitivity.
- Required gate: decoded and temporal MSE each improve by at least 3% with a
  paired-bootstrap lower bound of at least 1%; latent NMSE must be non-worse at
  point estimate with lower bound at least -1%. All conditions must pass.
- No protected test is accepted by any command or opened by the dataset.

## Status

Implementation only. This branch intentionally does not register a study,
submit Slurm jobs, or inspect validation outcomes.
