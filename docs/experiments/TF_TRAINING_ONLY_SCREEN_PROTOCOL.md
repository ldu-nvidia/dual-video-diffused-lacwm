# Training-only spatiotemporal spectrum screen

## Question

Can a privileged frequency-domain objective improve deployable one-step video
generation even when inference has no clean future feature, auxiliary state,
second diffusion clock, teacher, or TF network? This is a different mechanism
from synchronous dual diffusion: spectrum is a training loss on Wan's own
predicted clean endpoint and is discarded after optimization.

## Matched arms

Both arms start from the same frozen update-1000 VPM parent, after discarding
the parent's three dormant dual-TF module prefixes. Both instantiate the
ordinary explicit-action Wan model, with exactly the same parameter names and
count, train512 order, seed 1234, eight B200 ranks (global batch eight), AdamW,
schedule, action inputs, noise/timestep stream, and 200 optimizer updates.

- `TFREG-OFF`: spectral loss weight 0.
- `TFREG-ON`: spectral loss weight 0.05.

No dose is selected from validation. The 0.05 treatment is prospectively fixed
as a modest regularizer because its component losses are dimensionless after
per-sample/channel target-RMS normalization.

## Exact feature and loss

For flow convention

\[
x_\sigma=(1-\sigma)x_0+\sigma\epsilon,\qquad
v=\epsilon-x_0,
\]

the model's clean estimate is

\[
\hat{x}_0=x_\sigma-\sigma\hat v.
\]

Only the two future Wan tokens are selected. Thus each signal has shape
`[B, 16, 2, 23, 120]`: batch, VAE channel, future latent time, latent height,
and the width-stacked three-camera grid. We compute one orthonormal real 3-D
FFT over `(T,H,W)`. All temporal frequencies are retained. Spatial frequencies
are fixed to the rectangle `|fy| <= 0.25` and `fx <= 0.25` cycles/sample, i.e.
the lower 50% of spatial Nyquist on each axis. This captures temporal/spatial
phase jointly; it is not the earlier per-pixel temporal-only RFFT packing.

The amplitude term is Smooth-L1 between `log1p(|F(hat x0)|/s)` and
`log1p(|F(x0)|/s)`, where `s` is detached target spectral RMS per
sample/channel. The phase term is

\[
1-\frac{\Re(\hat X X^*)}
{\sqrt{|\hat X|^2+\delta^2}\sqrt{|X|^2+\delta^2}},
\]

weighted by capped target magnitude; global DC is excluded. The combined
training objective is `flow MSE + 0.05 * (amplitude + 0.25 * phase)`. Invalid
views/time slots are zeroed in both signals before FFT and receive no gradient.
FFT arithmetic is float32; model arithmetic remains bfloat16.

## Causal deployment proof

Before either arm trains, a real one-clip B200 NFE1 canary passes an independent
five-frame history allocation plus actions to the deployment sampler. A patched
spectral-loss function raises if reached. Changing the scalar loss weight from
0 to 0.05 must leave both native bfloat16 Wan endpoint and decoded output
bitwise identical under the same keyed noise. The canary additionally requires
one Wan call, zero auxiliary modules/parameters/inputs, no cached V-JEPA target
open, and no future RGB input. Training depends on this canary.

Final evaluation repeats that target-free sampler for all 64 validation clips.
Clean future RGB is moved to the accelerator only after aligned, episode-
shuffled, and zero-action generations finish, and is then used solely to score.
The protected test split has no CLI argument and remains unopened.

## Preregistered evaluation and gate

The primary cell is aligned actions, NFE1, 64 paired clips. Lower is better for
latent NMSE, decoded `[0,1]` uint8-grid MSE, and temporal-difference MSE. Each
effect is the relative change in the ratio of paired arm means, with a
deterministic 10,000-draw paired clip bootstrap 95% interval.

- Decoded MSE: at least 3% point improvement and at least 1% lower bound.
- Temporal MSE: at least 3% point improvement and at least 1% lower bound.
- Latent NMSE: nonnegative point improvement and lower bound at least -1%.

All three are required for PASS. Episode-disjoint shuffled and zero actions are
diagnostic controls: they test whether either arm's factual score is masking an
action-insensitive shortcut, but do not replace the primary treatment gate.

## Interpretation boundary

A PASS would establish that Fourier information can help through training-only
representation shaping, without solving feature prediction at inference. A
FAIL rejects this one loss, band, dose, parent, and 200-update/NFE1 regime; it
does not prove that every spectral regularizer is useless. Because only two
future latent time tokens exist, temporal frequency resolution is intrinsically
coarse; a longer-horizon latent model is the proper next venue if phase is the
only promising component.
