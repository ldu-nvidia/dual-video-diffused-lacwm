# Causal Spectral Inverse-Dynamics Probe (CSIP), Phase 0

Date frozen: 2026-08-08, before latent extraction, probe fitting, or validation

Status: implementation and prospective protocol only. This commit creates no
run, W&B record, checkpoint, metric, or generator change.

The implementation must be a descendant of reviewed integration commit
`e4fda453ed4b92173848ea12b5538749069c9032`; registration verifies this ancestry.

## Decision question

Before spending another video-generation training budget, test the necessary
premise behind action-aware spectral forcing:

> Do clean, real Wan future-motion latents contain phase-sensitive spectral
> information that identifies the action which produced the motion, beyond an
> episode-disjoint action shuffle, zero action, and sign-inverted action?

This is a frozen inverse-dynamics probe, not dual diffusion. A pass justifies a
later training-only generator regularizer in which this probe is frozen and
discarded at inference. A failure stops that path. It cannot itself establish
better video fidelity, faster denoising, fewer NFE, or real-time DAgger.

## Immutable data boundary

The screen uses the existing content-addressed planner populations:

```text
train: 512 unique ABC clips/episodes, RGB [512,13,3,180,960]
       actions [512,13,5,23]
val:    64 unique and train-disjoint ABC clips/episodes
test:    0 paths accepted, 0 clips read
```

The train population is prospectively partitioned by immutable
`auxiliary_index`:

```text
fit448: auxiliary_index % 8 != 0
cal64:  auxiliary_index % 8 == 0
```

Only fit448 fits the action PCA and probe weights. Cal64 is monitored at fixed
updates 0, 100, 200, 300, and 400; it cannot select a checkpoint, stop fitting,
or tune a threshold. The only checkpoint is the final update 400.

Registration performs full SHA-256 verification of train and validation RGB
and action arrays and binds the exact source commit, Python executable,
Python package versions, VideoX-Fun commit
`1d6d9c3e1540968466937129fef4b288041e06de`, Wan configuration, and Wan VAE
weights. After registration,
train latent extraction and fitting validate train records only. Validation
latent extraction is rejected until the update-400 checkpoint and training
report have been sealed. Thus fitting never opens validation RGB, actions, or
latents. Protected test is absent from every CLI.

## Clean real-video motion latent

Encode each 13-frame clip and the five observed frames independently with the
same frozen Wan VAE:

```text
full latent       Z   [B,16,4,24,120]
history latent    ZH  [B,16,2,24,120]
```

Require

\[
\max|Z_{0:2}-Z^H|\le 10^{-4}.
\]

The independently encoded history prevents a full-clip temporal encoder from
leaking future information into the observed anchor. Define the two future
motion tokens

\[
D_0=Z_2-Z^H_1,\qquad D_1=Z_3-Z_2.
\]

These are ordinary clean training/evaluation targets. They are never proposed
as deployment-time features.

## Per-view coarse spectral representation

The latent width contains three stacked camera views. Split it first:

\[
D^{(v)}=D[:,:,:,:,40v:40(v+1)],\qquad v\in\{0,1,2\}.
\]

No FFT crosses a camera seam. For every sample, view, and Wan channel, compute
the orthonormal spatiotemporal transform

\[
F^{(v,c)}(k_t,k_y,k_x)=
\frac{1}{\sqrt{2\cdot24\cdot40}}
\sum_{t=0}^{1}\sum_{y=0}^{23}\sum_{x=0}^{39}
D^{(v,c)}_{t,y,x}
e^{-2\pi i(k_tt/2+k_yy/24+k_xx/40)}.
\]

After FFT-shifting all axes, retain both temporal bins and the centered
`4 x 6` spatial crop. Let `A=|F|` and

\[
M=\mathbf 1\left[A\ge 10^{-3}
\sqrt{\operatorname{mean}_{k_t,k_y,k_x}A^2}\right].
\]

The fixed volume channels are

\[
\log(1+A),\qquad M\operatorname{Re}\frac{F}{\max(A,\epsilon)},\qquad
M\operatorname{Im}\frac{F}{\max(A,\epsilon)}.
\]

To retain direction without an angle branch cut, also take a 2-D spatial FFT
`S_t` of each motion token and form the unit phase increment

\[
\Delta U=U_1\overline{U_0}
=e^{i(\phi_1-\phi_0)},\qquad U_t=S_t/\max(|S_t|,\epsilon),
\]

masked only where both endpoint energies pass the same relative-energy rule.
Its real and imaginary components use the same `4 x 6` crop. Concatenating all
views and channels gives exactly

```text
3-D log magnitude/unit phase: 3 * 3 * 16 * 2 * 4 * 6 = 6,912
phase-increment real/imag:     2 * 3 * 16 * 1 * 4 * 6 = 2,304
total feature:                                              9,216
```

The negative exponential above fixes the phase sign. A positive one-pixel
spatial shift must multiply frequency `kx` by `exp(-2*pi*i*kx/40)`. Unit tests
freeze this sign, the energy mask, the phase-increment conjugacy, and per-view
isolation.

## Train-only action target

Use raw, unpadded action chunks `4:12`. For each five-step chunk form its four
within-chunk differences, then flatten:

\[
q=\operatorname{vec}(A_{4:12,1:5,:}-A_{4:12,0:4,:})\in\mathbb R^{736}.
\]

This avoids the nearly rank-one absolute-action shortcut documented in the
earlier action-conditioning audit. Fit a 16-component PCA on fit448 only,
canonicalize every component sign by making its largest-magnitude loading
positive, and whiten each retained score by its fit448 population standard
deviation. Persist the mean, components, scales, and fit indexes in the fixed
checkpoint. Cal64 and val64 contribute zero PCA elements.

## Frozen probe and fit

The probe is fixed before metrics:

```text
LayerNorm(9,216)
Linear(9,216,256) + SiLU
Linear(256,16)
```

Fit MSE with AdamW, learning rate `3e-4`, weight decay `1e-4`, global-norm
clip 1.0, batch 64, seed 1234, and exactly 400 updates. Batch 64 divides fit448
into seven batches, with a seeded permutation at each new pass. Calibration
telemetry cannot change this schedule. Training must create an online run in
the verified personal private W&B project
`zijiandu/dual-video-diffusion-private`, with `group=null` and `resume=never`.

## Sealed val64 controls

For each val clip, compare the same probe prediction with four targets:

1. `aligned`: its own action descriptor;
2. `episode_disjoint_cyclic_shuffled`: the first whole-population cyclic shift
   for which no clip receives an action from its own source episode;
3. `zero`: the PCA transform of the all-zero raw delta descriptor;
4. `inverse`: the PCA transform of the sign-negated local raw delta descriptor.

The shuffled control tests action identity rather than dataset-level motion;
zero tests collapse to an average/no-action target; inverse tests directional
phase/action sign. Every row records the local and donor clip/episode IDs.

## Fixed statistical gate

For each of three controls, evaluate paired per-clip MSE and cosine:

```text
MSE effect:    control MSE - aligned MSE       (positive favors aligned)
cosine effect: aligned cosine - control cosine (positive favors aligned)
```

Use exactly 10,000 paired clip bootstrap resamples, seed 20260808. There are
six preregistered cells, with one-sided Bonferroni cell alpha `0.05/6`. Phase 0
passes only if every cell has a positive point effect and a strictly positive
simultaneous lower bound. No subset, metric, threshold, checkpoint, or seed may
rescue a failure.

On pass, the next experiment is a matched generator ablation using the frozen
probe as a training-only action/spectral consistency loss on the denoiser's
predicted-clean future latent. The probe is removed at inference, directly
avoiding the unavailable-clean-feature problem. On failure, do not add a CSIP
branch to VPM; investigate other causal objectives such as cycle consistency
or learned action-state prediction under a new preregistration.
