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
> episode-disjoint paired action shuffle, raw-no-action descriptor,
> sign-inverted action, and a capacity-matched angle-neutral probe?

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

Registration is the one permitted pre-seal read and full SHA-256 verification
of train and validation RGB/action source records. It binds the exact source
commit, Python executable, Python package versions, VideoX-Fun commit
`1d6d9c3e1540968466937129fef4b288041e06de`, Wan configuration, and Wan VAE
weights. After registration, every train-stage render, train latent extraction,
fit, and seal operation validates train records only; it does not reopen or
rehash a validation file. A validation-stage render first validates the
update-400 checkpoint seal using train-only records, and only then reopens the
registered validation records. Validation latent extraction is rejected until
that checkpoint and training report are sealed. Thus fitting never opens
validation RGB, actions, or latents. Protected test is absent from every CLI.

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

The latent width contains three stacked camera views. Split it before every
FFT:

\[
D^{(v)}=D[:,:,:,:,40v:40(v+1)],\qquad v\in\{0,1,2\}.
\]

No FFT crosses a camera seam. This is explicitly a per-view transform of the
established width-stacked Wan latent; the Wan VAE encoded the width-stacked
pixels jointly, so this protocol does **not** claim the VAE itself is
seam-independent. For every sample, view, and Wan channel, compute
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
For this endpoint mask, the RMS scale is intentionally shared across the two
motion tokens for each sample/view/channel (it is not fit separately per
endpoint). Its real and imaginary components use the same `4 x 6` crop.
Concatenating all
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

## Frozen matched probes and fit

Two probes are fixed before metrics. Both have the same 9,216-input shape and
architecture:

```text
LayerNorm(9,216)
Linear(9,216,256) + SiLU
Linear(256,16)
```

The `full` probe receives all coordinates. The `angle_neutral` probe receives
the identical tensor, except every masked unit phasor `(Re, Im)` in both the
volume-phase and phase-increment blocks is replaced by `(M, 0)`. Invalid
phasors therefore remain `(0, 0)`, valid phasors become `(1, 0)`, and all 2,304
log-magnitude coordinates are unchanged. This preserves the reliability/energy
support carried implicitly by a nonzero phasor while removing its spectral
angle. Its input dimension is not reduced. Both probes start from the exact
same initial state and see the same batch order, targets, AdamW configuration,
update count, and checkpoint boundary. Thus the prospective difference is
access to spectral angle, not access to magnitude, support, or capacity.

Fit each probe's MSE with AdamW, learning rate `3e-4`, weight decay `1e-4`, global-norm
clip 1.0, batch 64, seed 1234, and exactly 400 updates. Batch 64 divides fit448
into seven batches, with a seeded permutation at each new pass. Calibration
telemetry cannot change this schedule. Training must create an online run in
the verified personal private W&B project
`zijiandu/dual-video-diffusion-private`, with `group=null` and `resume=never`.

## Sealed val64 controls

For each val clip, compare each probe prediction with four targets. The three
target-control gates below use the full probe; both probes are retained for the
matched spectral-angle-contribution gate:

1. `aligned`: its own action descriptor;
2. `episode_disjoint_paired_shuffled`: pair adjacent immutable manifest indexes
   `(0,1), (2,3), ..., (62,63)` and swap the two targets inside each pair; all
   32 pairs contain distinct source episodes and no row is its own donor;
3. `raw_no_action`: the PCA transform of the all-zero raw 736-D delta
   descriptor. Because the train-fit PCA subtracts its mean, this is generally
   not the zero vector in PCA coordinates and is not described as an average
   target;
4. `inverse`: the PCA transform of the sign-negated local raw delta descriptor.

The paired shuffle tests action identity rather than dataset-level motion;
raw-no-action tests a stationary raw descriptor; inverse tests directional
phase/action sign. Every row records the local and donor clip/episode IDs and
its pair-block ID.

## Fixed statistical and practical-effect gate

For the full probe against each of three target controls, evaluate paired
per-clip MSE and cosine. Also compare the full and angle-neutral probes on the
same aligned target:

```text
MSE effect:      control MSE - aligned MSE       (positive favors aligned)
cosine effect:   aligned cosine - control cosine (positive favors aligned)
angle MSE:       angle-neutral MSE - full MSE    (positive favors angle)
angle cosine:    full cosine - neutral cosine    (positive favors angle)
```

First average the two paired clip effects within each of the 32 disjoint donor
pairs. Use exactly 10,000 bootstrap resamples of those 32 pair blocks, seed
20260808. The same sampled pair-index matrix is reused for all eight
preregistered cells, with one-sided Bonferroni cell alpha `0.05/8`. Statistical
positivity alone is insufficient. Every full-probe target-control MSE cell must
have at least 5% relative point gain and a 1% relative simultaneous lower
bound; its cosine cell must have at least `0.05` point gain and `0.01` lower
bound. Full versus angle-neutral must have at least 3% relative MSE point gain
and a 1% relative lower bound, plus cosine point gain of `0.02` and lower bound
of `0.005`. Relative MSE values use the comparator's mean MSE as denominator;
every bootstrap replicate recomputes both its paired effect numerator and
comparator-mean denominator before the relative lower quantile is taken. Phase
0 passes only if all eight cells meet their fixed practical and
simultaneous-bound thresholds. No subset, metric, threshold, checkpoint, or
seed may rescue a failure.

On pass, the next experiment is a matched generator ablation using the frozen
probe as a training-only action/spectral consistency loss on the denoiser's
predicted-clean future latent. The probe is removed at inference, directly
avoiding the unavailable-clean-feature problem. On failure, do not add a CSIP
branch to VPM; investigate other causal objectives such as cycle consistency
or learned action-state prediction under a new preregistration.
