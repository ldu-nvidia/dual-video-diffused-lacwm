# Frozen VPM causal learnable-wavelet qualification

Date frozen: 2026-08-09

Status: prospective code-and-protocol precommit; post-selection exploratory
analysis on previously inspected ABC-train development rows only. No fresh
reserve, validation, or protected-test row may be opened.

## Question and claim boundary

Does a spatial wavelet basis learned without any VPM target make an equal-rank
component of the exact VPM one-step residual materially more causally
predictable and useful than the same component under fixed Haar, or than an
equal-capacity full-residual head?

This is the smallest qualification probe that can falsify the hypothesis that
data adaptation, rather than merely subspace restriction, changes the residual
signal seen in the completed fixed-Haar exploration. It is **not** a
Frequency-Forcing reproduction. The primary source
`literature/arxiv_2604.20902/main.tex` (SHA-256
`34ea62bf37e300012a3d5911c27a80048c81d8c58284d92d982c5983de3ce133`)
jointly optimizes a learnable wavelet/threshold transform and an asynchronous
auxiliary generator stream, and reports generator training for 400 epochs.
Here the basis is instead prefit only by an unsupervised clean-latent
sparsity/admissibility objective, frozen, and used to define targets for small
external causal residual heads. A negative result rejects only this narrow
frozen-basis prerequisite; it does not reject the paper's joint generator
training. Rows 416--479 have already been inspected, so no outcome can support
a paper claim. LACWM uses `sigma_video=1` for noise and `sigma_video=0` for
clean video. `dual_diffusion.enabled` remains false.

## Train-only, target-blind spatial QMF prefit

The frozen VAE encodes each ordinary clean training video in rows 128--383 to
`[16,4,24,120]`. Exactly the two clean future latent frames and all 16 channels,
`[16,2,24,120]`, enter the basis objective; the two history tokens are excluded
so static observed content cannot dominate a basis intended for the future
residual marginal. The artificial width is split into three independent
`[16,2,24,40]` camera views before any transform. The optimizer never receives
VPM velocities, residuals, hidden states, actions, metrics, or development
material.

The learned one-dimensional low-pass filter has nominal length eight. It is
parameterized by a four-rotation two-channel paraunitary polyphase lattice,
with three free angles and the fourth angle constrained so that their sum is
`-pi/4`. This hard-enforces unit norm, the low-pass sum `sqrt(2)`, zero DC in
the QMF high-pass

\[
g[k]=(-1)^k h[7-k],
\]

and zero even-shift autocorrelations. Separable outer products form the four
one-level spatial packet bands. Analysis uses a fixed periodic boundary and
phase separately within each 40-column view. Learned and Haar transforms use
the same matrix implementation, nominal length-eight window, phase, boundary,
one-level packet depth, and rank. Frozen Haar is represented as
`[1/sqrt(2),1/sqrt(2),0,0,0,0,0,0]`.

There are no learnable hard-threshold gates in this qualification: retention
is fixed to 100% in every terminal band. This deliberate departure from the
paper prevents sparsity from being manufactured by all-zero gates and
preserves equal rank and exact reconstruction. The fixed objective is the
paper-style constrained coefficient objective

\[
L=L_{\rm sparse}+100L_{\rm sum}+100L_{\rm hp}+100L_{\rm ortho},
\]

where `L_sparse` is mean absolute one-level packet coefficient magnitude
normalized by the input RMS and the other terms are exactly the sum,
high-pass-zero-DC, and even-shift-orthogonality penalties written in the
source. The three free angles start from a seed-`20261001` Gaussian perturbation
with standard deviation 0.01 around the zero-angle (Haar) lattice. Adam uses
512 updates, batch size 8, learning rate 0.02, no weight decay, and a fixed
deterministic permutation stream. The
final update is frozen; neither a best checkpoint nor a hyperparameter is
selected from residual predictability or held-out outcomes.

Rows 384--415 are used only after freezing to compute clean-latent transform
receipts. They do not select or alter the basis or residual head. Receipts
include filter taps/angles, cosine distance from Haar, QMF sums and even-shift
autocorrelations, full analysis-matrix orthogonality, analysis--synthesis
reconstruction, coefficient/input energy ratio, per-terminal-band energy,
normalized L1, Hoyer sparsity, small-coefficient retention, and top-10%-energy
concentration for learned and Haar. Passing basis qualification requires:

- maximum filter/QMF admissibility residual at most `2e-6`;
- maximum analysis-matrix orthogonality error at most `3e-6`;
- maximum reconstruction error at most `3e-6` and relative reconstruction
  energy at most `1e-11`;
- coefficient/input energy ratio within `1 +/- 1e-5`, 100% gate retention,
  and every terminal band carrying at least 0.5% of clean-latent energy;
- learned-filter absolute cosine with zero-padded Haar below `0.9999`; and
- at least 5% held-out reduction in normalized L1 versus Haar, with an
  episode-paired 95% bootstrap lower bound above zero. Hoyer and energy-
  concentration changes are reported but are not alternative pass routes.

Failure is retained as evidence and cannot be repaired by changing gates or
choosing another optimizer seed after outcomes.

## Frozen orthogonal video-residual partitions

For the exact initial one-step residual

\[
r=(z_1-z_0)-v_{\rm VPM}(S),
\]

only the two future latent frames are eligible. For each basis `b` in
`{HAAR,QMF}`, let `S_b` be the exact orthogonal projection onto the separable
spatial low-pass row space. Let `T` be exact length-two temporal Haar low-pass;
the temporal transform is fixed because only two future latent frames exist
and is never described as learned. The registered partitions are

\[
B^b_{\rm coarse}=TS_b r,\qquad
B^b_{\rm change}=(I-T)S_b r,\qquad
B^b_{\rm detail}=(I-S_b)r.
\]

Each basis therefore has identical ranks: one eighth, one eighth, and three
quarters of future residual coordinates. Natural band energies are not
rescaled or outcome-matched; exact projection/reconstruction is retained and
basis-induced energy allocation is part of the hypothesis. Energy fractions,
target RMS, correction RMS, and scale-normalized R2/cosine are all audited so
an energy shift cannot masquerade as causal predictability. For numerical
comparability, all ridge sufficient statistics
use the same FP32 implementation. Each partition must reconstruct the future
residual, be pairwise orthogonal, leave history exactly zero, and carry between
2% and 96% of residual energy per grouped band. Every spatial operation is
view-isolated.

## Access, seeds, and equal controls

The immutable 512-row ABC training manifest is partitioned as follows:

| Role | Indices | Use |
|---|---:|---|
| historical exclusion | 0--127 | never opened |
| clean basis prefit and residual-head fit | 128--383 | ordinary RGB/action arrays only |
| clean transform receipt | 384--415 | basis audit only; no outcome selection |
| prior-inspected exploratory development | 416--479 | sole reported outcomes |
| fresh reserve | 480--510 | never opened |
| historical constructor probe | 511 | never opened |

Future-validity substitution is disabled. The V-JEPA target array and teacher
features are prohibited. Parent manifests, cache identities, resolved config,
and VPM snapshot SHA-256 are revalidated. No W&B write is allowed.

Frozen randomness is disjoint from previous VPM studies: basis `20261001`,
head-fit noises `20261002,20261003`, development noises
`20261005,20261006,20261007,20261008`, Gaussian projection `20261009`, and
episode bootstrap `20261010`. There is one full fit dose of 256 clips. The
head capacity 1024 and ridge penalty 1.0 are inherited prospectively from the
completed full-dose fixed-Haar calibration; there is no new outcome-based
selection.

All heads consume the same inference-visible shared VPM trunk tokens from one
feature-off Wan call and have identical token-local affine ridge capacity,
token samples, precision, and fit rows. The 15 correction arms are exact zero;
aligned and adjacent-episode-shuffled `FULL_DIRECT`; and aligned/shuffled
coarse, change, and detail targets under each of frozen Haar and frozen learned
QMF. Predicted band corrections are projected into the registered subspace
before addition. `FULL_DIRECT` is unprojected. `ZERO` must reproduce VPM-off
bit-for-bit. All arms use one shared Wan call and zero feature/teacher calls.

## Metrics and fixed decision

Rows 416--479 are evaluated under all four development noises. Preserve
corrected future-velocity MSE, future-latent NMSE, decoded RGB MSE in `[0,1]`,
decoded temporal-difference MSE including the history boundary, band R2 and
cosine, energy, leakage, tensor hashes, calls, access counts, and synchronized
latency. Confidence intervals use 10,000 episode-clustered paired bootstrap
samples.

For population-shrinkage diagnosis, preregistered VPM-off error quartiles are
formed from per-episode VPM-off error, and aligned/shuffled effects are reported
within each quartile for velocity, latent, decoded, and temporal metrics. These
heterogeneity summaries cannot choose a basis, band, head, or pass.

The decision is `EXPLORATORY_LEARNED_QMF_SPECIAL` only if the basis passes all
qualification checks and at least one learned-QMF band satisfies all of:

1. aligned beats its matched shuffled arm by at least 3% on velocity, decoded,
   and temporal MSE, with positive paired 95% lower bounds and at least 60%
   favorable episodes for decoded and temporal metrics;
2. aligned beats VPM-off by at least 3% on decoded and temporal MSE with
   positive lower bounds, while latent NMSE has a nonnegative point effect and
   lower bound above -1%;
3. aligned beats both its rank-matched frozen-Haar band and
   `FULL_DIRECT_ALIGNED` by at least 1% on decoded and temporal MSE, with
   positive paired lower bounds for all four comparisons; and
4. mean held-out band R2 and cosine are positive, post-projection leakage is at
   most `2e-6`, the grouped-band energy/reconstruction contracts pass, one-Wan
   zero-feature serving holds, and `ZERO == VPM_OFF` bit-exactly.

If the basis qualifies but no band passes, the decision is
`EXPLORATORY_BASIS_ADAPTS_BUT_NO_CAUSAL_ADVANTAGE`; if the basis itself fails,
it is `EXPLORATORY_NO_LEARNED_QMF_QUALIFICATION`. No result authorizes reserve,
validation, test, full generator training, a video-quality claim, or a
real-time DAgger claim. A pass only motivates a separate prospective joint
auxiliary-stream experiment on untouched data.
