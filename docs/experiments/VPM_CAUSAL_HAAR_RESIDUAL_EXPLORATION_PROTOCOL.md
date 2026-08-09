# Frozen VPM causal Haar-residual exploration

Date frozen: 2026-08-09

Status: prospective code-and-protocol precommit; post-selection exploratory
analysis on previously inspected ABC-train development rows only. No reserve,
validation, or protected-test row may be opened.

## Question and claim boundary

At the native VPM one-step call, is an orthogonal coarse or temporal-change
component of the exact video-latent flow residual easier to predict causally
*and* more useful than an ordinary full-residual correction with the same
input and head capacity?

This is a falsification experiment for a time-frequency/coarse-residual
mechanism. Predictability alone is insufficient. A component is called
special only if it improves either held-out quality or frozen learning-curve
sample efficiency beyond `FULL_DIRECT`, with aligned-versus-shuffled
attribution. Because rows 416--479 were inspected by the causal-compressibility
ladder, every outcome is explicitly post-selection exploratory and cannot
support a paper claim. LACWM uses `sigma_video=1` for noise and
`sigma_video=0` for clean video.

## Exact, view-isolated orthogonal decomposition

The exact VPM residual at the initial one-step state is

\[
r = v^* - v_{\mathrm{VPM}}(S), \qquad v^*=z_1-z_0,
\]

where `S` contains only inference-visible observed history, actions, time, and
initial noise. For the pinned Wan geometry, the full video latent is
`[B,16,4,24,120]`, the history occupies the first two latent frames, and the
future residual is `[B,16,2,24,120]`. The width is three camera views of 40
latent columns each. Every transform is computed separately in the three
`[B,16,2,24,40]` view tensors; no operation crosses the artificial view
boundary.

For each channel and each aligned `2 x 2 x 2` future-latent block, let `S` be
the orthogonal projection onto the spatial Haar `LL` subspace and `T` the
orthogonal projection onto the temporal Haar lowpass subspace:

\[
(Sr)_{t,2i+a,2j+b}=\frac14\sum_{a',b'\in\{0,1\}}
r_{t,2i+a',2j+b'},
\]

\[
(Tx)_{2k+q,y,x}=\frac12\sum_{q'\in\{0,1\}}x_{2k+q',y,x}.
\]

The three registered bands are

| Band | Projection | Orthonormal 3-D Haar coefficient group |
|---|---|---|
| `ST_COARSE` | `T S r` | temporal-low, spatial-`LL` (`LLL`) |
| `TEMPORAL_CHANGE` | `(I-T) S r` | temporal-high, spatial-`LL` (`HLL`) |
| `SPATIOTEMPORAL_DETAIL` | `(I-S) r` | the other six spatial-detail groups |

Thus

\[
r=B_{\rm coarse}+B_{\rm change}+B_{\rm detail}
\]

and all three terms are pairwise orthogonal under the latent Euclidean inner
product. History corrections are exactly zero. Registration records the
maximum reconstruction error, pairwise inner products normalized by residual
energy, per-band energy fractions, latent shape, future boundary, and per-view
shape. The run is invalid unless reconstruction error is at most `2e-6` in
FP32, relative reconstruction energy is at most `1e-12`, and every normalized
pairwise inner product has magnitude at most `2e-6` on every checked fit,
calibration, and development batch.

## Frozen train-only access contract

The immutable 512-row ABC training manifest is partitioned by canonical
`auxiliary_index`:

| Role | Indices | Use |
|---|---:|---|
| historical exclusion | 0--127 | never opened |
| nested fit | 128--383 | ordinary RGB/action target construction only |
| direct-only calibration | 384--415 | select one shared capacity/ridge pair |
| prior-inspected exploratory development | 416--479 | sole reported outcomes |
| fresh reserve | 480--510 | never opened |
| historical constructor probe | 511 | excluded and never opened |

The nested fit doses are the first `32`, `64`, `128`, and `256` rows of the
fit interval. They are frozen by row identity, not chosen from outcomes.
Future-validity retry/substitution is disabled (`enabled=false`,
`max_retries=0`). Each process validates only its two in-role endpoint rows,
and an index-auditing wrapper rejects and inventories every row access.

Noise is deterministically keyed by `(seed, clip_index, stream)` and is
disjoint from every prior ladder/frontier seed:

- fit: `20260910, 20260911`;
- calibration: `20260912, 20260913`;
- exploratory development: `20260914, 20260915, 20260916, 20260917`;
- frozen projection: `20260918`;
- episode bootstrap: `20260919`.

The V-JEPA target array and all teacher features are prohibited. The parent
VPM snapshot, manifests, resolved config, RGB/actions cache identity, and
runtime receipt are inherited and revalidated from the completed immutable
causal-compressibility registration. The snapshot bytes are rehashed. No W&B
write is allowed.

## Equal-input, equal-capacity heads and controls

An external hook captures the inference-visible shared Wan trunk state before
the native video head during one feature-off VPM call. A single frozen nested
Gaussian projection maps each token to `{64,256,1024}` candidate channels.
All heads at a selected rung are token-local affine ridge heads with identical
input rows, output shape, parameter count, token subsampling, and numerical
precision. The ridge grid is `{1e-4,1e-3,1e-2,1e-1,1}`. Capacity and ridge are
selected once using only the full-dose `FULL_DIRECT_ALIGNED` corrected-
velocity MSE on rows 384--415; ties prefer smaller capacity and then larger
ridge. That locked pair is used for every band, control, and fit dose.

For every dose, fit these nine heads:

- exact-zero `ZERO`;
- `FULL_DIRECT_ALIGNED` and `FULL_DIRECT_SHUFFLED`;
- aligned and episode-shuffled targets for each of `ST_COARSE`,
  `TEMPORAL_CHANGE`, and `SPATIOTEMPORAL_DETAIL`.

The shuffled donor is the other member of each fixed adjacent two-episode
fit batch. At inference, a band prediction is projected back into its named
orthogonal subspace before addition to VPM velocity. This prevents off-band
leakage. `FULL_DIRECT` is not projected. All arms use one shared Wan call and
zero teacher/feature calls. `ZERO` must reproduce VPM-off bit-for-bit.

## Frozen metrics and exploratory decision

For each arm, dose, episode, and development noise seed, preserve:

- band-target R2, cosine, target energy, prediction energy, and off-band
  leakage before/after the enforced projection;
- corrected future-velocity MSE, future-latent NMSE, decoded RGB MSE in
  `[0,1]`, and decoded temporal-difference MSE including the history boundary;
- aligned-versus-off, aligned-versus-matched-shuffled, and aligned-versus-
  `FULL_DIRECT_ALIGNED` paired effects and favorable episode fractions;
- synchronized adapter and full one-step endpoint latency;
- complete tensor hashes, row access counts, Wan/teacher calls, and Haar
  reconstruction/orthogonality receipts.

Confidence intervals use 10,000 episode-clustered paired bootstrap samples;
all four development noises remain grouped within an episode. Learning-curve
sample efficiency is the trapezoidal area over normalized `log2(dose)` for
per-episode relative decoded and temporal improvement over VPM-off. Its paired
bootstrap compares each band curve with the `FULL_DIRECT` curve.

The result is `EXPLORATORY_HAAR_SPECIAL` only if at least one band satisfies
all of the following:

1. at some dose, aligned beats its matched shuffled control by at least 3% on
   corrected velocity, decoded MSE, and temporal MSE, with paired 95% lower
   bounds above zero and at least 60% favorable episodes for both decoded
   metrics;
2. aligned beats VPM-off by at least 3% on decoded and temporal MSE, with
   positive lower bounds, while latent NMSE has a nonnegative point effect and
   lower bound above -1%;
3. either (a) it beats matched-dose `FULL_DIRECT_ALIGNED` by at least 1% on
   decoded and temporal MSE with positive lower bounds, or (b) its decoded and
   temporal learning-curve areas both beat `FULL_DIRECT` with positive lower
   bounds while its full-dose decoded and temporal effects versus full direct
   are each no worse than -1%; and
4. mean held-out band R2 and cosine are positive, one-Wan/zero-feature serving
   holds, post-projection leakage is below `2e-6` relative energy, Haar checks
   pass, and `ZERO == VPM_OFF` bit-exactly.

Otherwise the result is `EXPLORATORY_NO_HAAR_ADVANTAGE`. High band R2 without
condition 3 is explicitly reported as ordinary residual predictability, not a
time-frequency advantage. Even a pass authorizes only a prospectively
replicated nonlinear study on untouched data; it is not evidence of better
video generation, real-time DAgger, or generalization.
