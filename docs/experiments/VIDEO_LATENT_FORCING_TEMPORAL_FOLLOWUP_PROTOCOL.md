# Temporal Video Latent Forcing follow-up protocol

Date preregistered: 2026-08-07

Status: exploratory follow-up frozen before any new checkpoint, autonomous
sample, candidate metric, or candidate comparison exists

## Question

Can an inference-generated auxiliary state improve few-step video generation
if it represents future **change** rather than repeatedly encoding the static
scene, and if it is trained on its own rollout distribution rather than only
true forward-noised states?

The completed absolute prefix-causal V-JEPA screen is the immutable incumbent.
Its best autonomous result was at one NFE; additional Euler calls worsened both
ordinary and temporal metrics. Action shuffling barely changed its prediction.
This follow-up separates three possible causes:

1. the denoiser may be inaccurate even on true forward-noised states;
2. the denoiser may be accurate on-distribution but inconsistent on its own
   rollout states; or
3. the absolute target and loss may reward static scene recovery much more
   than action-conditioned motion.

No clean future RGB, semantic target, optical flow, or V-JEPA teacher output is
available to a deployable sampler. A clean target may only construct an
explicitly labelled training-distribution diagnostic input or a supervised
training loss.

## Existing evidence and fixed inputs

The producer checkpoint is source commit
`c11487f6e83908687f27026ce2ac2e7d8d41461c`, update 5,000, SHA-256
`f7586f23030a489fc6a673ea3bb6c6cfecccdbe5269c62f6de697d1dc4f9f9cc`.
The immutable train and validation targets have SHA-256 values
`547c4579cf978ac2b9527cb038693259af678a2b07268ab1434706dc128051c4`
and `ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034`.
The 64,000-train/890-validation DROID split, protected-test exclusion, one-view
RGB, 16-action input, target shape `[48,8,8,14]`, model geometry, and clean-time
clock remain unchanged.

A target-only audit performed before choosing the candidate arms found that
the validation target has mean square `0.983072`; `88.4731%` of its energy is
in the per-clip temporal mean and only `11.5269%` in within-clip temporal
variation. Adjacent differences contain `17.0629%` as much energy per temporal
slot as the absolute target. Although this audit contains no prediction, it
influenced the decision to test delta packing. Validation is therefore a
**development/selection split**, not an untouched test split. The same audit
will be recomputed and reported on training targets only; no confirmatory claim
may rely on the validation audit. Exactly one selected arm/NFE pair may later
be evaluated once on the protected test split as described in D3.

The design is motivated by trajectory-consistency and on-policy video work,
including [Consistency Models](https://arxiv.org/abs/2303.01469),
[Consistency Trajectory Models](https://arxiv.org/abs/2310.02279),
[Self-Forcing](https://arxiv.org/abs/2506.08009), and
[Diffusion Forcing](https://arxiv.org/abs/2407.01392). The motion-state
alternative is supported by [LaMD](https://arxiv.org/abs/2304.11603) and
[MoVideo](https://arxiv.org/abs/2311.11325). These references motivate the
candidate mechanisms but do not substitute for the frozen DROID controls.

## Stage D0: denoiser-versus-trajectory diagnosis

Use every validation clip and the same clip-addressed Gaussian auxiliary noise
for all comparisons. A training-distribution diagnostic state at clean time
`t` is

\[
z_t=t s+(1-t)\epsilon.
\]

This state is labelled **training-distribution**, not "on-manifold": it follows
the training corruption law, but may still be statistically atypical. Query
the frozen model and report clean-state NMSE, token cosine,
temporal-difference NMSE/cosine, and velocity MSE. Separately roll out from
`epsilon`, never using `s`, and query the same model at matched pre-update
states. The frozen primary comparison is the state before call 2 on the exact
uniform-Euler four-call trajectory, namely `t=.25`; call 1 at `t=0` is shared.
Also report every pre-call node from the exact original uniform-Euler call
budgets `C in {2,4,8}`. At `t=0`, the training-distribution and autonomous
state and prediction must be bit-identical. A supplemental dense Euler trace
uses the registered grid
`{0,.025,.05,.10,.20,.35,.50,.65,.80,.90,.95}` but cannot determine the
primary diagnosis. Repeat autonomous paths with the true actions and three
fixed, episode-disjoint action permutations with manifest offsets
`{1,17,101}` modulo 890.

From identical noise, compare solvers at equal **actual transformer-call**
budgets. Direct-x is exactly one call at `t=0` and has no multi-call variant.
Euler uses `C in {1,2,4,8}` intervals and call times `t_i=i/C`, followed by an
Euler update to `(i+1)/C`. Explicit midpoint uses even call budgets
`C in {2,4,8}` and `M=C/2` intervals: for each interval of width `h=1/M`, call
at `t`, Euler-predict the state at `t+h/2`, call there, and update the interval
start with that midpoint velocity to `t+h`. It has no one-call entry. The
low-noise-dense Euler schedule uses the same actual call budgets and boundaries
`t_i=1-(1-i/C)^2`. Any Heun result is supplemental and must obey the same
actual-call accounting. Record nominal intervals and actual calls separately;
exclude duplicate one-call schedule labels from statistical comparisons. The
training-distribution arm is nondeployable and diagnostic only. Its target hash
must never occur in an autonomous sampler-input hash chain. Autonomous calls
must use a separate target-free call graph, preserve hashes of all fixed inputs,
and reject any changed state/noise/context/action hash unexpectedly. Signature
inspection is an additional guard, not proof of no leakage. Applicable metrics
must be finite; inapplicable fields serialize as JSON `null`, never NaN.

Classify rollout drift as the primary failure only from the preregistered
uniform-Euler-four primary cell at `t=.25`: training-distribution temporal NMSE
must be at most `0.50`, and autonomous temporal NMSE must be at least 25% worse
with the paired 95% bootstrap confidence-interval lower bound also at least
25%. Otherwise report the denoiser and trajectory errors separately without
declaring that D0 alone has identified a unique primary cause. D0 distinguishes
training-distribution denoising from failure along a specified numerical
trajectory; it cannot by itself prove that the endpoint parameterization or
representation is causal.

## Stage D1: train-only temporal packing statistics

Fit per-channel anchor and delta mean/standard deviation using only the 64,000
training targets stored as float16. For `X` with shape `[N,48,8,H,W]`, anchor
statistics reduce the `N,H,W` axes of `X[:,:,0]`; delta statistics reduce the
`N,7,H,W` axes of `X[:,:,1:]-X[:,:,:-1]`. Accumulate sums and squared sums in
float64 and use population variance `E[x^2]-E[x]^2`, clamped below at zero,
with `std=max(sqrt(var),1e-6)`. Publish the resulting double-precision JSON
values, counts, source hashes, and an exclusive completion record. Encoding
and decoding use float32. Validation and protected-test targets must not affect
these values.

Define the invertible delta pack `y` from absolute semantic tokens `s`:

\[
y_0=(s_0-\mu_a)/\sigma_a,
\qquad
y_j=((s_j-s_{j-1})-\mu_d)/\sigma_d,\quad j>0.
\]

Decode with the inverse affine transforms followed by a cumulative sum. Tests
must require tolerance-stable float32 round-trip maximum absolute error at most
`2e-5` from the declared float16 source storage. Unlike temporal de-meaning,
this transform retains the entire clean target while representing static
content once instead of eight times.

## Stage D2: controlled representation and trajectory DOE

Run all primary arms from scratch with seed 1234, identical initialization,
data order, global batch 256, optimizer, EMA, 200-update numerical calibration,
and 5,000-update budget:

| Arm | Target | Additional supervision |
|---|---|---|
| `ABS` | absolute V-JEPA | incumbent flow loss |
| `ABS-T` | absolute V-JEPA | normalized temporal-velocity loss, weight 1 |
| `DELTA` | normalized invertible delta pack | incumbent flow loss |
| `DELTA-T` | normalized invertible delta pack | normalized temporal-velocity loss, weight 1 |
| `DELTA-R` | normalized invertible delta pack | 50% stopped one-step self-roll-in plus the same clean target |

For all arms, temporal-velocity loss is defined only after decoding predicted
and target representation-space velocities into absolute semantic-space
velocities. Take their temporal first differences, normalize each original
semantic channel with the train delta standard deviation, and then compute
MSE. This avoids accidentally comparing second differences for delta-packed
arms.

For `DELTA-R`, the ordinary final supervised clock `t_1` is drawn by the exact
incumbent auxiliary-clock law, with the same clean target, Gaussian noise, and
data order as `DELTA`. A counter/hash RNG keyed by `(seed, update,
global_sample_id)`—without consuming the baseline RNG stream—draws a Bernoulli
mask with probability `.5` and `u ~ Uniform[0,1]`; selected samples set
`t_0=u*t_1`. Construct the true forward state `z(t_0)`, make one no-gradient
prediction, detach it, and apply

\[
z_R(t_1)=z(t_0)+(t_1-t_0)
          \frac{\hat{x}(z(t_0),t_0)-z(t_0)}
               {\max(1-t_0,0.05)}.
\]

Replace the true `z(t_1)` by `z_R(t_1)` only for selected samples. Unselected
samples retain the exact true `z(t_1)`. One no-gradient batched call operates
only on the selected subset when that subset is nonempty; the expected selected
fraction is one half, not one call per selected example. One gradient-bearing
forward/loss on the full mixed batch at `t_1` is supervised against the same
clean packed target. The global sample ID is
`(update-1)*global_batch_size + rank*local_optimizer_batch_size +
microstep*microbatch_size + row_within_microbatch`. Report selected counts,
gradient-bearing and total batched transformer calls, wall time, and GPU-hours.

Action shuffling is evaluated with the three frozen episode-disjoint manifest
permutations above and labelled only **factual action attribution**. It is not
counterfactual-causal evidence because DROID supplies no ground-truth future
for a changed action, and it is not a promotion criterion. An action-swap
training loss is not a primary arm and cannot manufacture apparent action
sensitivity.

## Stage D3: generated-only gate

Evaluate the final update-5,000 EMA weights of every primary arm on all 890
development/validation clips using NFE
`{1,2,4,8,12,20,25}` and the previous controls: autonomous, donor-target,
context-shuffled, history-shuffled, actions-shuffled, zero, and oracle-clean.
Delta predictions are decoded to absolute semantic space before applying the
existing semantic gate; packed-space errors are additional diagnostics.
The frozen deployable sampler is uniform-clean-time Euler with float32 state
updates and the same bfloat16 model autocast as the incumbent. D0 solver results
cannot change it. Only NFE `{1,2,4}` is eligible for selection; larger budgets
are descriptive.

An arm is eligible only at NFE at most 4 and must satisfy all prior requirements:

- autonomous NMSE at most `0.50`, cosine at least `0.70`, and temporal NMSE at
  most `0.50`;
- at least 5% paired improvement in ordinary and temporal NMSE over the `ABS`
  incumbent at the same NFE, with the simultaneous confidence lower bound also
  at least 5%;
- at least 5% paired improvement in ordinary and temporal NMSE over both
  donor-target and context-shuffled, with the simultaneous confidence lower
  bound also at least 5%; and
- positive paired cosine advantages over both controls.

`ABS` is a non-promotable reference, leaving four candidate arms and three
selectable NFEs, or 12 candidate cells. Development gates use 10,000 paired
clip-level bootstrap resamples and seed 20260807. The same vector of 890 clip
indices is resampled jointly across every metric, cell, and control. For a
lower-is-better metric, relative improvement is the ratio of sample means
`(mean(control)-mean(candidate))/mean(control)`; rollout degradation in D0 is
`(mean(autonomous)-mean(training_distribution)) /
mean(training_distribution)`. Cosine advantage is the difference of sample
means. Tests use a one-sided percentile lower bound. To account for selecting
among 12 cells, each candidate cell uses confidence `1-.05/12 = .995833...`.
This is a 12-cell Bonferroni screen of composite intersection-union gates; it is
not described as simultaneous coverage of every component interval. If
multiple cells pass, select deterministically by: smallest NFE, lowest temporal
NMSE, lowest ordinary NMSE, then arm order `ABS-T`, `DELTA`, `DELTA-T`,
`DELTA-R`.
No alternative tie-break or additional arm may be introduced after metrics are
visible.

After selection of exactly one arm/NFE pair, create the protected-test semantic
cache with the already frozen PCA/teacher pipeline. Evaluate the incumbent and
the selected pair exactly once on the protected clips with the same
clip-addressed noise and nominal 95% paired bootstrap interval. Do not tune,
rerun with another seed, or inspect other candidate arms on this lockbox. A
claim and D4 require the selected pair, at its selected NFE, to retain NMSE at
most `.50`, cosine at least `.70`, temporal NMSE at most `.50`, point
improvements of at least 5% in ordinary and temporal NMSE over the incumbent,
and nominal paired 95% lower bounds above zero for both improvements. Before
any metric is published or read, an operationally failed attempt may be retried
only with byte-identical source/config/data/checkpoint/noise identities; the
failed output remains recorded, and this is not a new scientific look. After
any lockbox metric is visible, no retry is eligible for inference. If no
development cell passes, do not open the lockbox.

## Stage D4: video experiment, conditional on D3

Do not enable video training if no arm passes D3 including lockbox confirmation.
If one passes, run fresh
parameter-matched video-only, generated-auxiliary, auxiliary-shuffled, and
oracle-clean controls. The training video branch must receive stopped
self-generated or calibrated forward-corrupted auxiliary states, not only
clean features. A positive video result requires better generated-only
temporal quality at equal total NFE, no regression in spatial/perceptual
quality, and a lower-NFE equivalent-quality point with end-to-end auxiliary
generation included in latency.

Semantic NMSE is an unvalidated proxy for downstream video utility. If every
semantic arm fails, a separately preregistered, cheap, generated-only video
probe may test whether that proxy rejected a useful conditioner. Such a probe
cannot support a video-quality claim or replace the controlled D4 comparison.

## If all semantic arms fail

Freeze the result rather than tuning against validation. The next independently
preregistered family is an observed-content anchor plus explicitly generative
motion state: video-VAE temporal residual, optical flow/occlusion, or depth
change. Frequency information remains eligible as a differentiable training
regularizer or self-generated initialization correction, but not as an oracle
future condition.
