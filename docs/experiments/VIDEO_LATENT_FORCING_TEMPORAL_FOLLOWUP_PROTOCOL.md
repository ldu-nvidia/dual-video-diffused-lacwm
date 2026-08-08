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
explicitly labelled on-manifold diagnostic input or a supervised training
loss.

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
slot as the absolute target. This audit contains no prediction and cannot
select a model.

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
for all comparisons. At clean times
`{0,.025,.05,.10,.20,.35,.50,.65,.80,.90,.95}`, construct the diagnostic
on-manifold state

\[
z_t=t s+(1-t)\epsilon.
\]

Query the frozen model once and report clean-state NMSE, token cosine,
temporal-difference NMSE/cosine, and velocity MSE. Separately roll out from
`epsilon`, never using `s`, and query the same model at matched times. Repeat
the autonomous path with destination actions and episode-disjoint shuffled
actions.

From identical noise, compare direct clean prediction, uniform Euler, midpoint
second-order integration, and a low-noise-dense schedule at nominal NFE
`{1,2,4,8}`. Record actual transformer calls separately from nominal steps.
The on-manifold arm is nondeployable and diagnostic only. Its target hash must
never occur in any autonomous sampler-input hash chain.

Classify rollout drift as the primary failure only if an on-manifold state at
`t <= .50` reaches temporal NMSE at most `0.50` while its paired autonomous
state is at least 25% worse with a positive 95% bootstrap lower bound. If no
such cell exists, classify endpoint/representation learning as the primary
failure. Both may coexist and all raw cells remain reportable.

## Stage D1: train-only temporal packing statistics

Fit per-channel anchor and delta mean/standard deviation using only the 64,000
training targets. Use deterministic float64 accumulation and publish counts,
moments, source hashes, and an exclusive completion record. Validation and
protected-test targets must not affect these values.

Define the invertible delta pack `y` from absolute semantic tokens `s`:

\[
y_0=(s_0-\mu_a)/\sigma_a,
\qquad
y_j=((s_j-s_{j-1})-\mu_d)/\sigma_d,\quad j>0.
\]

Decode with the inverse affine transforms followed by a cumulative sum. Tests
must require bit-stable round-trip accuracy within the declared storage
precision. Unlike temporal de-meaning, this transform retains the entire clean
target while representing static content once instead of eight times.

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

For `DELTA-R`, the first prediction is stop-gradient. Its Euler state at a
later randomly sampled time is fed back to the model, and the second prediction
is supervised against the same clean packed target. This deliberately trains
on model-induced errors without making a clean target an inference input.

An action-swap margin may be evaluated as a diagnostic arm, but it is not
eligible for promotion because DROID supplies no ground-truth counterfactual
future for the shuffled action. It cannot be used to manufacture apparent
action sensitivity.

## Stage D3: generated-only gate

Evaluate every primary arm on all 890 validation clips using NFE
`{1,2,4,8,12,20,25}` and the previous controls: autonomous, donor-target,
context-shuffled, history-shuffled, actions-shuffled, zero, and oracle-clean.
Delta predictions are decoded to absolute semantic space before applying the
existing semantic gate; packed-space errors are additional diagnostics.

An arm is eligible only at NFE at most 4 and must satisfy all prior requirements:

- autonomous NMSE at most `0.50`, cosine at least `0.70`, and temporal NMSE at
  most `0.50`;
- at least 5% paired improvement in ordinary and temporal NMSE over both
  donor-target and context-shuffled, with positive 95% confidence lower bounds;
  and
- positive paired cosine advantages over both controls.

This follow-up additionally requires actions-shuffled temporal NMSE to be at
least 5% worse than autonomous with a positive paired 95% confidence lower
bound. The bootstrap uses 10,000 clip-level resamples and seed 20260807.

## Stage D4: video experiment, conditional on D3

Do not enable video training if no arm passes D3. If one passes, run fresh
parameter-matched video-only, generated-auxiliary, auxiliary-shuffled, and
oracle-clean controls. The training video branch must receive stopped
self-generated or calibrated forward-corrupted auxiliary states, not only
clean features. A positive video result requires better generated-only
temporal quality at equal total NFE, no regression in spatial/perceptual
quality, and a lower-NFE equivalent-quality point with end-to-end auxiliary
generation included in latency.

## If all semantic arms fail

Freeze the result rather than tuning against validation. The next independently
preregistered family is an observed-content anchor plus explicitly generative
motion state: video-VAE temporal residual, optical flow/occlusion, or depth
change. Frequency information remains eligible as a differentiable training
regularizer or self-generated initialization correction, but not as an oracle
future condition.
