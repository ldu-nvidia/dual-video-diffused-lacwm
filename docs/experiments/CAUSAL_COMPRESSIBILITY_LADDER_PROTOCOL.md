# Frozen causal-compressibility ladder

Date: 2026-08-08

Status: prospective train-only preflight; code and protocol must be committed
before outcomes are opened; no protected test or validation split is permitted

## Question

The disjoint high-noise teacher probe showed that clean target-video V-JEPA
changes the frozen J1 velocity in a highly favorable direction at the sole
`sigma_video=1` update. That is useful only if the correction can be predicted
from information available at inference. This experiment asks whether the
privileged residual

\[
r_T = v_T(z_1,H,A,u^*)-v_S(z_1,H,A,\varnothing)
\]

is more causally compressible than an ordinary direct residual or a shuffled
teacher residual. LACWM uses `sigma=1` for Gaussian noise and `sigma=0` for
clean data.

This is a frozen-backbone, closed-form probe. It is not a continuation of J1,
and it cannot establish a paper claim from train-only development data.

## Exact information seam

An external forward pre-hook captures the first tensor passed to
`model.forward_model.transformer.head`. `WanForwardModel` already hooks this
same tensor to drive its auxiliary head, so this seam is the shared Wan trunk
state after all transformer blocks and before the native video head. The hook
is active only for the feature-off J1 call. Neither clean V-JEPA nor the
teacher prediction may enter the captured state.

The shared token order is retained exactly. Each token predicts one native Wan
output patch, and the adapter uses Wan's patch order
`[F,H,W,p_f,p_h,p_w,C]`. A round-trip test must prove that probe patchification
and Wan-style unpatchification are exact. History-token corrections are forced
to zero before the final history reference is clamped.

The causal feature for a token is a frozen nested Gaussian projection of the
shared trunk token. Projection seed is `20260828`. The three capacity rungs are
`64`, `256`, and `1024` projected channels; every target arm at a given rung
has exactly the same affine patch head and parameter count. Projection
features are centered and diagonally standardized using optimization rows
only.

## Frozen train-only partitions

The immutable 512-row ABC training manifest is partitioned by its canonical
`auxiliary_index`:

| Role | Indices | Episodes | Future feature allowed? |
|---|---:|---:|---|
| prior teacher eligibility, excluded | 0--127 | 128 | historical only |
| optimization | 128--383 | 256 | yes, teacher-target construction only |
| numerical calibration | 384--415 | 32 | no |
| development outcome | 416--479 | 64 | no |
| untouched train reserve | 480--511 | 32 | no access |

Every selected row must be train-scoped, and episode and clip identities must
be mutually disjoint across all five partitions. Any substitution by dataset
retry logic is fatal.

Gaussian noise is also prospective and disjoint:

- optimization seeds: `20260820`, `20260821`;
- calibration seeds: `20260822`, `20260823`;
- development seeds: `20260824`, `20260825`, `20260826`, `20260827`.

Noise is deterministically keyed by `(seed, clip_index, stream)`. Thus no
episode/noise pair used for fitting is reused for model selection or outcome
reporting. Optimization and calibration use 256 deterministic future-token
positions per `(clip, noise seed)`; development scores every future token.

## Equal-capacity targets

With the J1 backbone and native head frozen, fit four affine patch heads from
the identical feature rows:

| Arm | Optimization target |
|---|---|
| `ZERO` | exact zero |
| `DIRECT` | `v* - v_S` |
| `PFD_ALIGNED` | `stopgrad(v_T(aligned u*) - v_S)` |
| `PFD_SHUFFLED` | `stopgrad(v_T(episode-shuffled u*) - v_S)` |

The shuffled donor is the next optimization item in each fixed two-item
batch; all donors must be from different episodes. A teacher call is forbidden
outside optimization indices 128--383.

Each head is fitted by centered ridge regression. Candidate ridge penalties
are `1e-4`, `1e-3`, `1e-2`, `1e-1`, and `1`. The single shared
capacity/penalty pair is selected using only `DIRECT` corrected-velocity MSE
on the calibration partition. Ties prefer smaller capacity and then larger
penalty. The locked pair is then used for all four arms. This prevents the
privileged target from receiving more capacity or outcome-aware tuning.

`ZERO` must solve to exact zero and reproduce J1-off bit-for-bit after the
history clamp. Any failure of that no-op invariant invalidates the run.

## Fresh feature-free endpoint processes

Fitting, J1 development evaluation, and VPM development evaluation are three
different Python processes. The two endpoint processes instantiate
`ABCVideoResidualAnchorDataset`, whose implementation can open only the pinned
RGB and action arrays. An `np.load` guard fails immediately if the V-JEPA
target path is opened. The endpoint input graph records every NumPy array
opened and must contain exactly RGB and actions.

The J1 endpoint receives only shared off tokens and its sealed affine head. It
makes one J1 Wan call at the NFE-2 schedule's sole `sigma_video=1` video node,
with state and clock conditioning hard-off. It does not execute the auxiliary
prefix, because that prefix cannot affect the hard-off shared video trunk.
The corrected velocity is integrated directly from sigma 1 to sigma 0.

The actual frozen VPM update-1,000 snapshot is freshly evaluated at NFE 1 on
the identical 64 clips and four noise seeds. J1 and VPM must hash-identically
match their initial video noise for every paired row. Stored historical VPM
aggregates are context only and are not accepted as the frontier comparison.

Normal clean RGB is evaluator-owned only: it constructs the flow target and
quality metrics after the model input has been assembled. Future RGB, direct
residuals, V-JEPA, the target array, and teacher outputs never enter an
adapter.

## Metrics and inference accounting

For each arm and paired `(clip, noise seed)`, report:

- direct-residual explained energy (`R2` against the exact-zero correction)
  and cosine;
- corrected velocity MSE;
- one-step future-latent NMSE;
- decoded MSE and decoded temporal-difference MSE in `[0,1]`;
- Wan calls, teacher calls, auxiliary-array opens, and adapter latency.

All confidence intervals use 10,000 paired bootstrap resamples, seed
`20260829`, clustered by episode so all four noise draws stay together.
Positive relative improvement means the named candidate has lower error.

## Frozen decision

The privileged correction is `CAUSALLY_COMPRESSIBLE_ADVANCE` only if the
locked `PFD_ALIGNED` arm satisfies all of the following on development:

1. corrected velocity MSE beats `DIRECT` and `PFD_SHUFFLED` with positive
   paired 95% lower bounds;
2. decoded and temporal MSE improve at least 3% over both `ZERO`/J1-off and
   `DIRECT`, with paired lower bounds above 1%, and beat `PFD_SHUFFLED` by at
   least 1% with positive lower bounds;
3. decoded and temporal MSE beat freshly evaluated VPM@1 with positive paired
   lower bounds;
4. latent NMSE has a nonnegative point effect versus `DIRECT` and VPM@1, with
   lower bounds above -1%;
5. mean direct-residual R2 and cosine both exceed those of `DIRECT` and
   `PFD_SHUFFLED`;
6. exactly one Wan call per J1 endpoint, zero teacher calls, zero auxiliary
   target opens, and an exact `ZERO == J1-off` no-op hold; and
7. no validation, protected test, or reserve row is opened.

Otherwise the decision is `STOP_PRIVILEGED_NOT_CAUSALLY_COMPRESSIBLE`. A stop
closes the clean-future-feature distillation direction at this seam; it does
not reject causal geometry, action-conditioned predictors, or a sampled
posterior/prior interaction latent.

## Claim boundary

This experiment uses one frozen J1 parent, one frozen VPM parent, a linear
token-local adapter, and train-only development episodes. Even an advance
authorizes a multi-seed nonlinear student continuation; it is not evidence of
general video-generation or closed-loop DAgger improvement.
