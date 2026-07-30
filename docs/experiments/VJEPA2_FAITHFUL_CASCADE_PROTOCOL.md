# V-JEPA 2.1 Faithful-Cascade Screening Protocol

Date preregistered: 2026-07-30

## Question

Does a generated V-JEPA 2.1 state improve an action-conditioned LACWM video
generator when it is trained and sampled with the ordering used by Latent
Forcing?

This is a warm-started ABC robotics-video screen. It is not evidence for
from-scratch training, other datasets, or a general world-model claim.

LACWM uses `sigma=1` for Gaussian noise and `sigma=0` for clean data.

## Why this run is new

The completed V-JEPA study tested aligned clocks and an overlapping
one-logit-leading schedule. It did **not** test the released Latent Forcing
recipe in which the semantic state is generated first and then frozen while
the visual state is generated. The prior J1 state gate also remained nearly
closed: its final state-to-native activation ratio was 0.7526%.

The hypothesis is that strict ordering plus a nonzero semantic residual lets
the video branch exploit V-JEPA structure. This remains a hypothesis until the
autonomous, off, and shuffled controls are measured.

## Frozen representation and data

- Existing immutable ABC clips: 512 train, 64 validation, and 128 inspected
  test clips, each with 13 RGB frames and three camera views.
- Cached, offline V-JEPA 2.1 ViT-B targets:
  `[B,64,4,24,120]`, PCA-whitened using train-only statistics.
- Video latent: `[B,16,4,24,120]`.
- Warm start, clip order, video corruption, V-JEPA target cache, PCA basis,
  initial evaluation noise, optimizer, and 1,000-update allocation are shared.
- The frozen V-JEPA teacher is never loaded by trainer or autonomous sampler.

No future RGB or clean future V-JEPA is accepted by the deployable sampler.
Oracle sources remain leakage-only diagnostics.

## Arms

| Code | Intervention | Purpose |
|---|---|---|
| V0 | Original ExplicitActionDiT | Architecture reference |
| VPM | Dual parameter schema; V-JEPA loss and fusion disabled | Primary video-only baseline |
| A1 | V-JEPA-first branch schedule; video fusion disabled | Schedule/multitask control |
| J0 | Joint aligned clocks and fusion | Previous overlapping-schedule control |
| J1 | V-JEPA-first branch schedule and fusion | Primary candidate |

A1 and J1 use identical branch probabilities, clocks, auxiliary loss
coefficient, target representation, and total inference call accounting. Their
conditioning flag is the primary difference.

## Training clocks

For A1 and J1, draw `b ~ Bernoulli(0.4)`.

- If `b=1` (V-JEPA branch), hold video at `sigma_video=1`, sample the V-JEPA
  clock from the configured logit-normal law, and apply only V-JEPA loss.
- If `b=0` (video branch), sample the native shifted Wan clock, draw
  `sigma_vjepa ~ Uniform(0,0.25)`, and apply only video loss.

Loss masks are averaged over the unchanged distributed global batch, matching
the released Latent Forcing implementation. The frozen V-JEPA loss coefficient
is the released recipe's `dino_weight=0.333`. Its DINO clean-time
`logit_mean=-1.2, std=1.0` becomes V-JEPA noise-sigma
`logit_mean=+1.2, std=1.0` under LACWM's opposite clock convention.

J1 initializes both effective semantic gates to `+0.02`. This was fixed before
examining any new validation result. A1's injection is disabled at runtime;
VPM retains its exact-zero no-op contract.

## Autonomous inference

For a total Wan-call budget `K >= 2`, split calls as evenly as possible:

1. V-JEPA phase: video remains bit-identical at `sigma_video=1` while V-JEPA
   moves from `sigma_vjepa=1` to `0`.
2. Video phase: V-JEPA remains bit-identical at `sigma_vjepa=0` while video
   follows the native Wan scheduler from `sigma_video=1` to `0`.

Every call executes the shared Wan backbone and counts toward NFE. NFE=1 is an
explicitly labeled degenerate aligned negative control, not a faithful
two-phase cascade.

The J1 same-checkpoint controls share an identical generated V-JEPA trajectory:

- `autonomous`: inject the sample-aligned final generated V-JEPA state;
- `off`: disable state and V-JEPA-clock injection only for the video phase;
- `autonomous_shuffled`: roll the final generated V-JEPA state across clips
  only for the video phase.

Thus, off and shuffled do not alter how the auxiliary state itself is
generated.

## Metrics and claim gates

All quality effects are paired by immutable clip ID. Primary saved metrics are
video-latent NMSE, decoded RGB MSE, decoded PSNR, and decoded temporal-
difference MSE. Auxiliary NMSE/cosine and activation ratios are mechanism
diagnostics, not primary quality metrics.

The one-seed screen passes only if all of the following hold:

1. **Training efficiency:** at autonomous NFE=4, J1 reaches VPM's
   update-1,000 temporal-MSE quality by update 800 or earlier, does so in less
   cumulative wall time, and has a positive paired temporal learning-curve AUC
   confidence-interval lower bound.
2. **Same-budget quality:** at update 1,000, the paired 95% confidence-interval
   lower bound favors J1 over VPM by at least 3% on temporal MSE, while video
   NMSE and decoded MSE lower bounds are no worse than -1%.
3. **Fusion attribution:** J1 beats A1 at the same update and total NFE, and
   J1 autonomous beats both its off and shuffled controls. A package-level
   gain without both contrasts is not attributed to sample-aligned V-JEPA.
4. **Phase-boundary integrity (`K >= 2`):** the generated `tf_final` tensor is
   exactly identical for autonomous, off, and shuffled controls before the
   video-only intervention. NFE=1 has no auxiliary-first phase and is excluded
   from this integrity check. Oracle gains do not satisfy any deployable gate.

Generated V-JEPA own-target NMSE/cosine is reported as mechanism telemetry.
This screen does not claim own-versus-rolled semantic identification because
that separate cross-sample target metric is not in the frozen evaluator.

Inference acceleration is evaluated separately:

1. Build a fresh VPM validation frontier using actual total Wan calls.
2. Freeze one candidate `J1@K` versus `VPM@M` with `K < M`.
3. Require quality preservation on all three primary metrics and at least 20%
   favorable p95 end-to-end latency, with a positive paired latency interval.

The historical `J1@4` versus `VPM@8` timing job is diagnostic only. It cannot
support a speed claim unless that pair is selected by the fresh frontier. If
VPM at NFE=1 remains the quality frontier, a strict cascade cannot be called
faster because it requires at least two calls.

LPIPS and a pinned non-V-JEPA video-feature metric are required before a
publication-strength "higher perceptual quality" claim. Reconstruction metrics
alone support only a reconstruction-quality screen.

## Repetition and decision rule

One seed (`1234`) is the first screen. An "obvious advantage" requires the
screen to pass and then reproduce in at least two additional frozen seeds. If
the screen fails, the result is negative for this representation, schedule,
warm start, and data budget; no positive result will be inferred from oracle
conditioning or a hand-selected NFE.

Implementation, validation, cluster monitoring, evaluation, and evidence
review will continue for at least eight hours. Monitoring is read-only and
will not stop or mutate unrelated or long-running jobs.
