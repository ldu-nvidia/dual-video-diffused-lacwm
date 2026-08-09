# Next twelve-hour dual-video execution protocol

Date: 2026-08-08

Status: prospective decision protocol, committed before reading any new
compressibility, corrected-renderer, or V-JEPA 2-AC outcome

## Question

Can an auxiliary state that is available without unknown future RGB improve the
feature-free, low-NFE video endpoint beyond the actual VPM@1 frontier?

The previous study established only three prerequisites:

1. a clean target-video V-JEPA teacher is unusually accurate at the initial
   `sigma_video=1` update, but is nondeployable and harmful near the final
   low-noise update;
2. observed history and planned actions predict a compact command-tracking
   correction;
3. nominal robot rendering is temporally informative and inexpensive, but the
   bounded geometry probe used observed poses rather than a deployable predicted
   trajectory.

No previous result establishes a feature-free video-quality gain. This protocol
tests whether any of those prerequisites survives its missing causal step.

## Frozen priority and branching

The three Stage-0 gates can run in parallel, but their outcomes have fixed
consequences:

1. **Causal compressibility:** can a feature-free residual head predict the
   useful part of the privileged high-noise teacher correction?
2. **Corrected rendering:** does the action/history-predicted tracking
   correction improve projected robot alignment beyond raw commands?
3. **V-JEPA 2-AC qualification:** does the official action-conditioned
   predictor use inference-available actions and nominal states to improve
   future-feature prediction beyond controls?

Only a passing Stage-0 mechanism may enter Wan. If more than one passes, first
advance the mechanism with the largest paired lower-bound improvement per
measured millisecond and GPU-hour. Do not combine mechanisms before each has an
individually attributed feature-free gain.

## Shared safety and evidence contract

- No protected-test or prior lockbox sample is opened during this phase.
- Episode identity, rather than row index alone, defines split disjointness.
- Training-only future RGB, clean V-JEPA, measured future state, and optical
  flow may define targets or privileged diagnostics, but never enter a reported
  feature-free sampler.
- Evaluation noise seeds are disjoint from adapter-fitting noise seeds.
- Every shuffled control uses a different episode and preserves marginal scale.
- Selection metrics, thresholds, seeds, and split construction are serialized
  before outcome rows are read.
- Effects use paired episode-clustered bootstrap intervals with 10,000 fixed
  resamples. Positive percentages always mean lower error.
- Equal Wan calls are necessary but not sufficient: encoder, predictor,
  renderer, adapter, VAE, decoder, and policy-side latency are reported.
- Large artifacts live on `/lustre/fsw`; source, protocols, registrations, and
  result summaries live in Git.
- W&B, if actual optimization is launched, uses the user's private project with
  no organizational group. Read-only probes keep W&B disabled.

## Gate A: causal compressibility ladder

### Frozen inputs and targets

At the J1 initial pure-noise video update, capture the input-visible shared Wan
trunk tokens from the feature-off forward pass. The four equal-capacity residual
heads receive exactly those tokens, observed history/actions, time, and noise:

| Arm | Training target |
|---|---|
| `ZERO` | exact zero |
| `DIRECT` | stopped ordinary flow residual `v* - vS` |
| `PFD-ALIGNED` | stopped clean-feature teacher residual `vT - vS` |
| `PFD-SHUFFLED` | stopped different-episode teacher residual |

The target cache and teacher are absent while scoring corrected velocities and
rollouts. The adapter-fitting split and development split are episode- and
noise-seed-disjoint. The actual frozen VPM@1 sampler is evaluated on identical
clips and noise; its previously reported aggregate is not substituted for a
paired comparison.

### Advance gate

Advance `PFD-ALIGNED` only if all conditions hold:

1. held-out corrected velocity MSE beats `DIRECT` and `PFD-SHUFFLED` by at
   least 3%, with paired lower bounds above zero;
2. privileged-residual prediction has positive held-out R2 and materially
   exceeds the shuffled target on both R2 and cosine;
3. one-step decoded and temporal MSE improve at least 3% over J1-off, with
   paired lower bounds above 1%;
4. the feature-free endpoint improves decoded and temporal MSE over actual
   VPM@1 with positive paired lower bounds, while latent NMSE does not regress
   by more than 1%; and
5. feature-free evaluation records exactly zero teacher/cache calls and no
   material serving latency beyond the registered adapter allowance.

If `DIRECT` matches or beats PFD, the result supports ordinary residual
optimization rather than privileged semantic transfer. If PFD improves J1 but
does not beat VPM@1, it repairs a weaker parent and stops.

## Gate B: predicted tracking-corrected rendering

Refit the compact tracking predictor on a deterministic episode-disjoint
training subset and freeze it before image scoring. Select the scoring pool by
robot/camera availability and action/state motion metadata only. Compare:

| Arm | Render trajectory |
|---|---|
| `RAW` | nominal raw-command trajectory |
| `PREDICTED` | raw command plus predicted tracking correction |
| `HOLD` | current measured state held fixed |
| `SHUFFLED` | different-episode predicted correction |
| `MEASURED` | measured future state, privileged diagnostic only |

Use nonwrapping timing controls and score robot-support edge distance, soft
support overlap, rendered-layer flow direction/endpoint error, and per-view
latency. Report robot-support, nearby nonrobot foreground, and background
regions separately where the observable evidence permits it.

Advance to fixed-flow Wan conditioning only if `PREDICTED` beats both `RAW` and
`SHUFFLED` on the registered primary alignment metric by at least 5%, both
paired lower bounds are positive, at least 60% of clips favor `PREDICTED`, and
no registered secondary metric materially regresses. `MEASURED` cannot make a
failed causal arm pass.

## Gate C: V-JEPA 2-AC causal qualification

Pin the official source and official `vjepa2-ac-vitg.pt` bytes. The released
predictor was trained on monocular DROID clips using 7-D Cartesian state and
action tokens, with only two autoregressive future steps in the official
configuration. Therefore:

1. first run a native-contract DROID positive control;
2. distinguish teacher-forced/logged-state diagnostics from inference-valid
   autoregressive prediction;
3. for ABC, construct future state tokens only from the observed state and
   planned/FK-integrated trajectory—never from future measured state;
4. compare aligned actions against zero, episode-shuffled, and signed
   time-shifted actions;
5. compare with raw-action/history baselines and a from-scratch matched
   predictor wherever feasible; and
6. measure encoder, predictor, and autoregressive rollout latency separately.

The route advances only if the native positive control passes, the causal ABC
arm improves future-feature error at least 5% over both zero and shuffled
actions with positive paired lower bounds, at least 60% of clips are favorable,
and the signal remains beyond the raw-action/history baseline. Failure on ABC
after a passing native control is a transfer/action-contract failure, not proof
that V-JEPA 2-AC itself is invalid.

## Stage 1: low-NFE video attribution

For each passing mechanism, start from one registered parent and use identical
data order, optimization budget, noise, trainable capacity, and Wan calls.
Evaluate NFE 1 and 2 with at minimum:

- feature off;
- aligned causal auxiliary;
- episode-shuffled auxiliary;
- time-shifted or corrupted auxiliary where defined; and
- a privileged target-derived diagnostic that never participates in selection
  of a deployable endpoint.

The causal arm advances only if decoded and temporal MSE improve at least 3%
over both feature-off and the actual VPM@1 frontier, with paired lower bounds
above 1%; latent NMSE has a nonnegative point effect and lower bound above -1%;
aligned beats shuffled by at least 1%; complete latency is reported; and all
future-feature call counters remain zero.

For a stochastic interaction state, paired MSE is secondary rather than the
sole primary endpoint. Use equal-count multi-sample distributional/perceptual
metrics, calibration/coverage, contact and object-region consistency, and
downstream planning utility. Any best-of-K comparison uses the same K and total
compute for every arm. Robot and object motion are layered with masks, depth,
visibility, and warp composition; unrestricted pixelwise flow subtraction is
not treated as an interaction latent.

## Stop and claim rules

- A privileged teacher gain is not a deployable video gain.
- A gain over J1 that remains below VPM@1 is not progress over the frontier.
- A rendered robot-silhouette gain is not evidence of better object/contact
  prediction unless object-region metrics also improve.
- A V-JEPA 2-AC result using future measured state is a diagnostic, not a
  causal auxiliary.
- No few-step distillation begins until a feature-free teacher beats VPM@1.
- No paper-level claim is made without multi-seed confirmation, a frozen
  untouched split, perceptual/distributional metrics, long-horizon rollouts,
  and closed-loop utility.

