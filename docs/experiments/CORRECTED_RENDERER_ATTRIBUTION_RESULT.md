# Predicted tracking-corrected renderer attribution result

Date completed: 2026-08-08

Decision: **`STOP_RENDERER_ATTRIBUTION`; do not integrate this predictor into
Wan yet**

## Central finding

The causal tracking correction is real and survives forward kinematics for
**pose and silhouette**, but it does not reliably improve the **robot motion
field** over the raw command trajectory.

On 24 motion-stratified D405 train episodes held out from predictor fitting,
predicted correction reduced rendered-silhouette boundary error by 34.40% and
fixed-region RGB edge error by 5.19% versus raw-command rendering. Both paired
intervals were positive. However, robot-only flow EPE improved by only 3.16%,
its paired interval crossed zero, and only 13/24 clips favored correction. This
single preregistered failure correctly stops Wan integration.

This is more informative than either “the renderer works” or “the idea fails.”
The current all-at-once ridge predictor is a useful absolute-pose corrector, but
not yet a sufficiently reliable trajectory/velocity corrector. The next model
should predict command-response increments and optimize projected motion—not
add another video feature or launch RGB training.

## Executed prospective contract

- Source manifest: immutable ABC train512 only; no validation or protected
  test.
- Predictor fit: manifest rows 0--383, 384 unique episodes.
- Scoring pool: rows 384--511, 128 different episodes.
- Camera scan before RGB/state scoring found 103 D405, 13 OAK, 11 ZED-X, and
  one `decxin` episode.
- Selection: three planned-command-motion strata, eight D405 clips per stratum,
  chosen only by the frozen SHA-256 rank; 24 unique score episodes.
- Shuffle donors: next selected episode within the same motion stratum. Native
  history and nominal endpoint stayed fixed; only the future action feature was
  donated.
- Predictor: independently standardized history/action PCA64 plus multi-output
  ridge. Five-fold fit384 CV selected alpha 10.0. History/action PCA retained
  99.767%/99.949% of fit variance.
- Frames: observed history through boundary 4; future boundaries 5--12 scored.
- Arms: raw command, predicted correction, hold current, within-stratum
  episode-shuffled correction, and measured-state oracle diagnostic.
- Camera/model: official ABC commit
  `6bc6586721cf0c409ccee80f675a28de9b9b2f5e`, nominal top D405 extrinsic,
  recorded `fy`, centered principal point, no distortion.
- Fixed robot region: measured-oracle articulated mask, scoring-only. Static
  bases/gate/camera were excluded.
- Flow: exact geom-local point transport on oracle-visible articulated pixels,
  two-pixel grid, oracle motion at least 0.25 px, clip-pooled by qualifying
  pixel count.
- Statistics: 20,000 paired clip bootstrap samples. The primary silhouette
  metric required at least 5% improvement versus raw and shuffled; every
  silhouette, regional edge, and flow contrast required a positive lower bound
  and at least 60% favorable clips. IoU could not regress.

The registered v3 execution source is commit
`9499c05ecec504650ad06c96e878005f073ced49`, SHA-256
`f70898acf3411b7f43137ccb9ab4c46a489c0a7aebc324ade0d6a296e60ffc60`.

## Absolute arm metrics

Lower is better except IoU.

| Arm | Silhouette boundary Chamfer (px) | Silhouette IoU | RGB robot-band Chamfer (px) | Robot-flow EPE (px) |
|---|---:|---:|---:|---:|
| Raw command | 2.4335 | 0.8521 | 7.7504 | 4.5257 |
| **Predicted corrected** | **1.5964** | **0.8921** | **7.3482** | **4.3828** |
| Episode shuffled | 3.3085 | 0.8120 | 7.6686 | 6.4440 |
| Hold current | 7.2631 | 0.7061 | 10.3809 | 7.7307 |
| Measured oracle diagnostic | 0.0000 | 1.0000 | 7.2591 | 0.0000 |

The oracle's nonzero RGB-edge error is expected: observed robot-band edges
contain texture, grasped objects and background, while the render has nominal
extrinsics, a centered principal point and no distortion. Candidate RGB-band
Chamfer is only 0.0891 px above this diagnostic ceiling, whereas raw command is
0.4913 px above it. Correction therefore closes about 81.9% of the raw-to-
oracle RGB-band gap, but only 3.16% of the raw-to-oracle flow gap.

## Registered paired effects and gates

Positive differences favor predicted correction. Intervals are paired
clip-bootstrap 95% intervals.

| Contrast | Relative gain | Favorable difference (px or IoU) | 95% interval | Favorable clips | Gate |
|---|---:|---:|---:|---:|---|
| Silhouette vs raw | +34.40% | +0.8370 px | `[+0.5256,+1.1507]` | 20/24 | pass |
| RGB-band edge vs raw | +5.19% | +0.4022 px | `[+0.2163,+0.5989]` | 19/24 | pass |
| **Robot flow vs raw** | **+3.16%** | **+0.1429 px** | **`[-0.3054,+0.5747]`** | **13/24** | **fail** |
| IoU vs raw | +4.70% | +0.0401 | `[+0.0251,+0.0547]` | 21/24 | pass |
| Silhouette vs shuffled | +51.75% | +1.7120 px | `[+1.3056,+2.1283]` | 23/24 | pass |
| RGB-band edge vs shuffled | +4.18% | +0.3204 px | `[+0.1131,+0.5348]` | 16/24 | pass |
| Robot flow vs shuffled | +31.99% | +2.0612 px | `[+1.5338,+2.6228]` | 23/24 | pass |
| IoU vs shuffled | +9.86% | +0.0801 | `[+0.0621,+0.0981]` | 23/24 | pass |

The shuffled results establish sample-specific planned-action attribution. They
do not rescue the candidate: deployment compares against the native raw command,
and that flow contrast failed both the positive-interval and 60%-clip rules.

## Why pose improves but motion does not

The state-space diagnostic already favored correction:

| Held-out score24 state endpoint | MSE |
|---|---:|
| Predicted corrected | 0.001430 |
| Raw command | 0.003249 |
| Episode shuffled | 0.005410 |
| Hold current | 0.041950 |

The average unprojected trajectory-increment MSE also fell by 30.03% versus raw
with a positive exploratory interval and 20/24 favorable clips. Yet visible
pixel flow weights joints, link surfaces and time transitions very differently
from uniform joint/gripper MSE. The horizon breakdown identifies the failure:

| Transition horizon | Raw flow EPE | Corrected flow EPE | Corrected minus raw |
|---:|---:|---:|---:|
| 1 | 6.798 | 4.061 | -2.738 |
| 2 | 3.727 | 3.715 | -0.012 |
| 3 | 3.540 | 4.286 | +0.746 |
| 4 | 4.031 | 3.845 | -0.186 |
| 5 | 4.425 | 4.716 | +0.291 |
| 6 | 4.499 | 3.725 | -0.775 |
| 7 | 5.242 | 5.392 | +0.150 |
| 8 | 4.516 | 6.054 | +1.538 |

The first transition is substantially better, but independent absolute
corrections across the remaining horizon do not produce consistently better
frame-to-frame motion; the final transition is materially worse. Descriptively,
flow gains versus raw were +0.25%, -0.41%, and +6.73% in the low-, medium-, and
high-command-motion strata. This is not simply a low-motion threshold artifact.

The next model should therefore predict transition response and integrate it,
rather than predict eight absolute residuals with only uniform residual MSE.

## Timing and operational cost

Aligned scoring beat both nonwrapping one-step shifts. For predicted correction,
aligned silhouette Chamfer was 1.596 px versus 2.770 at -1 and 3.206 at +1;
aligned RGB-band Chamfer was 7.348 versus 7.799 and 7.634. The measured oracle
also had the best RGB-band score when aligned. This supports clip-step alignment
but cannot recover sub-frame controller latency from ceiling-resampled streams.

Full causal predictor replay—including history/residual construction, two
PCA64 projections, ridge inference, and destandardization—reproduced stored
predictions to maximum absolute error `3.05e-7` and measured:

| Predictor mode | Mean | p50 | p95 |
|---|---:|---:|---:|
| Batch 1, per clip | 0.0191 ms | 0.0183 ms | 0.0238 ms |
| Batch 24 total | 0.4673 ms | 0.4669 ms | 0.4823 ms |
| Batch 24 amortized | 0.0195 ms | 0.0195 ms | 0.0201 ms |

MuJoCo segmentation+depth pose rendering measured 3.467 ms mean and 4.656 ms
p95 after two warmups per clip. A nine-pose candidate trajectory is therefore
about 31.2 ms at the per-pose mean, or a conservative 41.9 ms from nine p95
poses, before Wan and decoding. Auxiliary compute is compatible with a 5--10 Hz
budget; end-to-end video generation has not been measured.

## Qualification failures and hardened audit

Two incomplete attempts are preserved rather than overwritten:

- v1, source `3fa5676`: stopped before metric rows because a fine-motion
  transition had zero pixels above the flow threshold;
- v2, source `10f7550`: stopped before metric rows because the evaluator assumed
  one shared D405 resolution.

Both contain registration, preparation and bundles, but no `analysis.json`, no
completion receipt and zero emitted frame-metric files. V3 froze symmetric
zero-support pooling and per-clip native-resolution renderers before restarting
from a fresh registration. Selected IDs, donors, predictor, metrics, bootstrap
and gates remained unchanged.

The independent read-only audit passed after recomputing all eight 20,000-
sample effects and all nine gates. It verified:

- 72 artifact hashes;
- 24 bundles, 960 frame/transition rows, 120 clip rows, 1,680 timing rows and
  408 provenance rows;
- all sealed identity bindings and the final stop decision;
- 3,270 explicit protected-test false flags;
- different-episode within-score-pool donors;
- per-clip moving-flow support and source reprojection RMSE below `1e-5` px.

Key identities:

- registration: `494ae93b5b87ca4ed19e1497f6cdd64d51dba0f527ac51a83a8eff3eaa676052`;
- preparation: `2cb1f84a2e813cbc256798ef868ae1788e35d0accf1f1d2f5b1399a44d3ea2b4`;
- analysis: `44b155aa334ec57a893609e47c0b03cbbd7ccbb525c9ca032a57f7dbc7ec715b`;
- completion: `b2d30521d2686bd8be72d87e0fa263e2b02a3e24caeb2183d37961a43ac307e1`;
- hardened audit: `d619ae35c0a72a05141cad93821a8d7479f73d98a4b8425b3dae59f682c1fe9b`;
- latency replay: `2390eefcb55a4075fa6c7306767725e18ecea198276e78cb71745ce02ba83bd9`.

Local artifact:

```text
/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/
  corrected_renderer_attribution/
  corrected-renderer-train384-score24-seed20260808-9499c05-v3/
```

Canonical Lustre artifact:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/corrected_renderer_attribution/
  corrected-renderer-train384-score24-seed20260808-9499c05-v3/
```

## Evidence-based next experiment

Do not spend a Wan run on the current corrected trajectory. Implement a compact
**increment-response predictor**:

\[
\Delta q^*_t=q^{meas}_{t+1}-q^{meas}_{t},\qquad
\hat q_{t+1}=\hat q_t+f_\phi(H,a_t,\hat q_t),
\]

with the observed frame-4 state as the initial condition. Train with a joint
objective

\[
L=L_{\Delta q}+\lambda_q L_q+lambda_g L_{\text{projected robot flow}},
\]

where the last term weights errors by visible link geometry/camera projection.
Mandatory arms should be raw command, current absolute-residual ridge, direct
increment predictor, recurrent increment predictor, action-shuffled predictor,
and measured oracle diagnostic. Freeze hyperparameters using fit384 only.

Use a new prospective score24 selected from the 79 D405 episodes in the current
score pool whose RGB/state outcomes were never evaluated. Require the new
predictor to beat both raw command and the current ridge on robot-flow EPE by at
least 5%, with a positive paired lower bound and at least 60% favorable clips,
while preserving the current silhouette/RGB gains. Only then launch the frozen
Wan flow-conditioning arms.

This remains an inference-causal deterministic geometry branch, not dual
diffusion by itself. Its value is to remove predictable robot motion so a later
stochastic auxiliary state can focus on objects, contacts and visibility.

## Limitations

- One fixed predictor seed, 24 train-only score episodes and top view only.
- Current score24 is now adaptive development evidence; it must not be reused as
  confirmation for the redesigned predictor.
- Raw command is a zero-order-hold endpoint proxy, not controller simulation.
- Ceiling-resampled state/action streams prevent precise controller latency.
- Nominal extrinsic, centered principal point and no D405 distortion.
- Robot-only rendered flow excludes object/contact dynamics and real scene
  occlusion.
- No Wan adapter, generated RGB, FVD/perceptual metric, policy success,
  protected test, or closed-loop DAgger measurement.
