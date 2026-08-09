# Predicted tracking-corrected renderer attribution

Date frozen: 2026-08-08

Status: **prospective train-only protocol; outcomes have not been inspected**

## Question and claim boundary

Does the compact, inference-causal tracking-residual predictor from
`NOMINAL_TRACKING_RESIDUAL_STAGE0_RESULT.md` improve the projected robot
geometry and robot-only motion field relative to rendering the raw command
trajectory?

This is a renderer-attribution prerequisite, not a video-generator experiment.
It cannot establish improved RGB generation, FVD, perceptual quality, DAgger
performance, or calibrated controller simulation. The nominal endpoint remains
the last recorded raw absolute-position command in each five-sample action
chunk. Measured future state is scoring-only and is always labelled a
privileged diagnostic.

## Prospective split and selection

The immutable 512-row ABC **train** manifest is the only data source. Validation
and protected test are forbidden.

- Manifest rows `0:384`, each from a unique episode, fit the predictor.
- Rows `384:512`, also unique episodes and disjoint from the fit rows, form the
  untouched train-only scoring pool.
- The registration stage may read only the train manifest, cached raw action
  tensor, MCAP metadata, and camera-info calibration. It must write a sealed
  registration before opening score-pool RGB or measured state.
- Eligible score rows must have a complete preprocessed episode and a top
  camera declared as `Intel RealSense D405`.
- For each eligible row, the selection scalar is planned joint-command motion

  \[
  m_i=\sqrt{\frac{1}{8\cdot 12}
  \sum_{t=4}^{11}\left\|a_{i,t,4,1:12}-a_{i,3,4,1:12}\right\|_2^2}.
  \]

  RGB, measured future pose, image edges, and alignment metrics are forbidden
  selection inputs.
- Eligible rows are rank-split into three planned-motion strata with
  `numpy.array_split`. Eight rows per stratum are chosen by the smallest
  SHA-256 rank of `clip_id|20260808`, producing exactly 24 score clips. If any
  stratum has fewer than eight rows, registration fails rather than changing
  the design.
- Within each stratum, the episode-shuffled donor is the next selected row with
  cyclic wrap. Donors are therefore different episodes and approximately
  motion-stratum matched.

The registration records exact row indices, clip IDs, strata, donor IDs,
manifest/cache hashes, camera metadata, and the committed tool hash. A failed
or partial attempt is never reused as a completed artifact.

## Frozen predictor and trajectories

The predictor is refit from scratch on rows `0:384`; the prior train512 model is
not evaluated on its fitting rows. The feature and target contract is unchanged
from Stage 0:

- history: measured states at frames 0--4, raw commands in chunks 0--3, and
  observed tracking residuals in transitions 0--3;
- candidate-only causal input: planned raw command chunks 4--11;
- target-only residual:
  `q_measured[t+1] - action[t,4]`, `t=4..11`;
- independently train-standardized history/action PCA64 features;
- multi-output ridge, alpha in
  `{0.1,1,10,100,1000,10000}`, selected by five-fold fit384 cross-validation;
- one fixed seed, 1234.

For target video boundaries 5--12, define five frozen arms in cache order
`[left joints 1..6, right joints 1..6, left grip, right grip]`:

| Arm | Rendered trajectory | Availability |
|---|---|---|
| `raw_command` | native `action[t,4]` endpoint | inference-causal |
| `predicted_corrected` | native endpoint plus aligned predicted residual | inference-causal candidate |
| `hold_current` | measured frame-4 state repeated | inference-causal negative control |
| `episode_shuffled` | native endpoint plus residual predicted with the registered donor action feature | attribution control |
| `measured_oracle` | measured states at frames 5--12 | privileged scoring diagnostic only |

Every trajectory begins at the native measured frame-4 history state for the
first flow transition. Predicted/raw grippers are clipped to `[0,1]` only at the
renderer boundary; both unclipped and rendered values are retained. The exact
cache-to-official permutation remains
`[0,1,2,3,4,5,12,6,7,8,9,10,11,13]`.

## Renderer and fixed scoring support

The renderer uses official `amazon-far/abc` commit
`6bc6586721cf0c409ccee80f675a28de9b9b2f5e`, the robot-only YAM visual scene,
the nominal top D405 extrinsic, and recorded `fy` as MuJoCo vertical FOV. The
principal point remains centered and distortion is not applied. Since all arms
share the same camera model, this study attributes relative trajectory changes;
it does not claim full image calibration.

Static gate/camera geometry and fixed arm bases are excluded. Only visible
geometries descending from the two articulated arm roots are robot support.
For every target frame, the **fixed scoring region** is derived from the
measured-oracle robot mask and never from a candidate arm:

- silhouette boundary Chamfer: symmetric candidate-to-oracle rendered boundary
  distance;
- silhouette IoU: candidate mask versus measured-oracle mask;
- RGB robot-band Chamfer: candidate boundary to observed Canny edges restricted
  to a 12-pixel dilation of the measured-oracle robot mask;
- 3-pixel edge support and candidate-boundary retention inside the fixed band
  are reported so background/object edges cannot silently dominate.

The RGB edge metric uses future RGB only for scoring. It is never a predictor or
renderer input.

## Robot-only geometry flow

Robot flow is not inferred from full-scene RGB. For each oracle-source pixel on
visible articulated geometry, MuJoCo depth and geom ID recover a geom-local
surface point. The same local point is transported through source/target geom
transforms for each arm and projected through the shared nominal camera. Flow
endpoint error is measured against the measured-oracle transport on a fixed
two-pixel grid, restricted to oracle-visible source points whose oracle target
is in-frame and whose oracle motion is at least 0.25 pixels. Source reprojection
RMSE, valid support, in-frame support, and moving-pixel counts are mandatory
audit fields.

A transition with fewer than ten qualifying moving pixels has no identifiable
robot-flow endpoint and is excluded symmetrically for every arm, rather than
being assigned an arbitrary zero or failure value. Clip flow EPE is pooled by
qualifying oracle pixel count across the remaining transitions; each clip must
retain at least ten pixels or execution fails. This zero-support rule was
frozen after a tooling-qualification attempt stopped before writing any frame
metric or analysis because a fine-motion transition had zero pixels above the
registered 0.25-pixel threshold. No renderer comparison outcome was available
from that incomplete attempt.

This is a rendered robot-geometry flow diagnostic. It excludes objects,
contacts, occlusions by scene objects, camera distortion, and real optical-flow
estimation.

## Timing diagnostics

All primary comparisons are aligned and non-cyclic. As a diagnostic for the
known ceiling-resampling ambiguity, every already-rendered trajectory is also
scored at `-1` and `+1` clip-step shifts using only nonwrapping pairs. The
measured-oracle lead/lag scores bound whether nominal timing is locally
identifiable. No arm or timing offset is selected from these results, and these
diagnostics cannot replace the aligned primary gate because one clip step is
about 0.17 seconds rather than sub-frame controller timing.

## Statistics and strict decision gate

Each metric is first averaged over target frames/transitions within each clip.
All inference uses 20,000 paired clip bootstrap resamples with seed 20260808
plus a registered contrast offset. Lower is better for Chamfer/flow EPE; higher
is better for IoU/support.

`GO_FOR_WAN_SCREEN` requires `predicted_corrected` to beat **both**
`raw_command` and `episode_shuffled` on all three mandatory lower-is-better
endpoints:

1. symmetric rendered-silhouette boundary Chamfer;
2. fixed-region RGB robot-band Chamfer;
3. rendered robot-only flow EPE.

Rendered-silhouette boundary Chamfer is the registered **primary alignment
metric**. Against both raw and shuffled references it must improve by at least
5% at the aggregate paired point estimate. For every one of the six metric
contrasts, the paired point difference must favor the candidate, the 95% lower
bound must be strictly positive, and at least 60% of clips must be favorable.
Thus RGB-band and robot-flow metrics are stricter no-secondary-regression
conditions rather than substitutes for the primary endpoint. Predicted
silhouette IoU must have a positive point difference against raw and shuffled
and may not have a negative 95% lower bound. Every condition is mandatory.
Otherwise the decision is
`STOP_RENDERER_ATTRIBUTION` and no Wan integration is authorized from this
study.

Hold-current and measured-oracle results are diagnostics, never alternative
gate references. No threshold, clip, camera, timing shift, or metric may be
changed after registration based on observed outcomes.

## Immutable artifacts and audit

The workflow writes:

- sealed `registration.json` before score RGB/state access;
- input provenance and exact selected/donor rows;
- refit model state and residual predictions;
- hash-addressed compact RGB/trajectory/calibration bundles;
- frame/transition rows and clip-aggregate metrics;
- sealed analysis and completion receipts;
- overlays for failure inspection;
- a read-only audit checking hashes, identities, train-only flags, split
  disjointness, donor identity, row counts, fixed controls, flow support, and
  explicit `protected_test_accessed=false` flags.

Results and any recommendation may be appended only after the registration and
execution source are committed.

## Predeclared limitations

- Only 24 deliberately motion-stratified ABC train episodes and one seed.
- The score pool is held out from predictor fitting but is not a public test or
  paper-level benchmark.
- Raw-command endpoints are not official controller/dynamics rollouts.
- State and command streams were independently ceiling-resampled; raw sample
  timestamps are unavailable.
- Nominal extrinsics, centered principal point, and no distortion correction.
- Measured state defines the fixed scoring region and oracle flow target; it is
  strictly evaluation-only.
- Rendered flow models robot geometry, not objects, contacts, visibility by
  scene objects, or stochastic futures.
- Passing would authorize only a controlled Wan flow-conditioning screen, not
  a dual-diffusion or generation-quality claim.
