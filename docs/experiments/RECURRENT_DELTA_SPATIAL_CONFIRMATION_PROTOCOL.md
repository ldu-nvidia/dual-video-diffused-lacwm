# Recurrent-delta spatial non-inferiority confirmation

Date frozen: 2026-08-09

Status: **prospective; no renderer-endpoint-55 future measured target indices or RGB
frames may be extracted until registration is sealed**

## Question and claim boundary

The completed fresh24 renderer gate found a large, horizon-consistent causal
motion gain for the recurrent delta-response predictor, but four zero-margin
spatial-retention intervals crossed zero.  This bounded confirmation asks
whether that already fitted predictor preserves practically equivalent spatial
alignment on the complete renderer-endpoint-unopened D405 census while
reproducing its flow gain.

A joint pass authorizes only an equal-Wan-call recurrent-flow screen against
VPM@1.  It is not evidence of improved generated RGB, FVD, policy success, or
real-time DAgger.  A failure closes the deterministic corrected-geometry
branch; endpoints, margins, families, or population may not be changed after
registration.

## Immutable population and target blindness

Only rows 384--511 of the immutable 512-row ABC **train** manifest are allowed.
Validation and protected test are forbidden.

- The canonical score pool contains exactly 103 top-D405 episodes.
- The completed Gate-0c registration selected 24 episodes and the completed
  trajectory-consistent fresh24 registration selected a disjoint 24.  Their
  exact renderer-outcome-unopened complement contains 55 episodes.
- A pre-registration, ID/path-metadata-only canonical scan found that unrelated
  later completed development/provenance populations collectively contain all
  55 IDs.  Therefore the global cross-experiment-unopened reserve is zero.  No
  renderer metric or outcome value was read, and none of those later artifacts
  contains this study's renderer endpoints.
- Registration takes **all 55** renderer-endpoint-unopened episodes, with no
  subsampling.  Under the already frozen fresh24 planned-command motion-bin
  boundaries, it must contain exactly 19 low-, 18 medium-, and 18 high-motion
  episodes.  This is an endpoint-specific census, not a globally fresh dataset.
- There is no outcome-based subsampling.  Donors are the next different episode
  in sorted `clip_id` order within the same frozen stratum, with wraparound.
- Before registration is sealed, the workflow may read only the manifest,
  cached planned actions and metadata, file presence, MCAP camera type, the two
  prior registrations, and the prior fitted-model artifact.  It may not open a
  score state container, decode score RGB, inspect earlier per-episode metrics
  for the 55, or access validation/protected-test data.
- Registration hash-binds both prior selections, the final canonical
  metadata-only freshness audit, all 55 IDs and
  donor IDs, input files, source commit and tool, prior model artifact,
  thresholds, seeds, and software versions.  Any prior completed endpoint
  artifact containing any of this study's renderer metric records blocks launch.

The population is a train-only bounded confirmation, not a public holdout or a
claim of untouched model-training data.

## Frozen model and causal preparation

The candidate is the exact recurrent delta-response model fitted on train rows
0--383 by the completed fresh24 study.  Its PCA state, standardizers, ridge
coefficients/intercept, selected alpha `0.1`, gain `1.0`, and three-round
self-rollout recipe are copied byte-for-byte from the canonical
`model_state_and_predictions.npz` and hash-bound.  This study performs no
refit, CV, hyperparameter choice, calibration, or outcome-dependent repair.

For transition `t`, the frozen rollout is

\[
\hat q_4=q^{meas}_4,\qquad
\hat q_{t+1}=\hat q_t+\Delta q^{raw}_t+
f_\phi(H,A,\hat q_t,t).
\]

Allowed predictor inputs are measured history only through boundary 4,
observed and proposed future commands, the current predicted state, prior
predicted residual, frozen fit384 transforms, robot geometry, and camera
calibration.  Future measured boundaries 5--12 and future RGB remain scoring
targets only.

Preparation writes and hash-closes every raw, absolute-ridge, recurrent,
recurrent-shuffled, and measured-history trajectory before evaluation is
allowed to extract future target indices or decode future RGB.  The NPZ format
stores complete state arrays as indivisible members, so requesting an array can
decompress its full member in memory; the audited preparation source indexes,
retains, hashes, and supplies only boundaries through 4, and a separate process
indexes boundaries 5--12 after closure.  This is source-enforced target-index
blindness, not an operating-system claim that future bytes were unread.  The
prior absolute-ridge model is copied unchanged; it is a mandatory reference,
not a retuned candidate.

## Frozen arms and renderer

The confirmatory arms are:

| Arm | Construction | Role |
|---|---|---|
| `raw_command` | proposed native command endpoints from measured `q4` | causal reference |
| `absolute_ridge` | frozen fresh24 absolute-residual predictor | learned reference |
| `recurrent_delta` | frozen closed-loop delta-response rollout | candidate |
| `recurrent_delta_shuffled` | recipient history/`q4`, donor future commands | attribution control |
| `measured_oracle` | future measured state | scoring-only target |

Rendering and metrics retain the prior contract exactly: official ABC/YAM
commit `6bc6586721cf0c409ccee80f675a28de9b9b2f5e`, nominal top D405 camera,
native resolution, robot-only articulated geometry, centered principal point,
recorded `fy`, no distortion, the measured-oracle mask as the fixed scoring
region, 12-pixel RGB robot band, stride-two exact geom-local point transport,
and a 0.25-pixel minimum oracle motion.  Lower is better for robot-flow EPE,
silhouette boundary Chamfer, and RGB-band edge Chamfer; higher is better for
silhouette IoU.

## Frozen estimands, margins, and multiplicity

The analysis unit is an episode.  All effects are paired and positive values
favor recurrent delta.  One common 100,000-draw stratified episode-bootstrap
index matrix (sampling 19/18/18 within the frozen strata), seed `20260809`, is
used for every endpoint.  There are two scientifically distinct Holm step-down
families at one-sided familywise alpha 0.05; neither family can rescue the
other.

The three flow-superiority hypotheses are mean
`reference EPE - recurrent EPE` versus raw, absolute ridge, and recurrent
shuffled.  Holm tests use the zero null.  Raw and ridge comparisons also require
at least 5% point relative improvement.  Every flow comparison requires a
strictly positive Holm lower bound and at least 60% favorable episodes (at
least 33 of 55);
recurrent versus shuffled requires no additional relative-effect margin.

The six spatial non-inferiority hypotheses compare recurrent with raw and
absolute ridge:

- silhouette and RGB-band Chamfer effect is
  `reference - recurrent`, with non-inferiority margin `-0.25` native pixel;
- silhouette-IoU effect is `recurrent - reference`, with non-inferiority margin
  `-0.005` absolute IoU.

The margins were frozen before opening the 55 renderer outcomes.  They are
**pilot-informed operational tolerances**, not camera/renderer resolution
constants and not a reinterpretation of fresh24's failed zero-margin test.
A quarter native pixel is a deliberately stringent subpixel *mean* tolerance;
0.005 IoU is a half-percentage-point mask-overlap tolerance.  They are practical
study tolerances, not claims that all downstream generation differences below
them are imperceptible.

For each family, bootstrap p-values test its stated null and Holm-adjusted lower
bounds use the corresponding step-down quantile.  Every member of both families
must pass.  Flow gains cannot compensate for spatial loss, spatial NI cannot
compensate for flow failure, and point estimates cannot override intervals.

The sole positive decision is `GO_RECURRENT_DELTA_WAN_SCREEN`; otherwise the
decision is `STOP_RECURRENT_DELTA_GEOMETRY`.  No Wan job is submitted by this
workflow.

## Power and limitations frozen before outcomes

The 55-episode size is fixed by the complete renderer-endpoint census, not
selected from a power search.  A normal approximation using fresh24 paired
standard deviations and the six-test spatial-family critical value puts the
bottleneck ridge-IoU NI gate near 83% power under the pilot effect; transport
and winner-selection make that conditional.  A failure rejects advancement of
the configured predictor from this evidence package but does not prove that
recurrent rendering is universally harmful.  The earlier fresh24 remains pilot
evidence and is not pooled into the primary interval.

Other limitations are one fitted predictor, train-only D405 episodes that were
present in unrelated later experiment populations, endpoint-specific rather
than dataset-level freshness, nominal calibration, independently resampled
command/state streams, robot-only geometry without contact or occlusion, and no
generated-video or serving measurement.

## Artifact and audit contract

The immutable workflow is `register -> prepare -> evaluate -> analyze -> audit`.
It records identities and SHA-256 hashes for source, prior registrations/model,
selection/donors, causal trajectories, bundles, rows, bootstrap indices,
analysis, completion, and environment.  The read-only audit replays causal
rollouts, confirms all trajectories were closed before future-target indexing
and RGB decoding, recomputes
the two Holm families and machine decision, checks exact 55 membership and
19/18/18 strata, validates all hashes and protected-data false flags, and emits
no mutation of evidence.  Partial or failed runs remain immutable and can never
be promoted as completed evidence.
