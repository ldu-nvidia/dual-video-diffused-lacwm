# Causal interaction-event bottleneck Stage 0

Date frozen: 2026-08-08

Status: **prospective; no score target has been opened by this study**

## Question and claim boundary

After explicitly removing rendered robot pixels, is a severe, 32-dimensional
summary of future top-camera change and transport both:

1. causally predictable from five observed frames plus the planned action; and
2. materially more predictable with the correctly paired action than with
   history alone or a different episode's action?

This is a train-only Stage-0 identifiability screen. It is not a video-model
training run and cannot establish better RGB quality, FVD, few-step sampling,
control fidelity, contact recognition, or real-time DAgger. The event maps are
image-change proxies, not semantic object masks or physical contact labels.

## Immutable population and prospective split

Only the 512-row immutable ABC **train** manifest/cache may be opened.
Validation and protected test are unsupported.

Before any selected RGB or measured state is read, registration scans only:

- the train manifest and cache metadata;
- cached planned actions;
- file existence and raw-MCAP camera metadata/calibration.

An eligible row must be a unique train episode with complete cached inputs,
`states.npz`, and a top camera declared as `Intel RealSense D405`. Eligible
clips are sorted by planned future joint-command motion, split into four equal
strata, then ranked within stratum by
`SHA256(clip_id|interaction-event-bottleneck-v1)`. Each stratum contributes 64
fit clips and 16 score clips: **256 fit and 64 score episodes**. Registration
fails if a stratum cannot supply 80 clips. Fit and score episodes are disjoint.

Within each score stratum, the episode-shuffled donor is the next score clip
under the frozen hash order with cyclic wrap. A donor can never equal the
native clip or episode. Selection never uses RGB, measured state, event-map
content, model error, or a previous study outcome.

## Causal inputs and privileged target boundary

For each 13-frame clip, boundaries 0--4 are observed and boundaries 5--12 are
future.

Predictor-available inputs are:

- cached top-view RGB frames 0--4;
- measured robot state at boundaries 0--4, used only to mask the observed
  robot in the causal history maps;
- requested action chunks 4--11, each with five controller samples and the 14
  active ABC coordinates.

Target construction alone may open future RGB boundaries 5--12 and measured
future robot states 5--12. Future RGB, future masks, target PCA coefficients,
and any target-derived statistic are forbidden predictor inputs. The official
YAM geometry and nominal D405 extrinsic are fixed. Per-episode recorded
vertical focal length and native image height set the render FOV; principal
point offset and distortion remain unmodelled.

## Raw event field and 32-dimensional bottleneck

All calculations use the cached 180x320 top view, reduced to a 45x80 work
grid. For transition `t -> t+1`, render the measured robot silhouette at both
boundaries, take their union, dilate it by 12 pixels at 180x320, and exclude
that region. This target-only use of future state prevents robot appearance or
robot-link motion from defining the future event target.

Let `g_t` be grayscale in `[0,1]`, `d = g_{t+1}-g_t`, and `(u,v)` be fixed
Farneback flow at 45x80. With `tau=8/255`, `s=32/255`, and
`e=clip((|d|-tau)/s,0,1)`, the four raw channels are:

1. `positive_change = clip((d-tau)/s,0,1)`;
2. `negative_change = clip((-d-tau)/s,0,1)`;
3. `horizontal_transport = e * clip(u/8,-1,1)`;
4. `vertical_transport = e * clip(v/8,-1,1)`.

Every channel is zero in the dilated robot region and is multiplied by
`0.25 + 0.75 exp(-distance_to_robot/12)` on the 45x80 grid. This fixed spatial
weight emphasizes nonrobot changes near the manipulator while retaining a
quarter-weight far-field control. It does not assert that a near-robot change
is contact.

The four observed transitions produce a raw history tensor `[4,4,45,80]`; the
eight future transitions produce `[8,4,45,80]`. Separate fit256-only randomized
PCA32 transforms compress history and future fields. The future PCA32
coefficient is the proposed interaction-event state. The score episodes never
fit either PCA.

## Models and fixed controls

Planned actions `[8,5,14]` are train-standardized on active dimensions and
compressed by fit256-only PCA32. Two multi-output ridge models predict the
standardized future PCA32 coefficient:

- `history_only_equal_width`: `[history_PCA32, zeros32]`;
- `history_plus_action`: `[history_PCA32, action_PCA32]`.

Both therefore receive a 64-wide design, use the same target, estimator, alpha
grid `{0.1,1,10,100,1000,10000}`, and five-fold fit-only alpha selection.
The zero block has no sample-varying information; equal width does not pretend
that it has equal effective information.

The fitted action model is also evaluated with:

- native aligned action chunks 4--11;
- episode-shuffled, stratum-matched action;
- train-mean action;
- an all-zero raw action (explicitly OOD and diagnostic only);
- nonwrapping within-episode shifts `3:11` and `5:13`.

Shifts diagnose clip-step timing but are not selection alternatives because
ABC state and command streams were independently ceiling-resampled. No arm,
offset, PCA dimension, threshold, or alpha grid may be changed from score
outcomes.

## Metrics, bootstrap, multiplicity, and GO gate

Each error is averaged within episode first. Coefficient error is MSE after
fit-target standardization. Field error decodes a predicted PCA coefficient
through the fixed future PCA and compares it with the complete masked raw event
field. Lower is better.

All paired intervals use one common 10,000-draw episode bootstrap with seed
20260813. Relative improvement is
`100 * (reference_mean - candidate_mean) / reference_mean`.

The six causal comparisons form one prospective family:

- aligned versus history-only, shuffled, and train-mean in coefficient MSE;
- aligned versus the same three references in decoded-field MSE.

Each uses a one-sided Bonferroni lower bound at confidence
`1 - 0.05/6 = 99.1667%`. Every comparison must have:

- point improvement at least **5%**;
- simultaneous lower bound strictly above zero;
- at least **60%** favorable score episodes.

In particular, the mandatory aligned-versus-history and aligned-versus-shuffle
conditions cannot be rescued by another control. Aligned-versus-mean is the
additional action-sensitivity gate. Zero and +/-1 shifts are reported but do
not gate.

A separate two-comparison reconstruction family checks whether the bottleneck
retains substantial event-field structure. The exact score coefficient decoded
through fit PCA (the `oracle_reconstruction`, not an inference input) must beat
both the fit-mean field and the all-zero field by at least **20%**, with a
one-sided Bonferroni 97.5% lower bound above zero and at least 60% favorable
episodes.

`GO_FOR_GENERATOR_SCREEN` requires all eight comparisons. Otherwise the fixed
decision is `STOP_INTERACTION_EVENT_BOTTLENECK`; no Wan integration is
authorized by this study.

## Evidence and immutable audit

The workflow writes a sealed registration before target extraction, exact
selected/donor rows and calibrations, a frozen source copy, input provenance,
derived PCA/model/prediction arrays, per-episode metrics, analysis, completion
receipt, and a read-only audit report. The audit validates hashes and identity
links, source bytes, row counts, train-only flags, fit/score/donor separation,
array geometry, common bootstrap recomputation, and explicit
`protected_test_accessed=false` flags.

## Predeclared limitations

- One train-only population and one split/model seed; bootstrap measures
  episode sampling uncertainty, not retraining variance.
- Nominal robot/camera geometry, centered principal point, no distortion, and
  independently resampled command/state streams.
- Rendering masks robot geometry but cannot remove shadows, grasped-object
  pixels, calibration residuals, or occlusion-boundary artifacts perfectly.
- Farneback and luminance change are observational proxies, not object tracks,
  segmentation, contact, force, or causal intervention labels.
- Deterministic prediction cannot identify irreducible or multimodal futures.
- Passing would justify only a matched, predicted-condition generator screen;
  it would not itself demonstrate video-generation improvement.
