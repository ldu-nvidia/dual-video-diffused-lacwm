# Trajectory-consistent corrected-renderer gate

Date frozen: 2026-08-08

Status: **prospective; no fresh score-pool RGB or measured-state outcome may be
opened before the registration and execution source are committed**

## Question and claim boundary

The previous corrected-renderer study showed that an absolute tracking-residual
ridge predictor improves robot pose and silhouette but does not reliably improve
the rendered robot motion field. This fresh experiment asks three distinct,
predeclared questions:

1. Does an inference-causal, recurrent increment-response predictor improve
   robot-flow EPE over both the native raw-command trajectory and the current
   absolute-ridge trajectory while retaining their spatial alignment?
2. Does a transition-local hybrid that uses the absolute-ridge pose as its
   source anchor and the native raw-command increment as its motion improve the
   same endpoints?
3. Independently of either learned correction, is the complete native raw
   trajectory sufficiently sample-specific to beat a motion-stratum-matched
   shuffled raw trajectory and hold-current controls?

The third question is a raw-geometry fallback gate. It cannot rescue either
learned candidate, and neither learned candidate can rescue it. A pass
authorizes only a fixed causal robot-geometry conditioning screen in Wan. It is
not evidence of better generated RGB, dual diffusion, FVD, policy success, or
real-time DAgger.

## Fresh prospective split and selection

Only the immutable 512-row ABC **train** manifest is allowed. Validation and
protected test are forbidden.

- Manifest rows `0:384` fit and tune every predictor and transform.
- Rows `384:512` are the train-only score pool and are episode-disjoint from
  fit384.
- Registration hash-binds the completed Gate-0c v3 registration and excludes
  all 24 of its selected clip IDs before opening any score-pool RGB or measured
  state. The prior registration contains 103 D405-eligible rows, so exactly 79
  D405 episodes must remain. Any count mismatch aborts registration.
- Registration may inspect only the manifest, cached raw commands, MCAP camera
  metadata, file presence, and the previous selected-ID receipt. It records
  `score_state_opened=false` and `score_rgb_opened=false`.
- The 79 untouched D405 rows are ordered by planned joint-command motion, split
  into three strata with `numpy.array_split`, and eight per stratum are selected
  by SHA-256 rank of
  `clip_id|trajectory-consistent-renderer|20260808`. This produces 24 fresh
  score clips without using an outcome.
- Within each selected stratum, the next selected episode under deterministic
  selection order is the donor. The donor must be a different episode. The same
  donor map is used by every shuffled arm.

The selection, donor map, thresholds, source hash, input hashes, and prior
registration identity are sealed before score state or RGB is opened.

## Causal data contract

Allowed inputs at inference are measured states through boundary 4, raw command
chunks through the observed history, the complete proposed future command
chunks 4--11, the frozen model and train-only normalizers, robot geometry, and
camera calibration. Future measured states at boundaries 5--12 and future RGB
are target/scoring-only. They are never a predictor feature or a recursive
state input.

Every predicted trajectory starts at native measured state `q4`. Gripper
coordinates are retained unclipped in artifacts and clipped to `[0,1]` only at
the render boundary.

## Frozen predictors

### Current absolute-ridge reference

This exactly refits the Gate-0c model on fit384: independently standardized
PCA64 history and complete-future-action features, concatenated and
standardized, followed by multi-output ridge on the eight absolute tracking
residuals. Alpha is selected from
`{0.1,1,10,100,1000,10000}` by fixed five-fold fit384 CV. It is a mandatory
reference, not a candidate selected from the fresh outcomes.

### Recurrent integrated delta-response candidate

For future transition `t=0..7`, define

\[
\Delta q^{raw}_t=q^{raw}_{t+1}-q^{raw}_{t},\qquad
\Delta q^*_t=q^{meas}_{t+1}-q^{meas}_{t}.
\]

The shared transition model predicts
`Delta q* - Delta qraw`. Its causal features are frozen fit-only PCA64 history
and full-action context, the native local five-command chunk, raw endpoint and
raw increment, an eight-way horizon indicator, the current **predicted** state,
its offset from the raw source anchor, and the previous predicted increment
residual. No future measured state appears in a feature.

Inference is a closed-loop rollout:

\[
\hat q_4=q^{meas}_4,\qquad
\hat q_{t+1}=\hat q_t+\Delta q^{raw}_t+
g\,f_\phi(H,A_t,\hat q_t,t).
\]

The ridge alpha and shrinkage `g` are selected only by fixed five-fold fit384
clip CV using a predeclared composite of joint increment MSE and integrated
joint-pose MSE. The candidate is refit through three fixed self-rollout rounds:
round zero uses the raw causal rollout state, and rounds one and two rebuild
training features only from the preceding model's predictions. Thus future
measured state remains target-only even during refitting. Joint coordinates
determine the CV objective; grippers are learned and reported but cannot select
the motion model.

### Hybrid anchor/raw-delta candidate

For each transition independently, its source pose is the current
absolute-ridge prediction and its target pose is

\[
q^{hybrid}_{t+1}=q^{abs}_{t}+\Delta q^{raw}_t.
\]

This is deliberately a transition-local mechanistic attribution, not a
coherent recurrent trajectory: every next transition is re-anchored to the
absolute-ridge pose. Silhouette/RGB at target `t+1` use the hybrid target pose;
flow uses the paired absolute-ridge source and hybrid target. It tests the
specific Gate-0c observation that ridge supplied better absolute anchoring
while raw increments sometimes preserved motion better. This arm is frozen
before outcomes and is never synthesized post hoc.

## Frozen arms

| Arm | Target/source construction | Role |
|---|---|---|
| `raw_command` | native raw endpoints, starting at native q4 | fixed causal candidate/reference |
| `raw_episode_shuffled` | donor complete raw endpoints, native q4 start | raw sample-attribution control |
| `absolute_ridge` | native raw endpoints plus current aligned ridge residual | mandatory learned reference |
| `recurrent_delta` | native causal closed-loop integrated delta rollout | co-primary learned candidate |
| `recurrent_delta_shuffled` | native history/q4 with donor future action plan through the same recurrent model | learned sample-attribution control |
| `hybrid_anchor_raw_delta` | absolute-ridge source anchor plus native raw increment per transition | co-primary mechanistic candidate |
| `hold_current` | native q4 repeated | negative control |
| `measured_oracle` | measured future state | privileged scoring diagnostic only |

For the shuffled learned arm, only future action-derived inputs are donated;
recipient history and q4 remain native. The donor's future measured state is
never opened or supplied.

## Renderer and metrics

Rendering and scoring retain Gate-0c's frozen geometry contract: official ABC
commit `6bc6586721cf0c409ccee80f675a28de9b9b2f5e`, robot-only YAM visuals,
nominal top D405 extrinsic, recorded `fy`, centered principal point, no
distortion, and articulated descendants only. Native resolution is preserved.

The fixed scoring region is the measured-oracle articulated mask. Metrics are:

- primary robot-only flow EPE from exact geom-local point transport on
  oracle-visible articulated pixels, stride two, minimum oracle motion 0.25 px;
- silhouette boundary Chamfer and IoU against the oracle render;
- observed RGB-edge Chamfer inside the fixed 12-pixel oracle robot band.

Transitions with no qualifying moving support are excluded symmetrically and
clip flow EPE is pooled by qualifying oracle pixels. Each clip must retain at
least ten qualifying pixels. Future RGB and measured state are scoring-only.

## Multiplicity and immutable decisions

All effects use the same 20,000 paired clip-bootstrap index matrix, seed
20260808. Positive differences favor the named candidate. Seven primary flow
contrasts form one global family:

1. recurrent delta versus raw;
2. recurrent delta versus absolute ridge;
3. recurrent delta versus recurrent-delta shuffled;
4. hybrid versus raw;
5. hybrid versus absolute ridge;
6. raw versus raw-trajectory shuffled;
7. raw versus hold current.

For each contrast, its one-sided bootstrap p-value is
`(1 + count(bootstrap_mean <= 0))/(B + 1)`. Contrasts are ordered by this
p-value and assigned the Holm step-down alpha `0.05/(m-rank+1)`. The reported
Holm lower bound is the corresponding bootstrap quantile. A primary contrast
passes only if the Holm procedure has not stopped before it, its Holm lower
bound is strictly positive, its point gain is positive, and at least 60% of
clips favor the candidate. The two candidate-versus-raw/reference contrasts
and both raw-fallback contrasts additionally require at least 5% relative flow
improvement. Recurrent-versus-shuffled requires positive attribution but not a
5% margin.

Spatial retention is not allowed to trade against flow. For recurrent and
hybrid versus both raw and absolute ridge, silhouette Chamfer and fixed-band
RGB Chamfer must have nonnegative ordinary paired 95% lower bounds, and IoU's
lower bound must be nonnegative. For raw fallback versus both shuffled raw and
hold, silhouette/RGB point effects and lower bounds must be strictly positive,
and IoU must have a positive point effect with a nonnegative lower bound.

Decisions are independent:

- `recurrent_delta_pass` requires all recurrent primary and retention tests;
- `hybrid_anchor_raw_delta_pass` requires all hybrid primary and retention
  tests;
- `raw_geometry_scaffold_pass` requires all raw-fallback primary and spatial
  tests.

No family can rescue another. Wan may receive only the arm(s) whose own frozen
gate passes. If none passes, the decision is `STOP_ALL_GEOMETRY_HANDOFFS`.

## Artifact and audit contract

The workflow writes a sealed registration before outcome access, then exact
fit/score provenance, model state and predictions, hash-addressed clip bundles,
frame/transition and clip-level rows, Holm calculations, analysis and completion
receipts, and overlays. The read-only audit must recompute row counts, paired
effects, the common bootstrap/Holm family, every gate, identities, file hashes,
freshness relative to Gate-0c, donor constraints, fit/score disjointness, flow
support, and explicit `protected_test_accessed=false` flags.

Partial or failed attempts remain immutable and are never promoted as completed
evidence.

## Predeclared limitations

- Twenty-four deliberately stratified train-only episodes and one fixed seed.
- Fit384 CV informs the compact model but this is not a public benchmark.
- Raw endpoints are zero-order-hold command proxies, not controller simulation.
- State/action streams are independently ceiling-resampled and do not preserve
  raw controller timestamps.
- Nominal extrinsics, centered principal point, and no D405 distortion.
- Robot-only geometry excludes objects, contacts, scene occlusion, and
  stochastic future effects.
- The hybrid is transition-local and must not be presented as a coherent
  rollout.
- A passing renderer gate authorizes a controlled Wan screen only; generated
  video and end-to-end latency remain unmeasured.
