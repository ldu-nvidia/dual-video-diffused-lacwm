# Fixed causal robot-flow Wan screen

Date frozen: 2026-08-08

Status: prospective implementation; launch is forbidden unless the sealed
predicted tracking-corrected renderer attribution study returns
`GO_FOR_WAN_SCREEN`

## Question and claim boundary

Does a deterministic robot-motion field, computed only from the observed
history, proposed actions, a train-fitted command-tracking model, robot
geometry, and fixed camera calibration, improve an action-conditioned Wan
video model at one, two, or four transformer calls?

This is a fixed-conditioning feasibility screen. It is not dual diffusion:
the robot field has no noise clock, no velocity target, and no sampling calls.
Passing authorizes a separately preregistered stochastic object/contact
residual experiment. It does not establish a video-generation, FVD, DAgger,
or paper-level claim.

## Hard prerequisite

The exact committed renderer study must be complete, pass its read-only audit,
and report `GO_FOR_WAN_SCREEN`. Its source, registration, preparation,
analysis, completion, audit, predictor state, official ABC commit, and model
assets are content-bound in this study's registration. A renderer stop, an
incomplete artifact, or any changed threshold prevents cache construction and
training.

## Causal condition

For each clip, the train-fitted predictor receives measured robot state only
through observed video frame 4 and the candidate action chunks. It predicts
the eight future command-tracking residuals. The causal pose path is the
observed frame-4 pose followed by the eight raw command endpoints plus those
predicted residuals. Future measured states and future RGB are neither read nor
accepted by the cache builder for validation clips.

At each of the eight future transitions, articulated visible robot pixels are
transported analytically between two MuJoCo poses. The per-pixel field is

\[
  f_t(p)=\left(\Delta x/W,\;\Delta y/H,\;m,\;
  m\log\frac{z_{t+1}+10^{-6}}{z_t+10^{-6}}\right),
\]

where `m` is one only when the source robot surface is valid, its transported
point remains in frame, and it is z-buffer visible in the target render.
Invalid values are exactly zero. Each native 180-by-320 top-camera field is
bottom-padded to 192-by-320 exactly like the Wan VAE, area-pooled by eight to
24-by-40, and inserted into columns 0:40 of the three-view 24-by-120 grid. The
two unvalidated wrist views are exactly zero.

Wan has four temporal latents for 13 RGB frames. Latents 0 and 1 correspond to
the observed five-frame history and receive exact-zero flow. Latent 2 packs
the four transitions 4->5 through 7->8; latent 3 packs transitions 8->9
through 11->12. Four components times four sub-transitions produce the fixed
condition shape `[B,16,4,24,120]`. Packing and view placement are tested by an
exact round trip.

All flow scaling constants are analytic (`W=320`, `H=180`); no validation
normalization is fit. Grippers are clipped to `[0,1]` only at the renderer
boundary. Cache metadata binds every row, input, predictor, calibration,
source commit, tensor hash, clipping count, visibility count, and render
latency. Protected test data are unsupported.

## Frozen data and cache controls

- Training uses the immutable ABC train512 cache and manifest.
- Development evaluation uses the existing immutable val64 cache and
  manifest. It is not a lockbox and supports only an exploratory decision.
- The tracking predictor is fit only on train indices 0:384 under the already
  frozen PCA64/ridge contract. The same sealed parameters are applied to all
  train512 and val64 clips.
- Camera eligibility is frozen from MCAP metadata before any generator outcome.
  D405 rows receive the rendered field. Non-D405 rows remain in the identical
  train/validation stream but receive an exact-zero 16-channel field and an
  explicit ineligible flag; they are never rendered using fabricated D405
  calibration. A D405 row without raw state history, exact cached/raw action
  equality, nonempty articulated support, or the registered geometry is fatal.
  There is no retry or substitution. At least 32 val64 episodes must be D405 or
  the screen stops before training.

The cache builder emits aligned flow plus three deterministic controls:

| Source | Construction | Purpose |
|---|---|---|
| `aligned` | local predicted-corrected render | deployable candidate |
| `episode_shuffled` | another episode's complete aligned tensor, fixed cyclic donor | sample attribution |
| `timeshift_plus_one` | local transitions shifted one future step without wrap; last is zero | timing attribution |
| `wrong_calibration` | local trajectory rendered after a fixed +5 cm local camera-x perturbation | geometry attribution |

Donors are fixed before video outcomes, episode-disjoint, D405-to-D405, and
matched by a train-only planned-motion tertile. Validation donors are
constructed within the registered D405 subset of val64 without reading
validation RGB or measured future state. Non-D405 control tensors are all
exact zero. The wrong calibration perturbation is fixed before cache
generation and is never tuned.

## Matched training arms

Both arms start from the exact frozen VPM update-1,000 snapshot, use seed 1234,
the same global batch, clip order, augmentation, video noise, timestep draws,
optimizer, learning-rate schedule, 200 updates, and one Wan call per training
example. Both instantiate the same 16-channel Conv3d adapter and unused
auxiliary head. The parent auxiliary modules are excluded identically at load.

The parent video corruption path is preserved exactly: during training,
history latents follow the same forward-noise draw as future latents; during
sampling, known history follows `(1-sigma) * reference + sigma * initial_noise`
and reaches the clean reference only at `sigma=0`. Clean-clamping history at
intermediate calls is forbidden because it changes the parent state
distribution even when flow fusion is off.

| Arm | Training condition | Difference |
|---|---|---|
| `FLOW-OFF` | aligned tensor is projected but hard-masked at the Wan seam | parameter-matched video control |
| `FLOW-ON` | aligned tensor is injected through the trainable bounded residual gate | fixed causal condition |

The fixed flow clock is always zero; auxiliary loss is zero; the flow tensor is
never corrupted, predicted, or updated. Both traces must hash-identically match
all 200 clip/timestep/video-noise/action/flow batches. Trainable parameter
names and counts must match. `FLOW-OFF` must reproduce an explicit off call
bit-for-bit.

## Evaluation grid and metrics

Use every val64 clip and four stateless video-noise seeds. Clean future video
is evaluator-owned and is opened only after every causal endpoint for that
clip/noise pair has materialized. Evaluate:

- `FLOW-OFF` at NFE 1/2/4 with condition hard-off;
- `FLOW-ON` at NFE 1/2/4 with aligned flow;
- the same `FLOW-ON` checkpoint at NFE 1/2/4 with condition off,
  episode-shuffled, time-shifted, and wrong-calibration flow.

The primary population is the prospectively registered D405 subset of val64,
and the primary spatial scope is its top view (pixel columns 0:320; latent
columns 0:40), because only that camera passed the prerequisite. The full
val64 population and all-three-view metrics are secondary guardrails. For each
scope report latent NMSE, decoded MSE in `[0,1]`, temporal-difference MSE
including the observed-to-first-future boundary, LPIPS, and action
sensitivity. Report adapter, render, VAE, Wan, decoder, and end-to-end latency
separately. Confidence intervals use 10,000 paired episode-clustered bootstrap
samples with frozen seed 20260831.

## Frozen development decision

The primary endpoint is top-view decoded MSE at NFE 1. `ADVANCE_TO_RESIDUAL`
requires all conditions:

1. aligned `FLOW-ON` improves top-view decoded and temporal MSE by at least 3%
   over matched `FLOW-OFF`, with paired 95% lower bounds above 1%;
2. aligned `FLOW-ON` beats its same-checkpoint off, shuffled, time-shifted, and
   wrong-calibration controls on both metrics, with positive paired lower
   bounds and at least 1% point improvement;
3. top-view LPIPS improves with a positive paired lower bound;
4. all-view decoded and temporal MSE have nonnegative point effects and lower
   bounds above -1%;
5. latent NMSE has a nonnegative point effect and lower bound above -1%;
6. aligned action sensitivity exceeds shuffled-action sensitivity with a
   positive paired lower bound;
7. exactly one Wan call is reported, no teacher/flow-model calls occur, no
   future RGB or measured future state enters any sampler or cache predictor,
   the full paired training trace passes, and protected test remains unopened.

NFE 2/4 are secondary dose-response checks. If NFE 1 fails, a higher-NFE gain
does not authorize the real-time residual direction. Otherwise the decision is
`STOP_FIXED_FLOW`; no stochastic residual diffusion or distillation is
launched from this mechanism.

## Next stage if and only if this passes

The next state must model object/contact uncertainty rather than duplicate
known robot kinematics. It will use the deterministic robot field as a frozen
scaffold and generate a severe, layered residual (object SE(3), particles,
contact mode, visibility, or another preregistered bottleneck) from history and
actions before RGB. Clean future flow may supervise it but may never condition
deployment. Generated-versus-zero, generated-versus-shuffled, and oracle
attribution remain mandatory at equal total calls and latency.
