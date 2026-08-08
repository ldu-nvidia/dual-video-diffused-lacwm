# Physics-anchored residual-flow protocol

Date: 2026-08-08

Status: prospective protocol selected after the learned dense-flow Stage-0
`NO_GO`; no generator-quality claim is made here

## Research question

Can an inference-causal motion state improve low-call action-conditioned video
generation when clean features of the unknown future video are unavailable?

The proposed factorization is

\[
u_{robot}=G(q_0,a_{1:T},\mathcal R,\mathcal C),\qquad
r\sim p_\psi(r\mid h,a,u_{robot}),\qquad
x\sim p_\theta(x\mid h,a,u_{robot}+r).
\]

`G` is deterministic kinematic rendering. It supplies only motion implied by
the known robot, proposed command sequence, and calibrated cameras. The small
stochastic state `r` represents object motion, contact consequences,
command-tracking error, and calibration/model residuals. It must be generated
before RGB and then held fixed. Neither state may read the future target video
at inference.

The learned ABC dense-flow proxy is not substituted for `G`: it improved dense
MSE only 2.91% over history and 2.94% over shuffled actions, below both
preregistered 10% gates. Its positive directional-cosine signal motivates a
kinematic scaffold but does not make its predicted field clean enough for the
generator handoff.

## Causal data contract

At the time a candidate trajectory is scored, the allowed inputs are:

- observed RGB history `h` through frame 4;
- measured current joint state `q0`;
- the complete proposed action/joint trajectory for the rollout horizon;
- a versioned robot model, link geometry, camera calibration, and frame graph;
- train-split statistics and model parameters.

Future RGB, future optical flow extracted from RGB, future executed joint state,
target V-JEPA/TF features, object masks derived from future RGB, and future
contact labels are forbidden predictor inputs. They may be training targets or
explicitly labelled oracle controls only. If commands do not uniquely determine
future joint state, `G` must propagate command-tracking uncertainty rather than
replay recorded future states.

Train, development validation, and final lockbox episodes must remain disjoint.
All transforms, calibration fitting, PCA/codebooks, normalizers, and thresholds
are fit or frozen without the lockbox.

## Deterministic state construction

For each future time `t`, use forward kinematics and a z-buffered mesh render to
obtain the visible robot surface point `P_t(p)` and link identity at pixel `p`.
Transport that same surface point with its link to time `t+1`, project it through
camera `C`, and define

\[
u_t(p)=\pi_C(P_{t+1}(p))-p.
\]

The minimal per-view field contains:

1. signed horizontal displacement `u_x/W`;
2. signed vertical displacement `u_y/H`;
3. visibility/validity in `{0,1}`;
4. normalized depth change `log((z_{t+1}+eps)/(z_t+eps))`.

Occluded, newly revealed, and out-of-frame points are invalid rather than zero
motion. Keep values signed; HSV flow images and magnitude-only features discard
direction or introduce a second image codec. Rasterize directly at a modest
resolution, then area-pool each view independently to the Wan latent grid. For
the current ABC layout, concatenate the three views only after per-view
rendering and normalization; never filter across a view seam.

The existing low-cost LACWM seam can project a tensor
`[B,4,T_lat,H_lat,W_lat]` to Wan tokens with a small Conv3d adapter. Set its
clock to clean (`sigma_flow=0`), hold the state fixed, predict no flow velocity,
and keep the RGB transformer-call count unchanged. This is a conditioning
screen, not yet dual diffusion. A full separate flow-token stream with joint
self-attention is justified only after the cheap screen passes.

## Gate 0: geometry and timing

Start with the 456 ABC D405 train/validation clips that have all three
intrinsics, top camera first. Pin the official YAM MJCF/meshes and nominal D405
transform. Re-extract raw MCAP timestamps and joint trajectories.

On observed transitions, compare the rendered robot mask/flow against image
evidence using:

- silhouette edge distance and soft mask IoU;
- signed flow cosine and endpoint error within the rendered mask;
- correct calibration versus wrong-station calibration;
- aligned trajectory versus +/-1, +/-2-frame shifts and episode-shuffled
  trajectories;
- median and p95 render-to-latent latency.

Proceed only if aligned rendering beats every wrong-calibration and time-shift
control with a positive paired lower bound and p95 preprocessing is below 20 ms.
Failure means calibration must be fitted on train data or the dataset changed;
it does not authorize an RGB generator run.

### First nominal-geometry diagnostic (2026-08-08)

A deliberately bounded train-only probe passed its registered diagnostic on
three clips and 39 frames. Correct-pose YAM rendering reached 8.444 px mean
edge Chamfer versus 13.025 px for a +4-clip-step wrong-pose control. The paired
shifted-minus-aligned mean was +4.581 px with 95% interval [3.320, 5.865], all
three clip means were positive, 3-pixel edge support improved by 5.902
percentage points, and render latency was 2.769 ms p95 at 640x480. This supports
constructing the planned-action geometry scaffold.

It does **not** close Gate 0 in full. The probe uses nominal extrinsics, a
centered principal point, no D405 distortion, observed image edges rather than
robot masks, and a coarse approximately 0.667-second control. Before Gate 1,
fit/refine calibration on train data only and run non-wrapping fine time shifts,
wrong-calibration controls, and planned-action controller replay. Full evidence
and hashes are in `ABC_D405_NOMINAL_GEOMETRY_PROBE.md`.

A post-hoc signed sensitivity has since run on the same three clips. Non-wrap
Chamfer deltas for shifts `[-2,-1,+1,+2]` were respectively
`[+2.114,+1.068,+0.537,+2.029]` px; all paired lower bounds were positive
except `+1`, whose interval was `[-0.018,1.108]`. This directional asymmetry
keeps timestamp interpolation/fine timing explicitly open before Gate 1.

### Gate 0b: predicted command-tracking correction (2026-08-08)

The leakage-controlled train512/validation64 Stage-0 passed all four registered
10% gates. A ridge predictor using measured history through frame 4 and planned
commands only reached standardized residual MSE 0.371876, improving 60.08%
`[54.38,65.04]` over history-only, 72.45% `[67.29,77.18]` over episode-
shuffled actions, 68.52% `[63.73,72.35]` over raw-command/zero residual, and
97.39% `[96.22,98.08]` over hold-current. Future measured states were target-
only. Joint tracking residual RMS was 0.06180 rad (3.54 degrees), and candidate
joint RMSE was 0.03305 rad (1.89 degrees).

This compact state can refine `u_robot`, but it is not yet the stochastic
object/contact residual `r`. It also uses independently ceiling-resampled raw
state/command streams and does not replay a controller. Before Gate 1, render
raw-command, predicted-corrected, hold-current, shuffled-correction, and
measured-state-oracle trajectories. Proceed only if the predicted correction
improves silhouette/flow alignment over raw and shuffled controls with positive
paired lower bounds. Full evidence is in
`NOMINAL_TRACKING_RESIDUAL_STAGE0_RESULT.md`.

## Gate 1: fixed causal flow conditioning

Use one parent snapshot, identical data order/noise/optimizer, equal trainable
parameter count, equal updates, and equal Wan calls. The mandatory arms are:

| Arm | Flow supplied to RGB | Purpose |
|---|---|---|
| `FLOW-OFF` | exact masked no-op | video-only reference |
| `FLOW-CAUSAL` | aligned rendered robot flow | deployable candidate |
| `FLOW-SHUFFLED` | another episode's marginally matched flow | sample attribution |
| `FLOW-TIMESHIFT` | same episode, wrong trajectory time | temporal attribution |
| `FLOW-WRONG-CAL` | deliberately perturbed extrinsic | geometry attribution |
| `FLOW-ORACLE` | target-video full-scene flow | non-deployable ceiling only |

Evaluate NFE 1/2/4 without spending a transformer call on flow. Register one
primary endpoint before training. A reasonable development gate is:

- at least 3% improvement in both decoded and temporal MSE, with paired 95%
  lower bounds above 1%;
- nonnegative latent-NMSE point effect and lower bound above -1%;
- `FLOW-CAUSAL` materially better than shuffled, time-shifted, and wrong-cal;
- causal retention of at least 25% of the oracle improvement;
- full render + adapter + Wan + decoder latency reported, not model-only time.

FVD/perceptual metrics, long rollouts, and a lockbox are reserved until this
development gate passes. Reusing val64 makes the current phase exploratory.

## Gate 2: stochastic residual motion

Only after `FLOW-CAUSAL` improves RGB, define a training target from future RGB
flow:

\[
r^* = M_{valid}\odot (u_{scene} - u_{robot}).
\]

Use confidence and occlusion channels so RAFT/Farneback errors are not treated
as physical residuals. Train a small residual-state flow/consistency model from
`h,a,u_robot`; future RGB creates `r*` for supervision only. During RGB
training, progressively replace clean `r*` with stopped, self-generated
`r_hat`, ending with the inference distribution. This avoids the exact
oracle-to-autonomous mismatch that invalidated clean V-JEPA/TF conditioning.

Inference is strictly ordered:

1. render `u_robot`;
2. generate `r_hat` in one or two small-model calls;
3. freeze `u_robot+r_hat`;
4. denoise RGB in one, two, or four Wan calls.

Compare generated residuals against zero, train mean, within-episode time shift,
episode shuffle, and clean oracle residual at the same total call and latency
budget. Require generated-residual attribution; an oracle-only gain is not a
deployable result.

## Gate 3: few-call deployment

Distill only a teacher that passed Gate 1 or Gate 2. Train the student on its own
generated intermediate states (self-forcing/consistency), not only clean teacher
states. Compare two and four total Wan calls with the same causal scaffold.
Report:

- latent NMSE, decoded MSE, temporal MSE, perceptual quality, and FVD;
- action sensitivity and shuffled-action controls;
- 10th/50th/95th-percentile end-to-end latency and peak memory;
- long-horizon drift and closed-loop task success under DAgger;
- latent-only policy latency separately from decoded-video latency.

The operational success condition is 5--10 Hz end-to-end candidate evaluation
with no material task-success loss, not merely a fast DiT kernel.

## Stop rules and claim boundary

- Do not launch residual diffusion if fixed causal flow does not improve RGB.
- Do not distill a teacher that is worse than the NFE-1 video-only frontier.
- Do not call future-RGB flow, V-JEPA, or TF a deployable condition.
- Do not claim “optical-flow-conditioned video” as the innovation by itself.
- Do not access the protected lockbox until a mechanism is frozen.

The potentially novel contribution is the factorization of known embodiment
motion and an **explicitly generated stochastic** contact/object residual,
validated at few-call closed-loop robotics latency—not flow or robot rendering
in isolation. [FlowWAM](https://arxiv.org/html/2607.13017) already covers dual
RGB/flow modeling and fixed desired-flow world-model conditioning;
[RealWonder](https://arxiv.org/html/2603.05449) covers physics flow plus a
four-step video generator; and
[Robot-Factored World Models](https://arxiv.org/html/2607.22535) already covers
the deployment-available nominal-controller trajectory, URDF mesh rendering,
end-effector/scene depth, and Wan latent conditioning. In addition,
[OSCAR](https://arxiv.org/html/2606.04463) covers 2-D kinematic skeleton
conditioning, [ContactFlow](https://arxiv.org/html/2607.26579) covers sparse
3-D object-contact trajectories rendered as a seven-channel control video,
[iMaC](https://arxiv.org/html/2606.09813) covers URDF/FK motion and deterministic
contact-distance images, and [EA-WM](https://arxiv.org/html/2605.06192) covers
event-supervised gated fusion of kinematic visual fields. Gate 1 is therefore a
reproduction/feasibility baseline. A research claim requires a generated
interaction residual beyond deterministic geometry/contact intent, clear
generated-versus-oracle attribution, and measured few-call closed-loop utility.
