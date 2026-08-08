# Deployable dual-video diffusion: twelve-hour research report

Date: 2026-08-08

Scope: low-NFE action-conditioned ABC/LACWM development screens; one seed,
train512/val64 unless stated; no protected-test access

## Executive conclusion

The absence of a clean encoder feature from unknown future video does **not**
make dual video diffusion impossible. It makes oracle future-video conditioning
non-deployable. An auxiliary must instead be generated before RGB, computed
causally from known inputs, or discarded after serving as training supervision.

The completed screens do not support autonomous V-JEPA/TF features, generated
frequency states, one-call scratchpads, explicit action tokens, or the tested
training-only relation/spectrum losses as quality-improving mechanisms at this
budget. The most credible remaining **technical** route is:

1. compute robot-only image motion from the proposed joint trajectory, robot
   model, and calibrated cameras before RGB denoising;
2. hold that causal flow scaffold fixed while RGB is generated;
3. if needed, generate a small stochastic residual-flow state for object/contact
   consequences before RGB;
4. distill a demonstrably stronger flow-conditioned teacher to two or four RGB
   calls with inference-consistent/self-forced training.

This is not merely conceptual. A leakage-controlled train512/val64 proxy screen
found that planned actions add statistically positive information about future
per-view motion summaries. It is still too weak to authorize generator training,
and exact ABC rendering first required a robot/camera calibration diagnostic.
That first train-only diagnostic now passes: nominal top-camera YAM rendering
is significantly closer to observed image edges than a +4-clip-step wrong-pose
control and is cheap enough to be causal. It remains a three-clip nominal-
calibration feasibility result, not evidence that rendered flow improves Wan.
A current literature audit also shows that nominal robot rendering is now an
important baseline rather than a standalone novelty claim; the publishable
increment would need generated uncertainty/contact residuals, privileged-to-
causal transfer, or a demonstrated low-call closed-loop advantage.

## Causal taxonomy

| Mechanism | Future clean video needed at inference? | Status |
|---|---:|---|
| target-video TF/V-JEPA condition | yes | oracle upper bound only |
| autonomous semantic/frequency branch before RGB | no | tested variants negative |
| same-call midpoint scratchpad | no | tested variant negative and sample-insensitive |
| training-only relation/spectrum/motion loss | no | TRD, TFREG, and low-frequency motion regularization negative |
| learned action-to-dense-flow proxy | no | real direction signal, but failed the strict generator-handoff gate |
| nominal D405 robot geometry | no | first three-clip train-only alignment diagnostic passed; finer timing/calibration and generator gates remain |
| analytic planned-action robot-only flow | no | strongest next generator-conditioning baseline; not yet evaluated in Wan |
| physics-anchored stochastic residual flow | no | prospective genuine dual-diffusion extension |
| privileged teacher to on-policy video-only student | no | strongest training-only salvage of the oracle gain; newly frozen, not yet run |
| consistency/self-forcing distillation | no auxiliary required | primary speed route after a stronger teacher exists |
| latent-only policy adapter | no decoded RGB for policy | measured systems fallback near 8.49 rollouts/s p95 |

## Controlled evidence

Positive numbers below mean lower error. All are development evidence, not FVD,
long-horizon, protected-test, or closed-loop task claims.

| Screen | Main NFE-1 result | Mechanism diagnosis |
|---|---|---|
| generated V-JEPA/full-clip state | fusion-on approximately equalled off/shuffled; flow training worse | autonomous feature lacked useful causal identity |
| generated Haar forcing | best joint arm -6.39% latent, -3.86% decoded, -0.74% temporal | synchronous auxiliary output arrives too late for the same Wan call |
| causal pre-Wan motion plan | -0.33%/-0.94%/-0.22%; aligned approximately shuffled | planner itself was weak: NMSE 1.548, cosine 0.166 |
| midpoint generated scratchpad | -0.18%/-1.13%/-0.26%; aligned approximately shuffled | nonzero injection, no sample-specific use |
| action-delta residual | -1.06%/-1.09%/-0.023% | gate/action path collapsed |
| fixed-dose ordered action tokens | +0.405% latent, +0.259% decoded, -0.086% temporal; all gates failed | route opened, but action-specific effect remained near zero |
| VideoREPA relation loss, 0.5/0.05 | teacher relation improved 90.2%/84.1%, while deployable video worsened at both doses | representation imitation conflicted with or failed to help flow learning |
| TF spectral loss, 0.05 | -0.564%/-0.134%/-0.110% | zero inference dependency, but no quality gain; exploratory geometry mismatch |
| per-view low-frequency motion loss, 0.05 | -0.256% latent, +0.662% decoded, +0.099% temporal; every registered gate failed | tiny uncertain appearance/motion points, latent regression, and the same effect under aligned/shuffled/zero actions |
| action-to-Farneback-summary proxy | +6.59% versus history-only; +9.04% versus shuffled actions | planned actions contain incremental motion information, but target is coarse |
| action-to-dense-top-flow proxy | +2.91% dense MSE versus history and +2.94% versus shuffled; cosine 0.040 to 0.216 | genuine action-specific direction, but both preregistered 10% handoff gates failed |
| confidence gate over midpoint scratchpad | no valid inference confidence was preserved; a forbidden perfect temporal oracle gained only +0.0447% | always-off is the only honest no-regret policy for this auxiliary |
| nominal D405/YAM geometry | correct-pose Chamfer 8.444 px versus 13.025 px at +4 clip steps; shifted-minus-aligned +4.581 px [3.320, 5.865] | narrow train-only feasibility pass; 3/3 clips positive and render p95 2.769 ms, but no masks, calibrated distortion, flow, or video-generation endpoint |

## Why image Latent Forcing and these video attempts diverged

Image Latent Forcing first denoises its auxiliary latent and then supplies that
generated, relatively clean latent while pixels denoise. Its best reported
cascade allocates 25 Heun steps to the latent and 25 to pixels. A synchronous
one-call video implementation instead lets RGB see only auxiliary input noise;
the clean auxiliary estimate is emitted after the RGB velocity has already been
computed. Compressing 13 RGB frames into four Wan temporal tokens further makes
motion features coarse, and the existing LACWM action interface collapses much
of its sample-specific variation.

The correct analogue is causal ordering, not access to oracle future features:

\[
\hat u \sim p(u\mid h,a),\qquad
\hat x\sim p(x\mid h,a,\hat u).
\]

The action-derived-flow route makes the first state easier by separating known
embodiment kinematics from unknown scene consequences.

## Action-derived flow feasibility

The released FlowWAM WorldArena implementation is external evidence for the
mechanism: it replays a known future joint trajectory in SAPIEN, computes
robot-only optical flow, VAE-encodes that flow, holds it clean/fixed, and lets
RGB and flow interact through joint self-attention. Its default 50-step pipeline
is not real-time evidence. RealWonder independently combines simulator-derived
flow/coarse RGB with a distilled four-step causal generator, but at far greater
training scale.

Our raw ABC audit found 576/576 train/val MCAPs, but no per-episode robot model,
base pose, camera extrinsics/TF, depth, masks, or attachments. The official ABC
repository provides nominal YAM MJCF/meshes and D405 transforms. The next
defensible subset is 456 D405 clips with all three intrinsics, top view first.
Rendered silhouettes must beat time-shifted and wrong-calibration controls
before the resulting flow is called pixel-aligned.

A bounded first probe has now checked that premise on three explicitly selected
train clips and 39 frames. Correct-pose robot-only rendering reached 8.444 px
mean edge Chamfer versus 13.025 px under a cyclic +4-clip-step pose control;
the paired shifted-minus-aligned difference was +4.581 px with 95% frame
bootstrap interval [3.320, 5.865], all three clip means were positive, and
3-pixel edge support improved by 5.902 percentage points. A no-wrap sensitivity
check gave +4.790 px [3.067, 6.525]. Render-only latency was 2.415 ms mean and
2.769 ms p95 at 640x480. This authorizes building a deterministic segmentation,
depth, link-ID, and robot-flow scaffold for a controlled screen. It does not
complete the full calibration gate: the camera model still uses nominal
extrinsics, a centered principal point, and no D405 distortion; observed edges
are not robot masks; and the +4 control is roughly 0.667 seconds rather than a
sub-frame perturbation. A post-hoc signed timing audit sharpened this warning:
non-wrapping ±2-step controls passed, as did -1 step, but the +1-step
approximately 0.167-second interval narrowly crossed zero (`[-0.018,1.108]`
px). The direction asymmetry is compatible with the known next/ceiling state
resampling but is not causally identified. The immutable evidence is documented in
`ABC_D405_NOMINAL_GEOMETRY_PROBE.md`.

The current DROID LeRobot cache is also not renderer-ready: it lacks
intrinsics/extrinsics, joint positions, robot geometry, depth, and masks, and
stores Cartesian end-effector actions. More urgently, its metadata places the
gripper at state index 7 while the current loader reads index 6; sampled index 6
is zero and index 7 varies. That action-loader contract must be corrected and
revalidated before DROID is used for the next causal-motion experiment.

The completed causal proxy used only observed frames 0--4 and planned action
chunks 4--11 as predictor inputs. Future RGB produced eight low-resolution
Farneback target transitions for training/scoring only. On val64:

- history plus aligned actions: standardized MSE 0.8947;
- history only: 0.9578, so aligned actions improve 6.59% [4.96, 8.05];
- same action model with episode-shuffled actions: 0.9837, so alignment improves
  9.04% [7.19, 11.10];
- aggregate standardized R2 rises from 0.098 to 0.158.

This passes its preregistered exploratory 1% gates. It does not meet the later
10% all-view integration threshold, and 84% of standardized target variance
remains unexplained. It motivates top-view dense/calibrated flow; it is not a
clean-enough learned condition or a video-quality result by itself.

The diagnostic top-camera slice improved 12.61% [10.23, 15.08], whereas the
left/right wrist slices improved only 3.57%/3.18%. Signed horizontal/vertical
motion carried about 9.34%/9.83% gain, compared with 2.82%/2.69% for mean/q75
magnitude. This is consistent with wrist ego-motion contaminating the target
and supports a top-view directional-flow first experiment. Because the slice
was not the registered all-view endpoint, it is hypothesis-selection evidence,
not a passing confirmatory gate.

The preregistered dense top-view follow-up did not reproduce that coarse 12.61%
effect. It predicted eight raw `(u,v)` Farneback fields at `12x20` from observed
top-view motion plus planned actions. Aligned actions improved dense MSE by
2.91% [2.49, 3.40] over history-only and 2.94% [2.41, 3.60] over shuffled
actions, below both 10% gates. Endpoint error improved 1.71%/2.24%. There is a
real signed-direction signal: cosine increased from 0.040 to 0.216, a +0.175
absolute gain with a positive paired interval, and aligned dense MSE beat
history for all 64 clips. A PCA192 oracle retained 98.60% of train variance and
reconstructed validation flow at cosine 0.911, so output compression is not the
main explanation. The result is nevertheless `NO_GO` for a learned proxy: it
does not predict flow magnitude/detail strongly enough to condition Wan under
the registered handoff rule. It strengthens the case for an analytic kinematic
scaffold while weakening the case for learning the entire flow field from this
small screen. Full identities and post-hoc per-horizon/motion-quartile slices are
preserved in `DENSE_TOP_FLOW_STAGE0.md`.

## Recommended architecture

First deployable stage:

\[
u_{robot}=G(a,\mathcal R,\mathcal C),\qquad
x_0=\operatorname{RGBDenoise}(z_s,h,a,u_{robot}).
\]

Use analytic rasterized link scene flow `(dx,dy,visibility)` where possible.
For the proof screen, project a fixed flow latent through the existing 16-channel
auxiliary seam, keep its clock at zero, never update it, and set its velocity
loss to zero. Compare flow-off, aligned, episode-shuffled, and target-video
oracle flow under identical RGB calls and optimization.

Only if fixed robot flow passes, add the genuine second stochastic state:

\[
r\sim p_\psi(r\mid h,a,u_{robot}),\qquad
x\sim p_\theta(x\mid h,a,u_{robot}+r),
\]

where `r` captures object motion/contact residuals. Require generated residuals
to beat zero, train-mean, and shuffled residuals at equal total calls.
The leakage contract, state channels, attribution arms, thresholds, and stop
rules are frozen in `PHYSICS_ANCHORED_RESIDUAL_FLOW_PROTOCOL.md`.

## Novelty boundary

“Condition video on optical flow or rendered robot motion” is no longer a
sufficient contribution. The closest current primary work occupies these
pieces:

| Existing work | Inference-time condition already demonstrated |
|---|---|
| [FlowWAM](https://arxiv.org/html/2607.13017) | fixed desired optical flow jointly attended with RGB |
| [RealWonder](https://arxiv.org/html/2603.05449) | simulator flow/coarse RGB plus a four-step causal generator |
| [Robot-Factored World Models](https://arxiv.org/html/2607.22535) | controller-realized nominal trajectory rendered as URDF mesh RGB and end-effector depth, paired with static RGB/depth |
| [OSCAR](https://arxiv.org/html/2606.04463) | 2-D kinematic skeleton video across embodiments |
| [ContactFlow](https://arxiv.org/html/2607.26579) | sparse 3-D object-contact points and displacement projected as a seven-channel control video |
| [iMaC](https://arxiv.org/html/2606.09813) / [EA-WM](https://arxiv.org/html/2605.06192) | deterministic FK/contact-distance fields and event-supervised kinematic fusion |

Robot-Factored World Models is especially close to our Stage-1 scaffold: it
already separates command realization from scene response, replays commands in
a robot-only controller, and conditions a latent Wan model on nominal rendered
geometry. OSCAR makes a coarse skeleton alternative explicit, and ContactFlow
occupies deterministic planned-contact conditioning. These papers are external
evidence that a calibrated causal visual action interface can help; they also
mean that reproducing one on ABC/LACWM is a baseline, not the paper.

A defensible new claim would need the combination of:

- a **physics-anchored stochastic residual motion state** that separates known
  robot kinematics from uncertain object/contact consequences;
- attributed quality gains from generated rather than oracle residuals;
- two- or four-call inference including scaffold cost;
- measured 5--10 Hz closed-loop DAgger utility rather than video metrics alone;
- calibration/command-tracking uncertainty that survives robot, view, and task
  transfer.

That could be a substantive physical-AI/world-model contribution. A modest MSE
gain from fixed clean flow, skeletons, mesh RGB, depth, or deterministic contact
points alone would be useful reproduction/engineering evidence but not, by
itself, a major algorithmic innovation.

## Other inference-causal directions

Ranked after the completed evidence and prior-art audit:

1. **Interaction-residual motion forcing.** Subtract analytic robot flow from
   target scene flow during training, then generate the uncertain residual for
   objects, contact, occlusion, and tracking error before RGB. This is the main
   dual-diffusion hypothesis; raw pixelwise subtraction must be confidence- and
   occlusion-masked.
2. **Generated contact/event tokens.** If a dense residual is too hard, predict
   an 8--32-token state for contacted object/slot, onset/release, attachment,
   signed displacement, and visibility change. The novelty would be generating
   the uncertain outcome before RGB, not ContactFlow-style planned contact,
   iMaC-style deterministic proximity, or EA-WM-style training supervision.
   Require action-shuffle sensitivity before generator integration.
3. **Cross-view 3-D interaction tracks.** Generate a compact, view-consistent
   object-trajectory state and project it into all three cameras. This may be
   lower entropy than RGB but carries substantially higher calibration/tracking
   risk and follows the first two options.
4. **Confidence-gated no-regret forcing.** Predict per-token auxiliary
   reliability and train with dropout, shuffling, time shifts, and corruption so
   unreliable states fall back to the video-only path. This can make a partly
   useful auxiliary safe, but it cannot manufacture missing information.

The retrospective confidence screen closes option 4 for the existing midpoint
scratchpad.  Its generated tensor was preserved only by hash; auxiliary NMSE
and cosine require the hidden clean future; its learned state/clock gates are
global constants; and all other varying numerics are target quality or timing
metadata.  Aligned versus same-checkpoint off changed decoded/temporal error by
only +0.0636%/+0.0326%, while aligned versus future-shuffled temporal error was
-0.0047% with interval [-0.0227, +0.0129]%.  More decisively, a prohibited
target-aware chooser selected 36/64 clips and still gained only +0.0447%
temporal MSE.  Therefore no fitted confidence gate or generator continuation is
justified; `always off` is the exact no-regret policy for this artifact.  This
does not reject confidence gating for a future auxiliary that emits calibrated,
inference-observable uncertainty and first shows material sample-specific
utility.

Few-step self-/causal-forcing is an acceleration layer after one of these
mechanisms produces a stronger teacher; it is not a substitute for a causally
informative state.

There is one additional route that does not require a second state at serving:
**privileged on-policy distillation**.  During training, the teacher receives
the clean target-video TF/V-JEPA feature, while the causal student receives only
history, actions, and its own rollout state.  Teacher and student velocities are
matched at student-visited states; a training-only reachability gate discards
teacher corrections that are either not better than the direct flow target or
too far from the causal student's prediction.  At inference the teacher,
feature extractor, and clean-feature cache are absent.  This is materially
different from the failed VideoREPA arms, which aligned intermediate relation
matrices at ordinary forward-noised states and never distilled the teacher's
actual denoising policy.

Recent external evidence makes this a credible controlled experiment rather
than a guarantee.  [Branch-aware on-policy diffusion distillation](https://arxiv.org/html/2607.24731)
transfers dense controls to sparse-control video students, and
[Privileged Self-Distillation](https://arxiv.org/html/2607.27055) uses a
future-aware teacher plus a reachability gate for causal sequential prediction.
Neither is an ABC/LACWM result, and a causal student cannot recover irreducible
future detail.  The exact eligibility gate, arms, leakage assertions, and stop
rules are frozen in `PRIVILEGED_ON_POLICY_VIDEO_DISTILLATION_PROTOCOL.md`.

## Phased execution decision

1. Extend the passing three-clip nominal-YAM diagnostic to train-only
   calibration fitting, distortion-aware projection, masks, and finer
   non-wrapping time-shift/wrong-calibration controls.
2. Replay planned actions through the robot-only controller, render
   masks/depth/link IDs and analytic top-view flow, and require alignment over
   those controls before integration.
3. Only after that gate, run paired `FLOW-OFF/CAUSAL/SHUFFLED/ORACLE` generator
   arms at NFE 1/2/4; do not use the failed learned dense proxy as the condition.
4. If analytic causal flow has an attributed quality gain, port full per-block joint
   RGB/flow attention and test the physics-anchored residual state.
5. In parallel, run the train-only eligibility test for privileged on-policy
   distillation; proceed only if the aligned oracle teacher beats both the
   causal student and a shuffled-feature teacher on student-visited states.
6. Distill only the strongest causal teacher to two/four calls; report complete
   rendering, flow, Wan, and decode latency.
7. Evaluate multi-seed DROID/ABC lockbox, perceptual/FVD metrics, long rollouts,
   and closed-loop DAgger utility. In parallel, test a latent-only policy path
   to avoid the approximately 0.130-second RGB decoder cost.

## Claim boundary

The current evidence is enough to reject several implementations and select a
better causal hypothesis. It is not enough to claim that dual diffusion improves
video quality, that TF features are generally useless, or that real-time DAgger
has been achieved. Val64 has been adaptively reused; all generator screens are
short, one-seed studies; the protected test remains untouched.
