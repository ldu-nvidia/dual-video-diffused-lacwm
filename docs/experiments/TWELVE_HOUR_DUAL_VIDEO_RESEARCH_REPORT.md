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
budget. The most credible remaining route is:

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
and exact ABC rendering first requires a robot/camera calibration gate.

## Causal taxonomy

| Mechanism | Future clean video needed at inference? | Status |
|---|---:|---|
| target-video TF/V-JEPA condition | yes | oracle upper bound only |
| autonomous semantic/frequency branch before RGB | no | tested variants negative |
| same-call midpoint scratchpad | no | tested variant negative and sample-insensitive |
| training-only relation/spectrum/motion loss | no | TRD and TFREG negative; per-view motion result pending |
| planned-action robot-only flow | no | strongest next route; geometry/calibration prerequisite |
| physics-anchored stochastic residual flow | no | prospective genuine dual-diffusion extension |
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
| action-to-Farneback-summary proxy | +6.59% versus history-only; +9.04% versus shuffled actions | planned actions contain incremental motion information, but target is coarse |

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

## Phased execution decision

1. Pin official YAM assets; reconstruct D405 top-view timing/intrinsics and pass
   a silhouette/flow alignment gate on train/val only.
2. Run the dense action-to-flow information gate, including aligned, shuffled,
   time-shifted, history-only, and latency controls.
3. Run paired `FLOW-OFF/CAUSAL/SHUFFLED/ORACLE` generator arms at NFE 1/2/4.
4. If causal flow has an attributed quality gain, port full per-block joint
   RGB/flow attention and test the physics-anchored residual state.
5. Distill only the strongest causal teacher to two/four calls; report complete
   rendering, flow, Wan, and decode latency.
6. Evaluate multi-seed DROID/ABC lockbox, perceptual/FVD metrics, long rollouts,
   and closed-loop DAgger utility. In parallel, test a latent-only policy path
   to avoid the approximately 0.130-second RGB decoder cost.

## Claim boundary

The current evidence is enough to reject several implementations and select a
better causal hypothesis. It is not enough to claim that dual diffusion improves
video quality, that TF features are generally useless, or that real-time DAgger
has been achieved. Val64 has been adaptively reused; all generator screens are
short, one-seed studies; the protected test remains untouched.
