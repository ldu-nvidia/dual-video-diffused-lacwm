# Action-derived flow scaffold: staged causal protocol

Date: 2026-08-08

Status: prospective protocol; no downstream generator claim has been made

## Scientific question

Can a dense motion field that is available from the proposed robot trajectory
*before* RGB generation improve a low-NFE action-conditioned video world model?

This differs from oracle TF or V-JEPA conditioning. The causal input is

\[
u_{0:T}=G(a_{0:T},h_{\le 0},\mathcal R,\mathcal C),
\]

where `G` is either a deterministic kinematic renderer or a train-only-fitted
causal predictor. It may not read target future scene RGB at inference. Future
RGB may be used to construct supervised flow targets during training and to
score validation only after the causal prediction has been materialized.

## Causality contract

The deployable scaffold may receive:

- five observed RGB frames;
- the complete candidate action sequence already being evaluated;
- morphology identity;
- fixed robot geometry, joint conventions, and camera calibration;
- parameters and normalization constants fitted on train only.

It may not receive future scene RGB, target video latents, cached V-JEPA
targets, recorded executed joint states that are not derivable from the proposed
commands, validation-fitted normalization, or an online future-video encoder.

## Stage 0A: deterministic robot-only flow prerequisite

Audit the selected dataset for all of:

1. robot URDF/meshes and joint-name ordering;
2. gripper mapping, root pose, and command-to-position convention;
3. intrinsics/extrinsics for every fixed camera;
4. camera-to-link transforms for wrist cameras;
5. time alignment between images and proposed commands.

If complete, replay planned positions without scene objects, rasterize link ID
and depth, and analytically project corresponding link surfaces between poses.
This produces `(dx,dy,visibility)` directly. A fixed train-derived magnitude
ceiling is used for all splits. RAFT/HSV/VAE encoding is retained only as an
external FlowWAM reproduction arm because it adds avoidable latency and
quantization.

The current immutable ABC train512/val64 cache does not contain these assets.
It has 12 joint plus two gripper action values and three RGB views, but lacks the
robot/camera mapping needed for image-aligned deterministic rendering. Do not
fabricate a robot-only condition from that cache.

A subsequent audit of the 576 corresponding raw MCAPs found intrinsics but no
extrinsics, TF tree, base pose, robot model, depth, segmentation, mask, or MCAP
attachment. The official ABC repository supplies nominal YAM MJCF/meshes and
D405 transforms. Pin those assets, begin with the 456 D405 clips that have all
three intrinsics (or the top camera alone), and require rendered silhouette
alignment to beat time-shifted/wrong-calibration controls before extracting
conditioning flow. This is a calibration gate, not permission to assume the
nominal geometry matches every recorded station.

## Stage 0B: causal action-to-full-scene-flow proxy gate

If exact geometry is unavailable, use this explicitly labelled proxy only to
test whether the planned actions contain enough visual-motion information to
justify generator integration.

### Targets

- Frozen train512 and val64 only; protected test is inaccessible.
- Split the width-concatenated RGB into three 320-pixel views.
- Begin with the fixed top camera to avoid wrist-camera ego-motion ambiguity.
- Estimate raw flow for transitions `frame 4 -> 5` through `frame 11 -> 12`.
- Store low-resolution `(dx,dy,validity)` rather than HSV color.
- Fit every scale/normalizer on train512 only and seal target identities before
  fitting a predictor.

### Parameter-matched predictors

| Code | Inference inputs | Purpose |
|---|---|---|
| `H` | observed history; action adapter instantiated but hard-masked | causal history baseline |
| `HA` | observed history plus aligned planned actions | deployable candidate |
| `A` | planned actions with history hard-masked | diagnostic, not primary |

Evaluate the trained `HA` checkpoint with aligned, episode-disjoint shuffled,
and zero actions using identical observed input. Fit no validation parameters.

### Registered Stage-0 gate

All conditions are required on paired val64 clips:

1. `HA` improves motion-weighted endpoint error over `H` by at least 10%, with
   paired-bootstrap lower bound above zero;
2. ordinary endpoint error and angular/cosine error do not regress materially;
3. episode-disjoint shuffling erases at least 80% of the aligned `HA` gain;
4. the target is moved to the predictor device only after every causal output
   and latency measurement has completed;
5. p95 scaffold latency is below 20 ms or below 10% of measured LACWM NFE-1
   latency, whichever is stricter.

Failure stops the proxy path before video-generator training. A generic motion
prediction gain that survives action shuffling is not evidence for an
action-derived scaffold.

The completed exploratory summary probe was weaker than this dense-flow gate:
it predicted per-view 2x2 Farneback statistics with ridge regression. Planned
actions improved all-view val64 standardized MSE by 6.59% over history-only and
9.04% over episode-shuffled actions, with both paired intervals above zero. It
passed its separately preregistered 1% information gate but did not meet the
10% all-view threshold above. It motivates the top-view dense/calibration test;
it does not authorize Stage 1 by itself. A diagnostic top-view slice was
+12.61% [10.23,15.08], versus +3.57%/+3.18% for the two wrist views; signed
direction was more predictable than magnitude. Freeze that top-view hypothesis
before the next run rather than retroactively redefining the completed endpoint.

## Stage 1: fixed-flow RGB conditioning

Only after Stage 0 passes, compare paired 200-update arms from the same frozen
VPM parent, seed, clip order, noise, optimizer, and Wan-call budget:

| Arm | Flow supplied before RGB denoising | Interpretation |
|---|---|---|
| `FLOW-OFF` | exact zero / hard-masked adapter | parameter-matched control |
| `FLOW-CAUSAL` | aligned deterministic or Stage-0 causal flow | deployable candidate |
| `FLOW-SHUFFLED` | episode-disjoint causal flow | attribution control |
| `FLOW-ORACLE` | flow extracted from target future scene RGB | non-deployable ceiling only |

The first implementation projects a fixed 16-channel flow latent through the
existing auxiliary seam; its flow clock is zero, flow velocity loss is zero,
and the flow state is never updated. This tests conditioning cheaply but is not
the full dual-attention architecture. If it passes, port separate flow patch
embedding/head and per-block concatenated RGB/flow self-attention.

Primary endpoints are NFE-1 decoded MSE and temporal MSE with latent NMSE as a
guardrail. Report NFE 1/2/4, action-shuffle attribution, end-to-end scaffold
latency, RGB decode latency, peak memory, and training GPU-hours. A deployable
gain must be statistically positive, disappear under flow shuffling, and retain
a preregistered fraction of the oracle ceiling.

## Stage 2: speed, only after a stronger causal teacher exists

Distill the strongest `FLOW-CAUSAL` model to two or four RGB calls with an
inference-consistent or self-forced objective. The current VPM multi-step Euler
trajectory is worse than its one-call output, so it is not a valid teacher.
Report complete candidate-rollout latency including action-to-flow rendering,
flow encoding, Wan calls, and optional RGB decode. For latent-policy DAgger,
also report the path that omits RGB decoding entirely.

## Claim boundary

A Stage-0 proxy pass says only that planned actions predict a useful dense
motion target. A Stage-1 pass establishes a causal flow-conditioning mechanism
for this data/model/budget. Neither alone establishes literal joint diffusion,
FVD improvement, long-horizon stability, or real-time closed-loop DAgger; those
require multi-seed evaluation, an untouched lockbox, perceptual metrics, and a
task-utility study.
