# Nominal-to-realized tracking residual Stage 0

Date frozen: 2026-08-08

Status: preregistered train512/validation64 exploratory screen, now completed;
the registration was written before this tool opened validation state targets;
no protected test or generator was run. See
`NOMINAL_TRACKING_RESIDUAL_STAGE0_RESULT.md` for the sealed result.

## Question and claim boundary

Can an inference-causal, compact robot command-tracking residual be predicted
well enough to become the stochastic state alongside deterministic robot
geometry?

This probe has measured future robot state, so it is not a command-smoothness
test. However, it does **not** replay the official controller or MuJoCo
dynamics. Its nominal trajectory is the recorded raw absolute position command
available to the model. The target is therefore called a **raw-command
tracking-residual proxy**, never a controller-simulated, physics-realized,
object-motion, or contact residual.

## Exact immutable tensors and ordering

Each raw `states.npz` stores:

- `joint_states[T,12]` and `joint_actions[T,12]` in
  `[left_joint1..6,right_joint1..6]` order;
- `gripper_states[T,2]` and `gripper_actions[T,2]` in `[left,right]` order;
- `frame_ts[T]` at top-camera frame times.

The immutable cached action tensor is `[N,13,5,23]`. For every clip, this tool
must prove bit-exactly that its first 14 coordinates equal
`concat(joint_actions,gripper_actions)[start:start+65].reshape(13,5,14)`
and that its last nine camera-padding coordinates are zero. Every manifest row
must use 13 boundaries `frame_indices[t] = start + 5*t`.

LACWM/cache order is
`[left6,right6,left_gripper,right_gripper]`. The official ABC simulator order is
`[left6,left_gripper,right6,right_gripper]`, with permutation
`[0,1,2,3,4,5,12,6,7,8,9,10,11,13]`. This screen audits that mapping but stays
in cache order because it does not launch the simulator.

For future transition `t -> t+1`, `t in {4,...,11}`, define the zero-order-hold
command endpoint and measured tracking residual

\[
q^{nom}_{t+1}=a_{t,4},\qquad
r^*_{t+1}=q^{meas}_{t+1}-q^{nom}_{t+1}.
\]

The target has shape `[8,14]`: 12 joint-angle errors in radians plus two
normalized gripper-aperture errors. The choice of the final command in each
five-sample chunk is fixed from the chunk/frame semantics, not selected on
validation.

## Causal predictor inputs

The history vector contains only:

- measured state boundaries at frames 0--4: `5x14`;
- raw command chunks 0--3: `4x5x14`;
- observed tracking residuals for transitions 0--3: `4x14`.

The candidate additionally receives planned raw commands in chunks 4--11:
`8x5x14`. Future measured states at frames 5--12 are target/scoring-only. RGB,
future RGB features, cached V-JEPA targets, and protected-test data are never
opened.

History and planned-action vectors are independently standardized on train512
and compressed with train-only PCA64. Targets are dimension-standardized on
train512. Multi-output ridge models use alpha
`{0.1,1,10,100,1000,10000}`, selected by five-fold train-only cross-validation.

## Frozen controls

All endpoints use the same val64 target and native raw-command nominal:

| Endpoint | Residual prediction |
|---|---|
| `history_only` | ridge from observed state/commands/residuals |
| `aligned` | same history plus native planned actions |
| `episode_shuffled` | same candidate with a different episode's planned-action feature |
| `mean_action` | same candidate with the train-mean planned-action feature |
| `train_mean_residual` | train mean residual |
| `zero_residual_raw_command` | raw command endpoint is assumed perfectly realized |
| `hold_current_state` | predicts all future measured states equal `q(frame4)` |

The hold-current control is mandatory because a residual model could otherwise
learn the algebraic shortcut `q(frame4)-command` without learning controller
response. Episode shuffling diagnoses sample-specific action use, but is not a
randomized physical intervention.

## Metrics and strict gate

The primary metric is per-clip mean dimension-standardized residual MSE.
Effects are `100*(1-mean(candidate)/mean(reference))`. One common causal
interpretation is not assigned to clip ratios: paired 10,000-resample
clip-bootstrap intervals use ratios of resampled means, seed 20260808 plus a
fixed contrast offset.

`GO` requires the aligned candidate to improve by at least **10%**, with a
strictly positive paired 95% lower bound, against every one of:

1. history-only ridge;
2. episode-shuffled actions in the same candidate model;
3. zero residual / raw-command nominal;
4. hold-current-state cancellation baseline.

Every condition is mandatory. A failure is `NO_GO` and forbids generator
integration. Train-mean residual and mean-action comparisons are reported as
diagnostics.

Absolute target residual RMS/MAE, candidate RMSE/MAE, joint and gripper RMSE,
hold-current error, and residual RMS divided by realized future-state-motion
RMS are reported so a relative percentage cannot hide a physically negligible
effect.

## Timestamp caveat

The ABC preprocessor comments that state/action streams are resampled by
"nearest" timestamp, but executes
`np.searchsorted(ts, frame_ts)` without comparing the preceding sample. It
therefore selects the next/ceiling state or action independently for each
stream. The raw state/action sample timestamps are not retained in
`states.npz`. This screen can audit clip/frame ordering but cannot recover or
claim sub-frame controller timing. That limitation must remain prominent even
if the statistical gate passes.

## Reproducibility

The tool writes a registration before opening `states.npz`, input provenance
with hashes for all 576 state files, derived arrays, model state/predictions,
ordered val64 per-clip rows, sealed analysis, and a completion receipt. Run:

```bash
python tools/stage0_nominal_tracking_residual.py \
  --output /approved/artifact/root \
  --train-cache /immutable/cache/train \
  --val-cache /immutable/cache/val \
  --train-manifest /immutable/manifests/train.jsonl \
  --val-manifest /immutable/manifests/val.jsonl

python tools/audit_nominal_tracking_residual_stage0.py /approved/artifact/root
```

Results may be appended only after the registration and source are committed.
