# Nominal tracking-residual Stage-0 result

Date: 2026-08-08

Decision: **GO for a geometry-alignment follow-up; not GO for video-generator
integration**

## Finding

A compact, inference-causal robot command-tracking residual is strongly
predictable on the immutable ABC train512/validation64 development split.
Observed state/history plus the aligned planned action trajectory cleared all
four preregistered 10% gates, including the explicit hold-current cancellation
control.

The result is narrower than a dual-video result. It predicts measured robot
joint/gripper tracking error relative to a raw command endpoint. It does not
predict object motion or contact, replay the real controller, improve rendered
geometry yet, or improve generated video.

## Executed causal contract

For each clip, the source audit established:

- measured boundaries: `[13,14]`, ordered
  `[left_joint1..6,right_joint1..6,left_gripper,right_gripper]`;
- cached commands: `[13,5,23]`, with the first 14 values bit-exactly equal to
  raw joint/gripper actions and all nine padded camera coordinates exactly zero;
- boundaries: `frame_indices[t] = start + 5*t`;
- predictor history: measured frames 0--4, raw command chunks 0--3, and four
  observed tracking residuals;
- candidate-only input: planned action chunks 4--11;
- target/scoring-only state: measured frames 5--12;
- target: `q_measured[t+1] - action[t,4]`, `t=4..11`, shape `[8,14]`.

Future measured state was not a predictor input. No RGB, cached V-JEPA target,
or protected test was opened. Train and validation contain 512 and 64 unique,
episode-disjoint rows.

The registered LACWM-to-official simulator permutation is
`[0,1,2,3,4,5,12,6,7,8,9,10,11,13]`; it round-tripped exactly. No simulator was
used in this probe.

## Registered paired result

Positive values favor history plus aligned planned actions. Effects use the
ratio of paired clip means and 10,000 common-unit bootstrap resamples.

| Reference endpoint | Standardized MSE, reference | Standardized MSE, aligned | Improvement | Paired 95% interval | Favorable clips | Gate |
|---|---:|---:|---:|---:|---:|---|
| History-only ridge | 0.931550 | 0.371876 | +60.08% | `[54.38,65.04]%` | 64/64 | pass |
| Episode-shuffled actions, same model | 1.349931 | 0.371876 | +72.45% | `[67.29,77.18]%` | 64/64 | pass |
| Zero residual / raw command | 1.181187 | 0.371876 | +68.52% | `[63.73,72.35]%` | 64/64 | pass |
| Hold current state | 14.245022 | 0.371876 | +97.39% | `[96.22,98.08]%` | 61/64 | pass |

The two diagnostics also favored aligned actions: +67.21%
`[62.60,70.98]%` versus the train-mean residual and +59.55%
`[53.39,64.72]%` versus the same model receiving the train-mean action. The
aggregate standardized validation R2 was 0.672 for aligned actions, 0.179 for
history-only, and -0.190 for shuffled actions.

Aligned actions beat the zero-residual raw-command baseline at all eight
horizons. Point gains ranged from 57.64% at horizon 8 to 77.35% at horizon 2;
all paired lower bounds were above 43%.

## Absolute scale and cancellation audit

The effect is not only a large percentage on a vanishing target:

| Quantity | Value |
|---|---:|
| Target residual RMS, mixed stored units | 0.08050 |
| Target residual mean absolute value | 0.04243 |
| Target joint residual RMS | 0.06180 rad (3.54 degrees) |
| Target gripper residual RMS | 0.14981 normalized aperture |
| Measured future state motion from frame 4, RMS | 0.20669 |
| Residual RMS / future-state-motion RMS | 0.38945 |
| Aligned predictor RMSE | 0.04124 |
| Aligned predictor MAE | 0.02547 |
| History-only predictor RMSE | 0.06256 |
| Hold-current predictor RMSE | 0.20669 |
| Aligned residual cosine, clip mean | 0.80665 |

The candidate joint RMSE is 0.03305 rad (1.89 degrees), a 71.40% raw-MSE
reduction from the zero-residual joint baseline. Gripper RMSE is 0.07313, a
76.17% raw-MSE reduction from zero residual. Overall RMSE is 48.77% lower than
zero residual, 34.09% lower than the history-only learned predictor, and 80.05%
lower than holding the current state. Thus the registered hold-current guard
rules out the simplest `q(frame4)-command` cancellation explanation.

This does not rule out all linear command-cancellation structure. Residual and
realized-state parameterizations are algebraically related, and the planned
action PCA64 retains 99.940% of train action variance. The justified claim is
therefore that the observed history/action pair predicts future measured robot
state beyond raw command, history alone, and trivial persistence—not that a
new stochastic physical variable has been discovered.

## Timestamp limitation

This is the main measurement caveat. The pinned ABC preprocessing source says
"nearest" resampling but executes `np.searchsorted(ts, frame_ts)` directly.
State and command streams are therefore assigned independently to the
next/ceiling sample. Their original controller timestamps are not retained in
`states.npz`, so sub-frame command/state latency cannot be recovered.

The registered 5-frame boundary periods have median 0.16671 seconds, but range
from 0.08348 to 0.40071 seconds in train and from 0.08348 to 0.25044 seconds in
validation. This probe establishes clip-level predictive signal, not precise
controller timing or a reusable controller dynamics model.

## Decision and next gate

The strict statistical Stage-0 gate passes, but the correct next experiment is
not generator conditioning yet. Use the predicted measured trajectory to drive
the already validated nominal YAM/D405 renderer, then compare on observed
transitions:

1. raw-command nominal render;
2. predicted tracking-corrected render;
3. hold-current render;
4. episode-shuffled predicted residual;
5. measured-state render as a nondeployable geometry ceiling.

Require predicted correction to improve silhouette/edge alignment and
robot-only flow against raw command and shuffled residual with positive paired
lower bounds. Only that establishes that proprioceptive prediction survives
forward kinematics and camera projection. A later video experiment can then
hold the corrected robot field fixed while generating a separate stochastic
object/contact residual. The present 14-D state is a robot-geometry correction;
it is not the object/contact residual proposed as the potentially novel dual
state.

## Reproducibility and artifact identities

- Frozen source commit: `5ceb2515a9f950d861e26ed66754ff6363713d54`
- Executed tool SHA-256:
  `ba28f707460e15e7b17ca9142ceb6449ec30dc99e3341624e4e1021e91cf87da`
- Registration identity:
  `113b3c8a1d3549d41e96ee7942517f570abe1dd8ae36ce5e0ad89a2f6984e586`
- Analysis identity:
  `5a949333c7bd3d61d98d5c727dadc31b427811c5b44836ddb2640f8dff544ebc`
- Completion identity:
  `b1295203ba0be74268dedea20b4079425e5257b41f6aabbd5654abae6454c7dc`
- Analysis JSON SHA-256:
  `d7cf4a4050714234661966f6dd4a3001d5acbf4c8cc757af6af5f614456d4a83`
- Derived-array SHA-256:
  `2c7f8417439b5d5a6af1a106a6fc95cd6a5c64e6b1d065182698e4299d043601`
- Model/prediction SHA-256:
  `2bae3e53c40293c597d4fbe3a51bdb764999523270629970a483b271f346825d`
- Input-provenance SHA-256:
  `de493a502bb362d4953ae73936e69f1358ae5003d9fa4ff403db9af2d92eb7e0`
- Per-clip SHA-256:
  `abbd87d3f283dbef58c4b1ea396daca40726b5aec5efb6fe63a1847c382ef573`
- Preprocessing source SHA-256:
  `c6275d5a99fc9fad9701041baeace53a798924a80f106c7a26ac41d85a565ff0`

Artifact root:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/nominal_tracking_residual_stage0/
  nominal-tracking-residual-stage0-seed1234-20260808-5ceb251-v1
```

The committed read-only audit checked seven artifact hashes, all three sealed
JSON identities, one ordered val64 pass, all 576 input-provenance rows, 644
explicit protected-test false flags, and different-episode shuffle donors. It
passed. Unit qualification was 6/6 tests in the pinned B200 Python environment.

## Limitations

- One fixed seed and a repeatedly reused exploratory validation64 split.
- Residual targets use measured future proprioception and cannot be supplied at
  inference; only the predictor output is deployable.
- The raw command endpoint is not a controller/dynamics rollout.
- Independent ceiling resampling prevents precise temporal interpretation.
- The proxy contains robot tracking only, not object motion, contact, occlusion,
  or scene uncertainty.
- No renderer-alignment validation, video model, FVD/perceptual metric,
  generation latency, or closed-loop DAgger result is measured here.
