# Privileged-TF sequential DOE

## Objective

Test whether matched time-frequency (TF) exposure during training improves the
video model even when deployment uses neither TF state content nor the separate
TF sigma-clock residual. This stage retains the per-view Parseval-normalized
RFFT representation so it isolates training exposure from representation and
inference-schedule changes. LACWM uses `sigma=1` for noise and `sigma=0` for
clean data.

## Stage A: posthoc checkpoint attribution

Strict-load the completed 200-update checkpoints:

| Evaluation arm | Parent training arm | Training intervention |
|---|---|---|
| `trained_matched` | `cascade_matched_s010` | aligned TF content during video-loss examples |
| `trained_shuffled` | `cascade_shuffled_s010` | wrong future TF content with matched marginal/noise |
| `trained_off` | `cascade_off_s000` | no TF state content |

All three use the same deployment intervention:

- native all-video Wan schedule;
- NFE 1, 2, 4, and 8, with every call updating video;
- TF state-content and TF clock residuals exactly disabled;
- condition mode `off`, evaluation seed `20260726`;
- fifth visualization batch at iteration 199;
- zero optimizer updates and zero new training observations.

The `autonomous` and `off` renders must be bitwise identical and serve only as
a runtime no-op audit. Paired-input identity is established from raw actions,
raw morphology indices, clean/initial latents, noise, and targets. The learned
`z_control` is retained as a diagnostic but is not required to match because
each checkpoint's action encoder was trained end-to-end.

### Primary comparisons and promotion gate

The primary contrasts are `trained_matched - trained_shuffled` and
`trained_matched - trained_off`. At NFE 4, matched must satisfy every criterion
against both references:

1. temporal-difference MSE improves by at least 3%;
2. video-latent NMSE and decoded MSE point estimates do not regress;
3. the paired-bootstrap upper interval for temporal error is below zero;
4. video and decoded upper intervals remain below a +2% noninferiority margin;
5. temporal direction agrees at NFE 2 and NFE 8.

The previously validated true-video-only checkpoint is a secondary package
comparison only: it received video loss on every update, so it is not an
isolated causal control for these cascade checkpoints.

## Conditional Stage B: fresh training

Run Stage B only if Stage A passes its gate, or retains a practically meaningful
and sign-supported decoded/temporal signal at NFE 4 or 8. All arms would start
from the same warm start, seed, data order, native video clocks, and optimizer:

| Arm | TF auxiliary loss | TF state in video pass | Deployment |
|---|---:|---|---|
| `V0_video_only` | 0 | never; structural no-op | direct video |
| `A1_aux_only` | 2/3 | never | direct video |
| `P2_priv_matched` | 2/3 | matched with fixed 50% dropout | TF off |
| `P3_priv_shuffled` | 2/3 | wrong future TF on the same mask | TF off |

Every arm supplies native-schedule video loss on every update. TF arms add a
separate TF-only auxiliary pass whose RNG cannot change later video data,
video noise, or LoRA-dropout streams. The initial screen is one seed and 400
updates, evaluated at updates 0, 50, 100, 200, and 399.

## Claim boundary

The eight ranks are paired evaluation units, not independent model seeds.
Temporal-difference MSE can reward under-motion and is not perceptual temporal
quality. No FVD, perceptual-quality, wall-clock speed, or FPS claim is allowed
without the corresponding measurement.
