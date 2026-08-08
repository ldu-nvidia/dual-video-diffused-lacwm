# Action-cycle Stage-1 protocol and STOP record

## Status

**Prototype only; do not register or launch.** The Stage-0 prerequisite named
in this prospective design returned `stop_or_revise_action_cycle_path`, not the
required `go_to_generated_latent_action_cycle_stage1`. The Stage-1 registration
tool therefore fails closed on the available result. No Stage-1 training,
validation, W&B run, checkpoint, or generator result exists.

The implementation was frozen before the Stage-0 validation result was read.
It is retained only to make the rejected continuation concrete and reusable if
a *new preregistered* recoverability representation later passes Stage 0. The
current Stage-0 validation output must not be used to tune its scientific
hyperparameters.

## Hypothesis and intervention

The discarded hypothesis was that generated Wan-latent motion can be made more
action-faithful without any clean future feature at inference by applying a
frozen inverse-action critic during training only. LACWM uses `sigma=1` for
noise and `sigma=0` for clean data, so one ordinary Wan call supplies

\[
  \hat{x}_0=x_\sigma-\sigma v_\theta(x_\sigma,\sigma,a).
\]

For each of four causal Wan bins, Stage 1 would reproduce Stage 0 exactly:
split the 120-wide latent into three 40-column camera views after Wan encoding,
normalize every bin independently over channel/height/per-view width, average
pool to `6 x 10`, and subtract adjacent endpoints. This gives
`[B,3 transitions,3 views,960]`. The frozen train-only feature normalization and
nine aligned ridge weights predict the standardized `[4,5,23]` requested-action
segment for each transition. Predictions are averaged across views exactly as
in Stage 0.

Only transitions 1 and 2 enter the loss: transition 1 crosses the
observed/generated boundary and transition 2 lies wholly in generated future.
The history-only transition 0 is not optimized. The proposed candidate loss
was frozen at

\[
  L_{\rm AC}=L_{\rm video}+0.05\,\frac{1}{2}
  \sum_{b\in\{1,2\}}
  \operatorname{MSE}_{j\in A_b}
  \left(\hat a^{\rm ridge}_{b,j}(\hat x_0),
  \tilde a_{b,j}\right),
\]

where `A_b` contains only active train-population target coordinates. The
weight `0.05`, transitions, pooling, normalization, ridge, update count, and
validation gate were fixed without consulting Stage-0 validation metrics.

## Controlled arms

Both arms would start from the exact update-1000 parameter-matched VPM snapshot
with SHA-256
`f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`.
They use seed 1234, 8 B200 ranks, local batch 1, 200 updates, a fresh identical
AdamW optimizer, identical clip order, timesteps, video noise, dropout RNG,
Wan calls, learning-rate schedule, validation schedule, and no EMA.

| Arm | Diagnostic computed | Returned objective |
|---|---|---|
| `AC-OFF` | same frozen critic under `no_grad` | exact untouched VPM video loss |
| `AC-ON` | same frozen critic with gradient to `x0_hat` | video loss + `0.05 * cycle loss` |

The critic is a plain Python object rather than `nn.Module`. It has zero
trainable parameters, registers no buffer, is excluded from the optimizer and
checkpoint, and is never constructed in deployment mode. The checkpoint state
key/shape/dtype inventory must match the parent VPM exactly. Both arms make one
Wan call per training example. Cached V-JEPA targets, clean future features,
and protected test data are forbidden.

## Intended target-free evaluation

Had Stage 0 passed, both checkpoints would be instantiated in `deploy` mode
with null critic path/digest/identity and strict state loading. Each val64 clip
would use only five history frames, morphology, requested actions, and
clip-keyed deterministic noise. Ground-truth future RGB and clean scoring
latents would be constructed only after all endpoints in a batch finish.

The fixed endpoint inventory was:

| Endpoint | NFE | Action input | Role |
|---|---:|---|---|
| `aligned_nfe_1` | 1 | matching requested trajectory | primary quality |
| `aligned_nfe_4` | 4 | matching requested trajectory | descriptive |
| `shuffled_nfe_1` | 1 | episode-disjoint bijective donor | action attribution |
| `zero_nfe_1` | 1 | exact zero action, same morphology | action attribution |

All endpoints share exact initial video/auxiliary noise hashes and actual Wan
call counts. W&B is locked to private, ungrouped
`zijiandu/dual-video-diffusion-private`. No online teacher, clean feature,
critic, or protected-test access is permitted at inference.

Nine preregistered val64 contrasts form one Bonferroni family with 10,000
paired clip bootstraps:

1. `AC-ON` versus `AC-OFF`, aligned NFE-1: decoded MSE and decoded temporal
   MSE must each improve by at least 3% with simultaneous lower bound at least
   1%; latent NMSE point and lower bound must each be no worse than -1%.
2. Within `AC-ON`, aligned must beat both shuffled and zero actions in decoded
   and temporal MSE, with positive point and simultaneous lower bound.
3. The aligned-versus-shuffled advantage for `AC-ON` must exceed the same
   advantage for `AC-OFF` in decoded and temporal MSE, with positive point and
   simultaneous lower bound.

Every check is required. NFE-4 and zero-action difference-in-differences are
descriptive and cannot rescue a failed NFE-1 gate.

## Why Stage 0 stopped this path

The sealed Stage-0 result identity is
`b81caa863afc87f3d38409be9e2f76ea956f04bf4cb22063da058f3cd49e70ee`.
Its complete 16-contrast gate returned `all_passed=false`. For the two
future-relevant transitions:

| Stage-0 comparison | Observed | Required | Outcome |
|---|---:|---:|---|
| aligned cosine | 0.117504 | — | weak absolute signal |
| same-clip temporal-negative cosine | 0.103518 | — | nearly the same |
| cosine gap vs temporal negative | 0.013985; simultaneous LB -0.022456 | gap >= 0.10 and LB > 0 | fail |
| aligned normalized MSE | 1.994852 | — | worse than mean |
| train-mean normalized MSE | 1.070248 | — | — |
| relative MSE gain vs train mean | -86.392%; LB -114.062% | >= +20% and LB > 0 | fail |
| temporal-negative MSE | 2.020338 | — | nearly the same |
| relative MSE gain vs temporal negative | +1.261%; LB -3.817% | >= +20% and LB > 0 | fail |
| retrieval aligned vs shuffled | 0.046875 vs 0.0; LB 0.0 | positive point and LB | fail |

The ridge separates episode-disjoint shuffles somewhat, but it does not recover
sample-specific temporal action alignment and is substantially worse than the
train mean in normalized MSE. Backpropagating this critic would therefore be
more likely to impose task/motion bias than a reliable action-consistency
gradient. Running the expensive generator continuation would not answer the
intended causal question, so the correct decision is STOP.

## Prototype boundary

The branch contains the feature/critic implementation, exact parameter-schema
guard, OFF/ON configs, paired trainer telemetry, a fail-closed GO-bound
registration extractor, and draft target-free evaluator/analyzer. Only local
unit/syntax checks were run. There is deliberately no registered study,
submitted Slurm workflow, deployment canary result, W&B run, checkpoint, or
quality claim. The evaluator/analyzer remain prospective code, not validated
experimental evidence.
