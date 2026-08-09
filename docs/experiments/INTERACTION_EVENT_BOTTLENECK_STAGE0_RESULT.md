# Causal interaction-event bottleneck Stage-0 result

Date completed: 2026-08-08

Decision: **`STOP_INTERACTION_EVENT_BOTTLENECK`; do not integrate PCA32 into
the video generator**

## Central finding

Removing rendered robot pixels did not uncover a deployable, action-predictable
future interaction state. The correctly paired planned action changed the
PCA32 prediction and was directionally better than a different episode's
action in coefficient space, but it added essentially no held-out value beyond
five-frame history or the train-mean action. The effect also vanished when the
predictions were decoded back to the masked event field.

The learned bottleneck itself failed its relevance prerequisite. Exact held-out
PCA32 coefficients reconstructed the masked future event field only 4.52%
better than the fit-mean field, far below the frozen 20% gate. This means the
dense change/flow subspace learned on fit episodes transferred poorly to new
episodes, even before asking a causal predictor to generate it.

These results stop this representation. They do not prove that object
interaction is action-independent or that every discrete interaction state
must fail.

## Executed prospective contract

- Source: immutable ABC train512 cache/manifest only; validation and protected
  test were unsupported and unopened.
- Metadata scan: 415 D405, 37 OAK, 58 ZED-X, and 2 decxin episodes.
- Split: four planned-command-motion strata, each with 64 fit and 16 score
  clips; 256 fit and 64 episode-disjoint score clips.
- Donors: next held-out score episode within the same motion stratum.
- Inputs: top RGB frames 0--4, measured history state only for observed robot
  masks, and planned actions `[4:12,5,14]`.
- Target-only data: future top RGB and measured future state for robot masks.
- Raw target: eight transitions of four 45x80 channels—positive/negative
  luminance change and event-weighted signed Farneback flow—outside the union
  of source/target robot silhouettes dilated by 12 pixels at 180x320.
- Bottleneck: fit256-only randomized PCA32 over `[8,4,45,80]`.
- Predictors: equal-width history-only ridge and history+action ridge, with
  independent five-fold fit-only alpha selection. Both selected alpha 1000.
- Controls: aligned, episode shuffled, train mean, all-zero raw action, and
  nonwrapping action shifts -1/+1.
- Statistics: one common 10,000-draw episode bootstrap, seed 20260813. Six
  causal contrasts used one-sided Bonferroni 99.1667% lower bounds; two
  reconstruction contrasts used 97.5% lower bounds.

The registered source is commit
`0c3a4a0c6c9147db59fa9cef4a082e83598855bb`, SHA-256
`f652f22201b14c69bdd810116d050907464d6bb0729640ac286b68685d174ea5`.
Registration was sealed before selected state/RGB access.

## Mandatory causal effects

Positive percentages favor aligned action. The lower bound is the frozen
familywise Bonferroni lower bound; every contrast required at least 5%, a
strictly positive bound, and at least 60% favorable episodes.

| Metric | Aligned reference | Relative gain | Simultaneous lower bound | Favorable clips | Gate |
|---|---|---:|---:|---:|---|
| PCA32 coefficient MSE | history only | +0.204% | -11.643% | 22/64 | fail |
| PCA32 coefficient MSE | episode shuffled | +9.400% | -1.829% | 39/64 | fail |
| PCA32 coefficient MSE | train mean | +0.297% | -11.548% | 21/64 | fail |
| Decoded event-field MSE | history only | +0.010% | -0.366% | 26/64 | fail |
| Decoded event-field MSE | episode shuffled | +0.464% | -0.091% | 36/64 | fail |
| Decoded event-field MSE | train mean | +0.017% | -0.360% | 26/64 | fail |

The aligned-versus-shuffled coefficient point estimate is the only large
directional signal. Its ordinary paired 95% interval was already
`[-0.272%, +20.035%]`; it is therefore suggestive, not evidence that survives
sampling uncertainty. Its field-space counterpart is below one half percent.

Absolute mean errors were:

| Arm | Standardized PCA32 coefficient MSE | Decoded-field MSE |
|---|---:|---:|
| History only | 0.076544 | 0.000517869 |
| **Aligned** | **0.076387** | **0.000517815** |
| Episode shuffled | 0.084312 | 0.000520228 |
| Train mean | 0.076615 | 0.000517901 |
| Raw zero | 0.087997 | 0.000522836 |
| Shift -1 | 0.077673 | 0.000518213 |
| Shift +1 | 0.076225 | 0.000517759 |

The raw-zero action made aligned look 13.19% better in coefficient space, but
zero absolute joint commands are far outside the native action distribution.
This diagnostic cannot substitute for history, mean, or shuffled controls. The
+1 shift was slightly better than aligned on both metrics, so the data do not
support clip-step timing attribution either.

## Representation relevance failure

Fit PCA32 retained 49.90% of fit-target variance, yet its exact score
coefficient reconstruction achieved:

| Oracle reconstruction reference | Gain | 97.5% lower bound | Favorable clips | Frozen 20% gate |
|---|---:|---:|---:|---|
| Fit-mean field | +4.521% | +3.561% | 64/64 | fail |
| All-zero field | +5.662% | +4.313% | 44/64 | fail |

The intervals confirm a small nonzero projection benefit, but the magnitude is
not useful enough for a proposed generative scaffold. The likely reason is
episode-specific spatial change: a 32-dimensional dense-map PCA learned on 256
tasks did not span most held-out event energy. The event target was also sparse:
mean nonrobot event mass was 0.00588, while the dilated robot exclusion covered
31.98% of the work grid. Sparse compression and task-specific object location
make a global linear subspace a poor cross-episode state.

## Operational evidence and artifact identities

- Slurm job: `507307`, B200 `pool0-0015`, `COMPLETED 0:0` in 1m16s.
- Peak RSS: 10.51 GiB; disk read 9.65 GiB.
- Robot rendering: 3.586 ms mean, 4.284 ms p95 over 3,520 post-warmup poses.
- Registration identity:
  `f985c94c708950a6e90530044a9b91f58e539a50a3f7ac2f91b89b1dc945f393`.
- Analysis identity:
  `d170b30e27477687da2e403b151e90e065d0c17937e3291de1ba45a0980165cd`.
- Completion identity:
  `17051018c4e51d8e3764549b306807a77cae6e28ca30046675d9fa16c5e863ee`.
- Read-only audit identity:
  `b018d0996ca550dac812a34f3d662b3e566f4749c83755b7500fe6d4fa4380cb`.
- Audit recomputed eight mandatory and six diagnostic effects, checked 320
  provenance rows and 64 score rows, and validated 1,418 explicit false flags.
- Canonical artifact root:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/interaction_event_bottleneck/interaction-event-train256-score64-seed20260813-0c3a4a0-v1`.

## Interpretation and bounded fallback

The prior all-view flow proxy's action gain was substantially driven by robot
or camera motion. Once articulated robot geometry is excluded, dense nonrobot
change is sparse, task-specific, and not improved by the planned action over
history in this model. A dual branch based on this PCA32 would therefore add a
hard-to-predict state that carries little stable held-out field information.

The final bounded fallback is a separately frozen, explicit event-token state:
per future transition, global positive mass, negative mass, signed horizontal
transport, and signed vertical transport—32 scalars total, with no learned
target PCA. It tests whether spatial PCA instability hid an action-predictable
interaction-rate signal. It must use this same train-only split and controls
and cannot retroactively alter this result. If that fallback also fails the
5% history/shuffle gates, this line should stop before generator training.
