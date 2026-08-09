# Explicit interaction-event token Stage-0 result

Date completed: 2026-08-08

Decision: **`STOP_EXPLICIT_EVENT_TOKENS`; end the masked interaction-event
line before video-generator training**

## Central finding

Replacing the poorly transferring dense PCA32 target with exact global event
integrals did not rescue a useful causal bottleneck. The representation itself
was active: all 32 fit dimensions varied and 62/64 score clips contained
nontrivial nonrobot event mass. Correctly aligned planned action also beat an
action borrowed from another episode by 23.62%. However, it improved the
held-out all-token error by only 2.77% over history alone and 1.98% over the
fit-mean action. Both effects had negative simultaneous lower bounds and helped
only 29/64 clips, well below the frozen 5%, positive-bound, and 60% gates.

The action signal was confined mostly to global change mass and did not extend
to signed transport. Even change mass was not reliable across episodes:
aligned action was 7.06% better than history at the point estimate, but helped
only 34/64 clips and had a -0.73% simultaneous lower bound. Transport was
0.54% worse than history. Nearby action shifts were statistically
indistinguishable from aligned action, so the result also lacks temporal
specificity.

This closes the preregistered interaction-event fallback. It does **not** show
that actions cannot predict object interactions. It shows that global masked
photometric-change and flow integrals are not a sufficiently reliable causal
state for this ABC population and linear Stage-0 test.

## Executed prospective contract

- Sequential diagnostic only: the exact fit256/score64 D405 train split,
  motion strata, masks, and shuffled donor map from the preceding PCA32 study
  were reused without reselection.
- Validation and protected test were unsupported and unopened.
- Inputs were top RGB frames 0--4, measured history state only for observed
  robot masks, and planned action chunks 4--11.
- Future top RGB/state at boundaries 5--12 were target-only.
- The rendered YAM robot union at each transition was dilated by 12 pixels and
  removed before computing the 45x80 event fields.
- The explicit future target was `[8,4]`: at each transition, spatial means of
  positive and negative luminance change and signed horizontal and vertical
  event-weighted flow. Flattening produced 32 scalars with no learned target
  encoder.
- Equal-width ridge designs compared `[history16, zeros32]` with
  `[history16, action_PCA32]`; fit-only five-fold CV selected alpha 100 for
  both models. Action PCA retained 99.34% of fit variance.
- Controls were aligned, same-stratum episode shuffled, fit-mean, raw zero, and
  nonwrapping action shifts -1/+1.
- Three mandatory effects shared a 10,000-draw score-episode bootstrap with
  seed 20260814 and one-sided Bonferroni 98.3333% lower bounds.

The registered implementation is commit
`376ee3bf41302b33af1d02c42f08d5f5a36e287d`, source SHA-256
`2d9037ccbb9baf16e93155c0c0bc589f1191f41e398436bea2bf858f0f9138c8`.
Registration was sealed before token scoring.

## Mandatory all-token effects

Positive percentages favor aligned action. Every mandatory comparison required
at least 5% gain, a strictly positive familywise lower bound, and at least 60%
favorable clips.

| Aligned reference | Reference MSE | Aligned MSE | Relative gain | Simultaneous lower bound | Favorable clips | Gate |
|---|---:|---:|---:|---:|---:|---|
| History only | 0.590755 | 0.574391 | +2.770% | -6.393% | 29/64 | fail |
| Episode shuffled | 0.752012 | 0.574391 | +23.619% | +2.190% | 43/64 | pass |
| Train mean | 0.585993 | 0.574391 | +1.980% | -7.621% | 29/64 | fail |

The shuffled comparison establishes that arbitrary other-episode commands can
be harmful; it does not establish that the aligned command contributes beyond
observed history or a typical command. Those two stronger causal controls both
failed all three gate components.

Raw-zero action had MSE 0.700362, making aligned appear 17.99% better, but zero
absolute joint commands are outside the native action distribution. Shift -1
and +1 had MSE 0.592351 and 0.576735, respectively. Aligned gained only 3.03%
and 0.41% against them, with lower bounds below zero.

## Component diagnostics

| Token family | Reference | Relative gain | Simultaneous lower bound | Favorable clips |
|---|---|---:|---:|---:|
| Change mass | History only | +7.060% | -0.730% | 34/64 |
| Change mass | Episode shuffled | +15.974% | +3.162% | 40/64 |
| Change mass | Train mean | +5.171% | -2.705% | 33/64 |
| Signed transport | History only | -0.537% | -9.726% | 24/64 |
| Signed transport | Episode shuffled | +28.269% | +6.256% | 37/64 |
| Signed transport | Train mean | -0.428% | -10.039% | 21/64 |

These were prespecified diagnostics, not substitutes for the all-token gate.
Change mass contains a directional action signal, but its episode consistency
and uncertainty fail the same standard. Signed transport contributes no gain
beyond history or mean action. The disagreement between shuffled and
history/mean controls is therefore substantive, not an averaging artifact.

## Target salience and scope

The fallback did not fail because all masked targets were zero:

- 32/32 fit token dimensions had raw standard deviation above `1e-6`;
- 96.875% (62/64) of score clips exceeded the `1e-5` event-mass threshold;
- mean score event mass was 0.02956;
- the global integrals reconstruct exactly from the explicit tokens by
  construction.

It failed predictability and causal-value gates. Unlike PCA32, this result is
not confounded by a learned target encoder that transfers poorly across
episodes. Conversely, global integrals deliberately discard object identity
and spatial location, so the result cannot rule out an explicitly
object-centric/contact-centric state.

## Operational evidence and artifact identities

- Slurm job: `507317`, B200 `pool0-0015`, `COMPLETED 0:0` in 51 seconds.
- Peak RSS: 7.20 GiB; disk read: 332.18 MiB.
- Robot rendering: 3.580 ms mean and 4.241 ms p95 over 3,520 post-warmup poses.
- Registration identity:
  `f6c6e70a12ff35f462d9d0f712d455b7c60b8cc93bd296d43f2683c5adb95ebb`.
- Analysis identity:
  `f05e713bbd3a7642ebd6f1cc04667fe173c5bcdc280fe097c7dc6459e35b0caf`.
- Completion identity:
  `620c5fc699a96586d301afdeb9711c8bc1f133da58d52ba7599c4c2f56c4a3c7`.
- Read-only audit identity:
  `d7ca3ea764f4f5d5f9df953a7af14176a46a47b10251894c33974a42da6ecd79`.
- Audit status: `audit_passed`; it checked 320 provenance rows, 64 score
  rows, 1,418 explicit false flags, three mandatory effects, and 15 diagnostic
  effects.
- Canonical artifact root:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/interaction_event_bottleneck/interaction-token-train256-score64-seed20260814-376ee3b-v1`.

## Evidence-based next step

Do not spend a video-training budget on PCA32 masked event fields or these
global event tokens. The honest next hypothesis, if the broader interaction
direction is retained, is a new object/contact-centric representation whose
state includes location and identity—for example tracked object slots with
relative gripper pose, contact/proximity, and object displacement. It should
first pass the same history, mean, shuffled, and timing controls on unseen
train episodes and demonstrate target relevance before any generator
integration. That would be a new research line, not a continuation or rescue
of the failed preregistered representation.
