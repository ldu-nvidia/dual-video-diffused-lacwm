# Final compact object/contact distribution Stage-0 result

Date completed: 2026-08-08

Decision: **`CLOSE_INTERACTION_BRANCH`; do not launch a stochastic
interaction generator or integrate this proxy into video diffusion**

## Central finding

Exact planned action did not materially improve the frozen compact
distribution over future object/contact outcomes. The aligned arm had slightly
better mean NLL than the three controls, but its gains were only 0.0173,
0.00734, and 0.00391 nats per target dimension versus history-only,
within-stratum shuffled action, and action-stratum centroid. All were below the
registered 0.05 threshold and all familywise simultaneous lower bounds were
negative.

The other proper scores supplied no rescue. CRPS and WIS gains were mostly
below 1.5%; energy was worse than history-only and shuffled action. Every one
of the 12 mandatory proper-score contrast gates failed. Coverage and target
salience passed, so neither badly calibrated intervals nor an empty target
explains the stop.

Together with the prior deterministic conditional-mean failure, this rejects
both tested uses of this interaction proxy: a fixed linear mean and this fixed
K=32 compact conditional mixture. It does **not** reject every possible
semantic or physical world state, but it provides no evidence-based authority
to train a stochastic interaction residual or video generator from this
branch.

## Frozen study

- Fit: 256 train episodes; bandwidth calibration: the previously opened 64;
  final score: 31 train episodes split 8/8/8/7 across the four motion strata.
- At registration, the final31 had not been opened by this interaction-event
  sequence. This is branch-scoped provenance only; it is not a claim that the
  episodes were globally unused across the entire research program.
- Target: seven standardized scalars for log event mass, event-time centroid,
  x/y event centroid, signed u/v motion, and peak contact score.
- Arms: aligned action, history-only, within-stratum episode-shuffled action,
  and action-stratum centroid. Every arm used 32 equal-weight Gaussian
  components; calibrated sigma was 0.65 for history and 0.45 for action arms.
- Scores: NLL per target dimension, multivariate energy, marginal CRPS, and
  WIS. The 12 contrasts shared a 20,000-draw episode bootstrap and one-sided
  12-way Bonferroni confidence of 99.5833%.
- Validation and protected test were not accessed. No generator, video metric,
  or control rollout was evaluated.

## Frozen mean scores

Lower is better for all four proper scores.

| Arm | NLL (nat/dim) | Energy | CRPS | WIS |
|---|---:|---:|---:|---:|
| **Aligned action** | **1.079394** | 1.501916 | 0.463989 | **0.293728** |
| History only | 1.096659 | 1.495667 | 0.467983 | 0.295813 |
| Episode shuffled | 1.086729 | **1.489745** | **0.462001** | 0.295018 |
| Action-stratum centroid | 1.083302 | 1.548651 | 0.469603 | 0.297953 |

The marginal ranking is mixed: aligned is best for NLL and WIS, while shuffled
action is best for energy and CRPS. The registered decision depends on the
simultaneous contrasts below, not rank alone.

## Mandatory proper-score effects

Positive values favor aligned action. NLL required at least 0.05 nat/dim;
energy, CRPS, and WIS required at least 5% relative gain. Every contrast also
required a strictly positive simultaneous lower bound and at least 60%
favorable final episodes.

| Score | Aligned reference | Gain | 99.5833% simultaneous lower bound | Favorable | Gate |
|---|---|---:|---:|---:|---|
| NLL | History only | +0.017265 nat/dim | -0.159344 nat/dim | 20/31 | fail |
| NLL | Episode shuffled | +0.007335 nat/dim | -0.074063 nat/dim | 16/31 | fail |
| NLL | Action-stratum centroid | +0.003908 nat/dim | -0.066480 nat/dim | 13/31 | fail |
| Energy | History only | -0.418% | -7.241% | 17/31 | fail |
| Energy | Episode shuffled | -0.817% | -7.281% | 12/31 | fail |
| Energy | Action-stratum centroid | +3.018% | -2.342% | 18/31 | fail |
| CRPS | History only | +0.853% | -6.152% | 20/31 | fail |
| CRPS | Episode shuffled | -0.430% | -8.088% | 18/31 | fail |
| CRPS | Action-stratum centroid | +1.196% | -3.703% | 17/31 | fail |
| WIS | History only | +0.705% | -6.645% | 17/31 | fail |
| WIS | Episode shuffled | +0.437% | -5.671% | 14/31 | fail |
| WIS | Action-stratum centroid | +1.418% | -3.523% | 14/31 | fail |

All 12 proper-score gates failed. Even the two rows with 20/31 favorable clips
missed the registered magnitude and simultaneous-lower-bound requirements.

## Coverage and salience

| Check | Observed | Registered requirement | Gate |
|---|---:|---:|---|
| Aligned central-80% coverage | 88.02% | 70--90% | pass |
| Aligned central-95% coverage | 97.24% | 87--100% | pass |
| Nonzero final targets | 24/31 (77.42%) | at least 50% | pass |
| Motion strata with nonzero targets | 4/4 | all four | pass |

The representation therefore generated nontrivial outcomes with calibrated
uncertainty. The failed proper-score family is specifically a failure to show
incremental predictive value from correctly aligned planned action.

## Execution and audit chain

- Registered scientific source: commit
  `c0e7ea456634f64790146c458f1301197f08e5e8`, SHA-256
  `99c1d5e363827b43c0ddc5162b8c1d321006f7f81610a1bf70011fe13724fa9f`.
- Protocol SHA-256:
  `c5180f4f629b149cb0ef5b3eb69ffff34102ec0511a2c4d97d6536248f2cb863`.
- Registration identity:
  `b39d6b2760bd970c9903cf663ab28387e0497668f044664789cc739475ee7225`;
  calibration identity:
  `ff6faca4537b7000572d8189135599b6138099690a0cbbf8d82ff9f18b2ffaea`.
- Two target-blind execution repairs left the registered scientific source
  unchanged. Repair-1 registration identity was
  `55410442f1c61c7409d570546b0b1eb7a557f0469f4d879d42c573a959d39653`;
  repair-2 registration identity was
  `200371a4fdba9c99e874770ad49df3edcd081d9ff1933f9389640cb5c8aa23de`.
- Before repair-2, job `507412` had written proper-score bytes but no terminal
  analysis. They were copied and hashed opaquely without JSON inspection.
  Job `507475` then completed 0:0 on `pool0-0242` in 31 seconds and reproduced
  the score JSONL byte-for-byte: 44,342 bytes, SHA-256
  `6f5db5bea50f5df9857a8f4b55b3ac1db726fad9e022cce6799b2e00437e1b83`.
  Provenance also matched exactly: 214,213 bytes, SHA-256
  `66ca7daa0055ead0b4fd1bbb98abc36054e00c16767a6b9531ce4a496dacf200`.
- Analysis identity:
  `1712b3de280179505b1588b033df3307ce093cdab5c7494d37251ae2309fa472`;
  completion identity:
  `57d61c742993cf5b7e1e043e72b49d1b3610be5352716d923534fde7a7e4eb4d`.
- In-job scientific, repair-1, and repair-2 audits all returned
  `audit_passed`, with identities `b9d0083454d9805b18ca88fab37a35c9e63d454293da03eba498956528c2463f`,
  `ddd94e883f755454d8f58c931de533a5251178f04cda5b8fc6bb9719ef4caf09`,
  and `e17d05ba660b4d2946cf1ad82c995681c5a97065b5ecd7c460dd98f18d9f12e1`.
- Independently rerun scientific, repair-1, and repair-2 audits also returned
  `audit_passed`, with identities `bf99753b765f52b70e5a330c022b60ab85368aa5a1f237b70b36bb65ee321786`,
  `d8b9565a6fc54451bfc80ecf53104fb30ac6b276ee087e371cdba6ece001dd6e`,
  and `9395f522ba4c1db1ebc17a0bc70c02fc0f7849fdc0b4821300374fb4eaa610ce`.
- A separate root-process replay of the unchanged scientific audit passed with
  identity `1690918e9a6c8edc686bbc7720869859061b12da304b5d4cc96532683d69af6b`.
  It verified 31 provenance rows, 31 score rows, all 12 recomputed effects,
  zero overlap within the registered partitions, and the same close decision.
- Canonical artifact root:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/interaction_event_bottleneck/object-contact-distribution-fit256-cal64-final31-seed20260816-c0e7ea4-v1`.

## Evidence-based consequence

Close the interaction-event/object-contact proxy branch. The earlier
deterministic test showed that its conditional mean was not action-predictable;
this final proper-score test shows that the preregistered compact conditional
distribution is not materially improved by aligned action either. No
stochastic object residual screen, Wan run, dual-denoising integration, or
real-time DAgger claim is authorized from these results.

Failure is scoped to this fixed representation, population, and simple
conditional mixture under a strict small-sample familywise gate. Reopening the
broader physical-state direction would require a genuinely distinct,
prospectively registered representation and new evidence; post-hoc rescoring
of these 31 episodes is not an admissible continuation.
