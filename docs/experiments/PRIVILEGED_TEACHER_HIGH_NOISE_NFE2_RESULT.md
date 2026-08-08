# Privileged teacher high-noise NFE=2 result

Date: 2026-08-08

Status: **PASS — `ELIGIBLE_FOR_STUDENT_SCREEN` on the frozen disjoint-train
calibration gate. No student was trained.**

## Result in one sentence

On 64 train episodes probe-disjoint from the NFE-4 diagnostic, the clean-future V-JEPA teacher improved the
sole NFE=2 high-noise video-velocity query on **64/64 units**, reducing mean
velocity MSE by **89.949%** versus feature-off and **80.020%** versus an
episode-shuffled clean-feature teacher; all five preregistered eligibility
conditions passed.

## Frozen experiment

- Probe source: `d028fc759940459b803c8b5e4b6ea28a31beeba5`.
- Checkpoint source: `656086686dae723c942a4209a9d71cdb17ed6ccc`.
- Data: immutable ABC/V-JEPA **train** manifest indices 64--127. The 64 selected
  episodes were unique and had zero episode overlap with the adaptively
  inspected NFE=4 indices 0--63.
- Schedule: NFE 2, observed as one auxiliary-only call followed by one
  video-active call. Every scored row was rollout step 1 with
  `video_sigma=1.0` and `tf_sigma=0.0`.
- Query: feature-off, aligned clean-future V-JEPA, and episode-disjoint shuffled
  clean V-JEPA were evaluated at the same feature-off video/auxiliary state.
- Statistics: 64 clip units and 10,000 paired clip-bootstrap replicates, seed
  `20260809`; thresholds were unchanged from the NFE=4 probe.
- Validation, test, and lockbox samples were not opened. W&B was disabled and
  no student optimization occurred.

## Frozen gate result

Positive percentages mean lower error for the aligned teacher.

| Requirement | Observed | Paired 95% interval | Result |
|---|---:|---:|---|
| Aligned teacher better on at least 60% of units | 64/64 = 100.000% | not applicable | pass |
| Aligned vs feature-off velocity gain at least 5% | 89.949% | [89.418%, 90.463%] | pass |
| Aligned vs shuffled velocity gain at least 3% | 80.020% | [78.233%, 81.685%] | pass |
| Aligned full rollout improves decoded MSE over off | 79.158% | [77.686%, 80.589%] | pass |
| Aligned full rollout improves temporal MSE over off | 63.519% | [60.393%, 66.833%] | pass |

The conjunctive decision is `ELIGIBLE_FOR_STUDENT_SCREEN`.

## Exact velocity and rollout metrics

The velocity-MSE means were **1.726449 off**, **0.173518 aligned**, and
**0.868438 shuffled**. Mean squared teacher--student velocity distance was
`1.232693` (range `0.781677`--`1.540031`), so teacher reachability by a small
causal adapter is not established. All 64 aligned feature hashes were unique,
and every aligned/shuffled pair differed.

| Complete NFE=2 rollout metric | Off | Aligned | Shuffled | Aligned vs off | Aligned vs shuffled |
|---|---:|---:|---:|---:|---:|
| Future latent NMSE | 2.679009 | 0.267112 | 1.350598 | 90.029% [89.480%, 90.542%] | 80.223% [78.518%, 81.805%] |
| Decoded MSE | 0.071235 | 0.014847 | 0.080484 | 79.158% [77.686%, 80.589%] | 81.553% [79.711%, 83.347%] |
| Temporal-difference MSE | 0.046924 | 0.017118 | 0.047459 | 63.519% [60.393%, 66.833%] | 63.930% [60.439%, 67.452%] |

Unlike the earlier NFE=4 full oracle diagnostic, this rollout has only one
video-active call. Its endpoint improvement therefore isolates clean-feature
conditioning at the high-noise video update under the NFE=2 sampler.

## What this does and does not establish

This disjoint calibration confirms that the earlier high-noise effect was not
limited to the 64 clips on which it was discovered. It establishes a strong,
sample-specific **privileged teacher** at the first NFE=2 video update.

It does **not** establish deployable video improvement: clean future V-JEPA is
unavailable at inference, no feature-free student was trained, and no
validation or protected split was evaluated. The teacher--student distance is
large, so a causal student may still fail to recover the privileged residual.

The sole NFE=2 video query occurs at `video_sigma=1` before any video update.
At this initial pure-noise node, a student-visited state and the ordinary
forward-noised pure-noise state coincide. Consequently this result does not
demonstrate a specifically **on-policy** advantage and weakens an OPD novelty
claim. The first controlled student screen should prioritize the closest prior,
`PFD-VIDEO`: a detached teacher-minus-student video-velocity residual learned by
a small zero-initialized causal adapter, with matched continuation, aligned,
and episode-shuffled controls. An on-policy advantage cannot be claimed unless
a later study demonstrates benefit at noninitial student-visited states.

Most importantly, the original frozen NFE=4 result remains
`STOP_NO_ELIGIBLE_TEACHER`. Its all-video-state gate failed because the aligned
teacher helped 64/64 units at `video_sigma=1` but 0/64 near
`video_sigma=0.024414`. This NFE=2 mechanistic follow-up neither reinterprets nor
overwrites that decision.

## Independent audit and immutable evidence

- Slurm job `506079`, B200 `pool0-0193`, completed in 2m19s with exit code 0.
- Artifact root:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/privileged_on_policy_teacher/privileged-opd-high-noise-nfe2-train64-index64-seed20260809-d028fc7-v1`
- Registration identity:
  `1bc1d8aeba1c132ec132cc1d348d2a5ad856beee11f084e78f405ca4da94c282`.
- Analysis identity:
  `0a4cbb08c8d11cd1702e06980be604851dbb2baec7ddadb19f8b611d6239f0ad`.
- Completion identity:
  `bce3d10d1ed2e34e553df9a1afd9791015a9eef8585bb1e137c241c10f8eac04`.
- Embedded audit identity:
  `6dd971b9175dfc760267b9068bb4d8854e6d8ee38379954be5b47894b73373ef`.
- Unit rows SHA-256:
  `28c83c5c8f0cf963b9e1271d9dfedb9100147225908bd1ff4ef47617679cea7b`.
- Rollout rows SHA-256:
  `ba0c25cf8be4f012a74e1b64c45e8386b09d367114308a6300545ca738d841e8`.
- Analysis file SHA-256:
  `79abf7bd237ee1481678b062765c449e689ba0497d211561fd89ce9f674a0eb6`.
- Snapshot SHA-256:
  `70d79533460680d836c7178baea2232cd4b8f309146a7571db0002f91ce53f61`.
- Resolved config SHA-256:
  `88ef78f4aa12050f50f393b1732c7804b39047513da7d7486cd932d4d2525a0d`.
- Train manifest SHA-256:
  `eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74`.

The independent read-only audit revalidated all registration, completion,
analysis, audit, unit-row, and rollout-row identities; rehashed every output
and registered input including the checkpoint; reproduced the 10,000-replicate
analysis exactly; and confirmed 64 unique follow-up episodes, zero parent
episode overlap, 320 expected/observed Wan calls, one `sigma=1` active video
node, and `student_training_launched=false`.
