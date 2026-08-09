# VPM one-step direct-residual frontier result

Date completed: 2026-08-09

Decision: **`STOP_VPM_DIRECT_RESIDUAL`**

## Central finding

A small causal residual head does not materially improve the actual VPM@1
frontier.  On the fresh 31-episode reserve, the aligned head improved corrected
velocity MSE by only 0.55%, decoded MSE by 0.50%, and latent NMSE by 0.41%
relative to unmodified VPM.  Every corresponding confidence interval crossed
zero and every point estimate was far below the frozen 3% threshold.  Temporal
MSE instead worsened by 0.44%, with the entire paired interval below zero and
only 7/31 episodes favorable.

The episode-shuffled head was essentially identical to VPM-off.  Aligned beat
shuffled by only 0.55% velocity, 0.61% decoded, and 0.46% latent error, while
temporal error worsened by 0.16%.  This is not evidence for a sample-specific
ordinary-residual correction at the VPM one-step seam.

This negative control sharpens the prior privileged-residual result.  A direct
head repaired the deliberately weaker J1 parent, but neither clean-V-JEPA
privileged transfer nor ordinary direct residual learning adds a useful
one-step correction to the stronger VPM parent under this token-local linear
capacity.  VPM@1 therefore remains the feature-free baseline for every
subsequent auxiliary claim.

## Frozen execution

- Slurm job `507331`: `COMPLETED`, exit `0:0`, elapsed `00:05:00`, one B200 on
  `pool0-0024`.
- Source commit: `4f75f9c08dbd64f7ad7d13373a98f25e397f7f4b`.
- Fit rows: immutable ABC train 128--383; calibration rows 384--415.
- Fresh outcome rows: 480--510, 31 episode-disjoint clips and four new Gaussian
  noise seeds per clip.
- Row 511 was prospectively excluded because the historical dataset constructor
  had already read it for structural array validation.  Rows 416--479, all
  validation data, and protected test data were excluded.
- The selected head used a frozen 1,024-channel projection, ridge penalty 1.0,
  and 65,600 affine parameters.  Selection used calibration corrected-velocity
  MSE only.
- Serving used history, planned actions, initial noise, and the frozen VPM trunk
  only.  Teacher calls and V-JEPA target-array opens were exactly zero.

## Fresh endpoint metrics

Lower is better.  Means cover 31 episodes x four outcome noise seeds.

| Endpoint | Velocity MSE | Latent NMSE | Decoded MSE | Temporal MSE | Residual R2 | Residual cosine |
|---|---:|---:|---:|---:|---:|---:|
| VPM-OFF / ZERO | 0.129447 | 0.201646 | 0.0154269 | **0.0127203** | 0 | 0 |
| DIRECT-SHUFFLED | 0.129456 | 0.201758 | 0.0154436 | 0.0127561 | -0.00132 | 0.0350 |
| DIRECT-ALIGNED | **0.128740** | **0.200827** | **0.0153498** | 0.0127762 | 0.00265 | 0.0662 |

The aligned residual R2 of 0.00265 is qualitatively different from the 0.8785
R2 observed when a matched direct head repaired J1.  At this VPM seam, the
frozen trunk exposes almost none of the remaining target residual to this
linear token-local correction.

## Paired effects

Positive percentages mean lower error for DIRECT-ALIGNED.  Intervals are the
registered 10,000-replicate episode-clustered bootstrap; the four noise draws
remain inside each episode.

| Comparison | Velocity | Latent | Decoded | Temporal |
|---|---:|---:|---:|---:|
| aligned vs VPM-off | +0.546% `[-0.248,+1.279]` | +0.406% `[-0.334,+1.078]` | +0.500% `[-0.167,+1.168]` | **-0.439% `[-0.657,-0.231]`** |
| aligned vs shuffled | +0.553% `[-0.068,+1.180]` | +0.461% `[-0.090,+1.025]` | +0.608% `[-0.132,+1.370]` | -0.158% `[-0.371,+0.071]` |

Episode-favorable fractions versus VPM-off were 19/31 for velocity and latent,
18/31 for decoded MSE, and only 7/31 for temporal MSE.  Versus shuffled they
were 20/31, 20/31, 17/31, and 16/31, respectively.  All registered quality
gates failed; only the latent non-regression guardrail, serving constraints,
latency, and evidence-integrity gates passed.

## Serving and evidence audit

- 64 shared Wan invocations / 124 sample calls evaluated four endpoints from
  identical noise with one Wan call per sample.
- `ZERO` and VPM-off were bit-exact for every final latent and metric row.
- The evaluator constructed clean targets only after all endpoints were
  materialized; future targets never entered the correction graph.
- Adapter p95 was 0.497 ms per batch, below the frozen 1 ms bound.
- Full aligned one-step latency, from resident observed RGB/actions through
  preparation, Wan, adapter, Euler update, and decode, was 402.20 ms mean and
  402.56 ms p95 per batch.  This is an evidence endpoint, not a real-time
  single-rollout claim.
- The completion contains 496 outcome rows and 64 complete timing rows.
- The independent audit recomputed row inventory, paired hashes, zero no-op,
  timing trace, role-scoped accesses, and eight artifact digests.  A separate
  post-run digest check also found zero mismatches.

Artifact identities:

- registration: `e9d4ce9c6299c42d4da0f92302cd8d9ba03ca0b3fcaf6b5c8e622e9707621448`;
- fit: `4bfb8a6085db657bbf846f03d3880d4c115e2a4ca025a9395bd75a62e9e1fe7d`;
- endpoint: `6d3d4f4f1afeef5bb85046655d33bf019f1fb5367028edbee9d9e9468034242c`;
- analysis: `6b15b3865925745f2e889585078f66238707315d695ddebad63ae43307ecf67d`;
- completion: `0bf783dccb6105cbd7166014e71bfa24d77d3a91aff624b6e8602532ba16dffc`;
- audit: `db093d20647829b67e515f72f4b56f240ddf935a8c11973d2bc917d20b3140fb`.

Canonical artifact:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/vpm_direct_residual_frontier/
  vpm-direct-residual-fit256-outcome31-seed20260832-4f75f9c-v2/
```

## Consequence

Do not spend the next run on a nonlinear continuation of this ordinary
residual head: the registered pass condition was not close, and the temporal
effect is directionally harmful.  Retain unmodified VPM@1 as the frontier.

The next auxiliary must add a causal inductive variable rather than merely
relearn the same clean-video residual.  The already frozen tests target that
claim directly:

1. fixed raw-command robot geometry/flow, which imports known embodiment
   computation before Wan;
2. localized object/contact slots, which test compact interaction state rather
   than global video summaries; and
3. orthogonal 3-D Haar residual bands, which must beat FULL-DIRECT to establish
   a frequency-specific optimization advantage.

A pass in any of these screens must still beat VPM-off at equal Wan calls and
cannot be attributed to direct residual capacity.  A fail would be evidence to
stop that mechanism, not evidence that every possible dual-video architecture
is impossible.

## Claim boundary

This is one train-only reserve, one frozen VPM checkpoint, four noise draws,
and a linear token-local residual head.  It does not exclude nonlinear
fine-tuning, other checkpoints, other datasets, or a genuinely informative
causal auxiliary.  It establishes that ordinary direct residual learning at
this seam is not a hidden stronger baseline and does not improve the tested
one-step endpoint.
