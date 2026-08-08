# Temporal Video Latent Forcing D0 result

Date executed: 2026-08-07

Status: complete development-split diagnostic; no protected-test access and no
candidate-model metric

## Immutable execution

- Slurm job: `504808`, `COMPLETED`, exit `0:0`, one NVIDIA B200, 11m04s
- source: `a3729a3a05cb4d4864c9308c3b590ddac385455e`, clean detached checkout
- frozen checkpoint SHA-256:
  `f7586f23030a489fc6a673ea3bb6c6cfecccdbe5269c62f6de697d1dc4f9f9cc`
- validation target SHA-256:
  `ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034`
- protocol SHA-256:
  `d1d5b22853c598fdfa62984f12975220534341eaf5b6f492d8757a1b3b2a7947`
- clips: 890; records: 143,290; expected records per clip: 161
- exact `t=0` training-distribution/autonomous state and prediction identity:
  passed
- teacher calls: zero
- clean future target entered autonomous sampler: false
- protected test accessed: false

The complete record is at
`$BASE/artifacts/dual_video_diffusion/temporal_followup/temporal-followup-seed1234-20260807-v1/diagnose-incumbent-flow-a3729a3-seed20260801-v1`.
The merged per-clip record SHA-256 is
`be229fa9d3b06963d2d52445362891dfd60c48b453c142ad36bc724fd23ab933`
and the aggregate summary SHA-256 is
`0cf58155e7929cebc0aa96a8fae9325a77c5f1aec80c3c77de2caa6d1c1b7175`.

## Frozen primary diagnosis

The primary cell is the prediction before call 2 of exact four-call uniform
Euler, at clean time `t=0.25`.

| state/query | semantic NMSE | token cosine | temporal NMSE | temporal cosine | velocity NMSE |
|---|---:|---:|---:|---:|---:|
| training-distribution corruption | 0.291555 | 0.834952 | 0.957870 | 0.173932 | 0.255496 |
| autonomous four-call trajectory | 0.686286 | 0.570520 | 1.094709 | 0.023666 | 0.483028 |

Autonomous temporal NMSE is 14.2857% worse than the paired
training-distribution query. The preregistered 10,000-resample paired bootstrap
(seed 20260807) gives a one-sided 95% lower bound of 14.0571% and descriptive
5th/95th percentiles `[14.0571%, 14.5145%]`.

The frozen rollout-drift-primary rule does **not** pass: training-distribution
temporal NMSE is not at most 0.50, and the rollout worsening and its lower bound
are not at least 25%. The required classification is therefore
`d0_does_not_identify_a_unique_primary_cause`.

## Endpoint and attribution observations

For the registered uniform Euler sampler, increasing calls makes the autonomous
endpoint worse:

| actual calls | semantic NMSE | token cosine | temporal NMSE | temporal cosine |
|---:|---:|---:|---:|---:|
| 1 | 0.614729 | 0.613074 | 1.043984 | 0.020876 |
| 2 | 0.672051 | 0.575996 | 1.185991 | 0.016516 |
| 4 | 0.768252 | 0.533453 | 1.374751 | 0.013159 |
| 8 | 0.855209 | 0.505115 | 1.546234 | 0.013086 |

At the primary autonomous state, replacing actions with any of the three fixed,
episode-disjoint action permutations changes temporal NMSE only from 1.094709
to 1.095509--1.095663. This is factual attribution evidence, not a
counterfactual-causal claim.

## Interpretation and next registered test

This diagnostic observes both denoiser error under the training corruption law
and additional autonomous-trajectory degradation. It does not uniquely assign
causality. The especially poor training-distribution temporal metrics and weak
action attribution are consistent with the already measured static-scene loss
dominance: the incumbent can recover average semantic appearance much better
than future change.

The result therefore preserves the frozen D2 comparison and prioritizes its
representation/loss arms (`ABS-T`, `DELTA`, `DELTA-T`) before interpreting the
self-roll-in arm (`DELTA-R`). No solver from D0 replaces the preregistered D3
sampler, and no protected-test cache is opened unless a candidate passes the
generated-only development gate.
