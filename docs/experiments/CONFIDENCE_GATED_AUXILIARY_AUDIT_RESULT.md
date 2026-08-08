# Confidence-gated generated-auxiliary audit result

## Decision

**No deployable confidence gate can be fit from the preserved artifact, and
even a perfect target-leaking selector has a negligible ceiling. Do not launch
a generator experiment from this result.**

The exact no-regret policy supported by this artifact is to keep the
same-checkpoint midpoint injection off for every clip. This retains 0% of the
unattainable oracle gain. It is safe under corrupted auxiliary input because
it reuses the already generated `off` endpoint exactly.

This is an exploratory audit of a one-seed, repeatedly inspected val64 set. It
does not reject prospective confidence gating if the missing inference-time
telemetry is logged in a new study.

## Frozen evidence and safety

The protocol was committed before the full per-clip quality table was read:

```text
protocol commit                    68ccce0cb199c8b6f489b172e9d1596a04c998c0
protocol file SHA-256              d223c47398b4ce02d4d84b214fc7491c06dc1e2b0b2f1d8b357fbacbf30237d1
MID-ON rows SHA-256                1373c2487cdfba00774a48e6ec20e21641e331b477758f789c1ee8e531a2907c
MID-OFF rows SHA-256               e174ffc11952f12a902f562063ea424e341b6f961b09042297694e3500c57165
analysis identity SHA-256          ab738404616755f49bbf938332fe6642ed068eaeeabd9854f0e9a4158ad92016
analysis file SHA-256              9d3b133216b5ad37e33ebf012e4ee5554e1fd641ef86a99e55c39e959e93554e
```

The source study is:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/
intra-forward-seed1234-20260808-d93d5ee-v3
```

Only NFE-1 `MID-ON` validation rows were used for the primary comparisons:

- `autonomous`: this clip's generated midpoint state is injected;
- `off`: the same checkpoint computes the state but injection is exactly zero;
- `autonomous_future_shuffled`: observed bins stay local while generated
  future bins are cyclically moved between samples; and
- `MID-OFF/autonomous`: a separately trained, parameter-matched package
  reference only.

All three primary cells contain the same 64 clip indices with paired target,
video-noise, and auxiliary-noise identities. Every row says deployable, no
protected-test access, no oracle leakage, no future RGB or clean auxiliary
passed to the sampler, and zero online-teacher calls. The audit launched and
changed no job.

## Why a confidence model could not be fit

The row schema has no varying, per-clip, inference-observable semantic
confidence value:

- generated auxiliary content is retained only as
  `auxiliary_final_sha256`; a hash is an identity, not a numeric feature;
- `auxiliary_future_nmse`, DC/motion NMSE, and future cosine all compare with
  the clean future target and are therefore unavailable at inference;
- `effective_state_gate=0.0139036868` and `effective_clock_gate=0` are global
  constants, not clip confidence; and
- the other varying numeric columns are target quality, latency, memory, call
  counts, or batch-shape-equivalence diagnostics. The prospective protocol
  forbids using them as semantic confidence.

Consequently, nested leave-one-out fitting was not attempted. Using a clean
future auxiliary error, output MSE, latency noise, or a hash would have created
a misleading post-hoc gate.

## Paired results

Values below are relative error improvements; positive is better. Intervals
are paired 10,000-replicate percentile-bootstrap 95% intervals over the 64
clips, seed 20260808.

| NFE-1 contrast | Video-latent NMSE | Decoded MSE | Temporal MSE |
|---|---:|---:|---:|
| aligned vs same-checkpoint off | -0.0038% `[-0.0414,+0.0346]` | +0.0636% `[+0.0109,+0.1159]` | +0.0326% `[+0.0153,+0.0503]` |
| future-shuffled vs same-checkpoint off | -0.0108% `[-0.0540,+0.0319]` | +0.0455% `[-0.0132,+0.1038]` | +0.0373% `[+0.0206,+0.0550]` |
| aligned vs future-shuffled | +0.0069% `[-0.0317,+0.0451]` | +0.0181% `[-0.0428,+0.0730]` | -0.0047% `[-0.0227,+0.0129]` |
| aligned vs separately trained MID-OFF | -0.1835% `[-0.8287,+0.4422]` | -1.1334% `[-1.9670,-0.3609]` | -0.2573% `[-0.4161,-0.0990]` |

The small same-checkpoint decoded/temporal changes are not evidence of useful
sample-specific forcing: corrupting the future auxiliary bins produces nearly
the same result, and every aligned-versus-shuffled interval contains zero.
Against the matched trained-off arm, MID-ON is worse in decoded and temporal
error.

Aligned beats same-checkpoint off on 34/64 latent, 39/64 decoded, and 36/64
temporal rows. The respective selection fractions are 53.1%, 60.9%, and 56.2%;
none supplies a deployable selection rule because the sign is known only after
clean-target scoring.

## Unattainable oracle ceilings

The perfect temporal oracle reads both future-target temporal errors and picks
aligned only when aligned is lower. It selects 36/64 clips (56.25%, bootstrap
95% interval 43.75%--68.75%):

| Target-leaking selector | Video-latent NMSE | Decoded MSE | Temporal MSE |
|---|---:|---:|---:|
| perfect temporal oracle | +0.0236% `[-0.0053,+0.0539]` | +0.0712% `[+0.0247,+0.1199]` | **+0.0447%** `[+0.0320,+0.0592]` |
| Pareto oracle | +0.0425% `[+0.0220,+0.0666]` | +0.0737% `[+0.0327,+0.1209]` | **+0.0241%** `[+0.0137,+0.0362]` |

The Pareto oracle reads all three target metrics and selects only the 19/64
clips on which aligned is no worse in every metric. Its selection rate is
29.69% (18.75%--40.63%). Both selectors are impossible at inference.

Most importantly, the *perfect* temporal oracle improves temporal MSE by
`0.0447%`, not `4.47%`. That is only 0.0121 percentage points above always-on
aligned. Even perfect knowledge of which clips benefit cannot turn this
midpoint scratchpad into a material quality improvement.

## What would make a future gate testable

A new evaluator would need to prospectively preserve, before target scoring,
at least one content-sensitive signal for aligned and corrupted states, such
as generated future-bin RMS by frequency band, midpoint residual RMS,
auxiliary self-consistency, two-noise/head disagreement, or a separately
trained confidence prediction. Calibration must use training-only episodes;
evaluation should be episode-grouped and use fresh validation clips.

That prospective experiment is worth running only for an auxiliary mechanism
whose same-checkpoint aligned-versus-shuffled effect is already materially
positive. Confidence gating cannot rescue an auxiliary path with a
`0.0447%` perfect-selection ceiling. The causal action-to-motion direction is
a better candidate because it has an attributable Stage-0 signal before RGB
generation; confidence should be treated as an uncertainty output of that
future scaffold, not as a post-hoc repair for this null scratchpad.

## Reproduction

```bash
python tools/audit_confidence_gated_auxiliary.py \
  --mid-on-rows /path/to/evaluation/mid_on/rows.jsonl \
  --mid-off-rows /path/to/evaluation/mid_off/rows.jsonl \
  --protocol docs/experiments/CONFIDENCE_GATED_AUXILIARY_AUDIT_PROTOCOL.md \
  --output /mnt/data1/path/to/confidence_gate_analysis.json
```
