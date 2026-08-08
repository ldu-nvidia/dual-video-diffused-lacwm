# Confidence-gated generated-auxiliary audit (prospective exploratory protocol)

Date frozen: 2026-08-08, before reading the full per-clip quality table for
this audit

Status: exploratory analysis of an existing one-seed, repeatedly inspected
64-clip validation set. This audit opens no protected split, launches no GPU
job, changes no checkpoint, and cannot support a confirmatory quality claim.

## Question

Can an inference-observable confidence signal make an already generated
auxiliary state **no regret** by using it only on clips for which it is likely
to improve one-call RGB generation, while falling back to the exact
same-checkpoint auxiliary-off endpoint otherwise?

The pinned study is the one-call intra-forward scratchpad experiment:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/
intra-forward-seed1234-20260808-d93d5ee-v3
```

The sole primary endpoint is NFE 1. The analysis uses only the deployable
`MID-ON` checkpoint cells `autonomous` (aligned generated state), `off` (exact
zero injection), and `future_shuffled` (corrupted future auxiliary bins with
observed bins preserved). `MID-OFF/autonomous` is a package-level secondary
reference, not a gating endpoint. Rows must cover the same 64 clip indices,
have no protected-test access, no oracle leakage, no future RGB, no clean
auxiliary input, and no online teacher calls.

## Leakage-safe confidence contract

A confidence feature is eligible only if all of the following are true:

1. it is a finite numeric value available before the final RGB prediction is
   accepted at inference;
2. it is computed from observed history, actions, noise, the generated
   auxiliary state, or model-internal predictions only;
3. it does not compare against future RGB, the clean auxiliary target, a
   decoded future target, or any target-derived metric; and
4. it is recorded per clip for aligned and corrupted-auxiliary cells.

The following row fields are explicitly **forbidden** as confidence features:
`video_nmse`, `decoded_mse`, `temporal_mse`, `auxiliary_future_nmse`,
`auxiliary_dc_nmse`, `auxiliary_motion_nmse`, and
`auxiliary_future_cosine`. Hashes identify tensors but do not expose numeric
confidence. Timing, memory, call-count, and batch-shape equivalence diagnostics
are audit metadata, not semantic confidence and are also ineligible.

Before quality values are analyzed, the artifact-schema audit will enumerate
every remaining per-clip numeric field. If no eligible, varying field exists,
the protocol forbids fitting a gate. In that case the honest result is an
oracle-selection ceiling plus an exact statement of the telemetry blocker.

If eligible features exist, fit the following rule without using the held-out
clip's target:

- outer leave-one-clip-out prediction;
- within each outer training fold, standardize using training clips only;
- fit ridge regression to the signed temporal-MSE improvement
  `(off - autonomous) / off` with `lambda` selected by inner leave-one-out
  from `{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`;
- select autonomous only when the out-of-fold predicted improvement is
  positive; otherwise select the exact off row;
- apply the already fitted rule to the same clip's `future_shuffled`
  confidence telemetry. A corrupted state that is rejected uses the exact off
  endpoint, not a new generation.

Because clips may share episodes and no episode identifier is promised by the
row schema, even a leave-one-clip-out result remains exploratory and may be
optimistic.

## Frozen estimands

For each metric, lower error is better. Relative improvement of candidate
`c` over reference `r` is

\[
100\,(\bar r-\bar c)/\bar r.
\]

Report paired 10,000-replicate percentile-bootstrap 95% intervals over clips
with seed 20260808 for:

1. always-on aligned versus always-off;
2. always-on future-shuffled versus always-off;
3. the confidence-gated aligned endpoint versus always-off, if a valid rule
   can be fit;
4. the same fitted rule under future-shuffled corruption versus always-off;
5. a target-leaking, per-clip perfect temporal oracle that chooses aligned
   exactly when its temporal MSE is below off; and
6. a stricter Pareto oracle that chooses aligned only when all three quality
   errors are no greater than off.

The perfect and Pareto oracles are unattainable ceilings, never deployable
gates. Report their autonomous selection rates, metric effects, and the
fraction of clips on which aligned improves each metric. A fitted gate's
retention is its temporal gain divided by the positive perfect-oracle temporal
gain.

The exploratory success criteria are:

- gated aligned retains at least 50% of the perfect-oracle temporal gain;
- gated aligned has nonnegative point improvement in all three metrics and a
  temporal 95% lower bound no worse than -0.5%; and
- under future-shuffled corruption, the rule selects off on at least 95% of
  clips, has nonnegative temporal point improvement over always-off, and has
  95% lower bounds no worse than -0.5% for all three metrics.

If no eligible confidence telemetry exists, `always off` is the only honest
no-regret policy in this artifact. Its oracle-gain retention is defined as
zero, and the audit must not substitute target-derived auxiliary accuracy,
latency noise, hashes, or validation quality metrics to manufacture a gate.

## Claim boundary

Passing would motivate prospective logging and a fresh validation experiment;
it would not demonstrate FVD, perceptual quality, multi-seed robustness,
real-time DAgger, or general video diffusion gains. Failing can reject only
post-hoc gating with the telemetry preserved by this study, not confidence
gating in principle.
