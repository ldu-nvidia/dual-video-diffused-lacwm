# VPM generated-motion-prior probe (preregistered)

## Status and question

This is a **validation-only, inference-only, training-free** probe. It is
specified before any metric from these endpoints is computed. It asks whether
a motion prior extracted from a cheap, autonomous VPM video sample can improve
a second VPM sample without clean future features at inference.

The protected test split will not be opened, scored, selected on, or used to
tune this probe. A positive result is validation-only mechanistic evidence; a
negative result retires this exact configuration, not all self-derived video
conditioning.

The frozen generator is the update-1,000 parameter-matched video-only arm
(`VPM`) from `vjepa2-controlled-20260730-seed1234-9cf8e69-v3`, trained at
commit `9cf8e6922f35a5d6645e3128545953723bf54da2`. Its auxiliary state and clock
gates are exactly zero, so this probe tests video-latent guidance rather than
the failed V-JEPA branch.

## Motivation and source

[PhaseLock](https://arxiv.org/abs/2606.06361) extracts temporal latent deltas
from a two-step generated video and injects the delta residual during the early
half of a second trajectory. The public reference implementation inspected for
this adaptation is
[`dnwjddl/phaselock@4ebb65b`](https://github.com/dnwjddl/phaselock/tree/4ebb65ba1348e754723f08a190723cf82294db19).

The adaptation is intentionally narrow:

- PhaseLock's `[B,T,C,H,W]` delta operator is translated to Wan's
  `[B,C,T,H,W]` layout.
- The generated VPM latent is used directly; there is no avoidable
  decode/re-encode round trip.
- Only generated-future deltas are extracted. The first future latent is the
  anchor, observed history is never changed by guidance, and no clean
  history-to-future or future-to-future delta is supplied.
- The preliminary and refinement trajectories reuse the exact same initial
  video and auxiliary Gaussian noise, keyed by immutable validation clip ID.
- A sample-shuffled generated-prior control tests whether any benefit requires
  sample alignment rather than generic smoothing.
- Every comparator has exactly the same total number of Wan/transformer calls.

## Deployable input boundary

The sampler accepts only:

- five observed RGB frames;
- the action tensor and morphology ID available to LACWM; and
- deterministic Gaussian noise.

It does not accept clean future RGB, cached V-JEPA targets, any clean future
feature, or an online encoder/teacher call. The evaluator owns the remaining
eight RGB frames only after sampling, for scoring. A forward hook counts every
Wan call. The run is invalid if declared and observed call counts differ.
The input loader reads the registered RGB and action arrays directly and
reproduces ABC's fixed action padding/morphology mapping; it never opens the
clean V-JEPA target array at all.

## Latent-delta operator and guidance

Let the generated video latent be

```text
z in R^[B,C,T,H,W],  h = number of observed-history latent frames.
```

VPM encodes 5 observed and 8 future RGB frames into 2 history and 2 future Wan
latent frames. For an autonomous preliminary sample `z_few`, the one available
future-future motion delta is

```text
M_prior = D(z_few)
D(z) = z[:,:,h+1:] - z[:,:,h:-1].
```

At refinement step `k`, after the ordinary scheduler update and exact history
clamp, compute

```text
M_k = D(z_k)
G_k = M_prior - M_k
z_k[:,:,h+1:] <- z_k[:,:,h+1:] + lambda_k * G_k.
```

`z_k[:,:,:h]` and the first future latent `z_k[:,:,h]` remain unchanged. The
fixed PhaseLock strength is `lambda_0 = 0.05`. It decays linearly over the
early interval

```text
k in [0, max(1, floor(K_full/2)))
lambda_k = 0.05 * (1 - k / max(1, floor(K_full/2))).
```

The shuffled control replaces `M_prior` with a cyclic roll by one across each
fixed local two-clip batch. It preserves prior marginal scale and extraction
compute while breaking clip alignment. The evaluator records the donor sample
ID.

## Fixed endpoints and call accounting

`K_few` is the preliminary trajectory and `K_full` is the guided refinement.
Each aligned and shuffled endpoint reruns its own preliminary trajectory; it
does not amortize calls across endpoints.

| Endpoint | `K_few` | `K_full` | Total Wan calls | Role |
|---|---:|---:|---:|---|
| `ordinary_b1` | 0 | 1 | 1 | previously established VPM frontier |
| `ordinary_b3` | 0 | 3 | 3 | matched comparator |
| `ordinary_b4` | 0 | 4 | 4 | matched comparator |
| `ordinary_b6` | 0 | 6 | 6 | matched comparator |
| `phaselock_k1_f2_{aligned,shuffled}` | 1 | 2 | 3 | model-adapted early prior |
| `phaselock_k1_f3_{aligned,shuffled}` | 1 | 3 | 4 | model-adapted early prior |
| `phaselock_k2_f2_{aligned,shuffled}` | 2 | 2 | 4 | faithful two-step prior |
| `phaselock_k2_f4_{aligned,shuffled}` | 2 | 4 | 6 | faithful two-step prior |

`K_few=1` is primary because the frozen validation study found VPM NFE=1 to
be its sole conservative frontier point. `K_few=2` tests the setting used by
PhaseLock. These choices and `lambda_0` are not tuned after seeing probe
metrics.

Exact call equality isolates solver/guidance quality, but it does not prove
equal wall-clock latency: PhaseLock has a second scheduler pass and small
tensor operations. No latency or real-time DAgger claim will be made from this
probe.

## Dataset and execution

- split: all 64 pinned immutable validation clips;
- no image augmentation;
- one B200 node, eight ranks, two clips per rank;
- fixed rank shard: `rank, rank+8, ...`;
- sampling ID: `1,000,000 + clip_index`;
- same VPM checkpoint, history, actions, and initial noise for every endpoint;
- raw RGB decoded metrics and VAE-latent metrics only;
- no W&B logging and no protected test.

The registration step must run from a clean tool commit and the exact clean
historical-model checkout recorded by the frozen study. The study is pinned by
kind, ID, root, and identity SHA-256. Registration rehashes the 4.25 GB VPM
checkpoint before any candidate metric. Evaluation refuses a changed protocol,
checkpoint, validation manifest/RGB/action cache, VideoX checkout, or
repository. Validation rows and cache metadata must explicitly name the `val`
split. The validation RGB/action arrays are fully rehashed; the clean V-JEPA
target array is neither required nor opened.

## Metrics

Primary metric (lower is better):

- decoded temporal-difference MSE on `[last observed RGB, 8 predicted RGB]`,
  measured against the corresponding raw validation sequence.

Guardrails (lower is better):

- future video-latent NMSE;
- decoded raw-RGB MSE.

Diagnostics, never substitutes for a failed primary/guardrail gate:

- future-latent delta NMSE;
- decoded PSNR;
- per-clip tensor hashes, donor identities, guidance schedule, and actual Wan
  calls.

## Fixed comparisons and simultaneous decision rule

Each of four aligned candidates is compared, using the same 64 clips, with:

1. its same-configuration shuffled-prior control (sample-alignment
   attribution);
2. ordinary Euler at the exact same total Wan calls (equal-compute benefit);
3. `ordinary_b1` (the already established VPM validation frontier).

This creates 12 predeclared contrasts. For each metric, relative improvement is

```text
I = (mean(reference) - mean(candidate)) / mean(reference),
```

so positive is favorable. The analyzer uses 10,000 paired clip bootstrap
replicates with seed `20260807`. It reports descriptive two-sided 95% CIs and a
one-sided Bonferroni lower bound at confidence

```text
1 - 0.05/12 = 0.9958333333333333.
```

A contrast passes only if:

- decoded temporal MSE point estimate and simultaneous lower bound are both
  at least `+1%`; and
- future video-latent NMSE and decoded MSE point estimates and simultaneous
  lower bounds are each greater than `-1%`.

A candidate demonstrates a deployable generated-motion-prior benefit on
validation only if **all three** contrasts pass. If multiple candidates pass,
the deterministic tie break is:

1. fewer total Wan calls;
2. larger temporal improvement versus `ordinary_b1`;
3. fixed order `k1_f2`, `k1_f3`, `k2_f2`, `k2_f4`.

Irrespective of outcome, this probe does not open protected test. A failed
alignment contrast means any observed effect cannot be attributed to the
sample-aligned generated motion prior. A passed equal-call contrast but failed
`ordinary_b1` contrast may diagnose higher-NFE phase erosion, but it is not an
improved low-compute frontier.

## Entry points and output layout

The implementation is:

- `tools/vpm_phaselock_probe.py` — registration, sampler, scoring, inventories;
- `tools/analyze_vpm_phaselock_probe.py` — paired bootstrap and fixed gates;
- `tools/slurm/vpm_phaselock_probe.sbatch` — non-requeueable B200 entrypoint.

Large outputs must be under Lustre:

```text
OUTPUT_ROOT/
  registration.json
  evaluation/
    rank_000.jsonl ... rank_007.jsonl
    rank_000.json  ... rank_007.json
    inventory.json
  analysis/
    analysis.json
```

All files are fresh-only and exclusively created. Failed attempts are retained;
the protocol does not overwrite or reinterpret them.
