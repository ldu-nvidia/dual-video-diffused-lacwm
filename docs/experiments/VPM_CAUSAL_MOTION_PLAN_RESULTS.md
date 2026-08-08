# VPM causal action motion-plan forcing screen: results

Date completed: 2026-08-08

This prospective screen tested whether a compact motion plan generated only
from five observed RGB frames, requested actions, morphology, and Gaussian
noise could mature before the first Wan call and improve low-NFE video
generation.  It used one seed, a planner fit on 448 training clips, 64 held-out
training-calibration clips, two matched 200-update video continuations, all 64
validation clips, and no protected-test access.

The complete artifact is:

`causal_motion_plan/camp-plan-first-seed1234-20260808-e4fda45-v3`

- sealed registration identity:
  `216ae036e688474e7ea3bdf3c07b6e2b412f16bd4870be1d94e34cdd5f899c6f`;
- planner snapshot SHA-256:
  `d53d1debdab1c8ac36a6c1fde445082ab320bc7268beb0ee2f2eb2f3e2b1bb04`;
- paired-training audit identity:
  `ee789da3005e7a55cceef5836fc3d8aad053da3d7418a02e4cafec1cf5e78c91`;
- structural audit identity:
  `e8c36c5b4fba1373f04a3bd3503c913bc19b77a1508160d3ff0c43478eaca581`;
- analysis identity:
  `8a396a260d5819dade891b8e49803f95f91644c86458729569c8e4093c8d2dc9`;
- audited validation rows: `1,280`;
- all twelve preregistered cells pass: false;
- protected test accessed: false.

## Configuration and causal contract

The planner generated a normalized `[B,16,2,6,30]` plan in two small Euler
calls.  The plan was upsampled per camera and injected into the existing Wan
adapter before video denoising.  Both video arms loaded the same frozen planner
and historical VPM parent:

| Arm | Planner calls | Generated plan retained | Runtime fusion |
|---|---:|---|---|
| `PLAN-OFF` | 2 | yes | exact off |
| `PLAN-ON` | 2 | yes | aligned on |

The paired audit verified all 200 updates had identical clip order, diffusion
timesteps, video noise, planner noise, action probes, parent/planner artifacts,
and parameter schema.  A sampler received zero future RGB, clean future latent,
teacher feature, or oracle plan inputs.  Evaluation used aligned, fusion-off,
globally plan-shuffled, and planner-action-shuffled endpoints.

## Planner result

The train-only two-call calibration diagnostic improved from NMSE `2.4008` and
cosine `-0.0007` at update 0 to NMSE `1.5923` and cosine `0.1690` at update
399.  On val64, the autonomous aligned plan reached mean NMSE `1.5480` and
cosine `0.1657`.

This is learned signal, but it is much too weak to serve as a useful future
motion backbone: its squared error exceeds the target energy and its direction
has only low positive alignment.

## Matched training result

At update 199, validation flow loss was `0.0974953` for `PLAN-OFF` and
`0.0977481` for `PLAN-ON`; plan fusion was therefore `0.259%` worse.  The
generated plan did not accelerate optimization in this controlled continuation.

## Primary NFE-1 result

Lower is better.  Relative improvement is candidate `PLAN-ON/aligned` versus
the named reference; positive values favor the candidate.

| Reference | latent NMSE | decoded MSE | temporal MSE |
|---|---:|---:|---:|
| candidate mean | 0.250582 | 0.0200753 | 0.0135421 |
| independent `PLAN-OFF/aligned` | -0.333% | -0.941% | -0.218% |
| same checkpoint, fusion off | +0.334% | +0.580% | +0.082% |
| same checkpoint, plan shuffled | +0.026% | +0.013% | +0.074% |
| same checkpoint, planner actions shuffled | -0.010% | -0.028% | +0.011% |

All three comparisons against the independently trained control regressed.
The small gains over fusion-off did not approach the preregistered 3% temporal
point threshold or 1% simultaneous lower bound.  Aligned versus plan-shuffled
and action-shuffled effects were essentially zero.  Consequently:

- the nine-cell generated-plan quality gate failed;
- the three-cell planner-action attribution gate failed;
- all twelve cells failed overall;
- descriptive NFE 2/4 results cannot rescue the primary NFE-1 failure.

The same-checkpoint fusion-off comparison indicates a small generic effect from
activating the adapter, but the shuffled controls show it is not useful
sample-specific motion guidance and is not attributable to requested actions.

## Latency claim boundary

The original evaluator measured `aligned_nfe_1` first and performed no explicit
CUDA/model warm-up.  Each rank's first sample therefore included cold-start
cost; the resulting `1.9506 s` p95 and `0.513` rollout/s value are not a valid
steady-state deployment estimate.  Later matched NFE-1 endpoints measured about
`0.233--0.265 s` mean and `0.249--0.313 s` p95, or roughly `3.19--4.02`
rollouts/s, but none is a warmed aligned endpoint with the exact declared
deployment condition.

Thus this screen establishes neither the required 5 rollout/s nor a precise
steady-state aligned rate.  Future endpoint evaluators must execute and record
an excluded causal warm-up before latency collection.  This latency limitation
does not affect the paired quality metrics or the failed causal-quality gate.

## Interpretation

Moving auxiliary generation before Wan fixes the NFE-1 causal-ordering defect
of synchronous dual diffusion, but does not solve the information problem.  A
small plan predicted only from history, actions, and noise must itself forecast
future motion accurately.  Here it did not: weak plan recovery, negligible
aligned-versus-shuffled effects, and absent action attribution all agree.

This rejects this two-call coarse latent-increment planner, dataset, one-seed
budget, and adapter seam.  It does not reject all dual-video mechanisms.  The
next useful tests should avoid making inference quality depend on a separately
forecast clean future feature: (1) an intra-forward feature produced by early
Wan blocks and consumed by later blocks in the same call, and (2) a frozen
spectral inverse-dynamics probe used only as training supervision and discarded
at inference.
