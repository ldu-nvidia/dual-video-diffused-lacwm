# VPM one-step direct-residual frontier

Date frozen: 2026-08-08

Status: prospective train-only protocol revision 2, committed before fitting
or reading any eligible reserve outcome

Revision 2 was forced by an independent pre-outcome source audit. The first
sealed implementation was never submitted and no reserve outcome was opened.
The audit found that the historical RGB/action dataset constructor had sampled
global rows 0 and 511 for structural finite/range validation. Row 511 therefore
cannot honestly be called untouched and is prospectively excluded below. The
same audit required role-scoped row access, stronger content/identity binding,
and true end-to-end serving latency before launch.

## Question

Can a small inference-causal residual head, trained on the ordinary clean-video
rectified-flow residual at VPM's initial one-step state, improve the actual
VPM@1 quality frontier?

This is a hard control, not dual diffusion. The completed privileged-residual
ladder showed that an equal-capacity `DIRECT` head dominated the clean-V-JEPA
teacher head on the weaker J1 parent, while unmodified VPM@1 still dominated
both. The missing comparison is therefore direct residual learning on VPM
itself. A passing endpoint becomes the baseline that every later causal
auxiliary must beat. It does not rescue or support privileged-feature transfer.

## Frozen data partitions

Use only the immutable 512-row ABC training cache and canonical manifest.
Episode identity, not row number alone, must be disjoint.

| Role | Manifest indices | Status |
|---|---:|---|
| excluded teacher history | 0--127 | never opened |
| residual-head fit | 128--383 | ordinary RGB/action target only |
| direct-only calibration | 384--415 | capacity and ridge selection only |
| prior inspected development | 416--479 | excluded from this experiment |
| fresh reserve outcome | 480--510 | sole new outcome population (31 episodes) |
| historical constructor probe | 511 | excluded; previously read for array validation |

Validation and protected test remain unopened. Fit, calibration, and outcome
noise seeds are disjoint:

- fit: `20260832, 20260833`;
- calibration: `20260834, 20260835`;
- outcome: `20260836, 20260837, 20260838, 20260839`;
- projection: `20260840`;
- bootstrap: `20260841`.

The parent is the same content-bound VPM update-1,000 snapshot and resolved
configuration used by the completed causal-compressibility ladder. The run
must bind the caller's snapshot digest to the parent receipt and rehash the
actual snapshot bytes, then revalidate arm/stage manifests, cache metadata,
manifest, RGB, and action identities. Teacher caches and V-JEPA arrays are
prohibited from the fit and serving input graph.

Each phase instantiates the RGB/action dataset with validation probes inside
that phase only: fit rows 128/383, calibration rows 384/415, and outcome rows
480/510. An index-auditing wrapper rejects every access outside the active
partition and proves the exact expected count for every allowed row. This is a
row-level contract; recording only which NumPy files were opened is
insufficient. The parent training config's future-validity retry/substitution
is hard-disabled (`enabled=false`, `max_retries=0`) and asserted after Hydra
instantiation, so an invalid allowed row cannot be silently replaced by a
random row from another partition.

## Model and arms

At the native VPM@1 `sigma_video=1` call, capture the shared Wan trunk tokens
from the feature-off forward pass. Patchify the exact ordinary target residual

\[
r^*(S)=v^* - v_{\mathrm{VPM}}(S),
\]

where `v* = noise - clean_video_latent` and `S` contains only inference-visible
history, actions, time, and initial noise. Fit token-local affine ridge heads
after one frozen Gaussian projection of the trunk feature. Reuse the previous
capacity ladder `{64,256,1024}` and ridge grid
`{1e-4,1e-3,1e-2,1e-1,1}`. Select solely by calibration corrected-velocity
MSE; ties prefer smaller capacity, then larger regularization. No outcome row
may influence selection.

Fit three equal-schema arms:

| Arm | Target | Purpose |
|---|---|---|
| `ZERO` | exact zero | bit-exact no-op control |
| `DIRECT_ALIGNED` | local `v* - vVPM` | primary causal correction |
| `DIRECT_SHUFFLED` | different-episode residual, fixed cyclic donor | sample-specific target control |

The shuffled donor is episode-disjoint and fixed before target values are
opened. Every arm uses identical projected inputs, token subsampling, numerical
precision, parameter count, and regularization. The frozen Wan parent and
residual heads receive no optimizer update during endpoint evaluation.

## Fresh one-step endpoint

For every reserve episode and each of four outcome noise seeds, materialize
`VPM_OFF`, `ZERO`, `DIRECT_ALIGNED`, and `DIRECT_SHUFFLED` from the same initial
noise and one shared Wan call. The adapter modifies the returned velocity
before the one native scheduler update; it does not add a Wan call. Decode the
eight future frames and report:

- corrected future-velocity MSE;
- video-latent NMSE;
- decoded RGB MSE in `[0,1]`;
- temporal-difference MSE including observed-to-first-future boundary;
- adapter-only mean/p50/p95 latency and full one-step latency;
- exact registered clip/episode identity and hashes for noise, all five raw
  history frames, encoded history reference, actions, morphology, encoded
  action control, auxiliary noise, latent flow target, exact pre-quantization
  cache-space future/history-boundary RGB, their uint8 scoring conversions,
  and final latent, plus call counters.

`VPM_OFF` and `ZERO` must be bit-exact. Endpoint processes may open only the
pinned RGB/action arrays. The residual target, clean latent, and future RGB are
evaluator-owned: they may score the already-materialized endpoint but may not
enter its model signature or correction inputs. Teacher/cache call counts are
exactly zero.

Full aligned latency is synchronized wall time from resident observed RGB and
actions through history/control preparation, the one shared Wan call, residual
adapter, Euler update, and RGB decode. Clean-future encoding and evaluator
scoring occur only after all endpoints are materialized and are excluded.
Component timings are retained, but cannot substitute for this full number.
All 64 batch timing rows—including `(noise_seed, clip_indices, batch_size)` and
the four singleton `[510]` batches—are persisted with identities and a file
digest. Analysis and audit recompute count, mean, p50, and p95 from this trace;
the adapter gate requires a finite `0 <= p95 < 1 ms`.

## Frozen gate

Effects use 10,000 episode-clustered bootstrap resamples over the 31 eligible
episodes; the four noise draws remain inside each episode. Positive means lower
error. Advance the direct endpoint only if all conditions hold:

1. `DIRECT_ALIGNED` improves corrected-velocity MSE by at least 3% over both
   `VPM_OFF` and `DIRECT_SHUFFLED`, with paired lower bounds above 1%;
2. decoded and temporal MSE each improve at least 3% over both controls, with
   paired lower bounds above 1%;
3. latent NMSE has a nonnegative point effect versus VPM_OFF and lower bound
   above -1%;
4. at least 60% of reserve episodes favor aligned on decoded and temporal MSE;
5. `ZERO` is bit-exact to VPM_OFF for every final latent and metric row;
6. the endpoint uses exactly one Wan call, zero teacher/feature calls, no
   target/cache serving input, and adapter p95 below 1 ms on B200; and
7. registration, fit, endpoints, analysis, completion, and independent audit
   receipts all pass content-hash and inventory recomputation.

If aligned improves VPM but not shuffled, the head learned only a population
correction and is not a sample-specific residual mechanism. If it fails VPM,
ordinary one-step residual capacity is not a stronger frontier at this seam.

## Consequence

A pass authorizes a separately preregistered nonlinear/continuation
replication and makes `VPM+DIRECT@1` the development baseline. It does not
authorize a dual-feature claim. A causal auxiliary must subsequently beat
both this endpoint and capacity-matched direct supervision at equal Wan calls
and complete latency. A fail retains unmodified VPM@1 as the frontier.
