# VPM one-step direct-residual frontier

Date frozen: 2026-08-08

Status: prospective train-only protocol, committed before fitting or reading the
new reserve outcomes

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
| fresh reserve outcome | 480--511 | sole new outcome population |

Validation and protected test remain unopened. Fit, calibration, and outcome
noise seeds are disjoint:

- fit: `20260832, 20260833`;
- calibration: `20260834, 20260835`;
- outcome: `20260836, 20260837, 20260838, 20260839`;
- projection: `20260840`;
- bootstrap: `20260841`.

The parent is the same content-bound VPM update-1,000 snapshot and resolved
configuration used by the completed causal-compressibility ladder. The run
must revalidate its snapshot, arm/stage manifests, cache metadata, manifest,
RGB, and action identities. Teacher caches and V-JEPA arrays are prohibited
from the fit and serving input graph.

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
- exact sample/noise/history/action/target hashes and call counters.

`VPM_OFF` and `ZERO` must be bit-exact. Endpoint processes may open only the
pinned RGB/action arrays. The residual target, clean latent, and future RGB are
evaluator-owned: they may score the already-materialized endpoint but may not
enter its model signature or correction inputs. Teacher/cache call counts are
exactly zero.

## Frozen gate

Effects use 10,000 episode-clustered bootstrap resamples; the four noise draws
remain inside each episode. Positive means lower error. Advance the direct
endpoint only if all conditions hold:

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
