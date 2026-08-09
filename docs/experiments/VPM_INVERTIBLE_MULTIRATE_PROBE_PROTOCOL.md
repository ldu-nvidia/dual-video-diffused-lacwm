# Frozen VPM invertible multirate probe (ILSF-2)

Date frozen: 2026-08-09

Status: prospective, post-selection exploratory protocol. This document is
committed before registration, endpoint materialization, or outcome access.
The implementation is additive and may not change the production sampler.

Source base: clean `integration/dual-video-next-12h` commit
`997a9dae79d63627a65773ef19cc41462db85c7d`.

## Question and claim boundary

Can a frozen VPM use an **inference-generated, invertible low-frequency state**
as an earlier clock to improve a two-call future-video sample? The experiment
tests ordering and factorization, not extra information: the only serving
inputs are observed RGB history, planned actions, morphology, and
sample-keyed Gaussian noise.

No clean future video, TF target, V-JEPA feature, teacher, fitted adapter, or
new parameter is available to any endpoint. Rows 416--479 have already been
inspected by earlier causal-compressibility work. Every result is therefore
post-selection exploratory and cannot establish FVD, generalization,
real-time DAgger, or a paper claim. Validation and protected test remain
unopened.

An equivalence audit before freezing found no completed experiment that applies
this algebra. Generated Haar forcing denoised a separate lossy auxiliary;
PhaseLock used a separate preliminary video trajectory and motion delta;
fixed/learned wavelet probes predicted residual corrections. None split the
native video latent into exact complementary states and put only one state at
the clean clock between the two ordinary VPM calls.

## Exact future-only, view-isolated transform

The pinned VPM latent has shape `[B,16,4,24,120]`. The first two latent frames
are observed history and the final two are future. Width is the concatenation
of three camera views, each 40 latent columns wide. All transforms use FP32 and
operate separately on the three `[B,16,2,24,40]` future-view tensors. No
operation may cross a view seam or modify history.

For each `2 x 2` spatial block (x), define the orthogonal Haar-LL projector

\[
P_{LL}x = \bar x\begin{bmatrix}1&1\\1&1\end{bmatrix},
\qquad
\bar x=\tfrac14\sum_{a,b}x_{ab},
\]

and its complement (Q_{LL}=I-P_{LL}). This keeps both future latent frames,
so the early state contains spatially coarse appearance **and** temporal
change. Its algebraic rank is one of four coefficients per spatial block.

The equal-rank high-frequency control is the Haar-HH projector

\[
P_{HH}x = \overline{x\odot h}\,h,
\qquad
h=\begin{bmatrix}1&-1\\-1&1\end{bmatrix},
\]

with (Q_{HH}=I-P_{HH}). `LL` and `HH` are rank matched but intentionally not
energy matched; natural energy fractions are reported and no arm is rescaled.

For both projectors the run must verify, on every endpoint batch:

- future reconstruction maximum absolute error at most `2e-6`;
- relative reconstruction-error energy at most `1e-12`;
- normalized `P`/`Q` inner-product magnitude at most `2e-6`;
- idempotence error at most `2e-6`;
- exact-zero history support and exact view isolation;
- algebraic rank fraction exactly `0.25` for both `LL` and `HH`.

## Frozen two-call sampler

LACWM uses sigma one for Gaussian noise and sigma zero for clean video. Let the
native two-step `FlowMatchEulerDiscreteScheduler` return

\[
s_0=1>s_1>s_2=0
\]

and its two model timesteps. Let (c) denote observed history, actions,
morphology, and fixed null context. Starting from the deterministic video
noise (z_0), one **shared** first Wan call produces

\[
v_0=f_\theta(z_0,s_0;c),\qquad
\hat z_0^{(0)}=z_0-s_0v_0,
\]

\[
z_1^{std}=z_0+(s_1-s_0)v_0.
\]

History follows the native noisy-reference path at (s_1). The primary
lowpass-first midpoint is

\[
z_1^{LL}=P_{LL}\hat z_0^{(0)}+Q_{LL}z_1^{std}
\]

on future slots, with ordinary history restored. Thus LL jumps from clock one
to zero while detail follows the native clock from one to (s_1). A second Wan
call sees the generated clean LL state in the ordinary video-token stream:

\[
v_1=f_\theta(z_1^{LL},s_1;c),
\]

\[
z_2^{LL}=P_{LL}\hat z_0^{(0)}+
Q_{LL}\left[z_1^{LL}+(s_2-s_1)v_1\right].
\]

The output history is clamped clean. No separate feature branch is added.
This midpoint is outside the checkpoint's scalar-sigma training distribution:
LL is clean while its complement is at (s_1), although Wan receives the
single native timestep for (s_1). A negative result rejects only this frozen,
untrained two-call mixed-clock schedule; it does not reject a model trained on
band-specific clocks.

## Endpoints and exact controls

All endpoints share clip, history, actions, morphology, video noise, dormant
auxiliary noise, native timesteps, and the first Wan call.

| Endpoint | Calls | Frozen construction |
|---|---:|---|
| `VPM1` | 1 | first clean estimate `z0 - s0*v0`; actual one-call frontier |
| `VPM2_ORDINARY` | 2 | native two-step FlowMatchEuler baseline |
| `LL_FIRST_ALIGNED` | 2 | sample-aligned generated LL is locked early |
| `LL_FIRST_EPISODE_SHUFFLED` | 2 | roll only generated future LL across the adjacent episode-disjoint batch; retain local history/actions/noise/complement |
| `LL_FIRST_TIME_REVERSED` | 2 | swap only the two generated future LL latent frames within sample |
| `HH_FIRST_RANK_MATCHED` | 2 | early-lock generated HH under the identical rank-one-of-four schedule |

The batch size is exactly two. The two clip IDs and episode directories must
differ before the shuffled endpoint is legal. `torch.roll(..., 1, dim=0)` is
frozen as the donor map. The time-reversed arm swaps the two future latent
tokens and nothing else.

`SPLIT_SYNCHRONOUS` is an algebraic receipt rather than another model call:
`P(z2_ordinary)+Q(z2_ordinary)` must equal `z2_ordinary` on future slots within
`2e-6`. It proves that a transform without the clock intervention is an exact
no-op.

The model is strict-loaded from the immutable VPM snapshot and kept in eval
mode. All evaluator logic is external to the module, parameter count cannot
change, and the snapshot bytes are rehashed. `condition_on_tf=false` and
`condition_on_tf_clock=false` on every Wan call. The checkpoint's auxiliary
noise input is dormant and identical across arms; auxiliary target arrays,
teacher calls, V-JEPA calls, optimizer updates, and W&B writes are forbidden.

## Causal materialization and access order

The sole outcome population is the immutable ABC-train index interval
`416--479`, processed as adjacent two-episode batches. Noise is keyed by
`(clip_index, seed, stream)` with four new seeds:

- endpoint seeds: `20261101, 20261102, 20261103, 20261104`;
- episode-bootstrap seed: `20261105`.

Frozen exclusions are:

| Indices/split | Contract |
|---|---|
| train 0--415 | excluded from this run; previously used or inspected |
| train 416--479 | only exploratory endpoint population |
| train 480--510 | barred; consumed by earlier direct-residual work |
| train 511 | constructor-excluded and barred |
| validation | unopened |
| protected test | unopened |

Future-validity retry and row substitution are disabled. An index-audited
dataset must observe each permitted row exactly once per endpoint seed and
must reject every other index.

For each paired batch and seed, the evaluator follows this irreversible order:

1. form an inference-only structure containing observed RGB history, actions,
   morphology, IDs, reference, contexts, and sample-keyed noises;
2. materialize and hash all six endpoint latents, with exactly one shared
   first call and five second calls;
3. close the endpoint barrier;
4. only then encode the full clean RGB clip and construct latent/raw/decoded
   scoring targets;
5. decode and score the already materialized endpoints.

Endpoint builders cannot accept the full RGB clip or a target tensor. The
event ledger records sequence numbers for history encoding, every Wan call,
all endpoint materializations, the first target construction, and scoring.
The run is invalid unless target construction occurs strictly after every
endpoint. Tensor hashes prove the shared first call and all paired serving
inputs are identical where required. Teacher, feature, auxiliary-target, and
pre-barrier clean-future counters must be zero.

## Metrics and simultaneous family

For every `(endpoint, clip, seed)` preserve:

- future video-latent NMSE;
- raw decoded RGB MSE in `[0,1]`;
- decoded temporal-difference MSE including the observed-history boundary;
- LL- and complementary-band future latent NMSE;
- future LL temporal-delta MSE and cosine;
- LL/HH and complement energy fractions;
- complement error at `VPM1` and after the second call;
- P-lock maximum error;
- hashes, event order, access counts, actual Wan calls, forbidden-call counts,
  transform receipts, and synchronized latency.

The paired unit is episode. The four noise values are averaged inside episode
before resampling. Use `10,000` deterministic episode-clustered bootstrap
replicates. Positive percentage means lower error for the first endpoint.

The frozen simultaneous family is five contrasts by three primary metrics,
exactly 15 tests:

1. aligned versus `VPM2_ORDINARY`;
2. aligned versus `VPM1`;
3. aligned versus episode-shuffled;
4. aligned versus time-reversed;
5. aligned versus rank-matched HH-first;

crossed with latent NMSE, decoded MSE, and decoded temporal MSE. Every claimed
positive lower bound uses a one-sided Bonferroni family level
`alpha=0.05/15` (`0.0033333333333333335`). Unadjusted 95% intervals may be
reported descriptively but cannot satisfy a gate.

## Frozen gates and decision

Audit gates require all transform tolerances above, identical first-call and
paired-input hashes, exact endpoint/call inventory, one call for `VPM1`, two
calls for every other endpoint, zero new parameters, zero target/teacher/
feature/auxiliary reads before the endpoint barrier, and the synchronous
split identity. Projection latency p95 must be at most `2 ms`; aligned
end-to-end p95 must be no more than `5%` above ordinary VPM2.

`GO_INVERTIBLE_MULTIRATE` requires **all** of the following:

1. aligned beats both ordinary VPM2 and VPM1 by at least `3%` point gain on
   decoded and temporal MSE, with simultaneous lower bounds above zero;
2. aligned has nonnegative latent-NMSE point gain against both baselines and
   simultaneous lower bounds above `-1%`;
3. at least `60%` of episodes favor aligned on decoded and temporal MSE against
   both baselines;
4. aligned beats shuffled and time-reversed by at least `1%` on decoded and
   temporal MSE, with simultaneous lower bounds above zero;
5. aligned beats rank-matched HH-first by at least `1%` on decoded and temporal
   MSE, with simultaneous lower bounds above zero;
6. the aligned complement-band error improves by at least `3%` from `VPM1`
   while its early LL state remains locked within `2e-6`;
7. every causal, reconstruction, call, capacity, access, and latency audit
   passes.

If aligned beats VPM2 but not VPM1, the decision is
`NO_GO_SOLVER_STABILIZATION_ONLY`. If aligned is not specifically better than
shuffled/reversed, the decision is `NO_GO_GENERIC_MIXED_STATE`. If LL is not
specifically better than HH, the decision is `NO_GO_GENERIC_EARLY_SUBSPACE`.
Every other failure is `NO_GO_UNTRAINED_MIXED_CLOCK`.

Even a pass authorizes only a new, prospectively trained band-clock study on a
genuinely fresh dataset. It does not itself authorize deployment or a speed,
FVD, closed-loop, or general-video claim.

## Frozen execution envelope

The evaluator runs 64 clips times four seeds in batch-two pairs. Per pair and
seed it uses one shared first Wan call plus five second calls: `128 + 640 =
768` Wan batch calls, followed by six decoded endpoints per pair. The launcher
requests one B200, 32 CPUs, 384 GB RAM, the `batch` partition, `short` QoS,
account `coreai_chef_posttrain`, and a two-hour non-requeueable allocation.
Outputs live under the registered Lustre dual-video artifact root. No large
artifact may be written to the repository or root filesystem.
