# VPM causal action motion-plan forcing screen (prospective)

Date frozen: 2026-08-08, before planner fitting, arm continuation, or candidate
validation metrics

Status: implementation and protocol only; no job or metric is created by this
commit

## Question and claim boundary

Can a compact motion state generated entirely from observed history, requested
actions, and Gaussian noise mature before video denoising and improve the
one-Wan-call video frontier?

This screen contains no protected test. It uses the immutable 512-train and
64-validation ABC populations. A positive result is one-seed validation
evidence only, not a general video-quality, real-time DAgger, or test-set
claim.

## Causal factorization

The model factorizes

\[
p(X,Q\mid H,A)=p_\phi(Q\mid H,A)p_\theta(X\mid H,A,\hat Q),
\]

where `H` is the five observed RGB frames, `A` is the requested action
sequence, `Q` is a compact low-frequency motion plan, and `X` is the eight
future RGB frames. At deployment `Q_hat` is generated before the first Wan
call. No clean future video or future-derived feature enters either sampler.

This differs from the completed PhaseLock probe, which spends full Wan calls on
a preliminary video; Frequency-Forcing, which generates a larger auxiliary
inside the shared Wan trajectory; observed-anchor V-JEPA, which uses a teacher
representation; and LaMo/two-clock, whose second signal is training-only.

## Frozen representation

The fixed geometry is:

```text
RGB clip          [B,13,3,180,960]
observed RGB      [B, 5,3,180,960]
Wan full latent   [B,16,4,24,120]
Wan history       [B,16,2,24,120]
future latent     [B,16,2,24,120] = 92,160 values
motion plan       [B,16,2, 6, 30] =  5,760 values
```

Split the width-stacked latent into three `24 x 40` camera views. Average each
view independently by `4 x 4` to `6 x 10`, then stack it back to `6 x 30`.
Call this operator `P`. Given independently encoded history `Z_H` and the full
training latent `Z`, define the raw increments

\[
R_0=P(Z_2-Z^H_1),\qquad R_1=P(Z_3-Z_2).
\]

Before planner fitting, compute immutable per-channel population moments
`mu_c,s_c` from `R` on the 448 planner-fit rows only and define
`Q_c=(R_c-mu_c)/s_c`. The 64 train-calibration rows, val64, and protected test
contribute zero elements. Statistics are accumulated in float64, persisted as
one identity-bearing JSON artifact, content-hashed, and loaded strictly by
planner calibration and both video arms. The update-400 planner snapshot must
contain byte-identical persistent mean/std buffers. Every generated plan and
Wan plan condition is therefore in this normalized coordinate system.

The statistics job independently encodes the five observed frames and also
encodes the full 13-frame training target. It fails if the first two full-clip
Wan tokens differ from the history-only tokens by more than `1e-4`; the
observed maximum, exact 448 indexes, train manifest/RGB hashes, Wan runtime,
mean/std, and element count are registered. Full-clip future tokens are used
only to construct this train-fit target. Clean `Q` is constructed only inside
the statistics job and planner calibration loss. The CAMP video continuation
never constructs it.

## Planner and train-only calibration

The planner is a roughly 2--3M parameter four-block 3-D convolutional RF model.
Its inputs are noisy `Q`, the pooled final observed Wan token, action tokens,
morphology, and its scalar clock. ABC actions have padded shape
`[B,13,5,157]`; transition chunks `4:12` are encoded and grouped four-at-a-time
into two future-Wan-token action states.

With the repository clock convention (`sigma=1` noise, `sigma=0` clean):

\[
Q_\sigma=(1-\sigma)Q+\sigma\epsilon_Q,\qquad
v_Q^*=\epsilon_Q-Q.
\]

Planner fitting minimizes

\[
L_P=\lVert g_\phi(Q_\sigma,\sigma,H,A)-v_Q^*\rVert^2
+0.25\,\operatorname{NMSE}(R_2(\epsilon_Q;H,A),Q),
\]

where the exact deployment rollout is two Euler calls on
`sigma=(1,0.5,0)`. The output head starts at exact zero.

Planner fitting uses a prospective partition of the 512 **training** rows:

```text
train-calibration: auxiliary_index % 8 == 0  (64 rows)
planner-fit:       auxiliary_index % 8 != 0 (448 rows)
```

The fixed endpoint is update 400. The 64 train-calibration rows report
two-step NMSE/cosine at updates 0, 100, 200, 300, and 399. These diagnostics do
not gate execution, choose a checkpoint, tune a threshold, or stop the fixed
workflow. Final val64 is not opened during planner calibration and never
decides whether video continuation proceeds.

The resulting update-400 planner snapshot is content-addressed, loaded strictly
into both arms, frozen, and required to have identical parameter and
normalization-buffer bytes. Its normalization artifact SHA-256 is also bound
independently in the resolved calibration and video configurations.

## Video conditioning and matched arms

The two-call generated plan is upsampled independently within each view. The
Wan conditioning tensor is

```text
C[:,:,0:2] = 0
C[:,:,2:4] = per_view_upsample(Q_hat)
C shape     = [B,16,4,24,120]
```

Observed appearance already enters the native reference channel, so the plan
cannot replace or shuffle history content. `C` enters the existing
`ZeroInitTFTokenAdapter` / `y_camera` residual seam. Its auxiliary clock is
fixed at clean zero and is not injected. The auxiliary velocity output is
computed identically but has zero loss and is ignored.

Both arms start from the exact update-1,000 historical VPM snapshot and receive
200 updates:

| Arm | Frozen planner | Plan adapter | Planner calls | Runtime fusion |
|---|---|---|---:|---|
| `PLAN-OFF` | identical | identical | 2 | exact off |
| `PLAN-ON` | identical | identical | 2 | aligned generated plan |

The arm model state, parameter shapes/count, planner output, initial adapter
bytes, optimizer schema, RGB/actions, clip order, video/plan noise, timesteps,
LoRA dropout path, and Wan calls are paired. The only treatment is the
non-parametric runtime fusion Boolean. Both arms generate and retain the plan
even when fusion is off, so planner compute and latency are matched.

During video training the condition is always a stopped autonomous two-call
planner sample. A clean, noised-clean, or oracle plan is forbidden from the
model signature. Future RGB remains an ordinary evaluator/video-flow target;
it is not a planner input.

## Deployment endpoints and call accounting

For every val64 clip and NFE `{1,2,4}`, run:

- `aligned`: local generated plan;
- `off`: same checkpoint and generated plan, runtime fusion disabled;
- `shuffled`: globally cyclically shuffled generated plan, with local history,
  actions, video noise, and planner noise retained.

At NFE 1 only, also run the `action_shuffled` attribution control: retain
the local history, morphology, Wan actions, video noise, and plan noise, but
globally cyclically shuffle the actions passed to the two-call planner. This
still executes exactly two planner calls and one Wan call; it adds no oracle.

`PLAN-OFF` executes every applicable label and is required to be bit-identical
across them at a fixed NFE because its fusion Boolean is false. `PLAN-ON` provides the causal
interventions. NFE 1 is the sole selectable endpoint; NFE 2/4 are descriptive
and cannot rescue a failed NFE-1 result. An optional clean-plan oracle may be
added only in a later preregistration; it is absent here.

At NFE 1 the declared deployment path is:

```text
one history VAE encode
two small planner calls
one Wan call
one VAE decode
zero teacher, RAFT, V-JEPA, or clean-future calls
```

Every row records planner calls, Wan calls, history encode, planner, Wan,
decode, and full end-to-end wall time with device synchronization. Each clip
also records hashes of cached RGB, cached/local actions, planner actions,
history latents, video noise, plan noise, generated/injected plan, final latent,
and decode. Registration rejects unequal data/action/noise hashes across arms
or endpoints where they must be paired. NFE never hides planner work. The
screen reports both generated-frame throughput (`8 / latency`) and
action-conditioned rollout/decision rate (`1 / latency`). A real-time DAgger
claim requires at least 5 rollout Hz using p95 end-to-end latency; mean frame
throughput cannot satisfy that gate.

## Metrics and fixed gate

Primary, lower is better:

- decoded temporal-difference MSE including the history/future boundary.

Guardrails, lower is better:

- future Wan-latent NMSE;
- decoded future RGB MSE.

Diagnostics are plan NMSE/cosine, future latent-delta NMSE, PSNR, all stage
latencies, and peak memory. NFE 2/4 are descriptive only and cannot rescue the
NFE-1 candidate. The NFE-1 action-shuffle comparison is required for an
action-conditioned mechanism or DAgger claim; without it, a planner that
ignores requested actions could pass by using history and noise alone.

At NFE 1 compare `PLAN-ON/aligned` with:

1. independently trained `PLAN-OFF/aligned`;
2. the same `PLAN-ON` checkpoint with fusion `off`;
3. the same `PLAN-ON` checkpoint with globally `shuffled` generated plan;
4. the same `PLAN-ON` checkpoint with globally shuffled planner actions while
   retaining local actions in Wan.

The family is four references by three claim metrics. Use 10,000 paired clip
bootstraps, seed 20260808, and one-sided simultaneous lower bounds at confidence
`1 - .05/12`. Every comparison must satisfy:

- temporal-MSE improvement at least 3% by point estimate and 1% by simultaneous
  lower bound;
- latent-NMSE and decoded-MSE point estimates no worse than 0%, with each
  simultaneous lower bound greater than -1%.

The first three references form the nine-cell generated-plan quality gate. The
fourth forms a three-cell planner-action attribution gate. The
action-conditioned mechanism passes only if all twelve comparisons pass and
every artifact proves:

- exactly two planner calls and one Wan call at the primary endpoint;
- zero clean-future/teacher inputs;
- identical planner tensor hashes between PLAN-OFF and PLAN-ON before fusion;
- identical cached RGB, local action, video-noise, and plan-noise hashes for
  every paired clip/endpoint and explicit planner-action donor hashes for the
  action-shuffled diagnostic;
- identical registered parameter schema and planner checkpoint;
- identical registered normalization artifact and persistent buffers;
- complete synchronized stage-latency and exact call-count records;
- complete val64 paired coverage;
- no protected-test access.

Aligned beating off but not shuffled is generic conditioning or regularization,
not sample-specific motion guidance. Aligned beating plan-shuffled but not
action-shuffled can establish sample-specific history/noise guidance, but not
planner action attribution or DAgger utility. The nine-cell quality result is
still reported separately if the three-cell action gate fails.

## Provenance and execution policy

Registration must occur from one clean full commit containing this protocol,
bind the parent/planner snapshots, RGB/action arrays, train/validation
manifests, Wan/VideoX runtime, Python, configs, and protocol bytes, and create a
fresh Lustre root. W&B must revalidate the private personal project
`zijiandu/dual-video-diffusion-private`, entity `zijiandu`, `group=null`, with
non-resuming content-bound run IDs.

Jobs are non-requeueable. Failed attempts remain immutable and are never
overwritten. The workflow refuses a changed source, input, planner, existing
output, missing completion receipt, changed call grid, or any protected-test
path.
