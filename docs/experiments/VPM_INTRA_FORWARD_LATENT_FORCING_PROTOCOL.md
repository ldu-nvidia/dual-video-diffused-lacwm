# One-call intra-forward video latent forcing screen (prospective)

Date frozen: 2026-08-08, before candidate training or validation metrics

Status: design and implementation only. No metric may be read and no job may
be launched until the implementation, tests, source commit, immutable inputs,
and registration described below are complete.

## Question and claim boundary

Can an auxiliary low-frequency/motion state predicted halfway through one Wan
backbone evaluation improve the final video prediction later in that **same**
evaluation?

The screen addresses a causal defect in ordinary synchronous dual diffusion.
At one sampling step, the video and auxiliary states enter the model as pure
noise and are updated together only after the forward call. The video head can
therefore never see the auxiliary estimate produced by that call. Moving the
auxiliary prediction and injection boundary inside the backbone gives the
video head a generated estimate without adding a second Wan call or using a
clean future feature.

This is a one-seed ABC development-validation screen. It contains no protected
test and cannot by itself support a general quality, acceleration, or real-time
DAgger claim.

## Representation and causal inputs

Use the already frozen six-channel per-view Haar scratchpad:

```text
RGB clip             [B,13,3,180,960]
observed RGB         [B, 5,3,180,960]
Wan state            [B,16,4, 24,120]
auxiliary state      [B, 6,4, 24,120]
```

Each camera view is spatially box-low-passed independently. Every four-frame
window retains its temporal Haar DC and coarsest signed motion coordinate.
The clean target is computed only for the supervised training loss and
post-generation validation scoring. No oracle sampling cell is registered. At deployable inference every
auxiliary bin starts from sample-keyed Gaussian noise; the sampler accepts only
the five observed frames, requested actions, morphology, and explicit video
and auxiliary noise.

## One-call architecture

Wan has 30 transformer blocks. Let `h_15` be the output after block index 14.
At auxiliary noise level `s`, a small patch head predicts rectified-flow
velocity

\[
\hat v_q = g_\phi(h_{15}, q_s, s), \qquad
\hat q_0 = q_s-s\hat v_q.
\]

The identity `q_0=q_s-sv_q` follows from the repository convention
`q_s=q_0+s(epsilon-q_0)`. A stopped copy of `q_0_hat` is patch-projected,
LayerNorm-normalized, multiplied by a bounded scalar residual gate, and added
to `h_15`. Blocks 15--29 and the native Wan video head then predict video
velocity from this generated estimate. The auxiliary velocity returned to the
sampler is exactly the midpoint prediction. Consequently one Euler call both
produces the auxiliary estimate internally and lets later blocks use it.

The midpoint injection must be implemented by a scoped block hook that is
installed and removed inside one forward call. Registration and tests bind:

- exactly 30 Wan blocks and midpoint index 14;
- one execution of the midpoint head per Wan call;
- no additional Wan/backbone call;
- `q_0_hat` is stopped before video injection;
- no clean target, future RGB, teacher feature, or oracle tensor reaches the
  midpoint hook or deployable sampler;
- the generated estimate, midpoint residual, and final output are finite.
- Wan activation checkpointing is disabled. A checkpoint recomputation after
  the scoped hook has been removed would differentiate a different function
  from the function used to compute the loss, so runtime drift fails closed.

At NFE 1 the architecture is:

```text
patch embed -> blocks 0..14 -> predict q0_hat -> inject -> blocks 15..29
            -> video head
```

This is not an early-exit latency optimization: all 30 blocks still execute.
It is a causal ordering intervention with small extra head/projection cost.

## Matched training arms

Both arms start from the same historical VPM update-1,000 checkpoint and
receive 200 updates over the immutable 512 ABC training clips. They instantiate
identical midpoint head, adapter, gate, optimizer schema, LoRA, action encoder,
parameter count, auxiliary target/loss, clocks, data order, noise, timesteps,
and Wan-call topology.

Before either optimizer runs, deterministic model-only loading hashes every
initialized state tensor byte plus the parameter, trainable-parameter, and
optimizer schemas. `MID-OFF` creates an immutable anchor and `MID-ON` must
match it exactly. The Slurm array is deliberately sequential so the treatment
cannot train before this byte-level check succeeds.

| Arm | midpoint auxiliary loss | generated midpoint injection |
|---|---:|---:|
| `MID-OFF` | 1 | exact off |
| `MID-ON` | 1 | aligned `stopgrad(q0_hat)` |

Both arms compute the midpoint estimate. The sole treatment is the
non-parametric runtime injection Boolean. The auxiliary and video clocks are
synchronous in this first screen, and the native input-level auxiliary state
and clock residuals are exact zero so generated structure first enters at the
registered midpoint.

Common factors are seed 1234, global batch eight on eight B200 GPUs, AdamW,
learning rate `1e-4`, 20-step warmup, cosine decay, validation after updates
0/50/100/150/199, and the private personal W&B project
`zijiandu/dual-video-diffusion-private` with `group=null`.

## Deployable intervention controls

Evaluate all 64 registered validation clips at NFE `{1,2,4}` with sample-keyed
identical video and auxiliary noise. Every cell executes two deterministic
generations: one artifact-audit generation and one synchronized latency
generation. Each generation independently hooks the Wan forward, block 14,
and the midpoint head and records exactly `NFE` calls; the receipt therefore
also records exactly `2*NFE` total evaluation calls per cell. This duplication
is evaluation instrumentation, not an extra call in one generated rollout.

For the `MID-ON` checkpoint run:

- `aligned`: inject this sample's midpoint `q0_hat`;
- `off`: compute the same estimate but multiply its residual by exact zero;
- `shuffled`: cyclically roll only `q0_hat` across the global batch while
  retaining local history, actions, video/auxiliary noise, clocks, and video
  state.

Also evaluate `MID-OFF/aligned`; it must be bit-identical to its own `off` and
`shuffled` labels at every NFE. No clean-auxiliary oracle is included in this
screen.

Rows record latent NMSE, decoded RGB MSE, decoded temporal-difference MSE
including the history/future boundary, auxiliary NMSE/DC/motion NMSE, peak
memory, independent midpoint-head/block/Wan hook counts, VAE history encode,
Wan, midpoint overhead, decode, and externally measured end-to-end wall time
with CUDA synchronization. The synchronized timed generation collects no
artifacts, and its decoded bytes must match the audit generation exactly.

## Frozen gate

NFE 1 is the sole selectable primary endpoint; NFE 2/4 are descriptive. Use
10,000 paired clip bootstraps with seed 20260808 and one-sided simultaneous
lower bounds at confidence `1-.05/9` for three references by three claim
metrics.

`MID-ON/aligned` must beat each of `MID-OFF/aligned`, same-checkpoint `off`, and
same-checkpoint `shuffled` as follows:

- temporal-MSE improvement at least 3% by point estimate and 1% by
  simultaneous lower bound;
- latent-NMSE and decoded-MSE point estimates no worse than 0%, with lower
  bounds greater than -1%;
- exact one midpoint-head execution and one Wan call per NFE-1 row;
- complete paired val64 coverage, zero protected-test access, and zero
  clean-future/teacher inputs.

Training validation is one reusable infinite iterator bounded to exactly four
local batches per event: 4 batches x 2 clips x 8 ranks = 64 clips, at updates
0, 50, 100, 150, and 199.

Aligned beating the independently trained arm but not same-checkpoint off and
shuffled is capacity or regularization, not evidence that the generated
midpoint state guides video. A negative result retires this midpoint/target/
budget combination, not every intra-forward hierarchy.

## Immutable execution policy

Registration must run from one clean commit and create a fresh Lustre root. It
binds source/config/protocol bytes, the historical VPM snapshot and SHA-256,
Wan and VideoX assets, Python executable, 512-train and 64-validation manifests
and cache metadata, W&B destination, midpoint index, block count, arm table,
call grid, and protected-test prohibition. Output roots are create-once;
failed attempts are retained and never resumed or overwritten.
