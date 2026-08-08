# Deployable video Frequency-Forcing screen (prospective)

Scientific design frozen: 2026-08-07, before candidate training or validation metrics

Operational iterator amendment frozen: 2026-08-08, before any retry is
registered or launched

Status: implementation/protocol only; no job has been launched by this change

The pre-amendment `frequency-forcing-seed1234-20260808-56e2e24-v2`
allocation is preserved as immutable failed-attempt evidence. Its FPM arm
requested eight batches from a finite rank-local validation stream containing
only four batches and stopped at the first scheduled validation. No checkpoint,
metric, or partial state from that root is eligible for a retry. This amendment
only makes that operational iterator reusable and bounds each event to one
complete pass; it does not change an arm, target, optimizer update, evaluation
endpoint, or decision threshold. A retry requires a fresh source commit,
registration, study root, and run identity.

Clock convention: `sigma=1` is Gaussian noise and `sigma=0` is clean data.

## Question

Can a small, deterministic low-frequency video scratchpad improve the quality
of a few-call action-conditioned Wan/LACWM rollout when the scratchpad itself
must be generated from noise at inference?

This test does not use a pretrained semantic teacher. The auxiliary clean
target exists only during supervised training and explicitly labelled oracle
diagnostics. Autonomous inference accepts exactly five observed RGB frames and
the proposed actions. It receives no clean future RGB, no clean future
low-frequency target, and no online encoder output.

## Representation

The frozen ABC input is `[B,13,3,180,960]`, with three `320`-pixel camera views
stacked along width. Each view is independently padded from `180 x 320` to
`192 x 320` using the production value `-1`, then box-averaged by `8 x 8` to
`24 x 40`. This box average is repeated Haar-LL averaging expressed in
mean-valued coordinates. Views are never mixed.

Let `P_t[c,h,w]` denote one pooled RGB view. Bin zero repeats `P_0` four times.
Bins one through three use frames `1:5`, `5:9`, and `9:13`. For each four-frame
window `(P_0,P_1,P_2,P_3)`, retain only

```text
DC     = (P_0 + P_1 + P_2 + P_3) / 2
motion = (P_2 + P_3 - P_0 - P_1) / 2.
```

These are the orthonormal length-four Haar scaling coordinate and the coarsest
signed temporal detail. For the repeated anchor, `DC=2 P_0` and `motion=0`.
The output channel order is

```text
[R_DC, G_DC, B_DC, R_motion, G_motion, B_motion].
```

After the views are width-stacked again, the auxiliary state is
`[B,6,4,24,120]`; the Wan VAE state is `[B,16,4,24,120]`. The auxiliary is
lossy: it discards two fine temporal Haar details and every spatial high-pass
band. This is intentional. The earlier complete length-four RFFT was an
invertible repackaging of low-resolution video; the current question is
whether an easier structural state matures early enough to guide video.

The transform has no learned parameters, checkpoint, PCA, feature cache, or
teacher call. All four auxiliary bins start from independent Gaussian noise in
autonomous inference (`auxiliary_history_mode=diffuse_all`). Thus even the
known-history bins are not clamped in this minimal screen; this keeps the
sampler identical across arms and tests a strict jointly generated state.

## Common initialization and data

All arms start model-only from the completed historical V-JEPA study's VPM
checkpoint. Its trained video LoRA, action encoder/control, VAE interface, and
other non-auxiliary weights are retained. The old 64-channel auxiliary adapter,
clock, and head are excluded because their shapes and semantics differ. All
stochastic weights in the new six-channel modules are initialized identically
under seed 1234; only the preregistered scalar gate values differ between the
no-fusion and fusion arms.

The old VPM metric is not reused as this experiment's baseline. A fresh FPM
arm receives the same 200 updates as every candidate. Consequently the screen
estimates continued adaptation from a common checkpoint, not training speed
from scratch.

The immutable ABC cache supplies 512 fixed training clips and 64 fixed
validation clips. Registration requires unique clip IDs, exactly one clip per
episode, and zero train/validation overlap by clip ID, episode path, or
episode/start pair. Its previously cached V-JEPA target array is never
memory-mapped and no target is returned to the model. No test manifest or test
cache is accepted by the registration or evaluation interfaces.

Common training factors:

- seed 1234;
- one node, eight B200 GPUs;
- batch one per rank, global batch eight;
- 200 optimizer updates;
- AdamW, learning rate `1e-4`, 20-update warmup, cosine decay;
- identical RGB/action order, Wan checkpoint, LoRA schema, optimizer, and
  evaluation noise;
- private W&B destination `zijiandu/dual-video-diffusion-private`, no group.

Trainer-side validation occurs after iterations `0,50,100,150,199`. One
iterator is constructed and reused across those events, so its deterministic
seed-1234, unaugmented rank shard is configured as infinite. Each event
consumes exactly four local batches of two:
`4 batches x 2 clips x 8 ranks = 64 clips`, one complete pass
over the registered validation split. `drop_last=false` is safe because each
rank owns exactly eight clips. Registration records this contract, the
warm-start preflight verifies the resolved Hydra config, and post-training
evaluation rejects a checkpoint whose saved config differs. Validation-time
future-validity retries are disabled, so no registered clip can be silently
replaced by another sample.

W&B finalization is bounded at 120 seconds and non-raising after the final
checkpoint and completion receipt are durable. Any unsent run files remain in
the arm-local W&B directory for later `wandb sync`; timeout does not discard
telemetry or turn a completed scientific run into a failed Slurm allocation.

## Arms

All four arms instantiate the same six-channel dual module schema.

| Arm | Auxiliary loss | Video sees auxiliary | Clock | Estimand |
|---|---:|---:|---|---|
| FPM | 0 | no | aligned/inert | parameter-matched video-only baseline |
| FAUX | 1 | no | aligned; head only | multitask regularization without fusion |
| FSYNC | 1 | yes | `sigma_aux=sigma_video` | synchronous joint generation/fusion |
| FLEAD | 1 | yes | `sigmoid(logit(sigma_video)-1)` | effect of making the same scratchpad mature earlier |

FSYNC and FLEAD initialize both bounded residual gates at `0.02` to avoid an
unexposed feature path. FPM and FAUX keep video-trunk state/clock injection at
exact zero. In all arms the auxiliary velocity head receives its raw auxiliary
clock; this does not enter the video trunk when fusion is disabled.

The comparisons isolate:

1. `FAUX - FPM`: auxiliary multitask regularization;
2. `FSYNC - FAUX`: the synchronous fusion package;
3. `FLEAD - FSYNC`: the leading-clock intervention;
4. autonomous versus same-checkpoint fusion-off and shuffled-generated-state:
   whether generated, sample-aligned content causally helps at inference.

## Validation-only inference

Every arm evaluates all 64 validation clips at total Wan-call budgets
`NFE={1,2,4,8}`. Each source/NFE run starts from sample-ID-keyed identical
video and auxiliary noise. The iterable dataset captures rank/world before
iteration and performs its native distributed sharding; evaluation audits one
and only one row for every global clip/source/NFE cell. The reported NFE must
equal both the sampler's counter and an independent Wan forward hook.
Because the shared validation dataset now cycles for trainer safety, the
standalone evaluator explicitly consumes exactly four local batches and then
stops; iterating it to exhaustion is forbidden.

Sources:

- `autonomous`: generated sample-aligned state;
- `off`: same checkpoint, initial noises, and auxiliary update machinery, but
  state and clock are not injected into the video trunk. Because the auxiliary
  head reads shared Wan tokens, its later trajectory may diverge after this
  intervention; the contrast is a joint-system fusion ablation, not frozen
  auxiliary-trajectory replay;
- `autonomous_shuffled`: roll only the generated, noise-subtracted state across
  the global batch while retaining each sample's corruption noise;
- `oracle_matched`: scheduled clean low-frequency target, leakage diagnostic;
- `oracle_shuffled`: wrong-sample scheduled clean target, oracle control.

The leading arm gets no extra call. Its auxiliary and video clocks share every
Wan forward, so FPM, FAUX, FSYNC, and FLEAD each make exactly `NFE` Wan calls
per cell. Encoder, decoder, and target-extraction costs are reported outside
the Wan-call count. Oracle cells can diagnose usable headroom but can never
support a deployment or acceleration claim.

At NFE 1 the exact noise/data endpoints make the aligned and leading inference
clock paths identical. That cell is a schedule negative control (although the
two checkpoints can still differ because their training clocks differed).
Within every checkpoint, autonomous and residual-shuffled NFE-1 video and
auxiliary outputs must be bit-identical; analysis fails if this invariant does
not hold.

Per-clip endpoints are future video-latent NMSE, raw decoded RGB MSE, temporal
MSE including the history-to-first-future boundary, future auxiliary NMSE, and
separate DC/motion auxiliary NMSE. Comparisons use paired 10,000-replicate
bootstrap intervals with seed 20260807. Before reading a table, analysis must
validate its sibling rank-zero completion receipt, exact registered study and
arm identities, frozen global grid, and SHA-256 of `rows.jsonl`. It refuses
duplicate or missing arm receipts.

## Decision rule

A deployable candidate is promising at NFE 1, 2, or 4 only if:

1. versus FPM at the same NFE, point improvements in video NMSE, decoded MSE,
   and temporal MSE are each at least 3%;
2. no primary metric's paired 95% interval permits more than 1% regression;
3. autonomous video NMSE is significantly better than both same-checkpoint
   fusion-off and shuffled-generated-state controls; and
4. every deployable artifact records zero teacher calls, no clean auxiliary,
   no future RGB passed to the sampler, and exact equal Wan calls.

FLEAD supports a scheduling claim only if it also beats FSYNC at equal total
calls. An oracle-only benefit means the representation contains useful target
information but its autonomous generator is still inadequate. A package-level
gain with no autonomous-versus-off/shuffled mechanism effect is reported as
multitask regularization, not dual-denoising inference guidance.

This screen has no protected-test phase. Passing results motivate a fresh,
separately preregistered multi-seed DROID confirmation; failing results retire
this exact six-channel representation/schedule, not every possible
Frequency-Forcing design.
