# VPM invertible P/Q two-clock latent pilot (IPQ-TC1)

Date frozen: 2026-08-09

Status: **prospective protocol only**. This document is committed before model
implementation, registration, training, endpoint materialization, or outcome
access. The user has authorized additive implementation and evidence-driven
research execution. Registration and launch nevertheless remain contingent on
the parent ILSF-2 frozen handoff rule saying to advance and an independent
auditor acknowledging one exact clean implementation commit.

Terminal handoff note, 2026-08-09: ILSF-2 concluded
`NO_GO_GENERIC_EARLY_SUBSPACE` with audit status `PASS`. The required advance
condition was therefore not met. IPQ-TC1 remains an additive, tested,
unregistered and unlaunched contingency; no endpoint outcome was opened.

Protocol source base: clean integration commit
`ee4855b3314e0864dece9e832158928570dc93fc`.

Parent-lineage amendment, 2026-08-09: protocol-only commit `c32c54b`
incorrectly named the legacy `f67…` checkpoint. An independent read-only audit
caught the error before implementation or any outcome access. This prospective
amendment replaces it with the faithful `de65…/d79c…/d123…` frontier and adds
the historical-source parity gate below. No arm, schedule, metric, threshold,
dataset, or outcome was changed.

## Question and claim boundary

Does end-to-end training on two exact complementary subspaces of the native
future Wan VAE latent make low-call video generation materially better than an
equal-compute, parameter-identical single-clock control?

The candidate does not receive information that the control lacks. Both states
are deterministic orthogonal projections of the same native latent, both begin
from projected Gaussian noise at inference, and their sum is exactly the native
video state. The intervention is only **factorized corruption and timing**:
the candidate trains on independently corrupted coarse/detail states, exposes
both clocks to one shared Wan adapter, predicts both velocities in one Wan
call, and may denoise the coarse state earlier at inference.

This is a one-seed, short-continuation ABC pilot. Its endpoint population has
been used by prior program-level studies, so it is not a globally fresh
lockbox. It is nevertheless untouched within this run: it is not opened for
training, validation, checkpoint selection, schedule selection, debugging, or
early stopping. A pass is exploratory evidence for a prospective multi-seed
study on genuinely untouched episodes. It is not an FVD, general-video,
real-time DAgger, policy-success, or physical-realism claim.

## Completed-experiment equivalence audit

The following audit is frozen before implementation. A future implementation
is invalid if it collapses to any completed mechanism in this table.

| Completed study | State and clock algebra | Why IPQ-TC1 is not equivalent |
|---|---|---|
| clean/oracle TF and V-JEPA conditioning | a feature extracted from the realized clean future conditions video | IPQ-TC1 has no future encoder or clean inference feature; both states are generated from noise |
| low-resolution Video Latent Forcing | a separate, lossy coarse-RGB or semantic target is generated before/fused into video | IPQ-TC1 splits the native VAE latent losslessly and has no RGB/semantic auxiliary decoder |
| generated Frequency-Forcing | a lossy six-channel RGB Haar DC/motion scratchpad is jointly denoised | IPQ-TC1 uses full-rank complementary native-latent P/Q states; no RGB target or discarded band exists |
| VPM two-clock consistency | two *global* points on one shared RF trajectory require two Wan calls per update; the low-noise prediction is a stopped teacher; inference remains ordinary single-clock VPM | IPQ-TC1 presents simultaneous P/Q states with independent clocks in one call and integrates those clocks at inference; it has no stopped consistency loss |
| fixed-Haar and learned-QMF residual probes | a frozen trunk plus external linear heads predicts projections of the one-step residual | IPQ-TC1 trains the shared Wan LoRA, adapter, and both velocity heads end to end; P/Q are diffusion states, not post-hoc corrections |
| spectral/low-frequency training losses | transformed clean targets regularize an otherwise ordinary video flow and disappear at inference | IPQ-TC1 directly corrupts, predicts, and integrates both invertible states at train and inference |
| ILSF-2 invertible multirate probe | an unmodified frozen VPM receives a generated LL-locked midpoint while still seeing one scalar clock; no band-clock training occurs | IPQ-TC1 is the contingent trained follow-up: mixed clocks are in-distribution, both clocks are visible, and P/Q velocities are separately supervised |

The diagonal-clock control below is intentionally algebraically equivalent to
ordinary full-latent rectified flow. That equivalence is a control, not the
claimed innovation. The candidate is nonduplicate only if independent clocks
remain separately visible and P/Q are integrated with separate step sizes.

## Exact view-isolated orthogonal state

The pinned 13-frame, three-view ABC clip encodes to native Wan geometry
`z0 in R^[B,16,4,24,120]`. Five observed RGB frames occupy the first two
latent frames; the final two latent frames are the prediction target. Width is
three adjacent views of 40 latent columns. Every transform is applied in FP32
to each future `[B,16,2,24,40]` view independently. No kernel crosses a view
seam and history is never assigned to the auxiliary subspace.

For every channel, future time, view, and aligned spatial `2 x 2` block `x`,
define the orthogonal Haar-LL projector

\[
Sx = \bar x\,\mathbf 1_{2\times2},\qquad
\bar x=\frac14\sum_{a,b\in\{0,1\}}x_{ab},
\]

and its complement `Q = I-S`. The clean future states are

\[
p_0=Sz_0,\qquad q_0=Qz_0,\qquad z_0=p_0+q_0.
\]

`p0` retains one of four spatial degrees of freedom in every block and keeps
both future latent frames, so it contains spatial layout and temporal change.
`q0` retains the three complementary spatial-detail degrees of freedom. Both
are stored in the native full tensor geometry; neither is rescaled or decoded
through a separate codec.

For one canonical isotropic Gaussian `epsilon`, use

\[
\epsilon_p=S\epsilon,\qquad \epsilon_q=Q\epsilon.
\]

Orthogonality makes these projected components independent Gaussian variables
on their subspaces and `epsilon_p + epsilon_q = epsilon`. The independently
corrupted future states are

\[
p_{s_p}=(1-s_p)p_0+s_p\epsilon_p,
\qquad
q_{s_q}=(1-s_q)q_0+s_q\epsilon_q,
\]

with `sigma=1` denoting noise and `sigma=0` clean. The native Wan input is
`x = p_sp + q_sq` on future slots. Observed-history slots follow the ordinary
reference/noise path at `s_p`. The explicit Q adapter input is zero on history
and equals `q_sq` on future slots. Thus serving inputs remain exactly observed
history, actions, morphology, null text/CLIP context, and one sample-keyed
Gaussian video noise.

Every checked batch must satisfy:

- maximum P+Q reconstruction error `<=2e-6` in FP32;
- relative reconstruction-error energy `<=1e-12`;
- normalized P/Q inner-product magnitude `<=2e-6`;
- projector idempotence error `<=2e-6`;
- rank fraction exactly `0.25` for P and `0.75` for Q;
- exact-zero Q history support; and
- exact view isolation under single-view impulses.

## One shared Wan model and two visible clocks

Both arms instantiate the same 16-channel dual adapter schema. The native Wan
input receives `p_sp+q_sq` and the native timestep corresponding to `s_p`.
The existing additive token adapter receives `q_sq`; its separate sigma
embedding receives `s_q`. The state and clock gates are fixed open at `0.02`
in both arms so a collapsed zero gate cannot silently turn the candidate into
the control. Adapter projections, the Wan LoRA, action/control modules, native
video head, and Q velocity head remain trainable. The Q head also receives its
ungated raw Q-clock embedding.

One and only one Wan trunk call emits a native-head tensor and a Q-head tensor.
Hard projections define the two predictions:

\[
\hat v_p=S\,v_{\rm native},\qquad
\hat v_q=Q\,v_{\rm qhead}.
\]

Projection occurs before loss and before every Euler update, so neither head
can hide error or capacity in the other subspace. The targets are

\[
v_p^*=\epsilon_p-p_0,\qquad v_q^*=\epsilon_q-q_0.
\]

With future validity mask `M` expanded to all 16 native channels, the loss is

\[
\mathcal L_{P/Q}=
\frac{\sum M\left[(\hat v_p-v_p^*)^2+(\hat v_q-v_q^*)^2\right]}
{\sum M}.
\]

Each band is normalized by the *same full-coordinate denominator*. There is
no 50/50 band averaging and no inverse-energy weighting. Consequently, by
orthogonality,

\[
\mathcal L_{P/Q}=
\operatorname{MSE}_M(\hat v_p+\hat v_q,\epsilon-z_0).
\]

This identity must hold within `2e-6` relative error in FP32 on every audit
batch. It prevents the 1/4-rank coarse band from receiving an accidental 2x
or 4x objective advantage.

## Frozen matched training arms

| Arm | Training clocks | Architecture, calls, objective | Estimand |
|---|---|---|---|
| `IPQ-SYNC` | draw `s_p` and an audited spare `s_q`; set effective `s_q=s_p` | same dual adapter/heads, one Wan call, same P/Q loss | parameter- and compute-matched single-clock control |
| `IPQ-INDEP` | independently draw `s_p,s_q` from the full native discrete scheduler | same dual adapter/heads, one Wan call, same P/Q loss | independent mixed-clock training |

Both arms draw the canonical video noise, P clock, and spare Q clock in the
same order. `IPQ-SYNC` draws and records the spare Q clock but does not use it.
No arm-specific draw may touch Wan dropout RNG. Per-update hashes must prove
identical clip IDs, actions, clean latent, canonical epsilon, P clock, spare Q
clock, RNG state before Wan, optimizer LR, and observation count. The only
permitted tensor differences are the effective Q clock, the resulting Q
corruption, model outputs, gradients, and later learned parameters.

Both arms start core weights from the immutable **faithful-cascade** update-1,000
VPM frontier:

```text
snapshot SHA-256
de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a

run identity SHA-256
d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f

canonical 1,686-tensor model-state SHA-256
d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0

resolved update-1,000 config SHA-256
ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38

training source commit
656086686dae723c942a4209a9d71cdb17ed6ccc
```

The older `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`
snapshot is explicitly forbidden. A completed parent-lineage audit found that
its canonical state is `2b298196…`, not `d1231b8b…`: 495/1,686 tensors differ,
including 480 Wan LoRA tensors and 15 action encoder/pool/control/morphology
tensors. It is a distinct older model, not a container variant of the current
frontier. Registration must bind the full
`PHYSICS_FLOW_PARENT_LINEAGE.md` evidence, including the corrected comparison
log SHA-256 `048bcddd35ecd2458e5b17f28a48a8a4967111dbb888e8cd2bc9d5ff669c0e2a`.

The historical unused 64-channel auxiliary adapter/head keys are excluded and
replaced by the same deterministic 16-channel P/Q modules in both arms. Every
other key must load strictly from `de65…`. Registration records the parent
resolved-config bytes, core-load coverage, exact excluded keys, initial state
hashes, total/trainable parameter counts, and per-parameter shapes; the two
arms must match bit-for-bit before update zero. It also binds parent training
source `6560866`, the native sampler source SHA-256/git blob
`a10fe3730f7bb3bacd20bd14ebbcfab2b3cf8c63a2db27783f1e8a9787b87ee6` /
`abc6d4df165f684c2c93920d079585c600562dbb`, and the complete audited delta
between that source and the current model path. This is a matched continuation,
not training-efficiency evidence from scratch.

Before any endpoint manifest row is opened, the untouched `de65…` parent must
also be instantiated from its exact resolved configuration and all 1,686
tensors strict-loaded in a separate clean source worktree at commit `6560866`.
On a fixed target-blind train history at NFE 1/2/4, its public deployable sampler
must match the current repository's parent adapter bit-for-bit in initial noise,
final latent, and decoded uint8 video. This historical/current parity receipt
guards against attributing a shared continuation regression to the candidate.

Each arm trains exactly 400 optimizer updates, seed `1234`, on the immutable
512-clip ABC training manifest using eight B200 ranks, local batch one, global
batch eight, AdamW (`lr=1e-4`, betas `0.9/0.95`), 40-update warmup, cosine
decay to `1e-6`, AMP, gradient clipping at 1.0, and no EMA. Checkpoints at 200
exist only for recovery; only the fixed update-400 checkpoint is eligible for
endpoint evaluation. There is no trainer validation, visualization, early
stopping, checkpoint selection, or outcome-dependent retry.

## Frozen inference schedules

At evaluation, start with `p_1=S epsilon` and `q_1=Q epsilon`; their sum is
the exact paired ordinary Gaussian video noise. Let the native NFE schedule
provide decreasing nodes `s_0=1>...>s_K=0` and native timesteps. Every endpoint
uses exactly `K` shared Wan calls.

The primary candidate schedule makes coarse P mature before detail Q:

\[
s^p_i=s_i,\qquad s^q_i=\sqrt{s_i},
\]

with endpoints assigned exactly to one and zero. The reverse-order control is
`s^p_i=s_i, s^q_i=s_i^2`; the synchronous control is `s^p_i=s^q_i=s_i`.
After one shared prediction,

\[
p_{i+1}=p_i+(s^p_{i+1}-s^p_i)\hat v_p,
\qquad
q_{i+1}=q_i+(s^q_{i+1}-s^q_i)\hat v_q,
\]

and `z_{i+1}=p_{i+1}+q_{i+1}`. History is restored to the native reference
path at `s^p_{i+1}` and is clean at the final decode.

For NFE 1 every monotone two-clock schedule necessarily has only `[1,0]` in
both bands. P-leading, Q-leading, and synchronous schedules must therefore be
bit-identical within a checkpoint. Any NFE-1 advantage can support only a
mixed-clock **training** claim, not an inference-order claim.

Evaluate NFE `{1,2,4}` with paired clip-keyed noise under these endpoints:

| Checkpoint | Endpoint | Intervention |
|---|---|---|
| `IPQ-SYNC` | `SYNC` | synchronous P/Q schedule; primary baseline |
| `IPQ-INDEP` | `P_LEADS_ALIGNED` | primary trained package |
| `IPQ-INDEP` | `SYNCHRONOUS` | isolates mixed-clock training from inference order |
| `IPQ-INDEP` | `Q_LEADS_ALIGNED` | predeclared reverse-order control |
| `IPQ-INDEP` | `P_LEADS_STATE_OFF` | disable Q state and Q clock injection into the shared video trunk; retain states, heads, and integration |
| `IPQ-INDEP` | `P_LEADS_STATE_SHUFFLED` | roll only Q conditioning tokens across an adjacent episode-disjoint batch; retain each sample's actual P/Q states, noises, clocks, heads, and integration |
| `IPQ-INDEP` | `P_LEADS_CLOCK_TIED` | keep P-leading states/integration but inject `s_p` in place of the true `s_q`; diagnostic clock-use ablation |

The shuffled condition changes only the adapter input, never the Q state being
integrated. Batch size is exactly two and the donor clips must have distinct
episodes. These same-checkpoint interventions cannot promote a result on their
own; they test whether explicit state/clock content is causally used.

## Access, causality, and call gates

The immutable 512-row train manifest is the sole training population. The
immutable 64-row ABC validation manifest is the within-run untouched endpoint
population. Registration may hash metadata, identities, and paths but no
validation RGB/action row may be constructed until both update-400 training
completion receipts exist. The two checkpoints are evaluated completely
before comparative analysis. Protected test paths are absent from every
interface.

At inference the sampler accepts only observed RGB history, requested actions,
morphology, IDs, and sample-keyed Gaussian noise. It cannot accept a full RGB
clip, clean future latent, auxiliary target, target cache, V-JEPA/TF encoder,
teacher output, oracle feature, or alternative decoder. Endpoint tensors,
decoded predictions, and their hashes are closed before the evaluator may
encode clean future RGB for scoring. The endpoint process must not instantiate
the production ABC dataset (whose item constructor serves all 13 RGB frames),
call a whole-array content hash, or expose a generic RGB-slice interface before
this boundary. A typed memmap reader may serve only `RGB[row,0:5]` and planned
`actions[row,0:13]`, recording every file, row, slice, and returned-tensor
hash. All eight ranks must finish and hash every endpoint for all assigned
clips, close their pre-barrier memmaps, and cross one global distributed
barrier. Only then may a distinct scoring reader open full RGB or a process
rehash the validation arrays. Event sequence counters and the access ledger
must prove this ordering for every autonomous endpoint and row.

The run fails closed unless:

- every training update has exactly one shared Wan call;
- every endpoint has exactly its declared NFE and an independent Wan hook
  agrees;
- teacher, V-JEPA, TF-encoder, auxiliary-target-array, and pre-barrier
  clean-future counters are all zero;
- arm data/RNG/LR traces pair exactly for all 400 updates;
- parameter names, shapes, counts, initialization hashes, optimizer policy,
  and model-call counts match;
- diagonal corruption, diagonal loss, diagonal Euler, NFE-1 schedule, P/Q
  reconstruction, orthogonality, rank, history, and view-isolation receipts
  pass their fixed tolerances; and
- no production sampling method is modified. IPQ-TC1 sampling remains an
  isolated evaluator/model method.

## Metrics, statistics, and frozen decisions

For all 64 endpoint clips and four fixed evaluation-noise seeds preserve:

- future video-latent NMSE;
- decoded future RGB MSE in `[0,1]`;
- decoded temporal-difference MSE including the observed/future boundary;
- P-band and Q-band future latent NMSE;
- future temporal-delta latent NMSE;
- per-band target/prediction/error energy;
- P/Q trajectory tensors for a fixed eight-clip evidence subset;
- actual calls, synchronized end-to-end latency, peak memory, tensor hashes,
  event order, and all forbidden-access counters.

The paired unit is episode. Average the four noise draws within episode, then
use 10,000 deterministic episode-clustered bootstrap replicates with seed
`20261201`. Positive relative change means lower error.

The primary simultaneous family is `3 NFE x 3 metrics = 9` comparisons of
`IPQ-INDEP/P_LEADS_ALIGNED` against `IPQ-SYNC/SYNC`. One-sided Bonferroni lower
bounds use family alpha `0.05/9`.

`GO_IPQ_TWO_CLOCK` requires one NFE to satisfy all of:

1. decoded MSE and temporal MSE point improvements are each at least **3%**
   and their simultaneous lower bounds are above zero;
2. latent NMSE has a nonnegative point improvement and simultaneous lower
   bound above `-1%`;
3. at least 60% of episodes favor the candidate on decoded and temporal MSE;
4. at the same NFE, aligned beats both state-off and state-shuffled by at least
   1% on decoded and temporal MSE with positive paired 95% lower bounds;
5. for NFE 2 or 4, aligned beats reverse-order by at least 1% on decoded and
   temporal MSE with positive paired 95% lower bounds and clock-tied changes at
   least one primary metric by 1%; NFE 1 is exempt because order collapses;
6. every access, call, pairing, capacity, algebra, latency, and identity audit
   passes.

If the package beats the matched arm but not state/off/shuffled attribution,
the decision is `NO_GO_OPTIMIZATION_ONLY`. If independent-clock training helps
under the synchronous evaluator but P-leading does not beat reverse/sync, the
decision is `NO_GO_TRAINING_ONLY_NO_ORDER`. If P-leading improves P or Q latent
error without both decoded thresholds, the decision is
`NO_GO_LATENT_ONLY`. Every other quality failure is
`NO_GO_IPQ_TWO_CLOCK`.

A secondary acceleration diagnostic compares candidate NFE 1 with control NFE
2 and candidate NFE 2 with control NFE 4. It may be called
`FEWER_CALLS_NONINFERIOR` only when decoded, temporal, and latent point effects
are each no worse than `-1%`, paired lower bounds exceed `-2%`, and measured
end-to-end p95 is lower. It cannot override a failed `GO_IPQ_TWO_CLOCK` gate.

No hyperparameter, schedule exponent, seed, checkpoint, endpoint, split, NFE,
metric, threshold, or decision label may change after registration.

## Resource, logging, and authorization envelope

The later authorized plan is two training allocations (one node/eight B200s,
four-hour non-requeueable limit each), two endpoint allocations (one
node/eight B200s, two-hour limit each), and CPU-only analysis. Maximum reserved
budget is 96 B200-hours; actual elapsed GPU-hours, peak memory, and artifact
bytes are mandatory outputs. Large checkpoints, logs, rows, and trajectories
must live under the registered Lustre dual-video artifact root, never the repo
or root filesystem. Each job must preflight quota and fail before training if
the registered output root is not fresh or has less than 1 TiB free.

W&B is disabled in the prepared source. An independently audited contingent
launch may enable online telemetry only after verifying that
`zijiandu/dual-video-diffusion-private` reports private access, bind each run ID
to its arm identity, set `group=null`, `resume=never`, and unset inherited group,
job-type, and notes variables. Existing credentials alone cannot bypass the
ILSF handoff and exact-SHA audit gate.

## Implementation and launch seal

Implementation must be additive and isolated under new IPQ-TC1-named model,
trainer, evaluator, analyzer, workflow, config, test, and Slurm files. Existing
production sampler files may be imported but not edited. Before handoff, a
read-only readiness command must seal:

- protocol SHA-256 and protocol commit;
- exact clean implementation commit and remote reachability;
- diff scope relative to `ee4855b`;
- nonduplication ledger and algebraic-control version;
- full unit/contract test command and result hash;
- model/config inventory and source hashes;
- resource plan and output-root policy;
- `registration_created=false`, `jobs_submitted=0`, `wandb_writes=0`, and
  `outcomes_opened=0`.

This protocol authorizes additive implementation and local synthetic/unit
validation only. It does not itself authorize registration, W&B writes,
`sbatch`, model training, validation-row reads, or endpoint evaluation. After
ILSF-2 completes, an independent auditor must verify that its frozen handoff
rule advances this contingency and explicitly acknowledge the exact sealed
source SHA. That acknowledgment—not a further requirement that the user recite
an internal hash—unlocks the already user-authorized execution chain.
