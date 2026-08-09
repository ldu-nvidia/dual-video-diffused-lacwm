# VPM adjacent consistency-distillation comparator (ACD-P0)

Date frozen: 2026-08-09

Status: prospective, implemented pilot. No research job has been launched from
this worktree. Registration and Slurm entrypoints are fail-closed behind an
exact-source test receipt, a one-B200 memory receipt, and a fresh explicit
user-authorization receipt. W&B is disabled.

Source base:
`92fd9ed8480f12084015c043f1fd5dfcfe40205a`

## Decision question and claim boundary

Can direct, feature-free consistency distillation improve one-, two-, or
four-call causal VPM video generation enough to serve as the mandatory
low-NFE comparator for physics-flow and dual-state proposals?

This is deliberately **not** a dual-diffusion method. It adds no
time-frequency, V-JEPA, rendered-flow, or other feature state. Both arms retain
the ordinary VPM parameter schema and deployment inputs. A positive result
would strengthen the direct low-NFE baseline that any dual method must beat;
it would not support a dual-feature novelty claim.

The bounded port is grounded in exact released implementations:

- Causal Forcing commit
  `1fc7bbc19a503c1bce80ecef08158b20e702f386`,
  `model/naive_consistency.py`, advances a forward-corrupted state by one
  frozen-teacher Euler step between adjacent rectified-flow nodes, then matches
  an online consistency prediction to a stopped EMA-student prediction at the
  next node.
- Flash-WAM commit
  `5b8df13e9db24fb15ce42ff5ccc60a4015195960`,
  `distillation/step.py`, uses the same teacher-step/online/EMA structure with
  a Karras boundary parameterization and pseudo-Huber loss. Its released video
  schedule uses SNR shift 5, `sigma_data=0.5`, pseudo-Huber `c=0.001`, and EMA
  decay `0.995` in `distillation/config.py`.
- The rCM release at
  `ed3cb14dd936f92cdc9f9381af7369991509b41f` recommends a teacher-forced
  consistency initialization before more complex self-forced distribution
  matching.

This P0 ports only the video-only, discrete teacher-step initialization. It is
not Flash-WAM's joint video/action method, Causal-rCM's complete training
recipe, Causal Forcing++ in full, or a self-forcing/DMD phase. Their reported
speed and quality remain external evidence, not LACWM results.

## Why this is not the completed two-clock screen

The rejected two-clock experiment drew one high and one low state independently
on the same clean-data/noise line. The current online model predicted both,
the low prediction was a stopped target, and inference remained ordinary
Euler. ACD-P0 changes all four defining elements:

1. the low-clock state is reached by an **online frozen-teacher Euler step**
   from the actual high-clock state;
2. the target is a separate **EMA student**, not the current low-clock branch;
3. the loss matches a **Karras-boundary consistency function**, not raw clean
   predictions; and
4. deployment evaluates both objective-compatible readouts and the complete
   objective-by-readout diagnostic cross, rather than silently handicapping
   the RF control with an untrained boundary map.

The old result therefore does not answer this question. Conversely, an ACD-P0
failure rejects only this fixed teacher-forced port and budget.

## Rectified-flow convention and forward pair

LACWM uses `sigma=1` for Gaussian noise and `sigma=0` for clean data:

```text
x_sigma = (1-sigma) x0 + sigma epsilon
v*      = epsilon - x0
x0_hat  = x_sigma - sigma v_theta(x_sigma, sigma, c).
```

The fixed Wan training scheduler has 1,000 descending nodes, shift 5, and
native timesteps `t=1000*sigma`. Each update samples one native start index
`i` using the existing VPM logit-normal clock sampler, then sets

```text
j = min(i + 500, 999).
```

Thus the released Flash-WAM two-anchor stride is preserved exactly. A zero
step (`j=i`) is forbidden. If the existing sampler selects an index above 998,
the implementation deterministically resamples from the same rank-local RNG
stream until `j>i`; the number of rejected draws is traced. Both arms execute
the identical draw path.

For source state `x_i`, one frozen causal teacher call gives

```text
x_j^T = x_i + (sigma_j-sigma_i) v_T(x_i,sigma_i,c).
```

After the step, the observed latent prefix is reset to its known forward-noise
trajectory using the observed history reference and the same `epsilon`.
Generated future tokens are never clamped.

The teacher consumes no separate clean future, target feature, cache, or metric.
However, `x_i` was constructed by forward-corrupting the clean training video.
This is teacher-forced initialization, not evidence that the training state is
causally constructible at deployment.

## Boundary parameterization and loss

For `sigma_data=0.5`, define

```text
c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
c_out(sigma)  = sigma*sigma_data / sqrt(sigma^2 + sigma_data^2)

F_theta(x,sigma)
  = c_skip(sigma) x
    + c_out(sigma) [x - sigma v_theta(x,sigma,c)].
```

The implementation must prove exactly, not approximately, that at `sigma=0`,
`c_skip=1`, `c_out=0`, and `F_theta(x,0)=x` for arbitrary finite velocity.

The online student predicts

```text
p_i = F_theta(x_i,sigma_i).
```

The frozen EMA target student predicts

```text
q_j = stopgrad[F_bar_theta(x_j^T,sigma_j)].
```

Only the two future Wan tokens receive loss. The candidate uses the released
pseudo-Huber form with fixed `c=0.001`:

```text
L_ACD = mean_M [sqrt((p_i-q_j)^2+c^2)-c].
```

No ordinary-flow mixture, coefficient sweep, adaptive weighting, action loss,
feature loss, or extra trainable parameter is allowed in P0. The EMA target is
initialized bit-exactly from the parent and updated only after a successful
student optimizer step:

```text
bar_theta <- 0.995 bar_theta + 0.005 theta.
```

The update covers only source parameters with `requires_grad=true` in a stable,
name-matched order. Frozen backbone/VAE parameters and all buffers are copied
bit-exactly from the online model; applying EMA arithmetic to already-equal
frozen floating tensors is forbidden because rounding alone can create drift.
The teacher is never updated. Tests require the EMA update to occur after—not
before—the target forward and successful optimizer step.

## Frozen teacher and matched arms

The teacher, online student, and EMA target all begin at the current faithful
VPM frontier:

```text
snapshot SHA-256
de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a

run identity SHA-256
d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f

canonical model-state SHA-256
d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0
```

The older `f67c7bae...` snapshot is forbidden. Registration must bind the
parent resolved config and strict-load every model key. External teacher/EMA
copies are created only after this load, excluded from the optimizer's
parameter groups, and never reachable from the public deployment sampler.
Student checkpoints contain separately labelled online and EMA states; the
teacher state is represented only by its immutable parent identity, not
serialized again.

Registration also composes each arm through the registered Python/Hydra
runtime with every interpolation resolved. The complete semantic mapping—not
only selected fields—is hashed into the arm identity. This binds inherited
AMP/dtype, optimizer epsilon/weight decay, scheduler final LR, LoRA
rank/alpha/dropout, data-loader semantics, and all other job fields. The live
job recomputes this digest before trainer construction; the emitted
`.hydra/config.yaml`, trace header, snapshot, and independent completion
receipt must all agree.

Both arms execute one teacher, one online-student, and one EMA-target Wan call
per training update. The sole objective difference is:

| Arm | Returned student loss | Teacher/online/EMA calls | Role |
|---|---|---:|---|
| `ACD-RF-CONT` | ordinary masked RF MSE at `x_i` | 1 / 1 / 1 | compute/input/EMA matched control |
| `ACD-CONS` | fixed pseudo-Huber `L_ACD` | 1 / 1 / 1 | adjacent consistency candidate |

The control computes the full teacher step, both consistency outputs, and the
consistency diagnostic, but does not attach that graph to its returned
objective. The candidate computes ordinary RF MSE diagnostically. Both arms
maintain EMA with decay 0.995. The setup supports a same-update/same-Wan-call
comparison; measured training time and memory remain mandatory.

LACWM has no text CFG branch: both teacher and students receive the same fixed
null prompt plus explicit planned actions. Adding an invented unconditional
action/text call would change the parent policy and is forbidden. This is an
explicit departure from Flash-WAM's two-call CFG teacher.

The historical 64-channel auxiliary topology remains a parameter-matched
no-op: state/clock conditioning is off, auxiliary loss is zero, and no target
extractor/cache is reachable. Production VPM sampler and Wan files are
imported but not modified.

## Training population and fixed budget

If separately authorized after exact-source review, each arm trains exactly
400 optimizer updates, seed 1234, on the immutable 512-clip ABC training
population, eight B200 ranks, local batch one, global batch eight, AdamW
(`lr=5e-6`, betas `0.9/0.95`), 40-update warmup, cosine decay to `1e-6`, AMP,
gradient clipping at 1.0, and no gradient accumulation. There is no trainer
validation, visualization, early stopping, loss/EMA/stride sweep, resume, or
checkpoint selection. W&B is disabled.

The actual 512-clip training RGB and action NumPy files are byte-hashed and
schema-checked at registration. Rank zero repeats both complete byte hashes
before each arm and broadcasts a fail-closed receipt before any dataset worker
starts. Metadata-carried digest strings alone are not accepted. Validation
array bytes remain unopened until the global endpoint-materialization barrier.

Every update seals rank-combined hashes for clip ID, actions, clean latent,
Gaussian noise, rejected and accepted clock draws, start/end indices, sigmas
and timesteps, source state, teacher-stepped state, online/EMA consistency
outputs, RF and consistency targets, exact three-call count, deterministic
teacher/EMA state probes before/after update, and RNG state before each Wan
call. Complete named-tensor state hashes plus exact parameter/buffer counts are
sealed at initialization and the final endpoint; the immutable teacher full
hash must match at both boundaries. Paired-arm analysis compares only the
treatment-invariant clip/action/clean/noise/clock/source/teacher-step/call/RNG
fields for all 400 updates. EMA/teacher probes must additionally be bit-exact
across all eight ranks at every update. Complete initial and final
student/teacher/EMA state receipts are gathered from all ranks and compared.
Online/EMA outputs and post-update EMA states are
treatment outcomes and must not be required to match after the arms diverge.
The trainer appends exactly one hash-chained `phase=optimizer_update` record
directly after each successful optimizer/EMA transaction. Validation or logger
callbacks cannot create audit rows. An AMP-skipped optimizer step fails before
EMA and invalidates the run.

## Objective-compatible deployment and 2x2 readout audit

The evaluator constructs only the online or EMA **student**. Its public sampler
accepts exactly five observed RGB frames, planned actions, morphology, fixed
null context, and sample-keyed Gaussian streams. It has no teacher, target,
clean-future, feature, or cache argument.

For a fixed native NFE grid `1=sigma_0>...>sigma_K=0`, start with
sample-keyed `x_0~N(0,I)`. The consistency readout computes

```text
z_hat_k = F_theta(x_k,sigma_k).
```

If this is the final call, `z_hat_k` is the decoded latent. Otherwise draw a
separate sample-keyed Gaussian `eta_(k+1)` and re-noise:

```text
x_(k+1) = (1-sigma_(k+1)) z_hat_k + sigma_(k+1) eta_(k+1).
```

The observed history prefix is overwritten by the same formula using the
history reference. This is the standard consistency-style re-noising analogue
for the repository's linear RF path. The rCM release commit
`ed3cb14dd936f92cdc9f9381af7369991509b41f` provides evidence for this
re-noising structure, but its ordinary RF clean estimate is not substituted
for the Flash-WAM Karras map optimized here. No alternative noise-reuse
diagnostic is registered in P0.

The native RF readout instead advances the same initial state once per call:

```text
x_(k+1) = x_k + (sigma_(k+1)-sigma_k) v_theta(x_k,sigma_k,c).
```

Its observed history is reset after every step to the known RF corruption path
using the *initial* sample-keyed noise. At NFE 1 this returns exactly
`x-sigma*v` because the appended terminal is `sigma=0`. This is the compatible
primary deployment for the RF-MSE control and freshly replayed parent.

The frozen endpoint family is the full 2x2 objective-by-readout cross for both
online and EMA students:

| Training objective | Primary compatible readout | Nonselectable cross-readout diagnostic |
|---|---|---|
| `ACD-RF-CONT` RF MSE | native RF Euler | Karras `F` plus re-noising |
| `ACD-CONS` pseudo-Huber | Karras `F` plus re-noising | native RF Euler |

The fresh parent uses native RF Euler only. Cross-readout diagnostics can
explain whether an outcome comes from the learned objective or the sampler,
but cannot promote `GO_ACD` or select an NFE. All readouts consume the same
registered Gaussian stream; native RF deliberately uses only its first slice.

Three additional nonselectable action interventions use the primary EMA
readout for candidate/control and native RF for the parent. Planned action
sequences are permuted by the smallest cyclic validation-index offset for
which every donor belongs to a different episode. RGB history, noise, model
state, and NFE remain aligned. These endpoints measure whether a quality gain
collapsed action sensitivity; they are not extra candidate variants.

At NFE `K`, an independent hook must observe exactly `K` student Wan calls.
Teacher calls, feature calls, target-cache reads, and pre-barrier future-RGB
reads must all equal zero. The final latent—not an intermediate velocity—is
passed to the unchanged causal Wan decoder.

## Evaluation and fixed gates

All 64 development clips and four preregistered noise seeds are evaluated at
NFE `{1,2,4}` for the full online/EMA 2x2 family above plus a freshly replayed
frozen VPM parent under native RF. The protected split remains closed. Metrics are future
latent NMSE, decoded RGB MSE, decoded temporal-difference MSE, LPIPS, PSNR,
complete latency (history VAE, action path, Wan calls, re-noising, decode), and
peak allocated/reserved memory. Timing uses counterbalanced endpoint order,
warmup, CUDA synchronization, and at least 120 repeats per endpoint.

Positive relative improvement means lower error. Clip/noise pairs are the
resampling unit. The analyzer uses 10,000 paired bootstraps with seed 20260809
and one-sided Bonferroni lower bounds across the frozen NFE/metric family.

An NFE passes only if the candidate EMA **consistency-readout** endpoint
satisfies all of:

1. decoded and temporal MSE improve at least 3% over the objective-compatible
   `ACD-RF-CONT` EMA native-RF endpoint with simultaneous lower bounds at
   least 1%;
2. latent NMSE and LPIPS have nonnegative point improvements over the control
   and simultaneous lower bounds above -1%;
3. decoded and temporal MSE beat the freshly replayed parent native-RF VPM@1 frontier
   with strictly positive simultaneous lower bounds, while latent and LPIPS
   meet the same -1% noninferiority bounds;
4. candidate online-versus-EMA conclusions agree in sign for all primary
   metrics. For decoded and temporal MSE, define action sensitivity as
   `100*(M_shuffled-M_aligned)/M_shuffled`; its candidate-minus-parent@1
   difference must have point estimate and simultaneous one-sided lower bound
   no worse than -1 percentage point;
5. exact deployment call/access identities pass, and complete p95 latency is
   no more than 5% above the parameter-identical control at equal NFE; and
6. no selection or metric uses the protected split.

`GO_ACD` requires at least one fixed NFE to pass, and reports the lowest passing
NFE. Any failure is `NO_GO_ACD`. A one-seed development pass authorizes a
multi-seed confirmation, not a paper claim.

### Speed and equivalence

No acceleration claim is allowed from call count alone. Relative to a
higher-call quality-matched endpoint, the selected student must have:

- paired decoded/temporal/LPIPS noninferiority bounds no worse than -1%;
- at least 20% lower complete p95 latency;
- exact student calls and zero deployment teacher calls; and
- measured control-cycle latency, not extrapolated frames, meeting 5 Hz
  (`<=200 ms`) or 10 Hz (`<=100 ms`) where claimed.

Eight predicted frames per chunk must not be reported as eight independent
closed-loop control updates.

## Resource ceiling and current readiness

No job is authorized by this document. The maximum proposed allocation is:

| Phase | Allocations | B200/allocation | Time limit | Maximum B200-hours |
|---|---:|---:|---:|---:|
| full-geometry three-copy memory smoke | 1 | 1 | 1 h | 1 |
| matched 400-update training | 2 | 8 | 6 h | 96 |
| paired NFE 1/2/4 evaluation/timing | 1 | 8 | 2 h | 16 |
| **total** | 4 |  |  | **113** |

Three model copies per rank make peak-memory preflight mandatory before any
full run. The output ceiling is 100 GiB under `/mnt/data1`, `/mnt/data2`, or
the exact user Lustre root
`/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train`;
other Lustre prefixes fail closed. Jobs are non-requeueable. The present readiness seal
must report registration, jobs, W&B writes, data rows, outcome rows, and
protected-test accesses all equal zero.

The existing few-step audit shows that the current parent's NFE 2 is mixed and
NFE 4 is materially worse than VPM@1. This does not invalidate local
consistency training, but it means the frozen teacher is not already a
quality-dominant multi-step oracle. This is a scientific risk and a mandatory
interpretation caveat, not a hidden execution override. Before memory smoke,
readiness is `READY_SOURCE_MEMORY_PREFLIGHT_REQUIRED`; only exact-source tests
plus a passing memory receipt can produce `READY_FOR_REGISTRATION`. Every
submission still requires a phase-specific, fresh user-authorization receipt.

## Stop rules

- A target-leaking or noncausal teacher invalidates the run.
- A consistency-loss reduction is not an autonomous video gain.
- An improvement over the matched control that remains worse than VPM@1 is not
  progress over the deployed frontier.
- A direct-distillation gain is a stronger baseline, not dual-diffusion
  evidence.
- A call-count reduction without quality equivalence and complete latency is
  not acceleration.
- P0 does not test self-forced DMD, full Causal-rCM, full Flash-WAM, decoder
  distillation, long-horizon rollout, multi-seed generalization, or closed-loop
  DAgger utility.
