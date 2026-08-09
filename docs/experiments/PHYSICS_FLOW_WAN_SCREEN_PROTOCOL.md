# Prospective raw-physics-flow Wan Stage-1 gate

Date frozen: 2026-08-08

Status: **prospective; no video-model training or generated-video outcome may
open until the exact source commit, caches, audits, parent, runtime, and this
protocol are sealed in `registration.json`**

“Prospective” is strictly branch-scoped: the Stage-1 endpoints, thresholds,
analysis, and D405 subset are frozen before this branch's training or generated
video outcomes. The immutable val64 rows may have been used elsewhere in the
broader research program; this protocol makes no claim that val64 is globally
new, untouched, or a protected final test set.

## Question and claim boundary

At equal Wan transformer calls, does a deterministic robot-motion field built
only from the observed frame-4 robot state, candidate action chunks, fixed
D405 calibration, and official ABC robot geometry improve a matched
action-conditioned Wan continuation model at NFE 1, 2, or 4?

This is a fixed-conditioning feasibility screen. The field is not diffused,
predicted, or updated; it has no noise clock, velocity target, or model call.
A pass would establish only that causal raw geometry is a useful scaffold for
a separately preregistered stochastic object/contact residual. It would not
establish dual diffusion, FVD, real-time DAgger, or a paper-level video-quality
claim.

The fresh direct-residual continuation screen did not improve its frozen
quality family. It is therefore supporting motivation, not an imported arm:
the faithful-cascade update-1,000 VPM remains the sole parent/control, and
RAW-FLOW must show that action-aligned causal geometry contributes structure
beyond ordinary residual continuation capacity. No direct-residual weights or
outcomes enter this study.

The pre-outcome parent audit in `PHYSICS_FLOW_PARENT_LINEAGE.md` rejected the
older `f67c7bae…` snapshot: although its 1,686-tensor schema matches, 495 model
tensors differ from the faithful-cascade frontier. Both arms therefore use
only `de65e832…/d79c3699…`; the registration binds both canonical model-state
hashes, the full mismatch receipt, its failed predecessor, and the ladder and
direct-residual lineages. It also binds the exact parent resolved configuration
with SHA-256
`ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38`.
The public sampler source is additionally proven byte-identical to parent
training commit `656086686dae723c942a4209a9d71cdb17ed6ccc`: file SHA-256
`a10fe3730f7bb3bacd20bd14ebbcfab2b3cf8c63a2db27783f1e8a9787b87ee6`,
git blob `abc6d4df165f684c2c93920d079585c600562dbb`.
Two transitive files differ only by the later `preserve_zero_support` option:
`adapters.py` historical/current blobs `5b861945…`/`e9085860…` and
`wan_forward_model.py` `e8b2fd44…`/`53bbbb77…`. Registration binds both old
and new files/blobs and the exact changed-file inventory. The historical YAML
must omit the option, the current default and instantiated runtime value must
both be false, and this source argument alone is not accepted as behavioral
proof.

Recurrent-delta and hybrid-anchor trajectories are deliberately excluded. The
fresh renderer study found a large recurrent flow improvement but did not pass
its complete frozen flow-plus-spatial family. No failed family may be imported
or used to rescue this raw-only screen.

## Independent prerequisite

The exact fresh 24-episode trajectory-consistent renderer gate must remain
complete and independently auditable with:

- registration identity
  `9cc556aba53d1defb69b0049dab67d12a3991decb917015bba5153c16cb8c2b1`;
- preparation identity
  `11444ee94ae6707869f44f00409386701c93b886d650ebc928bc4edec4f41a3a`;
- analysis identity
  `1471d0bb1f40aabc44d08c71b7eeeba3f0e88b9cb2a07e56dc6f7eb8b11034a0`;
- completion identity
  `715fc288ca3e5526cda3a5bb3329db650aa8ed15a61d2959be2349dd3df6dcfc`;
- decision `GO_raw_geometry_scaffold_pass`; and
- read-only audit status `audit_passed` with causal replay error at most
  `1e-6`.

That study prospectively showed raw commands outperform episode-shuffled raw
geometry on flow and every registered spatial metric, and outperform
hold-current geometry on the same family. Its raw-family pass—and only that
pass—authorizes this Wan screen.

## Causal field and tensor contract

For clip (i), let (q_{i,4}) be measured robot state at the last observed
video frame and let (a_{i,t,-1}) be the final absolute command in candidate
action chunk (t). The nine-pose raw path is

\[
  Q_i=(q_{i,4},a_{i,4,-1},a_{i,5,-1},\ldots,a_{i,11,-1}).
\]

No measured state value at frames 5 through 12 is indexed or enters the cache
predictor, and no future RGB value is indexed. Raw action samples are checked
bit-exactly against the immutable cached generator input.

For each transition, every source-visible articulated robot pixel is
back-projected into the source geometry, transported in its MuJoCo geom-local
coordinates, projected into the target pose, and checked against target
z-buffer visibility. The native field is

\[
 f_t(p)=\left(\frac{x' - x}{320},\frac{y' - y}{180},m,
 m\log\frac{z'+10^{-6}}{z+10^{-6}}\right),
\]

where (m=1) only for a valid, in-frame, z-buffer-visible target point.
Visibility uses frozen tolerance
`max(0.01 m, 0.02 * target_depth)`. Unsupported components are exact zero.

Each 180-by-320 top field is bottom-padded by 12 rows to 192-by-320. In each
8-by-8 block, visibility is area-averaged; displacement and log depth are
support-weighted averages. This yields eight transitions of shape
`[8,4,24,40]` per row. Compact float16 caches avoid storing known-zero history
and wrist values. At dataset read time, four transitions are packed into each
future Wan token, giving `[16,4,24,120]`: history tokens 0 and 1 and wrist
columns 40:120 are exact zero; future tokens 2 and 3 contain the top-view
field. Pack/unpack is exact and tested.

The recorded D405 vertical focal length is scaled to 180 rows. The official
ABC nominal camera extrinsics and a centered principal point are fixed; lens
distortion is recorded but not corrected. This limitation is part of the
claim boundary.

## Immutable cache registration and controls

The cache registration scans only manifest/action/camera metadata and D405
calibration. It freezes D405 eligibility, calibration identities, and shuffled
donors before rendering, training, validation
target access, or any generator outcome. At least 32 immutable val64 rows must
be D405 or the study stops. D405 rows missing observed state, exact action
equality, calibration, geometry, or articulated support are fatal. Non-D405
rows remain in the train stream with exact-zero fields and an explicit
ineligible flag; the primary evaluation population is the prospectively
registered D405 subset.

Registration also runs a pre-output renderer-runtime qualification in an
isolated cache-only virtual environment. Its lexical `bin/python` symlink,
complete symlink chain and resolved CPython binary, `pyvenv.cfg`, package
module files, and the NumPy 2.0.1, MCAP 1.4.0, and MuJoCo 3.3.7 distribution
`RECORD` files are content-bound. The active calibration/compression stack is
also direct-import and RECORD-bound: mcap-protobuf-support 0.5.4,
protobuf 7.35.1, zstandard 0.25.0, and the alternate MCAP codec lz4 4.4.5.
The qualification requires user-site isolation, `MUJOCO_GL=egl`, exact
imports/versions, a successful 8x8 offscreen render, and a successful decode
of the deterministic first registered train-D405 `/top-camera-info` message
from its zstd-compressed MCAP. The LACWM registration process and cache child
must produce the identical calibration identity. Every unambiguous file with
a declared RECORD SHA-256 is
rehashed and size-checked, including all native MuJoCo/NumPy libraries;
existing unhashed RECORD entries are also content-bound. The sole allowed
ambiguity is NumPy 2.0.1's duplicate generated `conv_template` bytecode row,
whose exact unhashed declaration, stale declared hash/size, and observed
hash/size are source-pinned. Zero other mismatches are accepted. Counts, total
bytes, and deterministic inventory identities are sealed. Each cache builder
must reproduce the identical receipt in its current process before creating
its split directory. Registration,
auditing/sealing, training, and evaluation remain in the separately registered
full LACWM runtime; the cache-only interpreter is never resolved before
execution and is used as the main process only for the two render jobs.

Four compact arrays are materialized and independently reconstructed by the
cache auditor:

| Source | Frozen construction | Attribution question |
|---|---|---|
| `raw` | local observed anchor plus raw action endpoints | deployable candidate |
| `episode_shuffled` | complete raw tensor from another D405 episode in the same planned-motion tertile | sample identity |
| `timeshift_plus_one` | local transitions shifted one slot without wrap; final slot zero | temporal alignment |
| `hold_current` | observed frame-4 pose repeated for all nine poses | action-dependent motion |
| `off` | runtime exact-zero packed tensor; not redundantly stored | condition presence |

Donors use fixed seed 20260831, are episode-disjoint, and are chosen within
split without RGB or measured future state. Row lineage binds ordered clip ID,
manifest index, action window/hash, observed state hash, calibration identity,
camera eligibility, donor, pose path, tensor hashes, visibility, and latency.
For `states.npz`, provenance records only absolute path, byte count, mtime,
device, and inode—never a whole-file content digest—and enumerates the only
indexed values: `joint_states[frame4]`, `gripper_states[frame4]`, and the
registered `[start:stop)` joint/gripper action slices. The auditor rejects a
state-file SHA field or any different slice declaration.
Global metadata binds the immutable RGB/action hashes, source commit, official
ABC commit, renderer prerequisite, row lineage, and all arrays. Protected test
data are unsupported.

## Matched training

Both arms start from the exact faithful-cascade update-1,000 VPM snapshot with
SHA-256
`de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a`
and parent run identity
`d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f`.
Its canonical model-state SHA-256 is
`d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0`.
They use seed 1234, eight B200 ranks, batch one per rank, 200 updates, fresh
identical AdamW, the same learning-rate schedule, no EMA, and one Wan call per
example.

| Arm | Field at training | Sole intervention |
|---|---|---|
| `FLOW-OFF` | registered raw tensor loaded/projected but hard-disabled at the Wan seam | parameter-matched control |
| `RAW-FLOW` | same registered raw tensor fused by the bounded residual adapter | causal raw geometry |

Both instantiate identical auxiliary parameters from the same seed; parent
auxiliary parameters are excluded identically from warm start. The field clock
and field velocity loss are zero. The parent VPM history corruption and native
sampling scheduler are unchanged.

Every update records all-rank SHA-256 identities for clip indices, actions,
raw field, video noise, and timesteps. The two 200-update traces, trainable
parameter schema, parent identity, and initial auxiliary state must compare
exactly before evaluation. A preemption may resume only through the existing
atomic checkpoint/trace contract; this prospective run is submitted
non-requeueable.

## Causal equal-call evaluation

Use every prospectively registered D405 val64 row, stateless noise IDs
0/1/2/3, and NFE 1/2/4. For each evaluation batch, materialize the complete
noise-by-endpoint grid below to CPU before opening any clean future RGB bytes:

- the untouched update-1,000 `PARENT-VPM` with its native condition-off public
  deployable sampler and no continuation updates;
- `FLOW-OFF` checkpoint with condition off;
- `RAW-FLOW` checkpoint with `raw`;
- the same `RAW-FLOW` checkpoint with `off`, `episode_shuffled`,
  `timeshift_plus_one`, and `hold_current`.

All same-NFE contrasts use identical explicit video noise and exactly NFE Wan
calls. No endpoint makes a field-model, teacher, or online feature call.

Before the validation mmap is constructed, a separate fail-closed receipt uses
only frames 0:5 and candidate actions from prospectively fixed immutable train
clip 0. First, a separate Python process imports the exact de65 model and all
repository-owned transitive model code from an independently registered clean
worktree at exact commit 6560866. The historical and current source paths use
the same separately registered Python, Wan, and VideoX external runtime. At
NFE 1/2/4 the historical process runs the public
`sample_future_deployable` method under condition off, its native aligned VPM
schedule, and fixed sample ID 7,000,000. The current evaluation adapter then
must equal those historical-process latents and decoded uint8 tensors bit for
bit; it must also equal a direct current public-method invocation bit for bit.
The snapshot, canonical 1,686-tensor state identity, resolved configuration,
both source worktrees/deltas, input hashes, explicit video noise,
full-precision equality between that noise and each first Wan input, Wan-call
counts, final latent, and decoded output are bound by
`parent_historical_reference.pt` and `parent_sampler_parity.json`. Any
non-bitwise result stops before validation outcomes; no train future RGB,
validation RGB, future measured state, clean video latent, teacher, or
protected test data are opened by this proof.

Primary scope is top-view pixels 0:320 / latent columns 0:40 at NFE 1. Report:

- decoded MSE in `[0,1]`;
- temporal-difference MSE including observed-frame-4 to first-future boundary;
- AlexNet LPIPS averaged over eight top-view future frames;
- all-view decoded and temporal MSE;
- future video-latent NMSE; and
- history VAE, adapter-plus-Wan, decoder, and end-to-end latency.

Component latency is reported for the two continuation models. To keep the
untouched parent's public sampler unmodified, its timing field reports only the
native public-call end-to-end duration; it is a quality/catastrophic-regression
control and is excluded from any component-latency claim.

LPIPS is a registered offline instrument, not a runtime dependency fetched
after outcomes. Registration hashes the installed LPIPS implementation and
packaged linear weights, the already-cached torchvision AlexNet checkpoint,
and the exact loaded state dictionary. The LACWM Python entry remains the
absolute lexical venv symlink: registration separately binds its symlink chain,
resolved base executable, `pyvenv.cfg`, isolated site-packages, and
LPIPS/torch/torchvision module and distribution `RECORD` files. Resolving the
entry is forbidden because it drops the LACWM site-packages. Every evaluation
rank must reconstruct the identical receipt without network access, and the
final auditor verifies that one identity served every arm and endpoint. The
frozen preflight is job `507388`: LPIPS 0.1.4 / torch 2.7.1+cu128 /
torchvision 0.22.1+cu128,
receipt `04f5013b7161fbf91ed6116d25f7e6ec66afc661024236ad27564b1899cb94be`,
loaded-state SHA-256
`abc218a76418de010923a57c9c55afb1c1040503b46e5015694ee79ea7c90a7d`,
and cached AlexNet SHA-256
`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.

A future-state `MEASURED_GEOMETRY_ORACLE` is deliberately not improvised into
this raw-only causal gate: constructing it would require a separate privileged
cache/access contract. Instead, nonselectable seam diagnostics report the
effective adapter gate, condition RMS/nonzero support, and same-checkpoint
raw-versus-off output-hash sensitivity. They can identify an unresponsive seam
but cannot rescue any quality decision. A measured-geometry oracle, if needed,
requires its own prospective study after this one.

NFE 2/4 are secondary dose-response checks and cannot rescue an NFE-1 failure.
Confidence intervals use 10,000 paired episode-cluster bootstrap replicates
with seed 20260831; all four noise seeds remain inside their episode cluster.

## Frozen decision

`ADVANCE_RAW_FLOW_SCAFFOLD` requires every condition:

1. Versus the untouched de65 `PARENT-VPM`, NFE-1 raw improves top-view decoded
   and temporal MSE by at least 3%, with paired lower bounds strictly above 1%,
   and has a strictly positive LPIPS improvement lower bound. All-view decoded
   and temporal MSE and latent NMSE must have nonnegative point effects and
   lower bounds strictly above -1%.
2. Versus matched `FLOW-OFF`, NFE-1 raw improves top-view decoded and temporal
   MSE by at least 3%, with paired lower bounds strictly above 1%.
3. Versus matched `FLOW-OFF`, top-view LPIPS has a strictly positive lower
   improvement bound.
4. At the same `RAW-FLOW` checkpoint, raw beats each of off, episode-shuffled,
   +1 time, and hold on both top decoded and temporal MSE by at least 1%, with
   strictly positive lower bounds.
5. Versus matched `FLOW-OFF`, all-view decoded and temporal MSE and latent NMSE
   have nonnegative point effects and lower bounds strictly above -1%.
6. Native-parent train-history parity passes bit-for-bit before validation is
   opened; the exact 200-update pairing trace passes; every NFE-1 endpoint
   reports one Wan call; flow-model calls are zero; future RGB and future
   measured state do not enter cache prediction or sampling; and protected test
   remains unopened.

Otherwise the decision is `STOP_FIXED_RAW_FLOW`. No secondary endpoint or
qualitative video can override the gate.

## Conditional next step

Only a pass authorizes a new prospective experiment in which the deterministic
robot field remains frozen and a small causal stochastic residual models
object/contact uncertainty (for example object SE(3), contact mode, visibility,
or particles) before RGB generation. Generated-versus-zero,
generated-versus-shuffled, oracle attribution, equal total calls, and serving
latency would remain mandatory.
