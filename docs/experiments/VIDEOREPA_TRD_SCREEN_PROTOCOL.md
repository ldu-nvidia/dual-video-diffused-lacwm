# Training-only VideoREPA relation-distillation screen

Status: prospective and unlaunched. This document and the implementation must
be committed in a clean checkout before registration. No validation or test
metric has been observed under this protocol.

## Question

Can a clean video representation help low-step LACWM generation **without being
available at inference**? The candidate uses cached clean V-JEPA 2.1 features
only as a training loss. At deployment it takes the same five observed RGB
frames and robot actions as the video-only VPM baseline. There is no online
teacher, predicted feature, cached target, projection, TRD hook, or auxiliary
branch in its video-only sampling path.

This is a controlled adaptation of VideoREPA (arXiv:2505.23656), not a claim
that the ABC cache encodes a complete physics model. VideoREPA's key observation
is that direct feature alignment can destabilize a pretrained video generator;
it instead compares within-representation token relations. Its loss is

\[
R^{d}_{ij}=\frac{y_{d,i}^{\mathsf T}y_{d,j}}
                 {\lVert y_{d,i}\rVert\lVert y_{d,j}\rVert},
\qquad
R^{d,e}_{ij}=\frac{y_{d,i}^{\mathsf T}y_{e,j}}
                 {\lVert y_{d,i}\rVert\lVert y_{e,j}\rVert},
\]

With the released VideoREPA hinge

\[
\ell_m(a,b)=\max(|a-b|-m,0),
\]

the registered loss is

\[
L_{\mathrm{TRD}}=
\operatorname{mean}_{d,i,j}\ell_{0.05}(R^{d}_{S,ij},R^{d}_{T,ij})+
\operatorname{mean}_{d\ne e,i,j}\ell_{0.05}(R^{d,e}_{S,ij},R^{d,e}_{T,ij}).
\]

Thus 0.05 is a tolerance: sub-margin errors contribute zero and 0.05 is
subtracted from every supra-margin error. The candidate objective is

\[
L_{\mathrm{ON}}=L_{\mathrm{video\ flow}}+0.5L_{\mathrm{TRD}},
\]

while the control returns exactly `L_video flow`. Both compute the same
detached relation telemetry, so the only gradient-level intervention is whether
the weighted relation loss enters the objective.

## Representation and alignment

- Teacher: immutable clean V-JEPA 2.1 PCA64 cache, one target per 13-frame ABC
  clip, shape `[B, 64, 4, 24, 120]`. Registration pins the exact train512 and
  val64 manifest/cache IDs plus the V-JEPA checkpoint and PCA SHA-256 values;
  a shape-compatible substitute is rejected.
- Student: output of zero-indexed Wan block 14 (the midpoint of the pinned
  30-block model) from the ordinary noisy-video training forward,
  `[B, 2880, 1536]` for the registered grid.
- The Wan tokens are reshaped to `[B, 1536, 4, 12, 60]` using the pinned
  `(1,2,2)` patch stride. No learned projection is used; teacher and student
  channel dimensions need not match when comparing cosine Gram matrices.
- Following VideoREPA Appendix C, temporal bin zero is excluded from both
  representations because the first 3D-VAE bin primarily anchors static
  semantics. Alignment therefore uses bins 1--3 rather than silently treating
  the semantic anchor as an ordinary motion bin.
- Width is split into three camera views before any pooling. Each view is
  independently average-pooled spatially to `H=2, W=4`; after the first-bin
  exclusion this yields `F=3`, 24 tokens per view, and 72 tokens per clip.
- Spatial relations compare the eight tokens within each frame and view.
  Temporal relations compare all spatial pairs across every ordered pair of
  distinct frames within a view. No pooling crosses a camera seam.

The scoped block hook exists only in model `forward`, which the trainer uses.
Gradient checkpointing is disabled in both arms so backward differentiates the
exact captured activation. The model rejects any block, pooling, VPM gate, or
checkpointing drift.

## Why this is not the failed A1 auxiliary head

The prior A1 arm trained a second velocity head to denoise V-JEPA state. Its
auxiliary state then had to be generated from noise, and the generated state was
not accurate enough to help the video branch at inference. TRD does not predict,
diffuse, inject, or reconstruct V-JEPA features. It only shapes the Wan hidden
geometry during training. Therefore teacher quality at inference is irrelevant:
the teacher is discarded.

The exact VPM checkpoint still contains its historical parameter-matching
modules in the state dict so the warm start is strict. The deployment override
does not execute them. It invokes the Wan video transformer directly, and the
evaluator hooks the V-JEPA/TF adapter, clock, and velocity head to require zero
calls.

## Matched arms

| Arm | Start | Updates | Objective | Inference |
|---|---|---:|---|---|
| TRD-OFF | exact VPM SHA `f67c7bae...` | 200 | video flow only | video-only Wan |
| TRD-ON | same exact VPM | 200 | video flow + `0.5 TRD` | same video-only Wan |

Both arms use seed 1234, eight B200 GPUs, batch one per GPU, fresh AdamW,
identical LR schedule, train clip order, corruption/timestep stream, LoRA
dropout, parameter schema, and one Wan call per update. The parent state is
loaded with `strict=True`; exclusions and remapping are forbidden. A production
shape 8-GPU forward/backward canary must finish first. Before that canary, a
separate single-process GPU check generates one train clip at NFE 1 through the
actual history-only deployment sampler. It also runs ordinary VPM
condition-off inference from the same keyed noise and requires bitwise equality
of the initial state, history reference, native BF16 Wan video velocity,
returned FP32 future, and decoded uint8 video. The persisted FP16 final-latent
evidence must also match bitwise, but is not presented as a native-FP32 check.
The custom path must make exactly one Wan call and zero calls to each watched
auxiliary/TRD module; the ordinary reference must call each of the control
adapter, TF projection, TF norm, clock net, and TF velocity head exactly once.
This prevents the comparison from silently using a partially bypassed
reference. Any difference fails the dependency chain before either full arm
can start.

Trainer validation and visualization use only the train512 cache and are
monitoring telemetry, not selection data. The sealed val64 paths are not
exported to either training process. After both update-200 checkpoints and
traces are complete, a CPU job writes an immutable post-training seal. Only a
successful seal can release the val64 jobs. No test argument exists.

## Evaluation

Each arm generates all 64 validation clips through the target-free RGB/action
dataset view. That view resolves only cached RGB and actions; its V-JEPA target
memmap must remain unopened. Ground-truth future RGB is encoded only after all
generation calls for the batch complete and is used solely for scoring. During
generation, future RGB remains in the CPU loader batch; the GPU sampler input
owns a separate allocation containing exactly the five observed frames.

Primary endpoint: aligned actions, NFE 1. NFE 2 and 4 are descriptive and
cannot rescue a failed primary endpoint. Every clip uses deterministic initial
noise keyed by immutable clip index, identical across arms and controls.

Reported paired metrics (lower is better):

- future video-latent NMSE;
- decoded RGB MSE;
- temporal-difference RGB MSE.

Action controls are also evaluated at every NFE:

- `aligned`: registered action chunks;
- `episode_shuffled`: a global cyclic donor proven to come from another
  episode in that batch;
- `zero`: exact zero action tensor.

They diagnose action dependence but are not model-selection endpoints.
The analysis reports paired within-arm error degradation for shuffled and zero
actions relative to aligned actions, with the same 10,000-sample bootstrap;
these diagnostic intervals cannot change the primary pass/fail decision.
Every cell records the SHA-256, shape, and dtype of the exact action tensor
passed to the sampler. The contract requires those hashes to remain fixed
across NFE, requires every shuffled hash to equal its declared donor clip's
aligned hash, requires all zero hashes to agree, and requires the complete hash
grid to be identical between arms.

## Preregistered decision gate

For each clip and metric, relative improvement is
`100 * (TRD_OFF - TRD_ON) / TRD_OFF`. A deterministic 10,000-sample paired
bootstrap supplies the 95% interval. TRD passes only if all six checks hold at
aligned NFE 1:

| Metric | point estimate | 95% lower bound |
|---|---:|---:|
| decoded MSE | at least +3% | at least +1% |
| temporal MSE | at least +3% | at least +1% |
| latent NMSE | non-negative | at least -1% |

This screen supports only a quick controlled claim on the registered ABC
distribution. It cannot establish physical commonsense, policy success, FVD,
or broad video quality. A pass motivates a longer multi-seed run and perceptual
metrics; a failure rejects this block/pooling/weight/budget combination, not all
training-only relational distillation.

## Safety and provenance

- personal private W&B only: `zijiandu/dual-video-diffusion-private`, group null;
- exact clean research and VideoX commits at every stage;
- exact VPM SHA, complete train target/RGB/action hashes, and complete val64
  RGB/action hashes; the unused val clean-feature array is never opened;
- multi-GiB hashes run before any NCCL process group;
- persisted B200 runtime receipts;
- identity-bound, revalidated deployment-sampler equivalence receipt;
- non-requeueable jobs and fresh outputs;
- `short` QOS wall time fixed to `02:00:00`, and one allocated GPU for
  seal/analysis bookkeeping because the `batch` partition rejects GPU-less
  jobs;
- default node exclusion:
  `pool0-0081,pool0-0089,pool0-0200,pool0-0343`;
- no protected test path, metric, or fallback.
