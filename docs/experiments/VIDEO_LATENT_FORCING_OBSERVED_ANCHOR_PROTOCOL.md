# Observed-anchor residual Video Latent Forcing protocol

Date preregistered: 2026-08-07

Status: contingent family and matched-control continuation frozen before any
observed-anchor cache, checkpoint, candidate sample, or candidate metric exists

## Question and execution condition

Can a dual video diffusion state remain useful at inference if clean scene
content comes only from observed history and diffusion generates future change,
rather than attempting to regenerate eight mostly redundant absolute future
features?

This family may run only if no primary temporal-target arm in
`VIDEO_LATENT_FORCING_TEMPORAL_FOLLOWUP_PROTOCOL.md` passes its generated-only
development gate, or as an explicitly labelled cheap proxy-validity probe that
cannot support a video-quality claim. The already-running five-arm temporal DOE
at clean commit `f638b493bc5bf0fad0faa4283a9c56deb5a5f764` may establish that
contingency, but none of its models, checkpoints, or metrics may serve as the
observed-anchor comparison baseline. It does not receive clean future RGB,
future semantics, optical flow, or a target-derived state at inference.

## Observed-only semantic anchor

Let the five observed RGB frames be `h0,...,h4`, the eight future frames be
`f0,...,f7`, and the existing prefix-causal V-JEPA targets be

\[
S\in\mathbb{R}^{48\times8\times8\times14}.
\]

Construct one exactly 16-frame observed prefix

\[
P_{-1}=[h_0]^{11}\mathbin\Vert[h_0,h_1,h_2,h_3,h_4].
\]

Run the same official V-JEPA 2.1 encoder, select the final temporal tubelet,
apply the same non-overlapping `3 x 3` spatial pooling, and apply the same
train-only `768 -> 48` PCA whitening transform. This produces

\[
A=\Phi(P_{-1})\in\mathbb{R}^{48\times8\times14}.
\]

The final tubelet of `P[-1]` contains `(h3,h4)`, immediately preceding the
existing target `S[0]`, whose final tubelet contains `(h4,f0)`. Repeating `h4`
for all 16 slots is forbidden because it would create an artificial static
video outside the established causal-prefix construction.

The public anchor extractor accepts `history` only. Tests must perturb every
future frame and prove bit or declared-tolerance invariance of `A`. The online
path quantizes the extracted anchor to float16 and back to float32 exactly as
the training cache does.

## Increment state

Define `S[-1]=A` and

\[
D_j=S_j-S_{j-1},\qquad j=0,\ldots,7.
\]

Fit one train-only mean and standard deviation per channel over the
`64,000 x 8 x 8 x 14` increment population:

\[
Q_{c,j,h,w}=\frac{D_{c,j,h,w}-\mu_c}
                  {\max(\sigma_c,10^{-6})}.
\]

The observed anchor has shape `[B,48,8,14]`; the diffused state has shape
`[B,48,8,8,14]`, identical to the existing auxiliary geometry. Accumulation,
subtraction, and moment computation use float64 over float16 source values.
Training encode/decode uses float32. Decode an autonomous prediction by

\[
\hat D_j=\sigma\hat Q_j+\mu,\qquad
\hat S_j=A+\sum_{k=0}^{j}\hat D_k.
\]

Unlike the temporal `DELTA` arm, this state contains no absolute future anchor:
all clean absolute content comes through the observed-only deterministic skip.

## Cache and provenance

Do not weaken the existing producer-bound causal-cache or PCA validators. The
new anchor implementation must either:

1. rebuild the PCA plus train/validation future-semantic caches under its exact
   clean commit, then require the rebuilt future target bytes to match the c114
   targets before comparison; or
2. use a separately reviewed producer-attestation bridge that runs the exact
   recorded validator and independently rechecks every file/content hash.

The anchor cache contains train and validation only and records:

- exact prefix frame map and final pair `(h3,h4)`;
- V-JEPA source, license, checkpoint, encoder, and PCA identities;
- tensor shape, float16 dtype, byte count, and SHA-256;
- manifest order and disjoint episode identities;
- `future_tensor_read=false` inside the anchor extractor;
- `protected_test_accessed=false` and `test_rows_extracted=0`.

Cached and online anchor extraction must agree on a frozen real validation
sample before training. Protected-test RGB or identifiers beyond the already
frozen exclusion evidence remain unopened until selection.

## Minimal semantic screen

Run a self-contained, prospectively matched two-arm continuation from one later
clean commit:

- `C-ABS`, a fresh continuation-local rerun of the registered absolute-semantic
  `ABS` arm; and
- `AINC-OFF`, which diffuses normalized anchored increments while the model
  still receives the existing raw history/actions and uses `A` only for
  deterministic decoding.

`C-ABS` is not a rerun of the five-arm temporal DOE: it is the new experiment's
single control arm. The old DOE is used only to authorize the contingency. Both
continuation arms start from scratch under the same later clean source commit;
there is no architecture or parameter-count difference. This costs one extra
control training run but removes code-drift, initialization, data-order, RNG,
optimizer, EMA, and evaluator-version ambiguity from the primary comparison.
For compatibility with the already registered trainer/evaluator, `C-ABS`
retains the machine-readable DOE arm label `ABS`; `C-ABS` denotes its
continuation-local experimental scope.

Freeze seed 1234, initialization, data order, global batch 256, optimizer, EMA,
one independent 200-update calibration per continuation arm, one from-scratch
5,000-update primary run per arm, final EMA weights, bfloat16 model autocast,
float32 state, and uniform-clean-time Euler sampling. Finish both calibrations
before either primary run. Evaluate all 890 development clips with identical
clip-addressed noise. `AINC-OFF` uses NFE `{1,2,4}`; `C-ABS` may additionally
materialize the already-registered descriptive higher-NFE cells, but only its
same-NFE `{1,2,4}` autonomous cells enter this gate.

Primary metrics are decoded absolute semantic NMSE/cosine and temporal-
difference NMSE/cosine. Normalized increment-space NMSE is diagnostic. Controls
are:

- autonomous generated increments and the correct observed anchor;
- `anchor_static`, which sets decoded increments exactly to zero and repeats
  `A` over all eight future slots (normalized value `-mu/sigma`);
- `mean_increment`, which sets normalized increments to zero and therefore
  decodes the train mean increment at each slot;
- donor target and context-shuffled generation;
- history shuffled;
- three fixed episode-disjoint action offsets `{1,17,101}`, factual attribution
  only;
- `anchor_decode_shuffled`, which reconstructs with a donor observed anchor;
- zero and oracle-clean, metric-only.

A cell is eligible only if it has decoded NMSE at most `.50`, cosine at least
`.70`, and temporal NMSE at most `.50`; improves ordinary and temporal NMSE by
at least 5% over continuation-local `C-ABS` at the same NFE; improves temporal
NMSE by at least 5%
over each of `anchor_static`, `mean_increment`, `context_shuffled`, and
`donor_target`; and attributes ordinary semantic content to the correct
observed anchor by improving ordinary NMSE by at least 5% and ordinary cosine
by a strictly positive amount over `anchor_decode_shuffled`. Relative NMSE
improvement is the ratio of paired sample means; cosine attribution is the
difference of paired sample means. Use common 10,000 clip bootstrap resamples,
seed 20260807, and one-sided lower bounds at confidence
`1-.05/3=.983333...` for the three selectable cells. Every required relative
NMSE point estimate and lower bound must be at least 5%; the ordinary-cosine
point estimate and lower bound against `anchor_decode_shuffled` must be
strictly positive. These multiple requirements form an intersection-union
gate within each cell, so the family-wise correction remains over the three
selectable NFE cells rather than over each conjunct.

### Prospective mathematical correction before execution

This paragraph and the immediately preceding eligibility rule were amended
before any observed-anchor cache, normalization, checkpoint, generated sample,
or metric existed. The original frozen text required temporal-NMSE improvement
over the donor-anchor reconstruction control. That test is structurally
impossible: if only the time-constant reconstruction anchor is replaced, then

\[
\left(A' + \sum_{k=0}^{j}\hat D_k\right)
-\left(A' + \sum_{k=0}^{j-1}\hat D_k\right)=\hat D_j,
\]

so `autonomous` and `anchor_decode_shuffled` have identical decoded temporal
differences (apart from irrelevant floating-point cancellation). The corrected
gate therefore tests donor-anchor attribution only with ordinary semantic
NMSE/cosine. Temporal attribution is tested against controls that change the
generated increments, their target association, or the trivial increment
trajectory: `context_shuffled`, `donor_target`, `anchor_static`, and
`mean_increment`. This is a prospective repair of an invalid test, not a
response to candidate results.

### Prospective matched-control recovery before execution

The self-contained `C-ABS` rule was added while the five-arm temporal DOE was
already running from commit `f638b493bc5bf0fad0faa4283a9c56deb5a5f764`, but
before any observed-anchor artifact or metric existed. A later observed-anchor
implementation cannot honestly claim exact source matching against that run.
Rather than weaken comparability with a cross-commit exception or rerun the
five-arm family, this continuation trains its own single `C-ABS` control beside
`AINC-OFF`. The frozen no-pass record from the running DOE is authorization
evidence only. The analyzer must reject that DOE's `ABS` summary as the numeric
baseline and require `C-ABS` and `AINC-OFF` to share the continuation commit,
model initialization recipe, manifests, cache bytes, data order, optimizer,
EMA, training geometry, seeds, evaluator, and paired clip-addressed noise.
This is a prospective repair of experimental matching, not a response to an
observed AINC result.

If multiple cells pass, select smallest NFE, then lowest temporal NMSE, then
lowest ordinary NMSE. Select at most one cell. Only it may receive the same
one-shot protected-test procedure and thresholds as the temporal follow-up;
the lockbox cannot be opened if an earlier family already opened it.

## Explicit-anchor factorial after a positive minimal screen

Only after `AINC-OFF` passes development and lockbox gates, add a parameter-
matched anchor projection

```python
anchor_projection = nn.Conv2d(48, 512, kernel_size=1, bias=False)
```

It maps `A` to `[B,512,8,14]`, broadcasts it over eight future positions in
exact `(time,height,width)` order, and adds it to aligned auxiliary tokens. It
adds 24,576 parameters; every factorial arm instantiates it and masking occurs
after projection so the off state is an exact zero no-op.

| Arm | Diffused target | Anchor-token projection |
|---|---|---|
| `ABS-OFF` | absolute `S` | off |
| `ABS-ON` | absolute `S` | on |
| `AINC-OFF` | anchored increments `Q` | off |
| `AINC-ON` | anchored increments `Q` | on |

For the on arms, `anchor_input_shuffled` gives the model a donor anchor but
decodes with the correct destination anchor; `anchor_full_shuffled` uses the
donor anchor for both. This separates learned anchor conditioning from the
deterministic reconstruction skip.

## Conditional video stage

Only a semantic arm confirmed on the one-shot lockbox may enter video training.
At inference:

\[
A=\Phi(H),\quad
\hat Q=G_\theta(\epsilon_Q,A,H,U),\quad
\hat X=F_\theta(\epsilon_X,\hat Q,A,H,U).
\]

After semantic warmup, video-loss examples receive stopped EMA-generated
`Q_hat`, never only clean or forward-noised future increments. Required controls
are parameter-matched video-only, auxiliary-trained/fusion-off, generated-
increment fusion, evaluation-only shuffled generated increments, and
evaluation-only oracle-clean increments.

Compare at equal total transformer calls `C_Q+C_X`; report online V-JEPA anchor
latency separately and in end-to-end throughput. A positive result requires
better generated-only temporal quality at equal total calls, no spatial or
perceptual regression, and a lower-call equivalent-quality point. If ViT-B
anchor latency prevents real-time DAgger, distillation into a lightweight
history-only causal encoder is a subsequent experiment, not excluded latency.
