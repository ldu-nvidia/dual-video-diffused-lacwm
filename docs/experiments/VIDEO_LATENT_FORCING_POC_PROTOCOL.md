# Video Latent Forcing Proof-of-Principle Protocol

Date preregistered: 2026-08-01

Status: low-resolution Phase 1 frozen; semantic Phase 3 amended and refrozen
before any semantic target, training run, or prediction result was produced

Pre-quality amendment (2026-08-01): an independent source audit found that a
fixed `0.9999` EMA would retain `0.9999^5000 = 60.65%` of initialization at the
Phase-1 endpoint, and that the first implementation had not copied the released
Latent Forcing zero initialization.  Before any training-quality sample or
metric existed, the model was changed to zero-initialize adaptive modulation
and both output heads, and EMA was changed to the deterministic warm-start
schedule `min(0.9999,(1+u)/(10+u))` after completed update `u`. Slurm job
`486176`, run `phase1-calibration-seed1234`, was already submitted from obsolete
commit `ce965305110eab95c47ea111b69509786a18b1be`; it remains systems-only and
cannot authorize training under the amended source. An exact-commit calibration
is required.

Phase-2 gate-definition amendment (2026-08-01): before any Phase-2 training,
generated sample, or quality metric existed, the previously qualitative
phrases "reaches quality," "LPIPS or the temporal metric," and "same budget"
were replaced below by executable numerical rules. The amendment also fixes
the full 3-arm/7-checkpoint evaluation inventory, primary control/NFE cells,
evaluator world/batch size, metric-provenance audit, target-feature identity,
and explicit mechanism/leakage evidence. B0's freeze-assertion field is now
correctly false because B0 has no auxiliary tensor; dual rows must be true.
Raw generated-video hashes were added so A1's fusion-off no-op is tested
bit-for-bit. These changes require a new exact-commit Phase-1 handoff and
calibration; no earlier cache or result can satisfy the amended gate.
Slurm job `486235`, run `phase1-calibration-seed1234`, completed successfully
at source `230fdc6ca0864e15636d6b0bef7fbce51f474958` with exit status `0:0` before
this amendment. It remains valid numerical systems evidence, but it cannot
authorize training under the post-amendment source.

Pre-quality R3D preprocessing correction (2026-08-01): exact-source Slurm job
`486355` at commit `dddd9f382c435531883380abb13d3ab628da152a`
completed the 200-update calibration and 5,000-update Phase-1 training without
nonfinite updates, then failed on the first validation batch before sampling,
per-clip metrics, a summary, or a gate existed.  The pinned torchvision video
preset uses `view` internally, but the required `[B,C,T,H,W]` to
`[B,T,C,H,W]` permutation was passed to it as a non-contiguous tensor.  The
evaluator now materializes that layout with `contiguous()` before invoking the
unchanged pinned preset, with a regression test exercising a genuinely
strided input.  Because cache, calibration, checkpoint, evaluator, and analyzer
source identities are intentionally one exact-commit contract, the completed
`dddd9f3` checkpoint remains systems evidence only.  A fresh cache,
calibration, 5,000-update training run, evaluation, and gate are required from
the corrected clean commit.  No quality threshold or scientific decision rule
was changed after the failed run.

Post-Phase-1 semantic-screen amendment (2026-08-01): corrected source commit
`6d35fbda76b3fdac6c04687ea95831d664c22231` completed the low-resolution-RGB
screen on all 890 validation clips. Its frozen gate, SHA-256
`33e1ab99092caf2a2820936082f3aa848b0b2406c8e92a20ceac03489a64269d`,
failed at every selectable NFE because autonomous temporal-difference MSE did
not beat both zero and episode-disjoint shuffled controls. The result stops
the low-resolution-RGB Phase 2 but, as preregistered, does not stop a semantic
screen. Before extracting any new V-JEPA target, training any semantic model,
or observing any semantic prediction metric, the previously underspecified
phrase "same ... gates" in Phase 3 was replaced below with an executable
semantic gate. The amendment also freezes a prefix-causal V-JEPA target,
train-only PCA population, controls, bootstrap rules, and the rule that a
semantic failure cannot be rescued by old full-clip oracle experiments.

Latent Forcing implementation audited: `AlanBaade/LatentForcing` commit
`fde8fc40377eaeeea49e6043e01c999b69779a53`.

## Research question

Can a future-video scratchpad that is generated autonomously from robot history
and actions make a from-scratch video flow model learn faster, generate better
video, or reach the same quality with fewer total network evaluations?

The first candidate is deliberately simple: a low-resolution RGB future.  It
tests the Latent Forcing mechanism without a pretrained video generator, VAE,
or semantic teacher.  V-JEPA 2 and DINOv2 are later representation arms, not
assumptions built into the first proof.

The deployable path never receives clean future RGB, clean future scratchpads,
or features extracted from clean future RGB.  Such inputs are labelled
`oracle_clean` and are mechanism diagnostics only.

## Clock and flow convention

This standalone experiment uses the original Latent Forcing **clean-time**
clock, not LACWM's native sigma clock:

\[
z(t)=t x + (1-t)\epsilon,\qquad v^*=x-\epsilon,
\]

where `t=0` is Gaussian noise and `t=1` is clean data.  Forward Euler therefore
updates `z <- z + (t_next - t) * v`.  LACWM elsewhere in this repository uses
`sigma=1` for noise, `sigma=0` for clean data, and the opposite velocity sign;
the two conventions must not be mixed.

## Frozen data contract

- Source: the locally complete portion of LeRobot DROID under the pinned input
  root recorded in each run's provenance.
- Camera: `exterior_image_1_left` only.  No artificial multiview concatenation
  and no transform across a camera boundary are allowed.
- Locally eligible population: the 9,780 episodes with at least 66 RGB/action
  frames observed in the 2026-08-01 read-only audit.
- The canonical JSON map from eligible episode ID to parquet row count has
  SHA-256 `3bc6f2c06abe74f1a60ddc4f9a44ce734fb8fa85f9ec94ac99e7bcc954993651`.
  The ordered train/validation/test episode-ID lists have SHA-256 values
  `cea3449f2a7cf3e9251b1fb1859f4fd8c3717a4ce29a344f2f48c65452e3ac12`,
  `58f1a863a7be8f273212030c902c568b32ed75df5aa79993a8aa5c1a7a0252e6`,
  and `aa36a731cde260ddb94875a0678435b339be18ebeaa4584b507cdbaeac71ba11`.
- Split: sort eligible episode IDs by
  `sha256("video-latent-forcing-poc-v1:<episode_id>")`; allocate the first
  8,000 to train, the next 890 to validation, and the final 890 to protected
  test.
- The protected manifest may contain episode IDs, lengths, and hashes.  Its
  video payload must not be decoded or scored until a validation-selected
  configuration and NFE are frozen.
- Clip: 5 history frames followed by 8 future frames, sampled at temporal
  stride 2 from the native 15 fps stream.  The selected 13-frame span covers
  24 native intervals, about 1.6 seconds.
- RGB is resized without cropping to `64 x 112`, quantized once to canonical
  uint8 values, scaled to `[-1,1]`, and stored
  as history `[3,5,64,112]` and future `[3,8,64,112]`.
- Conditioning actions are the 16 native 7-D DROID commands covering every
  transition from the final history frame to the final future frame.  Two
  commands occur between adjacent selected future frames because RGB stride is
  2.  The exact native indices are saved per sample.
- Clip starts use deterministic seed 20260801.  Training uses up to eight
  distinct starts per train episode (about 64,000 clips); validation and the
  protected manifest use one start per episode for paired comparisons.

All generated manifests and caches are immutable, content-hashed, and written
outside the Git repository on approved `/lustre`, `/mnt/data1`, or `/mnt/data2`
storage.  Run artifacts record source manifest hash, Git commit, environment,
command, seed, and Slurm identity. Data provenance additionally records the
clean builder Git commit and builder-tool SHA-256; training rejects a cache
built by a different source revision.

## Low-resolution RGB scratchpad

For each clean future clip `x`:

1. area-resize each frame from `64 x 112` to `32 x 56`;
2. split it into non-overlapping `1 x 4 x 4` spatiotemporal patches;
3. flatten each RGB patch into 48 scalars.

The resulting clean auxiliary state is

\[
s \in \mathbb{R}^{48\times8\times8\times14}.
\]

This mapping is deterministic and exactly invertible back to the `32 x 56`
low-resolution clip after resizing.  It is a coarse future video, not a
semantic representation.  That is useful for the first mechanism test, but it
does not dominate semantic features: low-resolution RGB retains unpredictable
appearance detail that DINOv2 or V-JEPA may discard.  A failure therefore
rejects only this scratchpad arm, not the broader video-forcing hypothesis.

## Model frozen for the primary comparison

- Future RGB patches: `(1,8,8)`, producing `8 x 8 x 14 = 896` video tokens.
- Auxiliary tokens: the aligned `48 x 8 x 8 x 14` scratchpad grid.
- Width 512, depth 12, 8 attention heads, MLP ratio 4.
- The resolved implementation has 41,963,760 parameters in both dual and
  parameter-matched video-only modes.
- Learned video and auxiliary input projections are added symmetrically with
  no learned near-zero gate.
- Both clean-time clocks enter every transformer block through adaptive layer
  normalization.
- Matching the released Latent Forcing/DiT initialization, the adaptive
  modulation projection and both clean-state output projections start at
  exactly zero. Positional and causal-context projections retain ordinary
  nonzero initialization.
- History RGB and the aligned 16 future-transition action commands are explicit
  conditions.
- One shared transformer trunk and separate video and scratchpad clean-state
  (`x`) heads.  Each prediction is converted to velocity as
  `(x_pred-z)/max(1-t,0.05)` for the velocity-equivalent loss and Euler sampler,
  matching the released Latent Forcing `t_eps=t_eps_inference=0.05`
  parameterization.
- The video-only baseline instantiates the same parameter schema; auxiliary
  parameters stay present but are strict runtime no-ops.  Parameter counts and
  executed-call accounting are reported.

The model is trained from scratch.  It does not load Wan, a VAE, V-JEPA,
DINOv2, or any previous LACWM checkpoint.

## Phases and arms

### Phase 0: implementation and numerical calibration

Run CPU tests and then exactly 200 optimizer updates per arm.  Calibration may
fix only numerical or systems defects: nonfinite values, shape/sign/mask
errors, out-of-memory, dataloader failure, or an obviously unusable loss-scale
ratio measured without comparing generated quality.  Any changed constant is
recorded and the 200-update calibration restarts for every arm.

### Phase 1: autonomous scratchpad screen

Train the scratchpad path for 5,000 updates with seed 1234.  Save checkpoints
at updates 500, 1,000, 2,000, and 5,000 and evaluate exact total NFE
`{1,2,4,8,12,20,25}` from fixed per-clip Gaussian noise.  It uses the same
global batch, optimizer, warmup, clipping, bf16, and EMA contract frozen below.

The screen passes only if, at NFE at most 12:

1. autonomous scratchpad NMSE is at most 0.50 and the clip mean of aligned
   48-D token cosines is at least 0.70;
2. decoding the generated scratchpad beats both zero and cross-clip shuffled
   scratchpads by at least 5% on LPIPS and temporal-difference MSE, with paired
   bootstrap 95% confidence intervals favoring the generated input;
3. at least 50% of the clean-scratchpad utility over zero is retained on both
   metrics; and
4. shuffled history/actions measurably degrade sample-aligned scratchpad
   generation, preventing an unconditional low-resolution-video model from
   satisfying the gate.

The formal gate uses only the update-5,000 warm-started-EMA checkpoint and
selects the smallest passing NFE in `{1,2,4,8,12}`. For a lower-is-better metric, relative
improvement of generated `G` over reference `R` is
`(mean(R)-mean(G))/mean(R)`. Each required 5% comparison must also have a
strictly positive paired-bootstrap 95% confidence-interval lower bound.
Bootstrap uses 10,000 clip-level resamples, base seed 20260801, and a
per-statistic seed derived from `sha256(base_seed + NUL + statistic_label)`.
“Measurably degrade” requires both a positive CI lower bound for autonomous
versus `context_shuffled` auxiliary-NMSE relative improvement and a positive
CI lower bound for the paired autonomous-minus-context-shuffled cosine
advantage. Retained utility is
`(mean(off)-mean(G))/(mean(off)-mean(oracle_clean))`; a nonpositive denominator
fails the gate rather than being silently divided.

This is deliberately a paired predictability gate against the one recorded
future, not a complete conditional-distribution test. A plausible alternative
future can score poorly against that realization. If the gate fails, the
low-resolution-RGB Phase-2 run stops and its result is reported as arm-specific.
Oracle-clean evidence cannot override the stop. The causal semantic/V-JEPA
screen remains independently eligible because semantic features can remove
unpredictable appearance entropy.

### Phase 2: faithful dual-video screen

Before any Phase-2 full training or evaluation, qualify exact deterministic
evaluation in two independently scheduled jobs. This qualification is a
pre-quality correctness gate and does not alter Phase 1. Both jobs must use the
same clean source commit and Python environment, the same passed same-source
Phase-1 gate, the pinned 890-clip validation manifest and R3D-18 weights, and a
fresh, nontrivial same-source A1 200-update calibration EMA. A checkpoint whose
raw or EMA clock-modulation, video-output-head, or auxiliary-output-head weight
remains identically zero is rejected.

Each qualification job is one exclusive node, eight B200s, eight torchrun
ranks, and evaluation batch size eight. It runs the evaluator's exact
determinism contract: deterministic PyTorch algorithms, deterministic cuDNN,
cuDNN benchmarking and TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`NVIDIA_TF32_OVERRIDE=0`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, and
CUDA bf16 autocast for sampling. It uses the production evaluation's
pair-preserving rank/batch layout and canonical RGB clamp, extracts pinned
R3D-18 real-target features for all 890 validation clips, and persists the
manifest-ordered little-endian C-contiguous float32 890-by-512 matrix, its
canonical raw-byte SHA-256, and a clip-ID-sorted digest matching the production
quality merger. The Phase-2 analyzer binds to the latter.

On every rank, the first complete eight-clip production-evaluation batch is
also sampled through the real A1 `sample_control` path at conceptual 25+25 NFE
under bf16 autocast. The generated auxiliary is computed once, exactly as in
evaluation, and reused by `autonomous`, `off`, `shuffled`, and `oracle_clean`.
Because A1 masks auxiliary fusion from the video branch, the raw generated
video tensor must be bit-identical across all four controls at both batch and
per-clip granularity. Thus each job probes 64 clips while exercising all eight
GPUs. Source files, Python/package/CUDA environment, Slurm job and node, GPU
UUIDs, manifests/provenance, gate/checkpoint/config/completion/weight files,
every target/probe input, the R3D matrix, per-clip and batch phase boundaries,
and generated outputs are content-hashed.

Finalize only two passed job records with distinct Slurm job IDs **and disjoint
node sets**. Environment, inputs, the full real-feature matrix digest, and all
probe outputs must match exactly. The immutable final qualification binds both
job records and the Phase-1 gate. Full B0/A1/L1 training requires this record,
stores it in every resolved training configuration, and evaluation must use the
identical record under the qualified runtime. The Phase-2 analyzer reproduces
the qualification and fails closed on a missing, changed, or unrelated record.
Qualification authorizes execution only; it is not evidence of model quality,
speed, controllability, or a Latent Forcing advantage.

The first seed contains these parameter-matched arms:

| Arm | Training intervention | Deployable inference |
|---|---|---|
| `B0` | video loss only; auxiliary path is a strict no-op | video from noise |
| `A1` | Latent Forcing branch clocks/losses; auxiliary tokens enter only while predicting the auxiliary state | auxiliary first, then video with auxiliary fusion off |
| `L1` | same branch clocks/losses as A1; symmetric auxiliary fusion on | generated auxiliary first, frozen during video phase |

For A1 and L1, draw `b ~ Bernoulli(0.4)` per example.

- Scratchpad branch (`b=1`): set video clean-time to exactly 0, draw
  `t_aux = sigmoid(N(-1.2, 1.0))`, and apply only auxiliary velocity loss.
- Video branch (`b=0`): draw video time from
  `sigmoid(N(-0.4, 0.8))`, replacing 10% of draws with `Uniform(0,0.5)`;
  draw `t_aux ~ Uniform(0.75,1.0)` and apply only video velocity loss.
- As in Latent Forcing, L1's **training** video branch fuses the corrupted
  auxiliary `z_aux=t_aux*s+(1-t_aux)*epsilon`; it is therefore teacher-forced
  from the clean auxiliary target during training. A1 masks this fusion off.
  This is distinct from deployable inference, which must first generate the
  auxiliary from history, actions, and noise and never receives `s`.
- The nominal auxiliary coefficient is 0.333.  Loss masks are normalized over
  the unchanged global batch, matching the released Latent Forcing recipe.

The video-clock mean above was corrected from the library default `-0.8` to
the released Latent Forcing training command's explicit `--P_mean -0.4`
during the pre-run source audit. No model-quality output had been generated or
inspected when this preregistration correction was made.

A1 enables the auxiliary projection for its scratchpad-branch examples—without
that input the shared trunk could not denoise the scratchpad—but disables it
for every video-branch example.  L1 enables it in both branches.  This is
implemented with a per-example fusion mask in one parameter- and call-matched
forward pass.

Train each arm for 20,000 optimizer updates after calibration, with global
batch 256 (32 per B200 on eight GPUs), bf16, and seed 1234.  Optimization is
AdamW with betas `(0.9,0.95)`, weight decay 0, learning rate `5e-5`, 500-update
linear warmup followed by a constant rate, and global gradient-norm clipping
at 1.0.  Track target EMA decay 0.9999 with the frozen short-run warm-start
`min(0.9999,(1+u)/(10+u))`, and use that EMA for every reported sample;
raw-weight diagnostics are labelled separately.  If the identical 200-update
systems calibration proves batch 256 infeasible, use gradient accumulation to
preserve global batch 256 rather than changing the optimizer contract.  Evaluate
updates `{500,1000,2000,5000,10000,16000,20000}`; the explicit 16,000
endpoint makes the preregistered training-efficiency gate directly observable.

Primary inference is a strict 25+25 cascade: generate the auxiliary state from
noise while video remains exactly noise, freeze the final generated auxiliary
state, then generate video while auxiliary remains bit-identical.  Report
**actual transformer calls**, not solver step labels.  Euler is the initial
solver; a higher-order solver is a new experiment and cannot silently change
NFE accounting.

Every checkpoint is evaluated as one complete immutable frontier with eight
ranks and evaluation batch size eight. B0 uses
`{0+1,0+2,0+4,0+8,0+12,0+20,0+25,0+50}` with only `off`; A1 and L1 use
`{1+1,2+2,4+4,6+6,10+10,25+25}` with
`{autonomous,off,shuffled,oracle_clean}`. The primary cells are B0 `off 0+50`,
A1 `off 25+25`, and L1 `autonomous 25+25`. A1 is a schedule/multitask
attribution control: because its auxiliary fusion is disabled, its 25
auxiliary inference calls are intentionally wasted and it is not a competitive
deployment baseline. B0 `0+50` is the practical equal-call video-only
baseline.

For every L1 checkpoint, use the identical generated auxiliary trajectory for:

- `autonomous`: aligned generated auxiliary injected during video generation;
- `off`: auxiliary injection disabled only during the video phase;
- `shuffled`: final generated auxiliary swapped by a manifest-global,
  content-hashed adjacent-pair derangement only during the video phase;
- `oracle_clean`: clean future auxiliary, clearly separated as a nondeployable
  ceiling diagnostic.

### Phase 3: representation extension

After the low-resolution result is frozen, repeat the representation screen
with temporally aligned, train-statistics-only normalized features. A failed
low-resolution gate does not block this screen because it is not evidence that
semantic targets are equally unpredictable. The first semantic arm is
**prefix-causal V-JEPA 2.1**. Per-frame DINOv2 remains a later alternative if
its local checkpoint, license, and preprocessing are separately pinned.

#### Prefix-causal V-JEPA target

Let the observed frames be `h0,...,h4` and the future targets be `f0,...,f7`.
For future index `j`, construct the exactly 16-frame teacher prefix

\[
P_j=[\underbrace{h_0,\ldots,h_0}_{10-j\ \text{copies}},
     h_0,h_1,h_2,h_3,h_4,f_0,\ldots,f_j].
\]

The last two frames are `(h4,f0)` when `j=0` and `(f[j-1],f[j])` otherwise.
Thus the last V-JEPA tubelet is aligned to the transition ending at `f_j`, and
the exact future-support invariant is

\[
\frac{\partial s_j}{\partial f_k}=0\quad\text{for every }k>j.
\]

The model may still condition on the complete planned 16-action sequence,
which is available at deployment; the invariant concerns unavailable future
RGB only.

Use the canonical cached `64 x 112` DROID RGB without crop, resize it to
`384 x 672`, map it from `[-1,1]` to `[0,1]`, and apply the pinned ImageNet
mean/std used by the official encoder. Run the official V-JEPA 2.1 ViT-B
encoder from source commit
`45d025f636dfc58fc2426905fc4a1ab755b1c3e5` with checkpoint SHA-256
`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.
The clean source checkout's MIT `LICENSE` has SHA-256
`cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173`.
The canonical RGB-cache provenance has SHA-256
`3320244e843ccaa84828b4bbecb9c227870706be3bbbc0e6e1c28eda1ac317e0`;
its train, validation, and identifier-only protected-test manifests have
SHA-256 values
`cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e`,
`b8773a8627e887bf0c0a31cfae6ff537ba6a99e0b1b4f11efec011e3983d8d99`,
and `e05bb46087152c4ea07820ce8290ae99c0a084dbcf5d576aac370a61486e925d`.
Semantic extraction must reject any other base population, any overlap with
the protected episode IDs, or any protected row carrying cached RGB.

Teacher extraction is frozen to CUDA bfloat16 with one DROID clip per rank per
forward. Fit the PCA by exact float32 centered covariance eigendecomposition
on CUDA with TF32 disabled. Encoder device, dtype, extraction batch size, PCA
device, and Python/PyTorch/CUDA/NumPy runtime identities are part of the
immutable PCA and cache identities; a resumed build may not mix them. Before
the full cache, require a real-checkpoint non-square `384 x 672` forward, a
later-future perturbation causality check, and an eight-rank interrupted/resumed
mini-cache whose resumed bytes equal an uninterrupted reference. This
systems-only mini-cache uses a fixed synthetic projection (zero 768-D mean,
the first 48 coordinate axes, unit eigenvalues, epsilon `1e-6`) so it can run
before the scientific PCA is fitted. The reference build, graceful partial
stop, and resume execute as three separate eight-rank processes. Every smoke
artifact is labelled non-production/non-adoptable and must be rejected by the
production PCA/cache validators.
Its 16-frame output has grid `[8,24,42,768]`. Keep only temporal token 7,
average each non-overlapping `3 x 3` spatial cell, and obtain
`r_j in R^[8,14,768]`:

\[
r_{j,a,b}=\frac{1}{9}\sum_{u=0}^{2}\sum_{v=0}^{2}
E(P_j)_{7,3a+u,3b+v}.
\]

Fit one 48-component PCA whitening transform to **all** 896 tokens from the
256 training clips with the smallest
`sha256("causal-vjepa2-pca-v1:" + clip_id)` ranks. No validation or test token
may influence the mean, components, eigenvalues, signs, or whitening epsilon
(`1e-6`). Canonicalize each component sign by making its largest-absolute
loading positive. Apply the frozen transform to every train/validation token
and stack future indices to obtain

\[
s\in\mathbb{R}^{48\times8\times8\times14}.
\]

This exactly preserves the low-resolution arm's 896-token grid, 48 auxiliary
channels, shared-transformer geometry, and 41,963,760-parameter budget. The
teacher is an offline target extractor only. It is forbidden in the optimizer,
validation sampler, and deployable sampler. Protected test RGB remains unread
and no protected semantic cache is built during selection.

The already completed legacy V-JEPA studies do not answer this arm: their
`[64,4,24,120]` target was one bidirectional encoding spanning observed and
future frames and all three views. At inference all four future-dependent bins
started from noise. Those studies reject that exact full-clip target, not the
one-view prefix-causal target above.

#### Semantic predictor screen

Train the same from-scratch Phase-1 model for exactly 5,000 optimizer updates,
after its own exact 200-update numerical calibration. The only target change is
low-resolution RGB `s` to prefix-causal V-JEPA `s`; initialization, clean-time
clock, optimizer, global batch 256, EMA, history/actions, fixed clip noise,
validation population, and NFE grid `{1,2,4,8,12,20,25}` remain unchanged.

For each destination validation clip, pair the manifest-adjacent clip as an
episode-disjoint donor. The fixed validation split contains one clip per
episode, so no donor shares an episode. Evaluate these sources from the same
destination-addressed Gaussian auxiliary noise:

- `autonomous`: destination history and actions; compare to its own clean
  semantic target;
- `donor_target`: reuse the autonomous generated state but compare it with the
  donor's clean target; this target is metric-only and never a model input;
- `context_shuffled`: donor history and donor actions; compare the generated
  state with the destination clean target;
- `history_shuffled`: donor history and destination actions, diagnostic only;
- `actions_shuffled`: destination history and donor actions, diagnostic only;
- `zero` and `oracle_clean`: metric-only lower/upper diagnostics, never
  deployable model inputs.

Report per clip: auxiliary NMSE, the mean aligned 48-D token cosine,
temporal-difference auxiliary NMSE, and the mean temporal-difference token
cosine. Temporal differences are along the eight aligned future indices.
Every record binds destination/donor IDs, source hashes, initial noise, target,
generated state, actual transformer calls, checkpoint/EMA, cache/PCA, and the
zero teacher-call assertion.

At the update-5,000 EMA checkpoint, select the smallest NFE in
`{1,2,4,8,12}` only if all of the following hold simultaneously:

1. autonomous auxiliary NMSE is at most `0.50`, token cosine is at least
   `0.70`, and temporal-difference auxiliary NMSE is at most `0.50`;
2. versus `donor_target`, autonomous improves ordinary NMSE and
   temporal-difference NMSE by at least 5%, with paired-bootstrap 95% CI lower
   bounds above zero, and has an ordinary token-cosine advantage whose paired
   CI lower bound is above zero;
3. versus `context_shuffled`, autonomous improves ordinary NMSE and
   temporal-difference NMSE by at least 5%, with paired-bootstrap 95% CI lower
   bounds above zero, and has an ordinary token-cosine advantage whose paired
   CI lower bound is above zero; and
4. cache validation proves the prefix support, PCA population, train/validation
   separation, source/checkpoint hashes, and zero teacher calls during both
   training and inference.

The bootstrap is the same 10,000 paired clip-level resamples and deterministic
SHA-derived statistic seeding used in Phase 1. `history_shuffled` and
`actions_shuffled` are reported separately but do not hard-gate the arm because
visual history can legitimately dominate short-horizon actions. For the
whitened target, zero has NMSE one and clean oracle has NMSE zero, so
`1 - autonomous_nmse` is also reported as retained zero-to-oracle utility.
NFE 20 and 25 remain diagnostic and cannot pass selection.

If this semantic gate fails, freeze the result and do not launch semantic
Phase 2. If it passes, run a fresh parameter-matched B0/A1/L1 Phase-2 study
with the same video-quality and generated-state attribution gates described
above; RGB LPIPS, temporal metrics, and R3D18-Frechet become valid again only
when the video branch is actually evaluated.

### Phase 4: acceleration

Distillation, consistency training, or solver tuning is prohibited until a
generated-only representation passes the quality and attribution gates.  The
selected dual checkpoint is then compared with the fresh B0 quality/NFE
frontier.  A speed claim requires equivalent quality with at least 20% lower
p95 end-to-end latency, including scratchpad generation.

## Metrics and claim gates

All comparisons use fixed clip IDs and initial noise.  Report curves versus
optimizer update, wall time, actual total NFE, and measured latency.

Primary quality metrics are:

- a pinned, non-candidate video feature Frechet distance;
- frame LPIPS;
- temporal-difference LPIPS/MSE; and
- action-conditioned trajectory consistency when an independent evaluator is
  available.

The non-candidate R3D-18 Kinetics-400 V1 weight is the official torchvision
file with SHA-256
`b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d`.

Generated RGB is deterministically clamped to `[-1,1]` before LPIPS,
R3D18-Frechet, and saved media, matching the valid image domain.  The fraction
of clamped values is reported and raw, unclamped RGB MSE remains secondary.
Phase-1 `32 x 56` scratchpad videos are bilinearly upsampled to `64 x 112`
before the same frozen perceptual extractors are applied.

Raw `oracle_clean` RGB, scratchpad, and phase-boundary media are not persisted
or uploaded. Its nondeployable validation diagnostics do persist metric scalars
and local pinned R3D feature vectors so distributional calculations remain
recomputable; those records are never represented as deployable evidence.

RGB MSE, PSNR, SSIM, token NMSE/cosine, losses, gradient norms, activation
ratios, memory, throughput, and per-phase latency are secondary or mechanism
telemetry.  Paired bootstrap intervals and raw per-clip records are saved.

The gate analyzer requires every cell in the complete 3-arm by 7-checkpoint
matrix above; a partial, duplicate, differently batched, or differently
controlled frontier fails closed. The four primary quality metrics are
R3D18-Frechet, frame LPIPS-Alex, temporal-difference LPIPS-Alex, and
temporal-difference MSE. It verifies exact pinned extractor versions,
preprocessing, and weight hashes, recomputes every Frechet value used by a gate
from saved 890-by-512 features, and requires one bit-identical real-target
feature matrix across all gate cells.

The one-seed L1 screen passes only if all five criteria are true:

1. **Training efficiency:** among L1 checkpoints
   `{500,1000,2000,5000,10000,16000}`, select the earliest whose primary
   `autonomous 25+25` cell is no worse than B0 update 20,000 `off 0+50` on all
   four primary metrics and whose recorded cumulative optimizer wall time is
   strictly lower than B0 update 20,000. All three arms must expose positive,
   finite, strictly increasing cumulative wall times at every registered
   checkpoint.
2. **Same-update/sample/NFE quality:** L1 update 20,000 `autonomous 25+25`
   must improve R3D18-Frechet by at least 10% relative to B0 update 20,000
   `off 0+50`; each of frame LPIPS, temporal-difference LPIPS, and
   temporal-difference MSE must be at most `1.01` times B0. This matches 20,000
   updates, 5.12 million examples, and 50 inference transformer calls. It does
   **not** match training FLOPs: L1 trains an auxiliary prediction head, so
   this criterion is a quality/sample-efficiency screen and never a speed
   claim.
3. **Attribution:** each comparison must pass separately: L1 `autonomous`
   versus A1 `off`, L1 `autonomous` versus its own `off`, and L1 `autonomous`
   versus its own `shuffled`, all at update 20,000 and 25+25. For each, at
   least one of the three clip-paired metrics above must have a 10,000-resample
   candidate-minus-reference percentile-bootstrap 95% interval with upper
   bound strictly below zero. All three paired metric means and
   distribution-level R3D18-Frechet must independently regress by no more than
   1% relative to the reference; a nonpositive denominator fails.
4. **Auxiliary mechanism:** B0 is a strict auxiliary-state no-op. In every dual
   row, the generated phase-boundary tensor is shared bit-for-bit across
   controls, each control is bound to its registered aligned/donor/oracle
   tensor, and the conditioning tensor remains bit-identical after every video
   step. The shuffled donor mapping is manifest-global and content-hashed. A1
   raw generated video must be bit-identical across all four controls.
5. **No inference leakage:** the explicitly documented L1 training branch may
   condition on a corrupted clean-target auxiliary, matching Latent Forcing.
   Every deployable validation/test row must receive only history, actions,
   and its clip-keyed video/auxiliary noise, execute with zero teacher calls,
   and bind its video phase only to an autonomously generated aligned or donor
   auxiliary. Clean-future conditioning is permitted only for explicitly
   nondeployable `oracle_clean` rows and can never satisfy this gate.

The analyzer emits schema `video-latent-forcing-poc-phase2-gate-v1`, the exact
decision rules and cells, paired statistics, checkpoint-wall provenance,
R3D-target digest, extractor/weight evidence, and explicit mechanism/leakage
counts. Any nonfinite metric, nonpositive relative-comparison denominator,
missing file/hash, source/data/noise mismatch, dirty source, or mutable output
fails closed.

The one-seed result is only a preregistered screen: its repeated
checkpoint/metric looks use nominal paired intervals without familywise error
control, and R3D18-Frechet has no uncertainty interval. An "obvious advantage"
is claimed only after the frozen one-seed gate passes,
the selected configuration is repeated with optimizer seeds
`{1234,2234,3234}` paired respectively with evaluation-noise seeds
`{20260801,20260802,20260803}`, and the three-seed result is confirmed once on
the protected test set.
The initial implementation code-locks optimizer seeds 2234/3234 and all
protected-test decoding. Those paths are enabled only in a later committed
stage that recomputes the one-seed/three-seed selection from raw validation
evidence; a hand-authored pass flag is never sufficient authorization.
Failed gates remain negative results for the stated representation, data,
model, schedule, and budget; they are not generalized into impossibility
claims.

## Required tests and saved evidence

- clean-time endpoints and velocity sign;
- branch masks and loss normalization;
- shape, patchify/unpatchify, and camera-view isolation;
- B0 auxiliary strict no-op under arbitrary auxiliary inputs and clocks;
- phase-boundary and frozen-auxiliary invariants;
- shuffled-control identity before the intervention point;
- action/history causality and protected-split non-overlap;
- exact model-call accounting;
- resume equivalence and distributed metric aggregation;
- immutable resolved config, provenance, logs, checkpoints, sample videos,
  per-clip metrics, and final gate decision.

Production LACWM runs and checkpoints remain immutable.  Legacy
`dual_diffusion.enabled` remains `false`; this standalone proof uses a new,
isolated package and artifact root.

No independent action-conditioned evaluator is currently implemented.
Therefore this proof cannot support a controllability, policy-quality,
real-time, or DAgger claim. Those require a separately preregistered evaluator
and measured end-to-end latency after the representation gate is confirmed.
