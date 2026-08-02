# Causal V-JEPA 2 semantic-screen result

Date: 2026-08-01

## Decision

The frozen prefix-causal V-JEPA 2 semantic gate **failed** at every selectable
auxiliary NFE. No NFE was selected, so the protocol stops this representation
before jointly training or evaluating the video branch.

This is a valid negative result for one precisely defined representation and
training/sampling recipe: a PCA-whitened, 48-channel prefix-causal V-JEPA 2.1
target, a 41,963,760-parameter from-scratch predictor trained for 5,000
updates, and its autonomous Euler sampler. It is not evidence that V-JEPA,
semantic forcing, or dual video diffusion cannot improve video generation.
The video loss and video output head were deliberately disabled, so this run
did not measure decoded video, FID/FVD, or generation latency.

## What was tested

For observed frames `h0,...,h4` and future frames `f0,...,f7`, the offline
teacher encoded eight 16-frame prefixes. Prefix `j` ended at `f_j`, so its
feature could not depend on any later future frame. The final V-JEPA temporal
token was pooled to an `8 x 14 x 768` grid, projected by a train-only 48-D PCA
whitening transform, and stacked into the target
`[48, 8, 8, 14]`.

The deployed path never called V-JEPA and never received a clean future
feature. It generated the semantic state from Gaussian noise using only the
destination history and planned actions. The full validation compared:

- `autonomous`: destination history and actions against its own target;
- `donor_target`: the identical autonomous prediction against an
  episode-disjoint donor target;
- `context_shuffled`: donor history and actions against the destination
  target;
- `history_shuffled` and `actions_shuffled`: attribution diagnostics; and
- `zero` and `oracle_clean`: metric-only references.

The selection rule required, at some NFE in `{1,2,4,8,12}`, autonomous NMSE
at most `0.50`, cosine at least `0.70`, temporal-difference NMSE at most
`0.50`, and at least 5% paired improvement over both donor-target and
context-shuffled temporal NMSE with positive 95% confidence bounds.

## Execution and evidence

- Scientific source: `c11487f6e83908687f27026ce2ac2e7d8d41461c`
- Data: 64,000 DROID training clips and 890 episode-disjoint validation
  clips; protected test RGB was not accessed
- PCA: 256 train clips, 229,376 tokens, 48 components; SHA-256
  `52348899c2e1c7aa73434c0db7fa9679e1c450d37fdcfa10e64a7d95ac86a5df`
- Cached targets: train `[64000,48,8,8,14]`, SHA-256
  `547c4579cf978ac2b9527cb038693259af678a2b07268ab1434706dc128051c4`;
  validation `[890,48,8,8,14]`, SHA-256
  `ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034`
- Calibration: 200 updates, zero nonfinite updates, checkpoint SHA-256
  `6025b1958892ddececcaa8ad85079b5f5c4b231cca245a64047b6bb4ad065292`
- Training: 5,000 updates, global batch 256 on eight B200s, zero nonfinite
  updates, checkpoint SHA-256
  `f7586f23030a489fc6a673ea3bb6c6cfecccdbe5269c62f6de697d1dc4f9f9cc`
- Training auxiliary loss: `2.717255` at update 1 and `0.523571` at update
  5,000; first-500 and last-500 means were `2.1278` and `0.53785`
- W&B: private project `zijiandu/dual-video-diffusion-private`, calibration
  run [`b0hjqijn`](https://wandb.ai/zijiandu/dual-video-diffusion-private/runs/b0hjqijn),
  training run
  [`wajx851w`](https://wandb.ai/zijiandu/dual-video-diffusion-private/runs/wajx851w),
  and evaluation run
  [`valz3i71`](https://wandb.ai/zijiandu/dual-video-diffusion-private/runs/valz3i71);
  all had no group
- Evaluation job `486939`: `COMPLETED 0:0` on eight B200s after an independent
  NCCL preflight; exactly 43,610 rows (`890 x 7 x 7`) and 49 cells
- Evaluation per-clip SHA-256:
  `4f5f4c764209d59fd463d7bdc02d4a9d17bbea48375b39264b53fa526cca0c19`
- Evaluation summary SHA-256:
  `de677312309693e2948d55f406053c1b3447328488654991e208c3eb248f961d`
- Gate job `486964`: CPU-only, `COMPLETED 0:0`; gate file SHA-256
  `ffdb339f72fc7eb0e85dccf86b55c242d41312646f4409d3f8757bd6826065be`
- Gate decision SHA-256:
  `5cdce25196c84f823787d7f5a4df2082b34a51bb9ead3ad28b341818c828870d`

Every metric was finite. All 43,610 rows recorded zero teacher calls and
`clean_future_target_entered_sampler=false`. The donor-target control reused
the autonomous generated tensor bit-for-bit. An independent read-only audit
recomputed every row inventory, summary mean, control mapping, input/call
hash, bootstrap decision, and artifact digest.

Operational failures and scheduler rejections produced no scientific
observation. Job `486842`
published valid distributed-cache evidence and then exited because its wrapper
prematurely checked a later-stage gate; the adopted evidence is bound by the
r1 recovery attestation. Job `486935` failed its mandatory NCCL preflight with
a node-local CUDA OOM before creating the evaluation directory, W&B run,
sampler call, or metric row. The byte-identical retry was job `486939` on a
different node. The first C5 submission was rejected before a job existed
because QoS `normal` requires a GPU; job `486964` used the cluster's
zero-GPU `cpu-normal` QoS with the byte-identical CPU analyzer. These
recoveries changed no model, data, seed, NFE, control, metric, or threshold.

## Frozen validation result

| NFE | Semantic NMSE | Token cosine | Temporal NMSE | Semantic gain vs donor | Temporal gain vs donor | Temporal gain vs shuffled context | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.6147 | 0.6131 | 1.0440 | 49.98% | 0.24% | 0.19% | fail |
| 2 | 0.6721 | 0.5760 | 1.1860 | 49.45% | 0.70% | 0.51% | fail |
| 4 | 0.7683 | 0.5335 | 1.3748 | 47.17% | 1.02% | 0.69% | fail |
| 8 | 0.8552 | 0.5051 | 1.5462 | 44.97% | 1.36% | 0.89% | fail |
| 12 | 0.8960 | 0.4930 | 1.6692 | 43.77% | 1.53% | 1.00% | fail |

At NFE 1, the semantic result is meaningful but insufficient. Autonomous
NMSE `0.6147` was much better than zero (`1.0`) and both donor/context controls
(approximately `1.228`), with large positive paired cosine and NMSE effects.
The model therefore learned destination-specific semantic content rather than
merely reproducing a population mean.

That content came overwhelmingly from visual history. At NFE 1,
history-shuffled NMSE/cosine were `1.2206/0.1447`, while actions-shuffled
values were `0.6233/0.6056`, close to autonomous `0.6147/0.6131`. The generated
state consequently carries little measured action-specific information.

Temporal prediction is the decisive failure. NFE-1 temporal NMSE `1.0440`
was already worse than the zero reference (`1.0`), and the paired temporal
gains over donor and shuffled context were only `0.24%` and `0.19%`, far below
the required 5%. Every selectable NFE failed all three absolute-quality checks
and both temporal-effect checks. Semantic attribution and cosine checks passed.

More Euler evaluations monotonically worsened semantic NMSE (`0.615` to
`0.950` by diagnostic NFE 25), cosine (`0.613` to `0.475`), and temporal NMSE
(`1.044` to `1.924`). This is consistent with sampler/off-trajectory drift,
but the present evidence does not distinguish that mechanism from a weak
endpoint denoiser or an unsuitable absolute-feature target.

## Conclusion

The clean V-JEPA target is useful as an offline supervisory signal, and the
network can predict coarse scene semantics from robot history. The autonomous
latent available at inference is not yet a usable video Latent Forcing prior:
it misses the future dynamics, barely responds to actions, and degrades rather
than improves with additional denoising calls. Wiring this state into the
video model now would confound representation failure with video-fusion
quality and would not support a faster-generation claim.

## Next experiment

Run a small sampler-versus-denoiser diagnostic on this frozen checkpoint
before retraining:

1. At a dense clean-time grid, apply the model once to true forward-noised
   semantic targets and measure direct clean-state and velocity error.
2. At the same times and identical noise, compare autonomous rollout states
   produced by direct-`x`, Euler, Heun, and nonuniform low-NFE schedules.
3. Keep history/actions fixed and repeat action shuffling to localize when the
   action signal disappears.

If single-call, on-distribution denoising is good but rollout is poor, the next
model change should add rollout/consistency supervision and select a stable
solver. If the endpoint itself is poor, replace the absolute semantic target
with an action-sensitive temporal target such as anchored or first-difference
V-JEPA features and add explicit temporal-difference/contrastive losses. Re-run
the same semantic gate. Only a generated-only state passing at NFE 4 or lower
should advance to parameter-matched B0/A1/L1 video training and actual
FID/FVD, temporal-quality, and end-to-end latency evaluation.
