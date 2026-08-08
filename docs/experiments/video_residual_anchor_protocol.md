# Video residual-anchor matched quick screen

Status: prospective, validation-only, and not launched by this implementation.

## Question and claim boundary

Does a direct, invertible future-video-latent coordinate system improve the
historical VPM/LACWM checkpoint after an equally short continuation, without
requiring a clean future feature at inference?

This is an **adjacent structural baseline**, not dual diffusion. It introduces
no second generative state, teacher, auxiliary condition, or trainable
parameter. A positive result would justify a larger residual-coordinate study;
it would not establish the proposed dual-diffusion mechanism.

## Frozen parent and data

- Parent: the exact historical VPM update-1000 model snapshot, SHA-256
  `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`.
- Parent training source: commit
  `9cf8e6922f35a5d6645e3128545953723bf54da2`.
- Training: the immutable 512 ABC clips already frozen by the V-JEPA study.
- Evaluation: all 64 immutable ABC validation clips.
- Registration reparses both pinned manifests and requires zero overlap in
  both `clip_id` and `episode_dir` (512 unique train episodes and 64 unique
  validation episodes).
- Protected test data are neither accepted nor opened.
- The RGB/action dataset never resolves the cached V-JEPA target array.

The Wan geometry is fixed: 13 RGB frames (5 history + 8 future) encode to
`[B,16,4,24,120]`. The first two Wan tokens are observable history and the
last two are generated future.

## Coordinate transform

Let `Z = [Z0,Z1,Z2,Z3]` be an absolute Wan latent and let `H0,H1` be the
history-only VAE encoding of the five observed frames. The reference tensor is

```text
Href = [H0, H1, 0, 0].
```

Both arms replace the first two model-state tokens with exactly `H0,H1`; they
are never noised and never scored. The control arm is

```text
Pabs = [H0, H1, Z2, Z3].
```

The candidate arm is

```text
Pres = [H0, H1, Z2 - H1, Z3 - Z2].
```

The inverse is

```text
Zhat2 = H1 + Pres2
Zhat3 = H1 + Pres2 + Pres3.
```

Thus the candidate generates displacement from the last observed latent and a
subsequent increment. The inverse is applied before VAE decoding and before
every latent, decoded-video, and temporal metric. The first screen uses no
data-dependent normalization: this avoids adding a second treatment and keeps
the checkpoint adaptation interpretable. Train-only residual whitening is a
separate follow-up only if this unnormalized screen is numerically unstable.

## Matched continuation

Two arms start model-only from the same parent:

| Arm | Future coordinates | Updates |
|---|---|---:|
| `VPM-ABS` | absolute `Z2,Z3` | 200 |
| `VPM-RESIDUAL` | `Z2-H1,Z3-Z2` | 200 |

The arms have exactly the same parameter names/shapes, trainable parameters,
parent weights, data order, seed 1234, global batch 8, optimizer, learning-rate
schedule, gradient clipping, validation cadence, and lack of EMA. Every update
records clip-index moments, timestep moments, and a Gaussian-noise probe. The
analyzer requires these pairing fields to be exactly equal for all 200 updates.
The otherwise zero-weighted auxiliary head retains the parent's
`head_condition_on_tf_clock=true` setting; only the future-video coordinate
mode differs between resolved arm configs. The reusable validation iterator is
infinite in both arms, as in the parent, so all three matched validation calls
complete rather than exhausting after the first pass.

Each arm uses a fresh identical AdamW optimizer (`lr=1e-4`, betas 0.9/0.95),
20-step warmup, cosine decay to `1e-6`, eight B200 ranks, and one sample per
rank. The parent has no EMA; neither continuation creates one.

Checkpoint equality is parameter matching, not functional matching. The
parent learned absolute video coordinates, whereas `VPM-RESIDUAL` immediately
changes both its noisy state coordinates and velocity target. Its 200 updates
therefore include adaptation away from the absolute-coordinate warm start. In
addition, both screen arms use the same exact-clean-history clamp, which differs
from the parent's noisy-history training trajectory. A negative result means
only that the residual coordinate did not win under this short, shared-clamp
continuation; it cannot rule out from-scratch, transformed-head, or
full-convergence residual training.

## Deployable sampler

The public sampler accepts only:

- exactly five observed RGB frames;
- the action sequence and morphology ID;
- explicit video and parameter-matched auxiliary Gaussian noise;
- NFE.

It has no argument for clean future RGB, a clean video latent, V-JEPA/DINO, a
teacher, or any auxiliary target. The known history tokens are clamped exactly
at every call. The candidate is cumulatively inverted only after the final
transformer call and is then decoded by the frozen Wan VAE.

## Paired validation endpoints

Every arm evaluates all 64 validation clips at autonomous NFE 1, 2, and 4.
The same clip-keyed video and no-op-auxiliary Gaussian tensors are reused
across NFE and arms. A forward hook must observe exactly the declared number
of Wan transformer calls.

An additional NFE-1 action-shuffled endpoint cyclically exchanges actions in
each fixed two-clip local batch. It is diagnostic only and cannot select the
candidate. Receipts bind each shuffled action tensor to the recorded,
episode-disjoint donor clip and prove that it differs from the recipient's
registered action tensor.

Evaluator-owned clean targets are constructed only after all four sampling
endpoints for a batch have completed. Metrics are:

- future absolute video-latent NMSE;
- future latent temporal-difference NMSE;
- decoded RGB MSE in `[0,1]`;
- decoded PSNR;
- decoded temporal-difference MSE in `[0,1]`.

## Screen decision

For each lower-is-better metric and NFE, compute the paired relative
improvement of `VPM-RESIDUAL` over `VPM-ABS`. Use 10,000 paired bootstrap
replicates and one-sided Bonferroni correction over 12 preregistered
NFE/metric contrasts (`alpha = 0.05/12`).

The earliest NFE passes only if:

1. decoded MSE improves by at least 2% and its corrected lower bound is above
   zero;
2. decoded temporal MSE improves by at least 2% and its corrected lower bound
   is above zero;
3. absolute latent NMSE point estimate and lower bound are both above -1%; and
4. latent temporal NMSE point estimate and lower bound are both above -1%.

A pass means only `screening_signal_for_residual_coordinate_followup`. A miss
means `no_controlled_residual_coordinate_advantage_in_quick_screen`. Neither
outcome supports a dual-diffusion, FVD, generalization, latency, downstream
DAgger, or real-time claim.

## Execution outline (not executed during implementation)

After these files are independently reviewed, committed, pushed, and cloned
into a fresh clean B200 checkout:

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
PY=$BASE/envs/lacwm-b200-py310/bin/python
TOOL_REPO=$BASE/src/dual-video-diffused-lacwm-video-residual-anchor-<commit7>
HISTORICAL_REPO=$BASE/src/vjepa2-latent-forcing/dual-video-diffused-lacwm
STUDY=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3
OUT=$BASE/artifacts/dual_video_diffusion/video_residual_anchor/video-residual-anchor-seed1234-<commit7>-v1

$PY $TOOL_REPO/tools/video_residual_anchor_evaluate.py register \
  --tool-repo "$TOOL_REPO" --expected-commit <full-commit> \
  --historical-repo "$HISTORICAL_REPO" --study-root "$STUDY" \
  --output-root "$OUT" \
  --train-manifest <absolute-train-manifest> \
  --train-cache-metadata <absolute-train-metadata> --python "$PY"

$PY $TOOL_REPO/tools/video_residual_anchor_workflow.py plan \
  --registration "$OUT/registration.json"
```

Submit one non-requeued two-hour, eight-B200 job per arm using
`tools/slurm/video_residual_anchor_screen.sbatch`, then run:

```bash
$PY $TOOL_REPO/tools/video_residual_anchor_analyze.py \
  --registration "$OUT/registration.json" \
  --absolute-inventory "$OUT/evaluation/vpm-abs/inventory.json" \
  --residual-inventory "$OUT/evaluation/vpm-residual/inventory.json" \
  --output "$OUT/analysis/analysis.json"
```

W&B is fixed to the private personal project
`zijiandu/dual-video-diffusion-private` with no group.
The launcher exports the registered VideoX/Wan runtime before sourcing the B200
activation guard, and each evaluation inventory hashes all eight rank-row files
and all eight rank receipts. Analysis revalidates the 200-update training trace,
base completion receipt, snapshot identity, rank receipts, and cross-arm tensor
pairing before computing an effect.
