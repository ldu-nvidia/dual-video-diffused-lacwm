# VPM per-substep action-token cross-attention screen (prospective)

## Question and boundary

This one-seed, 512-train/64-validation ABC continuation asks whether repairing
the collapsed explicit-action path improves a one-call video generator and
makes its output causally specific to the requested action. The protocol is
frozen before either arm is trained or scored. The protected test split is not
configured, opened, or authorized after this screen.

The motivating train-split audit found that raw future action chunks retained
sample variation (cyclic-shuffle cosine `0.728`) and within-chunk temporal
deltas had centered effective rank about `92`, while the 32-value control seen
by Wan had cyclic-shuffle cosine `0.99885` and centered effective rank `1.164`.
This is an observational bottleneck diagnosis, not itself a causal result.

## Audited native paths

The current explicit-action path selects dataset chunks `actions[:,4:12]`,
encodes each complete `[5,157]` chunk to one `[1,64]` latent, then groups four
adjacent transition latents for each generated Wan temporal token. A learned
pool maps the resulting 256 values back to 64. `ActionToControl` maps that to 16
channels and spatially broadcasts each value over the entire latent grid. Wan
concatenates noisy video (16 channels), control (16), and reference (16) before
its pretrained 3-D patch embedding. This explains how raw action rank can be
lost before the transformer sees it.

The pinned VideoX Wan implementation already has a native cross-attention in
every `WanAttentionBlock`. Its raw context width is 4096, the frozen
`text_embedding` maps it to the Wan hidden width, and the fixed text sequence
length is 512. The registered null-prompt artifact is `[1,4096]`; Wan normally
pads it with 511 exact-zero raw tokens. The intervention below materializes
that same padding and occupies only the final 40 padded positions. It therefore
uses existing block APIs and pretrained parameter shapes. All video queries can
attend all 40 planned-action tokens: there is no query-to-action temporal mask.
That is the principal limitation of this minimally invasive route, but it does
not leak outcomes because the complete planned action sequence is a legitimate
deployment input.

## Intervention

For each of the eight future requested action chunks, retain all five low-level
commands as distinct tokens in transition-major order:

```text
token_index(t,k) = 5t + k,  t=0,...,7, k=0,...,4.
```

Per-coordinate population mean and standard deviation are fit once over only
the registered 512 training clips (`512 x 8 x 5` observations). Coordinates
with training standard deviation at most `1e-6`, including padded dimensions,
are mapped to exact zero; all other standardized values are clipped to
`[-8,8]`. The immutable statistics file records its source manifest/action
array hashes and is bound into registration, configs, checkpoints, and
analysis. Registration independently recomputes every fitted value and
whitening diagnostic from that exact action array; self-declared provenance is
insufficient. Validation and protected-test actions never contribute statistics.

A fixed-seed adapter combines each standardized 157-D command with the existing
64-D morphology embedding, adds separate learned transition and within-chunk
position embeddings, and projects it to Wan's raw 4096-D context width. This
produces 40 ordered tokens `r[t,k]`. The ordinary historical action encoder,
four-to-one temporal pool, and spatially broadcast control route remain
unchanged. The new route adds its tokens to the final 40 positions of the
existing 512-token null-text context:

```text
z_control = HistoricalActionPath(a, morphology)
context[-40:] = null_context[-40:] + tanh(g) r[0:40]
```

Wan's frozen `text_embedding` maps that context to the native hidden width, and
each pretrained transformer block consumes it through its existing
cross-attention. No Wan parameter shape, serialized key, block signature, or
sequence length changes. This is the smallest architecture-faithful explicit
token route supported by the current Wan API; it does not add a separately
masked temporal cross-attention operator inside every block.

`g` initializes to exact zero, so the current-code context is the parent null
context at initialization. This is not a claim that a fresh stochastic forward
is bit-identical to execution under the historical training commit.

| Arm | 40-token adapter computed | Gate parameter/schema | Context effect |
|---|---:|---:|---:|
| `AT-OFF` | yes | same trainable scalar | hard-masked to exact zero |
| `AT-ON` | yes | same trainable scalar | `tanh(g) r[t,k]` |

Both arms instantiate the same parameters, buffers, action branch, action pool,
Wan `ActionToControl`, LoRA transformer, and unused parameter-matched auxiliary
topology. The control computes all tokens but a non-serialized arm flag
multiplies its effective gate by zero. Both use one Wan call per training
update and the same Wan calls at evaluation. The small candidate token-route
gradient is the intended intervention; no claim of bit-exact backward FLOPs is
made.

The inherited parameter-matched auxiliary head receives an all-zero placeholder
in both arms. Its input and clock gates and loss are exact zero, and the dataset
must not expose or open an auxiliary-target array. The placeholder preserves the
parent parameter/call topology without supplying any clean future feature.

## Frozen continuation

Both arms load every historical key from the update-1,000 VPM snapshot:

```text
f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21
```

Only the new `action_token_adapter.*` keys may be absent from the parent.
They are initialized identically with local seed `20260808`, without shifting
the parent's RNG stream. Both arms run exactly 200 updates, seed `1234`, eight
B200 ranks, local batch one/global batch eight, fresh AdamW (`lr=1e-4`, betas
`0.9/0.95`), 20-step warmup, cosine decay to `1e-6`, BF16 AMP, gradient norm
one, no EMA, and the exact same immutable clip order.

Every update records rank-combined exact hashes of actions, selected future
actions, standardized actions, clean video latent, noisy video latent, native
timestep, reference latent, and the unchanged raw null context. Analysis fails before quality inspection if
these data/noise hashes differ between arms. A canonical identity over the
complete resolved scientific Hydra configuration is written before training and
revalidated from the saved config. The parent is loaded exactly once: all
historical tensors must match and the only permitted missing state is
`action_token_adapter.*`. Base action latents, raw/gated action tokens, native
action control, and final Wan context are also logged as diagnostics. The
initial adapter-state hashes, total/trainable parameter counts, and zero gates
must match exactly.

Before `torchrun`, a single-process boundary rehashes every registered input,
including the multi-GB parent, and seals those records in the immutable arm
plan. The B200 runtime verifier output is likewise identity-sealed as a per-arm
receipt and bound into the arm plan, training completion, evaluation inventory,
and final analysis. Distributed ranks validate those receipts before NCCL and
do not make rank zero rehash a large snapshot before an object collective. The
trained snapshot is hashed by the single-process W&B completion seal before the
evaluation `torchrun` starts.

W&B is restricted to `zijiandu/dual-video-diffusion-private` with
`group=null`, online mode, and registration-derived run identities. Training
fails unless the actual W&B run ID equals the registered identity, and the same
ID must be present in the final snapshot. Evaluation is blocked until the W&B
API reports that exact ungrouped private run as `finished`, with the registered
identity in its summary; a signed local completion receipt binds that result to
the training-completion artifact.

## Deployable evaluation

Each of all 64 validation clips is sampled independently at batch size one from
identical deterministic noise. One unmeasured aligned NFE-1 rollout per rank
warms the runtime before latency collection.
The sampler receives exactly five observed RGB frames, morphology, and one of:

- the clip's aligned requested actions;
- all-zero actions; or
- an episode-disjoint action sequence from the paired global validation index.

It receives no future RGB, clean video latent, auxiliary target, pretrained
feature, or teacher output. Scoring targets are constructed only after every
endpoint completes. The aligned endpoints use NFE `1,2,4`; zero and shuffled
attribution diagnostics use NFE `1`. A sixth endpoint evaluates each trained arm
with the token gate temporarily hard-masked at aligned NFE 1. The runtime-only
mask does not mutate model state. A forward hook verifies actual Wan calls.
Preparation, Wan trajectory, decode, and total batch-one latency are reported
per endpoint with CUDA synchronization. `total` is the arithmetic sum of those
three separately timed components, not one contiguous end-to-end wall-clock
measurement. Preparation is measured once per unique action-source/token-mode
pair for a clip and reused by aligned endpoints at different NFE. Host-to-device
loading, scoring, and the read-only `ActionToControl` probe are excluded. These
numbers are therefore descriptive component latency, not a closed-loop latency
claim.

The evaluator records the exact 32-value historical control after Wan's
`ActionToControl`, plus hashes and 40 per-token RMS values for the new context
delta. The hard-mask endpoint must have an exact-zero context delta. These
read-only probes enable effective-rank and shuffled-action diagnostics without
modifying sampling.

## Frozen decision

For a lower-is-better metric, paired relative improvement is:

```text
I = [mean(reference) - mean(candidate)] / mean(reference).
```

The analyzer uses 10,000 paired clip bootstraps, seed `20260808`. Quality has
nine fixed contrasts (`3 NFE x 3 metrics`), incremental action attribution has
six (`2 controls x 3 metrics`), and the trained-candidate hard-mask family has
three, each with its own one-sided Bonferroni correction.

The screen passes only if all conditions hold at NFE 1:

1. versus `AT-OFF`, decoded temporal-difference MSE point estimate and
   simultaneous lower bound are at least `+1%`;
2. video-latent NMSE and decoded MSE point estimates and bounds are each above
   `-1%`;
3. for both zero and episode-disjoint shuffled actions, the paired incremental
   action-specific separation `[(candidate diagnostic - candidate aligned) -
   (control diagnostic - control aligned)] / mean(control aligned)` is at least
   `+0.5%` in temporal-MSE point estimate and bound, with the same `-1%`
   guardrails;
4. within the trained candidate, native aligned sampling beats the otherwise
   identical token-hard-masked endpoint by at least `+0.5%` in temporal-MSE
   point estimate and bound, with the same `-1%` guardrails.

Within-candidate aligned-versus-zero/shuffled contrasts remain descriptive;
they cannot satisfy causal attribution because the inherited control action
path may already separate those endpoints. In `AT-OFF`, native and explicit
hard-mask aligned outputs must be tensor-identical, qualifying the ablation.

NFE 2/4 are descriptive and cannot rescue a failed NFE-1 claim. Effective
rank, shuffled-control cosine, latent temporal NMSE, decoded PSNR, learned gate,
and latency are diagnostics, not promotion criteria. No protected test follows
either outcome.

## Interpretation

A pass would establish that causal, inference-available action variation—not a
clean future feature—can improve the low-NFE VPM and make a subsequent causal
motion-plan branch scientifically plausible. It would require multi-seed and
longer training confirmation before a paper claim.

A null result would reject this exact per-substep context representation, adapter, and short
continuation. If the learned control remains low-rank, the next intervention
should enforce counterfactual/inverse action consistency. If control rank rises
but video attribution still fails, the bottleneck lies downstream in Wan's use
of control rather than in the action encoder.

## Implementation

- `robot_wm/modeling/networks/action_token_context.py`
- `projects/latent_action_models/lam/action_token_conditioned_model.py`
- `robot_wm/utils/action_token_trainer.py`
- `projects/latent_action_models/train_action_token.py`
- `tools/action_token_screen.py`
- `tools/action_token_evaluate.py`
- `tools/analyze_action_token.py`
- `tools/slurm/action_token_workflow.py`
- `tools/slurm/action_token.sbatch`
- `tools/slurm/submit_action_token.sh`

The registration preserves the absolute virtual-environment launcher path and
separately hashes its resolved interpreter target.  This is intentional:
invoking the target directly would bypass the virtual environment's
`pyvenv.cfg`.  The analysis dependency requests one GPU because this cluster's
`batch` partition rejects jobs without a GPU allocation, although analysis does
not use that device for a scientific computation.
