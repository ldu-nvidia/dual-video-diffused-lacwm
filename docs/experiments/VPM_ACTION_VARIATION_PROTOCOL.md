# VPM action-variation preservation screen (prospective)

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

## Intervention

For each of the eight future requested action chunks, with five low-level
commands per chunk, define four within-chunk velocities:

```text
delta_a[t,k,d] = a[t,k+1,d] - a[t,k,d],  k=0,...,3.
```

Per-coordinate population mean and standard deviation are fit once over only
the registered 512 training clips (`512 x 8 x 4` observations). Coordinates
with training standard deviation at most `1e-6`, including padded dimensions,
are mapped to exact zero; all other standardized values are clipped to
`[-8,8]`. The immutable statistics file records its source manifest/action
array hashes and is bound into registration, configs, checkpoints, and
analysis. Registration independently recomputes every fitted value and
whitening diagnostic from that exact action array; self-declared provenance is
insufficient. Validation and protected-test actions never contribute statistics.

A fixed-seed three-layer MLP maps the flattened standardized deltas plus the
existing morphology embedding to a 64-D residual `r_delta`. The ordinary
historical action encoder remains unchanged:

```text
z_base = ActionEncoder(a, morphology)
z_aug  = z_base + tanh(g) r_delta
```

`g` initializes to exact zero. Therefore both arms have the parent VPM
function at initialization. No arbitrary output separation is imposed.

| Arm | Delta MLP computed | Gate parameter/schema | Effective residual |
|---|---:|---:|---:|
| `AV-CONT` | yes | same trainable scalar | hard-masked to exact zero |
| `AV-DELTA` | yes | same trainable scalar | `tanh(g) r_delta` |

Both arms instantiate the same parameters, buffers, action branch, action pool,
Wan `ActionToControl`, LoRA transformer, and unused parameter-matched auxiliary
topology. The control computes the residual but a non-serialized arm flag
multiplies its effective gate by zero. Both use one Wan call per training
update and the same Wan calls at evaluation. The small candidate residual
gradient is the intended intervention; no claim of bit-exact backward FLOPs is
made.

## Frozen continuation

Both arms load every historical key from the update-1,000 VPM snapshot:

```text
f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21
```

Only the new `action_delta_residual.*` keys may be absent from the parent.
They are initialized identically with local seed `20260808`, without shifting
the parent's RNG stream. Both arms run exactly 200 updates, seed `1234`, eight
B200 ranks, local batch one/global batch eight, fresh AdamW (`lr=1e-4`, betas
`0.9/0.95`), 20-step warmup, cosine decay to `1e-6`, BF16 AMP, gradient norm
one, no EMA, and the exact same immutable clip order.

Every update records rank-combined exact hashes of actions, selected future
actions, standardized deltas, clean video latent, noisy video latent, native
timestep, and reference latent. Analysis fails before quality inspection if
these data/noise hashes differ between arms. A canonical identity over the
complete resolved scientific Hydra configuration is written before training and
revalidated from the saved config. The parent is loaded exactly once: all
historical tensors must match and the only permitted missing state is
`action_delta_residual.*`. Base/residual/augmented action
latents and the final action control are also logged as diagnostics. The
initial adapter-state hashes and zero gates must match.

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
with the residual gate temporarily hard-masked at aligned NFE 1. The runtime-only
mask does not mutate model state. A forward hook verifies actual Wan calls.
Preparation, Wan trajectory, decode, and total batch-one latency are reported
per endpoint with CUDA synchronization. The read-only `ActionToControl` probe is
timed outside deployment preparation and never enters total latency.

The evaluator records the exact 32-value future control after the trained Wan
`ActionToControl` projection, enabling effective-rank and shuffled-cosine
diagnostics without modifying sampling.

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

1. versus `AV-CONT`, decoded temporal-difference MSE point estimate and
   simultaneous lower bound are at least `+1%`;
2. video-latent NMSE and decoded MSE point estimates and bounds are each above
   `-1%`;
3. for both zero and episode-disjoint shuffled actions, the paired incremental
   action-specific separation `[(candidate diagnostic - candidate aligned) -
   (control diagnostic - control aligned)] / mean(control aligned)` is at least
   `+0.5%` in temporal-MSE point estimate and bound, with the same `-1%`
   guardrails;
4. within the trained candidate, native aligned sampling beats the otherwise
   identical residual-hard-masked endpoint by at least `+0.5%` in temporal-MSE
   point estimate and bound, with the same `-1%` guardrails.

Within-candidate aligned-versus-zero/shuffled contrasts remain descriptive;
they cannot satisfy causal attribution because the inherited control action
path may already separate those endpoints. In `AV-CONT`, native and explicit
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

A null result would reject this exact delta representation, adapter, and short
continuation. If the learned control remains low-rank, the next intervention
should enforce counterfactual/inverse action consistency. If control rank rises
but video attribution still fails, the bottleneck lies downstream in Wan's use
of control rather than in the action encoder.

## Implementation

- `robot_wm/modeling/networks/action_delta_residual.py`
- `projects/latent_action_models/lam/action_variation_conditioned_model.py`
- `robot_wm/utils/action_variation_trainer.py`
- `projects/latent_action_models/train_action_variation.py`
- `tools/action_variation_screen.py`
- `tools/action_variation_evaluate.py`
- `tools/analyze_action_variation.py`
- `tools/slurm/action_variation_workflow.py`
- `tools/slurm/action_variation.sbatch`
- `tools/slurm/submit_action_variation.sh`
