# Action-cycle Stage-0 recoverability protocol

## Question and claim boundary

This prospective screen asks whether clean, frozen Wan VAE latent motion is
predictive of the aligned robot-action segment. A pass licenses a later
training-only inverse-action critic; it does **not** show better video quality,
few-step generation, or inference speed. The critic would be discarded at
inference, so this route never requires a clean future feature at deployment.

Only the immutable 512-train and 64-validation RGB/action arrays are opened.
The cached V-JEPA target and every protected-test artifact are forbidden.
Validation contributes zero observations to normalization, ridge fitting, or
regularization selection.

## Audited temporal alignment

The actual pinned `AutoencoderKLWan_.encode` implementation consumes a
13-frame video in blocks

\[
  [0],\quad[1,2,3,4],\quad[5,6,7,8],\quad[9,10,11,12],
\]

and emits four causal latent bins. The run must additionally encode real RGB
prefixes of lengths 1, 5, 9, and 13 and show that each prefix output agrees
with the corresponding full-video latent prefix within the preregistered
`atol=2e-5, rtol=1e-5`. Shape alone is not accepted as evidence.

There are therefore three adjacent latent displacements, not two. Because
cached action chunk `i` contains the five low-level actions from sampled frame
`i` toward sampled frame `i+1`, the exact mapping is:

| latent displacement | new sampled-frame block | action chunks | low-level actions |
|---|---:|---:|---:|
| `z1-z0` | 1–4 | 0–3 | 0–19 |
| `z2-z1` | 5–8 | 4–7 | 20–39 |
| `z3-z2` | 9–12 | 8–11 | 40–59 |

The terminal cached action chunk 12 has no following sampled video frame and
is excluded. All three transitions are primary; transitions 1 and 2 are also
a required future-relevant guardrail so history-only recoverability cannot
produce a go decision.

## Feature, target, and frozen probe

Wan encodes the exact three-view panorama used by LACWM into
`[B,16,4,24,120]`; only then is width split into three 40-column views. For
bin `b` and view `v`, the fixed feature is

\[
x_{b,v}=\operatorname{vec}\!\left(
  P_{6\times10}(\operatorname{LN}(z_{b+1,v}))-
  P_{6\times10}(\operatorname{LN}(z_{b,v}))\right)\in\mathbb{R}^{960},
\]

where LN spans channel and spatial axes and `P` is fixed average pooling. The
target is the flattened aligned `[4,5,23]` action segment (460 scalars). It is
called an action-displacement target because it causes the latent displacement;
the action values are not differenced again. The constant ABC morphology need
not be regressed, and zero padding to the heterogeneous 157-wide action space
is excluded from metrics.

Nine linear ridges (3 transitions × 3 views) are fit on train, with predictions
averaged over views. Train-only population statistics normalize inputs and
targets. A single alpha is selected from
`1e-4,1e-3,1e-2,1e-1,1,10,100` by the exact ridge leave-one-out formula averaged
over all nine strata, then frozen before validation is opened.

Controls are the transition-specific train mean, a ridge fit to a deterministic
episode-disjoint bijection of train targets, and correct predictions scored
against an episode-disjoint bijection of validation targets. Retrieval is
scored against all 64 validation action trajectories. Shuffles preserve each
split and are bijective; no donor may share an episode with its recipient.

## Preregistered gate

Clip is the bootstrap unit; views and transitions are aggregated inside each
of the 64 paired clips. Ten thousand deterministic paired resamples use seed
`20260808`. One-sided percentile lower bounds use Bonferroni alpha `0.05/K`
over the complete preregistered comparison family.

For both all-three and future-relevant transition sets, aligned normalized MSE
must improve by at least 20% over each of the three controls, aligned cosine
must exceed shuffled-fit and shuffled-target cosine by at least 0.10, and
top-1 retrieval must exceed its episode-disjoint target assignment. Every
simultaneous lower bound must be strictly positive. Any failure yields
`stop_or_revise_action_cycle_path`; no post-hoc subset can override it.

All inputs, code, Wan assets, encodings, fitted ridge, rows, shuffles, and
results are content-hashed and bound to the preregistration identity. W&B is
locked to the authenticated owner `zijiandu`, private project
`dual-video-diffusion-private`, with no group.

