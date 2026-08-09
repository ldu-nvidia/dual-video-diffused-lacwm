# V-JEPA 2-AC inference-causal qualification

Date: 2026-08-08

Status: protocol and evaluator implemented; native-DROID train-only run pending
completion at the time of this protocol commit

## Decision question

Before connecting another feature branch to the video generator, test the
released V-JEPA 2-AC model at the seam it was actually trained to support:

> Given an observed frame embedding, current robot state, and a candidate
> action sequence, does the released predictor produce a sample-specific future
> video embedding that is more accurate for the aligned action than for
> causally available negative controls?

This is different from encoding the clean future video.  The causal rollout
never receives a future RGB frame, future target token, or future measured
state.  It predicts a compact interaction hypothesis before RGB generation, so
it remains a possible video analogue of Latent Forcing even though the original
time-frequency and future-encoder routes did not survive inference.

Stage 0 is deliberately a feature-prediction qualification, not a video-quality
experiment.  A failure stops this route before Wan/LACWM integration.  A pass
is necessary but not sufficient for a generator claim.

## Exact external source and checkpoint

- Official V-JEPA 2 source checkout: commit
  `45d025f636dfc58fc2426905fc4a1ab755b1c3e5`.
- Released file: `vjepa2-ac-vitg.pt` from
  `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`.
- Exact byte count: `11760743310`.
- Exact SHA-256:
  `0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b`.
- HTTP ETag: `12c945a28cc4f447e728fd2ba810607c-1402`.
- HTTP Last-Modified: `Fri, 30 May 2025 00:00:12 GMT`.

The pinned upstream `src/hub/backbones.py` has the production URL commented
out and activates a localhost testing URL.  The evaluator therefore creates
`vjepa2_ac_vit_giant(pretrained=False)` and manually loads the hash-pinned
`encoder` and `predictor` states.  At this exact source/checkpoint pair, both
state dictionaries have no missing keys, no unexpected keys, and no shape
mismatches.  The evaluator encodes those empty sets as an explicit fail-closed
contract.

As in `app/vjepa_droid/train.py`, checkpoint parameters and state/action inputs
remain FP32 while model forwards run under CUDA BF16 autocast.  The evaluator
does not cast parameter storage itself to BF16, which would change residual and
normalization numerics relative to the released training recipe.

The hub default `num_frames=64` is preserved.  It controls the predictor's
causal-mask capacity (32 tubelet groups); it is not changed to the sampled
eight-frame clip length.  A value of eight would allocate only four groups and
cannot cover the seven context/action groups in the official DROID recipe.

## Native DROID contract

The positive control uses only unprotected rows from the immutable DROID train
manifest, whose exact SHA-256 is
`cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e`.
Episodes are ranked deterministically and one clip is selected per episode, so
the cohort and shuffled donors are episode-disjoint.  Validation and the
identifier-only protected test manifest are not opened.

The official recipe is reproduced as follows:

| Field | Frozen value |
|---|---|
| Camera | `exterior_image_1_left` |
| Native/source rate | 15 FPS |
| AC sampling rate | 4 FPS (`ceil(15/4)=4` frame stride) |
| Sampled source offsets | `[0,4,8,12,16,20,24,28]` |
| Decoded geometry | exactly `8 x 180 x 320 x 3` |
| Spatial augmentation | scale `(1.777,1.777)`, aspect `(0.75,1.35)`, no flip |
| 180x320 fallback crop | `(top,left,height,width)=(0,38,180,243)` |
| Model input | eight independent RGB frames, each duplicated into a two-frame tubelet |
| Encoder output | `[1,8,256,1408]` after per-token layer normalization |
| State/action | `[xyz,euler_xyz,gripper]`, seven dimensions each |
| Predictor | 24 blocks, width 1024, 16 heads; ViT-g encoder |
| Primary rollout horizons | one and two AC steps; the checkpoint trained `auto_steps=2` |

Direct parquet inspection resolves an ambiguity in the LeRobot metadata.  The
payload uses `state[:3]` for Cartesian position, `state[3:6]` for Euler XYZ,
`state[6]` as constant zero padding, and `state[7]` as gripper.  The evaluator
asserts those observations and explicitly constructs
`concat(state[:6], state[7:8])`; it does not interpret columns 3:7 as a
quaternion and does not use the existing loader's erroneous padding-as-gripper
slice.

Actions are reconstructed exactly as in the official DROID loader.  For
consecutive poses `p_t=[x_t,theta_t,g_t]`,

\[
a_t = [x_{t+1}-x_t,\;
       \operatorname{Euler}_{xyz}(R_{t+1}R_t^\top),\;
       g_{t+1}-g_t].
\]

The inference-causal state update uses the same left-multiplied convention,

\[
x_{t+1}=x_t+\Delta x_t,\qquad
R_{t+1}=\Delta R_tR_t,\qquad
g_{t+1}=\operatorname{clip}(g_t+\Delta g_t,0,1).
\]

Reintegration error is recorded for every clip.  The raw logged seven-value
`action` column is retained only as a diagnostic comparator; official AC
training derives actions from measured pose pairs and does not use that column.

## Two evaluations and six action conditions

The nondeployable teacher-forced positive control supplies true encoder tokens
for frames 0--6 and measured states 0--6, exactly matching the official
one-step training path.  It tests whether the released checkpoint and local
data adapter reproduce the checkpoint's native behavior.

The deployable autoregressive evaluation starts with only the embedding of
frame 0 and measured state 0.  At every subsequent step, it appends its own
predicted frame tokens and integrates the same candidate action sequence to
obtain the next state.  Future RGB, target embeddings, and measured future
states are excluded.

Both modes evaluate:

| Condition | Construction |
|---|---|
| `aligned` | exact official pose-difference actions for the clip |
| `zero` | all seven action channels zero |
| `episode_shuffled` | aligned actions from the next deterministic, distinct episode |
| `time_shifted_plus1` | one AC step later, zero-filled without wraparound |
| `time_shifted_minus1` | one AC step earlier, zero-filled without wraparound |
| `logged_raw` | dataset's logged action column; diagnostic only |

Persistence of the initial observed embedding is also scored as a
history-only baseline.  Each predicted token tensor is compared with the
encoder target using L1, MSE, target-normalized MSE, and token cosine.  The
primary endpoint is paired per-clip L1 at horizons one and two.

## Frozen gates

For every combination of

- teacher-forced and inference-causal autoregressive mode;
- horizon one and horizon two; and
- zero, episode-shuffled, plus-one, and minus-one controls,

aligned actions must improve mean L1 by at least 5%, and the paired 10,000-draw
bootstrap 95% lower bound must be strictly positive.  All 16 comparisons must
pass.  This stringent native gate is intended to reject a generic video
extrapolator whose outputs are not materially action specific.

If the native gate passes, the following remain mandatory before ABC or Wan:

1. Compare the pretrained predictor with persistence, the logged-raw-action
   diagnostic, and a parameter-matched predictor trained from scratch under an
   identical train-only budget.  Pretraining must provide material value beyond
   those baselines.
2. Convert ABC planned joint commands with pinned YAM forward kinematics into
   the checkpoint's Cartesian state/action convention.  The rollout may use
   only the initial measured state and planned actions, never recorded future
   states.
3. Freeze a single view/arm mapping without looking at target outcomes.  ABC's
   `[N,13,3,180,960]` three-view RGB and `[N,13,5,23]` joint-action payload do
   not directly match the single-camera, single-arm DROID checkpoint.
4. Repeat aligned versus zero, episode-shuffled, and nonwrapping time-shifted
   tests on episode-disjoint ABC train data.  Include the raw action adapter and
   matched scratch predictor.
5. Measure observed-frame encoder latency, predictor-only latency, total
   auxiliary latency, peak memory, and amortized latency across candidate
   action sequences.  All auxiliary work counts against the 5--10 Hz target.

Only a pass through those gates authorizes a fixed-budget generator experiment:
`OFF`, `AC-PRED`, `AC-SHUFFLED`, and target-feature `ORACLE`, with equal Wan
calls and parameter count.  The oracle is a ceiling, never an inference claim.

## Artifact contract and entrypoint

`tools/vjepa2_ac_stage0.py` writes a hash-sealed `registration.json` before RGB
decode or model forward, followed by per-clip metrics, timings, a summary, and
`run_complete.json`.  The separate `audit` command rehashes every artifact and
the registered implementation/checkpoint.  Any source change, dirty upstream
checkout, checkpoint mismatch, payload-schema drift, protected row, output
reuse, or non-CUDA execution fails closed.

The Slurm wrapper is `tools/slurm/vjepa2_ac_stage0.sbatch`.  Large checkpoints,
logs, and run artifacts remain on `/lustre/fsw`; none are committed to Git.

## Claim boundary

A native-DROID pass would show only that the released V-JEPA 2-AC prior carries
action-attributed, inference-causal predictive information under its own data
contract.  It would not show improved generated video, transfer to ABC,
few-step diffusion, FVD/FID gains, real-time DAgger, or a new dual-diffusion
method.  Those claims require the staged controls above, a fixed-call generator
study, multiple seeds, and one untouched lockbox after all development choices
are frozen.
