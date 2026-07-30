# V-JEPA 2.1 dual-video-diffusion controlled study

## Decision

The tested design did **not** achieve the joint objective of faster training,
higher reconstruction quality, and faster deployable inference.

The V-JEPA auxiliary task learned, and J1 showed a small temporal-curve
regularization effect. However:

- final J1 video-flow validation loss was 31.6% worse than the
  parameter-matched video-only baseline;
- at four inference steps, J1 regressed video-latent and decoded-RGB error;
- turning generated V-JEPA fusion off, or shuffling it between samples, did not
  reduce quality;
- fresh validation selected VPM at NFE 1 as the only conservative baseline
  frontier point and found no eligible J1 reduced-step candidate.

This rejects the exact experiment in which **all four full-clip,
future-dependent V-JEPA bins are generated from noise and jointly denoised
with the video latent**. It does not reject history-only V-JEPA conditioning
or an action-conditioned V-JEPA future predictor.

## Controlled question

Does a frozen V-JEPA 2.1 representation, used as a second jointly diffused
state, make a Wan/LACWM video generator:

1. reach lower video error in fewer training updates; and
2. preserve or improve held-out quality with fewer deployable Wan calls?

The quantitative baseline is VPM, not V0. VPM has the auxiliary-branch
parameters needed to match A1/J0/J1, but its auxiliary loss and video fusion
are disabled.

| Arm | Intervention |
|---|---|
| V0 | Original explicit-action architecture; provenance control only |
| VPM | Parameter-matched video-only quantitative baseline |
| A1 | V-JEPA auxiliary denoising loss; fusion into video disabled |
| J0 | Joint video/V-JEPA denoising with aligned noise clocks |
| J1 | Joint denoising with V-JEPA leading video by one logit |

For J0, `sigma_aux = sigma_video`. For J1:

```text
sigma_aux = sigmoid(logit(sigma_video) - 1)
```

LACWM uses `sigma=1` for noise and `sigma=0` for clean state, so the J1
auxiliary state is cleaner than the video state at the same joint step.

## Representation and deployability

Each example contains:

- RGB: `[13, 3, 180, 960]`;
- five observed history frames and eight future frames to generate;
- frozen V-JEPA target: `[64, 4, 24, 120]`.

The offline target extractor:

1. maps the 13-frame, three-view clip to V-JEPA's 16-frame input;
2. extracts dense 768-dimensional ViT-B tokens;
3. pools them into four temporal bins and a `24 x 40` spatial grid per view;
4. applies train-split-only PCA/whitening to 64 channels; and
5. stacks the three views along width, producing `[64, 4, 24, 120]`.

The frozen V-JEPA encoder is a training-target extractor only. It is never
called by the optimizer or deployable sampler. At inference:

- exactly five history RGB frames and actions are observable;
- the future video latent starts from Gaussian noise;
- all four full-clip V-JEPA bins start from Gaussian noise because the
  bidirectional embeddings depend on future frames;
- there is no future RGB, clean V-JEPA target, or online teacher call.

The core implementation is in:

- `robot_wm/modeling/dual_diffusion/vjepa2_target.py`;
- `projects/latent_action_models/lam/dual_explicit_action_dit_model.py`;
- `robot_wm/modeling/networks/wan_forward_model.py`;
- `projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/vjepa_{vpm,a1,j0,j1}.yaml`.

## Protocol

- warm-started Wan/LACWM model;
- seed 1234;
- 1,000 optimizer updates, global batch 8;
- 512 train, 64 validation, and 128 inspected-test clips;
- NFE grid: `1, 2, 4, 6, 8, 12, 20`;
- paired 10,000-bootstrap confidence intervals;
- reconstruction metrics only;
- private W&B project `zijiandu/dual-video-diffusion-private`, no group.

The authenticated W&B viewer was `zijiandu@asu.edu`, not the requested
`ldu@nvidia.edu`; the study manifest records that identity mismatch explicitly.

## Training observation

Final validation video-flow telemetry at update 1,000:

| Arm | Video-flow loss | Relative to VPM |
|---|---:|---:|
| VPM | 0.124243 | — |
| A1 | 0.163582 | 31.7% worse |
| J0 | 0.163788 | 31.8% worse |
| J1 | 0.163548 | 31.6% worse |

J1 learned a cleaner auxiliary predictor than A1/J0, but its learned coupling
to the video backbone remained small:

| Coupling diagnostic | J1 |
|---|---:|
| State-to-native activation ratio | 0.7526% |
| Total auxiliary-to-native ratio | 1.7681% |
| Learned state gate | -0.001626 |
| Learned clock gate | -0.002361 |

Observation: the auxiliary branch learned. The evidence does not show that the
video generator materially used the generated sample-aligned auxiliary state.

## Inspected-test quality

At update 1,000 and autonomous NFE 4:

| Arm | Video NMSE | Decoded MSE | Temporal MSE |
|---|---:|---:|---:|
| VPM | 0.300855 | 0.0244325 | 0.0167427 |
| J1 | 0.311760 | 0.0250796 | 0.0166098 |

Paired J1-versus-VPM effects:

| Metric | Relative improvement | 95% CI | Gate |
|---|---:|---:|---:|
| Temporal MSE | +0.794% | [-0.200%, +1.820%] | Fail |
| Video NMSE | -3.625% | [-5.151%, -2.120%] | Fail |
| Decoded MSE | -2.649% | [-4.392%, -0.911%] | Fail |

The temporal training-curve AUC improved by 1.969%, with 95% CI
`[+1.427%, +2.553%]`. This is a narrow temporal-regularization result. It is
not overall faster learning because the final video and RGB guardrails
regressed.

## Causal use of generated V-JEPA state

At J1 NFE 4:

| Contrast | Video | Decoded | Temporal |
|---|---:|---:|---:|
| Autonomous vs fusion off | +0.0711% | -0.0223% | -0.0843% |
| Autonomous vs shuffled generated state | -0.0075% | -0.0155% | -0.0094% |

The autonomous-versus-off temporal CI is entirely negative:
`[-0.1320%, -0.0385%]`. Autonomous-versus-shuffled intervals cross zero.
Oracle-matched and oracle-shuffled effects are also nearly identical.

Therefore, the generated sample-aligned V-JEPA state did not causally improve
deployable inference in this design. Any same-NFE temporal benefit at larger
NFE is consistent with a package-level or multitask effect, rather than a
benefit from the sample-aligned generated state.

## Fresh validation frontier

The original J1@4-versus-VPM@8 comparison was not a sound acceleration test
because VPM@8 was not established as the quality-matched baseline. A fresh,
validation-only frontier evaluated all 64 pinned validation clips:

| NFE | VPM video | J1 video | VPM decoded | J1 decoded | VPM temporal | J1 temporal |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.240274 | 0.256344 | 0.0191015 | 0.0199946 | 0.0132795 | 0.0134748 |
| 2 | 0.240546 | 0.253188 | 0.0190951 | 0.0198737 | 0.0132848 | 0.0133751 |
| 4 | 0.299599 | 0.304895 | 0.0235729 | 0.0238977 | 0.0154756 | 0.0153257 |
| 8 | 0.348931 | 0.345745 | 0.0269056 | 0.0266078 | 0.0181446 | 0.0171930 |
| 20 | 0.384714 | 0.375308 | 0.0291431 | 0.0281955 | 0.0201011 | 0.0182517 |

Authoritative selection fields:

```text
vpm_non_dominated_nfe_frontier = [1]
candidate_count                 = 0
selected_pair                   = null
confirmatory_eligible           = false
```

No new 128-clip lockbox scoring was launched. J1's higher-NFE temporal
improvements do not demonstrate acceleration because both arms at those NFEs
are worse than the lower-compute VPM@1 point.

## Latency

The preregistered but scientifically diagnostic comparison was measured in one
process with both checkpoints resident on the same B200, 20 warmup pairs and
100 counterbalanced timed pairs:

| Endpoint | Mean | p50 | p95 | Complete 8-frame rollouts/s at p95 |
|---|---:|---:|---:|---:|
| J1@4 | 405.91 ms | 394.27 ms | 459.22 ms | 2.18 |
| VPM@8 | 645.15 ms | 618.79 ms | 750.65 ms | 1.33 |

The paired mean speed effect is +37.083%, with stratified 95% CI
`[+36.428%, +37.755%]`, and both execution-order strata are favorable.
This proves that four Wan calls are faster than eight; it does not prove that
V-JEPA enabled a quality-preserving call reduction.

The relevant low-compute baseline is VPM@1:

```text
p95 latency             = 223.81 ms
complete rollouts/s     = 4.47
generated frames/s      = 35.74
```

Generated frames per second is not a DAgger control-decision rate. A complete
eight-frame rollout every 223.81 ms is 4.47 decisions/s before policy,
environment, and communication overhead, so this experiment does not
demonstrate a 5–10 Hz real-time DAgger loop.

## Preregistered outcome

| Evidence group | Result |
|---|---:|
| Training gates | Fail |
| Deployable-inference gates | Fail |
| Generated-state mechanism gate | Fail |
| Joint faster-and-higher-quality claim | **Not demonstrated** |

The I4 and I6 gates pass for J1@4 versus VPM@8, but the fresh validation
frontier shows that VPM@8 is an inflated-compute reference. Those passes are
diagnostic and cannot support the central acceleration claim.

## Limitations

- One seed and 1,000 warm-started updates; this is neither from-scratch
  training nor full convergence.
- The 128 original test clips became inspected/exploratory. The fresh
  validation gate rejected every candidate, so the untouched lockbox remained
  unscored.
- Metrics are latent and raw-RGB reconstruction only. There is no FVD, LPIPS,
  diversity, rollout-stability, or downstream DAgger-success claim.
- Higher NFE generally worsened VPM quality; under the preregistered
  conservative rule, VPM@1 was the sole frontier point. A solver/training
  mismatch is a hypothesis, not an established cause.
- The conclusion applies to generated, future-dependent full-clip V-JEPA
  co-denoising, not to every possible V-JEPA-conditioned video model.

## Recommended next experiment

Do not spend another long run on the same full-future joint-noise design.
Instead, preflight one of these inference-available representations:

1. frozen V-JEPA tokens computed from the five observed history frames only;
2. a small action-conditioned predictor of future V-JEPA tokens, trained
   separately and frozen for the video-generator ablation.

For either option, first require autonomous to beat both fusion-off and
sample-shuffled controls at the same NFE on validation. Only after that causal
gate passes should the experiment compare against the actual VPM NFE frontier
and open a fresh lockbox. Because VPM@1 was already the best point here,
one-step consistency/distillation is a more direct acceleration direction than
adding a second state that must itself be generated from noise.

## Reproducibility and evidence

- Training/scientific commit:
  `9cf8e6922f35a5d6645e3128545953723bf54da2`
- Final analyzer/recovery commit:
  `3ef18aeec70f441f923e818c6b0de9997d0b015d`
- Study:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3`
- Fresh frontier:
  `frontier_selection.json`,
  SHA-256 `90a641947a27402d5e32fa8580cd661344cc5e5b5ee7c918f45e4c140f2fabda`
- Paired timing:
  `paired-481133-43ed5d3-r1/paired_latency/paired_j1_nfe4_vs_vpm_nfe8.json`,
  SHA-256 `314a40090a96cd06c6e331f833b689aa06059ba1988cb83955ab9a2b9733d84c`
- Final analysis:
  `paired-analysis-481826-3ef18ae-r2/analysis/analysis.json`,
  SHA-256 `99522b758ab2ed576dd8883dd142864f569f014be30383785887ed15c20c442b`
- Human-readable analyzer output:
  `paired-analysis-481826-3ef18ae-r2/analysis/analysis.md`,
  SHA-256 `ff97fcd5b67a2e7974589d64ca9f9a6a3f8aaaa688296cd009bd31d0e8c70ef0`
- Analyzer-only recovery job: `481978`, `COMPLETED/0:0`, 1m49s.
