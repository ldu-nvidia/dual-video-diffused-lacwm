# Causal-compressibility ladder result

Date: 2026-08-08

Decision: **`STOP_PRIVILEGED_NOT_CAUSALLY_COMPRESSIBLE`**

## Executive conclusion

The clean-video V-JEPA teacher residual contains a large, sample-specific
signal that can be predicted from the feature-free J1 shared trunk. The locked
linear `PFD_ALIGNED` head improved one-step J1-off velocity MSE by 83.85%,
latent NMSE by 84.10%, raw decoded MSE by 58.30%, and raw temporal MSE by
47.68%. It also beat the episode-shuffled teacher head by 49--60% on all four
metrics, with every paired lower confidence bound positive.

That is not the required privileged-transfer result. An equal-capacity head
trained directly on the ordinary flow residual was significantly better than
the aligned privileged head on every primary metric. Fresh VPM@1 was then
substantially better than both. The supported interpretation is therefore:

1. J1's inference-visible state already contains enough information to repair
   much of J1's damaged one-step endpoint;
2. clean V-JEPA supplies a structured but lossy training target relative to
   the ordinary clean-video flow target that training already owns; and
3. neither correction establishes an improvement over the actual video-only
   frontier.

The clean-future-feature distillation direction should stop at this initial
high-noise J1 seam. It is not justified to spend a full nonlinear continuation
trying to reproduce a target that is dominated by direct supervision.

## Frozen run

- Slurm job: `507229`, `COMPLETED`, exit `0:0`, elapsed `00:10:36`
- node: `pool0-0036`, one B200
- source commit: `4d4db76cb905b8f10630038938ade240fdf4302c`
- training lineage commit: `656086686dae723c942a4209a9d71cdb17ed6ccc`
- J1 update-1,000 snapshot SHA-256:
  `70d79533460680d836c7178baea2232cd4b8f309146a7571db0002f91ce53f61`
- VPM update-1,000 snapshot SHA-256:
  `de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a`
- artifact:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/causal_compressibility_ladder/causal-compressibility-train256-dev64-seed20260820-4d4db76-v1`

The optimization partition was canonical train indices 128--383 with noise
seeds 20260820/20260821. Direct-only numerical selection used train indices
384--415 with seeds 20260822/20260823. Outcomes used 64 episode-disjoint train
indices 416--479 with four unseen seeds 20260824--20260827. Indices 0--127
were excluded due to previous teacher use; train reserve 480--511, validation,
and protected test were not opened.

The fit used 131,072 optimization token rows. DIRECT-only calibration selected
the 1,024-channel rung and ridge penalty `0.01`, giving exactly 65,600 affine
patch-head parameters per arm. Its calibration corrected-velocity MSE was
0.220817, versus 0.369830 at the best 256-channel candidate and 1.109627 at
the best 64-channel candidate.

## Feature-free development outcomes

All means below cover 64 episodes x four unseen Gaussian noise seeds. Primary
decoded targets are cached raw held-out RGB quantized to uint8. Temporal MSE
includes the last-observed-history to first-future boundary.

| Endpoint | Velocity MSE | Residual R2 | Residual cosine | Latent NMSE | Raw decoded MSE | Raw temporal MSE |
|---|---:|---:|---:|---:|---:|---:|
| J1-OFF / ZERO | 1.719084 | 0.0000 | 0.0000 | 2.792255 | 0.0702530 | 0.0492802 |
| PFD-SHUFFLED | 0.699738 | 0.5951 | 0.8341 | 1.129281 | 0.0615423 | 0.0510008 |
| PFD-ALIGNED | 0.277599 | 0.8415 | 0.9278 | 0.444007 | 0.0292969 | 0.0257848 |
| DIRECT | **0.211921** | **0.8785** | **0.9389** | **0.339340** | **0.0230908** | **0.0176271** |
| fresh VPM@1 | **0.124175** | n/a | n/a | **0.196498** | **0.0155103** | **0.0123595** |

VPM's residual R2/cosine fields are zero by construction because no probe is
attached; they are not comparisons of VPM representation quality.

### Paired clustered effects

Positive favors PFD-ALIGNED. Confidence intervals are the preregistered 10,000
episode-clustered bootstrap; all four noise draws remain within an episode.

| Comparison | Velocity | Latent | Decoded | Temporal |
|---|---:|---:|---:|---:|
| aligned vs J1-off | +83.85% [83.11, 84.60] | +84.10% [83.41, 84.77] | +58.30% [55.24, 61.39] | +47.68% [42.68, 52.61] |
| aligned vs shuffled | +60.33% [58.74, 61.88] | +60.68% [59.21, 62.12] | +52.40% [49.03, 55.79] | +49.44% [44.91, 54.02] |
| aligned vs DIRECT | **-30.99% [-33.57, -28.45]** | **-30.84% [-33.56, -28.24]** | **-26.88% [-30.22, -23.99]** | **-46.28% [-52.90, -40.70]** |
| aligned vs VPM@1 | **-123.55% [-135.38, -112.81]** | **-125.96% [-137.44, -115.28]** | **-88.89% [-101.88, -77.32]** | **-108.62% [-128.39, -93.08]** |

DIRECT itself improved over J1-off by 87.67% velocity, 87.85% latent, 67.13%
decoded, and 64.23% temporal MSE, all with strictly positive confidence
bounds. Nevertheless, relative to DIRECT, fresh VPM@1 was better by 41.40%
velocity, 42.09% latent, 32.83% decoded, and 29.88% temporal MSE. Restoring a
weak J1 parent is not progress over the frontier.

## Serving and integrity audit

- J1 endpoint rows: 1,280 = 64 clips x 4 seeds x 5 endpoints
- VPM rows: 256 = 64 clips x 4 seeds
- J1 and VPM: 128 Wan invocations / 256 sample calls each
- development teacher calls: exactly zero
- development V-JEPA target-array opens: exactly zero
- NumPy input graph in both endpoint processes: exactly the pinned RGB and
  action arrays
- J1-OFF and ZERO: bit-exact on all 256 final latents and metric records
- initial Gaussian noise, raw future target, and raw history-boundary hashes:
  identical across all six endpoints for every `(clip, seed)`
- adapter-only PFD-ALIGNED latency per batch: mean 0.380 ms, p95 0.418 ms
- row inventory, paired noise, and zero no-op were recomputed by the final
  audit

Artifact identities:

- registration: `3517eb729ee1b47576ca6a91159121f24fda2b73b7127dcb468628807356b36a`
- fit: `9bc62b9a67bacd2618d6c162a7a9573611a44991959d9dabb2a3ef0b9914354a`
- J1 endpoint: `c4ace726681e09fb5efefd219e4f02ec0534fbb9efa4fff400a6719eddf6ec9f`
- VPM endpoint: `103259c642d1d30ee08c5f94aaa94db66d174cc46360a1ed2f48e9de0e9addcc`
- analysis: `298ee8e64722380062efc25ba18d7f3624b6fb063f17738cd2e70efe832317c4`
- completion: `f333f6d639cfa7473af59d750e042271cb24fc2d00b0936e2018de3c918d106a`
- audit: `5ec80c660e2adfc2fba061667e981d481bb1c1e551916f9623482a46071a7f75`

All nine artifact digests recorded by `audit.json` were independently
recomputed after job completion and matched.

## Meaning for the research program

This experiment resolves the most immediate ambiguity in the high-noise
teacher result. The oracle V-JEPA correction was not merely unlearnable noise:
its high R2 and aligned-over-shuffled advantage show substantial causal
structure. But it is the wrong optimization target when clean video already
provides the exact flow residual. The semantic teacher discards information
needed by the pixel/video endpoint and does not regularize enough to offset
that loss.

The next dual-video experiment should therefore obey three constraints:

1. start from the actual VPM frontier, not J1;
2. include an equal-capacity DIRECT residual arm in every attribution table;
3. admit an auxiliary only if it improves over both DIRECT and fresh VPM under
   feature-free serving.

The evidence favors moving compute to causally available structure that adds
information the base state does not already supervise directly: calibrated
robot rendering plus a compact stochastic object/contact interaction state,
or a genuinely action-predictive V-JEPA2-AC representation. In either case,
the auxiliary must be tested against VPM and a matched direct-residual control.

## Claim boundary

This is one frozen J1/VPM parent pair, a linear token-local head, and train-only
development. It does not rule out every nonlinear privileged representation.
It does show that a full clean-V-JEPA continuation at this seam is not
evidence-based, because the privileged target is already dominated by direct
supervision and both are dominated by VPM@1. No FVD, protected test,
multi-checkpoint replication, long-horizon rollout, or closed-loop DAgger
claim is made.
