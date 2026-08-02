# Video Latent Forcing Phase-1 result

Date: 2026-08-01

## Decision

The preregistered low-resolution-RGB scratchpad gate **failed** at every
selectable auxiliary NFE. The protocol therefore stops this representation
before full B0/A1/L1 video training and advances only the independently
eligible prefix-causal semantic screen.

This is not a general failure of video Latent Forcing. It is a negative result
for an invertible `48 x 8 x 8 x 14` coarse-RGB future target, trained from
scratch for 5,000 updates. Phase 1 ran no video denoising calls, so it cannot
establish either a gain or a regression in full-resolution video generation.

## Execution and evidence

- Source: `6d35fbda76b3fdac6c04687ea95831d664c22231`
- Data: 64,000 train clips and 890 episode-disjoint validation clips from the
  frozen one-camera DROID split; protected test was not accessed
- Training: exact 200-update calibration and 5,000-update run, zero nonfinite
  updates, global batch 256 on eight B200s
- Checkpoint SHA-256:
  `66efe5ec0b80d5de92655741c7cb2546e9c4879834c4ef12d3bfb8d8238851f1`
- Evaluation Slurm job: `486628`, `COMPLETED 0:0` on `pool0-0201`
- W&B: `zijiandu/dual-video-diffusion-private`, run `vzt122rm`, no group
- Evaluation: all 31,150 paired clip/control/NFE rows, 35 R3D-18 feature cells,
  and 112 sample records completed
- Per-clip table SHA-256:
  `628275055994c62ddbf802280ed8be1e948bfde7b215b6fe1058d13f6f71f075`
- Summary SHA-256:
  `3b7ea34cb9681e375c978f7dd71d6e0d453ad10123f61d321f7e35a29ea34805`
- Gate SHA-256:
  `33e1ab99092caf2a2820936082f3aa848b0b2406c8e92a20ceac03489a64269d`

The evaluation recovery changed no scientific source or checkpoint. It added
a pinned job-local ffmpeg executable after the first evaluation attempt failed
before producing metrics. The successful job revalidated the immutable
training handoff, all eight GPU mappings, NCCL barrier/all-reduce, and the real
H.264 production writer before evaluating.

## Frozen validation result

| Aux NFE | Aux NMSE | Aux cosine | Frame-LPIPS gain vs off | Temporal-MSE gain vs off | Temporal-MSE gain vs shuffled | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.0693 | 0.9480 | +92.62% | -13.10% | +0.23% | fail |
| 2 | 0.0733 | 0.9468 | +93.41% | -32.02% | +0.25% | fail |
| 4 | 0.0801 | 0.9432 | +93.08% | -69.70% | +0.24% | fail |
| 8 | 0.0873 | 0.9403 | +92.65% | -102.48% | +0.20% | fail |
| 12 | 0.0914 | 0.9390 | +92.39% | -118.62% | +0.19% | fail |

At one call, autonomous frame LPIPS was `0.06378`, versus `0.86423` for
scratchpad-off and `0.49062` for a donor-generated scratchpad. Yet its
temporal-difference MSE was `0.006502`, versus `0.005749` for off and
`0.006517` for shuffled. The paired improvement over shuffled was only 0.23%,
with 95% CI `[-0.31%, +0.79%]`; the required gain was at least 5% with a
strictly positive lower bound. Every NFE failed improvement over off,
improvement over shuffled, and retained oracle temporal utility.

Context shuffling strongly degraded feature reconstruction (NFE-1 NMSE
relative effect 95.64%; cosine advantage 0.7738), so the predictor did use the
correct history/actions. The narrow conclusion is therefore:

> The model recovered sample-aligned coarse appearance in very few calls, but
> its generated temporal changes were no more aligned with the recorded future
> than another clip's trajectory. Additional Euler calls amplified, rather
> than repaired, that temporal error.

R3D18-Frechet was diagnostic and is not FVD. Oracle-clean was a nondeployable
ceiling and could not override the gate.

## Authorized next experiment

Use the prefix-causal V-JEPA 2.1 target defined in
`VIDEO_LATENT_FORCING_POC_PROTOCOL.md`. It discards pixel-level appearance
entropy and gives every future slot a semantic feature whose teacher support
ends at that slot. Train and gate the autonomous semantic predictor before any
full video-fusion run. Distillation and solver tuning remain blocked until a
generated-only representation passes both predictability and attribution.
