# Recurrent-delta spatial confirmation result

Date: 2026-08-09

Status: **`GO_RECURRENT_DELTA_WAN_SCREEN` at the robot-renderer endpoint only**

## Major finding

The frozen recurrent delta-response predictor is the first inference-causal
auxiliary in this research program to clear its complete prerequisite gate. On
the full 55-episode renderer-endpoint-unopened D405 census, it reduced mean
robot-flow EPE from `4.628886` to `2.690298` pixels versus raw planned commands
(`+41.880%`) and from `4.120211` to `2.690298` versus the frozen absolute-ridge
predictor (`+34.705%`). It also beat its episode-shuffled-command control by
`88.274%`. All three superiority tests and all six preregistered spatial
non-inferiority tests passed their separate Holm-controlled families.

This is evidence that a closed-loop action-conditioned robot-motion scaffold is
substantially more accurate than the fixed raw-command field previously sent to
Wan. It is **not** evidence that recurrent conditioning improves generated
video. No Wan call, generated RGB endpoint, validation split, protected test,
FVD, or policy rollout was used in this confirmation.

## Frozen execution

- Successful Slurm job: `508064`, `COMPLETED`, exit `0:0`, elapsed `2m24s`, one
  B200, 32 CPUs, 192 GiB requested; maximum RSS was approximately 8.38 GiB.
- Source commit: `92cc9584298d2a395f3a79152768c9d0b88b3864`.
- Population: all 55 eligible train-only D405 episodes, with frozen motion
  strata `19/18/18`; no outcome-based selection or exclusion.
- Predictor: the exact fit384 recurrent model, alpha `0.1`, gain `1.0`, copied
  without refitting from the prior fresh24 study.
- Statistical contract: one common 100,000-draw, motion-stratified paired
  episode bootstrap, seed `20260809`; separate Holm families of three flow
  superiority and six spatial non-inferiority hypotheses.
- Flow gates additionally required at least 5% improvement versus raw and
  ridge, a positive multiplicity-adjusted lower bound, and at least 33/55
  favorable episodes.
- Spatial margins were `-0.25` native pixels for both Chamfer metrics and
  `-0.005` absolute IoU.

## Absolute endpoint means

Lower is better except IoU.

| Arm | Flow EPE (px) | Silhouette Chamfer (px) | RGB-band Chamfer (px) | Silhouette IoU |
|---|---:|---:|---:|---:|
| raw command | 4.628886 | 2.486389 | 8.477115 | 0.854634 |
| absolute ridge | 4.120211 | 1.506476 | **8.025143** | 0.900289 |
| **recurrent delta** | **2.690298** | **1.464827** | 8.038219 | **0.901731** |
| recurrent, shuffled commands | 22.943593 | 29.272493 | 26.593334 | 0.242289 |
| measured oracle | 0 | 0 | 7.979369 | 1 |

The recurrent arm is slightly worse than absolute ridge on RGB-band Chamfer by
`0.013076` pixel (`0.163%`), but this is well inside the prospectively frozen
quarter-pixel non-inferiority margin. The candidate is better on the other
three means.

## Flow superiority family

The paired effect is `reference - recurrent`, so positive values favor the
candidate.

| Comparison | Mean effect (px) | Relative gain | Holm one-sided lower bound | Favorable episodes | Gate |
|---|---:|---:|---:|---:|---|
| raw - recurrent | +1.938588 | +41.880% | +1.568414 | 52/55 | pass |
| absolute ridge - recurrent | +1.429912 | +34.705% | +1.119935 | 53/55 | pass |
| shuffled - recurrent | +20.253294 | +88.274% | +18.752062 | 55/55 | pass |

The very large shuffled-control failure is important attribution evidence: the
predictor is using the recipient-aligned future command sequence rather than
acting as a generic smoothing prior.

## Spatial non-inferiority family

Positive effects favor recurrent. A lower bound above `-0.25` pixel or
`-0.005` IoU passes.

| Reference | Endpoint | Mean effect | Holm lower bound | Margin | Gate |
|---|---|---:|---:|---:|---|
| raw | silhouette Chamfer | +1.021562 px | +0.757131 | -0.25 px | pass |
| raw | RGB-band Chamfer | +0.438896 px | +0.277376 | -0.25 px | pass |
| raw | silhouette IoU | +0.047098 | +0.036175 | -0.005 | pass |
| absolute ridge | silhouette Chamfer | +0.041649 px | -0.083437 | -0.25 px | pass |
| absolute ridge | RGB-band Chamfer | -0.013076 px | -0.105089 | -0.25 px | pass |
| absolute ridge | silhouette IoU | +0.001443 | **-0.002832** | -0.005 | pass |

The ridge-IoU comparison is the tightest gate. It passes the registered
stratified Holm procedure and a post-hoc unstratified nonparametric Bonferroni
bootstrap (`LB=-0.004702`), but a parametric t/Bonferroni sensitivity gives
`LB=-0.005151`, narrowly outside the margin. Therefore the machine GO is valid
under the frozen analysis, while equivalence to ridge is not robust enough to
support a broad renderer claim. This sensitivity is why the result authorizes
only a controlled Wan screen with complete controls.

## Horizon behavior

Pooled flow EPE by future transition was:

| Arm | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 7.531 | 4.986 | 4.660 | 5.813 | 4.428 | 5.216 | 5.212 | 4.835 |
| absolute ridge | **3.719** | 3.883 | 4.311 | 5.173 | 4.800 | 5.417 | 5.152 | 5.731 |
| recurrent | 4.066 | **2.902** | **3.350** | **3.173** | **2.873** | **3.382** | **2.892** | **2.355** |
| shuffled | 90.046 | 21.089 | 14.817 | 14.382 | 14.303 | 15.347 | 12.323 | 10.987 |

The recurrent rollout is slightly worse than absolute ridge on only the first
transition and substantially better on transitions two through eight. That is
the expected signature of the recurrent correction: its benefit accumulates
after the first predicted state rather than merely improving the initial
anchor.

## Causality and freshness boundary

The candidate trajectories were materialized and hash-closed before future
measured boundaries or future RGB were used for scoring. The successful rerun
retained exactly the failed run's selection, donors, causal trajectories, and
provenance hashes:

- selection: `d4a341df...`;
- causal trajectories: `8d54a7ec2cb2d25ec57c1b8a404f6a9d6988e2d80a308745d154a6fc11511e45`;
- causal provenance: `20ea3d160646...`.

The canonical metadata-only freshness audit established that all 55 episodes
were unopened for this renderer endpoint, but all 55 had appeared in unrelated
later experiment populations. Consequently this is a complete
renderer-endpoint census/replication, not a globally untouched confirmation or
dataset-level generalization result. Future measured-state arrays are stored in
indivisible NPZ members, so the enforced claim is target-index blindness by
source and causal closure—not an operating-system claim that no future byte was
decompressed.

## Preserved failed run and decoder-only repair

Job `508051` from source `065c159` failed after 17 seconds while OpenCV randomly
sought to frame 2277 of the first MP4. No RGB frame, score bundle, metric row,
analysis, or decision was produced. Sequential decoding independently reached
all 2,934 frames, showing a decoder seek failure rather than missing data.

Commit `92cc958` validates the requested landing position and falls back to
sequential frame-zero decoding when random seeking lies or fails. The exact
failed frame set then decoded as `[9,480,640,3]`; 22 focused tests passed on the
cluster. The failed root remains immutable. Because the successful rerun used
the identical frozen population, target indices, trajectories, controls,
metrics, and gates, this was a decoder-only rerun rather than an outcome-driven
repair.

## Integrity receipts

- registration identity: `db1c29e5a89c35b8b1b8c9a4ba8d1be2ef796c3793961fcd4204db46d363d42d`;
- causal-closure identity: `0b8f50bf3ac84e9932870b3c7cf430618bc49006f1306cf869328a5f65ebdcfc`;
- evaluation identity: `d0f09573e9f91f84c86007730f39a0fd348d6bd6dc723859de9bdcd329525483`;
- analysis identity: `2426a1546a43403fc09f7d8bb60475526e4d342586727c880dd005aad99010f8`;
- completion identity: `a94afb1e39ccee4b80298182f287431409fbce47ae260c94d4324fa23081e5ff`;
- built-in audit output SHA-256: `1398f3353f5b4c041abb93a85fd67efa76c869c211e888b8499995b38753c72a`.

The built-in audit passed 168 artifact hashes, the 2,200 transition rows, 275
episode/arm rows, causal replay, bootstrap, identities, and protected-data false
flags. An additional strict read-only reconstruction independently rehashed the
registered inputs and caches, reconstructed all causal inputs from observed
prefixes, reproduced every trajectory/control with maximum error zero, checked
row/index uniqueness, and regenerated the exact 100,000-draw bootstrap. A
separately sealed strict-audit receipt is required before this package is used
as the lineage root for a Wan run.

Canonical cluster root:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/
  artifacts/dual_video_diffusion/recurrent_delta_spatial_confirmation/
  recurrent-delta-spatial-confirmation-renderer55-seed20260809-92cc958-v1
```

Small evidence mirror:

```text
/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/
  recurrent_delta_spatial_confirmation/renderer55_92cc958
```

## Next registered question

The next experiment must ask a narrower question than “does dual diffusion
work?”: **does the already trained RAW-FLOW adapter produce better low-NFE video
when its inference-causal field is replaced by this materially more accurate
recurrent field?**

Use one frozen Wan checkpoint, identical initial noise and sampler calls, and
the exact same video set for `FLOW-OFF`, recurrent-aligned, recurrent
episode-shuffled, held-current, and nonwrapping wrong-time controls. Start with
the existing checkpoint as a zero-training substitution screen; only a clearly
attributed gain warrants matched recurrent-field training. The primary endpoint
remains NFE-1 decoded and temporal MSE versus unmodified VPM@1, with NFE 2/4,
LPIPS, latent NMSE, action sensitivity, complete serving latency, and exact Wan
call parity as guardrails. A positive aligned effect must disappear under the
causal controls. The renderer result by itself must never be reported as a
video-quality improvement.
