# Trajectory-consistent corrected-renderer result

Date completed: 2026-08-08

Decision: **advance the native raw-command geometry scaffold only; do not
advance recurrent delta or hybrid correction into Wan**

Machine decision: `GO_raw_geometry_scaffold_pass`

## Central finding

The fresh experiment separates two conclusions that must not be conflated.

First, the recurrent delta-response model fixed the motion defect observed in
Gate-0c. Its robot-flow EPE was 2.988 px, versus 5.664 for raw command and 5.045
for the current absolute ridge. These are 47.25% and 40.78% improvements,
respectively; every one of the 24 fresh clips favored recurrent delta, and both
Holm-adjusted lower bounds were strongly positive. Its silhouette point estimate
also improved over both references.

However, the **predeclared recurrent family does not pass**. Four spatial-
retention intervals cross zero: RGB edge alignment versus raw and absolute
ridge, plus silhouette and IoU versus absolute ridge. The recurrent RGB point
estimate is effectively tied with absolute ridge (7.155 versus 7.148 px), but a
tie with uncertainty is not the registered nonnegative-lower-bound result. The
strong flow result therefore cannot override the frozen all-endpoint rule.

Second, the independent native raw-geometry fallback passes every one of its
own gates. Native raw flow beats a same-stratum shuffled complete raw trajectory
by 75.63% and hold-current by 38.25%, with Holm-positive lower bounds, at least
75% favorable clips, and strongly positive silhouette/RGB/IoU controls. This is
direct prospective evidence that known planned robot motion is a sufficiently
sample-specific fixed causal scaffold to justify an equal-Wan-call conditioning
screen. It is not evidence that conditioning will improve generated video.

The transition-local hybrid is rejected. It improves flow over raw by only
1.60%, below the frozen 5% margin, and is 10.47% worse than absolute ridge. The
Gate-0c hypothesis that absolute anchoring and raw increments could simply be
spliced together is therefore not supported.

## Immutable prospective execution

- Source: commit `856cd553c051b90f5b8ddf3649103ae1090f4a4d`, tool
  SHA-256 `834fad1a3113bd4a34b046061203955c1d72664f0728f5c55f1314a2115c538d`.
- Fit: immutable ABC train rows 0--383 only.
- Score pool: train rows 384--511; validation and protected test forbidden.
- Registration rescanned 103 D405 episodes and hash-bound/excluded all 24
  Gate-0c clips, leaving exactly 79 untouched D405 episodes.
- Fresh selection: 24 previously unevaluated episodes, eight per planned-motion
  stratum, selected by the frozen SHA rank before score state or RGB access.
- Donors: next selected different episode within the same fresh motion stratum.
- Recurrent tuning: five-fold fit384-only CV over six ridge alphas and four
  gains; selected alpha 0.1 and gain 1.0. Three fixed self-rollout refits were
  used. Future measured state was target-only.
- Statistics: one common 20,000-sample paired bootstrap and one global Holm
  family over seven primary flow contrasts.
- Renderer: official ABC/YAM commit
  `6bc6586721cf0c409ccee80f675a28de9b9b2f5e`, nominal D405 top camera,
  native image resolution, articulated robot geometry only.

The sealed registration explicitly recorded `score_state_opened=false`,
`score_rgb_opened=false`, `validation_accessed=false`, and
`protected_test_accessed=false` before selection outcomes were opened.

## Absolute renderer metrics

Lower is better except IoU.

| Arm | Robot flow EPE (px) | Silhouette Chamfer (px) | RGB robot-band Chamfer (px) | Silhouette IoU |
|---|---:|---:|---:|---:|
| Raw command | 5.6640 | 2.7079 | 7.3781 | 0.8502 |
| Raw trajectory shuffled | 23.2427 | 24.6165 | 22.2664 | 0.2987 |
| Current absolute ridge | 5.0455 | 1.8804 | **7.1478** | 0.8868 |
| **Recurrent delta** | **2.9877** | **1.7108** | 7.1547 | **0.8954** |
| Recurrent delta shuffled | 22.5569 | 25.0431 | 22.4919 | 0.2978 |
| Hybrid anchor/raw delta | 5.5737 | 2.2512 | 7.2252 | 0.8712 |
| Hold current | 9.1724 | 6.7220 | 9.3896 | 0.7089 |
| Measured oracle diagnostic | 0.0000 | 0.0000 | 6.9026 | 1.0000 |

The oracle RGB-band value is nonzero because the observed edge band contains
texture, objects, and background while the renderer contains nominal robot
geometry. It is a diagnostic floor, not a perfect RGB segmentation target.

## Holm-corrected primary flow family

Positive differences favor the named candidate. The Holm lower bound is the
registered one-sided bootstrap quantile after step-down multiplicity control.

| Contrast | Relative gain | Difference (px) | Holm lower bound | Favorable clips | Frozen primary gate |
|---|---:|---:|---:|---:|---|
| Recurrent vs raw | +47.25% | +2.6763 | +1.6757 | 24/24 | pass |
| Recurrent vs absolute ridge | +40.78% | +2.0577 | +1.2223 | 24/24 | pass |
| Recurrent vs recurrent shuffled | +86.75% | +19.5691 | +15.8900 | 24/24 | pass |
| Hybrid vs raw | +1.60% | +0.0904 | +0.0308 | 18/24 | **fail: below 5%** |
| Hybrid vs absolute ridge | -10.47% | -0.5282 | -1.0669 | 11/24 | **fail** |
| Raw vs raw shuffled | +75.63% | +17.5786 | +13.9295 | 24/24 | pass |
| Raw vs hold | +38.25% | +3.5084 | +1.2220 | 18/24 | pass |

The first six contrasts had bootstrap p-values at or below `0.00030`; the
hybrid-versus-absolute contrast had p=0.966 and stopped the Holm sequence. The
hybrid-versus-raw contrast still fails its independent 5% effect-size rule even
though its adjusted lower bound is positive.

## Why recurrent delta did not receive a go decision

The recurrent point estimates are encouraging, but the protocol prohibited
trading spatial uncertainty for motion improvement.

| Required retention contrast | Favorable point difference | Paired 95% interval | Gate |
|---|---:|---:|---|
| Silhouette vs raw | +0.9971 px | `[+0.5022,+1.5269]` | pass |
| IoU vs raw | +0.0452 | `[+0.0249,+0.0660]` | pass |
| **RGB band vs raw** | **+0.2235 px** | **`[-0.0548,+0.5439]`** | **fail** |
| **Silhouette vs absolute ridge** | **+0.1695 px** | **`[-0.1007,+0.4529]`** | **fail** |
| **RGB band vs absolute ridge** | **-0.0068 px** | **`[-0.1027,+0.0936]`** | **fail** |
| **IoU vs absolute ridge** | **+0.0085** | **`[-0.0031,+0.0206]`** | **fail** |

This result is not evidence that recurrent delta damages spatial alignment.
Its point estimates improve silhouette and IoU, and RGB is essentially tied
with the current ridge. It is evidence that the registered 24-clip experiment
did not establish nonnegative lower bounds for every required spatial endpoint.
The correct claim is therefore “large confirmed flow gain, incomplete joint
flow-plus-spatial confirmation,” not “recurrent correction failed to help.”

## Raw scaffold attribution

Raw command passed both motion controls and every spatial control:

| Contrast | Metric | Favorable difference | Paired 95% interval | Favorable clips |
|---|---|---:|---:|---:|
| Raw vs shuffled | Flow EPE | +17.5786 px | `[+14.5977,+20.7029]` | 24/24 |
| Raw vs shuffled | Silhouette | +21.9086 px | `[+17.7016,+26.4784]` | 24/24 |
| Raw vs shuffled | RGB band | +14.8882 px | `[+9.9142,+20.5429]` | 24/24 |
| Raw vs shuffled | IoU | +0.5515 | `[+0.4987,+0.6045]` | 24/24 |
| Raw vs hold | Flow EPE | +3.5084 px | `[+1.3815,+5.9627]` | 18/24 |
| Raw vs hold | Silhouette | +4.0141 px | `[+2.4905,+5.6157]` | 20/24 |
| Raw vs hold | RGB band | +2.0115 px | `[+0.9457,+3.1979]` | 17/24 |
| Raw vs hold | IoU | +0.1413 | `[+0.0886,+0.1949]` | 20/24 |

The shuffled effect is intentionally large because a donor supplies a complete
future raw trajectory. The hold contrast is the more conservative question:
does planned motion add useful future geometry beyond the currently observed
pose? It passes on flow and all spatial metrics.

## State-space and horizon diagnostics

The fresh target-only state diagnostic anticipated the projection result:

| Arm | Target-pose MSE | Joint-transition MSE |
|---|---:|---:|
| Raw | 0.004179 | 0.002730 |
| Absolute ridge | 0.002068 | 0.002240 |
| **Recurrent delta** | **0.001679** | **0.000275** |
| Hybrid | 0.003022 | 0.002730 |
| Hold | 0.039760 | 0.003486 |

Recurrent flow remained better than both raw and absolute ridge at every one of
the eight transitions. Its flow EPE by transition was
`[4.353,2.647,3.239,2.866,3.521,3.634,3.178,3.520]` px. Raw was
`[8.251,6.317,5.382,4.578,6.692,7.588,5.917,6.822]`; absolute ridge was
`[4.678,4.224,5.240,5.026,5.887,5.895,6.949,8.569]`. Thus the flow result is
not driven by one favorable horizon—the recurrent formulation fixes the late-
horizon degradation seen in Gate-0c.

Across low/medium/high command-motion strata, recurrent flow EPE was
`[1.502,2.733,4.729]` px versus raw `[2.560,5.890,8.542]` and absolute ridge
`[2.547,5.524,7.066]`. The gain is not confined to low-motion clips.

## Runtime and audit

Deduplicated native-resolution MuJoCo pose rendering measured 3.592 ms mean,
3.309 ms median, and 4.598 ms p95 over 1,303 post-warmup poses. This excludes
Wan and decoding.

The frozen read-only audit passed locally and directly on canonical Lustre. It
verified/recomputed:

- 113 artifact hashes and 24 hash-addressed bundles;
- 1,536 frame/transition rows, 192 clip rows, and 408 provenance rows;
- all 25 paired effects and all seven Holm primary tests;
- fresh-score exclusion and within-stratum different-episode donors;
- per-clip moving-flow support and source reprojection;
- causal recurrent replay with maximum absolute error below `4.2e-7` locally
  and exactly zero on canonical Lustre;
- 2,286 explicit protected-test false flags.

Artifact identities:

- registration: `9cc556aba53d1defb69b0049dab67d12a3991decb917015bba5153c16cb8c2b1`;
- preparation: `11444ee94ae6707869f44f00409386701c93b886d650ebc928bc4edec4f41a3a`;
- analysis: `1471d0bb1f40aabc44d08c71b7eeeba3f0e88b9cb2a07e56dc6f7eb8b11034a0`;
- completion: `715fc288ca3e5526cda3a5bb3329db650aa8ed15a61d2959be2349dd3df6dcfc`.

Canonical artifact:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/trajectory_consistent_renderer_gate/
  trajectory-renderer-train384-fresh24-20260808-856cd55-v1/
```

Local mirror:

```text
/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/
  trajectory_consistent_renderer_gate/
  trajectory-renderer-train384-fresh24-20260808-856cd55-v1/
```

## Evidence-based next step

Run the independently authorized **raw-only** Stage-1 conditioning screen. Use
equal Wan calls, one frozen parent checkpoint/data order/noise schedule, and the
same registered clips for these arms:

- `RAW-FLOW`: native raw-command robot geometry/flow;
- `FLOW-OFF`: exact masked no-op;
- `RAW-SHUFFLED`: same-stratum donor raw geometry;
- `RAW-TIME`: nonwrapping wrong-time raw geometry;
- `RAW-HOLD`: current-pose/no-motion geometry.

The recurrent and hybrid arms are excluded from that confirmatory screen. The
raw Stage-1 protocol must be committed and registered before generated-video
outcomes, with equal total Wan calls and a guarded launcher. Passing the
renderer gate does not predetermine the video result.

Recurrent delta remains a scientifically useful follow-up because its motion
gain is large and horizon-consistent. It should be revisited only in a separate
larger spatial-equivalence confirmation or after the raw Stage-1 mechanism is
understood; it must not be silently inserted into the raw-only screen.

## Limitations

- Twenty-four deliberately motion-stratified train-only episodes and one seed.
- No generated RGB, perceptual metric, FVD, task success, validation, or
  protected test.
- Raw endpoints are not controller/dynamics replay.
- Nominal top-camera calibration and robot-only rendering.
- The hybrid is transition-local rather than a coherent trajectory.
- Flow is deterministic robot geometry, not object/contact/visibility dynamics.
- A raw renderer pass authorizes experimentation, not a video-quality claim.
