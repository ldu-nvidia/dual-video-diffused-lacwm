# V-JEPA 2-AC native-DROID Stage-0 result

Date: 2026-08-08

Status: 32-episode RTX Blackwell qualification and preregistered 32-episode
B200 hardware replication completed and audited

Decision: **`STOP_NATIVE_ACTION_GATE_FAIL`**

## Bottom line

The released V-JEPA 2-AC checkpoint is a competent observed-frame/state
predictor, but its future embedding is only weakly specific to the aligned
action on this deterministic native-DROID train cohort.  In the inference-
causal two-step rollout, aligned actions improved L1 by only `0.710288%` over
zero actions at horizon 1 and `0.727121%` at horizon 2.  Against an
episode-disjoint action donor the gains were `2.012733%` and `1.657419%`.
Every point estimate was below the frozen 5% materiality threshold.

The preregistered B200 replication independently returned the same decision.
Its largest registered point estimate was `2.016380%`, and the largest absolute
change from the RTX point estimates was only `0.045155` percentage points.
Thus the stop is replicated across two Blackwell GPU classes; it is not a
workstation-only numerical artifact.

The result is not “no action signal.”  Most paired intervals are above zero,
and aligned actions beat the raw logged-action diagnostic and persistence by
material margins.  The narrower conclusion is that the released predictor did
not provide enough *sample-specific aligned-action advantage* under its own
DROID contract to justify adapting it to ABC or connecting it to Wan.

Per the registered hierarchy, this result stops the matched-scratch, ABC, and
generator stages.  The exact B200 replication is now complete and did not pass
any registered endpoint.  This branch must not weaken the controls or advance
the released checkpoint to scratch, ABC, or Wan integration.

## Frozen run

- RTX evaluator commit: `6caf638337e763e3522abcdab4aa0da40e558127`.
- B200 repository commit: `29bf325c8e8e54a922fb34e4513fe668e929f36a`;
  the evaluator bytes were unchanged.
- Evaluator SHA-256:
  `758e42349c49c6e7714c3cff1a85e590c82813c066afd87d4bbf255a00fb36c2`.
- Official V-JEPA 2 source commit:
  `45d025f636dfc58fc2426905fc4a1ab755b1c3e5`.
- Official checkpoint byte count: `11760743310`.
- Official checkpoint SHA-256:
  `0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b`.
- Frozen train-manifest SHA-256:
  `cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e`.
- Cohort seed: `20260809`; 32 distinct train episodes.
- Cohort identity:
  `acd9f6bbc351fdef29ffff20f4d4bc6feeaad3d37b31c87f4e851b3fcbb4c75f`.
- Primary metric: per-clip future-token L1.
- Primary horizons: one and two 4-FPS AC steps.
- Uncertainty: paired 10,000-draw clip bootstrap, seed `20260810`.
- Gate: at least 5% relative improvement and a strictly positive 95% lower
  bound for all four controls, both modes, and both horizons.
- Protected test accessed: `false`.

The run preserved the official 64-frame predictor mask capacity, exact
180x320-to-256 crop fallback, pose-difference convention, FP32 parameter/input
storage with BF16 autocast, and two-step autoregressive state integration.  It
used only frame 0 and state 0 at causal inference.  Future RGB, future target
tokens, and measured future states were unavailable to that rollout.

## Registered action-attribution endpoints

Positive percentages favor aligned actions.  Values below are rounded to six
decimal places; `summary.json` is authoritative at full precision.  “Favored”
is the number of 32 paired clips with lower aligned L1.

| Mode/control | Horizon | Relative L1 gain | Paired 95% interval | Favored | Gate |
|---|---:|---:|---:|---:|---:|
| causal / zero | 1 | 0.710288% | [0.004775%, 1.474089%] | 20/32 | FAIL |
| causal / zero | 2 | 0.727121% | [0.094065%, 1.398454%] | 20/32 | FAIL |
| causal / episode-shuffled | 1 | 2.012733% | [0.817742%, 3.356564%] | 21/32 | FAIL |
| causal / episode-shuffled | 2 | 1.657419% | [0.644890%, 2.745274%] | 21/32 | FAIL |
| causal / time-shifted +1 | 1 | 0.940467% | [0.221271%, 1.743821%] | 21/32 | FAIL |
| causal / time-shifted +1 | 2 | 0.329602% | [-0.060436%, 0.683008%] | 21/32 | FAIL |
| causal / time-shifted -1 | 1 | 0.710288% | [0.004775%, 1.474089%] | 20/32 | FAIL |
| causal / time-shifted -1 | 2 | 0.721437% | [0.083142%, 1.393387%] | 20/32 | FAIL |
| teacher-forced / zero | 1 | 0.710288% | [0.004775%, 1.474089%] | 20/32 | FAIL |
| teacher-forced / zero | 2 | 0.345903% | [0.003867%, 0.716183%] | 21/32 | FAIL |
| teacher-forced / episode-shuffled | 1 | 2.012733% | [0.817742%, 3.356564%] | 21/32 | FAIL |
| teacher-forced / episode-shuffled | 2 | 0.983581% | [0.400159%, 1.641133%] | 21/32 | FAIL |
| teacher-forced / time-shifted +1 | 1 | 0.940467% | [0.221271%, 1.743821%] | 21/32 | FAIL |
| teacher-forced / time-shifted +1 | 2 | 0.456541% | [0.109279%, 0.841907%] | 21/32 | FAIL |
| teacher-forced / time-shifted -1 | 1 | 0.710288% | [0.004775%, 1.474089%] | 20/32 | FAIL |
| teacher-forced / time-shifted -1 | 2 | 0.344953% | [0.003160%, 0.715106%] | 21/32 | FAIL |

Horizon-1 teacher-forced and causal results are identical by construction: both
begin with the same observed frame/state and make one prediction.  Their
divergence at horizon 2 verifies that the causal path is using its own predicted
tokens and action-integrated state rather than teacher future context.

## Preregistered B200 replication

Slurm job `507254` ran the unchanged 32-episode cohort, seed, checkpoint,
implementation bytes, controls, horizons, metric, bootstrap, and 5% gate on one
NVIDIA B200.  It completed with exit code `0:0` in `00:01:57`.  The dependency
job `507232` first completed the resumable official-checkpoint transfer and
verified its frozen byte count and SHA-256 before atomically exposing it.

Positive percentages again favor aligned actions.  Full-precision values are
sealed in `summary.json`; this table rounds only for display.

| Mode/control | Horizon | Relative L1 gain | Paired 95% interval | Favored | Gate |
|---|---:|---:|---:|---:|---:|
| causal / zero | 1 | 0.712241% | [-0.010479%, 1.494590%] | 18/32 | FAIL |
| causal / zero | 2 | 0.685613% | [0.061334%, 1.344774%] | 20/32 | FAIL |
| causal / episode-shuffled | 1 | 2.016380% | [0.818682%, 3.366309%] | 23/32 | FAIL |
| causal / episode-shuffled | 2 | 1.612264% | [0.608153%, 2.702134%] | 21/32 | FAIL |
| causal / time-shifted +1 | 1 | 0.936596% | [0.215653%, 1.745866%] | 21/32 | FAIL |
| causal / time-shifted +1 | 2 | 0.311210% | [-0.074197%, 0.673325%] | 20/32 | FAIL |
| causal / time-shifted -1 | 1 | 0.712241% | [-0.010479%, 1.494590%] | 18/32 | FAIL |
| causal / time-shifted -1 | 2 | 0.679824% | [0.053035%, 1.341188%] | 21/32 | FAIL |
| teacher-forced / zero | 1 | 0.712241% | [-0.010479%, 1.494590%] | 18/32 | FAIL |
| teacher-forced / zero | 2 | 0.346374% | [-0.003194%, 0.726715%] | 19/32 | FAIL |
| teacher-forced / episode-shuffled | 1 | 2.016380% | [0.818682%, 3.366309%] | 23/32 | FAIL |
| teacher-forced / episode-shuffled | 2 | 0.984814% | [0.400940%, 1.640711%] | 23/32 | FAIL |
| teacher-forced / time-shifted +1 | 1 | 0.936596% | [0.215653%, 1.745866%] | 21/32 | FAIL |
| teacher-forced / time-shifted +1 | 2 | 0.453503% | [0.104728%, 0.839027%] | 21/32 | FAIL |
| teacher-forced / time-shifted -1 | 1 | 0.712241% | [-0.010479%, 1.494590%] | 18/32 | FAIL |
| teacher-forced / time-shifted -1 | 2 | 0.345088% | [-0.004758%, 0.725843%] | 19/32 | FAIL |

All 16 registered B200 comparisons failed the 5% materiality requirement.
The B200 diagnostics remained qualitatively consistent: causal aligned versus
raw logged action was `6.556620%` at horizon 1 (95% interval
`[4.650369%, 8.490357%]`) and `5.797807%` at horizon 2
(`[4.068254%, 7.598319%]`); aligned versus persistence was `7.170527%`
(`[5.118251%, 9.122798%]`) and `7.577390%`
(`[5.769540%, 9.243213%]`).  These remain diagnostics rather than action-
attribution gates.

The two hardware runs agree on the scientific decision and effect scale.  The
maximum absolute RTX-to-B200 change across the 16 registered point estimates
was `0.045155` percentage points.  Small per-clip BF16 kernel differences moved
some favored counts and confidence bounds, including several lower bounds
slightly across zero, but every point remained less than half the 5% threshold.

## Baseline diagnostics

These comparisons were recorded diagnostics, not substitutes for the failed
registered controls.

| Causal comparison | Horizon | Relative L1 gain | Paired 95% interval | Favored |
|---|---:|---:|---:|---:|
| aligned vs raw logged action | 1 | 6.530465% | [4.646302%, 8.422716%] | 31/32 |
| aligned vs raw logged action | 2 | 5.783672% | [4.075580%, 7.536948%] | 31/32 |
| aligned vs embedding persistence | 1 | 7.261545% | [5.240073%, 9.190839%] | 27/32 |
| aligned vs embedding persistence | 2 | 7.617314% | [5.827473%, 9.304260%] | 27/32 |

The raw-action result validates the official adapter choice: the checkpoint was
trained on differences derived from successive measured poses, not the logged
LeRobot action field.  The persistence result shows that the model is not
broken.  However, the zero-action predictor already captures almost all of that
gain.  A post-hoc loss decomposition attributes only about 9.14% of the
horizon-1 aligned-versus-persistence gap and 8.88% of the horizon-2 gap to the
aligned-versus-zero difference.  This decomposition is explanatory only and did
not alter the preregistered decision.

Representative causal L1 aggregates were:

| Condition | Horizon 1 | Horizon 2 |
|---|---:|---:|
| aligned | 0.349936656 | 0.357522919 |
| zero | 0.352439996 | 0.360141585 |
| episode-shuffled | 0.357124622 | 0.363548440 |
| raw logged action | 0.374385788 | 0.379470233 |
| persistence | 0.377337163 | 0.387002083 |

## Runtime and data audits

The first completed 32-clip run used one NVIDIA RTX PRO 6000 Blackwell
Workstation Edition, Torch `2.7.1+cu128`, CUDA runtime `12.8`, FP32 parameter
storage, and BF16 forward autocast.  One full-shape warmup was excluded.

| Runtime endpoint | Result |
|---|---:|
| observed eight-frame encoder mean | 41.948 ms |
| observed eight-frame encoder p95 | 42.300 ms |
| six-control teacher-forced batch mean | 62.236 ms |
| two-call six-control causal batch mean | 64.511 ms |
| causal batch amortized per condition | 10.752 ms |
| causal call amortized per condition | 5.376 ms |
| peak allocated memory | 7,477,983,232 bytes (6.964 GiB) |
| peak reserved memory | 8,394,899,456 bytes (7.818 GiB) |

The six-control encoder-plus-causal batch took about 106.458 ms, excluding RGB
generation and decoding.  Per-condition amortization benefits from batching and
is not singleton latency.  These numbers therefore do not establish a 5--10 Hz
complete world-model loop; the quality gate already stopped integration.

The preregistered B200 replication used Torch `2.7.1+cu128`, CUDA runtime
`12.8`, FP32 parameter storage, BF16 forward autocast, and one excluded
full-shape warmup:

| B200 runtime endpoint | Result |
|---|---:|
| observed eight-frame encoder mean | 56.616933 ms |
| observed eight-frame encoder p95 | 69.987751 ms |
| six-control teacher-forced batch mean | 67.039998 ms |
| two-call six-control causal batch mean | 115.745770 ms |
| causal batch amortized per condition | 19.290962 ms |
| causal call mean | 57.872885 ms |
| causal call amortized per condition | 9.645481 ms |
| peak allocated memory | 7,478,409,216 bytes (6.965 GiB) |
| peak reserved memory | 8,378,122,240 bytes (7.803 GiB) |

The measured B200 encoder-plus-causal control batch was `172.362703 ms`.
This is the actual warmed measurement for this small-batch protocol, not an
optimized throughput claim, and it excludes RGB generation and decoding.

All 32 AV1 videos were decoded sequentially with PyAV `17.1.0` / `libdav1d` to
`rgb24`.  Declared and decoded frame counts matched for every episode.  All
sampled parquet rows had zero in padding column 6, gripper range `[0,1]` in
column 7, and maximum pose reintegration error
`1.6763806343078613e-08`.

## Sealed artifacts

Artifact root:

```text
/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/vjepa2_ac_stage0/
  native-droid-train32-6caf638-rtxpro-v1/
```

- Registration identity:
  `3f197601e6877f48fb01c4b1940fb786c03d149f1303c736d434f381c1708218`;
  serialized SHA-256
  `cc507d2765af23f71923c9996e3e396a709f66382d7212321ad0501d9399fea9`.
- Per-clip metrics SHA-256:
  `865f7edd80c71546978c2c9a2d3a986fdb5999ed0015f6663c71c91dc858807a`.
- Timings SHA-256:
  `59cd4560d80528fc55a675f44d10162d0aac8791515bda57f4c70f71399248f2`.
- Summary identity:
  `10baf9e028b3ae60a23e443204eb192b963c5e38563cb6ef2046337b67faab4a`;
  serialized SHA-256
  `16ebec8c0726a4be90cc3ae1c7398cc3a37b18998831603b6746cbf7490c4f5d`.
- Completion identity:
  `ce06130afeb81fc6f799aed517444a570bb2a614b609bbeb493e23c3190699b6`.

The independent audit command rehashed the implementation, checkpoint,
registration, metrics, timings, summary, and completion record and returned
`status=passed`, `row_count=832`, and `protected_test_accessed=false`.

The B200 artifact root is preserved both on the cluster and in a byte-identical
local mirror:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/vjepa2_ac_stage0/
  native-droid-train32-29bf325-b200-v1/

/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/vjepa2_ac_stage0/
  native-droid-train32-29bf325-b200-v1/
```

- Registration identity:
  `ba8fed9aa195844f9404befcff656342d39517624f6bcee387175b8de6cf8410`;
  serialized SHA-256
  `7584310f5ed0a4aede91e27f229711091001c7e527f3bc89bd534ed4535a7767`.
- Per-clip metrics SHA-256:
  `d20c8127386c18a68ae604def77845a7a4252360c0a4f8b76738e2b67c6bbd59`.
- Timings SHA-256:
  `e3b95226d11f13fac5a7e4f3edb802baecb55483b21ccd578dc4f51edad4edf3`.
- Summary identity:
  `d67fb85c9fe32173fa31e9f3097c56bae82cf386cc48c4ddfb4b1c26e0600978`;
  serialized SHA-256
  `6ed2b5020135b766627ed90b07c0dffd9523fbf6aadd7d095f1cbee03896150a`.
- Completion identity:
  `a396dbd97b5a21be1254fbdbb7753c9a5acc05842c545146388f6e1c2b5b085e`;
  serialized SHA-256
  `d8956017c9b40967982e4484a64f2c8a8b2f83bd912d72486c3efb5285924851`.

The post-job independent B200 audit again returned `status=passed`,
`clip_count=32`, `row_count=832`, decision
`STOP_NATIVE_ACTION_GATE_FAIL`, and `protected_test_accessed=false`.

## Consequence for the dual-video hypothesis

This closes one specific route: using the released DROID V-JEPA 2-AC predictor
as an action-specific, inference-causal latent prior for the current generator.
It does not refute predictive video features in general, V-JEPA representations,
or dual video diffusion.  It says the released checkpoint's incremental action
information is too small under the preregistered native test to warrant paying
its adaptation and inference cost.

The B200 replication confirmed `STOP`, so this qualification branch is complete.
Effort should return to causal features with a stronger action-to-future
bottleneck (for example an analytic robot-geometry scaffold or a task-trained
compact interaction state), rather than advancing this checkpoint to ABC or Wan
despite a failed prerequisite.
