# V-JEPA 2-AC native-DROID Stage-0 result

Date: 2026-08-08

Status: 32-episode RTX Blackwell qualification completed and audited;
preregistered B200 hardware replication pending

Decision: **`STOP_NATIVE_ACTION_GATE_FAIL`**

## Bottom line

The released V-JEPA 2-AC checkpoint is a competent observed-frame/state
predictor, but its future embedding is only weakly specific to the aligned
action on this deterministic native-DROID train cohort.  In the inference-
causal two-step rollout, aligned actions improved L1 by only `0.710288%` over
zero actions at horizon 1 and `0.727121%` at horizon 2.  Against an
episode-disjoint action donor the gains were `2.012733%` and `1.657419%`.
Every point estimate was below the frozen 5% materiality threshold.

The result is not “no action signal.”  Most paired intervals are above zero,
and aligned actions beat the raw logged-action diagnostic and persistence by
material margins.  The narrower conclusion is that the released predictor did
not provide enough *sample-specific aligned-action advantage* under its own
DROID contract to justify adapting it to ABC or connecting it to Wan.

Per the registered hierarchy, this result stops the matched-scratch, ABC, and
generator stages.  The only remaining action in this branch is an exact B200
replication of the same native gate.  It may not weaken the controls or advance
downstream work unless every native comparison unexpectedly passes.

## Frozen run

- Evaluator commit: `6caf638337e763e3522abcdab4aa0da40e558127`.
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

## Consequence for the dual-video hypothesis

This closes one specific route: using the released DROID V-JEPA 2-AC predictor
as an action-specific, inference-causal latent prior for the current generator.
It does not refute predictive video features in general, V-JEPA representations,
or dual video diffusion.  It says the released checkpoint's incremental action
information is too small under the preregistered native test to warrant paying
its adaptation and inference cost.

The scientifically valid next step for this branch is only the B200 replication.
If it confirms `STOP`, effort should return to causal features with a stronger
action-to-future bottleneck (for example an analytic robot-geometry scaffold or
a task-trained compact interaction state), rather than advancing this checkpoint
to ABC or Wan despite a failed prerequisite.
