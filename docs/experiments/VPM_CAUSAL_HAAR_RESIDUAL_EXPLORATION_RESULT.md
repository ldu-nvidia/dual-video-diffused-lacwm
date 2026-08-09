# VPM causal Haar-residual exploration result

Date completed: 2026-08-09

Decision: **`EXPLORATORY_NO_HAAR_ADVANTAGE`**

## Central finding

An exact orthogonal coarse/change/detail decomposition does not expose a
materially useful causal residual at the frozen VPM one-step seam. The largest
residual component, spatiotemporal coarse (`44.03%` of held-out residual
energy), was not linearly predictable and generally harmed quality. Temporal
change (`25.31%`) was also effectively unpredictable. Spatial/detail content
(`30.66%`) was the only component with positive held-out R2, but it explained
only `0.593%` of its target energy at the full 256-episode fit dose.

The detail projection produced a statistically consistent but practically tiny
correction: versus VPM-off at dose 256 it improved decoded MSE by `0.164%` and
temporal MSE by `0.104%`. Versus its episode-shuffled control the effects were
only `0.084%` and `0.057%`. These are one to two orders of magnitude below the
frozen `3%` off/attribution gates. It beat matched `FULL_DIRECT` on temporal
MSE by `0.468%`, but not on decoded MSE, and did not approach the registered
`1%` matched-quality criterion.

Therefore the signal is best interpreted as weak subspace regularization:
discarding hard-to-predict residual directions is less harmful than fitting
the complete residual. It is not evidence that a time-frequency latent or dual
denoising process improves video generation.

## Exact tested decomposition

For the VPM latent `[B,16,4,24,120]`, only the two future latent frames were
transformed. The three camera views were split into width-40 tensors, so no
operation crossed an artificial view boundary. With spatial-LL projection `S`
and temporal-low projection `T`, the registered components were:

| Component | Projection | 3-D Haar group |
|---|---|---|
| `ST_COARSE` | `T S r` | `LLL` |
| `TEMPORAL_CHANGE` | `(I-T) S r` | `HLL` |
| `SPATIOTEMPORAL_DETAIL` | `(I-S) r` | remaining six detail groups |

They reconstructed the residual with maximum development error
`2.384e-7`, relative error energy `3.284e-16`, and worst normalized pairwise
inner product `3.366e-9`. The decomposition and view-isolation gates passed.

## Frozen execution

- Successful Slurm job: `507414`, `COMPLETED`, exit `0:0`, elapsed `27m50s`,
  one B200 on `pool0-0015`.
- Source commit: `2fcc5ceef84276002d012297979cf978bf81e994`.
- Fit rows: ABC train `128--383`, in nested doses `32/64/128/256`.
- Calibration rows: `384--415`; selection used full-dose `FULL_DIRECT` only.
- Exploratory outcome rows: prior-inspected ABC train `416--479`, with four
  new noise seeds per episode.
- The selected shared head configuration was projection capacity `1024`, ridge
  penalty `1.0`, and `65,600` parameters per head. All heads were parameter
  matched and `ZERO` was exact.
- Rows `480--510` and row `511` were not opened by this run. However,
  `480--510` had already been consumed by the completed VPM direct-residual
  frontier and are **not** a globally untouched reserve. Validation and
  protected test were unopened.
- The first allocation (`507410`) stopped safely before `fit.json` because its
  zero-head schema audit looked for `bias` instead of the ridge helper's
  `mean_y`. Its registration and log were preserved. The one-field fix,
  regression test, new commit, and fresh `v2` namespace were reviewed before
  the successful run.

## Full-dose endpoint metrics

Lower is better for the four error columns. R2/cosine are measured against
each endpoint's named local target component. Means cover 64 episodes x four
development noises.

| Endpoint | Target R2 | Cosine | Velocity MSE | Latent NMSE | Decoded MSE | Temporal MSE |
|---|---:|---:|---:|---:|---:|---:|
| VPM-off / ZERO | 0 | 0 | 0.124765 | 0.197353 | 0.0155737 | 0.0123772 |
| `FULL_DIRECT` | -0.000204 | 0.05305 | **0.124554** | 0.197057 | 0.0155629 | 0.0124225 |
| `ST_COARSE` | -0.002984 | 0.05693 | 0.124684 | 0.197266 | 0.0155798 | 0.0124322 |
| `TEMPORAL_CHANGE` | -0.000087 | 0.02684 | 0.124729 | 0.197310 | 0.0155630 | 0.0123728 |
| `SPATIOTEMPORAL_DETAIL` | **0.005928** | **0.07582** | 0.124567 | **0.197027** | **0.0155482** | **0.0123644** |

Positive paired percentages below mean lower error for the named aligned band.
Intervals are 10,000-replicate episode-clustered 95% bootstrap intervals.

| Dose-256 endpoint vs control | Velocity | Latent | Decoded | Temporal |
|---|---:|---:|---:|---:|
| `FULL_DIRECT` vs off | +0.168% `[-0.322,+0.628]` | +0.150% `[-0.308,+0.589]` | +0.069% `[-0.356,+0.466]` | **-0.365% `[-0.542,-0.202]`** |
| coarse vs off | +0.065% `[-0.359,+0.489]` | +0.044% `[-0.362,+0.434]` | -0.040% `[-0.362,+0.260]` | **-0.444% `[-0.552,-0.336]`** |
| change vs off | +0.028% `[-0.019,+0.078]` | +0.022% `[-0.026,+0.070]` | +0.068% `[+0.006,+0.130]` | +0.036% `[-0.006,+0.076]` |
| detail vs off | **+0.158% `[+0.139,+0.181]`** | **+0.165% `[+0.146,+0.186]`** | **+0.164% `[+0.131,+0.199]`** | **+0.104% `[+0.075,+0.137]`** |
| detail vs shuffled | +0.080% `[+0.069,+0.093]` | +0.083% `[+0.072,+0.095]` | +0.084% `[+0.067,+0.102]` | +0.057% `[+0.042,+0.075]` |
| detail vs matched full direct | -0.010% `[-0.486,+0.474]` | +0.015% `[-0.437,+0.478]` | +0.094% `[-0.308,+0.536]` | **+0.468% `[+0.297,+0.648]`** |

The high favorable fractions for detail (61/64 decoded and 58/64 temporal
episodes versus off) show that the tiny effect is systematic, not that it is
large enough to matter. Aligned-versus-shuffled effects below `0.1%` also make
clear that most of the correction is a population/subspace effect rather than
a strong sample-specific causal prediction.

## Learning-curve interpretation

The registered log-dose AUC subgate favored temporal-change and detail over
`FULL_DIRECT`, but only because the full direct learning curve was harmful on
this development population:

| Component | Decoded absolute AUC | Band minus full direct (95% CI) | Temporal absolute AUC | Band minus full direct (95% CI) |
|---|---:|---:|---:|---:|
| `ST_COARSE` | -0.580 | -0.146 `[-0.286,-0.020]` | -0.434 | +0.061 `[-0.008,+0.134]` |
| `TEMPORAL_CHANGE` | +0.054 | +0.487 `[+0.193,+0.791]` | +0.022 | +0.517 `[+0.397,+0.642]` |
| `SPATIOTEMPORAL_DETAIL` | +0.178 | +0.611 `[+0.278,+0.961]` | +0.074 | +0.569 `[+0.430,+0.710]` |

The AUC units are percentage improvement integrated over normalized log2 dose.
Absolute gains of `0.02--0.18` percentage points do not rescue a mechanism
that misses the frozen `3%` quality and sample-attribution gates. Positive AUC
or R2 alone was prospectively declared insufficient.

## Serving and evidence integrity

- One shared Wan call served all 34 endpoints in each batch: 128 batch calls / 
  256 sample calls total. Teacher and V-JEPA calls were zero.
- Clean future targets were constructed only after every endpoint was
  materialized, and no target entered the correction graph.
- `ZERO` and VPM-off were bit-exact for every row.
- The endpoint contained all `8,704` registered rows and `128` timing rows.
- The primary complete one-step endpoint was `411.14 ms` mean and `415.26 ms`
  p95 per two-sample batch. Dose-256 adapter p95 values were `0.593 ms` full,
  `0.775 ms` coarse, `0.767 ms` change, and `1.021 ms` detail.
- The in-run audit passed. A separate read-only post-run check rehashed every
  registered artifact and independently verified all 8,704 row keys, split
  flags, paired zero hashes, and zero metrics. A second root-level replay
  independently validated all six JSON identities, all 8,704 row identities,
  the 34 x 256 endpoint inventory, pairing, call counts, scoring order, and
  forbidden-access flags with zero discrepancies.

Immutable identities:

- registration: `373bf8673015975bef7a3b10a3c5928290853c4660d5f789adc0a7093c19a7e7`;
- fit: `c0b6fb53d481bcf4117d2c09eb08b8ddd9dfe90be7d78f79129683114e1ccf97`;
- endpoint: `13deb463829c923831c86b607d481fbaad741da1a1ce73dc4901734b4bb7bd2d`;
- analysis: `11f5c081b4b2dd43b41acb00232c78e5a279d93a30748b00e6f662dd91c0f723`;
- completion: `3787eb12b2710af43ebe0e9087af2dcb0831653010e0f0eae4f619dd114cb238`;
- audit: `8d864cd98a6fcf57770ac8a0fbd7d27428b46bc007facdeb1d49349cfc868f7e`.

Canonical artifact:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/
  vpm_causal_haar_residual_exploration/
  vpm-causal-haar-fit256-dev64-seed20260910-2fcc5ce-v2/
```

## Consequence

Do not advance coarse or temporal-change residuals into a dual-diffusion
training run. The causal VPM trunk does not expose them to the tested linear
token-local head, despite their large oracle residual energy.

The detail result permits at most one narrow hypothesis: an explicitly
detail-projected nonlinear residual adapter may be a safer regularizer than an
unconstrained full-residual adapter. It is not yet a dual latent and its current
effect is too small to justify a major claim. Any follow-up must use a genuinely
new dataset/split, retain equal-capacity `FULL_DIRECT` and shuffled controls,
and require materially larger (`>=3%`) decoded and temporal improvements.
Rows 480--510 cannot serve as that confirmation because the direct-residual
frontier already consumed them.

A learnable, view-isolated wavelet/filter bank remains a distinct **untested**
hypothesis informed by Frequency-Forcing: it could learn a task-aligned basis
instead of accepting fixed first-level Haar bands. The present result neither
supports nor falsifies that mechanism. It would still need a prospective new
split, equal total parameters/calls, full-direct and shuffled controls, an
identity/reconstruction constraint, and the same material quality thresholds;
merely learning a basis or reporting spectral predictability would not pass.

In the broader program, this result strengthens the stop on generic
frequency/coarse conditioning. The remaining plausible directions must add a
causal interaction variable or embodiment computation that is not already a
weak projection of VPM's residual.

## Claim boundary

This is branch-scoped evidence from `research/vpm-multiscale-residual`: a
post-selection exploratory analysis on one frozen VPM checkpoint, one
prior-inspected ABC-train development population, four noises, and linear
token-local heads. It does not rule out nonlinear spatiotemporal adapters,
end-to-end training, learnable wavelets, other datasets, or other
representations. It does establish that this exact orthogonal Haar grouping
does not provide a material causal or sample-efficiency advantage over matched
full-direct correction at VPM@1.
