# VPM invertible-multirate probe result

Date completed: 2026-08-09

Decision: **`NO_GO_GENERIC_EARLY_SUBSPACE`**

## Central finding

The tested fixed, exactly invertible LL/HH mixed-clock construction is not a
useful video analogue of Latent Forcing.  The aligned LL-first two-call endpoint
did not materially beat either ordinary VPM baseline, and a rank-matched HH
first step was slightly better on all three registered quality metrics.  The
aligned endpoint did strongly beat episode-shuffled and time-reversed controls,
so coherent sample identity and temporal direction matter.  That advantage is
not specific to the proposed low-frequency subspace, however.

There were two very small favorable effects.  Relative to one-call VPM1,
aligned improved decoded MSE by `0.04860%`; relative to ordinary two-call VPM2,
it improved video-latent NMSE by `0.02427%`.  Both effects had positive
familywise lower bounds, but each was accompanied by regressions in the other
quality metrics and was roughly two orders of magnitude below the registered
`3%` threshold.  Aligned worsened the LL-complement NMSE from call 1 to call 2
by `0.13311%` instead of improving it.

This is evidence against this exact **untrained fixed-subspace mixed-clock
schedule**, not against learned dual-video diffusion in general.  It says that
splitting an already useful first denoising prediction into fixed multirate
subspaces and feeding one coherent part through the second call does not itself
create an informative auxiliary latent.

## Frozen execution

- Slurm job `507706`: `COMPLETED`, exit `0:0`, elapsed `00:07:25`, batch MaxRSS
  `5,923,940 KiB`, one B200 on `pool0-0363`.
- Source commit: `9c4fb880a46cbc0e0376972b76bb3922e26e88fa`.
- Prior-inspected development population: ABC-train rows `416--479`, 64 clips
  and four frozen noise seeds per clip.
- Six endpoints produced `1,536` paired rows and `128` timing rows.  The run made
  exactly `768` Wan calls and `768` decoder calls; the public-entrypoint parity
  preflight's three Wan and four decoder calls were separately excluded.
- VPM1 used one Wan call.  VPM2 and each mixed-clock endpoint used two.  The
  first Wan result, noise, actions, history, and scoring target were paired.
- There were zero new parameters, optimizer updates, teachers, feature
  encoders, or auxiliary-target calls.  Full future RGB was read only after all
  six endpoints for a batch were closed in the event ledger.
- Fresh-reserve rows `480--510`, constructor-probe row `511`, validation, the
  protected test split, and the V-JEPA target array were unopened.
- Public `sample_future_deployable` VPM1 and VPM2 outputs were bit-exact with the
  independent manual sampler in both final latent and decoded uint8 output.

The earlier job `507690` failed before registration, model loading, or data
access because the runtime-receipt parent did not exist.  The audited repair
created only the canonical artifact-family parent, retained fresh output and
runtime guards, moved the successful run to the commit-derived `v2` namespace,
and added executable missing-parent and no-mutation regressions.

## Aggregate endpoint metrics

Lower is better.  Values are mean +/- sample standard deviation over 64 clips x
four noise seeds.  The complement column is the mean future NMSE outside the LL
projection.

| Endpoint | Video NMSE | Decoded MSE | Temporal MSE | LL-complement NMSE |
|---|---:|---:|---:|---:|
| VPM1 | **0.196379622 +/- 0.067773394** | 0.0154957649 +/- 0.0104934484 | 0.0123679334 +/- 0.0106344890 | **0.353170387** |
| VPM2 ordinary | 0.196503990 +/- 0.067834390 | **0.0154854984 +/- 0.0104968678** | 0.0123671718 +/- 0.0106351805 | 0.353742001 |
| LL first, aligned | 0.196456305 +/- 0.067789039 | 0.0154882337 +/- 0.0104968586 | 0.0123708905 +/- 0.0106388467 | 0.353640509 |
| LL first, episode shuffled | 0.915093568 +/- 0.209930700 | 0.0623185208 +/- 0.0219452940 | 0.0174909820 +/- 0.0108337207 | 0.356628394 |
| LL first, time reversed | 0.241063967 +/- 0.076559817 | 0.0174564663 +/- 0.0112024475 | 0.0135609155 +/- 0.0111518666 | 0.354695082 |
| HH first, rank matched | 0.196450926 +/- 0.067831842 | 0.0154856871 +/- 0.0104950062 | **0.0123654059 +/- 0.0106330357** | 0.353455448 |

Additional transform diagnostics are means except for the final p-lock column,
which is the maximum over all rows.

| Endpoint | LL future NMSE | LL delta MSE | LL delta cosine | LL prediction energy | HH prediction energy | p-lock max abs |
|---|---:|---:|---:|---:|---:|---:|
| VPM1 | 0.164871924 | 0.127055301 | 0.425577403 | 0.867571889 | 0.006982002 | 0 |
| VPM2 ordinary | 0.164908105 | 0.127137292 | 0.424791678 | 0.867380987 | 0.007063076 | 0 |
| LL first, aligned | 0.164871925 | 0.127055301 | 0.425577400 | 0.867265872 | 0.007063641 | 2.38418579e-7 |
| LL first, episode shuffled | 1.030919721 | 0.172339122 | 0.085729995 | 0.868498415 | 0.006976944 | 2.38418579e-7 |
| LL first, time reversed | 0.218431576 | 0.237253148 | -0.425577401 | 0.867449379 | 0.007059643 | 2.38418579e-7 |
| HH first, rank matched | 0.164901568 | 0.127131317 | 0.424843378 | 0.867434055 | 0.006982637 | 2.38418579e-7 |

## Paired effects and attribution

Positive percentages mean lower error for LL-FIRST-ALIGNED.  Each cell is
`point effect / one-sided Bonferroni lower bound / favorable episodes`.  The
registered simultaneous family contains five contrasts x three metrics, uses
10,000 episode-clustered bootstrap replicates, and keeps the four noise seeds
inside each episode.

| Control | Decoded MSE | Temporal MSE | Video NMSE |
|---|---:|---:|---:|
| VPM1 | +0.048602% / +0.018146% / 45/64 | -0.023909% / -0.051397% / 21/64 | -0.039048% / -0.052829% / 9/64 |
| VPM2 ordinary | -0.017664% / -0.036976% / 24/64 | -0.030069% / -0.048541% / 19/64 | +0.024267% / +0.007362% / 43/64 |
| Episode shuffled | +75.146660% / +70.169322% / 64/64 | +29.272751% / +22.870765% / 63/64 | +78.531561% / +75.465083% / 64/64 |
| Time reversed | +11.275092% / +8.672796% / 63/64 | +8.775403% / +6.817521% / 64/64 | +18.504492% / +15.582442% / 64/64 |
| Rank-matched HH | -0.016445% / -0.033631% / 25/64 | -0.044354% / -0.064748% / 22/64 | -0.002738% / -0.020071% / 28/64 |

The LL-complement comparison from VPM1 call 1 to aligned call 2 was
`-0.13311497%`, with 95% interval
`[-0.16303356%, -0.10282289%]`, Bonferroni lower bound `-0.17395712%`,
and only `9/64` favorable episodes.  Its mean changed from `0.3531703867` to
`0.3536405093`; the registered `>=3%` complement-improvement gate failed.

The aligned endpoint therefore did improve *something*, but not a coherent
quality frontier: decoded MSE moved by only `+0.04860%` versus VPM1 while
latent and temporal errors regressed; latent NMSE moved by only `+0.02427%`
versus VPM2 while decoded and temporal errors regressed.  Its large gains over
shuffling and reversal establish input coherence, not an LL-specific dual
denoising advantage.  The rank-matched HH result directly rejects that
specificity claim.

## Latency

Times are milliseconds per two-sample batch on the successful allocation.

| Endpoint | Projection p95 | Composed mean | Composed p95 |
|---|---:|---:|---:|
| VPM1 | 0 | 341.634970 | 345.598430 |
| VPM2 ordinary | 0 | 418.334675 | 424.117304 |
| LL first, aligned | 0.684314 | 418.440983 | 422.123677 |
| LL first, episode shuffled | 0.685192 | 418.553922 | 423.945528 |
| LL first, time reversed | 0.680461 | 418.599716 | 422.998704 |
| HH first, rank matched | 1.020072 | 418.934467 | 422.981391 |

Aligned's p95 was `-0.470065%` relative to ordinary VPM2, consistent with no
meaningful added overhead, and its `0.684314 ms` projection passed the `2 ms`
bound.  It remained `22.142822%` slower at p95 than one-call VPM1 because it
still requires two Wan calls.  The timing gate passed; there is no acceleration
result.

## Gate accounting

| Gate | Registered condition | Result |
|---|---|---|
| Aligned vs VPM2 | decoded and temporal each >=3%, lower bound >0, favorable fraction >=0.60; latent point >=0 and lower bound >-1% | **Failed quality**; latent guardrail passed |
| Aligned vs VPM1 | same quality and latent requirements | **Failed both subgates** |
| Aligned vs shuffled | decoded and temporal each >=1% with lower bound >0 | Passed |
| Aligned vs reversed | decoded and temporal each >=1% with lower bound >0 | Passed |
| Aligned vs rank-matched HH | decoded and temporal each >=1% with lower bound >0 | **Failed** |
| Complement | LL-complement improvement >=3% | **Failed** (`-0.133115%`) |
| P-lock | maximum absolute error <=2e-6 | Passed (`2.38418579e-7`) |
| Latency | projection p95 <=2 ms and aligned-vs-VPM2 p95 overhead <=5% | Passed (`0.684314 ms`, `-0.470065%`) |
| Evidence receipts | exact projector, pairing, call/capacity, access-order, split, and zero-target-leak contracts | All passed |

The aggregate gate failed.  The decision label is
`NO_GO_GENERIC_EARLY_SUBSPACE` because the aligned endpoint passed the two
coherence controls but failed the prospectively rank-matched HH specificity
control.  Independently, it also failed both practical baseline-quality gates
and the complement-mechanism gate.

## Evidence audit and mirror

The in-run audit status is `PASS`, and the completion status is `COMPLETE`.
Independent root replay found all five JSON identities valid, all `1,536` row
and `128` timing identities valid, exact reproduction of every computed
analysis field, and equality of all five audit artifact hashes.

Immutable identities:

- registration: `0b1554938e132814e5ab211c6a02f533c63d299b421253e83209bb10d35cecaa`;
- endpoint: `41f65cc8dd43c92825099a4816766afa6fa062b24b3c823ab36178824359b7e7`;
- analysis: `1e88eef5f626fcbb3ffb80fd22605e672876184a319cb2dedf661c147f221b4e`;
- audit: `272d7677928993b3463d8b75fa294275945939776add8079b9e0e5df3acdeb0e`;
- completion: `872c32ef92d82663dede9e15aba9c41e5652529c41874c89179ee98760a4d5df`.

Only the eight small JSON/JSONL receipts were copied to the local mirror; no
video, checkpoint, or model artifact was copied.  Remote and local SHA-256 were
equal for every file:

| Receipt | SHA-256 |
|---|---|
| runtime verification | `6ef868c07d7ad9101e3c948ecab75b7a2b6dd346b1b403021866b57c28f60014` |
| analysis | `252cde200a7fa08f14e44a987928f93c08f8ff8b489bcc5f5e61536ca0a45bfc` |
| audit | `eec195848b81ddb88d417ed9dc87a814913aa6cc2de4121a7405564f9deb4af4` |
| endpoint completion | `466038cfddbdb9e8972e41917513a6d0cfe03f16fd014a4ba19d248b0eff1d76` |
| endpoint rows | `1941fcc5392a018ea8f6604cd6d2758bf7e0396e7cf910ebdeb4a7ecf95a30e0` |
| registration | `dce0a2c070acb2daff329663517499b78d041d1af4939f6b02e7d54a8fbd3479` |
| run completion | `e157522d37cbbfdabb1d84d2afe146a65f6de9b673158910ea6af25256e2683f` |
| timing rows | `143bbca5e2fe08cf403aa70a30c6d1a4fe6602edcbabfed32d2fb597d5a41a4f` |

Canonical remote artifact:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/vpm_invertible_multirate_probe/
  vpm-invertible-multirate-dev64-seed20261101-9c4fb88-v2/
```

Local receipt mirror:

```text
/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/
  vpm_invertible_multirate_probe/
  vpm-invertible-multirate-dev64-seed20261101-9c4fb88-v2/
```

## Consequence

Do not scale this fixed LL/HH mixed-clock schedule or describe it as a
dual-denoising quality or speed gain.  Its only clear signal is that the second
call is sensitive to coherent identity and time order, while an arbitrary
rank-matched coherent subspace works at least as well as LL.

A defensible next experiment must learn or predict a causal auxiliary from
history and actions, make that auxiliary available at inference, and then beat
VPM1, equal-call VPM2, shuffled/reversed controls, and a rank/capacity-matched
generic subspace.  Fixed frequency projection alone is not enough; any learned
subspace or auxiliary-transition proposal must retain the same target-blind
event ledger and require material decoded and temporal gains before scaling.

## Claim boundary

This is post-selection exploratory evidence from one frozen VPM checkpoint,
one prior-inspected ABC-train development population, four noises, and an
untrained exactly invertible rank-1/4 split.  It does not test end-to-end joint
training, learned filters, predictive auxiliary dynamics, other checkpoints,
new datasets, FVD, or protected test performance.  It rejects only the tested
claim that fixed aligned LL-first mixed-clock inference creates a material,
frequency-specific, faster video-generation advantage.
