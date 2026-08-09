# VPM causal learnable-wavelet qualification result

Date completed: 2026-08-09

Decision: **`EXPLORATORY_NO_LEARNED_QMF_QUALIFICATION`**

## Central finding

Learning an admissible, view-isolated spatial QMF basis on clean future VAE
latents did not turn the frozen VPM one-step residual into a materially useful
causal auxiliary signal. The learned filter was valid and genuinely different
from zero-padded Haar, but it reduced held-out normalized coefficient L1 by
only `1.534%` (episode-paired 95% CI `[1.451%, 1.616%]`). This missed the
prospectively frozen `5%` basis-qualification gate.

The downstream result was equally small. The best learned-QMF arm was the
spatiotemporal-detail projection. Relative to VPM-off, it improved decoded MSE
by `0.155%` and decoded temporal MSE by `0.103%`. Relative to its
episode-shuffled control, the improvements were only `0.076%` and `0.049%`.
Those effects were statistically consistent but roughly 40--60 times below
the frozen `3%` materiality/attribution gate. The learned detail basis did not
beat matched Haar: decoded MSE was `0.0086%` worse and temporal MSE was only
`0.0025%` better, with both confidence intervals crossing zero.

This result strengthens the fixed-Haar finding. A data-adapted orthogonal
frequency partition changes where residual energy falls, but it does not
expose a Latent-Forcing-like causal feature to this frozen trunk plus linear
token-local head. It is evidence against this **frozen learned-basis
qualification**, not against jointly trained auxiliary streams, nonlinear
adapters, or Frequency-Forcing itself.

## Basis qualification

The basis was fit for 512 Adam updates on only the two clean future VAE latent
frames from ABC-train rows 128--383. Actions were loaded by the ordinary
dataset but never entered the objective. Rows 384--415 produced sealed,
identity-bearing clean-latent receipts after the filter was frozen. No VPM
forward, residual target, hidden state, development metric, teacher feature,
or future outcome was available to the basis optimizer.

The learned low-pass taps were:

```text
[ 0.5815994,  0.7903976,  0.1561449, -0.1074356,
 -0.0252583,  0.0201865, -0.0053793,  0.0039582 ]
```

| Basis check | Result | Frozen gate |
|---|---:|---:|
| absolute cosine with zero-padded Haar | 0.970148 | `< 0.9999` pass |
| maximum admissibility residual | `1.434e-7` | `<= 2e-6` pass |
| held-out coefficient/input energy ratio | within `2.39e-7` of 1 | `<= 1e-5` pass |
| normalized-L1 reduction vs Haar | **1.534%** `[1.451,1.616]` | `>= 5%`, lower CI `>0`: **fail** |
| favorable calibration episodes | 32/32 | diagnostic |
| minimum mean terminal-band energy | 1.566% (`HH`) | `>= 0.5%` pass |
| threshold retention | 100% | pass |

Mean normalized L1 was `0.586878` for Haar and `0.577876` for learned QMF.
The QMF terminal energy fractions were `85.643% LL`, `3.206% LH`, `9.585%
HL`, and `1.566% HH`. The filter therefore learned a modestly sparser,
low-pass-heavier representation, but not the predeclared material adaptation.

## Frozen execution

- Successful Slurm job: `507595`, `COMPLETED`, exit `0:0`, elapsed `15m58s`,
  one B200 on `pool0-0171`.
- Source commit: `bbaf29cf11dc9e94689add884db55b77bc7d4ecb`.
- Basis/head-fit rows: ABC train `128--383`; two head-fit noise seeds.
- Basis receipt rows: `384--415`; no basis update or outcome selection.
- Exploratory outcome rows: prior-inspected ABC train `416--479`, four new
  noise seeds per episode.
- Rows `480--510` were not opened by this run, but had already been consumed
  by the direct-residual frontier and are not a globally fresh reserve. Row
  `511`, validation, and protected test remained unopened.
- Each ridge head used projection capacity 1024, penalty 1.0, and 65,600
  parameters. Those values were inherited from the completed fixed-Haar run;
  no new outcome-based head selection occurred.
- The experiment evaluated VPM-off, exact zero, aligned/shuffled full direct,
  and aligned/shuffled coarse/change/detail targets under both Haar and learned
  QMF: 4,096 sealed rows total.

The learned-QMF prefit is additional compute and repeated train-row exposure
that fixed Haar does not require. Therefore this experiment cannot claim
end-to-end efficiency parity or improved sample efficiency even if a quality
effect had appeared.

## Endpoint metrics

Lower is better for the four error columns. R2/cosine use each arm's named
local target. Means cover 64 episodes x four development noises.

| Endpoint | Target R2 | Cosine | Velocity MSE | Latent NMSE | Decoded MSE | Temporal MSE |
|---|---:|---:|---:|---:|---:|---:|
| VPM-off / ZERO | 0 | 0 | 0.124618 | 0.197162 | 0.0155538 | 0.0123735 |
| `FULL_DIRECT` | -0.000695 | 0.05149 | 0.124476 | 0.196947 | 0.0155492 | 0.0124245 |
| QMF `ST_COARSE` | -0.003257 | 0.05331 | 0.124589 | 0.197135 | 0.0155585 | 0.0124266 |
| QMF `TEMPORAL_CHANGE` | -0.000045 | 0.02743 | 0.124577 | 0.197108 | 0.0155421 | 0.0123699 |
| QMF `SPATIOTEMPORAL_DETAIL` | **0.006607** | **0.07993** | **0.124419** | **0.196834** | **0.0155296** | **0.0123608** |
| Haar `SPATIOTEMPORAL_DETAIL` | 0.005745 | 0.07482 | 0.124426 | 0.196845 | 0.0155283 | 0.0123611 |

Positive paired percentages below mean lower error for learned QMF detail.
Intervals are 10,000-replicate episode-clustered 95% bootstrap intervals.

| Learned QMF detail vs control | Velocity | Latent | Decoded | Temporal |
|---|---:|---:|---:|---:|
| VPM-off | **+0.160%** `[+0.140,+0.183]` | **+0.166%** `[+0.146,+0.188]` | **+0.155%** `[+0.126,+0.191]` | **+0.103%** `[+0.070,+0.142]` |
| matched shuffled QMF | **+0.078%** `[+0.068,+0.089]` | **+0.080%** `[+0.071,+0.091]` | **+0.076%** `[+0.058,+0.095]` | **+0.049%** `[+0.029,+0.072]` |
| matched Haar detail | +0.0056% `[+0.0001,+0.0115]` | +0.0057% `[+0.0002,+0.0117]` | -0.0086% `[-0.0197,+0.0029]` | +0.0025% `[-0.0026,+0.0078]` |
| matched full direct | +0.046% `[-0.447,+0.553]` | +0.057% `[-0.385,+0.533]` | +0.126% `[-0.307,+0.589]` | +0.513% `[+0.335,+0.706]` |

Beating full direct on temporal MSE is not a positive mechanism result: the
full-direct correction itself made temporal MSE `0.412%` worse than VPM-off.
The decisive comparisons are versus off, shuffled, and matched Haar, and none
approached a material effect.

For comparison, matched Haar detail improved decoded/temporal MSE over off by
`0.164%`/`0.101%`, nearly reproducing the prior fixed-Haar result. Learned QMF
shifted development residual energy from detail into low-frequency groups:
Haar coarse/change/detail fractions were `44.021%/25.272%/30.707%`, versus
QMF `45.886%/26.342%/27.773%`. That redistribution did not improve quality.

## Serving and evidence integrity

- One shared feature-off Wan call served every endpoint in a batch: 128 batch
  calls / 256 sample calls. Teacher and V-JEPA calls were zero.
- Clean future targets were encoded only after every deployable correction and
  decode was materialized; no future target entered a correction graph.
- `ZERO` and VPM-off were bit-exact for every episode/noise pair.
- Haar and QMF partitions were view-isolated and passed reconstruction,
  orthogonality, energy, history-mask, and projection-leakage contracts. The
  maximum development reconstruction error was `2.384e-7`.
- The primary complete one-step endpoint took `407.19 ms` mean, `401.07 ms`
  median, and `402.98 ms` p95 per two-sample batch. The QMF-detail adapter
  itself took `1.79 ms` median and `1.99 ms` p95. These timings do not rescue
  a quality effect and are not an end-to-end training-efficiency comparison.
- The in-run audit passed. A separate read-only post-run replay validated all
  seven JSON identities, 32 calibration rows, 4,096 development rows, 128
  timing rows, basis gates, endpoint timing summaries, paired hashes, and the
  full analysis with zero discrepancies. A second independent rehash matched
  all 11 sealed artifact digests in `audit.json`; the audit file itself has
  SHA-256 `29bcd870bae1dc80b77e1ce5f1d48799bd299027a9815f7b6464721807fab2f5`.

Immutable identities:

- registration: `741e040b313cab11062784c0e516cdb148494e63d345f880738be287fc71915b`;
- basis: `e9166097adf9f30caa6c0212d22a7692ef3b3a7f5deb0cb482e7db70bf7e49f1`;
- fit: `1051e8255aa3ba753b45392e244d1a515200f5238bb1a79e81d330668ba3e37f`;
- endpoint: `909b8a58b7eef4f18268d4b06503216980f98e44845f0789ee253e9d682066d5`;
- analysis: `54e510f6c7b9ef8b144e9393511c7df9f01466b0a5bf63ae628a73c0f745678e`;
- completion: `5683e2c41a01da35b9dee9e442b2fba4237afd7c2a32beefb9f9c9136abeca06`;
- audit: `72ca526909c167391816e33c3844c2b3ae36ff105abaab640d56992b302b38ce`.

Canonical artifact:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/
  vpm_causal_learnable_wavelet_qualification/
  vpm-causal-learnable-wavelet-fit256-dev64-seed20261001-bbaf29c-v1/
```

Full log:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/logs/slurm-507595.out
```

The completed 5,071-byte log has SHA-256
`366cad229d0815d1558835105ecd7aebee95a2ec296a7285710adeae5c0b672b`.

## Consequence and next step

Stop tuning frozen wavelet filters, thresholds, seeds, or linear residual
heads on this population. Fixed Haar and learned QMF both yield only a
repeatable sub-`0.2%` detail regularization effect, while learned QMF adds a
512-step prefit and does not beat Haar. More basis search would be
post-selection optimization around an effect that is already far below the
registered bar.

If the program continues toward a video analogue of Latent Forcing, the next
experiment must change the mechanism rather than the Fourier/wavelet basis:
jointly train an inference-available auxiliary predictive stream and the video
generator so the auxiliary state becomes clean/predictive early and can affect
the backbone during denoising. It should use genuinely untouched data, an
equal-total-compute no-auxiliary control, shuffled/sample-mismatched controls,
multiple generation NFEs, and the same `>=3%` decoded and temporal gates.
This frozen external-head result gives no evidence for fewer denoising steps,
training-epoch savings, or real-time DAgger quality.

An orthogonal alternative is to prioritize a causal interaction/action-motion
state that is not merely another projection of VPM's residual. The accumulated
TF, V-JEPA, fixed-Haar, full-direct, and learned-QMF results all point to the
same bottleneck: inference-visible future structure must be *predicted and
jointly useful*, not extracted from clean future video or recovered as a weak
post-hoc residual subspace.

## Claim boundary

This is branch-scoped, post-selection exploratory evidence on one frozen VPM
checkpoint, one prior-inspected ABC-train development population, four noises,
and linear token-local heads. It does not falsify Frequency-Forcing, joint
generator/transform training, nonlinear spatiotemporal adapters, other video
models, or other datasets. It does establish that this exact frozen,
clean-latent-prefit QMF basis does not provide a material causal or quality
advantage over matched Haar, full direct correction, or VPM-off at the tested
one-step seam.
