# Prospective object/contact-slot Stage-0 result

Date completed: 2026-08-08

Decision: **`STOP_OBJECT_CONTACT_SLOT`; do not integrate this deterministic
slot branch into the video generator**

## Central finding

The localized slot representation was meaningful but planned action did not
make it more predictable. This cleanly separates two hypotheses that were
confounded in the earlier dense PCA screen:

- Oracle target slots reconstructed the fresh held-out component field 34.70%
  better than the fit-mean field and 34.65% better than zero. Both frozen
  relevance gates passed with simultaneous lower bounds above 30%.
- Correctly aligned action improved standardized slot prediction only 0.22%
  over history, 0.42% over a shuffled episode, and was 0.27% worse than the
  fit-mean action. Decoded-field effects were similarly near zero.
- Aligned action was indistinguishable from action shifted by one chunk in
  either direction. None of the 12 mandatory causal/timing contrasts reached
  the frozen 5% effect threshold.

Thus spatial compression was not the blocker: these four persistent,
location-bearing slots retained substantial held-out event-field information.
The blocker was incremental causal predictability from the planned joint
commands. This result does not establish that physical object state is
action-independent; it rejects this observational slot extractor plus linear
history/action predictor as an auxiliary dual-diffusion state.

## Fresh prospective population

The exact parent fit256 was reused for fitting. Every one of the earlier 64
score episodes was excluded before constructing the new score set. Recomputing
the immutable ABC train512 D405 scan yielded:

| Original motion stratum | Eligible D405 | Reused fit | Excluded prior score | Untouched pool | Fresh score | Still unused |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 104 | 64 | 16 | 24 | 16 | 8 |
| 1 | 104 | 64 | 16 | 24 | 16 | 8 |
| 2 | 104 | 64 | 16 | 24 | 16 | 8 |
| 3 | 103 | 64 | 16 | 23 | 16 | 7 |
| **Total** | **415** | **256** | **64** | **95** | **64** | **31** |

Fresh clips were ranked with the new frozen SHA salt. Fit/prior-score/fresh-score
overlap was exactly zero. Registration was sealed and reported before fresh
score RGB pixels or measured state arrays were opened. Validation and
protected test remained unopened.

## Executed representation and predictor

- Rendered source/target robot masks used the official ABC/YAM model and a
  severe 16-pixel union dilation at 180x320.
- At 45x80, connected moving support required both luminance-event magnitude
  at least 0.20 and Farneback flow magnitude at least 0.20 pixels. Components
  below four pixels, above 20% of the work image, or farther than 12 pixels
  from robot support were rejected.
- A deterministic past-only tracker matched at most 12 candidates into four
  persistent slots, tolerating one missed transition. Null slots were exactly
  zero; no backward matching consulted future observations.
- Each slot contained 14 explicit values: presence, centroid x/y, area,
  positive/negative mass, covariance xx/xy/yy, signed flow u/v, contact
  proximity, contact onset, and track age.
- History was `[4,4,14]`; future target was `[8,4,14]`. All 448 flattened future
  coordinates were active on fit256. There was no learned target encoder.
- Equal-width ridge designs compared history+zeros32 with history+action PCA32.
  Fit-only five-fold CV selected alpha 1000 for both; action PCA retained
  99.34% of fit variance.
- Mandatory references were history-only, same-stratum shuffled action,
  fit-mean action, raw-zero action, and nonwrapping shifts -1/+1. Two metrics
  times six controls formed a 12-way one-sided Bonferroni family.

The registered source is commit
`f8e722646ffe7c1720ec8b3bc994df64c9a55dd1`, SHA-256
`c19dd64c7199885420ec8d270934fb4a1d6b403850d292e7363ac806c5367906`.
The protocol SHA-256 is
`1ddd51429465f7be41eb6444679f4c2b621c50e49512fb3f83c16ee4b1af4507`.

## Mandatory causal and timing effects

Positive percentages favor aligned action. Every row required at least 5%
gain, a strictly positive familywise 99.5833% lower bound, and at least 60%
favorable fresh episodes.

| Metric | Aligned reference | Relative gain | Simultaneous lower bound | Favorable clips | Gate |
|---|---|---:|---:|---:|---|
| Slot MSE | History only | +0.217% | -0.514% | 29/64 | fail |
| Slot MSE | Episode shuffled | +0.422% | -0.286% | 37/64 | fail |
| Slot MSE | Fit-mean action | -0.272% | -1.080% | 24/64 | fail |
| Slot MSE | Raw-zero action | +0.820% | +0.058% | 50/64 | fail: magnitude |
| Slot MSE | Shift -1 | +0.056% | -0.243% | 26/64 | fail |
| Slot MSE | Shift +1 | +0.002% | -0.626% | 38/64 | fail |
| Decoded-field MSE | History only | +0.164% | -0.103% | 38/64 | fail |
| Decoded-field MSE | Episode shuffled | +0.080% | -0.306% | 30/64 | fail |
| Decoded-field MSE | Fit-mean action | -0.014% | -0.222% | 37/64 | fail |
| Decoded-field MSE | Raw-zero action | +0.564% | +0.302% | 53/64 | fail: magnitude |
| Decoded-field MSE | Shift -1 | +0.053% | -0.205% | 21/64 | fail |
| Decoded-field MSE | Shift +1 | -0.035% | -0.581% | 41/64 | fail |

Absolute mean errors were:

| Arm | Standardized slot MSE | Decoded-field MSE |
|---|---:|---:|
| History only | 0.806165 | 0.0000203860 |
| **Aligned** | **0.804418** | **0.0000203525** |
| Episode shuffled | 0.807824 | 0.0000203688 |
| Fit mean | 0.802239 | 0.0000203497 |
| Raw zero | 0.811069 | 0.0000204680 |
| Shift -1 | 0.804871 | 0.0000203632 |
| Shift +1 | 0.804431 | 0.0000203454 |

Raw zero was the only control with a positive simultaneous lower bound, but
its sub-1% effect was far below the registered minimum and zero absolute joint
commands are distributionally atypical. History, mean, shuffled, and timing
controls show that it cannot support a causal-action claim.

## Representation relevance and salience

| Oracle reconstruction reference | Gain | 97.5% lower bound | Favorable clips | Gate |
|---|---:|---:|---:|---|
| Fit-mean component field | +34.697% | +30.524% | 64/64 | pass |
| All-zero component field | +34.645% | +30.507% | 55/64 | pass |

The oracle decoder therefore cleared the substantive 20% relevance bar. Slot
capacity was also sufficient: the tracker retained 98.73% of qualified
candidate mass on average, with 0.999 candidates and 0.923 assigned slots per
transition.

Salience was mixed:

- 448/448 fit target coordinates were active;
- 55/64 fresh clips (85.94%) had a nonnull future slot, below the frozen 90%
  gate;
- 43/64 (67.19%) had an identity present in two adjacent future transitions,
  above the frozen 60% persistence gate;
- fresh clips averaged 6.27 present slot-transitions across the eight-step
  future.

The salience miss independently stops the branch, although the causal family
already failed decisively. It also quantifies a practical limitation: a
near-robot moving-component state is absent in roughly one of seven clips.

## Operational and audit evidence

- Registration-only Slurm job: `507347`, completed 0:0 in 30 seconds.
- Run-only Slurm job: `507374`, B200 `pool0-0025`, completed 0:0 in 59 seconds.
- Peak RSS: 7.84 GiB; disk read: 304.91 MiB.
- Robot rendering: 3.659 ms mean and 4.274 ms p95 over 3,520 post-warmup poses.
- Registration identity:
  `1acfdb4222bac5057a2ccf774cff626f95d9b14fc6a0cde5938bca57c5ff7125`.
- Analysis identity:
  `497e0cd68eaaa7b2cdaf5a1eb0eaad3e91d82360ad06bb3772961cf7115bc619`.
- Completion identity:
  `6250a77dc921939764d5c02b504c96157847c72c74601eebf7afdfc4bf25c258`.
- In-job audit identity:
  `22ad763aa3a59789d5b24fc800d850c8de48e7cc67814515abb65fa6b7470ea6`.
- Independently rerun audit identity:
  `c8ec9aafe25f05bfcb8a909a10ae9a0e1658678fcaf277d906df2ead1c12b8a9`.
- Both audits returned `audit_passed`, independently recomputed all 12 causal
  and two relevance effects, verified 320 provenance and 64 score rows, found
  zero fresh/prior-score overlap, and checked 1,736 explicit false flags.
- Canonical artifact root:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/interaction_event_bottleneck/object-contact-slot-fit256-fresh64-seed20260815-f8e7226-v1`.

## Evidence-based consequence

Do not launch Wan or a deterministic dual-denoising branch from these slots.
The useful oracle state is not enough: at inference the auxiliary branch must
predict it, and aligned planned action supplied no stable advantage over
history or mean action and no timing-specific advantage over adjacent action
windows.

A distinct remaining question is whether the slot future is genuinely
multimodal: a conditional mean can fail even when action changes a compact
conditional distribution. Any such test must use a fixed low-dimensional
target and proper probabilistic scores, treat all 64 outcomes here only as
calibration, and prospectively reserve the still-unopened 31 episodes. It must
not reinterpret this deterministic stop or proceed to a stochastic generator
without a strong fresh distributional pass.
