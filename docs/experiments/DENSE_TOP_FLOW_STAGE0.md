# Dense top-view action-to-flow Stage-0

## Result in one sentence

The earlier coarse top-camera improvement of 12.61% did **not** survive as a comparable dense directional-flow gain: aligned actions improved dense MSE by 2.91% over observed history and 2.94% over shuffled actions, so this preregistered follow-up is **NO-GO** for a generator handoff. Actions nevertheless carry a statistically clear signal about future motion direction.

## Frozen causal protocol

- Predictor input: observed top-view RGB frames 0--4 and planned action chunks 4--11 only.
- History feature: raw Farneback `(u,v)` for observed transitions 0->1 through 3->4, computed at 80x45, downsampled with displacement rescaling to `[4,2,12,20]` (1,920 values), then train-only PCA64.
- Action feature: `[8,5,23]` planned actions; train-active coordinates 0--13 produce 560 standardized values, then train-only PCA64.
- Target/scoring only: raw top-view Farneback `(u,v)` for future transitions 4->5 through 11->12, represented as `[8,2,12,20]` (3,840 values), then train-only centered PCA192.
- Models: independently train-tuned multi-output ridge regressions for history-only and history+action. Five-fold train-only CV selected alpha 10,000 for both.
- Controls: native aligned actions, deterministic different-episode shuffled actions, and the train-mean/zero action.
- Splits: immutable ABC train512/validation64, with zero episode overlap. This reused exploratory validation split contains 64 unique episodes.
- Forbidden inputs: no future RGB or future-derived feature entered the predictor. No protected test, cached V-JEPA target, video generator, or generator checkpoint was opened.
- Primary preregistered gate: at least 10% dense-MSE improvement for aligned versus history-only and aligned versus shuffled, with a nonnegative paired-bootstrap lower bound. All gate criteria were required.

The exact executed source is preserved byte-for-byte at `tools/stage0_dense_top_flow_proxy.py`, SHA-256 `70f8cf1ba9f1246745ecceb0e8e7b35f94b3b52c3370d4f6e33790f54c80a1a8`.

## Validation result

Positive values favor aligned actions. Every confidence interval is a 10,000-sample paired episode bootstrap.

| Paired validation contrast | Dense MSE improvement | Endpoint-error improvement | Directional-cosine gain |
|---|---:|---:|---:|
| Aligned vs history-only | 2.9096% [2.4917, 3.4009] | 1.7079% [1.4378, 1.9980] | +0.17540 [+0.14277, +0.21061] |
| Aligned vs episode-shuffled | 2.9373% [2.4144, 3.6002] | 2.2422% [1.7630, 2.8523] | +0.17534 [+0.13940, +0.21486] |
| Aligned vs train-mean/zero | 2.9259% [2.5119, 3.4102] | 1.7104% [1.4329, 2.0027] | +0.17638 [+0.14378, +0.21188] |

Aligned actions reduced dense MSE for 64/64 clips relative to history-only and 63/64 relative to shuffled actions. Mean cosine rose from 0.04014 to 0.21555 versus history-only. Thus the action signal is real and action-specific, but its dense field accuracy is far below the fixed 10% bar.

Target PCA retained 98.596% of train variance. On validation, oracle PCA reconstruction achieved dense MSE 0.002173, endpoint error 0.03635, and cosine 0.91085. Compression therefore leaves headroom but is not a plausible explanation for the small aligned-action effect. The history+action feature-to-dense-field predictor had 0.389 ms p95 batch-one CPU latency; that number excludes RGB decode, Farneback extraction, and data loading.

## Why this differs from the coarse result

The coarse diagnostic predicted 2x2-cell summaries (mean `dx`, mean `dy`, mean magnitude, and magnitude q75). That representation rewards learning the broad signed direction of motion, where planned actions are informative. This follow-up asks the harder question: can the same inputs reconstruct a 20x12 directional field at eight horizons? The large cosine gain but modest MSE/EPE gain says the model learns overall direction much better than local displacement magnitude and spatial structure. The coarse +12.61% and dense +2.91% values are therefore evidence about different targets, not contradictory estimates of one effect.

## Post-hoc horizon and motion-magnitude diagnostics

These slices were requested after the registered result and do not change the decision. Cosine is computed per clip-horizon and then averaged. No confidence interval below is multiplicity-adjusted.

| Future transition | Target mean magnitude | MSE gain vs history | EPE gain vs history | Cosine gain vs history |
|---|---:|---:|---:|---:|
| 4->5 | 0.0739 | 1.66% | 0.55% | +0.031 |
| 5->6 | 0.0746 | 3.06% | 1.80% | +0.146 |
| 6->7 | 0.0751 | 3.02% | 2.09% | +0.145 |
| 7->8 | 0.0611 | 2.80% | 1.86% | +0.170 |
| 8->9 | 0.0538 | 4.85% | 2.52% | +0.195 |
| 9->10 | 0.0561 | 4.06% | 2.19% | +0.194 |
| 10->11 | 0.0747 | 3.14% | 1.62% | +0.184 |
| 11->12 | 0.0819 | 2.24% | 1.38% | +0.142 |

Every horizon is favorable on MSE and EPE, but no horizon reaches even 5% MSE improvement. The effect is not confined to one transition, and it does not grow monotonically with prediction horizon.

For a second descriptive slice, the 512 validation clip-horizon units were divided into target-motion quartiles (cut points 0.0161, 0.0371, and 0.0896):

| Target-motion bin | Mean magnitude | MSE gain vs history | MSE gain vs shuffled | EPE gain vs history | Cosine gain vs history |
|---|---:|---:|---:|---:|---:|
| Q1 low | 0.00783 | 4.10% | 8.97% | 1.43% | +0.028 |
| Q2 | 0.02550 | 4.56% | 5.46% | 2.05% | +0.102 |
| Q3 | 0.05804 | 3.88% | 3.59% | 1.99% | +0.165 |
| Q4 high | 0.18422 | 2.76% | 2.78% | 1.58% | +0.308 |

Direction becomes much more action-predictable as motion increases, but relative dense MSE/EPE improvement does not: the high-motion quartile has the largest cosine gain and only 2.76% MSE gain. Its absolute MSE reduction is largest because its errors are much larger, yet considerable local residual remains. This pattern is consistent with actions explaining broad kinematic direction while local geometry, occlusion, contact/object motion, or flow noise determines much of the dense field. It cannot isolate contact motion because the cache has no contact, robot, object, or background masks.

Run `python tools/diagnose_dense_top_flow_stage0.py <artifact-root>` to reproduce these read-only post-hoc slices. The script writes nothing into the sealed run.

## Decision and implication

The two 10% dense-MSE point gates fail, while the nonnegative EPE/cosine and target-PCA gates pass. Under the frozen all-required rule the decision is **NO-GO**. This result does not justify launching a video generator conditioned on this ridge-predicted flow field.

It does support a narrower research hypothesis: planned actions can provide a cheap motion-direction prior, but local geometry or learned visual dynamics is still needed to predict a sufficiently accurate dense scaffold. A future experiment should improve that causal scaffold directly (for example, object/robot masks, calibrated kinematics, occlusion/confidence, or a learned history+action dynamics encoder) and must clear a new frozen dense-field gate before generator work.

## Sealed identities

- Artifact root: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/action_flow_proxy_dense_top_stage0/dense-top-flow-stage0-seed1234-20260808-v1`
- Registration: `1d2f45fbd7209b94fff8b0703e3b5c670d834acb0fd3a3dc991feae13b39bf5b`
- Analysis: `0cce95d9fd971dbae243ebd10d078225a759d6dad696a9591be0d4b72c94c8df`
- Completion: `e2dbe55dce37159cd23c6dbd4fe29cc28047496f62eb1d1ed7cb9b8239e7eb1e`

Run `python tools/audit_dense_top_flow_stage0.py <artifact-root>` to recompute all JSON identities and artifact hashes, verify the exact protocol and source, confirm 64 unique validation rows with different-episode donors, and reject any protected-test flag.

## Limitations

- One seed and a previously reused exploratory validation64 split; the intervals do not capture split or training-seed uncertainty.
- Top view only; left/right wrist views and an all-view dense target were not evaluated.
- Farneback is a pseudo-label, not ground-truth 3D scene flow, and mixes robot, object, background, lighting, and camera effects.
- The 20x12 target is low-resolution, and PCA192 constrains predictions to a train-derived linear subspace.
- Ridge is a deliberately cheap screen, not a learned nonlinear dynamics model.
- Latency excludes online RGB-to-history-flow extraction.
- No video-generator training, FVD, perceptual metric, task success, or rollout stability was evaluated.
