# Stage-0 causal action-to-motion proxy

## Result in one sentence

On the immutable ABC train512/validation64 split, planned actions improved prediction of future low-resolution motion summaries beyond observed history, and the gain disappeared under episode-shuffled actions. This supports an inference-valid learned motion scaffold, but it does not yet establish a video-generation gain.

## Causal contract

- Predictor input: observed RGB frames 0--4 and planned action chunks 4--11.
- History feature: Farneback flow summaries for observed transitions 0->1 through 3->4.
- Target/scoring only: future RGB flow summaries for transitions 4->5 through 11->12.
- Target geometry: 8 transitions x 3 views x 4 spatial cells x 4 statistics = 384 dimensions. Statistics are mean dx, mean dy, mean magnitude, and 75th-percentile magnitude.
- Action geometry: `[N,8,5,23]`; train-only variance retained coordinates 0--13, the 560 active values were train-standardized and compressed to PCA64.
- Models: independently train-tuned multi-output ridge regressions for history-only (192 inputs) and history+action (256 inputs).
- Controls: aligned actions, deterministic different-episode shuffled actions, and the train-mean action.
- Validation contains 64 unique episodes; train contains 512 unique episodes; overlap is zero. No protected test or cached V-JEPA target was opened.

The exact executed source is preserved byte-for-byte at `tools/stage0_action_motion_proxy.py`, SHA-256 `3be829af2fc71c83e211709fc28fd9c101bcabecad9a134b5d92ef17aff21818`.

## Registered exploratory result

Positive values favor aligned actions.

| Paired validation contrast | Improvement | Paired bootstrap 95% CI | Favorable clips | Registered gate |
|---|---:|---:|---:|---|
| Aligned action vs history-only | 6.5882% | [4.9581%, 8.0550%] | 59/64 | Pass |
| Aligned vs episode-shuffled, same action model | 9.0430% | [7.1863%, 11.0997%] | 57/64 | Pass |
| Aligned vs train-mean action, same action model | 6.4819% | [4.9141%, 7.9076%] | 58/64 | Diagnostic pass |

Mean standardized validation MSE was 0.957815 for history-only, 0.894713 for aligned actions, 0.983666 for shuffled actions, and 0.956727 for train-mean actions. Aggregate standardized R2 rose from 0.09827 to 0.15767 with aligned actions.

## Exploratory gate versus downstream bar

The executed protocol preregistered a deliberately permissive Stage-0 gate: at least 1% all-view improvement with a positive paired-bootstrap lower bound for both incremental prediction and shuffled-action specificity. It passed.

A later downstream protocol proposes a stricter **10% all-view** improvement requirement. The all-view result is 6.5882%, so that stricter gate **fails**. This distinction must not be blurred.

The top-camera diagnostic is stronger: aligned actions improve top-view error by 12.6083%, paired bootstrap 95% CI [10.2324%, 15.0794%], favorable on 57/64 clips. That clears a top-view-only 10% threshold, but it cannot substitute for the failed all-view gate. Left- and right-wrist improvements are 3.5681% and 3.1767%, respectively.

Signed flow is where most of the gain lies: dx improves 9.3403% and dy 9.8274%, while mean magnitude and q75 magnitude improve 2.8155% and 2.6861%. All eight forecast horizons improve; horizon gains range from 1.45% to 10.60%.

## Artifact identities

- Registration: `830adba27f3e0573210a83ddefcbd987b7ca886f29ee8049a3ea60efecdcbe1d`
- Analysis: `077d75e3be29e8ad7cd8aaff5141b9303d51ef4921023ddec3e04ee12ed2a2d7`
- Completion: `88378787f6ea0959671be609e86524a480a3f43193d1824ddf32770f69f365da`
- Artifact root: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/action_flow_proxy_stage0/action-flow-proxy-stage0-seed1234-20260808-v1`

Run `python tools/audit_stage0_action_motion_proxy.py <artifact-root>` to recompute JSON identities, artifact hashes, row count, and protected-test flags.

## Interpretation and next experiment

The result establishes that history plus requested action contains deployable information about future signed image motion. It does not establish that the current 2x2 summary is a sufficient conditioning signal or that a generator will use it.

The next controlled study should predict a multi-scale dense `(u,v,occlusion,confidence)` scaffold from history+actions, train the video generator on a mixture of predicted, corrupted, and dropped scaffolds, and compare aligned, shuffled, zero, hard-masked, and oracle scaffolds. The primary gate should be all-view and should be fixed before generator training.

## Limitations

- One seed and a previously reused exploratory validation64 split.
- Farneback targets mix robot, object, background, and wrist-camera ego-motion.
- Shuffling diagnoses action specificity but is not a randomized physical intervention.
- The ridge target is a low-resolution summary, not a dense rendered robot field.
- No FVD, perceptual quality, generator training, latency, or long-rollout metric was measured.
