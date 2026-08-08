# Privileged on-policy teacher eligibility: first-64 train result

Date: 2026-08-08

Status: **STOP — the frozen teacher-eligibility gate failed.** No student was
trained, no validation or protected-test sample was opened, and no deployable
quality improvement is claimed.

## Result in one sentence

At the same NFE-4 causal-student-visited states, the aligned clean-V-JEPA
teacher reduced mean video-velocity MSE by **18.004%** versus the feature-off
student and by **9.681%** versus the episode-shuffled teacher, but it was better
than the student on only **64/128 = 50%** of clip/timestep units. The
preregistered minimum was 60%, so the route stops before student training.

## Frozen question and design

The probe asked only whether the existing J1 checkpoint supplies a sufficiently
better privileged teacher to justify a later on-policy distillation experiment.
It did not test distillation.

- Checkpoint: faithful-cascade `J1`, update 1000, trained at commit
  `656086686dae723c942a4209a9d71cdb17ed6ccc`.
- Probe code: `a375870ad9daa940085b525305ec1c386debba39`.
- Data: immutable ABC/V-JEPA **train** cache, indices 0--63. All 64 clips came
  from distinct episodes. The resolved validation and test loaders were never
  instantiated.
- Sampler: NFE 4 with the observed schedule split of two auxiliary-only calls
  followed by two video-active calls. The two scored states were rollout steps
  2 and 3.
- Student roll-in: the production `off` semantics from the checkpoint: a
  causal generated-feature prefix, then V-JEPA state and clock conditioning
  hard-off during video-active calls. The manual roll-in was required to equal
  the production `off` video and auxiliary final tensors after evidence
  serialization.
- Teacher queries: the same checkpoint at the **same student video and
  auxiliary states**, once with scheduled clean target-video V-JEPA and once
  with the next batch item's clean V-JEPA. Donors were episode-disjoint and
  retained the receiving sample's local auxiliary noise and clock.
- Target: the ordinary rectified-flow video target, fixed as initial video
  noise minus clean video latent, scored on future latent slots only.
- Statistics: 64 clip clusters, 128 clip/timestep units, 10,000 paired bootstrap
  replicates. Both visited timesteps remained together in every aggregate
  clip resample.

All 64 aligned clean-feature hashes were unique. Every aligned/shuffled pair
had different feature hashes and a different donor episode. Expected and
observed Wan forward calls were both 640.

## Frozen gate

Positive percentages mean lower MSE for the aligned teacher.

| Requirement | Observed | Paired 95% interval | Result |
|---|---:|---:|---|
| Aligned teacher better on at least 60% of clip/timestep units | 50.000% | not applicable | **FAIL** |
| Aligned vs feature-off velocity-MSE gain at least 5%, positive lower bound | 18.004% | [17.742%, 18.264%] | pass |
| Aligned vs episode-shuffled velocity-MSE gain at least 3%, positive lower bound | 9.681% | [8.970%, 10.398%] | pass |
| Aligned full rollout improves decoded MSE over off | 80.853% | [79.421%, 82.261%] | pass |
| Aligned full rollout improves temporal MSE over off | 68.133% | [64.896%, 71.364%] | pass |

Decision: `STOP_NO_ELIGIBLE_TEACHER`. The unit-coverage condition is conjunctive;
large average or oracle-rollout gains cannot override it.

## Full-rollout oracle diagnostic

These NFE-4 rollouts use clean target-video features and are explicitly
nondeployable. Decoded metrics compare VAE-decoded predictions with the
original ground-truth future RGB converted to unit range; temporal MSE compares
differences across the eight future frames.

| Metric (lower is better) | Off mean | Aligned mean | Shuffled mean | Aligned vs off | Aligned vs shuffled |
|---|---:|---:|---:|---:|---:|
| Future latent NMSE | 2.516992 | 0.249541 | 1.314691 | 90.086% [89.632%, 90.524%] | 81.019% [79.410%, 82.435%] |
| Decoded MSE | 0.068797 | 0.013172 | 0.072447 | 80.853% [79.421%, 82.261%] | 81.818% [80.072%, 83.437%] |
| Temporal-difference MSE | 0.047430 | 0.015114 | 0.045740 | 68.133% [64.896%, 71.364%] | 66.956% [62.921%, 70.805%] |

The shuffled control establishes that the oracle rollout gain is
sample-specific, not merely a consequence of injecting any clean V-JEPA tensor.
It does not establish that a causal student can recover the missing information.

## Why the favorable-unit fraction is exactly 50%

The aggregate hides a complete timestep split. The following timestep analysis
is diagnostic and post hoc; it was not used to change the frozen decision.

| Student-visited state | Video sigma | Aligned better than student | Student / aligned / shuffled velocity MSE | Aligned vs student | Aligned vs shuffled |
|---|---:|---:|---:|---:|---:|
| Step 2, first video update | 1.000000 | 64/64 (100%) | 1.731481 / 0.165469 / 0.868974 | 90.444% [89.997%, 90.867%] | 80.958% [79.329%, 82.455%] |
| Step 3, last video update | 0.024414 | 0/64 (0%) | 6.296369 / 6.417073 / 6.419097 | -1.917% [-1.986%, -1.847%] | 0.032% [-0.053%, 0.115%] |

Both video-active calls had auxiliary sigma 0: the teacher received the clean
V-JEPA condition. At the pure-noise video state, that condition was strongly
useful and sample-specific. Near the clean endpoint, aligned conditioning made
the local flow target slightly worse for every clip and was statistically
indistinguishable from shuffled conditioning.

The teacher/student velocity-distance mean also collapsed from 1.244455 at
step 2 to 0.002189 at step 3. Thus the late teacher is close but directionally
unhelpful; the early teacher is far but highly advantageous. Counting both
states equally gives 64 favorable units out of 128. Averaging squared error
instead gives a positive result because the step-2 reduction (about 1.566 MSE)
is much larger than the step-3 degradation (about 0.121 MSE).

The full oracle rollout can simultaneously improve dramatically because its
first video update changes the entire downstream trajectory. That is evidence
that privileged information is valuable early, not evidence that the current
teacher is uniformly safe to imitate or that its information is causally
recoverable.

## Interpretation and next decision

The defensible finding is narrower than “privileged distillation works”:

1. clean target-video V-JEPA defines a strong, genuinely sample-specific
   high-noise video-velocity teacher;
2. the same teacher is not better at the final NFE-4 video-active state;
3. therefore the frozen all-video-state teacher fails eligibility, and neither
   unrestricted OPD nor a student-quality claim is justified.

An early-state-only residual teacher is a plausible new hypothesis, but it is
selected after seeing these 64 clips. It must be preregistered and rechecked on
a disjoint train-only calibration set before any optimization. If that fresh
gate passes, the first student screen should include:

- an ordinary continuation baseline;
- an aligned early-state residual adapter;
- an otherwise identical episode-shuffled residual adapter; and
- a `PFD-VIDEO` baseline that predicts the detached teacher-minus-student video
  velocity residual with a zero-initialized small causal adapter.

No validation or lockbox data should be opened for that eligibility check.
Student inference must remain feature-free, with teacher/cache call count zero.

## Relationship to current primary work

[Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation](https://arxiv.org/html/2607.24731)
supports querying teacher and student at student-visited states and, in its
dense-to-sparse video-control study, supervises an early subset of the rollout.
Its main Positive--Direction Matching contribution addresses ambiguity between
positive and negative CFG branches. The present LACWM checkpoint does not expose
that two-branch CFG interface, so PDM is not the eligibility mechanism here.

[Privileged Foresight Distillation](https://arxiv.org/html/2604.25859) is even
closer: it uses a shared-backbone future-aware teacher, detaches the
teacher-minus-current-only action-denoising residual, trains a small
zero-initialized causal adapter, discards the teacher at inference, and includes
a shuffled-future control. Consequently, broad “future-aware teacher to causal
student” or “privileged denoising residual” novelty is already occupied. A
remaining contribution would have to be demonstrated specifically for
target-video/V-JEPA-conditioned **video velocity**, student-visited low-NFE
robot-video generation, and downstream closed-loop utility.

Official implementations inspected:

- OPD/PDM: <https://github.com/rethinking-cfg-opd/Rethinking-CFG-OPD>, commit
  `46d1e5afa49d01d09bd3e9f293afa22a402ab425`.
- PFD: <https://github.com/PengchengFang-cs/PFD>; released implementation commit
  `338f00b` confirms the detached residual and causal adapter construction.

## Audit and immutable evidence

- Slurm job: `506047`, B200 `pool0-0158`, completed in 2m50s with exit code 0.
- Artifact root:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/privileged_on_policy_teacher/privileged-opd-train64-seed20260808-a375870-v1`
- Registration identity:
  `ed3016345b8bd0dae7ff518858ccff3c923beb2de74d8c0d8379bea6243a2d70`.
- Analysis identity:
  `160012e5777ffaa4f52d2c95cbf3d1b33f6d04d8c84a93086b7cf40496990fbe`.
- Completion identity:
  `0392930de33caf4591ee65b6f62636fa48aa7aab75ed3cfd4c3cfef024de5832`.
- Independent audit identity:
  `2bc637e3f8b66b75d25c3f8d22652175e39ce8220a3f420ae7c7b3a130c2afbd`.
- Raw unit rows SHA-256:
  `71041332ee0b5fe76e62c247ff8b4efe06132638f7a40a53d5d2070ae49c397a`.
- Raw rollout rows SHA-256:
  `5fd4c8a6dec9f40efcd8966558cf5429e14254693b04a3207c36d11b55013281`.
- Analysis file SHA-256:
  `0d32067b164c6a612f6eb63bcc362e9c5fea369dd85eaf0fa6088d6385b43651`.
- Snapshot SHA-256:
  `70d79533460680d836c7178baea2232cd4b8f309146a7571db0002f91ce53f61`.
- Resolved config SHA-256:
  `88ef78f4aa12050f50f393b1732c7804b39047513da7d7486cd932d4d2525a0d`.
- Train manifest SHA-256:
  `eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74`.

The audit independently reproduced the analysis from identity-hashed raw rows,
rehash-verified every registered input and output, confirmed the 2+2 cascade
phase inventory, and recorded `protected_test_opened=false`,
`validation_opened=false`, and `student_training_launched=false`.
