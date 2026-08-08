# Generated-motion PhaseLock probe result

Date: 2026-08-08

## Decision

The frozen validation-only probe did **not** demonstrate a deployable,
sample-specific generated-motion benefit. All four fixed candidates improved
over an ordinary sampler with the same total number of transformer calls, but
none improved over the existing one-call frontier and none passed the aligned
versus shuffled-prior attribution gate. No candidate was selected and the
protected test must remain unopened.

This is a negative result for one training-free adaptation of PhaseLock to the
frozen VPM checkpoint. It does not rule out a learned motion prior, a different
backbone, or jointly trained coarse/motion forcing.

## What was tested

For each of 64 immutable validation clips, a short preliminary trajectory was
generated from history, actions, and Gaussian noise. Its future video-latent
frame differences were stopped and injected during the first half of a second
trajectory with fixed strength `0.05`. The sampler never opened a clean
auxiliary target, received future RGB, or called an online teacher.

The four fixed candidates used preliminary/full call allocations `(1,2)`,
`(1,3)`, `(2,2)`, and `(2,4)`, for total transformer-call budgets `3`, `4`,
`4`, and `6`. Each was compared with:

- an ordinary sampler at the same total call budget;
- the established ordinary one-call frontier; and
- the same guided sampler with the generated motion prior cyclically shuffled
  between validation samples.

The primary gate required at least 1% paired improvement in decoded temporal-
difference MSE, both in the point estimate and a one-sided simultaneous lower
bound. Video-latent NMSE and decoded MSE could regress by no more than 1%.
Twelve candidate/control contrasts shared a 10,000-sample paired bootstrap,
seed `20260807`, with Bonferroni familywise alpha `0.05`.

## Frozen result

| Candidate | Calls | Temporal gain vs equal-call ordinary | Latent-NMSE gain vs equal-call ordinary | Temporal gain vs one-call | Temporal gain vs shuffled prior | Decision |
|---|---:|---:|---:|---:|---:|:---:|
| `k1_f2` | 3 | 7.80% (LB 6.44%) | 12.03% (LB 10.17%) | -0.033% | -0.017% | fail |
| `k1_f3` | 4 | 6.85% (LB 5.73%) | 8.69% (LB 7.29%) | -8.55% | -0.617% | fail |
| `k2_f2` | 4 | 14.16% (LB 12.12%) | 19.71% (LB 17.20%) | -0.035% | -0.020% | fail |
| `k2_f4` | 6 | 9.44% (LB 7.72%) | 9.11% (LB 7.05%) | -16.88% | -1.47% | fail |

Ordinary sampling itself degraded with more Euler calls: video-latent NMSE was
`0.24027`, `0.27341`, `0.29960`, and `0.33096` at budgets `1`, `3`, `4`, and
`6`. The best guided candidates at budgets three and four returned close to
the one-call result (`0.24052` and `0.24054`) but did not surpass it.

Crucially, aligned and shuffled priors were effectively indistinguishable.
For `k1_f2`, decoded temporal MSE was `0.01328389` with the aligned prior and
`0.01328161` with the shuffled prior. For `k2_f2`, the corresponding values
were `0.01328424` and `0.01328156`. Every attribution confidence bound crossed
or fell below zero.

## Interpretation

The apparent equal-call improvement is not evidence that the generated future
motion belongs to, or helps, its own sample. The same effect survives when the
prior comes from a different clip. In this checkpoint, the guidance operation
mainly arrests a harmful multi-step trajectory and recovers approximately the
existing one-call quality; it does not supply useful sample-specific dynamics.

The immediate implication is that spending extra inference calls to extract a
motion prior from this frozen model is not justified. The more promising tests
are to improve the motion state during training, generate an earlier-maturing
low-frequency/motion stream jointly from noise, or distill the strong endpoint
map directly into the requested one- or two-call sampler.

## Evidence

- Tool source: `a9c37fab24a143400f50a72d0f0be0564c37785d`
- Frozen model source: `9cf8e6922f35a5d6645e3128545953723bf54da2`
- Frozen VPM snapshot SHA-256:
  `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`
- Slurm job `504887`: `COMPLETED 0:0` on eight B200s in `00:09:33`
- Evaluation inventory: 768 unique rows (`64 clips x 12 endpoints`), all
  expected clip/endpoint keys present exactly once, exact paired inputs and
  initial-noise checks passed
- Registration identity:
  `25194940073a38513fb6ca00bd38ddcd15f8f672780ed406aee6d6a1b2ed5254`
- Inventory identity:
  `f31e704e59b865ce883d3af26841d37bfc9e2b0c39237d19255fabc8e57a9f97`
- Analysis identity:
  `a83ed4cdf38bdee22655938318783b1807bcdb1a77f33a1ec9f197987e9ddd78`
- Protected test accessed: `false`
- Cluster artifact:
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/deployable_motion_prior/vpm-phaselock-validation-seed20260807-a9c37fa-v1`

