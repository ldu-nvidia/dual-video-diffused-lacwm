# Privileged teacher high-noise NFE=2 calibration protocol

Date frozen: 2026-08-08

Status: **preregistered; not launched; no student training authorized.**

## Why this exists

The frozen NFE=4 eligibility probe remains `STOP_NO_ELIGIBLE_TEACHER` at result
commit `7546e33ebf8cb5400aebc596637efbda42409ec2` and analysis identity
`160012e5777ffaa4f52d2c95cbf3d1b33f6d04d8c84a93086b7cf40496990fbe`.
It scored two video-active states: at step 2 (`video_sigma=1`) the aligned clean
V-JEPA teacher was better than feature-off on 64/64 clips, while at step 3
(`video_sigma=0.024414`) it was better on 0/64 and had nearly zero
sample-specific effect. Because that split was observed after the frozen gate,
this NFE=2 experiment is a new mechanistic follow-up. It cannot reinterpret,
replace, or overwrite the NFE=4 STOP.

## Frozen question

Does the same clean-future V-JEPA teacher pass the **unchanged** eligibility
thresholds when a faithful NFE=2 cascade exposes exactly one auxiliary-only
call followed by exactly one video-active call at `video_sigma=1`?

This is teacher eligibility only. It does not train a student, test a deployable
condition, or establish an inference-time quality gain.

## Frozen design

- Checkpoint, resolved config, immutable ABC/V-JEPA train cache, aligned teacher,
  episode-shuffled teacher, flow target, and production-off roll-in are identical
  to the NFE=4 probe.
- Data are the 64 episode-disjoint **train** clips at manifest indices 64--127.
  These are disjoint from the adaptively inspected indices 0--63. Validation,
  test, and lockbox data remain forbidden.
- NFE is 2 with cascade fraction 0.5. Runtime must observe one auxiliary-only
  call and one video-active call, whose video sigma must equal 1.0 exactly within
  `1e-8`; otherwise the run fails closed.
- Probe/bootstrap seed is `20260809`; paired bootstrap replicates are 10,000.
- The output must be fresh and named
  `privileged-opd-high-noise-nfe2-train64-index64-seed20260809-<commit7>-v1`.
- W&B is disabled. The run is non-array, non-requeueable, one B200, and performs
  no optimization.

## Unchanged conjunctive gate

All conditions must pass:

1. aligned teacher is better than feature-off on at least 60% of scored units;
2. aligned vs feature-off velocity-MSE gain is at least 5%, with positive paired
   clip-bootstrap lower bound;
3. aligned vs episode-shuffled gain is at least 3%, with positive lower bound;
4. the aligned full rollout has positive decoded-MSE point gain over off; and
5. the aligned full rollout has positive temporal-MSE point gain over off.

Pass authorizes only design of a separate controlled student screen. Failure
stops this high-noise teacher route. Neither outcome changes the original NFE=4
decision.

## Implementation and launch boundary

The immutable launcher is
`tools/slurm/probe_privileged_on_policy_teacher_high_noise_nfe2.sbatch`. It
fixes indices, NFE, seed, thresholds, clip count, and profile. The probe records
source/input hashes, parent-result lineage, selected clip identities, raw-row
identities, phase counts, active sigma nodes, forward-call accounting, and
explicit `student_training_launched=false`.

This commit only preregisters the experiment. **Do not submit it without a
separate launch instruction.**
