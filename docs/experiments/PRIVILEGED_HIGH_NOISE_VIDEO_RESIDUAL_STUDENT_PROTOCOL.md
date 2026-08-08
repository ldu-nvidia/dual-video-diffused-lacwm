# Privileged high-noise video-residual student protocol

Date: 2026-08-08

Status: prospective controlled student screen; not implemented or launched;
no deployable improvement is claimed

## Decision this protocol follows

The original NFE-4 teacher gate remains `STOP_NO_ELIGIBLE_TEACHER`: clean
V-JEPA helped every clip at the first pure-noise video update but hurt every
clip at the final low-noise update. A preregistered replication on disjoint
train indices 64--127 then passed at the sole NFE-2 `sigma_video=1` update:
aligned teacher velocity MSE improved 89.949% over feature-off and 80.020% over
episode-shuffled V-JEPA, with 64/64 favorable units.

That pass proves only that a strong sample-specific **training teacher** exists.
It does not prove that its correction is predictable from causal inputs or that
a feature-free model improves.

## Attribution that changes the next experiment

At `sigma_video=1`, no video update precedes the teacher query. The student-
visited state is the same initial Gaussian state used by ordinary forward
noising. Therefore an NFE-2 high-noise screen cannot attribute a gain to
on-policy state coverage. The primary method is the simpler PFD-style detached
teacher-minus-student residual. On-policy/reachability machinery is deferred
until a noninitial student-visited state has a qualified teacher.

Ordinary flow matching already observes the clean-video velocity target during
training. Extra adapter capacity trained directly on that target is therefore
a mandatory stronger control. Without it, a gain from the privileged arm could
be ordinary residual fitting rather than V-JEPA structure.

## Information contract

At training only, use the frozen J1 checkpoint to compute

\[
v_S=v_{base}(z_1,h,a,\varnothing),\qquad
v_T=v_{base}(z_1,h,a,u^*),\qquad
r_T=\operatorname{sg}(v_T-v_S),
\]

where `z_1` is initial video Gaussian noise and `u*` is clean target-video
V-JEPA. A small zero-initialized causal adapter predicts `r_hat` from only the
student-side hidden state, observed history, actions, noise, and time. The
deployed velocity is `v_S+r_hat`. The teacher, V-JEPA extractor, target cache,
future RGB, and target-derived gate are absent from validation and inference.

All feature-free evaluations must hard-assert zero teacher/cache calls and
byte-audit their input graph. The normal clean video may remain evaluator-owned
for metrics after sampling.

## Controlled arms

Every arm starts from one parent snapshot and uses the same data order, noise,
optimizer, updates, trainable backbone subset, adapter architecture, and
inference calls.

| Arm | Adapter training target at sigma 1 | Purpose |
|---|---|---|
| `BASE-PM` | zero target; adapter instantiated and held at exact zero | parameter/interface control |
| `DIRECT-RESIDUAL` | stopped ordinary `v* - v_S` | tests whether extra causal residual capacity is sufficient without V-JEPA |
| `PFD-ALIGNED` | stopped aligned `v_T - v_S` | primary privileged semantic teacher |
| `PFD-SHUFFLED` | stopped episode-shuffled-teacher residual | sample-specific V-JEPA attribution |
| `PFD-GATED` | aligned residual only where `e_T < e_S`, training-only | tests whether rejecting harmful teacher units matters |

Do not label any arm `OPD` at this initial state. If `DIRECT-RESIDUAL` matches
`PFD-ALIGNED`, the supported mechanism is added residual capacity/optimization,
not privileged semantic transfer. If `PFD-ALIGNED` matches `PFD-SHUFFLED`, the
clean feature is not sample-specifically useful to the causal student.

## Training and data separation

- Keep eligibility clips 0--127 out of loss-weight selection and outcome
  reporting. Use remaining training episodes for optimization and a separate
  train-only calibration slice for numerical stability.
- The already reused val64 split may serve only as an exploratory paired
  development screen. Do not open the protected split.
- Normalize residual loss by a train-frozen target RMS and use Smooth-L1 or
  clipped MSE so the large observed teacher/student distance does not dominate
  ordinary flow learning.
- Log ordinary flow loss, residual loss, adapter norm, teacher advantage,
  admitted fraction, gradient norms, train GPU-hours, and all teacher calls.
- W&B must remain in the user's private project with no organizational group.

## Evaluation and decision

Evaluate feature-free NFE 1 and 2 with the same seeds and total Wan calls.
Primary comparison is `PFD-ALIGNED` versus both `BASE-PM` and
`DIRECT-RESIDUAL`; `PFD-SHUFFLED` is the attribution control. Positive means
lower error.

Advance only if one preregistered NFE satisfies all of:

1. decoded and temporal MSE improve at least 3% over both `BASE-PM` and
   `DIRECT-RESIDUAL`, with paired 95% lower bounds above 1%;
2. latent NMSE has a nonnegative point effect and lower bound above -1%;
3. decoded and temporal MSE beat `PFD-SHUFFLED` by at least 1%;
4. aligned-versus-shuffled and aligned-versus-zero action sensitivity do not
   regress relative to `BASE-PM`;
5. inference makes exactly the baseline number of Wan calls, zero teacher/
   cache calls, and adds no material end-to-end latency; and
6. no metric is selected from the protected split.

A point-only gain or an oracle-only gain is `NO_GO`. A passing one-seed val64
screen authorizes multi-seed confirmation, not a paper claim. Few-step
distillation begins only after the feature-free student itself is a stronger
causal teacher than the current NFE-1 frontier.

## Novelty boundary

[Privileged Foresight Distillation](https://arxiv.org/abs/2604.25859) already
uses a future-aware teacher and detached residual adapter for causal robot
**action denoising**. The residual-adapter idea is not novel. A contribution
would require a demonstrated target-video semantic teacher for low-NFE robot
**video velocity**, advantage over direct-residual and shuffled controls,
feature-free serving, and preferably closed-loop world-model utility. At the
initial pure-noise state, no on-policy novelty can be claimed.

