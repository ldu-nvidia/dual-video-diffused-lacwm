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

The J1 parent is not the current video-only frontier. In the prior controlled
study, its update-1000 video-flow loss was 31.6% worse than VPM, and at NFE 1
its validation decoded MSE was 0.0199946 versus 0.0191015 for VPM. Therefore a
student that improves only over J1-off is not sufficient; it must also beat the
existing VPM@1 endpoint under a matched fresh evaluation.

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

## Causal-compressibility preflight

Before a full continuation, freeze J1 and collect only optimization-split rows
at `sigma=1`: the inference-visible off hidden state, direct residual
`v*-v_S`, and privileged residual `v_T-v_S`. Fit identical small probes to the
direct, aligned-privileged, shuffled-privileged, and zero targets. Evaluate on
episode-disjoint development rows with zero teacher/cache calls. Report
held-out residual R2/cosine, corrected velocity MSE, one-step latent/decoded/
temporal quality, latency, and the comparison with VPM@1. If aligned privileged
prediction does not beat both direct and shuffled prediction, stop before the
expensive student continuation: the oracle correction is not shown to be
causally compressible.

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

The five attribution arms below start from the same J1 parent snapshot and use
the same data order, noise, optimizer, updates, trainable backbone subset,
adapter architecture, and inference calls.

| Arm | Adapter training target at sigma 1 | Purpose |
|---|---|---|
| `J1-OFF-PM` | zero target; adapter instantiated and held at exact zero | within-J1 parameter/interface control |
| `J1-DIRECT-RESIDUAL` | stopped ordinary `v* - v_S` | tests whether extra causal residual capacity is sufficient without V-JEPA |
| `J1-PFD-ALIGNED` | stopped aligned `v_T - v_S` | primary privileged semantic teacher |
| `J1-PFD-SHUFFLED` | stopped episode-shuffled-teacher residual | sample-specific V-JEPA attribution |
| `J1-PFD-GATED` | aligned residual only where `e_T < e_S`, training-only | tests whether rejecting harmful teacher units matters |

The outcome table must also include the frozen `VPM@1` frontier, evaluated on
the identical clips and seeds. Preferably add a fresh `VPM-DIRECT-RESIDUAL`
continuation with matched adapter capacity and optimization budget. This
separates recovery of a damaged J1 parent from an actual improvement over the
best existing video-only model.

Do not label any arm `OPD` at this initial state. If `J1-DIRECT-RESIDUAL`
matches `J1-PFD-ALIGNED`, the supported mechanism is added residual
capacity/optimization, not privileged semantic transfer. If `J1-PFD-ALIGNED`
matches `J1-PFD-SHUFFLED`, the
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
Primary within-parent comparison is `J1-PFD-ALIGNED` versus both `J1-OFF-PM`
and `J1-DIRECT-RESIDUAL`; `J1-PFD-SHUFFLED` is the attribution control. The
deployable endpoint must additionally beat frozen `VPM@1` and, if trained,
`VPM-DIRECT-RESIDUAL`. Positive means lower error.

Advance only if one preregistered NFE satisfies all of:

1. decoded and temporal MSE improve at least 3% over both `J1-OFF-PM` and
   `J1-DIRECT-RESIDUAL`, with paired 95% lower bounds above 1%, and improve
   over the actual `VPM@1` frontier with positive paired lower bounds;
2. latent NMSE has a nonnegative point effect and lower bound above -1%;
3. decoded and temporal MSE beat `J1-PFD-SHUFFLED` by at least 1%;
4. aligned-versus-shuffled and aligned-versus-zero action sensitivity do not
   regress relative to `J1-OFF-PM`;
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
