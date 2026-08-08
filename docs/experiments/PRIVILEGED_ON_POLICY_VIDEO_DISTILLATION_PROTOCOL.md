# Privileged on-policy video-distillation protocol

Date: 2026-08-08

Status: prospective, inference-valid alternative selected after autonomous
future-feature and training-only relation-loss failures; no quality claim has
been made

## Question

Can the large **training-time** advantage of a clean future-video feature be
transferred into a student that receives no future feature at inference?

This changes the role of the second branch.  The clean TF/V-JEPA state is not a
deployment condition and is never approximated by a weak autonomous feature
sampler.  It instead defines a privileged teacher available only on training
examples:

\[
v_T=v_{\bar\theta}(z_t^S,t,h,a,u^*),\qquad
v_S=v_\theta(z_t^S,t,h,a,\varnothing),
\]

where `u*` is extracted from the target future video and `z_t^S` is a state
visited by the current causal student's own rollout.  Only `v_S` exists in the
deployed graph.  The hypothesis is that the teacher may expose a smoother or
better conditioned denoising direction whose **reachable** component can be
learned from history, actions, the noisy video state, and the sampled noise.
It cannot transfer irreducible information about a particular future.

## Why this is not the completed VideoREPA/TRD experiment

The completed TRD arms aligned a block-14 student relation matrix to a frozen
clean-video encoder relation at ordinary teacher-forced noisy samples.  They
successfully reduced relation error by 84--90% but worsened deployable video.
That is feature imitation, not policy distillation.

This protocol instead:

1. rolls the causal student from its own noise and conditions;
2. queries teacher and student at the **same student-visited state**;
3. matches their video velocity/flow predictions rather than an arbitrary
   hidden coordinate system;
4. admits a teacher target only when it is both better than the causal target
   and sufficiently close to be plausibly reachable; and
5. evaluates the ordinary feature-free student at inference.

It therefore tests an untried mechanism.  The TRD failure remains a strong
warning against adding another unconstrained representation loss.

## Frozen information contract

Teacher-only training inputs may include the clean target-video TF/V-JEPA
feature.  Student inputs are restricted to observed RGB history, proposed
actions, diffusion time/noise, and the student's current video state.  The
teacher output is stopped.  Target future RGB may compute the normal flow-
matching target and teacher-only feature but may never enter the student
forward path, validation sampler, confidence model, or inference artifact.

Train, development validation, and lockbox episodes remain disjoint.  The
existing val64 split has already been adaptively reused, so the first run is an
exploratory screen.  No lockbox is opened until the architecture, weight,
rollout schedule, and primary endpoint are frozen.

## Teacher-eligibility gate before training

Do not distill merely because an oracle condition improves a same-checkpoint
ablation.  On train-only calibration clips and student-visited states, require:

- the aligned privileged teacher has lower video-velocity MSE than the causal
  student on at least 60% of clip/timestep units;
- mean velocity-MSE improvement is at least 5%, with a positive paired lower
  bound;
- aligned teacher features beat episode-shuffled teacher features by at least
  3%, with a positive paired lower bound; and
- the teacher's full rollout improves both decoded and temporal MSE over the
  causal student at the same RGB NFE.

Failure stops this route.  A teacher that is not sample-specifically better
cannot create useful privileged supervision.

## On-policy objective

For each batch, generate a one- or two-call student trajectory without target
features and sample its visited state `z_t^S`.  Let `v*` be the ordinary video
flow target.  Compute per-unit errors

\[
e_T=\lVert v_T-v^*\rVert_2^2,\quad
e_S=\lVert v_S-v^*\rVert_2^2,\quad
d=\lVert v_T-v_S\rVert_2^2.
\]

Use a training-only advantage/reachability gate

\[
g=\mathbf 1[e_T<e_S]\,
  \mathbf 1[d\leq Q_\delta(d\mid\text{batch},t)],
\]

so a privileged target is used only when the teacher is actually better and
its discrepancy lies in the lowest frozen percentile `delta`.  The student
objective is

\[
\mathcal L=\mathcal L_{flow}(v_S,v^*)+
\lambda\frac{\sum g\,w(t)\lVert v_S-\operatorname{sg}(v_T)\rVert_2^2}
                    {\max(1,\sum g)}.
\]

Use an EMA teacher to stabilize targets.  The first screen fixes
`delta in {25%,50%}` and selects it using train-only calibration; it does not
tune a threshold on val64.  Log admitted fraction and effects by diffusion
time.  If the model uses classifier-free guidance, do not match only the
composed guided velocity.  Separately match the positive prediction and
conditional direction

\[
\lVert v_T^+-v_S^+\rVert_2^2+
\eta\lVert (v_T^+-v_T^-)-(v_S^+-v_S^-)\rVert_2^2,
\]

to avoid compensating positive/negative branch errors.

## Controlled arms

All trainable arms start from one parent snapshot and use identical examples,
noise, data order, optimizer, updates, inference parameters, and RGB calls.

| Arm | Training-only teacher | Student rollout | Purpose |
|---|---|---|---|
| `BASE` | none | on-policy | incumbent continuation |
| `OFFPOLICY-KD` | aligned oracle | target-forward-noised states | distinguishes state-distribution mismatch |
| `OPD-ALL` | aligned oracle | on-policy | tests unrestricted privileged transfer |
| `OPD-GATED` | aligned oracle | on-policy | primary reachable-transfer candidate |
| `OPD-SHUFFLED` | episode-shuffled oracle feature | on-policy | sample-specific teacher attribution |

`ORACLE-INFER` is an evaluation-only, explicitly nondeployable ceiling.  It
cannot satisfy a success gate.  Every deployable evaluation hard-asserts that
the clean feature extractor, target cache, and teacher module were not called.

## Primary development decision

Evaluate NFE 1 and 2 on the paired val64 screen.  Positive means lower error.
`OPD-GATED` advances only if all conditions hold at one preregistered NFE:

- decoded and temporal MSE improve by at least 3% over `BASE`, with paired 95%
  lower bounds above 1%;
- latent NMSE is nonnegative in point estimate with lower bound above -1%;
- it beats `OPD-SHUFFLED` by at least 1% decoded and temporal MSE;
- aligned-versus-shuffled and aligned-versus-zero action sensitivity does not
  regress relative to `BASE`;
- the feature-free sampler graph and transformer-call count are byte-audited;
  and
- complete NFE latency is unchanged within measurement noise.

Compare `OPD-GATED` with `OPD-ALL` to test the central reachability mechanism,
not just the existence of another loss.  A point-only gain or an oracle-only
gain is a `NO_GO`, not a reason to tune on val64.

## Interpretation and novelty boundary

This direction is attractive because it converts the clean-feature result into
training supervision and removes the feature problem at serving.  It does not
violate the information limit: a causal student cannot reproduce target detail
that is independent of history/actions/noise.  Gating can retain learnable
teacher structure and reject unreachable leakage, but cannot manufacture
future information.

[Rethinking CFG in On-Policy Diffusion Distillation](https://arxiv.org/html/2607.24731)
already demonstrates privileged dense-to-sparse transfer for video controls
and motivates student-visited states plus branch-aware matching.
[Privileged Foresight Distillation](https://arxiv.org/html/2604.25859)
is closer in robotics: a full-future attention mask defines a shared-backbone
teacher, and a small current-only adapter learns the detached teacher-minus-
student **action-denoising** residual, with no future video at inference.
[Privileged Self-Distillation](https://arxiv.org/html/2607.27055) demonstrates
future-aware teacher to causal-student transfer and a reachability gate in a
different sequential domain.  These are external results, not evidence for
ABC/LACWM.  A defensible contribution here would require action-conditioned
robot **video-velocity** generation rather than action prediction, a target-
video semantic/TF teacher queried on student-visited video states, explicit
reachable-versus-unreachable attribution, equal-cost one/two-call sampling,
and closed-loop world-model utility.  Plain privileged distillation or a
teacher-minus-student residual adapter is not by itself novel.

## Stop rules

- Do not run student training if the aligned teacher fails the train-only
  eligibility gate or equals its shuffled-feature control.
- Do not use target-derived quality as an inference-time confidence signal.
- Do not select the gate percentile or loss weight on the repeatedly reused
  val64 split.
- Do not claim that distillation recovered future information; claim only a
  deployable quality/optimization gain if the feature-free student passes.
- Do not distill to fewer calls until the causal teacher/student at the source
  call budget actually dominates the NFE-1 incumbent.
