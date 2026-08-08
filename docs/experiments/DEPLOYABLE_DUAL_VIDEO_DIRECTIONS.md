# Deployable directions after oracle-feature dual diffusion

Date: 2026-08-07

Status: living research map; completed observations are separated from
prospective hypotheses and every result retains its protocol-specific claim
boundary

Decision summary: the experiments reject oracle future-video features as a
deployment mechanism and reject the tested autonomous semantic, Fourier,
scratchpad, action-token, and training-only relation/spectrum formulations at
this budget. They do **not** imply that every dual state requires unavailable
future RGB. The best remaining causal hypothesis is an action-derived dense
motion scaffold computed before RGB denoising from the proposed joint trajectory
plus robot geometry/camera calibration, followed—if it actually improves a
strong teacher—by inference-consistent few-step distillation.

## Completed deployable evidence through 2026-08-08

All results below are one-seed development-validation evidence on the frozen
ABC screen, not protected-test or general video-quality claims.

| Intervention | Primary observation | Decision |
|---|---|---|
| generated full-clip V-JEPA state | autonomous conditioning was indistinguishable from fusion-off and shuffled state; J1 video-flow loss was 31.6% worse than VPM | no deployable mechanism |
| PhaseLock-style generated motion prior | stabilized an ordinary multi-call trajectory, but did not beat the one-call VPM frontier and aligned was indistinguishable from shuffled | no sample-specific guidance |
| temporal semantic target DOE | absolute, delta, temporal, and self-roll-in variants failed; more calls generally worsened prediction | no passing semantic motion state |
| video residual coordinates | worse latent/decoded quality at every NFE; a +7.90% NFE-4 latent-temporal effect came with substantial appearance regressions | rejected |
| LaMo macro-motion drift loss | effects were approximately neutral; best temporal point effect was +0.47% at NFE 4 with a negative simultaneous bound | rejected at this budget |
| two-clock consistency | NFE-1 latent/decoded MSE improved +0.49%/+0.68%, but temporal MSE worsened 0.28%; all metrics worsened by NFE 4 | rejected at this budget |
| observed V-JEPA anchor plus generated increments | semantic NMSE improved 46--49%, but increment NMSE was 1.00--1.17, temporal attribution versus shuffled/donor controls was below 0.3%, and a mean/static increment was as good or better | static identity works; future motion does not |
| generated Haar frequency forcing | auxiliary-only, synchronous-joint, and leading-joint arms all regressed the NFE-1 video-only control; the best joint arm was 6.39%/3.86%/0.74% worse in latent/decoded/temporal error and aligned equaled shuffled at NFE 1 | rejected at this budget and ordering |
| causal pre-Wan action motion plan (CAMP) | the frozen causal planner remained weak on validation (plan NMSE 1.548, cosine 0.166); its aligned NFE-1 video was 0.33%/0.94%/0.22% worse than the independently trained no-plan arm in latent/decoded/temporal error, and aligned differed from a plan-shuffled control by at most 0.074% | rejected for this representation and 400-update budget |
| one-call intra-forward generated scratchpad | the generated state was causally injected after Wan block 14 in the same call, at 5.96% of native-token RMS, but NFE-1 latent/decoded/temporal error was 0.18%/1.13%/0.26% worse than the matched trained-off arm; enabling it in the same checkpoint changed decoded/temporal error by only +0.064%/+0.033%, and aligned versus future-shuffled effects were at most 0.018% | rejected for this midpoint representation and 200-update budget; no sample-specific one-call benefit |
| within-chunk action-delta residual | the learned residual gate collapsed to -0.00140; NFE-1 latent/decoded/temporal error was 1.06%/1.09%/0.023% worse than the matched control, hard-masking the trained residual changed temporal error by only -0.0029%, and effective control rank stayed 1.18 versus 1.20 in control with shuffled cosine 0.9983 | rejected; a residual before the existing pool/broadcast does not repair action collapse |
| ordered native Wan action tokens (zero-start) | NFE-1 decoded/temporal error improved only 0.176%/0.081% while latent NMSE worsened 0.126%; the learned gate stayed at -3.88e-5, injected context RMS was only 5.45e-7, aligned-versus-shuffled/zero interaction effects were below 0.043% in magnitude, and native versus hard-masked temporal improvement was 0.0095% | reject this zero-start, 200-update route; it did not receive a meaningful optimization dose, so it does not reject action tokens generally |
| ordered native Wan action tokens (fixed 0.1 dose) | the route was genuinely open (context RMS 0.00140; effective rank 1.45), but NFE-1 latent/decoded error improved only 0.405%/0.259% while temporal error worsened 0.086%; hard-masking the trained tokens changed temporal error by only +0.010%, aligned-versus-shuffled interaction was +0.0003%, and every causal gate failed | rejected at this fixed dose and budget; the zero-start result was not merely a closed-gate artifact |
| training-only video relation distillation (weight 0.5) | the student relation L1 fell 90.2%, from 0.9033 to 0.0889, proving that the privileged clean-video target was learnable, but video-flow loss rose 6.62%; at NFE 1 latent/decoded/temporal error was 3.08%/0.98%/0.21% worse than the matched loss-off arm, with the latent regression's paired interval entirely below zero | rejected at this dose; successful representation imitation did not improve deployable generation and conflicted with the primary flow objective |
| training-only video relation distillation (weight 0.05) | the relation L1 still fell 84.1%, but flow loss rose 0.84%; the registered mean-of-clip-relative NFE-1 effects were -3.09%/-0.65%/-1.12% in latent/decoded/temporal error, all six point/lower-bound gates failed, and the temporal interval was entirely below zero | rejected at this lower dose; reducing the auxiliary-loss conflict did not expose a deployable quality gain, and this adaptive val64 reuse is exploratory |
| training-only 3-D spectral endpoint loss (weight 0.05) | aligned NFE-1 latent/decoded/temporal error changed by -0.564%/-0.134%/-0.110%, with paired intervals crossing zero; training flow loss and teacher-forced clean-latent NMSE were also 1.44%/1.74% worse | rejected for this low-band log-amplitude/phase objective; a preregistered H=23 versus actual padded H=24 geometry mismatch makes the otherwise identity-verified result exploratory |
| clean-latent inverse-action recoverability | a frozen train-only ridge probe achieved future-transition cosine 0.118, but a train-mean action predictor had much lower standardized MSE (1.070 versus 1.995), same-clip temporal misalignment was nearly as good (cosine 0.104), retrieval was 4.69%, and the 16-comparison simultaneous gate failed | stop the action-cycle generator continuation; this Wan displacement representation does not identify the requested within-clip action strongly enough |
| causal spectral-information probe (CSIP) | spectral angle improved target cosine over a matched magnitude/support-only probe by +0.127 (simultaneous lower bound +0.023), but improved MSE only 1.57% with a negative lower bound (-7.61%); aligned actions were 314% worse than raw-no-action in relative MSE | phase is partly informative, but no action-specific causal spectral route was demonstrated; stop this generator path |
| causal action-to-motion-summary Stage 0 | a train-only ridge model using observed motion plus planned actions improved val64 standardized future Farneback-summary MSE by 6.59% over history-only (paired 95% interval 4.96--8.05%) and by 9.04% over the same model with episode-shuffled actions (7.19--11.10%); aggregate R2 rose from 0.098 to 0.158 | passes its preregistered exploratory 1% information/specificity gates, but not the later proposed 10% all-view integration gate; supports a calibrated dense-flow follow-up, not a generator-quality claim |

Across the residual, LaMo, two-clock, and observed-anchor screens, action or
sample shuffling changes the relevant outputs by at most small fractions of a
percent. The recurring failure is therefore not merely a bad Fourier basis.
The learned future variable is weakly action-sensitive and its autonomous
sample identity is not causally used by video.

A direct replay of the VPM action path localizes part of this failure. Raw
future actions remain sample-specific (cyclic-shuffle cosine 0.728), but the
trained Wan control has cosine 0.99885 after shuffling, only 17.0% sample
standard deviation relative to its RMS, and centered effective rank 1.16 over
32 values. Thus the conditioning path largely collapses diverse requested
actions into one common direction before video generation. A better auxiliary
motion state cannot become action-specific if its causal action input has
already lost most of that information.

There is also a one-call ordering problem in the existing synchronous joint
sampler. At NFE 1, both video and auxiliary enter the sole Wan evaluation at
their noise endpoint. The video velocity can see the auxiliary's input noise,
but not the clean auxiliary estimate produced by that same call, because both
Euler updates occur afterward. A schedule can make the auxiliary useful only
on a later call. A true one-call dual mechanism must instead generate its plan
before Wan (the causal action motion-plan screen) or predict and inject it
between early and late blocks of the same Wan evaluation (the intra-forward
latent-forcing screen).

This is a materially harder latency target than the published Latent Forcing
result. Its best cascaded image sampler uses 50 Heun steps, with 25 latent
steps followed by 25 pixel steps; during pixel training the latent is normally
clean and the pixel loss is trained separately. The paper also reports that
DINOv2 features do not inform pixels well while the pixel state is still at
very high noise. Thus, a synchronous one-call video implementation is not a
faithful low-cost reproduction of the paper's successful causal ordering. It
removes precisely the interval in which the generated latent becomes a usable
condition. CAMP tests an explicit cheap pre-Wan phase; intra-forward forcing
tests whether that phase can be compressed inside one Wan evaluation.

## Constraint exposed by the completed screens

A clean feature of the unknown future can be used as a training target or an
explicitly labelled oracle diagnostic, but it cannot enter a deployable video
sampler. A useful second state must instead be:

1. computed only from observed history and proposed actions;
2. generated by the model from noise and trained on that generated
   distribution; or
3. used only as a training loss and removed at inference.

This does not make dual diffusion impossible. It rules out interpreting an
oracle-conditioned gain as an inference gain.

## Direction 1: observed anchor plus generated motion increments

This is the first contingent experiment and is specified exactly in
`VIDEO_LATENT_FORCING_OBSERVED_ANCHOR_PROTOCOL.md`. Let an observed-only encoder
produce `A = Phi(history)` and define

\[
D_j=S_j-S_{j-1},\qquad S_{-1}=A.
\]

Only normalized increments are diffused. At inference,

\[
\hat S_j=A+\sum_{k=0}^{j}\hat D_k.
\]

The anchor is available before generation and carries static scene content;
the stochastic branch spends its capacity on future change. The first semantic
screen uses the existing V-JEPA/PCA space for comparability. A later video
version should use the already available causal video-VAE history latent so
that anchor extraction does not dominate end-to-end latency.

Main risks are cumulative increment error, multimodal future motion, and weak
action attribution. Required controls are a repeated-anchor predictor, a
train-mean increment, wrong anchor, wrong history/action, and oracle increments.

## Direction 2: self-derived feature conditioning

Instead of generating an independent feature branch, derive the next feature
from the model's previous clean-video estimate:

\[
u_k=\operatorname{stopgrad}F(\hat z^{k-1}_0),\qquad
\hat z^k_0=f_\theta(z_{t_k},t_k,h,a,u_k).
\]

`F` should be deterministic and inexpensive: temporal finite differences,
3-D Haar low-pass/detail coefficients, or normalized complex temporal spectra
of the current video latent. The first call uses an all-zero self-condition;
later calls reuse information already produced by the preceding call. No clean
future feature and no separately denoised auxiliary trajectory exist at
inference.

Training must construct `u` from a stopped preliminary model prediction, not
only from the clean target. A parameter-matched off arm, sample-shuffled `u`,
and an oracle-clean diagnostic distinguish useful generated feedback from extra
capacity. NFE counts charge only transformer calls; the cost of `F` remains in
end-to-end latency.

This is a hypothesis until separately preregistered and run.

## Direction 2b: generated early-motion prior with equal-call accounting

The incumbent's current one-call output is better than its longer Euler
trajectories. That makes the early generated estimate itself a candidate
scratchpad. A cheap preliminary trajectory can provide latent frame
differences, which are then held as a stopped motion prior while a second
trajectory refines appearance. The prior is generated from the same observed
history, actions, and noise; it never uses target video.

This idea is motivated by
[PhaseLock](https://arxiv.org/abs/2606.06361), which reports that a two-step
image-to-video sample can retain motion structure that later visual refinement
erases. Its evidence is literature evidence, not evidence for this robot world
model. Our probe must compare against an ordinary sampler with the same **total**
transformer calls and against a shuffled generated-motion prior; otherwise an
apparent gain could be caused by extra compute or generic regularization.

## Direction 2c: one-call intra-forward latent forcing

The closest one-call analogue of Latent Forcing is to predict the auxiliary
clean state from early generator blocks and inject it before the remaining
blocks of that **same** Wan evaluation. With midpoint block `m`,

\[
h_m=G_{\theta,0:m}(z_s,q_s,h,a),\qquad
\hat q_0=q_s-s\,v_q(h_m,q_s,s),
\]

\[
\hat z_0=G_{\theta,m+1:L}\!\left(
h_m+A_q\bigl(\operatorname{stopgrad}(\hat q_0)\bigr)
\right).
\]

Unlike synchronous output-level dual diffusion, the video path can causally
read the newly predicted auxiliary before its own velocity is emitted. Unlike
CAMP, the auxiliary also reads the noisy future video tokens through the first
half of Wan, matching the key information route used by image Latent Forcing.
It requires no clean future feature, extra Wan call, pretrained V-JEPA encoder,
or oracle at inference. The prospective screen uses block 14 of the 30-block
Wan, exact parameter-matched injection-off/on arms, and an observed-history-
preserved future shuffle to test whether the generated auxiliary is actually
sample-specific.

The independently audited B200 screen in
`VPM_INTRA_FORWARD_LATENT_FORCING_PROTOCOL.md` completed with 1,152 sealed
rows and no protected-test access.  It did not pass.  At NFE 1 the trained
MID-ON arm regressed decoded MSE by 1.13% and temporal MSE by 0.26% versus
MID-OFF.  More importantly, within the same MID-ON checkpoint, autonomous
injection improved decoded/temporal MSE by only 0.064%/0.033% over hard-off,
and history-preserved future shuffling changed all metrics by at most 0.018%.
This separates an optimization effect from mechanism: the scratchpad was
nonzero and cheap, but its sample-specific content was not causally used.

## Direction 3: invertible coarse-to-detail latent forcing

Apply an orthonormal spatiotemporal transform to the video-VAE latent,

\[
(L,R)=Wz,\qquad z=W^{-1}(L,R),
\]

where `L` is a low-resolution 3-D Haar/Gaussian-pyramid component and `R`
contains the complementary detail coefficients. Both states start from noise
at inference. A dual clock makes `L` clean earlier so it can act as an internal
layout/motion scratchpad while `R` renders detail. Because the transform is
invertible, no pretrained teacher, decoder, or clean inference feature is
required.

This is closer to the ordering mechanism in Latent Forcing than conditioning on
a wholly separate semantic domain. It is also the direct video analogue of
[Frequency-Forcing](https://arxiv.org/abs/2604.20902), whose image experiments
generate an earlier-maturing learnable low-pass wavelet stream rather than
supplying its clean target at inference. It is not automatically positive: the
completed coarse-RGB experiment predicted appearance well but not aligned
temporal change. Any new screen must therefore separate temporal low-pass from
static DC content, use generated-only trajectories, and compare against a
parameter-matched single-stream transform baseline at equal total calls.

## Direction 4: spectral supervision without an inference branch

Time-frequency information can supervise the predicted clean video latent
without becoming an input:

\[
\mathcal L_{spec}=\lVert\log(1+|\mathcal F\hat z_0|)
-\log(1+|\mathcal Fz|)\rVert_1
+\lambda_\phi\,\mathbb E[1-\cos(\Delta\hat\phi-\Delta\phi)].
\]

The transform should be per-view and spatiotemporal; phase terms use circular
differences and mask coefficients whose magnitude is too small for stable
phase. This intervention has zero inference dependency and directly tests
whether spectrum/phase is valuable as supervision rather than as an oracle
condition. The completed 200-update screen did not pass. Relative to an exact
loss-off arm, the aligned NFE-1 latent/decoded/temporal effects were
-0.564%/-0.134%/-0.110%, where negative is worse; every registered gate failed.
Training flow loss and teacher-forced clean-latent NMSE also rose 1.44% and
1.74%. The artifact chain and no-inference-feature contract were independently
verified, but the registration said latent height 23 while the Wan VAE actually
padded RGB height 180 to 192 and produced height 24. The 806 selected bins
confirm the latter geometry, so this is exploratory rather than a strict
confirmation. It rejects the exact joint-view, two-token, low-band endpoint
loss, not all motion-aware or per-view supervision.

## Direction 5: exact-few-step and on-policy distillation

The completed NFE frontiers and D0 diagnostic show that extra Euler calls can
worsen the incumbent. If representation changes do not repair this, optimize
the requested one- or two-step map directly instead of relying on numerical
integration of a flow trained at isolated corruptions. Candidate methods are
consistency/shortcut objectives and causal consistency distillation from a
strong multi-step teacher. Training should roll on self-generated contexts to
avoid exposure bias.

This is not a dual-state claim, but it is the strongest direct route to the
5--10 Hz DAgger objective. Relevant primary references include
[Consistency Models](https://arxiv.org/abs/2303.01469),
[Consistency Trajectory Models](https://arxiv.org/abs/2310.02279),
[Self-Forcing](https://arxiv.org/abs/2506.08009), and
[Causal Forcing++](https://arxiv.org/abs/2605.15141). Two especially relevant
2026 systems are
[Causal-rCM](https://arxiv.org/abs/2606.25473), which combines teacher-forced
consistency initialization with self-forced distribution matching for one- or
two-step causal video, and
[Flash-WAM](https://arxiv.org/abs/2606.05254), which uses different consistency
parameterizations for video and action noise regimes. Their reported gains are
external baselines, not results for this repository.

## Direction 6: separate causal planning from rendering

A history/action encoder can compute a compact plan once per generated frame or
chunk, while a lightweight diffusion decoder renders it in one or two calls.
The plan is a learned internal state, not a clean future feature. A
parameter-matched encoder whose plan is stopped, zeroed, or shuffled is needed
to establish mechanism rather than capacity. This direction is motivated by
[Separable Causal Diffusion](https://arxiv.org/abs/2602.10095), which reports
that temporal reasoning and iterative rendering can be separated, but its
reported result is literature evidence rather than evidence for this DROID
model.

The completed CAMP screen is a negative instance of this family, not a proof
against the family: a pre-Wan plan cannot help if it does not first predict
future motion. Any successor must pass a representation gate before generator
training and must beat a history-preserved, episode-disjoint shuffled plan.

## Direction 7: repair action information before adding another state

The measured action bottleneck motivates an upstream intervention. Preserve
within-chunk action deltas (or use explicit action cross-attention) before the
current temporal pool and spatial broadcast, and ask whether both output
quality and counterfactual action sensitivity improve. This is not itself dual
diffusion, but it is a prerequisite for an action-conditioned auxiliary future:
a second branch cannot invent sample-specific control information that its
input path has discarded.

The completed `VPM_ACTION_VARIATION_PROTOCOL.md` screen compared an exact
parameter-matched hard-off residual against a zero-initialized action-delta
residual. Its causal gate uses aligned, zero-action, episode-disjoint shuffled-
action, and runtime residual-hard-mask endpoints. A quality gain without this
difference-in-differences attribution is not evidence that richer action
conditioning caused the gain.

It failed all three required gates.  At NFE 1 the residual arm regressed
latent/decoded/temporal error by 1.06%/1.09%/0.023%; its aligned-versus-shuffled
incremental temporal effect was -0.0061%, and hard-masking the trained residual
changed temporal error by -0.0029%.  The effective rank remained essentially
unchanged (1.18 candidate versus 1.20 control).  The next controlled repair
therefore bypassed the temporal pool and spatial broadcast entirely by retaining
all 40 planned low-level actions as ordered native Wan cross-attention tokens.
That zero-start route also failed: temporal error improved only 0.081%, its
action-attribution effects were negligible, and its gate remained effectively
closed. Because a zero scalar blocks adapter-weight gradients until the random
initial projection happens to move the scalar, this is a narrow optimization
failure rather than a fair rejection of explicit action tokens. An adaptive
follow-up fixes the same scalar at 0.1 in both arms and hard-masks only the
control, forcing the candidate projection to train from its first update. It
reuses val64 and is therefore exploratory, not confirmatory.

That follow-up completed with a verified effective gate of 0.1 and a nonzero
40-token context, so it removes the optimization ambiguity. It still did not
pass: NFE-1 latent and decoded errors improved 0.405% and 0.259%, temporal error
worsened 0.086%, and the trained route's native-versus-hard-mask temporal effect
was only +0.010%. Aligned-versus-shuffled and aligned-versus-zero
difference-in-differences effects were also far below the 0.5% causal threshold.
The existing Wan action control retained effective rank 1.186, essentially the
same as control. Thus the particular ordered-token adapter is not the missing
dual state at this budget; a successor should change the causal training target
or backbone action interface, not just raise this adapter's dose again.

## Direction 8: use representation learning on the training side or policy side

Two routes avoid an online clean-future condition entirely:

1. A frozen causal spectral-information probe can test whether spatiotemporal
   magnitude and phase carry future- and action-specific information. If the
   probe passes against zero, episode-shuffled, inverse-action, and magnitude-
   only controls, its loss can supervise Wan during training and the probe can
   be discarded at inference.
2. A robot policy can consume generated Wan latents directly through a small
   adapter, decoding RGB only for visualization or intervention review. The
   current warmed latent-only endpoint is about 0.102 s mean and 0.118 s
   empirical p95 over 64 rollouts (8.49 latent rollouts/s at p95), whereas RGB
   decoding adds roughly 0.130 s. This is a systems observation, not yet a
   closed-loop DAgger result or a video-quality claim.

The first route was screened by `CSIP_PHASE0_PROTOCOL.md`. Its full spectral
probe beat the matched angle-neutral probe on cosine but not MSE, and failed
the action-specific gate because raw-no-action was substantially better on
MSE. That closes this particular causal spectral-generator path without
claiming that phase is universally useless. The second route is consistent
with recent latent-world-action work, but requires a downstream policy-utility
study rather than FID alone. For longer autoregressive rollouts,
[Self Gradient Forcing](https://arxiv.org/abs/2607.20368) offers a complementary
training-only mechanism: it re-encodes stopped self-generated histories while
retaining future-loss gradients through the causal context K/V writer, so no
privileged feature is needed at inference.

## Direction 9: training-only video relation distillation

A stronger interpretation of the failed V-JEPA oracle study is to remove the
feature from the sampler, not necessarily from training. Let a frozen video
encoder produce clean token relations `R(E(x))`, and align those relations to
an intermediate Wan hidden state computed from the noisy video:

\[
\mathcal L_{\mathrm{TRD}}=
\lVert R_s(P(h_{14}(z_t)))-R_s(E(x))\rVert_1+
\lambda_t\lVert R_t(P(h_{14}(z_t)))-R_t(E(x))\rVert_1.
\]

Here `R_s` and `R_t` are per-view spatial and temporal cosine-similarity
matrices. The frozen encoder and clean future video are training-only; ordinary
video-only Wan sampling is unchanged at inference. Relational rather than
coordinate-wise alignment avoids requiring the diffusion hidden state to copy
the teacher's basis. This is the deployable lesson from
[VideoREPA](https://arxiv.org/abs/2505.23656): a video foundation model cannot
be supplied at text/image-to-video inference, so its pairwise token structure
is distilled into intermediate denoiser features during training.

The controlled weight-0.5 screen used the immutable clean V-JEPA cache only on
train, exact TRD-off/on warm starts, no new inference parameters, and val64
NFE-1/2/4 evaluation. It learned the teacher relation strongly but failed every
NFE-1 quality gate: latent/decoded/temporal error changed by
-3.08%/-0.98%/-0.21%, where a negative improvement is worse. A weight-0.05
follow-up was frozen from training telemetry before the high-dose val64 result;
because it reuses val64, it is an adaptive development screen rather than an
independent confirmation. That lower-dose run also failed: relation error fell
84.1%, but video-flow loss rose 0.84%, and the registered NFE-1
latent/decoded/temporal effects were -3.09%/-0.65%/-1.12%. The temporal paired
interval was wholly below zero. Together these doses show that the exact
block-14 PCA64 relation target is easy to imitate but does not improve the
deployable generator in this quick screen; they do not establish that all
training-only representation objectives are ineffective.

## Direction 10: training-only action-cycle structure

The action-collapse audit suggests that the next auxiliary objective should
first make generated motion identify the requested action. Let `z_hat_0` be the
predicted clean video latent and let `p(Delta z_hat_0)` pool adjacent latent-bin
displacements separately in each camera view. A frozen inverse-action critic
`C`, fitted only on clean train latents, supplies

\[
\mathcal L_{cycle}=\left\lVert
C\!\left(p(\Delta\hat z_0)\right)-\Delta a
\right\rVert_2^2.
\]

The critic and target disappear at inference, so sampling remains one ordinary
Wan call per NFE. Before any generator continuation, a representation probe
must show on disjoint val64 clips that clean Wan latent displacement predicts
the matching action delta better than both a train-mean predictor and an
episode-disjoint shuffled target. This prevents spending a generator run on an
inverse problem that the chosen latent scale does not contain.

If that gate passes, aligned-target, globally shuffled-target, and loss-off
continuations distinguish causal action alignment from generic regularization.
The idea is related to latent-displacement action decoding in
[Delta-JEPA](https://arxiv.org/abs/2606.31232), but its value for this Wan model
is prospective. A complementary, more expensive successor is counterfactual
consistency: compare reference, zero-action, and physically valid inverse-action
rollouts and penalize action-independent drift. Recent
[CoCo](https://arxiv.org/abs/2608.04653) results show why reconstruction quality
alone can hide an action shortcut; any adaptation here must first verify that
the continuous robot actions admit a valid inverse rather than assuming that
simple sign reversal is physically meaningful.

The Stage-0 probe failed, so no generator continuation was launched. On the two
future-relevant transitions, aligned validation cosine was 0.118 versus -0.013
for a shuffled-fit critic, but this apparent identity signal was not a useful
inverse action: standardized MSE was 1.995, substantially worse than the
train-mean predictor's 1.070, and the same-clip temporally misaligned target had
cosine 0.104. The simultaneous 16-comparison gate therefore rejected the chosen
pooled Wan-displacement representation before it could consume another video
training run. A future inverse-dynamics objective would need a more local,
contact-aware representation and a stronger pre-generator recoverability gate.

## Direction 11: action-derived robot-only flow before RGB denoising

The strongest remaining dual-state hypothesis does not predict an auxiliary
from unknown future RGB. It computes a dense future scaffold from the proposed
robot trajectory, which is already known when an action-conditioned world model
evaluates that candidate:

\[
(q_{0:T},\mathcal R,\mathcal C)
\xrightarrow{\text{kinematic render}} I^{robot}_{0:T}
\xrightarrow{\text{flow}} U^{robot}_{1:T}
\xrightarrow{\text{encode}} z_{flow}.
\]

Here `q` is the planned joint trajectory, `R` contains the URDF, joint mapping,
and robot root pose, and `C` contains calibrated cameras. This is a clean future
*motion* condition but not a clean future *scene-video* feature: object outcomes,
background changes, and contact consequences remain unknown and must be
generated. Its causality holds only when the complete candidate action sequence
is available before rendering; replaying recorded executed states that are not
known from the proposed commands would violate the contract.

The released [FlowWAM](https://arxiv.org/abs/2607.13017) WorldArena path is an
external existence proof for this design class. It directly sets planned joint
positions in SAPIEN, renders robot-only views from calibrated cameras, computes
masked RAFT flow, VAE-encodes that flow, and holds the clean flow latent fixed
while RGB is denoised. RGB and flow have separate patch embeddings and heads,
but share each DiT block through concatenated bidirectional self-attention.
Its published quality result is not our result, and its default 50-step sampler
plus renderer, RAFT, VAE encode, and refiner is not evidence of real-time DAgger.
[RealWonder](https://arxiv.org/abs/2603.05449) provides related external evidence:
physics-derived optical flow and coarse RGB condition a distilled four-step
causal generator, with reported 13.2 FPS, but at a much larger training budget.

The current immutable ABC cache cannot faithfully instantiate this route. It
retains three RGB views and 14 real action values (12 joints plus two grippers),
but not a URDF, joint-name convention, base pose, camera intrinsics/extrinsics,
or wrist-camera transforms. Therefore no claim should be made from pseudo
robot-only flow fabricated from the present cache. The exact staged design is
frozen in `ACTION_DERIVED_FLOW_PROTOCOL.md`; its bounded Stage-0 gate is:

1. audit raw ABC/DROID assets for exact robot geometry and calibration;
2. if absent, train an action-to-flow proxy on train512 only and test whether
   aligned actions beat history-only flow prediction by at least 10% with a
   paired lower bound above zero;
3. require episode-disjoint action shuffling to erase at least 80% of that gain
   and require p95 scaffold latency below 20 ms;
4. only after that gate, compare flow-off, aligned causal flow, shuffled flow,
   and future-RGB-derived oracle flow at fixed RGB training and Wan-call budgets;
5. if the causal arm retains a material fraction of the oracle ceiling, replace
   the proxy with deterministic robot rendering and then distill the strongest
   flow-conditioned teacher to two or four RGB calls.

A train/validation-only raw-MCAP audit sharpened the prerequisite. All 576
manifest episodes exist, but their topics contain only RGB, camera intrinsics,
joint/gripper state and action, instruction, and occasional end-effector pose.
They contain no URDF/MJCF, TF tree, robot-base pose, extrinsics, depth, mask, or
attachment. The official ABC repository does contain nominal YAM MJCF/meshes
and D405 top/wrist transforms. A defensible next dataset is therefore the 456
D405 clips with all three intrinsics, beginning with the top camera; nominal
rendered silhouettes must pass an image-alignment gate before their flow can
condition video. Heterogeneous ZED/OAK/decxin rigs and wrist timestamps require
separate calibration or raw re-extraction.

The existing DROID LeRobot cache is not a shortcut. It has three RGB streams
and Cartesian end-effector action/state, but no camera calibration, joint
positions, robot model, depth, or masks. A read-only schema audit also found
that metadata defines the gripper at state index 7, while the current loader
reads index 6; sampled parquet has index 6 identically zero and index 7 varying.
Correct the loader to use the native action-7/gripper contract before using
DROID for any scaffold or action-attribution study. Deterministic rendering
would additionally need calibration and either joint trajectories or a fixed IK
convention.

The first causal information screen is encouraging but deliberately weak. A
train-only ridge predictor used four observed Farneback-summary transitions and
planned actions to predict eight future per-view 2x2 flow summaries. On val64,
aligned actions improved standardized MSE 6.59% over history-only (paired 95%
interval 4.96--8.05%) and 9.04% over episode-shuffled actions (7.19--11.10%).
This clears its preregistered 1% exploratory gate and demonstrates incremental,
action-specific future-motion information. It does not clear the stricter 10%
all-view integration threshold, predict dense flow, or show that Wan will use
the signal. Wrist ego-motion, scene motion, and task/state correlation remain
confounds. A post-hoc top-view slice improved 12.61% with paired interval
10.23--15.08%, while left/right wrist gains were only 3.57%/3.18%; signed
direction carried more signal than magnitude. This makes top-view-only
calibration and a dense target the next gates, but the slice is diagnostic and
cannot replace the failed preregistered 10% all-view threshold.

A low-cost first integration can reuse the existing 16-channel auxiliary
projection and keep `z_flow` fixed with flow loss zero. A positive causal result
would justify the more faithful and expensive per-block joint-attention port.
For deployment, analytical rasterized scene flow from link depth/IDs is likely
preferable to SAPIEN RGB, RAFT, HSV encoding, and a second VAE pass.

The more novel second stage is a physics-anchored residual motion diffusion:

\[
u_{robot}=G(a,\mathcal R,\mathcal C),\quad
r\sim p_\psi(r\mid h,a,u_{robot}),\quad
x\sim p_\theta(x\mid h,a,u_{robot}+r).
\]

`u_robot` supplies the known embodiment trajectory; the small stochastic
residual `r` represents object motion, contact consequences, and camera/model
error, and is generated before RGB. This is a genuine inference-valid dual
diffusion candidate. It should be attempted only if deterministic robot flow
first helps and if a generated residual beats zero/mean/shuffled residuals at
equal calls. Otherwise it repeats the already failed autonomous-feature path
with a more elaborate parameterization.

## Direction 12: per-view low-frequency motion supervision

The joint-view TFREG screen mixed three camera views along the FFT width and had
only two future Wan tokens. A narrower training-only follow-up splits the three
views, prepends the last observed latent token, applies a fixed per-view Gaussian
low-pass, and supervises the two adjacent signed motion deltas of `x0_hat` with
target-RMS-normalized Smooth-L1. It has no auxiliary parameters, teacher, FFT,
or inference call. The paired loss-off/on screen is preregistered at weight
0.05, 200 updates, one seed, train512/val64, and NFE 1. Because eight future RGB
frames compress to two future Wan tokens, it can test only coarse latent motion;
it cannot support a contact-timing or general time-frequency claim. The shared
anchor also cancels algebraically inside the first delta error, so the objective
combines a low-pass endpoint loss for future token zero with one genuine
future-to-future motion-difference loss. Any positive result must be described
as composite structural regularization rather than pure motion supervision.

## Execution priority

1. Finish and audit the per-view low-frequency motion screen. Treat it as the
   last cheap training-only structural-loss test on the two-token ABC latent.
2. Acquire or recover the exact robot URDF, joint mapping, base pose, and camera
   calibration for one robot dataset, then run the action-to-flow causality gate
   before modifying the RGB generator.
3. If that gate passes, run flow-off/aligned/shuffled/oracle conditioning at
   equal Wan calls and report both end-to-end scaffold cost and RGB quality.
4. Distill only a demonstrably stronger causal flow-conditioned teacher with an
   inference-consistent/self-forced objective to two or four calls. The current
   VPM NFE-4 trajectory is worse than NFE 1 and is not a valid teacher.
5. In parallel, prototype a policy adapter over generated Wan latents and
   measure closed-loop task utility at the observed latent-only p95 latency.
   Optimize or distill the VAE decoder separately if RGB is operationally
   required at 5--10 Hz.
6. Advance a mechanism to multi-seed DROID and one untouched lockbox only after
   it passes action-attribution, latency, and quality gates on development data.

Every comparison must report quality at the same total transformer calls,
complete rollout latency including auxiliary work and decoding, peak memory,
training GPU-hours, action attribution controls, and one untouched lockbox
evaluation after development selection.
