# Next twelve-hour dual-video diffusion result

Status: final evidence ledger; the raw-flow and adjacent-consistency paths are
terminal, while recurrent delta passed its renderer prerequisite and has
advanced to a controlled low-NFE integration handoff.

## Question

Can an inference-available auxiliary state improve low-NFE robotic video
generation, or is the apparent gain limited to privileged clean-future
conditioning? The program compares causal feature proposals against direct,
feature-free controls and advances only mechanisms that survive prospective
causal and low-NFE gates.

## Evidence matrix

| Mechanism | Inference-available input | Prospective test | Observed result | Decision |
|---|---|---|---|---|
| Clean V-JEPA residual | no | privileged-residual repair versus equal-capacity direct residual and VPM@1 | repairs an intentionally weak endpoint, but direct residual is 27--46% better and VPM@1 is substantially better | `STOP_PRIVILEGED_NOT_CAUSALLY_COMPRESSIBLE` |
| Direct residual on VPM@1 | yes | frozen 256/31 frontier | decoded MSE +0.50%; temporal MSE -0.44%; registered intervals fail | `STOP_VPM_DIRECT_RESIDUAL` |
| V-JEPA2-AC action alignment | yes | aligned action versus causal controls, RTX and B200 replication | 0.7--2.0% aligned-action effect; controls below 5% gate | `STOP_VJEPA2_AC` |
| Object/contact distribution | yes | 12 registered gates | 12/12 gates fail | `CLOSE` |
| Fixed Haar detail | yes | matched low-frequency/detail screen | decoded +0.164%; temporal +0.104% | too small to advance |
| Learned QMF | yes | basis-reduction plus generation screen | basis reduction 1.534% versus 5%; generated detail +0.155%/+0.103% | `STOP_NO_HAAR_ADVANTAGE` |
| Early latent subspace forcing | yes | matched two-call NFE screen | lowpass redundant with VPM; HH slightly better; aligned path 22.1% slower | `NO_GO_GENERIC_EARLY_SUBSPACE` |
| Raw predicted robot flow | yes | frozen v7 RAW-FLOW versus matched FLOW-OFF, unmodified VPM parent, and shuffled/held/time-shifted controls at NFE 1/2/4 | NFE-1 top decoded MSE +1.587% versus matched-off, but temporal +0.201% is uncertain, LPIPS is -1.672%, all-frame guardrails fail, the parent is better, and causal controls do not attribute the gain to correct flow | `STOP_FIXED_RAW_FLOW` |
| Feature-free adjacent consistency | yes | RF-control versus consistency, 400 updates, full objective/readout factorial at NFE 1/2/4 | closest selectable point is NFE-4: temporal +4.963%, but decoded -22.009%, LPIPS -97.295%, and latent -55.963%; no NFE passes | `NO_GO_ACD` |
| Recurrent delta robot flow | yes | frozen fit384 predictor on the full 55-episode renderer-endpoint-unopened D405 census; three flow-superiority plus six spatial-NI gates | EPE reduction 41.880% versus raw, 34.705% versus absolute ridge, 88.274% versus shuffled; all nine Holm-controlled gates pass | `GO_RECURRENT_DELTA_WAN_SCREEN` |

## Operational evidence

The raw-flow v7 repair reproduced all eight cache arrays byte-for-byte, all
576 train/validation rows, all 2,000 per-update input hashes, the complete
1,686-tensor schema, every frozen tensor, and update-zero forward loss. The
first backward gradient differed slightly across different B200 nodes, with
loss divergence starting at update two. The registered historical gate was
bitwise equality, so v7 remains a failed operational repair and contributes no
efficacy endpoint. Raw-flow v8 therefore treated the immutable v7 pair as the
prospective paired realization, without retroactively adding a numerical
tolerance or selecting from v5 outcomes. Before evaluation it independently
replayed all 2,000 causal input hashes, 9,200 input/probe/order values, 7,200
finite-output checks, and both complete model schemas. The frozen evaluation
then scored 4,032 rows (48 episode clusters, four noise seeds, NFE 1/2/4) and
passed its artifact audit.

At primary NFE-1, RAW-FLOW versus matched FLOW-OFF improved top decoded MSE by
1.587% (episode-cluster bootstrap 95% CI [0.251%, 2.754%]), but top temporal
MSE improved only 0.201% ([-0.482%, 0.832%]) and LPIPS worsened 1.672%
([-2.674%, -0.790%]). All-frame decoded and temporal guardrails failed. Against
the unmodified VPM parent, RAW-FLOW was worse in top decoded MSE (-0.679%),
LPIPS (-2.423%), latent NMSE (-2.817%), and all-frame decoded MSE (-4.552%).
The same RAW-FLOW checkpoint changed 100% of latent and decoded output hashes
when flow was toggled off, but the effects were only +0.213% decoded and
+0.122% temporal; episode-shuffled, held-current, and time-shifted controls did
not establish correct-flow attribution. NFE-4 worsened both all-frame decoded
MSE (-1.373%) and temporal MSE (-0.728%) relative to matched-off. This supports
the mechanism-specific `STOP_FIXED_RAW_FLOW`, not a claim against learned,
recurrent, hybrid, or measured-geometry conditions.

ACD memory preflight job `507938` passed on source
`6063ec53d42faf13bb30d2e1c0ed2f30350ef183`. It executed one full synthetic
`[1,13,3,180,960]` transaction with one frozen teacher, one online student,
and one EMA target. Peak reserved memory was 15,919,480,832 bytes (8.31% of
device memory), leaving 175,583,657,984 bytes headroom. The production mask
contained three valid views, five history frames, eight future frames, two
future latent tokens, and 92,160 expanded future elements; no dataset,
endpoint, protected test, or W&B state was opened.

The final preregistered ACD study used source `6063ec5`, registration identity
`5e0ce96ca050d32b9e992587274f15dd2c239d72d9e4db0a0393c9a457ff6aa7`, and
separate exactly paired jobs: RF-control `507946` and adjacent-consistency
`507947`. Both completed all 400 updates with finite losses, one frozen teacher,
one online student, one EMA target, and zero auxiliary-feature calls. The frozen
pair validator replayed 8,000 treatment-invariant comparisons (400 updates by
20 fields) with zero mismatches, both 401-line trace hash chains, all initial
and final states, and all model-call counts. RF-control snapshot SHA-256 is
`44beb6ca6ee6f75415692efb09cbd1e402e2ca9e907665d93624fbae8387c5db`;
consistency snapshot SHA-256 is
`6d3277ea846aed80f91f56290a3f18d01e18aed6da0af048af4b80e5d387390e`.
Evaluation job `507964` compared both arms at NFE 1/2/4 with four noise seeds,
native and consistency readouts, LPIPS, and an action-shuffle control. It
completed `0:0` in 31m58s.

The sealed training traces show that the candidate did optimize its named
objective: mean consistency loss fell from `0.20697` over updates 1--100 to
`0.10799` over updates 301--400 (`-47.8%`). Its diagnostic RF loss increased
from `0.14890` to `0.91225` (`6.13x`), while RF-control loss was essentially
flat (`0.09531` to `0.09598`). Thus a negative endpoint should not be described
as a stalled optimizer, and blindly extending the same objective would not be
justified; the frozen consistency readout must demonstrate that it compensates
for the sacrificed parent velocity field.

## Feature-free adjacent consistency result

The frozen decision is **`NO_GO_ACD`**; no NFE passed. Positive percentages
below favor `CONS_EMA_CONSISTENCY`; negative percentages mean that the
consistency candidate is worse than the equal-NFE `RF_EMA_NATIVE_RF` control.
The bounds are the registered simultaneous one-sided lower bounds.

| NFE | Decoded MSE | Temporal MSE | LPIPS | Video-latent NMSE | candidate/control p95 | Pass |
|---:|---:|---:|---:|---:|---:|---|
| 1 | -44.353% / -54.153% | -8.720% / -10.261% | -79.717% / -87.485% | -87.578% / -96.353% | 0.9914 | no |
| 2 | -44.715% / -54.072% | -8.818% / -10.428% | -80.759% / -89.045% | -88.037% / -96.781% | 0.9861 | no |
| 4 | -22.009% / -29.694% | **+4.963% / +3.356%** | -97.295% / -108.121% | -55.963% / -63.707% | 0.9998 | no |

Against the fresh parent at NFE-1, the candidate was also worse at every NFE.
At NFE-1 its decoded, temporal, LPIPS, and latent effects were -45.090%,
-8.878%, -79.869%, and -88.233%; at NFE-4 they were -50.946%, -13.153%,
-45.532%, and -93.485%. P95 latency parity passed at all three NFEs: candidate
times were 0.41054, 0.48904, and 0.64791 seconds per two-sample batch.

The full 2x2 objective/readout factorial localizes the failure. Applying the
consistency readout to the RF-trained model was very poor. Consistency training
partly repaired that readout, but its native-RF path deteriorated as the sealed
training trace predicted. The best diagnostic--not selectable--was the online
consistency model at NFE-4: temporal MSE improved 6.180% versus online RF, but
decoded MSE, LPIPS, and latent NMSE worsened 2.258%, 17.778%, and 14.418%.
This is a temporal-only tradeoff, not a better quality frontier. The saved
qualitative evidence shows the same failure as visible spatial
smearing/mosaicing rather than a metric-only disagreement.

The Boolean action guard passed only because it was a relative non-degradation
guard. Actual action sensitivity was essentially zero: at NFE-1, shuffled
actions changed parent decoded/temporal MSE by -0.0131%/-0.0036% and candidate
MSE by +0.0144%/-0.0004%. Even at NFE-4 the candidate effects were only
+0.1165%/-0.0125%. Aligned-to-shuffled actions changed both latent and decoded
hashes for 256/256 pairs at every NFE, so the action path is computationally
active; it simply has no demonstrated target-aligned quality sensitivity. No
action-controllability conclusion is supported.

The evaluation inventory contains exactly 9,216 rows (64 clips x four noises x
12 endpoints x three NFEs), 43,680 observed Wan calls, and zero teacher,
feature, protected-test, or W&B calls. All 4,608 batch endpoints were
materialized and hash-closed before future RGB was opened. Independent replay
validated all 9,233 identity payloads, all file hashes, all row pairings, and
the exact frozen analysis. Analyzer-bound use was 9.18363 B200-hours; full
Slurm accounting including memory smoke and startup/teardown was approximately
9.7422 B200-hours. Artifact use was 16,543,983,840 bytes. Both remained within
the registered ceiling. Inventory
identity is `1d13287243b01101a9112647e20681f9804592444ecc2def177c981a428efb9c`;
analysis identity is
`caf55629f5914fe52250fe395d5fe63f8c223d750ad44968915bb3a8eb34e826`
(file SHA-256
`a14b9af6c6b871e6d8f68ee3550ecdcfa86f35a22742d841d314dc5cf946457c`).

This remains a one-training-seed, 64-clip development screen. Its frozen
bootstrap treats four clip/noise pairs as separate resampling units rather
than clustering all noises within 64 clips, so its intervals may be optimistic.
The action gate has no absolute sensitivity floor. It measures neither FVD,
long-horizon rollout, closed-loop DAgger utility, nor 5--10 Hz serving. A
six-second read-only `nvidia-smi` step was recorded inside the allocation; no
primary timing distribution had a value above twice its median, and the quality
failure is invariant to latency. These limitations do not rescue ACD-P0, but
they bound the claim against broader consistency methods.

## Canonical evidence

Raw-flow v8 canonical root:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/raw_physics_flow_stage1/
  raw-physics-flow-stage1-20260809-3424c73-v8-exploratory
```

Its registration, inventory, analysis, and audit identities are respectively
`83582e7ebe5cd3857831508f4fbec4734fc38e50d60d7a63614330bddc167209`,
`232506d33d292fd23842b0a85ddf960f27fd6bdde449008ba377ebfb52f3dabd`,
`721724ccef367a9c7e17986780d8be7c54e5374d4df9bf81aeb30ec374316c27`, and
`92f29b7d088e086f5337b8574e6b8ac37ce5184facd4fa015004dfe1a7a71edc`.

ACD-P0 canonical root:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/acd_p0/
  acd-p0-20260809-6063ec5-v7
```

The official result files are `acd_p0_result.json` and `acd_p0_result.md` at
that root. A local post-run visual contact sheet is stored outside Git at
`artifacts/acd_p0/acd-p0-20260809-6063ec5-v7/acd_p0_contact_sheet.png`, with
SHA-256 `223225be4855456b969c42d7eb15da6a4bb7c56a2214cc10c847c2a4a505cde5`.

Recurrent-delta renderer confirmation canonical root:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/recurrent_delta_spatial_confirmation/
  recurrent-delta-spatial-confirmation-renderer55-seed20260809-92cc958-v1
```

Its registration, causal-closure, evaluation, analysis, and completion
identities are respectively
`db1c29e5a89c35b8b1b8c9a4ba8d1be2ef796c3793961fcd4204db46d363d42d`,
`0b8f50bf3ac84e9932870b3c7cf430618bc49006f1306cf869328a5f65ebdcfc`,
`d0f09573e9f91f84c86007730f39a0fd348d6bd6dc723859de9bdcd329525483`,
`2426a1546a43403fc09f7d8bb60475526e4d342586727c880dd005aad99010f8`,
and `a94afb1e39ccee4b80298182f287431409fbce47ae260c94d4324fa23081e5ff`.

Recurrent-flow cache handoff canonical root:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/artifacts/dual_video_diffusion/recurrent_physics_flow_cache/
  recurrent-flow-wan-103e2d0-v1
```

Standalone strict-audit job `508106` completed `0:0` and reproduced the GO
after 731 record checks over 356 unique files / 9,069,194,268 referenced bytes,
25 reconstructed causal arrays with maximum error zero, all 2,200 transition
rows, all 275 episode/arm rows, and the exact `[100000,55]` bootstrap. Its
external receipt identity is
`d7afdf0bddcd8dee32351b2356137503cf5485dcdaa5f8e0b699a63d1dc86b69`
(file SHA-256
`8e0475d40aabb7ed3d200ef900cf012e58f0096b27fa46642384869b92f545ea`).
The first audit job `508100` failed closed on an auditor-only fixed-width
assumption; the repair independently binds each bundle to its variable native
raw-MCAP camera geometry. No canonical evidence was mutated.

## Recurrent-flow generator bridge

The renderer GO has been advanced only across the causal cache/evidence seam,
not yet into a Wan efficacy claim. Source commit
`103e2d019b12ee2a88519a96afaf91de4157305e` implements a distinct
`recurrent-physics-flow-cache-v1` family, an inference-only recurrent predictor,
five frozen condition sources, a family-bound dataset/model adapter, and an
NFE-1 seven-endpoint protocol. The exact source bundle SHA-256 is
`8a34d80377461b474d563624f7fb6cc0bf401652f7887484756595e53563316e`.
Cluster validation passed `49/49` tests in the main Torch/HDF5 environment and
`13/13` applicable tests in the MuJoCo cache environment, with only the
HDF5-dependent dataset test skipped there. A NumPy-2.0-specific fixture failure
was repaired and then reproduced green rather than waived.

Registration froze all 576 train/validation causal prefixes before any field
rendering. Each row binds 280 observed-state bytes and one logical 3,640-byte
candidate-action slice; it records zero returned future-state bytes, no future
RGB opening, no protected-test access, and no generator outcome. The privileged
confirmation NPZ is whole-file hashed but never copied into runtime; only its
explicit predictor whitelist is deserialized into a 283,672-byte inference-only
archive. Registration identity is
`c1c9793df19caa3ce32c662eef3aaef39403afbf8dfdaf26ba065fd3e89e5a44`;
its file SHA-256 is
`f1e542193a31e764e063bf6ab8ebfa5c4d7fe49cc267ae65893f044acbb18c14`.
The rebound cache-runtime receipt is
`f0c831c9db48bec33c215fbff34281f9bdfc0f976c11acabecedbf6656a8100d`;
it records raw-v7 receipt identity
`8cb5b92d5fde78e2379a72e33f2d9cd6bc76afc43b0c7b16ab200f7a1330e99d`
as its base, then reseals after changing only the byte-identical helper's
canonical checkout path.

An independent source audit found and blocked an alternate-checkout provenance
gap before materialization. The repaired bridge requires registration, build,
and replay to execute from the same physical, clean checkout with exact commit,
Git tree, source-file inventory, and non-symlinked evidence paths. The audit's
current decision is GO for cache-only build/full replay and strict NO-GO for Wan
training: frozen arm registration, recurrent-specific matched training,
rank-wise pairing receipts, a control-capable target-opening-safe evaluator,
parent parity, and preregistered analysis are not implemented. The planner has
no submission path and must remain `launch_ready: false`.

The immutable `19490c5` v1 registration and jobs `508144`, `508145`, `508148`,
`508149`, `508160`, and `508161` are retained as failed operational attempts.
The first pair used a non-portable shell option, the second omitted the sealed
no-user-site flag, and the third exposed that the inherited raw-v7 receipt
encoded its old source path. All failed before split creation. The last issue
was repaired by byte-checking, resealing, and independently reviewing the
runtime receipt rather than weakening runtime equality.

Canonical jobs `508171` (train) and `508172` (validation) then completed
`0:0`. Train rendered in 100.885 seconds and replayed all 512 causal rows,
including 415 eligible rows / 830 aligned-and-hold arrays; validation rendered
in 12.255 seconds and replayed all 64 rows, including 48 eligible rows / 96
arrays. Both full replays were byte-exact with maximum absolute error zero.
Train metadata/audit identities are
`3f965ee76dc4722565de6609e683fc394deaf1f04f54900a9af83d122af72c85` /
`a61a4f53a0560b8d85cbb5ca4cd8629d745c3539621a89cd28c7431ab9413cca`;
validation identities are
`01f41160c91519e473faaf3fd26dc381f24c9feaa42f83fe915c14112f84fc50` /
`bdb7c53b3caf8c0eb7b3a85aa53ffbc34b733406c8e8c83b7769becf4334c892`.
The complete cache occupies 147,081,947 bytes. Its dry-run plan identity is
`742abe01a53e621e6910670a2559283c7e9a72dc3895ef22f7316bb617a5eabd`;
the externally stored plan file SHA-256 is
`92a9ad57756d7c6b1693254bc633bb9ca4490a000ca33e13dfc71993207300af`.
The plan reports `launch_ready: false`, `submission_performed: false`, and a null
submission command. Thus no generated-video endpoint, quality gain, or Wan
training result has been produced by this handoff.

A posthoc, non-outcome diagnostic then checked condition distinguishability
without opening future RGB or a generator. Every eligible aligned row differed
from off, shuffled, wrong-time, and hold in both splits. Shuffled fields had
mean relative L2 distance `1.0452` (train) / `1.0538` (validation), while the
harder wrong-time control remained nondegenerate at `0.4235` / `0.4202` and
hold at `0.4455` / `0.4265`. Mean aligned support was `15.79%` / `15.13%` of
the pooled spatiotemporal field. This shows the planned controls are numerically
separable; it says nothing about whether Wan will use them. Diagnostic identity
is `f91debf83e94aa2a1a5792dca2850715eb9f0ed0c6fe975a8f4c14b15a10eab2`
and file SHA-256 is
`5adc647ccc912839d50e9278ae0be2922ece0546ec02660bb39d4fb49c666ba5`.

A second posthoc cache-only diagnostic measured whether zero-training
substitution would be grossly out of distribution for the raw-flow adapter.
Recurrent versus raw aligned fields had mean cosine `0.9727` train / `0.9739`
validation, support ratios `0.9903` / `0.9923`, and relative L2 distances
`0.2115` / `0.2068`. Recurrent visible-motion magnitude averaged `0.8907` /
`0.8876` of raw. This does not prove compatibility, but it rules out a simple
support/scale collapse and makes a frozen-checkpoint substitution screen more
diagnostic. Its identity is
`589cf1398ad7014086fd59a81f66105177de1330b73b6e1fc443c00e5300b7c1`
and file SHA-256 is
`29cbc6c4e0df57a51cd50d11eaac8c38e19c36f48942e7a8fd3a0927a3f47233`.

Stored cache telemetry gives only a component-level speed bound. The sum of
nine per-pose renderer timers for one aligned trajectory averaged 51.16 ms in train and 44.48 ms in
validation (p95 57.62 / 58.28 ms) on the allocated B200. This excludes state
ingestion, recurrent prediction, packing, Wan, decode, and serving overhead;
the cache job also rendered a hold control. It therefore neither establishes
5--10 Hz generation nor rules it out. The earlier measured NFE-1 Wan path, not
geometry rendering, is the larger measured component in this evidence set, but
the two timings are not an end-to-end serving benchmark.

## Claim boundary

The completed screens support a narrower and more useful conclusion: clean
future features contain a strong high-noise oracle signal, but no tested
inference-causal TF, semantic, residual, fixed-subspace, action, or fixed-flow
mechanism has converted it into a material, attributed gain over VPM@1. This
rejects the broad claim that an arbitrary auxiliary state supplies a
Latent-Forcing-like video advantage. It does not reject dual diffusion,
structured learned causal fields, or full consistency methods in general. A
positive claim requires an inference-available state, a matched stronger
baseline, a prospective low-NFE endpoint, action sensitivity, and a gain large
enough to survive uncertainty and complete latency accounting.

The recurrent delta-response robot-state predictor has now cleared the causal
renderer prerequisite. The exact fit384 model was frozen and evaluated on the
complete 55-episode renderer-endpoint-unopened D405 census. Mean rendered robot-flow EPE
fell from `4.628886` to `2.690298` pixels versus raw planned commands
(`+41.880%`), from `4.120211` to `2.690298` versus absolute ridge (`+34.705%`),
and from `22.943593` to `2.690298` versus episode-shuffled commands
(`+88.274%`). All three flow-superiority and all six spatial non-inferiority
gates passed their separate Holm families. The tightest ridge-IoU lower bound
was `-0.002832` against the frozen `-0.005` margin; a parametric sensitivity
misses that margin narrowly, so the result authorizes a controlled generator
screen, not a broad spatial-equivalence claim.

This confirmation is train-only and endpoint-specific. Metadata auditing found
that all 55 episodes had appeared in unrelated experiment populations, although
their renderer endpoints had not been opened. No Wan call or generated-video
metric was involved. The exact result and repair chronology are recorded in
`RECURRENT_DELTA_SPATIAL_CONFIRMATION_RESULT.md`.

## Final handoff

Raw-flow v8 is terminal at `STOP_FIXED_RAW_FLOW`; ACD-P0 is terminal at
`NO_GO_ACD`. Do not extend the same ACD objective merely because its training
loss was still falling, and do not launch a stochastic object/contact residual
that was conditional on raw-flow success.

The justified next experiment is now an equal-Wan-call recurrent-flow screen
against VPM@1 with recurrent-flow off, aligned, episode-shuffled, held-current,
and nonwrapping wrong-time controls. First substitute the recurrent field into
the frozen RAW-FLOW checkpoint to separate field quality from retraining. A
clear generated-video gain that disappears under controls may authorize matched
recurrent-field training; otherwise stop without spending another full training
pair. A full Causal-rCM/Flash-WAM-style consistency recipe remains a separate
baseline-reproduction project; this 400-update ACD-P0 result must not be
generalized to those untested methods.

Interpret that substitution conservatively. The recurrent/raw cache shift is
modest enough that gross support/scale mismatch is not the obvious confound,
but the frozen raw-flow checkpoint previously changed decoded and temporal MSE
by only `+0.213%` and `+0.122%` when its condition was toggled on. A negative
substitution therefore rejects usefulness through that frozen adapter; it does
not show that the recurrent field is inaccurate. Any later condition-dropout,
flow-reconstruction, or stronger-gating recipe would be a separately
preregistered generator-utilization hypothesis, not a posthoc rescue of this
screen.

The raw Stage-1 controller cannot safely launch this screen unchanged: it
hard-codes raw schemas, arm names, endpoints, trace kinds, and v5/v7 lineage.
The smallest safe next controller must freeze the exact Holm family, score
future-latent NMSE with the immutable parent tokenizer, prove cross-arm initial
adapter equality, assert recurrent family/schema/SHA before update zero, and
seal the complete transitive source inventory. The zero-training NFE-1 grid is
48 distinct validation episodes x four noises x seven endpoints = 1,344 rows
and 672 two-sample Wan batches. Only after that evaluation-only controller and
its independent readiness audit exist should the frozen raw adapter be tested;
matched recurrent training remains a later, separately paired stage.
