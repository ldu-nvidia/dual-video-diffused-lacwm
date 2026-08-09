# Recurrent-Delta Flow → Wan NFE-1 Screen

Status: **source implemented; launch disabled pending frozen study registration,
trainer/evaluator binding, and independent readiness audit**.

This is the smallest causal handoff justified by the completed renderer study.
It asks one question: does replacing raw-command geometry with the sealed
recurrent-delta geometry make a fixed-field Wan continuation measurably better
at one denoising step, and is any gain attributable to the aligned field?

## Immutable prerequisite

Only this completed artifact may authorize the bridge:

`/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/recurrent_delta_spatial_confirmation/recurrent-delta-spatial-confirmation-renderer55-seed20260809-92cc958-v1`

| Receipt | Required identity |
|---|---|
| source commit | `92cc9584298d2a395f3a79152768c9d0b88b3864` |
| registration | `db1c29e5a89c35b8b1b8c9a4ba8d1be2ef796c3793961fcd4204db46d363d42d` |
| preparation | `ea0c1383da10cb654aea3dd3d6682312fc00d1f68ace13b38f65587e6990397e` |
| trajectory closure | `0b8f50bf3ac84e9932870b3c7cf430618bc49006f1306cf869328a5f65ebdcfc` |
| analysis | `2426a1546a43403fc09f7d8bb60475526e4d342586727c880dd005aad99010f8` |
| completion | `a94afb1e39ccee4b80298182f287431409fbce47ae260c94d4324fa23081e5ff` |
| frozen source NPZ | `8602fda20b3f0c8365c69efcc48e363a8e8f93666725e1af8665a18d03e80688` |
| external strict audit | identity `d7afdf0bddcd8dee32351b2356137503cf5485dcdaa5f8e0b699a63d1dc86b69`, file SHA `8e0475d40aabb7ed3d200ef900cf012e58f0096b27fa46642384869b92f545ea` |
| external auditor commit | `ff3244912bfe62dfa0208a4a38cdbc1c1053f40f` |
| decision | `GO_RECURRENT_DELTA_WAN_SCREEN` |

All three flow-superiority and all six spatial-noninferiority gates must be
true. There is no partial-GO path.

## Privileged-artifact boundary

The frozen source NPZ contains training/score products, including
`target_measured_trajectory`. It is therefore **not a runtime artifact**.
Registration derives a new inference-only NPZ by opening exactly these members:

- history and action normalization/PCA tensors;
- recurrent ridge normalization, coefficient, and intercept tensors;
- `recurrent_alpha` and `recurrent_gain`.

The extractor uses individual ZIP-member reads, never `np.load(source_npz)`,
and records each member deserialized. The source archive is whole-file SHA-256
hashed, so its opaque bytes—including bytes belonging to forbidden members—are
physically traversed for identity. That is explicitly recorded as
`source_archive_bytes_hashed: true`; it is not claimed as byte-level unopened.
Score indexes, score contexts, stored trajectories/predictions, absolute-ridge
state, and measured targets are never deserialized or semantically loaded and
the privileged archive is never copied into runtime. The resulting runtime
archive must contain the whitelist exactly. Any extra runtime member,
pickle/object dtype, hash mismatch, or forbidden-member deserialization is
terminal.

## Causal input and representation

Registration and build each use the same strict reader. Registration seals the
value hashes for every train/validation row before cache materialization; build
must reproduce the registered hashes exactly. For each clip it reads exactly:

- 280 bytes of measured robot state at the five observed video indexes;
- one logical 3,640-byte registered candidate-action slice (joint and gripper
  members each have one contiguous `pread`);
- registered D405 calibration and pinned official ABC robot geometry.

It reads no state after the fifth observed frame and no future RGB. The sealed
predictor rolls out nine robot poses from observed history plus planned actions.
The existing analytic renderer converts the eight pose transitions to
`[8,4,24,40]` float16 fields: normalized `dx`, normalized `dy`, visibility, and
log-depth ratio. Packing remains exactly `[16,4,24,120]`; only the top-view
third and two future Wan tokens are populated. The predictor runs at cache time,
not inside the Wan sampling loop.

The cache schema is `recurrent-physics-flow-cache-v1`, family
`sealed_recurrent_delta_geometry`. It is deliberately not
`raw-physics-flow-cache-v3`, makes no v5 raw-cache numeric-equivalence claim,
and cannot be loaded by the raw-family dataset validator.

## Frozen conditions

| Condition | Definition |
|---|---|
| `recurrent_aligned` | same-episode recurrent-delta trajectory |
| `off` | exact-zero packed tensor, constructed at runtime |
| `recurrent_shuffled` | complete aligned tensor from the already-frozen episode-disjoint same-motion-stratum donor |
| `recurrent_wrong_time` | transition `t+1` injected at `t`, no wrap; final transition is zero |
| `recurrent_hold` | observed fifth-frame pose held for all nine poses |

Shuffling a complete tensor (rather than only actions) tests whether Wan uses
episode-aligned geometry. Wrong-time tests temporal registration; hold tests
whether static robot support alone explains an effect.

## Matched continuation

Only two 200-update continuations are allowed, both warm-started model-only
from the same canonical VPM@1 parent:

| Arm | Aligned cache supplied | Adapter fused |
|---|---:|---:|
| `FLOW-OFF` | yes, byte-identical loader path | no |
| `RECURRENT-FLOW` | yes | yes |

Seed, clip order, RGB/actions, video timestep/noise, optimizer, learning rate,
batch size, Wan call count, parent weights, excluded auxiliary keys, and update
count must match exactly. Each example executes one Wan call and zero online
predictor/teacher/feature calls. Exact all-rank hashes must prove paired clip,
action, timestep, noise, and field streams. No validation loader may run during
training.

## Frozen seven-endpoint evaluation

The first screen is NFE-1 only:

1. native parent VPM, off;
2. matched `FLOW-OFF`, off;
3. `RECURRENT-FLOW`, aligned;
4. `RECURRENT-FLOW`, off;
5. `RECURRENT-FLOW`, shuffled;
6. `RECURRENT-FLOW`, wrong-time;
7. `RECURRENT-FLOW`, hold.

Each endpoint uses the same four stateless noise seeds and exactly one Wan
transformer call. Every endpoint/noise output for a batch must be resident on
CPU before future RGB or clean video latents are opened. Score top-view decoded
MSE, temporal MSE, AlexNet LPIPS, and future-latent NMSE; retain all-view MSE
and latency as diagnostics. Clip is the resampling unit; noise repeats stay
clustered within clip.

The candidate advances only if the preregistered paired analysis shows:

- versus **both** matched off and the native parent, aligned improves top-view
  decoded MSE and top-view temporal MSE by at least 3%, with clustered 95%
  lower bounds strictly above 1%;
- versus both baselines, top-view LPIPS has a clustered lower bound strictly
  above zero;
- versus both baselines, all-view decoded MSE, all-view temporal MSE, and
  future-latent NMSE each have nonnegative point improvement and clustered
  lower bound above -1%;
- aligned beats candidate-checkpoint off, shuffled, wrong-time, and hold by at
  least 1% point estimate with clustered lower bound above zero on **both**
  top-view decoded and temporal MSE;
- all registered call, causality, tensor-identity, source, and latency audits
  pass.

Use a common paired episode bootstrap and Holm correction across the four
attribution comparisons. Failure of any required member yields
`STOP_RECURRENT_FLOW_WAN`. Passing yields only
`GO_RECURRENT_FLOW_NFE_FRONTIER`; it does not establish DAgger speed, FVD, or
policy benefit. NFE2/4, distillation, and real-time claims remain unopened.

## Operator runbook (currently cache-only)

Pinned reusable input registration:

`/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/raw_physics_flow_cache/raw-physics-flow-cache-20260808-19717d3-v7/cache_registration.json`

After archiving a clean committed source tree on the cluster, set:

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
SRC="$BASE/src/recurrent-flow-wan-<COMMIT>"
CONF="$BASE/artifacts/dual_video_diffusion/recurrent_delta_spatial_confirmation/recurrent-delta-spatial-confirmation-renderer55-seed20260809-92cc958-v1"
RAW_REG="$BASE/artifacts/dual_video_diffusion/raw_physics_flow_cache/raw-physics-flow-cache-20260808-19717d3-v7/cache_registration.json"
OUT="$BASE/artifacts/dual_video_diffusion/recurrent_physics_flow_cache/recurrent-flow-wan-<COMMIT>-v1"
STRICT_AUDIT="$BASE/artifacts/dual_video_diffusion/recurrent_delta_spatial_confirmation/strict_audits/recurrent-delta-spatial-confirmation-renderer55-seed20260809-92cc958-v1/strict_postrun_audit-ff32449.json"
MAIN_PY="$BASE/envs/lacwm-b200-py310/bin/python"
CACHE_PY="$BASE/envs/interaction-event-py310-v1/bin/python"
```

Then verify and register without opening any condition array:

```bash
"$MAIN_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" verify-confirmation \
  --confirmation-root "$CONF" --strict-audit "$STRICT_AUDIT"

"$MAIN_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" register-cache \
  --source-repo "$SRC" \
  --expected-commit "$(git -C "$SRC" rev-parse HEAD)" \
  --raw-cache-registration "$RAW_REG" \
  --confirmation-root "$CONF" \
  --strict-audit "$STRICT_AUDIT" \
  --output "$OUT"
```

The exact allocated-node commands are:

```bash
"$CACHE_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" build-cache \
  --registration "$OUT/cache_registration.json" --split train
"$CACHE_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" audit-cache \
  --metadata "$OUT/train/metadata.json" --split train --full-replay
"$CACHE_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" build-cache \
  --registration "$OUT/cache_registration.json" --split val
"$CACHE_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" audit-cache \
  --metadata "$OUT/val/metadata.json" --split val --full-replay
```

Dry-run planning never submits:

```bash
"$MAIN_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" plan \
  --registration "$OUT/cache_registration.json"
```

The optional explicit operator switch is receipt-only and still cannot submit:

```bash
"$MAIN_PY" "$SRC/tools/recurrent_flow_wan_bridge.py" plan \
  --registration "$OUT/cache_registration.json" \
  --authorize-screen \
  --authorization-token AUTHORIZE_RECURRENT_FLOW_WAN_NFE1
```

`plan` must continue to report `launch_ready: false`,
`submission_performed: false`, and no submission command until a later commit
adds and independently audits the non-masquerading trainer, seven-endpoint
evaluator, frozen study registration, parent parity receipt, and analysis.
