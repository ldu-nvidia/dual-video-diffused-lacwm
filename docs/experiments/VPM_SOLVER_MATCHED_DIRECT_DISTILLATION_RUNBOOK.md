# ACD-P0 execution runbook

Status: implementation prepared; no job launched from this worktree. The
default workflow is plan-only. A submission command requires both `--submit`
and a fresh, phase-specific user-authorization receipt.

## Frozen scope

ACD-P0 is the feature-free low-NFE comparator, not a dual-diffusion arm. It
trains two 400-update, eight-B200 arms from faithful parent `de65…`: RF MSE
with native RF Euler and adjacent consistency with the trained Karras
consistency readout. Evaluation is the registered 12-endpoint objective/readout
and action-intervention family at NFE 1/2/4, four seeds, and development 64.
W&B and the protected test split are forbidden.

Set site-specific absolute paths without writing into the source checkout:

```bash
export ACD_SOURCE=/path/to/clean/dual-video-diffused-lacwm
export ACD_COMMIT=$(git -C "$ACD_SOURCE" rev-parse HEAD)
export ACD_STORAGE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
export ACD_ROOT="$ACD_STORAGE/artifacts/dual_video_diffusion/acd_p0/run-id"
export ACD_RECEIPTS="$ACD_STORAGE/artifacts/dual_video_diffusion/acd_p0/receipts"
export ACD_PYTHON=/absolute/path/to/lacwm-venv/bin/python
export ACD_EXTERNAL_REPOS=/absolute/path/containing/Causal-Forcing-Flash-WAM-rcm
```

The exact commit must be clean and reachable from the audited GitHub remote.
All output, log, and receipt paths must be under `/mnt/data1`, `/mnt/data2`, or
the exact user root
`/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train`.
No other Lustre prefix is admitted. Required parents must already exist.

## 1. Exact-source test receipt

This runs only the frozen ACD unit/static suite. It submits no job and opens no
training or endpoint arrays.

```bash
"$ACD_PYTHON" "$ACD_SOURCE/tools/low_nfe_direct_baseline.py" test-report \
  --source-repo "$ACD_SOURCE" \
  --expected-commit "$ACD_COMMIT" \
  --python "$ACD_PYTHON" \
  --videox-home /absolute/path/to/VideoX-Fun \
  --output "$ACD_RECEIPTS/exact_source_tests.json"
```

## 2. Source readiness and one-B200 memory smoke

First render source readiness without a memory receipt:

```bash
"$ACD_PYTHON" "$ACD_SOURCE/tools/low_nfe_direct_baseline.py" readiness \
  --source-repo "$ACD_SOURCE" \
  --expected-commit "$ACD_COMMIT" \
  --external-repos-root "$ACD_EXTERNAL_REPOS" \
  --test-report "$ACD_RECEIPTS/exact_source_tests.json" \
  --output "$ACD_RECEIPTS/source_readiness.json"
```

The expected status is `READY_SOURCE_MEMORY_PREFLIGHT_REQUIRED`. Render the
memory-smoke submission plan with
`tools/slurm/adjacent_consistency_workflow.py`; submission requires a fresh
authorization receipt for phase `memory_smoke`. The Slurm job executes one
synthetic `[1,13,3,180,960]` forward/backward/optimizer/EMA transaction with
three full model copies on one B200. It opens no training/validation data and
must retain at least 16 GiB headroom with at most 90% reserved memory. Before
the transaction it independently recomputes the parent's canonical
`snapshot_model_state_receipt_v1` hash (`d123…`) and runtime
`tensor_state_sha256_v1` hash (`82ff…`), then requires the strictly loaded
student, teacher, and EMA target to match `82ff…`. Comparing a runtime hash to
the canonical value is a contract failure, not a checkpoint mismatch.

Preflight job `507903` on source `9348814…` passed those dual-hash checks and
reached the synthetic forward, then failed before backward. It produced no
passing memory receipt and opened no training or endpoint data. The cause was
the old constant-zero synthetic RGB tensor: production `_build_loss_mask`
correctly classified all three constant views as absent, so the future loss
mask was empty. The all-true 13-frame temporal mask was already correct and is
preserved. Receipt schema 2 instead requires the dataset-free deterministic
bounded fixture to prove three nonconstant views, five valid history frames,
eight valid future frames, two valid future latent tokens, and positive support
through the production mask builder and `_expanded_mask` before forward.

After it passes, create the registration-ready seal:

```bash
"$ACD_PYTHON" "$ACD_SOURCE/tools/low_nfe_direct_baseline.py" readiness \
  --source-repo "$ACD_SOURCE" \
  --expected-commit "$ACD_COMMIT" \
  --external-repos-root "$ACD_EXTERNAL_REPOS" \
  --test-report "$ACD_RECEIPTS/exact_source_tests.json" \
  --memory-smoke-receipt "$ACD_RECEIPTS/memory_smoke.json" \
  --output "$ACD_RECEIPTS/registration_readiness.json"
```

The expected status is `READY_FOR_REGISTRATION`.

## 3. Prospective registration

Registration byte-hashes the actual train RGB/action arrays, validates their
NumPy schemas, keeps validation bytes deferred, composes both full resolved
Hydra jobs, binds every semantic field into the arm identities, and seals both
non-interchangeable parent model-state hashes plus their algorithm names.

```bash
"$ACD_PYTHON" "$ACD_SOURCE/tools/low_nfe_direct_baseline.py" register \
  --output "$ACD_ROOT" \
  --source-repo "$ACD_SOURCE" \
  --expected-commit "$ACD_COMMIT" \
  --readiness-seal "$ACD_RECEIPTS/registration_readiness.json" \
  --memory-smoke-receipt "$ACD_RECEIPTS/memory_smoke.json" \
  --parent-snapshot /absolute/path/to/de65/snapshot.pt \
  --parent-resolved-config /absolute/path/to/de65/resolved_update_1000.yaml \
  --train-manifest /absolute/path/to/train/clip_manifest.jsonl \
  --train-cache-metadata /absolute/path/to/train/metadata.json \
  --val-manifest /absolute/path/to/val/clip_manifest.jsonl \
  --val-cache-metadata /absolute/path/to/val/metadata.json \
  --python "$ACD_PYTHON" \
  --wan-dir /absolute/path/to/Wan2.1 \
  --videox-home /absolute/path/to/VideoX-Fun \
  --lpips-linear-weight /absolute/path/to/lpips/alex.pth \
  --alexnet-weight /absolute/path/to/alexnet-owt-7be5be79.pth
```

Registration requires at least 100 GiB free and a fresh output root. It does
not submit jobs. Inspect the exact no-write plan with:

```bash
"$ACD_PYTHON" "$ACD_SOURCE/tools/low_nfe_direct_baseline.py" plan \
  --registration "$ACD_ROOT/registration.json"
```

## 4. Guarded training and evaluation

Use `tools/slurm/adjacent_consistency_workflow.py` in its default plan mode.
For either training arm, the authorization receipt must bind the exact source
commit, registration identity, phase, and arm and must still be fresh. The
wrapper revalidates registration, exports its exact arm/config identities, and
passes the registered save and visualization paths. Rank zero rehashes all
train RGB/action bytes before each arm; all ranks compare full initial/final
state receipts. The trainer rechecks the raw parent under both algorithms and
the strictly loaded model/teacher/EMA state under the runtime algorithm before
the first update. Either mismatch stops before useful training.

Evaluation authorization is accepted only after both independent 400-update
completion receipts exist. The evaluator materializes and hashes every
endpoint before the global future-target barrier, then opens validation RGB
for scoring. It uses fresh unbound models, exact Wan-call hooks, pinned LPIPS
weights, episode-disjoint action shuffling, and no teacher/feature calls.

## 5. Frozen analysis

After `evaluation/inventory.json` exists:

```bash
"$ACD_PYTHON" "$ACD_SOURCE/tools/analyze_adjacent_consistency.py" \
  --registration "$ACD_ROOT/registration.json" \
  --evaluation "$ACD_ROOT/evaluation" \
  --output-json "$ACD_ROOT/acd_p0_result.json" \
  --output-markdown "$ACD_ROOT/acd_p0_result.md"
```

The analyzer requires exact cross-endpoint history/noise/target hashes and
aligned/shuffled action hashes before computing its frozen 10,000-bootstrap
gate. A pass is evidence that direct feature-free consistency is the stronger
low-NFE baseline. It is not evidence for time-frequency or other dual-state
novelty.

## Resource and stop contracts

Maximum reservation is 113 B200-hours: one B200-hour memory smoke, two
eight-B200 six-hour training allocations, and one eight-B200 two-hour
evaluation. Persistent artifacts are capped at 100 GiB; jobs are non-requeue.
Stop on any source/config/data/state/call/access mismatch, nonfinite update,
AMP-skipped optimizer transaction, memory-gate failure, or authorization
failure. Do not repair, resume, retune, or select an unregistered endpoint
after observing development outcomes.
