# CSIP Phase-0 execution runbook

This runbook is prepared but has not been executed. The reviewed launcher uses
two non-requeueable one-node/eight-B200 jobs with an `afterok` dependency: the
first extracts train latents, fits both matched probes, and seals update 400;
the second opens val64 only after that seal, then evaluates and analyzes. Do not
launch while another eight-B200 study owns the intended allocation unless its
numeric job ID is explicitly included in the exact active-job allowlist.

## Pinned cluster paths

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
REPO=$BASE/src/dual-video-diffused-lacwm-csip-phase0
PY=$BASE/envs/lacwm-b200-py310/bin/python
CACHE=$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690
PARENT=$BASE/artifacts/dual_video_diffusion/csip_phase0
FINAL_COMMIT=<reviewed-40-character-commit>
SHORT=${FINAL_COMMIT:0:7}
STUDY_ID=csip-phase0-seed1234-20260808-$SHORT-v1
STUDY=$PARENT/$STUDY_ID
REG=$STUDY/registration.json
```

Large artifacts remain under Lustre, never the repository or root filesystem.
The known immutable source paths are:

```text
train manifest: $CACHE/manifests/train.jsonl
train metadata: $CACHE/caches/train/metadata.json
val manifest:   $CACHE/manifests/val.jsonl
val metadata:   $CACHE/caches/val/metadata.json
Wan root:       $BASE/wan_fun_1.3b_control
VideoX:         $BASE/VideoX-Fun-1d6d9c3
```

## 1. Independent pre-launch checks

Confirm no active job is using the intended GPUs and that the checkout is the
reviewed clean commit. Never stop another run for this screen.

```bash
git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --short
squeue -u "$USER"
```

The first command must equal `FINAL_COMMIT`; the second must emit nothing.
Review the protocol, all source changes, test output, expected artifact size,
and W&B destination before continuing.

## 2. Verify the personal private W&B destination

```bash
"$PY" "$REPO/tools/csip_workflow.py" wandb-check \
  --entity zijiandu \
  --project dual-video-diffusion-private
```

Require `access=PRIVATE`. Do not proceed with a team/group project, a different
entity, or a failed identity lookup.

## 3. Register before extraction or fitting

Registration fully hashes both source caches and the Wan assets, so allow time
for I/O. It creates `STUDY`; therefore the study root must not exist first.

```bash
mkdir -p "$PARENT"
"$PY" "$REPO/tools/csip_workflow.py" register \
  --study-id "$STUDY_ID" \
  --study-root "$STUDY" \
  --tool-repo "$REPO" \
  --expected-commit "$FINAL_COMMIT" \
  --python "$PY" \
  --wan-dir "$BASE/wan_fun_1.3b_control" \
  --videox-home "$BASE/VideoX-Fun-1d6d9c3" \
  --train-manifest "$CACHE/manifests/train.jsonl" \
  --train-cache-metadata "$CACHE/caches/train/metadata.json" \
  --validation-manifest "$CACHE/manifests/val.jsonl" \
  --validation-cache-metadata "$CACHE/caches/val/metadata.json" \
  --wandb-entity zijiandu \
  --wandb-project dual-video-diffusion-private \
  --ack-private-wandb-project
```

There is deliberately no test argument. Preserve the registration stdout and
file hash in the launch-review record.

## 4. Render and review the train stage

```bash
"$PY" "$REPO/tools/csip_workflow.py" render --registration "$REG" \
  --stage train \
  > "$STUDY/rendered-commands.review.json"
```

Because study files are immutable outputs, writing the review capture inside
the study is optional operational evidence and is not consumed by any tool.
The train render validates train records only and records that validation
sources were not opened. A validation render intentionally fails before the
checkpoint seal exists. Review the rendered commands and
`tools/slurm/csip_phase0_stage.sbatch`. The stage entrypoint revalidates the
exact clean registered source and VideoX checkouts, registered Python, complete
B200 environment, output freshness, and non-requeue state before reading data.
It writes the stage-specific boundary receipt to
`$STUDY/runtime/<stage>-job-<job-id>/stage-boundary-receipt.json`.

Dry-run the dependency-safe submitter first. It changes no directory or job:

```bash
"$REPO/tools/slurm/submit_csip_phase0.sh" \
  --registration "$REG" \
  --expected-commit "$FINAL_COMMIT" \
  --python "$PY"
```

Immediately before execution, enumerate `squeue -u "$USER"`. Repeat
`--allow-active-job-id ID` for every and only currently active base allocation
that is safe to coexist; the execute gate requires exact set equality. Then add
`--execute`. The launcher submits train and validation as separate jobs and
passes `--dependency=afterok:<train-job-id>` plus
`--kill-on-invalid-dep=yes` to the validation job. It never stops another job.

Train extraction writes 512 full and independent-history Wan latents in eight
content-hashed shards. Training opens only those train shards and raw train
actions. It refuses to run if a validation latent cache exists. Both the full
and matched angle-neutral probes use the same initialization, batches, targets,
optimizers, and 400-update endpoint. The comparator retains every
magnitude/support signal and maps each masked unit phasor to `(mask, 0)`, so
only spectral angle is removed.

Expected durable training outputs are:

```text
$STUDY/training/checkpoint-u000400.pt
$STUDY/training/report.json
$STUDY/wandb/
```

The checkpoint contains `full` and `angle_neutral` model states. Do not select
an earlier update from cal64. Preserve W&B local state even if upload
finalization is incomplete.

## 5. Verify the fixed checkpoint seal before val64 opens

The train Slurm stage runs the rendered `seal` operation only after both probe
fits return successfully. It must report fixed update 400,
`validation_clips_read=0`, and `protected_test_clips_read=0` in both checkpoint
and report. The resulting immutable boundary is:

```text
$STUDY/checkpoint-seal.json
```

Do not create or substitute a checkpoint after this point.

## 6. Extract sealed validation latents and evaluate

Only after the seal exists and the train allocation exits successfully does
Slurm release the dependent validation stage. Its `render --stage validation`
receipt first proves the registered seal using train-only records and only then
reopens and hashes validation sources. It executes validation extraction under
the same exact eight-B200 runtime. The extraction command includes `--seal`;
without it the tool fails before opening validation RGB. It then executes the
rendered `evaluate` command and writes exactly one val64 result:

```text
$STUDY/evaluation/val64.json
```

Check that the donor audit reports 32 adjacent-index, non-overlapping,
episode-disjoint symmetric pairs, zero self donors, and zero same-episode
donors. The sealed result contains both matched probe predictions and all four
targets: aligned, paired-shuffled, raw-no-action, and inverse. Do not inspect or
iterate on condition metrics before running the preregistered analysis.

## 7. Apply the one-shot gate

The validation stage executes the rendered `analyze` command only after sealed
evaluation returns successfully. It writes:

```text
$STUDY/analysis/bootstrap-gate.json
```

Report all eight point effects, simultaneous lower bounds, and fixed practical
thresholds, not only passing cells. The analysis averages within each of the 32
donor pairs and bootstraps those pair blocks; exactly the same 10,000 resamples
are used for all eight cells. A pass is probe feasibility only. A fail ends the
CSIP generator path for this representation and seed; no post-hoc checkpoint,
crop, phase mask, target dimension, control, threshold, or bootstrap change is
permitted.

No command in this runbook pushes a branch, registers a model artifact in an
external registry, or edits a generator. The submit helper is read-only by
default and submits jobs only with the explicit `--execute` flag.
