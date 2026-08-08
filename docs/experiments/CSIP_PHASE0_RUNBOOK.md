# CSIP Phase-0 execution runbook

This runbook is prepared but has not been executed. It contains no Slurm
submission helper on purpose: commands are rendered for independent review,
then the reviewer may wrap them in the cluster's approved scheduler template.
Do not launch while another eight-B200 study owns the intended allocation.

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

## 4. Render, review, then execute the train-only stages

```bash
"$PY" "$REPO/tools/csip_workflow.py" render --registration "$REG" \
  > "$STUDY/rendered-commands.review.json"
```

Because study files are immutable outputs, writing the review capture inside
the study is optional operational evidence and is not consumed by any tool.
The rendered train extraction uses eight ranks. Submit it through an approved
one-node/eight-B200 allocation; do not run it on a login node.

Execute only `extract_train`, then `train`. Train extraction writes 512 full
and independent-history Wan latents in eight content-hashed shards. Training
opens only those train shards and raw train actions. It refuses to run if a
validation latent cache exists.

Expected durable training outputs are:

```text
$STUDY/training/checkpoint-u000400.pt
$STUDY/training/report.json
$STUDY/wandb/
```

Do not select an earlier update from cal64. Preserve W&B local state even if
upload finalization is incomplete.

## 5. Seal the fixed checkpoint before opening val64

Run the rendered `seal` command. It must report fixed update 400,
`validation_clips_read=0`, and `protected_test_clips_read=0` in both checkpoint
and report. The resulting immutable boundary is:

```text
$STUDY/checkpoint-seal.json
```

Do not create or substitute a checkpoint after this point.

## 6. Extract sealed validation latents and evaluate

Only after the seal exists, execute the rendered validation extraction under
the same eight-B200 runtime. The command includes `--seal`; without it the tool
fails before opening validation RGB. Then execute the rendered `evaluate`
command. It writes exactly one val64 result:

```text
$STUDY/evaluation/val64.json
```

Check that the donor audit reports zero self donors and zero same-episode
donors. Do not inspect or iterate on condition metrics before running the
preregistered analysis.

## 7. Apply the one-shot gate

Execute the rendered `analyze` command. It writes:

```text
$STUDY/analysis/bootstrap-gate.json
```

Report all six point effects and simultaneous lower bounds, not only passing
cells. A pass is probe feasibility only. A fail ends the CSIP generator path
for this representation and seed; no post-hoc checkpoint, crop, phase mask,
target dimension, control, or bootstrap change is permitted.

No command in this runbook pushes a branch, registers a model artifact in an
external registry, edits a generator, or submits a job automatically.
