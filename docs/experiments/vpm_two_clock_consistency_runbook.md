# VPM two-clock consistency launch runbook

This runbook is prepared but has not been executed. Launch only from the final
reviewed commit and only when two independent eight-B200 jobs fit the active
allocation. Never reuse or overwrite a failed study root.

## Pinned inputs

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
REPO=$BASE/src/dual-video-diffused-lacwm-two-clock-consistency
HIST=$BASE/src/vjepa2-latent-forcing/dual-video-diffused-lacwm
PY=$BASE/envs/lacwm-b200-py310/bin/python
CACHE=$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690
PARENT_STUDY=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3
PARENT=$BASE/artifacts/dual_video_diffusion/two_clock_consistency
STUDY=$PARENT/two-clock-consistency-seed1234-20260808-v1
REG=$STUDY/registration.json
```

The parent VPM snapshot must hash to:

```text
f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21
```

## 1. Prepare exact clean source

Replace `FINAL_COMMIT` only after review and commit. Do not overwrite an
existing worktree.

```bash
FINAL_COMMIT=<40-character-commit>
mkdir -p "$PARENT/_slurm_logs"
git clone git@github.com:ldu-nvidia/dual-video-diffused-lacwm.git "$REPO"
git -C "$REPO" checkout --detach "$FINAL_COMMIT"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
test "$(git -C "$HIST" rev-parse HEAD)" = 9cf8e6922f35a5d6645e3128545953723bf54da2
test -z "$(git -C "$HIST" status --porcelain --untracked-files=all)"
```

Before any submission, check active jobs and do not stop them:

```bash
squeue -u "$USER" -o '%.18i %.9T %.10M %.6D %R'
```

## 2. Register before training or metrics

Registration rehashes the clean implementation and historical checkout, the
4.25 GB parent snapshot, train/validation RGB and actions, manifests, runtime,
Wan assets, protocol, and private W&B project.

```bash
"$PY" "$REPO/tools/two_clock_consistency_evaluate.py" register \
  --tool-repo "$REPO" \
  --historical-repo "$HIST" \
  --expected-commit "$FINAL_COMMIT" \
  --study-root "$PARENT_STUDY" \
  --train-manifest "$CACHE/manifests/train.jsonl" \
  --train-cache-metadata "$CACHE/caches/train/metadata.json" \
  --python "$PY" \
  --output-root "$STUDY" \
  > "$PARENT/registration-two-clock-consistency.stdout.json"

"$PY" "$REPO/tools/slurm/two_clock_consistency_workflow.py" plan \
  --registration "$REG" \
  > "$PARENT/two-clock-consistency-dry-run.json"
```

Inspect the dry run. It must show exactly `TC-CONT` and `TC-CONS`, 200 updates,
validation NFE `1,2,4`, personal private W&B, and no test path.
Each rendered W&B ID must equal that arm's 64-character run identity and use
`resume=never`; the human-readable run name remains the fixed arm name.
The arms match parameters, data, optimizer, and two Wan calls per update. The
candidate's small elementwise consistency-gradient overhead means the screen is
not an exact wall-clock or total-training-FLOP comparison.

## 3. Submit the matched arms

Each non-requeueable job trains one 200-update arm and immediately evaluates
all 64 validation clips. Source cleanliness and exact commit are checked again
inside Slurm. Parent/model schema and all input hashes are checked before the
first optimizer update.

```bash
export TWO_CLOCK_CONSISTENCY_REPO_ROOT="$REPO"
export TWO_CLOCK_CONSISTENCY_PYTHON="$PY"

CONTROL_JOB=$(sbatch --parsable \
  "$REPO/tools/slurm/two_clock_consistency.sbatch" \
  --mode arm --arm TC-CONT --registration "$REG" \
  --repo-root "$REPO" --python "$PY" --expected-commit "$FINAL_COMMIT")

CANDIDATE_JOB=$(sbatch --parsable \
  "$REPO/tools/slurm/two_clock_consistency.sbatch" \
  --mode arm --arm TC-CONS --registration "$REG" \
  --repo-root "$REPO" --python "$PY" --expected-commit "$FINAL_COMMIT")

echo "$CONTROL_JOB $CANDIDATE_JOB"
```

If only eight B200s are available, add
`--dependency="afterok:$CONTROL_JOB"` to the candidate submission. Sequential
execution does not change the statistical pairing.

Monitoring is read-only:

```bash
squeue -j "$CONTROL_JOB,$CANDIDATE_JOB" -o '%.18i %.9T %.10M %.6D %R'
sacct -j "$CONTROL_JOB,$CANDIDATE_JOB" \
  --format=JobID,State,Elapsed,ExitCode,NodeList
```

Do not stop, requeue, edit, or resume an arm. A failure remains evidence and
requires diagnosis plus a new study ID.

## 4. Analyze after both jobs succeed

```bash
test -s "$STUDY/evaluation/tc-cont/inventory.json"
test -s "$STUDY/evaluation/tc-cons/inventory.json"

"$PY" "$REPO/tools/slurm/two_clock_consistency_workflow.py" analyze \
  --registration "$REG"

test -s "$STUDY/analysis/analysis.json"
```

The analyzer first requires exact equality of all 200 paired data, epsilon,
clock, noisy-state, and RNG hashes. It then checks paired evaluation hashes,
actual NFE calls, the fixed bootstrap family, and the preregistered gate.

No protected test follows this screen. A pass authorizes a separately
preregistered multi-seed confirmation; a failure does not authorize weight or
clock-band tuning inside this root.
