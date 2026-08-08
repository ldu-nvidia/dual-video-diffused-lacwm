# Video Frequency-Forcing launch runbook

This runbook is prepared but has not been executed. Use a fresh clean checkout
of the final committed implementation. Do not launch while another eight-B200
study owns the intended allocation.

## Pinned cluster inputs

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
REPO=$BASE/src/dual-video-diffused-lacwm-frequency-forcing
PY=$BASE/envs/lacwm-b200-py310/bin/python
CACHE=$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690
WARM=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3/vpm_parameter_matched_video/snapshot.pt
PARENT=$BASE/artifacts/dual_video_diffusion/frequency_forcing
STUDY=$PARENT/frequency-forcing-seed1234-20260807-v1
REG=$STUDY/protocol_registration.json
```

Verified warm-start SHA-256:

```text
f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21
```

The runtime, Wan asset, VideoX-Fun checkout, 512-clip training manifest/cache,
64-clip validation manifest/cache, and checkpoint paths were observed on
`gcp-nrt-cs-001-vscode-01` on 2026-08-07. Registration revalidates them.

## 1. Prepare the exact source and log parent

Replace `FINAL_COMMIT` only after the implementation and protocol are reviewed
and committed. The registration command refuses a dirty checkout.

```bash
FINAL_COMMIT=<40-character-commit>
mkdir -p "$PARENT/_slurm_logs"
git clone git@github.com:ldu-nvidia/dual-video-diffused-lacwm.git "$REPO"
git -C "$REPO" checkout --detach "$FINAL_COMMIT"
git -C "$REPO" status --short
```

The final command must emit nothing. Reuse an existing exact clean checkout
instead of cloning if one already exists; do not overwrite a working tree.

## 2. Freeze registration before metrics

```bash
"$PY" "$REPO/tools/frequency_forcing_screen.py" register \
  --expected-commit "$FINAL_COMMIT" \
  --study-root "$STUDY" \
  --python "$PY" \
  --wan-dir "$BASE/wan_fun_1.3b_control" \
  --videox-home "$BASE/VideoX-Fun-1d6d9c3" \
  --warmstart "$WARM" \
  --warmstart-sha256 f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21 \
  --train-manifest "$CACHE/manifests/train.jsonl" \
  --train-metadata "$CACHE/caches/train/metadata.json" \
  --validation-manifest "$CACHE/manifests/val.jsonl" \
  --validation-metadata "$CACHE/caches/val/metadata.json" \
  > "$PARENT/registration.stdout.json"

"$PY" "$REPO/tools/frequency_forcing_screen.py" verify \
  --registration "$REG"
```

Registration has no test arguments and records `protected_test.allowed=false`.
Do not add or substitute a test path.

## 3. Submit the four controlled training arms

The scripts request the `coreai_chef_posttrain` account, `short` QOS, two-hour
limit, eight B200s per arm, excluded nodes `pool0-0081,pool0-0089`, no requeue,
and an array concurrency of two.

```bash
export FREQ_SCREEN_REGISTRATION="$REG"
export FREQ_SCREEN_REPO_ROOT="$REPO"
export FREQ_SCREEN_PYTHON="$PY"

TRAIN_JOB=$(sbatch --parsable "$REPO/tools/slurm/frequency_forcing_screen.sbatch")
echo "$TRAIN_JOB"
```

Array mapping:

```text
0 FPM    fresh six-channel parameter-matched video-only continuation
1 FAUX   aligned auxiliary multitask loss, no video fusion
2 FSYNC  aligned joint low-frequency fusion
3 FLEAD  identical joint model, auxiliary clock leads by logit 1
```

Each arm trains for 200 updates. Historical telemetry is about `0.52 s/update`
after warmup; model construction, four small validations, one NFE-1
visualization, W&B, checkpoint hashing, and final snapshot dominate. Budget
roughly 15–25 minutes per arm. With concurrency two, expected allocated wall
time is roughly 30–50 minutes plus queue delay. Treat this as an estimate, not
a scheduler guarantee.

W&B finalization is capped at 120 seconds without raising after the durable
snapshot and completion receipt exist. If online upload is incomplete, preserve
the arm's `wandb`, `wandb-data`, `wandb-cache`, and `wandb-config` directories;
they are the source for a later `wandb sync` and must not be deleted.

## 4. Submit validation-only mechanism evaluation

```bash
EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:$TRAIN_JOB" \
  "$REPO/tools/slurm/frequency_forcing_evaluate.sbatch")
echo "$EVAL_JOB"
```

Each evaluation arm scores all 64 validation clips at NFE `1,2,4,8` under
autonomous, fusion-off, generated-state-shuffled, oracle-matched, and
oracle-shuffled sources. Every cell audits the sampler counter, an independent
Wan hook, zero transform calls inside sampling, zero teacher calls, and clean
target availability. Expect about 10–25 minutes per arm; the one-hour request
leaves substantial headroom.

Monitoring is read-only:

```bash
squeue -j "$TRAIN_JOB,$EVAL_JOB" -o '%.18i %.9T %.10M %.6D %R'
sacct -j "$TRAIN_JOB,$EVAL_JOB" --format=JobID,State,Elapsed,ExitCode,NodeList
```

Do not stop, requeue, or modify an active arm. Failed runs remain evidence and
must receive a new study ID after diagnosis; they are not overwritten.

## 5. Analyze only after all four evaluation receipts exist

The analyzer itself verifies every sibling `complete.json`, binds its declared
study and arm identities to every row, checks the frozen grid, and recomputes
the SHA-256 of `rows.jsonl`; the shell existence check is only an early error.

```bash
for slug in \
  fpm_video_only \
  faux_multitask_only \
  fsync_joint_aligned \
  flead_joint_leading; do
  test -f "$STUDY/evaluation/$slug/complete.json"
done

"$PY" "$REPO/tools/frequency_forcing_screen.py" analyze \
  --registration "$REG" \
  --rows \
    "$STUDY/evaluation/fpm_video_only/rows.jsonl" \
    "$STUDY/evaluation/faux_multitask_only/rows.jsonl" \
    "$STUDY/evaluation/fsync_joint_aligned/rows.jsonl" \
    "$STUDY/evaluation/flead_joint_leading/rows.jsonl" \
  --output "$STUDY/analysis.json"
```

No protected test follows this screen. A pass only authorizes a separately
preregistered, multi-seed DROID confirmation.
