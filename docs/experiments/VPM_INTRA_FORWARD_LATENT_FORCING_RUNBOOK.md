# One-call intra-forward latent-forcing runbook

This runbook executes the prospective protocol in
`VPM_INTRA_FORWARD_LATENT_FORCING_PROTOCOL.md`. It has no protected-test phase.
Use one fresh, clean source commit and one create-once Lustre study root; retain
failed roots rather than resuming or overwriting them.

## 1. Register immutable inputs

Set absolute cluster paths for `REPO`, `STUDY`, `PYTHON_BIN`, `WAN_DIR`,
`VIDEOX_HOME`, `WARMSTART`, `TRAIN_MANIFEST`, `TRAIN_METADATA`, `VAL_MANIFEST`,
and `VAL_METADATA`. Then run:

```bash
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
"$PYTHON_BIN" "$REPO/tools/intra_forward_forcing_screen.py" register \
  --expected-commit "$COMMIT" \
  --study-root "$STUDY" \
  --python "$PYTHON_BIN" \
  --wan-dir "$WAN_DIR" \
  --videox-home "$VIDEOX_HOME" \
  --warmstart "$WARMSTART" \
  --warmstart-sha256 f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21 \
  --train-manifest "$TRAIN_MANIFEST" \
  --train-metadata "$TRAIN_METADATA" \
  --validation-manifest "$VAL_MANIFEST" \
  --validation-metadata "$VAL_METADATA"
```

Registration rejects dirty source, overlapping train/validation episodes,
noncanonical inputs, any test split, and any pre-existing study root.

## 2. Train byte-matched arms

Create the static Slurm log parent before submission, then submit the two-arm
array. Its `%1` throttle is intentional: `MID-OFF` creates the update-zero byte
anchor and `MID-ON` must match it before its optimizer can run.

```bash
mkdir -p /lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/_slurm_logs
TRAIN_JOB="$(sbatch --parsable \
  --export=ALL,MID_SCREEN_REGISTRATION="$STUDY/protocol_registration.json",MID_SCREEN_REPO_ROOT="$REPO",MID_SCREEN_PYTHON="$PYTHON_BIN" \
  "$REPO/tools/slurm/intra_forward_forcing_screen.sbatch")"
```

Do not manually resume either arm. A failed array task leaves its root as
evidence and requires a freshly registered study.

## 3. Evaluate deployable cells

After both training tasks succeed, evaluate validation-only NFE 1/2/4 cells.
Each cell makes one artifact-audit generation and one synchronized timing
generation; each generation independently proves exactly NFE Wan, midpoint
head, and block-14 calls.

```bash
EVAL_JOB="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,MID_SCREEN_REGISTRATION="$STUDY/protocol_registration.json",MID_SCREEN_REPO_ROOT="$REPO",MID_SCREEN_PYTHON="$PYTHON_BIN" \
  "$REPO/tools/slurm/intra_forward_forcing_evaluate.sbatch")"
```

The only sources are aligned generated midpoint state, exact off, and global
sample-shuffled generated midpoint state. No clean future auxiliary enters any
sampler call.

## 4. Run the frozen analyzer

After both evaluation tasks complete:

```bash
"$PYTHON_BIN" "$REPO/tools/intra_forward_forcing_screen.py" analyze \
  --registration "$STUDY/protocol_registration.json" \
  --rows \
    "$STUDY/evaluation/mid_off/rows.jsonl" \
    "$STUDY/evaluation/mid_on/rows.jsonl" \
  --output "$STUDY/analysis.json"
```

NFE 1 is the sole selectable endpoint. NFE 2/4 are descriptive. Never inspect
or add a protected-test manifest to this workflow.
