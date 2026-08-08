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

## 2. Submit the memory canary, byte-matched arms, and evaluation

Use the guarded wrapper. It submits four stages with explicit dependencies:

```text
8xB200 update-zero forward/backward memory canary
  -> MID-OFF task 0 (creates the initialization anchor)
  -> MID-ON task 1 (must match the anchor)
  -> parallel validation-only evaluation tasks 0 and 1
```

The two training arms are separate Slurm jobs. Do not replace this dependency
chain with a throttled array: a `%1` throttle does not guarantee that task 0
runs before task 1.

```bash
export MID_SCREEN_REGISTRATION="$STUDY/protocol_registration.json"
export MID_SCREEN_REPO_ROOT="$REPO"
export MID_SCREEN_PYTHON="$PYTHON_BIN"
"$REPO/tools/slurm/submit_intra_forward_forcing_screen.sh"
```

The wrapper prints all four job IDs. The memory canary performs the exact
production-shape BF16 forward and backward on each GPU using a deterministic
non-constant RGB fixture that remains valid under the model's per-view loss
mask, but never executes an optimizer update or reports a scientific metric.
Do not manually resume any stage. A failed stage leaves its root as evidence
and requires a freshly registered study.

## 3. Inspect deployable evaluation artifacts

The dependency-created evaluation jobs cover validation-only NFE 1/2/4 cells.
Before torchrun/NCCL, each job writes a create-once content preflight after
fully hashing the registration, warm start, final snapshot, and training
receipts. This prevents nonzero ranks from waiting at the first NCCL collective
while rank zero alone hashes multi-GB files.

For each two-clip cell they make one batch-2 artifact audit, one batch-2
profiled-path equivalence audit, and two batch-1 synchronized endpoint timings.
Every rollout independently proves exactly NFE Wan, midpoint-head, and block-14
calls. The two batch-2 paths must match byte-for-byte. Batch-1 versus batch-2
differences are recorded only as diagnostics because BF16 arithmetic can vary
with batch shape; the batch-2 artifact path alone supplies quality metrics.
Latency is descriptive at equal NFE only.

The only sources are aligned generated midpoint state, exact off, and global
future-bin-shuffled generated midpoint state with bins 0--1 preserved. No
clean future auxiliary enters any sampler call.

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
