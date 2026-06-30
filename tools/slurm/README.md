# Slurm continuation launcher

These scripts run the checked-in 60,000-step experiment across a sequence of
single-node 8xB200 allocations. The latent default is a physical batch of 4 per GPU
with four-way gradient accumulation (effective global batch 128) and a 78000 MiB
minimum-memory B200 profile. Those values may be supplied explicitly, are propagated
through every allocation, and are bound into the immutable run identity. Optimizer,
checkpoint cadence, model configuration, and total steps remain checked-in settings.

`submit_8xb200.sh` is dry-run-only unless `--execute` is supplied. It can
therefore render and review the exact `sbatch` command on a machine without
Slurm installed.

```bash
tools/slurm/submit_8xb200.sh \
  --partition b200 --account MY_ACCOUNT --qos normal --constraint b200 \
  --time 24:00:00 --cpus 160 --mem 1000G --signal-seconds 1200 \
  --max-requeues 12 \
  --batch-size 4 --gradient-accumulation-steps 4 \
  --min-gpu-memory-mib 78000 \
  --python /mnt/data1/shared/lacwm/env/bin/python \
  --wan-dir /mnt/data1/shared/lacwm/wan_fun_1.3b_control \
  --videox-home /mnt/data1/shared/lacwm/VideoX-Fun-1d6d9c3 \
  --data-root /mnt/data1/shared/lacwm/data \
  --run-root /mnt/data1/shared/lacwm/runs \
  --run-name lacwm_latent_v1 \
  --smoke-report /mnt/data1/shared/lacwm/reports/gradient_report.json \
  --data-validation-report /mnt/data1/shared/lacwm/reports/data_validation.json \
  --wandb-mode online --wandb-project lacwm --wandb-entity MY_ENTITY

# Add --execute only after reviewing the command.
```

The batch job requests one Slurm task. That task invokes the existing guarded
launcher, which creates eight local ranks with `torchrun`. Do not change this to
eight Slurm tasks.

## Time-limit protocol

The submitter requests `--signal=B:USR1@1200` by default. Near the allocation's
time limit, the batch shell atomically publishes a unique request and exports:

- `LACWM_CHECKPOINT_REQUEST_FILE`
- `LACWM_CHECKPOINT_ACK_FILE`

The Trainer creates the ACK only after every rank has participated in a durable
checkpoint, or after confirming that a request received before update zero is a
safe zero-state retry. The batch job continues waiting after the signal and
calls `scontrol requeue` only when that ACK validates. A missing or malformed
ACK is a failed attempt, never an unsafe automatic retry.

These mechanics use Slurm's documented batch-shell signal and restart behavior:
[`sbatch --signal/--requeue`](https://slurm.schedmd.com/sbatch.html) and
[`scontrol requeue`](https://slurm.schedmd.com/scontrol.html).

Before requeueing, the batch job validates the ACK's status, Slurm attempt ID,
run identity, iteration bounds, and snapshot path. Twelve requeues are allowed
by default (thirteen total allocations); reaching the cap stops for inspection.

`RUN_ROOT/RUN_NAME/training_complete.json` is authoritative. Once present, the
batch job exits successfully without requeueing. Otherwise a restarted job
selects `--resume` automatically when `run_identity.json` exists. Resume slots
accept the original strict data report for up to 30 days while the launcher
continues to recompute and compare all active-file fingerprints on every slot.
Completion JSON is likewise accepted only when it reports all 60,000 updates,
matches the immutable run identity, and references the existing live snapshot.

## Storage and scheduler requirements

- The run root and its `_slurm` control directory must be persistent and visible
  at the same path from every eligible B200 node.
- The shared filesystem must provide working `flock(2)` semantics. A stable
  run-level lock prevents two nodes from modifying one checkpoint concurrently.
- Repo, Python environment, assets, data, reports, and manifests must be visible
  at the same absolute paths on every allocation.
- The site must allow the job owner to run `scontrol requeue`.
- `--dependency=singleton` provides an additional scheduler-level guard against
  concurrently running jobs with the same user and Slurm job name.

For an estimated five compute days, a 24-hour limit generally means five to six
allocations plus queue delays. Prefer the longest limit that still receives
reasonable backfill priority on the target cluster.
