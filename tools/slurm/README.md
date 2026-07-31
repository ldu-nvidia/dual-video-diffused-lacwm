# Slurm continuation launcher

For the post-study V-JEPA validation-selected NFE-frontier workflow, use
`submit_vjepa2_frontier_workflow.sh`. It has direct gcp-nrt defaults for the
immutable v3 study, including partition `batch`, account
`coreai_chef_posttrain`, and QOS `normal`, and is dry-run-only until `--execute`
is supplied. These scheduler values remain CLI-overridable. The launcher queues
cache extraction, final-artifact validation, and the two validation arms. Its
selection gate submits lockbox scoring only after a confirmatory-eligible
candidate independently reproduces from raw validation rows. After lockbox
confirmation, a controller outcome gate fully reproduces the hashed raw
confirmation evidence: held-out quality failure writes an explicit negative
final report without running timing, while held-out quality success delegates
to the frozen scientific paired-timing and finalization entrypoint. A
study-local lock spans either publication or delegated timing. Negative reports
are atomically published without replacement; retry accepts only a byte- and
identity-equivalent complete report. Run the launcher from a clean evaluator
clone/worktree distinct from the immutable training checkout recorded in the
study.

The frontier launcher also supports the exact cache-only partial-submission
case with `--adopt-cache-job-id ID`. This is not a general resume flag. It
requires an empty pending-job log directory, no frontier outputs or submission
record, and an exact still-pending Slurm cache allocation whose complete
`SubmitLine`, resources, account/QOS, dependency, immutable inputs, and original
clean evaluator worktree reproduce. Recovery uses a separate controller
worktree but keeps all scientific evaluation on the cache producer's original
evaluator commit. All `batch` control submissions explicitly request one GPU.
See `docs/experiments/VJEPA2_NFE_FRONTIER_PROTOCOL.md` for the fail-closed
contract.

If that adopted cache completed but the original final-artifact gate failed
before validation, use `recover_vjepa2_frontier_workflow.sh` from a clean
controller commit. This is a second, even narrower recovery path for the
recorded `481556 -> 481577 -> {481578,481579} -> 481580` attempt. Its dry-run
preflight proves the original submission, terminal Slurm rows and complete
`SubmitLine` values, failed-gate stderr, full cache hashes/construction, and
absence of scientific outputs. `--execute` creates a unique recovery control
root with one immutable acceptance receipt per new job and submits only a
fresh controller gate, the two validation jobs from the frozen scientific
commit, and the conditional selection controller. It never submits or mutates
the completed cache.

These scripts run on one to 32 nodes with eight B200s per node. The latent default
is a physical batch of 4 per GPU with four-way gradient accumulation. Node count,
GPUs per node, world size, batching, and the minimum-memory B200 profile are bound
into the immutable run identity. The default schedule remains 60,000 updates with
2,000 warmup updates, logging every 50 updates, and checkpoint/validation/video
cadences of 1,000; each value can be explicitly overridden at submission.

`submit_8xb200.sh` is dry-run-only unless `--execute` is supplied. It can
therefore render and review the exact `sbatch` command on a machine without
Slurm installed.

```bash
tools/slurm/submit_8xb200.sh \
  --partition b200 --account MY_ACCOUNT --qos normal --constraint b200 \
  --nodes 2 --master-port 29400 \
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

The allocation requests one Slurm task per node. The guarded launcher resolves the
first allocated hostname as the c10d rendezvous host, then uses `srun` to start one
torchrun agent per node; each agent creates eight local GPU ranks. The NCCL probe,
real-data DDP smoke, and training use separate attempt-scoped rendezvous IDs and ports.
Every node runs the B200/runtime/data preflight and records local GPU and network
topology before collectives begin.

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
The batch entrypoint also refuses to start when `SLURM_RESTART_COUNT` already
exceeds the selected cap. The configurable cap has a hard ceiling of 100.

`RUN_ROOT/RUN_NAME/training_complete.json` is authoritative. Once present, the
batch job exits successfully without requeueing. Otherwise a restarted job
selects `--resume` automatically when `run_identity.json` exists. Resume slots
accept the original strict data report for up to 30 days while the launcher
continues to recompute and compare all active-file fingerprints on every slot.
Completion JSON is likewise accepted only when it reports all configured updates,
matches the immutable run identity, and references the existing live snapshot.

Exact resume is fixed-world-size: a run started on N nodes must resume on the same
N nodes, with one ordered RNG/loader state per global rank. Changing node count is a
new run, never an exact stochastic continuation. A schema-2
\`topology_migration_reset_rank_state\` handoff may preserve model, optimizer,
scheduler, scaler, iteration, and observation count only when the parent checkpoint
ACK and run identity are bound, the effective global batch is unchanged, and the
new run explicitly resets all rank-local loader/RNG state and uses a distinct run
and W&B identity.

## Unexpected failures

Automatic requeue is deliberately limited to an announced time limit with a validated
checkpoint ACK. A nonzero launcher, node, torchrun agent, or rank exit terminates the
allocation and is not automatically requeued. This prevents repeated hardware,
network, data, or numerical failures from consuming allocations or resuming ambiguous
state. Inspect the Slurm log, per-node preflight, NCCL/DDP logs, and last durable
snapshot before manually resubmitting the same fixed topology with `--resume`.

## Storage and scheduler requirements

- The run root and its `_slurm` control directory must be persistent and visible
  at the same path from every eligible B200 node.
- `/mnt/data1` and `/mnt/data2` are allowed by default. For a cluster filesystem,
  set `LACWM_ALLOWED_RUN_ROOTS` before submission to a colon-separated list of
  additional absolute canonical roots, for example:

  ```bash
  export LACWM_ALLOWED_RUN_ROOTS=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
  ```

  Empty entries, relative paths, `/`, symlink aliases, and paths containing
  non-canonical traversal are rejected. The submitter canonicalizes the selected
  run root, and the batch job plus every node preflight enforce the same captured
  policy before creating run or control state. Use `readlink -f PATH` to obtain the
  canonical spelling when the cluster exposes a friendlier symlink alias.
- The shared filesystem must provide working `flock(2)` semantics. A stable
  run-level lock prevents two nodes from modifying one checkpoint concurrently.
- Repo, Python environment, assets, data, reports, and manifests must be visible
  at the same absolute paths on every allocation.
- The site must allow the job owner to run `scontrol requeue`.
- Allocated hostnames must resolve from every node, and the selected rendezvous base
  port plus the next two ports must be available inside the allocation.
- `--dependency=singleton` provides an additional scheduler-level guard against
  concurrently running jobs with the same user and Slurm job name.

For an estimated five compute days, a 24-hour limit generally means five to six
allocations plus queue delays. Prefer the longest limit that still receives
reasonable backfill priority on the target cluster.

## V-JEPA paired-latency recovery

`submit_vjepa2_paired_latency_recovery.sh` is the incident-specific, fail-closed
recovery for validator-only failure `481133`. It preserves the immutable v3
study and writes every new receipt, log, timing artifact, and analysis file
under a fresh external `_paired_recoveries` root. The job is submitted held;
the launcher records its exact ID and tokenized `sbatch` command before
release, and the allocation validates its own accounting before timing.

Run without `--execute` for the read-only preflight. See
[`docs/experiments/VJEPA2_PAIRED_LATENCY_RECOVERY.md`](../../docs/experiments/VJEPA2_PAIRED_LATENCY_RECOVERY.md)
for the pinned contract and launch command.
