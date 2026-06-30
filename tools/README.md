# Guarded training tools

These tools are deliberately separate from `setup_training.sh`: they never download
assets or data, stop processes, choose a GPU implicitly, or launch in the background.
Every launch is fail-closed and remains attached to the terminal.

## 1. Inspect a host without launching

```bash
python3 tools/training_preflight.py \
  --profile assets \
  --python /path/to/lacwm-env/bin/python \
  --wan-dir /mnt/data1/.../wan_fun_1.3b_control \
  --videox-home /mnt/data1/.../VideoX-Fun \
  --data-root /mnt/data1/.../lacwm_data \
  --run-root /mnt/data1/.../lacwm_runs
```

The preflight checks repository state, the selected Python environment, model files,
the first entries and counts of all four datasets, writable/free output storage, host
RAM/CPUs, GPU inventory, importability of the actual Wan/VideoX modules, and active jobs.
Smoke mode requires its selected GPU to be idle and warns about jobs on other GPUs. Full
mode selects all eight GPUs, so any GPU job blocks it. The preflight also requires the
`colorlog` import used by the active Hydra logging configuration; `debugpy` is not needed
unless training is explicitly launched with `debug=true`.

## 2. Validate gradients on one idle GPU

First run without `--execute`; this performs preflight and prints the exact command.
W&B must be acknowledged as disabled because this validator never contacts W&B.

```bash
tools/run_gradient_smoke.sh \
  --gpu 0 \
  --variant latent \
  --data-mode real \
  --python /path/to/lacwm-env/bin/python \
  --wan-dir /mnt/data1/.../wan_fun_1.3b_control \
  --videox-home /mnt/data1/.../VideoX-Fun \
  --data-root /mnt/data1/.../lacwm_data \
  --run-root /mnt/data1/.../lacwm_runs \
  --wandb-mode disabled

# Add only after reviewing the preflight and command:
#   --execute
```

The executed validator covers DROID, EgoDex, AgiBot, and ABC individually. It performs
four forward/backward/AdamW updates without saving a checkpoint and verifies finite
losses and gradients in LoRA, ActionToControl, temporal action pooling, morphology
conditioning, and either the inverse/action-decoder path or explicit action encoder.
Its timestamped `gradient_report.json` is required by the full launcher.

To validate model loading and the complete gradient graph before datasets are present,
use `--data-mode synthetic` and omit `--data-root`. Synthetic mode still loads the real
Wan DiT/VAE weights and creates deterministic `[13,3,180,960]` RGB plus
`[13,5,157]` action tensors for morphology IDs 0, 2, 6, and 9. Its report is clearly
tagged `lacwm_gradient_smoke_synthetic` / `data_mode=synthetic` and **cannot** satisfy
the full launcher. A passing real-data smoke remains mandatory for an official run.

## 3. Strictly validate the full dataset

Run the deep, read-only validator with the training environment and keep its JSON under
the data volume. The launcher rejects files-only reports and reports older than 24 hours.
The report binds the exact validator/commit/options, canonical manifests, and a selected-
file stat fingerprint; any manifest or payload size/mtime change requires revalidation.

```bash
mkdir -p /mnt/data1/.../lacwm_runs
/path/to/lacwm-env/bin/python tools/validate_training_data.py \
  --data-root /mnt/data1/.../lacwm_data \
  --workers 16 \
  --json > /mnt/data1/.../lacwm_runs/data_validation.json
```

## 4. Launch the full 8xB200 run

The full wrapper additionally requires eight B200 GPUs, at least 78000 MiB per GPU by
default,
the intended dataset counts, a clean worktree, and a passing smoke report from the same
commit, model variant, and canonical paths. W&B mode is always explicit.

```bash
tools/launch_8xb200.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --variant latent \
  --python /path/to/lacwm-env/bin/python \
  --wan-dir /mnt/data1/.../wan_fun_1.3b_control \
  --videox-home /mnt/data1/.../VideoX-Fun \
  --data-root /mnt/data1/.../lacwm_data \
  --run-root /mnt/data1/.../lacwm_runs \
  --run-name lacwm_latent_v1 \
  --smoke-report /mnt/data1/.../gradient_report.json \
  --data-validation-report /mnt/data1/.../data_validation.json \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --min-gpu-memory-mib 78000 \
  --wandb-mode online \
  --wandb-project lacwm \
  --wandb-entity YOUR_ENTITY

# Add only after reviewing the preflight and command:
#   --execute
```

The stable default run directory is `RUN_ROOT/RUN_NAME`. If training is interrupted,
repeat the same command with `--resume`; the wrapper verifies run identity and the
Trainer automatically loads `snapshot.pt`. `--run-dir` selects another stable directory
beneath the run root. An existing checkpoint is never overwritten as a fresh run.
Immediately before the long run, the wrapper repeats exact verification under its lock,
runs an eight-rank NCCL collective probe, then performs three real-data accumulated DDP
optimizer/scheduler updates at the requested physical per-GPU microbatch and exact
production loader settings. This exercises DDP reducer reuse, delayed
conditioning gradients, all morphologies, and cross-rank parameter synchronization.
These gate logs are retained with provenance.

Use `--wandb-mode offline --wandb-project lacwm` for local W&B files, or
`--wandb-mode disabled` to make the no-W&B choice explicit. All Hydra outputs, console
logs, W&B files, preflight results, package inventory, Git state, GPU topology, accepted
smoke report, and the shell-escaped command are stored beneath the selected `/mnt/data1`
or `/mnt/data2` run root.

Both wrappers take a fixed per-host/account advisory lock (independent of the selected
run root) and repeat preflight immediately before the command. Smoke can use an idle GPU
while an unrelated job runs on another GPU. The lock coordinates these wrappers under
one account; the cluster scheduler remains authoritative across users and other launch
systems.

## 5. Resilient Slurm time slices

Use `tools/slurm/submit_8xb200.sh` when the cluster wall-time limit is shorter than
the full run. It submits one 8xB200 allocation that re-enters the queue under the
same Slurm job ID after each acknowledged time-limit checkpoint. The production
batching defaults, 60,000 updates, optimizer, and normal 1,000-iteration checkpoint
cadence come from the checked-in experiment. Explicit batching and memory-profile
arguments are identity-bound and must remain identical across resumed allocations.

Slurm sends `B:USR1` to the batch shell before the slot ends. The shell publishes a
per-attempt request file; all training ranks finish a safe iteration boundary, save
the exact distributed state atomically, and publish an ACK. The job requeues only
after that ACK. Application errors, failed writes, and missing ACKs stop instead of
creating an unsafe retry loop. A shared-filesystem lock also prevents two nodes from
writing one run directory concurrently.

The submitter is a dry run unless `--execute` is supplied. See
[`tools/slurm/README.md`](slurm/README.md) for the full command and cluster storage
requirements. Checkpoints and the `_slurm` control directory must be visible at the
same absolute path from every eligible node; do not place them in node-local scratch.
