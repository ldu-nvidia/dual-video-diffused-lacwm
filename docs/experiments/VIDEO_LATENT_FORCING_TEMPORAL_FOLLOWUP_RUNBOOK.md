# Temporal-target DOE execution runbook

This runbook operationalizes stages D2 and D3 of
`VIDEO_LATENT_FORCING_TEMPORAL_FOLLOWUP_PROTOCOL.md`. It does not change the
preregistered science and cannot access the protected test split.

## Entry points

- `tools/slurm/temporal_followup_doe.py` performs a read-only preflight,
  renders the exact command graph by default, and executes only with the
  explicit `--execute` flag inside Slurm.
- `tools/slurm/temporal_followup_doe.sbatch` fixes one node, eight B200 GPUs,
  one launcher task, 160 CPUs, 1000 GiB RAM, a two-hour non-requeueable
  allocation, account `coreai_chef_posttrain`, and QOS `short`. It invokes the
  Python controller with `--execute`; it does not submit itself.

The controller uses `torch.distributed.run --standalone --nproc-per-node=8`.
Every arm therefore has global batch 256, per-rank optimizer batch 32,
microbatch 32, accumulation 1, four workers per rank, and seed 1234. It first
runs all five 200-update numerical calibrations and validates their receipts;
only then does it run the five fresh 5,000-update models. The arm order is
`ABS`, `ABS-T`, `DELTA`, `DELTA-T`, `DELTA-R`.

W&B is always enabled at `zijiandu/dual-video-diffusion-private`, with no
group. The operator must first verify that this project is private and pass
`--ack-private-wandb-project`. The child environment is allow-listed; an
authentication key may pass at runtime, but it is never written to the plan or
workflow state.

Slurm stdout, stderr, and the job working directory are fixed outside the Git
checkout at the Lustre `_slurm_logs` directory shown below. Create that
directory before submission. This is part of the source-integrity contract:
letting Slurm create `slurm-<job>.out` in the checkout would make the repository
dirty before the controller's fail-closed source check.

All run artifacts are required to use a fresh absolute path under `/lustre`;
the controller rejects repository-local, root-filesystem, `/mnt/data1`, and
`/mnt/data2` study roots for this DOE.

## Frozen cluster inputs

The current immutable paths are rooted at:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
```

Use these values:

```text
data root:
  .../data/production_v1/droid_lerobot
train manifest:
  .../data/video_latent_forcing_poc/droid_v1_6d35fbd/train.jsonl
validation manifest:
  .../data/video_latent_forcing_poc/droid_v1_6d35fbd/val.jsonl
semantic cache root:
  .../data/causal_vjepa2/causal-vjepa2-semantic-screen-seed1234-20260801-v2/semantic
normalization:
  .../artifacts/dual_video_diffusion/temporal_followup/temporal-followup-seed1234-20260807-v1/normalization/train_temporal_normalization.json
Slurm logs and working directory:
  .../artifacts/dual_video_diffusion/temporal_followup/_slurm_logs
```

The controller rehashes both manifests, both cache metadata files, both target
arrays, and the D1 normalization before creating an artifact. It also checks
the cache's PCA identity, shape, dtype, population, split, and protected-test
flags. The repository must be a clean checkout at the exact full commit passed
through `--expected-commit`.

## Dry run

Run the Python controller on a cluster login node with the pinned environment.
Omit `--execute` (or pass `--dry-run`) to perform the complete read-only
preflight and print every command. No study directory is created.

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
PYTHON="$BASE/envs/lacwm-b200-py310/bin/python"
REPO=/absolute/path/to/clean/dual-video-diffused-lacwm
STUDY="$BASE/artifacts/dual_video_diffusion/temporal_followup/temporal-doe-seed1234-20260807-v1"

"$PYTHON" "$REPO/tools/slurm/temporal_followup_doe.py" \
  --dry-run \
  --phase train \
  --expected-commit "$(git -C "$REPO" rev-parse HEAD)" \
  --study-root "$STUDY" \
  --data-root "$BASE/data/production_v1/droid_lerobot" \
  --semantic-cache-root "$BASE/data/causal_vjepa2/causal-vjepa2-semantic-screen-seed1234-20260801-v2/semantic" \
  --train-manifest "$BASE/data/video_latent_forcing_poc/droid_v1_6d35fbd/train.jsonl" \
  --validation-manifest "$BASE/data/video_latent_forcing_poc/droid_v1_6d35fbd/val.jsonl" \
  --normalization "$BASE/artifacts/dual_video_diffusion/temporal_followup/temporal-followup-seed1234-20260807-v1/normalization/train_temporal_normalization.json" \
  --ack-private-wandb-project
```

Review the rendered commands and preserve the output as launch evidence. This
command does not authorize or perform an `sbatch` submission.

## Authorized Slurm execution

Only after explicit authorization, export the pinned Python and submit the
source-controlled entrypoint with the same arguments. Prefer separate
allocations for training and evaluation because the latter runs all 63
arm/control/NFE cells per arm.

```bash
export TEMPORAL_DOE_PYTHON="$PYTHON"
mkdir -p "$BASE/artifacts/dual_video_diffusion/temporal_followup/_slurm_logs"
sbatch "$REPO/tools/slurm/temporal_followup_doe.sbatch" \
  --phase train \
  --expected-commit "$(git -C "$REPO" rev-parse HEAD)" \
  --study-root "$STUDY" \
  --data-root "$BASE/data/production_v1/droid_lerobot" \
  --semantic-cache-root "$BASE/data/causal_vjepa2/causal-vjepa2-semantic-screen-seed1234-20260801-v2/semantic" \
  --train-manifest "$BASE/data/video_latent_forcing_poc/droid_v1_6d35fbd/train.jsonl" \
  --validation-manifest "$BASE/data/video_latent_forcing_poc/droid_v1_6d35fbd/val.jsonl" \
  --normalization "$BASE/artifacts/dual_video_diffusion/temporal_followup/temporal-followup-seed1234-20260807-v1/normalization/train_temporal_normalization.json" \
  --ack-private-wandb-project
```

After `training_receipts.json` exists and validates, use the same command with
`--phase evaluate`. That phase refuses to run without the pre-metric
implementation registration and exact completed calibration/full receipts. It
evaluates all five arms on validation at NFE `{1,2,4,8,12,20,25}`, then runs
the preregistered paired analyzer. `--phase all` is supported, but is useful
only if one two-hour allocation has enough margin for both stages.

## Failure behavior

Every run directory is fresh and immutable. The controller stops at the first
nonzero command, nonfinite training receipt, wrong source/config/checkpoint,
input hash mismatch, unexpected W&B group, or protected-test evidence. It
never deletes, overwrites, resumes, requeues, or silently skips a partial run.
An operational retry must use a fresh study ID and retain the failed artifact;
the protocol's byte-identity and no-new-look rules still apply.
