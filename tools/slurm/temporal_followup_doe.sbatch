#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089
#SBATCH --chdir=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/temporal_followup/_slurm_logs
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/temporal_followup/_slurm_logs/temporal-followup-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/temporal_followup/_slurm_logs/temporal-followup-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint must run inside Slurm"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "the DOE uses one non-array allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "the DOE is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "the DOE requires one node"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || die "the DOE requires one launcher task"

REPO_ROOT="${TEMPORAL_DOE_REPO_ROOT:-}"
PYTHON_BIN="${TEMPORAL_DOE_PYTHON:-}"

[[ -n "$REPO_ROOT" ]] || \
  die "export TEMPORAL_DOE_REPO_ROOT to the clean exact-commit checkout"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT/.git" ]] || \
  die "TEMPORAL_DOE_REPO_ROOT must be an absolute Git worktree root"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
ORCHESTRATOR="$REPO_ROOT/tools/slurm/temporal_followup_doe.py"
[[ -n "$PYTHON_BIN" ]] || \
  die "export TEMPORAL_DOE_PYTHON to the pinned B200 Python executable"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || \
  die "TEMPORAL_DOE_PYTHON must be an absolute executable path"
[[ -f "$ORCHESTRATOR" && ! -L "$ORCHESTRATOR" ]] || \
  die "missing or symlinked orchestrator: $ORCHESTRATOR"

exec "$PYTHON_BIN" "$ORCHESTRATOR" --execute "$@"
