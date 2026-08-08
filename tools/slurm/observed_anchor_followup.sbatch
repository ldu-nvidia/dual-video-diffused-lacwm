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
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/temporal_followup/_slurm_logs/observed-anchor-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/temporal_followup/_slurm_logs/observed-anchor-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint must run inside Slurm"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "observed-anchor phases are not arrays"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "requeue is forbidden"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "exactly one node is required"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || die "exactly one launcher task is required"

REPO_ROOT="${OBSERVED_ANCHOR_REPO_ROOT:-}"
PYTHON_BIN="${OBSERVED_ANCHOR_PYTHON:-}"
[[ -n "$REPO_ROOT" && "$REPO_ROOT" == /* ]] || \
  die "export OBSERVED_ANCHOR_REPO_ROOT to the exact clean checkout"
[[ "$(git -C "$REPO_ROOT" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || \
  die "OBSERVED_ANCHOR_REPO_ROOT must be an absolute Git worktree root"
[[ -n "$PYTHON_BIN" && "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || \
  die "export OBSERVED_ANCHOR_PYTHON to the pinned B200 Python executable"
REPO_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-toplevel)"
WORKFLOW="$REPO_ROOT/tools/slurm/observed_anchor_followup.py"
[[ -f "$WORKFLOW" && ! -L "$WORKFLOW" ]] || die "missing workflow: $WORKFLOW"

exec "$PYTHON_BIN" "$WORKFLOW" --execute "$@"
