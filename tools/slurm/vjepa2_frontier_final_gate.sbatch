#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

REPO_ROOT=""
STUDY_ROOT=""
TRAINING_COMMIT=""
EVALUATOR_COMMIT=""
SCIENTIFIC_REPO_ROOT=""
SCIENTIFIC_EVALUATOR_COMMIT=""
PYTHON_BIN=""
while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --scientific-repo-root) SCIENTIFIC_REPO_ROOT="${2:?}"; shift 2 ;;
    --scientific-evaluator-commit) SCIENTIFIC_EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -z "$SCIENTIFIC_REPO_ROOT" ]]; then
  SCIENTIFIC_REPO_ROOT="$REPO_ROOT"
fi
if [[ -z "$SCIENTIFIC_EVALUATOR_COMMIT" ]]; then
  SCIENTIFIC_EVALUATOR_COMMIT="$EVALUATOR_COMMIT"
fi
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "final gate requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "final gate is non-requeueable"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ "$(git -C "$SCIENTIFIC_REPO_ROOT" rev-parse HEAD)" == \
  "$SCIENTIFIC_EVALUATOR_COMMIT" ]] || \
  die "scientific evaluator worktree HEAD changed"
[[ -z "$(git -C "$SCIENTIFIC_REPO_ROOT" status \
  --porcelain --untracked-files=all)" ]] || \
  die "scientific evaluator worktree is dirty"

"$PYTHON_BIN" "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
  check-final \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT"
