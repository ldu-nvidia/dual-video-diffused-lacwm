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
RECOVERY_ARGS=()
while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --scientific-repo-root) SCIENTIFIC_REPO_ROOT="${2:?}"; shift 2 ;;
    --scientific-evaluator-commit) SCIENTIFIC_EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --recovery-prior-submission)
      RECOVERY_ARGS+=(--prior-submission "${2:?}")
      shift 2
      ;;
    --recovery-cache-job-id)
      RECOVERY_ARGS+=(--cache-job-id "${2:?}")
      shift 2
      ;;
    --recovery-failed-gate-job-id)
      RECOVERY_ARGS+=(--failed-gate-job-id "${2:?}")
      shift 2
      ;;
    --recovery-cancelled-vpm-job-id)
      RECOVERY_ARGS+=(--cancelled-vpm-job-id "${2:?}")
      shift 2
      ;;
    --recovery-cancelled-j1-job-id)
      RECOVERY_ARGS+=(--cancelled-j1-job-id "${2:?}")
      shift 2
      ;;
    --recovery-cancelled-selection-job-id)
      RECOVERY_ARGS+=(--cancelled-selection-job-id "${2:?}")
      shift 2
      ;;
    --recovery-final-job-id)
      RECOVERY_ARGS+=(--final-job-id "${2:?}")
      shift 2
      ;;
    --recovery-partition)
      RECOVERY_ARGS+=(--partition "${2:?}")
      shift 2
      ;;
    --recovery-account)
      RECOVERY_ARGS+=(--account "${2:?}")
      shift 2
      ;;
    --recovery-qos)
      RECOVERY_ARGS+=(--qos "${2:?}")
      shift 2
      ;;
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

"$PYTHON_BIN" "$REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
  check-final \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT"

if (("${#RECOVERY_ARGS[@]}")); then
  "$PYTHON_BIN" "$REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
    validate-completed-recovery \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --controller-repo-root "$REPO_ROOT" \
    --controller-commit "$EVALUATOR_COMMIT" \
    --scientific-repo-root "$SCIENTIFIC_REPO_ROOT" \
    --scientific-commit "$SCIENTIFIC_EVALUATOR_COMMIT" \
    "${RECOVERY_ARGS[@]}"
fi
