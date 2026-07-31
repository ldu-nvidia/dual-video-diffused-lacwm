#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
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
CONTROLLER_COMMIT=""
SCIENTIFIC_REPO_ROOT=""
SCIENTIFIC_EVALUATOR_COMMIT=""
STUDY_ROOT=""
TRAINING_COMMIT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --controller-commit) CONTROLLER_COMMIT="${2:?}"; shift 2 ;;
    --scientific-repo-root) SCIENTIFIC_REPO_ROOT="${2:?}"; shift 2 ;;
    --scientific-evaluator-commit)
      SCIENTIFIC_EVALUATOR_COMMIT="${2:?}"
      shift 2
      ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "conditional timing gate requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "conditional timing gate is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "conditional timing gate requires one node"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
for repo in "$REPO_ROOT" "$SCIENTIFIC_REPO_ROOT"; do
  [[ -d "$repo" && ! -L "$repo" ]] || die "repository is invalid: $repo"
  [[ "$(cd "$repo" && pwd -P)" == "$repo" ]] || \
    die "repository is not canonical: $repo"
  [[ "$(git -C "$repo" rev-parse --show-toplevel)" == "$repo" ]] || \
    die "repository is not a worktree root: $repo"
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || \
    die "repository must be clean: $repo"
done
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$CONTROLLER_COMMIT" ]] || \
  die "controller worktree HEAD changed"
[[ "$(git -C "$SCIENTIFIC_REPO_ROOT" rev-parse HEAD)" == \
  "$SCIENTIFIC_EVALUATOR_COMMIT" ]] || \
  die "scientific evaluator worktree HEAD changed"
git -C "$REPO_ROOT" merge-base --is-ancestor \
  "$SCIENTIFIC_EVALUATOR_COMMIT" "$CONTROLLER_COMMIT" || \
  die "controller does not descend from the scientific evaluator"
git -C "$REPO_ROOT" diff --quiet \
  "$SCIENTIFIC_EVALUATOR_COMMIT" "$CONTROLLER_COMMIT" -- \
  tools/vjepa2_nfe_frontier.py || \
  die "controller changed the frozen scientific frontier implementation"
command -v flock >/dev/null 2>&1 || die "flock is unavailable"

SELECTION="$STUDY_ROOT/frontier_selection.json"
CONFIRMATION="$STUDY_ROOT/frontier_lockbox_confirmation.json"
FINAL_REPORT="$STUDY_ROOT/frontier_final_report.json"
LATENCY_PARENT="$STUDY_ROOT/frontier_latency"
OUTCOME_LOCK="$STUDY_ROOT/.vjepa2_frontier_outcome.lock"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || \
  die "study root is not a canonical directory"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "study root path is not canonical"
[[ -s "$SELECTION" && ! -L "$SELECTION" ]] || \
  die "frontier selection is unavailable"
[[ -s "$CONFIRMATION" && ! -L "$CONFIRMATION" ]] || \
  die "lockbox confirmation is unavailable"
[[ ! -L "$FINAL_REPORT" ]] || \
  die "frontier final report must not be a symlink"
[[ ! -e "$LATENCY_PARENT" && ! -L "$LATENCY_PARENT" ]] || \
  die "fresh frontier timing parent already exists: $LATENCY_PARENT"
[[ ! -L "$OUTCOME_LOCK" ]] || die "outcome lock must not be a symlink"
if [[ -e "$OUTCOME_LOCK" ]]; then
  [[ -f "$OUTCOME_LOCK" ]] || die "outcome lock is not a regular file"
fi
exec 9>>"$OUTCOME_LOCK"
[[ -f "$OUTCOME_LOCK" && ! -L "$OUTCOME_LOCK" ]] || \
  die "outcome lock is not a regular file"
flock -n 9 || die "another frontier outcome handler holds the study lock"

unset PYTHONPATH
set +e
"$PYTHON_BIN" "$REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
  finalize-quality-failure \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT" \
  --selection "$SELECTION" \
  --confirmation "$CONFIRMATION" \
  --output "$FINAL_REPORT" \
  --controller-commit "$CONTROLLER_COMMIT" \
  --scientific-commit "$SCIENTIFIC_EVALUATOR_COMMIT"
QUALITY_STATUS=$?
set -e

case "$QUALITY_STATUS" in
  0)
    [[ -s "$FINAL_REPORT" && ! -L "$FINAL_REPORT" ]] || \
      die "negative final report was not produced"
    [[ ! -e "$LATENCY_PARENT" ]] || \
      die "paired timing artifact exists after held-out quality failure"
    echo "Held-out quality gate failed; finalized without paired timing: $FINAL_REPORT"
    exit 0
    ;;
  4)
    [[ ! -e "$FINAL_REPORT" && ! -e "$LATENCY_PARENT" ]] || \
      die "quality-pass preflight unexpectedly created an output"
    ;;
  *)
    die "held-out quality outcome is invalid"
    ;;
esac

SCIENTIFIC_TIMING="$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_latency.sbatch"
[[ -x "$SCIENTIFIC_TIMING" ]] || \
  die "frozen scientific timing entrypoint is unavailable"
exec "$SCIENTIFIC_TIMING" \
  --repo-root "$SCIENTIFIC_REPO_ROOT" \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT" \
  --evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT" \
  --python "$PYTHON_BIN" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE"
