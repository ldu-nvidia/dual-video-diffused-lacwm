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
STUDY_ROOT=""
TRAINING_COMMIT=""
EVALUATOR_COMMIT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "frontier timing requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "frontier timing is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "frontier timing requires one node"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime inputs are unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

SELECTION="$STUDY_ROOT/frontier_selection.json"
CONFIRMATION="$STUDY_ROOT/frontier_lockbox_confirmation.json"
FINAL_REPORT="$STUDY_ROOT/frontier_final_report.json"
LATENCY_PARENT="$STUDY_ROOT/frontier_latency"
LATENCY_OUTPUT="$LATENCY_PARENT/paired.json"
[[ -f "$CONFIRMATION" && ! -L "$CONFIRMATION" ]] || \
  die "lockbox confirmation is unavailable"
[[ ! -e "$LATENCY_PARENT" ]] || \
  die "fresh frontier timing parent already exists: $LATENCY_PARENT"
[[ ! -e "$FINAL_REPORT" ]] || \
  die "fresh final report already exists: $FINAL_REPORT"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE=disabled
unset PYTHONPATH
source "$REPO_ROOT/tools/env/activate_b200.sh"
"$PYTHON_BIN" "$REPO_ROOT/tools/env/verify_b200_runtime.py" \
  --expected-gpus 1 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

# The benchmark writes exclusively, but its parent must exist first.
mkdir "$LATENCY_PARENT"
"$PYTHON_BIN" "$REPO_ROOT/tools/benchmark_vjepa2_frontier_latency.py" \
  --repo-root "$REPO_ROOT" \
  --expected-commit "$TRAINING_COMMIT" \
  --benchmark-commit "$EVALUATOR_COMMIT" \
  --study-root "$STUDY_ROOT" \
  --selection "$SELECTION" \
  --device cuda:0 \
  --warmup-rounds 18 \
  --timed-rounds 120 \
  --bootstrap-samples 10000 \
  --confidence 0.95 \
  --seed 1234 \
  --output "$LATENCY_OUTPUT"
[[ -s "$LATENCY_OUTPUT" && ! -L "$LATENCY_OUTPUT" ]] || \
  die "frontier latency evidence was not produced"

"$PYTHON_BIN" "$REPO_ROOT/tools/vjepa2_nfe_frontier.py" finalize \
  --selection "$SELECTION" \
  --confirmation "$CONFIRMATION" \
  --latency "$LATENCY_OUTPUT" \
  --output "$FINAL_REPORT"
[[ -s "$FINAL_REPORT" && ! -L "$FINAL_REPORT" ]] || \
  die "frontier final report was not produced"
echo "Frontier workflow finalized: $FINAL_REPORT"
