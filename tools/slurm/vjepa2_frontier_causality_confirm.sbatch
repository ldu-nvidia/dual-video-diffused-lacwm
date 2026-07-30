#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
PYTHON_BIN=""
SELECTION=""
CONTINUATION=""

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --selection) SELECTION="${2:?}"; shift 2 ;;
    --continuation) CONTINUATION="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "causality confirmation requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "causality confirmation is non-requeueable"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen sidecar evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "sidecar evaluator repository must be clean"

CONTRACT="$REPO_ROOT/tools/vjepa2_lockbox_causality.py"
mapfile -t VALUES < <(
  "$PYTHON_BIN" "$CONTRACT" values \
    --repo-root "$REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --sidecar-evaluator-commit "$EVALUATOR_COMMIT" \
    --selection "$SELECTION" \
    --continuation "$CONTINUATION" \
    --require-autonomous-inventory
)
[[ "${#VALUES[@]}" == "10" ]] || die "causality confirmation values are incomplete"
AUTONOMOUS_INVENTORY="${VALUES[3]}"
SIDECAR_INVENTORY="${VALUES[5]}"
OUTPUT="${VALUES[6]}"
[[ -s "$AUTONOMOUS_INVENTORY" && -s "$SIDECAR_INVENTORY" ]] || \
  die "causality inventories are incomplete"
[[ ! -e "$OUTPUT" ]] || die "fresh causality confirmation already exists: $OUTPUT"

unset PYTHONPATH
"$PYTHON_BIN" "$CONTRACT" confirm \
  --repo-root "$REPO_ROOT" \
  --training-commit "$TRAINING_COMMIT" \
  --selection "$SELECTION" \
  --continuation "$CONTINUATION" \
  --autonomous-inventory "$AUTONOMOUS_INVENTORY" \
  --sidecar-inventory "$SIDECAR_INVENTORY" \
  --sidecar-evaluator-commit "$EVALUATOR_COMMIT" \
  --output "$OUTPUT"

[[ -s "$OUTPUT" && ! -L "$OUTPUT" ]] || \
  die "causality confirmation was not written"
echo "Separate J1 lockbox causality confirmation complete: $OUTPUT"
