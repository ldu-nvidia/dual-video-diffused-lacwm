#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
EVALUATOR_COMMIT=""
PYTHON_BIN=""
while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "confirmation requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "confirmation is non-requeueable"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
unset PYTHONPATH

SELECTION="$STUDY_ROOT/frontier_selection.json"
CONFIRMATION="$STUDY_ROOT/frontier_lockbox_confirmation.json"
[[ ! -e "$CONFIRMATION" ]] || \
  die "fresh confirmation output already exists: $CONFIRMATION"
"$PYTHON_BIN" "$REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
  require-selection --selection "$SELECTION"

mapfile -t IDENTITIES < <(
  "$PYTHON_BIN" - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path
with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["lockbox_registration"]["identity_sha256"])
print(payload["identity_sha256"])
PY
)
[[ "${#IDENTITIES[@]}" == "2" ]] || die "selection identities are incomplete"
[[ "${IDENTITIES[0]}" =~ ^[0-9a-f]{64}$ && \
  "${IDENTITIES[1]}" =~ ^[0-9a-f]{64}$ ]] || \
  die "selection identities must be full SHA-256 values"

VPM_DIR="$STUDY_ROOT/vpm_parameter_matched_video/frontier_quality/lockbox/${IDENTITIES[0]}/${IDENTITIES[1]}/update_1000"
J1_DIR="$STUDY_ROOT/j1_joint_auxiliary_leads/frontier_quality/lockbox/${IDENTITIES[0]}/${IDENTITIES[1]}/update_1000"
shopt -s nullglob
VPM_ROWS=("$VPM_DIR"/rank_*.jsonl)
J1_ROWS=("$J1_DIR"/rank_*.jsonl)
shopt -u nullglob
[[ "${#VPM_ROWS[@]}" == "8" && "${#J1_ROWS[@]}" == "8" ]] || \
  die "lockbox quality must contain exactly eight rank files per arm"
[[ -s "$VPM_DIR/inventory.json" && -s "$J1_DIR/inventory.json" ]] || \
  die "lockbox quality inventories are incomplete"

"$PYTHON_BIN" "$REPO_ROOT/tools/vjepa2_nfe_frontier.py" confirm \
  --selection "$SELECTION" \
  --j1-rows "${J1_ROWS[@]}" \
  --vpm-rows "${VPM_ROWS[@]}" \
  --expected-clips 128 \
  --output "$CONFIRMATION"
[[ -s "$CONFIRMATION" && ! -L "$CONFIRMATION" ]] || \
  die "lockbox confirmation was not produced"
echo "Lockbox quality confirmed: $CONFIRMATION"
