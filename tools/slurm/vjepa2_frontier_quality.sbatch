#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
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
ARM=""
ARM_FROM_ARRAY=0
SPLIT=""
SELECTION=""

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
    --arm-from-array) ARM_FROM_ARRAY=1; shift ;;
    --split) SPLIT="${2:?}"; shift 2 ;;
    --selection) SELECTION="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if ((ARM_FROM_ARRAY)); then
  [[ -z "$ARM" ]] || die "--arm and --arm-from-array are mutually exclusive"
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || \
    die "--arm-from-array requires array task 0 or 1"
  if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    ARM="VPM"
  else
    ARM="J1"
  fi
else
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
    die "a direct --arm quality job must not be a Slurm array task"
fi
[[ "$ARM" == "VPM" || "$ARM" == "J1" ]] || die "arm must be VPM or J1"
[[ "$SPLIT" == "validation" || "$SPLIT" == "lockbox" ]] || \
  die "split must be validation or lockbox"
if [[ "$SPLIT" == "validation" ]]; then
  [[ -z "$SELECTION" ]] || die "validation must precede selection"
else
  [[ -n "$SELECTION" ]] || die "lockbox evaluation requires a selection"
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || die "quality evaluation requires Slurm"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "quality evaluation is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "quality evaluation requires one node"
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid training commit"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid evaluator commit"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime inputs are unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

case "$ARM" in
  VPM) RUN_DIR="$STUDY_ROOT/vpm_parameter_matched_video" ;;
  J1) RUN_DIR="$STUDY_ROOT/j1_joint_auxiliary_leads" ;;
esac
ARM_MANIFEST="$RUN_DIR/arm_manifest.json"
STUDY_MANIFEST="$STUDY_ROOT/study_manifest.json"
STAGE_MANIFEST="$RUN_DIR/stage_manifest_update_1000.json"
STAGE_OUTCOME="$RUN_DIR/stage_outcome_update_1000.json"
RESOLVED_CONFIG="$RUN_DIR/resolved_update_1000.yaml"
SNAPSHOT="$RUN_DIR/snapshot.pt"
EVALUATOR="$REPO_ROOT/tools/evaluate_vjepa2_quality.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in \
  "$ARM_MANIFEST" "$STUDY_MANIFEST" "$STAGE_MANIFEST" "$STAGE_OUTCOME" \
  "$RESOLVED_CONFIG" "$SNAPSHOT" "$EVALUATOR" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required final input is unavailable: $path"
done

if [[ "$SPLIT" == "validation" ]]; then
  OUTPUT_DIR="$RUN_DIR/frontier_quality/validation/update_1000"
  EXTRA_ARGS=(--frontier-split validation)
else
  [[ -f "$SELECTION" && ! -L "$SELECTION" ]] || \
    die "selection is unavailable: $SELECTION"
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
  OUTPUT_DIR="$RUN_DIR/frontier_quality/lockbox/${IDENTITIES[0]}/${IDENTITIES[1]}/update_1000"
  EXTRA_ARGS=(--frontier-split lockbox --frontier-selection "$SELECTION")
fi
[[ ! -e "$OUTPUT_DIR" ]] || die "fresh quality output already exists: $OUTPUT_DIR"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset PYTHONPATH
source "$ACTIVATE"
"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "$EVALUATOR" \
  --repo-root "$REPO_ROOT" \
  --expected-commit "$TRAINING_COMMIT" \
  --evaluator-commit "$EVALUATOR_COMMIT" \
  --run-dir "$RUN_DIR" \
  --arm-manifest "$ARM_MANIFEST" \
  --study-manifest "$STUDY_MANIFEST" \
  --stage-manifest "$STAGE_MANIFEST" \
  --stage-outcome "$STAGE_OUTCOME" \
  --resolved-config "$RESOLVED_CONFIG" \
  --snapshot "$SNAPSHOT" \
  --completed-updates 1000 \
  --batch-size-per-rank 2 \
  "${EXTRA_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR"

[[ -s "$OUTPUT_DIR/inventory.json" && ! -L "$OUTPUT_DIR/inventory.json" ]] || \
  die "$ARM $SPLIT evaluation did not produce an inventory"
echo "$ARM $SPLIT quality complete: $OUTPUT_DIR/inventory.json"
