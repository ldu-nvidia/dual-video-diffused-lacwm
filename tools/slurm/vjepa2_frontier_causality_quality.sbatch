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

# Separate within-J1 lockbox controls. This job never writes any main-frontier
# selection, confirmation, timing, or final-report path.

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
SELECTION=""
CONTINUATION=""

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --selection) SELECTION="${2:?}"; shift 2 ;;
    --continuation) CONTINUATION="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "causality quality requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "causality quality is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "causality quality requires one node"
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid training commit"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid evaluator commit"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime inputs are unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen sidecar evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "sidecar evaluator repository must be clean"

CONTRACT="$REPO_ROOT/tools/vjepa2_lockbox_causality.py"
EVALUATOR="$REPO_ROOT/tools/evaluate_vjepa2_quality.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$CONTRACT" "$EVALUATOR" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required entrypoint is unavailable: $path"
done

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
[[ "${#VALUES[@]}" == "10" ]] || die "causality prerequisite values are incomplete"
OUTPUT_DIR="${VALUES[4]}"
AUTONOMOUS_INVENTORY="${VALUES[3]}"
[[ -s "$AUTONOMOUS_INVENTORY" && ! -L "$AUTONOMOUS_INVENTORY" ]] || \
  die "existing J1 autonomous lockbox inventory is unavailable"
[[ ! -e "$OUTPUT_DIR" ]] || die "fresh causality output already exists: $OUTPUT_DIR"

RUN_DIR="$STUDY_ROOT/j1_joint_auxiliary_leads"
ARM_MANIFEST="$RUN_DIR/arm_manifest.json"
STUDY_MANIFEST="$STUDY_ROOT/study_manifest.json"
STAGE_MANIFEST="$RUN_DIR/stage_manifest_update_1000.json"
STAGE_OUTCOME="$RUN_DIR/stage_outcome_update_1000.json"
RESOLVED_CONFIG="$RUN_DIR/resolved_update_1000.yaml"
SNAPSHOT="$RUN_DIR/snapshot.pt"
for path in \
  "$ARM_MANIFEST" "$STUDY_MANIFEST" "$STAGE_MANIFEST" "$STAGE_OUTCOME" \
  "$RESOLVED_CONFIG" "$SNAPSHOT"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required final J1 input is unavailable: $path"
done

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
  --frontier-split lockbox \
  --frontier-selection "$SELECTION" \
  --frontier-causality-sidecar \
  --frontier-continuation "$CONTINUATION" \
  --output-dir "$OUTPUT_DIR"

[[ -s "$OUTPUT_DIR/inventory.json" && ! -L "$OUTPUT_DIR/inventory.json" ]] || \
  die "J1 causality sidecar did not produce an inventory"
echo "J1 fresh-lockbox causality controls complete: $OUTPUT_DIR/inventory.json"
