#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

# Account, QOS, partition, stdout, and stderr are intentionally absent here.
# The submitter must provide every one explicitly.

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
SPLIT=""
OUTPUT_DIR=""
VALIDATION_REPORT=""
LOCKBOX_REGISTRATION=""

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --split) SPLIT="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --validation-report) VALIDATION_REPORT="${2:?}"; shift 2 ;;
    --lockbox-registration) LOCKBOX_REGISTRATION="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "native-all-video evaluation requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "native-all-video evaluation is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "native-all-video evaluation requires exactly one node"
[[ "$TRAINING_COMMIT" == \
  "656086686dae723c942a4209a9d71cdb17ed6ccc" ]] || \
  die "training commit is not the frozen faithful-cascade commit"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "evaluator commit must be one full lowercase SHA-1"
[[ "$SPLIT" == "validation" || "$SPLIT" == "lockbox" ]] || \
  die "split must be validation or lockbox"
[[ -x "$PYTHON_BIN" ]] || die "explicit LACWM Python is unavailable"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || \
  die "explicit Wan directory is unavailable"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "explicit VideoX-Fun checkout is unavailable"
[[ -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] || \
  die "evaluator repository is unavailable"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || \
  die "immutable study root is unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the requested evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "evaluator repository must be clean"
[[ "$OUTPUT_DIR" == /* && ! -e "$OUTPUT_DIR" ]] || \
  die "explicit output must be an absent absolute path"
[[ -d "$(dirname "$OUTPUT_DIR")" && ! -L "$(dirname "$OUTPUT_DIR")" ]] || \
  die "explicit output parent is unavailable"

if [[ "$SPLIT" == "validation" ]]; then
  [[ -z "$VALIDATION_REPORT" && -z "$LOCKBOX_REGISTRATION" ]] || \
    die "validation forbids lockbox arguments"
  EXTRA_ARGS=()
else
  [[ -n "$VALIDATION_REPORT" && -n "$LOCKBOX_REGISTRATION" ]] || \
    die "lockbox requires validation report and registration"
  # Validate the complete K=4 composite gate before even stat'ing the
  # registration path.
  "$PYTHON_BIN" \
    "$REPO_ROOT/tools/vjepa2_native_all_video.py" \
    check-validation-gate \
    --repo-root "$REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --evaluator-commit "$EVALUATOR_COMMIT" \
    --validation-report "$VALIDATION_REPORT"
  [[ -s "$LOCKBOX_REGISTRATION" && ! -L "$LOCKBOX_REGISTRATION" ]] || \
    die "registered lockbox is unavailable after validation eligibility"
  EXTRA_ARGS=(
    --validation-report "$VALIDATION_REPORT"
    --lockbox-registration "$LOCKBOX_REGISTRATION"
  )
fi

EVALUATOR="$REPO_ROOT/tools/vjepa2_native_all_video.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$EVALUATOR" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || \
    die "required evaluator entrypoint is unavailable: $path"
done
for arm_dir in \
  vpm_parameter_matched_video \
  a1_auxiliary_objective_only \
  j1_joint_auxiliary_leads; do
  for path in \
    "$STUDY_ROOT/$arm_dir/arm_manifest.json" \
    "$STUDY_ROOT/$arm_dir/stage_manifest_update_1000.json" \
    "$STUDY_ROOT/$arm_dir/stage_outcome_update_1000.json" \
    "$STUDY_ROOT/$arm_dir/resolved_update_1000.yaml" \
    "$STUDY_ROOT/$arm_dir/snapshot.pt"; do
    [[ -s "$path" && ! -L "$path" ]] || \
      die "required final immutable input is unavailable: $path"
  done
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
  "$EVALUATOR" evaluate \
  --repo-root "$REPO_ROOT" \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT" \
  --evaluator-commit "$EVALUATOR_COMMIT" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size-per-rank 2 \
  "${EXTRA_ARGS[@]}"

[[ -s "$OUTPUT_DIR/report.json" && ! -L "$OUTPUT_DIR/report.json" ]] || \
  die "native-all-video evaluation did not produce report.json"
echo "$SPLIT native-all-video evaluation complete: $OUTPUT_DIR/report.json"
