#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=B200
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --open-mode=append
#SBATCH --no-requeue

# ACD-P0 three-copy memory preflight. This entrypoint never submits another
# job and cannot run from a raw sbatch call: the workflow-created held-job
# receipt and its source-bound user authorization are both required.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

SOURCE_REPO=""
EXPECTED_COMMIT=""
PYTHON_BIN=""
PARENT_SNAPSHOT=""
PARENT_CONFIG=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
OUTPUT=""
USER_AUTHORIZATION=""
SUBMISSION_RECEIPT=""
LOG_DIR=""

while (($#)); do
  case "$1" in
    --source-repo) SOURCE_REPO="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --parent-snapshot) PARENT_SNAPSHOT="${2:?}"; shift 2 ;;
    --parent-resolved-config) PARENT_CONFIG="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --output) OUTPUT="${2:?}"; shift 2 ;;
    --user-authorization) USER_AUTHORIZATION="${2:?}"; shift 2 ;;
    --submission-receipt) SUBMISSION_RECEIPT="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "memory smoke requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "memory smoke is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" && "${SLURM_NTASKS:-1}" == "1" ]] || \
  die "memory smoke requires one node and one launcher task"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "source commit is invalid"
for value in "$SOURCE_REPO" "$PYTHON_BIN" "$PARENT_SNAPSHOT" "$PARENT_CONFIG" \
  "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE" "$OUTPUT" "$USER_AUTHORIZATION" \
  "$SUBMISSION_RECEIPT" "$LOG_DIR"; do
  [[ "$value" == /* ]] || die "all paths must be absolute: $value"
done
case "$OUTPUT" in /mnt/data1/*) ;; *) die "output must be under /mnt/data1" ;; esac
case "$LOG_DIR" in /mnt/data1/*) ;; *) die "logs must be under /mnt/data1" ;; esac
[[ -d "$SOURCE_REPO" && ! -L "$SOURCE_REPO" ]] || die "source repository is invalid"
[[ -x "$PYTHON_BIN" ]] || die "Python is unavailable"
for path in "$PARENT_SNAPSHOT" "$PARENT_CONFIG" "$USER_AUTHORIZATION" \
  "$SUBMISSION_RECEIPT"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required receipt/input is invalid: $path"
done
for path in "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE" "$LOG_DIR"; do
  [[ -d "$path" && ! -L "$path" ]] || die "required directory is invalid: $path"
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || die "memory receipt output is not fresh"
[[ "$(pwd -P)" == "$LOG_DIR" ]] || die "allocation started outside its log directory"

SOURCE_REPO="$(cd "$SOURCE_REPO" && pwd -P)"
[[ "$(git -C "$SOURCE_REPO" rev-parse --show-toplevel)" == "$SOURCE_REPO" ]] || \
  die "source repository root differs"
[[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "source repository commit differs"
[[ -z "$(git -C "$SOURCE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
  die "source repository must be clean"

WORKFLOW="$SOURCE_REPO/tools/slurm/adjacent_consistency_workflow.py"
SMOKE="$SOURCE_REPO/tools/adjacent_consistency_memory_smoke.py"
ACTIVATE="$SOURCE_REPO/tools/env/activate_b200.sh"
VERIFY="$SOURCE_REPO/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$SMOKE" "$ACTIVATE" "$VERIFY"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation is missing: $path"
done

"$PYTHON_BIN" "$WORKFLOW" validate-submission \
  --phase memory_smoke \
  --source-commit "$EXPECTED_COMMIT" \
  --user-authorization "$USER_AUTHORIZATION" \
  --submission-receipt "$SUBMISSION_RECEIPT" \
  --slurm-job-id "$SLURM_JOB_ID" >/dev/null

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_GROUP WANDB_NAME WANDB_RESUME \
  WANDB_JOB_TYPE WANDB_NOTES WANDB_TAGS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
source "$ACTIVATE"
"$PYTHON_BIN" "$VERIFY" \
  --expected-gpus 1 --require-b200 --wan-dir "$WAN_DIR_VALUE"

IPC_TMPDIR="/tmp/acd-p0-memory-${UID}-${SLURM_JOB_ID}"
[[ ! -e "$IPC_TMPDIR" ]] || die "IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"
PROJECT_ROOT="$SOURCE_REPO/projects/latent_action_models"
export PYTHONPATH="$SOURCE_REPO:$PROJECT_ROOT:$SOURCE_REPO/tools/env/videox_shim:$VIDEOX_HOME_VALUE"
cd "$SOURCE_REPO"
"$PYTHON_BIN" "$SMOKE" \
  --source-repo "$SOURCE_REPO" \
  --expected-commit "$EXPECTED_COMMIT" \
  --parent-snapshot "$PARENT_SNAPSHOT" \
  --parent-resolved-config "$PARENT_CONFIG" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --output "$OUTPUT"

[[ -s "$OUTPUT" && ! -L "$OUTPUT" ]] || die "memory-smoke receipt is missing"
echo "ACD-P0 one-B200 memory smoke complete: $OUTPUT"
