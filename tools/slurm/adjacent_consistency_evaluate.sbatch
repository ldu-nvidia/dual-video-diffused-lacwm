#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --constraint=B200
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Single ACD-P0 endpoint-factorial allocation. This entrypoint never submits
# work and opens registered validation RGB only inside the evaluator's sealed
# post-endpoint barrier.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

REGISTRATION=""
REPO_ROOT=""
PYTHON_BIN=""
EXPECTED_COMMIT=""
USER_AUTHORIZATION=""
SUBMISSION_RECEIPT=""
LOG_DIR=""

while (($#)); do
  case "$1" in
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --user-authorization) USER_AUTHORIZATION="${2:?}"; shift 2 ;;
    --submission-receipt) SUBMISSION_RECEIPT="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "evaluation requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "evaluation is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" && "${SLURM_NTASKS:-1}" == "1" ]] || \
  die "evaluation requires one node and one launcher task"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "source commit is invalid"
for value in "$REGISTRATION" "$REPO_ROOT" "$PYTHON_BIN" "$USER_AUTHORIZATION" \
  "$SUBMISSION_RECEIPT" "$LOG_DIR"; do
  [[ "$value" == /* ]] || die "all paths must be absolute: $value"
done
case "$LOG_DIR" in /mnt/data1/*) ;; *) die "logs must be under /mnt/data1" ;; esac
for path in "$REGISTRATION" "$USER_AUTHORIZATION" "$SUBMISSION_RECEIPT"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required receipt is invalid: $path"
done
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ -d "$LOG_DIR" && ! -L "$LOG_DIR" ]] || die "log directory is invalid"
[[ -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$(pwd -P)" == "$LOG_DIR" ]] || die "allocation started outside its log directory"

REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || \
  die "repository root differs"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || \
  die "repository must be clean"

WORKFLOW="$REPO_ROOT/tools/slurm/adjacent_consistency_workflow.py"
EVALUATOR="$REPO_ROOT/tools/adjacent_consistency_evaluate.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$EVALUATOR" "$ACTIVATE" "$VERIFY"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation is missing: $path"
done

"$PYTHON_BIN" "$WORKFLOW" validate-submission \
  --phase evaluate \
  --source-commit "$EXPECTED_COMMIT" \
  --registration "$REGISTRATION" \
  --user-authorization "$USER_AUTHORIZATION" \
  --submission-receipt "$SUBMISSION_RECEIPT" \
  --slurm-job-id "$SLURM_JOB_ID" >/dev/null

IFS=$'\t' read -r \
  REGISTRATION_ID SOURCE_COMMIT OUTPUT_ROOT EVALUATION_DIR ANALYSIS_JSON \
  ANALYSIS_MARKDOWN REGISTERED_LOG_DIR REGISTERED_PYTHON WAN_DIR_VALUE VIDEOX_HOME_VALUE \
  < <(
    "$PYTHON_BIN" "$WORKFLOW" evaluation-values \
      --registration "$REGISTRATION" --format tsv --require-evaluation-fresh
  )

[[ "$SOURCE_COMMIT" == "$EXPECTED_COMMIT" ]] || die "registered source changed"
[[ "$REGISTRATION_ID" =~ ^[0-9a-f]{64}$ ]] || die "registration identity is invalid"
[[ "$(readlink -f "$REGISTERED_PYTHON")" == "$(readlink -f "$PYTHON_BIN")" ]] || \
  die "Python differs from registration"
[[ "$REGISTERED_LOG_DIR" == "$LOG_DIR" ]] || die "log directory differs from registration"
case "$OUTPUT_ROOT" in /mnt/data1/*) ;; *) die "output root must be under /mnt/data1" ;; esac
[[ ! -e "$EVALUATION_DIR" && ! -L "$EVALUATION_DIR" ]] || \
  die "evaluation output is not fresh"
for arm_dir in \
  "$OUTPUT_ROOT/training/acd-p0-rf-control-seed1234-u000400" \
  "$OUTPUT_ROOT/training/acd-p0-consistency-seed1234-u000400"; do
  for name in snapshot.pt training_complete.json acd_training_trace.jsonl \
    acd_training_trace_complete.json .hydra/config.yaml; do
    path="$arm_dir/$name"
    [[ -s "$path" && ! -L "$path" ]] || die "training prerequisite is missing: $path"
  done
done
FREE_BYTES="$(df -PB1 "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')"
[[ "$FREE_BYTES" =~ ^[0-9]+$ && "$FREE_BYTES" -ge 107374182400 ]] || \
  die "registered artifact filesystem has less than 100 GiB free"

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
  --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE"

IPC_TMPDIR="/tmp/acd-p0-evaluate-${UID}-${SLURM_JOB_ID}"
[[ ! -e "$IPC_TMPDIR" ]] || die "IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME_VALUE"
cd "$REPO_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$EVALUATOR" \
  --registration "$REGISTRATION" \
  --output "$EVALUATION_DIR"

[[ -s "$EVALUATION_DIR/inventory.json" && ! -L "$EVALUATION_DIR/inventory.json" ]] || \
  die "evaluation inventory is missing"
ARTIFACT_BYTES="$(du -sb "$OUTPUT_ROOT" | awk '{print $1}')"
[[ "$ARTIFACT_BYTES" =~ ^[0-9]+$ && "$ARTIFACT_BYTES" -le 107374182400 ]] || \
  die "ACD-P0 artifact ceiling exceeded"
echo "ACD-P0 endpoint evaluation complete: $EVALUATION_DIR/inventory.json"
