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

# One independently authorized ACD-P0 matched-arm allocation. This entrypoint
# never submits another job and requires the held-job workflow receipt before
# it can open the registered training population.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

LUSTRE_ARTIFACT_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
approved_artifact_path() {
  case "$1" in
    /mnt/data1|/mnt/data1/*|/mnt/data2|/mnt/data2/*|\
      "$LUSTRE_ARTIFACT_ROOT"|"$LUSTRE_ARTIFACT_ROOT"/*) ;;
    *) die "$2 must be under an approved artifact root" ;;
  esac
}

REGISTRATION=""
ARM=""
REPO_ROOT=""
PYTHON_BIN=""
EXPECTED_COMMIT=""
USER_AUTHORIZATION=""
SUBMISSION_RECEIPT=""
LOG_DIR=""

while (($#)); do
  case "$1" in
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
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
  die "training requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "training is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" && "${SLURM_NTASKS:-1}" == "1" ]] || \
  die "training requires one node and one launcher task"
[[ "$ARM" == "ACD-RF-CONT" || "$ARM" == "ACD-CONS" ]] || die "invalid arm"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "source commit is invalid"
for value in "$REGISTRATION" "$REPO_ROOT" "$PYTHON_BIN" "$USER_AUTHORIZATION" \
  "$SUBMISSION_RECEIPT" "$LOG_DIR"; do
  [[ "$value" == /* ]] || die "all paths must be absolute: $value"
done
approved_artifact_path "$LOG_DIR" "logs"
approved_artifact_path "$REGISTRATION" "registration receipt"
approved_artifact_path "$USER_AUTHORIZATION" "user authorization receipt"
approved_artifact_path "$SUBMISSION_RECEIPT" "submission receipt"
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
TRAINER="$REPO_ROOT/projects/latent_action_models/train_adjacent_consistency.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$TRAINER" "$ACTIVATE" "$VERIFY"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation is missing: $path"
done

"$PYTHON_BIN" "$WORKFLOW" validate-submission \
  --phase train \
  --source-commit "$EXPECTED_COMMIT" \
  --registration "$REGISTRATION" \
  --arm "$ARM" \
  --user-authorization "$USER_AUTHORIZATION" \
  --submission-receipt "$SUBMISSION_RECEIPT" \
  --slurm-job-id "$SLURM_JOB_ID" >/dev/null

"$PYTHON_BIN" "$WORKFLOW" write-arm-plan \
  --registration "$REGISTRATION" --arm "$ARM" >/dev/null
IFS=$'\t' read -r \
  ARM_CODE ARM_MODE CONFIG_NAME RUN_NAME RUN_IDENTITY ARM_CONFIG_SEMANTIC_SHA256 \
  REGISTERED_ARM_CONFIG REGISTRATION_ID SOURCE_COMMIT OUTPUT_ROOT RUN_DIR ARM_PLAN \
  REGISTERED_LOG_DIR PARENT_SNAPSHOT PARENT_CONFIG TRAIN_MANIFEST TRAIN_METADATA \
  REGISTERED_PYTHON WAN_DIR_VALUE VIDEOX_HOME_VALUE \
  < <(
    "$PYTHON_BIN" "$WORKFLOW" arm-values \
      --registration "$REGISTRATION" --arm "$ARM" --format tsv \
      --require-training-fresh
  )

[[ "$ARM_CODE" == "$ARM" ]] || die "registered arm changed"
[[ "$SOURCE_COMMIT" == "$EXPECTED_COMMIT" ]] || die "registered source changed"
[[ "$REGISTRATION_ID" =~ ^[0-9a-f]{64}$ ]] || die "registration identity is invalid"
[[ "$RUN_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "run identity is invalid"
[[ "$ARM_CONFIG_SEMANTIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "registered arm config identity is invalid"
[[ -s "$REGISTERED_ARM_CONFIG" && ! -L "$REGISTERED_ARM_CONFIG" ]] || \
  die "registered arm config is unavailable"
[[ "$(readlink -f "$REGISTERED_PYTHON")" == "$(readlink -f "$PYTHON_BIN")" ]] || \
  die "Python differs from registration"
[[ "$REGISTERED_LOG_DIR" == "$LOG_DIR" ]] || die "log directory differs from registration"
approved_artifact_path "$OUTPUT_ROOT" "output root"
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || die "training output is not fresh"
for path in "$PARENT_SNAPSHOT" "$PARENT_CONFIG" "$TRAIN_MANIFEST" "$TRAIN_METADATA"; do
  [[ -f "$path" && ! -L "$path" ]] || die "registered training input is unavailable: $path"
done
FREE_BYTES="$(df -PB1 "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')"
[[ "$FREE_BYTES" =~ ^[0-9]+$ && "$FREE_BYTES" -ge 107374182400 ]] || \
  die "registered artifact filesystem has less than 100 GiB free"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export ACD_P0_SOURCE_COMMIT="$SOURCE_COMMIT"
export ACD_P0_REGISTRATION_IDENTITY="$REGISTRATION_ID"
export ACD_P0_REGISTRATION_PATH="$REGISTRATION"
export ACD_P0_ARM_CONFIG_SEMANTIC_SHA256="$ARM_CONFIG_SEMANTIC_SHA256"
export ACD_P0_PARENT_SNAPSHOT="$PARENT_SNAPSHOT"
export ACD_P0_PARENT_RESOLVED_CONFIG="$PARENT_CONFIG"
export ACD_P0_PARENT_SNAPSHOT_SHA256="de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
export ACD_P0_PARENT_CONFIG_SHA256="ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
export ACD_P0_PARENT_CANONICAL_MODEL_STATE_SHA256="d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0"
export ACD_P0_PARENT_RUNTIME_TENSOR_STATE_SHA256="82ffa76e99574f5202831897411e4b22eb281c1e2512dc358238bef72d5a4d5a"
export ACD_P0_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export ACD_P0_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export ACD_P0_RUN_ROOT="$OUTPUT_ROOT/training"
export LACWM_RUN_IDENTITY_SHA256="$RUN_IDENTITY"
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

IPC_TMPDIR="/tmp/acd-p0-${UID}-${SLURM_JOB_ID}-${ARM_CODE}"
[[ ! -e "$IPC_TMPDIR" ]] || die "IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME_VALUE"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$TRAINER" \
  "+experiments_0908=$CONFIG_NAME" \
  "hydra.run.dir=$RUN_DIR" \
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt" \
  "trainer.config.visualization.viz_path=$RUN_DIR/visualization" \
  "wandb.enabled=false" \
  "wandb.mode=disabled"

for path in \
  "$RUN_DIR/snapshot.pt" \
  "$RUN_DIR/training_complete.json" \
  "$RUN_DIR/acd_training_trace.jsonl" \
  "$RUN_DIR/acd_training_trace_complete.json" \
  "$RUN_DIR/.hydra/config.yaml"; do
  [[ -s "$path" && ! -L "$path" ]] || die "training output is missing: $path"
done
ARTIFACT_BYTES="$(du -sb "$OUTPUT_ROOT" | awk '{print $1}')"
[[ "$ARTIFACT_BYTES" =~ ^[0-9]+$ && "$ARTIFACT_BYTES" -le 107374182400 ]] || \
  die "ACD-P0 artifact ceiling exceeded"
echo "ACD-P0 training complete: $ARM_CODE"
