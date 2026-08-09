#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Future, independently authorized IPQ-TC1 training allocation.  This file
# never submits another job. Submit from the registered `_slurm_logs` directory
# so Slurm's default stdout/stderr remain under the registered artifact root.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

ARM=""; REGISTRATION=""; REPO_ROOT=""; PYTHON_BIN=""; EXPECTED_COMMIT=""
while (($#)); do
  case "$1" in
    --arm) ARM="${2:?}"; shift 2 ;;
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "training requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "training is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" && "${SLURM_NTASKS:-1}" == "1" ]] || \
  die "training requires one node and one launcher task"
[[ "$ARM" == "IPQ-SYNC" || "$ARM" == "IPQ-INDEP" ]] || die "invalid arm"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid source SHA"
for value in "$REGISTRATION" "$REPO_ROOT" "$PYTHON_BIN"; do
  [[ "$value" == /* ]] || die "all paths must be absolute: $value"
done
[[ -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || die "registration is invalid"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ -x "$PYTHON_BIN" ]] || die "registered Python is unavailable"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || \
  die "repository root differs"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

WORKFLOW="$REPO_ROOT/tools/slurm/invertible_two_clock_workflow.py"
TRAINER="$REPO_ROOT/projects/latent_action_models/train_invertible_two_clock.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$TRAINER" "$ACTIVATE" "$VERIFY"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required file is missing: $path"
done

"$PYTHON_BIN" "$WORKFLOW" write-arm-plan \
  --registration "$REGISTRATION" --arm "$ARM" >/dev/null
IFS=$'\t' read -r \
  ARM_CODE ARM_MODE CONFIG_NAME RUN_NAME RUN_IDENTITY OUTPUT_ROOT RUN_DIR EVAL_DIR \
  ARM_PLAN SLURM_LOG_DIR PARENT_SNAPSHOT PARENT_CONFIG TRAIN_MANIFEST TRAIN_METADATA \
  REGISTERED_PYTHON WAN_DIR_VALUE VIDEOX_HOME_VALUE WANDB_ENABLED \
  < <("$PYTHON_BIN" "$WORKFLOW" arm-values --registration "$REGISTRATION" \
      --arm "$ARM" --format tsv --require-training-fresh)

[[ "$ARM_CODE" == "$ARM" ]] || die "registered arm changed"
[[ "$REGISTERED_PYTHON" == "$(readlink -f "$PYTHON_BIN")" ]] || \
  die "Python differs from registration"
[[ "$RUN_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "run identity is invalid"
[[ "$(pwd -P)" == "$SLURM_LOG_DIR" ]] || \
  die "submit from registered _slurm_logs so scheduler logs stay in scope"
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || die "training output is not fresh"
[[ -f "$PARENT_SNAPSHOT" && ! -L "$PARENT_SNAPSHOT" ]] || \
  die "parent snapshot is unavailable"
[[ -f "$PARENT_CONFIG" && ! -L "$PARENT_CONFIG" ]] || \
  die "parent config is unavailable"

FREE_BYTES="$(df -PB1 "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')"
[[ "$FREE_BYTES" =~ ^[0-9]+$ && "$FREE_BYTES" -ge 1099511627776 ]] || \
  die "registered artifact filesystem has less than 1 TiB free"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
source "$ACTIVATE"
"$PYTHON_BIN" "$VERIFY" --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE"

export IPQ_TC1_PARENT_SNAPSHOT="$PARENT_SNAPSHOT"
export IPQ_TC1_PARENT_RESOLVED_CONFIG="$PARENT_CONFIG"
export IPQ_TC1_PARENT_SNAPSHOT_SHA256="de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
export IPQ_TC1_PARENT_CONFIG_SHA256="ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
export IPQ_TC1_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export IPQ_TC1_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export IPQ_TC1_RUN_ROOT="$(dirname "$RUN_DIR")"
export LACWM_RUN_IDENTITY_SHA256="$RUN_IDENTITY"
unset WANDB_RUN_GROUP WANDB_GROUP WANDB_JOB_TYPE WANDB_NOTES WANDB_TAGS

WANDB_OVERRIDES=()
if [[ "$WANDB_ENABLED" == "True" ]]; then
  "$PYTHON_BIN" "$WORKFLOW" wandb-check --registration "$REGISTRATION" >/dev/null
  export IPQ_TC1_WANDB_AUTHORIZED=1
  export WANDB_ENTITY=zijiandu
  export WANDB_PROJECT=dual-video-diffusion-private
  export WANDB_MODE=online
  export WANDB_SILENT=true
  export WANDB_DIR="$OUTPUT_ROOT/wandb"
  export WANDB_CACHE_DIR="$OUTPUT_ROOT/wandb-cache"
  export WANDB_CONFIG_DIR="$OUTPUT_ROOT/wandb-config"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
  WANDB_OVERRIDES=(
    "wandb.enabled=true" "wandb.mode=online" "wandb.entity=zijiandu"
    "wandb.project=dual-video-diffusion-private" "wandb.group=null"
    "+wandb.id=$RUN_IDENTITY" "+wandb.resume=never"
  )
else
  export IPQ_TC1_WANDB_AUTHORIZED=0
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  unset WANDB_RUN_ID WANDB_NAME WANDB_RESUME
fi

IPC_TMPDIR="/tmp/ipq-tc1-${UID}-${SLURM_JOB_ID}-${ARM_CODE}"
[[ ! -e "$IPC_TMPDIR" ]] || die "IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"

PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME_VALUE"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$TRAINER" "+experiments_0908=$CONFIG_NAME" \
  "hydra.run.dir=$RUN_DIR" "hydra.sweep.dir=$RUN_DIR" \
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt" \
  "${WANDB_OVERRIDES[@]}"

for path in "$RUN_DIR/snapshot.pt" "$RUN_DIR/training_complete.json" \
  "$RUN_DIR/ipq_training_trace.jsonl" "$RUN_DIR/ipq_training_trace_complete.json" \
  "$RUN_DIR/.hydra/config.yaml"; do
  [[ -s "$path" && ! -L "$path" ]] || die "training output is missing: $path"
done
echo "IPQ-TC1 training complete: $ARM_CODE"
