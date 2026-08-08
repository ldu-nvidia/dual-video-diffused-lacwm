#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089
#SBATCH --chdir=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/two_clock_consistency/_slurm_logs
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/two_clock_consistency/_slurm_logs/two-clock-consistency-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/two_clock_consistency/_slurm_logs/two-clock-consistency-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

MODE=""
ARM=""
REGISTRATION=""
REPO_ROOT="${TWO_CLOCK_CONSISTENCY_REPO_ROOT:-}"
PYTHON_BIN="${TWO_CLOCK_CONSISTENCY_PYTHON:-}"
EXPECTED_COMMIT=""

while (($#)); do
  case "$1" in
    --mode) MODE="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" ]] || die "entrypoint requires Slurm"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "screen is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "screen requires one node"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || die "screen requires one launcher task"
[[ "$MODE" == "arm" || "$MODE" == "analyze" ]] || die "--mode must be arm or analyze"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute Git worktree"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

WORKFLOW="$REPO_ROOT/tools/slurm/two_clock_consistency_workflow.py"
EVALUATOR="$REPO_ROOT/tools/two_clock_consistency_evaluate.py"
TRAINER="$REPO_ROOT/projects/latent_action_models/train_two_clock_consistency.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$EVALUATOR" "$TRAINER" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation missing: $path"
done

unset PYTHONPATH
export LACWM_PYTHON="$PYTHON_BIN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

if [[ "$MODE" == "analyze" ]]; then
  [[ -z "$ARM" ]] || die "analysis mode does not accept --arm"
  exec "$PYTHON_BIN" "$WORKFLOW" analyze --registration "$REGISTRATION"
fi

[[ "$ARM" == "TC-CONT" || "$ARM" == "TC-CONS" ]] || \
  die "arm mode requires TC-CONT or TC-CONS"

"$PYTHON_BIN" "$WORKFLOW" write-arm-plan \
  --registration "$REGISTRATION" --arm "$ARM" >/dev/null

IFS=$'\t' read -r \
  ARM_CODE CONFIG_NAME RUN_NAME RUN_IDENTITY WANDB_RUN_ID TRAINING_ROOT RUN_DIR EVAL_DIR \
  PARENT_SNAPSHOT TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  WAN_DIR_VALUE VIDEOX_HOME_VALUE REGISTERED_PYTHON \
  < <(
    "$PYTHON_BIN" "$WORKFLOW" arm-values \
      --registration "$REGISTRATION" --arm "$ARM" --format tsv
  )

[[ "$ARM_CODE" == "$ARM" ]] || die "arm values changed"
[[ "$REGISTERED_PYTHON" == "$(readlink -f "$PYTHON_BIN")" ]] || \
  die "runtime Python differs from registration"
[[ "$RUN_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "run identity is invalid"
[[ "$WANDB_RUN_ID" == "$RUN_IDENTITY" ]] || die "W&B run ID is not arm-bound"
[[ ! -e "$RUN_DIR" && ! -e "$EVAL_DIR" ]] || die "arm outputs are not fresh"
[[ -f "$PARENT_SNAPSHOT" && ! -L "$PARENT_SNAPSHOT" ]] || \
  die "parent snapshot is unavailable"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
# The activation guard validates these directories while it is sourced.  Bind
# them from the immutable registration before activation instead of allowing
# the workstation-only /mnt/data2 defaults to take effect on the cluster.
source "$ACTIVATE"

export TWO_CLOCK_CONSISTENCY_VPM_SNAPSHOT="$PARENT_SNAPSHOT"
export TWO_CLOCK_CONSISTENCY_VPM_SNAPSHOT_SHA256="f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
export TWO_CLOCK_CONSISTENCY_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export TWO_CLOCK_CONSISTENCY_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export TWO_CLOCK_CONSISTENCY_VAL_CLIP_MANIFEST="$VAL_MANIFEST"
export TWO_CLOCK_CONSISTENCY_VAL_CACHE_METADATA="$VAL_METADATA"
export TWO_CLOCK_CONSISTENCY_RUN_ROOT="$TRAINING_ROOT"
export LACWM_RUN_IDENTITY_SHA256="$RUN_IDENTITY"
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
export WANDB_MODE=online
export WANDB_SILENT=true
unset WANDB_RUN_GROUP WANDB_GROUP WANDB_JOB_TYPE WANDB_NOTES
export WANDB_DIR="$(dirname "$TRAINING_ROOT")/wandb"
export WANDB_CACHE_DIR="$(dirname "$TRAINING_ROOT")/wandb-cache"
export WANDB_CONFIG_DIR="$(dirname "$TRAINING_ROOT")/wandb-config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

IPC_TMPDIR="/tmp/lacwm-two-clock-${UID}-${SLURM_JOB_ID}-${ARM_CODE}"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"

"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE"
"$PYTHON_BIN" "$WORKFLOW" wandb-check >/dev/null

PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME_VALUE"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 "$TRAINER" \
  "+experiments_0908=$CONFIG_NAME" \
  "hydra.run.dir=$RUN_DIR" \
  "hydra.sweep.dir=$RUN_DIR" \
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt" \
  "wandb.enabled=true" \
  "wandb.mode=online" \
  "wandb.entity=zijiandu" \
  "wandb.project=dual-video-diffusion-private" \
  "wandb.group=null" \
  "+wandb.id=$WANDB_RUN_ID" \
  "+wandb.resume=never"

for path in \
  "$RUN_DIR/snapshot.pt" \
  "$RUN_DIR/training_complete.json" \
  "$RUN_DIR/two_clock_consistency_training_trace.jsonl" \
  "$RUN_DIR/two_clock_consistency_training_trace_complete.json" \
  "$RUN_DIR/.hydra/config.yaml"; do
  [[ -s "$path" && ! -L "$path" ]] || die "training output missing: $path"
done

cd "$REPO_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 "$EVALUATOR" evaluate \
  --registration "$REGISTRATION" \
  --arm "$ARM_CODE" \
  --output-dir "$EVAL_DIR" \
  --batch-size-per-rank 2

[[ -s "$EVAL_DIR/inventory.json" ]] || die "evaluation inventory is absent"
echo "Two-clock consistency arm complete: $ARM_CODE -> $EVAL_DIR/inventory.json"
