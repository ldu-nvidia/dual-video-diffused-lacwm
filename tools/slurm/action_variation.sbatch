#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

MODE=""
ARM=""
REGISTRATION=""
REPO_ROOT=""
PYTHON_BIN=""
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
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || die "screen is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "screen requires one node"
[[ "${SLURM_NTASKS:-1}" == 1 ]] || die "screen requires one launcher task"
[[ "$MODE" == arm || "$MODE" == analyze ]] || die "invalid --mode"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute Git worktree"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

WORKFLOW="$REPO_ROOT/tools/slurm/action_variation_workflow.py"
TRAINER="$REPO_ROOT/projects/latent_action_models/train_action_variation.py"
EVALUATOR="$REPO_ROOT/tools/action_variation_evaluate.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$TRAINER" "$EVALUATOR" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation missing: $path"
done

unset PYTHONPATH
export LACWM_PYTHON="$PYTHON_BIN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

if [[ "$MODE" == analyze ]]; then
  [[ -z "$ARM" ]] || die "analysis mode does not accept --arm"
  exec "$PYTHON_BIN" "$WORKFLOW" analyze --registration "$REGISTRATION"
fi

if [[ -z "$ARM" ]]; then
  case "${SLURM_ARRAY_TASK_ID:-}" in
    0) ARM="AV-CONT" ;;
    1) ARM="AV-DELTA" ;;
    *) die "arm mode requires --arm or array index 0/1" ;;
  esac
fi
[[ "$ARM" == AV-CONT || "$ARM" == AV-DELTA ]] || die "invalid arm"

IFS=$'\t' read -r \
  ARM_CODE CONFIG_NAME RUN_NAME RUN_IDENTITY TRAINING_ROOT RUN_DIR EVAL_DIR PLAN \
  RUNTIME_RECEIPT \
  PARENT_SNAPSHOT TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  STATS STATS_SHA PYTHON_REGISTERED WAN_DIR_VALUE VIDEOX_HOME_VALUE \
  < <(
    "$PYTHON_BIN" "$WORKFLOW" arm-values --registration "$REGISTRATION" \
      --arm "$ARM" --format tsv
  )
[[ "$ARM_CODE" == "$ARM" ]] || die "arm values changed"
[[ "$PYTHON_REGISTERED" == "$PYTHON_BIN" ]] || die "runtime Python differs"
[[ "$RUN_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "run identity is invalid"
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || die "training output is not fresh"
[[ ! -e "$EVAL_DIR" && ! -L "$EVAL_DIR" ]] || die "evaluation output is not fresh"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
source "$ACTIVATE"
export ACTION_VARIATION_VPM_SNAPSHOT="$PARENT_SNAPSHOT"
export ACTION_VARIATION_VPM_SNAPSHOT_SHA256="f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
export ACTION_VARIATION_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export ACTION_VARIATION_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export ACTION_VARIATION_VAL_CLIP_MANIFEST="$VAL_MANIFEST"
export ACTION_VARIATION_VAL_CACHE_METADATA="$VAL_METADATA"
export ACTION_VARIATION_STATS="$STATS"
export ACTION_VARIATION_STATS_SHA256="$STATS_SHA"
export ACTION_VARIATION_RUN_ROOT="$TRAINING_ROOT"
export ACTION_VARIATION_REGISTRATION="$REGISTRATION"
export ACTION_VARIATION_ARM_CODE="$ARM_CODE"
export LACWM_RUN_IDENTITY_SHA256="$RUN_IDENTITY"
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
export WANDB_MODE=online
export WANDB_SILENT=true
unset WANDB_RUN_GROUP WANDB_GROUP WANDB_JOB_TYPE WANDB_NOTES
STUDY_ROOT="$(dirname "$TRAINING_ROOT")"
export WANDB_DIR="$STUDY_ROOT/wandb"
export WANDB_CACHE_DIR="$STUDY_ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$STUDY_ROOT/wandb-config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
TMPDIR_VALUE="/tmp/lacwm-action-variation-${UID}-${SLURM_JOB_ID}-${ARM_CODE}"
[[ ! -e "$TMPDIR_VALUE" ]] || die "fresh IPC directory already exists"
mkdir "$TMPDIR_VALUE"
export TMPDIR="$TMPDIR_VALUE"

RUNTIME_VERIFIER_JSON="$TMPDIR/runtime_verifier_output.json"
[[ ! -e "$RUNTIME_VERIFIER_JSON" && ! -L "$RUNTIME_VERIFIER_JSON" ]] || \
  die "fresh runtime verifier output exists"
"$PYTHON_BIN" "$VERIFY_RUNTIME" --expected-gpus 8 --require-b200 \
  --wan-dir "$WAN_DIR" > "$RUNTIME_VERIFIER_JSON"
"$PYTHON_BIN" "$WORKFLOW" wandb-check >/dev/null
"$PYTHON_BIN" "$WORKFLOW" write-runtime-receipt \
  --registration "$REGISTRATION" --arm "$ARM_CODE" \
  --verifier-json "$RUNTIME_VERIFIER_JSON" >/dev/null
[[ -s "$RUNTIME_RECEIPT" && ! -L "$RUNTIME_RECEIPT" ]] || \
  die "sealed runtime verification receipt is absent"
"$PYTHON_BIN" "$WORKFLOW" write-arm-plan \
  --registration "$REGISTRATION" --arm "$ARM_CODE" >/dev/null

PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" \
  "+experiments_0908=$CONFIG_NAME" \
  "hydra.run.dir=$RUN_DIR" \
  "hydra.sweep.dir=$RUN_DIR" \
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt" \
  "wandb.enabled=true" "wandb.mode=online" "wandb.entity=zijiandu" \
  "wandb.project=dual-video-diffusion-private" "wandb.group=null" \
  "+wandb.id=$RUN_IDENTITY" "+wandb.resume=never"

for path in "$RUN_DIR/snapshot.pt" "$RUN_DIR/training_complete.json" \
  "$RUN_DIR/action_variation_training_trace.jsonl" \
  "$RUN_DIR/action_variation_training_trace_complete.json" \
  "$RUN_DIR/.hydra/config.yaml"; do
  [[ -s "$path" && ! -L "$path" ]] || die "training output missing: $path"
done

cd "$REPO_ROOT"
"$PYTHON_BIN" "$WORKFLOW" seal-wandb \
  --registration "$REGISTRATION" --arm "$ARM_CODE" >/dev/null
[[ -s "$RUN_DIR/wandb_run_complete.json" && ! -L "$RUN_DIR/wandb_run_complete.json" ]] || \
  die "verified W&B completion receipt is absent"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" \
  evaluate --registration "$REGISTRATION" --arm "$ARM_CODE" \
  --output-dir "$EVAL_DIR" --batch-size-per-rank 1
[[ -s "$EVAL_DIR/inventory.json" && ! -L "$EVAL_DIR/inventory.json" ]] || \
  die "evaluation inventory is absent"
echo "Action-variation arm complete: $ARM_CODE"
