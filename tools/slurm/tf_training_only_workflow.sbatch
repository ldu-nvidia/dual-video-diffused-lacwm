#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
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
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || die "workflow is non-requeueable"
[[ "$MODE" == canary || "$MODE" == train || "$MODE" == seal || \
   "$MODE" == evaluate || "$MODE" == analyze ]] || die "invalid mode"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute worktree"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

HELPER="$REPO_ROOT/tools/tf_training_only_screen.py"
CANARY="$REPO_ROOT/tools/tf_training_only_deployment_canary.py"
EVALUATOR="$REPO_ROOT/tools/evaluate_tf_training_only.py"
TRAINER="$REPO_ROOT/projects/latent_action_models/train.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$HELPER" "$CANARY" "$EVALUATOR" "$TRAINER" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation missing: $path"
done

STUDY_ROOT="$("$PYTHON_BIN" - "$REGISTRATION" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve(strict=True)
v = json.loads(p.read_text())
root = pathlib.Path(v["study_root"])
if p != (root / "registration.json").resolve(strict=True):
    raise SystemExit("registration root mismatch")
print(root)
PY
)"

if [[ "$MODE" == train || "$MODE" == evaluate ]]; then
  if [[ -z "$ARM" ]]; then
    case "${SLURM_ARRAY_TASK_ID:-}" in
      0) ARM="TFREG-OFF" ;;
      1) ARM="TFREG-ON" ;;
      *) die "$MODE requires arm or array index 0/1" ;;
    esac
  fi
fi
if [[ "$MODE" == canary ]]; then ARM="TFREG-ON"; fi

if [[ "$MODE" == seal ]]; then
  exec "$PYTHON_BIN" "$HELPER" seal --registration "$REGISTRATION"
fi
if [[ "$MODE" == analyze ]]; then
  exec "$PYTHON_BIN" "$HELPER" analyze --registration "$REGISTRATION"
fi
[[ "$ARM" == TFREG-OFF || "$ARM" == TFREG-ON ]] || die "invalid arm"

STAGE=train
if [[ "$MODE" == evaluate ]]; then STAGE=validation; fi
"$PYTHON_BIN" "$HELPER" verify-stage --registration "$REGISTRATION" \
  --stage "$STAGE" >/dev/null

IFS=$'\t' read -r \
  ARM_CODE CONFIG_NAME RUN_NAME RUN_IDENTITY RUN_DIR SNAPSHOT PLAN EVAL_DIR \
  PYTHON_REGISTERED VIDEOX_HOME_VALUE WAN_DIR_VALUE PARENT_SNAPSHOT \
  TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  < <("$PYTHON_BIN" "$HELPER" values --registration "$REGISTRATION" \
      --arm "$ARM" --format tsv)
[[ "$ARM_CODE" == "$ARM" && "$PYTHON_REGISTERED" == "$PYTHON_BIN" ]] || \
  die "registered runtime/arm changed"

unset PYTHONPATH
export LACWM_PYTHON="$PYTHON_BIN"
export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
source "$ACTIVATE"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-}"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
  GPU_COUNT="$(nvidia-smi -L | wc -l)"
fi
RUNTIME_DIR="$STUDY_ROOT/runtime_receipts"
mkdir -p -m 700 "$RUNTIME_DIR"
RUNTIME_JSON="$RUNTIME_DIR/${MODE}-${ARM_CODE,,}.json"
[[ ! -e "$RUNTIME_JSON" && ! -L "$RUNTIME_JSON" ]] || die "runtime receipt exists"
"$PYTHON_BIN" "$VERIFY_RUNTIME" --expected-gpus "$GPU_COUNT" --require-b200 \
  --wan-dir "$WAN_DIR" > "$RUNTIME_JSON"

export TFREG_PARENT_SNAPSHOT="$PARENT_SNAPSHOT"
export TFREG_TRAIN_MANIFEST="$TRAIN_MANIFEST"
export TFREG_TRAIN_METADATA="$TRAIN_METADATA"
export TFREG_VAL_MANIFEST="$VAL_MANIFEST"
export TFREG_VAL_METADATA="$VAL_METADATA"
export TFREG_RUN_IDENTITY="$RUN_IDENTITY"

if [[ "$MODE" == canary ]]; then
  [[ "$GPU_COUNT" == 1 ]] || die "deployment canary requires exactly one B200"
  export WANDB_MODE=disabled WANDB_DISABLED=true
  exec "$PYTHON_BIN" "$CANARY" --registration "$REGISTRATION" \
    --output "$STUDY_ROOT/deployment_canary.json"
fi

if [[ "$MODE" == evaluate ]]; then
  [[ "$GPU_COUNT" == 8 ]] || die "evaluation requires exactly eight B200s"
  [[ ! -e "$EVAL_DIR" && ! -L "$EVAL_DIR" ]] || die "evaluation output exists"
  export WANDB_MODE=disabled WANDB_DISABLED=true
  unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS
  cd "$REPO_ROOT"
  exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$EVALUATOR" --registration "$REGISTRATION" --arm "$ARM_CODE" \
    --output-dir "$EVAL_DIR"
fi

[[ "$MODE" == train ]] || die "unexpected mode"
[[ "$GPU_COUNT" == 8 ]] || die "training requires exactly eight B200s"
"$PYTHON_BIN" "$HELPER" verify-canary --registration "$REGISTRATION" >/dev/null
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || die "training output exists"
"$PYTHON_BIN" "$HELPER" plan --registration "$REGISTRATION" --arm "$ARM_CODE" >/dev/null
export TFREG_RUN_ROOT="$STUDY_ROOT/training"
export LACWM_RUN_IDENTITY_SHA256="$RUN_IDENTITY"
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
export WANDB_MODE=online
export WANDB_SILENT=true
unset WANDB_RUN_GROUP WANDB_GROUP WANDB_JOB_TYPE WANDB_NOTES
export WANDB_DIR="$STUDY_ROOT/wandb"
export WANDB_CACHE_DIR="$STUDY_ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$STUDY_ROOT/wandb-config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" \
  "+experiments_0908=$CONFIG_NAME" \
  "hydra.run.dir=$RUN_DIR" "hydra.sweep.dir=$RUN_DIR" \
  "trainer.config.saving.save_path=$SNAPSHOT"
for path in "$SNAPSHOT" "$RUN_DIR/training_complete.json" "$RUN_DIR/.hydra/config.yaml"; do
  [[ -s "$path" && ! -L "$path" ]] || die "training output missing: $path"
done
echo "TFREG arm complete: $ARM_CODE"
