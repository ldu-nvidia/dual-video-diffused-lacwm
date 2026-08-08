#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
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
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || die "TRD workflow is non-requeueable"
[[ "$MODE" == canary || "$MODE" == train || "$MODE" == seal || \
   "$MODE" == evaluate || "$MODE" == analyze ]] || die "invalid --mode"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute Git worktree"
[[ "$(git -C "$REPO_ROOT" rev-parse --is-inside-work-tree 2>/dev/null)" == true ]] || \
  die "repository is not a Git worktree"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

HELPER="$REPO_ROOT/tools/videorepa_trd_screen.py"
EVALUATOR="$REPO_ROOT/tools/evaluate_videorepa_trd.py"
DEPLOYMENT_CANARY="$REPO_ROOT/tools/videorepa_trd_deployment_canary.py"
TRAINER="$REPO_ROOT/projects/latent_action_models/train.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$HELPER" "$EVALUATOR" "$DEPLOYMENT_CANARY" "$TRAINER" \
  "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation missing: $path"
done

STUDY_ROOT="$("$PYTHON_BIN" - "$REGISTRATION" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve(strict=True)
value = json.loads(p.read_text())
root = pathlib.Path(value["study_root"])
if p != (root / "registration.json").resolve(strict=True):
    raise SystemExit("registration/study-root mismatch")
print(root)
PY
)"
RUNTIME_DIR="$STUDY_ROOT/runtime_receipts"
mkdir -p -m 700 "$RUNTIME_DIR"

unset PYTHONPATH
export LACWM_PYTHON="$PYTHON_BIN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

if [[ "$MODE" == seal ]]; then
  [[ -z "$ARM" ]] || die "seal does not accept --arm"
  "$PYTHON_BIN" "$HELPER" verify-stage --registration "$REGISTRATION" --stage train >/dev/null
  exec "$PYTHON_BIN" "$HELPER" seal --registration "$REGISTRATION"
fi
if [[ "$MODE" == analyze ]]; then
  [[ -z "$ARM" ]] || die "analysis does not accept --arm"
  exec "$PYTHON_BIN" "$HELPER" analyze --registration "$REGISTRATION"
fi

if [[ "$MODE" == train || "$MODE" == evaluate ]]; then
  if [[ -z "$ARM" ]]; then
    case "${SLURM_ARRAY_TASK_ID:-}" in
      0) ARM="TRD-OFF" ;;
      1) ARM="TRD-ON" ;;
      *) die "$MODE requires --arm or array index 0/1" ;;
    esac
  fi
  [[ "$ARM" == TRD-OFF || "$ARM" == TRD-ON ]] || die "invalid arm"
fi

if [[ "$MODE" == canary || "$MODE" == train ]]; then
  # Rehash the 4.25 GiB parent and all multi-GiB TRAIN arrays before torchrun
  # creates NCCL. Validation paths are not exported to the training process.
  "$PYTHON_BIN" "$HELPER" verify-stage --registration "$REGISTRATION" --stage train >/dev/null
else
  # The post-training seal and the val64 RGB/action inputs are rehashed before
  # NCCL. The unused clean-feature target file is deliberately never opened by
  # the target-free evaluation stage.
  "$PYTHON_BIN" "$HELPER" verify-stage --registration "$REGISTRATION" --stage validation >/dev/null
  "$PYTHON_BIN" "$HELPER" verify-arm --registration "$REGISTRATION" --arm "$ARM" >/dev/null
fi

if [[ "$MODE" == train ]]; then
  # A successful dependency is necessary but not sufficient: revalidate the
  # identity-bound report in every arm before allocating model state/NCCL.
  "$PYTHON_BIN" "$DEPLOYMENT_CANARY" verify \
    --registration "$REGISTRATION" >/dev/null
fi

if [[ "$MODE" == canary ]]; then
  ARM="TRD-ON"
fi
IFS=$'\t' read -r \
  ARM_CODE CONFIG_NAME RUN_NAME RUN_IDENTITY RUN_DIR SNAPSHOT TRACE TRACE_COMPLETE \
  EVAL_DIR PLAN PYTHON_REGISTERED WAN_DIR_VALUE VIDEOX_HOME_VALUE WARMSTART \
  TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  < <("$PYTHON_BIN" "$HELPER" values --registration "$REGISTRATION" \
      --arm "$ARM" --format tsv)
[[ "$ARM_CODE" == "$ARM" && "$PYTHON_REGISTERED" == "$PYTHON_BIN" ]] || \
  die "registered arm/runtime values changed"
[[ "$RUN_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "run identity is invalid"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
source "$ACTIVATE"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME"

# The sealed Hydra config intentionally keeps these paths as environment
# interpolations.  Restore every registered training-side value before any
# mode-specific consumer (including evaluation) loads that config.
export TRD_VPM_SNAPSHOT="$WARMSTART"
export TRD_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export TRD_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export TRD_RUN_ROOT="$STUDY_ROOT/training"
unset TRD_VALIDATION_CLIP_MANIFEST TRD_VALIDATION_CACHE_METADATA \
  VJEPA_VAL_CLIP_MANIFEST VJEPA_VAL_CACHE_METADATA VJEPA_TEST_CLIP_MANIFEST \
  VJEPA_TEST_CACHE_METADATA

RUNTIME_LABEL="${MODE}-${ARM_CODE,,}"
RUNTIME_JSON="$RUNTIME_DIR/$RUNTIME_LABEL.json"
[[ ! -e "$RUNTIME_JSON" && ! -L "$RUNTIME_JSON" ]] || \
  die "runtime receipt already exists: $RUNTIME_JSON"
"$PYTHON_BIN" "$VERIFY_RUNTIME" --expected-gpus 8 --require-b200 \
  --wan-dir "$WAN_DIR" > "$RUNTIME_JSON"
[[ -s "$RUNTIME_JSON" && ! -L "$RUNTIME_JSON" ]] || die "runtime receipt missing"

if [[ "$MODE" == evaluate ]]; then
  [[ ! -e "$EVAL_DIR" && ! -L "$EVAL_DIR" ]] || die "evaluation output is not fresh"
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS
  cd "$REPO_ROOT"
  exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$EVALUATOR" evaluate --registration "$REGISTRATION" \
    --arm "$ARM_CODE" --output-dir "$EVAL_DIR"
fi

if [[ "$MODE" == canary ]]; then
  CANARY_DIR="$STUDY_ROOT/memory_canary"
  DEPLOYMENT_CANARY_DIR="$STUDY_ROOT/deployment_sampler_canary"
  [[ ! -e "$CANARY_DIR" && ! -L "$CANARY_DIR" ]] || \
    die "memory-canary output is not fresh"
  [[ ! -e "$DEPLOYMENT_CANARY_DIR" && ! -L "$DEPLOYMENT_CANARY_DIR" ]] || \
    die "deployment-canary output is not fresh"
  CANARY_ID="$("$PYTHON_BIN" - "$RUN_IDENTITY" <<'PY'
import hashlib, sys
print(hashlib.sha256((sys.argv[1] + ':trd-memory-canary-v1').encode()).hexdigest())
PY
)"
  export TRD_RUN_ROOT="$STUDY_ROOT"
  export TRD_TRAIN_TRACE="$CANARY_DIR/trd_training_trace.jsonl"
  export LACWM_RUN_IDENTITY_SHA256="$CANARY_ID"
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
  # This is a real one-clip, NFE-1 deployment generation on GPU.  It also
  # compares the custom auxiliary-free Wan path to the ordinary VPM
  # condition-off sampler from identical keyed noise and fails on any bit.
  "$PYTHON_BIN" "$DEPLOYMENT_CANARY" run --registration "$REGISTRATION"
  "$PYTHON_BIN" "$DEPLOYMENT_CANARY" verify \
    --registration "$REGISTRATION" >/dev/null
  cd "$PROJECT_ROOT"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" \
    "+experiments_0908=$CONFIG_NAME" \
    "name=videorepa-trd-memory-canary" \
    "hydra.run.dir=$CANARY_DIR" "hydra.sweep.dir=$CANARY_DIR" \
    "trainer.config.max_iter=1" \
    "trainer.config.saving.save_path=$CANARY_DIR/snapshot.pt" \
    "trainer.config.saving.save_every=1" \
    "trainer.config.validation.val_every=1" \
    "trainer.config.validation.n_val_samples=1" \
    "viz_data_loader=[]" "wandb.enabled=false"
  for path in "$CANARY_DIR/snapshot.pt" "$CANARY_DIR/training_complete.json" \
    "$CANARY_DIR/trd_training_trace_complete.json" \
    "$DEPLOYMENT_CANARY_DIR/report.json"; do
    [[ -s "$path" && ! -L "$path" ]] || die "canary output missing: $path"
  done
  echo "TRD deployment-equivalence and production-shape memory canaries complete"
  exit 0
fi

[[ "$MODE" == train ]] || die "unexpected mode fallthrough"
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || die "training output is not fresh"
WANDB_RECEIPT="$RUNTIME_DIR/wandb-${ARM_CODE,,}.json"
[[ ! -e "$WANDB_RECEIPT" && ! -L "$WANDB_RECEIPT" ]] || \
  die "W&B privacy receipt already exists"
"$PYTHON_BIN" "$HELPER" wandb-check > "$WANDB_RECEIPT"
"$PYTHON_BIN" "$HELPER" write-plan --registration "$REGISTRATION" --arm "$ARM_CODE" >/dev/null
export TRD_TRAIN_TRACE="$TRACE"
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
  "trainer.config.saving.save_path=$SNAPSHOT" \
  "wandb.enabled=true" "wandb.mode=online" "wandb.entity=zijiandu" \
  "wandb.project=dual-video-diffusion-private" "wandb.group=null" \
  "+wandb.id=$RUN_IDENTITY" "+wandb.resume=never"
for path in "$SNAPSHOT" "$RUN_DIR/training_complete.json" "$TRACE" \
  "$TRACE_COMPLETE" "$RUN_DIR/.hydra/config.yaml"; do
  [[ -s "$path" && ! -L "$path" ]] || die "training output missing: $path"
done
echo "VideoREPA TRD arm complete: $ARM_CODE"
