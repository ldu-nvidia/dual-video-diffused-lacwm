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
#SBATCH --exclude=pool0-0081,pool0-0089
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
REPO_ROOT=""
EXPECTED_COMMIT=""
PYTHON_BIN=""

while (($#)); do
  case "$1" in
    --mode) MODE="${2:?}"; shift 2 ;;
    --arm) ARM="${2:?}"; shift 2 ;;
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$MODE" == "train" || "$MODE" == "evaluate" ]] || \
  die "--mode must be train or evaluate"
if [[ "$MODE" == "train" ]]; then
  [[ "$ARM" == "FLOW-OFF" || "$ARM" == "RAW-FLOW" ]] || \
    die "training requires --arm FLOW-OFF or RAW-FLOW"
else
  [[ -z "$ARM" ]] || die "evaluation does not accept --arm"
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || die "entrypoint requires Slurm"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "Stage-1 is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "Stage-1 requires one node"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || die "Stage-1 requires one launcher task"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -e "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute Git worktree"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

WORKFLOW="$REPO_ROOT/tools/physics_flow_stage1_workflow.py"
STAGE_TOOL="$REPO_ROOT/tools/physics_flow_stage1.py"
EVALUATOR="$REPO_ROOT/tools/physics_flow_stage1_evaluate.py"
PARENT_PARITY="$REPO_ROOT/tools/physics_flow_parent_parity.py"
PARENT_ADAPTER="$REPO_ROOT/tools/physics_flow_parent_vpm.py"
TRAINER="$REPO_ROOT/projects/latent_action_models/physics_flow_train.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$STAGE_TOOL" "$EVALUATOR" "$PARENT_PARITY" \
  "$PARENT_ADAPTER" "$TRAINER" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation missing: $path"
done

unset PYTHONPATH
export LACWM_PYTHON="$PYTHON_BIN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

REGISTRATION_SHA256="$(sha256sum "$REGISTRATION" | awk '{print $1}')"
STUDY_IDENTITY="$($PYTHON_BIN - "$REGISTRATION" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["identity_sha256"])
PY
)"
[[ "$STUDY_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "study identity is invalid"

if [[ "$MODE" == "train" ]]; then
  "$PYTHON_BIN" "$WORKFLOW" write-arm-plan \
    --registration "$REGISTRATION" --arm "$ARM" >/dev/null
  IFS=$'\t' read -r \
    ARM_CODE CONFIG_NAME RUN_NAME RUN_IDENTITY RUN_DIR TRAINING_ROOT ARM_PLAN \
    PARENT_SNAPSHOT PARENT_RESOLVED_CONFIG TRAIN_MANIFEST TRAIN_RGB_METADATA TRAIN_FLOW_METADATA \
    TRAIN_FLOW_METADATA_SHA256 TRAIN_RAW_SHA256 VAL_MANIFEST VAL_RGB_METADATA \
    VAL_FLOW_METADATA VAL_FLOW_METADATA_SHA256 VAL_RAW_SHA256 \
    RENDERER_ANALYSIS_IDENTITY REGISTERED_STUDY_IDENTITY REGISTERED_PYTHON \
    WAN_DIR_VALUE VIDEOX_HOME_VALUE \
    < <(
      "$PYTHON_BIN" "$WORKFLOW" arm-values \
        --registration "$REGISTRATION" --arm "$ARM" --format tsv
    )
  [[ "$ARM_CODE" == "$ARM" ]] || die "arm values changed"
  [[ "$REGISTERED_PYTHON" == "$PYTHON_BIN" ]] || die "runtime Python differs"
  [[ "$REGISTERED_STUDY_IDENTITY" == "$STUDY_IDENTITY" ]] || \
    die "study identity linkage differs"
  [[ "$RUN_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || die "run identity is invalid"
  [[ ! -e "$RUN_DIR" ]] || die "fresh run output exists: $RUN_DIR"
else
  IFS=$'\t' read -r \
    _ _ _ _ _ TRAINING_ROOT _ PARENT_SNAPSHOT PARENT_RESOLVED_CONFIG TRAIN_MANIFEST TRAIN_RGB_METADATA \
    TRAIN_FLOW_METADATA TRAIN_FLOW_METADATA_SHA256 TRAIN_RAW_SHA256 VAL_MANIFEST \
    VAL_RGB_METADATA VAL_FLOW_METADATA VAL_FLOW_METADATA_SHA256 VAL_RAW_SHA256 \
    RENDERER_ANALYSIS_IDENTITY REGISTERED_STUDY_IDENTITY REGISTERED_PYTHON \
    WAN_DIR_VALUE VIDEOX_HOME_VALUE \
    < <(
      "$PYTHON_BIN" "$WORKFLOW" arm-values \
        --registration "$REGISTRATION" --arm FLOW-OFF --format tsv
    )
  [[ "$REGISTERED_PYTHON" == "$PYTHON_BIN" ]] || die "runtime Python differs"
  [[ "$REGISTERED_STUDY_IDENTITY" == "$STUDY_IDENTITY" ]] || \
    die "study identity linkage differs"
fi

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
source "$ACTIVATE"
export PHYSICS_FLOW_VPM_SNAPSHOT="$PARENT_SNAPSHOT"
export PHYSICS_FLOW_VPM_SNAPSHOT_SHA256="de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
export PHYSICS_FLOW_VPM_RESOLVED_CONFIG="$PARENT_RESOLVED_CONFIG"
export PHYSICS_FLOW_VPM_RESOLVED_CONFIG_SHA256="ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
export PHYSICS_FLOW_TRAIN_MANIFEST="$TRAIN_MANIFEST"
export PHYSICS_FLOW_TRAIN_RGB_METADATA="$TRAIN_RGB_METADATA"
export PHYSICS_FLOW_TRAIN_FLOW_METADATA="$TRAIN_FLOW_METADATA"
export PHYSICS_FLOW_TRAIN_FLOW_METADATA_SHA256="$TRAIN_FLOW_METADATA_SHA256"
export PHYSICS_FLOW_TRAIN_RAW_SHA256="$TRAIN_RAW_SHA256"
export PHYSICS_FLOW_VAL_MANIFEST="$VAL_MANIFEST"
export PHYSICS_FLOW_VAL_RGB_METADATA="$VAL_RGB_METADATA"
export PHYSICS_FLOW_VAL_FLOW_METADATA="$VAL_FLOW_METADATA"
export PHYSICS_FLOW_VAL_FLOW_METADATA_SHA256="$VAL_FLOW_METADATA_SHA256"
export PHYSICS_FLOW_VAL_RAW_SHA256="$VAL_RAW_SHA256"
export PHYSICS_FLOW_RENDERER_ANALYSIS_IDENTITY_SHA256="$RENDERER_ANALYSIS_IDENTITY"
export PHYSICS_FLOW_STUDY_REGISTRATION="$REGISTRATION"
export PHYSICS_FLOW_STUDY_REGISTRATION_SHA256="$REGISTRATION_SHA256"
export PHYSICS_FLOW_STUDY_REGISTRATION_IDENTITY_SHA256="$STUDY_IDENTITY"
export PHYSICS_FLOW_RUN_ROOT="$TRAINING_ROOT"

export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
export WANDB_MODE=online
export WANDB_SILENT=true
unset WANDB_RUN_GROUP WANDB_GROUP WANDB_JOB_TYPE WANDB_NOTES
STUDY_ROOT="$(dirname "$REGISTRATION")"
export WANDB_DIR="$STUDY_ROOT/wandb"
export WANDB_CACHE_DIR="$STUDY_ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$STUDY_ROOT/wandb-config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

IPC_TMPDIR="/tmp/lacwm-physics-flow-${UID}-${SLURM_JOB_ID}-${MODE}"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC directory exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"
cleanup_ipc() { rmdir "$IPC_TMPDIR" 2>/dev/null || true; }
trap cleanup_ipc EXIT

"$PYTHON_BIN" "$VERIFY_RUNTIME" --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
export PYTHONPATH="$REPO_ROOT:$PROJECT_ROOT:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME_VALUE"

if [[ "$MODE" == "train" ]]; then
  "$PYTHON_BIN" "$WORKFLOW" wandb-check >/dev/null
  export LACWM_RUN_IDENTITY_SHA256="$RUN_IDENTITY"
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
    "+wandb.id=$RUN_NAME" \
    "+wandb.resume=never"
  for path in \
    "$RUN_DIR/snapshot.pt" \
    "$RUN_DIR/training_complete.json" \
    "$RUN_DIR/physics_flow_training_trace.jsonl" \
    "$RUN_DIR/physics_flow_training_trace_complete.json" \
    "$RUN_DIR/.hydra/config.yaml"; do
    [[ -s "$path" && ! -L "$path" ]] || die "training output missing: $path"
  done
  echo "Raw physics-flow training complete: $ARM -> $RUN_DIR"
else
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$STAGE_TOOL" compare-traces --registration "$REGISTRATION" >/dev/null
  "$PYTHON_BIN" "$PARENT_PARITY" \
    --registration "$REGISTRATION" \
    --output "$STUDY_ROOT/parent_sampler_parity.json" >/dev/null
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node=8 "$EVALUATOR" \
    --registration "$REGISTRATION" \
    --output-dir "$STUDY_ROOT/evaluation" \
    --batch-size 2
  "$PYTHON_BIN" "$STAGE_TOOL" analyze --registration "$REGISTRATION" >/dev/null
  "$PYTHON_BIN" "$STAGE_TOOL" audit-study --registration "$REGISTRATION"
  [[ -s "$STUDY_ROOT/analysis/audit.json" ]] || die "final audit is absent"
  echo "Raw physics-flow Stage-1 complete: $STUDY_ROOT/analysis/audit.json"
fi
