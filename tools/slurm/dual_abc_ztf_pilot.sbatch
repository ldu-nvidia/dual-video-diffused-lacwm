#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
HELPER="$REPO_ROOT/tools/dual_abc_pilot.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: dual_abc_ztf_pilot.sbatch REQUIRED_OPTIONS

Internal Slurm array entrypoint. Submit only through
tools/slurm/submit_dual_abc_ztf_pilot.sh.
EOF
}

PAIR_ID=""
EXPECTED_COMMIT=""
PAIR_ROOT=""
RUN_ROOT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
DATA_ROOT=""
CHECKPOINT=""
CHECKPOINT_SHA256=""
NO_ZTF_SMOKE_REPORT=""
WITH_ZTF_SMOKE_REPORT=""
WANDB_ENTITY_VALUE=""
WANDB_PROJECT_VALUE=""

while (($#)); do
  case "$1" in
    --pair-id) [[ $# -ge 2 ]] || die "--pair-id requires a value"; PAIR_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --pair-root) [[ $# -ge 2 ]] || die "--pair-root requires a value"; PAIR_ROOT="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || die "--run-root requires a value"; RUN_ROOT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; DATA_ROOT="$2"; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || die "--checkpoint requires a value"; CHECKPOINT="$2"; shift 2 ;;
    --checkpoint-sha256) [[ $# -ge 2 ]] || die "--checkpoint-sha256 requires a value"; CHECKPOINT_SHA256="$2"; shift 2 ;;
    --no-ztf-smoke-report) [[ $# -ge 2 ]] || die "--no-ztf-smoke-report requires a value"; NO_ZTF_SMOKE_REPORT="$2"; shift 2 ;;
    --with-ztf-smoke-report) [[ $# -ge 2 ]] || die "--with-ztf-smoke-report requires a value"; WITH_ZTF_SMOKE_REPORT="$2"; shift 2 ;;
    --wandb-entity) [[ $# -ge 2 ]] || die "--wandb-entity requires a value"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --wandb-project) [[ $# -ge 2 ]] || die "--wandb-project requires a value"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "pair ID:$PAIR_ID" \
  "expected commit:$EXPECTED_COMMIT" \
  "pair root:$PAIR_ROOT" \
  "run root:$RUN_ROOT" \
  "Python:$PYTHON_BIN" \
  "Wan directory:$WAN_DIR_VALUE" \
  "VideoX-Fun checkout:$VIDEOX_HOME_VALUE" \
  "data root:$DATA_ROOT" \
  "checkpoint:$CHECKPOINT" \
  "checkpoint SHA-256:$CHECKPOINT_SHA256" \
  "no-ZTF smoke report:$NO_ZTF_SMOKE_REPORT" \
  "with-ZTF smoke report:$WITH_ZTF_SMOKE_REPORT" \
  "W&B entity:$WANDB_ENTITY_VALUE" \
  "W&B project:$WANDB_PROJECT_VALUE"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires a Slurm allocation"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "SLURM_ARRAY_TASK_ID is unset"
[[ "$SLURM_ARRAY_TASK_ID" == "0" || "$SLURM_ARRAY_TASK_ID" == "1" ]] || \
  die "array task must be 0 or 1"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "pilot jobs are non-requeueable; refusing restart ${SLURM_RESTART_COUNT}"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "each arm requires exactly one node"

if command -v scontrol >/dev/null 2>&1; then
  JOB_RECORD="$(scontrol show job "$SLURM_JOB_ID" -o)"
  [[ "$JOB_RECORD" == *"Requeue=0"* ]] || die "Slurm job is unexpectedly requeueable"
fi

case "$SLURM_ARRAY_TASK_ID" in
  0)
    ARM="no-ztf-condition"
    CONDITION_ON_TF="false"
    CONFIG_SELECTOR="ravenhuang/wan-dit/dual_abc_no_ztf_condition.yaml"
    SMOKE_VARIANT="dual-no-ztf"
    SMOKE_REPORT="$NO_ZTF_SMOKE_REPORT"
    ;;
  1)
    ARM="with-ztf-condition"
    CONDITION_ON_TF="true"
    CONFIG_SELECTOR="ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml"
    SMOKE_VARIANT="dual-with-ztf"
    SMOKE_REPORT="$WITH_ZTF_SMOKE_REPORT"
    ;;
esac

RUN_ID="${PAIR_ID}-${ARM}"
RUN_DIR="$PAIR_ROOT/$ARM"
PAIR_MANIFEST="$PAIR_ROOT/pair_manifest.json"
ARM_CONFIG="$PROJECT_ROOT/configs/experiments_0908/$CONFIG_SELECTOR"
COMMON_CONFIG="$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml"
RESOLVED_CONFIG="$RUN_DIR/resolved_config.yaml"
ARM_MANIFEST="$RUN_DIR/arm_manifest.json"
OUTCOME="$RUN_DIR/outcome.json"

[[ -f "$PAIR_MANIFEST" && ! -L "$PAIR_MANIFEST" ]] || \
  die "immutable pair manifest is missing: $PAIR_MANIFEST"
[[ ! -e "$RUN_DIR" ]] || die "fresh arm run directory already exists: $RUN_DIR"
mkdir "$RUN_DIR" || die "could not exclusively create fresh run directory: $RUN_DIR"

mkdir \
  "$RUN_DIR/wandb" \
  "$RUN_DIR/wandb-data" \
  "$RUN_DIR/wandb-cache" \
  "$RUN_DIR/wandb-config" \
  "$RUN_DIR/cache" \
  "$RUN_DIR/tmp"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_DATA="$DATA_ROOT"
export LACWM_RUNS="$RUN_ROOT"
export LACWM_PYTHON="$PYTHON_BIN"
export LACWM_ALLOWED_RUN_ROOTS="$RUN_ROOT"
export LACWM_NNODES=1
export LACWM_WORLD_SIZE=8
export LACWM_GPUS_PER_NODE=8
export WANDB_MODE=online
export WANDB_ENTITY="$WANDB_ENTITY_VALUE"
export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
unset WANDB_RUN_GROUP WANDB_RUN_ID WANDB_NAME WANDB_TAGS WANDB_RESUME
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_DATA_DIR="$RUN_DIR/wandb-data"
export WANDB_CACHE_DIR="$RUN_DIR/wandb-cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb-config"
export HF_HOME="$RUN_DIR/cache/huggingface"
export TORCH_HOME="$RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export TMPDIR="$RUN_DIR/tmp"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# This validates the pinned VideoX commit and establishes the B200 NCCL profile.
source "$ACTIVATE"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit changed: $ACTUAL_COMMIT != $EXPECTED_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository is dirty: ${GIT_STATUS//$'\n'/; }"

"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE" > "$RUN_DIR/runtime_verification.json"

"$PYTHON_BIN" "$HELPER" wandb-private \
  --entity "$WANDB_ENTITY_VALUE" \
  --project "$WANDB_PROJECT_VALUE" > "$RUN_DIR/wandb_private_preflight.json"

"$PYTHON_BIN" "$REPO_ROOT/tools/validate_smoke_report.py" \
  --report "$SMOKE_REPORT" \
  --variant "$SMOKE_VARIANT" \
  --git-commit "$EXPECTED_COMMIT" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --data-root "$DATA_ROOT" \
  --warmstart-model "$CHECKPOINT" \
  --warmstart-sha256 "$CHECKPOINT_SHA256" > "$RUN_DIR/gradient_smoke_preflight.txt"

TRAIN_OVERRIDES=(
  "+experiments_0908=$CONFIG_SELECTOR"
  "name=$RUN_ID"
  "seed=1234"
  "data_loader.batch_size=1"
  "trainer.config.max_iter=100"
  "trainer.config.gradient_accumulation_steps=1"
  "trainer.config.load_path=$CHECKPOINT"
  "lr_scheduler_factory.lr_lambda.warmup_steps=20"
  "lr_scheduler_factory.lr_lambda.total_steps=100"
  "trainer.config.logging.log_every=5"
  "trainer.config.saving.save_every=50"
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt"
  "trainer.config.validation.val_every=10"
  "trainer.config.visualization.viz_every=50"
  "trainer.config.visualization.viz_path=$RUN_DIR/visualization"
  "trainer.config.visualization.require_success=true"
  "model.dual_diffusion.condition_on_tf=$CONDITION_ON_TF"
  "hydra.run.dir=$RUN_DIR"
  "hydra.sweep.dir=$RUN_DIR"
  "wandb.enabled=true"
  "wandb.mode=online"
  "wandb.entity=$WANDB_ENTITY_VALUE"
  "wandb.project=$WANDB_PROJECT_VALUE"
  "+wandb.id=$RUN_ID"
  "+wandb.group=null"
  "+wandb.resume=never"
)

PREPARE_ARGS=(
  "$PYTHON_BIN" "$HELPER" prepare-arm
  --arm "$ARM"
  --condition-on-tf "$CONDITION_ON_TF"
  --pair-id "$PAIR_ID"
  --git-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --project-root "$PROJECT_ROOT"
  --pair-root "$PAIR_ROOT"
  --run-dir "$RUN_DIR"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --data-root "$DATA_ROOT"
  --checkpoint "$CHECKPOINT"
  --checkpoint-sha256 "$CHECKPOINT_SHA256"
  --config-selector "$CONFIG_SELECTOR"
  --arm-config "$ARM_CONFIG"
  --common-config "$COMMON_CONFIG"
  --wandb-entity "$WANDB_ENTITY_VALUE"
  --wandb-project "$WANDB_PROJECT_VALUE"
  --wandb-run-id "$RUN_ID"
  --pair-manifest "$PAIR_MANIFEST"
  --smoke-report "$SMOKE_REPORT"
  --slurm-job-id "$SLURM_JOB_ID"
  --slurm-array-job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
  --slurm-array-task-id "$SLURM_ARRAY_TASK_ID"
  --resolved-config-output "$RESOLVED_CONFIG"
  --manifest-output "$ARM_MANIFEST"
)
for override in "${TRAIN_OVERRIDES[@]}"; do
  PREPARE_ARGS+=(--override "$override")
done
ARM_IDENTITY_SHA256="$("${PREPARE_ARGS[@]}")"
[[ "$ARM_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "arm helper returned an invalid identity SHA-256"
export LACWM_RUN_IDENTITY_SHA256="$ARM_IDENTITY_SHA256"

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node=8
  train.py
  "${TRAIN_OVERRIDES[@]}"
)

printf 'Launching matched-pair arm %q from %q:' "$ARM" "$PROJECT_ROOT"
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Pair: $PAIR_ID"
echo "Arm: $ARM (condition_on_tf=$CONDITION_ON_TF)"
echo "Clock: sigma=1 noise, sigma=0 clean"
echo "Warm start: $CHECKPOINT ($CHECKPOINT_SHA256), model weights only; fresh optimizer"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE, run=$RUN_ID, group=null"

set +e
(
  cd "$PROJECT_ROOT"
  "${COMMAND[@]}"
)
TRAIN_STATUS=$?
set -e

"$PYTHON_BIN" "$HELPER" record-outcome \
  --manifest "$ARM_MANIFEST" \
  --snapshot "$RUN_DIR/snapshot.pt" \
  --exit-status "$TRAIN_STATUS" \
  --output "$OUTCOME"
exit "$TRAIN_STATUS"
