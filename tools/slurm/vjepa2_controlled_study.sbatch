#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: vjepa2_controlled_study.sbatch REQUIRED_OPTIONS

Internal, non-requeueable stage-array entrypoint. Submit only through
tools/slurm/submit_vjepa2_controlled_study.sh.

LACWM clock convention: sigma=1 noise, sigma=0 clean.
EOF
}

STUDY_ID=""
STAGE_ENDPOINT=""
EXPECTED_COMMIT=""
REPO_ROOT=""
PROJECT_ROOT=""
STUDY_ROOT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
BASELINE_CONFIG=""
BASELINE_SELECTOR=""
DUAL_CONFIG=""
DUAL_SELECTOR=""
WARMSTART=""
TRAIN_MANIFEST=""
TRAIN_CACHE_METADATA=""
VALIDATION_MANIFEST=""
VALIDATION_CACHE_METADATA=""
TEST_MANIFEST=""
TEST_CACHE_METADATA=""
WANDB_ENTITY_VALUE=""
WANDB_PROJECT_VALUE=""

while (($#)); do
  case "$1" in
    --study-id) [[ $# -ge 2 ]] || die "--study-id requires a value"; STUDY_ID="$2"; shift 2 ;;
    --stage-endpoint) [[ $# -ge 2 ]] || die "--stage-endpoint requires a value"; STAGE_ENDPOINT="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "--repo-root requires a value"; REPO_ROOT="$2"; shift 2 ;;
    --project-root) [[ $# -ge 2 ]] || die "--project-root requires a value"; PROJECT_ROOT="$2"; shift 2 ;;
    --study-root) [[ $# -ge 2 ]] || die "--study-root requires a value"; STUDY_ROOT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --baseline-config) [[ $# -ge 2 ]] || die "--baseline-config requires a value"; BASELINE_CONFIG="$2"; shift 2 ;;
    --baseline-selector) [[ $# -ge 2 ]] || die "--baseline-selector requires a value"; BASELINE_SELECTOR="$2"; shift 2 ;;
    --dual-config) [[ $# -ge 2 ]] || die "--dual-config requires a value"; DUAL_CONFIG="$2"; shift 2 ;;
    --dual-selector) [[ $# -ge 2 ]] || die "--dual-selector requires a value"; DUAL_SELECTOR="$2"; shift 2 ;;
    --warmstart) [[ $# -ge 2 ]] || die "--warmstart requires a value"; WARMSTART="$2"; shift 2 ;;
    --train-manifest) [[ $# -ge 2 ]] || die "--train-manifest requires a value"; TRAIN_MANIFEST="$2"; shift 2 ;;
    --train-cache-metadata) [[ $# -ge 2 ]] || die "--train-cache-metadata requires a value"; TRAIN_CACHE_METADATA="$2"; shift 2 ;;
    --validation-manifest) [[ $# -ge 2 ]] || die "--validation-manifest requires a value"; VALIDATION_MANIFEST="$2"; shift 2 ;;
    --validation-cache-metadata) [[ $# -ge 2 ]] || die "--validation-cache-metadata requires a value"; VALIDATION_CACHE_METADATA="$2"; shift 2 ;;
    --test-manifest) [[ $# -ge 2 ]] || die "--test-manifest requires a value"; TEST_MANIFEST="$2"; shift 2 ;;
    --test-cache-metadata) [[ $# -ge 2 ]] || die "--test-cache-metadata requires a value"; TEST_CACHE_METADATA="$2"; shift 2 ;;
    --wandb-entity) [[ $# -ge 2 ]] || die "--wandb-entity requires a value"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --wandb-project) [[ $# -ge 2 ]] || die "--wandb-project requires a value"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "study ID:$STUDY_ID" \
  "stage endpoint:$STAGE_ENDPOINT" \
  "expected commit:$EXPECTED_COMMIT" \
  "repository root:$REPO_ROOT" \
  "project root:$PROJECT_ROOT" \
  "study root:$STUDY_ROOT" \
  "Python:$PYTHON_BIN" \
  "Wan directory:$WAN_DIR_VALUE" \
  "VideoX-Fun checkout:$VIDEOX_HOME_VALUE" \
  "baseline config:$BASELINE_CONFIG" \
  "baseline selector:$BASELINE_SELECTOR" \
  "dual config:$DUAL_CONFIG" \
  "dual selector:$DUAL_SELECTOR" \
  "warm start:$WARMSTART" \
  "train manifest:$TRAIN_MANIFEST" \
  "train cache metadata:$TRAIN_CACHE_METADATA" \
  "validation manifest:$VALIDATION_MANIFEST" \
  "validation cache metadata:$VALIDATION_CACHE_METADATA" \
  "test manifest:$TEST_MANIFEST" \
  "test cache metadata:$TEST_CACHE_METADATA" \
  "W&B entity:$WANDB_ENTITY_VALUE" \
  "W&B project:$WANDB_PROJECT_VALUE"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

[[ "$STAGE_ENDPOINT" =~ ^(1|50|100|200|400|600|800|1000)$ ]] || \
  die "stage endpoint is not one of 1,50,100,200,400,600,800,1000"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "this entrypoint requires a Slurm array allocation"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-4]$ ]] || die "array task must be 0 through 4"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "V-JEPA study stages are non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "each arm requires one node"

[[ "$REPO_ROOT" == /* && "$PROJECT_ROOT" == /* && "$STUDY_ROOT" == /* ]] || \
  die "repository, project, and study roots must be absolute"
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || die "invalid repository root"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository root is not canonical"

HELPER="$REPO_ROOT/tools/vjepa2_controlled_study.py"
QUALITY_EVALUATOR="$REPO_ROOT/tools/evaluate_vjepa2_quality.py"
LATENCY_BENCHMARK="$REPO_ROOT/tools/benchmark_vjepa2_inference.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
STUDY_MANIFEST="$STUDY_ROOT/study_manifest.json"
for path in \
  "$HELPER" "$QUALITY_EVALUATOR" "$LATENCY_BENCHMARK" \
  "$ACTIVATE" "$VERIFY_RUNTIME" "$STUDY_MANIFEST" \
  "$BASELINE_CONFIG" "$DUAL_CONFIG" "$WARMSTART" \
  "$TRAIN_MANIFEST" "$TRAIN_CACHE_METADATA" \
  "$VALIDATION_MANIFEST" "$VALIDATION_CACHE_METADATA" \
  "$TEST_MANIFEST" "$TEST_CACHE_METADATA"; do
  [[ -f "$path" && ! -L "$path" ]] || die "missing or symlinked required file: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"

# This runs for every allocation before creating an arm/stage/runtime artifact,
# including evidence-only resumption paths that intentionally skip preparation.
"$PYTHON_BIN" "$HELPER" validate-study-inputs \
  --study-manifest "$STUDY_MANIFEST" \
  --expected-commit "$EXPECTED_COMMIT" \
  --repo-root "$REPO_ROOT"

# Bash collapses adjacent tab characters because tab is IFS whitespace. V0 has
# three intentionally empty nullable fields, so use a non-whitespace delimiter
# and preserve their exact column positions.
IFS='|' read -r \
  ARM_CODE ARM_NAME SELECTOR_KIND DUAL_ENABLED CONDITION_MODE \
  CONDITION_ON_STATE CONDITION_ON_CLOCK SCHEDULE_MODE AUXILIARY_LEAD_LOGIT \
  AUXILIARY_LOSS_WEIGHT PARAMETER_MATCHED_CONTROL \
  < <(
    "$PYTHON_BIN" "$HELPER" arm-contract \
      --array-task-id "$SLURM_ARRAY_TASK_ID" \
      --format pipe
  )
[[ -n "$ARM_CODE" && -n "$ARM_NAME" && -n "$SELECTOR_KIND" ]] || \
  die "arm helper returned an incomplete contract"
for boolean_value in \
  "$DUAL_ENABLED" "$CONDITION_ON_STATE" "$CONDITION_ON_CLOCK" \
  "$PARAMETER_MATCHED_CONTROL"; do
  [[ "$boolean_value" == "true" || "$boolean_value" == "false" ]] || \
    die "arm helper returned an invalid Boolean: $boolean_value"
done
case "$SELECTOR_KIND" in
  baseline)
    [[ "$DUAL_ENABLED" == "false" ]] || die "baseline unexpectedly enables dual diffusion"
    CONFIG_FILE="$BASELINE_CONFIG"
    CONFIG_SELECTOR="$BASELINE_SELECTOR"
    ;;
  dual)
    [[ "$DUAL_ENABLED" == "true" ]] || die "dual arm unexpectedly disables dual diffusion"
    CONFIG_FILE="$DUAL_CONFIG"
    CONFIG_SELECTOR="$DUAL_SELECTOR"
    ;;
  *) die "unknown selector kind: $SELECTOR_KIND" ;;
esac

RUN_ID="${STUDY_ID}-${ARM_CODE}"
RUN_DIR="$STUDY_ROOT/$ARM_NAME"
ARM_MANIFEST="$RUN_DIR/arm_manifest.json"
RESOLVED_CONFIG="$RUN_DIR/resolved_update_$(printf '%04d' "$STAGE_ENDPOINT").yaml"
STAGE_MANIFEST="$RUN_DIR/stage_manifest_update_$(printf '%04d' "$STAGE_ENDPOINT").json"
STAGE_OUTCOME="$RUN_DIR/stage_outcome_update_$(printf '%04d' "$STAGE_ENDPOINT").json"
EVIDENCE_ONLY_RESUME=0
if [[ -f "$STAGE_OUTCOME" && ! -L "$STAGE_OUTCOME" ]]; then
  EVIDENCE_ONLY_RESUME=1
  for completed_path in \
    "$ARM_MANIFEST" "$RESOLVED_CONFIG" "$STAGE_MANIFEST" \
    "$RUN_DIR/snapshot.pt" "$RUN_DIR/training_complete.json"; do
    [[ -f "$completed_path" && ! -L "$completed_path" ]] || \
      die "evidence-only resume lacks completed stage input: $completed_path"
  done
fi

if [[ "$STAGE_ENDPOINT" == "1" ]]; then
  if ((EVIDENCE_ONLY_RESUME == 0)); then
    [[ ! -e "$RUN_DIR" ]] || die "fresh arm directory already exists: $RUN_DIR"
    mkdir "$RUN_DIR"
    mkdir \
      "$RUN_DIR/wandb" \
      "$RUN_DIR/wandb-data" \
      "$RUN_DIR/wandb-cache" \
      "$RUN_DIR/wandb-config" \
      "$RUN_DIR/cache"
    "$PYTHON_BIN" "$HELPER" create-arm \
      --array-task-id "$SLURM_ARRAY_TASK_ID" \
      --study-id "$STUDY_ID" \
      --run-id "$RUN_ID" \
      --git-commit "$EXPECTED_COMMIT" \
      --repo-root "$REPO_ROOT" \
      --study-root "$STUDY_ROOT" \
      --run-dir "$RUN_DIR" \
      --study-manifest "$STUDY_MANIFEST" \
      --slurm-array-job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" \
      --output "$ARM_MANIFEST"
  fi
else
  [[ -d "$RUN_DIR" && ! -L "$RUN_DIR" ]] || \
    die "prior stage did not create the arm directory: $RUN_DIR"
  [[ -f "$ARM_MANIFEST" && ! -L "$ARM_MANIFEST" ]] || \
    die "prior stage did not create the arm manifest"
  [[ -f "$RUN_DIR/snapshot.pt" && ! -L "$RUN_DIR/snapshot.pt" ]] || \
    die "prior stage snapshot is unavailable"
fi
if ((EVIDENCE_ONLY_RESUME == 0)); then
  [[ ! -e "$RESOLVED_CONFIG" && ! -e "$STAGE_MANIFEST" && ! -e "$STAGE_OUTCOME" ]] || \
    die "fresh stage provenance path already exists"
fi

# Unix-domain data-loader sockets must fit sockaddr_un. Durable outputs remain
# under the arm directory, while ephemeral IPC is node-local.
IPC_TMPDIR="/tmp/lacwm-vjepa-${UID}-${SLURM_JOB_ID}"
[[ ${#IPC_TMPDIR} -le 64 ]] || die "IPC TMPDIR is unexpectedly long"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC TMPDIR already exists"
(umask 077 && mkdir "$IPC_TMPDIR")
cleanup_ipc_tmpdir() {
  rmdir "$IPC_TMPDIR" 2>/dev/null || true
}
trap cleanup_ipc_tmpdir EXIT

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_NNODES=1
export LACWM_WORLD_SIZE=8
export LACWM_GPUS_PER_NODE=8
export WANDB_MODE=online
export WANDB_ENTITY="$WANDB_ENTITY_VALUE"
export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
unset WANDB_RUN_GROUP WANDB_RUN_ID WANDB_NAME WANDB_TAGS
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_DATA_DIR="$RUN_DIR/wandb-data"
export WANDB_CACHE_DIR="$RUN_DIR/wandb-cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb-config"
export HF_HOME="$RUN_DIR/cache/huggingface"
export TORCH_HOME="$RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export TMPDIR="$IPC_TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

source "$ACTIVATE"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit changed: $ACTUAL_COMMIT != $EXPECTED_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository is dirty: ${GIT_STATUS//$'\n'/; }"

if ((EVIDENCE_ONLY_RESUME == 0)); then
  "$PYTHON_BIN" "$VERIFY_RUNTIME" \
    --expected-gpus 8 \
    --require-b200 \
    --wan-dir "$WAN_DIR_VALUE" > "$RUN_DIR/runtime_update_$(printf '%04d' "$STAGE_ENDPOINT").json"

  "$PYTHON_BIN" "$HELPER" wandb-private \
    --entity "$WANDB_ENTITY_VALUE" \
    --project "$WANDB_PROJECT_VALUE" > "$RUN_DIR/wandb_private_update_$(printf '%04d' "$STAGE_ENDPOINT").json"
fi

if [[ "$STAGE_ENDPOINT" == "1" ]]; then
  LOAD_PATH_OVERRIDE="trainer.config.load_path=$WARMSTART"
else
  # Trainer loads the live snapshot exactly when it exists. Supplying the
  # original model-only warm start again would overwrite resumed model weights.
  LOAD_PATH_OVERRIDE="trainer.config.load_path=null"
fi

TRAIN_OVERRIDES=(
  "+experiments_0908=$CONFIG_SELECTOR"
  "name=$RUN_ID"
  "seed=1234"
  "dataset.img_augment=false"
  "val_dataset.img_augment=false"
  "viz_dataset.img_augment=false"
  "dataset.datasets.ABC.clip_manifest=$TRAIN_MANIFEST"
  "dataset.datasets.ABC.cache_metadata=$TRAIN_CACHE_METADATA"
  "val_dataset.datasets.ABC.clip_manifest=$VALIDATION_MANIFEST"
  "val_dataset.datasets.ABC.cache_metadata=$VALIDATION_CACHE_METADATA"
  "viz_dataset.datasets.ABC.clip_manifest=$TEST_MANIFEST"
  "viz_dataset.datasets.ABC.cache_metadata=$TEST_CACHE_METADATA"
  "data_loader.batch_size=1"
  "trainer.config.max_iter=$STAGE_ENDPOINT"
  "trainer.config.gradient_accumulation_steps=1"
  "trainer.config.throughput_warmup_steps=50"
  "$LOAD_PATH_OVERRIDE"
  "lr_scheduler_factory.lr_lambda.warmup_steps=50"
  "lr_scheduler_factory.lr_lambda.total_steps=1000"
  "trainer.config.logging.log_every=5"
  "trainer.config.saving.save_every=1000"
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt"
  "trainer.config.validation.val_every=1000"
  "trainer.config.visualization.viz_every=1000"
  "trainer.config.visualization.viz_path=$RUN_DIR/visualization"
  "hydra.run.dir=$RUN_DIR"
  "hydra.sweep.dir=$RUN_DIR"
  "wandb.enabled=true"
  "wandb.mode=online"
  "wandb.entity=$WANDB_ENTITY_VALUE"
  "wandb.project=$WANDB_PROJECT_VALUE"
  "wandb.tags=[abc,dual-video-diffusion,vjepa2-controlled-study,$ARM_CODE,seed-1234]"
  "+wandb.id=$RUN_ID"
  "+wandb.group=null"
  "+wandb.resume=allow"
)

if [[ "$DUAL_ENABLED" == "true" ]]; then
  TRAIN_OVERRIDES+=(
    "trainer.config.visualization.require_success=true"
    "model.time_frequency_transform=null"
    "model.dual_diffusion.enabled=true"
    "model.dual_diffusion.tf_channels=64"
    "model.dual_diffusion.auxiliary_history_mode=diffuse_all"
    "model.dual_diffusion.condition_mode='$CONDITION_MODE'"
    "model.dual_diffusion.condition_on_tf=$CONDITION_ON_STATE"
    "model.dual_diffusion.condition_on_tf_clock=$CONDITION_ON_CLOCK"
    "model.dual_diffusion.schedule_mode=$SCHEDULE_MODE"
    "model.dual_diffusion.tf_lead_logit=$AUXILIARY_LEAD_LOGIT"
    "model.dual_diffusion.tf_loss_weight=$AUXILIARY_LOSS_WEIGHT"
    "model.dual_diffusion.state_gate_init=0.0"
    "model.dual_diffusion.state_gate_trainable=true"
    "model.dual_diffusion.clock_gate_init=0.0"
    "model.dual_diffusion.clock_gate_trainable=true"
    "model.dual_diffusion.head_condition_on_tf_clock=true"
    "model.dual_diffusion.video_only_control=false"
    "model.dual_diffusion.parameter_matched_control=$PARAMETER_MATCHED_CONTROL"
    "model.dual_diffusion.capture_latent_trajectories=true"
    "model.dual_diffusion.evaluation_nfe_steps=[1,2,4,6,8,12,20]"
    "model.dual_diffusion.evaluation_noise_seed=20260729"
    "model.dual_diffusion.evaluation_condition_sources=[autonomous,off,autonomous_shuffled,oracle_matched,oracle_shuffled]"
  )
else
  # V0's built-in visualization is an NFE-8 architecture sanity check. The
  # immutable study contract still requires posthoc paired 1/2/4/6/8/12/20-NFE
  # evaluation before any final inference claim.
  TRAIN_OVERRIDES+=(
    "trainer.config.visualization.require_success=false"
    "+model.viz_num_steps=8"
  )
fi

PREPARE=(
  "$PYTHON_BIN" "$HELPER" prepare-stage
  --array-task-id "$SLURM_ARRAY_TASK_ID"
  --stage-endpoint "$STAGE_ENDPOINT"
  --git-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --project-root "$PROJECT_ROOT"
  --python "$PYTHON_BIN"
  --run-dir "$RUN_DIR"
  --arm-manifest "$ARM_MANIFEST"
  --study-manifest "$STUDY_MANIFEST"
  --config-selector "$CONFIG_SELECTOR"
  --config-file "$CONFIG_FILE"
  --resolved-config-output "$RESOLVED_CONFIG"
  --stage-manifest-output "$STAGE_MANIFEST"
)
for override in "${TRAIN_OVERRIDES[@]}"; do
  PREPARE+=(--override "$override")
done
if ((EVIDENCE_ONLY_RESUME == 0)); then
  STAGE_IDENTITY_SHA256="$("${PREPARE[@]}")"
else
  STAGE_IDENTITY_SHA256="$(
    "$PYTHON_BIN" - "$STAGE_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["identity_sha256"])
PY
  )"
fi
[[ "$STAGE_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "prepare-stage returned an invalid identity"

ARM_IDENTITY_SHA256="$(
  "$PYTHON_BIN" - "$ARM_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["identity_sha256"])
PY
)"
[[ "$ARM_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "arm manifest contains an invalid identity"
export LACWM_RUN_IDENTITY_SHA256="$ARM_IDENTITY_SHA256"

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node=8
  train.py
  "${TRAIN_OVERRIDES[@]}"
)

echo "V-JEPA study: $STUDY_ID"
echo "Arm: $ARM_CODE / $ARM_NAME"
echo "Stage endpoint: $STAGE_ENDPOINT completed optimizer updates"
echo "Primary milestones: [1,50,100,200,400,800,1000]"
echo "Inference NFE: [1,2,4,6,8,12,20]"
echo "Clock: sigma=1 noise, sigma=0 clean"
echo "Teacher calls in training/inference: 0"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE, group=null, run=$RUN_ID"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if ((EVIDENCE_ONLY_RESUME == 0)); then
  STARTED_AT_EPOCH="$(date +%s)"
  set +e
  (
    cd "$PROJECT_ROOT"
    "${COMMAND[@]}"
  )
  TRAIN_STATUS=$?
  set -e
  FINISHED_AT_EPOCH="$(date +%s)"
  ((TRAIN_STATUS == 0)) || exit "$TRAIN_STATUS"

  "$PYTHON_BIN" "$HELPER" record-stage \
    --array-task-id "$SLURM_ARRAY_TASK_ID" \
    --stage-endpoint "$STAGE_ENDPOINT" \
    --git-commit "$EXPECTED_COMMIT" \
    --run-dir "$RUN_DIR" \
    --arm-manifest "$ARM_MANIFEST" \
    --stage-manifest "$STAGE_MANIFEST" \
    --started-at-epoch "$STARTED_AT_EPOCH" \
    --finished-at-epoch "$FINISHED_AT_EPOCH" \
    --output "$STAGE_OUTCOME"
else
  echo "Evidence-only resume: preserving completed training and stage provenance."
fi

# The live snapshot is overwritten by the next allocation stage, so scientific
# fixed-test quality evidence must be completed and bound to this stage outcome
# before this job can satisfy the next stage's afterok dependency. Trainer
# visualization remains diagnostic only.
if [[ "$DUAL_ENABLED" == "true" && "$STAGE_ENDPOINT" =~ ^(1|50|100|200|400|800|1000)$ ]]; then
  QUALITY_OUTPUT_DIR="$RUN_DIR/quality/update_$(printf '%04d' "$STAGE_ENDPOINT")"
  if [[ -f "$QUALITY_OUTPUT_DIR/inventory.json" && ! -L "$QUALITY_OUTPUT_DIR/inventory.json" ]]; then
    echo "Evidence-only resume: preserving complete quality inventory."
  else
    [[ ! -e "$QUALITY_OUTPUT_DIR" ]] || \
      die "partial quality directory requires read-only diagnosis before retry: $QUALITY_OUTPUT_DIR"
    QUALITY_COMMAND=(
      "$PYTHON_BIN" -m torch.distributed.run
      --standalone
      --nproc_per_node=8
      "$QUALITY_EVALUATOR"
      --repo-root "$REPO_ROOT"
      --expected-commit "$EXPECTED_COMMIT"
      --run-dir "$RUN_DIR"
      --arm-manifest "$ARM_MANIFEST"
      --study-manifest "$STUDY_MANIFEST"
      --stage-manifest "$STAGE_MANIFEST"
      --stage-outcome "$STAGE_OUTCOME"
      --resolved-config "$RESOLVED_CONFIG"
      --snapshot "$RUN_DIR/snapshot.pt"
      --completed-updates "$STAGE_ENDPOINT"
      --batch-size-per-rank 2
      --output-dir "$QUALITY_OUTPUT_DIR"
    )
    printf 'Quality evaluation command:'
    printf ' %q' "${QUALITY_COMMAND[@]}"
    printf '\n'
    (
      cd "$PROJECT_ROOT"
      "${QUALITY_COMMAND[@]}"
    )
  fi
fi

if [[ "$DUAL_ENABLED" == "true" && "$STAGE_ENDPOINT" == "1000" ]]; then
  LATENCY_DIR="$RUN_DIR/latency"
  if [[ ! -e "$LATENCY_DIR" ]]; then
    mkdir "$LATENCY_DIR"
  else
    ((EVIDENCE_ONLY_RESUME == 1)) || \
      die "fresh latency directory already exists"
    [[ -d "$LATENCY_DIR" && ! -L "$LATENCY_DIR" ]] || \
      die "latency evidence path is not a regular directory"
  fi
  LATENCY_SOURCES=(autonomous off)
  LATENCY_NFES=(1 2 4 6 8 12 20)
  LATENCY_PIDS=()
  # Fourteen independent batch-one benchmarks are sharded round-robin across
  # eight B200s. Each GPU runs at most two benchmarks sequentially, avoiding
  # concurrent model residency on one device.
  for gpu in 0 1 2 3 4 5 6 7; do
    (
      task_index="$gpu"
      while ((task_index < 14)); do
        source_index=$((task_index / 7))
        nfe_index=$((task_index % 7))
        source_name="${LATENCY_SOURCES[$source_index]}"
        nfe_value="${LATENCY_NFES[$nfe_index]}"
        output_path="$LATENCY_DIR/source_${source_name}_nfe_${nfe_value}.json"
        if [[ -f "$output_path" && ! -L "$output_path" ]]; then
          echo "Evidence-only resume: preserving $output_path"
        else
          [[ ! -e "$output_path" ]] || \
            die "invalid existing latency evidence path: $output_path"
          CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 \
            "$PYTHON_BIN" "$LATENCY_BENCHMARK" \
              --repo-root "$REPO_ROOT" \
              --expected-commit "$EXPECTED_COMMIT" \
              --run-dir "$RUN_DIR" \
              --arm-manifest "$ARM_MANIFEST" \
              --study-manifest "$STUDY_MANIFEST" \
              --stage-manifest "$STAGE_MANIFEST" \
              --stage-outcome "$STAGE_OUTCOME" \
              --resolved-config "$RESOLVED_CONFIG" \
              --snapshot "$RUN_DIR/snapshot.pt" \
              --source "$source_name" \
              --nfe "$nfe_value" \
              --sample-index 0 \
              --device cuda:0 \
              --output "$output_path"
        fi
        task_index=$((task_index + 8))
      done
    ) &
    LATENCY_PIDS+=("$!")
  done
  LATENCY_STATUS=0
  for latency_pid in "${LATENCY_PIDS[@]}"; do
    wait "$latency_pid" || LATENCY_STATUS=$?
  done
  ((LATENCY_STATUS == 0)) || \
    die "one or more final deployable latency benchmarks failed"
fi

if [[ "$STAGE_ENDPOINT" == "1000" ]]; then
  if [[ -f "$RUN_DIR/outcome.json" && ! -L "$RUN_DIR/outcome.json" ]]; then
    echo "Evidence-only resume: preserving existing final arm outcome."
  else
    [[ ! -e "$RUN_DIR/outcome.json" ]] || \
      die "invalid existing final arm outcome path"
    "$PYTHON_BIN" "$HELPER" record-outcome \
      --array-task-id "$SLURM_ARRAY_TASK_ID" \
      --git-commit "$EXPECTED_COMMIT" \
      --run-dir "$RUN_DIR" \
      --arm-manifest "$ARM_MANIFEST" \
      --output "$RUN_DIR/outcome.json"
  fi
fi
