#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/train_8xb200.sbatch"
STATE_VALIDATOR="$REPO_ROOT/tools/slurm/validate_state.py"
RUN_ROOT_POLICY="$REPO_ROOT/tools/run_root_policy.py"
MAX_AUTOMATIC_REQUEUES=100

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_8xb200.sh [options]

Build and preview a 1-32 node, 8xB200-per-node Slurm submission. This command is
a dry run by default and does not require Slurm client commands until --execute.

Training inputs:
  --python PATH                   Provisioned lacwm Python executable
  --wan-dir PATH                  Wan2.1-Fun-1.3B-Control assets
  --videox-home PATH              Pinned VideoX-Fun checkout
  --data-root PATH                Prepared four-dataset root
  --run-root PATH                 Shared, persistent allowed run root
  --run-name NAME                 Stable run name
  --dataset-stage NAME            Immutable dataset stage label (default: all-four)
  --datasets CSV                  Exact dataset subset (default: all four)
  --smoke-report PATH             Passing real-data gradient report
  --data-validation-report PATH   Passing strict or explicitly files-only data report
  --data-validation-policy NAME   strict (default) or files_only_user_waived_v1
  --fast-training-authorization PATH
                                  Required explicit authorization for the fast policy
  --mixed-loader-report PATH      Required commit/data-bound loader evidence for fast policy
  --transition-handoff PATH       Optional immutable parent-stage handoff JSON.
                                  Its byte digest and parent checkpoint identity
                                  are bound into the child run identity.
  --wandb-mode MODE               online, offline, or disabled
  --wandb-project NAME            Required for online/offline W&B
  --wandb-entity NAME             Required for online W&B
  --wandb-run-id ID               Stable W&B identity (default: run name)
  --variant NAME                  latent (default) or explicit
  --batch-size N                  Physical per-GPU microbatch (default: latent=4,
                                  explicit=2)
  --gradient-accumulation-steps N Microbatches per update (default: latent=4,
                                  explicit=1)
  --min-gpu-memory-mib N          Minimum memory on each B200 (default: 78000)
  --max-iter N                    Optimizer updates (default: 60000)
  --warmup-steps N                LR warmup updates (default: 2000)
  --log-every N                   Metric logging cadence (default: 50)
  --save-every N                  Checkpoint cadence (default: 1000)
  --val-every N                   Validation cadence (default: 1000)
  --viz-every N                   Visualization cadence (default: 1000)

Slurm resources:
  --nodes N                       Number of 8xB200 nodes (default: 1; maximum: 32)
  --master-port PORT              Base c10d rendezvous port (default: 29400)
  --partition NAME                Optional partition
  --account NAME                  Optional account
  --qos NAME                      Optional QOS
  --constraint VALUE              Optional node constraint, e.g. b200
  --time VALUE                    Slot wall time (default: 24:00:00)
  --cpus N                        CPUs assigned to each node agent (default: 160)
  --mem VALUE                     Node memory request (default: 1000G)
  --signal-seconds N              USR1 lead time before slot end (default: 1200)
  --max-requeues N                Maximum self-requeues (default: 12)
  --job-name NAME                 Slurm name (default: lacwm-RUN_NAME)

Actions:
  --execute                       Create the log directory and call sbatch
  -h, --help                      Show this help

Defaults reproduce the checked-in batching profile. Explicit batching and GPU-memory
arguments are propagated into the immutable run identity on every allocation.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  [[ $# -ge 2 ]] || die "$1 requires a value"
}

require_absolute() {
  local name="$1"
  local value="$2"
  [[ "$value" == /* ]] || die "$name must be an absolute path: $value"
}

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

VARIANT="latent"
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
DATA_ROOT=""
RUN_ROOT=""
RUN_NAME=""
DATASET_STAGE="all-four"
DATASETS_CSV="droid,egodex,agibot,abc"
SMOKE_REPORT=""
DATA_VALIDATION_REPORT=""
DATA_VALIDATION_POLICY="strict"
FAST_TRAINING_AUTHORIZATION=""
MIXED_LOADER_REPORT=""
TRANSITION_HANDOFF=""
WANDB_MODE_VALUE=""
WANDB_PROJECT_VALUE=""
WANDB_ENTITY_VALUE=""
WANDB_RUN_ID=""
BATCH_SIZE=""
GRADIENT_ACCUMULATION_STEPS=""
MIN_GPU_MEMORY_MIB="78000"
MAX_ITER="60000"
WARMUP_STEPS="2000"
LOG_EVERY="50"
SAVE_EVERY="1000"
VAL_EVERY="1000"
VIZ_EVERY="1000"
NODES="1"
MASTER_PORT="29400"

PARTITION=""
ACCOUNT=""
QOS=""
CONSTRAINT=""
TIME_LIMIT="24:00:00"
CPUS="160"
MEMORY="1000G"
SIGNAL_SECONDS="1200"
MAX_REQUEUES="12"
JOB_NAME=""
EXECUTE=0

while (($#)); do
  case "$1" in
    --variant) require_value "$@"; VARIANT="$2"; shift 2 ;;
    --python) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) require_value "$@"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) require_value "$@"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --data-root) require_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --run-root) require_value "$@"; RUN_ROOT="$2"; shift 2 ;;
    --run-name) require_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --dataset-stage) require_value "$@"; DATASET_STAGE="$2"; shift 2 ;;
    --datasets) require_value "$@"; DATASETS_CSV="$2"; shift 2 ;;
    --smoke-report) require_value "$@"; SMOKE_REPORT="$2"; shift 2 ;;
    --data-validation-report) require_value "$@"; DATA_VALIDATION_REPORT="$2"; shift 2 ;;
    --data-validation-policy) require_value "$@"; DATA_VALIDATION_POLICY="$2"; shift 2 ;;
    --fast-training-authorization) require_value "$@"; FAST_TRAINING_AUTHORIZATION="$2"; shift 2 ;;
    --mixed-loader-report) require_value "$@"; MIXED_LOADER_REPORT="$2"; shift 2 ;;
    --transition-handoff) require_value "$@"; TRANSITION_HANDOFF="$2"; shift 2 ;;
    --wandb-mode) require_value "$@"; WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-project) require_value "$@"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    --wandb-entity) require_value "$@"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --wandb-run-id) require_value "$@"; WANDB_RUN_ID="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation-steps) require_value "$@"; GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --min-gpu-memory-mib) require_value "$@"; MIN_GPU_MEMORY_MIB="$2"; shift 2 ;;
    --max-iter) require_value "$@"; MAX_ITER="$2"; shift 2 ;;
    --warmup-steps) require_value "$@"; WARMUP_STEPS="$2"; shift 2 ;;
    --log-every) require_value "$@"; LOG_EVERY="$2"; shift 2 ;;
    --save-every) require_value "$@"; SAVE_EVERY="$2"; shift 2 ;;
    --val-every) require_value "$@"; VAL_EVERY="$2"; shift 2 ;;
    --viz-every) require_value "$@"; VIZ_EVERY="$2"; shift 2 ;;
    --nodes) require_value "$@"; NODES="$2"; shift 2 ;;
    --master-port) require_value "$@"; MASTER_PORT="$2"; shift 2 ;;
    --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
    --account) require_value "$@"; ACCOUNT="$2"; shift 2 ;;
    --qos) require_value "$@"; QOS="$2"; shift 2 ;;
    --constraint) require_value "$@"; CONSTRAINT="$2"; shift 2 ;;
    --time) require_value "$@"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) require_value "$@"; CPUS="$2"; shift 2 ;;
    --mem) require_value "$@"; MEMORY="$2"; shift 2 ;;
    --signal-seconds) require_value "$@"; SIGNAL_SECONDS="$2"; shift 2 ;;
    --max-requeues) require_value "$@"; MAX_REQUEUES="$2"; shift 2 ;;
    --job-name) require_value "$@"; JOB_NAME="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$VARIANT" == "latent" || "$VARIANT" == "explicit" ]] || \
  die "--variant must be latent or explicit"
[[ -n "$PYTHON_BIN" && -n "$WAN_DIR_VALUE" && -n "$VIDEOX_HOME_VALUE" ]] || \
  die "--python, --wan-dir, and --videox-home are required"
[[ -n "$DATA_ROOT" && -n "$RUN_ROOT" && -n "$RUN_NAME" ]] || \
  die "--data-root, --run-root, and --run-name are required"
[[ -n "$SMOKE_REPORT" && -n "$DATA_VALIDATION_REPORT" && -n "$WANDB_MODE_VALUE" ]] || \
  die "--smoke-report, --data-validation-report, and --wandb-mode are required"
[[ "$DATA_VALIDATION_POLICY" == "strict" || "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]] || \
  die "--data-validation-policy must be strict or files_only_user_waived_v1"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  [[ -n "$FAST_TRAINING_AUTHORIZATION" && -n "$MIXED_LOADER_REPORT" ]] || \
    die "fast policy requires --fast-training-authorization and --mixed-loader-report"
else
  [[ -z "$FAST_TRAINING_AUTHORIZATION" && -z "$MIXED_LOADER_REPORT" ]] || \
    die "strict policy forbids fast authorization/mixed evidence"
fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "--run-name may contain only letters, digits, dot, underscore, and dash"
[[ "$DATASET_STAGE" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe --dataset-stage"
[[ "$DATASETS_CSV" =~ ^(droid|egodex|agibot|abc)(,(droid|egodex|agibot|abc))*$ ]] || \
  die "--datasets must be a comma-separated subset of droid,egodex,agibot,abc"
[[ -n "$WANDB_RUN_ID" ]] || WANDB_RUN_ID="$RUN_NAME"
[[ "$WANDB_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe --wandb-run-id"
[[ "$WANDB_MODE_VALUE" == "online" || "$WANDB_MODE_VALUE" == "offline" || "$WANDB_MODE_VALUE" == "disabled" ]] || \
  die "--wandb-mode must be online, offline, or disabled"
if [[ "$WANDB_MODE_VALUE" != "disabled" ]]; then
  [[ -n "$WANDB_PROJECT_VALUE" ]] || \
    die "--wandb-project is required for W&B $WANDB_MODE_VALUE mode"
fi
if [[ "$WANDB_MODE_VALUE" == "online" ]]; then
  [[ -n "$WANDB_ENTITY_VALUE" ]] || \
    die "--wandb-entity is required for W&B online mode"
fi
if [[ -z "$BATCH_SIZE" ]]; then
  [[ "$VARIANT" == "latent" ]] && BATCH_SIZE=4 || BATCH_SIZE=2
fi
if [[ -z "$GRADIENT_ACCUMULATION_STEPS" ]]; then
  [[ "$VARIANT" == "latent" ]] && GRADIENT_ACCUMULATION_STEPS=4 || GRADIENT_ACCUMULATION_STEPS=1
fi
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "$NODES" =~ ^[1-9][0-9]*$ ]] && ((10#$NODES <= 32)) || \
  die "--nodes must be between 1 and 32"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "--master-port must be numeric"
((10#$MASTER_PORT >= 1024 && 10#$MASTER_PORT <= 65533)) || \
  die "--master-port must be between 1024 and 65533"
[[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || \
  die "--gradient-accumulation-steps must be a positive integer"
[[ "$MIN_GPU_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] || \
  die "--min-gpu-memory-mib must be a positive integer"
((10#$MIN_GPU_MEMORY_MIB >= 78000)) || \
  die "--min-gpu-memory-mib must be at least 78000 for the planned 80 GiB profile"
[[ "$MAX_ITER" =~ ^[1-9][0-9]*$ ]] || die "--max-iter must be a positive integer"
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || die "--warmup-steps must be a non-negative integer"
((10#$WARMUP_STEPS < 10#$MAX_ITER)) || die "--warmup-steps must be less than --max-iter"
for cadence_pair in \
  "--log-every=$LOG_EVERY" \
  "--save-every=$SAVE_EVERY" \
  "--val-every=$VAL_EVERY" \
  "--viz-every=$VIZ_EVERY"; do
  cadence_value="${cadence_pair#*=}"
  [[ "$cadence_value" =~ ^[1-9][0-9]*$ ]] || \
    die "${cadence_pair%%=*} must be a positive integer"
done

[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$SIGNAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  die "--signal-seconds must be a positive integer"
[[ "$MAX_REQUEUES" =~ ^(0|[1-9][0-9]{0,2})$ ]] || \
  die "--max-requeues must be an integer between 0 and $MAX_AUTOMATIC_REQUEUES"
((10#$MAX_REQUEUES <= MAX_AUTOMATIC_REQUEUES)) || \
  die "--max-requeues must not exceed $MAX_AUTOMATIC_REQUEUES"
MAX_REQUEUES="$((10#$MAX_REQUEUES))"
((SIGNAL_SECONDS <= 65535)) || die "--signal-seconds must not exceed Slurm's 65535-second limit"
[[ -n "$TIME_LIMIT" ]] || die "--time may not be empty"
[[ -n "$MEMORY" ]] || die "--mem may not be empty"
for scalar_pair in \
  "--time=$TIME_LIMIT" \
  "--mem=$MEMORY" \
  "--partition=$PARTITION" \
  "--account=$ACCOUNT" \
  "--qos=$QOS" \
  "--constraint=$CONSTRAINT"; do
  scalar_value="${scalar_pair#*=}"
  [[ "$scalar_value" != *[[:space:]]* ]] || \
    die "${scalar_pair%%=*} may not contain whitespace"
done

for path_pair in \
  "--python=$PYTHON_BIN" \
  "--wan-dir=$WAN_DIR_VALUE" \
  "--videox-home=$VIDEOX_HOME_VALUE" \
  "--data-root=$DATA_ROOT" \
  "--run-root=$RUN_ROOT" \
  "--smoke-report=$SMOKE_REPORT" \
  "--data-validation-report=$DATA_VALIDATION_REPORT"; do
  require_absolute "${path_pair%%=*}" "${path_pair#*=}"
done
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  require_absolute "--fast-training-authorization" "$FAST_TRAINING_AUTHORIZATION"
  require_absolute "--mixed-loader-report" "$MIXED_LOADER_REPORT"
fi
if [[ -n "$TRANSITION_HANDOFF" ]]; then
  require_absolute "--transition-handoff" "$TRANSITION_HANDOFF"
  [[ -s "$TRANSITION_HANDOFF" ]] || \
    die "transition handoff not found or empty: $TRANSITION_HANDOFF"
fi

[[ -x "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is not executable: $SBATCH_SCRIPT"
[[ -f "$STATE_VALIDATOR" ]] || die "state validator is missing: $STATE_VALIDATOR"
[[ -f "$RUN_ROOT_POLICY" ]] || die "run-root policy is missing: $RUN_ROOT_POLICY"
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ -f "$SMOKE_REPORT" ]] || die "smoke report not found: $SMOKE_REPORT"
[[ -f "$DATA_VALIDATION_REPORT" ]] || \
  die "data validation report not found: $DATA_VALIDATION_REPORT"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  [[ -f "$FAST_TRAINING_AUTHORIZATION" && ! -L "$FAST_TRAINING_AUTHORIZATION" ]] || \
    die "fast training authorization not found or symlinked: $FAST_TRAINING_AUTHORIZATION"
  [[ -f "$MIXED_LOADER_REPORT" && ! -L "$MIXED_LOADER_REPORT" ]] || \
    die "mixed-loader report not found or symlinked: $MIXED_LOADER_REPORT"
fi

if ! RUN_ROOT="$("$PYTHON_BIN" "$RUN_ROOT_POLICY" --run-root "$RUN_ROOT")"; then
  die "run root rejected by policy; configure canonical cluster roots with LACWM_ALLOWED_RUN_ROOTS"
fi
RUN_DIR="$RUN_ROOT/$RUN_NAME"
COMPLETE_MARKER="$RUN_DIR/training_complete.json"
validate_completion_marker() {
  "$PYTHON_BIN" "$STATE_VALIDATOR" completion \
    --path "$COMPLETE_MARKER" \
    --identity "$RUN_DIR/run_identity.json" \
    --run-dir "$RUN_DIR" \
    --expected-max-iter "$MAX_ITER"
}
if [[ -f "$COMPLETE_MARKER" ]]; then
  validate_completion_marker || \
    die "refusing malformed completion marker: $COMPLETE_MARKER"
  die "training is already complete: $COMPLETE_MARKER"
fi

if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="lacwm-$RUN_NAME"
fi
[[ "$JOB_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "--job-name may contain only letters, digits, dot, underscore, and dash"

LOG_DIR="$RUN_ROOT/_slurm/logs"
SBATCH_ARGS=(
  --parsable
  --export=ALL
  "--nodes=$NODES"
  "--ntasks=$NODES"
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --signal="B:USR1@$SIGNAL_SECONDS"
  --requeue
  --dependency=singleton
  --open-mode=append
  --job-name="$JOB_NAME"
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.out"
)
[[ -n "$PARTITION" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos="$QOS")
[[ -n "$CONSTRAINT" ]] && SBATCH_ARGS+=(--constraint="$CONSTRAINT")

JOB_ARGS=(
  --nodes "$NODES"
  --master-port "$MASTER_PORT"
  --variant "$VARIANT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --data-root "$DATA_ROOT"
  --run-root "$RUN_ROOT"
  --run-name "$RUN_NAME"
  --dataset-stage "$DATASET_STAGE"
  --datasets "$DATASETS_CSV"
  --smoke-report "$SMOKE_REPORT"
  --data-validation-report "$DATA_VALIDATION_REPORT"
  --data-validation-policy "$DATA_VALIDATION_POLICY"
  --wandb-mode "$WANDB_MODE_VALUE"
  --wandb-run-id "$WANDB_RUN_ID"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
  --max-iter "$MAX_ITER"
  --warmup-steps "$WARMUP_STEPS"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --val-every "$VAL_EVERY"
  --viz-every "$VIZ_EVERY"
  --max-requeues "$MAX_REQUEUES"
)
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  JOB_ARGS+=(
    --fast-training-authorization "$FAST_TRAINING_AUTHORIZATION"
    --mixed-loader-report "$MIXED_LOADER_REPORT"
  )
fi
[[ -n "$TRANSITION_HANDOFF" ]] && \
  JOB_ARGS+=(--transition-handoff "$TRANSITION_HANDOFF")
[[ -n "$WANDB_PROJECT_VALUE" ]] && JOB_ARGS+=(--wandb-project "$WANDB_PROJECT_VALUE")
[[ -n "$WANDB_ENTITY_VALUE" ]] && JOB_ARGS+=(--wandb-entity "$WANDB_ENTITY_VALUE")

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
COMMAND=("$SBATCH_BIN" "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")

echo "Planned persistent run directory: $RUN_DIR"
echo "Planned Slurm log directory: $LOG_DIR"
echo "Planned topology: nodes=$NODES gpus_per_node=8 world_size=$((10#$NODES * 8))"
echo "Planned schedule: max_iter=$MAX_ITER warmup_steps=$WARMUP_STEPS log_every=$LOG_EVERY save_every=$SAVE_EVERY val_every=$VAL_EVERY viz_every=$VIZ_EVERY"
if [[ -n "$TRANSITION_HANDOFF" ]]; then
  echo "Planned immutable parent transition: $TRANSITION_HANDOFF"
else
  echo "Planned immutable parent transition: none"
fi
printf 'Validated sbatch command:'
print_command "${COMMAND[@]}"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --execute to create the log directory and submit."
  exit 0
fi

command -v "$SBATCH_BIN" >/dev/null 2>&1 || \
  die "Slurm submission command is unavailable: $SBATCH_BIN"
mkdir -p "$LOG_DIR"
if [[ -f "$COMPLETE_MARKER" ]]; then
  validate_completion_marker || \
    die "completion marker appeared but is malformed: $COMPLETE_MARKER"
  die "training completed before submission: $COMPLETE_MARKER"
fi

JOB_ID="$("${COMMAND[@]}")" || die "sbatch submission failed"
[[ -n "$JOB_ID" ]] || die "sbatch returned an empty job identifier"
echo "Submitted Slurm job: $JOB_ID"
echo "The job will self-requeue only after a trainer checkpoint ACK."
