#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/train_8xb200.sbatch"
STATE_VALIDATOR="$REPO_ROOT/tools/slurm/validate_state.py"

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_8xb200.sh [options]

Build and preview a single-node 8xB200 Slurm submission. This command is a dry
run by default and does not require Slurm client commands until --execute.

Training inputs:
  --python PATH                   Provisioned lacwm Python executable
  --wan-dir PATH                  Wan2.1-Fun-1.3B-Control assets
  --videox-home PATH              Pinned VideoX-Fun checkout
  --data-root PATH                Prepared four-dataset root
  --run-root PATH                 Shared, persistent run root
  --run-name NAME                 Stable run name
  --smoke-report PATH             Passing real-data gradient report
  --data-validation-report PATH   Passing strict data report
  --wandb-mode MODE               online, offline, or disabled
  --wandb-project NAME            Required for online/offline W&B
  --wandb-entity NAME             Required for online W&B
  --variant NAME                  latent (default) or explicit
  --batch-size N                  Physical per-GPU microbatch (default: latent=4,
                                  explicit=2)
  --gradient-accumulation-steps N Microbatches per update (default: latent=4,
                                  explicit=1)
  --min-gpu-memory-mib N          Minimum memory on each B200 (default: 78000)

Slurm resources:
  --partition NAME                Optional partition
  --account NAME                  Optional account
  --qos NAME                      Optional QOS
  --constraint VALUE              Optional node constraint, e.g. b200
  --time VALUE                    Slot wall time (default: 24:00:00)
  --cpus N                        CPUs assigned to the one torchrun task (default: 160)
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
SMOKE_REPORT=""
DATA_VALIDATION_REPORT=""
WANDB_MODE_VALUE=""
WANDB_PROJECT_VALUE=""
WANDB_ENTITY_VALUE=""
BATCH_SIZE=""
GRADIENT_ACCUMULATION_STEPS=""
MIN_GPU_MEMORY_MIB="78000"

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
    --smoke-report) require_value "$@"; SMOKE_REPORT="$2"; shift 2 ;;
    --data-validation-report) require_value "$@"; DATA_VALIDATION_REPORT="$2"; shift 2 ;;
    --wandb-mode) require_value "$@"; WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-project) require_value "$@"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    --wandb-entity) require_value "$@"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation-steps) require_value "$@"; GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --min-gpu-memory-mib) require_value "$@"; MIN_GPU_MEMORY_MIB="$2"; shift 2 ;;
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
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "--run-name may contain only letters, digits, dot, underscore, and dash"
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
[[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || \
  die "--gradient-accumulation-steps must be a positive integer"
[[ "$MIN_GPU_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] || \
  die "--min-gpu-memory-mib must be a positive integer"
((10#$MIN_GPU_MEMORY_MIB >= 78000)) || \
  die "--min-gpu-memory-mib must be at least 78000 for the planned 80 GiB profile"

[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$SIGNAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  die "--signal-seconds must be a positive integer"
[[ "$MAX_REQUEUES" =~ ^[0-9]+$ ]] || die "--max-requeues must be a non-negative integer"
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

[[ -x "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is not executable: $SBATCH_SCRIPT"
[[ -f "$STATE_VALIDATOR" ]] || die "state validator is missing: $STATE_VALIDATOR"
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ -f "$SMOKE_REPORT" ]] || die "smoke report not found: $SMOKE_REPORT"
[[ -f "$DATA_VALIDATION_REPORT" ]] || \
  die "data validation report not found: $DATA_VALIDATION_REPORT"

RUN_ROOT="${RUN_ROOT%/}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
COMPLETE_MARKER="$RUN_DIR/training_complete.json"
validate_completion_marker() {
  "$PYTHON_BIN" "$STATE_VALIDATOR" completion \
    --path "$COMPLETE_MARKER" \
    --identity "$RUN_DIR/run_identity.json" \
    --run-dir "$RUN_DIR" \
    --expected-max-iter 60000
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
  --nodes=1
  --ntasks=1
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
  --variant "$VARIANT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --data-root "$DATA_ROOT"
  --run-root "$RUN_ROOT"
  --run-name "$RUN_NAME"
  --smoke-report "$SMOKE_REPORT"
  --data-validation-report "$DATA_VALIDATION_REPORT"
  --wandb-mode "$WANDB_MODE_VALUE"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
  --max-requeues "$MAX_REQUEUES"
)
[[ -n "$WANDB_PROJECT_VALUE" ]] && JOB_ARGS+=(--wandb-project "$WANDB_PROJECT_VALUE")
[[ -n "$WANDB_ENTITY_VALUE" ]] && JOB_ARGS+=(--wandb-entity "$WANDB_ENTITY_VALUE")

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
COMMAND=("$SBATCH_BIN" "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")

echo "Planned persistent run directory: $RUN_DIR"
echo "Planned Slurm log directory: $LOG_DIR"
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
