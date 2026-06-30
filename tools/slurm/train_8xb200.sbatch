#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="$REPO_ROOT/tools/launch_8xb200.sh"
STATE_VALIDATOR="$REPO_ROOT/tools/slurm/validate_state.py"

usage() {
  cat <<'EOF'
Usage: tools/slurm/train_8xb200.sbatch [training arguments]

This file is an internal Slurm job entrypoint. Submit it through
tools/slurm/submit_8xb200.sh so the resource and signal options are complete.

Required:
  --variant NAME                  latent or explicit
  --python PATH                   Provisioned lacwm Python executable
  --wan-dir PATH                  Wan2.1-Fun-1.3B-Control assets
  --videox-home PATH              Pinned VideoX-Fun checkout
  --data-root PATH                Prepared four-dataset root
  --run-root PATH                 Shared, persistent run root
  --run-name NAME                 Stable run name
  --smoke-report PATH             Passing real-data gradient report
  --data-validation-report PATH   Passing strict data report
  --wandb-mode MODE               online, offline, or disabled

Required for online/offline W&B:
  --wandb-project NAME
Required for online W&B:
  --wandb-entity NAME

Optional:
  --batch-size N                  Physical per-GPU microbatch
  --gradient-accumulation-steps N Microbatches per optimizer update
  --min-gpu-memory-mib N          Minimum total memory on each B200
  --max-requeues N                Maximum self-requeues (default: 12)
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local name="$1"
  local value="$2"
  [[ "$value" == /* ]] || die "$name must be an absolute path: $value"
}

VARIANT=""
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
MIN_GPU_MEMORY_MIB=""
MAX_REQUEUES="12"

while (($#)); do
  case "$1" in
    --variant) [[ $# -ge 2 ]] || die "--variant requires a value"; VARIANT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; DATA_ROOT="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || die "--run-root requires a value"; RUN_ROOT="$2"; shift 2 ;;
    --run-name) [[ $# -ge 2 ]] || die "--run-name requires a value"; RUN_NAME="$2"; shift 2 ;;
    --smoke-report) [[ $# -ge 2 ]] || die "--smoke-report requires a value"; SMOKE_REPORT="$2"; shift 2 ;;
    --data-validation-report) [[ $# -ge 2 ]] || die "--data-validation-report requires a value"; DATA_VALIDATION_REPORT="$2"; shift 2 ;;
    --wandb-mode) [[ $# -ge 2 ]] || die "--wandb-mode requires a value"; WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-project) [[ $# -ge 2 ]] || die "--wandb-project requires a value"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    --wandb-entity) [[ $# -ge 2 ]] || die "--wandb-entity requires a value"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || die "--batch-size requires a value"; BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation-steps) [[ $# -ge 2 ]] || die "--gradient-accumulation-steps requires a value"; GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --min-gpu-memory-mib) [[ $# -ge 2 ]] || die "--min-gpu-memory-mib requires a value"; MIN_GPU_MEMORY_MIB="$2"; shift 2 ;;
    --max-requeues) [[ $# -ge 2 ]] || die "--max-requeues requires a value"; MAX_REQUEUES="$2"; shift 2 ;;
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
[[ -n "$BATCH_SIZE" && -n "$GRADIENT_ACCUMULATION_STEPS" && -n "$MIN_GPU_MEMORY_MIB" ]] || \
  die "--batch-size, --gradient-accumulation-steps, and --min-gpu-memory-mib are required"
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
[[ "$MAX_REQUEUES" =~ ^[0-9]+$ ]] || die "--max-requeues must be a non-negative integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || \
  die "--gradient-accumulation-steps must be a positive integer"
[[ "$MIN_GPU_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] || \
  die "--min-gpu-memory-mib must be a positive integer"
((10#$MIN_GPU_MEMORY_MIB >= 78000)) || \
  die "--min-gpu-memory-mib must be at least 78000 for the planned 80 GiB profile"

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

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint must run inside a Slurm batch job"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "the production run requires exactly one node"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || \
  die "the batch allocation must contain one Slurm task; torchrun creates the eight GPU ranks"
[[ -x "$LAUNCHER" ]] || die "guarded launcher is not executable: $LAUNCHER"
[[ -f "$STATE_VALIDATOR" ]] || die "state validator is missing: $STATE_VALIDATOR"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v srun >/dev/null 2>&1 || die "srun is required"
command -v scontrol >/dev/null 2>&1 || die "scontrol is required for safe self-requeue"

RUN_ROOT="${RUN_ROOT%/}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
COMPLETE_MARKER="$RUN_DIR/training_complete.json"
IDENTITY_FILE="$RUN_DIR/run_identity.json"

validate_completion() {
  "$PYTHON_BIN" "$STATE_VALIDATOR" completion \
    --path "$COMPLETE_MARKER" \
    --identity "$IDENTITY_FILE" \
    --run-dir "$RUN_DIR" \
    --expected-max-iter 60000
}

# The existing launcher has a host-local lock. This second lock is intentionally
# on the persistent run filesystem so two allocations on different nodes cannot
# write the same snapshot concurrently. The filesystem must support flock(2).
LOCK_DIR="$RUN_ROOT/_slurm/locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="$LOCK_DIR/$RUN_NAME.lock"
exec 8>"$LOCK_FILE"
flock -n 8 || die "another allocation holds the shared run lock: $LOCK_FILE"

if [[ -f "$COMPLETE_MARKER" ]]; then
  validate_completion || die "refusing malformed completion marker: $COMPLETE_MARKER"
  echo "Training is already complete: $COMPLETE_MARKER"
  exit 0
fi

RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || die "SLURM_RESTART_COUNT is not numeric: $RESTART_COUNT"
ATTEMPT_ID="${SLURM_JOB_ID}.${RESTART_COUNT}"
CONTROL_DIR="$RUN_ROOT/_slurm/control/$RUN_NAME"
mkdir -p "$CONTROL_DIR"
REQUEST_FILE="$CONTROL_DIR/${ATTEMPT_ID}.checkpoint-request"
ACK_FILE="$CONTROL_DIR/${ATTEMPT_ID}.checkpoint-ack"
[[ ! -e "$REQUEST_FILE" && ! -e "$ACK_FILE" ]] || \
  die "stale control files exist for Slurm attempt $ATTEMPT_ID"

export LACWM_CHECKPOINT_REQUEST_FILE="$REQUEST_FILE"
export LACWM_CHECKPOINT_ACK_FILE="$ACK_FILE"
export LACWM_SLURM_ATTEMPT_ID="$ATTEMPT_ID"

CHECKPOINT_REQUESTED=0
request_checkpoint() {
  local temporary
  CHECKPOINT_REQUESTED=1
  if [[ -e "$REQUEST_FILE" ]]; then
    return 0
  fi
  temporary="${REQUEST_FILE}.tmp.$$"
  if ! {
    printf 'schema_version=1\n'
    printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
    printf 'slurm_restart_count=%s\n' "$RESTART_COUNT"
    printf 'requested_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$temporary"; then
    echo "ERROR: could not write checkpoint request temporary file: $temporary" >&2
    return 0
  fi
  if mv -- "$temporary" "$REQUEST_FILE"; then
    echo "Checkpoint-and-stop requested by Slurm for attempt $ATTEMPT_ID"
  else
    echo "ERROR: could not publish checkpoint request: $REQUEST_FILE" >&2
  fi
}
trap request_checkpoint USR1

RESUME_ARGS=()
if [[ -f "$IDENTITY_FILE" ]]; then
  # The original full report was content-validated before the run began. On
  # later allocations the guarded launcher still recomputes every active-file
  # fingerprint, so permit that immutable report for the expected multi-day
  # queue horizon without spending an allocated B200 node revalidating data.
  RESUME_ARGS+=(--resume)
  echo "Existing run identity found; this allocation will resume $RUN_DIR"
elif [[ -e "$RUN_DIR/snapshot.pt" ]]; then
  die "snapshot exists without wrapper run identity: $RUN_DIR/snapshot.pt"
elif [[ -e "$RUN_DIR" ]]; then
  die "run directory exists without wrapper identity or completion marker: $RUN_DIR"
else
  echo "No run identity exists; this allocation will create $RUN_DIR"
fi

LAUNCH_COMMAND=(
  "$LAUNCHER"
  --gpus 0,1,2,3,4,5,6,7
  --variant "$VARIANT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --data-root "$DATA_ROOT"
  --run-root "$RUN_ROOT"
  --run-dir "$RUN_DIR"
  --run-name "$RUN_NAME"
  --smoke-report "$SMOKE_REPORT"
  --data-validation-report "$DATA_VALIDATION_REPORT"
  --max-data-report-age-hours 720
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
  --wandb-mode "$WANDB_MODE_VALUE"
)
if [[ -n "$WANDB_PROJECT_VALUE" ]]; then
  LAUNCH_COMMAND+=(--wandb-project "$WANDB_PROJECT_VALUE")
fi
if [[ -n "$WANDB_ENTITY_VALUE" ]]; then
  LAUNCH_COMMAND+=(--wandb-entity "$WANDB_ENTITY_VALUE")
fi
LAUNCH_COMMAND+=("${RESUME_ARGS[@]}" --execute)

printf 'Slurm attempt %q launching:' "$ATTEMPT_ID"
printf ' %q' "${LAUNCH_COMMAND[@]}"
printf '\n'

# Keep srun in the background so Bash handles B:USR1 immediately. `wait` can
# return because the trap ran while the child remains alive, so always check the
# PID and wait again until the actual launcher process has terminated.
srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=8 \
  --kill-on-bad-exit=1 \
  "${LAUNCH_COMMAND[@]}" &
CHILD_PID=$!

CHILD_STATUS=0
set +e
while true; do
  wait "$CHILD_PID"
  CHILD_STATUS=$?
  if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    break
  fi
  echo "wait was interrupted while launcher PID $CHILD_PID is still active; continuing to wait"
done
set -e
trap - USR1

# Completion is authoritative and always wins over a late time-limit signal.
if [[ -f "$COMPLETE_MARKER" ]]; then
  validate_completion || die "refusing malformed completion marker: $COMPLETE_MARKER"
  echo "Training completed; no continuation will be queued: $COMPLETE_MARKER"
  exit 0
fi

if ((CHECKPOINT_REQUESTED == 1)); then
  # The Trainer writes ACK only after a durable snapshot, or after confirming
  # that a request before update zero is safe to retry. Allow a short metadata-
  # propagation window on distributed filesystems after torchrun exits.
  for ((ack_poll = 0; ack_poll < 30; ack_poll++)); do
    [[ -f "$ACK_FILE" ]] && break
    sleep 1
  done
  if [[ ! -f "$ACK_FILE" ]]; then
    echo "ERROR: checkpoint request was not acknowledged; refusing unsafe requeue" >&2
    echo "Request: $REQUEST_FILE" >&2
    echo "Expected ACK: $ACK_FILE" >&2
    exit 75
  fi

  if ! "$PYTHON_BIN" "$STATE_VALIDATOR" ack \
    --path "$ACK_FILE" \
    --identity "$IDENTITY_FILE" \
    --run-dir "$RUN_DIR" \
    --attempt-id "$ATTEMPT_ID" \
    --expected-max-iter 60000; then
    echo "ERROR: refusing requeue for an invalid checkpoint ACK: $ACK_FILE" >&2
    exit 75
  fi

  if ((RESTART_COUNT >= MAX_REQUEUES)); then
    echo "ERROR: checkpoint is safe, but the configured requeue cap ($MAX_REQUEUES) has been reached" >&2
    echo "Resume manually after inspecting progress and queue policy." >&2
    exit 75
  fi

  echo "Safe scheduler continuation acknowledged: $ACK_FILE"
  echo "Requeueing Slurm job $SLURM_JOB_ID for the next allocation"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi

if ((CHILD_STATUS != 0)); then
  echo "ERROR: guarded launcher exited with status $CHILD_STATUS; not requeueing" >&2
  exit "$CHILD_STATUS"
fi

die "launcher exited successfully without a completion marker or a scheduler checkpoint request"
