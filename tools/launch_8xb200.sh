#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/tools"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"

usage() {
  cat <<'EOF'
Usage: tools/launch_8xb200.sh [options]

Required:
  --gpus LIST              Eight physical GPU indices, e.g. 0,1,2,3,4,5,6,7.
  --variant NAME           latent or explicit.
  --python PATH            Python from the provisioned lacwm environment.
  --wan-dir PATH           Wan2.1-Fun-1.3B-Control asset directory.
  --videox-home PATH       VideoX-Fun checkout.
  --data-root PATH         Root containing all four prepared datasets.
  --run-root PATH          Output root under /mnt/data1 or /mnt/data2.
  --run-name NAME          Unique W&B/Hydra run name.
  --smoke-report PATH      Passing report from run_gradient_smoke.sh for the same
                           Git commit and model variant.
  --data-validation-report PATH
                           Passing strict JSON output from:
                           tools/validate_training_data.py --json
  --wandb-mode MODE        online, offline, or disabled.

Required for online/offline W&B:
  --wandb-project NAME
Required for online W&B:
  --wandb-entity NAME

Optional:
  --batch-size N           Physical per-GPU microbatch (default latent=4, explicit=2).
  --gradient-accumulation-steps N
                           Microbatches per optimizer update (default latent=4,
                           explicit=1).
  --min-gpu-memory-mib N   Minimum total memory on every B200 (default: 78000;
                           values below the planned 80 GiB profile are rejected).
  --max-data-report-age-hours N
                           Cached report age limit (default 24). Long Slurm runs
                           may raise this; fingerprints are still recomputed.
  --run-dir PATH           Stable Hydra/checkpoint directory beneath --run-root
                           (default: RUN_ROOT/RUN_NAME).
  --resume                 Reuse an existing wrapper-owned RUN_DIR. Trainer resumes
                           snapshot.pt when present; otherwise this retries iteration 0.
                           Without this, existing run identity/snapshot fails closed.
  --execute                Launch in the foreground. Without this, preflight and
                           command preview only.
  -h, --help               Show this help.

The wrapper enforces eight B200 GPUs with >=78000 MiB each by default, a clean worktree,
complete configured datasets, a prior gradient smoke, and an otherwise idle host.
It never kills or pauses another job and never launches in the background.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

assert_repo_identity() {
  local actual_commit actual_status
  actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  [[ "$actual_commit" == "$CURRENT_COMMIT" ]] || die \
    "repository HEAD changed during launch: $actual_commit != $CURRENT_COMMIT"
  actual_status="$(git -C "$REPO_ROOT" status --porcelain)"
  [[ -z "$actual_status" ]] || die \
    "repository became dirty during launch; refusing to mix code revisions"
}

validate_smoke_report() {
  "$PYTHON_BIN" "$TOOLS_DIR/validate_smoke_report.py" \
    --report "$1" \
    --variant "$VARIANT" \
    --git-commit "$CURRENT_COMMIT" \
    --wan-dir "$WAN_PATH" \
    --videox-home "$VIDEOX_PATH" \
    --data-root "$DATA_PATH"
}

GPUS=""
VARIANT=""
PYTHON_BIN=""
WAN_PATH=""
VIDEOX_PATH=""
DATA_PATH=""
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
MAX_DATA_REPORT_AGE_HOURS="24"
REQUESTED_RUN_DIR=""
RESUME=0
EXECUTE=0

while (($#)); do
  case "$1" in
    --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; GPUS="$2"; shift 2 ;;
    --variant) [[ $# -ge 2 ]] || die "--variant requires a value"; VARIANT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_PATH="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_PATH="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; DATA_PATH="$2"; shift 2 ;;
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
    --max-data-report-age-hours) [[ $# -ge 2 ]] || die "--max-data-report-age-hours requires a value"; MAX_DATA_REPORT_AGE_HOURS="$2"; shift 2 ;;
    --run-dir) [[ $# -ge 2 ]] || die "--run-dir requires a value"; REQUESTED_RUN_DIR="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$VARIANT" == "latent" || "$VARIANT" == "explicit" ]] || die "--variant must be latent or explicit"
[[ -n "$GPUS" && -n "$PYTHON_BIN" && -n "$WAN_PATH" && -n "$VIDEOX_PATH" && -n "$DATA_PATH" && -n "$RUN_ROOT" ]] || die "all GPU/path options are required"
[[ -n "$RUN_NAME" && -n "$SMOKE_REPORT" && -n "$DATA_VALIDATION_REPORT" && -n "$WANDB_MODE_VALUE" ]] || die "--run-name, --smoke-report, --data-validation-report, and --wandb-mode are required"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-name may contain only letters, digits, dot, underscore, and dash"
[[ "$WANDB_MODE_VALUE" == "online" || "$WANDB_MODE_VALUE" == "offline" || "$WANDB_MODE_VALUE" == "disabled" ]] || die "--wandb-mode must be online, offline, or disabled"
if [[ "$WANDB_MODE_VALUE" != "disabled" ]]; then
  [[ -n "$WANDB_PROJECT_VALUE" ]] || die "--wandb-project is required for W&B $WANDB_MODE_VALUE mode"
fi
if [[ "$WANDB_MODE_VALUE" == "online" ]]; then
  [[ -n "$WANDB_ENTITY_VALUE" ]] || die "--wandb-entity is required for W&B online mode"
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
[[ "$MAX_DATA_REPORT_AGE_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  die "--max-data-report-age-hours must be a non-negative number"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
[[ ${#GPU_ARRAY[@]} -eq 8 ]] || die "--gpus must contain exactly eight indices"
declare -A GPU_SEEN=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
  [[ -z "${GPU_SEEN[$gpu]:-}" ]] || die "duplicate GPU index: $gpu"
  GPU_SEEN[$gpu]=1
done
EFFECTIVE_GLOBAL_BATCH_SIZE=$((10#$BATCH_SIZE * 10#$GRADIENT_ACCUMULATION_STEPS * 8))

# Normalize every executable/data path once while still in the caller's working
# directory. The training process later changes directory into PROJECT_ROOT.
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
PYTHON_BIN="$("$PYTHON_BIN" - "$PYTHON_BIN" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().absolute())
PY
)"
mapfile -t NORMALIZED_PATHS < <("$PYTHON_BIN" - "$WAN_PATH" "$VIDEOX_PATH" "$DATA_PATH" "$RUN_ROOT" "$SMOKE_REPORT" "$DATA_VALIDATION_REPORT" <<'PY'
from pathlib import Path
import sys
for value in sys.argv[1:]:
    print(Path(value).expanduser().resolve(strict=False))
PY
)
WAN_PATH="${NORMALIZED_PATHS[0]}"
VIDEOX_PATH="${NORMALIZED_PATHS[1]}"
DATA_PATH="${NORMALIZED_PATHS[2]}"
RUN_ROOT="${NORMALIZED_PATHS[3]}"
SMOKE_REPORT="${NORMALIZED_PATHS[4]}"
DATA_VALIDATION_REPORT="${NORMALIZED_PATHS[5]}"

[[ -f "$SMOKE_REPORT" ]] || die "smoke report not found: $SMOKE_REPORT"
[[ -f "$DATA_VALIDATION_REPORT" ]] || die "strict data validation report not found: $DATA_VALIDATION_REPORT"
CURRENT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
validate_smoke_report "$SMOKE_REPORT"

RUN_ROOT_RESOLVED="$RUN_ROOT"
if [[ -n "$REQUESTED_RUN_DIR" ]]; then
  RUN_DIR="$("$PYTHON_BIN" - "$REQUESTED_RUN_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"
else
  RUN_DIR="$RUN_ROOT_RESOLVED/$RUN_NAME"
fi
case "$RUN_DIR/" in
  "$RUN_ROOT_RESOLVED"/*) ;;
  *) die "--run-dir must resolve beneath --run-root ($RUN_ROOT_RESOLVED): $RUN_DIR" ;;
esac
[[ "$RUN_DIR" != "$RUN_ROOT_RESOLVED" ]] || die "--run-dir cannot equal --run-root"
RUN_ROOT="$RUN_ROOT_RESOLVED"

SNAPSHOT="$RUN_DIR/snapshot.pt"
IDENTITY_FILE="$RUN_DIR/run_identity.json"
if ((RESUME == 1)); then
  [[ -f "$IDENTITY_FILE" ]] || die "--resume requires wrapper identity metadata: $IDENTITY_FILE"
  "$PYTHON_BIN" - "$IDENTITY_FILE" "$VARIANT" "$CURRENT_COMMIT" "$BATCH_SIZE" \
    "$GRADIENT_ACCUMULATION_STEPS" "$EFFECTIVE_GLOBAL_BATCH_SIZE" \
    "$MIN_GPU_MEMORY_MIB" "$RUN_NAME" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
variant, commit, batch_size, accumulation, effective_batch, min_memory, run_name = sys.argv[2:]
payload = json.loads(path.read_text())
expected = {
    "schema_version": 3,
    "variant": variant,
    "git_commit": commit,
    "batch_size": int(batch_size),
    "gradient_accumulation_steps": int(accumulation),
    "world_size": 8,
    "effective_global_batch_size": int(effective_batch),
    "gpu_profile": {"model": "B200", "minimum_memory_mib": int(min_memory)},
    "run_name": run_name,
}
problems = [f"{key}: {payload.get(key)!r} != {value!r}" for key, value in expected.items() if payload.get(key) != value]
if problems:
    raise SystemExit(f"resume identity mismatch in {path}: " + "; ".join(problems))
print(f"Validated resume identity: {path}")
PY
  if [[ -f "$SNAPSHOT" ]]; then
    echo "Resume checkpoint found: $SNAPSHOT"
  else
    echo "No checkpoint exists yet; --resume will retry this identified run from iteration 0."
  fi
else
  [[ ! -e "$SNAPSHOT" ]] || die "$SNAPSHOT exists; pass --resume to preserve and continue it"
  [[ ! -e "$IDENTITY_FILE" ]] || die "$IDENTITY_FILE exists; choose another --run-dir or pass --resume with a snapshot"
  [[ ! -e "$RUN_DIR" ]] || die "fresh --run-dir must not already exist: $RUN_DIR"
fi

export WAN_DIR="$WAN_PATH"
export VIDEOX_HOME="$VIDEOX_PATH"
export LACWM_DATA="$DATA_PATH"
export LACWM_RUNS="$RUN_ROOT"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE="$WANDB_MODE_VALUE"
export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
export WANDB_ENTITY="$WANDB_ENTITY_VALUE"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export PYTHONPATH="$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME:$PROJECT_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PREFLIGHT=(
  "$PYTHON_BIN" "$TOOLS_DIR/training_preflight.py"
  --profile full
  --gpus "$GPUS"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR"
  --videox-home "$VIDEOX_HOME"
  --data-root "$LACWM_DATA"
  --run-root "$LACWM_RUNS"
  --data-validation-report "$DATA_VALIDATION_REPORT"
  --max-data-report-age-hours "$MAX_DATA_REPORT_AGE_HOURS"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
)

echo "Running guarded 8xB200 preflight..."
"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR"
"${PREFLIGHT[@]}"

EXPERIMENT="ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml"
if [[ "$VARIANT" == "explicit" ]]; then
  EXPERIMENT="ravenhuang/wan-dit/wan_dit_explicit_abc_agibot_droid_egodex.yaml"
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LAUNCH_DIR="$RUN_DIR/launches/${TIMESTAMP}_pid$$"
WANDB_ARGS=("wandb.enabled=false")
if [[ "$WANDB_MODE_VALUE" != "disabled" ]]; then
  WANDB_ARGS=("wandb.enabled=true" "wandb.project=$WANDB_PROJECT_VALUE" "+wandb.mode=$WANDB_MODE_VALUE")
  [[ -n "$WANDB_ENTITY_VALUE" ]] && WANDB_ARGS+=("+wandb.entity=$WANDB_ENTITY_VALUE")
fi
COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node=8
  train.py
  "+experiments_0908=$EXPERIMENT"
  "name=$RUN_NAME"
  "data_loader.batch_size=$BATCH_SIZE"
  "trainer.config.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  "hydra.run.dir=$RUN_DIR"
  "hydra.sweep.dir=$RUN_DIR"
  "${WANDB_ARGS[@]}"
)

printf 'Validated command (working directory %q):' "$PROJECT_ROOT"
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Planned output: $RUN_DIR"
echo "Batching: physical_per_gpu=$BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS effective_global=$EFFECTIVE_GLOBAL_BATCH_SIZE"
echo "GPU profile: B200 with at least $MIN_GPU_MEMORY_MIB MiB total memory per device"
echo "W&B mode: $WANDB_MODE_VALUE${WANDB_PROJECT_VALUE:+, project=$WANDB_PROJECT_VALUE}${WANDB_ENTITY_VALUE:+, entity=$WANDB_ENTITY_VALUE}"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --execute after reviewing the command."
  exit 0
fi

# Fixed per host/account rather than per run root: separate output roots must
# not allow two guarded wrappers to race through the idle-GPU check together.
LOCK_ROOT="${LACWM_HOST_LOCK_ROOT:-/run/user/${UID}}"
if [[ ! -d "$LOCK_ROOT" ]]; then
  [[ -z "${LACWM_HOST_LOCK_ROOT:-}" ]] || die \
    "configured runtime lock directory is unavailable: $LOCK_ROOT"
  LOCK_ROOT="/tmp/lacwm-runtime-${UID}"
  if [[ ! -e "$LOCK_ROOT" ]]; then
    (umask 077 && mkdir "$LOCK_ROOT") || die \
      "could not create fallback runtime lock directory: $LOCK_ROOT"
  fi
fi
[[ -d "$LOCK_ROOT" && ! -L "$LOCK_ROOT" ]] || die "runtime lock path is not a secure directory: $LOCK_ROOT"
[[ "$(stat -c %u "$LOCK_ROOT")" == "$UID" && "$(stat -c %a "$LOCK_ROOT")" == "700" ]] || \
  die "runtime lock directory must be owned by UID $UID with mode 700: $LOCK_ROOT"
LOCK_FILE="$LOCK_ROOT/lacwm-training.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another guarded lacwm launch holds $LOCK_FILE"

# Python multiprocessing creates AF_UNIX sockets beneath TMPDIR when loader
# workers transfer tensors. Linux limits those socket paths to roughly 108
# bytes, while the immutable run/provenance path can be much longer. Keep only
# this small, private socket directory under the short per-user runtime path;
# caches, checkpoints, logs, and all large files remain on the selected data
# volume. The directory is normally empty after workers exit.
MP_TMPDIR="$(mktemp -d "$LOCK_ROOT/lacwm-mp.XXXXXX")"
chmod 700 "$MP_TMPDIR"
cleanup_mp_tmpdir() {
  rmdir -- "$MP_TMPDIR" 2>/dev/null || true
}
trap cleanup_mp_tmpdir EXIT

if ((RESUME == 0)); then
  [[ ! -e "$RUN_DIR" ]] || die "fresh run directory appeared during launch: $RUN_DIR"
fi

# Stage fallible post-lock gates outside RUN_DIR. A transient gate failure leaves
# diagnostics but cannot strand a half-created run that is neither fresh nor resumable.
GATE_DIR="$RUN_ROOT/_prelaunch/${RUN_NAME}_${TIMESTAMP}_pid$$"
mkdir -p "$GATE_DIR/cache" "$GATE_DIR/tmp"
export HF_HOME="$GATE_DIR/cache/huggingface"
export TORCH_HOME="$GATE_DIR/cache/torch"
export XDG_CACHE_HOME="$GATE_DIR/cache/xdg"
export MPLCONFIGDIR="$GATE_DIR/cache/matplotlib"
export TRITON_CACHE_DIR="$GATE_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$GATE_DIR/cache/torchinductor"
export TMPDIR="$GATE_DIR/tmp"

# Re-check exact packages/source/assets and host/data state after taking the lock.
"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR" > "$GATE_DIR/runtime_verification.json"
LOCKED_DATA_REPORT="$GATE_DIR/data_validation_report.json"
LOCKED_SMOKE_REPORT="$GATE_DIR/gradient_smoke_report.json"
cp -- "$DATA_VALIDATION_REPORT" "$LOCKED_DATA_REPORT"
cp -- "$SMOKE_REPORT" "$LOCKED_SMOKE_REPORT"
validate_smoke_report "$LOCKED_SMOKE_REPORT"
assert_repo_identity
"${PREFLIGHT[@]}" \
  --data-validation-report "$LOCKED_DATA_REPORT" \
  --json-out "$GATE_DIR/preflight.json"

STAGED_IDENTITY="$GATE_DIR/run_identity.json"
IDENTITY_TARGET="$IDENTITY_FILE"
if ((RESUME == 0)); then
  IDENTITY_TARGET="$STAGED_IDENTITY"
fi
IDENTITY_ARGS=(
  --identity "$IDENTITY_TARGET"
  --variant "$VARIANT"
  --git-commit "$CURRENT_COMMIT"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
  --run-name "$RUN_NAME"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR"
  --videox-home "$VIDEOX_HOME"
  --data-root "$LACWM_DATA"
  --run-root "$RUN_ROOT"
  --run-dir "$RUN_DIR"
  --wandb-mode "$WANDB_MODE_VALUE"
  --wandb-project "$WANDB_PROJECT_VALUE"
  --wandb-entity "$WANDB_ENTITY_VALUE"
  --data-report "$LOCKED_DATA_REPORT"
  --runtime-report "$GATE_DIR/runtime_verification.json"
  --smoke-report "$LOCKED_SMOKE_REPORT"
)
if ((RESUME == 0)); then
  "$PYTHON_BIN" "$TOOLS_DIR/run_identity.py" create "${IDENTITY_ARGS[@]}"
else
  "$PYTHON_BIN" "$TOOLS_DIR/run_identity.py" validate "${IDENTITY_ARGS[@]}"
fi
export LACWM_RUN_IDENTITY_SHA256="$("$PYTHON_BIN" - "$IDENTITY_TARGET" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("identity_sha256")
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit(f"invalid identity_sha256 in {sys.argv[1]}")
print(value)
PY
)"

if ((RESUME == 0)); then
  [[ ! -e "$RUN_DIR" ]] || die "fresh run directory appeared before identity publication: $RUN_DIR"
  mkdir -p "$(dirname "$RUN_DIR")"
  mkdir "$RUN_DIR" || die "could not exclusively create fresh run directory: $RUN_DIR"
  "$PYTHON_BIN" - "$STAGED_IDENTITY" "$IDENTITY_FILE" <<'PY'
import os
import pathlib
import sys

source, target = map(pathlib.Path, sys.argv[1:])
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(source.read_bytes())
    handle.flush()
    os.fsync(handle.fileno())
PY
else
  [[ -d "$RUN_DIR" ]] || die "resume run directory disappeared: $RUN_DIR"
fi
mkdir -p "$LAUNCH_DIR/wandb" "$LAUNCH_DIR/wandb-data" "$LAUNCH_DIR/wandb-cache" "$LAUNCH_DIR/wandb-config" "$LAUNCH_DIR/cache" "$LAUNCH_DIR/tmp"
cp -- "$GATE_DIR/runtime_verification.json" "$LAUNCH_DIR/runtime_verification.json"
cp -- "$GATE_DIR/preflight.json" "$LAUNCH_DIR/preflight.json"

export WANDB_DIR="$LAUNCH_DIR/wandb"
export WANDB_DATA_DIR="$LAUNCH_DIR/wandb-data"
export WANDB_CACHE_DIR="$LAUNCH_DIR/wandb-cache"
export WANDB_CONFIG_DIR="$LAUNCH_DIR/wandb-config"
export HF_HOME="$LAUNCH_DIR/cache/huggingface"
export TORCH_HOME="$LAUNCH_DIR/cache/torch"
export XDG_CACHE_HOME="$LAUNCH_DIR/cache/xdg"
export MPLCONFIGDIR="$LAUNCH_DIR/cache/matplotlib"
export TRITON_CACHE_DIR="$LAUNCH_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAUNCH_DIR/cache/torchinductor"
export TMPDIR="$MP_TMPDIR"

echo "Validating torchrun + NCCL collectives across all eight selected GPUs..."
timeout --signal=TERM 300s "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "$TOOLS_DIR/env/nccl_probe.py" 2>&1 | tee "$LAUNCH_DIR/nccl_probe.log"

echo "Validating three real-data DDP optimizer updates at the configured batching profile..."
timeout --signal=TERM 1800s "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "$TOOLS_DIR/ddp_training_smoke.py" \
  --variant "$VARIANT" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --steps 3 2>&1 | tee "$LAUNCH_DIR/ddp_training_smoke.log"
{
  printf '#!/usr/bin/env bash\n'
  printf '%s\n' 'echo "This file records provenance only; resume through tools/launch_8xb200.sh." >&2' 'exit 2' ''
  for variable in \
    WAN_DIR VIDEOX_HOME LACWM_DATA LACWM_RUNS LACWM_PYTHON \
    WANDB_MODE WANDB_PROJECT WANDB_ENTITY CUDA_VISIBLE_DEVICES \
    PYTORCH_CUDA_ALLOC_CONF TORCH_NCCL_ASYNC_ERROR_HANDLING PYTHONPATH \
    LACWM_RUN_IDENTITY_SHA256 WANDB_DIR WANDB_DATA_DIR WANDB_CACHE_DIR \
    WANDB_CONFIG_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR \
    TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TMPDIR \
    LACWM_CHECKPOINT_REQUEST_FILE LACWM_CHECKPOINT_ACK_FILE \
    LACWM_SLURM_ATTEMPT_ID; do
    [[ -v "$variable" ]] || continue
    printf 'export %s=%q\n' "$variable" "${!variable}"
  done
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf 'exec'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$LAUNCH_DIR/command.sh"
chmod 600 "$LAUNCH_DIR/command.sh"

"$PYTHON_BIN" "$TOOLS_DIR/capture_provenance.py" \
  --output-dir "$LAUNCH_DIR" \
  --python "$PYTHON_BIN" \
  --kind full-8xb200 \
  --variant "$VARIANT" \
  --data-mode real \
  --run-name "$RUN_NAME" \
  --command-file "$LAUNCH_DIR/command.sh"
cp -- "$LOCKED_SMOKE_REPORT" "$LAUNCH_DIR/accepted_smoke_report.json"
cp -- "$LOCKED_DATA_REPORT" "$LAUNCH_DIR/accepted_data_validation_report.json"

# The qualification step can take many minutes.  Do not start the long run if
# another shell changed branches or edited tracked/untracked code meanwhile.
assert_repo_identity
echo "Launching in foreground. Console log: $LAUNCH_DIR/console.log"
cd "$PROJECT_ROOT"
set +e
"${COMMAND[@]}" 2>&1 | tee "$LAUNCH_DIR/console.log"
PIPE_STATUSES=("${PIPESTATUS[@]}")
STATUS=${PIPE_STATUSES[0]}
TEE_STATUS=${PIPE_STATUSES[1]}
set -e
printf '%s\n' "$STATUS" > "$LAUNCH_DIR/exit_code.txt"
if ((STATUS != 0)); then
  die "training exited with code $STATUS; artifacts preserved at $LAUNCH_DIR"
fi
if ((TEE_STATUS != 0)); then
  die "training completed but console logging failed with code $TEE_STATUS; artifacts at $LAUNCH_DIR"
fi
echo "Training completed successfully; artifacts: $LAUNCH_DIR"
