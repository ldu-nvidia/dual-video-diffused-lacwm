#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/tools"

usage() {
  cat <<'EOF'
Usage: tools/run_gradient_smoke.sh [options]

Required:
  --gpu INDEX             One physical GPU index.
  --variant NAME          latent or explicit.
  --data-mode MODE        real or synthetic. Synthetic loads the real model/weights
                          with deterministic tensors but cannot qualify a full launch.
  --python PATH           Python from the fully provisioned lacwm environment.
  --wan-dir PATH          Wan2.1-Fun-1.3B-Control asset directory.
  --videox-home PATH      VideoX-Fun checkout.
  --data-root PATH        Root containing all four prepared datasets; required only
                          for --data-mode real.
  --run-root PATH         Output root under /mnt/data1, /mnt/data2, or a root
                          explicitly allowed by LACWM_ALLOWED_RUN_ROOTS.
  --wandb-mode disabled   Gradient validation never contacts W&B; this explicit
                          acknowledgement is required.

Optional:
  --run-name NAME         Provenance folder name (default is timestamped).
  --execute               Actually run four forward/backward/update checks.
                          Without this flag, preflight and command preview only.
  -h, --help              Show this help.

The selected GPU must be idle. Jobs on other GPUs are reported but left untouched;
the script never kills, pauses, or modifies another process.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

GPU=""
VARIANT=""
DATA_MODE=""
PYTHON_BIN=""
WAN_PATH=""
VIDEOX_PATH=""
DATA_PATH=""
RUN_ROOT=""
WANDB_MODE_VALUE=""
RUN_NAME=""
EXECUTE=0

while (($#)); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; GPU="$2"; shift 2 ;;
    --variant) [[ $# -ge 2 ]] || die "--variant requires a value"; VARIANT="$2"; shift 2 ;;
    --data-mode) [[ $# -ge 2 ]] || die "--data-mode requires a value"; DATA_MODE="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_PATH="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_PATH="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; DATA_PATH="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || die "--run-root requires a value"; RUN_ROOT="$2"; shift 2 ;;
    --wandb-mode) [[ $# -ge 2 ]] || die "--wandb-mode requires a value"; WANDB_MODE_VALUE="$2"; shift 2 ;;
    --run-name) [[ $# -ge 2 ]] || die "--run-name requires a value"; RUN_NAME="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$GPU" =~ ^[0-9]+$ ]] || die "--gpu must be one non-negative integer"
[[ "$VARIANT" == "latent" || "$VARIANT" == "explicit" ]] || die "--variant must be latent or explicit"
[[ "$DATA_MODE" == "real" || "$DATA_MODE" == "synthetic" ]] || die "--data-mode must be real or synthetic"
[[ -n "$PYTHON_BIN" && -n "$WAN_PATH" && -n "$VIDEOX_PATH" && -n "$RUN_ROOT" ]] || die "--python, --wan-dir, --videox-home, and --run-root are required"
if [[ "$DATA_MODE" == "real" ]]; then
  [[ -n "$DATA_PATH" ]] || die "--data-root is required for --data-mode real"
fi
[[ "$WANDB_MODE_VALUE" == "disabled" ]] || die "gradient validation requires the explicit option: --wandb-mode disabled"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="gradient_smoke_${DATA_MODE}_${VARIANT}_$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-name may contain only letters, digits, dot, underscore, and dash"

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
mapfile -t NORMALIZED_PATHS < <("$PYTHON_BIN" - "$WAN_PATH" "$VIDEOX_PATH" "$DATA_PATH" "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
for value in sys.argv[1:]:
    print("" if not value else Path(value).expanduser().resolve(strict=False))
PY
)
WAN_PATH="${NORMALIZED_PATHS[0]}"
VIDEOX_PATH="${NORMALIZED_PATHS[1]}"
DATA_PATH="${NORMALIZED_PATHS[2]}"
RUN_ROOT="${NORMALIZED_PATHS[3]}"

export WAN_DIR="$WAN_PATH"
export VIDEOX_HOME="$VIDEOX_PATH"
export LACWM_DATA="$DATA_PATH"
export LACWM_RUNS="$RUN_ROOT"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE="disabled"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME:$REPO_ROOT/projects/latent_action_models:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PREFLIGHT=(
  "$PYTHON_BIN" "$TOOLS_DIR/training_preflight.py"
  --profile smoke
  --gpus "$GPU"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR"
  --videox-home "$VIDEOX_HOME"
  --data-root "$LACWM_DATA"
  --run-root "$LACWM_RUNS"
)
if [[ "$DATA_MODE" == "synthetic" ]]; then
  PREFLIGHT+=(--skip-data)
fi

echo "Running guarded one-GPU preflight..."
"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 1 \
  --wan-dir "$WAN_DIR"
"${PREFLIGHT[@]}"

PREVIEW_REPORT="$RUN_ROOT/_launches/$RUN_NAME/gradient_report.json"
LAUNCH_DIR="$RUN_ROOT/_launches/$RUN_NAME"
[[ ! -e "$LAUNCH_DIR" ]] || die "smoke output already exists; choose another --run-name: $LAUNCH_DIR"
COMMAND=("$PYTHON_BIN" "$TOOLS_DIR/gradient_smoke.py" --variant "$VARIANT" --data-mode "$DATA_MODE" --steps 4 --report "$PREVIEW_REPORT")
printf 'Validated command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --execute after reviewing the command."
  exit 0
fi

# This lock is independent of the chosen output root, so two wrappers under the
# same account cannot both pass preflight before either one allocates CUDA.
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

[[ ! -e "$LAUNCH_DIR" ]] || die "smoke output appeared during launch: $LAUNCH_DIR"
GATE_DIR="$RUN_ROOT/_prelaunch/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
mkdir -p "$GATE_DIR/cache" "$GATE_DIR/tmp"

export HF_HOME="$GATE_DIR/cache/huggingface"
export TORCH_HOME="$GATE_DIR/cache/torch"
export XDG_CACHE_HOME="$GATE_DIR/cache/xdg"
export MPLCONFIGDIR="$GATE_DIR/cache/matplotlib"
export TRITON_CACHE_DIR="$GATE_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$GATE_DIR/cache/torchinductor"
export TMPDIR="$GATE_DIR/tmp"

# Close the check-to-launch race as far as possible: re-check after taking the lock.
"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 1 \
  --wan-dir "$WAN_DIR" > "$GATE_DIR/runtime_verification.json"
"${PREFLIGHT[@]}" --json-out "$GATE_DIR/preflight.json"

[[ ! -e "$LAUNCH_DIR" ]] || die "smoke output appeared before publication: $LAUNCH_DIR"
mkdir -p "$RUN_ROOT/_launches"
mkdir "$LAUNCH_DIR" || die "could not exclusively create smoke output: $LAUNCH_DIR"
mkdir -p "$LAUNCH_DIR/cache" "$LAUNCH_DIR/tmp"
cp -- "$GATE_DIR/runtime_verification.json" "$LAUNCH_DIR/runtime_verification.json"
cp -- "$GATE_DIR/preflight.json" "$LAUNCH_DIR/preflight.json"

export HF_HOME="$LAUNCH_DIR/cache/huggingface"
export TORCH_HOME="$LAUNCH_DIR/cache/torch"
export XDG_CACHE_HOME="$LAUNCH_DIR/cache/xdg"
export MPLCONFIGDIR="$LAUNCH_DIR/cache/matplotlib"
export TRITON_CACHE_DIR="$LAUNCH_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAUNCH_DIR/cache/torchinductor"
export TMPDIR="$LAUNCH_DIR/tmp"

REPORT="$LAUNCH_DIR/gradient_report.json"
COMMAND=("$PYTHON_BIN" "$TOOLS_DIR/gradient_smoke.py" --variant "$VARIANT" --data-mode "$DATA_MODE" --steps 4 --report "$REPORT")
{
  printf '#!/usr/bin/env bash\n'
  printf '%s\n' 'echo "This file records provenance only; rerun through tools/run_gradient_smoke.sh." >&2' 'exit 2' ''
  for variable in \
    WAN_DIR VIDEOX_HOME LACWM_DATA LACWM_RUNS LACWM_PYTHON WANDB_MODE \
    CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF PYTHONPATH HF_HOME \
    TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR TRITON_CACHE_DIR \
    TORCHINDUCTOR_CACHE_DIR TMPDIR; do
    printf 'export %s=%q\n' "$variable" "${!variable}"
  done
  printf 'exec'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$LAUNCH_DIR/command.sh"
chmod 600 "$LAUNCH_DIR/command.sh"

"$PYTHON_BIN" "$TOOLS_DIR/capture_provenance.py" \
  --output-dir "$LAUNCH_DIR" \
  --python "$PYTHON_BIN" \
  --kind gradient-smoke \
  --variant "$VARIANT" \
  --data-mode "$DATA_MODE" \
  --run-name "$RUN_NAME" \
  --command-file "$LAUNCH_DIR/command.sh"

echo "Starting gradient validation; no checkpoint will be written."
set +e
"${COMMAND[@]}" 2>&1 | tee "$LAUNCH_DIR/console.log"
PIPE_STATUSES=("${PIPESTATUS[@]}")
STATUS=${PIPE_STATUSES[0]}
TEE_STATUS=${PIPE_STATUSES[1]}
set -e
if ((STATUS != 0)); then
  die "gradient validation failed with exit code $STATUS; see $LAUNCH_DIR"
fi
if ((TEE_STATUS != 0)); then
  die "gradient validation ran but console logging failed with code $TEE_STATUS; see $LAUNCH_DIR"
fi

"$PYTHON_BIN" - "$REPORT" "$DATA_MODE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data_mode = sys.argv[2]
payload = json.loads(path.read_text())
if payload.get("status") != "passed":
    raise SystemExit(f"smoke report did not pass: {path}")
if payload.get("data_mode") != data_mode or payload.get("kind") != f"lacwm_gradient_smoke_{data_mode}":
    raise SystemExit(f"smoke report mode mismatch: {path}")
print(f"PASSED: {path}")
PY
