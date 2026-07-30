#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: vjepa2_phase_gate.sbatch REQUIRED_OPTIONS

Run the dedicated eight-rank, one-node B200 J0 phase gate. This entrypoint
does not initialize W&B, write a model checkpoint, or invoke the V-JEPA
teacher. The output directory must be fresh and outside the Git repository.

Required:
  --expected-commit SHA
  --repo-root PATH
  --project-root PATH
  --output-dir PATH
  --python PATH
  --wan-dir PATH
  --videox-home PATH
  --cache-build-complete PATH
  --train-manifest PATH
  --train-cache-metadata PATH
  --pca PATH
  --warmstart PATH
  --warmstart-sha256 HEX

LACWM clock convention: sigma=1 noise, sigma=0 clean.
EOF
}

EXPECTED_COMMIT=""
REPO_ROOT=""
PROJECT_ROOT=""
OUTPUT_DIR=""
PYTHON_BIN=""
PYTHON_REAL_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
CACHE_BUILD_COMPLETE=""
TRAIN_MANIFEST=""
TRAIN_CACHE_METADATA=""
PCA_PATH=""
WARMSTART=""
WARMSTART_SHA256=""

while (($#)); do
  case "$1" in
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "--repo-root requires a value"; REPO_ROOT="$2"; shift 2 ;;
    --project-root) [[ $# -ge 2 ]] || die "--project-root requires a value"; PROJECT_ROOT="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || die "--output-dir requires a value"; OUTPUT_DIR="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --cache-build-complete) [[ $# -ge 2 ]] || die "--cache-build-complete requires a value"; CACHE_BUILD_COMPLETE="$2"; shift 2 ;;
    --train-manifest) [[ $# -ge 2 ]] || die "--train-manifest requires a value"; TRAIN_MANIFEST="$2"; shift 2 ;;
    --train-cache-metadata) [[ $# -ge 2 ]] || die "--train-cache-metadata requires a value"; TRAIN_CACHE_METADATA="$2"; shift 2 ;;
    --pca) [[ $# -ge 2 ]] || die "--pca requires a value"; PCA_PATH="$2"; shift 2 ;;
    --warmstart) [[ $# -ge 2 ]] || die "--warmstart requires a value"; WARMSTART="$2"; shift 2 ;;
    --warmstart-sha256) [[ $# -ge 2 ]] || die "--warmstart-sha256 requires a value"; WARMSTART_SHA256="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "expected commit:$EXPECTED_COMMIT" \
  "repository root:$REPO_ROOT" \
  "project root:$PROJECT_ROOT" \
  "output directory:$OUTPUT_DIR" \
  "Python:$PYTHON_BIN" \
  "Wan directory:$WAN_DIR_VALUE" \
  "VideoX-Fun checkout:$VIDEOX_HOME_VALUE" \
  "cache completion record:$CACHE_BUILD_COMPLETE" \
  "training manifest:$TRAIN_MANIFEST" \
  "training cache metadata:$TRAIN_CACHE_METADATA" \
  "PCA artifact:$PCA_PATH" \
  "warm start:$WARMSTART" \
  "warm-start SHA-256:$WARMSTART_SHA256"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "expected commit must be a full lowercase SHA-1"
[[ "$WARMSTART_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "warm-start SHA-256 must be 64 lowercase hexadecimal characters"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires a Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "the V-JEPA phase gate is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "the V-JEPA phase gate requires exactly one node"

for path in "$REPO_ROOT" "$PROJECT_ROOT" "$OUTPUT_DIR"; do
  [[ "$path" == /* ]] || die "repository, project, and output paths must be absolute"
done
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || die "invalid repository root"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository root is not canonical"
[[ -d "$PROJECT_ROOT" && ! -L "$PROJECT_ROOT" ]] || die "invalid project root"
[[ "$(cd "$PROJECT_ROOT" && pwd -P)" == "$PROJECT_ROOT" ]] || \
  die "project root is not canonical"
[[ "$PROJECT_ROOT" == "$REPO_ROOT/projects/latent_action_models" ]] || \
  die "project root is not the repository LACWM project"
[[ ! -e "$OUTPUT_DIR" ]] || die "phase-gate output already exists: $OUTPUT_DIR"
[[ -d "$(dirname "$OUTPUT_DIR")" && ! -L "$(dirname "$OUTPUT_DIR")" ]] || \
  die "phase-gate output parent is unavailable or symlinked"
case "$OUTPUT_DIR/" in
  "$REPO_ROOT/"*) die "phase-gate output must be outside the Git repository" ;;
esac

PHASE_GATE="$REPO_ROOT/tools/vjepa2_phase_gate.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in \
  "$PHASE_GATE" "$ACTIVATE" "$VERIFY_RUNTIME" \
  "$CACHE_BUILD_COMPLETE" "$TRAIN_MANIFEST" "$TRAIN_CACHE_METADATA" \
  "$PCA_PATH" "$WARMSTART"; do
  [[ -f "$path" && ! -L "$path" ]] || \
    die "missing or symlinked required file: $path"
done
[[ "$PYTHON_BIN" == /* ]] || die "Python path must be absolute"
[[ -f "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || \
  die "Python is unavailable or non-executable"
# Keep the venv launcher path for execution. Invoking its resolved base-CPython
# target bypasses pyvenv.cfg and therefore loses the pinned site-packages.
PYTHON_REAL_BIN="$(readlink -f -- "$PYTHON_BIN")" || \
  die "failed to resolve Python executable: $PYTHON_BIN"
[[ -n "$PYTHON_REAL_BIN" && -f "$PYTHON_REAL_BIN" && \
   -x "$PYTHON_REAL_BIN" && ! -L "$PYTHON_REAL_BIN" ]] || \
  die "resolved Python is not a canonical regular executable"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || die "invalid Wan directory"
[[ -d "$VIDEOX_HOME_VALUE/.git" && ! -L "$VIDEOX_HOME_VALUE" ]] || \
  die "invalid VideoX-Fun checkout"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit changed: $ACTUAL_COMMIT != $EXPECTED_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository is dirty: ${GIT_STATUS//$'\n'/; }"
[[ "$(sha256sum "$WARMSTART" | awk '{print $1}')" == "$WARMSTART_SHA256" ]] || \
  die "warm-start SHA-256 changed"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "$GPU_COUNT" == "8" ]] || die "phase gate expected 8 GPUs, found $GPU_COUNT"
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -vi 'B200' >/dev/null; then
  die "phase gate requires an all-B200 node"
fi

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export LACWM_NNODES=1
export LACWM_WORLD_SIZE=8
export LACWM_GPUS_PER_NODE=8
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset WANDB_API_KEY WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_TAGS
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source "$ACTIVATE"

# This writes only runtime facts to stdout; the immutable phase-gate report
# records the same per-rank topology after NCCL initialization.
"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node=8
  "$PHASE_GATE"
  --repo-root "$REPO_ROOT"
  --project-root "$PROJECT_ROOT"
  --expected-commit "$EXPECTED_COMMIT"
  --output-dir "$OUTPUT_DIR"
  --cache-build-complete "$CACHE_BUILD_COMPLETE"
  --train-manifest "$TRAIN_MANIFEST"
  --train-cache-metadata "$TRAIN_CACHE_METADATA"
  --pca "$PCA_PATH"
  --warmstart "$WARMSTART"
  --warmstart-sha256 "$WARMSTART_SHA256"
)

echo "V-JEPA 2.1 phase gate: real J0, 8-rank DDP, four optimizer updates"
echo "Inference gate: public future-only sampler, autonomous NFE=1"
echo "Teacher calls: training=0 inference=0"
echo "W&B: disabled; checkpoints: disabled"
echo "Clock: sigma=1 noise, sigma=0 clean"
echo "Python launcher: $PYTHON_BIN (resolved binary: $PYTHON_REAL_BIN)"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

(
  cd "$PROJECT_ROOT"
  "${COMMAND[@]}"
)

REPORT="$OUTPUT_DIR/phase_gate_report.json"
[[ -f "$REPORT" && ! -L "$REPORT" ]] || die "phase gate did not publish its report"
"$PYTHON_BIN" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("artifact_type") != "vjepa2-j0-phase-gate":
    raise SystemExit("unexpected phase-gate report type")
if report.get("passed") is not True:
    raise SystemExit("V-JEPA phase gate did not pass")
if report.get("world_size") != 8:
    raise SystemExit("phase-gate report world size is not eight")
if report.get("training", {}).get("optimizer_updates") != 4:
    raise SystemExit("phase-gate report lacks four optimizer updates")
if report.get("teacher_calls_training") != 0 or report.get("teacher_calls_inference") != 0:
    raise SystemExit("phase-gate report records a teacher call")
PY

echo "V-JEPA J0 phase gate passed: $REPORT"
