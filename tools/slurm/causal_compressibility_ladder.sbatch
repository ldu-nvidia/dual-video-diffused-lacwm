#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Prospective train-only causal-compressibility ladder. Fit is the only process
# permitted to open clean V-JEPA; both endpoints start as fresh Python processes.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

EXPECTED_COMMIT=""; REPO_ROOT=""; PYTHON_BIN=""; WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""; TRAINING_COMMIT=""; TRAIN_MANIFEST=""; CACHE_METADATA=""
J1_CONFIG=""; J1_SNAPSHOT=""; J1_SHA256=""; J1_ARM=""; J1_STAGE=""; J1_OUTCOME=""
VPM_CONFIG=""; VPM_SNAPSHOT=""; VPM_SHA256=""; VPM_ARM=""; VPM_STAGE=""; VPM_OUTCOME=""
OUTPUT_DIR=""

while (($#)); do
  case "$1" in
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST="${2:?}"; shift 2 ;;
    --cache-metadata) CACHE_METADATA="${2:?}"; shift 2 ;;
    --j1-config) J1_CONFIG="${2:?}"; shift 2 ;;
    --j1-snapshot) J1_SNAPSHOT="${2:?}"; shift 2 ;;
    --j1-sha256) J1_SHA256="${2:?}"; shift 2 ;;
    --j1-arm) J1_ARM="${2:?}"; shift 2 ;;
    --j1-stage) J1_STAGE="${2:?}"; shift 2 ;;
    --j1-outcome) J1_OUTCOME="${2:?}"; shift 2 ;;
    --vpm-config) VPM_CONFIG="${2:?}"; shift 2 ;;
    --vpm-snapshot) VPM_SNAPSHOT="${2:?}"; shift 2 ;;
    --vpm-sha256) VPM_SHA256="${2:?}"; shift 2 ;;
    --vpm-arm) VPM_ARM="${2:?}"; shift 2 ;;
    --vpm-stage) VPM_STAGE="${2:?}"; shift 2 ;;
    --vpm-outcome) VPM_OUTCOME="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "ladder requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "ladder is non-requeueable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "training commit is invalid"
[[ "$J1_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "J1 snapshot hash is invalid"
[[ "$VPM_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "VPM snapshot hash is invalid"
for path in \
  "$REPO_ROOT" "$PYTHON_BIN" "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE" \
  "$TRAIN_MANIFEST" "$CACHE_METADATA" "$J1_CONFIG" "$J1_SNAPSHOT" \
  "$J1_ARM" "$J1_STAGE" "$J1_OUTCOME" "$VPM_CONFIG" "$VPM_SNAPSHOT" \
  "$VPM_ARM" "$VPM_STAGE" "$VPM_OUTCOME" "$OUTPUT_DIR"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || die "repository is noncanonical"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die "output must be fresh"
EXPECTED_BASENAME="causal-compressibility-train256-dev64-seed20260820-${EXPECTED_COMMIT:0:7}-v1"
[[ "$(basename "$OUTPUT_DIR")" == "$EXPECTED_BASENAME" ]] || \
  die "output basename must be $EXPECTED_BASENAME"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
source "$REPO_ROOT/tools/env/activate_b200.sh"

RUNTIME_RECORD="${OUTPUT_DIR}.runtime_verification.json"
[[ ! -e "$RUNTIME_RECORD" ]] || die "runtime record must be fresh"
(set -o noclobber; "$PYTHON_BIN" "$REPO_ROOT/tools/env/verify_b200_runtime.py" \
  --expected-gpus 1 --require-b200 --wan-dir "$WAN_DIR_VALUE" > "$RUNTIME_RECORD")

COMMON=("$PYTHON_BIN" "$REPO_ROOT/tools/causal_compressibility_ladder.py")
"${COMMON[@]}" fit \
  --repo-root "$REPO_ROOT" \
  --expected-source-commit "$EXPECTED_COMMIT" \
  --training-source-commit "$TRAINING_COMMIT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --cache-metadata "$CACHE_METADATA" \
  --j1-resolved-config "$J1_CONFIG" \
  --j1-snapshot "$J1_SNAPSHOT" \
  --j1-snapshot-sha256 "$J1_SHA256" \
  --j1-arm-manifest "$J1_ARM" \
  --j1-stage-manifest "$J1_STAGE" \
  --j1-stage-outcome "$J1_OUTCOME" \
  --vpm-resolved-config "$VPM_CONFIG" \
  --vpm-snapshot "$VPM_SNAPSHOT" \
  --vpm-snapshot-sha256 "$VPM_SHA256" \
  --vpm-arm-manifest "$VPM_ARM" \
  --vpm-stage-manifest "$VPM_STAGE" \
  --vpm-stage-outcome "$VPM_OUTCOME" \
  --output-dir "$OUTPUT_DIR"

# These are intentionally fresh processes; neither accepts a teacher target.
"${COMMON[@]}" evaluate-j1 --output-dir "$OUTPUT_DIR"
"${COMMON[@]}" evaluate-vpm --output-dir "$OUTPUT_DIR"
"${COMMON[@]}" analyze --output-dir "$OUTPUT_DIR"
"${COMMON[@]}" audit --output-dir "$OUTPUT_DIR"
