#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Prospective train-only VPM@1 direct-residual frontier. Fit, reserve
# materialization, analysis, and audit are separate fail-closed processes.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

EXPECTED_COMMIT=""; REPO_ROOT=""; PYTHON_BIN=""; WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""; PARENT_REGISTRATION=""; VPM_SHA256=""; OUTPUT_DIR=""

while (($#)); do
  case "$1" in
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --parent-registration) PARENT_REGISTRATION="${2:?}"; shift 2 ;;
    --vpm-sha256) VPM_SHA256="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "frontier requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "frontier is non-requeueable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
[[ "$VPM_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "VPM snapshot hash is invalid"
for path in \
  "$REPO_ROOT" "$PYTHON_BIN" "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE" \
  "$PARENT_REGISTRATION" "$OUTPUT_DIR"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository is noncanonical"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -f "$PARENT_REGISTRATION" && ! -L "$PARENT_REGISTRATION" ]] || \
  die "parent registration is invalid"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die "output must be fresh"
EXPECTED_BASENAME="vpm-direct-residual-fit256-outcome31-seed20260832-${EXPECTED_COMMIT:0:7}-v2"
[[ "$(basename "$OUTPUT_DIR")" == "$EXPECTED_BASENAME" ]] || \
  die "output basename must be $EXPECTED_BASENAME"

VPM_SNAPSHOT="$("$PYTHON_BIN" -c \
  'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(pathlib.Path(json.loads(p.read_text())["inputs"]["vpm_snapshot"]["path"]).resolve(strict=True))' \
  "$PARENT_REGISTRATION")"
[[ "$VPM_SNAPSHOT" == /* && -f "$VPM_SNAPSHOT" && ! -L "$VPM_SNAPSHOT" ]] || \
  die "registered VPM snapshot is invalid"
[[ "$(sha256sum "$VPM_SNAPSHOT" | awk '{print $1}')" == "$VPM_SHA256" ]] || \
  die "registered VPM snapshot SHA-256 differs"

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

COMMON=("$PYTHON_BIN" "$REPO_ROOT/tools/vpm_direct_residual_frontier.py")
"${COMMON[@]}" fit \
  --repo-root "$REPO_ROOT" \
  --expected-source-commit "$EXPECTED_COMMIT" \
  --parent-registration "$PARENT_REGISTRATION" \
  --vpm-snapshot-sha256 "$VPM_SHA256" \
  --runtime-record "$RUNTIME_RECORD" \
  --output-dir "$OUTPUT_DIR"

"${COMMON[@]}" evaluate --output-dir "$OUTPUT_DIR"
"${COMMON[@]}" analyze --output-dir "$OUTPUT_DIR"
"${COMMON[@]}" audit --output-dir "$OUTPUT_DIR"
