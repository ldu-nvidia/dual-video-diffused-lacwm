#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

EXPECTED_COMMIT=""
REPO_ROOT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
TRAINING_COMMIT=""
RESOLVED_CONFIG=""
SNAPSHOT=""
SNAPSHOT_SHA256=""
TRAIN_MANIFEST=""
CACHE_METADATA=""
ARM_MANIFEST=""
STAGE_MANIFEST=""
STAGE_OUTCOME=""
OUTPUT_DIR=""

while (($#)); do
  case "$1" in
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --resolved-config) RESOLVED_CONFIG="${2:?}"; shift 2 ;;
    --snapshot) SNAPSHOT="${2:?}"; shift 2 ;;
    --snapshot-sha256) SNAPSHOT_SHA256="${2:?}"; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST="${2:?}"; shift 2 ;;
    --cache-metadata) CACHE_METADATA="${2:?}"; shift 2 ;;
    --arm-manifest) ARM_MANIFEST="${2:?}"; shift 2 ;;
    --stage-manifest) STAGE_MANIFEST="${2:?}"; shift 2 ;;
    --stage-outcome) STAGE_OUTCOME="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "probe requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "probe is non-requeueable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "training commit is invalid"
[[ "$SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "snapshot hash is invalid"
for path in \
  "$REPO_ROOT" "$PYTHON_BIN" "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE" \
  "$RESOLVED_CONFIG" "$SNAPSHOT" "$TRAIN_MANIFEST" "$CACHE_METADATA" \
  "$ARM_MANIFEST" "$STAGE_MANIFEST" "$STAGE_OUTCOME" "$OUTPUT_DIR"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || die "repository is noncanonical"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die "probe output must be fresh"

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

COMMAND=(
  "$PYTHON_BIN" "$REPO_ROOT/tools/probe_privileged_on_policy_teacher.py" run
  --repo-root "$REPO_ROOT"
  --expected-source-commit "$EXPECTED_COMMIT"
  --training-source-commit "$TRAINING_COMMIT"
  --resolved-config "$RESOLVED_CONFIG"
  --snapshot "$SNAPSHOT"
  --snapshot-sha256 "$SNAPSHOT_SHA256"
  --train-manifest "$TRAIN_MANIFEST"
  --cache-metadata "$CACHE_METADATA"
  --arm-manifest "$ARM_MANIFEST"
  --stage-manifest "$STAGE_MANIFEST"
  --stage-outcome "$STAGE_OUTCOME"
  --output-dir "$OUTPUT_DIR"
  --num-clips 64
  --batch-size 2
  --nfe 4
  --seed 20260808
  --bootstrap-replicates 10000
)
printf 'Launching frozen train-only teacher eligibility probe:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
(cd "$REPO_ROOT"; "${COMMAND[@]}")
"$PYTHON_BIN" "$REPO_ROOT/tools/probe_privileged_on_policy_teacher.py" audit \
  --output-dir "$OUTPUT_DIR"
