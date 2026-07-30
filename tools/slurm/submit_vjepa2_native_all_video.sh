#!/usr/bin/env bash
# Dry-run-first launcher for the post-study native-all-video evaluator.

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  submit_vjepa2_native_all_video.sh \
    --study-root ABS --evaluator-commit SHA --python ABS \
    --wan-dir ABS --videox-home ABS --split validation|lockbox \
    --output-dir ABS --log-dir ABS \
    --partition NAME --account NAME --qos NAME [lockbox options] [--execute]

Required lockbox options:
  --validation-report ABS --lockbox-registration ABS

Validation forbids both lockbox options. The launcher dry-runs by default.
Account, QOS, partition, scientific output, and log output are always explicit.
For lockbox, the complete fixed-K=4 composite validation gate is checked before
the registration path is inspected or a job is submitted.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TRAINING_COMMIT="656086686dae723c942a4209a9d71cdb17ed6ccc"
STUDY_ROOT=""
EVALUATOR_COMMIT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
SPLIT=""
OUTPUT_DIR=""
LOG_DIR=""
PARTITION=""
ACCOUNT=""
QOS=""
TIME_LIMIT="06:00:00"
CPUS="160"
MEMORY="1000G"
VALIDATION_REPORT=""
LOCKBOX_REGISTRATION=""
EXECUTE=0

while (($#)); do
  case "$1" in
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --split) SPLIT="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --account) ACCOUNT="${2:?}"; shift 2 ;;
    --qos) QOS="${2:?}"; shift 2 ;;
    --time) TIME_LIMIT="${2:?}"; shift 2 ;;
    --cpus) CPUS="${2:?}"; shift 2 ;;
    --mem) MEMORY="${2:?}"; shift 2 ;;
    --validation-report) VALIDATION_REPORT="${2:?}"; shift 2 ;;
    --lockbox-registration) LOCKBOX_REGISTRATION="${2:?}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$TRAINING_COMMIT" == \
  "656086686dae723c942a4209a9d71cdb17ed6ccc" ]] || \
  die "training commit is not the frozen faithful-cascade commit"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "--evaluator-commit must be a full lowercase SHA-1"
[[ "$SPLIT" == "validation" || "$SPLIT" == "lockbox" ]] || \
  die "--split must be validation or lockbox"
[[ -x "$PYTHON_BIN" ]] || die "--python must be an executable absolute path"
[[ "$PYTHON_BIN" == /* ]] || die "--python must be absolute"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || \
  die "--study-root must be a non-symlink directory"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "--study-root must be canonical and absolute"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || \
  die "--wan-dir must be a non-symlink directory"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "--videox-home must be a Git checkout"
[[ "$OUTPUT_DIR" == /* && ! -e "$OUTPUT_DIR" ]] || \
  die "--output-dir must be an absent absolute path"
[[ -d "$(dirname "$OUTPUT_DIR")" && ! -L "$(dirname "$OUTPUT_DIR")" ]] || \
  die "output parent must already be a non-symlink directory"
[[ "$LOG_DIR" == /* && -d "$LOG_DIR" && ! -L "$LOG_DIR" ]] || \
  die "--log-dir must be an existing non-symlink absolute directory"
for scalar in "$PARTITION" "$ACCOUNT" "$QOS" "$TIME_LIMIT" "$MEMORY"; do
  [[ -n "$scalar" && "$scalar" != *[[:space:]]* ]] || \
    die "partition/account/qos/time/memory must be explicit scalars"
done
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "launcher repository is not at --evaluator-commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "launcher repository must be clean"

EVALUATOR="$REPO_ROOT/tools/vjepa2_native_all_video.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/vjepa2_native_all_video.sbatch"
[[ -x "$EVALUATOR" && -x "$SBATCH_SCRIPT" ]] || \
  die "native evaluator/Slurm wrapper must be executable"

EXTRA_ARGS=()
if [[ "$SPLIT" == "validation" ]]; then
  [[ -z "$VALIDATION_REPORT" && -z "$LOCKBOX_REGISTRATION" ]] || \
    die "validation forbids lockbox arguments"
else
  [[ -n "$VALIDATION_REPORT" && -n "$LOCKBOX_REGISTRATION" ]] || \
    die "lockbox requires --validation-report and --lockbox-registration"
  "$PYTHON_BIN" "$EVALUATOR" check-validation-gate \
    --repo-root "$REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --evaluator-commit "$EVALUATOR_COMMIT" \
    --validation-report "$VALIDATION_REPORT"
  # This is deliberately after check-validation-gate.
  [[ -s "$LOCKBOX_REGISTRATION" && ! -L "$LOCKBOX_REGISTRATION" ]] || \
    die "lockbox registration is unavailable after validation gate"
  EXTRA_ARGS=(
    --validation-report "$VALIDATION_REPORT"
    --lockbox-registration "$LOCKBOX_REGISTRATION"
  )
fi

JOB_NAME="vjepa2-native-all-video-${SPLIT}-${EVALUATOR_COMMIT:0:7}"
SBATCH_COMMAND=(
  sbatch
  --parsable
  --job-name="$JOB_NAME"
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --qos="$QOS"
  --time="$TIME_LIMIT"
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.err"
  "$SBATCH_SCRIPT"
  --repo-root "$REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$EVALUATOR_COMMIT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  "${EXTRA_ARGS[@]}"
)

echo "Native-all-video submission preflight passed."
echo "Split: $SPLIT"
echo "Scientific output: $OUTPUT_DIR"
echo "Explicit account/QOS/partition: $ACCOUNT / $QOS / $PARTITION"
printf 'Command:'
printf ' %q' "${SBATCH_COMMAND[@]}"
printf '\n'
if ((!EXECUTE)); then
  echo "Dry run only; pass --execute to submit."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
JOB_ID="$("${SBATCH_COMMAND[@]}")"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || \
  die "Slurm returned an invalid job ID: $JOB_ID"
echo "Submitted native-all-video $SPLIT job: $JOB_ID"
