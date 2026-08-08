#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/tf_training_only_workflow.sbatch"
REGISTRATION=""
PYTHON_BIN=""
EXPECTED_COMMIT=""
PARTITION=batch
ACCOUNT=coreai_chef_posttrain
QOS=short
TIME_LIMIT=02:00:00
EXCLUDE_NODES="pool0-0081,pool0-0089,pool0-0200,pool0-0343"
EXECUTE=0

die() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --account) ACCOUNT="${2:?}"; shift 2 ;;
    --qos) QOS="${2:?}"; shift 2 ;;
    --time) TIME_LIMIT="${2:?}"; shift 2 ;;
    --exclude) EXCLUDE_NODES="${2:?}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "--registration must be an absolute regular file"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "--python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "--expected-commit is invalid"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
for node in pool0-0081 pool0-0089 pool0-0200 pool0-0343; do
  [[ ",$EXCLUDE_NODES," == *",$node,"* ]] || die "known bad node exclusion missing: $node"
done
STUDY_ROOT="$("$PYTHON_BIN" - "$REGISTRATION" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve(strict=True)
v = json.loads(p.read_text())
root = pathlib.Path(v["study_root"])
if p != (root / "registration.json").resolve(strict=True):
    raise SystemExit("registration root mismatch")
print(root)
PY
)"
LOG_DIR="$STUDY_ROOT/_slurm_logs"
JOB_BASE="tfreg-$(basename "$STUDY_ROOT")"
COMMON=(--registration "$REGISTRATION" --repo-root "$REPO_ROOT" \
  --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT")
BASE=(--parsable --partition "$PARTITION" --account "$ACCOUNT" --qos "$QOS" \
  --time "$TIME_LIMIT" --exclude "$EXCLUDE_NODES" --chdir "$LOG_DIR")
echo "Canary -> OFF/ON train -> seal -> target-free val64 -> analysis"
if (( ! EXECUTE )); then
  echo "Dry-run only; pass --execute to submit."
  exit 0
fi
command -v sbatch >/dev/null || die "sbatch unavailable"
[[ ! -e "$LOG_DIR" && ! -L "$LOG_DIR" ]] || die "fresh Slurm log directory exists"
mkdir -m 700 "$LOG_DIR"
CANARY_ID="$(sbatch "${BASE[@]}" --job-name "$JOB_BASE-canary" --gpus-per-node 1 \
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err" \
  "$SBATCH_SCRIPT" --mode canary "${COMMON[@]}")"
TRAIN_ID="$(sbatch "${BASE[@]}" --job-name "$JOB_BASE" --gpus-per-node 8 \
  --dependency "afterok:$CANARY_ID" --array 0-1%2 \
  --output "$LOG_DIR/%x-%A_%a.out" --error "$LOG_DIR/%x-%A_%a.err" \
  "$SBATCH_SCRIPT" --mode train "${COMMON[@]}")"
SEAL_ID="$(sbatch "${BASE[@]}" --job-name "$JOB_BASE-seal" --gpus-per-node 1 \
  --dependency "afterok:$TRAIN_ID" --cpus-per-task 16 --mem 128G \
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err" \
  "$SBATCH_SCRIPT" --mode seal "${COMMON[@]}")"
EVAL_ID="$(sbatch "${BASE[@]}" --job-name "$JOB_BASE-eval" --gpus-per-node 8 \
  --dependency "afterok:$SEAL_ID" --array 0-1%2 \
  --output "$LOG_DIR/%x-%A_%a.out" --error "$LOG_DIR/%x-%A_%a.err" \
  "$SBATCH_SCRIPT" --mode evaluate "${COMMON[@]}")"
ANALYSIS_ID="$(sbatch "${BASE[@]}" --job-name "$JOB_BASE-analysis" --gpus-per-node 1 \
  --dependency "afterok:$EVAL_ID" --cpus-per-task 16 --mem 128G \
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err" \
  "$SBATCH_SCRIPT" --mode analyze "${COMMON[@]}")"
for value in "$CANARY_ID" "$TRAIN_ID" "$SEAL_ID" "$EVAL_ID" "$ANALYSIS_ID"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "unexpected job ID: $value"
done
echo "Submitted canary=$CANARY_ID train=$TRAIN_ID seal=$SEAL_ID eval=$EVAL_ID analysis=$ANALYSIS_ID"
