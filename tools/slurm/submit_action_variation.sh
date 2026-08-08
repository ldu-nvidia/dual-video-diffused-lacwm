#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/action_variation.sbatch"
REGISTRATION=""
PYTHON_BIN=""
EXPECTED_COMMIT=""
PARTITION=batch
ACCOUNT=""
QOS=""
TIME_LIMIT=04:00:00
EXCLUDE_NODES="pool0-0081,pool0-0089,pool0-0200,pool0-0343"
EXECUTE=0

die() { echo "ERROR: $*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage: submit_action_variation.sh --registration PATH --python PATH
       --expected-commit SHA [--partition NAME] [--account NAME] [--qos NAME]
       [--time HH:MM:SS] [--exclude NODELIST] [--execute]

Dry-run is the default. --execute submits two non-requeueable 8xB200 arm jobs,
then a dependency-gated analysis job. Existing outputs and same-name active jobs
fail closed; this launcher never registers the study or accesses a test split.
Arm jobs exclude pool0-0081,pool0-0089,pool0-0200,pool0-0343 by default;
pass --exclude with a different Slurm node list.
EOF
}
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
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "--registration must be an absolute regular file"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "--python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "--expected-commit is invalid"
[[ "$EXCLUDE_NODES" =~ ^[A-Za-z0-9_,.-]+$ ]] || die "--exclude is invalid"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -f "$SBATCH_SCRIPT" && ! -L "$SBATCH_SCRIPT" ]] || die "sbatch script is missing"

OUTPUT_ROOT="$("$PYTHON_BIN" - "$REGISTRATION" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve(strict=True)
value = json.loads(p.read_text())
root = pathlib.Path(value["output_root"])
if p != (root / "registration.json").resolve(strict=True):
    raise SystemExit("registration/output root mismatch")
print(root)
PY
)"
LOG_DIR="$OUTPUT_ROOT/_slurm_logs"
JOB_NAME="action-variation-$(basename "$OUTPUT_ROOT")"
COMMON=(--mode arm --registration "$REGISTRATION" --repo-root "$REPO_ROOT"
  --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT")
SBATCH_ARGS=(--parsable --job-name "$JOB_NAME" --partition "$PARTITION"
  --time "$TIME_LIMIT" --array 0-1%2 --gpus-per-node 8 --chdir "$LOG_DIR"
  --output "$LOG_DIR/%x-%A_%a.out" --error "$LOG_DIR/%x-%A_%a.err")
[[ -z "$ACCOUNT" ]] || SBATCH_ARGS+=(--account "$ACCOUNT")
[[ -z "$QOS" ]] || SBATCH_ARGS+=(--qos "$QOS")
[[ -z "$EXCLUDE_NODES" ]] || SBATCH_ARGS+=(--exclude "$EXCLUDE_NODES")

echo "Arm submission: sbatch ${SBATCH_ARGS[*]} $SBATCH_SCRIPT ${COMMON[*]}"
echo "Analysis will run only after both arm jobs succeed."
if (( ! EXECUTE )); then
  echo "Dry-run only; pass --execute to submit."
  exit 0
fi
command -v sbatch >/dev/null || die "sbatch is unavailable"
command -v squeue >/dev/null || die "squeue is unavailable"
[[ ! -e "$LOG_DIR" && ! -L "$LOG_DIR" ]] || die "fresh Slurm log directory exists"
if [[ -n "$(squeue -h -u "${USER:?}" -n "$JOB_NAME" -o '%A')" ]]; then
  die "an action-variation job with this immutable name is already active"
fi
mkdir -m 700 "$LOG_DIR"
ARM_JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${COMMON[@]}")"
[[ "$ARM_JOB_ID" =~ ^[0-9]+$ ]] || die "unexpected arm job ID: $ARM_JOB_ID"
ANALYSIS_ARGS=(--parsable --job-name "$JOB_NAME-analysis"
  --dependency "afterok:$ARM_JOB_ID" --partition "$PARTITION" --time 00:30:00
  --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 64G --chdir "$LOG_DIR"
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err")
[[ -z "$ACCOUNT" ]] || ANALYSIS_ARGS+=(--account "$ACCOUNT")
[[ -z "$QOS" ]] || ANALYSIS_ARGS+=(--qos "$QOS")
[[ -z "$EXCLUDE_NODES" ]] || ANALYSIS_ARGS+=(--exclude "$EXCLUDE_NODES")
ANALYSIS_JOB_ID="$(sbatch "${ANALYSIS_ARGS[@]}" "$SBATCH_SCRIPT" \
  --mode analyze --registration "$REGISTRATION" \
  --repo-root "$REPO_ROOT" --python "$PYTHON_BIN" \
  --expected-commit "$EXPECTED_COMMIT")"
[[ "$ANALYSIS_JOB_ID" =~ ^[0-9]+$ ]] || die "unexpected analysis job ID"
echo "Submitted arms=$ARM_JOB_ID analysis=$ANALYSIS_JOB_ID"
