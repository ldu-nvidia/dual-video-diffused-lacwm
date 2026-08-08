#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/videorepa_trd_workflow.sbatch"
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
usage() {
  cat <<'EOF'
Usage: submit_videorepa_trd_screen.sh --registration PATH --python PATH
       --expected-commit SHA [--partition NAME] [--account NAME] [--qos NAME]
       [--time HH:MM:SS] [--exclude NODELIST] [--execute]

Dry-run is the default. --execute submits a one-clip NFE-1 deployment/equivalence
canary, a production-shape 8xB200 memory canary, matched TRD-OFF/TRD-ON jobs,
a post-training seal, target-free val64 evaluation at NFE 1/2/4 with exact
action-tensor hashes, and final analysis. Every stage is non-requeueable;
validation is dependency-gated after the seal. No protected test path is
accepted. The four known bad nodes are excluded by default.
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
if [[ "$QOS" == short && "$TIME_LIMIT" != 02:00:00 ]]; then
  die "short QOS requires the cluster-compatible --time 02:00:00"
fi
[[ "$EXCLUDE_NODES" =~ ^[A-Za-z0-9_,.-]+$ ]] || die "--exclude is invalid"
for required_node in pool0-0081 pool0-0089 pool0-0200 pool0-0343; do
  [[ ",$EXCLUDE_NODES," == *",$required_node,"* ]] || \
    die "--exclude must retain known bad node $required_node"
done
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -f "$SBATCH_SCRIPT" && ! -L "$SBATCH_SCRIPT" ]] || die "sbatch script missing"

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
JOB_BASE="videorepa-trd-$(basename "$STUDY_ROOT")"
COMMON=(--registration "$REGISTRATION" --repo-root "$REPO_ROOT" \
  --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT")
GPU_BASE=(--parsable --partition "$PARTITION" --account "$ACCOUNT" --qos "$QOS" \
  --time "$TIME_LIMIT" --gpus-per-node 8 --exclude "$EXCLUDE_NODES" \
  --chdir "$LOG_DIR")

echo "Canary -> matched arms -> seal -> val64 arms -> analysis"
echo "Account/QOS: $ACCOUNT/$QOS; bad-node exclusion: $EXCLUDE_NODES"
echo "No validation process starts until both training arms are sealed."
if (( ! EXECUTE )); then
  echo "Dry-run only; pass --execute to submit."
  exit 0
fi
command -v sbatch >/dev/null || die "sbatch is unavailable"
command -v squeue >/dev/null || die "squeue is unavailable"
[[ ! -e "$LOG_DIR" && ! -L "$LOG_DIR" ]] || die "fresh Slurm log directory exists"
while IFS= read -r active_name; do
  case "$active_name" in
    "$JOB_BASE"|"$JOB_BASE-"*)
      die "a same-name TRD workflow stage is already active: $active_name"
      ;;
  esac
done < <(squeue -h -u "${USER:?}" -o '%j')
mkdir -m 700 "$LOG_DIR"

CANARY_ID="$(sbatch "${GPU_BASE[@]}" --job-name "$JOB_BASE-canary" \
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err" \
  "$SBATCH_SCRIPT" --mode canary "${COMMON[@]}")"
[[ "$CANARY_ID" =~ ^[0-9]+$ ]] || die "unexpected canary job ID"

TRAIN_ID="$(sbatch "${GPU_BASE[@]}" --job-name "$JOB_BASE" \
  --dependency "afterok:$CANARY_ID" --array 0-1%2 \
  --output "$LOG_DIR/%x-%A_%a.out" --error "$LOG_DIR/%x-%A_%a.err" \
  "$SBATCH_SCRIPT" --mode train "${COMMON[@]}")"
[[ "$TRAIN_ID" =~ ^[0-9]+$ ]] || die "unexpected training job ID"

SEAL_ID="$(sbatch --parsable --job-name "$JOB_BASE-seal" \
  --dependency "afterok:$TRAIN_ID" --partition "$PARTITION" --account "$ACCOUNT" \
  --qos "$QOS" --time 01:00:00 --nodes 1 --ntasks 1 --cpus-per-task 16 \
  --gpus-per-node 1 \
  --mem 128G --exclude "$EXCLUDE_NODES" --chdir "$LOG_DIR" \
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err" \
  "$SBATCH_SCRIPT" --mode seal "${COMMON[@]}")"
[[ "$SEAL_ID" =~ ^[0-9]+$ ]] || die "unexpected seal job ID"

EVAL_ID="$(sbatch "${GPU_BASE[@]}" --job-name "$JOB_BASE-eval" \
  --dependency "afterok:$SEAL_ID" --array 0-1%2 \
  --output "$LOG_DIR/%x-%A_%a.out" --error "$LOG_DIR/%x-%A_%a.err" \
  "$SBATCH_SCRIPT" --mode evaluate "${COMMON[@]}")"
[[ "$EVAL_ID" =~ ^[0-9]+$ ]] || die "unexpected evaluation job ID"

ANALYSIS_ID="$(sbatch --parsable --job-name "$JOB_BASE-analysis" \
  --dependency "afterok:$EVAL_ID" --partition "$PARTITION" --account "$ACCOUNT" \
  --qos "$QOS" --time 01:00:00 --nodes 1 --ntasks 1 --cpus-per-task 16 \
  --gpus-per-node 1 \
  --mem 128G --exclude "$EXCLUDE_NODES" --chdir "$LOG_DIR" \
  --output "$LOG_DIR/%x-%j.out" --error "$LOG_DIR/%x-%j.err" \
  "$SBATCH_SCRIPT" --mode analyze "${COMMON[@]}")"
[[ "$ANALYSIS_ID" =~ ^[0-9]+$ ]] || die "unexpected analysis job ID"
echo "Submitted canary=$CANARY_ID train=$TRAIN_ID seal=$SEAL_ID eval=$EVAL_ID analysis=$ANALYSIS_ID"
