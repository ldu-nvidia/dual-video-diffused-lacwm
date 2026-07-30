#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: vjepa2_paired_latency.sbatch REQUIRED_OPTIONS

Internal one-B200 post-study benchmark. Submit only through
tools/slurm/submit_vjepa2_controlled_study.sh. It compares J1 autonomous NFE=4
with VPM autonomous NFE=8 in one process using counterbalanced paired rounds.

LACWM clock convention: sigma=1 noise, sigma=0 clean.
EOF
}

STUDY_ID=""
EXPECTED_COMMIT=""
REPO_ROOT=""
STUDY_ROOT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
SUBMISSION_RECORD=""

while (($#)); do
  case "$1" in
    --study-id) [[ $# -ge 2 ]] || die "--study-id requires a value"; STUDY_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "--repo-root requires a value"; REPO_ROOT="$2"; shift 2 ;;
    --study-root) [[ $# -ge 2 ]] || die "--study-root requires a value"; STUDY_ROOT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --submission-record) [[ $# -ge 2 ]] || die "--submission-record requires a value"; SUBMISSION_RECORD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "study ID:$STUDY_ID" \
  "expected commit:$EXPECTED_COMMIT" \
  "repository root:$REPO_ROOT" \
  "study root:$STUDY_ROOT" \
  "Python:$PYTHON_BIN" \
  "Wan directory:$WAN_DIR_VALUE" \
  "VideoX-Fun checkout:$VIDEOX_HOME_VALUE" \
  "submission record:$SUBMISSION_RECORD"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "expected commit must be a full lowercase SHA"
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "paired latency requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "paired latency is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "paired latency requires exactly one node"
[[ "$REPO_ROOT" == /* && "$STUDY_ROOT" == /* && "$SUBMISSION_RECORD" == /* ]] || \
  die "repository, study, and submission paths must be absolute"
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "invalid repository root"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository root is not canonical"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || \
  die "study root is missing or symlinked"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "study root is not canonical"
[[ "$STUDY_ROOT" == */"$STUDY_ID" ]] || \
  die "study root basename differs from study ID"

BENCHMARK="$REPO_ROOT/tools/benchmark_vjepa2_paired_latency.py"
ANALYZER="$REPO_ROOT/tools/analyze_vjepa2_controlled_study.py"
HELPER="$REPO_ROOT/tools/vjepa2_controlled_study.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
STUDY_MANIFEST="$STUDY_ROOT/study_manifest.json"
for path in \
  "$BENCHMARK" "$ANALYZER" "$HELPER" "$ACTIVATE" "$VERIFY_RUNTIME" "$STUDY_MANIFEST" \
  "$SUBMISSION_RECORD"; do
  [[ -f "$path" && ! -L "$path" ]] || \
    die "missing or symlinked required file: $path"
done
[[ "$SUBMISSION_RECORD" == "$STUDY_ROOT/slurm_submission.json" ]] || \
  die "submission record must be study_root/slurm_submission.json"
[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime assets are unavailable"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs: $ACTUAL_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || \
  die "repository must be clean: ${GIT_STATUS//$'\n'/; }"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
source "$ACTIVATE"
"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 1 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

ROBOT_WM_ORIGIN="$(
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import robot_wm; print(Path(robot_wm.__file__).resolve())'
)"
case "$ROBOT_WM_ORIGIN" in
  "$REPO_ROOT"/robot_wm/*) ;;
  *) die "Python resolved robot_wm outside the study repository: $ROBOT_WM_ORIGIN" ;;
esac

# Re-hash and semantically revalidate the phase gate, cache build, and all
# immutable study inputs before creating paired-benchmark output.
"$PYTHON_BIN" "$HELPER" validate-study-inputs \
  --study-manifest "$STUDY_MANIFEST" \
  --expected-commit "$EXPECTED_COMMIT" \
  --repo-root "$REPO_ROOT"

OUTPUT_DIR="$STUDY_ROOT/paired_latency"
[[ ! -e "$OUTPUT_DIR" ]] || \
  die "paired-latency output directory already exists: $OUTPUT_DIR"
mkdir "$OUTPUT_DIR"
OUTPUT="$OUTPUT_DIR/paired_j1_nfe4_vs_vpm_nfe8.json"

"$PYTHON_BIN" "$BENCHMARK" \
  --repo-root "$REPO_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --study-root "$STUDY_ROOT" \
  --submission-record "$SUBMISSION_RECORD" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda:0 \
  --output "$OUTPUT"

[[ -s "$OUTPUT" && ! -L "$OUTPUT" ]] || \
  die "paired latency did not produce its immutable evidence record"

RUN_ROOT="$(dirname "$STUDY_ROOT")"
ANALYSIS_PARENT="$RUN_ROOT/_analysis"
mkdir -p "$ANALYSIS_PARENT"
[[ -d "$ANALYSIS_PARENT" && ! -L "$ANALYSIS_PARENT" ]] || \
  die "analysis parent is invalid: $ANALYSIS_PARENT"
ANALYSIS_ROOT="$ANALYSIS_PARENT/$STUDY_ID"
[[ ! -e "$ANALYSIS_ROOT" ]] || \
  die "final analysis output already exists: $ANALYSIS_ROOT"
mkdir "$ANALYSIS_ROOT"
ANALYSIS_JSON="$ANALYSIS_ROOT/analysis.json"
ANALYSIS_MARKDOWN="$ANALYSIS_ROOT/analysis.md"

"$PYTHON_BIN" "$ANALYZER" \
  --study-root "$STUDY_ROOT" \
  --output-json "$ANALYSIS_JSON" \
  --output-markdown "$ANALYSIS_MARKDOWN" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260729 \
  --confidence 0.95

[[ -s "$ANALYSIS_JSON" && ! -L "$ANALYSIS_JSON" ]] || \
  die "final analyzer did not produce JSON evidence"
[[ -s "$ANALYSIS_MARKDOWN" && ! -L "$ANALYSIS_MARKDOWN" ]] || \
  die "final analyzer did not produce its Markdown report"
