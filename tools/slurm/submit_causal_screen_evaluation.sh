#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$REPO_ROOT/tools/evaluate_causal_screen_snapshots.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/evaluate_causal_screen_snapshots.sbatch"

LACWM_BASE="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
PYTHON_BIN="$LACWM_BASE/envs/lacwm-b200-py310/bin/python"
WAN_DIR_VALUE="$LACWM_BASE/wan_fun_1.3b_control"
VIDEOX_HOME_VALUE="$LACWM_BASE/VideoX-Fun-1d6d9c3"
DATA_ROOT="$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1"
ANALYSIS_ROOT="$LACWM_BASE/runs/dual_video_diffusion/analysis"

PARTITION="batch"
TIME_LIMIT="04:00:00"
CPUS="160"
MEMORY="1000G"
ACCOUNT=""
QOS=""
SCREEN_ROOT=""
EVALUATION_ID=""
EXPECTED_EVALUATION_COMMIT=""
BATCHES_PER_RANK=1
ARMS="off_s000,matched_s003,shuffled_s003,matched_s010,shuffled_s010"
ALLOW_ACTIVE_JOB_IDS=()
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_causal_screen_evaluation.sh [options]

Validate or submit a fresh eight-B200 post-training evaluation of completed
causal-screen snapshots. The evaluator caches deterministic rank-sharded ABC
batches once and replays them across every selected arm. Every selected arm
always emits autonomous/off/oracle-matched/oracle-shuffled and the
same-checkpoint autonomous-shuffled control at NFE 1/2/4/8.

Options:
  --screen-root PATH       Completed five-arm causal-screen root (required)
  --evaluation-id ID       Fresh analysis ID (default: UTC time + commit)
  --expected-evaluation-commit SHA
                           Exact clean corrected commit (default: current HEAD)
  --batches-per-rank N     Deterministic batches per rank, 1-4 (default: 1)
  --arms CSV               At least two arm names (default: all five)
  --partition NAME         Slurm partition (default: batch)
  --time HH:MM:SS          Limit (default/max intended: 04:00:00)
  --cpus N                 CPUs (default: 160)
  --mem VALUE              Memory (default: 1000G)
  --account NAME           Optional Slurm account
  --qos NAME               Optional Slurm QOS
  --allow-active-job-id NUMERIC_ID
                           Repeat to allow exact active Slurm job IDs. The
                           allowlist must exactly equal all active user job
                           IDs at the submission gate. No wildcard or job-name
                           bypass is supported; default is an empty allowlist.
  --execute                Enforce the active-job gate, create a fresh
                           evaluation manifest, and submit. Omit for a
                           read-only dry run.
  -h, --help               Show this help

Safe staged examples:
  Full screen, n=8:
    --batches-per-rank 1
  Fresh high-power s010 comparison, n=32:
    --batches-per-rank 4 --arms matched_s010,shuffled_s010

The launcher never resumes, requeues, writes beneath source arms, or enables
W&B. Existing evaluation roots fail closed.
EOF
}

while (($#)); do
  case "$1" in
    --screen-root) [[ $# -ge 2 ]] || die "--screen-root requires a value"; SCREEN_ROOT="$2"; shift 2 ;;
    --evaluation-id) [[ $# -ge 2 ]] || die "--evaluation-id requires a value"; EVALUATION_ID="$2"; shift 2 ;;
    --expected-evaluation-commit) [[ $# -ge 2 ]] || die "--expected-evaluation-commit requires a value"; EXPECTED_EVALUATION_COMMIT="$2"; shift 2 ;;
    --batches-per-rank) [[ $# -ge 2 ]] || die "--batches-per-rank requires a value"; BATCHES_PER_RANK="$2"; shift 2 ;;
    --arms) [[ $# -ge 2 ]] || die "--arms requires a value"; ARMS="$2"; shift 2 ;;
    --partition) [[ $# -ge 2 ]] || die "--partition requires a value"; PARTITION="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || die "--time requires a value"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) [[ $# -ge 2 ]] || die "--cpus requires a value"; CPUS="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "--mem requires a value"; MEMORY="$2"; shift 2 ;;
    --account) [[ $# -ge 2 ]] || die "--account requires a value"; ACCOUNT="$2"; shift 2 ;;
    --qos) [[ $# -ge 2 ]] || die "--qos requires a value"; QOS="$2"; shift 2 ;;
    --allow-active-job-id) [[ $# -ge 2 ]] || die "--allow-active-job-id requires a value"; ALLOW_ACTIVE_JOB_IDS+=("$2"); shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$SCREEN_ROOT" ]] || die "--screen-root is required"
[[ "$SCREEN_ROOT" == /* ]] || die "--screen-root must be absolute"
[[ "$BATCHES_PER_RANK" =~ ^[1-4]$ ]] || \
  die "--batches-per-rank must be 1, 2, 3, or 4"
for scalar_pair in \
  "--partition=$PARTITION" \
  "--time=$TIME_LIMIT" \
  "--mem=$MEMORY" \
  "--account=$ACCOUNT" \
  "--qos=$QOS" \
  "--arms=$ARMS"; do
  scalar_value="${scalar_pair#*=}"
  [[ "$scalar_value" != *[[:space:]]* ]] || \
    die "${scalar_pair%%=*} may not contain whitespace"
done
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"

declare -A SEEN_ALLOWED_ACTIVE_JOB_IDS=()
for job_id in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || \
    die "--allow-active-job-id must be a positive numeric Slurm job ID"
  [[ -z "${SEEN_ALLOWED_ACTIVE_JOB_IDS[$job_id]+present}" ]] || \
    die "--allow-active-job-id may not repeat job ID $job_id"
  SEEN_ALLOWED_ACTIVE_JOB_IDS["$job_id"]=1
done
if ((${#ALLOW_ACTIVE_JOB_IDS[@]})); then
  mapfile -t ALLOW_ACTIVE_JOB_IDS < <(
    printf '%s\n' "${ALLOW_ACTIVE_JOB_IDS[@]}" | sort -n
  )
fi

for required_file in \
  "$HELPER" \
  "$SBATCH_SCRIPT" \
  "$SCREEN_ROOT/screen_manifest.json" \
  "$DATA_ROOT/abc_pp/manifest.txt"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] || \
    die "required regular file is missing or symlinked: $required_file"
done
[[ -x "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is not executable"
[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository root is unavailable or symlinked"
[[ -d "$SCREEN_ROOT" && ! -L "$SCREEN_ROOT" ]] || \
  die "screen root is unavailable or symlinked"
[[ -d "$ANALYSIS_ROOT" && ! -L "$ANALYSIS_ROOT" ]] || \
  die "analysis root is unavailable or symlinked"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || \
  die "Wan assets are unavailable or symlinked"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || die "VideoX-Fun checkout is unavailable"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || \
  die "data root is unavailable or symlinked"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "could not resolve a full Git commit"
if [[ -n "$EXPECTED_EVALUATION_COMMIT" ]]; then
  [[ "$EXPECTED_EVALUATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
    die "--expected-evaluation-commit must be a full lowercase commit"
  [[ "$ACTUAL_COMMIT" == "$EXPECTED_EVALUATION_COMMIT" ]] || \
    die "repository is at $ACTUAL_COMMIT, expected $EXPECTED_EVALUATION_COMMIT"
else
  EXPECTED_EVALUATION_COMMIT="$ACTUAL_COMMIT"
fi
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || \
  die "evaluation repository must be clean: ${GIT_STATUS//$'\n'/; }"

if [[ -z "$EVALUATION_ID" ]]; then
  SCREEN_BASENAME="$(basename "$SCREEN_ROOT")"
  EVALUATION_ID="posthoc-${SCREEN_BASENAME:0:45}-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_EVALUATION_COMMIT:0:10}"
fi
[[ "$EVALUATION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$ ]] || \
  die "--evaluation-id contains unsafe characters or exceeds 127 characters"
EVALUATION_ROOT="$ANALYSIS_ROOT/$EVALUATION_ID"
[[ ! -e "$EVALUATION_ROOT" ]] || \
  die "evaluation root already exists: $EVALUATION_ROOT"

MANIFEST_ARGS=(
  --evaluation-id "$EVALUATION_ID"
  --expected-evaluation-commit "$EXPECTED_EVALUATION_COMMIT"
  --repo-root "$REPO_ROOT"
  --screen-root "$SCREEN_ROOT"
  --analysis-root "$ANALYSIS_ROOT"
  --data-root "$DATA_ROOT"
  --batches-per-rank "$BATCHES_PER_RANK"
  --arms "$ARMS"
)
for job_id in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
  MANIFEST_ARGS+=(--allow-active-job-id "$job_id")
done

LOG_DIR="$ANALYSIS_ROOT/_slurm/logs"
JOB_NAME="dual-tf-posthoc-${EVALUATION_ID:0:65}"
SBATCH_ARGS=(
  --parsable
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --partition="$PARTITION"
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="$JOB_NAME"
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.err"
)
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos="$QOS")

EVALUATION_MANIFEST="$EVALUATION_ROOT/evaluation_manifest.json"
JOB_ARGS=(
  --expected-evaluation-commit "$EXPECTED_EVALUATION_COMMIT"
  --repo-root "$REPO_ROOT"
  --evaluation-root "$EVALUATION_ROOT"
  --evaluation-manifest "$EVALUATION_MANIFEST"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --data-root "$DATA_ROOT"
)
COMMAND=(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")

printf 'Validated posthoc evaluation sbatch command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Screen root: $SCREEN_ROOT (read-only)"
echo "Evaluation root: $EVALUATION_ROOT (fresh, outside screen)"
echo "Evaluation commit: $EXPECTED_EVALUATION_COMMIT (clean, corrected)"
echo "Arms: $ARMS"
echo "Paired units: 8 ranks x $BATCHES_PER_RANK batch(es)"
echo "Sources: autonomous, off, oracle_matched, oracle_shuffled, autonomous_shuffled"
echo "NFE: 1, 2, 4, 8"
echo "Requested active-job allowlist: ${ALLOW_ACTIVE_JOB_IDS[*]:-(empty)}"
echo "W&B: disabled; resume=false; requeue=false"

if ((EXECUTE == 0)); then
  "$PYTHON_BIN" "$HELPER" validate "${MANIFEST_ARGS[@]}"
  echo "Dry run only. Re-run with --evaluation-id '$EVALUATION_ID' --expected-evaluation-commit '$EXPECTED_EVALUATION_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"

# Enumerate every active job by Slurm's numeric base allocation ID. Array
# tasks may repeat the same %F and are deliberately deduplicated. Submission
# proceeds only when this observed set exactly equals the explicit allowlist.
# Existing jobs are inspected read-only and are never modified.
OBSERVED_ACTIVE_JOB_IDS=()
assert_active_job_allowlist() {
  local active_job_output job_id
  active_job_output="$(
    squeue \
      --noheader \
      --user "${USER:?USER is unset}" \
      --format='%F'
  )" || die "could not enumerate active user jobs"
  declare -A seen_active_job_ids=()
  while IFS= read -r job_id; do
    [[ -n "$job_id" ]] || continue
    [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || \
      die "squeue returned a non-numeric active base job ID: $job_id"
    seen_active_job_ids["$job_id"]=1
  done <<<"$active_job_output"
  OBSERVED_ACTIVE_JOB_IDS=()
  if ((${#seen_active_job_ids[@]})); then
    mapfile -t OBSERVED_ACTIVE_JOB_IDS < <(
      printf '%s\n' "${!seen_active_job_ids[@]}" | sort -n
    )
  fi
  [[ "${OBSERVED_ACTIVE_JOB_IDS[*]-}" == "${ALLOW_ACTIVE_JOB_IDS[*]-}" ]] || \
    die "active user job IDs do not exactly match --allow-active-job-id: observed=[${OBSERVED_ACTIVE_JOB_IDS[*]-}] allowed=[${ALLOW_ACTIVE_JOB_IDS[*]-}]"
}
assert_active_job_allowlist
echo "Active-job coexistence gate passed: [${OBSERVED_ACTIVE_JOB_IDS[*]-}]"

"$PYTHON_BIN" "$HELPER" prepare "${MANIFEST_ARGS[@]}"
[[ -f "$EVALUATION_MANIFEST" && ! -L "$EVALUATION_MANIFEST" ]] || \
  die "evaluation manifest was not created safely"
mkdir -p "$LOG_DIR"
[[ -d "$LOG_DIR" && ! -L "$LOG_DIR" ]] || \
  die "Slurm log directory is unavailable or symlinked"

# Source validation may be nontrivial, so close the race window by requiring
# the exact same active-job set again immediately before calling sbatch.
assert_active_job_allowlist
echo "Immediate pre-sbatch active-job gate passed: [${OBSERVED_ACTIVE_JOB_IDS[*]-}]"

JOB_ID="$("${COMMAND[@]}")" || die "Slurm rejected the posthoc evaluation"
[[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
  die "sbatch returned an unexpected job identifier: $JOB_ID"

"$PYTHON_BIN" "$HELPER" record-submission \
  --evaluation-root "$EVALUATION_ROOT" \
  --evaluation-manifest "$EVALUATION_MANIFEST" \
  --job-id "$JOB_ID"

echo "Submitted posthoc evaluation: $JOB_ID"
echo "Logs: $LOG_DIR/$JOB_NAME-${JOB_ID}.out"
