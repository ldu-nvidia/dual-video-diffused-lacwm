#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
HELPER="$REPO_ROOT/tools/dual_abc_representation_screen.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/dual_abc_representation_screen.sbatch"

LACWM_BASE="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
PYTHON_BIN="$LACWM_BASE/envs/lacwm-b200-py310/bin/python"
WAN_DIR_VALUE="$LACWM_BASE/wan_fun_1.3b_control"
VIDEOX_HOME_VALUE="$LACWM_BASE/VideoX-Fun-1d6d9c3"
DATA_ROOT="$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1"
RUN_ROOT="$LACWM_BASE/runs/dual_video_diffusion/ztf_representation_screen"
CHECKPOINT="$LACWM_BASE/runs/production/lora8n-stage2-fastall4-v1-f227b3b-posttrain-mig1/snapshot.pt"
CHECKPOINT_SHA256="5c132cb5ed6df7b840eef075d13140a73b1c6d5b3d1be4299b01e8365866224b"
WANDB_ENTITY_VALUE="zijiandu"
WANDB_PROJECT_VALUE="dual-video-diffusion-private"

PARTITION="batch"
TIME_LIMIT="04:00:00"
CPUS="160"
MEMORY="1000G"
ACCOUNT=""
QOS=""
SCREEN_ID=""
EXPECTED_COMMIT=""
NO_ZTF_SMOKE_REPORT=""
WITH_ZTF_SMOKE_REPORT=""
MAX_CONCURRENT_ARMS=4
ALLOW_ACTIVE_JOB_IDS=()
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_dual_abc_representation_screen.sh [options]

Preview or submit the seven-arm ABC TF-representation screen as a 0-6 Slurm array:

  0  Parseval RFFT, off,      fixed state scale 0.00
  1  Parseval RFFT, matched,  fixed state scale 0.10
  2  Parseval RFFT, shuffled, fixed state scale 0.10
  3  time packed,   off,      fixed state scale 0.00
  4  time packed,   matched,  fixed state scale 0.10
  5  time packed,   shuffled, fixed state scale 0.10
  6  video only,    off,      state/clock scales fixed 0.00, TF loss 0.00

All arms use the exact same clean commit, reviewed production warm start, ABC
manifest, seed 1234, 200-update optimizer schedule, fixed evaluation noise, and
independent 1/2/4/8-NFE evaluations. Each arm gets one node with eight B200s.
W&B is locked to the owner's private project and has no group.

Options:
  --screen-id ID           Immutable screen ID (default: UTC timestamp + commit)
  --expected-commit SHA    Require this exact 40-character Git commit
  --no-ztf-smoke-report PATH
                           Passing exact-commit real-data dual-no-ztf report
  --with-ztf-smoke-report PATH
                           Passing exact-commit real-data dual-with-ztf report
  --max-concurrent-arms N  Array concurrency in [1,7] (default: 4)
  --allow-active-job-id ID Allow one pre-existing active numeric Slurm job ID.
                           Repeat for each allowed ID; no wildcards or names.
  --partition NAME         Slurm partition (default: batch)
  --time HH:MM:SS          Per-arm limit (default: 04:00:00; batch QoS maximum)
  --cpus N                 CPUs per arm (default: 160)
  --mem VALUE              Memory per arm (default: 1000G)
  --account NAME           Optional Slurm account
  --qos NAME               Optional Slurm QOS
  --execute                Check for active jobs, create immutable provenance,
                           and submit. Without this option the command is dry-run.
  -h, --help               Show this help

This launcher never resumes or requeues. Existing screen and arm directories
fail closed. LACWM uses sigma=1 for noise and sigma=0 for clean data.
EOF
}

while (($#)); do
  case "$1" in
    --screen-id) [[ $# -ge 2 ]] || die "--screen-id requires a value"; SCREEN_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --no-ztf-smoke-report) [[ $# -ge 2 ]] || die "--no-ztf-smoke-report requires a value"; NO_ZTF_SMOKE_REPORT="$2"; shift 2 ;;
    --with-ztf-smoke-report) [[ $# -ge 2 ]] || die "--with-ztf-smoke-report requires a value"; WITH_ZTF_SMOKE_REPORT="$2"; shift 2 ;;
    --max-concurrent-arms) [[ $# -ge 2 ]] || die "--max-concurrent-arms requires a value"; MAX_CONCURRENT_ARMS="$2"; shift 2 ;;
    --allow-active-job-id)
      [[ $# -ge 2 ]] || die "--allow-active-job-id requires a value"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || \
        die "--allow-active-job-id must be a canonical positive decimal integer"
      for existing_job_id in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
        [[ "$existing_job_id" != "$2" ]] || \
          die "--allow-active-job-id may not be repeated for the same ID: $2"
      done
      ALLOW_ACTIVE_JOB_IDS+=("$2")
      shift 2
      ;;
    --partition) [[ $# -ge 2 ]] || die "--partition requires a value"; PARTITION="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || die "--time requires a value"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) [[ $# -ge 2 ]] || die "--cpus requires a value"; CPUS="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "--mem requires a value"; MEMORY="$2"; shift 2 ;;
    --account) [[ $# -ge 2 ]] || die "--account requires a value"; ACCOUNT="$2"; shift 2 ;;
    --qos) [[ $# -ge 2 ]] || die "--qos requires a value"; QOS="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if ((${#ALLOW_ACTIVE_JOB_IDS[@]})); then
  mapfile -t ALLOW_ACTIVE_JOB_IDS < <(
    printf '%s\n' "${ALLOW_ACTIVE_JOB_IDS[@]}" | LC_ALL=C sort -n
  )
fi
ALLOW_ACTIVE_JOB_ARGS=()
for allowed_job_id in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
  ALLOW_ACTIVE_JOB_ARGS+=(--allow-active-job-id "$allowed_job_id")
done

for scalar_pair in \
  "--partition=$PARTITION" \
  "--time=$TIME_LIMIT" \
  "--mem=$MEMORY" \
  "--account=$ACCOUNT" \
  "--qos=$QOS"; do
  scalar_value="${scalar_pair#*=}"
  [[ "$scalar_value" != *[[:space:]]* ]] || \
    die "${scalar_pair%%=*} may not contain whitespace"
done
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$MAX_CONCURRENT_ARMS" =~ ^[1-7]$ ]] || \
  die "--max-concurrent-arms must be an integer from 1 through 7"
[[ "$TIME_LIMIT" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]] || \
  die "--time must use HH:MM:SS with a two-digit hour"
TIME_SECONDS="$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))"
((TIME_SECONDS > 0 && TIME_SECONDS <= 4 * 3600)) || \
  die "--time must be greater than zero and no longer than 04:00:00"
[[ -n "$NO_ZTF_SMOKE_REPORT" && -n "$WITH_ZTF_SMOKE_REPORT" ]] || \
  die "--no-ztf-smoke-report and --with-ztf-smoke-report are required"
[[ "$NO_ZTF_SMOKE_REPORT" == /* && "$WITH_ZTF_SMOKE_REPORT" == /* ]] || \
  die "smoke report paths must be absolute"

for required_path in \
  "$HELPER" \
  "$SBATCH_SCRIPT" \
  "$CHECKPOINT" \
  "$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml" \
  "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml" \
  "$DATA_ROOT/abc_pp/manifest.txt"; do
  [[ -f "$required_path" && ! -L "$required_path" ]] || \
    die "required regular file is missing or symlinked: $required_path"
done
for smoke_report in "$NO_ZTF_SMOKE_REPORT" "$WITH_ZTF_SMOKE_REPORT"; do
  [[ -f "$smoke_report" && ! -L "$smoke_report" ]] || \
    die "smoke report is missing or symlinked: $smoke_report"
done
[[ -f "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || \
  die "Python is not executable: $PYTHON_BIN"
[[ -x "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is not executable: $SBATCH_SCRIPT"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || die "Wan assets are unavailable"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || die "VideoX-Fun checkout is unavailable"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "data root is unavailable"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "could not resolve a full Git commit"
if [[ -n "$EXPECTED_COMMIT" ]]; then
  [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
    die "--expected-commit must be a full lowercase hexadecimal commit"
  [[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
    die "repository is at $ACTUAL_COMMIT, expected $EXPECTED_COMMIT"
else
  EXPECTED_COMMIT="$ACTUAL_COMMIT"
fi
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository must be clean: ${GIT_STATUS//$'\n'/; }"

ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_CHECKPOINT_SHA256" == "$CHECKPOINT_SHA256" ]] || \
  die "production checkpoint changed: $ACTUAL_CHECKPOINT_SHA256 != $CHECKPOINT_SHA256"

"$PYTHON_BIN" "$REPO_ROOT/tools/validate_smoke_report.py" \
  --report "$NO_ZTF_SMOKE_REPORT" \
  --variant dual-no-ztf \
  --git-commit "$EXPECTED_COMMIT" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --data-root "$DATA_ROOT" \
  --warmstart-model "$CHECKPOINT" \
  --warmstart-sha256 "$CHECKPOINT_SHA256"
"$PYTHON_BIN" "$REPO_ROOT/tools/validate_smoke_report.py" \
  --report "$WITH_ZTF_SMOKE_REPORT" \
  --variant dual-with-ztf \
  --git-commit "$EXPECTED_COMMIT" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --data-root "$DATA_ROOT" \
  --warmstart-model "$CHECKPOINT" \
  --warmstart-sha256 "$CHECKPOINT_SHA256"

if [[ -z "$SCREEN_ID" ]]; then
  SCREEN_ID="abc200-tf-representation-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_COMMIT:0:10}"
fi
[[ "$SCREEN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$ ]] || \
  die "--screen-id contains unsafe characters or exceeds 127 characters"

export LACWM_ALLOWED_RUN_ROOTS="$LACWM_BASE"
CANONICAL_RUN_ROOT="$("$PYTHON_BIN" "$REPO_ROOT/tools/run_root_policy.py" --run-root "$RUN_ROOT")"
[[ "$CANONICAL_RUN_ROOT" == "$RUN_ROOT" ]] || \
  die "run root canonicalization changed: $CANONICAL_RUN_ROOT"
SCREEN_ROOT="$RUN_ROOT/$SCREEN_ID"
[[ ! -e "$SCREEN_ROOT" ]] || die "screen root already exists: $SCREEN_ROOT"

"$PYTHON_BIN" "$HELPER" wandb-private \
  --entity "$WANDB_ENTITY_VALUE" \
  --project "$WANDB_PROJECT_VALUE"

LOG_DIR="$RUN_ROOT/_slurm/logs"
JOB_NAME="dual-tf-representation-${SCREEN_ID:0:70}"
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
  --array="0-6%$MAX_CONCURRENT_ARMS"
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="$JOB_NAME"
  --output="$LOG_DIR/%x-%A_%a.out"
  --error="$LOG_DIR/%x-%A_%a.err"
)
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos="$QOS")

JOB_ARGS=(
  --screen-id "$SCREEN_ID"
  --expected-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --screen-root "$SCREEN_ROOT"
  --run-root "$RUN_ROOT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --data-root "$DATA_ROOT"
  --checkpoint "$CHECKPOINT"
  --checkpoint-sha256 "$CHECKPOINT_SHA256"
  --no-ztf-smoke-report "$NO_ZTF_SMOKE_REPORT"
  --with-ztf-smoke-report "$WITH_ZTF_SMOKE_REPORT"
  --wandb-entity "$WANDB_ENTITY_VALUE"
  --wandb-project "$WANDB_PROJECT_VALUE"
)

COMMAND=(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")
printf 'Validated representation-screen sbatch command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Git commit: $EXPECTED_COMMIT (clean)"
echo "Checkpoint: $CHECKPOINT ($CHECKPOINT_SHA256)"
echo "Screen root: $SCREEN_ROOT (fresh)"
for task_id in 0 1 2 3 4 5 6; do
  "$PYTHON_BIN" "$HELPER" arm-contract --array-task-id "$task_id" --format json
done
echo "Schedule: ABC, seed=1234, 200 updates, one sample/GPU, 8xB200 per arm"
echo "Evaluation: independent NFE=[1,2,4,8], noise_seed=20260726"
echo "Inference diagnostics: autonomous, off, oracle_matched, oracle_shuffled (oracle=leakage)"
echo "Clock: sigma=1 noise, sigma=0 clean"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE (PRIVATE), group=null"
echo "Array concurrency: $MAX_CONCURRENT_ARMS"
if ((${#ALLOW_ACTIVE_JOB_IDS[@]})); then
  printf 'Explicitly allowed pre-existing active job IDs:'
  printf ' %s' "${ALLOW_ACTIVE_JOB_IDS[@]}"
  printf '\n'
else
  echo "Explicitly allowed pre-existing active job IDs: <none> (fail closed)"
fi
echo "Raw-RFFT candidate: Slurm 472562 at b1738f9 (not accepted until terminal provenance compatibility is validated)"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --screen-id '$SCREEN_ID' --expected-commit '$EXPECTED_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"

is_allowed_active_job_id() {
  local candidate="$1"
  local allowed
  for allowed in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
    [[ "$candidate" != "$allowed" ]] || return 0
  done
  return 1
}

check_active_user_jobs() {
  local active_records
  if ! active_records="$(
    squeue \
      --noheader \
      --user "${USER:?USER is unset}" \
      --states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED \
      --format='%A|%i|%T|%j'
  )"; then
    die "could not enumerate active user jobs"
  fi

  local rejected_records=()
  local allowed_records=()
  local base_job_id displayed_job_id job_state job_name
  while IFS='|' read -r base_job_id displayed_job_id job_state job_name; do
    [[ -n "$base_job_id" ]] || continue
    [[ "$base_job_id" =~ ^[1-9][0-9]*$ ]] || \
      die "squeue returned a non-numeric base job ID: $base_job_id"
    if is_allowed_active_job_id "$base_job_id"; then
      allowed_records+=(
        "$base_job_id|$displayed_job_id|$job_state|$job_name"
      )
    else
      rejected_records+=(
        "$base_job_id|$displayed_job_id|$job_state|$job_name"
      )
    fi
  done <<< "$active_records"

  if ((${#rejected_records[@]})); then
    local joined_rejected
    printf -v joined_rejected '%s; ' "${rejected_records[@]}"
    die "refusing to submit while non-allow-listed user jobs are active: ${joined_rejected%; }"
  fi
  if ((${#allowed_records[@]})); then
    local joined_allowed
    printf -v joined_allowed '%s; ' "${allowed_records[@]}"
    echo "Pre-existing active jobs accepted only by numeric ID: ${joined_allowed%; }"
  else
    echo "No pre-existing active user jobs observed."
  fi
}

# Check before creating immutable state, then enumerate again immediately before
# sbatch so a newly active, non-allow-listed job cannot slip through the setup.
check_active_user_jobs

mkdir -p "$RUN_ROOT"
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
  die "run root is unavailable or symlinked: $RUN_ROOT"
mkdir -p "$LOG_DIR"
mkdir "$SCREEN_ROOT" || \
  die "could not exclusively create fresh screen root: $SCREEN_ROOT"
SCREEN_MANIFEST="$SCREEN_ROOT/screen_manifest.json"

"$PYTHON_BIN" "$HELPER" create-screen \
  --screen-id "$SCREEN_ID" \
  --git-commit "$EXPECTED_COMMIT" \
  --repo-root "$REPO_ROOT" \
  --run-root "$RUN_ROOT" \
  --screen-root "$SCREEN_ROOT" \
  --python "$PYTHON_BIN" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --no-ztf-smoke-report "$NO_ZTF_SMOKE_REPORT" \
  --with-ztf-smoke-report "$WITH_ZTF_SMOKE_REPORT" \
  --common-config "$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml" \
  --arm-config "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml" \
  --wandb-entity "$WANDB_ENTITY_VALUE" \
  --wandb-project "$WANDB_PROJECT_VALUE" \
  --max-concurrent-arms "$MAX_CONCURRENT_ARMS" \
  "${ALLOW_ACTIVE_JOB_ARGS[@]}" \
  --output "$SCREEN_MANIFEST"

check_active_user_jobs
JOB_ID="$("${COMMAND[@]}")" || die "Slurm rejected the representation-screen submission"
[[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
  die "sbatch returned an unexpected job identifier: $JOB_ID"

"$PYTHON_BIN" "$HELPER" record-submission \
  --screen-manifest "$SCREEN_MANIFEST" \
  --job-id "$JOB_ID" \
  --max-concurrent-arms "$MAX_CONCURRENT_ARMS" \
  "${ALLOW_ACTIVE_JOB_ARGS[@]}" \
  --output "$SCREEN_ROOT/slurm_submission.json"

echo "Submitted representation-screen Slurm array: $JOB_ID"
echo "Logs: $LOG_DIR/$JOB_NAME-${JOB_ID}_<task>.out"
