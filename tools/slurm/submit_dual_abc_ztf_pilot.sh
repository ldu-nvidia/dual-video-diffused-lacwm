#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
HELPER="$REPO_ROOT/tools/dual_abc_pilot.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/dual_abc_ztf_pilot.sbatch"

LACWM_BASE="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
PYTHON_BIN="$LACWM_BASE/envs/lacwm-b200-py310/bin/python"
WAN_DIR_VALUE="$LACWM_BASE/wan_fun_1.3b_control"
VIDEOX_HOME_VALUE="$LACWM_BASE/VideoX-Fun-1d6d9c3"
DATA_ROOT="$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1"
RUN_ROOT="$LACWM_BASE/runs/dual_video_diffusion/ztf_conditioning_pilot"
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
PAIR_ID=""
EXPECTED_COMMIT=""
NO_ZTF_SMOKE_REPORT=""
WITH_ZTF_SMOKE_REPORT=""
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_dual_abc_ztf_pilot.sh [options]

Preview or submit the matched two-arm ABC pilot as a 0-1 Slurm array. The
command is a dry run unless --execute is supplied. Task 0 disables current-TF
conditioning; task 1 enables it. Both tasks use the same committed source,
production warm-start model weights, ABC data, seed, optimizer schedule, eight
B200 GPUs, and private personal W&B project. No W&B group is assigned.

Options:
  --pair-id ID             Immutable pair ID (default: UTC timestamp + commit)
  --expected-commit SHA    Require this exact 40-character Git commit
  --no-ztf-smoke-report PATH
                            Passing real-data dual-no-ztf report
  --with-ztf-smoke-report PATH
                            Passing real-data dual-with-ztf report
  --partition NAME         Slurm partition (default: batch)
  --time HH:MM:SS          Per-arm limit (default: 04:00:00)
  --cpus N                 CPUs per arm (default: 160)
  --mem VALUE              Memory per arm (default: 1000G)
  --account NAME           Optional Slurm account
  --qos NAME               Optional Slurm QOS
  --execute                Check for existing user jobs, create immutable pair
                           provenance, and submit the array
  -h, --help               Show this help

The launcher never resumes or requeues a run. Existing pair/run directories
fail closed. LACWM uses sigma=1 for noise and sigma=0 for clean data.
EOF
}

while (($#)); do
  case "$1" in
    --pair-id) [[ $# -ge 2 ]] || die "--pair-id requires a value"; PAIR_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --no-ztf-smoke-report) [[ $# -ge 2 ]] || die "--no-ztf-smoke-report requires a value"; NO_ZTF_SMOKE_REPORT="$2"; shift 2 ;;
    --with-ztf-smoke-report) [[ $# -ge 2 ]] || die "--with-ztf-smoke-report requires a value"; WITH_ZTF_SMOKE_REPORT="$2"; shift 2 ;;
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
[[ -n "$NO_ZTF_SMOKE_REPORT" && -n "$WITH_ZTF_SMOKE_REPORT" ]] || \
  die "--no-ztf-smoke-report and --with-ztf-smoke-report are required"
[[ "$NO_ZTF_SMOKE_REPORT" == /* && "$WITH_ZTF_SMOKE_REPORT" == /* ]] || \
  die "smoke report paths must be absolute"

for required_path in \
  "$HELPER" \
  "$SBATCH_SCRIPT" \
  "$CHECKPOINT" \
  "$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml" \
  "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_no_ztf_condition.yaml" \
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

if [[ -z "$PAIR_ID" ]]; then
  PAIR_ID="abc100-ztf-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_COMMIT:0:10}"
fi
[[ "$PAIR_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$ ]] || \
  die "--pair-id contains unsafe characters or exceeds 127 characters"

export LACWM_ALLOWED_RUN_ROOTS="$RUN_ROOT"
CANONICAL_RUN_ROOT="$("$PYTHON_BIN" "$REPO_ROOT/tools/run_root_policy.py" --run-root "$RUN_ROOT")"
[[ "$CANONICAL_RUN_ROOT" == "$RUN_ROOT" ]] || \
  die "run root canonicalization changed: $CANONICAL_RUN_ROOT"
PAIR_ROOT="$RUN_ROOT/$PAIR_ID"
[[ ! -e "$PAIR_ROOT" ]] || die "pair root already exists: $PAIR_ROOT"

"$PYTHON_BIN" "$HELPER" wandb-private \
  --entity "$WANDB_ENTITY_VALUE" \
  --project "$WANDB_PROJECT_VALUE"

LOG_DIR="$RUN_ROOT/_slurm/logs"
JOB_NAME="dual-ztf-${PAIR_ID:0:80}"
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
  --array=0-1%2
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
  --pair-id "$PAIR_ID"
  --expected-commit "$EXPECTED_COMMIT"
  --pair-root "$PAIR_ROOT"
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
printf 'Validated matched-pair sbatch command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Git commit: $EXPECTED_COMMIT (clean)"
echo "Checkpoint: $CHECKPOINT ($CHECKPOINT_SHA256)"
echo "Pair root: $PAIR_ROOT (fresh)"
echo "Task 0: no-ztf-condition"
echo "Task 1: with-ztf-condition"
echo "Schedule: ABC, seed=1234, 100 updates, one sample/GPU, 8xB200 per arm"
echo "Clock: sigma=1 noise, sigma=0 clean"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE (PRIVATE), group=null"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --pair-id '$PAIR_ID' --expected-commit '$EXPECTED_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"

# Check once before submitting the pair. Doing this in each array task would
# make the second arm see the first arm and fail a matched submission.
ACTIVE_JOBS="$(
  squeue \
    --noheader \
    --user "${USER:?USER is unset}" \
    --states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED \
    --format='%i|%T|%j'
)"
[[ -z "$ACTIVE_JOBS" ]] || \
  die "refusing to submit while user jobs are active: ${ACTIVE_JOBS//$'\n'/; }"

mkdir -p "$LOG_DIR"
mkdir "$PAIR_ROOT" || die "could not exclusively create fresh pair root: $PAIR_ROOT"
PAIR_MANIFEST="$PAIR_ROOT/pair_manifest.json"

"$PYTHON_BIN" "$HELPER" create-pair \
  --pair-id "$PAIR_ID" \
  --git-commit "$EXPECTED_COMMIT" \
  --repo-root "$REPO_ROOT" \
  --run-root "$RUN_ROOT" \
  --pair-root "$PAIR_ROOT" \
  --python "$PYTHON_BIN" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --no-ztf-smoke-report "$NO_ZTF_SMOKE_REPORT" \
  --with-ztf-smoke-report "$WITH_ZTF_SMOKE_REPORT" \
  --common-config "$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml" \
  --no-ztf-config "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_no_ztf_condition.yaml" \
  --with-ztf-config "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml" \
  --wandb-entity "$WANDB_ENTITY_VALUE" \
  --wandb-project "$WANDB_PROJECT_VALUE" \
  --output "$PAIR_MANIFEST"

JOB_ID="$("${COMMAND[@]}")" || die "Slurm rejected the matched-pair submission"
[[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
  die "sbatch returned an unexpected job identifier: $JOB_ID"

"$PYTHON_BIN" "$HELPER" record-submission \
  --pair-manifest "$PAIR_MANIFEST" \
  --job-id "$JOB_ID" \
  --output "$PAIR_ROOT/slurm_submission.json"

echo "Submitted matched-pair Slurm array: $JOB_ID"
echo "Logs: $LOG_DIR/$JOB_NAME-${JOB_ID}_<task>.out"
