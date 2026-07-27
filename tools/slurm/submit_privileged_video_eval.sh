#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
HELPER="$REPO_ROOT/tools/privileged_video_eval_launch.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/privileged_video_eval.sbatch"

LACWM_BASE="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
PYTHON_BIN="$LACWM_BASE/envs/lacwm-b200-py310/bin/python"
PYTHON_LINK_TARGET="/lustre/fsw/portfolios/coreai/users/ldu/lacwm_train/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
PYTHON_REAL_BIN="$LACWM_BASE/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
WAN_DIR_VALUE="$LACWM_BASE/wan_fun_1.3b_control"
VIDEOX_HOME_VALUE="$LACWM_BASE/VideoX-Fun-1d6d9c3"
DATA_ROOT="$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1"
RUN_ROOT="$LACWM_BASE/runs/dual_video_diffusion/privileged_video_eval"
PARENT_BASE="$LACWM_BASE/runs/dual_video_diffusion/ztf_first_cascade_screen/abc200-tf-cascade3-s1234-18318ed-v1"
ABC_MANIFEST="$DATA_ROOT/abc_pp/manifest.txt"
ABC_MANIFEST_SHA256="e52232b49ffec39600aa22e2d708497f22a4ea57fc89f84bc289ae4b1e0a5c09"
SNAPSHOT_BYTES="4249340573"
CORE_COMMIT="af77f8a556cb7204e1aa55b347c123d68b24482b"
WANDB_ENTITY_VALUE="zijiandu"
WANDB_PROJECT_VALUE="dual-video-diffusion-private"
SLURM_ACCOUNT="coreai_chef_posttrain"
SLURM_QOS="normal"

EXPECTED_COMMIT=""
ALLOWED_ACTIVE_JOB_IDS=()
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_privileged_video_eval.sh [options]

Preview or submit one fixed evaluation-only Slurm array. Array tasks 0, 1,
and 2 evaluate trained_off, trained_matched, and trained_shuffled. The fixed
array specification is 0-2%1: at most one task runs concurrently to reduce
inode and scheduler pressure. Each task uses one node, eight B200s, 64 CPUs,
600G, batch, account coreai_chef_posttrain, QOS normal, and 00:30:00.

Options:
  --expected-commit SHA    Require this exact clean 40-character Git commit
  --allow-active-job-id ID Permit one already-active positive numeric Slurm
                           base array/job ID. Repeat for every allowed live ID.
                           Any unlisted job fails closed.
  --execute                Create three fresh immutable output roots and submit.
                           Without this option the command is a dry run.
  -h, --help               Show this help

The three output IDs are bound to the new source commit. Existing output fails
closed. This command never resumes, requeues, stops, or modifies another job.
LACWM uses sigma=1 for noise and sigma=0 for clean data.
EOF
}

while (($#)); do
  case "$1" in
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --allow-active-job-id) [[ $# -ge 2 ]] || die "--allow-active-job-id requires a value"; ALLOWED_ACTIVE_JOB_IDS+=("$2"); shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

declare -A SEEN_ALLOWED_ACTIVE_JOB_IDS=()
for allowed_job_id in "${ALLOWED_ACTIVE_JOB_IDS[@]}"; do
  [[ "$allowed_job_id" =~ ^[1-9][0-9]*$ ]] || \
    die "--allow-active-job-id accepts only positive numeric Slurm IDs"
  [[ -z "${SEEN_ALLOWED_ACTIVE_JOB_IDS[$allowed_job_id]:-}" ]] || \
    die "duplicate --allow-active-job-id: $allowed_job_id"
  SEEN_ALLOWED_ACTIVE_JOB_IDS["$allowed_job_id"]=1
done

for required_path in \
  "$HELPER" "$SBATCH_SCRIPT" \
  "$PROJECT_ROOT/evaluate_privileged_video.py" \
  "$PROJECT_ROOT/configs/train.yaml" \
  "$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml" \
  "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml" \
  "$ABC_MANIFEST"; do
  [[ -f "$required_path" && ! -L "$required_path" ]] || \
    die "required regular file is missing or symlinked: $required_path"
done
for parent_arm in cascade_off_s000 cascade_matched_s010 cascade_shuffled_s010; do
  for parent_file in \
    snapshot.pt resolved_config.yaml arm_manifest.json outcome.json \
    training_complete.json; do
    required_path="$PARENT_BASE/$parent_arm/$parent_file"
    [[ -f "$required_path" && ! -L "$required_path" ]] || \
      die "required parent file is missing or symlinked: $required_path"
  done
done
[[ -L "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || \
  die "pinned virtual-environment Python symlink is unavailable: $PYTHON_BIN"
[[ "$(readlink "$PYTHON_BIN")" == "$PYTHON_LINK_TARGET" ]] || \
  die "pinned Python symlink target changed"
[[ "$(readlink -f "$PYTHON_BIN")" == "$PYTHON_REAL_BIN" ]] || \
  die "pinned Python canonical executable changed"
[[ -f "$PYTHON_REAL_BIN" && ! -L "$PYTHON_REAL_BIN" && -x "$PYTHON_REAL_BIN" ]] || \
  die "pinned canonical Python executable is unavailable"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || \
  die "Wan assets are unavailable"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "VideoX-Fun checkout is unavailable"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "data root is unavailable"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "could not resolve a full Git commit"
if [[ -n "$EXPECTED_COMMIT" ]]; then
  [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
    die "--expected-commit must be a full lowercase hexadecimal commit"
  [[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
    die "repository is at $ACTUAL_COMMIT, expected $EXPECTED_COMMIT"
else
  EXPECTED_COMMIT="$ACTUAL_COMMIT"
fi
[[ "$EXPECTED_COMMIT" != "$CORE_COMMIT" ]] || \
  die "privileged evaluation requires a new immutable source commit"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || \
  die "repository must be clean: ${GIT_STATUS//$'\n'/; }"
git -C "$REPO_ROOT" merge-base --is-ancestor "$CORE_COMMIT" "$EXPECTED_COMMIT" || \
  die "launch commit is not based on privileged core $CORE_COMMIT"

[[ "$(sha256sum "$ABC_MANIFEST" | awk '{print $1}')" == "$ABC_MANIFEST_SHA256" ]] || \
  die "ABC manifest SHA-256 changed"
declare -A EXPECTED_SNAPSHOT_SHA=(
  [cascade_off_s000]="a147acb27dec8fb9f793d665861149ebc8d203b63ab1e6d107760f62d0b36e6b"
  [cascade_matched_s010]="5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d"
  [cascade_shuffled_s010]="1b5e70982d1a93b4069b8ad1c33b25ba1b4d106c560dcaad63e1d2dd23c3eb76"
)
for parent_arm in "${!EXPECTED_SNAPSHOT_SHA[@]}"; do
  checkpoint="$PARENT_BASE/$parent_arm/snapshot.pt"
  [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "${EXPECTED_SNAPSHOT_SHA[$parent_arm]}" ]] || \
    die "$parent_arm checkpoint SHA-256 changed"
  [[ "$(stat -c %s "$checkpoint")" == "$SNAPSHOT_BYTES" ]] || \
    die "$parent_arm checkpoint byte count changed"
done

export LACWM_ALLOWED_RUN_ROOTS="$LACWM_BASE"
CANONICAL_RUN_ROOT="$(
  "$PYTHON_BIN" "$REPO_ROOT/tools/run_root_policy.py" --run-root "$RUN_ROOT"
)"
[[ "$CANONICAL_RUN_ROOT" == "$RUN_ROOT" ]] || \
  die "run root canonicalization changed: $CANONICAL_RUN_ROOT"

"$PYTHON_BIN" "$HELPER" wandb-private \
  --entity "$WANDB_ENTITY_VALUE" \
  --project "$WANDB_PROJECT_VALUE"

ARM_LABELS=()
PARENT_ARMS=()
EVAL_IDS=()
OUTPUT_ROOTS=()
for task_id in 0 1 2; do
  IFS=$'\t' read -r arm_label parent_arm eval_id < <(
    "$PYTHON_BIN" "$HELPER" arm-contract \
      --array-task-id "$task_id" \
      --git-commit "$EXPECTED_COMMIT" \
      --format tsv
  )
  [[ -n "$arm_label" && -n "$parent_arm" && -n "$eval_id" ]] || \
    die "incomplete arm contract for task $task_id"
  output_root="$RUN_ROOT/$eval_id"
  [[ ! -e "$output_root" && ! -L "$output_root" ]] || \
    die "fresh evaluation output already exists: $output_root"
  ARM_LABELS+=("$arm_label")
  PARENT_ARMS+=("$parent_arm")
  EVAL_IDS+=("$eval_id")
  OUTPUT_ROOTS+=("$output_root")
done

LOG_DIR="$RUN_ROOT/_slurm/logs"
JOB_NAME="priv-video-${EXPECTED_COMMIT:0:10}"
SBATCH_ARGS=(
  --parsable
  --array=0-2%1
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task=64
  --mem=600G
  --time=00:30:00
  --partition=batch
  --account="$SLURM_ACCOUNT"
  --qos="$SLURM_QOS"
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="$JOB_NAME"
  --output="$LOG_DIR/%x-%A_%a.out"
  --error="$LOG_DIR/%x-%A_%a.err"
)
JOB_ARGS=(
  --expected-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --run-root "$RUN_ROOT"
)
COMMAND=(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")

printf 'Validated privileged-video evaluation sbatch command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Git commit: $EXPECTED_COMMIT (new, clean, based on $CORE_COMMIT)"
for task_id in 0 1 2; do
  echo "Array task $task_id: ${ARM_LABELS[$task_id]} <- ${PARENT_ARMS[$task_id]}"
  echo "  Output: ${OUTPUT_ROOTS[$task_id]} (fresh)"
done
echo "Resources: array=0-2%1, max concurrency=1, 1 node/task, 8 B200, 00:30:00, no-requeue"
echo "Slurm allocation: partition=batch, account=$SLURM_ACCOUNT, QOS=$SLURM_QOS"
echo "Evaluation: iter=199 after skip=4, NFE=[1,2,4,8], sources=[autonomous,off]"
echo "TF content=false, TF clock=false, aligned video schedule, stage flag=false"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE (PRIVATE), group=null, resume=never"
echo "Clock: sigma=1 noise, sigma=0 clean"
if ((${#ALLOWED_ACTIVE_JOB_IDS[@]} == 0)); then
  echo "Active-job policy: fail closed on every active user job"
else
  echo "Active-job policy: allow only numeric base IDs ${ALLOWED_ACTIVE_JOB_IDS[*]}"
fi

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --expected-commit '$EXPECTED_COMMIT', exact live --allow-active-job-id values, and --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"

ACTIVE_JOB_ROWS=()
check_active_user_jobs() {
  local active_jobs
  active_jobs="$(
    squeue \
      --noheader \
      --user "${USER:?USER is unset}" \
      --format='%F|%i|%T'
  )" || die "could not enumerate active user jobs"
  ACTIVE_JOB_ROWS=()
  if [[ -n "$active_jobs" ]]; then
    mapfile -t ACTIVE_JOB_ROWS <<< "$active_jobs"
  fi
  local unallowed_active_job_rows=()
  local -A observed_allowed_active_job_ids=()
  local row base_id display_id state extra
  for row in "${ACTIVE_JOB_ROWS[@]}"; do
    IFS='|' read -r base_id display_id state extra <<< "$row"
    [[ -z "${extra:-}" && "$base_id" =~ ^[1-9][0-9]*$ ]] || \
      die "squeue returned an unsafe active-job record: $row"
    [[ -n "$display_id" && "$display_id" != *[[:space:]]* ]] || \
      die "squeue returned an unsafe display job ID: $row"
    [[ "$state" =~ ^[A-Z_]+$ ]] || \
      die "squeue returned an unsafe active-job state: $row"
    if [[ -n "${SEEN_ALLOWED_ACTIVE_JOB_IDS[$base_id]:-}" ]]; then
      observed_allowed_active_job_ids["$base_id"]=1
    else
      unallowed_active_job_rows+=("$row")
    fi
  done
  if ((${#unallowed_active_job_rows[@]} > 0)); then
    die "refusing to submit with unlisted active user jobs: ${unallowed_active_job_rows[*]}"
  fi
  for allowed_job_id in "${ALLOWED_ACTIVE_JOB_IDS[@]}"; do
    [[ -n "${observed_allowed_active_job_ids[$allowed_job_id]:-}" ]] || \
      die "allowed active job ID is not active: $allowed_job_id"
  done
}

# First exact coexistence snapshot, before creating any output state.
check_active_user_jobs

mkdir -p "$RUN_ROOT"
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
  die "approved run root is unavailable or symlinked"
mkdir -p "$LOG_DIR"
for task_id in 0 1 2; do
  output_root="${OUTPUT_ROOTS[$task_id]}"
  mkdir "$output_root" || \
    die "could not exclusively create fresh output root: $output_root"
  manifest="$output_root/eval_manifest.json"
  CREATE_ARGS=(
    "$PYTHON_BIN" "$HELPER" create-manifest
    --arm-label "${ARM_LABELS[$task_id]}"
    --eval-id "${EVAL_IDS[$task_id]}"
    --git-commit "$EXPECTED_COMMIT"
    --repo-root "$REPO_ROOT"
    --run-root "$RUN_ROOT"
    --output-root "$output_root"
    --output "$manifest"
  )
  for allowed_job_id in "${ALLOWED_ACTIVE_JOB_IDS[@]}"; do
    CREATE_ARGS+=(--allow-active-job-id "$allowed_job_id")
  done
  for row in "${ACTIVE_JOB_ROWS[@]}"; do
    CREATE_ARGS+=(--observed-active-job "$row")
  done
  "${CREATE_ARGS[@]}"
done

# Any change in live jobs invalidates the exact scope immediately before sbatch.
check_active_user_jobs
IMMEDIATE_PRE_SBATCH_JOB_ROWS=("${ACTIVE_JOB_ROWS[@]}")
ARRAY_JOB_ID="$("${COMMAND[@]}")" || \
  die "Slurm rejected the privileged-video array submission"
[[ "$ARRAY_JOB_ID" =~ ^[1-9][0-9]*$ ]] || \
  die "sbatch returned a non-numeric array base job identifier: $ARRAY_JOB_ID"

for task_id in 0 1 2; do
  output_root="${OUTPUT_ROOTS[$task_id]}"
  RECORD_ARGS=(
    "$PYTHON_BIN" "$HELPER" record-submission
    --manifest "$output_root/eval_manifest.json"
    --array-job-id "$ARRAY_JOB_ID"
    --output "$output_root/slurm_submission.json"
  )
  for row in "${IMMEDIATE_PRE_SBATCH_JOB_ROWS[@]}"; do
    RECORD_ARGS+=(--observed-active-job "$row")
  done
  "${RECORD_ARGS[@]}"
done

echo "Submitted privileged-video evaluation array: $ARRAY_JOB_ID (0-2%1)"
echo "Logs: $LOG_DIR/$JOB_NAME-${ARRAY_JOB_ID}_TASK.out"
