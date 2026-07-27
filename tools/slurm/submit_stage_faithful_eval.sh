#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
HELPER="$REPO_ROOT/tools/stage_faithful_eval_launch.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/stage_faithful_eval.sbatch"

LACWM_BASE="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
PYTHON_BIN="$LACWM_BASE/envs/lacwm-b200-py310/bin/python"
WAN_DIR_VALUE="$LACWM_BASE/wan_fun_1.3b_control"
VIDEOX_HOME_VALUE="$LACWM_BASE/VideoX-Fun-1d6d9c3"
DATA_ROOT="$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1"
RUN_ROOT="$LACWM_BASE/runs/dual_video_diffusion/ztf_first_cascade_stage_eval"
PARENT_ROOT="$LACWM_BASE/runs/dual_video_diffusion/ztf_first_cascade_screen/abc200-tf-cascade3-s1234-18318ed-v1/cascade_matched_s010"
CHECKPOINT="$PARENT_ROOT/snapshot.pt"
CHECKPOINT_SHA256="5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d"
CHECKPOINT_BYTES="4249340573"
ABC_MANIFEST="$DATA_ROOT/abc_pp/manifest.txt"
ABC_MANIFEST_SHA256="e52232b49ffec39600aa22e2d708497f22a4ea57fc89f84bc289ae4b1e0a5c09"
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
Usage: tools/slurm/submit_stage_faithful_eval.sh [options]

Preview or submit one evaluation-only, non-array, non-requeueable Slurm job.
The job uses one node, eight B200s, 64 CPUs, 600G, batch, account
coreai_chef_posttrain, QOS normal, and 00:30:00.

Options:
  --expected-commit SHA    Require this exact clean 40-character Git commit
  --allow-active-job-id ID Permit one already-active positive numeric Slurm
                           array/job ID. Repeat for every allowed live ID.
                           Any unlisted job fails closed.
  --execute                Create fresh immutable provenance and submit.
                           Without this option the command is a dry run.
  -h, --help               Show this help

The output is fixed to the approved Lustre root and the evaluation ID is
`abc200-tf-cascade-stage-eval-s1234-${COMMIT:0:10}-v1`. Existing output fails
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
  "$PROJECT_ROOT/evaluate_stage_faithful.py" \
  "$PROJECT_ROOT/configs/train.yaml" \
  "$PROJECT_ROOT/configs/experiments_0908/dual_abc_ztf_common.yaml" \
  "$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml" \
  "$CHECKPOINT" "$ABC_MANIFEST" \
  "$PARENT_ROOT/resolved_config.yaml" "$PARENT_ROOT/arm_manifest.json" \
  "$PARENT_ROOT/outcome.json" "$PARENT_ROOT/training_complete.json"; do
  [[ -f "$required_path" && ! -L "$required_path" ]] || \
    die "required regular file is missing or symlinked: $required_path"
done
[[ -f "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || \
  die "pinned Python is not executable: $PYTHON_BIN"
[[ -x "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is not executable"
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
[[ -z "$GIT_STATUS" ]] || \
  die "repository must be clean: ${GIT_STATUS//$'\n'/; }"
git -C "$REPO_ROOT" merge-base --is-ancestor \
  5880d047347ee572fcbdd6b38df98e87bb40e335 "$EXPECTED_COMMIT" || \
  die "launch commit is not based on the audited stage-faithful core commit"

[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA256" ]] || \
  die "matched checkpoint SHA-256 changed"
[[ "$(stat -c %s "$CHECKPOINT")" == "$CHECKPOINT_BYTES" ]] || \
  die "matched checkpoint byte count changed"
[[ "$(sha256sum "$ABC_MANIFEST" | awk '{print $1}')" == "$ABC_MANIFEST_SHA256" ]] || \
  die "ABC manifest SHA-256 changed"

EVAL_ID="abc200-tf-cascade-stage-eval-s1234-${EXPECTED_COMMIT:0:10}-v1"
OUTPUT_ROOT="$RUN_ROOT/$EVAL_ID"
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
  die "fresh evaluation output already exists: $OUTPUT_ROOT"

export LACWM_ALLOWED_RUN_ROOTS="$LACWM_BASE"
CANONICAL_RUN_ROOT="$(
  "$PYTHON_BIN" "$REPO_ROOT/tools/run_root_policy.py" --run-root "$RUN_ROOT"
)"
[[ "$CANONICAL_RUN_ROOT" == "$RUN_ROOT" ]] || \
  die "run root canonicalization changed: $CANONICAL_RUN_ROOT"

"$PYTHON_BIN" "$HELPER" wandb-private \
  --entity "$WANDB_ENTITY_VALUE" \
  --project "$WANDB_PROJECT_VALUE"

LOG_DIR="$RUN_ROOT/_slurm/logs"
JOB_NAME="tf-stage-eval-${EXPECTED_COMMIT:0:10}"
SBATCH_ARGS=(
  --parsable
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
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.err"
)
JOB_ARGS=(
  --eval-id "$EVAL_ID"
  --expected-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --run-root "$RUN_ROOT"
  --output-root "$OUTPUT_ROOT"
)
COMMAND=(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")

printf 'Validated stage-faithful evaluation sbatch command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Git commit: $EXPECTED_COMMIT (clean; based on 5880d047)"
echo "Checkpoint: $CHECKPOINT ($CHECKPOINT_SHA256; $CHECKPOINT_BYTES bytes)"
echo "Output: $OUTPUT_ROOT (fresh)"
echo "Resources: non-array, 1 node, 8 B200, 64 CPUs, 600G, 00:30:00, no-requeue"
echo "Slurm allocation: partition=batch, account=$SLURM_ACCOUNT, QOS=$SLURM_QOS"
echo "Evaluation: iter=199 after skip=4, NFE=[2,4,8], 8 rank artifacts"
echo "Sources: autonomous, autonomous_shuffled, autonomous_legacy, off"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE (PRIVATE), group=null, resume=never"
echo "Clock: sigma=1 noise, sigma=0 clean"
if ((${#ALLOWED_ACTIVE_JOB_IDS[@]} == 0)); then
  echo "Active-job policy: fail closed on every active user job"
else
  echo "Active-job policy: allow only numeric IDs ${ALLOWED_ACTIVE_JOB_IDS[*]}"
fi

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --expected-commit '$EXPECTED_COMMIT' and the exact live --allow-active-job-id values, then --execute."
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

# First immutable coexistence snapshot, before creating any output state.
check_active_user_jobs

mkdir -p "$RUN_ROOT"
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
  die "approved run root is unavailable or symlinked"
mkdir -p "$LOG_DIR"
mkdir "$OUTPUT_ROOT" || \
  die "could not exclusively create fresh evaluation output root"
MANIFEST="$OUTPUT_ROOT/eval_manifest.json"

CREATE_ARGS=(
  "$PYTHON_BIN" "$HELPER" create-manifest
  --eval-id "$EVAL_ID"
  --git-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --run-root "$RUN_ROOT"
  --output-root "$OUTPUT_ROOT"
  --output "$MANIFEST"
)
for allowed_job_id in "${ALLOWED_ACTIVE_JOB_IDS[@]}"; do
  CREATE_ARGS+=(--allow-active-job-id "$allowed_job_id")
done
for row in "${ACTIVE_JOB_ROWS[@]}"; do
  CREATE_ARGS+=(--observed-active-job "$row")
done
"${CREATE_ARGS[@]}"

# A changed active-job set invalidates the exact scope immediately before sbatch.
check_active_user_jobs
IMMEDIATE_PRE_SBATCH_JOB_ROWS=("${ACTIVE_JOB_ROWS[@]}")
JOB_ID="$("${COMMAND[@]}")" || \
  die "Slurm rejected the stage-faithful evaluation submission"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || \
  die "sbatch returned a non-numeric or array job identifier: $JOB_ID"

RECORD_ARGS=(
  "$PYTHON_BIN" "$HELPER" record-submission
  --manifest "$MANIFEST"
  --job-id "$JOB_ID"
  --output "$OUTPUT_ROOT/slurm_submission.json"
)
for row in "${IMMEDIATE_PRE_SBATCH_JOB_ROWS[@]}"; do
  RECORD_ARGS+=(--observed-active-job "$row")
done
"${RECORD_ARGS[@]}"

echo "Submitted stage-faithful evaluation job: $JOB_ID"
echo "Logs: $LOG_DIR/$JOB_NAME-$JOB_ID.out"
