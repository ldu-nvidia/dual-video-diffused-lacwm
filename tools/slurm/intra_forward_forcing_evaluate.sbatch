#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343
#SBATCH --array=0-1%2
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/_slurm_logs/eval-%A_%a.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/_slurm_logs/eval-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "this entrypoint requires a Slurm array allocation"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-1]$ ]] || die "array task must be 0 or 1"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "evaluation is non-requeueable"
[[ -n "${MID_SCREEN_REGISTRATION:-}" ]] || die "registration is not exported"
[[ -n "${MID_SCREEN_REPO_ROOT:-}" ]] || die "repository root is not exported"
[[ -n "${MID_SCREEN_PYTHON:-}" ]] || die "runtime Python is not exported"

REPO_ROOT="$(cd "$MID_SCREEN_REPO_ROOT" && pwd -P)"
PYTHON_BIN="$MID_SCREEN_PYTHON"
HELPER="$REPO_ROOT/tools/intra_forward_forcing_screen.py"
EVALUATOR="$REPO_ROOT/tools/evaluate_intra_forward_forcing.py"
[[ -f "$HELPER" && -f "$EVALUATOR" && -x "$PYTHON_BIN" ]] || \
  die "evaluation implementation/runtime is missing"

IFS='|' read -r \
  EXPECTED_COMMIT STUDY_ROOT REGISTERED_PYTHON WAN_DIR_VALUE VIDEOX_HOME_VALUE \
  WARMSTART TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  REGISTRATION_ID \
  < <(
    "$PYTHON_BIN" "$HELPER" runtime \
      --registration "$MID_SCREEN_REGISTRATION"
  )
[[ "$REGISTERED_PYTHON" == "$PYTHON_BIN" ]] || die "runtime Python differs"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "source commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "evaluation requires a clean worktree"

IFS='|' read -r \
  ARM_CODE ARM_SLUG CONFIG_SELECTOR CONDITION_MODE CONDITION_ON_STATE \
  CONDITION_ON_CLOCK SCHEDULE_MODE LEAD_LOGIT AUXILIARY_LOSS_WEIGHT \
  STATE_GATE_INIT CLOCK_GATE_INIT PARAMETER_MATCHED MIDPOINT_INJECTION \
  < <(
    "$PYTHON_BIN" "$HELPER" arm \
      --array-task-id "$SLURM_ARRAY_TASK_ID" --format pipe
  )
RUN_DIR="$STUDY_ROOT/runs/$ARM_SLUG"
OUTPUT_DIR="$STUDY_ROOT/evaluation/$ARM_SLUG"
[[ -f "$RUN_DIR/snapshot.pt" && -f "$RUN_DIR/training_complete.json" \
   && -f "$RUN_DIR/initialization_match.json" \
   && -f "$RUN_DIR/training_content.json" ]] || \
  die "arm training is incomplete"
[[ ! -e "$OUTPUT_DIR" ]] || die "fresh evaluation output already exists"
[[ ! -e "$RUN_DIR/evaluation_runtime.json" ]] || \
  die "fresh evaluation runtime receipt already exists"
[[ ! -e "$RUN_DIR/evaluation_input_preflight.json" ]] || \
  die "fresh evaluation input preflight already exists"

# The activation guard validates these paths while it is sourced.  They must
# come from the immutable cluster registration, never its workstation defaults.
export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
source "$REPO_ROOT/tools/env/activate_b200.sh"

export MID_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export MID_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export MID_VAL_CLIP_MANIFEST="$VAL_MANIFEST"
export MID_VAL_CACHE_METADATA="$VAL_METADATA"
export MID_CLIP_MANIFEST="$VAL_MANIFEST"
export MID_CACHE_METADATA="$VAL_METADATA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MID_EVAL_RUNTIME_RECEIPT="$RUN_DIR/evaluation_runtime.json"

"$PYTHON_BIN" "$REPO_ROOT/tools/env/verify_b200_runtime.py" \
  --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE" \
  > "$MID_EVAL_RUNTIME_RECEIPT"

# Perform every multi-GB registration, warm-start, and snapshot hash in this
# single-process phase. The resulting immutable receipt is the only content
# preflight read after torchrun creates the NCCL process group.
"$PYTHON_BIN" "$HELPER" bind-evaluation-input \
  --registration "$MID_SCREEN_REGISTRATION" \
  --array-task-id "$SLURM_ARRAY_TASK_ID" \
  --run-dir "$RUN_DIR" \
  --runtime-receipt "$MID_EVAL_RUNTIME_RECEIPT" >/dev/null

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 "$EVALUATOR" \
  --registration "$MID_SCREEN_REGISTRATION" \
  --array-task-id "$SLURM_ARRAY_TASK_ID" \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR"

[[ -f "$OUTPUT_DIR/complete.json" && -f "$OUTPUT_DIR/rows.jsonl" ]] || \
  die "evaluation returned without complete result artifacts"
