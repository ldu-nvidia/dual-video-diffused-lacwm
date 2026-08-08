#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Recover evaluation after a completed action-variation training job whose
# post-training W&B seal failed.  This entrypoint never trains or overwrites an
# evaluation.  It consumes the already registered checkpoint and exact source
# commit, seals the completed private W&B run if needed, and runs only the
# preregistered evaluator.

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

ARM=""
REGISTRATION=""
REPO_ROOT=""
PYTHON_BIN=""
EXPECTED_COMMIT=""
RECOVERY_WRAPPER=""
while (($#)); do
  case "$1" in
    --arm) ARM="${2:?}"; shift 2 ;;
    --registration) REGISTRATION="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --recovery-wrapper) RECOVERY_WRAPPER="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" ]] || die "entrypoint requires Slurm"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || die "recovery is non-requeueable"
[[ "$ARM" == AV-CONT || "$ARM" == AV-DELTA ]] || die "invalid arm"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute Git worktree"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python is unavailable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
[[ "$RECOVERY_WRAPPER" == /* && -f "$RECOVERY_WRAPPER" && ! -L "$RECOVERY_WRAPPER" ]] || \
  die "recovery wrapper must be an absolute regular file"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "registered repository must be clean"

WORKFLOW="$REPO_ROOT/tools/slurm/action_variation_workflow.py"
EVALUATOR="$REPO_ROOT/tools/action_variation_evaluate.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for path in "$WORKFLOW" "$EVALUATOR" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required implementation missing: $path"
done

IFS=$'\t' read -r \
  ARM_CODE CONFIG_NAME RUN_NAME RUN_IDENTITY TRAINING_ROOT RUN_DIR EVAL_DIR PLAN \
  RUNTIME_RECEIPT \
  PARENT_SNAPSHOT TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  STATS STATS_SHA PYTHON_REGISTERED WAN_DIR_VALUE VIDEOX_HOME_VALUE \
  < <(
    "$PYTHON_BIN" "$WORKFLOW" arm-values --registration "$REGISTRATION" \
      --arm "$ARM" --format tsv
  )
[[ "$ARM_CODE" == "$ARM" ]] || die "arm values changed"
[[ "$PYTHON_REGISTERED" == "$PYTHON_BIN" ]] || die "runtime Python differs"
[[ -s "$RUN_DIR/snapshot.pt" && ! -L "$RUN_DIR/snapshot.pt" ]] || \
  die "completed training snapshot is absent"
[[ -s "$RUN_DIR/training_complete.json" && ! -L "$RUN_DIR/training_complete.json" ]] || \
  die "training completion is absent"
[[ ! -e "$EVAL_DIR" && ! -L "$EVAL_DIR" ]] || die "evaluation output is not fresh"

unset PYTHONPATH
export LACWM_PYTHON="$PYTHON_BIN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
source "$ACTIVATE"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/projects/latent_action_models:$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME"

# The sealed Hydra configuration deliberately retains environment-backed
# absolute paths.  Training exported these values in the original task; an
# evaluation-only recovery must restore the identical registered values before
# the immutable evaluator resolves that configuration.  These exports do not
# change data selection: every value came from the registration's arm-values
# command above, and the evaluator revalidates the resulting canonical config.
export ACTION_VARIATION_VPM_SNAPSHOT="$PARENT_SNAPSHOT"
export ACTION_VARIATION_VPM_SNAPSHOT_SHA256="f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
export ACTION_VARIATION_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export ACTION_VARIATION_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export ACTION_VARIATION_VAL_CLIP_MANIFEST="$VAL_MANIFEST"
export ACTION_VARIATION_VAL_CACHE_METADATA="$VAL_METADATA"
export ACTION_VARIATION_STATS="$STATS"
export ACTION_VARIATION_STATS_SHA256="$STATS_SHA"
export ACTION_VARIATION_RUN_ROOT="$TRAINING_ROOT"
export ACTION_VARIATION_REGISTRATION="$REGISTRATION"
export ACTION_VARIATION_ARM_CODE="$ARM_CODE"

if [[ ! -s "$RUN_DIR/wandb_run_complete.json" ]]; then
  "$PYTHON_BIN" "$WORKFLOW" seal-wandb \
    --registration "$REGISTRATION" --arm "$ARM_CODE" >/dev/null
fi
[[ -s "$RUN_DIR/wandb_run_complete.json" && ! -L "$RUN_DIR/wandb_run_complete.json" ]] || \
  die "verified W&B completion receipt is absent"

TMPDIR_VALUE="/tmp/lacwm-action-variation-eval-recovery-${UID}-${SLURM_JOB_ID}-${ARM_CODE}"
[[ ! -e "$TMPDIR_VALUE" ]] || die "fresh IPC directory already exists"
mkdir "$TMPDIR_VALUE"
export TMPDIR="$TMPDIR_VALUE"

RUNTIME_VERIFIER_JSON="$TMPDIR/runtime_verifier_output.json"
"$PYTHON_BIN" "$VERIFY_RUNTIME" --expected-gpus 8 --require-b200 \
  --wan-dir "$WAN_DIR" > "$RUNTIME_VERIFIER_JSON"
RECOVERY_RECEIPT="$(dirname "$REGISTRATION")/recovery/${ARM_CODE,,}-runtime-${SLURM_JOB_ID}.json"
[[ ! -e "$RECOVERY_RECEIPT" && ! -L "$RECOVERY_RECEIPT" ]] || \
  die "fresh recovery runtime receipt already exists"
"$PYTHON_BIN" "$RECOVERY_WRAPPER" seal-runtime \
  --registration "$REGISTRATION" --arm "$ARM_CODE" \
  --verifier-json "$RUNTIME_VERIFIER_JSON" --output "$RECOVERY_RECEIPT" >/dev/null
[[ -s "$RECOVERY_RECEIPT" && ! -L "$RECOVERY_RECEIPT" ]] || \
  die "recovery runtime receipt is absent"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 "$RECOVERY_WRAPPER" \
  evaluate --registration "$REGISTRATION" --arm "$ARM_CODE" \
  --output-dir "$EVAL_DIR" --batch-size-per-rank 1 \
  --recovery-runtime-receipt "$RECOVERY_RECEIPT"
[[ -s "$EVAL_DIR/inventory.json" && ! -L "$EVAL_DIR/inventory.json" ]] || \
  die "evaluation inventory is absent"
echo "Action-variation evaluation recovery complete: $ARM_CODE"
