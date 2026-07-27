#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=normal
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

EVAL_ID=""
EXPECTED_COMMIT=""
REPO_ROOT=""
RUN_ROOT=""
OUTPUT_ROOT=""

while (($#)); do
  case "$1" in
    --eval-id) [[ $# -ge 2 ]] || die "--eval-id requires a value"; EVAL_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "--repo-root requires a value"; REPO_ROOT="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || die "--run-root requires a value"; RUN_ROOT="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "evaluation ID:$EVAL_ID" \
  "expected commit:$EXPECTED_COMMIT" \
  "repository root:$REPO_ROOT" \
  "run root:$RUN_ROOT" \
  "output root:$OUTPUT_ROOT"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

LACWM_BASE="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
PYTHON_BIN="$LACWM_BASE/envs/lacwm-b200-py310/bin/python"
PYTHON_LINK_TARGET="/lustre/fsw/portfolios/coreai/users/ldu/lacwm_train/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
PYTHON_REAL_BIN="$LACWM_BASE/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
WAN_DIR_VALUE="$LACWM_BASE/wan_fun_1.3b_control"
VIDEOX_HOME_VALUE="$LACWM_BASE/VideoX-Fun-1d6d9c3"
DATA_ROOT="$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1"
ABC_MANIFEST="$DATA_ROOT/abc_pp/manifest.txt"
ABC_MANIFEST_SHA256="e52232b49ffec39600aa22e2d708497f22a4ea57fc89f84bc289ae4b1e0a5c09"
PARENT_ROOT="$LACWM_BASE/runs/dual_video_diffusion/ztf_first_cascade_screen/abc200-tf-cascade3-s1234-18318ed-v1/cascade_matched_s010"
CHECKPOINT="$PARENT_ROOT/snapshot.pt"
CHECKPOINT_SHA256="5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d"
CHECKPOINT_BYTES="4249340573"
WANDB_ENTITY_VALUE="zijiandu"
WANDB_PROJECT_VALUE="dual-video-diffusion-private"
SLURM_ACCOUNT="coreai_chef_posttrain"
SLURM_QOS="normal"

PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
EVALUATOR="$PROJECT_ROOT/evaluate_stage_faithful.py"
HELPER="$REPO_ROOT/tools/stage_faithful_eval_launch.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
MANIFEST="$OUTPUT_ROOT/eval_manifest.json"
INVENTORY="$OUTPUT_ROOT/eval_artifact_inventory.json"
COMPLETION="$OUTPUT_ROOT/evaluation_complete.json"
OUTCOME="$OUTPUT_ROOT/outcome.json"
SNAPSHOT_SENTINEL="$OUTPUT_ROOT/_never_write_snapshot.pt"

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires a Slurm allocation"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "evaluation must be a non-array job"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "evaluation is non-requeueable; refusing restart ${SLURM_RESTART_COUNT}"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "evaluation requires exactly one node"
if command -v scontrol >/dev/null 2>&1; then
  JOB_RECORD="$(scontrol show job "$SLURM_JOB_ID" -o)"
  [[ "$JOB_RECORD" == *"Requeue=0"* ]] || die "Slurm job is unexpectedly requeueable"
  [[ "$JOB_RECORD" != *"ArrayTaskId="* ]] || die "Slurm job is unexpectedly an array"
  [[ "$JOB_RECORD" == *"Account=$SLURM_ACCOUNT "* ]] || \
    die "Slurm job has an unexpected account"
  [[ "$JOB_RECORD" == *"QOS=$SLURM_QOS "* ]] || \
    die "Slurm job has an unexpected QOS"
fi

[[ "$REPO_ROOT" == /* && "$RUN_ROOT" == /* && "$OUTPUT_ROOT" == /* ]] || \
  die "repository and output paths must be absolute"
[[ "$OUTPUT_ROOT" == "$RUN_ROOT/$EVAL_ID" ]] || \
  die "output root is outside the exact evaluation scope"
[[ -e "$REPO_ROOT/.git" && ! -L "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository root is unavailable or symlinked"
[[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
  die "evaluation output root is unavailable or symlinked"
for required_file in \
  "$EVALUATOR" "$HELPER" "$ACTIVATE" "$VERIFY_RUNTIME" \
  "$MANIFEST" "$CHECKPOINT" "$ABC_MANIFEST"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] || \
    die "required regular file is missing or symlinked: $required_file"
done
[[ -L "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || \
  die "pinned virtual-environment Python symlink is unavailable"
[[ "$(readlink "$PYTHON_BIN")" == "$PYTHON_LINK_TARGET" ]] || \
  die "pinned Python symlink target changed"
[[ "$(readlink -f "$PYTHON_BIN")" == "$PYTHON_REAL_BIN" ]] || \
  die "pinned Python canonical executable changed"
[[ -f "$PYTHON_REAL_BIN" && ! -L "$PYTHON_REAL_BIN" && -x "$PYTHON_REAL_BIN" ]] || \
  die "pinned canonical Python executable is unavailable"
for forbidden in "$SNAPSHOT_SENTINEL" "$OUTPUT_ROOT/training_complete.json"; do
  [[ ! -e "$forbidden" && ! -L "$forbidden" ]] || \
    die "evaluation-only forbidden output already exists: $forbidden"
done

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_DATA="$DATA_ROOT"
export LACWM_RUNS="$RUN_ROOT"
export LACWM_PYTHON="$PYTHON_BIN"
export LACWM_ALLOWED_RUN_ROOTS="$RUN_ROOT"
export LACWM_NNODES=1
export LACWM_WORLD_SIZE=8
export LACWM_GPUS_PER_NODE=8
export WANDB_MODE=online
export WANDB_ENTITY="$WANDB_ENTITY_VALUE"
export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
unset WANDB_RUN_GROUP WANDB_RUN_ID WANDB_NAME WANDB_TAGS WANDB_RESUME
export WANDB_DIR="$OUTPUT_ROOT/wandb"
export WANDB_DATA_DIR="$OUTPUT_ROOT/wandb-data"
export WANDB_CACHE_DIR="$OUTPUT_ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$OUTPUT_ROOT/wandb-config"
export HF_HOME="$OUTPUT_ROOT/cache/huggingface"
export TORCH_HOME="$OUTPUT_ROOT/cache/torch"
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache/xdg"
export MPLCONFIGDIR="$OUTPUT_ROOT/cache/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

for fresh_dir in \
  "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
  "$OUTPUT_ROOT/cache"; do
  [[ ! -e "$fresh_dir" && ! -L "$fresh_dir" ]] || \
    die "fresh runtime directory already exists: $fresh_dir"
  mkdir "$fresh_dir"
done

IPC_TMPDIR="/tmp/lacwm-stage-eval-${UID}-${SLURM_JOB_ID}"
[[ ${#IPC_TMPDIR} -le 72 ]] || die "IPC TMPDIR is unexpectedly long"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC TMPDIR already exists"
(umask 077 && mkdir "$IPC_TMPDIR")
export TMPDIR="$IPC_TMPDIR"
cleanup_ipc_tmpdir() {
  rmdir "$IPC_TMPDIR" 2>/dev/null || true
}
trap cleanup_ipc_tmpdir EXIT

source "$ACTIVATE"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit changed: $ACTUAL_COMMIT != $EXPECTED_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository is dirty: ${GIT_STATUS//$'\n'/; }"
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_CHECKPOINT_SHA256" == "$CHECKPOINT_SHA256" ]] || \
  die "matched checkpoint SHA-256 changed: $ACTUAL_CHECKPOINT_SHA256"
[[ "$(stat -c %s "$CHECKPOINT")" == "$CHECKPOINT_BYTES" ]] || \
  die "matched checkpoint byte count changed"
[[ "$(sha256sum "$ABC_MANIFEST" | awk '{print $1}')" == "$ABC_MANIFEST_SHA256" ]] || \
  die "ABC manifest SHA-256 changed"

MANIFEST_IDENTITY="$(
  "$PYTHON_BIN" "$HELPER" validate-manifest \
    --manifest "$MANIFEST" \
    --git-commit "$EXPECTED_COMMIT" \
    --repo-root "$REPO_ROOT"
)"
[[ "$MANIFEST_IDENTITY" =~ ^[0-9a-f]{64}$ ]] || \
  die "manifest validator returned an invalid identity"
export LACWM_RUN_IDENTITY_SHA256="$MANIFEST_IDENTITY"

OUTCOME_WRITTEN=0
record_failed_outcome() {
  local status=$?
  if ((status != 0 && OUTCOME_WRITTEN == 0)) && [[ ! -e "$OUTCOME" ]]; then
    set +e
    "$PYTHON_BIN" "$HELPER" record-outcome \
      --manifest "$MANIFEST" \
      --completion "$COMPLETION" \
      --exit-status "$status" \
      --slurm-job-id "$SLURM_JOB_ID" \
      --output "$OUTCOME"
    set -e
  fi
  cleanup_ipc_tmpdir
}
trap record_failed_outcome EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

(
  set -o noclobber
  "$PYTHON_BIN" "$VERIFY_RUNTIME" \
    --expected-gpus 8 \
    --require-b200 \
    --wan-dir "$WAN_DIR_VALUE" > "$OUTPUT_ROOT/runtime_verification.json"
  "$PYTHON_BIN" "$HELPER" wandb-private \
    --entity "$WANDB_ENTITY_VALUE" \
    --project "$WANDB_PROJECT_VALUE" > "$OUTPUT_ROOT/wandb_private_preflight.json"
)

mapfile -t HYDRA_OVERRIDES < <(
  "$PYTHON_BIN" "$HELPER" hydra-overrides \
    --eval-id "$EVAL_ID" \
    --output-root "$OUTPUT_ROOT"
)
[[ ${#HYDRA_OVERRIDES[@]} -eq 52 ]] || \
  die "expected exactly 52 audited Hydra overrides, got ${#HYDRA_OVERRIDES[@]}"

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node=8
  evaluate_stage_faithful.py
  "${HYDRA_OVERRIDES[@]}"
)
printf 'Launching evaluation-only stage-faithful diagnostic from %q:' "$PROJECT_ROOT"
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Evaluation ID: $EVAL_ID"
echo "Checkpoint: $CHECKPOINT ($CHECKPOINT_SHA256)"
echo "World size: 8 B200 ranks; non-array; no requeue"
echo "Slurm allocation: account=$SLURM_ACCOUNT, QOS=$SLURM_QOS"
echo "Evaluation: NFE=[2,4,8], sources=[autonomous,autonomous_shuffled,autonomous_legacy,off]"
echo "Clock: sigma=1 noise, sigma=0 clean"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE (PRIVATE), id=$EVAL_ID, group=null, resume=never"

(
  cd "$PROJECT_ROOT"
  "${COMMAND[@]}"
)

for forbidden in "$SNAPSHOT_SENTINEL" "$OUTPUT_ROOT/training_complete.json"; do
  [[ ! -e "$forbidden" && ! -L "$forbidden" ]] || \
    die "evaluation wrote forbidden training state: $forbidden"
done
[[ -f "$OUTPUT_ROOT/stage_faithful_evaluation_provenance.json" ]] || \
  die "rank-zero evaluation provenance is missing"

"$PYTHON_BIN" "$HELPER" complete \
  --manifest "$MANIFEST" \
  --inventory-output "$INVENTORY" \
  --completion-output "$COMPLETION"
"$PYTHON_BIN" "$HELPER" record-outcome \
  --manifest "$MANIFEST" \
  --completion "$COMPLETION" \
  --exit-status 0 \
  --slurm-job-id "$SLURM_JOB_ID" \
  --output "$OUTCOME"
OUTCOME_WRITTEN=1
echo "Stage-faithful evaluation completed with exactly eight rank artifacts."
