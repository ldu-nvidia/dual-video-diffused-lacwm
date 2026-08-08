#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089
#SBATCH --array=0-3%2
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/frequency_forcing/_slurm_logs/train-%A_%a.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/frequency_forcing/_slurm_logs/train-%A_%a.err
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
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]] || die "array task must be 0 through 3"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "screen is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "each arm uses one node"
[[ -n "${FREQ_SCREEN_REGISTRATION:-}" ]] || \
  die "export FREQ_SCREEN_REGISTRATION before sbatch"
[[ "$FREQ_SCREEN_REGISTRATION" == /* && -f "$FREQ_SCREEN_REGISTRATION" ]] || \
  die "FREQ_SCREEN_REGISTRATION must be an absolute file"

# The registration supplies every mutable path and the exact source commit.
BOOTSTRAP_REPO="${FREQ_SCREEN_REPO_ROOT:-}"
[[ "$BOOTSTRAP_REPO" == /* && -d "$BOOTSTRAP_REPO/.git" ]] || \
  die "export FREQ_SCREEN_REPO_ROOT to the exact clean checkout"
BOOTSTRAP_REPO="$(cd "$BOOTSTRAP_REPO" && pwd -P)"
BOOTSTRAP_HELPER="$BOOTSTRAP_REPO/tools/frequency_forcing_screen.py"
[[ -f "$BOOTSTRAP_HELPER" && ! -L "$BOOTSTRAP_HELPER" ]] || \
  die "frequency-forcing helper is missing"

BOOTSTRAP_PYTHON="${FREQ_SCREEN_PYTHON:-}"
[[ "$BOOTSTRAP_PYTHON" == /* && -x "$BOOTSTRAP_PYTHON" ]] || \
  die "export FREQ_SCREEN_PYTHON to the pinned B200 environment"

IFS='|' read -r \
  EXPECTED_COMMIT STUDY_ROOT PYTHON_BIN WAN_DIR_VALUE VIDEOX_HOME_VALUE \
  WARMSTART TRAIN_MANIFEST TRAIN_METADATA VAL_MANIFEST VAL_METADATA \
  REGISTRATION_ID \
  < <(
    "$BOOTSTRAP_PYTHON" "$BOOTSTRAP_HELPER" runtime \
      --registration "$FREQ_SCREEN_REGISTRATION"
  )
[[ "$BOOTSTRAP_REPO" == "$(git -C "$BOOTSTRAP_REPO" rev-parse --show-toplevel)" ]] || \
  die "repository root is not canonical"
[[ "$(git -C "$BOOTSTRAP_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs from registration"
[[ -z "$(git -C "$BOOTSTRAP_REPO" status --porcelain --untracked-files=all)" ]] || \
  die "training requires a clean worktree"
[[ "$PYTHON_BIN" == "$BOOTSTRAP_PYTHON" ]] || \
  die "bootstrap Python differs from registered Python"

"$PYTHON_BIN" "$BOOTSTRAP_HELPER" verify \
  --registration "$FREQ_SCREEN_REGISTRATION" >/dev/null

IFS='|' read -r \
  ARM_CODE ARM_SLUG CONFIG_SELECTOR CONDITION_MODE CONDITION_ON_STATE \
  CONDITION_ON_CLOCK SCHEDULE_MODE LEAD_LOGIT AUXILIARY_LOSS_WEIGHT \
  STATE_GATE_INIT CLOCK_GATE_INIT PARAMETER_MATCHED \
  < <(
    "$PYTHON_BIN" "$BOOTSTRAP_HELPER" arm \
      --array-task-id "$SLURM_ARRAY_TASK_ID" --format pipe
  )
RUN_ID="$(basename "$STUDY_ROOT")-${ARM_CODE}"
RUN_DIR="$STUDY_ROOT/runs/$ARM_SLUG"
"$PYTHON_BIN" "$BOOTSTRAP_HELPER" bind-arm \
  --registration "$FREQ_SCREEN_REGISTRATION" \
  --array-task-id "$SLURM_ARRAY_TASK_ID" \
  --run-dir "$RUN_DIR" \
  --slurm-array-job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" >/dev/null

ARM_ID="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["identity_sha256"])' "$RUN_DIR/arm_manifest.json")"
mkdir \
  "$RUN_DIR/wandb" \
  "$RUN_DIR/wandb-data" \
  "$RUN_DIR/wandb-cache" \
  "$RUN_DIR/wandb-config" \
  "$RUN_DIR/cache"

ACTIVATE="$BOOTSTRAP_REPO/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$BOOTSTRAP_REPO/tools/env/verify_b200_runtime.py"
WARMSTART_PREFLIGHT="$BOOTSTRAP_REPO/tools/validate_frequency_forcing_warmstart.py"
[[ -f "$ACTIVATE" && -f "$VERIFY_RUNTIME" && -f "$WARMSTART_PREFLIGHT" ]] || \
  die "B200 runtime/preflight tools are missing"
# The activation guard validates these paths while it is sourced.  They must
# come from the immutable cluster registration, never its workstation defaults.
export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
source "$ACTIVATE"

export FREQ_WARM_START="$WARMSTART"
export FREQ_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export FREQ_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export FREQ_VAL_CLIP_MANIFEST="$VAL_MANIFEST"
export FREQ_VAL_CACHE_METADATA="$VAL_METADATA"
export FREQ_RUN_ROOT="$STUDY_ROOT/runs"
export FREQ_CLIP_MANIFEST="$TRAIN_MANIFEST"
export FREQ_CACHE_METADATA="$TRAIN_METADATA"
export LACWM_RUN_IDENTITY_SHA256="$ARM_ID"
export LACWM_NNODES=1
export LACWM_WORLD_SIZE=8
export LACWM_GPUS_PER_NODE=8
export WANDB_MODE=online
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
unset WANDB_RUN_GROUP WANDB_RUN_ID WANDB_NAME WANDB_TAGS
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_DATA_DIR="$RUN_DIR/wandb-data"
export WANDB_CACHE_DIR="$RUN_DIR/wandb-cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb-config"
export HF_HOME="$RUN_DIR/cache/huggingface"
export TORCH_HOME="$RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
export MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

IPC_TMPDIR="/tmp/lacwm-freq-${UID}-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"

"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE" \
  > "$RUN_DIR/runtime.json"

"$PYTHON_BIN" "$WARMSTART_PREFLIGHT" \
  --registration "$FREQ_SCREEN_REGISTRATION" \
  --selector "$CONFIG_SELECTOR" \
  --warmstart "$WARMSTART" \
  --output "$RUN_DIR/warmstart_preflight.json" >/dev/null

PROJECT_ROOT="$BOOTSTRAP_REPO/projects/latent_action_models"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 train.py \
  "+experiments_0908=$CONFIG_SELECTOR" \
  "name=$RUN_ID" \
  "seed=1234" \
  "trainer.config.load_path=$WARMSTART" \
  "trainer.config.max_iter=200" \
  "val_dataset.infinite=true" \
  "val_dataset.seed=1234" \
  "val_dataset.img_augment=false" \
  "val_dataset.future_validity.enabled=false" \
  "val_dataset.future_validity.max_retries=0" \
  "val_data_loader.0.batch_size=2" \
  "val_data_loader.0.drop_last=false" \
  "val_data_loader.0.num_workers=2" \
  "val_data_loader.0.persistent_workers=false" \
  "trainer.config.validation.n_val_samples=4" \
  "trainer.config.saving.save_path=$RUN_DIR/snapshot.pt" \
  "trainer.config.visualization.viz_path=$RUN_DIR/visualization" \
  "hydra.run.dir=$RUN_DIR" \
  "hydra.sweep.dir=$RUN_DIR" \
  "wandb.enabled=true" \
  "wandb.mode=online" \
  "wandb.entity=zijiandu" \
  "wandb.project=dual-video-diffusion-private" \
  "+wandb.id=$RUN_ID" \
  "+wandb.group=null" \
  "+wandb.resume=never"

[[ -f "$RUN_DIR/snapshot.pt" && -f "$RUN_DIR/training_complete.json" ]] || \
  die "training returned without its final snapshot/receipt"
