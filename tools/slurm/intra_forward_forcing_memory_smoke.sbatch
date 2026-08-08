#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/_slurm_logs/memory-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/_slurm_logs/memory-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires Slurm"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "memory smoke is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "memory smoke uses one node"
[[ -n "${MID_SCREEN_REGISTRATION:-}" ]] || die "registration is not exported"
[[ -n "${MID_SCREEN_REPO_ROOT:-}" ]] || die "repository root is not exported"
[[ -n "${MID_SCREEN_PYTHON:-}" ]] || die "runtime Python is not exported"

REPO_ROOT="$(cd "$MID_SCREEN_REPO_ROOT" && pwd -P)"
PYTHON_BIN="$MID_SCREEN_PYTHON"
HELPER="$REPO_ROOT/tools/intra_forward_forcing_screen.py"
SMOKE="$REPO_ROOT/tools/smoke_intra_forward_memory.py"
[[ -f "$HELPER" && -f "$SMOKE" && -x "$PYTHON_BIN" ]] || \
  die "memory-smoke implementation/runtime is missing"

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
  die "memory smoke requires a clean worktree"
[[ ! -e "$STUDY_ROOT/memory_smoke.json" ]] || die "memory smoke receipt exists"
[[ ! -e "$STUDY_ROOT/memory_smoke_runtime.json" ]] || \
  die "memory smoke runtime receipt exists"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
source "$REPO_ROOT/tools/env/activate_b200.sh"

export MID_WARM_START="$WARMSTART"
export MID_TRAIN_CLIP_MANIFEST="$TRAIN_MANIFEST"
export MID_TRAIN_CACHE_METADATA="$TRAIN_METADATA"
export MID_VAL_CLIP_MANIFEST="$VAL_MANIFEST"
export MID_VAL_CACHE_METADATA="$VAL_METADATA"
export MID_CLIP_MANIFEST="$TRAIN_MANIFEST"
export MID_CACHE_METADATA="$TRAIN_METADATA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON_BIN" "$HELPER" verify \
  --registration "$MID_SCREEN_REGISTRATION" >/dev/null
"$PYTHON_BIN" "$REPO_ROOT/tools/env/verify_b200_runtime.py" \
  --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE" \
  > "$STUDY_ROOT/memory_smoke_runtime.json"

IPC_TMPDIR="/tmp/lacwm-mid-memory-${UID}-${SLURM_JOB_ID}"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC directory already exists"
mkdir "$IPC_TMPDIR"
export TMPDIR="$IPC_TMPDIR"

cd "$REPO_ROOT/projects/latent_action_models"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 "$SMOKE" \
  --registration "$MID_SCREEN_REGISTRATION" \
  --runtime-receipt "$STUDY_ROOT/memory_smoke_runtime.json" \
  --output "$STUDY_ROOT/memory_smoke.json"

"$PYTHON_BIN" "$HELPER" verify-memory-smoke \
  --registration "$MID_SCREEN_REGISTRATION" >/dev/null
