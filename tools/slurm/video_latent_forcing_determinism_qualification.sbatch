#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

die() { echo "ERROR: $*" >&2; exit 2; }

[[ "${1:-}" == "--repo-root" && $# -ge 8 ]] || die "missing --repo-root"
REPO_ROOT="$2"; shift 2
[[ "${1:-}" == "--python" && $# -ge 6 ]] || die "missing --python"
PYTHON_BIN="$2"; shift 2
[[ "${1:-}" == "--expected-commit" && $# -ge 4 ]] || die "missing --expected-commit"
EXPECTED_COMMIT="$2"; shift 2
[[ "${1:-}" == "--" ]] || die "expected -- before qualifier arguments"
shift
[[ "${1:-}" == "run" ]] || die "Slurm qualifier accepts only the run command"

[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || \
  die "repo root must be an absolute non-symlink directory"
REPO_ROOT="$(readlink -f "$REPO_ROOT")"
TOOL="$REPO_ROOT/tools/qualify_video_latent_forcing_determinism.py"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python must be an absolute executable"
[[ -f "$TOOL" && ! -L "$TOOL" ]] || die "qualifier tool is missing or symlinked"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit must be a full SHA"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "queued source commit changed"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || die "queued source is dirty"
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-0}" == "1" ]] || \
  die "qualification requires a fresh one-node Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "qualification jobs cannot be requeued"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" == "8" ]] || die "expected exactly eight allocated GPUs"
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *B200* ]] || die "expected B200 GPU, found: $gpu_name"
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 \
  "$TOOL" "$@"
