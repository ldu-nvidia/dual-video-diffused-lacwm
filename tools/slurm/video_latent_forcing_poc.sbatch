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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$REPO_ROOT/tools/video_latent_forcing_poc.py"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${1:-}" == "--python" && $# -ge 6 ]] || \
  die "usage: video_latent_forcing_poc.sbatch --python PATH --expected-commit SHA -- calibrate|train|eval ARGS..."
PYTHON_BIN="$2"
shift 2
[[ "${1:-}" == "--expected-commit" && $# -ge 4 ]] || die "missing --expected-commit"
EXPECTED_COMMIT="$2"
shift 2
[[ "${1:-}" == "--" ]] || die "expected -- before tool arguments"
shift
[[ "$1" == "calibrate" || "$1" == "train" || "$1" == "eval" ]] || \
  die "tool command must be calibrate, train, or eval"
[[ "$PYTHON_BIN" == /* && -f "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || \
  die "--python must resolve to an absolute executable"
[[ -f "$TOOL" && ! -L "$TOOL" ]] || die "tool entrypoint is missing or symlinked"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit must be a full SHA"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "queued source commit changed before execution"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || \
  die "queued source worktree is not clean"
[[ "${SLURM_NNODES:-1}" == "1" ]] || die "this POC is frozen to one node"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "POC jobs must not be requeued"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" == "8" ]] || die "expected exactly eight allocated GPUs"
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *B200* ]] || die "expected B200 GPU, found: $gpu_name"
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  "$TOOL" "$@"
