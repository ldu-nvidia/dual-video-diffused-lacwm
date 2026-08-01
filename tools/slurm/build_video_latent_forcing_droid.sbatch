#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --constraint=B200
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$REPO_ROOT/tools/build_video_latent_forcing_droid.py"

die() { echo "ERROR: $*" >&2; exit 2; }

[[ $# -eq 8 ]] || die "expected --python, --expected-commit, --data-root, --output-root"
[[ "$1" == "--python" ]] || die "missing --python"
PYTHON_BIN="$2"
[[ "$3" == "--expected-commit" ]] || die "missing --expected-commit"
EXPECTED_COMMIT="$4"
[[ "$5" == "--data-root" ]] || die "missing --data-root"
DATA_ROOT="$6"
[[ "$7" == "--output-root" ]] || die "missing --output-root"
OUTPUT_ROOT="$8"

[[ "$PYTHON_BIN" == /* && -f "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "invalid Python"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid commit"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "invalid DROID source root"
[[ "$OUTPUT_ROOT" == /* && ! -e "$OUTPUT_ROOT" ]] || die "output must be a fresh absolute path"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "source commit changed"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || die "source worktree is dirty"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "cache build cannot be requeued"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$TOOL" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --cache-split train \
  --cache-split val \
  --cache-workers 32
