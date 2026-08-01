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

die() { echo "ERROR: $*" >&2; exit 2; }

[[ $# -eq 10 ]] || die "expected --repo-root, --python, --expected-commit, --data-root, --output-root"
[[ "$1" == "--repo-root" ]] || die "missing --repo-root"
REPO_ROOT="$2"
[[ "$3" == "--python" ]] || die "missing --python"
PYTHON_BIN="$4"
[[ "$5" == "--expected-commit" ]] || die "missing --expected-commit"
EXPECTED_COMMIT="$6"
[[ "$7" == "--data-root" ]] || die "missing --data-root"
DATA_ROOT="$8"
[[ "$9" == "--output-root" ]] || die "missing --output-root"
OUTPUT_ROOT="${10}"

[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "invalid repository root"
REPO_ROOT="$(readlink -f "$REPO_ROOT")"
TOOL="$REPO_ROOT/tools/build_video_latent_forcing_droid.py"
[[ "$PYTHON_BIN" == /* && -f "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "invalid Python"
[[ -f "$TOOL" && ! -L "$TOOL" ]] || die "tool entrypoint is missing or symlinked"
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
