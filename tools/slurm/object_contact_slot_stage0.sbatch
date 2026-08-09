#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Registration is intentionally not performed here. The fresh split must be
# sealed and reported before this run-only job may index fresh score RGB/state.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

EXPECTED_COMMIT=""
REPO_ROOT=""
PYTHON_BIN=""
OUTPUT_DIR=""
OFFICIAL_ABC_ROOT=""

while (($#)); do
  case "$1" in
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --official-abc-root) OFFICIAL_ABC_ROOT="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "qualification requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "qualification is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "qualification requires one node"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
for path in "$REPO_ROOT" "$PYTHON_BIN" "$OUTPUT_DIR" "$OFFICIAL_ABC_ROOT"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || die "repository is noncanonical"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -d "$OFFICIAL_ABC_ROOT" && ! -L "$OFFICIAL_ABC_ROOT" ]] || \
  die "official ABC root is invalid"
[[ "$(git -C "$OFFICIAL_ABC_ROOT" rev-parse HEAD)" == \
  "6bc6586721cf0c409ccee80f675a28de9b9b2f5e" ]] || \
  die "official ABC commit differs"
[[ -z "$(git -C "$OFFICIAL_ABC_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "official ABC checkout must be clean"
[[ -d "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die "registered output is unavailable"
EXPECTED_BASENAME="object-contact-slot-fit256-fresh64-seed20260815-${EXPECTED_COMMIT:0:7}-v1"
[[ "$(basename "$OUTPUT_DIR")" == "$EXPECTED_BASENAME" ]] || \
  die "output basename must be $EXPECTED_BASENAME"
[[ -s "$OUTPUT_DIR/registration.json" ]] || die "sealed registration is unavailable"
[[ ! -e "$OUTPUT_DIR/analysis.json" && ! -e "$OUTPUT_DIR/run_complete.json" ]] || \
  die "output already contains an outcome"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS PYTHONPATH

TOOL="$REPO_ROOT/tools/object_contact_slot_stage0.py"
"$PYTHON_BIN" "$TOOL" run \
  --output "$OUTPUT_DIR" \
  --official-abc-root "$OFFICIAL_ABC_ROOT" \
  --mujoco-gl egl

"$PYTHON_BIN" "$TOOL" audit \
  --output "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/audit.json"
