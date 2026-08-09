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

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "one non-array job required"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "repair run is non-requeueable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
for path in "$REPO_ROOT" "$PYTHON_BIN" "$OUTPUT_DIR" "$OFFICIAL_ABC_ROOT"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die "repository must be clean"
[[ "$(git -C "$OFFICIAL_ABC_ROOT" rev-parse HEAD)" == "6bc6586721cf0c409ccee80f675a28de9b9b2f5e" ]] || die "ABC commit differs"
[[ -z "$(git -C "$OFFICIAL_ABC_ROOT" status --porcelain --untracked-files=all)" ]] || die "ABC checkout must be clean"
[[ -s "$OUTPUT_DIR/registration.json" && -s "$OUTPUT_DIR/calibration.json" ]] || die "original registration is unavailable"
[[ -s "$OUTPUT_DIR/mechanical_repair_registration.json" ]] || die "mechanical repair is not registered"
[[ ! -e "$OUTPUT_DIR/analysis.json" && ! -e "$OUTPUT_DIR/run_complete.json" ]] || die "outcome already exists"
[[ "$(basename "$OUTPUT_DIR")" == "object-contact-distribution-fit256-cal64-final31-seed20260816-c0e7ea4-v1" ]] || die "original artifact basename differs"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS PYTHONPATH

WRAPPER="$REPO_ROOT/tools/object_contact_distribution_pca_repair.py"
"$PYTHON_BIN" "$WRAPPER" run \
  --output "$OUTPUT_DIR" \
  --official-abc-root "$OFFICIAL_ABC_ROOT" \
  --mujoco-gl egl
"$PYTHON_BIN" "$WRAPPER" audit \
  --output "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/audit.json"
"$PYTHON_BIN" "$WRAPPER" repair-audit \
  --output "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/mechanical_repair_audit.json"
