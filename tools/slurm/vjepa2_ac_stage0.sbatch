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
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Train-only native-DROID qualification of the released V-JEPA 2-AC model.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() { echo "ERROR: $*" >&2; exit 2; }

REPO_ROOT=""
EXPECTED_COMMIT=""
SOURCE_DIR=""
CHECKPOINT=""
CHECKPOINT_SHA256=""
TRAIN_MANIFEST=""
DATA_ROOT=""
OUTPUT_DIR=""
PYTHON_BIN=""
CLIP_COUNT=32
MAX_HORIZON=2

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --source-dir) SOURCE_DIR="${2:?}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
    --checkpoint-sha256) CHECKPOINT_SHA256="${2:?}"; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST="${2:?}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --clip-count) CLIP_COUNT="${2:?}"; shift 2 ;;
    --max-horizon) MAX_HORIZON="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "qualification requires one non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "qualification is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "qualification requires exactly one node"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
[[ "$CHECKPOINT_SHA256" == \
  "0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b" ]] || \
  die "checkpoint SHA-256 differs from the frozen official digest"
[[ "$CLIP_COUNT" =~ ^[0-9]+$ && "$CLIP_COUNT" -gt 1 ]] || \
  die "clip count must be an integer greater than one"
[[ "$MAX_HORIZON" =~ ^[2-7]$ ]] || die "max horizon must be in [2,7]"

for path in \
  "$REPO_ROOT" "$SOURCE_DIR" "$CHECKPOINT" "$TRAIN_MANIFEST" \
  "$DATA_ROOT" "$OUTPUT_DIR" "$PYTHON_BIN"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository is noncanonical"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || \
  die "repository is not a worktree root"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" ]] || die "official source is invalid"
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == \
  "45d025f636dfc58fc2426905fc4a1ab755b1c3e5" ]] || \
  die "official source commit differs"
[[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=all)" ]] || \
  die "official source must be clean"
[[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" ]] || die "checkpoint is invalid"
[[ "$(stat -c %s "$CHECKPOINT")" == "11760743310" ]] || \
  die "checkpoint byte count differs"
[[ -f "$TRAIN_MANIFEST" && ! -L "$TRAIN_MANIFEST" ]] || \
  die "train manifest is invalid"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "DROID root is invalid"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || \
  die "qualification output must be fresh"

unset PYTHONPATH
export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS

COMMAND=(
  "$PYTHON_BIN" "$REPO_ROOT/tools/vjepa2_ac_stage0.py" evaluate
  --source-dir "$SOURCE_DIR"
  --checkpoint "$CHECKPOINT"
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256"
  --train-manifest "$TRAIN_MANIFEST"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --clip-count "$CLIP_COUNT"
  --max-horizon "$MAX_HORIZON"
  --seed 20260809
  --device cuda:0
  --dtype bfloat16
)
printf 'Launching frozen native-DROID V-JEPA 2-AC qualification:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
(cd "$REPO_ROOT"; "${COMMAND[@]}")

"$PYTHON_BIN" "$REPO_ROOT/tools/vjepa2_ac_stage0.py" audit \
  --output-dir "$OUTPUT_DIR"
