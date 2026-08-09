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

EXPECTED_COMMIT=""; REPO_ROOT=""; PYTHON_BIN=""; OUTPUT_DIR=""
PARENT_ARTIFACT=""; OFFICIAL_ABC_ROOT=""
while (($#)); do
  case "$1" in
    --expected-commit) EXPECTED_COMMIT="${2:?}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --parent-artifact) PARENT_ARTIFACT="${2:?}"; shift 2 ;;
    --official-abc-root) OFFICIAL_ABC_ROOT="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "token fallback requires one non-array allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "job is non-requeueable"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
for path in \
  "$REPO_ROOT" "$PYTHON_BIN" "$OUTPUT_DIR" "$PARENT_ARTIFACT" \
  "$OFFICIAL_ABC_ROOT"; do
  [[ "$path" == /* ]] || die "all paths must be absolute: $path"
done
[[ -x "$PYTHON_BIN" ]] || die "pinned Python is unavailable"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "repository is invalid"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -f "$PARENT_ARTIFACT/audit.json" ]] || die "parent audited artifact is missing"
[[ "$(git -C "$OFFICIAL_ABC_ROOT" rev-parse HEAD)" == \
  "6bc6586721cf0c409ccee80f675a28de9b9b2f5e" ]] || \
  die "official ABC commit differs"
[[ -z "$(git -C "$OFFICIAL_ABC_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "official ABC checkout must be clean"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die "output must be fresh"
EXPECTED_BASENAME="interaction-token-train256-score64-seed20260814-${EXPECTED_COMMIT:0:7}-v1"
[[ "$(basename "$OUTPUT_DIR")" == "$EXPECTED_BASENAME" ]] || \
  die "output basename must be $EXPECTED_BASENAME"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS PYTHONPATH

TOOL="$REPO_ROOT/tools/interaction_event_token_stage0.py"
PROTOCOL="$REPO_ROOT/docs/experiments/INTERACTION_EVENT_TOKEN_STAGE0_PROTOCOL.md"
"$PYTHON_BIN" "$TOOL" register \
  --output "$OUTPUT_DIR" \
  --parent-artifact "$PARENT_ARTIFACT" \
  --protocol "$PROTOCOL" \
  --expected-commit "$EXPECTED_COMMIT"
"$PYTHON_BIN" "$TOOL" run \
  --output "$OUTPUT_DIR" \
  --official-abc-root "$OFFICIAL_ABC_ROOT" \
  --mujoco-gl egl
"$PYTHON_BIN" "$TOOL" audit \
  --output "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/audit.json"
