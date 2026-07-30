#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

readonly VJEPA_SOURCE_COMMIT=45d025f636dfc58fc2426905fc4a1ab755b1c3e5

die() {
  echo "ERROR: $*" >&2
  exit 2
}

REPO_ROOT=""
STUDY_ROOT=""
TRAINING_COMMIT=""
EVALUATOR_COMMIT=""
PYTHON_BIN=""
EXTRACTOR_PYTHON=""
VJEPA_SOURCE=""
VJEPA_CHECKPOINT=""
VJEPA_CHECKPOINT_SHA256=""
PCA_STATS=""
PCA_STATS_SHA256=""
TRAIN_MANIFEST=""

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --extractor-python) EXTRACTOR_PYTHON="${2:?}"; shift 2 ;;
    --vjepa-source) VJEPA_SOURCE="${2:?}"; shift 2 ;;
    --vjepa-checkpoint) VJEPA_CHECKPOINT="${2:?}"; shift 2 ;;
    --vjepa-checkpoint-sha256) VJEPA_CHECKPOINT_SHA256="${2:?}"; shift 2 ;;
    --pca) PCA_STATS="${2:?}"; shift 2 ;;
    --pca-sha256) PCA_STATS_SHA256="${2:?}"; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for pair in \
  "repository:$REPO_ROOT" "study root:$STUDY_ROOT" \
  "training commit:$TRAINING_COMMIT" "evaluator commit:$EVALUATOR_COMMIT" \
  "LACWM Python:$PYTHON_BIN" "extractor Python:$EXTRACTOR_PYTHON" \
  "V-JEPA source:$VJEPA_SOURCE" "V-JEPA checkpoint:$VJEPA_CHECKPOINT" \
  "checkpoint SHA-256:$VJEPA_CHECKPOINT_SHA256" "PCA:$PCA_STATS" \
  "PCA SHA-256:$PCA_STATS_SHA256" \
  "training manifest:$TRAIN_MANIFEST"; do
  [[ -n "${pair#*:}" ]] || die "${pair%%:*} is required"
done
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "cache extraction requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "cache extraction is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "cache extraction requires one node"
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid training commit"
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid evaluator commit"
[[ "$VJEPA_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "invalid V-JEPA checkpoint SHA-256"
[[ "$PCA_STATS_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid PCA SHA-256"
[[ "$REPO_ROOT" == /* && "$STUDY_ROOT" == /* ]] || die "roots must be absolute"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "invalid repository"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository root is not canonical"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || \
  die "repository must be a clone/worktree root"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is unavailable"
[[ -x "$PYTHON_BIN" && -x "$EXTRACTOR_PYTHON" ]] || \
  die "both Python interpreters must be executable"
for path in "$VJEPA_CHECKPOINT" "$PCA_STATS" "$TRAIN_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required file is unavailable: $path"
done
[[ -d "$VJEPA_SOURCE/.git" && ! -L "$VJEPA_SOURCE" ]] || \
  die "V-JEPA source checkout is unavailable"

LOCKBOX_TOOL="$REPO_ROOT/tools/vjepa2_frontier_lockbox.py"
EXTRACTOR="$REPO_ROOT/tools/extract_vjepa2_targets.py"
STUDY_MANIFEST="$STUDY_ROOT/study_manifest.json"
LOCKBOX_ROOT="$STUDY_ROOT/frontier_lockbox"
CONSTRUCTION="$LOCKBOX_ROOT/lockbox_construction.json"
CLIP_MANIFEST="$LOCKBOX_ROOT/lockbox.jsonl"
CACHE_ROOT="$LOCKBOX_ROOT/cache"
CACHE_METADATA="$CACHE_ROOT/metadata.json"
REGISTRATION="$LOCKBOX_ROOT/registration.json"
for path in "$LOCKBOX_TOOL" "$EXTRACTOR" "$STUDY_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required input is unavailable: $path"
done
[[ ! -e "$LOCKBOX_ROOT" ]] || \
  die "fresh lockbox root already exists: $LOCKBOX_ROOT"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
git -C "$REPO_ROOT" merge-base --is-ancestor \
  "$TRAINING_COMMIT" "$EVALUATOR_COMMIT" || \
  die "evaluator commit is not a descendant of the training commit"
[[ "$(git -C "$VJEPA_SOURCE" rev-parse HEAD)" == "$VJEPA_SOURCE_COMMIT" ]] || \
  die "V-JEPA source checkout commit changed"
[[ -z "$(git -C "$VJEPA_SOURCE" status --porcelain --untracked-files=all)" ]] || \
  die "V-JEPA source checkout is dirty"
[[ "$(sha256sum "$VJEPA_CHECKPOINT" | awk '{print $1}')" == \
  "$VJEPA_CHECKPOINT_SHA256" ]] || die "V-JEPA checkpoint digest changed"
[[ "$(sha256sum "$PCA_STATS" | awk '{print $1}')" == "$PCA_STATS_SHA256" ]] || \
  die "PCA digest changed"

"$EXTRACTOR_PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"cache extraction expected one visible GPU, found {torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_properties(0).name
if "B200" not in name.upper():
    raise SystemExit(f"cache extraction requires one B200, found {name}")
PY

# Build the episode-disjoint manifest before importing the V-JEPA environment.
"$PYTHON_BIN" "$LOCKBOX_TOOL" build-manifest \
  --repo-root "$REPO_ROOT" \
  --registration-commit "$EVALUATOR_COMMIT" \
  --study-manifest "$STUDY_MANIFEST" \
  --output-dir "$LOCKBOX_ROOT"

[[ -f "$CONSTRUCTION" && -f "$CLIP_MANIFEST" && ! -e "$CACHE_ROOT" ]] || \
  die "lockbox manifest construction did not produce a fresh extraction target"

export PYTHONPATH="$REPO_ROOT:$VJEPA_SOURCE"
export TORCH_HOME="$LOCKBOX_ROOT/runtime_cache/torch"
export HF_HOME="$LOCKBOX_ROOT/runtime_cache/huggingface"
export XDG_CACHE_HOME="$LOCKBOX_ROOT/runtime_cache/xdg"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

"$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-inputs \
  --source-path "$VJEPA_SOURCE" \
  --checkpoint "$VJEPA_CHECKPOINT" \
  --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"
"$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-pca \
  --pca "$PCA_STATS" \
  --train-manifest "$TRAIN_MANIFEST"

"$EXTRACTOR_PYTHON" "$EXTRACTOR" extract \
  --source-path "$VJEPA_SOURCE" \
  --checkpoint "$VJEPA_CHECKPOINT" \
  --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256" \
  --split test \
  --clip-manifest "$CLIP_MANIFEST" \
  --train-manifest "$TRAIN_MANIFEST" \
  --pca "$PCA_STATS" \
  --cache-dir "$CACHE_ROOT" \
  --device cuda \
  --encoder-dtype bfloat16 \
  --batch-size 1 \
  --finite-check-rows 8

"$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-cache \
  --cache-metadata "$CACHE_METADATA" \
  --clip-manifest "$CLIP_MANIFEST" \
  --train-manifest "$TRAIN_MANIFEST" \
  --pca "$PCA_STATS" \
  --source-path "$VJEPA_SOURCE" \
  --checkpoint "$VJEPA_CHECKPOINT" \
  --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256" \
  --finite-check-rows 8

[[ ! -e "$REGISTRATION" ]] || die "fresh lockbox registration already exists"
"$PYTHON_BIN" "$LOCKBOX_TOOL" register \
  --repo-root "$REPO_ROOT" \
  --registration-commit "$EVALUATOR_COMMIT" \
  --study-manifest "$STUDY_MANIFEST" \
  --construction "$CONSTRUCTION" \
  --cache-metadata "$CACHE_METADATA" \
  --attest-never-scored \
  --output "$REGISTRATION"
[[ -s "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "lockbox registration was not produced"

echo "Fresh 128-clip lockbox cache registered: $REGISTRATION"
