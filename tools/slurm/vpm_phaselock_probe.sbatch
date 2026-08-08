#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=coreai_chef_posttrain
#SBATCH --qos=short
#SBATCH --exclude=pool0-0081,pool0-0089
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

TOOL_REPO=""
TOOL_COMMIT=""
MODEL_REPO=""
STUDY_ROOT=""
OUTPUT_ROOT=""
PYTHON_BIN=""

while (($#)); do
  case "$1" in
    --tool-repo) TOOL_REPO="${2:?}"; shift 2 ;;
    --tool-commit) TOOL_COMMIT="${2:?}"; shift 2 ;;
    --model-repo) MODEL_REPO="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" ]] || die "probe requires a Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "probe is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "probe requires one node"
[[ "$TOOL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid tool commit"
for path in "$TOOL_REPO" "$MODEL_REPO" "$STUDY_ROOT" "$OUTPUT_ROOT"; do
  [[ "$path" == /* ]] || die "all paths must be absolute"
done
[[ "$OUTPUT_ROOT" == /lustre/* ]] || die "large outputs must be on Lustre"
[[ ! -e "$OUTPUT_ROOT" ]] || die "fresh output root already exists: $OUTPUT_ROOT"
[[ -d "$TOOL_REPO" && ! -L "$TOOL_REPO" ]] || die "invalid tool repository"
[[ -d "$MODEL_REPO" && ! -L "$MODEL_REPO" ]] || die "invalid model repository"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "invalid study root"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ "$(git -C "$TOOL_REPO" rev-parse HEAD)" == "$TOOL_COMMIT" ]] || \
  die "tool checkout differs from frozen commit"
[[ -z "$(git -C "$TOOL_REPO" status --porcelain --untracked-files=all)" ]] || \
  die "tool checkout must be clean"
[[ "$(git -C "$MODEL_REPO" rev-parse HEAD)" == \
  "9cf8e6922f35a5d6645e3128545953723bf54da2" ]] || \
  die "historical model checkout differs from training commit"
[[ -z "$(git -C "$MODEL_REPO" status --porcelain --untracked-files=all)" ]] || \
  die "historical model checkout must be clean"

PROBE="$TOOL_REPO/tools/vpm_phaselock_probe.py"
ANALYZER="$TOOL_REPO/tools/analyze_vpm_phaselock_probe.py"
ACTIVATE="$TOOL_REPO/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$TOOL_REPO/tools/env/verify_b200_runtime.py"
for path in "$PROBE" "$ANALYZER" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required tool is unavailable: $path"
done

WAN_DIR_VALUE="$(
  "$PYTHON_BIN" - "$STUDY_ROOT/study_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["inputs"]["runtime"]["wan_dir"])
PY
)"
VIDEOX_HOME_VALUE="$(
  "$PYTHON_BIN" - "$STUDY_ROOT/study_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["inputs"]["runtime"]["videox_home"])
PY
)"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "registered Wan/VideoX runtime is unavailable"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset PYTHONPATH
source "$ACTIVATE"
"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

# Registration is the first metric-independent scientific action. It fully
# rehashes the frozen 4.25 GB checkpoint and validation metadata before any
# candidate endpoint is evaluated.
"$PYTHON_BIN" "$PROBE" register \
  --tool-repo "$TOOL_REPO" \
  --tool-commit "$TOOL_COMMIT" \
  --model-repo "$MODEL_REPO" \
  --study-root "$STUDY_ROOT" \
  --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "$PROBE" evaluate \
  --registration "$OUTPUT_ROOT/registration.json" \
  --batch-size-per-rank 2 \
  --output-dir "$OUTPUT_ROOT/evaluation"

[[ -s "$OUTPUT_ROOT/evaluation/inventory.json" ]] || \
  die "validation evaluation did not produce an inventory"

"$PYTHON_BIN" "$ANALYZER" \
  --inventory "$OUTPUT_ROOT/evaluation/inventory.json" \
  --output-dir "$OUTPUT_ROOT/analysis"

[[ -s "$OUTPUT_ROOT/analysis/analysis.json" ]] || \
  die "probe analysis did not complete"
echo "VPM PhaseLock validation-only probe complete: $OUTPUT_ROOT/analysis/analysis.json"
