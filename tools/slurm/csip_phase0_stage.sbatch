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
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ $# -eq 5 ]] || \
  die "usage: csip_phase0_stage.sbatch STAGE REGISTRATION REPO COMMIT PYTHON"
STAGE="$1"
REGISTRATION="$2"
REPO_ROOT="$3"
EXPECTED_COMMIT="$4"
PYTHON_BIN="$5"

[[ "$STAGE" == "train" || "$STAGE" == "validation" ]] || \
  die "stage must be train or validation"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires Slurm"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "CSIP is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "CSIP requires one node"
[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registration must be an absolute regular file"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || \
  die "repository must be an absolute non-symlinked Git checkout"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is malformed"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || \
  die "Python must be an absolute executable"

REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$REPO_ROOT" == "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" ]] || \
  die "repository root is not canonical"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must remain clean"

# Bind imports before the first project Python process.  The registered runtime
# and the activation guard add only the reviewed repo, shim, and pinned VideoX.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$REPO_ROOT"
"$PYTHON_BIN" "$REPO_ROOT/tools/csip_workflow.py" render \
  --registration "$REGISTRATION" >/dev/null

mapfile -t REGISTERED < <(
  "$PYTHON_BIN" - "$REGISTRATION" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
paths = value["planned_paths"]
for item in (
    value["source"]["path"],
    value["source"]["git_commit"],
    value["runtime"]["python"]["launcher_path"],
    value["runtime"]["wan_dir"],
    value["runtime"]["videox_home"],
    value["study_root"],
    paths["train_latent_root"],
    paths["train_latent_metadata"],
    paths["validation_latent_root"],
    paths["validation_latent_metadata"],
    paths["checkpoint"],
    paths["training_report"],
    paths["checkpoint_seal"],
    paths["evaluation"],
    paths["analysis"],
    paths["wandb_local_dir"],
):
    print(item)
PY
)
[[ ${#REGISTERED[@]} -eq 16 ]] || die "registration runtime fields are incomplete"
SOURCE_ROOT="${REGISTERED[0]}"
SOURCE_COMMIT="${REGISTERED[1]}"
REGISTERED_PYTHON="${REGISTERED[2]}"
WAN_DIR_VALUE="${REGISTERED[3]}"
VIDEOX_HOME_VALUE="${REGISTERED[4]}"
STUDY_ROOT="${REGISTERED[5]}"
TRAIN_LATENT_ROOT="${REGISTERED[6]}"
TRAIN_LATENT_METADATA="${REGISTERED[7]}"
VAL_LATENT_ROOT="${REGISTERED[8]}"
VAL_LATENT_METADATA="${REGISTERED[9]}"
CHECKPOINT="${REGISTERED[10]}"
TRAINING_REPORT="${REGISTERED[11]}"
CHECKPOINT_SEAL="${REGISTERED[12]}"
EVALUATION="${REGISTERED[13]}"
ANALYSIS="${REGISTERED[14]}"
WANDB_LOCAL_DIR="${REGISTERED[15]}"

[[ "$SOURCE_ROOT" == "$REPO_ROOT" && "$SOURCE_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "source differs from registration"
[[ "$REGISTERED_PYTHON" == "$PYTHON_BIN" ]] || \
  die "Python differs from registration"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is unavailable"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
unset PYTHONPATH
source "$REPO_ROOT/tools/env/activate_b200.sh"
export PYTHONNOUSERSITE=1

RUNTIME_ROOT="$STUDY_ROOT/runtime/${STAGE}-job-${SLURM_JOB_ID}"
[[ ! -e "$RUNTIME_ROOT" && ! -L "$RUNTIME_ROOT" ]] || \
  die "runtime output must be fresh"
mkdir -p "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/wandb-data" \
  "$RUNTIME_ROOT/wandb-cache" "$RUNTIME_ROOT/wandb-config"
export WANDB_DATA_DIR="$RUNTIME_ROOT/wandb-data"
export WANDB_CACHE_DIR="$RUNTIME_ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$RUNTIME_ROOT/wandb-config"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export TORCH_HOME="$RUNTIME_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache/xdg"
export MPLCONFIGDIR="$RUNTIME_ROOT/cache/matplotlib"
export WANDB_MODE=online
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
unset WANDB_RUN_GROUP WANDB_RUN_ID WANDB_NAME WANDB_TAGS

TMP_ROOT="/tmp/csip-${UID}-${SLURM_JOB_ID}-${STAGE}"
[[ ! -e "$TMP_ROOT" ]] || die "temporary root must be fresh"
mkdir "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"

"$PYTHON_BIN" "$REPO_ROOT/tools/env/verify_b200_runtime.py" \
  --expected-gpus 8 --require-b200 --wan-dir "$WAN_DIR_VALUE" \
  > "$RUNTIME_ROOT/runtime.json"

if [[ "$STAGE" == "train" ]]; then
  for path in \
    "$TRAIN_LATENT_ROOT" "$CHECKPOINT" "$TRAINING_REPORT" "$CHECKPOINT_SEAL" \
    "$VAL_LATENT_ROOT" "$EVALUATION" "$ANALYSIS" "$WANDB_LOCAL_DIR"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "train output is not fresh: $path"
  done
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$REPO_ROOT/tools/csip_latent_cache.py" extract \
    --registration "$REGISTRATION" --split train
  "$PYTHON_BIN" "$REPO_ROOT/tools/csip_train.py" \
    --registration "$REGISTRATION"
  "$PYTHON_BIN" "$REPO_ROOT/tools/csip_workflow.py" seal \
    --registration "$REGISTRATION" \
    --checkpoint "$CHECKPOINT" \
    --training-report "$TRAINING_REPORT"
  [[ -f "$TRAIN_LATENT_METADATA" && -f "$CHECKPOINT" && \
     -f "$TRAINING_REPORT" && -f "$CHECKPOINT_SEAL" ]] || \
    die "train stage returned without fixed sealed outputs"
else
  for path in "$TRAIN_LATENT_METADATA" "$CHECKPOINT" "$TRAINING_REPORT" "$CHECKPOINT_SEAL"; do
    [[ -f "$path" && ! -L "$path" ]] || die "sealed train input is absent: $path"
  done
  for path in "$VAL_LATENT_ROOT" "$EVALUATION" "$ANALYSIS"; do
    [[ ! -e "$path" && ! -L "$path" ]] || \
      die "validation output is not fresh: $path"
  done
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$REPO_ROOT/tools/csip_latent_cache.py" extract \
    --registration "$REGISTRATION" --split validation --seal "$CHECKPOINT_SEAL"
  "$PYTHON_BIN" "$REPO_ROOT/tools/csip_evaluate.py" --seal "$CHECKPOINT_SEAL"
  "$PYTHON_BIN" "$REPO_ROOT/tools/csip_analyze.py" \
    --seal "$CHECKPOINT_SEAL" --evaluation "$EVALUATION"
  [[ -f "$VAL_LATENT_METADATA" && -f "$EVALUATION" && -f "$ANALYSIS" ]] || \
    die "validation stage returned without sealed analysis outputs"
fi
