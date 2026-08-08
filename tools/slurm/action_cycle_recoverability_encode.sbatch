#!/usr/bin/env bash
#SBATCH --job-name=ac-recover-enc
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343

set -euo pipefail

[[ $# -eq 4 ]] || { echo "usage: $0 REGISTRATION PYTHON VIDEOX_HOME WAN_DIR" >&2; exit 2; }
REGISTRATION=$1
PYTHON_BIN=$2
VIDEOX_HOME=$3
WAN_DIR=$4
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
TOOL=$REPO/tools/action_cycle_recoverability.py
LUSTRE_BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train

[[ -f $REGISTRATION && -x $PYTHON_BIN && -d $VIDEOX_HOME && -d $WAN_DIR ]] || {
  echo "registered runtime input is missing" >&2
  exit 2
}
export VIDEOX_HOME WAN_DIR
[[ -d $LUSTRE_BASE && ! -L $LUSTRE_BASE && $(realpath -e -- "$LUSTRE_BASE") == "$LUSTRE_BASE" ]] || {
  echo "reviewed Lustre base is missing or noncanonical" >&2
  exit 2
}
export LACWM_ALLOWED_RUN_ROOTS=$LUSTRE_BASE
export PYTHONPATH="$REPO:$VIDEOX_HOME"
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
unset WANDB_RUN_GROUP WANDB_GROUP || true
ROOT=$(dirname -- "$REGISTRATION")
export WANDB_DIR=$ROOT/wandb
export WANDB_CACHE_DIR=$ROOT/wandb-cache
export WANDB_CONFIG_DIR=$ROOT/wandb-config
export OMP_NUM_THREADS=8

"$PYTHON_BIN" "$TOOL" validate-registration --registration "$REGISTRATION" --full
"$PYTHON_BIN" "$TOOL" wandb-check >/dev/null
for SPLIT in train val; do
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$TOOL" encode --registration "$REGISTRATION" --split "$SPLIT" --wandb
done
