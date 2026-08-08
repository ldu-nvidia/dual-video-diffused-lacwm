#!/usr/bin/env bash
#SBATCH --job-name=ac-recover-fit
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --exclude=pool0-0081,pool0-0089,pool0-0200,pool0-0343

set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 REGISTRATION PYTHON REPO" >&2; exit 2; }
REGISTRATION=$1
PYTHON_BIN=$2
REPO=$3
[[ $REPO == /* && -d $REPO && ! -L $REPO ]] || {
  echo "repository must be an explicit absolute directory" >&2
  exit 2
}
REPO=$(realpath -e -- "$REPO")
TOOL=$REPO/tools/action_cycle_recoverability.py
LUSTRE_BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
[[ -f $REGISTRATION && -x $PYTHON_BIN ]] || { echo "registered runtime input is missing" >&2; exit 2; }
[[ -d $LUSTRE_BASE && ! -L $LUSTRE_BASE && $(realpath -e -- "$LUSTRE_BASE") == "$LUSTRE_BASE" ]] || {
  echo "reviewed Lustre base is missing or noncanonical" >&2
  exit 2
}
export LACWM_ALLOWED_RUN_ROOTS=$LUSTRE_BASE
export PYTHONPATH="$REPO"
export WANDB_ENTITY=zijiandu
export WANDB_PROJECT=dual-video-diffusion-private
unset WANDB_RUN_GROUP WANDB_GROUP || true
ROOT=$(dirname -- "$REGISTRATION")
export WANDB_DIR=$ROOT/wandb
export WANDB_CACHE_DIR=$ROOT/wandb-cache
export WANDB_CONFIG_DIR=$ROOT/wandb-config
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON_BIN" "$TOOL" validate-registration --registration "$REGISTRATION"
"$PYTHON_BIN" "$TOOL" wandb-check >/dev/null
"$PYTHON_BIN" "$TOOL" analyze --registration "$REGISTRATION" --wandb
