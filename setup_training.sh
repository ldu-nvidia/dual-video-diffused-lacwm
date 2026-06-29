#!/usr/bin/env bash
#
# setup_training.sh — provision a new server to train the Wan-DiT latent-action
# world model: install the env, fetch the Wan weights + VideoX-Fun, fetch the 4
# training datasets, (re)generate their manifests, repoint paths, and validate.
#
# The active run trains on ABC + Agibot + DROID + EgoDex (config
# transformed_multi_abc_agibot_droid_egodex). Libero data/manifests are optional.
#
# Usage:
#   1. Edit the CONFIG block below.
#   2. ./setup_training.sh            # full: env + fetch + manifests + validate
#   3. ./setup_training.sh manifests  # only (re)generate manifests on present data
#   4. ./setup_training.sh validate   # only check the layout
#
# Manifests store ABSOLUTE paths, so they MUST be regenerated on every new server
# (that is the main reason this script exists) — never copy them between machines.

set -euo pipefail

# ─────────────────────────────── CONFIG ──────────────────────────────────────
# Root that holds data + weights + VideoX-Fun. The repo/configs reference
# /scr/ravenh by default; keep BASE=/scr/ravenh to avoid any path edits, or set a
# new root and the script will repoint every /scr/ravenh reference in the repo.
BASE="${BASE:-/scr/ravenh}"

DATA_ROOT="$BASE/lacwm_data"                 # datasets land here
WAN_DIR="$BASE/wan_fun_1.3b_control"         # Wan2.1-Fun weights + null prompt
VIDEOX_DIR="$BASE/VideoX-Fun"                # VideoX-Fun library (provides videox_fun.*)
REPO_DIR="${REPO_DIR:-$HOME/lacwm-dit}"      # this repository
CONDA_ENV="${CONDA_ENV:-lacwm-dit}"

# How to obtain weights + datasets:
#   rsync  — pull from a box that already has them (default; you have them on the source)
#   skip   — data is already in place; only (re)generate manifests
FETCH="${FETCH:-rsync}"
SOURCE_HOST="${SOURCE_HOST:-ravenh@38.213.24.3}"   # box that currently holds the data
SOURCE_BASE="${SOURCE_BASE:-/scr/ravenh}"          # its BASE

# Datasets to set up (the 4 the active run uses). Add/remove as needed.
DATASETS=(droid_lerobot egodex_cdn agibot abc_pp)
# ──────────────────────────────────────────────────────────────────────────────

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n'  "$*" >&2; exit 1; }

# ─── 1. Environment ───────────────────────────────────────────────────────────
setup_env() {
  log "Setting up conda env '$CONDA_ENV' + installing the package"
  command -v conda >/dev/null || die "conda not found; install Miniconda first"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env list | grep -qE "^\s*$CONDA_ENV\s" || conda create -y -n "$CONDA_ENV" python=3.10
  conda activate "$CONDA_ENV"

  # Torch must match the server's CUDA. This pins the stack the run was built on
  # (torch 2.7.1 + cu128); change the index-url/version for a different CUDA.
  pip install --upgrade pip
  pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128 || \
    warn "torch install failed — install a torch build matching your CUDA, then re-run"

  # Wan + training + eval deps (the repo's setup.py has no pinned requirements).
  pip install diffusers==0.38.0 accelerate ftfy librosa sentencepiece \
              hydra-core omegaconf peft einops wandb \
              opencv-python decord av h5py pandas pyarrow scipy \
              lpips imageio imageio-ffmpeg

  pip install -e "$REPO_DIR"
  log "Cloning VideoX-Fun -> $VIDEOX_DIR"
  [ -d "$VIDEOX_DIR/.git" ] || git clone https://github.com/aigc-apps/VideoX-Fun.git "$VIDEOX_DIR"
}

# ─── 2. Weights (Wan2.1-Fun-1.3B-Control + null prompt) ───────────────────────
fetch_weights() {
  mkdir -p "$WAN_DIR"
  if [ "$FETCH" = rsync ]; then
    log "rsync Wan weights from $SOURCE_HOST:$SOURCE_BASE/wan_fun_1.3b_control"
    rsync -ah --info=progress2 "$SOURCE_HOST:$SOURCE_BASE/wan_fun_1.3b_control/" "$WAN_DIR/"
  else
    # HF alternative for the OFFICIAL files (VAE, DiT, umT5, clip):
    #   huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-Control --local-dir "$WAN_DIR"
    # NOTE: null_prompt_umt5.pt is a cached null umT5 embedding we added — it is NOT
    # in the official repo. Copy it from the source box (it is tiny, ~18 KB):
    #   scp $SOURCE_HOST:$SOURCE_BASE/wan_fun_1.3b_control/null_prompt_umt5.pt "$WAN_DIR/"
    warn "FETCH=$FETCH: skipping weight download — ensure $WAN_DIR is populated"
  fi
  [ -f "$WAN_DIR/Wan2.1_VAE.pth" ]        || warn "missing $WAN_DIR/Wan2.1_VAE.pth"
  [ -f "$WAN_DIR/null_prompt_umt5.pt" ]   || warn "missing $WAN_DIR/null_prompt_umt5.pt (required; copy from source box)"
}

# ─── 3. Datasets ──────────────────────────────────────────────────────────────
fetch_datasets() {
  mkdir -p "$DATA_ROOT"
  if [ "$FETCH" != rsync ]; then
    warn "FETCH=$FETCH: skipping dataset download — ensure $DATA_ROOT/{${DATASETS[*]}} are present"
    return
  fi
  for d in "${DATASETS[@]}"; do
    log "rsync dataset '$d' from $SOURCE_HOST:$SOURCE_BASE/lacwm_data/$d"
    # Bring manifests too. egodex/abc manifests get regenerated below (their stored
    # paths are absolute); agibot's curated manifest is portable and is preserved.
    rsync -ah --info=progress2 \
      "$SOURCE_HOST:$SOURCE_BASE/lacwm_data/$d/" "$DATA_ROOT/$d/"
  done
  # If your data lives on HuggingFace instead, replace the rsync above with, e.g.:
  #   huggingface-cli download <your-repo-id> --repo-type dataset --local-dir "$DATA_ROOT/$d"
}

# ─── 4. Manifests (regenerated for THIS server's absolute paths) ──────────────
make_manifests() {
  log "Regenerating manifests under $DATA_ROOT"

  # EgoDex: one .hdf5 path per line.
  if [ -d "$DATA_ROOT/egodex_cdn" ]; then
    find "$DATA_ROOT/egodex_cdn" -name '*.hdf5' | sort > "$DATA_ROOT/egodex_cdn/manifest.csv"
    log "  egodex_cdn/manifest.csv : $(wc -l < "$DATA_ROOT/egodex_cdn/manifest.csv") episodes"
  else warn "  egodex_cdn/ missing"; fi

  # ABC: one episode directory per line (each holds *.mp4 + states.npz).
  if [ -d "$DATA_ROOT/abc_pp" ]; then
    find "$DATA_ROOT/abc_pp" -mindepth 2 -maxdepth 2 -type d -name 'episode_*' | sort \
      > "$DATA_ROOT/abc_pp/manifest.txt"
    log "  abc_pp/manifest.txt     : $(wc -l < "$DATA_ROOT/abc_pp/manifest.txt") episodes"
  else warn "  abc_pp/ missing"; fi

  # Agibot: CSV "task_id,episode_id,dataset"; key 'scr' -> DATASET_ROOTS['scr'].
  # IMPORTANT: the training manifest is a *curated subset* (e.g. 5671 of the ~15k
  # episodes on disk) that is NOT derivable from the data — every on-disk episode has
  # complete files. The manifest is path-portable (no absolute paths), so we PRESERVE
  # it if present and only fall back to enumerating ALL episodes (with a warning).
  if [ -f "$DATA_ROOT/agibot/manifest.csv" ]; then
    log "  agibot/manifest.csv     : $(( $(wc -l < "$DATA_ROOT/agibot/manifest.csv") - 1 )) episodes (preserved curated subset)"
  elif [ -d "$DATA_ROOT/agibot/observations" ]; then
    warn "  agibot: no curated manifest found -> enumerating ALL episodes (NOT the training subset)."
    warn "  copy the curated manifest from the source box to match the original run:"
    warn "    scp $SOURCE_HOST:$SOURCE_BASE/lacwm_data/agibot/manifest.csv $DATA_ROOT/agibot/"
    {
      echo "task_id,episode_id,dataset"
      for t in "$DATA_ROOT/agibot/observations"/*/; do
        [ -d "$t" ] || continue
        tid=$(basename "$t")
        for e in "$t"*/; do [ -d "$e" ] && echo "${tid},$(basename "$e"),scr"; done
      done
    } > "$DATA_ROOT/agibot/manifest.csv"
    log "  agibot/manifest.csv     : $(( $(wc -l < "$DATA_ROOT/agibot/manifest.csv") - 1 )) episodes (ALL — review before training)"
  else warn "  agibot/ missing"; fi

  # DROID (LeRobot) needs no manifest — the loader globs data/chunk-*/episode_*.parquet.
  if [ -d "$DATA_ROOT/droid_lerobot/data" ]; then
    log "  droid_lerobot           : $(find "$DATA_ROOT/droid_lerobot/data" -name 'episode_*.parquet' | wc -l) episode parquets (no manifest needed)"
  else warn "  droid_lerobot/data/ missing"; fi
}

# ─── 5. Repoint repo paths if BASE != /scr/ravenh ─────────────────────────────
repoint_paths() {
  [ "$BASE" = /scr/ravenh ] && { log "BASE is /scr/ravenh — no path edits needed"; return; }
  log "Repointing /scr/ravenh -> $BASE across the repo configs + module defaults"
  grep -rl '/scr/ravenh' "$REPO_DIR" --include='*.py' --include='*.yaml' 2>/dev/null \
    | grep -v '/wandb/' \
    | xargs -r sed -i "s|/scr/ravenh|$BASE|g"
  warn "Also update any launch scripts (e.g. run_wan_full.sh) that reference /scr/ravenh"
}

# ─── 6. Validate ──────────────────────────────────────────────────────────────
validate() {
  log "Validating layout"
  local ok=1
  for f in "$WAN_DIR/Wan2.1_VAE.pth" "$WAN_DIR/diffusion_pytorch_model.safetensors" \
           "$WAN_DIR/null_prompt_umt5.pt" "$VIDEOX_DIR/config/wan2.1/wan_civitai.yaml"; do
    [ -e "$f" ] && echo "  ok  $f" || { echo "  MISSING  $f"; ok=0; }
  done
  for m in "$DATA_ROOT/egodex_cdn/manifest.csv" "$DATA_ROOT/abc_pp/manifest.txt" \
           "$DATA_ROOT/agibot/manifest.csv" "$DATA_ROOT/droid_lerobot/data"; do
    [ -e "$m" ] && echo "  ok  $m" || { echo "  MISSING  $m"; ok=0; }
  done
  [ "$ok" = 1 ] && log "Validation passed." || die "Validation found missing pieces (see above)."
  cat <<EOF

Ready. Launch the full 8-GPU run from $REPO_DIR/projects/latent_action_models with:

  cd $REPO_DIR/projects/latent_action_models
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  torchrun --standalone --nproc_per_node=8 train.py \\
    +experiments_0908=ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml \\
    data_loader.batch_size=16

(smoke test first: +experiments_0908=ravenhuang/wan-dit/wan_dit_smoke.yaml)
EOF
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────
case "${1:-all}" in
  all)       setup_env; fetch_weights; fetch_datasets; make_manifests; repoint_paths; validate ;;
  env)       setup_env ;;
  weights)   fetch_weights ;;
  datasets)  fetch_datasets ;;
  manifests) make_manifests ;;
  repoint)   repoint_paths ;;
  validate)  validate ;;
  *) die "usage: $0 [all|env|weights|datasets|manifests|repoint|validate]" ;;
esac
