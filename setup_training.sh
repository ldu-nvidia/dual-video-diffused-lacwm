#!/usr/bin/env bash
#
# setup_training.sh — provision a new server to train the Wan-DiT latent-action
# world model: env + Wan weights + VideoX-Fun, the 4 datasets, their manifests,
# path repointing, and validation.
#
# The active run trains on ABC + Agibot + DROID + EgoDex
# (config transformed_multi_abc_agibot_droid_egodex).
#
# Data (FETCH below):
#   download  — pull data from the public sources (default). Each dataset has a toggle
#               and a size limit. NOTE: some raw sources need a preprocessing/format step
#               before training (flagged loudly per dataset, see "PREP" notes).
#   skip      — data already in place; only (re)generate manifests.
#
# Public sources:
#   DROID   https://huggingface.co/datasets/lerobot/droid_1.0.1
#   Agibot  https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
#   ABC     https://huggingface.co/datasets/XDOF/ABC-130k
#   EgoDex  https://github.com/apple/ml-egodex  (zips on Apple's CDN)
#
# Usage:
#   ./setup_training.sh                 # all: env + fetch + manifests + repoint + validate
#   ./setup_training.sh datasets        # just fetch the enabled datasets
#   ./setup_training.sh manifests       # (re)generate manifests for present data
#   ./setup_training.sh validate
#
# Manifests store ABSOLUTE paths, so they are regenerated per-server here — never
# copy egodex/abc manifests between machines. (Agibot's manifest is path-portable.)

set -euo pipefail

# ─────────────────────────────── CONFIG ──────────────────────────────────────
BASE="${BASE:-/scr/ravenh}"                  # holds data + weights + VideoX-Fun
DATA_ROOT="$BASE/lacwm_data"
WAN_DIR="$BASE/wan_fun_1.3b_control"
VIDEOX_DIR="$BASE/VideoX-Fun"
REPO_DIR="${REPO_DIR:-$HOME/lacwm-dit}"
CONDA_ENV="${CONDA_ENV:-lacwm-dit}"

FETCH="${FETCH:-download}"                    # download | skip

# Per-dataset: enable (1/0) and a cap on episodes used (manifest is truncated to it; "all" = no cap).
DROID_ENABLE="${DROID_ENABLE:-1}";   DROID_LIMIT="${DROID_LIMIT:-all}"
AGIBOT_ENABLE="${AGIBOT_ENABLE:-1}"; AGIBOT_LIMIT="${AGIBOT_LIMIT:-all}"
EGODEX_ENABLE="${EGODEX_ENABLE:-1}"; EGODEX_LIMIT="${EGODEX_LIMIT:-all}"
ABC_ENABLE="${ABC_ENABLE:-1}";       ABC_LIMIT="${ABC_LIMIT:-all}"

# download-mode volume knobs (avoid pulling everything):
DROID_FILES="${DROID_FILES:-4}"               # v3 parquet "file-NNN" shards to fetch (chunk-000)
AGIBOT_TASKS="${AGIBOT_TASKS:-all}"           # number of Alpha task dirs to fetch ("all" = 36)
EGODEX_PARTS="${EGODEX_PARTS:-part2}"         # space-sep: part1..part5 test extra (existing run used part2)
# ──────────────────────────────────────────────────────────────────────────────

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*"; }
prep() { printf '\033[1;36m[PREP]\033[0m %s\n'  "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n'  "$*" >&2; exit 1; }
cap()  { if [ "$2" = all ]; then cat; else head -n "$2"; fi; }   # truncate stdin to a limit

# ─── 1. Environment ───────────────────────────────────────────────────────────
setup_env() {
  log "conda env '$CONDA_ENV' + package install"
  command -v conda >/dev/null || die "conda not found"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env list | grep -qE "^\s*$CONDA_ENV\s" || conda create -y -n "$CONDA_ENV" python=3.10
  conda activate "$CONDA_ENV"
  pip install --upgrade pip
  pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128 || \
    warn "torch install failed — install a build matching your CUDA, then re-run"
  pip install diffusers==0.38.0 accelerate ftfy librosa sentencepiece \
              hydra-core omegaconf peft einops wandb "huggingface_hub[cli]" \
              opencv-python decord av h5py pandas pyarrow scipy lpips imageio imageio-ffmpeg
  pip install -e "$REPO_DIR"
  [ -d "$VIDEOX_DIR/.git" ] || git clone https://github.com/aigc-apps/VideoX-Fun.git "$VIDEOX_DIR"
}

# ─── 2. Weights ───────────────────────────────────────────────────────────────
fetch_weights() {
  [ "$FETCH" = download ] || { log "weights: FETCH=$FETCH (using existing $WAN_DIR)"; return; }
  mkdir -p "$WAN_DIR"
  log "hf download alibaba-pai/Wan2.1-Fun-1.3B-Control"
  huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-Control --local-dir "$WAN_DIR"
  prep "null_prompt_umt5.pt is NOT in the official repo (a cached null umT5 embedding) —"
  prep "copy it into $WAN_DIR/ from a machine that has it (~18 KB)."
  [ -f "$WAN_DIR/Wan2.1_VAE.pth" ]      || warn "missing $WAN_DIR/Wan2.1_VAE.pth"
  [ -f "$WAN_DIR/null_prompt_umt5.pt" ] || warn "missing $WAN_DIR/null_prompt_umt5.pt (required)"
}

# ─── 3. Datasets (download from public sources) ───────────────────────────────
fetch_droid() {                                  # -> $DATA_ROOT/droid_lerobot {data,meta,videos}
  [ "$DROID_ENABLE" = 1 ] || { log "droid: disabled"; return; }
  [ "$FETCH" = download ] || { log "droid: FETCH=$FETCH (using existing data)"; return; }
  mkdir -p "$DATA_ROOT/droid_lerobot"
  log "hf download lerobot/droid_1.0.1 (meta + $DROID_FILES data shards + videos)"
  local inc=(--include "meta/*")
  for i in $(seq 0 $((DROID_FILES-1))); do inc+=(--include "data/chunk-000/file-$(printf %03d $i).parquet"); done
  inc+=(--include "videos/*")
  huggingface-cli download lerobot/droid_1.0.1 --repo-type dataset --local-dir "$DATA_ROOT/droid_lerobot" "${inc[@]}"
  prep "lerobot/droid_1.0.1 is LeRobot v3.0 (data/chunk-000/file-*.parquet). The loader"
  prep "(DroidLeRobotDataset) expects v2.1 (data/chunk-*/episode_*.parquet). Convert to v2.1"
  prep "or update the loader before training on this download."
}

fetch_agibot() {                                 # -> $DATA_ROOT/agibot {observations,proprio_stats,parameters,task_info}
  [ "$AGIBOT_ENABLE" = 1 ] || { log "agibot: disabled"; return; }
  [ "$FETCH" = download ] || { log "agibot: FETCH=$FETCH (using existing data)"; return; }
  mkdir -p "$DATA_ROOT/agibot"
  log "hf download agibot-world/AgiBotWorld-Alpha (tasks=$AGIBOT_TASKS) -> matches our layout"
  local inc=(--include "task_info/*")
  if [ "$AGIBOT_TASKS" = all ]; then
    inc+=(--include "observations/*" --include "proprio_stats/*" --include "parameters/*")
  else
    # first N task ids (sorted) across the three per-task trees
    local tasks; tasks=$(python - "$AGIBOT_TASKS" <<'PY'
import sys; from huggingface_hub import HfApi
n=int(sys.argv[1]); api=HfApi()
ts=sorted(i.path.split("/")[-1] for i in api.list_repo_tree("agibot-world/AgiBotWorld-Alpha","observations",repo_type="dataset"))
print(" ".join(ts[:n]))
PY
)
    for t in $tasks; do inc+=(--include "observations/$t/*" --include "proprio_stats/$t/*" --include "parameters/$t/*"); done
  fi
  huggingface-cli download agibot-world/AgiBotWorld-Alpha --repo-type dataset --local-dir "$DATA_ROOT/agibot" "${inc[@]}"
  prep "Agibot loader reads *_aligned.json camera params; if Alpha ships only the un-aligned"
  prep "variants, run your alignment preprocessing before training."
}

fetch_egodex() {                                 # -> $DATA_ROOT/egodex_cdn/<part>/<task>/<n>.hdf5
  [ "$EGODEX_ENABLE" = 1 ] || { log "egodex: disabled"; return; }
  [ "$FETCH" = download ] || { log "egodex: FETCH=$FETCH (using existing data)"; return; }
  mkdir -p "$DATA_ROOT/egodex_cdn"
  for p in $EGODEX_PARTS; do
    log "egodex: download + unzip $p (Apple CDN, ~300GB each for part1-5)"
    curl -L "https://ml-site.cdn-apple.com/datasets/egodex/${p}.zip" -o "$DATA_ROOT/egodex_cdn/${p}.zip"
    unzip -q -o "$DATA_ROOT/egodex_cdn/${p}.zip" -d "$DATA_ROOT/egodex_cdn/" && rm -f "$DATA_ROOT/egodex_cdn/${p}.zip"
  done
}

fetch_abc() {                                    # -> $DATA_ROOT/abc_pp/<task>/episode_*/{*.mp4,states.npz}
  [ "$ABC_ENABLE" = 1 ] || { log "abc: disabled"; return; }
  [ "$FETCH" = download ] || { log "abc: FETCH=$FETCH (using existing data)"; return; }
  log "hf download XDOF/ABC-130k (raw train split) -> $DATA_ROOT/abc_raw"
  mkdir -p "$DATA_ROOT/abc_raw"
  huggingface-cli download XDOF/ABC-130k --repo-type dataset --local-dir "$DATA_ROOT/abc_raw" --include "data/train/*" "meta/*"
  prep "ABC-130k is raw. Preprocess it into abc_pp/<task>/episode_*/ (top.mp4,left_wrist.mp4,"
  prep "right_wrist.mp4,states.npz) with robot_wm/datasets/abc/preprocessing/abc_batch_preprocess.py"
  prep "  python -m robot_wm.datasets.abc.preprocessing.abc_batch_preprocess --src $DATA_ROOT/abc_raw --dst $DATA_ROOT/abc_pp"
}

fetch_datasets() { fetch_droid; fetch_agibot; fetch_egodex; fetch_abc; }

# ─── 4. Manifests (regenerated for THIS server; capped by *_LIMIT) ────────────
make_manifests() {
  log "Regenerating manifests under $DATA_ROOT"

  if [ "$EGODEX_ENABLE" = 1 ] && [ -d "$DATA_ROOT/egodex_cdn" ]; then
    find "$DATA_ROOT/egodex_cdn" -name '*.hdf5' | sort | cap - "$EGODEX_LIMIT" > "$DATA_ROOT/egodex_cdn/manifest.csv"
    log "  egodex : $(wc -l < "$DATA_ROOT/egodex_cdn/manifest.csv") episodes (limit=$EGODEX_LIMIT)"
  fi

  if [ "$ABC_ENABLE" = 1 ] && [ -d "$DATA_ROOT/abc_pp" ]; then
    find "$DATA_ROOT/abc_pp" -mindepth 2 -maxdepth 2 -type d -name 'episode_*' | sort | cap - "$ABC_LIMIT" \
      > "$DATA_ROOT/abc_pp/manifest.txt"
    log "  abc    : $(wc -l < "$DATA_ROOT/abc_pp/manifest.txt") episodes (limit=$ABC_LIMIT)"
  fi

  if [ "$AGIBOT_ENABLE" = 1 ] && [ -d "$DATA_ROOT/agibot/observations" ]; then
    # Preserve a curated/portable manifest if one is already present and no cap is requested.
    if [ -f "$DATA_ROOT/agibot/manifest.csv" ] && [ "$AGIBOT_LIMIT" = all ] && [ "$FETCH" = skip ]; then
      log "  agibot : $(( $(wc -l < "$DATA_ROOT/agibot/manifest.csv") - 1 )) episodes (preserved existing manifest)"
    else
      { echo "task_id,episode_id,dataset"
        for t in "$DATA_ROOT/agibot/observations"/*/; do
          [ -d "$t" ] || continue; tid=$(basename "$t")
          for e in "$t"*/; do [ -d "$e" ] && echo "${tid},$(basename "$e"),scr"; done
        done | cap - "$AGIBOT_LIMIT"
      } > "$DATA_ROOT/agibot/manifest.csv"
      log "  agibot : $(( $(wc -l < "$DATA_ROOT/agibot/manifest.csv") - 1 )) episodes (enumerated, limit=$AGIBOT_LIMIT)"
    fi
  fi

  if [ "$DROID_ENABLE" = 1 ] && [ -d "$DATA_ROOT/droid_lerobot/data" ]; then
    log "  droid  : $(find "$DATA_ROOT/droid_lerobot/data" -name 'episode_*.parquet' 2>/dev/null | wc -l) v2.1 episodes (no manifest; loader globs them)"
  fi
}

# ─── 5. Repoint repo paths if BASE != /scr/ravenh ─────────────────────────────
repoint_paths() {
  [ "$BASE" = /scr/ravenh ] && { log "BASE=/scr/ravenh — no path edits"; return; }
  log "Repointing /scr/ravenh -> $BASE in repo configs + module defaults"
  grep -rl '/scr/ravenh' "$REPO_DIR" --include='*.py' --include='*.yaml' 2>/dev/null \
    | grep -v '/wandb/' | xargs -r sed -i "s|/scr/ravenh|$BASE|g"
}

# ─── 6. Validate + launch hint ────────────────────────────────────────────────
validate() {
  log "Validating"
  local ok=1
  for f in "$WAN_DIR/Wan2.1_VAE.pth" "$WAN_DIR/diffusion_pytorch_model.safetensors" \
           "$WAN_DIR/null_prompt_umt5.pt" "$VIDEOX_DIR/config/wan2.1/wan_civitai.yaml"; do
    [ -e "$f" ] && echo "  ok  $f" || { echo "  MISSING  $f"; ok=0; }
  done
  [ "$EGODEX_ENABLE" = 1 ] && { [ -f "$DATA_ROOT/egodex_cdn/manifest.csv" ] && echo "  ok  egodex manifest" || { echo "  MISSING egodex manifest"; ok=0; }; }
  [ "$ABC_ENABLE" = 1 ]    && { [ -f "$DATA_ROOT/abc_pp/manifest.txt" ]     && echo "  ok  abc manifest"    || { echo "  MISSING abc manifest"; ok=0; }; }
  [ "$AGIBOT_ENABLE" = 1 ] && { [ -f "$DATA_ROOT/agibot/manifest.csv" ]     && echo "  ok  agibot manifest" || { echo "  MISSING agibot manifest"; ok=0; }; }
  [ "$DROID_ENABLE" = 1 ]  && { [ -d "$DATA_ROOT/droid_lerobot/data" ]      && echo "  ok  droid data"      || { echo "  MISSING droid data"; ok=0; }; }
  [ "$ok" = 1 ] && log "Validation passed." || warn "Validation found gaps (see above + any [PREP] notes)."
  cat <<EOF

Launch (from $REPO_DIR/projects/latent_action_models):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  torchrun --standalone --nproc_per_node=8 train.py \\
    +experiments_0908=ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml \\
    data_loader.batch_size=16
  (smoke: +experiments_0908=ravenhuang/wan-dit/wan_dit_smoke.yaml)
EOF
}

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
