#!/usr/bin/env bash
#
# setup_training.sh — provision a new server to train the Wan-DiT latent-action
# world model: env + Wan weights + VideoX-Fun, the 4 datasets, their manifests,
# path repointing, and validation.
#
# The active run trains on ABC + Agibot + DROID + EgoDex
# (config transformed_multi_abc_agibot_droid_egodex).
#
# Data is opt-in. The default is FETCH=skip with every dataset disabled, so a
# bare invocation cannot start a multi-terabyte transfer. Downloads require:
#   FETCH=download ALLOW_DATA_DOWNLOAD=1 <DATASET>_ENABLE=1
# plus a finite per-dataset limit. ABC and AgiBot additionally require exact,
# reviewed file plans. AgiBot's plan pins each official archive by SHA-256; the
# preparer extracts genuine aligned camera/base streams and refuses synthesis.
#
# Public sources:
#   DROID   https://huggingface.co/datasets/cadene/droid   (LeRobot v2.1, matches the loader;
#           NOTE lerobot/droid_1.0.1 is v3.0 file-packed parquet and is NOT loadable as-is)
#   Agibot  https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
#   ABC     https://huggingface.co/datasets/XDOF/ABC-130k
#   EgoDex  https://github.com/apple/ml-egodex  (zips on Apple's CDN)
#
# Usage:
#   ./setup_training.sh                 # safe default: no dataset download; fails validation if assets are absent
#   ./setup_training.sh env
#   FETCH=download WAN_ASSET_DEVICE=cuda:0 ./setup_training.sh weights
#   FETCH=download ALLOW_DATA_DOWNLOAD=1 DROID_ENABLE=1 DROID_LIMIT=10000 \
#     ./setup_training.sh datasets
#   FETCH=download ALLOW_DATA_DOWNLOAD=1 AGIBOT_ENABLE=1 AGIBOT_LIMIT=5671 \
#     AGIBOT_ARCHIVE_PLAN=/path/to/agibot_archives.plan \
#     AGIBOT_EPISODE_PLAN=/path/to/agibot_episodes.csv ./setup_training.sh datasets
#   FETCH=skip DROID_ENABLE=1 EGODEX_ENABLE=1 AGIBOT_ENABLE=1 ABC_ENABLE=1 \
#     ./setup_training.sh manifests
#   ./setup_training.sh manifests       # (re)generate manifests for present data
#   ./setup_training.sh validate
#
# Manifests store ABSOLUTE paths, so they are regenerated per-server here — never
# copy egodex/abc manifests between machines. (Agibot's manifest is path-portable.)

set -euo pipefail

# ─────────────────────────────── CONFIG ──────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="${BASE:-/mnt/data2/${USER}/lacwm_runtime}" # large data, weights, runtimes, and outputs
# These four are EXPORTED so the configs + module defaults (which read them via
# ${oc.env:VAR,default} / os.environ.get) resolve to this server.
export LACWM_DATA="${LACWM_DATA:-$BASE/data}"             # datasets
export LACWM_RUNS="${LACWM_RUNS:-$BASE/runs}"             # training outputs (run dirs)
export WAN_DIR="${WAN_DIR:-$BASE/wan_fun_1.3b_control}"   # Wan2.1-Fun weights + null prompt
export VIDEOX_HOME="${VIDEOX_HOME:-$BASE/VideoX-Fun-1d6d9c3}" # pinned VideoX-Fun checkout
DATA_ROOT="$LACWM_DATA"; VIDEOX_DIR="$VIDEOX_HOME"        # internal aliases
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
ENV_DIR="${ENV_DIR:-$BASE/envs/lacwm-b200-py310}"
PYTHON_BIN="${LACWM_PYTHON:-$ENV_DIR/bin/python}"

FETCH="${FETCH:-skip}"                        # download | skip
ALLOW_DATA_DOWNLOAD="${ALLOW_DATA_DOWNLOAD:-0}"

# Per-dataset opt-in and finite active counts. "all" is intentionally rejected.
DROID_ENABLE="${DROID_ENABLE:-0}";   DROID_LIMIT="${DROID_LIMIT:-10000}"
AGIBOT_ENABLE="${AGIBOT_ENABLE:-0}"; AGIBOT_LIMIT="${AGIBOT_LIMIT:-5671}"
EGODEX_ENABLE="${EGODEX_ENABLE:-0}"; EGODEX_LIMIT="${EGODEX_LIMIT:-10000}"
ABC_ENABLE="${ABC_ENABLE:-0}";       ABC_LIMIT="${ABC_LIMIT:-10000}"

# Download/preparation controls. These have no broad defaults.
EGODEX_PARTS="${EGODEX_PARTS:-}"               # explicit space-separated part1..part5/test/extra
EGODEX_SHA256_PLAN="${EGODEX_SHA256_PLAN:-}"   # lines: <part> <sha256>, required for downloads
ABC_DOWNLOAD_PLAN="${ABC_DOWNLOAD_PLAN:-}"     # exact HF paths, one episode.mcap per line
ABC_SUCCESS_MANIFEST="${ABC_SUCCESS_MANIFEST:-$DATA_ROOT/abc_pp/manifest.success.txt}"
AGIBOT_ARCHIVE_PLAN="${AGIBOT_ARCHIVE_PLAN:-}" # lines: <section> <HF .tar path> <sha256>
AGIBOT_EPISODE_PLAN="${AGIBOT_EPISODE_PLAN:-}" # exact task_id,episode_id[,dataset] CSV
AGIBOT_RAW_ROOT="${AGIBOT_RAW_ROOT:-$DATA_ROOT/agibot_raw}"
AGIBOT_SUCCESS_MANIFEST="${AGIBOT_SUCCESS_MANIFEST:-$DATA_ROOT/agibot/manifest.success.csv}"
REBUILD_AGIBOT_MANIFEST="${REBUILD_AGIBOT_MANIFEST:-0}"
DELETE_ARCHIVES_AFTER_EXTRACT="${DELETE_ARCHIVES_AFTER_EXTRACT:-0}"
VALIDATE_WORKERS="${VALIDATE_WORKERS:-16}"
MANIFEST_HELPER="$REPO_DIR/tools/build_training_manifests.py"
DATA_VALIDATOR="$REPO_DIR/tools/validate_training_data.py"
AGIBOT_PREPARER="$REPO_DIR/tools/prepare_agibot.py"
# ──────────────────────────────────────────────────────────────────────────────

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*"; }
prep() { printf '\033[1;36m[PREP]\033[0m %s\n'  "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n'  "$*" >&2; exit 1; }

require_positive_int() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive finite integer (got '$value')"
}

require_bool() {
  local name="$1" value="$2"
  [ "$value" = 0 ] || [ "$value" = 1 ] || die "$name must be 0 or 1 (got '$value')"
}

enabled_dataset_count() {
  echo $((DROID_ENABLE + AGIBOT_ENABLE + EGODEX_ENABLE + ABC_ENABLE))
}

validate_config() {
  local action="${1:-all}" part
  case "$FETCH" in download|skip) ;; *) die "FETCH must be 'download' or 'skip'" ;; esac
  require_bool DROID_ENABLE "$DROID_ENABLE"
  require_bool AGIBOT_ENABLE "$AGIBOT_ENABLE"
  require_bool EGODEX_ENABLE "$EGODEX_ENABLE"
  require_bool ABC_ENABLE "$ABC_ENABLE"
  require_bool ALLOW_DATA_DOWNLOAD "$ALLOW_DATA_DOWNLOAD"
  require_bool REBUILD_AGIBOT_MANIFEST "$REBUILD_AGIBOT_MANIFEST"
  require_bool DELETE_ARCHIVES_AFTER_EXTRACT "$DELETE_ARCHIVES_AFTER_EXTRACT"
  require_positive_int DROID_LIMIT "$DROID_LIMIT"
  require_positive_int AGIBOT_LIMIT "$AGIBOT_LIMIT"
  require_positive_int EGODEX_LIMIT "$EGODEX_LIMIT"
  require_positive_int ABC_LIMIT "$ABC_LIMIT"
  require_positive_int VALIDATE_WORKERS "$VALIDATE_WORKERS"
  [ -s "$MANIFEST_HELPER" ] || die "missing manifest helper: $MANIFEST_HELPER"
  [ -s "$DATA_VALIDATOR" ] || die "missing data validator: $DATA_VALIDATOR"
  [ -s "$AGIBOT_PREPARER" ] || die "missing AgiBot preparer: $AGIBOT_PREPARER"

  if [ "$action" = all ] || [ "$action" = validate ]; then
    [ "$DROID_ENABLE" = 1 ] && [ "$EGODEX_ENABLE" = 1 ] && \
      [ "$AGIBOT_ENABLE" = 1 ] && [ "$ABC_ENABLE" = 1 ] || die \
      "production validation requires all four datasets enabled"
    [ "$DROID_LIMIT" = 10000 ] && [ "$EGODEX_LIMIT" = 10000 ] && \
      [ "$AGIBOT_LIMIT" = 5671 ] && [ "$ABC_LIMIT" = 10000 ] || die \
      "production limits must be DROID=10000, EGODEX=10000, AGIBOT=5671, ABC=10000"
  fi

  if [ "$FETCH" = download ] && { [ "$action" = all ] || [ "$action" = datasets ]; }; then
    [ "$(enabled_dataset_count)" -gt 0 ] || die \
      "FETCH=download requires at least one explicit <DATASET>_ENABLE=1"
    require_download_ack
    if [ "$AGIBOT_ENABLE" = 1 ]; then
      [ -n "$AGIBOT_ARCHIVE_PLAN" ] || die \
        "AGIBOT_ARCHIVE_PLAN is required before any AgiBot download"
      [ -s "$AGIBOT_ARCHIVE_PLAN" ] || die \
        "AgiBot archive plan is missing or empty: $AGIBOT_ARCHIVE_PLAN"
      [ -n "$AGIBOT_EPISODE_PLAN" ] || die \
        "AGIBOT_EPISODE_PLAN is required before any AgiBot download"
      [ -s "$AGIBOT_EPISODE_PLAN" ] || die \
        "AgiBot episode plan is missing or empty: $AGIBOT_EPISODE_PLAN"
    fi
    if [ "$EGODEX_ENABLE" = 1 ]; then
      [ -n "$EGODEX_PARTS" ] || die "EGODEX_PARTS must explicitly list one or more parts"
      [ -s "$EGODEX_SHA256_PLAN" ] || die \
        "EGODEX_SHA256_PLAN with expected archive hashes is required"
      for part in $EGODEX_PARTS; do
        case "$part" in part1|part2|part3|part4|part5|test|extra) ;; *) die "unsupported EgoDex part '$part'" ;; esac
      done
    fi
    if [ "$ABC_ENABLE" = 1 ]; then
      [ -n "$ABC_DOWNLOAD_PLAN" ] || die \
        "ABC_DOWNLOAD_PLAN is required before any download begins"
      [ -s "$ABC_DOWNLOAD_PLAN" ] || die "ABC download plan is missing or empty: $ABC_DOWNLOAD_PLAN"
    fi
  fi
}

require_download_ack() {
  [ "$ALLOW_DATA_DOWNLOAD" = 1 ] || die \
    "dataset downloads are disabled; set ALLOW_DATA_DOWNLOAD=1 after reviewing limits and storage"
}

require_runtime() {
  [ -x "$PYTHON_BIN" ] || die \
    "training Python is missing at $PYTHON_BIN; run '$0 env' first or set LACWM_PYTHON"
}

# ─── 1. Environment ───────────────────────────────────────────────────────────
setup_env() {
  log "creating pinned PyTorch/CUDA 12.8 B200 environment at $ENV_DIR"
  LACWM_BASE="$BASE" ENV_DIR="$ENV_DIR" VIDEOX_HOME="$VIDEOX_HOME" \
    "$REPO_DIR/tools/env/create_b200_env.sh"
}

# ─── 2. Weights ───────────────────────────────────────────────────────────────
fetch_weights() {
  [ "$FETCH" = download ] || { log "weights: FETCH=$FETCH (using existing $WAN_DIR)"; return; }
  [ -n "${WAN_ASSET_DEVICE:-}" ] || die \
    "FETCH=download weights requires an explicit WAN_ASSET_DEVICE (for example cuda:0 or cpu)"
  log "preparing pinned Wan assets and cached null prompt"
  LACWM_BASE="$BASE" ENV_DIR="$ENV_DIR" WAN_DIR="$WAN_DIR" VIDEOX_HOME="$VIDEOX_HOME" \
    "$REPO_DIR/tools/env/prepare_wan_assets.sh" --device "$WAN_ASSET_DEVICE"
}

# ─── 3. Datasets (download from public sources) ───────────────────────────────
fetch_droid() {                                  # -> $DATA_ROOT/droid_lerobot {data,meta,videos}
  [ "$DROID_ENABLE" = 1 ] || { log "droid: disabled"; return; }
  [ "$FETCH" = download ] || { log "droid: FETCH=$FETCH (using existing data)"; return; }
  require_download_ack
  require_runtime
  mkdir -p "$DATA_ROOT/droid_lerobot"
  # cadene/droid is LeRobot v2.1 (data/chunk-*/episode_*.parquet + per-episode mp4 with the
  # exact camera names the loader uses) -> loads directly, no conversion. ~1000 episodes/chunk.
  local nchunks=$(( (DROID_LIMIT + 999) / 1000 ))
  log "hf download cadene/droid (v2.1): meta + $nchunks chunk(s) (~$((nchunks * 1000)) episodes)"
  local patterns=("meta/*") i c
  for i in $(seq 0 $((nchunks - 1))); do
    c=$(printf %03d "$i"); patterns+=("data/chunk-$c/*" "videos/chunk-$c/*")
  done
  "$PYTHON_BIN" - "$DATA_ROOT/droid_lerobot" "${patterns[@]}" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id="cadene/droid",
    repo_type="dataset",
    revision="c8fbc029c9786fd4377ea20d9535e86d88199c0f",
    local_dir=sys.argv[1],
    allow_patterns=sys.argv[2:],
)
PY
}

fetch_agibot() {                                 # -> $DATA_ROOT/agibot {observations,proprio_stats,parameters}
  [ "$AGIBOT_ENABLE" = 1 ] || { log "agibot: disabled"; return; }
  [ "$FETCH" = download ] || { log "agibot: FETCH=$FETCH (using existing data)"; return; }
  require_download_ack
  require_runtime
  [ -s "$AGIBOT_ARCHIVE_PLAN" ] || die \
    "AGIBOT_ARCHIVE_PLAN is required (section, exact HF .tar path, SHA-256 per line)"
  [ -s "$AGIBOT_EPISODE_PLAN" ] || die \
    "AGIBOT_EPISODE_PLAN is required (exact task_id,episode_id[,dataset] CSV)"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$AGIBOT_EPISODE_PLAN" "$AGIBOT_LIMIT" "$AGIBOT_ARCHIVE_PLAN" <<'PY'
import sys
from pathlib import Path
from tools.prepare_agibot import (
    parse_archive_plan,
    parse_episode_plan,
    verify_official_archive_plan,
)

episodes = parse_episode_plan(Path(sys.argv[1]), expected_dataset_id="scr")
expected = int(sys.argv[2])
if len(episodes) != expected:
    raise SystemExit(
        f"AgiBot episode plan has {len(episodes)} entries; expected exactly AGIBOT_LIMIT={expected}"
    )
verify_official_archive_plan(parse_archive_plan(Path(sys.argv[3])))
PY
  mkdir -p "$AGIBOT_RAW_ROOT" "$DATA_ROOT"
  log "agibot: downloading only checksummed archives in $AGIBOT_ARCHIVE_PLAN"
  "$PYTHON_BIN" - "$AGIBOT_ARCHIVE_PLAN" "$AGIBOT_RAW_ROOT" <<'PY'
import re
import sys
from pathlib import Path, PurePosixPath
from huggingface_hub import snapshot_download

plan, destination = Path(sys.argv[1]), sys.argv[2]
sections = {"observations", "parameters", "proprio_stats"}
sha = re.compile(r"[0-9a-fA-F]{64}")
paths = []
seen_sections = set()
for line_number, raw in enumerate(plan.read_text(encoding="utf-8").splitlines(), 1):
    value = raw.strip()
    if not value or value.startswith("#"):
        continue
    fields = value.split()
    if len(fields) != 3:
        raise SystemExit(f"{plan}:{line_number}: expected section, HF tar path, SHA-256")
    section, path, digest = fields
    pure = PurePosixPath(path)
    if section not in sections or pure.is_absolute() or ".." in pure.parts:
        raise SystemExit(f"{plan}:{line_number}: unsafe section/path")
    if pure.parts[0] != section or pure.suffix != ".tar" or not sha.fullmatch(digest):
        raise SystemExit(f"{plan}:{line_number}: invalid section/path/checksum")
    paths.append(pure.as_posix())
    seen_sections.add(section)
if not paths or len(paths) != len(set(paths)) or seen_sections != sections:
    raise SystemExit("AgiBot plan must contain unique archives covering all three sections")
snapshot_download(
    repo_id="agibot-world/AgiBotWorld-Alpha",
    repo_type="dataset",
    revision="128665c9e0244c45d1cbe5c13f5a4706afd24f27",
    local_dir=destination,
    allow_patterns=paths,
    token=True,
)
PY
  prep "agibot: verifying hashes, safely extracting, and deep-qualifying genuine motion streams"
  "$PYTHON_BIN" "$AGIBOT_PREPARER" \
    --root "$DATA_ROOT/agibot" \
    --archive-root "$AGIBOT_RAW_ROOT" \
    --archive-plan "$AGIBOT_ARCHIVE_PLAN" \
    --episode-plan "$AGIBOT_EPISODE_PLAN" \
    --limit "$AGIBOT_LIMIT" \
    --manifest "$DATA_ROOT/agibot/manifest.csv" \
    --success-manifest "$AGIBOT_SUCCESS_MANIFEST" \
    --report "$DATA_ROOT/agibot/preparation_report.json" \
    --execute
  if [ "$DELETE_ARCHIVES_AFTER_EXTRACT" = 1 ]; then
    "$PYTHON_BIN" - "$AGIBOT_ARCHIVE_PLAN" "$AGIBOT_RAW_ROOT" <<'PY'
import sys
from pathlib import Path, PurePosixPath
plan, root = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
for raw in plan.read_text(encoding="utf-8").splitlines():
    value = raw.strip()
    if not value or value.startswith("#"):
        continue
    relative = PurePosixPath(value.split()[1])
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f"refusing unsafe archive deletion: {path}")
    path.unlink()
PY
  fi
}

fetch_egodex() {                                 # -> $DATA_ROOT/egodex_cdn/<part>/<task>/<n>.hdf5
  [ "$EGODEX_ENABLE" = 1 ] || { log "egodex: disabled"; return; }
  [ "$FETCH" = download ] || { log "egodex: FETCH=$FETCH (using existing data)"; return; }
  require_download_ack
  [ -n "$EGODEX_PARTS" ] || die "EGODEX_PARTS must explicitly list one or more parts"
  [ -s "$EGODEX_SHA256_PLAN" ] || die "EGODEX_SHA256_PLAN is missing or empty"
  command -v curl >/dev/null || die "curl not found"
  command -v unzip >/dev/null || die "unzip not found"
  command -v sha256sum >/dev/null || die "sha256sum not found"
  mkdir -p "$DATA_ROOT/egodex_cdn"
  local p archive partial expected_hash actual_hash
  for p in $EGODEX_PARTS; do
    case "$p" in part1|part2|part3|part4|part5|test|extra) ;; *) die "unsupported EgoDex part '$p'" ;; esac
    log "egodex: download + unzip $p (Apple CDN, ~300GB each for part1-5)"
    archive="$DATA_ROOT/egodex_cdn/${p}.zip"
    partial="${archive}.partial"
    expected_hash="$(awk -v part="$p" '$1 == part {print $2}' "$EGODEX_SHA256_PLAN")"
    [[ "$expected_hash" =~ ^[0-9a-fA-F]{64}$ ]] || die \
      "EGODEX_SHA256_PLAN must contain exactly one valid SHA-256 for $p"
    curl --fail --show-error --location "https://ml-site.cdn-apple.com/datasets/egodex/${p}.zip" -o "$partial"
    actual_hash="$(sha256sum "$partial" | awk '{print $1}')"
    [ "${actual_hash,,}" = "${expected_hash,,}" ] || die \
      "EgoDex $p checksum mismatch: $actual_hash != $expected_hash"
    unzip -tq "$partial" >/dev/null || die "downloaded EgoDex archive failed integrity check: $partial"
    mv -f "$partial" "$archive"
    unzip -q -o "$archive" -d "$DATA_ROOT/egodex_cdn/"
    if [ "$DELETE_ARCHIVES_AFTER_EXTRACT" = 1 ]; then
      rm -f -- "$archive"
    fi
  done
}

fetch_abc() {                                    # -> $DATA_ROOT/abc_pp/<task>/episode_*/{*.mp4,states.npz}
  [ "$ABC_ENABLE" = 1 ] || { log "abc: disabled"; return; }
  [ "$FETCH" = download ] || { log "abc: FETCH=$FETCH (using existing data)"; return; }
  require_download_ack
  require_runtime
  [ -n "$ABC_DOWNLOAD_PLAN" ] || die \
    "ABC_DOWNLOAD_PLAN is required; provide exactly ABC_LIMIT HF episode.mcap paths, one per line"
  [ -s "$ABC_DOWNLOAD_PLAN" ] || die "ABC download plan is missing or empty: $ABC_DOWNLOAD_PLAN"
  mkdir -p "$DATA_ROOT/abc_raw"
  log "abc: selectively downloading $ABC_LIMIT planned MCAP episodes -> $DATA_ROOT/abc_raw"
  "$PYTHON_BIN" - "$ABC_DOWNLOAD_PLAN" "$ABC_LIMIT" "$DATA_ROOT/abc_raw" <<'PY'
import re
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

plan_path, expected, local_dir = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
patterns = [line.strip() for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
valid = re.compile(r"^data/train/[^/]+/episode_[^/]+/episode\.mcap$")
bad = [path for path in patterns if not valid.fullmatch(path)]
if bad:
    raise SystemExit(f"invalid ABC plan paths (first 5): {bad[:5]}")
if len(patterns) != expected:
    raise SystemExit(f"ABC plan has {len(patterns)} paths; expected exactly {expected}")
if len(set(patterns)) != len(patterns):
    raise SystemExit("ABC plan contains duplicate paths")
snapshot_download(
    repo_id="XDOF/ABC-130k",
    repo_type="dataset",
    revision="fad18a5f891a47e665756d4cab2a67a7a080d8bb",
    local_dir=local_dir,
    allow_patterns=patterns,
)

active_plan = Path(local_dir) / "plan.active.txt"
temporary = active_plan.with_suffix(active_plan.suffix + ".tmp")
temporary.write_text(
    "".join(str((Path(local_dir) / item).resolve()) + "\n" for item in patterns),
    encoding="utf-8",
)
temporary.replace(active_plan)
PY
  # Preprocess mcap -> abc_pp/<task>/<ep>/{top,left_wrist,right_wrist}.mp4 + states.npz
  log "abc: preprocessing planned MCAPs; success provenance stays in $ABC_SUCCESS_MANIFEST"
  ABC_RAW="$DATA_ROOT/abc_raw/data/train" ABC_PP="$DATA_ROOT/abc_pp" \
    ABC_INPUT_MANIFEST="$DATA_ROOT/abc_raw/plan.active.txt" ABC_MANIFEST="$ABC_SUCCESS_MANIFEST" \
    "$PYTHON_BIN" -m robot_wm.datasets.abc.preprocessing.abc_batch_preprocess
}

fetch_datasets() {
  if [ "$(enabled_dataset_count)" -eq 0 ]; then
    [ "$FETCH" = skip ] && { log "no datasets enabled; nothing to fetch"; return; }
    die "FETCH=download requires at least one explicit <DATASET>_ENABLE=1"
  fi
  [ "$FETCH" = skip ] || require_download_ack
  fetch_droid
  fetch_agibot
  fetch_egodex
  fetch_abc
}

# ─── 4. Manifests (atomic, finite, complete assets only) ─────────────────────
make_manifests() {
  require_runtime
  [ "$(enabled_dataset_count)" -gt 0 ] || die "no datasets enabled; set at least one <DATASET>_ENABLE=1"
  log "Building finite manifests under $DATA_ROOT"

  if [ "$EGODEX_ENABLE" = 1 ]; then
    [ -n "$EGODEX_PARTS" ] || die \
      "EGODEX_PARTS must list the extracted parts used by the runtime manifest"
    local egodex_args=() p
    for p in $EGODEX_PARTS; do
      case "$p" in part1|part2|part3|part4|part5|test|extra) ;; *) die "unsupported EgoDex part '$p'" ;; esac
      egodex_args+=(--include-root "$DATA_ROOT/egodex_cdn/$p")
    done
    "$PYTHON_BIN" "$MANIFEST_HELPER" egodex \
      --root "$DATA_ROOT/egodex_cdn" \
      --output "$DATA_ROOT/egodex_cdn/manifest.csv" \
      --limit "$EGODEX_LIMIT" \
      "${egodex_args[@]}"
  fi

  if [ "$ABC_ENABLE" = 1 ]; then
    [ -s "$ABC_SUCCESS_MANIFEST" ] || die \
      "ABC preprocessing success manifest is missing/empty: $ABC_SUCCESS_MANIFEST (partial directories are never enumerated)"
    "$PYTHON_BIN" "$MANIFEST_HELPER" abc \
      --success-manifest "$ABC_SUCCESS_MANIFEST" \
      --output "$DATA_ROOT/abc_pp/manifest.txt" \
      --limit "$ABC_LIMIT"
  fi

  if [ "$AGIBOT_ENABLE" = 1 ]; then
    if [ "$REBUILD_AGIBOT_MANIFEST" = 1 ]; then
      die "REBUILD_AGIBOT_MANIFEST cannot certify an existing unbound tree. Use the checksummed AGIBOT_ARCHIVE_PLAN + exact AGIBOT_EPISODE_PLAN flow with a new output root."
    else
      [ -s "$DATA_ROOT/agibot/manifest.csv" ] || die \
        "AgiBot manifest missing/empty. Supply the curated extracted/aligned manifest, or set REBUILD_AGIBOT_MANIFEST=1 after preparation."
      log "agibot: preserving curated manifest; strict validation will enforce cap=$AGIBOT_LIMIT and aligned assets"
    fi
  fi

  if [ "$DROID_ENABLE" = 1 ]; then
    [ -d "$DATA_ROOT/droid_lerobot/data" ] || die "DROID data directory missing: $DATA_ROOT/droid_lerobot/data"
    log "droid: no manifest; strict validation will inspect the loader glob with cap=$DROID_LIMIT"
  fi
}

# ─── 5. Write a sourceable env file (configs read these vars at train time) ───
write_env() {
  local f="$REPO_DIR/.lacwm_env"
  {
    printf '%s\n' '# Source before training/evaluation so configs resolve to this server paths.'
    printf 'export LACWM_DATA=%q\n' "$LACWM_DATA"
    printf 'export LACWM_RUNS=%q\n' "$LACWM_RUNS"
    printf 'export WAN_DIR=%q\n' "$WAN_DIR"
    printf 'export VIDEOX_HOME=%q\n' "$VIDEOX_HOME"
    printf 'export LACWM_PYTHON=%q\n' "$PYTHON_BIN"
    printf 'export LACWM_BASE=%q\n' "$BASE"
  } > "$f"
  log "wrote $f  (run 'source $f' before launching training)"
}

# ─── 6. Strict validation gate + launch hint ─────────────────────────────────
validate() {
  require_runtime
  log "Validating exact runtime/assets, manifests, fields, views, and production counts"
  VIDEOX_HOME="$VIDEOX_HOME" \
  PYTHONPATH="$REPO_DIR/tools/env/videox_shim:$VIDEOX_HOME:$REPO_DIR/projects/latent_action_models:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$REPO_DIR/tools/env/verify_b200_runtime.py" --wan-dir "$WAN_DIR"

  [ "$(enabled_dataset_count)" -gt 0 ] || die \
    "no datasets enabled; explicitly set the datasets that must pass the launch gate"

  local datasets=() validator_args=(--data-root "$DATA_ROOT" --workers "$VALIDATE_WORKERS")
  if [ "$DROID_ENABLE" = 1 ]; then
    datasets+=(droid)
    validator_args+=(--droid-cap "$DROID_LIMIT" --droid-expected "$DROID_LIMIT")
  fi
  if [ "$EGODEX_ENABLE" = 1 ]; then
    datasets+=(egodex)
    validator_args+=(--egodex-cap "$EGODEX_LIMIT" --egodex-expected "$EGODEX_LIMIT")
  fi
  if [ "$AGIBOT_ENABLE" = 1 ]; then
    datasets+=(agibot)
    # The Hydra loader cap is 10k; the curated official corpus contains 5,671.
    validator_args+=(--agibot-cap 10000 --agibot-expected "$AGIBOT_LIMIT" --agibot-profile production)
  fi
  if [ "$ABC_ENABLE" = 1 ]; then
    datasets+=(abc)
    validator_args+=(--abc-cap "$ABC_LIMIT" --abc-expected "$ABC_LIMIT")
  fi

  "$PYTHON_BIN" "$DATA_VALIDATOR" "${validator_args[@]}" --datasets "${datasets[@]}" || die \
    "strict training-data preflight failed; no training launch should proceed"
  log "Strict validation passed."
  cat <<EOF

Next gates:
  source $REPO_DIR/.lacwm_env
  source $ENV_DIR/bin/activate
  source $REPO_DIR/tools/env/activate_b200.sh
  Review $REPO_DIR/tools/README.md, then run the real-data gradient smoke and
  guarded tools/launch_8xb200.sh dry run. Do not bypass those launch gates.
EOF
}

ACTION="${1:-all}"
validate_config "$ACTION"

case "$ACTION" in
  all)       setup_env; fetch_weights; fetch_datasets; make_manifests; write_env; validate ;;
  env)       setup_env ;;
  weights)   fetch_weights ;;
  datasets)  fetch_datasets ;;
  manifests) make_manifests ;;
  paths)     write_env ;;
  validate)  validate ;;
  *) die "usage: $0 [all|env|weights|datasets|manifests|paths|validate]" ;;
esac
