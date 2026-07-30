#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

readonly TRAIN_CLIPS=512
readonly VAL_CLIPS=64
readonly TEST_CLIPS=128
readonly CLIPS_PER_EPISODE=1
readonly BUILD_SEED=20260729
readonly PCA_MAX_CLIPS=256
readonly PCA_MAX_TOKENS=250000
readonly CHECKPOINT_BYTES=1664223428
readonly SOURCE_COMMIT=45d025f636dfc58fc2426905fc4a1ab755b1c3e5

die() {
  echo "ERROR: $*" >&2
  exit 2
}

BUILD_ID=""
EXPECTED_COMMIT=""
REPO_ROOT=""
BUILD_ROOT=""
EPISODE_MANIFEST=""
EPISODE_MANIFEST_SHA256=""
EXTRACTOR_PYTHON=""
VJEPA_SOURCE=""
VJEPA_CHECKPOINT=""
VJEPA_CHECKPOINT_SHA256=""

while (($#)); do
  case "$1" in
    --build-id) [[ $# -ge 2 ]] || die "--build-id requires a value"; BUILD_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "--repo-root requires a value"; REPO_ROOT="$2"; shift 2 ;;
    --build-root) [[ $# -ge 2 ]] || die "--build-root requires a value"; BUILD_ROOT="$2"; shift 2 ;;
    --episode-manifest) [[ $# -ge 2 ]] || die "--episode-manifest requires a value"; EPISODE_MANIFEST="$2"; shift 2 ;;
    --episode-manifest-sha256) [[ $# -ge 2 ]] || die "--episode-manifest-sha256 requires a value"; EPISODE_MANIFEST_SHA256="$2"; shift 2 ;;
    --extractor-python) [[ $# -ge 2 ]] || die "--extractor-python requires a value"; EXTRACTOR_PYTHON="$2"; shift 2 ;;
    --vjepa-source) [[ $# -ge 2 ]] || die "--vjepa-source requires a value"; VJEPA_SOURCE="$2"; shift 2 ;;
    --vjepa-checkpoint) [[ $# -ge 2 ]] || die "--vjepa-checkpoint requires a value"; VJEPA_CHECKPOINT="$2"; shift 2 ;;
    --vjepa-checkpoint-sha256) [[ $# -ge 2 ]] || die "--vjepa-checkpoint-sha256 requires a value"; VJEPA_CHECKPOINT_SHA256="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "build ID:$BUILD_ID" \
  "expected commit:$EXPECTED_COMMIT" \
  "repository root:$REPO_ROOT" \
  "build root:$BUILD_ROOT" \
  "episode manifest:$EPISODE_MANIFEST" \
  "episode-manifest SHA-256:$EPISODE_MANIFEST_SHA256" \
  "extractor Python:$EXTRACTOR_PYTHON" \
  "V-JEPA source:$VJEPA_SOURCE" \
  "V-JEPA checkpoint:$VJEPA_CHECKPOINT" \
  "V-JEPA checkpoint SHA-256:$VJEPA_CHECKPOINT_SHA256"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

[[ "$BUILD_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$ ]] || die "unsafe build ID"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit must be full lowercase SHA-1"
[[ "$EPISODE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid episode-manifest SHA-256"
[[ "$VJEPA_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid checkpoint SHA-256"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires a Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "cache builds are non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || die "cache build requires exactly one node"
[[ "$REPO_ROOT" == /* && "$BUILD_ROOT" == /* ]] || die "repository/build roots must be absolute"
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || die "invalid repository root"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || die "repository root is not canonical"
[[ -d "$BUILD_ROOT" && ! -L "$BUILD_ROOT" ]] || die "build root is missing or symlinked"
[[ "$(cd "$BUILD_ROOT" && pwd -P)" == "$BUILD_ROOT" ]] || die "build root is not canonical"
[[ -f "$EPISODE_MANIFEST" && ! -L "$EPISODE_MANIFEST" ]] || die "episode manifest is unavailable"
[[ -f "$VJEPA_CHECKPOINT" && ! -L "$VJEPA_CHECKPOINT" ]] || die "checkpoint is unavailable"
[[ -x "$EXTRACTOR_PYTHON" && ! -L "$EXTRACTOR_PYTHON" ]] || die "extractor Python is unavailable"
[[ -d "$VJEPA_SOURCE/.git" && ! -L "$VJEPA_SOURCE" ]] || die "V-JEPA source is unavailable"

readonly EXTRACTOR="$REPO_ROOT/tools/extract_vjepa2_targets.py"
readonly MANIFEST_BUILDER="$REPO_ROOT/tools/build_vjepa2_clip_manifests.py"
readonly REQUEST="$BUILD_ROOT/build_request.json"
readonly COMPLETE="$BUILD_ROOT/complete.json"
[[ -f "$EXTRACTOR" && -f "$MANIFEST_BUILDER" ]] || die "repository lacks V-JEPA cache tools"
[[ -f "$REQUEST" && ! -L "$REQUEST" ]] || die "immutable build request is missing"
[[ ! -e "$COMPLETE" ]] || die "cache build is already marked complete"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit changed: $ACTUAL_COMMIT != $EXPECTED_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository is dirty: ${GIT_STATUS//$'\n'/; }"
[[ "$(git -C "$VJEPA_SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || \
  die "V-JEPA source commit changed"
SOURCE_STATUS="$(git -C "$VJEPA_SOURCE" status --porcelain --untracked-files=all)"
[[ -z "$SOURCE_STATUS" ]] || die "V-JEPA source is dirty: ${SOURCE_STATUS//$'\n'/; }"
[[ "$(stat -c %s "$VJEPA_CHECKPOINT")" == "$CHECKPOINT_BYTES" ]] || \
  die "V-JEPA checkpoint byte count changed"
[[ "$(sha256sum "$VJEPA_CHECKPOINT" | awk '{print $1}')" == "$VJEPA_CHECKPOINT_SHA256" ]] || \
  die "V-JEPA checkpoint SHA-256 changed"
[[ "$(sha256sum "$EPISODE_MANIFEST" | awk '{print $1}')" == "$EPISODE_MANIFEST_SHA256" ]] || \
  die "episode manifest SHA-256 changed"

"$EXTRACTOR_PYTHON" - \
  "$REQUEST" "$BUILD_ID" "$EXPECTED_COMMIT" "$REPO_ROOT" "$BUILD_ROOT" \
  "$EPISODE_MANIFEST" "$EPISODE_MANIFEST_SHA256" "$EXTRACTOR_PYTHON" \
  "$VJEPA_SOURCE" "$VJEPA_CHECKPOINT" "$VJEPA_CHECKPOINT_SHA256" <<'PY'
import json
import sys
from pathlib import Path

(
    request_path,
    build_id,
    commit,
    repo,
    build_root,
    episode_manifest,
    episode_sha,
    python,
    source,
    checkpoint,
    checkpoint_sha,
) = sys.argv[1:]
with Path(request_path).open(encoding="utf-8") as handle:
    actual = json.load(handle)
expected = {
    "artifact_type": "vjepa2.1-immutable-cache-build-request",
    "format_version": 1,
    "build_id": build_id,
    "git_commit": commit,
    "repo_root": repo,
    "build_root": build_root,
    "episode_manifest": episode_manifest,
    "episode_manifest_sha256": episode_sha,
    "extractor_python": python,
    "vjepa_source": source,
    "vjepa_source_commit": "45d025f636dfc58fc2426905fc4a1ab755b1c3e5",
    "vjepa_checkpoint": checkpoint,
    "vjepa_checkpoint_sha256": checkpoint_sha,
    "train_clips": 512,
    "val_clips": 64,
    "test_clips": 128,
    "clips_per_episode": 1,
    "seed": 20260729,
    "pca_max_clips": 256,
    "pca_max_tokens": 250000,
}
if actual != expected:
    raise SystemExit("immutable build request differs from this allocation")
PY

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "$GPU_COUNT" == "8" ]] || die "full-node cache build expected 8 GPUs, found $GPU_COUNT"
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -vi 'B200' >/dev/null; then
  die "cache build requires an all-B200 node"
fi

export PYTHONPATH="$REPO_ROOT:$VJEPA_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$BUILD_ROOT/runtime_cache/torch"
export HF_HOME="$BUILD_ROOT/runtime_cache/huggingface"
export XDG_CACHE_HOME="$BUILD_ROOT/runtime_cache/xdg"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$BUILD_ROOT/job_logs"

ATTEMPT_RECORD="$BUILD_ROOT/attempt_${SLURM_JOB_ID}.json"
"$EXTRACTOR_PYTHON" - "$ATTEMPT_RECORD" "$SLURM_JOB_ID" <<'PY'
import datetime
import json
import sys
from pathlib import Path

payload = {
    "artifact_type": "vjepa2.1-cache-build-attempt",
    "format_version": 1,
    "slurm_job_id": sys.argv[2],
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

"$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-inputs \
  --source-path "$VJEPA_SOURCE" \
  --checkpoint "$VJEPA_CHECKPOINT" \
  --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"

MANIFEST_DIR="$BUILD_ROOT/manifests"
MANIFEST_METADATA="$MANIFEST_DIR/manifest_metadata.json"
if [[ -e "$MANIFEST_DIR" ]]; then
  [[ -d "$MANIFEST_DIR" && ! -L "$MANIFEST_DIR" && -f "$MANIFEST_METADATA" ]] || \
    die "existing manifest directory is incomplete"
  "$EXTRACTOR_PYTHON" "$MANIFEST_BUILDER" validate \
    --metadata "$MANIFEST_METADATA" \
    --verify-assets
else
  TEMP_MANIFEST_DIR="$BUILD_ROOT/.manifests.${SLURM_JOB_ID}"
  [[ ! -e "$TEMP_MANIFEST_DIR" ]] || die "fresh temporary manifest path already exists"
  "$EXTRACTOR_PYTHON" "$MANIFEST_BUILDER" build \
    --episode-manifest "$EPISODE_MANIFEST" \
    --output-dir "$TEMP_MANIFEST_DIR" \
    --train-clips "$TRAIN_CLIPS" \
    --val-clips "$VAL_CLIPS" \
    --test-clips "$TEST_CLIPS" \
    --seed "$BUILD_SEED" \
    --clips-per-episode "$CLIPS_PER_EPISODE"
  "$EXTRACTOR_PYTHON" "$MANIFEST_BUILDER" validate \
    --metadata "$TEMP_MANIFEST_DIR/manifest_metadata.json" \
    --verify-assets
  mv "$TEMP_MANIFEST_DIR" "$MANIFEST_DIR"
fi

"$EXTRACTOR_PYTHON" - "$MANIFEST_METADATA" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    metadata = json.load(handle)
expected_counts = {"train": 512, "val": 64, "test": 128}
if metadata.get("seed") != 20260729 or metadata.get("clips_per_episode") != 1:
    raise SystemExit("manifest seed/clips-per-episode contract mismatch")
for split, count in expected_counts.items():
    if metadata.get("splits", {}).get(split, {}).get("clip_count") != count:
        raise SystemExit(f"{split} manifest count mismatch")
PY

TRAIN_MANIFEST="$MANIFEST_DIR/train.jsonl"
VAL_MANIFEST="$MANIFEST_DIR/val.jsonl"
TEST_MANIFEST="$MANIFEST_DIR/test.jsonl"
PCA_DIR="$BUILD_ROOT/pca"
PCA_PATH="$PCA_DIR/train_pca64.pt"
PCA_METADATA="$PCA_PATH.json"
if [[ -e "$PCA_DIR" ]]; then
  [[ -d "$PCA_DIR" && ! -L "$PCA_DIR" && -f "$PCA_PATH" && -f "$PCA_METADATA" ]] || \
    die "existing PCA directory is incomplete"
  "$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-pca \
    --pca "$PCA_PATH" \
    --train-manifest "$TRAIN_MANIFEST"
else
  TEMP_PCA_DIR="$BUILD_ROOT/.pca.${SLURM_JOB_ID}"
  [[ ! -e "$TEMP_PCA_DIR" ]] || die "fresh temporary PCA path already exists"
  mkdir "$TEMP_PCA_DIR"
  TEMP_PCA_PATH="$TEMP_PCA_DIR/train_pca64.pt"
  CUDA_VISIBLE_DEVICES=0 "$EXTRACTOR_PYTHON" "$EXTRACTOR" fit-pca \
    --source-path "$VJEPA_SOURCE" \
    --checkpoint "$VJEPA_CHECKPOINT" \
    --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256" \
    --train-manifest "$TRAIN_MANIFEST" \
    --output "$TEMP_PCA_PATH" \
    --device cuda \
    --pca-device cuda \
    --encoder-dtype bfloat16 \
    --batch-size 1 \
    --max-tokens "$PCA_MAX_TOKENS" \
    --max-clips "$PCA_MAX_CLIPS" \
    --seed "$BUILD_SEED"
  "$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-pca \
    --pca "$TEMP_PCA_PATH" \
    --train-manifest "$TRAIN_MANIFEST"
  mv "$TEMP_PCA_DIR" "$PCA_DIR"
fi

"$EXTRACTOR_PYTHON" - "$PCA_PATH" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
expected = {
    "pca_clip_count": 256,
    "sampled_token_count": 250000,
    "pca_seed": 20260729,
    "target_channels": 64,
    "source_dimension": 768,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"PCA contract mismatch for {key}: {payload.get(key)!r} != {value!r}")
PY

mkdir -p "$BUILD_ROOT/caches"
SPLITS=(train val test)
MANIFESTS=("$TRAIN_MANIFEST" "$VAL_MANIFEST" "$TEST_MANIFEST")
GPUS=(0 1 2)
PIDS=()
EXTRACT_LOGS=()
for index in 0 1 2; do
  split="${SPLITS[$index]}"
  manifest="${MANIFESTS[$index]}"
  gpu="${GPUS[$index]}"
  cache_dir="$BUILD_ROOT/caches/$split"
  log="$BUILD_ROOT/job_logs/extract_${split}_${SLURM_JOB_ID}.log"
  EXTRACT_LOGS+=("$log")
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$EXTRACTOR_PYTHON" "$EXTRACTOR" extract \
      --source-path "$VJEPA_SOURCE" \
      --checkpoint "$VJEPA_CHECKPOINT" \
      --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256" \
      --split "$split" \
      --clip-manifest "$manifest" \
      --train-manifest "$TRAIN_MANIFEST" \
      --pca "$PCA_PATH" \
      --cache-dir "$cache_dir" \
      --device cuda \
      --encoder-dtype bfloat16 \
      --batch-size 1 \
      --finite-check-rows 8
  ) >"$log" 2>&1 &
  PIDS+=("$!")
done

EXTRACTION_FAILED=0
for index in 0 1 2; do
  if ! wait "${PIDS[$index]}"; then
    echo "Extraction failed for ${SPLITS[$index]}; log: ${EXTRACT_LOGS[$index]}" >&2
    EXTRACTION_FAILED=1
  fi
done
((EXTRACTION_FAILED == 0)) || die "one or more split extractions failed"
for log in "${EXTRACT_LOGS[@]}"; do
  echo "Completed extraction log: $log"
  tail -n 5 "$log"
done

for index in 0 1 2; do
  split="${SPLITS[$index]}"
  manifest="${MANIFESTS[$index]}"
  "$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-cache \
    --cache-metadata "$BUILD_ROOT/caches/$split/metadata.json" \
    --clip-manifest "$manifest" \
    --train-manifest "$TRAIN_MANIFEST" \
    --pca "$PCA_PATH" \
    --source-path "$VJEPA_SOURCE" \
    --checkpoint "$VJEPA_CHECKPOINT" \
    --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256" \
    --finite-check-rows 8
done

"$EXTRACTOR_PYTHON" - \
  "$COMPLETE" "$REQUEST" "$MANIFEST_METADATA" "$PCA_PATH" \
  "$BUILD_ROOT/caches/train/metadata.json" \
  "$BUILD_ROOT/caches/val/metadata.json" \
  "$BUILD_ROOT/caches/test/metadata.json" \
  "$SLURM_JOB_ID" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

(
    output,
    request_path,
    manifest_metadata,
    pca_path,
    train_metadata,
    val_metadata,
    test_metadata,
    job_id,
) = sys.argv[1:]

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

with Path(request_path).open(encoding="utf-8") as handle:
    request = json.load(handle)
splits = {}
for name, path in (
    ("train", train_metadata),
    ("val", val_metadata),
    ("test", test_metadata),
):
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("complete") is not True:
        raise SystemExit(f"{name} cache is not complete")
    splits[name] = {
        "metadata": str(Path(path).resolve()),
        "metadata_sha256": digest(path),
        "cache_id": metadata["cache_id"],
        "clip_count": metadata["clip_count"],
        "target_sha256": metadata["target_sha256"],
        "rgb_sha256": metadata["rgb_sha256"],
        "actions_sha256": metadata["actions_sha256"],
    }
payload = {
    "artifact_type": "vjepa2.1-immutable-cache-build",
    "format_version": 1,
    "build_request_sha256": digest(request_path),
    "build_id": request["build_id"],
    "git_commit": request["git_commit"],
    "slurm_job_id": job_id,
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest_metadata": str(Path(manifest_metadata).resolve()),
    "manifest_metadata_sha256": digest(manifest_metadata),
    "pca": str(Path(pca_path).resolve()),
    "pca_sha256": digest(pca_path),
    "splits": splits,
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Immutable V-JEPA cache build complete: $COMPLETE"
