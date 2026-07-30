#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/build_vjepa2_immutable_cache.sbatch"
EXTRACTOR="$REPO_ROOT/tools/extract_vjepa2_targets.py"
MANIFEST_BUILDER="$REPO_ROOT/tools/build_vjepa2_clip_manifests.py"

LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
EXTRACTOR_PYTHON="${VJEPA_EXTRACTOR_PYTHON:-$LACWM_BASE/envs/vjepa2-extractor-py311/bin/python}"
EPISODE_MANIFEST="${ABC_EPISODE_MANIFEST:-$LACWM_BASE/data/production_v1/fast_mixed_user_waived_v1/abc_pp/manifest.txt}"
VJEPA_SOURCE="${VJEPA_SOURCE:-$LACWM_BASE/assets/vjepa2/source-release-45d025f636dfc58fc2426905fc4a1ab755b1c3e5}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-$LACWM_BASE/assets/vjepa2/45d025f636dfc58fc2426905fc4a1ab755b1c3e5/vjepa2_1_vitb_dist_vitG_384.pt}"
VJEPA_CHECKPOINT_SHA256="${VJEPA_CHECKPOINT_SHA256:-848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d}"
ARTIFACT_ROOT="${VJEPA_CACHE_BUILD_ROOT:-$LACWM_BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds}"

PARTITION="batch"
TIME_LIMIT="04:00:00"
CPUS="160"
MEMORY="1000G"
ACCOUNT=""
QOS=""
BUILD_ID=""
BUILD_ID_EXPLICIT=0
EXPECTED_COMMIT=""
RESUME_EXISTING=0
EXECUTE=0
ALLOW_ACTIVE_JOB_IDS=()

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_build_vjepa2_immutable_cache.sh [options]

Dry-run or submit the pinned V-JEPA 2.1 offline cache build:

  train/val/test clips: 512 / 64 / 128
  split isolation:      one deterministic clip per episode
  manifest seed:        20260729
  PCA:                  first 256 train clips, 250000 sampled tokens
  target:               PCA64-whitened [64,4,24,120] FP16
  trainer inputs:       exact paired FP16 RGB and FP32 explicit actions

The initial build root must be fresh and external to the Git repository.
Extraction never passes --overwrite, so an interrupted per-split cache resumes
from its atomically recorded written_rows boundary. To resubmit an incomplete
root, pass the same explicit --build-id with --resume-existing; the immutable
request must match byte for byte and complete builds cannot be resubmitted.

Options:
  --build-id ID                 Build ID (default: UTC timestamp + commit)
  --expected-commit SHA         Require this exact clean 40-character commit
  --artifact-root PATH          Parent for fresh build directories
  --episode-manifest PATH       ABC preprocessing-success manifest
  --extractor-python PATH       Python >=3.11 V-JEPA extraction environment
  --vjepa-source PATH           Clean official source at the pinned commit
  --vjepa-checkpoint PATH       Official V-JEPA 2.1 ViT-B checkpoint
  --vjepa-checkpoint-sha256 HEX Full pinned checkpoint SHA-256
  --allow-active-job-id ID      Allow one existing numeric Slurm job ID;
                                repeat for every unrelated active job
  --partition NAME              Default: batch
  --time HH:MM:SS               Default/max: 04:00:00
  --cpus N                      Default: 160
  --mem VALUE                   Default: 1000G
  --account NAME                Optional Slurm account
  --qos NAME                    Optional Slurm QOS
  --resume-existing             Resume the matching incomplete build root
  --execute                     Create/validate provenance and submit; without
                                this flag all checks are read-only
  -h, --help

The allocation is one exclusive node with all eight B200 GPUs, is never
requeued, and never stops or changes any pre-existing job.
EOF
}

while (($#)); do
  case "$1" in
    --build-id) [[ $# -ge 2 ]] || die "--build-id requires a value"; BUILD_ID="$2"; BUILD_ID_EXPLICIT=1; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --artifact-root) [[ $# -ge 2 ]] || die "--artifact-root requires a value"; ARTIFACT_ROOT="$2"; shift 2 ;;
    --episode-manifest) [[ $# -ge 2 ]] || die "--episode-manifest requires a value"; EPISODE_MANIFEST="$2"; shift 2 ;;
    --extractor-python) [[ $# -ge 2 ]] || die "--extractor-python requires a value"; EXTRACTOR_PYTHON="$2"; shift 2 ;;
    --vjepa-source) [[ $# -ge 2 ]] || die "--vjepa-source requires a value"; VJEPA_SOURCE="$2"; shift 2 ;;
    --vjepa-checkpoint) [[ $# -ge 2 ]] || die "--vjepa-checkpoint requires a value"; VJEPA_CHECKPOINT="$2"; shift 2 ;;
    --vjepa-checkpoint-sha256) [[ $# -ge 2 ]] || die "--vjepa-checkpoint-sha256 requires a value"; VJEPA_CHECKPOINT_SHA256="$2"; shift 2 ;;
    --allow-active-job-id)
      [[ $# -ge 2 ]] || die "--allow-active-job-id requires a value"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "active job ID must be a positive integer"
      for existing in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
        [[ "$existing" != "$2" ]] || die "active job ID repeated: $2"
      done
      ALLOW_ACTIVE_JOB_IDS+=("$2")
      shift 2
      ;;
    --partition) [[ $# -ge 2 ]] || die "--partition requires a value"; PARTITION="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || die "--time requires a value"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) [[ $# -ge 2 ]] || die "--cpus requires a value"; CPUS="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "--mem requires a value"; MEMORY="$2"; shift 2 ;;
    --account) [[ $# -ge 2 ]] || die "--account requires a value"; ACCOUNT="$2"; shift 2 ;;
    --qos) [[ $# -ge 2 ]] || die "--qos requires a value"; QOS="$2"; shift 2 ;;
    --resume-existing) RESUME_EXISTING=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if ((${#ALLOW_ACTIVE_JOB_IDS[@]})); then
  mapfile -t ALLOW_ACTIVE_JOB_IDS < <(
    printf '%s\n' "${ALLOW_ACTIVE_JOB_IDS[@]}" | LC_ALL=C sort -n
  )
fi

[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$TIME_LIMIT" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]] || \
  die "--time must use HH:MM:SS"
TIME_SECONDS="$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))"
((TIME_SECONDS > 0 && TIME_SECONDS <= 4 * 3600)) || \
  die "--time must be in (00:00:00, 04:00:00]"
for value in "$PARTITION" "$MEMORY" "$ACCOUNT" "$QOS"; do
  [[ "$value" != *[[:space:]]* ]] || die "Slurm scalar options cannot contain whitespace"
done
[[ "$VJEPA_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "checkpoint SHA-256 must be 64 lowercase hexadecimal characters"
((RESUME_EXISTING == 0 || BUILD_ID_EXPLICIT == 1)) || \
  die "--resume-existing requires an explicit --build-id"

for path in \
  "$SBATCH_SCRIPT" "$EXTRACTOR" "$MANIFEST_BUILDER" \
  "$EPISODE_MANIFEST" "$VJEPA_CHECKPOINT"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required regular file is unavailable or symlinked: $path"
done
[[ -x "$SBATCH_SCRIPT" && -x "$EXTRACTOR_PYTHON" ]] || \
  die "Slurm entrypoint/extractor Python must be executable"
[[ -d "$VJEPA_SOURCE/.git" && ! -L "$VJEPA_SOURCE" ]] || \
  die "V-JEPA source must be a non-symlink Git checkout"

REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
EPISODE_MANIFEST="$(readlink -f "$EPISODE_MANIFEST")"
EXTRACTOR_PYTHON="$(readlink -f "$EXTRACTOR_PYTHON")"
VJEPA_SOURCE="$(cd "$VJEPA_SOURCE" && pwd -P)"
VJEPA_CHECKPOINT="$(readlink -f "$VJEPA_CHECKPOINT")"
[[ -f "$EXTRACTOR_PYTHON" && -x "$EXTRACTOR_PYTHON" && ! -L "$EXTRACTOR_PYTHON" ]] || \
  die "resolved extractor Python is not a regular executable"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "could not resolve repository commit"
if [[ -n "$EXPECTED_COMMIT" ]]; then
  [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit must be full lowercase SHA-1"
  [[ "$EXPECTED_COMMIT" == "$ACTUAL_COMMIT" ]] || \
    die "repository commit differs: $ACTUAL_COMMIT != $EXPECTED_COMMIT"
else
  EXPECTED_COMMIT="$ACTUAL_COMMIT"
fi
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository must be clean: ${GIT_STATUS//$'\n'/; }"
[[ "$(git -C "$VJEPA_SOURCE" rev-parse HEAD)" == "45d025f636dfc58fc2426905fc4a1ab755b1c3e5" ]] || \
  die "V-JEPA source is not at the pinned release commit"
SOURCE_STATUS="$(git -C "$VJEPA_SOURCE" status --porcelain --untracked-files=all)"
[[ -z "$SOURCE_STATUS" ]] || die "V-JEPA source must be clean: ${SOURCE_STATUS//$'\n'/; }"
[[ "$(stat -c %s "$VJEPA_CHECKPOINT")" == "1664223428" ]] || \
  die "V-JEPA checkpoint has the wrong byte count"
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$VJEPA_CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_CHECKPOINT_SHA256" == "$VJEPA_CHECKPOINT_SHA256" ]] || \
  die "checkpoint SHA-256 mismatch: $ACTUAL_CHECKPOINT_SHA256"
EPISODE_MANIFEST_SHA256="$(sha256sum "$EPISODE_MANIFEST" | awk '{print $1}')"

PYTHON_VERSION="$("$EXTRACTOR_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION#*.}"
((PYTHON_MAJOR > 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR >= 11))) || \
  die "extractor Python must be >=3.11, found $PYTHON_VERSION"

"$EXTRACTOR_PYTHON" "$EXTRACTOR" validate-inputs \
  --source-path "$VJEPA_SOURCE" \
  --checkpoint "$VJEPA_CHECKPOINT" \
  --checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"

if [[ -z "$BUILD_ID" ]]; then
  BUILD_ID="vjepa2-cache-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_COMMIT:0:10}"
fi
[[ "$BUILD_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$ ]] || die "unsafe build ID"
[[ "$ARTIFACT_ROOT" == /* ]] || die "artifact root must be absolute"
case "$ARTIFACT_ROOT/" in
  "$REPO_ROOT/"*) die "artifact root must be external to the repository" ;;
esac
case "$ARTIFACT_ROOT/" in
  "$LACWM_BASE/"*) ;;
  *) die "artifact root must remain under the approved LACWM base: $LACWM_BASE" ;;
esac
BUILD_ROOT="$ARTIFACT_ROOT/$BUILD_ID"
REQUEST="$BUILD_ROOT/build_request.json"
COMPLETE="$BUILD_ROOT/complete.json"

REQUEST_ARGS=(
  "$REQUEST" "$BUILD_ID" "$EXPECTED_COMMIT" "$REPO_ROOT" "$BUILD_ROOT"
  "$EPISODE_MANIFEST" "$EPISODE_MANIFEST_SHA256" "$EXTRACTOR_PYTHON"
  "$VJEPA_SOURCE" "$VJEPA_CHECKPOINT" "$VJEPA_CHECKPOINT_SHA256"
)
validate_or_write_request() {
  local mode="$1"
  "$EXTRACTOR_PYTHON" - "$mode" "${REQUEST_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    mode,
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
payload = {
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
path = Path(request_path)
if mode == "write":
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
elif mode == "validate":
    with path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    if actual != payload:
        raise SystemExit("existing immutable build request differs")
else:
    raise SystemExit(f"unknown request mode: {mode}")
PY
}

if ((RESUME_EXISTING)); then
  [[ -d "$BUILD_ROOT" && ! -L "$BUILD_ROOT" ]] || \
    die "resume build root is unavailable or symlinked: $BUILD_ROOT"
  [[ -f "$REQUEST" && ! -L "$REQUEST" ]] || die "resume build request is missing"
  [[ ! -e "$COMPLETE" ]] || die "completed cache builds cannot be resubmitted"
  validate_or_write_request validate
else
  [[ ! -e "$BUILD_ROOT" ]] || die "fresh build root already exists: $BUILD_ROOT"
fi

SBATCH_ARGS=(
  --parsable
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --partition="$PARTITION"
  --exclusive
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="vjepa2-cache-${BUILD_ID:0:70}"
  --output="$ARTIFACT_ROOT/_slurm/logs/%x-%j.out"
  --error="$ARTIFACT_ROOT/_slurm/logs/%x-%j.err"
)
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos="$QOS")
JOB_ARGS=(
  --build-id "$BUILD_ID"
  --expected-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --build-root "$BUILD_ROOT"
  --episode-manifest "$EPISODE_MANIFEST"
  --episode-manifest-sha256 "$EPISODE_MANIFEST_SHA256"
  --extractor-python "$EXTRACTOR_PYTHON"
  --vjepa-source "$VJEPA_SOURCE"
  --vjepa-checkpoint "$VJEPA_CHECKPOINT"
  --vjepa-checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"
)
COMMAND=(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" "${JOB_ARGS[@]}")

printf 'Validated immutable-cache sbatch command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Git commit: $EXPECTED_COMMIT (clean)"
echo "Build root: $BUILD_ROOT"
echo "ABC manifest: $EPISODE_MANIFEST ($EPISODE_MANIFEST_SHA256)"
echo "V-JEPA source: $VJEPA_SOURCE @ 45d025f636dfc58fc2426905fc4a1ab755b1c3e5 (clean)"
echo "V-JEPA checkpoint: $VJEPA_CHECKPOINT ($VJEPA_CHECKPOINT_SHA256)"
echo "Contract: train/val/test=512/64/128, clips-per-episode=1, seed=20260729"
echo "PCA contract: first 256 train clips, 250000 tokens, PCA64 whitening"
echo "Estimated cache bytes: 10.53 GB (9.81 GiB), excluding PCA/log overhead"
echo "Extraction: train/val/test use GPUs 0/1/2 concurrently; no --overwrite"
if ((RESUME_EXISTING)); then
  echo "Mode: resume matching incomplete immutable root"
else
  echo "Mode: fresh immutable root"
fi

if ((EXECUTE == 0)); then
  echo "Dry run only. No files or jobs were created."
  echo "Re-run with --build-id '$BUILD_ID' --expected-commit '$EXPECTED_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"

is_allowed_active_job_id() {
  local candidate="$1"
  local allowed
  for allowed in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
    [[ "$candidate" != "$allowed" ]] || return 0
  done
  return 1
}

check_active_user_jobs() {
  local records
  records="$(
    squeue \
      --noheader \
      --user "${USER:?USER is unset}" \
      --states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED \
      --format='%A|%i|%T|%j'
  )" || die "could not enumerate active user jobs"
  local rejected=()
  local observed=()
  local base_id displayed state name
  while IFS='|' read -r base_id displayed state name; do
    [[ -n "$base_id" ]] || continue
    [[ "$base_id" =~ ^[1-9][0-9]*$ ]] || die "squeue returned an invalid job ID"
    observed+=("$base_id")
    if ! is_allowed_active_job_id "$base_id"; then
      rejected+=("$base_id|$displayed|$state|$name")
    fi
  done <<< "$records"
  ((${#rejected[@]} == 0)) || \
    die "refusing submission with non-allow-listed active jobs: ${rejected[*]}"
  local unique_observed=()
  if ((${#observed[@]})); then
    mapfile -t unique_observed < <(printf '%s\n' "${observed[@]}" | sort -nu)
  fi
  [[ "${unique_observed[*]-}" == "${ALLOW_ACTIVE_JOB_IDS[*]-}" ]] || \
    die "active job IDs must exactly equal the explicit allow-list"
}

check_active_user_jobs
mkdir -p "$ARTIFACT_ROOT"
[[ -d "$ARTIFACT_ROOT" && ! -L "$ARTIFACT_ROOT" ]] || die "artifact root is invalid"
mkdir -p "$ARTIFACT_ROOT/_slurm/logs"
if ((RESUME_EXISTING == 0)); then
  mkdir "$BUILD_ROOT"
  validate_or_write_request write
else
  validate_or_write_request validate
fi
check_active_user_jobs

JOB_ID="$("${COMMAND[@]}")" || die "Slurm rejected the cache-build submission"
[[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
  die "sbatch returned an unexpected job identifier: $JOB_ID"

"$EXTRACTOR_PYTHON" - "$BUILD_ROOT/submission_${JOB_ID}.json" "$JOB_ID" "$REQUEST" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

output, job_id, request = sys.argv[1:]
digest = hashlib.sha256(Path(request).read_bytes()).hexdigest()
payload = {
    "artifact_type": "vjepa2.1-cache-build-submission",
    "format_version": 1,
    "slurm_job_id": job_id,
    "build_request_sha256": digest,
    "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Submitted immutable V-JEPA cache build: $JOB_ID"
echo "Logs: $ARTIFACT_ROOT/_slurm/logs/vjepa2-cache-${BUILD_ID:0:70}-${JOB_ID}.out"
