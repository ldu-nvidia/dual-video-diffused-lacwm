#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"
HELPER="$REPO_ROOT/tools/vjepa2_controlled_study.py"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/vjepa2_controlled_study.sbatch"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"

LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
PYTHON_BIN="${LACWM_PYTHON:-$LACWM_BASE/envs/lacwm-b200-py310/bin/python}"
EXTRACTOR_PYTHON="${VJEPA_EXTRACTOR_PYTHON:-}"
WAN_DIR_VALUE="${WAN_DIR:-$LACWM_BASE/wan_fun_1.3b_control}"
VIDEOX_HOME_VALUE="${VIDEOX_HOME:-$LACWM_BASE/VideoX-Fun-1d6d9c3}"
RUN_ROOT="${VJEPA_STUDY_RUN_ROOT:-$LACWM_BASE/runs/dual_video_diffusion/vjepa2_controlled_study}"

BASELINE_CONFIG="${VJEPA_BASELINE_CONFIG:-$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/vjepa_v0.yaml}"
BASELINE_SELECTOR="${VJEPA_BASELINE_SELECTOR:-ravenhuang/wan-dit/vjepa_v0.yaml}"
DUAL_CONFIG="${VJEPA_DUAL_CONFIG:-$PROJECT_ROOT/configs/experiments_0908/ravenhuang/wan-dit/vjepa_vpm.yaml}"
DUAL_SELECTOR="${VJEPA_DUAL_SELECTOR:-ravenhuang/wan-dit/vjepa_vpm.yaml}"

WARMSTART="${LACWM_WARM_START:-$LACWM_BASE/runs/production/lora8n-stage2-fastall4-v1-f227b3b-posttrain-mig1/snapshot.pt}"
WARMSTART_SHA256="${LACWM_WARM_START_SHA256:-5c132cb5ed6df7b840eef075d13140a73b1c6d5b3d1be4299b01e8365866224b}"
VJEPA_SOURCE="${VJEPA_SOURCE:-}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-}"
VJEPA_CHECKPOINT_SHA256="${VJEPA_CHECKPOINT_SHA256:-}"
PCA_STATS="${VJEPA_PCA_STATS:-}"
PCA_STATS_SHA256="${VJEPA_PCA_STATS_SHA256:-}"
TRAIN_MANIFEST="${VJEPA_TRAIN_CLIP_MANIFEST:-}"
TRAIN_CACHE_METADATA="${VJEPA_TRAIN_CACHE_METADATA:-}"
VALIDATION_MANIFEST="${VJEPA_VAL_CLIP_MANIFEST:-}"
VALIDATION_CACHE_METADATA="${VJEPA_VAL_CACHE_METADATA:-}"
TEST_MANIFEST="${VJEPA_TEST_CLIP_MANIFEST:-}"
TEST_CACHE_METADATA="${VJEPA_TEST_CACHE_METADATA:-}"

WANDB_ENTITY_VALUE="zijiandu"
WANDB_PROJECT_VALUE="dual-video-diffusion-private"
PARTITION="batch"
TIME_LIMIT="12:00:00"
CPUS="160"
MEMORY="1000G"
ACCOUNT=""
QOS=""
STUDY_ID=""
EXPECTED_COMMIT=""
MAX_CONCURRENT_ARMS=2
ALLOW_ACTIVE_JOB_IDS=()
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_vjepa2_controlled_study.sh [options]

Validate or submit the immutable five-arm V-JEPA 2.1 controlled study:

  0 V0   original ExplicitActionDiTModel
  1 VPM  parameter-matched dual wrapper, video objective only
  2 A1   V-JEPA auxiliary objective, no video-trunk fusion
  3 J0   joint diffusion, aligned clocks
  4 J1   joint diffusion, V-JEPA clock leads by logit 1

Primary completed-update milestones are exactly
[1,50,100,200,400,800,1000]. A 600-update allocation-only stage ensures no
Slurm allocation performs more than 200 new updates. The eight stage arrays
form an afterok chain; a failed arm stops all later stages. Inference NFE is
[1,2,4,6,8,12,20]. V-JEPA teacher calls during training/inference are zero.

Required path/provenance options can instead be supplied through the matching
environment variables shown in parentheses:

  --extractor-python PATH       Python for cache validation
                                (VJEPA_EXTRACTOR_PYTHON)
  --vjepa-source PATH           Pinned official source checkout (VJEPA_SOURCE)
  --vjepa-checkpoint PATH       Official V-JEPA checkpoint (VJEPA_CHECKPOINT)
  --vjepa-checkpoint-sha256 HEX (VJEPA_CHECKPOINT_SHA256)
  --pca-stats PATH              Train-split PCA64 artifact (VJEPA_PCA_STATS)
  --pca-stats-sha256 HEX        (VJEPA_PCA_STATS_SHA256)
  --train-manifest PATH         (VJEPA_TRAIN_CLIP_MANIFEST)
  --train-cache-metadata PATH   (VJEPA_TRAIN_CACHE_METADATA)
  --validation-manifest PATH    (VJEPA_VAL_CLIP_MANIFEST)
  --validation-cache-metadata PATH
                                (VJEPA_VAL_CACHE_METADATA)
  --test-manifest PATH          (VJEPA_TEST_CLIP_MANIFEST)
  --test-cache-metadata PATH    (VJEPA_TEST_CACHE_METADATA)

Other options:
  --study-id ID                 Fresh immutable study ID
  --expected-commit SHA         Require this exact clean 40-char commit
  --baseline-config PATH        V0 Hydra experiment file
  --baseline-selector VALUE     V0 experiments_0908 selector
  --dual-config PATH            Shared dual Hydra experiment file
  --dual-selector VALUE         Shared dual experiments_0908 selector
  --warmstart PATH              Production model-only warm start
  --warmstart-sha256 HEX        Warm-start SHA-256
  --run-root PATH               Approved external artifact root
  --max-concurrent-arms N       Array concurrency in [1,5] (default: 2)
  --allow-active-job-id ID      Allow one pre-existing numeric Slurm job ID;
                                repeat for each read-only unrelated job
  --partition NAME              Default: batch
  --time HH:MM:SS               Default: 12:00:00; max: 24:00:00
  --cpus N                      Default: 160
  --mem VALUE                   Default: 1000G
  --account NAME                Optional Slurm account
  --qos NAME                    Optional Slurm QOS
  --execute                     Create provenance and submit. Without this,
                                perform a complete read-only preflight only.
  -h, --help

The launcher never stops, requeues, or mutates an existing job. W&B is locked
to zijiandu/dual-video-diffusion-private with group=null.
LACWM uses sigma=1 for noise and sigma=0 for clean data.
EOF
}

while (($#)); do
  case "$1" in
    --study-id) [[ $# -ge 2 ]] || die "--study-id requires a value"; STUDY_ID="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --extractor-python) [[ $# -ge 2 ]] || die "--extractor-python requires a value"; EXTRACTOR_PYTHON="$2"; shift 2 ;;
    --vjepa-source) [[ $# -ge 2 ]] || die "--vjepa-source requires a value"; VJEPA_SOURCE="$2"; shift 2 ;;
    --vjepa-checkpoint) [[ $# -ge 2 ]] || die "--vjepa-checkpoint requires a value"; VJEPA_CHECKPOINT="$2"; shift 2 ;;
    --vjepa-checkpoint-sha256) [[ $# -ge 2 ]] || die "--vjepa-checkpoint-sha256 requires a value"; VJEPA_CHECKPOINT_SHA256="$2"; shift 2 ;;
    --pca-stats) [[ $# -ge 2 ]] || die "--pca-stats requires a value"; PCA_STATS="$2"; shift 2 ;;
    --pca-stats-sha256) [[ $# -ge 2 ]] || die "--pca-stats-sha256 requires a value"; PCA_STATS_SHA256="$2"; shift 2 ;;
    --train-manifest) [[ $# -ge 2 ]] || die "--train-manifest requires a value"; TRAIN_MANIFEST="$2"; shift 2 ;;
    --train-cache-metadata) [[ $# -ge 2 ]] || die "--train-cache-metadata requires a value"; TRAIN_CACHE_METADATA="$2"; shift 2 ;;
    --validation-manifest) [[ $# -ge 2 ]] || die "--validation-manifest requires a value"; VALIDATION_MANIFEST="$2"; shift 2 ;;
    --validation-cache-metadata) [[ $# -ge 2 ]] || die "--validation-cache-metadata requires a value"; VALIDATION_CACHE_METADATA="$2"; shift 2 ;;
    --test-manifest) [[ $# -ge 2 ]] || die "--test-manifest requires a value"; TEST_MANIFEST="$2"; shift 2 ;;
    --test-cache-metadata) [[ $# -ge 2 ]] || die "--test-cache-metadata requires a value"; TEST_CACHE_METADATA="$2"; shift 2 ;;
    --baseline-config) [[ $# -ge 2 ]] || die "--baseline-config requires a value"; BASELINE_CONFIG="$2"; shift 2 ;;
    --baseline-selector) [[ $# -ge 2 ]] || die "--baseline-selector requires a value"; BASELINE_SELECTOR="$2"; shift 2 ;;
    --dual-config) [[ $# -ge 2 ]] || die "--dual-config requires a value"; DUAL_CONFIG="$2"; shift 2 ;;
    --dual-selector) [[ $# -ge 2 ]] || die "--dual-selector requires a value"; DUAL_SELECTOR="$2"; shift 2 ;;
    --warmstart) [[ $# -ge 2 ]] || die "--warmstart requires a value"; WARMSTART="$2"; shift 2 ;;
    --warmstart-sha256) [[ $# -ge 2 ]] || die "--warmstart-sha256 requires a value"; WARMSTART_SHA256="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || die "--run-root requires a value"; RUN_ROOT="$2"; shift 2 ;;
    --max-concurrent-arms) [[ $# -ge 2 ]] || die "--max-concurrent-arms requires a value"; MAX_CONCURRENT_ARMS="$2"; shift 2 ;;
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
ALLOW_ACTIVE_JOB_ARGS=()
for job_id in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
  ALLOW_ACTIVE_JOB_ARGS+=(--allow-active-job-id "$job_id")
done

[[ "$MAX_CONCURRENT_ARMS" =~ ^[1-5]$ ]] || \
  die "--max-concurrent-arms must be 1 through 5"
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$TIME_LIMIT" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]] || \
  die "--time must use HH:MM:SS"
TIME_SECONDS="$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))"
((TIME_SECONDS > 0 && TIME_SECONDS <= 24 * 3600)) || \
  die "--time must be in (00:00:00, 24:00:00]"
for value in "$PARTITION" "$MEMORY" "$ACCOUNT" "$QOS"; do
  [[ "$value" != *[[:space:]]* ]] || die "Slurm scalar options cannot contain whitespace"
done
for required_pair in \
  "extractor Python:$EXTRACTOR_PYTHON" \
  "V-JEPA source:$VJEPA_SOURCE" \
  "V-JEPA checkpoint:$VJEPA_CHECKPOINT" \
  "V-JEPA checkpoint SHA-256:$VJEPA_CHECKPOINT_SHA256" \
  "PCA statistics:$PCA_STATS" \
  "PCA statistics SHA-256:$PCA_STATS_SHA256" \
  "train manifest:$TRAIN_MANIFEST" \
  "train cache metadata:$TRAIN_CACHE_METADATA" \
  "validation manifest:$VALIDATION_MANIFEST" \
  "validation cache metadata:$VALIDATION_CACHE_METADATA" \
  "test manifest:$TEST_MANIFEST" \
  "test cache metadata:$TEST_CACHE_METADATA"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done
[[ "$VJEPA_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "V-JEPA checkpoint SHA-256 must be 64 lowercase hex characters"
[[ "$PCA_STATS_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "PCA SHA-256 must be 64 lowercase hex characters"
[[ "$WARMSTART_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "warm-start SHA-256 must be 64 lowercase hex characters"

for path in \
  "$HELPER" "$SBATCH_SCRIPT" "$ACTIVATE" "$PYTHON_BIN" "$EXTRACTOR_PYTHON" \
  "$BASELINE_CONFIG" "$DUAL_CONFIG" "$WARMSTART" "$VJEPA_CHECKPOINT" \
  "$PCA_STATS" "$TRAIN_MANIFEST" "$TRAIN_CACHE_METADATA" \
  "$VALIDATION_MANIFEST" "$VALIDATION_CACHE_METADATA" \
  "$TEST_MANIFEST" "$TEST_CACHE_METADATA"; do
  [[ -f "$path" ]] || die "required file is missing: $path"
done
[[ -x "$PYTHON_BIN" && -x "$EXTRACTOR_PYTHON" ]] || \
  die "training/extractor Python must be executable"
[[ -x "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is not executable"
[[ -d "$VJEPA_SOURCE/.git" ]] || die "V-JEPA source is not a Git checkout"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime assets are unavailable"

# The submit host may already have another LACWM checkout on PYTHONPATH. Bind
# Hydra's package search and every imported module to this exact clean commit
# before the read-only preflight composes any study configuration.
export LACWM_BASE
export LACWM_PYTHON="$PYTHON_BIN"
export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
source "$ACTIVATE"
ROBOT_WM_ORIGIN="$(
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import robot_wm; print(Path(robot_wm.__file__).resolve())'
)"
case "$ROBOT_WM_ORIGIN" in
  "$REPO_ROOT"/robot_wm/*) ;;
  *) die "Python resolved robot_wm outside the study repository: $ROBOT_WM_ORIGIN" ;;
esac

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "could not resolve Git commit"
if [[ -n "$EXPECTED_COMMIT" ]]; then
  [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
    die "--expected-commit must be a full lowercase commit"
  [[ "$EXPECTED_COMMIT" == "$ACTUAL_COMMIT" ]] || \
    die "repository commit differs: $ACTUAL_COMMIT"
else
  EXPECTED_COMMIT="$ACTUAL_COMMIT"
fi
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository must be clean: ${GIT_STATUS//$'\n'/; }"

if [[ -z "$STUDY_ID" ]]; then
  STUDY_ID="vjepa2-controlled-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_COMMIT:0:10}"
fi
[[ "$STUDY_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$ ]] || \
  die "study ID is unsafe"
[[ "$RUN_ROOT" == /* ]] || die "run root must be absolute"
STUDY_ROOT="$RUN_ROOT/$STUDY_ID"
[[ ! -e "$STUDY_ROOT" ]] || die "study root already exists: $STUDY_ROOT"

INPUT_ARGS=(
  --git-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --project-root "$PROJECT_ROOT"
  --python "$PYTHON_BIN"
  --extractor-python "$EXTRACTOR_PYTHON"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --baseline-config "$BASELINE_CONFIG"
  --baseline-selector "$BASELINE_SELECTOR"
  --dual-config "$DUAL_CONFIG"
  --dual-selector "$DUAL_SELECTOR"
  --warmstart "$WARMSTART"
  --warmstart-sha256 "$WARMSTART_SHA256"
  --vjepa-source "$VJEPA_SOURCE"
  --vjepa-checkpoint "$VJEPA_CHECKPOINT"
  --vjepa-checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"
  --pca-stats "$PCA_STATS"
  --pca-stats-sha256 "$PCA_STATS_SHA256"
  --train-manifest "$TRAIN_MANIFEST"
  --train-cache-metadata "$TRAIN_CACHE_METADATA"
  --validation-manifest "$VALIDATION_MANIFEST"
  --validation-cache-metadata "$VALIDATION_CACHE_METADATA"
  --test-manifest "$TEST_MANIFEST"
  --test-cache-metadata "$TEST_CACHE_METADATA"
)

# Read-only preflight includes the official validate-cache command for all three
# splits; each command streams and compares the complete target-array SHA-256.
"$PYTHON_BIN" "$HELPER" preflight "${INPUT_ARGS[@]}" > /tmp/vjepa2-study-preflight-"$$".json
trap 'rm -f "/tmp/vjepa2-study-preflight-$$.json"' EXIT
"$PYTHON_BIN" "$HELPER" wandb-private \
  --entity "$WANDB_ENTITY_VALUE" \
  --project "$WANDB_PROJECT_VALUE"

echo "V-JEPA controlled-study preflight passed."
echo "Git commit: $EXPECTED_COMMIT (clean)"
echo "Study root: $STUDY_ROOT (fresh)"
echo "Primary completed updates: [1,50,100,200,400,800,1000]"
echo "Allocation endpoints: [1,50,100,200,400,600,800,1000]"
echo "Inference NFE: [1,2,4,6,8,12,20]"
echo "Inference sources: autonomous, off, autonomous_shuffled; oracle matched/shuffled are leakage-only"
echo "V-JEPA teacher calls during trainer/inference: 0"
echo "W&B: $WANDB_ENTITY_VALUE/$WANDB_PROJECT_VALUE (PRIVATE), group=null"
for task_id in 0 1 2 3 4; do
  "$PYTHON_BIN" "$HELPER" arm-contract \
    --array-task-id "$task_id" \
    --format json
done

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --study-id '$STUDY_ID' --expected-commit '$EXPECTED_COMMIT' --execute."
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
    [[ "$base_id" =~ ^[1-9][0-9]*$ ]] || die "squeue returned invalid ID"
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
mkdir -p "$RUN_ROOT"
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "run root is invalid"
mkdir "$STUDY_ROOT"
LOG_DIR="$RUN_ROOT/_slurm/logs"
mkdir -p "$LOG_DIR"
STUDY_MANIFEST="$STUDY_ROOT/study_manifest.json"

"$PYTHON_BIN" "$HELPER" create-study \
  "${INPUT_ARGS[@]}" \
  --study-id "$STUDY_ID" \
  --study-root "$STUDY_ROOT" \
  --run-root "$RUN_ROOT" \
  "${ALLOW_ACTIVE_JOB_ARGS[@]}" \
  --output "$STUDY_MANIFEST"

check_active_user_jobs

SBATCH_COMMON=(
  --parsable
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --partition="$PARTITION"
  --array="0-4%$MAX_CONCURRENT_ARMS"
  --no-requeue
  --open-mode=append
  --export=ALL
)
[[ -n "$ACCOUNT" ]] && SBATCH_COMMON+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_COMMON+=(--qos="$QOS")

JOB_COMMON=(
  --study-id "$STUDY_ID"
  --expected-commit "$EXPECTED_COMMIT"
  --repo-root "$REPO_ROOT"
  --project-root "$PROJECT_ROOT"
  --study-root "$STUDY_ROOT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --baseline-config "$BASELINE_CONFIG"
  --baseline-selector "$BASELINE_SELECTOR"
  --dual-config "$DUAL_CONFIG"
  --dual-selector "$DUAL_SELECTOR"
  --warmstart "$WARMSTART"
  --train-manifest "$TRAIN_MANIFEST"
  --train-cache-metadata "$TRAIN_CACHE_METADATA"
  --validation-manifest "$VALIDATION_MANIFEST"
  --validation-cache-metadata "$VALIDATION_CACHE_METADATA"
  --test-manifest "$TEST_MANIFEST"
  --test-cache-metadata "$TEST_CACHE_METADATA"
  --wandb-entity "$WANDB_ENTITY_VALUE"
  --wandb-project "$WANDB_PROJECT_VALUE"
)

STAGE_ENDPOINTS=(1 50 100 200 400 600 800 1000)
JOB_IDS=()
PREVIOUS_JOB_ID=""
for endpoint in "${STAGE_ENDPOINTS[@]}"; do
  job_name="vjepa2-${STUDY_ID:0:54}-u${endpoint}"
  stage_sbatch=(
    "${SBATCH_COMMON[@]}"
    --job-name="$job_name"
    --output="$LOG_DIR/%x-%A_%a.out"
    --error="$LOG_DIR/%x-%A_%a.err"
  )
  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    stage_sbatch+=(--dependency="afterok:$PREVIOUS_JOB_ID")
  fi
  command=(
    sbatch
    "${stage_sbatch[@]}"
    "$SBATCH_SCRIPT"
    "${JOB_COMMON[@]}"
    --stage-endpoint "$endpoint"
  )
  JOB_ID="$("${command[@]}")" || die "Slurm rejected endpoint $endpoint"
  [[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
    die "unexpected Slurm job identifier: $JOB_ID"
  JOB_IDS+=("$JOB_ID")
  PREVIOUS_JOB_ID="${JOB_ID%%[_;]*}"
  echo "Submitted endpoint $endpoint array: $JOB_ID"
done

RECORD_ARGS=(
  "$PYTHON_BIN" "$HELPER" record-submission
  --study-manifest "$STUDY_MANIFEST"
  --max-concurrent-arms "$MAX_CONCURRENT_ARMS"
  "${ALLOW_ACTIVE_JOB_ARGS[@]}"
  --output "$STUDY_ROOT/slurm_submission.json"
)
for job_id in "${JOB_IDS[@]}"; do
  RECORD_ARGS+=(--job-id "$job_id")
done
"${RECORD_ARGS[@]}"

echo "Submitted eight afterok-chained five-arm arrays."
echo "Stage job IDs: ${JOB_IDS[*]}"
echo "Logs: $LOG_DIR/vjepa2-${STUDY_ID:0:54}-u<endpoint>-<job>_<arm>.out"
