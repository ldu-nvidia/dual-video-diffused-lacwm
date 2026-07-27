#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1000G
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: evaluate_causal_screen_snapshots.sbatch REQUIRED_OPTIONS

Internal, non-requeueable eight-rank post-training evaluator. Submit only via
tools/slurm/submit_causal_screen_evaluation.sh.
EOF
}

EXPECTED_EVALUATION_COMMIT=""
REPO_ROOT=""
EVALUATION_ROOT=""
EVALUATION_MANIFEST=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
DATA_ROOT=""

while (($#)); do
  case "$1" in
    --expected-evaluation-commit) [[ $# -ge 2 ]] || die "--expected-evaluation-commit requires a value"; EXPECTED_EVALUATION_COMMIT="$2"; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "--repo-root requires a value"; REPO_ROOT="$2"; shift 2 ;;
    --evaluation-root) [[ $# -ge 2 ]] || die "--evaluation-root requires a value"; EVALUATION_ROOT="$2"; shift 2 ;;
    --evaluation-manifest) [[ $# -ge 2 ]] || die "--evaluation-manifest requires a value"; EVALUATION_MANIFEST="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_DIR_VALUE="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_HOME_VALUE="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; DATA_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for required_pair in \
  "evaluation commit:$EXPECTED_EVALUATION_COMMIT" \
  "repository root:$REPO_ROOT" \
  "evaluation root:$EVALUATION_ROOT" \
  "evaluation manifest:$EVALUATION_MANIFEST" \
  "Python:$PYTHON_BIN" \
  "Wan directory:$WAN_DIR_VALUE" \
  "VideoX-Fun checkout:$VIDEOX_HOME_VALUE" \
  "data root:$DATA_ROOT"; do
  [[ -n "${required_pair#*:}" ]] || die "${required_pair%%:*} is required"
done

[[ "$EXPECTED_EVALUATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "evaluation commit must be a full lowercase hexadecimal commit"
for absolute_path in \
  "$REPO_ROOT" "$EVALUATION_ROOT" "$EVALUATION_MANIFEST" \
  "$PYTHON_BIN" "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE" "$DATA_ROOT"; do
  [[ "$absolute_path" == /* ]] || die "all paths must be absolute: $absolute_path"
done
[[ -d "$REPO_ROOT/.git" && ! -L "$REPO_ROOT" ]] || \
  die "repository root is unavailable or symlinked: $REPO_ROOT"
CANONICAL_REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
[[ "$CANONICAL_REPO_ROOT" == "$REPO_ROOT" ]] || \
  die "repository root canonicalization changed: $CANONICAL_REPO_ROOT"
[[ -d "$EVALUATION_ROOT" && ! -L "$EVALUATION_ROOT" ]] || \
  die "evaluation root is unavailable or symlinked: $EVALUATION_ROOT"
[[ -f "$EVALUATION_MANIFEST" && ! -L "$EVALUATION_MANIFEST" ]] || \
  die "evaluation manifest is unavailable or symlinked"
[[ "$(dirname "$EVALUATION_MANIFEST")" == "$EVALUATION_ROOT" ]] || \
  die "evaluation manifest must be directly under evaluation root"
[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -d "$WAN_DIR_VALUE" && ! -L "$WAN_DIR_VALUE" ]] || \
  die "Wan assets are unavailable or symlinked"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || die "VideoX-Fun checkout is unavailable"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || \
  die "data root is unavailable or symlinked"

HELPER="$REPO_ROOT/tools/evaluate_causal_screen_snapshots.py"
ACTIVATE="$REPO_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$REPO_ROOT/tools/env/verify_b200_runtime.py"
for required_file in "$HELPER" "$ACTIVATE" "$VERIFY_RUNTIME"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] || \
    die "required repository file is missing or symlinked: $required_file"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this entrypoint requires a Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "posthoc evaluation is non-requeueable; refusing restart ${SLURM_RESTART_COUNT}"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "posthoc evaluation requires exactly one node"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "posthoc evaluation must not run as a Slurm array"
if command -v scontrol >/dev/null 2>&1; then
  JOB_RECORD="$(scontrol show job "$SLURM_JOB_ID" -o)"
  [[ "$JOB_RECORD" == *"Requeue=0"* ]] || \
    die "Slurm job is unexpectedly requeueable"
fi

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_EVALUATION_COMMIT" ]] || \
  die "repository commit changed: $ACTUAL_COMMIT != $EXPECTED_EVALUATION_COMMIT"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || \
  die "evaluation repository must be clean: ${GIT_STATUS//$'\n'/; }"

# DataLoader and multiprocessing sockets must remain short and node-local.
IPC_TMPDIR="/tmp/lacwm-eval-${UID}-${SLURM_JOB_ID}"
[[ ${#IPC_TMPDIR} -le 64 ]] || die "IPC TMPDIR is unexpectedly long"
[[ ! -e "$IPC_TMPDIR" ]] || die "fresh IPC TMPDIR already exists: $IPC_TMPDIR"
(umask 077 && mkdir "$IPC_TMPDIR") || die "could not create IPC TMPDIR"
[[ ! -L "$IPC_TMPDIR" ]] || die "IPC TMPDIR must not be a symlink"
[[ "$(stat -c %u "$IPC_TMPDIR")" == "$UID" ]] || \
  die "IPC TMPDIR has the wrong owner"
[[ "$(stat -c %a "$IPC_TMPDIR")" == "700" ]] || \
  die "IPC TMPDIR must have mode 700"
cleanup_ipc_tmpdir() {
  rmdir "$IPC_TMPDIR" 2>/dev/null || true
}
trap cleanup_ipc_tmpdir EXIT

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_DATA="$DATA_ROOT"
export LACWM_PYTHON="$PYTHON_BIN"
export LACWM_WORLD_SIZE=8
export LACWM_NNODES=1
export LACWM_GPUS_PER_NODE=8
export WANDB_MODE=disabled
export WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RUN_GROUP WANDB_NAME WANDB_RESUME WANDB_TAGS
export TMPDIR="$IPC_TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Establish the pinned B200/NCCL environment. No training run environment or
# W&B directory is created by this evaluation-only job.
source "$ACTIVATE"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_EVALUATION_COMMIT" ]] || \
  die "repository commit changed after environment activation"
GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)"
[[ -z "$GIT_STATUS" ]] || die "repository became dirty"

RUNTIME_VERIFICATION="$EVALUATION_ROOT/runtime_verification.json"
[[ ! -e "$RUNTIME_VERIFICATION" ]] || \
  die "evaluation cannot resume: runtime verification already exists"
(
  set -o noclobber
  "$PYTHON_BIN" "$VERIFY_RUNTIME" \
    --expected-gpus 8 \
    --require-b200 \
    --wan-dir "$WAN_DIR_VALUE" > "$RUNTIME_VERIFICATION"
)

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  --nproc_per_node=8
  "$HELPER"
  run
  --expected-evaluation-commit "$EXPECTED_EVALUATION_COMMIT"
  --repo-root "$REPO_ROOT"
  --evaluation-root "$EVALUATION_ROOT"
  --evaluation-manifest "$EVALUATION_MANIFEST"
)

printf 'Launching immutable post-training evaluation:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Evaluation root: $EVALUATION_ROOT"
echo "Evaluation commit: $EXPECTED_EVALUATION_COMMIT (clean)"
echo "World size: 8; W&B disabled; resume=false; requeue=false"
echo "Clock: sigma=1 noise, sigma=0 clean"

(
  cd "$REPO_ROOT"
  "${COMMAND[@]}"
)
