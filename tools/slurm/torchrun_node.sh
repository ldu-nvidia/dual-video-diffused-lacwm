#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

PYTHON_BIN=""
NNODES=""
NPROC_PER_NODE="8"
MASTER_ADDR=""
MASTER_PORT=""
RDZV_ID=""
DRY_RUN=0

while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --nnodes) [[ $# -ge 2 ]] || die "--nnodes requires a value"; NNODES="$2"; shift 2 ;;
    --nproc-per-node) [[ $# -ge 2 ]] || die "--nproc-per-node requires a value"; NPROC_PER_NODE="$2"; shift 2 ;;
    --master-addr) [[ $# -ge 2 ]] || die "--master-addr requires a value"; MASTER_ADDR="$2"; shift 2 ;;
    --master-port) [[ $# -ge 2 ]] || die "--master-port requires a value"; MASTER_PORT="$2"; shift 2 ;;
    --rdzv-id) [[ $# -ge 2 ]] || die "--rdzv-id requires a value"; RDZV_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] && ((10#$NNODES <= 32)) || \
  die "--nnodes must be between 1 and 32"
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "--nproc-per-node must be positive"
[[ -n "$MASTER_ADDR" && -n "$RDZV_ID" ]] || die "--master-addr and --rdzv-id are required"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "--master-port must be numeric"
((10#$MASTER_PORT >= 1024 && 10#$MASTER_PORT <= 65535)) || \
  die "--master-port must be between 1024 and 65535"
[[ $# -gt 0 ]] || die "a Python entrypoint is required after --"

NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-}}"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || \
  die "SLURM_NODEID or SLURM_PROCID must provide a numeric node rank"
((10#$NODE_RANK < 10#$NNODES)) || \
  die "node rank $NODE_RANK is outside nnodes=$NNODES"
if [[ -n "${SLURM_NODEID:-}" && -n "${SLURM_PROCID:-}" ]]; then
  [[ "$SLURM_NODEID" == "$SLURM_PROCID" ]] || \
    die "one-task-per-node requires SLURM_NODEID == SLURM_PROCID"
fi
[[ "${SLURM_LOCALID:-0}" == "0" ]] || \
  die "one-task-per-node requires SLURM_LOCALID=0"

COMMAND=(
  "$PYTHON_BIN" -m torch.distributed.run
  "--nnodes=$NNODES"
  "--nproc_per_node=$NPROC_PER_NODE"
  "--node_rank=$NODE_RANK"
  --rdzv_backend=c10d
  "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
  "--rdzv_id=$RDZV_ID"
  "$@"
)

printf 'node_rank=%s host=%s torchrun:' "$NODE_RANK" "$(hostname)"
printf ' %q' "${COMMAND[@]}"
printf '\n'
if ((DRY_RUN == 1)); then
  exit 0
fi

# The coordinator's TMPDIR may be on node-local storage and therefore absent on
# every other host. Give each torchrun agent a short private local path for
# DataLoader multiprocessing sockets, and isolate compiler caches per node.
LOCAL_TMP_ROOT="${SLURM_TMPDIR:-/tmp}"
[[ -d "$LOCAL_TMP_ROOT" && -w "$LOCAL_TMP_ROOT" ]] || LOCAL_TMP_ROOT="/tmp"
LOCAL_TMPDIR="$(mktemp -d "$LOCAL_TMP_ROOT/lacwm-${UID}-node${NODE_RANK}.XXXXXX")"
chmod 700 "$LOCAL_TMPDIR"
export TMPDIR="$LOCAL_TMPDIR"
export XDG_RUNTIME_DIR="$LOCAL_TMPDIR"
if [[ -z "${CUDA_CACHE_PATH:-}" ]]; then
  CUDA_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/lacwm-${UID}-cuda-cache-${SLURM_JOB_ID:-manual}"
fi
export CUDA_CACHE_PATH
mkdir -p "$CUDA_CACHE_PATH"
chmod 700 "$CUDA_CACHE_PATH"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
for cache_variable in TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR; do
  if [[ -n "${!cache_variable:-}" ]]; then
    printf -v "$cache_variable" '%s/node_%s' "${!cache_variable}" "$NODE_RANK"
    export "$cache_variable"
    mkdir -p "${!cache_variable}"
  fi
done
cleanup() {
  rmdir -- "$LOCAL_TMPDIR" 2>/dev/null || true
}
trap cleanup EXIT

set +e
"${COMMAND[@]}"
STATUS=$?
set -e
exit "$STATUS"
