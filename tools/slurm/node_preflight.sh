#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 15 ]] || {
  echo "Usage: node_preflight.sh PYTHON TOOLS GPUS WAN VIDEOX DATA RUN_ROOT DATA_REPORT MAX_AGE MIN_MEMORY OUTPUT_ROOT DATASETS_CSV POLICY FAST_AUTH MIXED_REPORT" >&2
  exit 2
}

PYTHON_BIN="$1"
TOOLS_DIR="$2"
GPUS="$3"
WAN_DIR_VALUE="$4"
VIDEOX_HOME_VALUE="$5"
DATA_ROOT="$6"
RUN_ROOT="$7"
DATA_REPORT="$8"
MAX_REPORT_AGE="$9"
MIN_MEMORY="${10}"
OUTPUT_ROOT="${11}"
DATASETS_CSV="${12}"
DATA_VALIDATION_POLICY="${13}"
FAST_TRAINING_AUTHORIZATION="${14}"
MIXED_LOADER_REPORT="${15}"
IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
NODE_DIR="$OUTPUT_ROOT/node_${NODE_RANK}_$(hostname)"
mkdir -p "$NODE_DIR"

hostname > "$NODE_DIR/hostname.txt"
nvidia-smi topo -m > "$NODE_DIR/nvidia_topology.txt"
if command -v ip >/dev/null 2>&1; then
  ip -brief link > "$NODE_DIR/network_interfaces.txt"
fi

"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE" > "$NODE_DIR/runtime_verification.json"

PREFLIGHT=(
  "$PYTHON_BIN" "$TOOLS_DIR/training_preflight.py"
  --profile full \
  --gpus "$GPUS" \
  --python "$PYTHON_BIN" \
  --wan-dir "$WAN_DIR_VALUE" \
  --videox-home "$VIDEOX_HOME_VALUE" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --data-validation-report "$DATA_REPORT" \
  --data-validation-policy "$DATA_VALIDATION_POLICY" \
  --max-data-report-age-hours "$MAX_REPORT_AGE" \
  --min-gpu-memory-mib "$MIN_MEMORY" \
  --datasets "${DATASETS[@]}"
  --json-out "$NODE_DIR/preflight.json"
)
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  PREFLIGHT+=(
    --fast-training-authorization "$FAST_TRAINING_AUTHORIZATION"
    --mixed-loader-report "$MIXED_LOADER_REPORT"
  )
elif [[ "$DATA_VALIDATION_POLICY" != "strict" ]]; then
  echo "unsupported data validation policy: $DATA_VALIDATION_POLICY" >&2
  exit 2
fi
"${PREFLIGHT[@]}"

printf 'node_rank=%s hostname=%s status=passed output=%s\n' \
  "$NODE_RANK" "$(hostname)" "$NODE_DIR"
