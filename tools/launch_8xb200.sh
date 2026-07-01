#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/tools"
PROJECT_ROOT="$REPO_ROOT/projects/latent_action_models"

usage() {
  cat <<'EOF'
Usage: tools/launch_8xb200.sh [options]

Required:
  --gpus LIST              Eight physical GPU indices, e.g. 0,1,2,3,4,5,6,7.
  --variant NAME           latent or explicit.
  --python PATH            Python from the provisioned lacwm environment.
  --wan-dir PATH           Wan2.1-Fun-1.3B-Control asset directory.
  --videox-home PATH       VideoX-Fun checkout.
  --data-root PATH         Root containing all four prepared datasets.
  --run-root PATH          Output root under /mnt/data1, /mnt/data2, or a root
                           explicitly allowed by LACWM_ALLOWED_RUN_ROOTS.
  --run-name NAME          Unique W&B/Hydra run name.
  --dataset-stage NAME     Immutable dataset stage label (default: all-four).
  --datasets CSV           Exact dataset subset in canonical order
                           (default: droid,egodex,agibot,abc).
  --smoke-report PATH      Passing report from run_gradient_smoke.sh for the same
                           Git commit and model variant.
  --data-validation-report PATH
                           Passing strict or explicitly files-only JSON report.
  --data-validation-policy NAME
                           strict (default) or files_only_user_waived_v1.
  --fast-training-authorization PATH
                           Required immutable certificate for the fast policy.
  --mixed-loader-report PATH
                           Required commit/data-bound loader evidence for fast policy.
  --transition-handoff PATH
                           Optional immutable parent-stage handoff JSON. The
                           handoff and parent snapshot digests are bound into
                           the new child run identity.
  --wandb-mode MODE        online, offline, or disabled.

Required for online/offline W&B:
  --wandb-project NAME
Required for online W&B:
  --wandb-entity NAME
  --wandb-run-id ID        Stable W&B identity, independent of run directory
                           (default: --run-name).

Optional:
  --nodes N                Number of 8-GPU nodes (1-32; default: 1).
  --master-addr HOST       Rendezvous host for multi-node execution. Required
                           when --nodes is greater than one.
  --master-port PORT       Base rendezvous port (default: 29400). The NCCL,
                           DDP-smoke, and training phases use PORT..PORT+2.
  --rdzv-id ID             Attempt-unique rendezvous identifier. Required for
                           multi-node execution.
  --batch-size N           Physical per-GPU microbatch (default latent=4, explicit=2).
  --gradient-accumulation-steps N
                           Microbatches per optimizer update (default latent=4,
                           explicit=1).
  --min-gpu-memory-mib N   Minimum total memory on every B200 (default: 78000;
                           values below the planned 80 GiB profile are rejected).
  --max-iter N             Optimizer updates (default: 60000).
  --warmup-steps N         LR warmup updates (default: 2000).
  --log-every N            Metric logging cadence (default: 50).
  --save-every N           Checkpoint cadence (default: 1000).
  --val-every N            Validation cadence (default: 1000).
  --viz-every N            Visualization cadence (default: 1000).
  --max-data-report-age-hours N
                           Cached report age limit (default 24). Long Slurm runs
                           may raise this; fingerprints are still recomputed.
  --run-dir PATH           Stable Hydra/checkpoint directory beneath --run-root
                           (default: RUN_ROOT/RUN_NAME).
  --resume                 Reuse an existing wrapper-owned RUN_DIR. Trainer resumes
                           snapshot.pt when present; otherwise this retries iteration 0.
                           Without this, existing run identity/snapshot fails closed.
  --execute                Launch in the foreground. Without this, preflight and
                           command preview only.
  -h, --help               Show this help.

The wrapper enforces eight B200 GPUs per node with >=78000 MiB each by default, a clean
worktree, complete configured datasets, a prior gradient smoke, and idle allocated GPUs.
Multi-node execution requires Slurm and launches one torchrun agent per node through srun.
It never kills or pauses another job and never launches in the background.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

assert_repo_identity() {
  local actual_commit actual_status
  actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  [[ "$actual_commit" == "$CURRENT_COMMIT" ]] || die \
    "repository HEAD changed during launch: $actual_commit != $CURRENT_COMMIT"
  # User-authored reports are intentionally outside the executable training
  # surface. Preserve them without making guarded launches unusable, while all
  # tracked changes and untracked files elsewhere still fail closed.
  actual_status="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all -- . ':(exclude)reports/**')"
  [[ -z "$actual_status" ]] || die \
    "repository became dirty during launch; refusing to mix code revisions"
}

validate_smoke_report() {
  "$PYTHON_BIN" "$TOOLS_DIR/validate_smoke_report.py" \
    --report "$1" \
    --variant "$VARIANT" \
    --git-commit "$CURRENT_COMMIT" \
    --wan-dir "$WAN_PATH" \
    --videox-home "$VIDEOX_PATH" \
    --data-root "$DATA_PATH"
}

bind_transition_identity() {
  local source_identity="$1"
  local target_identity="$2"
  "$PYTHON_BIN" - "$source_identity" "$target_identity" \
    "$TRANSITION_HANDOFF" "$TRANSITION_HANDOFF_SHA256" \
    "$TRANSITION_PARENT_IDENTITY_SHA256" \
    "$TRANSITION_PARENT_SNAPSHOT_SHA256" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(source_s, target_s, handoff_path, handoff_sha256,
 parent_identity_sha256, parent_snapshot_sha256) = sys.argv[1:]
source = pathlib.Path(source_s)
target = pathlib.Path(target_s)
payload = json.loads(source.read_text(encoding="utf-8"))
state = payload.pop("state", None)
payload.pop("identity_sha256", None)
payload["transition_handoff"] = {
    "path": handoff_path,
    "sha256": handoff_sha256,
    "parent_run_identity_sha256": parent_identity_sha256,
    "parent_snapshot_sha256": parent_snapshot_sha256,
}
canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
if state is not None:
    payload["state"] = state
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
PY
}

GPUS=""
NODES="1"
MASTER_ADDR_VALUE=""
MASTER_PORT="29400"
RDZV_ID=""
VARIANT=""
PYTHON_BIN=""
WAN_PATH=""
VIDEOX_PATH=""
DATA_PATH=""
RUN_ROOT=""
RUN_NAME=""
DATASET_STAGE="all-four"
DATASETS_CSV="droid,egodex,agibot,abc"
SMOKE_REPORT=""
DATA_VALIDATION_REPORT=""
DATA_VALIDATION_POLICY="strict"
FAST_TRAINING_AUTHORIZATION=""
MIXED_LOADER_REPORT=""
TRANSITION_HANDOFF=""
WANDB_MODE_VALUE=""
WANDB_PROJECT_VALUE=""
WANDB_ENTITY_VALUE=""
WANDB_RUN_ID=""
BATCH_SIZE=""
GRADIENT_ACCUMULATION_STEPS=""
MIN_GPU_MEMORY_MIB="78000"
MAX_ITER="60000"
WARMUP_STEPS="2000"
LOG_EVERY="50"
SAVE_EVERY="1000"
VAL_EVERY="1000"
VIZ_EVERY="1000"
MAX_DATA_REPORT_AGE_HOURS="24"
REQUESTED_RUN_DIR=""
RESUME=0
EXECUTE=0

while (($#)); do
  case "$1" in
    --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; GPUS="$2"; shift 2 ;;
    --nodes) [[ $# -ge 2 ]] || die "--nodes requires a value"; NODES="$2"; shift 2 ;;
    --master-addr) [[ $# -ge 2 ]] || die "--master-addr requires a value"; MASTER_ADDR_VALUE="$2"; shift 2 ;;
    --master-port) [[ $# -ge 2 ]] || die "--master-port requires a value"; MASTER_PORT="$2"; shift 2 ;;
    --rdzv-id) [[ $# -ge 2 ]] || die "--rdzv-id requires a value"; RDZV_ID="$2"; shift 2 ;;
    --variant) [[ $# -ge 2 ]] || die "--variant requires a value"; VARIANT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --wan-dir) [[ $# -ge 2 ]] || die "--wan-dir requires a value"; WAN_PATH="$2"; shift 2 ;;
    --videox-home) [[ $# -ge 2 ]] || die "--videox-home requires a value"; VIDEOX_PATH="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; DATA_PATH="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || die "--run-root requires a value"; RUN_ROOT="$2"; shift 2 ;;
    --run-name) [[ $# -ge 2 ]] || die "--run-name requires a value"; RUN_NAME="$2"; shift 2 ;;
    --dataset-stage) [[ $# -ge 2 ]] || die "--dataset-stage requires a value"; DATASET_STAGE="$2"; shift 2 ;;
    --datasets) [[ $# -ge 2 ]] || die "--datasets requires a value"; DATASETS_CSV="$2"; shift 2 ;;
    --smoke-report) [[ $# -ge 2 ]] || die "--smoke-report requires a value"; SMOKE_REPORT="$2"; shift 2 ;;
    --data-validation-report) [[ $# -ge 2 ]] || die "--data-validation-report requires a value"; DATA_VALIDATION_REPORT="$2"; shift 2 ;;
    --data-validation-policy) [[ $# -ge 2 ]] || die "--data-validation-policy requires a value"; DATA_VALIDATION_POLICY="$2"; shift 2 ;;
    --fast-training-authorization) [[ $# -ge 2 ]] || die "--fast-training-authorization requires a value"; FAST_TRAINING_AUTHORIZATION="$2"; shift 2 ;;
    --mixed-loader-report) [[ $# -ge 2 ]] || die "--mixed-loader-report requires a value"; MIXED_LOADER_REPORT="$2"; shift 2 ;;
    --transition-handoff) [[ $# -ge 2 ]] || die "--transition-handoff requires a value"; TRANSITION_HANDOFF="$2"; shift 2 ;;
    --wandb-mode) [[ $# -ge 2 ]] || die "--wandb-mode requires a value"; WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-project) [[ $# -ge 2 ]] || die "--wandb-project requires a value"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    --wandb-entity) [[ $# -ge 2 ]] || die "--wandb-entity requires a value"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --wandb-run-id) [[ $# -ge 2 ]] || die "--wandb-run-id requires a value"; WANDB_RUN_ID="$2"; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || die "--batch-size requires a value"; BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation-steps) [[ $# -ge 2 ]] || die "--gradient-accumulation-steps requires a value"; GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --min-gpu-memory-mib) [[ $# -ge 2 ]] || die "--min-gpu-memory-mib requires a value"; MIN_GPU_MEMORY_MIB="$2"; shift 2 ;;
    --max-iter) [[ $# -ge 2 ]] || die "--max-iter requires a value"; MAX_ITER="$2"; shift 2 ;;
    --warmup-steps) [[ $# -ge 2 ]] || die "--warmup-steps requires a value"; WARMUP_STEPS="$2"; shift 2 ;;
    --log-every) [[ $# -ge 2 ]] || die "--log-every requires a value"; LOG_EVERY="$2"; shift 2 ;;
    --save-every) [[ $# -ge 2 ]] || die "--save-every requires a value"; SAVE_EVERY="$2"; shift 2 ;;
    --val-every) [[ $# -ge 2 ]] || die "--val-every requires a value"; VAL_EVERY="$2"; shift 2 ;;
    --viz-every) [[ $# -ge 2 ]] || die "--viz-every requires a value"; VIZ_EVERY="$2"; shift 2 ;;
    --max-data-report-age-hours) [[ $# -ge 2 ]] || die "--max-data-report-age-hours requires a value"; MAX_DATA_REPORT_AGE_HOURS="$2"; shift 2 ;;
    --run-dir) [[ $# -ge 2 ]] || die "--run-dir requires a value"; REQUESTED_RUN_DIR="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$VARIANT" == "latent" || "$VARIANT" == "explicit" ]] || die "--variant must be latent or explicit"
[[ -n "$GPUS" && -n "$PYTHON_BIN" && -n "$WAN_PATH" && -n "$VIDEOX_PATH" && -n "$DATA_PATH" && -n "$RUN_ROOT" ]] || die "all GPU/path options are required"
[[ -n "$RUN_NAME" && -n "$SMOKE_REPORT" && -n "$DATA_VALIDATION_REPORT" && -n "$WANDB_MODE_VALUE" ]] || die "--run-name, --smoke-report, --data-validation-report, and --wandb-mode are required"
[[ "$DATA_VALIDATION_POLICY" == "strict" || "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]] || \
  die "--data-validation-policy must be strict or files_only_user_waived_v1"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  [[ -n "$FAST_TRAINING_AUTHORIZATION" && -n "$MIXED_LOADER_REPORT" ]] || \
    die "fast policy requires --fast-training-authorization and --mixed-loader-report"
else
  [[ -z "$FAST_TRAINING_AUTHORIZATION" && -z "$MIXED_LOADER_REPORT" ]] || \
    die "strict policy forbids fast authorization/mixed evidence"
fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-name may contain only letters, digits, dot, underscore, and dash"
[[ "$DATASET_STAGE" =~ ^[A-Za-z0-9._-]+$ ]] || die "--dataset-stage contains unsafe characters"
[[ -n "$WANDB_RUN_ID" ]] || WANDB_RUN_ID="$RUN_NAME"
[[ "$WANDB_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "--wandb-run-id contains unsafe characters"
IFS=',' read -r -a REQUESTED_DATASETS <<< "$DATASETS_CSV"
declare -A DATASET_SEEN=()
for dataset_name in "${REQUESTED_DATASETS[@]}"; do
  [[ "$dataset_name" == droid || "$dataset_name" == egodex || "$dataset_name" == agibot || "$dataset_name" == abc ]] || \
    die "unsupported dataset in --datasets: $dataset_name"
  [[ -z "${DATASET_SEEN[$dataset_name]:-}" ]] || die "duplicate dataset: $dataset_name"
  DATASET_SEEN[$dataset_name]=1
done
(( ${#DATASET_SEEN[@]} > 0 )) || die "--datasets must select at least one dataset"
DATASET_ARRAY=()
for dataset_name in droid egodex agibot abc; do
  [[ -n "${DATASET_SEEN[$dataset_name]:-}" ]] && DATASET_ARRAY+=("$dataset_name")
done
DATASETS_CSV="$(IFS=,; echo "${DATASET_ARRAY[*]}")"
[[ "$WANDB_MODE_VALUE" == "online" || "$WANDB_MODE_VALUE" == "offline" || "$WANDB_MODE_VALUE" == "disabled" ]] || die "--wandb-mode must be online, offline, or disabled"
if [[ "$WANDB_MODE_VALUE" != "disabled" ]]; then
  [[ -n "$WANDB_PROJECT_VALUE" ]] || die "--wandb-project is required for W&B $WANDB_MODE_VALUE mode"
fi
if [[ "$WANDB_MODE_VALUE" == "online" ]]; then
  [[ -n "$WANDB_ENTITY_VALUE" ]] || die "--wandb-entity is required for W&B online mode"
fi
if [[ -z "$BATCH_SIZE" ]]; then
  [[ "$VARIANT" == "latent" ]] && BATCH_SIZE=4 || BATCH_SIZE=2
fi
if [[ -z "$GRADIENT_ACCUMULATION_STEPS" ]]; then
  [[ "$VARIANT" == "latent" ]] && GRADIENT_ACCUMULATION_STEPS=4 || GRADIENT_ACCUMULATION_STEPS=1
fi
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "$NODES" =~ ^[1-9][0-9]*$ ]] && ((10#$NODES <= 32)) || \
  die "--nodes must be between 1 and 32"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "--master-port must be numeric"
((10#$MASTER_PORT >= 1024 && 10#$MASTER_PORT <= 65533)) || \
  die "--master-port must be between 1024 and 65533"
if ((10#$NODES > 1)); then
  [[ -n "$MASTER_ADDR_VALUE" ]] || die "--master-addr is required when --nodes is greater than one"
  [[ -n "$RDZV_ID" ]] || die "--rdzv-id is required when --nodes is greater than one"
  [[ "$MASTER_ADDR_VALUE" != *[[:space:]]* ]] || die "--master-addr may not contain whitespace"
  [[ "$RDZV_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "--rdzv-id may contain only letters, digits, dot, underscore, and dash"
fi
[[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || \
  die "--gradient-accumulation-steps must be a positive integer"
[[ "$MIN_GPU_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]] || \
  die "--min-gpu-memory-mib must be a positive integer"
((10#$MIN_GPU_MEMORY_MIB >= 78000)) || \
  die "--min-gpu-memory-mib must be at least 78000 for the planned 80 GiB profile"
[[ "$MAX_ITER" =~ ^[1-9][0-9]*$ ]] || die "--max-iter must be a positive integer"
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || die "--warmup-steps must be a non-negative integer"
((10#$WARMUP_STEPS < 10#$MAX_ITER)) || die "--warmup-steps must be less than --max-iter"
for cadence_pair in \
  "--log-every=$LOG_EVERY" \
  "--save-every=$SAVE_EVERY" \
  "--val-every=$VAL_EVERY" \
  "--viz-every=$VIZ_EVERY"; do
  cadence_value="${cadence_pair#*=}"
  [[ "$cadence_value" =~ ^[1-9][0-9]*$ ]] || \
    die "${cadence_pair%%=*} must be a positive integer"
done
[[ "$MAX_DATA_REPORT_AGE_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  die "--max-data-report-age-hours must be a non-negative number"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
[[ ${#GPU_ARRAY[@]} -eq 8 ]] || die "--gpus must contain exactly eight indices"
declare -A GPU_SEEN=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
  [[ -z "${GPU_SEEN[$gpu]:-}" ]] || die "duplicate GPU index: $gpu"
  GPU_SEEN[$gpu]=1
done
WORLD_SIZE=$((10#$NODES * 8))
EFFECTIVE_GLOBAL_BATCH_SIZE=$((10#$BATCH_SIZE * 10#$GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE))

# Normalize every executable/data path once while still in the caller's working
# directory. The training process later changes directory into PROJECT_ROOT.
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
PYTHON_BIN="$("$PYTHON_BIN" - "$PYTHON_BIN" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().absolute())
PY
)"
mapfile -t NORMALIZED_PATHS < <("$PYTHON_BIN" - "$WAN_PATH" "$VIDEOX_PATH" "$DATA_PATH" "$RUN_ROOT" "$SMOKE_REPORT" "$DATA_VALIDATION_REPORT" "$FAST_TRAINING_AUTHORIZATION" "$MIXED_LOADER_REPORT" <<'PY'
from pathlib import Path
import sys
for value in sys.argv[1:]:
    print("" if not value else Path(value).expanduser().resolve(strict=False))
PY
)
WAN_PATH="${NORMALIZED_PATHS[0]}"
VIDEOX_PATH="${NORMALIZED_PATHS[1]}"
DATA_PATH="${NORMALIZED_PATHS[2]}"
RUN_ROOT="${NORMALIZED_PATHS[3]}"
SMOKE_REPORT="${NORMALIZED_PATHS[4]}"
DATA_VALIDATION_REPORT="${NORMALIZED_PATHS[5]}"
FAST_TRAINING_AUTHORIZATION="${NORMALIZED_PATHS[6]}"
MIXED_LOADER_REPORT="${NORMALIZED_PATHS[7]}"

TRANSITION_HANDOFF_SHA256=""
TRANSITION_PARENT_IDENTITY_SHA256=""
TRANSITION_PARENT_SNAPSHOT_SHA256=""
if [[ -n "$TRANSITION_HANDOFF" ]]; then
  [[ "$TRANSITION_HANDOFF" == /* ]] || \
    die "--transition-handoff must be an absolute path: $TRANSITION_HANDOFF"
  mapfile -t TRANSITION_FIELDS < <("$PYTHON_BIN" - "$TRANSITION_HANDOFF" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True)
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit(f"transition handoff is not a non-empty file: {path}")
encoded = path.read_bytes()
try:
    payload = json.loads(encoded)
except json.JSONDecodeError as exc:
    raise SystemExit(f"transition handoff is not valid JSON: {path}: {exc}") from exc
if payload.get("schema_version") != 1 or payload.get("status") != "complete":
    raise SystemExit("transition handoff must have schema_version=1 and status='complete'")

def require_sha256(name):
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SystemExit(f"transition handoff has invalid {name}")
    return value

parent_snapshot = pathlib.Path(str(payload.get("parent_snapshot", ""))).expanduser()
if not parent_snapshot.is_absolute():
    raise SystemExit("transition handoff parent_snapshot must be absolute")
parent_snapshot = parent_snapshot.resolve(strict=True)
if not parent_snapshot.is_file() or parent_snapshot.stat().st_size <= 0:
    raise SystemExit(f"transition parent snapshot is not a non-empty file: {parent_snapshot}")
print(path)
print(hashlib.sha256(encoded).hexdigest())
print(require_sha256("parent_run_identity_sha256"))
print(require_sha256("parent_snapshot_sha256"))
PY
  ) || die "transition handoff validation failed: $TRANSITION_HANDOFF"
  (( ${#TRANSITION_FIELDS[@]} == 4 )) || \
    die "transition handoff validation returned incomplete metadata"
  TRANSITION_HANDOFF="${TRANSITION_FIELDS[0]}"
  TRANSITION_HANDOFF_SHA256="${TRANSITION_FIELDS[1]}"
  TRANSITION_PARENT_IDENTITY_SHA256="${TRANSITION_FIELDS[2]}"
  TRANSITION_PARENT_SNAPSHOT_SHA256="${TRANSITION_FIELDS[3]}"
fi

[[ -f "$SMOKE_REPORT" ]] || die "smoke report not found: $SMOKE_REPORT"
[[ -f "$DATA_VALIDATION_REPORT" ]] || die "data validation report not found: $DATA_VALIDATION_REPORT"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  [[ -f "$FAST_TRAINING_AUTHORIZATION" && ! -L "$FAST_TRAINING_AUTHORIZATION" ]] || \
    die "fast training authorization not found or symlinked: $FAST_TRAINING_AUTHORIZATION"
  [[ -f "$MIXED_LOADER_REPORT" && ! -L "$MIXED_LOADER_REPORT" ]] || \
    die "mixed-loader report not found or symlinked: $MIXED_LOADER_REPORT"
fi
CURRENT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
validate_smoke_report "$SMOKE_REPORT"

RUN_ROOT_RESOLVED="$RUN_ROOT"
if [[ -n "$REQUESTED_RUN_DIR" ]]; then
  RUN_DIR="$("$PYTHON_BIN" - "$REQUESTED_RUN_DIR" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"
else
  RUN_DIR="$RUN_ROOT_RESOLVED/$RUN_NAME"
fi
case "$RUN_DIR/" in
  "$RUN_ROOT_RESOLVED"/*) ;;
  *) die "--run-dir must resolve beneath --run-root ($RUN_ROOT_RESOLVED): $RUN_DIR" ;;
esac
[[ "$RUN_DIR" != "$RUN_ROOT_RESOLVED" ]] || die "--run-dir cannot equal --run-root"
RUN_ROOT="$RUN_ROOT_RESOLVED"

SNAPSHOT="$RUN_DIR/snapshot.pt"
IDENTITY_FILE="$RUN_DIR/run_identity.json"
if ((RESUME == 1)); then
  [[ -f "$IDENTITY_FILE" ]] || die "--resume requires wrapper identity metadata: $IDENTITY_FILE"
  "$PYTHON_BIN" - "$IDENTITY_FILE" "$VARIANT" "$CURRENT_COMMIT" "$BATCH_SIZE" \
    "$GRADIENT_ACCUMULATION_STEPS" "$NODES" "$WORLD_SIZE" "$EFFECTIVE_GLOBAL_BATCH_SIZE" \
    "$MIN_GPU_MEMORY_MIB" "$RUN_NAME" "$DATASET_STAGE" "$DATASETS_CSV" \
    "$MAX_ITER" "$WARMUP_STEPS" "$LOG_EVERY" "$SAVE_EVERY" "$VAL_EVERY" \
    "$VIZ_EVERY" "$WANDB_RUN_ID" "$DATA_VALIDATION_POLICY" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
(
    variant, commit, batch_size, accumulation, nodes, world_size,
    effective_batch, min_memory, run_name, dataset_stage, datasets_csv,
    max_iter, warmup_steps, log_every, save_every, val_every, viz_every,
    wandb_run_id, data_validation_policy,
) = sys.argv[2:]
payload = json.loads(path.read_text())
expected = {
    "schema_version": 6,
    "dataset_stage": dataset_stage,
    "dataset_names": datasets_csv.split(","),
    "variant": variant,
    "git_commit": commit,
    "batch_size": int(batch_size),
    "gradient_accumulation_steps": int(accumulation),
    "node_count": int(nodes),
    "gpus_per_node": 8,
    "world_size": int(world_size),
    "effective_global_batch_size": int(effective_batch),
    "gpu_profile": {"model": "B200", "minimum_memory_mib": int(min_memory)},
    "schedule": {
        "max_iter": int(max_iter),
        "warmup_steps": int(warmup_steps),
        "log_every": int(log_every),
        "save_every": int(save_every),
        "val_every": int(val_every),
        "viz_every": int(viz_every),
    },
    "run_name": run_name,
}
problems = [f"{key}: {payload.get(key)!r} != {value!r}" for key, value in expected.items() if payload.get(key) != value]
if payload.get("wandb", {}).get("run_id") != wandb_run_id:
    problems.append(
        f"wandb.run_id: {payload.get('wandb', {}).get('run_id')!r} != {wandb_run_id!r}"
    )
if payload.get("data", {}).get("validation_policy") != data_validation_policy:
    problems.append(
        f"data.validation_policy: {payload.get('data', {}).get('validation_policy')!r} "
        f"!= {data_validation_policy!r}"
    )
if problems:
    raise SystemExit(f"resume identity mismatch in {path}: " + "; ".join(problems))
print(f"Validated resume identity: {path}")
PY
  if [[ -f "$SNAPSHOT" ]]; then
    echo "Resume checkpoint found: $SNAPSHOT"
  else
    echo "No checkpoint exists yet; --resume will retry this identified run from iteration 0."
  fi
else
  [[ ! -e "$SNAPSHOT" ]] || die "$SNAPSHOT exists; pass --resume to preserve and continue it"
  [[ ! -e "$IDENTITY_FILE" ]] || die "$IDENTITY_FILE exists; choose another --run-dir or pass --resume with a snapshot"
  [[ ! -e "$RUN_DIR" ]] || die "fresh --run-dir must not already exist: $RUN_DIR"
fi

export WAN_DIR="$WAN_PATH"
export VIDEOX_HOME="$VIDEOX_PATH"
export LACWM_DATA="$DATA_PATH"
export LACWM_RUNS="$RUN_ROOT"
export LACWM_PYTHON="$PYTHON_BIN"
export WANDB_MODE="$WANDB_MODE_VALUE"
export WANDB_PROJECT="$WANDB_PROJECT_VALUE"
export WANDB_ENTITY="$WANDB_ENTITY_VALUE"
export CUDA_VISIBLE_DEVICES="$GPUS"
export LACWM_NNODES="$NODES"
export LACWM_WORLD_SIZE="$WORLD_SIZE"
export LACWM_GPUS_PER_NODE="8"
export LACWM_MASTER_ADDR="$MASTER_ADDR_VALUE"
export LACWM_MASTER_PORT="$MASTER_PORT"
export LACWM_RDZV_ID="$RDZV_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export PYTHONPATH="$REPO_ROOT/tools/env/videox_shim:$VIDEOX_HOME:$PROJECT_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Apply the validated B200 RoCE/NCCL and node-local CUDA-cache defaults to
# non-interactive Slurm launches as well as interactive shells.
source "$TOOLS_DIR/env/activate_b200.sh"

PREFLIGHT=(
  "$PYTHON_BIN" "$TOOLS_DIR/training_preflight.py"
  --profile full
  --gpus "$GPUS"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR"
  --videox-home "$VIDEOX_HOME"
  --data-root "$LACWM_DATA"
  --run-root "$LACWM_RUNS"
  --data-validation-report "$DATA_VALIDATION_REPORT"
  --data-validation-policy "$DATA_VALIDATION_POLICY"
  --max-data-report-age-hours "$MAX_DATA_REPORT_AGE_HOURS"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
  --datasets "${DATASET_ARRAY[@]}"
)
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  PREFLIGHT+=(
    --fast-training-authorization "$FAST_TRAINING_AUTHORIZATION"
    --mixed-loader-report "$MIXED_LOADER_REPORT"
  )
fi

echo "Running guarded 8xB200 preflight..."
"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR"
"${PREFLIGHT[@]}"

EXPERIMENT="ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml"
if [[ "$VARIANT" == "explicit" ]]; then
  EXPERIMENT="ravenhuang/wan-dit/wan_dit_explicit_abc_agibot_droid_egodex.yaml"
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LAUNCH_DIR="$RUN_DIR/launches/${TIMESTAMP}_pid$$"
WANDB_ARGS=("wandb.enabled=false")
if [[ "$WANDB_MODE_VALUE" != "disabled" ]]; then
  WANDB_ARGS=("wandb.enabled=true" "wandb.project=$WANDB_PROJECT_VALUE" "+wandb.mode=$WANDB_MODE_VALUE" "+wandb.id=$WANDB_RUN_ID")
  [[ -n "$WANDB_ENTITY_VALUE" ]] && WANDB_ARGS+=("+wandb.entity=$WANDB_ENTITY_VALUE")
fi
DATASET_DELETE_ARGS=()
for dataset_pair in droid:Droid egodex:EgoDex agibot:Agibot abc:ABC; do
  dataset_key="${dataset_pair%%:*}"
  hydra_key="${dataset_pair#*:}"
  if [[ -z "${DATASET_SEEN[$dataset_key]:-}" ]]; then
    DATASET_DELETE_ARGS+=(
      "~dataset.datasets.$hydra_key"
      "~val_dataset.datasets.$hydra_key"
      "~viz_dataset.datasets.$hydra_key"
    )
  fi
done
TRAIN_ARGS=(
  train.py
  "+experiments_0908=$EXPERIMENT"
  "${DATASET_DELETE_ARGS[@]}"
  "name=$RUN_NAME"
  "data_loader.batch_size=$BATCH_SIZE"
  "trainer.config.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  "trainer.config.max_iter=$MAX_ITER"
  "lr_scheduler_factory.lr_lambda.warmup_steps=$WARMUP_STEPS"
  "trainer.config.logging.log_every=$LOG_EVERY"
  "trainer.config.saving.save_every=$SAVE_EVERY"
  "trainer.config.validation.val_every=$VAL_EVERY"
  "trainer.config.visualization.viz_every=$VIZ_EVERY"
  "hydra.run.dir=$RUN_DIR"
  "hydra.sweep.dir=$RUN_DIR"
  "${WANDB_ARGS[@]}"
)
if [[ -n "$TRANSITION_HANDOFF" ]]; then
  TRAIN_ARGS+=("trainer.config.transition_handoff_path=$TRANSITION_HANDOFF")
fi

build_distributed_command() {
  local -n output=$1
  local phase=$2
  local port=$3
  shift 3
  if ((10#$NODES == 1)); then
    output=(
      "$PYTHON_BIN" -m torch.distributed.run
      --standalone
      --nproc_per_node=8
      "$@"
    )
  else
    output=(
      srun
      "--nodes=$NODES"
      "--ntasks=$NODES"
      --ntasks-per-node=1
      --gpus-per-task=8
      --kill-on-bad-exit=1
      "$TOOLS_DIR/slurm/torchrun_node.sh"
      --python "$PYTHON_BIN"
      --nnodes "$NODES"
      --nproc-per-node 8
      --master-addr "$MASTER_ADDR_VALUE"
      --master-port "$port"
      --rdzv-id "${RDZV_ID}-${phase}"
      --
      "$@"
    )
  fi
}

build_distributed_command COMMAND train "$((10#$MASTER_PORT + 2))" "${TRAIN_ARGS[@]}"

printf 'Validated command (working directory %q):' "$PROJECT_ROOT"
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "Planned output: $RUN_DIR"
echo "Topology: nodes=$NODES gpus_per_node=8 world_size=$WORLD_SIZE${MASTER_ADDR_VALUE:+ master=$MASTER_ADDR_VALUE:$MASTER_PORT}"
echo "Batching: physical_per_gpu=$BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS effective_global=$EFFECTIVE_GLOBAL_BATCH_SIZE"
echo "Schedule: max_iter=$MAX_ITER warmup_steps=$WARMUP_STEPS log_every=$LOG_EVERY save_every=$SAVE_EVERY val_every=$VAL_EVERY viz_every=$VIZ_EVERY"
echo "Dataset stage: $DATASET_STAGE datasets=$DATASETS_CSV"
if [[ -n "$TRANSITION_HANDOFF" ]]; then
  echo "Immutable parent transition: path=$TRANSITION_HANDOFF handoff_sha256=$TRANSITION_HANDOFF_SHA256 parent_identity=$TRANSITION_PARENT_IDENTITY_SHA256 parent_snapshot_sha256=$TRANSITION_PARENT_SNAPSHOT_SHA256"
else
  echo "Immutable parent transition: none"
fi
echo "GPU profile: $NODES node(s) x 8 B200 with at least $MIN_GPU_MEMORY_MIB MiB total memory per device"
echo "W&B mode: $WANDB_MODE_VALUE${WANDB_PROJECT_VALUE:+, project=$WANDB_PROJECT_VALUE}${WANDB_ENTITY_VALUE:+, entity=$WANDB_ENTITY_VALUE}"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --execute after reviewing the command."
  exit 0
fi

if ((10#$NODES > 1)); then
  [[ "${SLURM_JOB_NUM_NODES:-}" == "$NODES" ]] || die \
    "multi-node execution requires a matching Slurm allocation: expected $NODES nodes, got ${SLURM_JOB_NUM_NODES:-unset}"
  [[ "${SLURM_NTASKS:-}" == "$NODES" ]] || die \
    "multi-node execution requires one Slurm task per node: expected $NODES, got ${SLURM_NTASKS:-unset}"
  command -v srun >/dev/null 2>&1 || die "srun is required for multi-node execution"
  [[ -x "$TOOLS_DIR/slurm/torchrun_node.sh" ]] || die \
    "multi-node torchrun helper is not executable: $TOOLS_DIR/slurm/torchrun_node.sh"
  [[ -x "$TOOLS_DIR/slurm/node_preflight.sh" ]] || die \
    "multi-node preflight helper is not executable: $TOOLS_DIR/slurm/node_preflight.sh"
fi

# Fixed per host/account rather than per run root: separate output roots must
# not allow two guarded wrappers to race through the idle-GPU check together.
LOCK_ROOT="${LACWM_HOST_LOCK_ROOT:-/run/user/${UID}}"
if [[ ! -d "$LOCK_ROOT" ]]; then
  [[ -z "${LACWM_HOST_LOCK_ROOT:-}" ]] || die \
    "configured runtime lock directory is unavailable: $LOCK_ROOT"
  LOCK_ROOT="/tmp/lacwm-runtime-${UID}"
  if [[ ! -e "$LOCK_ROOT" ]]; then
    (umask 077 && mkdir "$LOCK_ROOT") || die \
      "could not create fallback runtime lock directory: $LOCK_ROOT"
  fi
fi
[[ -d "$LOCK_ROOT" && ! -L "$LOCK_ROOT" ]] || die "runtime lock path is not a secure directory: $LOCK_ROOT"
[[ "$(stat -c %u "$LOCK_ROOT")" == "$UID" && "$(stat -c %a "$LOCK_ROOT")" == "700" ]] || \
  die "runtime lock directory must be owned by UID $UID with mode 700: $LOCK_ROOT"
LOCK_FILE="$LOCK_ROOT/lacwm-training.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another guarded lacwm launch holds $LOCK_FILE"

# Python multiprocessing creates AF_UNIX sockets beneath TMPDIR when loader
# workers transfer tensors. Linux limits those socket paths to roughly 108
# bytes, while the immutable run/provenance path can be much longer. Keep only
# this small, private socket directory under the short per-user runtime path;
# caches, checkpoints, logs, and all large files remain on the selected data
# volume. The directory is normally empty after workers exit.
MP_TMPDIR="$(mktemp -d "$LOCK_ROOT/lacwm-mp.XXXXXX")"
chmod 700 "$MP_TMPDIR"
cleanup_mp_tmpdir() {
  rmdir -- "$MP_TMPDIR" 2>/dev/null || true
}
trap cleanup_mp_tmpdir EXIT

if ((RESUME == 0)); then
  [[ ! -e "$RUN_DIR" ]] || die "fresh run directory appeared during launch: $RUN_DIR"
fi

# Stage fallible post-lock gates outside RUN_DIR. A transient gate failure leaves
# diagnostics but cannot strand a half-created run that is neither fresh nor resumable.
GATE_DIR="$RUN_ROOT/_prelaunch/${RUN_NAME}_${TIMESTAMP}_pid$$"
mkdir -p "$GATE_DIR/cache" "$GATE_DIR/tmp"
export HF_HOME="$GATE_DIR/cache/huggingface"
export TORCH_HOME="$GATE_DIR/cache/torch"
export XDG_CACHE_HOME="$GATE_DIR/cache/xdg"
export MPLCONFIGDIR="$GATE_DIR/cache/matplotlib"
export TRITON_CACHE_DIR="$GATE_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$GATE_DIR/cache/torchinductor"
export TMPDIR="$GATE_DIR/tmp"

# Re-check exact packages/source/assets and host/data state after taking the lock.
"$PYTHON_BIN" "$TOOLS_DIR/env/verify_b200_runtime.py" \
  --expected-gpus 8 \
  --require-b200 \
  --wan-dir "$WAN_DIR" > "$GATE_DIR/runtime_verification.json"
LOCKED_DATA_REPORT="$GATE_DIR/data_validation_report.json"
LOCKED_SMOKE_REPORT="$GATE_DIR/gradient_smoke_report.json"
cp -- "$DATA_VALIDATION_REPORT" "$LOCKED_DATA_REPORT"
cp -- "$SMOKE_REPORT" "$LOCKED_SMOKE_REPORT"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  cp -- "$FAST_TRAINING_AUTHORIZATION" "$GATE_DIR/fast_training_authorization.json"
  cp -- "$MIXED_LOADER_REPORT" "$GATE_DIR/mixed_loader_report.json"
fi
validate_smoke_report "$LOCKED_SMOKE_REPORT"
assert_repo_identity
LOCKED_PREFLIGHT_ARGS=(--json-out "$GATE_DIR/preflight.json")
if [[ "$DATA_VALIDATION_POLICY" == "strict" ]]; then
  LOCKED_PREFLIGHT_ARGS+=(--data-validation-report "$LOCKED_DATA_REPORT")
fi
"${PREFLIGHT[@]}" "${LOCKED_PREFLIGHT_ARGS[@]}"

NODE_DATA_REPORT="$LOCKED_DATA_REPORT"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  NODE_DATA_REPORT="$DATA_VALIDATION_REPORT"
fi
if ((10#$NODES > 1)); then
  echo "Running guarded runtime, data, GPU, and topology preflight on all $NODES nodes..."
  mkdir -p "$GATE_DIR/nodes"
  timeout --signal=TERM 1800s srun \
    "--nodes=$NODES" \
    "--ntasks=$NODES" \
    --ntasks-per-node=1 \
    --gpus-per-task=8 \
    --kill-on-bad-exit=1 \
    "$TOOLS_DIR/slurm/node_preflight.sh" \
    "$PYTHON_BIN" "$TOOLS_DIR" "$GPUS" "$WAN_DIR" "$VIDEOX_HOME" \
    "$LACWM_DATA" "$LACWM_RUNS" "$NODE_DATA_REPORT" \
    "$MAX_DATA_REPORT_AGE_HOURS" "$MIN_GPU_MEMORY_MIB" "$GATE_DIR/nodes" \
    "$DATASETS_CSV" "$DATA_VALIDATION_POLICY" \
    "$FAST_TRAINING_AUTHORIZATION" "$MIXED_LOADER_REPORT" \
    2>&1 | tee "$GATE_DIR/multinode_preflight.log"
fi

STAGED_IDENTITY="$GATE_DIR/run_identity.json"
IDENTITY_TARGET="$IDENTITY_FILE"
IDENTITY_DATA_REPORT="$LOCKED_DATA_REPORT"
IDENTITY_SMOKE_REPORT="$LOCKED_SMOKE_REPORT"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  # The immutable certificate binds the original evidence paths as well as
  # their bytes. Copies remain in GATE_DIR for forensics, while identity
  # validation intentionally consumes the path-bound originals.
  IDENTITY_DATA_REPORT="$DATA_VALIDATION_REPORT"
  IDENTITY_SMOKE_REPORT="$SMOKE_REPORT"
fi
if ((RESUME == 0)); then
  IDENTITY_TARGET="$STAGED_IDENTITY"
fi
IDENTITY_ARGS=(
  --dataset-stage "$DATASET_STAGE"
  --datasets "${DATASET_ARRAY[@]}"
  --variant "$VARIANT"
  --git-commit "$CURRENT_COMMIT"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --max-iter "$MAX_ITER"
  --warmup-steps "$WARMUP_STEPS"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --val-every "$VAL_EVERY"
  --viz-every "$VIZ_EVERY"
  --node-count "$NODES"
  --gpus-per-node 8
  --world-size "$WORLD_SIZE"
  --min-gpu-memory-mib "$MIN_GPU_MEMORY_MIB"
  --run-name "$RUN_NAME"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR"
  --videox-home "$VIDEOX_HOME"
  --data-root "$LACWM_DATA"
  --run-root "$RUN_ROOT"
  --run-dir "$RUN_DIR"
  --wandb-mode "$WANDB_MODE_VALUE"
  --wandb-project "$WANDB_PROJECT_VALUE"
  --wandb-entity "$WANDB_ENTITY_VALUE"
  --wandb-run-id "$WANDB_RUN_ID"
  --data-report "$IDENTITY_DATA_REPORT"
  --runtime-report "$GATE_DIR/runtime_verification.json"
  --smoke-report "$IDENTITY_SMOKE_REPORT"
  --data-validation-policy "$DATA_VALIDATION_POLICY"
)
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  IDENTITY_ARGS+=(
    --fast-training-authorization "$FAST_TRAINING_AUTHORIZATION"
    --mixed-loader-report "$MIXED_LOADER_REPORT"
  )
fi
if [[ -n "$TRANSITION_HANDOFF" ]]; then
  BASE_IDENTITY="$GATE_DIR/run_identity.base.json"
  EXPECTED_IDENTITY="$GATE_DIR/run_identity.expected.json"
  "$PYTHON_BIN" "$TOOLS_DIR/run_identity.py" create \
    --identity "$BASE_IDENTITY" "${IDENTITY_ARGS[@]}"
  bind_transition_identity "$BASE_IDENTITY" "$EXPECTED_IDENTITY"
  if ((RESUME == 0)); then
    mv -- "$EXPECTED_IDENTITY" "$STAGED_IDENTITY"
    IDENTITY_TARGET="$STAGED_IDENTITY"
  else
    "$PYTHON_BIN" - "$EXPECTED_IDENTITY" "$IDENTITY_FILE" <<'PY'
import json
import pathlib
import sys

expected_path, actual_path = map(pathlib.Path, sys.argv[1:])
expected = json.loads(expected_path.read_text(encoding="utf-8"))
actual = json.loads(actual_path.read_text(encoding="utf-8"))
if actual != expected:
    keys = sorted(
        key for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )
    raise SystemExit(
        "resume identity does not match immutable transition binding; "
        f"mismatched fields: {keys}"
    )
print(f"Validated transition-bound resume identity: {actual_path}")
PY
  fi
elif ((RESUME == 0)); then
  "$PYTHON_BIN" "$TOOLS_DIR/run_identity.py" create \
    --identity "$IDENTITY_TARGET" "${IDENTITY_ARGS[@]}"
else
  "$PYTHON_BIN" "$TOOLS_DIR/run_identity.py" validate \
    --identity "$IDENTITY_TARGET" "${IDENTITY_ARGS[@]}"
fi
export LACWM_RUN_IDENTITY_SHA256="$("$PYTHON_BIN" - "$IDENTITY_TARGET" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("identity_sha256")
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit(f"invalid identity_sha256 in {sys.argv[1]}")
print(value)
PY
)"

if ((RESUME == 0)); then
  [[ ! -e "$RUN_DIR" ]] || die "fresh run directory appeared before identity publication: $RUN_DIR"
  mkdir -p "$(dirname "$RUN_DIR")"
  mkdir "$RUN_DIR" || die "could not exclusively create fresh run directory: $RUN_DIR"
  "$PYTHON_BIN" - "$STAGED_IDENTITY" "$IDENTITY_FILE" <<'PY'
import os
import pathlib
import sys

source, target = map(pathlib.Path, sys.argv[1:])
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(source.read_bytes())
    handle.flush()
    os.fsync(handle.fileno())
PY
else
  [[ -d "$RUN_DIR" ]] || die "resume run directory disappeared: $RUN_DIR"
fi
mkdir -p "$LAUNCH_DIR/wandb" "$LAUNCH_DIR/wandb-data" "$LAUNCH_DIR/wandb-cache" "$LAUNCH_DIR/wandb-config" "$LAUNCH_DIR/cache" "$LAUNCH_DIR/tmp"
cp -- "$GATE_DIR/runtime_verification.json" "$LAUNCH_DIR/runtime_verification.json"
cp -- "$GATE_DIR/preflight.json" "$LAUNCH_DIR/preflight.json"
if [[ -d "$GATE_DIR/nodes" ]]; then
  cp -a -- "$GATE_DIR/nodes" "$LAUNCH_DIR/node_preflight"
  cp -- "$GATE_DIR/multinode_preflight.log" "$LAUNCH_DIR/multinode_preflight.log"
fi

export WANDB_DIR="$LAUNCH_DIR/wandb"
export WANDB_DATA_DIR="$LAUNCH_DIR/wandb-data"
export WANDB_CACHE_DIR="$LAUNCH_DIR/wandb-cache"
export WANDB_CONFIG_DIR="$LAUNCH_DIR/wandb-config"
export HF_HOME="$LAUNCH_DIR/cache/huggingface"
export TORCH_HOME="$LAUNCH_DIR/cache/torch"
export XDG_CACHE_HOME="$LAUNCH_DIR/cache/xdg"
export MPLCONFIGDIR="$LAUNCH_DIR/cache/matplotlib"
export TRITON_CACHE_DIR="$LAUNCH_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LAUNCH_DIR/cache/torchinductor"
export TMPDIR="$MP_TMPDIR"

build_distributed_command NCCL_COMMAND nccl "$MASTER_PORT" \
  "$TOOLS_DIR/env/nccl_probe.py" \
  --expected-nodes "$NODES" \
  --gpus-per-node 8 \
  --timeout-seconds 1200
echo "Validating torchrun + NCCL collectives across $NODES node(s) and $WORLD_SIZE GPUs..."
timeout --signal=TERM --kill-after=60s 1260s "${NCCL_COMMAND[@]}" \
  2>&1 | tee "$LAUNCH_DIR/nccl_probe.log"

echo "Validating three real-data DDP optimizer updates at the configured batching profile..."
build_distributed_command DDP_SMOKE_COMMAND ddp-smoke "$((10#$MASTER_PORT + 1))" \
  "$TOOLS_DIR/ddp_training_smoke.py" \
  --variant "$VARIANT" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --steps 3 \
  --expected-nodes "$NODES" \
  --gpus-per-node 8 \
  --datasets "${DATASET_ARRAY[@]}"
timeout --signal=TERM 1800s "${DDP_SMOKE_COMMAND[@]}" \
  2>&1 | tee "$LAUNCH_DIR/ddp_training_smoke.log"
{
  printf '#!/usr/bin/env bash\n'
  printf '%s\n' 'echo "This file records provenance only; resume through tools/launch_8xb200.sh." >&2' 'exit 2' ''
  for variable in \
    WAN_DIR VIDEOX_HOME LACWM_DATA LACWM_RUNS LACWM_PYTHON \
    WANDB_MODE WANDB_PROJECT WANDB_ENTITY CUDA_VISIBLE_DEVICES \
    LACWM_NNODES LACWM_WORLD_SIZE LACWM_GPUS_PER_NODE \
    LACWM_MASTER_ADDR LACWM_MASTER_PORT LACWM_RDZV_ID \
    PYTORCH_CUDA_ALLOC_CONF TORCH_NCCL_ASYNC_ERROR_HANDLING PYTHONPATH \
    LACWM_RUN_IDENTITY_SHA256 WANDB_DIR WANDB_DATA_DIR WANDB_CACHE_DIR \
    WANDB_CONFIG_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR \
    TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TMPDIR \
    LACWM_CHECKPOINT_REQUEST_FILE LACWM_CHECKPOINT_ACK_FILE \
    LACWM_SLURM_ATTEMPT_ID; do
    [[ -n "${!variable+x}" ]] || continue
    printf 'export %s=%q\n' "$variable" "${!variable}"
  done
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf 'exec'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$LAUNCH_DIR/command.sh"
chmod 600 "$LAUNCH_DIR/command.sh"

"$PYTHON_BIN" "$TOOLS_DIR/capture_provenance.py" \
  --output-dir "$LAUNCH_DIR" \
  --python "$PYTHON_BIN" \
  --kind full-8xb200 \
  --variant "$VARIANT" \
  --data-mode real \
  --run-name "$RUN_NAME" \
  --command-file "$LAUNCH_DIR/command.sh"
cp -- "$LOCKED_SMOKE_REPORT" "$LAUNCH_DIR/accepted_smoke_report.json"
cp -- "$LOCKED_DATA_REPORT" "$LAUNCH_DIR/accepted_data_validation_report.json"
if [[ "$DATA_VALIDATION_POLICY" == "files_only_user_waived_v1" ]]; then
  cp -- "$FAST_TRAINING_AUTHORIZATION" "$LAUNCH_DIR/accepted_fast_training_authorization.json"
  cp -- "$MIXED_LOADER_REPORT" "$LAUNCH_DIR/accepted_mixed_loader_report.json"
fi

# The qualification step can take many minutes.  Do not start the long run if
# another shell changed branches or edited tracked/untracked code meanwhile.
assert_repo_identity
echo "Launching in foreground. Console log: $LAUNCH_DIR/console.log"
cd "$PROJECT_ROOT"
set +e
"${COMMAND[@]}" 2>&1 | tee "$LAUNCH_DIR/console.log"
PIPE_STATUSES=("${PIPESTATUS[@]}")
STATUS=${PIPE_STATUSES[0]}
TEE_STATUS=${PIPE_STATUSES[1]}
set -e
printf '%s\n' "$STATUS" > "$LAUNCH_DIR/exit_code.txt"
if ((STATUS != 0)); then
  die "training exited with code $STATUS; artifacts preserved at $LAUNCH_DIR"
fi
if ((TEE_STATUS != 0)); then
  die "training completed but console logging failed with code $TEE_STATUS; artifacts at $LAUNCH_DIR"
fi
echo "Training completed successfully; artifacts: $LAUNCH_DIR"
