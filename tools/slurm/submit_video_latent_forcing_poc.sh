#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/video_latent_forcing_poc.sbatch"

MODE="calibrate"
PYTHON_BIN=""
ARM=""
DATA_ROOT=""
TRAIN_MANIFEST=""
VALIDATION_MANIFEST=""
ARTIFACT_ROOT=""
RUN_ID=""
OPTIMIZER_SEED="1234"
CALIBRATION_RECORD=""
PHASE1_GATE_RECORD=""
RESUME=""
MICRO_BATCH_SIZE=""
WANDB_ENTITY_VALUE=""
WANDB_PROJECT_VALUE=""
EXPECTED_COMMIT=""
PARTITION="batch"
ACCOUNT=""
QOS=""
CONSTRAINT="b200"
TIME_LIMIT="04:00:00"
CPUS="128"
MEMORY="900G"
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_video_latent_forcing_poc.sh [options]

Dry-run by default. The default mode is an exact 200-update calibration on one
8xB200 node. Training uses the frozen 5k/20k update schedule and requires its
arm-matched successful calibration record.

Required:
  --python PATH
  --arm phase1|B0|A1|L1
  --data-root PATH
  --train-manifest PATH
  --validation-manifest PATH
  --artifact-root PATH          Must be below /lustre, /mnt/data1, or /mnt/data2
  --run-id ID

Modes/options:
  --mode calibrate|train        Default: calibrate
  --calibration-record PATH     Required for train
  --phase1-gate-record PATH     Required for B0/A1/L1 train
  --resume PATH                 Train-only checkpoint resume in the same run
  --micro-batch-size N          Per-GPU microbatch; accumulation keeps batch 256
  --seed N                      Frozen optimizer seed (default: 1234)
  --wandb-entity NAME           Must be paired with --wandb-project
  --wandb-project NAME          Explicitly acknowledged as private by launcher

Submission:
  --expected-commit SHA         Required with --execute
  --partition NAME              Default: batch
  --account NAME
  --qos NAME
  --constraint NAME             Default: b200
  --time HH:MM:SS               Default: 04:00:00
  --cpus N                      Default: 128
  --mem VALUE                   Default: 900G
  --execute                     Submit after all fail-closed checks
EOF
}

while (($#)); do
  case "$1" in
    --mode) [[ $# -ge 2 ]] || die "$1 requires a value"; MODE="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "$1 requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --arm) [[ $# -ge 2 ]] || die "$1 requires a value"; ARM="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "$1 requires a value"; DATA_ROOT="$2"; shift 2 ;;
    --train-manifest) [[ $# -ge 2 ]] || die "$1 requires a value"; TRAIN_MANIFEST="$2"; shift 2 ;;
    --validation-manifest) [[ $# -ge 2 ]] || die "$1 requires a value"; VALIDATION_MANIFEST="$2"; shift 2 ;;
    --artifact-root) [[ $# -ge 2 ]] || die "$1 requires a value"; ARTIFACT_ROOT="$2"; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || die "$1 requires a value"; RUN_ID="$2"; shift 2 ;;
    --seed) [[ $# -ge 2 ]] || die "$1 requires a value"; OPTIMIZER_SEED="$2"; shift 2 ;;
    --calibration-record) [[ $# -ge 2 ]] || die "$1 requires a value"; CALIBRATION_RECORD="$2"; shift 2 ;;
    --phase1-gate-record) [[ $# -ge 2 ]] || die "$1 requires a value"; PHASE1_GATE_RECORD="$2"; shift 2 ;;
    --resume) [[ $# -ge 2 ]] || die "$1 requires a value"; RESUME="$2"; shift 2 ;;
    --micro-batch-size) [[ $# -ge 2 ]] || die "$1 requires a value"; MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --wandb-entity) [[ $# -ge 2 ]] || die "$1 requires a value"; WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --wandb-project) [[ $# -ge 2 ]] || die "$1 requires a value"; WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "$1 requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --partition) [[ $# -ge 2 ]] || die "$1 requires a value"; PARTITION="$2"; shift 2 ;;
    --account) [[ $# -ge 2 ]] || die "$1 requires a value"; ACCOUNT="$2"; shift 2 ;;
    --qos) [[ $# -ge 2 ]] || die "$1 requires a value"; QOS="$2"; shift 2 ;;
    --constraint) [[ $# -ge 2 ]] || die "$1 requires a value"; CONSTRAINT="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || die "$1 requires a value"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) [[ $# -ge 2 ]] || die "$1 requires a value"; CPUS="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "$1 requires a value"; MEMORY="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$MODE" == "calibrate" || "$MODE" == "train" ]] || die "invalid --mode"
[[ "$ARM" == "phase1" || "$ARM" == "B0" || "$ARM" == "A1" || "$ARM" == "L1" ]] || \
  die "invalid --arm"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || die "unsafe --run-id"
[[ "$OPTIMIZER_SEED" == "1234" || "$OPTIMIZER_SEED" == "2234" || "$OPTIMIZER_SEED" == "3234" ]] || die "invalid frozen optimizer seed"
[[ "$ARM" != "phase1" || "$OPTIMIZER_SEED" == "1234" ]] || die "Phase-1 seed is frozen to 1234"
[[ "$PYTHON_BIN" == /* ]] || die "--python must be absolute"
PYTHON_TARGET="$(readlink -f "$PYTHON_BIN")"
[[ -f "$PYTHON_TARGET" && -x "$PYTHON_TARGET" ]] || die "resolved --python is not executable"
for required in "$DATA_ROOT" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$ARTIFACT_ROOT"; do
  [[ "$required" == /* ]] || die "all data/artifact paths must be absolute"
done
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "data root is unavailable or symlinked"
[[ -f "$TRAIN_MANIFEST" && ! -L "$TRAIN_MANIFEST" ]] || die "train manifest is unavailable or symlinked"
[[ -f "$VALIDATION_MANIFEST" && ! -L "$VALIDATION_MANIFEST" ]] || die "validation manifest is unavailable or symlinked"
[[ -f "$SBATCH_SCRIPT" && ! -L "$SBATCH_SCRIPT" ]] || die "Slurm entrypoint is unavailable or symlinked"
ARTIFACT_ROOT="$(readlink -m "$ARTIFACT_ROOT")"
case "$ARTIFACT_ROOT/" in
  /lustre/*|/mnt/data1/*|/mnt/data2/*) ;;
  *) die "artifact root must be below /lustre, /mnt/data1, or /mnt/data2" ;;
esac
[[ "$ARTIFACT_ROOT" != "$REPO_ROOT" && "$ARTIFACT_ROOT" != "$REPO_ROOT/"* ]] || \
  die "artifact root cannot be inside this Git repository"

if [[ "$MODE" == "calibrate" ]]; then
  [[ -z "$CALIBRATION_RECORD" && -z "$PHASE1_GATE_RECORD" && -z "$RESUME" ]] || \
    die "calibration cannot take gate, calibration, or resume records"
else
  [[ "$CALIBRATION_RECORD" == /* && -f "$CALIBRATION_RECORD" && ! -L "$CALIBRATION_RECORD" ]] || \
    die "train requires an absolute, non-symlink --calibration-record"
fi
if [[ "$MODE" == "train" && "$ARM" != "phase1" ]]; then
  [[ "$PHASE1_GATE_RECORD" == /* && -f "$PHASE1_GATE_RECORD" && ! -L "$PHASE1_GATE_RECORD" ]] || \
    die "dual-arm train requires an absolute, non-symlink --phase1-gate-record"
else
  [[ -z "$PHASE1_GATE_RECORD" ]] || die "--phase1-gate-record is valid only for dual-arm train"
fi
if [[ -n "$RESUME" ]]; then
  [[ "$MODE" == "train" && "$RESUME" == /* && -f "$RESUME" && ! -L "$RESUME" ]] || \
    die "--resume requires a train-mode absolute checkpoint"
  [[ -d "$ARTIFACT_ROOT/$RUN_ID" ]] || die "resume run directory does not exist"
else
  [[ ! -e "$ARTIFACT_ROOT/$RUN_ID" ]] || die "new run directory already exists"
fi
if [[ -n "$MICRO_BATCH_SIZE" ]]; then
  [[ "$MICRO_BATCH_SIZE" =~ ^(1|2|4|8|16|32)$ ]] || \
    die "micro batch must divide the frozen per-GPU batch 32"
fi
if [[ -n "$WANDB_ENTITY_VALUE" || -n "$WANDB_PROJECT_VALUE" ]]; then
  [[ -n "$WANDB_ENTITY_VALUE" && -n "$WANDB_PROJECT_VALUE" ]] || \
    die "W&B entity/project must be supplied together"
fi
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be positive"
[[ "$TIME_LIMIT" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || die "invalid --time"

TOOL_ARGS=(
  "$MODE"
  --arm "$ARM"
  --data-root "$DATA_ROOT"
  --train-manifest "$TRAIN_MANIFEST"
  --validation-manifest "$VALIDATION_MANIFEST"
  --artifact-root "$ARTIFACT_ROOT"
  --run-id "$RUN_ID"
  --seed "$OPTIMIZER_SEED"
)
[[ -n "$CALIBRATION_RECORD" ]] && TOOL_ARGS+=(--calibration-record "$CALIBRATION_RECORD")
[[ -n "$PHASE1_GATE_RECORD" ]] && TOOL_ARGS+=(--phase1-gate-record "$PHASE1_GATE_RECORD")
[[ -n "$RESUME" ]] && TOOL_ARGS+=(--resume "$RESUME")
[[ -n "$MICRO_BATCH_SIZE" ]] && TOOL_ARGS+=(--micro-batch-size "$MICRO_BATCH_SIZE")
if [[ -n "$WANDB_ENTITY_VALUE" ]]; then
  TOOL_ARGS+=(
    --wandb
    --wandb-entity "$WANDB_ENTITY_VALUE"
    --wandb-project "$WANDB_PROJECT_VALUE"
    --wandb-private-project-ack
  )
fi

JOB_NAME="vlf-${MODE}-${ARM}-${RUN_ID}"
SBATCH_ARGS=(
  --job-name "$JOB_NAME"
  --partition "$PARTITION"
  --constraint "$CONSTRAINT"
  --time "$TIME_LIMIT"
  --cpus-per-task "$CPUS"
  --mem "$MEMORY"
  --output "$ARTIFACT_ROOT/_slurm/%x-%j.out"
  --error "$ARTIFACT_ROOT/_slurm/%x-%j.err"
)
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account "$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos "$QOS")

printf 'Resolved command (dry-run unless --execute):\n'
printf ' %q' sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT" -- "${TOOL_ARGS[@]}"
printf '\n'

if ((EXECUTE == 0)); then
  exit 0
fi
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "--execute requires full --expected-commit"
ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || die "Git commit differs from --expected-commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || die "--execute requires a clean worktree"
command -v sbatch >/dev/null || die "sbatch is unavailable"
command -v squeue >/dev/null || die "squeue is unavailable"
if squeue --noheader --user "$(id -un)" --name "$JOB_NAME" | grep -q .; then
  die "an active job already has the exact run job name"
fi
mkdir -p "$ARTIFACT_ROOT/_slurm"
JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT" -- "${TOOL_ARGS[@]}")"
[[ "$JOB_ID" =~ ^[0-9]+([.;][A-Za-z0-9._-]+)?$ ]] || die "unexpected sbatch response: $JOB_ID"
printf 'Submitted %s\n' "$JOB_ID"
