#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

CONTROLLER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
STUDY_ROOT="${VJEPA_PAIRED_STUDY_ROOT:-$LACWM_BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3}"
SCIENTIFIC_ROOT="${VJEPA_PAIRED_SCIENTIFIC_ROOT:-$LACWM_BASE/src/vjepa2-latent-forcing/dual-video-diffused-lacwm}"
SCIENTIFIC_COMMIT="9cf8e6922f35a5d6645e3128545953723bf54da2"
CONTROLLER_COMMIT=""
FAILED_JOB_ID="481133"
FINAL_STAGE_JOB_ID="481132"
RECOVERY_ID=""
RECOVERY_ROOT=""
PYTHON_BIN="${LACWM_PYTHON:-$LACWM_BASE/envs/lacwm-b200-py310/bin/python}"
WAN_DIR_VALUE="${WAN_DIR:-$LACWM_BASE/wan_fun_1.3b_control}"
VIDEOX_HOME_VALUE="${VIDEOX_HOME:-$LACWM_BASE/VideoX-Fun-1d6d9c3}"
PARTITION="batch"
ACCOUNT="coreai_chef_posttrain"
QOS="normal"
TIME_LIMIT="04:00:00"
CPUS="32"
MEMORY="256G"
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_vjepa2_paired_latency_recovery.sh [options]

Recover the exact validator-only failure of paired timing job 481133 without
editing the immutable v3 study. The new allocation repeats the preregistered
same-B200 comparison:

  J1 autonomous NFE=4 versus VPM autonomous NFE=8
  sample 0, batch 1, 20 warmup pairs, 100 counterbalanced timed pairs

The original empty paired_latency directory, original submission, and failed
logs remain untouched. All new receipts, logs, timing evidence, and analysis
are created exclusively under a fresh external recovery root.

Pinned provenance:
  failed paired job:       481133 (FAILED 2:0 before timing)
  final update-1000 array: 481132 (five COMPLETED 0:0 tasks)
  scientific repository:  clean commit 9cf8e6922f35a5d6645e3128545953723bf54da2

Options:
  --study-root PATH
  --controller-commit SHA       Defaults to current clean HEAD
  --scientific-repo-root PATH
  --scientific-commit SHA       Must be the pinned 9cf8e69 commit
  --failed-paired-job-id ID     Must be 481133
  --final-stage-job-id ID       Must be 481132
  --recovery-id ID              Default: paired-481133-<commit7>-r1
  --recovery-root PATH          Must be the derived external path
  --python PATH
  --wan-dir PATH
  --videox-home PATH
  --partition NAME              Must be batch
  --account NAME                Must be coreai_chef_posttrain
  --qos NAME                    Must be normal
  --time HH:MM:SS               Must be 04:00:00
  --cpus COUNT                  Must be 32
  --memory SIZE                 Must be 256G
  --execute                     Submit held, write receipt, then release
  -h, --help

Without --execute, every check is read-only and the recovery root must be
absent. No live Slurm dependency is added: the original afterok predecessor
has completed and is revalidated from its exact five accounting rows.
EOF
}

while (($#)); do
  case "$1" in
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --controller-commit) CONTROLLER_COMMIT="${2:?}"; shift 2 ;;
    --scientific-repo-root) SCIENTIFIC_ROOT="${2:?}"; shift 2 ;;
    --scientific-commit) SCIENTIFIC_COMMIT="${2:?}"; shift 2 ;;
    --failed-paired-job-id) FAILED_JOB_ID="${2:?}"; shift 2 ;;
    --final-stage-job-id) FINAL_STAGE_JOB_ID="${2:?}"; shift 2 ;;
    --recovery-id) RECOVERY_ID="${2:?}"; shift 2 ;;
    --recovery-root) RECOVERY_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --account) ACCOUNT="${2:?}"; shift 2 ;;
    --qos) QOS="${2:?}"; shift 2 ;;
    --time) TIME_LIMIT="${2:?}"; shift 2 ;;
    --cpus) CPUS="${2:?}"; shift 2 ;;
    --memory) MEMORY="${2:?}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

ACTUAL_COMMIT="$(git -C "$CONTROLLER_ROOT" rev-parse HEAD)"
if [[ -z "$CONTROLLER_COMMIT" ]]; then
  CONTROLLER_COMMIT="$ACTUAL_COMMIT"
fi
if [[ -z "$RECOVERY_ID" ]]; then
  RECOVERY_ID="paired-481133-${CONTROLLER_COMMIT:0:7}-r1"
fi
if [[ -z "$RECOVERY_ROOT" ]]; then
  RECOVERY_ROOT="$(dirname "$STUDY_ROOT")/_paired_recoveries/$(basename "$STUDY_ROOT")/$RECOVERY_ID"
fi

[[ "$CONTROLLER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "invalid controller commit"
[[ "$SCIENTIFIC_COMMIT" == \
  "9cf8e6922f35a5d6645e3128545953723bf54da2" ]] || \
  die "scientific commit must be the immutable trained commit"
[[ "$FAILED_JOB_ID" == "481133" ]] || die "failed paired job must be 481133"
[[ "$FINAL_STAGE_JOB_ID" == "481132" ]] || \
  die "final stage job must be 481132"
[[ "$PARTITION" == "batch" ]] || die "partition must preserve batch"
[[ "$ACCOUNT" == "coreai_chef_posttrain" ]] || \
  die "account must preserve coreai_chef_posttrain"
[[ "$QOS" == "normal" ]] || die "QOS must preserve normal"
[[ "$TIME_LIMIT" == "04:00:00" ]] || die "time limit must preserve 04:00:00"
[[ "$CPUS" == "32" ]] || die "CPU count must preserve 32"
[[ "$MEMORY" == "256G" ]] || die "memory must preserve 256G"
for scalar in \
  "$PARTITION" "$ACCOUNT" "$QOS" "$TIME_LIMIT" "$CPUS" "$MEMORY" \
  "$RECOVERY_ID"; do
  [[ "$scalar" != *[[:space:]]* ]] || die "scalar contains whitespace"
done
for path in \
  "$CONTROLLER_ROOT" "$SCIENTIFIC_ROOT" "$STUDY_ROOT" "$RECOVERY_ROOT" \
  "$PYTHON_BIN" "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE"; do
  [[ "$path" == /* ]] || die "all paths must be absolute"
done
[[ "$ACTUAL_COMMIT" == "$CONTROLLER_COMMIT" ]] || \
  die "controller worktree differs from --controller-commit"
[[ -z "$(git -C "$CONTROLLER_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "controller worktree must be clean"
[[ "$(git -C "$CONTROLLER_ROOT" rev-parse --show-toplevel)" == \
  "$CONTROLLER_ROOT" ]] || die "launcher must run from a worktree root"
[[ -d "$SCIENTIFIC_ROOT" && ! -L "$SCIENTIFIC_ROOT" ]] || \
  die "scientific worktree is invalid"
[[ "$(cd "$SCIENTIFIC_ROOT" && pwd -P)" == "$SCIENTIFIC_ROOT" ]] || \
  die "scientific worktree must use its canonical path"
[[ "$(git -C "$SCIENTIFIC_ROOT" rev-parse HEAD)" == "$SCIENTIFIC_COMMIT" ]] || \
  die "scientific worktree HEAD changed"
[[ -z "$(git -C "$SCIENTIFIC_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "scientific worktree must be clean"
[[ "$CONTROLLER_ROOT" != "$SCIENTIFIC_ROOT" ]] || \
  die "controller and scientific worktrees must differ"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is invalid"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "study root is not canonical"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime assets are unavailable"
[[ "$(cd "$WAN_DIR_VALUE" && pwd -P)" == "$WAN_DIR_VALUE" ]] || \
  die "Wan directory must use its canonical path"
[[ "$(cd "$VIDEOX_HOME_VALUE" && pwd -P)" == "$VIDEOX_HOME_VALUE" ]] || \
  die "VideoX checkout must use its canonical path"
[[ ! -e "$RECOVERY_ROOT" ]] || die "fresh recovery root already exists"
cd "$CONTROLLER_ROOT"

HELPER="$CONTROLLER_ROOT/tools/vjepa2_paired_recovery.py"
SBATCH_SCRIPT="$CONTROLLER_ROOT/tools/slurm/vjepa2_paired_latency_recovery.sbatch"
for path in "$HELPER" "$SBATCH_SCRIPT"; do
  [[ -x "$path" ]] || die "required recovery entrypoint is not executable: $path"
done
for command in awk git sacct squeue; do
  command -v "$command" >/dev/null 2>&1 || die "$command is unavailable"
done

LOG_DIR="$STUDY_ROOT/../_slurm/logs"
# Construct the exact original shell truncation without truncating LOG_DIR.
FAILED_JOB_NAME="vjepa2-$(basename "$STUDY_ROOT" | cut -c1-49)-paired-latency"
FAILED_STDOUT="$LOG_DIR/$FAILED_JOB_NAME-481133.out"
FAILED_STDERR="$LOG_DIR/$FAILED_JOB_NAME-481133.err"

FAILED_ACCOUNTING="$(
  sacct -j "$FAILED_JOB_ID" -X -n -P \
    -o "JobIDRaw,JobName%128,Partition,Account,QOS,ReqCPUS,ReqMem,ReqNodes,ReqTRES%256,State,ExitCode,Elapsed,Timelimit,SubmitLine%4096,WorkDir%1024,NodeList%128,Submit,Start,End"
)" || die "could not query failed paired-job accounting"
mapfile -t FINAL_ACCOUNTING < <(
  sacct -j "$FINAL_STAGE_JOB_ID" --array -X -n -P \
    -o "JobID%64,State,ExitCode,JobName%128"
) || die "could not query final-array accounting"
[[ -n "$FAILED_ACCOUNTING" ]] || die "failed paired-job accounting is empty"
[[ "${#FINAL_ACCOUNTING[@]}" -eq 5 ]] || \
  die "final-array accounting does not contain five tasks"

COMMON=(
  --study-root "$STUDY_ROOT"
  --controller-repo-root "$CONTROLLER_ROOT"
  --controller-commit "$CONTROLLER_COMMIT"
  --scientific-repo-root "$SCIENTIFIC_ROOT"
  --scientific-commit "$SCIENTIFIC_COMMIT"
  --recovery-id "$RECOVERY_ID"
  --recovery-root "$RECOVERY_ROOT"
  --failed-job-id "$FAILED_JOB_ID"
  --final-stage-job-id "$FINAL_STAGE_JOB_ID"
  --failed-stdout "$FAILED_STDOUT"
  --failed-stderr "$FAILED_STDERR"
  --failed-accounting-row "$FAILED_ACCOUNTING"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
)
for row in "${FINAL_ACCOUNTING[@]}"; do
  COMMON+=(--final-accounting-row "$row")
done

"$PYTHON_BIN" "$HELPER" preflight "${COMMON[@]}"

JOB_NAME="vjepa2-paired-recovery-481133-${CONTROLLER_COMMIT:0:7}"
ACTIVE="$(
  squeue -h -u "${USER:?}" -o "%i|%j|%T|%R" |
    awk -F'|' '$2 ~ /vjepa2-paired-recovery-481133-/'
)" || die "could not query active paired-recovery jobs"
[[ -z "$ACTIVE" ]] || die "a paired-recovery job is already active: $ACTIVE"

echo "V-JEPA paired-latency recovery preflight passed."
echo "Study remains read-only: $STUDY_ROOT"
echo "Controller: $CONTROLLER_COMMIT at $CONTROLLER_ROOT"
echo "Scientific code: $SCIENTIFIC_COMMIT at $SCIENTIFIC_ROOT"
echo "Failed predecessor: $FAILED_JOB_ID"
echo "Completed afterok predecessor: $FINAL_STAGE_JOB_ID"
echo "Fresh external root: $RECOVERY_ROOT"
echo "Scheduler: $PARTITION / $ACCOUNT / $QOS, 1 B200, 32 CPU, 256G, 04:00:00"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --controller-commit '$CONTROLLER_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v scontrol >/dev/null 2>&1 || die "scontrol is unavailable"

RECOVERY_PARENT="$(dirname "$RECOVERY_ROOT")"
mkdir -p "$RECOVERY_PARENT"
mkdir "$RECOVERY_ROOT"
mkdir "$RECOVERY_ROOT/logs"
mkdir "$RECOVERY_ROOT/paired_latency"
mkdir "$RECOVERY_ROOT/analysis"
PROTOCOL="$RECOVERY_ROOT/protocol.json"
RECOVERY_SUBMISSION="$RECOVERY_ROOT/submission.json"

"$PYTHON_BIN" "$HELPER" write-protocol \
  "${COMMON[@]}" \
  --output "$PROTOCOL"

SBATCH_ARGS=(
  --parsable
  --hold
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=1
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --qos="$QOS"
  --chdir="$CONTROLLER_ROOT"
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="$JOB_NAME"
  --output="$RECOVERY_ROOT/logs/%x-%j.out"
  --error="$RECOVERY_ROOT/logs/%x-%j.err"
)
SUBMIT_TOKENS=(
  sbatch
  "${SBATCH_ARGS[@]}"
  "$SBATCH_SCRIPT"
  --controller-root "$CONTROLLER_ROOT"
  --controller-commit "$CONTROLLER_COMMIT"
  --scientific-root "$SCIENTIFIC_ROOT"
  --scientific-commit "$SCIENTIFIC_COMMIT"
  --study-root "$STUDY_ROOT"
  --protocol "$PROTOCOL"
  --recovery-submission "$RECOVERY_SUBMISSION"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
)
JOB_ID="$(
  "${SUBMIT_TOKENS[@]}"
)" || die "Slurm rejected the held paired-recovery job"
JOB_ID="${JOB_ID%%;*}"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || die "Slurm returned an invalid job ID"
[[ "$JOB_ID" != "$FAILED_JOB_ID" ]] || die "Slurm reused the failed job ID"

SUBMISSION_ARGS=(
  write-submission
  --protocol "$PROTOCOL"
  --job-id "$JOB_ID"
  --job-name "$JOB_NAME"
  --output "$RECOVERY_SUBMISSION"
)
for token in "${SUBMIT_TOKENS[@]}"; do
  SUBMISSION_ARGS+=("--submit-line-token=$token")
done
"$PYTHON_BIN" "$HELPER" "${SUBMISSION_ARGS[@]}"

scontrol release "$JOB_ID" || \
  die "receipt is durable but held recovery job $JOB_ID could not be released"

echo "Submitted paired-recovery job: $JOB_ID"
echo "Protocol: $PROTOCOL"
echo "Submission receipt: $RECOVERY_SUBMISSION"
echo "Timing evidence: $RECOVERY_ROOT/paired_latency/paired_j1_nfe4_vs_vpm_nfe8.json"
echo "Analysis: $RECOVERY_ROOT/analysis/analysis.json"
