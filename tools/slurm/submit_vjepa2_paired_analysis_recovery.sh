#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

CONTROLLER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
STUDY_ROOT="${VJEPA_PAIRED_STUDY_ROOT:-$LACWM_BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3}"
TIMING_ROOT=""
PAIRED_LATENCY=""
TIMING_SUBMISSION=""
CONTROLLER_COMMIT=""
RECOVERY_ID=""
RECOVERY_ROOT=""
PYTHON_BIN="${LACWM_PYTHON:-$LACWM_BASE/envs/lacwm-b200-py310/bin/python}"
FAILED_ANALYZER_JOB_ID="481826"
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_vjepa2_paired_analysis_recovery.sh [options]

Recover only the post-timing analyzer failure of job 481826. The preserved
r1 paired evidence is reused by exact SHA-256; no model or benchmark runs.

Options:
  --study-root PATH
  --controller-commit SHA       Defaults to current clean HEAD
  --recovery-id ID              Default: paired-analysis-481826-<commit7>-r2
  --recovery-root PATH          Must be the derived external sibling
  --python PATH                 Must match the immutable study runtime
  --failed-analyzer-job-id ID   Must be 481826
  --execute                     Submit held, write receipt, then release
  -h, --help

Without --execute all checks are read-only. Execution uses the known cluster
contract: batch/coreai_chef_posttrain/normal, one B200, 8 CPU, 64G, one hour.
The allocated GPU satisfies the partition contract but is never used to load
a model or repeat timing.
EOF
}

while (($#)); do
  case "$1" in
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --controller-commit) CONTROLLER_COMMIT="${2:?}"; shift 2 ;;
    --recovery-id) RECOVERY_ID="${2:?}"; shift 2 ;;
    --recovery-root) RECOVERY_ROOT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --failed-analyzer-job-id)
      FAILED_ANALYZER_JOB_ID="${2:?}"
      shift 2
      ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

ACTUAL_COMMIT="$(git -C "$CONTROLLER_ROOT" rev-parse HEAD)"
TIMING_ROOT="$(
  dirname "$STUDY_ROOT"
)/_paired_recoveries/$(
  basename "$STUDY_ROOT"
)/paired-481133-43ed5d3-r1"
PAIRED_LATENCY="$TIMING_ROOT/paired_latency/paired_j1_nfe4_vs_vpm_nfe8.json"
TIMING_SUBMISSION="$TIMING_ROOT/submission.json"
if [[ -z "$CONTROLLER_COMMIT" ]]; then
  CONTROLLER_COMMIT="$ACTUAL_COMMIT"
fi
if [[ -z "$RECOVERY_ID" ]]; then
  RECOVERY_ID="paired-analysis-481826-${CONTROLLER_COMMIT:0:7}-r2"
fi
if [[ -z "$RECOVERY_ROOT" ]]; then
  RECOVERY_ROOT="$(
    dirname "$STUDY_ROOT"
  )/_paired_recoveries/$(
    basename "$STUDY_ROOT"
  )/$RECOVERY_ID"
fi

[[ "$CONTROLLER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "invalid controller commit"
[[ "$FAILED_ANALYZER_JOB_ID" == "481826" ]] || \
  die "failed analyzer job must be 481826"
for path in \
  "$CONTROLLER_ROOT" "$STUDY_ROOT" "$TIMING_ROOT" "$PAIRED_LATENCY" \
  "$TIMING_SUBMISSION" "$RECOVERY_ROOT" "$PYTHON_BIN"; do
  [[ "$path" == /* ]] || die "all paths must be absolute"
done
[[ "$ACTUAL_COMMIT" == "$CONTROLLER_COMMIT" ]] || \
  die "controller worktree differs from --controller-commit"
[[ "$(git -C "$CONTROLLER_ROOT" rev-parse --show-toplevel)" == \
  "$CONTROLLER_ROOT" ]] || die "launcher must run from a worktree root"
[[ -z "$(git -C "$CONTROLLER_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "controller worktree must be clean"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || \
  die "study root is invalid"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "study root is not canonical"
[[ -d "$TIMING_ROOT" && ! -L "$TIMING_ROOT" ]] || \
  die "r1 timing root is invalid"
[[ -f "$PAIRED_LATENCY" && ! -L "$PAIRED_LATENCY" ]] || \
  die "r1 paired timing is invalid"
[[ -f "$TIMING_SUBMISSION" && ! -L "$TIMING_SUBMISSION" ]] || \
  die "r1 timing receipt is invalid"
[[ -x "$PYTHON_BIN" ]] || die "recorded Python is unavailable"
[[ ! -e "$RECOVERY_ROOT" ]] || \
  die "fresh analysis recovery root already exists"

HELPER="$CONTROLLER_ROOT/tools/vjepa2_paired_analysis_recovery.py"
SBATCH_SCRIPT="$CONTROLLER_ROOT/tools/slurm/vjepa2_paired_analysis_recovery.sbatch"
for path in "$HELPER" "$SBATCH_SCRIPT"; do
  [[ -x "$path" ]] || die "required analyzer recovery entrypoint is not executable"
done
for command in git sacct squeue; do
  command -v "$command" >/dev/null 2>&1 || die "$command is unavailable"
done

FAILED_ACCOUNTING="$(
  sacct -j "$FAILED_ANALYZER_JOB_ID" -X -n -P \
    -o "JobIDRaw,JobName%128,Partition,Account,QOS,ReqCPUS,ReqMem,ReqNodes,ReqTRES%256,State,ExitCode,Elapsed,Timelimit,SubmitLine%4096,WorkDir%1024,NodeList%128,Submit,Start,End"
)" || die "could not query failed analyzer accounting"
[[ -n "$FAILED_ACCOUNTING" ]] || die "failed analyzer accounting is empty"

COMMON=(
  --study-root "$STUDY_ROOT"
  --controller-repo-root "$CONTROLLER_ROOT"
  --controller-commit "$CONTROLLER_COMMIT"
  --recovery-id "$RECOVERY_ID"
  --recovery-root "$RECOVERY_ROOT"
  --failed-accounting-row "$FAILED_ACCOUNTING"
  --python "$PYTHON_BIN"
)
"$PYTHON_BIN" "$HELPER" preflight "${COMMON[@]}"

ACTIVE="$(
  squeue -h -u "${USER:?}" -o "%i|%j|%T|%R" |
    awk -F'|' '$2 ~ /vjepa2-paired-analysis-481826-/'
)" || die "could not query active analyzer-recovery jobs"
[[ -z "$ACTIVE" ]] || \
  die "an analyzer-recovery job is already active: $ACTIVE"

echo "V-JEPA paired analyzer-only recovery preflight passed."
echo "Preserved r1 timing: $PAIRED_LATENCY"
echo "Failed post-run analyzer: $FAILED_ANALYZER_JOB_ID"
echo "Controller: $CONTROLLER_COMMIT at $CONTROLLER_ROOT"
echo "Fresh external root: $RECOVERY_ROOT"
echo "Scheduler: batch / coreai_chef_posttrain / normal, 1 B200, 8 CPU, 64G, 01:00:00"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --controller-commit '$CONTROLLER_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v scontrol >/dev/null 2>&1 || die "scontrol is unavailable"

mkdir "$RECOVERY_ROOT"
mkdir "$RECOVERY_ROOT/logs"
mkdir "$RECOVERY_ROOT/analysis"
PROTOCOL="$RECOVERY_ROOT/protocol.json"
SUBMISSION="$RECOVERY_ROOT/submission.json"
"$PYTHON_BIN" "$HELPER" write-protocol \
  "${COMMON[@]}" \
  --output "$PROTOCOL"

JOB_NAME="vjepa2-paired-analysis-481826-${CONTROLLER_COMMIT:0:7}"
SBATCH_ARGS=(
  --parsable
  --hold
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=1
  --cpus-per-task=8
  --mem=64G
  --time=01:00:00
  --partition=batch
  --account=coreai_chef_posttrain
  --qos=normal
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
  --study-root "$STUDY_ROOT"
  --protocol "$PROTOCOL"
  --submission "$SUBMISSION"
  --python "$PYTHON_BIN"
  --paired-latency "$PAIRED_LATENCY"
  --timing-submission "$TIMING_SUBMISSION"
)
JOB_ID="$("${SUBMIT_TOKENS[@]}")" || \
  die "Slurm rejected the held analyzer-recovery job"
JOB_ID="${JOB_ID%%;*}"
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || \
  die "Slurm returned an invalid analyzer job ID"
[[ "$JOB_ID" != "$FAILED_ANALYZER_JOB_ID" ]] || \
  die "Slurm reused the failed analyzer job ID"

SUBMISSION_ARGS=(
  write-submission
  --protocol "$PROTOCOL"
  --job-id "$JOB_ID"
  --job-name "$JOB_NAME"
  --output "$SUBMISSION"
)
for token in "${SUBMIT_TOKENS[@]}"; do
  SUBMISSION_ARGS+=("--submit-line-token=$token")
done
"$PYTHON_BIN" "$HELPER" "${SUBMISSION_ARGS[@]}"

scontrol release "$JOB_ID" || \
  die "receipt is durable but held analyzer job $JOB_ID could not be released"

echo "Submitted analyzer-only recovery job: $JOB_ID"
echo "Protocol: $PROTOCOL"
echo "Submission receipt: $SUBMISSION"
echo "Analysis JSON: $RECOVERY_ROOT/analysis/analysis.json"
echo "Analysis Markdown: $RECOVERY_ROOT/analysis/analysis.md"
