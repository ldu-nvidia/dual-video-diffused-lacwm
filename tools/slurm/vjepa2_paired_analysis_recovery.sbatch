#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

CONTROLLER_ROOT=""
CONTROLLER_COMMIT=""
STUDY_ROOT=""
PROTOCOL=""
SUBMISSION=""
PYTHON_BIN=""
PAIRED_LATENCY=""
TIMING_SUBMISSION=""

while (($#)); do
  case "$1" in
    --controller-root) CONTROLLER_ROOT="${2:?}"; shift 2 ;;
    --controller-commit) CONTROLLER_COMMIT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --protocol) PROTOCOL="${2:?}"; shift 2 ;;
    --submission) SUBMISSION="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --paired-latency) PAIRED_LATENCY="${2:?}"; shift 2 ;;
    --timing-submission) TIMING_SUBMISSION="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in \
  "$CONTROLLER_ROOT" "$CONTROLLER_COMMIT" "$STUDY_ROOT" "$PROTOCOL" \
  "$SUBMISSION" "$PYTHON_BIN" "$PAIRED_LATENCY" "$TIMING_SUBMISSION"; do
  [[ -n "$value" ]] || die "all analyzer-recovery arguments are required"
done
for path in \
  "$CONTROLLER_ROOT" "$STUDY_ROOT" "$PROTOCOL" "$SUBMISSION" \
  "$PYTHON_BIN" "$PAIRED_LATENCY" "$TIMING_SUBMISSION"; do
  [[ "$path" == /* ]] || die "all analyzer-recovery paths must be absolute"
done
[[ "$CONTROLLER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "controller commit must be a full lowercase SHA-1"
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "analyzer recovery requires one non-array Slurm job"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "analyzer recovery is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "analyzer recovery requires one node"
[[ "${SLURM_CPUS_PER_TASK:-8}" == "8" ]] || \
  die "analyzer recovery requires eight CPUs"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || \
  die "batch partition did not provide its required single GPU"
[[ -x "$PYTHON_BIN" ]] || die "recorded Python is unavailable"
[[ -f "$PROTOCOL" && ! -L "$PROTOCOL" ]] || \
  die "analysis protocol is invalid"
[[ -f "$SUBMISSION" && ! -L "$SUBMISSION" ]] || \
  die "analysis submission is invalid"
[[ -f "$PAIRED_LATENCY" && ! -L "$PAIRED_LATENCY" ]] || \
  die "paired timing evidence is invalid"
[[ -f "$TIMING_SUBMISSION" && ! -L "$TIMING_SUBMISSION" ]] || \
  die "timing recovery submission is invalid"
[[ "$(git -C "$CONTROLLER_ROOT" rev-parse HEAD)" == \
  "$CONTROLLER_COMMIT" ]] || die "analysis controller HEAD changed"
[[ -z "$(git -C "$CONTROLLER_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "analysis controller worktree is dirty"

HELPER="$CONTROLLER_ROOT/tools/vjepa2_paired_analysis_recovery.py"
ANALYZER="$CONTROLLER_ROOT/tools/analyze_vjepa2_controlled_study.py"
for path in "$HELPER" "$ANALYZER"; do
  [[ -f "$path" && ! -L "$path" ]] || \
    die "required analyzer-only file is invalid: $path"
done

ANALYSIS_JSON="$(
  "$PYTHON_BIN" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["paths"]["analysis_json"])
PY
)"
ANALYSIS_MARKDOWN="$(
  "$PYTHON_BIN" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["paths"]["analysis_markdown"])
PY
)"
[[ ! -e "$ANALYSIS_JSON" && ! -e "$ANALYSIS_MARKDOWN" ]] || \
  die "analyzer recovery outputs already exist"

command -v sacct >/dev/null 2>&1 || die "sacct is unavailable"
CURRENT_ACCOUNTING="$(
  sacct -j "$SLURM_JOB_ID" -X -n -P \
    -o "JobIDRaw,JobName%128,Partition,Account,QOS,ReqCPUS,ReqMem,ReqNodes,ReqTRES%256,State,ExitCode,Elapsed,Timelimit,SubmitLine%4096,WorkDir%1024,NodeList%128,Submit,Start,End"
)" || die "could not query analyzer-recovery accounting"
[[ -n "$CURRENT_ACCOUNTING" ]] || \
  die "analyzer-recovery accounting is empty"

"$PYTHON_BIN" "$HELPER" validate-job \
  --protocol "$PROTOCOL" \
  --submission "$SUBMISSION" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --current-accounting-row "$CURRENT_ACCOUNTING" \
  --paired-latency "$PAIRED_LATENCY" \
  --timing-submission "$TIMING_SUBMISSION" \
  --output-json "$ANALYSIS_JSON" \
  --output-markdown "$ANALYSIS_MARKDOWN"

# Analyzer-only recovery: this worker intentionally never imports a model
# configuration, loads a checkpoint, invokes a teacher, reads future RGB
# tensors, or calls either benchmark entrypoint.
"$PYTHON_BIN" "$ANALYZER" \
  --study-root "$STUDY_ROOT" \
  --paired-latency "$PAIRED_LATENCY" \
  --paired-recovery-submission "$TIMING_SUBMISSION" \
  --paired-analysis-recovery-submission "$SUBMISSION" \
  --paired-analysis-recovery-job-id "$SLURM_JOB_ID" \
  --output-json "$ANALYSIS_JSON" \
  --output-markdown "$ANALYSIS_MARKDOWN" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260729 \
  --confidence 0.95

[[ -s "$ANALYSIS_JSON" && ! -L "$ANALYSIS_JSON" ]] || \
  die "analyzer recovery did not create JSON evidence"
[[ -s "$ANALYSIS_MARKDOWN" && ! -L "$ANALYSIS_MARKDOWN" ]] || \
  die "analyzer recovery did not create Markdown evidence"
[[ "$(git -C "$CONTROLLER_ROOT" rev-parse HEAD)" == \
  "$CONTROLLER_COMMIT" ]] || die "analysis controller changed during recovery"
[[ -z "$(git -C "$CONTROLLER_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "analysis controller became dirty during recovery"
