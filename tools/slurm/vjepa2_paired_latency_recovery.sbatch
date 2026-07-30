#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
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
SCIENTIFIC_ROOT=""
SCIENTIFIC_COMMIT=""
STUDY_ROOT=""
PROTOCOL=""
RECOVERY_SUBMISSION=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
VIDEOX_COMMIT="1d6d9c3e1540968466937129fef4b288041e06de"

while (($#)); do
  case "$1" in
    --controller-root) CONTROLLER_ROOT="${2:?}"; shift 2 ;;
    --controller-commit) CONTROLLER_COMMIT="${2:?}"; shift 2 ;;
    --scientific-root) SCIENTIFIC_ROOT="${2:?}"; shift 2 ;;
    --scientific-commit) SCIENTIFIC_COMMIT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --protocol) PROTOCOL="${2:?}"; shift 2 ;;
    --recovery-submission) RECOVERY_SUBMISSION="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in \
  "$CONTROLLER_ROOT" "$CONTROLLER_COMMIT" "$SCIENTIFIC_ROOT" \
  "$SCIENTIFIC_COMMIT" "$STUDY_ROOT" "$PROTOCOL" "$RECOVERY_SUBMISSION" \
  "$PYTHON_BIN" "$WAN_DIR_VALUE" "$VIDEOX_HOME_VALUE"; do
  [[ -n "$value" ]] || die "all recovery arguments are required"
done
for path in \
  "$CONTROLLER_ROOT" "$SCIENTIFIC_ROOT" "$STUDY_ROOT" "$PROTOCOL" \
  "$RECOVERY_SUBMISSION" "$PYTHON_BIN" "$WAN_DIR_VALUE" \
  "$VIDEOX_HOME_VALUE"; do
  [[ "$path" == /* ]] || die "all recovery paths must be absolute"
done
for commit in "$CONTROLLER_COMMIT" "$SCIENTIFIC_COMMIT"; do
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "commits must be full lowercase SHA-1s"
done

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "paired recovery requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || \
  die "paired recovery is non-requeueable"
[[ "${SLURM_JOB_NUM_NODES:-1}" == "1" ]] || \
  die "paired recovery requires exactly one node"
[[ "${SLURM_CPUS_PER_TASK:-32}" == "32" ]] || \
  die "paired recovery requires exactly 32 CPUs per task"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$CONTROLLER_ROOT" && ! -L "$CONTROLLER_ROOT" ]] || \
  die "controller worktree is invalid"
[[ -d "$SCIENTIFIC_ROOT" && ! -L "$SCIENTIFIC_ROOT" ]] || \
  die "scientific worktree is invalid"
[[ "$CONTROLLER_ROOT" != "$SCIENTIFIC_ROOT" ]] || \
  die "controller and scientific worktrees must differ"
[[ "$(git -C "$CONTROLLER_ROOT" rev-parse HEAD)" == "$CONTROLLER_COMMIT" ]] || \
  die "controller worktree HEAD changed"
[[ "$(git -C "$SCIENTIFIC_ROOT" rev-parse HEAD)" == "$SCIENTIFIC_COMMIT" ]] || \
  die "scientific worktree HEAD changed"
[[ -z "$(git -C "$CONTROLLER_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "controller worktree is dirty"
[[ -z "$(git -C "$SCIENTIFIC_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "scientific worktree is dirty"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is invalid"
[[ -f "$PROTOCOL" && ! -L "$PROTOCOL" ]] || die "recovery protocol is invalid"
[[ -f "$RECOVERY_SUBMISSION" && ! -L "$RECOVERY_SUBMISSION" ]] || \
  die "recovery submission is invalid"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime assets are unavailable"
[[ "$(git -C "$VIDEOX_HOME_VALUE" rev-parse HEAD)" == "$VIDEOX_COMMIT" ]] || \
  die "VideoX checkout HEAD changed before model loading"
[[ -z "$(git -C "$VIDEOX_HOME_VALUE" status --porcelain --untracked-files=all)" ]] || \
  die "VideoX checkout is dirty before model loading"

HELPER="$CONTROLLER_ROOT/tools/vjepa2_paired_recovery.py"
BENCHMARK="$CONTROLLER_ROOT/tools/benchmark_vjepa2_paired_latency.py"
ANALYZER="$CONTROLLER_ROOT/tools/analyze_vjepa2_controlled_study.py"
ACTIVATE="$SCIENTIFIC_ROOT/tools/env/activate_b200.sh"
VERIFY_RUNTIME="$SCIENTIFIC_ROOT/tools/env/verify_b200_runtime.py"
STUDY_HELPER="$SCIENTIFIC_ROOT/tools/vjepa2_controlled_study.py"
for path in \
  "$HELPER" "$BENCHMARK" "$ANALYZER" "$ACTIVATE" "$VERIFY_RUNTIME" \
  "$STUDY_HELPER"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required recovery file is invalid: $path"
done

OUTPUT="$(
  "$PYTHON_BIN" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["paths"]["paired_latency"])
PY
)"
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
RUNTIME_EVIDENCE="$(
  "$PYTHON_BIN" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["paths"]["runtime"])
PY
)"

command -v sacct >/dev/null 2>&1 || die "sacct is unavailable"
CURRENT_ACCOUNTING="$(
  sacct -j "$SLURM_JOB_ID" -X -n -P \
    -o "JobIDRaw,JobName%128,Partition,Account,QOS,ReqCPUS,ReqMem,ReqNodes,ReqTRES%256,State,ExitCode,Elapsed,Timelimit,SubmitLine%4096,WorkDir%1024,NodeList%128,Submit,Start,End"
)" || die "could not query current paired-recovery accounting"
[[ -n "$CURRENT_ACCOUNTING" ]] || die "current recovery accounting is empty"

"$PYTHON_BIN" "$HELPER" validate-runtime \
  --protocol "$PROTOCOL" \
  --submission "$RECOVERY_SUBMISSION" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --output "$OUTPUT" \
  --controller-repo-root "$CONTROLLER_ROOT" \
  --controller-commit "$CONTROLLER_COMMIT" \
  --scientific-repo-root "$SCIENTIFIC_ROOT" \
  --scientific-commit "$SCIENTIFIC_COMMIT" \
  --current-accounting-row "$CURRENT_ACCOUNTING" \
  --slurm-node "${SLURMD_NODENAME:?}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:?}" \
  --runtime-evidence "$RUNTIME_EVIDENCE"

export WAN_DIR="$WAN_DIR_VALUE"
export VIDEOX_HOME="$VIDEOX_HOME_VALUE"
export LACWM_PYTHON="$PYTHON_BIN"
source "$ACTIVATE"
"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 1 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

ROBOT_WM_ORIGIN="$(
  cd "$SCIENTIFIC_ROOT"
  "$PYTHON_BIN" -c \
    'from pathlib import Path; import robot_wm; print(Path(robot_wm.__file__).resolve())'
)"
case "$ROBOT_WM_ORIGIN" in
  "$SCIENTIFIC_ROOT"/robot_wm/*) ;;
  *) die "Python resolved robot_wm outside the scientific repository: $ROBOT_WM_ORIGIN" ;;
esac

"$PYTHON_BIN" "$STUDY_HELPER" validate-study-inputs \
  --study-manifest "$STUDY_ROOT/study_manifest.json" \
  --expected-commit "$SCIENTIFIC_COMMIT" \
  --repo-root "$SCIENTIFIC_ROOT"

"$PYTHON_BIN" "$BENCHMARK" \
  --repo-root "$SCIENTIFIC_ROOT" \
  --expected-commit "$SCIENTIFIC_COMMIT" \
  --study-root "$STUDY_ROOT" \
  --submission-record "$STUDY_ROOT/slurm_submission.json" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda:0 \
  --output "$OUTPUT" \
  --recovery-protocol "$PROTOCOL" \
  --recovery-submission-record "$RECOVERY_SUBMISSION" \
  --recovery-runtime-record "$RUNTIME_EVIDENCE" \
  --controller-commit "$CONTROLLER_COMMIT"

[[ -s "$OUTPUT" && ! -L "$OUTPUT" ]] || \
  die "paired recovery did not produce its immutable evidence record"

[[ "$(git -C "$CONTROLLER_ROOT" rev-parse HEAD)" == "$CONTROLLER_COMMIT" ]] || \
  die "controller worktree changed during paired timing"
[[ "$(git -C "$SCIENTIFIC_ROOT" rev-parse HEAD)" == "$SCIENTIFIC_COMMIT" ]] || \
  die "scientific worktree changed during paired timing"
[[ -z "$(git -C "$CONTROLLER_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "controller worktree became dirty during paired timing"
[[ -z "$(git -C "$SCIENTIFIC_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "scientific worktree became dirty during paired timing"
[[ "$(git -C "$VIDEOX_HOME_VALUE" rev-parse HEAD)" == "$VIDEOX_COMMIT" ]] || \
  die "VideoX checkout changed during paired timing"
[[ -z "$(git -C "$VIDEOX_HOME_VALUE" status --porcelain --untracked-files=all)" ]] || \
  die "VideoX checkout became dirty during paired timing"
"$PYTHON_BIN" "$VERIFY_RUNTIME" \
  --expected-gpus 1 \
  --require-b200 \
  --wan-dir "$WAN_DIR_VALUE"

"$PYTHON_BIN" "$ANALYZER" \
  --study-root "$STUDY_ROOT" \
  --paired-latency "$OUTPUT" \
  --paired-recovery-submission "$RECOVERY_SUBMISSION" \
  --output-json "$ANALYSIS_JSON" \
  --output-markdown "$ANALYSIS_MARKDOWN" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260729 \
  --confidence 0.95

[[ -s "$ANALYSIS_JSON" && ! -L "$ANALYSIS_JSON" ]] || \
  die "paired recovery analyzer did not produce JSON evidence"
[[ -s "$ANALYSIS_MARKDOWN" && ! -L "$ANALYSIS_MARKDOWN" ]] || \
  die "paired recovery analyzer did not produce its Markdown report"
