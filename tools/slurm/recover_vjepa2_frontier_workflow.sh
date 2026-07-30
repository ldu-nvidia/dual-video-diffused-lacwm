#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
STUDY_ROOT="${VJEPA_FRONTIER_STUDY_ROOT:-$LACWM_BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3}"
TRAINING_COMMIT="${VJEPA_FRONTIER_TRAINING_COMMIT:-9cf8e6922f35a5d6645e3128545953723bf54da2}"
CONTROLLER_COMMIT=""
SCIENTIFIC_REPO_ROOT="${VJEPA_FRONTIER_SCIENTIFIC_REPO_ROOT:-$LACWM_BASE/src/vjepa2-frontier-evaluator-6b659a9}"
SCIENTIFIC_COMMIT="${VJEPA_FRONTIER_SCIENTIFIC_COMMIT:-87f3f8f969e160c86f5a8149b3ee8d0b32758f99}"

FINAL_JOB_ID="${VJEPA_FINAL_U1000_JOB_ID:-481132}"
CACHE_JOB_ID="${VJEPA_FRONTIER_CACHE_JOB_ID:-481556}"
FAILED_GATE_JOB_ID="${VJEPA_FRONTIER_FAILED_GATE_JOB_ID:-481577}"
CANCELLED_VPM_JOB_ID="${VJEPA_FRONTIER_CANCELLED_VPM_JOB_ID:-481578}"
CANCELLED_J1_JOB_ID="${VJEPA_FRONTIER_CANCELLED_J1_JOB_ID:-481579}"
CANCELLED_SELECTION_JOB_ID="${VJEPA_FRONTIER_CANCELLED_SELECTION_JOB_ID:-481580}"

PRIOR_SUBMISSION="${VJEPA_FRONTIER_PRIOR_SUBMISSION:-}"
RECOVERY_ROOT="${VJEPA_FRONTIER_RECOVERY_ROOT:-}"
PYTHON_BIN="${LACWM_PYTHON:-$LACWM_BASE/envs/lacwm-b200-py310/bin/python}"
WAN_DIR_VALUE="${WAN_DIR:-$LACWM_BASE/wan_fun_1.3b_control}"
VIDEOX_HOME_VALUE="${VIDEOX_HOME:-$LACWM_BASE/VideoX-Fun-1d6d9c3}"

PARTITION="batch"
QUALITY_TIME="04:00:00"
CONTROL_TIME="01:00:00"
TIMING_TIME="04:00:00"
QUALITY_CPUS="160"
QUALITY_MEMORY="1000G"
CONTROL_CPUS="16"
CONTROL_MEMORY="64G"
TIMING_CPUS="32"
TIMING_MEMORY="256G"
ACCOUNT="coreai_chef_posttrain"
QOS="normal"
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/recover_vjepa2_frontier_workflow.sh [options]

Recover the single failed frontier DAG whose immutable lockbox cache completed
as job 481556. The recovery validates and reuses that cache, but never submits,
changes, removes, or replaces it.

Pinned predecessor:
  final u1000 array: 481132 (five exact COMPLETED/0:0 tasks)
  completed cache:  481556
  failed gate:      481577
  cancelled jobs:  481578 (VPM), 481579 (J1), 481580 (selection)
  scientific code: 87f3f8f969e160c86f5a8149b3ee8d0b32758f99

Fresh recovery DAG:
  controller artifact/cache gate
      -> frozen-87f VPM validation
      -> frozen-87f J1 validation
      -> controller selection and conditional continuation

Options:
  --study-root PATH
  --training-commit SHA
  --controller-commit SHA
  --scientific-repo-root PATH
  --scientific-commit SHA
  --prior-submission PATH
  --recovery-root PATH
  --final-u1000-job-id ID
  --cache-job-id ID
  --failed-gate-job-id ID
  --cancelled-vpm-job-id ID
  --cancelled-j1-job-id ID
  --cancelled-selection-job-id ID
  --python PATH
  --wan-dir PATH
  --videox-home PATH
  --partition NAME
  --quality-time HH:MM:SS
  --control-time HH:MM:SS
  --timing-time HH:MM:SS
  --account NAME
  --qos NAME
  --execute
  -h, --help

Without --execute, all checks are read-only. Scientific outputs and the
recovery root must be absent. The original workflow root is never modified.
EOF
}

while (($#)); do
  case "$1" in
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --controller-commit) CONTROLLER_COMMIT="${2:?}"; shift 2 ;;
    --scientific-repo-root) SCIENTIFIC_REPO_ROOT="${2:?}"; shift 2 ;;
    --scientific-commit) SCIENTIFIC_COMMIT="${2:?}"; shift 2 ;;
    --prior-submission) PRIOR_SUBMISSION="${2:?}"; shift 2 ;;
    --recovery-root) RECOVERY_ROOT="${2:?}"; shift 2 ;;
    --final-u1000-job-id) FINAL_JOB_ID="${2:?}"; shift 2 ;;
    --cache-job-id) CACHE_JOB_ID="${2:?}"; shift 2 ;;
    --failed-gate-job-id) FAILED_GATE_JOB_ID="${2:?}"; shift 2 ;;
    --cancelled-vpm-job-id) CANCELLED_VPM_JOB_ID="${2:?}"; shift 2 ;;
    --cancelled-j1-job-id) CANCELLED_J1_JOB_ID="${2:?}"; shift 2 ;;
    --cancelled-selection-job-id) CANCELLED_SELECTION_JOB_ID="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --quality-time) QUALITY_TIME="${2:?}"; shift 2 ;;
    --control-time) CONTROL_TIME="${2:?}"; shift 2 ;;
    --timing-time) TIMING_TIME="${2:?}"; shift 2 ;;
    --account) ACCOUNT="${2:?}"; shift 2 ;;
    --qos) QOS="${2:?}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ -z "$CONTROLLER_COMMIT" ]]; then
  CONTROLLER_COMMIT="$ACTUAL_COMMIT"
fi
for commit in "$TRAINING_COMMIT" "$CONTROLLER_COMMIT" "$SCIENTIFIC_COMMIT"; do
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit: $commit"
done
for job_id in \
  "$FINAL_JOB_ID" "$CACHE_JOB_ID" "$FAILED_GATE_JOB_ID" \
  "$CANCELLED_VPM_JOB_ID" "$CANCELLED_J1_JOB_ID" \
  "$CANCELLED_SELECTION_JOB_ID"; do
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || die "invalid Slurm job ID: $job_id"
done
EXPECTED_RECOVERY_ROOT="$STUDY_ROOT/_frontier_slurm_recovery_${CACHE_JOB_ID}_${FAILED_GATE_JOB_ID}"
if [[ -z "$RECOVERY_ROOT" ]]; then
  RECOVERY_ROOT="$EXPECTED_RECOVERY_ROOT"
fi
if [[ -z "$PRIOR_SUBMISSION" ]]; then
  PRIOR_SUBMISSION="$STUDY_ROOT/_frontier_slurm/submission.json"
fi
for scalar in \
  "$PARTITION" "$QUALITY_MEMORY" "$CONTROL_MEMORY" "$TIMING_MEMORY" \
  "$ACCOUNT" "$QOS"; do
  [[ "$scalar" != *[[:space:]]* ]] || die "scheduler scalar contains whitespace"
done
for value in "$QUALITY_TIME" "$CONTROL_TIME" "$TIMING_TIME"; do
  [[ "$value" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || \
    die "time limits must use HH:MM:SS"
done
for path in \
  "$REPO_ROOT" "$STUDY_ROOT" "$SCIENTIFIC_REPO_ROOT" \
  "$PRIOR_SUBMISSION" "$RECOVERY_ROOT"; do
  [[ "$path" == /* ]] || die "all recovery paths must be absolute"
done
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "invalid controller repository"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || \
  die "launcher must run from a Git worktree root"
[[ "$ACTUAL_COMMIT" == "$CONTROLLER_COMMIT" ]] || \
  die "controller worktree differs from --controller-commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "controller worktree must be clean"
[[ -d "$SCIENTIFIC_REPO_ROOT" && ! -L "$SCIENTIFIC_REPO_ROOT" ]] || \
  die "scientific evaluator repository is invalid"
[[ "$(cd "$SCIENTIFIC_REPO_ROOT" && pwd -P)" == \
  "$SCIENTIFIC_REPO_ROOT" ]] || die "scientific evaluator path is not canonical"
[[ "$(git -C "$SCIENTIFIC_REPO_ROOT" rev-parse --show-toplevel)" == \
  "$SCIENTIFIC_REPO_ROOT" ]] || die "scientific evaluator is not a worktree root"
[[ "$(git -C "$SCIENTIFIC_REPO_ROOT" rev-parse HEAD)" == \
  "$SCIENTIFIC_COMMIT" ]] || die "scientific evaluator worktree HEAD changed"
[[ -z "$(git -C "$SCIENTIFIC_REPO_ROOT" status \
  --porcelain --untracked-files=all)" ]] || \
  die "scientific evaluator worktree must be clean"
[[ "$SCIENTIFIC_REPO_ROOT" != "$REPO_ROOT" ]] || \
  die "controller and scientific evaluator require distinct worktrees"
git -C "$REPO_ROOT" merge-base --is-ancestor \
  "$SCIENTIFIC_COMMIT" "$CONTROLLER_COMMIT" || \
  die "controller is not a scientific-evaluator descendant"
git -C "$REPO_ROOT" diff --quiet \
  "$SCIENTIFIC_COMMIT" "$CONTROLLER_COMMIT" -- \
  tools/vjepa2_nfe_frontier.py || \
  die "controller and scientific selection implementations differ"

[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is invalid"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "study root is not canonical"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$WAN_DIR_VALUE" ]] || die "Wan directory is unavailable"
[[ -d "$VIDEOX_HOME_VALUE/.git" ]] || die "VideoX checkout is unavailable"
[[ -f "$PRIOR_SUBMISSION" && ! -L "$PRIOR_SUBMISSION" ]] || \
  die "predecessor submission is unavailable"
[[ "$RECOVERY_ROOT" == "$EXPECTED_RECOVERY_ROOT" ]] || \
  die "recovery root must equal the unique predecessor-bound path: $EXPECTED_RECOVERY_ROOT"
[[ ! -e "$RECOVERY_ROOT" ]] || die "fresh recovery root already exists"

WORKFLOW_HELPER="$REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py"
FINAL_GATE_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_final_gate.sbatch"
SELECTION_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_select_and_submit.sbatch"
QUALITY_SBATCH="$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_quality.sbatch"
for path in \
  "$WORKFLOW_HELPER" "$FINAL_GATE_SBATCH" "$SELECTION_SBATCH" \
  "$QUALITY_SBATCH"; do
  [[ -x "$path" ]] || die "required workflow entrypoint is not executable: $path"
done

PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - \
  "$REPO_ROOT" "$TRAINING_COMMIT" "$CONTROLLER_COMMIT" <<'PY'
import json
import sys
from pathlib import Path
from tools.vjepa2_nfe_frontier import git_inference_compatibility
print(json.dumps(git_inference_compatibility(
    Path(sys.argv[1]), training_commit=sys.argv[2], tool_commit=sys.argv[3]
), sort_keys=True))
PY

"$PYTHON_BIN" "$WORKFLOW_HELPER" check-submission \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT" \
  --final-job-id "$FINAL_JOB_ID"
FINAL_MODE="$(
  "$PYTHON_BIN" "$WORKFLOW_HELPER" classify-final-job \
    --final-job-id "$FINAL_JOB_ID"
)" || die "could not classify final update-1000 job"
[[ "$FINAL_MODE" == "terminal_success" ]] || \
  die "recovery requires the final update-1000 array to be terminal-success"
"$PYTHON_BIN" "$WORKFLOW_HELPER" check-final \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT"

RECOVERY_CHECK_ARGS=(
  validate-completed-recovery
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --controller-repo-root "$REPO_ROOT"
  --controller-commit "$CONTROLLER_COMMIT"
  --scientific-repo-root "$SCIENTIFIC_REPO_ROOT"
  --scientific-commit "$SCIENTIFIC_COMMIT"
  --prior-submission "$PRIOR_SUBMISSION"
  --cache-job-id "$CACHE_JOB_ID"
  --failed-gate-job-id "$FAILED_GATE_JOB_ID"
  --cancelled-vpm-job-id "$CANCELLED_VPM_JOB_ID"
  --cancelled-j1-job-id "$CANCELLED_J1_JOB_ID"
  --cancelled-selection-job-id "$CANCELLED_SELECTION_JOB_ID"
  --final-job-id "$FINAL_JOB_ID"
  --partition "$PARTITION"
  --account "$ACCOUNT"
  --qos "$QOS"
)
RECOVERY_EVIDENCE="$(
  "$PYTHON_BIN" "$WORKFLOW_HELPER" "${RECOVERY_CHECK_ARGS[@]}"
)" || die "completed frontier DAG failed recovery validation"

echo "V-JEPA frontier recovery preflight passed."
echo "Study: $STUDY_ROOT"
echo "Controller: $CONTROLLER_COMMIT at $REPO_ROOT"
echo "Scientific evaluator: $SCIENTIFIC_COMMIT at $SCIENTIFIC_REPO_ROOT"
echo "Completed cache reused: $CACHE_JOB_ID (no cache job will be submitted)"
echo "Failed predecessor gate: $FAILED_GATE_JOB_ID"
echo "Cancelled predecessor jobs: $CANCELLED_VPM_JOB_ID, $CANCELLED_J1_JOB_ID, $CANCELLED_SELECTION_JOB_ID"
echo "Fresh recovery root: $RECOVERY_ROOT"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --controller-commit '$CONTROLLER_COMMIT' --execute."
  exit 0
fi

for command in sbatch sacct squeue flock; do
  command -v "$command" >/dev/null 2>&1 || die "$command is unavailable"
done
mkdir "$RECOVERY_ROOT"
LOG_DIR="$RECOVERY_ROOT/logs"
mkdir "$LOG_DIR"
exec 9>"$RECOVERY_ROOT/workflow.lock"
flock -n 9 || die "another process owns the recovery workflow lock"

RECOVERY_RECHECK="$(
  "$PYTHON_BIN" "$WORKFLOW_HELPER" "${RECOVERY_CHECK_ARGS[@]}"
)" || die "recovery evidence changed before submission"
[[ "$RECOVERY_RECHECK" == "$RECOVERY_EVIDENCE" ]] || \
  die "recovery evidence changed before submission"
[[ "$(
  "$PYTHON_BIN" "$WORKFLOW_HELPER" classify-final-job \
    --final-job-id "$FINAL_JOB_ID"
)" == "terminal_success" ]] || die "final update-1000 accounting changed"
"$PYTHON_BIN" "$WORKFLOW_HELPER" check-final \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT" >/dev/null

EVIDENCE_RECORD="$RECOVERY_ROOT/predecessor_evidence.json"
"$PYTHON_BIN" - "$EVIDENCE_RECORD" "$RECOVERY_EVIDENCE" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(sys.argv[2])
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

normalize_job_id() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
    die "Slurm returned an invalid job ID: $value"
  printf '%s\n' "${value%%[_;]*}"
}
record_accepted_job() {
  local label="$1"
  local job_id="$2"
  local dependency="$3"
  local executor_repo="$4"
  local executor_commit="$5"
  shift 5
  local output="$RECOVERY_ROOT/accepted_${label}.json"
  "$PYTHON_BIN" - \
    "$output" "$label" "$job_id" "$dependency" \
    "$executor_repo" "$executor_commit" "$@" <<'PY'
import datetime
import hashlib
import json
import shlex
import sys
from pathlib import Path
output, label, job_id, dependency, executor_repo, executor_commit = sys.argv[1:7]
submit_tokens = sys.argv[7:]
if not submit_tokens or submit_tokens[0] != "sbatch":
    raise SystemExit("accepted-job receipt requires the exact sbatch token vector")
submit_line = shlex.join(submit_tokens)
payload = {
    "kind": "vjepa2_nfe_frontier_recovery_accepted_job",
    "schema_version": 1,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "label": label,
    "job_id": job_id,
    "dependency": None if dependency == "none" else dependency,
    "executor_repo_root": executor_repo,
    "executor_git_commit": executor_commit,
    "cache_job": False,
    "submit_line_tokens": submit_tokens,
    "submit_line_sha256": hashlib.sha256(
        submit_line.encode("utf-8")
    ).hexdigest(),
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}
SCHEDULER_OPTIONAL=(--account="$ACCOUNT" --qos="$QOS")
RECOVERY_GATE_ARGS=(
  --recovery-prior-submission "$PRIOR_SUBMISSION"
  --recovery-cache-job-id "$CACHE_JOB_ID"
  --recovery-failed-gate-job-id "$FAILED_GATE_JOB_ID"
  --recovery-cancelled-vpm-job-id "$CANCELLED_VPM_JOB_ID"
  --recovery-cancelled-j1-job-id "$CANCELLED_J1_JOB_ID"
  --recovery-cancelled-selection-job-id "$CANCELLED_SELECTION_JOB_ID"
  --recovery-final-job-id "$FINAL_JOB_ID"
  --recovery-partition "$PARTITION"
  --recovery-account "$ACCOUNT"
  --recovery-qos "$QOS"
)

GATE_SUBMIT_ARGS=(
  --parsable
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=1
  --cpus-per-task="$CONTROL_CPUS"
  --mem="$CONTROL_MEMORY"
  --time="$CONTROL_TIME"
  --partition="$PARTITION"
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="vjepa2-frontier-recovery-gate"
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.err"
  "${SCHEDULER_OPTIONAL[@]}"
  "$FINAL_GATE_SBATCH"
  --repo-root "$REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$CONTROLLER_COMMIT"
  --scientific-repo-root "$SCIENTIFIC_REPO_ROOT"
  --scientific-evaluator-commit "$SCIENTIFIC_COMMIT"
  --python "$PYTHON_BIN"
  "${RECOVERY_GATE_ARGS[@]}"
)
GATE_JOB_ID="$(sbatch "${GATE_SUBMIT_ARGS[@]}")" || \
  die "Slurm rejected the recovery gate"
GATE_DEPENDENCY="$(normalize_job_id "$GATE_JOB_ID")"
record_accepted_job \
  gate "$GATE_JOB_ID" none "$REPO_ROOT" "$CONTROLLER_COMMIT" \
  sbatch "${GATE_SUBMIT_ARGS[@]}"

QUALITY_JOB_ARGS=(
  --repo-root "$SCIENTIFIC_REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$SCIENTIFIC_COMMIT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --split validation
)
submit_validation() {
  local arm="$1"
  local label="validation_${arm,,}"
  local submit_args=(
    --parsable
    --nodes=1
    --ntasks=1
    --ntasks-per-node=1
    --gpus-per-node=8
    --cpus-per-task="$QUALITY_CPUS"
    --mem="$QUALITY_MEMORY"
    --time="$QUALITY_TIME"
    --partition="$PARTITION"
    --dependency="afterok:$GATE_DEPENDENCY"
    --no-requeue
    --open-mode=append
    --export=ALL
    --job-name="vjepa2-frontier-recovery-val-${arm,,}"
    --output="$LOG_DIR/%x-%j.out"
    --error="$LOG_DIR/%x-%j.err"
    "${SCHEDULER_OPTIONAL[@]}"
    "$QUALITY_SBATCH"
    "${QUALITY_JOB_ARGS[@]}"
    --arm "$arm"
  )
  local job_id
  job_id="$(sbatch "${submit_args[@]}")" || return
  normalize_job_id "$job_id" >/dev/null || return
  record_accepted_job \
    "$label" "$job_id" "afterok:$GATE_DEPENDENCY" \
    "$SCIENTIFIC_REPO_ROOT" "$SCIENTIFIC_COMMIT" \
    sbatch "${submit_args[@]}" || return
  printf '%s\n' "$job_id"
}
VPM_VALIDATION_JOB_ID="$(submit_validation VPM)" || \
  die "Slurm rejected recovery VPM validation"
J1_VALIDATION_JOB_ID="$(submit_validation J1)" || \
  die "Slurm rejected recovery J1 validation"
VPM_DEPENDENCY="$(normalize_job_id "$VPM_VALIDATION_JOB_ID")"
J1_DEPENDENCY="$(normalize_job_id "$J1_VALIDATION_JOB_ID")"

SELECTION_SUBMIT_ARGS=(
  --parsable
  --nodes=1
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node=1
  --cpus-per-task="$CONTROL_CPUS"
  --mem="$CONTROL_MEMORY"
  --time="$CONTROL_TIME"
  --partition="$PARTITION"
  --dependency="afterok:$VPM_DEPENDENCY:$J1_DEPENDENCY"
  --no-requeue
  --open-mode=append
  --export=ALL
  --job-name="vjepa2-frontier-recovery-select"
  --output="$LOG_DIR/%x-%j.out"
  --error="$LOG_DIR/%x-%j.err"
  "${SCHEDULER_OPTIONAL[@]}"
  "$SELECTION_SBATCH"
  --repo-root "$REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$CONTROLLER_COMMIT"
  --scientific-repo-root "$SCIENTIFIC_REPO_ROOT"
  --scientific-evaluator-commit "$SCIENTIFIC_COMMIT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --partition "$PARTITION"
  --quality-time "$QUALITY_TIME"
  --control-time "$CONTROL_TIME"
  --timing-time "$TIMING_TIME"
  --quality-cpus "$QUALITY_CPUS"
  --quality-mem "$QUALITY_MEMORY"
  --control-cpus "$CONTROL_CPUS"
  --control-mem "$CONTROL_MEMORY"
  --timing-cpus "$TIMING_CPUS"
  --timing-mem "$TIMING_MEMORY"
  --account "$ACCOUNT"
  --qos "$QOS"
  --log-dir "$LOG_DIR"
)
SELECTION_JOB_ID="$(sbatch "${SELECTION_SUBMIT_ARGS[@]}")" || \
  die "Slurm rejected the recovery selection gate"
normalize_job_id "$SELECTION_JOB_ID" >/dev/null
record_accepted_job \
  selection "$SELECTION_JOB_ID" \
  "afterok:$VPM_DEPENDENCY:$J1_DEPENDENCY" \
  "$REPO_ROOT" "$CONTROLLER_COMMIT" \
  sbatch "${SELECTION_SUBMIT_ARGS[@]}"

SUBMISSION_RECORD="$RECOVERY_ROOT/submission.json"
"$PYTHON_BIN" - \
  "$SUBMISSION_RECORD" "$EVIDENCE_RECORD" "$STUDY_ROOT" \
  "$TRAINING_COMMIT" "$REPO_ROOT" "$CONTROLLER_COMMIT" \
  "$SCIENTIFIC_REPO_ROOT" "$SCIENTIFIC_COMMIT" "$CACHE_JOB_ID" \
  "$GATE_JOB_ID" "$VPM_VALIDATION_JOB_ID" "$J1_VALIDATION_JOB_ID" \
  "$SELECTION_JOB_ID" "$RECOVERY_ROOT" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path
(
    output,
    evidence_path,
    study_root,
    training_commit,
    controller_repo,
    controller_commit,
    scientific_repo,
    scientific_commit,
    cache_job,
    gate_job,
    vpm_job,
    j1_job,
    selection_job,
    recovery_root,
) = sys.argv[1:]
evidence = Path(evidence_path)
evidence_bytes = evidence.read_bytes()
accepted = {}
for label in ("gate", "validation_vpm", "validation_j1", "selection"):
    path = Path(recovery_root) / f"accepted_{label}.json"
    raw = path.read_bytes()
    accepted[label] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "record": json.loads(raw),
    }
payload = {
    "kind": "vjepa2_nfe_frontier_slurm_recovery_submission",
    "schema_version": 1,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "study_root": study_root,
    "training_git_commit": training_commit,
    "controller_repo_root": controller_repo,
    "controller_git_commit": controller_commit,
    "scientific_evaluator_repo_root": scientific_repo,
    "scientific_evaluator_git_commit": scientific_commit,
    "predecessor_evidence": {
        "path": str(evidence.resolve()),
        "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "bytes": len(evidence_bytes),
    },
    "completed_cache_job_id": cache_job,
    "completed_cache_reused": True,
    "cache_job_submitted_by_recovery": False,
    "recovery_gate_job_id": gate_job,
    "validation_job_ids": {"VPM": vpm_job, "J1": j1_job},
    "selection_gate_job_id": selection_job,
    "accepted_job_receipts": accepted,
    "dependencies": {
        "recovery_gate": None,
        "validation": f"afterok:{gate_job.split(';', 1)[0].split('_', 1)[0]}",
        "selection": (
            f"afterok:{vpm_job.split(';', 1)[0].split('_', 1)[0]}:"
            f"{j1_job.split(';', 1)[0].split('_', 1)[0]}"
        ),
    },
    "lockbox_jobs_submitted_at_recovery_submission": False,
    "lockbox_submission_requires_confirmatory_eligible": True,
    "original_workflow_artifacts_modified": False,
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Frontier recovery DAG submitted."
echo "Completed cache reused (no cache submission): $CACHE_JOB_ID"
echo "Recovery gate: $GATE_JOB_ID"
echo "VPM validation: $VPM_VALIDATION_JOB_ID"
echo "J1 validation: $J1_VALIDATION_JOB_ID"
echo "Selection/conditional continuation: $SELECTION_JOB_ID"
echo "Recovery receipt: $SUBMISSION_RECORD"
