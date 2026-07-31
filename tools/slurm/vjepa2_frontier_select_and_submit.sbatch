#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --no-requeue
#SBATCH --open-mode=append

# This is the only workflow component permitted to create lockbox-scoring
# jobs.  It does so only after the raw validation selection has been written
# exclusively and independently reproduced as confirmatory-eligible.

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

REPO_ROOT=""
STUDY_ROOT=""
TRAINING_COMMIT=""
EVALUATOR_COMMIT=""
SCIENTIFIC_REPO_ROOT=""
SCIENTIFIC_EVALUATOR_COMMIT=""
PYTHON_BIN=""
WAN_DIR_VALUE=""
VIDEOX_HOME_VALUE=""
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
ACCOUNT=""
QOS=""
LOG_DIR=""

while (($#)); do
  case "$1" in
    --repo-root) REPO_ROOT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --scientific-repo-root) SCIENTIFIC_REPO_ROOT="${2:?}"; shift 2 ;;
    --scientific-evaluator-commit) SCIENTIFIC_EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --quality-time) QUALITY_TIME="${2:?}"; shift 2 ;;
    --control-time) CONTROL_TIME="${2:?}"; shift 2 ;;
    --timing-time) TIMING_TIME="${2:?}"; shift 2 ;;
    --quality-cpus) QUALITY_CPUS="${2:?}"; shift 2 ;;
    --quality-mem) QUALITY_MEMORY="${2:?}"; shift 2 ;;
    --control-cpus) CONTROL_CPUS="${2:?}"; shift 2 ;;
    --control-mem) CONTROL_MEMORY="${2:?}"; shift 2 ;;
    --timing-cpus) TIMING_CPUS="${2:?}"; shift 2 ;;
    --timing-mem) TIMING_MEMORY="${2:?}"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -z "$SCIENTIFIC_REPO_ROOT" ]]; then
  SCIENTIFIC_REPO_ROOT="$REPO_ROOT"
fi
if [[ -z "$SCIENTIFIC_EVALUATOR_COMMIT" ]]; then
  SCIENTIFIC_EVALUATOR_COMMIT="$EVALUATOR_COMMIT"
fi
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
  die "selection gate requires a non-array Slurm allocation"
[[ "${SLURM_RESTART_COUNT:-0}" == "0" ]] || die "selection gate is non-requeueable"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$LOG_DIR" && ! -L "$LOG_DIR" ]] || die "Slurm log directory is invalid"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
[[ -d "$SCIENTIFIC_REPO_ROOT" && ! -L "$SCIENTIFIC_REPO_ROOT" ]] || \
  die "scientific evaluator repository is invalid"
[[ "$(cd "$SCIENTIFIC_REPO_ROOT" && pwd -P)" == "$SCIENTIFIC_REPO_ROOT" ]] || \
  die "scientific evaluator repository is not canonical"
[[ "$(git -C "$SCIENTIFIC_REPO_ROOT" rev-parse --show-toplevel)" == \
  "$SCIENTIFIC_REPO_ROOT" ]] || die "scientific evaluator is not a worktree root"
[[ "$(git -C "$SCIENTIFIC_REPO_ROOT" rev-parse HEAD)" == \
  "$SCIENTIFIC_EVALUATOR_COMMIT" ]] || \
  die "scientific evaluator worktree HEAD changed"
[[ -z "$(git -C "$SCIENTIFIC_REPO_ROOT" status \
  --porcelain --untracked-files=all)" ]] || \
  die "scientific evaluator worktree is dirty"
git -C "$SCIENTIFIC_REPO_ROOT" merge-base --is-ancestor \
  "$TRAINING_COMMIT" "$SCIENTIFIC_EVALUATOR_COMMIT" || \
  die "scientific evaluator is not a training-commit descendant"
git -C "$REPO_ROOT" merge-base --is-ancestor \
  "$SCIENTIFIC_EVALUATOR_COMMIT" "$EVALUATOR_COMMIT" || \
  die "controller does not descend from the scientific evaluator"
git -C "$REPO_ROOT" diff --quiet \
  "$SCIENTIFIC_EVALUATOR_COMMIT" "$EVALUATOR_COMMIT" -- \
  tools/vjepa2_nfe_frontier.py || \
  die "selection implementation differs between scientific and controller commits"
for scalar in \
  "$PARTITION" "$QUALITY_MEMORY" "$CONTROL_MEMORY" "$TIMING_MEMORY" \
  "$ACCOUNT" "$QOS"; do
  [[ "$scalar" != *[[:space:]]* ]] || die "scheduler scalar contains whitespace"
done
for integer in "$QUALITY_CPUS" "$CONTROL_CPUS" "$TIMING_CPUS"; do
  [[ "$integer" =~ ^[1-9][0-9]*$ ]] || die "CPU counts must be positive integers"
done
command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
unset PYTHONPATH

VPM_VALIDATION="$STUDY_ROOT/vpm_parameter_matched_video/frontier_quality/validation/update_1000"
J1_VALIDATION="$STUDY_ROOT/j1_joint_auxiliary_leads/frontier_quality/validation/update_1000"
REGISTRATION="$STUDY_ROOT/frontier_lockbox/registration.json"
SELECTION="$STUDY_ROOT/frontier_selection.json"
CONTINUATION="$STUDY_ROOT/frontier_continuation.json"
[[ -s "$VPM_VALIDATION/inventory.json" && -s "$J1_VALIDATION/inventory.json" ]] || \
  die "validation inventories are incomplete"
[[ -s "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "registered lockbox evidence is unavailable"
[[ ! -e "$SELECTION" ]] || die "fresh selection output already exists"
[[ ! -e "$CONTINUATION" ]] || die "fresh continuation record already exists"

shopt -s nullglob
VPM_ROWS=("$VPM_VALIDATION"/rank_*.jsonl)
J1_ROWS=("$J1_VALIDATION"/rank_*.jsonl)
shopt -u nullglob
[[ "${#VPM_ROWS[@]}" == "8" && "${#J1_ROWS[@]}" == "8" ]] || \
  die "validation must contain exactly eight rank files per arm"

"$PYTHON_BIN" "$SCIENTIFIC_REPO_ROOT/tools/vjepa2_nfe_frontier.py" select \
  --j1-rows "${J1_ROWS[@]}" \
  --vpm-rows "${VPM_ROWS[@]}" \
  --split validation \
  --expected-clips 64 \
  --bootstrap-samples 10000 \
  --confidence 0.95 \
  --seed 1234 \
  --lockbox-registration "$REGISTRATION" \
  --output "$SELECTION"

set +e
"$PYTHON_BIN" "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
  require-selection --selection "$SELECTION"
ELIGIBILITY_STATUS=$?
set -e
case "$ELIGIBILITY_STATUS" in
  0) ;;
  3)
    echo "No confirmatory validation candidate. Lockbox scoring was not submitted."
    exit 0
    ;;
  *) die "selection failed confirmatory reproduction" ;;
esac

QUALITY_SBATCH="$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_quality.sbatch"
CONFIRM_SBATCH="$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_confirm.sbatch"
TIMING_GATE_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_timing_gate.sbatch"
SCIENTIFIC_TIMING_SBATCH="$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_latency.sbatch"
for path in \
  "$QUALITY_SBATCH" "$CONFIRM_SBATCH" "$TIMING_GATE_SBATCH" \
  "$SCIENTIFIC_TIMING_SBATCH"; do
  [[ -x "$path" ]] || die "workflow entrypoint is not executable: $path"
done

normalize_job_id() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
    die "Slurm returned an invalid job ID: $value"
  printf '%s\n' "${value%%[_;]*}"
}

SCHEDULER_OPTIONAL=()
[[ -n "$ACCOUNT" ]] && SCHEDULER_OPTIONAL+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SCHEDULER_OPTIONAL+=(--qos="$QOS")
COMMON_JOB_ARGS=(
  --repo-root "$SCIENTIFIC_REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
)

# One atomic array submission creates two independent one-node/eight-B200
# allocations.  This happens only after the eligibility check above.
LOCKBOX_JOB_ID="$(
  sbatch \
    --parsable \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=8 \
    --cpus-per-task="$QUALITY_CPUS" \
    --mem="$QUALITY_MEMORY" \
    --time="$QUALITY_TIME" \
    --partition="$PARTITION" \
    --array=0-1 \
    --dependency="afterok:$SLURM_JOB_ID" \
    --no-requeue \
    --open-mode=append \
    --export=ALL \
    --job-name="vjepa2-frontier-lockbox" \
    --output="$LOG_DIR/%x-%A_%a.out" \
    --error="$LOG_DIR/%x-%A_%a.err" \
    "${SCHEDULER_OPTIONAL[@]}" \
    "$QUALITY_SBATCH" \
    "${COMMON_JOB_ARGS[@]}" \
    --arm-from-array \
    --split lockbox \
    --selection "$SELECTION"
)" || die "Slurm rejected lockbox quality array"
LOCKBOX_DEPENDENCY="$(normalize_job_id "$LOCKBOX_JOB_ID")"

CONFIRM_JOB_ID="$(
  sbatch \
    --parsable \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --cpus-per-task="$CONTROL_CPUS" \
    --mem="$CONTROL_MEMORY" \
    --time="$CONTROL_TIME" \
    --partition="$PARTITION" \
    --dependency="afterok:$LOCKBOX_DEPENDENCY" \
    --no-requeue \
    --open-mode=append \
    --export=ALL \
    --job-name="vjepa2-frontier-confirm" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${SCHEDULER_OPTIONAL[@]}" \
    "$CONFIRM_SBATCH" \
    --repo-root "$SCIENTIFIC_REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT" \
    --python "$PYTHON_BIN"
)" || die "Slurm rejected lockbox confirmation job"
CONFIRM_DEPENDENCY="$(normalize_job_id "$CONFIRM_JOB_ID")"

TIMING_JOB_ID="$(
  sbatch \
    --parsable \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --cpus-per-task="$TIMING_CPUS" \
    --mem="$TIMING_MEMORY" \
    --time="$TIMING_TIME" \
    --partition="$PARTITION" \
    --dependency="afterok:$CONFIRM_DEPENDENCY" \
    --no-requeue \
    --open-mode=append \
    --export=ALL \
    --job-name="vjepa2-frontier-outcome" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${SCHEDULER_OPTIONAL[@]}" \
    "$TIMING_GATE_SBATCH" \
    --repo-root "$REPO_ROOT" \
    --controller-commit "$EVALUATOR_COMMIT" \
    --scientific-repo-root "$SCIENTIFIC_REPO_ROOT" \
    --scientific-evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --python "$PYTHON_BIN" \
    --wan-dir "$WAN_DIR_VALUE" \
    --videox-home "$VIDEOX_HOME_VALUE"
)" || die "Slurm rejected quality-outcome/conditional-timing job"
normalize_job_id "$TIMING_JOB_ID" >/dev/null

"$PYTHON_BIN" - \
  "$CONTINUATION" "$SELECTION" "$SLURM_JOB_ID" \
  "$LOCKBOX_JOB_ID" "$CONFIRM_JOB_ID" "$TIMING_JOB_ID" \
  "$REPO_ROOT" "$EVALUATOR_COMMIT" "$SCIENTIFIC_REPO_ROOT" \
  "$SCIENTIFIC_EVALUATOR_COMMIT" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

(
    output,
    selection_path,
    selection_job,
    lockbox_job,
    confirmation_job,
    timing_job,
    controller_repo_root,
    controller_commit,
    scientific_repo_root,
    scientific_evaluator_commit,
) = sys.argv[1:]
selection_bytes = Path(selection_path).read_bytes()
selection = json.loads(selection_bytes)
payload = {
    "kind": "vjepa2_nfe_frontier_slurm_continuation",
    "schema_version": 1,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection": {
        "path": str(Path(selection_path).resolve()),
        "sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "identity_sha256": selection["identity_sha256"],
        "confirmatory_eligible": selection["confirmatory_eligible"],
    },
    "selection_gate_job_id": selection_job,
    "controller_repo_root": controller_repo_root,
    "controller_git_commit": controller_commit,
    "scientific_evaluator_repo_root": scientific_repo_root,
    "scientific_evaluator_git_commit": scientific_evaluator_commit,
    "lockbox_quality_array_job_id": lockbox_job,
    "lockbox_quality_dependency": f"afterok:{selection_job}",
    "confirmation_job_id": confirmation_job,
    "confirmation_dependency": f"afterok:{lockbox_job.split(';', 1)[0].split('_', 1)[0]}",
    "timing_and_finalization_job_id": timing_job,
    "timing_dependency": f"afterok:{confirmation_job.split(';', 1)[0].split('_', 1)[0]}",
    "paired_timing_requires_held_out_quality_gate_pass": True,
    "quality_failure_writes_negative_final_report_without_timing": True,
    "outcome_uses_study_local_exclusive_lock": True,
    "negative_report_publication_is_atomic_no_replace": True,
    "lockbox_jobs_created_only_after_confirmatory_eligibility": True,
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Confirmatory selection passed; downstream afterok chain submitted."
echo "Lockbox quality array: $LOCKBOX_JOB_ID"
echo "Confirmation: $CONFIRM_JOB_ID"
echo "Quality outcome / conditional timing / finalization: $TIMING_JOB_ID"
