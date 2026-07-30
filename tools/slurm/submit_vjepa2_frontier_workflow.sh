#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
STUDY_ROOT="${VJEPA_FRONTIER_STUDY_ROOT:-$LACWM_BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3}"
TRAINING_COMMIT="${VJEPA_FRONTIER_TRAINING_COMMIT:-9cf8e6922f35a5d6645e3128545953723bf54da2}"
EVALUATOR_COMMIT=""
FINAL_JOB_ID="${VJEPA_FINAL_U1000_JOB_ID:-481132}"
ADOPT_CACHE_JOB_ID=""

PYTHON_BIN="${LACWM_PYTHON:-$LACWM_BASE/envs/lacwm-b200-py310/bin/python}"
EXTRACTOR_PYTHON="${VJEPA_EXTRACTOR_PYTHON:-$LACWM_BASE/envs/vjepa2-extractor-py311/bin/python3.11}"
WAN_DIR_VALUE="${WAN_DIR:-$LACWM_BASE/wan_fun_1.3b_control}"
VIDEOX_HOME_VALUE="${VIDEOX_HOME:-$LACWM_BASE/VideoX-Fun-1d6d9c3}"
VJEPA_SOURCE="${VJEPA_SOURCE:-$LACWM_BASE/assets/vjepa2/source-release-45d025f636dfc58fc2426905fc4a1ab755b1c3e5}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-$LACWM_BASE/assets/vjepa2/45d025f636dfc58fc2426905fc4a1ab755b1c3e5/vjepa2_1_vitb_dist_vitG_384.pt}"
VJEPA_CHECKPOINT_SHA256="${VJEPA_CHECKPOINT_SHA256:-848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d}"
IMMUTABLE_CACHE_ROOT="${VJEPA_IMMUTABLE_CACHE_ROOT:-$LACWM_BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690}"
PCA_STATS="${VJEPA_PCA_STATS:-$IMMUTABLE_CACHE_ROOT/pca/train_pca64.pt}"
PCA_STATS_SHA256="${VJEPA_PCA_STATS_SHA256:-f0f086178a40a0451a5a989260cef1967fed2ff5ff7647d18c79b632e780c3fe}"
TRAIN_MANIFEST="${VJEPA_TRAIN_CLIP_MANIFEST:-$IMMUTABLE_CACHE_ROOT/manifests/train.jsonl}"

PARTITION="batch"
CACHE_TIME="01:00:00"
QUALITY_TIME="04:00:00"
CONTROL_TIME="01:00:00"
TIMING_TIME="04:00:00"
CACHE_CPUS="32"
CACHE_MEMORY="256G"
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
Usage: tools/slurm/submit_vjepa2_frontier_workflow.sh [options]

Submit the validation-selected V-JEPA NFE-frontier repair for the immutable v3
study.  Defaults are directly runnable on gcp-nrt-cs-001:

  study: vjepa2-controlled-20260730-seed1234-9cf8e69-v3
  final update-1000 array: 481132
  LACWM Python: lacwm-b200-py310
  extractor Python: vjepa2-extractor-py311
  lockbox cache: one B200, about 1.92 GB

Initial DAG:

  final-u1000 job -> cache/register (1 B200) --------+
                   -> final-artifact gate -----------+
                                                      |
                         +-> VPM validation (8 B200) -+
                         +-> J1 validation  (8 B200) -+
                                                      |
                         validation selection gate ---+

The selection gate creates no lockbox jobs unless the exclusively written
selection independently reproduces with confirmatory_eligible=true.  If it
does, it submits an afterok chain: VPM/J1 lockbox array (two separate 8-B200
allocations), confirmation, one-B200 paired timing, and finalization.

Path/provenance options:
  --study-root PATH
  --training-commit SHA
  --evaluator-commit SHA       Clean descendant; defaults to current HEAD
  --final-u1000-job-id ID      Default: 481132
  --adopt-cache-job-id ID      Recover one exact pending cache-only submission
  --python PATH                LACWM runtime interpreter
  --extractor-python PATH      Official V-JEPA extraction interpreter
  --wan-dir PATH
  --videox-home PATH
  --vjepa-source PATH
  --vjepa-checkpoint PATH
  --vjepa-checkpoint-sha256 HEX
  --pca PATH
  --pca-sha256 HEX
  --train-manifest PATH

Scheduler options:
  --partition NAME             Default: batch
  --cache-time HH:MM:SS        Default: 01:00:00
  --quality-time HH:MM:SS      Default: 04:00:00
  --control-time HH:MM:SS      Default: 01:00:00
  --timing-time HH:MM:SS       Default: 04:00:00
  --account NAME               Default: coreai_chef_posttrain
  --qos NAME                   Default: normal
  --execute                    Submit; otherwise run read-only preflight
  -h, --help

All scientific outputs are fresh-only.  The launcher never changes, cancels,
or requeues the active controlled study. Run it from a separate evaluator
worktree/clone; it refuses the training checkout recorded by the study.
EOF
}

while (($#)); do
  case "$1" in
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --final-u1000-job-id) FINAL_JOB_ID="${2:?}"; shift 2 ;;
    --adopt-cache-job-id) ADOPT_CACHE_JOB_ID="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --extractor-python) EXTRACTOR_PYTHON="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --vjepa-source) VJEPA_SOURCE="${2:?}"; shift 2 ;;
    --vjepa-checkpoint) VJEPA_CHECKPOINT="${2:?}"; shift 2 ;;
    --vjepa-checkpoint-sha256) VJEPA_CHECKPOINT_SHA256="${2:?}"; shift 2 ;;
    --pca) PCA_STATS="${2:?}"; shift 2 ;;
    --pca-sha256) PCA_STATS_SHA256="${2:?}"; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --cache-time) CACHE_TIME="${2:?}"; shift 2 ;;
    --quality-time) QUALITY_TIME="${2:?}"; shift 2 ;;
    --control-time) CONTROL_TIME="${2:?}"; shift 2 ;;
    --timing-time) TIMING_TIME="${2:?}"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid training commit"
[[ "$FINAL_JOB_ID" =~ ^[1-9][0-9]*$ ]] || die "invalid final update-1000 job ID"
[[ -z "$ADOPT_CACHE_JOB_ID" || "$ADOPT_CACHE_JOB_ID" =~ ^[1-9][0-9]*$ ]] || \
  die "invalid adopted cache job ID"
[[ "$VJEPA_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "invalid checkpoint SHA-256"
[[ "$PCA_STATS_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid PCA SHA-256"
for value in \
  "$PARTITION" "$CACHE_MEMORY" "$QUALITY_MEMORY" "$CONTROL_MEMORY" \
  "$TIMING_MEMORY" "$ACCOUNT" "$QOS"; do
  [[ "$value" != *[[:space:]]* ]] || die "scheduler scalar contains whitespace"
done
for value in "$CACHE_TIME" "$QUALITY_TIME" "$CONTROL_TIME" "$TIMING_TIME"; do
  [[ "$value" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || \
    die "time limits must use HH:MM:SS"
done
[[ "$REPO_ROOT" == /* && "$STUDY_ROOT" == /* ]] || die "roots must be absolute"
[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] || die "invalid repository"
[[ "$(cd "$REPO_ROOT" && pwd -P)" == "$REPO_ROOT" ]] || \
  die "repository root is not canonical"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || \
  die "launcher must run from the root of a Git clone or worktree"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is unavailable"
[[ -x "$PYTHON_BIN" && -x "$EXTRACTOR_PYTHON" ]] || \
  die "both Python interpreters must be executable"
for path in "$VJEPA_CHECKPOINT" "$PCA_STATS" "$TRAIN_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required immutable input is unavailable: $path"
done
[[ -d "$VJEPA_SOURCE/.git" && ! -L "$VJEPA_SOURCE" ]] || \
  die "V-JEPA source checkout is unavailable"
[[ -d "$WAN_DIR_VALUE" && -d "$VIDEOX_HOME_VALUE/.git" ]] || \
  die "Wan/VideoX runtime inputs are unavailable"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ -z "$EVALUATOR_COMMIT" ]]; then
  EVALUATOR_COMMIT="$ACTUAL_COMMIT"
fi
[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid evaluator commit"
[[ "$ACTUAL_COMMIT" == "$EVALUATOR_COMMIT" ]] || \
  die "repository differs from frozen evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
git -C "$REPO_ROOT" merge-base --is-ancestor \
  "$TRAINING_COMMIT" "$EVALUATOR_COMMIT" || \
  die "evaluator commit is not a descendant of the training commit"
[[ "$(git -C "$VJEPA_SOURCE" rev-parse HEAD)" == \
  "45d025f636dfc58fc2426905fc4a1ab755b1c3e5" ]] || \
  die "V-JEPA source checkout changed"
[[ -z "$(git -C "$VJEPA_SOURCE" status --porcelain --untracked-files=all)" ]] || \
  die "V-JEPA source checkout is dirty"
[[ "$(sha256sum "$VJEPA_CHECKPOINT" | awk '{print $1}')" == \
  "$VJEPA_CHECKPOINT_SHA256" ]] || die "V-JEPA checkpoint digest changed"
[[ "$(sha256sum "$PCA_STATS" | awk '{print $1}')" == "$PCA_STATS_SHA256" ]] || \
  die "PCA digest changed"

WORKFLOW_HELPER="$REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py"
CACHE_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_cache.sbatch"
FINAL_GATE_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_final_gate.sbatch"
QUALITY_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_quality.sbatch"
SELECT_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_select_and_submit.sbatch"
for path in \
  "$WORKFLOW_HELPER" "$CACHE_SBATCH" "$FINAL_GATE_SBATCH" \
  "$QUALITY_SBATCH" "$SELECT_SBATCH"; do
  [[ -x "$path" ]] || die "workflow entrypoint is not executable: $path"
done

# Validate both ancestry and exact inference-tree equality before any job can
# inherit this evaluator commit.
PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - \
  "$REPO_ROOT" "$TRAINING_COMMIT" "$EVALUATOR_COMMIT" <<'PY'
import json
import sys
from pathlib import Path
from tools.vjepa2_nfe_frontier import git_inference_compatibility
print(json.dumps(git_inference_compatibility(
    Path(sys.argv[1]), training_commit=sys.argv[2], tool_commit=sys.argv[3]
), sort_keys=True))
PY

mapfile -t STUDY_VALUES < <(
  "$PYTHON_BIN" "$WORKFLOW_HELPER" study-values \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT"
)
[[ "${#STUDY_VALUES[@]}" == "13" ]] || die "study input inventory is incomplete"
[[ "${STUDY_VALUES[0]}" == "$STUDY_ROOT" ]] || die "study root differs"
STUDY_ID="${STUDY_VALUES[1]}"
TRAINING_REPO_ROOT="${STUDY_VALUES[2]}"
[[ "$TRAINING_REPO_ROOT" != "$REPO_ROOT" ]] || \
  die "evaluator must use an isolated clone/worktree, not the study training checkout"
[[ -d "$TRAINING_REPO_ROOT" && ! -L "$TRAINING_REPO_ROOT" ]] || \
  die "study-recorded training checkout is unavailable"
[[ "$(git -C "$TRAINING_REPO_ROOT" rev-parse --show-toplevel)" == \
  "$TRAINING_REPO_ROOT" ]] || die "study-recorded training checkout is invalid"
[[ "$(git -C "$TRAINING_REPO_ROOT" rev-parse HEAD)" == "$TRAINING_COMMIT" ]] || \
  die "study-recorded training checkout no longer has its immutable commit"
[[ -z "$(git -C "$TRAINING_REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "study-recorded training checkout is dirty"
[[ "${STUDY_VALUES[3]}" == "$(readlink -f "$PYTHON_BIN")" ]] || \
  die "study LACWM Python target differs"
[[ "${STUDY_VALUES[4]}" == "$(readlink -f "$EXTRACTOR_PYTHON")" ]] || \
  die "study extractor Python differs"
[[ "${STUDY_VALUES[5]}" == "$WAN_DIR_VALUE" ]] || die "study Wan directory differs"
[[ "${STUDY_VALUES[6]}" == "$VIDEOX_HOME_VALUE" ]] || \
  die "study VideoX checkout differs"
[[ "${STUDY_VALUES[7]}" == "$VJEPA_SOURCE" ]] || die "study V-JEPA source differs"
[[ "${STUDY_VALUES[8]}" == "$VJEPA_CHECKPOINT" ]] || \
  die "study V-JEPA checkpoint differs"
[[ "${STUDY_VALUES[9]}" == "$VJEPA_CHECKPOINT_SHA256" ]] || \
  die "study checkpoint digest differs"
[[ "${STUDY_VALUES[10]}" == "$PCA_STATS" ]] || die "study PCA path differs"
[[ "${STUDY_VALUES[11]}" == "$PCA_STATS_SHA256" ]] || die "study PCA digest differs"
[[ "${STUDY_VALUES[12]}" == "$TRAIN_MANIFEST" ]] || \
  die "study training manifest differs"
"$PYTHON_BIN" "$WORKFLOW_HELPER" check-submission \
  --study-root "$STUDY_ROOT" \
  --training-commit "$TRAINING_COMMIT" \
  --final-job-id "$FINAL_JOB_ID"

SCIENTIFIC_FRONTIER_PATHS=(
  "$STUDY_ROOT/frontier_lockbox"
  "$STUDY_ROOT/vpm_parameter_matched_video/frontier_quality"
  "$STUDY_ROOT/j1_joint_auxiliary_leads/frontier_quality"
  "$STUDY_ROOT/frontier_selection.json"
  "$STUDY_ROOT/frontier_continuation.json"
  "$STUDY_ROOT/frontier_lockbox_confirmation.json"
  "$STUDY_ROOT/frontier_latency"
  "$STUDY_ROOT/frontier_final_report.json"
)
for path in "${SCIENTIFIC_FRONTIER_PATHS[@]}"; do
  [[ ! -e "$path" ]] || die "fresh frontier path already exists: $path"
done

WORKFLOW_ROOT="$STUDY_ROOT/_frontier_slurm"
LOG_DIR="$WORKFLOW_ROOT/logs"
SUBMISSION_RECORD="$WORKFLOW_ROOT/submission.json"
SCIENTIFIC_REPO_ROOT="$REPO_ROOT"
SCIENTIFIC_EVALUATOR_COMMIT="$EVALUATOR_COMMIT"
ADOPTION_EVIDENCE_JSON="null"
if [[ -n "$ADOPT_CACHE_JOB_ID" ]]; then
  [[ -d "$WORKFLOW_ROOT" && ! -L "$WORKFLOW_ROOT" ]] || \
    die "cache adoption requires the existing non-symlink workflow directory"
  [[ "$(cd "$WORKFLOW_ROOT" && pwd -P)" == "$WORKFLOW_ROOT" ]] || \
    die "cache adoption workflow directory is not canonical"
  [[ -d "$LOG_DIR" && ! -L "$LOG_DIR" ]] || \
    die "cache adoption requires the existing non-symlink log directory"
  [[ ! -e "$SUBMISSION_RECORD" ]] || \
    die "cache adoption refuses an existing submission record"
  [[ -z "$(find "$WORKFLOW_ROOT" -mindepth 1 -maxdepth 1 \
    ! -name logs -print -quit)" ]] || \
    die "cache adoption found unexpected workflow state"
  [[ -z "$(find "$LOG_DIR" -mindepth 1 -print -quit)" ]] || \
    die "cache adoption requires an empty pending-job log directory"
  command -v sacct >/dev/null 2>&1 || die "sacct is unavailable"
  command -v scontrol >/dev/null 2>&1 || die "scontrol is unavailable"
  ADOPTION_CHECK_ARGS=(
    validate-adopted-cache
    --job-id "$ADOPT_CACHE_JOB_ID"
    --study-root "$STUDY_ROOT"
    --training-commit "$TRAINING_COMMIT"
    --current-repo-root "$REPO_ROOT"
    --current-evaluator-commit "$EVALUATOR_COMMIT"
    --final-job-id "$FINAL_JOB_ID"
    --python "$PYTHON_BIN"
    --extractor-python "$EXTRACTOR_PYTHON"
    --vjepa-source "$VJEPA_SOURCE"
    --vjepa-checkpoint "$VJEPA_CHECKPOINT"
    --vjepa-checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256"
    --pca "$PCA_STATS"
    --pca-sha256 "$PCA_STATS_SHA256"
    --train-manifest "$TRAIN_MANIFEST"
    --partition "$PARTITION"
    --account "$ACCOUNT"
    --qos "$QOS"
    --cache-time "$CACHE_TIME"
    --cache-cpus "$CACHE_CPUS"
    --cache-memory "$CACHE_MEMORY"
    --log-dir "$LOG_DIR"
  )
  ADOPTION_OUTPUT="$(
    "$PYTHON_BIN" "$WORKFLOW_HELPER" "${ADOPTION_CHECK_ARGS[@]}"
  )" || die "pending cache job failed fail-closed adoption validation"
  mapfile -t ADOPTION_VALUES <<<"$ADOPTION_OUTPUT"
  [[ "${#ADOPTION_VALUES[@]}" == "3" ]] || \
    die "cache adoption evidence is incomplete"
  SCIENTIFIC_REPO_ROOT="${ADOPTION_VALUES[0]}"
  SCIENTIFIC_EVALUATOR_COMMIT="${ADOPTION_VALUES[1]}"
  ADOPTION_EVIDENCE_JSON="${ADOPTION_VALUES[2]}"
  [[ "$SCIENTIFIC_REPO_ROOT" != "$TRAINING_REPO_ROOT" ]] || \
    die "adopted scientific evaluator cannot be the training checkout"
  for path in \
    "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_workflow.py" \
    "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_quality.sbatch" \
    "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_confirm.sbatch" \
    "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_latency.sbatch"; do
    [[ -x "$path" ]] || \
      die "adopted scientific evaluator entrypoint is unavailable: $path"
  done
else
  [[ ! -e "$WORKFLOW_ROOT" ]] || \
    die "fresh frontier path already exists: $WORKFLOW_ROOT"
fi

echo "V-JEPA NFE-frontier workflow preflight passed."
echo "Study: $STUDY_ID"
echo "Isolated evaluator checkout: $REPO_ROOT"
echo "Untouched training checkout: $TRAINING_REPO_ROOT"
echo "Training commit: $TRAINING_COMMIT"
echo "Frozen evaluator commit: $EVALUATOR_COMMIT (clean compatible descendant)"
if [[ -n "$ADOPT_CACHE_JOB_ID" ]]; then
  echo "Adopted pending cache: $ADOPT_CACHE_JOB_ID"
  echo "Scientific evaluator: $SCIENTIFIC_EVALUATOR_COMMIT at $SCIENTIFIC_REPO_ROOT"
fi
echo "Final update-1000 job: $FINAL_JOB_ID (active/terminal mode resolved at execute)"
echo "Cache: one B200, fresh 128-clip lockbox, approximately 1.92 GB"
echo "Validation: separate VPM and J1 one-node/eight-B200 jobs"
echo "Lockbox: not submitted unless confirmatory_eligible=true"
echo "Timing: one B200, 18 warmups + 120 counterbalanced rounds"

if ((EXECUTE == 0)); then
  echo "Dry run only. Re-run with --evaluator-commit '$EVALUATOR_COMMIT' --execute."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v sacct >/dev/null 2>&1 || die "sacct is unavailable"
command -v flock >/dev/null 2>&1 || die "flock is unavailable"
FINAL_JOB_MODE="$(
  "$PYTHON_BIN" "$WORKFLOW_HELPER" classify-final-job \
    --final-job-id "$FINAL_JOB_ID"
)" || die "could not classify final update-1000 job"
FINAL_JOB_DEPENDENCY_ARGS=()
case "$FINAL_JOB_MODE" in
  active_afterok)
    FINAL_JOB_DEPENDENCY_ARGS+=(--dependency="afterok:$FINAL_JOB_ID")
    FINAL_JOB_DEPENDENCY_RECORD="afterok:$FINAL_JOB_ID"
    ;;
  terminal_success)
    # A completed job can disappear from slurmctld after MinJobAge. The full
    # final-artifact gate remains mandatory, so no stale dependency is needed.
    FINAL_JOB_DEPENDENCY_RECORD="terminal_success_verified_via_sacct"
    ;;
  *) die "unknown final-job scheduling mode: $FINAL_JOB_MODE" ;;
esac
if [[ -n "$ADOPT_CACHE_JOB_ID" ]]; then
  exec 9>"$WORKFLOW_ROOT/workflow.lock"
  flock -n 9 || die "another frontier recovery owns the workflow lock"
  ADOPTION_RECHECK="$(
    "$PYTHON_BIN" "$WORKFLOW_HELPER" "${ADOPTION_CHECK_ARGS[@]}"
  )" || die "adopted cache changed before recovery submission"
  [[ "$ADOPTION_RECHECK" == "$ADOPTION_OUTPUT" ]] || \
    die "adopted cache evidence changed before recovery submission"
fi
WORKFLOW_ROOT="$STUDY_ROOT/_frontier_slurm"
LOG_DIR="$WORKFLOW_ROOT/logs"
if [[ -z "$ADOPT_CACHE_JOB_ID" ]]; then
  mkdir "$WORKFLOW_ROOT"
  mkdir "$LOG_DIR"
  exec 9>"$WORKFLOW_ROOT/workflow.lock"
  flock -n 9 || die "another frontier submission owns the workflow lock"
fi

SCHEDULER_OPTIONAL=()
[[ -n "$ACCOUNT" ]] && SCHEDULER_OPTIONAL+=(--account="$ACCOUNT")
[[ -n "$QOS" ]] && SCHEDULER_OPTIONAL+=(--qos="$QOS")
normalize_job_id() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
    die "Slurm returned an invalid job ID: $value"
  printf '%s\n' "${value%%[_;]*}"
}

if [[ -n "$ADOPT_CACHE_JOB_ID" ]]; then
  CACHE_JOB_ID="$ADOPT_CACHE_JOB_ID"
  CACHE_DEPENDENCY_RECORD="afterok:$FINAL_JOB_ID (adopted pending cache)"
else
  CACHE_JOB_ID="$(
    sbatch \
      --parsable \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --gpus-per-node=1 \
      --cpus-per-task="$CACHE_CPUS" \
      --mem="$CACHE_MEMORY" \
      --time="$CACHE_TIME" \
      --partition="$PARTITION" \
      "${FINAL_JOB_DEPENDENCY_ARGS[@]}" \
      --no-requeue \
      --open-mode=append \
      --export=ALL \
      --job-name="vjepa2-frontier-cache" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "${SCHEDULER_OPTIONAL[@]}" \
      "$CACHE_SBATCH" \
      --repo-root "$REPO_ROOT" \
      --study-root "$STUDY_ROOT" \
      --training-commit "$TRAINING_COMMIT" \
      --evaluator-commit "$EVALUATOR_COMMIT" \
      --python "$PYTHON_BIN" \
      --extractor-python "$EXTRACTOR_PYTHON" \
      --vjepa-source "$VJEPA_SOURCE" \
      --vjepa-checkpoint "$VJEPA_CHECKPOINT" \
      --vjepa-checkpoint-sha256 "$VJEPA_CHECKPOINT_SHA256" \
      --pca "$PCA_STATS" \
      --pca-sha256 "$PCA_STATS_SHA256" \
      --train-manifest "$TRAIN_MANIFEST"
  )" || die "Slurm rejected lockbox cache job"
  CACHE_DEPENDENCY_RECORD="$FINAL_JOB_DEPENDENCY_RECORD"
fi
CACHE_DEPENDENCY="$(normalize_job_id "$CACHE_JOB_ID")"

FINAL_GATE_JOB_ID="$(
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
    "${FINAL_JOB_DEPENDENCY_ARGS[@]}" \
    --no-requeue \
    --open-mode=append \
    --export=ALL \
    --job-name="vjepa2-frontier-u1000-gate" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${SCHEDULER_OPTIONAL[@]}" \
    "$FINAL_GATE_SBATCH" \
    --repo-root "$REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --evaluator-commit "$EVALUATOR_COMMIT" \
    --scientific-repo-root "$SCIENTIFIC_REPO_ROOT" \
    --scientific-evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT" \
    --python "$PYTHON_BIN"
)" || die "Slurm rejected final-artifact gate"
FINAL_GATE_DEPENDENCY="$(normalize_job_id "$FINAL_GATE_JOB_ID")"
VALIDATION_DEPENDENCY="afterok:$CACHE_DEPENDENCY:$FINAL_GATE_DEPENDENCY"

QUALITY_JOB_ARGS=(
  --repo-root "$SCIENTIFIC_REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT"
  --python "$PYTHON_BIN"
  --wan-dir "$WAN_DIR_VALUE"
  --videox-home "$VIDEOX_HOME_VALUE"
  --split validation
)
submit_validation() {
  local arm="$1"
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
    --dependency="$VALIDATION_DEPENDENCY" \
    --no-requeue \
    --open-mode=append \
    --export=ALL \
    --job-name="vjepa2-frontier-val-${arm,,}" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${SCHEDULER_OPTIONAL[@]}" \
    "$SCIENTIFIC_REPO_ROOT/tools/slurm/vjepa2_frontier_quality.sbatch" \
    "${QUALITY_JOB_ARGS[@]}" \
    --arm "$arm"
}
VPM_VALIDATION_JOB_ID="$(submit_validation VPM)" || \
  die "Slurm rejected VPM validation"
J1_VALIDATION_JOB_ID="$(submit_validation J1)" || \
  die "Slurm rejected J1 validation"
VPM_VALIDATION_DEPENDENCY="$(normalize_job_id "$VPM_VALIDATION_JOB_ID")"
J1_VALIDATION_DEPENDENCY="$(normalize_job_id "$J1_VALIDATION_JOB_ID")"

SELECT_JOB_ID="$(
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
    --dependency="afterok:$VPM_VALIDATION_DEPENDENCY:$J1_VALIDATION_DEPENDENCY" \
    --no-requeue \
    --open-mode=append \
    --export=ALL \
    --job-name="vjepa2-frontier-select" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "${SCHEDULER_OPTIONAL[@]}" \
    "$SELECT_SBATCH" \
    --repo-root "$REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --evaluator-commit "$EVALUATOR_COMMIT" \
    --scientific-repo-root "$SCIENTIFIC_REPO_ROOT" \
    --scientific-evaluator-commit "$SCIENTIFIC_EVALUATOR_COMMIT" \
    --python "$PYTHON_BIN" \
    --wan-dir "$WAN_DIR_VALUE" \
    --videox-home "$VIDEOX_HOME_VALUE" \
    --partition "$PARTITION" \
    --quality-time "$QUALITY_TIME" \
    --control-time "$CONTROL_TIME" \
    --timing-time "$TIMING_TIME" \
    --quality-cpus "$QUALITY_CPUS" \
    --quality-mem "$QUALITY_MEMORY" \
    --control-cpus "$CONTROL_CPUS" \
    --control-mem "$CONTROL_MEMORY" \
    --timing-cpus "$TIMING_CPUS" \
    --timing-mem "$TIMING_MEMORY" \
    --account "$ACCOUNT" \
    --qos "$QOS" \
    --log-dir "$LOG_DIR"
)" || die "Slurm rejected selection gate"
normalize_job_id "$SELECT_JOB_ID" >/dev/null

"$PYTHON_BIN" - \
  "$SUBMISSION_RECORD" "$STUDY_ROOT" "$TRAINING_COMMIT" \
  "$REPO_ROOT" "$EVALUATOR_COMMIT" \
  "$SCIENTIFIC_REPO_ROOT" "$SCIENTIFIC_EVALUATOR_COMMIT" \
  "$FINAL_JOB_ID" "$CACHE_JOB_ID" "$FINAL_GATE_JOB_ID" \
  "$VPM_VALIDATION_JOB_ID" "$J1_VALIDATION_JOB_ID" "$SELECT_JOB_ID" \
  "$PYTHON_BIN" "$EXTRACTOR_PYTHON" "$FINAL_JOB_MODE" \
  "$FINAL_JOB_DEPENDENCY_RECORD" "$CACHE_DEPENDENCY_RECORD" \
  "$ADOPTION_EVIDENCE_JSON" <<'PY'
import datetime
import json
import sys
from pathlib import Path

(
    output,
    study_root,
    training_commit,
    controller_repo_root,
    controller_commit,
    scientific_repo_root,
    scientific_evaluator_commit,
    final_job,
    cache_job,
    final_gate_job,
    vpm_validation_job,
    j1_validation_job,
    selection_job,
    lacwm_python,
    extractor_python,
    final_job_mode,
    final_job_dependency,
    cache_dependency,
    adoption_evidence_json,
) = sys.argv[1:]
adoption_evidence = json.loads(adoption_evidence_json)
payload = {
    "kind": "vjepa2_nfe_frontier_slurm_submission",
    "schema_version": 1,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "study_root": study_root,
    "training_git_commit": training_commit,
    "evaluator_git_commit": scientific_evaluator_commit,
    "scientific_evaluator_repo_root": scientific_repo_root,
    "controller_repo_root": controller_repo_root,
    "controller_git_commit": controller_commit,
    "interpreters": {
        "lacwm": lacwm_python,
        "vjepa2_extractor": extractor_python,
    },
    "final_update_1000_job_id": final_job,
    "final_update_1000_accounting_mode": final_job_mode,
    "cache_job_id": cache_job,
    "cache_job_adopted": adoption_evidence is not None,
    "cache_adoption_evidence": adoption_evidence,
    "final_artifact_gate_job_id": final_gate_job,
    "validation_job_ids": {
        "VPM": vpm_validation_job,
        "J1": j1_validation_job,
    },
    "selection_gate_job_id": selection_job,
    "dependencies": {
        "cache": cache_dependency,
        "final_artifact_gate": final_job_dependency,
        "validation": (
            f"afterok:{cache_job.split(';', 1)[0].split('_', 1)[0]}:"
            f"{final_gate_job.split(';', 1)[0].split('_', 1)[0]}"
        ),
        "selection": (
            f"afterok:{vpm_validation_job.split(';', 1)[0].split('_', 1)[0]}:"
            f"{j1_validation_job.split(';', 1)[0].split('_', 1)[0]}"
        ),
    },
    "lockbox_jobs_submitted_at_initial_submission": False,
    "lockbox_submission_requires_confirmatory_eligible": True,
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Initial frontier DAG submitted."
echo "Final job scheduling mode: $FINAL_JOB_MODE ($FINAL_JOB_DEPENDENCY_RECORD)"
echo "Cache/register: $CACHE_JOB_ID"
echo "Final artifact gate: $FINAL_GATE_JOB_ID"
echo "VPM validation: $VPM_VALIDATION_JOB_ID"
echo "J1 validation: $J1_VALIDATION_JOB_ID"
echo "Selection/conditional continuation: $SELECT_JOB_ID"
echo "Submission record: $SUBMISSION_RECORD"
