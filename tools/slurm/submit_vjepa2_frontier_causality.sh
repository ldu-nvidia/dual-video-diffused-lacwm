#!/usr/bin/env bash
# Submit the preregistered, separate within-J1 lockbox causality sidecar.
#
# This launcher becomes runnable only after the main selection gate has written
# an eligible frontier_continuation.json. It depends on the existing J1
# autonomous lockbox array task and never submits or modifies a main-frontier
# job. Per-job exclusive receipts allow a rerun to finish a partially submitted
# two-job chain without duplicating the accepted job.

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LACWM_BASE="${LACWM_BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train}"
STUDY_ROOT="${VJEPA_FRONTIER_STUDY_ROOT:-$LACWM_BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3}"
TRAINING_COMMIT="${VJEPA_FRONTIER_TRAINING_COMMIT:-9cf8e6922f35a5d6645e3128545953723bf54da2}"
EVALUATOR_COMMIT=""
PYTHON_BIN="${LACWM_PYTHON:-$LACWM_BASE/envs/lacwm-b200-py310/bin/python}"
WAN_DIR_VALUE="${WAN_DIR:-$LACWM_BASE/wan_fun_1.3b_control}"
VIDEOX_HOME_VALUE="${VIDEOX_HOME:-$LACWM_BASE/VideoX-Fun-1d6d9c3}"
PARTITION="batch"
ACCOUNT="coreai_chef_posttrain"
QOS="normal"
QUALITY_TIME="04:00:00"
CONTROL_TIME="01:00:00"
QUALITY_CPUS="160"
QUALITY_MEMORY="1000G"
CONTROL_CPUS="16"
CONTROL_MEMORY="64G"
EXECUTE=0

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: submit_vjepa2_frontier_causality.sh --evaluator-commit SHA [--execute]

Dry-runs by default. The immutable sidecar evaluator must be a clean
training-commit descendant with unchanged inference-critical trees.

The launcher is conditional: frontier_selection.json and
frontier_continuation.json must already prove confirmatory_eligible=true.
It submits:

  existing main J1 lockbox array task -> J1 off+shuffled quality (8 B200)
                                      -> separate confirmation (1 B200)

Accepted jobs are recorded individually under _frontier_causality_slurm.
Rerunning the same command recovers from a partial submission by adopting those
exclusive receipts instead of submitting duplicate jobs.
EOF
}

while (($#)); do
  case "$1" in
    --evaluator-commit) EVALUATOR_COMMIT="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --training-commit) TRAINING_COMMIT="${2:?}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR_VALUE="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME_VALUE="${2:?}"; shift 2 ;;
    --partition) PARTITION="${2:?}"; shift 2 ;;
    --account) ACCOUNT="${2:?}"; shift 2 ;;
    --qos) QOS="${2:?}"; shift 2 ;;
    --quality-time) QUALITY_TIME="${2:?}"; shift 2 ;;
    --control-time) CONTROL_TIME="${2:?}"; shift 2 ;;
    --quality-cpus) QUALITY_CPUS="${2:?}"; shift 2 ;;
    --quality-mem) QUALITY_MEMORY="${2:?}"; shift 2 ;;
    --control-cpus) CONTROL_CPUS="${2:?}"; shift 2 ;;
    --control-mem) CONTROL_MEMORY="${2:?}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$EVALUATOR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "--evaluator-commit must be a full lowercase SHA-1"
[[ "$TRAINING_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid training commit"
[[ -x "$PYTHON_BIN" ]] || die "LACWM Python is unavailable"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is unavailable"
[[ "$(cd "$STUDY_ROOT" && pwd -P)" == "$STUDY_ROOT" ]] || \
  die "study root must be canonical"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EVALUATOR_COMMIT" ]] || \
  die "repository is not at the requested sidecar evaluator commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "sidecar evaluator repository must be clean"
for scalar in \
  "$PARTITION" "$ACCOUNT" "$QOS" "$QUALITY_TIME" "$CONTROL_TIME" \
  "$QUALITY_MEMORY" "$CONTROL_MEMORY"; do
  [[ -n "$scalar" && "$scalar" != *[[:space:]]* ]] || \
    die "scheduler scalar is empty or contains whitespace"
done
for integer in "$QUALITY_CPUS" "$CONTROL_CPUS"; do
  [[ "$integer" =~ ^[1-9][0-9]*$ ]] || die "CPU counts must be positive integers"
done

SELECTION="$STUDY_ROOT/frontier_selection.json"
CONTINUATION="$STUDY_ROOT/frontier_continuation.json"
CONTRACT="$REPO_ROOT/tools/vjepa2_lockbox_causality.py"
QUALITY_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_causality_quality.sbatch"
CONFIRM_SBATCH="$REPO_ROOT/tools/slurm/vjepa2_frontier_causality_confirm.sbatch"
for path in "$CONTRACT" "$QUALITY_SBATCH" "$CONFIRM_SBATCH"; do
  [[ -x "$path" ]] || die "sidecar entrypoint is not executable: $path"
done

mapfile -t VALUES < <(
  "$PYTHON_BIN" "$CONTRACT" values \
    --repo-root "$REPO_ROOT" \
    --study-root "$STUDY_ROOT" \
    --training-commit "$TRAINING_COMMIT" \
    --sidecar-evaluator-commit "$EVALUATOR_COMMIT" \
    --selection "$SELECTION" \
    --continuation "$CONTINUATION"
)
[[ "${#VALUES[@]}" == "10" ]] || die "sidecar prerequisite values are incomplete"
SELECTED_NFE="${VALUES[0]}"
SELECTION_ID="${VALUES[1]}"
LOCKBOX_ID="${VALUES[2]}"
AUTONOMOUS_INVENTORY="${VALUES[3]}"
SIDECAR_OUTPUT="${VALUES[4]}"
SIDECAR_INVENTORY="${VALUES[5]}"
CONFIRMATION_OUTPUT="${VALUES[6]}"
MAIN_J1_DEPENDENCY="${VALUES[7]}"
CONTINUATION_SHA256="${VALUES[8]}"
PRIMARY_EVALUATOR_COMMIT="${VALUES[9]}"

WORKFLOW_ROOT="$STUDY_ROOT/_frontier_causality_slurm"
LOG_DIR="$WORKFLOW_ROOT/logs"
QUALITY_RECEIPT="$WORKFLOW_ROOT/quality_job.json"
CONFIRM_RECEIPT="$WORKFLOW_ROOT/confirmation_job.json"
SUBMISSION="$WORKFLOW_ROOT/submission.json"

echo "Conditional sidecar is eligible and preregistered."
echo "Frozen J1 NFE: $SELECTED_NFE"
echo "Main J1 dependency: $MAIN_J1_DEPENDENCY"
echo "Primary evaluator: $PRIMARY_EVALUATOR_COMMIT"
echo "Sidecar evaluator: $EVALUATOR_COMMIT"
echo "Sidecar inventory: $SIDECAR_INVENTORY"
echo "Separate confirmation: $CONFIRMATION_OUTPUT"
if ((!EXECUTE)); then
  echo "Dry run only; pass --execute to submit the separate two-job chain."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
if [[ ! -e "$WORKFLOW_ROOT" ]]; then
  mkdir "$WORKFLOW_ROOT"
  chmod 700 "$WORKFLOW_ROOT"
  mkdir "$LOG_DIR"
  chmod 700 "$LOG_DIR"
else
  [[ -d "$WORKFLOW_ROOT" && ! -L "$WORKFLOW_ROOT" ]] || \
    die "workflow root is not a non-symlink directory"
  [[ -d "$LOG_DIR" && ! -L "$LOG_DIR" ]] || \
    die "workflow log directory is unavailable"
fi
[[ -z "$(find "$WORKFLOW_ROOT" -mindepth 1 -maxdepth 1 \
  ! -name logs \
  ! -name quality_job.json \
  ! -name confirmation_job.json \
  ! -name submission.json \
  -print -quit)" ]] || die "workflow root contains unexpected state"
[[ ! -e "$SUBMISSION" ]] || die "sidecar submission already finalized: $SUBMISSION"
if [[ -e "$SIDECAR_OUTPUT" ]]; then
  [[ -s "$SIDECAR_INVENTORY" && -e "$QUALITY_RECEIPT" ]] || \
    die "sidecar output is partial or lacks its accepted-job receipt"
fi
if [[ -e "$CONFIRMATION_OUTPUT" ]]; then
  [[ -s "$CONFIRMATION_OUTPUT" && -e "$CONFIRM_RECEIPT" ]] || \
    die "causality confirmation is partial or lacks its accepted-job receipt"
fi

normalize_job_id() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$ ]] || \
    die "Slurm returned an invalid job ID: $value"
  printf '%s\n' "${value%%[_;]*}"
}

write_receipt() {
  local output="$1"
  local kind="$2"
  local job_id="$3"
  local dependency="$4"
  "$PYTHON_BIN" - \
    "$output" "$kind" "$job_id" "$dependency" "$SELECTION_ID" "$LOCKBOX_ID" \
    "$CONTINUATION_SHA256" "$TRAINING_COMMIT" "$PRIMARY_EVALUATOR_COMMIT" \
    "$EVALUATOR_COMMIT" "$REPO_ROOT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output, kind, job_id, dependency, selection_id, lockbox_id,
    continuation_sha, training_commit, primary_commit, sidecar_commit, repo_root,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "kind": kind,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": job_id,
    "dependency": dependency,
    "selection_identity_sha256": selection_id,
    "lockbox_registration_identity_sha256": lockbox_id,
    "frontier_continuation_sha256": continuation_sha,
    "training_git_commit": training_commit,
    "primary_frontier_evaluator_git_commit": primary_commit,
    "sidecar_evaluator_git_commit": sidecar_commit,
    "sidecar_evaluator_repo_root": repo_root,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

read_receipt_job() {
  local receipt="$1"
  local expected_kind="$2"
  local expected_dependency="$3"
  "$PYTHON_BIN" - \
    "$receipt" "$expected_kind" "$expected_dependency" "$SELECTION_ID" "$LOCKBOX_ID" \
    "$CONTINUATION_SHA256" "$TRAINING_COMMIT" "$PRIMARY_EVALUATOR_COMMIT" \
    "$EVALUATOR_COMMIT" "$REPO_ROOT" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(path, kind, dependency, selection, lockbox, continuation, training, primary, sidecar, repo) = sys.argv[1:]
payload = json.loads(Path(path).read_text(encoding="utf-8"))
identity = payload.pop("identity_sha256", None)
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
if (
    identity != hashlib.sha256(canonical).hexdigest()
    or payload.get("kind") != kind
    or payload.get("dependency") != dependency
    or payload.get("selection_identity_sha256") != selection
    or payload.get("lockbox_registration_identity_sha256") != lockbox
    or payload.get("frontier_continuation_sha256") != continuation
    or payload.get("training_git_commit") != training
    or payload.get("primary_frontier_evaluator_git_commit") != primary
    or payload.get("sidecar_evaluator_git_commit") != sidecar
    or payload.get("sidecar_evaluator_repo_root") != repo
    or re.fullmatch(r"[1-9][0-9]*", str(payload.get("job_id", ""))) is None
):
    raise SystemExit("invalid sidecar job receipt")
print(payload["job_id"])
PY
}

SCHEDULER_OPTIONAL=(--account="$ACCOUNT" --qos="$QOS")
COMMON_ARGS=(
  --repo-root "$REPO_ROOT"
  --study-root "$STUDY_ROOT"
  --training-commit "$TRAINING_COMMIT"
  --evaluator-commit "$EVALUATOR_COMMIT"
  --python "$PYTHON_BIN"
  --selection "$SELECTION"
  --continuation "$CONTINUATION"
)

if [[ -e "$QUALITY_RECEIPT" ]]; then
  QUALITY_JOB_ID="$(read_receipt_job \
    "$QUALITY_RECEIPT" "vjepa2_frontier_causality_quality_job" \
    "$MAIN_J1_DEPENDENCY")" || \
    die "quality recovery receipt is invalid"
  echo "Recovered accepted causality quality job: $QUALITY_JOB_ID"
else
  RAW_QUALITY_JOB_ID="$(
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
      --dependency="$MAIN_J1_DEPENDENCY" \
      --no-requeue \
      --open-mode=append \
      --export=ALL \
      --job-name="vjepa2-j1-causality-quality" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "${SCHEDULER_OPTIONAL[@]}" \
      "$QUALITY_SBATCH" \
      "${COMMON_ARGS[@]}" \
      --wan-dir "$WAN_DIR_VALUE" \
      --videox-home "$VIDEOX_HOME_VALUE"
  )" || die "Slurm rejected J1 causality quality"
  QUALITY_JOB_ID="$(normalize_job_id "$RAW_QUALITY_JOB_ID")"
  write_receipt \
    "$QUALITY_RECEIPT" "vjepa2_frontier_causality_quality_job" \
    "$QUALITY_JOB_ID" "$MAIN_J1_DEPENDENCY"
fi

CONFIRM_DEPENDENCY="afterok:$QUALITY_JOB_ID"
if [[ -e "$CONFIRM_RECEIPT" ]]; then
  CONFIRM_JOB_ID="$(read_receipt_job \
    "$CONFIRM_RECEIPT" "vjepa2_frontier_causality_confirmation_job" \
    "$CONFIRM_DEPENDENCY")" || \
    die "confirmation recovery receipt is invalid"
  echo "Recovered accepted causality confirmation job: $CONFIRM_JOB_ID"
else
  RAW_CONFIRM_JOB_ID="$(
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
      --dependency="$CONFIRM_DEPENDENCY" \
      --no-requeue \
      --open-mode=append \
      --export=ALL \
      --job-name="vjepa2-j1-causality-confirm" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "${SCHEDULER_OPTIONAL[@]}" \
      "$CONFIRM_SBATCH" \
      "${COMMON_ARGS[@]}"
  )" || die "Slurm rejected J1 causality confirmation"
  CONFIRM_JOB_ID="$(normalize_job_id "$RAW_CONFIRM_JOB_ID")"
  write_receipt \
    "$CONFIRM_RECEIPT" "vjepa2_frontier_causality_confirmation_job" \
    "$CONFIRM_JOB_ID" "$CONFIRM_DEPENDENCY"
fi

"$PYTHON_BIN" - \
  "$SUBMISSION" "$QUALITY_RECEIPT" "$CONFIRM_RECEIPT" \
  "$AUTONOMOUS_INVENTORY" "$SIDECAR_INVENTORY" "$CONFIRMATION_OUTPUT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

output, quality_path, confirmation_path, autonomous, sidecar, result = sys.argv[1:]
quality = json.loads(Path(quality_path).read_text(encoding="utf-8"))
confirmation = json.loads(Path(confirmation_path).read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "kind": "vjepa2_frontier_causality_slurm_submission",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "quality_job": quality,
    "confirmation_job": confirmation,
    "existing_autonomous_inventory": autonomous,
    "sidecar_inventory": sidecar,
    "separate_confirmation": result,
    "main_frontier_artifacts_modified": False,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "Separate J1 causality chain submitted."
echo "Quality: $QUALITY_JOB_ID ($MAIN_J1_DEPENDENCY)"
echo "Confirmation: $CONFIRM_JOB_ID ($CONFIRM_DEPENDENCY)"
echo "Submission record: $SUBMISSION"
