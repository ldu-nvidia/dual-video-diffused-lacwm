#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/csip_phase0_stage.sbatch"
REGISTRATION=""
EXPECTED_COMMIT=""
PYTHON_BIN=""
EXECUTE=0
ALLOW_ACTIVE_JOB_IDS=()

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: tools/slurm/submit_csip_phase0.sh --registration PATH --expected-commit SHA --python PATH [options]

Dry-run and, only with --execute, submit the two dependency-ordered CSIP jobs:
train latent extraction + paired probe fit + seal, followed by sealed validation
extraction + evaluation + bootstrap analysis. Both allocations are one-node,
eight-B200, short-QOS, non-requeueable jobs. Existing jobs are never modified.

Options:
  --registration PATH          Existing prospective registration (required)
  --expected-commit SHA        Exact reviewed clean source commit (required)
  --python PATH                Registered B200 Python launcher (required)
  --allow-active-job-id ID     Repeat for every currently active user allocation
  --execute                    Enforce gates and submit; omit for read-only dry run
  -h, --help                   Show this text
EOF
}

while (($#)); do
  case "$1" in
    --registration) [[ $# -ge 2 ]] || die "--registration requires a value"; REGISTRATION="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "--expected-commit requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --allow-active-job-id) [[ $# -ge 2 ]] || die "--allow-active-job-id requires a value"; ALLOW_ACTIVE_JOB_IDS+=("$2"); shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$REGISTRATION" == /* && -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || \
  die "--registration must be an absolute regular file"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "--expected-commit must be a full lowercase commit"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || \
  die "--python must be an absolute executable"
[[ -f "$SBATCH_SCRIPT" && ! -L "$SBATCH_SCRIPT" ]] || die "CSIP sbatch script is absent"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || \
  die "repository commit differs"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"

declare -A ALLOWED=()
for job_id in "${ALLOW_ACTIVE_JOB_IDS[@]}"; do
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || die "active-job allowlist IDs must be numeric"
  [[ -z "${ALLOWED[$job_id]+present}" ]] || die "active-job allowlist contains a duplicate"
  ALLOWED["$job_id"]=1
done

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$REPO_ROOT"
"$PYTHON_BIN" "$REPO_ROOT/tools/csip_workflow.py" render \
  --registration "$REGISTRATION" --stage train >/dev/null

mapfile -t FIELDS < <(
  "$PYTHON_BIN" - "$REGISTRATION" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
paths = value["planned_paths"]
for item in (
    value["source"]["path"],
    value["source"]["git_commit"],
    value["runtime"]["python"]["launcher_path"],
    value["study_root"],
    paths["train_latent_root"],
    paths["checkpoint"],
    paths["training_report"],
    paths["checkpoint_seal"],
    paths["validation_latent_root"],
    paths["evaluation"],
    paths["analysis"],
    paths["wandb_local_dir"],
):
    print(item)
PY
)
[[ ${#FIELDS[@]} -eq 12 ]] || die "registration launch fields are incomplete"
[[ "${FIELDS[0]}" == "$REPO_ROOT" && "${FIELDS[1]}" == "$EXPECTED_COMMIT" ]] || \
  die "registration source differs"
[[ "${FIELDS[2]}" == "$PYTHON_BIN" ]] || die "registration Python differs"
STUDY_ROOT="${FIELDS[3]}"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || die "study root is unavailable"
for output in "${FIELDS[@]:4}"; do
  [[ ! -e "$output" && ! -L "$output" ]] || die "planned output is not fresh: $output"
done

TRAIN_NAME="csip-train-$(basename "$STUDY_ROOT")"
VAL_NAME="csip-val-$(basename "$STUDY_ROOT")"
LOG_ROOT="$STUDY_ROOT/slurm-logs"
TRAIN_COMMAND=(
  sbatch --parsable --no-requeue --job-name="$TRAIN_NAME"
  --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err"
  "$SBATCH_SCRIPT" train "$REGISTRATION" "$REPO_ROOT" "$EXPECTED_COMMIT" "$PYTHON_BIN"
)

printf 'Validated train command:'
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
echo "Validation command will use --dependency=afterok:<train-job-id> --kill-on-invalid-dep=yes"
echo "Study: $STUDY_ROOT"
echo "Source: $EXPECTED_COMMIT (clean)"
echo "W&B: zijiandu/dual-video-diffusion-private, online, group=null, resume=never"
echo "Protected test: no accepted path"
echo "Requested active-job allowlist: ${ALLOW_ACTIVE_JOB_IDS[*]:-(empty)}"

if ((EXECUTE == 0)); then
  echo "Dry run only; no directory or job was created."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"
mapfile -t OBSERVED < <(
  squeue --noheader --user "${USER:?USER is unset}" --format='%F' | \
    sed '/^[[:space:]]*$/d' | sort -n -u
)
if [[ ${#OBSERVED[@]} -ne ${#ALLOW_ACTIVE_JOB_IDS[@]} ]]; then
  die "active jobs differ from the exact allowlist: observed=${OBSERVED[*]:-(empty)}"
fi
for job_id in "${OBSERVED[@]}"; do
  [[ -n "${ALLOWED[$job_id]+present}" ]] || \
    die "active job $job_id is not explicitly allowed"
done

mkdir "$LOG_ROOT"
TRAIN_JOB_ID="$("${TRAIN_COMMAND[@]}")"
[[ "$TRAIN_JOB_ID" =~ ^[1-9][0-9]*$ ]] || die "unexpected train job ID: $TRAIN_JOB_ID"
VAL_COMMAND=(
  sbatch --parsable --no-requeue --dependency="afterok:$TRAIN_JOB_ID"
  --kill-on-invalid-dep=yes --job-name="$VAL_NAME"
  --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err"
  "$SBATCH_SCRIPT" validation "$REGISTRATION" "$REPO_ROOT" "$EXPECTED_COMMIT" "$PYTHON_BIN"
)
VAL_JOB_ID="$("${VAL_COMMAND[@]}")"
[[ "$VAL_JOB_ID" =~ ^[1-9][0-9]*$ ]] || die "unexpected validation job ID: $VAL_JOB_ID"
echo "Submitted train job: $TRAIN_JOB_ID"
echo "Submitted sealed validation job: $VAL_JOB_ID (afterok:$TRAIN_JOB_ID)"
