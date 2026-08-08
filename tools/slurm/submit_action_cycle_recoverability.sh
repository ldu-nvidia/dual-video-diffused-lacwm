#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --study-root PATH --python PATH --train-metadata PATH --train-manifest PATH --validation-metadata PATH --validation-manifest PATH --videox-home PATH --wan-dir PATH" >&2
  exit 2
}

STUDY_ROOT= PYTHON_BIN= TRAIN_METADATA= TRAIN_MANIFEST=
VALIDATION_METADATA= VALIDATION_MANIFEST= VIDEOX_HOME= WAN_DIR=
while [[ $# -gt 0 ]]; do
  case $1 in
    --study-root) STUDY_ROOT=$2; shift 2 ;;
    --python) PYTHON_BIN=$2; shift 2 ;;
    --train-metadata) TRAIN_METADATA=$2; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST=$2; shift 2 ;;
    --validation-metadata) VALIDATION_METADATA=$2; shift 2 ;;
    --validation-manifest) VALIDATION_MANIFEST=$2; shift 2 ;;
    --videox-home) VIDEOX_HOME=$2; shift 2 ;;
    --wan-dir) WAN_DIR=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n $STUDY_ROOT && -n $PYTHON_BIN && -n $TRAIN_METADATA && -n $TRAIN_MANIFEST ]] || usage
[[ -n $VALIDATION_METADATA && -n $VALIDATION_MANIFEST && -n $VIDEOX_HOME && -n $WAN_DIR ]] || usage

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
TOOL=$REPO/tools/action_cycle_recoverability.py
LUSTRE_BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
[[ -d $LUSTRE_BASE && ! -L $LUSTRE_BASE && $(realpath -e -- "$LUSTRE_BASE") == "$LUSTRE_BASE" ]] || {
  echo "reviewed Lustre base is missing or noncanonical" >&2
  exit 2
}
STUDY_PARENT=$LUSTRE_BASE/artifacts/dual_video_diffusion/action_cycle_recoverability
case $STUDY_ROOT in
  "$STUDY_PARENT"/*) ;;
  *) echo "study root must be a strict child of $STUDY_PARENT" >&2; exit 2 ;;
esac
mkdir -p -- "$STUDY_PARENT"
[[ ! -L $STUDY_PARENT && $(realpath -e -- "$STUDY_PARENT") == "$STUDY_PARENT" ]] || {
  echo "study parent is noncanonical" >&2
  exit 2
}
export LACWM_ALLOWED_RUN_ROOTS=$LUSTRE_BASE
# Registration imports both the repository's tools package and VideoX-backed
# tokenizer module. Set the exact reviewed paths before the first tool import;
# ambient PYTHONPATH entries are intentionally discarded.
export PYTHONPATH="$REPO:$VIDEOX_HOME"
COMMIT=$(git -C "$REPO" rev-parse HEAD)
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || {
  echo "probe repository must be clean before registration" >&2
  exit 2
}

"$PYTHON_BIN" "$TOOL" register \
  --repo "$REPO" --expected-commit "$COMMIT" --output-root "$STUDY_ROOT" \
  --lustre-base "$LUSTRE_BASE" \
  --train-metadata "$TRAIN_METADATA" --train-manifest "$TRAIN_MANIFEST" \
  --validation-metadata "$VALIDATION_METADATA" --validation-manifest "$VALIDATION_MANIFEST" \
  --python "$PYTHON_BIN" --videox-home "$VIDEOX_HOME" --wan-dir "$WAN_DIR" >/dev/null
REGISTRATION=$STUDY_ROOT/registration.json

ENCODE_JOB=$(sbatch --parsable --output="$STUDY_ROOT/logs/encode-%j.out" --error="$STUDY_ROOT/logs/encode-%j.err" \
  "$REPO/tools/slurm/action_cycle_recoverability_encode.sbatch" \
  "$REGISTRATION" "$PYTHON_BIN" "$VIDEOX_HOME" "$WAN_DIR")
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterok:$ENCODE_JOB" --kill-on-invalid-dep=yes \
  --output="$STUDY_ROOT/logs/analyze-%j.out" --error="$STUDY_ROOT/logs/analyze-%j.err" \
  "$REPO/tools/slurm/action_cycle_recoverability_analyze.sbatch" "$REGISTRATION" "$PYTHON_BIN")
"$PYTHON_BIN" "$TOOL" record-submission --registration "$REGISTRATION" \
  --encode-job-id "$ENCODE_JOB" --analysis-job-id "$ANALYSIS_JOB" >/dev/null
echo "encode_job=$ENCODE_JOB analysis_job=$ANALYSIS_JOB registration=$REGISTRATION"
