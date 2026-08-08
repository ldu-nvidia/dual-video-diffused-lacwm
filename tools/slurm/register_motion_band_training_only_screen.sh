#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN=""
STUDY_ROOT=""
PARENT_SNAPSHOT=""
TRAIN_MANIFEST=""
TRAIN_METADATA=""
VAL_MANIFEST=""
VAL_METADATA=""
VIDEOX_HOME=""
WAN_DIR=""

die() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="${2:?}"; shift 2 ;;
    --study-root) STUDY_ROOT="${2:?}"; shift 2 ;;
    --parent-snapshot) PARENT_SNAPSHOT="${2:?}"; shift 2 ;;
    --train-manifest) TRAIN_MANIFEST="${2:?}"; shift 2 ;;
    --train-metadata) TRAIN_METADATA="${2:?}"; shift 2 ;;
    --val-manifest) VAL_MANIFEST="${2:?}"; shift 2 ;;
    --val-metadata) VAL_METADATA="${2:?}"; shift 2 ;;
    --videox-home) VIDEOX_HOME="${2:?}"; shift 2 ;;
    --wan-dir) WAN_DIR="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "--python is required"
for value in STUDY_ROOT PARENT_SNAPSHOT TRAIN_MANIFEST TRAIN_METADATA \
  VAL_MANIFEST VAL_METADATA VIDEOX_HOME WAN_DIR; do
  [[ "${!value}" == /* ]] || die "--${value,,} must be absolute"
done
EXPECTED_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || \
  die "repository must be clean"
exec "$PYTHON_BIN" "$REPO_ROOT/tools/motion_band_training_only_screen.py" register \
  --study-root "$STUDY_ROOT" --expected-commit "$EXPECTED_COMMIT" \
  --python "$PYTHON_BIN" --videox-home "$VIDEOX_HOME" --wan-dir "$WAN_DIR" \
  --parent-snapshot "$PARENT_SNAPSHOT" \
  --train-manifest "$TRAIN_MANIFEST" --train-metadata "$TRAIN_METADATA" \
  --val-manifest "$VAL_MANIFEST" --val-metadata "$VAL_METADATA"
