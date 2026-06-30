#!/usr/bin/env bash
set -euo pipefail

readonly VIDEOX_REPO="https://github.com/aigc-apps/VideoX-Fun.git"
readonly VIDEOX_COMMIT="1d6d9c3e1540968466937129fef4b288041e06de"
readonly DEFAULT_ROOT="${LACWM_BASE:-/mnt/data2/${USER}/lacwm_runtime}"
readonly DEST="${VIDEOX_HOME:-${DEFAULT_ROOT}/VideoX-Fun-${VIDEOX_COMMIT:0:7}}"

if [[ -e "${DEST}" && ! -d "${DEST}/.git" ]]; then
  echo "error: ${DEST} exists but is not a Git checkout" >&2
  exit 1
fi

if [[ ! -d "${DEST}/.git" ]]; then
  mkdir -p "$(dirname "${DEST}")"
  git clone --filter=blob:none --no-checkout "${VIDEOX_REPO}" "${DEST}"
  git -C "${DEST}" fetch --depth 1 origin "${VIDEOX_COMMIT}"
  git -C "${DEST}" checkout --detach "${VIDEOX_COMMIT}"
fi

actual_commit="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${VIDEOX_COMMIT}" ]]; then
  echo "error: ${DEST} is at ${actual_commit}, expected ${VIDEOX_COMMIT}" >&2
  echo "Use a new VIDEOX_HOME; this script will not switch an existing checkout." >&2
  exit 1
fi

if [[ -n "$(git -C "${DEST}" status --short)" ]]; then
  echo "error: pinned VideoX-Fun checkout is dirty: ${DEST}" >&2
  exit 1
fi

for required in \
  config/wan2.1/wan_civitai.yaml \
  videox_fun/models/wan_transformer3d.py \
  videox_fun/models/wan_vae.py; do
  if [[ ! -f "${DEST}/${required}" ]]; then
    echo "error: pinned checkout lacks ${required}" >&2
    exit 1
  fi
done

echo "VideoX-Fun ready: ${DEST}"
echo "commit: ${actual_commit}"
echo "source tools/env/activate_b200.sh after setting VIDEOX_HOME=${DEST}"
