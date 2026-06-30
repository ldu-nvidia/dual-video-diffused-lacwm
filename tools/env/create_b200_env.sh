#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python3.10}"
readonly RUNTIME_ROOT="${LACWM_BASE:-/mnt/data2/${USER}/lacwm_runtime}"
readonly ENV_DIR="${ENV_DIR:-${RUNTIME_ROOT}/envs/lacwm-b200-py310}"
readonly VIDEOX_DIR="${VIDEOX_HOME:-${RUNTIME_ROOT}/VideoX-Fun-1d6d9c3}"
readonly PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUNTIME_ROOT}/cache/pip}"
readonly UV_CACHE_DIR="${UV_CACHE_DIR:-${RUNTIME_ROOT}/cache/uv}"
readonly UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${RUNTIME_ROOT}/python}"
readonly UV_VERSION="0.10.9"
readonly UV_INSTALLER_SHA256="7fc46e39cb97290b57169c0c813a17970585ac519139f19006453c99b5f2f45f"
readonly UV_INSTALL_DIR="${RUNTIME_ROOT}/tools/uv-${UV_VERSION}"
export PIP_CACHE_DIR
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR

UV_CMD=""
ensure_uv() {
  if command -v uv >/dev/null && [[ "$(uv --version)" == "uv ${UV_VERSION}" ]]; then
    UV_CMD="$(command -v uv)"
  elif [[ -x "${UV_INSTALL_DIR}/uv" ]] && \
    [[ "$("${UV_INSTALL_DIR}/uv" --version)" == "uv ${UV_VERSION}" ]]; then
    UV_CMD="${UV_INSTALL_DIR}/uv"
  else
    mkdir -p "${RUNTIME_ROOT}/tools" "${UV_INSTALL_DIR}"
    installer="${RUNTIME_ROOT}/tools/uv-installer-${UV_VERSION}.sh"
    if [[ ! -f "${installer}" ]] || \
      [[ "$(sha256sum "${installer}" | awk '{print $1}')" != "${UV_INSTALLER_SHA256}" ]]; then
      command -v curl >/dev/null || {
        echo "error: curl is required to bootstrap pinned uv ${UV_VERSION}" >&2
        exit 1
      }
      temporary="${installer}.tmp"
      curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" \
        --output "${temporary}"
      [[ "$(sha256sum "${temporary}" | awk '{print $1}')" == "${UV_INSTALLER_SHA256}" ]] || {
        echo "error: uv installer checksum mismatch" >&2
        exit 1
      }
      mv -f "${temporary}" "${installer}"
    fi
    UV_UNMANAGED_INSTALL="${UV_INSTALL_DIR}" sh "${installer}"
    UV_CMD="${UV_INSTALL_DIR}/uv"
  fi
  [[ "$("${UV_CMD}" --version)" == "uv ${UV_VERSION}" ]] || {
    echo "error: pinned uv ${UV_VERSION} is unavailable" >&2
    exit 1
  }
}

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  mkdir -p \
    "$(dirname "${ENV_DIR}")" \
    "${PIP_CACHE_DIR}" \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}"
  if command -v "${PYTHON_BIN}" >/dev/null && \
    [[ "$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')" == "3.10.20" ]]; then
    "${PYTHON_BIN}" -m venv "${ENV_DIR}"
  else
    ensure_uv
    # uv installs managed CPython under the selected data-volume runtime when
    # the host does not provide the exact patch release.
    "${UV_CMD}" venv --python 3.10.20 --python-preference only-managed --seed "${ENV_DIR}"
  fi
fi

"${ENV_DIR}/bin/python" -m pip install \
  pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
"${ENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/requirements-b200-lock.txt"
"${ENV_DIR}/bin/python" -m pip install --no-deps -e "${REPO_ROOT}"

# PyPI's decord 0.6.0 archive is named as a generic Python 3 wheel but embeds
# an obsolete cp36 tag in its WHEEL metadata. pip therefore reports it as
# unsupported even though its bundled shared library imports on CPython 3.10.
# Keep pip check strict for every real dependency error and explicitly import
# decord when this one known metadata defect is the only complaint.
pip_check_status=0
pip_check_output="$("${ENV_DIR}/bin/python" -m pip check 2>&1)" || pip_check_status=$?
if (( pip_check_status != 0 )); then
  pip_check_remainder="$(
    printf '%s\n' "${pip_check_output}" \
      | grep -vFx 'decord 0.6.0 is not supported on this platform' \
      || true
  )"
  if [[ -n "${pip_check_remainder}" ]]; then
    printf '%s\n' "${pip_check_output}" >&2
    exit "${pip_check_status}"
  fi
  "${ENV_DIR}/bin/python" - <<'PY'
import decord

assert decord.__version__ == "0.6.0", decord.__version__
print("warning: ignoring decord 0.6.0's stale cp36 wheel tag; import succeeded")
PY
else
  printf '%s\n' "${pip_check_output}"
fi

VIDEOX_HOME="${VIDEOX_DIR}" "${SCRIPT_DIR}/prepare_videox_fun.sh"

VIDEOX_HOME="${VIDEOX_DIR}" \
PYTHONPATH="${REPO_ROOT}/tools/env/videox_shim:${VIDEOX_DIR}:${REPO_ROOT}/projects/latent_action_models:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_b200_runtime.py"

echo "Environment ready: ${ENV_DIR}"
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
echo "  source ${SCRIPT_DIR}/activate_b200.sh"
echo "  python ${SCRIPT_DIR}/verify_b200_runtime.py --expected-gpus 8 --require-b200"
