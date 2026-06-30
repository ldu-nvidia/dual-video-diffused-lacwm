#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly MODEL_ID="alibaba-pai/Wan2.1-Fun-1.3B-Control"
readonly MODEL_REVISION="ce96ebd52b1134d2c8a903ceb491ab27aa1e5b7c"
readonly RUNTIME_ROOT="${LACWM_BASE:-/mnt/data2/${USER}/lacwm_runtime}"
readonly ENV_DIR="${ENV_DIR:-${RUNTIME_ROOT}/envs/lacwm-b200-py310}"
readonly WAN_PATH="${WAN_DIR:-${RUNTIME_ROOT}/wan_fun_1.3b_control}"
readonly VIDEOX_PATH="${VIDEOX_HOME:-${RUNTIME_ROOT}/VideoX-Fun-1d6d9c3}"

usage() {
  cat <<'EOF'
Usage: tools/env/prepare_wan_assets.sh --device DEVICE

Downloads the exact Wan DiT, VAE, tokenizer, and umT5 encoder revision used by
lacwm, then creates null_prompt_umt5.pt. DEVICE is explicit because building the
cached prompt briefly loads the umT5 encoder; for example, use cuda:0 or cpu.

If null_prompt_umt5.pt already exists and passes validation, --device is not
required and the large text encoder is not downloaded again.
EOF
}

DEVICE=""
while (($#)); do
  case "$1" in
    --device) [[ $# -ge 2 ]] || { echo "error: --device requires a value" >&2; exit 2; }; DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "${ENV_DIR}/bin/python" ]] || {
  echo "error: environment missing: ${ENV_DIR}; run tools/env/create_b200_env.sh" >&2
  exit 1
}
[[ -d "${VIDEOX_PATH}/.git" ]] || {
  echo "error: pinned VideoX-Fun checkout missing: ${VIDEOX_PATH}" >&2
  exit 1
}

mkdir -p "${WAN_PATH}"

# Verify package/API compatibility and the pinned checkout before any large
# download or text-encoder execution.
VIDEOX_HOME="${VIDEOX_PATH}" \
PYTHONPATH="${REPO_ROOT}/tools/env/videox_shim:${VIDEOX_PATH}:${REPO_ROOT}/projects/latent_action_models:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_b200_runtime.py"

# Core training files are always reconciled to the pinned model revision first.
"${ENV_DIR}/bin/python" - "${WAN_PATH}" "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id=sys.argv[2],
    revision=sys.argv[3],
    local_dir=sys.argv[1],
    allow_patterns=[
        "config.json",
        "configuration.json",
        "diffusion_pytorch_model.safetensors",
        "Wan2.1_VAE.pth",
        "google/umt5-xxl/*",
    ],
)
PY

if VIDEOX_HOME="${VIDEOX_PATH}" \
  PYTHONPATH="${REPO_ROOT}/tools/env/videox_shim:${VIDEOX_PATH}:${REPO_ROOT}/projects/latent_action_models:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_b200_runtime.py" --wan-dir "${WAN_PATH}"; then
  echo "Reusing verified null prompt and provenance."
else
  [[ -n "${DEVICE}" ]] || {
    echo "error: assets or null-prompt provenance are invalid; rerun with --device cuda:N (or cpu) to rebuild the prompt" >&2
    exit 2
  }
  echo "Null prompt is absent or invalid; downloading the pinned umT5 encoder for rebuild."
  "${ENV_DIR}/bin/python" - "${WAN_PATH}" "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id=sys.argv[2],
    revision=sys.argv[3],
    local_dir=sys.argv[1],
    allow_patterns=[
        "config.json",
        "google/umt5-xxl/*",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ],
)
PY

  PYTHONPATH="${REPO_ROOT}/tools/env/videox_shim:${VIDEOX_PATH}:${REPO_ROOT}/projects/latent_action_models:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/build_null_prompt.py" \
      --wan-dir "${WAN_PATH}" \
      --videox-home "${VIDEOX_PATH}" \
      --device "${DEVICE}"
fi

VIDEOX_HOME="${VIDEOX_PATH}" \
PYTHONPATH="${REPO_ROOT}/tools/env/videox_shim:${VIDEOX_PATH}:${REPO_ROOT}/projects/latent_action_models:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify_b200_runtime.py" \
    --wan-dir "${WAN_PATH}"

echo "Wan assets ready: ${WAN_PATH}"
echo "Pinned model revision: ${MODEL_ID}@${MODEL_REVISION}"
