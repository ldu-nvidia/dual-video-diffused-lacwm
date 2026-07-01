#!/usr/bin/env bash
# Source this file after activating the lacwm-b200 Python environment.

_lacwm_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_lacwm_repo="$(cd "${_lacwm_env_dir}/../.." && pwd)"
_lacwm_pin="1d6d9c3e1540968466937129fef4b288041e06de"
_lacwm_base="${LACWM_BASE:-/mnt/data2/${USER}/lacwm_runtime}"

export VIDEOX_HOME="${VIDEOX_HOME:-${_lacwm_base}/VideoX-Fun-${_lacwm_pin:0:7}}"
export WAN_DIR="${WAN_DIR:-${_lacwm_base}/wan_fun_1.3b_control}"
export LACWM_DATA="${LACWM_DATA:-${_lacwm_base}/data}"
export LACWM_RUNS="${LACWM_RUNS:-${_lacwm_base}/runs}"
export LACWM_PYTHON="${LACWM_PYTHON:-$(command -v python)}"

if [[ ! -x "${LACWM_PYTHON}" ]]; then
  echo "error: activate a Python environment before sourcing this file" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${VIDEOX_HOME}/.git" ]]; then
  echo "error: pinned VideoX-Fun checkout missing at ${VIDEOX_HOME}" >&2
  return 1 2>/dev/null || exit 1
fi

_lacwm_actual_pin="$(git -C "${VIDEOX_HOME}" rev-parse HEAD)"
if [[ "${_lacwm_actual_pin}" != "${_lacwm_pin}" ]]; then
  echo "error: VideoX-Fun is ${_lacwm_actual_pin}; expected ${_lacwm_pin}" >&2
  return 1 2>/dev/null || exit 1
fi

# The overlay is first so importing a Wan submodule does not execute VideoX-Fun's
# eager initializers for unrelated image, audio, and sequence-parallel models.
export PYTHONPATH="${_lacwm_repo}/tools/env/videox_shim:${VIDEOX_HOME}:${_lacwm_repo}/projects/latent_action_models:${_lacwm_repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Tested bare-metal B200 RoCE profile. Keep the defaults overridable for a
# future fabric/runtime, while protecting NCCL 2.26.2 from its 32-rank NVLS
# communicator memory failure.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-20}"
export NCCL_IB_SL="${NCCL_IB_SL:-0}"
export NCCL_IB_TC="${NCCL_IB_TC:-52}"
export NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-84}"
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-0}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=rocep145s0:1,rocep146s0:1,rocep152s0:1,rocep153s0:1,rocep198s0:1,rocep199s0:1,rocep205s0:1,rocep206s0:1}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SLURM_TMPDIR:-/tmp}/lacwm-${UID}-cuda-cache-${SLURM_JOB_ID:-manual}}"

unset _lacwm_env_dir _lacwm_repo _lacwm_pin _lacwm_base _lacwm_actual_pin
