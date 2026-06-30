"""Training-only compatibility surface for VideoX-Fun's Wan transformer.

lacwm uses ordinary PyTorch DDP and never enables VideoX-Fun sequence-parallel
inference. Keeping these symbols local avoids importing every optional xFuser
backend merely to construct the Wan model.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

xFuserLongContextAttention = None


def get_sequence_parallel_rank() -> int:
    return 0


def get_sequence_parallel_world_size() -> int:
    return 1


def get_sp_group():
    raise RuntimeError(
        "VideoX-Fun sequence-parallel inference is disabled in the lacwm "
        "training runtime; use the repository's PyTorch DDP launcher."
    )


def usp_attn_forward(*args, **kwargs):
    raise RuntimeError(
        "VideoX-Fun sequence-parallel inference is disabled in the lacwm "
        "training runtime."
    )
