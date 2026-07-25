"""Foundational components for video/time-frequency dual flow matching.

These modules are intentionally independent of VideoX-Fun so their tensor and
schedule contracts can be tested on CPU before they are connected to Wan.
"""

from .adapters import TFVelocityHead, ZeroInitTFTokenAdapter
from .flow import (
    DualClockBatch,
    DualClockSampler,
    DualFlowCorruption,
    FlowCorruption,
    PairedSigmaSchedule,
    corrupt_dual_flow,
    corrupt_flow,
    derive_tf_sigma,
    euler_flow_step,
    make_paired_sigma_schedule,
    pair_video_sigma_schedule,
)
from .time_frequency import PerViewCausalRFFT, PerViewTemporalSTFT

__all__ = [
    "DualClockBatch",
    "DualClockSampler",
    "DualFlowCorruption",
    "FlowCorruption",
    "PairedSigmaSchedule",
    "PerViewCausalRFFT",
    "PerViewTemporalSTFT",
    "TFVelocityHead",
    "ZeroInitTFTokenAdapter",
    "corrupt_dual_flow",
    "corrupt_flow",
    "derive_tf_sigma",
    "euler_flow_step",
    "make_paired_sigma_schedule",
    "pair_video_sigma_schedule",
]
