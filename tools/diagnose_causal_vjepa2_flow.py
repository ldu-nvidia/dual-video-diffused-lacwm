#!/usr/bin/env python3
"""Diagnose the frozen causal V-JEPA denoiser separately from its sampler.

The completed causal V-JEPA screen established that one autonomous call was
useful, while additional Euler calls moved the semantic state farther from the
target.  This additive evaluator distinguishes two explanations without
changing the frozen trainer or sampler:

* ``training_distribution`` calls receive the exact forward-corrupted validation target
  at a dense clean-time grid.  These calls are deliberately non-deployable and
  measure the denoiser on its training distribution.
* ``autonomous`` calls start from clip-addressed Gaussian noise and receive
  only history RGB, planned actions, and their evolving generated state.

The autonomous API has no clean-target argument. Direct-x is one call only;
Euler uses actual-call budgets 1, 2, 4, and 8; and explicit midpoint uses even
actual-call budgets 2, 4, and 8. Every autonomous path is repeated with the
true actions and the three preregistered episode-disjoint manifest offsets.

Clock convention: clean time ``t=0`` is Gaussian noise and ``t=1`` is clean
data.  No V-JEPA teacher is called by this executable.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_vjepa2_cache_bridge as cache_bridge  # noqa: E402
from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402


SCHEMA = "causal-vjepa2-flow-diagnostic-v1"
FROZEN_TRAINING_SOURCE_COMMIT = "c11487f6e83908687f27026ce2ac2e7d8d41461c"
FROZEN_CHECKPOINT_SHA256 = "f7586f23030a489fc6a673ea3bb6c6cfecccdbe5269c62f6de697d1dc4f9f9cc"
FROZEN_TRAIN_TARGET_SHA256 = "547c4579cf978ac2b9527cb038693259af678a2b07268ab1434706dc128051c4"
FROZEN_VALIDATION_TARGET_SHA256 = "ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034"
FROZEN_PROTOCOL_SOURCE_COMMIT = "be7e76d97543ccc97253e76d1d234abe1c5c4387"
FROZEN_PROTOCOL_SHA256 = "d1d5b22853c598fdfa62984f12975220534341eaf5b6f492d8757a1b3b2a7947"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "VIDEO_LATENT_FORCING_TEMPORAL_FOLLOWUP_PROTOCOL.md"
)
DENSE_CLEAN_TIMES = (
    0.0,
    0.025,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    0.90,
    0.95,
)
NFE_BUDGETS = (1, 2, 4, 8)
UNIFORM_TRACE_NFE_BUDGETS = (2, 4, 8)
MIDPOINT_NFE_BUDGETS = (2, 4, 8)
SOLVERS = ("direct", "euler", "midpoint")
SCHEDULES = ("uniform", "clean_dense")
ACTION_PERMUTATION_OFFSETS = (1, 17, 101)
ACTION_CONTROLS = (
    "matched",
    *(f"offset_{offset:03d}" for offset in ACTION_PERMUTATION_OFFSETS),
)
FORBIDDEN_AUTONOMOUS_ARGUMENT_FRAGMENTS = ("clean", "target", "future")
SEMANTIC_METRIC_NAMES = (
    "semantic_nmse",
    "semantic_token_cosine",
    "temporal_difference_nmse",
    "temporal_difference_token_cosine",
    "retained_utility",
    "temporal_retained_utility",
)
DENSE_TRAJECTORY_METRIC_NAMES = (
    "trajectory_state_nmse",
    "velocity_mse",
    "velocity_nmse",
)
METRIC_NAMES = SEMANTIC_METRIC_NAMES + DENSE_TRAJECTORY_METRIC_NAMES

Solver = Literal["direct", "euler", "midpoint"]
Schedule = Literal["uniform", "clean_dense"]


class DiagnosticError(RuntimeError):
    """A scientific, shape, or provenance contract failed closed."""


def _frozen_protocol_record() -> dict[str, Any]:
    record = vlf.file_record(PROTOCOL_PATH)
    if record.get("sha256") != FROZEN_PROTOCOL_SHA256:
        raise DiagnosticError("the preregistered D0 protocol bytes changed")
    return record


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _batch_time(value: float | Tensor, reference: Tensor) -> Tensor:
    time = torch.as_tensor(value, device=reference.device, dtype=torch.float32)
    if time.ndim == 0:
        time = time.expand(reference.shape[0])
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError(f"clean time must have shape [B={reference.shape[0]}]")
    if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
        raise ValueError("clean time must be finite and lie in [0,1]")
    return time


def _clean_time_velocity_from_x(state: Tensor, prediction: Tensor, time: Tensor) -> Tensor:
    """Convert a clean-state prediction to increasing-clean-time velocity."""
    denominator = (1.0 - vlf.expand_clock(time, state)).clamp_min(
        vlf.FROZEN_CLEAN_TIME_EPS
    )
    return (prediction - state) / denominator


def clean_time_schedule(
    intervals: int,
    kind: Schedule,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return exact noise/data endpoints for a monotone clean-time schedule."""
    if intervals < 1:
        raise ValueError("a schedule requires at least one interval")
    if kind not in SCHEDULES:
        raise ValueError(f"unsupported schedule: {kind}")
    unit = torch.linspace(0.0, 1.0, intervals + 1, device=device, dtype=dtype)
    schedule = unit if kind == "uniform" else 1.0 - (1.0 - unit).square()
    # Preserve endpoint identity independently of dtype and implementation.
    schedule[0] = 0.0
    schedule[-1] = 1.0
    if bool((torch.diff(schedule) <= 0).any()):
        raise DiagnosticError("clean-time schedule is not strictly increasing")
    return schedule


def training_distribution_times(
    dense_times: Sequence[float] = DENSE_CLEAN_TIMES,
) -> tuple[float, ...]:
    """Union the supplemental grid with all preregistered uniform pre-call nodes."""
    values = {float(value) for value in dense_times}
    for calls in UNIFORM_TRACE_NFE_BUDGETS:
        values.update(index / calls for index in range(calls))
    ordered = tuple(sorted(values))
    if not ordered or ordered[0] != 0.0 or ordered[-1] >= 1.0:
        raise DiagnosticError("training-distribution time grid is malformed")
    return ordered


def endpoint_solver_configs() -> tuple[tuple[Solver, Schedule | None, int], ...]:
    """Return the exact nonduplicated endpoint cells frozen in D0."""
    return (
        ("direct", None, 1),
        *(("euler", "uniform", calls) for calls in NFE_BUDGETS),
        # C=1 has identical boundaries to uniform Euler and is excluded.
        *(("euler", "clean_dense", calls) for calls in NFE_BUDGETS if calls > 1),
        *(("midpoint", "uniform", calls) for calls in MIDPOINT_NFE_BUDGETS),
    )


def _per_example_hashes(value: Tensor) -> tuple[str, ...]:
    return screen.tensor_sha256_by_example(value)


class CallAuditor:
    """Hash every student call without accepting a metric target."""

    def __init__(
        self,
        *,
        history: Tensor,
        actions: Tensor,
        video_noise: Tensor,
        auxiliary_noise: Tensor,
        target_entered_model_call: bool = False,
    ) -> None:
        batch = history.shape[0]
        if not (
            actions.shape[0]
            == video_noise.shape[0]
            == auxiliary_noise.shape[0]
            == batch
        ):
            raise ValueError("audited sampler inputs must share their batch dimension")
        self._history = history.clone()
        self._actions = actions.clone()
        self._video_noise = video_noise.clone()
        self._auxiliary_noise = auxiliary_noise.clone()
        self._target_entered_model_call = bool(target_entered_model_call)
        history_hashes = _per_example_hashes(history)
        action_hashes = _per_example_hashes(actions)
        video_hashes = _per_example_hashes(video_noise)
        auxiliary_hashes = _per_example_hashes(auxiliary_noise)
        self._fixed = tuple(
            {
                "history_sha256": history_hashes[index],
                "actions_sha256": action_hashes[index],
                "initial_video_noise_sha256": video_hashes[index],
                "initial_auxiliary_noise_sha256": auxiliary_hashes[index],
            }
            for index in range(batch)
        )
        self._traces: list[list[str]] = [[] for _ in range(batch)]
        self._state_hashes: list[list[str]] = [[] for _ in range(batch)]

    @property
    def calls(self) -> int:
        lengths = {len(trace) for trace in self._traces}
        if len(lengths) != 1:
            raise DiagnosticError("per-example call traces have different lengths")
        return next(iter(lengths))

    @property
    def fixed_records(self) -> tuple[dict[str, str], ...]:
        return self._fixed

    @property
    def state_hash_chains(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(values) for values in self._state_hashes)

    def record(
        self,
        state: Tensor,
        clean_time: Tensor,
        *,
        phase: str,
        solver: str | None,
        schedule: str | None,
    ) -> None:
        state_hashes = _per_example_hashes(state)
        time_hashes = _per_example_hashes(clean_time[:, None])
        call_index = self.calls
        for index, trace in enumerate(self._traces):
            self._state_hashes[index].append(state_hashes[index])
            trace.append(
                _sha256_json(
                    {
                        "schema": f"{SCHEMA}-student-call-v1",
                        "call_index": call_index,
                        "phase": phase,
                        "solver": solver,
                        "schedule": schedule,
                        "clean_time_sha256": time_hashes[index],
                        "noisy_auxiliary_sha256": state_hashes[index],
                        **self._fixed[index],
                        "teacher_model_calls": 0,
                        "clean_future_target_entered_call": self._target_entered_model_call,
                    }
                )
            )

    def trace_digests(self) -> tuple[str, ...]:
        return tuple(
            _sha256_json(
                {
                    "schema": f"{SCHEMA}-student-call-trace-v1",
                    "calls": trace,
                }
            )
            for trace in self._traces
        )

    def assert_inputs_unchanged(
        self,
        *,
        history: Tensor,
        actions: Tensor,
        video_noise: Tensor,
        auxiliary_noise: Tensor,
    ) -> None:
        if not (
            torch.equal(history, self._history)
            and torch.equal(actions, self._actions)
            and torch.equal(video_noise, self._video_noise)
            and torch.equal(auxiliary_noise, self._auxiliary_noise)
        ):
            raise DiagnosticError("an immutable autonomous sampler input changed")


def _predict_clean(
    model: nn.Module,
    *,
    state: Tensor,
    clean_time: Tensor,
    history: Tensor,
    actions: Tensor,
    video_noise: Tensor,
    auditor: CallAuditor,
    phase: str,
    solver: str | None = None,
    schedule: str | None = None,
) -> Tensor:
    """Make one audited student call; no clean target is accepted here."""
    time = _batch_time(clean_time, state)
    auditor.record(
        state,
        time,
        phase=phase,
        solver=solver,
        schedule=schedule,
    )
    with vlf._autocast(state.device):  # noqa: SLF001 - frozen runtime primitive
        _, prediction = vlf.model_forward(
            model,
            noisy_video=video_noise,
            noisy_auxiliary=state,
            t_video=torch.zeros_like(time),
            t_auxiliary=time,
            history=history,
            actions=actions,
            condition_on_auxiliary=True,
            predict_video=False,
        )
    if prediction.shape != state.shape:
        raise DiagnosticError("student clean prediction changed auxiliary shape")
    if not bool(torch.isfinite(prediction).all()):
        raise DiagnosticError("student clean prediction is nonfinite")
    return prediction.float()


@dataclass(frozen=True)
class DensePoint:
    clean_time: float
    state: Tensor
    clean_prediction: Tensor
    model_calls_so_far: int
    call_trace_sha256_by_example: tuple[str, ...]
    state_sha256_chain_by_example: tuple[tuple[str, ...], ...]
    fixed_input_records: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class SolverSample:
    prediction: Tensor
    model_calls: int
    integration_intervals: int
    call_trace_sha256_by_example: tuple[str, ...]
    state_sha256_chain_by_example: tuple[tuple[str, ...], ...]
    fixed_input_records: tuple[dict[str, str], ...]
    pre_call_points: tuple[DensePoint, ...]


@dataclass(frozen=True)
class ActionVariant:
    """One factual action source registered by manifest offset."""

    control: str
    manifest_offset: int
    actions: Tensor
    source_clip_ids: tuple[str, ...]
    source_episode_indices: tuple[int, ...]


def predict_training_distribution(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    clean_target: Tensor,
    *,
    time_grid: Sequence[float] = DENSE_CLEAN_TIMES,
) -> list[DensePoint]:
    """Evaluate target-derived training corruptions (nondeployable diagnostic)."""
    if clean_target.shape != auxiliary_noise.shape:
        raise ValueError("clean target and auxiliary noise must have identical shape")
    points: list[DensePoint] = []
    for value in time_grid:
        time = _batch_time(float(value), clean_target)
        state = vlf.corrupt_clean_time(clean_target, auxiliary_noise, time)
        auditor = CallAuditor(
            history=history,
            actions=actions,
            video_noise=video_noise,
            auxiliary_noise=auxiliary_noise,
            target_entered_model_call=True,
        )
        prediction = _predict_clean(
            model,
            state=state,
            clean_time=time,
            history=history,
            actions=actions,
            video_noise=video_noise,
            auditor=auditor,
            phase="training_distribution",
        )
        auditor.assert_inputs_unchanged(
            history=history,
            actions=actions,
            video_noise=video_noise,
            auxiliary_noise=auxiliary_noise,
        )
        points.append(
            DensePoint(
                clean_time=float(value),
                state=state.float(),
                clean_prediction=prediction,
                model_calls_so_far=auditor.calls,
                call_trace_sha256_by_example=auditor.trace_digests(),
                state_sha256_chain_by_example=auditor.state_hash_chains,
                fixed_input_records=auditor.fixed_records,
            )
        )
    return points


def sample_autonomous_dense(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    *,
    time_grid: Sequence[float] = DENSE_CLEAN_TIMES,
) -> list[DensePoint]:
    """Trace Euler states using deployable inputs only.

    Deliberately note the absence of any clean/future/target argument.  The
    clean target is available only to the caller after this function returns.
    """
    values = tuple(float(value) for value in time_grid)
    if not values or values[0] != 0.0 or any(
        right <= left for left, right in zip(values, values[1:], strict=False)
    ) or values[-1] >= 1.0:
        raise ValueError("autonomous dense times must increase from 0 and stop below 1")
    state = auxiliary_noise.clone().float()
    auditor = CallAuditor(
        history=history,
        actions=actions,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
    )
    points: list[DensePoint] = []
    for index, value in enumerate(values):
        time = _batch_time(value, state)
        prediction = _predict_clean(
            model,
            state=state,
            clean_time=time,
            history=history,
            actions=actions,
            video_noise=video_noise,
            auditor=auditor,
            phase="autonomous_dense",
            solver="euler",
            schedule="registered_dense",
        )
        points.append(
            DensePoint(
                clean_time=value,
                state=state.clone(),
                clean_prediction=prediction.clone(),
                model_calls_so_far=auditor.calls,
                call_trace_sha256_by_example=auditor.trace_digests(),
                state_sha256_chain_by_example=auditor.state_hash_chains,
                fixed_input_records=auditor.fixed_records,
            )
        )
        if index + 1 < len(values):
            next_time = _batch_time(values[index + 1], state)
            state = vlf.clean_time_euler_from_x(state, prediction, time, next_time)
    auditor.assert_inputs_unchanged(
        history=history,
        actions=actions,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
    )
    return points


def sample_autonomous_solver(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    *,
    solver: Solver,
    schedule: Schedule | None,
    nfe: int,
) -> SolverSample:
    """Generate one semantic state without accepting any clean-future tensor."""
    if solver not in SOLVERS:
        raise ValueError(f"unsupported solver: {solver}")
    if nfe not in NFE_BUDGETS:
        raise ValueError(f"NFE must be one of {NFE_BUDGETS}")
    if solver == "direct":
        if nfe != 1 or schedule is not None:
            raise ValueError("direct-x is exactly one call and has no schedule label")
    elif solver == "euler":
        if schedule not in SCHEDULES:
            raise ValueError("Euler requires a registered schedule")
        if schedule == "clean_dense" and nfe == 1:
            raise ValueError("duplicate one-call dense Euler is excluded")
    elif schedule != "uniform" or nfe not in MIDPOINT_NFE_BUDGETS:
        raise ValueError("midpoint is uniform-only with even call budgets 2, 4, and 8")

    state = auxiliary_noise.clone().float()
    auditor = CallAuditor(
        history=history,
        actions=actions,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
    )
    intervals = nfe if solver in {"direct", "euler"} else nfe // 2
    nodes = (
        torch.tensor((0.0, 1.0), device=state.device, dtype=torch.float32)
        if solver == "direct"
        else clean_time_schedule(
            intervals,
            schedule,  # type: ignore[arg-type]
            device=state.device,
            dtype=torch.float32,
        )
    )
    pre_call_points: list[DensePoint] = []

    for index in range(intervals):
        time = _batch_time(nodes[index], state)
        next_time = _batch_time(nodes[index + 1], state)
        first = _predict_clean(
            model,
            state=state,
            clean_time=time,
            history=history,
            actions=actions,
            video_noise=video_noise,
            auditor=auditor,
            phase="autonomous_sampler",
            solver=solver,
            schedule=schedule,
        )
        if solver == "euler":
            pre_call_points.append(
                DensePoint(
                    clean_time=float(nodes[index]),
                    state=state.clone(),
                    clean_prediction=first.clone(),
                    model_calls_so_far=auditor.calls,
                    call_trace_sha256_by_example=auditor.trace_digests(),
                    state_sha256_chain_by_example=auditor.state_hash_chains,
                    fixed_input_records=auditor.fixed_records,
                )
            )
        if solver == "direct":
            state = first
        elif solver == "euler":
            state = vlf.clean_time_euler_from_x(state, first, time, next_time)
        else:
            midpoint_time = 0.5 * (time + next_time)
            first_velocity = _clean_time_velocity_from_x(state, first, time)
            midpoint_state = state + vlf.expand_clock(
                midpoint_time - time, state
            ) * first_velocity
            midpoint_prediction = _predict_clean(
                model,
                state=midpoint_state,
                clean_time=midpoint_time,
                history=history,
                actions=actions,
                video_noise=video_noise,
                auditor=auditor,
                phase="autonomous_sampler_midpoint",
                solver=solver,
                schedule=schedule,
            )
            midpoint_velocity = _clean_time_velocity_from_x(
                midpoint_state,
                midpoint_prediction,
                midpoint_time,
            )
            state = state + vlf.expand_clock(next_time - time, state) * midpoint_velocity

    if auditor.calls != nfe:
        raise DiagnosticError(
            f"{solver}/{schedule} used {auditor.calls} calls for NFE={nfe}"
        )
    auditor.assert_inputs_unchanged(
        history=history,
        actions=actions,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
    )
    return SolverSample(
        prediction=state,
        model_calls=auditor.calls,
        integration_intervals=intervals,
        call_trace_sha256_by_example=auditor.trace_digests(),
        state_sha256_chain_by_example=auditor.state_hash_chains,
        fixed_input_records=auditor.fixed_records,
        pre_call_points=tuple(pre_call_points),
    )


def assert_autonomous_api_has_no_clean_future() -> dict[str, list[str]]:
    """Fail if either deployable sampler gains a target-like argument."""
    result: dict[str, list[str]] = {}
    for function in (sample_autonomous_dense, sample_autonomous_solver):
        names = list(inspect.signature(function).parameters)
        forbidden = [
            name
            for name in names
            if any(fragment in name.lower() for fragment in FORBIDDEN_AUTONOMOUS_ARGUMENT_FRAGMENTS)
        ]
        if forbidden:
            raise DiagnosticError(
                f"autonomous API {function.__name__} accepts forbidden arguments: {forbidden}"
            )
        result[function.__name__] = names
    return result


def _validate_action_variants(
    variants: Sequence[ActionVariant],
    *,
    actions: Tensor,
    clip_ids: Sequence[str],
    episode_indices: Sequence[int],
) -> tuple[ActionVariant, ...]:
    by_control = {variant.control: variant for variant in variants}
    if tuple(by_control) != ACTION_CONTROLS or len(by_control) != len(variants):
        raise DiagnosticError(
            f"action variants must be ordered exactly as {ACTION_CONTROLS}"
        )
    batch = actions.shape[0]
    for variant in variants:
        expected_offset = (
            0
            if variant.control == "matched"
            else int(variant.control.removeprefix("offset_"))
        )
        if variant.manifest_offset != expected_offset:
            raise DiagnosticError("action variant label/offset mismatch")
        if (
            variant.actions.shape != actions.shape
            or len(variant.source_clip_ids) != batch
            or len(variant.source_episode_indices) != batch
        ):
            raise DiagnosticError("action variant batch metadata is malformed")
        if not bool(torch.isfinite(variant.actions).all()):
            raise DiagnosticError("action variant contains nonfinite values")
        if variant.control == "matched":
            if (
                not torch.equal(variant.actions, actions)
                or variant.source_clip_ids != tuple(str(value) for value in clip_ids)
                or variant.source_episode_indices
                != tuple(int(value) for value in episode_indices)
            ):
                raise DiagnosticError("matched actions do not preserve the source batch")
        elif any(
            int(source) == int(destination)
            for source, destination in zip(
                variant.source_episode_indices, episode_indices, strict=True
            )
        ):
            raise DiagnosticError("permuted actions are not episode-disjoint")
    return tuple(variants)


def _assert_target_absent_from_autonomous_chain(
    *,
    target: Tensor,
    state_hash_chains: tuple[tuple[str, ...], ...],
    fixed_input_records: tuple[dict[str, str], ...],
) -> None:
    target_hashes = _per_example_hashes(target)
    for item, target_hash in enumerate(target_hashes):
        if target_hash in state_hash_chains[item] or target_hash in fixed_input_records[
            item
        ].values():
            raise DiagnosticError("metric-target hash entered an autonomous input chain")


def _assert_finite_metric_mapping(values: Mapping[str, Tensor]) -> None:
    for name, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise DiagnosticError(f"applicable metric {name} is nonfinite")


def _metric_values(
    prediction: Tensor,
    target: Tensor,
    *,
    state: Tensor | None = None,
    training_distribution_state: Tensor | None = None,
    clean_time: float | None = None,
) -> dict[str, Tensor]:
    metrics = dict(screen.semantic_metrics(prediction, target))
    if state is None or training_distribution_state is None or clean_time is None:
        _assert_finite_metric_mapping(metrics)
        return metrics
    target_power = target.float().square().flatten(1).mean(1).clamp_min(1e-12)
    state_error = (
        state.float() - training_distribution_state.float()
    ).square().flatten(1).mean(1)
    time = _batch_time(clean_time, state)
    denominator = (1.0 - vlf.expand_clock(time, state)).clamp_min(
        vlf.FROZEN_CLEAN_TIME_EPS
    )
    predicted_velocity = (prediction.float() - state.float()) / denominator
    target_velocity = (target.float() - state.float()) / denominator
    velocity_error = (
        predicted_velocity - target_velocity
    ).square().flatten(1).mean(1)
    velocity_power = target_velocity.square().flatten(1).mean(1).clamp_min(1e-12)
    metrics.update(
        trajectory_state_nmse=state_error / target_power,
        velocity_mse=velocity_error,
        velocity_nmse=velocity_error / velocity_power,
    )
    _assert_finite_metric_mapping(metrics)
    return metrics


def _row_shell() -> dict[str, Any]:
    """Return a complete JSON schema whose inapplicable values stay null."""
    return {
        "solver": None,
        "schedule": None,
        "nfe": None,
        "actual_model_calls": None,
        "nominal_intervals": None,
        "clean_time": None,
        "call_index": None,
        "model_calls_so_far": None,
        "sampler_input_sha256": None,
        "autonomous_state_sha256_chain": None,
        **{name: None for name in METRIC_NAMES},
    }


def _rows_from_trajectory_points(
    points: Sequence[DensePoint],
    *,
    state_source: str,
    schedule: str | None,
    trajectory_calls: int | None,
    nominal_intervals: int | None,
    action_variant: ActionVariant,
    target: Tensor,
    auxiliary_noise: Tensor,
    clip_ids: Sequence[str],
    episode_indices: Sequence[int],
    supplemental: bool,
) -> list[dict[str, Any]]:
    deployable = state_source != "training_distribution"
    target_hashes = _per_example_hashes(target)
    if deployable and points:
        _assert_target_absent_from_autonomous_chain(
            target=target,
            state_hash_chains=points[-1].state_sha256_chain_by_example,
            fixed_input_records=points[-1].fixed_input_records,
        )
    rows: list[dict[str, Any]] = []
    for point in points:
        time = _batch_time(point.clean_time, target)
        reference_state = vlf.corrupt_clean_time(target, auxiliary_noise, time)
        tensors = _metric_values(
            point.clean_prediction,
            target,
            state=point.state,
            training_distribution_state=reference_state,
            clean_time=point.clean_time,
        )
        values = {
            name: tensor.detach().cpu().tolist() for name, tensor in tensors.items()
        }
        prediction_hashes = _per_example_hashes(point.clean_prediction)
        state_hashes = _per_example_hashes(point.state)
        uniform_grid_member = any(
            point.clean_time == index / calls
            for calls in UNIFORM_TRACE_NFE_BUDGETS
            for index in range(calls)
        )
        registered_dense_grid_member = point.clean_time in DENSE_CLEAN_TIMES
        primary_cell = (
            action_variant.control == "matched"
            and point.clean_time == 0.25
            and (
                state_source == "training_distribution"
                or (
                    state_source == "autonomous_uniform_euler"
                    and trajectory_calls == 4
                )
            )
        )
        for item, clip_id in enumerate(clip_ids):
            row = {
                **_row_shell(),
                "schema": SCHEMA,
                "family": "trajectory",
                "state_source": state_source,
                "solver": None if not deployable else "euler",
                "schedule": schedule,
                "nfe": trajectory_calls,
                "actual_model_calls": trajectory_calls,
                "nominal_intervals": nominal_intervals,
                "action_control": action_variant.control,
                "action_manifest_offset": action_variant.manifest_offset,
                "action_source_clip_id": action_variant.source_clip_ids[item],
                "action_source_episode_index": action_variant.source_episode_indices[
                    item
                ],
                "clip_id": str(clip_id),
                "episode_index": int(episode_indices[item]),
                "clean_time": point.clean_time,
                "call_index": point.model_calls_so_far - 1,
                "model_calls_so_far": point.model_calls_so_far,
                "student_model_calls_for_point": 1,
                "teacher_model_calls": 0,
                "deployable": deployable,
                "supplemental": (
                    supplemental
                    or (
                        state_source == "training_distribution"
                        and not uniform_grid_member
                    )
                ),
                "primary_diagnosis_cell": primary_cell,
                "uniform_trace_grid_member": uniform_grid_member,
                "registered_dense_grid_member": registered_dense_grid_member,
                "target_derived_state_entered_model": not deployable,
                "clean_future_target_entered_model_call": not deployable,
                "clean_future_target_entered_autonomous_sampler": False,
                "call_trace_sha256": point.call_trace_sha256_by_example[item],
                "state_sha256": state_hashes[item],
                "prediction_sha256": prediction_hashes[item],
                "metric_target_sha256": target_hashes[item],
                **point.fixed_input_records[item],
                "sampler_input_sha256": (
                    screen.sampler_input_sha256(point.fixed_input_records[item])
                    if deployable
                    else None
                ),
                "autonomous_state_sha256_chain": (
                    list(point.state_sha256_chain_by_example[item])
                    if deployable
                    else None
                ),
            }
            row.update({name: float(values[name][item]) for name in METRIC_NAMES})
            rows.append(row)
    return rows


def _rows_from_solver_sample(
    sample: SolverSample,
    *,
    solver: str,
    schedule: str | None,
    nfe: int,
    action_variant: ActionVariant,
    target: Tensor,
    clip_ids: Sequence[str],
    episode_indices: Sequence[int],
) -> list[dict[str, Any]]:
    _assert_target_absent_from_autonomous_chain(
        target=target,
        state_hash_chains=sample.state_sha256_chain_by_example,
        fixed_input_records=sample.fixed_input_records,
    )
    tensors = _metric_values(sample.prediction, target)
    values = {
        name: tensor.detach().cpu().tolist() for name, tensor in tensors.items()
    }
    target_hashes = _per_example_hashes(target)
    prediction_hashes = _per_example_hashes(sample.prediction)
    rows = []
    for item, clip_id in enumerate(clip_ids):
        row = {
            **_row_shell(),
            "schema": SCHEMA,
            "family": "endpoint",
            "state_source": "autonomous",
            "solver": solver,
            "schedule": schedule,
            "nfe": nfe,
            "actual_model_calls": sample.model_calls,
            "nominal_intervals": sample.integration_intervals,
            "action_control": action_variant.control,
            "action_manifest_offset": action_variant.manifest_offset,
            "action_source_clip_id": action_variant.source_clip_ids[item],
            "action_source_episode_index": action_variant.source_episode_indices[item],
            "clip_id": str(clip_id),
            "episode_index": int(episode_indices[item]),
            "teacher_model_calls": 0,
            "deployable": True,
            "supplemental": schedule == "clean_dense",
            "primary_diagnosis_cell": False,
            "target_derived_state_entered_model": False,
            "clean_future_target_entered_model_call": False,
            "clean_future_target_entered_autonomous_sampler": False,
            "call_trace_sha256": sample.call_trace_sha256_by_example[item],
            **sample.fixed_input_records[item],
            "sampler_input_sha256": screen.sampler_input_sha256(
                sample.fixed_input_records[item]
            ),
            "autonomous_state_sha256_chain": list(
                sample.state_sha256_chain_by_example[item]
            ),
            "prediction_sha256": prediction_hashes[item],
            "metric_target_sha256": target_hashes[item],
        }
        row.update(
            {name: float(values[name][item]) for name in SEMANTIC_METRIC_NAMES}
        )
        rows.append(row)
    return rows


@dataclass(frozen=True)
class BatchEvaluation:
    records: tuple[dict[str, Any], ...]
    actual_batched_student_calls: int
    t0_training_autonomous_bit_identical: bool


def evaluate_batch(
    model: nn.Module,
    *,
    history: Tensor,
    actions: Tensor,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    target: Tensor,
    clip_ids: Sequence[str],
    episode_indices: Sequence[int],
    action_variants: Sequence[ActionVariant],
    dense_clean_times: Sequence[float] = DENSE_CLEAN_TIMES,
) -> BatchEvaluation:
    """Run all paired paths; target is used only outside autonomous APIs."""
    batch = target.shape[0]
    if len(clip_ids) != batch or len(episode_indices) != batch:
        raise ValueError("clip metadata does not match batch")
    variants = _validate_action_variants(
        action_variants,
        actions=actions,
        clip_ids=clip_ids,
        episode_indices=episode_indices,
    )
    records: list[dict[str, Any]] = []
    actual_calls = 0
    teacher_points = predict_training_distribution(
        model,
        history,
        actions,
        video_noise,
        auxiliary_noise,
        target,
        time_grid=training_distribution_times(dense_clean_times),
    )
    actual_calls += len(teacher_points)
    records.extend(
        _rows_from_trajectory_points(
            teacher_points,
            state_source="training_distribution",
            schedule=None,
            trajectory_calls=None,
            nominal_intervals=None,
            action_variant=variants[0],
            target=target,
            auxiliary_noise=auxiliary_noise,
            clip_ids=clip_ids,
            episode_indices=episode_indices,
            supplemental=False,
        )
    )
    teacher_zero = next(point for point in teacher_points if point.clean_time == 0.0)
    if not torch.equal(teacher_zero.state, auxiliary_noise.float()):
        raise DiagnosticError("t=0 training-distribution state is not exact fixed noise")

    primary_autonomous_zero: DensePoint | None = None
    for variant in variants:
        dense_trace = sample_autonomous_dense(
            model,
            history,
            variant.actions,
            video_noise,
            auxiliary_noise,
            time_grid=dense_clean_times,
        )
        actual_calls += len(dense_trace)
        records.extend(
            _rows_from_trajectory_points(
                dense_trace,
                state_source="autonomous_registered_dense",
                schedule="registered_dense",
                trajectory_calls=len(dense_clean_times),
                nominal_intervals=max(len(dense_clean_times) - 1, 0),
                action_variant=variant,
                target=target,
                auxiliary_noise=auxiliary_noise,
                clip_ids=clip_ids,
                episode_indices=episode_indices,
                supplemental=True,
            )
        )
        for solver, schedule, nfe in endpoint_solver_configs():
            sample = sample_autonomous_solver(
                model,
                history,
                variant.actions,
                video_noise,
                auxiliary_noise,
                solver=solver,
                schedule=schedule,
                nfe=nfe,
            )
            actual_calls += sample.model_calls
            records.extend(
                _rows_from_solver_sample(
                    sample,
                    solver=solver,
                    schedule=schedule,
                    nfe=nfe,
                    action_variant=variant,
                    target=target,
                    clip_ids=clip_ids,
                    episode_indices=episode_indices,
                )
            )
            if (
                solver == "euler"
                and schedule == "uniform"
                and nfe in UNIFORM_TRACE_NFE_BUDGETS
            ):
                records.extend(
                    _rows_from_trajectory_points(
                        sample.pre_call_points,
                        state_source="autonomous_uniform_euler",
                        schedule="uniform",
                        trajectory_calls=nfe,
                        nominal_intervals=nfe,
                        action_variant=variant,
                        target=target,
                        auxiliary_noise=auxiliary_noise,
                        clip_ids=clip_ids,
                        episode_indices=episode_indices,
                        supplemental=False,
                    )
                )
                if variant.control == "matched" and nfe == 4:
                    primary_autonomous_zero = sample.pre_call_points[0]

    if primary_autonomous_zero is None or not (
        torch.equal(teacher_zero.state, primary_autonomous_zero.state)
        and torch.equal(
            teacher_zero.clean_prediction,
            primary_autonomous_zero.clean_prediction,
        )
    ):
        raise DiagnosticError(
            "t=0 training-distribution and autonomous state/prediction are not bit-identical"
        )
    return BatchEvaluation(tuple(records), actual_calls, True)


def expected_rows_per_clip(
    *,
    dense_clean_times: Sequence[float] = DENSE_CLEAN_TIMES,
) -> int:
    return len(training_distribution_times(dense_clean_times)) + len(
        ACTION_CONTROLS
    ) * (
        len(dense_clean_times)
        + len(endpoint_solver_configs())
        + sum(UNIFORM_TRACE_NFE_BUDGETS)
    )


def expected_batched_calls(
    *,
    dense_clean_times: Sequence[float] = DENSE_CLEAN_TIMES,
) -> int:
    return len(training_distribution_times(dense_clean_times)) + len(
        ACTION_CONTROLS
    ) * (
        len(dense_clean_times)
        + sum(nfe for _, _, nfe in endpoint_solver_configs())
    )


def _summary_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    if record["family"] == "trajectory":
        return (
            "trajectory",
            record["state_source"],
            record["schedule"],
            record["nfe"],
            record["action_control"],
            float(record["clean_time"]),
        )
    return (
        "endpoint",
        record["solver"],
        record["schedule"],
        int(record["nfe"]),
        record["action_control"],
    )


def summarize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_summary_key(record), []).append(record)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        first = group[0]
        identity = (
            {
                "family": "trajectory",
                "state_source": first["state_source"],
                "schedule": first["schedule"],
                "nfe": first["nfe"],
                "action_control": first["action_control"],
                "clean_time": float(first["clean_time"]),
            }
            if first["family"] == "trajectory"
            else {
                "family": "endpoint",
                "solver": first["solver"],
                "schedule": first["schedule"],
                "nfe": int(first["nfe"]),
                "action_control": first["action_control"],
            }
        )
        summary = {**identity, "clips": len(group)}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group
                    if row.get(metric) is not None
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            summary[metric] = float(finite.mean()) if finite.size else None
            summary[f"{metric}_finite_clips"] = int(finite.size)
        summaries.append(summary)
    return summaries


def validate_record_contract(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject leakage flags, nonfinite metrics, or non-null inapplicable fields."""
    for row in records:
        if row.get("teacher_model_calls") != 0:
            raise DiagnosticError("a D0 row reports a V-JEPA teacher call")
        deployable = row.get("deployable") is True
        if row.get("clean_future_target_entered_autonomous_sampler") is not False:
            raise DiagnosticError("an autonomous sampler reports clean-target entry")
        if deployable and (
            row.get("target_derived_state_entered_model") is not False
            or row.get("clean_future_target_entered_model_call") is not False
        ):
            raise DiagnosticError("a deployable model call reports target-derived input")
        if not deployable and row.get("state_source") == "training_distribution":
            if (
                row.get("target_derived_state_entered_model") is not True
                or row.get("clean_future_target_entered_model_call") is not True
            ):
                raise DiagnosticError("training-distribution target entry is unmarked")
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if value is not None and not np.isfinite(float(value)):
                raise DiagnosticError(f"row contains nonfinite metric {metric}")
        if row.get("family") == "endpoint" and any(
            row.get(metric) is not None for metric in DENSE_TRAJECTORY_METRIC_NAMES
        ):
            raise DiagnosticError("endpoint row populated an inapplicable trajectory metric")
        chain = row.get("autonomous_state_sha256_chain")
        if deployable:
            fixed_hashes = {
                row.get("history_sha256"),
                row.get("actions_sha256"),
                row.get("initial_video_noise_sha256"),
                row.get("initial_auxiliary_noise_sha256"),
            }
            if (
                not isinstance(chain, list)
                or row.get("metric_target_sha256") in chain
                or row.get("metric_target_sha256") in fixed_hashes
            ):
                raise DiagnosticError("autonomous target/hash-chain contract failed")
        elif chain is not None or row.get("sampler_input_sha256") is not None:
            raise DiagnosticError("training-distribution row has autonomous-only hashes")


def primary_diagnosis(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 20_260_807,
) -> dict[str, Any]:
    """Apply the preregistered paired D0 rollout-drift classification."""
    teacher = {
        str(row["clip_id"]): float(row["temporal_difference_nmse"])
        for row in records
        if row.get("family") == "trajectory"
        and row.get("state_source") == "training_distribution"
        and row.get("action_control") == "matched"
        and float(row.get("clean_time", -1.0)) == 0.25
    }
    autonomous = {
        str(row["clip_id"]): float(row["temporal_difference_nmse"])
        for row in records
        if row.get("family") == "trajectory"
        and row.get("state_source") == "autonomous_uniform_euler"
        and row.get("action_control") == "matched"
        and row.get("nfe") == 4
        and float(row.get("clean_time", -1.0)) == 0.25
    }
    if not teacher or set(teacher) != set(autonomous):
        raise DiagnosticError("primary paired D0 cell is incomplete")
    clip_ids = sorted(teacher)
    teacher_values = np.asarray([teacher[key] for key in clip_ids], dtype=np.float64)
    autonomous_values = np.asarray(
        [autonomous[key] for key in clip_ids], dtype=np.float64
    )
    if not np.isfinite(teacher_values).all() or not np.isfinite(
        autonomous_values
    ).all():
        raise DiagnosticError("primary D0 inputs are nonfinite")
    teacher_mean = float(teacher_values.mean())
    autonomous_mean = float(autonomous_values.mean())
    denominator = max(teacher_mean, 1e-12)
    relative_worsening = autonomous_mean / denominator - 1.0
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_resamples, dtype=np.float64)
    for begin in range(0, bootstrap_resamples, 512):
        end = min(begin + 512, bootstrap_resamples)
        indexes = generator.integers(
            0, len(clip_ids), size=(end - begin, len(clip_ids))
        )
        teacher_means = teacher_values[indexes].mean(axis=1)
        autonomous_means = autonomous_values[indexes].mean(axis=1)
        bootstrap[begin:end] = autonomous_means / np.maximum(
            teacher_means, 1e-12
        ) - 1.0
    lower, upper = np.quantile(bootstrap, (0.05, 0.95)).tolist()
    rollout_drift_primary = (
        teacher_mean <= 0.50 and relative_worsening >= 0.25 and lower >= 0.25
    )
    return {
        "schema": f"{SCHEMA}-primary-diagnosis-v1",
        "cell": {
            "trajectory": "uniform_euler",
            "actual_calls": 4,
            "pre_call_index": 1,
            "clean_time": 0.25,
            "action_control": "matched",
        },
        "paired_clips": len(clip_ids),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "training_distribution_temporal_nmse": teacher_mean,
        "autonomous_temporal_nmse": autonomous_mean,
        "autonomous_relative_worsening": relative_worsening,
        "relative_worsening_one_sided_lower_bound_95": float(lower),
        "relative_worsening_bootstrap_quantiles_05_95": [
            float(lower),
            float(upper),
        ],
        "rollout_drift_primary": rollout_drift_primary,
        "classification": (
            "rollout_drift_primary"
            if rollout_drift_primary
            else "d0_does_not_identify_a_unique_primary_cause"
        ),
    }


def _validate_checkpoint(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_config: Mapping[str, Any],
    cache_metadata: Mapping[str, Any],
    producer_attestation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Strict-load the frozen endpoint from a clean descendant checkout."""
    current_source = vlf.git_record()
    if current_source.get("dirty") is not False:
        raise DiagnosticError("production diagnostics require clean committed source")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_TRAINING_SOURCE_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode:
        raise DiagnosticError("diagnostic source is not descended from the frozen trainer")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.parent.name != "checkpoints":
        raise DiagnosticError("checkpoint must remain in its immutable checkpoints directory")
    config_path = checkpoint_path.parent.parent / "resolved_config.json"
    complete_path = checkpoint_path.parent.parent / "complete.json"
    checkpoint_record = vlf.file_record(checkpoint_path)
    if checkpoint_record.get("sha256") != FROZEN_CHECKPOINT_SHA256:
        raise DiagnosticError("checkpoint differs from the preregistered update-5,000 EMA")
    training_config = vlf.load_json(config_path, "semantic training config")
    complete = vlf.load_json(complete_path, "semantic training completion")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    datasets = training_config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise DiagnosticError("training config lacks dataset evidence")
    caches = datasets.get("semantic_cache")
    if not isinstance(caches, Mapping):
        raise DiagnosticError("training config lacks semantic-cache evidence")
    train_cache = caches.get("train")
    validation_cache = caches.get("validation")
    if (
        not isinstance(train_cache, Mapping)
        or train_cache.get("target_sha256") != FROZEN_TRAIN_TARGET_SHA256
        or not isinstance(validation_cache, Mapping)
        or validation_cache.get("target_sha256")
        != FROZEN_VALIDATION_TARGET_SHA256
        or cache_metadata.get("target_sha256")
        != FROZEN_VALIDATION_TARGET_SHA256
    ):
        raise DiagnosticError("semantic target bytes differ from the preregistration")
    screen._validate_training_cache_pair(  # noqa: SLF001 - frozen validator
        caches.get("train", {}),
        caches.get("validation", {}),
        train_manifest=datasets.get("train", {}),
        validation_manifest=datasets.get("validation", {}),
    )
    entrypoint = training_config.get("entrypoint")
    dataset_source = training_config.get("dataset_source")
    current_entrypoint = vlf.file_record(screen.__file__)
    producer = producer_attestation.get("producer")
    producer_dataset_source = (
        producer.get("dataset_source") if isinstance(producer, Mapping) else None
    )
    source = training_config.get("source")
    if (
        not isinstance(entrypoint, Mapping)
        or entrypoint.get("sha256") != current_entrypoint["sha256"]
        or entrypoint.get("bytes") != current_entrypoint["bytes"]
        or not isinstance(dataset_source, Mapping)
        or not isinstance(producer_dataset_source, Mapping)
        or dataset_source != producer_dataset_source
        or not isinstance(source, Mapping)
        or source.get("commit") != FROZEN_TRAINING_SOURCE_COMMIT
        or source.get("dirty") is not False
        or payload.get("schema") != screen.CHECKPOINT_SCHEMA
        or payload.get("arm") != "phase1"
        or payload.get("completed_updates") != screen.TRAIN_UPDATES
        or payload.get("model_config") != model_config
        or training_config.get("schema") != screen.RUN_SCHEMA
        or training_config.get("target_kind") != screen.TARGET_KIND
        or training_config.get("target_shape") != list(screen.TARGET_SHAPE)
        or training_config.get("model") != model_config
        or training_config.get("seed") != screen.FROZEN_SEED
        or training_config.get("source") != source
        or screen.sha256_json(training_config) != payload.get("config_sha256")
        or caches.get("validation") != cache_metadata
        or complete.get("schema") != screen.RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("completed_updates") != screen.TRAIN_UPDATES
        or complete.get("nonfinite_updates") != 0
        or complete.get("only_supervised_target") != "auxiliary_target"
        or complete.get("video_loss_enabled") is not False
        or complete.get("resolved_config") != vlf.file_record(config_path)
        or complete.get("checkpoint") != checkpoint_record
    ):
        raise DiagnosticError("frozen semantic checkpoint/config binding is invalid")
    ema = payload.get("ema")
    if (
        not isinstance(ema, Mapping)
        or ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or ema.get("num_updates") != screen.TRAIN_UPDATES
        or not isinstance(ema.get("shadow"), Mapping)
    ):
        raise DiagnosticError("checkpoint lacks the exact 5k EMA state")
    model.load_state_dict(ema["shadow"], strict=True)
    return payload, training_config, checkpoint_record, vlf.file_record(config_path)


def _load_manifest_action_bank(
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Tensor, dict[str, Any]]:
    """Read only cached action arrays in immutable manifest order."""
    actions: list[Tensor] = []
    for row in rows:
        relative = row.get("cache_relpath")
        if not isinstance(relative, str):
            raise DiagnosticError("D0 action permutations require frozen cached samples")
        path = (manifest_path.parent / relative).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise DiagnosticError("cached action path escapes the manifest directory")
        with np.load(path, allow_pickle=False) as payload:
            value = payload["actions"]
        if value.shape != (16, 7) or value.dtype != np.float32:
            raise DiagnosticError("cached action array is not float32 [16,7]")
        tensor = torch.from_numpy(value.copy())
        if not bool(torch.isfinite(tensor).all()):
            raise DiagnosticError("cached action array is nonfinite")
        actions.append(tensor)
    bank = torch.stack(actions).contiguous()
    if len(bank) != screen.FROZEN_VALIDATION_CLIPS:
        raise DiagnosticError("action bank does not cover the frozen validation split")
    mappings: dict[str, Any] = {}
    for offset in ACTION_PERMUTATION_OFFSETS:
        donors = tuple((index + offset) % len(rows) for index in range(len(rows)))
        if any(
            int(rows[index]["episode_index"])
            == int(rows[donor]["episode_index"])
            for index, donor in enumerate(donors)
        ):
            raise DiagnosticError(f"action offset {offset} is not episode-disjoint")
        mappings[f"offset_{offset:03d}"] = {
            "offset": offset,
            "ordered_donor_clip_ids_sha256": screen.sha256_json(
                [str(rows[index]["clip_id"]) for index in donors]
            ),
            "ordered_donor_episode_indices_sha256": screen.sha256_json(
                [int(rows[index]["episode_index"]) for index in donors]
            ),
        }
    return bank, {
        "schema": f"{SCHEMA}-action-bank-v1",
        "shape": list(bank.shape),
        "dtype": str(bank.dtype),
        "sha256": screen.tensor_sha256(bank),
        "permutations": mappings,
    }


def _batch_action_variants(
    *,
    action_bank: Tensor,
    global_indexes: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    batch_actions: Tensor,
) -> tuple[ActionVariant, ...]:
    own = torch.as_tensor(global_indexes, dtype=torch.long)
    expected = action_bank.index_select(0, own).to(batch_actions.device)
    if not torch.equal(expected, batch_actions):
        raise DiagnosticError("loader actions differ from the manifest action bank")
    variants = [
        ActionVariant(
            control="matched",
            manifest_offset=0,
            actions=batch_actions,
            source_clip_ids=tuple(str(rows[index]["clip_id"]) for index in global_indexes),
            source_episode_indices=tuple(
                int(rows[index]["episode_index"]) for index in global_indexes
            ),
        )
    ]
    for offset in ACTION_PERMUTATION_OFFSETS:
        donor_indexes = tuple((index + offset) % len(rows) for index in global_indexes)
        donor_tensor = torch.as_tensor(donor_indexes, dtype=torch.long)
        variants.append(
            ActionVariant(
                control=f"offset_{offset:03d}",
                manifest_offset=offset,
                actions=action_bank.index_select(0, donor_tensor).to(
                    batch_actions.device
                ),
                source_clip_ids=tuple(
                    str(rows[index]["clip_id"]) for index in donor_indexes
                ),
                source_episode_indices=tuple(
                    int(rows[index]["episode_index"]) for index in donor_indexes
                ),
            )
        )
    return tuple(variants)


def _production_command(args: argparse.Namespace) -> int:
    determinism = screen._configure_deterministic_eval()  # noqa: SLF001
    context = vlf.initialize_distributed()
    try:
        api_contract = assert_autonomous_api_has_no_clean_future()
        output_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
        manifest_path, rows, manifest_record = screen._manifest_record(  # noqa: SLF001
            args.manifest,
            split="val",
            expected_clips=screen.FROZEN_VALIDATION_CLIPS,
        )
        action_bank, action_bank_record = _load_manifest_action_bank(
            manifest_path, rows
        )
        dataset = cache_bridge.construct_producer_attested_dataset(
            manifest_path,
            args.data_root,
            args.semantic_cache_root,
        )
        cache_metadata = dataset.validated_cache_metadata()
        producer_attestation = dict(dataset.producer_attestation)
        vlf.seed_everything(args.seed, 0)
        model, model_config = screen.instantiate_model(args)
        model.to(context.device)
        payload, training_config, checkpoint_record, training_config_record = (
            _validate_checkpoint(
                args,
                model=model,
                model_config=model_config,
                cache_metadata=cache_metadata,
                producer_attestation=producer_attestation,
            )
        )
        model.eval()
        local_indexes, local_batches = vlf.paired_rank_evaluation_layout(
            len(dataset),
            args.eval_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        loader = DataLoader(
            torch.utils.data.Subset(dataset, local_indexes),
            batch_sampler=local_batches,
            num_workers=args.workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        config = {
            "schema": SCHEMA,
            "source": vlf.git_record(),
            "entrypoint": vlf.file_record(__file__),
            "frozen_training_source_commit": FROZEN_TRAINING_SOURCE_COMMIT,
            "protocol": _frozen_protocol_record(),
            "frozen_protocol_source_commit": FROZEN_PROTOCOL_SOURCE_COMMIT,
            "checkpoint": checkpoint_record,
            "training_config": training_config_record,
            "checkpoint_update": int(payload["completed_updates"]),
            "weights": "EMA",
            "model": model_config,
            "target_kind": screen.TARGET_KIND,
            "target_shape": list(screen.TARGET_SHAPE),
            "clock_convention": vlf.CLOCK_CONVENTION,
            "manifest": manifest_record,
            "semantic_cache": cache_metadata,
            "cache_bridge": vlf.file_record(cache_bridge.__file__),
            "producer_cache_attestation": producer_attestation,
            "dense_clean_times": list(DENSE_CLEAN_TIMES),
            "training_distribution_times": list(training_distribution_times()),
            "nfe_budgets": list(NFE_BUDGETS),
            "uniform_trace_nfe_budgets": list(UNIFORM_TRACE_NFE_BUDGETS),
            "midpoint_nfe_budgets": list(MIDPOINT_NFE_BUDGETS),
            "solvers": list(SOLVERS),
            "schedules": list(SCHEDULES),
            "endpoint_solver_configs": [
                {"solver": solver, "schedule": schedule, "actual_calls": calls}
                for solver, schedule, calls in endpoint_solver_configs()
            ],
            "action_controls": list(ACTION_CONTROLS),
            "action_permutation_offsets": list(ACTION_PERMUTATION_OFFSETS),
            "action_bank": action_bank_record,
            "fixed_clip_noise": True,
            "fixed_noise_key": "sha256(f'{clip_id}:{eval_seed}:video|aux')",
            "training_distribution_policy": (
                "target-derived forward-corruption state; nondeployable diagnostic"
            ),
            "autonomous_policy": "history+actions+fixed noise only",
            "primary_diagnosis": (
                "uniform Euler C=4 pre-call index 1 at clean_time=0.25"
            ),
            "t0_identity_requirement": "state and prediction bit-identical",
            "metric_serialization": "applicable finite; inapplicable JSON null",
            "autonomous_api_contract": api_contract,
            "teacher_model_calls": 0,
            "world_size": context.world_size,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "determinism": determinism,
        }
        config_sha256 = screen.sha256_json(config)
        screen._assert_distributed_config(context, config_sha256)  # noqa: SLF001
        if context.is_primary:
            output_dir.mkdir(parents=True, exist_ok=False)
            vlf.atomic_write_json(
                output_dir / "resolved_config.json", config, exclusive=True
            )
            vlf.atomic_write_json(
                output_dir / "provenance.json",
                {
                    "schema": SCHEMA,
                    "source": config["source"],
                    "runtime": vlf.runtime_record(),
                    "command": [sys.executable, *sys.argv],
                    "resolved_config_sha256": config_sha256,
                    "secrets_persisted": False,
                },
                exclusive=True,
            )
        context.barrier()

        local_records: list[dict[str, Any]] = []
        local_calls = 0
        local_t0_identity = True
        for batch_number, raw_batch in enumerate(loader):
            batch = screen._validate_batch(raw_batch, context.device)  # noqa: SLF001
            clip_ids = [str(value) for value in raw_batch["clip_id"]]
            episodes = [int(value) for value in raw_batch["episode_index"]]
            screen._validate_pair_batch(clip_ids, episodes)  # noqa: SLF001
            subset_indexes = local_batches[batch_number]
            global_indexes = tuple(local_indexes[index] for index in subset_indexes)
            if tuple(str(rows[index]["clip_id"]) for index in global_indexes) != tuple(
                clip_ids
            ):
                raise DiagnosticError("loader order differs from the frozen manifest")
            action_variants = _batch_action_variants(
                action_bank=action_bank,
                global_indexes=global_indexes,
                rows=rows,
                batch_actions=batch["actions"],
            )
            video_noise = vlf.stable_noise_like(
                batch["future"], clip_ids, args.seed, "video"
            )
            auxiliary_noise = vlf.stable_noise_like(
                batch["auxiliary_target"], clip_ids, args.seed, "aux"
            )
            result = evaluate_batch(
                model,
                history=batch["history"],
                actions=batch["actions"],
                video_noise=video_noise,
                auxiliary_noise=auxiliary_noise,
                target=batch["auxiliary_target"],
                clip_ids=clip_ids,
                episode_indices=episodes,
                action_variants=action_variants,
            )
            if result.actual_batched_student_calls != expected_batched_calls():
                raise DiagnosticError("batch diagnostic transformer-call count changed")
            local_calls += result.actual_batched_student_calls
            local_t0_identity = (
                local_t0_identity
                and result.t0_training_autonomous_bit_identical
            )
            local_records.extend(result.records)

        validate_record_contract(local_records)

        rank_path = output_dir / "rank_metrics" / f"rank_{context.rank:04d}.jsonl"
        screen._atomic_jsonl(rank_path, local_records)  # noqa: SLF001
        shard = {
            "rank": context.rank,
            "records": len(local_records),
            "actual_batched_student_calls": local_calls,
            "t0_training_autonomous_bit_identical": local_t0_identity,
            "file": vlf.file_record(rank_path),
        }
        shards = context.gather_objects(shard)
        context.barrier()
        if context.is_primary:
            all_records: list[dict[str, Any]] = []
            for item in sorted(shards, key=lambda value: int(value["rank"])):
                path = Path(item["file"]["path"])
                if vlf.file_record(path) != item["file"]:
                    raise DiagnosticError("rank shard changed before aggregation")
                with path.open("r", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                if len(records) != int(item["records"]):
                    raise DiagnosticError("rank shard record count changed")
                all_records.extend(records)
            expected = len(dataset) * expected_rows_per_clip()
            if len(all_records) != expected:
                raise DiagnosticError(
                    f"expected {expected} aggregated rows, found {len(all_records)}"
                )
            if not all(
                item.get("t0_training_autonomous_bit_identical") is True
                for item in shards
            ):
                raise DiagnosticError("a rank failed the exact t=0 identity assertion")
            validate_record_contract(all_records)
            all_records.sort(key=lambda row: (_summary_key(row), str(row["clip_id"])))
            merged_path = output_dir / "per_clip_metrics.jsonl"
            screen._atomic_jsonl(merged_path, all_records)  # noqa: SLF001
            summaries = summarize_records(all_records)
            diagnosis = primary_diagnosis(all_records)
            vlf.atomic_write_json(
                output_dir / "summary.json",
                {
                    "schema": SCHEMA,
                    "records": len(all_records),
                    "clips": len(dataset),
                    "cells": len(summaries),
                    "expected_rows_per_clip": expected_rows_per_clip(),
                    "expected_batched_calls_per_batch": expected_batched_calls(),
                    "rank_shards": shards,
                    "per_clip_metrics": vlf.file_record(merged_path),
                    "t0_training_autonomous_bit_identical": True,
                    "primary_diagnosis": diagnosis,
                    "cells_summary": summaries,
                },
                exclusive=True,
            )
            vlf.atomic_write_json(
                output_dir / "complete.json",
                {
                    "schema": SCHEMA,
                    "status": "complete",
                    "records": len(all_records),
                    "clips": len(dataset),
                    "teacher_model_calls": 0,
                    "clean_future_target_entered_autonomous_sampler": False,
                    "t0_training_autonomous_bit_identical": True,
                    "primary_diagnosis_classification": diagnosis["classification"],
                    "resolved_config": vlf.file_record(
                        output_dir / "resolved_config.json"
                    ),
                    "provenance": vlf.file_record(output_dir / "provenance.json"),
                    "summary": vlf.file_record(output_dir / "summary.json"),
                },
                exclusive=True,
            )
        context.barrier()
        return 0
    finally:
        vlf.close_distributed(context)


class _SyntheticPerfectModel(nn.Module):
    """Small deterministic conditional denoiser for CPU contract tests."""

    def __init__(self) -> None:
        super().__init__()
        channel = torch.linspace(-0.75, 0.75, screen.TARGET_SHAPE[0])
        temporal = torch.linspace(-1.0, 1.0, screen.TARGET_SHAPE[1])
        height = torch.linspace(-0.2, 0.2, screen.TARGET_SHAPE[2])
        width = torch.linspace(-0.1, 0.1, screen.TARGET_SHAPE[3])
        pattern = (
            channel[:, None, None, None]
            + temporal[None, :, None, None]
            + height[None, None, :, None]
            + width[None, None, None, :]
        )
        self.register_buffer("pattern", pattern)

    def clean_from_conditions(self, history: Tensor, actions: Tensor) -> Tensor:
        scale = 1.0 + history.float().flatten(1).mean(1) + 0.25 * actions.float().flatten(1).mean(1)
        return scale[:, None, None, None, None] * self.pattern[None]

    def forward(
        self,
        noisy_video: Tensor,
        noisy_auxiliary: Tensor,
        t_video: Tensor,
        t_auxiliary: Tensor,
        history: Tensor,
        actions: Tensor,
        **_: Any,
    ) -> SimpleNamespace:
        del t_video, t_auxiliary
        clean = self.clean_from_conditions(history, actions).to(
            device=noisy_auxiliary.device,
            dtype=noisy_auxiliary.dtype,
        )
        return SimpleNamespace(video_x=torch.zeros_like(noisy_video), auxiliary_x=clean)


def run_synthetic_smoke() -> dict[str, Any]:
    """Exercise every diagnostic path on CPU without data or checkpoint files."""
    screen._configure_deterministic_eval()  # noqa: SLF001
    assert_autonomous_api_has_no_clean_future()
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    batch = 2
    history = torch.stack(
        (
            torch.full((3, 5, 64, 112), -0.25),
            torch.full((3, 5, 64, 112), 0.50),
        )
    )
    actions = torch.stack(
        (
            torch.full((16, 7), -0.5),
            torch.full((16, 7), 0.75),
        )
    )
    video_noise = torch.randn(
        (batch, 3, 8, 64, 112), generator=generator, dtype=torch.float32
    )
    auxiliary_noise = torch.randn(
        (batch, *screen.TARGET_SHAPE), generator=generator, dtype=torch.float32
    )
    model = _SyntheticPerfectModel().eval()
    target = model.clean_from_conditions(history, actions)
    donor = torch.tensor((1, 0), dtype=torch.long)
    action_variants = (
        ActionVariant(
            "matched",
            0,
            actions,
            ("synthetic-0", "synthetic-1"),
            (0, 1),
        ),
        *(
            ActionVariant(
                f"offset_{offset:03d}",
                offset,
                actions.index_select(0, donor),
                ("synthetic-1", "synthetic-0"),
                (1, 0),
            )
            for offset in ACTION_PERMUTATION_OFFSETS
        ),
    )
    dense = DENSE_CLEAN_TIMES[:4]
    result = evaluate_batch(
        model,
        history=history,
        actions=actions,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        target=target,
        clip_ids=("synthetic-0", "synthetic-1"),
        episode_indices=(0, 1),
        action_variants=action_variants,
        dense_clean_times=dense,
    )
    expected_rows = batch * expected_rows_per_clip(dense_clean_times=dense)
    expected_calls = expected_batched_calls(dense_clean_times=dense)
    if (
        len(result.records) != expected_rows
        or result.actual_batched_student_calls != expected_calls
    ):
        raise DiagnosticError("synthetic row/call inventory changed")
    validate_record_contract(result.records)
    deployable_matched_primary_solvers = [
        row
        for row in result.records
        if row["deployable"] and row["action_control"] == "matched"
        and not (
            row["family"] == "endpoint" and row["schedule"] == "clean_dense"
        )
    ]
    max_nmse = max(
        float(row["semantic_nmse"])
        for row in deployable_matched_primary_solvers
    )
    if max_nmse > 1e-10:
        raise DiagnosticError("perfect synthetic denoiser did not recover its clean target")
    return {
        "schema": f"{SCHEMA}-synthetic-smoke-v1",
        "status": "pass",
        "records": len(result.records),
        "actual_batched_student_calls": result.actual_batched_student_calls,
        "expected_rows_per_clip": expected_rows_per_clip(
            dense_clean_times=dense
        ),
        "max_matched_deployable_semantic_nmse": max_nmse,
        "teacher_model_calls": 0,
        "clean_future_target_entered_autonomous_sampler": False,
        "t0_training_autonomous_bit_identical": (
            result.t0_training_autonomous_bit_identical
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("synthetic-smoke")
    run = subparsers.add_parser("run")
    screen._add_model_arguments(run)  # noqa: SLF001
    run.add_argument("--artifact-root", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--semantic-cache-root", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--seed", type=int, default=screen.FROZEN_EVALUATION_SEED)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--eval-batch-size", type=int, default=8)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    if (
        args.width != vlf.FROZEN_MODEL_WIDTH
        or args.depth != vlf.FROZEN_MODEL_DEPTH
        or args.heads != vlf.FROZEN_MODEL_HEADS
        or args.mlp_ratio != vlf.FROZEN_MODEL_MLP_RATIO
    ):
        raise DiagnosticError("model contract is frozen at width512/depth12/heads8/MLP4")
    if args.seed != screen.FROZEN_EVALUATION_SEED:
        raise DiagnosticError(f"evaluation seed is frozen at {screen.FROZEN_EVALUATION_SEED}")
    if args.workers < 0:
        raise DiagnosticError("worker count must be nonnegative")
    if args.eval_batch_size < 2 or args.eval_batch_size % 2:
        raise DiagnosticError("evaluation batch size must be even and at least two")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.command == "synthetic-smoke":
        print(json.dumps(run_synthetic_smoke(), sort_keys=True))
        return 0
    return _production_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
