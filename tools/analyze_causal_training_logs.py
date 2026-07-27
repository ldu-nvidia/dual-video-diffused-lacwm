#!/usr/bin/env python3
"""Audit causal-screen training logs without modifying the source runs.

The analyzer snapshots and hashes caller-declared log files, parses the
distributed validation summaries, verifies that repeated rank records agree,
and compares matched-vs-shuffled learning curves at the same fixed TF scale.
Its only write is an exclusively-created JSON document outside every arm root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import numbers
import os
import re
import stat
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ANALYZER_VERSION = "1.0.0"
SIGMA_CONVENTION = "sigma=1 noise, sigma=0 clean"
ARM_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
VALIDATION_PREFIX_RE = re.compile(
    r"Validation completed for iteration\s+([0-9]+),\s+losses:\s*"
)

CONDITION_MODES = frozenset({"off", "matched", "shuffled"})
CLOCK_SUFFIXES = (
    "clock/tf_sigma_mean",
    "clock/video_sigma_mean",
)
EXPOSURE_SUFFIXES = (
    "condition/raw_state_rms",
    "condition/state_residual_rms",
    "condition/clock_residual_rms",
    "condition/combined_rms",
    "condition/native_patch_embedding_rms",
    "condition/state_to_native_ratio",
    "condition/combined_to_native_ratio",
)
PRIMARY_METRIC_SUFFIXES = (
    "video_flow_loss",
    "teacher_forced/video_x0_nmse",
)
TEACHER_SIGMA_RE = re.compile(
    r"(?:^|/)(teacher_forced/video_x0_nmse/sigma_[0-9]+(?:[.][0-9]+)?)$"
)

# These inventories are serialized verbatim into every report. Keep the
# patterns narrow: warning counts are operational telemetry, not sentiment
# analysis over arbitrary log prose.
WARNING_REGEX_INVENTORY: dict[str, dict[str, str]] = {
    "abc_sample_decode_or_transform_retry": {
        "pattern": (
            r"ABCDataset sample [0-9]+ failed "
            r"[(][^\r\n]*[)]; retrying another"
        ),
        "meaning": (
            "ABCDataset caught a decode, transform, or sample-read exception "
            "and drew another episode"
        ),
    },
    "future_validity_retry": {
        "pattern": (
            r"Rejected sample with no model-supervised future pixels; "
            r"retrying global index [0-9]+[.] diagnostic="
        ),
        "meaning": (
            "MultiDataset rejected a decoded sample whose future supervision "
            "mask was empty and drew another global index"
        ),
    },
    "future_validity_exhausted": {
        "pattern": r"future-validity retries exhausted before batching",
        "meaning": "bounded future-validity retries were exhausted",
    },
}

FATAL_REGEX_INVENTORY: dict[str, dict[str, str]] = {
    "python_traceback": {
        "pattern": r"Traceback [(]most recent call last[)]:",
        "meaning": "Python traceback header",
    },
    "error_or_critical_log_level": {
        "pattern": r"(?m)^[^\r\n]*\s(?:ERROR|CRITICAL)\s[^\r\n]*$",
        "meaning": "logger record at ERROR or CRITICAL level",
    },
    "python_exception_terminator": {
        "pattern": (
            r"(?m)^\s*(?:(?:[A-Za-z_][A-Za-z0-9_]*[.])*)"
            r"(?:AssertionError|RuntimeError|ValueError|TypeError|"
            r"FloatingPointError|MemoryError|OSError|Exception):"
        ),
        "meaning": "uncaught Python exception terminator",
    },
    "slurm_error": {
        "pattern": r"(?im)^\s*(?:srun|slurmstepd):\s+error:",
        "meaning": "Slurm step error",
    },
    "out_of_memory": {
        "pattern": (
            r"(?i)\b(?:CUDA out of memory|Out Of Memory|OUT_OF_MEMORY|"
            r"oom-kill(?:er|ed)?)\b"
        ),
        "meaning": "host, scheduler, or CUDA out-of-memory marker",
    },
}

NONFINITE_REGEX_INVENTORY: dict[str, dict[str, str]] = {
    "numeric_nan_or_inf_token": {
        "pattern": (
            r"(?i)(?<![A-Za-z0-9_])"
            r"(?:nan|[+-]?inf(?:inity)?)"
            r"(?![A-Za-z0-9_])"
        ),
        "meaning": "standalone NaN, Inf, or Infinity numeric token",
    },
    "nonfinite_quantity_phrase": {
        "pattern": (
            r"(?i)\bnon[- ]?finite\s+(?:dual\s+)?"
            r"(?:loss(?:es)?|gradient(?:s)?|metric(?:s)?|tensor(?:s)?|"
            r"value(?:s)?|state(?:s)?|parameter(?:s)?|input(?:s)?|"
            r"output(?:s)?|energy)\b"
        ),
        "meaning": "explicit non-finite numerical quantity",
    },
    "quantity_nonfinite_phrase": {
        "pattern": (
            r"(?i)\b(?:loss(?:es)?|gradient(?:s)?|metric(?:s)?|tensor(?:s)?|"
            r"value(?:s)?|state(?:s)?|parameter(?:s)?|input(?:s)?|"
            r"output(?:s)?|energy)\b[^\r\n]{0,80}\bnon[- ]?finite\b"
        ),
        "meaning": "numerical quantity followed by an explicit non-finite marker",
    },
}

# These substrings describe configured protections, not observed numerical
# failures. They are removed from each line before applying the non-finite
# detectors; any remaining actual marker on the same line is still counted.
NONFINITE_IGNORE_REGEX_INVENTORY: dict[str, dict[str, str]] = {
    "guard_or_check_enabled": {
        "pattern": (
            r"(?i)\bnon[- ]?finite\s+"
            r"(?:guard|check|checking|detection|validation)\s+"
            r"(?:(?:is)\s+)?(?:enabled|active)\b"
        ),
        "meaning": "configuration prose stating that a safeguard is active",
    },
    "error_if_nonfinite_setting": {
        "pattern": r"(?i)\berror_if_nonfinite\s*[:=]\s*(?:true|false)\b",
        "meaning": "configuration value, not an observed failure",
    },
    "nan_inf_check_enabled": {
        "pattern": (
            r"(?i)\b(?:NaN/Inf|NaN or Inf)\s+"
            r"(?:guard|check|checking|detection)\s+"
            r"(?:(?:is)\s+)?(?:enabled|active)\b"
        ),
        "meaning": "configuration prose stating that a NaN/Inf check is active",
    },
}

_WARNING_PATTERNS = {
    name: re.compile(spec["pattern"]) for name, spec in WARNING_REGEX_INVENTORY.items()
}
_FATAL_PATTERNS = {
    name: re.compile(spec["pattern"]) for name, spec in FATAL_REGEX_INVENTORY.items()
}
_NONFINITE_PATTERNS = {
    name: re.compile(spec["pattern"])
    for name, spec in NONFINITE_REGEX_INVENTORY.items()
}
_NONFINITE_IGNORE_PATTERNS = {
    name: re.compile(spec["pattern"])
    for name, spec in NONFINITE_IGNORE_REGEX_INVENTORY.items()
}


class TrainingLogAnalysisError(RuntimeError):
    """Raised when provenance or comparison invariants are not satisfied."""


@dataclass(frozen=True)
class ArmInput:
    """Caller-declared causal-screen arm and its read-only training logs."""

    name: str
    condition_mode: str
    state_scale: str | float | Decimal
    root: str | Path
    logs: tuple[str | Path, ...]


@dataclass(frozen=True)
class LogSnapshot:
    path: Path
    data: bytes
    text: str
    sha256: str
    bytes: int
    mtime_ns_before: int
    size_after: int
    mtime_ns_after: int


@dataclass(frozen=True)
class ValidationOccurrence:
    iteration: int
    losses: dict[str, float]
    path: Path
    line: int


@dataclass(frozen=True)
class CanonicalArm:
    name: str
    condition_mode: str
    state_scale: str
    root: Path
    logs: tuple[Path, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise TrainingLogAnalysisError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TrainingLogAnalysisError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise TrainingLogAnalysisError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise TrainingLogAnalysisError(
            f"{label} must be a non-empty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _canonical_scale(value: str | float | Decimal) -> str:
    try:
        scale = Decimal(str(value))
    except InvalidOperation as exc:
        raise TrainingLogAnalysisError(
            f"state scale is not a decimal number: {value!r}"
        ) from exc
    if not scale.is_finite() or scale < 0:
        raise TrainingLogAnalysisError(
            f"state scale must be finite and nonnegative: {value!r}"
        )
    normalized = format(scale.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def _normalize_arms(arms: Sequence[ArmInput]) -> tuple[CanonicalArm, ...]:
    if len(arms) < 2:
        raise TrainingLogAnalysisError("at least two arms are required")
    normalized: list[CanonicalArm] = []
    names: set[str] = set()
    roots: set[Path] = set()
    all_logs: set[Path] = set()
    for arm in arms:
        if not ARM_NAME_RE.fullmatch(arm.name):
            raise TrainingLogAnalysisError(f"invalid arm name: {arm.name!r}")
        if arm.name in names:
            raise TrainingLogAnalysisError(f"duplicate arm name: {arm.name!r}")
        names.add(arm.name)
        if arm.condition_mode not in CONDITION_MODES:
            raise TrainingLogAnalysisError(
                f"arm {arm.name!r} has invalid condition mode {arm.condition_mode!r}"
            )
        if not arm.logs:
            raise TrainingLogAnalysisError(
                f"arm {arm.name!r} must declare at least one log"
            )
        root = _canonical_directory(arm.root, f"arm {arm.name!r} root")
        if root in roots:
            raise TrainingLogAnalysisError("arm roots must be distinct")
        roots.add(root)
        logs = tuple(
            _canonical_regular_file(path, f"arm {arm.name!r} log") for path in arm.logs
        )
        outside_root = [path for path in logs if not path.is_relative_to(root)]
        if outside_root:
            raise TrainingLogAnalysisError(
                f"arm {arm.name!r} logs must be inside its declared read-only "
                f"root {root}: {outside_root}"
            )
        if len(set(logs)) != len(logs):
            raise TrainingLogAnalysisError(
                f"arm {arm.name!r} declares a duplicate log path"
            )
        overlap = all_logs.intersection(logs)
        if overlap:
            raise TrainingLogAnalysisError(
                "a log file cannot belong to multiple arms: "
                + ", ".join(str(path) for path in sorted(overlap))
            )
        all_logs.update(logs)
        normalized.append(
            CanonicalArm(
                name=arm.name,
                condition_mode=arm.condition_mode,
                state_scale=_canonical_scale(arm.state_scale),
                root=root,
                logs=logs,
            )
        )
    return tuple(normalized)


def _snapshot_log(path: Path) -> LogSnapshot:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read(before.st_size)
        after = os.fstat(handle.fileno())
    if len(data) != before.st_size:
        raise TrainingLogAnalysisError(
            f"short read while snapshotting log {path}: {len(data)} != {before.st_size}"
        )
    return LogSnapshot(
        path=path,
        data=data,
        text=data.decode("utf-8", errors="replace"),
        sha256=_sha256_bytes(data),
        bytes=len(data),
        mtime_ns_before=before.st_mtime_ns,
        size_after=after.st_size,
        mtime_ns_after=after.st_mtime_ns,
    )


def _matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        raise TrainingLogAnalysisError(
            "validation losses payload does not begin with a mapping"
        )
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                break
    raise TrainingLogAnalysisError("validation losses mapping is truncated")


def _parse_loss_mapping(
    payload: str,
    *,
    path: Path,
    line: int,
) -> dict[str, float]:
    try:
        expression = ast.parse(payload, mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise TrainingLogAnalysisError(
            f"invalid validation mapping at {path}:{line}: {exc}"
        ) from exc
    if not isinstance(expression, ast.Dict):
        raise TrainingLogAnalysisError(
            f"validation losses must be a dict at {path}:{line}"
        )
    result: dict[str, float] = {}
    for key_node, value_node in zip(expression.keys, expression.values):
        if key_node is None:
            raise TrainingLogAnalysisError(
                f"dict expansion is forbidden in validation losses at {path}:{line}"
            )
        try:
            key = ast.literal_eval(key_node)
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise TrainingLogAnalysisError(
                f"non-literal validation entry at {path}:{line}: {exc}"
            ) from exc
        if not isinstance(key, str) or not key:
            raise TrainingLogAnalysisError(
                f"validation loss keys must be non-empty strings at {path}:{line}"
            )
        if key in result:
            raise TrainingLogAnalysisError(
                f"duplicate validation loss key {key!r} at {path}:{line}"
            )
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TrainingLogAnalysisError(
                f"validation loss {key!r} is not a real scalar at {path}:{line}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise TrainingLogAnalysisError(
                f"validation loss {key!r} is non-finite at {path}:{line}"
            )
        result[key] = number
    if not result:
        raise TrainingLogAnalysisError(
            f"validation losses mapping is empty at {path}:{line}"
        )
    return result


def _parse_validation_occurrences(
    snapshot: LogSnapshot,
) -> list[ValidationOccurrence]:
    occurrences: list[ValidationOccurrence] = []
    for match in VALIDATION_PREFIX_RE.finditer(snapshot.text):
        payload_start = match.end()
        while (
            payload_start < len(snapshot.text)
            and snapshot.text[payload_start].isspace()
        ):
            payload_start += 1
        payload_end = _matching_brace(snapshot.text, payload_start)
        line = snapshot.text.count("\n", 0, match.start()) + 1
        payload = snapshot.text[payload_start:payload_end]
        occurrences.append(
            ValidationOccurrence(
                iteration=int(match.group(1)),
                losses=_parse_loss_mapping(
                    payload,
                    path=snapshot.path,
                    line=line,
                ),
                path=snapshot.path,
                line=line,
            )
        )
    return occurrences


def _mapping_difference(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> list[str]:
    differences = []
    for key in sorted(set(left) | set(right)):
        if key not in left:
            differences.append(f"{key}: missing from first")
        elif key not in right:
            differences.append(f"{key}: missing from duplicate")
        elif left[key] != right[key]:
            differences.append(f"{key}: {left[key]!r} != {right[key]!r}")
    return differences


def _deduplicate_validation(
    arm_name: str,
    occurrences: Sequence[ValidationOccurrence],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not occurrences:
        raise TrainingLogAnalysisError(f"arm {arm_name!r} has no validation records")
    grouped: dict[int, list[ValidationOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.iteration, []).append(occurrence)

    records: list[dict[str, Any]] = []
    occurrence_counts: dict[str, int] = {}
    for iteration in sorted(grouped):
        duplicates = grouped[iteration]
        reference = duplicates[0]
        for duplicate in duplicates[1:]:
            differences = _mapping_difference(reference.losses, duplicate.losses)
            if differences:
                raise TrainingLogAnalysisError(
                    f"duplicate rank validation records disagree for arm "
                    f"{arm_name!r}, iteration {iteration}: "
                    f"{reference.path}:{reference.line} vs "
                    f"{duplicate.path}:{duplicate.line}; " + "; ".join(differences[:12])
                )
        occurrence_counts[str(iteration)] = len(duplicates)
        records.append(
            {
                "iteration": iteration,
                "losses": dict(sorted(reference.losses.items())),
                "occurrences": [
                    {"path": str(item.path), "line": item.line} for item in duplicates
                ],
            }
        )
    return records, {
        "raw_occurrence_count": len(occurrences),
        "deduplicated_iteration_count": len(records),
        "identical_duplicate_occurrence_count": len(occurrences) - len(records),
        "occurrences_per_iteration": occurrence_counts,
        "records_sha256": _canonical_json_sha256(
            [
                {
                    "iteration": record["iteration"],
                    "losses": record["losses"],
                }
                for record in records
            ]
        ),
    }


def _regex_marker_summary(
    snapshots: Sequence[LogSnapshot],
    patterns: Mapping[str, re.Pattern[str]],
) -> dict[str, Any]:
    per_log: dict[str, dict[str, int]] = {}
    total = {name: 0 for name in patterns}
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in patterns}
    unique_lines: set[tuple[str, int]] = set()
    for snapshot in snapshots:
        counts: dict[str, int] = {}
        for name, pattern in patterns.items():
            matches = list(pattern.finditer(snapshot.text))
            counts[name] = len(matches)
            total[name] += len(matches)
            for match in matches:
                line = snapshot.text.count("\n", 0, match.start()) + 1
                unique_lines.add((str(snapshot.path), line))
            for match in matches[: max(0, 5 - len(examples[name]))]:
                line = snapshot.text.count("\n", 0, match.start()) + 1
                text = snapshot.text.splitlines()[line - 1]
                examples[name].append(
                    {
                        "path": str(snapshot.path),
                        "line": line,
                        "text": text[:500],
                    }
                )
        per_log[str(snapshot.path)] = counts
    return {
        "total_by_pattern": total,
        "total_matches": sum(total.values()),
        "unique_matching_line_count": len(unique_lines),
        "per_log": per_log,
        "examples_first_five": examples,
    }


def _nonfinite_marker_summary(
    snapshots: Sequence[LogSnapshot],
) -> dict[str, Any]:
    total = {name: 0 for name in _NONFINITE_PATTERNS}
    ignored = {name: 0 for name in _NONFINITE_IGNORE_PATTERNS}
    unique_lines: list[dict[str, Any]] = []
    per_log: dict[str, dict[str, int]] = {}
    for snapshot in snapshots:
        log_counts = {name: 0 for name in _NONFINITE_PATTERNS}
        for line_number, original_line in enumerate(snapshot.text.splitlines(), 1):
            masked = original_line
            for name, pattern in _NONFINITE_IGNORE_PATTERNS.items():
                matches = list(pattern.finditer(masked))
                ignored[name] += len(matches)
                if matches:
                    masked = pattern.sub("", masked)
            line_matched = False
            for name, pattern in _NONFINITE_PATTERNS.items():
                count = len(list(pattern.finditer(masked)))
                if count:
                    line_matched = True
                    log_counts[name] += count
                    total[name] += count
            if line_matched and len(unique_lines) < 20:
                unique_lines.append(
                    {
                        "path": str(snapshot.path),
                        "line": line_number,
                        "text": original_line[:500],
                    }
                )
        per_log[str(snapshot.path)] = log_counts
    return {
        "total_by_pattern": total,
        "total_matches": sum(total.values()),
        "unique_matching_line_count": sum(
            1
            for snapshot in snapshots
            for original_line in snapshot.text.splitlines()
            if _line_has_nonfinite_after_mask(original_line)
        ),
        "ignored_configuration_matches": ignored,
        "per_log": per_log,
        "examples_first_twenty_lines": unique_lines,
    }


def _line_has_nonfinite_after_mask(line: str) -> bool:
    masked = line
    for pattern in _NONFINITE_IGNORE_PATTERNS.values():
        masked = pattern.sub("", masked)
    return any(pattern.search(masked) for pattern in _NONFINITE_PATTERNS.values())


def _metric_matches(losses: Mapping[str, float], suffix: str) -> list[str]:
    return sorted(key for key in losses if key == suffix or key.endswith("/" + suffix))


def _extract_curve(
    arm_name: str,
    records: Sequence[Mapping[str, Any]],
    suffix: str,
) -> dict[str, Any]:
    source_key: str | None = None
    points = []
    for record in records:
        matches = _metric_matches(record["losses"], suffix)
        if len(matches) != 1:
            raise TrainingLogAnalysisError(
                f"arm {arm_name!r}, iteration {record['iteration']} needs "
                f"exactly one metric ending in {suffix!r}; found {matches}"
            )
        if source_key is None:
            source_key = matches[0]
        elif source_key != matches[0]:
            raise TrainingLogAnalysisError(
                f"arm {arm_name!r} changed the full key for suffix "
                f"{suffix!r}: {source_key!r} -> {matches[0]!r}"
            )
        points.append(
            {
                "iteration": int(record["iteration"]),
                "value": float(record["losses"][matches[0]]),
            }
        )
    return {
        "metric_suffix": suffix,
        "source_key": source_key,
        "points": points,
        "summary": _curve_summary(points),
    }


def _curve_summary(points: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if not points:
        raise TrainingLogAnalysisError("cannot summarize an empty curve")
    iterations = [int(point["iteration"]) for point in points]
    values = [float(point["value"]) for point in points]
    if iterations != sorted(set(iterations)):
        raise TrainingLogAnalysisError(
            f"curve iterations are not strictly increasing: {iterations}"
        )
    auc = None
    normalized_auc = None
    if len(points) >= 2:
        auc = sum(
            (values[index] + values[index + 1])
            * 0.5
            * (iterations[index + 1] - iterations[index])
            for index in range(len(points) - 1)
        )
        span = iterations[-1] - iterations[0]
        if span <= 0:
            raise TrainingLogAnalysisError(
                "multi-point curve has a nonpositive iteration span"
            )
        normalized_auc = auc / span
    return {
        "count": len(points),
        "first_iteration": iterations[0],
        "last_iteration": iterations[-1],
        "first_value": values[0],
        "final_value": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "trapezoid_auc": auc,
        "trapezoid_auc_per_iteration_span": normalized_auc,
    }


def _teacher_sigma_suffixes(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    suffixes = set()
    for record in records:
        for key in record["losses"]:
            match = TEACHER_SIGMA_RE.search(key)
            if match:
                suffixes.add(match.group(1))
    return tuple(sorted(suffixes))


def _arm_curves(
    arm: CanonicalArm,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sigma_suffixes = _teacher_sigma_suffixes(records)
    if not sigma_suffixes:
        raise TrainingLogAnalysisError(
            f"arm {arm.name!r} has no teacher-forced video NMSE sigma bins"
        )
    primary = {
        suffix: _extract_curve(arm.name, records, suffix)
        for suffix in (*PRIMARY_METRIC_SUFFIXES, *sigma_suffixes)
    }
    clocks = {
        suffix: _extract_curve(arm.name, records, suffix) for suffix in CLOCK_SUFFIXES
    }
    exposure = {
        suffix: _extract_curve(arm.name, records, suffix)
        for suffix in EXPOSURE_SUFFIXES
    }
    return {
        "primary": primary,
        "clocks": clocks,
        "exposure": exposure,
        "teacher_forced_sigma_metric_suffixes": list(sigma_suffixes),
    }


def _first_iteration_at_or_below(
    points: Sequence[Mapping[str, float]],
    threshold: float,
) -> int | None:
    for point in points:
        if float(point["value"]) <= threshold:
            return int(point["iteration"])
    return None


def _relative_percent(delta: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return 100.0 * delta / reference


def _compare_metric(
    matched_curve: Mapping[str, Any],
    shuffled_curve: Mapping[str, Any],
) -> dict[str, Any]:
    matched_points = matched_curve["points"]
    shuffled_points = shuffled_curve["points"]
    matched_iterations = [int(point["iteration"]) for point in matched_points]
    shuffled_iterations = [int(point["iteration"]) for point in shuffled_points]
    if matched_iterations != shuffled_iterations:
        raise TrainingLogAnalysisError(
            "matched and shuffled metric curves have different iteration grids"
        )
    if len(matched_points) < 2:
        raise TrainingLogAnalysisError(
            "learning-curve comparisons require at least two validation points"
        )
    matched_summary = matched_curve["summary"]
    shuffled_summary = shuffled_curve["summary"]
    final_delta = matched_summary["final_value"] - shuffled_summary["final_value"]
    auc_delta = matched_summary["trapezoid_auc"] - shuffled_summary["trapezoid_auc"]
    normalized_auc_delta = (
        matched_summary["trapezoid_auc_per_iteration_span"]
        - shuffled_summary["trapezoid_auc_per_iteration_span"]
    )
    reference_final = float(shuffled_summary["final_value"])
    matched_reach = _first_iteration_at_or_below(matched_points, reference_final)
    shuffled_reach = _first_iteration_at_or_below(shuffled_points, reference_final)
    first_iteration = matched_iterations[0]
    matched_updates = None if matched_reach is None else matched_reach - first_iteration
    shuffled_updates = (
        None if shuffled_reach is None else shuffled_reach - first_iteration
    )
    lead = (
        None
        if matched_updates is None or shuffled_updates is None
        else shuffled_updates - matched_updates
    )
    deltas = [
        {
            "iteration": iteration,
            "matched_minus_shuffled": (
                float(matched["value"]) - float(shuffled["value"])
            ),
        }
        for iteration, matched, shuffled in zip(
            matched_iterations, matched_points, shuffled_points
        )
    ]
    return {
        "definition": "matched minus shuffled at the same state scale",
        "favorable_direction": (
            "negative final/AUC delta and positive matched_lead_updates"
        ),
        "matched": matched_summary,
        "shuffled": shuffled_summary,
        "final_matched_minus_shuffled": final_delta,
        "final_relative_percent_vs_shuffled": _relative_percent(
            final_delta, shuffled_summary["final_value"]
        ),
        "trapezoid_auc_matched_minus_shuffled": auc_delta,
        "normalized_auc_matched_minus_shuffled": normalized_auc_delta,
        "normalized_auc_relative_percent_vs_shuffled": _relative_percent(
            normalized_auc_delta,
            shuffled_summary["trapezoid_auc_per_iteration_span"],
        ),
        "time_to_reference_final": {
            "reference_arm": "shuffled",
            "threshold": reference_final,
            "criterion": "first observed validation value <= threshold",
            "interpolation": "none",
            "matched_first_iteration": matched_reach,
            "shuffled_first_iteration": shuffled_reach,
            "matched_updates_from_first_validation": matched_updates,
            "shuffled_updates_from_first_validation": shuffled_updates,
            "matched_lead_updates": lead,
        },
        "pointwise_delta": deltas,
    }


def _curve_values(curve: Mapping[str, Any]) -> list[float]:
    return [float(point["value"]) for point in curve["points"]]


def _same_scale_comparisons(
    arms: Sequence[CanonicalArm],
    arm_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for arm in arms:
        if arm.condition_mode not in {"matched", "shuffled"}:
            continue
        grouped.setdefault(arm.state_scale, {}).setdefault(
            arm.condition_mode, []
        ).append(arm.name)
    if not grouped:
        raise TrainingLogAnalysisError(
            "no matched/shuffled same-scale arms were supplied"
        )

    comparisons: dict[str, Any] = {}
    for scale in sorted(grouped, key=Decimal):
        modes = grouped[scale]
        if set(modes) != {"matched", "shuffled"}:
            raise TrainingLogAnalysisError(
                f"state scale {scale} must contain both matched and shuffled "
                f"arms; found {sorted(modes)}"
            )
        if len(modes["matched"]) != 1 or len(modes["shuffled"]) != 1:
            raise TrainingLogAnalysisError(
                f"state scale {scale} has ambiguous matched/shuffled arms: {modes}"
            )
        matched_name = modes["matched"][0]
        shuffled_name = modes["shuffled"][0]
        matched = arm_payloads[matched_name]
        shuffled = arm_payloads[shuffled_name]

        matched_iterations = matched["validation"]["iterations"]
        shuffled_iterations = shuffled["validation"]["iterations"]
        if matched_iterations != shuffled_iterations:
            raise TrainingLogAnalysisError(
                f"same-scale arms {matched_name!r}/{shuffled_name!r} have "
                f"different validation iteration grids: "
                f"{matched_iterations} != {shuffled_iterations}"
            )

        clock_checks = {}
        for suffix in CLOCK_SUFFIXES:
            matched_values = _curve_values(matched["curves"]["clocks"][suffix])
            shuffled_values = _curve_values(shuffled["curves"]["clocks"][suffix])
            if matched_values != shuffled_values:
                raise TrainingLogAnalysisError(
                    f"same-scale arms {matched_name!r}/{shuffled_name!r} "
                    f"have different {suffix} values"
                )
            clock_checks[suffix] = {
                "exact_match": True,
                "values": matched_values,
            }

        matched_sigma = set(matched["curves"]["teacher_forced_sigma_metric_suffixes"])
        shuffled_sigma = set(shuffled["curves"]["teacher_forced_sigma_metric_suffixes"])
        if matched_sigma != shuffled_sigma:
            raise TrainingLogAnalysisError(
                f"same-scale arms {matched_name!r}/{shuffled_name!r} "
                f"have different teacher-forced sigma bins: "
                f"{sorted(matched_sigma)} != {sorted(shuffled_sigma)}"
            )
        metric_suffixes = (
            *PRIMARY_METRIC_SUFFIXES,
            *sorted(matched_sigma),
        )
        metric_comparisons = {
            suffix: _compare_metric(
                matched["curves"]["primary"][suffix],
                shuffled["curves"]["primary"][suffix],
            )
            for suffix in metric_suffixes
        }
        comparisons[scale] = {
            "state_scale": scale,
            "matched_arm": matched_name,
            "shuffled_arm": shuffled_name,
            "validation_iteration_grid": matched_iterations,
            "validation_grid_exact_match": True,
            "clock_mean_exact_match": clock_checks,
            "metric_comparisons": metric_comparisons,
        }
    return comparisons


def _prepare_output_path(
    output: str | Path,
    roots: Sequence[Path],
) -> Path:
    raw = Path(output).expanduser()
    if raw.suffix.lower() != ".json":
        raise TrainingLogAnalysisError("output path must end in .json")
    parent = _canonical_directory(raw.parent, "output parent")
    path = parent / raw.name
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise TrainingLogAnalysisError(f"output path already exists: {path}")
    for root in roots:
        if path == root or path.is_relative_to(root):
            raise TrainingLogAnalysisError(
                f"output must be outside every read-only arm root: {path}"
            )
    return path


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def analyze(
    arms: Sequence[ArmInput],
    *,
    output: str | Path,
) -> dict[str, Any]:
    """Analyze immutable log snapshots and exclusively write one JSON report."""

    canonical_arms = _normalize_arms(arms)
    output_path = _prepare_output_path(output, [arm.root for arm in canonical_arms])

    arm_payloads: dict[str, dict[str, Any]] = {}
    for arm in canonical_arms:
        snapshots = tuple(_snapshot_log(path) for path in arm.logs)
        occurrences = [
            occurrence
            for snapshot in snapshots
            for occurrence in _parse_validation_occurrences(snapshot)
        ]
        records, deduplication = _deduplicate_validation(arm.name, occurrences)
        curves = _arm_curves(arm, records)
        warnings = _regex_marker_summary(snapshots, _WARNING_PATTERNS)
        fatal = _regex_marker_summary(snapshots, _FATAL_PATTERNS)
        nonfinite = _nonfinite_marker_summary(snapshots)
        arm_payloads[arm.name] = {
            "condition_mode": arm.condition_mode,
            "state_scale": arm.state_scale,
            "root": str(arm.root),
            "logs": [
                {
                    "path": str(snapshot.path),
                    "sha256_of_snapshotted_bytes": snapshot.sha256,
                    "snapshotted_bytes": snapshot.bytes,
                    "mtime_ns_before": snapshot.mtime_ns_before,
                    "size_after_snapshot": snapshot.size_after,
                    "mtime_ns_after": snapshot.mtime_ns_after,
                    "source_changed_during_snapshot": (
                        snapshot.size_after != snapshot.bytes
                        or snapshot.mtime_ns_after != snapshot.mtime_ns_before
                    ),
                    "utf8_replacement_character_count": snapshot.text.count("\ufffd"),
                }
                for snapshot in snapshots
            ],
            "validation": {
                **deduplication,
                "iterations": [int(record["iteration"]) for record in records],
                "records": records,
            },
            "operational_markers": {
                "warnings": warnings,
                "fatal": fatal,
                "nonfinite": nonfinite,
                "fatal_or_nonfinite_present": bool(
                    fatal["total_matches"] or nonfinite["unique_matching_line_count"]
                ),
            },
            "curves": curves,
            "available_validation_metric_keys": sorted(
                {key for record in records for key in record["losses"]}
            ),
        }

    comparisons = _same_scale_comparisons(canonical_arms, arm_payloads)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "kind": "causal_tf_training_log_telemetry_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sigma_convention": SIGMA_CONVENTION,
        "read_only_contract": {
            "source_runs_modified": False,
            "source_jobs_modified": False,
            "only_write": str(output_path),
            "output_exclusive_create": True,
            "output_outside_every_arm_root": True,
        },
        "regex_inventory": {
            "flags": (
                "flags embedded in each exact Python regular expression; "
                "warning patterns are case-sensitive"
            ),
            "warnings": WARNING_REGEX_INVENTORY,
            "fatal": FATAL_REGEX_INVENTORY,
            "nonfinite": NONFINITE_REGEX_INVENTORY,
            "nonfinite_ignored_configuration_prose": (NONFINITE_IGNORE_REGEX_INVENTORY),
        },
        "metric_definitions": {
            "comparison": (
                "matched minus shuffled at identical fixed state scale; "
                "all requested losses/NMSE are lower-is-better"
            ),
            "trapezoid_auc": ("sum((y_i+y_(i+1))/2 * (iteration_(i+1)-iteration_i))"),
            "normalized_auc": (
                "trapezoid AUC divided by last_iteration-first_iteration"
            ),
            "time_to_reference_final": (
                "first observed validation iteration whose value is <= the "
                "same-scale shuffled arm's final value; no interpolation"
            ),
            "teacher_forced_scope": (
                "one-call clean-estimate diagnostics whose TF corruption "
                "comes from the ground-truth clip; not autonomous generation"
            ),
            "exposure_keys": list(EXPOSURE_SUFFIXES),
        },
        "provenance": {
            "arm_count": len(canonical_arms),
            "input_log_count": sum(len(arm.logs) for arm in canonical_arms),
            "arms": arm_payloads,
        },
        "same_scale_matched_vs_shuffled": comparisons,
        "limitations": [
            (
                "Validation summaries are aggregate ABC loader records; "
                "identical DDP rank repeats are deduplicated, while any "
                "disagreement aborts analysis."
            ),
            (
                "Warning and fatal-marker counts are regex matches in the "
                "snapshotted log bytes, not reconstructed exception events."
            ),
            (
                "Teacher-forced video NMSE can diagnose supervised denoising "
                "but cannot by itself establish autonomous low-NFE quality."
            ),
            (
                "Time-to-threshold uses observed validation checkpoints and "
                "does not interpolate between them."
            ),
        ],
    }
    payload["payload_identity_sha256"] = _canonical_json_sha256(payload)
    _exclusive_json(output_path, payload)
    return payload


def _parse_name_assignment(value: str, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{label} must use NAME=VALUE")
    name, raw_value = value.split("=", 1)
    if not ARM_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"{label} name must use letters, digits, dot, underscore, or dash"
        )
    if not raw_value:
        raise argparse.ArgumentTypeError(f"{label} value must not be empty")
    return name, raw_value


def _parse_arm_cli(value: str) -> tuple[str, str, str, Path]:
    name, raw = _parse_name_assignment(value, "arm")
    parts = raw.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("arm must use NAME=MODE,SCALE,ROOT")
    mode, scale, root = parts
    if mode not in CONDITION_MODES:
        raise argparse.ArgumentTypeError(
            f"condition mode must be one of {sorted(CONDITION_MODES)}"
        )
    if not scale or not root:
        raise argparse.ArgumentTypeError("arm scale/root must not be empty")
    return name, mode, scale, Path(root)


def _parse_log_cli(value: str) -> tuple[str, Path]:
    name, raw = _parse_name_assignment(value, "log")
    return name, Path(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm_cli,
        required=True,
        metavar="NAME=MODE,SCALE,ROOT",
        help=(
            "declare an arm and its read-only root; MODE is off, matched, "
            "or shuffled (repeat)"
        ),
    )
    parser.add_argument(
        "--log",
        action="append",
        type=_parse_log_cli,
        required=True,
        metavar="NAME=PATH",
        help="attach one read-only training log to a declared arm (repeat)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=("fresh .json path outside every arm root; parent must already exist"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arm_metadata: dict[str, tuple[str, str, Path]] = {}
    for name, mode, scale, root in args.arm:
        if name in arm_metadata:
            raise TrainingLogAnalysisError(f"duplicate arm: {name!r}")
        arm_metadata[name] = (mode, scale, root)
    logs: dict[str, list[Path]] = {}
    for name, path in args.log:
        if name not in arm_metadata:
            raise TrainingLogAnalysisError(f"log references undeclared arm: {name!r}")
        logs.setdefault(name, []).append(path)
    missing_logs = sorted(set(arm_metadata) - set(logs))
    if missing_logs:
        raise TrainingLogAnalysisError(f"declared arms lack logs: {missing_logs}")
    arms = [
        ArmInput(
            name=name,
            condition_mode=mode,
            state_scale=scale,
            root=root,
            logs=tuple(logs[name]),
        )
        for name, (mode, scale, root) in arm_metadata.items()
    ]
    analyze(arms, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
