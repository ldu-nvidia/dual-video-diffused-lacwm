#!/usr/bin/env python3
"""Validate and canonicalize persistent training run roots.

The local workstation defaults remain ``/mnt/data1`` and ``/mnt/data2``.
Clusters with another persistent filesystem can explicitly add canonical roots
through the colon-separated ``LACWM_ALLOWED_RUN_ROOTS`` environment variable.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


ENVIRONMENT_VARIABLE = "LACWM_ALLOWED_RUN_ROOTS"
DEFAULT_ALLOWED_RUN_ROOTS = (Path("/mnt/data1"), Path("/mnt/data2"))


class RunRootPolicyError(ValueError):
    """Raised when the configured policy or selected run root is unsafe."""


def _canonical(path: Path) -> Path:
    return path.resolve(strict=False)


def configured_allowed_run_roots(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return default roots plus validated, explicitly configured roots.

    Configured entries must already be absolute canonical paths.  Requiring the
    canonical spelling prevents a policy that appears narrow but resolves through
    ``..`` or a symlink to a broader or different filesystem location.
    """

    environment = os.environ if environ is None else environ
    roots = [_canonical(path) for path in DEFAULT_ALLOWED_RUN_ROOTS]
    if any(root == Path("/") for root in roots):
        raise RunRootPolicyError("a default run root unexpectedly resolves to '/'")
    if ENVIRONMENT_VARIABLE not in environment:
        return tuple(dict.fromkeys(roots))

    raw = environment[ENVIRONMENT_VARIABLE]
    entries = raw.split(":")
    if not raw or any(not entry for entry in entries):
        raise RunRootPolicyError(
            f"{ENVIRONMENT_VARIABLE} must be a colon-separated list without empty entries"
        )

    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            raise RunRootPolicyError(
                f"{ENVIRONMENT_VARIABLE} entries must be absolute: {entry!r}"
            )
        resolved = _canonical(candidate)
        if resolved == Path("/"):
            raise RunRootPolicyError(
                f"{ENVIRONMENT_VARIABLE} may not allow the filesystem root '/'"
            )
        if candidate != resolved:
            raise RunRootPolicyError(
                f"{ENVIRONMENT_VARIABLE} entries must be canonical: "
                f"{entry!r} resolves to {str(resolved)!r}"
            )
        roots.append(resolved)

    return tuple(dict.fromkeys(roots))


def canonical_allowed_run_root(
    value: str | os.PathLike[str],
    allowed_roots: Sequence[Path] | None = None,
) -> Path:
    """Return the canonical run root or raise when it is outside policy."""

    raw = os.fspath(value)
    if not raw:
        raise RunRootPolicyError("run root may not be empty")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise RunRootPolicyError(f"run root must be absolute: {raw!r}")
    resolved = _canonical(candidate)
    roots = tuple(
        configured_allowed_run_roots()
        if allowed_roots is None
        else allowed_roots
    )
    if not any(resolved == root or root in resolved.parents for root in roots):
        rendered = ":".join(str(root) for root in roots)
        raise RunRootPolicyError(
            f"run root {resolved} is outside allowed roots: {rendered}"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    try:
        print(canonical_allowed_run_root(args.run_root))
    except RunRootPolicyError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
