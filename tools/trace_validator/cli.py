"""Command-line wrapper for the trace validator (design Component 5, Req 7.8).

Usage::

    python -m tools.trace_validator <trace.json> [<trace2.json> ...]

Loads each trace file, runs :func:`validate_trace`, prints every violated
check together with the offending event(s), and exits non-zero when any
violation exists. An empty violation list means the trace is valid and the
process exits 0.

Exit codes:
    0  every trace validated cleanly (no violations)
    1  at least one trace produced one or more violations
    2  a trace file was malformed / could not be parsed, or was unreadable
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .validator import (
    TraceParseError,
    Violation,
    is_valid,
    load_trace,
    validate_trace,
)

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_PARSE_ERROR = 2


def _event_summary(event: object) -> str:
    """Render the offending event(s) of a violation compactly for the CLI."""
    if event is None:
        return "<whole trace>"
    if isinstance(event, tuple):
        return " <-> ".join(_event_summary(e) for e in event)
    if isinstance(event, dict):
        name = event.get("name")
        return (
            "{{name=%r, ts=%r, dur=%r, pid=%r, tid=%r, cat=%r}}"
            % (
                name,
                event.get("ts"),
                event.get("dur"),
                event.get("pid"),
                event.get("tid"),
                event.get("cat"),
            )
        )
    return repr(event)


def _print_report(path: str, violations: List[Violation], stream=sys.stdout) -> None:
    if is_valid(violations):
        print("%s: VALID (no violations)" % path, file=stream)
        return
    print(
        "%s: INVALID (%d violation%s)"
        % (path, len(violations), "" if len(violations) == 1 else "s"),
        file=stream,
    )
    for v in violations:
        print(
            "  - check=%s detail=%s event=%s"
            % (v.check, v.detail, _event_summary(v.event)),
            file=stream,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.trace_validator",
        description="Validate a PyTorch/Kineto Chrome-trace JSON file.",
    )
    parser.add_argument(
        "trace",
        nargs="+",
        help="path(s) to Chrome-trace JSON file(s) to validate",
    )
    args = parser.parse_args(argv)

    exit_code = EXIT_OK
    for path in args.trace:
        try:
            trace = load_trace(path)
        except TraceParseError as exc:
            print("%s: PARSE ERROR: %s" % (path, exc), file=sys.stderr)
            exit_code = max(exit_code, EXIT_PARSE_ERROR)
            continue
        except OSError as exc:
            print("%s: cannot read file: %s" % (path, exc), file=sys.stderr)
            exit_code = max(exit_code, EXIT_PARSE_ERROR)
            continue

        violations = validate_trace(trace)
        _print_report(path, violations)
        if not is_valid(violations):
            exit_code = max(exit_code, EXIT_VIOLATIONS)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
