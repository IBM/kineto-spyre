"""Trace validator package for the kineto-spyre 2.12 release pipeline.

Provides a pure, hardware-independent validator for Chrome-trace JSON
produced by the PyTorch profiler (Component 5 of the release design).

Public API:
    - validate_trace(trace) -> list[Violation]
    - parse_trace(text) -> dict
    - load_trace(path) -> dict
    - is_aiu_event(event) -> bool
    - is_valid(violations) -> bool
    - is_failing(violations) -> bool
    - Violation
    - TraceParseError
"""

from .validator import (
    TraceParseError,
    Violation,
    is_aiu_event,
    is_failing,
    is_valid,
    load_trace,
    parse_trace,
    validate_trace,
)

__all__ = [
    "TraceParseError",
    "Violation",
    "is_aiu_event",
    "is_failing",
    "is_valid",
    "load_trace",
    "parse_trace",
    "validate_trace",
]
