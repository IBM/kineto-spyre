"""Pure trace validator for kineto-spyre Profiler_Traces (design Component 5).

This module implements the release validation checks of Requirement 7 as a
pure function from a parsed Chrome-trace JSON document to a list of
``Violation`` records. It performs no I/O beyond an optional file-loading
helper and requires no AIU hardware, which is exactly why it is the part of
the release pipeline amenable to property-based testing (design "Correctness
Properties", Properties 1-5).

Chrome-trace shape::

    {"traceEvents": [{"name", "ts", "dur", "pid", "tid", "cat", "ph", ...}, ...]}

Validation operates over *complete* events (``ph == "X"``) only. Each complete
event models a recorded/completed activity with a timestamp (``ts``,
microseconds) and a duration (``dur``).

Checks implemented (Req 7.3-7.9):
    - ``well_formed``        -- ``name``/``ts``/``dur`` present and non-null (7.6)
    - ``at_least_one_aiu``   -- the trace has >= 1 AIU event (7.2/7.7)
    - ``ts_positive``        -- every AIU event has ``ts > 0`` (7.4)
    - ``dur_positive``       -- every AIU event has ``dur > 0`` (7.5)
    - ``no_overlap``         -- non-concurrent same-thread events do not
                                intersect on ``[ts, ts + dur)`` (7.3)

Malformed top-level JSON is distinguished from a parseable-but-invalid trace:
``parse_trace`` raises ``TraceParseError`` on JSON that cannot be parsed (or
that lacks a ``traceEvents`` list), whereas ``validate_trace`` never raises for
a structurally valid trace -- it returns a (possibly empty) violation list
(Req 7.8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# The check names this validator can emit. Every Violation.check is one of
# these (used by Property 5 to assert each entry names a *known* check).
KNOWN_CHECKS = frozenset(
    {
        "well_formed",
        "at_least_one_aiu",
        "ts_positive",
        "dur_positive",
        "no_overlap",
    }
)

# Category strings (compared case-insensitively) that mark an event as an AIU
# (PrivateUse1) device activity. PyTorch exposes the AIU device as the
# "PrivateUse1" backend; both spellings are accepted and documented here.
_AIU_CATEGORIES = frozenset({"privateuse1", "aiu"})

# Event = a single Chrome-trace JSON object (a dict). A Violation's `event`
# field carries the offending event, a tuple of events (for `no_overlap`), or
# ``None`` (for `at_least_one_aiu`, which is about the trace as a whole).
Event = Dict[str, Any]
ViolationEvent = Union[Event, Tuple[Event, ...], None]


@dataclass(frozen=True)
class Violation:
    """A single failed validation check.

    Attributes:
        check:  the name of the violated check (one of ``KNOWN_CHECKS``).
        event:  the offending event(s) -- a single event dict, a tuple of
                events (``no_overlap`` reports the intersecting pair), or
                ``None`` when the violation is about the trace as a whole
                (``at_least_one_aiu``).
        detail: a short human-readable explanation.
    """

    check: str
    event: ViolationEvent
    detail: str


class TraceParseError(Exception):
    """Raised when the top-level trace document is malformed/unparseable.

    This is distinct from a parseable-but-invalid trace: an invalid trace is
    reported via the violation list returned by :func:`validate_trace`, never
    by raising.
    """


# ---------------------------------------------------------------------------
# Parsing / loading
# ---------------------------------------------------------------------------


def parse_trace(text: str) -> Dict[str, Any]:
    """Parse Chrome-trace JSON text into a trace dict.

    Raises:
        TraceParseError: if ``text`` is not valid JSON, is not a JSON object,
            or does not contain a ``traceEvents`` list. These are *malformed*
            documents (as opposed to parseable-but-invalid traces).
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise TraceParseError(f"malformed trace JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise TraceParseError(
            f"trace top-level must be a JSON object, got {type(data).__name__}"
        )
    if not isinstance(data.get("traceEvents"), list):
        raise TraceParseError("trace is missing a 'traceEvents' list")
    return data


def load_trace(path: str) -> Dict[str, Any]:
    """Load and parse a trace file from ``path``.

    Raises:
        TraceParseError: if the file content is malformed (see ``parse_trace``).
    """
    with open(path, "r", encoding="utf-8") as fh:
        return parse_trace(fh.read())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_aiu_event(event: Event) -> bool:
    """Return True if ``event`` is an AIU (PrivateUse1) device activity.

    An AIU event is identified by its ``cat`` (category) field. The category
    is matched case-insensitively against the known AIU category names
    (``"privateuse1"`` / ``"aiu"``), so ``"PrivateUse1"``, ``"privateuse1"``
    and ``"AIU"`` all identify an AIU event.
    """
    cat = event.get("cat")
    if not isinstance(cat, str):
        return False
    return cat.strip().lower() in _AIU_CATEGORIES


def _is_number(value: Any) -> bool:
    """True if ``value`` is a real number (and not a bool, which is an int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _complete_events(trace: Dict[str, Any]) -> List[Event]:
    """Return the complete (``ph == "X"``) event objects from a trace."""
    return [
        e
        for e in trace["traceEvents"]
        if isinstance(e, dict) and e.get("ph") == "X"
    ]


def intervals_overlap(a: Event, b: Event) -> bool:
    """True if the half-open intervals ``[ts, ts + dur)`` of ``a`` and ``b`` intersect.

    Uses half-open semantics: two events that merely *touch* (one ends exactly
    when the other begins) do not overlap. Events whose ``ts``/``dur`` are
    missing/null/non-numeric are treated as non-overlapping here; their
    structural problem is reported separately by the ``well_formed`` check.
    """
    a_ts, a_dur = a.get("ts"), a.get("dur")
    b_ts, b_dur = b.get("ts"), b.get("dur")
    if not (_is_number(a_ts) and _is_number(a_dur)):
        return False
    if not (_is_number(b_ts) and _is_number(b_dur)):
        return False
    a_start, a_end = a_ts, a_ts + a_dur
    b_start, b_end = b_ts, b_ts + b_dur
    # Half-open [start, end) intersection.
    return a_start < b_end and b_start < a_end


def _thread_key(event: Event) -> Tuple[Any, Any]:
    """The ``(pid, tid)`` thread identity of an event."""
    return (event.get("pid"), event.get("tid"))


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_trace(trace: Dict[str, Any]) -> List[Violation]:
    """Validate a parsed Chrome-trace ``trace`` dict against Requirement 7.

    Returns a list of :class:`Violation`. An empty list means no check failed
    (the trace MAY be reported valid, Req 7.9). A non-empty list both forbids
    reporting the trace as valid and identifies, per entry, the violated check
    and the offending event(s) (Req 7.8).

    This function does not raise for a parseable-but-invalid trace. It does
    raise ``TraceParseError`` only if handed a structurally malformed object
    (no ``traceEvents`` list), mirroring ``parse_trace`` for defensive use.
    """
    if not isinstance(trace, dict) or not isinstance(trace.get("traceEvents"), list):
        raise TraceParseError("validate_trace requires a dict with a 'traceEvents' list")

    events = _complete_events(trace)
    violations: List[Violation] = []

    # --- Req 7.6: each event well-formed (name, ts, dur present & non-null) ---
    for e in events:
        for attr in ("name", "ts", "dur"):
            if e.get(attr) is None:
                violations.append(
                    Violation("well_formed", e, "missing or null '%s'" % attr)
                )

    aiu_events = [e for e in events if is_aiu_event(e)]

    # --- Req 7.2 / 7.7: at least one AIU event ---
    if not aiu_events:
        violations.append(
            Violation("at_least_one_aiu", None, "trace contains no AIU events")
        )

    # --- Req 7.4 / 7.5: AIU device activities have ts > 0 and dur > 0 ---
    # Positivity is orthogonal to well-formedness: only numeric values are
    # checked here; null/missing values are reported by `well_formed` above.
    for e in aiu_events:
        ts = e.get("ts")
        if _is_number(ts) and ts <= 0:
            violations.append(Violation("ts_positive", e, "AIU event ts <= 0"))
        dur = e.get("dur")
        if _is_number(dur) and dur <= 0:
            violations.append(Violation("dur_positive", e, "AIU event dur <= 0"))

    # --- Req 7.3: no overlap on the same thread for non-concurrent activities ---
    # Complete ('X') events on a single (pid, tid) thread model serial,
    # non-concurrent activities (the generated traces carry no parallelism
    # annotation), so any intersection of their [ts, ts+dur) intervals is a
    # violation. Touching intervals (half-open) are allowed.
    by_thread: Dict[Tuple[Any, Any], List[Event]] = {}
    for e in events:
        by_thread.setdefault(_thread_key(e), []).append(e)

    for thread_events in by_thread.values():
        n = len(thread_events)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = thread_events[i], thread_events[j]
                if intervals_overlap(a, b):
                    violations.append(
                        Violation(
                            "no_overlap",
                            (a, b),
                            "same-thread intervals intersect",
                        )
                    )

    return violations


# ---------------------------------------------------------------------------
# Result indicators (Req 7.8 / 7.9)
# ---------------------------------------------------------------------------


def is_failing(violations: List[Violation]) -> bool:
    """True when at least one check failed (a non-empty violation list)."""
    return len(violations) > 0


def is_valid(violations: List[Violation]) -> bool:
    """Whether the trace may be reported valid.

    ``valid`` and ``failing`` are independent indicators derived from the
    single violation list (Req 7.9). A non-empty list forbids reporting valid;
    an empty list permits it. The mutual exclusion between "valid" and
    "failing" therefore holds exactly when one or more checks actually fail.
    """
    return len(violations) == 0
