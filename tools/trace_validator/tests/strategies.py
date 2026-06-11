"""Shared Hypothesis generators and oracle helpers for the validator property tests.

The generator emits randomized Chrome-trace structures that exercise the full
validator input space described in the design's "Correctness Properties":

    - random counts of complete (``ph == "X"``) events (plus some non-``X``
      events that the validator must ignore),
    - random ``(pid, tid)`` thread assignments drawn from a small pool so that
      same-thread collisions are common,
    - random ``ts``/``dur`` over a small integer range that includes zero,
      negative, and boundary-touching values,
    - a random subset of events marked as AIU via the ``cat`` field, and
    - random omission / null-ing of ``name``/``ts``/``dur`` to exercise
      malformedness.

The module also exposes small *independent* oracle helpers used by the tests
to compute the expected validation outcome without reusing the validator's own
implementation, so each property asserts a genuine iff.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from hypothesis import strategies as st

# Small pools so that same-(pid, tid) collisions happen frequently.
_PIDS = st.integers(min_value=0, max_value=2)
_TIDS = st.integers(min_value=0, max_value=2)

# Names, timestamps and durations. The -5..20 integer range makes touching and
# overlapping intervals (and zero / negative timing) common.
_NAMES = st.sampled_from(["k", "matmul", "add", "kernel_a", "relu"])
_TS_VALUES = st.integers(min_value=-5, max_value=20)
_DUR_VALUES = st.integers(min_value=-5, max_value=20)

_AIU_CATS = ["privateuse1", "PrivateUse1", "AIU", "aiu"]
_NON_AIU_CATS = ["cpu_op", "user_annotation", "Runtime", "python_function"]

# Mostly complete ("X") events, with some other phases mixed in so the
# validator's ph filtering is exercised.
_PH_VALUES = st.sampled_from(["X", "X", "X", "X", "i", "M", "B", "E"])

# Field presence: weighted toward "present" so valid-ish events also appear.
_FIELD_STATE = st.sampled_from(["present", "present", "present", "null", "omit"])

# Category state: a mix of AIU, non-AIU, null and omitted categories.
_CAT_STATE = st.sampled_from(
    _AIU_CATS + _NON_AIU_CATS + [None, "__omit__"]
)


@st.composite
def events(draw) -> Dict[str, Any]:
    """Draw a single randomized Chrome-trace event object."""
    event: Dict[str, Any] = {
        "ph": draw(_PH_VALUES),
        "pid": draw(_PIDS),
        "tid": draw(_TIDS),
    }

    cat = draw(_CAT_STATE)
    if cat != "__omit__":
        event["cat"] = cat

    name_state = draw(_FIELD_STATE)
    if name_state == "present":
        event["name"] = draw(_NAMES)
    elif name_state == "null":
        event["name"] = None

    ts_state = draw(_FIELD_STATE)
    if ts_state == "present":
        event["ts"] = draw(_TS_VALUES)
    elif ts_state == "null":
        event["ts"] = None

    dur_state = draw(_FIELD_STATE)
    if dur_state == "present":
        event["dur"] = draw(_DUR_VALUES)
    elif dur_state == "null":
        event["dur"] = None

    return event


@st.composite
def traces(draw, min_events: int = 0, max_events: int = 8) -> Dict[str, Any]:
    """Draw a randomized Chrome-trace document ``{"traceEvents": [...]}``."""
    n = draw(st.integers(min_value=min_events, max_value=max_events))
    return {"traceEvents": [draw(events()) for _ in range(n)]}


# ---------------------------------------------------------------------------
# Independent oracle helpers (do NOT call the validator implementation)
# ---------------------------------------------------------------------------

_AIU_CAT_SET = {"privateuse1", "aiu"}


def oracle_complete_events(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The complete (``ph == "X"``) events, mirroring the validator's filter."""
    return [
        e
        for e in trace["traceEvents"]
        if isinstance(e, dict) and e.get("ph") == "X"
    ]


def oracle_is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def oracle_is_aiu(event: Dict[str, Any]) -> bool:
    cat = event.get("cat")
    return isinstance(cat, str) and cat.strip().lower() in _AIU_CAT_SET


def oracle_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Independent half-open ``[ts, ts+dur)`` intersection test."""
    a_ts, a_dur, b_ts, b_dur = (
        a.get("ts"),
        a.get("dur"),
        b.get("ts"),
        b.get("dur"),
    )
    for v in (a_ts, a_dur, b_ts, b_dur):
        if not oracle_is_number(v):
            return False
    return a_ts < b_ts + b_dur and b_ts < a_ts + a_dur


def oracle_overlapping_pairs(
    trace: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """Indices of same-thread complete-event pairs whose intervals intersect."""
    evs = oracle_complete_events(trace)
    pairs: List[Tuple[int, int]] = []
    for i in range(len(evs)):
        for j in range(i + 1, len(evs)):
            a, b = evs[i], evs[j]
            same_thread = (a.get("pid"), a.get("tid")) == (
                b.get("pid"),
                b.get("tid"),
            )
            if same_thread and oracle_overlap(a, b):
                pairs.append((i, j))
    return pairs
