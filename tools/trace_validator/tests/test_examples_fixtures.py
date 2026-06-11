"""Example / edge-case fixtures for the trace validator (task 6.8).

Deterministic tests over known traces asserting the exact set of violated
checks (or the lack of any). Covers a known-good golden trace, an empty trace,
an all-CPU (no AIU) trace, overlapping AIU kernels on one thread, a zero-``ts``
event, a zero-``dur`` completed event, and a null ``name`` event.

Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.6, 7.8
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tools.trace_validator.validator import (
    TraceParseError,
    is_failing,
    is_valid,
    load_trace,
    parse_trace,
    validate_trace,
)


def _checks(violations):
    return sorted({v.check for v in violations})


# A known-good golden trace: one CPU op and two serial AIU kernels (disjoint
# intervals) on the same thread, all well-formed, positive timing.
GOLDEN_TRACE = {
    "traceEvents": [
        {"name": "cpu_op", "ph": "X", "ts": 10, "dur": 5, "pid": 1, "tid": 1, "cat": "cpu_op"},
        {"name": "aiu_matmul", "ph": "X", "ts": 100, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
        {"name": "aiu_add", "ph": "X", "ts": 120, "dur": 8, "pid": 1, "tid": 7, "cat": "PrivateUse1"},
        # A non-complete event that must be ignored by the validator.
        {"name": "meta", "ph": "M", "pid": 1, "tid": 7},
    ]
}


class GoldenTrace(unittest.TestCase):
    def test_golden_trace_has_no_violations_and_is_valid(self):
        violations = validate_trace(GOLDEN_TRACE)
        self.assertEqual(violations, [])
        self.assertTrue(is_valid(violations))
        self.assertFalse(is_failing(violations))

    def test_touching_intervals_do_not_overlap(self):
        # Half-open [ts, ts+dur): event ending at 110 and one starting at 110
        # on the same thread must NOT be a no_overlap violation.
        trace = {
            "traceEvents": [
                {"name": "a", "ph": "X", "ts": 100, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
                {"name": "b", "ph": "X", "ts": 110, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
            ]
        }
        self.assertEqual(validate_trace(trace), [])


class EmptyTrace(unittest.TestCase):
    def test_empty_trace_fails_at_least_one_aiu_only(self):
        violations = validate_trace({"traceEvents": []})
        self.assertEqual(_checks(violations), ["at_least_one_aiu"])
        self.assertFalse(is_valid(violations))


class AllCpuTrace(unittest.TestCase):
    def test_all_cpu_trace_fails_at_least_one_aiu(self):
        trace = {
            "traceEvents": [
                {"name": "cpu_a", "ph": "X", "ts": 1, "dur": 2, "pid": 1, "tid": 1, "cat": "cpu_op"},
                {"name": "cpu_b", "ph": "X", "ts": 5, "dur": 2, "pid": 1, "tid": 1, "cat": "user_annotation"},
            ]
        }
        violations = validate_trace(trace)
        self.assertEqual(_checks(violations), ["at_least_one_aiu"])


class OverlappingAiuKernels(unittest.TestCase):
    def test_overlapping_aiu_kernels_same_thread_report_no_overlap(self):
        trace = {
            "traceEvents": [
                {"name": "k1", "ph": "X", "ts": 100, "dur": 50, "pid": 1, "tid": 7, "cat": "privateuse1"},
                {"name": "k2", "ph": "X", "ts": 120, "dur": 50, "pid": 1, "tid": 7, "cat": "privateuse1"},
            ]
        }
        violations = validate_trace(trace)
        self.assertIn("no_overlap", _checks(violations))
        no_overlap = [v for v in violations if v.check == "no_overlap"]
        self.assertEqual(len(no_overlap), 1)
        self.assertIsInstance(no_overlap[0].event, tuple)

    def test_overlap_on_different_threads_is_allowed(self):
        trace = {
            "traceEvents": [
                {"name": "k1", "ph": "X", "ts": 100, "dur": 50, "pid": 1, "tid": 7, "cat": "privateuse1"},
                {"name": "k2", "ph": "X", "ts": 120, "dur": 50, "pid": 1, "tid": 8, "cat": "privateuse1"},
            ]
        }
        self.assertEqual(validate_trace(trace), [])


class ZeroTimestamp(unittest.TestCase):
    def test_zero_ts_aiu_event_fails_ts_positive(self):
        trace = {
            "traceEvents": [
                {"name": "k", "ph": "X", "ts": 0, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
            ]
        }
        violations = validate_trace(trace)
        self.assertEqual(_checks(violations), ["ts_positive"])


class ZeroDuration(unittest.TestCase):
    def test_zero_dur_completed_aiu_event_fails_dur_positive(self):
        trace = {
            "traceEvents": [
                {"name": "k", "ph": "X", "ts": 100, "dur": 0, "pid": 1, "tid": 7, "cat": "privateuse1"},
            ]
        }
        violations = validate_trace(trace)
        self.assertEqual(_checks(violations), ["dur_positive"])


class NullName(unittest.TestCase):
    def test_null_name_fails_well_formed(self):
        trace = {
            "traceEvents": [
                {"name": None, "ph": "X", "ts": 100, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
            ]
        }
        violations = validate_trace(trace)
        self.assertIn("well_formed", _checks(violations))
        wf = [v for v in violations if v.check == "well_formed"]
        self.assertEqual(len(wf), 1)

    def test_missing_ts_and_dur_report_two_well_formed_violations(self):
        trace = {
            "traceEvents": [
                {"name": "k", "ph": "X", "pid": 1, "tid": 7, "cat": "privateuse1"},
            ]
        }
        violations = validate_trace(trace)
        wf = [v for v in violations if v.check == "well_formed"]
        self.assertEqual(len(wf), 2)  # missing ts and missing dur


class ParseErrors(unittest.TestCase):
    def test_malformed_json_raises_parse_error(self):
        with self.assertRaises(TraceParseError):
            parse_trace("{ this is not json ")

    def test_non_object_top_level_raises_parse_error(self):
        with self.assertRaises(TraceParseError):
            parse_trace("[1, 2, 3]")

    def test_missing_trace_events_raises_parse_error(self):
        with self.assertRaises(TraceParseError):
            parse_trace('{"foo": "bar"}')

    def test_parseable_but_invalid_trace_does_not_raise(self):
        # An empty trace is parseable; it must return violations, not raise.
        trace = parse_trace('{"traceEvents": []}')
        violations = validate_trace(trace)
        self.assertTrue(is_failing(violations))

    def test_load_trace_roundtrip(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(GOLDEN_TRACE, fh)
            loaded = load_trace(path)
            self.assertEqual(validate_trace(loaded), [])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
