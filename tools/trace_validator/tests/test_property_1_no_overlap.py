# Feature: kineto-spyre-2-12-release, Property 1: Same-thread non-concurrent events do not overlap
"""Property 1: Same-thread non-concurrent events do not overlap.

Validates: Requirements 7.3

For any generated trace, ``validate_trace`` reports a ``no_overlap`` violation
if and only if there exist two events on the same ``(pid, tid)`` thread whose
(non-concurrent, serial) intervals ``[ts, ts+dur)`` intersect; when all
same-thread intervals are disjoint, no ``no_overlap`` violation is reported.
"""

from __future__ import annotations

import unittest

from hypothesis import given, settings

from tools.trace_validator.tests.strategies import oracle_overlapping_pairs, traces
from tools.trace_validator.validator import validate_trace


class Property1NoOverlap(unittest.TestCase):
    @settings(max_examples=200)
    @given(trace=traces())
    def test_no_overlap_reported_iff_same_thread_intervals_intersect(self, trace):
        violations = validate_trace(trace)
        no_overlap = [v for v in violations if v.check == "no_overlap"]

        expected_pairs = oracle_overlapping_pairs(trace)

        # iff: a no_overlap violation exists exactly when an overlapping
        # same-thread pair exists.
        self.assertEqual(bool(no_overlap), bool(expected_pairs))

        # Completeness: one no_overlap violation per overlapping same-thread pair.
        self.assertEqual(len(no_overlap), len(expected_pairs))

        # Soundness: every reported no_overlap names exactly the two offending
        # events as a pair.
        for v in no_overlap:
            self.assertIsInstance(v.event, tuple)
            self.assertEqual(len(v.event), 2)


if __name__ == "__main__":
    unittest.main()
