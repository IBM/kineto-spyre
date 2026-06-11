# Feature: kineto-spyre-2-12-release, Property 3: Event well-formedness
"""Property 3: Event well-formedness.

Validates: Requirements 7.6

For any generated trace, ``validate_trace`` reports a ``well_formed`` violation
for exactly those complete events missing or having a null ``name``, ``ts``, or
``dur``, and reports none when every complete event carries all three as
present, non-null values.
"""

from __future__ import annotations

import unittest

from hypothesis import given, settings

from tools.trace_validator.tests.strategies import oracle_complete_events, traces
from tools.trace_validator.validator import validate_trace


class Property3WellFormed(unittest.TestCase):
    @settings(max_examples=200)
    @given(trace=traces())
    def test_well_formed_violation_iff_missing_or_null_required_attr(self, trace):
        violations = validate_trace(trace)
        wf = [v for v in violations if v.check == "well_formed"]

        # Oracle: one violation per (event, attr) where the attr is None/absent.
        expected_count = 0
        any_malformed = False
        for e in oracle_complete_events(trace):
            for attr in ("name", "ts", "dur"):
                if e.get(attr) is None:
                    expected_count += 1
                    any_malformed = True

        self.assertEqual(len(wf), expected_count)
        self.assertEqual(bool(wf), any_malformed)

        for v in wf:
            self.assertIsInstance(v.event, dict)


if __name__ == "__main__":
    unittest.main()
