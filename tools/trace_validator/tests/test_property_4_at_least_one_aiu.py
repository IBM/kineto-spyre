# Feature: kineto-spyre-2-12-release, Property 4: At least one AIU event required
"""Property 4: At least one AIU event required.

Validates: Requirements 7.2, 7.7

For any generated trace, the ``at_least_one_aiu`` check fails if and only if
the trace contains no AIU complete event; any trace containing one or more AIU
events passes this check.
"""

from __future__ import annotations

import unittest

from hypothesis import given, settings

from tools.trace_validator.tests.strategies import (
    oracle_complete_events,
    oracle_is_aiu,
    traces,
)
from tools.trace_validator.validator import validate_trace


class Property4AtLeastOneAiu(unittest.TestCase):
    @settings(max_examples=200)
    @given(trace=traces())
    def test_at_least_one_aiu_fails_iff_no_aiu_events(self, trace):
        violations = validate_trace(trace)
        aiu_viol = [v for v in violations if v.check == "at_least_one_aiu"]

        has_aiu = any(oracle_is_aiu(e) for e in oracle_complete_events(trace))

        # iff: the check fails exactly when there are no AIU events.
        self.assertEqual(bool(aiu_viol), not has_aiu)

        # When it fails, it fails exactly once and refers to the whole trace.
        if not has_aiu:
            self.assertEqual(len(aiu_viol), 1)
            self.assertIsNone(aiu_viol[0].event)


if __name__ == "__main__":
    unittest.main()
