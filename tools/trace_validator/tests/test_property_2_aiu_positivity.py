# Feature: kineto-spyre-2-12-release, Property 2: AIU timing positivity
"""Property 2: AIU timing positivity.

Validates: Requirements 7.4, 7.5

For any generated trace, ``validate_trace`` reports a ``ts_positive`` violation
for exactly those AIU events whose numeric ``ts <= 0`` and a ``dur_positive``
violation for exactly those AIU events whose numeric ``dur <= 0``.
"""

from __future__ import annotations

import unittest

from hypothesis import given, settings

from tools.trace_validator.tests.strategies import (
    oracle_complete_events,
    oracle_is_aiu,
    oracle_is_number,
    traces,
)
from tools.trace_validator.validator import validate_trace


class Property2AiuPositivity(unittest.TestCase):
    @settings(max_examples=200)
    @given(trace=traces())
    def test_positivity_violation_iff_aiu_event_has_nonpositive_timing(self, trace):
        violations = validate_trace(trace)
        ts_viol = [v for v in violations if v.check == "ts_positive"]
        dur_viol = [v for v in violations if v.check == "dur_positive"]

        aiu = [e for e in oracle_complete_events(trace) if oracle_is_aiu(e)]
        expected_ts = [
            e for e in aiu if oracle_is_number(e.get("ts")) and e.get("ts") <= 0
        ]
        expected_dur = [
            e for e in aiu if oracle_is_number(e.get("dur")) and e.get("dur") <= 0
        ]

        # Completeness + soundness via exact count match.
        self.assertEqual(len(ts_viol), len(expected_ts))
        self.assertEqual(len(dur_viol), len(expected_dur))

        # iff at the trace level.
        self.assertEqual(bool(ts_viol), bool(expected_ts))
        self.assertEqual(bool(dur_viol), bool(expected_dur))

        # Each positivity violation names a single AIU event.
        for v in ts_viol + dur_viol:
            self.assertIsInstance(v.event, dict)
            self.assertTrue(oracle_is_aiu(v.event))


if __name__ == "__main__":
    unittest.main()
