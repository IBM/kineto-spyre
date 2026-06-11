# Feature: kineto-spyre-2-12-release, Property 5: Validator soundness and completeness
"""Property 5: Validator soundness and completeness.

Validates: Requirements 7.8, 7.9

For any generated trace, ``validate_trace`` returns an empty violation list if
and only if the trace satisfies all of Properties 1-4. Whenever the list is
non-empty, every entry identifies a known check and the offending event(s), the
trace is not reported valid, and the "valid"/"failing" indicators are mutually
exclusive exactly when at least one check fails.
"""

from __future__ import annotations

import unittest

from hypothesis import given, settings

from tools.trace_validator.tests.strategies import (
    oracle_complete_events,
    oracle_is_aiu,
    oracle_is_number,
    oracle_overlapping_pairs,
    traces,
)
from tools.trace_validator.validator import (
    KNOWN_CHECKS,
    is_failing,
    is_valid,
    validate_trace,
)


def _oracle_satisfies_all(trace) -> bool:
    """True iff the trace independently satisfies Properties 1-4."""
    events = oracle_complete_events(trace)

    # Property 3: every complete event well-formed.
    for e in events:
        for attr in ("name", "ts", "dur"):
            if e.get(attr) is None:
                return False

    aiu = [e for e in events if oracle_is_aiu(e)]

    # Property 4: at least one AIU event.
    if not aiu:
        return False

    # Property 2: AIU timing positivity.
    for e in aiu:
        if oracle_is_number(e.get("ts")) and e.get("ts") <= 0:
            return False
        if oracle_is_number(e.get("dur")) and e.get("dur") <= 0:
            return False

    # Property 1: no same-thread interval overlap.
    if oracle_overlapping_pairs(trace):
        return False

    return True


class Property5SoundnessCompleteness(unittest.TestCase):
    @settings(max_examples=200)
    @given(trace=traces())
    def test_empty_iff_all_properties_hold_and_entries_well_identified(self, trace):
        violations = validate_trace(trace)

        # Soundness + completeness: empty list iff the trace satisfies all checks.
        self.assertEqual(len(violations) == 0, _oracle_satisfies_all(trace))

        # Every entry identifies a known check and the offending event(s).
        for v in violations:
            self.assertIn(v.check, KNOWN_CHECKS)
            if v.check == "at_least_one_aiu":
                self.assertIsNone(v.event)
            elif v.check == "no_overlap":
                self.assertIsInstance(v.event, tuple)
                self.assertEqual(len(v.event), 2)
            else:
                self.assertIsInstance(v.event, dict)

        # Req 7.9: a non-empty list forbids reporting valid; the indicators are
        # mutually exclusive exactly when at least one check fails.
        if violations:
            self.assertTrue(is_failing(violations))
            self.assertFalse(is_valid(violations))
        else:
            self.assertTrue(is_valid(violations))
            self.assertFalse(is_failing(violations))


if __name__ == "__main__":
    unittest.main()
