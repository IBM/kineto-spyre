#!/usr/bin/env python3
"""Integration test for the AIU trace generator (``scripts/gen_trace.py``).

[needs AIU hardware] to RUN. This test drives the real PyTorch profiler over a
PrivateUse1 (AIU) workload via ``gen_trace.py`` and then asserts the exported
``trace.json`` parses and contains at least one AIU Trace_Event.

Requirements covered:
  - Req 7.1: ``gen_trace.py`` produces a parseable Profiler_Trace.
  - Req 7.2: the produced trace contains at least one AIU Trace_Event.

The test is SKIPPED (never failed/errored) when the environment cannot run it:
torch is not importable, or the profiler does not report PrivateUse1 in its
supported activities (i.e. no AIU hardware / no kineto-spyre enabled wheel). On
a machine without AIU hardware this is the expected outcome and the test should
report SKIPPED, not FAILED.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
GEN_TRACE = os.path.join(HERE, "gen_trace.py")


def _aiu_profiling_available():
    """True only when torch is importable AND PrivateUse1 profiling is supported.

    Any import or runtime error is treated as "not available" so the guard never
    raises during collection on a torch-less / AIU-less machine.
    """
    try:
        import torch  # noqa: F401

        supported = torch.profiler.supported_activities()
        return any("PrivateUse1" in str(activity) for activity in supported)
    except Exception:
        return False


def _is_aiu_event(event):
    """Return True if a Chrome-trace event represents an AIU (PrivateUse1) activity.

    Prefer the trace validator's canonical ``is_aiu_event`` when it is available
    in this checkout; otherwise fall back to an inline category check that
    matches the design's AIU-event definition (``cat`` identifying a
    PrivateUse1 device activity).
    """
    try:
        from tools.trace_validator import is_aiu_event  # type: ignore

        return is_aiu_event(event)
    except Exception:
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", "")).lower()
        return "privateuse1" in category or "aiu" in category or "aiu" in name


AIU_AVAILABLE = _aiu_profiling_available()


@unittest.skipUnless(
    AIU_AVAILABLE,
    "torch / PrivateUse1 (AIU) profiling support unavailable — needs AIU hardware",
)
class TestGenTraceIntegration(unittest.TestCase):
    """End-to-end: run gen_trace.py and validate the produced trace."""

    def test_gen_trace_produces_parseable_trace_with_aiu_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "trace.json")

            # Run the generator as a subprocess, exactly as CI / the smoke test
            # would invoke it. Pass the output path as a CLI argument.
            result = subprocess.run(
                [sys.executable, GEN_TRACE, out_path],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg="gen_trace.py failed:\nstdout:\n{}\nstderr:\n{}".format(
                    result.stdout, result.stderr
                ),
            )

            # Req 7.1: the produced trace must parse as JSON.
            self.assertTrue(
                os.path.exists(out_path), "gen_trace.py did not write the trace file"
            )
            with open(out_path) as fh:
                trace = json.load(fh)

            events = trace.get("traceEvents", [])
            self.assertIsInstance(events, list)

            # Req 7.2: at least one AIU Trace_Event must be present.
            aiu_events = [e for e in events if _is_aiu_event(e)]
            self.assertGreaterEqual(
                len(aiu_events),
                1,
                msg="trace contained no AIU (PrivateUse1) events",
            )


if __name__ == "__main__":
    unittest.main()
