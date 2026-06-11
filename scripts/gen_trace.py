#!/usr/bin/env python3
"""Generate a PyTorch profiler trace over an AIU (PrivateUse1) workload.

This is Component 5 of the kineto-spyre 2.12 release pipeline (the *generator*
half of the Trace Generator + Validator). It drives the PyTorch profiler over a
small ``privateuseone`` workload and exports a Chrome-trace JSON that the trace
validator (``tools/trace_validator``) can then check.

Requirements covered:
  - Req 3.3: assert the profiler reports PrivateUse1 in its supported activities
    (sanity check that the wheel was built with PrivateUse1 profiler
    registration enabled).
  - Req 7.1: produce a Profiler_Trace by running the profiler.
  - Req 7.2: the workload exercises the AIU device so that the trace contains at
    least one AIU Trace_Event.

[needs AIU hardware] to RUN. This script imports ``torch`` and drives a
PrivateUse1 (AIU) workload, which requires AIU hardware and a kineto-spyre
enabled PyTorch wheel. The ``torch`` import is kept *inside* :func:`generate_trace`
so this module can still be imported / byte-compiled on a machine without torch
installed (e.g. for authoring and CI lint stages that do not have AIU hardware).

Usage:
    python3 scripts/gen_trace.py [OUTPUT_PATH]
    GEN_TRACE_OUTPUT=/tmp/trace.json python3 scripts/gen_trace.py

If no output path is given on the command line, the ``GEN_TRACE_OUTPUT``
environment variable is used, falling back to ``trace.json`` in the current
working directory.
"""

import argparse
import os
import sys

# Default output path for the exported Chrome trace (Req 7.1). Overridable via a
# CLI argument or the GEN_TRACE_OUTPUT environment variable.
DEFAULT_OUTPUT = "trace.json"

# The PrivateUse1 device backend name PyTorch exposes for AIU.
PRIVATEUSE1_DEVICE = "privateuseone"


def resolve_output_path(argv=None):
    """Resolve the trace output path from CLI args / env / default.

    Pure (no torch needed) so it can be unit-checked without AIU hardware.
    """
    parser = argparse.ArgumentParser(
        description="Generate a PyTorch profiler trace over an AIU "
        "(PrivateUse1) workload and export it as Chrome-trace JSON.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=os.environ.get("GEN_TRACE_OUTPUT", DEFAULT_OUTPUT),
        help="Path to write the exported Chrome trace JSON "
        "(default: $GEN_TRACE_OUTPUT or %(default)r).",
    )
    args = parser.parse_args(argv)
    return args.output


def generate_trace(output_path):
    """Run the profiler over a PrivateUse1 workload and export the trace.

    torch is imported here (not at module scope) so the module remains
    importable / byte-compilable without torch installed. Running this function
    requires AIU hardware and a kineto-spyre enabled PyTorch wheel.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    # Req 3.3 sanity check: PrivateUse1 must be a supported profiler activity.
    # If it is not, the wheel was not built with PrivateUse1 profiler
    # registration (PR #172154) and no AIU events could ever be emitted.
    supported = torch.profiler.supported_activities()
    if not any("PrivateUse1" in str(activity) for activity in supported):
        raise RuntimeError(
            "PrivateUse1 is not in torch.profiler.supported_activities() "
            "({}). The PyTorch wheel was not built with PrivateUse1 profiler "
            "registration; AIU (PrivateUse1) profiling is unavailable.".format(
                [str(a) for a in supported]
            )
        )

    # Small PrivateUse1 (AIU) workload: a few matmuls on a device tensor so the
    # profiler records at least one AIU Trace_Event (Req 7.2).
    device = PRIVATEUSE1_DEVICE
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    ) as prof:
        x = torch.randn(256, 256, device=device)
        for _ in range(10):
            x = x @ x
        # Ensure queued device work has completed before the trace is exported.
        if hasattr(torch, PRIVATEUSE1_DEVICE):
            backend = getattr(torch, PRIVATEUSE1_DEVICE)
            if hasattr(backend, "synchronize"):
                backend.synchronize()

    # Req 7.1: produce a Profiler_Trace as a Chrome-trace JSON file.
    prof.export_chrome_trace(output_path)
    return output_path


def main(argv=None):
    output_path = resolve_output_path(argv)
    written = generate_trace(output_path)
    print("Wrote profiler trace to {}".format(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
