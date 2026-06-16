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

# Default PyTorch device type for the PrivateUse1 backend. The AIU backend
# usually renames this (e.g. to "aiu") via rename_privateuse1_backend(); the
# actual name is resolved at runtime by _privateuse1_device_name(). This is only
# the fallback when the renamed name cannot be queried.
PRIVATEUSE1_DEVICE = "privateuseone"

# Python module that registers the AIU PrivateUse1 backend (and its profiler
# activity) as a side effect of being imported. Overridable via the
# AIU_BACKEND_MODULE environment variable for forks that ship the backend under
# a different name. Set AIU_BACKEND_MODULE="" to skip the import entirely.
DEFAULT_AIU_BACKEND_MODULE = "torch_sendnn"


def _register_aiu_backend():
    """Import the AIU backend module so it registers the PrivateUse1 device.

    The AIU backend module (default ``torch_sendnn``) registers the
    ``privateuseone`` device backend as a side effect of being imported, which
    is what lets the workload below allocate device tensors. Importing torch
    alone is not enough.

    Returns ``True`` if the backend module was imported (or no module is
    configured), ``False`` if it is configured but not installed. The caller
    uses this to decide whether an AIU workload can run at all.
    """
    module_name = os.environ.get("AIU_BACKEND_MODULE", DEFAULT_AIU_BACKEND_MODULE)
    if not module_name:
        return True
    try:
        import importlib

        importlib.import_module(module_name)
        return True
    except ImportError:
        # Backend not installed; caller produces the actionable error.
        return False


def _privateuse1_device_name(torch):
    """Return the PyTorch device string for the PrivateUse1 (AIU) backend.

    The AIU backend module typically renames PrivateUse1 to a custom name (for
    example ``aiu``) via ``torch.utils.rename_privateuse1_backend``. After a
    rename, ``device="privateuseone"`` no longer resolves, so the workload must
    use the renamed name. Resolution order:

      1. ``AIU_DEVICE_NAME`` environment variable (explicit override).
      2. ``torch._C._get_privateuse1_backend_name()`` (the name the backend
         registered).
      3. ``PRIVATEUSE1_DEVICE`` default (``privateuseone``).
    """
    override = os.environ.get("AIU_DEVICE_NAME")
    if override:
        return override
    getter = getattr(getattr(torch, "_C", None), "_get_privateuse1_backend_name", None)
    if getter is not None:
        try:
            name = getter()
            if name:
                return name
        except Exception:
            pass
    return PRIVATEUSE1_DEVICE


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

    # The AIU backend module registers the ``privateuseone`` device as a side
    # effect of being imported; importing torch alone is not enough. Do this
    # before inspecting the profiler / running the workload.
    backend_available = _register_aiu_backend()

    # Two ways AIU (PrivateUse1) events can be captured:
    #   1. Native: the wheel was built with PrivateUse1 profiler registration,
    #      so ``ProfilerActivity.PrivateUse1`` is listed in
    #      ``supported_activities()`` and can be passed to ``profile(...)``.
    #   2. Fallback: the wheel lacks native registration, but the aiupti plugin
    #      enables its AIU activity kinds whenever the ``ProfilerActivity``
    #      environment variable is set to ``PrivateUse1`` (see
    #      libkineto/src/plugin/aiupti/AiuptiActivityApi.cpp). In that mode we
    #      profile with CPU only and the plugin attaches AIU events itself.
    supported = torch.profiler.supported_activities()
    native_privateuse1 = any(
        "PrivateUse1" in str(activity) for activity in supported
    )

    activities = [ProfilerActivity.CPU]
    if native_privateuse1:
        activities.append(ProfilerActivity.PrivateUse1)
    else:
        # Engage the aiupti env-var fallback so AIU activity kinds are enabled.
        if os.environ.get("ProfilerActivity") != "PrivateUse1":
            os.environ["ProfilerActivity"] = "PrivateUse1"
        # Without the backend module there is no PrivateUse1 device to run on
        # and no aiupti plugin to honor the fallback, so nothing could emit AIU
        # events. Fail with an actionable message (Req 3.3).
        if not backend_available:
            module_name = os.environ.get(
                "AIU_BACKEND_MODULE", DEFAULT_AIU_BACKEND_MODULE
            )
            raise RuntimeError(
                "PrivateUse1 is not in torch.profiler.supported_activities() "
                "({}) and the AIU backend module {!r} is not importable. The "
                "PyTorch wheel was not built with PrivateUse1 profiler "
                "registration and no AIU backend is installed to provide the "
                "ProfilerActivity=PrivateUse1 fallback; AIU profiling is "
                "unavailable.".format([str(a) for a in supported], module_name)
            )

    # Small PrivateUse1 (AIU) workload: a few matmuls on a device tensor so the
    # profiler records at least one AIU Trace_Event (Req 7.2). Resolve the
    # actual PrivateUse1 device name (the AIU backend usually renames it).
    device = _privateuse1_device_name(torch)
    with profile(activities=activities) as prof:
        x = torch.randn(256, 256, device=device)
        for _ in range(10):
            x = x @ x
        # Ensure queued device work has completed before the trace is exported.
        backend = getattr(torch, device, None)
        if backend is not None and hasattr(backend, "synchronize"):
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
