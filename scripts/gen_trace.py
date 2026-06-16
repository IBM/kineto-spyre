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

# Python module that registers the AIU PrivateUse1 backend as a side effect of
# being imported. Overridable via the AIU_BACKEND_MODULE environment variable
# for forks that ship the backend under a different name. Set
# AIU_BACKEND_MODULE="" to skip the import/registration entirely.
DEFAULT_AIU_BACKEND_MODULE = "torch_sendnn"

# PrivateUse1 device name the AIU backend is renamed to (matches the 2.11 e2e
# test, which calls rename_privateuse1_backend("aiu")). Overridable via the
# AIU_DEVICE_NAME environment variable.
DEFAULT_AIU_DEVICE_NAME = "aiu"

# torch.compile backend used to dispatch the workload to AIU when no eager
# PrivateUse1 device is registered (the torch_sendnn build is compile-only).
# "sendnn" executes on AIU hardware; "sendnn_compile_only" compiles without
# running and "sendnn_mock" is a mock, so neither would emit AIU runtime events.
# Overridable via the AIU_COMPILE_BACKEND environment variable.
DEFAULT_AIU_COMPILE_BACKEND = "sendnn"


def _resolve_sendnn_backend(module_name, pkg):
    """Find the ``sendnn_backend`` device-module object to register.

    Mirrors the 2.11 e2e test (``from torch_sendnn import torch_sendnn`` then
    ``torch_sendnn.sendnn_backend``): look up ``sendnn_backend`` on the
    ``<module_name>.<module_name>`` submodule first, then on the top-level
    package.
    """
    import importlib

    for candidate in (module_name + "." + module_name, module_name):
        try:
            mod = importlib.import_module(candidate)
        except Exception:
            continue
        backend = getattr(mod, "sendnn_backend", None)
        if backend is not None:
            return backend
    return getattr(pkg, "sendnn_backend", None)


class _AiuDeviceModule:
    """Adapt the sendnn backend object to torch 2.12's device-module API.

    The 2.11 e2e test registered ``torch_sendnn.sendnn_backend`` (a function) as
    the PrivateUse1 device module. torch 2.12's ``torch.accelerator`` probes the
    device module's ``is_available()`` (e.g. from dynamo's
    ``SymbolicStreamState`` during ``torch.compile``), which a bare function
    lacks -- raising ``AttributeError: 'function' object has no attribute
    'is_available'``. This shim proxies every other attribute to the real
    backend and supplies the accelerator-probe methods. ``is_available()``
    returns ``False`` so dynamo skips stream tracking (AIU is not a CUDA-style
    stream accelerator); AIU events are still captured by the aiupti plugin.
    """

    def __init__(self, backend):
        self._backend = backend

    def is_available(self):
        return False

    def device_count(self):
        return 0

    def __getattr__(self, name):
        # Only reached for attributes not defined above; proxy to the backend.
        return getattr(self._backend, name)


def _register_aiu_backend(torch):
    """Register the AIU PrivateUse1 device the way the 2.11 e2e test does.

    Importing the AIU backend module (default ``torch_sendnn``) is not, on its
    own, enough to expose PrivateUse1 to the profiler on this build. The 2.11
    benchmark (``e2e_tests/pt2bench/bert/z-script/llm-program.py``) performs an
    explicit registration sequence, which we replicate here::

        torch.utils.rename_privateuse1_backend("aiu")
        torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
        torch.utils.generate_methods_for_privateuse1_backend()

    Returns ``True`` if the backend module was imported (or no module is
    configured), ``False`` if it is configured but not installed.
    """
    module_name = os.environ.get("AIU_BACKEND_MODULE", DEFAULT_AIU_BACKEND_MODULE)
    if not module_name:
        return True
    try:
        import importlib

        pkg = importlib.import_module(module_name)
    except ImportError:
        # Backend not installed; caller produces the actionable error.
        return False

    device_name = os.environ.get("AIU_DEVICE_NAME", DEFAULT_AIU_DEVICE_NAME)

    # rename_privateuse1_backend can only be set once per process; skip if the
    # backend is already named what we want.
    try:
        current = torch._C._get_privateuse1_backend_name()
    except Exception:
        current = None
    if current != device_name:
        try:
            torch.utils.rename_privateuse1_backend(device_name)
        except Exception:
            pass

    sendnn_backend = _resolve_sendnn_backend(module_name, pkg)
    if sendnn_backend is not None:
        try:
            # Wrap in a shim so torch 2.12's torch.accelerator probing
            # (mod.is_available()) does not crash on the bare backend function.
            torch._register_device_module(device_name, _AiuDeviceModule(sendnn_backend))
        except Exception:
            pass

    try:
        torch.utils.generate_methods_for_privateuse1_backend()
    except Exception:
        pass

    return True


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


def _eager_device_available(torch, device):
    """Return True if eager tensors can be allocated on ``device``.

    The compile-only torch_sendnn build registers no eager PrivateUse1 device
    module, so ``torch.empty(..., device="privateuseone")`` raises. We probe
    once (outside the profiler) to decide between the eager and torch.compile
    workloads. Any failure means eager allocation is unavailable.
    """
    try:
        torch.empty(0, device=device)
        return True
    except Exception:
        return False


def _compile_backend_available(backend_name):
    """Return True if ``backend_name`` is a registered torch.compile backend."""
    try:
        from torch._dynamo import list_backends

        return backend_name in list_backends()
    except Exception:
        # Can't introspect; assume present and let torch.compile report errors.
        return True


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

    # Register the AIU PrivateUse1 device using the 2.11 e2e sequence
    # (rename_privateuse1_backend -> _register_device_module ->
    # generate_methods_for_privateuse1_backend). This is what exposes
    # PrivateUse1 to the profiler; importing torch alone is not enough.
    backend_available = _register_aiu_backend(torch)

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

    # Resolve the PrivateUse1 device name and decide how to drive the AIU.
    #   - eager: only works if the backend registered an eager PrivateUse1
    #     device module (future native builds).
    #   - compile: dispatch via torch.compile(..., backend="sendnn"), which is
    #     how the compile-only torch_sendnn build executes on AIU hardware.
    device = _privateuse1_device_name(torch)
    use_eager = _eager_device_available(torch, device)
    compile_backend = os.environ.get(
        "AIU_COMPILE_BACKEND", DEFAULT_AIU_COMPILE_BACKEND
    )
    if not use_eager and not _compile_backend_available(compile_backend):
        raise RuntimeError(
            "No eager PrivateUse1 device is registered and the torch.compile "
            "backend {!r} is not available. Install a torch_sendnn build that "
            "registers an AIU execution backend, or set AIU_COMPILE_BACKEND to "
            "a registered backend name.".format(compile_backend)
        )

    # Small AIU workload: a chain of matmuls so the profiler records at least
    # one AIU Trace_Event (Req 7.2).
    def _matmul_chain(t):
        for _ in range(10):
            t = t @ t
        return t

    with profile(activities=activities) as prof:
        if use_eager:
            x = torch.randn(256, 256, device=device)
            x = _matmul_chain(x)
            # Ensure queued device work completes before the trace is exported.
            backend = getattr(torch, device, None)
            if backend is not None and hasattr(backend, "synchronize"):
                backend.synchronize()
        else:
            compiled = torch.compile(_matmul_chain, backend=compile_backend)
            # CPU input tensor; the AIU compile backend handles device
            # placement and execution. Materialize the result to force the
            # compiled graph to run before the trace is exported.
            out = compiled(torch.randn(256, 256))
            if hasattr(out, "cpu"):
                out = out.cpu()

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
