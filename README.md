# Kineto

Kineto is a library used in the PyTorch Profiler.
# kineto-spyre with libaiupti support
Kineto extension for IBM Spyre card. This is a stop-gap repo before the kineto Spyre upstream version shows up in public PyTorch (2.10.x expected)
Specifically, this repo is a modified version of libKineto that implements the support to collect events from libAIUpti.

See the [here](docs/devel/README.md) for more details how to install and use it.

The last upstream sync was with the commit `b2103f78d13fde4937af010c0ef8e24313568bc5`.
This release targets PyTorch `2.12`.

# What this is (in plain terms)

PyTorch bundles a profiling library called **kineto** (at `third_party/kineto`).
Upstream kineto only profiles NVIDIA/AMD/Intel devices. **kineto-spyre is IBM's
fork of kineto that adds an AIU profiler plugin** (`AiuptiActivityApi`, backed by
the `libaiupti` runtime) so the PyTorch profiler can also capture events from the
IBM **AIU / Spyre** accelerator. That AIU plugin is the only real difference from
upstream kineto.

The **2.12 release** is not kineto on its own — it is a full **PyTorch 2.12 wheel
(Python 3.12 / `cp312`)** in which the bundled kineto has been replaced by this
fork, so AIU profiling is built in. At a high level the release pipeline:

1. **Sync** — find the exact upstream kineto commit PyTorch 2.12 pins and
   cherry-pick every upstream change since the last sync into this fork.
   (The fork's history was squashed, so there is no shared ancestor to merge —
   hence cherry-pick. PyTorch and kineto give no compatibility guarantee, so we
   pin to the exact commit.)
2. **Build** — clone PyTorch 2.12, replace `third_party/kineto` with this fork,
   and build the `cp312` wheel. The build **fails hard if `libaiupti` is not
   detected**, so it can never silently ship a wheel that emits no AIU events
   (the historical failure mode).
3. **Validate** — run the profiler over a small AIU workload and check the trace
   has real AIU events (positive timing, no impossible overlaps, well-formed).
4. **CI** — run the build + validation automatically (a self-hosted AIU runner
   does the real end-to-end build; GitHub-hosted runners do the hardware-free
   checks).
5. **Tag & package** — tag `torch-2.12.0.aiu.kineto.<x.y.z>` and publish the
   GitHub release (wheel + provenance).

These five stages map 1:1 to the five release PRs. See `RELEASE_NOTES.md` for the
integrated upstream commits and API changes.

# Dependencies

The 2.12 release builds and runs against:

- **Python 3.12** (the wheel is built as `cp312`).
- **PyTorch 2.12** (`v2.12.0`) — the build clones this source and replaces its
  bundled `third_party/kineto` with this fork.
- **libaiupti** — the AIU profiler runtime. It must be installed and reachable
  via `LIBAIUPTI_INSTALL_DIR` (or `LD_LIBRARY_PATH`); the build fails hard if it
  is not detected, so a wheel is never produced without AIU support.
- A C/C++ toolchain and CMake, plus the build prerequisites PyTorch 2.12
  requires. `scripts/build_pytorch.sh` provisions an isolated conda environment
  (Python 3.12 + setuptools/wheel/cmake and arch-specific BLAS) for the build.
- This repo's **git submodules** (`libkineto/third_party/{fmt,googletest,dynolog}`)
  must be checked out — clone with `--recurse-submodules` (see below). `fmt` is a
  build dependency of libkineto.
- Running the trace validator's tests additionally needs **Hypothesis**
  (`pip install hypothesis`).

# Build & test from source (dev environment)

These steps build the PyTorch 2.12 wheel with the AIU-enabled kineto fork and
validate it end to end. The build and trace generation require **AIU hardware**
with `libaiupti` installed.

```bash
# 1. Clone this fork WITH submodules (libkineto/third_party/{fmt,googletest,dynolog}
#    are required to build). The integration branch carries the full release pipeline.
git clone --recurse-submodules https://github.com/IBM/kineto-spyre.git
cd kineto-spyre
# If you already cloned without --recurse-submodules:
# git submodule update --init --recursive

# 2. Point the build at its inputs.
export KINETO_DIR="$PWD"                          # this fork -> becomes third_party/kineto
export PYTORCH_SRC="$(dirname "$PWD")/pt-build"   # build dir OUTSIDE the repo
export LIBAIUPTI_INSTALL_DIR=/opt/ibm/spyre/runtime   # REQUIRED — prefix with lib/libaiupti.so + include/libaiupti
export PYTHON_RELEASE_VERSION=3.12                # build a cp312 wheel

# 3. Build the wheel. Clones PyTorch v2.12.0, swaps in this kineto fork, runs the
#    version / PrivateUse1 / AIUPTI gates, and emits dist/*.whl.
#    The subcomponent-version gate is OFF for dev builds; the official release
#    sets VERIFY_SUBCOMPONENTS=1 (and real versions in release_record.json).
./scripts/build_pytorch.sh

# 4. Install the built wheel.
pip install --no-deps --force-reinstall --user $PYTORCH_SRC/pytorch/dist/*.whl

# 5. Generate a profiler trace over an AIU (PrivateUse1) workload and validate it.
python scripts/gen_trace.py trace.json
python -m tools.trace_validator trace.json   # expect: VALID (no violations)
```

> **Common pitfalls**
> - `PYTORCH_SRC` must be **outside** `KINETO_DIR`. If it is inside (e.g.
>   `$PWD/_build`), the kineto swap fails with *"cp: cannot copy a directory
>   into itself"*. The build now stops early with a clear message if so.
> - `LIBAIUPTI_INSTALL_DIR` must be the real install prefix (the dir containing
>   `lib/libaiupti.so` and `include/libaiupti/`), not a placeholder — otherwise
>   the AIUPTI gate aborts the build (by design).
> - Dev builds **skip** subcomponent version checks; set `VERIFY_SUBCOMPONENTS=1`
>   only for an official release build.

Hardware-independent checks (no AIU required) can be run anywhere with Python
3.12:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install hypothesis
python -m unittest discover -s tools/trace_validator/tests -t .   # validator + property tests
bash -n scripts/build_pytorch.sh scripts/build_lib.sh             # build-script lint
```

# Installation

Before installing, check your system configuration:

1. Check your PyTorch version:
   ```bash
   pip list | grep torch
   ```

2. Check your Python version:
   ```bash
   python --version
   ```

4. Visit the [releases page](https://github.com/IBM/kineto-spyre/releases) to find the appropriate wheel for your configuration:
   - **PyTorch version**: e.g., `torch-2.7.1`
   - **Python version**: indicated by `cp` prefix (e.g., `cp312` requires Python 3.12.x)
   - **System architecture**: e.g., `x86_64`, `ppc64le`, or `s390x`

5. Install the matching wheel:
   ```bash
   pip3 install --no-deps --force-reinstall --user https://github.com/IBM/kineto-spyre/releases/download/torch-2.7.1.aiu.kineto.1.1.1/torch-2.7.1+aiu.kineto.1.1.1-cp312-cp312-linux_x86_64.whl
   ```
   
   **Note:** Replace the URL with the appropriate wheel from the releases page that matches your PyTorch version, Python version, and system architecture.

# Kineto

Kineto is part of the PyTorch Profiler.

The Kineto project enables:
- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

The central component of Kineto is Libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## PyTorch TensorBoard Profiler (Deprecated)

> [!WARNING]
> The TensorBoard integration with PyTorch profiler (<code>tb_plugin</code> submodule) is deprecated and scheduled for permanent removal on 03/05/2026. 
> If you rely on <code>tb_plugin</code>, please comment on the <a href="https://github.com/pytorch/kineto/issues/1248">RFC issue</a> and consider migrating your workflow. 
> The code will be deleted after the feedback period.

The goal of the PyTorch TensorBoard Profiler is to provide a seamless and intuitive end-to-end profiling experience, including straightforward collection from PyTorch and insightful visualizations and recommendations in the TensorBoard UI.
Please refer to the [README](tb_plugin/README.md) file in the `tb_plugin` folder.

## Holistic Trace Analsysis
In order to compare Kineto traces across ranks, we reccomend using the [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) tool.

## Releases and Contributing
We will follow the PyTorch release schedule which roughly happens on a 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License
Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.
