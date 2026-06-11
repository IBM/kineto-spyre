#!/bin/bash
set -e
set -o pipefail

ARCH="$(uname -m)"

# Source the guard helpers (version/kineto/PrivateUse1/AIUPTI/subcomponent/wheel
# gates). Factored into scripts/build_lib.sh so each gate is independently
# unit-testable without cloning PyTorch or running a real build.
_BUILD_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/build_lib.sh
source "$_BUILD_LIB_DIR/build_lib.sh"

# ------------------------
# PyTorch Build Automation
# ------------------------

# 2.12 release (Req 2.3, 5.2): target PyTorch 2.12.0. Python 3.12 (cp312) is the
# release ABI — the conda env below pins python to 3.12 so the wheel is cp312
# rather than whatever the host default happens to be.
PYTORCH_VERSION="2.12.0"
PYTHON_RELEASE_VERSION="${PYTHON_RELEASE_VERSION:-3.12}"
# Kineto subcomponent version for the build suffix; sourced from
# release_record.json when present, else falls back to this default.
KINETO_VERSION="${KINETO_VERSION:-1.2.0}"
PYTORCH_BUILD_SUFFIX="+aiu.kineto."$KINETO_VERSION
CONDA_ENV_NAME="buildenv-torch"
CONDA_DIR="$HOME/miniconda"

_SRC=${PYTORCH_SRC:-/project_src/}
_KINETO_DIR=${KINETO_DIR:-$(pwd)}
# Release record providing recorded subcomponent versions (Req 9.1–9.5).
RELEASE_RECORD="${RELEASE_RECORD:-$_KINETO_DIR/release_record.json}"

# Pin the build Python to the release ABI (cp312) rather than inheriting the
# host default, so the wheel is the cp312 wheel the release ships.
PYTHON_VERSION="${PYTHON_RELEASE_VERSION}"

# Derive Python ABI tag for wheel naming
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
PYTHON_TAG="cp${PYTHON_MAJOR}${PYTHON_MINOR}-cp${PYTHON_MAJOR}${PYTHON_MINOR}"

function install_miniconda() {
  if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda..."
    if [[ "$ARCH" == "x86_64" ]]; then
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    elif [[ "$ARCH" == "ppc64le" ]]; then
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh -O miniconda.sh
    elif [[ "$ARCH" == "s390x" ]]; then
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-s390x.sh -O miniconda.sh
    else
      echo "Unknow architecture $ARCH"
      exit 1
    fi
    bash miniconda.sh -b -p "$CONDA_DIR"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
  fi
  export PATH="$CONDA_DIR/bin:$PATH"
  hash -r
}

function create_conda_env() {
  echo "Creating conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION..."
    
  CONDA_PKGS=(
    python=$PYTHON_VERSION
    # PyTorch 2.11's setup.py imports setuptools.command.bdist_wheel at module
    # scope, which requires setuptools>=70.1. Recent conda python packages no
    # longer bundle setuptools, so pin it here. Upper bound matches PyTorch's
    # own pyproject.toml dev group (<80.0: setup.py develop was deprecated in 80).
    "setuptools>=70.1,<80"
    wheel
    pip
  )

  ARCH="$(uname -m)"
  if [[ "$ARCH" == "x86_64" ]]; then
    # pytorch third_party/protobuf requires CMake < 3.5
    CMAKE_VERSION_MINIMUM="3.5"
    echo "Detected x86_64 — enabling MKL..."
    CONDA_PKGS+=(
      mkl-static mkl-include llvm-openmp
      cmake=$CMAKE_VERSION_MINIMUM
    )
  elif [[ "$ARCH" == "ppc64le" ]]; then
    CMAKE_VERSION_MINIMUM="3.26"
    echo "Detected $ARCH — enabling OpenBLAS instead of MKL..."
    CONDA_PKGS+=(
       cmake=$CMAKE_VERSION_MINIMUM
    )
  elif [[ "$ARCH" == "s390x" ]]; then
    CMAKE_VERSION_MINIMUM="3.18"
    echo "Detected $ARCH — enabling OpenBLAS instead of MKL..."
    CONDA_PKGS+=(
      cmake=$CMAKE_VERSION_MINIMUM
    )
  else
    echo "Unknow architecture $ARCH"
    exit 1
  fi

  conda create -y -n "$CONDA_ENV_NAME" -c conda-forge "${CONDA_PKGS[@]}"

  echo "Conda environment '$CONDA_ENV_NAME' created for architecture: $ARCH"
}

function clone_pytorch() {
  mkdir -p $_SRC
  cd $_SRC

  # 4a. Fetch the pinned PyTorch 2.12 source when absent (Req 5.1).
  if [ ! -d "pytorch" ]; then
    echo "Cloning PyTorch $PYTORCH_VERSION..."
    git clone --recursive -b "v$PYTORCH_VERSION" https://github.com/pytorch/pytorch.git
    cd pytorch
    git submodule sync
    git submodule update --init --recursive --jobs 1
  else
    echo "PyTorch repo already exists"
    cd pytorch
  fi

  # 4a. Verify the obtained source really is 2.12.x; report a mismatch and stop
  # before building otherwise (Req 2.7, 5.7).
  verify_pytorch_version "$PWD/version.txt"

  # 4b. Replace the entire third_party/kineto with Kineto_Spyre so no upstream
  # files remain (Req 5.3), then verify the replacement (AIU plugin present) and
  # stop on a replacement failure (Req 5.8).
  echo "Replacing Kineto with the aiu-kineto"
  rm -rf third_party/kineto
  cp -r ${_KINETO_DIR} third_party/kineto
  verify_kineto_replacement "$PWD"

  # 4c. PrivateUse1 registration must be present in the obtained source; halt
  # before the wheel build and emit a build.log entry naming the missing
  # dependency otherwise (Req 3.1, 3.2).
  check_privateuse1 "$PWD" "$PWD/build.log"
}

function build_pytorch() {
  echo "Building PyTorch $PYTORCH_VERSION"
  cd $_SRC/pytorch

  # Clear dist/ so no stale/partial wheel survives (Req 5.6). build/ is also
  # cleared, preserving the original behaviour.
  rm -rf build
  clean_dist dist

  # 4e. Verify each recorded subcomponent version before building (Req 9.2–9.5).
  # When a release record is present, build only against the recorded versions;
  # a missing, unobtainable, or mismatched version stops before any wheel.
  if [ -f "$RELEASE_RECORD" ]; then
    PYTORCH_ROOT="$PWD" verify_subcomponents "$RELEASE_RECORD"
  else
    echo "WARNING: no release record at $RELEASE_RECORD — skipping subcomponent version verification" >&2
  fi

  # 4d. libaiupti must be discoverable. CMake (FindAIUToolkit.cmake) searches
  # LIBAIUPTI_INSTALL_DIR first, then falls back to LD_LIBRARY_PATH (Req 4.1).
  # Require LIBAIUPTI_INSTALL_DIR for the release build so detection is exercised.
  export LIBAIUPTI_INSTALL_DIR="${LIBAIUPTI_INSTALL_DIR:?LIBAIUPTI_INSTALL_DIR must be set for the release build (Req 4.1)}"

  # Disable CUDA
  export USE_CUDA=0
  export USE_XPU=0

  # Disable Mobile support
  export USE_NNPACK=0
  export USE_QNNPACK=0
  export USE_XNNPACK=0
  export BUILD_JNI=0
  export BUILD_BINARY=0

  # Speedup the build and make the binary smaller
  export BUILD_TEST=0
  export BUILD_CAFFE2_OPS=0
  export USE_FBGEMM=0
  
  # Enable performant multi-thread support
  export CXXFLAGS="-w"
  export USE_MKLDNN=1
  export USE_OPENMP=1
  export NO_SHARED=1
  export ATEN_THREADING=OMP
  export USE_DISTRIBUTED=1
  export GLIBCXX_USE_CXX11_ABI=1
  export USE_STATIC_DISPATCH=1

  # For GCC12 https://github.com/pytorch/pytorch/issues/77939
  export CFLAGS="-Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull"
  export CXXFLAGS="-Wno-maybe-uninitialized -Wno-uninitialized -Wno-free-nonheap-object -Wno-nonnull"

  if [[ "$ARCH" == "x86_64" ]]; then
    echo "Detected x86_64 — building with MKL..."
    export USE_MKL=1
    export MKL_STATIC=1
    export BLAS=MKL
    export MKL_THREADING=OMP
    # For AVX512 support, as the original PyTorch 2.5.1 has
    export CXXFLAGS="$CXXFLAGS -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl"
  else
    echo "Detected $ARCH — building with OpenBLAS..."
    export USE_MKL=0
    export BLAS=OpenBLAS
  fi

  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}:$CMAKE_PREFIX_PATH

  # 4f. Set the build version to the 2.12 release version (Req 2.3, 5.2) and
  # assert it targets 2.12 before invoking the build.
  export PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}${PYTORCH_BUILD_SUFFIX}"
  export PYTORCH_BUILD_NUMBER=0
  assert_build_version "$PYTORCH_BUILD_VERSION"

  pip3 --no-cache-dir install -r requirements.txt

  # Re-assert setuptools bounds after requirements.txt install: requirements.txt
  # pulls requirements-build.txt which only sets setuptools>=70.1.0 with no upper
  # bound, and will happily install 81.x — but PyTorch's direct `python setup.py`
  # invocations need setuptools<80 (80.0 removed setup.py develop; some paths
  # misbehave at 81+).
  pip3 --no-cache-dir install --force-reinstall "setuptools>=70.1,<80" wheel

  python3 setup.py clean

  # Build PyTorch first to embed the OpenMP libraries into the wheel package.
  # Invoke PyTorch's own setup.py interface UNMODIFIED (Req 5.5); tee to build.log.
  if ! python3 setup.py build --verbose 2>&1 | tee build.log; then
    report_build_failure "setup.py build" "dist"
    exit 1
  fi

  # 4d. AIUPTI detection hard gate (Req 4.3, 4.4, 4.6): scan build.log for the
  # CMake "AIU library found:" line. Emits the detected/not-detected build.log
  # entry and aborts (non-zero, no wheel) when libaiupti was not detected. This
  # gate is unconditional — no override flag, no warning-and-continue path.
  assert_aiupti_detected "$PWD/build.log"

  # Copy the OpenMP libraries into the appropriate PyTorch lib directory within the build
  if [[ "$ARCH" == "x86_64" ]]; then
    cp $CONDA_PREFIX/lib/libgomp* build/lib.linux-$ARCH-cpython-${PYTHON_MAJOR}${PYTHON_MINOR}/torch/lib/
    cp $CONDA_PREFIX/lib/libomp* build/lib.linux-$ARCH-cpython-${PYTHON_MAJOR}${PYTHON_MINOR}/torch/lib/
  fi

  if ! python3 setup.py bdist_wheel --python-tag "$PYTHON_TAG" --verbose 2>&1 | tee -a build.log; then
    report_build_failure "setup.py bdist_wheel" "dist"
    exit 1
  fi

  # 4f. Postcondition: exactly one wheel in dist/ (Req 5.4); 0 or 2+ fails and
  # removes any partial wheel (Req 5.6).
  assert_single_wheel "dist"

  # 4e/4.5. Confirm the wheel was built with HAS_AIUPTI.
  confirm_wheel_has_aiupti "$PWD/build.log"

  echo "Build complete. Wheel is in: pytorch/dist/"
  cd ..
}

# ------------------------
# Main Flow
# ----------------------

install_miniconda
source "$CONDA_DIR/etc/profile.d/conda.sh"

create_conda_env
conda activate "$CONDA_ENV_NAME"

clone_pytorch
build_pytorch
