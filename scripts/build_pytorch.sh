#!/bin/bash
set -e

ARCH="$(uname -m)"

# ------------------------
# PyTorch Build Automation
# ------------------------

PYTORCH_VERSION="2.7.1"
KINETO_VERSION="1.0"
PYTORCH_BUILD_SUFFIX="+aiu.kineto."$KINETO_VERSION
CONDA_ENV_NAME="buildenv-torch"
CONDA_DIR="$HOME/miniconda"

_SRC=${PYTORCH_SRC:-/project_src/}
_KINETO_DIR=${KINETO_DIR:-$(pwd)}

PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")

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

  if [ ! -d "pytorch" ]; then
    echo "Cloning PyTorch $PYTORCH_VERSION..."
    git clone --recursive -b "v$PYTORCH_VERSION" https://github.com/pytorch/pytorch.git
    cd pytorch
    git submodule sync
    git submodule update --init --recursive --jobs 1

    echo "Replacing Kineto with the aiu-kineto"
    rm -rf third_party/kineto
    cp -r ${_KINETO_DIR} third_party/kineto
  else
    echo "PyTorch repo already exists. Doing nothing..."
  fi
}

function build_pytorch() {
  echo "Building PyTorch $PYTORCH_VERSION"
  cd $_SRC/pytorch

  rm -rf build dist

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

  export PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}${PYTORCH_BUILD_SUFFIX}"
  export PYTORCH_BUILD_NUMBER=0

  pip3 --no-cache-dir install -r requirements.txt

  python3 setup.py clean

  # Build PyTorch first to embed the OpenMP libraries into the wheel package
  python3 setup.py build --verbose 2>&1 | tee build.log

  # Copy the OpenMP libraries into the appropriate PyTorch lib directory within the build
  if [[ "$ARCH" == "x86_64" ]]; then
    cp $CONDA_PREFIX/lib/libgomp* build/lib.linux-$ARCH-cpython-${PYTHON_MAJOR}${PYTHON_MINOR}/torch/lib/
    cp $CONDA_PREFIX/lib/libomp* build/lib.linux-$ARCH-cpython-${PYTHON_MAJOR}${PYTHON_MINOR}/torch/lib/
  fi

  python3 setup.py bdist_wheel --python-tag "$PYTHON_TAG" --verbose 2>&1 | tee -a build.log

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
