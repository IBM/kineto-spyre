#!/bin/bash
set -e
echo "Build PyTorch with aiu-kineto wheel"

KINETO_SRC_DIR=${KINETO_SRC_DIR:-/project_src/kineto}
SEN_PROJECT_SRC=${SEN_PROJECT_SRC:-/project_src}

set -x #echo on
cd ${KINETO_SRC_DIR}
export PYTORCH_SRC=$SEN_PROJECT_SRC
./scripts/build_pytorch.sh
cd -
set +x #echo off
