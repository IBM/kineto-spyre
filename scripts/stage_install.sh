#!/bin/bash
set -e

SEN_PROJECT_SRC=${SEN_PROJECT_SRC:-/project_src}
PYTORCH_PROJECT_SRC=${PYTORCH_PROJECT_SRC:-$SEN_PROJECT_SRC/pytorch}

echo "Install PyTorch with aiu-kineto"
set -x #echo on
pip3 install --no-deps --force-reinstall --user $PYTORCH_PROJECT_SRC/dist/*.whl
set +x #echo off