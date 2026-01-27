#!/bin/bash
set -e

SEN_PROJECT_SRC=${SEN_PROJECT_SRC:-/project_src}
SEN_PROJECT_PACKAGE=${SEN_PROJECT_PACKAGE:-/project_package}

echo "Package PyTorch with aiu-kineto"
set -x #echo on
cp $SEN_PROJECT_SRC/pytorch/dist/*.whl $SEN_PROJECT_PACKAGE/
set +x #echo off