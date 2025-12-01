#!/bin/bash

echo "Testing PyTorch with aiu-kineto wheel"

KINETO_SRC_DIR=${KINETO_SRC_DIR:-/project_src/kineto}

echo "Install requirements"
pip3 install --upgrade --force-reinstall -r e2e_tests/requirements.txt

# If aiu pytorch is aready installed due to previous execution.
# we uninstall it for the test purpose
if [[ "$(pip3 show torch 2>/dev/null)" == *aiu* ]]; then
    pip3 uninstall -y torch
fi

cd $KINETO_SRC_DIR
pytest e2e_tests/test_build_run_benchmarks.py -s -v
pytest e2e_tests/test_verify_perf_regression.py -s -v
cd -

exit 0