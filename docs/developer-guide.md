# Developer's Guide

## Table of Contents

<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [Table of Contents](#table-of-contents)
- [Create a New Developer Image (Optional)](#create-a-new-developer-image-optional)
- [Clone repo](#clone-repo)
- [Build PyTorch with AIU-kineto from source](#build-pytorch-with-aiu-kineto-from-source)
- [Running the End-to-End (E2E) Tests](#running-the-end-to-end-e2e-tests)
- [General information for developers](#general-information-for-developers)

<!-- /TOC -->

## Request access to AIUs

Follow the official guide for both OpenShift and Baremetal environments.
[IBM AIU Developer Information](https://github.ibm.com/ai-chip-toolchain/aiu-release-information/wiki/IBM-AIU-Developer-Information)

## Create a New Developer Image (Optional)

Since the default container image `e2e_stable:latest` only allows a user with limited write permission, a developer might need a custom image to
rebuild the AIU software stack or install new packages.

For this usecase, reference the following instructions for [baremetal](https://github.ibm.com/ai-chip-toolchain/aiu-toolbox/blob/main/examples/baremetal/dev-image) or [OpenShift](https://github.ibm.com/ai-chip-toolchain/aiu-toolbox/tree/main/examples/openshift/dev-image).

## Clone repo

> **Note:** Do **not** forget to specify `--recurse-submodules` option.

Within the container, do:
```bash
cd /project_src
git clone --recurse-submodules git@github.ibm.com:ai-chip-toolchain/kineto.git
```

## Build PyTorch with AIU-kineto from source

Since kineto is statically linked with PyTorch, we need to rebuild PyTorch.

Run the following script in your container. The script will clone PyTorch 2.5.1 and replace its kineto sub-module with this repository.
The default folder is `/project_src`, but you can change the destination with the env variables `$KINETO_SRC_DIR` and `$PYTORCH_SRC`.

When kineto finds the `libaiupti` library in the environment, it will enable the code that collects AIU events from the runtime.

Also, the script follows the guideline of the official [PyTorch Documentation](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source) of using `conda` environment to install the necessary libraries,
such as static `mlk` for x86 among others.

```bash
./scripts/build_pytorch.sh
```

This script will generate a wheel file in `$PYTORCH_SRC/pytorch/dist`, and you can install PyTorch with the following command:

```bash
pip3 install --no-deps --force-reinstall --user $PYTORCH_SRC/pytorch/dist/*.whl
```

> We recommend using the `--no-deps` option with `pip3 install` to prevent unwanted library upgrades that could potentially break the environment.

## Running the End-to-End (E2E) Tests

> **Note:** Stage test can build PyTorch, then you do not need to build it before running the test.

```bash
./scripts/stage_test.sh
```

The End-to-End builds PyTorch (if not already built) and runs performance benchmarks for the following scenarios:

| Backed  | PyTorch version  |     | PyTorch version  | Profiler |
|---------|------------------|-----|------------------|--------|
| Sendnn  | 2.5.1+cpu        | vs. | 2.5.1+aiu.kineto | OFF |
| Sendnn  | 2.5.1+cpu        | vs. | 2.5.1+aiu.kineto | ON |

The performance regression test will fail if the compiled PyTorch with the `aiu.kineto` version shows a slowdown greater than 10% compared to the original version.

We compare the average inference time over 300 requests, excluding the warm up phase of 5 iterations.


## General information for developers

Commit messages should be signed with `git commit -s` and follow the [Conventional Commits](https://www.conventionalcommits.org).