---
sidebar_label: AIU Torch Profiler
---
# User's Guide

## Table of Contents

<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [User's Guide](#users-guide)
  - [Table of Contents](#table-of-contents)
  - [Request access to AIUs](#request-access-to-aius)
  - [Install PyTorch Profiler with Support for AIU Events](#install-pytorch-profiler-with-support-for-aiu-events)
    - [1. Request Access to Artifactory](#1-request-access-to-artifactory)
    - [2. Install the PyTorch Profiler with AIU Support](#2-install-the-pytorch-profiler-with-aiu-support)
  - [Use PyTorch profiler with support for AIU events](#use-pytorch-profiler-with-support-for-aiu-events)
    - [1. Register and Activate the AIU Runtime Backend](#1-register-and-activate-the-aiu-runtime-backend)
    - [2. Profile Your Code Using the Profiler Context](#2-profile-your-code-using-the-profiler-context)
    - [3. Print profiling results (Optional)](#3-print-profiling-results-optional)
    - [4. Advanced PyTorch Features](#4-advanced-pytorch-features)
  - [Using PyTorch Profiler with vLLM](#using-pytorch-profiler-with-vllm)
  - [Visualize the results](#visualize-the-results)
    - [1. Chrome](#1-chrome)
    - [2. TensorBoard](#2-tensorboard)
      - [Environment](#environment)
      - [Pre-Installation (recommended)](#pre-installation-recommended)
      - [Installation](#installation)
      - [Launching TensorBoard](#launching-tensorboard)
      - [Viewing TensorBoard](#viewing-tensorboard)
    - [3. Perfetto](#3-perfetto)
  - [Understanding the AIU Runtime and Hardware events](#understanding-the-aiu-runtime-and-hardware-events)
    - [1. Runtime events](#1-runtime-events)
    - [2. Hardware events](#2-hardware-events)
  - [Known Issues](#known-issues)
  
<!-- /TOC -->

## Request access to AIUs

Follow the official guide for both OpenShift and Baremetal environments.
[IBM AIU Developer Information](https://github.ibm.com/ai-chip-toolchain/aiu-release-information/wiki/IBM-AIU-Developer-Information)

## Install PyTorch Profiler with Support for AIU Events

### 1. Request Access to Artifactory

To use the current release container, you will need **read access to the IBM internal artifactory repository `sys-power-hpc-pypi-local` **.

1. [Request artifactory **read** access at SWAT](https://github.ibm.com/ai-chip-toolchain/aiu-release-information/wiki/SWAT-Access-to-AIU-Development-Resources#Artifactory-Access)
2. Once access is granted, follow the instructions in the following link to retrieve your access token:
[Artifactory Authentication Access Tokens](https://taas.cloud.ibm.com/guides/artifactory-authentication-access-tokens.md)
3. Contact Eun Kyung Lee (eunkyung.lee@us.ibm.com) for any issues. 

### 2. Install the PyTorch Profiler with AIU Support

After obtaining your token, you can install the AIU-Kineto PyTorch profiler using the appropriate wheel for your system architecture.

It will request your W3 username and artifactory access token.

> We recommend using the `--no-deps` option with `pip3 install` to prevent unwanted library upgrades that could potentially break the environment.

**For x86:**

Pytorch 2.5.1
```bash
pip3 install --no-deps https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/torch/x86_64/stable/2.5.1%2Baiu.kineto.1.0/torch-2.5.1%2Baiu.kineto.1.0-cp312-cp312-linux_x86_64.whl
```
Pytorch 2.7.1
```bash
pip3 install --no-deps https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/torch/x86_64/stable/2.7.1%2Baiu.kineto.1.0/torch-2.7.1%2Baiu.kineto.1.0-cp312-cp312-linux_x86_64.whl
```

**For IBM Z (s390x):**
```bash
pip3 install --no-deps https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/torch/s390x/stable/2.5.1%2Baiu.kineto.1.0/torch-2.5.1%2Baiu.kineto.1.0-cp312-cp312-linux_s390x.whl
```

**For IBM Power:**
```bash
pip3 install --no-deps https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/torch/ppc64le/stable/2.5.1%2Baiu.kineto.1.0/torch-2.5.1%2Baiu.kineto.1.0-cp312-cp312-linux_ppc64le.whl
```

> To install from source, check the [Developer's Guide](https://github.ibm.com/ai-chip-toolchain/kineto/blob/main/docs/developer-guide.md) (This repository is configured with restricted access permissions) 

## Use PyTorch profiler with support for AIU events

To use the PyTorch profiler with support for AIU events, follow these steps:

### 1. Register and Activate the AIU Runtime Backend

PyTorch provides the `PrivateUse1` key as an official mechanism for integrating custom device backends. By registering a backend under this key, we can enable support for profiling events related to that backend.

The following code registers the AIU backend, generating the required methods and properties, and making PrivateUse1 available for use with the PyTorch profiler:
```python
from torch.profiler import profile, record_function, ProfilerActivity

torch.utils.rename_privateuse1_backend("aiu")
torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
torch.utils.generate_methods_for_privateuse1_backend()
```

### 2. Profile Your Code Using the Profiler Context

Wrap your target code in a `with` statement using the profiler context manager. This can enable both CPU and AIU profiling:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                record_shapes=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/roberta')) as prof:
    model = torch.compile(model, backend="sendnn")
    outputs = model(**inputs)
```

Alternatively, the following non-context manager start/stop is supported as well:

```python
prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                record_shapes=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/roberta'))
prof.start()
model = torch.compile(model, backend="sendnn")
outputs = model(**inputs)
prof.stop()
```

> See [torch_roberta_pytorch.py](https://github.ibm.com/ai-chip-toolchain/kineto/blob/main/docs/example/torch_roberta_pytorch.py) for a full example. (Only available if you have access to Kineto Repo)

### 3. Print profiling results (Optional)

After the profiler context ends, you can print a summary of the profiling results. 

Note that "CUDA" is replaced with "AIU" for consistency, we will support it natively in the future.

```python
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10).replace("CUDA", "AIU"))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10).replace("CUDA", "AIU"))
```

The print will show a table with the summary of the kernel/ops profile statistics:
```
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                       Name    Self CPU %      Self CPU   CPU total %     CPU total   CPU time avg     Self AIU   Self AIU %    AIU total     AIU time avg    # of Calls
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
               Memcpy (HtoD)         0.00%       0.000us         0.00%       0.000us       0.000us      102.424s        88.10%      102.424s        1.552s            66
               Memcpy (DtoH)         0.00%       0.000us         0.00%       0.000us       0.000us        7.169s         6.17%        7.169s        3.585s             2
                   embedding         0.00%       0.000us         0.00%       0.000us       0.000us        3.583s         3.08%        3.583s        3.583s             1
             Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us        2.844s         2.45%        2.844s     142.192ms            20
              add_1_transfer         0.00%       0.000us         0.00%       0.000us       0.000us     236.821ms         0.20%     236.821ms     236.821ms             1
    TorchDynamo Cache Lookup         0.00%       4.000us         0.00%       4.000us       0.333us       0.000us         0.00%       0.000us       0.000us            12
compile_inner (dynamo_timed)         3.31%        1.416s        92.72%       39.700s       39.700s       0.000us         0.00%       0.000us       0.000us             1
                 aten::clone         0.13%      57.415ms         0.16%      69.754ms     100.221us       0.000us         0.00%       0.000us       0.000us           696
         aten::empty_strided         0.09%      36.838ms         0.09%      36.936ms       6.488us       0.000us         0.00%       0.000us       0.000us          5693
                 aten::copy_         0.00%       1.894ms         0.00%       1.894ms       5.774us       0.000us         0.00%       0.000us       0.000us           328
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
```


### 4. Advanced PyTorch Features

The PyTorch profiler includes several advanced features, see the [official documentation](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) for a comprehensive guide.

Four particularly useful features to be aware of are:
- [record_function](https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html): This allows you to annotate specific regions of your code with custom tags for more detailed profiling insights.
- [schedule](https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.profile): This lets you define a profiling schedule that can skip initial iterations and reduce overhead by profiling only selected phases (e.g., warm-up, active, repeat).
- [on_trace_ready](https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.profile): This allows to write the profiler output in tensorboard json files.
- [with_stack](https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.profile): Add additional overhead but enables additional information (file and line number) for the python operations.

## Using PyTorch Profiler with vLLM

The [vllm-spyre](https://github.com/vllm-project/vllm-spyre) project now supports starting and stopping the PyTorch Profiler with AIU events at runtime.

To get started, follow the instructions provided [here](https://github.com/vllm-project/vllm-spyre/pull/176).

## Visualize the results

The output json file can be viewed in chrome, perfetto, or tensorboard.

### 1. Chrome

 * Open url `chrome://tracing` and click the `load` button (or drag and drop the json file onto the tracing window)

 * Users will find these online references useful:
   1. [A beginner’s guide to Chrome tracing](https://nolanlawson.com/2022/10/26/a-beginners-guide-to-chrome-tracing).
   2. [Google’s Trace Viewer Guide](https://techblog.greeneye.ag/blog/googles-trace-viewer-as-a-tool-for-code-profiling).


### 2. TensorBoard

*TensorBoard* is a standalone dashboard visualization toolkit for performance trace data in JSON format. Given its popularity, PyTorch developed a torch-profiler plugin to TensorBoard, `torch-tb-profiler`.

We have extended the functionality of the `torch-tb-profiler` plugin to enhance support for AIU traces. The upstream version of the plugin is officially [deprecated](https://github.com/pytorch/kineto/issues/857) and is not guaranteed to be stable with modern PyTorch Profiler traces. For these reasons, it is 
suggested that you install our modified version of the plugin rather than the version currently available on PyPi.

#### Environment

It is recommended to install and a run TensorBoard on your local machine (or somewhere else where you can access a graphic web-browser). Alternatively, [port-forwarding](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server) can be used.

#### Pre-Installation (recommended)

Create an empty Python environment to avoid dependency conflicts

```bash
# create the environment
python3 -m venv tb-env

# activate the environment
source tb-env/bin/activate
```

#### Installation

```bash
cd kineto/tb_plugin
pip3 install -e .
```

#### Launching TensorBoard

Note: Ensure your Python environment is active if using one

```
tensorboard --logdir=$TRACE_DIR
```

#### Viewing TensorBoard

1. Launch Google Chrome (other browsers may not load correctly)
2. Navigate to the TensorBoard url
    * By default, TensorBoard will be available at http://localhost:6006/, but this can be checked in the command output


### 3. Perfetto

Go to `https://ui.perfetto.dev/` and select `open trace file` (or drag and drop the json file onto the tracing window). Note that this is an online service even if the current claim is that everything runs locally in your browser.

## Understanding the AIU Runtime and Hardware events

### 1. Runtime events

| Runtime Events               | Summary                                                                                |
|------------------------------|----------------------------------------------------------------------------------------|
| aiuLaunchControlBlocks       | Launches a set of AIU ControlBlocks (kernel and memory copies) to run on the device.   |
| aiuInitGraph                 | Initializes a Sendnn graph to be executed on the device.                          |
| aiuMalloc                    | Allocates memory on the device.                                                        |
| aiuResizeTensorAllocation    | Resizes the memory allocation for a tensor.                                            |
| aiuLaunchSuperNode           | Launches a supernode, which is a higher-level computational unit composed of sub-nodes representing operations.|
| aiuSuperNodeExecution        | Executes a supernode on the device. If autopilot is off each sub-nodes becomes a supernode.                    |
| aiuGraphExecution            | Executes the entire computation graph on the device, i.e all the supernodes.                                   |
| aiuNodeCompute               | Executes a single node within the computation graph, both CPU or AIU operations                                |
| aiuDataConvert               | Converts data between different formats or precisions, e.g. `float16` \<-\> `sen_fp16`.                      |
| aiuInitScheduler             | Initializes the scheduler managing graph execution.                                    |
| aiuCreateVirtualAddresses    | Creates virtual addresses for memory management.                                       |
| aiuLaunchScheduleCompute     | Launches a scheduled computation task.                                                 |
| aiuScheduleWait              | Waits for the completion of a scheduled task.                                          |
| aiuPrepareDMAs               | Prepares DMA (Direct Memory Access) operations.                                        |
| aiuPrepareAndSyncRDMA        | Prepares and synchronizes RDMA operations.                                             |
| aiuClearCache                | Clears the device's scratchpad cache.                                                             |
| aiuPreloadCache              | Preloads data into the device scratchpad cache.                                                   |
| aiuLaunchComputeStream       | Launches a computation stream of CBs and wait for their completion.                                  |
| aiuRDMABarrier1              | First RDMA synchronization barrier. (Multi-AIUs communication only)                     |
| aiuPostRDMAKeys              | Posts RDMA keys for access control. (Multi-AIUs communication only)                     |
| aiuRDMABarrier2              | Second RDMA synchronization barrier. (Multi-AIUs communication only)                    |
| aiuFetchRDMAKeys             | Fetches RDMA keys for access. (Multi-AIUs communication only)                           |
| aiuUpdateRDMACBs             | Updates RDMA command blocks. (Multi-AIUs communication only)                           |
| aiuRDMABarrier3              | Third RDMA synchronization barrier. (Multi-AIUs communication only)                    |
| aiuCheckRDMADeadlock         | Checks for potential RDMA deadlocks. Only visible when `FLEX_HDMA_CHECK_FOR_DEADLOCK=True`. (Multi-AIUs communication only)|
| aiuFileTransferDtoF          | Transfers data from device to file.                                                    |
| aiuFileTransferMtoF          | Transfers data from memory to file.                                                    |
| aiuFileTransferFtoD          | Transfers data from file to device.                                                    |
| aiuFileTransferFtoM          | Transfers data from file to memory.                                                    |
| aiuDataTransferDtoH          | Transfers data from device to host.                                                    |
| aiuDataTransferHtoD          | Transfers data from host to device.                                                    |
| aiuClockCalibration          | Calibrates the device's clock for accurate timing.                                     |
| aiuCompileGraph              | Compiles the computation graph for execution, this is part of sendnn operation.        |


### 2. Hardware events

| Hardware Events              | Summary                                                       |
|------------------------------|---------------------------------------------------------------|
| Kernel                       | AIU tensor core or specialized core operation. Represents AIU kernel execution. |
| Memcpy (HtoD)                | Data transfer from host memory to AIU (DMI).  |
| Memcpy (DtoH)                | Data transfer from AIU to host memory (DMO).  |
| Memcpy (DtoD)                | Data transfer from AIU to AIU but passing through host memory. (TBD) |
| Memcpy (PtoP)                | Directly data transfer between AIUs. (TBD) |
| Memory (Allocation) or Memset (Device) | Allocates memory on the AIU device.|
| Memory (Release)             | Frees previously allocated memory on the AIU device.|


## Known Issues

- Some executions may show slightly misaligned events due to clock synchronization issues between the device and the CPU. We are working on improving this.
