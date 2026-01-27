#!/usr/bin/env pytest

import os
import re
import json
import glob
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pytest

# Config
PT2BENCH_DIR = Path("e2e_tests/pt2bench/bert/z-script")
BUILD_PYTORCH_SCRIPT = "./scripts/build_pytorch.sh"
WHELL_FILE = "/project_src/pytorch/dist/*.whl"

ENV_VARS = {
    "DTLOG_LEVEL": "error",
    "TORCH_SENDNN_LOG": "CRITICAL",
    "DT_DEEPRT_VERBOSE": "-1",
    "FLEX_OVERWRITE_NMB_FRAME": "1",
    "FLEX_UNLINK_DEVMEM": "false",
    "DATA_PREC": "fp16",
    "SENCORES": "32",
    "LOG_LEVEL": "INFO",
}

@pytest.fixture
def benchmark_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(ENV_VARS)
    return env

@pytest.fixture
def curr_torch_version() -> str:
    return get_torch_version()

def get_torch_version() -> str:
    return subprocess.check_output(
        ["python3", "-c", "import torch; print(torch.__version__)"],
        text=True,
    ).strip()

# Parametrize benchmark cases
# The order of metafunc.parametrize calls matters. Parameters defined earlier sweep slower; later ones sweep faster.
def pytest_generate_tests(metafunc):
    metafunc.parametrize("torch_version", ["2.5.1+cpu", "2.5.1+aiu.kineto.0.5"])
    metafunc.parametrize("profile", [False, True])
    metafunc.parametrize("backend", ["sendnn"])
    # metafunc.parametrize("model", ["roberta-base", "bert-large-uncased"])
    metafunc.parametrize("model", ["roberta-base"])
    metafunc.parametrize("batch_size", ["08mb"])
    metafunc.parametrize("sequence", ["128seq"])
    metafunc.parametrize("num_tx", ["300"])

def build_aiu_kineto_pytorch():
    wheels = glob.glob(WHELL_FILE)
    print("Found PyTorch wheels\ in:", wheels)
    if not wheels:
        process = subprocess.Popen(
            [BUILD_PYTORCH_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream and print output for debug purpose
        for line in process.stdout:
            print(line, end='')

        process.stdout.close()
        returncode = process.wait()

        assert returncode == 0
        wheels = glob.glob(WHELL_FILE)
        assert wheels

def install_aiu_kineto_pytorch():
    wheels = glob.glob(WHELL_FILE)
    if not wheels:
        raise FileNotFoundError("No .whl files found in pytorch/dist/")

    cmd = [
        "pip3",
        "install",
        "--no-deps",
        *wheels,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')  # Live printing of each line

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

def uninstall_aiu_kineto_pytorch():
    cmd = [
        "pip3",
        "uninstall",
        "-y",
        "torch"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')  # Live printing of each line

    process.wait()

def test_benchmark_scenario(
    torch_version: str,
    backend: str,
    model: str,
    batch_size: str,
    sequence: str,
    num_tx: int,
    benchmark_env: Dict[str, str],
    curr_torch_version: str,
    profile: str,
):
    batch_size_val = str(int(re.search(r'\d+', batch_size).group()))
    seq_len_val = str(int(re.search(r'\d+', sequence).group()))
    benchmark_env["DT_OPT"] = "autopilot=0"

    # Install the PyTorch version with AIU-profiler if needed
    print("curr_torch_version", curr_torch_version)
    print("torch_version", torch_version)
    print("curr_torch_version != torch_version", curr_torch_version != torch_version)
    print("aiu in torch_version", "aiu" in torch_version)
    print("profile", profile)

    if curr_torch_version != torch_version:
        if "aiu" in torch_version:
            print("Going to install pytorch")
            build_aiu_kineto_pytorch()
            install_aiu_kineto_pytorch()
        else:
            # make sure that we only uninstall the aiu-kineto pytorch
            if "aiu" in curr_torch_version:
                uninstall_aiu_kineto_pytorch(torch_version)

        curr_torch_version = torch_version = get_torch_version()

    output_file = f"torchversion--{torch_version}-backend--{backend}-model--{model}-n--{num_tx}-len--{seq_len_val}-b--{batch_size_val}-prof--{profile}.json"
    program = PT2BENCH_DIR / "llm-program.py"
    cmd = [
        "python3", program,
        "--model", model,
        "--batch-size", batch_size_val,
        "--seq-len", seq_len_val,
        "--warmup-runs", "5",
        "--test-runs", str(num_tx),
        "--compile-backend", backend,
        "--mode", "inference",
        "--output", str(output_file)
    ]

    if profile:
        cmd.append("--profile")

    subprocess.run(cmd, check=True, env=benchmark_env)

    output_dir = PT2BENCH_DIR / "../output"
    output_file = output_dir / output_file
    assert output_file.exists(), f"Output file not found: {output_file}"
    with open(output_file) as f:
        data = json.load(f)
        assert isinstance(data, dict), "Output file must be a JSON object"
        assert "avg_latency" in data or "measurement" in data, "Expected 'avg_latency' data in output"