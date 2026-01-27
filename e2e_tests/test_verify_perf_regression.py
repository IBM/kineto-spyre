#!/usr/bin/env pytest

import os
import re
import json
from pathlib import Path

import pytest

THRESHOLD = 10  # 10%

def extract_version(filename):
    match = re.search(r"torchversion--([\d\.]+)\+([\w\.]+)", filename)
    if match:
        return match.group(2)
    return None

def extract_prof(filename):
    match = re.search(r"prof--(?P<prof>[^.]+)", filename)
    return match.group('prof') if match else None

def load_experiment_data(directory):
    results = dict()
    print("\nScenarios:")
    for file in Path(directory).glob("torch*.json"):
        try:
            version = extract_version(file.name)
            prof = extract_prof(file.name)
            with open(file, 'r') as f:
                data = json.load(f)

            elapsed_time_str = data.get("measurement", {}).get("avg_latency", "")
            model_name = data.get("model", {}).get("config", {}).get("_name_or_path", "unknown")

            match = re.match(r"([\d.]+)", elapsed_time_str)
            elapsed_time = float(match.group(1)) if match else None

            if version and elapsed_time and prof is not None:
                if version == "cpu":
                    version = "original"
                if model_name not in results:
                    results[model_name] = dict()
                if prof not in results[model_name]:
                    results[model_name][prof] = dict()
                results[model_name][prof][version] = elapsed_time
            print(f"{model_name}, Backend = Sendnn, Enabled Profiler = {prof}, PyTorch version = {version}, Elapsed Time = {elapsed_time}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file.name}: {e}")
    return results


@pytest.mark.parametrize("data_dir", ["e2e_tests/pt2bench/bert/output"])
def test_latency_deltas_within_threshold(data_dir):
    results = load_experiment_data(data_dir)

    assert results, f"No results loaded from {data_dir}"

    print("\nScenario Comparison:")

    # results -> {'roberta-base': {
    #   'sentient': {'cpu': 0.003368, 'aiu.kineto': 0.003366},
    #   'cpu': {'cpu': 0.00847, 'aiu.kineto.0.5': 0.008446}
    for model_name, profs in results.items():
        for prof, torch_versions in profs.items():
            orig_torch_time = None
            aiu_kineto_time = None

            for k in torch_versions:
                if 'original' in k:
                    orig_torch_time = torch_versions[k]
                elif 'aiu' in k:
                    aiu_kineto_time = torch_versions[k]

            assert orig_torch_time is not None, f"Missing CPU time for {model_name}"
            assert aiu_kineto_time is not None, f"Missing AIU time for {model_name}"

            ratio = ((aiu_kineto_time - orig_torch_time) / orig_torch_time) * 100
            
            print(f"{model_name}, prof = {prof}, ratio ((aiu_kineto_time - orig_torch_time) / orig_torch_time) * 100 = {ratio}")

            # If ratio > THRESHOLD: AIU is THRESHOLD percent slower
            # If ratio < 0: AIU is faster
            assert ratio <= THRESHOLD, (
                f"Latency difference for {model_name} exceeds threshold: aiu.kineto is {abs(ratio):.2f}% slower (threshold = {THRESHOLD}%)"
            )
