## Running Performance Tests

Run tests from the root source directory: `kineto`.

These tests are based on the script designed for performance benchmarking:

[`pt2bench/bert/z-script/llm-program.py`](https://github.ibm.com/ai-chip-toolchain/pt2bench/tree/master/bert/z-script/llm-program.py)

### Example of pt2bench usage:

```bash
python3 llm-program.py 
   --model bert-base-uncased 
   --batch-size 8 
   --seq-len 128 
   --test-runs 300 
   --compile-backend sendnn 
   --mode inference
```

## Building PyTorch and Running the benchmarks:
```bash
pytest e2e_tests/test_build_run_benchmarks.py -s -v
```

## Verifying the results:
```bash
pytest e2e_tests/test_verify_perf_regression.py -s -v
```

> Note: The performance regression test will fail if the compiled PyTorch model with aiu-kineto backend shows a slowdown greater than 2 milliseconds.