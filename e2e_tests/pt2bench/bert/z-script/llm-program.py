import os
import time
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import transformers
import logging as log
import json
import psutil
import numpy as np
import platform
from torch.profiler import profile, ProfilerActivity
from datetime import datetime


from torch_sendnn import torch_sendnn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SUPPORTED_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "deepset/bert-base-uncased-squad2",
    "deepset/bert-large-uncased-whole-word-masking-squad2"
]

QA_MODELS = [
    "deepset/bert-base-uncased-squad2",
    "deepset/bert-large-uncased-whole-word-masking-squad2"
]

BACKENDS = ["inductor", "sendnn"]
MODES = ["tokenizer", "inference", "combined"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the random seed for reproducibility
random.seed(42)

log.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# Create directories
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def trace_handler(p):
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    summary_file = f"{RESULT_DIR}/profile_summary_{p.step_num}.txt"
    with open(summary_file, "w") as f:
        f.write(output)

    p.export_chrome_trace(f"{RESULT_DIR}/trace_{p.step_num}.json")
    ops_file = f"{RESULT_DIR}/profile_operations_{p.step_num}.txt"
    with open(ops_file, "w") as f:
        f.write(
            f"{'Operation':<30} | {'CPU Time (us)':<15} | {'CUDA Time (us)':<15} | {'CPU Memory (KB)':<15} | {'CUDA Memory (KB)':<15} | {'Device Type':<15} | {'Input Shapes':<30}\n")
        f.write("-" * 150 + "\n")
        for op in p.events():
            f.write(
                f"{op.name:<30} | {op.cpu_time_total:<15} | {op.cuda_time_total:<15} | {op.cpu_memory_usage / 1024:<15.2f} | {str(op.device_type):<15} | {str(op.input_shapes):<30}\n")


def get_machine_info():
    """
    Return the machine information
    :return:
    """
    import psutil
    return {
        "platform": {
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "information": platform.platform(),
            "network_name": platform.node(),
            "OS": platform.system(),
            "type": platform.machine(),
        },
        "memory": {
            "total": f"{psutil.virtual_memory()[0] / 1000000000:.2f} GB",
            "available": f"{psutil.virtual_memory()[1] / 1000000000:.2f} GB",
            "used": f"{psutil.virtual_memory()[3] / 1000000000:.2f} GB"
        },
        "core": {
            "physical": psutil.cpu_count(logical=False),
            "logical": psutil.cpu_count(logical=True),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build_no_and_date": platform.python_build()
        }
    }


class ModelBenchmark:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")
        self.is_qa_model = args.model in QA_MODELS

        # Load model and tokenizer
        log.info(f"Loading {args.model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if self.is_qa_model:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                args.model
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                args.model
            ).to(self.device)

        # compile model
        if not args.no_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, backend=args.compile_backend)
                log.info(f"Model compiled with {args.compile_backend} backend")
            except Exception as e:
                err_msg = f"Compilation failed: {e}"
                log.error(err_msg)
                raise Exception(err_msg)

    def generate_sentence(self, target_sequence_length):
        word_list = [
            # nouns
            "cat", "dog", "car", "tree", "house", "sun", "moon", "bird", "computer", "phone",
            "book", "city", "road", "ship", "train", "cup", "chair", "table", "desk", "ball",
            # verbs
            'talk', 'run', 'cook', 'play', 'dance', 'teach', 'climb', 'fly', 'write', 'walk',
            'buy', 'read', 'eat', 'listen', 'learn', 'sing', 'jump', 'sleep', 'sell', 'think', 'drive',
            # Adjectives
            'tall', 'sad', 'new', 'old', 'loud', 'happy', 'hot', 'big', 'small', 'dark',
            'strong', 'fast', 'slow', 'short',
            # Adverbs
            "quickly", "slowly", "silently", "loudly", "happily", "sadly", "angrily", "easily",
            "freely", "safely", "deeply", "honestly", "sincerely", "sharply", "warmly",
            # Action words
            "a", "the", "these", "those", "this", "that", "they", "we", "who", "which", "whom",
            "where", "is", "are", "were", "was", "and", "or", "but", "if", "in", "on",
            "at", "by", "with", "for", "to", "from", "of", "as", "because", "so"
        ]

        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            # Start with a shorter sentence and build up
            sentence = ""
            tokens = []

            while len(tokens) < target_sequence_length:
                word = random.choice(word_list)
                new_sentence = f"{sentence} {word}" if sentence else word
                new_tokens = self.tokenizer.encode(new_sentence, add_special_tokens=False)

                # If adding this word would exceed our target, try a different word
                if len(new_tokens) > target_sequence_length:
                    continue

                sentence = new_sentence
                tokens = new_tokens

                # Early exit if we hit our target exactly
                if len(tokens) == target_sequence_length:
                    break

            if len(tokens) == target_sequence_length:
                return {
                    'sentence': sentence,
                    'actual_length': len(tokens)
                }

            attempt += 1
            log.debug(f"Attempt {attempt}: Generated {len(tokens)} tokens (target: {target_sequence_length})")

        raise ValueError(
            f"Failed to generate sentence with exactly {target_sequence_length} tokens after {max_attempts} attempts")

    def generate_sentences(self, requested_batch, target_sequence_length):
        """Generate multiple sentences with exact sequence length"""
        return [self.generate_sentence(target_sequence_length) for _ in range(requested_batch)]

    def generate_qa_pair(self, target_sequence_length):
        """Build QA pair based on sequence length"""
        topics = [
            "artificial intelligence", "machine learning", "quantum computing",
            "climate change", "neuroscience", "ancient civilizations",
            "space exploration", "biotechnology", "renewable energy"
        ]
        question_templates = [
            "What is {topic}?",
            "Explain {topic}",
            "Describe {topic}",
            "How does {topic} work?",
            "Why is {topic} important?",
            "What is the significance of {topic}?",
            "How has {topic} evolved in recent years?",
            "What are the main challenges in {topic}?",
            "Why is {topic} important for future development?",
            "Explain the concept of {topic} in simple terms."
        ]

        # Generate question
        topic = random.choice(topics)
        question = random.choice(question_templates).format(topic=topic)

        # Figure out context tokens
        question_tokens = self.tokenizer.tokenize(question)
        overhead_tokens = 3  # [CLS], 2x[SEP]
        max_context_tokens = target_sequence_length - len(question_tokens) - overhead_tokens

        # Generate context with exact token length
        base_context = (
            f"In the field of {topic}, researchers have made significant progress. "
            f"Recent studies show promising results. The technology has evolved. "
            f"Experts believe it will transform industries. Challenges remain."
        )
        context_tokens = self.tokenizer.tokenize(base_context)

        # Adjust to exact length needed
        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
        else:
            # Adding topic is important until we meet the requested sequence length
            while len(context_tokens) < max_context_tokens:
                context_tokens.extend(self.tokenizer.tokenize(f" {topic} is important."))
                if len(context_tokens) > max_context_tokens:
                    context_tokens = context_tokens[:max_context_tokens]

        context = self.tokenizer.convert_tokens_to_string(context_tokens)

        # Verify length
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=False
        )

        actual_length = inputs['input_ids'].shape[1]
        assert actual_length == target_sequence_length, \
            f"Length mismatch: {actual_length} vs {target_sequence_length}"

        return {
            'question': question,
            'context': context,
            'actual_length': actual_length,
            'topic': topic
        }

    def generate_qa_pairs(self, num_pairs, target_sequence_length):
        """Generate multiple QA pairs with exact sequence length"""
        return [self.generate_qa_pair(target_sequence_length) for _ in range(num_pairs)]

    def generate_input_batch(self, requested_batch, target_sequence_length):
        """Generate either sentences or QA pairs based on model type"""
        if self.is_qa_model:
            return self.generate_qa_pairs(requested_batch, target_sequence_length)
        else:
            return self.generate_sentences(requested_batch, target_sequence_length)

    def run_benchmark(self):
        results = {}
        final_results = self.get_final_results(self.args, num_threads_torch=torch.get_num_threads())

        final_results["system"] = {
            "cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None,
            "total_memory": psutil.virtual_memory().total,
            "torch_threads": torch.get_num_threads()
        }

        log.info(f"Testing - Batch: {self.args.batch_size}, Seq Len: {self.args.seq_len}, Mode: {self.args.mode}")

        # Generate input batch (either sentences or QA pairs)
        input_batch = self.generate_input_batch(
            requested_batch=self.args.batch_size,
            target_sequence_length=self.args.seq_len
        )

        # tokenize input in the case of testing inference only
        tokenized_input, _ = self.process_tokenizer_batch(input_batch, self.args.seq_len)

        # Warmup test
        self.warmup(input_batch, self.args.seq_len)

        # Measurement phase
        total_time = 0
        total_tokenization_time = 0
        total_inference_time = 0
        total_input_tokens = 0
        latencies = []
        log.info(f"Start measurement at {datetime.now()}")

        run_start = time.time()

        if self.args.profile:
            activities = [ProfilerActivity.CPU]
            if self.device == "cuda":
                activities.append(ProfilerActivity.CUDA)
            if args.compile_backend == "sendnn":
                activities.append(ProfilerActivity.PrivateUse1)
                torch.utils.rename_privateuse1_backend("aiu")
                torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
                torch.utils.generate_methods_for_privateuse1_backend()

            with profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=self.args.test_runs),
                    on_trace_ready=trace_handler,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
            ) as prof:
                for run in range(self.args.test_runs):
                    if self.args.mode == "tokenizer":
                        outputs, token_time = self.process_tokenizer_batch(input_batch, self.args.seq_len)
                        total_tokenization_time += token_time
                        batch_time = token_time
                        input_lengths = outputs["attention_mask"].sum(dim=1)
                    elif self.args.mode == "inference":
                        prof.step()
                        outputs, infer_time = self.process_inference_batch(tokenized_input)
                        total_inference_time += infer_time
                        batch_time = infer_time
                        input_lengths = tokenized_input["attention_mask"].sum(dim=1)
                    else:
                        prof.step()
                        outputs, tokenized_input, batch_time, token_time, infer_time = self.process_combined_batch(
                            input_batch, self.args.seq_len)
                        total_tokenization_time += token_time
                        total_inference_time += infer_time
                        input_lengths = tokenized_input["attention_mask"].sum(dim=1)
                    prof.step()
                    total_time += batch_time
                    total_input_tokens += input_lengths.sum().item()
                    latencies.append(batch_time)
        else:
            # Run without profiler
            for run in range(self.args.test_runs):
                if self.args.mode == "tokenizer":
                    outputs, token_time = self.process_tokenizer_batch(input_batch, self.args.seq_len)
                    total_tokenization_time += token_time
                    batch_time = token_time
                    input_lengths = outputs["attention_mask"].sum(dim=1)
                elif self.args.mode == "inference":
                    outputs, infer_time = self.process_inference_batch(tokenized_input)
                    total_inference_time += infer_time
                    batch_time = infer_time
                    input_lengths = tokenized_input["attention_mask"].sum(dim=1)
                else:
                    outputs, tokenized_input, batch_time, token_time, infer_time = self.process_combined_batch(
                        input_batch, self.args.seq_len)
                    total_tokenization_time += token_time
                    total_inference_time += infer_time
                    input_lengths = tokenized_input["attention_mask"].sum(dim=1)
                total_time += batch_time
                total_input_tokens += input_lengths.sum().item()
                latencies.append(batch_time)

        run_end = time.time()
        measure_time = run_end - run_start
        log.info(f"End measurement at {datetime.now()}")

        # Calculate final metrics
        avg_tokens_per_sec = total_input_tokens / total_time
        avg_latency_sec = np.mean(latencies)
        latency_variance = np.var(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        # Transaction rate calculation (batches per second)
        trans_per_sec = self.args.test_runs / total_time * self.args.batch_size
        avg_tokenization_time = (total_tokenization_time / self.args.test_runs) * 1000  # ms
        avg_inference_time = (total_inference_time / self.args.test_runs) * 1000  # ms

        # Result to display on screen
        results[(self.args.batch_size, self.args.seq_len)] = {
            "trans_per_sec": trans_per_sec,
            "avg_latency_ms": avg_latency_sec * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "p99_latency_ms": p99_latency * 1000,
            "latency_variance": latency_variance,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_input_tokens": total_input_tokens,
            "total_tokenization_time": total_tokenization_time,
            "total_inference_time": total_inference_time,
            "avg_tokenization_latency_ms": avg_tokenization_time,
            "avg_inference_latency_ms": avg_inference_time,
            "total_time": total_time,
        }

        final_results["measurement"].update({
            "start_time": run_start,
            "end_time": run_end,
            "elapsed_time": f"{measure_time:.3f} sec(s)",
            "trans_rate": f"{trans_per_sec:.3f}",
            "avg_latency": f"{avg_latency_sec:.6f} sec(s)",
            "throughput": {
                "transactions_per_second": trans_per_sec,
                "tokens_per_second": avg_tokens_per_sec
            },

            "latency": {
                "end_to_end": {
                    "average_seconds": avg_latency_sec,
                    "average_milliseconds": avg_latency_sec * 1000,
                    "p95_seconds": p95_latency,
                    "p99_seconds": p99_latency,
                    "variance": latency_variance
                },
                "components": {
                    "tokenization": {
                        "total_seconds": total_tokenization_time,
                        "avg_seconds": avg_tokenization_time,
                        "avg_milliseconds": avg_tokenization_time * 1000
                    },
                    "inference": {
                        "total_seconds": total_inference_time,
                        "avg_seconds": avg_inference_time,
                        "avg_milliseconds": avg_inference_time * 1000
                    }
                }
            }
        })

        # Log results
        self.log_results(results)

        # Display test summary
        self.print_results(results)

        # Write final results
        if self.args.output:
            outfile = os.path.join(RESULT_DIR, args.output)
            with open(outfile, "w") as f:
                json.dump(final_results, f, sort_keys=True, indent=4)
            log.info(f"Test result: {outfile}")

    def warmup(self, input_batch, seq_len):
        log.info(f"Starting warmup ({len(input_batch)} batch(s))...")

        # needed for the case of measuring only inference
        tokenized_input, _ = self.process_tokenizer_batch(input_batch, seq_len)

        for _ in range(self.args.warmup_runs):
            if self.args.mode == "tokenizer":
                self.process_tokenizer_batch(input_batch, seq_len)
            elif self.args.mode == "inference":
                self.process_inference_batch(tokenized_input)
            else:
                self.process_combined_batch(input_batch, seq_len)
        log.info(f"Warmup completed - {self.args.warmup_runs} loops")

    def process_tokenizer_batch(self, input_batch, seq_len):
        token_start = time.time()

        if self.is_qa_model:
            tokenized_input = self.tokenizer(
                [q['question'] for q in input_batch],
                [q['context'] for q in input_batch],
                max_length=seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            tokenized_input = self.tokenizer(
                [s['sentence'] for s in input_batch],
                max_length=seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

        token_time = time.time() - token_start
        return tokenized_input, token_time

    def process_inference_batch(self, tokenized_input):
        infer_start = time.time()
        with torch.no_grad():
            outputs = self.model(**tokenized_input)
        infer_time = time.time() - infer_start
        return outputs, infer_time

    def process_combined_batch(self, input_batch, seq_len):
        start_time = time.time()

        # Tokenization phase
        tokenized_input, token_time = self.process_tokenizer_batch(input_batch=input_batch, seq_len=seq_len)

        # Inference phase
        outputs, infer_time = self.process_inference_batch(tokenized_input=tokenized_input)
        total_time = time.time() - start_time
        return outputs, tokenized_input, total_time, token_time, infer_time

    def _count_model_parameters(self):
        """Counts the total number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def get_final_results(self, args, num_threads_torch):
        final_results = {
            "model": {
                "name": self.args.model,
                **({"num_parameters": f"{self._count_model_parameters():,}"} if hasattr(self, 'model') else {}),
                **({"config": self.model.config.to_dict()} if hasattr(self, 'model') else {})
            },
            **get_machine_info(),
            "config": {
                "cmdline": str(args),
                "num_transactions": self.args.test_runs,
                "seq_len": self.args.seq_len,
                "batch_size": self.args.batch_size,
                "pytorch_threads": num_threads_torch,
                "hugging_face_version": transformers.__version__,
                "mode": self.args.mode,
                "profiling_enabled": self.args.profile
            },
            "measurement": {},
        }
        return final_results

    def log_results(self, results):
        log.info("\nMETRICS SUMMARY:")
        log.info(f"{'Metric':<30} {'Value':>15}")
        log.info("-" * 50)

        for (batch_size, seq_len), res in results.items():
            log.info(f"{'Batch Size':<30} {batch_size:>15}")
            log.info(f"{'Sequence Length':<30} {seq_len:>15}")
            log.info(f"{'Throughput (trans/sec)':<30} {res['trans_per_sec']:>15.2f}")
            log.info(f"{'Throughput (tokens/sec)':<30} {res['avg_tokens_per_sec']:>15.2f}")
            log.info(f"{'Avg Latency (ms)':<30} {res['avg_latency_ms']:>15.2f}")
            log.info(f"{'P95 Latency (ms)':<30} {res['p95_latency_ms']:>15.2f}")
            log.info(f"{'P99 Latency (ms)':<30} {res['p99_latency_ms']:>15.2f}")

            if self.args.mode in ["tokenizer", "combined"]:
                log.info(f"{'Total Tokenization Time':<30} {res['total_tokenization_time']:>15.4f}s")

            if self.args.mode in ["inference", "combined"]:
                log.info(f"{'Total Inference Time':<30} {res['total_inference_time']:>15.4f}s")

            log.info(f"{'Total Input Tokens':<30} {res['total_input_tokens']:>15}")
            log.info(f"{'Total Time':<30} {res['total_time']:>15.4f}s")

            log.info("-" * 50)

    def print_results(self, results):
        log.info("\nTEST SUMMARY:")
        if self.args.mode == "tokenizer":
            log.info("Batch Size | Seq Length | trans/sec | Latency (ms) | Token Time (s) | Avg Token Lat (ms)")
            log.info("----------|------------|------------|--------------|----------------|-------------------")
            for (batch_size, seq_len), res in sorted(results.items()):
                log.info(
                    f"{batch_size:9} | {seq_len:10} | {res['trans_per_sec']:10.2f} | {res['avg_latency_ms']:12.3f} | {res['total_tokenization_time']:14.4f} | {res['avg_tokenization_latency_ms']:17.3f}")
        elif self.args.mode == "inference":
            log.info("Batch Size | Seq Length | trans/sec | Latency (ms) | Infer Time (s) | Avg Infer Lat (ms)")
            log.info("----------|------------|------------|--------------|----------------|-------------------")
            for (batch_size, seq_len), res in sorted(results.items()):
                log.info(
                    f"{batch_size:9} | {seq_len:10} | {res['trans_per_sec']:10.2f} | {res['avg_latency_ms']:12.3f} | {res['total_inference_time']:14.4f} | {res['avg_inference_latency_ms']:17.3f}")
        else:
            log.info(
                "Batch Size | Seq Length | trans/sec | Latency (ms) | Token Time (s) | Infer Time (s) | Total Time (s) | Avg Token Lat (ms) | Avg Infer Lat (ms)")
            log.info(
                "----------|------------|------------|--------------|----------------|----------------|----------------|--------------------|-------------------")
            for (batch_size, seq_len), res in sorted(results.items()):
                log.info(
                    f"{batch_size:9} | {seq_len:10} | {res['trans_per_sec']:10.2f} | {res['avg_latency_ms']:12.3f} | {res['total_tokenization_time']:14.4f} | {res['total_inference_time']:14.4f} | {res['total_time']:14.4f} | {res['avg_tokenization_latency_ms']:18.3f} | {res['avg_inference_latency_ms']:18.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Model Performance Benchmark")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS,
                        help="Model name or path")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Batch size to test")
    parser.add_argument("--seq-len", type=int, required=True,
                        help="Context sequence length to test")
    parser.add_argument("--test-runs", type=int, required=True,
                        help="Number of transactions to be executed")
    parser.add_argument("--warmup-runs", type=int, default=5,
                        help="Number of warmup runs")
    parser.add_argument("--compile-backend", required=True,
                        choices=BACKENDS,
                        help="PyTorch compilation backend")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip model compilation")
    parser.add_argument("--mode", choices=MODES, default="combined",
                        help="Test mode: tokenizer, inference, or combined")
    parser.add_argument("-o", "--output", default="results.json",
                        help="Test result filename. Default: 'results.json'")
    parser.add_argument("-p", "--profile", action="store_true",
                        help="Enable profiling during the test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark = ModelBenchmark(args)
    benchmark.run_benchmark()