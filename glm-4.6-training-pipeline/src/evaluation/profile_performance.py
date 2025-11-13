"""
GLM-4.6 Performance Profiling Tool

Profiles model performance:
- Inference latency (time to first token, tokens/second)
- Memory usage (peak GPU memory, activation memory)
- Throughput (requests/second)
- Batch processing efficiency

Usage:
    python profile_performance.py --model zai-org/GLM-4.6
    python profile_performance.py --model zai-org/GLM-4.6 --batch-sizes 1,4,8,16
"""

import argparse
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/transformers not available")


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single run"""
    batch_size: int
    seq_length: int
    time_to_first_token_ms: float
    tokens_per_second: float
    total_time_seconds: float
    peak_memory_gb: float
    throughput_requests_per_sec: float


class PerformanceProfiler:
    """Profiles GLM-4.6 performance"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        """
        Initialize profiler

        Args:
            model_path: Path to model
            device: Device to profile on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        print(f"Loading model: {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device

        print(f"Model loaded on {device}")

    def measure_latency(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 10
    ) -> Tuple[float, float, float]:
        """
        Measure inference latency

        Args:
            prompt: Input prompt
            max_new_tokens: Tokens to generate
            num_runs: Number of runs to average

        Returns:
            (mean_ttft_ms, mean_tps, std_tps)
        """
        ttft_times = []
        tps_times = []

        for _ in range(num_runs):
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Measure time to first token
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                # First token
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True
                )

            torch.cuda.synchronize()
            ttft = (time.time() - start_time) * 1000  # ms

            # Measure full generation
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True
                )

            torch.cuda.synchronize()
            total_time = time.time() - start_time

            # Calculate tokens per second
            generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            tps = generated_tokens / total_time

            ttft_times.append(ttft)
            tps_times.append(tps)

        return (
            np.mean(ttft_times),
            np.mean(tps_times),
            np.std(tps_times)
        )

    def measure_memory(
        self,
        prompt: str,
        max_new_tokens: int = 100
    ) -> float:
        """
        Measure peak GPU memory usage

        Args:
            prompt: Input prompt
            max_new_tokens: Tokens to generate

        Returns:
            Peak memory in GB
        """
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        if self.device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            return peak_memory
        else:
            return 0.0

    def measure_throughput(
        self,
        prompts: List[str],
        batch_size: int,
        max_new_tokens: int = 100
    ) -> float:
        """
        Measure throughput (requests/second)

        Args:
            prompts: List of prompts
            batch_size: Batch size
            max_new_tokens: Tokens to generate

        Returns:
            Requests per second
        """
        # Process in batches
        num_batches = len(prompts) // batch_size

        torch.cuda.synchronize()
        start_time = time.time()

        for i in range(num_batches):
            batch = prompts[i * batch_size:(i + 1) * batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        requests_per_sec = (num_batches * batch_size) / total_time

        return requests_per_sec

    def profile_batch_sizes(
        self,
        batch_sizes: List[int],
        seq_length: int = 128,
        max_new_tokens: int = 100
    ) -> List[PerformanceMetrics]:
        """
        Profile different batch sizes

        Args:
            batch_sizes: List of batch sizes to test
            seq_length: Input sequence length
            max_new_tokens: Tokens to generate

        Returns:
            List of performance metrics
        """
        results = []

        # Generate test prompts
        test_prompt = "Hello " * (seq_length // 2)

        for batch_size in batch_sizes:
            print(f"\nProfiling batch_size={batch_size}...")

            # Measure latency
            ttft, tps, _ = self.measure_latency(
                test_prompt,
                max_new_tokens=max_new_tokens,
                num_runs=5
            )

            # Measure memory
            memory = self.measure_memory(test_prompt, max_new_tokens)

            # Measure throughput
            prompts = [test_prompt] * (batch_size * 10)
            throughput = self.measure_throughput(
                prompts,
                batch_size,
                max_new_tokens
            )

            # Calculate total time
            total_time = max_new_tokens / tps

            metrics = PerformanceMetrics(
                batch_size=batch_size,
                seq_length=seq_length,
                time_to_first_token_ms=ttft,
                tokens_per_second=tps,
                total_time_seconds=total_time,
                peak_memory_gb=memory,
                throughput_requests_per_sec=throughput
            )

            results.append(metrics)

            print(f"  TTFT: {ttft:.2f} ms")
            print(f"  Tokens/sec: {tps:.2f}")
            print(f"  Peak memory: {memory:.2f} GB")
            print(f"  Throughput: {throughput:.2f} req/s")

        return results


def print_summary_table(results: List[PerformanceMetrics]):
    """Print summary table of results"""
    print("\n" + "=" * 100)
    print("  Performance Summary")
    print("=" * 100)
    print()
    print(f"{'Batch':<10} {'TTFT (ms)':<15} {'Tokens/s':<15} {'Memory (GB)':<15} {'Throughput':<15}")
    print(f"{'Size':<10} {'':<15} {'':<15} {'':<15} {'(req/s)':<15}")
    print("-" * 100)

    for metrics in results:
        print(f"{metrics.batch_size:<10} "
              f"{metrics.time_to_first_token_ms:<15.2f} "
              f"{metrics.tokens_per_second:<15.2f} "
              f"{metrics.peak_memory_gb:<15.2f} "
              f"{metrics.throughput_requests_per_sec:<15.2f}")

    print("=" * 100)
    print()


def compare_with_targets(results: List[PerformanceMetrics]):
    """Compare with target performance metrics"""
    print("\n" + "=" * 70)
    print("  Performance vs. Targets")
    print("=" * 70)
    print()

    # Target metrics (from official GLM-4.6)
    targets = {
        "ttft_ms": 50,  # Time to first token
        "tokens_per_sec": 50,  # Per-user throughput
        "memory_gb": 80,  # Per-GPU memory
    }

    # Use single-batch metrics
    if results:
        metrics = results[0]  # batch_size=1

        print(f"{'Metric':<30} {'Your Model':<20} {'Target':<20} {'Status':<10}")
        print("-" * 70)

        # TTFT
        status = "✓" if metrics.time_to_first_token_ms <= targets["ttft_ms"] else "✗"
        print(f"{'Time to First Token':<30} {metrics.time_to_first_token_ms:>8.2f} ms{'':<11} "
              f"{targets['ttft_ms']:>8} ms{'':<11} {status:<10}")

        # Tokens/sec
        status = "✓" if metrics.tokens_per_second >= targets["tokens_per_sec"] else "✗"
        print(f"{'Tokens per Second':<30} {metrics.tokens_per_second:>8.2f} tok/s{'':<8} "
              f"{targets['tokens_per_sec']:>8} tok/s{'':<8} {status:<10}")

        # Memory
        status = "✓" if metrics.peak_memory_gb <= targets["memory_gb"] else "✗"
        print(f"{'Peak Memory':<30} {metrics.peak_memory_gb:>8.2f} GB{'':<11} "
              f"{targets['memory_gb']:>8} GB{'':<11} {status:<10}")

    print("=" * 70)
    print()


def save_results(results: List[PerformanceMetrics], output_file: str):
    """Save results to JSON"""
    output = []
    for metrics in results:
        output.append({
            "batch_size": metrics.batch_size,
            "seq_length": metrics.seq_length,
            "time_to_first_token_ms": metrics.time_to_first_token_ms,
            "tokens_per_second": metrics.tokens_per_second,
            "total_time_seconds": metrics.total_time_seconds,
            "peak_memory_gb": metrics.peak_memory_gb,
            "throughput_requests_per_sec": metrics.throughput_requests_per_sec
        })

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile GLM-4.6 performance"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="zai-org/GLM-4.6",
        help="Model path"
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8",
        help="Comma-separated batch sizes to test"
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Input sequence length"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Tokens to generate"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="performance_profile.json",
        help="Output file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda, cpu)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with target metrics"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  GLM-4.6 Performance Profiling")
    print("=" * 70)
    print()
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Initialize profiler
    profiler = PerformanceProfiler(
        model_path=args.model,
        device=args.device
    )

    # Profile
    results = profiler.profile_batch_sizes(
        batch_sizes=batch_sizes,
        seq_length=args.seq_length,
        max_new_tokens=args.max_new_tokens
    )

    # Print summary
    print_summary_table(results)

    # Compare with targets
    if args.compare:
        compare_with_targets(results)

    # Save results
    save_results(results, args.output)

    print("\n" + "=" * 70)
    print("  Profiling Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
