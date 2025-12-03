"""
GLM-4.6 Benchmark Evaluation Script

Evaluates model on standard benchmarks:
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math)
- HumanEval (Code Generation)
- AIME (AI Mathematical Exam)

Usage:
    python benchmarks.py --model zai-org/GLM-4.6 --benchmark mmlu
    python benchmarks.py --model zai-org/GLM-4.6 --benchmark all
"""

import os
import argparse
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, as dict
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not installed")


# Color output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


def print_info(msg):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def print_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def print_result(msg):
    print(f"{Colors.CYAN}[RESULT]{Colors.NC} {msg}")


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation"""
    benchmark: str
    accuracy: float
    num_samples: int
    correct: int
    duration_seconds: float
    details: Dict


class ModelEvaluator:
    """Evaluates GLM-4.6 on various benchmarks"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 2048,
        batch_size: int = 1
    ):
        """
        Initialize evaluator

        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to run on ("auto", "cuda", "cpu")
            max_length: Maximum generation length
            batch_size: Batch size for evaluation
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")

        print_info(f"Loading model: {model_path}")

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

        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        print_info(f"Model loaded on {self.model.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        return generated.strip()

    def evaluate_mmlu(self, num_samples: Optional[int] = None) -> BenchmarkResult:
        """
        Evaluate on MMLU benchmark

        Args:
            num_samples: Number of samples to evaluate (None = all)

        Returns:
            Benchmark results
        """
        print_info("Evaluating on MMLU (Massive Multitask Language Understanding)...")

        if not DATASETS_AVAILABLE:
            raise ImportError("datasets required. Install with: pip install datasets")

        start_time = time.time()

        # Load MMLU dataset
        dataset = load_dataset("cais/mmlu", "all", split="test")

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        correct = 0
        total = 0

        for example in dataset:
            # Format prompt
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]

            prompt = f"Question: {question}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += "\nAnswer: "

            # Generate answer
            generated = self.generate(prompt, max_new_tokens=10, temperature=0.0)

            # Extract answer (A, B, C, or D)
            predicted_idx = None
            if generated and generated[0] in "ABCD":
                predicted_idx = ord(generated[0]) - ord('A')

            # Check correctness
            if predicted_idx == answer_idx:
                correct += 1

            total += 1

            # Progress
            if total % 100 == 0:
                print_info(f"Progress: {total}/{len(dataset)} ({correct/total*100:.1f}% correct)")

        duration = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0

        result = BenchmarkResult(
            benchmark="MMLU",
            accuracy=accuracy,
            num_samples=total,
            correct=correct,
            duration_seconds=duration,
            details={}
        )

        return result

    def evaluate_gsm8k(self, num_samples: Optional[int] = None) -> BenchmarkResult:
        """
        Evaluate on GSM8K (Grade School Math) benchmark

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Benchmark results
        """
        print_info("Evaluating on GSM8K (Grade School Math)...")

        if not DATASETS_AVAILABLE:
            raise ImportError("datasets required")

        start_time = time.time()

        # Load GSM8K
        dataset = load_dataset("gsm8k", "main", split="test")

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        correct = 0
        total = 0

        for example in dataset:
            question = example["question"]
            answer = example["answer"]

            # Extract numerical answer
            answer_num = answer.split("####")[-1].strip()

            # Format prompt
            prompt = f"Question: {question}\n\nLet's solve this step by step:\n"

            # Generate solution
            generated = self.generate(prompt, max_new_tokens=512, temperature=0.0)

            # Extract answer from generation
            # Look for "####" or final number
            predicted = None
            if "####" in generated:
                predicted = generated.split("####")[-1].strip()
            else:
                # Try to extract last number
                import re
                numbers = re.findall(r'-?\d+\.?\d*', generated)
                if numbers:
                    predicted = numbers[-1]

            # Check correctness
            if predicted and predicted.replace(",", "") == answer_num.replace(",", ""):
                correct += 1

            total += 1

            if total % 50 == 0:
                print_info(f"Progress: {total}/{len(dataset)} ({correct/total*100:.1f}% correct)")

        duration = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0

        result = BenchmarkResult(
            benchmark="GSM8K",
            accuracy=accuracy,
            num_samples=total,
            correct=correct,
            duration_seconds=duration,
            details={}
        )

        return result

    def evaluate_humaneval(self) -> BenchmarkResult:
        """
        Evaluate on HumanEval (Code Generation) benchmark

        Returns:
            Benchmark results
        """
        print_info("Evaluating on HumanEval (Code Generation)...")

        try:
            from human_eval.data import write_jsonl, read_problems
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportError:
            print_error("human_eval not installed. Install from: https://github.com/openai/human-eval")
            return None

        start_time = time.time()

        # Load problems
        problems = read_problems()

        # Generate solutions
        solutions = []
        for task_id, problem in problems.items():
            prompt = problem["prompt"]

            # Generate code
            generated = self.generate(
                prompt,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95
            )

            solutions.append({
                "task_id": task_id,
                "completion": generated
            })

            if len(solutions) % 10 == 0:
                print_info(f"Progress: {len(solutions)}/{len(problems)}")

        # Save solutions
        output_file = "humaneval_solutions.jsonl"
        write_jsonl(output_file, solutions)

        # Evaluate
        print_info("Evaluating solutions...")
        results = evaluate_functional_correctness(output_file)

        duration = time.time() - start_time

        # Parse results
        pass_at_1 = results.get("pass@1", 0.0)

        result = BenchmarkResult(
            benchmark="HumanEval",
            accuracy=pass_at_1,
            num_samples=len(problems),
            correct=int(pass_at_1 * len(problems)),
            duration_seconds=duration,
            details=results
        )

        return result


def compare_with_official(results: Dict[str, BenchmarkResult]):
    """
    Compare results with official GLM-4.6 benchmarks

    Args:
        results: Dictionary of benchmark results
    """
    print("\n" + "=" * 70)
    print("  Comparison with Official GLM-4.6")
    print("=" * 70)
    print()

    # Official GLM-4.6 results
    official = {
        "MMLU": 87.2,
        "GSM8K": 94.8,
        "HumanEval": 74.4,
        "AIME": 98.6
    }

    print(f"{'Benchmark':<15} {'Your Model':<15} {'Official':<15} {'Difference':<15}")
    print("-" * 70)

    for benchmark, official_score in official.items():
        if benchmark in results:
            your_score = results[benchmark].accuracy * 100
            diff = your_score - official_score
            diff_str = f"{diff:+.1f}%"

            print(f"{benchmark:<15} {your_score:>6.1f}%{'':<8} {official_score:>6.1f}%{'':<8} {diff_str:<15}")
        else:
            print(f"{benchmark:<15} {'N/A':<15} {official_score:>6.1f}%{'':<8} {'N/A':<15}")

    print("=" * 70)
    print()


def save_results(results: Dict[str, BenchmarkResult], output_file: str):
    """Save results to JSON file"""
    output = {}
    for benchmark, result in results.items():
        output[benchmark] = {
            "accuracy": result.accuracy,
            "accuracy_percent": result.accuracy * 100,
            "num_samples": result.num_samples,
            "correct": result.correct,
            "duration_seconds": result.duration_seconds,
            "details": result.details
        }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print_info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GLM-4.6 on standard benchmarks"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="zai-org/GLM-4.6",
        help="Model path or HuggingFace model ID"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mmlu", "gsm8k", "humaneval", "all"],
        required=True,
        help="Benchmark to run"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cuda, cpu)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with official GLM-4.6 results"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  GLM-4.6 Benchmark Evaluation")
    print("=" * 70)
    print()
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Device: {args.device}")
    print()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        device=args.device
    )

    # Run benchmarks
    results = {}

    if args.benchmark in ["mmlu", "all"]:
        result = evaluator.evaluate_mmlu(num_samples=args.num_samples)
        results["MMLU"] = result
        print_result(f"MMLU: {result.accuracy * 100:.2f}% ({result.correct}/{result.num_samples})")

    if args.benchmark in ["gsm8k", "all"]:
        result = evaluator.evaluate_gsm8k(num_samples=args.num_samples)
        results["GSM8K"] = result
        print_result(f"GSM8K: {result.accuracy * 100:.2f}% ({result.correct}/{result.num_samples})")

    if args.benchmark in ["humaneval", "all"]:
        result = evaluator.evaluate_humaneval()
        if result:
            results["HumanEval"] = result
            print_result(f"HumanEval: {result.accuracy * 100:.2f}% (pass@1)")

    # Save results
    save_results(results, args.output)

    # Compare with official
    if args.compare:
        compare_with_official(results)

    print()
    print("=" * 70)
    print("  Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
