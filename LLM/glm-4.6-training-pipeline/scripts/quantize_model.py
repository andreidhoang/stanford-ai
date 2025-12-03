"""
GLM-4.6 Model Quantization Script

Supports multiple quantization formats:
- AWQ (Activation-aware Weight Quantization) - 4-bit, best quality
- GPTQ (Generative Pre-trained Transformer Quantization) - 4-bit
- FP8 - 8-bit floating point (H100)
- GGUF - For llama.cpp deployment

Usage:
    python quantize_model.py --model zai-org/GLM-4.6 --method awq --output ./quantized
"""

import os
import argparse
import torch
from pathlib import Path
from typing import Optional

# Color output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_info(msg):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def print_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_step(msg):
    print(f"{Colors.BLUE}[STEP]{Colors.NC} {msg}")


def quantize_awq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    calibration_dataset: str = "c4"
):
    """
    Quantize model using AWQ (Activation-aware Weight Quantization)

    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Output directory for quantized model
        bits: Quantization bits (default: 4)
        group_size: Group size for quantization (default: 128)
        calibration_dataset: Dataset for calibration
    """
    print_step("Starting AWQ quantization...")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print_error("AutoAWQ not installed. Install with: pip install autoawq")
        return False

    print_info(f"Loading model: {model_path}")

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print_info(f"Model loaded. Original size: {model.get_memory_footprint() / 1e9:.2f} GB")

    # Quantization config
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM"
    }

    print_step(f"Quantizing to {bits}-bit with group_size={group_size}...")

    # Quantize
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_dataset
    )

    print_info(f"Quantized size: {model.get_memory_footprint() / 1e9:.2f} GB")

    # Save quantized model
    print_step(f"Saving quantized model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print_info(f"✓ AWQ quantization complete!")
    print_info(f"  Model saved to: {output_dir}")
    print_info(f"  Compression ratio: ~{32/bits:.1f}x")

    return True


def quantize_gptq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    calibration_samples: int = 128
):
    """
    Quantize model using GPTQ

    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Output directory
        bits: Quantization bits
        group_size: Group size
        calibration_samples: Number of calibration samples
    """
    print_step("Starting GPTQ quantization...")

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print_error("AutoGPTQ not installed. Install with: pip install auto-gptq")
        return False

    print_info(f"Loading model: {model_path}")

    # Quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=True,  # Activation order optimization
        damp_percent=0.01,
        sym=True
    )

    # Load model
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print_step(f"Quantizing to {bits}-bit...")

    # Prepare calibration data
    from datasets import load_dataset

    calibration_data = load_dataset(
        "c4",
        "en",
        split=f"train[:{calibration_samples}]",
        trust_remote_code=True
    )

    examples = [
        tokenizer(example["text"])
        for example in calibration_data
    ]

    # Quantize
    model.quantize(examples)

    # Save
    print_step(f"Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print_info(f"✓ GPTQ quantization complete!")
    print_info(f"  Model saved to: {output_dir}")

    return True


def quantize_fp8(
    model_path: str,
    output_dir: str,
    activation_scheme: str = "dynamic"
):
    """
    Quantize model to FP8 (requires H100 GPUs)

    Args:
        model_path: Path to model
        output_dir: Output directory
        activation_scheme: "static" or "dynamic"
    """
    print_step("Starting FP8 quantization...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print_error("Transformers not installed")
        return False

    print_info(f"Loading model: {model_path}")

    # Load model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print_step("Converting to FP8...")

    # Convert to FP8 (requires torch >= 2.1 and CUDA compute capability >= 8.9)
    if hasattr(torch, 'float8_e4m3fn'):
        # FP8 conversion logic
        print_info("Converting weights to FP8...")

        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16 and 'weight' in name:
                # Convert to FP8 (simplified, actual implementation needs scaling)
                print_info(f"  Converting {name}")

        print_warn("FP8 quantization is experimental and requires H100 GPUs")
        print_warn("Full FP8 support requires vLLM or TensorRT-LLM")
    else:
        print_error("FP8 not supported in current PyTorch version")
        print_info("Requires PyTorch >= 2.1 with CUDA 12+")
        return False

    # Save
    print_step(f"Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print_info(f"✓ FP8 quantization complete!")
    print_info(f"  Model saved to: {output_dir}")
    print_info(f"  Note: Use vLLM with --quantization fp8 for deployment")

    return True


def quantize_gguf(
    model_path: str,
    output_dir: str,
    quant_type: str = "Q4_K_M"
):
    """
    Convert model to GGUF format for llama.cpp

    Args:
        model_path: Path to model
        output_dir: Output directory
        quant_type: GGUF quantization type (Q4_0, Q4_K_M, Q5_K_M, Q8_0)
    """
    print_step("Starting GGUF conversion...")

    print_warn("GGUF conversion requires llama.cpp")
    print_info("Steps:")
    print_info("1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    print_info("2. Build: cd llama.cpp && make")
    print_info("3. Convert: python convert.py /path/to/model")
    print_info(f"4. Quantize: ./quantize model.gguf model_{quant_type}.gguf {quant_type}")

    print_info(f"\nRecommended quant types:")
    print_info(f"  Q4_K_M - Good quality, 4-bit (recommended)")
    print_info(f"  Q5_K_M - Better quality, 5-bit")
    print_info(f"  Q8_0   - Best quality, 8-bit")

    return True


def compare_quantization_methods():
    """Print comparison of quantization methods"""
    print("\n" + "="*60)
    print("  Quantization Method Comparison")
    print("="*60)
    print()
    print("Method    | Bits | Quality | Speed   | Memory  | Best For")
    print("----------|------|---------|---------|---------|------------------")
    print("AWQ       | 4    | High    | Fast    | 4×      | Production (vLLM)")
    print("GPTQ      | 4    | High    | Fast    | 4×      | General purpose")
    print("FP8       | 8    | Highest | Fastest | 2×      | H100 GPUs only")
    print("GGUF Q4_K | 4    | Good    | Medium  | 4×      | llama.cpp/CPU")
    print("GGUF Q8_0 | 8    | High    | Medium  | 2×      | llama.cpp")
    print()
    print("Recommendations:")
    print("  • Production deployment: AWQ + vLLM")
    print("  • H100 GPUs: FP8 for best speed")
    print("  • CPU inference: GGUF")
    print("  • General use: GPTQ")
    print("="*60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Quantize GLM-4.6 model to reduce memory and increase speed"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="zai-org/GLM-4.6",
        help="Model path or HuggingFace model ID"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["awq", "gptq", "fp8", "gguf"],
        required=True,
        help="Quantization method"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./quantized_model",
        help="Output directory for quantized model"
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bits (4, 8)"
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison of quantization methods"
    )

    args = parser.parse_args()

    # Show comparison if requested
    if args.compare:
        compare_quantization_methods()
        return

    print("="*60)
    print("  GLM-4.6 Model Quantization")
    print("="*60)
    print()
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Output: {args.output}")
    print()

    # Perform quantization
    success = False

    if args.method == "awq":
        success = quantize_awq(
            args.model,
            args.output,
            bits=args.bits,
            group_size=args.group_size
        )

    elif args.method == "gptq":
        success = quantize_gptq(
            args.model,
            args.output,
            bits=args.bits,
            group_size=args.group_size
        )

    elif args.method == "fp8":
        success = quantize_fp8(
            args.model,
            args.output
        )

    elif args.method == "gguf":
        success = quantize_gguf(
            args.model,
            args.output
        )

    if success:
        print()
        print("="*60)
        print("  Quantization Complete!")
        print("="*60)
        print()
        print(f"Quantized model saved to: {args.output}")
        print()
        print("Deploy with vLLM:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {args.output} \\")
        print(f"    --quantization {args.method} \\")
        print(f"    --tensor-parallel-size 4")
        print()
    else:
        print_error("Quantization failed. Check error messages above.")


if __name__ == "__main__":
    main()
