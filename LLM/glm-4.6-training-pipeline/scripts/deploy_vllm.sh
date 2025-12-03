#!/bin/bash
# GLM-4.6 vLLM Deployment Script
#
# Deploys GLM-4.6 using vLLM for high-throughput inference
# Supports multiple deployment scenarios and configurations

set -e

# Default configuration
MODEL_NAME="${MODEL_NAME:-zai-org/GLM-4.6}"
DEPLOYMENT_SCENARIO="${DEPLOYMENT_SCENARIO:-balanced}"
GPU_COUNT="${GPU_COUNT:-4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check GPU availability
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA GPUs required for deployment."
        exit 1
    fi

    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    print_info "Detected $AVAILABLE_GPUS GPUs"

    if [ "$AVAILABLE_GPUS" -lt "$GPU_COUNT" ]; then
        print_warn "Requested $GPU_COUNT GPUs but only $AVAILABLE_GPUS available"
        print_warn "Adjusting GPU count to $AVAILABLE_GPUS"
        GPU_COUNT=$AVAILABLE_GPUS
    fi
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.10+"
        exit 1
    fi

    # Check vLLM
    if ! python -c "import vllm" 2>/dev/null; then
        print_error "vLLM not installed. Installing..."
        pip install vllm
    fi

    print_info "All dependencies satisfied"
}

# Function to get deployment configuration
get_deployment_config() {
    case "$DEPLOYMENT_SCENARIO" in
        "high-throughput")
            TENSOR_PARALLEL=$GPU_COUNT
            PIPELINE_PARALLEL=1
            MAX_NUM_SEQS=512
            MAX_BATCH_TOKENS=65536
            GPU_MEMORY_UTIL=0.95
            print_info "High-throughput deployment: TP=$TENSOR_PARALLEL, max_seqs=$MAX_NUM_SEQS"
            ;;

        "low-latency")
            TENSOR_PARALLEL=1
            PIPELINE_PARALLEL=1
            MAX_NUM_SEQS=1
            MAX_BATCH_TOKENS=8192
            GPU_MEMORY_UTIL=0.90
            print_info "Low-latency deployment: Single GPU, minimal batching"
            ;;

        "balanced")
            TENSOR_PARALLEL=$GPU_COUNT
            PIPELINE_PARALLEL=1
            MAX_NUM_SEQS=128
            MAX_BATCH_TOKENS=32768
            GPU_MEMORY_UTIL=0.92
            print_info "Balanced deployment: TP=$TENSOR_PARALLEL, max_seqs=$MAX_NUM_SEQS"
            ;;

        "memory-constrained")
            TENSOR_PARALLEL=$((GPU_COUNT > 2 ? GPU_COUNT : 2))
            PIPELINE_PARALLEL=1
            MAX_NUM_SEQS=64
            MAX_BATCH_TOKENS=16384
            GPU_MEMORY_UTIL=0.95
            QUANTIZATION="awq"
            print_info "Memory-constrained deployment: TP=$TENSOR_PARALLEL, quantization=$QUANTIZATION"
            ;;

        *)
            print_error "Unknown deployment scenario: $DEPLOYMENT_SCENARIO"
            print_info "Available scenarios: high-throughput, low-latency, balanced, memory-constrained"
            exit 1
            ;;
    esac
}

# Function to start vLLM server
start_vllm() {
    print_info "Starting vLLM server..."
    print_info "Model: $MODEL_NAME"
    print_info "Port: $PORT"
    print_info "GPUs: $GPU_COUNT"

    # Build command
    CMD="python -m vllm.entrypoints.openai.api_server"
    CMD="$CMD --model $MODEL_NAME"
    CMD="$CMD --host $HOST"
    CMD="$CMD --port $PORT"
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL"
    CMD="$CMD --pipeline-parallel-size $PIPELINE_PARALLEL"
    CMD="$CMD --max-num-seqs $MAX_NUM_SEQS"
    CMD="$CMD --max-num-batched-tokens $MAX_BATCH_TOKENS"
    CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTIL"
    CMD="$CMD --trust-remote-code"
    CMD="$CMD --enable-prefix-caching"

    # Add quantization if specified
    if [ ! -z "$QUANTIZATION" ]; then
        CMD="$CMD --quantization $QUANTIZATION"
        print_info "Using quantization: $QUANTIZATION"
    fi

    # Add API key if provided
    if [ ! -z "$API_KEY" ]; then
        CMD="$CMD --api-key $API_KEY"
        print_info "API authentication enabled"
    fi

    # Set environment variables
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0

    print_info "Starting server with command:"
    print_info "$CMD"
    echo ""

    # Run server
    eval $CMD
}

# Function to test deployment
test_deployment() {
    print_info "Testing deployment..."
    sleep 5  # Wait for server to start

    # Test health endpoint
    if curl -s "http://${HOST}:${PORT}/health" > /dev/null; then
        print_info "✓ Server health check passed"
    else
        print_warn "Server health check failed (might still be starting)"
    fi

    # Test generation
    print_info "Testing text generation..."
    curl -s "http://${HOST}:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_NAME"'",
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.8
        }' | python -m json.tool

    print_info "✓ Deployment test complete"
}

# Main deployment flow
main() {
    echo "==============================================="
    echo "    GLM-4.6 vLLM Deployment Script"
    echo "==============================================="
    echo ""

    # Check prerequisites
    check_gpus
    check_dependencies

    # Get configuration
    get_deployment_config

    # Start server
    start_vllm
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --scenario)
            DEPLOYMENT_SCENARIO="$2"
            shift 2
            ;;
        --gpus)
            GPU_COUNT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --test)
            test_deployment
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model name or path (default: zai-org/GLM-4.6)"
            echo "  --scenario SCENARIO    Deployment scenario: high-throughput, low-latency,"
            echo "                         balanced, memory-constrained (default: balanced)"
            echo "  --gpus N               Number of GPUs to use (default: 4)"
            echo "  --port PORT            Server port (default: 8000)"
            echo "  --host HOST            Server host (default: 0.0.0.0)"
            echo "  --api-key KEY          API key for authentication"
            echo "  --quantization TYPE    Quantization type: awq, gptq, fp8"
            echo "  --test                 Test existing deployment"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --scenario balanced --gpus 4"
            echo "  $0 --scenario high-throughput --gpus 8 --api-key mysecret"
            echo "  $0 --scenario memory-constrained --quantization awq"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main
main
