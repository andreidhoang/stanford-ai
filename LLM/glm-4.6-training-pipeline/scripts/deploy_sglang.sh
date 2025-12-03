#!/bin/bash
# GLM-4.6 SGLang Deployment Script
#
# Deploys GLM-4.6 using SGLang for structured generation and RadixAttention
# SGLang provides advanced prefix caching and better structured output support

set -e

# Default configuration
MODEL_NAME="${MODEL_NAME:-zai-org/GLM-4.6}"
GPU_COUNT="${GPU_COUNT:-4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
RADIX_CACHE_SIZE="${RADIX_CACHE_SIZE:-10GB}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_feature() {
    echo -e "${BLUE}[FEATURE]${NC} $1"
}

# Function to check GPU availability
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA GPUs required."
        exit 1
    fi

    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    print_info "Detected $AVAILABLE_GPUS GPUs"

    if [ "$AVAILABLE_GPUS" -lt "$GPU_COUNT" ]; then
        print_warn "Requested $GPU_COUNT GPUs but only $AVAILABLE_GPUS available"
        GPU_COUNT=$AVAILABLE_GPUS
    fi
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.10+"
        exit 1
    fi

    if ! python -c "import sglang" 2>/dev/null; then
        print_error "SGLang not installed. Installing..."
        pip install "sglang[all]"
    fi

    print_info "All dependencies satisfied"
}

# Function to configure SGLang features
configure_features() {
    print_info "Configuring SGLang features..."

    # RadixAttention for prefix caching
    ENABLE_RADIX_CACHE=${ENABLE_RADIX_CACHE:-true}
    if [ "$ENABLE_RADIX_CACHE" = true ]; then
        print_feature "✓ RadixAttention enabled (automatic prefix caching)"
        print_feature "  Cache size: $RADIX_CACHE_SIZE"
    fi

    # Torch compile for faster inference
    ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-false}
    if [ "$ENABLE_TORCH_COMPILE" = true ]; then
        print_feature "✓ Torch compile enabled (first run will be slower)"
    fi

    # CUDA graphs for fixed batch sizes
    CUDA_GRAPH_MAX_SEQ_LEN=${CUDA_GRAPH_MAX_SEQ_LEN:-8192}
    print_feature "✓ CUDA graphs enabled (max_seq_len: $CUDA_GRAPH_MAX_SEQ_LEN)"

    # Chunked prefill for long contexts
    ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-true}
    if [ "$ENABLE_CHUNKED_PREFILL" = true ]; then
        print_feature "✓ Chunked prefill enabled (better for long prompts)"
    fi
}

# Function to start SGLang server
start_sglang() {
    print_info "Starting SGLang server..."
    print_info "Model: $MODEL_NAME"
    print_info "Port: $PORT"
    print_info "Tensor Parallel: $GPU_COUNT"
    print_info "Context Length: $CONTEXT_LENGTH"

    # Build command
    CMD="python -m sglang.launch_server"
    CMD="$CMD --model-path $MODEL_NAME"
    CMD="$CMD --host $HOST"
    CMD="$CMD --port $PORT"
    CMD="$CMD --tp $GPU_COUNT"
    CMD="$CMD --context-length $CONTEXT_LENGTH"
    CMD="$CMD --trust-remote-code"

    # Add RadixAttention configuration
    if [ "$ENABLE_RADIX_CACHE" = true ]; then
        CMD="$CMD --enable-radix-cache"
        CMD="$CMD --radix-cache-size $RADIX_CACHE_SIZE"
    fi

    # Add torch compile
    if [ "$ENABLE_TORCH_COMPILE" = true ]; then
        CMD="$CMD --enable-torch-compile"
    fi

    # Add CUDA graphs
    CMD="$CMD --cuda-graph-max-seq-len $CUDA_GRAPH_MAX_SEQ_LEN"

    # Add chunked prefill
    if [ "$ENABLE_CHUNKED_PREFILL" = true ]; then
        CMD="$CMD --chunked-prefill-size 4096"
    fi

    # Add memory fraction
    if [ ! -z "$MEM_FRACTION" ]; then
        CMD="$CMD --mem-fraction-static $MEM_FRACTION"
    fi

    # Set environment variables
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TOKENIZERS_PARALLELISM=false

    print_info "Command: $CMD"
    echo ""

    eval $CMD
}

# Function to test SGLang deployment
test_deployment() {
    print_info "Testing SGLang deployment..."
    sleep 5

    # Test health endpoint
    if curl -s "http://${HOST}:${PORT}/health" > /dev/null; then
        print_info "✓ Server health check passed"
    else
        print_warn "Server health check failed"
    fi

    # Test generation
    print_info "Testing text generation..."
    curl -s "http://${HOST}:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Hello, how are you?",
            "sampling_params": {
                "max_new_tokens": 50,
                "temperature": 0.8
            }
        }' | python -m json.tool

    # Test structured generation (SGLang feature)
    print_info "Testing structured generation..."
    curl -s "http://${HOST}:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Generate a JSON with name and age:",
            "sampling_params": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "regex": "\\{\"name\":\\s*\"[^\"]+\",\\s*\"age\":\\s*\\d+\\}"
            }
        }' | python -m json.tool

    print_info "✓ Deployment tests complete"
}

# Function to show performance tips
show_performance_tips() {
    echo ""
    echo "==============================================="
    echo "    SGLang Performance Tips"
    echo "==============================================="
    echo ""
    echo "1. RadixAttention Prefix Caching:"
    echo "   - Automatically caches common prompts"
    echo "   - Significant speedup for repeated prefixes"
    echo "   - Great for chat applications"
    echo ""
    echo "2. Structured Generation:"
    echo "   - Use 'regex' parameter for JSON/XML output"
    echo "   - Better than post-processing"
    echo "   - Example: regex=\"\\{.*\\}\""
    echo ""
    echo "3. Batch Processing:"
    echo "   - Send multiple requests concurrently"
    echo "   - SGLang automatically batches"
    echo "   - Higher throughput"
    echo ""
    echo "4. Context Optimization:"
    echo "   - Keep prompts under $CONTEXT_LENGTH tokens"
    echo "   - Use chunked prefill for very long contexts"
    echo ""
    echo "==============================================="
}

# Main deployment flow
main() {
    echo "==============================================="
    echo "    GLM-4.6 SGLang Deployment Script"
    echo "==============================================="
    echo ""

    check_gpus
    check_dependencies
    configure_features

    echo ""
    start_sglang
}

# Handle arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
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
        --context-length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --radix-cache-size)
            RADIX_CACHE_SIZE="$2"
            shift 2
            ;;
        --no-radix-cache)
            ENABLE_RADIX_CACHE=false
            shift
            ;;
        --torch-compile)
            ENABLE_TORCH_COMPILE=true
            shift
            ;;
        --mem-fraction)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --test)
            test_deployment
            exit 0
            ;;
        --tips)
            show_performance_tips
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL               Model path (default: zai-org/GLM-4.6)"
            echo "  --gpus N                    Number of GPUs (default: 4)"
            echo "  --port PORT                 Server port (default: 8000)"
            echo "  --host HOST                 Server host (default: 0.0.0.0)"
            echo "  --context-length LEN        Context length (default: 32768)"
            echo "  --radix-cache-size SIZE     RadixAttention cache size (default: 10GB)"
            echo "  --no-radix-cache            Disable RadixAttention caching"
            echo "  --torch-compile             Enable torch.compile (experimental)"
            echo "  --mem-fraction FRAC         GPU memory fraction (0.0-1.0)"
            echo "  --test                      Test existing deployment"
            echo "  --tips                      Show performance optimization tips"
            echo "  --help                      Show this help"
            echo ""
            echo "SGLang Features:"
            echo "  • RadixAttention: Automatic prefix caching"
            echo "  • Structured Generation: Regex-constrained output"
            echo "  • Chunked Prefill: Better long-context handling"
            echo "  • CUDA Graphs: Fixed-size batch optimization"
            echo ""
            echo "Examples:"
            echo "  $0 --gpus 4 --context-length 32768"
            echo "  $0 --gpus 8 --torch-compile"
            echo "  $0 --gpus 2 --mem-fraction 0.85"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

main
