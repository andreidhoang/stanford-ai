# GLM-4.6 Deployment Scripts

Comprehensive deployment scripts for GLM-4.6 in production environments.

## 📋 Available Scripts

### 1. **vLLM Deployment** (`deploy_vllm.sh`)
High-throughput inference server with OpenAI-compatible API.

**Features:**
- Multiple deployment scenarios (high-throughput, low-latency, balanced, memory-constrained)
- Automatic GPU detection and configuration
- Prefix caching for common prompts
- Batch processing optimization

**Usage:**
```bash
# Basic deployment (4 GPUs, balanced)
bash deploy_vllm.sh

# High-throughput with 8 GPUs
bash deploy_vllm.sh --scenario high-throughput --gpus 8

# Low-latency single GPU
bash deploy_vllm.sh --scenario low-latency --gpus 1

# With authentication
bash deploy_vllm.sh --api-key mysecretkey

# Custom port
bash deploy_vllm.sh --port 8080

# Test existing deployment
bash deploy_vllm.sh --test
```

**Deployment Scenarios:**
- `high-throughput`: Max concurrent users, batch processing (512 seqs)
- `low-latency`: Minimal latency, single user (1 seq)
- `balanced`: Good throughput + latency (128 seqs) - **default**
- `memory-constrained`: Quantized, lower memory (64 seqs)

### 2. **SGLang Deployment** (`deploy_sglang.sh`)
Advanced inference with RadixAttention and structured generation.

**Features:**
- RadixAttention for automatic prefix caching
- Structured generation with regex constraints
- Chunked prefill for long contexts
- CUDA graphs optimization

**Usage:**
```bash
# Basic deployment
bash deploy_sglang.sh

# With custom context length
bash deploy_sglang.sh --context-length 32768 --gpus 4

# Disable prefix caching
bash deploy_sglang.sh --no-radix-cache

# Enable torch compile (faster after warmup)
bash deploy_sglang.sh --torch-compile

# Custom cache size
bash deploy_sglang.sh --radix-cache-size 20GB

# Show performance tips
bash deploy_sglang.sh --tips

# Test deployment
bash deploy_sglang.sh --test
```

**SGLang Advantages:**
- 2-5× speedup on chat workloads (prefix caching)
- Structured output (JSON, XML) with regex
- Better long-context handling
- Automatic batching

### 3. **Model Quantization** (`quantize_model.py`)
Reduce model size and increase inference speed.

**Supported Formats:**
- **AWQ**: 4-bit, best quality-size tradeoff (recommended for vLLM)
- **GPTQ**: 4-bit, general purpose
- **FP8**: 8-bit float, H100 only, best quality
- **GGUF**: For llama.cpp CPU/GPU inference

**Usage:**
```bash
# Compare quantization methods
python quantize_model.py --compare

# AWQ quantization (recommended)
python quantize_model.py \
    --model zai-org/GLM-4.6 \
    --method awq \
    --output ./quantized-awq \
    --bits 4

# GPTQ quantization
python quantize_model.py \
    --model zai-org/GLM-4.6 \
    --method gptq \
    --output ./quantized-gptq

# FP8 quantization (H100)
python quantize_model.py \
    --model zai-org/GLM-4.6 \
    --method fp8 \
    --output ./quantized-fp8

# GGUF conversion
python quantize_model.py \
    --model zai-org/GLM-4.6 \
    --method gguf \
    --output ./quantized-gguf
```

**Deploy Quantized Model:**
```bash
# vLLM with AWQ
python -m vllm.entrypoints.openai.api_server \
    --model ./quantized-awq \
    --quantization awq \
    --tensor-parallel-size 4

# vLLM with FP8 (H100)
python -m vllm.entrypoints.openai.api_server \
    --model ./quantized-fp8 \
    --quantization fp8 \
    --tensor-parallel-size 4
```

### 4. **Docker Deployment**

**Build Image:**
```bash
# Build vLLM image
docker build -f Dockerfile.vllm -t glm46-vllm:latest .
```

**Run Container:**
```bash
# Basic deployment
docker run --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    glm46-vllm:latest

# With environment variables
docker run --gpus all \
    -p 8000:8000 \
    -e MODEL_NAME=zai-org/GLM-4.6 \
    -e TENSOR_PARALLEL_SIZE=4 \
    -e GPU_MEMORY_UTILIZATION=0.95 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    glm46-vllm:latest

# With HuggingFace token (for gated models)
docker run --gpus all \
    -p 8000:8000 \
    -e HUGGING_FACE_HUB_TOKEN=your_token_here \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    glm46-vllm:latest
```

**Docker Compose:**
```bash
# Start service
docker-compose up -d glm46-vllm

# With monitoring
docker-compose --profile monitoring up -d

# With production setup (nginx)
docker-compose --profile production up -d

# View logs
docker-compose logs -f glm46-vllm

# Stop service
docker-compose down
```

## 📊 Performance Comparison

| Deployment | GPUs | Throughput | Latency | Memory | Best For |
|------------|------|------------|---------|--------|----------|
| **vLLM High-Throughput** | 8 | ~500 tok/s/user | ~100ms | 640GB | Production API |
| **vLLM Balanced** | 4 | ~300 tok/s/user | ~80ms | 320GB | General use |
| **vLLM Low-Latency** | 1 | ~50 tok/s/user | ~50ms | 80GB | Interactive |
| **SGLang** | 4 | ~400 tok/s/user | ~70ms | 320GB | Chat/structured |
| **AWQ Quantized** | 2 | ~200 tok/s/user | ~90ms | 80GB | Cost-effective |

## 🔧 Configuration

### Environment Variables

**Common:**
```bash
MODEL_NAME=zai-org/GLM-4.6          # Model to deploy
GPU_COUNT=4                          # Number of GPUs
PORT=8000                            # API port
HOST=0.0.0.0                         # Server host
```

**vLLM-specific:**
```bash
TENSOR_PARALLEL_SIZE=4               # TP degree
GPU_MEMORY_UTILIZATION=0.95          # GPU memory fraction
MAX_NUM_SEQS=256                     # Max concurrent sequences
QUANTIZATION=awq                     # Quantization method
```

**SGLang-specific:**
```bash
CONTEXT_LENGTH=32768                 # Context window
RADIX_CACHE_SIZE=10GB               # Prefix cache size
ENABLE_TORCH_COMPILE=false          # PyTorch compilation
```

### Port Configuration

- **8000**: Main API server (vLLM/SGLang)
- **9090**: Prometheus metrics
- **3000**: Grafana dashboard (if using monitoring)
- **80/443**: Nginx reverse proxy (if using production profile)

## 📈 Monitoring

### Prometheus Metrics

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'glm46'
    static_configs:
      - targets: ['glm46-vllm:9090']
```

### Health Checks

```bash
# Check server health
curl http://localhost:8000/health

# Check model info
curl http://localhost:8000/v1/models

# Get metrics
curl http://localhost:9090/metrics
```

### Load Testing

```bash
# Install vegeta
go install github.com/tsenart/vegeta@latest

# Create request file (requests.txt):
cat > requests.txt << EOF
POST http://localhost:8000/v1/completions
Content-Type: application/json
@request.json
EOF

# Create request body (request.json):
cat > request.json << EOF
{
  "model": "zai-org/GLM-4.6",
  "prompt": "Hello, how are you?",
  "max_tokens": 50
}
EOF

# Run load test (100 requests/sec for 30s)
vegeta attack -rate=100 -duration=30s -targets=requests.txt | vegeta report
```

## 🚀 Production Deployment Checklist

- [ ] Choose deployment method (vLLM recommended for production)
- [ ] Select hardware (minimum 4× A100/H100 80GB)
- [ ] Configure environment variables
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure reverse proxy (Nginx)
- [ ] Enable SSL/TLS certificates
- [ ] Set API authentication (--api-key)
- [ ] Configure rate limiting
- [ ] Set up logging
- [ ] Test failover and recovery
- [ ] Document API endpoints
- [ ] Load test under expected traffic

## 📞 API Usage

### OpenAI-Compatible API

```python
from openai import OpenAI

# Connect to GLM-4.6 server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # If authentication enabled
)

# Chat completion
response = client.chat.completions.create(
    model="zai-org/GLM-4.6",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=500,
    temperature=0.8
)

print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="zai-org/GLM-4.6",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

### Direct HTTP API

```bash
# Completion
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/GLM-4.6",
        "prompt": "Hello, how are you?",
        "max_tokens": 100,
        "temperature": 0.8
    }'

# Chat
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/GLM-4.6",
        "messages": [
            {"role": "user", "content": "What is AI?"}
        ],
        "max_tokens": 200
    }'
```

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `GPU_MEMORY_UTILIZATION` (e.g., 0.85)
- Decrease `MAX_NUM_SEQS`
- Use quantization (AWQ/GPTQ)
- Increase `TENSOR_PARALLEL_SIZE`

**"Model not found"**
- Set `HUGGING_FACE_HUB_TOKEN` environment variable
- Check model path is correct
- Verify internet connection for download

**"Slow inference"**
- Enable prefix caching (`--enable-prefix-caching`)
- Use more GPUs (increase tensor parallelism)
- Try SGLang for chat workloads
- Consider FP8 quantization on H100

**"Connection refused"**
- Wait for model loading (can take 2-5 minutes)
- Check port is not in use: `lsof -i :8000`
- Verify firewall settings
- Check Docker port mapping

## 📚 Additional Resources

- vLLM Documentation: https://docs.vllm.ai
- SGLang Documentation: https://github.com/sgl-project/sglang
- GLM-4.6 Official: https://z.ai/blog/glm-4.6
- HuggingFace Model: https://huggingface.co/zai-org/GLM-4.6

## 🤝 Support

For deployment issues:
1. Check logs: `docker logs glm46-vllm` or console output
2. Verify GPU availability: `nvidia-smi`
3. Test health endpoint: `curl http://localhost:8000/health`
4. Review configuration in this README
