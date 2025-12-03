# GLM-4.6 Pre-Training Methodology

## Table of Contents
1. [Pre-Training Overview](#pre-training-overview)
2. [Hardware Infrastructure](#hardware-infrastructure)
3. [Three-Phase Training Curriculum](#three-phase-training-curriculum)
4. [Training Hyperparameters](#training-hyperparameters)
5. [Expert Management](#expert-management)
6. [Distributed Training Setup](#distributed-training-setup)
7. [Training Step Anatomy](#training-step-anatomy)
8. [Monitoring and Checkpointing](#monitoring-and-checkpointing)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Pre-Training Overview

GLM-4.6 pre-training is a **23 trillion token** process spanning approximately **82-92 days** on **8,192 NVIDIA H800 GPUs**. The training follows a carefully designed three-phase curriculum that progresses from general language understanding to domain-specific expertise and long-context capability.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Training Tokens** | 23 trillion |
| **Training Duration** | 82-92 days |
| **GPU Cluster** | 8,192 × H800 80GB |
| **Total Compute** | 4.6 zettaFLOPs |
| **GPU-Hours** | 753,664 H800-hours |
| **Estimated Cost** | $27.7M (at $2.50/GPU-hour) |
| **Peak Memory** | ~208 GB per GPU |
| **Average Throughput** | ~2.8M tokens/second |

### Training Phases

```
Phase 1: General Pre-Training (15T tokens, 50 days)
  ├─ Context: 2,048 tokens
  ├─ Data: 52% web, 18% multilingual, 20% code, 10% synthetic
  └─ Goal: Foundational language understanding

Phase 2: Domain-Specific Training (7T tokens, 30 days)
  ├─ Context: 4,096 tokens
  ├─ Data: 35% code, 15% math, 12% reasoning, 38% mixed
  └─ Goal: Specialized domain expertise

Phase 3: Long-Context Training (1T tokens, 12 days)
  ├─ Context: 32K → 128K → 200K (gradual extension)
  ├─ Data: Repos, books, documents, agent trajectories
  └─ Goal: Extended context capability
```

---

## Hardware Infrastructure

### GPU Cluster Specification

**Hardware Configuration**:
- **GPUs**: 8,192 × NVIDIA H800 80GB
- **Nodes**: 1,024 nodes × 8 GPUs per node
- **GPU Memory**: 640 GB per node (8 × 80 GB)
- **System RAM**: 1-2 TB DDR5 per node
- **NVMe Storage**: 8 TB per node (for ZeRO-3 offloading)

**Network Topology**:
- **Intra-Node**: NVLink + NVSwitch
  - Bandwidth: 900 GB/s per GPU
  - Latency: <1 μs
- **Inter-Node**: InfiniBand HDR
  - Bandwidth: 400-800 Gbps
  - Latency: ~1-2 μs

### Compute Specifications

**Per-GPU Performance**:
- **FP32 Performance**: 60 TFLOPS
- **BF16 Performance**: 1,979 TFLOPS (with Tensor Cores)
- **Sustained Training**: ~0.9 PFLOPS per GPU
- **Memory Bandwidth**: 3.35 TB/s (H800)

**Cluster-Wide Performance**:
- **Peak Compute**: 16.2 exaFLOPS (theoretical)
- **Sustained Compute**: ~7.4 exaFLOPS (45% utilization)
- **Total Memory**: 640 TB VRAM
- **Total NVMe**: 8 PB

### FLOPs Calculation

For each token with 32B active parameters:

```python
# Forward pass FLOPs
forward_flops = 6 * num_active_params = 6 * 32B = 192 GFLOPS

# Backward pass (typically ~2× forward)
backward_flops = 2 * forward_flops = 384 GFLOPS

# Total per token
total_flops_per_token = forward_flops + backward_flops = 576 GFLOPS

# Add MoE routing overhead (~10%)
with_routing = 576 * 1.1 = 633.6 GFLOPS per token

# For 23T tokens
total_flops = 23e12 * 633.6e9 = 1.46e25 FLOPs = 14.6 yottaFLOPs
```

**Adjusted for actual implementation**:
- Activation checkpointing: +10% FLOPs
- Mixed precision overhead: +5% FLOPs
- Communication overhead: Reduces throughput by ~35%

**Realistic total**: ~4.6 zettaFLOPs

---

## Three-Phase Training Curriculum

### Phase 1: General Pre-Training

**Duration**: 50 days
**Tokens**: 15 trillion
**Context Length**: 2,048 tokens
**Batch Size**: 4M tokens (2,048 sequences)

**Data Mixture**:
```yaml
web_common_crawl: 35%      # 5.25T tokens
books: 15%                 # 2.25T tokens
wikipedia: 8%              # 1.2T tokens
news: 7%                   # 1.05T tokens
academic_papers: 5%        # 0.75T tokens
chinese_web: 12%           # 1.8T tokens
multilingual: 8%           # 1.2T tokens
conversation: 10%          # 1.5T tokens
```

**Objectives**:
1. Learn foundational language patterns
2. Build vocabulary coverage
3. Establish basic reasoning
4. Warm up expert routing

**Expert Configuration**:
- First 3 layers: Dense FFN only
- Layers 4-92: MoE activated
- Initial routing temperature: 1.0
- Gradual expert activation (first 100B tokens)

**Learning Rate Schedule**:
```python
warmup_steps = 2000
max_lr = 0.02  # Muon base LR
min_lr = max_lr * 0.1

# Cosine decay with warmup
if step < warmup_steps:
    lr = max_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Checkpoints**: Every 5,000 steps (~20B tokens)

---

### Phase 2: Domain-Specific Training

**Duration**: 30 days
**Tokens**: 7 trillion
**Context Length**: 4,096 tokens
**Batch Size**: 4M tokens (1,024 sequences)

**Data Mixture**:
```yaml
github_code: 35%           # 2.45T tokens
code_docs: 10%             # 0.7T tokens
math_problems: 15%         # 1.05T tokens
reasoning_traces: 12%      # 0.84T tokens
scientific_papers: 8%      # 0.56T tokens
technical_blogs: 10%       # 0.7T tokens
general_downsampled: 10%   # 0.7T tokens (maintain general knowledge)
```

**Objectives**:
1. Develop code understanding
2. Strengthen mathematical reasoning
3. Learn multi-step problem solving
4. Prepare for instruction following

**Data Quality Requirements**:
- Code: Must compile/execute successfully
- Math: Verified solutions with step-by-step reasoning
- Reasoning: Chain-of-thought format

**Learning Rate**: Continue cosine decay from Phase 1 (no reset)

**Checkpoints**: Every 3,000 steps (~12B tokens)

---

### Phase 3: Long-Context Training

**Duration**: 12 days
**Tokens**: 1 trillion
**Context Length**: Progressive (32K → 64K → 128K → 200K)
**Batch Size**: 2M tokens (adaptive based on context length)

**Context Extension Schedule**:
```yaml
tokens_0_to_250B:   32K context
tokens_250B_to_500B: 64K context
tokens_500B_to_750B: 128K context
tokens_750B_to_1T:   200K context
```

**Data Mixture**:
```yaml
long_documents: 25%        # 250B tokens - books, articles
full_repositories: 30%     # 300B tokens - complete codebases
books: 15%                 # 150B tokens - fiction and non-fiction
legal_documents: 10%       # 100B tokens - contracts, cases
agent_trajectories: 20%    # 200B tokens - multi-step tasks
```

**Objectives**:
1. Extend to 200K context window
2. Maintain coherence over long sequences
3. Learn to use information from distant context
4. Prepare for agentic applications

**Special Techniques**:
- Position interpolation for RoPE
- Gradual context extension (avoid sudden jumps)
- Sliding window attention during training
- Long-range dependency evaluation

**Checkpoints**: Every 1,000 steps (~2B tokens at 200K context)

---

## Training Hyperparameters

### Optimizer Configuration

**Muon Optimizer** (for weight matrices):
```python
optimizer_config = {
    "type": "muon",
    "lr": 0.02,
    "momentum": 0.95,
    "nesterov": True,
    "ns_steps": 5,  # Newton-Schulz iterations
}
```

**AdamW** (for biases, layer norms):
```python
adamw_config = {
    "type": "adamw",
    "lr": 3e-4,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "eps": 1e-8,
}
```

### Batch Size and Gradient Accumulation

```python
# Global batch size: 4M tokens
micro_batch_size = 2      # sequences per GPU
gradient_accumulation = 4
num_gpus = 8192

# Calculation
tokens_per_micro_batch = micro_batch_size * seq_length
tokens_per_gpu_step = tokens_per_micro_batch * gradient_accumulation
global_batch_tokens = tokens_per_gpu_step * num_gpus

# Phase 1: 2,048 context
# = 2 * 2048 * 4 * 8192 = 134M tokens per step (scale down to 4M)

# Adjusted
micro_batch_size = 1
gradient_accumulation = 8
# = 1 * 2048 * 8 * 8192 = ~134M (use sequence packing to reach 4M effective)
```

### Precision and Mixed Precision

```python
precision_config = {
    "model_dtype": "bfloat16",
    "optimizer_dtype": "float32",  # Optimizer states in FP32
    "gradient_dtype": "bfloat16",
    "master_weights": True,  # Keep FP32 master copy
    "loss_scale": "dynamic",
    "initial_loss_scale": 2**16,
    "loss_scale_window": 1000,
}
```

### Gradient Clipping

```python
gradient_config = {
    "clip_grad_norm": 1.0,
    "clip_grad_value": None,  # Not used
}
```

---

## Expert Management

### Loss-Free Expert Balancing

GLM-4.6 uses **loss-free balancing** instead of auxiliary load-balancing losses. This is achieved through **dynamic bias adjustment**:

```python
class ExpertBalancer:
    """
    Manages expert utilization without auxiliary losses

    Key idea: Adjust routing bias based on usage statistics
    """
    def __init__(self, num_experts=160, target_balance=0.05):
        self.num_experts = num_experts
        self.target_balance = target_balance  # ±5% tolerance

        # Tracking
        self.expert_counts = torch.zeros(num_experts)
        self.expert_bias = torch.zeros(num_experts)

        # EMA parameters
        self.alpha = 0.01  # Smoothing factor

    def update(self, router_logits, selected_experts):
        """
        Update expert bias based on utilization

        Called after each training step
        """
        # Count expert activations
        batch_counts = torch.bincount(
            selected_experts.flatten(),
            minlength=self.num_experts
        ).float()

        # Exponential moving average
        self.expert_counts = (
            (1 - self.alpha) * self.expert_counts +
            self.alpha * batch_counts
        )

        # Compute usage ratio
        total_tokens = self.expert_counts.sum()
        ideal_count = total_tokens / self.num_experts
        usage_ratio = self.expert_counts / (ideal_count + 1e-6)

        # Bias adjustment
        # usage_ratio > 1.0 → decrease bias (reduce selection probability)
        # usage_ratio < 1.0 → increase bias (increase selection probability)
        bias_adjustment = 0.001 * (usage_ratio - 1.0)

        # Apply adjustment
        self.expert_bias -= bias_adjustment
        self.expert_bias.clamp_(-5.0, 5.0)  # Prevent extreme values

        return self.expert_bias

    def get_balance_metrics(self):
        """Return expert balance metrics"""
        total = self.expert_counts.sum()
        expected = total / self.num_experts

        return {
            "expert_utilization_std": self.expert_counts.std().item(),
            "expert_utilization_cv": (self.expert_counts.std() / expected).item(),
            "max_usage_ratio": (self.expert_counts.max() / expected).item(),
            "min_usage_ratio": (self.expert_counts.min() / expected).item(),
        }
```

### Expert Warm-Up Strategy

During the first 100B tokens of Phase 1, experts are gradually activated:

```python
def expert_warmup_schedule(step, warmup_steps=50000):
    """
    Gradually increase expert routing strength

    Prevents early expert collapse
    """
    if step < warmup_steps:
        # Linear warm-up of routing temperature
        temperature = 1.0 + 0.5 * (step / warmup_steps)
        # temperature goes from 1.0 → 1.5
    else:
        temperature = 1.5

    return temperature
```

### Expert Capacity and Overflow

```python
def compute_expert_capacity(num_tokens, num_experts, capacity_factor=1.25):
    """
    Compute expert capacity to prevent overflow

    Args:
        num_tokens: Total tokens in batch
        num_experts: Number of experts
        capacity_factor: Buffer factor (typically 1.1-1.5)

    Returns:
        capacity: Max tokens per expert
    """
    # Expected tokens per expert
    expected = num_tokens / num_experts

    # Add buffer for load imbalance
    capacity = int(expected * capacity_factor)

    return capacity
```

---

## Distributed Training Setup

### Parallelism Strategy

GLM-4.6 uses **three-dimensional parallelism**:

```
┌─────────────────────────────────────────────────────────┐
│                8,192 GPU Cluster                         │
│                                                          │
│  Tensor Parallel (TP=8)                                  │
│  ├─ Splits model width across 8 GPUs                    │
│  └─ Intra-node (NVLink)                                  │
│                                                          │
│  Pipeline Parallel (PP=16)                               │
│  ├─ Splits 92 layers across 16 stages (~6 layers/stage) │
│  └─ Inter-node (InfiniBand)                              │
│                                                          │
│  Expert Parallel (EP=32)                                 │
│  ├─ Distributes 160 experts across 32 GPU groups        │
│  └─ Each group handles ~5 experts                        │
│                                                          │
│  Data Parallel (DP)                                      │
│  └─ Remaining dimension for batch parallelism            │
└─────────────────────────────────────────────────────────┘

Total: TP(8) × PP(16) × EP(32) = 4,096 GPUs for model
       Remaining 4,096 GPUs for data parallelism
```

### Tensor Parallelism (TP=8)

**What gets split**:
- Attention Q, K, V, O weight matrices
- FFN up/down projections
- Expert gate/down projections

**Communication pattern**:
- All-reduce after attention output projection
- All-reduce after FFN down projection
- Uses NVLink (high bandwidth, low latency)

```python
# Example: Attention with TP
def forward_with_tp(hidden_states, tp_group):
    # Q, K, V projections (column parallel)
    q = column_parallel_linear(hidden_states, q_weight, tp_group)
    k = column_parallel_linear(hidden_states, k_weight, tp_group)
    v = column_parallel_linear(hidden_states, v_weight, tp_group)

    # Attention computation (local)
    attn_output = attention(q, k, v)

    # Output projection (row parallel - requires all-reduce)
    output = row_parallel_linear(attn_output, o_weight, tp_group)

    return output  # All-reduce handled inside row_parallel_linear
```

### Pipeline Parallelism (PP=16)

**Stage assignment**:
```python
# 92 layers → 16 stages
layers_per_stage = 92 // 16  # = 5.75, so most stages have 6 layers

stage_0:  layers 0-5    (6 layers)
stage_1:  layers 6-11   (6 layers)
...
stage_15: layers 86-91  (6 layers)
```

**Pipeline schedule**: GPipe with micro-batching
```
Time →
Stage 0: [F0][F1][F2][F3][B0][B1][B2][B3]
Stage 1:     [F0][F1][F2][F3][B0][B1][B2][B3]
...
Stage 15:                         [F0][F1][F2][F3][B0][B1][B2][B3]

F = Forward, B = Backward
```

**Bubble overhead**: ~(PP-1) / num_micro_batches
- With 16 stages and 8 micro-batches: ~15/8 = 1.875x overhead
- Actual: ~12% efficiency loss

### Expert Parallelism (EP=32)

**Expert distribution**:
```python
# 160 routed experts → 32 GPU groups
experts_per_group = 160 // 32  # = 5 experts

GPU_group_0:  experts 0-4
GPU_group_1:  experts 5-9
...
GPU_group_31: experts 155-159

# Shared expert: replicated across all groups
```

**Routing with EP**:
```python
def route_to_experts(hidden_states, router_logits, ep_group):
    # Select top-8 experts (per token)
    top_k_experts = torch.topk(router_logits, k=8)

    # Determine which GPU group owns each expert
    group_ids = top_k_experts.indices // 5  # 5 experts per group

    # All-to-all communication
    # Send tokens to owning GPU groups
    expert_inputs = all_to_all(hidden_states, group_ids, ep_group)

    # Local expert computation
    expert_outputs = local_expert_forward(expert_inputs)

    # All-to-all to return results
    outputs = all_to_all(expert_outputs, group_ids, ep_group, reverse=True)

    return outputs
```

### ZeRO-3 Optimization

**Parameter partitioning**:
```python
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,

        # Partition everything
        "partition_optimizer_states": True,
        "partition_gradients": True,
        "partition_parameters": True,

        # Offload to NVMe
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "buffer_count": 4,
            "fast_init": False,
        },

        # Expert-specific
        "expert_parallel_size": 32,
        "moe_expert_offload": True,

        # Memory efficiency
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "sub_group_size": 1e9,
    }
}
```

**Memory savings**:
- Model parameters: 355B × 2 bytes (BF16) = 710 GB
- Without ZeRO: 710 GB per GPU (doesn't fit in 80 GB)
- With ZeRO-3: 710 GB / 8,192 = 87 MB per GPU
- Optimizer states: 120 GB / 8,192 = 15 MB per GPU
- Gradients: 710 GB / 8,192 = 87 MB per GPU

**Total per GPU**: ~87 + 15 + 87 + activations ≈ 200 MB + 40 GB activations = ~40 GB

---

## Training Step Anatomy

### Forward Pass (Single Micro-Batch)

```python
def training_step_forward(model, batch, balancer):
    """
    Complete forward pass for one micro-batch

    Returns:
        loss: Scalar loss value
        router_logits: Expert routing decisions (for balancing)
    """
    # 1. Token embedding
    input_ids = batch["input_ids"]  # (batch_size, seq_len)
    hidden_states = model.embeddings(input_ids)

    # 2. Pass through 92 transformer layers
    all_router_logits = []

    for layer_idx, layer in enumerate(model.layers):
        # Attention block
        hidden_states = layer.attention(hidden_states)

        # FFN block (Dense for first 3, MoE for rest)
        if layer_idx < 3:
            hidden_states = layer.dense_ffn(hidden_states)
        else:
            hidden_states, router_logits = layer.moe(
                hidden_states,
                expert_bias=balancer.expert_bias
            )
            all_router_logits.append(router_logits)

    # 3. Final layer norm
    hidden_states = model.final_norm(hidden_states)

    # 4. Language modeling head
    logits = model.lm_head(hidden_states)

    # 5. Compute loss
    labels = batch["labels"]
    loss = cross_entropy_loss(logits, labels)

    return loss, all_router_logits
```

### Backward Pass and Update

```python
def training_step_backward(loss, model, optimizer, scaler):
    """
    Backward pass with gradient accumulation and mixed precision
    """
    # 1. Scale loss for gradient accumulation
    scaled_loss = loss / gradient_accumulation_steps

    # 2. Backward pass (with automatic mixed precision)
    scaler.scale(scaled_loss).backward()

    # 3. Gradient accumulation check
    if (step + 1) % gradient_accumulation_steps == 0:
        # 4. Unscale gradients
        scaler.unscale_(optimizer)

        # 5. Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # 7. Zero gradients
        optimizer.zero_grad()
```

### Expert Bias Update

```python
def update_expert_bias(balancer, router_logits, selected_experts):
    """
    Update expert bias after optimizer step

    This is done OUTSIDE the gradient graph (no-grad)
    """
    with torch.no_grad():
        # Update utilization statistics
        new_bias = balancer.update(router_logits, selected_experts)

        # Apply new bias to model
        for layer in model.layers[3:]:  # MoE layers only
            layer.moe.expert_bias.data = new_bias
```

### Complete Training Step

```python
def full_training_step(
    model,
    optimizer,
    data_loader,
    balancer,
    scaler,
    step
):
    """
    Complete training step with all components
    """
    # Accumulate gradients over micro-batches
    accumulated_loss = 0.0
    all_router_logits = []
    all_selected_experts = []

    for micro_step in range(gradient_accumulation_steps):
        # Get batch
        batch = next(data_loader)

        # Forward pass
        loss, router_logits = training_step_forward(model, batch, balancer)
        accumulated_loss += loss.item()

        # Track routing decisions
        all_router_logits.append(router_logits)
        all_selected_experts.append(get_selected_experts(router_logits))

        # Backward pass
        training_step_backward(loss, model, optimizer, scaler)

    # Update expert bias (after full step)
    update_expert_bias(balancer, all_router_logits, all_selected_experts)

    # Logging
    avg_loss = accumulated_loss / gradient_accumulation_steps
    balance_metrics = balancer.get_balance_metrics()

    return {
        "loss": avg_loss,
        "expert_balance_std": balance_metrics["expert_utilization_std"],
        "step": step,
    }
```

---

## Monitoring and Checkpointing

### Key Metrics to Track

```python
training_metrics = {
    # Loss metrics
    "train/loss": loss,
    "train/perplexity": torch.exp(loss),

    # Expert metrics
    "experts/utilization_std": expert_std,
    "experts/max_usage_ratio": max_ratio,
    "experts/min_usage_ratio": min_ratio,
    "experts/overflow_count": overflow_count,

    # Optimization metrics
    "optim/learning_rate": current_lr,
    "optim/grad_norm": grad_norm,
    "optim/loss_scale": loss_scale,

    # Performance metrics
    "perf/tokens_per_second": tokens_per_sec,
    "perf/gpu_utilization": gpu_util_avg,
    "perf/memory_allocated": memory_allocated,

    # System metrics
    "system/gpu_temp": gpu_temp_avg,
    "system/power_usage": power_watts,
}
```

### Checkpointing Strategy

```python
checkpoint_config = {
    # Regular checkpoints
    "save_interval": 5000,  # Save every 5K steps

    # Keep only last N checkpoints
    "keep_last_n": 3,

    # Save best checkpoints based on validation
    "save_best": True,
    "metric": "validation/loss",

    # Save format
    "format": "deepspeed",  # Use DeepSpeed format for ZeRO-3

    # What to save
    "save_model": True,
    "save_optimizer": True,
    "save_scheduler": True,
    "save_rng_states": True,
}
```

### Checkpoint Contents

```python
checkpoint = {
    # Model state
    "model_state_dict": model.state_dict(),

    # Optimizer state (partitioned with ZeRO)
    "optimizer_state_dict": optimizer.state_dict(),

    # Scheduler state
    "scheduler_state_dict": scheduler.state_dict(),

    # Training state
    "step": global_step,
    "epoch": epoch,
    "tokens_seen": tokens_seen,

    # Expert balancing state
    "expert_counts": balancer.expert_counts,
    "expert_bias": balancer.expert_bias,

    # RNG states (for reproducibility)
    "rng_state": torch.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state_all(),
    "numpy_rng_state": np.random.get_state(),

    # Config
    "config": model.config,
    "training_args": training_args,
}
```

---

## Troubleshooting Guide

### Common Issues and Solutions

**1. Expert Imbalance (usage std >10%)**
```
Symptoms: Some experts heavily used, others barely activated
Solutions:
  - Increase bias adjustment rate (alpha=0.01 → 0.02)
  - Add temporary auxiliary loss for first 10K steps
  - Check for nan/inf in router logits
  - Verify bias clamping not too restrictive
```

**2. GPU Out of Memory (OOM)**
```
Symptoms: CUDA out of memory errors during training
Solutions:
  - Reduce micro_batch_size (2 → 1)
  - Increase gradient_accumulation (4 → 8)
  - Enable activation checkpointing
  - Verify ZeRO-3 offloading working
  - Check for memory leaks in custom ops
```

**3. Loss Spikes**
```
Symptoms: Sudden large increases in loss
Solutions:
  - Check for nan/inf gradients
  - Reduce learning rate temporarily
  - Increase gradient clipping threshold
  - Verify batch hasn't become corrupted
  - Check for hardware failures
```

**4. Slow Training (tokens/sec too low)**
```
Symptoms: Below 2M tokens/sec on full cluster
Solutions:
  - Check GPU utilization (should be >85%)
  - Profile communication overhead
  - Verify NVLink/InfiniBand working
  - Check for stragglers (slow GPUs)
  - Increase pipeline micro-batches
```

**5. Loss Not Decreasing**
```
Symptoms: Loss plateaus early or doesn't improve
Solutions:
  - Check learning rate (might be too low/high)
  - Verify data quality and diversity
  - Check for data leakage/duplication
  - Verify expert routing working
  - Check model initialization
```

---

## Pretraining Code Reference

See companion implementation files:
- `src/training/pretraining/pretrainer.py`: Main training loop
- `src/training/pretraining/curriculum.py`: Phase management
- `src/training/pretraining/distributed.py`: Parallelism setup
- `src/optimizer/muon.py`: Muon optimizer implementation
- `src/infrastructure/deepspeed_config.json`: DeepSpeed ZeRO-3 config

---

**Next**: See [03_MID_TRAINING.md](03_MID_TRAINING.md) for domain-specific training phases.
