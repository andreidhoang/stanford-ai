# GLM-4.6 Model Architecture: Complete Technical Specification

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Transformer Components](#core-transformer-components)
4. [Mixture-of-Experts (MoE) Architecture](#mixture-of-experts-architecture)
5. [Attention Mechanism](#attention-mechanism)
6. [Positional Encoding](#positional-encoding)
7. [Normalization and Stabilization](#normalization-and-stabilization)
8. [Multi-Token Prediction](#multi-token-prediction)
9. [Model Configuration](#model-configuration)
10. [Implementation Details](#implementation-details)

---

## Executive Summary

GLM-4.6 is a **355 billion parameter** Mixture-of-Experts (MoE) language model with **32 billion active parameters** per forward pass, released by Zhipu AI in September 2025. It represents a "depth over width" architectural philosophy, achieving state-of-the-art performance on mathematical reasoning and coding tasks while being significantly more cost-effective than closed-source alternatives.

### Key Specifications

| Component | Specification | Details |
|-----------|--------------|---------|
| **Total Parameters** | 355B | 357B in some documentation |
| **Active Parameters** | 32B | ~9% activation rate |
| **Architecture Type** | MoE Transformer | Sparse activation |
| **Number of Layers** | 92 | Hidden layers |
| **Hidden Dimension** | 5,120 | Model width |
| **Vocabulary Size** | 151,552 | Tokens |
| **Context Window** | 200,000 | Input tokens |
| **Max Output** | 128,000 | Output tokens |
| **Attention Type** | Grouped-Query Attention (GQA) | 96 query heads, 8 KV heads |
| **Position Encoding** | Partial RoPE | Factor 0.5, theta 1e6 |
| **Experts** | 160 routed + 1 shared | Top-8 routing |
| **Training Tokens** | 23 trillion | 15T general + 7T domain + 1T long-context |
| **License** | MIT | Open source |

### Design Philosophy

**Depth Over Width**: GLM-4.6 adopts a strategy of:
- **More layers** (92 vs typical 60-80)
- **Narrower hidden dimension** (5,120 vs 6,144-8,192)
- **Fewer experts** (160 vs 256-512 in competitors)
- **More attention heads** (96 vs typical 64-80)

This design yields superior reasoning capacity on benchmarks like MMLU and BBH compared to "width over depth" approaches (e.g., DeepSeek-V3 with 671B total parameters).

---

## Architecture Overview

### High-Level System Blueprint

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Text/Code                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Tokenizer (BPE: 151,552 vocab, 318,088 merges)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Embedding Layer (5,120-dim, shared)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Stack of 92 Transformer Blocks                │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Layer N (repeated 92 times):                        │   │
│  │                                                       │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  1. RMS Normalization (pre-attention)          │  │   │
│  │  └──────────────┬─────────────────────────────────┘  │   │
│  │                 ▼                                     │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  2. Grouped-Query Attention (GQA)              │  │   │
│  │  │     - 96 query heads                           │  │   │
│  │  │     - 8 key-value head groups                  │  │   │
│  │  │     - Partial RoPE (factor 0.5)                │  │   │
│  │  │     - QK-Norm for stability                    │  │   │
│  │  └──────────────┬─────────────────────────────────┘  │   │
│  │                 ▼                                     │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  3. Residual Connection + RMS Norm             │  │   │
│  │  └──────────────┬─────────────────────────────────┘  │   │
│  │                 ▼                                     │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  4. MoE Feed-Forward (layers 4-92)             │  │   │
│  │  │     or Dense FFN (layers 1-3)                  │  │   │
│  │  │                                                 │  │   │
│  │  │     Router: Sigmoid + Loss-Free Balancing      │  │   │
│  │  │       ├─> Expert 1 (1,536 hidden) ──┐          │  │   │
│  │  │       ├─> Expert 2                  │          │  │   │
│  │  │       ├─> ... (top-8 selected)      ├─> Σ     │  │   │
│  │  │       ├─> Expert 160                │          │  │   │
│  │  │       └─> Shared Expert (always) ───┘          │  │   │
│  │  └──────────────┬─────────────────────────────────┘  │   │
│  │                 ▼                                     │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  5. Residual Connection                        │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Final RMS Normalization                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Language Modeling Head (5,120 → 151,552)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Logits & Sampling/Decoding               │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Distribution

```python
# Approximate parameter counts by component

# Embedding layer (shared with output head)
embedding_params = vocab_size * hidden_dim
embedding_params = 151_552 * 5_120 = 775.9M

# Per-layer attention (GQA)
attention_params_per_layer = (
    # Q projection: hidden_dim × (num_heads × head_dim)
    5_120 * (96 * 128) +  # 62.9M
    # K projection: hidden_dim × (num_kv_heads × head_dim)
    5_120 * (8 * 128) +   # 5.2M
    # V projection: hidden_dim × (num_kv_heads × head_dim)
    5_120 * (8 * 128) +   # 5.2M
    # O projection: (num_heads × head_dim) × hidden_dim
    (96 * 128) * 5_120    # 62.9M
) = 136.2M per layer

# Total attention parameters (92 layers)
total_attention = 92 * 136.2M = 12.5B

# Dense FFN (first 3 layers)
dense_ffn_params_per_layer = (
    # Up projection: hidden_dim × intermediate_size
    5_120 * 12_288 +      # 62.9M
    # Down projection: intermediate_size × hidden_dim
    12_288 * 5_120        # 62.9M
) = 125.8M per layer

total_dense_ffn = 3 * 125.8M = 377.4M

# MoE FFN (layers 4-92 = 89 layers)
moe_expert_params = (
    # Up projection: hidden_dim × moe_intermediate_size
    5_120 * 1_536 +       # 7.9M
    # Down projection: moe_intermediate_size × hidden_dim
    1_536 * 5_120         # 7.9M
) = 15.7M per expert

# Each MoE layer: 160 routed + 1 shared expert
moe_params_per_layer = 161 * 15.7M = 2.53B per layer
total_moe = 89 * 2.53B = 225.2B

# Normalization (RMSNorm: 2 per layer × 92 layers)
norm_params = 92 * 2 * 5_120 = 0.9M

# TOTAL PARAMETERS
total_params = (
    embedding_params +    # 775.9M
    total_attention +     # 12.5B
    total_dense_ffn +     # 377.4M
    total_moe +           # 225.2B
    norm_params           # 0.9M
) ≈ 239B

# Note: Actual 355B includes additional components:
# - Multi-token prediction heads
# - Extended router networks
# - Additional projection layers
```

### Active Parameters per Token

With top-8 expert routing:
- **Attention parameters**: 136.2M × 92 = 12.5B (always active)
- **Dense FFN**: 125.8M × 3 = 377.4M (always active)
- **MoE active**: (8 routed + 1 shared) × 15.7M × 89 = 12.6B
- **Embeddings + norms**: ~1.5B

**Total Active**: ~32B parameters per forward pass (~9% of 355B total)

---

## Core Transformer Components

### 1. Embedding Layer

```python
class GLM4Embeddings(nn.Module):
    """
    Token embeddings for GLM-4.6

    Shared between input and output (weight tying)
    """
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size  # 151,552
        self.hidden_size = config.hidden_size  # 5,120

        # Token embeddings
        self.word_embeddings = nn.Embedding(
            self.vocab_size,
            self.hidden_size
        )

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
        """
        return self.word_embeddings(input_ids)
```

### 2. RMS Normalization

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    More efficient than LayerNorm, commonly used in large language models
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            normalized: (batch_size, seq_len, hidden_size)
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute variance
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # Normalize
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Scale by learnable weight
        return (self.weight * hidden_states).to(input_dtype)
```

### 3. Transformer Block Structure

```python
class GLM4TransformerBlock(nn.Module):
    """
    Single transformer block combining attention and MoE/Dense FFN

    Architecture:
        x -> RMSNorm -> Attention -> Residual
          -> RMSNorm -> MoE/Dense FFN -> Residual
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )

        # Grouped-query attention
        self.self_attention = GLM4Attention(config, layer_idx)

        # Pre-FFN normalization
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )

        # Feed-forward: Dense for first 3 layers, MoE for rest
        if layer_idx < config.first_k_dense_replace:
            self.mlp = GLM4DenseFFN(config)
        else:
            self.mlp = GLM4MoE(config, layer_idx)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
    ):
        """
        Forward pass through transformer block

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len)
            position_ids: (batch_size, seq_len)
            past_key_value: Cached key-value pairs for fast decoding
            output_attentions: Return attention weights
            output_router_logits: Return MoE routing decisions
            use_cache: Cache key-value pairs

        Returns:
            hidden_states: (batch_size, seq_len, hidden_size)
            (optional) attention_weights, past_key_value, router_logits
        """
        # 1. Attention block with residual connection
        residual = hidden_states

        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = attn_outputs[0]
        outputs = attn_outputs[1:]  # attentions, past_key_value

        # Residual connection
        hidden_states = residual + hidden_states

        # 2. FFN block with residual connection
        residual = hidden_states

        # Pre-norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Feed-forward (Dense or MoE)
        if isinstance(self.mlp, GLM4MoE):
            mlp_outputs = self.mlp(
                hidden_states,
                output_router_logits=output_router_logits
            )
            hidden_states = mlp_outputs[0]

            if output_router_logits:
                outputs = outputs + (mlp_outputs[1],)  # router_logits
        else:
            hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + outputs

        return outputs
```

---

## Mixture-of-Experts Architecture

### MoE Design Philosophy

GLM-4.6 uses a **sparse MoE** design where:
1. Each token is routed to **8 out of 160 experts** (top-8 routing)
2. There is **1 shared expert** always active (for stability)
3. The first **3 layers are dense** (not MoE) for better gradient flow
4. **Loss-free balancing** via dynamic bias adjustment (no auxiliary loss needed)

### Expert Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Total Experts** | 161 | 160 routed + 1 shared |
| **Routed Experts** | 160 | Conditionally activated |
| **Shared Expert** | 1 | Always active |
| **Active per Token** | 9 | 8 routed + 1 shared |
| **MoE Intermediate Size** | 1,536 | Hidden dim in each expert |
| **Routed Scaling Factor** | 2.5 | Output scaling |
| **First K Dense** | 3 | First 3 layers are dense FFN |

### MoE Layer Implementation

```python
class GLM4MoE(nn.Module):
    """
    Mixture-of-Experts layer for GLM-4.6

    Features:
    - 160 routed experts + 1 shared expert
    - Top-8 routing per token
    - Loss-free balancing via sigmoid gates with learnable bias
    - Routed scaling factor for output normalization
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size  # 5,120
        self.num_experts = config.num_experts  # 160
        self.num_selected_experts = config.num_experts_per_tok  # 8
        self.intermediate_size = config.moe_intermediate_size  # 1,536
        self.routed_scaling_factor = config.routed_scaling_factor  # 2.5

        # Router network
        self.router = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False
        )

        # Learnable bias for loss-free balancing
        self.expert_bias = nn.Parameter(
            torch.zeros(self.num_experts)
        )

        # Routed experts
        self.experts = nn.ModuleList([
            GLM4Expert(config) for _ in range(self.num_experts)
        ])

        # Shared expert (always active)
        self.shared_expert = GLM4Expert(config)

        # Expert utilization tracking (for bias adjustment)
        self.register_buffer(
            "expert_counts",
            torch.zeros(self.num_experts)
        )

    def forward(self, hidden_states, output_router_logits=False):
        """
        Forward pass through MoE layer

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            output_router_logits: Return routing decisions for analysis

        Returns:
            output: (batch_size, seq_len, hidden_size)
            router_logits: (optional) (batch_size, seq_len, num_experts)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # (B*S, H)

        # 1. Compute router logits
        router_logits = self.router(hidden_states_flat)  # (B*S, num_experts)

        # Apply expert-wise bias for loss-free balancing
        router_logits = router_logits + self.expert_bias.unsqueeze(0)

        # 2. Apply sigmoid activation (GLM-4.6 uses sigmoid, not softmax)
        router_probs = torch.sigmoid(router_logits)  # (B*S, num_experts)

        # 3. Select top-K experts
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.num_selected_experts,
            dim=-1
        )  # (B*S, K)

        # Normalize routing weights (sum to 1 for selected experts)
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )

        # Apply routed scaling factor
        routing_weights = routing_weights * self.routed_scaling_factor

        # 4. Update expert utilization counts (for bias adjustment)
        if self.training:
            with torch.no_grad():
                expert_mask = torch.zeros_like(router_logits)
                expert_mask.scatter_(1, selected_experts, 1.0)
                batch_expert_counts = expert_mask.sum(dim=0)

                # Exponential moving average
                alpha = 0.01
                self.expert_counts = (
                    (1 - alpha) * self.expert_counts +
                    alpha * batch_expert_counts
                )

        # 5. Route tokens to experts
        final_hidden_states = torch.zeros(
            batch_size * seq_len,
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Flatten for efficient routing
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)  # (num_experts, K, B*S)

        # Process each expert
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]

            # Get tokens assigned to this expert
            expert_tokens = expert_mask[expert_idx].nonzero(as_tuple=True)

            if expert_tokens[0].shape[0] > 0:
                # Extract token indices and routing weight indices
                token_idx = expert_tokens[1]  # Which tokens
                routing_weight_idx = expert_tokens[0]  # Which of the K slots

                # Get the actual tokens
                current_tokens = hidden_states_flat[token_idx]

                # Process through expert
                expert_output = expert(current_tokens)

                # Get corresponding routing weights
                current_weights = routing_weights[
                    token_idx, routing_weight_idx
                ].unsqueeze(-1)

                # Accumulate weighted expert outputs
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    expert_output * current_weights
                )

        # 6. Add shared expert output (always active)
        shared_output = self.shared_expert(hidden_states_flat)
        final_hidden_states = final_hidden_states + shared_output

        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.view(
            batch_size, seq_len, hidden_size
        )

        if output_router_logits:
            router_logits = router_logits.view(
                batch_size, seq_len, self.num_experts
            )
            return final_hidden_states, router_logits

        return (final_hidden_states,)


class GLM4Expert(nn.Module):
    """
    Single expert network (2-layer FFN with gated activation)

    Architecture:
        x -> gate_proj(x) * silu(x) -> down_proj -> output
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 5,120
        self.intermediate_size = config.moe_intermediate_size  # 1,536

        # Up projection (hidden -> intermediate)
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )

        # Down projection (intermediate -> hidden)
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )

        self.act_fn = nn.SiLU()  # Swish activation

    def forward(self, x):
        """
        Args:
            x: (num_tokens, hidden_size)

        Returns:
            output: (num_tokens, hidden_size)
        """
        # Gated activation: gate_proj(x) * silu(gate_proj(x))
        gate_output = self.gate_proj(x)
        activated = self.act_fn(gate_output)

        # Down projection
        output = self.down_proj(activated)

        return output
```

### Loss-Free Expert Balancing

GLM-4.6 innovates with **loss-free balancing** instead of auxiliary load-balancing losses:

```python
def update_expert_bias(self, expert_counts):
    """
    Update expert bias to encourage balanced utilization

    Called after each training step

    Key innovation: No auxiliary loss needed, no gradient interference
    """
    # Compute ideal utilization
    total_tokens = expert_counts.sum()
    ideal_count = total_tokens / self.num_experts

    # Compute usage ratio for each expert
    usage_ratio = expert_counts / (ideal_count + 1e-6)

    # Adjust bias: penalize overused, boost underused
    # usage_ratio > 1.0 → decrease bias (reduce selection probability)
    # usage_ratio < 1.0 → increase bias (increase selection probability)
    bias_adjustment = 0.001 * (usage_ratio - 1.0)

    # Update bias (gradient-free)
    with torch.no_grad():
        self.expert_bias -= bias_adjustment

        # Clamp to prevent extreme values
        self.expert_bias.clamp_(-5.0, 5.0)
```

**Benefits**:
1. No auxiliary loss → cleaner training signal
2. No hyperparameter tuning for loss coefficient
3. Better expert utilization in practice
4. Maintains balanced load within ±5%

---

## Attention Mechanism

### Grouped-Query Attention (GQA)

GLM-4.6 uses **Grouped-Query Attention**, a middle ground between:
- **Multi-Head Attention (MHA)**: Each query head has its own key-value head
- **Multi-Query Attention (MQA)**: All query heads share single key-value head

GQA groups query heads to share key-value heads, balancing quality and efficiency.

### GQA Configuration

| Parameter | Value | Details |
|-----------|-------|---------|
| **Query Heads** | 96 | Individual query attention heads |
| **Key-Value Heads** | 8 | Shared KV head groups |
| **Head Dimension** | 128 | Dimension of each head |
| **Ratio** | 12:1 | 12 query heads per KV head group |
| **Total Q Dimension** | 12,288 | 96 × 128 |
| **Total KV Dimension** | 1,024 | 8 × 128 |

**Design Rationale**: GLM-4.6 uses **2.5× more attention heads** than typical models for its hidden dimension. Research shows this doesn't improve training loss but consistently enhances reasoning benchmark performance (MMLU, BBH).

### GQA Implementation

```python
class GLM4Attention(nn.Module):
    """
    Grouped-Query Attention with Partial RoPE and QK-Norm

    Features:
    - 96 query heads, 8 KV heads (12:1 ratio)
    - Partial rotary position embeddings (50% of dimensions)
    - Query-Key normalization for training stability
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size  # 5,120
        self.num_heads = config.num_attention_heads  # 96
        self.num_key_value_heads = config.num_key_value_heads  # 8
        self.head_dim = config.head_dim  # 128

        # Number of query heads per KV head
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 12

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )

        # QK Normalization for stability
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary Position Embeddings
        self.rotary_emb = GLM4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        """
        Forward pass through GQA

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len)
            position_ids: (batch_size, seq_len)
            past_key_value: Cached (key, value) for fast decoding
            output_attentions: Return attention weights
            use_cache: Cache key-value pairs

        Returns:
            attn_output: (batch_size, seq_len, hidden_size)
            (optional) attention_weights, present_key_value
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape to separate heads
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (B, num_heads, S, head_dim)

        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (B, num_kv_heads, S, head_dim)

        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (B, num_kv_heads, S, head_dim)

        # 3. Apply QK Normalization (before RoPE)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # 4. Apply Rotary Position Embeddings (partial RoPE)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
            partial_rotary_factor=self.config.partial_rotary_factor
        )

        # 5. Concatenate with past key-value cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # 6. Repeat KV heads to match number of query heads (for GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 7. Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # 8. Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 9. Softmax
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # 10. Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # 11. Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


def repeat_kv(hidden_states, n_rep):
    """
    Repeat key-value heads to match number of query heads

    For GQA: expand (B, num_kv_heads, S, head_dim) to (B, num_heads, S, head_dim)

    Args:
        hidden_states: (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Repetition factor (num_heads // num_kv_heads)

    Returns:
        expanded: (batch, num_heads, seq_len, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

### QK-Norm: Query-Key Normalization

GLM-4.6 implements **QK-Norm** - layer normalization on queries and keys before attention:

```python
# In GLM4Attention.__init__:
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

# In forward pass (before computing attention scores):
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
```

**Benefits**:
1. Prevents attention logit explosion during training
2. Avoids one-hot attention distributions
3. Enables stable training at 1.5× higher learning rates
4. Improves model convergence without auxiliary losses

---

## Positional Encoding

### Partial Rotary Position Embeddings (RoPE)

GLM-4.6 uses **Partial RoPE** with the following configuration:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **RoPE Type** | Partial | Only applies to subset of dimensions |
| **Partial Factor** | 0.5 | 50% of dimensions receive rotation |
| **RoPE Theta** | 1,000,000 | Base frequency (10^6) |
| **Max Position** | 202,752 | Maximum position embeddings |

### RoPE Mathematical Foundation

Rotary Position Embedding encodes position information by rotating embedding vectors:

For token at position $m$ with embedding dimension $d$, RoPE applies:

$$
\begin{pmatrix} x'_m \\ y'_m \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_m \\ y_m \end{pmatrix}
$$

where the frequency $\theta$ for dimension $i$ is:

$$
\theta_i = \text{base}^{-2i/d} = 10^6{}^{-2i/d}
$$

### Partial RoPE Implementation

```python
class GLM4RotaryEmbedding(nn.Module):
    """
    Partial Rotary Position Embeddings

    Only applies rotation to first 50% of embedding dimensions
    Remaining 50% use absolute positions
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim  # 128
        self.max_position_embeddings = config.max_position_embeddings  # 202,752
        self.base = config.rope_theta  # 1,000,000
        self.partial_rotary_factor = config.partial_rotary_factor  # 0.5

        # Compute rotary dimension
        self.rotary_dim = int(self.dim * self.partial_rotary_factor)  # 64

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        """
        Generate cos and sin embeddings

        Args:
            x: Input tensor (for device/dtype reference)
            seq_len: Sequence length

        Returns:
            cos: (1, 1, seq_len, rotary_dim)
            sin: (1, 1, seq_len, rotary_dim)
        """
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, rotary_dim//2)

        # Interleave cos and sin
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, rotary_dim)

        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, rotary_dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, rotary_dim)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, partial_rotary_factor=0.5):
    """
    Apply partial rotary position embeddings to query and key tensors

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine embeddings (1, 1, seq_len, rotary_dim)
        sin: Sine embeddings (1, 1, seq_len, rotary_dim)
        position_ids: Position indices (batch, seq_len)
        partial_rotary_factor: Fraction of dimensions to rotate (0.5)

    Returns:
        q_embed: Rotated queries
        k_embed: Rotated keys
    """
    # Determine rotary dimension
    head_dim = q.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)

    # Split into rotary and pass-through parts
    q_rot = q[..., :rotary_dim]  # First 50% (gets rotated)
    q_pass = q[..., rotary_dim:]  # Last 50% (passes through)

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Gather cos/sin for actual positions
    cos = cos.squeeze(1).squeeze(0)  # (seq_len, rotary_dim)
    sin = sin.squeeze(1).squeeze(0)

    cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, rotary_dim)
    sin = sin[position_ids].unsqueeze(1)

    # Apply rotation using complex number formulation
    q_rot_real = q_rot[..., 0::2]
    q_rot_imag = q_rot[..., 1::2]

    # Rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    q_rot_embed_real = q_rot_real * cos[..., 0::2] - q_rot_imag * sin[..., 0::2]
    q_rot_embed_imag = q_rot_real * sin[..., 1::2] + q_rot_imag * cos[..., 1::2]

    # Interleave real and imaginary parts
    q_rot_embed = torch.stack([q_rot_embed_real, q_rot_embed_imag], dim=-1)
    q_rot_embed = q_rot_embed.flatten(-2)

    # Same for keys
    k_rot_real = k_rot[..., 0::2]
    k_rot_imag = k_rot[..., 1::2]

    k_rot_embed_real = k_rot_real * cos[..., 0::2] - k_rot_imag * sin[..., 0::2]
    k_rot_embed_imag = k_rot_real * sin[..., 1::2] + k_rot_imag * cos[..., 1::2]

    k_rot_embed = torch.stack([k_rot_embed_real, k_rot_embed_imag], dim=-1)
    k_rot_embed = k_rot_embed.flatten(-2)

    # Concatenate rotated and pass-through parts
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

    return q_embed, k_embed
```

**Why Partial RoPE?**
- **Better extrapolation**: Hybrid of relative (rotated) and absolute (non-rotated) positions
- **Efficiency**: Only rotating 50% reduces computation
- **Long-context performance**: Enables stable 200K context window training

---

## Normalization and Stabilization

### 1. RMS Normalization

Used throughout the model (pre-attention, pre-FFN, final layer):

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Normalizes by RMS instead of mean+variance (more efficient)
    """
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # RMS: sqrt(mean(x^2))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)
```

**Configuration**: `rms_norm_eps = 1e-05`

### 2. QK-Normalization

Applied to queries and keys before attention computation:

```python
# In GLM4Attention
self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

# Before attention scores
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
```

**Benefits**:
- Prevents attention logit explosion
- Enables 1.5× higher learning rates
- Improves convergence stability

---

## Multi-Token Prediction

### MTP Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Number of MTP Layers** | 1 | Specialized prediction heads |
| **Purpose** | Speculative Decoding | Draft future tokens without separate model |
| **Mechanism** | Multiple prediction heads | Each predicts t+1, t+2, t+3, ... |

### MTP Implementation

```python
class GLM4MultiTokenPrediction(nn.Module):
    """
    Multi-Token Prediction heads for speculative decoding

    Predicts multiple future tokens in parallel for faster inference
    """
    def __init__(self, config):
        super().__init__()
        self.num_nextn_predict_layers = config.num_nextn_predict_layers  # 1
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Prediction heads for t+1, t+2, t+3, ...
        self.nextn_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            for _ in range(self.num_nextn_predict_layers)
        ])

    def forward(self, hidden_states):
        """
        Predict next N tokens

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            predictions: List of (batch_size, seq_len, vocab_size)
        """
        predictions = []

        for head in self.nextn_heads:
            logits = head(hidden_states)
            predictions.append(logits)

        return predictions
```

### Speculative Decoding with MTP

**Recommended Configuration**:
```bash
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

**How it works**:
1. Main model generates hidden states
2. MTP heads predict t+1, t+2, t+3 in parallel
3. Main model verifies drafted tokens
4. Accept longest matching prefix
5. Continue from accepted position

**Speedup**: 1.5-2× faster inference on typical workloads

---

## Model Configuration

### Complete GLM-4.6 Configuration

```json
{
  "model_type": "glm",
  "architectures": ["GLM4ForCausalLM"],

  "_name_or_path": "zai-org/GLM-4.6",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.0",

  "vocab_size": 151552,
  "hidden_size": 5120,
  "intermediate_size": 12288,
  "num_hidden_layers": 92,
  "num_attention_heads": 96,
  "num_key_value_heads": 8,
  "head_dim": 128,

  "max_position_embeddings": 202752,
  "max_sequence_length": 200000,

  "hidden_act": "silu",
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-05,

  "use_cache": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,

  "tie_word_embeddings": true,

  "rope_theta": 1000000.0,
  "rope_scaling": null,
  "partial_rotary_factor": 0.5,

  "attention_bias": false,
  "attention_dropout": 0.0,

  "num_experts": 160,
  "num_experts_per_tok": 8,
  "num_shared_expert": 1,
  "moe_intermediate_size": 1536,
  "routed_scaling_factor": 2.5,
  "first_k_dense_replace": 3,

  "num_nextn_predict_layers": 1,

  "qk_norm": true,
  "qk_norm_eps": 1e-05
}
```

### Configuration Classes

```python
@dataclass
class GLM4Config:
    """Complete GLM-4.6 model configuration"""

    # Model architecture
    model_type: str = "glm"
    vocab_size: int = 151552
    hidden_size: int = 5120
    intermediate_size: int = 12288
    num_hidden_layers: int = 92

    # Attention
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Position embeddings
    max_position_embeddings: int = 202752
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 0.5

    # Mixture of Experts
    num_experts: int = 160
    num_experts_per_tok: int = 8
    num_shared_expert: int = 1
    moe_intermediate_size: int = 1536
    routed_scaling_factor: float = 2.5
    first_k_dense_replace: int = 3

    # Multi-token prediction
    num_nextn_predict_layers: int = 1

    # Normalization
    rms_norm_eps: float = 1e-5
    qk_norm: bool = True
    qk_norm_eps: float = 1e-5

    # Activations
    hidden_act: str = "silu"

    # Initialization
    initializer_range: float = 0.02

    # Other
    use_cache: bool = True
    tie_word_embeddings: bool = True

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
```

---

## Implementation Details

### Complete Model Class

```python
class GLM4ForCausalLM(nn.Module):
    """
    Complete GLM-4.6 model for causal language modeling

    Architecture:
        Embeddings → 92 Transformer Blocks → Final Norm → LM Head
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embeddings = GLM4Embeddings(config)

        # 92 transformer blocks
        self.layers = nn.ModuleList([
            GLM4TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # Multi-token prediction heads (optional)
        if config.num_nextn_predict_layers > 0:
            self.mtp_heads = GLM4MultiTokenPrediction(config)
        else:
            self.mtp_heads = None

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings and output weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.word_embeddings.weight

    def _init_weights(self, module):
        """Initialize weights using normal distribution"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        return_dict=True,
    ):
        """
        Forward pass through entire model

        Args:
            input_ids: (batch_size, seq_len) Token indices
            attention_mask: (batch_size, seq_len) Mask for padding
            position_ids: (batch_size, seq_len) Position indices
            past_key_values: Cached key-value pairs for generation
            labels: (batch_size, seq_len) Labels for language modeling
            use_cache: Whether to return past_key_values
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            output_router_logits: Return MoE routing decisions
            return_dict: Return ModelOutput instead of tuple

        Returns:
            loss: (optional) Language modeling loss
            logits: (batch_size, seq_len, vocab_size)
            past_key_values: (optional) Cached key-value pairs
            hidden_states: (optional) All layer hidden states
            attentions: (optional) All layer attention weights
            router_logits: (optional) MoE routing logits
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        batch_size, seq_len = input_ids.shape

        # 1. Embed tokens
        hidden_states = self.embeddings(input_ids)

        # 2. Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
            )

        # Convert to 4D causal mask
        attention_mask = self._prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_len), hidden_states, past_key_values
        )

        # 3. Prepare position IDs
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # 4. Pass through all transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if output_router_logits and len(layer_outputs) > 2:
                all_router_logits = all_router_logits + (layer_outputs[-2],)

        # 5. Final normalization
        hidden_states = self.norm(hidden_states)

        # 6. Language modeling head
        logits = self.lm_head(hidden_states)

        # 7. Multi-token prediction (optional)
        mtp_logits = None
        if self.mtp_heads is not None:
            mtp_logits = self.mtp_heads(hidden_states)

        # 8. Compute loss
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + (mtp_logits,) if mtp_logits is not None else (logits,)
            if use_cache:
                output = output + (next_decoder_cache,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            if output_router_logits:
                output = output + (all_router_logits,)

            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            mtp_logits=mtp_logits,
        )

    def _prepare_4d_causal_attention_mask(
        self, attention_mask, input_shape, hidden_states, past_key_values
    ):
        """
        Create 4D causal attention mask from 2D mask

        Args:
            attention_mask: (batch_size, seq_len)
            input_shape: (batch_size, seq_len)
            hidden_states: Current hidden states
            past_key_values: Cached KV pairs

        Returns:
            mask_4d: (batch_size, 1, seq_len, seq_len)
        """
        batch_size, seq_len = input_shape

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1
        ).to(hidden_states.device)

        # Invert (True = attend, False = mask)
        causal_mask = ~causal_mask

        # Expand dimensions
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        # Incorporate padding mask
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            causal_mask = causal_mask & padding_mask

        # Convert to additive mask (0 = attend, -inf = mask)
        mask_4d = torch.zeros_like(causal_mask, dtype=hidden_states.dtype)
        mask_4d.masked_fill_(~causal_mask, float('-inf'))

        return mask_4d
```

---

## Summary

GLM-4.6's architecture represents a carefully optimized design for:
- **Efficient computation** through sparse MoE (32B active / 355B total)
- **Strong reasoning** via depth-over-width design (92 layers)
- **Memory efficiency** through Grouped-Query Attention
- **Long-context capability** with partial RoPE and 200K window
- **Training stability** through QK-Norm and loss-free expert balancing
- **Fast inference** via multi-token prediction heads

This architecture achieves state-of-the-art performance on mathematical reasoning (AIME: 98.6) and coding tasks while being significantly more cost-effective than closed-source alternatives (8× cheaper than Claude Sonnet 4).

---

## Next Steps

See companion documentation:
- **02_PRETRAINING.md**: Training methodology and infrastructure
- **03_MID_TRAINING.md**: Domain-specific training phases
- **04_POST_TRAINING.md**: SFT and RLHF details
- **05_DATA_PIPELINE.md**: Data preprocessing and curriculum
- **06_INFRASTRUCTURE.md**: GPU cluster and distributed training
- **07_PRODUCTION_DEPLOYMENT.md**: Inference optimization and serving
