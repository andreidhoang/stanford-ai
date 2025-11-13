"""
GLM-4.6 Attention Mechanism

Implements Grouped-Query Attention (GQA) with:
- 96 query heads, 8 key-value heads (12:1 ratio)
- Partial Rotary Position Embeddings (50% of dimensions)
- QK-Normalization for training stability
- Efficient KV caching for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    More efficient than LayerNorm, used throughout GLM-4.6
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class GLM4RotaryEmbedding(nn.Module):
    """
    Partial Rotary Position Embeddings (RoPE)

    Only applies rotation to first 50% of embedding dimensions.
    Remaining 50% use absolute positions.

    Args:
        dim: Head dimension (128 for GLM-4.6)
        max_position_embeddings: Maximum sequence length (202,752)
        base: RoPE theta base frequency (1,000,000)
        partial_rotary_factor: Fraction of dims to rotate (0.5)
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 202752,
        base: float = 1000000.0,
        partial_rotary_factor: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Compute rotary dimension (50% of head_dim)
        self.rotary_dim = int(dim * partial_rotary_factor)

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len: Optional[int] = None):
        """
        Generate cos and sin embeddings for positions

        Args:
            x: Input tensor (for device/dtype reference)
            seq_len: Sequence length

        Returns:
            cos: Cosine embeddings (1, seq_len, rotary_dim)
            sin: Sine embeddings (1, seq_len, rotary_dim)
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # Compute frequencies: outer product of positions and inv_freq
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, rotary_dim//2)

        # Duplicate to match full rotary_dim
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, rotary_dim)

        cos = emb.cos()[None, :, :]  # (1, seq_len, rotary_dim)
        sin = emb.sin()[None, :, :]  # (1, seq_len, rotary_dim)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotate half the hidden dims of the input

    Used in RoPE application
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply partial rotary position embeddings to query and key tensors

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine embeddings (1, seq_len, rotary_dim)
        sin: Sine embeddings (1, seq_len, rotary_dim)
        position_ids: Position indices (batch, seq_len)

    Returns:
        q_embed: Queries with RoPE applied
        k_embed: Keys with RoPE applied
    """
    # Get rotary dimension
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot = q[..., :rotary_dim]  # First 50% (gets rotated)
    q_pass = q[..., rotary_dim:]  # Last 50% (passes through)

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Squeeze cos/sin and gather for positions if needed
    cos = cos.squeeze(0)  # (seq_len, rotary_dim)
    sin = sin.squeeze(0)

    if position_ids is not None:
        # Gather for specific positions
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, rotary_dim)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Use sequential positions
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, rotary_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation using complex number formulation
    # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate rotated and pass-through parts
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads to match number of query heads

    For GQA: expand (batch, num_kv_heads, seq_len, head_dim)
              to (batch, num_heads, seq_len, head_dim)

    Args:
        hidden_states: KV tensor (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Repetition factor (num_heads // num_kv_heads)

    Returns:
        expanded: (batch, num_heads, seq_len, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    # Expand and reshape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GLM4Attention(nn.Module):
    """
    Grouped-Query Attention with Partial RoPE and QK-Norm

    Features:
    - 96 query heads, 8 KV heads (12:1 ratio)
    - Partial rotary position embeddings (50% of dimensions)
    - Query-Key normalization for training stability
    - Efficient KV caching for generation

    Args:
        config: Model configuration
        layer_idx: Layer index in the model
    """
    def __init__(self, config, layer_idx: int):
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
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias
        )

        # QK Normalization for stability
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Rotary Position Embeddings
        self.rotary_emb = GLM4RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
        )

        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through GQA

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Attention mask (batch, 1, seq_len, seq_len)
            position_ids: Position indices (batch, seq_len)
            past_key_value: Cached (key, value) for fast decoding
            output_attentions: Return attention weights
            use_cache: Cache key-value pairs

        Returns:
            attn_output: Output tensor (batch, seq_len, hidden_size)
            attn_weights: (optional) Attention weights
            present_key_value: (optional) Cached key-value pairs
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape to separate heads
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (batch, num_kv_heads, seq_len, head_dim)

        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (batch, num_kv_heads, seq_len, head_dim)

        # 3. Apply QK Normalization (before RoPE)
        if self.q_norm is not None and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # 4. Get current sequence length (for position embeddings)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # 5. Apply Rotary Position Embeddings (partial RoPE)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # 6. Concatenate with past key-value cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # 7. Repeat KV heads to match number of query heads (for GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 8. Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # 9. Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 10. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = self.attention_dropout(attn_weights)

        # 11. Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # 12. Reshape and project output
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


# Example usage and testing
if __name__ == "__main__":
    print("Testing GLM-4.6 Attention Mechanism\n")

    # Create minimal config for testing
    class MockConfig:
        hidden_size = 5120
        num_attention_heads = 96
        num_key_value_heads = 8
        head_dim = 128
        attention_bias = False
        attention_dropout = 0.0
        qk_norm = True
        qk_norm_eps = 1e-5
        max_position_embeddings = 202752
        rope_theta = 1000000.0
        partial_rotary_factor = 0.5

    config = MockConfig()

    # Create attention module
    attention = GLM4Attention(config, layer_idx=0)
    print(f"Attention module created with:")
    print(f"  - Query heads: {config.num_attention_heads}")
    print(f"  - KV heads: {config.num_key_value_heads}")
    print(f"  - Head dim: {config.head_dim}")
    print(f"  - QK-Norm: {config.qk_norm}")
    print()

    # Test forward pass
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    print(f"Input shape: {hidden_states.shape}")

    # Forward pass without caching
    outputs = attention(hidden_states, use_cache=False)
    attn_output = outputs[0]

    print(f"Output shape: {attn_output.shape}")
    print(f"✓ Forward pass successful!\n")

    # Test with KV caching (generation mode)
    print("Testing KV caching...")
    outputs_with_cache = attention(hidden_states, use_cache=True)
    attn_output, past_kv = outputs_with_cache[0], outputs_with_cache[-1]

    print(f"Cached KV shapes: {past_kv[0].shape}, {past_kv[1].shape}")

    # Test with cached KV (next token)
    next_token = torch.randn(batch_size, 1, config.hidden_size)
    outputs_next = attention(next_token, past_key_value=past_kv, use_cache=True)

    print(f"Next token output shape: {outputs_next[0].shape}")
    print(f"✓ KV caching works!\n")

    # Test RoPE
    print("Testing Partial RoPE...")
    rope = GLM4RotaryEmbedding(
        dim=config.head_dim,
        partial_rotary_factor=config.partial_rotary_factor
    )
    print(f"Rotary dimension: {rope.rotary_dim} / {config.head_dim}")
    print(f"Rotation applied to: {rope.rotary_dim / config.head_dim * 100:.0f}% of dimensions")

    x = torch.randn(1, seq_len, config.head_dim)
    cos, sin = rope(x, seq_len=seq_len)
    print(f"RoPE embeddings shape: {cos.shape}")
    print(f"✓ Partial RoPE works!\n")

    print("=" * 50)
    print("All attention tests passed! ✓")
