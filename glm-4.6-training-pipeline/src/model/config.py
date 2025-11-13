"""
GLM-4.6 Model Configuration

Complete configuration for the 355B parameter MoE model with all hyperparameters
matching the official GLM-4.6 specifications.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class GLM4Config:
    """
    Configuration class for GLM-4.6 model

    Specifications:
    - 355B total parameters, 32B active
    - 92 transformer layers
    - 160 routed + 1 shared expert
    - Top-8 routing
    - Grouped-Query Attention (96 heads, 8 KV heads)
    - 200K context window
    """

    # Model architecture
    model_type: str = "glm"
    architectures: List[str] = field(default_factory=lambda: ["GLM4ForCausalLM"])

    # Vocabulary and embeddings
    vocab_size: int = 151552
    hidden_size: int = 5120
    intermediate_size: int = 12288

    # Transformer layers
    num_hidden_layers: int = 92

    # Attention configuration
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Position embeddings
    max_position_embeddings: int = 202752
    max_sequence_length: int = 200000
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[dict] = None
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

    # Activation functions
    hidden_act: str = "silu"

    # Initialization
    initializer_range: float = 0.02

    # Generation
    use_cache: bool = True
    tie_word_embeddings: bool = True

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Precision
    torch_dtype: str = "bfloat16"

    # Framework version
    transformers_version: str = "4.44.0"

    def __post_init__(self):
        """Validate configuration"""
        # Ensure num_attention_heads is divisible by num_key_value_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )

        # Ensure head_dim * num_attention_heads = hidden_size
        expected_hidden_size = self.head_dim * self.num_attention_heads
        if expected_hidden_size != self.hidden_size:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_attention_heads ({self.num_attention_heads}) "
                f"= {expected_hidden_size}, but hidden_size is {self.hidden_size}"
            )

    @property
    def num_key_value_groups(self) -> int:
        """Number of query heads per key-value head"""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def active_parameters(self) -> int:
        """
        Estimate active parameters per forward pass

        Returns:
            Approximate number of active parameters (in billions)
        """
        # Attention (always active)
        attn_params_per_layer = (
            # Q, K, V, O projections
            4 * self.hidden_size * (self.num_attention_heads * self.head_dim) +
            # K, V use fewer heads
            2 * self.hidden_size * (self.num_key_value_heads * self.head_dim -
                                    self.num_attention_heads * self.head_dim)
        )
        total_attn = self.num_hidden_layers * attn_params_per_layer

        # Dense FFN (first 3 layers)
        dense_ffn_per_layer = 2 * self.hidden_size * self.intermediate_size
        total_dense = self.first_k_dense_replace * dense_ffn_per_layer

        # MoE (active experts only)
        expert_params = 2 * self.hidden_size * self.moe_intermediate_size
        active_experts_per_layer = self.num_experts_per_tok + self.num_shared_expert
        moe_layers = self.num_hidden_layers - self.first_k_dense_replace
        total_moe = moe_layers * active_experts_per_layer * expert_params

        # Embeddings + norms
        embeddings = self.vocab_size * self.hidden_size
        norms = self.num_hidden_layers * 2 * self.hidden_size

        total = total_attn + total_dense + total_moe + embeddings + norms
        return total / 1e9  # Convert to billions

    @property
    def total_parameters(self) -> int:
        """
        Estimate total model parameters

        Returns:
            Approximate total parameters (in billions)
        """
        # Attention
        attn_params_per_layer = (
            4 * self.hidden_size * (self.num_attention_heads * self.head_dim) +
            2 * self.hidden_size * (self.num_key_value_heads * self.head_dim -
                                    self.num_attention_heads * self.head_dim)
        )
        total_attn = self.num_hidden_layers * attn_params_per_layer

        # Dense FFN
        dense_ffn_per_layer = 2 * self.hidden_size * self.intermediate_size
        total_dense = self.first_k_dense_replace * dense_ffn_per_layer

        # MoE (all experts)
        expert_params = 2 * self.hidden_size * self.moe_intermediate_size
        total_experts_per_layer = self.num_experts + self.num_shared_expert
        moe_layers = self.num_hidden_layers - self.first_k_dense_replace
        total_moe = moe_layers * total_experts_per_layer * expert_params

        # Router
        router_params = moe_layers * self.hidden_size * self.num_experts

        # Embeddings + norms
        embeddings = self.vocab_size * self.hidden_size
        norms = self.num_hidden_layers * 2 * self.hidden_size

        total = total_attn + total_dense + total_moe + router_params + embeddings + norms
        return total / 1e9

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """
        Load configuration from pretrained model

        Args:
            model_name_or_path: Model name or path containing config.json

        Returns:
            GLM4Config instance
        """
        import os
        config_file = os.path.join(model_name_or_path, "config.json")

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def save_pretrained(self, save_directory: str):
        """
        Save configuration to directory

        Args:
            save_directory: Directory to save config.json
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        config_dict = self.__dict__.copy()

        # Convert to JSON-serializable format
        for key, value in config_dict.items():
            if isinstance(value, type):
                config_dict[key] = value.__name__

        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return self.__dict__.copy()

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"GLM4Config(\n"
            f"  model_type='{self.model_type}',\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  num_experts={self.num_experts},\n"
            f"  num_experts_per_tok={self.num_experts_per_tok},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  max_sequence_length={self.max_sequence_length},\n"
            f"  total_parameters={self.total_parameters:.1f}B,\n"
            f"  active_parameters={self.active_parameters:.1f}B\n"
            f")"
        )


# Pre-defined configurations

def get_glm46_config() -> GLM4Config:
    """Get standard GLM-4.6 configuration (355B total, 32B active)"""
    return GLM4Config()


def get_glm46_small_config() -> GLM4Config:
    """
    Get smaller GLM-4.6 variant for experimentation

    Scaled-down version:
    - 24 layers (vs 92)
    - 2048 hidden size (vs 5120)
    - 32 experts (vs 160)
    - 4 active experts (vs 8)

    Total parameters: ~15B
    Active parameters: ~3B
    """
    return GLM4Config(
        num_hidden_layers=24,
        hidden_size=2048,
        intermediate_size=4096,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=64,
        num_experts=32,
        num_experts_per_tok=4,
        moe_intermediate_size=768,
    )


def get_glm46_medium_config() -> GLM4Config:
    """
    Get medium GLM-4.6 variant

    Medium-scale version:
    - 48 layers (vs 92)
    - 3584 hidden size (vs 5120)
    - 64 experts (vs 160)
    - 6 active experts (vs 8)

    Total parameters: ~100B
    Active parameters: ~15B
    """
    return GLM4Config(
        num_hidden_layers=48,
        hidden_size=3584,
        intermediate_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=56,
        num_experts=64,
        num_experts_per_tok=6,
        moe_intermediate_size=1024,
    )


# Export all configurations
__all__ = [
    'GLM4Config',
    'get_glm46_config',
    'get_glm46_small_config',
    'get_glm46_medium_config',
]


if __name__ == "__main__":
    # Test configurations
    print("=== GLM-4.6 Configurations ===\n")

    print("Standard GLM-4.6:")
    config_full = get_glm46_config()
    print(config_full)
    print(f"\nActive/Total ratio: {config_full.active_parameters / config_full.total_parameters * 100:.1f}%\n")

    print("\nSmall variant:")
    config_small = get_glm46_small_config()
    print(config_small)
    print(f"\nActive/Total ratio: {config_small.active_parameters / config_small.total_parameters * 100:.1f}%\n")

    print("\nMedium variant:")
    config_medium = get_glm46_medium_config()
    print(config_medium)
    print(f"\nActive/Total ratio: {config_medium.active_parameters / config_medium.total_parameters * 100:.1f}%\n")
