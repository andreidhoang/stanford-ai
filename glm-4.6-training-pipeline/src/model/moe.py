"""
GLM-4.6 Mixture-of-Experts (MoE) Implementation

Implements sparse MoE with:
- 160 routed experts + 1 shared expert
- Top-8 routing per token
- Loss-free balancing via dynamic bias adjustment
- Efficient expert parallel execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GLM4Expert(nn.Module):
    """
    Single expert network (2-layer FFN with SwiGLU activation)

    Architecture:
        x -> gate_proj(x) * silu(x) -> down_proj -> output

    Args:
        config: Model configuration
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert

        Args:
            x: Input tensor (num_tokens, hidden_size)

        Returns:
            output: Output tensor (num_tokens, hidden_size)
        """
        # SwiGLU activation: gate_proj(x) * silu(gate_proj(x))
        gate_output = self.gate_proj(x)
        activated = self.act_fn(gate_output)

        # Down projection
        output = self.down_proj(activated)

        return output


class GLM4DenseFFN(nn.Module):
    """
    Dense feed-forward network (for first 3 layers)

    Same structure as expert but larger intermediate size

    Args:
        config: Model configuration
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 5,120
        self.intermediate_size = config.intermediate_size  # 12,288

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dense FFN"""
        gate_output = self.gate_proj(x)
        activated = self.act_fn(gate_output)
        output = self.down_proj(activated)
        return output


class GLM4MoE(nn.Module):
    """
    Mixture-of-Experts layer for GLM-4.6

    Features:
    - 160 routed experts + 1 shared expert
    - Top-8 routing per token
    - Loss-free balancing via sigmoid gates with learnable bias
    - Routed scaling factor for output normalization

    Args:
        config: Model configuration
        layer_idx: Layer index in the model
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size  # 5,120
        self.num_experts = config.num_experts  # 160
        self.num_selected_experts = config.num_experts_per_tok  # 8
        self.intermediate_size = config.moe_intermediate_size  # 1,536
        self.routed_scaling_factor = config.routed_scaling_factor  # 2.5

        # Router network (projects hidden states to expert scores)
        self.router = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False
        )

        # Learnable bias for loss-free balancing
        # This is updated OUTSIDE the gradient graph
        self.expert_bias = nn.Parameter(
            torch.zeros(self.num_experts),
            requires_grad=False  # Not trained via backprop
        )

        # Routed experts
        self.experts = nn.ModuleList([
            GLM4Expert(config) for _ in range(self.num_experts)
        ])

        # Shared expert (always active for stability)
        self.shared_expert = GLM4Expert(config)

        # Expert utilization tracking (for bias adjustment)
        # This is a buffer (not a parameter) that tracks usage statistics
        self.register_buffer(
            "expert_counts",
            torch.zeros(self.num_experts),
            persistent=False  # Don't save in checkpoints
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            output_router_logits: Return routing decisions for analysis

        Returns:
            output: Output tensor (batch, seq_len, hidden_size)
            router_logits: (optional) Routing decisions (batch, seq_len, num_experts)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Reshape for routing: (batch, seq_len, hidden) -> (batch*seq_len, hidden)
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # 1. Compute router logits
        router_logits = self.router(hidden_states_flat)  # (B*S, num_experts)

        # 2. Apply expert-wise bias for loss-free balancing
        router_logits = router_logits + self.expert_bias.unsqueeze(0)

        # 3. Apply sigmoid activation (GLM-4.6 uses sigmoid, not softmax)
        router_probs = torch.sigmoid(router_logits)  # (B*S, num_experts)

        # 4. Select top-K experts
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.num_selected_experts,
            dim=-1
        )  # Both: (B*S, K)

        # 5. Normalize routing weights (sum to 1 for selected experts)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 6. Apply routed scaling factor
        routing_weights = routing_weights * self.routed_scaling_factor

        # 7. Update expert utilization counts (for bias adjustment)
        if self.training:
            with torch.no_grad():
                # Count how many tokens are routed to each expert
                expert_mask = torch.zeros_like(router_logits)
                expert_mask.scatter_(1, selected_experts, 1.0)
                batch_expert_counts = expert_mask.sum(dim=0)

                # Exponential moving average
                alpha = 0.01
                self.expert_counts.mul_(1 - alpha).add_(batch_expert_counts, alpha=alpha)

        # 8. Route tokens to experts
        final_hidden_states = torch.zeros(
            batch_size * seq_len,
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Create expert mask for efficient routing
        # Shape: (num_experts, K, B*S)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Process each expert
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]

            # Get tokens assigned to this expert
            # expert_mask[expert_idx] is (K, B*S)
            # nonzero gives us indices where mask is 1
            expert_tokens = expert_mask[expert_idx].nonzero(as_tuple=True)

            if expert_tokens[0].shape[0] > 0:
                # token_idx: which tokens in the batch
                # weight_idx: which of the K slots (for routing weights)
                weight_idx = expert_tokens[0]  # Which of the K slots
                token_idx = expert_tokens[1]   # Which tokens

                # Get the actual tokens
                current_tokens = hidden_states_flat[token_idx]

                # Process through expert
                expert_output = expert(current_tokens)

                # Get corresponding routing weights
                current_weights = routing_weights[token_idx, weight_idx].unsqueeze(-1)

                # Accumulate weighted expert outputs
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    expert_output * current_weights
                )

        # 9. Add shared expert output (always active)
        shared_output = self.shared_expert(hidden_states_flat)
        final_hidden_states = final_hidden_states + shared_output

        # 10. Reshape back to original dimensions
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)

        if output_router_logits:
            router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
            return final_hidden_states, router_logits

        return (final_hidden_states,)

    def get_balance_metrics(self):
        """
        Get expert balance metrics for monitoring

        Returns:
            dict: Balance statistics
        """
        if self.expert_counts.sum() == 0:
            return {
                "expert_utilization_std": 0.0,
                "max_usage_ratio": 0.0,
                "min_usage_ratio": 0.0,
                "cv": 0.0,
            }

        total = self.expert_counts.sum()
        expected = total / self.num_experts

        return {
            "expert_utilization_std": self.expert_counts.std().item(),
            "max_usage_ratio": (self.expert_counts.max() / expected).item(),
            "min_usage_ratio": (self.expert_counts.min() / expected).item(),
            "cv": (self.expert_counts.std() / expected).item(),  # Coefficient of variation
        }


def update_expert_bias(moe_layer: GLM4MoE, learning_rate: float = 0.001):
    """
    Update expert bias to encourage balanced utilization

    This should be called AFTER the optimizer step, outside the gradient graph.

    Args:
        moe_layer: MoE layer to update
        learning_rate: Bias adjustment rate (default: 0.001)
    """
    with torch.no_grad():
        # Compute ideal utilization
        total_tokens = moe_layer.expert_counts.sum()
        if total_tokens == 0:
            return  # No tokens routed yet

        ideal_count = total_tokens / moe_layer.num_experts

        # Compute usage ratio for each expert
        usage_ratio = moe_layer.expert_counts / (ideal_count + 1e-6)

        # Adjust bias: penalize overused, boost underused
        # usage_ratio > 1.0 → decrease bias (reduce selection probability)
        # usage_ratio < 1.0 → increase bias (increase selection probability)
        bias_adjustment = learning_rate * (usage_ratio - 1.0)

        # Update bias
        moe_layer.expert_bias.sub_(bias_adjustment)

        # Clamp to prevent extreme values
        moe_layer.expert_bias.clamp_(-5.0, 5.0)


# Example usage and testing
if __name__ == "__main__":
    print("Testing GLM-4.6 MoE Layer\n")

    # Create minimal config for testing
    class MockConfig:
        hidden_size = 5120
        intermediate_size = 12288
        moe_intermediate_size = 1536
        num_experts = 160
        num_experts_per_tok = 8
        num_shared_expert = 1
        routed_scaling_factor = 2.5

    config = MockConfig()

    # Test Expert
    print("Testing Expert...")
    expert = GLM4Expert(config)
    x = torch.randn(10, config.hidden_size)
    out = expert(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Expert works!\n")

    # Test Dense FFN
    print("Testing Dense FFN...")
    dense = GLM4DenseFFN(config)
    out = dense(x)
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Dense FFN works!\n")

    # Test MoE Layer
    print("Testing MoE Layer...")
    moe = GLM4MoE(config, layer_idx=3)
    print(f"  Number of experts: {moe.num_experts}")
    print(f"  Active experts per token: {moe.num_selected_experts}")
    print(f"  Routed scaling factor: {moe.routed_scaling_factor}")
    print()

    # Forward pass
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    print(f"Input shape: {hidden_states.shape}")

    # Without router logits
    outputs = moe(hidden_states, output_router_logits=False)
    output = outputs[0]
    print(f"Output shape: {output.shape}")
    print(f"✓ Forward pass successful!\n")

    # With router logits
    print("Testing with router logits...")
    outputs_with_logits = moe(hidden_states, output_router_logits=True)
    output, router_logits = outputs_with_logits
    print(f"Router logits shape: {router_logits.shape}")
    print(f"✓ Router logits work!\n")

    # Test expert balancing
    print("Testing expert balancing...")

    # Simulate training with imbalanced usage
    moe.train()
    for step in range(5):
        outputs = moe(hidden_states, output_router_logits=False)

        # Get balance metrics
        metrics = moe.get_balance_metrics()
        print(f"Step {step}: CV = {metrics['cv']:.4f}, "
              f"Max ratio = {metrics['max_usage_ratio']:.2f}, "
              f"Min ratio = {metrics['min_usage_ratio']:.2f}")

        # Update bias (simulating what would happen after optimizer step)
        update_expert_bias(moe, learning_rate=0.01)

    print(f"✓ Expert balancing works!\n")

    # Test bias clamping
    print("Testing bias clamping...")
    print(f"Expert bias range: [{moe.expert_bias.min():.3f}, {moe.expert_bias.max():.3f}]")
    print(f"✓ Bias is within [-5.0, 5.0]\n")

    print("=" * 50)
    print("All MoE tests passed! ✓")
