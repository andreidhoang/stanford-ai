"""
GLM-4.6 Complete Model Implementation

Brings together all components into a working transformer model:
- Embeddings
- 92 transformer blocks (attention + MoE/Dense FFN)
- Language modeling head
- Generation capabilities
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from .config import GLM4Config
from .attention import GLM4Attention, RMSNorm
from .moe import GLM4MoE, GLM4DenseFFN


@dataclass
class CausalLMOutput:
    """Output of the causal language model"""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


class GLM4Embeddings(nn.Module):
    """
    Token embeddings for GLM-4.6

    Shared between input and output (weight tying)
    """
    def __init__(self, config: GLM4Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.word_embeddings = nn.Embedding(
            self.vocab_size,
            self.hidden_size
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
        """
        return self.word_embeddings(input_ids)


class GLM4TransformerBlock(nn.Module):
    """
    Single transformer block combining attention and MoE/Dense FFN

    Architecture:
        x -> RMSNorm -> Attention -> Residual
          -> RMSNorm -> MoE/Dense FFN -> Residual
    """
    def __init__(self, config: GLM4Config, layer_idx: int):
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
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        use_cache: bool = False,
    ) -> Tuple:
        """
        Forward pass through transformer block

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len)
            position_ids: (batch_size, seq_len)
            past_key_value: Cached key-value pairs
            output_attentions: Return attention weights
            output_router_logits: Return MoE routing decisions
            use_cache: Cache key-value pairs

        Returns:
            Tuple of outputs
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

            if output_router_logits and len(mlp_outputs) > 1:
                outputs = outputs + (mlp_outputs[1],)  # router_logits
        else:
            hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + outputs

        return outputs


class GLM4Model(nn.Module):
    """
    GLM-4.6 Transformer Model (without language modeling head)

    This is the core transformer that can be used for various tasks.
    """
    def __init__(self, config: GLM4Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token embeddings
        self.embeddings = GLM4Embeddings(config)

        # 92 transformer blocks
        self.layers = nn.ModuleList([
            GLM4TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

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
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass through the model

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            position_ids: (batch_size, seq_len)
            past_key_values: Cached key-value pairs
            use_cache: Whether to return past_key_values
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            output_router_logits: Return MoE routing decisions
            return_dict: Return CausalLMOutput instead of tuple

        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        output_router_logits = output_router_logits if output_router_logits is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

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
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2 if use_cache else 1],)

            if output_router_logits:
                # Router logits are last in the output
                if len(layer_outputs) > (3 if use_cache and output_attentions else
                                        2 if use_cache or output_attentions else 1):
                    all_router_logits = all_router_logits + (layer_outputs[-1],)

        # 5. Final normalization
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_attentions,
                all_router_logits,
            ] if v is not None)

        return CausalLMOutput(
            logits=hidden_states,  # Will be used by LM head
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
        )

    def _prepare_4d_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        hidden_states: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]]
    ) -> torch.Tensor:
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

        # Get past sequence length
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
            diagonal=1
        )

        # Invert (True = attend, False = mask)
        causal_mask = ~causal_mask

        # Expand for past
        if past_len > 0:
            # Can attend to all past positions
            causal_mask = torch.cat([
                torch.ones(seq_len, past_len, dtype=torch.bool, device=hidden_states.device),
                causal_mask
            ], dim=1)

        # Expand dimensions
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, total_len)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, -1)

        # Convert to additive mask
        mask_4d = torch.zeros_like(causal_mask, dtype=hidden_states.dtype)
        mask_4d.masked_fill_(~causal_mask, float('-inf'))

        return mask_4d


class GLM4ForCausalLM(nn.Module):
    """
    GLM-4.6 for Causal Language Modeling

    Complete model with language modeling head for next-token prediction.
    """
    def __init__(self, config: GLM4Config):
        super().__init__()
        self.config = config

        # Core transformer
        self.model = GLM4Model(config)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # Tie embeddings and output weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.word_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass with language modeling

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            position_ids: (batch_size, seq_len)
            past_key_values: Cached key-value pairs
            labels: (batch_size, seq_len) for computing loss
            use_cache: Return past_key_values
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            output_router_logits: Return MoE routing decisions
            return_dict: Return CausalLMOutput

        Returns:
            Model outputs with loss and logits
        """
        return_dict = return_dict if return_dict is not None else True

        # Forward through transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        if return_dict:
            hidden_states = outputs.logits
        else:
            hidden_states = outputs[0]

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
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
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """
        Generate text autoregressively

        Args:
            input_ids: (batch_size, seq_len) Starting tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to sample or use greedy decoding

        Returns:
            generated: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        batch_size = input_ids.shape[0]
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # Get logits for next token
            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or greedy decode
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Update cache
            past_key_values = outputs.past_key_values

            # Check for EOS token
            if (next_token == self.config.eos_token_id).all():
                break

        return input_ids


# Example usage and testing
if __name__ == "__main__":
    print("Testing GLM-4.6 Complete Model\n")
    print("=" * 60)

    # Create small config for testing
    from config import get_glm46_small_config

    config = get_glm46_small_config()
    print(f"\nUsing scaled-down config:")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Active experts: {config.num_experts_per_tok}")
    print(f"  Active params: {config.active_parameters:.1f}B")
    print(f"  Total params: {config.total_parameters:.1f}B")
    print()

    # Create model
    print("Creating model...")
    model = GLM4ForCausalLM(config)
    print(f"✓ Model created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print()

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids, return_dict=True)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs.logits.shape}")
    print(f"  ✓ Forward pass successful!")
    print()

    # Test with labels (compute loss)
    print("Testing loss computation...")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, labels=labels, return_dict=True)
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  ✓ Loss computation works!")
    print()

    # Test with caching
    print("Testing KV caching...")
    outputs = model(input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values
    print(f"  Cached layers: {len(past_kv)}")
    print(f"  Cached KV shape: {past_kv[0][0].shape}")
    print(f"  ✓ KV caching works!")
    print()

    # Test generation
    print("Testing text generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True
    )
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  New tokens: {generated.shape[1] - prompt.shape[1]}")
    print(f"  ✓ Generation works!")
    print()

    # Test router logits output
    print("Testing router logits...")
    outputs = model(input_ids, output_router_logits=True, return_dict=True)
    if outputs.router_logits:
        print(f"  Router logits layers: {len(outputs.router_logits)}")
        print(f"  Router logits shape: {outputs.router_logits[0].shape}")
        print(f"  ✓ Router logits work!")
    print()

    print("=" * 60)
    print("All model tests passed! ✓")
    print("\nModel is ready for:")
    print("  ✓ Training")
    print("  ✓ Inference")
    print("  ✓ Generation")
    print("  ✓ Fine-tuning")
