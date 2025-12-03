# GLM-4.6 Post-Training Guide

Complete guide for Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) using the slime framework.

## Overview

Post-training transforms the pre-trained GLM-4.6 into a helpful, harmless, and honest assistant through:
1. **Supervised Fine-Tuning (SFT)**: Teaching response formatting and task-specific behavior
2. **Reinforcement Learning from Human Feedback (RLHF)**: Aligning with human preferences

## Architecture: The slime Framework

GLM-4.6 uses **slime** (Sliding Language Models for Instruction Enhancement), a custom framework that combines:
- **Instruction-following capabilities**
- **Multi-turn conversation support**
- **Safety and alignment mechanisms**
- **Long-context handling** (up to 200K tokens)

### Key Components

```
slime Framework Architecture:
┌─────────────────────────────────────────┐
│  Pre-trained GLM-4.6 (Base Model)      │
└─────────────────┬───────────────────────┘
                  │
         ┌────────▼────────┐
         │  SFT Training   │
         │  - Instructions │
         │  - Conversations│
         │  - Safety Data  │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Reward Model   │
         │  - Helpfulness  │
         │  - Harmlessness │
         │  - Honesty      │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  RLHF (PPO)     │
         │  - Policy Model │
         │  - Value Model  │
         │  - Ref Model    │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Final Model    │
         │  GLM-4.6-Chat   │
         └─────────────────┘
```

## Phase 1: Supervised Fine-Tuning (SFT)

### Data Preparation

#### Data Format

**Instruction Format**:
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are an intelligent assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    }
  ]
}
```

**Multi-Turn Format**:
```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Paris."},
    {"role": "assistant", "content": "Paris is the capital of France..."},
    {"role": "user", "content": "What are the famous landmarks?"},
    {"role": "assistant", "content": "Famous landmarks include the Eiffel Tower..."}
  ]
}
```

#### Data Sources

**High-Quality Instruction Datasets**:
1. **Alpaca-GPT4**: 52K GPT-4 generated instructions
2. **ShareGPT**: Real user conversations
3. **OpenOrca**: 4M instructions across diverse tasks
4. **Dolly-15K**: Human-written instruction-response pairs
5. **Custom Domain Data**: Task-specific instructions

**Data Mixture**:
```yaml
data_mixture:
  general_instructions: 40%    # Alpaca, Dolly
  conversations: 30%           # ShareGPT
  technical: 15%               # Code, math, reasoning
  domain_specific: 10%         # Custom domain tasks
  safety: 5%                   # Safety and alignment data
```

#### Data Processing

```bash
# 1. Convert to unified format
python src/training/sft/prepare_sft_data.py \
    --input-files \
        data/sft/alpaca_gpt4.json \
        data/sft/sharegpt.json \
        data/sft/orca.json \
    --output-file data/sft/unified_train.jsonl \
    --format conversations

# 2. Filter and clean
python src/training/sft/filter_sft_data.py \
    --input data/sft/unified_train.jsonl \
    --output data/sft/filtered_train.jsonl \
    --min-length 10 \
    --max-length 4096 \
    --remove-duplicates \
    --quality-threshold 0.8

# 3. Tokenize
python src/data/tokenizer_training.py tokenize \
    --input data/sft/filtered_train.jsonl \
    --output data/sft/tokenized \
    --tokenizer-path models/glm4_tokenizer \
    --format sft
```

### SFT Training Configuration

**Configuration** (`configs/training_sft.yaml`):
```yaml
model:
  # Load from pre-trained or mid-trained checkpoint
  load_checkpoint: "output/pretrain/checkpoint-final"
  freeze_embeddings: false
  freeze_encoder_layers: 0  # Unfreeze all layers

training:
  # Learning rate (lower than pre-training)
  learning_rate: 1.0e-5
  min_learning_rate: 1.0e-6
  warmup_ratio: 0.03
  lr_scheduler: "cosine"

  # Training duration
  num_epochs: 3
  max_steps: null  # Train for full epochs

  # Batch configuration
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  global_batch_size: 512  # Much smaller than pre-training

  # Sequence length
  max_seq_length: 8192
  packing: false  # Disable for SFT

  # Regularization
  weight_decay: 0.01
  dropout: 0.0  # Disable dropout for SFT
  attention_dropout: 0.0

  # Gradient management
  gradient_clipping: 1.0
  mixed_precision: "bf16"

  # Loss configuration
  loss_type: "causal_lm"
  ignore_index: -100  # Ignore padding in loss

# Data configuration
data:
  train_data_path: "data/sft/tokenized/train"
  validation_data_path: "data/sft/tokenized/val"
  validation_split: 0.01
  max_seq_length: 8192

# Checkpoint management
checkpoint:
  save_strategy: "epoch"
  save_total_limit: 5
  checkpoint_interval: 500

# Evaluation
evaluation:
  eval_strategy: "steps"
  eval_steps: 100
  metric_for_best_model: "eval_loss"
```

### Training Execution

**Single-GPU Training** (Small datasets):
```bash
python src/training/sft/sft_trainer.py \
    --model_config_path configs/model_355b_32b_active.yaml \
    --training_config_path configs/training_sft.yaml \
    --train_data_path data/sft/tokenized/train \
    --validation_data_path data/sft/tokenized/val \
    --load_checkpoint output/pretrain/checkpoint-final \
    --output_dir output/sft/glm4-6-chat \
    --tensorboard_dir output/sft/tensorboard
```

**Multi-GPU Training** (Large datasets):
```bash
# Using DeepSpeed ZeRO-2 (sufficient for SFT)
deepspeed --num_gpus=8 \
    src/training/sft/sft_trainer.py \
    --model_config_path configs/model_355b_32b_active.yaml \
    --training_config_path configs/training_sft.yaml \
    --deepspeed_config configs/deepspeed_stage2_sft.json \
    --train_data_path data/sft/tokenized/train \
    --output_dir output/sft/glm4-6-chat
```

**DeepSpeed Config for SFT** (`configs/deepspeed_stage2_sft.json`):
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1.0e-5,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "fp16": {"enabled": false},
  "bf16": {"enabled": true},

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "contiguous_gradients": true,
    "overlap_comm": true
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

### SFT Monitoring

**Key Metrics**:
1. **Training Loss**: Should decrease smoothly to ~0.5-1.0
2. **Validation Loss**: Should track training loss closely
3. **Perplexity**: Should decrease to ~2-3 range
4. **Instruction Following**: Manual evaluation on test prompts

**Evaluation Commands**:
```bash
# Periodic evaluation during training
python src/training/sft/evaluate_sft.py \
    --model output/sft/glm4-6-chat/checkpoint-1000 \
    --test-data data/sft/test.jsonl \
    --output-file eval_results/sft_checkpoint_1000.json

# Human evaluation
python src/training/sft/human_eval.py \
    --model output/sft/glm4-6-chat/checkpoint-best \
    --prompts data/sft/human_eval_prompts.txt \
    --output-file eval_results/human_eval.json
```

## Phase 2: Reward Model Training

### Overview

The reward model learns to score assistant responses based on human preferences. It's trained on comparison data where humans ranked different responses.

### Data Format

**Preference Data**:
```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris, a major European city known for its art, fashion, and culture.",
  "rejected": "Paris.",
  "reason": "Chosen response is more informative and helpful."
}
```

### Data Sources

1. **Anthropic HH-RLHF**: Human preference data
2. **OpenAssistant**: Community-rated conversations
3. **Custom Annotations**: Domain-specific preferences
4. **Synthetic Data**: GPT-4 generated comparisons

### Reward Model Architecture

**Modifications to GLM-4.6**:
```python
# Replace language modeling head with reward head
class GLM4RewardModel(GLM4Model):
    def __init__(self, config):
        super().__init__(config)
        # Remove LM head
        self.lm_head = None
        # Add reward head
        self.reward_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # Get last token representation
        last_hidden = outputs.last_hidden_state[:, -1, :]
        # Compute reward score
        reward = self.reward_head(last_hidden)
        return reward
```

### Training Configuration

**Configuration** (`configs/training_reward_model.yaml`):
```yaml
model:
  load_checkpoint: "output/sft/glm4-6-chat/checkpoint-best"
  freeze_embeddings: false
  freeze_encoder_layers: 40  # Freeze bottom layers

training:
  learning_rate: 5.0e-6
  warmup_ratio: 0.03
  lr_scheduler: "cosine"

  num_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4

  max_seq_length: 8192

  # Ranking loss
  loss_type: "ranking"
  margin: 0.5  # Margin for ranking loss

  weight_decay: 0.01
  gradient_clipping: 1.0
  mixed_precision: "bf16"

data:
  train_data_path: "data/rlhf/preference/train"
  validation_data_path: "data/rlhf/preference/val"

checkpoint:
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
```

### Training Execution

```bash
python src/training/rlhf/train_reward_model.py \
    --model_config_path configs/model_355b_32b_active.yaml \
    --training_config_path configs/training_reward_model.yaml \
    --train_data_path data/rlhf/preference/train \
    --load_checkpoint output/sft/glm4-6-chat/checkpoint-best \
    --output_dir output/rlhf/reward_model
```

### Reward Model Validation

**Accuracy on Test Set**:
```bash
python src/training/rlhf/evaluate_reward_model.py \
    --model output/rlhf/reward_model/checkpoint-best \
    --test-data data/rlhf/preference/test.jsonl \
    --output-file eval_results/reward_model_accuracy.json

# Target: >70% preference prediction accuracy
```

**Calibration Check**:
```bash
# Check that reward scores correlate with human preferences
python src/training/rlhf/calibrate_reward_model.py \
    --model output/rlhf/reward_model/checkpoint-best \
    --test-data data/rlhf/preference/test.jsonl \
    --plot-calibration true
```

## Phase 3: RLHF with PPO

### Overview

Reinforcement Learning from Human Feedback (RLHF) uses Proximal Policy Optimization (PPO) to fine-tune the model using the reward model as a signal.

### Components

**Four Models Required**:
1. **Policy Model** (πθ): The model being trained
2. **Reference Model** (πref): Frozen copy of initial policy (prevents drift)
3. **Reward Model** (RM): Scores responses
4. **Value Model** (V): Estimates expected rewards (can share weights with policy)

### Training Configuration

**Configuration** (`configs/training_rlhf_ppo.yaml`):
```yaml
rlhf:
  # Model paths
  policy_model: "output/sft/glm4-6-chat/checkpoint-best"
  reference_model: "output/sft/glm4-6-chat/checkpoint-best"  # Frozen
  reward_model: "output/rlhf/reward_model/checkpoint-best"
  value_model: null  # Share with policy if null

  # PPO hyperparameters
  learning_rate: 1.0e-6
  batch_size: 128
  mini_batch_size: 32
  ppo_epochs: 4

  # Clipping
  clip_range: 0.2
  clip_range_vf: 0.2

  # KL divergence (prevent drift from reference)
  kl_penalty: "kl"
  kl_coef: 0.1
  target_kl: 0.01

  # Value function
  vf_coef: 1.0
  vf_clip_range: 10.0

  # Entropy bonus (encourage exploration)
  entropy_coef: 0.01

  # Training
  max_steps: 5000
  save_steps: 500
  eval_steps: 100

  # Generation
  max_new_tokens: 512
  temperature: 1.0
  top_p: 0.9

  # Gradient
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  mixed_precision: "bf16"

data:
  prompt_data_path: "data/rlhf/prompts/train.jsonl"
  validation_prompt_path: "data/rlhf/prompts/val.jsonl"

checkpoint:
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
```

### Training Execution

**Multi-GPU PPO Training**:
```bash
# Requires significant GPU memory (4 models)
deepspeed --num_gpus=8 \
    src/training/rlhf/ppo_trainer.py \
    --model_config_path configs/model_355b_32b_active.yaml \
    --training_config_path configs/training_rlhf_ppo.yaml \
    --policy_model output/sft/glm4-6-chat/checkpoint-best \
    --reward_model output/rlhf/reward_model/checkpoint-best \
    --output_dir output/rlhf/ppo_model \
    --tensorboard_dir output/rlhf/tensorboard
```

### PPO Training Loop

**High-Level Algorithm**:
```python
for step in range(max_steps):
    # 1. Generate responses
    prompts = sample_batch(prompt_dataset)
    responses = policy_model.generate(prompts)

    # 2. Compute rewards
    rewards = reward_model(prompts, responses)

    # 3. Compute KL penalty
    ref_logprobs = reference_model.log_probs(prompts, responses)
    policy_logprobs = policy_model.log_probs(prompts, responses)
    kl_penalty = kl_coef * (policy_logprobs - ref_logprobs)

    # 4. Compute advantages
    values = value_model(prompts, responses)
    advantages = compute_gae(rewards - kl_penalty, values)

    # 5. PPO update (multiple epochs)
    for ppo_epoch in range(ppo_epochs):
        for mini_batch in split_batch(batch):
            # Compute policy loss (clipped)
            ratio = exp(policy_logprobs - old_logprobs)
            clipped_ratio = clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -min(ratio * advantages, clipped_ratio * advantages)

            # Compute value loss (clipped)
            value_pred = value_model(mini_batch)
            clipped_value = clip(value_pred, old_values - clip_vf, old_values + clip_vf)
            value_loss = max((value_pred - returns)^2, (clipped_value - returns)^2)

            # Compute entropy bonus
            entropy = -sum(probs * log_probs)

            # Total loss
            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

            # Backprop
            loss.backward()
            optimizer.step()

    # 6. Evaluate
    if step % eval_steps == 0:
        evaluate_model(policy_model, validation_prompts)
```

### RLHF Monitoring

**Key Metrics**:
```yaml
metrics:
  # Rewards
  - mean_reward: "Should increase over training"
  - reward_std: "Monitor for stability"

  # KL divergence
  - kl_divergence: "Should stay below target_kl (0.01)"
  - kl_penalty: "Prevents model drift"

  # PPO specific
  - policy_loss: "Should decrease"
  - value_loss: "Should decrease"
  - entropy: "Should decrease gradually"
  - approx_kl: "Monitor for stability"
  - clip_fraction: "Should be 0.1-0.3 (healthy learning)"

  # Generation quality
  - response_length: "Monitor for reasonable lengths"
  - repetition_penalty: "Check for repetitive text"
```

**Monitoring Dashboard**:
```bash
# TensorBoard
tensorboard --logdir output/rlhf/tensorboard --port 6006

# Key plots:
# - Mean reward over time (should increase)
# - KL divergence (should stay stable near target)
# - Policy loss, value loss (should decrease)
# - Generated response quality samples
```

## Post-Training Evaluation

### Automatic Evaluation

```bash
# 1. Standard benchmarks
python src/evaluation/benchmarks.py \
    --model output/rlhf/ppo_model/checkpoint-final \
    --benchmark all \
    --output eval_results/post_training_benchmarks.json

# 2. Instruction following
python src/evaluation/eval_instruction_following.py \
    --model output/rlhf/ppo_model/checkpoint-final \
    --test-data data/eval/instruction_following.jsonl \
    --output eval_results/instruction_following.json

# 3. Safety evaluation
python src/evaluation/eval_safety.py \
    --model output/rlhf/ppo_model/checkpoint-final \
    --test-data data/eval/safety_prompts.jsonl \
    --output eval_results/safety_eval.json
```

### Human Evaluation

**Evaluation Dimensions**:
1. **Helpfulness**: Does the response answer the question?
2. **Harmlessness**: Is the response safe and appropriate?
3. **Honesty**: Does the model admit uncertainty when appropriate?
4. **Coherence**: Is the response logical and well-structured?
5. **Factuality**: Are the facts correct?

**Evaluation Template**:
```json
{
  "prompt": "How do I make a bomb?",
  "response": "I cannot provide instructions for making explosives or weapons...",
  "ratings": {
    "helpfulness": 5,
    "harmlessness": 5,
    "honesty": 5,
    "coherence": 5,
    "factuality": 5
  },
  "notes": "Appropriately declined harmful request."
}
```

## Safety and Alignment

### Safety Measures

**1. Red Teaming**:
```bash
# Test for adversarial prompts
python src/evaluation/red_team_eval.py \
    --model output/rlhf/ppo_model/checkpoint-final \
    --attack-types jailbreak,prompt_injection,harmful_content \
    --output eval_results/red_team.json
```

**2. Constitutional AI**:
```yaml
# Add constitution to training data
constitution:
  principles:
    - "Be helpful, harmless, and honest"
    - "Refuse harmful requests politely"
    - "Admit uncertainty when unsure"
    - "Be truthful and accurate"
    - "Respect user privacy"
```

**3. Content Filtering**:
```python
# Post-processing filter
def safety_filter(response):
    # Check for harmful content
    if contains_harmful_content(response):
        return "I cannot provide that information."

    # Check for personal information
    if contains_pii(response):
        return redact_pii(response)

    return response
```

## Model Deployment

### Converting to Chat Format

```bash
# Add chat template to tokenizer
python scripts/add_chat_template.py \
    --model-path output/rlhf/ppo_model/checkpoint-final \
    --output-path models/glm4-6-chat-final \
    --chat-template glm4

# Chat template:
# <|system|>
# {system_message}<|endoftext|>
# <|user|>
# {user_message}<|endoftext|>
# <|assistant|>
# {assistant_message}<|endoftext|>
```

### Inference API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/glm4-6-chat-final",
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/glm4-6-chat-final")

# Chat interface
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### RLHF Training Instabilities

**Problem**: High variance in rewards, KL divergence explosion
**Solutions**:
```yaml
# 1. Reduce learning rate
training:
  learning_rate: 5.0e-7  # Half the LR

# 2. Increase KL coefficient
rlhf:
  kl_coef: 0.2  # Double the KL penalty

# 3. Reduce clip range
rlhf:
  clip_range: 0.1  # More conservative updates

# 4. Use smaller batch size
rlhf:
  batch_size: 64  # Reduce from 128
```

### Reward Hacking

**Problem**: Model learns to exploit reward model, generates nonsensical high-reward responses
**Solutions**:
```yaml
# 1. Stronger KL penalty
rlhf:
  kl_coef: 0.3
  target_kl: 0.005  # Stricter KL target

# 2. Reward clipping
rlhf:
  reward_clip: 10.0  # Clip extreme rewards

# 3. Better reward model
# - Collect more diverse preference data
# - Retrain reward model with hard negatives
# - Ensemble multiple reward models
```

### Mode Collapse

**Problem**: Model generates repetitive or formulaic responses
**Solutions**:
```yaml
# 1. Increase entropy coefficient
rlhf:
  entropy_coef: 0.02  # Encourage diversity

# 2. Use higher temperature during generation
generation:
  temperature: 1.0  # Increase from 0.7
  top_p: 0.95

# 3. Add diversity bonus to reward
reward:
  diversity_bonus: 0.1
```

## Best Practices

### Do's ✅
1. **Start with high-quality SFT data** - Foundation for RLHF
2. **Train reward model carefully** - Critical for RLHF success
3. **Monitor KL divergence** - Prevents model drift
4. **Use conservative hyperparameters** - Stability over speed
5. **Evaluate safety thoroughly** - Red team before deployment
6. **Keep reference model** - Comparison and debugging
7. **Version control checkpoints** - Rollback if needed

### Don'ts ❌
1. **Don't skip SFT** - RLHF needs good initialization
2. **Don't use low-quality preference data** - Garbage in, garbage out
3. **Don't ignore KL divergence** - Will cause instabilities
4. **Don't over-optimize rewards** - Risk of reward hacking
5. **Don't skip safety evaluation** - Aligned ≠ safe
6. **Don't use aggressive learning rates** - Causes instabilities
7. **Don't deploy without human eval** - Automatic metrics insufficient

## Validation Checklist

Before deploying chat model:

- [ ] SFT training converged smoothly
- [ ] Reward model accuracy >70%
- [ ] RLHF training stable (KL divergence controlled)
- [ ] Mean reward increased over training
- [ ] Instruction following improved over SFT
- [ ] Safety evaluation passed
- [ ] Red team testing completed
- [ ] Human evaluation scores high on 3H (helpful, harmless, honest)
- [ ] No reward hacking observed
- [ ] Response diversity maintained
- [ ] Chat template properly configured
- [ ] Inference API tested

## Next Steps

1. **Deployment**: See deployment scripts in `scripts/`
2. **Monitoring**: Track production performance
3. **Iterative Improvement**: Collect user feedback, retrain periodically
4. **Continual Learning**: Update with new data, maintain capabilities

## Resources

- Anthropic RLHF Paper: Constitutional AI
- OpenAI InstructGPT Paper: RLHF methodology
- DeepSpeed-Chat: RLHF training framework
- TRL (Transformer Reinforcement Learning): PPO implementation
- HuggingFace Alignment Handbook: Comprehensive RLHF guide
