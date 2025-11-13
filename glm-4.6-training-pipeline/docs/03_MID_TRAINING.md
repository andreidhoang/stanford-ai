# GLM-4.6 Mid-Training Guide

Complete guide for domain-specific continued pre-training and mid-training adaptation.

## Overview

Mid-training (continued pre-training on domain-specific data) bridges the gap between base pre-training and task-specific fine-tuning. This stage adapts the pre-trained GLM-4.6 model to specialized domains while preserving general capabilities.

## When to Use Mid-Training

### Use Cases
- **Domain Adaptation**: Finance, medicine, legal, scientific domains
- **Language Adaptation**: Adding new languages or improving existing ones
- **Knowledge Updates**: Incorporating recent information post-cutoff
- **Style Adaptation**: Technical writing, creative content, code generation

### Benefits
- Preserves general knowledge from pre-training
- Faster than training from scratch
- More data-efficient than pure fine-tuning
- Better performance on domain-specific tasks

## Data Preparation

### Domain-Specific Data Collection

```bash
# Data collection structure
data/mid_training/
├── domain_name/
│   ├── raw/              # Original collected data
│   ├── processed/        # Cleaned and formatted
│   ├── tokenized/        # Tokenized datasets
│   └── metadata.json     # Dataset statistics
```

### Data Quality Requirements

**Volume Requirements**:
- Minimum: 10B tokens for noticeable adaptation
- Recommended: 50-100B tokens for strong domain adaptation
- Optimal: 200B+ tokens for comprehensive coverage

**Quality Criteria**:
1. **Relevance**: >90% domain-relevant content
2. **Diversity**: Multiple sources, perspectives, styles
3. **Accuracy**: Verified factual correctness
4. **Recency**: Up-to-date information where applicable
5. **Cleanliness**: Minimal noise, proper formatting

### Data Processing Pipeline

```bash
# 1. Clean raw data
python src/data/preprocess_data.py \
    --input data/mid_training/domain/raw \
    --output data/mid_training/domain/processed \
    --domain-specific \
    --quality-threshold 0.9

# 2. Deduplicate
python src/data/deduplication.py \
    --input data/mid_training/domain/processed \
    --output data/mid_training/domain/deduplicated \
    --threshold 0.85

# 3. Tokenize
python src/data/tokenizer_training.py tokenize \
    --input data/mid_training/domain/deduplicated \
    --output data/mid_training/domain/tokenized \
    --tokenizer-path models/glm4_tokenizer
```

## Training Configuration

### Learning Rate Strategy

**Recommended Settings**:
```yaml
# Mid-training uses LOWER learning rates than pre-training
learning_rate: 5.0e-6  # 10× lower than pre-training (5e-5)
min_learning_rate: 5.0e-7
warmup_ratio: 0.05     # Shorter warmup
lr_scheduler: "cosine_with_restarts"  # Allow recovery
```

**Rationale**:
- Lower LR prevents catastrophic forgetting
- Gradual adaptation preserves general capabilities
- Restarts help recover from local minima

### Training Duration

**Token-Based Scheduling**:
```python
# Calculate training steps
domain_tokens = 50e9  # 50B domain tokens
batch_size = 8192     # Global batch size
seq_length = 8192     # Sequence length

tokens_per_step = batch_size * seq_length
total_steps = domain_tokens / tokens_per_step

# Example: 50B / (8192 * 8192) ≈ 750 steps
```

**Epoch-Based (Small Domains)**:
```yaml
# For smaller datasets (<50B tokens)
num_epochs: 3-5
max_steps: null  # Train for full epochs
```

### Hyperparameters

**Mid-Training Configuration** (`configs/training_mid_training.yaml`):
```yaml
training:
  # Learning rate (10× lower than pre-training)
  learning_rate: 5.0e-6
  min_learning_rate: 5.0e-7
  warmup_ratio: 0.05
  lr_scheduler: "cosine_with_restarts"

  # Training duration
  max_steps: 1000      # Adjust based on domain size
  save_steps: 100
  eval_steps: 50

  # Batch configuration
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  global_batch_size: 8192

  # Regularization (prevent overfitting)
  weight_decay: 0.1
  dropout: 0.1         # Slightly higher than pre-training
  attention_dropout: 0.1

  # Gradient management
  gradient_clipping: 1.0
  mixed_precision: "bf16"

  # Loss configuration
  loss_scaling: "dynamic"
  clip_grad_norm: 1.0

# Data configuration
data:
  train_data_path: "data/mid_training/domain/tokenized"
  validation_split: 0.01
  max_seq_length: 8192

# Checkpoint management
checkpoint:
  load_checkpoint: "output/pretrain/checkpoint-final"  # Start from pre-trained
  save_total_limit: 3
  checkpoint_interval: 100
```

## Training Execution

### Single-Node Training

```bash
# Small domain (<10B tokens)
python src/training/pretraining/pretrainer.py \
    --model_config_path configs/model_355b_32b_active.yaml \
    --training_config_path configs/training_mid_training.yaml \
    --deepspeed_config configs/deepspeed_stage3.json \
    --train_data_path data/mid_training/domain/tokenized \
    --load_checkpoint output/pretrain/checkpoint-final \
    --output_dir output/mid_training/domain \
    --tensorboard_dir output/mid_training/domain/tensorboard \
    --checkpoint_dir output/mid_training/domain/checkpoints
```

### Multi-Node Training

```bash
# Large domain (>50B tokens)
# 1. Setup cluster
bash scripts/setup_cluster.sh \
    --type slurm \
    --nodes 16 \
    --gpus-per-node 8

# 2. Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=glm46-midtrain
#SBATCH --nodes=16
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

source setup_env.sh

deepspeed --hostfile=hostfile \
    src/training/pretraining/pretrainer.py \
    --model_config_path configs/model_355b_32b_active.yaml \
    --training_config_path configs/training_mid_training.yaml \
    --deepspeed_config configs/deepspeed_stage3.json \
    --train_data_path data/mid_training/domain/tokenized \
    --load_checkpoint output/pretrain/checkpoint-final \
    --output_dir output/mid_training/domain
EOF
```

## Monitoring and Evaluation

### Key Metrics to Track

**1. Loss Metrics**:
```python
# Training loss should:
# - Start at pre-training final loss (e.g., 2.5)
# - Gradually decrease to ~2.0-2.3 range
# - NOT drop below pre-training loss significantly (indicates overfitting)

# Warning signs:
# - Sudden loss spikes → learning rate too high
# - Loss plateau → learning rate too low
# - Train loss << val loss → overfitting
```

**2. Perplexity**:
```python
# Monitor perplexity on:
# - Domain-specific validation set (should improve)
# - General validation set (should NOT degrade significantly)

# Target: <10% perplexity increase on general data
```

**3. Domain Performance**:
```bash
# Periodic evaluation on domain tasks
python src/evaluation/benchmarks.py \
    --model output/mid_training/domain/checkpoint-500 \
    --benchmark domain_specific \
    --num-samples 1000
```

### Monitoring Dashboard

```bash
# Start monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana: http://localhost:3000
# Username: admin, Password: admin

# Add custom domain metrics:
# - Domain perplexity
# - General capability retention
# - Training loss curves
```

## Catastrophic Forgetting Prevention

### Problem
Mid-training can cause the model to "forget" general knowledge while adapting to domain-specific data.

### Solutions

**1. Mixed Data Training**:
```yaml
# Mix 80% domain data + 20% general data
data:
  domain_data_path: "data/mid_training/domain/tokenized"
  general_data_path: "data/pretrain/tokenized"
  domain_ratio: 0.8
  general_ratio: 0.2
```

**2. Elastic Weight Consolidation (EWC)**:
```python
# Add EWC loss to preserve important weights
training:
  use_ewc: true
  ewc_lambda: 5000  # Importance weight
  ewc_checkpoint: "output/pretrain/checkpoint-final"
```

**3. Progressive Learning Rates**:
```yaml
# Different LR for different layers
optimizer:
  layer_lr_decay: 0.9  # Decrease LR for earlier layers
  # Layer 0 (embeddings): lr × 0.9^60 ≈ lr × 0.002
  # Layer 60 (output): lr × 1.0
```

**4. Regular General Evaluation**:
```bash
# Every 100 steps, evaluate on general benchmarks
python src/evaluation/benchmarks.py \
    --model output/mid_training/domain/checkpoint-100 \
    --benchmark mmlu \
    --compare-baseline output/pretrain/checkpoint-final
```

## Domain-Specific Considerations

### Code Domain

**Data Sources**:
- GitHub repositories (clean, well-documented)
- Technical documentation
- API references
- Programming tutorials

**Special Handling**:
```yaml
tokenizer:
  add_special_tokens:
    - "<code>"
    - "</code>"
    - "<comment>"
    - "</comment>"

training:
  max_seq_length: 16384  # Longer for code files
  learning_rate: 3.0e-6  # Lower for code (more structured)
```

### Medical Domain

**Data Sources**:
- Medical journals (PubMed)
- Clinical guidelines
- Medical textbooks
- Case studies (anonymized)

**Compliance**:
```yaml
data_processing:
  anonymization: true
  phi_removal: true  # Protected Health Information
  compliance_check: "HIPAA"

training:
  additional_safety_checks: true
  human_review_required: true
```

### Financial Domain

**Data Sources**:
- Financial reports (10-K, 10-Q)
- Market analysis
- Economic papers
- Regulatory documents

**Special Considerations**:
```yaml
data_processing:
  temporal_cutoff: "2024-01-01"  # Clear data cutoff
  fact_verification: true

training:
  validation_strategy: "time_series"  # Temporal validation split
```

## Checkpoint Management

### Checkpoint Selection

**Evaluation Criteria**:
1. **Domain Performance**: Best on domain-specific tasks
2. **General Retention**: Minimal degradation on general benchmarks
3. **Loss Metrics**: Lowest validation loss
4. **Perplexity**: Best domain perplexity with acceptable general perplexity

**Selection Process**:
```bash
# Evaluate all checkpoints
for ckpt in output/mid_training/domain/checkpoints/checkpoint-*; do
    python src/evaluation/benchmarks.py \
        --model $ckpt \
        --benchmark domain_specific \
        --output eval_results/$(basename $ckpt).json
done

# Compare results
python scripts/compare_checkpoints.py \
    --eval-dir eval_results \
    --baseline output/pretrain/checkpoint-final \
    --output best_checkpoint.txt
```

### Checkpoint Conversion

```bash
# Convert to HuggingFace format
python scripts/convert_to_hf.py \
    --checkpoint output/mid_training/domain/checkpoint-best \
    --output models/glm4-6-domain-adapted \
    --tokenizer models/glm4_tokenizer
```

## Best Practices

### Do's ✅
1. **Start from pre-trained checkpoint** - Never train from scratch
2. **Use lower learning rates** - 5-10× lower than pre-training
3. **Monitor general capabilities** - Evaluate on MMLU, GSM8K regularly
4. **Mix general data** - Include 10-20% general data in training
5. **Use gradual learning rate schedule** - Cosine with restarts
6. **Save frequent checkpoints** - Every 50-100 steps
7. **Validate on multiple datasets** - Both domain and general

### Don'ts ❌
1. **Don't use pre-training learning rates** - Will cause catastrophic forgetting
2. **Don't train for too long** - 1-3 epochs maximum on domain data
3. **Don't ignore general performance** - Balance domain and general capabilities
4. **Don't skip data quality checks** - Poor data = poor adaptation
5. **Don't overfit to small datasets** - Use regularization, early stopping
6. **Don't forget to evaluate** - Regular evaluation prevents overfitting

## Troubleshooting

### Loss Not Decreasing

**Symptoms**: Loss stays flat or increases
**Causes**:
- Learning rate too low
- Data quality issues
- Checkpoint loading failed

**Solutions**:
```bash
# 1. Increase learning rate
sed -i 's/learning_rate: 5.0e-6/learning_rate: 1.0e-5/g' configs/training_mid_training.yaml

# 2. Check data quality
python src/data/analyze_dataset.py \
    --input data/mid_training/domain/tokenized \
    --output data_quality_report.json

# 3. Verify checkpoint loading
python scripts/verify_checkpoint.py \
    --checkpoint output/pretrain/checkpoint-final
```

### Catastrophic Forgetting

**Symptoms**: Domain performance improves but general performance degrades significantly
**Solutions**:
```yaml
# Increase general data ratio
data:
  domain_ratio: 0.7  # Reduce from 0.8
  general_ratio: 0.3  # Increase from 0.2

# Lower learning rate further
training:
  learning_rate: 3.0e-6  # Reduce from 5.0e-6

# Enable EWC
training:
  use_ewc: true
  ewc_lambda: 10000
```

### Overfitting

**Symptoms**: Training loss much lower than validation loss
**Solutions**:
```yaml
# Increase regularization
training:
  dropout: 0.15      # Increase from 0.1
  weight_decay: 0.15  # Increase from 0.1

# Reduce training duration
training:
  max_steps: 500  # Reduce from 1000
  early_stopping: true
  patience: 3

# Mix more general data
data:
  general_ratio: 0.3  # Increase from 0.2
```

## Validation Checklist

Before deploying mid-trained model:

- [ ] Domain performance improved over base model
- [ ] General capabilities retained (MMLU, GSM8K within 5% of baseline)
- [ ] Training loss converged smoothly
- [ ] Validation loss decreased without overfitting
- [ ] Multiple checkpoints evaluated and best selected
- [ ] Model tested on diverse domain-specific examples
- [ ] Model tested on general knowledge questions
- [ ] Checkpoint converted to deployment format
- [ ] Documentation updated with domain-specific capabilities

## Next Steps

After successful mid-training:
1. **Post-Training (SFT/RLHF)**: See `04_POST_TRAINING.md`
2. **Evaluation**: Run comprehensive benchmarks
3. **Deployment**: Deploy adapted model for inference
4. **Monitoring**: Track performance in production

## Resources

- GLM-4 Technical Report: Architecture details
- DeepSpeed Documentation: Distributed training optimization
- Catastrophic Forgetting Papers: EWC, LwF, Progressive Networks
- Domain-Specific Datasets: HuggingFace Datasets, Papers with Code
