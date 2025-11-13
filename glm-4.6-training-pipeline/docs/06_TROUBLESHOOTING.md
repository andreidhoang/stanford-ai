# GLM-4.6 Troubleshooting Guide

Comprehensive troubleshooting guide for GLM-4.6 training pipeline covering common issues, diagnostics, and solutions.

## Quick Diagnostic Commands

```bash
# System health check
bash scripts/health_check.sh

# GPU status
nvidia-smi

# Training status
tail -f output/tensorboard/training.log

# Monitor resources
htop

# Check NCCL
python -c "import torch; print(torch.cuda.nccl.version())"

# DeepSpeed status
ds_report
```

## Training Issues

### Issue: Training Loss Not Decreasing

**Symptoms**:
- Loss stays flat or increases over time
- Perplexity not improving
- Model not learning

**Diagnostics**:
```bash
# Check loss curves
tensorboard --logdir output/tensorboard

# Verify data quality
python src/data/analyze_dataset.py \
    --input data/pretrain/tokenized \
    --output data_analysis.json

# Check learning rate schedule
grep "learning_rate" output/tensorboard/training.log
```

**Possible Causes & Solutions**:

**1. Learning Rate Too Low**:
```yaml
# Increase learning rate
training:
  learning_rate: 1.0e-4  # Increase from 5e-5
  warmup_steps: 2000
```

**2. Bad Data**:
```bash
# Re-process and deduplicate data
python src/data/preprocess_data.py --input data/raw --output data/processed
python src/data/deduplication.py --input data/processed --output data/clean
```

**3. Checkpoint Loading Failed**:
```bash
# Verify checkpoint integrity
python scripts/verify_checkpoint.py \
    --checkpoint output/checkpoints/checkpoint-1000

# If corrupted, rollback to previous checkpoint
```

**4. Incorrect Loss Masking**:
```python
# Check that padding tokens are properly masked
# In training config:
training:
  ignore_index: -100  # Tokenizer pad_token_id
```

---

### Issue: Training Loss Spikes

**Symptoms**:
- Sudden jumps in loss value
- Loss increases by >2.0 in single step
- Training becomes unstable

**Diagnostics**:
```bash
# Check loss history
grep "loss" output/tensorboard/training.log | tail -100

# Monitor gradients
# Enable in training config:
training:
  log_gradients: true
```

**Possible Causes & Solutions**:

**1. Learning Rate Too High**:
```yaml
# Reduce learning rate
training:
  learning_rate: 2.0e-5  # Reduce from 5e-5
  # Or add more warmup
  warmup_steps: 4000  # Increase from 2000
```

**2. Gradient Overflow**:
```yaml
# Enable gradient clipping
training:
  gradient_clipping: 0.5  # Reduce from 1.0
  mixed_precision: "bf16"  # Switch from fp16
```

**3. Bad Batch**:
```python
# Add gradient validation
training:
  skip_nan_gradients: true
  max_grad_norm: 1.0
```

**4. Optimizer State Corruption**:
```bash
# Reset optimizer state
python scripts/reset_optimizer_state.py \
    --checkpoint output/checkpoints/checkpoint-recent \
    --output output/checkpoints/checkpoint-reset
```

---

### Issue: Out of Memory (OOM)

**Symptoms**:
- CUDA OOM errors
- Training crashes
- "RuntimeError: CUDA out of memory"

**Diagnostics**:
```bash
# Check GPU memory usage
nvidia-smi

# Monitor memory during training
watch -n 1 nvidia-smi

# Profile memory
python scripts/profile_memory.py \
    --config configs/model_355b_32b_active.yaml
```

**Possible Causes & Solutions**:

**1. Batch Size Too Large**:
```yaml
# Reduce batch size, increase gradient accumulation
training:
  per_device_train_batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 2  # Increase from 1
```

**2. Sequence Length Too Long**:
```yaml
# Reduce max sequence length
model:
  max_seq_length: 4096  # Reduce from 8192
```

**3. Activation Memory**:
```yaml
# Enable activation checkpointing
model:
  use_gradient_checkpointing: true
  gradient_checkpointing_ratio: 0.5
```

**4. Insufficient Offloading**:
```json
// In DeepSpeed config
{
  "zero_optimization": {
    "offload_optimizer": {
      "device": "nvme",  // Change from "none"
      "nvme_path": "/local_nvme"
    },
    "offload_param": {
      "device": "cpu"  // Change from "none"
    }
  }
}
```

**5. Memory Fragmentation**:
```bash
# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

# Restart training with clean memory
```

---

### Issue: Slow Training Speed

**Symptoms**:
- Low tokens/second
- High time per iteration
- Poor GPU utilization (<80%)

**Diagnostics**:
```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet

# Profile training
python scripts/profile_training.py \
    --config configs/model_355b_32b_active.yaml

# Check data loading
# Add to training script:
import time
start = time.time()
batch = next(dataloader)
print(f"Data loading time: {time.time() - start:.3f}s")
```

**Possible Causes & Solutions**:

**1. Data Loading Bottleneck**:
```yaml
# Increase data workers
data:
  num_workers: 8  # Increase from 4
  prefetch_factor: 4
  persistent_workers: true
```

**2. CPU Bottleneck**:
```bash
# Pin CPU cores
export OMP_NUM_THREADS=8
taskset -c 0-95 python src/training/...

# Use better CPU
# Recommended: 2× CPU cores per GPU
```

**3. Network Bottleneck** (Distributed):
```bash
# Check NCCL bandwidth
mpirun -n 8 --hostfile hostfile \
    nccl-tests/build/all_reduce_perf -b 1G -e 8G

# Optimize NCCL settings
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple

# Enable InfiniBand
export NCCL_IB_DISABLE=0
```

**4. Small Batch Size**:
```yaml
# Increase batch size for better GPU utilization
training:
  per_device_train_batch_size: 4  # Increase from 1
  gradient_accumulation_steps: 1  # Reduce from 4
```

**5. Inefficient Model Implementation**:
```python
# Enable optimizations
model:
  use_flash_attention: true
  fused_ops: true
  compile: true  # PyTorch 2.0 compile
```

---

### Issue: Catastrophic Forgetting (Mid-Training/Fine-Tuning)

**Symptoms**:
- Domain performance improves but general performance degrades
- MMLU/GSM8K scores drop significantly
- Model loses general knowledge

**Diagnostics**:
```bash
# Compare with baseline
python src/evaluation/benchmarks.py \
    --model output/mid_training/checkpoint-1000 \
    --benchmark mmlu,gsm8k \
    --compare-baseline output/pretrain/checkpoint-final
```

**Solutions**:

**1. Mix General Data**:
```yaml
data:
  domain_data_ratio: 0.7  # Reduce from 0.8
  general_data_ratio: 0.3  # Increase from 0.2
```

**2. Lower Learning Rate**:
```yaml
training:
  learning_rate: 1.0e-6  # Reduce from 5e-6
```

**3. Enable Elastic Weight Consolidation**:
```yaml
training:
  use_ewc: true
  ewc_lambda: 10000
  ewc_checkpoint: "output/pretrain/checkpoint-final"
```

**4. Layer-wise Learning Rate Decay**:
```yaml
optimizer:
  layer_lr_decay: 0.9  # Lower layers get lower LR
```

---

### Issue: Overfitting

**Symptoms**:
- Training loss much lower than validation loss
- Validation loss increases while training loss decreases
- Poor generalization

**Diagnostics**:
```bash
# Plot loss curves
tensorboard --logdir output/tensorboard

# Check train/val gap
python scripts/analyze_overfitting.py \
    --logdir output/tensorboard
```

**Solutions**:

**1. Increase Regularization**:
```yaml
training:
  dropout: 0.2  # Increase from 0.1
  weight_decay: 0.15  # Increase from 0.1
  attention_dropout: 0.2
```

**2. Early Stopping**:
```yaml
training:
  early_stopping: true
  patience: 5
  metric: "eval_loss"
```

**3. Reduce Training Duration**:
```yaml
training:
  max_steps: 500  # Reduce from 1000
  # Or reduce epochs
  num_epochs: 2  # Reduce from 3
```

**4. Increase Training Data**:
```bash
# Add more diverse data
python src/data/augment_data.py \
    --input data/train \
    --output data/train_augmented
```

---

## Infrastructure Issues

### Issue: NCCL Timeout

**Symptoms**:
- "NCCL timeout" errors
- Training hangs during distributed operations
- Collective operations fail

**Diagnostics**:
```bash
# Test NCCL connectivity
mpirun -n 8 --hostfile hostfile \
    python -c "import torch; import torch.distributed as dist; \
    dist.init_process_group('nccl'); print('OK')"

# Check network latency
mpirun -n 2 --hostfile hostfile \
    ib_write_lat  # InfiniBand
    # or
    ping -c 100 node002  # TCP
```

**Solutions**:

**1. Increase Timeout**:
```bash
export NCCL_TIMEOUT=7200000  # 2 hours (in ms)
```

**2. Fix Network Issues**:
```bash
# Check InfiniBand status
ibstat
# Should show: State: Active

# Restart InfiniBand
sudo systemctl restart openibd

# Check firewall
sudo iptables -L
# Ensure ports 29500-29600 open
```

**3. Optimize NCCL**:
```bash
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_DEBUG=INFO  # Verbose logging
export NCCL_SOCKET_IFNAME=eth0  # Correct interface
```

---

### Issue: Node Failure

**Symptoms**:
- Training crashes when node fails
- "Connection refused" errors
- Missing ranks in distributed training

**Diagnostics**:
```bash
# Check node status
for host in $(cat hostfile | awk '{print $1}'); do
    ssh $host "echo OK" || echo "FAILED: $host"
done

# Check Slurm node status
sinfo -N -l
```

**Solutions**:

**1. Enable Elastic Training**:
```yaml
# DeepSpeed config
elasticity:
  enabled: true
  min_gpus: 1024
  max_gpus: 8192
  min_time_scale_in_ms: 30000
```

**2. Exclude Failed Nodes**:
```bash
# Slurm
scontrol update nodename=node042 state=drain reason="GPU failure"

# Manual
sed -i '/node042/d' hostfile
```

**3. Implement Fault Tolerance**:
```bash
# Save checkpoints frequently
training:
  checkpoint_interval: 100  # Every 100 steps

# Auto-restart on failure
while true; do
    sbatch submit_job.slurm && break
    sleep 60
done
```

---

### Issue: Checkpoint Corruption

**Symptoms**:
- "Unable to load checkpoint" errors
- Training fails to resume
- Checkpoint files incomplete

**Diagnostics**:
```bash
# Verify checkpoint
python scripts/verify_checkpoint.py \
    --checkpoint output/checkpoints/checkpoint-1000

# Check file integrity
find output/checkpoints/checkpoint-1000 -type f -exec md5sum {} \;
```

**Solutions**:

**1. Rollback to Previous Checkpoint**:
```bash
# Use earlier checkpoint
--load_checkpoint output/checkpoints/checkpoint-900
```

**2. Enable Checkpoint Verification**:
```yaml
checkpoint:
  verify_after_save: true
  checksum: "md5"
```

**3. Use Redundant Storage**:
```bash
# Copy to multiple locations
rsync -av output/checkpoints/checkpoint-1000 \
    /backup/checkpoints/checkpoint-1000
```

---

### Issue: Slow Checkpoint Saving

**Symptoms**:
- Checkpoint saving takes >30 minutes
- Training blocked during checkpoint save
- High disk I/O wait

**Diagnostics**:
```bash
# Monitor I/O
iostat -x 1

# Check NVMe performance
fio --name=test --directory=/local_nvme \
    --rw=write --bs=1M --size=10G
```

**Solutions**:

**1. Enable Parallel Write**:
```yaml
checkpoint:
  parallel_write:
    pipeline_stage: true
  use_node_local_storage: true
```

**2. Use Faster Storage**:
```bash
# Save to NVMe first, copy to shared later
checkpoint:
  node_local_path: "/local_nvme/checkpoints"
  shared_path: "/mnt/lustre/checkpoints"
  async_copy: true
```

**3. Reduce Checkpoint Frequency**:
```yaml
checkpoint:
  save_interval: 2000  # Increase from 1000
```

---

## Data Issues

### Issue: Poor Data Quality

**Symptoms**:
- Model learns slowly
- High validation loss
- Poor benchmark performance

**Diagnostics**:
```bash
# Analyze dataset
python src/data/analyze_dataset.py \
    --input data/pretrain/tokenized \
    --output data_quality_report.json

# Sample random examples
python src/data/sample_dataset.py \
    --input data/pretrain/tokenized \
    --num-samples 100 \
    --output samples.txt

# Check for duplicates
python src/data/deduplication.py \
    --input data/pretrain/processed \
    --output data/pretrain/deduplicated \
    --check-only true
```

**Solutions**:

**1. Re-process Data**:
```bash
# Clean and filter
python src/data/preprocess_data.py \
    --input data/raw \
    --output data/processed \
    --quality-threshold 0.8  # Increase threshold

# Deduplicate
python src/data/deduplication.py \
    --input data/processed \
    --output data/clean \
    --threshold 0.85
```

**2. Add High-Quality Data**:
```bash
# Mix with curated datasets
python src/data/mix_datasets.py \
    --inputs data/clean,data/curated \
    --ratios 0.8,0.2 \
    --output data/mixed
```

---

### Issue: Data Loading Slow

**Symptoms**:
- Low GPU utilization
- Long time between batches
- CPU bottleneck

**Diagnostics**:
```python
# Profile data loading
import time
dataloader = get_dataloader()
times = []
for i, batch in enumerate(dataloader):
    start = time.time()
    _ = next(dataloader)
    times.append(time.time() - start)
    if i > 100:
        break

print(f"Avg data loading time: {np.mean(times):.3f}s")
# Target: <0.1s
```

**Solutions**:

**1. Increase Workers**:
```yaml
data:
  num_workers: 16  # Increase from 4
  prefetch_factor: 4
  persistent_workers: true
```

**2. Pre-process and Cache**:
```bash
# Pre-tokenize all data
python src/data/tokenizer_training.py tokenize \
    --input data/processed \
    --output data/tokenized \
    --cache-dir /local_nvme/cache
```

**3. Use Faster Storage**:
```bash
# Move data to NVMe
rsync -av data/tokenized /local_nvme/data/
```

---

## Model Issues

### Issue: NaN Loss

**Symptoms**:
- Loss becomes NaN
- Training crashes
- Model outputs all zeros or infinities

**Diagnostics**:
```bash
# Check when NaN first appeared
grep "loss" output/logs/training.log | grep -i "nan"

# Enable gradient checking
training:
  detect_anomaly: true
```

**Possible Causes & Solutions**:

**1. Numerical Instability**:
```yaml
# Use bf16 instead of fp16
training:
  mixed_precision: "bf16"

# Lower learning rate
training:
  learning_rate: 1.0e-5  # Reduce from 5e-5
```

**2. Gradient Explosion**:
```yaml
# Enable gradient clipping
training:
  gradient_clipping: 0.5  # Reduce from 1.0
  max_grad_norm: 0.5
```

**3. Bad Initialization**:
```python
# Check weight initialization
model:
  init_method: "normal"
  init_std: 0.02  # Reduce if unstable
```

**4. Bad Data**:
```bash
# Find problematic samples
python scripts/find_nan_samples.py \
    --checkpoint output/checkpoints/checkpoint-before-nan \
    --data data/train
```

---

### Issue: Expert Imbalance (MoE)

**Symptoms**:
- High coefficient of variation in expert usage
- Some experts never used
- Poor MoE efficiency

**Diagnostics**:
```bash
# Check expert balance metrics
tensorboard --logdir output/tensorboard
# Look for: expert_balance_cv, expert_usage_*

# Analyze expert usage
python scripts/analyze_expert_usage.py \
    --checkpoint output/checkpoints/checkpoint-1000
```

**Solutions**:

**1. Adjust Load Balancing**:
```yaml
model:
  moe:
    load_balance_loss_coef: 0.01  # Increase from 0.001
    expert_balance_method: "sinkhorn"  # Try different method
```

**2. Increase Capacity Factor**:
```yaml
model:
  moe:
    capacity_factor: 1.5  # Increase from 1.25
```

**3. Use Auxiliary Load Balancing**:
```python
# GLM-4.6 uses loss-free balancing, but can enable aux loss for debugging
model:
  moe:
    use_auxiliary_loss: true  # Temporary for debugging
    aux_loss_coef: 0.01
```

---

## Evaluation Issues

### Issue: Benchmark Scores Lower Than Expected

**Symptoms**:
- MMLU < 85%
- GSM8K < 90%
- HumanEval < 70%

**Diagnostics**:
```bash
# Run detailed evaluation
python src/evaluation/benchmarks.py \
    --model output/checkpoints/checkpoint-final \
    --benchmark all \
    --num-samples null \
    --output eval_detailed.json

# Analyze failure modes
python scripts/analyze_failures.py \
    --eval-results eval_detailed.json
```

**Possible Causes & Solutions**:

**1. Insufficient Training**:
```bash
# Continue training
# Load checkpoint and train more steps
```

**2. Poor Data Quality**:
```bash
# Review training data quality
# Add more high-quality data
```

**3. Suboptimal Hyperparameters**:
```yaml
# Try different learning rate
training:
  learning_rate: 3.0e-5  # Adjust

# Try different batch size
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 2
```

**4. Evaluation Issues**:
```bash
# Check evaluation implementation
# Ensure proper prompt formatting
# Verify answer extraction logic
```

---

## Deployment Issues

### Issue: Slow Inference

**Symptoms**:
- High latency (>1s per request)
- Low throughput (<10 req/s)
- Poor GPU utilization during inference

**Diagnostics**:
```bash
# Profile inference
python src/evaluation/profile_performance.py \
    --model models/glm4-6-chat \
    --batch-sizes 1,4,8,16

# Check GPU usage
nvidia-smi dmon
```

**Solutions**:

**1. Use Optimized Inference Engine**:
```bash
# vLLM for high throughput
bash scripts/deploy_vllm.sh \
    --model models/glm4-6-chat \
    --scenario high-throughput

# SGLang for chat workloads
bash scripts/deploy_sglang.sh \
    --model models/glm4-6-chat
```

**2. Quantize Model**:
```bash
# AWQ 4-bit quantization (4× smaller, 3× faster)
python scripts/quantize_model.py \
    --model models/glm4-6-chat \
    --method awq \
    --bits 4 \
    --output models/glm4-6-chat-awq
```

**3. Increase Batch Size**:
```python
# vLLM config
max_num_seqs: 256  # Increase from 128
```

**4. Use Tensor Parallelism**:
```bash
# Distribute across multiple GPUs
python -m vllm.entrypoints.openai.api_server \
    --model models/glm4-6-chat \
    --tensor-parallel-size 8
```

---

## Debugging Tools

### Enable Debugging

**Training Debug Mode**:
```yaml
training:
  debug: true
  log_level: "DEBUG"
  detect_anomaly: true
  print_gradients: true
```

**Python Debugger**:
```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()
```

**Remote Debugging** (Multi-node):
```python
# debugpy for VS Code
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### Logging

**Comprehensive Logging**:
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
```

**DeepSpeed Logging**:
```bash
export DEEPSPEED_LOG_LEVEL=DEBUG
```

**NCCL Logging**:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=nccl_debug.log
```

## Getting Help

### Collecting Debug Information

```bash
# System info
python scripts/collect_debug_info.py --output debug_info.txt

# Include:
# - Hardware specs (nvidia-smi)
# - Software versions (torch, deepspeed, transformers)
# - Configuration files
# - Recent logs (last 1000 lines)
# - Error traces
# - Training metrics
```

### Community Resources

- **GitHub Issues**: https://github.com/THUDM/GLM-4
- **GLM Forum**: https://forum.thudm.ai/
- **DeepSpeed GitHub**: https://github.com/microsoft/DeepSpeed/issues
- **HuggingFace Forum**: https://discuss.huggingface.co/

### Creating Bug Reports

**Template**:
```markdown
**Environment**:
- GPU: H800 × 8
- CUDA: 12.1
- PyTorch: 2.1.0
- DeepSpeed: 0.12.0
- GLM-4.6 version: main branch

**Problem Description**:
[Clear description of the issue]

**Steps to Reproduce**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior**:
[What should happen]

**Actual Behavior**:
[What actually happens]

**Error Messages**:
```
[Paste error traces]
```

**Configurations**:
[Attach config files]

**Additional Context**:
[Any other relevant information]
```

## Prevention Checklist

**Before Training**:
- [ ] Test on single GPU first
- [ ] Validate data quality
- [ ] Run small-scale test (1 node)
- [ ] Verify checkpoint saving/loading
- [ ] Test data pipeline performance
- [ ] Check monitoring setup
- [ ] Verify NCCL connectivity
- [ ] Test node failure recovery
- [ ] Review hyperparameters
- [ ] Set up alerts

**During Training**:
- [ ] Monitor GPU utilization
- [ ] Watch loss curves
- [ ] Check expert balance (MoE)
- [ ] Monitor throughput
- [ ] Validate checkpoints
- [ ] Review sample outputs
- [ ] Check system resources
- [ ] Monitor network bandwidth

**After Training**:
- [ ] Evaluate on benchmarks
- [ ] Compare with baseline
- [ ] Test inference speed
- [ ] Validate model quality
- [ ] Document learnings
- [ ] Archive artifacts

## Common Error Messages

### "CUDA out of memory"
→ See [Out of Memory](#issue-out-of-memory-oom) section

### "NCCL timeout"
→ See [NCCL Timeout](#issue-nccl-timeout) section

### "Loss is NaN"
→ See [NaN Loss](#issue-nan-loss) section

### "Checkpoint loading failed"
→ See [Checkpoint Corruption](#issue-checkpoint-corruption) section

### "Training stalled"
→ See [Slow Training Speed](#issue-slow-training-speed) section

### "Expert imbalance"
→ See [Expert Imbalance](#issue-expert-imbalance-moe) section

### "Connection refused" (distributed)
→ Check network connectivity, firewall, SSH keys

### "RuntimeError: Expected tensor for argument"
→ Check input shapes, data types, device placement

### "ImportError: No module named"
→ Install missing dependencies: `pip install -r requirements.txt`

## Quick Fixes

**Reset Everything**:
```bash
# Nuclear option - start fresh
rm -rf output/
rm -rf ~/.cache/huggingface
pip install --upgrade --force-reinstall torch transformers deepspeed
```

**Clear GPU Memory**:
```bash
# Kill all Python processes
pkill -9 python

# Reset GPUs
sudo nvidia-smi --gpu-reset
```

**Restart Distributed Training**:
```bash
# Clean up stale processes
pdsh -w ^hostfile "pkill -9 python"

# Clear temporary files
pdsh -w ^hostfile "rm -rf /tmp/torch_*"

# Restart
bash scripts/launch_training.sh
```

## Summary

- Always test at small scale first
- Monitor everything during training
- Save checkpoints frequently
- Keep backups of working configurations
- Document issues and solutions
- Use community resources when stuck

For additional help, see individual guide sections:
- [Pre-training Guide](01_PRETRAINING.md)
- [Data Pipeline](02_DATA_PIPELINE.md)
- [Mid-training Guide](03_MID_TRAINING.md)
- [Post-training Guide](04_POST_TRAINING.md)
- [Infrastructure Guide](05_INFRASTRUCTURE.md)
