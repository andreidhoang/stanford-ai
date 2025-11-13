# GLM-4.6 Infrastructure Guide

Complete guide for cluster setup, distributed training, and infrastructure management.

## Overview

Training GLM-4.6 (355B parameters, 32B active) requires:
- **Multi-node cluster**: 128-1024 nodes
- **High-performance GPUs**: H800, A100, H100
- **High-speed interconnect**: InfiniBand or RoCE
- **Fast storage**: NVMe SSDs for offloading
- **Robust monitoring**: Prometheus + Grafana

## Hardware Requirements

### GPU Requirements

**Recommended GPUs**:
| GPU | Memory | Bandwidth | Performance | Cost |
|-----|--------|-----------|-------------|------|
| H800 | 80GB | 3.35TB/s | 1.98 PFLOPS | $$$$$ |
| A100 | 80GB | 2.0TB/s | 1.25 PFLOPS | $$$$ |
| H100 | 80GB | 3.35TB/s | 2.0 PFLOPS | $$$$$$ |

**Minimum Configuration**:
- **Pre-training**: 128 nodes × 8 GPUs = 1,024 GPUs (8,192 H800 GPUs for full training)
- **Fine-tuning**: 16 nodes × 8 GPUs = 128 GPUs
- **Inference**: 1-8 GPUs (depending on quantization)

### Network Requirements

**InfiniBand** (Recommended):
- **Bandwidth**: 200-400 Gb/s per port
- **Latency**: <2 μs
- **Topology**: Fat-tree or dragonfly
- **NICs**: ConnectX-6/7 or newer

**RoCE** (Alternative):
- **Bandwidth**: 100-200 Gb/s per port
- **Latency**: <5 μs
- **Requirements**: Lossless Ethernet, PFC, ECN

### Storage Requirements

**NVMe SSDs** (For ZeRO offloading):
- **Capacity**: 2TB+ per node
- **Bandwidth**: >7 GB/s sequential read/write
- **IOPS**: >1M random read/write
- **Endurance**: Enterprise-grade (>10 DWPD)

**Shared Storage** (For datasets and checkpoints):
- **Type**: Parallel file system (Lustre, GPFS, BeeGFS)
- **Capacity**: 500TB+ for full training
- **Bandwidth**: >100 GB/s aggregate
- **IOPS**: >1M aggregate

## Cluster Setup

### Prerequisites

**Operating System**:
```bash
# Ubuntu 22.04 LTS (Recommended)
lsb_release -a
# Description: Ubuntu 22.04.3 LTS

# RHEL/CentOS 8+ (Alternative)
cat /etc/redhat-release
```

**CUDA Installation**:
```bash
# Install CUDA 12.1+
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Verify installation
nvidia-smi
nvcc --version

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**NCCL Installation**:
```bash
# Install NCCL 2.18+
sudo apt-get install libnccl2=2.18.3-1+cuda12.1 libnccl-dev=2.18.3-1+cuda12.1

# For InfiniBand
sudo apt-get install libnccl-plugin-infiniband

# Verify
python -c "import torch; print(torch.cuda.nccl.version())"
```

### Automated Cluster Setup

```bash
# Run setup script
bash scripts/setup_cluster.sh \
    --type slurm \
    --nodes 128 \
    --gpus-per-node 8 \
    --master-addr 10.0.0.1 \
    --master-port 29500 \
    --nvme-path /local_nvme

# This will:
# 1. Check prerequisites (CUDA, NCCL, Python)
# 2. Setup NVMe offloading directories
# 3. Configure high-speed networking (InfiniBand/TCP)
# 4. Generate hostfile for distributed training
# 5. Setup SSH keys for passwordless access
# 6. Test connectivity and GPU accessibility
# 7. Create training launch scripts
# 8. Generate Slurm job submission script
```

### Manual Cluster Setup

#### 1. Network Configuration

**InfiniBand Setup**:
```bash
# Install InfiniBand drivers
sudo apt-get install infiniband-diags perftest

# Check InfiniBand status
ibstat
# Should show: State: Active, Physical state: LinkUp

# Test bandwidth
ib_write_bw -a -d mlx5_0

# Configure NCCL for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$(ibstat -l | head -1)
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
```

**TCP/IP Fallback**:
```bash
# If InfiniBand unavailable
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface

# For multi-NIC systems
export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3
```

#### 2. NVMe Offloading Setup

```bash
# Create NVMe directory
sudo mkdir -p /local_nvme
sudo chmod 777 /local_nvme

# Verify NVMe performance
sudo fio --name=sequential_write \
    --directory=/local_nvme \
    --rw=write \
    --bs=1M \
    --size=10G \
    --numjobs=1 \
    --runtime=60 \
    --time_based \
    --end_fsync=1

# Should see >7 GB/s bandwidth

# Configure DeepSpeed AIO
pip install py-libaio
export DEEPSPEED_NVME_PATH=/local_nvme
```

#### 3. Shared Storage Setup

**Lustre** (Recommended for large clusters):
```bash
# Mount Lustre filesystem
sudo mkdir -p /mnt/lustre
sudo mount -t lustre mgs_ip@tcp:/fsname /mnt/lustre

# Verify performance
sudo lfs df -h /mnt/lustre

# Stripe configuration for large files
lfs setstripe -c -1 -S 4M /mnt/lustre/glm4-training
```

**NFS** (Small clusters):
```bash
# Mount NFS
sudo mkdir -p /mnt/nfs
sudo mount -t nfs -o vers=4.2,rsize=1048576,wsize=1048576 \
    nfs_server:/export /mnt/nfs
```

#### 4. Hostfile Creation

```bash
# Format: hostname slots=num_gpus
cat > hostfile << EOF
node001 slots=8
node002 slots=8
node003 slots=8
...
node128 slots=8
EOF

# Verify connectivity
for host in $(awk '{print $1}' hostfile); do
    ssh $host "hostname && nvidia-smi -L" || echo "FAILED: $host"
done
```

#### 5. Environment Setup

```bash
# Create environment script
cat > setup_env.sh << 'EOF'
#!/bin/bash

# CUDA
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# NCCL
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600000

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# DeepSpeed
export DEEPSPEED_NVME_PATH=/local_nvme

# Master node
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500

# HuggingFace
export TRANSFORMERS_CACHE=/mnt/lustre/cache/huggingface
export HF_HOME=/mnt/lustre/cache/huggingface

# Tokenizers
export TOKENIZERS_PARALLELISM=false

echo "Environment configured for GLM-4.6 training"
EOF

chmod +x setup_env.sh
```

## Job Scheduling

### Slurm Configuration

**Job Script** (`submit_job.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=glm46-pretrain
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --exclusive

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Total GPUs: $((SLURM_NNODES * 8))"

# Setup environment
source setup_env.sh

# Generate hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile
while IFS= read -r host; do
    echo "$host slots=8"
done < hostfile > hostfile.tmp
mv hostfile.tmp hostfile

# Get master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch training
deepspeed --hostfile=hostfile \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/pretraining/pretrainer.py \
    --model_config_path=configs/model_355b_32b_active.yaml \
    --deepspeed_config=configs/deepspeed_stage3.json \
    --output_dir=/mnt/lustre/glm4-training/output \
    --tensorboard_dir=/mnt/lustre/glm4-training/tensorboard \
    --checkpoint_dir=/mnt/lustre/glm4-training/checkpoints

echo "Job finished at: $(date)"
```

**Submit Job**:
```bash
# Submit
sbatch submit_job.slurm

# Monitor
squeue -u $USER

# Check logs
tail -f logs/slurm-JOBID.out

# Cancel if needed
scancel JOBID
```

### PBS/Torque

**Job Script** (`submit_job.pbs`):
```bash
#!/bin/bash
#PBS -N glm46-pretrain
#PBS -l nodes=128:ppn=8:gpus=8
#PBS -l walltime=168:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o logs/pbs-$PBS_JOBID.log

cd $PBS_O_WORKDIR

# Setup environment
source setup_env.sh

# Generate hostfile
cat $PBS_NODEFILE | uniq | awk '{print $1 " slots=8"}' > hostfile

# Get master node
export MASTER_ADDR=$(head -n 1 hostfile | awk '{print $1}')
export MASTER_PORT=29500

# Launch training
deepspeed --hostfile=hostfile \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/pretraining/pretrainer.py \
    --model_config_path=configs/model_355b_32b_active.yaml \
    --deepspeed_config=configs/deepspeed_stage3.json \
    --output_dir=/mnt/lustre/glm4-training/output
```

**Submit Job**:
```bash
qsub submit_job.pbs
qstat -u $USER
```

## Distributed Training

### DeepSpeed Configuration

**ZeRO Stage 3** (`configs/deepspeed_stage3.json`):
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },

  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "warmup_min_lr": 0.0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },

  "fp16": {"enabled": false},
  "bf16": {"enabled": true},

  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000,
    "stage3_max_live_parameters": 3000000000,
    "stage3_max_reuse_distance": 3000000000,
    "stage3_gather_16bit_weights_on_model_save": true,

    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },

    "offload_param": {
      "device": "none"
    }
  },

  "gradient_clipping": 1.0,

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "synchronize_checkpoint_boundary": true
  },

  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  },

  "steps_per_print": 10,
  "tensorboard": {
    "enabled": true,
    "output_path": "logs/tensorboard",
    "job_name": "glm46_training"
  }
}
```

### Three-Dimensional Parallelism

**Parallelism Strategy**:
```yaml
# For 8,192 H800 GPUs (128 nodes × 8 GPUs × 8 nodes)
parallelism:
  tensor_parallel: 8      # Within node
  pipeline_parallel: 16   # Across nodes
  expert_parallel: 32     # For MoE experts
  data_parallel: 2        # Remaining dimension

# Total: 8 × 16 × 32 × 2 = 8,192 GPUs
```

**Configuration**:
```yaml
# In model config
model:
  tensor_parallel_size: 8
  pipeline_parallel_size: 16
  expert_parallel_size: 32

# Automatically computed
# data_parallel_size = total_gpus / (tp * pp * ep)
#                    = 8192 / (8 * 16 * 32) = 2
```

**Launch Command**:
```bash
deepspeed --hostfile=hostfile \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --num_gpus=8 \
    --num_nodes=128 \
    src/training/pretraining/pretrainer.py \
    --model_config_path=configs/model_355b_32b_active.yaml \
    --deepspeed_config=configs/deepspeed_stage3.json
```

## Monitoring and Observability

### Prometheus + Grafana Setup

```bash
# Setup monitoring stack
bash scripts/setup_monitoring.sh \
    --prometheus-port 9090 \
    --grafana-port 3000 \
    --start

# Access Grafana
# http://localhost:3000
# Username: admin, Password: admin
```

### Key Metrics to Monitor

**Training Metrics**:
- `training_loss`: Current training loss
- `training_learning_rate`: Current learning rate
- `training_tokens_per_second`: Throughput
- `training_steps_total`: Total steps completed
- `expert_balance_cv`: Expert utilization balance

**GPU Metrics** (NVIDIA DCGM):
- `nvidia_gpu_duty_cycle`: GPU utilization %
- `nvidia_gpu_memory_used_bytes`: GPU memory usage
- `nvidia_gpu_temperature`: GPU temperature
- `nvidia_gpu_power_usage`: Power consumption

**System Metrics** (Node Exporter):
- `node_cpu_seconds_total`: CPU usage
- `node_memory_MemAvailable_bytes`: Available memory
- `node_disk_io_time_seconds_total`: Disk I/O time
- `node_network_receive_bytes_total`: Network receive

**NCCL Metrics**:
- `nccl_allreduce_time_ms`: All-reduce communication time
- `nccl_bandwidth_gbps`: NCCL bandwidth
- `nccl_busbw_gbps`: NCCL bus bandwidth

### Alerting Rules

**Critical Alerts** (`monitoring/alerts.yml`):
```yaml
groups:
  - name: critical_alerts
    rules:
      - alert: TrainingStalled
        expr: rate(training_steps_total[5m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Training has stalled"

      - alert: GPUMemoryExhausted
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.98
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory exhausted"

      - alert: HighGPUTemperature
        expr: nvidia_gpu_temperature > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature too high"

      - alert: LossSpike
        expr: abs(delta(training_loss[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Training loss spike detected"
```

## Checkpoint Management

### Checkpoint Storage

**Distributed Checkpointing**:
```yaml
checkpoint:
  # Use node-local storage for speed
  use_node_local_storage: true
  node_local_path: "/local_nvme/checkpoints"

  # Parallel write for faster saving
  parallel_write:
    pipeline_stage: true

  # Checkpoint frequency
  save_interval: 1000  # Every 1000 steps

  # Keep only recent checkpoints
  save_total_limit: 3
```

**Checkpoint Aggregation**:
```bash
# After training, consolidate node-local checkpoints
python scripts/consolidate_checkpoints.py \
    --node-local-dir /local_nvme/checkpoints/step-10000 \
    --output-dir /mnt/lustre/checkpoints/step-10000 \
    --hostfile hostfile

# This will:
# 1. Collect checkpoint shards from all nodes
# 2. Verify integrity
# 3. Consolidate to shared storage
# 4. Clean up node-local copies
```

### Checkpoint Recovery

**Resume Training**:
```bash
# Resume from checkpoint
deepspeed --hostfile=hostfile \
    src/training/pretraining/pretrainer.py \
    --model_config_path=configs/model_355b_32b_active.yaml \
    --deepspeed_config=configs/deepspeed_stage3.json \
    --load_checkpoint=/mnt/lustre/checkpoints/step-10000 \
    --output_dir=/mnt/lustre/glm4-training/output

# Training will resume from step 10001
```

**Checkpoint Conversion**:
```bash
# Convert DeepSpeed checkpoint to HuggingFace
python scripts/convert_to_hf.py \
    --checkpoint /mnt/lustre/checkpoints/step-final \
    --output models/glm4-6-hf \
    --tokenizer models/glm4_tokenizer

# Now can use with transformers library
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("models/glm4-6-hf")
```

## Performance Optimization

### NCCL Tuning

**InfiniBand Optimization**:
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # All IB adapters
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=160
export NCCL_IB_SL=0
```

**TCP/IP Optimization**:
```bash
export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3  # All NICs
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4
```

**General NCCL Tuning**:
```bash
export NCCL_DEBUG=WARN           # Logging level
export NCCL_P2P_DISABLE=0        # Enable P2P
export NCCL_P2P_LEVEL=NVL        # NVLink for P2P
export NCCL_SHM_DISABLE=0        # Enable shared memory
export NCCL_TREE_THRESHOLD=0     # Always use tree algorithm
export NCCL_ALGO=Tree            # Communication algorithm
export NCCL_PROTO=Simple         # Protocol (Simple/LL/LL128)
```

### PyTorch Optimizations

```bash
# Memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8

# Tensor cores
export NVIDIA_TF32_OVERRIDE=1

# CUDNN
export CUDNN_BENCHMARK=1

# OpenMP
export OMP_NUM_THREADS=8  # Match CPU cores per GPU
```

### Profiling

**DeepSpeed Profiler**:
```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 10,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true
  }
}
```

**PyTorch Profiler**:
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Training step
    outputs = model(inputs)
    loss = outputs.loss
    loss.backward()

# Export trace
prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

**NCCL Tests**:
```bash
# All-reduce test
mpirun -n 8 --hostfile hostfile \
    /usr/local/bin/nccl-tests/build/all_reduce_perf \
    -b 8 -e 8G -f 2 -g 1

# Expected: >300 GB/s for H800 with InfiniBand
```

## Troubleshooting

### Common Issues

**1. NCCL Timeout**:
```bash
# Symptoms: "NCCL timeout" errors
# Solution: Increase timeout
export NCCL_TIMEOUT=7200000  # 2 hours

# Check network connectivity
mpirun -n 128 --hostfile hostfile hostname
```

**2. Out of Memory**:
```bash
# Symptoms: CUDA OOM errors
# Solutions:
# - Reduce batch size
# - Increase gradient accumulation
# - Enable CPU/NVMe offloading
# - Reduce sequence length
```

**3. Slow Training**:
```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet -i 0

# Should see >90% GPU utilization
# If low:
# - Check data loading (increase num_workers)
# - Check CPU bottlenecks (use profiler)
# - Check network bandwidth (NCCL tests)
```

**4. Loss Spikes**:
```bash
# Symptoms: Training loss suddenly increases
# Causes:
# - Learning rate too high
# - Gradient overflow
# - Bad batch

# Solutions:
# - Enable gradient clipping
# - Lower learning rate
# - Use bf16 instead of fp16
# - Check data quality
```

**5. Node Failures**:
```bash
# Symptoms: Training crashes when node fails
# Solutions:
# - Enable elastic training (DeepSpeed)
# - Save checkpoints frequently
# - Use fault-tolerant launcher
# - Monitor node health
```

## Best Practices

### Do's ✅
1. **Use InfiniBand** for >64 GPU clusters
2. **Enable NVMe offloading** for large models
3. **Monitor everything** with Prometheus + Grafana
4. **Save checkpoints frequently** (every 1000 steps)
5. **Test at small scale first** before full cluster
6. **Use shared filesystem** (Lustre) for checkpoints
7. **Tune NCCL settings** for your network
8. **Profile regularly** to find bottlenecks
9. **Validate data quality** before training
10. **Have rollback plan** for failed experiments

### Don'ts ❌
1. **Don't skip connectivity tests** before training
2. **Don't use NFS** for training data (too slow)
3. **Don't ignore GPU temperature** warnings
4. **Don't run without monitoring** (blind flying)
5. **Don't use fp16** on large models (use bf16)
6. **Don't save checkpoints to node-local** only
7. **Don't skip NCCL tests** before scaling up
8. **Don't ignore loss spikes** (investigate immediately)
9. **Don't run at 100% capacity** (leave headroom)
10. **Don't deploy without testing** recovery procedures

## Validation Checklist

Before starting large-scale training:

- [ ] All nodes accessible via SSH
- [ ] GPUs detected on all nodes (`nvidia-smi`)
- [ ] InfiniBand active (`ibstat`)
- [ ] NCCL tests passed (>300 GB/s)
- [ ] NVMe performance verified (>7 GB/s)
- [ ] Shared storage mounted and tested
- [ ] Environment variables set correctly
- [ ] Hostfile generated and validated
- [ ] Monitoring stack running
- [ ] Checkpoint directory writable
- [ ] Data accessible from all nodes
- [ ] Small-scale test job completed successfully
- [ ] Recovery procedure tested

## Next Steps

1. **Start Training**: Launch with `sbatch submit_job.slurm`
2. **Monitor Progress**: Check Grafana dashboards
3. **Evaluate Checkpoints**: Periodic evaluation on validation set
4. **Iterate**: Adjust hyperparameters based on results

## Resources

- DeepSpeed Documentation: https://www.deepspeed.ai/
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- Slurm Documentation: https://slurm.schedmd.com/
- InfiniBand Documentation: https://www.mellanox.com/
- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
