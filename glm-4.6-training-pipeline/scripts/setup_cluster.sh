#!/bin/bash
# GLM-4.6 Cluster Setup Script
#
# Sets up multi-node distributed training environment
# Supports Slurm, MPI, and manual cluster configuration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Configuration
CLUSTER_TYPE="${CLUSTER_TYPE:-slurm}"  # slurm, mpi, manual
NUM_NODES="${NUM_NODES:-128}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
NVME_PATH="${NVME_PATH:-/local_nvme}"

echo "================================================"
echo "    GLM-4.6 Cluster Setup"
echo "================================================"
echo ""
print_info "Cluster type: $CLUSTER_TYPE"
print_info "Nodes: $NUM_NODES"
print_info "GPUs per node: $GPUS_PER_NODE"
print_info "Total GPUs: $((NUM_NODES * GPUS_PER_NODE))"
echo ""

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."

    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA required."
        exit 1
    fi

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.10+"
        exit 1
    fi

    # Check DeepSpeed
    if ! python -c "import deepspeed" 2>/dev/null; then
        print_warn "DeepSpeed not installed. Installing..."
        pip install deepspeed
    fi

    # Check PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        print_error "PyTorch not installed. Please install PyTorch 2.0+"
        exit 1
    fi

    print_info "✓ Prerequisites satisfied"
}

# Function to setup NVMe offloading
setup_nvme() {
    print_step "Setting up NVMe offloading..."

    if [ ! -d "$NVME_PATH" ]; then
        print_warn "Creating NVMe directory: $NVME_PATH"
        sudo mkdir -p "$NVME_PATH"
        sudo chmod 777 "$NVME_PATH"
    fi

    # Check available space
    AVAILABLE_GB=$(df -BG "$NVME_PATH" | tail -1 | awk '{print $4}' | sed 's/G//')
    print_info "Available NVMe space: ${AVAILABLE_GB}GB"

    if [ "$AVAILABLE_GB" -lt 500 ]; then
        print_warn "Low NVMe space. Recommended: >500GB"
    fi

    print_info "✓ NVMe offloading configured"
}

# Function to setup networking
setup_networking() {
    print_step "Setting up high-speed networking..."

    # Check for InfiniBand
    if command -v ibstat &> /dev/null; then
        print_info "✓ InfiniBand detected"
        export NCCL_IB_DISABLE=0
        export NCCL_IB_HCA=$(ibstat -l | head -1)
        print_info "  HCA: $NCCL_IB_HCA"
    else
        print_warn "InfiniBand not detected. Using TCP/IP."
        export NCCL_IB_DISABLE=1
    fi

    # NCCL optimizations
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG=WARN
    export NCCL_P2P_DISABLE=0
    export NCCL_TREE_THRESHOLD=0

    # Set timeouts
    export NCCL_TIMEOUT=3600000  # 1 hour timeout

    print_info "✓ Network configuration complete"
}

# Function to generate hostfile
generate_hostfile() {
    print_step "Generating hostfile..."

    HOSTFILE="hostfile"

    if [ "$CLUSTER_TYPE" = "slurm" ]; then
        # Generate from Slurm
        if [ ! -z "$SLURM_JOB_NODELIST" ]; then
            scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE
            while IFS= read -r host; do
                echo "$host slots=$GPUS_PER_NODE" >> "${HOSTFILE}.tmp"
            done < $HOSTFILE
            mv "${HOSTFILE}.tmp" $HOSTFILE
        else
            print_error "SLURM_JOB_NODELIST not set. Not running under Slurm?"
            exit 1
        fi

    elif [ "$CLUSTER_TYPE" = "manual" ]; then
        # Manual hostfile
        if [ -f "$HOSTFILE" ]; then
            print_info "Using existing hostfile"
        else
            print_error "Please create hostfile with format: hostname slots=N"
            exit 1
        fi
    fi

    print_info "✓ Hostfile generated: $HOSTFILE"
    print_info "Cluster nodes:"
    cat $HOSTFILE
}

# Function to setup SSH keys
setup_ssh() {
    print_step "Setting up SSH keys..."

    if [ ! -f ~/.ssh/id_rsa ]; then
        print_info "Generating SSH key..."
        ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
    fi

    if [ -f "hostfile" ]; then
        print_info "Copying SSH keys to nodes..."
        while read -r line; do
            host=$(echo $line | awk '{print $1}')
            print_info "  Copying to $host..."
            ssh-copy-id -i ~/.ssh/id_rsa.pub "$host" 2>/dev/null || true
        done < hostfile
    fi

    print_info "✓ SSH keys configured"
}

# Function to test connectivity
test_connectivity() {
    print_step "Testing node connectivity..."

    if [ ! -f "hostfile" ]; then
        print_warn "No hostfile found. Skipping connectivity test."
        return
    fi

    local failed=0
    while read -r line; do
        host=$(echo $line | awk '{print $1}')
        print_info "Testing $host..."

        if ssh -o ConnectTimeout=5 "$host" "echo ok" &>/dev/null; then
            print_info "  ✓ $host reachable"
        else
            print_error "  ✗ $host unreachable"
            failed=$((failed + 1))
        fi
    done < hostfile

    if [ $failed -gt 0 ]; then
        print_error "$failed nodes unreachable"
        exit 1
    fi

    print_info "✓ All nodes reachable"
}

# Function to test GPUs on all nodes
test_gpus() {
    print_step "Testing GPUs on all nodes..."

    if [ ! -f "hostfile" ]; then
        print_warn "No hostfile found. Testing local GPUs only."
        nvidia-smi --query-gpu=name,memory.total --format=csv
        return
    fi

    while read -r line; do
        host=$(echo $line | awk '{print $1}')
        print_info "Testing GPUs on $host..."

        ssh "$host" "nvidia-smi --query-gpu=name,memory.total --format=csv" || {
            print_error "GPU test failed on $host"
            exit 1
        }
    done < hostfile

    print_info "✓ All GPUs accessible"
}

# Function to create training launch script
create_launch_script() {
    print_step "Creating training launch script..."

    cat > launch_training.sh << 'EOF'
#!/bin/bash
# Auto-generated training launch script

# Load environment variables
source setup_env.sh

# Training configuration
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_355b_32b_active.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training_8192_h800.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_stage3.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# DeepSpeed launch
deepspeed --hostfile=hostfile \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/pretraining/pretrainer.py \
    --model_config_path=$MODEL_CONFIG \
    --deepspeed_config=$DEEPSPEED_CONFIG \
    --output_dir=$OUTPUT_DIR \
    --tensorboard_dir=$OUTPUT_DIR/tensorboard \
    --checkpoint_dir=$OUTPUT_DIR/checkpoints
EOF

    chmod +x launch_training.sh
    print_info "✓ Launch script created: launch_training.sh"
}

# Function to create environment setup
create_env_script() {
    print_step "Creating environment setup script..."

    cat > setup_env.sh << EOF
#!/bin/bash
# Environment configuration for GLM-4.6 training

# CUDA and GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL settings
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600000

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Master node settings
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# DeepSpeed settings
export DEEPSPEED_NVME_PATH=$NVME_PATH

# HuggingFace settings
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Tokenizers
export TOKENIZERS_PARALLELISM=false

echo "Environment configured for GLM-4.6 training"
echo "  MASTER_ADDR: \$MASTER_ADDR"
echo "  MASTER_PORT: \$MASTER_PORT"
echo "  NCCL_IB_DISABLE: \$NCCL_IB_DISABLE"
EOF

    chmod +x setup_env.sh
    print_info "✓ Environment script created: setup_env.sh"
}

# Function to create Slurm job script
create_slurm_script() {
    if [ "$CLUSTER_TYPE" != "slurm" ]; then
        return
    fi

    print_step "Creating Slurm job script..."

    cat > submit_job.slurm << EOF
#!/bin/bash
#SBATCH --job-name=glm46-training
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Print job info
echo "Job started at: \$(date)"
echo "Running on nodes: \$SLURM_JOB_NODELIST"
echo "Total GPUs: \$((SLURM_NNODES * $GPUS_PER_NODE))"

# Setup environment
source setup_env.sh

# Generate hostfile
scontrol show hostnames \$SLURM_JOB_NODELIST > hostfile
while IFS= read -r host; do
    echo "\$host slots=$GPUS_PER_NODE"
done < hostfile > hostfile.tmp
mv hostfile.tmp hostfile

# Get master node
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)

# Launch training
./launch_training.sh

echo "Job finished at: \$(date)"
EOF

    chmod +x submit_job.slurm
    print_info "✓ Slurm script created: submit_job.slurm"
    print_info "Submit with: sbatch submit_job.slurm"
}

# Function to print summary
print_summary() {
    echo ""
    echo "================================================"
    echo "    Cluster Setup Complete!"
    echo "================================================"
    echo ""
    print_info "Cluster configuration:"
    print_info "  Type: $CLUSTER_TYPE"
    print_info "  Nodes: $NUM_NODES"
    print_info "  GPUs: $((NUM_NODES * GPUS_PER_NODE))"
    print_info "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo ""
    print_info "Generated files:"
    print_info "  ✓ hostfile - Node configuration"
    print_info "  ✓ setup_env.sh - Environment setup"
    print_info "  ✓ launch_training.sh - Training launcher"
    [ "$CLUSTER_TYPE" = "slurm" ] && print_info "  ✓ submit_job.slurm - Slurm job script"
    echo ""
    print_info "Next steps:"
    if [ "$CLUSTER_TYPE" = "slurm" ]; then
        print_info "  1. Review submit_job.slurm"
        print_info "  2. Submit: sbatch submit_job.slurm"
        print_info "  3. Monitor: squeue -u \$USER"
    else
        print_info "  1. Source environment: source setup_env.sh"
        print_info "  2. Start training: ./launch_training.sh"
        print_info "  3. Monitor logs in output/"
    fi
    echo ""
}

# Main execution
main() {
    check_prerequisites
    setup_nvme
    setup_networking
    generate_hostfile

    if [ "$CLUSTER_TYPE" = "manual" ]; then
        setup_ssh
        test_connectivity
    fi

    test_gpus
    create_env_script
    create_launch_script
    create_slurm_script
    print_summary
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            CLUSTER_TYPE="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --nvme-path)
            NVME_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --type TYPE              Cluster type: slurm, mpi, manual (default: slurm)"
            echo "  --nodes N                Number of nodes (default: 128)"
            echo "  --gpus-per-node N        GPUs per node (default: 8)"
            echo "  --master-addr ADDR       Master node address"
            echo "  --master-port PORT       Master port (default: 29500)"
            echo "  --nvme-path PATH         NVMe path for offloading (default: /local_nvme)"
            echo "  --help                   Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
