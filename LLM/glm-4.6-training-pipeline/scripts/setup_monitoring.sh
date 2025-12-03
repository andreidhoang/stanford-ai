#!/bin/bash
# GLM-4.6 Monitoring Setup Script
#
# Sets up Prometheus + Grafana monitoring stack for training

set -e

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
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
NODE_EXPORTER_PORT="${NODE_EXPORTER_PORT:-9100}"
NVIDIA_EXPORTER_PORT="${NVIDIA_EXPORTER_PORT:-9445}"

echo "================================================"
echo "    GLM-4.6 Monitoring Setup"
echo "================================================"
echo ""

# Function to check Docker
check_docker() {
    print_step "Checking Docker..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_warn "docker-compose not found. Installing..."
        pip install docker-compose
    fi

    print_info "✓ Docker available"
}

# Function to create Prometheus configuration
create_prometheus_config() {
    print_step "Creating Prometheus configuration..."

    mkdir -p monitoring

    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'glm46-training'

scrape_configs:
  # Training metrics from DeepSpeed
  - job_name: 'deepspeed'
    static_configs:
      - targets: ['localhost:29500']
    metrics_path: '/metrics'

  # Node metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # GPU metrics
  - job_name: 'nvidia'
    static_configs:
      - targets: ['nvidia-exporter:9445']

  # vLLM inference metrics (if deployed)
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - 'alerts.yml'
EOF

    print_info "✓ Prometheus config created"
}

# Function to create alert rules
create_alert_rules() {
    print_step "Creating alert rules..."

    cat > monitoring/alerts.yml << 'EOF'
groups:
  - name: training_alerts
    interval: 30s
    rules:
      # GPU utilization alerts
      - alert: LowGPUUtilization
        expr: nvidia_gpu_duty_cycle < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low GPU utilization on {{ $labels.instance }}"
          description: "GPU utilization is {{ $value }}% (< 50%)"

      - alert: HighGPUMemory
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High GPU memory on {{ $labels.instance }}"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"

      # Training metrics alerts
      - alert: TrainingStalled
        expr: rate(training_steps_total[5m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Training appears stalled"
          description: "No training steps completed in last 10 minutes"

      - alert: HighLoss
        expr: training_loss > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Training loss is high"
          description: "Loss is {{ $value }}, may indicate training instability"

      - alert: LossSpike
        expr: abs(delta(training_loss[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Training loss spike detected"
          description: "Loss changed by {{ $value }} in 5 minutes"

      # System alerts
      - alert: HighNodeMemory
        expr: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low memory on {{ $labels.instance }}"
          description: "Available memory is {{ $value | humanizePercentage }}"

      - alert: NodeDown
        expr: up{job="node"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Node {{ $labels.instance }} is down"
          description: "Cannot scrape metrics from node"
EOF

    print_info "✓ Alert rules created"
}

# Function to create Grafana dashboards
create_grafana_dashboard() {
    print_step "Creating Grafana dashboard..."

    mkdir -p monitoring/dashboards

    cat > monitoring/dashboards/glm46-training.json << 'EOF'
{
  "dashboard": {
    "title": "GLM-4.6 Training Dashboard",
    "tags": ["glm46", "training"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Training Loss",
        "type": "graph",
        "targets": [
          {
            "expr": "training_loss",
            "legendFormat": "Loss"
          }
        ],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "id": 2,
        "title": "Learning Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "training_learning_rate",
            "legendFormat": "LR"
          }
        ],
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "id": 3,
        "title": "Tokens per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(training_tokens_total[1m])",
            "legendFormat": "Tokens/s"
          }
        ],
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
      },
      {
        "id": 4,
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_duty_cycle",
            "legendFormat": "GPU {{gpu}}"
          }
        ],
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
      },
      {
        "id": 5,
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100",
            "legendFormat": "GPU {{gpu}}"
          }
        ],
        "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8}
      },
      {
        "id": 6,
        "title": "Expert Balance (Coefficient of Variation)",
        "type": "graph",
        "targets": [
          {
            "expr": "expert_balance_cv",
            "legendFormat": "CV"
          }
        ],
        "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8}
      }
    ]
  }
}
EOF

    print_info "✓ Grafana dashboard created"
}

# Function to create Docker Compose for monitoring
create_monitoring_compose() {
    print_step "Creating Docker Compose configuration..."

    cat > monitoring/docker-compose.monitoring.yml << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: glm46-prometheus
    ports:
      - "$PROMETHEUS_PORT:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: glm46-grafana
    ports:
      - "$GRAFANA_PORT:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://localhost:$GRAFANA_PORT
    volumes:
      - grafana-data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: glm46-node-exporter
    ports:
      - "$NODE_EXPORTER_PORT:9100"
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    restart: unless-stopped
    networks:
      - monitoring

  nvidia-exporter:
    image: nvidia/dcgm-exporter:latest
    container_name: glm46-nvidia-exporter
    runtime: nvidia
    ports:
      - "$NVIDIA_EXPORTER_PORT:9445"
    environment:
      - DCGM_EXPORTER_LISTEN=:9445
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring:
    driver: bridge
EOF

    print_info "✓ Monitoring compose created"
}

# Function to start monitoring stack
start_monitoring() {
    print_step "Starting monitoring stack..."

    cd monitoring
    docker-compose -f docker-compose.monitoring.yml up -d

    print_info "✓ Monitoring stack started"
    print_info ""
    print_info "Access points:"
    print_info "  Prometheus: http://localhost:$PROMETHEUS_PORT"
    print_info "  Grafana: http://localhost:$GRAFANA_PORT (admin/admin)"
    print_info "  Node Exporter: http://localhost:$NODE_EXPORTER_PORT/metrics"
    print_info "  NVIDIA Exporter: http://localhost:$NVIDIA_EXPORTER_PORT/metrics"
}

# Function to create monitoring README
create_monitoring_readme() {
    print_step "Creating monitoring documentation..."

    cat > monitoring/README.md << 'EOF'
# GLM-4.6 Training Monitoring

Prometheus + Grafana monitoring stack for GLM-4.6 training.

## Components

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Node Exporter**: System metrics (CPU, memory, disk)
- **NVIDIA DCGM Exporter**: GPU metrics

## Quick Start

```bash
# Start monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Stop monitoring
docker-compose -f docker-compose.monitoring.yml down
```

## Access

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Metrics

### Training Metrics
- `training_loss` - Current training loss
- `training_learning_rate` - Current learning rate
- `training_tokens_total` - Total tokens processed
- `training_steps_total` - Total training steps
- `expert_balance_cv` - Expert utilization coefficient of variation

### GPU Metrics
- `nvidia_gpu_duty_cycle` - GPU utilization %
- `nvidia_gpu_memory_used_bytes` - GPU memory used
- `nvidia_gpu_temperature` - GPU temperature

### System Metrics
- `node_cpu_seconds_total` - CPU usage
- `node_memory_MemAvailable_bytes` - Available memory
- `node_disk_io_time_seconds_total` - Disk I/O time

## Alerts

Configured alerts:
- Low GPU utilization (< 50% for 10min)
- High GPU memory (> 95% for 5min)
- Training stalled (no steps for 10min)
- High/spiking loss values
- Low system memory
- Node down

## Grafana Dashboards

Pre-configured dashboards:
- GLM-4.6 Training Overview
- GPU Performance
- System Resources
- Expert Utilization

## Custom Metrics

Add custom metrics in your training code:

```python
from prometheus_client import Counter, Gauge

# Define metrics
training_steps = Counter('training_steps_total', 'Total training steps')
training_loss = Gauge('training_loss', 'Current training loss')

# Update metrics
training_steps.inc()
training_loss.set(loss.item())
```

## Troubleshooting

**Prometheus not scraping:**
- Check target status: http://localhost:9090/targets
- Verify network connectivity
- Check firewall rules

**High memory usage:**
- Reduce retention period in prometheus.yml
- Limit number of metrics collected

**NVIDIA exporter not working:**
- Ensure NVIDIA Docker runtime is installed
- Check GPU accessibility: `nvidia-smi`
EOF

    print_info "✓ Documentation created"
}

# Main execution
main() {
    check_docker
    create_prometheus_config
    create_alert_rules
    create_grafana_dashboard
    create_monitoring_compose
    create_monitoring_readme

    echo ""
    print_info "Monitoring setup complete!"
    echo ""
    print_info "Start monitoring with:"
    print_info "  cd monitoring"
    print_info "  docker-compose -f docker-compose.monitoring.yml up -d"
    echo ""
    print_info "Or run now:"
    read -p "Start monitoring stack now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_monitoring
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prometheus-port)
            PROMETHEUS_PORT="$2"
            shift 2
            ;;
        --grafana-port)
            GRAFANA_PORT="$2"
            shift 2
            ;;
        --start)
            AUTO_START=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prometheus-port PORT   Prometheus port (default: 9090)"
            echo "  --grafana-port PORT      Grafana port (default: 3000)"
            echo "  --start                  Start monitoring immediately"
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
