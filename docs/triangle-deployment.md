# Triangle LLM Deployment Guide

**Confidential - For Authorized Use Only**

## Prerequisites

### Authorization
- Active Triangle OS employee or authorized contractor
- Valid credentials in Triangle OS identity system
- Approved access to `triangle-llm-dev` repository

### Infrastructure Requirements
- **GPU**: NVIDIA H100 (recommended) or A100 (minimum)
- **Memory**: 1.5TB+ for BF16, 400GB+ for FP8
- **Storage**: 500GB for model weights + code
- **Network**: 100+ Mbps connection to Triangle OS infrastructure
- **OS**: Ubuntu 22.04 LTS or CentOS 8+

### Software Stack
```bash
Python >= 3.10
CUDA >= 12.0
cuDNN >= 8.9.0
PyTorch >= 2.0.0
```

## Installation

### Step 1: Clone the Repository

```bash
# Authorized users only
git clone -b triangle-llm-dev https://github.com/TriangleOS/GLM-5.git triangle-llm
cd triangle-llm
```

### Step 2: Verify Model Authenticity

```bash
# Cryptographic verification is mandatory
python scripts/verify_signature.py \
  --model-path ./models/triangle-llm-1.0 \
  --cert-path ./certs/triangle-root-ca.pem
```

Expected output:
```
✓ Model signature verified successfully
✓ Certificate chain valid
✓ Timestamp: 2026-06-27T08:15:30Z
✓ Signer: Triangle LLM Release Authority
```

### Step 3: Install Dependencies

```bash
# Install base GLM-5 requirements
pip install -r requirements.txt

# Install Triangle LLM proprietary dependencies
pip install -r requirements-triangle.txt

# Pre-commit hooks (for development)
pre-commit install
```

### Step 4: Environment Configuration

Create `.env` file in the project root:

```bash
# Triangle OS Authentication
TRIANGLE_API_KEY="your-api-key-here"
TRIANGLE_AUTH_ENDPOINT="https://auth.internal.triangleos.com"

# Model Configuration
TRIANGLE_MODEL_PATH="/path/to/models/triangle-llm-1.0"
TRIANGLE_MODEL_VARIANT="bf16"  # or "fp8"

# Hardware Settings
CUDA_VISIBLE_DEVICES="0,1,2,3"
TORCH_CUDA_ARCH_LIST="90-A100"

# Logging & Telemetry
TRIANGLE_LOG_LEVEL="INFO"
TRIANGLE_AUDIT_ENABLED="true"
TRIANGLE_TELEMETRY_ENDPOINT="https://telemetry.internal.triangleos.com"

# Security
TRIANGLE_ENABLE_PII_FILTER="true"
TRIANGLE_ENABLE_COMPLIANCE_CHECK="true"
TRIANGLE_SIGNATURE_VERIFICATION="required"
```

### Step 5: Initialize the Model

```python
from triangle_llm import TriangleLLM, TriangleLLMConfig

# Load configuration
config = TriangleLLMConfig.from_env()

# Initialize model with verification
model = TriangleLLM.load(
    model_path="./models/triangle-llm-1.0",
    config=config,
    verify_signature=True  # Mandatory
)

print("✓ Triangle LLM initialized successfully")
```

## Deployment Architectures

### Architecture A: Single GPU Server

**Use Case**: Development, testing, small-scale inference

```
┌─────────────────────────────────┐
│  HTTP/gRPC API Server           │
├─────────────────────────────────┤
│  Triangle LLM Inference Engine  │
├─────────────────────────────────┤
│  Single NVIDIA H100 GPU         │
└─────────────────────────────────┘
```

**Configuration**:
```yaml
deployment:
  type: "single-gpu"
  gpu_count: 1
  model_variant: "fp8"
  max_batch_size: 16
  max_context_length: 100000
```

**Expected Performance**:
- Throughput: 200-300 tokens/sec
- Max concurrent requests: 4-8
- Latency (p50): 2-5s per 1k tokens

### Architecture B: Multi-GPU Cluster (Tensor Parallelism)

**Use Case**: Production inference with high throughput

```
┌──────────────────────────────────┐
│  Load Balancer / API Gateway     │
├──────────────────────────────────┤
│  Triangle LLM Distributed Engine │
├──────────────────────────────────┤
│  GPU 0   GPU 1   GPU 2   GPU 3   │
│  [Sharded Model - Tensor Parallel]
└──────────────────────────────────┘
```

**Configuration**:
```yaml
deployment:
  type: "multi-gpu-tp"
  gpu_count: 4
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
  model_variant: "bf16"
  max_batch_size: 64
  max_context_length: 1000000
```

**Expected Performance**:
- Throughput: 1,200-1,800 tokens/sec
- Max concurrent requests: 32-64
- Latency (p50): 1-3s per 1k tokens

### Architecture C: High Availability Multi-Region

**Use Case**: Enterprise production with geo-redundancy

```
┌─────────────────────────────────────┐
│  Global Load Balancer               │
├──────────────┬──────────────────────┤
│              │                      │
v              v                      v
[Region 1]  [Region 2]          [Region 3]
[Cluster]   [Cluster]            [Cluster]
4x H100     4x H100              4x H100
```

**Configuration**:
```yaml
deployment:
  type: "multi-region-ha"
  regions:
    - name: "us-west"
      endpoint: "triangle-llm-west.internal"
      gpu_count: 4
      replicas: 2
    - name: "us-east"
      endpoint: "triangle-llm-east.internal"
      gpu_count: 4
      replicas: 2
    - name: "eu-central"
      endpoint: "triangle-llm-eu.internal"
      gpu_count: 4
      replicas: 1
  failover_strategy: "active-active"
  health_check_interval: 30s
```

## API Server Deployment

### Starting the API Server

```bash
python -m triangle_llm.server \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --model-path ./models/triangle-llm-1.0 \
  --tensor-parallel-size 4
```

### API Endpoints

#### Health Check
```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "triangle-llm-1.0",
  "gpu_memory_used": "78%",
  "uptime_seconds": 3600
}
```

#### Generate Completion
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $TRIANGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyze this enterprise data:",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

#### Batch Inference
```bash
curl -X POST http://localhost:8000/v1/batch \
  -H "Authorization: Bearer $TRIANGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"prompt": "Task 1", "max_tokens": 256},
      {"prompt": "Task 2", "max_tokens": 256}
    ]
  }'
```

## Monitoring & Observability

### Prometheus Metrics

Triangle LLM exports metrics on `:8001/metrics`:

```
# Model performance
triangle_llm_inference_duration_seconds
triangle_llm_tokens_generated_total
triangle_llm_requests_total{status="success|error"}

# Hardware utilization
triangle_llm_gpu_memory_percent
triangle_llm_gpu_utilization_percent
triangle_llm_temperature_celsius

# Compliance
triangle_llm_pii_detected_total
triangle_llm_compliance_checks_total
triangle_llm_audit_events_total
```

### Structured Logging

All events logged with structured format:

```json
{
  "timestamp": "2026-06-27T10:30:45.123Z",
  "level": "INFO",
  "component": "inference",
  "event": "request_completed",
  "request_id": "req-xyz-789",
  "duration_ms": 1523,
  "tokens_generated": 256,
  "model_version": "triangle-llm-1.0"
}
```

## Security Checklist

- [ ] Model signatures verified before deployment
- [ ] API authentication enabled (OAuth2 / mTLS)
- [ ] Network traffic encrypted (TLS 1.3)
- [ ] Audit logging configured and monitored
- [ ] Data retention policies enforced
- [ ] Regular security scans scheduled
- [ ] Incident response procedures documented
- [ ] Backup and recovery procedures tested

## Troubleshooting

### Issue: Model Signature Verification Failed

```
ERROR: Model signature verification failed!
```

**Solution**:
```bash
# Re-download model from authorized source
rm -rf ./models/triangle-llm-1.0
python scripts/download_model.py --version 1.0

# Verify again
python scripts/verify_signature.py
```

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size
2. Use FP8 quantized variant
3. Enable gradient checkpointing
4. Increase number of GPUs for tensor parallelism

### Issue: High Latency

**Diagnosis**:
```bash
python scripts/benchmark.py \
  --model ./models/triangle-llm-1.0 \
  --batch-size 1 \
  --seq-length 1000
```

**Common Fixes**:
- Enable KV-cache (default enabled)
- Reduce context length if possible
- Increase batch size for throughput mode
- Check GPU clock speeds and thermal throttling

## Support

For deployment assistance, contact the Triangle LLM team:
- **Internal Slack**: #triangle-llm-deployment
- **Email**: triangle-llm-deployment@triangleos.com
- **On-call**: See rotation in Pagerduty (team: triangle-llm)

---

**Document Classification**: CONFIDENTIAL  
**Last Updated**: June 27, 2026  
**Next Review**: September 27, 2026
