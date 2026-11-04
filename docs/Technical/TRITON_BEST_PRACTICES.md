# NVIDIA Triton Inference Server - Best Practices

Comprehensive guide for optimizing and deploying models with Triton in production.

## Table of Contents
- [Batch Size Configuration](#batch-size-configuration)
- [Dynamic Batching](#dynamic-batching)
- [Instance Groups](#instance-groups)
- [GPU Utilization](#gpu-utilization)
- [Performance Tuning](#performance-tuning)
- [Security & Production Deployment](#security--production-deployment)
- [YOLO-Specific Recommendations](#yolo-specific-recommendations)
- [Our Current Configuration](#our-current-configuration)

> **Note:** Examples reference `yolov11_small_trt_end2end` (our active YOLO model).
> Batch sizes and instance counts vary by model — see `models/*/config.pbtxt` for current values.

---

## Batch Size Configuration

### max_batch_size Guidelines

**Official Guidance:**
- Set `max_batch_size >= 1` for models that support batching
- Set `max_batch_size = 0` for models that don't support batching
- Default via command line: `--backend-config=default-max-batch-size=<int>` (default: 4)

**Industry Insights:**
- **Typical range**: 4-128 depending on model size and GPU memory
- **YOLO models**: Common values are 4, 8, 16, 32, 50, 64
- **Auto-sizing**: Use `batch=-1` for ~60% CUDA memory utilization
- **Manual tuning**: Start with powers of 2 (8, 16, 32, 64)

**GPU Memory Considerations:**
- **RTX A6000 (48GB)**: Can handle larger batches (50-128) for YOLO11
- **Memory calculation**: `batch_size × model_memory × safety_margin`
- **Rule of thumb**: Leave 20-30% GPU memory free for operations

### preferred_batch_size Configuration

```protobuf
dynamic_batching {
  preferred_batch_size: [8, 16, 25, 50]  # Example: powers of 2 + GPU-optimal sizes
  max_queue_delay_microseconds: 100
}
```

**Best Practices:**
- Include powers of 2 for TensorRT optimization
- Add GPU-specific optimal sizes (multiples of CUDA cores)
- Smaller models benefit from larger batches
- Larger models may need smaller preferred sizes

**Our YOLO11 configuration** is in `models/yolov11_small_trt_end2end/config.pbtxt` (batch size, preferred sizes and queue delay, with the reasons in the file's comments).

---

## Dynamic Batching

### Why Dynamic Batching Matters

> **"Dynamic batching is the feature that provides the largest performance improvement for most models."**
> — NVIDIA Triton Documentation

**Performance Gains:**
- Without batching: ~73 inferences/sec (Inception model example)
- With dynamic batching: ~272 inferences/sec (**3.7x improvement**)
- Real-world: 70%+ throughput increase is common

### How It Works

```
Request Flow:
1. Requests arrive at varying times
2. Triton waits max_queue_delay_microseconds
3. Forms batch matching preferred_batch_size (or smaller)
4. Sends batch to GPU for inference
5. Distributes results back to individual requests
```

### Configuration Template

```protobuf
dynamic_batching {
  # Preferred batch sizes (Triton tries to form these)
  preferred_batch_size: [8, 16, 25, 50]

  # Maximum delay waiting for full batch (microseconds)
  max_queue_delay_microseconds: 100  # 0.1ms

  # Preserve order of responses
  preserve_ordering: false  # Better throughput

  # Queue policy
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 1000000  # 1 second
    max_queue_size: 100
  }
}
```

### Tuning Parameters

**max_queue_delay_microseconds:**
- **Too low (1-10)**: Smaller batches, lower latency, lower throughput
- **Optimal (50-500)**: Balance of latency and throughput
- **Too high (>1000)**: Larger batches, higher latency, higher throughput

**preserve_ordering:**
- `false`: Better throughput (recommended for most cases)
- `true`: Maintain request order (needed for some applications)

---

## Instance Groups

### Why Multiple Instances?

**Benefits:**
1. **Memory transfer overlap**: CPU ↔ GPU transfers happen during inference
2. **Better GPU utilization**: More parallel work on GPU
3. **Higher throughput**: Process multiple batches simultaneously

**Typical Configuration:**

```protobuf
instance_group [
  {
    count: 2  # Number of model instances
    kind: KIND_GPU
    gpus: [0]  # GPU device ID
  }
]
```

### Choosing Instance Count

**General Guidelines:**
- **Start with 2 instances** - most common sweet spot
- **Larger models**: 1-2 instances
- **Smaller models**: 2-4 instances
- **GPU memory limit**: `instances × model_memory < GPU_memory × 0.8`

**Our capacity.** Per-instance GPU memory differs by model and by TensorRT build; the figures are in the
comments of each `config.pbtxt` (for example SCRFD is about 1.7 GB for one instance). Instance counts
come from the installer GPU profile; `./openprocessor gpu plan` shows what fits the card.

**Testing Strategy:**
```bash
# Use perf_analyzer to find optimal count
perf_analyzer -m yolov11_small_trt_end2end --concurrency-range 1:64:8
# Then adjust instance count based on GPU utilization
```

---

## GPU Utilization

### Target Metrics

**Optimal GPU Utilization:**
- **50-95%**: Good - GPU is well utilized
- **<50%**: Under-utilized - increase batch size or instances
- **>95%**: Risk of saturation - may cause queuing

**Monitoring:**
```bash
# Real-time monitoring
nvidia-smi dmon -s u

# Triton metrics (port 8002)
curl http://localhost:8002/metrics | grep gpu_utilization
```

### Maximizing GPU Utilization

**Formula for Concurrency:**
```
Optimal Concurrency = 2 × max_batch_size × instance_count

Example (our config):
= 2 × 64 × 2 = 256 concurrent requests
```

**Tuning Process:**
1. Start with 1 instance, enable dynamic batching
2. Increase concurrency until throughput plateaus
3. Add 2nd instance, test again
4. Monitor GPU utilization - aim for 70-90%
5. Adjust batch sizes if needed

---

## Performance Tuning

### Optimization Hierarchy

**Priority Order (biggest impact first):**

1. **Enable Dynamic Batching** → 2-4x improvement
2. **Use TensorRT Backend** → 2-3x improvement over ONNX
3. **Add 2nd Model Instance** → 1.5-2x improvement
4. **Optimize Batch Sizes** → 10-30% improvement
5. **Fine-tune Queue Delays** → 5-15% improvement

### Concurrency vs Latency

**Latency-Optimized (real-time applications):**
```protobuf
max_batch_size: 1  # or small like 4
dynamic_batching { ... }  # Disabled or minimal
instance_group { count: 1 }
concurrency: 1
```

**Throughput-Optimized (batch processing):**
```protobuf
max_batch_size: 64  # or higher
dynamic_batching {
  preferred_batch_size: [16, 32, 64]
  max_queue_delay_microseconds: 500
}
instance_group { count: 2 }
concurrency: 128+
```

**Balanced (our use case):**
```protobuf
max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [8, 16, 32, 64]
  max_queue_delay_microseconds: 25000
}
instance_group { count: 2 }
concurrency: 16-64
```

### Performance Testing Tools

**perf_analyzer (included with Triton):**
```bash
# Basic throughput test
perf_analyzer -m yolov11_small_trt_end2end \
  --concurrency-range 1:64:8 \
  --measurement-interval 5000

# Latency test
perf_analyzer -m yolov11_small_trt_end2end \
  --concurrency-range 1:16:1 \
  --measurement-mode time_windows \
  --latency-threshold 100

# With custom data
perf_analyzer -m yolov11_small_trt_end2end \
  --input-data /path/to/images.json \
  --concurrency-range 1:32:4
```

**Model Analyzer:**
```bash
# Find optimal configuration automatically
model-analyzer profile \
  --model-repository /models \
  --profile-models yolov11_small_trt_end2end \
  --triton-launch-mode docker \
  --output-model-repository-path /optimized_models
```

---

## Security & Production Deployment

### Deployment Architecture

**Recommended Pattern:**
```
[Client] → [Gateway/Proxy] → [Triton Server] → [GPU]
            ↓
         - Authorization
         - Rate limiting
         - Load balancing
         - Encryption (TLS)
```

**Don't expose Triton directly to internet** - use nginx, traefik, or API gateway

### Security Best Practices

**Critical Settings:**
```bash
# Launch with security flags
tritonserver \
  --model-store=/models \
  --exit-on-error=true \              # Fail fast on errors
  --strict-model-config=true \        # Require explicit configs
  --strict-readiness=true \           # Only ready when all models loaded
  --model-control-mode=explicit \     # No auto-loading
  --load-model=yolov11_small_trt_end2end \
  --load-model=scrfd_10g_bnkps \
  --log-verbose=1
```

**Model Repository Access:**
- ⚠️ **CRITICAL**: Dynamic model loading allows arbitrary code execution
- Use read-only volumes in production
- Implement strict access control
- Audit model changes

**Container Security:**
```yaml
# Docker Compose security
services:
  triton:
    user: triton-server  # Non-root user
    read_only: true
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
```

**Kubernetes RBAC:**
- Use service accounts with minimal permissions
- Network policies to restrict traffic
- Pod security policies

### Monitoring & Observability

**Triton Metrics (Prometheus format on port 8002):**
```bash
# Key metrics to monitor
nv_inference_request_success      # Successful requests
nv_inference_request_failure      # Failed requests
nv_inference_queue_duration_us    # Queue time
nv_inference_compute_infer_duration_us  # Inference time
nv_gpu_utilization                # GPU usage %
nv_gpu_memory_used_bytes         # GPU memory
```

**Alerting Thresholds:**
- Request failure rate > 1%
- Queue duration > 1000ms
- GPU utilization < 30% or > 95%
- GPU memory > 90%

---

## YOLO-Specific Recommendations

### Export Format: TensorRT (not ONNX)

**Problem with ONNX Runtime + TensorRT EP:**
- Outputs copied to CPU (huge overhead)
- 4x slower than native TensorRT

**Solution:**
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(
    format="engine",  # Native TensorRT
    imgsz=640,
    device=0,
    half=True,  # FP16
    batch=50,   # Max batch
    workspace=4  # 4GB workspace
)
```

**Backend Configuration:**
```protobuf
platform: "tensorrt_plan"  # NOT "onnxruntime_onnx"

optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
      parameters {
        key: "precision_mode"
        value: "FP16"  # 2x faster than FP32
      }
    }
  }
}
```

### YOLO Batch Size Recommendations

**Based on Model Size:**
```
YOLO11-nano:   max_batch: 64-128, preferred: [8, 16, 32, 64]
YOLO11-small:  max_batch: 32-64,  preferred: [8, 16, 25, 50]
YOLO11-medium: max_batch: 16-50,  preferred: [4, 8, 16, 25]
YOLO11-large:  max_batch: 8-32,   preferred: [4, 8, 16]
```

### Input Preprocessing

**Let YOLO handle it:**
- ✅ YOLO automatically resizes to 640x640
- ✅ YOLO handles normalization
- ✅ YOLO applies padding
- ❌ Don't preprocess on host - waste of time

**FastAPI should only:**
1. Decode image format (JPEG/PNG → BGR array)
2. Validate (null checks, max dimension)
3. Pass to YOLO

### Measured performance

This guide holds no throughput numbers. The measured results, with hardware and method, are in [PERFORMANCE.md](../PERFORMANCE.md).

---

## Our Current Configuration

The model configs are the source of truth: read `models/*/config.pbtxt` for batch sizes, instance
counts and queue delays, and `./openprocessor gpu plan` for what the installer profile
(`minimal`, `small`, `standard`, `full`) puts on a card. Engines are built by the `model-export`
container ([export/README.md](../../export/README.md)); precision follows the ONNX dtypes
(TensorRT 11 builds are strongly typed).

Models Triton serves: `yolov11_small_trt_end2end` (YOLO11, GPU NMS), `yolo26_small_trt` (YOLO26,
NMS-free), `scrfd_10g_bnkps`, `arcface_w600k_r50`, `mobileclip2_s2_image_encoder`,
`mobileclip2_s2_text_encoder`, `paddleocr_det_trt`, `paddleocr_rec_trt`, `ocr_pipeline` (a Python
BLS), `pe_image_encoder`, `pe_text_encoder` and `gpu_image_prep` (decode-once ingest, a Python
backend).

Tuning for latency or throughput is done per model with the method below, and recorded in
[PERFORMANCE.md](../PERFORMANCE.md).

---

## Testing Methodology

### Step-by-Step Optimization

**1. Baseline (no optimization):**
```bash
# Single instance, no batching
perf_analyzer -m yolov11_small_trt_end2end --concurrency-range 1:16:1
```

**2. Enable dynamic batching:**
```bash
# Update config, restart Triton
perf_analyzer -m yolov11_small_trt_end2end --concurrency-range 1:64:8
# Expected: 2-4x throughput improvement
```

**3. Add 2nd instance:**
```bash
# Update config, restart Triton
perf_analyzer -m yolov11_small_trt_end2end --concurrency-range 1:128:16
# Expected: 1.5-2x additional improvement
```

**4. Tune batch sizes:**
```bash
# Experiment with preferred_batch_size values
# Monitor GPU utilization: nvidia-smi dmon
# Aim for 70-90% GPU util
```

### Key Metrics to Track

**For each configuration change:**
- Throughput (inferences/sec)
- Latency (p50, p95, p99)
- GPU utilization %
- GPU memory usage
- Queue time
- Error rate

---

## References

### Official Documentation
- [Triton Optimization Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html)
- [Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher)
- [Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [Performance Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
- [Model Analyzer](https://github.com/triton-inference-server/model_analyzer)

### Ultralytics + Triton
- [YOLO TensorRT Export](https://docs.ultralytics.com/integrations/tensorrt/)
- [YOLO Triton Deployment](https://docs.ultralytics.com/guides/triton-inference-server/)

### Community Resources
- [Triton GitHub Discussions](https://github.com/triton-inference-server/server/discussions)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/ai/triton-inference-server/)

---
