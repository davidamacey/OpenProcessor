# System Architecture Guide

Production-grade architecture for high-performance visual AI inference at scale.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Production Deployment Patterns](#production-deployment-patterns)
3. [Thread Safety and Concurrency](#thread-safety-and-concurrency)
4. [Best Practices](#best-practices)
5. [Scaling Strategies](#scaling-strategies)

---

## Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Load Balancer (NGINX/Envoy/Cloud LB)             │
│  - SSL termination                                          │
│  - Request routing                                          │
│  - Rate limiting                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: API Gateway (FastAPI) - Multiple Instances       │
│  - Authentication/Authorization                             │
│  - Input validation                                         │
│  - Request preprocessing                                    │
│  - Response formatting                                      │
│  - Shared Triton gRPC client pool                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (gRPC, persistent connections)
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Triton Inference Server - Multiple Instances     │
│  - Model serving                                            │
│  - Dynamic batching                                         │
│  - GPU execution                                            │
│  - Metrics export                                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Model Repository (S3/NFS/Local)                  │
│  - Version control                                          │
│  - Model artifacts                                          │
└─────────────────────────────────────────────────────────────┘
```

### Service Components

The system uses Docker Compose to orchestrate three core services:

1. **triton-api**: NVIDIA Triton Inference Server
   - GPU inference backend (device_ids: [`0`, `2`])
   - Ports: 4600 (HTTP), 4601 (gRPC), 4602 (metrics)
   - Serves TensorRT models with dynamic batching
   - Max batch size: 128

2. **yolo-api**: FastAPI Service
   - Python 3.12 with async support
   - Port: **4603** (all API endpoints)
   - Workers: 2 (dev) or 64 (production)
   - Located in `src/main.py`

3. **opensearch**: Vector Database
   - OpenSearch 3.0+ with k-NN plugin
   - Port: **4607** (REST API)
   - Indexes: images, faces, objects, ocr

---

## Production Deployment Patterns

### Preprocessing Strategy

**Client-Side (Browser/Mobile App):**
```javascript
✅ Image compression (JPEG quality 85-90%)
✅ Max resolution enforcement (e.g., 4K max)
✅ Format validation (reject unsupported formats)
❌ NO resizing/letterbox (server does this for accuracy)
❌ NO normalization (model-specific, server handles)
```

**Why?**
- Reduces bandwidth (5MB → 500KB)
- Faster uploads
- But server still controls model-specific preprocessing

**API Layer (FastAPI):**
```python
✅ Fast validation (file size, format, dimensions)
✅ Image decoding (OpenCV/Pillow)
✅ Error handling and retries
✅ Request batching/aggregation (advanced)
❌ NO heavy preprocessing (defeats GPU pipeline)
```

**Triton Layer:**
```
✅ Model-specific preprocessing (letterbox, normalize)
✅ GPU-accelerated (DALI for full GPU pipeline)
✅ Batch processing
```

### Production Configuration

#### FastAPI (docker-compose.yml)

```yaml
yolo-api:
  command:
    - uvicorn
    - src.main:app
    - --host=0.0.0.0
    - --port=4603
    # Workers: (2 × CPU cores) + 1
    - --workers=32

    # Concurrency: requests per worker
    # 512 × 32 workers = 16,384 total capacity
    - --limit-concurrency=512

    # Connection settings
    - --backlog=8192              # Socket queue (was 4096)
    - --timeout-keep-alive=120    # Reuse connections (was 75)

    # Memory management
    - --limit-max-requests=50000  # Recycle workers (was 10000)
    - --limit-max-requests-jitter=5000  # Spread recycling

    # Performance
    - --loop=uvloop               # 2-3x faster event loop
    - --http=httptools            # Faster HTTP parsing

  environment:
    # gRPC settings for Triton
    GRPC_ENABLE_FORK_SUPPORT: "1"
    GRPC_POLL_STRATEGY: "epoll1"  # Linux-optimized

  deploy:
    resources:
      limits:
        memory: 16G
      reservations:
        memory: 8G
```

#### Triton Server (docker-compose.yml)

```yaml
triton-api:
  command:
    - tritonserver
    - --model-store=/models

    # Batching configuration
    - --backend-config=default-max-batch-size=128

    # Thread pool (CPU cores × 2)
    - --backend-config=tensorflow,version=2
    - --backend-config=python,shm-default-byte-size=16777216

    # HTTP/gRPC settings
    - --grpc-keepalive-time=7200000        # 2 hours
    - --grpc-keepalive-timeout=20000       # 20 seconds
    - --grpc-keepalive-permit-without-calls=1
    - --grpc-http2-max-pings-without-data=2

    # Performance
    - --model-control-mode=explicit
    - --strict-model-config=false
    - --log-verbose=1

  deploy:
    resources:
      limits:
        memory: 32G
      reservations:
        memory: 16G
```

### Shared Triton Client Architecture

**CRITICAL**: Use shared gRPC client pool to enable dynamic batching.

**Current Architecture (BROKEN):**
```python
# ❌ WRONG - Creates new connection per request
@app.post("/detect")
def detect(image: UploadFile):
    client = TritonEnd2EndClient(...)  # NEW CONNECTION!
    result = client.infer(image)
    return result

# Result: 1000 requests → 1000 gRPC connections → NO BATCHING
```

**Production Architecture (CORRECT):**
```python
# ✅ RIGHT - Shared connection pool

# Global client pool (singleton)
from src.utils.triton_shared_client import get_triton_client

# At startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create shared client ONCE
    global triton_client
    triton_client = get_triton_client("triton-api:8001")

    # Configure gRPC connection
    # - Keep-alive to prevent connection drops
    # - Connection pooling for throughput

    yield

    # Cleanup on shutdown
    triton_client.close()

# In endpoint
@app.post("/detect")
async def detect(image: UploadFile):
    # Reuse shared client
    client = TritonEnd2EndClient(
        triton_url=TRITON_URL,
        model_name=model_name,
        shared_grpc_client=triton_client  # SHARED!
    )
    result = client.infer(image)
    return result

# Result: 1000 requests → 1 gRPC connection → BATCHING WORKS!
```

---

## Thread Safety and Concurrency

### The Thread Safety Problem

When using async/await with `asyncio.to_thread()`, blocking operations run in a ThreadPoolExecutor. This creates potential thread safety issues with shared resources.

### YOLO Model Thread Safety

**The Problem:**

From [Ultralytics documentation](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/):

> YOLO models contain internal state that can be corrupted when accessed by multiple threads simultaneously.

**Threading Architecture:**
- 32 worker **processes** (uvicorn --workers=32)
- Each process has **async event loop** + **ThreadPoolExecutor**
- `asyncio.to_thread()` → Runs blocking I/O in thread pool
- Multiple concurrent requests → **Multiple threads accessing same instance** → **RACE CONDITIONS**

**WRONG Approach (Cached, Unsafe):**
```python
@lru_cache(maxsize=32)  # ❌ Creates shared instance
def get_triton_yolo_client(model_url: str):
    return YOLO(model_url, task="detect")

# Request 1 (Thread A) → calls model(img1)
# Request 2 (Thread B) → calls model(img2) simultaneously
# Both threads modify same YOLO instance → CORRUPTION!
```

**CORRECT Approach (Per-Request, Safe):**
```python
def create_triton_yolo_client(model_url: str):
    """
    Create a new YOLO Triton client instance

    NOTE: Creates per-request for thread safety. Lightweight (no model loading).
    """
    return YOLO(model_url, task="detect")

# Each request gets its own client instance
# No shared state between threads
# No race conditions
```

### Performance Impact Analysis

**Before (Cached, Unsafe):**
```
First request:  2ms (create) + 20ms (inference) = 22ms
Second request: 0ms (cached)  + 20ms (inference) = 20ms ✅ 2ms saved
                                                         ❌ BUT UNSAFE!
```

**After (Per-Request, Safe):**
```
First request:  2ms (create) + 20ms (inference) = 22ms
Second request: 2ms (create) + 20ms (inference) = 22ms ✅ SAFE!
                                                        ⚠️  2ms slower
```

**Trade-off**: We lose 2ms per request (9% overhead), but gain **correctness and safety**.

### Why Triton Clients Are Lightweight

- `YOLO("grpc://triton-api:8001/...")` doesn't load PyTorch model
- It's just a gRPC client wrapper (~1-2ms creation overhead)
- No heavy model weights in memory
- Creation overhead is ~5-10% of total request time
- **Safety > marginal performance gain**

### Concurrency Best Practices

1. **Performance optimizations must preserve correctness**
   - Fast but wrong ❌
   - Reasonably fast + correct ✅

2. **Thread safety is non-negotiable**
   - Race conditions are hard to debug
   - Intermittent failures are worse than consistent slowness
   - Always check library thread-safety docs

3. **Measure before optimizing**
   - Don't cache things that are cheap to create
   - Profile to identify real bottlenecks

---

## Best Practices

### Health Checks and Circuit Breakers

```python
# Health check endpoint
@app.get("/health")
async def health():
    """Comprehensive health check."""
    checks = {
        "api": "healthy",
        "triton": await check_triton_health(),
        "gpu": check_gpu_availability(),
        "memory": check_memory_usage()
    }

    # Fail if Triton is down
    if checks["triton"] != "healthy":
        raise HTTPException(status_code=503, detail="Triton unavailable")

    return checks

async def check_triton_health():
    """Check Triton is responding."""
    try:
        client = get_triton_client(TRITON_URL)
        if client.is_server_live():
            return "healthy"
        return "unhealthy"
    except:
        return "unavailable"

# Circuit breaker pattern
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_triton_with_circuit_breaker(model_name, image):
    """Automatic fallback if Triton fails repeatedly."""
    client = TritonEnd2EndClient(...)
    return await client.infer(image)
```

### Monitoring and Metrics

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
requests_total = Counter('api_requests_total', 'Total requests', ['endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')
active_requests = Gauge('api_active_requests', 'Active requests')

# Triton metrics
triton_batch_size = Histogram('triton_batch_size', 'Triton batch sizes')
triton_queue_time = Histogram('triton_queue_time_ms', 'Time in Triton queue')
triton_inference_time = Histogram('triton_inference_time_ms', 'Triton inference time')

# Track in middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    active_requests.inc()
    start = time.time()

    try:
        response = await call_next(request)
        requests_total.labels(request.url.path, response.status_code).inc()
        return response
    finally:
        request_duration.observe(time.time() - start)
        active_requests.dec()
```

**Grafana Dashboards:**
1. Request rate (RPS)
2. Latency percentiles (P50, P95, P99)
3. Triton batch sizes (should be >1!)
4. GPU utilization
5. Error rates

---

## Scaling Strategies

### Single GPU (Current Setup)

**Capacity:** ~500-1000 RPS (with batching fixed)

```
Load Balancer
     │
     ▼
FastAPI (1 instance, 32 workers)
     │
     ▼
Triton (1 instance, 1 GPU)
```

### Multi-GPU (Single Node)

**Capacity:** ~2000-4000 RPS

```
Load Balancer
     │
     ├─▶ FastAPI (1 instance, 32 workers)
     │        │
     │        ├─▶ Triton GPU:0 (models A-C)
     │        └─▶ Triton GPU:1 (models D-F)
```

### Production Scale (Multi-Node)

**Capacity:** 10,000+ RPS

```
Cloud Load Balancer (AWS ALB/GCP LB)
     │
     ├─▶ FastAPI Pod 1 (K8s)
     │        └─▶ Triton Pod 1 (GPU Node 1)
     │
     ├─▶ FastAPI Pod 2 (K8s)
     │        └─▶ Triton Pod 2 (GPU Node 2)
     │
     ├─▶ FastAPI Pod 3 (K8s)
     │        └─▶ Triton Pod 3 (GPU Node 3)
     │
     └─▶ ... (autoscaling 3-20 pods)
```

### Deployment Options

1. **Docker Compose** (1-4 GPUs, single node) ← Current
2. **Docker Swarm** (4-16 GPUs, 2-4 nodes)
3. **Kubernetes** (16+ GPUs, 4+ nodes) ← Fortune 500 scale

### Request Aggregation (Advanced)

For **MAXIMUM** throughput, add client-side batching:

```python
# src/utils/request_aggregator.py
class RequestAggregator:
    """
    Collects individual requests and sends them as batches.

    Config:
    - max_batch_size: 32 (matches Triton preferred_batch_size)
    - max_wait_ms: 10 (balance latency vs throughput)
    """

    def __init__(self, max_batch_size=32, max_wait_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.queue = []
        self.lock = asyncio.Lock()
        self.processing = False

    async def submit(self, image_bytes: bytes):
        """Submit request and wait for batch processing."""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append((image_bytes, future))

            # Start batch processor if needed
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batches())

            # Flush immediately if full
            if len(self.queue) >= self.max_batch_size:
                await self._flush()

        return await future

    async def _process_batches(self):
        """Background task to flush batches."""
        while True:
            await asyncio.sleep(self.max_wait_ms)

            async with self.lock:
                if self.queue:
                    await self._flush()
                else:
                    self.processing = False
                    break

    async def _flush(self):
        """Send accumulated requests as batch."""
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        # Process batch
        try:
            images = [req[0] for req in batch]
            results = await self._infer_batch(images)

            # Complete futures
            for (_, future), result in zip(batch, results):
                future.set_result(result)
        except Exception as e:
            for _, future in batch:
                future.set_exception(e)
```

**When to use:**
- High-throughput scenarios (1000+ RPS)
- Batch workloads (offline video processing)
- GPU utilization optimization

**When NOT to use:**
- Real-time streaming (adds latency)
- Low request rate (<100 RPS)

---

## Reference Architectures

### Uber's ML Platform

```
API Gateway (FastAPI)
   └─▶ Request Router
       └─▶ Model Server (Triton)
           └─▶ Feature Store (Redis)
```

### Netflix Recommendation System

```
Zuul API Gateway
   └─▶ Microservices (Spring Boot/FastAPI)
       └─▶ TensorFlow Serving / Triton
           └─▶ Model Registry (S3)
```

### Current System (Production-Ready)

```
NGINX Load Balancer
   └─▶ FastAPI (3 instances, shared Triton client)
       └─▶ Triton (2 instances, 2 GPUs)
           └─▶ Model Repository (Local/NFS)
           └─▶ Prometheus/Grafana (monitoring)
```

---

## Summary

### Critical Architecture Decisions

1. ✅ **Shared Triton gRPC client** (enables dynamic batching)
2. ✅ **Per-request model instances** (thread safety)
3. ✅ **Production hardening** (health checks, metrics)
4. ✅ **Horizontal scaling** (when >1000 RPS)

### Current Stack is Production-Grade

- FastAPI ✅ (Netflix, Uber use this)
- Triton ✅ (NVIDIA's official solution)
- Docker Compose ✅ (Good for 1-4 GPUs)

**Next step:** Kubernetes when you need 10+ GPUs across multiple nodes.

---

## Further Reading

- [FastAPI Concurrency and async/await](https://fastapi.tiangolo.com/async/)
- [Ultralytics Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/)
- [NVIDIA Triton Optimization Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/optimization.html)
- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)

---

**Last Updated**: 2026-01-26
**Version**: 2.0 (Consolidated documentation)
