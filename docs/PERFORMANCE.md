# Performance Optimization Guide

Complete guide for optimizing FastAPI and Triton Inference Server performance.

---

## Table of Contents

1. [Overview](#overview)
2. [FastAPI Optimizations](#fastapi-optimizations)
3. [gRPC Connection Management](#grpc-connection-management)
4. [Benchmarking](#benchmarking)
5. [Profiling](#profiling)
6. [Tuning Parameters](#tuning-parameters)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Optimizations Applied

The system includes several production-grade optimizations:

1. **High-Performance JSON** (orjson) - 2-3x faster serialization
2. **Optimized Image Processing** (pillow-simd) - 4-10x faster operations
3. **Request Validation** - Early rejection of invalid/oversized requests
4. **Performance Monitoring** - Automatic request timing and metrics
5. **Optimized Uvicorn Configuration** - Tuned worker and connection settings

### Expected Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **API Overhead** | 8-15ms | 4-8ms | **~50% reduction** |
| **JSON Encoding** | 2-3ms | 1ms | **2-3x faster** |
| **Image Decode** | 5-10ms | 1-2ms | **4-5x faster** |
| **Throughput** | Baseline | +15-20% | **More req/sec** |

**Note**: Total end-to-end latency improvement is 10-15% because GPU inference still dominates total request time.

---

## FastAPI Optimizations

### 1. High-Performance JSON Serialization (orjson)

**Implementation**:
- Added `orjson` to requirements.txt
- Configured `ORJSONResponse` as default response class

```python
from fastapi.responses import ORJSONResponse

app = FastAPI(
    default_response_class=ORJSONResponse  # All responses use orjson
)
```

**Impact**: 2-3x faster JSON encoding/decoding

**Benchmark**:
```bash
# Before (stdlib json): ~500 MB/s
# After (orjson): ~1500 MB/s
```

### 2. Optimized Image Processing (pillow-simd)

**Implementation**:
- Replaced standard `Pillow` with `pillow-simd` in requirements.txt
- SIMD-accelerated (AVX2, SSE4) image operations
- Drop-in replacement, no code changes required

**Impact**: 4-10x faster image operations (resize, decode, color conversion)

**Affected operations**:
- Image decoding from bytes
- Resizing operations
- Color space conversions

### 3. Request Validation and Size Limits

**Implementation**:
Performance middleware in `src/main.py`:

```python
MAX_FILE_SIZE_MB = 50  # Adjust based on requirements
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    # Early validation - reject oversized files before processing
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"}
        )

    # Request timing
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    return response
```

**Impact**:
- Prevents DoS attacks
- Fast-fail for invalid requests
- Reduces memory exhaustion risk

### 4. Performance Monitoring Middleware

**Implementation**:
Automatic request timing and slow request detection:

```python
SLOW_REQUEST_THRESHOLD_MS = 100  # Log requests slower than this

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    # Add timing header
    response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

    # Log slow requests
    if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
        logger.warning(f"Slow request: {request.url.path} took {duration_ms:.2f}ms")

    return response
```

**Usage**:
```bash
# Check response time in headers
curl -I http://localhost:4603/detect
# Response includes: X-Process-Time: 23.45ms
```

### 5. Optimized Uvicorn Configuration

Tuned worker processes and connection handling in `docker-compose.yml`:

| Parameter | Value | Impact |
|-----------|-------|--------|
| `--limit-max-requests` | 10000 | Prevents memory leaks (worker recycling) |
| `--limit-max-requests-jitter` | 1000 | Avoids thundering herd |
| `--timeout-graceful-shutdown` | 30 | Clean restarts (drains connections) |
| `--loop` | uvloop | 2-3x faster event loop |
| `--http` | httptools | Faster HTTP parsing |

**Worker Tuning Formula**:
```
Workers = (2 × CPU cores) + 1

Examples:
- 8 cores → 17 workers
- 16 cores → 33 workers
- 32 cores → 65 workers
```

---

## gRPC Connection Management

### How gRPC Connections Work

Unlike HTTP/1.1 (one request per connection), gRPC uses HTTP/2 with:
- **Multiple concurrent streams** on one connection
- **Bidirectional streaming** (full duplex)
- **Header compression** (HPACK)
- **Flow control** per stream

```
HTTP/1.1 (Old):
Connection 1 → Request 1 (blocking)
Connection 2 → Request 2 (blocking)
...

gRPC/HTTP/2 (Modern):
Connection 1 → Stream 1, 2, 3, ..., 1000 (concurrent!)
```

### Single Connection Is Optimal

**Current Architecture:**
```
32 FastAPI Workers
    │
    └─▶ 1 Shared gRPC Client (HTTP/2 channel)
            │
            └─▶ 1 Triton Server (1 GPU)
                    │
                    └─▶ Dynamic Batching → GPU Processing
```

**Capacity Analysis:**

**Single gRPC Connection Limits:**
- Theoretical: ~2^31 concurrent streams (HTTP/2 spec)
- Practical: 10,000-100,000 concurrent requests
- Network bandwidth: 1-10 Gbps (local Docker network)

**System Limits (Actual Bottlenecks):**
- FastAPI: 32 workers × 512 concurrent = 16,384 max
- GPU: ~400-600 inferences/sec
- Triton: Queue depth 128 (config)

**Conclusion**: The gRPC connection can handle 10x more than the GPU can process.

### When You DON'T Need Multiple Connections

✅ Single Triton server
✅ 1-4 GPUs on one node
✅ <5,000 concurrent requests
✅ Local network (Docker, same datacenter)
✅ <1,000 RPS throughput

### When You DO Need Multiple Connections

**Scenario 1: Multiple Triton Servers (Horizontal Scaling)**

```python
# Multiple Triton instances (different URLs)
triton_servers = [
    "triton-1:8001",  # GPU 0
    "triton-2:8001",  # GPU 1
    "triton-3:8001",  # GPU 2
]

# Round-robin across servers
def get_triton_round_robin():
    import random
    server = random.choice(triton_servers)
    return get_triton_client(server)
```

**When**: >1000 RPS, multiple GPU nodes

**Scenario 2: High Concurrency (>10,000 requests)**

```python
class TritonConnectionPool:
    """Multiple connections to same Triton server."""

    def __init__(self, triton_url: str, pool_size: int = 4):
        self.clients = [
            InferenceServerClient(url=triton_url)
            for _ in range(pool_size)
        ]
        self.current = 0

    def get_client(self):
        """Round-robin across connections."""
        client = self.clients[self.current]
        self.current = (self.current + 1) % len(self.clients)
        return client
```

**When**: >10,000 concurrent requests

### Monitoring Connection Saturation

```bash
# Monitor active connections
watch -n 1 'docker compose exec yolo-api netstat -an | grep 8001 | grep ESTABLISHED'

# Monitor latency percentiles
# If P99 >1000ms with <5000 RPS = possible connection bottleneck
```

**Red Flags** (Connection Saturation):
- P99 latency >1000ms
- gRPC "stream limit reached" errors
- Connection refused errors
- Throughput plateaus despite more load

### Production Scaling Roadmap

**Phase 1: Current (1 GPU, <1000 RPS)**
```
✅ Single Triton server
✅ Single shared gRPC connection
✅ Dynamic batching enabled
```
**Capacity**: ~500-1000 RPS
**Bottleneck**: GPU processing power

**Phase 2: Multi-GPU Single Node (1-4 GPUs, <5000 RPS)**
```
Option A: Multiple Triton instances (1 per GPU)
  - Load balancer → 4 Triton servers
  - 4 shared connections (1 per server)

Option B: Single Triton with multiple models
  - 1 Triton, 4 model instances
  - 1 shared connection
  - Triton routes to available GPU
```
**Capacity**: ~2000-5000 RPS
**Bottleneck**: GPU memory, PCIe bandwidth

**Phase 3: Multi-Node (4+ GPUs, 5000+ RPS)**
```
Kubernetes with:
  - 4+ Triton pods (1 GPU each)
  - Service load balancer
  - Connection pool per FastAPI instance
  - Autoscaling based on queue depth
```
**Capacity**: 10,000+ RPS
**Bottleneck**: Network, orchestration overhead

---

## Benchmarking

### Using the Go Benchmark Tool

The repository includes `benchmarks/triton_bench.go` for testing.

#### 1. Baseline (Before Optimization)

```bash
# Record baseline metrics
cd benchmarks
go run triton_bench.go \
    --url http://localhost:4603/detect \
    --clients 50 \
    --requests 1000 \
    --image ../test_images/sample.jpg \
    > baseline_results.txt
```

#### 2. Rebuild with Optimizations

```bash
# Rebuild containers with new requirements
docker compose down
docker compose build --no-cache yolo-api
docker compose up -d

# Wait for warmup (~30 seconds)
sleep 30
```

#### 3. Optimized Benchmark

```bash
# Run same benchmark
cd benchmarks
go run triton_bench.go \
    --url http://localhost:4603/detect \
    --clients 50 \
    --requests 1000 \
    --image ../test_images/sample.jpg \
    > optimized_results.txt
```

#### 4. Compare Results

```bash
# Compare latency metrics
echo "=== BASELINE ==="
grep -A 5 "Latency" baseline_results.txt

echo "=== OPTIMIZED ==="
grep -A 5 "Latency" optimized_results.txt
```

### Recommended Test Matrix

Test with various concurrency levels:

```bash
for clients in 1 10 50 100 256; do
    echo "Testing with $clients concurrent clients..."
    go run triton_bench.go \
        --url http://localhost:4603/detect \
        --clients $clients \
        --requests 1000 \
        --image ../test_images/sample.jpg \
        > results_${clients}_clients.txt
done
```

### Key Metrics to Track

1. **Average Latency**: Should decrease 10-15%
2. **P95 Latency**: Should decrease 15-25% (better consistency)
3. **P99 Latency**: Should decrease 20-35% (fewer spikes)
4. **Throughput**: Should increase 15-20% (requests/sec)
5. **Error Rate**: Should remain 0%

---

## Profiling

### Using py-spy (Flamegraph Analysis)

#### Install py-spy in Container

Add to `requirements-dev.txt`:
```
py-spy>=0.3.14
```

Rebuild:
```bash
docker compose build yolo-api
docker compose up -d
```

#### Run Profiling Script

```bash
# Profile for 60 seconds (recommended during load test)
./scripts/profile_api.sh 60 profile_optimized.svg
```

#### Analyze Flamegraph

1. Open `profile_optimized.svg` in browser
2. Look for wide bars (expensive operations)
3. Check for:
   - ✅ Less time in JSON serialization
   - ✅ Less time in image decoding
   - ⚠️ Most time should be in GPU inference (expected)

#### Generate Load During Profiling

```bash
# Terminal 1: Start profiler
./scripts/profile_api.sh 60 profile.svg

# Terminal 2: Generate load
cd benchmarks
go run triton_bench.go \
    --url http://localhost:4603/detect \
    --clients 50 \
    --requests 500 \
    --image ../test_images/sample.jpg
```

### Using triton_bench (Comprehensive Load Testing)

Quick start:
```bash
cd benchmarks

# Quick validation (30 seconds, 16 clients)
./triton_bench --mode quick

# Full benchmark (60 seconds, 64 clients)
./triton_bench --mode full --clients 64 --duration 60

# High concurrency test (256 clients)
./triton_bench --mode full --clients 256 --duration 120

# Sustained throughput (auto-finds optimal client count)
./triton_bench --mode sustained
```

---

## Tuning Parameters

### Worker Count Optimization

Current: **32 workers** (assumes 16-core CPU)

**How to tune**:

1. Check CPU cores:
```bash
docker exec yolo-api nproc
```

2. Calculate optimal workers:
```
Workers = (2 × CPU cores) + 1
```

3. Update `docker-compose.yml`:
```yaml
- --workers=17  # For 8-core system
```

4. Restart:
```bash
docker compose restart yolo-api
```

**Signs you need fewer workers**:
- High memory usage (workers × model size)
- GPU contention (multiple workers fighting for GPU)
- CPU thrashing (too many context switches)

**Signs you need more workers**:
- Low CPU utilization (<50% during load)
- Request queueing (429 errors)
- High P99 latency (workers maxed out)

### File Size Limit

Current: **50MB** maximum upload size

**To adjust**:

Edit `src/main.py`:
```python
MAX_FILE_SIZE_MB = 100  # Increase to 100MB
```

Restart:
```bash
docker compose restart yolo-api
```

### Slow Request Threshold

Current: **100ms** (logs requests slower than this)

**To adjust**:

Edit `src/main.py`:
```python
SLOW_REQUEST_THRESHOLD_MS = 50  # More aggressive logging
```

**Useful for**:
- Development: Set to 50ms for detailed analysis
- Production: Set to 200ms to reduce log noise

---

## Troubleshooting

### Issue: No Performance Improvement

**Possible Causes**:

1. **GPU is the bottleneck** (expected!)
   - Compare different endpoints
   - Solution: Focus on model optimization (TensorRT)

2. **Not using optimized libraries**
   ```bash
   # Verify orjson is installed
   docker exec yolo-api python -c "import orjson; print('orjson OK')"

   # Verify pillow-simd is installed
   docker exec yolo-api python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
   ```

### Issue: Increased Memory Usage

**Cause**: Worker recycling not happening

**Solution**: Verify in `docker-compose.yml`:
```yaml
- --limit-max-requests=10000
- --limit-max-requests-jitter=1000
```

**Monitor**:
```bash
# Check memory usage
docker stats yolo-api

# Should see periodic drops as workers recycle
```

### Issue: Slow Requests Still Occurring

**Debug Steps**:

1. Check logs for slow request warnings:
```bash
docker compose logs -f yolo-api | grep "Slow request"
```

2. Profile during slow requests:
```bash
./scripts/profile_api.sh 30 slow_profile.svg
```

3. Check if GPU is the bottleneck:
```bash
# GPU utilization should be near 100%
nvidia-smi dmon -s u
```

### Issue: Connection Errors

**Symptom**: `429 Too Many Requests` or connection refused

**Cause**: Hit concurrency limit

**Solutions**:

1. Increase concurrency limit in `docker-compose.yml`:
```yaml
- --limit-concurrency=1024  # Increased from 512
```

2. Increase backlog:
```yaml
- --backlog=8192  # Increased from 4096
```

3. Add more workers (if CPU/memory available)

---

## Performance Monitoring Dashboard

### Health Endpoint

Use the `/health` endpoint for monitoring:

```bash
# Quick check
curl -s http://localhost:4603/health | python -m json.tool

# Monitor memory over time
watch -n 5 'curl -s http://localhost:4603/health | jq ".performance.memory_mb"'

# Check optimization status
curl -s http://localhost:4603/health | jq ".performance.optimizations"
```

### Integration with Prometheus/Grafana

Your Prometheus + Grafana setup can scrape these metrics:

1. Create `/metrics` endpoint (optional enhancement):
```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('api_requests_total', 'Total requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')
```

2. Add to Prometheus config:
```yaml
- job_name: 'yolo-api'
  static_configs:
    - targets: ['yolo-api:4603']
```

---

## Summary

### Quick Checklist

✅ **Optimizations Applied**:
- orjson for JSON (2-3x faster)
- pillow-simd for images (4-10x faster)
- Request size limits (prevents DoS)
- Performance monitoring (tracks latency)
- Optimized Uvicorn config (better throughput)
- Enhanced health check (observability)

✅ **Testing**:
- Run baseline benchmark
- Rebuild containers
- Run optimized benchmark
- Compare results (expect 10-15% improvement)
- Profile with py-spy
- Load test with triton_bench

✅ **Tuning**:
- Adjust worker count for your CPU
- Set appropriate file size limits
- Configure slow request threshold
- Monitor memory usage

### Expected Results

- ✅ **10-15% latency reduction** (total end-to-end)
- ✅ **15-20% throughput increase**
- ✅ **Better P99 latency** (fewer spikes)
- ✅ **Lower memory usage**

Remember: **GPU inference is still the bottleneck** (60-70% of total time). These optimizations maximize API efficiency!

---

**Last Updated**: 2026-01-26
**Version**: 2.0 (Consolidated documentation)
