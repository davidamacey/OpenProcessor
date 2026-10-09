# Performance Optimization Guide

Complete guide for optimizing FastAPI and Triton Inference Server performance.

> **Status (v0.4.1):** this guide covers the API-layer tuning and benchmarks that are in
> `main`. The GPU and Triton pipeline optimization (decode once, stay on the GPU) is
> open work, tracked in #40 and planned in
> [`design/triton_pipeline_optimization_plan.md`](design/triton_pipeline_optimization_plan.md);
> the numbers here are measured before it.

---

## Table of Contents

1. [Overview](#overview)
2. [FastAPI Optimizations](#fastapi-optimizations)
3. [gRPC Connection Management](#grpc-connection-management)
4. [Benchmarking](#benchmarking)
5. [Profiling](#profiling)
6. [Tuning Parameters](#tuning-parameters)
7. [Troubleshooting](#troubleshooting)
8. [Ingest cost per image](#ingest-cost-per-image)
9. [Baseline protocol](#baseline-protocol)

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
watch -n 1 'docker compose exec api netstat -an | grep 8001 | grep ESTABLISHED'

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

## Ingest cost per image

What a detector vocabulary and an embedding choice cost per 1,000 images.
Measured by: the live stack measurement of 2026-10-03 (project of 2,000 COCO
images, narrow vehicle detector) recorded in
`docs/design/generic_detector_and_selective_embedding_plan.md` (findings F12 and
section 4). The per-1,000 figures below are derived from those numbers with
`scripts/bench/ingest_cost_probe.py`; they are not a new run. A full re-measure
on the public 4,000-image COCO set is a later step.

Inputs: 1.37 stored items per image with the narrow detector, 7 per image with
the full 80-class vocabulary, about 1 KB of metadata per item (measured 0.85-1 KB), 8.4 KB per
1024-d vector (one per embedded item, one per image for the whole frame).

| Scenario (per 1,000 images) | Items | Total MB | vs narrow | Crop encoder calls |
|---|---:|---:|---:|---:|
| Narrow detector, every item embedded | 1,370 | 22.4 | 1.0x | 1,370 |
| Full vocabulary, every item embedded | 7,000 | 79.8 | 3.6x | 7,000 |
| Full vocabulary, no item embedded | 7,000 | 21.0 | 0.9x | 0 |
| Full vocabulary, only the narrow classes embedded | 7,000 | 32.5 | 1.45x | 1,370 |

Reading: keeping every detection costs about 3.6x the storage and 5x the crop
encoder and VLM calls of the narrow detector when each one gets a vector. Keeping
them without a vector costs about the same storage as the narrow setup, so
storage is the small number and encoder and VLM time is the real cost. The
detector itself runs once per image whatever the number of classes. Policy modes map onto those rows (computed from the same per-item sizes, not
measured): `all` is the "every item embedded" row for your vocabulary; `selected`
is the "only the narrow classes embedded" row; `lazy` is the "no item embedded"
row until you run the embed action, then it converges to `all` for the items you
chose. The ingest policy preview (`POST .../ingest/policy/preview`) gives the
same estimate for your stored data.

Reproduce
the table with your own per-image figures:

```bash
python scripts/bench/ingest_cost_probe.py --images 1000 --full-items 7
```

### Measured ingest speed

One run, not averaged, from `docs/design/storage_sizing_and_ingest_baselines.md`
section 6: `POST /ingest/batch`, 32 images per request, 4 client threads, one 48 GB
GPU for the detection models, API with 32 workers. A 2,000-image public COCO subset
ingested at **13.39 images/s** (0 failed, 1.37 crops per image, batch p50 10.0 s). A
4,000-image mixed set that is half 12 to 20 MP JPEGs ingested at 4.90 images/s
(about 15 for the small photos, about 2.5 for the large ones; decode and resize
dominate). Those runs used the narrow vehicle detector; the full-vocabulary
policy rows above are computed, not measured. The larger public baseline set (#45)
is not published yet.

### Measured VLM labeling speed

One run, repeated once, on a 200-crop public COCO project: `POST /vlm/label_batch`,
32 crops per request, 8 client threads, the local `gemma-4-e4b` model served by vLLM on
one RTX A6000 (GPU memory utilization 0.4, shared with Triton). 200 crops labeled in
20.0 s both times: **10.0 crops/s** (0 errors; vLLM saw 69 requests, about 3 crops per
request). The continuous VLM worker sustained a similar 9.8 crops/s average over a
1,277-crop scope run. Other models and GPUs will differ.

## Baseline protocol

Wave 0 of `docs/design/triton_pipeline_optimization_plan.md`: two fixed seeded
image sets, one command each, and a per-stage report. Run on a quiet host with
a dedicated GPU, three interleaved rounds per comparison (median and range),
and keep the raw JSON outside the repo. All paths below are arguments; nothing
has a default location.

Set A, public (4,000 COCO val2017 images, each image's license recorded):

```bash
python scripts/bench/select_baseline_set.py coco \
  --annotations <annotations_trainval2017.zip or instances_val2017.json, path or URL> \
  --images <local val2017 dir, or the val2017 image URL> --download-dir <dir for URL downloads> \
  --count 4000 --seed 42 --out <dir>/coco_4000.txt
python scripts/bench/run_baseline.py <dir>/coco_4000.txt --api-url http://<api-host>:<port> \
  --slug-prefix baseline-a --policy all --stages ingest,embed,region,vlm,cluster \
  --opensearch-url http://<opensearch-host>:<port> --out <dir>/a_before
```

Set B, private (4,000 photos of about 12 to 20 MP from a private archive, taken
round-robin across its first-level folders; decodable JPEG, min side 320 px,
at most 40 MB):

```bash
python scripts/bench/select_baseline_set.py local --root <archive dir> \
  --count 4000 --seed 42 --out <private dir>/set_b.txt
python scripts/bench/run_baseline.py <private dir>/set_b.txt --api-url http://<api-host>:<port> \
  --slug-prefix baseline-b --policy all --path-map <host archive dir>=<same dir inside the API container> \
  --out <private dir>/b_before
```

Compare two runs: `python scripts/bench/run_baseline.py --compare before.json after.json`.

The manifest is a text file: a `#` header (seed, count, bytes, per-source
counts, date, sha256) and one path per line. The sha256 covers the path lines,
and `run_baseline.py` refuses a manifest whose paths no longer match it. The
COCO run also writes `<manifest>.licenses.csv` (license id, name and URL per
image). `--api-prefix` (default `OP_API_PREFIX`, else `/curation`) selects the API mount. The batch ingest route reads each path inside the API container, so
the API must be able to see the manifest paths (`--path-map HOST=CONTAINER`
rewrites a prefix). The harness creates a new project per run and leaves it
in place; delete it when you are done. The first 100 images (`--warmup`) are
ingested but left out of the rates and the metric deltas.

Recorded in the JSON and markdown report (aggregates only, never image content
or paths): images/s, items/s, input MB/s, batch p50 and p99, failures; region,
embed (lazy policy), VLM and cluster stage wall time; GPU mean utilization and
peak memory per GPU from `nvidia-smi` (skipped if absent); the change in every
`op_*` series of the API, optionally the worker (`--worker-metrics-url`) and
the Triton metrics endpoint (`--triton-metrics-url`, whose `compute_input`
counters are the host-to-device time); store bytes per image after a refresh
and force-merge (`--opensearch-url`) and crop-cache bytes (`--crop-cache-dir`).

The per-stage timers are the `op_pipeline_stage_seconds` histogram and the
`op_pipeline_stage_bytes_total` counter, labelled `stage` = `decode`, `crop`,
`jpeg_encode`, `resize`, `embed`, `opensearch_write`. The compose file sets
`PROMETHEUS_MULTIPROC_DIR` on the API, so a scrape aggregates all workers.

Where numbers go: set A numbers go into this file, in a table with the commit,
GPU model and the Triton, TensorRT, driver and torch versions. Set B numbers
stay in private notes, and only an aggregate ratio or pass/fail is stated in
public text.

### Baseline v0.4.1 (set A, public COCO, 4,000 images; commit 15f7bbb9)

Host: single node, Triton + API on one A6000 (GPU 0, 32 API workers, multiprocess metrics on), PE image encoder + YOLO11s via Triton. Region profile off, open_vocab off, VLM/auto-label workers idle. Warmup 100 images excluded. One round per row (not the 3-round median).

| row | images/s | items/s | items/image | ingest wall s | cluster s | store B/image | GPU0 mean util % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A all (4,000 imgs, ingest+embed+cluster) | 4.25 | 32.1 | 7.6 | 918 | 172.7 | 75,410 | 30.4 |
| A lazy (first 1,000, ingest only) | 10.28 | 81.4 | 7.9 | 88 | n/a | 10,883 | 19.8 |
| A selected person,car,dog (first 1,000) | 6.10 | 48.2 | 7.9 | 148 | n/a | 67,583 | 44.3 |

Per-stage totals, A all (op_pipeline_stage_*; seconds are summed across concurrent workers, so they exceed wall time):

| stage | calls | sum s | mean ms | bytes |
|---|---:|---:|---:|---:|
| decode | 3,900 | 20.0 | 5.1 | 676 MB |
| resize | 3,876 | 95.0 | 24.5 | n/a |
| crop | 29,488 | 1.9 | 0.06 | n/a |
| jpeg_encode | 29,488 | 11.0 | 0.37 | 235 MB |
| embed | 8,165 | 6,012.6 | 736 | 45.2 GB |
| opensearch_write | 7,776 | 2,054.6 | 264 | n/a |

Triton compute_infer, A all: pe_image_encoder 1,590 s total for 33,388 inferences (3,120 execs, queue 1,680 s); yolov11_small 2.5 s for 3,900. Lazy policy still embeds one whole-image vector per image (900 embed calls on 900 images).

A second baseline on a set of real high-resolution (about 20 MP) photos was measured privately; its numbers are not published. It showed the same pipeline is decode and I/O bound at that image size (the GPU was mostly idle), which is what the GPU decode and crop-at-model-size waves in the optimization plan target. Single run per row; a 3-round median is planned before the optimization work starts.

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
docker compose build --no-cache api
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
docker compose build api
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
docker exec api nproc
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
docker compose restart api
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
docker compose restart api
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
   docker exec api python -c "import orjson; print('orjson OK')"

   # Verify pillow-simd is installed
   docker exec api python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
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
docker stats api

# Should see periodic drops as workers recycle
```

### Issue: Slow Requests Still Occurring

**Debug Steps**:

1. Check logs for slow request warnings:
```bash
docker compose logs -f api | grep "Slow request"
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
- job_name: 'api'
  static_configs:
    - targets: ['api:4603']
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
