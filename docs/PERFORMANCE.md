# Performance Optimization Guide

Complete guide for optimizing FastAPI and Triton Inference Server performance.

> **Status (v0.5.0):** this guide covers the API-layer tuning and benchmarks that are in
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
policy rows above are computed, not measured. The v0.5.0 re-measurement on the pinned
2,000-image set is in "v0.5.0 baseline" below.

### Measured VLM labeling speed

One run, repeated once, on a 200-crop public COCO project: `POST /vlm/label_batch`,
32 crops per request, 8 client threads, the local default VLM served by vLLM on
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

### Baseline 0.4.1 (set A, public COCO, 4,000 images; commit 15f7bbb9)

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

### v0.5.0 baseline (set A, public COCO, 2,000 images)

The Wave 0 baseline of `docs/design/triton_pipeline_optimization_plan.md`, measured on the
published v0.5.0 images before any optimization (issues #45 and #40). Raw JSON:
`docs/benchmarks/v050_baseline.json` (environment, ingest, Triton statistics,
perf_analyzer points, endpoints, VLM) and `docs/benchmarks/v050_storage.json` (the
storage run); idle-CPU readings in `docs/benchmarks/v050_idle_background.json`. (The VLM model label in the raw JSON metrics is replaced by `catalog-default`.) Every
number below is a median with the minimum and maximum of 3 repetitions in parentheses,
unless stated.

**Headline.** A 2,000-image public COCO set goes through detector plus embeddings at
**8.9 images/s** through `POST /curation/projects/{project}/ingest/upload` and
**9.2 images/s** through `.../ingest/batch`, with the
GPU 91 percent busy. The run-to-run spread is under 6 percent. 90 percent of the GPU time is
the PE image encoder (7.4 embeddings per image). That engine is FP32, and a fresh v0.5.0
install builds the same FP32 engine (see Caveats), so this is the true "before" for the
default install.

#### Environment

| Item | Value |
|---|---:|
| GPU (index 0) | NVIDIA RTX A6000, driver 615.71.09, CUDA 13.4 |
| CPU | Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz, 48 logical cores |
| RAM | 504 GB |
| Stack | OpenProcessor 0.5.0, Triton 2.70.0, OpenSearch 3.6.0, libnvinfer.so.11 |
| Harness commit | 877f62c9 |
| Dataset | 2000 images, 333 MB, seed 20261009, manifest edde8ae3d8cecfb2 |
| Foreign GPU 0 load before the runs | 0.0 % SM mean over 30 samples |
| Started | 2026-10-10T02:10:58.768502+00:00 |

Software under test: the images pinned by `images.lock` (API `openprocessor:0.5.0`
`sha256:b0af838e...`, Triton `openprocessor-triton:0.5.0` `sha256:1ac56ac9...`,
OpenSearch `3.6.0`), installed with `setup-openprocessor.sh --version v0.5.0 --tiers
core,curation --local-only --bind 127.0.0.1 --skip-models` into an isolated compose project on its own
ports, GPU 0 only. The VLM worker was stopped for the measured ingest runs. OpenSearch
heap 8 GB, no replicas, one node. Models: TensorRT engines built on 2026-09-26 and copied
from an earlier v0.5.0 install on the same host. The PE image encoder engine is FP32
(1.27 GB); every other engine is consistent with FP16 by file size.

#### Method

```bash
# 1. the pinned set (checksum-verified; 2,000 images, 333 MB, in data/samples/, gitignored)
python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_bench_2000 \
  --bench-set 2000 --seed 20261009 --manifest scripts/datasets/manifests/coco_bench_2000.json
python scripts/datasets/fetch_coco_subset.py --verify-only --out data/samples/coco_bench_2000 \
  --manifest scripts/datasets/manifests/coco_bench_2000.json
# 2. everything else, one entry point; the JSON is merged per phase
python scripts/bench/baseline_suite.py --manifest scripts/datasets/manifests/coco_bench_2000.json \
  --images data/samples/coco_bench_2000/images --out artifacts_local/bench/v050_baseline.json \
  --project <compose project> --api-url http://127.0.0.1:<api> --triton-url http://127.0.0.1:<triton> \
  --opensearch-url http://127.0.0.1:<opensearch> --source-map <host images dir>=<same dir in the API container> \
  --phases env,ingest,triton,endpoints
python scripts/bench/suite_report.py artifacts_local/bench/v050_baseline.json   # the tables below
```

- The pin draws 1,279 images from val2017 (all that carry an allowed license: Attribution,
  Attribution-ShareAlike, no known restrictions, US Government Work) and 721 from
  train2017. 10,000 and larger sets come from the same command with a larger `--bench-set`
  (only train2017 can supply 50k and 100k); only the 2,000-image pin is committed.
- Ingest: 32 images per request, 4 client threads, ingest policy `all` (the default: whole-image
  vector, all 80 detector classes, one vector per detection). A 100-image warm-up project is ingested and
  deleted first. Each repetition uses a fresh project, runs `train_clusters` auto-label
  afterwards (its wall time is in the table), then settles OpenSearch and deletes the
  project. Upload and batch repetitions are interleaved (upload, batch, upload, ...).
- Triton: `/v2/models/stats` read before and after each repetition; per-execution times come from
  the batch statistics (Triton's per-request compute figures credit every request in a batch with
  the whole execution and overcount). Wire bytes are inference counts times the tensor sizes
  in the model config. perf_analyzer runs from the 26.06 SDK container on the stack network,
  gRPC, `--measurement-interval 5000 --stability-percentage 10`, concurrency 1, 6, 11, 16.
- GPU: `nvidia-smi` utilization and memory once a second, plus `nvidia-smi pmon` SM share
  split into the stack's processes and everyone else's (foreign) by pid.
- CPU seconds are cgroup `usage_usec` deltas of the API and Triton containers around the
  ingest window.

#### Ingest, end to end

| Route | images/s | wall s | items/image | request p50 s | API CPU s/image | Triton CPU s/image | cluster training s | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `/ingest/upload` | 8.93 (8.89-9.13) | 224 (219-225) | 6.39 (6.39-6.39) | 14.1 (13.7-14.8) | 0.490 (0.299-0.954) | 0.031 (0.030-0.032) | 88 (68-125) | 0 |
| `/ingest/batch` | 9.19 (8.94-9.45) | 218 (212-224) | 6.39 (6.39-6.40) | 14.1 (13.5-14.7) | 0.358 (0.270-0.360) | 0.031 (0.030-0.032) | 78 (68-80) | 0 |

Per-stage timers of the API (`op_pipeline_stage_*`, `/ingest/upload`; seconds are summed
across concurrent requests, so they are not wall time):

| Stage | calls/image | mean ms/call | summed ms/image | KiB/image |
|---|---:|---:|---:|---:|
| crop | 6.39 (6.39-6.39) | 0.1 (0.1-0.1) | 0.6 (0.6-0.6) | n/a |
| decode | 1.00 (1.00-1.00) | 4.6 (4.5-4.7) | 4.6 (4.5-4.7) | 163 (163-163) |
| embed | 2.07 (2.07-2.07) | 2304.0 (1954.2-2332.0) | 4763.5 (4040.2-4821.4) | 9783 (9783-9783) |
| jpeg_encode | 6.39 (6.39-6.39) | 0.5 (0.5-0.5) | 3.3 (3.1-3.4) | 58 (58-58) |
| opensearch_write | 1.99 (1.99-1.99) | 203.4 (199.7-276.3) | 404.5 (397.1-549.5) | n/a |
| resize | 0.99 (0.99-0.99) | 24.4 (23.5-24.9) | 24.1 (23.2-24.6) | n/a |

Triton per model, `/ingest/upload` (`infer ms/image` = execution infer time x executions / images, the
GPU time one image costs; the three exec columns are per execution and overlap):

| Model | inferences/image | mean batch | queue ms/request | exec input ms | exec infer ms | exec output ms | infer ms/image | request KiB/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pe_image_encoder` | 7.39 (7.39-7.39) | 21.3 (20.8-22.4) | 1514 (1139-1523) | 0.1 (0.1-0.2) | 292.0 (285.3-307.2) | 285.8 (279.8-302.3) | 101.3 (101.2-101.4) | 9783 (9783-9783) |
| `yolov11_small_trt_end2end` | 1.00 (1.00-1.00) | 16.0 (16.0-16.1) | 1 (0-1) | 20.8 (19.9-21.0) | 78.0 (72.3-104.9) | 0.4 (0.4-0.7) | 4.9 (4.5-6.6) | 4800 (4800-4800) |

PE executions run at a mean batch of 21.3 (15 percent at the maximum of 32); the detector always runs at 16.
Estimated bytes on the wire per image: client to API 166,684 B for `/ingest/upload` (the JPEG plus
multipart framing) and 91 B for `/ingest/batch` (paths only; the API reads the file); API to Triton 10.0 MB
(PE) + 4.9 MB (detector) of FP32 tensors; responses to the client about 0.5 KB.

GPU 0 during the `/ingest/upload` repetitions:

| Ingest mode | GPU util mean % (nvidia-smi) | GPU memory peak MiB | own SM % (pmon) | foreign SM % (pmon) |
|---|---:|---:|---:|---:|
| `upload` | 91 (90-92) | 33020 (33020-33020) | 89 (89-91) | 0.0 (0.0-0.0) |

Foreign load: another project keeps 16.4 GB resident on GPU 0 (`pmon` showed 0.0 percent
foreign SM in all 6 measured repetitions and in 30 idle samples before them).

Cross-checks: a first run with the VLM worker still polling (6 repetitions, older statistics
code) gave 8.92 (8.70-8.98) images/s for upload and 8.92 (8.74-8.95) for batch; the storage run
below (4 repetitions) gave 9.30 and 9.15. Upload and batch are the same pipeline, so the
two routes agree within the spread.

#### Cluster training

`train_clusters` auto-label after each ingest, 12.8k items of 1,024 dimensions, IVF on CPU: 88 (68-125) s for upload and
78 (68-80) s for batch (almost all in `cluster_residuals`). That is 30 to 55 percent of the ingest wall time
(215-225 s). The spread is large; the cluster-refresh worker also fires every 200 new crops during the ingest.
The region stage was not measured: it needs the segmenter tier (gated weights) and a VLM
alongside Triton, which does not fit the 33 GB GPU 0 leaves free.

#### Storage per image and per vector

Measured in a separate run (2 repetitions per route) that merges until no deleted documents
remain and the size repeats; the first run measured too early and saw up to 2x inflation from
the cluster-refresh worker still rewriting documents. Reference: `docs/design/storage_sizing_and_ingest_baselines.md`.

| Route | bytes/image | bytes/vector | vs 8.35 KB reference |
|---|---:|---:|---:|
| `upload` | 65424 (65206-65642) | 8846 (8817-8876) | 1.06x |
| `batch` | 65298 (65230-65366) | 8830 (8820-8839) | 1.06x |

Details (upload, settled): images index 8,758 B per image, items index 8,826 to 8,894 B per
item (12,791 items, 6.4 per image, one vector each), 14,791 vectors in total, 65.4 KB per image.
The formula of the storage document (`8.7 KB + K x 10.4 KB`) predicts 75 KB for K = 6.4; the
measurement is 0.87x of that because this default pipeline stores no region boxes (about 1 KB of
metadata per item less). Per vector the cost is 1.06x the 8.35 KB reference, so the reference holds.

#### Triton model ceilings (perf_analyzer)

Inferences per second counts samples, not requests. Latency columns are at concurrency 1.

| Model | batch | infer/s at concurrency 1 | p50 ms at concurrency 1 | best infer/s | at concurrency |
|---|---:|---:|---:|---:|---:|
| `arcface_w600k_r50` | 1 | 52 | 18.8 | 2545 | 16 |
| `arcface_w600k_r50` | 8 | 1379 | 5.5 | 5238 | 16 |
| `arcface_w600k_r50` | 16 | 1762 | 8.4 | 5912 | 16 |
| `arcface_w600k_r50` | 32 | 1970 | 15.1 | 5948 | 16 |
| `mobileclip2_s2_image_encoder` | 1 | 24 | 40.2 | 551 | 16 |
| `mobileclip2_s2_image_encoder` | 8 | 258 | 28.7 | 897 | 16 |
| `mobileclip2_s2_image_encoder` | 16 | 312 | 47.8 | 905 | 11 |
| `mobileclip2_s2_image_encoder` | 32 | 372 | 82.7 | 913 | 11 |
| `pe_image_encoder` | 1 | 26 | 35.3 | 74 | 16 |
| `pe_image_encoder` | 8 | 64 | 120.9 | 77 | 6 |
| `pe_image_encoder` | 16 | 60 | 260.4 | 76 | 6 |
| `pe_image_encoder` | 32 | 62 | 516.5 | 78 | 6 |
| `scrfd_10g_bnkps` | 1 | 42 | 18.1 | 298 | 16 |
| `yolov11_small_trt_end2end` | 1 | 23 | 39.4 | 303 | 16 |
| `yolov11_small_trt_end2end` | 8 | 111 | 62.3 | 502 | 11 |
| `yolov11_small_trt_end2end` | 16 | 123 | 120.0 | 353 | 6 |
| `yolov11_small_trt_end2end` | 32 | 139 | 216.8 | 267 | 6 |

PE tops out at 74 to 78 inferences/s whatever the batch size or concurrency: it is compute
bound, and one `/ingest` image needs 7.39 of them. MobileCLIP (the `/embed/image` model) is
12x faster per sample.

#### Single-image endpoints

200 images, `/detect`, `/embed/image`, `/faces/detect`, `/ocr/predict`, 1 and 4 client threads,
3 repetitions. `/embed/image` at one thread has a cold first repetition (14 images/s; 128 warm).

| Endpoint | client threads | images/s | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| `detect` | 1 | 10.7 (10.6-10.7) | 94 (93-94) | 105 (104-105) |
| `detect` | 4 | 40.6 (39.2-40.9) | 98 (98-98) | 110 (110-112) |
| `embed_image` | 1 | 116.0 (14.1-127.7) | 8 (7-70) | 11 (11-80) |
| `embed_image` | 4 | 148.7 (147.4-165.4) | 10 (9-10) | 80 (77-80) |
| `faces_detect` | 1 | 13.6 (12.4-13.7) | 68 (66-79) | 99 (98-116) |
| `faces_detect` | 4 | 51.4 (51.1-51.8) | 74 (70-76) | 113 (112-113) |
| `ocr_predict` | 1 | 7.4 (6.4-7.5) | 116 (116-142) | 247 (236-282) |
| `ocr_predict` | 4 | 19.9 (19.5-19.9) | 166 (163-176) | 430 (418-442) |

`/detect` takes 94 ms at the API for a detector inference of about 5 ms: 95 percent of a single
request is host work (decode, letterbox, FP32 tensor, gRPC).

#### VLM labeling

`POST /curation/projects/{project}/vlm/label_batch`, 200 crops per repetition (a different 200 each
time), 32 crops per request, 8 client threads, the catalog default model (bfloat16) on
vLLM with GPU memory utilization 0.4 on GPU 0, Triton stopped for this run so it fits next to the 16 GB
of the other project: **10.0 crops/s** (9.9-10.1), 0 errors, GPU 0 utilization
83 percent. This reproduces the 10.0 crops/s measured earlier on 0.5.0. Re-labeling crops the model has
already seen is much faster (27 to 31 crops/s) because of vLLM's multimodal cache, so vary the crops.

#### Caveats

- **The PE engine is FP32, and the FP16 build does not work on this TensorRT release.** The
  `pe_image_encoder` plan is 1.27 GB, the size of the FP32 ONNX. The installer's model step bakes
  FP16 into the ONNX first (`trt_utils.py` reports `fp16` and writes a 636 MB ONNX), then runs
  `trtexec` with the profile `images` min 1x3x336x336, opt 8, max 32, workspace 8G, `--skipInference`
  (`scripts/lib/model_setup.sh`, `_ms_pe_trtexec`). On this stack (TensorRT 11.1 in the Triton 26.06
  image) that build fails at parse time on both `Einsum` nodes: `IEinsumLayer must have all
  inputs of same type. Input 1 has type Half and input 0 has type Float` (`/visual/Einsum`: the FP16
  graph rewrite converted its float `Constant` to half but left the explicit float32 `Cast` that
  feeds the other input). The installer then retries from the FP32 ONNX, which builds. A fresh v0.5.0
  install therefore ends up with the same FP32 engine as this baseline, so the baseline is the true
  "before" for the default install. The fix and its measured effect are in the next section.
- OpenSearch data and the upload store live on a RAID array (Docker data root), not on NVMe.
  This affects `opensearch_write` and the upload route's persistence; treat them as upper bounds.
- The VLM worker, polling a project that has data while no VLM is configured, burned about
  2.7 cores while idle (18 projects of 2,000 images: 0.66 in the worker, 1.2 in the API,
  0.85 in OpenSearch), so it was stopped for the throughput runs. Throughput did not change
  (the GPU is the limit) but API CPU per image roughly doubled with it running.
- Face and OCR models are measured only through their endpoints and perf_analyzer; ingest does not
  call them. OCR recognition and detection (dynamic shapes) have no perf_analyzer points.
- Closed loop, 4 threads x 32 images: Triton queue times (1.1 to 1.5 s for PE) are backlog, not a
  configured delay. A concurrency sweep was not run.
- Python harness and Triton share the host with other projects; the host load average was
  4 to 5 before the runs. The foreign GPU load was zero at every sample.
- COCO frames are 640 x 480, so costs that scale with pixels (decode, full-size crops) are
  small here. The private high-resolution set B was not run.
- Counts of 2,000 images only; the 10k pin and larger are not measured.

### v0.6.0 Wave 1: FP16 PE engine (set A, public COCO, 2,000 images)

The first change of `docs/design/triton_pipeline_optimization_plan.md` after the baseline above:
the PE image encoder is built as a real FP16 engine. Same pinned set, same harness
(`scripts/bench/baseline_suite.py --phases env,ingest`), same host and GPU 0, same published v0.5.0
images; the only difference is the engine behind `pe_image_encoder`. Raw JSON:
`docs/benchmarks/v060_fp16_pe.json` (ingest, Triton statistics, GPU) and
`docs/benchmarks/v060_fp16_pe_engine.json` (engine size, build time, parity). Medians with the
minimum and maximum of 3 repetitions in parentheses.

**Headline.** FP16 doubles ingest throughput: **17.4 images/s** through `/ingest/upload`
(8.9 before) and **18.8 images/s** through `/ingest/batch` (9.2 before), with the PE engine at
**5.5 ms per embedding** (13.7 before) in a plan of half the size (645 MB against 1.27 GB). Embeddings
stay equivalent to the FP32 engine (cosine similarity 0.9996 mean, 0.9976 minimum on 200 images).

#### Root cause and fix

The v0.5.0 FP16 bake (`bake_fp16_onnx` in `export/trt_utils.py`) never produced a buildable graph on
TensorRT 11.1, and the installer hid it by retrying from FP32. Two defects, both found by building
the baked ONNX with `trtexec`:

1. `onnxconverter-common` retypes tensors and initializers but never the `to` of a `Cast` already in
   the graph. PE computes its rotary-embedding table in float32 (`.float()` in the model code, an
   explicit `Cast(to=FLOAT)`), so the rewrite left an FP32 island whose consumers mix float32 with
   the FP16 weights. TensorRT 11.1 rejects mixed input types at ONNX parse: first at the two `Einsum`
   nodes (the converted half `Constant` against the float `Cast`), then, once `Einsum` was held
   in FP32, at the `Mul` and `Concat` nodes that consume the same table (221 nodes with mixed
   operands; `ElementWiseOperation PROD must have same input types`,
   `/visual/transformer/resblocks.0/attn/Mul`).
2. `Einsum` itself is a precision question: it multiplies position by frequency, where FP16 loses
   angle precision.

The fix keeps `Einsum` in FP32 (the converter inserts the casts around it) and adds a reconcile pass
after the rewrite that re-derives every tensor type from the nodes and casts mixed operands of
`Add`, `Concat`, `Div`, `Einsum`, `Equal`, `Gemm`, `MatMul`, `Mul`, `Sub`, `Where` and similar ops to
FP16 (FP32 for ops held in FP32). On PE it inserts 127 casts. It runs for every model baked by this
function; the other baked models are unchanged except the MobileCLIP image encoder (12 casts, see
Caveats).

#### Engine and parity

| Item | FP32 (v0.5.0) | FP16 (this change) |
|---|---:|---:|
| Plan size | 1,272,018,292 B | 644,778,380 B (0.51x) |
| `trtexec` engine build | 38 s | 81 s |
| Installer step wall (bake, parse, build) | 107 s (bake, failed FP16 parse, FP32 retry) | 142 s |

Parity, FP16 plan against the FP32 plan, same Triton, same inputs, 200 images (the first 200 by file
name of the pinned set, canonical PE whole-frame preprocessing):

| Metric | Value |
|---|---:|
| Cosine similarity, minimum | 0.9976 |
| Cosine similarity, mean | 0.9997 |
| Cosine similarity, 1st percentile | 0.9984 |
| Top-1 nearest neighbour agreement (200 queries, each against the other 199; FP16 gallery against FP32 gallery) | 97.0 % |
| Top-1 agreement, FP16 query against the FP32 gallery | 99.0 % |
| Top-5 neighbour overlap, mean | 97.5 % |

The Wave 1 accuracy gates of section 7 of the plan (retrieval and clustering on the labelled sets)
are not run here; this is an engine parity check on public images only.

#### Ingest, FP32 against FP16

| Metric | FP32 baseline | FP16 | Change |
|---|---:|---:|---:|
| `/ingest/upload` images/s | 8.93 (8.89-9.13) | 17.41 (16.44-17.97) | 1.95x |
| `/ingest/batch` images/s | 9.19 (8.94-9.45) | 18.84 (17.36-19.50) | 2.05x |
| Wall s, `/ingest/upload` (2,000 images) | 224 (219-225) | 115 (111-122) | 0.51x |
| Request p50 s, `/ingest/upload` | 14.1 (13.7-14.8) | 7.2 (6.9-7.3) | 0.51x |
| PE `infer ms/image` (7.4 embeddings) | 101.3 (101.2-101.4) | 41.0 (40.8-41.2) | 0.40x |
| PE ms per embedding | 13.7 | 5.5 | 0.40x |
| PE exec infer ms (per execution) | 292.0 (285.3-307.2) | 75.2 (69.0-85.3) | |
| PE mean batch | 21.3 (20.8-22.4) | 13.5 (12.5-15.5) | |
| PE queue ms per request | 1514 (1139-1523) | 214 (168-265) | |
| GPU util mean %, `/ingest/upload` (`nvidia-smi`) | 91 (90-92) | 74 (65-74) | |
| GPU own SM %, `/ingest/upload` (`pmon`) | 89 (89-91) | 71 (68-72) | |
| GPU memory peak MiB | 33020 | 30457 | |
| API CPU s/image, `/ingest/upload` | 0.490 (0.299-0.954) | 0.188 (0.186-0.204) | |
| API CPU s/image, `/ingest/batch` | 0.358 (0.270-0.360) | 0.177 (0.163-0.180) | |
| Triton CPU s/image | 0.031 | 0.035 | |
| Cluster training s, `/ingest/upload` | 88 (68-125) | 159 (153-169) | |
| Failed items | 0 | 0 | |

PE is still the largest GPU item at 41 of about 43 ms per image (the detector model is not
changed; its execution time here, 37 ms per batch of 16 against 78 ms, includes time-slicing against
PE on the same card). The pipeline is no longer GPU-bound: the GPU is 74 percent busy and PE batches
are smaller (13.5 against 21.3), so the host side now feeds it more slowly than it drains. The
per-image stage timers show decode (4.9 ms) and letterbox resize (23.7 ms) unchanged, and the
OpenSearch write latency summed per image higher (405 to 680 ms, the same writes arriving twice as
fast). That is the "API CPU outside the timed stages" item of section 11.1 of the plan becoming the
next limit; it is not measured here. The API CPU per image reading of the baseline varied from 0.30 to
0.95 s, so its drop (0.49 to 0.19 s) is not evidence of a saving. Cluster training (CPU, IVF) is
outside this change; its slower reading here coincides with host contention (load average 9 to 15
from other projects, against 4 to 5 for the baseline) and is not part of the images/s figure.

The earlier private FP16 measurement of about 170 embeddings/s (5.9 ms) matches the 5.5 ms here.

#### Caveats

- One host, one card, 3 repetitions; the host was busier than during the baseline (load average 9
  to 15 against 4 to 5). The foreign GPU load was zero at every sample, as before.
- Parity covers 200 public images on whole-frame preprocessing. Retrieval and clustering gates on
  labelled data (plan section 7) are open.
- The fix bakes FP16 for every model that calls `bake_fp16_onnx`. On TensorRT 11.1 the MobileCLIP
  image encoder's v0.5.0 bake also failed at parse (`ElementWiseOperation DIV must have same input
  types`); with the reconcile pass it parses, but `trtexec` then stops at
  `Could not find any implementation for node .../reparam_conv/Conv + PWN(...)`, so its exporter
  still falls back to FP32 (its existing, printed fallback). That is a separate TensorRT-side
  failure, tracked in #224; no other baked model has a mixed-type op.
- The installer's FP32 fallback is kept (an FP32 engine beats none) but is no longer silent: it
  prints a warning naming the model, the FP16 failure reason and the consequence, writes the model
  to `.install/precision.tsv`, records the group as `degraded` in `groups.tsv` (a re-run retries FP16
  instead of skipping), and `./openprocessor models status` lists it under "Degraded precision".

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
