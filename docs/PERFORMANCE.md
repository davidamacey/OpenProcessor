# Performance Optimization Guide

Complete guide for optimizing FastAPI and Triton Inference Server performance.

> **Status (v0.6.0):** this guide covers the API-layer tuning, the measured results of the GPU and
> Triton pipeline optimization milestone (summary table below; plan in
> [`design/triton_pipeline_optimization_plan.md`](design/triton_pipeline_optimization_plan.md)),
> and the benchmark protocol. The scale ladder at 10k and 100k images is in
> [Scale ladder: 10k and 100k](#scale-ladder-10k-and-100k); the 1M step is planned for the next release.

---

## Summary of the optimization milestone

Every headline result of the milestone, with the section that holds the method, the raw numbers and
the caveats. Nothing here is new: each figure is copied from the section in the last column (raw
JSON under `docs/benchmarks/`). Public data only: set A is the pinned 2,000-image COCO set, set B
the pinned Commons photos (12 to 24 MP). Ranges and the host conditions are in the sections.

| Result | Before | After | Source |
|---|---|---|---|
| PE image encoder engine, per embedding (FP16 engine) | 13.7 ms | 5.5 ms | [FP16 PE engine](#060-wave-1-fp16-pe-engine-set-a-public-coco-2000-images) |
| Ingest through `/ingest/upload`, set A, one card (FP16 engine) | 8.9 images/s | 17.4 images/s | [FP16 PE engine](#060-wave-1-fp16-pe-engine-set-a-public-coco-2000-images) |
| Request bytes to Triton per image, set A (uint8 inputs) | 14.94 MB per image | 3.73 MB per image | [uint8 inputs](#wp-21-uint8-inputs-for-the-detector-and-the-pe-image-encoder-214) |
| API CPU per image, set A, `/ingest/upload` (uint8 inputs) | 0.118 CPU-s | 0.062 CPU-s | [uint8 inputs](#wp-21-uint8-inputs-for-the-detector-and-the-pe-image-encoder-214) |
| Photo ingest, `/ingest/upload`, 8 / 16 / 32 / 64 clients (pixel work off the event loop) | photos 5.2 / 8.7 / 10.8 / 4.9 images/s | 21.5 / 27.3 / 23.6 / 24.1 images/s | [API CPU attribution and fixes](#060-wp-12-and-wp-13-api-cpu-attribution-and-fixes-210-211-239-241) |
| Ingest, set A, 1 x A6000, tuned detector engine and concurrency | 23.39 images/s | 28.81 to 29.07 images/s | [Stage 2 tuning](#stage-2-tuning-detector-engine-and-the-concurrency-sweep-226-259) |
| Ingest, set A, 2 x A6000 (data-parallel PE) | 1 card 23.39 images/s | 44.60 images/s (1.91x); A6000 + 3080 Ti 43.72 (1.87x); tuned 2 cards 55.08 to 56.37 | [Multi-GPU PE encoding](#multi-gpu-pe-encoding-228), [Stage 2 tuning](#stage-2-tuning-detector-engine-and-the-concurrency-sweep-226-259) |
| Photos (set B, 12 to 24 MP) ingest, detect + embed (GPU decode) | 27.0 images/s | 35.0 images/s | [Decode once and the GPU decode](#decode-once-and-the-gpu-decode-217) |
| Host CPU per photo at bulk (GPU decode) | 1.12 CPU-s | 0.34 CPU-s | [Decode once and the GPU decode](#decode-once-and-the-gpu-decode-217) |
| Cluster pass plus geometry stage, 10,000 crop vectors (IVF) | 176.4 s | 24.9 s | [Cluster pass plumbing](#cluster-pass-plumbing-fetch-and-write-back-218) |
| Cluster pass, 100,000 vectors (IVF, fetch + fit + write-back) | 310.6 s | 138.0 s | [Cluster pass plumbing](#cluster-pass-plumbing-fetch-and-write-back-218) |
| OpenSearch write ceiling, 100,000 real vectors, final mapping | 1 writer 473 documents/s | 4 writers 1,980; 8 writers 4,426 documents/s (target needs about 740) | [Stage 4](#stage-4-coalesced-ingest-writes-lucene-engine-write-ceiling-on-the-final-mapping-248-254-230) |
| Index build CPU, `ef_construction` 512 to 128 | 2.39 node CPU-s per 1,000 documents | 1.49 | [Stage 4](#stage-4-coalesced-ingest-writes-lucene-engine-write-ceiling-on-the-final-mapping-248-254-230) |
| Vector index size per item, faiss to Lucene engine | 8.6 KiB | 4.5 KiB | [Stage 4](#stage-4-coalesced-ingest-writes-lucene-engine-write-ceiling-on-the-final-mapping-248-254-230), [Vector store evaluation](#vector-store-evaluation-tuned-opensearch-against-milvus-qdrant-pgvector-and-lancedb-221-230) |
| PE engine: INT8 | | rejected: faster but fails the accuracy gates; FP16 stays the only precision | [INT8 evaluation](#int8-evaluation-of-the-pe-image-encoder-on-ampere-213) |
| System shared memory between the API and Triton | | not adopted: under 5 percent of API CPU | [shared memory](#wp-22-system-shared-memory-between-the-api-and-triton-215-measured-and-not-adopted) |
| gRPC ingest channel | | skipped: the decision rule said no, nothing built | [Ingest transports](#ingest-transports-263) |

---

## Table of Contents

0. [Summary of the optimization milestone](#summary-of-the-optimization-milestone)
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
- Routes with a response model return that model; FastAPI serializes it through Pydantic
  straight to JSON bytes (no `default_response_class`, which FastAPI now deprecates for orjson).
- The curation routes that return a hand-built wire dict (`response_model=None`) use
  `WireRoute` (`src/core/wire_json.py`), which renders with `orjson` and skips
  `jsonable_encoder`.
- `scripts/bench/serialization_bench.py` compares the old and new paths offline.

**Benchmark** (`.venv/bin/python -m scripts.bench.serialization_bench`, median ms per
in-process request, 30 runs; the old path is the former orjson response class):

| Case | Old | New |
|---|---|---|
| Review page, 100 items (wire dict) | 68.9 | 4.2 |
| `/embed/boxes`, 1000 boxes x 512 dims | 88.6 | 65.7 |
| `/ingest/batch`, 1000 detail rows | 6.0 | 5.9 |

Bodies are byte-identical in every case.

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
Performance middleware in `src/app/factory.py`:

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
slow_request_threshold_ms = 100  # settings value; log requests slower than this

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    # Add timing header
    response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

    # Log slow requests
    if duration_ms > settings.slow_request_threshold_ms:
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
policy rows above are computed, not measured. The 0.5.0 re-measurement on the pinned
2,000-image set is in "0.5.0 baseline" below.

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
`op_pipeline_stage_bytes_total` counter, labelled `stage`. The stages are `decode`,
`crop`, `jpeg_encode`, `pe_preprocess` (the PE crop resize and normalize, named `resize` before
0.6.0), `embed`, `opensearch_write`, and the work that ran untimed before WP-1.2 (#210):
`hash` (the file fingerprint), `persist` (the upload's content-addressed write), `letterbox`
(the detector letterbox and tensor), `triton_request` (building the Triton input),
`whole_frame_preprocess` (the whole-frame vector's decode and normalize), `blur` (the
Laplacian scores), `doc_build` (the images and items documents) and `bulk_serialize` (encoding
request bodies for OpenSearch). All but `embed` and `opensearch_write` time a synchronous
block, so on the event-loop thread their seconds are API CPU seconds. The compose file sets
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

### 0.5.0 baseline (set A, public COCO, 2,000 images)

The Wave 0 baseline of `docs/design/triton_pipeline_optimization_plan.md`, measured on the
published 0.5.0 images before any optimization (issues #45 and #40). Raw JSON:
`docs/benchmarks/v050_baseline.json` (environment, ingest, Triton statistics,
perf_analyzer points, endpoints, VLM) and `docs/benchmarks/v050_storage.json` (the
storage run); idle-CPU readings in `docs/benchmarks/v050_idle_background.json`. (The VLM model label in the raw JSON metrics is replaced by `catalog-default`.) Every
number below is a median with the minimum and maximum of 3 repetitions in parentheses,
unless stated.

**Headline.** A 2,000-image public COCO set goes through detector plus embeddings at
**8.9 images/s** through `POST /curation/projects/{project}/ingest/upload` and
**9.2 images/s** through `.../ingest/batch`, with the
GPU 91 percent busy. The run-to-run spread is under 6 percent. 90 percent of the GPU time is
the PE image encoder (7.4 embeddings per image). That engine is FP32, and a fresh 0.5.0
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
OpenSearch `3.6.0`), installed with `setup-openprocessor.sh --version 0.5.0 --tiers
core,curation --local-only --bind 127.0.0.1 --skip-models` into an isolated compose project on its own
ports, GPU 0 only. The VLM worker was stopped for the measured ingest runs. OpenSearch
heap 8 GB, no replicas, one node. Models: TensorRT engines built on 2026-09-26 and copied
from an earlier 0.5.0 install on the same host. The PE image encoder engine is FP32
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
  feeds the other input). The installer then retries from the FP32 ONNX, which builds. A fresh 0.5.0
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

### 0.6.0 Wave 1: FP16 PE engine (set A, public COCO, 2,000 images)

The first change of `docs/design/triton_pipeline_optimization_plan.md` after the baseline above:
the PE image encoder is built as a real FP16 engine. Same pinned set, same harness
(`scripts/bench/baseline_suite.py --phases env,ingest`), same host and GPU 0, same published 0.5.0
images; the only difference is the engine behind `pe_image_encoder`. Raw JSON:
`docs/benchmarks/v060_fp16_pe.json` (ingest, Triton statistics, GPU) and
`docs/benchmarks/v060_fp16_pe_engine.json` (engine size, build time, parity). Medians with the
minimum and maximum of 3 repetitions in parentheses.

**Headline.** FP16 doubles ingest throughput: **17.4 images/s** through `/ingest/upload`
(8.9 before) and **18.8 images/s** through `/ingest/batch` (9.2 before), with the PE engine at
**5.5 ms per embedding** (13.7 before) in a plan of half the size (645 MB against 1.27 GB). Embeddings
stay equivalent to the FP32 engine (cosine similarity 0.9996 mean, 0.9976 minimum on 200 images).

#### Root cause and fix

The 0.5.0 FP16 bake (`bake_fp16_onnx` in `export/trt_utils.py`) never produced a buildable graph on
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

| Item | FP32 (0.5.0) | FP16 (this change) |
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
  image encoder's 0.5.0 bake also failed at parse (`ElementWiseOperation DIV must have same input
  types`); with the reconcile pass it parses, but the build stopped at
  `Could not find any implementation for node .../reparam_conv/Conv + PWN(...)` (#224). The failing
  layers are the depthwise 7x7 stride-2 convs with a channel multiplier of 2 (the three
  `downsample/proj/proj.0/reparam_conv` convs) and the final 3x3 conv (640 groups, 1280 outputs):
  TensorRT 11.1 has no FP16 kernel for a grouped conv with one input channel per group and more
  outputs than groups. `bake_fp16_onnx` now leaves those four convs in FP32 and the engine builds as
  FP16 (see "MobileCLIP image encoder FP16" below). No other baked model has such a conv.
- The installer's FP32 fallback is kept (an FP32 engine beats none) but is no longer silent: it
  prints a warning naming the model, the FP16 failure reason and the consequence, writes the model
  to `.install/precision.tsv`, records the group as `degraded` in `groups.tsv` (a re-run retries FP16
  instead of skipping), and `./openprocessor models status` lists it under "Degraded precision".

### INT8 evaluation of the PE image encoder on Ampere (#213)

Measured question: does explicit-Q/DQ INT8 beat the FP16 PE engine on the RTX A6000 and the 12 GB
RTX 3080 Ti (both SM86), and do the vectors stay in the same space? **Result: no recipe ships.**
The activation-quantized (A8W8) engines are 1.2-1.4x faster but fail every vector-agreement gate by a
wide margin; the one recipe that passes the gates (weight-only INT8) gives no speed and no size gain.
FP16 stays the only PE precision; there is no INT8 option, build flag or config in the tree. Raw
numbers: `docs/benchmarks/int8_pe_eval.json`. The detector was not evaluated: it is 22 of the 406
GFLOP per image (PE is 384), so even a free speed-up there changes the GPU arithmetic per image by about 5 percent,
and the same Q/DQ route would have to clear a box-level gate first.

#### Method

- **Stack.** TensorRT 11.1.0.106 (strongly typed: no implicit calibration, no precision flags), NVIDIA
  Model Optimizer 0.47.0 for the Q/DQ insertion and calibration, FP16 as the high-precision type.
  Engines are built with `trtexec` (profile min 1 / opt 8 / max 32, 336 x 336) from the FP16-baked PE
  ONNX that `export/trt_utils.py` produces, and served as separate models in an isolated Triton
  (own container, loopback ports, one GPU each; nothing in `models/` or the live stack was touched).
- **Calibration set.** 1,000 crops from 300 COCO train2017 images that are not in the pinned 2,000
  (seed 20261011, license-filtered like the pin; overlap with the pinned set checked: 0). Boxes are
  COCO ground-truth boxes (shorter side at least 24 px, up to 6 per image), cut with the production
  crop rule and `resize_crop_rgb` / `normalize_chw`. The set and its manifest stay out of git.
- **Evaluation set.** The pinned 2,000 images: the 12,790 production crop rectangles of the FP16
  reference capture (`ref-fp16`) and the 2,000 whole frames, embedded through Triton by each engine.
  The reference for each GPU is the FP16 engine built from the same ONNX on that GPU. Each candidate
  is wrapped as a parity capture (clusters re-computed with k-means, K = 512, same seed, per
  vector set) and judged by `scripts/bench/parity.py compare --profile precision`, so the thresholds
  are exactly the ones every other precision change is held to. Noise floor: the FP16 engine built on
  the 3080 Ti against the one built on the A6000 passes every gate (crop cosine p01 0.99996, kNN
  overlap@10 0.992, cluster assignment 0.9992).
- **Recipes** (Q/DQ on `MatMul` only; Softmax, LayerNorm, GELU, the patch convolution and the
  attention score and context matmuls stay FP16 in every recipe):
  - `bmax`: per-channel weights, `max` calibration, all 98 linear MatMuls (QKV, output projection,
    MLP in both directions in all 24 blocks, the pooling head MLP, the projection).
  - `b`: as `bmax` with `entropy` calibration on the 1,000 crops.
  - `h2` / `c_e`: as `bmax` / `b` but blocks 0, 1, 22, 23, the pooling head and the final projection
    left in FP16 (`c_e` calibrated on 300 crops because the histogram stage is single-threaded).
  - `w8`: weight-only INT8 (per-channel INT8 weights, no activation Q/DQ), built by dropping the
    activation Q/DQ pairs of a `--dq_only` model.
  - Excluding only the MLP output projections (`mlp/c_proj`) gave an engine whose output is NaN for
    every input (the graph is finite under ONNX Runtime): not usable, not pursued.

#### Accuracy gates (candidate against the FP16 engine of the same GPU; pinned set)

Cosine is per vector; kNN overlap is at k = 10 on 1,000 queries; the purity and kNN-precision columns
are percentage points lost against COCO ground-truth classes (negative = gained).

| Recipe, GPU | crop cos p01 | crop cos median | frame cos p01 | frame cos median | kNN overlap@10 | near-dup agreement | cluster assign | purity drop | label vote | kNN precision drop | gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate | >= 0.995 | >= 0.999 | >= 0.995 | >= 0.999 | >= 0.95 | >= 0.995 | >= 0.98 | <= 0.5 | >= 0.97 | <= 0.5 | |
| FP16 3080 Ti vs FP16 A6000 (noise floor) | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.992 | 1.000 | 0.999 | 0.24 | 0.991 | 0.03 | 0 |
| `bmax`, A6000 | 0.736 | 0.938 | 0.805 | 0.927 | 0.524 | 0.999 | 0.671 | 0.37 | 0.847 | 0.82 | 8 |
| `bmax`, 3080 Ti | 0.735 | 0.937 | 0.805 | 0.927 | 0.524 | 0.999 | 0.670 | -0.49 | 0.849 | 0.75 | 8 |
| `b` (entropy), A6000 | 0.885 | 0.986 | 0.942 | 0.981 | 0.716 | 0.999 | 0.901 | -0.13 | 0.887 | 0.81 | 8 |
| `h2` (head and edge blocks FP16), A6000 | 0.929 | 0.980 | 0.928 | 0.976 | 0.694 | 0.999 | 0.879 | 0.00 | 0.896 | -0.23 | 7 |
| `c_e` (entropy, head and edge blocks FP16), A6000 | 0.977 | 0.992 | 0.945 | 0.984 | 0.820 | 0.999 | 0.955 | -0.54 | 0.926 | 0.36 | 7 |
| `w8` (weight-only), A6000 | 0.9990 | 0.9997 | 0.9984 | 0.9995 | 0.960 | 1.000 | 0.996 | 0.16 | 0.976 | -0.17 | 0 |

Reading it: the best A8W8 recipe (`c_e`) reaches a median crop cosine of 0.992 and a first percentile
of 0.977, against 0.999 and 0.995 required. The ground-truth task metrics (cluster purity, kNN class
precision) move by under a point either way (at most 0.82), so INT8 does not make the space useless; it makes it a
different space: neighbour identity changes (overlap 0.52-0.82, label vote 0.85-0.93), so stored
vectors and new vectors from an INT8 engine could not be mixed in one collection without a
re-embedding migration. The weight-only recipe stays inside the gates but only just on kNN overlap
(0.960 against 0.95), and TensorRT folds the INT8 weights back to FP16 constants: its plan is
644.8 MB, the same as FP16 (644.8 MB), and the speed is unchanged. Weight-only INT8 therefore
buys nothing on this stack.

#### Speed (two numbers, never compared with each other)

Engine only is `trtexec --loadEngine` (TensorRT 11 default CUDA graphs, no host transfers, 15 s
after a 1 s warm-up). Through Triton is `perf_analyzer` over HTTP against the isolated Triton with
the production dynamic-batching config (preferred 4, 8, 16, 32; 15 ms queue delay), random FP32
input, 6 s windows, batch 1, 8, 16, 32 at concurrency 1, 2, 4, 8, 16; saturation is the range at
concurrency >= 4 and batch >= 8 (the production request shape is a batch of crops, up to 32).
Images per second; ratio is INT8 over FP16 on the same GPU.

| Recipe, GPU | Engine only b8 | Engine only b32 | Through Triton, saturated | Plan size |
|---|---:|---:|---:|---:|
| FP16, A6000 | 190.0 | 186.5 | 189-192 | 644.8 MB |
| `bmax`, A6000 | 279.0 (1.47x) | 268.2 (1.44x) | 263-273 (1.39-1.43x) | 337.4 MB |
| `h2`, A6000 | 241.7 (1.27x) | 233.2 (1.25x) | not run | 393.6 MB |
| `c_e`, A6000 | 241.1 (1.27x) | 235.5 (1.26x) | 232-239 (1.22-1.25x) | 393.6 MB |
| FP16, 3080 Ti | 206.5 | 200.3 | 188-195 | 645.1 MB |
| `bmax`, 3080 Ti | 284.4 (1.38x) | 271.5 (1.36x) | 236-258 (1.25-1.36x) | 337.4 MB |
| `h2`, 3080 Ti | 256.1 (1.24x) | 246.3 (1.23x) | not run | 393.5 MB |

On SM86 the INT8 tensor cores do deliver: all-linear INT8 is 1.4x faster on the A6000 and 1.35-1.4x on
the 3080 Ti, near the 1.33x that a public ViT-L measurement on the same architecture reports (see the serving
backend trade study), and it saves about 300 MB of weights (relevant for the 12 GB card, but the
activation memory of 340-360 MB per engine is unchanged). But keeping the sensitive
layers in FP16, which is what the accuracy needs, takes the gain to 1.22-1.27x, under the 1.3x bar
the plan sets, and still fails the gates. The speed-up is real and the accuracy is not there.

#### Decision

| Plan rule | Observed | Action |
|---|---|---|
| p01 >= 0.995, all gates pass, >= 1.3x | p01 0.74-0.98, 7-8 gates fail; >= 1.3x only with the recipes that fail worst | Do not ship |
| p01 < 0.995 or cluster/kNN gates fail | all A8W8 recipes | Record the numbers (this section), no code |
| Changed space would need a migration | yes, kNN overlap 0.52-0.82 | Out of scope |

FP16 remains the default and the only PE precision. Nothing in `export/`, `models/`, the compose files
or the API changed.

#### What was not measured, and caveats

- The full ingest protocol through the API: the plan runs it for the best variant only, and no
  variant passed the gates. The Triton numbers above are the PE model alone; the end-to-end ingest
  rate also depends on the decode, detector, crop, serialization and write stages above, so a 1.4x PE
  speed-up would have moved ingest by less than 1.4x.
- The PE engines evaluated take FP32 normalized input (the graph before the uint8 input change); the
  input dtype changes transfer bytes, not the 98 quantized matmuls, so neither the accuracy nor the
  engine-only speed ratios depend on it. They were not repeated on the uint8-input graph.
- The detector, MobileCLIP, ArcFace and OCR models in INT8; a SmoothQuant or per-token-activation
  recipe, quantization-aware training or a learned rounding method (not in the ONNX flow of the
  tool used; they are the usual remedy for ViT activation outliers and are a research task, not a
  calibration choice); a VLM label-agreement run (the VLM does not consume vectors); a sample-count
  sweep for the entropy method (1,000 crops for `b`, 300 for `c_e`).
- Timing was taken while another project's process held 16 GB on GPU 0 and 6 GB on GPU 1 with no
  compute (0 percent utilization before and after every run); the A6000 is capped at 300 W, so
  its engine-only FP16 rate (190 images/s) is below the 3080 Ti's (206 images/s). Triton and
  `trtexec` figures were never compared with each other.
- Tooling findings, for anyone repeating it: ONNX Runtime in the evaluator image lacks the CUDA 12
  libraries its provider needs (the image carries CUDA 13 ones), so calibration falls back to the CPU
  unless they are supplied; either way the entropy method is bound by a single-threaded histogram
  stage (about 85 minutes for 1,000 crops, about 3.5 hours for 300 on the partially quantized graph) and
  the max method takes 5-15 minutes. `bmax` and `b` were quantized from the FP32 ONNX and converted by
  Model Optimizer, `h2`, `c_e` and `w8` from the production FP16 ONNX; the model's final `If` (squeeze of a one-row pooling output) breaks Model
  Optimizer's FP16 conversion and was inlined (its condition is always true for this model) for the
  quantization copy only.

To repeat: bake the FP16 ONNX (`python export/trt_utils.py pytorch_models/pe_image_encoder.onnx
OUT.fp16.onnx`), inline the `If`, run `python -m modelopt.onnx.quantization --onnx_path ... --quantize_mode
int8 --calibration_method {max,entropy} --calibration_data_path CALIB.npy --high_precision_dtype fp16
--op_types_to_quantize MatMul Gemm [--nodes_to_exclude REGEX ...]` on a `(N, 3, 336, 336)` array of
normalized crops, build with `trtexec` using the profile above, then capture and `parity.py compare`
as in the gates section below.

### Wave 0 parity and FP16 gates (set A, public COCO, 2,000 images)

Every 0.6.0 change is gated by `scripts/bench/parity.py` (WP-0.1, #208): it captures the per-stage
outputs of one ingest of the pinned set and compares two captures against the accuracy gates of
`docs/design/triton_pipeline_optimization_plan.md` section 7 and the cluster-agreement metric of
`docs/design/v060_wave_plan.md` section 2.1. This section describes the tool and records the gate run
that accepts the FP16 PE engine of the section above.

#### The tool

```bash
# Reference and candidate: same stack, same pinned set, one capture per configuration.
.venv/bin/python scripts/bench/parity.py capture --tag fp32 --images DIR_OF_PINNED_IMAGES \
    --api-url http://127.0.0.1:API_PORT --triton-url http://127.0.0.1:TRITON_HTTP_PORT \
    --opensearch-url http://127.0.0.1:OPENSEARCH_PORT --compose-project NAME \
    --engine-label fp32 --engine-file PATH/TO/model.plan \
    --annotations COCO_ANNOTATIONS_DIR --probe-images 200
.venv/bin/python scripts/bench/parity.py compare artifacts_local/bench/parity/fp32 \
    artifacts_local/bench/parity/fp16 --out docs/benchmarks/parity/fp32_vs_fp16
```

`capture` creates a fresh project, ingests the 2,000 images through `POST .../ingest/upload`
(embedding policy `all`), trains clusters once, and stores under `artifacts_local/bench/parity/<tag>/`:

| File | Content |
|---|---|
| `capture.json` | Schema number, hardware (GPU, driver, CPU, RAM, load), git sha and dirty flag, the image reference and image id of every container of the compose project, Triton model configuration, the engine record (label, free `KEY=VALUE` pairs, plan size and sha256), the dataset manifest hash, the sections captured, the SHA-256 of `arrays.npz` |
| `arrays.npz` | A fixed set of arrays (float16 vectors, float32 boxes, unicode strings): per item the image, box, detector class and score, integer crop rectangle, cluster id, distance, similarity and core flag, stored class, ground-truth class, VLM label and the PE crop vector; per image the PE whole-frame vector; MobileCLIP, face (box, score, embedding) and OCR (box, text) rows for a sample of images |

Items are keyed by image file name, box overlap and class name, never by stored id, so two captures of
different projects line up. The curation API never serves embedding vectors, so the PE vectors are read
by document id from the project's own OpenSearch indexes; everything else goes through the project-scoped
routes. The COCO ground-truth class of an item is the class of the best-IoU annotation (IoU at least 0.5,
else `none`).

`compare` pairs the two captures and measures box matching (greedy, best IoU first, same class name
only), cosine similarity of matched vectors, neighbour overlap at k = 10 on 1,000 seeded query items,
near-duplicate decisions at cosine 0.98, cluster agreement (adjusted Rand index, NMI, nearest-reference-
centroid agreement, k-means objective, purity against the COCO classes), class-name agreement, the VLM
label agreement and the OCR exact-match share. Thresholds live in `scripts/bench/parity_gates.json`
(values from the plan); the exit code is 0 on PASS, 1 on any FAIL (a gate with no data is SKIP and fails
only under `--strict`), 2 when the captures cannot be compared. `--out` writes a small
`parity_summary.json` and `.md`; the large capture files stay out of git.

| Gate | Rule |
|---|---|
| P-box | at least 98 % of images with every box matched at IoU 0.95 and confidence delta at most 0.02 |
| P-box-confident | zero unmatched boxes with confidence at least 0.5 |
| P-item | item count and class names identical on at least 99 % of images |
| P-pe, P-pew | cosine of crop vectors and of whole-frame vectors: 1st percentile at least 0.995, median at least 0.999 |
| P-knn, P-dup | neighbour overlap at least 0.95; near-duplicate decisions identical on at least 99.5 % of pairs |
| P-cluster-assign | at least 98 % of items keep their nearest reference centroid |
| P-cluster-purity | purity against the COCO classes not lower by more than 0.5 points |
| P-label-vote, P-knn-precision | majority neighbour class identical on at least 97 % of queries; kNN class precision not lower by more than 0.5 points |
| P-label, P-vlm | stored class name at least 99 % identical; VLM label at least 97 % identical |
| P-clip, P-face, P-ocr | MobileCLIP and ArcFace cosine 1st percentile at least 0.995; OCR exact match at least 98 % |

Crop-rectangle identity, the adjusted Rand index, NMI, the k-means objective difference and the cluster
similarity deltas are recorded but not gated (see the noise floor below).

When a rebuilt engine fails the gates against an old reference, decide on ground truth before capturing a new
reference: [parity_references.md](benchmarks/parity_references.md).

#### FP16 against FP32 PE engine

Same isolated stack (the published 0.5.0 images, GPU 0, own ports, no VLM), only the file behind
`pe_image_encoder` changes. The FP32 plan is the one of the 0.5.0 baseline (1,272,018,292 B). The FP16 plan
is built through the fixed installer path (`export/trt_utils.py` bake, then `trtexec` with the shapes of
`scripts/lib/model_setup.sh`): 641,185,140 B, built in 95 s. Each configuration was captured twice,
once with 4 and once with 1 upload request in flight (run 1 and run 2), and the FP32 plan was captured
a second time in each run as the noise floor (a repeat of the same configuration). All 12,789 to 12,792
items per capture are the same 6.4 items per image as the baseline.

**Result: the FP16 engine passes every gate in both runs** (P-vlm is skipped: no VLM is configured).

| Gate | Metric | Rule | FP16 run 1 | FP16 run 2 | FP32 repeat run 1 | FP32 repeat run 2 | FP16 verdict |
|---|---|---|---:|---:|---:|---:|---|
| P-box | `boxes.images_identical_share` | >= 0.98 | 0.9934 | 0.9934 | 0.9924 | 0.9939 | PASS |
| P-box-confident | `boxes.unmatched_confident` | <= 0 | 0 | 0 | 1 | 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | >= 0.99 | 0.9995 | 0.9995 | 0.9990 | 0.9995 | PASS |
| P-pe-p01 | `pe_crop.p01` | >= 0.995 | 0.9984 | 0.9990 | 1.0000 | 1.0000 | PASS |
| P-pe-median | `pe_crop.median` | >= 0.999 | 0.9999 | 0.9999 | 1.0000 | 1.0000 | PASS |
| P-pew-p01 | `pe_frame.p01` | >= 0.995 | 0.9975 | 0.9975 | 1.0000 | 1.0000 | PASS |
| P-pew-median | `pe_frame.median` | >= 0.999 | 0.9998 | 0.9998 | 1.0000 | 1.0000 | PASS |
| P-knn | `knn.overlap_at_k` | >= 0.95 | 0.9681 | 0.9706 | 0.9978 | 0.9974 | PASS |
| P-dup | `knn.near_dup_agreement` | >= 0.995 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | >= 0.98 | 0.9977 | 0.9977 | 0.9996 | 0.9996 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | <= 0.5 | -0.7115 | -0.4535 | -0.3988 | -0.1564 | PASS |
| P-label-vote | `knn.label_vote_agreement` | >= 0.97 | 0.9790 | 0.9820 | 0.9990 | 1.0000 | PASS |
| P-knn-precision | `knn.precision_drop_points` | <= 0.5 | 0.0382 | -0.0654 | 0.0127 | 0.0131 | PASS |
| P-label | `labels.class_agreement` | >= 0.99 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| P-vlm | `labels.vlm_agreement` | >= 0.97 | n/a | n/a | n/a | n/a | SKIP |
| P-clip | `clip.p01` | >= 0.995 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| P-face | `face.p01` | >= 0.995 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| P-ocr | `ocr.exact_share` | >= 0.98 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |

| Recorded metric | FP16 run 1 | FP16 run 2 | FP32 repeat run 1 | FP32 repeat run 2 |
|---|---:|---:|---:|---:|
| `boxes.crop_rect_identical_share` | 0.9899 | 0.9923 | 0.9917 | 0.9924 |
| `pe_crop.min` / `pe_crop.mean` | 0.9220 / 0.9998 | 0.9257 / 0.9998 | 0.9516 / 0.9999 | 0.9234 / 0.9999 |
| `pe_frame.min` / `pe_frame.mean` | 0.9915 / 0.9997 | 0.9915 / 0.9997 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `cluster.ari` / `cluster.nmi` | 0.3409 / 0.7764 | 0.3317 / 0.7701 | 0.3507 / 0.7791 | 0.3344 / 0.7698 |
| `cluster.objective_rel_diff` | -0.0016 | -0.0059 | 0.0024 | -0.0039 |
| `cluster.purity_ref` / `cluster.purity_cand` | 0.6162 / 0.6233 | 0.6202 / 0.6247 | 0.6162 / 0.6202 | 0.6202 / 0.6218 |
| `knn.precision_ref` / `knn.precision_cand` | 0.5743 / 0.5739 | 0.5849 / 0.5856 | 0.5743 / 0.5741 | 0.5857 / 0.5856 |
| `cluster.core_agreement`, `similarity_abs_delta_mean` | 0.962, 0.0215 | 0.960, 0.0222 | 0.962, 0.0213 | 0.961, 0.0219 |

Reading the numbers:

- **The embedding space is the same.** Crop vectors stay at median cosine 0.9999 (1st percentile 0.998 to
  0.999), whole-frame vectors at median 0.9998 (1st percentile 0.9975, worst single frame 0.9915). The
  neighbour overlap at 10 is 0.97, the reference nearest neighbour is always inside the FP16 top 10, and
  no near-duplicate decision at cosine 0.98 changes.
- **Clustering is unchanged within its own noise.** 99.8 % of items keep their nearest reference
  centroid. The adjusted Rand index is 0.33 to 0.35 for FP16 *and* for the repeat of the FP32 plan:
  k-means on 12.8k residual items is chaotic, so ARI is recorded and not gated, as the plan says.
  Purity against the COCO classes is 0.616 to 0.625 in every capture; FP16 is not lower than the
  reference.
- **Detection and labels do not change.** Matched detector scores differ by at most 0.0 in both FP16 runs
  (0.08 in one FP32 repeat). Boxes and crop rectangles differ for about 1 % of items in FP16 against FP32
  *and* in an FP32 repeat: the detector is not bit-reproducible between runs (0.8 to 1.0 % of crop
  rectangles move, worst matched IoU 0.77 to 0.78). The worst crop cosine (0.92) belongs to such a moved
  crop, not to precision. A class change would show as an unmatched box and fail P-item; none does. The
  kNN class precision against the COCO classes is the same to 0.1 point.
- **The noise floor decides how to read a single FAIL.** One unmatched box with confidence at least 0.5
  appeared between two FP32 captures of the same plan (run 1 repeat), which the strict P-box-confident
  rule counts as a failure. Capture the reference twice and compare the two before blaming a candidate.
- Ingest through `POST .../ingest/upload` took 119 s with the FP16 plan against 224 s with FP32 in
  run 1 (the rows of the Wave 1 section above); the captures are not a benchmark and the host carried
  other load.

#### Caveats

- One host, one GPU shared with other projects, the published 0.5.0 images; the container image ids are
  in each `capture.json`. The summaries of the four comparisons are in `docs/benchmarks/parity/`.
- The VLM gate and the label agreement of a registry-labelled project are not exercised: no VLM is
  configured and the project has no class registry, so stored class names are empty. The label gates here
  are the stored class name (trivially equal) and the majority detector class among the top 10 neighbours
  (P-label-vote), the label a propagation step would assign.
- Faces and OCR are probed on the first 200 pinned images through the core routes (200 faces, 34 text
  lines); MobileCLIP, ArcFace and OCR engines are unchanged by this run, so their agreement is exact.
- Cluster centroids are not served by the API; the reference centroids of the assignment-agreement
  metric are the normalised mean of each reference cluster's vectors.
- Rebuild of the FP16 plan: `export/trt_utils.py` bake inside the API image, then `trtexec` in the Triton
  image with the 1/8/32 x 3 x 336 x 336 profile and an 8 GB workspace; the bake log reports `fp16`.


### 0.6.0 Wave 0: full-pipeline throughput (one RTX A6000, FP16 PE engine)

The Wave 1 section above ends with the GPU 74 percent busy and the host named as the next
suspect. This section measures what the whole pipeline can do on one GPU at full speed and what
limits it, stage by stage: a client and Triton sweep, a set of large photos, the stages `ingest`
does not call (faces, OCR), where the API CPU goes, and how fast OpenSearch takes the documents.
Raw JSON: `docs/benchmarks/v060_wave0_*.json`; the tables are rendered by
`scripts/bench/ingest_sweep_report.py`. Medians with the minimum and maximum of 3 repetitions in
parentheses.

**Headline.**

- **On 640 x 480 images the GPU binds, as the trade study estimated.** With 16 or more clients
  `/ingest/upload` does **22.7 (22.5-23.4) images/s** and `/ingest/batch` **22.8 (22.8-23.1)** (2,000
  images, 32 per request) with the GPU **94 (92-94) percent** busy, 97 percent of the 23.5
  images/s ceiling of the trade study. The baseline's 4 clients leave the GPU 80 percent busy and
  give 18.7 images/s. The "GPU 26 percent idle" of Wave 1 was too few requests in flight, not the
  host: the API uses 4 of 48 cores at the plateau.
- **No Triton knob helps.** A second or third PE instance, one or four detector instances, a
  2 ms instead of 15 ms queue delay, and removing `preferred_batch_size` all stay within 3 percent
  of the shipped configuration (20.4 to 21.5 against 21.1 images/s).
- **On 12 to 24 MP photos the host binds and the GPU idles.** 1.0 to 1.3 CPU-seconds of API time
  per image (0.18 on COCO), 3 to 11 images/s, GPU 5 to 15 percent busy. 98 percent of the API CPU
  is in the event-loop thread of the workers. Adding clients beyond 32 lowers throughput (#241).
- **Faces and OCR are not part of ingest.** Called through their routes, faces cost 44 to 54 ms of
  API CPU per COCO image and almost no GPU; OCR is capped at 21 images/s by the single
  `ocr_pipeline` instance (42 with two, 80 with four, #240).
- **OpenSearch is not close to binding.** One writer sustains 1,011 documents/s (137 images/s
  equivalent); 16 writers 6,987 (944). `ef_construction` 128 instead of 512 cuts index CPU per
  document by 38 percent and raises documents/s by 5 percent.
- **Not measured:** the region stage (no segmenter weights), scaling across GPUs (the other two
  GPUs hold other projects), the cluster pass. See "Not measured".

#### Method and environment

| Item | Value |
|---|---|
| GPU | one RTX A6000 (GPU 2, 300 W cap, 48 GB), driver 615.71.09, CUDA 13.4; foreign SM load on it 0.0 percent in every sample of all 198 measured runs |
| Other GPUs | GPU 0 holds another project (16 GB resident, up to 95 percent utilization during one of my windows), GPU 1 another (7 GB): neither was used |
| Host | 2 x Xeon E5-2680 v3, 48 logical cores, 504 GB RAM, load average 5 to 59 over the runs (other projects), whole-host CPU busy 10 to 77 percent (columns below) |
| Stack | published 0.5.0 images (`images.lock` digests), isolated compose project, own ports, loopback only, OpenSearch heap 8 GB, no replicas, 32 API workers, curation workers and VLM not running |
| PE engine | FP16, baked with the fixed `export/trt_utils.py` and built with `trtexec` as in Wave 1: plan 644,771,508 bytes; all other engines as installed by 0.5.0 |
| Set A | the pinned COCO set (2,000 images, 640 x 480 median 166 KB); sweeps use its first 1,000 images, the headline rows all 2,000 |
| Set B | 500 public photos from the Wikimedia Commons featured-picture category, CC0, public domain, CC BY and CC BY-SA, 12.0 to 24.0 MP (median 17.0, mean 17.4), 5.2 GB (mean 10.4 MB), pinned in `scripts/datasets/manifests/commons_hires_500.json` (manifest sha256 `c013c1f9...`, seed 20261010, fetched with `scripts/datasets/fetch_hires_set.py`) |
| Protocol | the Wave 1 protocol: fresh project per repetition, warm-up project first, 3 repetitions, policy `all`; clients are threads of one Python process; cluster training and the storage merge are skipped in these runs (`--skip-cluster --skip-settle`) so that the window is the ingest alone |
| Setting changed | `MAX_FILE_SIZE_MB=1024` for set B: the shipped 50 MB counts the whole request and rejects 8 photos in one request (#239) |

"GPU busy" is the `nvidia-smi` utilization mean; "model exec / wall" is Triton execution seconds
of all models per wall second (above 1.0 when PE and the detector overlap). "API cores" is API CPU
seconds per image times images per second.

#### Probe 1: client concurrency and request size (set A, `/ingest/upload`)

| batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 x 4 | 14.90 (13.06-15.72) | 2.1 (1.9-2.4) | 60 (55-70) | 0.65 (0.57-0.70) | 0.181 (0.172-0.198) | 2.7 | 28 (23-53) | 1601 (1562-1647) | 12.1 (12.0-12.4) | 14 (12-21) |
| 8 x 16 | 22.05 (21.06-22.38) | 5.4 (5.1-5.6) | 93 (87-94) | 1.09 (1.05-1.13) | 0.181 (0.170-0.194) | 4.0 | 34 (31-58) | 1441 (1433-1462) | 22.5 (21.8-23.3) | 19 (19-31) |
| 32 x 1 | 5.97 (5.35-6.10) | 5.2 (5.1-5.6) | 24 (20-28) | 0.25 (0.23-0.26) | 0.187 (0.183-0.212) | 1.1 | 43 (28-56) | 1749 (1721-1749) | 7.5 (7.3-7.7) | 23 (16-48) |
| 32 x 2 | 9.82 (9.34-10.87) | 6.3 (5.8-6.8) | 37 (36-49) | 0.42 (0.40-0.46) | 0.189 (0.177-0.209) | 1.9 | 57 (26-77) | 1675 (1641-1679) | 8.7 (8.2-9.0) | 51 (23-59) |
| 32 x 4 | 16.80 (16.33-17.98) | 7.3 (6.7-7.3) | 66 (65-72) | 0.72 (0.70-0.78) | 0.172 (0.158-0.186) | 2.9 | 43 (38-71) | 1549 (1544-1554) | 15.3 (12.2-17.0) | 31 (27-43) |
| 32 x 8 | 20.18 (17.36-21.76) | 12.2 (11.0-12.4) | 87 (70-89) | 0.90 (0.76-0.99) | 0.176 (0.170-0.208) | 3.6 | 32 (28-73) | 1501 (1422-1517) | 20.4 (17.1-20.6) | 36 (23-53) |
| 32 x 16 | 21.10 (19.16-21.15) | 20.8 (20.4-21.1) | 86 (80-88) | 0.97 (0.88-0.98) | 0.190 (0.184-0.196) | 3.9 | 42 (36-54) | 1468 (1432-1477) | 25.0 (20.4-27.2) | 24 (19-29) |
| 32 x 32 | 19.97 (19.37-20.24) | 45.3 (43.6-46.4) | 80 (79-82) | 0.88 (0.87-0.88) | 0.204 (0.199-0.214) | 4.1 | 35 (34-72) | 1500 (1486-1504) | 25.2 (24.7-26.0) | 31 (18-39) |

The GPU is the limit from about 8 clients on: images/s stops rising at 20 to 22, PE batches grow
from 7.5 (1 client) to 25 (16), the GPU runs at 86 to 93 percent and API CPU stays at 0.17 to
0.20 CPU-s per image (3.6 to 4.1 cores). Below that the pipeline is request-bound: one client of
32 images does 6.0 images/s with the GPU 24 percent busy. More than 16 clients does not help
(32 clients: 20.0, with a 45 s request p50). Smaller requests fill the batcher as well as large
ones when there are enough of them: 8 images x 16 clients is the best row (22.1).

The SM clock falls from 1,720 to 1,750 MHz at light load to 1,420 to 1,500 MHz at the plateau; the
board draws 254 to 271 W on average at 8 or more clients and reaches its 300 W limit (the throttle bit is set in 92 to 98
percent of samples at 4 or more clients). The plateau is therefore a power-capped GPU, not 1.8 GHz.

Headline rows, all 2,000 images, the same 32 images per request:

Upload:

| batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 x 4 | 18.69 (17.91-19.78) | 6.4 (6.1-6.6) | 80 (72-82) | 0.81 (0.78-0.87) | 0.152 (0.148-0.158) | 3.0 | 24 (23-25) | 1534 (1502-1545) | 15.4 (13.6-16.2) | 10 (10-11) |
| 32 x 16 | 22.71 (22.53-23.36) | 23.1 (20.9-23.3) | 94 (92-94) | 1.05 (1.04-1.10) | 0.175 (0.172-0.177) | 4.0 | 27 (27-28) | 1442 (1438-1452) | 26.6 (26.1-27.4) | 11 (11-11) |

Batch (the API reads the files):

| batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 x 4 | 18.77 (18.73-19.90) | 6.6 (6.3-6.8) | 81 (76-86) | 0.82 (0.81-0.88) | 0.152 (0.152-0.155) | 2.9 | 24 (24-24) | 1510 (1504-1530) | 16.9 (16.2-17.5) | 10 (9-10) |
| 32 x 16 | 22.80 (22.78-23.09) | 22.5 (21.0-22.6) | 94 (94-94) | 1.06 (1.04-1.07) | 0.178 (0.172-0.179) | 4.1 | 27 (27-28) | 1447 (1447-1454) | 27.1 (26.8-27.5) | 10 (10-12) |

PE mean batch rises from 15 to 27, PE time stays at 41 ms per image (5.5 ms per embedding) and
the detector takes 5 ms per image once it competes with PE for the card.

#### Probe 1b: Triton instance counts, queue delay, preferred batch (set A, 32 x 16)

| Triton arm | batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det=1 | 32 x 16 | 21.40 (20.87-21.78) | 21.9 (20.2-22.7) | 89 (86-90) | 0.94 (0.94-0.98) | 0.183 (0.176-0.188) | 3.9 | 29 (27-36) | 1428 (1420-1456) | 26.0 (24.5-26.5) | 15 (15-20) |
| det=4 | 32 x 16 | 20.75 (20.16-21.56) | 22.0 (19.9-22.8) | 85 (84-87) | 0.92 (0.90-0.95) | 0.188 (0.179-0.193) | 3.9 | 30 (23-43) | 1460 (1428-1487) | 25.0 (24.5-25.4) | 14 (13-21) |
| pe=2 | 32 x 16 | 21.52 (20.19-21.56) | 20.6 (20.2-23.1) | 87 (82-91) | 1.90 (1.75-1.92) | 0.175 (0.172-0.183) | 3.8 | 27 (26-28) | 1463 (1415-1474) | 25.4 (25.0-26.4) | 13 (11-16) |
| pe=3 | 32 x 16 | 20.41 (19.68-21.01) | 22.3 (22.2-22.6) | 84 (82-85) | 2.47 (2.44-2.57) | 0.185 (0.181-0.188) | 3.8 | 28 (27-29) | 1486 (1439-1491) | 24.1 (23.0-25.5) | 14 (13-15) |
| pe_delay=2000 | 32 x 16 | 21.47 (20.82-21.81) | 21.0 (20.6-21.3) | 88 (86-88) | 0.97 (0.95-1.00) | 0.190 (0.183-0.200) | 4.0 | 34 (30-56) | 1431 (1423-1471) | 26.0 (26.0-26.4) | 17 (15-21) |
| pe_nopref=1 | 32 x 16 | 21.10 (20.92-21.72) | 21.6 (21.3-21.7) | 85 (83-88) | 0.95 (0.93-0.96) | 0.186 (0.182-0.196) | 3.9 | 28 (27-40) | 1447 (1431-1461) | 25.8 (24.7-26.0) | 16 (15-20) |

(The shipped row is the 32 x 16 row of probe 1.) Every arm is within the repetition spread of the
shipped configuration: the GPU is already the limit, so more instances of PE only share its time
slices (three instances are slightly lower, 20.4), the detector's instance count does not matter
(4 to 5 ms of GPU per image), and the batcher settings change the PE mean batch by less than 1.5.
Isolated PE through Triton, `perf_analyzer`, concurrency 6 (the best level), synthetic input:

| Model | batch | best concurrency | inferences/s | p50 ms | exec infer ms per batch |
|---|---:|---:|---:|---:|---:|
| `pe_image_encoder` | 8 | 6 | 195 | 243 | 83 |
| `pe_image_encoder` | 16 | 6 | 194 | 491 | 163 |
| `pe_image_encoder` | 32 | 6 | 195 | 984 | 164 |
| `yolov11_small_trt_end2end` | 8 | 11 | 485 | 161 | 9 |
| `yolov11_small_trt_end2end` | 16 | 6 | 388 | 232 | 11 |
| `yolov11_small_trt_end2end` | 32 | 6 | 262 | 740 | 35 |

The isolated PE engine does 195 embeddings/s at every batch size (5.1 ms each); at 7.4 embeddings
plus 5 ms of detector per image that is 23.3 images/s. The pipeline reaches 22.7 of it.

#### Probe 2: high-resolution photos (set B)

Set B, `/ingest/upload`:

| batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 x 4 | 2.90 (2.47-2.98) | 10.1 (10.1-10.3) | 5 (4-5) | 0.05 (0.04-0.05) | 1.301 (1.262-1.335) | 3.9 | 17 (14-21) | 1770 (1770-1772) | 4.0 (3.8-4.1) | 8 (7-11) |
| 8 x 16 | 4.76 (4.45-6.52) | 17.7 (13.0-19.5) | 9 (7-12) | 0.08 (0.07-0.11) | 1.233 (1.206-1.237) | 5.7 | 26 (19-26) | 1737 (1727-1742) | 4.1 (4.0-4.2) | 10 (9-13) |
| 32 x 16 | 3.37 (3.22-3.87) | 77.1 (70.6-88.4) | 5 (5-6) | 0.06 (0.05-0.07) | 1.265 (1.237-1.281) | 4.3 | 22 (22-24) | 1655 (1630-1673) | 3.6 (3.6-3.8) | 9 (8-9) |

Set B, `/ingest/batch`:

| batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 x 4 | 3.32 (3.18-3.35) | 9.2 (8.9-9.5) | 6 (5-6) | 0.05 (0.05-0.05) | 1.191 (1.186-1.244) | 3.9 | 16 (15-16) | 1770 (1761-1771) | 4.0 (4.0-4.1) | 8 (7-8) |
| 8 x 16 | 9.07 (6.02-9.53) | 10.4 (10.3-11.7) | 15 (6-16) | 0.15 (0.10-0.16) | 1.195 (1.153-1.226) | 11.1 | 32 (27-41) | 1714 (1676-1747) | 4.1 (4.0-4.3) | 12 (11-19) |
| 32 x 16 | 5.35 (4.09-5.77) | 45.5 (39.0-70.1) | 7 (6-10) | 0.09 (0.07-0.10) | 1.182 (1.175-1.204) | 6.3 | 25 (17-28) | 1597 (1592-1663) | 3.8 (3.6-3.9) | 8 (7-9) |

More clients on `/ingest/batch` (8 images per request):

| batch x clients | images/s | request p50 s | GPU 2 util % | model exec / wall | API CPU s/image | API cores | host busy % | SM MHz | PE mean batch | load avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 x 32 | 11.42 (10.99-12.02) | 14.3 (13.1-18.6) | 14 (13-19) | 0.19 (0.18-0.20) | 1.217 (1.196-1.252) | 13.9 | 47 (45-50) | 1674 (1665-1679) | 4.6 (4.5-4.8) | 18 (15-19) |
| 8 x 64 | 5.99 (5.54-6.02) | 49.6 (45.0-52.1) | 11 (8-14) | 0.10 (0.09-0.10) | 1.191 (1.188-1.198) | 7.1 | 29 (28-30) | 1737 (1733-1752) | 4.5 (4.4-4.6) | 11 (10-12) |

On photos the pipeline is host-bound with the GPU mostly idle. Per image the API spends 1.0 to
1.3 CPU-seconds (0.18 on COCO) and the photos yield only 1.6 items and 2.6 embeddings each
(6.4 and 7.4 on COCO), so the PE work per image falls to 15 ms while the decode path grows
from 35 ms to about 0.55 s of single-thread work (probe 4). Throughput scales
with clients up to about 32 clients (11.4 images/s, 14 cores in use) and then falls (64 clients:
6.0), while the stage timers show `opensearch_write` latency per image growing from 2.1 s to 7.2 s
on an unchanged document rate: the writes wait for event loops that are busy with decode (#241).
`/ingest/upload` is slower than `/ingest/batch` at the same settings because the single-process
Python client also has to read and encode 80 MB per request.

Per-image stage timers (summed per image across concurrent requests; `/ingest/batch`, 8 x 16):

| Stage | calls/image | mean ms per call | summed ms/image |
|---|---:|---:|---:|
| decode | 1.00 | 305 | 305 |
| resize (PE crop preprocessing) | 0.59 | 11 | 7 |
| jpeg_encode | 1.63 | 14 | 23 |
| crop | 1.63 | 5 | 8 |
| `embed` (wait for Triton) | 1.61 | 1,945 | 3,127 |
| `opensearch_write` (wait) | 1.59 | 806 | 1,281 |

Triton at that point: PE 2.6 inferences per image at mean batch 4.1, 15 ms of GPU per image;
detector 1.2 ms. The `embed` and `opensearch_write` rows are waits, inflated by the loop being
busy; they are not work.

#### Probe 3: faces and OCR (the stages ingest does not call)

`/ingest/*` calls the detector and PE only. Faces and OCR are reachable through `/faces/*`, `/ocr/*`
and `/analyze`; the region stage that reads text is a worker (see "Not measured"). Per endpoint,
200 images of set A, 1 client and 16 clients:

| Endpoint | clients | images/s | p50 ms | API CPU s/image | Triton CPU s/image | Triton exec ms/image by model |
|---|---:|---:|---:|---:|---:|---:|
| analyze | 1 | 5.4 (5.3-5.4) | 176 (175-177) | 0.132 (0.131-0.134) | 0.078 (0.077-0.078) | arcface_w600k_r50 1.7, mobileclip2_s2_image_encoder 31.8, scrfd_10g_bnkps 1.3, yolov11_small_trt_end2end 5.1 |
| analyze | 16 | 55.5 (54.4-65.1) | 224 (217-229) | 0.116 (0.102-0.119) | 0.046 (0.045-0.049) | arcface_w600k_r50 2.3, mobileclip2_s2_image_encoder 11.4, scrfd_10g_bnkps 2.2, yolov11_small_trt_end2end 3.7 |
| analyze_ocr | 1 | 5.1 (5.0-5.1) | 180 (178-180) | 0.178 (0.177-0.182) | 0.133 (0.132-0.136) | arcface_w600k_r50 2.0, mobileclip2_s2_image_encoder 32.3, ocr_pipeline 46.0, paddleocr_det_trt 4.1, paddleocr_rec_trt 5.2, scrfd_10g_bnkps 1.3, yolov11_small_trt_end2end 4.6 |
| analyze_ocr | 16 | 20.0 (19.9-20.1) | 708 (705-721) | 0.160 (0.157-0.163) | 0.110 (0.109-0.112) | arcface_w600k_r50 2.1, mobileclip2_s2_image_encoder 16.8, ocr_pipeline 45.9, paddleocr_det_trt 4.5, paddleocr_rec_trt 5.9, scrfd_10g_bnkps 1.8, yolov11_small_trt_end2end 5.4 |
| detect | 1 | 12.7 (12.4-12.7) | 79 (79-80) | 0.062 (0.061-0.065) | 0.016 (0.016-0.017) | yolov11_small_trt_end2end 4.7 |
| detect | 16 | 133.2 (115.5-134.0) | 93 (86-100) | 0.050 (0.049-0.051) | 0.010 (0.009-0.011) | yolov11_small_trt_end2end 1.4 |
| embed_image | 1 | 111.4 (14.3-112.3) | 9 (8-68) | 0.011 (0.009-0.043) | 0.001 (0.000-0.016) | mobileclip2_s2_image_encoder 11.8 |
| embed_image | 16 | 223.0 (212.7-303.1) | 71 (50-82) | 0.027 (0.019-0.028) | 0.004 (0.002-0.004) | mobileclip2_s2_image_encoder 1.7 |
| faces_detect | 1 | 16.8 (16.1-16.8) | 54 (53-56) | 0.054 (0.052-0.056) | 0.017 (0.017-0.018) | arcface_w600k_r50 1.8, scrfd_10g_bnkps 1.3 |
| faces_detect | 16 | 154.0 (132.3-158.8) | 66 (65-78) | 0.044 (0.043-0.052) | 0.012 (0.012-0.015) | arcface_w600k_r50 0.9, scrfd_10g_bnkps 0.9 |
| faces_recognize | 1 | 16.7 (16.3-17.3) | 54 (52-55) | 0.052 (0.052-0.054) | 0.017 (0.016-0.017) | arcface_w600k_r50 1.7, scrfd_10g_bnkps 1.3 |
| faces_recognize | 16 | 179.3 (172.7-179.6) | 76 (74-80) | 0.047 (0.046-0.050) | 0.014 (0.013-0.015) | arcface_w600k_r50 1.2, scrfd_10g_bnkps 1.0 |
| ocr_predict | 1 | 9.2 (8.8-9.2) | 93 (91-98) | 0.082 (0.080-0.084) | 0.055 (0.055-0.061) | ocr_pipeline 45.4, paddleocr_det_trt 4.3, paddleocr_rec_trt 5.1 |
| ocr_predict | 16 | 20.2 (20.1-20.2) | 723 (720-746) | 0.079 (0.079-0.083) | 0.056 (0.055-0.057) | ocr_pipeline 45.8, paddleocr_det_trt 4.1, paddleocr_rec_trt 5.3 |

The same endpoints on 100 photos of set B (16.1 MP mean, 8.5 MB):

| Endpoint | clients | images/s | p50 ms | API CPU s/image | Triton CPU s/image | Triton exec ms/image by model |
|---|---:|---:|---:|---:|---:|---:|
| analyze | 1 | 2.0 (2.0-2.0) | 460 (457-465) | 0.779 (0.767-0.781) | 0.063 (0.062-0.065) | arcface_w600k_r50 0.8, mobileclip2_s2_image_encoder 21.0, scrfd_10g_bnkps 1.3, yolov11_small_trt_end2end 4.3 |
| analyze | 16 | 22.8 (22.5-23.3) | 588 (576-603) | 0.789 (0.783-0.799) | 0.043 (0.041-0.043) | arcface_w600k_r50 0.8, mobileclip2_s2_image_encoder 11.2, scrfd_10g_bnkps 1.4, yolov11_small_trt_end2end 3.7 |
| analyze_ocr | 1 | 0.4 (0.4-0.4) | 2499 (2481-2529) | 3.050 (3.030-3.061) | 1.297 (1.268-1.301) | arcface_w600k_r50 0.8, mobileclip2_s2_image_encoder 21.2, ocr_pipeline 775.6, paddleocr_det_trt 3.8, paddleocr_rec_trt 15.1, scrfd_10g_bnkps 1.3, yolov11_small_trt_end2end 4.2 |
| analyze_ocr | 16 | 1.2 (1.2-1.2) | 10636 (10540-10770) | 2.755 (2.749-2.798) | 1.232 (1.215-1.268) | arcface_w600k_r50 0.8, mobileclip2_s2_image_encoder 21.1, ocr_pipeline 765.8, paddleocr_det_trt 3.5, paddleocr_rec_trt 15.3, scrfd_10g_bnkps 1.4, yolov11_small_trt_end2end 4.3 |
| detect | 1 | 2.5 (2.5-2.6) | 348 (339-351) | 0.446 (0.434-0.457) | 0.021 (0.020-0.021) | yolov11_small_trt_end2end 4.2 |
| detect | 16 | 33.7 (33.7-34.9) | 369 (365-380) | 0.376 (0.372-0.384) | 0.011 (0.011-0.012) | yolov11_small_trt_end2end 3.0 |
| embed_image | 1 | 8.3 (2.1-9.0) | 110 (101-430) | 0.149 (0.134-0.540) | 0.004 (0.003-0.024) | mobileclip2_s2_image_encoder 11.7 |
| embed_image | 16 | 40.2 (36.2-42.6) | 333 (267-391) | 0.299 (0.274-0.340) | 0.007 (0.006-0.009) | mobileclip2_s2_image_encoder 4.5 |
| faces_detect | 1 | 2.7 (2.7-2.8) | 324 (316-332) | 0.428 (0.420-0.437) | 0.020 (0.020-0.021) | arcface_w600k_r50 0.9, scrfd_10g_bnkps 1.2 |
| faces_detect | 16 | 36.3 (34.5-37.1) | 351 (321-356) | 0.355 (0.340-0.362) | 0.013 (0.013-0.014) | arcface_w600k_r50 0.7, scrfd_10g_bnkps 1.2 |
| faces_recognize | 1 | 2.7 (2.7-2.8) | 313 (313-333) | 0.429 (0.423-0.433) | 0.020 (0.020-0.021) | arcface_w600k_r50 0.9, scrfd_10g_bnkps 1.2 |
| faces_recognize | 16 | 36.8 (36.6-37.1) | 339 (335-348) | 0.364 (0.363-0.368) | 0.013 (0.013-0.013) | arcface_w600k_r50 0.8, scrfd_10g_bnkps 1.3 |
| ocr_predict | 1 | 0.4 (0.4-0.4) | 2439 (2429-2494) | 2.467 (2.457-2.498) | 1.159 (1.155-1.180) | ocr_pipeline 774.3, paddleocr_det_trt 3.7, paddleocr_rec_trt 14.9 |
| ocr_predict | 16 | 1.2 (1.2-1.2) | 10619 (10486-10732) | 2.139 (2.134-2.147) | 1.143 (1.113-1.150) | ocr_pipeline 763.6, paddleocr_det_trt 3.6, paddleocr_rec_trt 15.4 |

- **Faces** are host-bound: 1.8 to 3.1 ms of GPU per image (SCRFD plus ArcFace), 44 to 54 ms of API
  CPU per COCO image (0.36 to 0.43 s per photo), 154 to 179 images/s at 16 clients on COCO and
  36 on photos. At the top of the COCO rows the single Python client process may be the limit too.
- **OCR** is bound by one Python-backend instance: `ocr_pipeline` takes 42 to 46 ms per COCO image
  for about 9 ms of GPU in the detection and recognition engines, so 16 clients get 21 images/s
  and a 680 ms p50. The instance count scales it linearly:

| `ocr_pipeline` instances | Endpoint (16 clients) | images/s | p50 ms | API CPU s/image | Triton CPU s/image |
|---|---:|---:|---:|---:|---:|
| 1 (shipped) | `ocr_predict` | 21.4 (21.4-21.8) | 678 (658-680) | 0.073 (0.072-0.076) | 0.052 (0.052-0.052) |
| 1 (shipped) | `analyze_ocr` | 21.2 (20.9-21.5) | 646 (629-662) | 0.157 (0.152-0.164) | 0.104 (0.102-0.106) |
| 2 | `ocr_predict` | 42.6 (40.5-43.1) | 335 (334-355) | 0.067 (0.066-0.071) | 0.052 (0.052-0.056) |
| 2 | `analyze_ocr` | 41.3 (40.3-41.3) | 345 (340-357) | 0.155 (0.153-0.158) | 0.101 (0.099-0.104) |
| 4 | `ocr_predict` | 80.0 (72.6-83.9) | 162 (160-185) | 0.066 (0.058-0.070) | 0.052 (0.049-0.058) |
| 4 | `analyze_ocr` | 51.8 (51.4-52.8) | 272 (272-278) | 0.174 (0.166-0.176) | 0.106 (0.104-0.108) |

  On 16 MP photos OCR is 1.2 images/s: 765 ms in `ocr_pipeline`, 2.1 s of API CPU and 1.1 s of
  Triton CPU per image, because the whole decoded frame travels to Triton as FP32 (the case the
  faces and OCR plan WP-2.5, #220, addresses).
- **`/analyze`** (detector, faces, MobileCLIP, optional OCR): 55 images/s without OCR and 20 with
  it at 16 clients on COCO (52 with four OCR instances); 23 and 1.2 on photos.
- `/embed/image` at 1 client varies between 14 and 112 images/s across repetitions (the first
  repetition is eight times slower than the others; a warm-up or cache effect that was not
  investigated); the 16-client row is the stable one.

#### Probe 4: where the API CPU goes (f)

Per image the API container uses 0.18 CPU-s on COCO and 1.0 to 1.2 on photos (cgroup `usage_usec`).
The timed stages explain about 32 ms of the first. Three instruments:

1. **CPU census by thread** (every thread of the container, 30 s in the middle of a run, no
   profiler running): the work is in the event-loop thread of each worker (the main thread of the
   process; threads are named by the kernel, `python3.13` is the interpreter).

Set A, `/ingest/upload`, 32 x 16:

Over 30 s: 2.81 cores in the 34 processes present throughout (19 % system), 2.81 of them in threads that lived the whole window, and 0.00 cores of lifetime CPU in 0 processes started during it.

| Thread name | threads | cores | system % | share % |
|---|---:|---:|---:|---:|
| `python3.13 (main thread)` | 33 | 2.52 | 15 | 89.9 |
| `event_engine` | 512 | 0.14 | 80 | 5.2 |
| `python3.13` | 2325 | 0.13 | 32 | 4.6 |
| `uvicorn (main thread)` | 1 | 0.01 | 27 | 0.3 |

Set B, `/ingest/batch`, 8 x 16:

Over 30 s: 5.94 cores in the 34 processes present throughout (10 % system), 5.94 of them in threads that lived the whole window, and 0.00 cores of lifetime CPU in 0 processes started during it.

| Thread name | threads | cores | system % | share % |
|---|---:|---:|---:|---:|
| `python3.13 (main thread)` | 33 | 5.85 | 9 | 98.5 |
| `event_engine` | 512 | 0.06 | 70 | 0.9 |
| `python3.13` | 2326 | 0.02 | 30 | 0.4 |
| `uvicorn (main thread)` | 1 | 0.01 | 22 | 0.1 |

2. **Single-thread cost of the primitives** (`scripts/bench/ingest_micro_profile.py`, inside the API
   container, median of 20 images, milliseconds):

| Operation (ms) | set A frame | set B photo |
|---|---:|---:|
| file read | 0.1 | 3.2 |
| SHA-256 of the file | 0.4 | 28.5 |
| PIL decode + EXIF transpose | 3.3 | 316.5 |
| detector letterbox (640) | 4.6 | 80.5 |
| PE whole-frame preprocess | 5.5 | 136.8 |
| crop to array (central third) | 0.1 | 4.5 |
| PE crop resize + normalize | 2.3 | 2.8 |
| crop JPEG encode | 0.3 | 10.2 |
| megapixels | 0.27 | 18.4 |

   On an 18 MP photo, decode (316), the letterbox (80), the whole-frame preprocess (137) and the
   file hash (28) alone are about 0.56 s of the 1.0 to 1.2 s; on a COCO frame the same list is
   14 ms of 180.
3. **py-spy** (0.4.2, one process per worker, raw stacks kept in the JSON): it kept falling behind
   (about 1,000 non-idle samples where the workers' CPU implies tens of thousands; each worker has
   72 threads), 73 percent of what it sampled is the event loop waiting, and it costs 17 cores
   itself. Its stacks are not used for the numbers here.

What this attributes: more than 90 percent of the CPU is synchronous work on the event-loop
thread (COCO: 90 percent; photos: 98.5 percent), 9 to 19 percent of it system time (page faults
and copies of large arrays). On COCO the named primitives explain about 31 of the 180 ms per image
(decode, letterbox, whole-frame preprocess, 6.4 crop resizes and encodes), the 32 ms of timed stages
are part of that, and about 150 ms per image is unattributed: serialization of 14.9 MB of FP32
tensors per image to Triton, document and bulk-body construction (vectors as lists of numpy
floats), the gRPC client, and the 32 busy workers competing on a hyper-threaded host. Naming
that 150 ms is work package WP-1.2 (#210); this probe tells it where to look (the loop thread, not
a thread pool) and shows that on photos decode is the first item.

#### Probe 5: OpenSearch write ceiling (g, WP-S5)

100,000 item documents (1,024-dimension vector, 154 mapped fields) plus 15,625 image documents
(1,024- and 512-dimension vectors), the proportion of a 6.4-items-per-image ingest, written to
fresh indexes with the real mapping and settings of a project (one shard, no replica, `faiss`
HNSW `m` 16, `ef_construction` 512, cosine, 1 s refresh, 8 GB heap), vectors taken from a real
project plus noise so that the graph is built over clustered data (the `iid` arm uses independent
unit vectors). Bulk bodies are built before the clock starts. "Searchable" is the time from the
first request until a refresh has made everything visible. Bytes per document are measured right
after the load, before merges.

| Arm | bulk x writers | refresh / ef / vectors | docs/s | images/s equiv. | searchable s | bulk p95 s | bytes/doc | node cores | node CPU-s per 1k docs | 429 retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 1000 x 4 | shipped, ef shipped, real | 3206 (3093-3241) | 433 (418-438) | 43 (43-45) | 1.68 (1.67-1.71) | 10097 (9174-10354) | 7.7 (7.1-7.7) | 2.39 (2.19-2.50) | 0 (0-0) |
| bulk500 | 500 x 4 | shipped, ef shipped, real | 2818 (2799-2864) | 381 (378-387) | 49 (48-50) | 1.06 (1.06-1.11) | 9177 (9176-10051) | 6.1 (5.5-6.2) | 2.13 (1.96-2.20) | 0 (0-0) |
| bulk5000 | 5000 x 4 | shipped, ef shipped, real | 3617 (3426-3633) | 489 (463-491) | 37 (37-37) | 6.61 (6.21-6.88) | 10642 (10435-12136) | 8.6 (8.4-9.7) | 2.52 (2.30-2.69) | 0 (0-0) |
| writers1 | 1000 x 1 | shipped, ef shipped, real | 1011 (1007-1041) | 137 (136-141) | 127 (125-129) | 1.23 (1.16-1.26) | 9163 (8940-9164) | 2.2 (2.1-2.2) | 2.12 (2.10-2.18) | 0 (0-0) |
| writers8 | 1000 x 8 | shipped, ef shipped, real | 5048 (4688-5060) | 682 (634-684) | 24 (24-25) | 2.07 (2.04-2.22) | 14325 (11869-14686) | 13.2 (12.1-13.3) | 2.62 (2.40-2.81) | 0 (0-0) |
| writers16 | 1000 x 16 | shipped, ef shipped, real | 6987 (6971-7376) | 944 (942-997) | 17 (17-18) | 3.34 (2.69-3.43) | 12783 (11196-13273) | 19.3 (17.0-19.4) | 2.62 (2.44-2.78) | 0 (0-0) |
| refresh_off | 1000 x 4 | -1, ef shipped, real | 2703 (2693-2828) | 365 (364-382) | 59 (59-61) | 3.11 (2.91-3.54) | 9164 (9163-9164) | 5.9 (5.9-6.2) | 2.17 (2.10-2.30) | 0 (0-0) |
| ef128 | 1000 x 4 | shipped, ef 128, real | 3365 (3295-3456) | 455 (445-467) | 38 (36-39) | 1.52 (1.50-1.59) | 9175 (9174-9175) | 5.0 (4.7-5.2) | 1.49 (1.43-1.50) | 0 (0-0) |
| iid_vectors | 1000 x 4 | shipped, ef shipped, iid | 2667 (2541-2692) | 360 (343-364) | 70 (62-77) | 3.80 (2.62-4.01) | 10378 (9177-11714) | 11.2 (10.4-11.7) | 4.37 (3.86-4.39) | 0 (0-0) |

- One node takes 1,011 documents/s from one writer and 6,987 from sixteen. The GPU ceiling of 23
  images/s needs about 170 documents/s; the WP-S5 target of 100 images/s (about 740 vectors and 840
  documents/s) is met by a single writer.
- Bulk size 500, 1,000 and 5,000 differ by 25 percent in documents/s; larger bulks raise the
  request p95 to 6.6 s. `refresh_interval` -1 during the load was not faster (2,703 against 3,206).
- `ef_construction` 128 against 512: 3,365 against 3,206 documents/s, 1.49 against 2.39 node CPU
  seconds per 1,000 documents (-38 percent), searchable 38 against 43 s. Stored bytes per document
  do not differ beyond merge-state noise (9.2 to 10.1 KB). Recall at 128 was not measured here (WP-V, #221, owns that gate); the
  defaults are unchanged.
- Independent random vectors cost 1.8 times the CPU of clustered ones (4.37 against 2.39 CPU-s per
  1,000 documents): index CPU depends on the data, which is why the real-vector arm is the shipped row.
- 429 retries: none in any arm.

#### Binding resource per stage and per GPU

| Stage | Set A (COCO, 640 x 480) | Set B (12 to 24 MP photos) |
|---|---|---|
| Client requests in flight | binds below 8 clients (GPU 24 to 66 percent busy at 1 to 4) | not reached; clients scale throughput to 32 |
| API decode, letterbox, crop, tensor build (event-loop thread) | 0.18 CPU-s/image, 4 of 48 cores: not binding | **binds**: 1.0 to 1.3 CPU-s/image, 14 cores at 11.4 images/s |
| Detector (GPU) | 2 to 5 ms/image: not binding | 1 ms/image: not binding |
| PE embedding (GPU tensor compute at the 300 W cap) | **binds** at 16 or more clients: 41 ms/image, 94 percent busy, 22.7 images/s | not binding: 15 ms/image, 5 to 15 percent busy |
| Triton configuration (instances, delay, preferred batch) | not binding (all arms within 3 percent) | not binding |
| OpenSearch | not binding: 137 images/s equivalent from one writer | not binding in these runs; write latency inflates only because the API loops are busy |
| Faces (`/faces/*`) | API CPU, 44 to 54 ms/image | API CPU, 0.36 to 0.43 s/image |
| OCR (`/ocr/*`) | `ocr_pipeline` instance: 21 images/s with one | `ocr_pipeline` CPU and full-frame transfer: 1.2 images/s |
| Cluster pass, region stage | not measured | not measured |

Per GPU: **GPU 2** (RTX A6000) is the only one measured. Its binding resource is tensor compute
under the power cap on small images and nothing (85 to 95 percent idle) on photos. GPU 0 (the other
A6000) and GPU 1 (the 3080 Ti) host other projects' containers and were not touched.

#### What the trade study's ceiling table got right and wrong

Right:

- One A6000 at policy `all` has a PE-bound ceiling of about 23.5 images/s: measured 22.7 (97 percent)
  with the shipped Triton configuration, and 23.3 from `perf_analyzer` plus the detector.
- Host CPU is not the binder on COCO-size images: 0.15 to 0.19 CPU-s per image (estimate 0.22),
  4 cores at the plateau; the host-to-device copy phase of a PE execution is 0.4 ms of 145 ms, and the
  Triton instance counts do not matter.
- The X-1 target (GPU at least 90 percent busy and at least 21 images/s) is reached with no
  rebuild and no Triton change, only 16 clients instead of 4.
- Datastore writes (the "unknown" row) are far from binding on one node.

Wrong or incomplete:

- The 45 percent of peak FP16 efficiency uses the 1.8 GHz boost clock. The card runs 1.42 to 1.50 GHz
  at its 300 W limit, where the dense FP16 peak is about 125 TFLOP/s and PE's 70 TFLOP/s while
  executing is about 56 percent: at the power cap, the engine-tuning headroom (WP-S2) is smaller than
  the 10 to 25 percent estimate; more throughput has to come from fewer FLOPs per image (INT8, fewer
  embeddings) or more cards.
- Wave 1 read the 74 percent GPU busy as the host feeding PE too slowly. At 4 clients it was the
  request depth; with 16 the host is at 4 cores and the GPU is full.
- The host-CPU row ("110 to 135 images/s") is for COCO-size frames. On 12 to 24 MP photos the host
  binds at about 11 images/s with 14 cores, an order of magnitude below the GPU ceiling, and the
  ceiling table has no photo row. The "decode stays on the CPU unless set B says otherwise"
  rule of the study is answered by the public analogue of set B: it says otherwise (#241, #217).
- The single event loop is called "not binding with 32 workers". It is what makes photos slow: 98.5
  percent of the CPU runs on the loop thread, so a worker cannot overlap two images.
- The multi-GPU rows (2x, 55 to 58 images/s) and the "next binder is host CPU at about 55
  images/s" could not be checked (one GPU only). At 22.7 images/s the host uses 4 cores, so the
  host would carry 2 GPUs of COCO-size work (about 8 cores) if the loop-thread limit does not bite earlier.

#### Not measured

- **(d) Region stage.** It needs the segmenter next to Triton. GPU 2 has the memory (about 35 GB
  free beside the 14 GB Triton uses), but the segmenter image downloads gated SAM weights with an
  `HF_TOKEN`, none is available to this run and no copy of the weights exists on the host.
- **(e) Per-GPU scaling.** GPU 0 and GPU 1 hold containers of other projects (GPU 0 was busy at up
  to 95 percent during one of the windows), so PE on GPU 0, on both A6000s and on the 3080 Ti was
  not run. The harness reports per GPU (`--gpu 0,2`); its arm grammar has no GPU-list
  knob yet.
- **Cluster pass** after ingest: skipped in these runs; the Wave 1 figure (159 s) stands.
- **Faces and OCR inside ingest**: ingest does not call them; their cost is probe 3.

#### Caveats

- One host and one card, 3 repetitions, 1,000-image subsets for the sweeps. The host was shared:
  load average 5 to 59 and whole-host CPU busy up to 77 percent from other projects; the foreign
  load on the card itself was zero. The GPU-bound rows are insensitive to it; the host-bound rows
  (set B, light endpoints) are not, and their spread is wide (set B 8 x 16 batch: 6.0 to 9.5).
- The load generator is one Python process with threads. Rows above about 150 requests/s
  (`/detect`, `/embed/image`, `/faces/*` on COCO) may be partly client-bound.
- The py-spy run perturbs the API (17 cores); the census and micro-profile runs do not use it.
- Set B is a public stand-in for high-resolution photo archives: featured pictures are landscapes, wildlife
  and architecture, with fewer detections per image (1.6 items) than a street scene; the CPU per
  image is a property of the pixel count and holds, the PE share is lower than a crowded scene.
- `MAX_FILE_SIZE_MB` was raised for set B (#239).
- Issues filed from this work: #239 (oversize upload answers 500, and the 50 MB cap counts the
  whole request), #240 (one `ocr_pipeline` instance), #241 (photos are API-bound).

### 0.6.0 WP-1.2 and WP-1.3: API CPU attribution and fixes (#210, #211, #239, #241)

The Wave 0 section above ends with the API as the limit on photos (3 to 11 images/s, GPU 5 to 15
percent busy) and 150 of the 180 ms of API CPU per COCO frame unattributed. This section names
those costs, fixes the ones that measure as worth fixing, and repeats the Wave 0 measurements.
Medians with the minimum and maximum of 3 repetitions in parentheses.

**Method.** The Wave 0 stack and protocol on GPU 2 of the same host (an isolated compose project,
loopback ports, the published 0.5.0 images with the branch's `src/` mounted over `/app/src`,
the FP16 PE plan of the Wave 0 reference, plan sha256 `b9a6f397...`, 32 uvicorn workers, a fresh
OpenSearch for each series). Set A is the pinned COCO set (first 1,000 images, `/ingest/upload`, 32
images per request), set B the 500 photos of `commons_hires_500.json` (`/ingest/batch`, 8 images per
request). Foreign load on GPU 2 was 0.0 percent in all 42 measured runs; whole-host CPU busy
was 16 to 22 percent (A) and 18 to 34 percent (B) before the changes, 15 to 17 and 54 to 78 percent
after them (the difference is this stack's own load: it now uses the cores it left idle). The
"before" rows are the code of `origin/main` plus the two Wave 0 branches, measured in the same
session as the "after" rows. Commands: `scripts/bench/ingest_sweep_suite.py --phases sweep` with
`--sweep-points 32x4,32x16,32x32 --modes upload --limit 1000` and
`--sweep-points 8x8,8x16,8x32,8x64 --modes batch`.

#### Attribution (WP-1.2, #210)

The stage timer `op_pipeline_stage_seconds` gained the stages that ran untimed (`hash`, `persist`,
`letterbox`, `triton_request`, `whole_frame_preprocess`, `blur`, `doc_build`, `bulk_serialize`,
`admission_wait`); `resize` was renamed `pe_preprocess` because it is the PE crop preprocessing,
not the detector letterbox. Every stage but `embed` and `opensearch_write` times a synchronous
block, so on the event-loop thread its seconds are API CPU seconds. The table is the code before
any fix with the new timers, 16 clients, 2 repetitions.

| Span | 640 x 480 frames, ms/image | share of API CPU | 12-24 MP photos, ms/image | share of API CPU |
|---|---:|---:|---:|---:|
| `bulk_serialize` (OpenSearch bodies) | 34.3 | 17.2 % | 10.0 | 0.8 % |
| `pe_preprocess` (PE crop resize + normalize) | 25.0 | 12.5 % | 6.8 | 0.6 % |
| `triton_request` (input build, FP32 copy) | 13.9 | 6.9 % | 10.0 | 0.9 % |
| `blur` (Laplacian scores) | 10.4 | 5.2 % | 447.7 | 38.2 % |
| `whole_frame_preprocess` | 8.6 | 4.3 % | 187.8 | 16.0 % |
| `letterbox` | 5.1 | 2.5 % | 93.3 | 8.0 % |
| `decode` | 4.9 | 2.4 % | 303.7 | 25.9 % |
| `jpeg_encode` (crop cache) | 3.4 | 1.7 % | 22.6 | 1.9 % |
| `persist`, `doc_build`, `crop`, `hash` | 2.9 | 1.4 % | 8.3 | 0.7 % |
| **in a span** | **108.4** | **54 %** | **1,090** | **93 %** |
| API CPU per image | 199.8 | | 1,172.8 | |

On photos 93 percent is attributed and four primitives are 88 percent of the API CPU, led by the
blur score, which converted the frame to BGR and took a float64 Laplacian: 398 ms single-threaded
on an 18 MP frame, more than the decode. On frames the spans reach 54 percent. The rest, 91 ms,
was looked at with `py-spy record --native` on one worker (130 ms of API CPU per image with the same
code and two clients, so 32 contended workers cost 1.5 times as much per image):

| Not in a span (frames, one worker) | share of busy samples | ms/image |
|---|---:|---:|
| gRPC client: protobuf serialization and copies of 13.6 MB of FP32 tensors, the socket send | 13 % | 17 |
| the OpenSearch project guard decoding every line of each bulk body (the vectors again) | 6 % | 8 |
| other Python in the ingest functions (documents, OCC merge, ranking, events) | 12 % | 15 |
| uvicorn, multipart, aiohttp, Prometheus, kernel socket I/O | 2 % | 3 |

So with the native profile roughly 85 to 90 percent of the frame CPU is named. The wave plan expected that
the guard "does not parse bodies"; it parses every NDJSON line to find the target indexes. The
FP32 wire (the first row) is WP-2.1 and WP-2.2 and is not touched here.

#### Fixes, ranked by measured gain (WP-1.3, #211, #241)

| Fix | Where it shows | Before | After |
|---|---|---:|---:|
| Pixel work off the event loop on a bounded pool, images admitted by decoded megabytes (#241) | photos, images/s at 8 / 16 / 32 / 64 clients | 5.2 / 8.7 / 10.8 / 4.9 | 21.5 / 27.3 / 23.6 / 24.1 |
| `ndarray.tolist()` and orjson for OpenSearch bodies | frames, `bulk_serialize` ms/image | 34.3 | 0.9 |
| PE normalization through a lookup table (bit-identical) | frames, `pe_preprocess` ms/image | 25.0 | 9.8 |
| Blur score from exact integer sums of a 16-bit Laplacian | frames / photos, `blur` ms/image | 10.4 / 447.7 | 4.6 / 252.2 (398 to about 75 ms single-threaded on 18 MP) |
| Small work stays on the loop (under 4 MiB of pixels) | frames, one worker, API CPU s/image | 0.157 (4 threads; 0.147 with 1) | 0.109 |
| glibc keeps freed buffers (`MALLOC_TRIM_THRESHOLD_` and two more) | frames, one worker, API CPU s/image | 0.109 | 0.096 |
| Project guard parses only the action lines of a bulk body | frames, share of native samples | 6 % | below 1 % |
| Letterbox tensor in one lookup pass, guard checks each bulk target once | frames | not separately measurable (under 5 %) | |
| Triton `--log-verbose=0` | Triton-side logging only | 12,038 lines per 2,000 images | none |

The two rows that touch the thread pool carry the lesson of the work: moving everything off the loop
cost more CPU than it saved on small frames (a pool hop is a GIL hand-over; the same code with the
pool switched off measured 0.092 CPU-s per image against 0.147 to 0.157 with it), and on photos it is what
lets a worker use more than one core. `run_cpu` therefore takes the bytes of pixels a call touches and
runs it in place below 4 MiB. Connections stay on the worker that accepted them, so with 32 clients
only about 14 of the 32 workers hold requests; each needs 8 threads and room for about 20 photos
(1 GiB) to use its share of the cores. Four threads and 256 MiB gave 21.0 and 13.1 images/s at 16 and
32 clients, 8 threads and 1 GiB gave 25.1 and 26.0 (set B, 2 repetitions per arm; 4 threads with 512
MiB gave 19.8 and 15.2, 8 threads with 512 MiB 23.6 and 16.8).

Counting the gate in images instead of megabytes held the GPU-bound stream of frames back (4 images
per worker let a 32-frame request run alone): PE batches fell from 26.6 to 21.9 and the GPU from 92
to 84 percent busy at 16 clients, so it counts decoded bytes read from the image headers.

Not done: the file is decoded twice (Pillow in full for the detector letterbox and the crops,
OpenCV at 1/8 scale for the whole-frame vector) because both entropy-decode the same bitstream;
sharing one decode changes the pixels the whole-frame vector is computed from and so breaks the
bit-identical gate (WP-2.x GPU decode). No JPEG is encoded twice on this path: the crop cache
encodes each crop once and an upload is persisted as the bytes that arrived.

#### Before and after

Set A (COCO, 1,000 images, 32 per request, `/ingest/upload`):

| clients | images/s before | images/s after | API CPU s/image before | after | change | GPU busy % before / after | request p50 s before / after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 17.01 (17.00-17.35) | 19.56 (17.96-19.63) | 0.170 (0.169-0.170) | 0.113 (0.112-0.116) | -33 % | 71 / 80 | 7.4 / 6.5 |
| 16 | 21.59 (21.25-21.61) | 21.58 (21.48-21.64) | 0.187 (0.182-0.191) | 0.126 (0.121-0.129) | -33 % | 92 / 87 | 21.0 / 21.4 |
| 32 | 20.68 (20.11-21.01) | 20.46 (20.31-20.77) | 0.192 (0.186-0.194) | 0.136 (0.131-0.139) | -29 % | 85 / 84 | 45.9 / 47.0 |

Set B (500 photos of 12-24 MP, 8 per request, `/ingest/batch`):

| clients | images/s before | images/s after | API CPU s/image before | after | GPU busy % before / after | request p50 s before / after |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5.22 (5.13-5.87) | 21.50 (20.59-21.51) | 1.102 | 0.990 | 9 / 41 | 9.3 / 2.7 |
| 16 | 8.68 (8.29-9.31) | 27.26 (24.56-27.32) | 1.129 | 1.103 | 15 / 45 | 10.7 / 4.0 |
| 32 | 10.78 (10.20-11.28) | 23.57 (19.90-26.21) | 1.168 | 1.129 | 16 / 33 | 15.9 / 7.2 |
| 64 | 4.87 (4.85-5.24) | 24.13 (23.71-25.28) | 1.225 | 1.090 | 8 / 37 | 48.1 / 11.7 |

Against the targets:

- **Photos: met.** Throughput is 2.2 to 4.9 times higher and no longer falls past 32 clients (24.1
  at 64 against 23.6 at 32; it fell from 10.8 to 4.9). The CPU per image on photos barely moves
  (1.10-1.23 to 0.99-1.13): the gain is parallelism, not less work, and the host is now the limit
  (68 to 76 percent busy, 27 to 30 cores, GPU 33 to 45 percent busy).
- **COCO frames, API CPU per image down at least 40 percent: not met.** It is down 29 to 33 percent
  (0.170-0.192 to 0.113-0.136 CPU-s). What is left is the FP32 wire (13.6 MB per image through
  protobuf and the socket, about 25 ms of the 96 ms single-worker figure) and the decode, letterbox
  and whole-frame preprocessing that parity fixes in place. Images/s is not lower (GPU-bound, within
  the spread). The p99 per batch was not measured; the p50 is in the table.
- The attribution reaches 93 percent on photos and 54 percent from the stage timers on frames (roughly 85 to 90 percent with the native profile above).

#### Accuracy

`scripts/bench/parity.py` (PR #238 tool), one capture of this branch on the isolated stack (2,000 COCO
images, policy `all`, clusters trained by the auto-label worker, 200-image core-route probe) against the FP16
reference capture `ref-fp16` (same plan sha256, code of `0567748f`), profile `precision`:

| Gate | Metric | Observed | Rule | Status |
|---|---|---:|---|---|
| P-box | `boxes.images_identical_share` | 0.99191 | >= 0.98 | PASS |
| P-box-confident | `boxes.unmatched_confident` | 0 | <= 0 | PASS |
| P-item | `boxes.item_count_class_identical_share` | 0.99949 | >= 0.99 | PASS |
| P-pe-p01 | `pe_crop.p01` | 0.99993 | >= 0.995 | PASS |
| P-pe-median | `pe_crop.median` | 1.00000 | >= 0.999 | PASS |
| P-pew-p01 | `pe_frame.p01` | 0.99993 | >= 0.995 | PASS |
| P-pew-median | `pe_frame.median` | 1.00000 | >= 0.999 | PASS |
| P-knn | `knn.overlap_at_k` | 0.99610 | >= 0.95 | PASS |
| P-dup | `knn.near_dup_agreement` | 1.00000 | >= 0.995 | PASS |
| P-cluster-assign | `cluster.assignment_agreement` | 0.99922 | >= 0.98 | PASS |
| P-cluster-purity | `cluster.purity_drop_points` | -0.19547 | <= 0.5 | PASS |
| P-label-vote | `knn.label_vote_agreement` | 0.99700 | >= 0.97 | PASS |
| P-knn-precision | `knn.precision_drop_points` | -0.07792 | <= 0.5 | PASS |
| P-label | `labels.class_agreement` | 1.00000 | >= 0.99 | PASS |
| P-vlm | `labels.vlm_agreement` | n/a | >= 0.97 | SKIP |
| P-clip | `clip.p01` | 1.00000 | >= 0.995 | PASS |
| P-face | `face.p01` | 1.00000 | >= 0.995 | PASS |
| P-ocr | `ocr.exact_share` | 1.00000 | >= 0.98 | PASS |

Verdict PASS (`docs/benchmarks/parity/api_cpu_vs_fp16_ref/`). The candidate has 12,791 items against 12,790;
12,790 matched. The PE vectors are not bit-identical between two captures (identical share 0.946 for crops,
0.939 for frames) because the FP16 engine's result depends on the batch it ran in and the batches differ from
run to run; the median cosine is 1.00000 and the 1st percentile 0.99993. The reference configuration measured
against itself is the noise floor (about 1 percent of crop rectangles differ). P-vlm is skipped: no VLM was
configured. The changes that touch numbers were also checked against their old implementations
directly, on real images, which a GPU-in-the-loop capture cannot do: in the API container, on 400 COCO
frames and 40 photos (440 images, 1,320 random boxes), the PE normalization of the whole-frame and crop
tensors, the detector letterbox tensor, the full-frame Laplacian variance and the box variances were
compared with the previous code, bit for bit for the tensors and at the stored two decimals for the
variances: 0 of 3,960 comparisons differ. The same equalities are unit tests (every uint8 value on every
channel, random frames and crops, extreme and tiny images).

Raw results: `docs/benchmarks/v060_apicpu_{before,after}_set_{a,b}.json` and `v060_apicpu_attribution_set_{a,b}.json`.

### Bulk ingest runner

`POST /curation/projects/{project}/ingest/bulk` runs a folder or manifest through the same `POST /ingest/batch`
route as any other client (see [CURATION.md](CURATION.md#bulk-ingest)), so it is held to the same ceiling: the
GPU. Set A (public COCO, 2,000 images, 640 x 480), one RTX A6000 (GPU 0) shared with other projects'
resident processes (16.4, 1.4 and 0.3 GB), FP16 PE engine, 32 API workers, OpenSearch heap 8 GB, a fresh
project per run, policy `all`. The plain path is a Python client with threads posting `/ingest/batch` (16 images
per request).

| Path | Batches in flight | Run 1 | GPU 0 busy % in run 1 | Run 2 |
|---|---:|---:|---:|---:|
| `/ingest/batch` from client threads | 8 | 19.74 | 78 | 20.72 |
| Bulk ingest runner | 8 | 20.75 | 86 | 18.38 |
| `/ingest/batch` from client threads | 16 | 20.65 | 89 | |
| Bulk ingest runner | 16 | 21.29 | 90 | |

Images per second. The runner is within the spread of the plain path (18.4 to 21.3 across the six runs of both):
it adds no throughput and costs none, which is the point of sending each batch through the route that the data
path is tuned behind. The second pair of 8-in-flight runs was sampled with `nvidia-smi pmon`: the other processes on GPU 0 held 0 percent
SM throughout; the other runs sampled only the card's total utilization, so a foreign load there is not
excluded. A cancel at 784 images followed by a start of the same source finished at exactly 2,000 stored documents
with 0 duplicates and 784 images skipped from the ledger.

The first design ran the pipeline inside the task that owns the run, so all batches shared one `api`
worker's event loop. That reached 9.85 images/s (one process is limited by its host CPU, about 0.1 CPU-s per
image on frames; the stage timers showed `opensearch_write` at 2.5 s per image only because the loop was busy).
Sending the batches back to the API's own socket spreads them over every worker and gives the 18 to 21 above.

OpenSearch is not the limit at this rate (one writer sustains the equivalent of 137 images/s, probe 5 above), so
the writes stay per image inside the pipeline; the bulk size of 1,000 documents from probe 5 is a datastore
property, not a runner setting.

### WP-2.1: uint8 inputs for the detector and the PE image encoder (#214)

The detector and PE engines take uint8 pixels and do the scaling (`/255`; `(x / 255 - 0.5) / 0.5`)
inside the TensorRT graph. Before, the API built an FP32 tensor per image (4.9 MB for the detector,
10.0 MB for the PE crops on the COCO set), serialized it into the gRPC request and Triton copied it
back out. Raw numbers: `docs/benchmarks/uint8_inputs.json`; parity summaries:
`docs/benchmarks/parity/uint8_*`. Medians with the minimum and maximum of 3 repetitions in
parentheses unless noted.

**Headline.** Request bytes to Triton per image fall 75.0 percent (14.94 to 3.73 MB on set A, 8.49 to
2.12 MB on set B). On set A, API CPU per image falls 47 to 49 percent (0.118 to 0.062 CPU-s upload,
0.117 to 0.060 batch), Triton CPU per image falls 64 percent, and images/s rise 10 to 12 percent
(19.1 to 21.0 upload, 19.0 to 21.4 batch), with the GPU 88 to 91 percent busy: the ingest is GPU-bound
on the PE engine again. On 12-24 MP photos (set B) the API CPU per image is unchanged (1.08 to 1.10
CPU-s): the cost there is decode and resize, and the host was saturated by other projects, so that
throughput reading is not usable (below). The engines themselves cost nothing for the detector
(5.7 against 5.6 ms per batch of 8) and 6 to 8 percent more for PE (see Engine-only).

#### What changed

| | Before | After |
|---|---|---|
| Detector input | FP32 `[B, 3, 640, 640]`, `/255` on the CPU (lookup table) | UINT8, `/255` in the graph |
| PE input | FP32 `[B, 3, 336, 336]`, `(x / 255 - 0.5) / 0.5` on the CPU (lookup table) | UINT8, same arithmetic in the graph |
| Where the prefix is added | n/a | `export/uint8_input.py`, after the FP16 bake (the bake keeps the graph I/O FP32, so the prefix sits in front of its first cast) |
| Removed | | `fp32_images_input`, `normalize_chw`, `normalize_into`, `PE_MEAN`, `PE_STD`, the unit-range table of the letterbox, `infer_yolo_batch` (no callers) |

There is no FP32 fallback. An engine built before this change keeps its FP32 plan next to a
`config.pbtxt` that now declares `TYPE_UINT8`; Triton refuses to load that pair
(`unexpected datatype TYPE_FP32 for inference input 'images', expecting TYPE_UINT8`), the model is not
READY, and the installer's skip test (`group_should_skip`) rebuilds it. Models that keep float32 input
because they are not repository-built TensorRT engines: promoted trained models served by the ONNX
Runtime backend (`output0`, used by the region detector) and the bake-off engines
(`scripts/curation/bakeoff`). MobileCLIP, SCRFD, ArcFace and OCR are unchanged.

#### Method

One RTX A6000 (GPU 2), the published 0.5.0 images with the checkout's `src/` mounted over `/app/src`,
an isolated compose project on loopback, OpenSearch 8 GB heap, 32 API workers, a fresh stack per
configuration, policy `all`. Set A: the pinned COCO set (2,000 images, `coco_bench_2000.json`),
`scripts/bench/baseline_suite.py --phases env,ingest --reps 3 --skip-cluster --skip-settle` (32 images
per request, 4 clients; upload and batch). Set B: the 500 pinned photos, `scripts/bench/ingest_sweep_suite.py
--phases sweep --sweep-points 8x16 --modes batch`. Configurations: **base** is `origin/main` with the
detector plan of the 0.5.0 install (Sep 26 build) and an FP16 PE plan from `build_pe_trt.sh`;
**detector uint8** is the first commit of this change (PE still FP32); **final** is both. The uint8
detector plan was built with the real exporter path (`export_models.export_trt_end2end`), the PE plan
with `build_pe_trt.sh`, both on the TensorRT 11.1 of the Triton image.

#### Request bytes and ingest, set A (2,000 images)

| Metric | base | detector uint8 | final |
|---|---:|---:|---:|
| Detector request MB per image | 4.92 | 1.23 | 1.23 |
| PE request MB per image | 10.02 | 10.02 | 2.50 |
| Triton request MB per image | 14.94 | 11.25 | 3.73 |
| `/ingest/upload` images/s | 19.09 (17.89-19.10) | 20.77 (20.17-22.47) | 20.98 (20.97-21.05) |
| `/ingest/batch` images/s | 19.04 (18.97-20.38) | 20.43 (19.14-21.72), 2 reps | 21.41 (21.20-22.31) |
| API CPU s/image, upload | 0.118 (0.110-0.120) | 0.084 (0.083-0.087) | 0.062 (0.060-0.062) |
| API CPU s/image, batch | 0.117 (0.116-0.119) | 0.083 (0.082-0.084) | 0.060 (0.060-0.061) |
| Triton CPU s/image, upload | 0.036 (0.035-0.039) | 0.023 (0.022-0.024) | 0.013 (0.013-0.013) |
| `triton_request` stage ms/image, upload | 3.94 | 1.64 | 0.53 |
| `pe_preprocess` stage ms per image, upload | 9.60 | 9.26 | 5.79 |
| Request p50 s, upload | 6.6 (6.6-7.2) | 6.1 (5.6-6.4) | 6.1 (6.1-6.1) |
| GPU 2 utilization mean %, upload | 79 (72-80) | 83 (81-89) | 88 (87-91) |
| PE mean batch, upload | 18.0 | 18.6 | 20.4 |
| Failed items | 0 | 0 | 0 |

The three configurations were measured in sequence on the same host while other projects used it
(foreign SM load on GPU 2 was 0.0 percent in every sampled ingest run, but the host CPU was not
controlled), so the CPU-s columns carry host contention; the three ranges do not overlap, and the saving is the right size
for the bytes removed (serialization, the CPU copies and the lookup-table normalize). The batch row
of "detector uint8" has 2 repetitions: the harness failed on the cleanup `DELETE` of the third batch
project with a 409 after its ingest had finished (a harness or project-delete race, unrelated to the
input type; the other 5 runs completed).

#### High-resolution photos, set B (500 photos, `/ingest/batch`, 8 images x 16 clients)

| Metric | base | final |
|---|---:|---:|
| Triton request MB per image | 8.49 | 2.12 |
| API CPU s/image | 1.083 (1.071-1.113) | 1.101 (1.096-1.110) |
| Triton CPU s/image | 0.027 (0.027-0.029) | 0.019 (0.018-0.021) |
| images/s | 8.55 (7.86-13.52) | 14.90 (7.64-15.52) |
| Host CPU busy % | 97 (96-100) | 96 (95-99) |
| Host load average (1 min) | 191 (132-192) | 84 (84-98) |

The host was saturated by other projects during both series (load average 84 to 192 on 48 logical
cores), so the images/s columns and their 2x spread are not evidence of a gain or a loss; only the
bytes and the CPU-s per image are comparable. API CPU per photo does not move: it is decode and resize.

#### Engine-only (Triton alone, `perf_analyzer`, gRPC)

These numbers have no API, no preprocessing and no OpenSearch in them. Each pair is measured in one
Triton, back to back, on GPU 2. `compute_infer` is Triton's per-execution engine time.

| Model | Batch | Concurrency 1: images/s | `compute_infer` ms | Saturated: images/s |
|---|---:|---:|---:|---:|
| Detector, Sep-26 FP32 plan | 8 | 118.1 | 5.7 | 516.9 |
| Detector, rebuilt FP32 plan | 8 | 115.8 | 5.6 | 545.5 |
| Detector, uint8 plan | 8 | 341.5 | 5.7 | 1612.0 |
| Detector, Sep-26 FP32 plan | 32 | 141.1 | 18.3 | 265.2 |
| Detector, rebuilt FP32 plan | 32 | 142.1 | 18.4 | 306.1 |
| Detector, uint8 plan | 32 | 411.3 | 18.4 | 1501.9 |
| PE, FP32 plan | 8 | 132.6 | 38.8 | 193.8 |
| PE, uint8 plan | 8 | 152.4 | 42.0 | 181.1 |
| PE, FP32 plan | 32 | 129.0 | 160.2 | 195.6 |
| PE, uint8 plan | 32 | 156.1 | 169.8 | 182.7 |

The engine time of the detector is unchanged; what `perf_analyzer` gains is the request path (the
FP32 requests were the limit at 120 to 140 images/s). The PE engine is 6 to 8 percent slower with the
prefix (`compute_infer` 42.0 against 38.8 ms at batch 8, 169.8 against 160.2 ms at batch 32); under
the real ingest the PE time per image was the same (5.5 ms base, 5.6 ms final, from
`exec_infer_ms / mean_batch`), the GPU is simply the limit now. The cause of the 6 to 8 percent is not
isolated: the extra cast and arithmetic run in FP32 in front of the FP16 first layer, and the two PE
plans are separate builds, so tactic choice may contribute. A first pass of `perf_analyzer` on the
final stack (PE 94 images/s, detector 780, with `compute_infer` about twice the values above for both
models) is discarded: nothing in the code explains a 2x slowdown of both engines at once, the repeat
above does not reproduce it, and foreign load on GPU 2 was not sampled during `perf_analyzer` (it
was 0.0 percent in every ingest run). Both passes are in the JSON.

#### Parity

`scripts/bench/parity.py` captured the pinned set through each configuration (12,790 items, 2,000
frames, 200 faces, 200 MobileCLIP, 34 text lines) and compared against the FP16 reference of the
FP16 PE section (`ref-fp16`) with the `precision` profile.

| Comparison | P-box | P-item | P-pe-p01 | P-pew-p01 | P-pew-median | P-knn | P-cluster-assign | P-cluster-purity | P-label-vote | unmatched conf. | gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unchanged stack vs `ref-fp16` (noise floor) | 0.9939 | 0.9990 | 0.9996 | 0.9993 | 0.9999 | 0.9822 | 0.9982 | 0.0469 | 0.9880 | 0 | none |
| detector uint8, run 1 vs run 2 (noise floor) | 0.9919 | 0.9995 | 0.9999 | 1.0000 | 1.0000 | 0.9942 | 0.9995 | 0.9068 | 1.0000 | 0 | P-cluster-purity |
| detector rebuilt FP32 vs `ref-fp16` (rebuild, no uint8) | 0.9353 | 0.9707 | 0.9779 | 0.9993 | 0.9999 | 0.9437 | 0.9922 | 1.0973 | 0.9710 | 2 | P-box, P-box-confident, P-item, P-pe-p01, P-knn, P-cluster-purity |
| detector uint8 vs rebuilt FP32 | 0.9848 | 0.9960 | 0.9968 | 1.0000 | 1.0000 | 0.9934 | 0.9991 | -1.3685 | 0.9990 | 0 | none |
| detector uint8, run 2 vs rebuilt FP32 | 0.9818 | 0.9954 | 0.9964 | 1.0000 | 1.0000 | 0.9928 | 0.9988 | -0.4692 | 0.9980 | 0 | none |
| PE uint8 added, vs detector uint8 run 1 | 0.9934 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 0.9950 | 0.9995 | 1.1961 | 0.9980 | 0 | P-cluster-purity |
| PE uint8 added, vs detector uint8 run 2 | 0.9944 | 0.9995 | 0.9999 | 1.0000 | 1.0000 | 0.9950 | 0.9995 | 0.2892 | 0.9980 | 0 | none |
| final vs rebuilt FP32 | 0.9833 | 0.9960 | 0.9971 | 1.0000 | 1.0000 | 0.9930 | 0.9992 | -0.1799 | 0.9970 | 0 | none |
| final vs `ref-fp16` | 0.9343 | 0.9697 | 0.9779 | 0.9993 | 0.9999 | 0.9468 | 0.9923 | 0.9327 | 0.9740 | 2 | P-box, P-box-confident, P-item, P-pe-p01, P-knn, P-cluster-purity |

Three findings, in order of importance.

1. **The uint8 inputs pass every gate against the right control.** Against an FP32 detector rebuilt
   from the same ONNX with the same bake, the uint8 detector passes all gates in both runs, and adding
   the PE change moves the whole-frame vectors by cosine 1.0000 (1st percentile 0.99996) and the crop
   vectors by 0.9999 (1st percentile). The remaining spread is the detector's own run-to-run noise:
   crop rectangles are identical on 98.7 percent of items against the control and 99.1 to 99.2 percent
   between two runs of the same engine.
2. **Rebuilding the detector engine, with no input change, fails the gates against `ref-fp16`.**
   The 0.5.0 detector plan (Sep 26) was built from an ONNX baked before the FP16 cast fix of the FP16
   PE section; the plan rebuilt today from the same weights moves 6.5 percent of images beyond the box
   gate, 2 confident boxes go unmatched, and crop-vector agreement drops to 0.978 (1st percentile). This
   is independent of the input type (the "rebuilt FP32" row has no uint8 in it) and applies to every
   deployment that rebuilds its detector, so it is a finding for the engine pipeline, not for this
   change; it is why this change is gated against a rebuilt FP32 control and not only against the
   reference. The uint8 detector against `ref-fp16` fails exactly as the rebuilt FP32 one does.
3. **P-cluster-purity is below the noise of the capture.** Two runs of the same uint8 detector differ
   by 0.91 purity points (gate 0.5) because the detector is not bit-reproducible and the cluster
   training is seeded on the items it gets; the single failure of the PE comparison (1.20 against
   run 1) goes away against run 2 (0.29). The purity gate needs a repeat-run noise allowance before
   it can separate a 0.5 point effect.

On the question of exactness: the in-graph arithmetic is the same arithmetic as the removed lookup
table (cast, divide by 255, subtract, divide), verified bit for bit by ONNX Runtime on all 256 values
for both normalizations (`tests/test_uint8_input.py`); on the GPU, TensorRT fuses the prefix and
rounds to FP16 once, which is where the 0.9999 and the 98 percent box identity come from.

#### Caveats

- One host, one card, 3 repetitions, other projects running on the same host and CPUs. Set B
  throughput is not usable (host load average 84 to 192).
- Set A, public COCO only; set B, the public photo set. Retrieval and clustering gates on labelled
  private data were not run. The VLM gate is skipped (no VLM in the stack).
- The first `perf_analyzer` pass on the final stack is discarded as an unexplained 2x outlier (above).
- Core routes (`/detect`, `/embed`, `/analyze`) send uint8 to the detector through the same engine
  and were exercised by the 200-image probe of `parity.py capture`; MobileCLIP, SCRFD, ArcFace and
  the OCR models still receive FP32 (they are separate packages if they show up in the profile).
- The dual-head detector export and the YOLO26 export follow the same prefix and are covered by unit
  tests only; no engine of either was built or served in this run.
- The `DELETE /curation/projects/{project}` 409 seen once at the end of a batch series is not
  diagnosed here.

### WP-2.2: system shared memory between the API and Triton (#215), measured and not adopted

**Result.** Shared memory removes 1.9 ms of client CPU per detector image and 0.5 to 0.6 ms per PE image
when the transport is measured alone (about 6 ms of the 65 to 75 ms of API CPU per COCO frame), but the
ingest does not get cheaper: API CPU per image is 0 to 3 percent lower (inside a run-to-run spread of
8 to 10 percent), and Triton spends more user CPU than the wire path saved (+5.6 ms per image against
-2.6 ms of kernel time), so the host total does not fall. Images per second rise 2 to 12 percent and the
request p50 falls 5 to 8 percent, but the ranges overlap. The plan's rule (close the package when
the win is under 5 percent of API CPU) applies, so the package is closed as measured and **nothing is
shipped**: no setting, no compose change, no client code. The implementation and its tests are two
commits on the local branch `perf/shared-memory-impl` (`4de40001`, `8e20afe5`), for reference only.

**Method.** The isolated stack of the uint8 section (GPU 2, foreign load 0.0 percent in every sampled
run, published 0.5.0 images with the branch `src/` mounted, uint8 detector and PE plans, PE plan sha256
`a4feb7df...`, a fresh project per repetition, policy `all`), one extra tmpfs volume mounted as `/dev/shm`
in both containers and `--allow-client-shm=true` on Triton. Wire and shared memory were the same code with
the transport switch off or on, the API container recreated between launches, arms interleaved (wire, shm, shm,
wire and shm, wire, shm) so a drifting host load falls on both. Host load average (1 min) was 12 to 62
at launch and 15 to 43 at the end: other projects were running throughout. Medians with the minimum and
maximum of the repetitions in parentheses. Raw numbers: `docs/benchmarks/shared_memory_transport.json`.

**Attribution before.** Native `py-spy` on one worker (2 clients, 45 s, 50 Hz, uint8 tensors) put 13 percent
of the busy thread samples in client-side transport (protobuf serialization 4.9, other gRPC client 4.4,
socket send 2.9, the `tobytes` copy 1.1); that counts every thread of the process, so it is a ceiling, not a
CPU share. The cgroup split of one worker is the better number: 14.4 of 66.9 ms of API CPU per image is
kernel time (sockets, page faults, futex) and 52.5 ms is user time. Triton spends 18.7 ms per image
(14.1 user, 4.6 kernel).

**Isolated transport cost** (sequential `AsyncTritonPool.infer_images`, one process, 3 repetitions, client
CPU ms per image):

| Request | Wire | Shared memory |
|---|---:|---:|
| Detector, batch 16 (19.7 MB) | 2.25 (2.17-2.31) | 0.34 (0.33-0.36) |
| PE, batch 9 (3.0 MB) | 0.89 (0.83-0.93) | 0.32 (0.28-0.32) |
| PE, batch 32 (10.8 MB) | 0.67 (0.66-0.69) | 0.17 (0.15-0.18) |

One frame is one detector image and about 7.4 PE images, so the isolated saving is about 6 ms per frame.
With raw `tritonclient` calls, Triton CPU per image also falls in isolation (detector batch 8: 2.26 to
0.89 ms; PE batch 16: 0.92 to 0.47 ms).

**Ingest, set A (2,000 COCO images, 32 workers, 4 clients, 32 images per request, 9 repetitions per cell).**

| Endpoint | Transport | images/s | API CPU s/image | Triton CPU s/image | request p50 s | GPU 2 busy % |
|---|---|---:|---:|---:|---:|---:|
| `/ingest/upload` | wire | 20.68 (19.18-22.28) | 0.0747 (0.0697-0.0866) | 0.0133 (0.0123-0.0145) | 6.15 (5.79-6.40) | 83 (80-91) |
| `/ingest/upload` | shared memory | 21.66 (16.62-22.67) | 0.0739 (0.0679-0.0809) | 0.0131 (0.0120-0.0169) | 5.71 (4.92-7.39) | 88 (70-96) |
| `/ingest/batch` | wire | 20.86 (19.99-21.79) | 0.0754 (0.0713-0.0799) | 0.0131 (0.0120-0.0138) | 5.95 (5.67-6.51) | 87 (80-92) |
| `/ingest/batch` | shared memory | 21.29 (20.10-22.01) | 0.0755 (0.0682-0.0794) | 0.0143 (0.0124-0.0155) | 5.67 (5.08-5.83) | 91 (84-94) |

After the first two launches shared memory looked 4 to 8 percent cheaper (0.0695 against 0.0724 s/image
on batch, 0.0680 against 0.0742 on upload, non-overlapping ranges, 3 against 6 repetitions); the later
launches removed it. Three repetitions per arm are not enough on this host. Throughput is bound by the GPU in both arms (83 to 91 percent busy), so the transport cannot
raise images per second by more than the idle gaps it closes.

**One worker, 2 clients (1,000 images, 9 repetitions per cell, `/ingest/upload`).**

| Transport | images/s | API CPU s/image | Triton CPU s/image | request p50 s | GPU 2 busy % |
|---|---:|---:|---:|---:|---:|
| wire | 10.14 (7.83-12.66) | 0.0659 (0.0620-0.0733) | 0.0169 (0.0143-0.0191) | 5.22 (4.21-7.00) | 44 (33-57) |
| shared memory | 11.31 (10.15-11.93) | 0.0653 (0.0600-0.0733) | 0.0203 (0.0192-0.0211) | 4.82 (4.41-5.27) | 46 (42-54) |

User and kernel CPU of the two cgroups for the same run (ms per image, 3 repetitions per arm):

| Transport | API user | API kernel | Triton user | Triton kernel |
|---|---:|---:|---:|---:|
| wire | 52.5 (51.4-52.8) | 14.4 (13.8-14.8) | 14.1 (13.9-15.4) | 4.6 (4.2-5.2) |
| shared memory | 52.6 (51.9-54.6) | 12.3 (11.5-13.3) | 19.7 (19.5-20.0) | 2.0 (2.0-2.3) |

API kernel time falls 2 ms and Triton kernel time 2.6 ms, as the isolated numbers predict. Triton's user
time rises 5.6 ms. **Hypothesis, not verified:** a request whose input lives in a client-registered
region is copied to the GPU from pageable host memory, which the CUDA driver stages on the CPU, while a
gRPC request lands in memory Triton already owns; the isolated run is a single stream and does not show
it. Registering the regions as pinned memory would be the experiment to run before revisiting this.

**Photos, set B (500 pinned 12-24 MP photos, `/ingest/batch`, 8 images x 16 clients, 3 repetitions per
arm, one launch per arm).** Wire 20.43 (12.10-20.82) images/s, API CPU 1.187 (1.129-1.309) s/image;
shared memory 17.16 (15.05-19.51) images/s, 1.082 (1.061-1.129) s/image. The two arms differ by more
than the transport can explain (decode and resize are 1.1 s of the CPU, the transport at most 6 ms) and
the host load average was 29 to 44: these are noise, not a result. The ABBA series was stopped after two
launches.

**Parity.** `parity.py capture` of the shared memory stack (2,000 images, policy `all`, 200-image probe)
against the references of the uint8 section, profile `precision`; the last column is the two references
against each other (the noise floor of the pair):

| Gate | shm vs `ref-u8fp16` | shm vs `ref-u8fp16-repeat` | `ref-u8fp16` vs repeat |
|---|---:|---:|---:|
| P-box (images with identical boxes) | 0.9929 | 0.9949 | 0.9939 |
| P-item | 0.9995 | 0.9995 | 1.0000 |
| P-pe, 1st percentile (median 1.0000) | 0.99993 | 0.99993 | 0.99995 |
| P-pew, 1st percentile (median 1.0000) | 0.99991 | 0.99991 | 0.99994 |
| P-knn overlap | 0.9943 | 0.9949 | 0.9955 |
| P-cluster-assign | 0.9995 | 0.9995 | 0.9997 |
| P-cluster-purity drop (points, gate 0.5) | 0.27 | -0.18 | 0.45 |
| P-label-vote | 0.997 | 0.998 | 0.999 |
| Gates failed | none | none | none |

Direct check on real frames, the same tensors through both transports: PE embeddings are bit-identical
at batch 1, 12 and 32. The detector's `det_boxes` differ by up to 0.012 between the transports, and by the
same amount between two wire runs of the same tensor (the engine is not bit-reproducible, as the uint8
section found); `num_dets`, scores and classes are identical.

**What was built** (for a later revisit; none of it is in the tree). The detector and PE image batches
only; outputs (4.8 KB per detector image, 4 KB per PE image) stay on the wire. One tmpfs volume mounted as
`/dev/shm` in `triton-server` and `api` (project-scoped, sized by `SHM_SIZE`, so two compose projects never
share a region namespace) and `--allow-client-shm` on Triton. Each API process owned up to four regions
named `<namespace>.<boot>.<pid>.<n>` (`boot` random per start, so a recycled pid never collides), sized to
the next power of two, grown on demand, never shrunk. A request leased a region, copied the batch in and
sent the region name. Cleanup: the process held an `flock` on its lock file; at every start a process swept
the namespace, removing the Triton registrations and files of any owner whose lock it could take, and any
registration with no lock. A failed request retired its region. The start probed one region and refused to
start (no fallback to the wire) if Triton could not map it. Checked live: `docker kill -s KILL` of the API
left 16 registrations of 4 owners, the next start reaped 32 owners and left none; a graceful recreate left
none and removed every file; a Triton restart was recovered by registering the region again (the first
version matched the wrong error text and failed 3 requests of 128 before the slot was retired; fixed and
rerun with 0 failed of 256). The 700-line ceiling held (`triton_pool.py` at 698).

**Not measured.** CUDA shared memory (excluded by design: the API and Triton may sit on different GPUs, and
the known `tritonclient` issue with multithreaded clients); pinned or registered host regions; writing the
letterbox straight into the region (the `tobytes` copy is 1.1 percent of the samples above, so at most
that); the core routes (`/detect`, `/embed`), which use the synchronous client; MobileCLIP, SCRFD, ArcFace
and OCR, which still send FP32; more than one host; a Triton instance count other than the shipped one.

**Caveats.** One host, one card, 9 repetitions per cell on set A and a shared CPU with load averages of 12
to 62. A difference of 5 percent in API CPU would need more repetitions than this to resolve; the
conclusion is that the effect is smaller than that, not that it is zero. The `DELETE
/curation/projects/{project}` 409 of the uint8 section occurred once more (the third launch, `/ingest/batch`
repetition 1); it is still not diagnosed, and it cost that launch its batch repetitions, which were
repeated in a later launch.

### Multi-GPU PE encoding (#228)

The PE image encoder is about 99 percent of the GPU work of `/ingest`, and one RTX A6000 tops out
near 23 images per second on 640 x 480 frames (Wave 0). This section measures running the encoder
on several cards. Medians with the minimum and maximum of 3 repetitions in parentheses. Raw JSON
and logs stay in the scratch directory of the run; the commands are `scripts/bench/ingest_sweep_suite.py
--phases env,sweep --modes batch --reps 3 --sweep-points 32x16` (set A) and `8x16` (set B),
`perf_analyzer` for the engine-only rows, and `scripts/bench/pe_gpu_parity.py`.

**Design chosen.** One Triton, `pe_image_encoder` with `instance_group { count: 1, kind: KIND_GPU }`
and no `gpus` list, so Triton starts one instance on every card the container sees and one
dynamic-batcher queue feeds them all. `TRITON_GPU_ID` takes a comma list (`0,2`); every other model
stays pinned to `gpus: [ 0 ]`, the lowest listed card (`CUDA_DEVICE_ORDER=PCI_BUS_ID` makes that
true; see the finding below). No client change, no second service, and a card that is free takes the
next batch, so unequal cards share the work by speed. The alternative, one Triton per card behind a
client that balances, needs a balancing client in the API, one compose service and port per card,
and gains nothing the first layout does not already deliver (engine-only table).

| Item | Value |
|---|---|
| Cards | RTX A6000 (GPU 2, GPU 0), RTX 3080 Ti (GPU 1, 12 GB), PCIe 3.0 x16, GPU 0 and 1 on NUMA node 0, GPU 2 on node 1; same compute capability (8.6) |
| Stack | the repository `docker-compose.yml` with the checkout's `src/` mounted, project `mgpu`, loopback ports, OpenSearch heap 8 GB, 32 API workers, no curation workers (`--skip-cluster --skip-settle`) |
| Engines | the uint8 FP16 PE plan of `ref-u8fp16` (plan sha256 `a4feb7df...`), one plan file for every card |
| Load | set A (2,000 COCO images, 32 per request, 16 clients), set B (500 photos, 8 per request, 16 clients), policy `all` |
| Foreign load | the host and GPUs 0 and 2 were shared with other projects. Every repetition records the foreign SM share per card; only repetitions with 0.0 percent foreign load on every card used are reported as results |

#### Engine only (`perf_analyzer`, batch 32, gRPC)

| Layout | images/s |
|---|---:|
| RTX A6000 alone | 180.9 |
| RTX 3080 Ti alone | 197.3 |
| (a) one Triton, the encoder on both | 386.1 (102 percent of the two alone) |
| (b) two Triton processes, both driven at once | 377.9 (197.8 + 180.1) |

Ingest sends 7.4 encoder crops per image on set A (6.4 item crops and the whole frame), so 181
crops/s is 24.4 images/s for one A6000 and 386 crops/s is 52 images/s for the pair.

#### Ingest, set A (2,000 images, `/ingest/batch`)

| Cards | images/s | scaling | request p50 s | GPU util % | API CPU s/image | host busy % |
|---|---:|---:|---:|---|---:|---|
| 1 x A6000 (GPU 2) | 23.39 (22.96-23.41) | 1.00 | 22.3 (21.5-22.4) | 95-98 | 0.065-0.071 | 26-42 |
| 2 x A6000 (GPUs 0 and 2) | 44.60 (44.29-45.47) | 1.91 | 11.0 (10.9-11.1) | 87-93 / 89-93 | 0.060-0.068 | 22-26 |
| A6000 + 3080 Ti (GPUs 2 and 1) | 43.72 (42.46-45.94) | 1.87 | 10.2 (10.1-10.6) | 84-90 / 81-90 | 0.062-0.079 | 29-82 |

Two A6000 pass the 1.8x gate at 1.91x (95 percent of two cards), with the encoder instances 87-93
percent busy on both cards and the host 22-26 percent busy: the API (1.7 cores per card at 0.06 CPU-s
per image) is not the limit for frames. The three clean repetitions are 44.29 and 45.47 from one
series and 44.60 from another; the other three repetitions of those series ran while other projects
held 11-30 percent of GPU 0's SMs and are discarded (37.2-39.9 images/s).

One A6000 reaches 96 percent of its engine-only capacity (23.39 against 24.4 images/s); the
A6000 + 3080 Ti pair reaches 84 percent of its own (43.72 against 52), and the best repetition
(45.94, host 29 percent busy) 88 percent. The slow repetitions of the pair ran while other
projects pushed the host load average to 54-74 on 48 cores, which starves the API (GPU 81-84 percent
busy, PE mean batch 1.7 against 1.8). The first one-card series on GPU 0 and on GPU 2 are discarded:
other projects held 5-31 percent of those cards' SMs (15.9-22.2 images/s).

#### Ingest, set B (500 photos of 12-24 MP, `/ingest/batch`)

| Cards | images/s | API CPU s/image | GPU util % | host busy % |
|---|---:|---:|---|---|
| 1 x A6000 | 25.03 (23.36-26.39) | 1.17-1.20 | 32-41 | 71-81 |
| A6000 + 3080 Ti | 26.54 (24.47-28.61) | 1.09-1.16 | 14-29 / 21-26 | 78-82 |

Photos are bound by the API and host (28 of 48 cores), not by the encoder: a second card adds 6
percent. More cards help frames, not photos; the next lever for photos is the API CPU per image.

#### Same crops, any card

`pe_gpu_parity.py capture` encoded the same 256 crops (resize, centre crop, uint8, as in ingest)
on a Triton that sees one card, once per card.

| Pair | crops | cosine min | max abs diff | bit-identical |
|---|---:|---:|---:|---:|
| RTX A6000 (GPU 0) vs RTX 3080 Ti (GPU 1) | 256 | 1.0000 | 0.0 | 256 |
| RTX A6000 (GPU 0) vs RTX A6000 (GPU 2) | 256 | 1.0000 | 0.0 | 256 |
| RTX 3080 Ti (GPU 1) vs RTX A6000 (GPU 2) | 256 | 1.0000 | 0.0 | 256 |
| RTX 3080 Ti, run 1 vs run 2 | 256 | 1.0000 | 0.0 | 256 |

Which card serves a request does not change the stored vector. TensorRT logs that a plan is "not
supported across different models of devices"; with one compute capability the three cards loaded the
one plan and agreed bit for bit.

#### Parity gate (`precision` profile, set A, A6000 + 3080 Ti, policy `all`)

| Comparison | P-box | P-item | P-pe-p01 | P-pew-p01 | P-knn | P-cluster-assign | P-cluster-purity | P-label-vote | unmatched conf. | gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ref-u8fp16-repeat` vs `ref-u8fp16` (noise floor) | 0.9939 | 1.0000 | 0.99995 | 0.99994 | 0.9955 | 0.9997 | 0.453 | 0.9990 | 0 | none |
| two-card run vs `ref-u8fp16` | 0.9934 | 1.0000 | 0.99991 | 0.99989 | 0.9945 | 0.9997 | 0.274 | 0.9980 | 0 | none |
| two-card run vs `ref-u8fp16-repeat` | 0.9944 | 1.0000 | 0.99991 | 0.99989 | 0.9939 | 0.9995 | -0.180 | 0.9970 | 0 | none |

The capture used the detector and the encoder only (`--probe-images 0`: MobileCLIP, SCRFD and OCR
are untouched by this change and did not fit next to them on the 12 GB card); the CLIP, face and OCR
gates are therefore skipped. 12,792 items matched 12,792.

#### Memory

One encoder instance holds 1,312 MiB on either card at batch 32 (recorded as
`GPU_MIB_PER_INSTANCE` 1536 in the model config, which `openprocessor gpu plan` and the installer
read). The primary card holds every model: 13.6 GB on the A6000. With 2.5 GB held by another project
the full model set does not load on the 12 GB card (SCRFD and the detector fail with out of memory),
the detector and encoder alone do (5.1 GB). So a 12 GB card is a good secondary and not a primary;
`gpu plan` refuses it as the lowest id, and the installer's existing 16 GB export floor does too.

#### Finding: CUDA numbers cards fastest first

Without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, a Triton container given `TRITON_GPU_ID=1,2` put every
`gpus: [ 0 ]` model on the RTX A6000 (host index 2) and the encoder second on the RTX 3080 Ti,
because CUDA's default order is fastest first. The compose file now sets PCI order, so the pinned
models follow the lowest listed host index on any mix of cards.

#### Not measured

- Whether the cross-socket copies of a card on the other NUMA node cost anything: no pinned
  per-socket run was made, and the pair above already includes GPU 2 on node 1 against GPU 1 on
  node 0 (scaling 1.87, 84 percent of the engine-only capacity).
- The 10,000-image pin, p99 request latency per layout, the CLIP, face and OCR parity probes of the
  two-card run, and a client-balanced multi-Triton run through the API (only the engine-only
  comparison above).

### Vector store evaluation: tuned OpenSearch against Milvus, Qdrant, pgvector and LanceDB (#221, #230)

Question: is OpenSearch k-NN the right engine for the curation embedding index, or is a second
store worth its cost? Protocol: [v060_vector_db_evaluation_plan.md](design/v060_vector_db_evaluation_plan.md).
Raw documents: `docs/benchmarks/vectordb/` (one JSON per size, arm and repetition, plus
`environment.json` with versions, digests and the deviations from the plan). Harness:
`scripts/bench/vectordb/`, compose file `docker/bench/vectordb/compose.yml`.

#### Setup

- **Data: real embeddings only.** The product's PE-Core-L14-336 image tower (FP16, GPU 1) over
  19,000 pinned public COCO 2017 images: one whole-frame vector plus one crop per ground-truth box
  (largest first, at most 12, shorter side at least 10 px), 6.5 vectors per image, 1024 dimensions,
  L2-normalized. 114,987 vectors are indexed; 1,300 held-out images supply 1,000 query vectors and
  7,299 vectors for the streaming phase. Rungs of 2,000, 10,000 and 100,000 vectors are nested
  prefixes. Every row below is `real`; no noise-expanded or random vectors were used. **1M was not
  run**: the real set is 115k vectors, and embedding about 0.9M COCO vectors needs a 20 GB image
  download and a GPU for hours, which this run did not do. Every 1M figure below is a linear
  extrapolation and is labelled as such.
- **Filters** as the product builds them: none; `project` (9.9 percent); project + class (2.6);
  review default (project, not validated, not test holdout, not dismissed; 8.3); project +
  `cluster_id` (0.06); a selectivity sweep at 50, 10, 1 and 0.1 percent. The filter fields other
  than the COCO class are synthetic and seeded (plan section 3); `cluster_id` is a 512-way k-means
  of the vectors. Exact cosine top-100 per query per filter is the ground truth.
- **Engines**, each alone in a fresh container (16 cores, 64 GB cgroup, loopback only, no GPU):
  OpenSearch 3.6.0 (8 GB heap), Milvus 2.6.25 standalone (embedded etcd, local storage), Qdrant
  1.19.2, pgvector 0.8.7 on PostgreSQL 17 (16 GB shared buffers), LanceDB 0.40.0 (embedded), plus an
  exact flat scan as the reference. HNSW arms use `m` 16 and `ef_construction` 512 unless named.
- **Operating point.** For every filter the cheapest grid setting (`ef`, `nprobe`) whose recall@10
  reaches 0.95 on 120 queries is used for the timed runs (300 queries, 3 repetitions on the
  QPS filters). The grid floor is 16, and most engines reach 0.95 there on this data: recall is
  not the binding constraint, per-query overhead is. Recall is re-measured on 300 queries, so a row
  can read slightly under 0.95 (sampling error of a few thousandths).
- **Noise.** The host is shared: load average 10 to 195 and 25 to 100 percent whole-host CPU during
  the runs. The headline arms have 3 whole-arm repetitions at 100,000 (cells show median and
  min-max); secondary arms and the 2,000 / 10,000 rungs have one. A difference is claimed only when
  the ranges do not overlap. Some runs overlapped by mistake and were re-run (see
  `environment.json`); the small rungs show shape, not precise latency.

#### Write throughput, storage and memory at 100,000 vectors (real)

"Load" is the client-side send (bodies encoded before the clock starts); "to searchable" includes
refresh, merge or index build. LanceDB appends are cheap and the index build is its cost, so its
load column is not comparable; use "vectors per second including index build". Disk is the data
directory growth (container volume) and the engine's own figure where it reports one; RAM is the
growth of the container's anonymous plus shared memory after load (OpenSearch heap is a fixed
8 GB and not included).

| Arm | load vec/s (8 clients) | to searchable s | vec/s incl. index build | disk KiB/vec (dir) | disk KiB/vec (engine) | RAM KiB/vec | runs |
|---|---|---|---|---|---|---|---|
| os_shipped | 2822 (2658-3470) | 117 (102-124) | 855 (809-984) | 8.20 (8.19-8.20) | 10.99 (8.19-11.00) | 6.99 (6.94-7.71) | 3 |
| os_shipped_merged | 3319 (3130-3647) | 386 (360-428) | 259 (233-278) | 8.20 (8.20-8.20) | 8.20 (8.20-8.20) | 7.72 (7.53-7.99) | 3 |
| os_efc128 | 3117 | 102 | 982 | 8.20 | 8.20 | 7.43 | 1 |
| os_m32 | 3373 | 407 | 246 | 8.32 | 8.32 | 8.37 | 1 |
| os_lucene | 2071 (1662-2259) | 434 (401-513) | 230 (195-249) | 4.10 (4.10-4.10) | 4.10 (4.10-12.14) | 2.87 (2.43-3.40) | 3 |
| os_fp16 | 2493 | 355 | 282 | 6.20 | 10.24 | 6.31 | 1 |
| os_disk8x | 3973 | 109 | 920 | 4.70 | 4.70 | 3.64 | 1 |
| os_disk16x | 4191 | 81 | 1242 | 4.45 | 4.45 | 3.18 | 1 |
| os_disk32x | 4283 | 78 | 1284 | 4.34 | 8.53 | 2.52 | 1 |
| milvus_hnsw | 5423 (4997-6323) | 107 (93-109) | 930 (916-1077) | 23.44 (21.21-23.44) | n/a | 6.32 (5.93-6.36) | 3 |
| milvus_ivf_sq8 | 5243 | 58 | 1721 | 16.26 | n/a | 10.74 | 1 |
| qdrant_hnsw | 500 (500-551) | 215 (215-238) | 465 (420-466) | 10.71 (9.95-12.32) | n/a | 0.76 (0.73-0.93) | 3 |
| qdrant_deferred | 456 | 307 | 325 | 11.01 | n/a | 0.64 | 1 |
| qdrant_int8 | 532 | 203 | 492 | 10.97 | n/a | 4.24 | 1 |
| pg_vector | 9661 (6065-10941) | 74 (74-98) | 1351 (1019-1352) | 22.57 (22.57-22.57) | 13.55 (13.55-13.55) | 13.56 (13.56-13.56) | 3 |
| pg_halfvec | 15798 | 53 | 1879 | 10.30 | 5.55 | 5.56 | 1 |
| lance_ivfpq | 305789 (302346-330029) | 79 (78-80) | 1269 (1253-1276) | n/a | 8.50 (8.50-8.50) | n/a | 3 |
| lance_hnsw_sq | 34852 | 62 | 1604 | n/a | 11.53 | n/a | 1 |
| ref_flat | 11836 | 9 | 11190 | n/a | 4.00 | n/a | 1 |

Footnotes: `os_shipped` is the product's index as built (no forced merge, 13 segments); the other
OpenSearch arms are force-merged to one segment as the storage document's settle rule says.
`os_lucene` is the Lucene HNSW engine (same `m` and `ef_construction`, cosine). The `on_disk` arms use
OpenSearch's quantized mode with default rescoring. `qdrant_deferred` builds the graph after the load
(the documented bulk path); `qdrant_hnsw` indexes while loading, which is why its load is slow.
pgvector builds the HNSW index after a `COPY`, and its RAM column is `shared_buffers` holding the
table and index; pgvector's data directory includes WAL. Milvus's directory holds binlogs, index
files and the write-ahead log, which is why it is 3 times OpenSearch (it is a deployment cost, not
the index alone); on the smaller rungs the fixed overhead of Milvus and Qdrant dominates.

#### Query latency at the recall target, review-default filter (8.3 percent), 100,000 vectors (real)

Single client, k = 10, concurrency 1 for latency and 8 / 32 client threads for QPS (the Python client
may bind the QPS columns above about 800). `recall@100` is measured at the cheapest setting that
reaches 0.95 at k = 100.

| Arm | setting | recall@10 | p50 ms | p95 ms | QPS c8 | QPS c32 | recall@100 |
|---|---|---|---|---|---|---|---|
| os_shipped | {"ef_search": 16} | 1.000 (1.000-1.000) | 8.4 (8.3-8.5) | 9.6 (9.5-9.9) | 336 (331-362) | 347 (344-369) | 1.000 (1.000-1.000) ({"ef_search": 16}) |
| os_shipped_merged | {"ef_search": 100} | 0.967 (0.967-0.971) | 6.6 (6.1-6.9) | 8.1 (8.0-10.5) | 423 (365-448) | 387 (372-408) | 0.939 (0.935-0.940) ({"ef_search": 512}) |
| os_efc128 | {"ef_search": 100} | 0.957 | 6.3 | 7.0 | 366 | 369 | 0.911 ({"ef_search": 512}) |
| os_m32 | {"ef_search": 64} | 0.951 | 6.2 | 7.5 | 398 | 355 | 0.957 ({"ef_search": 512}) |
| os_lucene | {"ef_search": 16} | 0.997 (0.997-0.998) | 6.9 (6.1-7.3) | 8.8 (8.8-11.1) | 403 (344-414) | 392 (360-395) | 0.999 (0.999-0.999) ({"ef_search": 16}) |
| os_fp16 | {"ef_search": 100} | 0.963 | 6.6 | 8.1 | 382 | 371 | 0.937 ({"ef_search": 512}) |
| os_disk8x | {"ef_search": 128} | 0.952 | 6.0 | 7.5 | 370 | 388 | 0.851 ({"ef_search": 512}) |
| os_disk16x | {"ef_search": 128} | 0.945 | 5.6 | 7.8 | 388 | 394 | 0.857 ({"ef_search": 512}) |
| os_disk32x | {"ef_search": 16} | 0.976 | 6.1 | 8.2 | 384 | 441 | 0.933 ({"ef_search": 512}) |
| milvus_hnsw | {"ef": 16} | 0.954 (0.948-0.973) | 8.0 (7.2-8.1) | 9.8 (9.7-10.3) | 741 (568-747) | 726 (580-800) | 0.964 (0.964-0.986) ({"ef": 16}) |
| milvus_ivf_sq8 | {"nprobe": 32} | 0.958 | 7.8 | 9.5 | 896 | 819 | 0.978 ({"nprobe": 128}) |
| qdrant_hnsw | {"hnsw_ef": 16} | 1.000 (0.995-1.000) | 6.4 (6.2-7.6) | 7.3 (7.2-8.4) | 293 (286-298) | 293 (290-311) | 1.000 (0.999-1.000) ({"hnsw_ef": 16}) |
| qdrant_int8 | {"hnsw_ef": 16} | 1.000 | 4.5 | 6.3 | 317 | 319 | 1.000 ({"hnsw_ef": 16}) |
| pg_vector | {"ef_search": 16} | 1.000 (1.000-1.000) | 48.0 (47.3-50.3) | 52.2 (49.2-56.7) | 131 (130-135) | 103 (100-105) | 1.000 (1.000-1.000) ({"ef_search": 16}) |
| pg_halfvec | {"ef_search": 16} | 0.988 | 4.9 | 9.5 | 903 | 747 | 1.000 ({"ef_search": 16}) |
| lance_ivfpq | {"nprobes": 10, "refine": 10} | 0.977 (0.967-0.990) | 29.5 (29.5-30.4) | 33.9 (31.9-34.5) | 228 (210-246) | 275 (244-287) | 0.990 (0.987-0.994) ({"nprobes": 10, "refine": 10}) |
| lance_hnsw_sq | {"nprobes": 10, "ef": 64} | 0.986 | 13.6 | 18.8 | 175 | 264 | 0.990 ({"nprobes": 10, "ef": 64}) |
| ref_flat | {} | 1.000 | 9.5 | 12.9 | 271 | 319 | 1.000 ({}) |

#### Filtered search: recall@10 per filter (median and min-max of the runs), 100,000 vectors (real)

| filter | selectivity | os_shipped | os_shipped_merged | os_lucene | milvus_hnsw | qdrant_hnsw | pg_vector | lance_ivfpq |
|---|---|---|---|---|---|---|---|---|
| none | 100.00% | 0.987 (0.982-0.988) | 0.973 (0.972-0.976) | 0.978 (0.977-0.982) | 0.975 (0.954-0.979) | 0.991 (0.989-0.992) | 0.946 (0.946-0.981) | 0.956 (0.955-0.962) |
| project | 9.90% | 1.000 (1.000-1.000) | 0.977 (0.975-0.980) | 0.993 (0.993-0.994) | 0.955 (0.952-0.975) | 0.999 (0.998-1.000) | 1.000 (1.000-1.000) | 0.974 (0.970-0.985) |
| project_class | 2.56% | 1.000 (1.000-1.000) | 0.912 (0.908-0.917) | 1.000 (0.999-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.969 (0.958-0.980) |
| review_default | 8.30% | 1.000 (1.000-1.000) | 0.967 (0.967-0.971) | 0.997 (0.997-0.998) | 0.954 (0.948-0.973) | 1.000 (0.995-1.000) | 1.000 (1.000-1.000) | 0.977 (0.967-0.990) |
| project_cluster | 0.06% | 0.995 (0.995-0.995) | 0.995 (0.995-0.996) | 0.997 (0.995-0.997) | 0.998 (0.995-0.998) | 0.996 (0.995-0.997) | 0.995 (0.995-0.995) | 0.995 (0.995-0.995) |
| sel50 | 49.85% | 0.982 (0.981-0.988) | 0.959 (0.959-0.966) | 0.969 (0.962-0.969) | 0.958 (0.953-0.971) | 0.980 (0.979-0.994) | 0.963 (0.962-0.965) | 0.968 (0.964-0.973) |
| sel10 | 9.96% | 1.000 (1.000-1.000) | 0.975 (0.974-0.978) | 0.995 (0.995-0.995) | 0.947 (0.940-0.962) | 0.999 (0.998-1.000) | 0.989 (0.988-0.990) | 0.975 (0.972-0.987) |
| sel1 | 0.99% | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.997 (0.997-0.997) | 0.984 (0.982-0.991) |
| sel01 | 0.10% | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.999 (0.998-0.999) | 1.000 (1.000-1.000) |

Selectivity of `project_cluster` came out at 0.06 percent (the busiest cluster of project 0), below
the plan's 0.2.

p95 latency (ms) of the same runs:

| filter | selectivity | os_shipped | os_shipped_merged | os_lucene | milvus_hnsw | qdrant_hnsw | pg_vector | lance_ivfpq |
|---|---|---|---|---|---|---|---|---|
| none | 100.00% | 8.5 (7.9-8.7) | 7.0 (6.7-7.4) | 6.7 (6.1-6.7) | 5.4 (5.1-5.6) | 5.6 (5.2-6.4) | 1.9 (1.5-2.3) | 33.0 (31.5-39.5) |
| project | 9.90% | 10.2 (9.3-14.8) | 7.8 (7.6-16.3) | 8.9 (8.8-9.1) | 6.4 (5.9-8.4) | 7.2 (7.0-7.3) | 62.5 (59.5-98.3) | 34.9 (34.1-37.1) |
| project_class | 2.56% | 7.3 (7.3-7.8) | 12.8 (12.4-23.1) | 11.1 (11.1-12.0) | 8.2 (8.0-8.2) | 5.6 (5.0-7.4) | 20.3 (18.1-34.4) | 29.0 (28.3-29.5) |
| review_default | 8.30% | 9.6 (9.5-9.9) | 8.1 (8.0-10.5) | 8.8 (8.8-11.1) | 9.8 (9.7-10.3) | 7.3 (7.2-8.4) | 52.2 (49.2-56.7) | 33.9 (31.9-34.5) |
| project_cluster | 0.06% | 6.3 (6.0-6.9) | 6.1 (5.3-7.4) | 7.6 (7.3-7.8) | 5.6 (5.3-5.8) | 4.0 (4.0-4.6) | 2.0 (1.6-2.2) | 19.1 (16.1-20.4) |
| sel50 | 49.85% | 10.1 (10.0-10.5) | 7.1 (6.7-7.5) | 6.6 (6.5-7.0) | 11.6 (11.0-12.1) | 6.0 (5.3-6.0) | 2.4 (2.1-2.4) | 72.4 (71.8-72.5) |
| sel10 | 9.96% | 10.3 (9.3-10.4) | 7.4 (6.7-7.8) | 8.2 (8.0-8.4) | 7.2 (7.2-7.3) | 7.2 (6.9-7.7) | 5.4 (4.7-5.4) | 51.0 (34.9-79.3) |
| sel1 | 0.99% | 7.2 (6.4-7.3) | 7.2 (6.8-8.9) | 8.1 (7.6-8.1) | 6.4 (6.3-6.7) | 4.5 (4.0-5.6) | 28.9 (28.4-30.0) | 28.0 (26.4-54.1) |
| sel01 | 0.10% | 6.6 (6.3-6.8) | 6.4 (5.7-11.5) | 7.6 (7.4-7.7) | 5.6 (5.3-6.4) | 4.1 (4.1-6.5) | 144.9 (143.5-151.3) | 22.8 (21.5-49.3) |

#### Metadata rewrite, streaming insert and delete, 100,000 vectors (real)

`cluster_id` rewritten for 12.8k and for all 100k vectors (the cluster pass write-back), review-default
queries running throughout; then 200 vectors per second inserted for 20 s next to 4 query clients
(p95 ratio is mixed over idle); then the 10 percent project deleted and the store settled. LanceDB's
settle row reflects `optimize` with version cleanup. Only the headline arms ran these phases.

| Arm | rewrite 12.8k s | rewrite 100k s | recall@10 during | p95 ms during | stream vec/s achieved | p95 mixed / idle | delete 10% s | MiB before delete | MiB after settle |
|---|---|---|---|---|---|---|---|---|---|
| lance_ivfpq | 1.2 (1.2-1.2) | 3.7 (3.4-4.0) | 0.980 (0.976-0.985) | 89.2 (80.4-98.1) | 201 (201-201) | 1.03 (1.01-1.05) | 0.0 (0.0-0.1) | 1291 (1291-1291) | 397 (397-397) |
| milvus_hnsw | 2.1 (2.1-2.2) | 12.2 (11.4-13.0) | 0.976 (0.973-0.978) | 17.6 (17.4-17.9) | 199 (197-201) | 0.85 (0.67-1.03) | 1.7 (1.6-1.8) | 4772 (4707-4837) | 5027 (4947-5107) |
| os_lucene | 6.5 | 43.0 | 1.000 | 21.6 | 200 | 2.10 | 1.9 | 2257 | 1097 |
| os_shipped_merged | 7.8 (6.9-8.6) | 43.4 (35.1-51.6) | 0.982 (0.981-0.984) | 21.5 (16.5-26.4) | 176 (153-199) | 2.01 (1.90-2.11) | 3.6 (2.9-4.3) | 2689 (2418-2960) | 760 (756-764) |
| os_shipped | 6.0 (5.9-6.1) | 44.4 (38.4-50.4) | 1.000 (1.000-1.000) | 16.8 (13.8-19.9) | 196 (192-200) | 1.96 (1.85-2.06) | 2.2 (2.1-2.4) | 2266 (2181-2351) | 1670 (1600-1740) |
| pg_vector | 25.8 (25.0-26.7) | 202.4 (199.6-205.1) | 0.991 (0.991-0.991) | 31.3 (12.4-50.2) | 137 (123-151) | 1.82 (1.79-1.85) | 0.2 (0.2-0.2) | 2425 (2422-2428) | 2497 (2486-2508) |
| qdrant_hnsw | 62.7 (54.8-70.6) | 148.9 (136.2-161.6) | 0.998 (0.996-1.000) | 13.1 (12.2-14.0) | 200 (200-200) | 2.39 (2.28-2.51) | 1.8 (1.1-2.5) | 1206 (1205-1208) | 1243 (1208-1278) |

#### Scale rungs (review-default filter: recall@10 / p95 / QPS at 8 clients / disk per vector)

2,000 and 10,000 vectors have one run per arm; the disk column of Milvus and Qdrant at those sizes
is fixed overhead, not per-vector cost. All rows real.

| arm | 2,000 vectors | 10,000 vectors | 100,000 vectors |
|---|---|---|---|
| os_shipped | 1.000 / 5.9 ms / 365 QPS / 8.2 KiB | 1.000 / 6.5 ms / 386 QPS / 5.9 KiB | 1.000 (1.000-1.000) / 9.6 (9.5-9.9) ms / 336 (331-362) QPS / 11.0 (8.2-11.0) KiB |
| os_shipped_merged | 1.000 / 5.5 ms / 421 QPS / 8.2 KiB | 1.000 / 7.3 ms / 384 QPS / 8.2 KiB | 0.967 (0.967-0.971) / 8.1 (8.0-10.5) ms / 423 (365-448) QPS / 8.2 (8.2-8.2) KiB |
| os_lucene | 1.000 / 7.5 ms / 391 QPS / 4.1 KiB | 1.000 / 5.9 ms / 512 QPS / 4.1 KiB | 0.997 (0.997-0.998) / 8.8 (8.8-11.1) ms / 403 (344-414) QPS / 4.1 (4.1-12.1) KiB |
| milvus_hnsw | 0.978 / 6.3 ms / 904 QPS / 86.4 KiB | 0.969 / 7.0 ms / 923 QPS / 29.8 KiB | 0.954 (0.948-0.973) / 9.8 (9.7-10.3) ms / 741 (568-747) QPS / 24.9 (22.6-24.9) KiB |
| qdrant_hnsw | 1.000 / 3.8 ms / 378 QPS / 480.3 KiB | 1.000 / 6.7 ms / 292 QPS / 99.3 KiB | 1.000 (0.995-1.000) / 7.3 (7.2-8.4) ms / 293 (286-298) QPS / 10.7 (10.0-12.3) KiB |
| pg_vector | 1.000 / 1.8 ms / 4205 QPS / 13.7 KiB | 1.000 / 6.4 ms / 1333 QPS / 13.6 KiB | 1.000 (1.000-1.000) / 52.2 (49.2-56.7) ms / 131 (130-135) QPS / 13.6 (13.6-13.6) KiB |
| lance_ivfpq | 1.000 / 17.0 ms / 124 QPS / 10.0 KiB | 0.992 / 16.4 ms / 304 QPS / 8.8 KiB | 0.977 (0.967-0.990) / 33.9 (31.9-34.5) ms / 228 (210-246) QPS / 8.5 (8.5-8.5) KiB |
| ref_flat | 1.000 / 0.1 ms / 6516 QPS / 4.0 KiB | 1.000 / 0.7 ms / 4670 QPS / 4.0 KiB | 1.000 / 12.9 ms / 271 QPS / 4.0 KiB |


Single-client load at 10,000 vectors (vectors per second, one run): OpenSearch merged 634,
OpenSearch Lucene 595, Milvus 4,154, Qdrant 712, pgvector 2,832, LanceDB 57,978 (append only). Eight
clients: 2,124, 2,226, 5,080, 513, 6,448 and 168,286.

#### What the numbers say

1. **Recall is not where OpenSearch loses, with one exception.** On the product's filters the
   as-built index (`os_shipped`, 13 segments) holds recall@10 of 0.98 to 1.0 at `ef_search` 16 and
   above, better than any other engine's lowest setting. After a forced merge to one segment the faiss
   graph becomes the weak point: recall@10 at the review default needs `ef_search` 100 (0.617 at 16,
   0.795 at 32, 0.923 at 64, 0.966 at 100, 0.996 at 256), and at project + class (2.6 percent) it stays
   at 0.91 even at `ef_search` 512, where Milvus, Qdrant, pgvector and the Lucene engine read 1.0. The
   Lucene engine holds 0.969 to 1.0 on every filter at `ef_search` 16 to 32.
2. **Latency at the recall target is a tie.** On the review default the best OpenSearch rows (merged
   faiss 8.1 ms, `on_disk` 16x 7.8 ms, `ef_construction` 128 7.0 ms, Lucene 8.8 ms) and the best other
   rows (Qdrant 7.3 ms, Milvus 9.8 ms, pgvector `halfvec` 9.5 ms) overlap within the noise of this
   host. pgvector is the fastest unfiltered (1.9 ms) and on the 0.06 percent cluster filter (2.0 ms),
   and the slowest on the review default (52 ms for `vector`, 9.5 ms for `halfvec`) and on the 0.1
   percent selectivity filter (145 ms): its planner switches between the HNSW and B-tree paths.
   LanceDB IVF_PQ is 3 to 4 times slower (34 ms). Milvus's one clear lead is concurrent throughput:
   741 QPS at 8 clients against 423 for merged OpenSearch (1.7 times, ranges do not overlap).
3. **Writes while serving.** At 200 vectors per second, OpenSearch's query p95 doubles (ratio 2.0),
   Qdrant's grows 2.4 times, pgvector's 1.8 times; Milvus (0.85) and LanceDB (1.03) stay flat.
   Refreshes (which build graphs) take 87 s of wall time during a 100k load.
4. **Metadata rewrite.** In place in OpenSearch the `cluster_id` rewrite of 100k vectors takes 35 to
   52 s with recall@10 during the rewrite at 0.98 to 1.0, inside the plan's 60 s limit. It rewrites the
   vector with the document: disk grows to 2.4 to 2.7 GB (3.2 times the settled 0.8 GB) until a merge.
   Milvus does the same rewrite in 12 s, LanceDB in 4 s, Qdrant in 149 s and pgvector in 202 s.
5. **Storage and memory.** The faiss index costs 8.2 KiB per vector on disk and 4.1 KiB of native
   (off-heap) graph memory, as the storage document says. Lucene HNSW costs 4.1 KiB on disk and no
   native graph memory (the graph is memory mapped; heap used 0.7 to 2.8 GB against 4.7 to 6.1 GB).
   `on_disk` 8x / 16x: 4.7 / 4.5 KiB and 0.65 / 0.40 KiB native graph, but recall@10 needs
   `ef_search` 128 and recall@100 falls to 0.85. `on_disk` 32x ignores `ef_search` (0.975 at every
   value, no graph memory): it behaves as a quantized scan with rescoring. fp16 halves the native graph
   (215 MB against 415 MB per 100k) but disk falls only 24 percent. No other engine is below the
   OpenSearch Lucene or `on_disk` figures: Qdrant int8 11.0 KiB disk (it keeps the raw vectors),
   pgvector `halfvec` 5.6 KiB, Milvus 16 to 23 KiB (directory), LanceDB IVF_PQ 8.5 KiB.

   **Native-memory formula, checked against the stored data (#56).** For a faiss HNSW vector field
   the k-NN plugin's estimate is `1.1 x (4 x d + 8 x m)` bytes per vector (float32 vectors plus
   `m` neighbours of 8 bytes; the 1.1 is its overhead factor). At `d` 1,024 and `m` 16 that is
   4,646 bytes, 4.54 KiB; without the factor, 4,224 bytes, 4.125 KiB. The measured value is the
   node's `knn_graph_memory_kb` after the 100,000-vector load
   (`docs/benchmarks/vectordb/100000/os_shipped_r1.json`, field `footprint.engine_stats`): 414,845
   KB, which is 4.15 KiB per vector. The raw formula without the overhead factor predicts it to
   within 1 percent and the 1.1 factor over-estimates by 9 percent, so size the native memory of a
   faiss index at 4.5 KiB per vector (4.5 GiB per million) and expect 4.1 KiB. The same division
   gives the quantized and Lucene rows: fp16 (`os_fp16_r1.json`) 214,876 KB, 2.15 KiB per vector (the
   formula with 2-byte components is 2,176 bytes, 2.125 KiB); `on_disk` 8x (`os_disk8x_r1.json`)
   64,876 KB, 0.63 KiB; `on_disk` 16x (`os_disk16x_r1.json`) 39,876 KB, 0.39 KiB; Lucene
   (`os_lucene_r3.json`) 0 KB, because the Lucene graph is memory mapped and is paged by the
   operating system instead of counted in native memory. The table in "Write throughput, storage and
   memory" has the container-level RAM growth, which is larger (it includes heap, page cache and
   buffers).
6. **Build cost.** `ef_construction` 128 cuts the OpenSearch merge from 264 s to 67 s (searchable
   after 102 s against 386 s) for a recall@10 loss of 0.01 at `ef_search` 100 (0.957 against 0.967 on
   300 queries, inside the sampling error). The Lucene engine merges slower (387 to 485 s) and loads
   slower (2.1k against 3.3k vectors per second).

##### Cross-check against public statements

- OpenSearch k-NN issue 3606 (filtered Faiss recall falls at sparse-to-mid selectivity while Lucene
  stays near 1.0): **agree in direction**. Merged faiss reads 0.912 at 2.6 percent and 0.967 at 8 to
  10 percent; Lucene reads 1.0 and 0.993 to 0.997. The floor here (0.91) is higher than the issue's
  0.78 at 5 percent (nested queries, a different data set). The as-built unmerged index does not show
  it. Faiss p95 is not lower than Lucene's here (8.1 against 8.8 ms on the review default, 12.8
  against 11.1 on project + class, within noise).
- Concurrent segment search (37 to 15 ms in the 3.0 benchmark) was not tested; it was off.
- Lucene-on-Faiss quantization costing up to 4.5 percent recall: **partly disagree**. At the same
  `ef_search` 100 the 8x and 16x modes read 0.933 and 0.932 against 0.966 (3 points) on recall@10,
  and 0.85 against 0.94 on recall@100 at `ef_search` 512 (9 points); raising `ef_search` to 128 recovers 0.95.
- FP16 saving about 48 percent memory: **agree for the native graph** (415 to 215 MB per 100k),
  disagree if it is read as total memory or disk (RAM growth 7.7 to 6.3 KiB per vector, disk 8.2 to 6.2).
- Rankings among Milvus, Qdrant and pgvector differ across public sources; only the recall-gated rows
  above are used here.

#### Where OpenSearch's limits are, and what 1M would do (extrapolation, not measured)

Nothing in the 2,000 to 100,000 rungs is a hard limit: one shard, one node, 8 GB heap, p95 5 to 10 ms.
Linear extrapolation of the measured per-vector figures to the 7.4 million vectors of a 1M-image
project (one vector field; the items index has two):

| Quantity | faiss as shipped | Lucene engine | faiss `on_disk` 16x |
|---|---|---|---|
| Disk, one field | 61 GB (above the usual 50 GB shard guidance) | 30 GB | 33 GB |
| Native (off-heap) graph memory | 31 GB, counted against the k-NN circuit breaker | 0 (page cache instead) | 3 GB |
| Forced merge of the whole field (from 100k timings; optimistic, build cost grows faster than linearly) | 5.5 h (`ef_construction` 512), 1.4 h (128) | 8 to 10 h | 1.1 h |

The binding items at that scale are shard size, native memory, merge time and, today, query p95
doubling under sustained inserts, not recall or single-query latency. All are properties of the
engine choice (Lucene or `on_disk`) and of sharding, which stay inside OpenSearch. A real
1M-vector set is the next scale point; it needs the full COCO train download (about 4.5 hours at the
rate seen here) and GPU time for embedding.

#### Decision against the plan (section 8)

| Criterion (all must hold at D1M) | Result at 100,000 real vectors | Met |
|---|---|---|
| p95 at recall@10 >= 0.95 at least 2x better than best tuned OpenSearch on review default and cluster filters, or storage plus RAM per vector at least 40 percent lower than the best quantized OpenSearch | review default: best other 7.3 ms against best OpenSearch 7.0 to 8.1 ms (no gain); cluster filter: Qdrant 4.0 against 6.1 ms (1.5x), pgvector 2.0 ms but 52 ms on the review default; storage plus RAM: Lucene 4.1 + 2.9 = 7.0 KiB, `on_disk` 16x 4.5 + 3.2 = 7.6; best other pgvector `halfvec` 11 KiB, Qdrant int8 15 KiB | no |
| `cluster_id` rewrite of 100k in under 60 s with recall >= 0.9 | OpenSearch in place 35 to 52 s with recall 0.98 to 1.0; Milvus 12 s; Qdrant 149 s and pgvector 202 s fail | met by OpenSearch itself |
| Consistency design implementable with a reconciler and isolation guard test | not assessed: the first criterion failed | n/a |
| Operational weight acceptable | not assessed | n/a |

D1M was not run, so the criteria could not be met formally; at 100k there is no sign they would be.

#### End states compared

- **A. OpenSearch stays the single store, tuned (supported by the numbers).** No second service, OCC
  and the isolation guard unchanged, one backup. Costs to manage: query p95 doubles under 200 vectors
  per second of inserts, a 5 to 9 minute settle after deleting a project's items, and 2.4 to 3.2
  times the disk until a merge after a full `cluster_id` rewrite.
- **B. A vector store (Milvus is the strongest candidate) plus OpenSearch or a relational store for
  everything else.** Gains measured: 1.7 times the concurrent QPS, flat query latency under inserts, a
  12 s rewrite. Costs, none of which this benchmark removes: (1) every filter field the store needs
  (class, validated, holdout, dismissed, cluster, project) is duplicated, and each human validation,
  auto-promote and cluster pass writes twice; the lag is a correctness window (a validated item can
  still be returned) closed only by over-fetching and an `mget` re-check; (2) an outbox and a
  reconciler for item-without-vector, vector-without-item and stale-vector states after a crash; (3)
  OCC and the lock rule stay in the metadata store, but the vector store needs its own
  project-isolation guard (the transport guard covers only OpenSearch) and every delete, combine and
  undo flow has to cover two stores; (4) clustering reads every vector from OpenSearch by
  point-in-time slices today, so either vectors stay in OpenSearch (no storage win; Milvus adds 16 to
  23 KiB per vector on this host) or the cluster pass must read the second store; (5) backup and
  restore must be consistent across two stores; (6) migration is a backfill from the existing
  indexes, per project, plus a dual-write period. Host cost measured for Milvus: 73 MB idle, 6.3 KiB
  per vector resident, one container with embedded etcd (the stock compose uses three). pgvector as
  the relational store would also take the metadata, which rewrites the OCC, aggregation, sort and
  point-in-time code and is out of scope.

#### Recommendation

**Stay on OpenSearch as the single store; do not add Milvus now.** The numbers do not support a split
at the sizes measured: latency at the recall target ties, storage and memory are not better anywhere
else, and the consistency costs above are certain. Because no split is recommended, no split plan
document was written. Follow-ups, each its own work package with the product-stack parity gate (kNN
overlap@10 >= 0.95 on the review default, which was not run here because no product stack was
brought up):

1. **Lucene engine for new project indexes** (`_knn_field` in
   `src/clients/curation_opensearch/base.py`): half the disk, no native graph memory, filtered
   recall 0.97 to 1.0 on every filter, and identical `_score` for cosine (checked on 2,000 vectors: faiss
   and Lucene returned the same scores to four digits, so `min_score` and the UI chip are unaffected).
   Costs: 50 percent slower merge, 36 percent slower load, and existing indexes need a reindex. Not
   shipped here: it changes stored mappings and needs the owner's decision and the parity gate.
2. **`ef_construction` 128** (`OP_HNSW_EF_CONSTRUCTION`): merge 4 times faster, recall within sampling
   error; not yet tested on the Lucene engine.
3. **Do not force-merge** project indexes: the as-built segments gave the best recall at no extra cost.
4. **Leave `ef_search` at the default**: it was not a lever (p50 moved by under 1 ms from 16 to 512).
5. Re-run at 1M (and the cluster pass at 100k, #218) before revisiting a split. Revisit sooner if the
   sustained ingest rate pushes query p95 past the product's budget, or concurrent search load exceeds
   about 400 QPS per node. If a split is ever built, the smallest proof is a shadow Milvus collection
   for one project filled from the scroll of its items index, compared on recall and p95, with no
   change to the existing indexes.

No product code was changed by this evaluation.

#### Not measured

- 1M vectors (real or otherwise); GPU index arms (cuVS, GPU CAGRA, Milvus GPU, FAISS GPU); two
  shards and the `byte` data type in OpenSearch; concurrent segment search; Milvus `partition_key`,
  IVF_FLAT and cluster-filtered partitioning; Qdrant on-disk vectors; pgvector partial indexes and
  iterative-scan tuning; LanceDB GPU build; bulk size, refresh-off and writer-count sweeps (probe 5
  covers them).
- Whole-arm repetitions beyond one for the secondary arms and the small rungs; 120 and 300 queries
  per point instead of 1,000; the in-set ("more like this") queries are recorded in the JSON
  (`in_set@10`) but not tabulated.
- Per-engine RSS under steady query load (only idle and after-load figures) and operational
  procedures (backup, upgrade) beyond what is stated.
- The product-stack parity gate for any shipped setting.

### PE image encoder engine pass (#227)

Measured question: can the FP16 PE engine (384 GFLOP per embedding, 190 images/s engine only on the
RTX A6000, about 46 percent of its 154.8 TFLOP/s dense FP16 peak) go faster at identical output?
**Result: one change ships and everything else is a no-op.** The rotary embedding of the exported
graph runs on interleaved channel pairs, and TensorRT lowers that to slice, transpose and concat
kernels that took 23 percent of the engine time. Permuting the query and key channels of every
head (even channels first, then the odd ones) turns the pairs into two contiguous halves, which is the
same function (cosine 1.0000 in FP32 on ONNX Runtime) and a 17 to 25 percent faster engine, engine
only and through Triton, on the same card. No builder flag, optimization profile, batching setting or
instance count beat the shipped build. FP16 stays the only precision. The change is
`export/pe_rope_layout.py`, run by the end of `export/export_pe_image_encoder.py`. Raw numbers:
`docs/benchmarks/pe_engine_pass.json`.

#### Method

- **Engines.** The FP16-baked, uint8-input PE graph that the installer builds (TensorRT 11.1.0.106,
  `trtexec`, profile min 1 / opt 8 / max 32 at 336 x 336, 8 GB workspace), one plan per variant, all
  built the same way on the same GPU. Variants that change the ONNX (`rope`, `rope2`) start from the
  same FP32 ONNX as the baseline.
- **Engine only** is `trtexec --loadEngine` at a fixed batch (TensorRT 11 uses CUDA graphs and skips
  host transfers by default), 1 s warm-up, 15 s timed. **Through Triton** is `perf_analyzer` over HTTP
  against an isolated Triton (own container, loopback port, GPU 0), 6 s windows, the production
  dynamic-batching configuration unless the row says otherwise, random uint8 input. The two are never
  compared with each other. Every cell is the median of 3 repetitions, interleaved across variants,
  with the range beside it.
- **Machine state.** One RTX A6000 capped at 300 W: during the runs the SM clock averaged 1,430 to
  1,540 MHz (lower at larger batches) and the power draw 285 to 299 W, temperature up to 78 C. The
  GPU also holds other projects' idle Triton servers (30 GB resident, no compute). A run was kept only if
  `nvidia-smi pmon` showed no process other than the benchmark with SM activity before and during the
  run; runs that failed that check were repeated (the count is in the JSON). Timed commands ran under
  the shared bench lock.
- **Builds are sensitive to load.** A first set of variant plans built while another process loaded the
  GPU ran at 78 to 116 images/s against 184 for the same flags built on a quiet GPU (the builder times
  tactics on the shared card). That set was discarded; the plans below were built with the GPU idle and
  the build was repeated if any foreign activity showed. Two builds of the baseline (with and without a
  timing-cache file) agree within 0.1 percent.
- **Layer dump** is `trtexec --dumpLayerInfo --dumpProfile` on a build with detailed profiling
  verbosity, batch 16, separate profile run.

#### Layer dump: what the engine does

Both engines already fuse attention (`_gemm_mha_v2`, 24 blocks, 7.4 ms of 83.5 ms per batch of 16),
fuse the bias, residual add and GELU into the GEMM epilogues (`FcAdd`, `Fc`), fold the Cast nodes of the
FP16 bake into generated kernels (no standalone cast layer in the plan) and run LayerNorm as one generated
kernel per norm. The GEMMs run at 99 to 118 TFLOP/s each. The unfused cost was the rotary embedding.

| Per batch of 16, `--dumpProfile` | Interleaved pairs (shipped) | Half-split halves |
|---|---:|---:|
| Output projection and MLP down GEMMs (`FcAdd`) | 19.6 ms | 19.4 ms |
| MLP up GEMM with GELU (`Fc`) | 18.7 ms | 18.5 ms |
| QKV GEMM | 12.9 ms | 12.6 ms |
| Fused attention | 7.4 ms | 7.3 ms |
| Rotary slice, transpose, negate, concat | 19.2 ms (23 %) | 2.8 ms (4 %) |
| LayerNorm | 3.6 ms | 3.6 ms |
| Total per batch | 83.5 ms (191 images/s) | 66.2 ms (242 images/s) |

#### Engine only (`trtexec`, images/s, median [range] of 3)

The speed-up is the same in the two independent sets (the baseline reads 184 and 182 to 184 in them).

| Plan | Batch 8 | Batch 16 | Batch 32 |
|---|---:|---:|---:|
| Baseline (opt 8) | 184.3 [183.5-185.7] | 182.4 [181.8-183.9] | 183.7 [183.4-183.8] |
| Baseline, second build | 184.2 [183.6-186.2] | 182.1 [182.0-185.2] | 183.8 [183.3-183.9] |
| `--builderOptimizationLevel=5` (build 6 to 7 min instead of 2) | 184.2 [184.2-185.8] | 183.3 [182.3-185.1] | 184.3 [183.9-184.8] |
| `--maxAuxStreams=0` | 184.2 [183.8-184.3] | 182.6 [181.7-182.8] | 183.5 [183.4-184.2] |
| Profile opt 16 | 173.0 [172.8-173.6] | 194.2 [193.0-195.1] | 183.3 [183.3-184.3] |
| Profile opt 32 | 178.2 [178.0-178.2] | 186.0 [184.3-186.0] | 202.4 [201.5-202.5] |
| **Half-split rotary layout** | **214.9 [214.7-217.0]** | **226.0 [225.8-228.3]** | **228.8 [228.8-229.2]** |
| Half-split, profile opt 16 | 212.6 [212.4-213.0] | 228.0 [227.8-230.1] | 228.3 [228.0-228.6] |
| Half-split, optimization level 5 | 215.8 [215.7-216.5] | 228.1 [226.8-228.3] | 229.5 [229.0-230.1] |
| Half-split, tiling optimization level 3 | 215.6 [215.5-216.2] | 227.9 [226.2-229.3] | 229.2 [229.0-229.9] |
| Half-split, attention scales folded into the QKV weights | 215.7 [215.4-216.4] | 226.2 [225.8-229.4] | 229.1 [228.4-229.5] |

An optimization profile whose `opt` shape matches the batch is 6 to 10 percent faster at that batch
and 3 to 6 percent slower at the others; with ingest batches that vary from 1 to 32 the profile
does not pay and stays at `opt` 8. At batch 32 the half-split engine runs 229 images/s = 88 TFLOP/s =
57 percent of peak (the baseline: 184 = 71 TFLOP/s = 46 percent).

#### Through Triton (`perf_analyzer`, images/s, median [range] of 3)

| Batch x concurrency | Baseline | Half-split | Ratio | p50 latency, baseline / half-split |
|---|---:|---:|---:|---:|
| 8 x 4 | 181.0 [180.5-181.5] | 217.5 [217.1-218.2] | 1.20 | 176 / 146 ms |
| 8 x 8 | 181.1 [181.1-181.2] | 224.4 [224.2-224.4] | 1.24 | 352 / 284 ms |
| 16 x 4 | 181.1 [180.4-181.1] | 224.6 [224.5-225.3] | 1.24 | 352 / 283 ms |
| 16 x 8 | 181.1 [179.6-181.1] | 225.4 [225.3-232.0] | 1.24 | 705 / 566 ms |
| 32 x 4 | 180.5 [180.5-182.0] | 226.3 [224.8-227.8] | 1.25 | 701 / 558 ms |
| 32 x 8 | 180.3 [180.2-180.3] | 226.3 [226.2-226.3] | 1.26 | 1,404 / 1,128 ms |
| 1 x 32 (single images, batched by Triton) | 180.4 [180.2-180.4] | 217.0 [216.7-217.3] | 1.20 | 176 / 147 ms |

Serving settings, all on the same engine (images/s; 16 x 4, 16 x 8, 1 x 32):

| Setting | Half-split | Baseline |
|---|---|---|
| Shipped: 1 instance, preferred 4, 8, 16, 32, 15 ms queue delay | 224.6, 225.4, 217.0 | 181.1, 181.1, 180.4 |
| 2 instances on the GPU | 228.1, 228.2, 223.7 (+1 to +3 %) | 181.1, 178.9, 183.3 (-1 to +2 %) |
| Queue delay 0 | 223.1, 225.3, 213.4 | not run |
| Queue delay 5 ms | 224.5, 224.5, 217.8 | not run |
| No preferred batch sizes | 224.6, 226.0, 217.5 | not run |
| CUDA graphs for batches 1, 2, 4, 8, 12, 16, 20, 24, 32 | 221.0, 220.9, 215.8 (-2 %) | 178.1, 177.4, 174.2 (-2 to -3 %) |
| Profile opt 16 plan | 225.3, 226.0, 216.2 | not run |

The GPU is the bottleneck at one instance, so the batching knobs have nothing to win; a second instance
buys at most 3 percent for twice the activation memory, and CUDA graphs cost 2 percent.

#### Parity (candidate engine against the shipped FP16 engine, pinned 2,000 images)

The candidate plan is the half-split ONNX baked, uint8-wrapped and built through the installer path,
served in an isolated stack (project `pepass`, GPU 0, own ports, the reference models directory with only the
PE plan replaced) and captured with `scripts/bench/parity.py capture`; the references are
`ref-u8fp16` and `ref-u8fp16-repeat` of the current tree, judged with `compare --profile precision`.
Summary: `docs/benchmarks/parity/pe_rope_layout_vs_u8fp16/`.

| Gate (rule) | Against `ref-u8fp16` | Against `ref-u8fp16-repeat` | Reference against its repeat (noise) |
|---|---:|---:|---:|
| P-pe crop cosine p01 (>= 0.995) | 0.99976 | 0.99979 | 0.99995 |
| P-pe crop cosine median (>= 0.999) | 0.99996 | 0.99996 | 1.00000 |
| P-pew frame cosine p01 (>= 0.995) | 0.99945 | 0.99945 | 0.99994 |
| P-pew frame cosine median (>= 0.999) | 0.99994 | 0.99994 | 1.00000 |
| P-knn overlap@10 (>= 0.95) | 0.9826 | 0.9835 | 0.9955 |
| P-dup near-duplicate agreement (>= 0.995) | 1.0000 | 1.0000 | 1.0000 |
| P-cluster-assign (>= 0.98) | 0.9987 | 0.9984 | 0.9997 |
| P-cluster-purity drop in points (<= 0.5) | 0.44 | -0.02 | 0.45 |
| P-label-vote (>= 0.97) | 0.994 | 0.993 | 0.999 |
| P-knn-precision drop in points (<= 0.5) | -0.12 | -0.11 | -0.01 |
| P-box, P-item, P-label, P-clip, P-face, P-ocr | PASS | PASS | PASS |

All gates pass (`P-vlm` skipped: no VLM in the stack). The frame vectors, which share their input bits
with the reference (no detector in between), move by a minimum cosine of 0.9970 and a first percentile
of 0.99945; the crop minimum of 0.909 is the same as the reference's own repeat run (0.909: one crop
whose rectangle moved a pixel). The drift is FP16 rounding order, larger than the repeat noise
(first percentile 0.9998 against 0.99995) and well inside the gates. Issue #227 states a stricter bar
of p01 >= 0.9999 against `ref-fp16`; the engine does not meet that literal figure (0.99945 on frames),
and the repository gate for a precision-level change is the one above.

#### Decision

| Rule | Observed | Action |
|---|---|---|
| >= 10 percent PE images/s at batch 16 through Triton | 1.24x (181.1 to 224.6) | Ship the half-split layout |
| Reaches >= 55 percent of peak | 57 percent at batch 32, 56 percent at batch 16 | Closes the question of fused attention: it was fused all along |
| Builder flag, profile, batching, instance or graph variants | within 2 percent of the baseline except at the `opt` batch | Not adopted, nothing added |

`models/pe_image_encoder/config.pbtxt`, `build_pe_trt.sh` and the installer build step are unchanged.
Deployments pick the faster engine up by rebuilding it (`./openprocessor models install --only pe` re-exports the
ONNX and rebuilds the plan); a plan built before this change keeps working and keeps its speed.

#### What was not measured, and caveats

- Not run: the RTX 3080 Ti (GPU 1 belonged to another run), a maximum batch of 64, a second instance of
  the half-split engine with the production API in front, the full ingest protocol (the ingest phase
  of the two parity captures took 77 and 82 s against 97.6 s of the reference capture, but the host load
  differed; not a measurement), repeated builds of the half-split plan (the baseline's two builds agree
  within 0.1 percent), CUDA shared memory and pinned-memory settings (the shared-memory package covers them).
- No attention plugin or custom kernel was tried: TensorRT already selects its fused MHA kernel, the
  remaining non-GEMM time is LayerNorm at 4 percent and the fused attention at 11 percent. The
  `--tacticSources` flag only lists convolution tactics in this TensorRT release, and the engine has one
  convolution. LayerNorm and GELU are already fused (above). Precision mixing was not touched; INT8 stays
  rejected (see the INT8 section).
- Timing was taken at 1,430 to 1,540 MHz on a power-capped card shared with other projects; the
  ratios between rows are the finding, the absolute images/s move by a few percent with clocks.
- The rewrite is a fixed pattern of the PE-Core export (node names of the `perception_models` graph,
  24 blocks of 16 heads x 64, 577 tokens). It checks those assumptions and leaves any other graph
  as exported, with a warning, so a changed upstream export costs speed, not correctness.

To repeat: export (`python export/export_pe_image_encoder.py` runs the rewrite), bake and build with
`export/build_pe_trt.sh`, then `trtexec --loadEngine=model.plan --shapes=images:16x3x336x336
--warmUp=1000 --duration=15` for the engine and `perf_analyzer -m pe_image_encoder -b 16
--concurrency-range=4 --shape images:3,336,336` for Triton. A baseline plan for the comparison is the same
build from an ONNX exported before this change (or `python export/pe_rope_layout.py` skipped).

### WP-2.3: cross-image PE batching (#216)

The question: ingest sends the crops of one image (6.4 on set A) and, in a second request, its whole
frame, so every PE request carries a handful of rows while the engine is fastest at 32. Does packing
the rows of several images into one engine batch raise images/s? Medians with the minimum and maximum
of 3 repetitions in parentheses. Only repetitions with 0.0 percent foreign GPU load are counted.

**Result.** Not for ingest: it is already at the engine's rate, and a client-side coalescer built for
this (a per-process queue that packs the rows of concurrent callers into requests of up to 32 rows,
5 ms maximum wait) changed nothing beyond noise on set A and on set B, so it was removed again and
no batching code ships for ingest. The embed pass was the real gap: it embedded one image at a time,
so Triton saw batches of 3.7 rows. It now works on 16 images at once, in chunks of 64 instead of 20,
and runs 2.2 times faster (6.33 to 14.05 images/s on 500 images). Triton's dynamic batcher already
does the packing across concurrent requests; the fix is to have concurrent requests.

| Item | Value |
|---|---|
| GPU | RTX A6000, GPU 2, 0.0 percent foreign SM share in every counted repetition; another project held GPU 0 at 100 percent and was not touched. SM clock 1,416-1,444 MHz (c16), 1,449-1,482 MHz (c4) |
| Host | 48 logical cores shared with other projects; 1-minute load 5-9 on the c16 and c4 repetitions, 12-20 at c32, 14-25 on set B; host 12-15 percent busy (set A, c4 and c16) |
| Stack | project `xbatch`: Triton, API (the published 0.5.0 image with this checkout's `src/` mounted), OpenSearch 8 GB heap, 32 API workers, loopback ports; the uint8 FP16 PE plan of `ref-u8fp16` (sha256 `a4feb7df...`), `pe_image_encoder` dynamic batcher `preferred_batch_size: [4, 8, 16, 32]`, `max_queue_delay_microseconds: 15000` (unchanged) |
| Load | set A: 2,000 COCO images, `POST /ingest/batch`, 32 images per request, policy `all`, 1 warm-up and 3 counted repetitions (`ingest_sweep_suite.py --phases env,sweep --sweep-points 32x16,32x4 --modes batch --skip-cluster --skip-settle`); set B: 500 photos, 8 per request, 16 clients; the embed bench is `scripts/bench/embed_missing_bench.py` (500 images ingested, then 3 rewrites of every crop and frame vector) |

#### Measurement first: PE batch sizes Triton sees today (set A, unchanged ingest code)

| Clients x images per request | images/s | PE mean batch | PE requests per image | PE executions | GPU 2 busy % | request p50 s (p95) |
|---|---:|---:|---:|---:|---:|---:|
| 16 x 32 | 23.51 (23.36-23.65) | 27.7 (27.5-28.0) | 2.07 | 529-538 | 95-98 | 21.7 (24.0-26.0) |
| 32 x 32 | 23.42 (23.39-23.59) | 27.4 (27.1-27.9) | 2.07 | 530-546 | 96-98 | 40.5 (50.6-60.1) |
| 4 x 32 | 21.43 (21.07-21.62) | 20.3 (19.5-21.2) | 2.07 | 698-757 | 86-91 | 5.9 (7.1-7.7) |

Batch-size histogram of Triton's PE executions, first repetition, 16 clients (batch size: executions;
535 executions in all): 32: 155, 31: 57, 30: 53, 29: 44, 28: 33, 27: 35, 26: 29, 25: 18, 24: 20,
23: 15, 22: 10, 21: 20, 20: 8, 19: 9, 18: 6, 17: 5, 16: 3, 12-15: 8, 1-8: 12. Triton already
packs 27.7 of 32 rows on average. The sizes below 32 come from request granularity (a 7-row request does
not fit in the 3 rows left), not from a lack of company.

The per-crop engine cost does not depend on the batch size in this range: the execute time divided by
the mean batch is 5.5 ms per row at a mean batch of 21 (c4) and 5.5 ms at 27.7 (c16), the same 5.5 ms
(181 rows/s) as `perf_analyzer` at batch 32 (180.9 rows/s, multi-GPU section). With 7.3 rows per
image, 23.5 images/s is 172 rows/s, 95 percent of the engine; the packing of the last 5 percent
cannot be worth more than that, and it is hidden by the GPU being 95-98 percent busy. Larger batches
help only below about 16 rows, which ingest at 4 or more clients never sends.

#### The coalescer, and why it was removed

A `PEBatcher` in front of the Triton call (per process, per event loop; flush at 32 rows or 5 ms after
the first caller; two batches in flight; a failed batch retried one caller at a time; each caller
gets back its own rows) was built and measured as the "client-side micro-batching window" option.
Same stack, GPU 2, 0.0 percent foreign load:

| Case | Batcher off | Batcher on (5 ms) |
|---|---:|---:|
| Set A, 16 x 32, images/s | 23.51 (23.36-23.65) | 23.64 (23.28-23.91) |
| Set A, 4 x 32, images/s | 22.78 (21.81-22.82) | 22.63 (21.77-23.24) |
| Set A, 16 x 32, PE requests per image | 2.07 | 0.35-0.44 |
| Set A, 16 x 32, PE mean batch at Triton | 27.7 | 25.9 (25.2-26.7) |
| Set A, 16 x 32, request p95 s | 25.4 (24.0-26.0) | 27.4 (24.8-31.4) |
| Set B, 8 x 16, images/s | 28.63 (26.99-29.49) | 28.03 (25.16-29.54) |
| Embed pass, images/s (16 images at once) | 13.75 (13.63-13.84) | 13.17 (13.10-13.19) |

The batcher cut PE requests per image by 80 percent and gained nothing: throughput is the engine's,
the mean batch at Triton fell (a process sends fewer, larger requests, which Triton packs less
tightly: two 20-row requests do not share a batch), and the request p95 moved up. The 4-client
"off" row is a later series than the first 4-client row above (21.43); the 6 percent between the
two is the drift between series on a shared host, not a code difference (ingest code is identical),
which is why the on/off comparison was taken in the same series. The deterministic-output rule held (below), but a
negative result ships no code: the batcher module and its wait setting were deleted. Acceptance (PE requests per image below 0.3 from 2.07, throughput not lower) is
therefore not met for ingest, and not wanted: requests per image is not what limits ingest.

#### The embed pass (what shipped)

The `embed` reprocess scope (embed-missing, "embed all") embedded one image at a time: a one-row
whole-frame request, then that image's crop request, in chunks of 20 images that drained at each end.
`reembed_items` now runs up to 16 images at once (`_IMAGES_IN_FLIGHT`), so their requests overlap
and Triton packs them, and a chunk is 64 images (`reprocess_images.CHUNK`), so a chunk holds several
windows. `OP_MAX_INGEST_CONCURRENCY` is for ingest and is unrelated. The encoder client and the
Triton configuration are unchanged.

| Embed pass, 500 images, crop and frame vectors | before | after |
|---|---:|---:|
| images/s | 6.33 (6.33-6.34) | 14.05 (13.93-14.10) |
| PE mean batch | 3.7 | 13.0 (12.9-13.4) |
| PE requests per image | 1.98 | 1.98 |
| PE executions at batch 1 | 556 of 990 | 14 of about 280 |
| GPU 2 busy % | 25-29 | 57-60 |
| API cores / CPU-s per image | not recorded | 0.65 / 0.047 |

The settings that led here, all measured with the coalescer on, 500 images: 8 images at once and
20-image chunks 10.8 images/s, 24 at once 10.7 (the chunk end is the limit), chunks of 64 with 24 at
once 12.8, with 16 at once 13.2, with 32 at once 12.7. The pass is still not at the engine's
24 images/s: the GPU is 58 percent busy with the API using 0.65 of a core, so the next limit is probably on the
host side of the one process that runs the job. That was not diagnosed here.

Memory: 16 images are decoded at once in one API process, about 15 MB for 640 x 480 frames and up to
1.2 GB for 24 MP photos (3 bytes per pixel). The embed pass on photos was not run.

#### Same vector whatever the batch mates

The engine does not give the same bits for a crop at batch 1, at batch 32 or in a mixed batch (different
kernels per batch size); it agrees to FP16 rounding. 192 crops (2 random boxes on each of 96 COCO
images, the ingest resize), straight to Triton on GPU 2:

| Pair | cosine min | max abs difference | bit-identical |
|---|---:|---:|---:|
| each crop alone vs batches of 32 | 0.999995 | 9.4e-4 | 0 of 192 |
| each crop alone vs random batches of 2-31 | 0.999995 | 9.5e-4 | 0 of 192 |
| batches of 32 vs random batches of 2-31 | 0.999995 | 9.5e-4 | 0 of 192 |
| batches of 32, run twice | 1.000000 | 0.0 | 192 of 192 |

Through the shipped embed pass (96 COCO images, 746 crops, 96 frames, two rewrites in a row, whose
batches differ): crop cosine min 0.99989 and p01 0.99995; frame cosine min 0.99988. Against the vectors
ingest wrote the embed pass differs more for some crops (min 0.9855, p01 0.9995; frames 0.99988): the pass
cuts the crop from the stored normalized box on a re-decoded image, ingest from the detector's pixel box.
That difference is in the crop rectangle, not in batching (the same minimum appeared with the coalescer on and
off), and the pass was not compared against the old build, so whether it is older than this change
is not established.

#### Parity (`precision` profile, set A, GPU 2, policy `all`, capture of the final code)

| Comparison | P-box | P-item | P-pe-p01 | P-pew-p01 | P-knn | P-cluster-assign | P-cluster-purity | P-label-vote | unmatched conf. | gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ref-u8fp16-repeat` vs `ref-u8fp16` (noise floor) | 0.9939 | 1.0000 | 0.99995 | 0.99994 | 0.9955 | 0.9997 | 0.453 | 0.9990 | 0 | none |
| final vs `ref-u8fp16` | 0.9924 | 0.9995 | 0.99993 | 0.99990 | 0.9937 | 0.9996 | 0.164 | 0.9970 | 0 | none |
| final vs `ref-u8fp16-repeat` | 0.9924 | 0.9995 | 0.99993 | 0.99991 | 0.9942 | 0.9993 | -0.289 | 0.9990 | 0 | none |

The ingest path is the code of `origin/main`, so this capture is a regression check of the stack, not
a test of the change; the embed pass has no reference capture, its check is the table above. The
capture with the coalescer on passed the same gates (P-pe-p01 0.99995, P-pew-p01 0.99990, P-knn 0.9948,
P-cluster-purity 0.219). The VLM gate is skipped (no VLM in the stack).

#### Caveats and not measured

- One host and one card, 3 repetitions, other projects running on the host's CPUs and on GPU 0; the c4
  and c16 series ran at different times (drift above). Only 0.0 percent foreign GPU repetitions are
  counted; host load was not controlled.
- Set B was measured only for the on/off comparison (the host was loaded, load average 14-25; the
  throughput reading is noisy: 25.2-29.5 images/s).
- Not measured: Triton queue delay and preferred sizes (a knob sweep belongs to #226; the shipped 15 ms
  and `[4, 8, 16, 32]` were left alone because the engine, not the batch size, bounds ingest); two
  GPUs (one queue per Triton feeds both, unchanged by this); the `region` part of the embed pass; the
  embed pass on photos; what holds the embed pass at 58 percent GPU; a before/after of the embed pass
  API CPU; ingest with a coalescer cap of 16 rows to keep Triton's packing granularity.
- Before-and-after of the embed pass compares two builds on one stack, run back to back on GPU 2 (before
  first, after later); the before series ran with the coalescer absent.

### Cluster pass plumbing: fetch and write-back (#218)

The question: where does "ingest complete" to "clusters available" go, and how much of it is moving
vectors out of and assignments back into OpenSearch rather than clustering? The clustering methods
(IVF, AHC and its refinement, HDBSCAN) were not touched; a golden test pins what each stores.
Medians of 3 repetitions, range in the table of stage times in the bench output; CPU only.

**Result.** The write-back was the binding stage: a guarded painless update re-indexes the whole
document with its vector, so one 2,000-document request at a time gave about 550 documents/s. Eight
requests of 1,000 reach about 2,300 documents/s on this loaded host (more writers add nothing). The fetch
is bound by the node rendering 25 MB of floats per 2,000-document page (2.4 s) and then by the single
client thread decoding it; decoding is now 2.5 times faster, and the fetch is flat at about 19 s for
100k whatever the slices (8 to 32) and page size (1,000 to 2,000), so the node is the floor.

| Seconds (median) | 2k before | 2k after | 10k before | 10k after | 100k before | 100k after |
|---|---|---|---|---|---|---|
| IVF fetch | 0.79 | 0.55 | 3.31 | 2.46 | 30.0 | 19.3 |
| IVF fit (unchanged) | 2.38 | 2.33 | 7.55 | 7.96 | 93.3 | 86.9 |
| IVF write-back | 3.26 | 0.65 | 18.5 | 3.9 | 175.3 | 31.5 |
| **IVF total** | 6.71 | 3.87 | 28.3 | 14.4 | 310.6 | 138.0 |
| AHC total (fetch + fit + write) | 5.99 | 4.83 | 47.5 | 24.4 | not run | not run |
| HDBSCAN total | 11.6 | 9.25 | 201.6 | 189.1 | not run | not run |
| assign-only (incremental), IVF | 5.0 | 3.3 | 26.3 | 10.6 | 292.8 | 82.6 |

| Write-back alone, 100k documents | 2000 x 1 | 2000 x 4 | 1000 x 4 | 1000 x 8 | 2000 x 8 | 1000 x 16 | 500 x 16 |
|---|---|---|---|---|---|---|---|
| documents/s | 550 | 1,600 | 1,560 | 2,340 | 2,330 | 2,100 | 2,050 |

The first cell is chunk size by writers. Default: 1,000 documents, 8 writers; a pool too small to fill
them is split down to 250 documents per request. AHC and HDBSCAN were not run at 100k (see caveats).

**New bottleneck, found here.** The `cluster_geometry` stage the pipeline runs inside the same progress
stage read every clustered vector twice, one scroll per cluster, and wrote every item in 20-document
requests one after another: 43 s at 2k, 96 s at 10k, 442 s at 100k after IVF (the 100k figure is on the
vector set of the table above). It is now one point-in-time sliced read of the clustered items with the
centroids, distances and nearest clusters computed from that matrix (rows in index order, so every sum is
the one the per-cluster read made), items that already store the same geometry skipped, and the rest
written 8 requests at a time through the same writer. The stored geometry is identical to the old
algorithm's (`tests/curation/test_cluster_geometry_golden.py`).

Follow-up measured on the 12,792 real crop vectors of a parity capture (the 114,987-vector set used above
was no longer available, so 100k was not re-measured), IVF, median of 3, seconds, same node and host:

| Seconds | 2k main | 2k first PR commit | 2k now | 10k main | 10k first PR commit | 10k now |
|---|---|---|---|---|---|---|
| IVF fetch | 0.72 | 0.55 | 0.50 | 3.67 | 2.34 | 2.50 |
| IVF fit (untouched) | 2.28 | 2.83 | 2.37 | 9.99 | 8.87 | 7.17 |
| IVF write-back | 3.08 | 0.81 | 0.56 | 18.0 | 3.84 | 3.64 |
| geometry stage | 60.2 | 40.8 | 3.1 | 142.9 | 94.3 | 11.4 |
| **cluster pass + geometry** | 66.6 | 45.4 | 6.8 | 176.4 | 110.6 | 24.9 |
| assign-only (incremental) | 5.8 | 5.0 | 1.6 | 27.7 | 9.7 | 5.4 |

Assign-only now reads through 8 point-in-time slices (one scroll is one cursor) and skips a document that
already stores exactly what the update writes; a second pass on 10k found 7,181 of 10,000 already exact
(3.9 s against 5.4 s; the rest differ in the last bit of the distance because a page's composition changes
when its documents are rewritten, so the match is exact or the document is written). The new bottleneck
at 10k is the IVF fit (7.2 of 13.5 s of the pass, a method), then the geometry stage (11.4 s).

Options measured and not adopted (10k, one 2,000-document page): `fields` instead of `_source` 2.4 s
against 2.65 s (inside the noise); `docvalue_fields` is refused by the node for the vector field and
`stored_fields` returns nothing; `refresh_interval: -1` during the write-back 4.0-4.5 s against 3.7-4.5 s
with the default (not faster, and it adds 1-2 s to the closing refresh).

#### Caveats and not measured

- Host: 48 logical cores shared with other projects, 1-minute load 13-48 during the series; one
  single-node OpenSearch 3.6, 8 GB heap, loopback, `clusplumb` project. Before and after ran back to back.
- AHC and HDBSCAN are not run-to-run deterministic on real vectors (two runs of the unmodified code gave
  6 different `cluster_id` of 2,000), so real-index hashes cannot show equality for them; the golden test
  on a fixed set (deterministic) does, and IVF's stored hash is identical before and after at 2k and 10k.
- Not measured: AHC and HDBSCAN at 100k (AHC at 100k did not finish its merge loop in 25 minutes; HDBSCAN
  fit 180 s at 10k), GPU paths, the follow-up at 100k, concurrent search load during the write-back.
- Tool: `scripts/bench/cluster_plumbing_bench.py` (stage split, `--fetch-profile`, `--write-grid`,
  `--fetch-grid`, `--assign-only`, `--geometry`).

#### Cluster pass at a chosen size: `cluster_bench.py` (#218)

`scripts/bench/cluster_bench.py` times the production `cluster_residuals` at each size of a ladder
(`--sizes`), per method (`--methods`, default `ivf`), `--reps` times, in a throw-away project
(real mapping, deleted through the project lifecycle at the end) loaded with the first N rows of a dump of real
public COCO embeddings. It records method, size, wall seconds, residual and cluster counts and the
host load, and refuses a pass that did not cluster the whole pool (status not `success`, or fewer
residuals than rows loaded): a cluster-refresh daemon on the same node assigns the items of a new
project between the reset and the fetch, so such passes are discarded and repeated (`--attempts`).
The wall time is "fit and write-back as `cluster_residuals` does them": it excludes the geometry
stage (`cluster_plumbing_bench.py --geometry`) and the load. It calls the methods and changes none.

Run on the live development stack, 12,792 real crop vectors of the pinned COCO set A (the parity
capture `ref-u8fp16`), IVF and AHC, median (min-max) of 3, seconds:

| Method | 2,000 vectors | 10,000 vectors | Clusters (2,000 / 10,000) |
|---|---|---|---|
| IVF | 8.5 (6.7-10.9) | 72.2 (71.7-112.2) | 333 / 512 |
| AHC | 30.4 (21.3-42.5) | 231.8 (165.4-356.9) | 627 / 1,769 |

**These are not comparable with the tables above.** The host was saturated by other projects (1-minute
load 44 to 138 on 48 logical cores, against 13 to 48 during the plumbing series) and the cluster-refresh
daemon of the stack was working on the same node; the IVF pass at 10,000 vectors, 14.4 s for the whole
pass in the plumbing table, took 72 s here. Treat the run as a check that the harness works and as an
upper bound, not as a result: no number in this guide changes because of it. The 100,000-vector run
needs a dump of at least that size (the parity capture has 12,792 rows) and a quiet host, and is open (#218):

```bash
.venv/bin/python scripts/bench/cluster_bench.py --opensearch-url http://127.0.0.1:<opensearch port> \
    --data <vectors dump>.npz --source vecdb --sizes 2000,10000,100000 --methods ivf --reps 3 \
    --out <results dir>
```

`--source vecdb` reads `vectors` (float32 rows) and `class_name` arrays from the npz; AHC did not
finish at 100,000 and HDBSCAN was not run there (see above), so only `ivf` belongs on that rung.

### GPU decode measurements (#217)

Question: does decoding JPEGs on the GPU, and building each model's input there, speed up ingest?
The full design, the dataflow map and the implementation plan are in
[design/gpu_decode_plan.md](design/gpu_decode_plan.md); this section keeps the numbers. One RTX
A6000 (GPU 2; SM86 has no hardware JPEG decoder), the published 0.5.0 images, the `ref-u8fp16`
engines, an isolated compose project, only this work's processes on GPU 2 during every timed run.
Tools: `scripts/bench/gpu_decode_bench.py`, `scripts/bench/gpu_decode_parity.py`.

**Result.** On 12-24 MP photos the CPU decode is the ceiling: the two decodes ingest runs per photo
cost 0.50 CPU-s and 46 processes doing nothing else reach 26.3 photos/s, the measured photo ingest
rate. GPU decode cuts the host cost of a photo 3.7x (not to zero: there is no GPU-only Huffman
decoder on this card) and, with letterbox, whole frame and blur on the GPU too, the API CPU per photo
falls 65 percent through the real stack. Images/s did not rise in that run because the float32 GPU
kernels used made the GPU the limit; the plan sets a kernel budget and estimates about 1.9x on one
A6000 and 3x on two once it is met (cheaper kernels, 8 instead of 11 ms of GPU per photo, gave
25.5 against 26.8 photos/s: still GPU-bound). GPU decode changes the decoded pixels
slightly and the detector turns that into different boxes on 7 to 13 percent of images: the parity
gates against today's references fail, AP and kNN precision against ground truth do not move. COCO
frames gain nothing (they are bound by the PE engine) and stay on the CPU.

CPU decoders (single thread, ms per image, median; Pillow 12.3 and OpenCV 4.10 of the API image):

| Images (median MP) | Pillow + EXIF | Pillow at 1/8 | OpenCV | OpenCV at 1/8 | both decodes ingest runs | both, 46 processes: images/s |
|---|---:|---:|---:|---:|---:|---:|
| COCO set A (0.27) | 5.3 | 2.2 | 3.7 | 2.1 | 7.4 | 979 |
| Open Images 1,024 px (0.78) | 10.1 | 3.4 | 5.8 | 2.9 | 13.0 | 851 |
| Open Images originals (5.2) | 68.1 | 15.5 | 37.1 | 15.6 | 83.7 | 124 |
| photos, set B (17.7) | 369 | 130 | 283 | 133 | 502 | 26.3 |

GPU decoders (batch 32, 8 decoder threads): images/s and host CPU ms per image:

| Images | nvImageCodec | DALI | torchvision nvJPEG |
|---|---|---|---|
| COCO set A | 2,346, 2.8 ms | 1,941, 3.1 ms | 379, 2.6 ms (serial) |
| Open Images 1,024 px | 1,282, 4.9 ms | 1,176, 5.1 ms | 215, 4.6 ms |
| Open Images originals | 193, 21.6 ms | 154, 23.5 ms | 22, 45 ms |
| photos, set B | 35.4, 134 ms | 31.8, 132 ms | 4.5, 221 ms |

One nvImageCodec decoder saturates at about 35 photos/s whatever its thread count; its `GPU_ONLY`
backend returns no image on this card. Decoding beside the PE engine at saturation: a DALI GPU decode
(decode, letterbox, whole-frame resize) takes 4.5-5.8 percent of PE (about 2 ms of PE time per photo);
the float32 torch GPU decode with the blur takes 22 percent (about 15 ms).

Through the stack (`/ingest/batch`, 3 repetitions):

| Arm | Set B images/s | API CPU-s per photo | host busy % | GPU busy % | Set A images/s | API CPU-s per frame |
|---|---:|---:|---:|---:|---:|---:|
| CPU path, 32 workers | 24.32 (21.72-26.26) | 1.06-1.13 | 76-79 | 36-43 | 23.51 (23.39-23.54) | 0.069 |
| CPU path, 16 workers | 26.78 (24.59-28.10) | 1.07-1.15 | 80-82 | 36-41 | | |
| GPU decode in the API process, float32 kernels, 16 workers | 23.39 (23.00-24.38) | 0.36-0.41 | 28-33 | 70-80 | 20.23 (20.00-20.70) | 0.164 |
| same, half-precision resize input and convolution blur | 25.48 (24.45-26.48) | 0.35-0.36 | 34-38 | 79-83 | not run | |

With 32 workers the API-side GPU decode ran the 48 GB card out of memory (each worker held 2-5 GB).

Parity of that run through the stack against `ref-u8fp16` (profile `precision`): P-box 0.869, P-item
0.942, P-pe p01 0.974, P-pew p01 0.991, kNN overlap 0.911, label vote 0.961 fail; near-duplicate
agreement 0.9997, cluster assignment 0.988, purity drop 0.34 points, kNN precision drop 0.00 points
and class labels pass. Offline on the same engines, GPU decode alone accounts for most of the box
change (P-box 0.883 with CPU resizes, 0.869 with GPU resizes; the GPU letterbox passes the tensor
gate on every image), and COCO AP is 0.4376 on the CPU path against 0.4378 and 0.4377 on the two GPU
arms. Decoder coverage: nvImageCodec applies all eight EXIF orientations and, with
`fancy_upsampling=1`, matches Pillow within 0.01 grey levels on average; torchvision's GPU decode and
DALI's default ignore EXIF orientation, and every GPU decoder returns an image for truncated or
corrupted files that Pillow rejects (the plan routes those to the CPU).

Not measured: the Triton-resident GPU decode the plan recommends, two cards and the 12 GB card with GPU
decode, kernels that meet the plan's budget, the region stage and VLM on GPU-encoded crop JPEGs.


### Stage 2 tuning: detector engine and the concurrency sweep (#226, #259)

Question: after the uint8 inputs, the half-split PE engine and the two-card encoder, is there a
detector engine variant or a Triton, client or API setting that raises images/s? **Result: no engine
variant and no Triton or API setting beats the shipped one by 3 percent or more; the one default that
changes is the bulk-ingest concurrency (8 to 16).** One RTX A6000 now ingests 28.8 to 29.4 images/s on
set A (23.4 before the half-split engine), with the encoder 95 to 97 percent busy: the card is the limit.
Medians with the range of 3 repetitions in brackets. Only repetitions with 0.0 percent foreign GPU
share count. Raw JSON stays in the scratch directory of the run.

| Item | Value |
|---|---|
| Cards | one RTX A6000 (GPU 0) for the sweep; two A6000 (GPUs 0 and 2, 0.0 percent foreign utilisation in every counted repetition) for the confirmation runs |
| Stack | project `stage2`, the checkout's `src/` over the published image, 32 API workers, OpenSearch 8 GB heap; uint8 FP16 detector plan and the half-split PE plan (`models/` as in the previous section) |
| Load | set A (2,000 COCO images), `POST /ingest/batch`, policy `all`, 100-image warm-up, 3 repetitions per point |
| Host | 48 logical cores shared with other projects; 1-minute load 7-37, host 13-47 percent busy; SM clock 1,480 MHz mean |
| Design | one factor at a time around the shipped point (16 images x 16 clients), plus a 15-point client screen on 1,000 images with 2 repetitions (it ran while the host was loaded and only the direction is used); a full factorial of every factor was not run, see the end |

#### Detector layer cost (`trtexec --dumpLayerInfo --dumpProfile`, detailed profiling, batch 16)

Total 10.1 ms per batch of 16 (1,583 images/s engine only); the profiled run sums to 10.8 ms.

| Part | ms | share |
|---|---:|---:|
| Conv + SiLU (backbone, neck, head), 85 fused layers | 6.58 | 61 % |
| Generated fused kernels (box decode, Tanh/Mul/Add, softmax) | 1.19 | 11 % |
| Concat, split, pool | 0.93 | 9 % |
| Other (shuffles, constants) | 0.73 | 7 % |
| Upsample (Resize) and its casts | 0.66 | 6 % |
| Input and output reformat copies | 0.36 | 3 % |
| End2end head glue (transpose, slice, cast, concat) | 0.25 | 2 % |
| `EfficientNMS_TRT` plugin | 0.09 | 1 % |

The compiled NMS and the end2end head are 3 percent of the engine and the engine is about 2 percent of
the GPU time of an ingest (the encoder is the rest), so nothing in the head is worth a variant by
itself. The casts that the FP32 NMS input needs cost 0.1 ms.

#### Detector engine variants (images/s, `trtexec --loadEngine`, 5 s, median [range] of 3, GPU 0)

| Plan | Batch 1 | Batch 8 | Batch 16 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|---:|
| Shipped plan (opt 32, max 64, level 3) | 640 [638-642] | 1,464 [1,452-1,464] | 1,595 [1,585-1,600] | 1,685 [1,674-1,692] | 1,712 [1,702-1,720] |
| Rebuilt, same flags (two builds) | 611-634 | 1,450-1,460 | 1,592-1,594 | 1,682-1,691 | 1,712-1,722 |
| Profile opt 16 | 504 | 1,412 | 1,611 | 1,647 | 1,659 |
| Profile opt 8 | 600 | 1,485 | 1,514 | 1,579 | 1,594 |
| Max batch 32 | 640 | 1,466 | 1,596 | 1,689 | n/a |
| Builder optimization level 5 | 639 | 1,441 | 1,564 | 1,649 | 1,670 |
| NMS input in FP16 (plugin not blocked in the FP16 bake) | 638 | 1,463 | 1,597 | 1,684 | 1,713 |

The build-to-build spread (about 1 percent) is as large as every difference except the profile `opt`
shapes, which win 1 percent at their own batch and lose 4 to 7 percent elsewhere. FP16 for the NMS input
changes nothing, so it was not taken to a parity capture. CUDA graphs on the detector were not
measured separately: the encoder measurement (previous section) was 2 percent slower with graphs and the
detector is 2 percent of the GPU. Nothing shipped.

#### Client concurrency x images per request (ingest images/s, set A, one card)

| Point | images/s | request p50 s | encoder busy |
|---|---:|---:|---:|
| 16 x 4 | 21.97 [21.6-22.4] | 2.9 | 0.74 |
| 16 x 8 | 27.56 [26.9-27.7] | 4.9 | 0.92 |
| 8 x 16 | 28.58 [28.5-28.9] | 4.5 | 0.95 |
| **16 x 16** | **29.07 [28.3-29.2]** | 9.2 | 0.96 |
| 32 x 16 | 28.45 [28.3-29.0] | 17.6 | 0.95 |
| 16 x 32 | 28.19 [28.1-28.5] | 17.3 | 0.94 |
| 16 x 64 | 26.86 [26.7-27.9] | 32.3 | 0.89 |

The plateau starts at 16 clients; 4 clients lose 24 percent, 8 lose 5 percent, 64 lose 8 percent. Images
per request (8, 16, 32) is inside the noise. The bulk-ingest default concurrency was 8 and is now 16
(maximum 32 unchanged).

#### Triton arms (16 x 16, set A, one card)

| Arm | images/s | vs shipped |
|---|---:|---:|
| Shipped | 28.81 [28.6-28.8] | |
| Encoder 2 instances | 28.83 [28.4-29.5] | +0.1 % |
| Detector 1 instance | 28.67 [28.7-29.1] | -0.5 % |
| Detector 4 instances | 28.78 [28.7-28.9] | -0.1 % |
| Encoder queue delay 2 ms (15 ms) | 29.02 [28.9-29.2] | +0.7 % |
| Encoder without `preferred_batch_size` | 29.01 [28.7-29.1] | +0.7 % |
| Detector queue delay 5 ms (25 ms) | 29.37 [29.1-29.5] | +1.9 % |
| Detector without `preferred_batch_size` | 29.07 [28.4-29.3] | +0.9 % |

#### API workers (16 x 16, set A, one card)

| Workers | images/s | API CPU s/image |
|---|---:|---:|
| 16 | 28.58 [27.4-29.4] | 0.059 |
| 24 | 29.14 [28.6-29.7] | 0.063 |
| 32 (shipped) | 28.92 [27.7-29.1] | 0.064 |
| 48 | 29.35 [29.2-29.7] | 0.072 |

#### Recommended defaults per installer profile

| Profile | Instances, delays, preferred sizes, workers | Client |
|---|---|---|
| minimal, standard, full | unchanged: no setting moved images/s by 3 percent or more on a GPU-bound card; `full` (48 GB+) is the profile measured | 16 batches of 16 images in flight (the bulk-ingest default) |

The `minimal` and `standard` profiles (6 to 24 GB cards) were not measured; their instance counts are set by
memory, not speed, and a GPU-bound card ignores them.

#### Embed pass (#259): what holds it at 15 images/s

`reprocess` scope `embed` (rewrite of every crop and frame vector) runs at 14.4 to 15.9 images/s on 500 and
800 images against 29 for ingest on the same card. It runs as one asyncio task in one API worker (0.7
cores). py-spy on that worker for 30 s at 50 Hz: the event-loop thread was non-idle in 810 of 1,500
samples (54 percent), and 205 more samples were in the pool threads. The loop thread does the work
itself because both preprocess calls declare under 4 MiB (`OFFLOAD_MIN_BYTES`) and run in place: crop
resize 21 percent of all samples, whole-frame decode and resize 15 percent, tensor packing 6 percent, gRPC
serialization 9 percent, bulk-update JSON 3 percent. The encoder is 57 to 60 percent busy, so the pass is
latency-bound on one loop thread and the 64-image chunk boundary (read, process, write in sequence), not
on the engine. The fix (0.7.0) is output-preserving: the embed pass runs every crop, frame and
JPEG step on the CPU pool whatever its size (`offload_always` in `src/utils/cpu_pool.py`; ingest keeps the
4 MiB in-place rule it was measured with), and the next 64-image chunk is read while the current one is
processed and written. The acceptance bars were: at least 20 images/s on set A (500 and 800 images, 16 at a time, one card),
encoder busy above 80 percent, API CPU at most 0.047 CPU-s per image, three repetitions, re-embedded vectors
identical to the pre-change pass. **Measured live on release/0.7.0**
(`docs/research/v070_measurement_log.md`, raw files in `docs/benchmarks/v070/`): the offload and read-ahead
alone moved COCO (set A, 1,500 images) from 15.7 to 16.6 images/s at 0.065 API CPU-s per image (0.052 before),
so **the bar is not met on set A** (E1, F3). On set P (about 21 MP) the GPU decode route below raised the pass
to 28 images/s (F1) and the two-chunk window to 51 images/s at 0.025 CPU-s per image and 85 to 92 percent
encoder busy (G3), which meets all three bars on that set; vectors agree with the pre-change pass to a
median cosine of 0.9999965, not bit for bit (G4).

Re-embedded crop vectors against the ingest vectors (same project, 3,166 and 5,133 crops): median cosine
0.999997 (identical), first percentile 0.9981 and 0.99992, minimum 0.9728 and 0.9337; 35 and 44 crops are
below 0.999. Crops whose box touches the right or bottom edge are 5 percent of the crops and 57 percent
of those below 0.999 (20 of 35 and 25 of 44; 20 of 153 edge crops, 13 percent, against 15 of 3,013 others, 0.5 percent). Cause:
ingest crops the detector's pixel box, which can extend past the frame, and Pillow pads the outside with
black; the stored `bbox_norm` is clamped to the frame, so the re-embed crops less. The padding cannot be
recovered from the stored box. **Decision (0.7.0, owner, #259): the vector is computed from the box that is
stored.** Ingest now cuts the image-clamped rectangle on both decode paths (`geometry.crop_rect`, repeated in
`models/gpu_image_prep/1/prep_geometry.py` and held equal by `tests/curation/test_crop_is_stored_box.py`), so
a crop touching the right or bottom edge is embedded without the black padding, and the crop-cache JPEG is the
stored box too. Interior boxes are bit-identical to 0.6.0. Vectors ingested before this change keep the padded
input; re-embed them (`reprocess` scope `embed`, parts `crop`) to move them into the stored-box space. The
reprocess embed and the region-box refresh already cut the stored box. The P-pe parity capture against the 0.6.0
reference (crop cosine p01 at least 0.995) is still to run.

#### Embed pass on the GPU decode (#259, 0.7.0)

The offload and read-ahead above moved the work, they did not remove it: live on release/0.7.0
(`docs/research/v070_measurement_log.md`, E1) COCO went from 15.7 to 16.6 images/s at 0.065 API CPU-s per
image, and set P (about 21 MP) stayed at 10.5 to 11.9 images/s with the GPU 10 to 16 percent busy, because
the pass decoded every 21 MP photo in the API (0.52 CPU-s per image) while ingest sends the same photos to
`gpu_image_prep`. The pass now takes the same route for large photos.

**What runs.** When the ingest GPU decode is usable (`OP_INGEST_GPU_DECODE` on, the model loaded for the
ingest profile, no secondary detector: the same readiness rule as ingest) and the loaded model files declare
the new optional `boxes` input, a photo at or above `OP_INGEST_GPU_MIN_MEGAPIXELS` (3.0) is sent to
`gpu_image_prep` as its stored bytes plus its stored boxes: the item's `bbox_norm` (part `crop`) and its
embeddable region boxes (part `region`), as FP64 rows so the model cuts exactly the rectangle the CPU pass
cuts. A box asked for both parts is sent once. In this stored-box mode the model decodes once (nvJPEG, EXIF
orientation), skips the detector, the blur sums and the crop JPEGs, and returns the whole-frame vector and
one vector per box; the API only reads the file, sends it and writes vectors. Photos below the gate stay on
the CPU path unchanged (a COCO frame decodes in 5 ms on one core), so the COCO rows above do not move.
Code: `src/services/curation/reprocess_gpu_embed.py`, `reembed_items(..., gpu=...)` in
`reprocess_embed.py`, stored-box mode in `models/gpu_image_prep/1/model.py`, the rectangle in
`prep_geometry.stored_box_rects` (held equal to the CPU pass by `tests/models/test_gpu_image_prep_geometry.py`).

**Fallback (fail closed, the same cases as ingest).** Below the gate, an unreadable header, any status other
than `ok` (progressive, truncated, decoder, ...), a failed call, or an answer with a row count other than the
boxes sent: that photo is decoded in the API as before. Each photo is counted in
`op_reprocess_decode_total{route}` (`gpu` or `cpu_<reason>`) while the route is usable. Model files older
than the API (no `boxes` input) disable the route for the run (log `reprocess_gpu_embed_model_outdated`), with
no call per photo. The single-item region refresh after a box edit, the backfill script and the auto-label
embed stage keep the CPU path.

**Numerics against the CPU pass.** Same rectangle (exact, tested). Crop part: the GPU square is plain
bilinear as `resize_crop_rgb` (within one grey level in the host model of the kernel), frame part: Pillow
`BILINEAR` antialiased (within one grey level); nvJPEG's IDCT differs from libjpeg by a grey level here and
there. This is the parity ingest's GPU decode already has against its CPU path (#217,
`docs/design/gpu_decode_plan.md`: same boxes, crop cosine p01 0.996 to 0.998 on every set). Region part (#360): every route embeds the decoded pixels of the stored box. The CPU pass, the
backfill script and the detection worker's per-box stage used to embed each region crop after a JPEG round
trip at quality 92 (the worker cut it from the item-crop JPEG). On a 21 MP photo that moved the vector of a
small box a lot (live check on 5,177 boxes: cosine against the GPU vector median 0.965, p01 0.886, min 0.842;
median 0.933 for boxes under 2 percent of the frame), because the encoder client shrinks a large crop by a
large factor and so turns the JPEG noise into a vector change. The GPU vector agreed with the raw-pixel CPU
vector at a median of 0.9993. The CPU paths now cut the box from the decoded source (`crop_pil`, the same
rectangle as the crop vectors) and embed it without re-encoding, so a region box equal to the item box gets the
same vector as the crop. Vectors of a photo are not bit-identical between the two routes (decoder and resize
differences, as for crops); within one route repeated passes agree. The old "cosine above 0.99 for region
vectors" expectation held only for small photos and is withdrawn.

**Chunk window.** The first live run of the GPU route reached 28 img/s with the GPU about half busy while ingest
of the same photos reached 45 to 49. The in-flight bound was not the limit (16, 24, 32 and 48 photos gave 27.6,
27.5, 27.1 and 26.0 img/s); the barrier between 64-photo chunks was (a chunk of 256 gave 33.1, one chunk of all 1,500
photos 34.2). A pure `embed` pass now embeds two chunks at once (`EMBED_CHUNKS_IN_FLIGHT` in `reprocess_images.py`):
the next chunk starts on its photos while the last ones of this chunk finish and its vectors are written. The two
chunks share one in-flight bound of 16 photos, so decoded pixels in memory do not grow. A chunk is still read,
embedded and written as a unit: a failed chunk fails only its own images, progress is reported in chunk order, and
a cancel waits for the chunks already started. A pass that also runs `detect` or `open_vocab` keeps one chunk at a
time (the embed must see the items those scopes changed). Measured: 51 img/s (see the measurement log, section G).

**Model files.** The change touches `models/gpu_image_prep/config.pbtxt`, `1/model.py` and
`1/prep_geometry.py`. Triton reads them at load: copy the three files into the model repository the stack
mounts (in a checkout that is `models/` itself) and reload the model (unload and load through Triton model
control, or restart `triton-server`). In an installed directory `./openprocessor upgrade` copies the files
(`export/sync_model_seed.py` replaces a model directory whose `MODEL_VERSION` is older than the release's;
the installer used to keep the old files with `cp -rn`), and `./openprocessor restart triton-server` reloads
them. Ingest requests do not send `boxes`, so ingest is unchanged by the model change.

**Live procedure (run in the measurement log, sections F and G).** On the test stack (own compose project name, GPU 0 or 2, quiet host), for
set P (300 photos, 16 to 24 MP) and COCO 1,500 as in E1, three repetitions each, a fresh project per set:

```bash
P=<project>; API=http://127.0.0.1:4603; TRITON=http://127.0.0.1:4600; API_CTR=<api container>
# 1. the model declares the input (expect "boxes" in the list) after the reload
curl -s "$TRITON/v2/models/gpu_image_prep" | python3 -c 'import json,sys; print([i["name"] for i in json.load(sys.stdin)["inputs"]])'
# 2. counters, API CPU and GPU before the pass
curl -s "$API/metrics" | grep '^op_reprocess_decode_total'
docker exec "$API_CTR" cat /sys/fs/cgroup/cpu.stat | grep usage_usec
nvidia-smi -i <triton gpu> --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 500 > gpu.log &
# 3. the pass (every vector rewritten, as E1); poll the job until it finishes
curl -s -X POST "$API/curation/projects/$P/reprocess" -H 'content-type: application/json' \
  -d '{"targets":{"filter":{"all_images":true}},"scopes":["embed"],"embed":{"parts":["crop","frame","region"]},"dry_run":false}'
curl -s "$API/curation/projects/$P/reprocess/jobs/<job_id>"   # images_done, started_at, finished_at
# 4. the same three readings after; img/s = images / (finished_at - started_at)
```

Expected on set P: `op_reprocess_decode_total{route="gpu"}` grows by the photo count (any `cpu_*` growth is
a finding), at least 20 images/s (ingest reaches 31.8 with one `gpu_image_prep` instance and 48.0 with two on
the same photos, E4, while doing more per photo), API CPU at most 0.047 CPU-s per image (the API now reads,
sends and writes only) and GPU busy above 50 percent. Then the vectors: crop and frame vectors after the pass
against the same project's ingest vectors (median 0.99999 or better, p01 at least 0.995, as the GPU-decode
ingest parity), and region vectors after the pass against those of a no-round-trip CPU pass on the same project (GPU decode
off: the API restarted with `OP_INGEST_GPU_DECODE=0` for that run, on the build with the #360 fix; p01 at least
0.99; a CPU pass from before the fix is not a valid reference). COCO must reproduce the E1 release/0.7.0 row
(all photos counted `cpu_small`). If set P stays under 20 images/s with the GPU under 50 percent, the
in-flight bound of the pass (16 photos, `_IMAGES_IN_FLIGHT` in `reprocess_embed.py`) or the 64-photo chunk
barrier is the next limit to test (the chunk barrier was the limit, see "Chunk window").

#### Parity (final configuration against `ref-u8fp16` and `ref-u8fp16-repeat`, `precision` profile)

One capture of the final configuration (shipped engines and Triton settings, one card, set A, policy `all`,
200 probe images for the CLIP, face and OCR gates); summaries in `docs/benchmarks/parity/stage2_tuning_vs_u8fp16/`
and `stage2_tuning_vs_u8fp16-repeat/`. All gates pass; `P-vlm` is skipped (no VLM in the stack).

| Gate (rule) | Against `ref-u8fp16` | Against `ref-u8fp16-repeat` |
|---|---:|---:|
| P-box images identical (>= 0.98) | 0.99242 | 0.99343 |
| P-item (>= 0.99) | 0.99899 | 0.99899 |
| P-pe crop cosine p01 (>= 0.995) | 0.99977 | 0.99971 |
| P-pew frame cosine p01 (>= 0.995) | 0.99944 | 0.99945 |
| P-knn overlap@10 (>= 0.95) | 0.98210 | 0.98280 |
| P-cluster-assign (>= 0.98) | 0.99898 | 0.99836 |
| P-cluster-purity drop in points (<= 0.5) | 0.24 | -0.21 |
| P-label-vote (>= 0.97) | 0.992 | 0.992 |
| P-clip, P-face, P-ocr | 1.0000 | 1.0000 |

#### End to end, before and after (set A, 16 x 16, one card)

| | images/s | encoder busy |
|---|---:|---:|
| Before (multi-GPU section, one A6000, uint8 + interleaved PE engine) | 23.39 [22.96-23.41] | 0.95-0.98 |
| After (this checkout: half-split engine, default settings) | 28.81-29.07 (three series of 3 repetitions, 0.0 percent foreign load) | 0.95-0.96 |
| Two cards before (multi-GPU section) | 44.60 [44.29-45.47] | 0.87-0.93 per card |
| Two cards after, 16 x 16 | 55.08 [55.04-55.11] (1.89x one card) | 1.79-1.82 of 2 |
| Two cards after, 32 x 16 | 56.37 [55.24-56.97] | 1.82-1.88 of 2 |
| Two cards after, 16 x 32 | 50.17 [48.98-51.68] | 1.62-1.71 of 2 |

#### Not measured in the stage 2 pass

- Two-card sweep: only the confirmation points above were run (16 x 16, 32 x 16, 16 x 32 at shipped settings);
  instance counts, delays and worker counts were swept on one card only. 2-card API CPU stays 0.06 s/image.
- Set B (photos): the pinned images were removed from the scratch store by another process and the
  re-download was rate limited (HTTP 429); no photo numbers.
- The `minimal` and `standard` profiles, the RTX 3080 Ti, detector CUDA graphs in Triton, a full
  factorial of client x instances x delays x workers (one factor at a time was run), `priority_levels`.
- The embed-pass fix (above).

### Decode once and the GPU decode (#217)

Built after the owner decisions on the GPU decode plan (`docs/design/gpu_decode_plan.md`, section
13 has the full A/B and the go / no-go). One isolated stack on GPU 2, `/ingest/batch`, 3 repetitions,
median images/s; host CPU is the API plus Triton containers.

| Set B photos (12-24 MP) | CPU path | CPU single decode + libjpeg-turbo | GPU decode (`OP_INGEST_GPU_DECODE`, on by default) |
|---|---:|---:|---:|
| detect + embed, batch 8 x 16 clients | 27.0 | 23.5 (26.8 without libjpeg-turbo) | **35.0** |
| detect only, batch 8 x 16 | 28.3 | 31.0 | **44.0** |
| one client, batch 1 (p50 latency) | 0.97 (0.98 s) | 1.17 (0.81 s) | **2.40 (0.36 s)** |
| four clients, batch 1 | 3.45 | 4.48 | **8.30** |
| host CPU-s per photo at bulk | 1.12 | 1.00 | **0.34** |

COCO frames (0.3 MP) are faster on the CPU (forced onto the GPU decode: 22.7 against 29.0
images/s with embedding, 63.6 against 124.5 detection only), so the GPU decode takes only images at
or above `OP_INGEST_GPU_MIN_MEGAPIXELS` (3.0). The CPU single decode alone does not speed up bulk
ingest (the next binders are elsewhere); it is the CPU path the GPU decode falls back to.

The GPU decode is the default, sized by hardware. It adds about 8.5 GB (8,704 MiB, loaded on first use)
per model instance to the Triton card (`full` runs two instances, 17,408 MiB, #268; the other profiles
run one if you turn the GPU decode on), on top of the profile's own models (about 8, 12 and 17 GB for `minimal`,
`standard` and `full`). Only `full` (48 GB+) has room, so the installer writes
`OP_INGEST_GPU_DECODE=1` for it and `0` for `minimal` and `standard`; a 12 GB card does not hold
the full model set beside other projects' resident processes, let alone the GPU decode. A checkout
whose `.env` has no such line gets the code default (on), and an explicit value always wins.
`openprocessor gpu plan` prints whether it is on and why, and warns (never refuses) when it is on
for a card without room. If the model cannot load, Triton leaves the other models serving and
every photo is ingested on the CPU path, as with a declined file.

Instance count (#268, one RTX A6000, 3,275 photos of about 18 MP, three runs each): one
`gpu_image_prep` instance 31.8 images/s, two 48.0 (+51 percent; GPU utilization 70-75 to 78-87
percent, Triton CPU 610 to 865 percent), three adds 0.6 percent over two for another 6 to 7 GB. The
single instance is one Python process: its worker threads share one GIL. `gpu_image_prep_instances`
in `config_templates/profiles/*.json` sets the count (`full` 2, `minimal` and `standard` 1), the
installer writes it into `models/gpu_image_prep/config.pbtxt`, and `openprocessor gpu plan` sizes the
GPU decode as count times 8,704 MiB. The committed config carries 2, the `full` value.

### Stage 4: coalesced ingest writes, Lucene engine, write ceiling on the final mapping (#248, #254, #230)

Three storage changes measured on the product stack: one RTX 3080 Ti (GPU 1, 12 GB, shared with other
projects' resident processes, none busy), 24 API workers, FP16 PE engine, OpenSearch 3.6.0 with an 8 GB
heap on the same host (load average 13 to 36 from other projects), set A (public COCO, 2,000 images,
policy `all`, 12,803 items), 32 images per request, 8 clients, fresh project per run. "Before" is the
tree at the parent commit, "after" this branch, switched by changing the mounted `src/` of the same
containers. Raw files: `docs/benchmarks/stage4_*.json` (ingest, write path, parity, ceiling).

#### 1. Where the write path sits at the GPU-bound rate

| Route | Arm | images/s median (min-max, 4 runs) | API CPU s/image | request p50 / p95 s | `opensearch_write` mean ms |
|---|---|---:|---:|---:|---:|
| `/ingest/batch` | before | 26.6 (26.5-26.9) | 0.065 (0.059-0.069) | 9.7 / 11.3 | 184 |
| `/ingest/batch` | after | 26.4 (25.8-28.2) | 0.062 (0.045-0.065) | 9.9 / 11.0 | 392 |
| `/ingest/upload` | before | 23.5 (21.7-25.8) | 0.070 (0.069-0.071) | 10.7 / 14.6 | 432 |
| `/ingest/upload` | after | 23.7 (20.5-27.6) | 0.067 (0.062-0.069) | 10.1 / 14.0 | 937 |

At 23 to 27 images/s the coalesced writer wins nothing end to end: throughput, API CPU and request
latency are inside run-to-run noise, and the GPU is the binder as before. The `opensearch_write` stage
is wall time per image and grew because an image now waits for a shared flush; the request p50 did
not move. The CPU stages of the write path are small at this rate (`doc_build` 0.07 and
`bulk_serialize` 0.05 ms per call); the gain is headroom, below.

#### 2. The write path alone at the rates decode work will unlock

`scripts/bench/ingest_write_path.py` replays real ingest documents (the images doc and items docs of
a real project, vectors as stored; beyond the 2,000 source images they are reused with a small
perturbation, the last column is the exact-real share) through the product client and guard in one
process, open loop, 20 s per point, 2 repetitions in opposite order, with 20 review-default k-NN
queries per second running in the same process. `perimage` is the previous behaviour: an images bulk,
an `mget` and an items bulk per image (about 7 documents each), no limit on writers.

| Offered img/s | Arm | achieved img/s | API CPU ms/image | requests/image | docs per bulk | write p50 / p99 ms | k-NN p50 / p95 ms | exact real |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 25 | perimage | 24 | 10.1 | 3.00 | 7 | 593 / 2,494 | 23 / 449 | 1.00 |
| 25 | coalesced | 24 | 9.5 | 1.17 | 19 | 441 / 1,151 | 22 / 28 | 1.00 |
| 50 | perimage | 25 | 7.6 | 3.00 | 7 | 15,680 / 28,606 | 1,159 / 9,821 | 1.00 |
| 50 | coalesced | 48 | 5.8 | 0.55 | 41 | 519 / 1,539 | 21 / 29 | 1.00 |
| 100 | perimage | 26 | 7.3 | 3.00 | 7 | 45,720 / 62,996 | 2,945 / 23,521 | 0.99 |
| 100 | coalesced | 96 | 3.8 | 0.26 | 85 | 575 / 1,659 | 22 / 39 | 0.99 |
| 200 | perimage | 28 | 8.0 | 3.00 | 7 | 96,118 / 121,942 | 13,166 / 42,292 | 0.50 |
| 200 | coalesced | 175 | 3.2 | 0.05 | 530 | 2,687 / 5,091 | 42 / 179 | 0.49 |
| 200 | coalesced, 500-doc bulks | 180 | 3.1 | 0.07 | 324 | 1,363 / 2,871 | 24 / 71 | 0.51 |
| 200 | coalesced, 8 writers | 187 | 3.1 | 0.12 | 182 | 1,561 / 3,402 | 24 / 71 | 0.49 |

- One process on the per-image path stops at about 25 to 28 images/s (3 requests each through the
  client's 10-connection pool) and queues without bound above it; the coalesced writer keeps up to 100
  images/s and reaches 175 to 187 at 200. In production 24 workers share the load, so the per-process
  knee is the relevant number when a few workers carry the decoded frames.
- API CPU per image falls 62 percent at 100 images/s (7.3 to 3.8 ms) and requests fall 3.00 to 0.26 per
  image. At 25 images/s CPU is unchanged (10 ms) but k-NN p95 under the same writes falls from 449 to
  28 ms, because the node no longer takes 75 small bulks per second.
- No linger (`coalesced_nolinger`) and bulk caps of 500 against 1,000 differ by under 10 percent below
  200 images/s; 500 documents and 8 writers are slightly better at 200. The defaults stay at 1,000
  documents and 4 writers per process (24 processes in the stack).
- Not measured here: node CPU per image (the cgroup counter included merges and neighbours' load).

#### 3. Lucene engine and `ef_construction` 128 on the product stack

Two projects per engine from the same 2,000 images: faiss with `ef_construction` 512 (the previous
default, built from the parent tree) and Lucene with 128 (this branch). `scripts/bench/store_parity.py`
runs the product's k-NN query builder with the review-default filter (and a class and a cluster filter)
against the live indexes for 1,000 items and compares the top 10 with exact cosine search over the
stored vectors; cluster training ran on all four projects. The same-engine rows are the noise floor
(two ingests of identical images differ in about 1 percent of boxes).

| Pair | review default overlap@10 | review default recall vs exact (ref / cand) | class filter (33 %) overlap@10 | class filter recall (ref / cand) | cluster ARI | cluster purity change, points |
|---|---:|---|---:|---|---:|---:|
| faiss 512 against faiss 512 | 0.9935 | 0.9998 / 0.9997 | 0.904 | 0.938 / 0.912 | 0.338 | -0.56 |
| faiss 512 against Lucene 128 | 0.9923 | 0.9998 / 0.9995 | 0.930 | 0.938 / 1.000 | 0.344 | -0.79 |
| faiss 512 against Lucene 128 (second pair) | 0.9902 | 0.9997 / 0.9992 | 0.910 | 0.918 / 1.000 | 0.338 | +0.53 |
| Lucene 128 against Lucene 128 | 0.9883 | 0.9994 / 0.9993 | 0.987 | 1.000 / 1.000 | 0.341 | +0.77 |

- Gate (overlap@10 >= 0.95 on the review default): **passes**, 0.990 to 0.992 across engines against
  0.988 to 0.994 within an engine. Cosine `_score` of the same neighbour differs by under 0.0005 at the
  99th percentile, so `min_score` and the UI chip are unaffected.
- With a class filter the faiss index loses recall to the filter (0.92 to 0.94) and Lucene does not (1.000);
  the cross-engine overlap there (0.91 to 0.93) is faiss's miss, not Lucene's.
- Cluster agreement (ARI 0.338 to 0.344) and purity change (-0.79 to +0.77 points) are inside the spread
  of two k-means runs of one engine, because cluster training reads the stored vectors and does not
  query the index. The cluster-filter overlap is not meaningful across projects (cluster ids are
  per project); its recall against exact search is 1.000 on both engines.
- Disk of the items index for the same 12,803 items: 56 MiB on Lucene against 108 MiB on faiss (4.5
  against 8.6 KiB per item, whole documents included).

**Existing indexes.** The engine of a vector field is fixed when the field is created, so an existing
project keeps its faiss fields and `ef_construction` 512 and keeps working (queries and scores are
the same on both engines). New projects, and vector fields added to an old index by the startup
mapping migration, get Lucene with 128. No migration or reindex tool ships: to move a project, create
a new project on this build and re-ingest or re-import its images. `OP_HNSW_EF_CONSTRUCTION` applies
to indexes created after it is set.

#### 4. Write ceiling at 100,000 real vectors, final mapping

`scripts/bench/os_write_ceiling.py` with `--noise 0`: 100,000 distinct real item documents and 15,625 real
image documents, copied from a real 16,000-image COCO ingest on this stack (102,677 items), written
to fresh indexes with that project's mapping and settings (Lucene, `m` 16, `ef_construction` 128,
one shard, no replica, 1 s refresh). Medians of 2 repetitions on a loaded host.

| Writers x bulk | docs/s | images/s equivalent | searchable s | bulk p95 s | node cores |
|---|---:|---:|---:|---:|---:|
| 1 x 1,000 | 473 | 64 | 245 | 3.7 | 4.9 |
| 4 x 1,000 | 1,980 | 268 | 60 | 3.6 | 6.9 |
| 8 x 1,000 | 4,426 | 598 | 26 | 2.8 | 9.6 |
| 16 x 1,000 | 6,271 | 847 | 19 | 3.3 | 15.3 |

- Lucene is slower per writer than the faiss rows of probe 5 (473 against 1,011 documents/s with one
  writer; the host was also busier), so one writer no longer carries 100 images/s: four do (268
  images/s equivalent). That is what the coalescer's 4 writers per process provide. 429 retries: none.
- **Go / no-go against the 100 images/s target (#230): go.** An ingested image writes about 7.4
  documents and 7.4 embeddings (6.4 items and one image document, the proportions of this
  table), so 100 images/s asks the datastore for about 740 documents/s. The measured ceiling on the
  final mapping is 1,980 documents/s with 4 writers (268 images/s, 2.7 times the target) and 4,426
  with 8 (598 images/s), both with no 429 retries and `ef_construction` 128, which cuts node CPU
  per 1,000 documents from 2.39 to 1.49 CPU-s at equal throughput (probe 5, 3,365 against 3,206
  documents/s). One writer (473 documents/s, 64 images/s) does not carry 100 images/s on the Lucene
  engine; the coalescer's 4 writers per process do. So the datastore is not what holds ingest
  below 100 images/s: the measured ingest rates are bound by the GPU stages, and the write path
  has headroom of 2.7 to 6 times at the target. The conditions are one node, one shard, an 8 GB heap
  and a loaded host; the 1M point and the cluster pass at that size stay with #232.
- Stored size of the source project's items index: 448.5 MiB for 102,677 items (4.5 KiB each); heap in
  use 4.4 of 8 GB after the runs.
- **Extrapolation, not run (1M):** a linear extrapolation of the sizes above gives about 4.4 GiB of
  items and 0.5 GiB of images on disk for a million items, a load of 5 to 10 minutes at the 8-writer rate
  if it held (it will fall as merges grow), and the Lucene graph read through the page cache instead of
  heap. One shard per
  project is enough to this size; split into more shards only when a single shard passes tens of GB,
  because each shard merges alone. Keep the heap at 8 GB for the HNSW build buffers and let the rest
  of the RAM be page cache. This is a guide, not a measurement; the 1M point stays with #232.

#### Not measured

- 1M vectors (real or noisy), 250k and 500k points; a merge-time and load-time comparison of faiss against
  Lucene at 100k on this build (the vector store evaluation has the 100k figures: merge 50 percent slower,
  load 36 percent slower); node CPU per image of the two write paths; set B (photos) end to end.
- The 4-writer, 1,000-document default was not tuned past the points above; ingest throughput at the
  GPU-bound rate does not depend on it.

### Ingest transports (#263)

Question: would a gRPC ingest channel beat HTTP beyond noise? Decision protocol:
`docs/design/ingest_surfaces_plan.md` section 8. **Decision: no. The rule says skip, nothing is built.**

Measured by: `scripts/bench/ingest_transport_bench.py` (transport only, no pipeline, no GPU, no
OpenSearch) and, for the pipeline side, `scripts/bench/baseline_suite.py --phases ingest --modes upload`,
both on 2026-10-14 on the shared dev host (load average 5 to 6, other projects running; whole-host CPU busy
10 to 15 percent in every transport run, GPU 2 with no foreign load in every sample of the pipeline runs).
Medians with the minimum and maximum of 3 repetitions in parentheses. The server reads each image's bytes,
computes the imohash the real route computes first and returns a `BatchIngestResponse`-sized result; the client
is a separate process; HTTP is uvicorn (uvloop, httptools) with FastAPI `File` parsing, gRPC is `grpc.aio`
with hand-framed byte messages (a protobuf `bytes` field adds about 5 bytes per message). Server and client
CPU are `psutil` process times. Bytes on the wire are computed from request and response sizes plus header
and HTTP/2 frame estimates, not captured. HTTP/2 multipart was not measured: hypercorn and h2 are not in
the repo venv.

Data: set A is the pinned COCO set (2,000 images, 640 x 480, 333 MB, 32 per request). Set B is **only 8**
of the 500 pinned Commons photos (45 MB, 5.6 MB mean, 8 per request, sent 12 times for 96 images): the
Wikimedia server answered 429 to the bulk download, so the 500 were not fetched. The transport share for
set B uses the 1.10 CPU-seconds per image recorded above, not a new pipeline run (no set B pipeline run was
made).

**Transport only, set A** (2,000 images; an earlier full run, with the same picture, differs by less than the ranges shown; raw
JSON kept outside the repo):

| contender | clients | images/s | request p99 s | server CPU s/image | client CPU s/image | wire B/image |
|---|---:|---:|---:|---:|---:|---:|
| HTTP/1.1 multipart | 4 | 2433 (2303-2440) | 0.06 | 0.00039 (0.00038-0.00040) | 0.00041 | 166,863 |
| HTTP/1.1 multipart | 16 | 2203 (2197-2515) | 0.26 | 0.00041 (0.00036-0.00042) | 0.00045 | 166,863 |
| HTTP/1.1 multipart | 32 | 2244 (2147-2459) | 0.48 | 0.00039 (0.00037-0.00041) | 0.00044 | 166,863 |
| gRPC client stream | 4 | 3310 (2432-3572) | 0.07 | 0.00049 (0.00046-0.00056) | 0.00030 | 166,826 |
| gRPC client stream | 16 | 2919 (2770-3931) | 0.22 | 0.00046 (0.00042-0.00050) | 0.00031 | 166,826 |
| gRPC client stream | 32 | 3145 (2888-3478) | 0.38 | 0.00043 (0.00042-0.00046) | 0.00031 | 166,826 |
| gRPC unary | 4 | 3320 (3053-4382) | 0.07 | 0.00040 (0.00031-0.00045) | 0.00026 | 166,820 |
| gRPC unary | 16 | 3452 (3180-3566) | 0.21 | 0.00038 (0.00038-0.00040) | 0.00029 | 166,820 |
| gRPC unary | 32 | 3051 (2758-3620) | 0.44 | 0.00040 (0.00036-0.00042) | 0.00034 | 166,820 |

**Transport only, set B** (8 photos x 12, 8 per request):

| contender | clients | images/s | server CPU s/image | client CPU s/image | wire B/image | server peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| HTTP/1.1 multipart | 4 | 79.8 (74.8-86.2) | 0.0245 (0.0235-0.0263) | 0.0045 | 5,865,195 | 535 |
| HTTP/1.1 multipart | 16 | 87.2 (75.7-88.2) | 0.0272 (0.0243-0.0300) | 0.0043 | 5,865,195 | 549 |
| HTTP/1.1 multipart | 32 | 81.3 (79.9-88.8) | 0.0265 (0.0250-0.0285) | 0.0044 | 5,865,195 | 661 |
| gRPC client stream | 4 | 67.0 (54.2-68.7) | 0.0198 (0.0191-0.0217) | 0.0112 | 5,868,394 | 664 |
| gRPC client stream | 16 | 56.7 (50.2-60.1) | 0.0191 (0.0181-0.0215) | 0.0130 | 5,868,394 | 1520 |
| gRPC unary | 4 | 76.9 (73.3-83.8) | 0.0147 (0.0142-0.0152) | 0.0145 | 5,868,258 | 970 |
| gRPC unary | 16 | 63.0 (61.7-70.3) | 0.0143 (0.0140-0.0146) | 0.0151 | 5,868,258 | 1295 |

(The 32-client gRPC rows follow the same pattern; they are in the harness output.) On set B gRPC
moves server CPU to the client (client CPU rises by 0.007 to 0.011 s per image while server CPU falls by
0.005 to 0.013), uses more server memory, and is slower in median images/s than HTTP at every client count.

**Amdahl bound and the rule.** Pipeline API CPU per image on set A, measured live on the same day through
`/ingest/upload` (640 images, 16 clients x 32 per request, policy `all`, 3 repetitions, fresh project each):
**0.072 (0.070-0.075) s** at **29.0 (28.9-29.2) images/s**, GPU 2 **90.5 (90.5-95.2) percent** busy, no
foreign GPU load, 0 failed; Triton CPU 0.013 s per image. For set B: 1.10 s per image, 15 images/s, GPU
bound by CPU decode (the Wave 0 findings above). The rule (saving of at least 10 percent of that
CPU, with repetition ranges that do not overlap):

| set | best server CPU saving of gRPC, s/image | share of pipeline API CPU | passes |
|---|---:|---:|---|
| A | 0.00002 (unary, 16 clients; negative or zero in 5 of 6 rows) | 0.03 percent | no |
| B | 0.0129 (unary, 16 clients; ranges do not overlap) | 1.17 percent | no |

The best case is 8.5 times below the 10 percent bar. On set B the bar would need a pipeline API CPU of
under 0.13 s per image, against 1.0 to 1.3 s measured.

**Reading.**

- Transport is not where the time goes: 0.0004 s of server CPU per image on set A against 0.072 s for the
  pipeline (0.6 percent), 0.025 against 1.10 on set B (2.2 percent). The ingest rate is bound by the GPU on set A
  (90 percent busy) and by decode CPU on set B, not by the
  wire, so a faster transport cannot raise images/s.
- Bytes per image are the same to within 0.1 percent (a JPEG is already compressed; gRPC adds 5 bytes of
  framing per message, multipart a boundary and a few header lines).
- `POST .../ingest/batch` sends no image bytes at all (paths the API reads itself, under 100 bytes per image
  on the wire), so a byte channel could only compete with `/ingest/upload`, and the client already spends
  0.0004 s of CPU per image there.
- Caveats: one host with other jobs running (the ranges show the noise; set A server CPU differences of
  0.0001 s are inside it); client concurrency is asyncio tasks, not threads; set B is 8 distinct photos; the
  set B pipeline cost is a recorded figure, not a re-run; `/ingest/batch` was not re-measured live (its
  paths must sit under the configured source root, which holds no public set here); HTTP/2 multipart not
  measured; the pipeline run used 640 images, not 2,000. None of these can move a 1 percent share to 10.

Reproduce:

```bash
.venv/bin/python -m scripts.bench.ingest_transport_bench --images <dir> --label <name> \
  --pipeline-cpu <API CPU s/image> --concurrency 4,16,32 --reps 3 --out transport.json
```

### Scale ladder: 10k and 100k

Three runs of the bulk ingest runner ([Bulk ingest runner](#bulk-ingest-runner)) at larger sizes than the 2,000-image
baseline: 10,000 public COCO images, 100,000 public Open Images, and 100,000 camera-original photos. All three
use the 0.6.0 defaults (policy `all`, the 3-stage index policy defaults, 16 images per request, 16 requests in
flight, FP16 PE engine, GPU decode on) and a fresh project. The 1M-image step is not run (planned for the next release).

#### Sets and environment

| Run | Images | Source |
|---|---|---|
| coco-10k | 10,000, all under 1 MP, 1.65 GB | public COCO train and val 2017, seed 20261009 (`--bench-set 10000`, same command as the 2k pin) |
| oi-100k | 100,000, each at most 1,024 px (about 0.7 MP) | public Open Images v7 train, every image CC BY 2.0, seed 20261009; manifest sha256 `b9deca207878b569...` |
| hires-100k | 100,000, median about 20 MP, about 6.8 MB each | a private collection of 20 MP camera-original JPEGs, not redistributable; aggregates only |

One RTX A6000 (GPU 2) for Triton, VLM stopped. OpenSearch data and the source images are on a NAS (RAID6 over a
network share), Docker data is on the NAS too, and the host is shared with other workloads. One earlier attempt
window was invalid because the NAS saturated on IO from another job; the runs below are the repeat. Each run
was stopped once with a forced cancel part-way and resumed with the same source (see Stop and resume).

#### Results

| Measure | coco-10k | oi-100k | hires-100k |
|---|---:|---:|---:|
| Throughput, images/s | 29.04 to 29.08 | 35.96 | 21.00 |
| Share of the 2k baseline (29.0 images/s) | 100 % | 124 % | 72 % |
| Wall time, ingest | 311 s (run A, one leg); 155 s second leg (run B) | 2,781 s | 4,761 s |
| Legs (forced stop, then resume) | 2 (run B) | 1,384 s + 1,397 s | 2,582 s + 2,179 s |
| Failures | 0 | 2 (decode, see below) | 0 |
| Duplicate images | 0 | 0 | 2,120 |
| Crops indexed and embedded | 64,415 | 490,804 | 236,689 |
| Crops per stored image | 6.44 | 4.91 | 2.42 |
| Crops embedded per second | about 187 | about 177 | about 50 |
| Cluster pass (CPU, after the last ingest) | 269 s at 64,415 crops | 2,042 s at 490,804 | 1,802 s at 236,689 |
| Index size on disk | not measured (preview estimate 653 MB) | 8.86 GiB | 4.35 GiB |
| Index bytes per input image | not measured | 95.2 KB | 46.7 KB |
| Index bytes per crop | not measured | 19.4 KB | 19.7 KB |
| Peak GPU utilization and memory (card 2) | not recorded | 100 % and 21.1 GiB | 100 % and 25.8 GiB |
| Mean GPU utilization over the samples | not recorded | 94 % | 51 % |
| OpenSearch heap peak, segments peak | not recorded | 79 %, 159 | 79 %, 128 |
| Peak host CPU and memory | not recorded | 90 %, 152 GB | 79 %, 159 GB |

Images per second is images processed in the whole run divided by the ingest wall time including both legs
(the legs include the cost of the forced stop and the restart). The 29.0 baseline is the 2k pin on the same
build. Index bytes are the OpenSearch store size the driver sampled at the end of the run; the 2,120 duplicates in hires-100k are not stored, so its per-unique-image figure is 47.7 KB.
The two decode failures in oi-100k were fixed after this run by the Pillow header fallback (#333); the run was
not repeated.
The 10k cluster pass is the `cluster_residuals` stage (4 m 29 s); the other stages in it took 2 s or less.
The bytes per crop are the same in both 100k runs (about 19.5 KB), so the index grows with the crop count,
not the pixel count: a 20 MP photo costs the index less than a 0.7 MP one because it holds fewer objects.

#### Throughput over time

From the 15 s progress polls in `scale/oi-100k.jsonl` and `scale/hires-100k.jsonl` (images stored per second
over a rolling 5-minute window, per leg; a leg's first minutes include model and cache warm-up).

| Run and leg | Quarter 1 | Quarter 2 | Quarter 3 | Quarter 4 | 5-min window min, median, max |
|---|---:|---:|---:|---:|---|
| oi-100k leg 1 | 37.0 | 36.8 | 36.9 | 36.2 | 35.9, 36.9, 37.6 |
| oi-100k leg 2 | 36.6 | 37.3 | 36.3 | 31.9 | 32.1, 36.6, 37.7 |
| hires-100k leg 1 | 22.7 | 23.5 | 19.1 | 12.8 | 4.6, 19.5, 30.4 |
| hires-100k leg 2 | 23.5 | 22.4 | 21.5 | 23.7 | 15.1, 22.4, 28.6 |

- **oi-100k does not decay.** Leg 1 is flat to within 2 percent over 1,368 s (37.0 to 36.2 images/s). Leg 2 is flat for
  three quarters and drops to 31.9 in the last quarter, as the index passed 159 segments with merges running
  (the sampler saw three merges in flight in the second half); the whole-run rate of 35.96 includes it.
- **hires-100k is noisier and leg 1 decayed.** Leg 1 held 23 images/s for 20 minutes, then fell to 19, 16 and
  9 images/s over the last ten minutes before the forced stop (the minimum 5-minute window was 4.6). Leg 2, after the
  restart and on the same index, held 21.5 to 23.7 for its whole length. A restart that cures the slowdown points
  at something that builds up in a long-running process or on the host (the leg 1 GPU utilization also fell to 11 to 51 percent
  in that stretch) rather than at index size, because leg 2 started with the bigger index and did not slow down. This run
  cannot say which. The shared host and the NAS are both candidates (see Caveats).

#### Stop and resume

| Run | Stop | Resume | Result |
|---|---|---|---|
| coco-10k run A | cancel sent at about 7 % (the threshold used the growing discovery count; fixed in the driver) | restart returned the same run id, 944 images skipped | 10,000 stored, 0 failed, 29.08 images/s |
| coco-10k run B | forced cancel at 5,232 stored | the same run resumed (attempts 2), 5,488 skipped from the ledger | 10,000 stored, 0 failed, 29.04 images/s, 155 s second leg |
| oi-100k | forced cancel at 50,000 (50,654 stored by the time it took effect) | skipped 50,654, no re-embedding | 99,998 stored plus 2 failed |
| hires-100k | forced cancel at 50,000 (50,272 stored) | skipped 50,272 | 97,880 stored plus 2,120 duplicates, 0 failed |

In every case the stop and the restart changed no count: images stored plus skipped plus duplicates plus failed
equals the input exactly, and the project holds 99,998 images and 490,804 crops (oi-100k), 97,880 images and
236,689 crops (hires-100k), all embedded (`n_not_embedded` 0, `n_embed_failed` 0). `resumed` is only in the
response of the start call; `attempts` 2 on the stored run is the record of a resume.

#### Where the time goes

`stage_s` in the run record is cumulative worker-seconds summed over every concurrent worker, not wall seconds.
Dividing a stage by the run's `elapsed_s` gives its average concurrency; dividing by the 16 requests in flight gives
a loose share. Stages overlap (a request's time includes the stages inside it), so the columns do not add to 100 %.
For the resumed runs the record's `elapsed_s` (1,394.7 s, 2,176.7 s) matches the last leg, so these are the last
leg's figures.

| Stage (worker-s) | coco-10k | oi-100k | hires-100k |
|---|---:|---:|---:|
| `batch_requests` (whole request) | 4,899 | 21,813 | 33,999 |
| as average concurrency (over `elapsed_s`) | 15.7 | 15.6 | 15.6 |
| `embed` | 51,662 | 171,182 | 35 |
| `opensearch_write` | 5,042 | 28,971 | 49,432 |
| `admission_wait` | 0.04 | 0.2 | 13,919 |
| `decode` | 47 | 500 | 15 |
| `letterbox` | 26 | 590 | 7 |
| `whole_frame_preprocess` | 59 | 508 | 4 |
| `jpeg_encode` | 30 | 220 | 1 |
| `blur` | 28 | 309 | 3 |
| `triton_request`, `hash`, `crop`, `doc_build`, `bulk_serialize`, `pe_preprocess` (each) | under 45 | under 220 | under 18 |

- **All three runs keep 16 requests in flight all the time** (15.6 to 15.7 average), so the client side never
  starves; what differs is what the requests wait on.
- **The small-image runs wait on the GPU.** `embed` is 3.5 to 5.7 worker-seconds per image, summed over the
  crops of a request waiting in the Triton queue (about 123 waiters at once in oi-100k), while the CPU stages are
  each under 1 percent of it. Mean GPU utilization is 94 % and 100 % is the peak. The crops-per-second figures
  fit the same picture: 187 (coco-10k) and 177 (oi-100k) crops embedded per second on different image
  sets. That is why oi-100k beats the 29.0 baseline (124 %): its images carry 4.91 crops against 6.44, so
  at the same crop rate it stores 22 percent more images per second. The images-per-second figure is a property of the
  crops per image as much as of the pipeline.
- **The 20 MP run does not wait on the GPU.** `embed` is 35 worker-seconds, mean GPU utilization is 51 % (median 53),
  and the crop rate is 50 per second, a quarter of the small-image runs. The time sits in `opensearch_write`
  (49,432, 22.7 workers) and `admission_wait` (13,919, 6.4 workers). `admission_wait` is time spent waiting for
  the in-flight bytes budget (`OP_INGEST_INFLIGHT_MB`): at 6.8 MB per photo and 16 images per request, a share of requests waited for
  memory budget, about 0.28 s per image. At 21 images/s the runner moves about 143 MB/s of source files off the NAS.
- **What the stage timers cannot say.** In the 20 MP run `decode` is 15 s and `embed` 35 s, which are not credible
  as the cost of the work, so on the GPU decode path these timers do not cover the work and its time shows up
  in the stage that awaits it (`opensearch_write` here). Reading `opensearch_write` literally as the OpenSearch
  commit time would contradict probe 5 (one writer sustains the equivalent of 137 images/s). The defensible
  statements are: the GPU is idle half the time, so the limit is upstream of it (NAS reads of 6.8 MB files,
  the GPU decode and the in-flight bytes cap, or a combination) and the OpenSearch write stage is not shown to be
  the limit. Storage is on the same NAS as the sources, so a NAS-bound reading fits but is not proven; a run with
  the sources on local NVMe, or the same photos downsized, would separate them.
- **The small-image runs are not storage-write bound.** `opensearch_write` is 0.5 to 0.6 worker-seconds per image
  in coco-10k and oi-100k (21 workers in oi-100k) with the GPU saturated, and the throughput matches the crop rate
  of the GPU-bound baseline, so the NAS-hosted OpenSearch kept up with 177 documents per second of crops at
  100,000 images (heap peak 79 %, 159 segments).

#### Caveats

- One card, one run per configuration; the 100k runs are single observations, not medians. The coco-10k pair
  (29.08 and 29.04) is the only repeat, and it agrees within 0.2 percent.
- The host and the NAS are shared with other workloads. An earlier first attempt was lost to a NAS IO saturation
  episode. The hires-100k leg 1 slowdown may be the same kind of interference; the data cannot exclude it.
- The hires-100k set is private and cannot be reproduced; only its aggregates are published. Its crops
  per photo (2.42) depend on its content and do not transfer to other photo sets.
- `stage_s` for a resumed run covers only the final leg (see above), and the stage timers cover different work
  on the CPU and GPU decode paths.
- The cluster pass runs on the CPU and shares the host. Its time per crop was 4.2 ms in oi-100k and 7.6 ms in
  hires-100k, and the cause of the difference was not found.
- The coco-10k run recorded no GPU, heap or index-size samples; the preview route estimated 653 MB for it.
- Both 100k indexes were left unmerged (128 and 159 segments); size on disk after a force merge was not measured.

#### Bottlenecks and next steps

1. **Decode once on the GPU, stay on the GPU (#40).** The 20 MP run leaves the GPU half idle while the host
   moves 6.8 MB files and decodes; the small-image runs are GPU bound on embedding, so keeping the decoded frame
   resident removes the repeated host work.
2. **Multi-size views of one decode (#219).** Work crops at model size for the VLM and region consumers so the
   embed and region stages share one decode.
3. **Faces and OCR on resident crops (#220).** Ingest does not call them today (probe 3); on resident crops they cost
   a model pass, not a decode.
4. **ortloom head-to-head against Triton on the ingest-path models (#231)**, since the small-image runs are bound by
   the embedding model's throughput.
5. **The embed reprocess pass stops at about 58 percent GPU (#259)**, the same underuse the 20 MP ingest shows.
6. **Raise or tune the in-flight bytes gate (`OP_INGEST_INFLIGHT_MB`)** for photo-sized sets and re-run hires-100k
   with the sources on local storage, to split the NAS from the pipeline. Not run in 0.6.0.
7. The 1M-image step follows once 1 to 5 land; at 36 images/s it would be about 7.7 hours of ingest and a
   cluster pass that grows faster than linearly (269 s at 64k crops, 2,042 s at 491k).

#### Reproduce

```bash
# Public Open Images v7 train, seed 20261009, 100,000 images, CC BY 2.0 (checksum-pinned; the 39 MB pin is not in git)
python scripts/datasets/fetch_openimages_bench.py --out data/bench100k --n 100000 --seed 20261009 \
  --cache-dir data/oi-cache --manifest data/oi_bench_100000.json
python scripts/datasets/fetch_openimages_bench.py --verify-only --out data/bench100k --manifest data/oi_bench_100000.json
# pin: manifest_sha256 b9deca207878b569b7cd6e649dbd68d1b34b70db0a0fb0f5d43cc4d554da9f50
# (scripts/datasets/manifests/openimages_bench_100000.sha256)

# 10k COCO set (same command as the 2k pin, larger --bench-set)
# the driver: ingest, one forced cancel at 50 %, resume, 15 s polls, 30 s resource samples, cluster pass timing
.venv/bin/python scripts/bench/scale_run.py --api http://127.0.0.1:4603 \
  --project oi-100k --folder /data/bench100k/images --label oi-100k \
  --stop-at 0.5 --baseline-img-s 29.0 --out-dir artifacts_local/scale
# --manifest <path as the API container sees it> replaces --folder for a manifest source
```

The driver writes `<label>.jsonl` and a `<label>.md` table; `curl .../curation/projects/<p>/ingest/bulk` returns
the run record with `stage_s`. It reads the stack only (HTTP, `nvidia-smi`, `docker stats`); it needs `docker`
on the host for the resource samples.


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

Add to `requirements.txt`:
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

### MobileCLIP image encoder FP16

One RTX 3080 Ti (12 GB, no foreign load during the runs), TensorRT 11.1, dynamic batch 1/8/32 at
256 x 256, engine-only `trtexec` timings (no Triton, no preprocessing):

| Engine | Size | Build | Batch 8 GPU compute | Batch 8 images/s | Batch 32 GPU compute | Batch 32 images/s |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 146.6 MB | 62 s | 12.7 ms | 627 | 41.5 ms | 771 |
| FP16 (4 convs FP32) | 75.1 MB | 65 s | 8.3 ms | 966 | 19.9 ms | 1604 |

Parity of the FP16 engine against the FP32 engine on the first 200 images of the pinned COCO
benchmark set (hashes verified): cosine similarity min 0.99986, mean 0.99998; top-1 nearest
neighbour agrees for 98.5 percent of images and the top-5 neighbour overlap is 99.1 percent. The
engine built by `export_mobileclip_image_encoder.py` itself (71.5 MB, 72 s, no fallback) measures
cosine min 0.99984, mean 0.99998, top-1 99.5 percent, top-5 99.3 percent. Per-model Triton
`perf_analyzer` runs were not made; the engine-only figures above are the speedup bound.

## Tuning Parameters

### OCR pipeline instances

`ocr_pipeline` is a Python BLS model: an instance runs one request at a time, so its instance
count is the OCR concurrency (the detection and recognition engines need under 10 ms of GPU
time per image, the Python instance 42-47 ms). On one RTX A6000 with 640 x 480 frames and 16
clients, `/ocr/predict` does 21 images/s with one instance, 42 with two and 80 with four.

The installer sets the count per profile (`ocr_pipeline_instances` in
`config_templates/profiles/*.json`: minimal 1, standard 2, full 4) and `models/ocr_pipeline`
ships 2. A warm instance holds about 256 MiB of GPU memory (measured on a 12 GB card, 640 x 480
frames; whole-card use with the det and rec engines loaded was 11082, 11337 and 11850 MiB for
one, two and four instances, so +255 MiB per extra instance); the model config records 512 MiB per instance as the budget figure.
`python_backend_budget_mib` in each profile caps the Python instances, and an offline test
fails when a count exceeds it. Memory for very large frames was not measured. To change the
count on an installed stack, edit `count` in `models/ocr_pipeline/config.pbtxt` and restart
Triton.

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

Edit `src/app/factory.py`:
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

Set it in `.env` (the setting is `slow_request_threshold_ms` in `src/config/settings.py`;
environment names are case-insensitive):
```bash
slow_request_threshold_ms=50  # More aggressive logging
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
