# Rust vs Python Preprocessing Service Benchmark Guide

## 🎯 Current Status: ALL SERVICES RUNNING ✅

```bash
$ docker compose ps
SERVICE           STATUS                    PORTS
yolo-api          Up (healthy)             0.0.0.0:4603->8000/tcp  (Python)
rust-preprocess   Up                       0.0.0.0:4610->8000/tcp  (Rust)
triton-server     Up (healthy)             0.0.0.0:4601->8001/tcp  (gRPC)
opensearch        Up (healthy)             0.0.0.0:4607->9200/tcp  (REST)
```

## 📋 Services Verified

### Python Service (Port 4603)
```bash
$ curl http://localhost:4603/health
{"status":"ok","version":"0.1.0","api_version":"v1"}
```

### Rust Service (Port 4610)
```bash
$ curl http://localhost:4610/health
{"status":"ok","service":"rust-preprocess","version":"0.1.0"}
```

## 🔬 Benchmark Setup

### Prerequisites

1. **Build Go Benchmark Tool:**
```bash
cd /mnt/nvm/repos/triton-api/benchmarks
go build -tags batch -o ingest_benchmark_batch ingest_benchmark_batch.go
cd ..
```

2. **Verify Services:**
```bash
# All should return 200
curl -s http://localhost:4603/health | jq '.status'  # "ok"
curl -s http://localhost:4610/health | jq '.status'  # "ok"
```

## 🚀 Running Benchmarks

### Test 1: Python Baseline (Port 4603)

```bash
./benchmarks/ingest_benchmark_batch \
    -dir "/mnt/nas/killboy_data/killboy_hdd01/Killboy_Sorted_Photos" \
    -workers 16 \
    -batch 64 \
    -queue 1024 \
    -report 10 \
    -faces=false \
    -ocr=false \
    -port 4603 \
    2>&1 | tee benchmark_python.log
```

**Expected Output:**
- Throughput: ~XXX RPS
- Memory: ~6-8GB (32 workers × 200MB each)
- Processing time per batch
- Success rate

### Test 2: Rust Comparison (Port 4610)

```bash
./benchmarks/ingest_benchmark_batch \
    -dir "/mnt/nas/killboy_data/killboy_hdd01/Killboy_Sorted_Photos" \
    -workers 16 \
    -batch 64 \
    -queue 1024 \
    -report 10 \
    -faces=false \
    -ocr=false \
    -port 4610 \
    2>&1 | tee benchmark_rust.log
```

**Expected Output:**
- Throughput: ~Similar (GPU-bound)
- Memory: ~100-200MB (single process)
- Processing time per batch
- Success rate

## 📊 What's Being Measured

### Fair Comparison (Identical Settings)
- ✅ Same Triton server (triton-server:8001)
- ✅ Same GPU models (YOLO11 + MobileCLIP)
- ✅ Same OpenSearch instance (opensearch:9200)
- ✅ Same benchmark parameters (-workers 16 -batch 64)
- ✅ Same feature flags (-faces=false -ocr=false)
- ✅ Same preprocessing algorithms (letterbox, center-crop)

### Performance Metrics
1. **Throughput** (RPS) - GPU-bound, should be ~similar
2. **Memory Usage** - Rust expected 90% lower
3. **CPU Efficiency** - Rust expected better (no GIL, SIMD)
4. **Latency** - Single-request latency (Rust expected 2-3x better)
5. **Process Overhead** - Rust: 1 process vs Python: 32 workers

### Pipeline Components
```
HTTP Request → Image Decode (JPEG) → Letterbox 640x640 → YOLO Inference (GPU)
                                   ↘ Center Crop 256x256 → CLIP Inference (GPU)
                                                          ↘ OpenSearch Bulk Index
```

**CPU-Bound (Where Rust Wins):**
- JPEG decoding (turbojpeg SIMD)
- Image resizing (fast_image_resize SIMD)
- Memory allocation/deallocation
- HTTP parsing
- JSON serialization

**GPU-Bound (Same Performance):**
- YOLO inference (TensorRT)
- CLIP inference (TensorRT)

## 🔍 Current Implementation Details

### Feature Parity

Both services implement **exactly the same pipeline**:

**✅ Enabled:**
- Global CLIP embedding (512-dim per image)
- YOLO object detection (80 COCO classes)
- OpenSearch `visual_search_global` indexing
- imohash duplicate detection

**❌ Disabled (for fair comparison):**
- Per-detection box embeddings (Python doesn't extract them either)
- Face detection (enable_faces=false)
- OCR (enable_ocr=false)
- `visual_search_vehicles` / `visual_search_people` counts remain 0

### Response Schema Match

Both return identical JSON:
```json
{
  "status": "success",
  "total": 64,
  "processed": 64,
  "duplicates": 0,
  "errors_count": 0,
  "indexed": {
    "global": 64,
    "vehicles": 0,
    "people": 0,
    "faces": 0,
    "ocr": 0
  },
  "near_duplicates": 0,
  "total_time_ms": 150.5
}
```

## 🐛 Troubleshooting

### Service Not Responding

```bash
# Check service logs
docker compose logs --tail 100 yolo-api
docker compose logs --tail 100 rust-preprocess

# Restart services
docker compose restart yolo-api
docker compose restart rust-preprocess
```

### Triton Connection Issues

```bash
# Check Triton health
curl http://localhost:4600/v2/health/ready

# Check models loaded
curl http://localhost:4600/v2/models/yolov11_small_trt_end2end/ready
curl http://localhost:4600/v2/models/mobileclip2_s2_image_encoder/ready
```

### OpenSearch Issues

```bash
# Check OpenSearch health
curl http://localhost:4607/_cluster/health | jq

# Check indexes
curl http://localhost:4607/_cat/indices?v
```

## 📈 Expected Results

### Hypothesis

**Throughput:** ~Equal (both GPU-bound)
- Bottleneck: GPU inference time
- Triton dynamic batching handles both equally

**Memory:** Rust 90% lower
- Python: 32 workers × 200MB = ~6.4GB
- Rust: Single process ~100-200MB

**Latency:** Rust 2-3x better (single requests)
- No GIL contention
- No worker process scheduling
- Direct tokio async/await

**CPU Efficiency:** Rust better
- SIMD-optimized preprocessing
- Zero-copy operations
- No Python interpreter overhead

## 📝 Benchmark Checklist

- [ ] All services healthy (`docker compose ps`)
- [ ] Go benchmark built (`ls -lh benchmarks/ingest_benchmark_batch`)
- [ ] Test image directory accessible
- [ ] Python baseline run completed
- [ ] Rust comparison run completed
- [ ] Logs captured for both runs
- [ ] Memory stats collected (`docker stats` during runs)
- [ ] Results compared

## 🎬 Quick Start

```bash
# 1. Verify all services running
docker compose ps

# 2. Build benchmark
cd benchmarks && go build -tags batch -o ingest_benchmark_batch ingest_benchmark_batch.go && cd ..

# 3. Run Python baseline
./benchmarks/ingest_benchmark_batch -port 4603 -workers 16 -batch 64 \
    -faces=false -ocr=false \
    -dir "/mnt/nas/killboy_data/killboy_hdd01/Killboy_Sorted_Photos" \
    | tee benchmark_python.log

# 4. Run Rust comparison
./benchmarks/ingest_benchmark_batch -port 4610 -workers 16 -batch 64 \
    -faces=false -ocr=false \
    -dir "/mnt/nas/killboy_data/killboy_hdd01/Killboy_Sorted_Photos" \
    | tee benchmark_rust.log

# 5. Compare results
echo "=== Python ===" && tail -20 benchmark_python.log
echo "=== Rust ===" && tail -20 benchmark_rust.log
```

## 🔗 Additional Resources

- **Rust Implementation:** `/mnt/nvm/repos/triton-api/rust-preprocess/`
- **Python Implementation:** `/mnt/nvm/repos/triton-api/src/`
- **Status Document:** `/mnt/nvm/repos/triton-api/rust-preprocess/STATUS.md`
- **Go Benchmark:** `/mnt/nvm/repos/triton-api/benchmarks/ingest_benchmark_batch.go`

---

**Ready to benchmark!** Both services are running and waiting for your test. 🚀
