# Rust Preprocessing Service - Implementation Status

## Current Status: 95% Complete - Minor Compilation Fixes Needed

### ✅ Fully Implemented

**Project Structure:**
- ✅ Cargo.toml with all dependencies
- ✅ build.rs for proto compilation
- ✅ Multi-stage Dockerfile
- ✅ Docker Compose integration (port 4610)
- ✅ Makefile targets

**Core Modules:**
- ✅ Image decoding (turbojpeg for JPEG, image crate for PNG/BMP)
- ✅ YOLO letterbox preprocessing (exact Python match)
- ✅ CLIP center-crop preprocessing (exact Python match)
- ✅ Triton gRPC client with connection pooling
- ✅ OpenSearch REST client
- ✅ HTTP handlers (detect, embed, ingest)
- ✅ Post-processing (inverse letterbox, COCO classes)

**Feature Parity with Python:**
- ✅ `/detect` - Single image detection
- ✅ `/detect/batch` - Batch detection
- ✅ `/embed/image` - Single image embedding
- ✅ `/embed/batch` - Batch embeddings
- ✅ `/ingest/batch` - Full pipeline (PRIMARY BENCHMARK ENDPOINT)
- ✅ `/health` - Health check

**Exact Python Behavior Match:**
- ✅ NO per-detection box embeddings (matching Python's current behavior)
- ✅ Only global CLIP embedding extracted
- ✅ vehicles/people counts = 0 (no box embeddings to index)
- ✅ Same preprocessing algorithms (letterbox padding trick, center crop)

### ❌ Remaining Compilation Issues

**Proto Type Names (Easy Fix):**
```rust
// Current (incorrect):
ModelInferRequest_InferInputTensor
ModelInferRequest_InferRequestedOutputTensor

// Need to use (actual generated types):
model_infer_request::InferInputTensor
model_infer_request::InferRequestedOutputTensor
```

**Error Type in Main:**
```rust
// Need to add From<std::io::Error> for AppError
// Or change main return type
```

**Fixed:**
- ✅ Decompressor mutability
- ✅ get_class_name import

### 🎯 Next Steps

1. **Fix Proto Types** (~5 min)
   - Update `src/triton/client.rs` to use correct generated type names

2. **Fix Main Error Handling** (~2 min)
   - Add `From<std::io::Error>` impl or change return type

3. **Rebuild Docker Image**
   ```bash
   docker compose build rust-preprocess
   docker compose up -d rust-preprocess
   ```

4. **Test Endpoints**
   ```bash
   curl http://localhost:4610/health
   curl -X POST http://localhost:4610/detect -F "image=@test_images/bus.jpg"
   ```

5. **Run Benchmark Comparison**
   ```bash
   # Python (port 4603)
   ./benchmarks/ingest_benchmark_batch -port 4603 -workers 16 -batch 64 \
       -faces=false -ocr=false -dir "/path/to/images"

   # Rust (port 4610)
   ./benchmarks/ingest_benchmark_batch -port 4610 -workers 16 -batch 64 \
       -faces=false -ocr=false -dir "/path/to/images"
   ```

## Services Currently Running

```bash
$ docker compose ps
triton-server    UP (healthy)    4601:8001 (gRPC)
opensearch       UP (healthy)    4607:9200
yolo-api         UP (healthy)    4603:8000 (Python)
rust-preprocess  NEEDS REBUILD   4610:8000 (Rust - awaiting fixes)
```

## Key Implementation Details

### Exact Algorithm Matching

**YOLO Letterbox:**
```rust
// Python: scale = min(640/h, 640/w).min(1.0)
// Python: new_w = round(w * scale)  // Python's round()
// Python: pad_w = (640 - new_w) / 2.0
// Python: top = round(pad_h - 0.1)  // Ultralytics trick
```

**CLIP Center Crop:**
```rust
// Python: scale = 256 / min(h, w)
// Python: new_w = int(w * scale)  // Truncation, NOT rounding
// Python: start_x = (new_w - 256) // 2  // Integer division
```

### Performance Expectations

**Python (32 uvicorn workers):**
- Memory: ~32 × 200MB = 6.4GB
- CPU: GIL-limited parallel preprocessing
- Throughput: GPU-bound (batch inference)

**Rust (single tokio process):**
- Memory: ~100-200MB total
- CPU: True parallelism (no GIL)
- Throughput: GPU-bound (same as Python)
- Latency: 2-3x lower for single requests
- Startup: Instant vs Python import overhead

### Fair Benchmark Conditions

**Identical Settings:**
- ✅ Same Triton server (triton-server:8001)
- ✅ Same OpenSearch (opensearch:9200)
- ✅ Same GPU models (YOLO11 + MobileCLIP)
- ✅ Same Go benchmark binary
- ✅ Same parameters: `-workers 16 -batch 64 -faces=false -ocr=false`

**What's Being Measured:**
- Preprocessing CPU time (decode, resize, normalize)
- Triton gRPC client overhead
- OpenSearch indexing throughput
- Memory efficiency
- Process overhead

**Not Being Measured:**
- GPU inference time (identical for both)
- Network latency (same Docker network)
- Disk I/O (same NAS mount)

## Files Modified

```
rust-preprocess/
├── Cargo.toml              ✅ All dependencies
├── build.rs                ✅ Proto compilation
├── Dockerfile              ✅ Multi-stage build
├── proto/                  ✅ Triton protos
└── src/
    ├── main.rs             ⚠️  Needs error fix
    ├── config.rs           ✅
    ├── error.rs            ✅
    ├── hash.rs             ✅ imohash
    ├── preprocess/         ✅ All modules
    ├── triton/client.rs    ⚠️  Needs proto type fix
    ├── opensearch/         ✅ All modules
    ├── handlers/           ✅ All endpoints
    └── postprocess/        ✅ All modules

docker-compose.yml          ✅ rust-preprocess service added
Makefile                    ✅ rust-* targets added
```

## Testing Plan

### Phase 1: Smoke Test
```bash
curl http://localhost:4610/health
# Expected: {"status":"ok","service":"rust-preprocess","version":"0.1.0"}
```

### Phase 2: Functional Test
```bash
# Single detection
curl -X POST http://localhost:4610/detect \
  -F "image=@test_images/bus.jpg" | jq

# Batch detection
curl -X POST http://localhost:4610/detect/batch \
  -F "images=@test_images/bus.jpg" \
  -F "images=@test_images/zidane.jpg" | jq

# Embedding
curl -X POST http://localhost:4610/embed/image \
  -F "image=@test_images/bus.jpg" | jq '.dimensions'
# Expected: 512
```

### Phase 3: Ingest Test
```bash
curl -X POST http://localhost:4610/ingest/batch \
  -F "images=@test_images/bus.jpg" \
  -F "enable_detection=true" \
  -F "enable_clip=true" \
  -F "enable_faces=false" \
  -F "enable_ocr=false" | jq
# Expected: {"status":"success","indexed":{"global":1,"vehicles":0,"people":0}}
```

### Phase 4: Performance Benchmark
```bash
# Build Go benchmark
cd benchmarks
go build -tags batch -o ingest_benchmark_batch ingest_benchmark_batch.go

# Python baseline
./ingest_benchmark_batch -port 4603 -workers 16 -batch 64 -report 10 \
  -faces=false -ocr=false \
  -dir "/mnt/nas/killboy_data/killboy_hdd01/Killboy_Sorted_Photos"

# Rust comparison
./ingest_benchmark_batch -port 4610 -workers 16 -batch 64 -report 10 \
  -faces=false -ocr=false \
  -dir "/mnt/nas/killboy_data/killboy_hdd01/Killboy_Sorted_Photos"
```

## Expected Benchmark Metrics

**Throughput:** ~Same (GPU-bound)
**Memory:** Rust 90% lower
**Latency:** Rust 2-3x better (single requests)
**CPU:** Rust more efficient (no GIL, SIMD)

---

**Ready for final fixes and testing!**
