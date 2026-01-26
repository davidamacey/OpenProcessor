# Face Pipeline C++ Backend

High-performance face detection + recognition pipeline implemented as a Triton C++ backend.

## Architecture

```
Input: face_images (640x640) + original_image (HD) + orig_shape
  │
  ├─► BLS: YOLO11-face detection (internal call)
  │         ↓
  ├─► GPU: Transform boxes from YOLO coords to HD coords
  │         ↓
  ├─► GPU: Expand boxes with MTCNN-style 40% margin
  │         ↓
  ├─► GPU: Bilinear crop faces from HD image
  │         ↓
  ├─► BLS: ArcFace embedding extraction (internal call)
  │         ↓
  └─► GPU: L2 normalize embeddings
           ↓
Output: num_faces, boxes, landmarks, scores, embeddings, quality
```

## Performance Results (January 2026)

### Production Endpoints (Recommended)

| Endpoint | 16 Clients | 32 Clients | 64 Clients | 128 Clients |
|----------|------------|------------|------------|-------------|
| `/track_e/faces/fast/recognize` (direct gRPC) | 67 RPS | 126 RPS | **227 RPS** | 244 RPS |
| `/track_e/faces/recognize` (Python BLS) | 65 RPS | 125 RPS | **224 RPS** | 240 RPS |
| `/track_e/faces/yolo11/recognize` (YOLO11 pipeline) | 68 RPS | 123 RPS | **204 RPS** | 250 RPS |

**Target achieved: 227+ RPS (original target was 100 RPS)**

### C++ Backend Results

| Metric | C++ Backend | Python BLS |
|--------|-------------|------------|
| Single-request latency | 100ms | 112ms |
| Throughput @ 16 clients | 12 RPS | 67 RPS |
| Scaling | **Poor** | **Excellent** |

### Root Cause of Poor C++ Scaling

The C++ BLS implementation uses synchronous blocking waits:
```cpp
// This blocks the thread until BLS completes
ctx.cv.wait(lk, [&ctx] { return ctx.done; });
```

Each model instance thread blocks during BLS calls, limiting parallelism to the number of instances. Python BLS with FastAPI workers achieves better concurrency through:
1. Multiple FastAPI workers (32 uvicorn + 64 thread pool)
2. Async request handling with uvloop
3. Better integration with Triton's dynamic batching scheduler

### Triton Model Performance (per request)

| Model | Queue | Input | Inference | Output | Total |
|-------|-------|-------|-----------|--------|-------|
| yolo11_face_small_trt_end2end | 7ms | 5ms | 5ms | 0.5ms | ~18ms |
| arcface_w600k_r50 | 10ms | 1ms | 5ms | 0.3ms | ~17ms |

Both models have 8 GPU instances for high concurrency.

## Building

```bash
# Build the Docker build image (once)
docker build -f Dockerfile.build -t face-pipeline-builder .

# Build the backend
docker run --rm -v "$(pwd)":/workspace -w /workspace face-pipeline-builder \
    bash -c "rm -rf build && mkdir build && cd build && cmake .. && make -j"

# Deploy
cp build/triton_face_pipeline.so ../../backends/face_pipeline/libtriton_face_pipeline.so
docker compose restart triton-api
```

## Files

- `src/face_pipeline.cc` - Main backend implementation with BLS calls
- `src/gpu_crop.cu` - CUDA kernels (bilinear crop, box transforms, L2 normalize)
- `CMakeLists.txt` - Build configuration
- `config.pbtxt` - Triton model configuration
- `Dockerfile.build` - Build environment with Triton headers

## Future Improvements

To achieve better scaling, the BLS calls need to be made truly async:
1. Use non-blocking BLS with completion callbacks
2. Process multiple requests per Execute call
3. Or: Implement as Triton ensemble with GPU memory sharing

## Recommendation

**Use the Python-based endpoints for production.** All three endpoints now achieve 200+ RPS:

- `/track_e/faces/fast/recognize` - Direct gRPC calls (fastest)
- `/track_e/faces/recognize` - Python BLS pipeline (most features)
- `/track_e/faces/yolo11/recognize` - YOLO11 face pipeline

The C++ backend code is archived here as reference for future async BLS implementation attempts.

## Status: ARCHIVED

This C++ backend experiment demonstrated that synchronous BLS calls don't scale well.
The production Python endpoints exceed all performance targets (227+ RPS vs 100 RPS goal).
