# Rust Preprocessing Service

High-performance preprocessing and inference service built in Rust as a drop-in replacement for the Python yolo-api service.

## Overview

This service provides the same `/ingest/batch`, `/detect`, and `/embed` endpoints as the Python service, but with:

- **True parallelism** via tokio (no GIL)
- **Single process** instead of 32 uvicorn workers
- **Zero-copy operations** with SIMD-optimized image processing
- **Lower memory footprint** (~100MB vs 32 × 200MB Python workers)

## Architecture

```
HTTP Layer (axum)
    ↓
Preprocessing (turbojpeg + fast_image_resize)
    ↓
Triton gRPC Client (tonic, connection pool)
    ↓
Post-processing + OpenSearch Indexing
```

## Endpoints

- **GET /health** - Health check
- **POST /detect** - Single image object detection
- **POST /detect/batch** - Batch object detection (up to 64 images)
- **POST /embed/image** - Single image CLIP embedding
- **POST /embed/batch** - Batch CLIP embeddings
- **POST /ingest/batch** - Full pipeline (decode, detect, embed, index)

## Building

### Docker (Recommended)

```bash
# Build the image
docker compose build rust-preprocess

# Or via Make
make rust-build
```

### Local Development

Requires:
- Rust 1.83+
- NASM (for turbojpeg SIMD)
- CMake
- libjpeg-turbo development headers
- protobuf compiler

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install -y nasm cmake libjpeg62-turbo-dev protobuf-compiler

# Check compilation
cargo check

# Build release
cargo build --release

# Run locally (requires Triton and OpenSearch)
TRITON_URL=triton-server:8001 \
OPENSEARCH_URL=http://opensearch:9200 \
PORT=8000 \
cargo run --release
```

## Usage

### Start the service

```bash
# Via Docker Compose
make rust-up

# Service available on port 4610
curl http://localhost:4610/health
```

### Quick test

```bash
# Health check
curl http://localhost:4610/health

# Detection test
curl -X POST http://localhost:4610/detect \
  -F "image=@test_images/bus.jpg" | jq

# Embedding test
curl -X POST http://localhost:4610/embed/image \
  -F "image=@test_images/bus.jpg" | jq
```

### Run benchmark

```bash
# Benchmark against Rust service (port 4610)
make rust-bench

# Compare Python vs Rust
make bench-compare
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton-server:8001` | Triton gRPC endpoint |
| `OPENSEARCH_URL` | `http://opensearch:9200` | OpenSearch REST endpoint |
| `PORT` | `8000` | HTTP server port |
| `GRPC_POOL_SIZE` | `4` | Number of gRPC channels to Triton |
| `RUST_LOG` | `rust_preprocess=info` | Log level (tracing crate) |

## Performance

Expected improvements over Python:
- **2-3x lower latency** for single-image requests
- **Same or better throughput** for batch requests (limited by GPU)
- **90% lower memory usage** (single process vs 32 workers)
- **Faster startup time** (no Python import overhead)

## Implementation Status

✅ **Implemented:**
- JPEG/PNG/WebP image decoding (turbojpeg + image crate)
- YOLO letterbox preprocessing (exact Python match)
- CLIP center-crop preprocessing (exact Python match)
- Triton gRPC client with connection pooling and retry
- YOLO End2End + MobileCLIP inference
- OpenSearch bulk indexing (global, vehicles, people)
- imohash computation for duplicate detection
- `/detect`, `/detect/batch`, `/embed/image`, `/embed/batch`, `/ingest/batch`

❌ **Not Yet Implemented:**
- Face detection/recognition (SCRFD + ArcFace)
- OCR (PP-OCRv5)
- Near-duplicate detection (partially implemented)
- Per-detection box embeddings for vehicles/people indexing

## Project Structure

```
rust-preprocess/
├── Cargo.toml              # Dependencies
├── build.rs                # Proto compilation
├── Dockerfile              # Multi-stage build
├── proto/                  # Triton gRPC protos
│   ├── grpc_service.proto
│   └── model_config.proto
└── src/
    ├── main.rs             # Server setup
    ├── config.rs           # Env config
    ├── error.rs            # Error types
    ├── hash.rs             # imohash algorithm
    ├── preprocess/
    │   ├── decode.rs       # JPEG/PNG decode
    │   ├── yolo.rs         # Letterbox 640x640
    │   └── clip.rs         # Center-crop 256x256
    ├── triton/
    │   ├── client.rs       # gRPC pool + inference
    │   └── proto.rs        # Proto re-exports
    ├── opensearch/
    │   ├── client.rs       # REST client
    │   ├── bulk.rs         # Bulk operations
    │   └── duplicate.rs    # Near-duplicate detection
    ├── handlers/
    │   ├── detect.rs       # /detect endpoints
    │   ├── embed.rs        # /embed endpoints
    │   ├── ingest.rs       # /ingest/batch
    │   └── health.rs       # /health
    └── postprocess/
        ├── detection.rs    # Inverse letterbox transform
        └── coco_classes.rs # 80-class COCO names
```

## Development

View logs:
```bash
make rust-logs
```

Restart service:
```bash
make rust-restart
```

Stop service:
```bash
make rust-down
```

## License

Same as parent project (see main repository LICENSE).
