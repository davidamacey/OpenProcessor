# Benchmarks

This directory contains benchmarking tools and results for the Triton API.

## Directory Structure

```
benchmarks/
├── README.md           # This file
├── results/            # Benchmark result files (timestamped)
└── scripts/            # Benchmark scripts
    ├── benchmark.py    # Python httpx-based benchmark
    └── detect.lua      # wrk script for /detect endpoint
```

## Running Benchmarks

### Quick Latency Test

Test individual endpoint latency with curl:

```bash
# Single request timing
make bench-quick

# Or manually:
curl -s -w "Time: %{time_total}s\n" -o /dev/null \
    -X POST http://localhost:4603/detect \
    -F "image=@test_images/sample.jpg"
```

### Apache Bench (ab)

Simple load testing with Apache Bench:

```bash
# Install ab
apt install apache2-utils

# Benchmark detection (1000 requests, 32 concurrent)
ab -n 1000 -c 32 -p test_images/sample.jpg -T "image/jpeg" \
    http://localhost:4603/detect

# Benchmark face recognition
ab -n 1000 -c 32 -p test_images/sample.jpg -T "image/jpeg" \
    http://localhost:4603/faces/recognize

# Benchmark embedding
ab -n 1000 -c 32 -p test_images/sample.jpg -T "image/jpeg" \
    http://localhost:4603/embed/image
```

### wrk (High-Performance HTTP Benchmark)

For more sophisticated load testing with wrk:

```bash
# Install wrk
apt install wrk

# Benchmark with custom Lua script
wrk -t4 -c64 -d30s -s benchmarks/scripts/detect.lua \
    http://localhost:4603/detect
```

### Python httpx Benchmark

For custom benchmarks with detailed metrics:

```bash
# Run inside container
docker compose exec yolo-api python /app/benchmarks/scripts/benchmark.py

# Or use make target
make bench-python
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make bench-quick` | Quick latency test of all endpoints |
| `make bench-detect` | Benchmark /detect with wrk |
| `make bench-faces` | Benchmark /faces/recognize with ab |
| `make bench-embed` | Benchmark /embed/image with ab |
| `make bench-ingest` | Benchmark /ingest with ab |
| `make bench-search` | Benchmark /search/image with ab |
| `make bench-python` | Run Python benchmark script |
| `make bench-results` | Show recent benchmark results |

## Configuration

Override benchmark parameters with environment variables:

```bash
# Longer duration
BENCH_DURATION=60s make bench-detect

# More concurrent clients
BENCH_CLIENTS=64 make bench-faces

# More total requests
BENCH_REQUESTS=5000 make bench-ingest

# Different test image
TEST_IMAGE=test_images/faces/sample.jpg make bench-faces
```

## Example Results

Typical throughput on RTX A6000:

| Endpoint | Latency (p50) | Throughput |
|----------|---------------|------------|
| /detect | ~15ms | ~60 RPS |
| /faces/recognize | ~25ms | ~40 RPS |
| /embed/image | ~10ms | ~100 RPS |
| /ingest | ~50ms | ~20 RPS |
| /search/image | ~30ms | ~35 RPS |

Results vary based on image size, batch size, and GPU utilization.

## Saving Results

Benchmark results can be saved to timestamped files:

```bash
# Save ab output
ab -n 1000 -c 32 -p test_images/sample.jpg -T "image/jpeg" \
    http://localhost:4603/detect > benchmarks/results/detect_$(date +%Y%m%d_%H%M%S).txt

# Save wrk output
wrk -t4 -c64 -d30s http://localhost:4603/detect \
    > benchmarks/results/wrk_$(date +%Y%m%d_%H%M%S).txt
```
