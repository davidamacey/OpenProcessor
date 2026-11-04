# Benchmarks

This directory contains benchmarking tools and results for OpenProcessor. The `./openprocessor bench` subcommand is not available yet because `triton_bench` and `benchmarks/build.sh` are not in the repository; use the tools below. This page is the tool list; method and measurements are in [docs/PERFORMANCE.md](../docs/PERFORMANCE.md#benchmarking).

## Directory Structure

```
benchmarks/
├── README.md                   # This file
├── go.mod                      # Module for the two Go ingest benchmarks
├── ingest_benchmark.go         # Parallel single-image ingest (build tag `single`)
├── ingest_benchmark_batch.go   # Parallel batch ingest to /ingest/batch (build tag `batch`)
├── results/                    # Benchmark result files (timestamped)
└── scripts/
    ├── benchmark.py            # Python httpx-based benchmark
    └── detect.lua              # wrk script for /detect endpoint
```

### Go ingest benchmarks

```bash
cd benchmarks
go run -tags single ingest_benchmark.go -dir /path/to/images -workers 64 -clear
go run -tags batch ingest_benchmark_batch.go -dir /path/to/images -workers 16 -batch 32 -clear
```

The `-dir` argument is a host folder of images; `-clear` empties the visual-search indexes first.

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
docker compose exec api python /app/benchmarks/scripts/benchmark.py

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

## Results

Measured numbers, with the hardware and method for each, are in [docs/PERFORMANCE.md](../docs/PERFORMANCE.md). Results vary with image size, batch size and GPU utilization.

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
