# Documentation Index

Technical documentation for the Visual AI API.

---

## Quick Links

- **[Main README](../README.md)** - Project overview, API endpoints, quick start
- **[CLAUDE.md](../CLAUDE.md)** - Project instructions for AI assistants
- **[Benchmarks Guide](../benchmarks/README.md)** - Performance testing with triton_bench
- **[Model Export](../export/README.md)** - TensorRT model export documentation
- **[Attribution](../ATTRIBUTION.md)** - Third-party code attribution and licensing

---

## Core Documentation

### System Architecture

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, production patterns, thread safety, scaling strategies |

### Capabilities

| Document | Description |
|----------|-------------|
| [OCR.md](OCR.md) | PP-OCRv5 text detection and recognition - setup, deployment, usage |
| [FACE_RECOGNITION_IMPLEMENTATION.md](FACE_RECOGNITION_IMPLEMENTATION.md) | SCRFD face detection and ArcFace embeddings |

### Performance

| Document | Description |
|----------|-------------|
| [PERFORMANCE.md](PERFORMANCE.md) | FastAPI optimizations, gRPC connection management, benchmarking, profiling |

### Vector Search

| Document | Description |
|----------|-------------|
| [opensearch_schema_design.md](opensearch_schema_design.md) | FAISS IVF clustering and OpenSearch index design |

---

## API Reference

The API provides these endpoint groups (all on port 4603):

| Prefix | Description | Key Endpoints |
|--------|-------------|---------------|
| `/detect` | YOLO object detection | Single, batch |
| `/faces` | Face detection and recognition | detect, recognize, verify, search, identify |
| `/embed` | CLIP embeddings | image, text, batch, boxes |
| `/search` | Visual similarity search | image, text, face, ocr, object |
| `/ingest` | Data ingestion | single, batch, directory |
| `/ocr` | Text extraction | predict, batch |
| `/analyze` | Combined analysis | All models in one call |
| `/clusters` | FAISS clustering | train, stats, albums |
| `/query` | Data retrieval | image, stats, duplicates |
| `/health` | Monitoring | Service health, model status |

---

## Project Structure

```
OpenProcessor/
├── README.md                 # Main project documentation
├── CLAUDE.md                 # AI assistant instructions
├── ATTRIBUTION.md            # Third-party code attribution
├── Makefile                  # Development commands
├── docker-compose.yml        # Services orchestration
│
├── src/                      # FastAPI service
│   ├── main.py               # Application entry point
│   ├── routers/              # API endpoints
│   │   ├── detect.py         # /detect endpoints
│   │   ├── faces.py          # /faces endpoints
│   │   ├── embed.py          # /embed endpoints
│   │   ├── search.py         # /search endpoints
│   │   ├── ingest.py         # /ingest endpoints
│   │   ├── ocr.py            # /ocr endpoints
│   │   ├── analyze.py        # /analyze endpoints
│   │   ├── clusters.py       # /clusters endpoints
│   │   ├── query.py          # /query endpoints
│   │   └── health.py         # /health endpoints
│   ├── services/             # Business logic
│   ├── clients/              # Triton and OpenSearch clients
│   └── schemas/              # Pydantic models
│
├── export/                   # Model export scripts
├── models/                   # Triton model repository
├── benchmarks/               # Performance testing
├── docs/                     # This directory
└── monitoring/               # Prometheus and Grafana
```

---

## Common Tasks

### Start Services

```bash
docker compose up -d
curl http://localhost:4603/health
```

### Download Test Images

```bash
# Auto-downloads bus.jpg and zidane.jpg from Ultralytics
make download-test-images
```

### Run Tests

```bash
# Endpoint integration tests (auto-downloads test images)
make test-endpoints

# Comprehensive test suite
source .venv/bin/activate
python tests/test_full_system.py

# Individual endpoint tests
make test-faces           # Face detection + recognition
make test-detect          # Object detection
make test-embed           # CLIP embeddings
make test-ocr             # Text extraction
```

### Run Benchmarks

```bash
cd benchmarks
./build.sh
./triton_bench --mode quick
```

### Export Models

```bash
make export-models           # YOLO TensorRT
make export-mobileclip       # MobileCLIP encoders
make setup-face-pipeline     # SCRFD + ArcFace
make setup-ocr               # PP-OCRv5 models
```

### Check Model Status

```bash
curl -s http://localhost:4600/v2/models | jq '.models[] | {name, state}'
```

---

## External Resources

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Ultralytics YOLO](https://docs.ultralytics.com/)

---

**Last Updated:** 2026-01-27
**Version:** 2.1 - Updated for SCRFD face pipeline and test data setup
