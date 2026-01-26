# Documentation Index

Technical documentation for the Visual AI API.

---

## Quick Links

- **[Main README](../README.md)** - Project overview, API endpoints, quick start
- **[Benchmarks Guide](../benchmarks/README.md)** - Performance testing with triton_bench
- **[Model Export](../export/README.md)** - TensorRT model export documentation
- **[Attribution](../ATTRIBUTION.md)** - Third-party code attribution and licensing

---

## API Reference

The API provides these endpoint groups (all on port 4603):

| Prefix | Description | Endpoints |
|--------|-------------|-----------|
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

## Technical Documentation

### Model Documentation

| Document | Description |
|----------|-------------|
| [OCR_SETUP_GUIDE.md](OCR_SETUP_GUIDE.md) | PP-OCRv5 deployment guide |
| [OCR_DEPLOYMENT_CHECKLIST.md](OCR_DEPLOYMENT_CHECKLIST.md) | Step-by-step OCR setup |
| [OCR_IMPLEMENTATION_PLAN.md](OCR_IMPLEMENTATION_PLAN.md) | OCR architecture details |
| [FACE_RECOGNITION_IMPLEMENTATION.md](FACE_RECOGNITION_IMPLEMENTATION.md) | Face detection and ArcFace setup |

### Vector Search

| Document | Description |
|----------|-------------|
| [opensearch_schema_design.md](opensearch_schema_design.md) | FAISS IVF clustering and OpenSearch schema |

### Performance

| Document | Description |
|----------|-------------|
| [GRPC_CONNECTION_SCALING.md](GRPC_CONNECTION_SCALING.md) | gRPC connection pooling |
| [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) | FastAPI performance tuning |
| [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) | Quick optimization reference |
| [INGEST_BENCHMARK_METHODOLOGY.md](INGEST_BENCHMARK_METHODOLOGY.md) | Ingestion benchmarking |

### Architecture

| Document | Description |
|----------|-------------|
| [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md) | Production deployment patterns |
| [THREAD_SAFETY_FIX.md](THREAD_SAFETY_FIX.md) | Thread safety troubleshooting |

### Research

| Document | Description |
|----------|-------------|
| [PADDING_COMPARISON_GUIDE.md](PADDING_COMPARISON_GUIDE.md) | YOLO preprocessing comparison |

### Planning

| Document | Description |
|----------|-------------|
| [FEATURE_ROADMAP.md](FEATURE_ROADMAP.md) | Feature status and future plans |

---

## Project Structure

```
triton-api/
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

### Run Tests

```bash
# Comprehensive test suite (32 tests)
source .venv/bin/activate
python tests/test_full_system.py

# Visual validation with bounding boxes
python tests/validate_visual_results.py

# View annotated test images
ls test_results/*.jpg
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
make export-face-models      # YOLO11-face + ArcFace
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

---

**Last Updated:** January 2026
