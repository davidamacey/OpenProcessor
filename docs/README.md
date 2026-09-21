# Documentation Index

Technical documentation for the Visual AI API.

---

## Quick Links

- **[Main README](../README.md)** - Project overview, API endpoints, quick start
- **[CLAUDE.md](../CLAUDE.md)** - Project instructions for AI assistants
- **[CURATION.md](CURATION.md)** - Curation & active-learning subsystem user guide (experimental)
- **[Benchmarks Guide](../benchmarks/README.md)** - Performance testing with triton_bench
- **[Model Export](../export/README.md)** - TensorRT model export documentation
- **[Attribution](../ATTRIBUTION.md)** - Third-party code attribution and licensing
- **[SECURITY.md](../SECURITY.md)** - Security policy — no authentication, do not expose to the internet
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Dev setup, test suites, commit conventions

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

### Curation / Labeling (experimental)

| Document | Description |
|----------|-------------|
| [CURATION.md](CURATION.md) | User guide — what it is, required models, class-registry schema, workers, seed path, known gaps |
| [design/curation_design_rationale.md](design/curation_design_rationale.md) | Design rationale — the four config dataclasses, frozen wire contract, pre-commit ratchet |
| [design/curation_api_contract.md](design/curation_api_contract.md) | `/curation` HTTP wire contract — Pydantic model field names, frozen vs. configurable, capability discovery (`/methods`) |

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
| `/curation` | Curation + active-learning labeling subsystem | classes, crops, regions, clusters, review, scores, select, VLM labeling, training, export, pipeline — see [curation_api_contract.md](design/curation_api_contract.md) |

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
│   │   ├── health.py         # /health endpoints
│   │   └── curation/         # /curation endpoints (classes, crops, regions,
│   │                         #   clusters, review, scores, select, vlm, ...)
│   ├── services/             # Business logic
│   │   ├── curation/         # Curation subsystem services (clustering, scoring,
│   │   │                     #   selection, event hub, semantic search, export)
│   │   ├── detection/         # Detection cascade primitives
│   │   ├── labeling/          # VLM client/labeler/prompts
│   │   └── training/          # Training pipeline (jobs, profiles, promote)
│   ├── clients/              # Triton and OpenSearch clients
│   └── schemas/              # Pydantic models
│
├── scripts/curation/         # Curation worker entry points (vlm_worker.py,
│                             #   auto_label_worker.py, cluster_refresh_daemon.py,
│                             #   detection worker package)
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
# Full pytest suite (1000+ tests, offline — this is what CI runs)
make test
# equivalent to: .venv/bin/python -m pytest tests/ -q

# Endpoint integration tests (auto-downloads test images)
make test-endpoints

# Comprehensive smoke-test script
.venv/bin/python tests/test_full_system.py

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

**Last Updated:** 2026-09-21
**Version:** 0.3.0 - Curation subsystem, CI, and OSS furniture added
