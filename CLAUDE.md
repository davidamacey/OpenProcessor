# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Python Environment

**ALWAYS use the project's virtual environment for all Python operations:**

```bash
# Activate the venv first
source .venv/bin/activate

# Then run Python commands
python script.py
pytest tests/
pre-commit run --all-files
```

**When running Python scripts or commands:**
- ✅ CORRECT: `source .venv/bin/activate && python tests/test_full_system.py`
- ✅ CORRECT: `source .venv/bin/activate && pre-commit run --all-files`
- ❌ WRONG: `python tests/test_full_system.py` (uses system Python)
- ❌ WRONG: `python3 tests/test_full_system.py` (uses system Python3)

The `.venv` directory contains all required dependencies including:
- FastAPI, uvicorn, pydantic
- pre-commit, ruff, mypy, bandit
- pytest, requests
- All ML libraries (torch, transformers, opencv, etc.)

## Project Overview

High-performance visual AI API built on FastAPI and NVIDIA Triton Inference Server. The system provides a unified, capability-based REST API for computer vision tasks.

**Core Capabilities:**
- **Object Detection**: YOLO11 for 80-class COCO detection
- **Face Recognition**: YOLO11-face detection + ArcFace embeddings (512-dim)
- **CLIP Embeddings**: MobileCLIP for image/text embeddings (512-dim)
- **OCR**: PP-OCRv5 for text detection and recognition
- **Visual Search**: OpenSearch k-NN for similarity search across all embedding types

**Key Features:**
- Single service architecture on port 4603
- TensorRT-accelerated inference with GPU NMS
- Batch processing support (up to 64 images per request)
- Vector search with configurable indexes
- Face clustering and identification

## Architecture

### Services

The system uses Docker Compose to orchestrate three core services:

1. **triton-api**: NVIDIA Triton Inference Server
   - GPU inference backend (device_ids: [`0`, `2`])
   - Ports: 4600 (HTTP), 4601 (gRPC), 4602 (metrics)
   - Serves TensorRT models with dynamic batching
   - Max batch size: 128

2. **yolo-api**: FastAPI Service
   - Python 3.12 with async support
   - Port: **4603** (all API endpoints)
   - Workers: 2 (dev) or 64 (production)
   - Located in [src/main.py](src/main.py)

3. **opensearch**: Vector Database
   - OpenSearch 3.0+ with k-NN plugin
   - Port: **4607** (REST API)
   - Indexes: images, faces, objects, ocr

### Core Models

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| `yolov11_small_trt_end2end` | Object detection | 640x640 RGB | Boxes + classes |
| `yolo11_face_small_trt_end2end` | Face detection | 640x640 RGB | Face boxes + landmarks |
| `arcface_w600k_r50` | Face embedding | 112x112 RGB | 512-dim vector |
| `mobileclip2_s2_image_encoder` | Image embedding | 256x256 RGB | 512-dim vector |
| `mobileclip2_s2_text_encoder` | Text embedding | Token IDs | 512-dim vector |
| `paddleocr_det_trt` | Text detection | Variable RGB | Text boxes |
| `paddleocr_rec_trt` | Text recognition | Text crops | Characters |
| `yolo11_face_pipeline` | Face ensemble | 640x640 RGB | Faces + embeddings |

### Model Directory Structure

```
models/
├── yolov11_small_trt_end2end/
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolo11_face_small_trt_end2end/
│   ├── 1/model.plan
│   └── config.pbtxt
├── arcface_w600k_r50/
│   ├── 1/model.plan
│   └── config.pbtxt
├── mobileclip2_s2_image_encoder/
│   ├── 1/model.plan
│   └── config.pbtxt
├── mobileclip2_s2_text_encoder/
│   ├── 1/model.plan
│   └── config.pbtxt
├── paddleocr_det_trt/
│   ├── 1/model.plan
│   └── config.pbtxt
├── paddleocr_rec_trt/
│   ├── 1/model.plan
│   └── config.pbtxt
└── yolo11_face_pipeline/
    └── config.pbtxt
```

## API Endpoints

All endpoints available on port **4603**.

### /detect - Object Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Detect objects in single image |
| POST | `/detect/batch` | Batch detection (up to 64 images) |

### /faces - Face Detection and Recognition

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/faces/detect` | Detect faces with landmarks |
| POST | `/faces/recognize` | Detect faces + extract embeddings |
| POST | `/faces/verify` | Compare two face images (1:1) |
| POST | `/faces/search` | Search for similar faces in index |
| POST | `/faces/identify` | Identify face against known identities (1:N) |

### /embed - CLIP Embeddings

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/embed/image` | Generate image embedding |
| POST | `/embed/text` | Generate text embedding |
| POST | `/embed/batch` | Batch image embeddings |
| POST | `/embed/boxes` | Embeddings for cropped regions |

### /search - Visual Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search/image` | Find similar images by image |
| POST | `/search/text` | Find images by text description |
| POST | `/search/face` | Find images containing similar faces |
| POST | `/search/ocr` | Find images by text content |
| POST | `/search/object` | Find images containing similar objects |

### /ingest - Data Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Ingest single image (auto-indexes faces, OCR) |
| POST | `/ingest/batch` | Batch ingest (up to 64 images, 300+ RPS) |
| POST | `/ingest/directory` | Ingest entire directory |

### /analyze - Combined Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Full analysis: detection + faces + embedding + OCR |
| POST | `/analyze/batch` | Batch full analysis |

### /clusters - Clustering and Albums

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/clusters/train/{index}` | Train clustering model on index |
| POST | `/clusters/assign/{index}` | Assign vectors to clusters |
| GET | `/clusters/stats/{index}` | Get clustering statistics |
| GET | `/clusters/{index}/{cluster_id}` | Get items in specific cluster |

### /query - Data Retrieval

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/query/stats` | Get index statistics |
| GET | `/query/image/{id}` | Get image metadata by ID |
| DELETE | `/query/image/{id}` | Delete image from indexes |
| GET | `/query/duplicates` | Find duplicate images |

### /ocr - Text Extraction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ocr/predict` | Extract text from single image |
| POST | `/ocr/batch` | Batch OCR processing |

### /models - Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/models/upload` | Upload custom model |
| GET | `/models` | List available models |
| POST | `/models/{name}/export` | Export model to TensorRT |
| GET | `/models/{name}/status` | Get model loading status |

## Response Formats

### Detection Response

```json
{
  "detections": [
    {
      "x1": 0.15, "y1": 0.20, "x2": 0.45, "y2": 0.80,
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "image": {"width": 1920, "height": 1080},
  "inference_time_ms": 12.5
}
```

### Face Recognition Response

```json
{
  "faces": [
    {
      "box": {"x1": 0.30, "y1": 0.10, "x2": 0.50, "y2": 0.40},
      "confidence": 0.98,
      "landmarks": [[0.35, 0.20], [0.45, 0.20], [0.40, 0.28], [0.36, 0.35], [0.44, 0.35]],
      "embedding": [0.012, -0.034, ...]
    }
  ],
  "inference_time_ms": 18.3
}
```

### Embedding Response

```json
{
  "embedding": [0.012, -0.034, 0.056, ...],
  "dimensions": 512,
  "inference_time_ms": 8.2
}
```

### Search Response

```json
{
  "results": [
    {
      "image_id": "abc123",
      "score": 0.95,
      "image_path": "/data/photos/image001.jpg",
      "metadata": {}
    }
  ],
  "total_results": 42,
  "search_time_ms": 3.5
}
```

### Ingest Response

```json
{
  "image_id": "abc123",
  "indexed": {
    "global": true,
    "faces": 2,
    "objects": 5,
    "ocr": true
  },
  "processing_time_ms": 45.2
}
```

## Development Commands

### Deployment

```bash
# Start all services (requires GPU)
docker compose up -d

# View logs
docker compose logs -f triton-api
docker compose logs -f yolo-api

# Stop services
docker compose down
```

**Code Hot Reloading:**
- Volume mounts enable hot reloading for Python code
- To pick up changes: `docker compose stop yolo-api && docker compose rm -f yolo-api && docker compose up -d yolo-api`
- Only rebuild containers when `Dockerfile` or `requirements.txt` changes

### Testing

```bash
# Comprehensive test suite (all endpoints, ingest, search)
source .venv/bin/activate
python tests/test_full_system.py 2>&1 | tee test_results/test_results.txt

# Visual validation (draws bounding boxes on images)
python tests/validate_visual_results.py 2>&1 | tee test_results/visual_validation.txt

# View annotated images
ls test_results/*.jpg

# Quick API tests (legacy Makefile targets)
make test-all
make test-detect
make test-faces
make test-embed
make test-search
make test-ingest
make test-ocr

# Health check
make check-all
```

### Benchmarking

```bash
# Build benchmark tool
make bench-build

# Quick benchmark (30 seconds)
make bench-quick

# Full benchmark (60 seconds, 128 clients)
make bench-full

# Benchmark specific endpoints
make bench-detect
make bench-faces
make bench-ingest
```

### Model Management

```bash
# List loaded models
make models-list

# Export models to TensorRT
make export-models

# Restart Triton to reload models
make restart-triton
```

## Configuration

### Triton Server

- **Dynamic batching**: Preferred batch sizes [8, 16, 32, 64]
- **Max queue delay**: 5ms for balanced latency/throughput
- **TensorRT precision**: FP16 mode
- **Instance count**: 2 per model (configurable)

### OpenSearch Indexes

| Index | Embedding Dim | Distance | Purpose |
|-------|---------------|----------|---------|
| `images` | 512 | cosine | Global image embeddings |
| `faces` | 512 | cosine | Face embeddings |
| `objects` | 512 | cosine | Object crop embeddings |
| `ocr` | 512 | cosine | OCR text embeddings |

## Monitoring

The deployment includes a monitoring stack:

- **Prometheus**: Metrics collection (port 4604)
- **Grafana**: Visualization dashboards (port 4605, admin/admin)
- **Loki + Promtail**: Log aggregation (port 4606)
- **OpenSearch Dashboards**: Vector search management (port 4608)

View dashboards at http://localhost:4605

## Dependencies

Key Python packages in [requirements.txt](requirements.txt):

- `fastapi`, `uvicorn[standard]`: REST API server
- `tritonclient[all]`: Triton gRPC/HTTP client
- `opencv-python`, `pillow`: Image processing
- `numpy`: Numerical operations
- `opensearch-py>=2.3.0`: Async OpenSearch client
- `transformers>=4.30.0`: CLIP tokenizer

## Production Deployment

1. All endpoints on single port (4603) - simplifies load balancing
2. Configure `--workers=64` for production throughput
3. Enable OpenSearch security for production
4. Use Prometheus/Grafana for monitoring
5. Scale horizontally with multiple API instances behind load balancer

## Test Data

| Dataset | Location | Images | Purpose |
|---------|----------|--------|---------|
| LFW Deep Funneled | `test_images/faces/lfw-deepfunneled` | 13,233 | Face recognition validation |
| Sample Images | `test_images/` | varies | General testing |

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for third-party code attribution and licensing.
