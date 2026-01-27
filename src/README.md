# Visual AI API - Source Code

**Unified FastAPI service providing object detection, face recognition, visual search, and OCR through a single REST API.**

---

## Service Overview

**Single unified service** running on port **4603** with capability-based API design:

| Capability | Endpoint Prefix | Description |
|------------|----------------|-------------|
| **Object Detection** | `/detect` | YOLO11 detection (80 COCO classes) |
| **Face Recognition** | `/faces` | SCRFD + ArcFace embeddings |
| **Embeddings** | `/embed` | MobileCLIP image/text embeddings |
| **Visual Search** | `/search` | OpenSearch k-NN similarity search |
| **Data Ingestion** | `/ingest` | Batch processing and indexing |
| **OCR** | `/ocr` | PP-OCRv5 text extraction |
| **Combined Analysis** | `/analyze` | All models in single request |
| **Clustering** | `/clusters` | FAISS-based clustering |
| **Data Retrieval** | `/query` | Metadata and statistics |
| **Health** | `/health` | Service and model status |

**All inference powered by NVIDIA Triton with TensorRT acceleration.**

---

## File Structure

```
src/
├── main.py                      # Application entry point with lifespan management
│
├── routers/                     # FastAPI route handlers (one per capability)
│   ├── __init__.py              # Router exports
│   ├── health.py                # Health check endpoints
│   ├── detect.py                # Object detection (/detect)
│   ├── faces.py                 # Face recognition (/faces)
│   ├── embed.py                 # CLIP embeddings (/embed)
│   ├── search.py                # Visual search (/search)
│   ├── ingest.py                # Data ingestion (/ingest)
│   ├── ocr.py                   # Text extraction (/ocr)
│   ├── analyze.py               # Combined analysis (/analyze)
│   ├── clusters.py              # Clustering (/clusters)
│   └── query.py                 # Data retrieval (/query)
│
├── services/                    # Business logic layer
│   ├── __init__.py              # Service exports
│   ├── inference.py             # Core Triton inference operations
│   ├── embedding.py             # MobileCLIP embedding generation
│   ├── face_recognition.py      # Face detection and recognition
│   ├── face_identity.py         # Face identification and verification
│   ├── visual_search.py         # OpenSearch visual search operations
│   ├── ocr_service.py           # PP-OCRv5 text extraction
│   ├── clustering.py            # FAISS clustering
│   ├── duplicate_detection.py   # Image deduplication
│   └── image.py                 # Image processing utilities
│
├── clients/                     # External service clients
│   ├── __init__.py              # Client exports
│   ├── triton_client.py         # Triton gRPC client wrapper
│   ├── triton_pool.py           # Connection pooling for Triton
│   └── opensearch.py            # OpenSearch async client
│
├── schemas/                     # Pydantic models for validation
│   ├── __init__.py              # Schema exports
│   ├── common.py                # Shared schemas (ImageInfo, BoundingBox)
│   ├── detection.py             # Detection request/response models
│   ├── faces.py                 # Face recognition models
│   ├── embeddings.py            # Embedding models
│   ├── search.py                # Search models
│   ├── ingest.py                # Ingestion models
│   ├── ocr.py                   # OCR models
│   ├── analyze.py               # Analysis models
│   └── clusters.py              # Clustering models
│
├── config/                      # Configuration management
│   ├── __init__.py              # Config exports
│   └── settings.py              # Pydantic settings with validation
│
├── core/                        # Core application components
│   ├── __init__.py              # Core exports
│   ├── dependencies.py          # FastAPI dependencies & factories
│   └── exceptions.py            # Custom exception classes
│
└── utils/                       # Utility functions
    ├── __init__.py              # Utility exports
    ├── image_processing.py      # Image decode & validation
    └── cache.py                 # LRU caching utilities
```

---

## Architecture Overview

### Application Factory Pattern

The service uses FastAPI's application factory pattern in `main.py`:

```python
# main.py
from src.routers import (
    analyze_router,
    clusters_router,
    detect_router,
    embed_router,
    faces_router,
    health_router,
    ingest_router,
    ocr_router,
    query_router,
    search_router,
)

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(detect_router)
    app.include_router(faces_router)
    app.include_router(embed_router)
    app.include_router(search_router)
    app.include_router(ingest_router)
    app.include_router(ocr_router)
    app.include_router(analyze_router)
    app.include_router(clusters_router)
    app.include_router(query_router)
    return app
```

### Dependency Injection

Centralized in `core/dependencies.py`:
- `get_triton_pool()` - Manages Triton gRPC connection pool
- `get_opensearch_client()` - Manages OpenSearch async client
- `get_*_service()` - Service factory functions using `@lru_cache`

### Service Layer

Business logic separated into `services/`:
- `InferenceService` - Core Triton inference operations
- `EmbeddingService` - MobileCLIP image/text embeddings
- `FaceRecognitionService` - Face detection and ArcFace embeddings
- `FaceIdentityService` - Face verification and identification
- `VisualSearchService` - OpenSearch k-NN operations and ingestion
- `OCRService` - PP-OCRv5 text detection and recognition
- `ClusteringService` - FAISS clustering for albums
- `ImageService` - Image processing and validation

---

## API Capabilities

### Object Detection (`routers/detect.py`)

YOLO11 object detection with TensorRT End2End (GPU NMS).

**Endpoints:**
```
POST /detect              # Single image detection
POST /detect/batch        # Batch detection (up to 64 images)
```

**Example:**
```bash
curl -X POST http://localhost:4603/detect -F "image=@photo.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "x1": 0.094, "y1": 0.278, "x2": 0.870, "y2": 0.989,
      "confidence": 0.918,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "image": {"width": 1920, "height": 1080},
  "inference_time_ms": 145.2
}
```

---

### Face Recognition (`routers/faces.py`)

SCRFD-10G face detection with ArcFace embeddings (512-dim).

**Endpoints:**
```
POST /faces/detect        # Face detection with landmarks
POST /faces/recognize     # Detection + ArcFace embeddings
POST /faces/verify        # Compare two faces (1:1)
POST /faces/search        # Find similar faces in index
POST /faces/identify      # Identify face against known identities (1:N)
```

**Example:**
```bash
curl -X POST http://localhost:4603/faces/recognize -F "image=@photo.jpg"
```

**Response:**
```json
{
  "num_faces": 2,
  "faces": [
    {
      "box": {"x1": 0.30, "y1": 0.10, "x2": 0.50, "y2": 0.40},
      "confidence": 0.98,
      "landmarks": [[0.35, 0.20], [0.45, 0.20], ...]
    }
  ],
  "embeddings": [[...512 floats...]],
  "inference_time_ms": 120.5
}
```

---

### Embeddings (`routers/embed.py`)

MobileCLIP embeddings for image/text similarity (512-dim).

**Endpoints:**
```
POST /embed/image         # Single image embedding
POST /embed/text          # Text embedding
POST /embed/batch         # Batch image embeddings
POST /embed/boxes         # Per-box crop embeddings
```

**Example:**
```bash
# Image embedding
curl -X POST http://localhost:4603/embed/image -F "image=@photo.jpg"

# Text embedding
curl -X POST http://localhost:4603/embed/text \
  -H "Content-Type: application/json" \
  -d '{"text": "a red sports car"}'
```

**Response:**
```json
{
  "embedding": [0.012, -0.034, 0.056, ...],
  "dimensions": 512,
  "inference_time_ms": 7.2
}
```

---

### Visual Search (`routers/search.py`)

OpenSearch k-NN similarity search across all embedding types.

**Endpoints:**
```
POST /search/image        # Image-to-image similarity
POST /search/text         # Text-to-image search
POST /search/face         # Face similarity search
POST /search/ocr          # Search by text content
POST /search/object       # Object-level search
```

**Example:**
```bash
# Search by text
curl -X POST http://localhost:4603/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "sunset beach", "top_k": 10}'
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "image_id": "img_001",
      "score": 0.95,
      "image_path": "/data/photos/image001.jpg",
      "metadata": {}
    }
  ],
  "total_results": 10,
  "search_time_ms": 8.5
}
```

---

### Data Ingestion (`routers/ingest.py`)

Batch processing with automatic indexing to OpenSearch.

**Endpoints:**
```
POST /ingest              # Single image (auto-indexes faces, objects, OCR)
POST /ingest/batch        # Batch ingest (up to 64 images)
POST /ingest/directory    # Directory ingest with deduplication
```

**Example:**
```bash
curl -X POST http://localhost:4603/ingest \
  -F "image=@photo.jpg" \
  -F "image_id=photo_001"
```

**Response:**
```json
{
  "status": "success",
  "image_id": "photo_001",
  "num_detections": 5,
  "num_faces": 2,
  "indexed": {
    "global": true,
    "vehicles": 1,
    "people": 2,
    "faces": 2,
    "ocr": true
  },
  "ocr": {
    "num_texts": 3,
    "full_text": "Invoice Total: $100",
    "indexed": true
  },
  "total_time_ms": 850.4
}
```

---

### OCR (`routers/ocr.py`)

PP-OCRv5 text detection and recognition.

**Endpoints:**
```
POST /ocr/predict         # Extract text from single image
POST /ocr/batch           # Batch OCR processing
```

**Example:**
```bash
curl -X POST http://localhost:4603/ocr/predict -F "image=@document.jpg"
```

**Response:**
```json
{
  "num_texts": 5,
  "texts": ["Invoice", "Total: $100", ...],
  "regions": [
    {
      "text": "Invoice",
      "box": [0.1, 0.1, 0.3, 0.15],
      "confidence": 0.98
    }
  ],
  "full_text": "Invoice\\nTotal: $100\\n...",
  "inference_time_ms": 220.3
}
```

---

### Combined Analysis (`routers/analyze.py`)

Run all models on a single image in one request.

**Endpoints:**
```
POST /analyze             # All models (detection + faces + embedding + OCR)
POST /analyze/batch       # Batch combined analysis
```

**Example:**
```bash
curl -X POST http://localhost:4603/analyze -F "image=@photo.jpg"
```

**Response:**
```json
{
  "detections": [...],
  "faces": [...],
  "global_embedding": [...],
  "ocr": {...},
  "total_time_ms": 380.5
}
```

---

## Configuration

Settings managed via `config/settings.py` with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton-server:8001` | Triton gRPC address |
| `OPENSEARCH_URL` | `http://opensearch:4607` | OpenSearch REST address |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload size |
| `SLOW_REQUEST_THRESHOLD_MS` | `1000` | Log slow requests |

---

## Running the Service

### Via Docker Compose (Production)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f yolo-api

# Stop services
docker compose down
```

### Standalone (Development)

```bash
# Requires Triton and OpenSearch running
docker compose up -d triton-server opensearch

# Activate virtual environment
source .venv/bin/activate

# Run service with hot reload
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 4603 \
  --workers 2 \
  --reload
```

---

## Performance Metrics

**Measured Latency (single request):**

| Operation | Time | Throughput |
|-----------|------|------------|
| Object Detection | 140-170ms | ~6-7 RPS |
| Face Detection | 100-150ms | ~7-10 RPS |
| Face Recognition | 105-130ms | ~8-9 RPS |
| Image Embedding | 6-8ms | ~120 RPS |
| Text Embedding | 5-17ms | ~60-200 RPS |
| OCR Prediction | 170-350ms | ~3-6 RPS |
| Full Analyze | 280-430ms | ~2-3 RPS |
| Single Ingest | 750-950ms | ~1-1.3 RPS |

**Batch Processing:**
- Directory ingest: 6.8 images/sec (50 images in 7.3s)
- Batch provides ~2-3x throughput improvement over sequential processing

---

## Related Documentation

- [../README.md](../README.md) - Project overview and API reference
- [../CLAUDE.md](../CLAUDE.md) - Development instructions for AI assistants
- [../PROJECT_STATUS.md](../PROJECT_STATUS.md) - Current status and test results
- [utils/README.md](utils/README.md) - Utilities documentation
- [../docs/README.md](../docs/README.md) - Technical documentation index

---

**Last Updated:** 2026-01-26
**Version:** 2.0 - Unified capability-based API
