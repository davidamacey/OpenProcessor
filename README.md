# Visual AI API

**High-performance visual analysis API with NVIDIA Triton Inference Server.**

Object detection, face recognition, visual search, OCR, and embeddings - all through a unified REST API with TensorRT acceleration.

---

## Recent Updates

**✨ Latest (2026-01-26):**
- ✅ Fixed batch/directory ingest (94% success rate on 50-image batches)
- ✅ Comprehensive test suite added (32 tests, 100% passing)
- ✅ Visual validation with bounding box verification
- ✅ All pre-commit quality checks passing (ruff, mypy, bandit)
- ✅ gRPC connection optimization for high-throughput batch processing
- ✅ Test results organized in test_results/ directory

---

## Quick Start

```bash
# Clone and start services
cd /mnt/nvm/repos/triton-api
docker compose up -d

# Wait for models to load (2-3 minutes first time)
docker compose logs -f triton-api | grep "successfully loaded"

# Verify services are running
curl http://localhost:4603/health
```

---

## API Endpoints

All endpoints available on port **4603**.

### Object Detection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | YOLO object detection (single image) |
| `/detect/batch` | POST | Batch detection (up to 64 images) |

### Face Recognition

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/faces/detect` | POST | Face detection with landmarks (YOLO11-face) |
| `/faces/recognize` | POST | Detection + ArcFace 512-dim embeddings |
| `/faces/verify` | POST | 1:1 face comparison (two images) |
| `/faces/search` | POST | Find similar faces in index |
| `/faces/identify` | POST | 1:N face identification |

### Embeddings

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed/image` | POST | MobileCLIP image embedding (512-dim) |
| `/embed/text` | POST | MobileCLIP text embedding (512-dim) |
| `/embed/batch` | POST | Batch image embeddings |
| `/embed/boxes` | POST | Per-box crop embeddings |

### Visual Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search/image` | POST | Image-to-image similarity search |
| `/search/text` | POST | Text-to-image search |
| `/search/face` | POST | Face similarity search |
| `/search/ocr` | POST | Search images by text content |
| `/search/object` | POST | Object-level search (vehicles, people) |

### Data Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest image (auto-indexes faces, OCR, objects) |
| `/ingest/batch` | POST | Batch ingest (up to 64 images) |
| `/ingest/directory` | POST | Bulk ingest from server directory |

### OCR (Text Extraction)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ocr/predict` | POST | Extract text from image (PP-OCRv5) |
| `/ocr/batch` | POST | Batch OCR processing |

### Combined Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | All models on single image (YOLO + faces + CLIP + OCR) |
| `/analyze/batch` | POST | Batch combined analysis |

### Clustering & Albums

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/clusters/train/{index}` | POST | Train FAISS clustering for an index |
| `/clusters/stats/{index}` | GET | Get cluster statistics |
| `/clusters/{index}/{id}` | GET | Get cluster members |
| `/clusters/albums` | GET | List auto-generated albums |

### Data Retrieval

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query/image/{id}` | GET | Get stored image data/metadata |
| `/query/stats` | GET | Index statistics for all indexes |
| `/query/duplicates` | GET | List duplicate groups |

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/health/models` | GET | Triton model status |

---

## Usage Examples

### Python

```python
import requests

# Object Detection
with open('image.jpg', 'rb') as f:
    resp = requests.post('http://localhost:4603/detect', files={'image': f})
result = resp.json()
# {"detections": [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.95, "class_id": 0, "class_name": "person"}], ...}

# Face Recognition
with open('photo.jpg', 'rb') as f:
    resp = requests.post('http://localhost:4603/faces/recognize', files={'image': f})
print(resp.json())
# {"num_faces": 2, "faces": [...], "embeddings": [[...512 floats...], ...]}

# Image Embedding
with open('image.jpg', 'rb') as f:
    resp = requests.post('http://localhost:4603/embed/image', files={'image': f})
embedding = resp.json()['embedding']  # 512-dim vector

# Text-to-Image Search
resp = requests.post('http://localhost:4603/search/text',
                    json={'query': 'a red sports car', 'top_k': 10})
results = resp.json()['results']

# Image Ingestion (auto-indexes everything)
with open('photo.jpg', 'rb') as f:
    resp = requests.post('http://localhost:4603/ingest',
                        files={'image': f},
                        data={'image_id': 'photo_001'})
print(resp.json())
# {"status": "indexed", "image_id": "photo_001", "indexed": {"global": true, "faces": 2, "vehicles": 1}}

# OCR
with open('document.jpg', 'rb') as f:
    resp = requests.post('http://localhost:4603/ocr/predict', files={'image': f})
print(resp.json())
# {"num_texts": 5, "texts": ["Invoice", "Total: $100"], ...}

# Combined Analysis (everything in one call)
with open('scene.jpg', 'rb') as f:
    resp = requests.post('http://localhost:4603/analyze', files={'image': f})
result = resp.json()
# {"detections": [...], "faces": [...], "global_embedding": [...], "ocr": {...}}
```

### cURL

```bash
# Detection
curl -X POST http://localhost:4603/detect -F "image=@photo.jpg"

# Face Recognition
curl -X POST http://localhost:4603/faces/recognize -F "image=@face.jpg"

# Text Search
curl -X POST http://localhost:4603/search/text \
    -H "Content-Type: application/json" \
    -d '{"query": "sunset beach", "top_k": 10}'

# Ingestion
curl -X POST http://localhost:4603/ingest \
    -F "image=@photo.jpg" \
    -F "image_id=my_photo_001"
```

---

## Response Formats

### Detection Response

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
  "inference_time_ms": 12.5
}
```

**Note:** Coordinates are normalized (0.0-1.0). Multiply by image width/height for pixels.

### Face Recognition Response

```json
{
  "num_faces": 2,
  "faces": [
    {
      "box": {"x1": 0.30, "y1": 0.10, "x2": 0.50, "y2": 0.40},
      "confidence": 0.98,
      "landmarks": [[0.35, 0.20], [0.45, 0.20], [0.40, 0.28], [0.36, 0.35], [0.44, 0.35]]
    }
  ],
  "embeddings": [[...512 floats...]],
  "inference_time_ms": 25.3
}
```

### Search Response

```json
{
  "status": "success",
  "results": [
    {"image_id": "img_001", "score": 0.95, "image_path": "/path/to/image.jpg"}
  ],
  "total_results": 10,
  "search_time_ms": 15.2
}
```

### Ingest Response

```json
{
  "status": "success",
  "image_id": "photo_001",
  "num_detections": 5,
  "num_faces": 2,
  "embedding_norm": 1.0,
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

## Architecture

```
Client (Port 4603)
       |
       v
  +----------+
  | yolo-api |  FastAPI service (all endpoints)
  +----------+
       |
       v
  +-----------+     +------------+
  | triton-api|     | opensearch |
  | (GPU)     |     | (k-NN)     |
  +-----------+     +------------+
```

**Services:**
- `yolo-api` (port 4603): FastAPI service handling all requests
- `triton-api` (ports 4600-4602): NVIDIA Triton Inference Server with TensorRT models
- `opensearch` (port 4607): Vector database for similarity search
- `prometheus/grafana` (ports 4604/4605): Monitoring stack

---

## Models

| Model | Purpose | Backend |
|-------|---------|---------|
| YOLO11 | Object detection | TensorRT End2End |
| YOLO11-face | Face detection | TensorRT |
| ArcFace | Face embeddings (512-dim) | TensorRT |
| MobileCLIP | Image/text embeddings (512-dim) | TensorRT |
| PP-OCRv5 | Text detection + recognition | TensorRT |

All models use FP16 precision with dynamic batching for optimal throughput.

---

## Performance

**Measured Latency (single request):**
| Operation | Time | Throughput |
|-----------|------|------------|
| Object Detection | 140-170ms | ~6-7 RPS |
| Face Detection | 100-150ms | ~7-10 RPS |
| Face Recognition | 105-130ms | ~8-9 RPS |
| Image Embedding (CLIP) | 6-8ms | ~120 RPS |
| Text Embedding (CLIP) | 5-17ms | ~60-200 RPS |
| OCR Prediction | 170-350ms | ~3-6 RPS |
| Full Analyze | 280-430ms | ~2-3 RPS |
| Single Image Ingest | 750-950ms | ~1-1.3 RPS |
| **Batch Ingest (50 images)** | **7.3s total** | **~6.8 images/sec** |

**Batch processing** provides ~2-3x throughput improvement over sequential single-image processing.

---

## System Requirements

**Minimum:**
- NVIDIA GPU with 8GB+ VRAM (Ampere or newer)
- 16GB RAM, 16 CPU cores
- Docker with NVIDIA Container Toolkit

**Recommended:**
- NVIDIA A100/A6000/RTX 4090 (16GB+)
- 64GB RAM, 48+ CPU cores
- NVMe SSD for image storage

---

## Configuration

### Worker Count

```yaml
# docker-compose.yml
command: --workers=64  # Production
command: --workers=2   # Development
```

### GPU Selection

```yaml
# docker-compose.yml
device_ids: ['0', '2']  # Use GPUs 0 and 2
```

---

## Testing

Run comprehensive test suite to verify all functionality:

```bash
# Full system test (32 tests covering all endpoints)
source .venv/bin/activate
python tests/test_full_system.py 2>&1 | tee test_results/test_results.txt

# Visual validation (draws bounding boxes on test images)
python tests/validate_visual_results.py 2>&1 | tee test_results/visual_validation.txt

# View annotated test images
ls test_results/*.jpg
```

**Test Coverage:**
- ✅ All ML model endpoints (detection, faces, CLIP, OCR)
- ✅ Single and batch processing
- ✅ Directory ingest pipeline (50+ images)
- ✅ OpenSearch indexing and search
- ✅ Visual validation with bounding boxes

---

## Benchmarking

```bash
cd benchmarks
./build.sh
./triton_bench --mode quick    # 30-second test
./triton_bench --mode full     # Full benchmark
```

See [benchmarks/README.md](benchmarks/README.md) for detailed benchmarking guide.

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: AI assistant instructions and detailed architecture
- **[docs/](docs/)**: Technical documentation
  - [docs/OCR_SETUP_GUIDE.md](docs/OCR_SETUP_GUIDE.md): OCR model setup
  - [docs/FACE_RECOGNITION_IMPLEMENTATION.md](docs/FACE_RECOGNITION_IMPLEMENTATION.md): Face recognition details
  - [docs/opensearch_schema_design.md](docs/opensearch_schema_design.md): Vector search schema
- **[export/README.md](export/README.md)**: Model export documentation
- **[benchmarks/README.md](benchmarks/README.md)**: Benchmark tool guide

---

## Attribution

This project uses:
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork for End2End TensorRT export
- [Apple MobileCLIP](https://github.com/apple/ml-mobileclip)
- [InsightFace ArcFace](https://github.com/deepinsight/insightface)
- [YOLO11-face](https://github.com/akanametov/yolo-face)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete licensing information.

---

**Built for maximum throughput** - Process 100K+ images in minutes, visual search in milliseconds.
