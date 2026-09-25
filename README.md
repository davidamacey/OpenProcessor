# Visual AI API

**High-performance visual analysis API with NVIDIA Triton Inference Server.**

Object detection, face recognition, visual search, OCR, and embeddings - all through a unified REST API with TensorRT acceleration.

---

## Quick Start

### Copy & Run (One Line)

```bash
git clone https://github.com/davidamacey/OpenProcessor.git && cd OpenProcessor && ./scripts/setup.sh
```

That's it! The setup script automatically:
- Pulls pre-built Docker images from Docker Hub (~15GB)
- Detects your GPU and selects the optimal profile
- Downloads required models (~500MB, ~16 seconds)
- Exports models to TensorRT (~30-60 min first time, one-time only)
- Starts all services and runs smoke tests

**First-time setup takes ~30-60 minutes** (mostly TensorRT compilation). Subsequent starts take seconds.

### Non-Interactive Setup

```bash
# Clone and setup with defaults (no prompts)
git clone https://github.com/davidamacey/OpenProcessor.git && cd OpenProcessor && ./scripts/setup.sh --yes

# Or specify a profile explicitly
./scripts/setup.sh --profile=standard --gpu=0 --yes
```

### Verify Installation

```bash
curl http://localhost:4603/health
# {"status":"ready","version":"0.3.0",...}

# Quick test with an image
curl -X POST http://localhost:4603/detect -F "image=@your-image.jpg"
```

### Management Commands

```bash
./scripts/openprocessor.sh status    # Check service health
./scripts/openprocessor.sh logs -f   # View live logs
./scripts/openprocessor.sh restart   # Restart all services
./scripts/openprocessor.sh help      # See all commands
```

See [INSTALLATION.md](INSTALLATION.md) for manual installation, troubleshooting, and advanced options.

## Docker Hub

Pre-built images are available on Docker Hub (pulled automatically by setup):

```bash
docker pull davidamacey/openprocessor:latest        # FastAPI service (~14GB)
docker pull davidamacey/openprocessor-triton:latest  # Triton server (~18GB)
```

---

## GPU Compatibility

| Profile | VRAM | GPUs | Throughput |
|---------|------|------|------------|
| minimal | 6-8GB | RTX 3060, RTX 4060 | ~5 RPS |
| standard | 12-24GB | RTX 3080, RTX 4090 | ~15 RPS |
| full | 48GB+ | A6000, A100 | ~50 RPS |

Switch profiles: `./scripts/openprocessor.sh profile <name>`

---

## Setup Timing (RTX A6000, 48GB)

First-time setup is dominated by one-time TensorRT model compilation.
Subsequent starts take only seconds since compiled engines are cached.

| Step | Time | Notes |
|------|------|-------|
| Model downloads | ~16s | 5 models, ~500MB from HuggingFace/GitHub |
| Docker image pull | ~30s | Pre-built from Docker Hub (~32GB total) |
| **TensorRT exports** | **~31 min** | **One-time only, cached after first run** |
| Service startup | ~33s | Triton loads cached TRT engines |
| **Total first run** | **~32 min** | |
| **Subsequent starts** | **~30s** | Just `docker compose up -d` |

**TensorRT Export Breakdown:**

| Model | Export Time | Engine Size |
|-------|-----------|-------------|
| YOLO11 detection | ~8 min | ~45 MB |
| YOLO26 detection (optional) | ~6 min | ~40 MB |
| SCRFD face detection | ~2 min | ~20 MB |
| ArcFace embeddings | ~2 min | ~86 MB |
| MobileCLIP image encoder | ~4 min | ~35 MB |
| MobileCLIP text encoder | ~1 min | ~125 MB |
| PaddleOCR (det + rec) | ~15 min | ~15 MB |

Times vary by GPU. Faster GPUs with more CUDA cores export faster.
Lower VRAM GPUs (8-12GB) may take 45-60 minutes total.

**Dual YOLO families — YOLO11 and YOLO26 side by side.** One Triton
instance and one API serve both:

- **YOLO11** exports through the EfficientNMS end2end toolchain
  (`export/export_models.py`, GPU-NMS 4-tensor engines — the proven
  default path).
- **YOLO26** exports through the stock ultralytics native toolchain
  (`export/export_yolo26.py`, natively NMS-free single-tensor engines —
  no plugin required):

  ```bash
  docker compose exec yolo-api python /app/export/export_yolo26.py --models small
  curl -X POST http://localhost:4603/models/yolo26_small_trt/load
  ```

- The API resolves each model's output format from **Triton metadata**,
  so `/detect?model_name=yolo26_small_trt` and the YOLO11 default work
  interchangeably; set `YOLO_MODEL=yolo26_small_trt` to switch the
  default detector. Models load/unload at runtime via
  `POST /models/{name}/load` / `POST /models/{name}/unload`.

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
| `/faces/detect` | POST | Face detection with landmarks (SCRFD) |
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
| `/ready` | GET | Readiness probe — Triton + OpenSearch reachability (no separate `/health/models` route exists) |

---

### Curation & Active Learning

**Experimental for v0.3.0.** A generic, domain-agnostic active-learning
curation subsystem: ingest images, detect and crop regions of interest,
cluster and browse them, label by hand or via an OpenAI-compatible VLM,
track class registries and review queues, export labeled datasets, and
drive a training loop through a documented file-based protocol.

It's real, working, and tested — 25 route groups, 109 routes under
`/curation` as of this release (verify the live count with
`python -c "from src.main import app; print(len([r for r in app.routes if r.path.startswith('/curation')]))"`)
— but it's new, still evolving, ships opt-in behind the `curation`
Docker Compose profile, and is disabled by default:

```bash
docker compose --profile curation up -d
```

**Try it with a public sample.** No dataset ships in this repo (nothing
proprietary is bundled anywhere) -- fetch a small, license-filtered COCO
2017 subset instead:

```bash
make sample-coco-readme   # 200 images, 20 per class, ~1-2 min on a fast link
```

This writes `data/samples/coco_va_readme/` (images + `ATTRIBUTION.csv` +
`coco_gt.json`), gitignored, from a pinned, deterministic selection
(`scripts/datasets/manifests/coco_va_200.json`) filtered to Flickr
licenses safe to redistribute crops of (Attribution,
Attribution-ShareAlike, "No known copyright restrictions", "United
States Government Work" -- explicitly not NonCommercial/NoDerivs). Then:

```bash
# 1. Create a few classes (see docs/CURATION.md "Create classes from zero")
# 2. Set OP_INGEST_PRIMARY_DETECTOR_MODEL, and narrow ingest to the classes
#    you just created with OP_INGEST_PRIMARY_CLASS_IDS (otherwise a stock
#    detector's full label space -- all 80 COCO classes -- becomes item
#    proposals; e.g. 2,3,5,7 for car/motorcycle/bus/truck).
# 3. Point OP_SOURCE_ROOT_HOST at data/samples in .env (the compose mount
#    target is fixed at /data/source -- data/samples/coco_va_readme/images
#    is NOT under the default ./data/source, so a walker --root pointed
#    straight at the samples dir 404s every image as unservable_path):
echo 'OP_SOURCE_ROOT_HOST=./data/samples' >> .env
docker compose up -d --force-recreate yolo-api
# 4. --root is a container path under the /data/source mount, not a
#    host-relative one:
docker compose exec yolo-api python scripts/curation/ingest_walker.py \
  --root /data/source/coco_va_readme/images --api-base http://localhost:8000/curation
```

`make sample-coco` (the larger 800-image + 12-image upload + 24-image
re-ingest side-set default) and `make sample-plates` (300-image Open
Images V7 "Vehicle registration plate" region set) are the same tool at
a bigger scale — see [docs/CURATION.md](docs/CURATION.md#seed--bootstrap-path-for-a-fresh-install)
and `python scripts/datasets/fetch_coco_subset.py --help` /
`python scripts/datasets/fetch_openimages_plates.py --help` for every
flag. `make sample-clean` removes everything fetched.

See **[docs/CURATION.md](docs/CURATION.md)** for the full user guide —
what's required (you supply your own detector/VLM/trainer models), the
class-registry schema with a non-vehicle worked example, the complete
`OP_*` environment variable table, and the known gaps stated up front.
See **[docs/design/curation_api_contract.md](docs/design/curation_api_contract.md)**
for the hand-written wire-level API contract, or the generated,
always-current schema under **[contracts/](contracts/)**
(`contracts/openapi/curation.json` plus generated TypeScript types) —
and **[SECURITY.md](SECURITY.md)**: the curation surface has no
authentication, same as the rest of this API.

**Cropwright** (a separate, optional SvelteKit frontend) is one
consumer of the `/curation` API — a labeling UI for the cascade above.
It is not required: every curation route works from `curl`/`httpx`/the
generated OpenAPI client too. Cropwright is not published in this
repo; if you build one, wire it up like this:

- **Docker network:** join the compose network this stack created —
  `${COMPOSE_PROJECT_NAME:-openprocessor}_triton_net` (e.g.
  `openprocessor_triton_net` with an unset `COMPOSE_PROJECT_NAME`; see
  `docker network ls` after `docker compose up`).
- **API upstream:** `API_UPSTREAM=http://op-api:8000`. `yolo-api` carries a
  network alias `op-api` specifically so a frontend's out-of-the-box
  default (many default to `op-api`, not this repo's `yolo-api` service
  name) resolves without extra configuration — `http://yolo-api:8000`
  works identically if your frontend's default is the service name
  instead.
- **API prefix:** `PUBLIC_API_PREFIX=/curation` (must equal this API's
  `OP_API_PREFIX`, default `/curation`).

See `docs/CURATION.md` "Wiring up Cropwright" for the full env var list.

### Security: local tool, no authentication

**This service — core API and curation subsystem alike — has no
authentication, authorization, or rate limiting.** It is built to run
on a trusted machine or private network behind your own reverse proxy,
never exposed directly to a network you don't trust (and never to the
public internet). See **[SECURITY.md](SECURITY.md)** for the full list
of routes that are dangerous without access control and the other
default-open components (Grafana, OpenSearch) you should lock down
before wider deployment.

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

# Text-to-Image Search (query params, not a JSON body -- `text`, not `query`)
resp = requests.post('http://localhost:4603/search/text',
                    params={'text': 'a red sports car', 'top_k': 10})
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

# Text Search (query params, not a JSON body -- `text`, not `query`)
curl -X POST "http://localhost:4603/search/text?text=sunset+beach&top_k=10"

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
      "landmarks": [0.35, 0.20, 0.45, 0.20, 0.40, 0.28, 0.36, 0.35, 0.44, 0.35]
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
  +--------------+     +------------+
  | triton-server|     | opensearch |
  | (GPU)        |     | (k-NN)     |
  +--------------+     +------------+
```

**Services:**
- `yolo-api` (port 4603): FastAPI service handling all requests
- `triton-server` (ports 4600-4602): NVIDIA Triton Inference Server with TensorRT models
- `opensearch` (port 4607): Vector database for similarity search
- `prometheus/grafana` (ports 4604/4605): Monitoring stack (opt-in — `docker compose --profile monitoring up -d` / `make up-monitoring`; not started by `make up`)

---

## Models

| Model | Purpose | Backend |
|-------|---------|---------|
| YOLO11 | Object detection | TensorRT End2End |
| SCRFD-10G | Face detection + landmarks | TensorRT |
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

### GPU sizing — default core Triton loadout

Measured on a 48 GB card (fresh-start E2E run, 2026-09-25) by unloading
each model in turn via `POST /v2/repository/models/<name>/unload`. The
stock `models/*/config.pbtxt` loadout came to **~25.2 GB** at its
previous default instance counts — with the VLM service alone measured
at ~23 GB in that same run, that left no room for the segmenter or a
training run on the same card.

| Model | Instance count | Approx. VRAM (measured) | Notes |
|---|---:|---:|---|
| `scrfd_10g_bnkps` | 4 (old) → 1 (default) | 6.8 GB → ~1.7 GB | Face detection |
| `mobileclip2_s2_image_encoder` | 2 (old) → 1 (default) | 3.4 GB → ~1.7 GB | FP32 build (no FP16 baked into this export yet) |
| `arcface_w600k_r50` | 4 (old) → 1 (default) | 2.75 GB → ~0.7 GB | Face embeddings |
| `mobileclip2_s2_text_encoder` | 1 | 0.7 GB | Unchanged |
| yolo11/PE/OCR set (remaining core models) | as shipped | ~11.65 GB | Not reduced by this pass — no per-model breakdown measured yet |

The shipped defaults now use `count: 1` for the three over-provisioned
models above, bringing the core loadout down to roughly 16-17 GB — a
meaningful cut from 25.2 GB, though still above a strict 12-14 GB target
if you also need the segmenter/VLM/trainer on the same card (the
remaining ~11.65 GB yolo/PE/OCR set hasn't been broken down
per-model yet). Raise any
instance count back up (`config.pbtxt`, or re-export with
`export/export_scrfd.py` / `export/export_face_recognition.py`) on a
card with headroom to spare — hot-path models (face detection under
heavy face-search load, for example) benefit most from more instances.
Curation-only deployments that don't need the core face/vehicle path at
all can `unload` those 3 models entirely instead of exporting them.

`pe_text_encoder` (semantic-search query embeddings, added to the default
load list alongside `pe_image_encoder`) is `KIND_CPU` with one instance —
it adds **0 GB of GPU VRAM**, only host RAM, and is what
`OP_PE_TEXT_BACKEND=auto` now prefers over loading its own copy in every
uvicorn worker (see `export/README.md#pe-core-encoders-curation-embeddings`).

### GPU sizing — segmenter (curation region cascade)

The optional `segmenter` service (`--profile segmenter`) is the single
biggest curation VRAM line item after Triton. Two knobs trade VRAM for
throughput (`SEGMENTER_INSTANCES`, `SEGMENTER_SHARED_WEIGHTS` in
`.env` — see `env.template`):

| `SEGMENTER_INSTANCES` | `SEGMENTER_SHARED_WEIGHTS` | Approx. VRAM | Notes |
|---|---|---|---|
| 1 | 0 or 1 | ~2 GB | Fits a 12GB card alongside Triton |
| 2 | 1 (recommended) | ~4 GB | One weight copy, per-instance activations only |
| 2 | 0 | ~4-6 GB | Each instance loads its own weight copy — no benefit over shared, higher VRAM |
| 4 | 1 | ~7-8 GB | 48GB-class cards (A6000/A100) with headroom for training too |

`SEGMENTER_SHARED_WEIGHTS=1` is recommended whenever more than one
instance is configured: instances share one copy of the model weights and
only pay per-instance activation memory, instead of each loading its own
full copy. Measure your actual footprint with `nvidia-smi` after the
service reports `"loaded":true` on its `/health` endpoint — the numbers
above are a starting point, not a guarantee, and depend on image
resolution and batch size.

---

## Testing

Run comprehensive test suite to verify all functionality. `test_full_system.py`
and `validate_visual_results.py` hit the running stack over HTTP, so they read
their target ports from the environment (`API_PORT`, `TRITON_HTTP_PORT`,
`OPENSEARCH_PORT` — same names as `.env`/`docker-compose.yml`; default to
4603/4600/4607 if unset, so this is a no-op unless you remapped ports):

```bash
# Full system test (32 tests covering all endpoints) — host venv path
.venv/bin/python tests/test_full_system.py 2>&1 | tee test_results/test_results.txt

# Visual validation (draws bounding boxes on test images)
.venv/bin/python tests/validate_visual_results.py 2>&1 | tee test_results/visual_validation.txt

# View annotated test images
ls test_results/*.jpg

# Full offline pytest suite (see docs/CURATION.md for the curation-only suite)
.venv/bin/python -m pytest tests/ -q
```

**Docker-only path** (F-21/F-31) — the production `yolo-api` image installs
only `requirements.txt` (no `pytest`, no `requirements-test.txt`), so a bare
`docker compose exec yolo-api pytest ...` fails with `executable file not
found`. Install the test deps into the running container first (not
persisted across a recreate):

```bash
docker compose exec yolo-api pip install -r requirements-test.txt
docker compose exec yolo-api python -m pytest tests/ -q --ignore=tests/live
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
  - [docs/CURATION.md](docs/CURATION.md): Curation & active-learning subsystem user guide (experimental)
  - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Component and runtime topology
  - [docs/OCR.md](docs/OCR.md): OCR model setup
  - [docs/FACE_RECOGNITION_IMPLEMENTATION.md](docs/FACE_RECOGNITION_IMPLEMENTATION.md): Face recognition details
  - [docs/opensearch_schema_design.md](docs/opensearch_schema_design.md): Vector search schema
- **[contracts/](contracts/)**: Generated, always-current API schema — OpenAPI (`contracts/openapi/curation.json`) and TypeScript types, regenerated by `scripts/codegen/generate_contracts.py`
- **[export/README.md](export/README.md)**: Model export documentation
- **[benchmarks/README.md](benchmarks/README.md)**: Benchmark tool guide
- **[SECURITY.md](SECURITY.md)**: Security policy — read this before exposing the API beyond a trusted network
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Dev setup, test suites, and commit conventions
- **[CHANGELOG.md](CHANGELOG.md)**: Release history

---

## Attribution

This project uses:
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [levipereira/ultralytics](https://github.com/levipereira/ultralytics) fork for End2End TensorRT export
- [Apple MobileCLIP](https://github.com/apple/ml-mobileclip)
- [InsightFace ArcFace](https://github.com/deepinsight/insightface)
- [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Meta Perception Encoder (PE-Core)](https://github.com/facebookresearch/perception_models)
- [Meta Segment Anything 3](https://github.com/facebookresearch/sam3) (optional segmenter)

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete licensing information.

---

## License

This project is licensed under the **GNU Affero General Public License
v3.0 or later (AGPL-3.0-or-later)** — see [LICENSE](LICENSE). It was
previously MIT-badged; it is re-badged AGPL-3.0-or-later because it
vendors an AGPL-3.0 Ultralytics fork (`src/ultralytics_patches/`) whose
copyleft terms propagate to the combined work. Third-party components
retain their own licenses (BSD, Apache-2.0, MIT, and others) — see
[ATTRIBUTION.md](ATTRIBUTION.md) for the full per-component table.

---

**Built for maximum throughput** - Process 100K+ images in minutes, visual search in milliseconds.
