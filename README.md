# OpenProcessor

**A backend for computer-vision dataset curation and training, with a fast
inference API underneath.** Ingest images, detect and crop regions, label them
with people and a vision-language model, import and combine datasets, export,
train and promote models. It works for any image domain, not one built-in use
case.

Underneath is a unified REST API on NVIDIA Triton Inference Server with
TensorRT engines: object detection, face recognition, visual search, OCR and
embeddings.

The optional web UI, **Cropwright**, is a separate frontend for the curation
API. Every route also works from `curl`, `httpx` or the generated OpenAPI
client. See [`docs/VISION_AND_GOALS.md`](docs/VISION_AND_GOALS.md) for the
project's scope and standards.

> Screenshot pending: Cropwright (project list with per-project counts and a review grid showing items with several region boxes)

---

## Features

**Inference API** (port 4603, `/v1` twin for the core routes)

- Object detection (YOLO11 and YOLO26 side by side), batch up to 64 images.
- Face detection, ArcFace embeddings, 1:1 verify, search and 1:N identify.
- MobileCLIP image and text embeddings, visual search over OpenSearch k-NN.
- PP-OCRv5 text detection and recognition.
- Combined analysis, ingest into the visual-search indexes, FAISS clustering
  and albums.

**Curation and training** (`/curation`, all project scoped)

- **Isolated projects.** Each project has its own indexes, directories, class
  registry, settings and jobs. Create, archive, unarchive and delete through
  the API. Isolation is enforced at the OpenSearch transport layer.
- **Per-project settings and config store.** Shared defaults per axis
  (cluster method, sort, prompt pack, region profile, VLM), a vocabulary
  endpoint for editors, and clone-settings between projects.
- **Prompt packs.** What the VLM is asked and how it answers is versioned
  data: create, edit, activate, roll back, test on stored crops.
- **Region profiles.** The region stage (detector, segmenter prompt, text
  reading, thresholds) is versioned data with an impact report on activation,
  rollback and a test-on-crop route.
- **One or many regions per item.** `region_boxes` is always a list. A single
  region is a list of one. Each box has its own id, state, verdict, text,
  cluster and embedding. `max_regions_per_item` in the profile sets the cap.
- **Keymaps.** Per-project keyboard shortcuts for the review UI, validated
  server side.
- **VLM endpoint selection.** A deployment-wide registry of OpenAI-compatible
  endpoints with a local model catalog. Each project activates one. Remote
  endpoints need an explicit acknowledgement because crops leave the host.
- **Dataset import.** YOLO, COCO and OpenProcessor-export layouts, with
  preview, by-name class mapping, undo and resume.
- **Unified reprocess.** One route re-runs detect, open-vocabulary, region, VLM
  and embed scopes over selected items, with a dry run by default.
- **Open-vocabulary detection.** A project-level set of text prompts ("traffic
  cone") that SAM 3 runs on the whole image; every hit is a normal item, with
  a dry-run cost estimate, a per-image test route and a shared segmenter gate.
- **Combine projects.** Merge up to eight projects into a new one with class
  mapping, dedup and holdout handling.
- **Class identity by name.** Import, combine, export, train, promote and
  model sharing map classes by name, never by index.
- **The lock rule.** A human-set or import-validated label or box, and
  anything in the frozen test holdout, is never overwritten by an automated
  writer.
- Ingest, clustering, review queues, scoring, diverse selection, semantic
  search, export, training jobs, bake-off model comparison and promotion.
- A public example: cars from COCO, wheels as regions
  (`examples/region_profiles/vehicle_wheel.json`).

---

## Quick Start

### One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/setup-openprocessor.sh | bash
```

This installs the latest published release into `./openprocessor/`: no git
clone, no local image build, no host Python. The script you pipe in only
resolves the release, downloads that release's own `setup-openprocessor.sh`
and `SHA256SUMS`, checks the checksum, and runs the verified copy. Images are
pinned by digest (`images.lock`) and checked after the pull. It asks which
tiers you want, picks GPUs, exports the TensorRT engines inside the
containers, starts everything and runs a health check.

**Needs:** Linux, Docker with Compose v2, an NVIDIA GPU with the NVIDIA
Container Toolkit, and disk for the tiers you pick (about 60 GB for `core`,
about 135 GB for everything). First install takes 30-60 minutes, mostly image
pulls and TensorRT export.

### Tiers

| Tier | What you get | Extra images | Needs |
|---|---|---|---|
| `core` | Triton, the API (port 4603), OpenSearch | ~53 GB | ~16 GB VRAM (`--profile minimal` for 6-8 GB cards) |
| `curation` | curation workers, PE-Core embeddings, evaluator | + ~15 GB | + ~2 GB VRAM; implies `core` |
| `segmenter` | SAM 3 segmenter | + ~7 GB | 2-8 GB VRAM; a HuggingFace token with SAM 3 access (gated); implies `curation` |
| `vlm` | local vLLM serving a model from the VLM catalog | + ~19 GB | ~23 GB VRAM for the default; implies `curation` |
| `trainer` | training service + MLflow | + ~9 GB | >= 16 GB free VRAM while training; implies `curation` |
| `cropwright` | the Cropwright web UI (separate compose project) | + ~65 MB | implies `curation` |

Monitoring (Prometheus, Grafana, Loki) is not a tier: add `--with-monitoring`.
Its dashboards are default-open, so it is off unless you ask.

### Unattended install

```bash
curl -fsSL https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/setup-openprocessor.sh \
  | bash -s -- --tiers core,curation,cropwright --unattended
```

`--unattended` never prompts: every choice comes from flags or their defaults,
and a step that needs consent (a non-loopback `--bind`, an external VLM, a
purge) fails unless its consent variable is set. Use `--all` for every tier,
`--version vX.Y.Z` to pin a release, `--dry-run` to see every command without
running any. All flags: [INSTALLATION.md](INSTALLATION.md#installer-flags).

### Verify the download yourself

```bash
V=vX.Y.Z   # the release you want
curl -fsSLO https://github.com/davidamacey/OpenProcessor/releases/download/$V/setup-openprocessor.sh
curl -fsSLO https://github.com/davidamacey/OpenProcessor/releases/download/$V/SHA256SUMS
grep ' setup-openprocessor.sh$' SHA256SUMS | sha256sum -c -
bash setup-openprocessor.sh --version "$V"
```

`SHA256SUMS` comes from the same place as the files it covers. It proves
**integrity** (the download is complete and uncorrupted), **not authenticity**:
anyone who could replace the release files could replace `SHA256SUMS` too. The
same holds for `images.lock` and `cropwright.lock` (they pin exact digests
and are covered by `SHA256SUMS`). Signed releases are follow-up work.

### Network access: Cropwright on your LAN

**Cropwright is reachable on your LAN by default, for homelab or
small-business use. The API itself stays bound to 127.0.0.1. There is no login
on Cropwright — a warning is shown. Pass `--local-only` to opt out and keep
everything on 127.0.0.1.**

Do not port-forward Cropwright (or any OpenProcessor port) to the public
internet. If you need access beyond a trusted network, put a reverse proxy
with authentication in front. `--bind <ip>` publishes the API ports on that
address instead (with a warning and a typed confirmation), and a specific
address also narrows Cropwright to that interface. See
[SECURITY.md](SECURITY.md).

### After the install

```bash
cd openprocessor
./openprocessor status            # services and health
./openprocessor logs yolo-api -f  # live logs
./openprocessor sample coco       # fetch a public COCO sample (200 images)
./openprocessor vlm list          # the local VLM catalog
./openprocessor upgrade           # to the latest release (backs up first)
./setup-openprocessor.sh --repair | --rollback | --uninstall
```

```bash
curl http://127.0.0.1:4603/health
curl -X POST http://127.0.0.1:4603/detect -F "image=@your-image.jpg"
```

Every subcommand of the `openprocessor` CLI is listed in
[INSTALLATION.md](INSTALLATION.md#the-openprocessor-cli). The installer sizes
the OpenSearch heap from your RAM (RAM/8, 1-8 GB) and prints it in the summary;
see [INSTALLATION.md](INSTALLATION.md#opensearch-heap-sizing).

### Install from source

For development, or to build the images yourself:

```bash
git clone https://github.com/davidamacey/OpenProcessor.git && cd OpenProcessor && ./scripts/setup.sh
```

`scripts/setup.sh` detects your GPU, picks a profile, downloads the models,
exports them to TensorRT and starts the services. Add `--yes` for no prompts,
or `--profile=standard --gpu=0 --yes` to choose explicitly. Manage a checkout
with `./scripts/openprocessor.sh status|logs|restart|help` (it forwards to the
`openprocessor` CLI and adds the dev overlay `docker-compose.dev.yml`). See
[INSTALLATION.md](INSTALLATION.md#install-from-source) for manual steps.

### Docker Hub

Images are published on Docker Hub under versioned tags. The installer never
uses `:latest`; it runs the digests in the release's `images.lock`:

```bash
docker pull davidamacey/openprocessor:<version>         # FastAPI service
docker pull davidamacey/openprocessor-triton:<version>  # Triton server
```

`<version>` is the release number without the leading `v` (the `VERSION` file).

---

## Your first project

Curation routes live under `/curation/projects/{project}/...`. The `default`
project exists after the first start; create more with
`POST /curation/projects`. The slug is 2-32 characters: lowercase letters,
digits and single hyphens, starting with a letter.

```bash
API=http://127.0.0.1:4603/curation

# 1. create a project (slug is permanent; display_name is editable)
curl -X POST $API/projects -H 'Content-Type: application/json' \
  -d '{"slug": "cars", "display_name": "Cars"}'

# 2. list projects, then read this project's counts
curl $API/projects
curl $API/projects/cars/stats

# 3. add classes (class identity is the name)
curl -X POST $API/projects/cars/classes -H 'Content-Type: application/json' \
  -d '{"name": "car"}'
```

Then ingest images (below), activate a region profile and a VLM endpoint if
you want regions or automatic labels, and review. The route for each step:

| Step | Routes |
|---|---|
| Classes | `GET /curation/projects/{project}/classes`, `POST /curation/projects/{project}/classes` |
| Ingest | `POST /curation/projects/{project}/ingest/batch`, `POST /curation/projects/{project}/ingest/upload` |
| Import a labeled dataset | `POST /curation/projects/{project}/datasets/preview`, `POST /curation/projects/{project}/datasets/imports` |
| Region profile | `POST /curation/projects/{project}/region_profiles`, `POST /curation/projects/{project}/region_profiles/{name}/activate` |
| Prompt pack | `POST /curation/projects/{project}/prompt_packs`, `POST /curation/projects/{project}/prompt_packs/{name}/activate` |
| VLM endpoint | `POST /curation/vlm/endpoints`, `POST /curation/projects/{project}/vlm/endpoints/{name}/activate` |
| Review | `GET /curation/projects/{project}/review/tabs`, `GET /curation/projects/{project}/review/{tab}` |
| Export and train | `POST /curation/projects/{project}/export/yolo`, `POST /curation/projects/{project}/train/preflight`, `POST /curation/projects/{project}/train/start` |

Try it with a public sample. No dataset ships in this repo (nothing
proprietary is bundled anywhere). Fetch a small, license-filtered COCO 2017
subset instead:

```bash
make sample-coco-readme   # 200 images, 20 per class, ~1-2 min on a fast link
# or, from an installed deployment: ./openprocessor sample coco
```

This writes `data/samples/coco_va_readme/` (images + `ATTRIBUTION.csv` +
`coco_gt.json`), gitignored, from a pinned, deterministic selection filtered to
Flickr licenses safe to redistribute crops of (Attribution,
Attribution-ShareAlike, "No known copyright restrictions", "United States
Government Work"; never NonCommercial or NoDerivs). Then:

```bash
# 1. Create classes in your project (see docs/CURATION.md).
# 2. Narrow ingest to those classes with OP_INGEST_PRIMARY_CLASS_IDS in .env
#    (2,3,5,7 = car/motorcycle/bus/truck for COCO). Unset, a stock
#    detector proposes items for its whole label space (all 80 COCO classes).
#    OP_INGEST_PRIMARY_DETECTOR_MODEL picks the detector.
# 3. Point OP_SOURCE_ROOT_HOST at data/samples in .env (the compose mount
#    target is fixed at /data/source):
echo 'OP_SOURCE_ROOT_HOST=./data/samples' >> .env
docker compose up -d --force-recreate yolo-api
# 4. --root is a container path under the /data/source mount, not a
#    host-relative one:
docker compose exec yolo-api python scripts/curation/ingest_walker.py \
  --root /data/source/coco_va_readme/images --project cars \
  --api-base http://localhost:8000/curation
```

A single full dev deployment (dev overlay, GPU arbiter overlay, and the
curation, segmenter, vlm and training profiles) is one command:
`make dev-up` to build and start, `make dev-ps` to inspect, `make dev-down`
to stop (volumes are kept). Set `COMPOSE_PROJECT_NAME`, `OP_IMAGE_REPO` and
`OP_IMAGE_TAG` in `.env` to keep the locally built image tags separate from
published ones.

In a source checkout use `make up` and `make` targets, which add the dev
overlay; in an installed directory run `./openprocessor restart yolo-api` after
editing `.env`. `make sample-coco` (800 images plus side sets),
`make sample-plates` (300 Open Images V7 images) and `make sample-clean` use
the same tool. See
[docs/CURATION.md](docs/CURATION.md) for the walkthrough and every flag.

### Example: cars and wheels

`examples/` holds a complete, public, multi-region example: the primary
detector finds cars, then the region stage finds the wheels on each car.

```bash
make sample-coco-cars      # 60 CC BY car images into data/samples/coco_car
.venv/bin/python scripts/examples/wheel_example_live.py \
  --api http://localhost:4603 --project wheels \
  --container-dir /data/source/coco_car/images
```

The script creates the project, creates and activates
`examples/region_profiles/vehicle_wheel.json` (`max_regions_per_item: 4`,
text-free) and `examples/prompt_packs/vehicle_wheel.json`, ingests the images,
waits for the detection worker and exports the wheel boxes. It needs the live
stack with the segmenter and an activated VLM. The same walk runs offline in
`tests/integration/test_wheel_example_e2e.py`.

---

## The curation model in brief

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/CURATION.md](docs/CURATION.md) and
[docs/design/curation_api_contract.md](docs/design/curation_api_contract.md).

**Projects.** Every project owns six OpenSearch indexes named
`{OP_PROJECT_INDEX_PREFIX}{slug}__{role}` (default prefix `op_prj_`; roles
`images`, `items`, `labels_confirmed`, `classes`, `umap_state`, `configs`),
a class registry and exports under `OP_PROJECTS_DATA_ROOT/{slug}/`, and its
own upload, job and training directories. A project is `building`, `active`,
`archived`, `deleting`, `deleted` or `failed`. `default` can be archived but
not deleted. Delete is a dry run first (`?dry_run=true`), then needs
`?confirm=<slug>` and answers 202.

| Lifecycle | Route |
|---|---|
| List, create | `GET /curation/projects`, `POST /curation/projects` |
| Read, rename, delete | `GET /curation/projects/{project}`, `PATCH /curation/projects/{project}`, `DELETE /curation/projects/{project}` |
| Archive, restore | `POST /curation/projects/{project}/archive`, `POST /curation/projects/{project}/unarchive` |
| Copy settings | `POST /curation/projects/{project}/clone_settings` |
| Combine | `POST /curation/projects/combine/preview`, `POST /curation/projects/combine`, `GET /curation/projects/combine/{job_id}` |

**Items and regions.** The primary detector proposes **items** (crops) at
ingest. When the active region profile names `parent_classes`, only items of
those classes (matched by name) get the region stage; an empty list means every
item does, so the shipped sub-region example names `car`, `truck`, `bus` and
`motorcycle`. Region detection is off until you set a profile. The detection worker
runs the profile's detector and segmenter legs, merges candidates, keeps up to
`max_regions_per_item`, and writes them as the item's `region_boxes` list. The
VLM verifies each box, a text reader fills `text` when the profile asks for
it, each box is embedded and clustered on its own, and humans edit per box.
Box states are `proposed`, `accepted`, `rejected` and `false_positive`; the
item's `region_status` is derived from them.

| Per-box operation | Route |
|---|---|
| List boxes with filters | `GET /curation/projects/{project}/regions` |
| Replace an item's box set | `PUT /curation/projects/{project}/crops/{crop_id}/regions` |
| Change one box | `PATCH /curation/projects/{project}/crops/{crop_id}/regions/{box_id}` |
| Set many boxes' state | `POST /curation/projects/{project}/regions/batch_box_state` |
| Cluster boxes | `POST /curation/projects/{project}/regions/cluster` |

**Config store.** Prompt packs, region profiles and VLM activations are
revisioned documents. Saving writes a new revision, activating applies one to
the running workers, and rollback reactivates the previous one. Activation
takes `expected_active` so two editors cannot overwrite each other.

| Config | Routes |
|---|---|
| Prompt packs | `GET /curation/projects/{project}/prompt_packs`, `POST /curation/projects/{project}/prompt_packs/{name}/activate`, `POST /curation/projects/{project}/prompt_packs/active/rollback`, `POST /curation/projects/{project}/prompt_packs/test` |
| Region profiles | `GET /curation/projects/{project}/region_profiles`, `POST /curation/projects/{project}/region_profiles/{name}/activate`, `GET /curation/projects/{project}/region_profiles/active/impact`, `POST /curation/projects/{project}/region_profiles/active/rollback`, `POST /curation/projects/{project}/region_profiles/test` |
| Settings and vocabulary | `GET /curation/projects/{project}/settings`, `PUT /curation/projects/{project}/settings`, `GET /curation/projects/{project}/config/vocabulary` |
| Keymap | `GET /curation/projects/{project}/keymap`, `PUT /curation/projects/{project}/keymap`, `POST /curation/projects/{project}/keymap/validate`, `POST /curation/projects/{project}/keymap/reset` |

> Screenshot pending: Cropwright (the region-profile editor with the activation impact panel)

**VLM endpoints.** The endpoint registry is deployment-wide
(`GET /curation/vlm/endpoints`); the activation is per project
(`GET /curation/projects/{project}/vlm/endpoints/active`). The local catalog
(`GET /curation/vlm/catalog`) lists models the in-compose vLLM can serve. The
API records the wanted model (`POST /curation/vlm/local/select`) and the host
applies it with `./openprocessor vlm use <id>`. API keys are never served:
store one with `./openprocessor vlm key set <slug>` and reference it as
`secret:<slug>`. Run-time overrides take `?vlm=<name>` on the labeling routes.

**Datasets and reprocess.**

| Operation | Route |
|---|---|
| Supported formats and limits | `GET /curation/projects/{project}/datasets/formats` |
| Upload a zip or tar | `POST /curation/projects/{project}/datasets/uploads` |
| Preview, with class suggestions | `POST /curation/projects/{project}/datasets/preview` |
| Start, read, undo | `POST /curation/projects/{project}/datasets/imports`, `GET /curation/projects/{project}/datasets/imports/{import_id}`, `POST /curation/projects/{project}/datasets/imports/{import_id}/undo` |
| Re-run scopes on items | `POST /curation/projects/{project}/reprocess`, `GET /curation/projects/{project}/reprocess/jobs/{job_id}` |
| Open-vocabulary prompt sets, and a test on one image | `GET /curation/projects/{project}/open_vocab`, `POST /curation/projects/{project}/open_vocab/test` |

Every dataset class that has boxes needs a mapping decision (`map`, `create`,
`skip` or `region`); mapping is by class name. A COCO layout must say
`"format": "coco"`. `scripts/curation/import_labeled_dataset.py` is a client
of these routes.

The curation surface is large (more than 230 operations). The generated schema in
[`contracts/openapi/curation.json`](contracts/openapi/curation.json) is the
source of truth, and [`contracts/`](contracts/) also has the TypeScript item
types.

---

## GPU Compatibility

| Profile | VRAM | GPUs | Throughput |
|---------|------|------|------------|
| minimal | 6-8GB | RTX 3060, RTX 4060 | ~5 RPS |
| standard | 12-24GB | RTX 3080, RTX 4090 | ~15 RPS |
| full | 48GB+ | A6000, A100 | ~50 RPS |

Switch profiles in a checkout: `./scripts/openprocessor.sh profile <name>`,
or pass `--profile` to the installer. Figures are rough; measure on your
hardware ([docs/PERFORMANCE.md](docs/PERFORMANCE.md)).

First-time setup is dominated by one-time TensorRT compilation (about 30
minutes on a 48 GB card, 45-60 minutes on 8-12 GB cards). Engines are cached,
so later starts take seconds to a minute.

**Dual YOLO families.** One Triton instance and one API serve YOLO11 and
YOLO26.

- YOLO11 exports through the EfficientNMS end2end toolchain
  (`export/export_models.py`, GPU-NMS engines, the default path).
- YOLO26 exports through the stock Ultralytics toolchain
  (`export/export_yolo26.py`, natively NMS-free engines):

  ```bash
  docker compose exec yolo-api python /app/export/export_yolo26.py --models small
  curl -X POST http://localhost:4603/models/yolo26_small_trt/load
  ```

- The API reads each model's output format from Triton metadata, so
  `/detect?model_name=yolo26_small_trt` and the YOLO11 default work
  interchangeably. Set `YOLO_MODEL=yolo26_small_trt` to change the default
  detector. Models load and unload at runtime with `POST /models/{name}/load`
  and `POST /models/{name}/unload`.

---

## API Endpoints

All endpoints are on port **4603**. The routes below are also under `/v1`.

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
| `/ready` | GET | Readiness probe: Triton + OpenSearch reachability |

The `/curation` routes are **not** mounted under `/v1`; they exist only at
`/curation` (the prefix is `OP_API_PREFIX`).

### Curation route groups

Everything below is under `/curation/projects/{project}/` unless marked
global. The full table is in
[docs/design/curation_api_contract.md](docs/design/curation_api_contract.md).

| Group | What it covers |
|---|---|
| `classes`, `class_sources` | Class registry: create, edit, merge, deprecate, restore |
| `crops`, `images` | Items: browse, label, move, exclude, discard, undo; image and thumbnail serving |
| `regions`, `crops/{crop_id}/regions` | Per-box list, edit, state, text, clustering, false-positive pull |
| `clusters` | Cluster cards, representatives, refine, auto-promote |
| `review` | Review tabs, queues, new-class proposals |
| `scores`, `select`, `search/text` | Mistakenness and uniqueness scores, diverse selection, semantic search |
| `vlm` | Class labeling, region verification, endpoint activation |
| `ingest`, `datasets`, `reprocess` | Ingest, labeled-dataset import, re-running pipeline scopes |
| `prompt_packs`, `region_profiles`, `settings`, `keymap`, `config` | Config store |
| `pipeline/auto_label`, `probe`, `pause`, `resume` | Auto-label jobs and pipeline control |
| `export`, `train`, `models`, `bakeoff`, `test_holdout` | Export, training, promotion, model comparison, frozen test set |
| `events`, `stats`, `health`, `methods` | SSE stream, counts, capability discovery |
| global: `projects`, `vlm/endpoints`, `vlm/catalog`, `vlm/local`, `events` | Registry and cross-project routes |

**Status:** the curation subsystem is complete for this release and tested
offline end to end, but it has no authentication (see below), and it ships
opt-in behind the `curation` compose profile.

### Security: local tool, no authentication

**This service, the core API and the curation subsystem alike, has no
authentication, authorization, or rate limiting.** It is built to run on a
trusted machine or private network behind your own reverse proxy, never
exposed directly to a network you don't trust (and never to the public
internet). See **[SECURITY.md](SECURITY.md)** for the routes that are dangerous
without access control and the other default-open components (Grafana,
OpenSearch) you should lock down before wider deployment.

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

### Curation item (region boxes)

Every item-returning curation route emits the same item shape
(`contracts/json/item_wire.json`). Region data is a list:

```json
{
  "crop_id": "c_123",
  "class_name": "car",
  "region_status": "detected",
  "region_count": 2,
  "region_revision": 4,
  "region_boxes": [
    {"box_id": "b1", "bbox_norm": [0.10, 0.60, 0.30, 0.90], "state": "accepted", "score": 0.91, "detector": "sam3"},
    {"box_id": "b2", "bbox_norm": [0.62, 0.60, 0.82, 0.90], "state": "proposed", "score": 0.77, "detector": "sam3"}
  ],
  "label_locked": false
}
```

The example shows a subset of the keys. A box also carries `text*`,
`cluster_id`, `cluster_subid`, `cluster_distance`, `rejection_reason` and
`locked`. `bbox_norm` is `[x1, y1, x2, y2]` in the item crop's frame.

---

## Architecture

```
Client (Port 4603)
       |
       v
  +----------+     +-----------------------------+
  | yolo-api |---->| curation workers (profile)  |
  +----------+     | detection, VLM, auto-label, |
       |           | cluster refresh, evaluator  |
       v           +-----------------------------+
  +--------------+     +------------+
  | triton-server|     | opensearch |
  | (GPU)        |     | (k-NN)     |
  +--------------+     +------------+
```

**Services** (compose profile in brackets):

- `yolo-api` (4603): FastAPI service handling all requests.
- `triton-server` (4600-4602): Triton with TensorRT models.
- `opensearch` (4607): vector database and the curation datastore.
- `curation-detection-worker`, `curation-vlm-worker`,
  `curation-auto-label-worker`, `curation-cluster-refresh`,
  `curation-evaluator` [`curation`].
- `segmenter` (4611) [`segmenter`], `vlm` (4612) [`vlm`].
- `curation-trainer`, `curation-mlflow` (4609) [`training`].
- Prometheus (4604), Grafana (4605), Loki (4606), OpenSearch Dashboards (4608),
  DCGM exporter (4610), Alloy [`monitoring`]. Opt in with `make up-monitoring`
  or `--with-monitoring`; `make up` does not start them.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Models

| Model | Purpose | Backend |
|-------|---------|---------|
| YOLO11 / YOLO26 | Object detection | TensorRT |
| SCRFD-10G | Face detection + landmarks | TensorRT |
| ArcFace | Face embeddings (512-dim) | TensorRT |
| MobileCLIP | Image/text embeddings (512-dim) | TensorRT |
| PP-OCRv5 | Text detection + recognition | TensorRT |
| PE-Core-L14-336 | Curation embeddings (1024-dim, image and text) | TensorRT image, Triton CPU text |
| Segmenter | Region proposals from a text prompt (optional) | HTTP service |
| VLM | Class labels and region verdicts (optional) | OpenAI-compatible endpoint |

Core models use FP16 with dynamic batching. Export steps: [export/README.md](export/README.md).

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

Every setting is in `.env` (template: [`env.template`](env.template)). The
"Curation quick-config" block lists the keys the curation tiers need.

### GPU placement

`TRITON_GPU_ID`, `API_GPU_ID`, `SEGMENTER_GPU_ID`, `EVALUATOR_GPU_ID` and
`VLM_GPU_ID` pick a GPU per service. `./openprocessor gpu plan` prints them.
The installer chooses a placement and `--gpu-plan` overrides it.

### Triton model loadout

The default Triton instance counts are small: face detection, face embeddings,
the MobileCLIP image encoder and the PE image encoder run one instance each;
`paddleocr_det_trt`, the YOLO models and `mobileclip2_s2_text_encoder` run
two. `pe_text_encoder` is a CPU instance and uses no VRAM. Raise a count in
the model's `config.pbtxt` when a hot path needs it and the card has room.
Check `nvidia-smi` after loading.

### Segmenter VRAM

`SEGMENTER_INSTANCES` (default 2) and `SEGMENTER_SHARED_WEIGHTS` (default 1)
trade VRAM for throughput. With shared weights the instances share one copy of
the model and each pays only for its own activations. Measure with
`nvidia-smi` after the segmenter's `/health` reports loaded.

### Workers

The API runs `uvicorn --workers=32` by default (see `docker-compose.yml`).
Lower it on small hosts.

---

## Testing

Run the offline suite (no Docker, no GPU needed):

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/test_full_system.py` and `tests/validate_visual_results.py` hit the
running stack over HTTP, so they read their target ports from the environment
(`API_PORT`, `TRITON_HTTP_PORT`, `OPENSEARCH_PORT`; defaults 4603/4600/4607):

```bash
# Full system test (all endpoints) — host venv path
.venv/bin/python tests/test_full_system.py 2>&1 | tee test_results/test_results.txt

# Visual validation (draws bounding boxes on test images)
.venv/bin/python tests/validate_visual_results.py 2>&1 | tee test_results/visual_validation.txt
```

**Docker-only path.** The production `yolo-api` image installs only
`requirements.txt` (no `pytest`, no `requirements-test.txt`), so a bare
`docker compose exec yolo-api pytest ...` fails with `executable file not
found`. Install the test deps into the running container first (not persisted
across a recreate):

```bash
docker compose exec yolo-api pip install -r requirements-test.txt
docker compose exec yolo-api python -m pytest tests/ -q --ignore=tests/live
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the live write-path harness and the
pre-commit hooks.

---

## Benchmarking

```bash
cd benchmarks
./build.sh
./triton_bench --mode quick    # 30-second test
./triton_bench --mode full     # Full benchmark
```

See [benchmarks/README.md](benchmarks/README.md) and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

---

## Documentation

- **[docs/](docs/README.md)**: documentation index
  - [docs/CURATION.md](docs/CURATION.md): curation user guide
  - [docs/design/curation_api_contract.md](docs/design/curation_api_contract.md): route and wire-model reference
  - [docs/VISION_AND_GOALS.md](docs/VISION_AND_GOALS.md): scope and standards
  - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): components, data model, runtime topology
  - [docs/OCR.md](docs/OCR.md): OCR model setup
  - [docs/FACE_RECOGNITION_IMPLEMENTATION.md](docs/FACE_RECOGNITION_IMPLEMENTATION.md): face recognition details
  - [docs/opensearch_schema_design.md](docs/opensearch_schema_design.md): index schemas
- **[contracts/](contracts/)**: generated API schema (OpenAPI and TypeScript types), regenerated by `scripts/codegen/generate_contracts.py`
- **[INSTALLATION.md](INSTALLATION.md)**: installer, CLI and source install
- **[export/README.md](export/README.md)**: model export
- **[scripts/README.md](scripts/README.md)**: scripts and curation tooling
- **[benchmarks/README.md](benchmarks/README.md)**: benchmark tool
- **[SECURITY.md](SECURITY.md)**: read before exposing the API beyond a trusted network
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: dev setup, tests, commit conventions
- **[CLAUDE.md](CLAUDE.md)**: orientation for AI coding agents
- **[CHANGELOG.md](CHANGELOG.md)**: release history

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
v3.0 or later (AGPL-3.0-or-later)** — see [LICENSE](LICENSE). It vendors an
AGPL-3.0 Ultralytics fork (`src/ultralytics_patches/`) whose copyleft terms
propagate to the combined work. Third-party components retain their own
licenses (BSD, Apache-2.0, MIT, and others) — see
[ATTRIBUTION.md](ATTRIBUTION.md) for the full per-component table.
