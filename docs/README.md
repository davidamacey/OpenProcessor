# Documentation Index

Documentation for OpenProcessor: the inference API and the curation subsystem.

---

## Start here

| Document | What it covers |
|---|---|
| [README.md](../README.md) | Overview, feature list, quick start, first project, the cars and wheels example |
| [INSTALLATION.md](../INSTALLATION.md) | The one-line installer, every flag, the `openprocessor` CLI, VLM selection, install from source |
| [VISION_AND_GOALS.md](VISION_AND_GOALS.md) | What the project is for, the v0.4.0 scope and the standards the code is held to |
| [SECURITY.md](../SECURITY.md) | No authentication, LAN exposure, VLM URL policy, reporting |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup, tests, contracts, doc checks, commit conventions |
| [CLAUDE.md](../CLAUDE.md) | Orientation for AI coding agents working in the repo |

---

## Curation and labeling

| Document | What it covers |
|---|---|
| [CURATION.md](CURATION.md) | User guide: projects, ingest, region profiles, prompt packs, VLM endpoints, import, combine, export, training |
| [design/curation_api_contract.md](design/curation_api_contract.md) | Route table and wire models for `/curation` |
| [design/curation_design_rationale.md](design/curation_design_rationale.md) | Why the subsystem is built the way it is |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Services, project isolation, data model, the multi-box region cascade, config store, workers |
| [opensearch_schema_design.md](opensearch_schema_design.md) | Global and per-project index schemas, `region_boxes`, clustering |
| [../contracts/README.md](../contracts/README.md) | Generated OpenAPI and TypeScript contracts |

The generated schema, [`contracts/openapi/curation.json`](../contracts/openapi/curation.json),
is the source of truth for routes and models.

---

## Inference and models

| Document | What it covers |
|---|---|
| [OCR.md](OCR.md) | PP-OCRv5 detection and recognition setup |
| [FACE_RECOGNITION_IMPLEMENTATION.md](FACE_RECOGNITION_IMPLEMENTATION.md) | SCRFD and ArcFace |
| [../export/README.md](../export/README.md) | Exporting models to TensorRT, PE-Core encoders, dual-head detectors |
| [MIGRATION_TRITON_26.md](MIGRATION_TRITON_26.md) | Re-exporting engines for Triton 26.06 and TensorRT 11 |
| [PERFORMANCE.md](PERFORMANCE.md) | FastAPI and gRPC tuning, profiling |
| [Technical/TRITON_BEST_PRACTICES.md](Technical/TRITON_BEST_PRACTICES.md) | Triton batching, instance groups, tuning |
| [security/triton_cve_hardening.md](security/triton_cve_hardening.md) | Triton image CVE posture |
| [../benchmarks/README.md](../benchmarks/README.md) | The `triton_bench` tool |

---

## Scripts and tooling

| Document | What it covers |
|---|---|
| [../scripts/README.md](../scripts/README.md) | Setup, CLI, release, dataset, example and curation scripts |
| [../ATTRIBUTION.md](../ATTRIBUTION.md) | Third-party code and licenses |
| [../CHANGELOG.md](../CHANGELOG.md) | Release history |

---

## API surface

All routes are on one port (4603 by default).

| Prefix | Description |
|---|---|
| `/detect`, `/faces`, `/embed`, `/search`, `/ingest`, `/ocr`, `/analyze`, `/clusters`, `/query`, `/models`, `/health` | Inference and visual search; also under `/v1` |
| `/curation/projects` | Project registry: list, create, combine |
| `/curation/projects/{project}/...` | Everything project scoped: classes, crops, regions, clusters, review, scores, VLM, prompt packs, region profiles, settings, keymap, datasets, reprocess, ingest, export, training |
| `/curation/vlm/...` | Deployment-wide VLM endpoint registry, catalog and local model selection |
| `/curation/events` | Global event stream |

---

## Repository layout

```
OpenProcessor/
├── README.md, INSTALLATION.md, SECURITY.md, CONTRIBUTING.md, CLAUDE.md
├── openprocessor              # management CLI (bash)
├── setup-openprocessor.sh     # one-line installer
├── docker-compose.yml         # deploy-safe stack
├── docker-compose.dev.yml     # checkout overlay: local builds, hot reload
├── docker-compose.gpu-arbiter.yml  # opt-in overlay
├── env.template               # every setting
├── Makefile
│
├── src/
│   ├── main.py                # FastAPI app
│   ├── routers/               # core routers; routers/curation/ for /curation
│   ├── services/              # curation/, config_store/, projects/, labeling/,
│   │                          #   detection/, training/ and the core services
│   ├── clients/               # Triton, OpenSearch, OCC helpers, PE encoder
│   ├── config/                # CurationConfig, RegionFields, projects, retired env
│   └── schemas/               # Pydantic models
│
├── scripts/                   # setup, lib/, release/, codegen/, datasets/, docs/,
│   └── curation/              #   examples/; curation/ has workers and tools
├── models/                    # Triton model repository
├── export/                    # model export scripts
├── docker/                    # segmenter, trainer, evaluator, test harness
├── examples/                  # region profiles, prompt packs, bake-off, VLM catalog
├── contracts/                 # generated API contracts
├── benchmarks/                # triton_bench (Go)
├── docs/, docs-site/          # documentation, Docusaurus site
├── monitoring/                # Prometheus, Grafana, Loki configuration
└── tests/                     # offline suite; tests/live needs the harness
```

---

## Common tasks

```bash
make up                    # start the core stack
make curation-up           # start the curation workers
curl http://localhost:4603/health
make test                  # offline pytest suite
make contracts             # regenerate contracts/ after an API change
.venv/bin/python scripts/docs/check_docs_vs_code.py   # check docs against code
```

Model exports:

```bash
make export-models         # YOLO TensorRT
make export-mobileclip     # MobileCLIP encoders
make setup-face-pipeline   # SCRFD + ArcFace
make setup-ocr             # PP-OCRv5
make export-pe             # PE-Core encoders (curation)
```

Check model status:

```bash
make models-list           # or: ./openprocessor models
```

---

## External resources

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenSearch documentation](https://opensearch.org/docs/latest/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
