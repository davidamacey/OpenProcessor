# Documentation Index

This folder holds maintainer material: architecture, schemas, design plans, research,
measurements and release notes. User and operator guides live in the documentation site,
`docs-site/docs/`.

---

## Using OpenProcessor

| Where | What it covers |
|---|---|
| [../README.md](../README.md) | Overview, the one-line install, a first result |
| [../INSTALLATION.md](../INSTALLATION.md) | The installer, every flag, the `openprocessor` CLI, install from source |
| [../docs-site/docs/getting-started/quick-start.mdx](../docs-site/docs/getting-started/quick-start.mdx) | A first project end to end |
| [../docs-site/docs/guides/](../docs-site/docs/guides/) | Task guides: projects, ingest policy, region profiles, prompt packs, VLM selection, dataset import, combine |
| [../docs-site/docs/operations/](../docs-site/docs/operations/) | Workers, health, exports, training, monitoring |
| [../docs-site/docs/configuration/](../docs-site/docs/configuration/) | Basic and advanced settings |
| [../docs-site/docs/cropwright/](../docs-site/docs/cropwright/) | The Cropwright web UI |
| [CURATION.md](CURATION.md) | A map from each curation topic to its site page |
| [../SECURITY.md](../SECURITY.md) | No authentication, LAN exposure, VLM URL policy, reporting |

Build the site with `make docs-build`.

---

## Project and architecture

| Document | What it covers |
|---|---|
| [VISION_AND_GOALS.md](VISION_AND_GOALS.md) | What the project is for, the v0.6.0 scope and the standards the code is held to (canonical) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Services, project isolation, data model, the multi-box region cascade, config store, workers |
| [opensearch_schema_design.md](opensearch_schema_design.md) | Global and per-project index schemas, `region_boxes`, clustering |
| [design/curation_api_contract.md](design/curation_api_contract.md) | Route table and wire models for `/curation` |
| [design/curation_design_rationale.md](design/curation_design_rationale.md) | Why the subsystem is built the way it is |
| [../contracts/README.md](../contracts/README.md) | Generated OpenAPI and TypeScript contracts, the source of truth for routes |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup, tests, contracts, doc checks, commit conventions |
| [../CLAUDE.md](../CLAUDE.md) | Orientation for AI coding agents working in the repo |

---

## Design plans and research

| Folder | What it holds |
|---|---|
| [design/](design/README.md) | Design docs and plans with status and issue numbers |
| [research/](research/) | Measured snapshots and external-reference research (clustering, VLM sizing, GPU decode, quantization). Indexed in [design/README.md](design/README.md) |
| [market/](market/README.md) | Market research and competitive analysis, as of 2026-10-04 |

---

## Inference and models

| Document | What it covers |
|---|---|
| [OCR.md](OCR.md) | PP-OCRv5 detection and recognition setup |
| [FACE_RECOGNITION_IMPLEMENTATION.md](FACE_RECOGNITION_IMPLEMENTATION.md) | SCRFD and ArcFace |
| [../export/README.md](../export/README.md) | Exporting models to TensorRT, PE-Core encoders |
| [MIGRATION_TRITON_26.md](MIGRATION_TRITON_26.md) | Re-exporting engines for Triton 26.06 and TensorRT 11 |
| [Technical/TRITON_BEST_PRACTICES.md](Technical/TRITON_BEST_PRACTICES.md) | Triton batching, instance groups, tuning |
| [security/triton_cve_hardening.md](security/triton_cve_hardening.md) | Triton image CVE posture |
| [image_size_audit.md](image_size_audit.md) | Docker image size audit and plan (not implemented) |

---

## Measurements

| Document | What it covers |
|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | The measurement log: setup, hardware, method and results for every benchmark. Cite it for any number |
| [benchmarks/](benchmarks/) | Raw result JSON and parity references behind `PERFORMANCE.md` |
| [../benchmarks/README.md](../benchmarks/README.md) | The `triton_bench` tool |

---

## Releases

| Document | What it covers |
|---|---|
| [releases/](releases/) | Release notes per version and the release checklist |
| [../CHANGELOG.md](../CHANGELOG.md) | Release history |
| [../ATTRIBUTION.md](../ATTRIBUTION.md) | Third-party code and licenses |

---

## Scripts and tooling

| Document | What it covers |
|---|---|
| [../scripts/README.md](../scripts/README.md) | Setup, CLI, release, dataset, example and curation scripts |
| [../monitoring/README.md](../monitoring/README.md) | Prometheus, Grafana, Loki configuration |

Check the docs against the code:

```bash
.venv/bin/python scripts/docs/check_docs_vs_code.py
```

---

## External resources

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenSearch documentation](https://opensearch.org/docs/latest/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
