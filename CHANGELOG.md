# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-07-04

### Fixed
- Fresh-install path (`scripts/setup.sh`) on Triton 26.06: trtexec moved
  to `/usr/bin` and its `--fp16` flag was removed in TensorRT 11 — the
  PaddleOCR engine step now works out of the box.
- End2end `config.pbtxt` is written from the built engine's actual output
  dtypes (EfficientNMS_TRT precision varies across TRT releases/builds).
- Health checks in `setup.sh` accept the `/health` -> `/ready` status
  contract.
- CI: valid action pins (trivy-action v0.36.0, checkout v5,
  codeql-action v4) and a scan timeout suited to the image size.
- Endpoint suite: dual-family checks skip gracefully when the optional
  YOLO26 engine is not exported.

## [0.2.0] - 2026-07-04

### Added
- **YOLO26 support served alongside YOLO11** in the same Triton + API
  instance: native NMS-free export (`export/export_yolo26.py`), a
  detection-adapter registry that resolves each model's output contract
  from Triton metadata, `YOLO_MODEL` env for the default detector, and
  per-request selection via the existing `model_name` parameter.
- Dual export toolchains in one image: the proven YOLO11 EfficientNMS
  path keeps its exact pin (`ultralytics==8.3.253`) in an isolated
  `/opt/venv-y11`; `export_models.py` re-execs into it transparently.
- `/live` and `/ready` health endpoints with real per-dependency probes
  (Triton gRPC `is_server_live`, OpenSearch HTTP); `/health` is now an
  alias of `/ready`.
- Prometheus `/metrics` endpoint with an `http_request_duration_seconds`
  histogram labeled by route template.
- dcgm-exporter service + GPU Metrics Grafana dashboard (all host GPUs).
- `make scan` targets and a GitHub Actions Trivy workflow (filesystem +
  API image, SARIF upload, weekly cron).
- Integration test suites: 20 GPU-free pytest tests and a live endpoint
  suite (25 checks) including dynamic YOLO26 load/unload
  (`tests/test_endpoints.sh dual`).
- `docs/MIGRATION_TRITON_26.md` upgrade guide.

### Changed
- **BREAKING: Triton upgraded to 26.06 (CUDA 13.3, TensorRT 11.1)** —
  every existing TensorRT engine must be re-exported; see the migration
  guide. A build-time assertion keeps the server TRT and the
  `tensorrt-cu13` pip pin in lockstep.
- TensorRT 11 is strongly-typed: FP16 is baked into the ONNX at export
  (NVIDIA ModelOpt AutoCast / onnxconverter-common for EfficientNMS
  graphs). Text detection (`paddleocr_det`) defaults to FP32 for
  threshold robustness.
- Monitoring stack pinned (prometheus v3.12.0, grafana 13.1.0, loki
  3.6.12 non-root, node-exporter v1.10.2); Promtail (EOL 2026-03-02)
  replaced by Grafana Alloy v1.17.1.
- OpenSearch upgraded to 3.6.0 (3.0–3.2 carry known HIGH CVEs).
- Triton container runs as the non-root `triton-server` user; both built
  images apply apt security upgrades and scan clean of fixable
  HIGH/CRITICAL CVEs (Nsight Systems CLI removed from the runtime image).
- Triton batching: 25 ms max queue delay; instance counts are a 12 GB
  baseline with scale-up guidance in each `config.pbtxt`.
- gRPC message caps raised to 512 MB for large raw detector heads.
- `/detect` always applies its confidence filter (NMS-free engines emit
  all top-K candidates).
- Request-id context moved to `src.core.logging` (importable by worker
  processes); structlog `foreign_pre_chain` formatter bug fixed.

### Fixed
- `cluster_distance` sort no longer 400s on documents missing the field
  or on freshly created indices.
- Container HEALTHCHECK targets `/live` so a degraded downstream
  dependency cannot cascade restarts through `depends_on`.

## [0.1.0] - 2026-03-19

Initial public release: YOLO11 detection, SCRFD + ArcFace face
recognition, MobileCLIP embeddings, PP-OCRv5 OCR, OpenSearch visual
search, Triton 25.10 TensorRT serving, monitoring stack.
