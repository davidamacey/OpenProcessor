# Migration Guide: Triton 26.06 / TensorRT 11 Upgrade

This release moves the stack from Triton 25.10 (TensorRT 10.13) to
**Triton 26.06 (CUDA 13.3, TensorRT 11.1)** and pins every container in
`docker-compose.yml`. Existing deployments need the one-time steps below.

## 1. Re-export ALL TensorRT engines (required)

TensorRT serialized engines (`.plan`) are **not portable across TensorRT
major versions**. Every engine built under 25.10 (TRT 10.x) fails to load
on 26.06 (TRT 11.x) with a serialization-version error.

```bash
# Rebuild images first
docker compose build

# Start only what exports need
docker compose up -d triton-server yolo-api

# Re-export every model (runs inside the API container)
docker compose exec yolo-api python /app/export/export_models.py --formats onnx_end2end trt_end2end
docker compose exec yolo-api python /app/export/export_scrfd.py
docker compose exec yolo-api python /app/export/export_face_recognition.py
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py
docker compose exec yolo-api python /app/export/export_mobileclip_text_encoder.py
docker compose exec yolo-api python /app/export/export_paddleocr_det.py
docker compose exec yolo-api python /app/export/export_paddleocr_rec.py

# Restart Triton to load the new engines
docker compose restart triton-server
```

Notes:

- The client-side `tensorrt-cu13==11.1.0.106` pin **must** match the TRT
  bundled in the Triton image. Do not bump it independently; upgrade it
  together with the Triton base tag.
- GPU requirements: driver R570+ (CUDA 13.x) and compute capability >= 7.5
  (Turing or newer). Volta/Pascal are no longer supported by this Triton
  release line.

## 2. Loki data volume ownership (one-time)

Loki previously ran as root (`user: "0"`); it now runs as the image's
builtin non-root user (uid 10001). Existing volumes are root-owned and
will fail with permission errors until chowned:

```bash
docker compose stop loki
docker run --rm -v openprocessor_loki_data:/loki alpine chown -R 10001:10001 /loki
docker compose up -d loki
```

Fresh installs need nothing.

## 3. Promtail → Grafana Alloy

Promtail reached end-of-life on 2026-03-02 and has been replaced by
**Grafana Alloy** (`monitoring/alloy-config.alloy`). The shipped pipeline
is equivalent (Docker service discovery for the Triton + API containers).

If you customized `monitoring/promtail-config.yml`, convert it:

```bash
docker run --rm -v $(pwd)/monitoring:/cfg grafana/alloy:v1.17.1 \
  convert --source-format=promtail --output=/cfg/alloy-config.alloy /cfg/promtail-config.yml
```

## 4. YOLO26 support (new, optional)

ultralytics moved to the 8.4 line (YOLO26). The YOLO11 EfficientNMS export
toolchain keeps its exact production pin in an isolated venv inside the
image (`/opt/venv-y11`) — `export_models.py` re-execs into it
automatically, so existing export commands are unchanged.

To serve YOLO26 alongside YOLO11:

```bash
docker compose exec yolo-api python /app/export/export_yolo26.py --models small
curl -X POST http://localhost:4603/models/yolo26_small_trt/load
curl -F image=@test.jpg 'http://localhost:4603/detect?model_name=yolo26_small_trt'
```

Set `YOLO_MODEL=yolo26_small_trt` on the yolo-api service to make it the
default detector. Output-format differences between the families are
handled transparently (adapter resolved from Triton model metadata).

## 5. Health endpoint semantics

- `/live` — process liveness only (new; used by the container HEALTHCHECK).
- `/ready` — actively probes Triton (gRPC `is_server_live`) and OpenSearch;
  returns **503 with per-service detail** while any dependency is down.
- `/health` — now an alias of `/ready`. If you previously treated `/health`
  as an always-200 liveness signal, point that consumer at `/live`.

## 6. Pinned image matrix

| Service | Image |
|---|---|
| triton-server | `nvcr.io/nvidia/tritonserver:26.06-py3` (via `Dockerfile.triton`) |
| opensearch | `opensearchproject/opensearch:3.6.0` |
| opensearch-dashboards | `opensearchproject/opensearch-dashboards:3.6.0` |
| prometheus | `prom/prometheus:v3.12.0` |
| grafana | `grafana/grafana:13.1.0` |
| loki | `grafana/loki:3.6.12` (non-root) |
| alloy (replaces promtail) | `grafana/alloy:v1.17.1` |
| node-exporter | `prom/node-exporter:v1.10.2` |
| dcgm-exporter | `nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04` |

OpenSearch 3.3 → 3.6 is a same-major upgrade; existing indices roll
forward in place. Take a snapshot first if the data matters.
