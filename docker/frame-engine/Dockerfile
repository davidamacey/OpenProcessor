# =============================================================================
# OpenProcessor frame engine: decode each photo once on the GPU, cut every model input from that
# frame, call the TensorRT models in Triton as a client (frame_engine/, D1 of
# docs/design/v070_rebuild/13_backend_speed_plan.md).
# =============================================================================
# Build (from the repository root, the context must hold src/domain and frame_engine):
#   docker compose build frame-engine
#
# The CUDA user-space libraries come from NVIDIA's pip wheels pinned to what the Triton prep model
# loaded inside openprocessor-triton (Triton 26.10 base), read from /proc/self/maps of that image
# on 2026-11-14, so the kernels compile and the JPEGs decode exactly as before:
#
#   library          loaded by the Triton model                    here (wheel)
#   NVRTC            nvidia/cu13/lib/libnvrtc.so.13, 13.2           nvidia-cuda-nvrtc 13.2.86 (same sha256)
#   NVRTC builtins   libnvrtc-builtins.so.13.2                       nvidia-cuda-nvrtc 13.2.86 (same sha256)
#   nvJPEG           cuda-13.4/.../libnvjpeg.so.13.2.3.58            nvidia-nvjpeg 13.2.3.58 (same version)
#   CUDA runtime     13.2 (cupy runtimeGetVersion 13020)             nvidia-cuda-runtime 13.2.86
#   CuPy / cuda-bindings / OpenCV (JPEG encode) / NumPy              14.2.0 / 13.4.4 / 5.0.0.93 / 2.5.4
#   pyclipper (the ocr branch's box unclip, as PaddleOCR's DB post-process)   1.4.0 (as the Triton image)
#
# The driver (libcuda) is mounted by the NVIDIA container toolkit. The Python minor version is the
# API image's (python:3.13-slim-trixie, same digest), because the engine imports src/domain.
# =============================================================================

# -- Stage 1: the packages (a --user install, like every other image of this repository) ----------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

COPY docker/frame-engine/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-warn-script-location --no-deps --only-binary=:all: -r /tmp/requirements.txt \
    && pip check \
    && rm -rf /root/.local/lib/python3.13/site-packages/pip \
        /root/.local/lib/python3.13/site-packages/pip-*.dist-info \
        /root/.local/bin/pip*

# -- Stage 2: runtime ------------------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# Debian point-release security fixes; no build tools, no extra packages.
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && groupadd -r engine \
    && useradd -r -g engine -G video -u 1000 -m -s /usr/sbin/nologin engine

COPY --from=builder --chown=engine:engine /root/.local /home/engine/.local
# The engine's code and the one part of src/ it imports (the wire module, the decode-route
# vocabulary and the geometry the CPU paths share).
COPY src/domain /app/src/domain
COPY frame_engine /app/frame_engine
# The recogniser's character dictionary, the one copy the Triton pipeline model also reads.
COPY models/paddleocr_rec_trt/ppocrv6_dict.txt /app/ocr/ppocrv6_dict.txt

# nvJPEG is opened by name (ctypes); CuPy finds NVRTC and the runtime in the same wheel directory.
ENV PATH=/home/engine/.local/bin:$PATH \
    LD_LIBRARY_PATH=/home/engine/.local/lib/python3.13/site-packages/nvidia/cu13/lib \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUPY_CACHE_DIR=/tmp/cupy_cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
USER engine

EXPOSE 8001 8002

# Ready means the model server answers and the region is registered (KServe ServerReady).
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, tritonclient.grpc as g; sys.exit(0 if g.InferenceServerClient('127.0.0.1:8001').is_server_ready(client_timeout=4) else 1)"]

CMD ["python", "-m", "frame_engine"]

ARG OP_BUILD_SHA=unknown
LABEL org.opencontainers.image.title="OpenProcessor Frame Engine" \
      org.opencontainers.image.description="Decode once on the GPU, cut every model input, call Triton" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}
