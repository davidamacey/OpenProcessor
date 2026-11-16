# =============================================================================
# OpenProcessor native frame engine (Rust): decode each photo once on the GPU, cut every model
# input from that frame, call the TensorRT models in the model server as a client (native/,
# docs/design/v070_rebuild/17_native_engine_plan.md). The same protocol (version 1) as the Python
# engine in docker/frame-engine/, from a smaller image.
# =============================================================================
# Build (from the repository root; BuildKit, for the cache mounts):
#   docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml build engine-native
# or, without compose:
#   docker build -f docker/frame-engine-native/Dockerfile --build-context contracts_json=contracts/json .
#
# Three stages, each on a base pinned by digest (read from Docker Hub on 2026-11-15; a deliberate
# bump re-reads them):
#
#   wheels   the CUDA user-space libraries from NVIDIA's pip wheels, the exact builds the Python
#            engine pins (docker/frame-engine/requirements.txt, whose table says which build of
#            the model server's image each is): NVRTC, the CUDA runtime and nvJPEG. The base is the
#            Python engine's own, so one pip resolves them the same way.
#   build    libjpeg-turbo 3.1.2 from its release tarball (SHA-256 pinned, SIMD required: nasm),
#            built static, then `cargo build --release --locked` of the engine. 3.1.2 is the version
#            both simplejpeg 1.9.0 and OpenCV 5.0.0.93 bundle, so the crop JPEGs can match byte for
#            byte; turbojpeg-sys links it through TURBOJPEG_SOURCE=explicit.
#   runtime  Debian slim, a non-root user, the engine binary and the CUDA libraries. The driver
#            (libcuda, libnvidia-ml) is mounted by the NVIDIA container toolkit.

# Scanning: Trivy reads the Rust crates (Cargo.lock) and Debian packages, but not the C code built
# into the binary or the CUDA libraries copied from wheels. Trivy's documentation names no way to
# embed a bill of materials in an image, so those are listed in vendored-c-libraries.cdx.json and
# scanned beside the image:
#   trivy image --scanners vuln --severity HIGH,CRITICAL --exit-code 1 <image>
#   trivy sbom --severity HIGH,CRITICAL --exit-code 1 docker/frame-engine-native/vendored-c-libraries.cdx.json
# A version bump below changes that file in the same commit.

# -- Stage 1: the CUDA libraries ---------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS wheels

# The three nvidia-* pins are read from the Python engine's requirements, not copied: one copy.
COPY docker/frame-engine/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -E '^nvidia-(cuda-nvrtc|cuda-runtime|nvjpeg)==' /tmp/requirements.txt > /tmp/cuda.txt \
    && test "$(wc -l < /tmp/cuda.txt)" -eq 3 \
    && pip download --no-deps --only-binary=:all: -r /tmp/cuda.txt -d /wheels \
    && mkdir /unpacked \
    && for wheel in /wheels/*.whl; do python -m zipfile -e "$wheel" /unpacked; done \
    && mkdir /cuda-libs \
    && cp -a /unpacked/nvidia/cu13/lib/. /cuda-libs/

# -- Stage 2: libjpeg-turbo and the engine ------------------------------------------------------
FROM rust:1.99.1-slim-trixie@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7 AS build

ARG LIBJPEG_TURBO_VERSION=3.1.2
# SHA-256 of libjpeg-turbo-3.1.2.tar.gz from
# https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.2/ (read 2026-11-15)
ARG LIBJPEG_TURBO_SHA256=8f0012234b464ce50890c490f18194f913a7b1f4e6a03d6644179fa0f867d0cf

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake curl g++ make nasm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src
RUN curl -fsSL -o libjpeg-turbo.tar.gz \
        "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" \
    && printf '%s  %s\n' "${LIBJPEG_TURBO_SHA256}" libjpeg-turbo.tar.gz > libjpeg-turbo.sha256 \
    && sha256sum -c libjpeg-turbo.sha256 \
    && tar -xzf libjpeg-turbo.tar.gz \
    && cmake -S "libjpeg-turbo-${LIBJPEG_TURBO_VERSION}" -B /opt/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/libjpeg-turbo -DCMAKE_INSTALL_LIBDIR=lib \
        -DENABLE_SHARED=0 -DENABLE_STATIC=1 -DWITH_TURBOJPEG=1 -DREQUIRE_SIMD=1 \
        -DCMAKE_POSITION_INDEPENDENT_CODE=1 \
    && cmake --build /opt/build --parallel "$(nproc)" \
    && cmake --install /opt/build \
    && rm -rf /opt/src /opt/build

# The crate links the build above, not a copy of its own; the toolchain is the base image's, and
# naming it keeps rustup from fetching the components rust-toolchain.toml lists for development.
ENV TURBOJPEG_SOURCE=explicit \
    TURBOJPEG_LIB_DIR=/opt/libjpeg-turbo/lib \
    TURBOJPEG_INCLUDE_DIR=/opt/libjpeg-turbo/include \
    TURBOJPEG_STATIC=1 \
    RUSTUP_TOOLCHAIN=1.99.1

WORKDIR /repo
COPY native /repo/native
# the wire contract the engine embeds, from the named build context `contracts_json` (the root
# .dockerignore keeps contracts/ out of the main context; hadolint cannot know a build context)
# hadolint ignore=DL3022
COPY --from=contracts_json frame_engine_v1.json /repo/contracts/json/frame_engine_v1.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/repo/native/target \
    cargo build --release --locked --manifest-path native/Cargo.toml -p frame_engine \
    && install -D native/target/release/op-frame-engine /out/op-frame-engine

# -- Stage 3: runtime ---------------------------------------------------------------------------
FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2

# Debian point-release security fixes; nothing else is installed.
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && groupadd -r engine \
    && useradd -r -g engine -G video -u 1000 -m -s /usr/sbin/nologin engine

COPY --from=wheels /cuda-libs /opt/nvidia/lib
COPY --from=build /out/op-frame-engine /usr/local/bin/op-frame-engine
# The recogniser's dictionary belongs to the model build; the engine reads it at start
# (`ocr::ctc::DICTIONARY_PATH`) and refuses to start without it.
COPY models/paddleocr_rec_trt/ppocrv6_dict.txt /app/ocr/ppocrv6_dict.txt

# The engine opens NVRTC, the CUDA runtime and nvJPEG by name at run time.
ENV LD_LIBRARY_PATH=/opt/nvidia/lib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER engine

EXPOSE 8001 8002

# Ready means the model server answers and the region is registered: the running engine answers
# ModelReady (`op-frame-engine health` asks it on its own port).
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["op-frame-engine", "health"]

ENTRYPOINT ["op-frame-engine"]
CMD ["--pipeline", "gpu"]

ARG OP_BUILD_SHA=unknown
LABEL org.opencontainers.image.title="OpenProcessor Frame Engine (native)" \
      org.opencontainers.image.description="Decode once on the GPU, cut every model input, call the model server; Rust" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}
