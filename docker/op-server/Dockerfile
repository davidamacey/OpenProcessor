# =============================================================================
# OpenProcessor server (Rust): the `op-server` binary and its roles. The `api` role is the front on
# the API's port: it answers the routes Rust owns and forwards every other request to the Python
# API (docs/design/v070_rebuild/18_backend_rust_port_plan.md, 18a section 5 slice R1a). One image for
# every role, so the CPU side is one build.
# =============================================================================
# Build (from the repository root; BuildKit, for the cache mounts and the named build context):
#   docker compose build api              # the dev overlay names the context below
# or, without compose:
#   docker build -f docker/op-server/Dockerfile --build-context contracts_json=contracts/json .
#
# Two stages, each on a base pinned by digest (the same two as docker/frame-engine-native/Dockerfile,
# read from Docker Hub on 2026-11-15; a deliberate bump re-reads them):
#
#   build    `cargo build --release --locked` of the `op_server` package. The toolchain is the base
#            image's; naming it keeps rustup from fetching the components rust-toolchain.toml lists
#            for development.
#   runtime  Debian slim, a non-root user, the binary, ExifTool and Perl (the capture-metadata
#            reader of the ingest path, run by path as a `-stay_open` pool; the API image installs
#            the same Debian package). Nothing else: no shell tools beyond the base image.
# =============================================================================

# -- Stage 1: the binary ----------------------------------------------------------------------
FROM rust:1.99.1-slim-trixie@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7 AS build

ENV RUSTUP_TOOLCHAIN=1.99.1

WORKDIR /repo
COPY native /repo/native
# the retired-variable list the binary embeds, from the named build context `contracts_json` (the
# root .dockerignore keeps contracts/ out of the main context; hadolint cannot know a build context)
# hadolint ignore=DL3022
COPY --from=contracts_json retired_env.json /repo/contracts/json/retired_env.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/repo/native/target \
    cargo build --release --locked --manifest-path native/Cargo.toml -p op_server \
    && install -D native/target/release/op-server /out/op-server

# -- Stage 2: runtime -------------------------------------------------------------------------
FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2

# Debian point-release security fixes, then ExifTool (libimage-exiftool-perl brings Perl).
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && groupadd -r opserver \
    && useradd -r -g opserver -u 1000 -m -s /usr/sbin/nologin opserver

COPY --from=build /out/op-server /usr/local/bin/op-server

USER opserver

EXPOSE 8000

# Ready means the front answers on its own port; `op-server health` asks it (the image has no curl).
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["op-server", "health"]

ENTRYPOINT ["op-server"]
CMD ["api"]

ARG OP_BUILD_SHA=unknown
LABEL org.opencontainers.image.title="OpenProcessor Server" \
      org.opencontainers.image.description="The Rust front of the API: owns the routes ported to Rust, forwards the rest to the Python API" \
      org.opencontainers.image.vendor="OpenProcessor" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/davidamacey/OpenProcessor"
LABEL org.opencontainers.image.revision=${OP_BUILD_SHA}
