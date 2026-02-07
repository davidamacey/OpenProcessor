# Multi-stage Dockerfile for rust-preprocess service
#
# Builder stage: rust:bookworm with build tools
# Runtime stage: debian:bookworm-slim with minimal runtime deps

# ============================================================================
# Builder Stage
# ============================================================================
FROM rust:1.84-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    nasm \
    libjpeg62-turbo-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifests first for layer caching
COPY Cargo.toml Cargo.lock* build.rs ./
COPY proto/ ./proto/

# Create dummy main.rs to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source code
COPY src/ ./src/

# Build release binary with full optimizations
RUN cargo build --release --bin rust-preprocess

# ============================================================================
# Runtime Stage
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libjpeg62-turbo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/rust-preprocess /app/rust-preprocess

# Runtime config via environment variables
ENV TRITON_URL=triton-server:8001
ENV OPENSEARCH_URL=http://opensearch:9200
ENV PORT=8000
ENV GRPC_POOL_SIZE=4
ENV RUST_LOG=rust_preprocess=info

# Expose service port
EXPOSE 8000

# Run the service
CMD ["/app/rust-preprocess"]
