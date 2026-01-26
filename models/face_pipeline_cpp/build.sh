#!/bin/bash
# Build script for Face Pipeline C++ Backend
#
# Usage:
#   ./build.sh           # Build inside Docker
#   ./build.sh install   # Build and install to Triton

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building Face Pipeline C++ Backend ==="

# Check if we're inside Triton container
if [ -f /opt/tritonserver/bin/tritonserver ]; then
    echo "Building inside Triton container..."

    # Clone backend repo if not exists
    if [ ! -d /opt/triton-backend ]; then
        git clone --depth 1 --branch r24.01 \
            https://github.com/triton-inference-server/backend.git \
            /opt/triton-backend
    fi

    export TRITON_BACKEND_REPO=/opt/triton-backend

    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)

    if [ "$1" == "install" ]; then
        echo "Installing to /opt/tritonserver/backends/face_pipeline/"
        mkdir -p /opt/tritonserver/backends/face_pipeline
        cp libtriton_face_pipeline.so /opt/tritonserver/backends/face_pipeline/
        echo "Installed successfully!"
    fi

else
    echo "Building using Docker..."

    # Build the builder image
    docker build -f Dockerfile.build -t face-pipeline-builder .

    # Run the build
    docker run --rm \
        -v "$(pwd)":/workspace \
        -w /workspace \
        face-pipeline-builder \
        bash -c "
            if [ ! -d /opt/triton-backend ]; then
                git clone --depth 1 --branch r24.01 \
                    https://github.com/triton-inference-server/backend.git \
                    /opt/triton-backend
            fi
            export TRITON_BACKEND_REPO=/opt/triton-backend
            mkdir -p build && cd build && cmake .. && make -j\$(nproc)
            cp libtriton_face_pipeline.so ../
        "

    if [ -f libtriton_face_pipeline.so ]; then
        echo "Build successful: libtriton_face_pipeline.so"
        echo ""
        echo "To install, copy to Triton container:"
        echo "  docker cp libtriton_face_pipeline.so triton-api:/opt/tritonserver/backends/face_pipeline/"
    else
        echo "Build failed!"
        exit 1
    fi
fi

echo "=== Build Complete ==="
