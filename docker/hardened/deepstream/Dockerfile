# syntax=docker/dockerfile:1
#
# Hardened NVIDIA DeepStream 9.0 (Triton-enabled) — CVE-scan-gate build.
# Base: nvcr.io/nvidia/deepstream:9.0-triton-multiarch  (DS 9.0.0, Ubuntu 24.04,
# bundles Triton 2.60 / NGC 26.01 for the nvinferserver plugin).
#
# DeepStream has NO product-specific CVE bulletin; its HIGH/CRITICAL are inherited
# from the Ubuntu base, CUDA/TensorRT, the bundled Triton python env, and GStreamer.
# Hardening (see docs/security/triton_deepstream_cve_remediation.md):
#   Tier 1  apt upgrade      -> Ubuntu OS security fixes (kernel headers, curl, ...)
#   Tier 1  pip upgrade      -> starlette/fastapi, wheel, jaraco.context (bundled Triton)
#   Tier 2  purge linux-libc-dev -> cascade-removes compile-time toolchain (kernel CVEs)
#   Tier 2  purge Nsight     -> removes the Go profiler binary (stdlib CVEs), ~600MB
#   Runtime non-root user + cleaned caches.
#
# IMPORTANT (registry): do NOT run gst-inspect during build. Without a GPU the NVIDIA
# GStreamer plugins fail to load (no libcuda.so.1) and get BLACKLISTED into the baked
# registry cache, which makes nvstreammux "fail to create element" at runtime. We
# instead delete any registry cache so GStreamer rescans fresh (with GPU) at runtime.
#
# RESIDUAL / CANNOT-FIX-YET: gst-plugins-bad CVE-2025-3887 (H.265 parser RCE) has NO
# Ubuntu fix as of this build. gst-plugins-bad is required by DeepStream and cannot be
# removed, so this HIGH is handled by VEX (docker/hardened/deepstream/trivyignore.txt).
#
# Build:  docker build -f docker/hardened/deepstream/Dockerfile -t deepstream-hardened:9.0 docker/hardened/deepstream
# Scan:   trivy image --scanners vuln --severity CRITICAL,HIGH --timeout 40m \
#           --ignorefile docker/hardened/deepstream/trivyignore.txt deepstream-hardened:9.0

ARG DS_TAG=9.0-triton-multiarch
FROM nvcr.io/nvidia/deepstream:${DS_TAG}

# hadolint ignore=DL3008,DL3009,DL4006
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    # Patch Ubuntu OS packages only; keep NVIDIA/CUDA/TensorRT pinned to the DS release.
    for f in /etc/apt/sources.list.d/*; do \
      [ -f "$f" ] || continue; \
      if grep -qiE 'nvidia|cuda|developer\.download\.nvidia' "$f" 2>/dev/null; then \
        mv "$f" "$f.disabled"; fi; \
    done; \
    apt-get update; \
    apt-get -y upgrade; \
    # Tier 2 — drop kernel headers + build toolchain (cascade via linux-libc-dev).
    # Confirmed safe: only *-dev/toolchain + DOCA-DPDK *dev* SDKs go; nvstreammux and
    # nvinfer do not link them, so live inference keeps working.
    apt-get -y purge linux-libc-dev; \
    # Tier 2 — remove Nsight Systems/Compute profilers (Go binary -> stdlib CVEs).
    apt-get -y purge nsight-systems-cli-2025.6.1 nsight-compute-2025.6.1 || true; \
    rm -rf /usr/local/cuda-*/NsightSystems-cli-* /usr/local/cuda-*/nsight-compute-* \
           /opt/nvidia/nsight-compute* /opt/nvidia/nsight-systems* 2>/dev/null || true; \
    # Tier 1 — patch flagged Python runtime deps (clean pip upgrade; these have
    # RECORD files). wheel/jaraco.context are vendored inside setuptools (build
    # tooling, not runtime-reachable) and cannot be cleanly bumped -> VEX'd instead.
    python3 -m pip install --no-cache-dir --upgrade "starlette>=1.3.1" "fastapi>=0.139.0" || true; \
    # re-enable NVIDIA/CUDA repos for downstream installs
    for f in /etc/apt/sources.list.d/*.disabled; do \
      [ -f "$f" ] && mv "$f" "${f%.disabled}" || true; \
    done; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/*; \
    # CRITICAL: purge any GStreamer registry cache so it is rebuilt fresh (with GPU)
    # at runtime — otherwise blacklisted NVIDIA plugins break the pipeline.
    rm -rf /root/.cache/gstreamer-1.0 /home/*/.cache/gstreamer-1.0 \
           /var/cache/gstreamer-1.0 2>/dev/null || true; \
    # smoke test: deepstream-app must resolve its shared libs (NO gst-inspect here)
    echo "gst-plugins-bad: $(dpkg-query -W -f='${Version}' gstreamer1.0-plugins-bad 2>/dev/null || echo n/a)"; \
    if ldd "$(command -v deepstream-app)" | grep -q 'not found'; then echo 'SMOKE FAIL: deepstream-app libs'; exit 1; fi; \
    echo "smoke: deepstream-app shared libs resolved"

# Non-root runtime. Engine caches / output must go to a writable, mounted dir.
RUN useradd -m -u 1001 dsuser || true
USER dsuser

WORKDIR /opt/nvidia/deepstream/deepstream
ENTRYPOINT ["deepstream-app"]
CMD ["--version-all"]
