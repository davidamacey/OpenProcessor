# syntax=docker/dockerfile:1
#
# Hardened NVIDIA Triton Inference Server — CVE-scan-gate build.
# Base: latest NGC Triton (v2.70.0 / 26.06), which already fixes the Aug-2025
# HTTP RCE chain (CVE-2025-23310/23311/23317, fixed 25.07) and the Dec-2025
# DoS pair (CVE-2025-33201/33211, fixed 25.10).
#
# Hardening applied here (see docs/security/triton_deepstream_cve_remediation.md):
#   Tier 1  apt upgrade  -> pull noble-security fixes (curl, gnutls, perl, systemd, ...)
#   Tier 1  pip upgrade  -> starlette >= 1.3.1 (OpenAI-compatible frontend dep)
#   Tier 2  purge the compile-time toolchain (build-essential, *-dev headers,
#           linux-libc-dev, cuda-nvcc/crt) -> deletes the kernel-header CVEs
#           (16 CRITICAL + ~200 HIGH) from the scan because the package is gone.
#           trtexec (prebuilt) + libnvinfer + libnvrtc remain, so engine build
#           and serving still work.
#   Runtime non-root user, cleaned caches.
#
# Build:  docker build -f docker/hardened/triton/Dockerfile -t triton-hardened:26.06 .
# Scan:   trivy image --scanners vuln --severity CRITICAL,HIGH triton-hardened:26.06

ARG TRITON_TAG=26.06-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

# hadolint ignore=DL3008,DL3009,DL4006
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    # Disable NVIDIA/CUDA apt repos so `upgrade` patches ONLY Ubuntu OS packages.
    # NVIDIA's CUDA/TensorRT libs are pinned to this Triton release and must not
    # drift (upgrading them pulls GBs and breaks the tested backend versions).
    for f in /etc/apt/sources.list.d/*; do \
      [ -f "$f" ] || continue; \
      if grep -qiE 'nvidia|cuda|developer\.download\.nvidia' "$f" 2>/dev/null; then \
        mv "$f" "$f.disabled"; fi; \
    done; \
    apt-get update; \
    # Tier 1 — apply all available Ubuntu security updates in place
    apt-get -y upgrade; \
    # Tier 2 — remove the kernel-header package. EVERY CRITICAL + ~all HIGH in
    # the scan is `linux-libc-dev`; purging it makes apt cascade-remove the whole
    # compile-time toolchain that depends on it (libc6-dev -> build-essential,
    # g++, clang, cuda-nvcc, all *-dev headers). This is a targeted purge, NOT
    # `autoremove --purge` (which removed a runtime .so and broke the binary).
    # Runtime libs (libc6, libstdc++6, libpython3.12, libnvinfer, libnvrtc, ...)
    # are untouched, so serving AND trtexec engine builds keep working.
    apt-get -y purge linux-libc-dev; \
    # Tier 2 — remove Nsight Systems/Compute profilers (~600MB). They are dev-only
    # tooling (not used to serve) and ship a Go binary (efa_metrics/nic_sampler)
    # built on Go stdlib 1.26.1 that carries the remaining 11 HIGH CVEs. Explicit
    # names match this pinned base; the rm is a belt-and-suspenders fallback.
    apt-get -y purge nsight-systems-cli-2026.3.1 nsight-compute-2026.2.0 || true; \
    rm -rf /usr/local/cuda-*/NsightSystems-cli-* /usr/local/cuda-*/nsight-compute-* \
           /opt/nvidia/nsight-compute* 2>/dev/null || true; \
    # Tier 1 — patch flagged Python deps. starlette>=1.3.1 fixes CVE-2026-48818 /
    # CVE-2026-54283 but needs a matching fastapi (>=0.139) or the pin breaks; upgrade
    # the pair together. These back only Triton's optional OpenAI frontend, not core
    # KServe v2 serving, so this is safe.
    python3 -m pip install --no-cache-dir --upgrade "starlette>=1.3.1" "fastapi>=0.139.0"; \
    # re-enable NVIDIA/CUDA repos so downstream images can install if needed
    for f in /etc/apt/sources.list.d/*.disabled; do \
      [ -f "$f" ] && mv "$f" "${f%.disabled}" || true; \
    done; \
    # cleanup
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/*; \
    # smoke test: fail the build if the purge stripped a runtime shared library
    # (a missing .so is the failure mode). No unresolved libs => the binary loads.
    if ldd "$(command -v tritonserver)" | grep -q 'not found'; then echo 'SMOKE FAIL: tritonserver libs'; exit 1; fi; \
    if ldd "$(command -v trtexec)" | grep -q 'not found'; then echo 'SMOKE FAIL: trtexec libs'; exit 1; fi; \
    trtexec --help > /dev/null 2>&1 && echo "smoke: tritonserver + trtexec shared libs resolved"

# Non-root runtime (26.06 keeps /opt/tritonserver root-owned, triton-server user
# reads it via group/other perms). Model repos are mounted read-only at runtime.
USER triton-server

EXPOSE 8000 8001 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/v2/health/ready || exit 1

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--model-control-mode=explicit", "--strict-readiness=true"]
