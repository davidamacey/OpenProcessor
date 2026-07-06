# Hardened Triton + DeepStream Containers — Build / Scan / Test Plan

**Goal:** ship independent, latest-version Triton and DeepStream container images that
(a) pass a **0 un-waived CRITICAL/HIGH** CVE gate and (b) still **load models, build TensorRT
engines, and serve inference** (Triton) / **run a live-stream inference pipeline** (DeepStream),
verified with real example data on an A6000 (`CUDA_VISIBLE_DEVICES=2`).

Companion CVE analysis: [`triton_deepstream_cve_remediation.md`](./triton_deepstream_cve_remediation.md).

## Layout

```
docker/hardened/
  triton/Dockerfile          # FROM nvcr.io/nvidia/tritonserver:26.06-py3  (independent build)
  deepstream/Dockerfile      # FROM nvcr.io/nvidia/deepstream:<latest>     (independent build)
  test/
    make_test_model.py       # generate a small ONNX classifier via repo .venv (torch 2.10 / onnx 1.19)
    build_scan.sh            # docker build + trivy CRITICAL/HIGH scan helper
    test_triton.sh           # ONNX serve + trtexec engine build + TensorRT serve + HTTP inference
    test_deepstream.sh       # sample RTSP/file pipeline, optional nvinferserver->Triton
  README.md
```

## Hardening recipe (both images)

- **Tier 1 — patch:** `apt-get update && apt-get -y upgrade` (pulls noble-security fixes:
  linux-libc-dev 6.8.0-124→134, curl, gnutls, perl, systemd, …). `pip install -U starlette`.
- **Tier 2 — remove build surface:** purge `build-essential linux-libc-dev libc6-dev g++/gcc/cpp
  clang cuda-nvcc cuda-crt` in the final image (they are compile-time only). This deletes the
  16 CRITICAL + ~200 HIGH kernel-header findings because the package is gone. Verify runtime,
  engine build (`trtexec`), and serving still work afterward.
- **Tier 3 — VEX:** any residual kernel-header CVE with no upstream fix → OpenVEX /
  `.trivyignore` (`vulnerable_code_not_present`, container uses host kernel).
- **Runtime hardening:** non-root `USER`, clean apt/pip caches, keep only runtime libs.
- **DeepStream extra:** ensure `gst-plugins-bad ≥ 1.26.3` (H.265/H.266 RCEs); patch CUDA/TRT.

## Iteration loop (per image)

1. `docker build` → 2. `trivy image --scanners vuln --severity CRITICAL,HIGH` →
3. inspect residual findings → 4. patch/purge/VEX → 5. rebuild until threshold →
6. functional test on A6000 → 7. record results.

## Acceptance criteria

- **Triton:** trivy CRITICAL/HIGH = 0 (excluding signed-off VEX); container serves an ONNX model
  AND a `trtexec`-built TensorRT `.plan`; an HTTP inference request returns correct output shape.
- **DeepStream:** trivy CRITICAL/HIGH = 0 (excluding VEX); a sample video pipeline runs to
  completion producing detections; `nvinferserver` can target Triton (in-proc or gRPC).

## Host-side (out of image scope, noted for the gate)

NVIDIA Container Toolkit ≥ 1.17.8, current GPU driver — the escape-class CVEs
(NVIDIAScape CVE-2025-23266) are host-side and not fixed by rebuilding the image.
