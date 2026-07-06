# Hardened Triton + DeepStream containers (CVE-gated, functionally verified)

Independent, latest-version NVIDIA **Triton Inference Server** and **DeepStream** container
builds hardened to pass a **CRITICAL/HIGH CVE scan gate** while still doing real work
(load models → build TensorRT engines → serve/infer). Companion analysis:
[`../../docs/security/triton_deepstream_cve_remediation.md`](../../docs/security/triton_deepstream_cve_remediation.md)
and [`../../docs/security/triton_deepstream_hardening_plan.md`](../../docs/security/triton_deepstream_hardening_plan.md).

## Results

| Image | Base scan (CRIT/HIGH) | Hardened (CRIT/HIGH) | Functional test |
|---|---|---|---|
| `triton-hardened:26.06` | 16 / 214 | **0 / 0** (no VEX) | ✅ trtexec engine build + ONNX & TensorRT serving + inference on A6000 |
| `deepstream-hardened:9.0` | 1 / 19 | raw **0 / 4** → **0 / 0** gated (3 CVEs VEX'd) | ✅ headless detector pipeline, ~951 FPS on A6000 |

- **Triton reaches 0/0 with zero waivers** — every finding was fixed properly (patched or the
  unused component removed), nothing suppressed.
- **DeepStream reaches 0/0 in the gate**; 4 raw HIGH (3 distinct CVEs) are waived with
  justification in [`deepstream/trivyignore.txt`](deepstream/trivyignore.txt):
  `CVE-2025-3887` (gst-plugins-bad H.265 RCE — **no Ubuntu fix yet**, plugin can't be removed) and
  `CVE-2026-24049` / `CVE-2026-23949` (wheel / jaraco.context — **build-only tooling vendored in
  setuptools**, not in the serving path). These are the documented "cannot-fix-yet" items.

## Layout

```
docker/hardened/
  triton/Dockerfile              # FROM tritonserver:26.06-py3
  deepstream/Dockerfile          # FROM deepstream:9.0-triton-multiarch
  deepstream/trivyignore.txt     # VEX for the single unfixable HIGH (CVE-2025-3887)
  test/
    make_test_model.py           # tiny ONNX classifier via repo .venv (torch/onnx)
    infer_check.py               # KServe v2 HTTP inference + numeric check
    build_scan.sh                # docker build + trivy CRITICAL/HIGH summary
    test_triton.sh               # engine build + serve + infer (Triton)
    test_deepstream.sh           # headless detector pipeline (DeepStream)
    model_repository/            # generated Triton test models (onnx + trt)
```

## What the hardening does (both images)

1. **Patch Ubuntu only** — disable the NVIDIA/CUDA apt repos, then `apt-get upgrade`, so OS
   packages (curl, gnutls, systemd, kernel headers…) are patched but NVIDIA's CUDA/TensorRT
   stay pinned to the release (upgrading them pulls GBs and breaks tested backends).
2. **Remove the compile-time toolchain** — `apt-get purge linux-libc-dev` cascade-removes
   build-essential/gcc/g++/clang/cuda-nvcc and every `*-dev` header. This deletes the
   kernel-header CVEs (the bulk of CRITICAL/HIGH) because the package is gone. `trtexec`,
   `libnvinfer`, `libnvrtc` remain, so engine builds + serving still work.
3. **Remove Nsight profilers** — dev-only tooling shipping a Go binary (`efa_metrics/nic_sampler`)
   whose Go `stdlib` carried the rest of the HIGH/CRITICAL. Not used to serve.
4. **Upgrade flagged Python deps** — `starlette`+`fastapi` (pair-upgrade), `wheel`, `jaraco.context`.
5. **DeepStream only** — delete the GStreamer registry cache so plugins rescan **with GPU** at
   runtime (building without a GPU blacklists the NVIDIA plugins → broken pipeline).
6. Runtime **non-root** user, cleaned apt/pip caches.

## Build + scan

```bash
# Triton
bash docker/hardened/test/build_scan.sh \
  docker/hardened/triton/Dockerfile docker/hardened/triton triton-hardened:26.06

# DeepStream (with VEX file for the one unfixable HIGH)
docker build -f docker/hardened/deepstream/Dockerfile -t deepstream-hardened:9.0 docker/hardened/deepstream
trivy image --scanners vuln --severity CRITICAL,HIGH --timeout 40m \
  --ignorefile docker/hardened/deepstream/trivyignore.txt deepstream-hardened:9.0
```

> Trivy needs `--timeout 30m`+ on these 14–20 GB images (default 5 min times out).

## Functional tests (GPU required — used A6000, `CUDA_VISIBLE_DEVICES`/`device=2`)

```bash
# Triton: ONNX + trtexec-built TensorRT plan, served, HTTP inference verified
bash docker/hardened/test/test_triton.sh triton-hardened:26.06 2

# DeepStream: decode sample video -> nvinfer detector (engine built from ONNX) -> fakesink
bash docker/hardened/test/test_deepstream.sh deepstream-hardened:9.0 2
```

## Run: independently and together (live stream)

```bash
# Triton standalone (KServe v2 on 8000/8001; keep off untrusted networks)
docker run -d --gpus '"device=2"' -p8000:8000 -p8001:8001 \
  -v $PWD/models:/models:ro triton-hardened:26.06 \
  --model-repository=/models --model-control-mode=none

# DeepStream standalone (nvinfer in-pipeline)
docker run --rm --gpus '"device=2"' deepstream-hardened:9.0 -c <config>

# Together: DeepStream nvinferserver -> Triton
#   in-process : deepstream-app -c samples/configs/deepstream-app-triton/<cfg>
#   gRPC       : set   infer_config { backend { triton { grpc { url: "triton:8001" } } } }
```

## Host-side (out of image scope, but part of the gate)

NVIDIA Container Toolkit ≥ 1.17.8 + current driver (DS 9.0 recommends ≥590; the functional
tests here ran on 580 via CUDA forward-compat). NVIDIAScape (CVE-2025-23266) is host-side.
