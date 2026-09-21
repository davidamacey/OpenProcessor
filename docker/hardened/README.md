# Hardened Triton container (CVE-gated, functionally verified)

An independent, latest-version NVIDIA **Triton Inference Server** build hardened to pass a
**CRITICAL/HIGH CVE scan gate** while still doing real work (load models → build TensorRT
engines → serve/infer). This is a stricter, opt-in alternative to the production
[`Dockerfile.triton`](../../Dockerfile.triton) — that file already applies Tier 1 (`apt upgrade`,
Nsight purge, `starlette` upgrade); this variant additionally applies Tier 2 (purge the entire
compile-time toolchain) for a deployment with a strict "zero unwaived CRITICAL/HIGH" scan gate.

Companion analysis: [`../../docs/security/triton_cve_hardening.md`](../../docs/security/triton_cve_hardening.md).

Note: an earlier version of this directory also hardened a DeepStream image. DeepStream is not
part of this product — that half of the work now lives in the `run_deepstream` project, which
built on this same methodology and has since gone further (per-image baseline scans, remediation
log). See that project's `docs/security/cve-remediation.md`.

## Results

| Image | Base scan (CRIT/HIGH) | Hardened (CRIT/HIGH) | Functional test |
|---|---|---|---|
| `triton-hardened:26.06` | 16 / 214 | **0 / 0** (no VEX) | ✅ trtexec engine build + ONNX & TensorRT serving + inference on A6000 |

Triton reaches 0/0 with zero waivers — every finding was fixed properly (patched, or the unused
component removed), nothing suppressed.

## Layout

```
docker/hardened/
  triton/Dockerfile          # FROM tritonserver:26.06-py3
  test/
    make_test_model.py       # tiny ONNX classifier via repo .venv (torch/onnx)
    infer_check.py           # KServe v2 HTTP inference + numeric check
    build_scan.sh            # docker build + trivy CRITICAL/HIGH summary
    test_triton.sh           # engine build + serve + infer
    model_repository/        # generated Triton test models (onnx + trt) -- gitignored
```

## What the hardening does

1. **Patch Ubuntu only** — disable the NVIDIA/CUDA apt repos, then `apt-get upgrade`, so OS
   packages (curl, gnutls, systemd, kernel headers…) are patched but NVIDIA's CUDA/TensorRT
   stay pinned to the release (upgrading them pulls GBs and breaks tested backends).
2. **Remove the compile-time toolchain** — `apt-get purge linux-libc-dev` cascade-removes
   build-essential/gcc/g++/clang/cuda-nvcc and every `*-dev` header. This deletes the
   kernel-header CVEs (the bulk of CRITICAL/HIGH) because the package is gone. `trtexec`,
   `libnvinfer`, `libnvrtc` remain, so engine builds + serving still work.
3. **Remove Nsight profilers** — dev-only tooling shipping a Go binary (`efa_metrics/nic_sampler`)
   whose Go `stdlib` carried the rest of the HIGH/CRITICAL. Not used to serve.
4. **Upgrade flagged Python deps** — `starlette`+`fastapi` (pair-upgrade).
5. Runtime **non-root** user, cleaned apt/pip caches.

## Build + scan

```bash
bash docker/hardened/test/build_scan.sh \
  docker/hardened/triton/Dockerfile docker/hardened/triton triton-hardened:26.06
```

> Trivy needs `--timeout 30m`+ on these 14–20 GB images (default 5 min times out).

## Functional test (GPU required — used A6000, `CUDA_VISIBLE_DEVICES`/`device=2`)

```bash
bash docker/hardened/test/test_triton.sh triton-hardened:26.06 2
```

## Run

```bash
docker run -d --gpus '"device=2"' -p8000:8000 -p8001:8001 \
  -v $PWD/models:/models:ro triton-hardened:26.06 \
  --model-repository=/models --model-control-mode=none
```

Keep Triton's ports off untrusted networks regardless of image hardening — it has no built-in
auth (see the CVE doc's §6 on the unauthenticated-HTTP CVE class).

## Host-side (out of image scope, but part of the gate)

NVIDIA Container Toolkit ≥ 1.17.8 + current GPU driver. NVIDIAScape (CVE-2025-23266) is
host-side and not fixed by rebuilding the image.
