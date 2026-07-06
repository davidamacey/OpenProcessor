# Triton Inference Server + DeepStream — CVE Posture & Scan-Pass Remediation

**Prepared:** 2026-07-05 · **Scope:** NVIDIA Triton Inference Server and DeepStream SDK
container security for a company deployment with a "no unresolved HIGH/CRITICAL" scan gate.
**Evidence:** live Trivy 0.67.1 scan of `nvcr.io/nvidia/tritonserver:26.06-py3` + in-container
`apt` inspection (Ubuntu 24.04.4). Sources listed at the end.

---

## 1. The three CVEs that got `25.04-py3-min` denied

All three are from **NVIDIA's August 2025 Triton security bulletin (answer ID 5687)**, all in the
**HTTP request-handling** path (unsafe `alloca()` stack allocation driven by attacker-controlled
chunked-transfer input via libevent `evbuffer_peek`). Unauthenticated by default → remote code
execution / DoS / info-leak. They can be chained (Wiz "ModelWeasel" chain, CVE-2025-23319) into
full server takeover with no credentials.

| CVE | Type (CWE) | CVSS | Fixed in |
|---|---|---|---|
| **CVE-2025-23310** | Stack buffer overflow, HTTP handler | **9.8 Critical** | **Triton 25.07** |
| **CVE-2025-23311** | Stack-based buffer overflow (CWE-121), affects ≤25.06 | **9.8 Critical** | **Triton 25.07** |
| **CVE-2025-23317** | HTTP-server RCE (reverse shell via crafted request) | **9.1 Critical** | **Triton 25.07** |

**Fixable? → Yes, and already fixed.** `25.04` was correctly denied — it predates the fix.
Any build **≥ 25.07** clears all three. They are Triton *application* CVEs (NVIDIA source), so the
only remediation is upgrading the Triton version; there is no config workaround.

The same bulletin covers **CVE-2025-23310 … 23323** (OOB writes, improper validation) — all fixed in
25.07. A later **December 2025 bulletin (ID 5734)** added two HIGH (7.5) DoS bugs, **CVE-2025-33201 /
CVE-2025-33211**, fixed in **25.10**. **Net: run ≥ 25.10; the current release 26.06 covers everything
disclosed through Dec 2025.**

## 2. Latest version

**Triton 26.06** = product **v2.70.0**, Ubuntu 24.04 / Python 3.12 / CUDA 13.3 / TensorRT 11.0 /
vLLM 0.22.1. NVIDIA ships **monthly** (`YY.MM`) on NGC — always check for a newer tag, but 26.06 has
**zero outstanding Triton-application CVEs** as of this report. Variants: `-py3` (full), `-py3-min`
(minimal base for custom builds — **use this for production**), `-py3-sdk` (clients),
`-pyt-python-py3` / `-trtllm-python-py3` (trimmed backends).

## 3. Why the latest, fully-patched image still "fails" a naive scan

Trivy on the official `26.06-py3` reports **16 CRITICAL + 214 HIGH** — yet **none are Triton**:

| Source package | Findings | Nature |
|---|---|---|
| **`linux-libc-dev`** (kernel headers) | **16 CRITICAL + ~200 HIGH** | Ubuntu kernel CVEs — **not exploitable in a container** (containers use the *host* kernel; the header package is not running code) |
| `starlette` 0.49.3 | 2 HIGH | Real userspace — fix ≥ 1.3.1 |
| Go `stdlib` v1.26.1 | ~10 HIGH | Real userspace — fix ≥ Go 1.26.4 |

The kernel-header noise is a known industry problem: since Feb 2024, kernel.org assigns ~120 CVEs/month
to *any* bugfix, and scanners map them onto `linux-libc-dev`. NVIDIA documents the same for its GPU
Operator images ("known high CVEs from base images, not in libraries the software uses").

## 4. Can the base-image CVEs be fixed *properly*? — Yes. Three stackable tiers.

Verified against the live 26.06 image (`noble-security` has fixes staged):

**Tier 1 — Patch (proper fix).** `apt-get update && apt-get upgrade` pulls **78 upgradable packages**,
including security updates for `linux-libc-dev` (**6.8.0-124 → 6.8.0-134**, clears the CVEs marked
`fix=6.8.0-134.134`), `curl`, `libgnutls30`, `perl`, `systemd`. `pip install -U` for `starlette`
(≥1.3.1) and rebuild/drop Go binaries (Go ≥1.26.4). This genuinely lowers the count.

**Tier 2 — Remove the attack surface (biggest lever).** `linux-libc-dev` is only a **build-time**
dependency (`libc6-dev → build-essential`, g++, clang, cuda-nvcc). A runtime image doesn't need it.
Start from **`26.06-py3-min`** (or a multi-stage build) and purge the compiler toolchain in the final
stage → **all 16 CRITICAL + ~200 HIGH kernel-header findings disappear from the scan** because the
package is gone. This is the single most effective action for a "zero HIGH/CRITICAL" gate.

```dockerfile
FROM nvcr.io/nvidia/tritonserver:26.06-py3-min AS runtime
RUN apt-get update && apt-get -y upgrade \
 && apt-get -y purge build-essential linux-libc-dev libc6-dev g++ g++-13 cpp-13 clang clang-18 \
      cuda-nvcc-* cuda-crt-* 2>/dev/null || true \
 && apt-get -y autoremove --purge \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade "starlette>=1.3.1"
USER triton-server            # 26.06 already makes /opt/tritonserver root-owned + non-root user
```

**Tier 3 — VEX the irreducible tail.** A handful of 2026 kernel CVEs have **no fix from Ubuntu yet**
(Trivy `fix=` empty) — impossible for *anyone* to patch. If Tier 2 didn't already remove the package,
document these as **VEX "NOT AFFECTED / vulnerable_code_not_present"** (container uses host kernel).
This is the standard, accepted path; NVIDIA ships the **Vulnerability-Analysis NIM Agent Blueprint**
to auto-generate these justifications (analyst review required). Load into your gate as a
`.trivyignore` / Grype ignore rule / OpenVEX doc.

> **If the company rule allows *no* exceptions:** only Tiers 1+2 satisfy it — patch what's patchable
> and *remove* the rest. A pure "patch-only, zero-waiver" rule is technically unsatisfiable on any
> NVIDIA (or Ubuntu) image while upstream kernel fixes lag, so pair the rule with an approved VEX
> process. Keep the host's NVIDIA Container Toolkit **≥ 1.17.8** and GPU driver current — those
> escape-class CVEs (NVIDIAScape CVE-2025-23266, 9.0) are host-side and *not* fixed by rebuilding the image.

## 5. DeepStream — latest version + CVE posture

- **Latest: DeepStream 9.0** (supersedes 8.0 from Sep 2025); both support Ubuntu 24.04 / Blackwell /
  Jetson Thor. Containers on NGC: base, **`-triton`** (bundles Triton as inference backend), `-samples`.
- **No DeepStream-SDK-specific CVE bulletin exists.** DeepStream's HIGH/CRITICAL findings are **inherited
  from its layers**, and the fix approach is identical to §4:
  - **GStreamer** — the real remote attack surface (it parses untrusted RTSP/video). Recent RCEs:
    **CVE-2025-3887** (H.265, stack overflow), **CVE-2025-6663** (H.266, fixed in `gst-plugins-bad`
    **1.26.3**), plus 2026 batch CVE-2026-2920/2921/2923/3082/3083/3085 (8.8). **Ensure
    `gst-plugins-bad ≥ 1.26.3`.**
  - **CUDA / TensorRT / cuDNN** — baked into the image; patch to the versions in NVIDIA's CUDA bulletins.
  - **Base OS** — same Tier 1/2/3 treatment as Triton.
  - **Host — NVIDIA Container Toolkit** ≥ 1.17.8 (CVE-2025-23266/23267, CVE-2024-0132/CVE-2025-23359).

## 6. Running Triton + DeepStream — independent and together (live stream)

**Independent.** Triton = standalone inference microservice (`8000` HTTP / `8001` gRPC / `8002`
metrics). DeepStream = separate GStreamer video-analytics pipeline. Neither requires the other.

**Together** (DeepStream's `Gst-nvinferserver` plugin uses Triton as its backend):

- **CAPI / in-process** — the `deepstream:9.0-triton` container embeds Triton libs; `nvinferserver`
  calls Triton in-process. Lowest latency, one container. Best for a single edge box.
- **gRPC / remote** — `nvinferserver { grpc { url: "triton:8001" } }` connects to a standalone Triton.
  Decouples scaling (one Triton serves many DeepStream instances + other clients). Best for a cluster.

**Live-stream pipeline:** `RTSP camera → nvurisrcbin (NVDEC HW decode) → nvstreammux (batch) →
nvinferserver (→ Triton) → nvtracker → nvdsosd/encode → RTSP/WebRTC/Kafka out`.

**Security for the "together" path:** the Aug-2025 Triton CVEs are unauthenticated HTTP — so **never
expose 8000/8001 to untrusted networks.** Keep Triton on a private/overlay network or behind an
authenticating reverse proxy (Triton has no built-in auth); prefer mTLS on the gRPC link; run with
`--model-control-mode=none` and disable unused shared-memory/repo endpoints to shrink the surface.

## 7. Scan-gate checklist

1. **Upgrade Triton ≥ 26.06** (kills 23310/23311/23317 + all Aug/Dec-2025 app CVEs). — *the denial fix*
2. Build from **`-py3-min`**, `apt upgrade`, **purge the build toolchain** (removes kernel-header CVEs).
3. `pip -U starlette (≥1.3.1)`; rebuild/drop Go binaries (Go ≥1.26.4).
4. DeepStream: `gst-plugins-bad ≥ 1.26.3`; patch CUDA/TensorRT; DS **9.0**.
5. **Host:** Container Toolkit ≥ 1.17.8 + current GPU driver.
6. **VEX** the residual unpatchable kernel-header CVEs (`vulnerable_code_not_present`); load as
   `.trivyignore`/OpenVEX with analyst sign-off.
7. Run **non-root**, read-only rootfs, dropped caps, seccomp; keep inference ports off public networks.
8. Re-scan (`trivy image --scanners vuln --severity CRITICAL,HIGH`) → target **0 un-VEX'd HIGH/CRITICAL**.

---

### Sources
- NVIDIA Triton Security Bulletin — August 2025 (23310/23311/23317, fixed 25.07): https://nvidia.custhelp.com/app/answers/detail/a_id/5687
- NVIDIA Triton Security Bulletin — December 2025 (33201/33211, fixed 25.10): https://nvidia.custhelp.com/app/answers/detail/a_id/5734
- Wiz — CVE-2025-23319 Triton takeover chain: https://www.wiz.io/blog/nvidia-triton-cve-2025-23319-vuln-chain-to-ai-server
- The Hacker News — Triton unauthenticated RCE: https://thehackernews.com/2025/08/nvidia-triton-bugs-let-unauthenticated.html
- ZeroPath — 23310 / 23311 / 23317 technical summaries: https://zeropath.com/blog/cve-2025-23317-nvidia-triton-inference-server-rce-summary
- Triton 26.06 release notes (v2.70.0): https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-06.html
- Triton NGC catalog (image variants): https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
- Trivy issue #1596 — kernel CVEs / `linux-libc-dev` in containers: https://github.com/aquasecurity/trivy/issues/1596
- Ubuntu — kernel CVE volume & `linux-libc-dev`: https://bugs.launchpad.net/bugs/2083312
- NVIDIA Vulnerability-Analysis Blueprint (VEX automation): https://github.com/NVIDIA-AI-Blueprints/vulnerability-analysis
- DeepStream 9.0 docs / release notes: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html
- Gst-nvinferserver (Triton backend for DeepStream): https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html
- GStreamer H.266 RCE CVE-2025-6663 (fixed 1.26.3): https://zeropath.com/blog/gstreamer-h266-cve-2025-6663-buffer-overflow
- Wiz — NVIDIAScape CVE-2025-23266 (Container Toolkit): https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape

---

# Part 2 — Implementation results (hardened images built, scanned & tested 2026-07-06)

Two independent hardened images were built, scanned with Trivy 0.67.1, and functionally
verified on an RTX A6000 (`device=2`, driver 580.159 via CUDA forward-compat). Artifacts live
under [`docker/hardened/`](../../docker/hardened/) (Dockerfiles, test harness, VEX file, README).

## Scan-gate results (CRITICAL / HIGH)

| Image | Base image | Hardened (raw) | Gated (with VEX) | Waivers |
|---|---|---|---|---|
| `triton-hardened:26.06` (FROM `tritonserver:26.06-py3`) | **16 / 214** | **0 / 0** | **0 / 0** | none |
| `deepstream-hardened:9.0` (FROM `deepstream:9.0-triton-multiarch`) | **1 / 19** | **0 / 4** | **0 / 0** | 3 CVEs, documented |

## How it was done (both images)

1. **Ubuntu-only patching** — disable the NVIDIA/CUDA apt repos, then `apt-get upgrade`. Patches
   OS packages (curl, gnutls, systemd, perl, kernel headers) while keeping CUDA/TensorRT pinned to
   the vendor release. (A naive `apt upgrade` with NVIDIA repos on pulls ~6.5 GB and drifts
   TensorRT 11.0→11.1, breaking the tested backend — this was hit and corrected.)
2. **`apt-get purge linux-libc-dev`** — one surgical purge whose dependency cascade removes the
   entire compile-time toolchain (build-essential, gcc/g++, clang, cuda-nvcc, all `*-dev` headers).
   This deletes **every** kernel-header CVE (the bulk of CRITICAL/HIGH) because the package is gone.
   `trtexec`, `libnvinfer`, `libnvrtc` remain → engine builds + serving still work. (A blanket
   `autoremove --purge` was tried first and stripped a runtime `.so` → reverted to the surgical form.)
3. **Remove Nsight Systems/Compute** — dev-only profilers shipping a Go binary
   (`efa_metrics/nic_sampler`) whose Go `stdlib` carried the residual CRITICAL/HIGH. Not used to serve.
4. **`pip upgrade starlette + fastapi`** (as a pair — starlette≥1.3.1 needs fastapi≥0.139) — backs
   only the optional OpenAI frontend, not core KServe serving.
5. **DeepStream-only:** delete the GStreamer registry cache so plugins rescan **with the GPU** at
   runtime. (Root cause of a hard failure: running `gst-inspect` at build time — no GPU — blacklisted
   the NVIDIA plugins into the baked registry, so `nvstreammux` "failed to create element" at runtime.)

## What was fixed *properly* (no waiver)

- **All kernel-header CRITICAL/HIGH** (`linux-libc-dev`, 16 CRIT + ~200 HIGH on Triton) — removed.
- **Go `stdlib` CRITICAL/HIGH** (Nsight `nic_sampler`) — removed.
- **`starlette` / `fastapi`** HIGH — upgraded to fixed versions.
- **OS package HIGH** (curl, gnutls, systemd, perl, …) — patched via Ubuntu security updates.
- Triton finishes at a clean **0/0 with no exceptions at all.**

## What could NOT be fixed — remediation backlog (review later)

These are DeepStream-only and are the reason it needs 3 documented VEX entries
([`docker/hardened/deepstream/trivyignore.txt`](../../docker/hardened/deepstream/trivyignore.txt)):

| CVE | Package | Why it can't be fixed now | Exposure | Action to close |
|---|---|---|---|---|
| **CVE-2025-3887** (HIGH) | `gstreamer1.0-plugins-bad` / `libgstreamer-plugins-bad1.0-0` 1.24.2-1ubuntu4 | **No Ubuntu 24.04 fix published yet** (upstream fix only in 1.26.3). gst-plugins-bad is a hard DeepStream runtime dependency — can't remove it. | H.265 parser RCE — reachable **only if decoding attacker-supplied H.265/HEVC**. | Watch Ubuntu USN for `1.24.2-1ubuntu4.x`; drop the waiver + `apt upgrade` when it ships. Interim: restrict ingest to trusted cameras / prefer H.264. |
| **CVE-2026-24049** (HIGH) | `wheel` (vendored in `setuptools/_vendor` + stale debian `python3-wheel`) | setuptools pins the vendored copy; debian `python3-wheel` can't be pip-uninstalled (no RECORD) nor apt-purged (pip depends on it). | **Build/packaging tooling only** — not loaded by deepstream-app / nvinfer / Triton serving; no untrusted input reaches it. | Drop when a base image ships patched setuptools/wheel. |
| **CVE-2026-23949** (HIGH) | `jaraco.context` (vendored in `setuptools/_vendor`) | Same — vendored inside setuptools, not independently upgradable. | Same — build tooling, not runtime-reachable. | Same as above. |

**Also noted (accepted trade-offs, not CVEs):**
- The `linux-libc-dev` purge on DeepStream also removes the **DOCA-DPDK dev SDKs**. Confirmed *not*
  used by standard video-inference pipelines (`nvstreammux`/`nvinfer` don't link them). **Keep them
  (skip the purge) if you use BlueField DPU / GPUDirect-RDMA networking.**
- **Driver:** DS 9.0 recommends ≥590; these tests ran on **580.159** via forward-compat. Bump the host
  driver before production.
- **Host-side (unchanged, still required for the gate):** NVIDIA Container Toolkit ≥1.17.8 + current
  driver — NVIDIAScape (CVE-2025-23266) is host-side and not fixable by rebuilding the image.

## Functional verification (proves hardening didn't break serving)

- **Triton** — generated a tiny ONNX model, built a TensorRT `.plan` with `trtexec` *inside the
  hardened image*, served both (onnxruntime + tensorrt_plan), and ran KServe v2 HTTP inference:
  both returned correct shape, numerically matching the torch reference (max|Δ|≈2e-8). **PASS.**
- **DeepStream** — headless `deepstream-app`: decoded the sample video → `nvinfer` resnet18 detector
  (**TensorRT engine built + serialized from the ONNX on first run**) → ~**951 FPS**, "App run
  successful". **PASS.**

Reproduce: `docker/hardened/test/test_triton.sh` and `test_deepstream.sh` (see the README).
