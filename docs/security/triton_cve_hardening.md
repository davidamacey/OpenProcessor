# Triton Inference Server — CVE Posture & Scan-Pass Remediation

**Scope:** NVIDIA Triton Inference Server container security for a deployment with a
"no unresolved HIGH/CRITICAL" scan gate. Trimmed from an earlier combined Triton+DeepStream
analysis — DeepStream is not part of this product; that half now lives in the `run_deepstream`
project, which built on the same methodology (see its `docs/security/cve-remediation.md`).

**Evidence:** live Trivy 0.67.1 scan of `nvcr.io/nvidia/tritonserver:26.06-py3` + in-container
`apt` inspection (Ubuntu 24.04.4). Sources listed at the end.

---

## 1. The three CVEs that got an old `-min` image denied

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

**Fixable? → Yes, and already fixed.** They are Triton *application* CVEs (NVIDIA source), so the
only remediation is upgrading the Triton version; there is no config workaround.

The same bulletin covers **CVE-2025-23310 … 23323** (OOB writes, improper validation) — all fixed
in 25.07. A later **December 2025 bulletin (ID 5734)** added two HIGH (7.5) DoS bugs,
**CVE-2025-33201 / CVE-2025-33211**, fixed in **25.10**. **Net: run ≥ 25.10.**

## 2. Latest version (as of this writing)

**Triton 26.06** = product **v2.70.0**, Ubuntu 24.04 / Python 3.12 / CUDA 13.3 / TensorRT 11.0 /
vLLM 0.22.1. NVIDIA ships **monthly** (`YY.MM`) on NGC — always check for a newer tag. 26.06 has
**zero outstanding Triton-application CVEs** as of this report. Variants: `-py3` (full), `-py3-min`
(minimal base for custom builds — use this for production), `-py3-sdk` (clients).

## 3. Why the latest, fully-patched image still "fails" a naive scan

Trivy on the official `26.06-py3` reports **16 CRITICAL + 214 HIGH** — yet **none are Triton**:

| Source package | Findings | Nature |
|---|---|---|
| **`linux-libc-dev`** (kernel headers) | **16 CRITICAL + ~200 HIGH** | Ubuntu kernel CVEs — **not exploitable in a container** (containers use the *host* kernel; the header package is not running code) |
| `starlette` 0.49.3 | 2 HIGH | Real userspace — fix ≥ 1.3.1 |
| Go `stdlib` v1.26.1 | ~10 HIGH | Real userspace (Nsight profiler's bundled binaries) — remove the profiler |

The kernel-header noise is a known industry problem: since Feb 2024, kernel.org assigns ~120
CVEs/month to *any* bugfix, and scanners map them onto `linux-libc-dev`. NVIDIA documents the
same for its GPU Operator images ("known high CVEs from base images, not in libraries the
software uses").

## 4. Can the base-image CVEs be fixed *properly*? — Yes. Three stackable tiers.

**Tier 1 — Patch (proper fix).** `apt-get update && apt-get upgrade` pulls the available
`noble-security` fixes, including `linux-libc-dev`, `curl`, `libgnutls30`, `perl`, `systemd`.
`pip install -U starlette` (≥1.3.1, needs a matching `fastapi` ≥0.139 or the pin breaks). This is
already applied in this repo's production [`Dockerfile.triton`](../../Dockerfile.triton).

**Tier 2 — Remove the attack surface (biggest lever).** `linux-libc-dev` is only a **build-time**
dependency (`libc6-dev → build-essential`, g++, clang, cuda-nvcc). A runtime image doesn't need
it. Purge the compiler toolchain in the final stage → **all 16 CRITICAL + ~200 HIGH kernel-header
findings disappear from the scan** because the package is gone. This is the single most effective
action for a "zero HIGH/CRITICAL" gate — **not yet applied to the production Dockerfile** (it
changes the build surface more invasively than Tier 1); see
[`docker/hardened/triton/Dockerfile`](../../docker/hardened/triton/Dockerfile) for a verified,
opt-in build that does this. Also remove the Nsight Systems/Compute profilers (~600MB dev-only
tooling whose bundled Go binaries carry the remaining HIGH CVEs) — this part *is* already applied
in the production Dockerfile.

```dockerfile
RUN apt-get update && apt-get -y upgrade \
 && apt-get -y purge build-essential linux-libc-dev libc6-dev g++ g++-13 cpp-13 clang clang-18 \
      cuda-nvcc-* cuda-crt-* 2>/dev/null || true \
 && apt-get -y autoremove --purge \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade "starlette>=1.3.1"
USER triton-server
```

**Tier 3 — VEX the irreducible tail.** A handful of kernel CVEs may have **no fix from Ubuntu yet**
(Trivy `fix=` empty) — impossible for anyone to patch. If Tier 2 didn't already remove the
package, document these as **VEX "NOT AFFECTED / vulnerable_code_not_present"** (container uses
host kernel), loaded into the scan gate as a `.trivyignore` / OpenVEX doc.

> **If the deployment rule allows *no* exceptions:** only Tiers 1+2 satisfy it — patch what's
> patchable and *remove* the rest. A pure "patch-only, zero-waiver" rule is technically
> unsatisfiable on any NVIDIA (or Ubuntu) image while upstream kernel fixes lag, so pair the rule
> with an approved VEX process. Keep the host's NVIDIA Container Toolkit **≥ 1.17.8** and GPU
> driver current — escape-class CVEs (NVIDIAScape CVE-2025-23266, 9.0) are host-side and *not*
> fixed by rebuilding the image.

## 5. Scan-gate checklist

1. **Upgrade Triton ≥ 26.06** (kills 23310/23311/23317 + all Aug/Dec-2025 app CVEs).
2. Build from `-py3-min` or purge the build toolchain from `-py3` in the final stage (removes
   kernel-header CVEs) — see `docker/hardened/triton/Dockerfile` for the verified 0/0 build.
3. `pip -U starlette` (≥1.3.1).
4. **Host:** Container Toolkit ≥ 1.17.8 + current GPU driver.
5. **VEX** any residual unpatchable kernel-header CVEs (`vulnerable_code_not_present`); load as
   `.trivyignore`/OpenVEX with analyst sign-off.
6. Run **non-root**, read-only rootfs, dropped caps, seccomp; keep inference ports off public
   networks.
7. Re-scan (`trivy image --scanners vuln --severity CRITICAL,HIGH`) → target **0 un-VEX'd
   HIGH/CRITICAL**.

## 6. Unauthenticated HTTP — a deployment note, not a scan finding

Triton has **no built-in authentication**. Independent of any CVE, `8000`/`8001` must never be
exposed to untrusted networks — keep Triton on a private/overlay network or behind an
authenticating reverse proxy, prefer mTLS on the gRPC link, run with
`--model-control-mode=none`, and disable unused shared-memory/repo endpoints to shrink the
surface.

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
