# Desktop app: the end-goal direction

Status: direction only, recorded 2026-11-13 at the owner's request. Nothing here is built now. It is reviewed
after P2c and becomes its own plan then. Goal G6 in [00](00_photo_library_plan.md).

## 1. The goal

A standalone desktop app, like a self-contained Immich but more capable, that a person installs on their own
computer. It runs the same library features (organise, People, curate, create) and, where the hardware allows,
model training and segmentation, using the machine's own GPU or Apple chip. The core is written in Rust for
speed and a small footprint.

- **Platforms:** Linux, Windows and macOS; a reduced tier for small machines (mini PC, Raspberry Pi class).
- **Hardware:** NVIDIA GPUs, Apple Silicon, other GPUs where a runtime supports them, and CPU-only.
- **Apple extras, optional:** on macOS the app may use Apple's on-device Foundation Models for text tasks such as
  titles and captions. The app must work fully without them, so every such feature also has a local or template
  path that runs on Linux and Windows.
- **Capability tiers:** heavy features (training, segmentation, large VLMs) may be scaled down or switched off on
  small hardware. They are not removed from the product.

Today the product is the server: Linux with NVIDIA GPUs (Triton, OpenSearch, Postgres, Celery). The engine work
(P1) and phases P2a-0 to P2c come first.

## 2. Building blocks to research when this starts (candidates, not decisions)

Each needs its official docs read and a measured prototype before it is chosen.

| Need | Candidates |
|---|---|
| App shell reusing the Svelte UI | [Tauri 2](https://v2.tauri.app/) (Rust core, system web view) |
| Model runtime on any GPU or CPU | [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/) execution providers (CUDA, TensorRT, CoreML, DirectML, OpenVINO, CPU), from Rust via the [`ort`](https://ort.pyke.io/) crate |
| Local database and vectors | SQLite with a vector extension or an embedded vector store (for example sqlite-vec, LanceDB, USearch) |
| Media | FFmpeg; ExifTool (a Perl program, bundled) or a Rust metadata crate if its coverage is proven equal |
| Apple text model | Apple's Foundation Models framework (Swift) through a small bridge, macOS only |

## 3. What today's work keeps possible (no extra code now)

The no-dead-code and no-shim rules hold: nothing is built for the desktop ahead of time. These habits keep the
door open at no cost:

1. **The wire contract is the boundary.** Cropwright talks only to the generated contract, so a Rust backend that
   implements the same contract can host the same UI.
2. **ONNX is the source artifact for every model.** Engines for Triton are built from ONNX; the same files feed
   ONNX Runtime later.
3. **Domain rules stay pure and tested** (`src/domain/`). Their tests and fixtures are the specification a Rust
   port must pass.
4. **No server-only detail leaks to the wire** (for example OpenSearch query syntax or index names).
5. **Each phase records its hardware needs** (model sizes, GPU memory, time per photo), so tiers can be set later
   from measured numbers.
