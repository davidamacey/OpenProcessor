# 17: the native frame engine (Rust) behind protocol version 1, and its first spike

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15), written for the coordinator; not reviewed yet |
| Code base read | `int/b4-stateless` at `a9e16498d` (read only): `frame_engine/`, `src/clients/frame_engine.py`, `src/domain/frame_engine_wire.py`, `tests/frame_engine_tests/`, `tests/gpu/`, `docker/frame-engine/`, the `frame-engine` compose service |
| Reads with | [14](14_roadmap.md) section 4 (how a port is judged), [13](13_backend_speed_plan.md) D1 and the protocol, [13b](13b_crop_vectors_plan.md) section 6, [the serving trade study](../../research/v060_serving_backend_trade_study.md) sections 5 and 10, [the desktop direction](../photo_library/50_desktop_direction.md) |
| Replaces | track T1 of plan 13 (raised to a build); the order of plan 14 row 7 ("after phase 3") is overridden by the owner: the native engine starts now |
| Owner decisions in force | 2026-11-15: no further speed work goes into the Python engine; it is frozen as the reference for correct outputs until the native engine replaces it. Two engines are never kept alive after the switch |
| Not in scope | the API, jobs, storage and training stay Python; the model server is kept in the first steps (section 1.5) |
| Numbers | every number names its set and run; the raw outputs live outside the tracked tree and are cited in words (input-pipeline spike, engine host-cost lane, engine CPU lane, pooled-crops lane, configuration D refresh and its failure diagnosis, all of 2026-11-14/15). Set P is private: numbers only |

## 0. The user's job, and what the native engine changes for it

A person adds photos. Each photo is decoded once on the GPU; the detector, the frame encoder,
the crop encoder, faces, text and thumbnails all run from that one frame; the person then labels
by cluster and trains a model. The measure is photos per second from "photo in" to "stored and
findable". The frame engine is the part that touches pixels, and it is now the measured limit.
The limit is the Python process, not the card:

| Evidence (one RTX A6000, engine alone, 32 photos in flight unless stated) | Number |
|---|---|
| Python engine, today's plan (detect, frame vector, a crop-encoder call per crop, crop JPEGs), 20 MP | 26 to 33 photos/s, lock held in about 80% of samples; 46 to 48 after three host-cost fixes, card then 94 to 96% busy |
| Python engine, crop cost taken away (pooled-crop lane) | COCO 68.8 to 73.9, 20 MP 46.6 to 48.2 photos/s, card 55 to 60% busy; lock held in 0.78 of samples, 42% of it by the one network loop thread parsing 8 to 10 MB requests |
| The engine's own decode and resample code in a harness, one process, with the detector and frame encoder attached, no crops | COCO 187 to 192, 20 MP 122 to 125 (set P 127) photos/s |
| Models alone, inputs already on the device | 196 to 203 photos/s, card 97% busy |
| Input pipeline only (no models), one process against four | COCO 650 against 3,398 (5.2x); 20 MP 174 to 200 against 222 to 244 |
| Host CPU per photo, Python engine | 141 to 220 ms through the API rows; 57 ms per small photo engine alone; 44 to 46 (COCO) and 178 to 184 (20 MP) in the pooled rows, of which the crop JPEG encodes are 61 to 63 |
| JPEG entropy decoding | on host cores in every library on this card (no JPEG engine on GA102; the video decoder does not decode JPEG): 40 to 100 ms of CPU per 20 MP photo in nvJPEG's host phase; the host's arithmetic limit is about 280 to 330 photos/s at 20 MP on 24 cores |

So decode capacity is cores and must be spread over all of them, and the per-photo control work
(request parse, kernel launches, call building) must cost microseconds, not milliseconds under a
global lock. A native process does both in one process.

### 0.1 Targets per plan (one A6000, engine alone, 32 and 64 in flight)

GPU time per photo, from the measurements above: frame models about 5.0 ms (1000 / 196 to 203);
decode and resample kernels about 0.15 ms on a COCO photo and about 2.5 ms on a 20 MP photo; the
crop encoder at its 224 input (13b section 6: 2.1 times the 336 rate of 225 to 247 per second)
about 2.0 ms per crop; 4.67 crops per COCO photo at policy `all` (plan 13 section 2; 6.5 in a
200-photo sample) and 2.7 to 3.55 per set P photo; faces with tiles 15.1 ms (13.4 detector, 1.7
recogniser) and without tiles 3.1 ms per 20 MP photo; OCR about 9 ms per photo before batching (an
estimate from O1b, not an engine row).

| Plan | Branches | Arithmetic ceiling | Target (native) | Python today |
|---|---|---|---|---|
| F, frame only | `detect`, `embed_frame`, crops `none` | COCO about 194, 20 MP about 133; measured harness 187 to 192 and 122 to 127 | COCO at least 175 (0.9 of the models-alone row), 20 MP at least 110 (0.9 of the harness row) | not measured as such (N0 measures it); the pooled rows bound it near 70 and 47 |
| C, with crops at the crop encoder's cost | F plus `embed_crops` at 224, `crops: inline` | COCO about 69, set P about 68 | at least 60 on both (equals 13b's rate bar) | at today's 336 crops: COCO 25.9 to 26.1 (card 99%, encoder-bound), set P 46 to 48 (card 94 to 96%) |
| A, the full plan | C plus `faces`, `face_embed`, `ocr`, `derivatives` | 20 MP about 25 with face tiles, about 36 without | the card at least 90% busy and the rate at least 0.9 of 1000 / (GPU ms per photo from the model server's counters plus the engine's kernel time, same hold) | not measured |

Read plainly: a native engine wins on rate where the Python engine is bound by its own front
(plan F; plan C once the crop encoder runs at 224). Where the card is already the bound (today's
336-crop plan) it wins only on host CPU, latency and memory.

### 0.2 Adoption bars (plan 14 section 4's rule, per plan)

| Measure | Bar against the Python engine, same photos, same model server, same hold |
|---|---|
| Photos per second | higher by more than the run-to-run spread (two control runs of the Python engine in the hold; never less than 5%) on every plan where the Python engine is not GPU-bound; at least 0.97x where both are |
| Host CPU per photo, per process | at most 0.5x on COCO-sized photos, at most 0.75x on 20 MP (nvJPEG's host phase and the crop JPEG encodes are the same C code in both) |
| p50 and p99 at 32 and 64 in flight | p99 not higher; p50 not higher at equal rate |
| Memory | device memory idle and at 32 in flight not above the Python engine's at the same sizes (3,340 MiB idle at the 40 GB sizes; 7.4 GB at 16 threads and 10.6 to 11.8 GB at 32 threads under 20 MP load); host RSS not above 3.4 GB |
| Image size | below the Python image (786 MB) |
| Start time | container start to ready, model server already up: not above the Python engine's (measured in N0; no row exists) |
| Outputs | section 1.2, and the ground-truth gates of plan 13 per branch |
| Faults | section 1.3 |

## 1. What stays fixed

### 1.1 Protocol version 1 and every answer the API relies on

The API cannot tell the engines apart. The wire module `src/domain/frame_engine_wire.py` stays
the one definition; the native engine reads a contract generated from it (section 3, row
"Contract").

- One unary KServe v2 `ModelInfer` per photo to the model `frame_engine`, version `1`; inputs
  `image` (BYTES [1]), `plan` (BYTES [1], one JSON string), optional `boxes` (FP64 [n, 4]); every
  input in `raw_input_contents`; BYTES elements as a 4-byte little-endian length and the bytes; the
  outputs and datatypes of `OUTPUT_DATATYPES`, filtered to the request's `outputs` when it names
  any; `ServerLive`, `ServerReady`, `ModelReady`, `ModelMetadata` answered; every other KServe
  call `UNIMPLEMENTED`.
- Plan faults (`bad_json` ... `unused_boxes`) answer `INVALID_ARGUMENT` with the fault name first.
- Photo answers: `status` `ok` or one of `corrupt`, `huge`, `over_budget`, `gpu_memory`,
  `decoder`; `decode_route` `gpu`, `cpu_<reason>` (`not_jpeg`, `precision`, `components`,
  `progressive`, `truncated`, `small`) or `none`; `branch_status` one JSON object, one entry per
  reported branch: `ok`, `not_run`, or `error` with `unsupported_branch`, `model_error`,
  `dependency_failed`, `image_failed`; `size` once decoded.

The gRPC statuses, read from `src/clients/frame_engine.py` and `frame_engine/front.py`:

| Status | When | What the API does |
|---|---|---|
| `RESOURCE_EXHAUSTED` | overload: in-flight cap, frame budget wait, region (slab) wait, colour-table build wait | retries with jittered backoff until the photo's 120 s deadline; `overloaded` on the error |
| `UNAVAILABLE` | not ready, draining, model server unavailable, CUDA context lost | retries 3 times |
| `INTERNAL`, details starting `frame engine: ` | the engine's own failure for this photo | no retry (deterministic) |
| `INTERNAL` without that prefix | lost in transit (stream removed, RST_STREAM, flow control) | retries 3 times |
| `INVALID_ARGUMENT` | malformed request or plan | no retry |
| `NOT_FOUND` | unknown model | no retry |
| `DEADLINE_EXCEEDED` | still running at `OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS` (60 s, under the client's 120 s) | no retry |

Also fixed: readiness means the model server answers, the region is registered and the context is
not lost; the API probes `ModelReady` up to 3 times, 2 s each. The API pings every 30 s with no
call in flight (`ENGINE_CHANNEL_OPTIONS`); the Python server permits that explicitly
(`SERVER_OPTIONS`) because gRPC C-core sends GOAWAY `too_many_pings` otherwise (it did, configuration D,
two 503s). A tonic server enforces no ping policy at all (read from the h2 0.4.20 source: every
ping is answered, there is no strike counter), so the native front accepts the API's pings; the test that pins the two
sides changes from "equal options" to "the server accepts a ping every 30 s idle" (N2). Every
`OP_FRAME_ENGINE_*` setting keeps its name, default and range and still refuses a malformed value
at start. Every metric name and label of `frame_engine/stage_metrics.py` is kept (dashboards).

### 1.2 Outputs equal to the Python engine's

Measured two ways. At one photo in flight both engines send batch-1 calls, so the model server's
answers are bit-stable (the host-cost lane: the same photos one at a time gave bit-identical
vectors on both trees). At 32 in flight the dynamic batcher composes batches by timing, so the
floor is the Python engine against itself (minimum cosine 0.99995 frame, 0.99991 crop; the E1 row:
0.999889).

| Output | Equal means (1 in flight) | At 32 in flight | Source of the bar |
|---|---|---|---|
| `status`, `decode_route`, `branch_status`, `size`, `letterbox` | identical | identical | protocol; `LETTERBOX_ATOL` 1e-9 |
| Model input tensors (letterbox canvas, squares, aligned faces, text lines) | byte-identical (same kernel source, same NVRTC build, same host weights) | same | the Python engine's own E1 rule |
| `det_boxes`, `det_scores`, `det_classes` | identical | identical on at least 99% of photos; the rest inside the control-against-control spread | E1 row; TensorRT is not bit-stable across batch compositions |
| `whole_embedding`, `crop_embeddings` | bit-identical | cosine at least the control floor (0.9999) | host-cost lane section 9 |
| `crop_embedded`, `face_embedded`, overflow counts | identical | identical | |
| `blur_sums` | identical integers | identical | E1 row |
| `crop_jpegs` | same count and pixel sizes; decoded pixels identical; bytes identical if the libjpeg-turbo build and settings match OpenCV's, else sizes within 0.5% (question Q6) | same | E1 row asked for identical bytes |
| `derivative_jpegs` | bytes identical (same nvJPEG build, same parameters) | same | `tests/gpu/test_gpu_renditions.py` (PSNR 40 dB to Pillow stays) |
| Colour | the frame identical; each profile's table identical to littleCMS's (maximum error 0) | same | colour row of plan 13; `tests/gpu/test_gpu_colour.py` |
| Host decode route | JPEG frames identical (same libjpeg-turbo, accurate DCT, fancy upsampling as simplejpeg); lossless formats identical; lossy WebP within 1 level | same | input-pipeline spike section 5 |
| `face_*` | boxes within 0.01 px of the Python engine; recogniser cosine at least 0.9999 | `BOX_ATOL_PX` 1.0 and cosine p01 at least 0.995 against the reference path | `tests/gpu/test_gpu_faces.py`, plan 13 Q2 |
| `ocr_*` | texts identical on the GPU-tier set; on COCO-Text 400 photos strict F1 not below 0.238 and spotted words not below 0.495 | same | `tests/gpu/test_gpu_ocr.py`; the `ocr` branch row of plan 13 (ground truth, plan 14 section 3) |

### 1.3 The fault rule

A decode or prep fault must not take the models down. Kept by the same means: a separate process
from the model server; the kill test (killed at least 50 times at 32 in flight; the model server
stays up, logs no CUDA error, returns to its memory baseline, serves the next request); region
names with a fixed prefix purged at start; unregister before free at stop; a lost CUDA context
(the sticky codes 700, 702, 710, 714 to 719, plus 214, whose documentation carries the same
"terminate and relaunch" sentence) refuses work with `UNAVAILABLE`, turns readiness false and exits
non-zero for the restart policy. New for Rust: a panic in one photo's work fails that photo
(`INTERNAL` with the prefix) and the process goes on; the photo's device memory is released by its
owner's drop after a stream wait.

### 1.4 GPU profiles

Every profile of `config_templates/profiles/` keeps working, including a 12 GB card (the RTX 3080
Ti here). Today the 12 to 24 GB profiles run the Python engine at its smallest sizes (16 MiB
region, 64 MiB budget, one worker) and decode on the CPU; plan 13 item E4 asks the engine to be the
image path on every profile. The native engine owns E4 (N2): sizes per profile from measured
memory, a frame budget for at least one 50 MP photo, the 12 GB card serving the full plan.

### 1.5 The model server stays in the first steps

Triton over CUDA shared memory stays until the switch, for four reasons with evidence:

1. The engine's bound is its front, not the model calls: models alone reach 196 to 203 photos/s
   and the harness with the same calls 187 to 192, while the Python engine reaches about 70.
2. Process separation is what meets the fault rule today (50 of 50 kills, E0 and E1 rows). An
   in-process runtime puts a decode fault and the models in one process, and every restart then
   pays the engines' load time.
3. The models are native TensorRT plans built by `export/` for the model server. ONNX Runtime's
   TensorRT provider builds or caches its own engines (its EP-context models); its documentation
   says nothing of loading a plan built outside it, so every model would need a second build path
   and a parity pass (trade study section 5.3, "M per model").
4. Other callers share the models (the API's text tower, the core routes), and the model server's
   statistics are the batch-size evidence every row uses.

An in-process runtime is decision 4.1, settled by measurement, never assumed.

## 2. Architecture

### 2.1 Processes and threads

One OS process per card (instance `fe0`), in its own container, beside the card's model server
(plan 13 D7). Inside it:

| Role | Execution unit | Count | Does | Never does |
|---|---|---|---|---|
| Network | tokio runtime, multi-thread | 4 worker threads (measured) | the tonic KServe front; the tonic client channels to the model server (`OP_FRAME_ENGINE_TRITON_CHANNELS`); one async control task per photo (it reads the header markers up to the frame header for size, orientation and profile, which the budget needs: microseconds); the health tick | pixel work, decode, JPEG encode, blocking waits |
| Decode | OS threads, each owning one nvJPEG lane (decoder state, pinned and device buffers, jpeg stream, CUDA stream) | `OP_FRAME_ENGINE_LANES`, swept against cores (decision 4.4) | the end-of-image scan (the last routing check), nvJPEG parse and host phase, transfer, device decode, orientation, colour gather, the round-1 resamples into their region slices, one stream wait | model calls |
| Launch | OS threads, each with its own CUDA stream | 4 (measured) | round-2 kernels (crop squares, face and text warps), blur sums, renditions; one stream wait each | decode |
| Finish | OS threads | `OP_FRAME_ENGINE_ENCODE_THREADS` | crop pixels copied to pinned host memory and JPEG-encoded (libjpeg-turbo), host post-processing (face anchors and NMS, text boxes, CTC), nvJPEG encodes of renditions | anything that feeds the card |
| Host decode | the decode threads | as above | images nvJPEG does not take (libjpeg-turbo, PNG, WebP and the rest), uploaded on the thread's stream | |

The tokio documentation's rule decides the split: CPU-bound work goes to a dedicated pool and its
result comes back to the async task through a oneshot channel; `spawn_blocking` is not used for
decode (its pool grows to 512 threads by default). One owner per CUDA stream and per nvJPEG state,
as the nvJPEG thread-safety rules require (the library handle and the decoders are shared).

### 2.2 Stages, queues and their bounds

```text
gRPC request ──► admission ──► frame budget ──► decode queue ──► decode thread ──► round-1 calls ─┐
 (Bytes, no      in-flight      FIFO bytes,      bounded by       nvJPEG, orient,    over CUDA      │
  copy)          cap: else      bounded wait:    the in-flight    colour, round-1    shared memory  │
                 RESOURCE_      else RESOURCE_   cap              resamples, sync    (tonic client) │
                 EXHAUSTED      EXHAUSTED                                                           ▼
 response ◄── finish (outputs, release) ◄── finish pool (crop JPEGs, ◄── round-2 calls ◄── launch pool
                                             blur, post-process)        (crop squares,     (round-2 kernels,
                                                                         faces, text)       sync)
```

| Wait | Bound | Answer when exceeded |
|---|---|---|
| Admission | `OP_FRAME_ENGINE_MAX_IN_FLIGHT` photos admitted (running or queued) | `RESOURCE_EXHAUSTED` at once |
| Frame budget | `OP_FRAME_ENGINE_BUDGET_MIB`, first come first served (a tokio semaphore is fair; permits in KiB), bytes read from the header before any decode (frame, its oriented copy, a rendition's scratch: `prep_budget.py`'s arithmetic) | `RESOURCE_EXHAUSTED` after `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`; a photo larger than the whole budget answers `over_budget` |
| Region slices | first fit with coalescing in the one registered region (`slab.py`'s allocator), taken by the async task before work is dispatched, so a pool thread never waits for room; a photo holds at most its fair share with calls in flight | `RESOURCE_EXHAUSTED` after `OP_FRAME_ENGINE_SLAB_WAIT_SECONDS` |
| Model calls | each call `min(OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS, what is left of the request)`, as the gRPC deadline and a tokio timeout; dropping the future cancels the stream | `UNAVAILABLE`, with an immediate health probe; readiness follows the probe, never one call |
| Request | `OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS` | `DEADLINE_EXCEEDED`, logged and counted with its stage |

### 2.3 Memory ownership, network buffer to device

1. The request arrives in tonic's per-message buffer; with prost's `bytes` option every
   `raw_input_contents` element is a reference-counted slice of it (one copy from the HTTP/2
   frames into that buffer, none after).
2. The decode thread hands nvJPEG the slice's address and the length up to the end-of-image
   marker; the slice is dropped as soon as the decode returns (10 MB freed early).
3. nvJPEG writes through the lane's pinned and device buffers into the frame: one device
   allocation per photo from a stream-ordered pool whose release threshold is
   `OP_FRAME_ENGINE_POOL_MIB`, charged to the budget. The frame has one owner (the photo's
   state); pool jobs borrow it; it is freed only after every stream that touched it was waited for
   (each pool job waits for its own stream before it returns, the rule of `engine.py` made a type).
4. Model inputs are written by kernels into slices of the one region: a single `cudaMalloc` made at
   start, exported with `cudaIpcGetMemHandle` (the documentation requires the base of a
   `cudaMalloc` allocation, so no pool slice is exported), registered once with the 64-byte handle
   as `raw_handle`, addressed by offset. A slice returns only after its call ended and its writer's
   stream was waited for.
5. Answers arrive as gRPC messages; crop pixels come back once through pinned staging per finish
   thread; JPEG bytes become `Bytes` in the response without a copy.

### 2.4 Back-pressure, admission, overlap and prefetch

The three bounds above are the only queues; each answers with a status, none waits without a
limit. Decoding runs ahead continuously: while photo k waits for its calls, photos k+1 and on
decode on other lanes, limited only by admission and the budget (frames waiting for the models
are bounded by the budget). Round-2 kernels of one photo overlap other photos' decodes on other
streams; crop JPEG encodes overlap the round-2 calls; host post-processing never runs on a thread
that feeds the card. Batches form in the model server's dynamic batcher from inputs already
resident; an engine-side batcher across photos is decision 4.6.

### 2.5 The checklist (mechanisms that keep host and card busy together)

| # | Mechanism | Where in this design |
|---|---|---|
| K1 | Admission by count and by decoded bytes, read from the header, never an unbounded wait | 2.2 |
| K2 | Decode runs ahead continuously and hands over single photos, not batches | decode pool |
| K3 | One decode feeds every model input on the device; only the encoded file crosses the bus in, only small results come out | 2.3 |
| K4 | Batches form only from inputs already resident; no batch waits for a decode | model server batcher; 4.6 |
| K5 | Concurrent GPU decodes are a bounded pool of lanes, each with its own state and stream; the lane count is a measured knob | decode pool, 4.4 |
| K6 | Frames waiting for the models are bounded; a deep queue is a working-set fault | budget |
| K7 | A partial batch leaves on a timer that new arrivals do not extend; the mean batch is reported per row | model server config (P2), 4.6 |
| K8 | No large allocation per photo on the hot path: persistent region, stream-ordered frame pool, reused pinned staging, weights cached per size | 2.3 |
| K9 | Per-photo host work (JPEG encode, blur reduce, post-processing, response build) off the threads that feed the card | finish pool |
| K10 | Thread counts per role; the decode pool as wide as the cores the card needs | 2.1 |
| K11 | One owner per stream, per nvJPEG state, per model session | 2.1 |
| K12 | No process-wide lock on the request path | Rust; atomics, per-lane state, a fair semaphore |

## 3. Building blocks

Licences are recorded at selection (plan 14 section 3). Versions are the latest on 2026-11-15;
N0 pins exact versions in `Cargo.lock`.

| Need | Choice | Alternative | Why | Doc | Licence (last release) |
|---|---|---|---|---|---|
| Language and toolchain | Rust stable, pinned in `native/rust-toolchain.toml` (1.99.1 is installed on the host) | C++ | owner: Rust preferred; a later desktop build; ownership types enforce the stream-wait-before-free rules the Python code enforces by comment | https://rust-lang.github.io/rustup/overrides.html | MIT / Apache-2.0 |
| CUDA driver, streams, events, module loading, NVRTC | `cudarc` (raw `sys` calls for IPC: `cuIpcGetMemHandle` has no safe wrapper) | `cust` / Rust-CUDA (last crates.io release 2022; project "in early development"); hand FFI for everything (ortloom's way) | maintained, covers the driver API and NVRTC with run-time loading; frames from CUDA's stream-ordered allocator | https://docs.rs/cudarc ; https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html ; https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator ; https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html (IPC) | MIT OR Apache-2.0 (0.19.10, 2026-09-24) |
| nvJPEG decode and encode | own FFI module loaded with `libloading`: the about 30 symbols the Python lanes call through ctypes, same flags (`GPU_HYBRID` then `HYBRID`, RGBI output, interpolated chroma) | `baracuda-nvjpeg` (alpha), `nvjpeg-sys` (2024, no repository) | no maintained crate; the symbol set and the call order are already proven in `nvjpeg_lanes.py` and `nvjpeg_encode.py` | https://docs.nvidia.com/cuda/nvjpeg/index.html (thread safety 1.3, backends 2.2.1, decoupled phases 2.3.4) | `libloading` ISC (0.9.1, 2026-11-13); nvJPEG under the CUDA EULA |
| CUDA libraries in the image | the same NVIDIA wheel builds the Python image pins (NVRTC 13.2.86, nvJPEG 13.2.3.58, runtime 13.2.86), unpacked in a build stage | distribution CUDA packages | the same NVRTC build gives the same PTX, the same nvJPEG the same pixels | `docker/frame-engine/Dockerfile` table | CUDA EULA |
| Kernels | the existing CUDA C sources, moved once out of the Python strings into `native/frame_engine/kernels/*.cu`; Python reads the files (`cp.RawModule(code=...)`), Rust embeds them (`include_str!`) and compiles with NVRTC at start with CuPy 14.2.0's options; one new kernel (`orient.cu`, a pure permutation) replaces CuPy's transpose copy | CV-CUDA, NPP (different arithmetic: pixels drift) | one copy of the source for both engines during the transition; same source plus same compiler plus same options is the parity argument, proven by bytes | https://docs.nvidia.com/cuda/nvrtc/index.html ; https://github.com/cupy/cupy/blob/v14.2.0/cupy/cuda/compiler.py | repository licence (AGPL) |
| Host geometry and routing | a Rust port of `prep_geometry.py`, `src/domain/image_geometry.py` and `src/domain/jpeg_markers.py`, checked against golden cases generated from the Python functions (float32 weights compared bit for bit) | calling Python | pure functions; the goldens make drift a failing test | | AGPL |
| Contract | the wire module generates `contracts/json/frame_engine_v1.json` (tensor names, datatypes, shapes, statuses, plan faults, branch outputs, decode routes, the ping policy) and `contracts/json/frame_engine_plan_cases.json` (plan texts and their parse result or fault) through `scripts/codegen/export_api_contracts.py`; the `api-contracts-drift` hook checks them; Rust tests assert its constants and parser against both | a hand-kept second copy | the API keeps the Python wire module after the switch; the contract is the boundary, generated, never hand-edited | [contracts/README.md](../../../contracts/README.md) | AGPL |
| gRPC server and client | `tonic` + `prost` (generated from Triton's `grpc_service.proto` and `model_config.proto`, vendored at the model server's release tag); `bytes` fields as `Bytes`; `max_decoding_message_size` set from `OP_FRAME_ENGINE_MAX_MESSAGE_MIB` (tonic's default is 4 MB) | `grpcio` bindings; `triton-ng-client` (50 downloads), `triton-client` (2022) | one proto set serves the front and the client; maintained | https://docs.rs/tonic/0.14.9/tonic/server/struct.Grpc.html ; https://docs.rs/prost-build/latest/prost_build/struct.Config.html | tonic MIT (0.14.9, 2026-10-29); prost Apache-2.0 (0.14.6, 2026-11-12) |
| Async runtime and pools | `tokio` for the network; dedicated OS threads (`std::thread` plus `crossbeam-channel`) for decode, launch and finish | everything async; everything threads | tokio's own guidance for CPU-bound work; per-thread nvJPEG lanes need fixed threads | https://docs.rs/tokio/1.54.1/tokio/index.html (CPU-bound tasks and blocking code) | tokio MIT (1.54.1, 2026-10-21); crossbeam-channel MIT OR Apache-2.0 (0.5.17, 2026-09-05) |
| Host JPEG decode and crop JPEG encode | `turbojpeg` (libjpeg-turbo 3.1, vendored and linked statically, SIMD required); accurate DCT and fancy upsampling as simplejpeg; encoder at quality, 4:2:0, no optimisation as `cv2.imencode` | `zune-jpeg` (different IDCT: pixels drift); OpenCV bindings (a large native dependency) | the same library family the Python engine decodes and encodes with | https://docs.rs/turbojpeg ; https://libjpeg-turbo.org/Documentation/Documentation | Unlicense OR MIT (1.5.1, 2026-07-25); libjpeg-turbo BSD-style |
| Other host formats | `image` (PNG, WebP, GIF, TIFF, BMP), EXIF orientation from the port of `jpeg_markers.py` and the PNG/WebP chunks | Pillow through Python | lossless formats decode identically by definition; lossy WebP judged within 1 level | https://docs.rs/image | MIT OR Apache-2.0 (0.25.10, 2026-03-10) |
| Colour | `lcms2` (Little CMS 2): profile class by the tag rule of `src/utils/icc_srgb.py`, perceptual intent and no flags (Pillow's defaults), the 2^24-entry table gathered on the GPU by `colour.cu`; the host route uses the same GPU step instead of a host conversion | a 3D grid with interpolation (measured 8 to 32 levels off near the gamut clip) | one colour path for both routes; the table equals littleCMS's answer | https://docs.rs/lcms2 ; https://www.littlecms.com/ | MIT (6.2.0, 2026-08-26); Little CMS MIT (2.20, 2026-10-31) |
| OCR post-processing | a port of `ocr/db_boxes.py`: contours and minimum-area rectangles (`imageproc`), polygon offset (`clipper2`, a port of the Clipper2 library pyclipper wraps) | OpenCV bindings | judged on ground truth (1.2), not bit-parity | https://docs.rs/imageproc ; https://docs.rs/clipper2 | licences recorded in N1c |
| Metrics | `prometheus-client` with the names of `stage_metrics.py` | `prometheus` crate (no release for 8 months) | official client, maintained | https://docs.rs/prometheus-client | Apache-2.0 OR MIT (0.26.0, 2026-11-11) |
| Model calls, first steps | Triton over CUDA shared memory, as today | Triton's in-process C API; ONNX Runtime with the TensorRT provider (`ort`); TensorRT's runtime directly | section 1.5 | https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md ; https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inprocess_c_api.md ; https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html | Triton BSD-3; `ort` MIT OR Apache-2.0 (2.0.0-rc.15, no stable 2.x) |
| ortloom | learn from it and build on it: now its design, and the engine's GPU parts written as a separate crate that can move into ortloom; a crate dependency once ortloom is public | depend on it now; wrap `ortloom-serve` | see below | the trade study, sections 5 and 10 | MIT OR Apache-2.0 (0.1.1, pre-release) |

**The reuse decision on ortloom, from its code.** ortloom is the owner's Rust server; its
measured strength is server-side decode scheduling (decode overlapped with inference, one
batching stage, nvJPEG lanes, a GPU letterbox matched to the CPU filter), and its raw-tensor path
was measured behind Triton (trade study section 5.2). The owner intends to make it public beside
OpenProcessor once the native engine is proven, so a crate dependency is viable for the long run.
Its code today does not yet carry what this engine needs: (1) its batcher's seams (the
decode-and-letterbox step and the batch runner) are crate-private and typed to one image in, one
YOLO detection list out, with no room for a photo that fans out to a data-dependent number of
crops, faces and text lines; (2) its nvJPEG module's FFI is private and its public outputs are a
letterboxed tensor for its own session or a full-resolution frame copied to the host, not a
resident frame that many cuts read; (3) it runs models through ONNX Runtime's TensorRT provider,
not the native plans the model repository holds (trade study section 5.3). Taken now, in this
plan's own words: the checklist of 2.5, lanes as the decode knob, admission by decoded bytes read
from the header, a single owner per session and stream, a finisher stage off the feeding threads.

*Licences.* ortloom is `MIT OR Apache-2.0` (its `LICENSE-MIT` and `LICENSE-APACHE`); this
repository is AGPL-3.0-or-later. Permissive code may be used in an AGPL program (the FSF lists the
Expat (MIT) licence and Apache 2.0 as compatible with GPLv3, which AGPLv3 allows to be combined:
https://www.gnu.org/licenses/license-list.html#apache2 ,
https://www.gnu.org/licenses/license-list.html#Expat). The other direction does not hold: AGPL
code moved into ortloom would make ortloom AGPL. Code written here moves into ortloom only when
its copyright holder (the owner holds both copyrights) licenses it `MIT OR Apache-2.0`.

*Until ortloom is public.* A private repository cannot be a dependency of the public build, and
none of its code may be copied here before it is public. The interim: the GPU parts that are not
OpenProcessor-specific (nvJPEG lanes, the NVRTC kernel loader, the CUDA shared-memory region and
its registration, the frame budget and the stream-ordered frame pool) are written in their own
crate, `native/gpu_frame/`, with no type from the engine, the plan or the wire in its interface;
the engine (`native/frame_engine/`) depends on it through that boundary only. When ortloom is
public, that crate moves into ortloom (relicensed by the owner), the engine depends on the
published ortloom crate, and the copy here is deleted in the same change. Question Q3.

## 4. Decisions a measurement settles

Two measurements started on 2026-11-15 (coordinator) feed this section; neither serves protocol
version 1, N0 is the first thing that does.

- **M-a**: the owner's Rust server run as it is on the bake-off photos with the same two models.
- **M-b**: a throwaway Rust pipeline (decode threads with nvJPEG, the product's CUDA kernel
  source compiled at run time, the model server over CUDA shared memory) against the Python engine
  in the same lock holds.

| # | Decision | Settled by | Rule |
|---|---|---|---|
| 4.1 | Model calls through the model server over shared memory, or an in-process runtime | M-a for the in-process side, M-b and N0 for the shared-memory side; nothing changes before the switch | in-process only if, on plan F and C, it beats the shared-memory engine by more than the spread at equal parity, passes the kill and restart tests with a start time inside the bar, and the second build path per model passes every plan 13 precision gate. Evidence that would trigger the measurement: N0's profile showing the engine's model-client time or the model server's queue time as the bound |
| 4.2 | Async front or threads | M-b, then N0 | the front's CPU per photo (tokio threads, cgroup per-thread time) under 2 ms on 20 MP at 64 in flight and never the named bound; if it is, the response build and request parse move to the finish pool |
| 4.3 | One process or several per card | N0 rows at 1, 8, 32, 64 in flight | one process unless it fails to use the cores its decode needs (cores busy flat while the card idles and the lanes wait); several processes cost a CUDA context, a region and lanes each and time-slice the card |
| 4.4 | Decode threads against cores | N0 sweep of `OP_FRAME_ENGINE_LANES` 8, 16, 24, 32, 48 on 20 MP and COCO at 32 and 64 in flight; NUMA pinning to the card's socket as one arm (GPU 2 is on the second socket) | the knee: the smallest count within the spread of the best rate; device memory per lane recorded (lane buffers are not freed inside a process, so the count is fixed at start) |
| 4.5 | DALI-style deep batching for bulk ingest | already measured: deep batches gave the models larger batches (12 to 21 against 5 to 12) at 3 to 5x the latency and no more photos/s at 20 MP; nvJPEG's batched GPU Huffman ran 23 photos/s per card; DALI's 4.6x on small photos came from running no per-photo Python, which a native engine removes | no place in the engine, unless N0's COCO row stays under 0.9 of the models-alone row with the card and the cores both idle |
| 4.6 | An engine-side batcher across photos for crop, face and text calls (first-item timer, never reset by arrivals) | N1a: mean batch at the model server and model calls per second with and without it, on plan C at 32 and 64 in flight | kept only if it wins by more than the spread or lowers p99; otherwise the model server's batcher alone |

## 5. Phases

The owner wants the native engine now. The shortest path to the switch: N0, then three N1 lanes
in parallel, N2 overlapping the end of N1, then N3. Nothing waits for plan 14 phase 3 (the API
side keeps protocol version 1 and A1, E3, P1 are API-side work). What waits for the switch: E4's
profile sizes (done natively in N2, not in the frozen Python engine), any new engine branch or
plan field (question Q2), and the in-process runtime question (4.1).

**Evidence for the paper (applies to N0 and to the exit of every phase).** The owner intends a
research paper on the measured speed. Every speed claim is a table that names the public data set
(COCO, the Commons set, Open Images or Objects365; set P never appears in it), the hardware (card,
PCIe generation, CPU, cores, the other load on the card and the host at start and end), the
versions (engine commit, image digest, toolchain, CUDA libraries, model server, model builds), the
exact command, the bound that remains with its evidence, and the Python engine measured in the same
hold as the reference. The raw files (JSON lines, statistics snapshots, profiler output) are kept
under `artifacts_local/` and never bulk-deleted; the summary row goes into this plan's results.

### N0: the spike (first slice of the product, specified for a lane)

**Goal.** The smallest native engine that serves protocol version 1 for `detect` and
`embed_frame` (GPU decode, no colour step, letterbox and frame square, blur sums, model calls over
CUDA shared memory), measured against the Python engine in the same holds. It is written as the
product's first slice, not as throwaway code (no write-then-rewrite): what N1 adds is branches and
routes, not a rewrite.

**Entry.** This plan approved; GPU 2 time granted by the coordinator, who names the exclusive hold
file and checks the peer sessions' use of the card. N0 does not wait for M-a or M-b: M-b's
registration and kernel-compile findings are folded in when they arrive.

**Scope in.** Front (every KServe call of 1.1; the statuses; the request's `outputs` filter);
plan parse for the whole vocabulary (an unbuilt branch answers `unsupported_branch`, exactly as
the protocol defines); admission, budget, region slab; region create, purge by prefix, register,
re-register on "Unable to find shared memory region", unregister-then-free; tonic client with
deadlines and cancellation; nvJPEG lanes; orientation; the kernels `resample_h`, `resample_v`,
`lap_sums` from the shared file; host weights; the `detect` branch (letterbox, call, settle with
`undo_letterbox` and `crop_rect`, blur sums of the frame and every detection) and `embed_frame`
(centre square); readiness and health tick; context-lost handling; SIGTERM drain; settings and
metrics with the Python names.

**Scope out (N1).** The host route (a photo that needs it answers `decoder` in N0; the test sets
contain none), the colour step (the sets are filtered to untagged or sRGB-tagged files), crops,
stored boxes, conditions, faces, OCR, renditions.

**What it reuses.** The kernel sources (moved, byte-identical); the wire module through the
generated contract; the NVIDIA library builds of the Python image; the compose service's
environment names; the GPU tier `tests/gpu/test_gpu_decode_parity.py` unchanged (it speaks
protocol version 1 to whatever engine it is pointed at).

**Files.**

| Path | What |
|---|---|
| `native/Cargo.toml`, `native/Cargo.lock`, `native/rust-toolchain.toml` | the workspace (later Rust parts sit beside the engine) |
| `native/frame_engine/Cargo.toml`, `build.rs` | the crate `op-frame-engine`; `tonic-prost-build` over `native/frame_engine/proto/` |
| `native/frame_engine/proto/` | Triton's `grpc_service.proto` and `model_config.proto`, vendored at the model server's tag, with the tag in a header line |
| `native/frame_engine/kernels/prep.cu`, `colour.cu`, `faces_warp.cu`, `ocr_warp.cu`, `orient.cu` | the moved sources (first commit, with the Python modules reading them); `orient.cu` new |
| `frame_engine/prep_kernels.py`, `colour.py`, `faces/kernels.py`, `ocr/kernels.py`, `docker/frame-engine/Dockerfile` | the same first commit: the `_SRC` strings deleted, the files read instead; the Python image copies `native/frame_engine/kernels/` (the only change to the frozen engine, question Q5) |
| `native/gpu_frame/Cargo.toml`, `src/lib.rs`, `cuda.rs`, `nvjpeg.rs`, `nvjpeg/ffi.rs`, `kernels.rs` (NVRTC loader), `region.rs` (allocation, IPC handle, slab), `budget.rs`, `frame_pool.rs` | the GPU parts with no engine, plan or wire type in their interface (section 3, "Until ortloom is public") |
| `native/frame_engine/src/main.rs`, `settings.rs`, `front.rs`, `wire.rs`, `admission.rs`, `transport.rs` (model-server client, registration), `geometry.rs`, `jpeg_header.rs`, `photo.rs`, `branches/detect.rs`, `branches/embed.rs`, `metrics.rs`, `health.rs` | one concern each, named after the Python module it replaces |
| `native/frame_engine/tests/golden/` | weights, geometry and routing cases generated from the Python functions by `scripts/codegen/frame_engine_goldens.py` |
| `contracts/json/frame_engine_v1.json`, `contracts/json/frame_engine_plan_cases.json` | generated (section 3) |
| `native/frame_engine_loadgen/` | a closed-loop Rust client: N photos in flight from a file list held in memory, a plan, a warm-up, then photos/s, p50, p99 and failures by status as one JSON line (a Python client was measured as a doubt at 20 MP rates) |
| `scripts/bench/frame_engine_ab.py` | one hold: both engines, every set and in-flight count, alternating A then B, two runs each; samples host CPU from the engine container's cgroup `cpu.stat`, card busy and per-process device memory from NVML every 50 ms, the model server's statistics before and after (mean batch, queue time per request); parity mode at 1 and 32 in flight with `tritonclient` |
| `docker/frame-engine-native/Dockerfile` | multi-stage: the NVIDIA wheels unpacked for their libraries; a Rust build stage on the pinned toolchain (cmake and nasm for libjpeg-turbo); runtime on `debian:trixie-slim` by digest, non-root user, `op-frame-engine health` as the health check |
| `docker/frame-engine-native/compose.n0.yml` | the spike's test bed under its own project name (`-p op-n0`): the native engine, the Python engine and one model server on GPU 2 with `--allow-client-shm=true`, `init: true` on both engines, `memlock` unlimited |

**Tests in the lane (light, no GPU).** Unit tests for wire parsing against the plan cases,
admission and budget fairness, slab first fit and coalescing, deadline arithmetic, routing and
geometry against the goldens (weights bit for bit), the BYTES packing, status mapping of every
error class, settings ranges. Each test seen failing first (a deliberately wrong constant).

**The measurement (GPU 2, exclusive holds, bounded scripts that write files).** Sets: COCO 1,000
of `op-bench-v1` (the input-pipeline spike's `coco.txt`), the 120 public Commons photos at 18 to
24 MP (`mp20.txt`), set P 200 (`setp.txt`, numbers only); each list filtered by header to files
the GPU route takes with no converting profile, the excluded counts recorded. Models: the detector
`yolov11_small_trt_end2end` and `pe_image_encoder`, configs untouched. Every row: failed 0, the
1-minute load at start and end (gate under 16), the hold's model server restarted only between
holds.

| Comparison | Arms | Rows |
|---|---|---|
| Rate and latency | native, Python | plan F at 1, 8, 32, 64 in flight on COCO, 20 MP, set P; two runs per arm, alternating |
| Host CPU per photo | both | cgroup CPU over the measured window divided by photos; native split by thread role |
| Card | both | busy share, mean detector and encoder batch, queue time per request |
| Ceiling in the same hold | models alone | fixed inputs already on the device, 32 and 64 in flight (the input-pipeline spike's row Z) |
| Memory | both | device memory idle and peak per process; host RSS peak |
| Start time | both | container start to `ModelReady` true, model server up; and process start to ready |
| Image size | both | `docker image ls` |
| Lanes | native | 4.4's sweep at 32 and 64 in flight |
| Parity | native against Python | 200 COCO, 100 mp20 and the set P photos at 1 in flight (1.2, bit-level rows) and at 32 (floor rows); the Python engine against itself twice as the floor |
| Kernel bytes | native against CuPy | 50 frames: the letterbox canvas and frame square read back from both engines' region slices: identical |
| The kernel move | Python engine after the first commit | `tests/gpu/test_gpu_decode_parity.py`, `test_gpu_renditions.py`, `test_gpu_colour.py` pass |
| Faults (smoke) | native | 10 kills at 32 in flight; a model-server restart with the engine up |

**Go or no-go.**

| Go when all hold | No-go when any holds |
|---|---|
| Plan F, 32 and 64 in flight: COCO at least min(2.0x the Python engine, 0.85x the models-alone row of the same hold); 20 MP and set P at least min(1.5x the Python engine, 0.85x the input-pipeline harness rows); always above the Python engine by more than the spread | below the rate bar with the named bound inside the native engine and no fix inside the spike |
| Host CPU per photo at most 0.5x (COCO) and 0.75x (20 MP); p99 not above Python's | parity fails on model inputs, boxes or blur sums for a reason not found in the kernel loader, the compile options or the geometry port |
| Every parity row of 1.2 for `detect` and `embed_frame` passes | the region cannot be registered or read by the model server from the Rust process |
| Device memory and image size inside 0.2; start time measured | any kill leaves the model server restarted, not ready, or above its memory baseline |
| 10 of 10 kills survived; the restart re-registers by itself | |
| Every row recorded as the evidence clause asks (public sets for any claim, raw files kept) | |

A miss is reported with its number and its bound; the bar moves only in writing (plan 14
section 6).

**Thrown away after N0.** `compose.n0.yml` and the N0-only `decoder` answer for host-route photos
(N1a replaces it with the host route). The raw rows move to `artifacts_local/`. Everything else is
the product's code and stays.

### N1: every branch, each with its parity gate (three lanes in parallel)

**Entry.** N0 is a go and merged. **Exit.** Every branch any caller sends, and every branch the
Python engine builds, passes its rows of 1.2 on the GPU tier and on the public sets; plan C and the
branch rows recorded under the evidence clause.

| Lane | Adds | Parity gate (1.2) |
|---|---|---|
| N1a, ingest | `embed_crops` with `embed_classes`, `crops: inline` (finish pool, libjpeg-turbo), stored `boxes`, `if_detected` and the late rounds, the host route, the colour step (classify, table build single-flighted per profile, gather), every refusal status; decision 4.6 | the four-way ingest rows' photos through the API (`tests/gpu/`, plan 13's E1 end-to-end parity script): crop counts, pixel sizes and decoded pixels, crop vectors, blur sums; the 88-profile corpus (COCO's 85 plus the 3 fixtures): identical class and table bytes; host-route sets (PNG, progressive, CMYK, small) |
| N1b, faces | `faces` (whole frame, coarse pass, tiles), `face_embed` (warp kernel, chunks of `batch`), anchors and NMS on the finish pool | `tests/gpu/test_gpu_faces.py`; LFW, CALFW, CPLFW accuracy not below 0.9985 / 0.9592 / 0.9592; WIDER FACE AP not below the branch row |
| N1c, text and renditions | `ocr` (detector input, DB boxes, line warp, batches by width, CTC), `derivatives` (Lanczos weights, nvJPEG encode) | `tests/gpu/test_gpu_ocr.py`; COCO-Text 400 photos F1 and spotted words not below the branch row; `tests/gpu/test_gpu_renditions.py` and `test_gpu_colour.py` |

Each lane's GPU-tier runs go to the wave sweep, never into the lane (the multi-agent rule); a lane
lists them as owed.

### N2: packaging, profiles, faults, docs and the sweep

**Entry.** N1a merged (N2 starts while N1b and N1c finish). **Exit.** All rows below green in one
sweep on the merged tree, the sweep's speed rows recorded under the evidence clause.

- Image: Trivy scan clean of high and critical findings; non-root; size and start time inside 0.2.
- Settings: every `OP_FRAME_ENGINE_*` read with the Python ranges; the interpreter-only
  `OP_FRAME_ENGINE_SWITCH_INTERVAL_MS` refused at start as retired; the meaning of
  `OP_FRAME_ENGINE_WORKERS` decided from N0's rows (kept as the launch pool size, or retired).
- Profiles (E4): sizes per profile from measured memory, the 12 GB card starting inside its memory
  and running plan A on the mix with failed 0; the profile files, the installer mirror, its test and
  the sizing page changed together.
- Faults: 50 kills at 32 in flight; a model-server restart (engine ready within 15 s of the server's
  gRPC port, the restart check of plan 13); the server recreated on a new address (first good call
  within about 1 s, as the Python row); SIGTERM drain; a sticky error injected (a kernel with an
  out-of-range write in a test build) ends in exit 1 and a clean restart.
- Metrics: the names and labels of `/metrics` equal to the Python engine's (a scripted diff).
- Docs: `docker/test/README.md` GPU tier, the docs-site engine and sizing pages (concise user
  page, detailed developer page), `env.template`.
- The sweep: full offline suite, the GPU tier on the native engine, the four-way ingest rows and
  the 600 s soak of configuration D with failed 0; every number recorded with host CPU per photo per
  process (plan 14 section 4).

### N3: the switch and the deletion

**Entry.** N2's sweep green; the adoption bars of 0.2 met on plans F and C, and plan A measured.
**Exit.** One engine in the tree and in the stack, a green sweep, a ledger entry, and the final
table of plans F, C and A against the last Python measurement recorded under the evidence clause
(the paper's main table).

In one change: the compose service keeps its name and image name
(`openprocessor-frame-engine`) and points at the native build; `docker/frame-engine-native/`
becomes `docker/frame-engine/`; `frame_engine/` (Python) and `tests/frame_engine_tests/` are
deleted (their cases live on as Rust tests and goldens); the kernel files stay where they are;
`scripts/codegen/check_layering.py`, `tests/test_frame_engine_profile_sizes.py` and the
keepalive pin are updated; `docker/test/fake_frame_engine.py` (which imports the Python front)
gets a small KServe front of its own in `docker/test/` so the live harness keeps its CPU-only fake;
the `make` targets and the `openprocessor` CLI paths that name the Python engine are updated; plan
13's T1 row and this plan's status are closed. No compatibility path, no second engine kept.

## 6. Interim lever: several Python engine processes

Not done: the owner decided on 2026-11-15 that no further speed work goes into the Python engine.
The measurement stays as the reason the native engine must spread decode over all cores in one
process: the engine's input code in four processes did 3,398 photos/s on small photos against 650
in one (5.2x), and 1.2 to 1.4x at 12 to 20 MP.

## 7. Risks and their early signals

| Risk | Early signal | Answer |
|---|---|---|
| CUDA crate maturity: `cudarc` is pre-1.0 and has no safe IPC or nvJPEG wrappers | a minor version bump breaks the build; a raw call needed outside `cuda.rs` | exact pins; every unsafe call in `cuda.rs` and `nvjpeg/ffi.rs` only, each with the invariant it relies on |
| Shared-memory registration from Rust | N0's first day: the register call refused, or the model server reads zeros | the 64-byte handle of the region's base `cudaMalloc`; engine process id never 1 (`init: true`); prefix purge first |
| Kernel parity | the kernel-bytes row differs | the same NVRTC build and CuPy's options; if options differ the lane reads CuPy's compiler module, never guesses |
| Host weights drift (float64 arithmetic, `np.sinc` for Lanczos) | the golden test differs in the last bit | compute in f64 in the same order; a Lanczos-only difference touches renditions, judged by their PSNR bar |
| nvJPEG thread safety and lane memory | device memory grows across a soak; a decode error under load only | per-lane state, buffers and jpeg stream (documented); the lane count fixed at start |
| libjpeg-turbo builds differ from simplejpeg's and OpenCV's | host-route frames or crop bytes differ in N1a | pin the vendored version to the Python image's build where possible; otherwise question Q6's rule |
| OCR post-processing port | F1 below the branch row | ground-truth gate; the reference is the frozen Python branch and its captured outputs |
| Build time (tonic, CUDA FFI, libjpeg-turbo with cmake and nasm, lcms2) | a clean image build over 10 minutes | cached dependency layer in the Dockerfile; builds happen once per sweep |
| GPU test coverage | a GPU-tier test not run because a lane server lacks a model variant (faces and OCR need the uint8 builds) | listed as owed to the sweep, never skipped; the sweep's model server carries every variant |
| Contract drift while N1 runs (13b C1 changes the plan vocabulary) | the drift hook fails on `frame_engine_plan_cases.json` | the hook; the Rust parser is regenerated against the new cases in the lane that changes the vocabulary |
| The desktop direction | none now | the model-call seam is one trait with one implementation (Triton); an in-process implementation is added only when 4.1 or the desktop plan needs it (no code ahead of time) |

## 8. Questions for the coordinator

| # | Question | Evidence | Recommendation |
|---|---|---|---|
| Q1 | Do faces, OCR and renditions block the switch, or may the switch go with them answering `unsupported_branch`? | No caller sends them today: curation ingest sends `detect`, `embed_frame`, `embed_crops`, crops inline (`src/services/curation/ingest_gpu_decode.py`), reprocess and import send stored boxes with the embed branches. But switching early deletes validated Python branches whose GPU tests would then fail, and "never skip" forbids parking them | port all three in N1 in parallel; the switch waits for them, not for phase 3 |
| Q2 | Does 13b C1 (the crop encoder as a named model, letterboxed crops at 224) land in the frozen Python engine first? | It is a correctness and wire change, small; without it the native engine has no reference for the new crop rule | yes: C1 in the wire module and the Python engine first, the native engine implements it in N1a |
| Q3 | How the engine builds on ortloom while ortloom is private | A private repository cannot be a dependency of the public build; ortloom is `MIT OR Apache-2.0`, this repository AGPL (permissive into AGPL is fine, the reverse needs the owner's relicensing); ortloom's nvJPEG and batcher code is not yet shaped for a fanned-out photo (section 3) | write the shared GPU parts as `native/gpu_frame/`, AGPL here for now, with no engine type in its interface; when ortloom is public the owner relicenses that crate and moves it into ortloom, the engine takes the published crate and the copy here is deleted in the same change. Alternative: give `native/gpu_frame/` its own `MIT OR Apache-2.0` licence files from day one (faster move later, but an exception to the whole-repository AGPL decision) |
| Q4 | Two cards on one host share the host's decode capacity | plan F on two cards needs about 240 photos/s of 20 MP decode; the host measured 222 to 279 | accept the host bound; N2 measures NUMA pinning per engine |
| Q5 | The kernel sources move from Python strings to `.cu` files read by both engines: a no-behaviour change to the frozen engine | one source for both engines is the parity argument; the GPU tier proves it | yes, as N0's first commit |
| Q6 | Crop JPEG equality if the libjpeg-turbo builds differ | no consumer compares crop bytes; the crop cache stores what it receives | equal means decoded pixels identical and sizes within 0.5% |

## 9. Results

None yet. N0 adds the first rows here, in the form of the evidence clause (section 5).
